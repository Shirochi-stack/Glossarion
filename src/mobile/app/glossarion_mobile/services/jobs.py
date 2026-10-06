"""JobService: one exclusive backend job at a time on the ``gl-job`` thread.

Plan §1/§4, mobile-app design §4-§5.1, UI_SPEC §1.7-§1.9. The backend is
configured through process-global state (``os.environ``, ``sys.argv``, stdout,
``UnifiedClient`` key pools), so exactly one job runs at a time:

* ``submit(spec)`` queues a ``JobSpec`` ("run after current" while another job
  runs) and returns its id. One long-lived daemon thread named ``gl-job`` pulls
  the queue (it inherits the 16 MiB stack set by ``runtime_bootstrap``).
* At job start the thread flushes the ``MobileConfigStore`` and takes a
  ``snapshot()`` (edits made while the job runs apply to the next run;
  ``set_job_running`` drives the Settings banner), then enters
  ``job_runner.scoped_process_state(capture_stdout=host.log, lock=JOB_LOCK)``.
  Inside it, and only there, it calls ``stop_control.reset_for_new_run``,
  builds the ``HeadlessOwner`` from the snapshot (its init writes ~40 env vars)
  and hands a ``JobContext`` to the kind's adapter (``job_kinds/*``), which only
  calls shared owner methods. ``ProgressWatcher`` turns
  ``translation_progress.json`` and the API watchdog into ``progress`` /
  ``api_state`` events while the adapter runs.
* ``request_stop()`` replays the desktop ``stop_translation`` sequence without its
  widgets (``JobBackend.request_stop``: force branch -> watchdog reset ->
  ``stop_control.request_stop``, whose flag ordering is the desktop's ->
  ``stop_control.announce_stop``: HTTP log suppression and the stop log line): the
  first request is graceful when the config's ``graceful_stop`` is on (state
  STOPPING), otherwise immediate (FORCE_STOPPING); any further user request forces
  (UI_SPEC §2.4 "finishing: tap again = force"; the desktop forces on a second click
  within 1 s). The job's latch is set at once; on the UI loop the protocol itself runs
  on a ``gl-job-stop`` thread. A stop that arrives while the job is still starting is
  re-issued once the run has been reset (and the adapters skip the worker when the
  shared set-up erased it); the next job waits for the previous immediate stop's
  cleanup thread (``stop_control.wait_for_stop_cleanup``).
* ``<data>/jobs/active.state`` checkpoints the running job (spec, inputs,
  resolved output folders, progress) and the queue; ``recover()`` on the next
  launch turns what it finds into Interrupted jobs after restoring the
  in-progress rows of their progress files with
  ``shutdown_utils.restore_in_progress_rows_for_shutdown`` (the desktop
  force-exit cleanup). ``history.state`` keeps the last 50 finished jobs and
  ``interrupted.state`` the jobs waiting for Resume / Discard.
* Every job writes ``<logs>/jobs/<id>.log`` and a per-job ``LogBuffer``; every
  log message also feeds, whole, the job's ``direct_text_stream.DirectTextStream``
  (the desktop Direct Text request classifier; the chat's live cards and job detail
  read it through ``request_stream(job_id)``), which a ``gl-job-stream`` thread
  drains with the desktop's repaint budget while the job runs.

Threading: internal state is guarded by one lock and is safe to call from any
thread. Listeners (``subscribe``, ``on_transition``, ``on_question``,
``on_event``) run on the UI loop when a bound ``UiDispatcher`` is given
(state transitions are posted in order; progress is coalesced through a
latest-wins ``Channel`` and the view is rebuilt at delivery time, so a listener
never sees a stale view), synchronously otherwise (host tests).

Pure Python, no Flet import; shared backend modules are imported lazily on the
job thread through ``JobBackend`` (tests pass a fake). Python 3.10 compatible.
"""

from __future__ import annotations

import contextlib
import copy
import enum
import json
import logging
import os
import sys
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterable, Iterator, Mapping, Optional, Sequence

from glossarion_mobile.services.dispatcher import LogBuffer

__all__ = [
    "ACTIVE_STATE_FILE",
    "HISTORY_LIMIT",
    "HISTORY_STATE_FILE",
    "INTERRUPTED_STATE_FILE",
    "JobBackend",
    "JobContext",
    "JobError",
    "JobKind",
    "JobService",
    "JobSnapshot",
    "JobSpec",
    "JobState",
    "JobsView",
    "Progress",
    "STATE_LABELS",
    "TERMINAL_STATES",
    "ACTIVE_STATES",
    "format_duration",
    "progress_line",
    "status_key",
    "strip_model_for",
]

log = logging.getLogger("glossarion.jobs")

STATE_VERSION = 1
HISTORY_LIMIT = 50
ACTIVE_STATE_FILE = "active.state"
HISTORY_STATE_FILE = "history.state"
INTERRUPTED_STATE_FILE = "interrupted.state"
CHECKPOINT_INTERVAL = 10.0  # seconds between active.state rewrites while progress moves
JOB_LOG_MAXLEN = 5000  # lines held in a job's LogBuffer (the per-job file keeps everything)
JOB_BUFFERS_KEPT = 4  # finished jobs whose LogBuffer stays in memory
LAST_LINE_CHARS = 240
STREAM_DRAIN_INTERVAL = 0.1  # seconds between the job-side budgeted drains of the request stream

#: Lines ``authgpt_auth`` prints when a job has no usable ChatGPT login and falls back to the
#: interactive browser login (``get_valid_access_token`` / ``recover_from_unauthorized``, or the
#: headless refusal). The first one in a job sends a ``sign_in_required`` event (UI_SPEC §1.9).
SIGN_IN_MARKERS = (
    "AuthGPT: No valid token found",
    "AuthGPT: Session expired",
    "AuthGPT: Browser-based OAuth login is not available",
)


class JobKind(str, enum.Enum):
    """Job kinds with an adapter in ``job_kinds`` (more arrive with later milestones)."""

    TRANSLATE = "translate"
    DIRECT_TEXT = "direct_text"
    EXTRACT_GLOSSARY = "extract_glossary"
    GLOSSARY_REFINE = "glossary_refine"
    UNIFIED_GLOSSARY = "unified_glossary"
    PARALLEL_PAIR = "parallel_pair"
    COMPILE_EPUB = "compile_epub"
    COMPILE_PDF = "compile_pdf"
    SINGLE_CHAPTER = "single_chapter"
    # U6 tools
    QA_SCAN = "qa_scan"
    VALIDATE_EPUB = "validate_epub"
    RENAME_OUTPUTS = "rename_outputs"
    TRANSLATE_HEADERS = "translate_headers"
    METADATA = "metadata"


class JobState(str, enum.Enum):
    """Values equal the names, so ``state.value`` feeds ``send_state.SendInputs.own_job_state``."""

    QUEUED = "QUEUED"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    STOPPING = "STOPPING"  # graceful stop requested
    FORCE_STOPPING = "FORCE_STOPPING"  # immediate or forced stop requested
    DONE = "DONE"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"  # stopped by the user (or cancelled while queued)
    INTERRUPTED = "INTERRUPTED"  # found in active.state after the app was killed


TERMINAL_STATES = frozenset({JobState.DONE, JobState.FAILED, JobState.CANCELLED, JobState.INTERRUPTED})
ACTIVE_STATES = frozenset({JobState.STARTING, JobState.RUNNING, JobState.STOPPING, JobState.FORCE_STOPPING})

#: Jobs page state chip labels (UI_SPEC §1.8).
STATE_LABELS = {
    JobState.QUEUED: "Queued",
    JobState.STARTING: "Starting",
    JobState.RUNNING: "Running",
    JobState.STOPPING: "Stopping",
    JobState.FORCE_STOPPING: "Force stopping",
    JobState.DONE: "Done",
    JobState.FAILED: "Failed",
    JobState.CANCELLED: "Cancelled",
    JobState.INTERRUPTED: "Interrupted",
}

#: ``tokens.STATUS_STYLES`` key for each state (icon + colour of the state chip).
_STATUS_KEYS = {
    JobState.QUEUED: "queued",
    JobState.STARTING: "running",
    JobState.RUNNING: "running",
    JobState.STOPPING: "running",
    JobState.FORCE_STOPPING: "running",
    JobState.DONE: "done",
    JobState.FAILED: "failed",
    JobState.CANCELLED: "stopped",
    JobState.INTERRUPTED: "interrupted",
}

STOP_GRACEFUL = "graceful"
STOP_IMMEDIATE = "immediate"
STOP_FORCE = "force"


class JobError(RuntimeError):
    """A job could not run (bad spec, missing input, shared module missing). Message is user-facing."""


class _Cancelled(Exception):
    """The job was stopped before its adapter started."""


def _jsonable(value: Any, _depth: int = 0) -> Any:
    if _depth > 8:
        return repr(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v, _depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(v, _depth + 1) for v in value]
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    return repr(value)


def _norm(path: Any) -> str:
    try:
        return os.path.normcase(os.path.abspath(os.fspath(path)))
    except Exception:
        return str(path)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JobSpec:
    """What to run. ``params`` and ``origin`` must be JSON-friendly (persisted for Resume)."""

    kind: str
    title: str
    inputs: tuple = ()
    params: Mapping[str, Any] = field(default_factory=dict)
    #: Where the job came from: ``{"type": "chat", "cid": "3", "label": "Chat · My novel"}``,
    #: ``{"type": "library", "bid": "...", "label": "Library · Book"}``; the route opens it.
    origin: Mapping[str, Any] = field(default_factory=dict)
    resumable: bool = True

    def __post_init__(self) -> None:
        kind = self.kind.value if isinstance(self.kind, enum.Enum) else str(self.kind)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "inputs", tuple(os.fspath(p) for p in (self.inputs or ())))
        object.__setattr__(self, "params", dict(self.params or {}))
        object.__setattr__(self, "origin", dict(self.origin or {}))

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "title": self.title,
            "inputs": list(self.inputs),
            "params": _jsonable(self.params),
            "origin": _jsonable(self.origin),
            "resumable": bool(self.resumable),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "JobSpec":
        return cls(
            kind=str(data.get("kind") or ""),
            title=str(data.get("title") or ""),
            inputs=tuple(data.get("inputs") or ()),
            params=dict(data.get("params") or {}),
            origin=dict(data.get("origin") or {}),
            resumable=bool(data.get("resumable", True)),
        )


@dataclass(frozen=True)
class Progress:
    """Chapter progress from ``ProgressWatcher`` (``translation_progress.json`` summary)."""

    total: Optional[int] = None
    completed: int = 0
    in_progress: int = 0
    failed: int = 0
    label: str = ""

    @property
    def fraction(self) -> Optional[float]:
        if not self.total:
            return None
        return max(0.0, min(1.0, self.completed / float(self.total)))

    def to_dict(self) -> dict:
        return {"total": self.total, "completed": self.completed, "in_progress": self.in_progress,
                "failed": self.failed, "label": self.label}

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, Any]]) -> "Progress":
        data = data or {}

        def num(key: str, default: Any = 0) -> Any:
            value = data.get(key, default)
            try:
                return int(value) if value is not None else None
            except (TypeError, ValueError):
                return default

        return cls(total=num("total", None), completed=num("completed") or 0, in_progress=num("in_progress") or 0,
                   failed=num("failed") or 0, label=str(data.get("label") or ""))


@dataclass(frozen=True)
class JobSnapshot:
    """Immutable view of one job (what screens, the strip and notifications read)."""

    id: str
    spec: JobSpec
    state: JobState
    created: float
    started: Optional[float] = None
    finished: Optional[float] = None
    progress: Progress = field(default_factory=Progress)
    in_flight: int = 0
    peak_in_flight: int = 0
    api_queued: int = 0
    phase: str = ""
    warning: bool = False
    last_line: str = ""
    outputs: tuple = ()
    output_dirs: Mapping[str, str] = field(default_factory=dict)
    output_dir: Optional[str] = None
    error: Optional[str] = None
    stop_mode: Optional[str] = None
    stop_requested_at: Optional[float] = None
    run_id: Any = None
    log_path: Optional[str] = None
    question: Optional[Mapping[str, Any]] = None
    recovered: bool = False
    resolution: Optional[str] = None  # interrupted jobs: "resumed" | "discarded"
    restore_pending: bool = False  # recovered; its progress rows are not restored yet
    result: Mapping[str, Any] = field(default_factory=dict)  # what the kind adapter recorded (JobContext.set_result)

    @property
    def kind(self) -> str:
        return self.spec.kind

    @property
    def title(self) -> str:
        return self.spec.title

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    @property
    def is_active(self) -> bool:
        return self.state in ACTIVE_STATES

    @property
    def stopped(self) -> bool:
        """Ended because the user (or the system) stopped it."""
        return self.state is JobState.CANCELLED and self.started is not None

    @property
    def state_label(self) -> str:
        if self.state is JobState.CANCELLED and self.started is not None:
            return "Stopped"
        return STATE_LABELS.get(self.state, str(self.state.value).title())

    def elapsed(self, now: Optional[float] = None) -> Optional[float]:
        if self.started is None:
            return None
        end = self.finished if self.finished is not None else (now if now is not None else time.time())
        return max(0.0, end - self.started)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "spec": self.spec.to_dict(),
            "state": self.state.value,
            "created": self.created,
            "started": self.started,
            "finished": self.finished,
            "progress": self.progress.to_dict(),
            "in_flight": self.in_flight,
            "peak_in_flight": self.peak_in_flight,
            "api_queued": self.api_queued,
            "phase": self.phase,
            "last_line": self.last_line,
            "outputs": list(self.outputs),
            "output_dirs": dict(self.output_dirs),
            "output_dir": self.output_dir,
            "error": self.error,
            "stop_mode": self.stop_mode,
            "run_id": _jsonable(self.run_id),
            "log_path": self.log_path,
            "recovered": self.recovered,
            "resolution": self.resolution,
            "restore_pending": self.restore_pending,
            "result": _jsonable(dict(self.result)),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "JobSnapshot":
        try:
            state = JobState(str(data.get("state") or "INTERRUPTED"))
        except ValueError:
            state = JobState.INTERRUPTED
        return cls(
            id=str(data.get("id") or uuid.uuid4().hex[:12]),
            spec=JobSpec.from_dict(data.get("spec") or {}),
            state=state,
            created=float(data.get("created") or 0.0),
            started=data.get("started"),
            finished=data.get("finished"),
            progress=Progress.from_dict(data.get("progress")),
            in_flight=int(data.get("in_flight") or 0),
            peak_in_flight=int(data.get("peak_in_flight") or 0),
            api_queued=int(data.get("api_queued") or 0),
            phase=str(data.get("phase") or ""),
            last_line=str(data.get("last_line") or ""),
            outputs=tuple(data.get("outputs") or ()),
            output_dirs=dict(data.get("output_dirs") or {}),
            output_dir=data.get("output_dir"),
            error=data.get("error"),
            stop_mode=data.get("stop_mode"),
            run_id=data.get("run_id"),
            log_path=data.get("log_path"),
            recovered=bool(data.get("recovered", False)),
            resolution=data.get("resolution"),
            restore_pending=bool(data.get("restore_pending", False)),
            result=dict(data.get("result") or {}),
        )


@dataclass(frozen=True)
class JobsView:
    """Everything the Jobs page shows (UI_SPEC §1.8), built fresh at delivery time."""

    active: Optional[JobSnapshot] = None
    queue: tuple = ()
    interrupted: tuple = ()
    history: tuple = ()
    paused: bool = False

    @property
    def running_count(self) -> int:
        return 1 if self.active is not None else 0

    @property
    def queued_count(self) -> int:
        return len(self.queue)

    def find(self, job_id: str) -> Optional[JobSnapshot]:
        for snap in ((self.active,) if self.active is not None else ()) + tuple(self.queue) + tuple(
            self.interrupted
        ) + tuple(self.history):
            if snap is not None and snap.id == job_id:
                return snap
        return None


@dataclass
class _Job:
    """Mutable job record (guarded by ``JobService._lock``)."""

    id: str
    spec: JobSpec
    state: JobState = JobState.QUEUED
    created: float = 0.0
    started: Optional[float] = None
    finished: Optional[float] = None
    progress: Progress = field(default_factory=Progress)
    in_flight: int = 0
    peak_in_flight: int = 0
    api_queued: int = 0
    phase: str = ""
    warning: bool = False
    last_line: str = ""
    outputs: list = field(default_factory=list)
    output_dirs: dict = field(default_factory=dict)
    output_dir: Optional[str] = None
    error: Optional[str] = None
    stop_mode: Optional[str] = None
    stop_requested_at: Optional[float] = None
    issued_stop: Optional[str] = None  # the last stop mode whose protocol ran for this job
    sign_in_flagged: bool = False  # a "sign_in_required" event was sent for this job
    announced: threading.Event = field(default_factory=threading.Event)  # its QUEUED transition was posted
    run_id: Any = None
    log_path: Optional[str] = None
    question: Optional[dict] = None
    stop_event: threading.Event = field(default_factory=threading.Event)
    owner: Any = None
    buffer: Optional[LogBuffer] = None
    stream: Any = None
    log_file: Any = None
    log_lock: threading.Lock = field(default_factory=threading.Lock)
    last_checkpoint: float = 0.0
    result: dict = field(default_factory=dict)

    def snapshot(self) -> JobSnapshot:
        return JobSnapshot(
            id=self.id,
            spec=self.spec,
            state=self.state,
            created=self.created,
            started=self.started,
            finished=self.finished,
            progress=self.progress,
            in_flight=self.in_flight,
            peak_in_flight=self.peak_in_flight,
            api_queued=self.api_queued,
            phase=self.phase,
            warning=self.warning,
            last_line=self.last_line,
            outputs=tuple(self.outputs),
            output_dirs=dict(self.output_dirs),
            output_dir=self.output_dir,
            error=self.error,
            stop_mode=self.stop_mode,
            stop_requested_at=self.stop_requested_at,
            run_id=self.run_id,
            log_path=self.log_path,
            question=dict(self.question) if self.question else None,
            result=dict(self.result),
        )


# ---------------------------------------------------------------------------
# Presentation helpers (pure; used by the strip, the notifications and the screens)
# ---------------------------------------------------------------------------


def format_duration(seconds: Optional[float]) -> str:
    """``12:41`` / ``1:02:03`` (elapsed time on the strip and the job card)."""
    if seconds is None:
        return ""
    seconds = int(max(0, seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def status_key(snap: JobSnapshot) -> str:
    """``tokens.STATUS_STYLES`` key for the state chip."""
    return _STATUS_KEYS.get(snap.state, "info")


def _kind_info(kind: str) -> Any:
    try:
        from glossarion_mobile.job_kinds import get_kind

        return get_kind(kind)
    except Exception:
        return None


def kind_verb(kind: str) -> str:
    info = _kind_info(kind)
    return getattr(info, "verb", None) or str(kind).replace("_", " ").capitalize()


def kind_icon(kind: str) -> str:
    info = _kind_info(kind)
    return getattr(info, "icon", None) or "WORK_HISTORY"


def progress_line(snap: JobSnapshot, now: Optional[float] = None) -> str:
    """"Ch 12/80 · 3 in flight · 12:41" (UI_SPEC §1.7 subtitle); the phase while the total is unknown."""
    parts: list[str] = []
    if snap.question:
        return "Waiting for your glossary decision"
    if snap.state is JobState.STOPPING:
        parts.append("Stopping after current request…")
    elif snap.state is JobState.FORCE_STOPPING:
        parts.append("Force stopping…")
    progress = snap.progress
    if progress.total:
        parts.append(f"Ch {progress.completed}/{progress.total}")
    elif snap.phase and not parts:
        parts.append(snap.phase)
    if snap.in_flight:
        parts.append(f"{snap.in_flight} in flight")
    if progress.failed:
        parts.append(f"{progress.failed} failed")
    elapsed = snap.elapsed(now)
    if elapsed is not None and snap.is_active:
        parts.append(format_duration(elapsed))
    if not parts and snap.state is JobState.QUEUED:
        parts.append("Queued")
    return " · ".join(parts)


def notification_text(snap: JobSnapshot) -> str:
    """Ongoing FGS text: "Translating *Book*: 12/80 chapters · 3 in flight" (UI_SPEC §1.9)."""
    text = f"{kind_verb(snap.kind)} {snap.title}"
    progress = snap.progress
    details: list[str] = []
    if progress.total:
        details.append(f"{progress.completed}/{progress.total} chapters")
    if snap.in_flight:
        details.append(f"{snap.in_flight} in flight")
    if snap.state is JobState.STOPPING:
        details.append("stopping after current request")
    elif snap.state is JobState.FORCE_STOPPING:
        details.append("force stopping")
    if snap.question:
        details = ["waiting for your glossary decision"]
    return f"{text}: {' · '.join(details)}" if details else text


def strip_model_for(snap: Optional[JobSnapshot], queued: int = 0, now: Optional[float] = None) -> Any:
    """``AppState.JobStripModel`` for the global JobStrip (None hides it)."""
    if snap is None:
        return None
    from glossarion_mobile.state.app_state import JobStripModel

    origin = snap.spec.origin or {}
    owner_chat = str(origin.get("cid")) if origin.get("type") == "chat" and origin.get("cid") is not None else None
    if snap.is_terminal:
        if snap.state is JobState.FAILED:
            title, state = f"Failed · {snap.title}", "failed"
        elif snap.stopped:
            title, state = f"Stopped · {snap.title}", "done"
        else:
            title, state = f"Done · {snap.title}", "done"
        progress = snap.progress
        subtitle = f"{progress.completed}/{progress.total} chapters" if progress.total else (snap.error or "")
        return JobStripModel(title=title, subtitle=subtitle[:160], progress=1.0 if state == "done" else None,
                             kind_icon=kind_icon(snap.kind), queued=queued, state=state, owner_chat=owner_chat)
    if snap.state is JobState.STOPPING:
        state = "finishing"
    elif snap.state is JobState.FORCE_STOPPING:
        state = "stopping"
    else:
        state = "running"
    return JobStripModel(
        title=f"{kind_verb(snap.kind)} · {snap.title}",
        subtitle=progress_line(snap, now),
        progress=snap.progress.fraction,
        kind_icon=kind_icon(snap.kind),
        queued=queued,
        state=state,
        warning=bool(snap.question),
        owner_chat=owner_chat,
    )


# ---------------------------------------------------------------------------
# Persistence (atomic JSON; "state" files are plain JSON)
# ---------------------------------------------------------------------------


def _write_json(path: str, data: Any) -> None:
    from glossarion_mobile.state.prefs import atomic_write_json

    atomic_write_json(path, data)


def _read_json(path: str) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        return None
    except Exception as exc:
        log.warning("unreadable job state %s: %s", path, exc)
        try:
            os.replace(path, f"{path}.corrupt-{int(time.time())}")
        except OSError:
            pass
        return None


# ---------------------------------------------------------------------------
# Shared backend bridge
# ---------------------------------------------------------------------------


class JobBackend:
    """Lazy bridge to the shared GUI-free modules; tests pass a fake with the same methods.

    ``job_runner`` (JOB_LOCK, scoped_process_state, ProgressWatcher),
    ``stop_control`` (reset_for_new_run, request_stop), ``headless_owner`` and
    ``shutdown_utils``. Nothing here re-implements backend logic: a missing
    shared module fails the job with a clear message instead.
    """

    def __init__(self) -> None:
        self._fallback_lock = threading.RLock()
        #: the last immediate stop's ``translation-stop-cleanup`` thread (desktop:
        #: ``TranslatorGUI._translation_stop_cleanup_thread``); the next job waits for it
        self.stop_cleanup_thread: Optional[threading.Thread] = None

    @staticmethod
    def _module(name: str) -> Any:
        import importlib

        try:
            return importlib.import_module(name)
        except ImportError as exc:
            raise JobError(f"The shared module '{name}' is not in this build ({exc}).") from exc

    @property
    def job_lock(self) -> Any:
        try:
            import job_runner  # shared (U3)

            return job_runner.JOB_LOCK
        except Exception:
            return self._fallback_lock

    def scoped_process_state(self, *, capture_stdout: Callable[..., Any], lock: Any) -> Any:
        """``job_runner.job_process_state``: JOB_LOCK held, env / argv / large_env / cwd restored key
        by key, fresh ``UnifiedClient`` key pools (the previous pool state is put back), stdout and
        stderr lines teed to ``capture_stdout``. Falls back to the contract's ``scoped_process_state``."""
        job_runner = self._module("job_runner")
        scope = getattr(job_runner, "job_process_state", None)
        if callable(scope):
            return scope(capture_stdout, lock=lock)
        return job_runner.scoped_process_state(capture_stdout=capture_stdout, lock=lock)

    def make_owner(self, config: dict, *, host: Any) -> Any:
        headless_owner = self._module("headless_owner")
        return headless_owner.HeadlessOwner(config, host=host)

    def reset_for_new_run(self, kind: str) -> Any:
        stop_control = self._module("stop_control")
        return stop_control.reset_for_new_run(kind=kind)

    def wait_for_stop_cleanup(self, log: Callable[..., Any]) -> bool:
        """``stop_control.wait_for_stop_cleanup`` on the previous immediate stop's cleanup thread
        (the desktop ``run_translation_thread`` preflight): False when it is still closing HTTP
        transports after 3 s, so the next run must not clear the cancellation state yet."""
        thread = self.stop_cleanup_thread
        if thread is None:
            return True
        stop_control = self._module("stop_control")
        ready = stop_control.wait_for_stop_cleanup(thread, log)
        if ready and self.stop_cleanup_thread is thread:
            self.stop_cleanup_thread = None
        return bool(ready)

    def request_stop(
        self,
        *,
        graceful: bool,
        wait_for_chunks: bool,
        force: bool,
        set_stop_requested: Callable[[], Any],
        log: Callable[..., Any],
        kind: str = "translation",
        job_kind: str = "",
        owner: Any = None,
    ) -> None:
        """The desktop ``stop_translation`` sequence without its widgets.

        1. force (desktop: a second click within 1 s): ``graceful_stop_active`` and
           ``_last_stop_was_graceful`` drop to False, ``apply_force_stop_flags()``;
        2. a full (non-graceful) stop resets the API watchdog (the owner's
           ``_reset_api_watchdog_progress``);
        3. ``stop_control.request_stop``: env mode flags -> the latch
           (``set_stop_requested``) -> stop files -> module flags (with the desktop's
           ``translation_stop_flag`` hook) -> background cleanup with the same watchdog
           reset; the cleanup thread is kept (``stop_cleanup_thread``) so the next job
           waits for it like the desktop's next Run;
        4. compile jobs raise the EPUB converter's stop flag
           (``stop_control.stop_epub_converter``; the desktop does while the converter
           runs), then ``stop_control.announce_stop``: a graceful stop silences the HTTP
           client loggers and the desktop log line says which stop it is.

        Not replayed (recorded divergences): ``save_config`` (mobile never persists on a
        stop), the 500 ms ``_reset_stop_flags_if_idle`` timer (the next job's
        ``reset_for_new_run`` resets them). Glossary jobs (``kind='glossary'``) take the
        desktop ``stop_glossary_extraction`` protocol, ``stop_control.request_glossary_stop``
        (U6: GRACEFUL_STOP, HTTP log suppression, the latch, the immediate-stop flags and the
        run-id-guarded cleanup that kills the helper processes and touches GLOSSARY_STOP_FILE);
        a forced stop is the desktop's double-click (graceful False). A build without it falls
        back to the translation protocol above (it latches the owner's ``stop_requested``,
        which the glossary extractor's stop callback polls).
        """
        stop_control = self._module("stop_control")
        glossary_stop = getattr(stop_control, "request_glossary_stop", None) if kind == "glossary" else None
        if callable(glossary_stop):
            if force:
                graceful = False
                if owner is not None:
                    try:
                        owner.graceful_stop_active = False
                        owner._last_stop_was_graceful = False
                    except Exception:
                        pass
                log("⚡ Double-click detected — forcing immediate stop!")

            def glossary_stop_flag(value: bool) -> None:
                # Desktop: the lazily loaded ``glossary_stop_flag`` global
                # (extract_glossary_from_epub.set_stop_flag), set once the extractor is loaded.
                module = sys.modules.get("extract_glossary_from_epub")
                set_flag = getattr(module, "set_stop_flag", None) if module is not None else None
                if callable(set_flag):
                    set_flag(value)

            glossary_stop(graceful=graceful, set_stop_requested=set_stop_requested, log=log,
                          glossary_stop_flag=glossary_stop_flag,
                          get_run_id=lambda: getattr(owner, "_glossary_run_id", None))
            return
        if force:
            graceful = False
            if owner is not None:
                try:
                    owner.graceful_stop_active = False
                    owner._last_stop_was_graceful = False
                except Exception:
                    pass
            stop_control.apply_force_stop_flags()
            log("⚡ Force stop — aborting the current requests now")
        reset_progress = getattr(owner, "_reset_api_watchdog_progress", None)

        def clear_watchdog() -> None:
            if callable(reset_progress):
                reset_progress(clear_stale_external_files=True)
            else:
                stop_control.reset_api_watchdog(clear_stale_external_files=True)

        if not graceful:
            try:
                clear_watchdog()
            except Exception:
                pass

        def translation_stop_flag_hook() -> None:
            # Desktop: the lazily loaded ``translation_stop_flag`` (TransateKRtoEN.set_stop_flag)
            # once the backend is loaded.
            module = sys.modules.get("TransateKRtoEN")
            set_flag = getattr(module, "set_stop_flag", None) if module is not None else None
            if callable(set_flag):
                set_flag(True)

        def keep_cleanup_thread(thread: threading.Thread) -> None:
            self.stop_cleanup_thread = thread

        stop_control.request_stop(
            graceful=graceful,
            wait_for_chunks=wait_for_chunks,
            set_stop_requested=set_stop_requested,
            log=log,
            clear_watchdog=clear_watchdog,
            stop_flag_hook=translation_stop_flag_hook,
            cleanup_thread_created=keep_cleanup_thread,
        )
        if job_kind in ("compile_epub", "compile_pdf"):  # the converter runs in a compile job
            try:
                stop_control.stop_epub_converter()
            except Exception:
                pass
        stop_control.announce_stop(graceful, log)

    def progress_watcher(self, resolver: Callable[..., Any], host: Any, interval: float) -> Any:
        try:
            import job_runner
        except ImportError:
            return None
        cls = getattr(job_runner, "ProgressWatcher", None)
        return cls(resolver, host, interval=interval) if cls is not None else None

    def restore_in_progress(self, *, input_files: Sequence[str], output_dirs: Mapping[str, str], config: dict,
                            app_dir: Optional[str]) -> None:
        shutdown_utils = self._module("shutdown_utils")
        normalized = {_norm(k): v for k, v in (output_dirs or {}).items()}

        def resolver(path: str) -> Optional[str]:
            return normalized.get(_norm(path)) or (output_dirs or {}).get(path)

        shutdown_utils.restore_in_progress_rows_for_shutdown(
            input_files=list(input_files),
            entry_file=None,
            output_dir_resolver=resolver,
            config=config or {},
            app_dir=app_dir,
        )


# ---------------------------------------------------------------------------
# Job host (the owner's ``host``) and context (what adapters get)
# ---------------------------------------------------------------------------


class _JobHost:
    """``job_runner.JobHost`` implementation for one job (any thread may call it)."""

    def __init__(self, service: "JobService", job: _Job) -> None:
        self._service = service
        self._job = job
        self.job_id = job.id

    def log(self, text: Any = "", *args: Any, **kwargs: Any) -> None:
        self._service._job_log(self._job, text, kwargs)

    append_log = log  # desktop-style name

    def is_stop_requested(self) -> bool:
        return self._job.stop_event.is_set()

    def is_graceful_stop(self) -> bool:
        return self._job.stop_mode == STOP_GRACEFUL

    def emit(self, kind: str, **data: Any) -> None:
        self._service._job_event(self._job, str(kind), data)

    def ask(self, kind: str, **data: Any) -> Any:
        return self._service._job_ask(self._job, str(kind), data)


class _OutputDirResolver:
    """``ProgressWatcher`` resolver: ``resolver()`` -> the job's current output folder,
    ``resolver(input_path)`` -> the owner's ``_resolve_translation_output_dir``."""

    def __init__(self, job: _Job) -> None:
        self._job = job

    def __call__(self, *args: Any) -> Optional[str]:
        job = self._job
        if args:
            mapped = job.output_dirs.get(_norm(args[0])) or job.output_dirs.get(args[0])
            if mapped:
                return mapped
            owner = job.owner
            resolve = getattr(owner, "_resolve_translation_output_dir", None)
            if callable(resolve):
                try:
                    return resolve(args[0])
                except Exception:
                    return None
            return None
        return job.output_dir


class JobContext:
    """What a ``job_kinds`` adapter gets: the owner, the spec and a few reporting calls."""

    def __init__(self, service: "JobService", job: _Job, *, owner: Any, host: _JobHost, config: dict) -> None:
        self.service = service
        self._job = job
        self.owner = owner
        self.host = host
        self.config = config

    @property
    def job_id(self) -> str:
        return self._job.id

    @property
    def spec(self) -> JobSpec:
        return self._job.spec

    @property
    def params(self) -> Mapping[str, Any]:
        return self._job.spec.params

    @property
    def inputs(self) -> tuple:
        return self._job.spec.inputs

    @property
    def started(self) -> Optional[float]:
        return self._job.started

    @property
    def output_dir(self) -> Optional[str]:
        return self._job.output_dir

    def log(self, text: Any) -> None:
        self.host.log(text)

    def stop_requested(self) -> bool:
        return self.host.is_stop_requested()

    def phase(self, label: str) -> None:
        self.host.emit("phase", label=label)

    def set_output_dirs(self, mapping: Mapping[str, str]) -> None:
        """Record input -> output folder (checkpointed for recovery; first one feeds ProgressWatcher)."""
        self.host.emit("output_dirs", mapping=dict(mapping))

    def set_output_dir(self, path: Optional[str]) -> None:
        self.host.emit("output_dir", path=path)

    def add_outputs(self, paths: Iterable[str]) -> None:
        self.host.emit("outputs", paths=[os.fspath(p) for p in paths if p])

    def set_result(self, **data: Any) -> None:
        """Record JSON-safe facts about the run on ``JobSnapshot.result`` (e.g. the glossary
        the pipeline used), read by the front end after the job's process state is restored."""
        self.host.emit("run_result", **data)

    def ask(self, kind: str, **data: Any) -> Any:
        return self.host.ask(kind, **data)


# ---------------------------------------------------------------------------
# Request stream (the shared direct_text_stream model of the desktop Direct Text dialog)
# ---------------------------------------------------------------------------


def _make_request_stream(spec: JobSpec) -> Any:
    """The job's ``direct_text_stream.DirectTextStream`` (None when the module is missing).

    A chat send passes the run's attachment flag, the chat's next request number and the
    model (``params``: ``is_attachment``, ``request_number``, ``model``), like the dialog's
    ``_start_translation``; other kinds get a plain stream for the job detail cards.
    """
    try:
        import direct_text_stream  # shared (U3)
    except Exception:
        log.warning("direct_text_stream is not in this build; jobs show no request cards", exc_info=True)
        return None
    params = spec.params or {}
    request_number = params.get("request_number")
    try:
        request_number = int(request_number) if request_number else None
    except (TypeError, ValueError):
        request_number = None
    model = params.get("model") or (params.get("config_overrides") or {}).get("model") or None
    return direct_text_stream.make_stream(
        source_is_attachment=bool(params.get("is_attachment")),
        request_number=request_number,
        model=str(model) if model else None,
    )


# ---------------------------------------------------------------------------
# JobService
# ---------------------------------------------------------------------------


Listener = Callable[[JobsView], Any]
TransitionListener = Callable[[JobSnapshot, Optional[JobState]], Any]
QuestionListener = Callable[[JobSnapshot, Mapping[str, Any]], Any]
EventListener = Callable[[str, str, Mapping[str, Any]], Any]


class JobService:
    def __init__(
        self,
        *,
        jobs_dir: Any,
        logs_dir: Any = None,
        data_dir: Any = None,
        config_store: Any = None,
        config_loader: Optional[Callable[[], dict]] = None,
        dispatcher: Any = None,
        backend: Any = None,
        kinds: Optional[Callable[[str], Any]] = None,
        history_limit: int = HISTORY_LIMIT,
        checkpoint_interval: float = CHECKPOINT_INTERVAL,
        watcher_interval: float = 2.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.jobs_dir = os.fspath(jobs_dir)
        self.logs_dir = os.fspath(logs_dir) if logs_dir is not None else os.path.join(self.jobs_dir, "logs")
        self.data_dir = os.fspath(data_dir) if data_dir is not None else None
        self.config_store = config_store
        self._config_loader = config_loader
        self.dispatcher = dispatcher
        self.backend = backend if backend is not None else JobBackend()
        self._kinds = kinds
        self.history_limit = max(1, int(history_limit))
        self.checkpoint_interval = float(checkpoint_interval)
        self.watcher_interval = float(watcher_interval)
        self.clock = clock

        self._lock = threading.RLock()
        self._cond = threading.Condition(self._lock)
        self._queue: list[_Job] = []
        self._active: Optional[_Job] = None
        self._recent: list[_Job] = []  # finished jobs whose buffers are kept
        self._history: list[JobSnapshot] = []
        self._interrupted: list[JobSnapshot] = []
        self._paused = False
        self._closed = False
        self._thread: Optional[threading.Thread] = None
        self._answers: dict[str, tuple[threading.Event, list]] = {}
        self.adopted_ids: list[str] = []  # interrupted jobs found at this launch (launch banner)
        # Stop protocols requested on the UI loop run in order on one short-lived ``gl-job-stop``
        # thread (env writes, backend imports, watchdog resets never block the loop). The lock is
        # held while a protocol runs; a job takes it around its run reset, so a late protocol of
        # the previous job can never land on (or be erased by) the next job's start.
        self._stop_tasks: list[tuple[_Job, str]] = []
        self._stop_thread: Optional[threading.Thread] = None
        self._stop_lock = threading.RLock()
        self.stream_drain_interval = STREAM_DRAIN_INTERVAL

        self._listeners: list[Listener] = []
        self._transition_listeners: list[TransitionListener] = []
        self._question_listeners: list[QuestionListener] = []
        self._event_listeners: list[EventListener] = []
        self._channel = None
        if dispatcher is not None and hasattr(dispatcher, "channel"):
            self._channel = dispatcher.channel("jobs.view", None)
            self._channel.subscribe(lambda _token: self._deliver_view())

        os.makedirs(self.jobs_dir, exist_ok=True)
        self._load_history()
        self._adopt_leftovers()

    # ---- paths ----------------------------------------------------------------------------

    @property
    def active_state_path(self) -> str:
        return os.path.join(self.jobs_dir, ACTIVE_STATE_FILE)

    @property
    def history_state_path(self) -> str:
        return os.path.join(self.jobs_dir, HISTORY_STATE_FILE)

    @property
    def interrupted_state_path(self) -> str:
        return os.path.join(self.jobs_dir, INTERRUPTED_STATE_FILE)

    def job_log_path(self, job_id: str) -> str:
        return os.path.join(self.logs_dir, "jobs", f"{job_id}.log")

    # ---- kinds --------------------------------------------------------------------------------

    def _kind(self, kind: str) -> Any:
        if self._kinds is not None:
            return self._kinds(kind)
        from glossarion_mobile.job_kinds import get_kind

        return get_kind(kind)

    def has_kind(self, kind: str) -> bool:
        try:
            return self._kind(kind) is not None
        except Exception:
            return False

    # ---- listeners ------------------------------------------------------------------------------

    def subscribe(self, callback: Listener, *, immediate: bool = False) -> Callable[[], None]:
        """``callback(JobsView)`` on every change (progress coalesced)."""
        self._listeners.append(callback)
        if immediate:
            callback(self.view())
        return self._remover(self._listeners, callback)

    def on_transition(self, callback: TransitionListener) -> Callable[[], None]:
        """``callback(snapshot, previous_state)`` for every state change, in order."""
        self._transition_listeners.append(callback)
        return self._remover(self._transition_listeners, callback)

    def on_question(self, callback: QuestionListener) -> Callable[[], None]:
        """``callback(snapshot, question)`` when a job asks the UI (glossary approval, ...).

        The job thread blocks until ``answer(question['id'], value)``. Without
        any question listener ``ask`` returns the question's ``default``.
        """
        self._question_listeners.append(callback)
        return self._remover(self._question_listeners, callback)

    def on_event(self, callback: EventListener) -> Callable[[], None]:
        """``callback(job_id, kind, data)`` for job events the service does not consume itself."""
        self._event_listeners.append(callback)
        return self._remover(self._event_listeners, callback)

    @staticmethod
    def _remover(items: list, callback: Any) -> Callable[[], None]:
        def remove() -> None:
            try:
                items.remove(callback)
            except ValueError:
                pass

        return remove

    def _bound(self) -> bool:
        dispatcher = self.dispatcher
        return dispatcher is not None and getattr(dispatcher, "bound", False)

    def _post(self, fn: Callable[..., Any], *args: Any) -> None:
        if self._bound():
            if self.dispatcher.post(fn, *args):
                return
        try:
            fn(*args)
        except Exception:
            log.exception("job listener failed")

    def _deliver_view(self) -> None:
        view = self.view()
        for callback in list(self._listeners):
            try:
                callback(view)
            except Exception:
                log.exception("jobs view listener failed")

    def _deliver_transition(self, snap: JobSnapshot, previous: Optional[JobState]) -> None:
        for callback in list(self._transition_listeners):
            try:
                callback(snap, previous)
            except Exception:
                log.exception("job transition listener failed")
        self._deliver_view()

    def _changed(self) -> None:
        """Progress-type change: coalesced (latest wins) when a dispatcher is bound."""
        if self._channel is not None and self._bound():
            self._channel.set(object())
        else:
            self._deliver_view()

    # ---- view ---------------------------------------------------------------------------------------

    def view(self) -> JobsView:
        with self._lock:
            return JobsView(
                active=self._active.snapshot() if self._active is not None else None,
                queue=tuple(job.snapshot() for job in self._queue),
                interrupted=tuple(self._interrupted),
                history=tuple(self._history),
                paused=self._paused,
            )

    def snapshot(self, job_id: Optional[str] = None) -> Optional[JobSnapshot]:
        """The active job (``job_id=None``) or any known job."""
        with self._lock:
            if job_id is None:
                return self._active.snapshot() if self._active is not None else None
            job = self._find_live(job_id)
            if job is not None:
                return job.snapshot()
        return self.view().find(job_id)

    def _find_live(self, job_id: str) -> Optional[_Job]:
        if self._active is not None and self._active.id == job_id:
            return self._active
        for job in self._queue:
            if job.id == job_id:
                return job
        for job in self._recent:
            if job.id == job_id:
                return job
        return None

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._active is not None

    @property
    def paused(self) -> bool:
        with self._lock:
            return self._paused

    # ---- submit / queue -------------------------------------------------------------------------------

    def submit(self, spec: JobSpec, *, front: bool = False) -> str:
        """Queue ``spec`` (starts at once when idle); returns the job id."""
        if not isinstance(spec, JobSpec):
            raise TypeError("submit() takes a JobSpec")
        if not self.has_kind(spec.kind):
            raise JobError(f"Unknown job kind: {spec.kind}")
        job = _Job(id=uuid.uuid4().hex[:12], spec=spec, created=self.clock())
        job.log_path = self.job_log_path(job.id)
        with self._cond:
            if self._closed:
                raise JobError("The job service is closed")
            if front:
                self._queue.insert(0, job)
            else:
                self._queue.append(job)
            self._write_active_locked()
            snap = job.snapshot()
            self._ensure_worker_locked()
            self._cond.notify_all()
        log.info("job %s queued: %s %s", job.id, spec.kind, spec.title)
        self._post(self._deliver_transition, snap, None)
        job.announced.set()  # the worker starts it only now: listeners see QUEUED before STARTING
        return job.id

    def cancel_queued(self, job_id: str) -> bool:
        with self._cond:
            job = next((j for j in self._queue if j.id == job_id), None)
            if job is None:
                return False
            self._queue.remove(job)
            job.state = JobState.CANCELLED
            job.finished = self.clock()
            snap = job.snapshot()
            self._append_history_locked(snap)
            self._write_active_locked()
        self._post(self._deliver_transition, snap, JobState.QUEUED)
        return True

    def cancel_all_queued(self) -> int:
        with self._lock:
            ids = [job.id for job in self._queue]
        return sum(1 for job_id in ids if self.cancel_queued(job_id))

    def move_queued(self, job_id: str, index: int) -> bool:
        """Reorder the queue (Jobs page drag handles)."""
        with self._lock:
            job = next((j for j in self._queue if j.id == job_id), None)
            if job is None:
                return False
            self._queue.remove(job)
            index = max(0, min(int(index), len(self._queue)))
            self._queue.insert(index, job)
            self._write_active_locked()
        self._changed()
        return True

    def pause_queue(self) -> None:
        """Stop starting the next queued job (the running one continues)."""
        with self._cond:
            self._paused = True
        self._changed()

    def resume_queue(self) -> None:
        with self._cond:
            self._paused = False
            if self._queue:
                self._ensure_worker_locked()
            self._cond.notify_all()
        self._changed()

    def clear_finished(self) -> None:
        with self._lock:
            self._history = []
            self._write_history_locked()
        self._changed()

    # ---- interrupted / resume -------------------------------------------------------------------------------

    def resume(self, job_id: str) -> Optional[str]:
        """Resubmit an interrupted, stopped or failed job's spec (the backend resumes from its progress).

        An Interrupted job that was already resumed (its entry resolved ``resumed``) is refused:
        a second Resume (launch banner, Jobs page, the chat) would run the same work twice."""
        with self._lock:
            snap = next((s for s in self._interrupted if s.id == job_id), None)
            from_interrupted = snap is not None
            if snap is None:
                snap = next((s for s in self._history if s.id == job_id), None)
                if snap is not None and snap.resolution == "resumed":
                    return None
        if snap is None:
            return None
        new_id = self.submit(snap.spec)
        if from_interrupted:
            self._resolve_interrupted(job_id, "resumed")
        return new_id

    retry = resume

    def discard(self, job_id: str) -> bool:
        return self._resolve_interrupted(job_id, "discarded")

    def _resolve_interrupted(self, job_id: str, resolution: str) -> bool:
        with self._lock:
            snap = next((s for s in self._interrupted if s.id == job_id), None)
            if snap is None:
                return False
            self._interrupted.remove(snap)
            self._append_history_locked(replace(snap, resolution=resolution))
            self._write_interrupted_locked()
        self._changed()
        return True

    @property
    def interrupted(self) -> tuple:
        with self._lock:
            return tuple(self._interrupted)

    # ---- stop ---------------------------------------------------------------------------------------------------

    def _graceful_setting(self, job: _Job) -> tuple[bool, bool]:
        """(graceful_stop, wait_for_chunks) from the owner, else the job's config (desktop defaults True)."""
        owner = job.owner
        if owner is not None and hasattr(owner, "graceful_stop_var"):
            graceful = bool(getattr(owner, "graceful_stop_var", True))
            wait = bool(getattr(owner, "wait_for_chunks_var", True))
            return graceful, wait
        config: Mapping[str, Any] = {}
        store = self.config_store
        if store is not None:
            try:
                config = {"graceful_stop": store.get("graceful_stop", True),
                          "wait_for_chunks": store.get("wait_for_chunks", True)}
            except Exception:
                config = {}
        return bool(config.get("graceful_stop", True)), bool(config.get("wait_for_chunks", True))

    def graceful_stop(self) -> bool:
        """Whether the next first Stop is graceful (the running job's owner, else the config)."""
        with self._lock:
            job = self._active
        if job is not None:
            return self._graceful_setting(job)[0]
        store = self.config_store
        try:
            return bool(store.get("graceful_stop", True)) if store is not None else True
        except Exception:
            return True

    def request_stop(self, job_id: Optional[str] = None, *, force: bool = False, escalate: bool = True,
                     reason: str = "") -> Optional[str]:
        """Stop the running job (or cancel a queued one). Returns the stop mode used.

        First request: graceful when the config's graceful stop is on (state
        STOPPING), otherwise immediate (FORCE_STOPPING). A further request
        forces (``escalate``; user taps). System-initiated stops (Android FGS
        timeout, iOS expiry) pass ``escalate=False`` so a second one never forces.

        The job's own latch (``stop_event``) and state change at once; the stop protocol
        (``JobBackend.request_stop``: env flags, backend imports, watchdog reset) runs on the
        ``gl-job-stop`` thread when called on the UI loop (UI_SPEC §2.4: the button never
        touches ``os.environ``), inline from any other thread.
        """
        with self._lock:
            active = self._active
            if job_id is not None and (active is None or active.id != job_id):
                if any(j.id == job_id for j in self._queue):
                    queued_id: Optional[str] = job_id
                else:
                    queued_id = None
                job = None
            else:
                queued_id = None
                job = active if active is not None and active.state not in TERMINAL_STATES else None
        if job is None:
            if queued_id is not None:
                return "cancelled" if self.cancel_queued(queued_id) else None
            return None
        with self._lock:
            previous = job.state
            already = job.stop_mode is not None
            if already and not escalate and not force:
                return job.stop_mode
            graceful_enabled, _wait = self._graceful_setting(job)
            if force or already:
                mode = STOP_FORCE
            elif graceful_enabled:
                mode = STOP_GRACEFUL
            else:
                mode = STOP_IMMEDIATE
            job.stop_mode = mode
            job.stop_requested_at = self.clock()
            job.state = JobState.STOPPING if mode == STOP_GRACEFUL else JobState.FORCE_STOPPING
            snap = job.snapshot()
        # The job's own latch at once: queued questions give up, a starting job is cancelled
        # before its run reset and the adapters see the Stop (the backend reads the owner's
        # latch, which the protocol sets after the env mode flags, in the desktop order).
        job.stop_event.set()
        log.info("job %s stop requested (%s)%s", job.id, mode, f": {reason}" if reason else "")
        if reason:
            self._job_log(job, f"⏹️ {reason}", {})
        if self._on_ui_loop():
            self._issue_stop_soon(job, mode)
        else:
            with self._stop_lock:
                self._issue_stop(job, mode)
        if snap.state is not previous:
            self._post(self._deliver_transition, snap, previous)
        else:
            self._changed()
        return mode

    def _on_ui_loop(self) -> bool:
        dispatcher = self.dispatcher
        if dispatcher is None or not self._bound():
            return False
        on_loop = getattr(dispatcher, "on_loop_thread", None)
        try:
            return bool(on_loop()) if callable(on_loop) else False
        except Exception:
            return False

    def _issue_stop_soon(self, job: _Job, mode: str) -> None:
        """Run the stop protocol on the ``gl-job-stop`` thread (requests keep their order)."""
        with self._lock:
            self._stop_tasks.append((job, mode))
            if self._stop_thread is None:
                thread = threading.Thread(target=self._stop_worker, name="gl-job-stop", daemon=True)
                self._stop_thread = thread
                thread.start()

    def _stop_worker(self) -> None:
        while True:
            with self._lock:
                if not self._stop_tasks:
                    self._stop_thread = None
                    return
                job, mode = self._stop_tasks.pop(0)
            try:
                with self._stop_lock:
                    with self._lock:
                        # a protocol for a job that already ended must not reach the next one
                        current = job is self._active and job.state not in TERMINAL_STATES
                        repeat = job.issued_stop == mode
                    if current and not repeat:
                        self._issue_stop(job, mode)
            except Exception:
                log.exception("the stop protocol failed for job %s", job.id)

    def wait_stops_idle(self, timeout: Optional[float] = None) -> bool:
        """Block until no stop protocol is queued or running (tests, shutdown)."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            with self._lock:
                busy = bool(self._stop_tasks) or self._stop_thread is not None
            if not busy:
                return True
            if deadline is not None and time.monotonic() >= deadline:
                return False
            time.sleep(0.01)

    def _issue_stop(self, job: _Job, mode: str) -> None:
        graceful = mode == STOP_GRACEFUL
        with self._lock:
            job.issued_stop = mode
        latched: list = []

        def set_stop_requested() -> None:
            # The owner attributes the desktop latch sets (stop_translation's
            # _latch_stop_requested); the shared pipelines read graceful_stop_active.
            latched.append(True)
            job.stop_event.set()
            owner = job.owner
            if owner is not None:
                try:
                    owner.graceful_stop_active = graceful
                    owner._last_stop_translation_ts = time.time()
                    owner._last_stop_was_graceful = graceful
                    owner.stop_requested = True
                except Exception:
                    pass

        _graceful, wait = self._graceful_setting(job)
        try:
            stop_kind = getattr(self._kind(job.spec.kind), "stop_kind", "translation") or "translation"
        except Exception:
            stop_kind = "translation"
        try:
            self.backend.request_stop(
                graceful=graceful,
                wait_for_chunks=bool(wait),
                force=mode == STOP_FORCE,
                set_stop_requested=set_stop_requested,
                log=lambda text="", *a, **k: self._job_log(job, text, {}),
                kind=stop_kind,
                job_kind=job.spec.kind,
                owner=job.owner,
            )
        except Exception as exc:
            log.warning("stop_control.request_stop failed (%s); latching the stop flag only", exc)
        if not latched:  # the latch must always end up set
            set_stop_requested()

    # ---- questions ----------------------------------------------------------------------------------------------

    def answer(self, question_id: str, value: Any) -> bool:
        with self._lock:
            entry = self._answers.get(question_id)
            if entry is None:
                return False
            event, slot = entry
            slot[0] = value
            event.set()
        return True

    def pending_question(self, job_id: Optional[str] = None) -> Optional[dict]:
        with self._lock:
            job = self._active if job_id is None else self._find_live(job_id)
            return dict(job.question) if job is not None and job.question else None

    def _job_ask(self, job: _Job, kind: str, data: Mapping[str, Any]) -> Any:
        data = dict(data)
        default = data.pop("default", None)
        if not self._question_listeners:
            return default
        question = {"id": uuid.uuid4().hex[:12], "job_id": job.id, "kind": kind, "data": data,
                    "asked_at": self.clock(), "default": default}
        event = threading.Event()
        slot = [default]
        with self._lock:
            self._answers[question["id"]] = (event, slot)
            job.question = question
            job.warning = True
            snap = job.snapshot()
        self._post(self._deliver_question, snap, question)
        self._changed()
        try:
            while not event.wait(0.25):
                if job.stop_event.is_set():
                    break
            return slot[0]
        finally:
            with self._lock:
                self._answers.pop(question["id"], None)
                job.question = None
                job.warning = False
            self._changed()

    def _deliver_question(self, snap: JobSnapshot, question: Mapping[str, Any]) -> None:
        for callback in list(self._question_listeners):
            try:
                callback(snap, question)
            except Exception:
                log.exception("job question listener failed")

    # ---- worker -------------------------------------------------------------------------------------------------

    def _ensure_worker_locked(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._worker, name="gl-job", daemon=True)
        self._thread.start()

    def _worker(self) -> None:
        while True:
            with self._cond:
                while not self._closed and (self._paused or not self._queue):
                    self._cond.wait()
                if self._closed:
                    return
                job = self._queue.pop(0)
                self._active = job
            job.announced.wait(5.0)  # submit() posts QUEUED right after queueing; keep the order
            try:
                self._execute(job)
            except BaseException:  # noqa: BLE001 - the worker must survive anything
                log.exception("job worker crashed on %s", job.id)
            finally:
                with self._cond:
                    if self._active is job:
                        self._active = None
                    self._cond.notify_all()

    def _transition(self, job: _Job, state: JobState, *, only_from: Optional[Iterable[JobState]] = None) -> None:
        with self._lock:
            previous = job.state
            if only_from is not None and previous not in set(only_from):
                return
            if previous is state:
                return
            job.state = state
            snap = job.snapshot()
        self._post(self._deliver_transition, snap, previous)

    def _config_snapshot(self, job: _Job) -> dict:
        store = self.config_store
        config: dict = {}
        if store is not None:
            try:
                store.flush()
            except Exception:
                log.exception("config flush at job start failed")
            config = store.snapshot()
            try:
                store.set_job_running(True)
            except Exception:
                log.debug("set_job_running failed", exc_info=True)
        elif self._config_loader is not None:
            config = dict(self._config_loader() or {})
        config = copy.deepcopy(config if isinstance(config, dict) else {})
        overrides = job.spec.params.get("config_overrides")
        if isinstance(overrides, Mapping):
            config.update(copy.deepcopy(dict(overrides)))
        return config

    @contextlib.contextmanager
    def _watching(self, job: _Job, host: _JobHost) -> Iterator[None]:
        watcher = None
        try:
            watcher = self.backend.progress_watcher(_OutputDirResolver(job), host, self.watcher_interval)
        except Exception:
            log.exception("ProgressWatcher could not start")
        started = False
        if watcher is not None:
            try:
                if hasattr(watcher, "start"):
                    watcher.start()
                    started = True
                elif hasattr(watcher, "__enter__"):
                    watcher.__enter__()
                    started = True
            except Exception:
                log.exception("ProgressWatcher.start failed")
        try:
            yield
        finally:
            if watcher is not None and started:
                try:
                    if hasattr(watcher, "stop"):
                        watcher.stop()
                    elif hasattr(watcher, "__exit__"):
                        watcher.__exit__(None, None, None)
                except Exception:
                    log.exception("ProgressWatcher.stop failed")

    @contextlib.contextmanager
    def _draining(self, job: _Job) -> Iterator[None]:
        """Keep the job's request stream classified on a ``gl-job-stream`` thread while it runs.

        The budgeted drain (the desktop's 12 ms / 1200-record repaint budget) every
        ``stream_drain_interval`` s, so no backlog builds up while no screen drains it (a
        repaint then only copies the cards); whatever is left is drained when the job ends,
        still on the job thread."""
        stream = job.stream
        drain = getattr(stream, "drain", None) if stream is not None else None
        if not callable(drain):
            yield
            return
        stop = threading.Event()
        interval = max(0.01, float(self.stream_drain_interval))

        def loop() -> None:
            while not stop.wait(interval):
                try:
                    drain(final=False)
                except Exception:
                    log.debug("request stream drain failed", exc_info=True)

        thread = threading.Thread(target=loop, name="gl-job-stream", daemon=True)
        thread.start()
        try:
            yield
        finally:
            stop.set()
            thread.join(5.0)
            try:
                drain(final=True)
            except Exception:
                log.debug("the final request stream drain failed", exc_info=True)

    def _execute(self, job: _Job) -> None:
        try:
            kind = self._kind(job.spec.kind)
        except Exception as exc:
            job.started = self.clock()
            self._finish(job, None, f"Unknown job kind: {job.spec.kind} ({exc})")
            return
        host = _JobHost(self, job)
        with self._lock:
            job.started = self.clock()
            job.buffer = LogBuffer(JOB_LOG_MAXLEN, name=f"job:{job.id}")
            try:
                job.stream = _make_request_stream(job.spec)
            except Exception:
                log.exception("the request stream could not be created")
                job.stream = None
        self._transition(job, JobState.STARTING, only_from={JobState.QUEUED})
        self._checkpoint(job, force=True)
        self._job_log(job, f"▶ {kind_verb(job.spec.kind)} {job.spec.title}", {})
        result: Any = None
        error: Optional[str] = None
        exit_ok = True
        config_taken = False
        try:
            if job.stop_event.is_set():
                raise _Cancelled()
            config = self._config_snapshot(job)
            config_taken = True
            lock = self.backend.job_lock
            with self.backend.scoped_process_state(capture_stdout=host.log, lock=lock):
                # Held from the run reset to the owner latch: a stop protocol still running on
                # gl-job-stop (the previous job's, or this job's own) never interleaves with them.
                with self._stop_lock:
                    # The previous immediate stop's cleanup closes HTTP transports on a helper
                    # thread; do not reset the cancellation state underneath it (desktop preflight).
                    wait_cleanup = getattr(self.backend, "wait_for_stop_cleanup", None)
                    if callable(wait_cleanup) and not wait_cleanup(lambda text="", *a, **k: self._job_log(job, text, {})):
                        raise JobError("The previous translation is still stopping; tap Retry in a moment.")
                    if job.stop_event.is_set():
                        raise _Cancelled()
                    stop_kind = getattr(kind, "stop_kind", "translation") or "translation"
                    job.run_id = self.backend.reset_for_new_run(stop_kind)
                    owner = self.backend.make_owner(config, host=host)
                    with self._lock:
                        job.owner = owner
                        pending_mode = job.stop_mode
                    if pending_mode is not None:
                        # A stop arrived while the run was starting: reset_for_new_run cleared its
                        # flags. (A Stop before the shared set-up's own "Reset stop flags" block is
                        # caught by the adapter through ``ctx.stop_requested()``.)
                        self._issue_stop(job, pending_mode)
                self._transition(job, JobState.RUNNING, only_from={JobState.STARTING})
                ctx = JobContext(self, job, owner=owner, host=host, config=config)
                with self._watching(job, host), self._draining(job):
                    result = kind.run(ctx)
        except _Cancelled:
            pass
        except JobError as exc:
            error = str(exc)
            self._job_log(job, f"❌ {error}", {})
        except SystemExit as exc:  # backend mains may sys.exit(); the thread survives
            exit_ok = exc.code in (None, 0)
            if not exit_ok:
                error = f"The backend exited with code {exc.code}"
        except BaseException as exc:  # noqa: BLE001
            error = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            self._job_log(job, "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)), {"kind": "error"})
        finally:
            with self._lock:
                job.owner = None
            if config_taken and self.config_store is not None:
                try:
                    self.config_store.set_job_running(False)
                except Exception:
                    log.debug("set_job_running(False) failed", exc_info=True)
        self._finish(job, result, error)

    def _finish(self, job: _Job, result: Any, error: Optional[str]) -> None:
        from glossarion_mobile.job_kinds import result_fields

        ok, outputs, result_error = result_fields(result)
        with self._lock:
            previous = job.state
            for path in outputs:
                if path not in job.outputs:
                    job.outputs.append(path)
            if error is None and result_error and ok is False:
                error = result_error
            if job.stop_mode is not None and (error is None or job.stop_event.is_set()):
                state = JobState.CANCELLED
                if error and job.stop_mode == STOP_GRACEFUL:
                    job.error = error
            elif error is not None or ok is False:
                state = JobState.FAILED
                job.error = error or result_error or "The job failed"
            else:
                state = JobState.DONE
            job.state = state
            job.finished = self.clock()
            job.in_flight = 0
            job.question = None
            job.warning = False
            snap = job.snapshot()
            self._append_history_locked(snap)
            self._recent.insert(0, job)
            del self._recent[JOB_BUFFERS_KEPT:]
            self._active = None if self._active is job else self._active
            self._write_active_locked()
        self._job_log(job, f"■ {snap.state_label}: {job.spec.title}" + (f" — {snap.error}" if snap.error else ""), {})
        self._close_log(job)
        self._post(self._deliver_transition, snap, previous)

    # ---- events from the job (any thread) ------------------------------------------------------------------------

    def _job_log(self, job: _Job, text: Any, kwargs: Mapping[str, Any]) -> None:
        raw = "" if text is None else str(text)
        # The request stream gets every message whole, blank ones included, like the desktop
        # Direct Text listener (``append_log`` -> ``_on_log_line(message)``): the classifier
        # splits it itself and keeps blank lines inside streamed text (paragraph breaks).
        # Its "suppress-main-log" answer covers the whole message.
        suppress = False
        stream = job.stream
        if stream is not None:
            try:
                suppress = stream.feed(raw) == "suppress-main-log"
            except Exception:
                log.debug("request stream rejected a message", exc_info=True)
        if not raw.strip():
            return
        kind = kwargs.get("kind") if isinstance(kwargs, Mapping) else None
        if kind not in ("info", "error", "thinking", "api"):
            kind = None
        for line in raw.replace("\r\n", "\n").split("\n"):
            if not line.strip():
                continue
            self._write_log_file(job, line)
            if not job.sign_in_flagged and any(marker in line for marker in SIGN_IN_MARKERS):
                job.sign_in_flagged = True
                self._post(self._deliver_event, job.id, "sign_in_required", {"provider": "authgpt", "line": line[:200]})
            if suppress:
                continue
            buffer = job.buffer
            if buffer is not None:
                buffer.append(line, kind)
            job.last_line = line[:LAST_LINE_CHARS]

    def _write_log_file(self, job: _Job, line: str) -> None:
        if not job.log_path:
            return
        with job.log_lock:
            try:
                if job.log_file is None:
                    os.makedirs(os.path.dirname(job.log_path), exist_ok=True)
                    job.log_file = open(job.log_path, "a", encoding="utf-8", errors="replace")
                job.log_file.write(line + "\n")
                job.log_file.flush()
            except Exception:
                job.log_path = None  # stop trying for this job

    def _close_log(self, job: _Job) -> None:
        with job.log_lock:
            handle, job.log_file = job.log_file, None
        if handle is not None:
            try:
                handle.close()
            except Exception:
                pass

    def _job_event(self, job: _Job, kind: str, data: Mapping[str, Any]) -> None:
        if kind == "log":
            self._job_log(job, data.get("text", data.get("message", "")), data)
            return
        consumed = True
        with self._lock:
            if kind == "progress":
                job.progress = Progress.from_dict({**job.progress.to_dict(), **{
                    k: v for k, v in data.items() if k in ("total", "completed", "in_progress", "failed", "label")}})
            elif kind == "api_state":
                in_flight = data.get("in_flight")
                if in_flight is not None:
                    try:
                        job.in_flight = max(0, int(in_flight))
                    except (TypeError, ValueError):
                        pass
                for key in ("peak_in_flight", "peak"):
                    if data.get(key) is not None:
                        try:
                            job.peak_in_flight = max(job.peak_in_flight, int(data[key]))
                        except (TypeError, ValueError):
                            pass
                job.peak_in_flight = max(job.peak_in_flight, job.in_flight)
                for key in ("queued", "backlog", "scheduler_queued"):
                    if data.get(key) is not None:
                        try:
                            job.api_queued = int(data[key])
                        except (TypeError, ValueError):
                            pass
                        break
            elif kind == "phase":
                job.phase = str(data.get("label") or data.get("phase") or "")
            elif kind == "run_result":
                job.result.update(_jsonable(dict(data)))
            elif kind == "output_dirs":
                mapping = data.get("mapping") or {}
                for source, target in dict(mapping).items():
                    if target:
                        job.output_dirs[_norm(source)] = os.fspath(target)
                        if job.output_dir is None:
                            job.output_dir = os.fspath(target)
            elif kind == "output_dir":
                path = data.get("path")
                job.output_dir = os.fspath(path) if path else None
            elif kind in ("outputs", "result"):
                for path in data.get("paths") or data.get("outputs") or ():
                    if path and os.fspath(path) not in job.outputs:
                        job.outputs.append(os.fspath(path))
                consumed = kind == "outputs"
            else:
                consumed = False
            checkpoint_due = kind in ("progress", "output_dirs") and (
                self.clock() - job.last_checkpoint >= self.checkpoint_interval or kind == "output_dirs")
        if checkpoint_due:
            self._checkpoint(job)
        if not consumed:
            payload = dict(data)
            self._post(self._deliver_event, job.id, kind, payload)
        self._changed()

    def _deliver_event(self, job_id: str, kind: str, data: Mapping[str, Any]) -> None:
        for callback in list(self._event_listeners):
            try:
                callback(job_id, kind, data)
            except Exception:
                log.exception("job event listener failed")

    # ---- logs and request cards for screens -------------------------------------------------------------------------

    def log_buffer(self, job_id: str) -> Optional[LogBuffer]:
        """The job's in-memory LogBuffer (running job and the last few finished ones)."""
        with self._lock:
            job = self._find_live(job_id)
            return job.buffer if job is not None else None

    def read_log_tail(self, job_id: str, max_lines: int = 2000) -> list[str]:
        """Last lines of ``<logs>/jobs/<id>.log`` (finished jobs whose buffer is gone)."""
        path = self.job_log_path(job_id)
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as handle:
                lines = handle.read().splitlines()
        except OSError:
            return []
        return lines[-max_lines:]

    def request_stream(self, job_id: Optional[str]) -> Any:
        """The job's ``direct_text_stream.DirectTextStream`` (running and recently finished jobs)."""
        if job_id is None:
            return None
        with self._lock:
            job = self._find_live(job_id)
            return job.stream if job is not None else None

    def request_segments(self, job_id: str) -> list:
        """Copies of the job's live request segments, in spine order ([] when it has none).

        Never drains (screens call it on the UI loop): the job-side drain keeps the cards
        current while the job runs and drains the rest when it ends."""
        stream = self.request_stream(job_id)
        if stream is None:
            return []
        try:
            try:
                return list(stream.segments(drain=False))
            except TypeError:  # a stream without the keyword
                return list(stream.segments())
        except Exception:
            log.debug("reading the request segments failed", exc_info=True)
            return []

    def has_request_stream(self, job_id: str) -> bool:
        with self._lock:
            job = self._find_live(job_id)
            return job is not None and job.stream is not None

    # ---- persistence ----------------------------------------------------------------------------------------------------

    def _active_payload_locked(self) -> dict:
        active = self._active.snapshot().to_dict() if self._active is not None else None
        return {
            "version": STATE_VERSION,
            "saved_at": self.clock(),
            "active": active,
            "queue": [job.snapshot().to_dict() for job in self._queue],
        }

    def _write_active_locked(self) -> None:
        payload = self._active_payload_locked()
        path = self.active_state_path
        try:
            if payload["active"] is None and not payload["queue"]:
                if os.path.exists(path):
                    os.remove(path)
                return
            _write_json(path, payload)
        except Exception:
            log.exception("writing %s failed", path)

    def _checkpoint(self, job: Optional[_Job] = None, *, force: bool = False) -> None:
        with self._lock:
            if job is not None:
                now = self.clock()
                if not force and now - job.last_checkpoint < self.checkpoint_interval:
                    return
                job.last_checkpoint = now
            self._write_active_locked()

    def checkpoint(self) -> None:
        """Rewrite ``active.state`` now (lifecycle INACTIVE / HIDE)."""
        self._checkpoint(force=True)

    def _append_history_locked(self, snap: JobSnapshot) -> None:
        self._history = [s for s in self._history if s.id != snap.id]
        self._history.insert(0, snap)
        del self._history[self.history_limit:]
        self._write_history_locked()

    def _write_history_locked(self) -> None:
        try:
            _write_json(self.history_state_path, {"version": STATE_VERSION,
                                                   "jobs": [s.to_dict() for s in self._history]})
        except Exception:
            log.exception("writing %s failed", self.history_state_path)

    def _write_interrupted_locked(self) -> None:
        path = self.interrupted_state_path
        try:
            if not self._interrupted:
                if os.path.exists(path):
                    os.remove(path)
                return
            _write_json(path, {"version": STATE_VERSION, "jobs": [s.to_dict() for s in self._interrupted]})
        except Exception:
            log.exception("writing %s failed", path)

    def _load_history(self) -> None:
        def load(path: str) -> list[JobSnapshot]:
            data = _read_json(path)
            jobs = data.get("jobs") if isinstance(data, dict) else None
            out = []
            for entry in jobs or ():
                if isinstance(entry, dict):
                    try:
                        out.append(JobSnapshot.from_dict(entry))
                    except Exception:
                        log.debug("skipping a bad job entry", exc_info=True)
            return out

        with self._lock:
            self._history = load(self.history_state_path)[: self.history_limit]
            self._interrupted = load(self.interrupted_state_path)

    # ---- recovery -----------------------------------------------------------------------------------------------------

    def _adopt_leftovers(self) -> None:
        """Move a previous launch's ``active.state`` into ``interrupted.state`` (JSON only).

        Runs in ``__init__``, before anything can rewrite ``active.state``, so a job
        submitted right after launch never overwrites what the killed run left behind.
        Jobs that had started are marked ``restore_pending``; ``recover()`` restores
        their progress rows later (that needs the backend)."""
        data = _read_json(self.active_state_path)
        if not isinstance(data, dict):
            return
        entries: list[dict] = []
        if isinstance(data.get("active"), dict):
            entries.append(data["active"])
        entries.extend(e for e in data.get("queue") or () if isinstance(e, dict))
        adopted: list[JobSnapshot] = []
        for entry in entries:
            try:
                snap = JobSnapshot.from_dict(entry)
            except Exception:
                continue
            pending = snap.started is not None and bool(snap.spec.inputs or snap.output_dirs)
            adopted.append(replace(snap, state=JobState.INTERRUPTED, recovered=True, in_flight=0, question=None,
                                   finished=snap.finished or data.get("saved_at"), restore_pending=pending))
        with self._lock:
            known = {s.id for s in self._interrupted}
            for snap in adopted:
                if snap.id not in known:
                    self._interrupted.insert(0, snap)
                    self.adopted_ids.append(snap.id)
            self._write_interrupted_locked()
            try:
                os.remove(self.active_state_path)
            except OSError:
                pass
        if adopted:
            log.info("found %d interrupted job(s) from the previous launch", len(adopted))

    def recover(self) -> list[JobSnapshot]:
        """Restore the progress rows of interrupted jobs (blocking; run off the UI loop).

        Each recovered job that had started gets the in-progress rows of its
        progress files restored with ``shutdown_utils.restore_in_progress_rows_for_shutdown``
        (under the job lock), exactly as the desktop does before a forced exit;
        Resume then resubmits the same spec and the backend continues from
        ``translation_progress.json``. Queued jobs that never started are listed
        as Interrupted too. Returns the Interrupted jobs.
        """
        self._adopt_leftovers()  # a no-op unless active.state reappeared
        with self._lock:
            pending = [s for s in self._interrupted if s.restore_pending]
        if not pending:
            return list(self.interrupted)
        config: dict = {}
        if self.config_store is not None:
            try:
                config = self.config_store.snapshot()
            except Exception:
                config = {}
        done: set[str] = set()
        for snap in pending:
            lock = self.backend.job_lock
            acquired = lock.acquire(timeout=60) if hasattr(lock, "acquire") else False
            try:
                self.backend.restore_in_progress(
                    input_files=snap.spec.inputs,
                    output_dirs=snap.output_dirs,
                    config=config,
                    app_dir=self.data_dir,
                )
            except Exception as exc:
                log.warning("restoring in-progress rows of %s failed: %s", snap.id, exc)
            finally:
                if acquired:
                    lock.release()
            done.add(snap.id)
        with self._lock:
            self._interrupted = [replace(s, restore_pending=False) if s.id in done else s for s in self._interrupted]
            self._write_interrupted_locked()
        log.info("restored the progress rows of %d interrupted job(s)", len(done))
        self._changed()
        return list(self.interrupted)

    # ---- lifecycle ------------------------------------------------------------------------------------------------------

    def wait_idle(self, timeout: Optional[float] = None) -> bool:
        """Block until nothing runs and nothing (unpaused) is queued (tests, shutdown)."""
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cond:
            while self._active is not None or (self._queue and not self._paused):
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return False
                self._cond.wait(remaining if remaining is not None else 0.5)
        return True

    def close(self, timeout: float = 2.0) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout)
