"""How the chat talks to JobService (pure Python, no Flet, Python 3.10).

JobService (``services/jobs.py``, mobile-app design §5.1) runs one job at a time
on the ``gl-job`` thread and publishes ``snapshot`` (``JobSnapshot | None``) and
``queue`` Signals. The chat needs only a few things from it, collected in
``JobsAdapter`` so the screens never depend on more than this:

* ``submit(kind, title, inputs, params)`` - queue/run a ``JobSpec``
  (``kind="direct_text"`` for chat sends; params from ``run_request.job_params``);
* ``request_stop(force=False)`` - the Send/Stop button (graceful, then force);
* ``snapshot()`` / ``subscribe(callback)`` - state, progress, in-flight count
  (``on_transition`` delivers every state change in order, including the terminal
  one; ``subscribe`` the coalesced view; ``on_question`` the blocking questions -
  registering it is what makes ``JobHost.ask`` wait for the chat's answer);
* ``answer(question_id, value)`` - the glossary approval card replying to the
  blocking ``JobHost.ask("…glossary_approval…", path=...)`` of the running job;
* ``request_stream(job_id)`` / ``request_segments(job_id)`` - the job's
  ``direct_text_stream.DirectTextStream`` and its live request cards (JobService feeds
  it every raw log line, including the token payloads the job log suppresses);
* ``add_log_listener(callback)`` - optional raw ``(message, source_thread)`` lines.

The snapshot is duck-typed (``id``, ``spec.kind/title/params``, ``state`` (enum or
name), ``progress``, ``in_flight``, ``last_line``, ``outputs``, ``error``,
``question``) so tests can use plain objects.

Also here: the JobCard phase for a chat turn, the progress line
"Chapter 12/48 · 3 in flight · ETA 14 min · 12:41" and the UI-only ETA moving
average (UI_SPEC §2.12.3).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

__all__ = [
    "ACTIVE_STATES",
    "CardPhase",
    "EtaEstimator",
    "JobsAdapter",
    "TERMINAL_STATES",
    "chat_id_of",
    "ended_card",
    "ended_kind",
    "format_duration",
    "job_kind",
    "progress_counts",
    "progress_line",
    "question_of",
    "running_label",
    "state_name",
]

log = logging.getLogger("glossarion.chat.jobs")

ACTIVE_STATES = frozenset({"QUEUED", "STARTING", "RUNNING", "STOPPING", "FORCE_STOPPING"})
RUNNING_STATES = frozenset({"STARTING", "RUNNING", "STOPPING", "FORCE_STOPPING"})
TERMINAL_STATES = frozenset({"DONE", "FAILED", "CANCELLED", "INTERRUPTED"})


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def state_name(snapshot: Any) -> str:
    """``JobState.RUNNING`` / ``"running"`` -> ``"RUNNING"`` ("" without a snapshot)."""
    state = _get(snapshot, "state")
    if state is None:
        return ""
    return str(getattr(state, "name", None) or getattr(state, "value", None) or state).upper()


def job_kind(snapshot: Any) -> str:
    kind = _get(_get(snapshot, "spec"), "kind")
    return str(getattr(kind, "value", None) or getattr(kind, "name", None) or kind or "").lower()


def chat_id_of(snapshot_or_spec: Any) -> Optional[str]:
    """The chat a job belongs to (``params["chat_id"]``), as route-safe text."""
    spec = _get(snapshot_or_spec, "spec", snapshot_or_spec)
    params = _get(spec, "params") or {}
    value = params.get("chat_id") if isinstance(params, Mapping) else None
    return None if value is None else str(value)


def question_of(snapshot: Any) -> Optional[dict]:
    """The pending blocking question of a job (``{id, kind, data}``) or None."""
    question = _get(snapshot, "question") or _get(snapshot, "pending_question")
    if not question:
        return None
    if isinstance(question, Mapping):
        data = dict(question.get("data") or {k: v for k, v in question.items() if k not in ("id", "kind")})
        return {"id": question.get("id"), "kind": str(question.get("kind") or ""), "data": data}
    return {
        "id": _get(question, "id"),
        "kind": str(_get(question, "kind", "") or ""),
        "data": dict(_get(question, "data") or {}),
    }


def progress_counts(snapshot: Any) -> dict:
    """``{total, completed, in_progress, failed, in_flight}`` from a snapshot (missing -> 0)."""
    progress = _get(snapshot, "progress")
    counts = {"total": 0, "completed": 0, "in_progress": 0, "failed": 0, "in_flight": 0, "fraction": None}
    if isinstance(progress, Mapping):
        for key in ("total", "completed", "in_progress", "failed"):
            counts[key] = int(progress.get(key) or 0)
        if not counts["completed"] and progress.get("cur") is not None:
            counts["completed"] = int(progress.get("cur") or 0)
        counts["fraction"] = progress.get("fraction")
    elif progress is not None:
        counts["total"] = int(_get(progress, "total", 0) or 0)
        counts["completed"] = int(_get(progress, "completed", None) or _get(progress, "cur", 0) or 0)
        counts["in_progress"] = int(_get(progress, "in_progress", 0) or 0)
        counts["failed"] = int(_get(progress, "failed", 0) or 0)
        counts["fraction"] = _get(progress, "fraction")
    counts["in_flight"] = int(_get(snapshot, "in_flight", 0) or 0)
    if counts["fraction"] is None and counts["total"]:
        counts["fraction"] = min(1.0, counts["completed"] / counts["total"])
    return counts


def format_duration(seconds: Optional[float]) -> str:
    """``12:41`` / ``1:02:03`` (elapsed) for the progress line."""
    if seconds is None or seconds < 0:
        return ""
    seconds = int(seconds)
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


class EtaEstimator:
    """UI-only moving average of chapter completion times (UI_SPEC §2.12.3)."""

    def __init__(self, window: int = 8, clock: Callable[[], float] = time.monotonic) -> None:
        self._samples: deque = deque(maxlen=max(2, window))
        self._clock = clock
        self._last_completed: Optional[int] = None

    def update(self, completed: int) -> None:
        if self._last_completed is None or completed < self._last_completed:
            self._samples.clear()
            self._samples.append((self._clock(), completed))
        elif completed > self._last_completed:
            self._samples.append((self._clock(), completed))
        self._last_completed = completed

    def eta_seconds(self, total: int) -> Optional[float]:
        if len(self._samples) < 2 or not total:
            return None
        (t0, c0), (t1, c1) = self._samples[0], self._samples[-1]
        if c1 <= c0 or t1 <= t0:
            return None
        rate = (t1 - t0) / (c1 - c0)
        remaining = max(0, total - c1)
        return remaining * rate

    @staticmethod
    def format(seconds: Optional[float]) -> str:
        if seconds is None:
            return ""
        minutes = max(1, int(round(seconds / 60.0)))
        if minutes < 60:
            return f"ETA {minutes} min"
        return f"ETA {minutes // 60} h {minutes % 60:02d} min"


def progress_line(snapshot: Any, *, eta: Optional[EtaEstimator] = None, now: Optional[float] = None) -> str:
    """"Chapter 12/48 · 3 in flight · ETA 14 min · 12:41" (parts appear when known)."""
    counts = progress_counts(snapshot)
    parts = []
    if counts["total"]:
        parts.append(f"Chapter {counts['completed']}/{counts['total']}")
        if eta is not None:
            eta.update(counts["completed"])
    if counts["in_flight"]:
        parts.append(f"{counts['in_flight']} in flight")
    if counts["failed"]:
        parts.append(f"{counts['failed']} failed")
    if eta is not None and counts["total"]:
        text = EtaEstimator.format(eta.eta_seconds(counts["total"]))
        if text:
            parts.append(text)
    started = _get(snapshot, "started")
    if isinstance(started, (int, float)) and started > 0:
        parts.append(format_duration((now if now is not None else time.time()) - started))
    return " · ".join(parts)


def running_label(snapshot: Any, *, waiting_for_glossary: bool = False) -> str:
    """JobCard Running header (UI_SPEC §2.12.3)."""
    state = state_name(snapshot)
    if state == "STOPPING":
        return "Stopping after current request…"
    if state == "FORCE_STOPPING":
        return "Force stopping…"
    if waiting_for_glossary:
        return "Waiting for your glossary decision"
    if state in ("QUEUED", "STARTING"):
        return "Starting translation…"
    last = str(_get(snapshot, "last_line", "") or "").lower()
    if "glossary" in last and ("extract" in last or "generat" in last):
        return "Generating glossary…"
    if "extract" in last and "chapter" in last:
        return "Extracting chapters…"
    if "header" in last or "toc" in last:
        return "Translating headers…"
    if "compil" in last or "epub" in last and "build" in last:
        return "Compiling EPUB…"
    return "Translating"


@dataclass(frozen=True)
class CardPhase:
    """JobCard lifecycle (UI_SPEC §2.12): plan · queued · running · stopping · result."""

    name: str  # plan | queued | running | stopping | force_stopping | done | stopped | failed | interrupted
    job_state: str = ""

    @property
    def live(self) -> bool:
        return self.name in ("queued", "running", "stopping", "force_stopping")

    @classmethod
    def for_turn(cls, job_state: str, *, plan_pending: bool = False, stop_requested: bool = False) -> "CardPhase":
        state = str(job_state or "").upper()
        if plan_pending:
            return cls("plan", state)
        mapping = {
            "QUEUED": "queued",
            "STARTING": "running",
            "RUNNING": "running",
            "STOPPING": "stopping",
            "FORCE_STOPPING": "force_stopping",
            "FAILED": "failed",
            "INTERRUPTED": "interrupted",
            "CANCELLED": "stopped",
        }
        if state == "DONE":
            return cls("stopped" if stop_requested else "done", state)
        return cls(mapping.get(state, "done"), state)


def ended_kind(snapshot: Any) -> str:
    """How a remembered JobService job ended, for its chat JobCard:
    ``done`` | ``stopped`` | ``failed`` | ``interrupted`` | ``discarded``."""
    state = state_name(snapshot)
    if state == "INTERRUPTED":
        return "discarded" if _get(snapshot, "resolution") == "discarded" else "interrupted"
    return {"CANCELLED": "stopped", "FAILED": "failed"}.get(state, "done")


def ended_card(kind: str, snapshot: Any = None) -> tuple:
    """(CardPhase name, status text) of an ended chat turn (UI_SPEC §2.12.4): "Stopped · 12/48
    chapters", "Failed …", "Interrupted …" (both offer Resume), "Finished with issues · n failed"
    or "Done · 48/48 chapters"."""
    counts = progress_counts(snapshot)
    chapters = f" · {counts['completed']}/{counts['total']} chapters" if counts["total"] else ""
    if kind == "stopped":
        return "stopped", f"Stopped{chapters}"
    if kind == "failed":
        return "failed", f"Failed{chapters}"
    if kind == "interrupted":
        return "interrupted", f"Interrupted{chapters}"
    if kind == "discarded":
        return "stopped", f"Discarded{chapters}"
    if counts["failed"]:
        return "done", f"Finished with issues · {counts['failed']} failed"
    return "done", f"Done{chapters}"


class JobsAdapter:
    """Duck-typed access to JobService (see the module docstring)."""

    def __init__(self, jobs: Any) -> None:
        self.jobs = jobs

    @property
    def available(self) -> bool:
        return self.jobs is not None

    def _signal(self, name: str) -> Any:
        return getattr(self.jobs, name, None) if self.jobs is not None else None

    def snapshot(self) -> Any:
        sig = self._signal("snapshot")
        if sig is None:
            return None
        if callable(sig) and not hasattr(sig, "value"):
            try:
                return sig()
            except Exception:
                return None
        return getattr(sig, "value", None)

    def snapshot_of(self, job_id: Any) -> Any:
        """The current snapshot of one job (live or finished); None when the service cannot tell."""
        fn = getattr(self.jobs, "snapshot", None) if self.jobs is not None else None
        if job_id is None or not callable(fn) or hasattr(fn, "value"):
            return None
        try:
            return fn(job_id)
        except TypeError:
            return None
        except Exception:
            return None

    def queue(self) -> list:
        sig = self._signal("queue")
        value = getattr(sig, "value", sig) if sig is not None else None
        return list(value or []) if isinstance(value, (list, tuple)) else []

    def pending(self) -> list:
        """The active job's and the queued jobs' snapshots (``JobService.view``)."""
        view = self._signal("view")
        if callable(view):
            try:
                jobs_view = view()
            except Exception:
                return []
            active = getattr(jobs_view, "active", None)
            return ([active] if active is not None else []) + list(getattr(jobs_view, "queue", ()) or ())
        active = self.snapshot()
        return ([active] if active is not None else []) + self.queue()

    def subscribe(self, callback: Callable[[Any], Any]) -> Callable[[], None]:
        """``callback(snapshot)`` for every state change, question and progress update."""
        unsubs = []
        jobs = self.jobs
        on_transition = getattr(jobs, "on_transition", None)
        on_question = getattr(jobs, "on_question", None)
        view_subscribe = getattr(jobs, "subscribe", None)
        if callable(on_transition):
            unsubs.append(on_transition(lambda snap, previous=None: callback(snap)))
        if callable(on_question):
            unsubs.append(on_question(lambda snap, question=None: callback(snap)))
        if callable(view_subscribe):
            unsubs.append(view_subscribe(lambda view: callback(getattr(view, "active", None))))
        else:
            for name in ("snapshot", "queue"):
                sig = self._signal(name)
                subscribe = getattr(sig, "subscribe", None)
                if callable(subscribe):
                    unsubs.append(subscribe(lambda _v: callback(self.snapshot())))

        def unsubscribe() -> None:
            for unsub in unsubs:
                try:
                    unsub()
                except Exception:
                    pass

        return unsubscribe

    def build_spec(self, kind: str, title: str, inputs: tuple, params: Mapping[str, Any],
                   origin: Optional[Mapping[str, Any]] = None) -> Any:
        """A ``services.jobs.JobSpec`` when available, else a plain mapping."""
        try:
            from glossarion_mobile.services import jobs as jobs_module  # type: ignore[attr-defined]
        except Exception:
            jobs_module = None
        spec_cls = getattr(jobs_module, "JobSpec", None) if jobs_module is not None else None
        kind_enum = getattr(jobs_module, "JobKind", None) if jobs_module is not None else None
        kind_value: Any = kind
        if kind_enum is not None:
            for candidate in (kind.upper(), kind):
                try:
                    kind_value = kind_enum[candidate] if candidate.isupper() else kind_enum(candidate)
                    break
                except Exception:
                    continue
        if spec_cls is not None:
            try:
                return spec_cls(kind=kind_value, title=title, inputs=tuple(inputs), params=dict(params),
                                origin=dict(origin or {}))
            except TypeError:
                log.debug("JobSpec signature differs; submitting a mapping", exc_info=True)
        return {"kind": kind, "title": title, "inputs": tuple(inputs), "params": dict(params), "origin": dict(origin or {})}

    async def submit(self, kind: str, title: str, inputs: tuple, params: Mapping[str, Any],
                     origin: Optional[Mapping[str, Any]] = None) -> Any:
        if self.jobs is None:
            raise RuntimeError("The job service is not running")
        spec = self.build_spec(kind, title, inputs, params, origin)
        result = self.jobs.submit(spec)
        if asyncio.iscoroutine(result):
            result = await result
        return result

    def request_stop(self, force: bool = False) -> Any:
        if self.jobs is None:
            return None
        try:
            return self.jobs.request_stop(force=force)
        except TypeError:
            return self.jobs.request_stop(force)

    def cancel_queued(self, job_id: Any) -> Any:
        fn = getattr(self.jobs, "cancel_queued", None)
        return fn(job_id) if callable(fn) else None

    def resume(self, job_id: Any) -> Any:
        """``JobService.resume``: resubmit a remembered job's spec and resolve its Interrupted
        entry; the new job id, or None (unknown / already resumed / no such service call)."""
        fn = getattr(self.jobs, "resume", None)
        if job_id is None or not callable(fn):
            return None
        try:
            return fn(job_id)
        except Exception:
            log.exception("resuming job %s failed", job_id)
            return None

    def answer(self, question_id: Any, value: Any) -> bool:
        for name in ("answer", "answer_question", "respond"):
            fn = getattr(self.jobs, name, None)
            if callable(fn):
                result = fn(question_id, value)
                return True if result is None else bool(result)
        return False

    def add_log_listener(self, callback: Callable[..., Any]) -> Optional[Callable[[], None]]:
        for name in ("add_log_listener", "subscribe_log_lines", "add_line_listener"):
            fn = getattr(self.jobs, name, None)
            if callable(fn):
                result = fn(callback)
                if callable(result):
                    return result
                remove = getattr(self.jobs, "remove_log_listener", None)
                return (lambda: remove(callback)) if callable(remove) else (lambda: None)
        return None

    def log_buffer(self, job_id: Any = None) -> Any:
        fn = getattr(self.jobs, "log_buffer", None)
        if callable(fn) and job_id is not None:
            return fn(job_id)
        return getattr(self.jobs, "log", None)

    def request_stream(self, job_id: Any) -> Any:
        """The job's ``direct_text_stream.DirectTextStream`` (None before the job starts)."""
        fn = getattr(self.jobs, "request_stream", None)
        if job_id is None or not callable(fn):
            return None
        try:
            return fn(job_id)
        except Exception:
            return None

    def request_segments(self, job_id: Any) -> Optional[list]:
        """The service's live request segments for ``job_id``; None when it has no request stream."""
        has = getattr(self.jobs, "has_request_stream", None)
        fn = getattr(self.jobs, "request_segments", None)
        if job_id is None or not callable(fn):
            return None
        if callable(has):
            try:
                if not has(job_id):
                    return None
            except Exception:
                return None
        try:
            return list(fn(job_id) or [])
        except Exception:
            return None

    def graceful_stop_configured(self) -> bool:
        value = getattr(self.jobs, "graceful_stop", None)
        if callable(value):
            try:
                return bool(value())
            except Exception:
                return True
        return True if value is None else bool(value)
