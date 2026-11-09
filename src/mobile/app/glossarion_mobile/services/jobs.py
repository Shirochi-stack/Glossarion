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
* ``add_log_listener(job_id, callback)`` is the desktop ``add_log_listener``: the
  callback gets every message of that job whole (blank ones and the blank lines inside
  streamed text included) before any log-buffer filtering, on the thread that logged
  it, after a replay of the messages the job logged before (the Reader's live panel).
* Job safety (hotfix 1 issue 5): the same work is never queued twice. ``submit`` reuses an
  ACTIVE job (queued, starting, running, waiting at the glossary gate, refining, stopping)
  that already does the submission's work - the same request (``spec_key``: kind, inputs,
  params), or the same job family over a target it covers (``job_target``: a book's
  translation run, glossary phase included, a book's glossary extraction, a workspace's QA
  scan, one chapter's translation) - and returns that job's id; the check and the insert
  run under the service lock. ``submit_job`` says whether it reused (``SubmitResult``),
  ``active_job_for`` finds that job before anything is recorded. ``resume`` of an active
  job (or of one already resumed into a job that still runs) returns that job. A gate's
  answers are idempotent (the first one counts) and refused while ``refine_at_gate``
  refines its glossary: the job waits until the refinement ended, so the translation never
  starts before it.
* Multi-instance (design §6, §6b): a job submitted with ``instance=N`` (``JobSpec.params["instance"]``, kept for
  Resume) runs on instance N's keys only, in every key pool (``instance_plan.config_for_instance``; the main
  API key is instance 1's), and its API client is kept to them (``instance_worker.restrict_keys``). Instance 1 is the
  queue and ``gl-job`` worker above, unchanged; each instance N >= 2 has its own queue and ``gl-job-iN``
  worker, so one job per instance runs at a time and the instances run side by side. An instance N job runs
  inside its isolated in-process runtime (``services.instance_runtime``: its own environment, stdout, argv,
  module copies, stop flags, key pools and watchdog), set up when it starts - a job whose runtime cannot be
  set up fails, it never runs shared. The Stop protocol, the ``ProgressWatcher`` and the request stream of
  that job use the instance's own modules. The old no-argument forms (``snapshot()``, ``busy``,
  ``request_stop()``, ``stream_tail()``) still mean instance 1's job; ``running_snapshots``,
  ``request_stop_all`` and ``JobsView.running`` cover every instance. ``set_job_running`` stays true while
  any job runs, and ``active.state`` keeps every instance's running and queued jobs.

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

import collections
import contextlib
import copy
import enum
import importlib
import json
import logging
import os
import re
import sys
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterable, Iterator, Mapping, Optional, Sequence

from glossarion_mobile.job_kinds import INDEXING, result_fields
from glossarion_mobile.services.dispatcher import LogBuffer

__all__ = [
    "ACTIVE_STATE_FILE",
    "AUTO_ACCEPT_GLOSSARY_PARAM",
    "FAMILY_CHAPTER",
    "FAMILY_GLOSSARY",
    "FAMILY_QA",
    "FAMILY_TRANSLATION",
    "GLOSSARY_ACCEPTED_LINE",
    "GLOSSARY_AUTO_ACCEPTED_LINE",
    "GLOSSARY_QUESTION_KINDS",
    "HISTORY_LIMIT",
    "HISTORY_STATE_FILE",
    "INSTANCE_PARAM",
    "INTERRUPTED_STATE_FILE",
    "JOB_FAMILIES",
    "JOB_TARGET_PARAM",
    "JobAlreadyActive",
    "JobBackend",
    "JobContext",
    "JobError",
    "JobKind",
    "JobService",
    "JobSnapshot",
    "JobSpec",
    "JobState",
    "JobTarget",
    "JobsView",
    "Progress",
    "RESUMED_EVENT",
    "pending_jobs",
    "STATE_LABELS",
    "SubmitResult",
    "TERMINAL_STATES",
    "ACTIVE_STATES",
    "ISSUE_LABELS",
    "already_active_text",
    "classify_issue",
    "client_log_handlers",
    "coerce_instance",
    "format_duration",
    "instance_label",
    "is_glossary_question",
    "issue_label",
    "job_instance",
    "job_target",
    "notification_lines",
    "notification_summary",
    "notification_text",
    "position_line",
    "short_title",
    "progress_line",
    "spec_key",
    "status_key",
    "strip_model_for",
    "target_key",
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
JOB_MESSAGES_KEPT = 4000  # whole log messages a job keeps for listeners that attach late
LAST_LINE_CHARS = 240
#: what a job keeps of its current request's streamed text (``JobService.stream_tail``, hotfix 3)
STREAM_TAIL_KEEP = 2000
#: log lines that start a new request: the streamed text of the last one is dropped
STREAM_TAIL_RESETS = ("SDK stream start", "Sending API call")
#: hotfix 4: lowercased phrases that open / close a request's stream. The OpenAI-compatible routes (on-device
#: ``local/`` among them) log no "Text streaming" banner, so the request stream never marked their text as
#: streamed and the notification showed none; messages between these that are not status lines are the text
STREAM_OPEN_PHRASES = ("stream start (model=", "stream opened in ")
STREAM_CLOSE_PHRASES = ("received response", "stream finished", "stream complete", "sdk call finished",
                        "received translation", "sending api call")
#: lowercased phrases that end a request's thinking: what streams next is the answer (``_note_stream_phase``)
STREAM_TEXT_PHRASES = ("thinking complete", "text streaming", "first text token")
#: how a status line starts besides an emoji / symbol (``stream_status_line``), never streamed text
STREAM_STATUS_PREFIXES = ("[DEBUG]", "[INFO]", "[WARN", "[ERROR")


def stream_status_line(raw: str, stream: Any = None) -> bool:
    """Whether a log message is a status line, not streamed text: it starts with an emoji or other symbol
    (Unicode "So": the client's and the routes' status lines, "⏳ AuthND: …", "📤 …", "✓ …"; not "<" of
    the HTML text), a log tag, or the request stream calls it a pipeline status. Blank: not status."""
    import unicodedata

    stripped = str(raw or "").strip()
    if not stripped:
        return False
    if stripped.startswith(STREAM_STATUS_PREFIXES) or unicodedata.category(stripped[0]) == "So":
        return True
    status = getattr(stream, "_looks_like_pipeline_status", None)
    try:
        return bool(callable(status) and status(stripped))
    except Exception:
        return False
STREAM_DRAIN_INTERVAL = 0.1  # seconds between the job-side budgeted drains of the request stream
#: Lines written for the request stream's parser, not for people: neither Logs nor the log file keeps them
STREAM_MACHINE_MARKERS = ("[DIRECT_TEXT_RESPONSE_PAYLOAD]", "[DIRECT_TEXT_GLOSSARY_STREAM_START]")
ANSWERS_KEPT = 64  # answered questions remembered, so a repeated tap of the same answer stays a no-op

#: Lines the auth modules print when a job has no usable login and falls back to the interactive
#: browser login (``get_valid_access_token`` / ``recover_from_unauthorized``), refuses it (the
#: headless refusal) or, on mobile, sends the user to the Accounts screen (AuthGrok / the AuthCD CLI
#: fallback): marker -> provider. The first one in a job sends a ``sign_in_required`` event with the
#: provider and the slot the job's model uses (UI_SPEC §1.9 "Sign-in required for Claude #2").
SIGN_IN_PROVIDER_MARKERS = (
    ("AuthGPT: No valid token found", "authgpt"),
    ("AuthGPT: Session expired", "authgpt"),
    ("AuthGPT: Browser-based OAuth login is not available", "authgpt"),
    ("AuthCD: No valid token found", "authcd"),
    ("AuthCD: Browser-based OAuth login is not available", "authcd"),
    ("AuthCD: the Claude Code CLI is not available on Glossarion Mobile", "authcd"),
    ("AuthGem: No valid token found", "authgem"),
    ("AuthGem: Browser-based OAuth login is not available", "authgem"),
    ("AuthGrok has no valid token", "authgrok"),
    ("AuthGrok browser login is unavailable", "authgrok"),
    ("AuthGrok: sign in to Grok from Glossarion's Accounts screen", "authgrok"),
)
SIGN_IN_MARKERS = tuple(marker for marker, _provider in SIGN_IN_PROVIDER_MARKERS)


#: Owner #30: a job that never needs the model (``model_catalog.MODEL_FREE_JOB_KINDS``: Compile EPUB / PDF,
#: the file tools, the QA scan, ...) must never start a sign-in. The EPUB converter still translates headers /
#: the title / metadata through the configured model when those options are on, and with no saved login the
#: ChatGPT / Claude / Gemini / Grok auth modules fell back to their interactive browser login (on the phone the
#: in-app browser opened over the app). The shared modules refuse that login at once in a headless
#: environment (``get_valid_access_token``'s ``is_headless`` check), so such a job runs with this variable
#: set (``no_sign_in_scope``); its requests fail like any other unusable
#: route and no "Sign-in required" prompt is sent for it.
NO_SIGN_IN_ENV = ("HF_SPACES", "true")


def sign_in_allowed(kind: Any) -> bool:
    """False for a job kind that never needs the model (see ``NO_SIGN_IN_ENV``)."""
    from glossarion_mobile.services.model_catalog import MODEL_FREE_JOB_KINDS

    return str(getattr(kind, "value", kind) or "") not in MODEL_FREE_JOB_KINDS


@contextlib.contextmanager
def no_sign_in_scope(kind: Any) -> Iterator[None]:
    """``NO_SIGN_IN_ENV`` set while a job of ``kind`` runs, when its kind never needs the model."""
    if sign_in_allowed(kind):
        yield
        return
    with _env_scope(*NO_SIGN_IN_ENV):
        yield


@contextlib.contextmanager
def local_context_scope() -> Iterator[None]:
    """``CONTEXT_ENV`` (every on-device model's context window, ``setting_writes.local_context_env``)
    set while a job runs: the shared glossary code sizes its chunks to fit a ``local/`` model's context
    (``unified_api_client.model_context_window``). Nothing is set without the app's engine service."""
    from glossarion_mobile.state.setting_writes import local_context_env

    value = local_context_env()
    if not value:
        yield
        return
    from glossarion_mobile.services.local_llm.service import CONTEXT_ENV

    with _env_scope(CONTEXT_ENV, value):
        yield


def _without_key_env(env: Mapping[str, str]) -> dict:
    """``env`` (instance 1's environment) without its API keys, key-pool lists and switches and its key scope
    (``instance_plan.KEY_ENV``, ``GLOSSARION_KEY_INSTANCE``): an instance >= 2 exports its own from its config
    snapshot, never instance 1's leftovers (the desktop instance process: ``instance_worker.child_environment``)."""
    import instance_plan

    dropped = {name.upper() for name in instance_plan.KEY_ENV + (instance_plan.ENV_KEY_INSTANCE,)}
    return {key: value for key, value in env.items() if str(key).upper() not in dropped}


def _http_logger_levels() -> dict:
    """The levels of the HTTP client loggers a graceful stop silences (``stop_control.HTTP_LOGGER_NAMES``): they
    are the process's, every instance's job logs its requests through them."""
    try:
        names = importlib.import_module("stop_control").HTTP_LOGGER_NAMES
    except Exception:
        return {}
    return {name: logging.getLogger(name).level for name in names}


def _restore_http_logger_levels(levels: Mapping[str, int]) -> None:
    for name, level in levels.items():
        logging.getLogger(name).setLevel(level)


def _release_base_stop_flags() -> None:
    """Instance 1's stopped job has ended while another instance's job still runs: instance 1's immediate-stop
    flags (the base client's ``global_stop_flag`` / ``UnifiedClient._global_cancelled``) and ``GRACEFUL_STOP``
    must not stay set, since the app's on-device endpoint reads them for every instance's request
    (``local_llm.service.process_stop_state``) and would refuse the other instance's ``local/`` requests until
    instance 1 runs again (its next run resets them anyway). On the base thread: the base modules and env."""
    client = sys.modules.get("unified_api_client")
    if client is not None:
        try:
            client.global_stop_flag = False
            client.UnifiedClient._global_cancelled = False
        except Exception:
            log.debug("the base client's stop flags could not be cleared", exc_info=True)
    if os.environ.get("GRACEFUL_STOP") == "1":
        os.environ["GRACEFUL_STOP"] = "0"


@contextlib.contextmanager
def _env_scope(name: str, value: str) -> Iterator[None]:
    previous = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


def sign_in_provider(line: str) -> Optional[str]:
    """The provider whose login a job log line says is missing (``SIGN_IN_PROVIDER_MARKERS``)."""
    for marker, provider in SIGN_IN_PROVIDER_MARKERS:
        if marker in line:
            return provider
    return None


def sign_in_event(job_model: Optional[str], provider: str, line: str = "") -> dict:
    """``sign_in_required`` data: the provider and the slot the job's model targets
    (``authcd2/`` -> 2; a pool route or another provider's model -> slot #0)."""
    account_id = 0
    try:
        from glossarion_mobile.services.oauth import sign_in_slot

        account_id = int(sign_in_slot(job_model, provider) or 0)
    except Exception:
        account_id = 0
    return {"provider": provider, "account_id": account_id, "line": str(line or "")[:200]}


# ---------------------------------------------------------------------------
# Glossary questions (the shared gate's ``host.ask`` kinds)
# ---------------------------------------------------------------------------

#: Blocking job questions the glossary approval card answers: the chat run's Direct Text gate
#: (``translation_pipeline._await_direct_text_glossary_approval`` -> ``_ui_request`` -> ``host.ask``) and
#: the Library "Review glossary before translating" gate (``job_kinds.translate.GLOSSARY_REVIEW_QUESTION``).
#: The one home of the kinds (contract C1): the chat run controller, the chat cards, the Jobs feature and
#: the notifications import them from here.
GLOSSARY_QUESTION_KINDS = ("glossary_approval", "direct_text_glossary_approval")

#: ``JobSpec.params`` flag of a chat send (``run_request.job_params``, captured at Send from the chat's
#: "Always accept generated glossaries" setting): ``_job_ask`` answers the glossary gate Yes itself.
AUTO_ACCEPT_GLOSSARY_PARAM = "auto_accept_glossary"
#: The desktop's own log line for an accepted generated glossary (``translator_gui._resolve_glossary_approval``;
#: ``tests_host/test_glossary_auto_accept.py`` checks the desktop source still has it).
GLOSSARY_ACCEPTED_LINE = "✅ Direct Text: generated glossary accepted"
GLOSSARY_AUTO_ACCEPTED_LINE = GLOSSARY_ACCEPTED_LINE + " (Always accept is on)"


#: Issue 12: the progress line while the gate's glossary is refined (the review alerts are hidden meanwhile).
GLOSSARY_REFINING_TEXT = "Refining the glossary…"
#: Job event while the gate's glossary is refined: the "Glossary ready" notification is withdrawn (the
#: question is delivered again when the refinement ends and the question still waits).
QUESTION_SUSPENDED_EVENT = "question_suspended"
#: Job event of a stopped / failed / interrupted job that Resume carried on in another job (``{"job": its id}``):
#: its "Stopped: … Tap to resume" / "Paused" notifications have nothing left to resume.
RESUMED_EVENT = "resumed"


def is_glossary_question(kind: Any) -> bool:
    """The pipeline's Direct Text glossary gate (``_ui_request`` kind), whatever its exact name."""
    value = str(kind or "").lower()
    return value in GLOSSARY_QUESTION_KINDS or ("glossary" in value and "approv" in value)


def _question_kind(question: Any) -> str:
    return str(question.get("kind") or "") if isinstance(question, Mapping) else ""


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
    GLOSSARY_NAMES = "glossary_names"
    VALIDATE_EPUB = "validate_epub"
    RENAME_OUTPUTS = "rename_outputs"
    TRANSLATE_HEADERS = "translate_headers"
    METADATA = "metadata"
    # U7
    RETRANSLATE = "retranslate"
    RESOLVE_QA = "resolve_qa"
    ASYNC_BATCH = "async_batch"
    REVIEW = "review"
    RPGMAKER = "rpgmaker"
    GENERATE_MEDIA = "generate_media"
    TRANSLATE_IMAGE = "translate_image"
    # U8
    MANGA = "manga"
    MANGA_STEP = "manga_step"
    # U9
    MD_TXT_SIDECARS = "md_txt_sidecars"
    BR_TO_PARAGRAPHS = "br_to_paragraphs"


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

#: ``JobSnapshot.resolution`` of an ended entry a newer job of its book replaced
RESOLUTION_CARRIED_ON = "carried_on"

STOP_GRACEFUL = "graceful"
STOP_IMMEDIATE = "immediate"
STOP_FORCE = "force"


class JobError(RuntimeError):
    """A job could not run (bad spec, missing input, shared module missing). Message is user-facing."""


class JobAlreadyActive(JobError):
    """A submission whose work an active job already does (hotfix 1 issue 5): nothing new was queued. The
    message is user-facing (``already_active_text``); ``job_id`` / ``snapshot`` name the job the UI binds
    to or opens instead (Jobs › job)."""

    def __init__(self, snapshot: Any = None, message: str = "") -> None:
        self.snapshot = snapshot
        self.job_id = getattr(snapshot, "id", None)
        super().__init__(message or already_active_text(snapshot))


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
# Multi-instance (design §6): which instance a job runs in
# ---------------------------------------------------------------------------

#: ``JobSpec.params`` key of a job of a multi-instance run: its instance (``submit_job(spec, instance=N)``; kept in
#: the spec, so a Resume runs on the same instance). Absent: instance 1 exactly as before, on the whole Translation
#: key pool. Present (1 included): the job's config snapshot keeps only that instance's Translation keys.
INSTANCE_PARAM = "instance"
#: the instance runtime module, once something imported it (``_installed_runtime``)
_RUNTIME_MODULE = "glossarion_mobile.services.instance_runtime"


def coerce_instance(value: Any) -> int:
    """An instance number from a stored value (``instance_plan.coerce_instance``: integers >= 1, else 1); 1 in a
    build without the shared backend, where no other instance can run."""
    try:
        import instance_plan
    except ImportError:
        return 1
    return instance_plan.coerce_instance(value)


def job_instance(spec: Any) -> int:
    """The instance a job (a ``JobSpec``, a snapshot's spec or its dict) runs in: ``params["instance"]``, else 1."""
    params = _spec_field(spec, "params") if spec is not None else None
    if isinstance(params, Mapping) and params.get(INSTANCE_PARAM) is not None:
        return coerce_instance(params.get(INSTANCE_PARAM))
    return 1


def instance_label(snap: Any) -> str:
    """"Instance 2" (the Jobs page row, the notification's rows of a multi-instance run)."""
    return f"Instance {job_instance(getattr(snap, 'spec', None))}"


def _installed_runtime() -> Any:
    """``services.instance_runtime`` once an instance >= 2 job installed it, else None (never imports it, so a
    session without one runs exactly as before)."""
    module = sys.modules.get(_RUNTIME_MODULE)
    return module if module is not None and module.installed() else None


def _loaded(name: str) -> Any:
    """``sys.modules.get(name)`` as the calling thread's instance sees it (its own copy inside an instance)."""
    runtime = _installed_runtime()
    return runtime.loaded_module(name) if runtime is not None else sys.modules.get(name)


def _instance_key() -> int:
    """The calling thread's instance (1 outside every instance runtime)."""
    runtime = _installed_runtime()
    return (runtime.current() or 1) if runtime is not None else 1


# ---------------------------------------------------------------------------
# Job safety (hotfix 1 issue 5): what a job works on, for the duplicate check
# ---------------------------------------------------------------------------

#: ``JobSpec.params`` key: the path(s) whose work the job does, for the duplicate check (``job_target``).
#: A submitter sets it when its inputs are not the book itself: a chat attachment send runs the pipeline on
#: an input of its own run root, so it names the attached file here (``ChatRuns.send``).
JOB_TARGET_PARAM = "job_target"

FAMILY_TRANSLATION = "translation"
FAMILY_GLOSSARY = "glossary"
FAMILY_QA = "qa"
FAMILY_CHAPTER = "chapter"

#: ``params`` key of a translation run a chat started for a Library book (its origin is the Library): the chat whose
#: model a later Resume on the Jobs page follows (``ChatRuns._resume_params``)
MODEL_CHAT_PARAM = "model_chat"
#: kind -> job family. An active job of the same family whose target covers a submission's target already
#: does its work (``JobService.submit`` reuses it): the translation family is a book's translation run, its
#: glossary phase included (Library / Progress Manager / QA card "Run translation", the chat's "Save to:
#: Library" plan, a chat attachment send - which names its target with ``JOB_TARGET_PARAM``); the glossary
#: family a book's standalone glossary extraction; the QA family a workspace's QA scan; the chapter family
#: one chapter's "Translate this chapter" (Reader, Book page rows). Other kinds are only ever reused for the
#: very same request (``spec_key``).
JOB_FAMILIES = {
    "translate": FAMILY_TRANSLATION,
    "direct_text": FAMILY_TRANSLATION,
    "extract_glossary": FAMILY_GLOSSARY,
    "glossary_refine": FAMILY_GLOSSARY,
    "qa_scan": FAMILY_QA,
    "single_chapter": FAMILY_CHAPTER,
}


@dataclass(frozen=True)
class JobTarget:
    """A job's family and the normalised paths it works on (``target_key``; a chapter job's keys are
    ``<book>::<chapter file>``)."""

    family: str
    keys: frozenset

    def covers(self, other: Optional["JobTarget"]) -> bool:
        """All of ``other``'s work is part of this job's (same family, its paths a subset of ours)."""
        return (other is not None and self.family == other.family and bool(other.keys)
                and other.keys <= self.keys)

    def names(self, key: str) -> bool:
        return any(k == key or k.startswith(key + "::") for k in self.keys)


def target_key(path: Any) -> str:
    """A path as the duplicate check compares it: absolute, symbolic links resolved and case-folded where
    the file system is case-insensitive (``os.path.normcase``)."""
    try:
        return os.path.normcase(os.path.realpath(os.path.abspath(os.fspath(path))))
    except Exception:
        return _norm(path)


def _spec_kind(spec: Any) -> str:
    kind = spec.get("kind") if isinstance(spec, Mapping) else getattr(spec, "kind", None)
    return str(getattr(kind, "value", kind) or "")


def _spec_field(spec: Any, name: str, default: Any = None) -> Any:
    return spec.get(name, default) if isinstance(spec, Mapping) else getattr(spec, name, default)


def job_target(spec: Any) -> Optional[JobTarget]:
    """The family and paths of a submission (``JOB_FAMILIES``), or None when its kind has no family or it
    names nothing: ``params[JOB_TARGET_PARAM]`` when set, else the inputs (a typed-text chat send, which
    names no target, has none). Pure: no disk access beyond resolving the paths."""
    kind = _spec_kind(spec)
    family = JOB_FAMILIES.get(kind)
    if family is None:
        return None
    params = _spec_field(spec, "params") or {}
    params = params if isinstance(params, Mapping) else {}
    explicit = params.get(JOB_TARGET_PARAM)
    if explicit:
        paths = [explicit] if isinstance(explicit, (str, os.PathLike)) else list(explicit)
    elif kind == "direct_text":
        return None  # its input is a file of its own run root: only the target the chat names counts
    else:
        paths = list(_spec_field(spec, "inputs") or ())
    keys = {target_key(p) for p in paths if p and str(p).strip()}
    if kind == "single_chapter":
        chapter = str(params.get("chapter_file") or "").strip()
        if not chapter:
            return None
        keys = {f"{key}::{chapter}" for key in keys}
    return JobTarget(family, frozenset(keys)) if keys else None


#: Kinds whose job works on books (``job_books``): one of them queued or running replaces its books' ended entries.
BOOK_JOB_KINDS = frozenset(JOB_FAMILIES) | {
    "retranslate", "resolve_qa", "compile_epub", "compile_pdf", "metadata", "translate_headers", "validate_epub",
    "rename_outputs", "md_txt_sidecars", "br_to_paragraphs",
}


def job_books(spec: Any, output_dirs: Optional[Mapping[str, str]] = None, output_dir: Any = None) -> tuple:
    """The books a job works on, each as the set of keys that name that book: its source file (``job_target``, a QA
    scan's ``targets``), its workspace folder (the params, the output folders once known) and, for a job of one
    book, its Library id. Never the title: two books can share one, and one book has several (owner 2026-11-01:
    Library, QA card and Reader jobs of one book). () for a job on no book. Pure: no disk access beyond resolving
    the paths."""
    kind = _spec_kind(spec)
    if kind not in BOOK_JOB_KINDS:
        return ()
    params = _spec_field(spec, "params") or {}
    params = params if isinstance(params, Mapping) else {}
    targets = params.get("targets")
    books: list = []
    if targets and isinstance(targets, (list, tuple)):
        books = [{target_key(t[name]) for name in ("folder", "source") if t.get(name)}
                 for t in targets if isinstance(t, Mapping)]
    else:
        target = job_target(spec)
        if target is not None:
            paths = {key.split("::", 1)[0] for key in target.keys}
        elif kind == "direct_text":
            return ()  # a typed-text chat: its input is a file of its own run root
        else:
            paths = {target_key(p) for p in (_spec_field(spec, "inputs") or ()) if p and str(p).strip()}
        books = [{path} for path in sorted(paths)]
    dirs = dict(output_dirs or {})
    if len(books) > 1:  # each output folder to its own book; the params and the origin do not say which
        for source, folder in dirs.items():
            key = target_key(source)
            for book in books:
                if key in book:
                    book.add(target_key(folder))
        return tuple(frozenset(book) for book in books)
    keys = set(books[0]) if books else set()
    keys.update(target_key(params[name]) for name in ("folder", "output_dir", "source_path") if params.get(name))
    keys.update(target_key(folder) for folder in (*dirs.values(), output_dir) if folder)
    origin = _spec_field(spec, "origin") or {}
    if isinstance(origin, Mapping) and origin.get("bid"):
        keys.add(f"bid:{origin['bid']}")
    return (frozenset(keys),) if keys else ()


def same_books(ended: Sequence[frozenset], live: Sequence[frozenset]) -> bool:
    """Every book of ``ended`` is one of ``live``'s (a key in common): a job over two books is not replaced by a
    job over one of them."""
    return bool(ended) and all(any(book & other for other in live) for book in ended)


def spec_key(spec: Any) -> str:
    """What makes two submissions the very same request, for any kind: the kind, the inputs and the params
    (not the title or the origin, which only say where it was tapped). '' when it cannot be told."""
    try:
        params = _spec_field(spec, "params") or {}
        return json.dumps({"kind": _spec_kind(spec),
                           "inputs": [target_key(p) for p in (_spec_field(spec, "inputs") or ())],
                           "params": _jsonable(dict(params) if isinstance(params, Mapping) else {})},
                          sort_keys=True, ensure_ascii=False, default=repr)
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


#: a job title with one of these endings is still its source file's name (``_take_translated_title``)
FILE_TITLE_EXTENSIONS = (".epub", ".txt", ".pdf", ".md", ".html", ".xhtml", ".htm", ".docx", ".odt", ".rtf")

#: job kinds that work on one chapter of a book, and what its file is called (``book_job_title``)
CHAPTER_JOB_KINDS = frozenset({"single_chapter", "resolve_qa"})
CHAPTER_FILE_EXTENSIONS = (".xhtml", ".html", ".htm", ".txt")  # .txt: a TXT book's response_section_N.txt


def book_job_title(kind: str, title: str, inputs: Sequence[str]) -> str:
    """Owner 2026-11-01 ("using the title of the html file instead of the book title"): a chapter's job is labelled
    with its book. "Book · chapter0004.xhtml" (the Reader, the Book page's rows, Resolve QA, entries an older build
    stored) is "Book"; with no book name, the source file's name."""
    if kind not in CHAPTER_JOB_KINDS:
        return title
    head, _separator, name = str(title or "").rpartition("·")
    if os.path.splitext(name.strip())[1].lower() not in CHAPTER_FILE_EXTENSIONS:
        return title
    return head.strip() or (os.path.splitext(os.path.basename(str(inputs[0])))[0] if inputs else "") or title


def translated_book_title(path: str) -> str:
    """The translated ``title`` of a book's ``metadata.json`` (``title_translated`` set), else ''."""
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return ""
    if not isinstance(data, dict) or not data.get("title_translated"):
        return ""
    return " ".join(str(data.get("title") or "").split())


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
        object.__setattr__(self, "title", book_job_title(kind, self.title, self.inputs))
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
    resolution: Optional[str] = None  # interrupted jobs: "resumed" | "discarded"; RESOLUTION_CARRIED_ON
    restore_pending: bool = False  # recovered; its progress rows are not restored yet
    result: Mapping[str, Any] = field(default_factory=dict)  # what the kind adapter recorded (JobContext.set_result)
    # U9: the request wait the API client last logged (rate limit / key cooldown / network):
    # {"kind", "seconds", "since", "until", "line"}; transient (never persisted), see ``active_issue``.
    issue: Optional[Mapping[str, Any]] = None
    # Issue 12: the pending glossary question is being refined (``JobService.refine_at_gate``): its "action
    # required" alerts (notification, JobStrip warning, Book page review) stay hidden until it ends
    refining: bool = False
    # hotfix 8: where the job is, "[5] Ch. 3 (1/4)" (``request_position``); transient (never persisted)
    position: str = ""

    def active_issue(self, now: Optional[float] = None) -> Optional[dict]:
        """The issue chip state while the job runs and the wait has not expired, else None."""
        issue = self.issue
        if not issue or not self.is_active:
            return None
        current = now if now is not None else time.time()
        try:
            until = float(issue.get("until") or 0.0)
        except (TypeError, ValueError):
            until = 0.0
        return dict(issue) if until > current else None

    @property
    def kind(self) -> str:
        return self.spec.kind

    @property
    def title(self) -> str:
        return self.spec.title

    @property
    def instance(self) -> int:
        """The instance the job runs in (``JobSpec.params["instance"]``; 1 for a job of no multi-instance run)."""
        return job_instance(self.spec)

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
    """Everything the Jobs page shows (UI_SPEC §1.8), built fresh at delivery time.

    ``active`` is instance 1's running job, else the first running job of another instance (what the job strip
    and the old single-job callers show); ``running`` lists every instance's running job, instance 1 first, then
    by instance number; ``queue`` holds every instance's queued jobs (instance 1's in order first; a snapshot's
    ``instance`` says whose)."""

    active: Optional[JobSnapshot] = None
    queue: tuple = ()
    interrupted: tuple = ()
    history: tuple = ()
    paused: bool = False
    # ended entries a newer job of their book replaced (``JobService._drop_superseded``): not listed; what refers to
    # them by id (a chat's Result and QA cards, an open-job link, a notification) still reads them
    carried_on: tuple = ()
    running: tuple = ()

    @property
    def running_count(self) -> int:
        return len(self.running) if self.running else (1 if self.active is not None else 0)

    @property
    def queued_count(self) -> int:
        return len(self.queue)

    def instance_job(self, number: int) -> Optional[JobSnapshot]:
        """The running job of instance ``number`` (None when that instance is idle)."""
        wanted = coerce_instance(number)
        return next((snap for snap in (self.running or ((self.active,) if self.active is not None else ()))
                     if snap.instance == wanted), None)

    def find(self, job_id: str) -> Optional[JobSnapshot]:
        for snap in ((self.active,) if self.active is not None else ()) + tuple(self.running) + tuple(self.queue) + \
                tuple(self.interrupted) + tuple(self.history) + tuple(self.carried_on):
            if snap is not None and snap.id == job_id:
                return snap
        return None


def pending_jobs(view: Any) -> list:
    """Every job of ``view`` that runs or waits: each instance's running job (``running``; a view without it: its
    ``active`` job), then the queue. What a "does a job write this folder?" guard checks: with several instances
    running, ``active`` is only one of them."""
    running = [snap for snap in (getattr(view, "running", ()) or ()) if snap is not None]
    if not running:
        active = getattr(view, "active", None)
        running = [active] if active is not None else []
    return running + list(getattr(view, "queue", ()) or ())


@dataclass(frozen=True)
class SubmitResult:
    """``JobService.submit_job`` / ``resume_job``: the job that does the submitted work. ``reused``: it is an
    active job that already did (nothing new was queued); ``snapshot`` is that job as it was then."""

    job_id: str
    reused: bool = False
    snapshot: Optional[JobSnapshot] = None


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
    metadata_mtime: float = 0.0  # the output folder's metadata.json last read for its translated title
    issued_stop: Optional[str] = None  # the last stop mode whose protocol ran for this job
    sign_in_flagged: bool = False  # a "sign_in_required" event was sent for this job
    stream_tail: str = ""  # the end of the text the current request streams (the notification's rows)
    stream_open: bool = False  # between a request's stream start and its end (``_streamed_text``)
    stream_thinking: bool = False  # the streamed text is thinking (``_note_stream_phase``)
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
    # whole log messages (``add_log_listener``): the replay backlog and the listeners
    messages: Any = field(default_factory=lambda: collections.deque(maxlen=JOB_MESSAGES_KEPT))
    message_listeners: list = field(default_factory=list)
    message_lock: threading.RLock = field(default_factory=threading.RLock)
    issue: Optional[dict] = None  # U9: the last request wait (direct_text_stream.classify_request_issue)
    refining: bool = False  # issue 12: the gate's glossary is being refined (``refine_at_gate``)
    position: str = ""  # hotfix 8: the newest in-flight request, "[5] Ch. 3 (1/4)" (``request_position``)
    spine: dict = field(default_factory=dict)  # source file basename -> its 1-based spine position (or 0)
    spine_files: Optional[list] = None  # the book's spine basenames (``_spine_position``), read once
    chapter_spine: Optional[dict] = None  # chapter number -> its 1-based spine position (``_chapter_spine_position``)
    # hotfix 1 issue 5: what the job works on (``job_target``) and its request (``spec_key``), set at submit
    target: Optional[JobTarget] = None
    spec_key: str = ""
    instance: int = 1  # multi-instance: the instance whose queue and worker run it (``job_instance``)
    isolated: bool = False  # an instance >= 2 job: its instance runtime is set up (``_prepare_instance``)
    key_owners: Optional[dict] = None  # a multi-instance job: whose key is whose (``instance_plan.key_owners``)

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
            issue=dict(self.issue) if self.issue else None,
            refining=self.refining,
            position=self.position,
        )


#: How long an issue chip stays after its line when the line names no wait (seconds).
ISSUE_DEFAULT_SECONDS = 45.0
#: Labels of the running card's issue chips (UI_SPEC §2.12.3).
ISSUE_LABELS = {"rate_limited": "Rate limited", "key_cooling": "Key cooling", "network_wait": "Waiting for network…",
                "request_queue": "⏳ Request Queue"}
#: The issue kind of a ⏳ Request Queue wait (``request_schedule.QueueWaiter`` log lines).
REQUEST_QUEUE_ISSUE = "request_queue"
_RQ_PREFIX = "⏳ Request Queue"
_RQ_LEFT = re.compile(r"(?:waiting (\d+)h (\d+)m|(\d+)h (\d+)m left)")
_RQ_STARTS = re.compile(r"(?:requests start at|starts) (\d{1,2}:\d{2}(?: [A-Z][a-z]{2})?)")


def request_queue_issue(line: str) -> Optional[tuple]:
    """``(seconds left, start clock)`` of a ⏳ Request Queue wait line ("⏳ Request Queue: waiting 4h 59m —
    requests start at 13:00 (local, UTC+4) / 09:00 UTC", "… waited 0h 03m, 4h 56m left (starts 13:00
    local)"), else None. The wait itself is ``UnifiedClient._wait_for_request_queue``; a job that waits
    for hours is not stuck: this only labels it."""
    text = str(line or "")
    if _RQ_PREFIX not in text:
        return None
    left = _RQ_LEFT.search(text)
    if left is None:
        return None
    hours, minutes = (left.group(1), left.group(2)) if left.group(1) is not None else (left.group(3), left.group(4))
    starts = _RQ_STARTS.search(text)
    return int(hours) * 3600 + int(minutes) * 60, (starts.group(1) if starts else "")


def classify_issue(line: str) -> Optional[tuple]:
    """The shared API-wait classifier (``direct_text_stream.classify_request_issue``); None without it.
    A provider safety block (``is_safety_block_line``) is ``("safety_block", None)``."""
    try:
        from direct_text_stream import classify_request_issue, is_safety_block_line
    except Exception:
        return None
    try:
        if is_safety_block_line(line):
            return "safety_block", None
        return classify_request_issue(line)
    except Exception:
        return None


def issue_label(issue: Optional[Mapping[str, Any]], now: Optional[float] = None, *, cooling_keys: int = 0) -> str:
    """"Rate limited · retrying in 30 s" / "Key cooling (2)" / "Waiting for network…" ("" for none)."""
    if not issue:
        return ""
    kind = str(issue.get("kind") or "")
    label = ISSUE_LABELS.get(kind, kind.replace("_", " ").capitalize())
    if kind == REQUEST_QUEUE_ISSUE and issue.get("starts"):
        return f"{label} · until {issue.get('starts')}"
    if kind == "key_cooling" and cooling_keys:
        return f"{label} ({cooling_keys})"
    if kind == "rate_limited" and issue.get("seconds"):
        current = now if now is not None else time.time()
        try:
            remaining = int(round(float(issue.get("until") or 0.0) - current))
        except (TypeError, ValueError):
            remaining = 0
        if remaining > 0:
            return f"{label} · retrying in {remaining} s"
    return label


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


def _only_output_dir(snap: Any) -> Optional[str]:
    """The one workspace of a finished job; None for a job over several books, whose saved counts are one book's."""
    dirs = list((getattr(snap, "output_dirs", None) or {}).values())
    if len(dirs) > 1:
        return None
    return dirs[0] if dirs else getattr(snap, "output_dir", None)


def kind_verb(kind: str) -> str:
    info = _kind_info(kind)
    return getattr(info, "verb", None) or str(kind).replace("_", " ").capitalize()


def job_verb(snap: Any) -> str:
    """The word a job's lines start with: its kind's verb, "Indexing" while a book job is in its glossary stage
    (``job_kinds.translate.indexing_phase``)."""
    phase = getattr(snap, "phase", "")
    if phase in (INDEXING, "Multipass QA scan", "Multipass refinement"):
        return phase
    if phase and getattr(snap, "kind", "") == "glossary_names":
        return phase
    return kind_verb(getattr(snap, "kind", "") or "")


def kind_icon(kind: str) -> str:
    info = _kind_info(kind)
    return getattr(info, "icon", None) or "WORK_HISTORY"


def _request_queue_note(snap: Any, now: Optional[float] = None) -> str:
    """"⏳ waiting until 13:00" while the job waits for a Request Queue window (else "")."""
    active_issue = getattr(snap, "active_issue", None)
    issue = active_issue(now) if callable(active_issue) else None
    if not issue or issue.get("kind") != REQUEST_QUEUE_ISSUE:
        return ""
    starts = str(issue.get("starts") or "")
    return f"⏳ waiting until {starts}" if starts else "⏳ waiting (Request Queue)"


_POSITION_CHAPTER = re.compile(r"\b(chapter|section|merged)\s+(-?\d+(?:\.\d+)?(?:-\d+(?:\.\d+)?)?)", re.IGNORECASE)
_POSITION_CHUNK = re.compile(r"\bchunk\s+(\d+)\s*/\s*(\d+)", re.IGNORECASE)
_POSITION_FILE = re.compile(r"([\w.-]+\.(?:x?html?|txt))\b", re.IGNORECASE)


def _same_chapter(segment: Mapping, chapter: Any) -> bool:
    """The request card ``segment`` names the chapter of the ``_POSITION_CHAPTER`` match ``chapter``."""
    match = _POSITION_CHAPTER.search(str(segment.get("label") or ""))
    return bool(match) and match.group(1).lower() == chapter.group(1).lower() and match.group(2) == chapter.group(2)


def request_position(segments: Sequence[Any], spine_of: Callable[[str], int] = lambda name: 0,
                     spine_total: Any = 0, chapter_spine: Callable[[str], int] = lambda number: 0) -> str:
    """The newest in-flight request card of a job (``direct_text_stream`` segments, spine order) as
    "[17/182] Ch. 13 (1/2)": its spine position (``spine_of(file)``) out of ``spine_total`` (an int, or a
    callable asked after ``spine_of``, which may read the spine; the position alone without a total, both
    left out when unknown), chapter number and chunk; "Sec. 2" for text sources. Between two requests, or
    while only a request that names no chapter runs, the newest finished card's: a sequential run (the
    glossary stage, an on-device model) would blink back to no chapter after every answer. A card that names
    no file (the client's own "Chapter 2 ... API call in progress" line) takes the file of a card of the same
    chapter, else ``chapter_spine(number)``. '' when no request names a chapter."""
    cards = [s for s in reversed(list(segments or ())) if isinstance(s, Mapping)]
    cards = [s for s in cards if not s.get("complete")] + [s for s in cards if s.get("complete")]
    for segment in cards:
        label = str(segment.get("label") or "")
        chapter = _POSITION_CHAPTER.search(label)
        if not chapter:
            continue
        word = {"chapter": "Ch.", "section": "Sec.", "merged": "Merged"}[chapter.group(1).lower()]
        text = f"{word} {chapter.group(2)}"
        chunk = _POSITION_CHUNK.search(label)
        if chunk and int(chunk.group(2)) > 1:  # a chapter that is not split has no chunk to name
            text += f" ({chunk.group(1)}/{chunk.group(2)})"
        found = next(filter(None, (_POSITION_FILE.search(str(s.get("label") or "")) for s in cards
                                   if _same_chapter(s, chapter))), None)
        spine = spine_of(found.group(1)) if found else chapter_spine(chapter.group(2))
        if not spine:
            return text
        total = int((spine_total() if callable(spine_total) else spine_total) or 0)
        return f"[{spine}/{total}] {text}" if total >= spine else f"[{spine}] {text}"
    return ""


def progress_view(snap: Any) -> str:
    """Which view of the Progress window a job's Progress button opens, "glossary" or "chapters": a glossary
    job's glossary, and a book job still in its glossary stage (its requests name chapters and no translation
    progress exists yet); the chapters otherwise."""
    if _spec_kind(getattr(snap, "spec", None)) in ("extract_glossary", "glossary_refine"):
        return "glossary"
    if getattr(snap, "is_active", False) and str(getattr(snap, "phase", "") or "") == INDEXING:
        return "glossary"
    progress = getattr(snap, "progress", None)
    staged = getattr(snap, "is_active", False) and getattr(snap, "position", "") and not getattr(progress, "total", 0)
    return "glossary" if staged else "chapters"


def position_line(snap: Any) -> str:
    """The job's one-line status (owner 2026-10-30), for the Jobs page, the job page, the Book page and the
    notification: "Translating • [17/182] Ch. 13 (1/2)" from the newest in-flight request
    (``request_position``); before the first request names a chapter, "Translating • 14/15"."""
    verb = job_verb(snap)
    position = str(getattr(snap, "position", "") or "")
    if position and getattr(snap, "is_active", False):
        return f"{verb} • {position}"
    progress = getattr(snap, "progress", None)
    total = int(getattr(progress, "total", 0) or 0)
    if total:
        return f"{verb} • {int(getattr(progress, 'completed', 0) or 0)}/{total}"
    return verb


def progress_line(snap: JobSnapshot, now: Optional[float] = None, *, verb: bool = False, elapsed: bool = True) -> str:
    """"[17/182] Ch. 13 (1/2) · 3 in flight · 12:41" (UI_SPEC §1.7 subtitle; "Ch 12/80" before a request names
    a chapter); the phase while the total is unknown. ``verb``: it starts "Translating • …" (``position_line``),
    for the places that do not name the job's kind already (the Jobs page row, the job page). ``elapsed`` False:
    no time, for a place that only redraws when the job changes (the Reader's live panel)."""
    parts: list[str] = []
    if snap.question:
        question = snap.question if isinstance(snap.question, Mapping) else {}
        if is_glossary_question(_question_kind(question)):
            return GLOSSARY_REFINING_TEXT if getattr(snap, "refining", False) else "Waiting for your glossary decision"
        # Tools › Async batch asks the dialog's question (``async_batch_question``)
        data = question.get("data")
        title = str((data if isinstance(data, Mapping) else {}).get("title") or "")
        return f"Waiting for your answer: {title}" if title else "Waiting for your answer"
    if snap.state is JobState.STOPPING:
        parts.append("Stopping after current request…")
    elif snap.state is JobState.FORCE_STOPPING:
        parts.append("Force stopping…")
    progress = snap.progress
    if verb and ((snap.position and snap.is_active) or progress.total):
        parts.append(position_line(snap))
    elif snap.position and snap.is_active:
        parts.append(snap.position)
    elif progress.total:
        parts.append(f"Ch {progress.completed}/{progress.total}")
    elif snap.phase and not parts:
        parts.append(snap.phase)
    if snap.in_flight:
        parts.append(f"{snap.in_flight} in flight")
    queue_note = _request_queue_note(snap, now)
    if queue_note:
        parts.append(queue_note)
    if progress.failed:
        parts.append(f"{progress.failed} failed")
    seconds = snap.elapsed(now) if elapsed else None
    if seconds is not None and snap.is_active:
        parts.append(format_duration(seconds))
    if not parts and snap.state is JobState.QUEUED:
        parts.append("Queued")
    return " · ".join(parts)


#: the notification's second line, the book's title, is cut to this many characters (owner 2026-10-30)
NOTIFICATION_TITLE_CHARS = 40


def short_title(title: str, limit: int = NOTIFICATION_TITLE_CHARS) -> str:
    """``title`` on one line, cut to ``limit`` characters with "…"."""
    text = " ".join(str(title or "").split())
    return text if len(text) <= limit else text[:max(1, limit - 1)].rstrip() + "…"


def notification_lines(snap: JobSnapshot) -> tuple:
    """The ongoing FGS notification as (headline, book): "Translating • [17/182] Ch. 13 (1/2)" in the
    notification's large title line (``position_line``, then in flight / Request Queue / stopping), and the
    book's title cut to one short line in the smaller text under it (owner 2026-10-30). While the job asks,
    the headline is "Waiting for your glossary decision" (glossary gates) or "Waiting for your answer"."""
    details: list[str] = [position_line(snap)]
    if snap.in_flight:
        details.append(f"{snap.in_flight} in flight")
    queue_note = _request_queue_note(snap)
    if queue_note:
        details.append(queue_note)
    if snap.state is JobState.STOPPING:
        details.append("stopping after current request")
    elif snap.state is JobState.FORCE_STOPPING:
        details.append("force stopping")
    if snap.question:
        details = [("Refining the glossary" if getattr(snap, "refining", False) else "Waiting for your glossary decision")
                   if is_glossary_question(_question_kind(snap.question)) else "Waiting for your answer"]
    return " · ".join(details), short_title(snap.title)


#: the streamed text the job notification shows (about three rows on a phone, Android wraps it)
NOTIFICATION_STREAM_CHARS = 120
#: what leads the streamed text's rows in the notification (``live_rows``)
STREAM_ROW_MARK = "✍️ "
THINKING_ROW_MARK = "🧠 "
_TAG_RE = re.compile(r"<[^>]*>")


def stream_row(text: str, limit: int = NOTIFICATION_STREAM_CHARS) -> str:
    """The end of streamed text as one paragraph for the notification: tags and runs of spaces and new lines
    removed, the last ``limit`` characters (from a word start) behind "…"; '' for no text."""
    import html

    plain = " ".join(html.unescape(_TAG_RE.sub(" ", str(text or ""))).split())
    if len(plain) <= limit:
        return plain
    tail = plain[-limit:]
    space = tail.find(" ")
    if 0 <= space < limit // 4:
        tail = tail[space + 1:]
    return "…" + tail


def on_device_row(threads: bool = False) -> str:
    """The running on-device request's row ("📱 412 tokens written · 6.5 tok/s"; ``threads``: " · 4 threads"
    after it, the job screen's status), '' when none runs."""
    try:
        from glossarion_mobile.services.local_llm.service import default_service
    except Exception:
        return ""
    service = default_service()
    live = getattr(getattr(service, "engine", None), "live", None) if service is not None else None
    try:
        return str(live.notification_row(threads=threads)) if live is not None else ""
    except Exception:
        log.debug("the on-device notification row failed", exc_info=True)
        return ""


def live_rows(snap: Optional[JobSnapshot], stream_tail: str = "", thinking: bool = False, *,
              device: Optional[str] = None, limit: int = NOTIFICATION_STREAM_CHARS) -> list:
    """Hotfix 3: rows under the job notification's book line while a request runs: the on-device model's
    prompt reading / speed (one row) and, with Streaming on, the end of the text it streams (about three
    rows). None while the job waits for an answer or is not running. ``device``: the on-device row to show
    (``JobService.device_row``: '' for a job of a multi-instance run that does not use the on-device model);
    None reads it here."""
    if snap is None or snap.question or not snap.is_active:
        return []
    streamed = stream_row(stream_tail, limit)
    # hotfix 4 (owner: "why is there no visual styling for that log area"): the notification is plain text
    # (flutter_foreground_task's BigTextStyle takes a string), so the streamed text gets its own marker: 🧠 for
    # the request's thinking, ✍️ for its answer
    mark = THINKING_ROW_MARK if thinking else STREAM_ROW_MARK
    rows = [row for row in (on_device_row() if device is None else device, f"{mark}{streamed}" if streamed else "")
            if row]
    return rows


#: the streamed text each job's rows show in a multi-instance notification (one job's rows stay short)
NOTIFICATION_SUMMARY_STREAM_CHARS = 80


def notification_summary(snaps: Sequence[JobSnapshot],
                         rows_for: Optional[Callable[[JobSnapshot], Sequence[str]]] = None) -> tuple:
    """The ongoing notification while jobs of several instances run (design §6): (headline, lines) with one row
    per running job, "[2] Translating • [3/12] Ch. 3 · 2 in flight · Book", each followed by its own live rows
    (``rows_for(snap)``: its 🧠 / ✍️ stream row, its on-device row). One job: ``notification_lines`` as before."""
    snaps = [snap for snap in snaps if snap is not None]
    if len(snaps) == 1:
        headline, book = notification_lines(snaps[0])
        return headline, [book, *(rows_for(snaps[0]) if rows_for is not None else ())]
    lines: list = []
    for snap in snaps:
        headline, book = notification_lines(snap)
        lines.append(f"[{snap.instance}] {headline} · {book}" if book else f"[{snap.instance}] {headline}")
        if rows_for is not None:
            lines.extend(f"    {row}" for row in rows_for(snap) if row)
    stopping = all(snap.state in (JobState.STOPPING, JobState.FORCE_STOPPING) for snap in snaps) and snaps
    headline = f"{len(snaps)} jobs running" + (" · stopping" if stopping else "")
    return headline, lines


def notification_text(snap: JobSnapshot) -> str:
    """``notification_lines`` on one line ("Translating • [17/182] Ch. 13 (1/2) — Book"), where only one fits
    (the iOS continued-processing task, diagnostics)."""
    headline, book = notification_lines(snap)
    return f"{headline} — {book}" if book else headline


def already_active_text(snap: Any) -> str:
    """Hotfix 1 issue 5: why nothing new was queued - "Translating “Book.epub” is already running (Chat ·
    Novel)" / "… is already queued" / "… is still stopping"; a generic line without a snapshot."""
    if snap is None:
        return "This job is already running"
    spec = getattr(snap, "spec", None)
    kind = _spec_kind(spec) if spec is not None else ""
    state = getattr(snap, "state", None)
    state = str(getattr(state, "value", state) or "").upper()
    where = {"QUEUED": "already queued", "STOPPING": "still stopping",
             "FORCE_STOPPING": "still stopping"}.get(state, "already running")
    title = str(getattr(spec, "title", "") or "").strip()
    text = f"{kind_verb(kind)} “{title}” is {where}" if title else f"{kind_verb(kind)} is {where}"
    origin = getattr(spec, "origin", None) or {}
    label = str(origin.get("label") or "").strip() if isinstance(origin, Mapping) else ""
    return f"{text} ({label})" if label else text


def strip_model_for(snap: Optional[JobSnapshot], queued: int = 0, now: Optional[float] = None) -> Any:
    """``AppState.JobStripModel`` for the global JobStrip (None hides it)."""
    if snap is None:
        return None
    from glossarion_mobile.state.app_state import JobStripModel

    origin = snap.spec.origin or {}
    owner_chat = str(origin.get("cid")) if origin.get("type") == "chat" and origin.get("cid") is not None else None
    # The chat's own Job card is the progress UI only for its RunController runs (params["chat_id"]).
    params = snap.spec.params if isinstance(snap.spec.params, Mapping) else {}
    chat_card = owner_chat is not None and params.get("chat_id") is not None and str(params.get("chat_id")) == owner_chat
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
                             kind_icon=kind_icon(snap.kind), queued=queued, state=state, owner_chat=owner_chat,
                             chat_card=chat_card)
    if snap.state is JobState.STOPPING:
        state = "finishing"
    elif snap.state is JobState.FORCE_STOPPING:
        state = "stopping"
    else:
        state = "running"
    return JobStripModel(
        title=f"{job_verb(snap)} · {snap.title}",
        subtitle=progress_line(snap, now),
        progress=snap.progress.fraction,
        kind_icon=kind_icon(snap.kind),
        queued=queued,
        state=state,
        warning=bool(snap.question) and not getattr(snap, "refining", False),
        owner_chat=owner_chat,
        chat_card=chat_card,
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


@contextlib.contextmanager
def client_log_handlers(outer: Any) -> Iterator[None]:
    """The desktop ``_attach_gui_logging_handlers`` for ``outer`` (anything with ``append_log``) while the
    block runs: the shared GUI-free attach (``translate_headers_standalone._attach_logging_handlers``: the
    ``unified_api_client``, ``httpx``, ``requests.packages.urllib3`` and ``openai`` loggers), and every
    handler of ``outer`` (``outer_id``) detached again at the end - also one a shared function attached to
    it again meanwhile (``translate_headers_now`` re-attaches to its view: the same ``outer_id`` replaces
    ours, so a record is never written twice)."""
    attached = False
    try:
        headers_core = importlib.import_module("translate_headers_standalone")  # an instance's own copy there

        headers_core._attach_logging_handlers(outer)
        attached = True
    except Exception:
        log.debug("the API client log handlers could not be attached", exc_info=True)
    try:
        yield
    finally:
        if attached:
            outer_id = id(outer)
            for logger in [logging.getLogger()] + [
                    v for v in list(logging.Logger.manager.loggerDict.values()) if isinstance(v, logging.Logger)]:
                if any(getattr(h, "outer_id", None) == outer_id for h in logger.handlers):
                    logger.handlers = [h for h in logger.handlers if getattr(h, "outer_id", None) != outer_id]


class JobBackend:
    """Lazy bridge to the shared GUI-free modules; tests pass a fake with the same methods.

    ``job_runner`` (JOB_LOCK, scoped_process_state, ProgressWatcher),
    ``stop_control`` (reset_for_new_run, request_stop), ``headless_owner`` and
    ``shutdown_utils``. Nothing here re-implements backend logic: a missing
    shared module fails the job with a clear message instead.
    """

    def __init__(self) -> None:
        self._fallback_lock = threading.RLock()
        #: instance -> the last immediate stop's ``translation-stop-cleanup`` thread of that instance (desktop:
        #: ``TranslatorGUI._translation_stop_cleanup_thread``); that instance's next job waits for it
        self._stop_cleanup_threads: dict = {}

    @property
    def stop_cleanup_thread(self) -> Optional[threading.Thread]:
        """The calling thread's instance's last stop cleanup thread (instance 1 outside every instance)."""
        return self._stop_cleanup_threads.get(_instance_key())

    @stop_cleanup_thread.setter
    def stop_cleanup_thread(self, thread: Optional[threading.Thread]) -> None:
        if thread is None:
            self._stop_cleanup_threads.pop(_instance_key(), None)
        else:
            self._stop_cleanup_threads[_instance_key()] = thread

    @staticmethod
    def _module(name: str) -> Any:
        try:
            return importlib.import_module(name)
        except ImportError as exc:
            raise JobError(f"The shared module '{name}' is not in this build ({exc}).") from exc

    @property
    def job_lock(self) -> Any:
        try:
            # shared (U3); importlib, so an instance's thread gets its own copy (``services.instance_runtime``)
            return importlib.import_module("job_runner").JOB_LOCK
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

    @contextlib.contextmanager
    def client_logs(self, host: Any) -> Iterator[None]:
        """The desktop ``_attach_gui_logging_handlers`` for one job: the API client loggers' records
        (``unified_api_client`` turns its print()s into them: "Sending API call", the streamed text, the
        retry notices; httpx, openai) reach the job's ``host.append_log`` - its log, request stream and
        live listeners - as the desktop GuiLogHandler sends them to the log panel. The shared GUI-free
        attach (``client_log_handlers``); detached when the job ends (each job has its own host). A kind
        whose shared code attaches its own handler (``KindInfo.own_client_logs``: the headers job) runs
        without this one, or every record would reach the job twice."""
        with client_log_handlers(host):
            yield

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
        stop_engine: bool = True,
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

        Multi-instance: it runs as the job's instance (``JobService._in_instance``), so every module it reaches
        is that instance's own; ``stop_engine`` False (a job of one instance that does not use the on-device
        model while another instance runs) leaves the shared on-device engine's generation alone.
        """
        if (force or not graceful) and stop_engine:  # the on-device generation ends even if the flags clear soon
            engine = sys.modules.get("glossarion_mobile.services.local_llm.service")
            engine = engine.default_service() if engine is not None else None
            if engine is not None:
                engine.stop_generation()
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
                module = _loaded("extract_glossary_from_epub")
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
            module = _loaded("TransateKRtoEN")
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
            job_runner = importlib.import_module("job_runner")  # an instance's job: its own client's watchdog
        except ImportError:
            return None
        cls = getattr(job_runner, "ProgressWatcher", None)
        return cls(resolver, host, interval=interval) if cls is not None else None

    def progress_summary(self, path: str) -> Optional[Mapping[str, Any]]:
        """The counts ``ProgressWatcher`` reports for a progress file; None when it cannot be read."""
        try:
            job_runner = importlib.import_module("job_runner")

            return job_runner._default_summary_reader(path)
        except Exception:
            log.debug("reading the progress summary of %s failed", path, exc_info=True)
            return None

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

    def request_stop(self, reason: str = "") -> Optional[str]:
        """The job stops itself (U9: a declined glossary review): the same Stop protocol as a user's
        Stop (never escalated to a force stop); the run ends Stopped and can be resumed."""
        return self.service.request_stop(self._job.id, escalate=False, reason=reason)


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
        direct_text_stream = importlib.import_module("direct_text_stream")  # shared (U3); an instance's own copy
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
    # a book's translation from the Library / Jobs page / Reader is a file run too: its request cards name the
    # chapter and file ("Chapter 13 (chunk 1/2) · ch13.xhtml"), which the job line reads (``request_position``)
    book_run = spec.kind in ("translate", "single_chapter", "extract_glossary")
    return direct_text_stream.make_stream(
        source_is_attachment=bool(params.get("is_attachment")) or book_run,
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


def _job_epub(job: Any) -> str:
    """The EPUB a job works on (the file a chat attachment send names, else its inputs), '' without one."""
    spec = getattr(job, "spec", None)
    params = _spec_field(spec, "params") if spec is not None else None
    explicit = params.get(JOB_TARGET_PARAM) if isinstance(params, Mapping) else None
    if explicit:
        paths = [explicit] if isinstance(explicit, (str, os.PathLike)) else list(explicit)
    else:
        paths = list(_spec_field(spec, "inputs") or ()) if spec is not None else []
    return next((os.fspath(p) for p in paths if str(p).lower().endswith(".epub")), "")


class JobService:
    #: seconds between the repaints that keep a running job's elapsed time moving
    elapsed_tick_interval = 1.0

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
        self._carried_on: list[JobSnapshot] = []  # ended entries a newer job of their book replaced (``carried_on``)
        self._carried_by: dict = {}  # such an entry's id -> {"job": what replaced it, "covers": it does that work}
        self._folder_keys: dict = {}  # a workspace path -> its ``target_key`` (``_folder_key``)
        self._counts_unsaved = False  # ``_follow_progress`` changed counts the state file does not have yet
        self._counts_saved_at = 0.0
        self._summaries_seen: dict = {}  # progress file -> (mtime, size) the Jobs page's refresh last read
        self._interrupted: list[JobSnapshot] = []
        self._paused = False
        self._closed = False
        self._thread: Optional[threading.Thread] = None
        # Multi-instance: instance N >= 2's queue, running job and ``gl-job-iN`` worker (instance 1 is the above)
        self._iqueues: dict[int, list[_Job]] = {}
        self._iactive: dict[int, _Job] = {}
        self._ithreads: dict[int, threading.Thread] = {}
        self._jobs_running = 0  # jobs between their config snapshot and their end (``_note_job_running``)
        # instance 1's environment / argv / streams as they were before its running job's scope changed them: an
        # instance N job starts from it (``_pristine_process_state``), never from instance 1's job state
        self._pristine: Optional[tuple] = None
        self._pristine_lock = threading.Lock()
        # owner rule: while a job of an instance >= 2 runs, instance 1's plain jobs (chat, Book page, Chapters) use
        # instance 1's API keys only: [the running plain job, its scope's undo or None, its config] (``_instance_keys``)
        self._base_plain: Optional[list] = None
        self._base_scope_lock = threading.RLock()
        # question id -> (answered event, [value], the asking job); the last answers the jobs took (repeats)
        self._answers: dict[str, tuple[threading.Event, list, Optional[_Job]]] = {}
        self._answered: "collections.OrderedDict[str, Any]" = collections.OrderedDict()
        # hotfix 1 issue 5: Resume runs one at a time; a job resumed in this session -> the job it became
        # (a second Resume returns that job while it is active)
        self._resume_lock = threading.RLock()
        self._resumed_into: dict[str, str] = {}
        self._ended_targets: dict = {}  # ended job id -> its ``job_target`` (``_ended_target``)
        self._ended_books: dict = {}  # ended job id -> its ``job_books``
        self._resumed_interrupted: set = set()  # Interrupted jobs resumed this session (a second Resume is refused)
        self.adopted_ids: list[str] = []  # interrupted jobs found at this launch (launch banner)
        #: ``spec -> params | None``: the params a Resume without its own runs with (the chat sets it:
        #: a chat job resumed from the Jobs page / banner uses the chat's model now, ``ChatRuns.with_current_model``)
        self.resume_params: Optional[Callable[[JobSpec], Optional[Mapping[str, Any]]]] = None
        # Stop protocols requested on the UI loop run in order on one short-lived ``gl-job-stop``
        # thread (env writes, backend imports, watchdog resets never block the loop). The lock is
        # held while a protocol runs; a job takes it around its run reset, so a late protocol of
        # the previous job can never land on (or be erased by) the next job's start.
        self._stop_tasks: list[tuple[_Job, str]] = []
        self._stop_thread: Optional[threading.Thread] = None
        self._stop_lock = threading.RLock()
        # one such lock per instance (instance 1's is ``_stop_lock``): instances never wait for each other's starts
        self._stop_locks: dict[int, Any] = {1: self._stop_lock}
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
        with self._lock:
            self._bring_back_locked()  # the jobs that replaced them died with the last launch

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
        """``callback(job_id, kind, data)`` for job events the service does not consume itself, plus its own
        ``sign_in_required``, ``question_resolved`` (``{"id", "kind"}``: a delivered question was
        answered or given up) and ``RESUMED_EVENT`` (``{"job"}``: Resume carried the job on in that job)."""
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
            running = tuple(job.snapshot() for job in self._running_jobs_locked())
            return JobsView(
                active=running[0] if running else None,
                queue=tuple(job.snapshot() for job in self._queued_jobs_locked()),
                interrupted=tuple(self._interrupted),
                history=tuple(self._history),
                paused=self._paused,
                carried_on=tuple(self._carried_on),
                running=running,
            )

    def _running_jobs_locked(self) -> list:
        """Every instance's running job (``_active`` / ``_iactive``): instance 1 first, then by instance number."""
        jobs = [self._active] if self._active is not None else []
        jobs.extend(self._iactive[number] for number in sorted(self._iactive))
        return jobs

    def _queued_jobs_locked(self) -> list:
        jobs = list(self._queue)
        for number in sorted(self._iqueues):
            jobs.extend(self._iqueues[number])
        return jobs

    def _queue_of_locked(self, job: _Job) -> list:
        """The queue ``job`` waits in (its instance's)."""
        return self._queue if job.instance < 2 else self._iqueues.setdefault(job.instance, [])

    def _is_running_locked(self, job: _Job) -> bool:
        """``job`` is its instance's running job."""
        return job is self._active if job.instance < 2 else self._iactive.get(job.instance) is job

    def running_snapshots(self) -> list:
        """Every instance's running job (also while it starts, waits at a gate or stops), instance 1 first."""
        with self._lock:
            return [job.snapshot() for job in self._running_jobs_locked()]

    def instance_busy(self, number: int) -> bool:
        """Instance ``number`` runs a job now."""
        wanted = coerce_instance(number)
        with self._lock:
            return (self._active is not None) if wanted < 2 else wanted in self._iactive

    @property
    def any_busy(self) -> bool:
        """A job of any instance runs (``busy`` is instance 1's)."""
        with self._lock:
            return self._active is not None or bool(self._iactive)

    def instance_of(self, job_id: str) -> int:
        """The instance job ``job_id`` runs (or ran) in."""
        snap = self.snapshot(job_id)
        return snap.instance if snap is not None else 1

    def snapshot(self, job_id: Optional[str] = None) -> Optional[JobSnapshot]:
        """The active job (``job_id=None``) or any known job."""
        with self._lock:
            if job_id is None:
                return self._active.snapshot() if self._active is not None else None
            carried = next((s for s in self._carried_on if s.id == job_id), None)
            if carried is not None:
                return carried
            job = self._find_live(job_id)
            if job is not None:
                return job.snapshot()
        return self.view().find(job_id)

    def stream_tail(self, job_id: Optional[str] = None) -> str:
        """The end of the text the active job's (or ``job_id``'s) current request streams; '' when nothing
        streams (Streaming off, between requests)."""
        with self._lock:
            job = self._active if job_id is None else self._find_live(job_id)
            return job.stream_tail if job is not None else ""

    def stream_thinking(self, job_id: Optional[str] = None) -> bool:
        """Whether ``stream_tail`` is the request's thinking (hotfix 4: the notification marks it 🧠)."""
        with self._lock:
            job = self._active if job_id is None else self._find_live(job_id)
            return bool(job.stream_thinking) if job is not None else False

    def device_row(self, job_id: Optional[str] = None, *, threads: bool = False) -> str:
        """The on-device model's status row for job ``job_id`` (instance 1's job when None; ``on_device_row``)
        while it runs and asks nothing. The on-device engine is the app's, one for every instance: while jobs of
        several instances run, only a job whose model is ``local/`` shows it ('' for the others)."""
        with self._lock:
            job = self._active if job_id is None else self._find_live(job_id)
            if job is None or job.question or job.state not in ACTIVE_STATES:
                return ""
            shared = any(other is not job for other in self._running_jobs_locked())
        if shared and not str(self._job_model(job) or "").startswith("local/"):
            return ""
        return on_device_row(threads=threads)

    def watchdog_state(self, job_id: Optional[str] = None) -> Optional[dict]:
        """``unified_api_client.get_api_watchdog_state()`` of job ``job_id``'s own instance (its in-flight
        requests; instance 1's job when None); None while that instance has not loaded the client."""
        with self._lock:
            job = self._active if job_id is None else self._find_live(job_id)
        if job is None:
            return None
        with self._in_instance(job):
            module = _loaded("unified_api_client")
            getter = getattr(module, "get_api_watchdog_state", None) if module is not None else None
            try:
                state = getter() if callable(getter) else None
            except Exception:
                log.debug("the watchdog state of job %s is unavailable", job.id, exc_info=True)
                state = None
        return dict(state) if isinstance(state, Mapping) else None

    def _find_live(self, job_id: str) -> Optional[_Job]:
        if self._active is not None and self._active.id == job_id:
            return self._active
        for job in self._iactive.values():
            if job.id == job_id:
                return job
        for job in self._queued_jobs_locked():
            if job.id == job_id:
                return job
        for job in self._recent:
            if job.id == job_id:
                return job
        return None

    @property
    def busy(self) -> bool:
        """Instance 1 runs a job (``any_busy``: any instance)."""
        with self._lock:
            return self._active is not None

    @property
    def paused(self) -> bool:
        with self._lock:
            return self._paused

    # ---- submit / queue -------------------------------------------------------------------------------

    def submit(self, spec: JobSpec, *, front: bool = False, instance: Optional[int] = None) -> str:
        """Queue ``spec`` (starts at once when idle); returns the job id: the id of the active job that already
        does this work when there is one (``submit_job``, hotfix 1 issue 5), so the caller binds to it."""
        if instance is None:  # a plain job: the call of before (a wrapped ``submit_job`` sees no new argument)
            return self.submit_job(spec, front=front).job_id
        return self.submit_job(spec, front=front, instance=instance).job_id

    def submit_job(self, spec: JobSpec, *, front: bool = False, instance: Optional[int] = None) -> SubmitResult:
        """``submit`` that says whether it queued a job (``SubmitResult``; ``reused``: nothing new was queued).

        An ACTIVE job (queued, starting, running, waiting at the glossary gate, refining, stopping) that does
        the same request (``spec_key``) or, in the same family, a target that covers this one's
        (``job_target``) is reused: two surfaces, a double tap, two chats with the same file, a Resume of a
        book that is running never queue the same work twice - whichever instance runs it. The check and the
        insert run under the service lock, so concurrent submits from any thread give one job.

        ``instance``: the job of a multi-instance run for instance N (1 included) - it goes into that
        instance's queue and runs on that instance's Translation keys; recorded as ``params["instance"]``, so
        a Resume runs on the same instance. None (the default): the spec's own ``params["instance"]`` (a
        Resume), else instance 1 exactly as before. Each instance runs one job at a time."""
        if not isinstance(spec, JobSpec):
            raise TypeError("submit() takes a JobSpec")
        if not self.has_kind(spec.kind):
            raise JobError(f"Unknown job kind: {spec.kind}")
        if instance is not None:
            spec = replace(spec, params={**dict(spec.params), INSTANCE_PARAM: coerce_instance(instance)})
        target, key = job_target(spec), spec_key(spec)  # paths are resolved outside the lock
        job = _Job(id=uuid.uuid4().hex[:12], spec=spec, created=self.clock(), target=target, spec_key=key,
                   instance=job_instance(spec))
        job.log_path = self.job_log_path(job.id)
        with self._cond:
            if self._closed:
                raise JobError("The job service is closed")
            existing = self._covering_job_locked(target, key)
            if existing is not None:
                snap = existing.snapshot()
            else:
                queue = self._queue_of_locked(job)
                if front:
                    queue.insert(0, job)
                else:
                    queue.append(job)
                self._write_active_locked()
                snap = job.snapshot()
                self._ensure_worker_for_locked(job.instance)
                self._cond.notify_all()
        if existing is not None:
            log.info("job %s reused for %s %s (it is %s)", existing.id, spec.kind, spec.title, snap.state.value)
            return SubmitResult(existing.id, True, snap)
        self._drop_superseded(target, job.id)
        log.info("job %s queued: %s %s", job.id, spec.kind, spec.title)
        self._post(self._deliver_transition, snap, None)
        job.announced.set()  # the worker starts it only now: listeners see QUEUED before STARTING
        return SubmitResult(job.id, False, snap)

    def _ended_target(self, snap: JobSnapshot) -> Optional[JobTarget]:
        """``job_target`` of an ended job's spec, resolved once per job."""
        if snap.id not in self._ended_targets:
            self._ended_targets[snap.id] = job_target(snap.spec)
        return self._ended_targets[snap.id]

    def _books_of_ended(self, snap: JobSnapshot) -> tuple:
        """``job_books`` of an ended job, resolved once per job."""
        if snap.id not in self._ended_books:
            self._ended_books[snap.id] = job_books(snap.spec, snap.output_dirs, snap.output_dir)
        return self._ended_books[snap.id]

    def _drop_superseded(self, target: Optional[JobTarget] = None, new_id: str = "") -> None:
        """Owner 2026-10-30 ("the chat and the jobs menu in sync") and 2026-11-01 ("I somehow have 3 separate finished
        jobs"): a book has one job on the Jobs page, wherever it was started (the chat, the Library, the Reader, the
        Jobs page). While a job for a book is queued or running none of the book's ended entries is under Finished
        (``job_books``; ``_carry_on_locked`` keeps them readable by id). A new job (``target``, ``new_id``) also
        carries on the Interrupted entries its target covers. Blocking only on resolving the paths."""
        with self._lock:
            interrupted = list(self._interrupted)
            history = list(self._history)
            live = [(job.id, job.spec, dict(job.output_dirs), job.output_dir) for job in self._live_jobs_locked()]
        resumed = {snap.id for snap in interrupted
                   if snap.id != new_id and target is not None and target.covers(self._ended_target(snap))}
        carried: dict = {}
        for job_id, spec, dirs, folder in live:
            books = job_books(spec, dirs, folder)
            target_now = job_target(spec)
            for snap in history:
                if snap.id != job_id and snap.id not in carried and same_books(self._books_of_ended(snap), books):
                    carried[snap.id] = (job_id, target_now is not None and target_now.covers(self._ended_target(snap)))
        if not resumed and not carried:
            return
        with self._lock:
            if resumed:
                self._interrupted = [s for s in self._interrupted if s.id not in resumed]
                self._write_interrupted_locked()
            if carried:
                self._carry_on_locked(carried)
        log.info("%d earlier job(s) of the running books left the Jobs page", len(resumed) + len(carried))
        self._changed()

    def _carry_on_locked(self, carried: Mapping[str, str]) -> None:
        """``carried``: ended entry id -> (the job that replaces it, whether that job does the entry's work). The
        entries leave Finished, persisted (``_bring_back_locked`` says when one returns)."""
        moved = [s for s in self._history if s.id in carried]
        self._history = [s for s in self._history if s.id not in carried]
        self._carried_on[:0] = [replace(s, resolution=s.resolution or RESOLUTION_CARRIED_ON) for s in moved]
        del self._carried_on[self.history_limit:]
        for job_id, (into, covers) in carried.items():
            self._carried_by[job_id] = {"job": into, "covers": bool(covers)}
        kept = {s.id for s in self._carried_on}
        self._carried_by = {job_id: who for job_id, who in self._carried_by.items() if job_id in kept}
        self._write_history_locked()

    def _bring_back_locked(self, job_id: Optional[str] = None, *, never_ran: bool = False) -> None:
        """Job ``job_id`` ended (None: every job, at launch nothing is running any more). What it replaced stays out
        of Finished when it carried that work on. Two kinds of entry return: all of them when it never ran (cancelled
        while queued: the book's real result is still its result), and the unfinished entries of work it did not take
        over (a QA scan or a compile of a half-done book): their Resume row. ``_drop_superseded`` then takes out
        whatever another live job of the book replaces."""
        back = []
        for snap in self._carried_on:
            who = self._carried_by.get(snap.id)
            if not who or (job_id is not None and who.get("job") != job_id):
                continue
            unfinished = snap.state is not JobState.DONE and snap.resolution == RESOLUTION_CARRIED_ON
            if never_ran or (unfinished and not who.get("covers")):
                back.append(snap)
        if not back:
            return
        ids = {snap.id for snap in back}
        self._carried_on = [s for s in self._carried_on if s.id not in ids]
        for snap in back:
            self._carried_by.pop(snap.id, None)
            snap = replace(snap, resolution=None if snap.resolution == RESOLUTION_CARRIED_ON else snap.resolution)
            at = next((i for i, s in enumerate(self._history) if (s.finished or 0) <= (snap.finished or 0)),
                      len(self._history))
            self._history.insert(at, snap)
        del self._history[self.history_limit:]
        self._write_history_locked()

    def _carrier_live_locked(self, job_id: str) -> bool:
        """The job that replaced ended entry ``job_id`` has not ended."""
        who = self._carried_by.get(job_id)
        into = who.get("job") if who else None
        return any(job.id == into for job in self._live_jobs_locked())

    def latest_job_for(self, path: Any, family: str = FAMILY_TRANSLATION) -> Optional[JobSnapshot]:
        """The job that stands for ``path``'s work now: the active one, else the newest ended or Interrupted
        job whose target names it (of ``family``); None when no job worked on it. The chat's Result card of a
        book shows it, so the chat and the Jobs page say the same thing."""
        active = self.active_job_for(path, family)
        if active is not None or not path:
            return active
        wanted = target_key(path)
        with self._lock:
            # the entries out of Finished too: a QA scan of the book does not make its last translation not have run
            ended = list(self._interrupted) + list(self._history) + list(self._carried_on)
        best = None
        for snap in ended:
            found = self._ended_target(snap)
            if found is None or found.family != family or not found.names(wanted):
                continue
            if best is None or (snap.created or 0) > (best.created or 0):
                best = snap
        return best

    def _live_jobs_locked(self) -> list:
        """The jobs that have not ended: the running ones (also while they start, wait at a gate or stop; every
        instance's), then the queues in order."""
        jobs = [job for job in self._running_jobs_locked() if job.state not in TERMINAL_STATES]
        jobs.extend(job for job in self._queued_jobs_locked() if job.state not in TERMINAL_STATES)
        return jobs

    def _covering_job_locked(self, target: Optional[JobTarget], key: str) -> Optional[_Job]:
        for job in self._live_jobs_locked():
            if key and job.spec_key == key:
                return job
            if target is not None and job.target is not None and job.target.covers(target):
                return job
        return None

    def active_job_for(self, what: Any, family: Optional[str] = None) -> Optional[JobSnapshot]:
        """The active job that already does ``what``'s work, else None (hotfix 1 issue 5).

        ``what``: a ``JobSpec`` - the job ``submit`` would reuse for it (check before recording anything, e.g.
        a chat turn); or a path (a book's source file, a workspace folder) - an active job whose target names
        it, of ``family`` when given (``FAMILY_TRANSLATION`` / ``FAMILY_GLOSSARY`` / ``FAMILY_QA`` /
        ``FAMILY_CHAPTER``). Any thread; the answer can be stale by the time it is used, so a submission
        still goes through ``submit``, which decides under the lock."""
        if isinstance(what, JobSpec):
            target, key = job_target(what), spec_key(what)
            with self._lock:
                job = self._covering_job_locked(target, key)
                return job.snapshot() if job is not None else None
        if not what:
            return None
        wanted = target_key(what)
        with self._lock:
            for job in self._live_jobs_locked():
                found = job.target
                if found is None or (family is not None and found.family != family):
                    continue
                if found.names(wanted):
                    return job.snapshot()
        return None

    def cancel_queued(self, job_id: str) -> bool:
        with self._cond:
            job = next((j for j in self._queued_jobs_locked() if j.id == job_id), None)
            if job is None:
                return False
            self._queue_of_locked(job).remove(job)
            job.state = JobState.CANCELLED
            job.finished = self.clock()
            snap = job.snapshot()
            self._bring_back_locked(job.id, never_ran=True)
            self._append_history_locked(snap)
            self._write_active_locked()
        self._post(self._deliver_transition, snap, JobState.QUEUED)
        self._drop_superseded()
        return True

    def cancel_all_queued(self) -> int:
        with self._lock:
            ids = [job.id for job in self._queued_jobs_locked()]
        return sum(1 for job_id in ids if self.cancel_queued(job_id))

    def move_queued(self, job_id: str, index: int) -> bool:
        """Reorder the queue (Jobs page drag handles); a job moves within its own instance's queue."""
        with self._lock:
            job = next((j for j in self._queued_jobs_locked() if j.id == job_id), None)
            if job is None:
                return False
            queue = self._queue_of_locked(job)
            queue.remove(job)
            index = max(0, min(int(index), len(queue)))
            queue.insert(index, job)
            self._write_active_locked()
        self._changed()
        return True

    def pause_queue(self) -> None:
        """Stop starting the next queued job of every instance (the running ones continue)."""
        with self._cond:
            self._paused = True
        self._changed()

    def resume_queue(self) -> None:
        with self._cond:
            self._paused = False
            if self._queue:
                self._ensure_worker_locked()
            for number, queue in self._iqueues.items():
                if queue:
                    self._ensure_worker_for_locked(number)
            self._cond.notify_all()
        self._changed()

    def clear_finished(self) -> None:
        with self._lock:
            self._history = []
            self._carried_on = []
            self._carried_by = {}
            self._write_history_locked()
        self._changed()

    # ---- interrupted / resume -------------------------------------------------------------------------------

    def resume(self, job_id: str, params: Optional[Mapping[str, Any]] = None) -> Optional[str]:
        """Resubmit an interrupted, stopped or failed job's spec (the backend resumes from its progress).

        ``params``: the spec's params as the caller resubmits them (the chat's Resume moves the turn
        index, the request number and a prompt profile that no longer exists: ``ChatRuns.resubmit``);
        omitted, the spec runs as it was.

        An Interrupted job that was already resumed (its entry resolved ``resumed``) is refused:
        a second Resume (launch banner, Jobs page, the chat) would run the same work twice.

        Hotfix 1 issue 5: a job that is still active returns its own id, a stopped / failed job this session
        already resumed returns the job it became while that one is active, and the resubmission itself
        reuses an active job that does the same work (``submit_job``); ``resume_job`` says which."""
        result = self.resume_job(job_id, params)
        return result.job_id if result is not None else None

    retry = resume

    def resume_job(self, job_id: str, params: Optional[Mapping[str, Any]] = None) -> Optional[SubmitResult]:
        """``resume`` that says whether it reused an active job (``SubmitResult.reused``); None when there is
        nothing to resume. Resumes run one at a time (a double tap resumes once)."""
        with self._resume_lock:
            with self._lock:
                live = self._find_live(job_id)
                if live is not None and live.state not in TERMINAL_STATES:
                    return SubmitResult(live.id, True, live.snapshot())  # it is still active itself
                if job_id in self._resumed_interrupted:
                    return None  # an Interrupted entry resumed already: refused, as before
                snap = next((s for s in self._interrupted if s.id == job_id), None)
                from_interrupted = snap is not None
                if snap is None:
                    snap = next((s for s in self._history if s.id == job_id), None)
                    if snap is not None and snap.resolution == "resumed":
                        return None  # an Interrupted entry resumed already: refused, as before
                if snap is None and self._carrier_live_locked(job_id):
                    # out of Finished for a job still at work: submitting reuses it when it does this work, and
                    # queues the Resume behind it when it does not (a QA scan of the half-done book)
                    snap = next((s for s in self._carried_on if s.id == job_id), None)
                into = self._resumed_into.get(job_id)
                current = self._find_live(into) if into else None
                if current is not None and current.state not in TERMINAL_STATES:
                    # a stopped / failed job this session resumed already, into a job that is still active
                    return SubmitResult(current.id, True, current.snapshot())
            if snap is None:
                return None
            if params is None and self.resume_params is not None:
                try:
                    params = self.resume_params(snap.spec)
                except Exception:
                    log.exception("resume params hook failed")
                    params = None
            result = self.submit_job(snap.spec if params is None else replace(snap.spec, params=dict(params)))
            left_finished = False
            with self._lock:
                self._resumed_into[job_id] = result.job_id
                if from_interrupted:
                    self._resumed_interrupted.add(job_id)
                if not from_interrupted and result.job_id != job_id:
                    # owner 2026-10-29: a resumed Stopped / Failed job carries on under Running; its old entry
                    # leaves Finished (it was the same job shown twice). Its new run lands there when it ends.
                    if any(s.id == job_id for s in self._history):
                        self._carry_on_locked({job_id: (result.job_id, True)})
                        left_finished = True
                    elif job_id in self._carried_by:
                        self._carried_by[job_id] = {"job": result.job_id, "covers": True}
                        self._write_history_locked()
            if from_interrupted:
                self._resolve_interrupted(job_id, "resumed")
            if left_finished:
                self._changed()
            if result.job_id != job_id:
                # its "Stopped: … Tap to resume" notification goes (JobsFeature), whichever Resume button it was
                self._post(self._deliver_event, job_id, RESUMED_EVENT, {"job": result.job_id})
            return result

    def discard(self, job_id: str) -> bool:
        return self._resolve_interrupted(job_id, "discarded")

    def _resolve_interrupted(self, job_id: str, resolution: str) -> bool:
        with self._lock:
            snap = next((s for s in self._interrupted if s.id == job_id), None)
            if snap is None:
                return False
            self._interrupted.remove(snap)
            if resolution != "resumed":
                # a resumed one carries on as its new job (owner 2026-10-30: the same job was listed twice,
                # Interrupted under Finished next to the run that resumed it)
                self._bring_back_locked(job_id, never_ran=snap.started is None)
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

        ``job_id`` None: instance 1's running job (``request_stop_all`` stops every instance's); a job id: that
        job, whichever instance runs it - the protocol then runs as its instance, on its own modules.
        """
        with self._lock:
            if job_id is None:
                active = self._active
            else:
                active = next((j for j in self._running_jobs_locked() if j.id == job_id), None)
            if job_id is not None and active is None:
                if any(j.id == job_id for j in self._queued_jobs_locked()):
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
            if job.issue and job.issue.get("kind") == REQUEST_QUEUE_ISSUE:
                job.issue = None  # every Stop ends a Request Queue wait at once (nothing was sent)
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
            with self._stop_lock_for(job):
                self._issue_stop(job, mode)
        if snap.state is not previous:
            self._post(self._deliver_transition, snap, previous)
        else:
            self._changed()
        return mode

    def request_stop_all(self, *, force: bool = False, escalate: bool = True, reason: str = "") -> dict:
        """Stop every instance's running job (the notification's Stop, a system time limit): ``{job id: the stop
        mode used}``. Queued jobs stay queued (``cancel_all_queued``)."""
        with self._lock:
            ids = [job.id for job in self._running_jobs_locked() if job.state not in TERMINAL_STATES]
        return {job_id: self.request_stop(job_id, force=force, escalate=escalate, reason=reason) for job_id in ids}

    def _in_instance(self, job: Any) -> Any:
        """A context that runs as ``job``'s instance (its runtime's modules, environment and streams); instance 1,
        or an instance whose runtime was never made: nothing changes."""
        number = getattr(job, "instance", 1) or 1
        if number < 2:
            return contextlib.nullcontext()
        runtime = _installed_runtime()
        if runtime is None or runtime.get(number) is None:
            return contextlib.nullcontext()
        return runtime.activated(number)

    def _others_running(self, job: Any) -> bool:
        """A job of another instance runs besides ``job``."""
        with self._lock:
            return any(other is not job for other in self._running_jobs_locked())

    def _stop_lock_for(self, job: Any) -> Any:
        """The lock ``job``'s instance holds around a stop protocol and its jobs' run resets (``_stop_lock``)."""
        number = getattr(job, "instance", 1) or 1
        with self._lock:
            lock = self._stop_locks.get(number)
            if lock is None:
                lock = self._stop_locks[number] = threading.RLock()
            return lock

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
                with self._stop_lock_for(job):
                    with self._lock:
                        # a protocol for a job that already ended must not reach the next one
                        current = self._is_running_locked(job) and job.state not in TERMINAL_STATES
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

        if job.instance >= 2 and not job.isolated:
            # just picked, its instance runtime not set up yet: nothing of it runs, and the stop protocol would run
            # on instance 1's modules and environment (stopping instance 1's job); the latch alone ends it
            set_stop_requested()
            return
        _graceful, wait = self._graceful_setting(job)
        try:
            stop_kind = getattr(self._kind(job.spec.kind), "stop_kind", "translation") or "translation"
        except Exception:
            stop_kind = "translation"
        extra: dict = {}
        shared = job.instance >= 2 or self._others_running(job)
        if shared:
            # the on-device engine is the app's, not this job's: its generation belongs to whichever job uses it
            if not str(self._job_model(job) or "").startswith("local/"):
                extra["stop_engine"] = False
        # a graceful stop silences the HTTP loggers, which are the process's: not while another instance's job
        # writes its own log with them
        http_levels = _http_logger_levels() if shared else None
        try:
            with self._in_instance(job):  # an instance's job stops through its own stop_control / client
                self.backend.request_stop(
                    graceful=graceful,
                    wait_for_chunks=bool(wait),
                    force=mode == STOP_FORCE,
                    set_stop_requested=set_stop_requested,
                    log=lambda text="", *a, **k: self._job_log(job, text, {}),
                    kind=stop_kind,
                    job_kind=job.spec.kind,
                    owner=job.owner,
                    **extra,
                )
        except Exception as exc:
            log.warning("stop_control.request_stop failed (%s); latching the stop flag only", exc)
        finally:
            if http_levels is not None and self._others_running(job):
                _restore_http_logger_levels(http_levels)
        if not latched:  # the latch must always end up set
            set_stop_requested()
        if job.instance < 2 and self._others_running(job):
            with self._lock:
                ended = job.state in TERMINAL_STATES or not self._is_running_locked(job)
            if ended:  # the job saw the latch and ended before the protocol raised its module flags
                _release_base_stop_flags()

    # ---- questions ----------------------------------------------------------------------------------------------

    def answer(self, question_id: str, value: Any) -> bool:
        """Answer a job's pending question (the approval card's Yes / No, the Book page / Jobs sheet, the
        notification). Hotfix 1: idempotent - the first answer is the one the job gets; the same answer again
        is a no-op (True), a different one is refused (False). Refused while the question's glossary is being
        refined (``refine_at_gate``, issue 2): the job waits at the gate until the refinement ended, so its
        translation never starts before it."""
        with self._lock:
            entry = self._answers.get(question_id)
            if entry is None:
                # already taken by the job: a repeat of its answer is still a no-op success
                return question_id in self._answered and self._answered[question_id] == value
            event, slot, job = entry
            if job is not None and job.refining:
                log.info("answer to %s refused: its glossary is being refined", question_id)
                return False
            if event.is_set():
                return slot[0] == value
            slot[0] = value
            event.set()
        return True

    def is_refining(self, job_id: Optional[str]) -> bool:
        """The job's gate glossary is being refined now (``refine_at_gate``; its answers are refused)."""
        if job_id is None:
            return False
        with self._lock:
            job = self._find_live(str(job_id))
            return bool(job is not None and job.refining)

    def refine_at_gate(self, job_id: str, glossary_path: str, selected_types: Optional[Sequence[str]] = None) -> dict:
        """U12 item 5, "✨ Refine" on the glossary approval card: refine the generated glossary while the job
        waits at the gate (blocking; call it off the UI loop). A refinement *job* would queue behind the
        paused translation forever (one job at a time), so this runs the shared manual refinement
        (``glossary_progress_core.plan_manual_glossary_refinement`` / ``run_manual_glossary_refinement``)
        with the paused job's own owner and environment; its log lines go to the job's log. The question
        stays open: the user then accepts (Yes), edits or rejects the refined file.

        Hotfix 1 issue 2: from the tap on, the gate refuses answers (``answer``) and its alerts and approval
        card are hidden (``JobSnapshot.refining``), so Yes cannot start the translation while the refinement
        sends its requests; a second Refine while one runs, or a Refine after the gate was answered, is
        refused. A Stop meanwhile is honoured once the refinement has ended (it polls the owner's stop flag)."""
        if not glossary_path or not os.path.isfile(glossary_path):
            raise RuntimeError("The generated glossary file was not found")
        with self._lock:
            job = self._find_live(job_id)
            question = dict(job.question) if job is not None and job.question else None
            entry = self._answers.get(str(question.get("id"))) if question else None
            if job is None or question is None or job.owner is None or (entry is not None and entry[0].is_set()):
                raise RuntimeError("The job is not waiting for a glossary decision")
            if job.refining:
                raise RuntimeError("The glossary is already being refined")
            job.refining = True  # under the lock: from here on the gate refuses answers
        try:
            return self._refine_at_gate(job, glossary_path, selected_types)
        finally:
            with self._lock:
                job.refining = False  # also when the refinement could not start (``_resume_question`` did not run)

    def _refine_at_gate(self, job: _Job, glossary_path: str, selected_types: Optional[Sequence[str]]) -> dict:
        question = self._suspend_question(job)  # the card and the alerts go away at once
        try:
            with self._in_instance(job):  # the paused job's own instance: its modules, environment and keys
                return self._run_gate_refinement(job, glossary_path, selected_types)
        finally:
            self._resume_question(job, question)

    def _run_gate_refinement(self, job: _Job, glossary_path: str, selected_types: Optional[Sequence[str]]) -> dict:
        gpc = importlib.import_module("glossary_progress_core")  # the job's instance's own (``_in_instance``)

        from glossarion_mobile.services.library import bind_call

        owner = job.owner
        types = [str(t) for t in (selected_types or ()) if str(t or "").strip()]
        if not types:
            types = [str(t) for t, cfg in dict(getattr(owner, "custom_entry_types", None) or {}).items()
                     if not isinstance(cfg, Mapping) or cfg.get("enabled", True)] or ["character", "terms"]

        def log(text: Any = "", *args: Any, **kwargs: Any) -> None:
            self._job_log(job, text, kwargs)

        # Issue 14: the book's glossary progress file records the refined entry types as completed (the
        # Glossary Progress refinement rows), like the desktop's refinement pass
        from glossarion_mobile.job_kinds.glossary import progress_for_glossary

        progress_path = progress_for_glossary(glossary_path)
        planned = bind_call(gpc.plan_manual_glossary_refinement, owner, glossary_path=glossary_path,
                            progress_path=progress_path, source_path=None, selected_types=types,
                            target_chunk_count=None, log=log)
        if not planned:
            raise RuntimeError("The selected entry type(s) contain no glossary entries")
        options, plan = planned[0], planned[1]
        log(f"\n✨ Refining the generated glossary before translation: {', '.join(types)}")
        gpc.run_manual_glossary_refinement(owner, glossary_path, progress_path, options, plan)
        log("✨ Glossary refinement finished: review it, then choose Edit, Yes or No")
        return {"ok": True, "glossary_path": glossary_path}

    def _suspend_question(self, job: _Job) -> Optional[dict]:
        """Issue 12: the gate's "action required" alerts go away while its glossary is refined (the
        notification is withdrawn through ``QUESTION_SUSPENDED_EVENT``; the JobStrip warning and the Book
        page review follow ``JobSnapshot.refining``). Returns the question being refined."""
        with self._lock:
            question = dict(job.question) if job.question else None
            job.refining = True
            job.warning = False
        if question is not None:
            self._post(self._deliver_event, job.id, QUESTION_SUSPENDED_EVENT,
                       {"id": question.get("id"), "kind": question.get("kind")})
        self._changed()
        return question

    def _resume_question(self, job: _Job, question: Optional[dict]) -> None:
        """The refinement ended: a question that still waits raises its alerts again (delivered anew to the
        question listeners: the "Glossary ready" notification / snackbar unless its card is on screen)."""
        stop_event = getattr(job, "stop_event", None)
        with self._lock:
            job.refining = False
            waiting = question is not None and bool(job.question) and job.question.get("id") == question.get("id")
            if waiting and stop_event is not None and stop_event.is_set():
                waiting = False  # stopped meanwhile: the gate gives up now, nothing is asked again
            if waiting:
                job.warning = True
            snap = job.snapshot()
            current = dict(job.question) if waiting else None
        if current is not None:
            self._post(self._deliver_question, snap, current)
        self._changed()

    def pending_question(self, job_id: Optional[str] = None) -> Optional[dict]:
        with self._lock:
            job = self._active if job_id is None else self._find_live(job_id)
            return dict(job.question) if job is not None and job.question else None

    def _job_ask(self, job: _Job, kind: str, data: Mapping[str, Any]) -> Any:
        """The job thread's blocking question (``host.ask``): delivered to the question listeners (the
        chat's approval card, the Book page / Jobs sheet, the notification) and answered by ``answer``;
        Stop gives up with the default. A delivered question ends with the job event
        ``question_resolved`` ``{"id", "kind"}`` (posted, so it also arrives while the UI loop's pump is
        parked, e.g. to cancel its notification).

        "Always accept generated glossaries" (``params["auto_accept_glossary"]``, set at Send by a chat
        whose setting is on): a glossary gate is answered Yes here, on the job thread, with the desktop's
        accepted log line; nothing is delivered (no card, no notification, no UI loop needed, so it works
        with the screen off). Mobile divergence: the chat's request cards are then not frozen at the gate
        (``RunController.commit_gate`` = desktop ``_commit_active_request_phase`` runs only when the card
        shows). Other kinds (``async_batch_question``) always ask.
        """
        data = dict(data)
        default = data.pop("default", None)
        params = job.spec.params if isinstance(job.spec.params, Mapping) else {}
        if is_glossary_question(kind) and params.get(AUTO_ACCEPT_GLOSSARY_PARAM):
            if job.stop_event.is_set():
                return default  # stopping: the gate gives up like an unanswered question
            self._job_log(job, GLOSSARY_AUTO_ACCEPTED_LINE, {})
            return True
        if not self._question_listeners:
            return default
        question = {"id": uuid.uuid4().hex[:12], "job_id": job.id, "kind": kind, "data": data,
                    "asked_at": self.clock(), "default": default}
        event = threading.Event()
        slot = [default]
        with self._lock:
            self._answers[question["id"]] = (event, slot, job)
            job.question = question
            job.warning = True
            snap = job.snapshot()
        self._post(self._deliver_question, snap, question)
        self._changed()
        try:
            while not event.wait(0.25):
                # A Stop gives the gate up - never while its glossary is being refined (hotfix 1 issue 2):
                # the refinement runs on this job's owner and process state, so the job waits for it to
                # end (it stops early on the owner's stop flag) before the next job may start.
                if job.stop_event.is_set() and not job.refining:
                    break
            return slot[0]
        finally:
            with self._lock:
                self._answers.pop(question["id"], None)
                if event.is_set():
                    self._answered[question["id"]] = slot[0]
                    while len(self._answered) > ANSWERS_KEPT:
                        self._answered.popitem(last=False)
                job.question = None
                job.warning = False
            self._post(self._deliver_event, job.id, "question_resolved", {"id": question["id"], "kind": kind})
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

    def _ensure_worker_for_locked(self, number: int) -> None:
        """The worker of instance ``number``: instance 1's ``gl-job``, else that instance's ``gl-job-iN``."""
        if number < 2:
            self._ensure_worker_locked()
            return
        thread = self._ithreads.get(number)
        if thread is not None and thread.is_alive():
            return
        thread = threading.Thread(target=self._instance_worker, args=(number,), name=f"gl-job-i{number}", daemon=True)
        thread._gl_instance = number  # every thread it starts belongs to the instance too (instance_runtime)
        self._ithreads[number] = thread
        thread.start()

    def _instance_worker(self, number: int) -> None:
        """``_worker`` for instance ``number`` >= 2: its own queue, one job at a time, beside every other instance."""
        while True:
            with self._cond:
                while not self._closed and (self._paused or not self._iqueues.get(number)):
                    self._cond.wait()
                if self._closed:
                    return
                job = self._iqueues[number].pop(0)
                self._iactive[number] = job
            job.announced.wait(5.0)  # submit() posts QUEUED right after queueing; keep the order
            try:
                self._execute(job)
            except BaseException:  # noqa: BLE001 - the worker must survive anything
                log.exception("job worker of instance %d crashed on %s", number, job.id)
            finally:
                with self._cond:
                    if self._iactive.get(number) is job:
                        del self._iactive[number]
                    self._cond.notify_all()

    def _prepare_instance(self, job: _Job) -> Any:
        """Instance N's runtime for ``job`` (N >= 2), set up on its worker: installed on first use, its process
        state reset to instance 1's outside any job (``_pristine_process_state``) and its essential modules copied.
        ``JobError`` when it cannot be isolated: the job fails, it never runs on shared modules."""
        number = job.instance
        try:
            from glossarion_mobile.services import instance_runtime

            runtime = instance_runtime.ensure(number)
            env, argv, stdout, stderr = self._pristine_process_state()
            runtime.prepare(env=_without_key_env(env), argv=argv, stdout=stdout, stderr=stderr)
            runtime.warm(instance_runtime.ESSENTIAL_MODULES)
        except Exception as exc:
            log.exception("instance %d could not be set up for job %s", number, job.id)
            raise JobError(f"Instance {number} could not be isolated, so this job did not start: {exc}") from exc
        return runtime

    def _pristine_process_state(self) -> tuple:
        """(environment, argv, stdout, stderr) of instance 1 outside any job: what instance 1's running job saw
        before its scope changed them (``_capture_pristine``), else the process state now - read while instance
        1's job lock is free, so no owner built outside a job (the settings env preview) is caught halfway."""
        from glossarion_mobile.services import instance_runtime

        with self._pristine_lock:
            saved = self._pristine
        if saved is not None:
            return saved
        with instance_runtime.activated(None):
            lock = self.backend.job_lock
        acquired = False
        try:
            acquired = bool(lock.acquire(timeout=2.0)) if hasattr(lock, "acquire") else False
        except Exception:
            acquired = False
        try:
            with self._pristine_lock:
                saved = self._pristine  # instance 1's job may have started meanwhile
                if saved is None:
                    saved = (dict(instance_runtime.base_environ()), list(instance_runtime.base_argv()),
                             instance_runtime.base_stream("stdout"), instance_runtime.base_stream("stderr"))
        finally:
            if acquired:
                lock.release()
        return saved

    def _capture_pristine(self, job: _Job) -> None:
        """Instance 1's job is about to enter its scope: what an instance N job starts from meanwhile."""
        if job.instance >= 2:
            return
        with self._pristine_lock:
            self._pristine = (dict(os.environ), list(sys.argv), sys.stdout, sys.stderr)

    def _release_pristine(self, job: _Job) -> None:
        if job.instance >= 2:
            return
        with self._pristine_lock:
            self._pristine = None

    @contextlib.contextmanager
    def _instance_keys(self, job: _Job, config: Mapping[str, Any]) -> Iterator[None]:
        """A job of a multi-instance run (``params["instance"]``, instance 1 too): while it runs, the API client
        uses only its instance's keys, in every key pool - every key list it is given or reads (the glossary
        extractor loads the pools from config.json itself), the main-key retry and the fallback paths - and a pool
        without an enabled key of the instance is off for it, so its tasks use the instance's own Translation keys.
        The desktop instances' own guard (``instance_worker.restrict_keys``: ``GLOSSARION_KEY_INSTANCE`` in the
        job's environment, the client's pools detached and rebuilt from ``config``, the job's
        ``config_for_instance`` snapshot) on the job's instance's client, entered before the owner is built; its
        request guard knows whose key is whose (``job.key_owners``) and refuses another instance's key outright.

        A job of no multi-instance run (a chat translation, the Book page, Chapters) is left as it always was,
        except while a job of an instance >= 2 runs (owner rule: instance 2's keys are never instance 1's): then it
        uses instance 1's keys only, from its start, or from the moment such a job starts (``_scope_plain_job``)."""
        if job.spec.params.get(INSTANCE_PARAM) is None:
            with self._base_scope_lock:
                undo = None
                if self._instance_jobs_alive():
                    try:
                        undo = self._plain_scope(config)
                    except Exception as exc:
                        raise JobError(f"Instance 1's API keys could not be kept to it while another instance runs: "
                                       f"{exc}") from exc
                self._base_plain = [job, undo, config]
            try:
                yield
            finally:
                with self._base_scope_lock:  # inside the job's own process-state scope: undone before it restores
                    entry, self._base_plain = self._base_plain, None
                    if entry is not None and entry[1] is not None:
                        entry[1]()
            return
        if job.instance >= 2:
            self._scope_plain_job()
        try:
            restore = importlib.import_module("instance_worker").restrict_keys(
                job.instance, config, key_owners=job.key_owners)
        except Exception as exc:
            raise JobError(f"Instance {job.instance}'s API keys could not be kept to it: {exc}") from exc
        try:
            yield
        finally:
            restore()

    def _instance_jobs_alive(self) -> bool:
        """A job of an instance >= 2 is picked or running."""
        with self._lock:
            return any(number >= 2 for number in self._iactive)

    def _plain_scope(self, config: Mapping[str, Any]) -> Callable[[], None]:
        """Instance 1's keys only for instance 1's side (``restrict_keys(1, config_for_instance(config, 1))``, the
        request guard told every instance's keys); returns the undo. Runs on instance 1's modules and env."""
        instance_plan = importlib.import_module("instance_plan")
        try:
            scoped = instance_plan.config_for_instance(config, 1)
        except ValueError:
            scoped = dict(config)  # no Translation key of its own: the client's scope still narrows every list
        return importlib.import_module("instance_worker").restrict_keys(
            1, scoped, key_owners=instance_plan.key_owners(config))

    def _scope_plain_job(self) -> None:
        """A job of an instance >= 2 starts while a plain job of instance 1 runs unscoped: from now on that job
        uses instance 1's keys only (its own scope undoes it, ``_instance_keys``)."""
        with self._base_scope_lock:
            entry = self._base_plain
            if entry is None or entry[1] is not None:
                return
            try:
                from glossarion_mobile.services import instance_runtime

                with instance_runtime.activated(None):  # instance 1's modules and environment
                    entry[1] = self._plain_scope(entry[2])
            except Exception:
                log.exception("instance 1's running job could not be kept to instance 1's keys")

    def _kind_of(self, job: _Job) -> Any:
        """``job``'s ``KindInfo``: an instance's job runs the adapter of the instance's own copy of the job kinds."""
        if job.instance < 2 or self._kinds is not None:
            return self._kind(job.spec.kind)
        from glossarion_mobile.services import instance_runtime

        runtime = instance_runtime.get(job.instance)
        if runtime is None:
            raise JobError(f"Instance {job.instance} is not set up")
        return runtime.module("glossarion_mobile.job_kinds").get_kind(job.spec.kind)

    def _note_job_running(self, running: bool) -> None:
        """``set_job_running`` for the Settings banner: True when a job takes its snapshot, False once the last
        running job of every instance ended (one job: True then False, as before)."""
        with self._lock:
            self._jobs_running = max(0, self._jobs_running + (1 if running else -1))
            last = self._jobs_running == 0
        if running or last:
            try:
                self.config_store.set_job_running(bool(running))
            except Exception:
                log.debug("set_job_running(%s) failed", running, exc_info=True)

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
            self._note_job_running(True)
        elif self._config_loader is not None:
            config = dict(self._config_loader() or {})
        config = copy.deepcopy(config if isinstance(config, dict) else {})
        overrides = job.spec.params.get("config_overrides")
        if isinstance(overrides, Mapping):
            config.update(copy.deepcopy(dict(overrides)))
        if job.spec.params.get(INSTANCE_PARAM) is not None:
            # a job of a multi-instance run uses only its instance's keys, in every pool (a pool without one is off
            # for it); an instance >= 2 runs on its own Translation key in place of the main API key (instance 1's)
            import instance_plan

            # whose key is whose (fingerprints of the whole config): the request guard of the job's client
            job.key_owners = instance_plan.key_owners(config)
            try:
                config = instance_plan.config_for_instance(config, job.instance)
            except ValueError as exc:
                if store is not None:
                    self._note_job_running(False)  # the job never took its snapshot
                raise JobError(f"{exc}: assign a Translation key to instance {job.instance} in the Multi-Key Manager"
                               ) from exc
        # the mobile Streaming default (absent toggles ON) and the thoughts lock; config.json is untouched
        from glossarion_mobile.state.setting_writes import with_mobile_streaming_defaults

        return with_mobile_streaming_defaults(config)

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

    def _track_position(self, job: _Job) -> None:
        """Hotfix 8 (owner 2026-10-30: the "Ch 8/9" tracker "is not tracking the actual remaining chapters"):
        the newest in-flight request as "[spine] Ch. N (chunk/total)" for the job line and notification."""
        stream = job.stream
        try:
            segments = list(stream.segments(drain=False)) if stream is not None else []
        except Exception:
            return
        position = request_position(segments, lambda name: self._spine_position(job, name),
                                    lambda: len(job.spine_files or ()),
                                    lambda number: self._chapter_spine_position(job, number))
        if position == job.position:
            return
        with self._lock:
            job.position = position
        self._changed()

    @staticmethod
    def _spine_position(job: _Job, name: str) -> int:
        """``name``'s 1-based place in the book's spine (the workspace OPF, like the Progress manager's
        "[005]", else the job's own EPUB: the glossary stage runs before the workspace exists); 0 when
        unknown. The spine (``job.spine_files``, its length the "/182") is read once."""
        key = os.path.basename(str(name or "")).lower()
        if key not in job.spine:
            if not JobService._read_spine(job):
                return 0  # no OPF in the workspace yet: looked up again on the next request
            stem = os.path.splitext(key)[0]
            job.spine[key] = next((index + 1 for index, base in enumerate(job.spine_files)
                                   if base == key or os.path.splitext(base)[0] == stem), 0)
        return job.spine[key]

    @staticmethod
    def _read_spine(job: _Job) -> list:
        """``job.spine_files``, read when it has none yet."""
        if not job.spine_files:
            try:
                from TransateKRtoEN import _single_pass_spine_files

                job.spine_files = [os.path.basename(str(f or "")).lower()
                                   for f in _single_pass_spine_files(job.output_dir, _job_epub(job))]
            except Exception:
                log.debug("the book's spine is unavailable", exc_info=True)
        return job.spine_files or []

    @staticmethod
    def _chapter_spine_position(job: _Job, number: str) -> int:
        """The 1-based spine place of the file that the pipeline numbers as chapter ``number`` (a request card
        that names no file; "1-99", a merged request's first), by the pipeline's own display numbering. 0 when
        unknown, or when two spine files carry that number: no position beats a wrong one."""
        if job.chapter_spine is None:
            files = JobService._read_spine(job)
            if not files:
                return 0  # no OPF in the workspace yet: looked up again on the next request
            try:
                from TransateKRtoEN import _assign_translation_display_chapter_numbers, is_configured_special_filename

                numbers = _assign_translation_display_chapter_numbers([{"filename": base} for base in files],
                                                                      is_configured_special_filename)
            except Exception:
                log.debug("the book's chapter numbers are unavailable", exc_info=True)
                return 0
            job.chapter_spine = {value: index + 1 for index, value in enumerate(numbers)
                                 if value > 0 and numbers.count(value) == 1}
        first = re.match(r"\d+(?![\d.])", str(number or ""))
        return job.chapter_spine.get(int(first.group(0)), 0) if first else 0

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
        ticked = [self.clock()]

        def loop() -> None:
            while not stop.wait(interval):
                try:
                    drain(final=False)
                except Exception:
                    log.debug("request stream drain failed", exc_info=True)
                self._track_position(job)
                # the job line's elapsed time: nothing else repaints while one long request runs
                if self.clock() - ticked[0] >= self.elapsed_tick_interval:
                    ticked[0] = self.clock()
                    self._changed()

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
        if job.instance >= 2:
            try:
                # on gl-job-iN: from here on the job's imports, environment and streams are its instance's own
                self._prepare_instance(job)
                job.isolated = True  # a Stop now runs in the instance (before: it only latches, ``_issue_stop``)
            except JobError as exc:
                with self._lock:
                    job.started = self.clock()
                    job.buffer = LogBuffer(JOB_LOG_MAXLEN, name=f"job:{job.id}")
                self._transition(job, JobState.STARTING, only_from={JobState.QUEUED})
                self._job_log(job, f"❌ {exc}", {})
                self._finish(job, None, str(exc))
                return
        try:
            kind = self._kind_of(job)
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
        where = f" · Instance {job.instance}" if job.spec.params.get(INSTANCE_PARAM) is not None else ""
        self._job_log(job, f"▶ {kind_verb(job.spec.kind)} {job.spec.title}{where}", {})
        result: Any = None
        error: Optional[str] = None
        exit_ok = True
        config_taken = False
        try:
            if job.stop_event.is_set():
                raise _Cancelled()
            config = self._config_snapshot(job)
            config_taken = True
            lock = self.backend.job_lock  # an instance's job: its own job_runner copy's JOB_LOCK
            self._capture_pristine(job)
            with self.backend.scoped_process_state(capture_stdout=host.log, lock=lock), \
                    no_sign_in_scope(job.spec.kind), local_context_scope(), _env_scope("GLOSSARY_STREAM_CARDS", "1"), \
                    self._instance_keys(job, config):
                # Held from the run reset to the owner latch: a stop protocol still running on
                # gl-job-stop (the previous job's, or this job's own) never interleaves with them.
                with self._stop_lock_for(job):
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
                # the kind's shared code may attach the API client log handler itself (the headers job's
                # translate_headers_now, to its view): a second one would write every record twice
                client_logs = None if getattr(kind, "own_client_logs", False) else getattr(self.backend, "client_logs", None)
                with (client_logs(host) if callable(client_logs) else contextlib.nullcontext()), \
                        self._watching(job, host), self._draining(job):
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
            self._release_pristine(job)
            if job.instance < 2 and job.stop_mode is not None and self._others_running(job):
                _release_base_stop_flags()  # the on-device endpoint must not refuse the other instance's requests
            with self._lock:
                job.owner = None
            if config_taken and self.config_store is not None:
                self._note_job_running(False)
        self._finish(job, result, error)

    def _finish(self, job: _Job, result: Any, error: Optional[str]) -> None:
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
            self._bring_back_locked(job.id)
            self._append_history_locked(snap)
            self._recent.insert(0, job)
            del self._recent[JOB_BUFFERS_KEPT:]
            self._active = None if self._active is job else self._active
            if self._iactive.get(job.instance) is job:
                del self._iactive[job.instance]
            self._write_active_locked()
        self._job_log(job, f"■ {snap.state_label}: {job.spec.title}" + (f" — {snap.error}" if snap.error else ""), {})
        self._close_log(job)
        self._post(self._deliver_transition, snap, previous)
        self._drop_superseded()
        # a job that only names its workspace (Retranslate Selected) changed the book's marks too
        params = snap.spec.params if isinstance(snap.spec.params, Mapping) else {}
        folder = _only_output_dir(snap) or params.get("output_dir")
        if folder:
            self.refresh_finished_progress(folder=str(folder))

    def refresh_finished_progress(self, job_id: Optional[str] = None, *, folder: Optional[str] = None) -> bool:
        """Blocking: finished jobs show how their book stands now. A QA scan, a retranslation or a Progress manager
        edit changes the chapters' marks in the book's progress file, and the counts a job saved when it ended (its
        page's "N QA failed", its progress line, its Jobs row) do not follow. ``job_id``: that job; ``folder``: the
        jobs of that workspace; neither (the Jobs page shows): every job whose progress file changed since the last
        such read. True when a count changed."""
        reader = getattr(self.backend, "progress_summary", None)
        if not callable(reader):
            return False
        wanted = self._folder_key(folder) if folder else None
        targets: dict = {}
        with self._lock:
            for snap in self._history + self._carried_on:
                own = _only_output_dir(snap)
                if (own and snap.progress.total and (job_id is None or snap.id == job_id)
                        and (wanted is None or self._folder_key(own) == wanted)):
                    targets[snap.id] = os.path.join(own, "translation_progress.json")
        paths = set(targets.values())
        if job_id is None and wanted is None:
            paths = {path for path in paths if self._summary_file_changed(path)}
        summaries = {path: reader(path) for path in paths}

        def counts_of(snap: JobSnapshot) -> Optional[Mapping[str, Any]]:
            summary = summaries.get(targets.get(snap.id))
            return summary if isinstance(summary, Mapping) and summary.get("total") else None

        with self._lock:
            changed = self._update_counts_locked(counts_of)
            if changed or self._counts_unsaved:
                self._write_counts_locked()
        if changed:
            self._changed()
        return changed

    def _folder_key(self, path: Any) -> str:
        """``target_key``, once per path: Android names one folder /data/user/0/... and /data/data/..., and a progress
        event compares it for every ended entry, under the lock."""
        key = self._folder_keys.get(path)
        if key is None:
            key = self._folder_keys[path] = target_key(path)
        return key

    def _summary_file_changed(self, path: str) -> bool:
        try:
            stat = os.stat(path)
            seen = (stat.st_mtime_ns, stat.st_size)
        except OSError:
            seen = None
        changed = self._summaries_seen.get(path) != seen
        self._summaries_seen[path] = seen
        return changed and seen is not None

    def _update_counts_locked(self, counts_of: Callable[[JobSnapshot], Optional[Mapping[str, Any]]]) -> bool:
        """The ended entries take the counts ``counts_of`` gives them; True when one changed."""
        changed = False
        for entries in (self._history, self._carried_on):
            for index, snap in enumerate(entries):
                counts = counts_of(snap)
                if not counts:
                    continue
                progress = Progress.from_dict({**snap.progress.to_dict(), **{
                    key: counts[key] for key in ("total", "completed", "in_progress", "failed") if key in counts}})
                if progress != snap.progress:
                    entries[index] = replace(snap, progress=progress)
                    recent = next((job for job in self._recent if job.id == snap.id), None)
                    if recent is not None:  # ``snapshot(job_id)`` answers from it
                        recent.progress = progress
                    changed = True
        return changed

    def _write_counts_locked(self) -> None:
        self._counts_unsaved = False
        self._counts_saved_at = self.clock()
        self._write_history_locked()

    def _follow_progress(self, data: Mapping[str, Any]) -> None:
        """Owner 2026-11-01 ("The QA failed labels in jobs doesn't update when said failed chapters are in progress"):
        a running job's progress event is the summary of its book's progress file, so the book's ended entries take
        the same counts at once: a chapter the job took up is in progress, not QA failed. Saved at the checkpoint
        rate; the job's end saves the last ones (``refresh_finished_progress``)."""
        path = data.get("path")
        if not path or not data.get("total"):
            return  # not a progress file's summary (a manga run's image counts), or one with no chapters yet
        wanted = self._folder_key(os.path.dirname(os.fspath(path)))

        def counts_of(snap: JobSnapshot) -> Optional[Mapping[str, Any]]:
            own = _only_output_dir(snap)
            return data if own and snap.progress.total and self._folder_key(own) == wanted else None

        with self._lock:
            if self._update_counts_locked(counts_of):
                self._counts_unsaved = True
            if self._counts_unsaved and self.clock() - self._counts_saved_at >= self.checkpoint_interval:
                self._write_counts_locked()

    # ---- events from the job (any thread) ------------------------------------------------------------------------

    def _job_log(self, job: _Job, text: Any, kwargs: Mapping[str, Any]) -> None:
        raw = "" if text is None else str(text)
        # Whole-message listeners first (desktop ``append_log``: ``_extra_log_listeners`` get every
        # raw message before any GUI-side suppression or filtering).
        self._deliver_message(job, raw)
        # The request stream gets every message whole, blank ones included, like the desktop
        # Direct Text listener (``append_log`` -> ``_on_log_line(message)``): the classifier
        # splits it itself and keeps blank lines inside streamed text (paragraph breaks).
        # Its "suppress-main-log" answer marks the whole message as streamed text. The desktop hides
        # that from its log for the Direct Text dialog; here Logs shows it for every provider (owner),
        # and only the job line and the issue chip skip it.
        streamed_text = False
        stream = job.stream
        if stream is not None:
            try:
                streamed_text = stream.feed(raw) == "suppress-main-log"
            except Exception:
                log.debug("request stream rejected a message", exc_info=True)
        # hotfix 3: the notification's streaming rows read the end of the streamed text (Streaming off: none)
        self._note_stream_phase(job, raw)
        # hotfix 4: the stream's own hint also marks some routes' status lines ("⏳ AuthND: NVIDIA queue /
        # prefill …", owner: "the notification area just shows random logs"), so both paths skip status lines
        if (streamed_text or self._streamed_text(job, raw)) and not stream_status_line(raw, job.stream):
            job.stream_tail = (job.stream_tail + raw)[-STREAM_TAIL_KEEP:]
        elif any(marker in raw for marker in STREAM_TAIL_RESETS):
            job.stream_tail = ""
        if not raw.strip():
            return
        kind = kwargs.get("kind") if isinstance(kwargs, Mapping) else None
        if kind not in ("info", "error", "thinking", "api"):
            kind = None
        for line in raw.replace("\r\n", "\n").split("\n"):
            if not line.strip() or line.lstrip().startswith(STREAM_MACHINE_MARKERS):
                continue
            self._write_log_file(job, line)
            provider = (sign_in_provider(line)
                        if not job.sign_in_flagged and sign_in_allowed(job.spec.kind) else None)
            if provider is not None:
                job.sign_in_flagged = True
                self._post(self._deliver_event, job.id, "sign_in_required",
                           sign_in_event(self._job_model(job), provider, line))
            buffer = job.buffer
            if buffer is not None:
                buffer.append(line, kind)
            if streamed_text:
                continue
            job.last_line = line[:LAST_LINE_CHARS]
            self._note_issue(job, line)

    @staticmethod
    def _note_stream_phase(job: _Job, raw: str) -> None:
        """Hotfix 4 (owner: thinking should have a brain emoji in the notification): the client's markers
        ("🧠 [x] Thinking...", then "Thinking complete" / "📡 Text streaming...") switch the streamed text
        between thinking and answer; a switch starts the tail anew, so the two never share a row."""
        low = raw.strip().lower()
        if not low:
            return
        if raw.strip().startswith("🧠") and low.endswith("thinking..."):
            thinking = True
        elif any(phrase in low for phrase in STREAM_TEXT_PHRASES) or any(p in low for p in STREAM_OPEN_PHRASES):
            thinking = False
        else:
            return
        if thinking != job.stream_thinking:
            job.stream_thinking = thinking
            job.stream_tail = ""

    @staticmethod
    def _streamed_text(job: _Job, raw: str) -> bool:
        """Hotfix 4: whether ``raw`` is text of an open request stream (``STREAM_OPEN_PHRASES`` ..
        ``STREAM_CLOSE_PHRASES``) rather than a status line; opening / closing updates ``job.stream_open``."""
        low = raw.strip().lower()
        if any(phrase in low for phrase in STREAM_OPEN_PHRASES):
            job.stream_open = True
            return False
        if any(phrase in low for phrase in STREAM_CLOSE_PHRASES):
            job.stream_open = False
            return False
        return job.stream_open and not stream_status_line(raw, job.stream)

    @staticmethod
    def _job_model(job: _Job) -> Optional[str]:
        """The model the job runs (its owner's config, else the spec's overrides)."""
        config = getattr(job.owner, "config", None)
        if isinstance(config, Mapping) and config.get("model"):
            return str(config.get("model"))
        params = job.spec.params or {}
        model = params.get("model") or (params.get("config_overrides") or {}).get("model")
        return str(model) if model else None

    def _note_issue(self, job: _Job, line: str) -> None:
        """A rate-limit / key-cooldown / network wait line sets the job's issue chip (UI_SPEC §2.12.3); a
        ⏳ Request Queue wait line sets "⏳ Request Queue · until 13:00" for the time left of the wait."""
        queued = request_queue_issue(line)
        if queued is not None:
            left, starts = queued
            now = self.clock()
            with self._lock:
                if job.stop_mode is not None:
                    return  # a Stop ended the wait
                job.issue = {"kind": REQUEST_QUEUE_ISSUE, "seconds": left, "since": now,
                             "until": now + max(60.0, float(left)), "starts": starts, "line": line[:200]}
            self._changed()
            return
        found = classify_issue(line)
        if found is None:
            return
        issue_kind, seconds = found
        now = self.clock()
        if issue_kind == "safety_block":
            # kept on the job's result (persisted): the chat shows the "Blocked by the provider's
            # safety filter" ErrorCard after the run (UI_SPEC §2.13)
            with self._lock:
                job.result.setdefault("safety_block", line[:300])
            return
        wait = float(seconds) if seconds else ISSUE_DEFAULT_SECONDS
        with self._lock:
            job.issue = {"kind": issue_kind, "seconds": seconds, "since": now, "until": now + max(5.0, wait),
                         "line": line[:200]}
        self._changed()

    @staticmethod
    def _deliver_message(job: _Job, raw: str) -> None:
        # Under the job's message lock, so a listener that attaches meanwhile sees the backlog and
        # then every later message exactly once, in order (callbacks must be quick: queue + post).
        with job.message_lock:
            job.messages.append(raw)
            for callback in list(job.message_listeners):
                try:
                    callback(raw)
                except Exception:
                    log.debug("job log listener failed", exc_info=True)

    def add_log_listener(self, job_id: str, callback: Callable[[str], Any], *,
                         backlog: bool = True) -> Optional[Callable[[], None]]:
        """Desktop ``add_log_listener`` for one job: ``callback(message)`` gets every whole log
        message of ``job_id`` (blank ones, and blank lines inside streamed text, included) on the
        thread that logged it; with ``backlog`` the messages logged before are replayed first.

        Returns the remover (``remove_log_listener``), or None when the job is unknown. The
        callback must be thread-safe and quick (it runs under the job's message lock)."""
        with self._lock:
            job = self._find_live(job_id)
        if job is None:
            return None
        with job.message_lock:
            if backlog:
                for message in list(job.messages):
                    try:
                        callback(message)
                    except Exception:
                        log.debug("job log listener failed on the backlog", exc_info=True)
            if callback not in job.message_listeners:
                job.message_listeners.append(callback)

        def remove() -> None:
            self.remove_log_listener(job_id, callback, job=job)

        return remove

    def remove_log_listener(self, job_id: str, callback: Callable[[str], Any], *, job: Optional[_Job] = None) -> None:
        """Unregister a listener added via ``add_log_listener``."""
        if job is None:
            with self._lock:
                job = self._find_live(job_id)
        if job is None:
            return
        with job.message_lock:
            if callback in job.message_listeners:
                job.message_listeners.remove(callback)

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
        if kind in ("progress", "output_dirs", "output_dir") and self._take_translated_title(job):
            checkpoint_due = True
        if kind == "progress":
            self._follow_progress(data)
        elif kind in ("output_dirs", "output_dir"):
            self._drop_superseded()  # the job now names its workspace: entries known only by that folder
        if checkpoint_due:
            self._checkpoint(job)
        if not consumed:
            payload = dict(data)
            self._post(self._deliver_event, job.id, kind, payload)
        self._changed()

    def _take_translated_title(self, job: _Job) -> bool:
        """Owner 2026-10-29: a book job still named after its file ("[213524] 히어로….epub", a chat's
        attachment) takes its translated title once the metadata translation wrote it (the output folder's
        metadata.json: ``title`` with ``title_translated``). A Library book's job keeps the book's name.
        True when renamed."""
        folder = job.output_dir
        if not folder or os.path.splitext(job.spec.title)[1].lower() not in FILE_TITLE_EXTENSIONS:
            return False
        path = os.path.join(folder, "metadata.json")
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            return False
        if mtime == job.metadata_mtime:
            return False
        job.metadata_mtime = mtime
        title = translated_book_title(path)
        if not title or title == job.spec.title:
            return False
        with self._lock:
            job.spec = replace(job.spec, title=title)
        log.info("job %s renamed after its metadata translation: %s", job.id, title)
        return True

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
        """Last lines of ``<logs>/jobs/<id>.log`` (finished jobs whose buffer is gone). A file from before
        2026-11-03 still holds the stream's machine lines, which are skipped here."""
        path = self.job_log_path(job_id)
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as handle:
                lines = handle.read().splitlines()
        except OSError:
            return []
        return [line for line in lines if not line.lstrip().startswith(STREAM_MACHINE_MARKERS)][-max_lines:]

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
        payload = {
            "version": STATE_VERSION,
            "saved_at": self.clock(),
            "active": active,
            "queue": [job.snapshot().to_dict() for job in self._queue],
        }
        # multi-instance: instance N's running and queued jobs (only while it has some, so a session of one
        # instance writes the file it always wrote)
        instances = {}
        for number in sorted(set(self._iactive) | set(self._iqueues)):
            running, queued = self._iactive.get(number), self._iqueues.get(number) or []
            if running is not None or queued:
                instances[str(number)] = {"active": running.snapshot().to_dict() if running is not None else None,
                                          "queue": [job.snapshot().to_dict() for job in queued]}
        if instances:
            payload["instances"] = instances
        return payload

    def _write_active_locked(self) -> None:
        payload = self._active_payload_locked()
        path = self.active_state_path
        try:
            if payload["active"] is None and not payload["queue"] and not payload.get("instances"):
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
                                                   "jobs": [s.to_dict() for s in self._history],
                                                   "carried_on": [s.to_dict() for s in self._carried_on],
                                                   "carried_by": self._carried_by})
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
        def load(path: str, key: str = "jobs") -> list[JobSnapshot]:
            data = _read_json(path)
            jobs = data.get(key) if isinstance(data, dict) else None
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
            self._carried_on = load(self.history_state_path, "carried_on")[: self.history_limit]
            data = _read_json(self.history_state_path)
            by = data.get("carried_by") if isinstance(data, dict) else None
            self._carried_by = {job_id: who for job_id, who in by.items() if isinstance(who, dict)} if isinstance(
                by, dict) else {}
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
        instances = data.get("instances")
        for _number, part in sorted(instances.items()) if isinstance(instances, dict) else ():
            if isinstance(part, dict):  # every instance's running job, then its queue (Resume keeps the instance)
                if isinstance(part.get("active"), dict):
                    entries.append(part["active"])
                entries.extend(e for e in part.get("queue") or () if isinstance(e, dict))
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
        """Block until nothing runs and nothing (unpaused) is queued, in any instance (tests, shutdown)."""
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cond:
            while self._active is not None or self._iactive or (self._queued_jobs_locked() and not self._paused):
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return False
                self._cond.wait(remaining if remaining is not None else 0.5)
        return True

    def close(self, timeout: float = 2.0) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()
            threads = [self._thread, *self._ithreads.values()]
        for thread in threads:
            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout)
