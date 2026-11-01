"""ChatRuns: one chat send from the composer to the finished transcript (pure Python, no Flet).

Desktop flow (``_InputOutputDialog``): ``_on_enter_clicked`` -> ``_start_translation``
(record the turn, prepare the temp input, apply the run options, start
``run_translation_thread``) -> the log listener streams request cards -> the
glossary gate asks Edit / Yes / No -> ``_finish_translation`` (final drain, read the
translated file, persist the run tree into the chat folder, commit the cards and the
"Extraction report" / "Attachment actions" cards, save) -> ``_restore_run_context``.

Mobile splits it across threads and JobService; every Direct Text rule runs in the
shared code the dialog inherits (``direct_text_store`` / ``direct_text_stream``):

* ``send()`` (UI loop, file work in ``run_io``): records the user turn through the
  ``ChatStoreAdapter``, prepares the run (``run_request.prepare_direct_text_run`` =
  ``direct_text_store.prepare_direct_text_input``) and submits a ``direct_text`` job;
  the job adapter applies the options/env and runs the shared pipeline on the
  ``gl-job`` thread, and JobService feeds every raw log line into the job's
  ``direct_text_stream.DirectTextStream`` (``RunStream`` reads it).
* ``on_snapshot()`` (UI loop): job state -> run state; a pending
  ``direct_text_glossary_approval`` question freezes the cards into the chat
  (``ChatStore.commit_request_phase`` = ``_commit_active_request_phase``) and shows the
  approval card; a terminal state schedules ``finish()``.
* ``generate()`` (U7, UI_SPEC §2.6 "Generate from prompt (no input)"): the same run for the
  desktop generative-only path - the composer text is recorded as ``["user", prompt]`` and a
  ``generate_media`` job runs the shared ``_run_generative_prompt_mode`` with it as the prompt;
  the generated image / video / audio lands in the run root and ``finish_run`` promotes it into
  the chat folder like any Direct Text media response.
* ``finish()`` (worker thread): ``ChatStore.finish_run`` (the dialog's
  ``_finish_translation``: final drain, translated file, run tree persisted into the
  chat folder, cards + "Extraction report" / "Attachment actions" committed, history
  saved). A run that was stopped or failed keeps its run root (Resume / Retry continue
  from ``translation_progress.json``; the dialog deletes its temp root - recorded
  mobile divergence); a finished one is cleaned like the dialog's.

Runs are keyed by chat id; a resumed job (same params after a kill) is re-attached
from its ``params["run"]``. Listeners (``subscribe``) are told the chat id that changed;
the chat feature marshals them to the UI loop. ``subscribe_finished`` listeners get
``(cid, run, job state)`` on the finish thread once the run is committed and saved (the chat
feature moves a finished book into the Library from there, UI_SPEC §2.17).

A job belongs to the user turn at its submitted ``params["user_index"]`` until a message
before it is deleted: Delete message records the turn of every job the chat knows
(``chat_job_turns``) by fingerprint under the run's key (``turn_key``: its run root, shared
by Resume / Retry of the same run), and ``turn_index`` / ``persisted_job`` / ``resubmit``
resolve turns through the recorded ``ChatStoreAdapter.job_turns``.

Job safety (hotfix 1 issue 5): a chat has one run at a time and a file one job. ``send`` /
``resubmit`` / ``generate`` while the chat's run is live (queued, running, at the glossary gate,
refining, stopping, finishing) return that run; a second send of the same chat while the first
one prepares is a no-op (None). An attachment send names its file (``JOB_TARGET_PARAM``) and,
before it records anything, binds to this chat's active job for that file or refuses with
``JobAlreadyActive`` when another chat or the Library already runs it (``JobService.
active_job_for``); a send of the same file from another chat is refused while this one
prepares. ``JobService.submit_job`` decides last, under its lock: when it reuses a job anyway,
the prepared run is dropped and the turn gets the "could not be started" card saying where the
book runs. While the gate's glossary is refined (``JobSnapshot.refining`` -> ``ChatRun.refining``)
``answer_glossary`` refuses Yes / No (JobService refuses them too), so the translation never
starts before the refinement ended (issue 2).
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Mapping, Optional

from glossarion_mobile.services.jobs import (  # contract C1
    GLOSSARY_QUESTION_KINDS,
    GLOSSARY_REFINING_TEXT,
    JOB_TARGET_PARAM,
    JobAlreadyActive,
    is_glossary_question,
    target_key,
)
from glossarion_mobile.ui.chat.direct_text_rules import DirectTextSettings, ManualGlossarySource
from glossarion_mobile.ui.chat.job_binding import (
    RUNNING_STATES,
    TERMINAL_STATES,
    JobsAdapter,
    chat_id_of,
    ended_kind,
    job_kind,
    progress_counts,
    question_of,
    state_name,
)
from glossarion_mobile.ui.chat.run_request import (
    DIRECT_TEXT_JOB_KIND,
    GENERATE_MEDIA_JOB_KIND,
    GENERATIVE_SENTINEL,
    DirectTextRun,
    job_params,
    job_title,
    prepare_direct_text_run,
    replace_profile,
    user_turn,
    with_chat_overrides,
    with_current_glossary,
    with_run_config,
)
from glossarion_mobile.ui.chat.stream_bridge import RunStream

__all__ = ["CHAT_JOB_KINDS", "ChatRun", "ChatRuns", "GLOSSARY_QUESTION_KINDS", "JobAlreadyActive", "RESUMABLE_ENDINGS",
           "STATUS", "attachment_target", "is_glossary_question", "turn_key"]

log = logging.getLogger("glossarion.chat.runs")

#: Desktop footer strings (UI_SPEC §2.3 status caption; "Click" -> "Tap").
STATUS = {
    "translating": "Translating…",
    "starting": "Starting translation…",
    "glossary": "Glossary ready — choose Edit, Yes, or No",
    "finishing": "Finishing current request… Tap again to force stop",
    "force": "Force stopping…",
    "stopping": "Stopping…",
    "stopped": "Stopped",
    "no_output": "No translated output was produced",
    "streamed": "Run ended; showing streamed output",
    "could_not_start": "Could not start",
    "manual_glossary": "Manual glossary required",
    "ready": "Ready",
    "refining": GLOSSARY_REFINING_TEXT,  # hotfix 1 issue 2: the gate's glossary is being refined
}

#: Hotfix 1 issue 5: a send of a file another chat is starting a run for at this moment
STARTING_ELSEWHERE = "“{name}” is already starting in another chat"

#: The glossary gate's question kinds and predicate live in ``services.jobs`` (contract C1); the old
#: names stay importable from here.
_GLOSSARY_QUESTION_KINDS = GLOSSARY_QUESTION_KINDS
#: JobService kinds a chat run can be (Direct Text send, generative-only run).
CHAT_JOB_KINDS = (DIRECT_TEXT_JOB_KIND, GENERATE_MEDIA_JOB_KIND)
#: ``last_job_ending`` values whose job card still offers Resume (running, stopped, failed,
#: interrupted, discarded) or Retry failed (done with failed chapters, "issues"), UI_SPEC §2.12.4.
RESUMABLE_ENDINGS = ("live", "stopped", "failed", "interrupted", "discarded", "issues")


def _now_iso() -> str:
    """Desktop ``_direct_response_timestamp`` format (timezone-aware, seconds)."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def attachment_target(attachment: Optional[Mapping[str, Any]]) -> str:
    """The file an attachment send works on, for the duplicate check (``JOB_TARGET_PARAM``): the attached
    file's absolute path; '' for typed text and for a Vision attachment (an image / image archive the chat
    asks about - two questions about one picture are two requests)."""
    path = str((attachment or {}).get("path") or "") if attachment else ""
    if not path:
        return ""
    try:
        from glossarion_mobile.ui.chat.output_modes import is_vision_attachment

        if is_vision_attachment(path):
            return ""
    except Exception:
        pass
    return os.path.abspath(path)


def turn_key(params: Any, job_id: Any = None) -> Optional[str]:
    """The key a chat job's turn is recorded under: its run root (``params["run"]["temp_root"]``,
    the same for every Resume / Retry of that run), else its job id."""
    run = params.get("run") if isinstance(params, Mapping) else None
    root = str(run.get("temp_root") or "") if isinstance(run, Mapping) else ""
    if root:
        return "run:" + root
    return f"job:{job_id}" if job_id is not None else None


def _rechecked_profile(params: dict, check: Optional[Callable[[str], Optional[str]]]) -> dict:
    """``params`` with its Send-time prompt profile (``config_overrides["active_profile"]``) re-checked:
    ``check(name)`` answers the profile to run with (the same name while it exists) or None (the run
    inherits). A changed answer is written through ``run_request.replace_profile`` (the extraction method
    the profile switches follows it, like at Send)."""
    overrides = params.get("config_overrides")
    name = overrides.get("active_profile") if isinstance(overrides, Mapping) else None
    if not callable(check) or not isinstance(name, str) or not name:
        return params
    try:
        profile = check(name)
    except Exception:
        log.exception("re-checking the resumed run's prompt profile failed")
        return params
    if profile == name:
        return params
    log.info("resumed run: prompt profile %r no longer exists, running with %r", name, profile or "(inherited)")
    params = dict(params)
    params["config_overrides"] = replace_profile(overrides, profile or None)
    return params


def _snapshot_time(snapshot: Any) -> float:
    """When a remembered job last changed (finished, else started, else created)."""
    for name in ("finished", "started", "created"):
        value = getattr(snapshot, name, None)
        if isinstance(value, (int, float)) and value:
            return float(value)
    return 0.0


def _finish_run_root(run_state: Mapping[str, Any], result: Any) -> None:
    """A finished chat run's root goes away like ``finish_run(cleanup=True)`` removes it, after its glossary
    folder (``<run root>/Glossary/<book>/``: the glossary, its progress and its EPUB text cache) was kept
    inside the attachment workspace (``workspace_sync.stash_run_glossary``), which carries it into the
    Library with the book: a later Continue reuses the glossary progress and caches (issues 19 / 20)."""
    temp_root = str(run_state.get("temp_root") or "")
    if not temp_root:
        return
    result = result if isinstance(result, Mapping) else {}
    if result.get("preserve_temp_root"):
        return  # finish_run keeps this run root (its response links point into it)
    workspace = str(result.get("output_folder") or "")
    if run_state.get("is_attachment") and workspace and os.path.isdir(workspace):
        book = os.path.splitext(os.path.basename(str(run_state.get("source_path") or "")))[0]
        try:
            from glossarion_mobile.services.workspace_sync import stash_run_glossary

            if book and not _inside_folder(workspace, temp_root):
                stash_run_glossary(temp_root, book, workspace)
        except Exception:
            log.exception("keeping the run's glossary folder failed")
    shutil.rmtree(temp_root, ignore_errors=True)


def _inside_folder(path: str, root: str) -> bool:
    if not path or not root:
        return False
    try:
        a = os.path.normcase(os.path.abspath(path))
        b = os.path.normcase(os.path.abspath(root))
        return os.path.commonpath([a, b]) == b
    except ValueError:
        return False


def _streamed_text(model: Any, run_state: Mapping) -> bool:
    """The run streamed reply text (the stream's content or any request segment's)."""
    if str(run_state.get("streamed_content") or "").strip():
        return True
    if model is None:
        return False
    if str(getattr(model, "_streamed_content", "") or "").strip():
        return True
    segments = getattr(model, "_active_request_segments", None)
    if segments is None and isinstance(model, list):
        segments = model
    return any(str((s or {}).get("content") or "").strip() for s in (segments or ()) if isinstance(s, Mapping))


@dataclass
class ChatRun:
    cid: str
    run: DirectTextRun
    stream: RunStream
    user_index: int
    job_id: Any = None
    state: str = "preparing"  # preparing | queued | running | stopping | force_stopping | finishing | done | failed | stopped
    job_state: str = ""
    question: Optional[dict] = None
    stop_requested: bool = False
    finished: bool = False
    error: Optional[str] = None
    output_folder: str = ""
    output_dir: str = ""  # the pipeline's output folder (JobSnapshot.output_dir: translation_progress.json lives here)
    status: str = STATUS["translating"]
    last_snapshot: Any = None
    params: dict = field(default_factory=dict)  # the JobSpec params (Resume / Retry failed resubmit them)
    title: str = ""
    started: float = field(default_factory=time.time)
    kind: str = DIRECT_TEXT_JOB_KIND  # the JobSpec kind (Resume resubmits the same kind)
    #: The chat's message count when the run started: the request cards THIS run commits (the glossary
    #: gate's, the finish's) are the chat's messages from here on; the turn's earlier cards (a stopped
    #: run's, before a Resume) are not (``ChatView._update_live_job_card``).
    committed_from: int = 0
    #: Hotfix 1 issue 2: the gate's glossary is being refined (``JobSnapshot.refining``, ✨ Refine)
    refining: bool = False

    @property
    def live(self) -> bool:
        return not self.finished

    @property
    def awaiting_glossary(self) -> bool:
        """A glossary gate question waits for this run, also while its glossary is being refined (then
        ``refining`` is True and ``answer_glossary`` refuses Yes / No until the refinement ended)."""
        return bool(self.question) and is_glossary_question(self.question.get("kind"))

    @property
    def gate_open(self) -> bool:
        """The gate takes the user's Yes / No now: a glossary question waits and is not being refined."""
        return self.awaiting_glossary and not self.refining


class ChatRuns:
    def __init__(
        self,
        store: Any,  # ChatStoreAdapter
        jobs: JobsAdapter,
        *,
        run_io: Optional[Callable[..., Any]] = None,  # async run_io(fn, *args)
        temp_dir: Optional[str] = None,
        model_name: Callable[[], Optional[str]] = lambda: None,
    ) -> None:
        self.store = store
        self.jobs = jobs
        self.run_io = run_io
        self.temp_dir = temp_dir
        self.model_name = model_name
        self._lock = threading.RLock()
        self.runs: dict = {}  # cid -> ChatRun (live or the last finished one)
        self.status: dict = {}  # cid -> last status caption when idle
        self._listeners: list = []
        self._finished_listeners: list = []
        self._job_unsub: Optional[Callable[[], None]] = None
        self.finish_threads: list = []
        #: Hotfix 1 issue 5: chats whose send / resubmit is preparing (cid -> the file it works on, '' for
        #: none): a second send of that chat, or another chat's send of the same file, cannot start a job
        self._starting: dict = {}
        #: chats whose last send / resubmit queued nothing because the app's preflight refused it (the app said
        #: why, e.g. "… · Choose model"): ``take_refused`` tells that apart from "nothing to resume"
        self._refused: set = set()

    def take_refused(self, cid: Any) -> bool:
        """The chat's last send / resubmit was refused by the app's preflight, which already told the user why
        (cleared by this call)."""
        with self._lock:
            if str(cid) in self._refused:
                self._refused.discard(str(cid))
                return True
            return False

    def _mark_refused(self, cid: str) -> None:
        with self._lock:
            self._refused.add(str(cid))

    # ---- listeners ------------------------------------------------------------------

    @staticmethod
    def _add_listener(listeners: list, callback: Callable[..., Any]) -> Callable[[], None]:
        listeners.append(callback)

        def unsubscribe() -> None:
            try:
                listeners.remove(callback)
            except ValueError:
                pass

        return unsubscribe

    def subscribe(self, callback: Callable[[str], Any]) -> Callable[[], None]:
        return self._add_listener(self._listeners, callback)

    def subscribe_finished(self, callback: Callable[[str, "ChatRun", str], Any]) -> Callable[[], None]:
        """``callback(cid, run, state)`` once a run is committed: on the finish thread, after the
        chat is saved and marked idle (``state``: the job's state name, ``"DONE"``, ``"FAILED"``,
        ``"CANCELLED"``, ...). Returns the unsubscribe function."""
        return self._add_listener(self._finished_listeners, callback)

    def _emit_finished(self, run: "ChatRun", state: str) -> None:
        for callback in list(self._finished_listeners):
            try:
                callback(run.cid, run, state)
            except Exception:
                log.exception("chat run finished listener failed")

    def _emit(self, cid: str) -> None:
        for callback in list(self._listeners):
            try:
                callback(cid)
            except Exception:
                log.exception("chat run listener failed")

    def attach(self) -> None:
        """Follow JobService (state changes, questions, progress)."""
        if self._job_unsub is None and self.jobs.available:
            self._job_unsub = self.jobs.subscribe(self.on_snapshot)
            service = self.jobs.jobs
            if hasattr(service, "resume_params") and getattr(service, "resume_params", None) is None:
                service.resume_params = self._resume_params  # Jobs page / launch banner Resume

    def _resume_params(self, spec: Any) -> Optional[dict]:
        """``JobService.resume_params``: a chat job resumed outside the chat runs with the chat's model now and
        with the Run options its stopped card shows by default ("Only for this run" off: the settings now, never
        the Send-time values the Jobs page cannot show)."""
        origin = getattr(spec, "origin", None) or {}
        cid = origin.get("cid") if origin.get("type") == "chat" else None
        if cid is None or getattr(spec, "kind", None) not in CHAT_JOB_KINDS or self.store.session(cid) is None:
            return None
        params = dict(getattr(spec, "params", None) or {})
        if getattr(spec, "kind", None) == DIRECT_TEXT_JOB_KIND and isinstance(params.get("config_overrides"), Mapping):
            params["config_overrides"] = with_run_config(params["config_overrides"], {})
        return self.with_current_model(cid, params)

    def detach(self) -> None:
        if self._job_unsub is not None:
            try:
                self._job_unsub()
            except Exception:
                pass
            self._job_unsub = None
            service = self.jobs.jobs
            if getattr(service, "resume_params", None) == self._resume_params:
                service.resume_params = None

    # ---- queries ----------------------------------------------------------------------------

    def run_for(self, cid: Any) -> Optional[ChatRun]:
        with self._lock:
            return self.runs.get(str(cid))

    def live_run(self, cid: Any) -> Optional[ChatRun]:
        run = self.run_for(cid)
        return run if run is not None and run.live else None

    def own_job_state(self, cid: Any) -> Optional[str]:
        """The SendInputs ``own_job_state`` for this chat (JobState name) or None."""
        run = self.live_run(cid)
        if run is None:
            return None
        if run.state in ("preparing", "queued", "finishing"):
            return "RUNNING"  # Stop cancels a queued run; finishing is a short commit
        return {"running": "RUNNING", "stopping": "STOPPING", "force_stopping": "FORCE_STOPPING"}.get(run.state)

    def queued(self, cid: Any) -> bool:
        run = self.live_run(cid)
        return run is not None and run.state == "queued"

    def caption(self, cid: Any) -> Optional[str]:
        run = self.live_run(cid)
        if run is not None:
            if run.awaiting_glossary:
                return STATUS["refining"] if run.refining else STATUS["glossary"]
            return run.status
        text = self.status.get(str(cid))
        return None if not text or text == STATUS["ready"] else text

    # ---- send ------------------------------------------------------------------------------

    async def _io(self, fn: Callable[..., Any], *args: Any) -> Any:
        if self.run_io is None:
            return fn(*args)
        return await self.run_io(fn, *args)

    def _message_count(self, cid: str) -> int:
        """The chat's message count now (``ChatRun.committed_from``; 0 when the store cannot tell)."""
        reader = getattr(self.store, "messages", None)
        try:
            return len(reader(cid)) if callable(reader) else 0
        except Exception:
            return 0

    async def send(
        self,
        cid: Any,
        *,
        text: str,
        attachment: Optional[Mapping[str, Any]],
        settings: DirectTextSettings,
        output_mode: str,
        overrides: Optional[Mapping[str, Any]] = None,
        manual_glossary: Optional[ManualGlossarySource] = None,
        user_index: Optional[int] = None,
        config_extra: Optional[Mapping[str, Any]] = None,
    ) -> Optional[ChatRun]:
        """Record the turn (unless ``user_index`` points at an already recorded one, e.g. a
        Plan card), prepare the run off the loop and submit the ``direct_text`` job.

        ``config_extra``: config keys for this run only (the Plan card's "Only for this run" Run
        options), merged into the JobSpec ``config_overrides`` under the chat overrides.

        Hotfix 1 issue 5 (module docstring): returns this chat's live run when it has one and None while
        its previous send still prepares (nothing is recorded either way); raises ``JobAlreadyActive``
        when another chat or the Library already runs the attached file."""
        cid = str(cid)
        target = attachment_target(attachment)
        claimed = self._claim(cid, target)
        if claimed is not True:
            return claimed
        try:
            if target:
                active = self._active_job_for(DIRECT_TEXT_JOB_KIND, (target,),
                                              {JOB_TARGET_PARAM: [target], "is_attachment": True})
                if active is not None:
                    return self._bind_or_refuse(cid, active)
            return await self._send(cid, text=text, attachment=attachment, settings=settings,
                                    output_mode=output_mode, overrides=overrides, manual_glossary=manual_glossary,
                                    user_index=user_index, config_extra=config_extra, target=target)
        finally:
            self._release(cid)

    async def _send(
        self,
        cid: str,
        *,
        text: str,
        attachment: Optional[Mapping[str, Any]],
        settings: DirectTextSettings,
        output_mode: str,
        overrides: Optional[Mapping[str, Any]],
        manual_glossary: Optional[ManualGlossarySource],
        user_index: Optional[int],
        config_extra: Optional[Mapping[str, Any]],
        target: str,
    ) -> ChatRun:
        text = str(text or "").strip()
        role = settings.attachment_prompt_role
        if user_index is None:
            display_input = (attachment or {}).get("name") if attachment else text
            user_index = self.store.record_user_turn(cid, user_turn(text, attachment, role), display_input)
            self._emit(cid)  # show the turn right away (the run is prepared off the loop next)
        try:
            run = await self._io(
                lambda: prepare_direct_text_run(
                    text=text,
                    attachment=attachment,
                    output_mode=output_mode,
                    attachment_prompt_role=role,
                    manual_glossary=manual_glossary,
                    temp_dir=self.temp_dir,
                )
            )
        except Exception as exc:
            self._could_not_start(cid, exc)
            raise
        chat_run = ChatRun(cid=cid, run=run, stream=RunStream(auto_scroll_disabled=settings.disable_auto_scroll),
                           user_index=int(user_index), committed_from=self._message_count(cid))
        chat_run.stream.provider = self._stream_provider(chat_run)
        with self._lock:
            previous = self.runs.get(cid)
            self.runs[cid] = chat_run
            self.status.pop(cid, None)
        self.store.set_running(cid, True)
        await self._io(self.store.flush)  # history on disk before the job starts (desktop saves at send)
        params = job_params(chat_id=self._chat_id_value(cid), user_index=user_index, run=run, settings=settings, overrides=overrides)
        if target:
            params[JOB_TARGET_PARAM] = [target]  # the attached file, for JobService's duplicate check
        if config_extra:
            params["config_overrides"] = {**dict(config_extra), **dict(params.get("config_overrides") or {})}
        # The job's DirectTextStream starts at the chat's next request number (the dialog's
        # _active_request_next_number) and counts tokens for the run's model.
        params["request_number"] = self.store.request_count(cid) + 1
        model = (params.get("config_overrides") or {}).get("model") or self.model_name()
        if model:
            params["model"] = str(model)
        title = str((self.store.session(cid) or {}).get("title") or "Chat")
        origin = {"type": "chat", "cid": cid, "label": f"Chat · {title}"}
        try:
            job_id, reused = await self._submit_job(DIRECT_TEXT_JOB_KIND, job_title(run, title), (run.source_path,),
                                                    params, origin)
        except Exception as exc:
            with self._lock:
                chat_run.finished = True
            self.store.set_running(cid, False)
            self._could_not_start(cid, exc)
            raise
        if reused is not None:
            # another surface queued this file's job while the run was prepared: nothing new runs
            await self._drop_prepared(cid, chat_run, previous, remove_root=True)
            self._refuse_reused(cid, reused)
        if job_id is None:
            # not queued (the app's preflight refused the run and said why): the chat must not keep a run
            # without a job, which would hold its Send as Stop and bind every later send to it
            await self._drop_prepared(cid, chat_run, previous, remove_root=True)
            self._emit(cid)
            return None
        with self._lock:
            chat_run.job_id = job_id
            chat_run.params = dict(params)
            chat_run.title = job_title(run, title)
            if chat_run.state == "preparing":
                snapshot = self.jobs.snapshot()
                same = snapshot is not None and getattr(snapshot, "id", None) == job_id
                chat_run.state = "running" if same and state_name(snapshot) in RUNNING_STATES else "queued"
        self._emit(cid)
        return chat_run

    async def generate(
        self,
        cid: Any,
        *,
        prompt: str,
        output_mode: str,
        settings: DirectTextSettings,
        overrides: Optional[Mapping[str, Any]] = None,
    ) -> Optional[ChatRun]:
        """"Generate from prompt (no input)" (UI_SPEC §2.6): record ``["user", prompt]`` and submit a
        ``generate_media`` job (the desktop generative-only run, the composer text as the prompt).
        Hotfix 1 issue 5: this chat's live run is returned (None while its previous send prepares)."""
        cid = str(cid)
        prompt = str(prompt or "").strip()
        if not prompt:
            raise ValueError("Type a prompt in the composer first")
        claimed = self._claim(cid, "")
        if claimed is not True:
            return claimed
        try:
            return await self._generate(cid, prompt=prompt, output_mode=output_mode, settings=settings,
                                        overrides=overrides)
        finally:
            self._release(cid)

    async def _generate(self, cid: str, *, prompt: str, output_mode: str, settings: DirectTextSettings,
                        overrides: Optional[Mapping[str, Any]]) -> Optional[ChatRun]:
        user_index = self.store.record_user_turn(cid, user_turn(prompt, None, settings.attachment_prompt_role), prompt)
        self._emit(cid)
        try:
            run = await self._io(
                lambda: prepare_direct_text_run(text=prompt, attachment=None, output_mode=output_mode,
                                                attachment_prompt_role=settings.attachment_prompt_role,
                                                temp_dir=self.temp_dir)
            )
        except Exception as exc:
            self._could_not_start(cid, exc)
            raise
        chat_run = ChatRun(cid=cid, run=run, stream=RunStream(auto_scroll_disabled=settings.disable_auto_scroll),
                           user_index=int(user_index), kind=GENERATE_MEDIA_JOB_KIND,
                           committed_from=self._message_count(cid))
        chat_run.stream.provider = self._stream_provider(chat_run)
        with self._lock:
            previous = self.runs.get(cid)
            self.runs[cid] = chat_run
            self.status.pop(cid, None)
        self.store.set_running(cid, True)
        await self._io(self.store.flush)
        params = job_params(chat_id=self._chat_id_value(cid), user_index=user_index, run=run, settings=settings,
                            overrides=overrides)
        params["options"]["selected_files"] = [GENERATIVE_SENTINEL]
        params["options"]["force_no_glossary"] = True
        params["prompt"] = prompt
        params["request_number"] = self.store.request_count(cid) + 1
        model = (params.get("config_overrides") or {}).get("model") or self.model_name()
        if model:
            params["model"] = str(model)
        title = str((self.store.session(cid) or {}).get("title") or "Chat")
        try:
            job_id = await self.jobs.submit(GENERATE_MEDIA_JOB_KIND, title, (), params,
                                            {"type": "chat", "cid": cid, "label": f"Chat · {title}"})
        except Exception as exc:
            with self._lock:
                chat_run.finished = True
            self.store.set_running(cid, False)
            self._could_not_start(cid, exc)
            raise
        if job_id is None:  # not queued (the app's preflight said why): no run without a job
            await self._drop_prepared(cid, chat_run, previous, remove_root=True)
            self._emit(cid)
            return None
        with self._lock:
            chat_run.job_id = job_id
            chat_run.params = dict(params)
            chat_run.title = title
            if chat_run.state == "preparing":
                snapshot = self.jobs.snapshot()
                same = snapshot is not None and getattr(snapshot, "id", None) == job_id
                chat_run.state = "running" if same and state_name(snapshot) in RUNNING_STATES else "queued"
        self._emit(cid)
        return chat_run

    # ---- job safety (hotfix 1 issue 5) ----------------------------------------------------------

    def _claim(self, cid: str, target: str) -> Any:
        """Before a send / resubmit records anything: True when this chat may start a run now (claimed until
        ``_release``); else this chat's live run (a second Send / Start / Resume binds to it) or None (its
        previous send is still preparing: a double tap). Raises ``JobAlreadyActive`` while another chat
        prepares a send of the same file."""
        key = target_key(target) if target else ""
        with self._lock:
            live = self.runs.get(cid)
            if live is not None and live.live:
                return live
            if cid in self._starting:
                return None
            if key and any(other == key for other in self._starting.values()):
                raise JobAlreadyActive(None, STARTING_ELSEWHERE.format(name=os.path.basename(target)))
            self._starting[cid] = key
        return True

    def _release(self, cid: str) -> None:
        with self._lock:
            self._starting.pop(cid, None)

    def _service(self) -> Any:
        return getattr(self.jobs, "jobs", None)

    @staticmethod
    def _own(cid: str, snapshot: Any) -> bool:
        """``snapshot`` is a job of this chat (a chat job kind with its ``params["chat_id"]``)."""
        return job_kind(snapshot) in CHAT_JOB_KINDS and chat_id_of(snapshot) == str(cid)

    def _active_job_for(self, kind: str, inputs: tuple, params: Mapping[str, Any]) -> Any:
        """``JobService.active_job_for`` the job these parts make (None when the service cannot tell)."""
        service = self._service()
        finder = getattr(service, "active_job_for", None) if service is not None else None
        if not callable(finder):
            return None
        try:
            return finder(self.jobs.build_spec(kind, "", tuple(inputs), dict(params), {}))
        except Exception:
            log.debug("the active-job check failed", exc_info=True)
            return None

    async def _submit_job(self, kind: str, title: str, inputs: tuple, params: Mapping[str, Any],
                          origin: Mapping[str, Any]) -> tuple:
        """``(job id or None, the reused job's snapshot or None)``. The adapter's submit - the app's Send /
        Start path (``JobsFeature.submit``: the model preflight, the background preparation on the user's
        tap, then ``JobService.submit``) - and whether JobService answered with an active job that is not
        this submission (it reuses one that already does the same work, also one queued while the
        preparation awaited a permission dialog): the job's spec is not the one sent."""
        job_id = await self.jobs.submit(kind, title, inputs, params, origin)
        snapshot_of = getattr(self.jobs, "snapshot_of", None)
        build_spec = getattr(self.jobs, "build_spec", None)
        if job_id is None or not callable(snapshot_of) or not callable(build_spec):
            return job_id, None  # (a jobs object without per-job lookups cannot tell)
        snapshot = snapshot_of(job_id)
        spec = getattr(snapshot, "spec", None)
        try:
            ours = build_spec(kind, title, tuple(inputs), dict(params), dict(origin))
            same = spec is None or not hasattr(ours, "to_dict") or spec.to_dict() == ours.to_dict()
        except Exception:
            same = True
        return job_id, (None if same else snapshot)

    def _resume_job(self, job_id: Any, params: Mapping[str, Any]) -> tuple:
        """``(job id or None, the reused job's snapshot or None)``: ``JobService.resume_job`` when the service
        has it, else the adapter's resume."""
        service = self._service()
        resume_job = getattr(service, "resume_job", None) if service is not None else None
        if not callable(resume_job):
            return self.jobs.resume(job_id, params=params), None
        try:
            result = resume_job(str(job_id), params=dict(params))
        except Exception:
            log.exception("resuming job %s failed", job_id)
            return None, None
        if result is None:
            return None, None
        return result.job_id, (result.snapshot if result.reused else None)

    def _service_refining(self, job_id: Any) -> bool:
        """JobService says the job's gate glossary is being refined (``JobService.is_refining``)."""
        service = self._service()
        check = getattr(service, "is_refining", None) if service is not None else None
        if job_id is None or not callable(check):
            return False
        try:
            return bool(check(job_id))
        except Exception:
            return False

    def _bind(self, cid: str, snapshot: Any) -> Optional[ChatRun]:
        """This chat's active job: the chat's run follows it (re-attached when the chat did not track it)."""
        self.on_snapshot(snapshot)
        return self.live_run(cid)

    def _bind_or_refuse(self, cid: str, snapshot: Any) -> Optional[ChatRun]:
        if self._own(cid, snapshot):
            return self._bind(cid, snapshot)
        raise JobAlreadyActive(snapshot)

    async def _drop_prepared(self, cid: str, chat_run: ChatRun, previous: Optional[ChatRun], *,
                             remove_root: bool) -> None:
        """Nothing new was queued for ``chat_run``: the chat keeps its previous run; a send's fresh run root
        goes away (a resubmit's is the stopped run's own and stays)."""
        with self._lock:
            chat_run.finished = True
            chat_run.state = "stopped"
            if self.runs.get(cid) is chat_run:
                if previous is not None:
                    self.runs[cid] = previous
                else:
                    self.runs.pop(cid, None)
        self.store.set_running(cid, False)
        root = str(chat_run.run.temp_root or "") if remove_root else ""
        if root and os.path.isdir(root):
            try:
                await self._io(shutil.rmtree, root, True)
            except Exception:
                log.debug("removing the unused run root failed", exc_info=True)

    def _refuse_reused(self, cid: str, snapshot: Any) -> None:
        """A submission JobService answered with an active job: this chat's job is followed, the turn gets the
        "could not be started" card saying where the book runs, and ``JobAlreadyActive`` is raised."""
        if self._own(cid, snapshot):
            self._bind(cid, snapshot)
        exc = JobAlreadyActive(snapshot)
        self._could_not_start(cid, exc)
        raise exc

    def _store_for(self, cid: Any) -> Any:
        """The shared ChatStore that owns chat ``cid`` (a scratch chat has its own)."""
        binding_for = getattr(self.store, "binding_for", None)
        binding = binding_for(cid) if callable(binding_for) else getattr(self.store, "binding", None)
        return getattr(binding, "store", None)

    def _stream_provider(self, chat_run: ChatRun) -> Callable[[], Any]:
        """The job's ``direct_text_stream.DirectTextStream`` (JobService feeds it every raw line)."""
        return lambda: self.jobs.request_stream(chat_run.job_id)

    @staticmethod
    def _chat_id_value(cid: str) -> Any:
        try:
            return int(cid)
        except (TypeError, ValueError):
            return cid

    def _could_not_start(self, cid: str, exc: BaseException) -> None:
        """Desktop start-failure card: "**Translation could not be started.**" + ``ExcType: msg``."""
        content = "**Translation could not be started.**\n\n" f"`{type(exc).__name__}: {exc}`"
        created = _now_iso()
        try:
            self.store.append_messages(
                cid,
                [("assistant", content, f"❌ Could not start translation: {exc}\n", "Processing", "",
                  f"Request {self.store.request_count(cid) + 1}", {"created_at": created})],
            )
        except Exception:
            log.exception("recording the start failure failed")
        with self._lock:
            self.status[cid] = STATUS["could_not_start"]
        self._emit(cid)

    # ---- job events -------------------------------------------------------------------------

    def _run_for_snapshot(self, snapshot: Any) -> Optional[ChatRun]:
        if snapshot is None or job_kind(snapshot) not in ("",) + CHAT_JOB_KINDS:
            return None
        job_id = getattr(snapshot, "id", None)
        cid = chat_id_of(snapshot)
        with self._lock:
            for run in self.runs.values():
                if run.live and job_id is not None and run.job_id == job_id:
                    return run
            if cid is None:
                return None
            run = self.runs.get(cid)
            spec = getattr(snapshot, "spec", None)
            params = getattr(spec, "params", None) or {}
            run_dict = params.get("run") if isinstance(params, Mapping) else None
            if run is not None and run.live and (run.job_id in (None, job_id)):
                run.job_id = job_id
                return run
            if not isinstance(run_dict, Mapping):
                return None
            # A resumed / recovered job: re-attach a stream to it.
            resumed = DirectTextRun.from_dict(run_dict)
            turn = self._turn_of(cid, params, job_id)
            chat_run = ChatRun(
                cid=cid,
                run=resumed,
                stream=RunStream(),
                user_index=-1 if turn is None else turn,  # -1: its turn was deleted (no card)
                job_id=job_id,
                params=dict(params),
                title=str(getattr(spec, "title", "") or ""),
                kind=job_kind(snapshot) or DIRECT_TEXT_JOB_KIND,
                committed_from=self._message_count(cid),
            )
            chat_run.stream.provider = self._stream_provider(chat_run)
            self.runs[cid] = chat_run
        self.store.set_running(cid, True)
        return chat_run

    def on_snapshot(self, snapshot: Any) -> Optional[str]:
        """UI loop: apply a JobService snapshot; returns the chat id that changed."""
        run = self._run_for_snapshot(snapshot)
        if run is None:
            return self._detect_vanished(snapshot)
        state = state_name(snapshot)
        with self._lock:
            if getattr(snapshot, "output_dir", None):
                run.output_dir = str(snapshot.output_dir)
            spec = getattr(snapshot, "spec", None)
            if not run.params and isinstance(getattr(spec, "params", None), Mapping):
                run.params = dict(spec.params)
                run.title = str(getattr(spec, "title", "") or "")
            run.last_snapshot = snapshot
            run.job_state = state
            run.refining = bool(getattr(snapshot, "refining", False))
            question = question_of(snapshot)
            new_question = question is not None and (run.question or {}).get("id") != question.get("id")
            run.question = question
            if state in ("QUEUED",):
                run.state = "queued"
            elif state in ("STARTING", "RUNNING"):
                if run.state not in ("stopping", "force_stopping"):
                    run.state = "running"
                    if not run.question and run.status != STATUS["starting"]:
                        run.status = STATUS["translating"]
            elif state == "STOPPING":
                run.state = "stopping"
                run.status = STATUS["finishing"]
            elif state == "FORCE_STOPPING":
                run.state = "force_stopping"
                run.status = STATUS["force"]
        if new_question and run.awaiting_glossary:
            self.commit_gate(run)
        if state in TERMINAL_STATES and not run.finished and run.state != "finishing":
            self._schedule_finish(run, snapshot)
        self._emit(run.cid)
        return run.cid

    def _detect_vanished(self, snapshot: Any) -> Optional[str]:
        """Another job's snapshot (or none): finish tracked runs whose own job reached a terminal state.

        Each tracked job is looked up by id (``JobService.snapshot(job_id)``), so a queued or
        running neighbour never ends a run early; services without per-job lookup rely on
        the terminal snapshot of the job itself.
        """
        with self._lock:
            tracked = [r for r in self.runs.values() if r.live and r.job_id is not None and r.state != "finishing"]
        changed = None
        for run in tracked:
            own = self.jobs.snapshot_of(run.job_id)
            if own is not None and state_name(own) in TERMINAL_STATES:
                run.last_snapshot = own
                run.job_state = state_name(own)
                self._schedule_finish(run, own)
                changed = changed or run.cid
        return changed

    # ---- resubmit / follow-up jobs ------------------------------------------------------------

    def _chat_jobs(self, cid: str) -> list:
        """``[(snapshot, interrupted)]``: this chat's direct_text jobs JobService remembers (the
        Interrupted list, the history and the entries a newer job of their book took out of it), newest
        first. Survives a relaunch, unlike ``runs``."""
        view = getattr(self.jobs.jobs, "view", None) if self.jobs.jobs is not None else None
        if not callable(view):
            return []
        try:
            jobs_view = view()
            entries = [(snap, True) for snap in (getattr(jobs_view, "interrupted", ()) or ())]
            entries += [(snap, False) for snap in (getattr(jobs_view, "history", ()) or ())]
            entries += [(snap, False) for snap in (getattr(jobs_view, "carried_on", ()) or ())]
        except Exception:
            return []
        entries = [(snap, interrupted) for snap, interrupted in entries
                   if job_kind(snap) in CHAT_JOB_KINDS and chat_id_of(snap) == cid]
        entries.sort(key=lambda entry: _snapshot_time(entry[0]), reverse=True)
        return entries

    def _turn_resolver(self, cid: str) -> Callable[..., Optional[int]]:
        """``resolve(params, job_id=None, fallback=None)``: the current index of the turn a job ran
        for, or None once that turn was deleted (see the module docstring). The chat's
        fingerprints are computed at most once per resolver."""
        reader = getattr(self.store, "job_turns", None)
        turns = reader(cid) if callable(reader) else {}
        fingerprints: list = []

        def resolve(params: Any, job_id: Any = None, fallback: Any = None) -> Optional[int]:
            params = params if isinstance(params, Mapping) else {}
            submitted = params.get("user_index") if fallback is None else fallback
            key = turn_key(params, job_id)
            if key is None or key not in turns:
                try:
                    return int(submitted)
                except (TypeError, ValueError):
                    return None
            fp = turns.get(key)
            if not fp:
                return None
            if not fingerprints:
                fingerprints.append(self.store.fingerprints(cid))
            try:
                return fingerprints[0].index(fp)
            except ValueError:
                return None

        return resolve

    def _turn_of(self, cid: str, params: Any, job_id: Any = None, fallback: Any = None) -> Optional[int]:
        """The current index of the turn a job ran for (None: deleted)."""
        return self._turn_resolver(cid)(params, job_id, fallback)

    def turn_index(self, cid: Any, run: Optional[ChatRun]) -> Optional[int]:
        """The current index of ``run``'s user turn (a live run's turn cannot move: Delete waits)."""
        if run is None:
            return None
        if run.live:
            return run.user_index
        params = run.params or {"run": run.run.as_dict()}
        return self._turn_of(str(cid), params, run.job_id, run.user_index)

    def chat_job_turns(self, cid: Any) -> list:
        """``[(turn key, submitted user index)]`` of every job this chat knows (Delete message
        records their turns before it moves the messages)."""
        cid = str(cid)
        entries: list = []
        run = self.run_for(cid)
        if run is not None:
            entries.append((run.params or {"run": run.run.as_dict()}, run.job_id, run.user_index))
        for snap, _interrupted in self._chat_jobs(cid):
            params = getattr(getattr(snap, "spec", None), "params", None) or {}
            entries.append((params, getattr(snap, "id", None), params.get("user_index")))
        out, seen = [], set()
        for params, job_id, index in entries:
            key = turn_key(params, job_id)
            if key is None or key in seen:
                continue
            seen.add(key)
            out.append((key, index))
        return out

    def persisted_job(self, cid: Any, user_index: Any) -> Any:
        """The newest JobService snapshot of one chat turn, or None.

        After a relaunch ``runs`` is empty; the JobCard of an ended turn reads its real state
        (Interrupted, Stopped, Failed, Done and the chapter counts) from here instead of
        guessing it from the committed cards. A job's turn is its ``params["user_index"]``
        moved by any Delete message since (``_turn_of``)."""
        cid = str(cid)
        try:
            wanted = int(user_index)
        except (TypeError, ValueError):
            return None
        resolve = self._turn_resolver(cid)
        for snap, _interrupted in self._chat_jobs(cid):
            params = getattr(getattr(snap, "spec", None), "params", None) or {}
            if resolve(params, getattr(snap, "id", None)) == wanted:
                return snap
        return None

    def _last_job(self, cid: str, user_index: Optional[int] = None) -> tuple:
        """(params, title, interrupted job id, kind) of the chat's last chat job: this session's
        run, else the newest JobService entry (the id is set when that entry is an unresolved
        Interrupted job, which Resume must resolve). ``user_index``: the newest job of that turn
        instead of the chat's last one."""
        run = self.run_for(cid)
        if run is not None and run.params and (user_index is None or self.turn_index(cid, run) == user_index):
            return dict(run.params), run.title, None, run.kind
        resolve = self._turn_resolver(cid) if user_index is not None else None
        for snap, interrupted in self._chat_jobs(cid):
            spec = getattr(snap, "spec", None)
            if resolve is not None and resolve(getattr(spec, "params", None) or {}, getattr(snap, "id", None)) != user_index:
                continue
            pending = interrupted and getattr(snap, "resolution", None) is None
            return (dict(getattr(spec, "params", {}) or {}), str(getattr(spec, "title", "") or ""),
                    getattr(snap, "id", None) if pending else None, job_kind(snap) or DIRECT_TEXT_JOB_KIND)
        return {}, "", None, DIRECT_TEXT_JOB_KIND

    def _last_params(self, cid: str) -> tuple:
        """(params, title) of the chat's last chat job (see ``_last_job``)."""
        params, title, _interrupted_id, _kind = self._last_job(cid)
        return params, title

    def last_job_ending(self, cid: Any) -> tuple:
        """``(ending, snapshot, output folder, source path)`` of the job Resume / Retry failed would
        resubmit (``_last_job``: this session's run, else the newest job JobService remembers).

        ``ending`` follows the job card (``chat_view`` ended card): "live", "done", "issues" (done
        with failed chapters: Retry failed), "stopped", "failed", "interrupted", "discarded", or ""
        when the chat has no job. The output folder is the ``Attachments/<stem>`` workspace the run
        was committed into (this session's run only; '' otherwise)."""
        cid = str(cid)
        run = self.run_for(cid)
        if run is not None and run.params:
            snapshot = run.last_snapshot
            if run.live:
                ending = "live"
            else:
                ending = run.state if run.state in ("stopped", "failed") else "done"
            folder, source = run.output_folder, run.run.source_path
        else:
            entries = self._chat_jobs(cid)
            if not entries:
                return "", None, "", ""
            snapshot = entries[0][0]
            ending = ended_kind(snapshot)
            params = getattr(getattr(snapshot, "spec", None), "params", None) or {}
            run_dict = params.get("run") if isinstance(params, Mapping) else None
            folder = ""
            source = str(run_dict.get("source_path") or "") if isinstance(run_dict, Mapping) else ""
        if ending == "done" and progress_counts(snapshot)["failed"]:
            ending = "issues"
        return ending, snapshot, str(folder or ""), str(source or "")

    def relocate_output(self, cid: Any, old: str, new: str) -> bool:
        """A run's committed workspace moved (into the Library): point ``output_folder`` at it."""
        run = self.run_for(cid)
        if run is None or not old or not new:
            return False

        def same(a: str, b: str) -> bool:
            return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))

        with self._lock:
            if run.output_folder and same(run.output_folder, old):
                run.output_folder = str(new)
                return True
        return False

    def with_current_model(self, cid: Any, params: Mapping[str, Any]) -> dict:
        """``params`` of a resubmitted run with the chat's model now: its own (``overrides["model"]``),
        else the global model (``model_name``) - not the model it was sent with (owner: Resume kept
        using the Welcome's model after the model was changed). The chat's own model goes into
        ``config_overrides``; without one the run inherits the global model, as a new Send would."""
        try:
            own = str((self.store.overrides(str(cid)) or {}).get("model") or "").strip()
        except Exception:
            own = ""
        model = own or str(self.model_name() or "").strip()
        if not model:
            return dict(params)
        params = dict(params)
        overrides = dict(params.get("config_overrides") or {})
        if own:
            overrides["model"] = own
        else:
            overrides.pop("model", None)
        params["config_overrides"] = overrides
        params["model"] = model  # the stream's / job card's model name
        return params

    async def resubmit(self, cid: Any, *,
                       profile_check: Optional[Callable[[str], Optional[str]]] = None,
                       run_config: Optional[Mapping[str, Any]] = None,
                       run_config_for: Optional[str] = None,
                       overrides: Optional[Mapping[str, Any]] = None,
                       settings: Optional[DirectTextSettings] = None,
                       user_index: Optional[int] = None) -> Optional[ChatRun]:
        """Resume / Retry failed: the same JobSpec params (same run root), so the pipeline continues
        from its ``translation_progress.json`` and only redoes missing or failed chapters.

        ``user_index``: the turn whose newest job to resume (a Result card's own turn, hotfix 1 #4); None
        resumes the chat's last job.

        Four params follow the chat as it is now: the turn's index (``user_index``), the chat's next
        request number (``request_number``, as a new Send would get: the resumed run's cards do not
        repeat the stopped run's "Request N"), the model (:meth:`with_current_model`: the chat's model
        now, never the one it was sent with) and a prompt profile that no longer exists:
        ``profile_check(name)`` (``ChatView.resume_profile``) answers the profile to run with, None to
        inherit; a profile renamed or deleted since Send never reaches the backend, which would silently
        use the first profile (owner decisions #15 / #16). A profile that still exists stays as sent.

        An Interrupted job (killed app) is resumed through ``JobService.resume``, which resolves
        its Interrupted entry, so the Jobs page and the launch banner cannot run it again.

        ``run_config`` (owner #5: the Run options of a stopped / failed / interrupted card): the run's
        Run options are replaced by these (``run_request.with_run_config``; {} runs with the config's
        values), None keeps the ones it was sent with. ``run_config_for``: the ``turn_key`` of the run those
        Run options were taken from; when it is not the run resubmitted here (an older turn's card), they
        are ignored and the run keeps its own.

        ``overrides`` / ``settings`` (the Result card's chips: ``ChatView.run_overrides`` / ``settings`` now):
        the run's profile, target language and glossary prompts follow the chat's overrides now
        (``run_request.with_chat_overrides``) and its glossary the chat's policy now
        (``run_request.with_current_glossary``), like the model; None keeps the Send-time values.

        Hotfix 1 issue 5: Resume while the chat's run is live returns that run (None while a send of the chat
        still prepares); a run whose book an active job of this chat already runs (resumed from the Jobs page)
        binds to that job, one another chat or the Library runs raises ``JobAlreadyActive`` - nothing new is
        queued either way."""
        cid = str(cid)
        claimed = self._claim(cid, "")
        if claimed is not True:
            return claimed
        self.take_refused(cid)  # a refusal of an earlier submit is not this one's
        try:
            return await self._resubmit(cid, profile_check=profile_check, run_config=run_config,
                                        run_config_for=run_config_for, overrides=overrides, settings=settings,
                                        user_index=user_index)
        finally:
            self._release(cid)

    async def _resubmit(self, cid: str, *, profile_check: Optional[Callable[[str], Optional[str]]],
                        run_config: Optional[Mapping[str, Any]], run_config_for: Optional[str],
                        overrides: Optional[Mapping[str, Any]],
                        settings: Optional[DirectTextSettings],
                        user_index: Optional[int] = None) -> Optional[ChatRun]:
        params, title, interrupted_id, kind = self._last_job(cid, user_index)
        run_dict = params.get("run") if isinstance(params, Mapping) else None
        if not isinstance(run_dict, Mapping):
            return None
        user_index = self._turn_of(cid, params, interrupted_id)
        if user_index is None:
            return None  # the run's turn was deleted: nothing to resume into
        params = dict(params)
        params["user_index"] = user_index  # a new submission records the turn where it is now
        if run_config is not None and (run_config_for is None or run_config_for == turn_key(params)):
            params["config_overrides"] = with_run_config(params.get("config_overrides"), run_config)
        if overrides is not None:
            params["config_overrides"] = with_chat_overrides(params.get("config_overrides"), overrides)
        if settings is not None and kind != GENERATE_MEDIA_JOB_KIND and isinstance(params.get("options"), Mapping):
            params["options"] = with_current_glossary(params["options"], DirectTextRun.from_dict(run_dict), settings)
        counter = getattr(self.store, "request_count", None)
        if "request_number" in params and callable(counter):
            params["request_number"] = counter(cid) + 1
        params = self.with_current_model(cid, params)
        params = _rechecked_profile(params, profile_check)
        resumed = DirectTextRun.from_dict(run_dict)
        inputs = () if kind == GENERATE_MEDIA_JOB_KIND else (resumed.source_path,)
        # the run's book may be running already (the Jobs page's Resume, a Library run, another chat)
        active = self._active_job_for(kind, inputs, params)
        if active is not None:
            return self._bind_or_refuse(cid, active)
        chat_run = ChatRun(
            cid=cid, run=resumed, user_index=user_index,
            stream=RunStream(),
            params=params, title=title or job_title(resumed), kind=kind,
            committed_from=self._message_count(cid),
        )
        chat_run.stream.provider = self._stream_provider(chat_run)
        with self._lock:
            previous = self.runs.get(cid)
            self.runs[cid] = chat_run
            self.status.pop(cid, None)
        self.store.set_running(cid, True)
        try:
            job_id, reused = self._resume_job(interrupted_id, params) if interrupted_id else (None, None)
            if not job_id:
                session_title = str((self.store.session(cid) or {}).get("title") or "Chat")
                job_id, reused = await self._submit_job(kind, chat_run.title, inputs, params,
                                                        {"type": "chat", "cid": cid, "label": f"Chat · {session_title}"})
        except Exception:
            await self._drop_prepared(cid, chat_run, previous, remove_root=False)
            raise
        if reused is not None:
            # an active job already runs this book: nothing new was queued (the run root is the stopped run's)
            await self._drop_prepared(cid, chat_run, previous, remove_root=False)
            return self._bind_or_refuse(cid, reused)
        if not job_id:  # not queued (the app's preflight said why): the stopped run stays as it was
            await self._drop_prepared(cid, chat_run, previous, remove_root=False)
            self._mark_refused(cid)
            self._emit(cid)
            return None
        with self._lock:
            chat_run.job_id = job_id
            if chat_run.state == "preparing":
                chat_run.state = "queued"
        self._emit(cid)
        return chat_run

    async def compile(self, cid: Any, kind: str = "compile_epub", *, folder: Optional[str] = None) -> Any:
        """Compile EPUB / PDF from a chat output folder (shared ``text_jobs`` compile runners).

        ``folder``: the job card's own ``Direct Text/<chat>/Attachments/<stem>`` workspace (any
        card, also after a relaunch). Without one: the chat's run in this session, i.e. the
        pipeline's folder in the run root while it exists (a stopped / failed run keeps it), else
        the folder ``finish_run`` persisted the run into (a finished run's root is cleaned).
        """
        if folder and os.path.isdir(str(folder)):
            folder = str(folder)
        else:
            run = self.run_for(cid)
            candidates = (run.output_dir, run.output_folder) if run is not None else ()
            folder = next((str(c) for c in candidates if c and os.path.isdir(str(c))), "")
        if not folder:
            return None
        title = os.path.basename(folder.rstrip("\\/")) or "Book"
        return await self.jobs.submit(kind, title, (), {"folder": folder},
                                      {"type": "chat", "cid": str(cid), "label": "Chat"})

    # ---- stop / glossary ---------------------------------------------------------------------

    def request_stop(self, cid: Any, force: bool = False) -> None:
        run = self.live_run(cid)
        if run is None:
            return
        with self._lock:
            run.stop_requested = True
            if run.state == "queued":
                pass
            elif force:
                run.state = "force_stopping"
                run.status = STATUS["force"]
            elif run.state == "running":
                run.state = "stopping"
                run.status = STATUS["finishing"] if self.jobs.graceful_stop_configured() else STATUS["stopping"]
        if run.state == "queued" and run.job_id is not None:
            self.jobs.cancel_queued(run.job_id)
            self._schedule_finish(run, None, cancelled=True)
        else:
            self.jobs.request_stop(force=force)
        self._emit(run.cid)

    def commit_gate(self, run: ChatRun) -> list:
        """The glossary gate (the job waits for Edit / Yes / No): freeze the run's real request
        cards into the chat, like the dialog's ``_commit_active_request_phase``."""
        model = run.stream.model()
        session = self.store.session(run.cid)
        commit = getattr(self._store_for(run.cid), "commit_request_phase", None)
        if model is None or session is None or not callable(commit):
            return []
        try:
            committed = list(commit(session, model) or [])
        except Exception:
            log.exception("committing the glossary-phase cards failed")
            return []
        if committed:
            self.store.forget_bodies(run.cid)
            self.store.schedule_save()
        return committed

    def answer_glossary(self, cid: Any, accepted: bool) -> bool:
        """✓ Yes / ■ No on the approval card (desktop ``_resolve_glossary_approval``).

        Hotfix 1 issue 2: refused (False, nothing changes) while the gate's glossary is being refined - the
        refinement is for this translation, which must not start before it ended; JobService refuses the
        answer too (``JobService.answer``), also when the refinement started a moment ago."""
        run = self.live_run(cid)
        if run is None or not run.awaiting_glossary or run.refining:
            return False
        question = run.question or {}
        ok = self.jobs.answer(question.get("id"), bool(accepted))
        if not ok and self._service_refining(run.job_id):
            with self._lock:
                run.refining = True
            self._emit(run.cid)
            return False
        with self._lock:
            run.question = None
            run.status = STATUS["starting"] if accepted else STATUS["stopping"]
        if not accepted:
            self.request_stop(cid, force=False)
        self._emit(run.cid)
        return ok

    # ---- finishing ---------------------------------------------------------------------------

    def _schedule_finish(self, run: ChatRun, snapshot: Any, *, cancelled: bool = False) -> None:
        with self._lock:
            if run.finished or run.state == "finishing":
                return
            run.state = "finishing"
        thread = threading.Thread(target=self.finish, args=(run, snapshot, cancelled), name="gl-chat-finish", daemon=True)
        thread.start()  # started before it is listed: a joiner never sees an unstarted thread
        with self._lock:
            self.finish_threads = [t for t in self.finish_threads if t.is_alive()] + [thread]

    def finish(self, run: ChatRun, snapshot: Any = None, cancelled: bool = False) -> None:
        """Worker thread: commit the run into the chat (``ChatStore.finish_run``)."""
        state = state_name(snapshot) if snapshot is not None else ("CANCELLED" if cancelled else "DONE")
        if state == "CANCELLED" and snapshot is not None and getattr(snapshot, "started", 0) is None:
            cancelled = True  # taken out of the queue (Undo / Jobs page) before it ever ran
        status = STATUS["ready"]
        try:
            run.stream.drain(final=True)
            status = self._finish_with_store(run, snapshot, state, cancelled)
        except Exception as exc:
            log.exception("finishing the chat run failed")
            run.error = f"{type(exc).__name__}: {exc}"
            status = STATUS["streamed"]
        finally:
            with self._lock:
                run.finished = True
                run.state = {"FAILED": "failed", "CANCELLED": "stopped"}.get(state, "stopped" if run.stop_requested else "done")
                run.question = None
                self.status[run.cid] = status
            try:
                self.store.set_running(run.cid, False)
                self.store.refresh_attachments(run.cid)
                self.store.flush()
            except Exception:
                log.exception("saving the finished chat failed")
            self._emit(run.cid)
        self._emit_finished(run, state)

    def finish_run_state(self, run: ChatRun, snapshot: Any) -> dict:
        """``ChatStore.finish_run``'s run dict: the prepared run plus what the job ran with.

        ``force_no_glossary`` comes from the run options, ``glossary_path`` (the glossary the
        pipeline used: the run's MANUAL_GLOSSARY / the owner's ``manual_glossary_path``),
        ``run_env`` (the run's MANUAL_GLOSSARY itself) and ``model`` from the job's result (the
        job adapter records them before the job's process state is restored), so the
        attachment tree gets the run's glossary like on desktop. ``run_env`` is always set: the
        finish runs on its own thread, maybe while the next queued job has exported its own
        MANUAL_GLOSSARY, so the shared finish code must never read the live environment.
        """
        state = run.run.as_dict()
        params = run.params or dict(getattr(getattr(snapshot, "spec", None), "params", None) or {})
        options = params.get("options") if isinstance(params, Mapping) else None
        if isinstance(options, Mapping) and "force_no_glossary" in options:
            state["force_no_glossary"] = bool(options.get("force_no_glossary"))
        result = getattr(snapshot, "result", None) or {}
        state["run_env"] = {}
        if isinstance(result, Mapping):
            for key in ("glossary_path", "model", "force_no_glossary"):
                if result.get(key) not in (None, ""):
                    state[key] = result[key]
            if isinstance(result.get("run_env"), Mapping):
                state["run_env"] = dict(result["run_env"])
        if not state.get("model") and isinstance(params, Mapping) and params.get("model"):
            state["model"] = params["model"]
        return state

    def _finish_with_store(self, run: ChatRun, snapshot: Any, state: str, cancelled: bool) -> str:
        session = self.store.session(run.cid)
        store = self._store_for(run.cid)
        if cancelled or session is None or store is None:
            # cancelled while queued: nothing ran, nothing to commit
            return STATUS["stopped"] if cancelled else STATUS["ready"]
        model = run.stream.model()
        # Keep the run root of a stopped / failed run so Resume continues from its progress file.
        done = state == "DONE" and not run.stop_requested
        # Hotfix 1: also the run root of a run that ended with failed chapters ("Finished with issues"): its Retry
        # failed continues from that progress file and sends only the failed chapters, never the whole book again.
        final = snapshot if snapshot is not None else run.last_snapshot
        cleanup = done and not progress_counts(final)["failed"]
        run_state = self.finish_run_state(run, snapshot)
        if not done and not _streamed_text(model, run_state):
            # A failed / stopped run with no reply: its output file still holds the untranslated input
            # (the text pipeline writes the source for an unfinished chunk), which the card showed as the
            # answer. Not read: the card says no output was produced; the run root stays for Resume.
            run_state["expected_output"] = ""
        # The run root is removed here, not by finish_run: its glossary folder goes with the kept workspace first
        result = store.finish_run(session, run_state, model if model is not None else [], cleanup=False)
        status = None
        if isinstance(result, Mapping):
            run.output_folder = str(result.get("output_folder") or "")
            status = result.get("status")
        if cleanup:
            _finish_run_root(run_state, result)
        self.store.forget_bodies(run.cid)
        self.store.schedule_save()
        return str(status or STATUS["ready"])
