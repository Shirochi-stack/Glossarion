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
* ``finish()`` (worker thread): ``ChatStore.finish_run`` (the dialog's
  ``_finish_translation``: final drain, translated file, run tree persisted into the
  chat folder, cards + "Extraction report" / "Attachment actions" committed, history
  saved). A run that was stopped or failed keeps its run root (Resume / Retry continue
  from ``translation_progress.json``; the dialog deletes its temp root - recorded
  mobile divergence); a finished one is cleaned like the dialog's.

Runs are keyed by chat id; a resumed job (same params after a kill) is re-attached
from its ``params["run"]``. Listeners (``subscribe``) are told the chat id that changed;
the chat feature marshals them to the UI loop.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Mapping, Optional

from glossarion_mobile.ui.chat.direct_text_rules import DirectTextSettings, ManualGlossarySource
from glossarion_mobile.ui.chat.job_binding import (
    RUNNING_STATES,
    TERMINAL_STATES,
    JobsAdapter,
    chat_id_of,
    job_kind,
    question_of,
    state_name,
)
from glossarion_mobile.ui.chat.run_request import (
    DIRECT_TEXT_JOB_KIND,
    DirectTextRun,
    job_params,
    job_title,
    prepare_direct_text_run,
    user_turn,
)
from glossarion_mobile.ui.chat.stream_bridge import RunStream

__all__ = ["ChatRun", "ChatRuns", "STATUS"]

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
}

_GLOSSARY_QUESTION_KINDS = ("glossary_approval", "direct_text_glossary_approval")


def is_glossary_question(kind: Any) -> bool:
    """The pipeline's Direct Text glossary gate (``_ui_request`` kind), whatever its exact name."""
    value = str(kind or "").lower()
    return value in _GLOSSARY_QUESTION_KINDS or ("glossary" in value and "approv" in value)


def _now_iso() -> str:
    """Desktop ``_direct_response_timestamp`` format (timezone-aware, seconds)."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _snapshot_time(snapshot: Any) -> float:
    """When a remembered job last changed (finished, else started, else created)."""
    for name in ("finished", "started", "created"):
        value = getattr(snapshot, name, None)
        if isinstance(value, (int, float)) and value:
            return float(value)
    return 0.0


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

    @property
    def live(self) -> bool:
        return not self.finished

    @property
    def awaiting_glossary(self) -> bool:
        return bool(self.question) and is_glossary_question(self.question.get("kind"))


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
        self._job_unsub: Optional[Callable[[], None]] = None
        self.finish_threads: list = []

    # ---- listeners ------------------------------------------------------------------

    def subscribe(self, callback: Callable[[str], Any]) -> Callable[[], None]:
        self._listeners.append(callback)

        def unsubscribe() -> None:
            try:
                self._listeners.remove(callback)
            except ValueError:
                pass

        return unsubscribe

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

    def detach(self) -> None:
        if self._job_unsub is not None:
            try:
                self._job_unsub()
            except Exception:
                pass
            self._job_unsub = None

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
                return STATUS["glossary"]
            return run.status
        text = self.status.get(str(cid))
        return None if not text or text == STATUS["ready"] else text

    # ---- send ------------------------------------------------------------------------------

    async def _io(self, fn: Callable[..., Any], *args: Any) -> Any:
        if self.run_io is None:
            return fn(*args)
        return await self.run_io(fn, *args)

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
    ) -> ChatRun:
        """Record the turn (unless ``user_index`` points at an already recorded one, e.g. a
        Plan card), prepare the run off the loop and submit the ``direct_text`` job."""
        cid = str(cid)
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
                           user_index=int(user_index))
        chat_run.stream.provider = self._stream_provider(chat_run)
        with self._lock:
            self.runs[cid] = chat_run
            self.status.pop(cid, None)
        self.store.set_running(cid, True)
        await self._io(self.store.flush)  # history on disk before the job starts (desktop saves at send)
        params = job_params(chat_id=self._chat_id_value(cid), user_index=user_index, run=run, settings=settings, overrides=overrides)
        # The job's DirectTextStream starts at the chat's next request number (the dialog's
        # _active_request_next_number) and counts tokens for the run's model.
        params["request_number"] = self.store.request_count(cid) + 1
        model = (params.get("config_overrides") or {}).get("model") or self.model_name()
        if model:
            params["model"] = str(model)
        title = str((self.store.session(cid) or {}).get("title") or "Chat")
        origin = {"type": "chat", "cid": cid, "label": f"Chat · {title}"}
        try:
            job_id = await self.jobs.submit(DIRECT_TEXT_JOB_KIND, job_title(run, title), (run.source_path,), params, origin)
        except Exception as exc:
            with self._lock:
                chat_run.finished = True
            self.store.set_running(cid, False)
            self._could_not_start(cid, exc)
            raise
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
        if snapshot is None or job_kind(snapshot) not in ("", DIRECT_TEXT_JOB_KIND):
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
            chat_run = ChatRun(
                cid=cid,
                run=resumed,
                stream=RunStream(),
                user_index=int(params.get("user_index") or 0),
                job_id=job_id,
                params=dict(params),
                title=str(getattr(spec, "title", "") or ""),
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
        Interrupted list and the history), newest first. Survives a relaunch, unlike ``runs``."""
        view = getattr(self.jobs.jobs, "view", None) if self.jobs.jobs is not None else None
        if not callable(view):
            return []
        try:
            jobs_view = view()
            entries = [(snap, True) for snap in (getattr(jobs_view, "interrupted", ()) or ())]
            entries += [(snap, False) for snap in (getattr(jobs_view, "history", ()) or ())]
        except Exception:
            return []
        entries = [(snap, interrupted) for snap, interrupted in entries
                   if job_kind(snap) == DIRECT_TEXT_JOB_KIND and chat_id_of(snap) == cid]
        entries.sort(key=lambda entry: _snapshot_time(entry[0]), reverse=True)
        return entries

    def persisted_job(self, cid: Any, user_index: Any) -> Any:
        """The newest JobService snapshot of one chat turn (``params["user_index"]``), or None.

        After a relaunch ``runs`` is empty; the JobCard of an ended turn reads its real state
        (Interrupted, Stopped, Failed, Done and the chapter counts) from here instead of
        guessing it from the committed cards."""
        cid = str(cid)
        for snap, _interrupted in self._chat_jobs(cid):
            params = getattr(getattr(snap, "spec", None), "params", None) or {}
            try:
                if int(params.get("user_index")) == int(user_index):
                    return snap
            except (TypeError, ValueError):
                continue
        return None

    def _last_job(self, cid: str) -> tuple:
        """(params, title, interrupted job id) of the chat's last direct_text job: this session's
        run, else the newest JobService entry (the id is set when that entry is an unresolved
        Interrupted job, which Resume must resolve)."""
        run = self.run_for(cid)
        if run is not None and run.params:
            return dict(run.params), run.title, None
        for snap, interrupted in self._chat_jobs(cid):
            spec = getattr(snap, "spec", None)
            pending = interrupted and getattr(snap, "resolution", None) is None
            return (dict(getattr(spec, "params", {}) or {}), str(getattr(spec, "title", "") or ""),
                    getattr(snap, "id", None) if pending else None)
        return {}, "", None

    def _last_params(self, cid: str) -> tuple:
        """(params, title) of the chat's last direct_text job (see ``_last_job``)."""
        params, title, _interrupted_id = self._last_job(cid)
        return params, title

    async def resubmit(self, cid: Any) -> Optional[ChatRun]:
        """Resume / Retry failed: the same JobSpec params (same run root), so the pipeline continues
        from its ``translation_progress.json`` and only redoes missing or failed chapters.

        An Interrupted job (killed app) is resumed through ``JobService.resume``, which resolves
        its Interrupted entry, so the Jobs page and the launch banner cannot run it again."""
        cid = str(cid)
        if self.live_run(cid) is not None:
            return None
        params, title, interrupted_id = self._last_job(cid)
        run_dict = params.get("run") if isinstance(params, Mapping) else None
        if not isinstance(run_dict, Mapping):
            return None
        resumed = DirectTextRun.from_dict(run_dict)
        chat_run = ChatRun(
            cid=cid, run=resumed, user_index=int(params.get("user_index") or 0),
            stream=RunStream(),
            params=dict(params), title=title or job_title(resumed),
        )
        chat_run.stream.provider = self._stream_provider(chat_run)
        with self._lock:
            self.runs[cid] = chat_run
            self.status.pop(cid, None)
        self.store.set_running(cid, True)
        job_id = self.jobs.resume(interrupted_id) if interrupted_id else None
        if not job_id:
            session_title = str((self.store.session(cid) or {}).get("title") or "Chat")
            job_id = await self.jobs.submit(DIRECT_TEXT_JOB_KIND, chat_run.title, (resumed.source_path,), params,
                                            {"type": "chat", "cid": cid, "label": f"Chat · {session_title}"})
        with self._lock:
            chat_run.job_id = job_id
            if chat_run.state == "preparing":
                chat_run.state = "queued"
        self._emit(cid)
        return chat_run

    async def compile(self, cid: Any, kind: str = "compile_epub") -> Any:
        """Compile EPUB / PDF from the run's output folder (shared ``text_jobs`` compile runners).

        The pipeline's folder in the run root while it exists (a stopped / failed run keeps it),
        else the folder ``finish_run`` persisted the run into (an attachment's
        ``Direct Text/<chat>/Attachments/<stem>`` tree; a finished run's root is cleaned).
        """
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
        commit = getattr(getattr(self.store, "binding", None), "store", None)
        commit = getattr(commit, "commit_request_phase", None)
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
        """✓ Yes / ■ No on the approval card (desktop ``_resolve_glossary_approval``)."""
        run = self.live_run(cid)
        if run is None or not run.awaiting_glossary:
            return False
        question = run.question or {}
        ok = self.jobs.answer(question.get("id"), bool(accepted))
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
        store = getattr(getattr(self.store, "binding", None), "store", None)
        if cancelled or session is None or store is None:
            # cancelled while queued: nothing ran, nothing to commit
            return STATUS["stopped"] if cancelled else STATUS["ready"]
        model = run.stream.model()
        # Keep the run root of a stopped / failed run so Resume continues from its progress file.
        cleanup = state == "DONE" and not run.stop_requested
        result = store.finish_run(session, self.finish_run_state(run, snapshot),
                                  model if model is not None else [], cleanup=cleanup)
        status = None
        if isinstance(result, Mapping):
            run.output_folder = str(result.get("output_folder") or "")
            status = result.get("status")
        self.store.forget_bodies(run.cid)
        self.store.schedule_save()
        return str(status or STATUS["ready"])
