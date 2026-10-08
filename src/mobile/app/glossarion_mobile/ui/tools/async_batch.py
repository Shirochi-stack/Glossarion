"""Async batch (``/tools/async``, UI_SPEC §4.7; desktop "📦 Async Processing (50% Off)" dialog).

* **Model**: "Current Model" + the shared support check (``async_batch_core.async_support_status``:
  "✓ Supported (OPENAI)" / "✗ Not supported for async") and the 50% discount note.
* **Configuration**: "Wait for completion" (``async_wait_for_completion``) and "Poll interval
  (seconds)" (``async_poll_interval``, 10-600), saved like the dialog saves them; the note
  "Async processing will skip chapters that require chunking".
* **Source**: the SourcePicker (a Library book's raw file, Browse); "Start Async Processing"
  and "Estimate Cost Only" run ``async_batch`` jobs (``job_kinds.async_batch``: the dialog's
  own handlers through ``async_batch_core.HeadlessAsyncBatch``). The dialog's questions come
  back from the job (``JobService.on_question``, kind ``async_batch_question``) and are asked
  here with the desktop titles, texts and buttons; its notices (``async_batch_message``) show as
  snackbars or sheets.
* **Active Async Jobs**: the job list file (``async_jobs.json`` in the app data folder) as the
  dialog's rows (``job_display_row``: id, provider · model, status, progress, created, source,
  cost); tap selects. Refresh (check every pending batch) · Check Status · Retrieve Results
  (download and write the chapters into the output folder) · Cancel · Delete · Clear Completed.

The UI never builds a ``HeadlessOwner``: reading the job list needs only the processor (no
owner); every action that talks to a provider runs as a job.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import flet as ft

from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.components.info_sheet import InfoSheet
from glossarion_mobile.ui.router import RouteMatch
from glossarion_mobile.ui.screens.base import Screen
from glossarion_mobile.ui.theme import HIT_TARGET
from glossarion_mobile.ui.tools import targets as tg
from glossarion_mobile.ui.tools.common import ChoiceDialog, JobWatch, action_button, card, hint_text
from glossarion_mobile.ui.tools.source_picker import SourcePicker

__all__ = ["AsyncBatchScreen", "AsyncSnapshot", "async_spec", "load_async_snapshot", "question_options"]

log = logging.getLogger("glossarion.tools.async")

KIND = "async_batch"
QUESTION_KIND = "async_batch_question"
DISCOUNT_NOTE = "50% off · results in up to 24 h"
CHUNKING_NOTE = "Note: Async processing will skip chapters that require chunking"
WAIT_KEY = "async_wait_for_completion"
POLL_KEY = "async_poll_interval"
TERMINAL = ("completed", "failed", "cancelled", "expired")


@dataclass(frozen=True)
class AsyncSnapshot:
    rows: tuple = ()
    model: str = ""
    supported: bool = False
    status_text: str = ""
    jobs_file: str = ""
    error: Optional[str] = None
    missing: tuple = ()


def _core() -> Any:
    import async_batch_core

    return async_batch_core


def load_async_snapshot(model: str, jobs_file: Optional[str] = None) -> AsyncSnapshot:
    """Blocking: the job list rows and the model row (no owner: the processor only reads the file)."""
    try:
        core = _core()
    except Exception as exc:
        return AsyncSnapshot(model=model, error=f"The async batch core is not available in this build ({exc})",
                             missing=("async_batch_core",))
    try:
        path = jobs_file or core.default_jobs_file()
        processor = core.AsyncAPIProcessor(None, jobs_file=path)
        rows = tuple(core.job_display_row(job_id, job) for job_id, job in processor.jobs.items())
        supported, text = core.async_support_status(processor, model or "")
    except Exception as exc:
        log.exception("reading the async job list failed")
        return AsyncSnapshot(model=model, error=f"The async job list could not be read ({exc})")
    return AsyncSnapshot(rows=rows, model=model, supported=bool(supported), status_text=str(text), jobs_file=path)


def async_spec(action: str, *, jobs_file: str = "", source: str = "", job_ids: Sequence[str] = (),
               title: str = "") -> Any:
    from glossarion_mobile.services.jobs import JobSpec

    labels = {"submit": "Submit", "estimate": "Estimate cost", "refresh": "Refresh", "check": "Check status",
              "retrieve": "Retrieve results", "cancel": "Cancel", "delete": "Delete", "clear_completed": "Clear completed"}
    name = title or (os.path.basename(source) if source else "")
    return JobSpec(kind=KIND, title=f"{labels.get(action, action)}{' · ' + name if name else ''}",
                   inputs=(source,) if source else (),
                   params={"action": action, "jobs_file": jobs_file, "job_ids": [str(j) for j in job_ids]},
                   origin={"type": "tool", "label": "Tools · Async batch"}, resumable=False)


def question_options(buttons: Sequence[str]) -> list:
    """The dialog buttons of an ``async_batch_question`` as ChoiceDialog options (desktop order)."""
    labels = {"yes": "Yes", "no": "No", "cancel": "Cancel"}
    kinds = {"yes": "filled", "no": "text", "cancel": "text"}
    names = [str(b) for b in buttons or ()] or ["yes", "no"]
    ordered = [n for n in ("cancel", "no", "yes") if n in names] + [n for n in names if n not in labels]
    return [(name, labels.get(name, name.title()), kinds.get(name, "text")) for name in ordered]


class AsyncBatchScreen(Screen):
    title = "Async batch"

    def __init__(self, match: Optional[RouteMatch], ctx: Any) -> None:
        super().__init__(match)
        self.ctx = ctx
        self.state = ctx.tool_state.setdefault("async", {})
        self.target: Optional[tg.ToolTarget] = self.state.get("target")
        self.snapshot = AsyncSnapshot()
        self.selected: set = set(self.state.get("selected") or ())
        self.watch = JobWatch(ctx, self._on_job_end, self._on_job_change)
        self.active_job: Any = None
        self.picker: Optional[SourcePicker] = None
        self.row_controls: dict = {}
        self.dialogs: list = []
        self.messages: list = []
        self._unsubs: list = []

    # ---- layout -----------------------------------------------------------------------------------

    def actions(self) -> list:
        return [ft.IconButton(icon=ft.Icons.REFRESH, tooltip="Reload the job list", key="async-reload",
                              on_click=lambda e: self.ctx.spawn(self.reload()), size_constraints=HIT_TARGET)]

    def build_body(self) -> ft.Control:
        model = str(self.ctx.cfg("model", "") or "")
        self.model_text = ft.Text(f"Current Model: {model or 'Not selected'}", theme_style=ft.TextThemeStyle.BODY_MEDIUM,
                                  key="async-model")
        self.support_text = ft.Text("", theme_style=ft.TextThemeStyle.BODY_SMALL, key="async-support")
        info = card("Model", [self.model_text, self.support_text, hint_text(DISCOUNT_NOTE, key="async-discount")],
                    icon="SCHEDULE_SEND", key="async-info")
        self.wait_switch = ft.Switch(label="Wait for completion (the job keeps polling)",
                                     value=bool(self.ctx.cfg(WAIT_KEY, False)), on_change=self._on_wait,
                                     key="async-wait")
        self.poll_field = ft.TextField(label="Poll interval (seconds)", value=str(self.ctx.cfg(POLL_KEY, 60)),
                                       dense=True, keyboard_type=ft.KeyboardType.NUMBER, width=200,
                                       on_blur=self._on_poll, on_submit=self._on_poll, key="async-poll")
        config = card("Async Processing Configuration", [self.wait_switch, self.poll_field,
                                                         hint_text(CHUNKING_NOTE, key="async-chunk-note")],
                      icon="TUNE", key="async-config")
        self.source_text = ft.Text("", theme_style=ft.TextThemeStyle.BODY_MEDIUM, key="async-source")
        self.cost_text = ft.Text("Select chapters to see cost estimate", theme_style=ft.TextThemeStyle.BODY_SMALL,
                                 key="async-cost")
        self.start_button = ft.FilledButton(content="Start Async Processing", icon=ft.Icons.SEND,
                                            on_click=lambda e: self.ctx.spawn(self.start("submit")), key="async-start")
        self.estimate_button = ft.FilledTonalButton(content="Estimate Cost Only", icon=ft.Icons.CALCULATE,
                                                    on_click=lambda e: self.ctx.spawn(self.start("estimate")),
                                                    key="async-estimate")
        self.progress = ft.ProgressBar(visible=False, key="async-progress")
        self.run_status = ft.Text("", theme_style=ft.TextThemeStyle.BODY_SMALL, key="async-run-status")
        self.stop_button = ft.OutlinedButton(content="Stop", icon=ft.Icons.STOP, visible=False,
                                             on_click=self._on_stop, key="async-stop")
        source = card("Create batch from…", [
            self.source_text,
            ft.Row([ft.FilledTonalButton(content="Choose…", icon=ft.Icons.FOLDER_OPEN,
                                         on_click=lambda e: self.open_picker(), key="async-choose")], wrap=True),
            ft.Row([self.start_button, self.estimate_button, self.stop_button], wrap=True, spacing=8),
            self.progress, self.run_status, self.cost_text,
        ], icon="UPLOAD_FILE", key="async-source-card")
        self.jobs_column = ft.Column(spacing=4, key="async-jobs")
        self.selection_text = ft.Text("", theme_style=ft.TextThemeStyle.LABEL_MEDIUM, key="async-selection")
        self.job_actions = ft.Row(wrap=True, spacing=6, run_spacing=6, key="async-job-actions")
        jobs = card("Active Async Jobs", [self.selection_text, self.job_actions, self.jobs_column],
                    icon="LIST_ALT", key="async-jobs-card")
        self._render_source()
        self._render_job_actions()
        return ft.ListView(controls=[info, config, source, jobs], expand=True, spacing=tokens.SPACING["md"],
                           padding=ft.Padding.symmetric(horizontal=12, vertical=8), key="async-screen")

    # ---- lifecycle ----------------------------------------------------------------------------------

    def did_show(self) -> None:
        jobs = self.ctx.jobs
        for name, callback in (("on_question", self._on_question), ("on_event", self._on_event)):
            subscribe = getattr(jobs, name, None) if jobs is not None else None
            if callable(subscribe):
                try:
                    self._unsubs.append(subscribe(callback))
                except Exception:
                    log.debug("subscribing to %s failed", name, exc_info=True)
        for snap in self.watch.adopt((KIND,)):
            self._on_job_change(snap)
            pending = getattr(jobs, "pending_question", None)
            question = pending(snap.id) if callable(pending) else None
            if question:
                self._on_question(snap, question)
        self.ctx.spawn(self.reload())

    def dispose(self) -> None:
        self.watch.stop()
        for unsub in self._unsubs:
            try:
                unsub()
            except Exception:
                pass
        self._unsubs = []

    # ---- data ---------------------------------------------------------------------------------------

    async def reload(self) -> AsyncSnapshot:
        model = str(self.ctx.cfg("model", "") or "")
        jobs_file = self.state.get("jobs_file") or ""
        self.snapshot = await self.ctx.io(load_async_snapshot, model, jobs_file or None)
        if self.snapshot.jobs_file:
            self.state["jobs_file"] = self.snapshot.jobs_file
        self.model_text.value = f"Current Model: {model or 'Not selected'}"
        if self.snapshot.error:
            self.support_text.value = self.snapshot.error
            self.support_text.color = ft.Colors.ERROR
        else:
            self.support_text.value = self.snapshot.status_text
            self.support_text.color = ft.Colors.PRIMARY if self.snapshot.supported else ft.Colors.ERROR
        keys = {row.get("job_id") for row in self.snapshot.rows}
        self.selected &= keys
        self._render_jobs()
        self._render_job_actions()
        self._push(self.body)
        return self.snapshot

    def _render_source(self) -> None:
        target = self.target
        if target is None or not target.source:
            self.source_text.value = "Choose the book to submit"
        else:
            self.source_text.value = f"📖 {target.source_name}" + (f" · 📁 {target.folder_name}" if target.folder else "")
        busy = self.watch.active() is not None
        self.start_button.disabled = busy or target is None or not target.source
        self.estimate_button.disabled = self.start_button.disabled

    def _render_jobs(self) -> None:
        self.row_controls = {}
        rows = list(self.snapshot.rows)
        if not rows:
            self.jobs_column.controls = [hint_text("No async jobs yet.", key="async-empty")]
        else:
            self.jobs_column.controls = [self._job_row(row) for row in rows]
        count = len(self.selected)
        self.selection_text.value = f"{count} selected" if count else "Tap a job to select it"

    def _job_row(self, row: dict) -> ft.Control:
        job_id = str(row.get("job_id") or "")
        selected = job_id in self.selected
        state = str(row.get("state") or "")
        color = {"completed": ft.Colors.PRIMARY, "failed": ft.Colors.ERROR, "cancelled": ft.Colors.OUTLINE,
                 "expired": ft.Colors.OUTLINE}.get(state, ft.Colors.TERTIARY)
        lines = [
            ft.Row([ft.Text(str(row.get("display_id") or job_id), weight=ft.FontWeight.W_600, expand=True,
                            max_lines=1, overflow=ft.TextOverflow.ELLIPSIS),
                    ft.Text(str(row.get("status") or ""), color=color, weight=ft.FontWeight.W_600)]),
            ft.Text(f"{row.get('provider', '')} · {row.get('model', '')} · {row.get('progress', '')}",
                    theme_style=ft.TextThemeStyle.BODY_SMALL),
            ft.Text(" · ".join(p for p in (str(row.get("source_file") or ""), str(row.get("created") or ""),
                                           f"Cost {row.get('cost', 'N/A')}") if p),
                    theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT),
        ]
        control = ft.Container(
            content=ft.Column(lines, spacing=2, tight=True),
            padding=ft.Padding.symmetric(horizontal=10, vertical=8),
            border_radius=tokens.RADII["card"],
            bgcolor=ft.Colors.SECONDARY_CONTAINER if selected else ft.Colors.SURFACE_CONTAINER,
            border=ft.Border.all(2, ft.Colors.PRIMARY) if selected else None,
            on_click=lambda e, j=job_id: self.toggle_job(j),
            ink=True,
            key=f"async-job-{job_id}",
        )
        self.row_controls[job_id] = control
        return control

    def _selected_rows(self) -> list:
        return [row for row in self.snapshot.rows if row.get("job_id") in self.selected]

    def _render_job_actions(self) -> None:
        busy = self.watch.active() is not None
        rows = self._selected_rows()
        none = "Select a job first" if not rows else None
        one = "Select one job" if len(rows) != 1 else None
        done = [r for r in rows if r.get("state") == "completed"]
        self.job_actions.controls = [
            action_button("Refresh", "REFRESH", lambda e: self.ctx.spawn(self.start("refresh")), key="async-refresh",
                          reason="A batch action is running" if busy else None),
            action_button("Check Status", "INFO_OUTLINE", lambda e: self.ctx.spawn(self.start("check")),
                          key="async-check", reason="A batch action is running" if busy else one),
            action_button("Retrieve Results", "DOWNLOAD", lambda e: self.ctx.spawn(self.start("retrieve")),
                          key="async-retrieve", reason="A batch action is running" if busy else (
                              none or (None if done else "No completed job selected"))),
            action_button("Cancel", "CANCEL", lambda e: self.ctx.spawn(self.start("cancel")), key="async-cancel",
                          reason="A batch action is running" if busy else none),
            action_button("Delete", "DELETE_OUTLINE", lambda e: self.ctx.spawn(self.start("delete")),
                          key="async-delete", destructive=True,
                          reason="A batch action is running" if busy else none),
            action_button("Clear Completed", "CLEANING_SERVICES",
                          lambda e: self.ctx.spawn(self.start("clear_completed")), key="async-clear",
                          reason="A batch action is running" if busy else None),
        ]

    def toggle_job(self, job_id: str) -> None:
        if job_id in self.selected:
            self.selected.discard(job_id)
        else:
            self.selected.add(job_id)
        self.state["selected"] = set(self.selected)
        self._render_jobs()
        self._render_job_actions()
        self._push(self.jobs_column, self.selection_text, self.job_actions)

    # ---- settings -----------------------------------------------------------------------------------

    def _on_wait(self, e: Any = None) -> None:
        self.ctx.set_cfg(WAIT_KEY, bool(self.wait_switch.value))

    def _on_poll(self, e: Any = None) -> Optional[int]:
        try:
            value = int(str(self.poll_field.value or "").strip())
        except ValueError:
            self.poll_field.error = "Enter a number of seconds"
            self._push(self.poll_field)
            return None
        value = min(600, max(10, value))  # the dialog's spin box range
        self.poll_field.value = str(value)
        self.poll_field.error = None
        self.ctx.set_cfg(POLL_KEY, value)
        self._push(self.poll_field)
        return value

    # ---- source -------------------------------------------------------------------------------------

    def open_picker(self) -> SourcePicker:
        picker = SourcePicker(self.ctx, title="Create batch from…", multi=False,
                              eligible=lambda t: None if t.source else "No raw source file",
                              on_done=self.set_target, segment="library", find_folder=True)
        self.picker = picker
        if self.ctx.page is not None:
            picker.show(self.ctx.page)
            self.ctx.spawn(picker.load())
        return picker

    def set_target(self, targets: Sequence[tg.ToolTarget]) -> None:
        self.target = targets[0] if targets else None
        self.state["target"] = self.target
        self._render_source()
        self._push(self.source_text, self.start_button, self.estimate_button)

    # ---- actions ------------------------------------------------------------------------------------

    async def start(self, action: str) -> Optional[str]:
        if not self.ctx.has_kind(KIND):
            self.ctx.say("Async batch jobs are not available in this build")
            return None
        if self.watch.active() is not None:
            self.ctx.say("A batch action is already running")
            return None
        source = ""
        job_ids: list = []
        if action in ("submit", "estimate"):
            if self.target is None or not self.target.source:
                self.ctx.say("Please select a file to translate first")
                return None
            source = self.target.source
            self.ctx.remember_source("tools.async", self.target.title)
        elif action in ("check", "retrieve", "cancel", "delete"):
            job_ids = [str(r.get("job_id")) for r in self._selected_rows()]
            if not job_ids:
                self.ctx.say("Please select a job first")
                return None
            if action == "check":
                job_ids = job_ids[:1]
        spec = async_spec(action, jobs_file=self.state.get("jobs_file") or self.snapshot.jobs_file, source=source,
                          job_ids=job_ids)
        job_id = await self.ctx.submit(spec)
        if job_id:
            self.watch.watch(job_id)
            self.active_job = job_id
            self._set_running(True, f"{spec.title}…")
        return job_id

    def _set_running(self, running: bool, text: str = "") -> None:
        self.progress.visible = running
        self.stop_button.visible = running
        self.run_status.value = text
        self._render_source()
        self._render_job_actions()
        self._push(self.progress, self.stop_button, self.run_status, self.start_button, self.estimate_button,
                   self.job_actions)

    def _on_stop(self, e: Any = None) -> None:
        snap = self.watch.active()
        jobs = self.ctx.jobs
        if snap is not None and jobs is not None:
            try:
                jobs.request_stop(snap.id)
            except Exception:
                log.exception("stopping the async job failed")

    def _on_job_change(self, snap: Any) -> None:
        if not getattr(snap, "is_terminal", False):
            self._set_running(True, str(getattr(snap, "phase", "") or getattr(snap, "title", "") or "Working…"))

    def _on_job_end(self, snap: Any) -> None:
        result = dict(getattr(snap, "result", {}) or {})
        cost = result.get("async_cost") or result.get("async_cost_info")
        if cost and result.get("async_action") == "estimate":
            self.cost_text.value = str(cost)
        error = getattr(snap, "error", None)
        status = "Stopped" if getattr(snap, "stopped", False) else (f"Failed: {error}" if error else "Done")
        if result.get("async_action") == "submit" and not result.get("async_job_id") and not error:
            status = "Not submitted"
        self._set_running(False, status)
        self._push(self.cost_text)
        self.ctx.spawn(self.reload())

    # ---- job questions / notices --------------------------------------------------------------------

    def _on_question(self, snap: Any, question: Any = None) -> Optional[ChoiceDialog]:
        if not isinstance(question, dict) or question.get("kind") != QUESTION_KIND:
            return None
        if question.get("job_id") not in self.watch.job_ids:
            return None
        data = dict(question.get("data") or {})
        dialog = ChoiceDialog(str(data.get("title") or "Async batch"), str(data.get("text") or ""),
                              question_options(data.get("buttons") or ()), key="async-question")
        self.dialogs.append(dialog)
        scripted = self.ctx.extras.get("answers") if hasattr(self.ctx, "extras") else None
        if isinstance(scripted, list):
            answer = scripted.pop(0) if scripted else None
            self._answer(question, answer)
            return dialog
        if self.ctx.page is not None:
            dialog.show(self.ctx.page)
            self.ctx.spawn(self._await_answer(dialog, question))
        return dialog

    async def _await_answer(self, dialog: ChoiceDialog, question: dict) -> None:
        self._answer(question, await dialog.wait())

    def _answer(self, question: dict, answer: Optional[str]) -> None:
        jobs = self.ctx.jobs
        answer_fn = getattr(jobs, "answer", None) if jobs is not None else None
        if callable(answer_fn):
            answer_fn(question.get("id"), answer or question.get("default") or "no")

    def _on_event(self, job_id: str, kind: str, data: Any) -> None:
        if job_id not in self.watch.job_ids:
            return
        data = dict(data or {})
        if kind == "async_batch_message":
            self.messages.append(data)
            title, text = str(data.get("title") or ""), str(data.get("text") or "")
            if len(text) > 160 or "\n" in text:
                sheet = InfoSheet(title=title or "Async batch", body=text)
                if self.ctx.page is not None:
                    self.ctx.show(sheet)
            else:
                self.ctx.say(f"{title}: {text}" if title else text)
        elif kind == "async_batch_cost":
            self.cost_text.value = str(data.get("text") or "")
            self._push(self.cost_text)
        elif kind == "async_batch_jobs":
            rows = tuple(data.get("rows") or ())
            if rows:
                self.snapshot = AsyncSnapshot(rows=rows, model=self.snapshot.model, supported=self.snapshot.supported,
                                              status_text=self.snapshot.status_text, jobs_file=self.snapshot.jobs_file)
                self._render_jobs()
                self._push(self.jobs_column, self.selection_text)

    # ---- plumbing -------------------------------------------------------------------------------------

    def _push(self, *controls: Any) -> None:
        for control in controls:
            if control is None:
                continue
            try:
                control.update()
            except Exception:
                pass
