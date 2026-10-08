"""Job detail (``/jobs/<jid>``, UI_SPEC §1.8; also the target of ``/job/<jid>`` notification links).

* Header: title, origin link, state chip, progress bar (determinate once
  ``ProgressWatcher`` knows the chapter total), "Ch 12/80 · 3 in flight ·
  12:41", elapsed and ETA (UI-only moving average of chapter completions),
  the error for failed jobs and the pending question ("Waiting for your
  glossary decision", answered on the chat's approval card).
* Buttons: Stop (graceful, then force; same state words as Send) · Resume /
  Retry · Files (output folder in the file browser) · Progress (the output folder in the
  Progress manager, U9) · Open origin; a finished translation with failed / QA-failed chapters
  shows the chip "N QA failed" (Book › Chapters filtered on failures for a Library book, else
  the Progress manager over the job's workspace).
* Requests (N): request cards from the shared ``direct_text_stream`` request
  model fed by the job log (``JobService.request_segments``), in its order
  (spine order for attachments). A placeholder line until the build has it.
* Log: ``LogConsole`` over the job's ``LogBuffer`` (live), or the tail of
  ``<logs>/jobs/<id>.log`` for older jobs; Copy and Share log.
* Outputs: one chip per output file -> Share / Save to… / Save to Downloads /
  Show in Files (``FileBridge``).

The screen only binds to ``JobService`` snapshots; it never derives a status.
"""

from __future__ import annotations

import logging
import os
import time
from collections import deque
from typing import Any, Callable, Mapping, Optional, Sequence

import flet as ft

from glossarion_mobile.services.dispatcher import LogBuffer
from glossarion_mobile.services.jobs import (
    JobService,
    JobSnapshot,
    JobState,
    format_duration,
    issue_label,
    kind_icon,
    progress_line,
    status_key,
)
from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.components.action_sheet import ActionItem, ActionSheet
from glossarion_mobile.ui.components.empty_state import EmptyState
from glossarion_mobile.ui.components.log_console import LogConsole
from glossarion_mobile.ui.components.section_card import SectionCard
from glossarion_mobile.ui.components.status import StatusChip
from glossarion_mobile.ui.router import RouteMatch
from glossarion_mobile.ui.screens.base import Screen
from glossarion_mobile.ui.theme import HIT_TARGET, icon_data

__all__ = ["EtaEstimator", "JobDetailScreen", "RequestCardData", "origin_route", "request_card_data"]

log = logging.getLogger("glossarion.jobs")

MAX_REQUEST_CARDS = 60
PREVIEW_CHARS = 280
REQUEST_REFRESH_S = 1.0


class EtaEstimator:
    """Moving-average ETA from (time, completed) samples; None until two samples moved."""

    def __init__(self, window: int = 8) -> None:
        self.samples: deque = deque(maxlen=max(2, window))

    def update(self, completed: int, total: Optional[int], now: Optional[float] = None) -> Optional[float]:
        now = time.time() if now is None else now
        if not self.samples or self.samples[-1][1] != completed:
            self.samples.append((now, completed))
        if not total or len(self.samples) < 2:
            return None
        (t0, c0), (t1, c1) = self.samples[0], self.samples[-1]
        if c1 <= c0 or t1 <= t0:
            return None
        rate = (c1 - c0) / (t1 - t0)
        remaining = max(0, total - completed)
        return remaining / rate if rate > 0 else None


def origin_route(snap: JobSnapshot) -> Optional[tuple]:
    """(route name, params, label) for the job's origin, when it has one."""
    origin = snap.spec.origin or {}
    kind = origin.get("type")
    label = str(origin.get("label") or "")
    if kind == "chat" and origin.get("cid") is not None:
        cid = str(origin["cid"])
        return ("home", None, label or "Chat") if cid == "1" else ("chat", {"cid": cid}, label or "Chat")
    if kind == "library" and origin.get("bid"):
        return ("library.book", {"bid": str(origin["bid"])}, label or "Library")
    return None


class RequestCardData:
    """What one request card shows, read tolerantly from a ``direct_text_stream`` segment."""

    __slots__ = ("label", "phase", "tokens", "preview", "key")

    def __init__(self, label: str, phase: str, tokens_text: str, preview: str, key: str) -> None:
        self.label = label
        self.phase = phase
        self.tokens = tokens_text
        self.preview = preview
        self.key = key


def _field(segment: Any, *names: str) -> Any:
    for name in names:
        value = segment.get(name) if isinstance(segment, Mapping) else getattr(segment, name, None)
        if callable(value) and not isinstance(value, (str, bytes)):
            try:
                value = value()
            except TypeError:
                continue
        if value not in (None, ""):
            return value
    return None


def request_card_data(segment: Any, index: int) -> RequestCardData:
    label = _field(segment, "label", "title", "request_label", "name") or f"Request {index + 1}"
    phase = _field(segment, "phase", "status", "state") or ""
    text = _field(segment, "preview", "text", "content", "output") or ""
    text = str(text)
    preview = text[-PREVIEW_CHARS:] if len(text) > PREVIEW_CHARS else text
    parts = []
    for name, suffix in (("thinking_tokens", " thinking"), ("text_tokens", " text"), ("tokens", " tokens")):
        value = _field(segment, name)
        if isinstance(value, (int, float)) and value:
            parts.append(f"{int(value):,}{suffix}")
    key = str(_field(segment, "id", "key", "request_id") or index)
    return RequestCardData(str(label), str(phase), " · ".join(parts), preview, key)


class JobDetailScreen(Screen):
    title = "Job"

    def __init__(
        self,
        match: Optional[RouteMatch],
        *,
        service: JobService,
        files: Any = None,
        dispatcher: Any = None,
        page: Any = None,
        navigate: Optional[Callable[..., Any]] = None,
        notify: Optional[Callable[..., Any]] = None,
        copy_handler: Optional[Callable[[str], Any]] = None,
        file_ref: Optional[Callable[[str], str]] = None,
        tablet: bool = False,
        dark: bool = False,
        open_progress: Optional[Callable[[str], Any]] = None,
        roots: Optional[Callable[[], Mapping[str, str]]] = None,
        chat_workspace: Optional[Callable[[Any], str]] = None,
    ) -> None:
        super().__init__(match)
        self.open_progress = open_progress  # (output folder) -> the Progress manager over it
        self.roots = roots  # () -> the file browser roots (JobsFeature.file_roots)
        self.chat_workspace = chat_workspace  # (snapshot) -> a chat job's persisted Attachments/<stem> folder
        self._files_key: Any = None
        self._files_target: Optional[tuple] = None
        self.service = service
        self.files = files
        self.dispatcher = dispatcher
        self.page = page
        self.navigate = navigate
        self.notify = notify
        self.copy_handler = copy_handler
        self.file_ref = file_ref
        self.tablet = tablet
        self.dark = dark
        self.job_id = (match.params.get("jid") if match is not None else None) or ""
        self.snap: Optional[JobSnapshot] = None
        self.eta = EtaEstimator()
        self._unsubs: list[Callable[[], None]] = []
        self._last_requests = 0.0
        self._log_buffer: Optional[LogBuffer] = None
        known = service.snapshot(self.job_id) if self.job_id else None
        self.title = (known.title if known is not None else "") or "Job"  # the app bar is built first

    # ---- body -----------------------------------------------------------------------------

    def build_body(self) -> ft.Control:
        self.snap = self.service.snapshot(self.job_id)
        if self.snap is None:
            return EmptyState(icon="WORK_HISTORY", title="Job not found",
                              body="It may have been cleared from the finished list.", key="job-missing")
        self.title = self.snap.title or "Job"
        self.kind_icon = ft.Icon(icon_data(kind_icon(self.snap.kind)), color=ft.Colors.PRIMARY)
        self.title_text = ft.Text(self.snap.title, theme_style=ft.TextThemeStyle.TITLE_MEDIUM,
                                  weight=ft.FontWeight.W_600, selectable=True)
        self.origin_button = ft.TextButton(content="", on_click=self._open_origin, visible=False, key="job-origin")
        self.state_chip = StatusChip(status=status_key(self.snap), text=self.snap.state_label, dark=self.dark)
        self.progress_bar = ft.ProgressBar(value=None, visible=False, border_radius=4)
        self.progress_text = ft.Text("", theme_style=ft.TextThemeStyle.BODY_SMALL)
        self.time_text = ft.Text("", theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT)
        self.error_text = ft.Text("", color=ft.Colors.ERROR, selectable=True, visible=False)
        self.question_text = ft.Text("", visible=False)
        self.stop_button = ft.FilledTonalButton(content="Stop", icon=ft.Icons.STOP_CIRCLE, on_click=self._on_stop,
                                                key="job-stop")
        self.resume_button = ft.FilledButton(content="Resume", icon=ft.Icons.PLAY_ARROW, on_click=self._on_resume,
                                             key="job-resume")
        self.files_button = ft.OutlinedButton(content="Files", icon=ft.Icons.FOLDER_OPEN, on_click=self._on_files,
                                              key="job-files")
        self.progress_button = ft.OutlinedButton(content="Progress", icon=ft.Icons.TIMELINE, key="job-progress",
                                                 on_click=lambda e: self._on_progress(), visible=False)
        # U9: a book job's glossary review (the Library review gate) can be answered here too
        self.review_button = ft.FilledTonalButton(content="Review glossary", icon=ft.Icons.RATE_REVIEW,
                                                  on_click=lambda e: self.open_review(), visible=False,
                                                  key="job-review")
        self.review_sheet: Any = None
        self.issue_chip = ft.Chip(label=ft.Text(""), leading=ft.Icon(ft.Icons.HOURGLASS_TOP, size=16), visible=False,
                                  key="job-issue")
        self.qa_chip = ft.Chip(label=ft.Text(""), leading=ft.Icon(ft.Icons.ERROR_OUTLINE, color=ft.Colors.ERROR, size=16),
                               on_click=lambda e: self._on_progress(failed=True), visible=False, key="job-qa-failed")
        header = SectionCard(
            title="Job",
            icon=kind_icon(self.snap.kind),
            children=[
                ft.Row([self.title_text], wrap=True),
                ft.Row([self.state_chip, self.origin_button], wrap=True, spacing=8),
                self.progress_bar,
                self.progress_text,
                self.time_text,
                self.question_text,
                self.error_text,
                self.issue_chip,
                self.qa_chip,
                ft.Row([self.review_button, self.stop_button, self.resume_button, self.files_button,
                        self.progress_button], wrap=True, spacing=8, run_spacing=8),
            ],
            key="job-header",
        )
        self.requests_title = ft.Text("Requests", theme_style=ft.TextThemeStyle.TITLE_SMALL)
        self.requests_column = ft.Column([], spacing=6, tight=True)
        self.requests_tile = ft.ExpansionTile(
            title=self.requests_title,
            controls=[self.requests_column],
            expanded=self.snap.is_active,
            controls_padding=ft.Padding.only(left=8, right=8, bottom=8),
            key="job-requests",
        )
        self.outputs_row = ft.Row([], wrap=True, spacing=8, run_spacing=8)
        self.outputs_card = SectionCard(title="Outputs", icon="FOLDER", children=[self.outputs_row],
                                        key="job-outputs")
        self.console = LogConsole(
            buffer=self._buffer(),
            dispatcher=self.dispatcher,
            copy_handler=self.copy_handler,
            list_height=360,
            backlog=2000,
        )
        log_card = SectionCard(
            title="Log",
            icon="TERMINAL",
            trailing=ft.IconButton(icon=ft.Icons.SHARE, tooltip="Share log", on_click=self._on_share_log,
                                   size_constraints=HIT_TARGET),
            children=[self.console],
            key="job-log",
        )
        self._apply(self.snap)
        self._refresh_requests(force=True)
        return ft.ListView(
            controls=[header, self.outputs_card, ft.Container(content=self.requests_tile,
                                                               bgcolor=ft.Colors.SURFACE_CONTAINER_LOW,
                                                               border_radius=tokens.RADII["card"]), log_card],
            spacing=tokens.SPACING["md"],
            padding=ft.Padding.all(tokens.SPACING["md"]),
            expand=True,
        )

    def _buffer(self) -> LogBuffer:
        buffer = self.service.log_buffer(self.job_id)
        if buffer is None:
            buffer = LogBuffer(5000, name=f"job-file:{self.job_id}")
            buffer.extend(self.service.read_log_tail(self.job_id))
        self._log_buffer = buffer
        return buffer

    # ---- state ------------------------------------------------------------------------------

    def _apply(self, snap: JobSnapshot) -> None:
        self.snap = snap
        self.state_chip.status = status_key(snap)
        self.state_chip.text = snap.state_label
        origin = origin_route(snap)
        self.origin_button.visible = origin is not None
        if origin is not None:
            self.origin_button.content = f"Open {origin[2]}"
        fraction = snap.progress.fraction
        self.progress_bar.visible = snap.is_active or fraction is not None
        self.progress_bar.value = fraction if (fraction is not None or not snap.is_active) else None
        line = progress_line(snap)
        if not line and snap.progress.total:
            line = f"Ch {snap.progress.completed}/{snap.progress.total}"
        self.progress_text.value = line
        self.progress_text.visible = bool(line)
        elapsed = snap.elapsed()
        times = []
        if elapsed is not None:
            times.append(f"Elapsed {format_duration(elapsed)}")
        if snap.is_active:
            eta = self.eta.update(snap.progress.completed, snap.progress.total)
            if eta is not None:
                times.append(f"ETA {max(1, int(round(eta / 60)))} min")
        self.time_text.value = " · ".join(times)
        self.time_text.visible = bool(times)
        self.error_text.value = snap.error or ""
        self.error_text.visible = bool(snap.error) and snap.state is JobState.FAILED
        from glossarion_mobile.services.notifications import chat_of
        from glossarion_mobile.ui.chat.cards import glossary_question

        reviewable = glossary_question(snap) is not None and chat_of(snap) is None
        self.review_button.visible = reviewable
        self.question_text.value = ("Waiting for your glossary decision — review it to continue" if reviewable else
                                    "Waiting for your glossary decision — answer it in the chat") if snap.question else ""
        self.question_text.color = ft.Colors.TERTIARY
        self.question_text.visible = bool(snap.question)
        active = snap.is_active
        self.stop_button.visible = active or snap.state is JobState.QUEUED
        if snap.state is JobState.QUEUED:
            self.stop_button.content, self.stop_button.icon = "Cancel", ft.Icons.CLOSE
        elif snap.state is JobState.STOPPING:
            self.stop_button.content, self.stop_button.icon = "Force stop", ft.Icons.HOURGLASS_BOTTOM
        else:
            self.stop_button.content, self.stop_button.icon = "Stop", ft.Icons.STOP_CIRCLE
        self.stop_button.disabled = snap.state is JobState.FORCE_STOPPING
        can_resume = snap.spec.resumable and snap.state in (JobState.INTERRUPTED, JobState.CANCELLED, JobState.FAILED) \
            and snap.started is not None and snap.resolution is None
        self.resume_button.visible = can_resume
        self.resume_button.content = "Retry" if snap.state is JobState.FAILED else "Resume"
        folder = snap.output_dir or next(iter(snap.output_dirs.values()), None)
        self.files_button.visible = (self.navigate is not None and self.file_ref is not None
                                     and self.files_target(snap) is not None)
        library_bid = self._library_bid(snap)
        self.progress_button.visible = bool(folder) and (self.open_progress is not None or library_bid is not None)
        failed = int(getattr(snap.progress, "failed", 0) or 0)
        self.qa_chip.visible = failed > 0 and not snap.is_active and (folder is not None or library_bid is not None)
        self.qa_chip.label = ft.Text(f"{failed} QA failed")
        active_issue = snap.active_issue() if hasattr(snap, "active_issue") else None
        label = issue_label(active_issue) if active_issue else ""
        self.issue_chip.label = ft.Text(label)
        self.issue_chip.visible = bool(label)
        self.outputs_row.controls = [
            ft.Chip(label=ft.Text(os.path.basename(path)), leading=ft.Icon(ft.Icons.INSERT_DRIVE_FILE, size=16),
                    on_click=lambda e, p=path: self._on_output(p), key=f"job-output-{i}")
            for i, path in enumerate(snap.outputs)
        ]
        self.outputs_card.visible = bool(snap.outputs)

    def _refresh_requests(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_requests < REQUEST_REFRESH_S:
            return
        self._last_requests = now
        segments = self.service.request_segments(self.job_id)
        if not segments:
            text = ("Request cards appear as API requests start." if self.service.has_request_stream(self.job_id)
                    else "Request cards appear here when the shared request classifier is in this build; "
                         "the log below shows every request.")
            self.requests_column.controls = [ft.Text(text, theme_style=ft.TextThemeStyle.BODY_SMALL,
                                                     color=ft.Colors.ON_SURFACE_VARIANT)]
            self.requests_title.value = "Requests"
            return
        cards = [request_card_data(seg, i) for i, seg in enumerate(segments)][-MAX_REQUEST_CARDS:]
        self.requests_title.value = f"Requests ({len(segments)})"
        self.requests_column.controls = [self._request_card(card) for card in cards]

    def _request_card(self, card: RequestCardData) -> ft.Control:
        header = [ft.Text(card.label, theme_style=ft.TextThemeStyle.LABEL_LARGE, expand=True, max_lines=2)]
        if card.phase:
            header.append(ft.Chip(label=ft.Text(str(card.phase).title()), visual_density=ft.VisualDensity.COMPACT))
        parts: list[ft.Control] = [ft.Row(header, spacing=6)]
        if card.tokens:
            parts.append(ft.Text(card.tokens, theme_style=ft.TextThemeStyle.LABEL_SMALL,
                                 color=ft.Colors.ON_SURFACE_VARIANT))
        if card.preview:
            parts.append(ft.Text(card.preview, max_lines=3, overflow=ft.TextOverflow.ELLIPSIS,
                                 theme_style=ft.TextThemeStyle.BODY_SMALL))
        return ft.Container(content=ft.Column(parts, spacing=4, tight=True), padding=10,
                            bgcolor=ft.Colors.SURFACE_CONTAINER, border_radius=tokens.RADII["field"],
                            key=f"request-{card.key}")

    # ---- lifecycle -----------------------------------------------------------------------------

    def did_show(self) -> None:
        if self._unsubs or self.snap is None:
            return
        self._unsubs.append(self.service.subscribe(self._on_view))
        if self.dispatcher is not None and self._log_buffer is not None and hasattr(self.dispatcher, "subscribe_log"):
            self._unsubs.append(self.dispatcher.subscribe_log(self._log_buffer, self._on_lines))

    def dispose(self) -> None:
        for unsub in self._unsubs:
            unsub()
        self._unsubs = []

    def _on_view(self, view: Any) -> None:
        snap = view.find(self.job_id)
        if snap is None:
            return
        self._apply(snap)
        self._refresh_requests(force=snap.is_terminal)
        self._mark_dirty()

    def _on_lines(self, lines: Any, gap: int) -> None:
        before = self.requests_title.value
        self._refresh_requests()
        if self.requests_title.value != before or self.requests_tile.expanded:
            self._mark_dirty()

    def _mark_dirty(self) -> None:
        body = self.body
        if body is None:
            return
        dispatcher = self.dispatcher
        try:
            if dispatcher is not None and dispatcher.on_loop_thread() and getattr(dispatcher, "bound", False):
                dispatcher.mark_dirty(body)
            elif getattr(body, "page", None) is not None:
                body.update()
        except Exception:
            pass

    # ---- actions --------------------------------------------------------------------------------

    def _notify(self, text: str) -> None:
        if self.notify is not None:
            self.notify(text)

    def _on_stop(self, e: Any = None) -> None:
        snap = self.snap
        if snap is None:
            return
        if snap.state is JobState.QUEUED:
            self.service.cancel_queued(snap.id)
            self._notify("Job cancelled")
            return
        mode = self.service.request_stop(snap.id)
        if mode == "graceful":
            self._notify("Stopping after the current request · tap again to force stop")
        elif mode in ("force", "immediate"):
            self._notify("Force stopping…")

    def open_review(self) -> Any:
        """Review glossary: the approval sheet (✏️ Edit · ✓ Yes · ■ No) for the job's pending question."""
        from glossarion_mobile.ui.chat.cards import GlossaryReviewSheet, glossary_preview, glossary_question

        snap = self.service.snapshot(self.job_id) or self.snap
        question = glossary_question(snap)
        if question is None:
            self._notify("The job is not waiting for a glossary decision")
            return None
        path = str((question.get("data") or {}).get("path") or "")
        question_id = str(question.get("id") or "")

        def answer(accepted: bool) -> None:
            self.service.answer(question_id, bool(accepted))
            self._notify("Translating with the reviewed glossary" if accepted else
                         "Translation stopped: the glossary was not accepted")

        def edit(file_path: str) -> None:
            if self.review_sheet is not None:
                self.review_sheet.close()
            if self.navigate is not None and self.file_ref is not None:
                from glossarion_mobile.ui.tools.text_editor import request_open

                fid = self.file_ref(file_path)
                request_open(fid)
                self.navigate("tools.text", {"fid": fid})

        try:
            info = glossary_preview(path)
        except Exception:
            info = None
        self.review_sheet = GlossaryReviewSheet(path=path, info=info, on_answer=answer,
                                                on_edit=edit if self.file_ref is not None else None,
                                                title=f"{snap.title} · glossary ready")
        return self.review_sheet.show(self.page)

    def _on_resume(self, e: Any = None) -> None:
        snap = self.snap
        if snap is None:
            return
        new_id = self.service.resume(snap.id)
        if new_id is None:
            self._notify("This job can no longer be resumed")
            return
        self._notify("Resumed · continues from the saved progress")
        if self.navigate is not None:
            self.navigate("jobs.detail", {"jid": new_id})

    def files_target(self, snap: Optional[JobSnapshot]) -> Optional[tuple]:
        """``(root, folder)`` the Files button opens: a chat job's persisted ``Direct Text/<chat>/Attachments/
        <stem>`` workspace (its pipeline folder lives in the run root under the data folder, outside the
        file browser's roots, and a finished run's root is deleted), else the job's output folder when a
        root contains it; None hides the button."""
        if snap is None:
            return None
        folder = snap.output_dir or next(iter(snap.output_dirs.values()), None)
        key = (snap.id, folder, str(snap.state))
        if key == self._files_key:
            return self._files_target
        target: Optional[tuple] = None
        roots = None
        if self.roots is not None:
            try:
                roots = dict(self.roots() or {})
            except Exception:
                roots = {}
        from glossarion_mobile.ui.screens.files import root_for

        origin = snap.spec.origin or {}
        if origin.get("type") == "chat" and self.chat_workspace is not None:
            try:
                workspace = self.chat_workspace(snap) or ""
            except Exception:
                log.debug("chat workspace lookup failed", exc_info=True)
                workspace = ""
            if workspace:
                root = root_for(workspace, roots, ("chats", "output")) if roots is not None else "chats"
                if root:
                    target = (root, workspace)
        if target is None and folder:
            root = root_for(str(folder), roots) if roots is not None else "output"
            if root:
                target = (root, str(folder))
        self._files_key, self._files_target = key, target
        return target

    def _on_files(self, e: Any = None) -> None:
        snap = self.snap
        if snap is None or self.navigate is None or self.file_ref is None:
            return
        target = self.files_target(snap)
        if target is not None:
            self.navigate("tools.files.folder", {"root": target[0], "fid": self.file_ref(target[1])})

    @staticmethod
    def _library_bid(snap: JobSnapshot) -> Optional[str]:
        origin = snap.spec.origin or {}
        return str(origin["bid"]) if origin.get("type") == "library" and origin.get("bid") else None

    def _on_progress(self, failed: bool = False) -> Any:
        """Progress / "N QA failed": a Library book opens Book › Chapters (filtered on failures for the
        chip, QA report's link), any other workspace the Progress manager over the output folder."""
        snap = self.snap
        if snap is None:
            return None
        bid = self._library_bid(snap)
        if bid is not None and self.navigate is not None:
            query = {"tab": "chapters", "filter": "failed"} if failed else {"tab": "chapters"}
            try:
                return self.navigate("library.book", {"bid": bid}, query)
            except TypeError:
                return self.navigate("library.book", {"bid": bid})
        folder = snap.output_dir or next(iter(snap.output_dirs.values()), None)
        if folder and self.open_progress is not None:
            return self.open_progress(str(folder))
        self._notify("No output folder for this job")
        return None

    def _open_origin(self, e: Any = None) -> None:
        if self.snap is None or self.navigate is None:
            return
        origin = origin_route(self.snap)
        if origin is not None:
            self.navigate(origin[0], origin[1])

    def _on_output(self, path: str) -> Optional[ActionSheet]:
        if self.files is None or self.page is None:
            return None
        sheet = export_sheet(self.files, path, page=self.page, notify=self.notify, tablet=self.tablet)
        sheet.show(self.page)
        return sheet

    async def _on_share_log(self, e: Any = None) -> None:
        if self.files is None:
            return
        path = self.service.job_log_path(self.job_id)
        if not os.path.isfile(path) or not await self.files.share([path], title=f"{self.title} log"):
            self._notify("No log file to share")


def export_sheet(files: Any, path: str, *, page: Any, notify: Optional[Callable[..., Any]] = None,
                 tablet: bool = False, extra: Sequence[ActionItem] = ()) -> ActionSheet:
    """ExportSheet (UI_SPEC §5.5): Share · Save to… · Save to Downloads · Show in Files."""

    def say(text: str) -> None:
        if notify is not None:
            notify(text)

    async def run(option_id: str, confirmed: bool = False) -> None:
        result = await files.export(option_id, path, confirmed=confirmed)
        if option_id == "save":
            if getattr(result, "needs_confirm", False):
                from glossarion_mobile.ui.components.dialogs import ConfirmDialog

                size_mb = getattr(result, "size", 0) / (1024 * 1024)
                ConfirmDialog(title="Save a large file?", body=f"{os.path.basename(path)} is {size_mb:.0f} MB.",
                              confirm_label="Save", on_confirm=lambda: run("save", True)).show(page)
                return
            if getattr(result, "error", None):
                say(f"Save failed: {result.error}")
            elif getattr(result, "ok", False):
                say("Saved")
        elif option_id == "downloads":
            say("Saved to Downloads/Glossarion" if result else "Saving to Downloads failed")
        elif option_id == "files" and not result:
            say("Could not open the Files app")

    items = [
        ActionItem(option.label, (lambda oid=option.id: run(oid)), icon=option.icon,
                   disabled_reason=option.disabled_reason, key=f"export-{option.id}")
        for option in files.export_options(path)
    ]
    items.extend(extra)
    return ActionSheet(items, title=os.path.basename(path), tablet=tablet)
