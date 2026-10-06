"""Jobs page (``/jobs``, UI_SPEC §1.8) and ``JobsFeature``, which wires jobs, files and intents into the app.

Jobs page
  One list with sections **Running** (at most one) · **Queued**
  (``ReorderableListView`` with drag handles; ✕ Cancel asks "Cancel job / Keep
  job") · **Interrupted** (recovered from ``jobs/active.state``: Resume ·
  Discard) · **Finished** (last 50: Open · Retry). Rows: kind icon, title,
  origin line, progress bar (running), state chip. App bar ⋯: Pause / Resume
  queue · Cancel all queued · Clear finished. Extended FAB "Pause queue" /
  "Resume queue" while jobs are queued. Empty state "No jobs yet".

JobsFeature (``await JobsFeature.install(app)`` in ``GlossarionApp.start``,
after ``SettingsFeature.install`` so the config store and Prefs exist)
  * builds ``JobService`` (``<data>/jobs``), ``JobNotifications``,
    ``BackgroundExecution`` (+ ``Wakelock``), ``FileBridge`` (Inbox,
    Library/Raw) and ``IntentRouter``; exposes them as ``app.jobs`` (this
    feature), ``app.job_service``, ``app.files``, ``app.intents``;
  * wraps the shell's screen factory for ``/jobs``, ``/jobs/<jid>`` (and the
    ``/job/<jid>`` notification alias) and ``/tools/files/<root>[/<fid>]``;
  * binds ``AppState.job_strip`` / ``jobs_badge`` to JobService (Done / Failed
    for 10 s after a job ends) and the global strip's Open / Stop;
  * forwards native events: share -> IntentRouter, foreground / background
    task -> BackgroundExecution, notification taps -> routes / Share / Resume;
  * checkpoints ``active.state`` when the app goes to the background;
  * runs ``JobService.recover()`` on a worker thread and shows the launch
    banner "N interrupted jobs · Resume / Review".

``submit(spec)`` is what Send / Start call: it runs the background preparation
(Android permissions, iOS continued-processing request, which must happen on
the user's tap) and then ``JobService.submit``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Callable, Optional

import flet as ft

from glossarion_mobile.services.background import IOS_BACKGROUND_NOTICE, BackgroundExecution
from glossarion_mobile.services.files import FileBridge
from glossarion_mobile.services.intents import IntentImport, IntentRouter
from glossarion_mobile.services.jobs import (
    JobService,
    JobSnapshot,
    JobSpec,
    JobState,
    JobsView,
    kind_icon,
    progress_line,
    status_key,
    strip_model_for,
)
from glossarion_mobile.services.notifications import ACTION_RESUME, ACTION_SHARE, JobNotifications, chat_of
from glossarion_mobile.services.wakelock import SharedWakelock
from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.components.action_sheet import ActionItem, ActionSheet
from glossarion_mobile.ui.components.dialogs import ConfirmDialog
from glossarion_mobile.ui.components.empty_state import EmptyState
from glossarion_mobile.ui.components.status import StatusChip
from glossarion_mobile.ui.router import RouteMatch
from glossarion_mobile.ui.screens.base import Screen
from glossarion_mobile.ui.screens.job_detail import JobDetailScreen
from glossarion_mobile.ui.theme import HIT_TARGET, icon_data

__all__ = ["DONE_STRIP_SECONDS", "JobsFeature", "JobsScreen", "SCREEN_ROUTES"]

log = logging.getLogger("glossarion.jobs.ui")

SCREEN_ROUTES = ("jobs", "jobs.detail", "tools.files", "tools.files.folder", "tools.text")
DONE_STRIP_SECONDS = 10.0
QUEUE_ROW_HEIGHT = 76
QUEUE_VISIBLE_ROWS = 6
_FLUSH_STATES = ("inactive", "hide", "pause", "detach")


def _platform_name(page: Any) -> str:
    value = getattr(getattr(page, "platform", None), "value", getattr(page, "platform", None))
    value = str(value or "").lower()
    if getattr(page, "web", False):
        return "desktop"
    return value if value in ("android", "ios") else "desktop"


class JobsScreen(Screen):
    title = "Jobs"

    def __init__(
        self,
        match: Optional[RouteMatch],
        *,
        service: JobService,
        page: Any = None,
        dispatcher: Any = None,
        navigate: Optional[Callable[..., Any]] = None,
        notify: Optional[Callable[..., Any]] = None,
        tablet: bool = False,
        dark: bool = False,
        platform: str = "desktop",
    ) -> None:
        super().__init__(match)
        self.service = service
        self.page = page
        self.dispatcher = dispatcher
        self.navigate = navigate
        self.notify = notify
        self.tablet = tablet
        self.dark = dark
        self.platform = platform
        self.view: Optional[JobsView] = None
        self._unsubs: list[Callable[[], None]] = []
        self.pause_item = ft.PopupMenuItem(content="Pause queue", icon=ft.Icons.PAUSE, on_click=self._toggle_queue,
                                           key="jobs-menu-pause")
        self.cancel_all_item = ft.PopupMenuItem(content="Cancel all queued", icon=ft.Icons.CLEAR_ALL,
                                                on_click=self._cancel_all, key="jobs-menu-cancel-all")
        self.clear_item = ft.PopupMenuItem(content="Clear finished", icon=ft.Icons.DELETE_SWEEP,
                                           on_click=self._clear_finished, key="jobs-menu-clear")

    # ---- body --------------------------------------------------------------------------

    def actions(self) -> list[ft.Control]:
        return [ft.PopupMenuButton(icon=ft.Icons.MORE_VERT, items=[self.pause_item, self.cancel_all_item,
                                                                   self.clear_item], tooltip="More",
                                   key="jobs-menu")]

    def build_body(self) -> ft.Control:
        self.list_view = ft.ListView(controls=[], expand=True, spacing=6,
                                     padding=ft.Padding.only(left=12, right=12, top=8, bottom=96))
        self.fab = ft.FloatingActionButton(icon=ft.Icons.PAUSE, content="Pause queue", on_click=self._toggle_queue,
                                           visible=False, key="jobs-fab")
        self._render(self.service.view())
        return ft.Stack([self.list_view, ft.Container(content=self.fab, right=16, bottom=16)], expand=True)

    def _section(self, title: str, count: Optional[int] = None) -> ft.Control:
        text = f"{title} ({count})" if count is not None else title
        return ft.Container(
            content=ft.Text(text, theme_style=ft.TextThemeStyle.TITLE_SMALL, color=ft.Colors.PRIMARY,
                            weight=ft.FontWeight.W_600),
            padding=ft.Padding.only(left=4, top=12, bottom=2),
        )

    def _origin_line(self, snap: JobSnapshot) -> str:
        origin = snap.spec.origin or {}
        return str(origin.get("label") or "")

    def _row(self, snap: JobSnapshot, section: str) -> ft.Control:
        chip = StatusChip(status=status_key(snap), text=snap.state_label, dark=self.dark)
        lines: list[ft.Control] = [
            ft.Text(snap.title, theme_style=ft.TextThemeStyle.TITLE_SMALL, max_lines=2,
                    overflow=ft.TextOverflow.ELLIPSIS)
        ]
        origin = self._origin_line(snap)
        if origin:
            lines.append(ft.Text(origin, theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT))
        detail = progress_line(snap)
        if detail and section in ("running", "queued"):
            lines.append(ft.Text(detail, theme_style=ft.TextThemeStyle.BODY_SMALL))
        elif snap.error and snap.state is JobState.FAILED:
            lines.append(ft.Text(snap.error, theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ERROR,
                                 max_lines=2, overflow=ft.TextOverflow.ELLIPSIS))
        elif snap.progress.total:
            lines.append(ft.Text(f"{snap.progress.completed}/{snap.progress.total} chapters",
                                 theme_style=ft.TextThemeStyle.BODY_SMALL))
        top = ft.Row(
            [
                ft.Icon(icon_data(kind_icon(snap.kind)), color=ft.Colors.PRIMARY),
                ft.Column(lines, spacing=2, tight=True, expand=True),
                chip,
            ],
            spacing=10,
            vertical_alignment=ft.CrossAxisAlignment.START,
        )
        parts: list[ft.Control] = [top]
        if section == "running":
            parts.append(ft.ProgressBar(value=snap.progress.fraction, border_radius=4))
        buttons = self._row_actions(snap, section)
        if buttons:
            parts.append(ft.Row(buttons, alignment=ft.MainAxisAlignment.END, spacing=4, wrap=True))
        return ft.Container(
            content=ft.Column(parts, spacing=6, tight=True),
            padding=ft.Padding.only(left=12, right=8, top=10, bottom=6),
            bgcolor=ft.Colors.SURFACE_CONTAINER_LOW,
            border_radius=tokens.RADII["card"],
            on_click=lambda e, s=snap: self._open(s),
            ink=True,
            key=f"job-row-{snap.id}",
        )

    def _row_actions(self, snap: JobSnapshot, section: str) -> list[ft.Control]:
        if section == "running":
            label = "Force stop" if snap.state is JobState.STOPPING else "Stop"
            return [ft.TextButton(content=label, icon=ft.Icons.STOP_CIRCLE, on_click=lambda e: self._stop(snap),
                                  disabled=snap.state is JobState.FORCE_STOPPING, key=f"job-stop-{snap.id}")]
        if section == "queued":
            return [ft.TextButton(content="Cancel", icon=ft.Icons.CLOSE, on_click=lambda e: self._cancel(snap),
                                  key=f"job-cancel-{snap.id}")]
        if section == "interrupted":
            return [
                ft.TextButton(content="Discard", on_click=lambda e: self._discard(snap), key=f"job-discard-{snap.id}"),
                ft.FilledTonalButton(content="Resume", icon=ft.Icons.PLAY_ARROW, on_click=lambda e: self._resume(snap),
                                     key=f"job-resume-{snap.id}"),
            ]
        out: list[ft.Control] = []
        if snap.state is JobState.FAILED and snap.spec.resumable:
            out.append(ft.TextButton(content="Retry", icon=ft.Icons.REFRESH, on_click=lambda e: self._resume(snap),
                                     key=f"job-retry-{snap.id}"))
        elif snap.stopped and snap.spec.resumable:
            out.append(ft.TextButton(content="Resume", icon=ft.Icons.PLAY_ARROW, on_click=lambda e: self._resume(snap),
                                     key=f"job-resume-{snap.id}"))
        if snap.state is JobState.DONE:
            out.append(ft.TextButton(content="Open", icon=ft.Icons.OPEN_IN_NEW, on_click=lambda e: self._open(snap),
                                     key=f"job-open-{snap.id}"))
        return out

    def _render(self, view: JobsView) -> None:
        self.view = view
        controls: list[ft.Control] = []
        if self.platform == "ios":
            controls.append(ft.Container(
                content=ft.Row([ft.Icon(ft.Icons.INFO_OUTLINE, color=ft.Colors.TERTIARY, size=18),
                                ft.Text(IOS_BACKGROUND_NOTICE, theme_style=ft.TextThemeStyle.BODY_SMALL, expand=True)],
                               spacing=8, vertical_alignment=ft.CrossAxisAlignment.START),
                padding=10, bgcolor=ft.Colors.SURFACE_CONTAINER, border_radius=tokens.RADII["card"],
                key="jobs-ios-notice",
            ))
        if view.active is None and not view.queue and not view.interrupted and not view.history:
            controls.append(EmptyState(
                icon="FOLDER_OPEN",
                title="No jobs yet",
                body="Translations, glossary extractions, scans and compiles you start appear here.",
                key="jobs-empty",
            ))
        if view.active is not None:
            controls += [self._section("Running"), self._row(view.active, "running")]
        if view.queue:
            controls.append(self._section("Queued", len(view.queue)))
            self.queue_list = ft.ReorderableListView(
                controls=[self._row(snap, "queued") for snap in view.queue],
                on_reorder=self._on_reorder,
                height=QUEUE_ROW_HEIGHT * min(len(view.queue), QUEUE_VISIBLE_ROWS),
                key="jobs-queue",
            )
            controls.append(self.queue_list)
            if view.paused:
                controls.append(ft.Text("Queue paused: the next job starts when you resume it.",
                                        theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.TERTIARY))
        if view.interrupted:
            controls.append(self._section("Interrupted", len(view.interrupted)))
            controls += [self._row(snap, "interrupted") for snap in view.interrupted]
        if view.history:
            controls.append(self._section("Finished", len(view.history)))
            controls += [self._row(snap, "finished") for snap in view.history]
        self.list_view.controls = controls
        paused = view.paused
        self.fab.visible = bool(view.queue)
        self.fab.content = "Resume queue" if paused else "Pause queue"
        self.fab.icon = ft.Icons.PLAY_ARROW if paused else ft.Icons.PAUSE
        self.pause_item.content = "Resume queue" if paused else "Pause queue"
        self.pause_item.icon = ft.Icons.PLAY_ARROW if paused else ft.Icons.PAUSE
        self.cancel_all_item.disabled = not view.queue
        self.clear_item.disabled = not view.history

    # ---- lifecycle ------------------------------------------------------------------------

    def did_show(self) -> None:
        if not self._unsubs:
            self._unsubs.append(self.service.subscribe(self._on_view))

    def dispose(self) -> None:
        for unsub in self._unsubs:
            unsub()
        self._unsubs = []

    def _on_view(self, view: JobsView) -> None:
        if getattr(self, "list_view", None) is None:
            return
        self._render(view)
        body = self.body
        if body is None:
            return
        try:
            dispatcher = self.dispatcher
            if dispatcher is not None and getattr(dispatcher, "bound", False) and dispatcher.on_loop_thread():
                dispatcher.mark_dirty(body)
            elif getattr(body, "page", None) is not None:
                body.update()
        except Exception:
            pass

    # ---- actions ---------------------------------------------------------------------------

    def _say(self, text: str) -> None:
        if self.notify is not None:
            self.notify(text)

    def _open(self, snap: JobSnapshot) -> None:
        if self.navigate is not None:
            self.navigate("jobs.detail", {"jid": snap.id})

    def _stop(self, snap: JobSnapshot) -> None:
        mode = self.service.request_stop(snap.id)
        if mode == "graceful":
            self._say("Stopping after the current request · tap again to force stop")

    def _cancel(self, snap: JobSnapshot) -> ConfirmDialog:
        dialog = ConfirmDialog(title="Cancel job?", body=snap.title, confirm_label="Cancel job",
                               cancel_label="Keep job", destructive=True,
                               on_confirm=lambda: self.service.cancel_queued(snap.id))
        if self.page is not None:
            dialog.show(self.page)
        return dialog

    def _resume(self, snap: JobSnapshot) -> Optional[str]:
        new_id = self.service.resume(snap.id)
        self._say("Resumed · continues from the saved progress" if new_id else "This job can no longer be resumed")
        return new_id

    def _discard(self, snap: JobSnapshot) -> None:
        self.service.discard(snap.id)

    def _on_reorder(self, e: Any) -> None:
        view = self.view
        old, new = getattr(e, "old_index", None), getattr(e, "new_index", None)
        if view is None or old is None or new is None or not (0 <= old < len(view.queue)):
            return
        if new > old:
            new -= 1  # Flutter reports the index before removal
        self.service.move_queued(view.queue[old].id, new)

    def _toggle_queue(self, e: Any = None) -> None:
        if self.service.paused:
            self.service.resume_queue()
        else:
            self.service.pause_queue()

    def _cancel_all(self, e: Any = None) -> None:
        count = self.service.cancel_all_queued()
        self._say(f"Cancelled {count} queued job{'s' if count != 1 else ''}")

    def _clear_finished(self, e: Any = None) -> None:
        self.service.clear_finished()


# ---------------------------------------------------------------------------
# Feature wiring
# ---------------------------------------------------------------------------


class JobsFeature:
    def __init__(self, app: Any, *, service: Optional[JobService] = None) -> None:
        self.app = app
        self.page = getattr(app, "page", None)
        self.dispatcher = getattr(app, "dispatcher", None)
        self.state = getattr(app, "state", None)
        self.prefs = getattr(app, "prefs", None)
        self.platform = _platform_name(self.page)
        paths = getattr(app, "paths", None)
        self.paths = paths
        data = str(getattr(paths, "data", None) or os.getcwd())
        logs = str(getattr(paths, "logs", None) or os.path.join(data, "logs"))
        self.inbox_dir = os.path.join(data, "Inbox")
        self.service = service or JobService(
            jobs_dir=os.path.join(data, "jobs"),
            logs_dir=logs,
            data_dir=data,
            config_store=getattr(app, "config_store", None),
            dispatcher=self.dispatcher,
        )
        native = getattr(app, "native", None)
        self.notifications = JobNotifications(native, platform=self.platform)
        # One reference-counted owner of the platform wakelock: the job service and the
        # Reader ("Keep screen on") each hold it through their own holder (U5 review).
        raw_wakelock = self._make_wakelock()
        self.wakelock_owner = SharedWakelock(raw_wakelock) if raw_wakelock is not None else None
        self.wakelock = self.wakelock_owner.holder("jobs") if self.wakelock_owner is not None else None
        self.background = BackgroundExecution(
            native,
            platform=self.platform,
            jobs=self.service,
            notifications=self.notifications,
            prefs=self.prefs,
            wakelock=self.wakelock,
            confirm=self._confirm,
            navigate_route=self._navigate_route,
        )
        cache_dirs = [str(p) for p in (getattr(paths, "cache", None), getattr(paths, "temp", None)) if p]
        self.files = FileBridge(
            inbox_dir=self.inbox_dir,
            cache_dirs=cache_dirs,
            platform=self.platform,
            url_launcher=getattr(app, "url_launcher", None),
            native=native,
            run_io=self.run_io,
            files_visible_root=str(getattr(paths, "docs", "")) if self.platform == "ios" and paths else None,
        )
        self.intents = IntentRouter(files=self.files, native=native, present=self._present_imports,
                                    notify=self._notify)
        self._fallback_factory: Optional[Callable[[RouteMatch], Any]] = None
        self._unsubs: list[Callable[[], None]] = []
        self._ended: Optional[tuple] = None  # (snapshot, hide_at)
        self._ended_handle: Any = None
        self.banner: Any = None
        self.recovered: list = []
        self.screens_built: list[str] = []

    def __getattr__(self, name: str) -> Any:
        """Everything else is the JobService API (``snapshot``, ``subscribe``, ``request_stop``,
        ``on_transition``, ``on_question``, ``answer``, ``log_buffer``, ``request_segments``, ...),
        so ``app.jobs`` can be used wherever a JobService is expected; ``submit`` adds the
        background preparation."""
        if name.startswith("__") or name == "service":
            raise AttributeError(name)
        return getattr(self.service, name)

    # ---- install -----------------------------------------------------------------------------

    @classmethod
    async def install(cls, app: Any) -> "JobsFeature":
        feature = cls(app)
        feature.attach(app)
        feature.spawn(feature.recover())
        return feature

    def attach(self, app: Any) -> None:
        app.jobs = self
        app.job_service = self.service
        app.files = self.files
        app.intents = self.intents
        shell = getattr(app, "shell", None)
        if shell is not None and self._fallback_factory is None:
            self._fallback_factory = shell.screen_factory
            shell.screen_factory = self.screen_factory
            strip = getattr(shell, "global_strip", None)
            if strip is not None:
                strip.on_open = self.open_strip_job
                strip.on_stop = self.stop_from_strip
        native = getattr(app, "native", None)
        if native is not None and hasattr(native, "add_listener"):
            native.add_listener("share", self._on_share_event)
            native.add_listener("foreground", self.background.on_foreground_event)
            native.add_listener("background_task", self.background.on_background_task_event)
            native.add_listener("notification", self._on_notification)
        self._wrap_lifecycle()
        if not self._unsubs:
            self._unsubs.append(self.service.subscribe(self._on_view))
            self._unsubs.append(self.service.on_transition(self._on_transition))
            self._unsubs.append(self.service.on_question(self._on_question))
            self._unsubs.append(self.service.on_event(self._on_job_event))
        self._on_view(self.service.view())

    def close(self) -> None:
        for unsub in self._unsubs:
            unsub()
        self._unsubs = []
        self.service.close()

    # ---- helpers ------------------------------------------------------------------------------

    def _make_wakelock(self) -> Any:
        if self.platform not in ("android", "ios"):
            return None
        try:
            return ft.Wakelock()  # page service: this reference keeps it registered
        except Exception as exc:
            log.info("Wakelock unavailable: %s", exc)
            return None

    def spawn(self, coro: Any) -> Any:
        dispatcher = self.dispatcher
        if dispatcher is not None and getattr(dispatcher, "bound", False):
            return dispatcher.spawn(coro)
        return asyncio.ensure_future(coro)

    async def run_io(self, fn: Callable[..., Any], *args: Any) -> Any:
        dispatcher = self.dispatcher
        if dispatcher is not None and getattr(dispatcher, "bound", False):
            return await dispatcher.run_in_thread(fn, *args, name="gl-files")
        return await asyncio.to_thread(fn, *args)

    def _notify(self, message: str, action_label: Optional[str] = None, on_action: Any = None) -> Any:
        notify = getattr(self.app, "notify", None)
        return notify(message, action_label, on_action) if notify is not None else None

    def _navigate(self, name: str, params: Optional[dict] = None) -> None:
        navigate = getattr(self.app, "navigate_to", None)
        if navigate is not None:
            navigate(name, params)

    def _navigate_route(self, route: str) -> None:
        navigate = getattr(self.app, "navigate", None)
        if navigate is not None:
            self.spawn(navigate(route))

    def _tablet(self) -> bool:
        return bool(getattr(getattr(self.app, "shell", None), "tablet", False))

    def _dark(self) -> bool:
        try:
            from glossarion_mobile.ui.theme import is_dark

            return bool(is_dark(self.page))
        except Exception:
            return False

    def file_ref(self, path: str) -> str:
        if self.prefs is None:
            raise RuntimeError("Prefs unavailable")
        return self.prefs.file_ref(path)

    def file_roots(self) -> dict:
        paths = self.paths
        output = str(getattr(paths, "output", "") or "")
        return {
            "output": output,
            "library": str(getattr(paths, "library", "") or ""),
            "inbox": self.inbox_dir,
            "chats": os.path.join(output, "Direct Text") if output else "",
        }

    async def _confirm(self, title: str, body: str) -> bool:
        if self.page is None:
            return True
        loop = asyncio.get_running_loop()
        answer = loop.create_future()

        def done(value: bool) -> None:
            if not answer.done():
                answer.set_result(value)

        ConfirmDialog(title=title, body=body, confirm_label="Continue", cancel_label="Not now",
                      on_confirm=lambda: done(True), on_cancel=lambda: done(False)).show(self.page)
        return await answer

    # ---- submit / stop -------------------------------------------------------------------------

    async def submit(self, spec: JobSpec, *, long_job: bool = True) -> str:
        """Send / Start: background preparation on the user's tap, then queue the job."""
        try:
            await self.background.prepare_for_run(spec, long_job=long_job)
        except Exception:
            log.exception("background preparation failed; the job still runs in the foreground")
        job_id = self.service.submit(spec)
        active = self.service.snapshot()
        if active is not None and active.id != job_id:
            self._notify(f"Queued · runs after {active.title}", "Undo", lambda: self.undo_queue(job_id))
        return job_id

    def undo_queue(self, job_id: str) -> bool:
        """The Queue snackbar's Undo (UI_SPEC §2.4): take the job out of the queue again."""
        cancelled = bool(self.service.cancel_queued(job_id))
        if not cancelled:
            self._notify("It already started; stop it from the job strip")
        return cancelled

    def stop_from_strip(self) -> Optional[str]:
        mode = self.service.request_stop()
        if mode == "graceful":
            self._notify("Stopping after the current request · tap again to force stop")
        elif mode in ("force", "immediate"):
            self._notify("Force stopping…")
        return mode

    def open_strip_job(self) -> None:
        snap = self.service.snapshot()
        if snap is None and self._ended is not None:
            snap = self._ended[0]
        if snap is not None:
            cid = chat_of(snap)
            if snap.question and cid is not None:
                # a pending glossary approval is answered on the chat's approval card
                self._navigate("chat", {"cid": cid})
                return
            if snap.question and (snap.question or {}).get("kind") == "async_batch_question":
                self._navigate("tools.async")  # Tools › Async batch answers the dialog's questions
                return
            self._navigate("jobs.detail", {"jid": snap.id})
        else:
            self._navigate("jobs")

    # ---- screens --------------------------------------------------------------------------------

    def make_screen(self, match: RouteMatch) -> Any:
        tablet, dark = self._tablet(), self._dark()
        notify = getattr(self.app, "notify", None)
        copy = getattr(self.app, "_copy_text", None)
        if match.name == "jobs":
            screen: Any = JobsScreen(match, service=self.service, page=self.page, dispatcher=self.dispatcher,
                                     navigate=self._navigate, notify=notify, tablet=tablet, dark=dark,
                                     platform=self.platform)
        elif match.name == "jobs.detail":
            screen = JobDetailScreen(match, service=self.service, files=self.files, dispatcher=self.dispatcher,
                                     page=self.page, navigate=self._navigate, notify=notify, copy_handler=copy,
                                     file_ref=self.file_ref if self.prefs is not None else None, tablet=tablet,
                                     dark=dark)
        elif match.name in ("tools.files", "tools.files.folder"):
            from glossarion_mobile.ui.screens.files import FileBrowserScreen

            prefs = self.prefs
            screen = FileBrowserScreen(match, roots=self.file_roots(), files=self.files, page=self.page,
                                       navigate=self._navigate, notify=notify,
                                       file_ref=prefs.file_ref if prefs is not None else None,
                                       resolve_ref=prefs.resolve_file_ref if prefs is not None else None,
                                       run_io=self.run_io, tablet=tablet, open_reader=self._open_reader_path,
                                       push_overlay=self._push_overlay, pop_overlay=self._pop_overlay)
        elif match.name == "tools.text":
            from glossarion_mobile.ui.tools.text_editor import TextEditorScreen

            prefs = self.prefs
            screen = TextEditorScreen(match, roots=self.file_roots(), files=self.files, page=self.page, notify=notify,
                                      resolve_ref=prefs.resolve_file_ref if prefs is not None else None,
                                      run_io=self.run_io, tablet=tablet, on_close=self._back)
        else:
            return None
        self.screens_built.append(match.name)
        return screen

    def _push_overlay(self, view: Any) -> None:
        shell = getattr(self.app, "shell", None)
        if shell is not None:
            shell.push_overlay(view)
            self._page_update()

    def _pop_overlay(self, view: Any) -> None:
        shell = getattr(self.app, "shell", None)
        if shell is not None:
            shell.pop_view(view)
            self._page_update()

    def _page_update(self) -> None:
        try:
            self.page.update()
        except Exception:
            pass

    def _back(self) -> None:
        """Leave the top screen (a screen that asked before Back, e.g. the text editor's unsaved edits)."""
        back = getattr(self.app, "back", None)
        if callable(back):
            back()

    def _open_reader_path(self, *, path: str) -> Any:
        """File browser › Open with › Reader (the Reader feature, when installed)."""
        reader = getattr(self.app, "reader", None)
        opener = getattr(reader, "open_book", None)
        if not callable(opener):
            return None
        return opener(path=path)

    def screen_factory(self, match: RouteMatch) -> Any:
        screen = self.make_screen(match)
        if screen is None:
            if self._fallback_factory is None:
                raise LookupError(f"no screen for {match.name}")
            screen = self._fallback_factory(match)
        return screen

    # ---- JobService -> strip, badge, background, notifications ---------------------------------------

    def _strip_model(self, view: JobsView) -> Any:
        if view.active is not None:
            return strip_model_for(view.active, view.queued_count)
        ended = self._ended
        if ended is not None and time.monotonic() < ended[1]:
            return strip_model_for(ended[0], view.queued_count)
        return None

    def _on_view(self, view: JobsView) -> None:
        state = self.state
        if state is not None:
            try:
                from glossarion_mobile.state.app_state import JobsBadge

                state.job_strip.set(self._strip_model(view))
                state.jobs_badge.set(JobsBadge(running=view.running_count, queued=view.queued_count))
            except Exception:
                log.exception("updating the job strip failed")
        if view.active is not None:
            self.spawn(self.background.job_progress(view.active))

    def _on_transition(self, snap: JobSnapshot, previous: Optional[JobState]) -> None:
        if snap.is_terminal and previous is not None and snap.started is not None:
            self._ended = (snap, time.monotonic() + DONE_STRIP_SECONDS)
            self._schedule_strip_clear()
        elif snap.state is JobState.STARTING:
            self._ended = None
        self.spawn(self.background.on_transition(snap, previous))

    def _schedule_strip_clear(self) -> None:
        loop = getattr(self.dispatcher, "loop", None)
        if loop is None:
            return
        if self._ended_handle is not None:
            self._ended_handle.cancel()
        self._ended_handle = loop.call_later(DONE_STRIP_SECONDS + 0.05, self._clear_ended)

    def _clear_ended(self) -> None:
        self._ended_handle = None
        if self._ended is not None and time.monotonic() >= self._ended[1]:
            self._ended = None
        self._on_view(self.service.view())

    def _on_question(self, snap: JobSnapshot, question: Any) -> None:
        if not self.background.app_visible:
            self.spawn(self.notifications.glossary_review_needed(snap))

    def _on_job_event(self, job_id: str, kind: str, data: Any) -> None:
        """Job events JobService does not consume itself (UI loop)."""
        if kind == "sign_in_required":
            # UI_SPEC §1.9: the job lost (or never had) its ChatGPT login and the backend falls
            # back to the browser login; point the user at the Accounts sign-in instead.
            self.spawn(self.notifications.sign_in_required("ChatGPT"))
            self._notify("Sign-in required for ChatGPT", "Sign in", lambda: self._navigate("settings.accounts"))

    # ---- native events -----------------------------------------------------------------------------------

    async def _on_share_event(self, event: dict) -> None:
        await self.intents.handle(event.get("items") or [], source="share")

    async def handle_initial_shared(self, items: Any) -> list:
        """Open-with at cold start (``get_initial_shared``); launch links stay with the app router."""
        return await self.intents.handle(items or [], source="initial")

    async def _on_notification(self, event: dict) -> None:
        tap = JobNotifications.parse_event(event)
        if tap is None:
            return
        if tap.action == ACTION_SHARE and tap.job_id:
            snap = self.service.snapshot(tap.job_id)
            if snap is not None and snap.outputs:
                await self.files.share(list(snap.outputs))
                return
        if tap.action == ACTION_RESUME and tap.job_id:
            if self.service.resume(tap.job_id):
                self._notify("Resumed · continues from the saved progress")
                self._navigate("jobs")
                return
        if tap.route:
            navigate = getattr(self.app, "navigate", None)
            if navigate is not None:
                await navigate(tap.route)

    def _present_imports(self, imports: list) -> Optional[ActionSheet]:
        files = [imp for imp in imports if imp.imported is not None]
        texts = [imp for imp in imports if imp.imported is None and imp.text]
        errors = [imp for imp in imports if imp.error]
        if errors and not files and not texts:
            self._notify(errors[0].error or "The shared file could not be imported")
            return None
        if texts and not files:
            compose = getattr(self.app, "prefill_composer", None)
            if compose is not None:
                compose(texts[0].text)
            else:
                self._notify("Shared text received")
            return None
        if self.page is None or not files:
            return None
        first: IntentImport = files[0]
        title = first.label if len(files) == 1 else f"{len(files)} files imported"

        def run(action_id: str) -> Any:
            async def go() -> None:
                for imp in files:
                    try:
                        await self.intents.perform(action_id, imp)
                    except Exception as exc:
                        self._notify(f"{imp.label}: {exc}")

            return go()

        items = [
            ActionItem(action.label, (lambda aid=action.id: run(aid)), icon=action.icon,
                       disabled_reason=action.disabled_reason, key=f"intent-{action.id}")
            for action in first.actions
        ]
        sheet = ActionSheet(items, title=title, subtitle="Saved to the Inbox", tablet=self._tablet())
        sheet.show(self.page)
        return sheet

    # ---- lifecycle / recovery -------------------------------------------------------------------------

    def _wrap_lifecycle(self) -> None:
        page = self.page
        if page is None:
            return
        original = getattr(page, "on_app_lifecycle_state_change", None)
        if getattr(original, "_glossarion_jobs", False):
            return

        async def on_lifecycle(e: Any) -> None:
            state = getattr(e, "state", None)
            name = str(getattr(state, "value", state))
            self.background.on_lifecycle(name)
            if name in _FLUSH_STATES:
                try:
                    self.service.checkpoint()
                except Exception:
                    log.exception("job checkpoint failed")
            if original is not None:
                result = original(e)
                if hasattr(result, "__await__"):
                    await result

        on_lifecycle._glossarion_jobs = True  # type: ignore[attr-defined]
        page.on_app_lifecycle_state_change = on_lifecycle

    async def recover(self) -> list:
        """Interrupted jobs from the previous launch, then the launch banner."""
        try:
            self.recovered = await self.run_io(self.service.recover)
        except Exception:
            log.exception("job recovery failed")
            self.recovered = []
        if any(s.id in self.service.adopted_ids for s in self.recovered):  # killed at the last launch
            self.show_launch_banner()
        return self.recovered

    def show_launch_banner(self) -> Any:
        interrupted = self.service.interrupted
        if not interrupted or self.page is None:
            return None
        count = len(interrupted)
        newest = interrupted[0]

        def close() -> None:
            if self.banner is not None and getattr(self.banner, "open", False):
                try:
                    self.page.pop_dialog()
                except Exception:
                    pass

        def resume(e: Any = None) -> None:
            close()
            if self.service.resume(newest.id):
                self._notify(f"Resumed · {newest.title}")

        def review(e: Any = None) -> None:
            close()
            self._navigate("jobs")

        self.banner = ft.Banner(
            leading=ft.Icon(ft.Icons.RESTART_ALT, color=ft.Colors.TERTIARY),
            content=ft.Text(f"{count} interrupted job{'s' if count != 1 else ''}"
                            f" · {newest.title} can continue from its saved progress"),
            actions=[
                ft.TextButton(content="Resume", on_click=resume, key="banner-resume"),
                ft.TextButton(content="Review", on_click=review, key="banner-review"),
                ft.IconButton(icon=ft.Icons.CLOSE, on_click=lambda e: close(), tooltip="Dismiss",
                              size_constraints=HIT_TARGET, key="banner-close"),
            ],
        )
        try:
            self.page.show_dialog(self.banner)
        except Exception:
            log.exception("showing the interrupted-jobs banner failed")
        return self.banner
