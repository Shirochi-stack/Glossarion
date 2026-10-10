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
    task -> BackgroundExecution, notification taps -> routes / Share / Resume /
    Accept (``route_launch_notification`` routes the tap that cold-started the
    app, once the app is ready);
  * a job's blocking question (UI_SPEC §1.9, §2.11): a ``jobs.action``
    notification ("Glossary ready: review needed" with Accept / Review, "Answer
    needed: …" for Tools › Async batch) unless the surface that answers it (the
    owning chat, the Book page, Tools › Async batch) is on screen; while the app
    is visible on another screen also a snackbar with Review / Answer. The
    notification is cancelled when the question is answered (``question_resolved``);
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
    is_glossary_question,
    kind_icon,
    progress_line,
    status_key,
    strip_model_for,
)
from glossarion_mobile.services.notifications import (
    ACTION_ACCEPT,
    ACTION_RESUME,
    ACTION_SHARE,
    JobNotifications,
    action_notification_id,
    chat_of,
    question_route,
    question_title,
)
from glossarion_mobile.services.wakelock import SharedWakelock
from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.components import surface
from glossarion_mobile.ui.components.action_sheet import ActionItem, ActionSheet
from glossarion_mobile.ui.components.dialogs import ConfirmDialog, close_dialog
from glossarion_mobile.ui.components.empty_state import EmptyState
from glossarion_mobile.ui.components.status import StatusChip
from glossarion_mobile.ui.router import RouteError, RouteMatch, build_route, parse_route
from glossarion_mobile.ui.screens.base import Screen
from glossarion_mobile.ui.screens.job_detail import JobDetailScreen
from glossarion_mobile.ui.theme import HIT_TARGET, icon_data

__all__ = ["DONE_STRIP_SECONDS", "JobsFeature", "JobsScreen", "SCREEN_ROUTES", "open_job_in_panel"]

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


def open_job_in_panel(page: Any, jid: Any) -> bool:
    """Tablets: job ``jid``'s detail in the shell's SidePanel beside what is on screen (UI_SPEC
    §1.1, §1.7); False on phones (the caller pushes ``/jobs/<jid>``)."""
    if page is None or not surface.is_tablet(page):
        return False
    try:
        match = parse_route(build_route("jobs.detail", {"jid": str(jid)}))
    except RouteError:
        return False
    return match is not None and surface.open_route_in_panel(page, match)


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
        if open_job_in_panel(self.page, snap.id):  # tablets: the detail beside the list
            return
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


def sign_in_label(provider: str, account_id: int = 0) -> str:
    """"ChatGPT" / "Claude #2": the provider's name (``PROVIDER_INFO``) and a non-default slot."""
    try:
        from glossarion_mobile.services.oauth import PROVIDER_INFO

        info = PROVIDER_INFO.get(provider)
        name = info.label if info is not None else provider
    except Exception:
        name = provider
    return f"{name} #{account_id}" if account_id else name


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
            notify=self._notify,
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
        self._launch_notification_routed = False

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
        feature.spawn(feature.background.ask_notifications_on_launch())  # U11 item 8: the system dialog
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
        drawer = getattr(app, "drawer", None)
        if drawer is not None and hasattr(drawer, "register_search") and self.prefs is not None:
            drawer.register_search("files", self.search_files)  # unified search (UI_SPEC §1.3)
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
        self.background.close()
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

    def _navigate(self, name: str, params: Optional[dict] = None, query: Optional[dict] = None) -> None:
        navigate = getattr(self.app, "navigate_to", None)
        if navigate is not None:
            if query:
                navigate(name, params, query)
            else:
                navigate(name, params)

    def _open_progress_folder(self, folder: str) -> Any:
        """Job detail › Progress: the output folder in the Progress manager (the chat's workspace
        opener: ``ChatFeature.open_progress`` builds the Library row and its route id)."""
        chat_feature = getattr(self.app, "chat_feature", None)
        opener = getattr(chat_feature, "open_progress", None)
        if not callable(opener):
            self._notify("The Progress manager is not available in this session")
            return None
        return self.spawn(opener(folder))

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

    def _chat_workspace(self, snap: Any) -> str:
        """Job detail › Files of a chat job: its persisted Attachments workspace (``ChatFeature.job_workspace``)."""
        feature = getattr(self.app, "chat_feature", None)
        lookup = getattr(feature, "job_workspace", None)
        return str(lookup(snap) or "") if callable(lookup) else ""

    async def search_files(self, query: str) -> list:
        """Drawer › Files: names under the safe roots (Output, Library, Inbox), searched on the io pool;
        a hit opens its folder in the file browser."""
        return await self.run_io(self._search_files_blocking, query)

    def _search_files_blocking(self, query: str, *, max_entries: int = 20000, max_depth: int = 4) -> list:
        from glossarion_mobile.ui.shell.drawer import SEARCH_LIMIT, SearchHit

        needle = query.casefold()
        hits: list = []
        seen = 0
        roots = [(key, path) for key, path in self.file_roots().items() if key != "chats" and path]
        for root_key, root in roots:
            if not os.path.isdir(root):
                continue
            base_depth = root.rstrip(os.sep).count(os.sep)
            for folder, dirs, files in os.walk(root):
                if folder.count(os.sep) - base_depth >= max_depth:
                    dirs[:] = []
                dirs[:] = [d for d in dirs if not d.startswith(".")]
                for name in [*dirs, *files]:
                    seen += 1
                    if seen > max_entries or len(hits) >= SEARCH_LIMIT:
                        return hits
                    if needle not in name.casefold():
                        continue
                    path = os.path.join(folder, name)
                    target = path if os.path.isdir(path) else folder
                    try:
                        fid = self.file_ref(target)
                    except Exception:
                        continue
                    relative = os.path.relpath(folder, root)
                    hits.append(SearchHit(
                        title=name,
                        subtitle=f"{root_key.title()} › {relative}" if relative != "." else root_key.title(),
                        icon="FOLDER" if os.path.isdir(path) else "DESCRIPTION",
                        open=lambda r=root_key, f=fid: self._navigate("tools.files.folder", {"root": r, "fid": f}),
                        key=fid + name,
                    ))
        return hits

    def file_roots(self) -> dict:
        paths = self.paths
        output = str(getattr(paths, "output", "") or "")
        return {
            "output": output,
            "library": str(getattr(paths, "library", "") or ""),
            "inbox": self.inbox_dir,
            "chats": os.path.join(output, "Direct Text") if output else "",
            # U9 Logs & diagnostics: the API dumps and the log files
            "payloads": os.path.join(str(getattr(paths, "data", "") or ""), "Payloads") if getattr(paths, "data", "") else "",
            "logs": str(getattr(paths, "logs", "") or ""),
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

    def model_block(self, spec: JobSpec) -> Optional[tuple]:
        """``(reason, detail)`` when ``spec`` would run a model route excluded on mobile (U9 preflight,
        ``model_catalog.job_model_block``), else None."""
        store = getattr(self.app, "config_store", None)
        get = getattr(store, "get", None) if store is not None else None
        try:
            from glossarion_mobile.services.model_catalog import job_model_block

            return job_model_block(spec.kind, spec.params, get)
        except Exception:
            log.debug("model preflight failed", exc_info=True)
            return None

    def choose_model(self) -> Any:
        """"Choose model" of a refused start: the global ModelSheet (the chat's picker)."""
        chat_view = getattr(self.app, "chat_view", None)
        opener = getattr(chat_view, "open_model_sheet", None) if chat_view is not None else None
        if callable(opener):
            return opener("model")
        self._navigate("settings.models")
        return None

    async def submit(self, spec: JobSpec, *, long_job: bool = True) -> Optional[str]:
        """Send / Start: background preparation on the user's tap, then queue the job.

        U9 preflight: a job whose model is a route excluded on mobile (ocz/, ollamapull/, antigravity/ …)
        is not queued - it would fail inside the API client with desktop-only text - and the reason is
        shown with "Choose model" (Library › Translate, the Book page, Tools and glossary starts)."""
        block = self.model_block(spec)
        if block is not None:
            self._notify(block[0], "Choose model", self.choose_model)
            return None
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
            origin = snap.spec.origin or {}
            if snap.question and origin.get("type") == "library" and origin.get("bid"):
                # U9: the Library review gate is answered on the Book page's approval sheet
                self._navigate("library.book", {"bid": str(origin.get("bid"))})
                return
            if snap.question and (snap.question or {}).get("kind") == "async_batch_question":
                self._navigate("tools.async")  # Tools › Async batch answers the dialog's questions
                return
            if open_job_in_panel(self.page, snap.id):  # tablets: SidePanel (UI_SPEC §1.7)
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
                                     dark=dark, open_progress=self._open_progress_folder, roots=self.file_roots,
                                     chat_workspace=self._chat_workspace)
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

    def _shown_chat(self) -> Optional[str]:
        """The chat the chat home (``/``, ``/chat/<cid>``) shows: the chat view's own chat."""
        chat_view = getattr(self.app, "chat_view", None)
        cid = getattr(chat_view, "cid", None)
        if cid in (None, ""):
            current = getattr(getattr(self.state, "current_chat", None), "value", None)
            cid = current if current not in (None, "") else "1"
        return str(cid)

    def question_on_screen(self, snap: JobSnapshot) -> bool:
        """The surface that answers ``snap``'s question is what the user looks at: the owning chat (its
        approval card), the Book page of a Library job, Tools › Async batch."""
        if not self.background.app_visible:
            return False
        shell = getattr(self.app, "shell", None)
        current = str(getattr(shell, "current_route", "") or "") if shell is not None else ""
        if not current:
            return False
        try:
            match = parse_route(current)
        except Exception:
            match = None
        if getattr(match, "name", None) in ("home", "chat", "chat.message"):
            # The chat home shows the chat view's chat, whatever cid the route names: "New chat" and
            # the other chat switches change the chat without changing the shell route.
            cid = chat_of(snap)
            return cid is not None and self._shown_chat() == cid
        return current.split("?", 1)[0] == question_route(snap)

    def _on_question(self, snap: JobSnapshot, question: Any) -> None:
        """A job blocks on a question (UI loop; also while the app is hidden, the question is posted):
        notify unless its surface is on screen (UI_SPEC §1.9, §2.11)."""
        if self.question_on_screen(snap):
            return  # the approval card / sheet / dialog is already in front of the user
        question = question if isinstance(question, dict) else {}
        route = question_route(snap)
        if is_glossary_question(question.get("kind")):
            self.spawn(self.notifications.glossary_review_needed(snap))
            message, action = "Glossary ready: review needed", "Review"
        else:
            title = question_title(snap, question)
            self.spawn(self.notifications.answer_needed(snap, title))
            message, action = f"Answer needed: {title}", "Answer"
        if self.background.app_visible:
            self._notify(message, action, lambda: self._navigate_route(route))

    def _on_job_event(self, job_id: str, kind: str, data: Any) -> None:
        """Job events JobService does not consume itself (UI loop)."""
        if kind == "question_resolved":
            # answered in the app (card / sheet / dialog), from the notification, or given up by Stop: the
            # "Glossary ready" / "Answer needed" notification has nothing left to answer (posted by JobService,
            # so this also runs while the dispatcher's pump is parked with the app hidden).
            self.spawn(self.notifications.cancel_action(job_id))
            return
        if kind == "sign_in_required":
            # UI_SPEC §1.9: the job lost (or never had) its login (ChatGPT / Claude / Gemini / Grok)
            # and the backend falls back to the browser login or refuses it; point the user at that
            # provider's LoginSheet (OAuthBridge: Custom Tab, paste fallback, sign-in service).
            data = data if isinstance(data, dict) else {}
            provider = str(data.get("provider") or "authgpt")
            try:
                account_id = int(data.get("account_id") or 0)
            except (TypeError, ValueError):
                account_id = 0
            label = sign_in_label(provider, account_id)
            self.spawn(self.notifications.sign_in_required(label))
            self._notify(f"Sign-in required for {label}", "Sign in",
                         lambda: self.open_login_sheet(provider, account_id))

    def open_login_sheet(self, provider: str, account_id: int = 0) -> Any:
        """The provider's LoginSheet for that slot (Accounts when the bridge is not available)."""
        oauth = getattr(getattr(self.app, "chat_feature", None), "oauth", None)
        if oauth is None or self.page is None:
            self._navigate("settings.accounts")
            return None
        try:
            from glossarion_mobile.ui.screens.accounts import LoginSheet

            pages = getattr(self.app, "pages_feature", None)
            on_done = (lambda _status: pages._signed_in_changed()) if pages is not None else None
            sheet = LoginSheet(oauth, provider=provider, account_id=int(account_id or 0), on_done=on_done)
            sheet.show(self.page)
            self.login_sheet = sheet
            return sheet
        except Exception:
            log.exception("opening the %s sign-in failed", provider)
            self._navigate("settings.accounts")
            return None

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
        if tap.action == ACTION_ACCEPT and self.accept_glossary_from_notification(tap):
            self._notify("Glossary accepted · translating")
        if tap.route:
            navigate = getattr(self.app, "navigate", None)
            if navigate is not None:
                await navigate(tap.route)

    def accept_glossary_from_notification(self, tap: Any) -> bool:
        """"Glossary ready" › **Accept**: answer the running job's glossary gate Yes. A chat job is answered
        through its chat's run controller (the approval card's own ✓ Yes path, ``ChatRuns.answer_glossary``),
        any other job (the Library review gate) through ``JobService.answer``. A stale tap (the job ended,
        another job's notification, no glossary question pending) answers nothing: the caller only opens
        the route."""
        snap = self.service.snapshot()
        if snap is None:
            return False
        notification_id = getattr(tap, "notification_id", None)
        if notification_id is not None and notification_id != action_notification_id(snap.id):
            return False
        question = self.service.pending_question(snap.id)
        if not question or not is_glossary_question(question.get("kind")):
            return False
        if getattr(tap, "route", None) and tap.route != question_route(snap):
            return False
        answered = False
        cid = chat_of(snap)
        runs = getattr(getattr(self.app, "chat_feature", None), "runs", None)
        answer_glossary = getattr(runs, "answer_glossary", None)
        if cid is not None and callable(answer_glossary):
            try:
                answered = bool(answer_glossary(cid, True))
            except Exception:
                log.exception("accepting the glossary of chat %s failed", cid)
        if not answered:  # not a chat run (Library review gate, a chat's Library job): the job's question
            answered = bool(self.service.answer(question.get("id"), True))
        return answered

    async def route_launch_notification(self) -> bool:
        """The notification tap that cold-started the app (Android keeps it as the launch notification and
        never sends it as an event; iOS likewise): handled once, like a tap while running. Called by the app
        after its launch links are routed; a stale Accept only opens the route."""
        if self._launch_notification_routed:
            return False
        self._launch_notification_routed = True
        native = getattr(self.app, "native", None)
        if native is None:
            return False
        try:
            getter = getattr(native, "launch_notification", None)
            if callable(getter):
                event = await getter()
            else:
                event = await native.call("get_launch_notification", default=None)
        except Exception as exc:
            log.info("get_launch_notification failed: %s", exc)
            return False
        if not isinstance(event, dict) or not event.get("payload"):
            return False
        await self._on_notification(event)
        return True

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
            close_dialog(self.page, self.banner)

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
