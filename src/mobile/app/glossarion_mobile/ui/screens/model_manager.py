"""Model Manager (``/settings/models``; UI_SPEC §4.11) and the Models & keys feature install.

**Models** (desktop "Manage Models" › Models): search; chips Polled only (``hide_unpolled``, the
desktop "🧪 Hide unpolled models" toggle that also filters every picker), Custom (entries that are
not in the built-in / polled catalog) and Removed (the tombstones in
``model_manager_removed_models``); "🌐 Poll providers" with the desktop status text and a
per-provider status chip row; the list in saved order (``ReorderableListView``: drag to reorder;
long-press a row for Move to top / up / down / bottom, the desktop ⇈ ↑ ↓ ⇊ buttons; swipe to
remove with Undo; in Removed, swipe to restore); FAB "Add model"; ⋯ Reset to defaults.
The list is a ``WindowedList`` (100-row steps, 500-row windows, "Show 100 more"): a desktop config
brings thousands of models, and every mounted row is about 1 KB on the Flet wire (issue 19). A drag
stays inside the mounted rows; the long-press moves reach any position of the whole list.
Every edit is saved at once through ``ModelCatalogService`` → ``model_catalog_core`` (the desktop
saves on "Save"; tombstones are computed the same way, so removed models never come back from a
passive catalog merge).

**Custom prefixes**: the ``custom_prefix_routes`` table (prefix → base URL → endpoint type) with an
add/edit sheet validated like the desktop table.

``ModelsKeysFeature.install(app)`` wires the catalog service, the ModelSheet environment and the
screens for ``settings.models``, ``settings.keys``, ``settings.keys.pool`` and
``settings.endpoints`` into the app (Refusal patterns and Local AI open as full-screen views).
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections import OrderedDict
from typing import Any, Callable, Optional

import flet as ft

from glossarion_mobile.services import model_catalog as mc
from glossarion_mobile.services.model_catalog import CatalogSnapshot, ModelCatalogService
from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.components.action_sheet import ActionItem, ActionSheet
from glossarion_mobile.ui.components.dialogs import ConfirmDialog, close_dialog
from glossarion_mobile.ui.components.error_card import ErrorCard
from glossarion_mobile.ui.components.reason_chip import ReasonChip
from glossarion_mobile.ui.components.skeleton import Skeleton
from glossarion_mobile.ui.components.windowed_list import WindowedList
from glossarion_mobile.ui.foreground import poll_sleep
from glossarion_mobile.ui.router import RouteMatch
from glossarion_mobile.ui.screens.base import Screen, build_screen_view
from glossarion_mobile.ui.theme import HIT_TARGET, semantic

__all__ = ["IMPLEMENTED_ROUTES", "ModelManagerScreen", "ModelsKeysFeature", "PrefixEditor", "SCREEN_ROUTES"]

log = logging.getLogger("glossarion.models")

SCREEN_ROUTES = ("settings.models", "settings.keys", "settings.keys.pool", "settings.endpoints")
#: Settings home pages this feature ships (merged into ``SettingsHome.implemented`` once installed).
IMPLEMENTED_ROUTES = frozenset(("settings.models", "settings.keys", "settings.endpoints"))
READY_TEXT = "Ready to poll online catalogs\n✓ marks models confirmed within 7 days"
RESET_TEXT = ("Replace the current list with the built-in default model catalog?\n\n"
              "Any custom models you added will be lost.")
WHEEL_REASON = "Desktop mouse-wheel guard; touch screens have no wheel. The setting is kept."
ENDPOINT_TYPES_FALLBACK = ("/chat/completions", "/images/generations", "/v1/messages", "/v1/ocr", "/{model_id}")
#: UI_SPEC §4.11 / §7.3: a model row is about 1 KB on the wire, so the list mounts 100 rows at a time and at
#: most 500 (the page selector moves between windows of 500).
LIST_STEP = 100
LIST_WINDOW = 500
ROW_CACHE_MAX = 2 * LIST_WINDOW  # rows kept for reuse (search / filter round trips)
SEARCH_DEBOUNCE = 0.25  # the Glossary editor's typing pause (ui/glossary/editor.py SEARCH_DEBOUNCE)
STALE_CHECK_SECONDS = 0.25  # a covered screen with new data checks this often whether it is shown again
SLOW_LOAD_SECONDS = 10.0  # then "Still loading…" with Retry replaces the skeleton


def _push(*controls: Any) -> None:
    """``update()`` each control (None skipped); controls not on a page (tests) are ignored."""
    for control in controls:
        if control is None:
            continue
        try:
            control.update()
        except Exception:
            pass


def endpoint_type_options() -> tuple:
    try:
        from run_env import RunEnvMixin

        return tuple(RunEnvMixin._CUSTOM_PREFIX_ENDPOINT_TYPES)
    except Exception:
        return ENDPOINT_TYPES_FALLBACK


class ModelManagerScreen(Screen):
    """The Models / Custom prefixes page.

    Issue 19 (the list of 884-3,776+ models froze the app): the rows live in a ``WindowedList``
    (``LIST_STEP`` rows per step, at most ``LIST_WINDOW`` mounted, a "Show 100 more" footer and
    the page selector) over a ``ReorderableListView``; rows are cached by (model, polled,
    excluded, removed view) so a repaint re-sends only rows that changed. Every change paints
    once: catalog publishes are coalesced to one render on the next loop tick, a publish that only
    changes poll state updates the header (while Poll providers runs, its publishes only update the
    header and the list syncs once at the end; the user's own search, chips and edits still update
    the list at once), the search runs ``SEARCH_DEBOUNCE`` after typing stops (off the loop, a newer
    keystroke wins), and nothing paints while another screen covers this one (one render when it is
    shown again)."""

    title = "Model manager"

    def __init__(self, match: Optional[RouteMatch], *, catalog: ModelCatalogService, page: Any = None,
                 notify: Optional[Callable[..., Any]] = None, spawn: Optional[Callable[[Any], Any]] = None,
                 run_io: Optional[Callable[..., Any]] = None, tablet: bool = False,
                 is_top: Optional[Callable[[Any], bool]] = None) -> None:
        super().__init__(match)
        self.catalog = catalog
        self.page = page
        self.notify = notify
        self.spawn_fn = spawn
        self.run_io = run_io
        self.tablet = tablet
        self.is_top = is_top  # shell: is this screen shown (top of the stack)? Hidden screens do not paint
        self.tab = "models"
        self.query = ""  # the search the list shows (the field's text applies SEARCH_DEBOUNCE later)
        self.filter = "all"  # all | custom | removed
        self.poll_status = READY_TEXT
        self.polling = False
        self.snapshot: CatalogSnapshot = catalog.snapshot
        self._poll_known: Optional[frozenset] = None  # the last poll's catalog (desktop ``known``)
        self._unsub: Optional[Callable[[], None]] = None
        self.last_dialog: Any = None
        self.last_sheet: Optional[ActionSheet] = None  # the last row-action sheet (tests / UI driver)
        self.prefix_editor: Optional[PrefixEditor] = None
        self.rows: Optional[WindowedList] = None
        self._row_cache: "OrderedDict[tuple, tuple]" = OrderedDict()  # cache key -> (row, drag handle)
        self._cache_routes: Any = None
        self._list_inputs: Any = None  # what the mounted list was built from
        self._view: Any = None  # (filter, query, polled only): a new one starts at the first window
        self._visible_count = 0
        self._precomputed: Any = None  # (inputs, models) from the off-loop search
        self._shown_statuses: Any = None
        self._state_kind: Any = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._render_handle: Any = None
        self._stale = False  # a change arrived while hidden: paint once when shown again
        self._stale_task: Any = None
        self._search_task: Any = None
        self._search_generation = 0
        self._hold_list = False  # Poll providers: its publishes update the header; the list syncs when it ends
        self._user_change = False  # a user edit waits in the coalesced render: it syncs the list even then
        self._load_error: Optional[BaseException] = None
        self._slow = False
        self._slow_task: Any = None
        self._disposed = False
        self.renders = 0  # renders that ran (coalesced / not hidden)
        self.list_renders = 0  # of those, the ones that re-synced the list

    # ---- helpers --------------------------------------------------------------------------------

    def say(self, message: str, action: Optional[str] = None, on_action: Any = None) -> None:
        if self.notify is None:
            log.info("models: %s", message)
            return
        try:
            self.notify(message, action, on_action)
        except TypeError:
            self.notify(message)

    def spawn(self, coro: Any) -> Any:
        return self.spawn_fn(coro) if self.spawn_fn is not None else asyncio.ensure_future(coro)

    async def io(self, fn: Callable[..., Any], *args: Any) -> Any:
        if self.run_io is not None:
            return await self.run_io(fn, *args)
        return await asyncio.to_thread(fn, *args)

    def _edit(self, ok_error: tuple, success: Optional[str] = None) -> bool:
        """A catalog edit's result. The edit published a snapshot (one coalesced render); a failure
        only says why."""
        ok, error = ok_error
        if not ok:
            if error:
                self.say(error if isinstance(error, str) else error[1])
            return False
        self.snapshot = self.catalog.snapshot
        if success:
            self.say(success)
        self._user_change = True
        self._request_render()
        return True

    @property
    def builtin_keys(self) -> Optional[frozenset]:
        """Casefolded catalog ids for the Custom filter: the last poll's catalog, else the loaded
        catalog (``CatalogSnapshot.known_keys``); None until a load has read it."""
        if self._poll_known is not None:
            return self._poll_known
        return self.snapshot.known_keys

    @builtin_keys.setter
    def builtin_keys(self, value: Any) -> None:
        self._poll_known = frozenset(value) if value is not None else None

    # ---- body -----------------------------------------------------------------------------------

    def build_body(self) -> ft.Control:
        self._capture_loop()
        self.tabs = ft.SegmentedButton(
            segments=[ft.Segment(value="models", label="Models"), ft.Segment(value="prefixes", label="Custom prefixes")],
            selected=[self.tab], on_change=self._on_tab,
        )
        self.search = ft.TextField(hint_text="Search models", prefix_icon=ft.Icons.SEARCH, dense=True,
                                   on_change=lambda e: self._on_search_change(e.control.value or ""),
                                   border_radius=tokens.RADII["field"], key="mm-search")
        self.polled_chip = ft.Chip(label=ft.Text("Polled only"), show_checkmark=True,
                                   selected=self.snapshot.hide_unpolled, on_click=lambda e: self.toggle_polled_only(),
                                   key="mm-polled-only")
        self.custom_chip = ft.Chip(label=ft.Text("Custom"), selected=False, on_click=lambda e: self.set_filter("custom"),
                                   key="mm-custom")
        self.removed_chip = ft.Chip(label=ft.Text("Removed"), selected=False, on_click=lambda e: self.set_filter("removed"),
                                    key="mm-removed")
        self.poll_button = ft.FilledTonalButton(content="🌐 Poll providers", on_click=lambda e: self.spawn(self.poll()),
                                                key="mm-poll")
        self.poll_text = ft.Text(self.poll_status, theme_style=ft.TextThemeStyle.BODY_SMALL,
                                 color=ft.Colors.ON_SURFACE_VARIANT)
        self.status_row = ft.Row([], scroll=ft.ScrollMode.AUTO, spacing=6)
        self.hint = ft.Text("", theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT)
        self.models_header = ft.Column([
            self.search,
            ft.Row([self.polled_chip, self.custom_chip, self.removed_chip], wrap=True, spacing=6),
            ft.Row([self.poll_button], wrap=True),
            self.poll_text,
            self.status_row,
            self.hint,
        ], spacing=8, tight=True)
        self.rows = WindowedList(
            build_row=self._row_for, key_of=self._row_key, window_rows=LIST_WINDOW, step=LIST_STEP, key="mm-rows",
            list_view=ft.ReorderableListView(controls=[], expand=True, on_reorder=self._on_reorder,
                                             show_default_drag_handles=False, build_controls_on_demand=True,
                                             scroll_interval=120, key="mm-list",
                                             padding=ft.Padding.symmetric(horizontal=tokens.SPACING["md"])),
            scroll_keys=False, show_more=True, quiet_scroll=True, reset_scroll=True,
        )
        self.list_view = self.rows.list_view  # the mounted rows (tests / UI driver)
        self.state_box = ft.Container(visible=False, padding=ft.Padding.symmetric(horizontal=12), key="mm-state")
        self.prefix_list = ft.ListView(controls=[], expand=True, spacing=6,
                                       padding=ft.Padding.symmetric(horizontal=tokens.SPACING["md"]))
        self.add_fab = ft.FloatingActionButton(icon=ft.Icons.ADD, content="Add model", on_click=lambda e: self.open_add(),
                                               key="mm-add")
        self.core_notice = ft.Container(visible=not self.catalog.core.available, content=ReasonChip(
            reason="Editing needs the model catalog core", detail=mc.CORE_MISSING), padding=ft.Padding.only(left=12))
        self.body_column = ft.Column([
            ft.Container(content=self.tabs, padding=ft.Padding.symmetric(horizontal=12, vertical=6)),
            self.core_notice,
            ft.Container(content=self.models_header, padding=ft.Padding.symmetric(horizontal=12)),
            self.state_box,
            self.rows.control,
            self.prefix_list,
            ft.Container(content=ft.Row([self.add_fab], alignment=ft.MainAxisAlignment.END),
                         padding=ft.Padding.only(right=16, bottom=16)),
        ], expand=True, spacing=4)
        self.render(push=False)
        return self.body_column

    def actions(self) -> list:
        return [ft.PopupMenuButton(icon=ft.Icons.MORE_VERT, tooltip="More", items=[
            ft.PopupMenuItem(content="Reset to defaults", icon=ft.Icons.RESTART_ALT, on_click=lambda e: self.confirm_reset()),
            ft.PopupMenuItem(content="Lock mouse wheel (desktop only)", icon=ft.Icons.MOUSE, disabled=True,
                             on_click=lambda e: self.say(WHEEL_REASON)),
        ])]

    def did_show(self) -> None:
        self._disposed = False
        self._capture_loop()
        if self._unsub is None:
            self._unsub = self.catalog.subscribe(self._on_snapshot)
            latest = self.catalog.snapshot
            if latest is not self.snapshot:  # published between the build and now
                self.snapshot = latest
                self._stale = True
        if not self.catalog.snapshot.loaded:
            self.spawn(self._load())  # shares a load already in flight (ModelCatalogService.load)
            self._start_slow_timer()
        if self._stale and self.body is not None:
            self._render_now()

    def dispose(self) -> None:
        self._disposed = True
        if self._unsub is not None:
            self._unsub()
            self._unsub = None
        for task in (self._search_task, self._stale_task, self._slow_task):
            if task is not None and hasattr(task, "cancel"):
                task.cancel()
        self._search_task = self._stale_task = self._slow_task = None
        if self._render_handle is not None:
            self._render_handle.cancel()
            self._render_handle = None

    async def _load(self, *, fresh: bool = False) -> None:
        self._load_error = None
        try:
            await self.catalog.load(fresh=fresh)
        except Exception as exc:
            log.exception("loading the model list failed")
            self._load_error = exc
            self._request_render()

    async def retry_load(self) -> None:
        """ErrorCard / "Still loading…" Retry. A failed load reads the catalog again; a load still in
        flight is shared (``ModelCatalogService.load``: a second read would only queue behind it), with
        the skeleton again and "Still loading…" + Retry back after ``SLOW_LOAD_SECONDS``."""
        self._load_error = None
        self._slow = False
        if not self.snapshot.loaded:
            self._render_now()
            self._start_slow_timer()
        await self._load()
        self.snapshot = self.catalog.snapshot
        self._render_if_shown()

    def _start_slow_timer(self) -> None:
        task = self._slow_task
        if task is not None and not task.done():
            return

        async def later() -> None:
            await asyncio.sleep(SLOW_LOAD_SECONDS)
            if not self._disposed and not self.catalog.snapshot.loaded:
                self._slow = True
                self._request_render()

        try:
            self._slow_task = self.spawn(later())
        except RuntimeError:  # no running loop (a screen built outside the app)
            self._slow_task = None

    # ---- one render per change ---------------------------------------------------------------------

    def _capture_loop(self) -> None:
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            pass

    def _on_loop(self) -> bool:
        if self._loop is None:
            return True
        try:
            return asyncio.get_running_loop() is self._loop
        except RuntimeError:
            return False

    def _shown(self) -> bool:
        """On top of the stack (not covered by another screen); True without a shell hook."""
        if self.is_top is None:
            return True
        try:
            return bool(self.is_top(self))
        except Exception:
            return True

    def _on_snapshot(self, snapshot: CatalogSnapshot) -> None:
        if not self._on_loop():  # published on an I/O thread without a dispatcher: paint on the loop
            try:
                self._loop.call_soon_threadsafe(self._on_snapshot, snapshot)
            except RuntimeError:  # the loop is closed
                pass
            return
        if self._disposed or snapshot is self.snapshot:
            return
        self.snapshot = snapshot
        self._request_render()

    def _request_render(self) -> None:
        """Coalesce: every change before the next loop tick paints once."""
        if self._render_handle is not None or self._disposed:
            return
        loop = self._loop
        if loop is not None and not self._on_loop():  # an I/O thread: hand it to the loop
            try:
                loop.call_soon_threadsafe(self._request_render)
            except RuntimeError:  # the loop is closed
                pass
            return
        if loop is None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                self._flush()  # no loop at all (a screen driven outside the app)
                return
        self._render_handle = loop.call_soon(self._flush)

    def _flush(self) -> None:
        self._render_handle = None
        user_change, self._user_change = self._user_change, False
        if self._disposed or self.body is None:
            return
        if not self._shown():
            self._stale = True
            self._watch_until_shown()
            return
        self._stale = False
        # while Poll providers runs, its publishes only update the header; a user edit still syncs the list
        self.render(sync_list=user_change or not self._hold_list)

    def _render_now(self, *, whole: bool = False) -> None:
        """A user action paints at once (a pending coalesced render is folded into it)."""
        if self._render_handle is not None:
            self._render_handle.cancel()
            self._render_handle = None
        self._stale = False
        self._user_change = False
        self.render(whole=whole)

    def _render_if_shown(self) -> None:
        """The end of something the user started (a search, a poll, a Retry): paint now, unless another
        screen covers this one by now (then once when it is shown again)."""
        if self._shown():
            self._render_now()
        else:
            if self._render_handle is not None:
                self._render_handle.cancel()
                self._render_handle = None
            self._stale = True
            self._watch_until_shown()

    def _watch_until_shown(self) -> None:
        """Hidden with new data: check every ``STALE_CHECK_SECONDS`` (parked while the app is in the
        background) and paint once when the screen is on top again (Back does not call did_show)."""
        task = self._stale_task
        if task is not None and not task.done():
            return

        async def watch() -> None:
            while self._stale and not self._disposed:
                await poll_sleep(self.page, STALE_CHECK_SECONDS)
                if self._stale and not self._disposed and self._shown():
                    self._render_now()
                    return

        try:
            self._stale_task = self.spawn(watch())
        except RuntimeError:  # no running loop
            self._stale_task = None

    # ---- rendering ----------------------------------------------------------------------------------

    @staticmethod
    def _visible_for(snap: CatalogSnapshot, filter_: str, query: str, known: Optional[frozenset]) -> list:
        if filter_ == "removed":
            models = list(snap.removed)
        else:
            models = snap.visible_models()
            if filter_ == "custom":
                if known is None:  # the catalog was not read yet: never "everything is custom"
                    return []
                models = [m for m in models if m.casefold() not in known]
        if query.strip():
            models = mc.rank_models(models, query, limit=len(models) or 1)
        return models

    def visible_models(self) -> list:
        return self._visible_for(self.snapshot, self.filter, self.query, self.builtin_keys)

    def _inputs(self, snap: CatalogSnapshot, filter_: str, query: str, known: Optional[frozenset]) -> tuple:
        """Everything the list depends on (compared by identity first: an unchanged list costs nothing)."""
        return (snap.loaded, snap.models, snap.removed, snap.polled, snap.hide_unpolled, snap.custom_routes, filter_,
                query.strip(), known if filter_ == "custom" else None)

    @property
    def reorderable(self) -> bool:
        return self.filter == "all" and not self.query.strip() and not self.snapshot.hide_unpolled

    def render(self, push: bool = True, *, whole: bool = False, sync_list: bool = True) -> None:
        """Sync the header, the state card and (only when its inputs changed; never when ``sync_list``
        is False: a poll's own publishes) the list; push only the parts that can have changed
        (``whole``: the tab changed, push the body)."""
        self.renders += 1
        snap = self.snapshot
        models_tab = self.tab == "models"
        self.models_header.visible = models_tab
        self.rows.control.visible = models_tab
        self.prefix_list.visible = not models_tab
        self.add_fab.content = "Add model" if models_tab else "Add prefix"
        self._sync_header(snap)
        list_changed = False
        if models_tab and sync_list:
            list_changed = self._sync_list(snap)
        elif not models_tab:
            self.prefix_list.controls = self._prefix_rows()
        self._sync_state(snap)
        if not push:
            return
        if whole:
            _push(self.body_column)
        elif models_tab:
            _push(self.models_header, self.state_box, self.rows.control if list_changed else None)
        else:
            _push(self.state_box, self.prefix_list)

    def _sync_header(self, snap: CatalogSnapshot) -> None:
        self.polled_chip.selected = snap.hide_unpolled
        self.custom_chip.selected = self.filter == "custom"
        self.custom_chip.disabled = self.builtin_keys is None and self.filter != "custom"
        self.removed_chip.selected = self.filter == "removed"
        self.poll_button.disabled = self.polling or snap.is_polling()
        self.poll_button.content = "⏳ Polling…" if (self.polling or "*" in snap.polling) else "🌐 Poll providers"
        self.poll_text.value = self.poll_status
        if snap.statuses is not self._shown_statuses:
            self._shown_statuses = snap.statuses
            self.status_row.controls = self._status_chips(snap)

    def _sync_list(self, snap: CatalogSnapshot) -> bool:
        known = self.builtin_keys
        inputs = self._inputs(snap, self.filter, self.query, known)
        if inputs == self._list_inputs:
            return False
        precomputed = self._precomputed
        self._precomputed = None
        if precomputed is not None and precomputed[0] == inputs:
            models = precomputed[1]
        else:
            models = self._visible_for(snap, self.filter, self.query, known)
        if snap.custom_routes != self._cache_routes:  # provider labels follow the custom prefixes
            self._row_cache.clear()
            self._cache_routes = snap.custom_routes
        view = (self.filter, self.query.strip(), snap.hide_unpolled)
        fresh = view != self._view  # a new search / filter starts at the first rows
        self._view = view
        self._list_inputs = inputs
        self._visible_count = len(models)
        self.rows.set_items(self._items(models), keep_window=not fresh, keep_rendered=not fresh)
        self.list_renders += 1
        if not snap.loaded:
            self.hint.value = ""
        elif self.filter == "removed":
            self.hint.value = f"{len(models)} removed · swipe to restore"
        elif self.reorderable:
            self.hint.value = f"{len(models)} models · drag ☰ or long-press to move · swipe to remove"
        else:
            self.hint.value = f"{len(models)} shown · clear the search and filters to reorder"
        return True

    def _sync_state(self, snap: CatalogSnapshot) -> None:
        """The card above the list: loading skeleton, "Still loading…", the load error, or the empty text."""
        kind: Any = None
        build: Optional[Callable[[], ft.Control]] = None
        error = self._load_error if self._load_error is not None else snap.error
        if self.tab != "models":
            kind = None
        elif error:
            kind = ("error", str(error))
            build = lambda: ErrorCard(title="Couldn't load the model list", message=error,  # noqa: E731
                                      on_retry=self.retry_load, key="mm-error")
        elif not snap.loaded:
            kind = "slow" if self._slow else "loading"
            build = self._slow_card if self._slow else (
                lambda: Skeleton("rows", count=6, label="Loading models…", key="mm-skeleton").control)
        elif self.filter == "custom" and self.builtin_keys is None:
            kind = ("text", "Loading…")
        elif not self._visible_count:  # what the mounted list shows (synced or not during a poll)
            kind = ("text", "No removed models." if self.filter == "removed" else "No models match.")
        if kind == self._state_kind:
            return
        self._state_kind = kind
        if build is None and isinstance(kind, tuple) and kind[0] == "text":
            text = kind[1]
            build = lambda: ft.Container(key="mm-empty", padding=ft.Padding.all(4),  # noqa: E731
                                         content=ft.Text(text, color=ft.Colors.ON_SURFACE_VARIANT))
        self.state_box.content = build() if build is not None else None
        self.state_box.visible = build is not None

    def _slow_card(self) -> ft.Control:
        return ft.Row([
            ft.ProgressRing(width=16, height=16, stroke_width=2),
            ft.Text("Still loading the model list…", expand=True, color=ft.Colors.ON_SURFACE_VARIANT),
            ft.TextButton(content="Retry", icon=ft.Icons.REFRESH, on_click=lambda e: self.spawn(self.retry_load())),
        ], spacing=8, vertical_alignment=ft.CrossAxisAlignment.CENTER, key="mm-slow")

    def _status_chips(self, snap: CatalogSnapshot) -> list:
        chips = []
        for provider, status in sorted(snap.statuses.items()):
            text = str(status)
            if mc.provider_excluded(provider):  # a route excluded on mobile: its reason, not a poll warning
                chips.append(ReasonChip(reason=f"{mc.provider_label(provider)} · {mc.EXCLUDED_STATUS}",
                                        detail=mc.provider_excluded_detail(provider), key=f"mm-status-{provider}"))
                continue
            ok = text.startswith("online")
            color = semantic("success") if ok else (ft.Colors.OUTLINE if "credential" in text else semantic("warning"))
            short = text.replace("online ", "").replace("static fallback ", "")
            chips.append(ft.Chip(label=ft.Text(f"{mc.provider_label(provider)} · {short}",
                                               theme_style=ft.TextThemeStyle.LABEL_SMALL),
                                 leading=ft.Icon(ft.Icons.CIRCLE, size=10, color=color), tooltip=text,
                                 key=f"mm-status-{provider}"))
        return chips

    # ---- rows -----------------------------------------------------------------------------------------

    @staticmethod
    def _items(models: list) -> list:
        """(model, n): the n-th row of the same id (a hand-edited list may repeat one) keeps its own key."""
        seen: dict = {}
        items = []
        for model in models:
            n = seen.get(model, 0)
            seen[model] = n + 1
            items.append((model, n))
        return items

    @staticmethod
    def _row_key(item: tuple) -> str:
        model, n = item
        return f"mm-{model}" if not n else f"mm-{model}#{n}"

    def _cache_key(self, item: tuple) -> tuple:
        """What a row shows: (model, n, polled, excluded on mobile, Removed view)."""
        model, n = item
        return (model, n, self.snapshot.is_polled(model), bool(mc.excluded_route(model)), self.filter == "removed")

    def _row_for(self, item: tuple, _position: int = 0) -> ft.Control:
        """The row of ``item``, reused while its look is unchanged (WindowedList ``build_row``)."""
        cache_key = self._cache_key(item)
        _model, _n, polled, excluded, removed_view = cache_key
        cached = self._row_cache.get(cache_key)
        if cached is None:
            cached = self._row(item, polled=polled, excluded=excluded, removed_view=removed_view)
            self._row_cache[cache_key] = cached
            while len(self._row_cache) > ROW_CACHE_MAX:
                self._row_cache.popitem(last=False)
        else:
            self._row_cache.move_to_end(cache_key)
        row, handle = cached
        handle.visible = self.reorderable and not removed_view
        return row

    def _row(self, item: tuple, *, polled: bool, excluded: bool, removed_view: bool) -> tuple:
        model = item[0]
        provider = mc.provider_of(model, list(self.snapshot.custom_routes))
        subtitle = mc.provider_label(provider) + (" · ✓ polled" if polled else "")
        handle = ft.ReorderableDragHandle(content=ft.Icon(ft.Icons.DRAG_HANDLE),
                                          visible=self.reorderable and not removed_view)
        tile = ft.ListTile(
            title=ft.Text(model, max_lines=1, overflow=ft.TextOverflow.ELLIPSIS),
            subtitle=ft.Text(subtitle, theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT),
            leading=ft.Icon(ft.Icons.CHECK if polled else ft.Icons.SMART_TOY_OUTLINED,
                            color=semantic("success") if polled else ft.Colors.ON_SURFACE_VARIANT),
            trailing=ft.Row([
                *([ReasonChip(reason="Not on mobile", detail=mc.excluded_detail(model))] if excluded else []),
                handle,
                ft.IconButton(icon=ft.Icons.RESTORE if removed_view else ft.Icons.DELETE_OUTLINE,
                              tooltip="Restore" if removed_view else "Remove", size_constraints=HIT_TARGET,
                              on_click=lambda e, m=model: self.restore(m) if removed_view else self.remove(m)),
            ], spacing=0, tight=True),
            dense=True, min_height=52,
            # Move to top / up / down / bottom (any position, past the mounted window); Removed: swipe restores
            on_long_press=None if removed_view else (lambda e, it=item: self.open_row_actions(it)),
        )
        # one background (swipe toward the start): a shared background / secondary_background pair
        # serialised the same Container twice under one id
        background = ft.Container(bgcolor=semantic("success") if removed_view else ft.Colors.ERROR,
                                  padding=ft.Padding.symmetric(horizontal=16), alignment=ft.Alignment.CENTER_RIGHT,
                                  content=ft.Icon(ft.Icons.RESTORE if removed_view else ft.Icons.DELETE_OUTLINE,
                                                  color=ft.Colors.ON_ERROR))
        row = ft.Dismissible(
            content=tile,
            background=background,
            dismiss_direction=ft.DismissDirection.END_TO_START,
            on_dismiss=lambda e, m=model: self.restore(m) if removed_view else self.remove(m),
        )
        return row, handle

    def _take_rows(self, model: str) -> list:
        """Take ``model``'s rows out of the list at once (a dismissed ``Dismissible`` must leave the tree
        in this update); [(index, item)] to put back if the edit fails. A dismissed row object is never
        mounted again (Undo / a failed edit build a new one)."""
        rows = self.rows
        taken: list = []
        if rows is None:
            return taken
        n = 0
        while True:
            item = (model, n)
            index = rows.positions.get(self._row_key(item))
            if index is None:
                break
            taken.append((index, item))
            n += 1
        for index, item in reversed(taken):
            self._forget_row(rows.controls.get(self._row_key(item)))
            rows.remove_key(self._row_key(item))
        if taken:
            _push(rows.control)
        return taken

    def _forget_row(self, row: Any) -> None:
        """Drop a mounted row object from the cache (the next build of its item makes a new one)."""
        if row is not None:
            for cache_key in [k for k, cached in self._row_cache.items() if cached[0] is row]:
                del self._row_cache[cache_key]

    def _put_back(self, taken: list) -> None:
        if self.rows is None or not taken:
            return
        for index, item in taken:
            self.rows.insert_item(index, item)
        _push(self.rows.control)

    # ---- model edits -----------------------------------------------------------------------------------

    def _on_tab(self, e: Any = None) -> None:
        selected = list(getattr(getattr(e, "control", None), "selected", []) or []) if e is not None else []
        self.set_tab(selected[0] if selected else "models")

    def set_tab(self, tab: str) -> None:
        self.tab = tab if tab in ("models", "prefixes") else "models"
        self.tabs.selected = [self.tab]
        self._render_now(whole=True)

    def _on_search_change(self, text: str) -> None:
        """Search as you type: each keystroke restarts ``SEARCH_DEBOUNCE``, then :meth:`apply_search`."""
        task = self._search_task
        if task is not None and not task.done():
            task.cancel()

        async def later() -> None:
            try:
                await asyncio.sleep(SEARCH_DEBOUNCE)
            except asyncio.CancelledError:
                return
            await self.apply_search(text)

        self._search_task = self.spawn(later())
        _push(self.search)  # nothing to send: keeps the event from repainting the whole page

    async def apply_search(self, query: Optional[str] = None) -> None:
        """Rank off the loop; a newer keystroke (or ``set_query``) supersedes a running search."""
        if query is None:
            field = getattr(self, "search", None)
            query = (field.value or "") if field is not None else ""
        self._search_generation += 1
        generation = self._search_generation
        snap, filter_, known = self.snapshot, self.filter, self.builtin_keys
        try:
            models = await self.io(self._visible_for, snap, filter_, query, known)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.info("model search failed: %s", exc)
            return
        if generation != self._search_generation or self._disposed:
            return
        self.query = query
        self._precomputed = (self._inputs(snap, filter_, query, known), models)
        self._render_if_shown()

    def set_query(self, query: str) -> None:
        """Apply a search at once (no debounce)."""
        task = self._search_task
        if task is not None and not task.done():
            task.cancel()
        self._search_generation += 1
        self.query = query
        self._render_now()

    def set_filter(self, value: str) -> None:
        self.filter = "all" if self.filter == value else value
        self._render_now()

    def toggle_polled_only(self) -> bool:
        value = not self.snapshot.hide_unpolled
        self.catalog.set_hide_unpolled(value)
        self.snapshot = self.catalog.snapshot
        self._render_now()
        return value

    def _on_reorder(self, e: Any) -> None:
        old, new = int(getattr(e, "old_index", -1)), int(getattr(e, "new_index", -1))
        rows = self.rows
        mounted = len(rows.list_view.controls) if rows is not None else 0
        if rows is None or not (0 <= old < mounted and 0 <= new < mounted):
            return
        item = rows.items[rows.absolute(old)]
        old_abs, new_abs = rows.absolute(old), rows.absolute(new)
        if old_abs == new_abs:
            return
        if not self.reorderable:  # the client moved a row the list cannot reorder now: send it back
            self._shift(item, new_abs)
            self._shift(item, old_abs)
            return
        self._move(item, old_abs, new_abs, dragged=True)

    def _shift(self, item: tuple, index: int) -> None:
        """Move ``item``'s row to ``index`` of the whole list (the same row object) and push the list."""
        rows = self.rows
        if rows is None:
            return
        rows.remove_key(self._row_key(item))
        rows.insert_item(index, item)
        _push(rows.control)

    def _move(self, item: tuple, old_abs: int, new_abs: int, *, dragged: bool = False,
              success: Optional[str] = None) -> bool:
        """Save ``item``'s move in the whole list (``old_abs`` → ``new_abs``) and show it in the mounted
        rows (the same row object moves; the publish's render then finds nothing left to send).

        A drag is mirrored before the save: the client already shows it, and when the save fails the
        row is moved back, which the client applies. (Flet 1.0.3 matches rows by key, so re-sending the
        unchanged saved order would send nothing and leave the client's moved order on screen.) A row
        action moves the row only once the save succeeded."""
        if self.rows is None:
            return False
        if dragged:
            self._shift(item, new_abs)
        if self._edit(self.catalog.move_model(old_abs, new_abs), success):
            if not dragged:
                self._shift(item, new_abs)
            return True
        if dragged:
            self._shift(item, old_abs)
        return False

    def move_row(self, item: tuple, where: str) -> bool:
        """Row action Move to top / up / down / bottom (``where``: top | up | down | bottom), the desktop
        ⇈ ↑ ↓ ⇊ buttons. Indices are in the whole list, so a model in any window reaches any position
        (a drag stays inside the mounted rows)."""
        rows = self.rows
        if rows is None or not self.reorderable:
            return False
        old = rows.positions.get(self._row_key(item))
        if old is None:
            return False
        last = len(rows.items) - 1
        new = max(0, min(last, {"top": 0, "up": old - 1, "down": old + 1, "bottom": last}.get(where, old)))
        if new == old:
            return False
        model = item[0]
        # the row leaves the mounted rows: say where it went
        shown = rows.window_start <= new < rows.rendered
        success = None if shown else (f"Moved {model} to the top" if new == 0 else f"Moved {model} to row {new + 1:,}")
        return self._move(item, old, new, success=success)

    def open_row_actions(self, item: tuple) -> Optional[ActionSheet]:
        """Long-press on a row: Move to top / up / down / bottom and Remove (ActionSheet, UI_SPEC §5.2).
        The moves stay listed but unavailable, with the reason, while a search or filter is on."""
        rows = self.rows
        position = rows.positions.get(self._row_key(item)) if rows is not None else None
        if position is None:
            return None
        model, last = item[0], len(rows.items) - 1
        blocked = None if self.reorderable else "Clear the search and filters to reorder."
        at_top = blocked or ("Already at the top." if position == 0 else None)
        at_bottom = blocked or ("Already at the bottom." if position >= last else None)

        def move(where: str) -> Callable[[], Any]:
            return lambda: self.move_row(item, where)

        sheet = ActionSheet([
            ActionItem("Move to top", move("top"), icon=ft.Icons.VERTICAL_ALIGN_TOP, disabled_reason=at_top,
                       key="mm-move-top"),
            ActionItem("Move up", move("up"), icon=ft.Icons.ARROW_UPWARD, disabled_reason=at_top, key="mm-move-up"),
            ActionItem("Move down", move("down"), icon=ft.Icons.ARROW_DOWNWARD, disabled_reason=at_bottom,
                       key="mm-move-down"),
            ActionItem("Move to bottom", move("bottom"), icon=ft.Icons.VERTICAL_ALIGN_BOTTOM,
                       disabled_reason=at_bottom, key="mm-move-bottom"),
            ActionItem("Remove", lambda: self.remove(model), icon=ft.Icons.DELETE_OUTLINE, destructive=True,
                       key="mm-move-remove"),
        ], title=model, subtitle=f"Row {position + 1:,} of {last + 1:,}", tablet=self.tablet)
        self.last_sheet = sheet
        if self.page is not None:
            sheet.show(self.page)
        return sheet

    def remove(self, model: str) -> bool:
        before = list(self.snapshot.models)
        taken = self._take_rows(model)
        if not self._edit(self.catalog.remove_models([model])):
            self._put_back(taken)
            return False
        self.say(f"Removed {model}", "Undo", lambda: self._undo(before))
        return True

    def _undo(self, before: list) -> None:
        self._edit(self.catalog.save_order(before, list(self.catalog.snapshot.models)))

    def restore(self, model: str) -> bool:
        taken = self._take_rows(model) if self.filter == "removed" else []
        ok = self._edit(self.catalog.restore_removed([model], add_to_saved=True), f"Restored {model}")
        if not ok:
            self._put_back(taken)
        return ok

    def add(self, model: str) -> bool:
        return self._edit(self.catalog.add_model(model), f"Added {model.strip()}")

    def open_add(self) -> Any:
        if self.tab == "prefixes":
            return self.open_prefix_editor(None)
        field = ft.TextField(hint_text="Type a model ID to add…", autofocus=True, dense=True,
                             on_submit=lambda e: submit())

        def submit() -> None:
            text = (field.value or "").strip()
            if not text:
                return
            if not close_dialog(self.page, dialog) and self.page is not None:
                return  # already closed: a second tap / Enter never adds twice
            self.add(text)

        dialog = ft.AlertDialog(title=ft.Text("Add model"), content=field, actions=[
            ft.TextButton(content="Cancel", on_click=lambda e: close_dialog(self.page, dialog)),
            ft.FilledButton(content="➕ Add", on_click=lambda e: submit()),
        ])
        self.add_field, self.add_submit = field, submit
        self.last_dialog = dialog
        if self.page is not None:
            self.page.show_dialog(dialog)
        return dialog

    def confirm_reset(self) -> ConfirmDialog:
        dialog = ConfirmDialog(title="Reset Model List", body=RESET_TEXT, confirm_label="Reset", destructive=True,
                               on_confirm=lambda: self._edit(self.catalog.reset_to_defaults(), "Model list reset"))
        self.last_dialog = dialog
        if self.page is not None:
            dialog.show(self.page)
        return dialog

    async def poll(self) -> Any:
        """Desktop "🌐 Poll Providers": a full explicit poll, then the manager list becomes the polled
        catalog minus tombstones plus the genuinely custom entries (``manager_poll_models``).

        While it runs only the header changes (button, status text, provider chips); the list
        updates once at the end."""
        self.polling = True
        self._hold_list = True
        self.poll_status = "Contacting provider catalogs in the background…"
        self._render_now()
        previous = list(self.catalog.snapshot.models)
        outcome: Any = None
        try:
            if self.builtin_keys is None:
                await self._load()
            known = self.builtin_keys or frozenset()
            try:
                outcome = await self.catalog.refresh(None, explicit=True)
            finally:
                self.polling = False
            if not outcome.skipped:
                refreshed = self.catalog.manager_poll_models(previous, known)
                if refreshed:
                    ok, error = await self.io(lambda: self.catalog.save_order(refreshed, previous))
                    if not ok and error:
                        self.say(error)
                    known_fn = self.catalog.core.fn("known_catalog_keys")
                    result = self.catalog.last_result
                    if known_fn is not None and result is not None:
                        self.builtin_keys = known_fn(list(getattr(result, "models", []) or []))
            self.poll_status = outcome.message or READY_TEXT
        finally:
            self.polling = False
            self._hold_list = False
            self.snapshot = self.catalog.snapshot
            self._render_if_shown()
        return outcome

    # ---- custom prefixes ---------------------------------------------------------------------------------

    def routes(self) -> list:
        value = self.catalog.config.get("custom_prefix_routes", [])
        return [dict(r) for r in value if isinstance(r, dict)] if isinstance(value, list) else []

    def _prefix_rows(self) -> list:
        routes = self.routes()
        rows: list = [ft.Text("Use models like myprefix/model-name. Base URL is the provider root; Endpoint Type is "
                              "the path shape.", theme_style=ft.TextThemeStyle.BODY_SMALL,
                              color=ft.Colors.ON_SURFACE_VARIANT)]
        if not routes:
            rows.append(ft.Container(content=ft.Text("No custom prefixes yet.", color=ft.Colors.ON_SURFACE_VARIANT),
                                     padding=ft.Padding.all(12), key="prefix-empty"))
        for index, route in enumerate(routes):
            rows.append(ft.ListTile(
                title=ft.Text(str(route.get("prefix", ""))),
                subtitle=ft.Text(f"{route.get('routing', route.get('base_url', ''))} · "
                                 f"{route.get('endpoint_type', '/chat/completions')}", max_lines=2,
                                 overflow=ft.TextOverflow.ELLIPSIS),
                leading=ft.Icon(ft.Icons.ALT_ROUTE),
                trailing=ft.IconButton(icon=ft.Icons.DELETE_OUTLINE, tooltip="Delete", size_constraints=HIT_TARGET,
                                       on_click=lambda e, i=index: self.delete_prefix(i)),
                on_click=lambda e, i=index: self.open_prefix_editor(i),
                key=f"prefix-{index}",
            ))
        return rows

    def open_prefix_editor(self, index: Optional[int]) -> "PrefixEditor":
        routes = self.routes()
        current = routes[index] if index is not None and 0 <= index < len(routes) else None
        editor = PrefixEditor(route=current, on_save=lambda row: self.save_prefix(index, row))
        self.prefix_editor = editor
        if self.page is not None:
            editor.show(self.page)
        return editor

    def save_prefix(self, index: Optional[int], row: dict) -> Optional[str]:
        routes = self.routes()
        if index is not None and 0 <= index < len(routes):
            merged = dict(routes[index])
            merged.update(row)
            routes[index] = merged
        else:
            routes.append(row)
        ok, error = self.catalog.save_prefix_routes(routes)
        if not ok:
            return error[1] if isinstance(error, tuple) else str(error)
        self._render_now()
        return None

    def delete_prefix(self, index: int) -> ConfirmDialog:
        routes = self.routes()
        route = routes[index] if 0 <= index < len(routes) else {}

        def delete() -> None:
            remaining = [r for i, r in enumerate(self.routes()) if i != index]
            self.catalog.save_prefix_routes(remaining)
            self._render_now()

        dialog = ConfirmDialog(title="Delete custom prefix",
                               body=f"Delete 1 custom prefix route?\n\n{route.get('prefix', '')}",
                               confirm_label="Delete", destructive=True, on_confirm=delete)
        self.last_dialog = dialog
        if self.page is not None:
            dialog.show(self.page)
        return dialog



class PrefixEditor:
    """Add / edit one custom prefix route (prefix · base URL · endpoint type)."""

    def __init__(self, *, route: Optional[dict] = None, on_save: Callable[[dict], Optional[str]]) -> None:
        route = dict(route or {})
        self.on_save = on_save
        self.saved = False
        self._page: Any = None
        self.prefix = ft.TextField(value=str(route.get("prefix", "")), label="Prefix", hint_text="myprefix/", dense=True)
        self.routing = ft.TextField(value=str(route.get("routing", route.get("base_url", ""))), label="Base URL",
                                    hint_text="https://api.example.com/v1", dense=True, keyboard_type=ft.KeyboardType.URL)
        types = endpoint_type_options()
        current = str(route.get("endpoint_type", "/chat/completions") or "/chat/completions")
        self.endpoint_type = ft.Dropdown(value=current, label="Endpoint Type", editable=True, dense=True,
                                         options=[ft.DropdownOption(key=t, text=t) for t in dict.fromkeys([current, *types])])
        self.error = ft.Text("", color=ft.Colors.ERROR, visible=False)
        self.dialog = ft.BottomSheet(
            content=ft.Container(padding=ft.Padding.only(left=16, right=16, bottom=16), content=ft.Column([
                ft.Text("Custom prefix", theme_style=ft.TextThemeStyle.TITLE_LARGE, weight=ft.FontWeight.W_600),
                self.prefix, self.routing, self.endpoint_type, self.error,
                ft.Row([ft.TextButton(content="Cancel", on_click=lambda e: self.close()),
                        ft.FilledButton(content="Save", on_click=lambda e: self.save())],
                       alignment=ft.MainAxisAlignment.END),
            ], tight=True, spacing=10)),
            show_drag_handle=True, scrollable=True, bgcolor=ft.Colors.SURFACE_CONTAINER_HIGH,
        )

    def row(self) -> dict:
        return {"prefix": (self.prefix.value or "").strip(), "routing": (self.routing.value or "").strip(),
                "endpoint_type": (self.endpoint_type.value or "/chat/completions").strip()}

    def save(self) -> Optional[str]:
        if self.saved:  # a second tap while the sheet closes
            return None
        row = self.row()
        if not row["prefix"] and not row["routing"]:
            error: Optional[str] = "Row 1 needs both a prefix and Base URL."
        else:
            error = self.on_save(row)
        if error:
            self.error.value = error
            self.error.visible = True
            _push(self.error)
            return error
        self.saved = True
        self.close()
        return None

    def show(self, page: Any) -> None:
        self._page = page
        page.show_dialog(self.dialog)

    def close(self) -> None:
        close_dialog(self._page, self.dialog)


# ---- feature install ------------------------------------------------------------------------------------


class ModelsKeysFeature:
    """Catalog service + ModelSheet env + Models/Keys/Endpoints screens (call after the chat install)."""

    def __init__(self, app: Any) -> None:
        self.app = app
        self.page = getattr(app, "page", None)
        self.dispatcher = getattr(app, "dispatcher", None)
        settings = getattr(app, "settings", None)
        self.ctx = getattr(settings, "ctx", None)
        self.store = getattr(app, "config_store", None)
        self.prefs = getattr(app, "prefs", None)
        self.catalog = ModelCatalogService(self.store, run_io=self.run_io, post=self._post)
        from glossarion_mobile.ui.screens.endpoints import run_in_run_env
        from glossarion_mobile.ui.screens.keys import KeyBackend, KeysController

        self.backend = KeyBackend()
        self.keys = KeysController(self.store, self.backend, test_runner=run_in_run_env)
        self._fallback_factory: Optional[Callable[[RouteMatch], Any]] = None
        self._unsubs: list = []
        self.screens_built: list = []

    @classmethod
    async def install(cls, app: Any) -> "ModelsKeysFeature":
        feature = cls(app)
        feature.attach(app)
        return feature

    def attach(self, app: Any) -> None:
        from glossarion_mobile.ui.sheets.model_sheet import SheetEnv, install_sheet_env

        app.models_keys = self
        mc.set_default_service(self.catalog)
        install_sheet_env(SheetEnv(
            catalog=self.catalog, store=self.store, prefs=self.prefs, ctx=self.ctx,
            navigate=getattr(app, "navigate_to", None), notify=getattr(app, "notify", None),
            copy_text=getattr(app, "_copy_text", None),
            read_clipboard=getattr(getattr(app, "clipboard", None), "get", None),
            run_io=self.run_io, spawn=self.spawn, sign_in=self.sign_in, signed_in_keys=self.signed_in_keys,
            test_key=self.test_key, tablet=self._tablet, next_slot=self.next_slot,
            gemini_status=self.gemini_status, gcp_project_picker=self.gcp_project_picker,
        ))
        shell = getattr(app, "shell", None)
        if shell is not None and self._fallback_factory is None:
            self._fallback_factory = shell.screen_factory
            shell.screen_factory = self.screen_factory
        extras = getattr(self.ctx, "extras", None)
        if isinstance(extras, dict):
            # Settings › Response handling › "Manage refusal patterns…" (section_page STATIC_LINKS action)
            extras.setdefault("actions", {})["refusal_patterns"] = self.open_refusal_patterns
            # Settings › Custom API Endpoints (home, search hits, open_setting) opens the Endpoints page
            # (quick-paste chips, dependency ReasonChips, Local AI, Test connection), not a plain SectionPage
            extras.setdefault("section_screens", {})["other.endpoints"] = self.endpoints_section_screen
        if self.store is not None and not self._unsubs:
            observe = getattr(self.store, "observe_keys", None)
            if observe is not None:
                keys = ("custom_model_list", "model_manager_removed_models", "custom_prefix_routes",
                        "model_manager_hide_unpolled_models")
                self._unsubs.append(observe(keys, lambda key, value: self._post(self._schedule_reload)))
                # desktop: a model change auto-polls that provider once its 24 h TTL is due (debounced)
                self._unsubs.append(observe(("model", "api_key"), lambda key, value: self._post(self._schedule_auto_poll)))
        self.spawn(self.catalog.load())

    def gcp_project_picker(self, model: str) -> Any:
        """ModelSheet route row of ``authgem-vertex/`` models: the Accounts GCP project picker
        (``accounts.GcpProjectPicker``) for the model's Gemini slot; None without the OAuthBridge."""
        oauth = self._oauth()
        if oauth is None or self.store is None:
            return None
        from glossarion_mobile.ui.screens.accounts import GcpProjectPicker

        _route, account = mc.login_route(model)
        store = self.store
        return GcpProjectPicker(oauth, config_get=store.get, config_set=store.set_many, io=self.run_io,
                                slot=lambda: int(account or 0), key="route-gcp-project")

    def endpoints_section_screen(self, match: RouteMatch) -> Any:
        """``settings.section`` ``other.endpoints`` (``#key`` focuses that field): the Endpoints page."""
        if self.ctx is None:
            return None
        from glossarion_mobile.ui.screens.endpoints import EndpointsScreen

        return EndpointsScreen(match, self.ctx, run_io=self.run_io, open_local_ai=self.open_local_ai,
                               copy_text=getattr(self.app, "_copy_text", None),
                               read_clipboard=getattr(getattr(self.app, "clipboard", None), "get", None))

    async def gemini_status(self, account_id: int) -> Any:
        """ModelSheet › 📊: ``accounts.open_gemini_status`` with the chat feature's OAuthBridge."""
        oauth = getattr(getattr(self.app, "chat_feature", None), "oauth", None)
        if oauth is None:
            return None
        from glossarion_mobile.ui.screens.accounts import open_gemini_status

        notify = getattr(self.app, "notify", None)
        return await open_gemini_status(oauth, account_id, io=self.run_io, page=getattr(self.app, "page", None),
                                        say=notify)

    def detach(self) -> None:
        for unsub in self._unsubs:
            unsub()
        self._unsubs = []

    # ---- plumbing ---------------------------------------------------------------------------------------

    def _post(self, fn: Callable[..., Any], *args: Any) -> None:
        dispatcher = self.dispatcher
        if dispatcher is not None and getattr(dispatcher, "bound", False) and not dispatcher.on_loop_thread():
            dispatcher.post(fn, *args)
        else:
            fn(*args)

    def spawn(self, coro: Any) -> Any:
        dispatcher = self.dispatcher
        if dispatcher is not None and getattr(dispatcher, "bound", False):
            return dispatcher.spawn(coro)
        return asyncio.ensure_future(coro)

    async def run_io(self, fn: Callable[..., Any], *args: Any) -> Any:
        dispatcher = self.dispatcher
        if dispatcher is not None and getattr(dispatcher, "bound", False):
            return await dispatcher.run_in_thread(fn, *args, name="gl-models-io")
        return await asyncio.to_thread(fn, *args)

    def _schedule_reload(self) -> None:
        if getattr(self, "_reload_pending", False):
            return
        self._reload_pending = True

        async def reload() -> None:
            await asyncio.sleep(0.3)
            self._reload_pending = False
            # the config changed after any load in flight started; an unchanged result is not published
            await self.catalog.load(fresh=True)

        try:
            self.spawn(reload())
        except RuntimeError:  # no running loop (worker thread without a dispatcher)
            self._reload_pending = False

    def _schedule_auto_poll(self, delay: float = 1.2) -> None:
        """Desktop ``_schedule_current_provider_catalog_refresh``: poll after typing settles."""
        if getattr(self, "_auto_poll_pending", False):
            return
        self._auto_poll_pending = True

        async def poll() -> None:
            await asyncio.sleep(delay)
            self._auto_poll_pending = False
            try:
                await self.catalog.maybe_auto_poll()
            except Exception:
                log.debug("auto-poll failed", exc_info=True)

        try:
            self.spawn(poll())
        except RuntimeError:  # no running loop (a worker thread without a dispatcher)
            self._auto_poll_pending = False

    def _tablet(self) -> bool:
        return bool(getattr(getattr(self.app, "shell", None), "tablet", False))

    def _is_shown(self, screen: Any) -> bool:
        """``screen`` is what the user sees: the shell's top screen with no overlay View (Refusal
        patterns, Local AI) above it. True without a shell."""
        shell = getattr(self.app, "shell", None)
        if shell is None:
            return True
        if getattr(shell, "overlays", None):
            return False
        return getattr(shell, "top_screen", screen) is screen

    def _dark(self) -> bool:
        try:
            from glossarion_mobile.ui.theme import is_dark

            return bool(is_dark(self.page))
        except Exception:
            return False

    def _oauth(self) -> Any:
        return getattr(getattr(self.app, "chat_feature", None), "oauth", None)

    def signed_in_keys(self) -> frozenset:
        """``AppState.signed_in`` plus the OAuthBridge slot keys (``authgpt``, ``authgpt2``, ``authgem``…)."""
        state = getattr(self.app, "state", None)
        signed = set(getattr(getattr(state, "signed_in", None), "value", None) or ())
        oauth = self._oauth()
        try:
            signed.update(getattr(oauth, "signed_in", None) or ())
        except Exception:
            pass
        return frozenset(signed)

    def next_slot(self, route: str) -> int:
        """ModelSheet slot menu "+ Add account": OAuthBridge.next_slot (desktop slot allocation)."""
        oauth = self._oauth()
        if oauth is None:
            return 1
        try:
            return int(oauth.next_slot(route))
        except Exception:
            log.debug("next_slot failed for %s", route, exc_info=True)
            return 1

    def sign_in(self, route: str, account: int = 0) -> Any:
        """A sign-in chip / row: the Accounts LoginSheet for that provider and slot (OAuthBridge);
        the U3 ChatGPT sheet or the Accounts screen when the provider flow is not available."""
        oauth = self._oauth()
        if oauth is not None:
            try:
                from glossarion_mobile.ui.screens.accounts import LoginSheet

                sheet = LoginSheet(oauth, provider=route, account_id=int(account or 0),
                                   copy_text=getattr(self.app, "_copy_text", None))
                sheet.show(self.page)
                return sheet
            except (TypeError, KeyError, ImportError):
                log.debug("provider sign-in sheet unavailable for %s", route, exc_info=True)
        chat_view = getattr(self.app, "chat_view", None)
        opener = getattr(chat_view, "open_login_sheet", None)
        if route == "authgpt" and not account and opener is not None:
            return opener()
        navigate = getattr(self.app, "navigate_to", None)
        if navigate is not None:
            navigate("settings.accounts")
        return None

    async def test_key(self, api_key: str, model: str) -> dict:
        entry = {"api_key": api_key, "model": model}
        return await self.run_io(lambda: self.keys.test_entry(entry, "main"))

    # ---- screens -----------------------------------------------------------------------------------------

    def make_screen(self, match: RouteMatch) -> Any:
        notify = getattr(self.app, "notify", None)
        if match.name == "settings.models":
            screen: Any = ModelManagerScreen(match, catalog=self.catalog, page=self.page, notify=notify,
                                             spawn=self.spawn, run_io=self.run_io, tablet=self._tablet(),
                                             is_top=self._is_shown)
        elif match.name in ("settings.keys", "settings.keys.pool"):
            from glossarion_mobile.ui.screens.keys import KeysScreen
            from glossarion_mobile.ui.sheets.model_sheet import sheet_env

            paths = getattr(self.app, "paths", None)
            export_dir = os.path.join(str(paths.data), "Exports") if paths is not None else None
            screen = KeysScreen(match, controller=self.keys, ctx=self.ctx, sheet_env=sheet_env(),
                                files=getattr(self.app, "files", None), page=self.page, notify=notify,
                                copy_text=getattr(self.app, "_copy_text", None),
                                read_clipboard=getattr(getattr(self.app, "clipboard", None), "get", None),
                                run_io=self.run_io, spawn=self.spawn, open_refusal_patterns=self.open_refusal_patterns,
                                export_dir=export_dir, tablet=self._tablet(), dark=self._dark(), is_top=self._is_shown)
        elif match.name == "settings.endpoints" and self.ctx is not None:
            from glossarion_mobile.ui.screens.endpoints import EndpointsScreen

            screen = EndpointsScreen(match, self.ctx, run_io=self.run_io, open_local_ai=self.open_local_ai,
                                     copy_text=getattr(self.app, "_copy_text", None),
                                     read_clipboard=getattr(getattr(self.app, "clipboard", None), "get", None))
        else:
            return None
        self.screens_built.append(match.name)
        return screen

    def screen_factory(self, match: RouteMatch) -> Any:
        screen = None
        try:
            screen = self.make_screen(match)
        except Exception:
            log.exception("building the %s screen failed", match.name)
        if screen is None:
            if self._fallback_factory is None:
                raise LookupError(f"no screen for {match.name}")
            screen = self._fallback_factory(match)
            implemented = getattr(screen, "implemented", None)
            if match.name == "settings" and isinstance(implemented, frozenset):
                screen.implemented = implemented | IMPLEMENTED_ROUTES  # Settings home: no "Arrives in U4" chips
        return screen

    def _push_screen(self, screen: Screen) -> Optional[ft.View]:
        shell = getattr(self.app, "shell", None)
        if shell is None:
            return None
        view = build_screen_view(screen, shell.current_route)
        shell.push_overlay(view)
        try:
            self.page.update()
        except Exception:
            pass
        screen.did_show()
        return view

    def open_refusal_patterns(self) -> Any:
        from glossarion_mobile.ui.screens.refusal_patterns import RefusalModel, RefusalPatternsScreen

        screen = RefusalPatternsScreen(None, model=RefusalModel(self.store),
                                       page=self.page, files=getattr(self.app, "files", None),
                                       notify=getattr(self.app, "notify", None), run_io=self.run_io, spawn=self.spawn)
        self._push_screen(screen)
        return screen

    def open_local_ai(self) -> Any:
        from glossarion_mobile.ui.screens.local_ai import LocalAiScreen

        screen = LocalAiScreen(None, store=self.store, catalog=self.catalog, notify=getattr(self.app, "notify", None),
                               spawn=self.spawn, navigate=getattr(self.app, "navigate_to", None))
        self._push_screen(screen)
        return screen
