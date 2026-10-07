"""SectionPage (``/settings/s/<section>[#<key>]``, UI_SPEC §4.15, §5.7).

Built from ``settings_schema.sections()``/``spec()``: one schema-bound tile per
key, in schema order, inside ``ListView(build_controls_on_demand=False)`` so
every rendered tile is a valid ``scroll_to(scroll_key=)`` target. Sections
larger than ``WINDOW_SIZE`` tiles render a sliding window ("Show earlier /
Show more"); a jump (search result, ``#key`` deep link) first re-centres the
window on the target, then scrolls to it and highlights it for 1.5 s - the
procedure Flet 1.0.3 requires because ``scroll_to`` only reaches built items.

App bar: search (sheet with the shared ``SettingsSearch``) and ⋯ Save now ·
Discard changes since opening · Reset this section to defaults. A banner
says "Changes apply to the next run" while a job runs. Edits auto-save
through ``MobileConfigStore`` (600 ms debounce).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Optional

import flet as ft

from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.components.dialogs import ConfirmDialog
from glossarion_mobile.ui.components.empty_state import EmptyState
from glossarion_mobile.ui.router import RouteMatch
from glossarion_mobile.ui.screens.base import Screen
from glossarion_mobile.ui.settings.banners import JobBanner
from glossarion_mobile.ui.settings.model import WINDOW_SIZE, window_bounds
from glossarion_mobile.ui.settings.schema_access import SearchHit
from glossarion_mobile.ui.settings.search import SettingsSearch
from glossarion_mobile.ui.settings.tiles import EffectiveConfig, SettingTile, make_tile
from glossarion_mobile.ui.theme import HIT_TARGET

__all__ = ["HIGHLIGHT_SECONDS", "SectionPage"]

log = logging.getLogger("glossarion.settings")

HIGHLIGHT_SECONDS = 1.5


class SectionPage(Screen):
    def __init__(self, match: Optional[RouteMatch], ctx: Any, *, section_id: Optional[str] = None,
                 window_size: int = WINDOW_SIZE) -> None:
        super().__init__(match)
        self.ctx = ctx
        self.section_id = section_id or (match.params.get("section", "") if match is not None else "")
        self.focus_target = match.fragment if match is not None else None
        self.window_size = max(4, int(window_size))
        self.section = ctx.schema.section(self.section_id) if ctx.schema.available else None
        self.title = self.section.title if self.section is not None else "Settings"
        self.opened_snapshot = ctx.store.snapshot()
        self.config_view = EffectiveConfig(ctx.store)
        self.specs: list[Any] = []
        self.keys: list[str] = []
        self.tiles: dict[str, SettingTile] = {}
        self.window: tuple[int, int] = (0, 0)
        self.list_view: Optional[ft.ListView] = None
        self.banner = JobBanner(ctx)
        self.search_sheet: Optional[ft.BottomSheet] = None
        self.search: Optional[SettingsSearch] = None
        self.jumps: list[str] = []
        self._unsubs: list[Callable[[], None]] = []
        self._focus_task: Any = None

    # ---- body ------------------------------------------------------------------------------

    def build_body(self) -> ft.Control:
        if not self.ctx.schema.available:
            return ft.Column(
                [self.banner.control, EmptyState(
                    icon="SETTINGS", title="Settings schema unavailable",
                    body="This build has no settings schema "
                         f"({self.ctx.schema.error or 'settings_schema not importable'}). config.json is kept untouched.",
                    key="settings-no-schema")],
                expand=True,
            )
        if self.section is None:
            return EmptyState(icon="SETTINGS", title="Unknown settings section",
                              body=f"There is no section “{self.section_id}” in this build.", key="settings-unknown")
        self.specs = self.ctx.schema.specs_for(self.section)
        self.keys = [str(getattr(spec, "key", "")) for spec in self.specs]
        self.earlier_button = ft.TextButton(content="", icon=ft.Icons.EXPAND_LESS, on_click=self._on_earlier,
                                            key="settings-window-earlier")
        self.later_button = ft.TextButton(content="", icon=ft.Icons.EXPAND_MORE, on_click=self._on_later,
                                          key="settings-window-later")
        self.list_view = ft.ListView(
            expand=True,
            spacing=tokens.SPACING["xs"],
            padding=ft.Padding.symmetric(horizontal=tokens.SPACING["md"], vertical=tokens.SPACING["sm"]),
            build_controls_on_demand=False,
            auto_scroll=False,
        )
        start = 0
        if self.focus_target in self.keys:
            start = window_bounds(self.keys.index(self.focus_target), len(self.keys), self.window_size)[0]
        self.render_window(start, push=False)
        controls: list[ft.Control] = [self.banner.control]
        if not self.specs:
            controls.append(EmptyState(icon="SETTINGS", title=self.title, body="This section has no settings yet.",
                                       key="settings-empty-section"))
        controls.append(self.list_view)
        return ft.Column(controls, spacing=tokens.SPACING["sm"], expand=True)

    def actions(self) -> list[ft.Control]:
        self.search_button = ft.IconButton(icon=ft.Icons.SEARCH, tooltip="Search settings", on_click=self.open_search,
                                           size_constraints=HIT_TARGET)
        self.menu = ft.PopupMenuButton(
            icon=ft.Icons.MORE_VERT,
            tooltip="More",
            items=[
                ft.PopupMenuItem(content="Save now", icon=ft.Icons.SAVE_OUTLINED, on_click=self._on_save_now),
                ft.PopupMenuItem(content="Discard changes since opening", icon=ft.Icons.UNDO, on_click=self._on_discard),
                ft.PopupMenuItem(content="Reset this section to defaults", icon=ft.Icons.RESTART_ALT,
                                 on_click=self._on_reset_section),
            ],
        )
        return [self.search_button, self.menu]

    # ---- window ------------------------------------------------------------------------------

    def tile(self, key: str) -> SettingTile:
        tile = self.tiles.get(key)
        if tile is None:
            spec = self.specs[self.keys.index(key)]
            tile = make_tile(spec, self.ctx, config=self.config_view)
            self.tiles[key] = tile
        return tile

    @property
    def visible_keys(self) -> list[str]:
        start, end = self.window
        return self.keys[start:end]

    def render_window(self, start: int, push: bool = True) -> None:
        if self.list_view is None:
            return
        total = len(self.keys)
        start = max(0, min(start, max(0, total - self.window_size)))
        end = min(total, start + self.window_size)
        self.window = (start, end)
        controls: list[ft.Control] = []
        if start > 0:
            self.earlier_button.content = f"Show earlier settings ({start})"
            controls.append(self.earlier_button)
        for key in self.keys[start:end]:
            tile = self.tile(key)
            tile.refresh(push=False)
            controls.append(tile.control)
        if end < total:
            self.later_button.content = f"Show more settings ({total - end})"
            controls.append(self.later_button)
        self.list_view.controls = controls
        if push:
            self.ctx.push(self.list_view)

    def _on_earlier(self, e: Any = None) -> None:
        self.render_window(self.window[0] - self.window_size // 2)

    def _on_later(self, e: Any = None) -> None:
        self.render_window(self.window[0] + self.window_size // 2)

    async def focus_key(self, key: str, *, highlight: bool = True, settle: float = 0.05) -> bool:
        """Re-centre the window on ``key``, ``scroll_to`` its tile and highlight it."""
        if self.list_view is None or key not in self.keys:
            return False
        index = self.keys.index(key)
        start, end = self.window
        if not start <= index < end:
            self.render_window(window_bounds(index, len(self.keys), self.window_size)[0])
            await asyncio.sleep(settle)  # the client builds the new window first
        self.jumps.append(key)
        try:
            await self.list_view.scroll_to(scroll_key=ft.ScrollKey(key), duration=300)
        except Exception as exc:  # not mounted (yet) / client gone
            log.debug("scroll_to(%s) failed: %s", key, exc)
        tile = self.tiles.get(key)
        if tile is not None and highlight:
            tile.set_highlight(True)
            try:
                await asyncio.sleep(HIGHLIGHT_SECONDS)
            finally:
                tile.set_highlight(False)
        return True

    # ---- lifecycle -----------------------------------------------------------------------------

    def did_show(self) -> None:
        if not self._unsubs:
            self._unsubs.append(self.ctx.store.observe_all(lambda key, value: self.ctx.on_ui(self._on_config_change, key)))
            self.banner.attach()
            self._unsubs.append(self.banner.detach)
        if self.focus_target and self._focus_task is None and self.focus_target in self.keys:
            self._focus_task = self.ctx.spawn(self.focus_key(self.focus_target))

    def dispose(self) -> None:
        for unsub in self._unsubs:
            unsub()
        self._unsubs = []
        if self._focus_task is not None and not self._focus_task.done():
            self._focus_task.cancel()

    def _on_config_change(self, key: str) -> None:
        # Rules may depend on any key: refresh every rendered tile, then one diffed update.
        for visible in self.visible_keys:
            tile = self.tiles.get(visible)
            if tile is not None:
                tile.refresh(push=False)
        self.ctx.push(self.list_view)

    def refresh_all(self) -> None:
        """Re-render values (e.g. after the lazy schema defaults were resolved)."""
        self._on_config_change("")

    # ---- actions ---------------------------------------------------------------------------------

    async def _on_save_now(self, e: Any = None) -> bool:
        wrote = await self.ctx.run_io(self.ctx.store.flush)
        if self.ctx.store.save_error:
            self.ctx.say(self.ctx.store.save_error)
        else:
            self.ctx.say("Settings saved" if wrote else "No unsaved changes")
        return bool(wrote)

    def discard_changes(self) -> list[str]:
        changed = self.ctx.store.revert_to(self.opened_snapshot)
        self._on_config_change("")
        return changed

    def _on_discard(self, e: Any = None) -> None:
        changed = self.discard_changes()
        self.ctx.say(f"Discarded {len(changed)} change{'s' if len(changed) != 1 else ''}" if changed
                     else "Nothing changed since this page opened")

    def reset_section(self) -> list[str]:
        removed = []
        for key in self.keys:
            if not self.ctx.store.has(self.ctx.schema.path_of(key)):
                continue
            ok, _reason = self.ctx.schema.availability(key)
            if ok and self.ctx.schema.lock_reason(self.specs[self.keys.index(key)], self.config_view) is None:
                if self.ctx.store.unset(self.ctx.schema.path_of(key)):
                    removed.append(key)
        self._on_config_change("")
        return removed

    def _on_reset_section(self, e: Any = None) -> ConfirmDialog:
        stored = [k for k in self.keys if self.ctx.store.has(self.ctx.schema.path_of(k))]
        dialog = ConfirmDialog(
            title=f"Reset “{self.title}”?",
            body=(f"{len(stored)} stored value{'s' if len(stored) != 1 else ''} in this section go back to their "
                  "defaults (the keys are removed from config.json). Settings that are locked or unavailable on "
                  "mobile are kept. Use ⋯ Discard changes since opening to undo."),
            confirm_label="Reset",
            destructive=True,
            on_confirm=lambda: self.ctx.say(f"Reset {len(self.reset_section())} settings"),
        )
        dialog.show(self.ctx.page)
        return dialog

    # ---- search sheet ------------------------------------------------------------------------------

    def open_search(self, e: Any = None) -> ft.BottomSheet:
        self.search = SettingsSearch(self.ctx, on_open=self._open_hit, autofocus=True)
        self.search_sheet = ft.BottomSheet(
            content=ft.Container(
                padding=ft.Padding.only(left=12, right=12, bottom=12),
                height=520,
                content=ft.Column([self.search.field, self.search.filter_row, self.search.results],
                                  spacing=tokens.SPACING["sm"], expand=True,
                                  horizontal_alignment=ft.CrossAxisAlignment.STRETCH),
            ),
            show_drag_handle=True,
            scrollable=True,
            bgcolor=ft.Colors.SURFACE_CONTAINER_HIGH,
        )
        self.ctx.show_dialog(self.search_sheet)
        return self.search_sheet

    def _open_hit(self, hit: SearchHit) -> None:
        if self.search_sheet is not None and getattr(self.search_sheet, "open", False):
            self.ctx.pop_dialog(self.search_sheet)
        if hit.section_id == self.section_id:
            self.ctx.spawn(self.focus_key(hit.key))
        else:
            self.ctx.open_setting(hit.section_id, hit.key)
