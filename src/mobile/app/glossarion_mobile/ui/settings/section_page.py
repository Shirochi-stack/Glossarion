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
from glossarion_mobile.ui.settings.model import WINDOW_SIZE, grouped_specs, window_bounds
from glossarion_mobile.ui.settings.schema_access import SearchHit
from glossarion_mobile.ui.settings.search import SettingsSearch, search_sheet
from glossarion_mobile.ui.settings.tiles import MOBILE_READONLY_REASONS, EffectiveConfig, SettingTile, make_tile
from glossarion_mobile.ui.theme import HIT_TARGET

__all__ = ["HIGHLIGHT_SECONDS", "SECTION_POOL_KEYS", "STATIC_LINKS", "STATIC_ROWS", "SectionPage",
           "pool_toggle_key", "static_row"]

log = logging.getLogger("glossarion.settings")

HIGHLIGHT_SECONDS = 1.5

#: Desktop controls without a config key that mobile cannot offer: shown at the end of their
#: section as disabled rows with the reason (plan principle "nothing silently missing").
#: section id -> ((title, reason, detail), ...)
STATIC_ROWS: dict = {
    "other.processing.extraction": (
        ("Async chapter extraction (subprocess)", "No subprocesses on mobile",
         "The desktop can extract chapters in a helper process (USE_ASYNC_CHAPTER_EXTRACTION). Mobile apps "
         "cannot start processes, so extraction always runs in-process (run_env sets it off)."),
    ),
}


#: Actions the desktop shows next to a section's settings that live in a mobile tool or another
#: settings page: section id -> ((label, icon, route name[, params[, fragment]]), ...) shown as links at
#: the end of the section. A route ``action:<name>`` runs the feature action registered in
#: ``SettingsContext.extras['actions']`` (a full-screen view without a route, e.g. the Multi-Key
#: Manager's Refusal patterns); the link is hidden while no feature registered it.
STATIC_LINKS: dict = {
    # Output MD / Output TXT (Sidecars): the desktop "Generate MD" / "Generate TXT" retroactive buttons;
    # Validate EPUB Structure / Rename Files (retain source extension) / Load Font: Tools › Converter
    "epub_output": (("Generate MD / TXT for existing outputs…", "DESCRIPTION", "tools.convert"),
                    ("Validate EPUB · Rename files · Load font…", "BUILD_OUTLINED", "tools.convert")),
    # Convert <br> tags to <p>: the desktop "Apply to Existing Outputs" button
    "other.processing.extraction": (("Apply <br> → <p> to existing outputs…", "FORMAT_PARAGRAPH", "tools.convert"),),
    # Meta Data: Translate Headers Now / Delete Header Files / Delete TOC Files (Tools › Headers & metadata)
    "other.meta_data": (("Translate headers now / Delete header & TOC files…", "TITLE", "tools.headers"),),
    # Response handling › Duplicates: "Configure AI Hunter" (duplicate_detection_mode 'ai-hunter')
    # Response handling › Safety checks: "Manage refusal patterns" (FEATURE_MAP other-settings #80,
    # multikey-misc #20; the same RefusalPatternsScreen as the Multi-Key Manager footer)
    "other.response": (("Configure AI Hunter…", "MANAGE_SEARCH", "settings.section", {"section": "qa.ai_hunter"}),
                       ("Manage refusal patterns…", "BLOCK", "action:refusal_patterns")),
    # Image & vision: the global output mode lives with the run defaults (UI_SPEC §4.15); the image
    # compression master switch (shared with the Vision compression dialog) sits in EPUB output
    "other.image": (("Default output mode → Translation defaults", "TUNE", "settings.section",
                     {"section": "translation_defaults"}, "output_mode"),
                    ("Enable image compression → EPUB output", "PHOTO_SIZE_SELECT_LARGE", "settings.section",
                     {"section": "epub_output"}, "enable_image_compression")),
    # Direct Text: Chat settings › All chats edits these with their real types (the tiles here are read-only)
    "direct_text.settings": (("Edit in Chat settings › All chats…", "TUNE", "action:chat_settings_global"),),
    # PDF › Quality: the image compression switch and quality are shared with the EPUB output
    "pdf": (("Image compression & quality → EPUB output", "PHOTO_SIZE_SELECT_LARGE", "settings.section",
             {"section": "epub_output"}, "enable_image_compression"),),
    # Profile & System Prompt: profiles are edited on their own page (the JSON tile would bypass the desktop
    # save_profiles semantics)
    "main.prompt": (("Profiles & prompts…", "DESCRIPTION_OUTLINED", "settings.profiles"),
                    ("Assistant prefill…", "SHORT_TEXT", "settings.prefill")),
}

#: KeyPoolTiles (UI_SPEC §4.12 "KeyPoolTiles inside settings sections open /settings/keys/<pool>
#: directly"; they replace the desktop one-pool preview buttons RS Keys / Truncation Keys / Metadata
#: Keys / Fallback Keys and the Translation Keys status): section id -> pool config keys. Each renders
#: the pool's enable switch (``use_<pool>``) and its schema tile (count, tap -> the pool page).
SECTION_POOL_KEYS: dict = {
    "context_memory": ("rolling_summary_keys",),
    # Image & vision: the desktop "Vision Keys" (QA-scan pool) and "Image Keys" (inpainter pool) buttons next
    # to Configure Vision OCR Prompt / Output Resolution (the chat Mode options have them too)
    "other.image": ("qa_scan_keys", "inpainter_keys"),
    "other.response": ("multi_api_keys", "truncation_retry_keys"),
    "other.meta_data": ("metadata_keys",),
    "provider_options": ("fallback_keys",),
    "qa.settings": ("ai_truncation_detection_keys",),
    "glossary.refinement": ("glossary_refinement_keys",),  # the Glossary Refinement tab
}


#: STATIC_LINKS route prefix of a feature action (``SettingsContext.extras['actions'][name]``).
ACTION_PREFIX = "action:"


def pool_toggle_key(pool_key: str) -> str:
    """The enable switch of a pool stored under ``pool_key`` (``key_pool_service`` toggle_key)."""
    return {"multi_api_keys": "use_multi_api_keys"}.get(pool_key, "use_" + pool_key)


def static_row(title: str, reason: str, detail: str = "") -> ft.Control:
    """A disabled settings row with a ReasonChip (a desktop control that cannot exist on mobile)."""
    from glossarion_mobile.ui.components.reason_chip import ReasonChip

    return ft.Container(
        content=ft.ListTile(
            title=ft.Text(title, color=ft.Colors.ON_SURFACE_VARIANT),
            subtitle=ft.Row([ReasonChip(reason=reason, detail=detail or None)], wrap=True),
            trailing=ft.Switch(value=False, disabled=True),
            # not disabled=True: Flet would disable the ReasonChip too and its reason could not be opened
            min_height=tokens.SIZES["hit_target"],
        ),
        border_radius=tokens.RADII["tile"],
        bgcolor=ft.Colors.SURFACE_CONTAINER_LOW,
        key=f"static-{title}",
    )


class SectionPage(Screen):
    def __init__(self, match: Optional[RouteMatch], ctx: Any, *, section_id: Optional[str] = None,
                 window_size: int = WINDOW_SIZE) -> None:
        super().__init__(match)
        self.ctx = ctx
        self.section_id = section_id or (match.params.get("section", "") if match is not None else "")
        self.focus_target = match.fragment if match is not None else None
        display_key = getattr(ctx.schema, "display_key", None)
        if self.focus_target and callable(display_key):  # a key folded into a virtual control (Streaming)
            self.focus_target = display_key(self.focus_target)
        self.window_size = max(4, int(window_size))
        self.section = ctx.schema.section(self.section_id) if ctx.schema.available else None
        if self.section is not None and self.section.id != self.section_id and section_id is None:
            self.section_id = self.section.id  # a desktop id the curated map emptied (main.run)
        self.title = self.section.title if self.section is not None else "Settings"
        self.opened_snapshot = ctx.store.snapshot()
        self.config_view = EffectiveConfig(ctx.store)
        self.specs: list[Any] = []
        self.keys: list[str] = []
        self.headings: dict[str, str] = {}  # key -> sub-heading (desktop group box / nested path)
        self.tiles: dict[str, SettingTile] = {}
        self.pool_tiles: list[SettingTile] = []  # SECTION_POOL_KEYS (outside the window)
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
        # Sub-headings (U9): the desktop group boxes (QA Scanner › Foreign Character Detection,
        # Word Count Analysis, ...) or the nested path (AI Hunter › Thresholds / Weights).
        specs = self.ctx.schema.specs_for(self.section)
        curated = dict(getattr(self.section, "headings", ()) or ())
        if curated:  # UI_SPEC §4.15 curated section: its own sub-headings, in its order
            self.specs = list(specs)
            self.headings = {str(getattr(spec, "key", "")): curated.get(str(getattr(spec, "key", "")), "")
                             for spec in specs}
        else:
            self.specs, self.headings = grouped_specs(specs)
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
        heading = None
        for key in self.keys[start:end]:
            current = self.headings.get(key)
            if current and current != heading:
                controls.append(self._heading(current))
            heading = current
            tile = self.tile(key)
            tile.refresh(push=False)
            controls.append(tile.control)
        if end < total:
            self.later_button.content = f"Show more settings ({total - end})"
            controls.append(self.later_button)
        else:
            controls.extend(static_row(*row) for row in STATIC_ROWS.get(self.section_id, ()))
            pools = self.pool_controls()
            if pools:
                controls.append(self._heading("API key pools"))
                controls.extend(pools)
            for link in STATIC_LINKS.get(self.section_id, ()):
                label, icon, route = link[:3]
                params = link[3] if len(link) > 3 else None
                fragment = link[4] if len(link) > 4 else None
                if str(route).startswith(ACTION_PREFIX):
                    name = str(route)[len(ACTION_PREFIX):]
                    if not self.ctx_action(name):
                        continue
                    controls.append(ft.TextButton(
                        content=label, icon=getattr(ft.Icons, icon, None), key=f"link-action-{name}",
                        on_click=lambda e, n=name: self.run_action(n)))
                    continue
                slug = "-".join(str(v) for v in (route, *(params or {}).values()))
                controls.append(ft.TextButton(
                    content=label, icon=getattr(ft.Icons, icon, None), key=f"link-{slug}",
                    on_click=lambda e, r=route, p=params, f=fragment: self.ctx.go(r, p, fragment=f)))
        self.list_view.controls = controls
        if push:
            self.ctx.push(self.list_view)

    def ctx_action(self, name: str) -> Optional[Callable[[], Any]]:
        """The feature action a STATIC_LINKS ``action:<name>`` link runs (None: not registered)."""
        actions = (getattr(self.ctx, "extras", None) or {}).get("actions") or {}
        action = actions.get(name)
        return action if callable(action) else None

    def run_action(self, name: str) -> Any:
        action = self.ctx_action(name)
        if action is None:
            self.ctx.say("This page is not available in this session")
            return None
        result = action()
        if asyncio.iscoroutine(result):
            self.ctx.spawn(result)
        return result

    def pool_controls(self) -> list[ft.Control]:
        """The section's KeyPoolTiles: per pool its enable switch and its key-list tile (``make_tile`` of
        the schema spec: count, tap -> ``settings.keys.pool``); built once, refreshed on every render."""
        if not self.pool_tiles:
            for pool_key in SECTION_POOL_KEYS.get(self.section_id, ()):
                for key in (pool_toggle_key(pool_key), pool_key):
                    spec = self.ctx.schema.spec(key)
                    if spec is None or key in self.keys:
                        continue
                    try:
                        self.pool_tiles.append(make_tile(spec, self.ctx, config=self.config_view))
                    except Exception:
                        log.debug("pool tile %s failed", key, exc_info=True)
        for tile in self.pool_tiles:
            tile.refresh(push=False)
        return [tile.control for tile in self.pool_tiles]

    @staticmethod
    def _heading(text: str) -> ft.Control:
        return ft.Container(
            content=ft.Text(text, theme_style=ft.TextThemeStyle.TITLE_SMALL, color=ft.Colors.PRIMARY,
                            weight=ft.FontWeight.W_600),
            padding=ft.Padding.only(left=12, top=tokens.SPACING["md"], bottom=2),
        )

    def _on_earlier(self, e: Any = None) -> None:
        self.render_window(self.window[0] - self.window_size // 2)

    def _on_later(self, e: Any = None) -> None:
        self.render_window(self.window[0] + self.window_size // 2)

    async def focus_key(self, key: str, *, highlight: bool = True, settle: float = 0.05) -> bool:
        """Re-centre the window on ``key``, ``scroll_to`` its tile and highlight it."""
        display_key = getattr(self.ctx.schema, "display_key", None)
        key = display_key(key) if callable(display_key) else key
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
        for tile in self.pool_tiles:
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
            spec = self.specs[self.keys.index(key)]
            if getattr(spec, "virtual", ""):  # Context mode / Streaming: the tile resets the keys it stands for
                if self.tile(key).reset():
                    removed.append(key)
                continue
            if not self.ctx.store.has(self.ctx.schema.path_of(key)):
                continue
            ok, _reason = self.ctx.schema.availability(key)
            if getattr(spec, "readonly", ""):  # a mirror another control writes (Follows Output mode, ...)
                continue
            if key in MOBILE_READONLY_REASONS:  # edited on its own page (Prompt profiles: Profiles & prompts)
                continue
            if ok and self.ctx.schema.lock_reason(spec, self.config_view) is None:
                if self.ctx.store.unset(self.ctx.schema.path_of(key)):
                    removed.append(key)
        self._on_config_change("")
        return removed

    def _on_reset_section(self, e: Any = None) -> ConfirmDialog:
        is_stored = getattr(self.ctx.schema, "is_stored", None)
        stored = [k for k in self.keys if (is_stored(self.ctx.store, k) if callable(is_stored)
                                           else self.ctx.store.has(self.ctx.schema.path_of(k)))]
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
        self.search_sheet = search_sheet(self.search)
        self.ctx.show_dialog(self.search_sheet)
        return self.search_sheet

    def _open_hit(self, hit: SearchHit) -> None:
        if self.search_sheet is not None and getattr(self.search_sheet, "open", False):
            self.ctx.pop_dialog(self.search_sheet)
        if hit.section_id == self.section_id:
            self.ctx.spawn(self.focus_key(hit.key))
        else:
            self.ctx.open_setting(hit.section_id, hit.key)
