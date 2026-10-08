"""Manga translator (``/tools/manga?tab=files|settings|editor``, UI_SPEC §4.6).

Three tabs over one ``MangaSession``: **Files** (selection, range, groups, Start / Stop, log,
CBZ / image export), **Settings** (OCR provider, detection model, context, glossary workflow,
inpainting, rendering and the schema sections of ``manga.settings``) and **Editor** (page strip,
Source / Translated, canvas boxes, workflow steps, the per-box sheet, OCR JSON import /
export). The tab is part of the route (``?tab=``), so a deep link or Back restores it.

Wide screens (>= 1200 dp, UI_SPEC §1.1) lay it out master-detail: Files in a left pane, the
tab bar holding Editor · Settings; ``select_tab("files")`` then leaves the tabs alone. Each
tab builds its controls once; a size-class change (``apply_size_class``) only re-arranges them.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import flet as ft

from glossarion_mobile.ui.components import surface
from glossarion_mobile.ui.components.master_detail import MASTER_WIDTH
from glossarion_mobile.ui.router import RouteMatch
from glossarion_mobile.ui.screens.base import Screen

__all__ = ["MangaScreen", "TABS"]

log = logging.getLogger("glossarion.tools.manga")

TABS = ("files", "settings", "editor")
#: Tabs beside the Files pane on wide screens (master-detail).
WIDE_TABS = ("editor", "settings")
_TAB_LABELS = {"files": "Files", "settings": "Settings", "editor": "Editor"}


class MangaScreen(Screen):
    title = "Manga translator"

    def __init__(self, match: Optional[RouteMatch], ctx: Any, *, session: Any, feature: Any = None) -> None:
        super().__init__(match)
        self.ctx = ctx
        self.session = session
        self.feature = feature
        tab = (match.query.get("tab") if match is not None else None) or "files"
        self.initial_tab = tab if tab in TABS else "files"
        from glossarion_mobile.ui.tools.manga.editor import EditorTab
        from glossarion_mobile.ui.tools.manga.files import FilesTab
        from glossarion_mobile.ui.tools.manga.settings import SettingsTab

        self.files_tab = FilesTab(ctx, session, screen=self)
        self.settings_tab = SettingsTab(ctx, session, screen=self)
        self.editor_tab = EditorTab(ctx, session, screen=self)
        self.tab_controls = {"files": self.files_tab, "settings": self.settings_tab, "editor": self.editor_tab}
        self.tabs: Optional[ft.Tabs] = None
        self.shown = False
        self.wide = False  # master-detail: Files in its own pane (``_arrange``)
        self.tab_names: tuple = TABS
        self.tab_bodies: dict = {}
        self.layout_slot: Optional[ft.Container] = None
        self.layout_builds = 0

    def build_body(self) -> ft.Control:
        self.tab_bodies = {name: self.tab_controls[name].build() for name in TABS}
        self.layout_slot = ft.Container(expand=True, key="manga-layout")
        self._arrange(surface.is_wide(getattr(self.ctx, "page", None)), self.initial_tab)
        return self.layout_slot

    def _arrange(self, wide: bool, tab: str) -> None:
        """Tabs over the whole width (phone, tablet) or the Files pane + Editor · Settings (wide)."""
        self.wide = bool(wide)
        self.layout_builds += 1
        self.tab_names = WIDE_TABS if self.wide else TABS
        if tab not in self.tab_names:
            tab = self.tab_names[0]
        # The first build keeps the U8 keys; a re-arrangement gets fresh ones (Flet freezes a subtree
        # re-rendered under the key of the one it replaces).
        suffix = "" if self.layout_builds == 1 else f"-{self.layout_builds}"
        self.tabs = ft.Tabs(
            length=len(self.tab_names),
            selected_index=self.tab_names.index(tab),
            on_change=self._on_tab,
            expand=True,
            content=ft.Column([
                ft.TabBar(tabs=[ft.Tab(label=_TAB_LABELS[name]) for name in self.tab_names], key=f"manga-tabbar{suffix}"),
                ft.TabBarView(controls=[self.tab_bodies[name] for name in self.tab_names], expand=True,
                              key=f"manga-tabview{suffix}"),
            ], expand=True, spacing=0),
            key=f"manga-tabs{suffix}",
        )
        if self.wide:
            files = ft.Container(content=self.tab_bodies["files"], width=MASTER_WIDTH + 60,
                                 bgcolor=ft.Colors.SURFACE_CONTAINER_LOW, key=f"manga-files-pane{suffix}")
            self.layout_slot.content = ft.Row([files, self.tabs], spacing=0, expand=True,
                                              vertical_alignment=ft.CrossAxisAlignment.STRETCH)
        else:
            self.layout_slot.content = self.tabs

    def apply_size_class(self, size_class: Any) -> None:
        """Master-detail at >= 1200 dp; the tab on screen stays selected where it still exists."""
        wide = getattr(size_class, "value", size_class) == "wide"
        if self.layout_slot is None or wide == self.wide:
            return
        self._arrange(wide, self.current_tab)
        self.ctx.push(self.layout_slot)

    @property
    def current_tab(self) -> str:
        names = self.tab_names
        if self.tabs is None:
            return self.initial_tab if self.initial_tab in names else names[0]
        index = getattr(self.tabs, "selected_index", 0)
        return names[index] if 0 <= index < len(names) else names[0]

    def select_tab(self, name: str) -> None:
        if name not in TABS:
            return
        if self.tabs is None:
            self.initial_tab = name
            return
        if name not in self.tab_names:  # Files has its own pane on wide screens
            self._show_tab(name)
            return
        self.tabs.selected_index = self.tab_names.index(name)
        self.ctx.push(self.tabs)
        self._on_tab()

    def _show_tab(self, name: str) -> None:
        tab = self.tab_controls.get(name)
        if tab is not None:
            try:
                tab.did_show()
            except Exception:
                log.exception("manga tab %s did_show failed", name)

    def _on_tab(self, e: Any = None) -> None:
        self._show_tab(self.current_tab)

    def did_show(self) -> None:
        if self.feature is not None:
            self.feature.screen = self
        if not self.shown:
            self.shown = True
            self.ctx.spawn(self._first_show())
        else:
            self._on_tab()

    async def _first_show(self) -> None:
        session = self.session
        if not session.loaded:
            try:
                await self.ctx.io(session.load, self.ctx.config_snapshot())
            except Exception:
                log.exception("restoring the manga selection failed")
                session.loaded = True
        await self.take_pending()
        for tab in self.tab_controls.values():
            try:
                tab.on_session_loaded()
            except Exception:
                log.exception("manga tab refresh failed")
        if self.wide:
            self._show_tab("files")  # always on screen in the master pane
        self._on_tab()

    async def take_pending(self) -> int:
        """Files handed over by Open-with / the chat (``MangaFeature.open_with``)."""
        session = self.session
        if not session.pending or not session.loaded:
            return 0
        paths, session.pending = list(session.pending), []
        return await self.files_tab.add_paths(paths)

    def dispose(self) -> None:
        for tab in self.tab_controls.values():
            try:
                tab.dispose()
            except Exception:
                pass
        if self.feature is not None and self.feature.screen is self:
            self.feature.screen = None

    def handle_back(self) -> bool:
        """Android back (UI_SPEC §1.6) for the tab on screen: Files leaves selection mode first;
        the Editor drops the selected box, then the edit tool; otherwise the View pops."""
        tab = self.current_tab
        if tab == "files" or self.wide:  # wide: Files is always on screen
            if self.files_tab.selection_mode:
                self.files_tab.exit_selection()
                return True
            if not self.wide:
                return False
        if tab == "editor":
            return self.editor_tab.handle_back()
        return False

    def refresh_all(self) -> None:
        for tab in self.tab_controls.values():
            try:
                tab.refresh()
            except Exception:
                log.exception("manga tab refresh failed")
