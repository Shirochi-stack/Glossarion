"""Manga translator (``/tools/manga?tab=files|settings|editor``, UI_SPEC §4.6).

Three tabs over one ``MangaSession``: **Files** (selection, range, groups, Start / Stop, log,
CBZ / image export), **Settings** (OCR provider, detection model, context, glossary workflow,
inpainting, rendering and the schema sections of ``manga.settings``) and **Editor** (page strip,
Source / Translated, canvas boxes, workflow steps, the per-box sheet, OCR JSON import /
export). The tab is part of the route (``?tab=``), so a deep link or Back restores it.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import flet as ft

from glossarion_mobile.ui.router import RouteMatch
from glossarion_mobile.ui.screens.base import Screen

__all__ = ["MangaScreen", "TABS"]

log = logging.getLogger("glossarion.tools.manga")

TABS = ("files", "settings", "editor")
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

    def build_body(self) -> ft.Control:
        self.tabs = ft.Tabs(
            length=len(TABS),
            selected_index=TABS.index(self.initial_tab),
            on_change=self._on_tab,
            expand=True,
            content=ft.Column([
                ft.TabBar(tabs=[ft.Tab(label=_TAB_LABELS[name]) for name in TABS], key="manga-tabbar"),
                ft.TabBarView(controls=[self.tab_controls[name].build() for name in TABS], expand=True,
                              key="manga-tabview"),
            ], expand=True, spacing=0),
            key="manga-tabs",
        )
        return self.tabs

    @property
    def current_tab(self) -> str:
        index = getattr(self.tabs, "selected_index", TABS.index(self.initial_tab)) if self.tabs is not None else 0
        return TABS[index] if 0 <= index < len(TABS) else "files"

    def select_tab(self, name: str) -> None:
        if name not in TABS:
            return
        if self.tabs is None:
            self.initial_tab = name
            return
        self.tabs.selected_index = TABS.index(name)
        self.ctx.push(self.tabs)
        self._on_tab()

    def _on_tab(self, e: Any = None) -> None:
        tab = self.tab_controls.get(self.current_tab)
        if tab is not None:
            try:
                tab.did_show()
            except Exception:
                log.exception("manga tab %s did_show failed", self.current_tab)

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
        if tab == "files":
            if self.files_tab.selection_mode:
                self.files_tab.exit_selection()
                return True
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
