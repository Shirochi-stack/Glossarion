"""Glossary view (``/glossary/<gid>?tab=``; UI_SPEC §4.1): Editor · General · Balanced/Full · Minimal · Refinement.

A scrollable ``TabBar`` over the editor of one glossary file and the Glossary Manager's
settings tabs (global: the same tabs as Settings › Glossary). Opening the view runs the
Glossary Manager's mode lock pass once (``settings_rules.apply_glossary_mode_locks``), and
a mode change made while it is open runs it again, so the 🔒 badges and the forced toggle
values always follow the mode. Android back leaves the editor's selection mode first;
leaving with unsaved edits asks first - by Back, and through the shell's leave guard
(:meth:`GlossaryScreen.confirm_leave`) for a drawer / sidebar destination, a chat row, a link
from outside the app or a View popped below the editor. The app bar has Extract glossary and
⋯ (Glossary progress · Share · Reload); its title follows the open file. The Glossaries list
the ◀ ▶ / file name ▾ step through and the glossary's Library input are read on the io pool.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Optional

import flet as ft

from glossarion_mobile.ui.components.empty_state import EmptyState
from glossarion_mobile.ui.glossary.common import attach_mode_locks, mode_locks_changed
from glossarion_mobile.ui.glossary.editor import EditorPane
from glossarion_mobile.ui.glossary.settings_tabs import GlossarySettingsTab
from glossarion_mobile.ui.router import RouteMatch
from glossarion_mobile.ui.screens.base import Screen
from glossarion_mobile.ui.theme import HIT_TARGET

__all__ = ["GLOSSARY_TABS", "GlossaryScreen"]

log = logging.getLogger("glossarion.glossary.ui")

GLOSSARY_TABS = ("editor", "general", "balanced", "minimal", "refinement")
_TAB_LABELS = {"editor": "Editor", "general": "General", "balanced": "Balanced/Full", "minimal": "Minimal",
               "refinement": "Refinement"}


class GlossaryScreen(Screen):
    def __init__(self, match: Optional[RouteMatch], ctx: Any, *, path: Optional[str] = None,
                 source_path: Optional[str] = None) -> None:
        super().__init__(match)
        self.ctx = ctx
        self.service = ctx.service
        self.gid = str(match.params.get("gid") or "") if match is not None else ""
        self.path = path or (self.service.path_for_gid(self.gid) if self.gid else None)
        self._source_path = source_path
        self._source_resolved = bool(source_path)  # a "not found" is remembered too
        tab = match.get("tab") if match is not None else None
        self.initial_tab = tab if tab in GLOSSARY_TABS else "editor"
        self.title = os.path.basename(self.path) if self.path else "Glossary"
        self.editor = EditorPane(self)
        self.settings_tabs = {name: GlossarySettingsTab(ctx, name) for name in GLOSSARY_TABS[1:]}
        self._unsubs: list = []
        self.shown = False
        self.listing_task: Any = None  # the Glossaries list being read (load_listing)
        self._leave_task: Any = None  # the "Unsaved changes" question being asked (confirm_leave)

    # ---- helpers the editor uses ---------------------------------------------------------------------

    def sibling_files(self) -> list:
        """The Glossaries list ◀ ▶ / file name ▾ step through: the feature's cached listing only (read on the
        io pool by :meth:`load_listing`; never scanned on the UI loop)."""
        feature = self.ctx.feature
        return list(getattr(feature, "listing", None) or []) if feature is not None else []

    async def load_listing(self, *, force: bool = False) -> list:
        """Read the Glossaries list on the io pool when the feature has none yet (the editor opened directly:
        Book page, chat card, Add to glossary, a deep link; or after Save As), then show ◀ ▶."""
        feature = self.ctx.feature
        if feature is None or not self.path:
            return []
        if feature.listing and not force:
            return list(feature.listing)
        try:
            listing = list(await self.ctx.io(self.service.list_glossaries))
        except Exception:
            log.debug("listing the glossaries failed", exc_info=True)
            return []
        feature.listing = listing
        editor = self.editor
        if editor.root is not None:
            editor._sync_nav()
            self.ctx.push(editor.prev_button, editor.next_button)
        return listing

    def _spawn_listing(self, *, force: bool = False) -> None:
        self.listing_task = self.ctx.spawn(self.load_listing(force=force))

    def source_path(self) -> Optional[str]:
        """The input this glossary belongs to (Library raw source) as resolved so far
        (:meth:`resolve_source_path` resolves it)."""
        return self._source_path

    async def resolve_source_path(self) -> Optional[str]:
        """The input this glossary belongs to (Library raw source), for hide-unused / output files / Extract.
        Resolved once on the io pool (the Library resolvers read registry files and validate EPUBs)."""
        if self._source_path or self._source_resolved or not self.path:
            return self._source_path
        path = self.path
        try:
            found = await self.ctx.io(self._resolve_source_blocking, path)
        except Exception:
            found = None
        if path == self.path:  # not switched to another file meanwhile
            self._source_path = found
            self._source_resolved = True
        return found

    def _resolve_source_blocking(self, path: str) -> Optional[str]:
        from glossarion_mobile.services.glossary import GlossaryFile

        folder = os.path.dirname(path)
        row = GlossaryFile(path=path, kind="book", name="", book=os.path.basename(folder), folder=folder)
        book = self.service.book_for_glossary(row)
        if book is None or self.service.library is None:
            return None
        try:
            return self.service.library.raw_source(book) or None
        except Exception:
            return None

    def export_dir(self) -> str:
        files = self.ctx.files
        inbox = getattr(files, "inbox_dir", None) if files is not None else None
        base = inbox or (os.path.dirname(self.path) if self.path else os.getcwd())
        return os.path.join(base, "Exports")

    async def switch_to(self, path: str) -> bool:
        """Open another glossary in this screen; False when the user kept the open file ("Unsaved changes" ›
        Cancel), in which case the title, gid and input stay the open file's."""
        if not await self.editor.confirm_switch(path):
            return False
        self.path = path
        self.gid = self.service.gid_for(path)
        self.title = os.path.basename(path)
        self._source_path = None
        self._source_resolved = False
        self._set_title()
        await self.editor.open(path, confirmed=True)
        return True

    def note_switched(self, path: str) -> None:
        """Save As: the open document now is ``path`` (title, gid, the file list refreshed)."""
        self.path = path
        self.gid = self.service.gid_for(path)
        self.title = os.path.basename(path)
        feature = self.ctx.feature
        if feature is not None:
            feature.listing = []
            self._spawn_listing(force=True)
        self._set_title()

    def _set_title(self) -> None:
        """The app bar (phone View) / main-area title bar (tablet) shows the open file's name: the shell
        records the title Text it built for this screen as ``app_bar_title``."""
        bar = getattr(self, "app_bar_title", None)
        if bar is not None:
            bar.value = self.title
            self.ctx.push(bar)

    # ---- app bar ----------------------------------------------------------------------------------------

    async def open_extract(self) -> Any:
        """App bar "Extract glossary": the Extract sheet with this glossary's Library input as "This file"."""
        feature = self.ctx.feature
        if feature is None:
            return None
        return feature.open_extract_sheet(await self.resolve_source_path())

    def actions(self) -> list:
        feature = self.ctx.feature
        self.extract_button = ft.IconButton(icon=ft.Icons.AUTO_AWESOME, tooltip="Extract glossary",
                                            on_click=lambda e: self.ctx.spawn(self.open_extract())
                                            if feature else None, key="gv-extract", size_constraints=HIT_TARGET)
        self.menu = ft.PopupMenuButton(icon=ft.Icons.MORE_VERT, tooltip="Glossary", key="gv-menu", items=[
            ft.PopupMenuItem(content="Glossary progress", icon=ft.Icons.PLAYLIST_ADD_CHECK,
                             on_click=lambda e: feature.open_progress_for_path(self.path) if feature else None),
            ft.PopupMenuItem(content="Reload", icon=ft.Icons.REFRESH,
                             on_click=lambda e: self.ctx.spawn(self.editor.reload(force=True))),
            ft.PopupMenuItem(content="Share file", icon=ft.Icons.SHARE,
                             on_click=lambda e: self.ctx.spawn(self.editor.share_file())),
            ft.PopupMenuItem(content="Search settings", icon=ft.Icons.SEARCH,
                             on_click=lambda e: self.search_settings()),
            ft.PopupMenuItem(content="Discard settings changes since opening", icon=ft.Icons.UNDO,
                             on_click=lambda e: self.discard_settings_changes()),
        ])
        return [self.extract_button, self.menu]

    def _settings_page(self) -> Any:
        """The SectionPage of the selected settings tab (None on the Editor tab)."""
        tab = self.settings_tabs.get(self.current_tab) if hasattr(self, "tabs") else None
        return getattr(tab, "page", None) if tab is not None else None

    def search_settings(self) -> Any:
        page = self._settings_page()
        if page is None:
            settings = getattr(self.ctx, "settings", None)
            if settings is not None:
                from glossarion_mobile.ui.settings.search import SettingsSearch, search_sheet

                sheet = search_sheet(SettingsSearch(settings, on_open=lambda hit: settings.open_setting(
                    hit.section_id, hit.key), autofocus=True))
                settings.show_dialog(sheet)
                return sheet
            return None
        return page.open_search()

    def discard_settings_changes(self) -> list:
        """⋯ Discard changes since opening (desktop Glossary Manager Cancel): the selected settings
        tab's SectionPage restores the config it opened with."""
        page = self._settings_page()
        if page is None:
            self.ctx.say("Open a settings tab (General, Balanced/Full, Minimal, Refinement) first")
            return []
        changed = page.discard_changes()
        self.ctx.say(f"Discarded {len(changed)} change{'s' if len(changed) != 1 else ''}" if changed
                     else "Nothing changed since this tab opened")
        return changed

    # ---- body ---------------------------------------------------------------------------------------------

    def build_body(self) -> ft.Control:
        if not self.path:
            return EmptyState(icon="SPELLCHECK", title="Glossary not found",
                              body="This glossary is no longer available.", key="gv-missing",
                              primary=("Glossaries", lambda e: self.ctx.go("glossary")))
        self.tab_index = {name: i for i, name in enumerate(GLOSSARY_TABS)}
        self.tabs = ft.Tabs(
            length=len(GLOSSARY_TABS),
            selected_index=self.tab_index[self.initial_tab],
            on_change=self._on_tab,
            expand=True,
            content=ft.Column([
                ft.TabBar(tabs=[ft.Tab(label=_TAB_LABELS[name]) for name in GLOSSARY_TABS], scrollable=True,
                          key="gv-tabbar"),
                ft.TabBarView(controls=[self.editor.build()] + [tab.build() for tab in self.settings_tabs.values()],
                              expand=True, key="gv-tabview"),
            ], expand=True, spacing=0),
            key="gv-tabs",
        )
        return self.tabs

    @property
    def current_tab(self) -> str:
        tabs = getattr(self, "tabs", None)
        index = getattr(tabs, "selected_index", 0) if tabs is not None else 0
        return GLOSSARY_TABS[index] if 0 <= index < len(GLOSSARY_TABS) else "editor"

    def set_tab(self, name: str) -> None:
        if name not in GLOSSARY_TABS or getattr(self, "tabs", None) is None:
            return
        self.tabs.selected_index = self.tab_index[name]
        self.ctx.push(self.tabs)
        self._on_tab()

    def _on_tab(self, e: Any = None) -> None:
        tab = self.settings_tabs.get(self.current_tab)
        if tab is not None:
            tab.did_show()

    # ---- lifecycle ----------------------------------------------------------------------------------------

    def did_show(self) -> None:
        if not self.path:
            return
        if not self.shown:
            self.shown = True
            # the Glossary Manager's lock pass on open and on every mode change (shared with Settings › Glossary)
            self._unsubs.extend(attach_mode_locks(self.ctx))
            self.ctx.spawn(self.editor.open(self.path, source_path=self._source_path))
            if self.ctx.feature is not None and not self.ctx.feature.listing:
                self._spawn_listing()
        self.editor.start_polling()
        self._on_tab()

    def _on_mode_changed(self) -> None:
        mode_locks_changed(self.ctx)

    def handle_back(self) -> bool:
        if self.editor.selecting:
            self.editor.exit_selection()
            return True
        if self.editor.doc is not None and self.editor.doc.dirty:
            self.ctx.spawn(self._confirm_leave())
            return True
        return False

    async def _confirm_leave(self) -> None:
        """Back with unsaved edits: the question, then the screen pops on Discard."""
        if await self.confirm_leave():
            pop = getattr(self.ctx, "pop_overlay", None)
            if callable(pop):
                pop()

    async def confirm_leave(self) -> bool:
        """The shell's leave guard (``Screen.confirm_leave``): True when this screen may be disposed. With
        unsaved edits it asks "Unsaved changes" (Discard: the edits are dropped; Keep editing: False, the
        navigation is cancelled). Questions asked meanwhile share the one dialog."""
        doc = self.editor.doc
        if doc is None or not getattr(doc, "dirty", False):
            return True
        pending = self._leave_task
        if pending is None or pending.done():
            pending = self._leave_task = asyncio.ensure_future(self._ask_discard())
        return bool(await asyncio.shield(pending))

    async def _ask_discard(self) -> bool:
        from glossarion_mobile.ui.glossary.common import ask

        discard = await ask(self.ctx, title="Unsaved changes", body="Discard the unsaved glossary changes?",
                            confirm="Discard", cancel="Keep editing", destructive=True)
        if discard and self.editor.doc is not None:
            self.editor.doc.dirty = False
        return bool(discard)

    def dispose(self) -> None:
        self.editor.stop_polling()
        for tab in self.settings_tabs.values():
            tab.dispose()
        for unsub in self._unsubs:
            try:
                unsub()
            except Exception:
                pass
        self._unsubs = []
