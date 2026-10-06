"""Glossaries home (``/glossary``; UI_SPEC §4.1 "Home"): every glossary file in one list.

Rows: book glossaries (``Glossary/<book>/``), Minimal-mode output glossaries
(``<output>/<book>/glossary.csv``), manual / imported glossaries and the unified glossaries
(``Glossary/Unified Glossary/<lang>/``), each with its kind, entry count, book link and
modified time (``GlossaryService.list_glossaries``; counts fill in from the io pool).

* Search ("Filter glossaries…"), kind chips All · Book · Manual · Unified, the glossary mode
  chip ("Glossary mode: Balanced ▾" → the 8 desktop modes; the mode locks follow).
* Row tap → the glossary view (``/glossary/<gid>``); row ⋯ → Open · Use as manual glossary ·
  Share · Glossary progress · Delete glossary files · Restore backup (the desktop main-window
  🗑️ / ↩️ for that book, with their confirmation texts).
* App bar ⋯: Extract glossary · Parallel EPUB pair · Unified glossary · Glossary progress ·
  Refresh. FAB "Import glossary" (FileBridge picker: csv / json / txt / md into the Inbox).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import flet as ft

from glossarion_mobile.services.glossary import GlossaryFile
from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.components.action_sheet import ActionItem, ActionSheet
from glossarion_mobile.ui.components.empty_state import EmptyState
from glossarion_mobile.ui.glossary.common import ago, kind_icon, kind_label
from glossarion_mobile.ui.router import RouteMatch
from glossarion_mobile.ui.screens.base import Screen
from glossarion_mobile.ui.theme import HIT_TARGET, icon_data

__all__ = ["GlossariesScreen", "KIND_FILTERS"]

log = logging.getLogger("glossarion.glossary.ui")

KIND_FILTERS = (("all", "All"), ("book", "Book"), ("manual", "Manual"), ("unified", "Unified"))


class GlossariesScreen(Screen):
    title = "Glossaries"

    def __init__(self, match: Optional[RouteMatch], ctx: Any) -> None:
        super().__init__(match)
        self.ctx = ctx
        self.service = ctx.service
        self.files: list = []
        self.kind = "all"
        self.query = ""
        self.loaded = False
        self.rows: dict = {}
        self._counting: Any = None

    # ---- app bar -------------------------------------------------------------------------------------

    def actions(self) -> list:
        feature = self.ctx.feature

        def item(label: str, handler: Any, icon: Any) -> ft.PopupMenuItem:
            return ft.PopupMenuItem(content=label, icon=icon, on_click=lambda e: handler())

        self.menu = ft.PopupMenuButton(icon=ft.Icons.MORE_VERT, tooltip="Glossaries", key="gh-menu", items=[
            item("Extract glossary…", lambda: feature.open_extract_sheet() if feature else None, ft.Icons.AUTO_AWESOME),
            item("Parallel EPUB pair", lambda: self.ctx.go("glossary.parallel_pair"), ft.Icons.COMPARE_ARROWS),
            item("Unified glossary", lambda: self.ctx.go("glossary.unified"), ft.Icons.MERGE_TYPE),
            item("Glossary progress", lambda: self.ctx.go("tools.progress.glossary"), ft.Icons.PLAYLIST_ADD_CHECK),
            item("Refresh", lambda: self.ctx.spawn(self.refresh()), ft.Icons.REFRESH),
        ])
        return [self.menu]

    # ---- body -------------------------------------------------------------------------------------------

    def build_body(self) -> ft.Control:
        self.mode_chip = ft.Chip(label=ft.Text(self._mode_text()), leading=ft.Icon(ft.Icons.RULE, size=16),
                                 on_click=lambda e: self.open_mode_sheet(), key="gh-mode")
        self.search = ft.TextField(hint_text="Filter glossaries…", prefix_icon=ft.Icons.SEARCH, dense=True,
                                   on_change=lambda e: self._on_search(), key="gh-search")
        self.kind_buttons = ft.SegmentedButton(
            selected=[self.kind], show_selected_icon=False, on_change=self._on_kind, key="gh-kind",
            segments=[ft.Segment(value=k, label=ft.Text(label)) for k, label in KIND_FILTERS])
        self.count_text = ft.Text("", theme_style=ft.TextThemeStyle.LABEL_MEDIUM, key="gh-count")
        self.loading = ft.ProgressBar(visible=True, key="gh-loading")
        self.list_view = ft.ListView(spacing=4, padding=ft.Padding.only(left=8, right=8, bottom=96), expand=True,
                                     build_controls_on_demand=True, key="gh-list")
        self.list_holder = ft.Container(content=self.list_view, expand=True, key="gh-list-holder")
        self.fab = ft.FloatingActionButton(icon=ft.Icons.FILE_UPLOAD, content="Import glossary",
                                           on_click=lambda e: self.ctx.spawn(self.import_glossary()), key="gh-fab")
        self.root = ft.Column([
            ft.Container(content=ft.Column([
                ft.Row([self.mode_chip], wrap=True),
                self.search,
                ft.Row([self.kind_buttons], scroll=ft.ScrollMode.AUTO),
                self.count_text,
            ], spacing=6, tight=True), padding=ft.Padding.only(left=12, right=12, top=8)),
            self.loading,
            ft.Stack([self.list_holder, ft.Container(content=self.fab, right=16, bottom=16)], expand=True),
        ], spacing=4, expand=True, key="glossaries-home")
        return self.root

    def _mode_text(self) -> str:
        return f"Glossary mode: {self.service.mode_label()} ▾"

    def did_show(self) -> None:
        self.ctx.spawn(self.refresh())

    def dispose(self) -> None:
        task = self._counting
        if task is not None and not task.done():
            task.cancel()

    # ---- data -------------------------------------------------------------------------------------------

    async def refresh(self) -> list:
        try:
            files = await self.ctx.io(self.service.list_glossaries)
        except Exception as exc:
            log.exception("listing glossaries failed")
            self.ctx.say(f"Could not list the glossaries: {exc}")
            files = []
        self.files = list(files)
        self.loaded = True
        feature = self.ctx.feature
        if feature is not None:
            feature.listing = list(self.files)
        self.render()
        task = self._counting
        if task is not None and not task.done():
            task.cancel()
        self._counting = self.ctx.spawn(self._fill_counts())
        return self.files

    async def _fill_counts(self) -> None:
        for index, row in enumerate(list(self.files)):
            if row.entries is not None:
                continue
            count = await self.ctx.io(self.service.count_entries, row.path)
            if count is None:
                continue
            from dataclasses import replace

            updated = replace(row, entries=count)
            if index < len(self.files) and self.files[index].path == row.path:
                self.files[index] = updated
            control = self.rows.get(row.key)
            if control is not None:
                control.subtitle = ft.Text(self._subtitle(updated), max_lines=2, overflow=ft.TextOverflow.ELLIPSIS)
                self.ctx.push(control)

    def visible_files(self) -> list:
        rows = self.files
        if self.kind == "book":
            rows = [f for f in rows if f.kind in ("book", "output")]
        elif self.kind != "all":
            rows = [f for f in rows if f.kind == self.kind]
        query = self.query.casefold().strip()
        if query:
            rows = [f for f in rows if query in f.name.casefold() or query in f.book.casefold()]
        return rows

    def _subtitle(self, row: GlossaryFile) -> str:
        parts = [kind_label(row.kind)]
        if row.book:
            parts.append(row.book)
        elif row.language_key:
            parts.append(row.language_key)
        if row.entries is not None:
            parts.append(f"{row.entries:,} entries")
        if row.mtime:
            parts.append(ago(row.mtime))
        return " · ".join(parts)

    def render(self) -> None:
        if getattr(self, "root", None) is None:
            return
        self.loading.visible = not self.loaded
        self.mode_chip.label = ft.Text(self._mode_text())
        rows = self.visible_files()
        self.count_text.value = f"{len(rows)} glossar{'ies' if len(rows) != 1 else 'y'}" if self.loaded else ""
        self.rows = {}
        if self.loaded and not self.files:
            feature = self.ctx.feature
            self.list_holder.content = EmptyState(
                icon="SPELLCHECK", title="No glossaries yet",
                body="Extract a glossary from a book, or import a CSV / JSON glossary.", key="gh-empty",
                primary=("Extract glossary", lambda e: feature.open_extract_sheet() if feature else None),
                secondary=("Import", lambda e: self.ctx.spawn(self.import_glossary())))
        elif self.loaded and not rows:
            self.list_holder.content = EmptyState(icon="FILTER_LIST_OFF", title="No glossaries match",
                                                  body="Change the search or the kind filter.", key="gh-nomatch")
        else:
            controls = []
            for row in rows:
                tile = ft.ListTile(
                    leading=ft.Icon(icon_data(kind_icon(row.kind))),
                    title=ft.Text(row.name, max_lines=2, overflow=ft.TextOverflow.ELLIPSIS),
                    subtitle=ft.Text(self._subtitle(row), max_lines=2, overflow=ft.TextOverflow.ELLIPSIS),
                    trailing=ft.IconButton(icon=ft.Icons.MORE_VERT, tooltip="Glossary actions",
                                           size_constraints=HIT_TARGET, on_click=lambda e, r=row: self.open_row_sheet(r)),
                    on_click=lambda e, r=row: self.open(r),
                    on_long_press=lambda e, r=row: self.open_row_sheet(r),
                    min_height=tokens.SIZES["row_two_line"],
                    key=f"gh-row-{row.gid}",
                )
                self.rows[row.key] = tile
                controls.append(tile)
            self.list_view.controls = controls
            self.list_holder.content = self.list_view
        self.ctx.push(self.root)

    def _on_search(self) -> None:
        self.query = self.search.value or ""
        self.render()

    def _on_kind(self, e: Any = None) -> None:
        selected = list(self.kind_buttons.selected or ["all"])
        self.kind = selected[0] if selected else "all"
        self.render()

    # ---- actions -----------------------------------------------------------------------------------------

    def open(self, row: GlossaryFile) -> None:
        self.ctx.go("glossary.detail", {"gid": row.gid})

    def open_row_sheet(self, row: GlossaryFile) -> ActionSheet:
        feature = self.ctx.feature
        book_kind = row.kind in ("book", "output")
        files = self.ctx.files
        items = [
            ActionItem("Open", lambda: self.open(row), icon="EDIT_NOTE"),
            ActionItem("Use as manual glossary", lambda: self.ctx.spawn(feature.use_as_manual(row.path)),
                       icon="FILE_OPEN", disabled_reason=None if feature is not None else "Not available"),
            ActionItem("Share", lambda: self.ctx.spawn(files.share([row.path])), icon="IOS_SHARE",
                       disabled_reason=None if files is not None else "Sharing is not available"),
            ActionItem("Glossary progress", lambda: feature.open_progress_for(row) if feature else None,
                       icon="PLAYLIST_ADD_CHECK",
                       disabled_reason=None if book_kind else "Glossary progress is per book"),
            ActionItem("🗑️ Delete glossary files", lambda: self.ctx.spawn(self.delete_files(row)), icon="DELETE_SWEEP",
                       destructive=True, disabled_reason=None if book_kind else "Only book glossaries"),
            ActionItem("↩️ Restore backup", lambda: self.ctx.spawn(self.restore_backup(row)), icon="RESTORE",
                       disabled_reason=None if book_kind else "Only book glossaries"),
        ]
        sheet = ActionSheet(items, title=row.name, subtitle=self._subtitle(row), tablet=self.ctx.tablet)
        self.ctx.show(sheet)
        return sheet

    def _inputs_for(self, row: GlossaryFile) -> list:
        book = self.service.book_for_glossary(row)
        if book is not None:
            return self.service.input_paths_for_books([book])
        base = row.book or os.path.splitext(os.path.basename(row.path))[0]
        return [os.path.join(row.folder, f"{base}.epub")]

    async def delete_files(self, row: GlossaryFile) -> Any:
        feature = self.ctx.feature
        if feature is None:
            return None
        result = await feature.delete_glossary_files(self._inputs_for(row))
        await self.refresh()
        return result

    async def restore_backup(self, row: GlossaryFile) -> Any:
        feature = self.ctx.feature
        if feature is None:
            return None
        result = await feature.restore_glossary_backup(self._inputs_for(row))
        await self.refresh()
        return result

    async def import_glossary(self) -> Optional[str]:
        files = self.ctx.files
        if files is None:
            self.ctx.say("Importing files is not available in this session")
            return None
        try:
            picked = await files.pick_files(target="inbox", allowed_extensions=["csv", "json", "txt", "md"],
                                            allow_multiple=False, dialog_title="Import glossary")
        except Exception as exc:
            self.ctx.say(f"Import failed: {exc}")
            return None
        if not picked:
            return None
        path = picked[0].path
        self.service.record_import(path)
        await self.refresh()
        self.ctx.say(f"Imported {os.path.basename(path)}", "Open",
                     lambda: self.ctx.go("glossary.detail", {"gid": self.service.gid_for(path)}))
        return path

    def open_mode_sheet(self) -> Any:
        feature = self.ctx.feature
        if feature is None:
            return None
        return feature.open_mode_sheet(on_changed=lambda mode: self.render())
