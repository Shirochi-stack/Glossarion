"""Glossary › Editor (UI_SPEC §4.1 Editor, §5.6, §7.3): the desktop Glossary Editor tab on a phone.

Anatomy
  * top bar: ◀ · file name ▾ (switch file: every glossary of the Glossaries list) · ▶ ·
    the auto-reload dot (green = watching, amber = changed on disk while edited);
  * status line (the desktop stats label: "Total entries: N | Characters: … | Terms: …",
    hide-unused / filter counts);
  * toolbar: Search · Filter (column value filters, badge = active columns) · Sort ·
    Undo · Redo · Save (dirty dot) · ⋯;
  * the WindowedList of rows (10k+ entries: a 1,500-row window, 150 rows per scroll
    step): raw (title) → translated, type badge, gender chip, ⚠ tracked-gender conflict,
    "edited" when the translated name differs from the saved one (desktop orange rows);
    tap → EntrySheet, long-press → selection (Delete · Export selection · Change type),
    swipe left → delete with Undo;
  * FAB "＋ Entry".

⋯ menu: Find / Replace · Save As… · Export selection… · Backups · Edit raw · Use as manual
glossary · Update output files on save (``update_html_on_save``) · Hide unused entries ·
Advanced › (Reload · Clean Empty Fields · Remove Duplicates · Backup Settings · Trim Entries ·
Filter Entries · Convert Format · Export Selection · About Format) · Text size · Share file.

The open file is a shared ``glossary_document.GlossaryDocument``: every action is its method
(the desktop action's steps in the desktop order); this pane only asks the desktop questions
first ("Update output files", "Confirm Delete", "No glossary match", "Backup Failed …
Continue anyway?" through ``GlossaryService.ask_continue``) and shows the returned message
boxes. Like the desktop, actions that save also reload (clearing undo), and Undo of a
glossary step writes the restored rows back. After every reload the view is derived again
like the desktop's load: column filters of vanished columns are dropped
(``prune_column_filters``) and Hide unused is re-run (source indices shift). The swipe
delete's snackbar Undo restores the rows the delete removed (the shared undo snapshot taken
before it, saved and re-read) while nothing else changed the glossary since.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import os
from typing import Any, Optional, Sequence

import flet as ft

from glossarion_mobile.services.glossary import (
    CoreMissing,
    JobBusy,
    OpReport,
    RowSpec,
    ViewState,
    doc_count,
    doc_fields,
    is_list_doc,
    is_updated,
    raw_field,
    translated_field,
)
from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.components.action_sheet import ActionItem, ActionSheet
from glossarion_mobile.ui.components.empty_state import EmptyState
from glossarion_mobile.ui.components.info_sheet import InfoSheet
from glossarion_mobile.ui.components.windowed_list import WindowedList
from glossarion_mobile.ui.foreground import poll_sleep
from glossarion_mobile.ui.glossary.common import ask, chip
from glossarion_mobile.ui.glossary.entry_sheet import EntrySheet, GenderSheet, field_label
from glossarion_mobile.ui.library.selection_bar import BulkAction, BulkActionBar, SelectionTopBar
from glossarion_mobile.ui.theme import HIT_TARGET

__all__ = ["EditorPane", "SORTS", "entry_types", "new_entry_form"]

log = logging.getLogger("glossarion.glossary.ui")

#: (id, label, field, descending)
SORTS = (
    ("original", "Original order", None, False),
    ("raw", "Raw name A–Z", "raw_name", False),
    ("translated", "Translated name A–Z", "translated_name", False),
    ("type", "Type", "type", False),
)
POLL_SECONDS = 2.0
SEARCH_DEBOUNCE = 0.25  # seconds of typing pause before the search runs


def _error_text(exc: Exception) -> str:
    return str(exc) or exc.__class__.__name__


def _row_key(source_idx: int) -> str:
    return RowSpec(source_idx, None, {}).key


def entry_types(service: Any, doc: Any) -> list:
    """The EntrySheet's type options: the document's entry types, else the configured ones (list glossaries)."""
    if doc is None or not is_list_doc(doc):
        return []
    try:
        types = list(service.filter_types(doc))
    except Exception:
        types = []
    return types or [t for t in service.custom_entry_types()]


def new_entry_form(doc: Any, fields: Sequence[str], raw_term: str = "") -> tuple:
    """``(fields, values)`` of the "＋ Entry" sheet for ``doc``; ``raw_term`` fills the raw name."""
    shown = [f for f in fields if not f.startswith("_")] or (
        ["type", "raw_name", "translated_name", "gender", "description"] if is_list_doc(doc) else
        ["original", "translated"])
    values: dict = {"type": "character"} if is_list_doc(doc) else {}
    if raw_term:
        values["raw_name" if is_list_doc(doc) else "original"] = raw_term
    return shown, values


class EditorPane:
    def __init__(self, screen: Any) -> None:
        self.screen = screen
        self.ctx = screen.ctx
        self.service = screen.service
        self.doc: Any = None
        self.path: Optional[str] = None
        self.specs: list = []
        self.visible: list = []
        self.state = ViewState()
        self.selected: set = set()
        self.selecting = False
        self.current_key: Optional[str] = None
        self.last_find = ""
        self.last_replace = ""
        self.hide_unused = False
        self.unused_note = ""
        self.loading = False
        self.error: Optional[str] = None
        self.sort_id = "original"
        self.changed_on_disk = False
        self._poll_task: Any = None
        self._search_task: Any = None
        self._saving = False
        self._rows_selecting = False  # the mounted rows are built in the selection layout
        self.generation = 0  # bumped whenever the rows are recomputed (the swipe Undo's staleness check)
        self.root: Optional[ft.Control] = None
        self.sheets: list = []  # opened sheets (tests)
        self.reports: list = []  # OpReports shown (tests)
        self.windowed = WindowedList(build_row=self._row_control, key_of=lambda spec: spec.key, key="ge")
        size = self.service.cfg("glossary_editor_tree_font_size", None)
        try:
            self.text_size = max(8, min(32, int(float(size)))) if size not in (None, "") else 14
        except (TypeError, ValueError):
            self.text_size = 14

    # ---- doc helpers -------------------------------------------------------------------------------

    @property
    def dirty(self) -> bool:
        return bool(getattr(self.doc, "dirty", False)) if self.doc is not None else False

    @property
    def fields(self) -> list:
        return doc_fields(self.doc) if self.doc is not None else []

    # ---- build -------------------------------------------------------------------------------------

    def build(self) -> ft.Control:
        self.prev_button = ft.IconButton(icon=ft.Icons.CHEVRON_LEFT, tooltip="Previous glossary",
                                         size_constraints=HIT_TARGET, on_click=lambda e: self.ctx.spawn(self.nav(-1)),
                                         key="ge-prev")
        self.next_button = ft.IconButton(icon=ft.Icons.CHEVRON_RIGHT, tooltip="Next glossary",
                                         size_constraints=HIT_TARGET, on_click=lambda e: self.ctx.spawn(self.nav(1)),
                                         key="ge-next")
        self.file_button = ft.TextButton(content=self._file_label(), icon=ft.Icons.ARROW_DROP_DOWN, expand=True,
                                         on_click=lambda e: self.open_file_picker(), key="ge-file")
        self.reload_dot = ft.Container(width=10, height=10, border_radius=5, bgcolor=ft.Colors.GREEN_400,
                                       tooltip="Auto-reload on change", key="ge-reload-dot")
        self.stats = ft.Text("Loading glossary…", theme_style=ft.TextThemeStyle.BODY_SMALL, italic=True,
                             color=ft.Colors.ON_SURFACE_VARIANT, max_lines=3, key="ge-stats")
        self.search = ft.TextField(hint_text="Search entries", prefix_icon=ft.Icons.SEARCH, dense=True, expand=True,
                                   on_change=lambda e: self._on_search(), key="ge-search")
        self.filter_button = ft.IconButton(icon=ft.Icons.FILTER_LIST, tooltip="Filter columns",
                                           size_constraints=HIT_TARGET, on_click=lambda e: self.open_filter(),
                                           key="ge-filter")
        self.sort_menu = ft.PopupMenuButton(icon=ft.Icons.SORT, tooltip="Sort", key="ge-sort", items=[
            ft.PopupMenuItem(content=label, on_click=lambda e, s=sid: self.set_sort(s)) for sid, label, _f, _d in SORTS])
        self.undo_button = ft.IconButton(icon=ft.Icons.UNDO, tooltip="Nothing to undo", disabled=True,
                                         size_constraints=HIT_TARGET, on_click=lambda e: self.ctx.spawn(self.undo()),
                                         key="ge-undo")
        self.redo_button = ft.IconButton(icon=ft.Icons.REDO, tooltip="Nothing to redo", disabled=True,
                                         size_constraints=HIT_TARGET,
                                         on_click=lambda e: self.ctx.spawn(self.undo(redo=True)), key="ge-redo")
        self.save_button = ft.IconButton(icon=ft.Icons.SAVE_OUTLINED, tooltip="Save", size_constraints=HIT_TARGET,
                                         on_click=lambda e: self.ctx.spawn(self.save()), key="ge-save")
        self.menu = ft.PopupMenuButton(icon=ft.Icons.MORE_VERT, tooltip="Glossary editor", key="ge-menu",
                                       items=self._menu_items())
        self.filter_row = ft.Row(spacing=6, wrap=True, visible=False, key="ge-filter-chips")
        self.changed_banner = ft.Container(
            content=ft.Row([ft.Icon(ft.Icons.SYNC_PROBLEM, size=18),
                            ft.Text("The file changed on disk.", expand=True),
                            ft.TextButton(content="Reload", on_click=lambda e: self.ctx.spawn(self.reload(force=True)))],
                           spacing=6),
            bgcolor=ft.Colors.TERTIARY_CONTAINER, border_radius=tokens.RADII["card"], padding=6, visible=False,
            key="ge-changed")
        self.selection_bar = SelectionTopBar(on_close=self.exit_selection, on_select_all=self.select_all,
                                             key="ge-selection")
        self.bulk_bar = BulkActionBar(page=self.ctx.page, tablet=self.ctx.tablet,
                                      compact=self.ctx.text_scale >= tokens.COMPACT_TEXT_SCALE, key="ge-bulk")
        self.loading_bar = ft.ProgressBar(visible=True, key="ge-loading")
        self.list_holder = ft.Container(content=self.windowed.control, expand=True, key="ge-list-holder")
        self.fab = ft.FloatingActionButton(icon=ft.Icons.ADD, content="Entry", on_click=lambda e: self.open_new_entry(),
                                           key="ge-fab")
        self.fab_slot = ft.Container(content=self.fab, right=16, bottom=16, key="ge-fab-slot")
        self.root = ft.Column([
            ft.Container(content=ft.Column([
                ft.Row([self.prev_button, self.file_button, self.next_button, self.reload_dot], spacing=0,
                       vertical_alignment=ft.CrossAxisAlignment.CENTER),
                self.stats,
                self.changed_banner,
                ft.Row([self.search, self.filter_button, self.sort_menu], spacing=0,
                       vertical_alignment=ft.CrossAxisAlignment.CENTER),
                ft.Row([self.undo_button, self.redo_button, ft.Container(expand=True), self.save_button, self.menu],
                       spacing=0),
                self.filter_row,
                self.selection_bar.control,
            ], spacing=4, tight=True), padding=ft.Padding.only(left=8, right=8, top=4)),
            self.loading_bar,
            ft.Stack([self.list_holder, self.fab_slot], expand=True),
            self.bulk_bar.control,
        ], spacing=4, expand=True, key="glossary-editor")
        self._sync_nav()
        return self.root

    def _file_label(self) -> str:
        if self.path:
            return f"{os.path.basename(self.path)}{' •' if self.dirty else ''}"
        return "No glossary loaded"

    def _menu_items(self) -> list:
        update_on_save = bool(self.service.cfg("update_html_on_save", True))

        def item(label: str, handler: Any, icon: Any = None, checked: Optional[bool] = None) -> ft.PopupMenuItem:
            return ft.PopupMenuItem(content=label, icon=icon, checked=checked, on_click=lambda e: handler())

        return [
            item("Find / Replace", self.open_find_replace, ft.Icons.FIND_REPLACE),
            item("Save As…", self.open_save_as, ft.Icons.SAVE_AS),
            item("Export selection…", self.open_export_selection, ft.Icons.IOS_SHARE),
            item("Backups", lambda: self.ctx.spawn(self.open_backups()), ft.Icons.HISTORY),
            item("Edit raw", lambda: self.ctx.spawn(self.open_raw()), ft.Icons.CODE),
            item("Use as manual glossary", lambda: self.ctx.spawn(self.use_as_manual()), ft.Icons.FILE_OPEN),
            item("Update output files on save", self.toggle_update_on_save, None, update_on_save),
            item("Hide unused entries", lambda: self.ctx.spawn(self.toggle_hide_unused()), None, self.hide_unused),
            item("Advanced ›", self.open_advanced, ft.Icons.BUILD),
            item("Text size", self.open_text_size, ft.Icons.FORMAT_SIZE),
            item("Share file", lambda: self.ctx.spawn(self.share_file()), ft.Icons.SHARE),
        ]

    def _refresh_menu(self) -> None:
        self.menu.items = self._menu_items()

    # ---- loading -----------------------------------------------------------------------------------

    async def confirm_switch(self, path: str) -> bool:
        """The "Unsaved changes" question before another file replaces the open one (True: go ahead)."""
        if self.doc is not None and self.dirty and self.path and os.path.normcase(path) != os.path.normcase(self.path):
            return await ask(self.ctx, title="Unsaved changes",
                             body=f"Discard the unsaved changes to {os.path.basename(self.path)}?",
                             confirm="Discard", cancel="Cancel", destructive=True)
        return True

    async def open(self, path: str, *, source_path: Optional[str] = None, confirmed: bool = False) -> Any:
        """Open ``path`` (``confirmed``: the caller already asked :meth:`confirm_switch`).

        Like the desktop, "Hide unused entries" stays on across file loads (``_after_reload`` re-runs it,
        as the desktop re-runs ``_apply_hide_unused_entries_filter`` after every load); the search box's
        text keeps filtering the new file. Column filters, the selection and the sort start over."""
        if not confirmed and not await self.confirm_switch(path):
            return self.doc
        self.path = path
        self.doc = None
        query = (self.search.value or "") if self.root is not None else self.state.query
        self.state = ViewState(query=query)
        self.sort_id = "original"
        self.selected = set()
        self.selecting = False
        return await self.reload(source_path=source_path)

    async def reload(self, *, force: bool = False, source_path: Optional[str] = None, silent: bool = False) -> Any:
        """Read the file (desktop Reload / force refresh / the auto-reload)."""
        if not self.path:
            return None
        self.loading = True
        if not silent and self.root is not None:
            self.loading_bar.visible = True
            self.ctx.push(self.loading_bar)
        source = source_path or (getattr(self.doc, "source_path", None) if self.doc is not None else None) or \
            await self.screen.resolve_source_path()
        doc = self.doc if self.doc is not None and os.path.normcase(getattr(self.doc, "path", "")) == \
            os.path.normcase(self.path) else None
        try:
            if doc is not None:
                doc.source_path = source
                doc = await self.ctx.io(lambda: self.service.reload_document(doc))
            else:
                doc = await self.ctx.io(lambda: self.service.open_document(self.path, source_path=source))
            self.error = None
        except CoreMissing as exc:
            self.error = f"The glossary editor is not available in this build ({exc.name})."
            doc = None
        except Exception as exc:
            log.info("loading %s failed: %s", self.path, exc)
            self.error = _error_text(exc)
            doc = None
        finally:
            self.loading = False
        self.doc = doc
        if doc is not None:
            self.changed_on_disk = False
            if self.root is not None:
                self.reload_dot.bgcolor = ft.Colors.GREEN_400
        if self.root is not None:
            self.loading_bar.visible = False
        await self._after_reload(keep_window=silent)
        return doc

    async def _after_reload(self, *, keep_window: bool = True) -> None:
        """The document was re-read (Reload, the auto-reload, Delete, a tool, Undo of a glossary step,
        a restore): derive the view again like the desktop's load - column filters of columns the file
        no longer has are dropped (``prune_column_filters``) and Hide unused is re-run, because the
        stored used rows are source indices that shifted."""
        if self.doc is not None and self.state.filters:
            self.state.filters = self.service.prune_filters(self.state.filters, self.fields)
        self.state.used_rows = None
        self._recompute()
        self.render(keep_window=keep_window)
        if self.hide_unused and self.doc is not None:
            await self.refresh_hide_unused()

    def _recompute(self) -> None:
        self.generation += 1
        doc = self.doc
        if doc is None:
            self.specs, self.visible = [], []
            return
        self.specs = self.service.row_specs(doc)
        self.visible = self.service.visible(doc, self.state, self.specs)

    # ---- render ------------------------------------------------------------------------------------

    def render(self, *, keep_window: bool = False) -> None:
        if self.root is None:
            return
        doc = self.doc
        self.file_button.content = self._file_label()
        self.stats.value = self._stats_text()
        self.stats.italic = doc is None
        self.changed_banner.visible = self.changed_on_disk
        self._render_filter_chips()
        self._sync_history()
        self._sync_nav()
        self._refresh_menu()
        if self.error and doc is None:
            self.list_holder.content = EmptyState(icon="ERROR_OUTLINE", title="Glossary could not be opened",
                                                  body=self.error, key="ge-error",
                                                  primary=("Edit raw", lambda e: self.ctx.spawn(self.open_raw())))
        elif doc is not None and not self.specs:
            self.list_holder.content = EmptyState(icon="SPELLCHECK", title="No entries",
                                                  body="This glossary has no entries yet.", key="ge-empty",
                                                  primary=("＋ Entry", lambda e: self.open_new_entry()))
        elif doc is not None and not self.visible:
            self.list_holder.content = EmptyState(icon="FILTER_LIST_OFF", title="No entries match",
                                                  body="Change the search or the filters.", key="ge-nomatch",
                                                  primary=("Clear filters", lambda e: self.clear_filters()))
        else:
            self.list_holder.content = self.windowed.control
            self.windowed.set_items(self.visible, keep_window=keep_window, keep_rendered=keep_window)
            self._rows_selecting = self.selecting
        self.fab_slot.visible = doc is not None and not self.selecting
        self._sync_selection_bars()
        self.ctx.push(self.root)

    def _stats_text(self) -> str:
        doc = self.doc
        if doc is None:
            return self.error or ("Loading glossary…" if self.loading else "No glossary loaded")
        base = self.service.stats_text(doc)
        if self.hide_unused and self.unused_note:
            base = self.unused_note
        if self.state.filters:
            base += f" | Column filters: {len(self.visible)}/{len(self.specs)} shown"
        elif self.state.query:
            base += f" | Search: {len(self.visible)}/{len(self.specs)} shown"
        return base

    def _render_filter_chips(self) -> None:
        chips = []
        for name, allowed in sorted(self.state.filters.items()):
            chips.append(ft.Chip(label=ft.Text(f"{field_label(name)}: {len(allowed)}"),
                                 on_delete=lambda e, f=name: self.set_column_filter(f, None),
                                 key=f"ge-chip-{name}"))
        self.filter_row.controls = chips
        self.filter_row.visible = bool(chips)
        self.filter_button.badge = ft.Badge(label=str(len(chips))) if chips else None

    def _sync_history(self) -> None:
        doc = self.doc
        undo = len(getattr(doc, "_undo_stack", []) or []) if doc is not None else 0
        redo = len(getattr(doc, "_redo_stack", []) or []) if doc is not None else 0
        self.undo_button.disabled = undo == 0
        self.redo_button.disabled = redo == 0
        self.undo_button.tooltip = f"Undo ({undo} steps)" if undo else "Nothing to undo"
        self.redo_button.tooltip = f"Redo ({redo} steps)" if redo else "Nothing to redo"
        dirty = self.dirty
        self.save_button.icon = ft.Icons.SAVE if dirty else ft.Icons.SAVE_OUTLINED
        self.save_button.badge = ft.Badge(small_size=8) if dirty else None
        self.save_button.disabled = doc is None

    def _sync_nav(self) -> None:
        files = self.screen.sibling_files()
        show = len(files) > 1
        self.prev_button.visible = show
        self.next_button.visible = show

    # ---- rows --------------------------------------------------------------------------------------

    def _row_control(self, spec: RowSpec, position: int) -> ft.Control:
        doc = self.doc
        entry = spec.entry
        raw = str(entry.get(raw_field(doc), "") or entry.get("original", "") or "") if doc is not None else ""
        translated = str(entry.get(translated_field(doc), "") or "") if doc is not None else ""
        selected = spec.source_idx in self.selected
        updated = doc is not None and is_updated(doc, spec)
        badges: list = []
        entry_type = str(entry.get("type") or "")
        if entry_type:
            badges.append(chip(entry_type, key="type"))
        gender = str(entry.get("gender") or "")
        if gender:
            badges.append(chip(gender, bgcolor=ft.Colors.TERTIARY_CONTAINER, key="gender"))
        status = self.service.gender_status(doc, entry) if doc is not None else None
        if isinstance(status, dict) and status.get("conflict"):
            badges.append(chip(f"⚠ {status.get('label') or 'Gender conflict'}", bgcolor=ft.Colors.ERROR_CONTAINER,
                               key="conflict", tooltip="Tap the row › Resolve gender…"))
        if updated:
            badges.append(chip("edited", bgcolor=ft.Colors.ORANGE_200, key="edited"))
        lines: list = [ft.Text(raw or "—", size=self.text_size, weight=ft.FontWeight.W_600, max_lines=3,
                               overflow=ft.TextOverflow.ELLIPSIS),
                       ft.Text(f"→ {translated}" if translated else "→ (no translation)", size=self.text_size - 1,
                               max_lines=3, overflow=ft.TextOverflow.ELLIPSIS)]
        description = str(entry.get("description") or "")
        if description:
            lines.append(ft.Text(description, size=max(8, self.text_size - 3), color=ft.Colors.ON_SURFACE_VARIANT,
                                 max_lines=2, overflow=ft.TextOverflow.ELLIPSIS))
        if badges:
            lines.append(ft.Row(badges, spacing=4, wrap=True, run_spacing=2))
        leading: list = []
        check = None
        if self.selecting:
            check = ft.Icon(ft.Icons.CHECK_CIRCLE if selected else ft.Icons.RADIO_BUTTON_UNCHECKED,
                            color=ft.Colors.PRIMARY if selected else ft.Colors.OUTLINE, size=22)
            leading.append(check)
        base_bgcolor = ft.Colors.with_opacity(0.10, ft.Colors.ORANGE) if updated else ft.Colors.SURFACE_CONTAINER_LOW
        row = ft.Container(
            content=ft.Row(leading + [
                ft.Column(lines, spacing=1, expand=True, tight=True),
                ft.Text(f"{spec.source_idx + 1}", size=10, color=ft.Colors.ON_SURFACE_VARIANT),
            ], spacing=8, vertical_alignment=ft.CrossAxisAlignment.START),
            padding=ft.Padding.symmetric(horizontal=10, vertical=8),
            border_radius=tokens.RADII["card"],
            bgcolor=ft.Colors.SECONDARY_CONTAINER if selected else base_bgcolor,
            border=ft.Border.all(2, ft.Colors.PRIMARY) if spec.key == self.current_key else None,
            on_click=lambda e, s=spec: self.on_row_tap(s),
            on_long_press=lambda e, s=spec: self.on_row_long_press(s),
            ink=True,
        )
        if self.selecting:
            row.data = {"check": check, "bgcolor": base_bgcolor}  # toggled in place (_mark_selected)
            return row
        return ft.Dismissible(
            content=row,
            dismiss_direction=ft.DismissDirection.END_TO_START,
            background=ft.Container(bgcolor=ft.Colors.ERROR, border_radius=tokens.RADII["card"],
                                    alignment=ft.Alignment.CENTER_RIGHT, padding=ft.Padding.only(right=16),
                                    content=ft.Icon(ft.Icons.DELETE_OUTLINE, color=ft.Colors.ON_ERROR)),
            on_dismiss=lambda e, s=spec: self.ctx.spawn(self.swipe_delete(s)),
        )

    def spec_for_key(self, key: Optional[str]) -> Optional[RowSpec]:
        if key is None:
            return None
        for spec in self.specs:
            if spec.key == key:
                return spec
        return None

    def on_row_tap(self, spec: RowSpec) -> Any:
        if self.selecting:
            self.toggle(spec.source_idx)
            return None
        self.current_key = spec.key
        return self.open_entry(spec)

    def on_row_long_press(self, spec: RowSpec) -> None:
        if not self.selecting:
            self.ctx.haptic("medium_impact")
        self.selecting = True
        self.toggle(spec.source_idx, force=True)

    async def jump_to(self, key: str) -> bool:
        previous = self.current_key
        self.current_key = key
        for k in (previous, key):
            spec = self.spec_for_key(k)
            if spec is not None:
                self.windowed.replace(spec)  # no-op when that row is not mounted
        return await self.windowed.jump_to(key)

    # ---- search / filters / sort --------------------------------------------------------------------

    def _on_search(self) -> None:
        """Search as you type: each keystroke restarts a short wait, then :meth:`apply_search` runs."""
        task = self._search_task
        if task is not None and not task.done():
            task.cancel()

        async def later() -> None:
            try:
                await asyncio.sleep(SEARCH_DEBOUNCE)
            except asyncio.CancelledError:
                return
            await self.apply_search()

        self._search_task = self.ctx.spawn(later())

    async def apply_search(self) -> None:
        """Filter the rows by the search text off the UI loop; a newer keystroke or reload supersedes it."""
        query = (self.search.value or "") if self.root is not None else self.state.query
        doc, specs = self.doc, self.specs
        if doc is None:
            self.state.query = query
            return
        state = dataclasses.replace(self.state, query=query)
        try:
            visible = await self.ctx.io(lambda: self.service.visible(doc, state, specs))
        except Exception as exc:
            log.info("search failed: %s", exc)
            return
        if doc is not self.doc or (self.root is not None and (self.search.value or "") != query):
            return  # another document, or the text changed again (that search applies)
        self.state.query = query
        if specs is not self.specs or self.state != state:
            self._apply_view()  # the rows or the filters changed meanwhile: recompute with the new text
            return
        self.visible = visible
        self.render()

    def _apply_view(self) -> None:
        if self.doc is None:
            return
        self.visible = self.service.visible(self.doc, self.state, self.specs)
        self.render()

    def set_column_filter(self, field_name: str, allowed: Optional[frozenset]) -> None:
        if allowed is None:
            self.state.filters.pop(field_name, None)
        else:
            self.state.filters[field_name] = frozenset(allowed)
        self._apply_view()

    def clear_filters(self) -> None:
        self.state.filters = {}
        self.state.query = ""
        if self.root is not None:
            self.search.value = ""
        self._apply_view()

    def set_sort(self, sort_id: str) -> None:
        for sid, _label, field_name, desc in SORTS:
            if sid == sort_id:
                self.sort_id = sid
                self.state.sort_field = field_name
                self.state.sort_desc = desc
        fields = self.fields
        if self.state.sort_field and fields and self.state.sort_field not in fields:
            alias = {"raw_name": "original", "translated_name": "translated"}.get(self.state.sort_field)
            if alias in fields:
                self.state.sort_field = alias
        self._apply_view()

    def open_filter(self, field_name: Optional[str] = None) -> Any:
        if self.doc is None:
            return None
        from glossarion_mobile.ui.glossary.tools import ColumnFilterSheet

        fields = self.fields
        sheet = ColumnFilterSheet(self.ctx, fields=fields,
                                  values_for=lambda f: self.service.column_values(self.doc, f, self.specs),
                                  active=self.state.filters, on_apply=self.set_column_filter,
                                  initial_field=field_name or ("type" if "type" in fields else None))
        self.sheets.append(sheet)
        return sheet.show()

    # ---- selection ---------------------------------------------------------------------------------

    def toggle(self, source_idx: int, force: Optional[bool] = None) -> None:
        """Select / deselect one row. Entering or leaving selection mode rebuilds the mounted rows (the
        swipe-to-delete wrapper comes and goes); inside it only that row changes, in place (§7.3)."""
        on = (source_idx not in self.selected) if force is None else force
        if on:
            self.selected.add(source_idx)
        else:
            self.selected.discard(source_idx)
        if not self.selected:
            self.selecting = False
        if self.root is None or not (self.selecting and self._rows_selecting):
            self.render(keep_window=True)
            return
        self._mark_selected(source_idx)
        self._sync_selection_bars()
        self.ctx.push(self.selection_bar.control, self.bulk_bar.control)

    def _mark_selected(self, source_idx: int) -> None:
        """Show one mounted row's selection state (its check icon and background, mutated in place)."""
        key = _row_key(source_idx)
        control = self.windowed.controls.get(key)
        if control is None:
            return  # not mounted: built with its state when it scrolls in
        selected = source_idx in self.selected
        parts = control.data if isinstance(getattr(control, "data", None), dict) else None
        if parts is None or parts.get("check") is None:
            position = self.windowed.positions.get(key)
            if position is not None and self.windowed.replace(self.windowed.items[position]):
                self.ctx.push(self.windowed.list_view)
            return
        check = parts["check"]
        check.icon = ft.Icons.CHECK_CIRCLE if selected else ft.Icons.RADIO_BUTTON_UNCHECKED
        check.color = ft.Colors.PRIMARY if selected else ft.Colors.OUTLINE
        control.bgcolor = ft.Colors.SECONDARY_CONTAINER if selected else parts["bgcolor"]
        self.ctx.push(control)

    def select_all(self) -> None:
        self.selecting = True
        self.selected = {s.source_idx for s in self.visible}
        self.render(keep_window=True)

    def exit_selection(self) -> None:
        self.selecting = False
        self.selected = set()
        self.render(keep_window=True)

    def selected_specs(self) -> list:
        return [s for s in self.specs if s.source_idx in self.selected]

    def _sync_selection_bars(self) -> None:
        count = len(self.selected)
        active = self.selecting and count > 0
        self.selection_bar.set_count(count)
        self.selection_bar.show(active)
        self.bulk_bar.show(active)
        if active:
            typed = self.doc is not None and is_list_doc(self.doc) and "type" in self.fields
            self.bulk_bar.set_actions([
                BulkAction("delete", f"Delete {count}", "DELETE_OUTLINE",
                           lambda: self.ctx.spawn(self.delete_selected()), destructive=True),
                BulkAction("export", "Export selection", "IOS_SHARE", self.open_export_selection),
                BulkAction("type", "Change type", "CATEGORY", self.open_change_type,
                           None if typed else "Only list glossaries have entry types"),
            ])

    # ---- entry edits --------------------------------------------------------------------------------

    def _types(self) -> list:
        return entry_types(self.service, self.doc)

    def open_entry(self, spec: RowSpec) -> Optional[EntrySheet]:
        doc = self.doc
        if doc is None:
            return None
        status = self.service.gender_status(doc, spec.entry)
        conflict = (f"Tracked gender conflict: {status.get('label')}" if isinstance(status, dict)
                    and status.get("conflict") else None)
        sheet = EntrySheet(self.ctx, fields=self.fields, values=spec.entry, types=self._types(),
                           conflict=conflict, text_size=self.text_size,
                           on_save=lambda values, s=spec: self.save_entry(s, values),
                           on_delete=lambda s=spec: self.ctx.spawn(self.delete_rows([s], confirm=True)),
                           on_resolve_gender=lambda s=spec: self.open_gender(s))
        self.sheets.append(sheet)
        return sheet.show()

    def save_entry(self, spec: RowSpec, values: dict) -> Any:
        if self.doc is None or not values:
            return None
        new_ref = self.service.update_entry(self.doc, spec.ref, values)
        self._after_edit(spec)
        return new_ref

    def _after_edit(self, spec: Optional[RowSpec] = None) -> None:
        """Re-render after an in-memory edit: one row in place when the row set is unchanged."""
        if self.doc is None:
            return
        old_keys = [s.key for s in self.visible]
        self._recompute()
        if spec is not None and [s.key for s in self.visible] == old_keys and self.root is not None:
            for candidate in self.visible:
                if candidate.key == spec.key:
                    self.windowed.replace(candidate)
            self.stats.value = self._stats_text()
            self.file_button.content = self._file_label()
            self._sync_history()
            self.ctx.push(self.root)
            return
        self.render(keep_window=True)

    def open_new_entry(self, raw_term: str = "") -> Optional[EntrySheet]:
        """＋ Entry; ``raw_term`` pre-fills the raw name (Reader / chat "Add to glossary")."""
        doc = self.doc
        if doc is None:
            return None
        fields, values = new_entry_form(doc, self.fields, raw_term)
        sheet = EntrySheet(self.ctx, fields=fields, values=values,
                           types=self._types(), new=True, text_size=self.text_size, on_save=self.add_entry)
        self.sheets.append(sheet)
        return sheet.show()

    def add_entry(self, values: dict) -> Any:
        if self.doc is None:
            return None
        try:
            ref = self.service.add_entry(self.doc, values)
        except ValueError as exc:
            self.ctx.say(str(exc))
            return None
        self._recompute()
        self.render(keep_window=True)
        key = next((s.key for s in self.specs if s.ref == ref), None)
        if key is not None:
            self.ctx.spawn(self.jump_to(key))
        self.ctx.say("Entry added — Save to keep it")
        return ref

    def open_gender(self, spec: RowSpec) -> Optional[GenderSheet]:
        if self.doc is None:
            return None
        try:
            model = self.service.gender_model(self.doc, spec.ref)
        except CoreMissing as exc:
            self.ctx.say(f"Not available in this build ({exc.name})")
            return None
        if not model:
            self.ctx.say("This entry has no tracked gender conflict")
            return None
        sheet = GenderSheet(self.ctx, model, on_apply=lambda decision, s=spec: self.apply_gender(s, decision))
        self.sheets.append(sheet)
        return sheet.show()

    def apply_gender(self, spec: RowSpec, decision: str) -> bool:
        if self.doc is None:
            return False
        ok = self.service.resolve_gender(self.doc, spec.ref, decision)
        if ok:
            self._after_edit(spec)
        return ok

    def open_change_type(self) -> Optional[ActionSheet]:
        specs = self.selected_specs()
        if not specs or self.doc is None:
            return None
        items = [ActionItem(t, lambda t=t: self.change_type(specs, t), icon="CATEGORY") for t in self._types()]
        sheet = ActionSheet(items, title=f"Change type of {len(specs)} entr{'ies' if len(specs) != 1 else 'y'}",
                            tablet=self.ctx.tablet)
        self.ctx.show(sheet)
        self.sheets.append(sheet)
        return sheet

    def change_type(self, specs: Sequence[RowSpec], entry_type: str) -> int:
        if self.doc is None:
            return 0
        count = self.service.change_type(self.doc, [s.ref for s in specs], entry_type)
        self.selecting = False
        self.selected = set()
        self._recompute()
        self.render(keep_window=True)
        self.ctx.say(f"Changed the type of {count} entries — Save to keep it")
        return count

    # ---- find / replace -------------------------------------------------------------------------------

    def open_find_replace(self) -> Any:
        if self.doc is None:
            return None
        from glossarion_mobile.ui.glossary.find_replace import FindReplaceSheet

        sheet = FindReplaceSheet(self)
        self.sheets.append(sheet)
        return sheet.show()

    def view_rows(self) -> list:
        """The rows the desktop view holds for Replace All: all, or the used ones while Hide unused is on."""
        if self.state.used_rows is None:
            return list(self.specs)
        return [s for s in self.specs if s.source_idx in self.state.used_rows]

    def replace_in_row(self, spec: RowSpec, find: str, repl: str) -> int:
        if self.doc is None:
            return 0
        row = self.service.editor_rows(self.doc, [spec])[0]
        count = self.service.replace_in(self.doc, row, find, repl)
        if count:
            self._after_edit(spec)
        return count

    async def replace_all(self, find: str, repl: str) -> int:
        """Replace All over the rows the view holds, on the io pool (10k rows), then one re-render."""
        doc = self.doc
        if doc is None:
            return 0
        specs = self.view_rows()

        def run() -> int:
            return self.service.replace_all(doc, find, repl, self.service.editor_rows(doc, specs))

        try:
            total = await self.ctx.io(run)
        except Exception as exc:
            self.ctx.say(_error_text(exc))
            return 0
        if total and doc is self.doc:
            self._recompute()
            self.render(keep_window=True)
        return total

    async def replace_in_outputs(self, old: str, new: str) -> tuple:
        if self.doc is None:
            return 0, 0
        try:
            result = await self.ctx.io(lambda: self.service.replace_in_outputs(self.doc, old, new))
        except Exception as exc:
            self.ctx.say(_error_text(exc))
            return 0, 0
        self._sync_history()
        self.ctx.push(self.root)
        return result

    # ---- undo / redo / save ---------------------------------------------------------------------------

    async def undo(self, *, redo: bool = False) -> Optional[str]:
        if self.doc is None:
            return None
        try:
            kind, result = await self.ctx.io(lambda: self.service.undo(self.doc, redo=redo))
        except Exception as exc:
            self.ctx.say(_error_text(exc))
            return None
        if kind == "html":
            if result:
                files_updated, total = result
                verb = "↷ Redo: re-applied" if redo else "↶ Undo: reverted"
                self.ctx.say(f"{verb} {total} replacement(s) across {files_updated} output file(s).")
            else:
                self.ctx.say(f"⚠️ {'Redo' if redo else 'Undo'} (output files) failed")
            self._sync_history()
            if self.root is not None:
                self.ctx.push(self.root)
        elif kind == "glossary":
            if result:
                self.ctx.say(str(result))
            await self._after_reload()  # restored, saved and re-read
        return kind

    async def save(self) -> Optional[dict]:
        """``save_edited_glossary``: the "Update output files" question, then ``GlossaryDocument.save_edits``
        (backup "before_save", the shared save, the output-file update, the new baseline)."""
        doc = self.doc
        if doc is None or self._saving:
            return None
        self._saving = True
        try:
            changes = self.service.translated_changes(doc)
            update = bool(self.service.cfg("update_html_on_save", True))
            if update and changes:
                title, text = self.service.update_prompt(changes)
                if not await ask(self.ctx, title=title, body=text, confirm="Yes", cancel="No"):
                    return None
            try:
                report = await self.ctx.io(lambda: self.service.save_edits(doc, update_outputs=update))
            except CoreMissing as exc:
                self.ctx.say(f"Saving needs {exc.name} (not in this build)")
                return None
            except Exception as exc:
                log.info("saving %s failed: %s", doc.path, exc)
                self.ctx.say(_error_text(exc))
                return None
            self.reports.append(report)
            if report.get("saved"):
                files = int(report.get("files_updated") or 0)
                self.ctx.say("Glossary saved successfully" + (f" · {files} output file(s) updated" if files else ""))
                if self.hide_unused:
                    await self.refresh_hide_unused()
            self._recompute()
            self.render(keep_window=True)
            return report
        finally:
            self._saving = False

    async def swipe_delete(self, spec: RowSpec) -> Optional[OpReport]:
        """Swipe left: the delete without the question, then a snackbar whose Undo restores the row."""
        report = await self.delete_rows([spec], confirm=False, keep_snapshot=True)
        if report is not None and report.ok:
            details = report.details if isinstance(report.details, dict) else {}
            snapshot = details.get("snapshot")
            if snapshot is not None:
                doc, generation = self.doc, self.generation
                self.ctx.say(report.message, "Undo",
                             lambda: self.ctx.spawn(self.undo_swipe_delete(doc, generation, snapshot)))
            else:
                self.ctx.say(report.message)
        return report

    async def undo_swipe_delete(self, doc: Any, generation: int, snapshot: Any) -> bool:
        """The swipe delete's Undo: the rows before the delete written back (``restore_snapshot``) - only
        while nothing else changed the glossary since (no edit, save, reload or other delete)."""
        if doc is None or doc is not self.doc or generation != self.generation or self.dirty:
            self.ctx.say("The glossary changed since the delete — restore it from Backups instead")
            return False
        try:
            ok = await self.ctx.io(lambda: self.service.restore_snapshot(doc, snapshot))
        except Exception as exc:
            log.info("undoing the swipe delete failed: %s", exc)
            self.ctx.say(_error_text(exc))
            ok = False
        await self._after_reload()
        if ok:
            self.ctx.say("Entry restored")
        return ok

    async def delete_selected(self) -> Optional[OpReport]:
        return await self.delete_rows(self.selected_specs(), confirm=True)

    async def delete_rows(self, specs: Sequence[RowSpec], *, confirm: bool = True,
                          keep_snapshot: bool = False) -> Optional[OpReport]:
        """Delete Selected: "Confirm Delete" → ``GlossaryDocument.delete`` (backup, undo snapshot, save, reload)."""
        doc = self.doc
        specs = list(specs)
        if doc is None or not specs:
            self.ctx.say("Please select entries to delete")
            return None
        if confirm and not await ask(self.ctx, title="Confirm Delete", body=f"Delete {len(specs)} selected entries?",
                                     confirm="Yes", cancel="No", destructive=True):
            return None
        try:
            report = await self.ctx.io(lambda: self.service.delete(doc, [s.ref for s in specs],
                                                                   keep_snapshot=keep_snapshot))
        except Exception as exc:
            self.ctx.say(_error_text(exc))
            return None
        self.selecting = False
        self.selected = set()
        if report is not None and report.changed:
            await self._after_reload()
        else:
            self._recompute()
            self.render(keep_window=True)
        if report is not None:
            self.reports.append(report)
            if confirm:
                self.ctx.say(report.message)
        return report

    # ---- tools -------------------------------------------------------------------------------------

    def open_advanced(self) -> ActionSheet:
        reason = None if self.doc is not None else "No glossary loaded"
        list_reason = reason or (None if is_list_doc(self.doc) else "Only list glossaries")
        items = [
            ActionItem("Reload", lambda: self.ctx.spawn(self.reload(force=True)), icon="REFRESH"),
            ActionItem("Clean Empty Fields", lambda: self.ctx.spawn(self.run_tool("clean")),
                       icon="CLEANING_SERVICES", disabled_reason=list_reason),
            ActionItem("Remove Duplicates", lambda: self.ctx.spawn(self.run_tool("dedupe")), icon="CONTENT_COPY",
                       disabled_reason=list_reason),
            ActionItem("Backup Settings", self.open_backup_settings, icon="SETTINGS_BACKUP_RESTORE"),
            ActionItem("Trim Entries", self.open_trim, icon="CONTENT_CUT", disabled_reason=reason),
            ActionItem("Filter Entries", self.open_filter_entries, icon="FILTER_ALT", disabled_reason=reason),
            ActionItem("Convert Format", self.open_convert, icon="TRANSFORM", disabled_reason=reason),
            ActionItem("Export Selection", self.open_export_selection, icon="IOS_SHARE", disabled_reason=reason),
            ActionItem("About Format", self.open_about_format, icon="INFO_OUTLINE"),
        ]
        sheet = ActionSheet(items, title="Advanced editing", tablet=self.ctx.tablet)
        self.ctx.show(sheet)
        self.sheets.append(sheet)
        return sheet

    async def run_tool(self, tool: str, *args: Any) -> Optional[OpReport]:
        """One shared editor tool on the io pool; shows its desktop message box."""
        doc = self.doc
        if doc is None:
            self.ctx.say("No glossary loaded")
            return None
        calls = {
            "clean": lambda: self.service.clean_empty_fields(doc),
            "dedupe": lambda: self.service.remove_duplicates(doc),
            "trim": lambda: self.service.trim(doc, args[0]),
            "filter": lambda: self.service.apply_filter(doc, **args[0]),
            "convert": lambda: self.service.convert(doc, args[0]),
        }
        try:
            report = await self.ctx.io(calls[tool])
        except JobBusy as exc:
            self.ctx.say(str(exc))
            return None
        except CoreMissing as exc:
            self.ctx.say(f"Not available in this build ({exc.name})")
            return None
        except Exception as exc:
            log.info("glossary tool %s failed: %s", tool, exc)
            self.ctx.say(_error_text(exc))
            return None
        if report is not None:
            self.reports.append(report)
            self._show_report(report)
        details = report.details if report is not None and isinstance(report.details, dict) else {}
        if report is not None and report.changed and details.get("reload", True):
            await self._after_reload()  # the tool saved and re-read the file
        else:
            self._recompute()
            self.render(keep_window=True)
        return report

    def _show_report(self, report: OpReport) -> None:
        if not report.message:
            return
        if "\n" in report.message and self.ctx.page is not None:
            info = InfoSheet(title=report.title or "Glossary", body=report.message)
            info.show(self.ctx.page)
            self.sheets.append(info)
        else:
            self.ctx.say(report.message)

    def open_trim(self) -> Any:
        if self.doc is None:
            return None
        from glossarion_mobile.ui.glossary.tools import TrimSheet

        sheet = TrimSheet(self.ctx, total=doc_count(self.doc), type_summary=self.service.type_summary(self.doc),
                          preview=lambda n: self.service.trim_preview(self.doc, n),
                          on_apply=lambda n: self.ctx.spawn(self.run_tool("trim", n)))
        self.sheets.append(sheet)
        return sheet.show()

    def open_filter_entries(self) -> Any:
        if self.doc is None:
            return None
        from glossarion_mobile.ui.glossary.tools import FilterEntriesSheet

        types = self.service.filter_types(self.doc)
        sheet = FilterEntriesSheet(self.ctx, total=doc_count(self.doc), types=types, typed=bool(types),
                                   on_preview=self.preview_filter_entries,
                                   on_apply=lambda choices: self.ctx.spawn(self.run_tool("filter", choices)))
        self.sheets.append(sheet)
        return sheet.show()

    async def preview_filter_entries(self, choices: dict) -> Optional[OpReport]:
        if self.doc is None:
            return None
        try:
            matching, removed = await self.ctx.io(lambda: self.service.preview_filter(self.doc, **choices))
        except Exception as exc:
            self.ctx.say(_error_text(exc))
            return None
        return OpReport(True, f"Filter matches: {matching} entries ({removed} will be removed)", "Preview",
                        count=matching, details=removed)

    def open_convert(self) -> Any:
        if self.doc is None:
            return None
        from glossarion_mobile.ui.glossary.tools import ConvertSheet

        sheet = ConvertSheet(self.ctx, default_path=self.service.default_convert_path(self.doc),
                             legacy=bool(self.service.cfg("glossary_use_legacy_csv", False)),
                             on_convert=lambda dest: self.ctx.spawn(self.convert(dest)))
        self.sheets.append(sheet)
        return sheet.show()

    async def convert(self, dest: str) -> Optional[OpReport]:
        report = await self.run_tool("convert", dest)
        if report is not None and report.ok and self.ctx.files is not None and os.path.isfile(dest) and \
                os.path.normcase(dest) != os.path.normcase(self.path or ""):
            self.ctx.say(report.message.splitlines()[0], "Share", lambda: self.ctx.spawn(self.ctx.files.share([dest])))
        return report

    def open_about_format(self) -> InfoSheet:
        from glossarion_mobile.ui.glossary.tools import about_format

        title, text = about_format(self.service)
        info = InfoSheet(title=title, body=text)
        if self.ctx.page is not None:
            info.show(self.ctx.page)
        self.sheets.append(info)
        return info

    def open_backup_settings(self) -> Any:
        from glossarion_mobile.ui.glossary.tools import BackupSettingsSheet

        enabled, max_backups = self.service.backup_settings()
        sheet = BackupSettingsSheet(self.ctx, enabled=enabled, max_backups=max_backups, glossary_path=self.path or "",
                                    on_save=lambda on, n: self.ctx.say(self.service.set_backup_settings(on, n)),
                                    on_backup_now=lambda: self.ctx.spawn(self.backup_now()))
        self.sheets.append(sheet)
        return sheet.show()

    async def backup_now(self) -> bool:
        if self.doc is None or not getattr(self.doc, "current_glossary_data", None):
            self.ctx.say("No glossary loaded")
            return False
        try:
            ok = await self.ctx.io(lambda: self.service.manual_backup(self.doc))
        except Exception as exc:
            self.ctx.say(_error_text(exc))
            return False
        if ok:
            self.ctx.say("Manual backup created successfully!")
        return bool(ok)

    async def open_backups(self) -> Any:
        if not self.path or self.doc is None:
            self.ctx.say("No glossary file is loaded.")
            return None
        from glossarion_mobile.ui.glossary.backups import BackupsSheet

        error = None
        try:
            rows = await self.ctx.io(lambda: self.service.list_backups(self.doc))
        except Exception as exc:
            rows, error = [], f"Could not list the backups: {exc}"
        files = self.ctx.files
        sheet = BackupsSheet(self.ctx, glossary_path=self.path, backups=rows, error=error,
                             on_restore=lambda p: self.ctx.spawn(self.restore_backup(p)),
                             on_share=(lambda p: self.ctx.spawn(files.share([p]))) if files is not None else None,
                             on_settings=self.open_backup_settings,
                             on_backup_now=lambda: self.ctx.spawn(self.backup_now()))
        self.sheets.append(sheet)
        return sheet.show()

    async def restore_backup(self, backup_path: str) -> Optional[OpReport]:
        if self.doc is None:
            return None
        if not await ask(self.ctx, title="Restore backup",
                         body=f"Replace this glossary's entries with {os.path.basename(backup_path)}?\n\n"
                              "A backup of the current entries is made first.", confirm="Restore", cancel="Cancel"):
            return None
        try:
            report = await self.ctx.io(lambda: self.service.restore_backup(self.doc, backup_path))
        except Exception as exc:
            self.ctx.say(_error_text(exc))
            return None
        if report is not None and report.changed:
            await self._after_reload(keep_window=False)
        else:
            self._recompute()
            self.render()
        if report is not None:
            self.ctx.say(report.message)
        return report

    # ---- files: save as / export / raw / manual / share ------------------------------------------------

    def open_save_as(self) -> Any:
        if self.doc is None:
            return None
        from glossarion_mobile.ui.glossary.tools import NameSheet

        stem, ext = os.path.splitext(os.path.basename(self.path or "glossary.json"))
        sheet = NameSheet(self.ctx, title="Save Glossary As", folder=os.path.dirname(self.path or ""),
                          default_name=f"{stem}_copy{ext if ext in ('.json', '.csv') else '.json'}", confirm="Save",
                          note="The copy is written next to this glossary and opened in the editor.",
                          on_confirm=lambda dest: self.ctx.spawn(self.save_as(dest)))
        self.sheets.append(sheet)
        return sheet.show()

    async def save_as(self, dest: str) -> Optional[OpReport]:
        if self.doc is None:
            return None
        if os.path.exists(dest) and not await ask(self.ctx, title="Replace file?",
                                                  body=f"{os.path.basename(dest)} already exists. Replace it?",
                                                  confirm="Replace", cancel="Cancel", destructive=True):
            return None
        try:
            report = await self.ctx.io(lambda: self.service.save_as(self.doc, dest))
        except Exception as exc:
            self.ctx.say(_error_text(exc))
            return None
        self.ctx.say(report.message)
        self.path = self.doc.path
        self.screen.note_switched(self.path)
        self.render(keep_window=True)
        return report

    def open_export_selection(self) -> Any:
        if self.doc is None:
            return None
        if not self.selected:
            self.ctx.say("No entries selected")
            return None
        from glossarion_mobile.ui.glossary.tools import NameSheet

        stem = os.path.splitext(os.path.basename(self.path or "glossary"))[0]
        sheet = NameSheet(self.ctx, title="Export Selected Entries", folder=self.screen.export_dir(),
                          default_name=f"{stem}_selection.json", confirm="Export",
                          note="Then share it or save it where you like.",
                          on_confirm=lambda dest: self.ctx.spawn(self.export_selection(dest)))
        self.sheets.append(sheet)
        return sheet.show()

    async def export_selection(self, dest: str) -> Optional[OpReport]:
        if self.doc is None:
            return None
        refs = [s.ref for s in self.selected_specs()]
        try:
            os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
            report = await self.ctx.io(lambda: self.service.export_selection(self.doc, dest, refs))
        except Exception as exc:
            self.ctx.say(_error_text(exc))
            return None
        self.ctx.say(report.message)
        if self.ctx.files is not None and os.path.isfile(dest):
            await self.ctx.files.share([dest])
        return report

    async def share_file(self) -> bool:
        if not self.path or self.ctx.files is None:
            return False
        return bool(await self.ctx.files.share([self.path]))

    async def open_raw(self) -> Any:
        if not self.path:
            return None
        from glossarion_mobile.ui.glossary.raw_editor import RawGlossaryEditor

        editor = await RawGlossaryEditor.open(self.ctx, self.path,
                                              on_saved=lambda p: self.ctx.spawn(self.reload(force=True)))
        self.sheets.append(editor)
        return editor

    async def use_as_manual(self) -> Any:
        if not self.path or self.ctx.feature is None:
            return None
        return await self.ctx.feature.use_as_manual(self.path, source_path=getattr(self.doc, "source_path", None))

    def toggle_update_on_save(self) -> bool:
        value = not bool(self.service.cfg("update_html_on_save", True))
        self.service.set_cfg("update_html_on_save", value)
        self._refresh_menu()
        self.ctx.push(self.menu)
        self.ctx.say(("✅" if value else "❌") + " Update output files on save")
        return value

    def open_text_size(self) -> Any:
        from glossarion_mobile.ui.glossary.tools import TextSizeSheet

        sheet = TextSizeSheet(self.ctx, size=self.text_size, on_change=self.set_text_size)
        self.sheets.append(sheet)
        return sheet.show()

    def set_text_size(self, size: int) -> None:
        self.text_size = max(8, min(32, int(size)))
        self.service.set_cfg("glossary_editor_tree_font_size", self.text_size)
        self.render(keep_window=True)

    # ---- hide unused ---------------------------------------------------------------------------------

    async def toggle_hide_unused(self) -> bool:
        self.hide_unused = not self.hide_unused
        if not self.hide_unused:
            self.state.used_rows = None
            self.unused_note = ""
            self._apply_view()
            return False
        await self.refresh_hide_unused()
        return True

    async def refresh_hide_unused(self) -> Optional[dict]:
        """``_apply_hide_unused_entries_filter``: scan the translated output, show the used rows only."""
        doc = self.doc
        if doc is None:
            return None
        if self.root is not None:
            self.stats.value = "Scanning translated output for glossary usage..."
            self.ctx.push(self.stats)

        def progress(payload: dict) -> None:
            stage = payload.get("stage")
            if stage == "reading":
                text = f"Reading translated output files: {payload.get('current', 0)}/{payload.get('total_files', 0)}"
            elif stage == "matching":
                text = (f"Matching glossary entries: {payload.get('checked', 0)}/{payload.get('total', 0)} checked, "
                        f"{payload.get('used', 0)} used")
            elif stage == "starting":
                text = (f"Matching glossary entries: 0/{payload.get('total', 0)} checked, 0 used "
                        f"({payload.get('total_files', 0)} files)")
            else:
                return
            self.ctx.post_ui(self._progress_text, text)

        try:
            result = await self.ctx.io(lambda: self.service.used_rows(doc, progress))
        except Exception as exc:
            result = {"ok": False, "error": str(exc)}
        total = int(result.get("total") or len(self.specs))
        self.state.used_rows = None
        if not result.get("ok", True):
            self.unused_note = f"Hide unused failed: {result.get('error', 'Unknown error')}"
        elif result.get("no_output_dir"):
            self.unused_note = "Hide unused entries needs a translated output folder"
        elif result.get("no_files"):
            self.unused_note = f"No translated output files found in: {result.get('output_dir') or ''}"
        else:
            for line in result.get("errors", []) or []:
                self.service.log(line)
            used = frozenset(int(i) for i in result.get("used_rows", []) or [])
            self.state.used_rows = used
            self.unused_note = f"Showing {len(used)}/{total} used entries in translated output"
        self._apply_view()
        return result

    def _progress_text(self, text: str) -> None:
        if self.root is not None:
            self.stats.value = text
            self.ctx.push(self.stats)

    # ---- file switching / polling -----------------------------------------------------------------------

    def open_file_picker(self) -> Optional[ActionSheet]:
        files = self.screen.sibling_files()
        if not files:
            pending = getattr(self.screen, "listing_task", None)
            if pending is not None and not pending.done():  # the list is still being read (io pool)
                async def when_listed() -> None:
                    await asyncio.shield(pending)
                    if self.screen.sibling_files():
                        self.open_file_picker()

                self.ctx.spawn(when_listed())
                return None
            self.ctx.say("No other glossaries found")
            return None
        items = [ActionItem(f.name, lambda f=f: self.ctx.spawn(self.screen.switch_to(f.path)),
                            icon="CHECK" if self.path and os.path.normcase(f.path) == os.path.normcase(self.path) else
                            "DESCRIPTION", key=f"pick-{f.gid}") for f in files[:200]]
        sheet = ActionSheet(items, title="Glossary file", tablet=self.ctx.tablet)
        self.ctx.show(sheet)
        self.sheets.append(sheet)
        return sheet

    async def nav(self, direction: int) -> Optional[str]:
        """◀ ▶ through the glossary list (wraps around, like the desktop combo)."""
        files = self.screen.sibling_files()
        if len(files) <= 1:
            return None
        keys = [os.path.normcase(f.path) for f in files]
        current = keys.index(os.path.normcase(self.path)) if self.path and os.path.normcase(self.path) in keys else 0
        target = files[(current + direction) % len(files)]
        if await self.screen.switch_to(target.path) is False:
            return None
        return target.path

    def start_polling(self) -> None:
        if self._poll_task is not None and not self._poll_task.done():
            return
        self._poll_task = self.ctx.spawn(self._poll())

    def stop_polling(self) -> None:
        task = self._poll_task
        self._poll_task = None
        if task is not None and not task.done():
            task.cancel()

    async def _poll(self) -> None:
        """Auto-reload: an external change reloads a clean document; an edited one shows the banner.
        No checks while the app is in the background (``foreground.poll_sleep``)."""
        try:
            while True:
                await poll_sleep(getattr(self.ctx, "page", None), POLL_SECONDS)
                await self.check_disk()
        except asyncio.CancelledError:
            return

    async def check_disk(self) -> Optional[str]:
        doc = self.doc
        if doc is None or not self.path or self._saving or self.loading:
            return None
        try:
            mtime = await self.ctx.io(os.path.getmtime, self.path)
        except OSError:
            return None
        if not getattr(doc, "mtime", 0) or mtime == doc.mtime:
            return None
        if self.dirty:
            if not self.changed_on_disk:
                self.changed_on_disk = True
                if self.root is not None:
                    self.reload_dot.bgcolor = ft.Colors.AMBER_400
                    self.changed_banner.visible = True
                    self.ctx.push(self.root)
            return "changed"
        await self.reload(silent=True)
        return "reloaded"
