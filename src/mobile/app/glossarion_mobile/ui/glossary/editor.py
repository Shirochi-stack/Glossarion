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
    "edited" when the translated name differs from the saved one or a Resolve Gender
    decision waits for Save (desktop orange rows); on the right ✎ Edit (the EntrySheet)
    and, for a row whose tracked gender conflicts, the gender menu next to it (the desktop
    context menu's "Resolve Gender…": the decisions Auto / Male / Female applied at once,
    and "Resolve Gender…" for the full sheet with the tracking history);
    tap → EntrySheet, long-press → selection (Delete · Export selection · Change type),
    swipe left → delete with Undo;
  * FAB "＋ Entry".

Responsiveness (hotfix 1 issue 1: the app froze while the editor was used during a
translation): no event handler of the editor leaves Flet's whole-page auto-update behind
(``common.quiet`` / ``quiet_scroll``: a scroll event, a keystroke or a tap used to diff every
View, the chat with its live job card included); a render sends the editor's header and its
list, never the whole page; mounted rows are reused while their look is unchanged (a search,
a save or an auto-reload builds and sends only the rows that changed) and a one-row edit sends
that row alone. The auto-reload reads a file the running job rewrote into a new document off
the UI loop and swaps it in only while there are no unsaved edits; a half-written file leaves
the open document as it was.

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
from collections import OrderedDict
from typing import Any, Optional, Sequence

import flet as ft

from glossarion_mobile.services.glossary import (
    CoreMissing,
    GlossaryChanging,
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
from glossarion_mobile.ui.components.windowed_list import WINDOW_ROWS, WindowedList
from glossarion_mobile.ui.foreground import poll_sleep
from glossarion_mobile.ui.glossary.common import ask, chip, quiet
from glossarion_mobile.ui.glossary.entry_sheet import (
    GENDER_DECISIONS,
    RESOLVE_GENDER_LABEL,
    EntrySheet,
    GenderSheet,
    field_label,
)
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
#: Row controls kept for reuse (a window of mounted rows plus a search / filter round trip).
ROW_CACHE_MAX = 2 * WINDOW_ROWS
#: Rows one build step (one UI-loop callback: a scroll step, a search, a re-render) builds with their full
#: card; its further rows start as light placeholders (or keep their last look) and ``_fill`` completes them,
#: ``FILL_CHUNK`` per loop step. Sending a full row costs about 0.5 ms on the host (several on a phone), so a
#: whole 150-row step at once stalled the loop well past 100 ms.
FULL_ROWS_PER_STEP = 20
FILL_CHUNK = 12
PLACEHOLDER_HEIGHT = 72


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


#: desktop ``_apply_editor_gender_presentation``: the tint of a row whose tracked gender is unresolved
UNRESOLVED_GENDER_PINK = "#ec4899"


@ft.control(isolated=True)
class RowShell(ft.Dismissible):
    """A mounted editor row (swipe left: delete). Isolated: when the list is sent (a scroll step appended,
    a new search) Flet 1.0.3 compares the row object but not its card - without it every mounted row's
    whole subtree was diffed again on each append (hundreds of ms once a window filled up). A row that
    changes is sent on its own (``EditorPane._refresh_row`` / ``_rows_in_place``)."""


@ft.control(isolated=True)
class RowCard(ft.Container):
    """A mounted row in selection mode (the card itself, toggled in place); isolated like :class:`RowShell`."""


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
        self.header: Optional[ft.Control] = None
        self.sheets: list = []  # opened sheets (tests)
        self.reports: list = []  # OpReports shown (tests)
        self._by_key: dict = {}  # row key -> the current RowSpec (row handlers look their row up at tap time)
        #: row key -> (look, control): a mounted row is reused while what it shows is unchanged (``_row_view``)
        self._row_cache: "OrderedDict[str, tuple]" = OrderedDict()
        self._unreadable_mtime: Optional[float] = None  # an auto-reload that failed: retried once the file changes
        self._step_built = 0  # full rows built in this loop step (``FULL_ROWS_PER_STEP``)
        self._step_reset = False
        self._fill_queue: "OrderedDict[str, None]" = OrderedDict()  # mounted rows waiting for their card
        self._fill_task: Any = None
        self._filling = False
        # quiet_scroll: a scroll event that appends nothing sends nothing (no whole-page auto-update)
        self.windowed = WindowedList(build_row=self._row_control, key_of=lambda spec: spec.key, key="ge",
                                     quiet_scroll=True)
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
        # Every handler is ``quiet``: it pushes what it changes, Flet never diffs the whole page after it.
        self.prev_button = ft.IconButton(icon=ft.Icons.CHEVRON_LEFT, tooltip="Previous glossary",
                                         size_constraints=HIT_TARGET,
                                         on_click=quiet(lambda: self.ctx.spawn(self.nav(-1))),
                                         key="ge-prev")
        self.next_button = ft.IconButton(icon=ft.Icons.CHEVRON_RIGHT, tooltip="Next glossary",
                                         size_constraints=HIT_TARGET,
                                         on_click=quiet(lambda: self.ctx.spawn(self.nav(1))),
                                         key="ge-next")
        self.file_button = ft.TextButton(content=self._file_label(), icon=ft.Icons.ARROW_DROP_DOWN, expand=True,
                                         on_click=quiet(self.open_file_picker), key="ge-file")
        self.reload_dot = ft.Container(width=10, height=10, border_radius=5, bgcolor=ft.Colors.GREEN_400,
                                       tooltip="Auto-reload on change", key="ge-reload-dot")
        self.stats = ft.Text("Loading glossary…", theme_style=ft.TextThemeStyle.BODY_SMALL, italic=True,
                             color=ft.Colors.ON_SURFACE_VARIANT, max_lines=3, key="ge-stats")
        self.search = ft.TextField(hint_text="Search entries", prefix_icon=ft.Icons.SEARCH, dense=True, expand=True,
                                   on_change=quiet(self._on_search), key="ge-search")
        self.filter_button = ft.IconButton(icon=ft.Icons.FILTER_LIST, tooltip="Filter columns",
                                           size_constraints=HIT_TARGET, on_click=quiet(self.open_filter),
                                           key="ge-filter")
        self.sort_menu = ft.PopupMenuButton(icon=ft.Icons.SORT, tooltip="Sort", key="ge-sort", items=[
            ft.PopupMenuItem(content=label, on_click=quiet(lambda s=sid: self.set_sort(s)))
            for sid, label, _f, _d in SORTS])
        self.undo_button = ft.IconButton(icon=ft.Icons.UNDO, tooltip="Nothing to undo", disabled=True,
                                         size_constraints=HIT_TARGET,
                                         on_click=quiet(lambda: self.ctx.spawn(self.undo())),
                                         key="ge-undo")
        self.redo_button = ft.IconButton(icon=ft.Icons.REDO, tooltip="Nothing to redo", disabled=True,
                                         size_constraints=HIT_TARGET,
                                         on_click=quiet(lambda: self.ctx.spawn(self.undo(redo=True))), key="ge-redo")
        self.save_button = ft.IconButton(icon=ft.Icons.SAVE_OUTLINED, tooltip="Save", size_constraints=HIT_TARGET,
                                         on_click=quiet(lambda: self.ctx.spawn(self.save())), key="ge-save")
        self.menu = ft.PopupMenuButton(icon=ft.Icons.MORE_VERT, tooltip="Glossary editor", key="ge-menu",
                                       items=self._menu_items())
        self.filter_row = ft.Row(spacing=6, wrap=True, visible=False, key="ge-filter-chips")
        self.changed_banner = ft.Container(
            content=ft.Row([ft.Icon(ft.Icons.SYNC_PROBLEM, size=18),
                            ft.Text("The file changed on disk.", expand=True),
                            ft.TextButton(content="Reload",
                                          on_click=quiet(lambda: self.ctx.spawn(self.reload(force=True))))],
                           spacing=6),
            bgcolor=ft.Colors.TERTIARY_CONTAINER, border_radius=tokens.RADII["card"], padding=6, visible=False,
            key="ge-changed")
        self.selection_bar = SelectionTopBar(on_close=quiet(self.exit_selection), on_select_all=quiet(self.select_all),
                                             key="ge-selection")
        self.bulk_bar = BulkActionBar(page=self.ctx.page, tablet=self.ctx.tablet,
                                      compact=self.ctx.text_scale >= tokens.COMPACT_TEXT_SCALE, key="ge-bulk")
        self.loading_bar = ft.ProgressBar(visible=True, key="ge-loading")
        self.list_holder = ft.Container(content=self.windowed.control, expand=True, key="ge-list-holder")
        self.fab = ft.FloatingActionButton(icon=ft.Icons.ADD, content="Entry", on_click=quiet(self.open_new_entry),
                                           key="ge-fab")
        self.fab_slot = ft.Container(content=self.fab, right=16, bottom=16, key="ge-fab-slot")
        self.header = ft.Container(content=ft.Column([
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
        ], spacing=4, tight=True), padding=ft.Padding.only(left=8, right=8, top=4), key="ge-header")
        self.root = ft.Column([
            self.header,
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
            return ft.PopupMenuItem(content=label, icon=icon, checked=checked, on_click=quiet(handler))

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
        # the row cache stays: the new file's rows refill the mounted row objects (their looks differ)
        self._unreadable_mtime = None
        return await self.reload(source_path=source_path)

    async def reload(self, *, force: bool = False, source_path: Optional[str] = None, silent: bool = False) -> Any:
        """Read the file (desktop Reload / force refresh / the auto-reload: ``silent``)."""
        if not self.path:
            return None
        if silent and self.doc is not None:
            return await self._auto_reload()
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

    async def _auto_reload(self) -> Any:
        """The auto-reload (desktop ``_silent_glossary_reload``) of a file changed on disk - typically a running
        job writing the book's glossary: ``GlossaryService.reread_document`` reads it into a new document on
        the io pool. The open document is swapped for it only while it is still the open one, still has no
        unsaved edits (an edit made meanwhile keeps the "changed on disk" banner instead) and no save
        re-baselined it meanwhile. A file caught while it is written (``GlossaryChanging``) or that does
        not parse leaves the open document as it was: the first is read again on the next check, the
        second once the file changes again. Returns the new document, or None."""
        doc = self.doc
        if doc is None:
            return None
        baseline = getattr(doc, "mtime", None)
        self.loading = True
        try:
            fresh = await self.ctx.io(lambda: self.service.reread_document(doc))
        except GlossaryChanging:
            log.debug("%s is still being written; read again on the next check", self.path)
            return None
        except Exception as exc:
            log.info("auto-reloading %s failed (read again once it changes): %s", self.path, exc)
            try:
                self._unreadable_mtime = await self.ctx.io(os.path.getmtime, doc.path)
            except OSError:
                self._unreadable_mtime = None
            return None
        finally:
            self.loading = False
        if doc is not self.doc or self._saving or getattr(doc, "mtime", None) != baseline:
            return None  # another file, or a save wrote this one meanwhile (the next poll compares with it)
        if doc.dirty:
            self._show_changed_on_disk()
            return None
        self._unreadable_mtime = None
        self.doc = fresh
        self.changed_on_disk = False
        if self.root is not None:
            self.reload_dot.bgcolor = ft.Colors.GREEN_400
        await self._after_reload(keep_window=True)
        return fresh

    def _show_changed_on_disk(self) -> None:
        if self.changed_on_disk:
            return
        self.changed_on_disk = True
        if self.root is not None:
            self.reload_dot.bgcolor = ft.Colors.AMBER_400
            self.changed_banner.visible = True
            self.ctx.push(self.header)

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
            self.specs, self.visible, self._by_key = [], [], {}
            return
        self.specs = self.service.row_specs(doc)
        self._by_key = {spec.key: spec for spec in self.specs}
        self.visible = self.service.visible(doc, self.state, self.specs)

    # ---- render ------------------------------------------------------------------------------------

    def _render_header(self) -> None:
        """The header's state (file name, stats line, banner, filter chips, Undo / Redo / Save, ◀ ▶, ⋯)."""
        doc = self.doc
        self.file_button.content = self._file_label()
        self.stats.value = self._stats_text()
        self.stats.italic = doc is None
        self.changed_banner.visible = self.changed_on_disk
        self._render_filter_chips()
        self._sync_history()
        self._sync_nav()
        self._refresh_menu()

    def render(self, *, keep_window: bool = False) -> None:
        """The header, the list and the bars - sent as those controls, never as the whole page. Mounted rows
        whose look did not change are reused (``_row_control``), so only changed rows are built and sent."""
        if self.root is None:
            return
        doc = self.doc
        self._render_header()
        list_changed = True
        if self.error and doc is None:
            self.list_holder.content = EmptyState(icon="ERROR_OUTLINE", title="Glossary could not be opened",
                                                  body=self.error, key="ge-error",
                                                  primary=("Edit raw", quiet(lambda: self.ctx.spawn(self.open_raw()))))
        elif doc is not None and not self.specs:
            self.list_holder.content = EmptyState(icon="SPELLCHECK", title="No entries",
                                                  body="This glossary has no entries yet.", key="ge-empty",
                                                  primary=("＋ Entry", quiet(self.open_new_entry)))
        elif doc is not None and not self.visible:
            self.list_holder.content = EmptyState(icon="FILTER_LIST_OFF", title="No entries match",
                                                  body="Change the search or the filters.", key="ge-nomatch",
                                                  primary=("Clear filters", quiet(self.clear_filters)))
        else:
            in_place = (keep_window and self.list_holder.content is self.windowed.control
                        and self._rows_selecting == self.selecting and self._rows_in_place())
            self.list_holder.content = self.windowed.control
            if in_place:
                list_changed = False  # the same rows: only the ones that changed were sent
            else:
                self.windowed.set_items(self.visible, keep_window=keep_window, keep_rendered=keep_window)
            self._rows_selecting = self.selecting
        self.fab_slot.visible = doc is not None and not self.selecting
        self._sync_selection_bars()
        targets = [self.header, self.loading_bar, self.fab_slot, self.bulk_bar.control]
        if list_changed:
            targets.insert(2, self.list_holder)
        self.ctx.push(*targets)

    def _rows_in_place(self) -> bool:
        """``render(keep_window=True)`` when the list keeps its rows (same keys, same order - an auto-reload,
        a save, Select all): the mounted rows whose look changed get it in place and are sent one by one;
        the list itself is not sent (that diffs every mounted row). False when the rows changed, or a row
        cannot be refilled (the caller rebuilds the window)."""
        wl = self.windowed
        if len(wl.items) != len(self.visible) or any(a.key != b.key for a, b in zip(wl.items, self.visible)):
            return False
        wl.items = list(self.visible)  # the same keys at the same positions
        refilled: list = []
        for spec in self.visible[wl.window_start:wl.rendered]:
            shell = wl.controls.get(spec.key)
            if shell is None:
                return False
            view = self._row_view(spec)
            cached = self._row_cache.get(spec.key)
            if cached is not None and cached[1] is shell and cached[0] == view:
                continue
            if hasattr(shell, "_frozen") or not self._same_kind(shell):
                return False
            if not self._full_row_allowed():
                self._queue_fill(spec.key)  # the next loop steps send it (``_fill``)
                continue
            if not self._refill(shell, spec, view):
                return False
            self._remember_row(spec.key, view, shell)
            refilled.append(shell)
        self.ctx.push(*refilled)
        return True

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
                                 on_delete=quiet(lambda f=name: self.set_column_filter(f, None)),
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

    def _row_view(self, spec: RowSpec) -> tuple:
        """What a row shows (its look; also the row cache's check): every value of the entry, the tracked
        gender status, the edited flag, the selection and the highlight. ``_row_control`` reuses a mounted
        row while this is unchanged."""
        doc = self.doc
        entry = spec.entry
        status = self.service.gender_status(doc, entry) if doc is not None else None
        status = status if isinstance(status, dict) else {}
        updated = doc is not None and (is_updated(doc, spec) or self.service.gender_pending(doc, entry))
        return (
            spec.source_idx, str(spec.ref), tuple((str(k), str(v)) for k, v in entry.items()),
            str(status.get("label") or ""), bool(status.get("conflict")), bool(status.get("unresolved")),
            str(status.get("decision") or "auto"), bool(status) and self.service.can_resolve_gender(status),
            bool(updated), spec.source_idx in self.selected, self.selecting, spec.key == self.current_key,
            self.text_size,
        )

    def _row_control(self, spec: RowSpec, position: int = 0) -> ft.Control:
        """The WindowedList's ``build_row``: the cached row of ``spec`` while it looks the same; a cached row
        of the same kind refilled in place when it changed (the list then keeps that row object); else a new
        row. Keeping the row objects matters beyond the work saved: Flet 1.0.3 diffs a NEW object that takes
        the place of an old one with the same key as a frozen value, and a frozen row can never be changed
        in place again (``_refill`` / ``_mark_selected`` would raise). Past ``FULL_ROWS_PER_STEP`` rows in
        one loop step a row starts as a placeholder (or keeps its last look) and ``_fill`` completes it."""
        key = spec.key
        cached = self._row_cache.get(key)
        if cached is None and not self._full_row_allowed():
            return self._placeholder_for(key, spec)
        view = self._row_view(spec)
        if cached is not None:
            # hotfix 1 #1: a cached row coming back into the list (a search cleared, a filter removed, a scroll
            # step) is sent whole like a new row, so it counts towards this step's budget too; past it, it
            # starts as a placeholder that ``_fill`` completes (a mounted row costs nothing to keep)
            mounted = self._is_mounted(key, cached[1])
            if cached[0] == view:
                if not mounted and not self._full_row_allowed():
                    return self._placeholder_for(key, spec)
                self._row_cache.move_to_end(key)
                return cached[1]
            if not self._full_row_allowed():
                if mounted and not hasattr(cached[1], "_frozen") and self._same_kind(cached[1]):
                    self._queue_fill(key)
                    return cached[1]  # its last look until ``_fill`` reaches it
                return self._placeholder_for(key, spec)
        if cached is not None and self._refill(cached[1], spec, view):
            self._remember_row(key, view, cached[1])
            return cached[1]
        control = self._build_row(spec, view)
        self._remember_row(key, view, control)
        return control

    def _is_mounted(self, key: str, control: Any) -> bool:
        """``control`` is the row the list shows for ``key`` (while the list is rebuilt: what it showed until then,
        ``WindowedList.mounted_before``)."""
        wl = self.windowed
        before = wl.mounted_before
        return (before is not None and before.get(key) is control) or wl.controls.get(key) is control

    def _placeholder_for(self, key: str, spec: RowSpec) -> ft.Control:
        shell = self._placeholder_row(spec)
        self._remember_row(key, None, shell)
        self._queue_fill(key)
        return shell

    def _remember_row(self, key: str, view: Optional[tuple], control: Any) -> None:
        self._row_cache[key] = (view, control)
        self._row_cache.move_to_end(key)
        while len(self._row_cache) > ROW_CACHE_MAX:
            self._row_cache.popitem(last=False)

    def _same_kind(self, shell: Any) -> bool:
        """``shell`` is the kind of row the list shows now (a swipe row, or a selection-mode card)."""
        return isinstance(shell, RowCard) if self.selecting else isinstance(shell, ft.Dismissible)

    def _full_row_allowed(self) -> bool:
        """This loop step may still build a full row card: ``FULL_ROWS_PER_STEP`` per loop callback (always
        outside a running loop, and while ``_fill`` completes rows)."""
        if self._filling:
            return True
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return True
        if not self._step_reset:
            self._step_reset = True
            loop.call_soon(self._reset_step)
        self._step_built += 1
        return self._step_built <= FULL_ROWS_PER_STEP

    def _reset_step(self) -> None:
        self._step_built = 0
        self._step_reset = False

    def _placeholder_row(self, spec: RowSpec) -> ft.Control:
        """A row's light stand-in (an empty card of a row's height; no swipe) until ``_fill`` gives it its card."""
        if self.selecting:
            return RowCard(height=PLACEHOLDER_HEIGHT, border_radius=tokens.RADII["card"],
                           bgcolor=ft.Colors.SURFACE_CONTAINER_LOW, data={}, ink=True,
                           padding=ft.Padding.only(left=10, right=0, top=4, bottom=4),
                           on_click=quiet(lambda k=spec.key, s=spec: self._tap_key(k, s)),
                           on_long_press=quiet(lambda k=spec.key, s=spec: self._long_press_key(k, s)))
        return RowShell(content=ft.Container(height=PLACEHOLDER_HEIGHT, border_radius=tokens.RADII["card"],
                                             bgcolor=ft.Colors.SURFACE_CONTAINER_LOW),
                        dismiss_direction=ft.DismissDirection.NONE,
                        on_dismiss=quiet(lambda k=spec.key, s=spec: self._on_dismiss(k, s)))

    def _queue_fill(self, key: str) -> None:
        self._fill_queue[key] = None
        task = self._fill_task
        if task is None or task.done():
            self._fill_task = self.ctx.spawn(self._fill())

    async def _fill(self) -> None:
        """Complete the mounted rows a build step left as placeholders (or with their last look), up to
        ``FILL_CHUNK`` per loop step; each completed row is sent on its own (the list is not sent)."""
        while self._fill_queue:
            await asyncio.sleep(0)
            wl = self.windowed
            done: list = []
            rebuilt = False
            self._filling = True
            try:
                while self._fill_queue and len(done) < FILL_CHUNK:
                    key, _none = self._fill_queue.popitem(last=False)
                    shell, position = wl.controls.get(key), wl.positions.get(key)
                    if shell is None or position is None or position >= len(wl.items):
                        continue  # not mounted any more
                    spec = wl.items[position]
                    view = self._row_view(spec)
                    cached = self._row_cache.get(key)
                    if cached is not None and cached[1] is shell and cached[0] == view:
                        continue
                    if self._refill(shell, spec, view):
                        self._remember_row(key, view, shell)
                        done.append(shell)
                    elif wl.replace(spec):  # a frozen row: a new one takes its place
                        rebuilt = True
            finally:
                self._filling = False
            self.ctx.push(*done)
            if rebuilt:
                self.ctx.push(wl.list_view)

    def _refill(self, shell: Any, spec: RowSpec, view: tuple) -> bool:
        """Give a row object ``view``'s look in place: a swipe row gets a new card (a placeholder its swipe),
        a selection-mode card its new content, colour and frame. False when it cannot be (frozen by Flet, or
        the other kind of row)."""
        if hasattr(shell, "_frozen"):
            return False
        selecting = bool(view[10])
        if not selecting and isinstance(shell, ft.Dismissible):
            shell.content = self._row_content(spec, view)
            shell.dismiss_direction = ft.DismissDirection.END_TO_START
            if shell.background is None:
                shell.background = self._swipe_background()
            return True
        if selecting and isinstance(shell, RowCard) and isinstance(getattr(shell, "data", None), dict):
            card = self._row_content(spec, view)
            shell.content, shell.bgcolor, shell.border, shell.data = card.content, card.bgcolor, card.border, card.data
            shell.height = None
            card.content = None  # the Row now belongs to the mounted card
            return True
        return False

    @staticmethod
    def _swipe_background() -> ft.Control:
        return ft.Container(bgcolor=ft.Colors.ERROR, border_radius=tokens.RADII["card"],
                            alignment=ft.Alignment.CENTER_RIGHT, padding=ft.Padding.only(right=16),
                            content=ft.Icon(ft.Icons.DELETE_OUTLINE, color=ft.Colors.ON_ERROR))

    def _build_row(self, spec: RowSpec, view: tuple) -> ft.Control:
        if self.selecting:
            return self._row_content(spec, view, card=RowCard)  # toggled in place (_mark_selected: row.data)
        return RowShell(
            content=self._row_content(spec, view),
            dismiss_direction=ft.DismissDirection.END_TO_START,
            background=self._swipe_background(),
            on_dismiss=quiet(lambda k=spec.key, s=spec: self._on_dismiss(k, s)),
        )

    def _row_content(self, spec: RowSpec, view: tuple, card: type = ft.Container) -> ft.Container:
        """The row's card: the names, description and the chip line closed by the row number (expanding,
        ellipsized), then on the right the gender menu (rows with a tracked gender conflict) and ✎ Edit,
        48 dp targets each. ``card``: the card's class (``RowCard`` when the card is the mounted row)."""
        (_idx, _ref, _values, label, conflict, unresolved, decision, resolvable, updated, selected, selecting,
         current, text_size) = view
        doc = self.doc
        entry = spec.entry
        raw = str(entry.get(raw_field(doc), "") or entry.get("original", "") or "") if doc is not None else ""
        translated = str(entry.get(translated_field(doc), "") or "") if doc is not None else ""
        badges: list = []
        entry_type = str(entry.get("type") or "")
        if entry_type:
            badges.append(chip(entry_type, key="type"))
        gender = str(entry.get("gender") or "")
        # U12 item 6, desktop ``_apply_editor_gender_presentation``: the gender column shows the tracker's
        # label (the tracked decision, e.g. with its share) when the tracker knows the name
        tracked = label if label and not conflict else ""
        if gender or tracked:
            badges.append(chip(tracked or gender, bgcolor=ft.Colors.TERTIARY_CONTAINER, key="gender"))
        if conflict:
            badges.append(chip(f"⚠ {label or 'Gender conflict'}", bgcolor=ft.Colors.ERROR_CONTAINER,
                               key="conflict", tooltip=RESOLVE_GENDER_LABEL))
        if updated:
            badges.append(chip("edited", bgcolor=ft.Colors.ORANGE_200, key="edited"))
        lines: list = [ft.Text(raw or "—", size=text_size, weight=ft.FontWeight.W_600, max_lines=3,
                               overflow=ft.TextOverflow.ELLIPSIS),
                       ft.Text(f"→ {translated}" if translated else "→ (no translation)", size=text_size - 1,
                               max_lines=3, overflow=ft.TextOverflow.ELLIPSIS)]
        description = str(entry.get("description") or "")
        if description:
            lines.append(ft.Text(description, size=max(8, text_size - 3), color=ft.Colors.ON_SURFACE_VARIANT,
                                 max_lines=2, overflow=ft.TextOverflow.ELLIPSIS))
        # the row number (the desktop tree's # column) closes the chip line
        badges.append(ft.Text(f"#{spec.source_idx + 1}", size=10, color=ft.Colors.ON_SURFACE_VARIANT, key="number"))
        lines.append(ft.Row(badges, spacing=4, wrap=True, run_spacing=2,
                            vertical_alignment=ft.CrossAxisAlignment.CENTER))
        leading: list = []
        check = None
        if selecting:
            check = ft.Icon(ft.Icons.CHECK_CIRCLE if selected else ft.Icons.RADIO_BUTTON_UNCHECKED,
                            color=ft.Colors.PRIMARY if selected else ft.Colors.OUTLINE, size=22)
            leading.append(check)
        actions: list = [self._gender_menu(spec, decision)] if resolvable else []
        actions.append(ft.IconButton(icon=ft.Icons.EDIT_OUTLINED, tooltip="Edit", size_constraints=HIT_TARGET,
                                     on_click=quiet(lambda k=spec.key, s=spec: self.edit_key(k, s)),
                                     key=f"ge-edit-{spec.key}"))
        base_bgcolor = ft.Colors.with_opacity(0.10, ft.Colors.ORANGE) if updated else ft.Colors.SURFACE_CONTAINER_LOW
        if unresolved:  # desktop: unresolved rows are tinted pink
            base_bgcolor = ft.Colors.with_opacity(0.28, UNRESOLVED_GENDER_PINK)
        row = card(
            content=ft.Row(leading + [ft.Column(lines, spacing=1, expand=True, tight=True)] + actions,
                           spacing=4, vertical_alignment=ft.CrossAxisAlignment.START),
            padding=ft.Padding.only(left=10, right=0, top=4, bottom=4),
            border_radius=tokens.RADII["card"],
            bgcolor=ft.Colors.SECONDARY_CONTAINER if selected else base_bgcolor,
            border=ft.Border.all(2, ft.Colors.PRIMARY) if current else None,
            on_click=quiet(lambda k=spec.key, s=spec: self._tap_key(k, s)),
            on_long_press=quiet(lambda k=spec.key, s=spec: self._long_press_key(k, s)),
            ink=True,
        )
        if selecting:
            row.data = {"check": check, "bgcolor": base_bgcolor}
        return row

    def _gender_menu(self, spec: RowSpec, decision: str) -> ft.PopupMenuButton:
        """The row's gender menu (desktop context menu "Resolve Gender…" for a tracked conflict): the desktop
        decisions Auto / Male / Female (the current one checked), applied at once like the desktop dialog's
        Apply (undoable; written to the tracker on Save), and "Resolve Gender…" for the full sheet."""
        items = [ft.PopupMenuItem(content=label, checked=value == decision,
                                  on_click=quiet(lambda k=spec.key, s=spec, v=value: self.apply_gender_key(k, v, s)),
                                  key=f"ge-gender-{spec.key}-{value}")
                 for value, label in GENDER_DECISIONS]
        items.append(ft.PopupMenuItem(content=RESOLVE_GENDER_LABEL, icon=ft.Icons.WC,
                                      on_click=quiet(lambda k=spec.key, s=spec: self.open_gender_key(k, s)),
                                      key=f"ge-gender-{spec.key}-sheet"))
        return ft.PopupMenuButton(icon=ft.Icons.WC, tooltip=RESOLVE_GENDER_LABEL, size_constraints=HIT_TARGET,
                                  items=items, key=f"ge-gender-{spec.key}")

    def _refresh_row(self, spec: RowSpec) -> bool:
        """Show one row's new look at once: a mounted swipe row keeps its place in the list and only its card is
        replaced and sent (sending the list would diff every mounted row). False when it is not mounted."""
        key = spec.key
        position = self.windowed.positions.get(key)
        if position is not None and position < len(self.windowed.items):
            self.windowed.items[position] = spec
        shell = self.windowed.controls.get(key)
        if shell is None:
            return False
        view = self._row_view(spec)
        cached = self._row_cache.get(key)
        if cached is not None and cached[1] is shell and cached[0] == view:
            return True
        if self._refill(shell, spec, view):
            self._remember_row(key, view, shell)
            self.ctx.push(shell)
            return True
        if self._replace_row(spec):  # a frozen row: a new one takes its place (the list is sent)
            self.ctx.push(self.windowed.list_view)
            return True
        return False

    def _replace_row(self, spec: RowSpec) -> bool:
        """``WindowedList.replace`` with a full row (never a placeholder: one row is cheap)."""
        previous, self._filling = self._filling, True
        try:
            return self.windowed.replace(spec)
        finally:
            self._filling = previous

    def spec_for_key(self, key: Optional[str]) -> Optional[RowSpec]:
        if key is None:
            return None
        return self._by_key.get(key)

    def _current_spec(self, key: str, fallback: Optional[RowSpec] = None) -> Optional[RowSpec]:
        """The row's spec as it is now (rows are reused across recomputes, so their handlers look it up)."""
        spec = self._by_key.get(key)
        if spec is None and fallback is not None and not self._by_key:
            return fallback  # rows built without a computed document view
        return spec

    def _tap_key(self, key: str, fallback: Optional[RowSpec] = None) -> Any:
        spec = self._current_spec(key, fallback)
        return self.on_row_tap(spec) if spec is not None else None

    def _long_press_key(self, key: str, fallback: Optional[RowSpec] = None) -> None:
        spec = self._current_spec(key, fallback)
        if spec is not None:
            self.on_row_long_press(spec)

    def edit_key(self, key: str, fallback: Optional[RowSpec] = None) -> Optional[EntrySheet]:
        """The row's ✎ Edit: the entry sheet (also in selection mode, where a tap selects)."""
        spec = self._current_spec(key, fallback)
        if spec is None:
            return None
        self._set_current(spec.key)
        return self.open_entry(spec)

    def apply_gender_key(self, key: str, decision: str, fallback: Optional[RowSpec] = None) -> bool:
        """The gender menu's Auto / Male / Female."""
        spec = self._current_spec(key, fallback)
        return self.apply_gender(spec, decision) if spec is not None else False

    def open_gender_key(self, key: str, fallback: Optional[RowSpec] = None) -> Optional[GenderSheet]:
        """The gender menu's "Resolve Gender…"."""
        spec = self._current_spec(key, fallback)
        return self.open_gender(spec) if spec is not None else None

    def _on_dismiss(self, key: str, fallback: Optional[RowSpec] = None) -> Any:
        """Swipe left: a dismissed ``Dismissible`` must leave the tree in this update and never be mounted
        again, so its row leaves the list and the cache at once; the delete (and its re-render) follows."""
        self._row_cache.pop(key, None)
        spec = self._current_spec(key, fallback)
        if self.windowed.remove_key(key):
            self.ctx.push(self.windowed.control)
        return self.ctx.spawn(self.swipe_delete(spec)) if spec is not None else None

    def _set_current(self, key: Optional[str]) -> None:
        """Highlight ``key``'s row (and un-highlight the last one) in place."""
        previous, self.current_key = self.current_key, key
        if self.root is None:
            return
        for k in dict.fromkeys((previous, key)):
            spec = self.spec_for_key(k)
            if spec is not None:
                self._refresh_row(spec)

    def on_row_tap(self, spec: RowSpec) -> Any:
        if self.selecting:
            self.toggle(spec.source_idx)
            return None
        self._set_current(spec.key)
        return self.open_entry(spec)

    def on_row_long_press(self, spec: RowSpec) -> None:
        if not self.selecting:
            self.ctx.haptic("medium_impact")
        self.selecting = True
        self.toggle(spec.source_idx, force=True)

    async def jump_to(self, key: str) -> bool:
        self._set_current(key)
        jumped = await self.windowed.jump_to(key)
        spec = self.spec_for_key(key)
        if spec is not None:
            self._refresh_row(spec)  # the target gets its card now even when the jump built it as a placeholder
        return jumped

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
        if parts is None or parts.get("check") is None or hasattr(control, "_frozen"):
            # a placeholder (refilled now) or a frozen row (replaced): that row alone, in full
            position = self.windowed.positions.get(key)
            if position is not None and position < len(self.windowed.items):
                self._refresh_row(self.windowed.items[position])
            return
        check = parts["check"]
        check.icon = ft.Icons.CHECK_CIRCLE if selected else ft.Icons.RADIO_BUTTON_UNCHECKED
        check.color = ft.Colors.PRIMARY if selected else ft.Colors.OUTLINE
        control.bgcolor = ft.Colors.SECONDARY_CONTAINER if selected else parts["bgcolor"]
        spec = self.spec_for_key(key)
        cached = self._row_cache.get(key)
        if spec is not None and cached is not None and cached[1] is control:
            self._remember_row(key, self._row_view(spec), control)  # the cache knows the row's new look
        else:
            self._row_cache.pop(key, None)
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
                           quiet(lambda: self.ctx.spawn(self.delete_selected())), destructive=True),
                BulkAction("export", "Export selection", "IOS_SHARE", quiet(self.open_export_selection)),
                BulkAction("type", "Change type", "CATEGORY", quiet(self.open_change_type),
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
            # the same rows: that row's card and the header go out, not the list (or the page)
            self.windowed.items = list(self.visible)
            candidate = self._by_key.get(spec.key)
            if candidate is not None:
                self._refresh_row(candidate)
            self.stats.value = self._stats_text()
            self.file_button.content = self._file_label()
            self._sync_history()
            self.ctx.push(self.header)
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
        self.ctx.push(self.header)
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
            if not result:
                self.ctx.say(f"⚠️ {'Redo' if redo else 'Undo'} (output files) failed")
            self._sync_history()
            if self.root is not None:
                self.ctx.push(self.header)
        elif kind == "glossary":
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
            if report.get("saved") and self.hide_unused:
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
            if confirm and not report.ok:
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
        elif not report.ok:
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
        return await self.run_tool("convert", dest)

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
                                    on_save=self.service.set_backup_settings,
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
        if report is not None and not report.ok:
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
        if not report.ok:
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
        if not report.ok:
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
        """The screen is disposed: no more auto-reload checks, and no row completion either."""
        task = self._poll_task
        self._poll_task = None
        if task is not None and not task.done():
            task.cancel()
        self._fill_queue.clear()
        fill, self._fill_task = self._fill_task, None
        if fill is not None and not fill.done():
            fill.cancel()

    async def _poll(self) -> None:
        """Auto-reload: an external change reloads a clean document; an edited one shows the banner.
        No checks while the app is in the background (``foreground.poll_sleep``). A failed check (the file
        being rewritten, a render error) is logged and the next interval checks again."""
        try:
            while True:
                await poll_sleep(getattr(self.ctx, "page", None), POLL_SECONDS)
                try:
                    await self.check_disk()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    log.exception("glossary auto-reload check failed")
        except asyncio.CancelledError:
            return

    async def check_disk(self) -> Optional[str]:
        """One auto-reload check (desktop ``_silent_glossary_reload``, every 2 s): "reloaded" when a clean
        document was swapped for the file's new state, "changed" when unsaved edits keep it (the banner),
        else None. Nothing while a save writes the file or a read runs; a file state the last auto-reload
        could not read is not read again."""
        doc = self.doc
        if doc is None or not self.path or self._saving or self.loading:
            return None
        try:
            mtime = await self.ctx.io(os.path.getmtime, self.path)
        except OSError:
            return None
        if doc is not self.doc or self._saving or self.loading:
            return None  # another file, a save or a read started meanwhile (it sets the baseline)
        if not getattr(doc, "mtime", 0) or mtime == doc.mtime or mtime == self._unreadable_mtime:
            return None
        if self.dirty:
            self._show_changed_on_disk()
            return "changed"
        if await self.reload(silent=True) is not None:
            return "reloaded"
        return "changed" if self.changed_on_disk else None
