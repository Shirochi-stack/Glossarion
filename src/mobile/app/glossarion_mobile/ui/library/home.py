"""Library home (``/library``; UI_SPEC §3.1-§3.4, §3.12).

* App bar actions: search, filter sheet (``tune``), grid/list, ⋯ (Scan for raw (N) ·
  Organize (n) · Undo (n) · Refresh · Library settings).
* Shelf bar: ``SegmentedButton`` "In progress (N) · Completed (N)" (``epub_library_tab``
  0 / 1) + the teal "Scan for raw (N)" chip when N > 0.
* Body: ``GridView(max_extent=card_w)`` for the density preset, or a 72 dp list;
  the shelf is appended ``epub_library_page_size`` cards at a time as the user
  scrolls ("All" renders through the lazy grid); pull-to-refresh (top overscroll)
  and ⋯ Refresh run a full rescan.
* Refresh: a quiet ``library_core`` scan every 2 s only while this screen is on top
  and the app is in the foreground (``Poller``); results are diffed by the shared
  ``card_signature``: changed cards are updated in place, the grid is rebuilt only
  when the visible set changes (content-only changes do not re-sort, desktop rule).
* Long-press -> selection mode: top bar "N selected · Select all · ✕" and the bulk
  bar Translate · Metadata · Compile · Delete N · More (Delete glossary files ·
  Restore glossary backup · Clear saved raw link · Share · …).
* Extended FAB: "Import EPUB" (In progress) / "Add translation" (Completed); files
  are copied into Library/Raw (Translated) and registered with
  ``library_core.import_paths`` (FileBridge).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Mapping, Optional, Sequence

import flet as ft

from glossarion_mobile.services.library import Poller, ScanSnapshot, book_key
from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.components.action_sheet import ActionItem, ActionSheet
from glossarion_mobile.ui.components.empty_state import EmptyState
from glossarion_mobile.ui.library.book_card import BookCard, BookListRow, LIST_ROW_HEIGHT
from glossarion_mobile.ui.library.common import LibraryContext, icon_button
from glossarion_mobile.ui.library.delete_confirm import DeleteFlow
from glossarion_mobile.ui.library.filter_sheet import FilterSheet
from glossarion_mobile.ui.library.models import (
    CARD_TEXT_HEIGHT,
    CardModel,
    DEFAULT_DENSITY,
    DENSITY_ORDER,
    FilterState,
    build_card,
    density_preset,
    effective_density,
    page_size_value,
    visible_books,
)
from glossarion_mobile.ui.library.organize import clear_raw_link_flow, organize_flow, undo_flow
from glossarion_mobile.ui.library.selection_bar import BulkAction, BulkActionBar, SelectionTopBar
from glossarion_mobile.ui.library.translate_sheet import open_translate_sheet
from glossarion_mobile.ui.screens.base import Screen

__all__ = [
    "EMPTY_COMPLETED",
    "EMPTY_IN_PROGRESS",
    "LibraryScreen",
    "PREF_SHOW_LANGUAGE",
    "PREF_SHOW_PROGRESS",
    "PREF_VIEW",
    "SHELVES",
]

log = logging.getLogger("glossarion.library.ui")

SHELVES = ("progress", "completed")
EMPTY_IN_PROGRESS = "No translations in progress.\nUse “Import EPUB” to start one."
EMPTY_COMPLETED = "Your Library is empty.\n\nAdd finished .epub files with “Add translation” to see them here."
# Mobile-only display prefs (Prefs / mobile_state.json; config.json gets no new keys).
PREF_VIEW = "library_view"
PREF_SHOW_LANGUAGE = "library_show_language"
PREF_SHOW_PROGRESS = "library_show_progress"
GLOSSARY_FILES_REASON = "Arrives with the Glossary tools (U6)"
SERIES_REASON = "Arrives in U9"
RAW_IMPORT_EXTENSIONS = ["epub", "txt", "pdf", "html", "htm"]
SCROLL_APPEND_PX = 600


def _shelf_from_tab(value: Any) -> str:
    text = str(value if value is not None else "0").strip().lower()
    return "completed" if text in ("1", "completed", "comp") else "progress"


def _opens_in_another_app(book: Mapping[str, Any]) -> bool:
    """A TXT file, or a PDF without a translation workspace: library_core's "system" open
    decision (the desktop opens it in the system viewer; the Reader cannot show it)."""
    kind = str(book.get("type") or "")
    return kind == "txt" or (kind == "pdf" and not book.get("output_folder"))


class LibraryScreen(Screen):
    title = "Library"

    def __init__(self, match: Any, ctx: LibraryContext) -> None:
        super().__init__(match)
        self.ctx = ctx
        service = ctx.service
        self.service = service
        cfg = service.cfg
        query_shelf = match.get("shelf") if match is not None else None
        self.shelf = query_shelf if query_shelf in SHELVES else _shelf_from_tab(cfg("epub_library_tab", 0))
        self.filter = FilterState(fmt=str(cfg("epub_library_format_filter", "all") or "all"),
                                  sort=str(cfg("epub_library_sort", "date") or "date"))
        self.view_mode = str(self._pref(PREF_VIEW, "grid"))
        density = str(cfg("epub_library_card_size", DEFAULT_DENSITY) or DEFAULT_DENSITY)
        self.density = density if density in DENSITY_ORDER else DEFAULT_DENSITY
        self.raw_titles = bool(cfg("epub_library_show_raw_titles", False))
        self.show_language = bool(self._pref(PREF_SHOW_LANGUAGE, False))
        self.show_progress = bool(self._pref(PREF_SHOW_PROGRESS, True))
        self.page_size_raw = cfg("epub_library_page_size", 20)
        self.page_size = page_size_value(self.page_size_raw)
        self.selecting = False
        self.selected: dict = {"progress": set(), "completed": set()}
        self.cards: dict = {}  # key -> BookCard / BookListRow
        self.books_by_key: dict = {}
        self.visible_keys: list = []
        self.rendered = 0
        self.searching = False
        self.snapshot: ScanSnapshot = service.snapshot
        self._signatures: dict = {}
        self._cover_queue: list = []
        self._cover_running = False
        self._unsub: Any = None
        self._refresh_pending = False
        self._header_sig: Any = None  # what the shelf header last showed (_refresh_header)
        self._shown_error: Optional[str] = None  # the scan error the banner shows
        self.sheet: Any = None
        self.delete_flow: Optional[DeleteFlow] = None
        self.poller = Poller(self._tick, interval=2.0, visible=lambda: self.ctx.is_top(self),
                             foreground=self.ctx.foreground, spawn=self.ctx.spawn, name="library")

    # ---- prefs / config ------------------------------------------------------------------------

    def _pref(self, key: str, default: Any) -> Any:
        prefs = self.ctx.prefs
        if prefs is None:
            return default
        try:
            value = prefs.get(key, default)
        except Exception:
            return default
        return default if value is None else value

    def _set_pref(self, key: str, value: Any) -> None:
        prefs = self.ctx.prefs
        if prefs is not None:
            try:
                prefs.set(key, value)
            except Exception:
                log.debug("saving pref %s failed", key, exc_info=True)

    # ---- app bar --------------------------------------------------------------------------------

    def actions(self) -> list:
        self.search_button = icon_button("SEARCH", "Search", self._toggle_search, key="lib-search")
        self.filter_button = icon_button("TUNE", "Filter, sort and display", self.open_filter_sheet,
                                         key="lib-filter")
        self.view_button = icon_button("VIEW_LIST" if self.view_mode == "grid" else "GRID_VIEW",
                                       "List view" if self.view_mode == "grid" else "Grid view",
                                       self._toggle_view, key="lib-view")
        self.menu_items = {
            "scan": ft.PopupMenuItem(content="Scan for raw", icon=ft.Icons.SEARCH,
                                     on_click=lambda e: self.ctx.go("library.scan_raw")),
            "organize": ft.PopupMenuItem(content="Organize", icon=ft.Icons.DRIVE_FILE_MOVE,
                                         on_click=lambda e: self.ctx.spawn(organize_flow(self.ctx))),
            "undo": ft.PopupMenuItem(content="Undo", icon=ft.Icons.UNDO,
                                     on_click=lambda e: self.ctx.spawn(undo_flow(self.ctx))),
            "refresh": ft.PopupMenuItem(content="Refresh", icon=ft.Icons.REFRESH,
                                        on_click=lambda e: self.ctx.spawn(self.full_refresh())),
            "settings": ft.PopupMenuItem(content="Library settings", icon=ft.Icons.SETTINGS,
                                         on_click=lambda e: self.open_filter_sheet(tab=2)),
        }
        self.menu = ft.PopupMenuButton(icon=ft.Icons.MORE_VERT, tooltip="More", items=list(self.menu_items.values()),
                                       key="lib-menu")
        self._sync_menu()
        return [self.search_button, self.filter_button, self.view_button, self.menu]

    def _sync_menu(self) -> None:
        items = getattr(self, "menu_items", None)
        if not items:
            return
        snap = self.snapshot
        items["scan"].content = f"Scan for raw ({snap.missing_raw})"
        items["scan"].disabled = snap.missing_raw <= 0
        items["organize"].content = f"Organize ({snap.organize_count})"
        items["organize"].disabled = snap.organize_count <= 0
        items["undo"].content = f"Undo ({snap.undo_count})"
        items["undo"].disabled = snap.undo_count <= 0

    # ---- body ------------------------------------------------------------------------------------

    def build_body(self) -> ft.Control:
        snap = self.snapshot
        self.shelf_buttons = ft.SegmentedButton(
            segments=[ft.Segment(value="progress", label=ft.Text(self._shelf_label("progress"))),
                      ft.Segment(value="completed", label=ft.Text(self._shelf_label("completed")))],
            selected=[self.shelf],
            show_selected_icon=False,
            on_change=self._on_shelf,
            key="shelf",
        )
        self.scan_chip = ft.Chip(
            label=ft.Text(f"Scan for raw ({snap.missing_raw})", color="#ffffff"),
            leading=ft.Icon(ft.Icons.SEARCH, color="#ffffff", size=16),
            bgcolor="#17a2b8",
            on_click=lambda e: self.ctx.go("library.scan_raw"),
            visible=snap.missing_raw > 0,
            key="scan-raw-chip",
        )
        self.shelf_row = ft.Row([self.shelf_buttons, self.scan_chip], wrap=True, spacing=8, run_spacing=4,
                                vertical_alignment=ft.CrossAxisAlignment.CENTER, key="shelf-row")
        self.selection_bar = SelectionTopBar(on_close=self.exit_selection, on_select_all=self.select_all)
        self.search_field = ft.TextField(
            hint_text="Filter title or tag…", prefix_icon=ft.Icons.SEARCH, dense=True,
            on_change=self._on_query, value=self.filter.query, visible=self.searching,
            suffix=ft.IconButton(icon=ft.Icons.CLOSE, tooltip="Clear search", on_click=self._clear_search),
            key="lib-search-field",
        )
        self.count_text = ft.Text("", theme_style=ft.TextThemeStyle.LABEL_SMALL, color=ft.Colors.ON_SURFACE_VARIANT,
                                  key="lib-count")
        self.loading = ft.ProgressBar(visible=False, key="lib-loading")
        self.banner = ft.Container(visible=False, key="lib-banner", bgcolor=ft.Colors.ERROR_CONTAINER,
                                   border_radius=tokens.RADII["card"], padding=8)
        self.list_holder = ft.Container(expand=True, key="lib-list")
        self.fab = ft.FloatingActionButton(icon=ft.Icons.ADD, content=self._fab_label(), on_click=self._on_fab,
                                           key="lib-fab")
        self.fab_slot = ft.Container(content=self.fab, right=16, bottom=16, key="lib-fab-slot")
        self.bulk_bar = BulkActionBar(page=self.ctx.page, tablet=self.ctx.tablet,
                                      compact=self.ctx.text_scale >= tokens.COMPACT_TEXT_SCALE)
        self.body_stack = ft.Stack([self.list_holder, self.fab_slot], expand=True)
        self._header_sig = None  # fresh header / banner controls: the next scan fills them
        self._shown_error = None
        self._render(structural=True)
        return ft.Column([
            ft.Container(content=ft.Column([self.shelf_row, self.selection_bar.control, self.search_field,
                                            self.count_text], spacing=6, tight=True),
                         padding=ft.Padding.only(left=12, right=12, top=8)),
            self.loading,
            ft.Container(content=self.banner, padding=ft.Padding.symmetric(horizontal=12)),
            self.body_stack,
            self.bulk_bar.control,
        ], spacing=4, expand=True)

    def _fab_label(self) -> str:
        return "Import EPUB" if self.shelf == "progress" else "Add translation"

    def _shelf_label(self, shelf: str) -> str:
        count = len(self.snapshot.shelf(shelf))
        return f"In progress ({count})" if shelf == "progress" else f"Completed ({count})"

    # ---- lifecycle ------------------------------------------------------------------------------

    def did_show(self) -> None:
        if self._unsub is None:
            self._unsub = self.service.subscribe(self._on_snapshot)
        self.poller.start()
        if self.service.dirty or not self.snapshot.scanned_at:
            self.ctx.spawn(self.full_refresh(initial=not self.snapshot.scanned_at))

    def dispose(self) -> None:
        self.poller.stop()
        if self._unsub is not None:
            self._unsub()
            self._unsub = None

    async def _tick(self) -> None:
        await self.service.refresh(quiet=True, reason="poll")

    async def full_refresh(self, initial: bool = False) -> Optional[ScanSnapshot]:
        self.loading.visible = True
        if initial and not self.snapshot.scanned_at:
            self.count_text.value = "Scanning library…"
        self.ctx.push(self.loading, self.count_text)
        try:
            snap = await self.service.refresh(reason="full")
        finally:
            self.loading.visible = False
            self.ctx.push(self.loading)
        return snap

    # ---- snapshots and diffing ---------------------------------------------------------------------

    def _on_snapshot(self, snap: ScanSnapshot) -> None:
        """Apply a scan. The 2 s quiet scans usually change nothing: then nothing is pushed (the
        header only when its counts change, the list only on structural changes, changed cards
        in place)."""
        old_sigs = dict(self.snapshot.signatures)
        self.snapshot = snap
        banner_changed = snap.error != self._shown_error
        self._shown_error = snap.error
        if banner_changed:
            if snap.error:
                self.banner.content = ft.Row([
                    ft.Icon(ft.Icons.ERROR_OUTLINE, color=ft.Colors.ON_ERROR_CONTAINER),
                    ft.Text(snap.error, color=ft.Colors.ON_ERROR_CONTAINER, expand=True),
                    ft.TextButton(content="Retry", on_click=lambda e: self.ctx.spawn(self.full_refresh())),
                ])
                self.banner.visible = True
            else:
                self.banner.visible = False
        self._refresh_header()
        old_names = {k: str(b.get("name") or "") for k, b in self.books_by_key.items()}
        new_keys = self._compute_visible()
        changed = {k for k, sig in snap.signatures.items() if old_sigs.get(k) != sig}
        # A compile that ended without touching the workspace (cancelled while queued, failed early)
        # leaves the signature as it was: its card still shows the "⚙ COMPILING…" ribbon.
        changed |= {k for k, card in self.cards.items()
                    if (getattr(getattr(card, "model", None), "ribbon_state", "") == "compiling")
                    != self.service.is_compiling(self.books_by_key.get(k) or self._book(k))}
        # Desktop 2 s rule: rebuild when the visible set changes (or a name moves the A-Z order);
        # otherwise replace only the changed cards, in place, without re-sorting.
        name_changed = self.filter.sort == "name" and any(
            k in changed and old_names.get(k) != str(self.books_by_key.get(k, {}).get("name") or "")
            for k in new_keys)
        structural = set(new_keys) != set(self.visible_keys) or name_changed
        if structural:
            self._render(structural=True, keys=new_keys)
            self.ctx.push(self.list_holder, self.count_text)
        else:
            for key in changed:
                self.service.covers.pop(key, None)
                self._update_card(key)
                if key in self.cards:
                    self._cover_queue.append(key)
            if self._cover_queue and not self._cover_running:
                self.ctx.spawn(self._load_covers())
        if banner_changed:
            self.ctx.push(self.banner)

    def _book(self, key: str) -> dict:
        for book in self.snapshot.all_books():
            if book_key(book) == key:
                return dict(book)
        return {}

    def _refresh_header(self) -> bool:
        """Shelf counts, the Scan-for-raw chip and the ⋯ menu counts; pushed only when they change."""
        snap = self.snapshot
        segments = getattr(self.shelf_buttons, "segments", [])
        labels = tuple(self._shelf_label(segment.value) for segment in segments)
        header = (labels, snap.missing_raw, snap.organize_count, snap.undo_count)
        if header == self._header_sig:
            return False
        self._header_sig = header
        for segment, label in zip(segments, labels):
            segment.label = ft.Text(label)
        self.scan_chip.label = ft.Text(f"Scan for raw ({snap.missing_raw})", color="#ffffff")
        self.scan_chip.visible = snap.missing_raw > 0
        self._sync_menu()
        self.ctx.push(self.shelf_row, getattr(self, "menu", None))
        return True

    def _compute_visible(self) -> list:
        service = self.service
        books = list(self.snapshot.shelf(self.shelf))
        ordered = visible_books(books, self.filter, matches=service.matches_query, format_of=service.format_of,
                                sort=lambda bs, mode, rev: service.sort_books(bs, mode, reverse=rev))
        self.books_by_key = {book_key(b): dict(b) for b in ordered}
        return [book_key(b) for b in ordered]

    def _card_model(self, key: str) -> CardModel:
        service = self.service
        book = self.books_by_key.get(key) or self._book(key)
        bid = service.bid_for(book)
        has_continue = False
        prefs = self.ctx.prefs
        if prefs is not None and hasattr(prefs, "reader_position"):
            try:
                has_continue = prefs.reader_position(bid) is not None
            except Exception:
                has_continue = False
        badge, size = service.card_badge(book)
        return build_card(
            book, key=key, bid=bid, view=self.snapshot.views.get(key), raw_titles=self.raw_titles,
            raw_title=service.raw_title(book) if self.raw_titles else None, compiling=service.is_compiling(book),
            has_continue=has_continue, selected=key in self.selected[self.shelf],
            signature=self.snapshot.signatures.get(key), dark=self.ctx.dark, badge_text=badge, size_label=size)

    def _make_card(self, key: str) -> Any:
        model = self._card_model(key)
        common = dict(cover_src=self.service.covers.get(key), dark=self.ctx.dark, on_open=self._on_card_tap,
                      on_long_press=self._on_card_long_press, on_continue=self._on_continue,
                      on_more=self._on_card_more, show_language=self.show_language,
                      show_progress=self.show_progress)
        if self.view_mode == "list":
            return BookListRow(model, **common)
        density = effective_density(self.density, self.ctx.text_scale)
        card_w, cover_h = density_preset(density, self.service.size_presets())
        return BookCard(model, card_w=card_w, cover_h=cover_h,
                        title_lines=3 if self.ctx.text_scale >= tokens.COMPACT_TEXT_SCALE else 2, **common)

    def _update_card(self, key: str) -> None:
        card = self.cards.get(key)
        if card is None:
            return
        card.set_model(self._card_model(key))
        card.update()

    def _grid(self) -> ft.Control:
        density = effective_density(self.density, self.ctx.text_scale)
        card_w, cover_h = density_preset(density, self.service.size_presets())
        return ft.GridView(
            max_extent=card_w,
            child_aspect_ratio=card_w / float(cover_h + CARD_TEXT_HEIGHT),
            spacing=8,
            run_spacing=8,
            padding=ft.Padding.only(left=12, right=12, top=4, bottom=96),
            expand=True,
            build_controls_on_demand=True,
            on_scroll=self._on_scroll,
            scroll_interval=120,
            key="lib-grid",
        )

    def _list(self) -> ft.Control:
        return ft.ListView(spacing=4, padding=ft.Padding.only(left=8, right=8, top=4, bottom=96), expand=True,
                           build_controls_on_demand=True, on_scroll=self._on_scroll, scroll_interval=120,
                           key="lib-listview")

    def _render(self, *, structural: bool, keys: Optional[Sequence[str]] = None) -> None:
        self.visible_keys = list(keys) if keys is not None else self._compute_visible()
        if not self.visible_keys:
            self.cards = {}
            self.scroller = None
            scanning = not self.snapshot.scanned_at
            text = EMPTY_IN_PROGRESS if self.shelf == "progress" else EMPTY_COMPLETED
            if self.snapshot.shelf(self.shelf) and not scanning:
                title, body = "No books match", "Change the search or the filters to see more books."
            else:
                title, _, body = text.partition("\n")
                body = body.strip()
            self.list_holder.content = EmptyState(
                icon="LOCAL_LIBRARY", title="Scanning library…" if scanning else title,
                body=None if scanning else body, key="lib-empty")
            self._update_counts()
            return
        self.scroller = self._grid() if self.view_mode == "grid" else self._list()
        keep = {}
        self.rendered = 0
        self.list_holder.content = self.scroller
        self.cards = keep
        self._append_page()
        self._update_counts()

    def _page_increment(self) -> int:
        return self.page_size if self.page_size > 0 else len(self.visible_keys)

    def _append_page(self) -> int:
        if self.scroller is None:
            return 0
        start = self.rendered
        end = min(len(self.visible_keys), start + self._page_increment())
        controls = []
        for key in self.visible_keys[start:end]:
            card = self._make_card(key)
            self.cards[key] = card
            controls.append(card.control)
            if key not in self.service.covers:
                self._cover_queue.append(key)
        self.scroller.controls.extend(controls)
        self.rendered = end
        if self._cover_queue and not self._cover_running:
            self.ctx.spawn(self._load_covers())
        return end - start

    def _update_counts(self) -> None:
        total = len(self.snapshot.shelf(self.shelf))
        shown = len(self.visible_keys)
        noun = "novel" if self.shelf == "progress" else "book"
        if not self.snapshot.scanned_at:
            text = "Scanning library…"
        elif shown == total:
            text = f"{total} {noun}{'s' if total != 1 else ''}"
        else:
            text = f"{shown} of {total} {noun}{'s' if total != 1 else ''}"
        self.count_text.value = text

    async def _load_covers(self) -> None:
        self._cover_running = True
        try:
            while self._cover_queue:
                key = self._cover_queue.pop(0)
                book = self.books_by_key.get(key)
                if book is None:
                    continue
                try:
                    path = await self.ctx.io(self.service.cover_blocking, book)
                except Exception:
                    path = None
                card = self.cards.get(key)
                if card is not None and path:
                    card.set_cover(path)
                    card.update()
        finally:
            self._cover_running = False

    # ---- scrolling ---------------------------------------------------------------------------------

    def _on_scroll(self, e: Any) -> None:
        event_type = str(getattr(getattr(e, "event_type", None), "value", getattr(e, "event_type", "")))
        if event_type == "overscroll":
            overscroll = getattr(e, "overscroll", 0) or 0
            pixels = getattr(e, "pixels", 0) or 0
            minimum = getattr(e, "min_scroll_extent", 0) or 0
            if overscroll < 0 and pixels <= minimum + 1 and not self._refresh_pending:
                self._refresh_pending = True
                self.ctx.haptic("selection_click")
                self.ctx.spawn(self._pull_refresh())
            return
        pixels = getattr(e, "pixels", None)
        maximum = getattr(e, "max_scroll_extent", None)
        if pixels is None or maximum is None:
            return
        if maximum - pixels < SCROLL_APPEND_PX and self.rendered < len(self.visible_keys):
            if self._append_page():
                self.ctx.push(self.scroller)

    async def _pull_refresh(self) -> None:
        try:
            await self.full_refresh()
        finally:
            self._refresh_pending = False

    # ---- shelf / search / filters -----------------------------------------------------------------

    def _on_shelf(self, e: Any = None) -> None:
        selected = list(getattr(self.shelf_buttons, "selected", []) or [])
        shelf = selected[0] if selected else "progress"
        self.set_shelf(shelf)

    def set_shelf(self, shelf: str) -> None:
        if shelf not in SHELVES or shelf == self.shelf:
            return
        self.shelf = shelf
        self.shelf_buttons.selected = [shelf]
        self.service.set_cfg("epub_library_tab", 0 if shelf == "progress" else 1)
        self.fab.content = self._fab_label()
        if self.selecting:
            self._sync_selection()
        self._render(structural=True)
        self.ctx.push(self.shelf_row, self.list_holder, self.fab, self.count_text)

    def _toggle_search(self, e: Any = None) -> None:
        self.searching = not self.searching or bool(self.filter.query)
        self.search_field.visible = self.searching
        self.ctx.push(self.search_field)

    def _clear_search(self, e: Any = None) -> None:
        self.search_field.value = ""
        self.set_query("")
        self.searching = False
        self.search_field.visible = False
        self.ctx.push(self.search_field)

    def _on_query(self, e: Any = None) -> None:
        self.set_query(str(self.search_field.value or ""))

    def set_query(self, text: str) -> None:
        self.filter.query = text
        self._render(structural=True)
        self.ctx.push(self.list_holder, self.count_text)

    def open_filter_sheet(self, e: Any = None, tab: int = 0) -> FilterSheet:
        sheet = FilterSheet(self.filter, view_mode=self.view_mode, density=self.density, raw_titles=self.raw_titles,
                            show_language=self.show_language, show_progress=self.show_progress,
                            page_size=str(self.page_size_raw).lower(), on_change=self.apply_filter_change,
                            initial_tab=tab if isinstance(tab, int) else 0)
        self.sheet = sheet
        if self.ctx.page is not None:
            sheet.show(self.ctx.page)
        return sheet

    def apply_filter_change(self, name: str, value: Any) -> None:
        service = self.service
        if name == "fmt":
            self.filter.fmt = str(value)
            service.set_cfg("epub_library_format_filter", self.filter.fmt)
        elif name == "sort":
            self.filter.sort = str(value)
            service.set_cfg("epub_library_sort", self.filter.sort)
        elif name == "reverse":
            self.filter.reverse = bool(value)
        elif name == "states":
            self.filter.states = dict(value or {})
        elif name == "view":
            self.view_mode = "list" if value == "list" else "grid"
            self._set_pref(PREF_VIEW, self.view_mode)
            self._sync_view_button()
        elif name == "density":
            if value in DENSITY_ORDER:
                self.density = str(value)
                service.set_cfg("epub_library_card_size", self.density)
        elif name == "raw_titles":
            self.raw_titles = bool(value)
            service.set_cfg("epub_library_show_raw_titles", self.raw_titles)
        elif name == "show_language":
            self.show_language = bool(value)
            self._set_pref(PREF_SHOW_LANGUAGE, self.show_language)
        elif name == "show_progress":
            self.show_progress = bool(value)
            self._set_pref(PREF_SHOW_PROGRESS, self.show_progress)
        elif name == "page_size":
            text = str(value).lower()
            self.page_size_raw = "all" if text == "all" else int(text)
            self.page_size = page_size_value(self.page_size_raw)
            service.set_cfg("epub_library_page_size", self.page_size_raw)
        self._render(structural=True)
        self.ctx.push(self.list_holder, self.count_text)

    def _toggle_view(self, e: Any = None) -> None:
        self.apply_filter_change("view", "grid" if self.view_mode == "list" else "list")

    def _sync_view_button(self) -> None:
        button = getattr(self, "view_button", None)
        if button is None:
            return
        from glossarion_mobile.ui.theme import icon_data

        button.icon = icon_data("VIEW_LIST" if self.view_mode == "grid" else "GRID_VIEW")
        button.tooltip = "List view" if self.view_mode == "grid" else "Grid view"
        self.ctx.push(button)

    # ---- cards ---------------------------------------------------------------------------------------

    def selected_books(self) -> list:
        keys = self.selected[self.shelf]
        return [dict(self.books_by_key.get(k) or self._book(k)) for k in self.visible_keys if k in keys] + [
            self._book(k) for k in keys if k not in self.visible_keys and self._book(k)]

    def _on_card_tap(self, model: CardModel) -> None:
        if self.selecting:
            self.toggle(model.key)
            return
        book = self.books_by_key.get(model.key) or {}
        if _opens_in_another_app(book):
            # The Reader opens EPUBs and translation workspaces only (library_core's "system"
            # decision): share the file to an external app (UI_SPEC §3.3), else the Book page.
            self.ctx.spawn(self._share_or_open(book, model.bid))
            return
        self.ctx.go("library.book", {"bid": model.bid})

    async def _share_or_open(self, book: Mapping[str, Any], bid: str) -> bool:
        """A TXT file / a PDF without a workspace: the share sheet; True when it was shown."""
        path = str(book.get("path") or "")
        files = self.ctx.files
        if files is not None and path and os.path.isfile(path):
            try:
                if await files.share([path]):
                    return True
            except Exception:
                log.exception("sharing %s failed", path)
        self.ctx.go("library.book", {"bid": bid})
        return False

    def _on_card_long_press(self, model: CardModel) -> None:
        if not self.selecting:
            self.ctx.haptic("medium_impact")
            self.selecting = True
        self.toggle(model.key, force=True)

    def _on_continue(self, model: CardModel) -> None:
        """▶ Continue opens the Reader at the saved position (UI_SPEC §3.2)."""
        book = self.books_by_key.get(model.key) or self._book(model.key)
        self.ctx.open_reader(book, bid=model.bid, resume=True)

    def _on_card_more(self, model: CardModel) -> ActionSheet:
        book = self.books_by_key.get(model.key) or self._book(model.key)
        sheet = self.card_actions(book)
        self.ctx.show(sheet)
        return sheet

    # ---- selection -----------------------------------------------------------------------------------

    def toggle(self, key: str, force: Optional[bool] = None) -> None:
        keys = self.selected[self.shelf]
        on = (key not in keys) if force is None else force
        if on:
            keys.add(key)
        else:
            keys.discard(key)
        if not keys:
            self.exit_selection()
            self._update_card(key)
            return
        self.selecting = True
        self._update_card(key)
        self._sync_selection()

    def select_all(self) -> None:
        self.selecting = True
        self.selected[self.shelf] = set(self.visible_keys)
        for key in self.visible_keys:
            self._update_card(key)
        self._sync_selection()

    def handle_back(self) -> bool:
        """Android back leaves selection mode first (UI_SPEC §1.6 rule 2)."""
        if self.selecting:
            self.exit_selection()
            return True
        return False

    def exit_selection(self) -> None:
        previous = set(self.selected[self.shelf])
        self.selecting = False
        self.selected[self.shelf] = set()
        for key in previous:
            self._update_card(key)
        self._sync_selection()

    def _sync_selection(self) -> None:
        count = len(self.selected[self.shelf])
        active = self.selecting and count > 0
        self.selection_bar.set_count(count)
        self.selection_bar.show(active)
        self.shelf_row.visible = not active
        self.fab_slot.visible = not active
        self.bulk_bar.show(active)
        if active:
            primary, more = self.bulk_actions()
            self.bulk_bar.set_actions(primary, more)
        self.ctx.push(self.selection_bar.control, self.shelf_row, self.fab_slot, self.bulk_bar.control)

    def bulk_actions(self) -> tuple:
        books = self.selected_books()
        count = len(books)
        service = self.service
        with_raw = [b for b in books if b.get("raw_source_path") and not b.get("missing_raw_file")]
        epubs = [b for b in with_raw if str(b.get("raw_source_path") or "").lower().endswith(".epub")]
        workspaces = [b for b in books if b.get("output_folder")]
        metadata_reason = None if service.has_job_kind("metadata") else "Metadata translation jobs arrive in U6"
        if not epubs:
            metadata_reason = "No raw EPUB resolves for the selection"
        primary = [
            BulkAction("translate", f"Load {count} for translation" if count != 1 else "Load for translation",
                       "TRANSLATE", lambda: self.ctx.spawn(open_translate_sheet(self.ctx, books)),
                       None if with_raw else "No raw source file resolves for the selection"),
            BulkAction("metadata", f"Translate Metadata for {len(epubs)} EPUB{'s' if len(epubs) != 1 else ''}",
                       "LABEL", lambda: self.ctx.spawn(self.translate_metadata(epubs)), metadata_reason),
            BulkAction("compile", "Compile", "MENU_BOOK", lambda: self.compile_menu(workspaces),
                       None if workspaces else "No output workspace in the selection"),
            BulkAction("delete", f"Delete {count}", "DELETE_OUTLINE", lambda: self.ctx.spawn(self.delete_books(books)),
                       destructive=True),
        ]
        more = [
            BulkAction("glossary_delete", f"Delete glossary files ({count})", "DELETE_SWEEP",
                       disabled_reason=GLOSSARY_FILES_REASON),
            BulkAction("glossary_restore", "Restore glossary backup", "RESTORE", disabled_reason=GLOSSARY_FILES_REASON),
            BulkAction("clear_raw", f"Clear saved raw link (for {count} item{'s' if count != 1 else ''})",
                       "LINK_OFF", lambda: self.ctx.spawn(clear_raw_link_flow(self.ctx, books))),
            BulkAction("share", "Share", "IOS_SHARE", lambda: self.ctx.spawn(self.share_books(books))),
            BulkAction("series", "Add to Series", "COLLECTIONS_BOOKMARK", disabled_reason=SERIES_REASON),
        ]
        if count == 1:
            more.insert(0, BulkAction("card", "Book actions…", "MORE_HORIZ",
                                      lambda: self.ctx.show(self.card_actions(books[0]))))
        return primary, more

    # ---- actions ---------------------------------------------------------------------------------------

    def card_actions(self, book: Mapping[str, Any]) -> ActionSheet:
        """The single-card ⋯ sheet: desktop context-menu labels and visibility rules (epub_library 12326)."""
        service = self.service
        bid = service.bid_for(book)
        raw = str(book.get("raw_source_path") or "")
        has_raw = bool(raw) and not book.get("missing_raw_file")
        folder = str(book.get("output_folder") or "")
        kind = str(book.get("type") or "")
        has_progress = bool(book.get("progress_file")) or bool(folder and os.path.isfile(
            os.path.join(folder, "translation_progress.json")))
        is_epub = kind == "epub" or raw.lower().endswith(".epub")
        if not (is_epub or kind == "pdf"):
            reader_reason: Optional[str] = "The Reader opens EPUB and PDF books"
        elif _opens_in_another_app(book):
            # The card tap shares it; the desktop menu offers no Reader item for it either.
            reader_reason = "A PDF without a translation workspace opens in another app (↗ Share)"
        else:
            reader_reason = None
        items = [
            ActionItem("\U0001f4d1 Open Book Details", lambda: self.ctx.go("library.book", {"bid": bid}),
                       icon="MENU_BOOK"),
            ActionItem("\U0001f4d6 Open in Reader", lambda: self.ctx.open_reader(book, bid=bid), icon="AUTO_STORIES",
                       disabled_reason=reader_reason),
            ActionItem("\U0001f501 Load for translation",
                       lambda: self.ctx.spawn(open_translate_sheet(self.ctx, [book])), icon="TRANSLATE",
                       disabled_reason=None if has_raw else "The raw source file can't be found"),
            ActionItem("\U0001f310 Translate Metadata", lambda: self.ctx.spawn(self.translate_metadata([book])),
                       icon="LABEL",
                       disabled_reason=(None if has_raw and raw.lower().endswith(".epub")
                                        and service.has_job_kind("metadata")
                                        else "Needs a raw EPUB" if not (has_raw and raw.lower().endswith(".epub"))
                                        else "Metadata translation jobs arrive in U6")),
            ActionItem("\U0001f4d8 Compile EPUB", lambda: self.ctx.spawn(self.compile([book], "compile_epub")),
                       icon="MENU_BOOK", disabled_reason=None if has_progress else "No translation_progress.json"),
            ActionItem("\U0001f4c4 Compile PDF", lambda: self.ctx.spawn(self.compile([book], "compile_pdf")),
                       icon="PICTURE_AS_PDF", disabled_reason=None if has_progress else "No translation_progress.json"),
            ActionItem("\U0001f4c1 Files", lambda: self.open_files(book), icon="FOLDER_OPEN",
                       disabled_reason=None if folder else "No output folder"),
            ActionItem("↗ Share", lambda: self.ctx.spawn(self.share_books([book])), icon="IOS_SHARE"),
            ActionItem("\U0001f4cb Copy Path", lambda: self.copy_path(book), icon="CONTENT_COPY",
                       disabled_reason=None if self._developer() else "Developer setting"),
            ActionItem("✂️ Clear saved raw link", lambda: self.ctx.spawn(clear_raw_link_flow(self.ctx, [book])),
                       icon="LINK_OFF"),
            ActionItem("\U0001f5d1️ Delete glossary files", icon="DELETE_SWEEP",
                       disabled_reason=GLOSSARY_FILES_REASON),
            ActionItem("↩️ Restore glossary backup", icon="RESTORE", disabled_reason=GLOSSARY_FILES_REASON),
            ActionItem("\U0001f5d1️ Delete", lambda: self.ctx.spawn(self.delete_books([book])),
                       icon="DELETE_OUTLINE", destructive=True),
        ]
        return ActionSheet(items, title=str(book.get("name") or ""), tablet=self.ctx.tablet)

    def _developer(self) -> bool:
        return bool(self._pref("developer_mode", False))

    def copy_path(self, book: Mapping[str, Any]) -> Any:
        copy = self.ctx.copy_text
        if copy is None:
            return None
        return copy(str(book.get("path") or ""))

    def open_files(self, book: Mapping[str, Any]) -> None:
        folder = str(book.get("output_folder") or "")
        prefs = self.ctx.prefs
        if not folder or prefs is None:
            self.ctx.go("tools.files", {"root": "output"})
            return
        self.ctx.go("tools.files.folder", {"root": "output", "fid": prefs.file_ref(folder)})

    def compile_menu(self, books: Sequence[Mapping[str, Any]]) -> ActionSheet:
        sheet = ActionSheet([
            ActionItem("\U0001f4d8 Compile EPUB", lambda: self.ctx.spawn(self.compile(books, "compile_epub")),
                       icon="MENU_BOOK"),
            ActionItem("\U0001f4c4 Compile PDF", lambda: self.ctx.spawn(self.compile(books, "compile_pdf")),
                       icon="PICTURE_AS_PDF"),
        ], title=f"Compile {len(books)} book{'s' if len(books) != 1 else ''}", tablet=self.ctx.tablet)
        self.ctx.show(sheet)
        return sheet

    async def compile(self, books: Sequence[Mapping[str, Any]], kind: str) -> list:
        service = self.service
        if service.jobs is None or not service.has_job_kind(kind):
            self.ctx.say("The job service is not running")
            return []
        started = []
        for book in books:
            try:
                started.append(await service.submit(service.compile_spec(book, kind)))
            except Exception as exc:
                self.ctx.say(f"{book.get('name') or ''}: {exc}")
        if started:
            self.ctx.haptic("medium_impact")
            self.ctx.say(f"{'Compiling' if len(started) == 1 else f'Compiling {len(started)} books'}"
                         f" · {'EPUB' if kind == 'compile_epub' else 'PDF'}", "Jobs", lambda: self.ctx.go("jobs"))
            for book in books:
                self._update_card(book_key(book))
        return started

    async def translate_metadata(self, books: Sequence[Mapping[str, Any]]) -> Optional[str]:
        service = self.service
        if not service.has_job_kind("metadata"):
            self.ctx.say("Metadata translation jobs arrive in U6")
            return None
        try:
            spec = await self.ctx.io(service.metadata_spec, list(books))
            return await service.submit(spec)
        except Exception as exc:
            self.ctx.say(f"Could not start: {exc}")
            return None

    async def share_books(self, books: Sequence[Mapping[str, Any]]) -> bool:
        files = self.ctx.files
        if files is None:
            self.ctx.say("Sharing is not available in this session")
            return False

        def targets() -> list:
            out = []
            for book in books:
                outputs = self.service.compiled_outputs_blocking(book)
                if outputs:
                    out.append(outputs[0][0])
                elif book.get("path") and os.path.isfile(str(book.get("path"))):
                    out.append(str(book.get("path")))
                elif self.service.raw_source(book):
                    out.append(self.service.raw_source(book))
            return out

        paths = await self.ctx.io(targets)
        if not paths:
            self.ctx.say("Nothing to share for the selection")
            return False
        return bool(await files.share(paths))

    # ---- delete ------------------------------------------------------------------------------------------

    async def delete_books(self, books: Sequence[Mapping[str, Any]]) -> Any:
        """Two-level delete (``DeleteFlow``); selection mode ends after it."""
        self.delete_flow = DeleteFlow(self.ctx, on_done=lambda report: self.exit_selection())
        return await self.delete_flow.start(list(books))

    # ---- import (FAB) -------------------------------------------------------------------------------------

    def _on_fab(self, e: Any = None) -> Any:
        return self.ctx.spawn(self.import_files())

    async def import_files(self) -> list:
        files = self.ctx.files
        if files is None:
            self.ctx.say("Importing is not available in this session")
            return []
        translated = self.shelf == "completed"
        try:
            imported = await files.pick_files(target="translated" if translated else "library",
                                              allowed_extensions=["epub"] if translated else RAW_IMPORT_EXTENSIONS,
                                              dialog_title="Add translation" if translated else "Import EPUB")
        except Exception as exc:
            self.ctx.say(f"Import failed: {exc}")
            return []
        if not imported:
            return []
        names = ", ".join(f.name for f in imported[:3]) + (f" +{len(imported) - 3}" if len(imported) > 3 else "")
        self.ctx.say(f"Added to Library: {names}")
        self.service.mark_dirty()
        await self.service.refresh(reason="import")
        return imported
