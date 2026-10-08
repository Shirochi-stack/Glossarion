"""SourcePicker (UI_SPEC §4.2, §5.5): what a tool runs on.

A ``BottomSheet`` with the segments **Recent outputs** · **Library books** · **Chat
workspaces** · **Browse**. Rows are ``targets.ToolTarget`` (output folder + raw source),
loaded on the io pool. Single mode picks on tap; multi mode has checkboxes and a "Use N"
button (a bulk QA scan, metadata for several EPUBs). Each tool passes ``eligible(target)``:
rows it cannot use stay visible, disabled, with the reason (QA skips Direct Text
workspaces, the Converter needs an output folder, metadata needs a raw EPUB...). The
reasons are computed once per row on the io pool (``load`` / ``browse``), never again on a
render or a keystroke.

Browse picks EPUB / TXT / PDF files through ``FileBridge.pick_files`` (copied into the
Inbox) and finds each file's output folder with the QA Scanner's auto-search
(``targets.target_for_source``).

Opt-in options (the chat's ＋ › From Library, UI_SPEC §2.5; the Tools pickers keep the
defaults):

* ``segments`` - the segments offered; with one segment the SegmentedButton is hidden.
* ``searchable`` / ``query`` - the Library's "Filter title or tag…" field (prefilled with
  ``query``), filtered on every change with the shared ``book_matches_query``
  (``LibraryService.matches_query``), no debounce (like Library home).
* ``book_rows`` - Library rows are the Library's own list rows (``models.card_model_for`` +
  ``BookListRow``: cover, type badge, progress pill; no ⋯), newest first
  (``targets.order_library_rows``: the Library's ``visible_books`` with its Date sort); missing
  covers load through the Library's ``CoverQueue``. Rows are built once and kept across
  renders (Flet 1.0.3 freezes a control re-created under its key; covers arrive in place).
  They are paged like Library home (``models.next_page_end`` / ``wants_next_page``, the Library's
  ``epub_library_page_size``): the next page is built when the scroll nears the end, so a large
  Library opens (and a search clears) without building every row on the UI loop.
* ``long_press_selects`` - a long-press on a book row starts a multi-selection (the Library
  shelves' long-press): further taps add or remove rows and "Use N" finishes; a plain tap still
  picks one row at once.
* ``header_action`` - ``(label, callback)``: a TextButton in the header (the picker closes first).
* ``include_unresolved`` - also list Library books that have neither an output folder nor a
  raw file (``targets.library_targets``); the tool's ``eligible`` keeps them disabled.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any, Callable, Mapping, Optional, Sequence

import flet as ft

from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.components.dialogs import close_dialog
from glossarion_mobile.ui.components.reason_chip import ReasonChip
from glossarion_mobile.ui.components.sheet import sheet_frame
from glossarion_mobile.ui.theme import HIT_TARGET
from glossarion_mobile.ui.tools import targets as tg

__all__ = ["LIBRARY_SORT", "SEARCH_HINT", "SEGMENTS", "SourcePicker"]

log = logging.getLogger("glossarion.tools")

SEGMENTS = (("recent", "Recent outputs"), ("library", "Library books"), ("chat", "Chat workspaces"),
            ("browse", "Browse"))
BROWSE_EXTENSIONS = ("epub", "txt", "pdf")
#: The Library search field's copy (UI_SPEC §3.1).
SEARCH_HINT = "Filter title or tag…"
#: Book rows come newest first: the Library's Date sort (``library_core.sort_books`` SORT_DATE).
LIBRARY_SORT = tg.LIBRARY_SORT


def _subtitle(target: tg.ToolTarget) -> str:
    parts = []
    if target.folder:
        parts.append(f"📁 {target.folder_name}")
    else:
        parts.append("No output folder yet")
    if target.source:
        parts.append(f"📖 {target.source_name}")
    elif target.origin != "chat":
        parts.append("No raw source")
    return " · ".join(parts)


def _book_of(target: tg.ToolTarget) -> Optional[Mapping[str, Any]]:
    extra = target.extra
    book = extra.get("book") if isinstance(extra, Mapping) else None
    return book if isinstance(book, Mapping) else None


def _reason_of(eligible: Callable[[tg.ToolTarget], Optional[str]], target: tg.ToolTarget) -> Optional[str]:
    try:
        return eligible(target)
    except Exception:
        log.exception("checking %s for the picker failed", target.key)
        return "Could not check this source"


def _card_models(service: Any, rows: Sequence[tg.ToolTarget], dark: bool) -> dict:
    """``{target key: CardModel}`` of the Library rows (io): the Library's own card model."""
    from glossarion_mobile.ui.library.models import card_model_for

    snapshot = getattr(service, "snapshot", None)
    views = getattr(snapshot, "views", None) or {}
    cfg = getattr(service, "cfg", None)
    try:
        raw_titles = bool(cfg("epub_library_show_raw_titles", False)) if callable(cfg) else False
    except Exception:
        raw_titles = False
    models: dict = {}
    for target in rows:
        book = _book_of(target)
        if book is None or target.key in models:
            continue
        try:
            models[target.key] = card_model_for(service, book, views=views, raw_titles=raw_titles, dark=dark)
        except Exception:
            log.debug("card model for %s failed", target.key, exc_info=True)
    return models


class SourcePicker:
    def __init__(self, ctx: Any, *, title: str, multi: bool = False,
                 eligible: Optional[Callable[[tg.ToolTarget], Optional[str]]] = None,
                 on_done: Optional[Callable[[list], Any]] = None, selected: Sequence[tg.ToolTarget] = (),
                 segment: str = "recent", browse_label: str = "Pick a source file…",
                 find_folder: bool = True, segments: Optional[Sequence[str]] = None, searchable: bool = False,
                 query: str = "", book_rows: bool = False,
                 header_action: Optional[tuple] = None, include_unresolved: bool = False,
                 long_press_selects: bool = False) -> None:
        self.ctx = ctx
        self.find_folder = find_folder  # Browse: look for the picked file's output folder (QA auto-search)
        self.title = title
        self.multi = multi
        self.eligible = eligible or (lambda t: None)
        self.on_done = on_done
        wanted = tuple(segments) if segments is not None else None
        self.segments = tuple(value for value, _label in SEGMENTS if wanted is None or value in wanted) or tuple(
            value for value, _label in SEGMENTS)
        self.segment = segment if segment in self.segments else self.segments[0]
        self.searchable = searchable
        self.query = str(query or "")
        self.book_rows = book_rows
        self.header_action = header_action
        self.include_unresolved = include_unresolved
        self.long_press_selects = long_press_selects
        self.rows: dict = {"recent": [], "library": [], "chat": [], "browse": []}
        self.shown: list = []  # the rows ``render`` lists (filtered, ordered); ``rendered`` of them are built
        self.rendered = 0
        self.loaded = False
        self.selected: dict = {t.key: t for t in selected}
        self.result: Optional[list] = None
        self.tiles: dict = {}
        self.reasons: dict = {}  # target key -> eligible(target), computed once (io)
        self.models: dict = {}  # target key -> CardModel of a Library row (book_rows; io)
        self.book_cards: dict = {}  # target key -> (BookListRow, row control): kept, so covers update in place
        self._cover_rows: dict = {}  # book key -> BookListRow waiting for its cover
        self._covers: Any = None  # the Library's CoverQueue (book_rows)
        self.segmented = ft.SegmentedButton(
            segments=[ft.Segment(value=value, label=ft.Text(label)) for value, label in SEGMENTS
                      if value in self.segments],
            selected=[self.segment], allow_multiple_selection=False, show_selected_icon=False,
            on_change=self._on_segment, key="picker-segments")
        self.segment_row = ft.Row([self.segmented], scroll=ft.ScrollMode.AUTO, visible=len(self.segments) > 1)
        # The list takes the sheet's free height (Cancel / Done stay below it): a fixed 360 dp list
        # pushed the multi-select Done past the sheet at large text.
        self.list_view = ft.ListView(spacing=2, expand=True, build_controls_on_demand=True, key="picker-list",
                                     on_scroll=self._on_scroll, scroll_interval=120)
        self.status = ft.Text("", theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT,
                              key="picker-status")
        self.search_field: Optional[ft.TextField] = None
        if searchable:
            self.search_field = ft.TextField(
                hint_text=SEARCH_HINT, prefix_icon=ft.Icons.SEARCH, dense=True, value=self.query,
                on_change=self._on_query,
                suffix=ft.IconButton(icon=ft.Icons.CLOSE, tooltip="Clear search", on_click=self._clear_query,
                                     size_constraints=HIT_TARGET),
                key="picker-search")
        self.browse_button = ft.FilledTonalButton(content=browse_label, icon=ft.Icons.FILE_OPEN_OUTLINED,
                                                  on_click=self._on_browse, visible=self.segment == "browse",
                                                  key="picker-browse")
        self.done_button = ft.FilledButton(content=self._done_label(), on_click=self._on_done,
                                           visible=multi, disabled=not self.selected, key="picker-done")
        heading = ft.Text(title, theme_style=ft.TextThemeStyle.TITLE_LARGE, weight=ft.FontWeight.W_600,
                          key="picker-title")
        header: ft.Control = heading
        self.header_button: Optional[ft.TextButton] = None
        if header_action is not None:
            heading.expand = True
            self.header_button = ft.TextButton(content=str(header_action[0]), on_click=self._on_header_action,
                                               key="picker-header-action")
            header = ft.Row([heading, self.header_button], vertical_alignment=ft.CrossAxisAlignment.CENTER)
        controls: list = [header, self.segment_row]
        if self.search_field is not None:
            controls.append(self.search_field)
        controls += [
            self.browse_button,
            self.status,
            self.list_view,
            ft.Row([ft.TextButton(content="Cancel", on_click=lambda e: self.close(), key="picker-cancel"),
                    self.done_button], alignment=ft.MainAxisAlignment.END),
        ]
        content = ft.Column(controls, spacing=tokens.SPACING["sm"])
        self.sheet = ft.BottomSheet(content=sheet_frame(content, padding=tokens.SPACING["sheet_padding"]),
                                    show_drag_handle=True, scrollable=True, bgcolor=ft.Colors.SURFACE_CONTAINER_HIGH)
        self._page: Any = None

    # ---- lifecycle -------------------------------------------------------------------------------

    def show(self, page: Any) -> "SourcePicker":
        self._page = page
        page.show_dialog(self.sheet)
        self.ctx.spawn(self.load())
        return self

    def close(self) -> None:
        close_dialog(self._page, self.sheet)

    async def load(self) -> None:
        self.status.value = "Loading…"
        self.ctx.push(self.status)
        service = self.ctx.service
        if service is not None:
            snapshot = getattr(service, "snapshot", None)
            if snapshot is not None and not getattr(snapshot, "scanned_at", None) and hasattr(service, "refresh"):
                try:
                    await service.refresh(quiet=True, reason="tools source picker")
                except Exception:
                    log.debug("library refresh for the picker failed", exc_info=True)
        chats_root = self.ctx.chats_root
        segments = set(self.segments)
        include_unresolved = self.include_unresolved
        eligible = self.eligible
        book_rows = self.book_rows
        dark = bool(getattr(self.ctx, "dark", False))

        def gather() -> dict:
            library = (tg.library_targets(service, include_unresolved=include_unresolved)
                       if service is not None and segments & {"recent", "library"} else [])
            rows = {
                "recent": tg.recent_output_targets(service, library_rows=library)
                if service is not None and "recent" in segments else [],
                "library": library if "library" in segments else [],
                "chat": tg.chat_workspace_targets(chats_root) if "chat" in segments else [],
            }
            reasons: dict = {}
            for found_rows in rows.values():
                for target in found_rows:
                    if target.key not in reasons:
                        reasons[target.key] = _reason_of(eligible, target)
            models = _card_models(service, rows["library"], dark) if book_rows and service is not None else {}
            return {"rows": rows, "reasons": reasons, "models": models}

        try:
            found = await self.ctx.io(gather)
        except Exception as exc:
            log.exception("loading the source rows failed")
            found = {}
            self.status.value = f"Could not list sources: {exc}"
        for key, rows in ((found or {}).get("rows") or {}).items():
            self.rows[key] = list(rows)
        self.reasons.update((found or {}).get("reasons") or {})
        self.models.update((found or {}).get("models") or {})
        self.loaded = True
        self.render()

    # ---- rendering ----------------------------------------------------------------------------

    def _done_label(self) -> str:
        count = len(self.selected)
        return f"Use {count}" if count else "Use"

    def reason(self, target: tg.ToolTarget) -> Optional[str]:
        """``eligible(target)``: the reason computed when the row was loaded (else computed now, once)."""
        key = target.key
        if key not in self.reasons:
            self.reasons[key] = _reason_of(self.eligible, target)
        return self.reasons[key]

    def visible_rows(self) -> list:
        """The current segment's rows; a searchable / book-row picker filters them with the query and
        orders Library books newest first (``targets.order_library_rows``: the Library's own
        ``visible_books``)."""
        rows = list(self.rows.get(self.segment, []))
        if not (self.searchable or self.book_rows):
            return rows
        query = self.query.strip()
        service = self.ctx.service
        if self.segment == "library" and service is not None:
            return tg.order_library_rows(service, rows, query)
        return [t for t in rows if tg.target_matches(service, t, query)] if query else rows

    def page_size(self) -> int:
        """Rows built per page: book rows follow the Library's page size (``epub_library_page_size``,
        "All" = 0 builds every row); plain tiles are cheap and are all built."""
        if not self.book_rows:
            return 0
        from glossarion_mobile.ui.library.models import DEFAULT_PAGE_SIZE, page_size_value

        cfg = getattr(self.ctx.service, "cfg", None)
        try:
            raw = cfg("epub_library_page_size", DEFAULT_PAGE_SIZE) if callable(cfg) else DEFAULT_PAGE_SIZE
        except Exception:
            raw = DEFAULT_PAGE_SIZE
        return page_size_value(raw)

    def _row_control(self, target: tg.ToolTarget) -> ft.Control:
        if self.book_rows and target.key in self.models:
            return self._book_row(target)
        return self._tile(target)

    def _append_rows(self, minimum: int = 0) -> int:
        """Build the next page of ``shown`` (``models.next_page_end``; at least ``minimum`` rows in
        all); returns how many were added."""
        from glossarion_mobile.ui.library.models import next_page_end

        start = self.rendered
        end = max(next_page_end(start, len(self.shown), self.page_size()), min(int(minimum), len(self.shown)))
        self.list_view.controls.extend(self._row_control(target) for target in self.shown[start:end])
        self.rendered = end
        return end - start

    def _on_scroll(self, e: Any) -> None:
        from glossarion_mobile.ui.library.models import wants_next_page

        if wants_next_page(e, self.rendered, len(self.shown)) and self._append_rows():
            self.ctx.push(self.list_view)
            self._start_covers()

    def render(self, *, keep_rendered: bool = False) -> None:
        """List the current rows (the first page; ``keep_rendered``: as many as were built, so a
        selection change keeps the list where the user scrolled it)."""
        built = self.rendered if keep_rendered else 0
        rows = self.visible_rows()
        self.tiles = {}
        self.shown = rows
        self.rendered = 0
        self.list_view.controls = []
        self._append_rows(minimum=built)
        if not rows:
            query = self.query.strip()
            if query and self.rows.get(self.segment):
                self.status.value = (f"No Library book matches “{query}”" if self.segment == "library"
                                     else f"Nothing matches “{query}”")
            else:
                empty = {"recent": "No translation workspaces yet.", "library": "The Library is empty.",
                         "chat": "No chat attachment workspaces.",
                         "browse": "Pick an EPUB, TXT or PDF file; its output folder is found automatically."}
                self.status.value = empty.get(self.segment, "")
        else:
            total = len(self.rows.get(self.segment, []))
            if len(rows) != total:
                self.status.value = f"{len(rows)} of {total} item{'s' if total != 1 else ''}"
            else:
                self.status.value = f"{len(rows)} item{'s' if len(rows) != 1 else ''}"
        self.browse_button.visible = self.segment == "browse"
        self.done_button.content = self._done_label()
        self.done_button.visible = self.multi
        self.done_button.disabled = not self.selected
        self.ctx.push(self.list_view, self.status, self.browse_button, self.done_button)
        self._start_covers()

    def _tile(self, target: tg.ToolTarget) -> ft.Control:
        reason = self.reason(target)
        checked = target.key in self.selected
        leading: ft.Control
        if self.multi:
            leading = ft.Checkbox(value=checked, disabled=reason is not None,
                                  on_change=lambda e, t=target: self.toggle(t))
        else:
            leading = ft.Icon(ft.Icons.RADIO_BUTTON_CHECKED if checked else ft.Icons.RADIO_BUTTON_UNCHECKED)
        tile = ft.ListTile(
            leading=leading,
            title=ft.Text(target.title, max_lines=2, overflow=ft.TextOverflow.ELLIPSIS),
            subtitle=ft.Text(_subtitle(target), max_lines=2, overflow=ft.TextOverflow.ELLIPSIS,
                             theme_style=ft.TextThemeStyle.BODY_SMALL),
            trailing=ReasonChip(reason=reason if len(reason) <= 28 else reason[:26] + "…", detail=reason)
            if reason else None,
            disabled=reason is not None,
            on_click=lambda e, t=target: self.toggle(t),
            min_height=tokens.SIZES["hit_target"],
            key=f"pick-{target.key}",
        )
        self.tiles[target.key] = tile
        return tile

    def _book_row(self, target: tg.ToolTarget) -> ft.Control:
        """The Library's list row for a Library book. Built once and kept: a search re-render reuses
        the same objects (Flet 1.0.3 freezes a control re-created under its key, and the cover arrives
        later through ``set_cover``)."""
        cached = self.book_cards.get(target.key)
        if cached is not None:
            return cached[1]
        from glossarion_mobile.ui.library.book_card import BookListRow

        model = self.models[target.key]
        if (target.key in self.selected) != bool(model.selected):
            model = replace(model, selected=target.key in self.selected)
        service = self.ctx.service
        covers = getattr(service, "covers", None)
        covers = covers if isinstance(covers, Mapping) else {}
        row = BookListRow(model, cover_src=covers.get(model.key), dark=bool(getattr(self.ctx, "dark", False)),
                          on_open=lambda m, t=target: self.toggle(t), show_more=False,
                          on_long_press=(lambda m, t=target: self.start_selection(t)) if self.long_press_selects
                          else None)
        control: ft.Control = row.control
        reason = self.reason(target)
        if reason is not None:
            # Not a disabled row: Flet passes ``disabled`` down, so the chip's InfoSheet could never
            # open (``reason_chip.unavailable_tile``). The row ignores taps and reads muted instead.
            row.control.on_click = None
            row.control.on_long_press = None
            row.control.ink = False
            row.control.opacity = 0.55
            row.control.expand = True
            control = ft.Row([row.control,
                              ReasonChip(reason=reason if len(reason) <= 28 else reason[:26] + "…", detail=reason)],
                             spacing=4, vertical_alignment=ft.CrossAxisAlignment.CENTER, key=f"pick-{target.key}")
        self.book_cards[target.key] = (row, control)
        if model.key not in covers:
            self._cover_rows[model.key] = row
            self._queue_cover(model.key, _book_of(target) or {})
        return control

    # ---- covers (book rows) ----------------------------------------------------------------------

    def _queue_cover(self, key: str, book: Mapping[str, Any]) -> None:
        queue = self._covers
        if queue is None:
            from glossarion_mobile.ui.library.book_card import CoverQueue

            queue = self._covers = CoverQueue(self.ctx.service, self._on_cover, spawn=self.ctx.spawn)
        queue.add(key, dict(book))

    def _start_covers(self) -> None:
        """Look the queued covers up one at a time (io), in the order the rows were first shown."""
        queue = self._covers
        if queue is not None:
            queue.kick()

    def _on_cover(self, key: str, path: Optional[str]) -> None:
        row = self._cover_rows.pop(key, None)
        if row is not None and path:
            row.set_cover(path)
            row.update()

    # ---- actions ---------------------------------------------------------------------------------

    def _on_segment(self, e: Any = None) -> None:
        selected = list(getattr(getattr(e, "control", None), "selected", None) or self.segmented.selected or [])
        self.set_segment(selected[0] if selected else self.segments[0])

    def set_segment(self, segment: str) -> None:
        self.segment = segment if segment in self.segments else self.segments[0]
        self.segmented.selected = [self.segment]
        self.ctx.push(self.segmented)
        self.render()

    def _on_query(self, e: Any = None) -> None:
        value = getattr(getattr(e, "control", None), "value", None)
        if value is None and self.search_field is not None:
            value = self.search_field.value
        self.set_query(str(value or ""))

    def set_query(self, text: str) -> None:
        """Filter the rows (every change, no debounce: the Library home search)."""
        self.query = str(text or "")
        if self.search_field is not None and self.search_field.value != self.query:
            self.search_field.value = self.query
            self.ctx.push(self.search_field)
        self.render()

    def _clear_query(self, e: Any = None) -> None:
        self.set_query("")

    def _on_header_action(self, e: Any = None) -> None:
        if not self.header_action:
            return
        callback = self.header_action[1]
        self.close()
        from glossarion_mobile.ui.components._handlers import call_handler

        call_handler(callback)

    def start_selection(self, target: tg.ToolTarget) -> None:
        """A long-press on a row (``long_press_selects``): multi-selection starts with it (or, already
        selecting, toggles it); "Use N" appears."""
        if self.reason(target) is not None:
            return
        self.multi = True
        self.toggle(target)

    def toggle(self, target: tg.ToolTarget) -> None:
        if self.reason(target) is not None:
            return
        if not self.multi:
            self.selected = {target.key: target}
            self.finish()
            return
        if target.key in self.selected:
            del self.selected[target.key]
        else:
            self.selected[target.key] = target
        self._sync_book_row(target)
        self.render(keep_rendered=True)

    def _sync_book_row(self, target: tg.ToolTarget) -> None:
        cached = self.book_cards.get(target.key)
        if cached is None:
            return
        row = cached[0]
        selected = target.key in self.selected
        if bool(row.model.selected) != selected:
            row.set_model(replace(row.model, selected=selected))
            row.update()

    def finish(self) -> list:
        self.result = list(self.selected.values())
        self.close()
        if self.on_done is not None:
            from glossarion_mobile.ui.components._handlers import call_handler

            call_handler(self.on_done, list(self.result))
        return self.result

    def _on_done(self, e: Any = None) -> list:
        return self.finish()

    async def browse(self) -> list:
        files = self.ctx.files
        if files is None:
            self.ctx.say("Picking files is not available in this session")
            return []
        try:
            picked = await files.pick_files(target="inbox", allowed_extensions=list(BROWSE_EXTENSIONS),
                                            allow_multiple=self.multi, dialog_title=self.title)
        except Exception as exc:
            self.ctx.say(f"Could not pick the file: {exc}")
            return []
        paths = [getattr(f, "path", None) for f in picked or ()]
        paths = [p for p in paths if p]
        if not paths:
            return []
        output_root = self.ctx.output_root

        find_folder = self.find_folder
        eligible = self.eligible

        def resolve() -> list:
            if not find_folder:
                found = [tg.target_for_source(p, candidates=lambda *a, **k: []) for p in paths]
            else:
                found = [tg.target_for_source(p, output_root=output_root or None) for p in paths]
            return [(target, _reason_of(eligible, target)) for target in found]

        resolved = await self.ctx.io(resolve)
        added = [target for target, _reason in resolved]
        for target, reason in resolved:
            self.reasons[target.key] = reason
            self.rows["browse"] = [t for t in self.rows["browse"] if t.key != target.key] + [target]
        if self.multi:
            for target in added:
                if self.reason(target) is None:
                    self.selected[target.key] = target
            self.render()
        elif added and self.reason(added[0]) is None:
            self.selected = {added[0].key: added[0]}
            self.render()
            self.finish()
        else:
            self.render()
        return added

    async def _on_browse(self, e: Any = None) -> list:
        return await self.browse()
