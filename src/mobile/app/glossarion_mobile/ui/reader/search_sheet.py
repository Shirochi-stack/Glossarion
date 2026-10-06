"""ReaderSearchSheet (UI_SPEC §3.11 "Search", §5.8).

A ``BottomSheet`` with the field "Search across book…" and the matches as rows
(chapter title + excerpt of ±34 characters). Typing is debounced 650 ms (the
desktop ``_SEARCH_DEBOUNCE_MS``); the search itself is the shared
``reader_doc.search_chapters`` run on a worker thread by the Reader, which
delivers rows in batches of 120 (``add_batch``). Tapping a row closes the
sheet and the Reader jumps to the chapter and highlights that occurrence.
States: ``idle`` · ``searching`` · ``results`` · ``none``.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Mapping, Optional, Sequence

import flet as ft

from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.components._handlers import call_handler

__all__ = ["ReaderSearchSheet", "SEARCH_DEBOUNCE", "SEARCH_HINT"]

SEARCH_DEBOUNCE = 0.65
SEARCH_HINT = "Search across book…"
MAX_ROWS = 2000


class ReaderSearchSheet:
    def __init__(
        self,
        *,
        on_search: Callable[[int, str], Any],  # (search id, query): start a search; rows arrive via add_batch
        on_pick: Callable[[Mapping[str, Any]], Any],
        unavailable_reason: Optional[str] = None,
        debounce: float = SEARCH_DEBOUNCE,
        height: Optional[float] = None,
        initial_query: str = "",
    ) -> None:
        self.on_search = on_search
        self.on_pick = on_pick
        self.debounce = debounce
        self.search_id = 0
        self.query = ""
        self.state = "idle"
        self.rows: list[dict] = []
        self._timer: Optional[asyncio.TimerHandle] = None
        self._page: Any = None
        self.field = ft.TextField(
            hint_text=SEARCH_HINT,
            value=initial_query,
            autofocus=True,
            prefix_icon=ft.Icons.SEARCH,
            on_change=self._on_change,
            on_submit=lambda e: self.start(e.control.value),
            disabled=unavailable_reason is not None,
            dense=True,
            key="reader-search-field",
        )
        self.status = ft.Text(unavailable_reason or "", theme_style=ft.TextThemeStyle.LABEL_MEDIUM,
                              color=ft.Colors.ON_SURFACE_VARIANT, key="reader-search-status")
        self.progress = ft.ProgressBar(height=2, visible=False)
        self.results = ft.ListView(controls=[], expand=True, spacing=0, key="reader-search-results")
        body = ft.Container(
            padding=ft.Padding.only(left=16, right=16, bottom=8),
            content=ft.Column([self.field, self.progress, self.status, self.results], spacing=6, expand=True),
        )
        self.sheet = ft.BottomSheet(
            content=ft.Container(content=body, height=min(620.0, (height or 800) * 0.8)),
            show_drag_handle=True,
            scrollable=False,
            bgcolor=ft.Colors.SURFACE_CONTAINER_HIGH,
        )

    # ---- searching ---------------------------------------------------------------------------

    def _on_change(self, e: Any) -> None:
        text = str(e.control.value or "")
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self.start(text)
            return
        self._timer = loop.call_later(self.debounce, self.start, text)

    def start(self, query: Any) -> int:
        self._timer = None
        text = str(query or "").strip()
        self.search_id += 1
        self.query = text
        self.rows = []
        self.results.controls = []
        if not text:
            self.state = "idle"
            self.status.value = ""
            self.progress.visible = False
            self._push(self.results, self.status, self.progress)
            return self.search_id
        self.state = "searching"
        self.status.value = "Searching…"
        self.progress.visible = True
        self._push(self.results, self.status, self.progress)
        call_handler(self.on_search, self.search_id, text)
        return self.search_id

    def is_current(self, search_id: int) -> bool:
        return search_id == self.search_id

    def add_batch(self, search_id: int, rows: Sequence[Mapping[str, Any]], done: bool) -> None:
        """Rows from the worker (UI loop); stale searches are ignored."""
        if search_id != self.search_id:
            return
        room = MAX_ROWS - len(self.rows)
        fresh = [dict(r) for r in rows][: max(0, room)]
        self.rows.extend(fresh)
        self.results.controls.extend(self._row(r) for r in fresh)
        if done:
            self.progress.visible = False
            self.state = "results" if self.rows else "none"
            self.status.value = self.summary()
        else:
            self.status.value = f"Searching… {len(self.rows)}"
        self._push(self.results, self.status, self.progress)

    def fail(self, search_id: int, message: str) -> None:
        if search_id != self.search_id:
            return
        self.progress.visible = False
        self.state = "none"
        self.status.value = message
        self._push(self.status, self.progress)

    def summary(self) -> str:
        if not self.rows:
            return f"No matches for “{self.query}”"
        more = "+" if len(self.rows) >= MAX_ROWS else ""
        return f"{len(self.rows)}{more} match{'es' if len(self.rows) != 1 else ''}"

    def _row(self, row: Mapping[str, Any]) -> ft.Control:
        title = str(row.get("title") or f"Chapter {int(row.get('chapter_idx', 0)) + 1}")
        excerpt = str(row.get("excerpt") or "")
        return ft.ListTile(
            title=ft.Text(title, max_lines=1, overflow=ft.TextOverflow.ELLIPSIS,
                          theme_style=ft.TextThemeStyle.LABEL_LARGE),
            subtitle=ft.Text(excerpt, max_lines=2, overflow=ft.TextOverflow.ELLIPSIS,
                             theme_style=ft.TextThemeStyle.BODY_SMALL),
            dense=True,
            min_height=tokens.SIZES["row_two_line"],
            on_click=lambda e, r=dict(row): self._pick(r),
            key=f"hit-{row.get('global_occurrence', len(self.rows))}",
        )

    def _pick(self, row: dict) -> None:
        self.close()
        call_handler(self.on_pick, row)

    # ---- visibility --------------------------------------------------------------------------

    def show(self, page: Any) -> None:
        self._page = page
        page.show_dialog(self.sheet)

    def close(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        if self._page is not None and getattr(self.sheet, "open", False):
            try:
                self._page.pop_dialog()
            except Exception:
                pass

    @staticmethod
    def _push(*controls: Any) -> None:
        for control in controls:
            try:
                control.update()
            except Exception:
                pass
