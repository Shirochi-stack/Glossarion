"""ChaptersDrawer (UI_SPEC §3.11 "Chapters drawer", §5.8).

The Reader View's ``end_drawer`` (``NavigationDrawer``): the chapter list (the
reader's chapter titles, or the book's native TOC from ``TOC.txt`` → sidecar
``toc.ncx`` → the EPUB's ``toc.ncx`` when "Native TOC" is on, desktop
``_rebuild_toc_sidebar``), each row with its translation-status icon from the
overlay (``reader_overlay``) and the current chapter highlighted, plus the
"Native TOC" and "Show special files" switches (``epub_reader_native_toc`` /
``epub_details_show_special_files``). Rows are built lazily.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

import flet as ft

from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.components._handlers import call_handler
from glossarion_mobile.ui.components.reason_chip import ReasonChip
from glossarion_mobile.ui.reader.model import TocRow, toc_rows
from glossarion_mobile.ui.theme import icon_data, status_color

__all__ = ["ChaptersDrawer", "TocRow", "toc_rows"]


_STATUS_KEYS = {"completed", "in_progress", "pending", "failed", "qa_failed", "merged", "skipped", "error",
                "not_translated", "refine_failed"}


class ChaptersDrawer:
    def __init__(
        self,
        *,
        on_open: Callable[[TocRow], Any],
        on_native_toc: Optional[Callable[[bool], Any]] = None,
        on_special_files: Optional[Callable[[bool], Any]] = None,
        dark: bool = False,
        width: float = 320,
    ) -> None:
        self.on_open = on_open
        self.on_native_toc = on_native_toc
        self.on_special_files = on_special_files
        self.dark = dark
        self.rows: list[TocRow] = []
        self.current = 0
        self.title = ft.Text("Chapters", theme_style=ft.TextThemeStyle.TITLE_MEDIUM, weight=ft.FontWeight.W_600)
        self.count = ft.Text("", theme_style=ft.TextThemeStyle.LABEL_MEDIUM, color=ft.Colors.ON_SURFACE_VARIANT)
        self.native_switch = ft.Switch(label="Native TOC", value=False, on_change=self._on_native, key="toc-native")
        self.native_reason = ReasonChip(reason="No native TOC",
                                        detail="No TOC.txt or toc.ncx entries could be matched to this book's chapters.")
        self.native_reason.visible = False
        self.special_switch = ft.Switch(label="Show special files", value=False, on_change=self._on_special,
                                        key="toc-special")
        self.list_view = ft.ListView(controls=[], expand=True, spacing=0, build_controls_on_demand=True,
                                     key="toc-list")
        self.box = ft.Container(
            height=640,
            content=ft.SafeArea(
                content=ft.Column(
                    [
                        ft.Container(
                            padding=ft.Padding.only(left=16, right=8, top=12),
                            content=ft.Column([ft.Row([self.title, self.count], spacing=8),
                                               ft.Row([self.native_switch, self.native_reason], wrap=True),
                                               self.special_switch], spacing=2, tight=True),
                        ),
                        self.list_view,
                    ],
                    spacing=4,
                    expand=True,
                ),
                expand=True,
            ),
        )
        self.drawer = ft.NavigationDrawer(controls=[self.box], bgcolor=ft.Colors.SURFACE_CONTAINER_LOW, width=width)

    # ---- content -------------------------------------------------------------------------

    def set_rows(self, rows: Sequence[TocRow], current: int, *, native_available: bool, native_on: bool,
                 show_special: bool) -> None:
        self.rows = list(rows)
        self.current = int(current)
        self.native_switch.value = bool(native_on and native_available)
        self.native_switch.disabled = not native_available
        self.native_reason.visible = not native_available
        self.special_switch.value = bool(show_special)
        self.count.value = f"{len(self.rows)}"
        self.list_view.controls = [self._row(r) for r in self.rows]
        self._push(self.box)

    def set_current(self, chapter: int) -> None:
        """Move the highlight: only the rows of the old and the new chapter change."""
        if chapter == self.current:
            return
        previous, self.current = self.current, int(chapter)
        changed = []
        for index, row in enumerate(self.rows):
            if row.chapter in (previous, self.current) and index < len(self.list_view.controls):
                self.list_view.controls[index] = self._row(row)
                changed.append(self.list_view.controls[index])
        if changed:
            self._push(self.list_view)

    def set_height(self, height: Optional[float]) -> None:
        if height:
            self.box.height = max(320.0, float(height))

    def current_row_index(self) -> int:
        for index, row in enumerate(self.rows):
            if row.chapter == self.current:
                return index
        return -1

    def _row(self, row: TocRow) -> ft.Control:
        status = row.status if row.status in _STATUS_KEYS else ""
        leading = None
        if status:
            color = status_color(status, self.dark)
            leading = ft.Icon(icon_data(tokens.status_style(status).icon), color=color, size=18,
                              tooltip=tokens.status_style(status).label)
        title = f"{row.number}. {row.title}" if row.number is not None and row.title else (row.title or f"{row.number}")
        selected = row.chapter == self.current
        return ft.ListTile(
            title=ft.Text(title, max_lines=2, overflow=ft.TextOverflow.ELLIPSIS,
                          weight=ft.FontWeight.W_600 if selected else None),
            leading=leading,
            selected=selected,
            dense=True,
            min_height=tokens.SIZES["hit_target"],
            on_click=lambda e, r=row: call_handler(self.on_open, r),
            key=f"toc-{row.chapter}-{row.fragment}",
        )

    # ---- events -----------------------------------------------------------------------------

    def _on_native(self, e: Any) -> None:
        call_handler(self.on_native_toc, bool(e.control.value))

    def _on_special(self, e: Any) -> None:
        call_handler(self.on_special_files, bool(e.control.value))

    @staticmethod
    def _push(*controls: Any) -> None:
        for control in controls:
            try:
                control.update()
            except Exception:
                pass
