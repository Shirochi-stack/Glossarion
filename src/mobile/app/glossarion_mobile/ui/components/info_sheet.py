"""InfoSheet (UI_SPEC §5.2): title + body text in a bottom sheet (help, reasons).

A short note keeps a compact sheet; a body that may not fit ("Show full translation" opens it with the
whole output, long settings help) is measured: the sheet is as tall as the text, and scrolls only past
the screen (``show``: ``components.sheet.compact_or_fitted``).
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import flet as ft

from glossarion_mobile.ui.components.dialogs import close_dialog
from glossarion_mobile.ui.components.sheet import compact_or_fitted, page_width, sheet_frame, text_height

__all__ = ["InfoSheet"]


class InfoSheet:
    def __init__(self, *, title: str, body: str = "", actions: Optional[Sequence[ft.Control]] = None,
                 markdown: bool = False) -> None:
        self.title = title
        self.body = body
        self._page: Any = None
        controls: list[ft.Control] = [
            ft.Text(title, theme_style=ft.TextThemeStyle.TITLE_LARGE, weight=ft.FontWeight.W_600),
        ]
        if body and markdown:  # rich help (the desktop provider information, the header help)
            controls.append(ft.Markdown(body, selectable=True, extension_set=ft.MarkdownExtensionSet.GITHUB_WEB))
        elif body:
            controls.append(ft.Text(body, theme_style=ft.TextThemeStyle.BODY_MEDIUM, selectable=True))
        if actions:
            controls.append(ft.Row(list(actions), wrap=True, spacing=8, run_spacing=8))
        self.actions = list(actions or ())
        self.column = ft.Column(controls, tight=True, spacing=12)
        self.frame = sheet_frame(self.column, padding=ft.Padding.only(left=16, right=16, bottom=24))
        self.dialog = ft.BottomSheet(
            content=self.frame,
            show_drag_handle=True,
            scrollable=True,
            draggable=True,  # slides closed (hotfix 3 issue 1)
            bgcolor=ft.Colors.SURFACE_CONTAINER_HIGH,
        )

    def estimated_height(self, width: float) -> float:
        """Upper estimate (dp, at 200 % text) of the title, body and action rows at ``width``."""
        inner = width - 32
        total = 24.0 + text_height(self.title, inner, 22)
        if self.body:
            total += 12 + text_height(self.body, inner, 14)
        if self.actions:
            total += 12 + 56.0 * len(self.actions)
        return total

    def show(self, page: Any) -> None:
        self._page = page
        # compact when it surely fits, else as tall as the text measures (never full screen with empty space)
        self.frame.content.content = compact_or_fitted(self.column, page, self.estimated_height(page_width(page)))
        page.show_dialog(self.dialog)

    def close(self) -> None:
        close_dialog(self._page, self.dialog)
