"""InfoSheet (UI_SPEC §5.2): title + body text in a bottom sheet (help, reasons)."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import flet as ft

__all__ = ["InfoSheet"]


class InfoSheet:
    def __init__(self, *, title: str, body: str = "", actions: Optional[Sequence[ft.Control]] = None) -> None:
        self.title = title
        self.body = body
        self._page: Any = None
        controls: list[ft.Control] = [
            ft.Text(title, theme_style=ft.TextThemeStyle.TITLE_LARGE, weight=ft.FontWeight.W_600),
        ]
        if body:
            controls.append(ft.Text(body, theme_style=ft.TextThemeStyle.BODY_MEDIUM, selectable=True))
        if actions:
            controls.append(ft.Row(list(actions), wrap=True, spacing=8, run_spacing=8))
        self.dialog = ft.BottomSheet(
            content=ft.Container(
                padding=ft.Padding.only(left=16, right=16, bottom=24),
                content=ft.Column(controls, tight=True, spacing=12),
            ),
            show_drag_handle=True,
            scrollable=True,
            bgcolor=ft.Colors.SURFACE_CONTAINER_HIGH,
        )

    def show(self, page: Any) -> None:
        self._page = page
        page.show_dialog(self.dialog)

    def close(self) -> None:
        if self._page is not None and getattr(self.dialog, "open", False):
            self._page.pop_dialog()
