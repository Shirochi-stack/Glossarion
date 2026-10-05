"""SidePanel (UI_SPEC §1.1, §5.1): the 380 dp right panel on tablets.

Title row (title, pin on wide screens, ✕) over swappable content (chat
settings, job detail, compare, term sheet). Hidden until a surface opens it.
"""

from __future__ import annotations

from typing import Any, Optional

import flet as ft

from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.theme import HIT_TARGET

__all__ = ["SidePanel"]


class SidePanel:
    def __init__(self, width: int = tokens.SIZES["side_panel"]) -> None:
        self.pinned = False
        self.title_text = ft.Text("", theme_style=ft.TextThemeStyle.TITLE_MEDIUM, weight=ft.FontWeight.W_600, expand=True)
        self.pin_button = ft.IconButton(
            icon=ft.Icons.PUSH_PIN_OUTLINED, tooltip="Pin panel", on_click=self._toggle_pin, size_constraints=HIT_TARGET
        )
        self.close_button = ft.IconButton(icon=ft.Icons.CLOSE, tooltip="Close panel", on_click=self._close, size_constraints=HIT_TARGET)
        self.switcher = ft.AnimatedSwitcher(content=ft.Container(), duration=tokens.MOTION["state_ms"], expand=True)
        self.control = ft.Container(
            width=width,
            visible=False,
            bgcolor=ft.Colors.SURFACE_CONTAINER_LOW,
            padding=ft.Padding.only(left=12, right=4, top=4, bottom=8),
            content=ft.Column(
                [ft.Row([self.title_text, self.pin_button, self.close_button], spacing=0), self.switcher],
                spacing=4,
                expand=True,
            ),
        )

    @property
    def is_open(self) -> bool:
        return bool(self.control.visible)

    def open(self, title: str, content: ft.Control) -> None:
        self.title_text.value = title
        self.switcher.content = content
        self.control.visible = True
        self._push()

    def close(self) -> None:
        self.control.visible = False
        self.pinned = False
        self.pin_button.icon = ft.Icons.PUSH_PIN_OUTLINED
        self._push()

    def set_pin_available(self, available: bool) -> None:
        """Pinning (three panes) only on wide screens (>= 1200 dp)."""
        self.pin_button.visible = available

    def _toggle_pin(self, e: Any = None) -> None:
        self.pinned = not self.pinned
        self.pin_button.icon = ft.Icons.PUSH_PIN if self.pinned else ft.Icons.PUSH_PIN_OUTLINED
        self._push()

    def _close(self, e: Any = None) -> None:
        self.close()

    def _push(self) -> None:
        try:
            self.control.update()
        except Exception:
            pass

    @property
    def content(self) -> Optional[ft.Control]:
        return self.switcher.content
