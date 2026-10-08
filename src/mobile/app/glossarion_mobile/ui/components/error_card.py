"""ErrorCard (UI_SPEC §5.2, §7.4): an error a screen could not get past.

Error icon · bold title · the error as mono ``ExcType: message`` (selectable; six lines, then
"Show more") · actions: **Retry** · **Copy error** · **View log** · an optional fix action
("Edit raw", "Sign in"). While a Retry runs the card shows a ProgressRing and its buttons are
disabled (state ``retrying``).

Status is never colour alone (UI_SPEC §6.1): the card always has the error icon and a title.
``error_text(exc)`` formats an exception the way the card shows it.
"""

from __future__ import annotations

from dataclasses import field
from typing import Any, Callable, Optional

import flet as ft

from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.components._handlers import await_handler, call_handler
from glossarion_mobile.ui.theme import mono_family

__all__ = ["ErrorCard", "MAX_LINES", "error_text"]

MAX_LINES = 6

Action = tuple[str, Callable[..., Any]]


def error_text(error: Any) -> str:
    """``ExcType: message`` for an exception, the text itself for a string."""
    if isinstance(error, BaseException):
        message = str(error).strip()
        name = type(error).__name__
        return f"{name}: {message}" if message else name
    return str(error or "").strip()


@ft.control
class ErrorCard(ft.Container):
    title: str = field(default="Something went wrong", metadata={"skip": True})
    message: Any = field(default="", metadata={"skip": True})  # str or exception
    on_retry: Optional[Callable[..., Any]] = field(default=None, metadata={"skip": True})
    on_copy: Optional[Callable[[str], Any]] = field(default=None, metadata={"skip": True})
    on_view_log: Optional[Callable[..., Any]] = field(default=None, metadata={"skip": True})
    fix: Optional[Action] = field(default=None, metadata={"skip": True})
    actions: Optional[list] = field(default=None, metadata={"skip": True})  # more (label, handler) buttons

    def init(self) -> None:
        super().init()
        self.state = "error"
        self.expanded = False
        self.text = error_text(self.message)
        self.icon = ft.Icon(ft.Icons.ERROR_OUTLINE, color=ft.Colors.ERROR, size=24)
        self.title_text = ft.Text(self.title, theme_style=ft.TextThemeStyle.TITLE_SMALL, weight=ft.FontWeight.W_600,
                                  expand=True)
        self.message_text = ft.Text(self.text, selectable=True, size=tokens.MONO_STYLE.size, max_lines=MAX_LINES,
                                    overflow=ft.TextOverflow.ELLIPSIS, font_family=mono_family(None),
                                    visible=bool(self.text))
        self.more_button = ft.TextButton(content="Show more", on_click=self._toggle, visible=self._long())
        self.progress = ft.ProgressRing(width=18, height=18, stroke_width=2, visible=False)
        self.buttons: list[ft.Control] = []
        if self.on_retry is not None:
            self.retry_button = ft.FilledTonalButton(content="Retry", icon=ft.Icons.REFRESH, on_click=self._retry)
            self.buttons.append(self.retry_button)
        if self.on_copy is not None and self.text:
            self.buttons.append(ft.TextButton(content="Copy error", icon=ft.Icons.CONTENT_COPY, on_click=self._copy))
        if self.on_view_log is not None:
            self.buttons.append(ft.TextButton(content="View log", icon=ft.Icons.TERMINAL,
                                              on_click=lambda e: call_handler(self.on_view_log)))
        if self.fix is not None:
            label, handler = self.fix
            self.buttons.append(ft.TextButton(content=label, on_click=lambda e: call_handler(handler)))
        for label, handler in list(self.actions or ()):
            self.buttons.append(ft.TextButton(content=label, on_click=lambda e, h=handler: call_handler(h)))
        self.content = ft.Column(
            [
                ft.Row([self.icon, self.title_text, self.progress], spacing=8,
                       vertical_alignment=ft.CrossAxisAlignment.CENTER),
                self.message_text,
                self.more_button,
                ft.Row(self.buttons, wrap=True, spacing=8, run_spacing=4, visible=bool(self.buttons)),
            ],
            spacing=6,
            tight=True,
        )
        self.bgcolor = ft.Colors.SURFACE_CONTAINER_LOW
        self.border_radius = tokens.RADII["card"]
        self.padding = ft.Padding.all(tokens.SPACING["card_padding"])

    def _long(self) -> bool:
        return self.text.count("\n") + 1 > MAX_LINES or len(self.text) > MAX_LINES * 60

    def _toggle(self, e: Any = None) -> None:
        self.expanded = not self.expanded
        self.message_text.max_lines = None if self.expanded else MAX_LINES
        self.more_button.content = "Show less" if self.expanded else "Show more"
        self._push()

    def _copy(self, e: Any = None) -> Any:
        result = call_handler(self.on_copy, self.text)
        return result

    def set_retrying(self, retrying: bool) -> None:
        self.state = "retrying" if retrying else "error"
        self.progress.visible = retrying
        for button in self.buttons:
            button.disabled = retrying
        self._push()

    async def _retry(self, e: Any = None) -> None:
        if self.state == "retrying":
            return
        self.set_retrying(True)
        try:
            await await_handler(self.on_retry)
        finally:
            self.set_retrying(False)

    def _push(self) -> None:
        try:
            self.update()
        except Exception:  # not mounted (tests) / the screen replaced it
            pass
