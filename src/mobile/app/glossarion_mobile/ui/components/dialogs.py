"""ConfirmDialog and snackbars (UI_SPEC §5.2).

``ConfirmDialog``: title, body (the verbatim desktop text when one exists), an
optional item list and 2 buttons; the destructive one uses the error colour.
While an async ``on_confirm`` runs, both buttons are disabled and a
ProgressRing shows (state ``running``). Dialogs open with ``page.show_dialog``
and close with ``page.pop_dialog`` (Flet 1.0.3).
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

import flet as ft

from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.components._handlers import await_handler, call_handler

__all__ = ["ConfirmDialog", "show_snackbar"]


class ConfirmDialog:
    def __init__(
        self,
        *,
        title: str,
        body: str = "",
        confirm_label: str = "OK",
        cancel_label: str = "Cancel",
        destructive: bool = False,
        items: Optional[Sequence[str]] = None,
        on_confirm: Optional[Callable[[], Any]] = None,
        on_cancel: Optional[Callable[[], Any]] = None,
    ) -> None:
        self.title = title
        self.on_confirm = on_confirm
        self.on_cancel = on_cancel
        self.state = "idle"
        self.result: Optional[bool] = None
        self._page: Any = None
        content: list[ft.Control] = []
        if body:
            content.append(ft.Text(body, theme_style=ft.TextThemeStyle.BODY_MEDIUM))
        for item in items or ():
            content.append(ft.Text(f"• {item}", theme_style=ft.TextThemeStyle.BODY_SMALL, selectable=True))
        self.progress = ft.ProgressRing(width=18, height=18, stroke_width=2, visible=False)
        content.append(ft.Row([self.progress], alignment=ft.MainAxisAlignment.CENTER))
        self.cancel_button = ft.TextButton(content=cancel_label, on_click=self._on_cancel)
        confirm_style = (
            ft.ButtonStyle(bgcolor=ft.Colors.ERROR, color=ft.Colors.ON_ERROR) if destructive else None
        )
        self.confirm_button = ft.FilledButton(content=confirm_label, on_click=self._on_confirm, style=confirm_style)
        self.dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text(title),
            content=ft.Container(
                width=tokens.SIZES["dialog_max"],
                content=ft.Column(content, tight=True, spacing=8, scroll=ft.ScrollMode.AUTO),
            ),
            actions=[self.cancel_button, self.confirm_button],
            actions_alignment=ft.MainAxisAlignment.END,
        )

    def show(self, page: Any) -> None:
        self._page = page
        page.show_dialog(self.dialog)

    def _close(self) -> None:
        if self._page is not None and getattr(self.dialog, "open", False):
            self._page.pop_dialog()

    def _set_running(self, running: bool) -> None:
        self.state = "running" if running else "idle"
        self.cancel_button.disabled = running
        self.confirm_button.disabled = running
        self.progress.visible = running
        try:
            self.dialog.update()
        except Exception:  # not mounted (tests) or already closed
            pass

    def _on_cancel(self, e: Any = None) -> None:
        if self.state == "running":
            return
        self.result = False
        self._close()
        call_handler(self.on_cancel)

    async def _on_confirm(self, e: Any = None) -> None:
        if self.state == "running":
            return
        self._set_running(True)
        try:
            await await_handler(self.on_confirm)
        finally:
            self._set_running(False)
        self.result = True
        self._close()


def show_snackbar(
    page: Any,
    message: str,
    *,
    action_label: Optional[str] = None,
    on_action: Optional[Callable[[], Any]] = None,
    duration_ms: int = 4000,
) -> ft.SnackBar:
    """Show a SnackBar (``page.show_dialog``); returns it for tests."""
    bar = ft.SnackBar(
        content=ft.Text(message),
        action=action_label,
        on_action=(lambda e: call_handler(on_action)) if on_action is not None else None,
        duration=duration_ms,
        behavior=ft.SnackBarBehavior.FLOATING,
        show_close_icon=action_label is None,
    )
    page.show_dialog(bar)
    return bar
