"""ConfirmDialog and snackbars (UI_SPEC §5.2).

``ConfirmDialog``: title, body (the verbatim desktop text when one exists), an
optional item list and 2 buttons; the destructive one uses the error colour.
While an async ``on_confirm`` runs, both buttons are disabled and a
ProgressRing shows (state ``running``); once it succeeds the dialog closes and
ignores further taps (state ``closed``). Dialogs open with ``page.show_dialog``
and close with ``close_dialog`` (Flet 1.0.3).
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

import flet as ft

from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.components import surface
from glossarion_mobile.ui.components._handlers import await_handler, call_handler
from glossarion_mobile.ui.components.sheet import fits_compact, page_width, text_height
from glossarion_mobile.ui.theme import HIT_TARGET

__all__ = ["ConfirmDialog", "close_dialog", "dialog_height", "fit_dialog_column", "show_snackbar"]


def close_dialog(page: Any, dialog: Any) -> bool:
    """Close ``dialog`` itself; returns True when it was open.

    ``page.pop_dialog()`` closes the most recently opened dialog that is still open. After a
    handler has shown a SnackBar (also a dialog control in Flet 1.0.3) that is the SnackBar, not
    the sheet or confirm dialog that should close: Add key left its sheet open with no feedback
    and the same key was added again on every tap. Turning ``open`` off closes exactly this
    dialog's route (Flet 1.0.3 ``bottom_sheet.dart`` closes the route it pushed, as does
    ``alert_dialog.dart``); the client then sends ``dismiss`` and the page drops it.

    A sheet a tablet shows in the SidePanel (``components.surface.present_sheet``) closes the
    panel instead, so the sheet's own Save / Cancel paths work on every size class.
    """
    if dialog is not None and surface.dismiss(page, dialog):
        return True
    if dialog is None or not getattr(dialog, "open", False):
        return False
    if hasattr(dialog, "_frozen"):
        # Flet freezes a dialog in a keyed diff (one shown again under the same key before the
        # first one's dismiss arrived); its own use_dialog hook strips the marker the same way so
        # ``open`` can be turned off. page.pop_dialog() raised here and the dialog stayed open.
        try:
            del dialog._frozen
        except AttributeError:
            pass
    try:
        dialog.open = False
    except RuntimeError:
        return False
    try:
        dialog.update()
    except Exception:  # not mounted on a page (tests) or the session is gone
        if page is not None:
            try:
                page.update()
            except Exception:
                pass
    return True


def dialog_height(width: float, title: str, body: str, items: Sequence[str] = ()) -> float:
    """Upper estimate (dp, 200 % text) of an AlertDialog on a ``width`` dp page: title, body text,
    bullet items and the button row."""
    width = max(120.0, min(float(tokens.SIZES["dialog_max"]), width - 80))  # dialog insets + padding
    return (text_height(title, width, 22) + text_height(body, width) + 8 * len(items)
            + sum(text_height(f"• {item}", width, 12) for item in items) + 140)


def fit_dialog_column(page: Any, column: ft.Column, title: str, body: str, items: Sequence[str] = ()) -> None:
    """Scroll a dialog's body Column only when it may not fit (owner #21).

    Flet 1.0.3 lays a scrolling Column out as tall as the space it may take, so a short confirmation
    ("Delete selected", "Retranslate selected") opened screen-high with empty space under its text. As the
    compact sheets do (``fits_compact``), it keeps its natural height unless it may not fit."""
    fits = fits_compact(page, dialog_height(page_width(page), title, body, items))
    column.scroll = None if fits else ft.ScrollMode.AUTO


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
        self.body = body
        self.items = [str(item) for item in items or ()]
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
        self.column = ft.Column(content, tight=True, spacing=8, scroll=ft.ScrollMode.AUTO)
        self.dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text(title),
            content=ft.Container(width=tokens.SIZES["dialog_max"], content=self.column),
            actions=[self.cancel_button, self.confirm_button],
            actions_alignment=ft.MainAxisAlignment.END,
        )

    def show(self, page: Any) -> None:
        self._page = page
        fit_dialog_column(page, self.column, self.title, self.body, self.items)
        if self.state == "closed":  # shown again
            self.state = "idle"
            self._set_running(False)
        page.show_dialog(self.dialog)

    def _close(self) -> None:
        close_dialog(self._page, self.dialog)

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
        if self.state in ("running", "closed"):
            return
        self.state = "closed"
        self.result = False
        self._close()
        call_handler(self.on_cancel)

    async def _on_confirm(self, e: Any = None) -> None:
        if self.state in ("running", "closed"):  # a second tap never runs the action again
            return
        self._set_running(True)
        try:
            await await_handler(self.on_confirm)
        except BaseException:
            self._set_running(False)
            raise
        # Close this dialog by identity: a snackbar the action showed (often with Undo) stays.
        self.state = "closed"
        self.progress.visible = False
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
    """Show a SnackBar (``page.show_dialog``); returns it for tests.

    A bar without an action gets its dismiss ✕ on the LEFT (owner #15): Flutter's ``show_close_icon``
    puts it at the trailing edge, right over the chat composer's Stop button, so a user closing the
    message stopped the job instead."""
    text = ft.Text(message)
    bar = ft.SnackBar(
        content=text,
        action=action_label,
        on_action=(lambda e: call_handler(on_action)) if on_action is not None else None,
        duration=duration_ms,
        behavior=ft.SnackBarBehavior.FLOATING,
        show_close_icon=False,
    )
    if action_label is None:
        text.expand = True
        bar.content = ft.Row(
            [
                ft.IconButton(
                    icon=ft.Icons.CLOSE, tooltip="Dismiss", icon_color=ft.Colors.ON_INVERSE_SURFACE,
                    size_constraints=HIT_TARGET, key="snackbar-close",
                    on_click=lambda e: close_dialog(page, bar),
                ),
                text,
            ],
            spacing=4,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )
    page.show_dialog(bar)
    return bar
