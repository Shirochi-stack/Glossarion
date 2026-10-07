"""Shared plumbing of the Glossary screens: ``GlossaryContext`` and small helpers (UI_SPEC §4.1, §5).

``GlossaryContext`` extends the Library's ``LibraryContext`` (navigation, snackbars, io pool,
haptics, sheets) with the Glossary service, the Library service (book links, raw sources),
the settings context (schema tiles for the settings tabs) and the feature (cross-screen
actions: open the editor, submit jobs, load as manual glossary for the current chat).
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

import flet as ft

from glossarion_mobile.ui.components.dialogs import ConfirmDialog, close_dialog
from glossarion_mobile.ui.components.sheet import scroll_sheet
from glossarion_mobile.ui.library.common import LibraryContext

__all__ = ["GlossaryContext", "ask", "ago", "kind_icon", "kind_label"]

KIND_LABELS = {"book": "Book", "output": "Book (output)", "manual": "Manual", "unified": "Unified"}
KIND_ICONS = {"book": "MENU_BOOK", "output": "FOLDER_SPECIAL", "manual": "DESCRIPTION", "unified": "MERGE_TYPE"}


@dataclass
class GlossaryContext(LibraryContext):
    library: Any = None  # LibraryService (book rows, raw sources)
    settings: Any = None  # SettingsContext (schema tiles; may be None in tests)
    feature: Any = None  # GlossaryFeature

    def post_ui(self, fn: Callable[..., Any], *args: Any) -> None:
        """Run ``fn(*args)`` on the UI loop (progress callbacks arrive on the io pool)."""
        dispatcher = self.dispatcher
        if dispatcher is None or not getattr(dispatcher, "bound", False) or dispatcher.on_loop_thread():
            fn(*args)
            return
        dispatcher.post(fn, *args)


def kind_label(kind: str) -> str:
    return KIND_LABELS.get(kind, kind.title())


def kind_icon(kind: str) -> str:
    return KIND_ICONS.get(kind, "DESCRIPTION")


def ago(mtime: float, now: Optional[float] = None) -> str:
    """"just now" / "5 min ago" / "3 h ago" / "2 days ago" / a date."""
    if not mtime:
        return ""
    delta = max(0.0, (now if now is not None else time.time()) - float(mtime))
    if delta < 60:
        return "just now"
    if delta < 3600:
        return f"{int(delta // 60)} min ago"
    if delta < 86400:
        return f"{int(delta // 3600)} h ago"
    if delta < 86400 * 14:
        days = int(delta // 86400)
        return f"{days} day{'s' if days != 1 else ''} ago"
    return time.strftime("%Y-%m-%d", time.localtime(mtime))


def _scripted(ctx: Any, kind: str, title: str, body: str) -> Any:
    """Scripted answers (``ctx.extras['answers']``, host tests / the self-test); records what was asked."""
    extras = getattr(ctx, "extras", None)
    if not isinstance(extras, dict):
        return None
    extras.setdefault("asked", []).append((kind, title, body))
    answers = extras.get("answers")
    if isinstance(answers, list) and answers:
        return answers.pop(0)
    return None


async def ask(ctx: Any, *, title: str, body: str = "", confirm: str = "Yes", cancel: str = "No",
              destructive: bool = False, items: Any = None) -> bool:
    """A ConfirmDialog awaited as a bool (the desktop Yes/No message boxes). Without a page: True.

    A dialog closed any other way (the system back gesture closes Flutter dialog routes even
    when modal) answers No, so no caller waits forever."""
    scripted = _scripted(ctx, "ask", title, body)
    if scripted is not None:
        return bool(scripted)
    if getattr(ctx, "page", None) is None:
        return True
    loop = asyncio.get_running_loop()
    answer: asyncio.Future = loop.create_future()
    dialog = ConfirmDialog(title=title, body=body, confirm_label=confirm, cancel_label=cancel, destructive=destructive,
                           items=items, on_confirm=lambda: answer.done() or answer.set_result(True),
                           on_cancel=lambda: answer.done() or answer.set_result(False))
    # set before show_dialog (Flet wraps the handler present when the dialog opens)
    dialog.dialog.on_dismiss = lambda e: answer.done() or answer.set_result(False)
    ctx.show(dialog)
    ctx.extras["last_dialog"] = dialog
    return bool(await answer)


async def prompt_text(ctx: Any, *, title: str, label: str, value: str = "", ok: str = "OK") -> Optional[str]:
    """A one-field text dialog (the desktop QInputDialog); None when cancelled, dismissed or empty."""
    scripted = _scripted(ctx, "prompt", title, value)
    if scripted is not None:
        return str(scripted) if scripted is not False else None
    if getattr(ctx, "page", None) is None:
        return value or None
    loop = asyncio.get_running_loop()
    answer: asyncio.Future = loop.create_future()
    dismissed: list = []

    def on_dismiss(e: Any) -> None:  # back gesture / outside tap: the dialog is already gone
        dismissed.append(True)
        if not answer.done():
            answer.set_result(None)

    field = ft.TextField(label=label, value=value, autofocus=True,
                         on_submit=lambda e: answer.done() or answer.set_result(field.value))
    dialog = ft.AlertDialog(modal=True, title=ft.Text(title), content=field, on_dismiss=on_dismiss, actions=[
        ft.TextButton(content="Cancel", on_click=lambda e: answer.done() or answer.set_result(None)),
        ft.FilledButton(content=ok, on_click=lambda e: answer.done() or answer.set_result(field.value)),
    ])
    ctx.page.show_dialog(dialog)
    ctx.extras["last_dialog"] = dialog
    try:
        result = await answer
    finally:
        if not dismissed:  # by identity: never close a dialog or snackbar opened since
            close_dialog(ctx.page, dialog)
    text = str(result or "").strip()
    return text or None


def basename(path: str) -> str:
    return os.path.basename(str(path or ""))


def chip(text: str, *, color: Any = None, bgcolor: Any = None, key: Optional[str] = None,
         tooltip: Optional[str] = None) -> ft.Container:
    return ft.Container(
        content=ft.Text(text, size=11, weight=ft.FontWeight.W_600, color=color, no_wrap=True),
        bgcolor=bgcolor or ft.Colors.SECONDARY_CONTAINER,
        border_radius=6,
        padding=ft.Padding.symmetric(horizontal=6, vertical=1),
        key=key,
        tooltip=tooltip,
    )


def sheet(title: str, controls: list, *, actions: Optional[list] = None, key: Optional[str] = None,
          scroll: bool = True) -> ft.BottomSheet:
    """A scrollable bottom sheet: title, content and a bottom action row (primary actions in reach);
    the app-wide ``components.sheet.scroll_sheet``."""
    return scroll_sheet(title, controls, actions=actions, key=key, scroll=scroll,
                        padding=ft.Padding.only(left=16, right=16, bottom=20))


class SheetHost:
    """Opens / closes one ``ft.BottomSheet`` through the page (``page.show_dialog`` / ``close_dialog``)."""

    def __init__(self, ctx: Any) -> None:
        self.ctx = ctx
        self.dialog: Any = None

    def open(self, dialog: Any) -> Any:
        self.dialog = dialog
        page = getattr(self.ctx, "page", None)
        if page is not None:
            page.show_dialog(dialog)
        return dialog

    def close(self) -> None:
        page = getattr(self.ctx, "page", None)
        close_dialog(page, self.dialog)


def run_later(ctx: Any, fn: Callable[[], Any]) -> Any:
    result = fn()
    if asyncio.iscoroutine(result):
        return ctx.spawn(result)
    return result
