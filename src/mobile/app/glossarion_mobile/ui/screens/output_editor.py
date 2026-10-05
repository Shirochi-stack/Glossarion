"""Output editor (``/chat/<cid>/m/<mid>/edit``, UI_SPEC §2.10 "Edit translation").

A full-screen editor for one saved assistant response. Saving goes through the
shared ``direct_text_store`` writer (the desktop ``_save_response_output_edit``):
the ``Chat Messages/NNNNNN-response.{md,txt,html,xhtml}`` copies are replaced
atomically (all rolled back if one write fails) and, for an attachment chapter
card, the real translated chapter file found through ``translation_progress.json``
is rewritten too. Editing is unavailable when the backing response file is missing
(desktop rule). Mobile edits the response source (Markdown/HTML) as text; the
desktop's rich-text round trip is not needed because the source is edited directly.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Callable, Optional

import flet as ft

from glossarion_mobile.ui.screens.base import Screen

__all__ = ["OutputEditorScreen", "edit_availability", "save_response_edit"]

log = logging.getLogger("glossarion.chat.edit")


def edit_availability(chats: Any, cid: Any, index: int) -> Optional[str]:
    """None when the response can be edited, else the reason (desktop ``_response_context_action_availability``)."""
    message = chats.message(cid, index)
    if not message or message[0] != "assistant":
        return "Only assistant responses can be edited"
    storage = message[6] if len(message) > 6 and isinstance(message[6], dict) else {}
    if str(message[1] or ""):
        return None
    reference = str(storage.get("content_path") or "")
    if not reference:
        return "The saved response file is missing or unreadable."
    path = chats.binding.resolve_reference(reference)
    if not os.path.isfile(path):
        return "The saved response file is missing or unreadable."
    return None


def save_response_edit(chats: Any, cid: Any, index: int, source: str) -> Any:
    """Blocking: the shared ``_save_response_output_edit`` for chat ``cid`` (then save the history)."""
    return chats.save_response_edit(cid, index, source)


class OutputEditorScreen(Screen):
    title = "Edit output"

    def __init__(
        self,
        match: Any,
        *,
        chats: Any,
        cid: str,
        index: int,
        run_io: Optional[Callable[..., Any]] = None,
        on_saved: Optional[Callable[[], Any]] = None,
        on_close: Optional[Callable[[], Any]] = None,
        notify: Optional[Callable[..., Any]] = None,
        mono: str = "monospace",
    ) -> None:
        super().__init__(match)
        self.chats = chats
        self.cid = str(cid)
        self.index = int(index)
        self.run_io = run_io
        self.on_saved = on_saved
        self.on_close = on_close
        self.notify = notify
        self.reason = edit_availability(chats, cid, index)
        self.original = ""
        self.dirty = False
        self.saved = False
        self.field = ft.TextField(
            multiline=True,
            min_lines=20,
            expand=True,
            text_style=ft.TextStyle(font_family=mono, size=13),
            border=ft.OutlineInputBorder(),
            on_change=self._changed,
            disabled=self.reason is not None,
        )
        self.status = ft.Text("", theme_style=ft.TextThemeStyle.LABEL_SMALL, color=ft.Colors.ON_SURFACE_VARIANT)
        self.error = ft.Text("", color=ft.Colors.ERROR, visible=False, selectable=True)
        self.save_button = ft.TextButton(content="Save", on_click=self._save, disabled=self.reason is not None)

    def actions(self) -> list:
        return [ft.TextButton(content="Cancel", on_click=lambda e: self.close()), self.save_button]

    def build_body(self) -> ft.Control:
        if self.reason is None:
            self.original = self.chats.message_text(self.cid, self.index, "content")
            self.field.value = self.original
            self.status.value = f"{len(self.original):,} characters"
        else:
            self.error.value = self.reason
            self.error.visible = True
        return ft.Container(
            padding=12,
            content=ft.Column([self.field, self.status, self.error], expand=True, spacing=6),
        )

    def _changed(self, e: Any = None) -> None:
        value = self.field.value or ""
        self.dirty = value != self.original
        self.status.value = f"{len(value):,} characters" + (" · edited" if self.dirty else "")
        try:
            self.status.update()
        except Exception:
            pass

    async def save(self) -> bool:
        if self.reason is not None:
            return False
        source = self.field.value or ""
        try:
            if self.run_io is not None:
                await self.run_io(save_response_edit, self.chats, self.cid, self.index, source)
            else:
                await asyncio.to_thread(save_response_edit, self.chats, self.cid, self.index, source)
        except Exception as exc:
            log.warning("saving the edited response failed: %s", exc)
            self.error.value = f"Could not save the edited output:\n\n{exc}"
            self.error.visible = True
            try:
                self.error.update()
            except Exception:
                pass
            return False
        self.saved = True
        self.original = source
        self.dirty = False
        if self.on_saved is not None:
            self.on_saved()
        if self.notify is not None:
            self.notify("Output saved")
        self.close()
        return True

    async def _save(self, e: Any = None) -> None:
        await self.save()

    def close(self) -> None:
        if self.on_close is not None:
            self.on_close()
