"""Full-screen composer (``/chat/<cid>/compose``, UI_SPEC §2.3; Direct Text #0/#4 composer input + drafts).

The composer's expand button (shown once the draft reaches ``composer.EXPAND_ICON_LINES`` lines)
opens this screen: one multiline field seeded from the composer text, the token count the composer's
hint uses (``direct_text_rules.count_tokens`` / ``token_hint``, debounced in a worker) and **Done**,
which writes the text back to the composer and the chat draft (``ChatStore.set_draft``) and pops.
Back / Close keeps the composer as it was. No send happens here: the composer's Send does that.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Optional

import flet as ft

from glossarion_mobile.ui.chat.direct_text_rules import TOKEN_HINT_MIN_CHARS, count_tokens, token_hint
from glossarion_mobile.ui.screens.base import Screen
from glossarion_mobile.ui.theme import HIT_TARGET

__all__ = ["ComposeScreen"]

log = logging.getLogger("glossarion.chat")

TOKEN_DEBOUNCE_SECONDS = 0.45


class ComposeScreen(Screen):
    title = "Composer"

    def __init__(
        self,
        match: Any,
        *,
        text: str = "",
        model: str = "",
        on_done: Optional[Callable[[str], Any]] = None,
        on_close: Optional[Callable[[], Any]] = None,
        run_io: Optional[Callable[..., Any]] = None,
        spawn: Optional[Callable[[Any], Any]] = None,
    ) -> None:
        super().__init__(match)
        self.original = text or ""
        self.model = model or ""
        self.on_done = on_done
        self.on_close = on_close
        self.run_io = run_io
        self.spawn = spawn
        self.done = False
        self._token_task: Any = None
        self.field = ft.TextField(
            value=self.original,
            multiline=True,
            min_lines=12,
            expand=True,
            autofocus=True,
            hint_text="Paste or type text to translate…",
            border=ft.OutlineInputBorder(),
            on_change=self._changed,
            key="compose-field",
        )
        self.token_text = ft.Text("", theme_style=ft.TextThemeStyle.LABEL_SMALL, color=ft.Colors.ON_SURFACE_VARIANT,
                                  key="compose-tokens")
        self.done_button = ft.FilledButton(content="Done", on_click=lambda e: self.finish(), key="compose-done")

    @property
    def text(self) -> str:
        return self.field.value or ""

    def build_body(self) -> ft.Control:
        self._schedule_tokens()
        return ft.Container(
            content=ft.Column([self.field, ft.Row([self.token_text], alignment=ft.MainAxisAlignment.END)],
                              expand=True, spacing=6),
            padding=12,
            expand=True,
        )

    def actions(self) -> list[ft.Control]:
        return [ft.IconButton(icon=ft.Icons.CLOSE, tooltip="Close without changes", on_click=lambda e: self.close(),
                              size_constraints=HIT_TARGET, key="compose-close"),
                self.done_button]

    # ---- token count -------------------------------------------------------------------------

    def _changed(self, e: Any = None) -> None:
        self._schedule_tokens()

    def _schedule_tokens(self) -> None:
        if len(self.text) < TOKEN_HINT_MIN_CHARS:
            if self.token_text.value:
                self.token_text.value = ""
                self._push(self.token_text)
            return
        if self.spawn is None or (self._token_task is not None and not self._token_task.done()):
            return
        try:
            self._token_task = self.spawn(self.update_tokens())
        except Exception:
            self._token_task = None

    async def update_tokens(self, *, debounce: float = TOKEN_DEBOUNCE_SECONDS) -> str:
        if debounce:
            await asyncio.sleep(debounce)
        text = self.text
        try:
            if self.run_io is not None:
                count = await self.run_io(count_tokens, text, self.model)
            else:
                count = count_tokens(text, self.model)
        except Exception:
            return self.token_text.value or ""
        self.token_text.value = token_hint(count) if len(self.text) >= TOKEN_HINT_MIN_CHARS else ""
        self._push(self.token_text)
        self._token_task = None
        if self.text != text:
            self._schedule_tokens()
        return self.token_text.value

    # ---- done / close -------------------------------------------------------------------------

    def finish(self) -> str:
        """Done: the text goes back to the composer (and the chat draft), then the screen pops."""
        self.done = True
        if self.on_done is not None:
            try:
                self.on_done(self.text)
            except Exception:
                log.exception("writing the composer text back failed")
        self.close()
        return self.text

    def close(self) -> None:
        if self.on_close is not None:
            self.on_close()

    def dispose(self) -> None:
        if self._token_task is not None and not self._token_task.done():
            self._token_task.cancel()

    @staticmethod
    def _push(control: Any) -> None:
        try:
            control.update()
        except Exception:
            pass
