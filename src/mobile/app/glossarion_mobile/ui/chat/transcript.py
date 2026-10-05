"""Transcript (UI_SPEC §2.8, §2.13): the chat's message list.

``ListView(build_controls_on_demand=False, spacing=12, padding=12)`` over a
Python-side window (every rendered card is built, so each can be a
``ScrollKey`` target; ``auto_scroll`` stays False because ``scroll_to(scroll_key=)``
needs it off). U1 renders only the empty state, with the exact §2.13 copy and
its suggestion chips; message cards, windowing and streaming arrive in U3.
"""

from __future__ import annotations

from dataclasses import field
from typing import Any, Callable, Optional, Sequence

import flet as ft

from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.components.empty_state import HALGAKOS_ASSET, EmptyState

__all__ = ["EMPTY_BODY", "EMPTY_TITLE", "SUGGESTIONS", "Transcript"]

EMPTY_TITLE = "What would you like to translate?"
EMPTY_BODY = (
    "Paste text into the composer below, or attach a supported file. "
    "Translations stream into this conversation as they are generated."
)
# (id, chip label) for the empty-chat suggestion chips (§2.13)
SUGGESTIONS = (
    ("paste_text", "Paste text"),
    ("attach_book", "Attach a book"),
    ("open_library", "Open Library"),
    ("manga_page", "Translate a manga page"),
)


@ft.control
class Transcript(ft.ListView):
    on_suggestion: Optional[Callable[[str], Any]] = field(default=None, metadata={"skip": True})

    def init(self) -> None:
        super().init()
        self.build_controls_on_demand = False
        self.spacing = tokens.SPACING["md"]
        self.padding = ft.Padding.all(tokens.SPACING["md"])
        self.expand = True
        self.auto_scroll = False
        self.scroll_interval = 100
        self.messages: list[ft.Control] = []
        self.empty_state = EmptyState(
            image_src=HALGAKOS_ASSET,
            title=EMPTY_TITLE,
            body=EMPTY_BODY,
            suggestions=[(label, (lambda e, sid=sid: self._suggest(sid))) for sid, label in SUGGESTIONS],
            key="transcript-empty",
        )
        self.controls = [self.empty_state]

    @property
    def is_empty(self) -> bool:
        return not self.messages

    def set_messages(self, controls: Sequence[ft.Control]) -> None:
        self.messages = list(controls)
        self.controls = list(self.messages) if self.messages else [self.empty_state]

    def add_message(self, control: ft.Control) -> None:
        self.set_messages([*self.messages, control])

    def _suggest(self, suggestion_id: str) -> None:
        if self.on_suggestion is not None:
            self.on_suggestion(suggestion_id)
