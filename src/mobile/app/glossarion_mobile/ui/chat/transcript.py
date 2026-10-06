"""Transcript (UI_SPEC §2.8, §2.13): the chat's message list.

``ListView(build_controls_on_demand=False, spacing=12, padding=12)`` over a
Python-side window (every rendered card is built, so each can be a ``ScrollKey``
target; ``auto_scroll`` stays False because ``scroll_to(scroll_key=)`` needs it
off). The window itself is computed by ``transcript_model`` (desktop rule); this
control only shows:

* the empty state with the exact §2.13 copy and its suggestion chips;
* the loader rows "↑ Scroll for earlier messages (N hidden)" /
  "Scroll for newer messages (N hidden) ↓" (tap = slide the window);
* the rendered cards, then the live tail (streaming cards, approval card, plan).

Following the tail: ``follow_tail`` is True while the user is within one screen of
the bottom (``on_scroll`` events); ``scroll_to_end`` jumps with ``offset=-1``.

Card slots: every saved card sits in a ``CardSlot`` that carries its ``ScrollKey`` and
stays the same Python object across re-renders (``slot(key, card)``). Flet 1.0.3 diffs a
*new* control that an old one matches by key as an immutable copy and marks it frozen, so
a card re-created under its old key could never be updated in place again (Copy ✓, audio
playback, a Vision run's OCR section, the jump highlight, live job progress). A slot
matched by identity is diffed in place; a new card swapped into its ``content`` is a plain
replacement, and a card passed again unchanged costs nothing.
"""

from __future__ import annotations

from dataclasses import field
from typing import Any, Callable, Optional, Sequence

import flet as ft

from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.components.empty_state import HALGAKOS_ASSET, EmptyState

__all__ = ["CardSlot", "EARLIER_TEMPLATE", "EMPTY_BODY", "EMPTY_TITLE", "LATER_TEMPLATE", "SUGGESTIONS", "Transcript"]

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
EARLIER_TEMPLATE = "↑ Scroll for earlier messages ({n} hidden)"
LATER_TEMPLATE = "Scroll for newer messages ({n} hidden) ↓"


class CardSlot(ft.Container):
    """The stable, ``ScrollKey``-keyed place of one transcript card (its ``content``)."""

    def __init__(self, key: str) -> None:
        super().__init__(key=ft.ScrollKey(key))
        self.slot_key = key

    @property
    def card(self) -> Any:
        return self.content


@ft.control
class Transcript(ft.ListView):
    on_suggestion: Optional[Callable[[str], Any]] = field(default=None, metadata={"skip": True})
    on_load_earlier: Optional[Callable[[], Any]] = field(default=None, metadata={"skip": True})
    on_load_later: Optional[Callable[[], Any]] = field(default=None, metadata={"skip": True})
    on_follow_change: Optional[Callable[[bool], Any]] = field(default=None, metadata={"skip": True})

    def init(self) -> None:
        super().init()
        self.build_controls_on_demand = False
        self.spacing = tokens.SPACING["md"]
        self.padding = ft.Padding.all(tokens.SPACING["md"])
        self.expand = True
        self.auto_scroll = False
        self.scroll_interval = 100
        self.messages: list[ft.Control] = []
        self.tail: list[ft.Control] = []
        self._slots: dict = {}  # slot key -> CardSlot of the latest set_messages
        self.hidden_before = 0
        self.hidden_after = 0
        self.follow_tail = True
        self._outer_on_scroll = self.on_scroll
        self.on_scroll = self._handle_scroll
        self.empty_state = EmptyState(
            image_src=HALGAKOS_ASSET,
            title=EMPTY_TITLE,
            body=EMPTY_BODY,
            suggestions=[(label, (lambda e, sid=sid: self._suggest(sid))) for sid, label in SUGGESTIONS],
            key="transcript-empty",
        )
        self.earlier_row = ft.TextButton(content="", on_click=lambda e: self._load(self.on_load_earlier), key="load-earlier")
        self.later_row = ft.TextButton(content="", on_click=lambda e: self._load(self.on_load_later), key="load-later")
        self.controls = [self.empty_state]

    @property
    def is_empty(self) -> bool:
        return not self.messages and not self.tail

    def _rebuild(self) -> None:
        controls: list[ft.Control] = []
        if self.hidden_before:
            self.earlier_row.content = EARLIER_TEMPLATE.format(n=self.hidden_before)
            controls.append(ft.Row([self.earlier_row], alignment=ft.MainAxisAlignment.CENTER))
        controls.extend(self.messages)
        if self.hidden_after:
            self.later_row.content = LATER_TEMPLATE.format(n=self.hidden_after)
            controls.append(ft.Row([self.later_row], alignment=ft.MainAxisAlignment.CENTER))
        controls.extend(self.tail)
        self.controls = controls if controls else [self.empty_state]

    def set_messages(self, controls: Sequence[ft.Control], *, hidden_before: int = 0, hidden_after: int = 0) -> None:
        self.messages = list(controls)
        self.hidden_before = max(0, int(hidden_before))
        self.hidden_after = max(0, int(hidden_after))
        # Only the slots shown now are kept: a card that leaves the window comes back in a new slot.
        shown = {c.slot_key for c in self.messages if isinstance(c, CardSlot)}
        self._slots = {key: slot for key, slot in self._slots.items() if key in shown}
        self._rebuild()

    def slot(self, key: str, card: ft.Control) -> CardSlot:
        """The slot keyed ``key`` (created on first use) holding ``card``."""
        key = str(key)
        slot = self._slots.get(key)
        if slot is None:
            slot = self._slots[key] = CardSlot(key)
        if slot.content is not card:
            slot.content = card
        return slot

    def slot_for(self, key: Any) -> Optional[CardSlot]:
        """The shown slot of ``key`` (a str or a ``ScrollKey``), if any."""
        value = getattr(key, "value", key)
        return self._slots.get(str(value)) if value is not None else None

    @property
    def cards(self) -> list:
        """The shown message controls with their slots unwrapped (cards, switchers, banners)."""
        return [c.card if isinstance(c, CardSlot) else c for c in self.messages]

    def set_tail(self, controls: Sequence[ft.Control]) -> None:
        self.tail = [c for c in controls if c is not None]
        self._rebuild()

    def add_message(self, control: ft.Control) -> None:
        self.set_messages([*self.messages, control], hidden_before=self.hidden_before, hidden_after=self.hidden_after)

    # ---- scrolling -------------------------------------------------------------------------

    def _handle_scroll(self, e: Any) -> None:
        pixels = float(getattr(e, "pixels", 0) or 0)
        max_extent = float(getattr(e, "max_scroll_extent", 0) or 0)
        viewport = float(getattr(e, "viewport_dimension", 0) or 600)
        follow = max_extent - pixels <= viewport
        if follow != self.follow_tail:
            self.follow_tail = follow
            if self.on_follow_change is not None:
                self.on_follow_change(follow)
        if self._outer_on_scroll is not None:
            self._outer_on_scroll(e)

    async def scroll_to_end(self, duration: int = 200, settle: float = 0.15) -> None:
        """Jump to the newest card once the client has laid out the latest update."""
        import asyncio

        if settle:
            await asyncio.sleep(settle)  # scroll_to before layout would stop at the old max extent
        try:
            await self.scroll_to(offset=-1, duration=duration)
        except Exception:
            pass

    def _load(self, handler: Optional[Callable[[], Any]]) -> None:
        if handler is not None:
            handler()

    def _suggest(self, suggestion_id: str) -> None:
        if self.on_suggestion is not None:
            self.on_suggestion(suggestion_id)
