"""Transcript (UI_SPEC §2.8, §2.13): the chat's message list.

``ListView(build_controls_on_demand=False, spacing=12, padding=12)`` over a
Python-side window (every rendered card is built, so each can be a ``ScrollKey``
target; ``auto_scroll`` stays False because ``scroll_to(scroll_key=)`` needs it
off). The window itself is computed by ``transcript_model`` (desktop rule); this
control only shows:

* the empty state with the exact §2.13 copy and its suggestion chips;
* the loader rows "↑ Scroll for earlier messages (N hidden)" /
  "Scroll for newer messages (N hidden) ↓" (tap = slide the window; built once, never re-wrapped);
* the rendered cards, then the live tail (streaming cards, approval card, plan).

Following the tail: ``follow_tail`` is True while the user is within one screen of
the bottom (``on_scroll`` events); ``scroll_to_end`` jumps with ``offset=-1``.

Loading at the edges (UI_SPEC §2.8, desktop ``_update_history_window_for_scroll``): a user scroll
towards the start within ``EDGE_LOAD_PX`` of it (or a pull past the top) while cards are hidden
before the window calls ``on_load_earlier(True)``; the mirror at the end calls ``on_load_later(True)``.
One load at a time: the owner keeps the viewport on the card that was at the edge
(``keep_in_view``), which ends the load, so prepending at the top never cascades. Programmatic
scrolls (a jump, a restore) hold the edge loads (``hold_edge_loads``). A transcript shorter than
the screen sends no scroll events: the loader rows' tap always works.

Card slots: every saved card sits in a ``CardSlot`` that carries its ``ScrollKey`` and
stays the same Python object across re-renders (``slot(key, card)``). Flet 1.0.3 diffs a
*new* control that an old one matches by key as an immutable copy and marks it frozen, so
a card re-created under its old key could never be updated in place again (Copy ✓, audio
playback, a Vision run's OCR section, the jump highlight, live job progress). A slot
matched by identity is diffed in place; a new card swapped into its ``content`` is a plain
replacement, and a card passed again unchanged costs nothing.
"""

from __future__ import annotations

import time
from dataclasses import field
from typing import Any, Callable, Optional, Sequence

import flet as ft

from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.components.empty_state import HALGAKOS_ASSET, EmptyState
from glossarion_mobile.ui.components.pull_to_refresh import is_overscroll, is_top_pull

__all__ = ["CardSlot", "EARLIER_TEMPLATE", "EDGE_LOAD_PX", "EMPTY_BODY", "EMPTY_TITLE", "LATER_TEMPLATE",
           "SUGGESTIONS", "Transcript"]

EMPTY_TITLE = "What would you like to translate?"
EMPTY_BODY = (
    "Paste text into the composer below, or attach a supported file. "
    "Translations stream into this conversation as they are generated."
)
# (id, chip label) for the empty-chat suggestion chips (§2.13); "From Library" opens the in-chat
# Library picker (the book is attached here, never a trip to the Library screen)
SUGGESTIONS = (
    ("paste_text", "Paste text"),
    ("attach_book", "Attach a book"),
    ("from_library", "From Library"),
    ("manga_page", "Translate a manga page"),
)
EARLIER_TEMPLATE = "↑ Scroll for earlier messages ({n} hidden)"
LATER_TEMPLATE = "Scroll for newer messages ({n} hidden) ↓"
#: UI_SPEC §2.8: a user scroll this close to an edge loads the hidden cards beyond it
EDGE_LOAD_PX = 600.0
#: after a load or a programmatic scroll, edge loads wait this long (the client lays out first)
EDGE_HOLD_SECONDS = 0.35


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
        # the loader rows are built once: a render only changes their text (no remove + add per render)
        self.earlier_box = ft.Row([self.earlier_row], alignment=ft.MainAxisAlignment.CENTER)
        self.later_box = ft.Row([self.later_row], alignment=ft.MainAxisAlignment.CENTER)
        self._edge_busy = False  # an edge load is waiting for its viewport restore (``keep_in_view``)
        self._edge_hold_until = 0.0  # monotonic time before which scroll events load nothing
        self.edge_loads = 0  # edge loads started (diagnostics / tests)
        self.controls = [self.empty_state]

    @property
    def is_empty(self) -> bool:
        return not self.messages and not self.tail

    def _rebuild(self) -> None:
        controls: list[ft.Control] = []
        if self.hidden_before:
            self.earlier_row.content = EARLIER_TEMPLATE.format(n=self.hidden_before)
            controls.append(self.earlier_box)
        controls.extend(self.messages)
        if self.hidden_after:
            self.later_row.content = LATER_TEMPLATE.format(n=self.hidden_after)
            controls.append(self.later_box)
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
        self._maybe_load_edge(e)
        if self._outer_on_scroll is not None:
            self._outer_on_scroll(e)

    def _maybe_load_edge(self, e: Any) -> bool:
        """A user scroll near an edge with hidden cards beyond it loads one page of them (see the
        module docstring); True when a load started."""
        if self._edge_busy or time.monotonic() < self._edge_hold_until:
            return False
        kind = getattr(getattr(e, "event_type", ""), "value", getattr(e, "event_type", ""))
        delta = float(getattr(e, "scroll_delta", 0) or 0) if str(kind or "") == "update" else 0.0
        overscroll = float(getattr(e, "overscroll", 0) or 0) if is_overscroll(e) else 0.0
        pixels = float(getattr(e, "pixels", 0) or 0)
        minimum = float(getattr(e, "min_scroll_extent", 0) or 0)
        maximum = float(getattr(e, "max_scroll_extent", 0) or 0)
        if self.hidden_before and self.on_load_earlier is not None:
            if (delta < 0 and pixels - minimum <= EDGE_LOAD_PX) or is_top_pull(e):
                return self._edge_load(self.on_load_earlier)
        if self.hidden_after and self.on_load_later is not None:
            if (delta > 0 or overscroll > 0) and maximum - pixels <= EDGE_LOAD_PX:
                return self._edge_load(self.on_load_later)
        return False

    def _edge_load(self, handler: Callable[..., Any]) -> bool:
        self._edge_busy = True
        self.edge_loads += 1
        try:
            started = handler(True)
        except Exception:
            started = False
        if not started:  # nothing moved: no viewport restore will end this load
            self.release_edge_load()
        return bool(started)

    def release_edge_load(self, hold: float = EDGE_HOLD_SECONDS) -> None:
        """The edge load finished (its viewport is restored): scroll events may load again after ``hold``."""
        self._edge_busy = False
        self._edge_hold_until = time.monotonic() + max(0.0, float(hold))

    def hold_edge_loads(self, seconds: float = EDGE_HOLD_SECONDS) -> None:
        """No edge loads for ``seconds`` (a programmatic scroll is under way: a jump, a restore)."""
        self._edge_hold_until = max(self._edge_hold_until, time.monotonic() + max(0.0, float(seconds)))

    async def keep_in_view(self, key: Optional[str], settle: float = 0.15) -> None:
        """After an edge load re-rendered the window: put the card that was at the edge (slot ``key``)
        back at the top of the viewport (the desktop ``preserve_viewport``), then end the load."""
        import asyncio

        try:
            if key:
                if settle:
                    await asyncio.sleep(settle)  # the client lays the new cards out first
                await self.scroll_to(scroll_key=ft.ScrollKey(str(key)), duration=0)
        except Exception:
            pass
        finally:
            self.release_edge_load()

    async def scroll_to_end(self, duration: int = 200, settle: float = 0.15) -> None:
        """Jump to the newest card once the client has laid out the latest update."""
        import asyncio

        if settle:
            await asyncio.sleep(settle)  # scroll_to before layout would stop at the old max extent
        try:
            await self.scroll_to(offset=-1, duration=duration)
        except Exception:
            pass

    def _load(self, handler: Optional[Callable[..., Any]]) -> None:
        """A loader row's tap (the guaranteed path: a short transcript sends no scroll events)."""
        if handler is not None:
            handler()

    def _suggest(self, suggestion_id: str) -> None:
        if self.on_suggestion is not None:
            self.on_suggestion(suggestion_id)
