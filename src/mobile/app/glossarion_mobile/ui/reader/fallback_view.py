"""Native fallback page (UI_SPEC §3.11 "Native fallback").

Used where flet-webview has no platform view (Windows / Linux dev, web), for
the "Lightweight reader" switch and after a WebView failure: the chapter as a
scrolling ``ListView`` of ``Text`` / ``Image`` controls in the reader theme's
colours and typography (font size pt → px like the page, line spacing as the
text height). A ``GestureDetector`` maps taps (centre: chrome, edges: scroll a
screen, and at the end / start of the chapter :meth:`FallbackPage.page_by`
reports :data:`PAGE_EDGE` so the Reader opens the next / previous chapter, like
the page's ``edge`` event), pinch (font size steps, like the page's two-finger
events) and long-press on a paragraph (the selection actions for that
paragraph). The text is never OS-selectable: a ``SelectableText`` wins the
gesture arena over the tap zones and the paragraph long-press, and native
selection reports no text anyway.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Mapping, Optional, Sequence

import flet as ft

from glossarion_mobile.ui.components._handlers import call_handler
from glossarion_mobile.ui.reader import model as rm
from glossarion_mobile.ui.reader.blocks import Block

__all__ = ["PAGE_EDGE", "PAGE_SCROLLED", "FallbackPage", "font_family_for"]

log = logging.getLogger("glossarion.reader")

_HEADING_SCALE = {1: 1.6, 2: 1.4, 3: 1.25, 4: 1.12, 5: 1.05, 6: 1.0}
_PINCH_STEP = 0.15
#: ``page_by`` results: the list is at the end / start (turn the chapter), or it scrolled a screen.
PAGE_EDGE = "edge"
PAGE_SCROLLED = "scrolled"
PAGE_SCROLL_MS = 180  # the edge-tap scroll animation
#: Seconds after the animation for its scroll events to arrive. Flet sends ``on_scroll`` only
#: from scroll notifications, never on layout: a chapter shorter than the screen never sends one.
PAGE_SETTLE = 0.15
#: With nothing known about the list (no scroll event since the chapter was drawn) a page turn that
#: saw no event waits this much longer before it calls the chapter short: a busy phone's events late.
PAGE_LATE_GRACE = 0.2
EDGE_SLACK = 1.0  # logical pixels from the bound that count as "at the end"


def font_family_for(family: str) -> Optional[str]:
    text = (family or "").strip().lower()
    if text in ("", rm.EMBEDDED_CSS.lower(), "serif"):
        return "serif"
    if text == "sans":
        return None
    if text == "mono":
        return "monospace"
    return family


class FallbackPage:
    def __init__(
        self,
        *,
        on_tap_zone: Callable[[str], Any],  # "prev" | "next" | "centre"
        on_pinch: Callable[[int], Any],
        on_pinch_end: Callable[[], Any],
        on_paragraph: Callable[[str], Any],
        on_scroll: Callable[[float], Any],
    ) -> None:
        self.on_tap_zone = on_tap_zone
        self.on_pinch = on_pinch
        self.on_pinch_end = on_pinch_end
        self.on_paragraph = on_paragraph
        self.on_scroll = on_scroll
        self.blocks: list[Block] = []
        # the page area's size (the Reader passes the screen less the system bars / cutouts its
        # SafeArea keeps the page out of, and less the tablet chapters panel)
        self.width = 400.0
        self.height = 800.0
        self._pinch_base = 1.0
        # the list's scroll metrics from its last on_scroll event (None: none since the last render)
        self.pixels: Optional[float] = None
        self.min_extent = 0.0
        self.max_extent: Optional[float] = None
        self.viewport: Optional[float] = None  # its visible height (None: no event since the size changed)
        self.scroll_events = 0
        self.render_serial = 0  # counts render() calls (a page_by that saw a new chapter drawn says nothing)
        self.list_view = ft.ListView(controls=[], expand=True, spacing=0, on_scroll=self._on_scroll,
                                     key="reader-fallback-list")
        self.container = ft.Container(content=self.list_view, expand=True, bgcolor="#1e1e1e")
        self.control = ft.GestureDetector(
            content=self.container,
            expand=True,
            on_tap_up=self._on_tap,
            on_scale_start=self._on_scale_start,
            on_scale_update=self._on_scale_update,
            on_scale_end=lambda e: call_handler(self.on_pinch_end),
            key="reader-fallback",
        )

    def set_size(self, width: Optional[float], height: Optional[float]) -> None:
        if width:
            self.width = float(width)
        if height and float(height) != self.height:
            self.height = float(height)
            self.viewport = None  # rotated / resized: the last event's viewport is stale

    def page_step(self) -> float:
        """What an edge tap scrolls: 90 % of the list's visible height (its scroll events' viewport,
        else the page area's height), so a turn never skips a line the reader has not seen."""
        return 0.9 * max(1.0, min(self.viewport or self.height, self.height))

    def render(self, blocks: Sequence[Block], *, theme: Mapping[str, Any], settings: rm.ReaderSettings,
               images: Optional[Mapping[str, bytes]] = None, keep_offset: bool = False) -> None:
        """Draw ``blocks``. A new chapter starts at the top: its scroll metrics are unknown until the
        list sends a scroll event. ``keep_offset`` (the same chapter drawn again, e.g. an Aa change):
        the list keeps its offset, so ``pixels`` is kept; the extent is not (the layout changed)."""
        self.blocks = list(blocks)
        self.render_serial += 1
        self.pixels = self.pixels if keep_offset else None
        self.min_extent, self.max_extent = 0.0, None
        fg = str(theme.get("fg") or "#d4d4d4")
        heading = str(theme.get("heading") or fg)
        border = str(theme.get("border") or "#333333")
        px = round(settings.font_size * 96 / 72)
        family = font_family_for(settings.font_family)
        controls: list[ft.Control] = []
        for block in self.blocks:
            if block.kind == "heading":
                controls.append(ft.Container(
                    padding=ft.Padding.only(top=14, bottom=6),
                    content=ft.Text(block.text, size=round(px * _HEADING_SCALE.get(block.level or 2, 1.2)),
                                    weight=ft.FontWeight.W_600, color=heading, font_family=family, selectable=False),
                ))
            elif block.kind == "rule":
                controls.append(ft.Divider(color=border, height=24))
            elif block.kind == "image":
                data = (images or {}).get(block.src)
                if data:
                    controls.append(ft.Container(padding=ft.Padding.symmetric(vertical=8),
                                                 content=ft.Image(src=data, fit=ft.BoxFit.CONTAIN)))
                elif block.alt:
                    controls.append(ft.Text(f"🖼 {block.alt}", color=fg, italic=True, size=px - 2))
            else:
                prefix = "• " if block.kind == "item" else ""
                text = ft.Text(
                    spans=[ft.TextSpan(prefix + t if i == 0 else t,
                                       style=ft.TextStyle(weight=ft.FontWeight.W_600 if "bold" in s else None,
                                                          italic=True if "italic" in s else None))
                           for i, (t, s) in enumerate(block.spans)] if block.spans else None,
                    value="" if block.spans else prefix + block.text,
                    size=px,
                    color=fg,
                    font_family=family,
                    style=ft.TextStyle(height=rm.clamp_line_spacing(settings.line_spacing)),
                    italic=block.kind == "quote",
                    selectable=False,  # taps and the long-press below must reach their detectors
                )
                controls.append(ft.GestureDetector(
                    content=ft.Container(content=text, padding=ft.Padding.symmetric(vertical=4)),
                    on_long_press=lambda e, t=block.text: call_handler(self.on_paragraph, t),
                ))
        if not controls:
            controls.append(ft.Text("No readable content found for this chapter.", color=fg))
        self.list_view.controls = controls
        self.list_view.padding = ft.Padding.symmetric(horizontal=settings.margins, vertical=16)
        self.container.bgcolor = str(theme.get("bg") or "#1e1e1e")
        self._push(self.container)

    async def scroll_to_fraction(self, fraction: float) -> None:
        """The end of the chapter (``fraction`` >= 0.999) or its top (<= 0); other fractions keep
        the list where it is (the native page restores no in-chapter position)."""
        try:
            if fraction >= 0.999:
                await self.list_view.scroll_to(offset=-1)
            elif fraction <= 0.0:
                await self.list_view.scroll_to(offset=0)
        except Exception:
            pass

    def at_edge(self, direction: int) -> Optional[bool]:
        """Whether the list is at its end (``direction`` > 0) / start; None while the metrics are
        unknown (no scroll event since the chapter was drawn)."""
        if self.pixels is None or self.max_extent is None:
            return None
        if direction > 0:
            return self.pixels >= self.max_extent - EDGE_SLACK
        return self.pixels <= self.min_extent + EDGE_SLACK

    async def page_by(self, direction: int) -> str:
        """Scroll a screen (:meth:`page_step`) towards ``direction``: :data:`PAGE_EDGE` when the list is
        already at that end (known metrics), or when it does not move (one at a bound only overscrolls; with nothing
        known about the list, a chapter shorter than the screen sends no scroll event at all, waited
        for ``PAGE_LATE_GRACE`` longer); else :data:`PAGE_SCROLLED` (also when the list is known to
        be able to move but its events are late: a late turn would skip the rest of the chapter)."""
        step = 1 if direction > 0 else -1
        if self.at_edge(step):
            return PAGE_EDGE
        known = self.pixels is not None and self.max_extent is not None  # (and not at this edge)
        # unknown pixels: nothing scrolled since the chapter was drawn, so the list is at the top
        before = self.pixels if self.pixels is not None else self.min_extent
        events, serial = self.scroll_events, self.render_serial
        try:
            await self.list_view.scroll_to(delta=step * self.page_step(), duration=PAGE_SCROLL_MS)
        except Exception as exc:  # not on screen (yet): neither scrolled nor at an edge
            log.debug("fallback scroll failed: %s", exc)
            return PAGE_SCROLLED
        await asyncio.sleep(PAGE_SCROLL_MS / 1000 + PAGE_SETTLE)
        if self.render_serial != serial:
            return PAGE_SCROLLED  # another chapter was drawn meanwhile: never turn past it
        if self.scroll_events == events:
            if known:
                return PAGE_SCROLLED  # the list said it can still move: its events are only late
            await asyncio.sleep(PAGE_LATE_GRACE)
            if self.render_serial != serial:
                return PAGE_SCROLLED
            if self.scroll_events == events:
                return PAGE_EDGE  # a chapter shorter than the screen never sends one
        if self.pixels is not None and abs(self.pixels - before) < EDGE_SLACK:
            return PAGE_EDGE
        return PAGE_SCROLLED

    # ---- gestures ---------------------------------------------------------------------------

    def _on_tap(self, e: Any) -> None:
        x = None
        position = getattr(e, "local_position", None)
        if position is not None:
            x = getattr(position, "x", None)
        if x is None:
            x = getattr(e, "local_x", None)
        width = self.width or 1.0
        if x is None:
            zone = "centre"
        elif x < width / 3:
            zone = "prev"
        elif x > 2 * width / 3:
            zone = "next"
        else:
            zone = "centre"
        call_handler(self.on_tap_zone, zone)

    def _on_scale_start(self, e: Any) -> None:
        self._pinch_base = 1.0

    def _on_scale_update(self, e: Any) -> None:
        if int(getattr(e, "pointer_count", 2) or 0) < 2:
            return
        scale = getattr(e, "scale", None)
        if scale is None:
            scale = getattr(e, "horizontal_scale", 1.0)
        try:
            scale = float(scale)
        except (TypeError, ValueError):
            return
        ratio = scale / max(0.01, self._pinch_base)
        if ratio > 1 + _PINCH_STEP or ratio < 1 - _PINCH_STEP:
            self._pinch_base = scale
            call_handler(self.on_pinch, 1 if ratio > 1 else -1)

    def _on_scroll(self, e: Any) -> None:
        try:
            pixels = float(getattr(e, "pixels", 0.0) or 0.0)
            extent = float(getattr(e, "max_scroll_extent", 0.0) or 0.0)
            low = float(getattr(e, "min_scroll_extent", 0.0) or 0.0)
        except (TypeError, ValueError):
            return
        self.pixels, self.min_extent, self.max_extent = pixels, low, extent
        try:
            viewport = float(getattr(e, "viewport_dimension", 0.0) or 0.0)
        except (TypeError, ValueError):
            viewport = 0.0
        if viewport > 0:
            self.viewport = viewport
        self.scroll_events += 1
        fraction = max(0.0, min(1.0, pixels / extent)) if extent > 0 else 0.0
        call_handler(self.on_scroll, fraction)

    @staticmethod
    def _push(*controls: Any) -> None:
        for control in controls:
            try:
                control.update()
            except Exception:
                pass
