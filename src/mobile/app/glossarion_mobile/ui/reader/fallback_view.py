"""Native fallback page (UI_SPEC §3.11 "Native fallback").

Used where flet-webview has no platform view (Windows / Linux dev, web), for
the "Lightweight reader" switch and after a WebView failure: the chapter as a
scrolling ``ListView`` of ``Text`` / ``Image`` controls in the reader theme's
colours and typography (font size pt → px like the page, line spacing as the
text height). A ``GestureDetector`` maps taps (centre: chrome, edges: scroll a
screen), pinch (font size steps, like the page's two-finger events) and
long-press on a paragraph (the selection actions for that paragraph, since
native text selection reports no text).
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Optional, Sequence

import flet as ft

from glossarion_mobile.ui.components._handlers import call_handler
from glossarion_mobile.ui.reader import model as rm
from glossarion_mobile.ui.reader.blocks import Block

__all__ = ["FallbackPage", "font_family_for"]

_HEADING_SCALE = {1: 1.6, 2: 1.4, 3: 1.25, 4: 1.12, 5: 1.05, 6: 1.0}
_PINCH_STEP = 0.15


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
        self.width = 400.0
        self.height = 800.0
        self._pinch_base = 1.0
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
        if height:
            self.height = float(height)

    def render(self, blocks: Sequence[Block], *, theme: Mapping[str, Any], settings: rm.ReaderSettings,
               images: Optional[Mapping[str, bytes]] = None) -> None:
        self.blocks = list(blocks)
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
                                    weight=ft.FontWeight.W_600, color=heading, font_family=family, selectable=True),
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
                    selectable=True,
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
        try:
            if fraction >= 0.999:
                await self.list_view.scroll_to(offset=-1)
        except Exception:
            pass

    async def page_by(self, direction: int) -> None:
        try:
            await self.list_view.scroll_to(delta=direction * self.height * 0.9, duration=180)
        except Exception:
            pass

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
        except (TypeError, ValueError):
            return
        fraction = max(0.0, min(1.0, pixels / extent)) if extent > 0 else 0.0
        call_handler(self.on_scroll, fraction)

    @staticmethod
    def _push(*controls: Any) -> None:
        for control in controls:
            try:
                control.update()
            except Exception:
                pass
