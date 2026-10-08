"""Skeleton loading placeholders (UI_SPEC §5.2, §7.4).

Shimmer blocks shaped like the content that is loading: list rows (Glossaries, the Book
page's chapter rows, tools), cards (the Library grid), a hero (the Book page Overview) and
message bubbles (the chat while bodies load). Under reduce motion the blocks are a static
tint instead of a shimmer (``ui.motion.shimmer``, UI_SPEC §6.3).

The blocks are placeholders, not text: their fixed heights never clip anything, and the
whole skeleton carries one ``Semantics`` label ("Loading library…") so a screen reader
announces the state instead of a run of empty boxes.
"""

from __future__ import annotations

from typing import Any, Optional

import flet as ft

from glossarion_mobile.ui import motion, tokens

__all__ = ["Skeleton", "block", "bubble_skeleton", "card_skeleton", "hero_skeleton", "row_skeleton"]

_BLOCK = ft.Colors.SURFACE_CONTAINER_HIGHEST


def block(width: Optional[float] = None, height: float = 12, *, radius: float = tokens.RADII["badge"],
          expand: Any = None) -> ft.Container:
    """One grey block (a line of text, an avatar, a cover)."""
    return ft.Container(width=width, height=height, bgcolor=_BLOCK, border_radius=radius, expand=expand)


def row_skeleton(*, lines: int = 2, avatar: bool = True) -> ft.Control:
    """A list row: optional 32 dp avatar, a title line and ``lines - 1`` shorter lines."""
    text = [block(height=14, expand=True)] + [block(width=160, height=10) for _ in range(max(0, lines - 1))]
    parts: list[ft.Control] = []
    if avatar:
        parts.append(block(32, 32, radius=16))
    parts.append(ft.Column(text, spacing=6, expand=True, tight=True))
    return ft.Container(content=ft.Row(parts, spacing=12, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                        padding=ft.Padding.symmetric(horizontal=tokens.SPACING["list_item_h"],
                                                     vertical=tokens.SPACING["list_item_v"]))


def card_skeleton(width: float = 120, cover_height: float = 170) -> ft.Control:
    """A Library book card: cover, two title lines, an info line."""
    return ft.Column([
        block(width, cover_height, radius=tokens.RADII["cover"]),
        block(width, 12), block(width * 0.7, 12), block(width * 0.5, 10),
    ], spacing=6, tight=True, width=width)


def hero_skeleton(cover: tuple = (120, 180)) -> ft.Control:
    """The Book page Overview hero: cover beside title / author / progress lines."""
    return ft.Row([
        block(cover[0], cover[1], radius=tokens.RADII["cover"]),
        ft.Column([block(height=18, expand=False, width=200), block(140, 12), block(220, 8), block(100, 10)],
                  spacing=10, tight=True, expand=True),
    ], spacing=16, vertical_alignment=ft.CrossAxisAlignment.START)


def bubble_skeleton(*, mine: bool = False, width: float = 260) -> ft.Control:
    """A chat message while its body loads (right-aligned user bubble or full-width reply)."""
    lines = [block(height=12, width=width), block(height=12, width=width * 0.85), block(height=12, width=width * 0.6)]
    bubble = ft.Container(content=ft.Column(lines, spacing=6, tight=True), padding=12,
                          border_radius=tokens.RADII["bubble"], bgcolor=ft.Colors.SURFACE_CONTAINER_LOW)
    return ft.Row([bubble], alignment=ft.MainAxisAlignment.END if mine else ft.MainAxisAlignment.START)


class Skeleton:
    """A labelled shimmer over ``count`` placeholder items (``kind``: rows · cards · hero · bubbles).

    ``control`` is the widget to show while loading; build a fresh one each time it is shown
    (the shimmer or the static tint follows the reduce-motion switch at build time).
    """

    KINDS = ("rows", "cards", "hero", "bubbles")

    def __init__(self, kind: str = "rows", *, count: int = 6, label: str = "Loading…", key: Optional[str] = None,
                 card_width: float = 120, lines: int = 2, avatar: bool = True) -> None:
        self.kind = kind if kind in self.KINDS else "rows"
        self.count = max(1, int(count))
        self.label = label
        if self.kind == "cards":
            items: list[ft.Control] = [card_skeleton(card_width, card_width * 1.42) for _ in range(self.count)]
            body: ft.Control = ft.Row(items, wrap=True, spacing=12, run_spacing=16)
        elif self.kind == "hero":
            body = ft.Column([hero_skeleton(), *[row_skeleton(lines=lines, avatar=avatar) for _ in range(self.count)]],
                             spacing=12, tight=True)
        elif self.kind == "bubbles":
            body = ft.Column([bubble_skeleton(mine=index % 3 == 0) for index in range(self.count)], spacing=12,
                             tight=True)
        else:
            body = ft.Column([row_skeleton(lines=lines, avatar=avatar) for _ in range(self.count)], spacing=4,
                             tight=True)
        self.body = body
        self.control = ft.Semantics(
            label=label,
            live_region=True,
            content=ft.Container(content=motion.shimmer(body), padding=ft.Padding.all(tokens.SPACING["sm"])),
            key=key,
        )
