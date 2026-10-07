"""Bottom sheets whose rows can always be reached (UI_SPEC §5.2).

Flet 1.0.3 ``BottomSheet(scrollable=True)`` only maps to Flutter's ``isScrollControlled``: it
lifts the 9/16 height cap, but it never adds a scroll view (``bottom_sheet.dart`` passes the
content through), and a ``Column`` without ``scroll`` is a plain Flutter Column. Content taller
than the screen was clipped and its lower rows could not be reached (owner report: "chat
settings doesn't scroll down"). ``scroll_column`` gives a sheet a scrolling body, optionally with
an action row pinned under it; ``bottom_sheet`` builds the sheet around it.

Flet 1.0.3 lays a scrolling Column out at least as tall as its bounded parent
(``scrollable_control.dart`` ``_InnerConstraintsEnforcer``), so a sheet with a scrolling body
opens at full height. Sheets that are often short (``ActionSheet``, ``InfoSheet``) estimate their
height at 200 % text and keep a compact, non-scrolling body only when it surely fits
(``fits_compact``).

Flutter's modal sheet wraps the content in ``SafeArea(bottom: false)``, so with Android 15
edge-to-edge and 3-button navigation the last row sat under the navigation bar: ``sheet_frame``
adds the bottom system inset.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Sequence

import flet as ft

__all__ = [
    "SHEET_CHROME",
    "SHEET_PADDING",
    "WORST_TEXT_SCALE",
    "bottom_sheet",
    "fits_compact",
    "page_width",
    "scroll_column",
    "scroll_sheet",
    "sheet_frame",
    "text_height",
]

#: Default inner padding of a sheet body (the drag handle sits above it).
SHEET_PADDING = ft.Padding.only(left=16, right=16, bottom=16)
#: Largest system font scale Android offers (200 %); compact-sheet estimates assume it.
WORST_TEXT_SCALE = 2.0
#: Height a phone sheet never gets: status bar, drag handle and navigation bar (dp).
SHEET_CHROME = 48.0 + 48.0 + 48.0


def sheet_frame(content: ft.Control, *, padding: Any = None, key: Optional[str] = None) -> ft.SafeArea:
    """``content`` with the sheet padding, inset above the navigation bar (bottom system inset)."""
    return ft.SafeArea(
        content=ft.Container(content=content, padding=SHEET_PADDING if padding is None else padding),
        avoid_intrusions_top=False,
        key=key,
    )


def scroll_column(
    controls: Sequence[ft.Control],
    *,
    footer: Optional[Sequence[ft.Control]] = None,
    spacing: float = 10,
    scroll: bool = True,
) -> ft.Column:
    """A Column that scrolls inside a sheet; ``footer`` controls stay pinned below it (always visible).
    ``scroll=False`` builds the same column without scrolling (content that surely fits)."""
    column = ft.Column(list(controls), spacing=spacing, tight=True, scroll=ft.ScrollMode.AUTO if scroll else None)
    if not footer:
        return column
    column.expand = True
    column.tight = False
    return ft.Column([column, *footer], spacing=spacing)


def bottom_sheet(content: ft.Control, *, key: Optional[str] = None, draggable: bool = False) -> ft.BottomSheet:
    """The app's sheet chrome: drag handle, no 9/16 height cap, ``content`` as built by the caller."""
    return ft.BottomSheet(content=content, show_drag_handle=True, scrollable=True, draggable=draggable,
                          bgcolor=ft.Colors.SURFACE_CONTAINER_HIGH, key=key)


def scroll_sheet(
    title: Optional[str],
    controls: Sequence[ft.Control],
    *,
    actions: Optional[Sequence[ft.Control]] = None,
    spacing: float = 10,
    padding: Any = None,
    key: Optional[str] = None,
    scroll: bool = True,
) -> ft.BottomSheet:
    """A scrolling sheet (moved from ``glossary.common.sheet``): title, ``controls`` and an
    ``actions`` row at the end of the scroll."""
    body: list = []
    if title:
        body.append(ft.Text(title, theme_style=ft.TextThemeStyle.TITLE_MEDIUM, weight=ft.FontWeight.W_600))
    body.extend(controls)
    if actions:
        body.append(ft.Row(list(actions), alignment=ft.MainAxisAlignment.END, wrap=True, spacing=8))
    return bottom_sheet(sheet_frame(scroll_column(body, spacing=spacing, scroll=scroll), padding=padding), key=key)


# ---- compact-or-scroll estimate ---------------------------------------------------------------


def _page_size(page: Any) -> tuple:
    def number(name: str) -> float:
        try:
            return float(getattr(page, name, 0) or 0)
        except (TypeError, ValueError):
            return 0.0

    return number("width"), number("height")


def text_height(text: Any, width: float, font_size: float = 14.0) -> float:
    """Upper estimate (dp) of ``text`` wrapped to ``width`` at ``WORST_TEXT_SCALE``."""
    size = font_size * WORST_TEXT_SCALE
    per_line = max(4, int(max(width, 40.0) / (size * 0.62)))
    lines = sum(max(1, math.ceil(len(part) / per_line)) for part in str(text or "").split("\n"))
    return lines * size * 1.5


def fits_compact(page: Any, content_height: float) -> bool:
    """True when content estimated at ``content_height`` dp surely fits a sheet on ``page``.

    An unknown page size answers False (the caller then scrolls)."""
    _width, height = _page_size(page)
    return bool(height) and content_height + SHEET_CHROME <= height


def page_width(page: Any, default: float = 360.0) -> float:
    width, _height = _page_size(page)
    return width or default
