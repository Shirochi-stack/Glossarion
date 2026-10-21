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
    "CLOSE_LABEL",
    "bottom_sheet",
    "close_sheet",
    "install_close_buttons",
    "with_close_button",
    "fits_compact",
    "page_width",
    "scroll_column",
    "scroll_sheet",
    "sheet_body",
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


# ---- "Close Menu ✕" on every sheet -------------------------------------------------------------
# owner 2026-10-21: sliding a sheet down to close it can pull Android's notification shade down instead,
# so every BottomSheet gets a "Close Menu ✕" button at its top right. It sits in a strip that takes the
# place of Flutter's drag-handle area (the same pill in the middle when the sheet had one), so it never
# covers the sheet's own first row and the sheet keeps its height. ``install_close_buttons`` hooks
# ``BottomSheet.before_update``, which Flet runs for every sheet shown and on every update, so sheets that
# swap their content later keep the button.

CLOSE_LABEL = "Close Menu"
#: height of the top strip (Flutter's drag-handle area is 48 dp)
CLOSE_STRIP = 40.0
_CLOSE_MARK = "_glossarion_close_strip"


def close_sheet(sheet: ft.BottomSheet) -> None:
    """Close ``sheet`` as ``page.pop_dialog`` does (its on_dismiss runs as for a swipe)."""
    sheet.open = False
    try:
        sheet.update()
    except Exception:
        pass


def close_strip(sheet: ft.BottomSheet, *, handle: bool) -> ft.Control:
    """The strip on top of a sheet: the drag-handle pill (``handle``) and "Close Menu ✕" at the right."""
    button = ft.TextButton(
        content=ft.Row([ft.Text(CLOSE_LABEL), ft.Icon(ft.Icons.CLOSE, size=18)], spacing=4, tight=True),
        tooltip=CLOSE_LABEL, on_click=lambda e: close_sheet(sheet), key="sheet-close",
        style=ft.ButtonStyle(padding=ft.Padding.symmetric(horizontal=10, vertical=0)))
    controls: list = [ft.Container(content=button, right=4, top=0, bottom=0, alignment=ft.Alignment.CENTER_RIGHT)]
    if handle:
        pill = ft.Container(width=32, height=4, border_radius=2, bgcolor=ft.Colors.ON_SURFACE_VARIANT, opacity=0.4)
        controls.insert(0, ft.Container(content=pill, left=0, right=0, top=0, bottom=0, alignment=ft.Alignment.CENTER))
    return ft.Container(content=ft.Stack(controls), height=CLOSE_STRIP, left=0, right=0, top=0)


def with_close_button(sheet: ft.BottomSheet) -> ft.BottomSheet:
    """``sheet`` with the "Close Menu ✕" strip over its content (idempotent; a new content gets it too)."""
    content = getattr(sheet, "content", None)
    if content is None or getattr(content, _CLOSE_MARK, False):
        return sheet
    handle = bool(getattr(sheet, "show_drag_handle", False))
    body = ft.Container(content=content, padding=ft.Padding.only(top=CLOSE_STRIP),
                        expand=getattr(content, "expand", None))
    stack = ft.Stack([body, close_strip(sheet, handle=handle)], expand=getattr(content, "expand", None))
    object.__setattr__(stack, _CLOSE_MARK, True)
    sheet.show_drag_handle = False  # the strip draws the pill instead
    sheet.content = stack
    return sheet


def sheet_body(sheet: ft.BottomSheet) -> Any:
    """The sheet's own content, under the "Close Menu" strip once the sheet was shown."""
    content = getattr(sheet, "content", None)
    if getattr(content, _CLOSE_MARK, False):
        return content.controls[0].content
    return content


def install_close_buttons() -> None:
    """Give every ``ft.BottomSheet`` the "Close Menu ✕" strip (once per process, at app start)."""
    original = ft.BottomSheet.before_update
    if getattr(original, _CLOSE_MARK, False):
        return

    def before_update(self: ft.BottomSheet) -> None:
        with_close_button(self)
        original(self)

    setattr(before_update, _CLOSE_MARK, True)
    ft.BottomSheet.before_update = before_update


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
