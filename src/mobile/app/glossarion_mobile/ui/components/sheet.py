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

Closing (hotfix 3 issue 1, "the close menu button and slide to close is not working in some edge
cases, I have to backspace"): every sheet has "Close Menu ✕" (``close_dialog``: by identity, frozen
or in the tablet SidePanel, never raising) and slides closed (``draggable``) unless it pans / zooms
itself, is full screen, or opts out (``no_drag_dismiss``); ``install_close_buttons`` also installs
``dialogs.install_views_guard``, so a new top View never strands a sheet the client still shows.
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
    "no_drag_dismiss",
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
_NO_DRAG = "_glossarion_no_drag"
_DRAG_DECIDED = "_glossarion_drag_decided"
#: controls that own vertical drags themselves (a swipe-down would steal their pans / pinches): the image
#: viewer, and a web page (flet_webview builds ``WebViewWidget`` without gestureRecognizers, so a platform view
#: only gets the drags no Flutter recognizer claims: the sheet's drag would take every vertical swipe - the
#: DeepSeek sign-in page could not scroll, and a swipe closed the sheet mid-login)
_GESTURE_OWNERS = ("InteractiveViewer", "WebView")
#: the content walk that looks for them stops after this many controls (huge lists are plain rows)
_GESTURE_SCAN_LIMIT = 400


def _event_page(sheet: ft.BottomSheet, e: Any) -> Any:
    """The click's page, else the sheet's (None when neither is mounted: a panel copy, a gone session)."""
    for source in (e, sheet):
        if source is None:
            continue
        try:
            page = source.page  # Flet raises when the control is not on a page
        except Exception:
            continue
        if page is not None:
            return page
    return None


def close_sheet(sheet: ft.BottomSheet, e: Any = None) -> None:
    """Close ``sheet`` like its own Close / Cancel paths (``dialogs.close_dialog``: this sheet by identity,
    a frozen one too, the tablet SidePanel copy); its on_dismiss runs as for a swipe. Never raises."""
    from glossarion_mobile.ui.components.dialogs import close_dialog

    try:
        close_dialog(_event_page(sheet, e), sheet)
    except Exception:  # a click handler that raises shows Flet's crashed-session screen
        import logging

        logging.getLogger("glossarion.ui").exception("closing a sheet failed")


def close_strip(sheet: ft.BottomSheet, *, handle: bool) -> ft.Control:
    """The strip on top of a sheet: the drag-handle pill (``handle``) and "Close Menu ✕" at the right."""
    button = ft.TextButton(
        content=ft.Row([ft.Text(CLOSE_LABEL), ft.Icon(ft.Icons.CLOSE, size=18)], spacing=4, tight=True),
        tooltip=CLOSE_LABEL, on_click=lambda e: close_sheet(sheet, e), key="sheet-close",
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
    # the pill where Flutter's handle was, and on every sheet that slides closed (a swapped-in content too)
    handle = bool(getattr(sheet, "show_drag_handle", False) or getattr(sheet, "draggable", False))
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


def no_drag_dismiss(sheet: ft.BottomSheet) -> ft.BottomSheet:
    """Keep ``sheet`` from sliding closed (it keeps "Close Menu ✕" and Android back): content whose own
    vertical drags must not close it."""
    object.__setattr__(sheet, _NO_DRAG, True)
    sheet.draggable = False
    return sheet


def _holds_gestures(content: Any) -> bool:
    """True when ``content`` has a control that pans / zooms itself (``_GESTURE_OWNERS``)."""
    stack, seen = [content], 0
    while stack and seen < _GESTURE_SCAN_LIMIT:
        control = stack.pop()
        if control is None:
            continue
        seen += 1
        if type(control).__name__ in _GESTURE_OWNERS:
            return True
        for name in ("content", "title", "subtitle", "leading", "trailing"):
            child = getattr(control, name, None)
            if isinstance(child, ft.Control):
                stack.append(child)
        for name in ("controls", "actions"):
            items = getattr(control, name, None)
            if isinstance(items, list):
                stack.extend(item for item in items if isinstance(item, ft.Control))
    return False


def drag_dismissable(sheet: ft.BottomSheet) -> bool:
    """Whether ``sheet`` should slide closed: not full screen (an editor's typed text), not opted out,
    nothing in it that pans / zooms."""
    if getattr(sheet, _NO_DRAG, False) or getattr(sheet, "fullscreen", False):
        return False
    return not _holds_gestures(getattr(sheet, "content", None))


def _enable_drag(sheet: ft.BottomSheet) -> None:
    """Swipe-down closes the sheet (Flutter reads ``enableDrag`` when the sheet opens); decided once."""
    if getattr(sheet, _DRAG_DECIDED, False):
        return
    object.__setattr__(sheet, _DRAG_DECIDED, True)
    if not sheet.draggable and drag_dismissable(sheet):
        sheet.draggable = True  # the strip draws the drag pill for it


def install_close_buttons() -> None:
    """Every ``ft.BottomSheet`` gets the "Close Menu ✕" strip and slides closed, and a new top View waits
    for the sheets on the old one (``dialogs.install_views_guard``); once per process, at app start."""
    from glossarion_mobile.ui.components.dialogs import install_views_guard

    install_views_guard()
    original = ft.BottomSheet.before_update
    if getattr(original, _CLOSE_MARK, False):
        return

    def before_update(self: ft.BottomSheet) -> None:
        if getattr(self, "open", False):
            _enable_drag(self)
        with_close_button(self)
        original(self)

    setattr(before_update, _CLOSE_MARK, True)
    ft.BottomSheet.before_update = before_update


def bottom_sheet(content: ft.Control, *, key: Optional[str] = None, draggable: bool = True) -> ft.BottomSheet:
    """The app's sheet chrome: drag handle, slides closed (not when ``content`` pans / zooms itself or holds a
    web page, ``_GESTURE_OWNERS``), no 9/16 height cap, ``content`` as built by the caller."""
    return ft.BottomSheet(content=content, show_drag_handle=True, scrollable=True,
                          draggable=draggable and not _holds_gestures(content),
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
