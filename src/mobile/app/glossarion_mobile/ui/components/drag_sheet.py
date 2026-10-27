"""Sheets that open short and grow with a swipe up (hotfix 5: the composer's mini model picker, the ＋ sheet).

The phone app's extension has ``DraggableSheet`` (Flutter's DraggableScrollableSheet,
``flet_glossarion_native.draggable_sheet``): the sheet opens at a fraction of the screen, a swipe up grows it
before its rows scroll, a swipe down shrinks it and closes it. ``draggable_sheet_class(page)`` is that control
when ``page``'s client has it, else None: the desktop and web clients of ``flet run`` (and a run with the
native stub forced) keep an ordinary scrolling sheet. ``height_fraction`` turns a height in dp into the
fraction of what a phone sheet gets.
"""

from __future__ import annotations

import os
from typing import Any

from glossarion_mobile.services import native as native_mod
from glossarion_mobile.ui.components.sheet import CLOSE_STRIP

__all__ = ["NAV_INSET", "SHEET_TOP", "draggable_sheet_class", "height_fraction"]

#: above a phone sheet's rows: the status bar and the "Close Menu" strip (dp)
SHEET_TOP = 24.0 + CLOSE_STRIP
#: under them: the navigation bar the sheet's last row stays above (3-button navigation, dp)
NAV_INSET = 48.0


def draggable_sheet_class(page: Any) -> Any:
    """The extension's ``DraggableSheet`` when ``page``'s client has it (the built phone app), else None."""
    if os.environ.get("GLOSSARION_FORCE_NATIVE_STUB") == "1":
        return None
    module = native_mod.load_extension()
    available = getattr(module, "native_widgets_available", None) if module is not None else None
    if available is None:
        return None
    try:
        if not available(page):
            return None
    except Exception:
        return None
    return getattr(module, "DraggableSheet", None)


def height_fraction(page: Any, height: float, *, low: float = 0.1, high: float = 1.0) -> float:
    """``height`` dp of rows (plus the navigation bar under them) as a fraction of what a sheet on ``page``
    gets, within [low, high] (0.5 when the page has no size yet)."""
    try:
        page_height = float(getattr(page, "height", 0) or 0)
    except (TypeError, ValueError):
        page_height = 0.0
    if page_height <= 0:
        return max(low, min(high, 0.5))
    available = max(1.0, page_height - SHEET_TOP)
    return round(max(low, min(high, (float(height) + NAV_INSET) / available)), 3)
