"""Size classes and layout metrics (UI_SPEC §1.1). Pure Python, no Flet import.

| Class       | Width (dp) | Shell                                   | Chat column       |
|-------------|------------|-----------------------------------------|-------------------|
| phone       | < 600      | modal drawer, min(0.8*w, 360) wide      | full, 12 dp gutter|
| large_phone | 600-899    | modal drawer (max 360)                  | max 760, 16 dp    |
| tablet      | >= 900     | persistent 300 dp sidebar + SidePanel   | max 860, 24 dp    |
| wide        | >= 1200    | 320 dp sidebar, SidePanel can be pinned | max 860, 24 dp    |

The shell is rebuilt only when the class changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from glossarion_mobile.ui.tokens import COMPACT_TEXT_SCALE, SIZES, gutter

__all__ = [
    "LARGE_PHONE_MIN",
    "TABLET_MIN",
    "WIDE_MIN",
    "Layout",
    "SizeClass",
    "drawer_width",
    "layout_for",
    "output_row_style",
    "size_class_for",
]

LARGE_PHONE_MIN = 600
TABLET_MIN = 900
WIDE_MIN = 1200
OUTPUT_LABEL_MIN = 400  # below this the output-mode row shows icons only (§2.3)


class SizeClass(str, Enum):
    PHONE = "phone"
    LARGE_PHONE = "large_phone"
    TABLET = "tablet"
    WIDE = "wide"

    @property
    def persistent_sidebar(self) -> bool:
        return self in (SizeClass.TABLET, SizeClass.WIDE)


def size_class_for(width: Optional[float]) -> SizeClass:
    try:
        w = float(width or 0)
    except (TypeError, ValueError):
        w = 0.0
    if w >= WIDE_MIN:
        return SizeClass.WIDE
    if w >= TABLET_MIN:
        return SizeClass.TABLET
    if w >= LARGE_PHONE_MIN:
        return SizeClass.LARGE_PHONE
    return SizeClass.PHONE


def drawer_width(width: Optional[float]) -> float:
    """Modal drawer width: min(0.8 * w, 360) (phone and large phone)."""
    try:
        w = float(width or 0)
    except (TypeError, ValueError):
        w = 0.0
    if w <= 0:
        return float(SIZES["drawer_max"])
    return float(min(0.8 * w, SIZES["drawer_max"]))


def output_row_style(width: Optional[float], text_scale: float = 1.0) -> str:
    """How the composer's output-mode row renders (§2.3 width rules).

    ``"icons"``: six icon toggles, no label (< 400 dp, or text scale >= 160%);
    ``"label"``: "Output: Text" label + icon toggles (400-899 dp);
    ``"full"``: label + toggles that also show their text label (>= 900 dp).
    """
    try:
        w = float(width or 0)
    except (TypeError, ValueError):
        w = 0.0
    if text_scale >= COMPACT_TEXT_SCALE or w < OUTPUT_LABEL_MIN:
        return "icons"
    if w >= TABLET_MIN:
        return "full"
    return "label"


@dataclass(frozen=True)
class Layout:
    size_class: SizeClass
    width: float
    gutter: int
    drawer_width: float
    sidebar_width: int
    chat_max_width: Optional[int]
    side_panel_width: int
    output_row: str
    compact_text: bool

    @property
    def persistent_sidebar(self) -> bool:
        return self.size_class.persistent_sidebar


def layout_for(width: Optional[float], text_scale: float = 1.0) -> Layout:
    size_class = size_class_for(width)
    try:
        w = float(width or 0)
    except (TypeError, ValueError):
        w = 0.0
    if size_class is SizeClass.PHONE:
        chat_max = None
    elif size_class is SizeClass.LARGE_PHONE:
        chat_max = SIZES["large_phone_chat_max"]
    else:
        chat_max = SIZES["chat_max"]
    return Layout(
        size_class=size_class,
        width=w,
        gutter=gutter(size_class.value),
        drawer_width=drawer_width(w),
        sidebar_width=SIZES["sidebar_wide"] if size_class is SizeClass.WIDE else SIZES["sidebar"],
        chat_max_width=chat_max,
        side_panel_width=SIZES["side_panel"],
        output_row=output_row_style(w, text_scale),
        compact_text=text_scale >= COMPACT_TEXT_SCALE,
    )
