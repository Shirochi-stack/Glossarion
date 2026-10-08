"""Size classes and layout metrics (UI_SPEC §1.1). Pure Python, no Flet import.

| Class       | Width (dp) | Shell                                   | Chat column       |
|-------------|------------|-----------------------------------------|-------------------|
| phone       | < 600      | modal drawer, min(0.8*w, 360) wide      | full, 12 dp gutter|
| large_phone | 600-899    | modal drawer (max 360)                  | max 760, 16 dp    |
| tablet      | >= 900     | persistent 300 dp sidebar + SidePanel   | max 860, 24 dp    |
| wide        | >= 1200    | 320 dp sidebar, SidePanel can be pinned | max 860, 24 dp    |

The shell is rebuilt only when the class changes (the composer's output-mode style is
re-applied without a rebuild when only the chat column changes).

The composer's output-mode control sits in its action row (UI_SPEC §2.3 item 3), so it is
sized from the chat column (the composer's own width), not from the window: on tablets the
persistent sidebar takes 300-320 dp and the column is capped at 760 / 860 dp.
``output_row_style`` picks the control for a column width; ``action_row_width`` estimates
the action row (＋ · mode control · token hint · Send) so the rule, and the tests, can check
that the row never overflows.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from glossarion_mobile.ui.chat.output_modes import OUTPUT_MODES  # pure module, no Flet
from glossarion_mobile.ui.tokens import COMPACT_TEXT_SCALE, SIZES, gutter

__all__ = [
    "LARGE_PHONE_MIN",
    "TABLET_MIN",
    "WIDE_MIN",
    "Layout",
    "SizeClass",
    "action_row_width",
    "chat_column_width",
    "drawer_width",
    "layout_for",
    "mode_control_width",
    "output_row_style",
    "size_class_for",
]

LARGE_PHONE_MIN = 600
TABLET_MIN = 900
WIDE_MIN = 1200

# Composer action row (UI_SPEC §2.3 item 3). Widths in dp; they mirror ui/chat/composer.py
# and ui/chat/output_mode_row.py.
OUTPUT_TOGGLES_MIN = 600  # composer (chat column) width from which the six toggles sit inline
COMPOSER_INSET = 40  # composer margin 8 + padding 12, on both sides
ACTION_ROW_GAP = 4  # Row spacing between ＋ · mode control · pills · token hint · Send
PILL_ROOM = 112  # what the labelled toggles must leave for the option pills (one "Thinking off ×")
TOKEN_HINT_SAMPLE = "≈12.3k tok"  # the widest common token hint (direct_text_rules.token_hint)
MODE_EMOJI_SIZE = 16  # the emoji in the mode chip and in the icon toggles
CHIP_PADDING = (10, 4)  # mode chip visual: left / right padding around emoji + ▾
CHIP_ARROW = 16  # the ▾ icon (an Icon: it does not follow the text scale)
TOGGLE_GAP = 2  # spacing between the six toggles
FULL_TOGGLE_PADDING = 10  # labelled toggle: horizontal padding on each side
_LABEL_MEDIUM = (12, 0.5)  # font size, letter spacing (tokens.TYPE_SCALE["label_medium"])
_LABEL_SMALL = (11, 0.5)  # the token hint


class SizeClass(str, Enum):
    PHONE = "phone"
    LARGE_PHONE = "large_phone"
    TABLET = "tablet"
    WIDE = "wide"

    @property
    def persistent_sidebar(self) -> bool:
        return self in (SizeClass.TABLET, SizeClass.WIDE)


def _width(width: Optional[float]) -> float:
    try:
        return float(width or 0)
    except (TypeError, ValueError):
        return 0.0


def size_class_for(width: Optional[float]) -> SizeClass:
    w = _width(width)
    if w >= WIDE_MIN:
        return SizeClass.WIDE
    if w >= TABLET_MIN:
        return SizeClass.TABLET
    if w >= LARGE_PHONE_MIN:
        return SizeClass.LARGE_PHONE
    return SizeClass.PHONE


def drawer_width(width: Optional[float]) -> float:
    """Modal drawer width: min(0.8 * w, 360) (phone and large phone)."""
    w = _width(width)
    if w <= 0:
        return float(SIZES["drawer_max"])
    return float(min(0.8 * w, SIZES["drawer_max"]))


def _sidebar_width(size_class: SizeClass) -> int:
    return SIZES["sidebar_wide"] if size_class is SizeClass.WIDE else SIZES["sidebar"]


def _chat_max(size_class: SizeClass) -> Optional[int]:
    if size_class is SizeClass.PHONE:
        return None
    if size_class is SizeClass.LARGE_PHONE:
        return SIZES["large_phone_chat_max"]
    return SIZES["chat_max"]


def chat_column_width(width: Optional[float], *, side_panel: bool = False) -> float:
    """The chat column (the composer's width with its margins) for a window width: the window
    minus the persistent sidebar on tablets (and the 380 dp SidePanel while it is open,
    ``side_panel``), capped at the class's chat max."""
    w = _width(width)
    size_class = size_class_for(w)
    if size_class.persistent_sidebar:
        w -= _sidebar_width(size_class)
        if side_panel:
            w -= SIZES["side_panel"]
    chat_max = _chat_max(size_class)
    return float(max(0.0, min(w, chat_max) if chat_max is not None else w))


# ---- action-row estimate ---------------------------------------------------------------------


def _glyph_em(ch: str) -> float:
    """Rough advance of one character in em (Roboto; emoji are wider, joiners take no room)."""
    code = ord(ch)
    if code in (0xFE0F, 0x200D):
        return 0.0
    if code >= 0x1F000 or 0x2600 <= code <= 0x27BF:
        return 1.25
    if ch == " ":
        return 0.28
    if ch.isupper() or ch in "mw":
        return 0.68
    return 0.56


def _text_width(text: str, size: float, text_scale: float, letter_spacing: float = 0.0) -> float:
    return sum(_glyph_em(ch) * size * text_scale + letter_spacing for ch in text if _glyph_em(ch))


def mode_control_width(style: str, text_scale: float = 1.0) -> float:
    """Estimated width of the output-mode control inside the action row (``OutputModeRow`` inline).

    ``chip``: the active mode's emoji + ▾ in one 48 dp target; ``icons``: six 48 dp toggles;
    ``full``: six toggles that also show their label ("📝 Text"); no "Output:" label inline.
    """
    hit = SIZES["hit_target"]
    emoji = max(_text_width(mode.emoji, MODE_EMOJI_SIZE, text_scale) for mode in OUTPUT_MODES)
    if style == "chip":
        return max(float(hit), CHIP_PADDING[0] + emoji + CHIP_ARROW + CHIP_PADDING[1])
    gaps = TOGGLE_GAP * (len(OUTPUT_MODES) - 1)
    if style == "full":
        size, spacing = _LABEL_MEDIUM
        return gaps + sum(
            2 * FULL_TOGGLE_PADDING + _text_width(f"{mode.emoji} {mode.label}", size, text_scale, spacing)
            for mode in OUTPUT_MODES
        )
    return gaps + len(OUTPUT_MODES) * max(float(hit), emoji + 8)


def action_row_width(style: str, text_scale: float = 1.0, *, token_hint: str = TOKEN_HINT_SAMPLE) -> float:
    """Estimated width of the action row's fixed children for a mode-control style.

    ＋ (48) · mode control · [pills: flexible, 0 here] · token hint · Send (48), with the row's
    gaps. The token hint hides at >= 160% text scale (UI_SPEC §7.5), as the composer does.
    """
    hit = SIZES["hit_target"]
    widths = [float(hit), mode_control_width(style, text_scale), 0.0, float(hit)]
    if token_hint and text_scale < COMPACT_TEXT_SCALE:
        size, spacing = _LABEL_SMALL
        widths.insert(3, _text_width(token_hint, size, text_scale, spacing))
    return sum(widths) + ACTION_ROW_GAP * (len(widths) - 1)


def output_row_style(width: Optional[float], text_scale: float = 1.0) -> str:
    """How the composer's output-mode control renders for a chat-column width (§2.3 width rules).

    ``"chip"``: one compact chip (active emoji + ▾) that opens a menu of the six modes
    (< 600 dp, or text scale >= 160%);
    ``"icons"``: the six icon toggles inline in the action row (from 600 dp);
    ``"full"``: toggles that also show their text label, once they fit next to ＋, the token
    hint, Send and room for one option pill (about 810 dp at 100% text: wide screens and
    tablets from about 1110 dp).

    The ＋ sheet's row shows ``"chip"`` as the six icon toggles (``OutputModeRow(inline=False)``).
    """
    w = _width(width)
    if text_scale >= COMPACT_TEXT_SCALE or w < OUTPUT_TOGGLES_MIN:
        return "chip"
    room = w - COMPOSER_INSET
    if action_row_width("full", text_scale) + PILL_ROOM <= room:
        return "full"
    if action_row_width("icons", text_scale) <= room:
        return "icons"
    return "chip"


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
    chat_width: float = 0.0  # the chat column (``chat_column_width``); ``output_row`` is picked for it
    text_scale: float = 1.0  # the effective text scale the layout was computed for
    side_panel: bool = False  # the tablet SidePanel was open (``chat_width`` excludes it)

    @property
    def persistent_sidebar(self) -> bool:
        return self.size_class.persistent_sidebar

    @property
    def wide(self) -> bool:
        """Master-detail layouts (Settings, Book page, Manga; UI_SPEC §1.1)."""
        return self.size_class is SizeClass.WIDE


def layout_for(width: Optional[float], text_scale: float = 1.0, *, side_panel: bool = False) -> Layout:
    """The layout for a window width and the effective text scale (app × system,
    ``ui.text_scale``). ``side_panel``: the tablet SidePanel is open, so the chat column (and the
    composer's output-mode style) is narrower; the size class never depends on it."""
    w = _width(width)
    size_class = size_class_for(w)
    column = chat_column_width(w, side_panel=side_panel)
    return Layout(
        size_class=size_class,
        width=w,
        gutter=gutter(size_class.value),
        drawer_width=drawer_width(w),
        sidebar_width=_sidebar_width(size_class),
        chat_max_width=_chat_max(size_class),
        side_panel_width=SIZES["side_panel"],
        output_row=output_row_style(column, text_scale),
        compact_text=text_scale >= COMPACT_TEXT_SCALE,
        chat_width=column,
        text_scale=float(text_scale or 1.0),
        side_panel=bool(side_panel and size_class.persistent_sidebar),
    )
