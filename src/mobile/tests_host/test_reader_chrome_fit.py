"""The Reader's bottom bar fits every phone (owner report: on a 320 dp Android screen the ▶ next-chapter
button was cut off).

The old action row was ◀ · ☰ · Aa · "🌐 Translate" · ▶ in one ``Row`` with Flet's default 10 dp spacing:
four 48 dp targets, the labelled button (about 116 dp) and 40 dp of gaps in a bar with 8 dp padding
needed about 364 dp, so 320, 344 and 360 dp phones clipped ▶. Now ◀ and ▶ are fixed at the bar's edges
(the middle group expands between them, no spacing) and the Translate / Live view button turns
icon-only (``chrome.compact_actions``) where its label would not fit.

These tests build the real ReaderScreen on a stand-in page at each width (and text scale), lay the
action row out the way Flutter's ``Row`` does (fixed children first, the expanded middle group gets
what is left) with this file's own, deliberately generous text metrics, and check that every control
lies inside the bar and ◀ / ▶ sit at its edges.

Run from src/mobile (mobile venv):
    python -m pytest -p no:cacheprovider -W ignore tests_host/test_reader_chrome_fit.py
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = MOBILE_DIR / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

needs_flet = pytest.mark.skipif(importlib.util.find_spec("flet") is None, reason="flet not installed")

ROUTE = "/reader/ab12cd34ef56"
# (width, height, (left, top, right, bottom) insets): small phones, the folded Fold cover screen,
# common phones, the unfolded Fold, a landscape phone with a cutout, tablets (side panel open)
SCREENS = [
    ("320x640", 320, 640, (0, 24, 0, 48)),
    ("344x882 fold cover", 344, 882, (0, 32, 0, 24)),
    ("360x740", 360, 740, (0, 24, 0, 48)),
    ("374x842 fold cover", 374, 842, (0, 32, 0, 24)),
    ("412x915", 412, 915, (0, 52, 0, 48)),
    ("840x1000 fold open", 840, 1000, (0, 32, 0, 24)),
    ("740x360 landscape", 740, 360, (48, 24, 0, 0)),
]
SCALES = [1.0, 1.3, 2.0]
HIT = 48.0


def _glyph(ch: str) -> float:
    """A generous advance in em (wider than ``responsive._glyph_em``), so the check has its own margin."""
    code = ord(ch)
    if code in (0xFE0F, 0x200D):
        return 0.0
    if code >= 0x1F000 or 0x2600 <= code <= 0x27BF:
        return 1.3
    if ch == " ":
        return 0.3
    return 0.7 if ch.isupper() or ch in "mw" else 0.58


def _text_width(text: str, size: float, scale: float) -> float:
    return sum(_glyph(ch) * size * scale + 0.1 for ch in text if _glyph(ch))


def _control_width(control: Any, scale: float) -> float:
    import flet as ft

    if isinstance(control, ft.IconButton):
        c = control.size_constraints
        assert c.min_width == HIT and c.min_height == HIT, control.key  # thumb-sized targets
        return HIT
    if isinstance(control, ft.TextButton):
        content = control.content
        if isinstance(content, str):
            return max(64.0, 24 + _text_width(content, 14, scale))
        assert isinstance(content, ft.Icon), content  # icons do not follow the text scale
        return 64.0
    raise AssertionError(f"unexpected control in the action row: {control!r}")


def _screen(width: float, height: float, insets: tuple, scale: float = 1.0) -> Any:
    import flet as ft

    from glossarion_mobile.ui.reader.reader_view import ReaderDeps, ReaderScreen
    from glossarion_mobile.ui.router import parse_route

    left, top, right, bottom = insets
    page = types.SimpleNamespace(width=width, height=height, views=[], platform=None, web=False,
                                 media=types.SimpleNamespace(padding=ft.Padding(left=left, top=top, right=right,
                                                                                bottom=bottom)),
                                 show_dialog=lambda d: None, pop_dialog=lambda: None)
    state = types.SimpleNamespace(text_scale=types.SimpleNamespace(value=scale), system_text_scale=None)
    screen = ReaderScreen(parse_route(ROUTE), ReaderDeps(page=page, webview_ok=lambda: False,
                                                         extras={"state": state}))
    screen.build_view(ROUTE)
    return screen


def _layout(screen: Any, scale: float) -> dict:
    """x ranges of the action row's visible controls inside the bar (Flutter ``Row`` rules)."""
    from glossarion_mobile.ui.reader import chrome as ch

    chrome = screen.chrome
    row = chrome.action_row
    assert row.controls == [chrome.prev_button, chrome.middle_actions, chrome.next_button]
    assert row.spacing == 0 and chrome.middle_actions.spacing == 0 and chrome.middle_actions.expand
    left, _top, right, _bottom = screen._page_insets()
    width = screen._size()[0] - (ch_panel(screen))
    inner = width - left - right - 2 * ch.BAR_PADDING  # the bar's SafeArea keeps the side insets clear
    middle = [c for c in chrome.middle_actions.controls if c.visible is not False]
    widths = [_control_width(c, scale) for c in middle]
    room = inner - 2 * HIT
    boxes = {"reader-prev": (0.0, HIT), "reader-next": (inner - HIT, inner)}
    gap = (room - sum(widths)) / (len(widths) + 1)  # SPACE_EVENLY
    x = HIT + gap
    for control, w in zip(middle, widths):
        boxes[control.key] = (x, x + w)
        x += w + gap
    return {"inner": inner, "room": room, "used": sum(widths), "boxes": boxes}


def ch_panel(screen: Any) -> float:
    from glossarion_mobile.ui.reader.reader_view import TOC_PANEL_WIDTH

    return TOC_PANEL_WIDTH if screen.panel_open else 0.0


@needs_flet
@pytest.mark.parametrize("live", [False, True])
@pytest.mark.parametrize("scale", SCALES)
@pytest.mark.parametrize("name,width,height,insets", SCREENS, ids=[s[0] for s in SCREENS])
def test_every_bottom_control_fits_the_bar(name, width, height, insets, scale, live):
    screen = _screen(width, height, insets, scale)
    screen.chrome.set_translate(visible=True, live=live)
    lay = _layout(screen, scale)
    assert lay["used"] <= lay["room"], (name, scale, lay)
    for key, (x0, x1) in lay["boxes"].items():
        assert 0 <= x0 < x1 <= lay["inner"], (name, scale, key, lay)
    # ◀ and ▶ are always there, at the two edges, never overlapped by the middle group
    boxes = lay["boxes"]
    assert boxes["reader-prev"][0] == 0 and boxes["reader-next"][1] == lay["inner"]
    assert boxes["reader-chapters"][0] >= boxes["reader-prev"][1]
    assert boxes["reader-translate"][1] <= boxes["reader-next"][0]
    assert screen.chrome.prev_button.visible is not False and screen.chrome.next_button.visible is not False


@needs_flet
@pytest.mark.parametrize("name,width,height,insets", SCREENS, ids=[s[0] for s in SCREENS])
def test_labelled_translate_where_it_fits(name, width, height, insets):
    """At 100 % text the label shows from 360 dp up; 320 / 344 dp get the icon (tooltip says what it does)."""
    import flet as ft

    screen = _screen(width, height, insets)
    chrome = screen.chrome
    chrome.set_translate(visible=True, live=False)
    bar = screen._page_size()[0]
    if bar >= 360:
        assert not chrome.compact and chrome.translate_button.content == "🌐 Translate", name
    if bar <= 344:
        assert chrome.compact and isinstance(chrome.translate_button.content, ft.Icon), name
        assert chrome.translate_button.tooltip == "Translate this chapter"
    assert chrome.translate_button.key == "reader-translate"  # device flows tap this key at every width
    # the short mode labels under 400 dp
    label = chrome.segments["original"].label
    assert label.value == ("Orig" if bar < 400 else "Original"), name


@needs_flet
def test_the_row_refits_on_rotation_the_side_panel_and_live_view():
    import flet as ft

    screen = _screen(320, 640, (0, 24, 0, 48))
    chrome = screen.chrome
    chrome.set_translate(visible=True, live=True)
    assert chrome.compact and chrome.translate_button.content.icon == ft.Icons.SATELLITE_ALT
    assert chrome.translate_button.tooltip == "Show the live translation"
    screen.page.width, screen.page.height = 640, 320  # rotate
    screen._on_width()
    assert not chrome.compact and chrome.translate_button.content == "🛰️ Live view"
    chrome.set_translate(visible=True, live=False)
    assert chrome.translate_button.content == "🌐 Translate"
    # a tablet with the chapters panel open: the bar is the screen less the panel
    screen.page.width, screen.page.height = 1000, 800
    screen._on_width()
    screen.panel_open = True
    screen._apply_side_panel()
    assert screen._page_size()[0] == 1000 - 320 and not chrome.compact
    # back to the phone
    screen.page.width, screen.page.height = 320, 640
    screen._on_width()
    assert chrome.compact and isinstance(chrome.translate_button.content, ft.Icon)


@needs_flet
def test_the_compact_row_fits_even_a_280_dp_bar_at_any_text_scale():
    from glossarion_mobile.ui.reader import chrome as ch

    for scale in (1.0, 2.0, 3.0):
        assert ch.action_row_width(True, scale) + 2 * ch.BAR_PADDING <= 280
    assert ch.compact_actions(320) and not ch.compact_actions(412)
    assert ch.compact_actions(412, 2.0)  # a large text scale on a common phone: icon-only too
    assert not ch.compact_actions(0)  # no size known yet: keep the label


@needs_flet
def test_the_progress_line_cannot_push_the_page_count_off():
    import flet as ft

    screen = _screen(320, 640, (0, 24, 0, 48), 2.0)
    chrome = screen.chrome
    assert chrome.progress_text.expand and chrome.progress_text.overflow == ft.TextOverflow.ELLIPSIS
    assert chrome.progress_text.max_lines == 1 and chrome.page_text.max_lines == 1
