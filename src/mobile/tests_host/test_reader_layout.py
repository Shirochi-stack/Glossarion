"""Owner's device report #10 on the U8 APK (2026-10-08): "the epub reader text clips out at the bottom".

Two defects stacked up (diagnosis wf_89d52c08-c74, skeptic-checked):

1. The APK targets SDK 36, so Android 15+ runs the app edge to edge. Every ordinary screen sits in a
   ``SafeArea``, but the Reader builds its own full-screen View and put the page (the WebView, or the
   native page) straight into its Stack: the page lay under the status bar and the navigation bar.
2. The mobile paged page shell (``reader_doc._MOBILE_PAGED_CSS``) put the safe-area padding on both
   ``html`` and ``body`` (the top inset counted twice) and never shrank ``#columns`` (the desktop
   ``_setupColumns`` keeps it ``innerHeight - 36`` tall), so with any reported inset the column box ran
   past the bottom of the screen.

The fix (both mobile-only; the desktop page is byte-identical, ``tests/test_reader_doc.py``): the page
layer sits in a ``SafeArea`` while the chrome bars stay edge to edge, and the mobile CSS adds the insets
once (on ``body``) and gives ``#columns`` the height that is left. The native page, the selection chips
and the tablet panel follow the page area.

This file proves it:

* the Reader's View: only the page is in a SafeArea (the chrome bars keep their own); in the real app
  shell too (the shell wraps the body with its JobStrip footer);
* the page area's size (insets, tablet chapters panel), the native page's edge-tap step (90 % of what is
  visible, never of the whole screen) and the selection chips' placement over the inset page;
* the theme background reaches the strips beside the page;
* the shell CSS: insets once, ``#columns`` sized from what is left, desktop page unchanged;
* the owner's complaint, measured the way the diagnosis did: real Reader pages (``DocumentBuilder`` +
  ``ReaderServer``, the page the phone loads) in headless Chrome at 412x915 and 360x800, with the page
  in its SafeArea (3-button and gesture navigation), WebViews that report insets anyway (spurious
  system-bar or cutout insets) and a full-screen page that reports the bars; several font sizes, line
  spacings and both families; paged and scroll layouts: no text line past the bottom of the page, none
  under a bar, the column box exactly where the insets leave room. ``GLOSSARION_READER_LAYOUT_FULL=1``
  runs the diagnosis' full typography matrix.

Run from src/mobile (mobile venv):
    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_reader_layout.py
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import os
import struct
import sys
import types
import zlib
from pathlib import Path
from typing import Any, Optional

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = MOBILE_DIR / "app"
SRC_DIR = MOBILE_DIR.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

_SPEC = importlib.util.spec_from_file_location("_glossarion_devfix1_helpers_layout",
                                               Path(__file__).with_name("test_devfix_issue1.py"))
D1 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(D1)

phone = D1.phone  # the real app on the fake Flet session as Android (isolated storage), from devfix issue 1


def _has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


needs_flet = pytest.mark.skipif(not _has("flet"), reason="flet not installed")
needs_cores = pytest.mark.skipif(not all(_has(m) for m in ("reader_doc", "library_core", "ebooklib")),
                                 reason="the shared reader cores are not importable here")

SCREEN = (412.0, 915.0)
INSETS = (0.0, 52.0, 0.0, 48.0)  # status bar (with a cutout) and a 3-button navigation bar
ROUTE = "/reader/ab12cd34ef56"


def _md5(path: Path) -> Optional[str]:
    return hashlib.md5(path.read_bytes()).hexdigest() if path.is_file() else None


# =====================================================================================
# The Reader's layers (Flet, no book)
# =====================================================================================


def _bare_screen(size: tuple = SCREEN, insets: tuple = INSETS) -> Any:
    """A ReaderScreen on a stand-in page (no book opened): ``page.media.padding`` carries the insets
    Flutter reports for an edge-to-edge window."""
    import flet as ft

    from glossarion_mobile.ui.reader.reader_view import ReaderDeps, ReaderScreen
    from glossarion_mobile.ui.router import parse_route

    left, top, right, bottom = insets
    page = types.SimpleNamespace(width=size[0], height=size[1], views=[], platform=None, web=False,
                                 media=types.SimpleNamespace(padding=ft.Padding(left=left, top=top, right=right,
                                                                                bottom=bottom)),
                                 show_dialog=lambda d: None, pop_dialog=lambda: None)
    return ReaderScreen(parse_route(ROUTE), ReaderDeps(page=page, webview_ok=lambda: False))


def _path_to(root: Any, target: Any) -> Optional[list]:
    """The controls from ``root`` down to ``target`` (``content`` / ``controls`` children), or None."""
    if root is target:
        return [root]
    for name in ("content", "controls"):
        child = getattr(root, name, None)
        for item in (child if isinstance(child, list) else [child] if child is not None else []):
            if hasattr(item, "__dict__"):
                found = _path_to(item, target)
                if found is not None:
                    return [root, *found]
    return None


@needs_flet
def test_page_sits_in_a_safe_area_and_the_chrome_stays_edge_to_edge():
    """The page layer (WebView / native page / empty state, all swapped into ``page_slot``) is inside a
    SafeArea that avoids every system intrusion; the View, the body and the Stack are not, so the chrome
    bars (each with its own SafeArea) still reach under the status and navigation bars."""
    import flet as ft

    screen = _bare_screen()
    view = screen.build_view(ROUTE)
    assert screen.edge_to_edge and view.appbar is None and view.padding == 0
    area = screen.page_area
    assert screen.stack.controls[0] is area and isinstance(area, ft.SafeArea)
    assert area.content is screen.page_slot
    assert (area.avoid_intrusions_top, area.avoid_intrusions_bottom, area.avoid_intrusions_left,
            area.avoid_intrusions_right) == (True, True, True, True)
    path = _path_to(view, screen.stack)
    assert path is not None and not any(isinstance(c, ft.SafeArea) for c in path)
    # the chrome: full-width overlays pinned to the top / bottom edge, each in its own SafeArea
    assert (screen.top_slot.top, screen.top_slot.left, screen.top_slot.right) == (0, 0, 0)
    assert (screen.bottom_slot.bottom, screen.bottom_slot.left, screen.bottom_slot.right) == (0, 0, 0)
    top_safe, bottom_safe = screen.chrome.top.content, screen.chrome.bottom.content
    assert isinstance(top_safe, ft.SafeArea) and top_safe.avoid_intrusions_bottom is False
    assert isinstance(bottom_safe, ft.SafeArea) and bottom_safe.avoid_intrusions_top is False
    assert screen.stack.controls[1:] == [screen.top_slot, screen.bottom_slot, screen.selection.container,
                                         screen.loading]
    # every page the Reader shows goes into page_slot, so it is always inside the SafeArea
    screen._show_error("x")
    assert screen.stack.controls[0] is area and area.content is screen.page_slot


@needs_flet
def test_page_area_size_follows_the_insets_the_tablet_panel_and_the_shell():
    from glossarion_mobile.ui.reader.reader_view import TOC_PANEL_WIDTH

    screen = _bare_screen()
    assert screen._page_insets() == (0.0, 0.0, 0.0, 0.0)  # not on screen yet: nothing is inset
    screen.build_view(ROUTE)
    assert screen._page_insets() == INSETS
    assert screen._page_size() == (412.0, 815.0)
    # a width / rotation change: the native page gets the page area, the chapters drawer the full height
    screen.session, screen.state = types.SimpleNamespace(), "ready"
    screen.layout = screen._effective_layout()
    screen._on_width()
    assert (screen.fallback.width, screen.fallback.height) == (412.0, 815.0)
    assert screen.toc.box.height == 915.0  # the drawer / side panel sit outside the page area (own SafeArea)
    # landscape with a cutout on the left
    screen.page.width, screen.page.height = 915.0, 412.0
    screen.page.media.padding = type(screen.page.media.padding)(left=48, top=24, right=0, bottom=0)
    screen._on_width()
    assert (screen.fallback.width, screen.fallback.height) == (867.0, 388.0)
    # tablet: the open chapters panel takes its 320 dp from the page area
    screen.page.width, screen.page.height = 1280.0, 800.0
    screen.page.media.padding = type(screen.page.media.padding)(left=0, top=24, right=0, bottom=48)
    screen.toggle_side_panel(True)
    assert screen.panel_open and screen._page_size() == (1280.0 - TOC_PANEL_WIDTH, 728.0)
    assert screen.fallback.width == 1280.0 - TOC_PANEL_WIDTH and screen.fallback.height == 728.0
    # without the Reader's own View the shell's SafeArea already holds the body: no second inset
    shell_screen = _bare_screen()
    assert shell_screen._page_size() == SCREEN


@needs_flet
def test_selection_chips_float_next_to_the_selection_on_the_inset_page():
    """The page reports a selection rect as fractions of its own viewport (the page area); the chips
    float in the full-screen Stack, so the rect moves down by the top inset and shrinks to the page."""
    from glossarion_mobile.ui.reader.chrome import SelectionChipRow

    screen = _bare_screen()
    screen.build_view(ROUTE)
    rect = (0.1, 0.5, 0.4, 0.05)
    mapped = screen._stack_rect(rect)
    assert mapped == pytest.approx((0.1, (52 + 0.5 * 815) / 915, 0.4, 0.05 * 815 / 915))
    assert SelectionChipRow._top_for(mapped, 915) == pytest.approx(52 + 0.5 * 815 - 60)
    # the page's events: the selection, then its rect (the extras stamp their page id)
    screen.current_doc = "d1"
    screen.deduper.set_document("d1", 0)
    screen.handle_payload({"type": "sel", "seq": 1, "text": "검", "chapter": 0})
    assert screen.selection.visible
    screen.handle_payload({"t": "selrect", "seq": 2, "doc": "d1", "x": 0.1, "y": 0.9, "w": 0.4, "h": 0.03})
    selection_bottom = 52 + 0.93 * 815
    assert screen.selection.container.top == pytest.approx(52 + 0.9 * 815 - 60)
    assert screen.selection.container.top < selection_bottom < 915 - 48  # above the navigation bar
    screen.handle_payload({"t": "selrect", "seq": 3, "doc": "d1", "x": 0, "y": 0.01, "w": 1, "h": 0.04})
    assert screen.selection.container.top == pytest.approx(52 + 0.05 * 815 + 12)  # below a selection at the top


@needs_flet
def test_native_page_edge_tap_scrolls_what_is_visible(monkeypatch):
    """The native page's edge tap scrolls 90 % of what the list shows (its scroll events' viewport,
    else the page area), never 90 % of the whole screen: 0.9 x 915 = 823.5 is more than the 815 dp a
    phone with 52 + 48 dp bars shows, and a turn would skip text."""
    import flet as ft

    from glossarion_mobile.ui.reader import blocks as rb
    from glossarion_mobile.ui.reader import fallback_view as fv
    from glossarion_mobile.ui.reader import model as rm

    monkeypatch.setattr(fv, "PAGE_SCROLL_MS", 5)
    monkeypatch.setattr(fv, "PAGE_SETTLE", 0.01)
    deltas: list = []
    page = fv.FallbackPage(on_tap_zone=lambda z: None, on_pinch=lambda s: None, on_pinch_end=lambda: None,
                           on_paragraph=lambda t: None, on_scroll=lambda f: None)

    def event(pixels: float, viewport: Optional[float]) -> Any:
        data = {"pixels": pixels, "min_scroll_extent": 0.0, "max_scroll_extent": 5000.0}
        if viewport is not None:
            data["viewport_dimension"] = viewport
        return types.SimpleNamespace(**data)

    async def scroll_to(self, offset=None, delta=None, scroll_key=None, duration=0, curve=None):
        deltas.append(delta)
        page._on_scroll(event((page.pixels or 0.0) + (delta or 0.0), page.viewport))

    monkeypatch.setattr(ft.ListView, "scroll_to", scroll_to)
    theme = {"bg": "#1e1e1e", "fg": "#d4d4d4"}

    async def scenario():
        page.set_size(412, 815)  # the Reader passes the page area
        page.render(rb.html_to_blocks("<p>long</p>"), theme=theme, settings=rm.ReaderSettings())
        assert page.page_step() == pytest.approx(0.9 * 815)
        assert await page.page_by(1) == fv.PAGE_SCROLLED and deltas[-1] == pytest.approx(733.5)
        page._on_scroll(event(800.0, 791.0))  # the list's own viewport (a JobStrip footer below it)
        assert page.viewport == 791.0 and page.page_step() == pytest.approx(0.9 * 791)
        assert await page.page_by(1) == fv.PAGE_SCROLLED and deltas[-1] == pytest.approx(0.9 * 791)
        assert await page.page_by(-1) == fv.PAGE_SCROLLED and deltas[-1] == pytest.approx(-0.9 * 791)
        page.set_size(412, 815)  # the same size: the viewport stays known
        assert page.viewport == 791.0
        page.set_size(915, 364)  # rotated: the old viewport is stale until the next scroll event
        assert page.viewport is None and page.page_step() == pytest.approx(0.9 * 364)
        page._on_scroll(event(100.0, 2000.0))  # never more than the page area
        assert page.page_step() == pytest.approx(0.9 * 364)
        assert all(abs(d) < 815 for d in deltas)

    asyncio.run(scenario())


@needs_flet
def test_theme_background_reaches_the_strips_beside_the_page():
    """The page sits in its SafeArea, so the strips under the system bars show the body and the View:
    a theme change (Aa) paints them too, not just the page."""
    from glossarion_mobile.ui.reader.aa_sheet import SCOPE_ALL

    screen = _bare_screen()
    view = screen.build_view(ROUTE)
    first = screen.theme.get("bg")
    assert view.bgcolor == first and screen.body.bgcolor == first and screen.page_slot.bgcolor == first
    other = next(i for i, theme in enumerate(screen.themes) if theme.get("bg") != first)
    screen._apply_settings({"theme": other}, SCOPE_ALL, save=False)
    painted = screen.theme.get("bg")
    assert painted != first
    assert view.bgcolor == painted and screen.body.bgcolor == painted and screen.page_slot.bgcolor == painted


@needs_flet
@pytest.mark.skipif(not D1._has("msgpack"), reason="msgpack not installed")
def test_reader_in_the_app_shell_puts_only_the_page_in_a_safe_area(phone):
    """The real app on the fake session as Android: the Reader's View (wrapped by the shell with its
    JobStrip footer) has no SafeArea above the Stack, the page is in its own, and the native page gets
    the page area once Flutter reports the insets."""
    import flet as ft

    async def scenario():
        await phone.start()
        phone.app.prefs.set("reader_prefs", {"lightweight": True})
        phone.session.apply_page_patch({"height": 915})
        phone.page.media = ft.PageMediaData(padding=ft.Padding(left=0, top=52, right=0, bottom=48),
                                            view_padding=ft.Padding(left=0, top=52, right=0, bottom=48),
                                            view_insets=ft.Padding(left=0, top=0, right=0, bottom=0),
                                            device_pixel_ratio=2.625, orientation=ft.Orientation.PORTRAIT,
                                            always_use_24_hour_format=False)
        try:
            screen = await D1._open(phone, chapter=0)
            view = phone.page.views[-1]
            assert view is screen.view and screen.edge_to_edge and view.appbar is None
            path = _path_to(view, screen.stack)
            assert path is not None and not any(isinstance(c, ft.SafeArea) for c in path), \
                [type(c).__name__ for c in path]
            assert _path_to(screen.page_area, screen.fallback.control) is not None
            assert isinstance(screen.page_area, ft.SafeArea) and screen.stack.controls[0] is screen.page_area
            assert (screen.fallback.width, screen.fallback.height) == (412.0, 815.0)
        finally:
            await phone.stop()

    asyncio.run(scenario())


# =====================================================================================
# The shared page shell's CSS (mobile only)
# =====================================================================================


@needs_cores
def test_mobile_paged_css_adds_the_insets_once_and_sizes_the_columns_from_what_is_left():
    import reader_doc

    css = reader_doc._MOBILE_PAGED_CSS
    rules = [rule.strip() for rule in css.replace("}", "}\n").splitlines() if rule.strip()]
    padded = [r for r in rules if "env(safe-area-inset-top" in r or "env(safe-area-inset-bottom" in r]
    # no rule pads html and body together with an inset (the U8 rule counted the top inset twice)
    assert not any(r.startswith("html, body") for r in padded), padded
    assert "html { height: 100%; height: 100dvh; padding: 0; touch-action: pan-y; }" in css
    assert ("body { height: 100%; height: 100dvh; padding-top: calc(20px + env(safe-area-inset-top, 0px)); "
            "padding-bottom: calc(16px + env(safe-area-inset-bottom, 0px)); touch-action: pan-y; }") in css
    insets = "env(safe-area-inset-top, 0px) - env(safe-area-inset-bottom, 0px)"
    for unit in ("100vh", "100dvh"):
        assert f"height: calc({unit} - 36px - {insets}) !important;" in css  # beats the inline innerHeight - 36
        assert f"max-height: calc({unit} - 60px - {insets});" in css
        assert f"min-height: calc({unit} - 40px - {insets});" in css
        assert f"max-height: calc({unit} - 100px - {insets});" in css
    assert "-webkit-column-break-before" in css and "max(18px, env(safe-area-inset-left))" in css
    # the desktop page keeps its own geometry (tests/test_reader_doc.py proves it byte-identical)
    desktop = reader_doc.wrap_reader_html("<p>x</p>", 0, paginated=True, embedded_css="")
    assert "padding: 10px 0 26px 0" in desktop and "c.style.height = (window.innerHeight - 36) + 'px';" in desktop
    assert "env(" not in desktop and "100dvh" not in desktop
    mobile = reader_doc.wrap_reader_html("<p>x</p>", 0, paginated=True, embedded_css="", mobile=True)
    assert mobile.index(css) > mobile.index("padding: 10px 0 26px 0")  # after the desktop rules: it wins


# =====================================================================================
# The owner's complaint, measured in headless Chrome
# =====================================================================================

FULL = bool(os.environ.get("GLOSSARION_READER_LAYOUT_FULL"))
SCREENS = [(412, 915), (360, 800)]
STATUS, NAV_3BUTTON, NAV_GESTURE = 24, 48, 24
# (name, status bar, navigation bar, WebView box: "safe" = in the page SafeArea / "full" = under the bars,
#  insets the WebView reports through env(safe-area-inset-*) (top, bottom))
SCENARIOS = [
    ("safe area, 3-button navigation", STATUS, NAV_3BUTTON, "safe", (0, 0)),
    ("safe area, gesture navigation", STATUS, NAV_GESTURE, "safe", (0, 0)),
    ("safe area, WebView also reports the bars", STATUS, NAV_3BUTTON, "safe", (STATUS, NAV_3BUTTON)),
    ("safe area, WebView reports a cutout", 32, NAV_3BUTTON, "safe", (32, 0)),
    ("full screen, WebView reports the bars", STATUS, NAV_3BUTTON, "full", (STATUS, NAV_3BUTTON)),
]
if FULL:
    TYPOGRAPHY = [(size, spacing, family) for size in (12, 14, 18, 24) for spacing in (1.0, 1.2, 1.8, 2.4)
                  for family in ("Embedded CSS", "Serif")]
else:
    TYPOGRAPHY = [(12, 1.2, "Embedded CSS"), (14, 1.8, "Embedded CSS"), (18, 1.8, "Serif"), (24, 2.4, "Serif"),
                  (14, 1.0, "Embedded CSS")]
LINE_SLACK = 1.0  # px: sub-pixel rounding of line boxes at a fractional device pixel ratio

_EN_TEXT = ("The rain kept falling on the old harbour city while the night watch walked its rounds, and the "
            "lanterns swung in the wind above the narrow streets. ")
_KO_TEXT = "비가 오래된 항구 도시에 계속 내렸고 야경꾼들은 밤새 순찰을 돌았다. 바람에 등불이 흔들렸다. "

_MEASURE_JS = r"""
(async () => {
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  for (let i = 0; i < 300 && !(window.GLRDR && document.readyState === 'complete'); i++) { await sleep(20); }
  await sleep(80);
  const H = window.innerHeight, W = window.innerWidth;
  const probe = document.createElement('div');
  probe.style.cssText = 'position:fixed;left:0;top:0;width:1px;visibility:hidden;' +
    'padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)';
  document.documentElement.appendChild(probe);
  const ps = getComputedStyle(probe);
  const envTop = parseFloat(ps.paddingTop) || 0, envBottom = parseFloat(ps.paddingBottom) || 0;
  probe.remove();
  // A text line as its line box: Range.getClientRects() gives the font's content area (ascent + descent),
  // which overhangs the line box when the line spacing is tighter than the font (1.0 with a serif face).
  function lineRects() {
    const out = [];
    const root = document.getElementById('content') || document.body;
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, null);
    let node;
    while ((node = walker.nextNode())) {
      if (!node.nodeValue.trim()) { continue; }
      const range = document.createRange();
      range.selectNodeContents(node);
      const lineHeight = parseFloat(getComputedStyle(node.parentElement).lineHeight);
      for (const q of range.getClientRects()) {
        if (q.width > 0.5 && q.height > 0.5 && q.left >= -1 && q.left < W - 1) {
          const over = lineHeight > 0 ? Math.max(0, (q.height - lineHeight) / 2) : 0;
          out.push([q.top + over, q.bottom - over]);
        }
      }
    }
    return out;
  }
  function imageRects() {
    return Array.prototype.slice.call(document.querySelectorAll('img')).map((img) => img.getBoundingClientRect())
      .filter((r) => r.width > 1 && r.left >= -1 && r.left < W - 1).map((r) => [r.top, r.bottom]);
  }
  const res = {H: H, W: W, envTop: envTop, envBottom: envBottom, lineTops: [], lineBottoms: [], images: []};
  const c = document.getElementById('columns');
  if (c) {
    const cr = c.getBoundingClientRect();
    res.paged = true;
    res.col = [cr.top, cr.bottom];
    res.count = GLRDR.count();
    res.pages = 0;
    for (let p = 0; p < res.count; p++) {
      GLRDR.goTo(p);
      await sleep(0);
      const rects = lineRects();
      if (rects.length) { res.pages++; }
      for (const [t, b] of rects) { res.lineTops.push(t); res.lineBottoms.push(b); }
      for (const r of imageRects()) { res.images.push(r); }
    }
  } else {
    res.paged = false;
    const first = lineRects();
    res.lineTops = first.map((r) => r[0]);
    window.scrollTo(0, 1e9);
    await sleep(80);
    res.lineBottoms = lineRects().map((r) => r[1]);
    res.images = imageRects();
  }
  return res;
})()
"""


def _png(width: int, height: int) -> bytes:
    """A plain RGB PNG (a tall illustration page)."""
    row = b"\x00" + bytes((200, 120, 40)) * width
    data = zlib.compress(row * height)

    def chunk(kind: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", data) + chunk(b"IEND", b""))


def _layout_epub(path: Path) -> Path:
    """Long chapters the way books come: the book's own stylesheet, headings, paragraphs of every
    length, a block quote, Korean text and a tall illustration."""
    from ebooklib import epub

    book = epub.EpubBook()
    book.set_identifier("glossarion-reader-layout")
    book.set_title("Layout Book")
    book.set_language("en")
    style = epub.EpubItem(uid="style", file_name="style/book.css", media_type="text/css",
                          content=b"p { text-indent: 1.5em; margin: 0 0 0.7em 0; } h1 { font-size: 1.5em; }"
                                  b" blockquote { margin: 1em 2em; font-style: italic; }")
    book.add_item(style)
    book.add_item(epub.EpubItem(uid="plate", file_name="images/plate.png", media_type="image/png",
                                content=_png(300, 1300)))
    items = []
    for number in (1, 2):
        paragraphs = []
        for n in range(34):
            text = (_KO_TEXT if number == 2 and n % 2 else _EN_TEXT) * (1 + (n * 7) % 5)
            paragraphs.append(f"<p>{number}.{n + 1} {text}</p>")
            if n == 9:
                paragraphs.append(f"<blockquote><p>{_EN_TEXT * 2}</p></blockquote>")
            if n == 17 and number == 2:
                # a full-page illustration on its own (reader_doc pulls a preceding <p> or heading into the
                # full-page wrapper, which clips what does not fit: that shared heuristic is not tested here)
                paragraphs.append('<hr/><p><img src="../images/plate.png" alt="plate"/></p>')
        chapter = epub.EpubHtml(title=f"Chapter {number}", file_name=f"text/chapter{number:04d}.xhtml", lang="en")
        chapter.content = (f"<html><head><title>Chapter {number}</title>"
                           f'<link rel="stylesheet" href="../style/book.css" type="text/css"/></head>'
                           f"<body><h1>Chapter {number}</h1>{''.join(paragraphs)}</body></html>")
        chapter.add_item(style)
        book.add_item(chapter)
        items.append(chapter)
    book.toc = tuple(items)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav", *items]
    path.parent.mkdir(parents=True, exist_ok=True)
    epub.write_epub(str(path), book)
    return path


def _bridge_with_cdp(script: str) -> str:
    """test_devfix_issue1's DevTools bridge plus a ``cdp`` command (any DevTools method: the device
    metrics and the safe-area insets of a scenario)."""
    if "case 'cdp':" in script:
        return script
    marker = "    case 'quit':"
    assert marker in script, "test_devfix_issue1's bridge changed: teach _bridge_with_cdp the new layout"
    return script.replace(marker, "    case 'cdp':\n      return await send(cmd.method, cmd.params || {});\n"
                          + marker, 1)


@pytest.fixture(scope="module")
def chrome(tmp_path_factory):
    found = D1.ChromeEngine.locate()
    if found is None:
        pytest.skip("headless Chrome and node >= 22 are needed for the page measurements")
    patch = pytest.MonkeyPatch()
    patch.setattr(D1, "_CDP_BRIDGE_JS", _bridge_with_cdp(D1._CDP_BRIDGE_JS))
    engine = D1.ChromeEngine(tmp_path_factory.mktemp("layout-chrome"), *found)
    try:
        engine.start()
    except RuntimeError as exc:
        pytest.skip(str(exc))
    finally:
        patch.undo()
    yield engine
    engine.stop()


@pytest.fixture(scope="module")
def pages(tmp_path_factory):
    """Real Reader pages: ``DocumentBuilder`` over the layout book, published by a ``ReaderServer``
    (the page, CSP and images exactly as the phone loads them). Library / output / HOME under tmp."""
    if not all(_has(m) for m in ("reader_doc", "library_core", "ebooklib")):
        pytest.skip("the shared reader cores are not importable here")
    tmp = tmp_path_factory.mktemp("layout-book")
    config = SRC_DIR / "config.json"
    config_md5 = _md5(config)
    patch = pytest.MonkeyPatch()
    for name in ("GLOSSARION_LIBRARY_DIR", "OUTPUT_DIRECTORY", "HOME", "APPDATA", "GLOSSARION_DATA_DIR"):
        folder = tmp / "env" / name.lower()
        folder.mkdir(parents=True)
        patch.setenv(name, str(folder))
    patch.setenv("GLOSSARION_HTTP_LOG", "0")
    import reader_doc

    (tmp / "epubcache").mkdir()
    patch.setattr(reader_doc, "_EPUB_CACHE_DIR_OVERRIDE", str(tmp / "epubcache"))
    from glossarion_mobile.services.reader_server import ReaderServer
    from glossarion_mobile.ui.reader import model as rm
    from glossarion_mobile.ui.reader import session as rs
    from glossarion_mobile.ui.reader.document import DocumentBuilder

    server = ReaderServer()
    server.start()
    engine = rs.DocEngine()
    session = rs.ReaderSession(rs.plan_for_file(str(_layout_epub(tmp / "Layout Book.epub"))), engine=engine,
                               cache_dir=str(tmp / "cache"))
    session.load()
    builder = DocumentBuilder(session, server.register_image)
    themes = [dict(t) for t in reader_doc.READER_THEMES]
    serial = iter(range(1, 1 << 30))

    def publish(index: int, layout: str, size: int, spacing: float, family: str) -> str:
        settings = rm.ReaderSettings().with_changes(font_size=size, line_spacing=spacing, font_family=family,
                                                    layout=layout)
        built = builder.build(index, settings=settings, layout=layout, theme=rm.theme_for(themes, settings,
                                                                                            app_dark=False),
                              doc_id=f"d{next(serial)}", event_url=server.event_path)
        return server.publish(built.html, script_nonce=built.nonce or None)

    state = types.SimpleNamespace(publish=publish, session=session, rm=rm)
    try:
        yield state
    finally:
        builder.close()
        session.close()
        server.stop()
        patch.undo()
        assert _md5(config) == config_md5, "src/config.json changed"


def _set_screen(chrome, width: int, height: int, env: tuple) -> None:
    chrome.call("cdp", method="Emulation.setDeviceMetricsOverride",
                params={"width": width, "height": height, "deviceScaleFactor": 2.625, "mobile": True})
    top, bottom = env
    try:
        chrome.call("cdp", method="Emulation.setSafeAreaInsetsOverride", params={"insets": {
            "top": top, "topMax": top, "bottom": bottom, "bottomMax": bottom,
            "left": 0, "leftMax": 0, "right": 0, "rightMax": 0}})
    except RuntimeError as exc:
        if top or bottom:
            pytest.skip(f"this Chrome cannot emulate safe-area insets ({exc})")


def _measure(chrome, url: str) -> dict:
    chrome.navigate(url)
    return chrome.call("eval", timeout=60.0, expr=_MEASURE_JS)


def _problems(result: dict, *, covered: tuple, env: tuple, slack: float = LINE_SLACK) -> list:
    """What would be cut or hidden on the phone: lines past the bottom of the page or under a bar, a
    column box not where the insets leave room, an illustration taller than its column. Lines are line
    boxes (``_MEASURE_JS``), within ``slack`` px."""
    height = float(result["H"])
    cover_top, cover_bottom = covered
    problems = []
    if abs(result["envTop"] - env[0]) > 0.5 or abs(result["envBottom"] - env[1]) > 0.5:
        problems.append(f"reported insets {result['envTop']}/{result['envBottom']} != {env}")
    bottoms, tops = result["lineBottoms"], result["lineTops"]
    if not bottoms or not tops:
        return problems + ["no text measured"]
    past_page = [b for b in bottoms if b > height + 0.5]
    under_bar = [b for b in bottoms if b > height - cover_bottom + slack] + [t for t in tops if t < cover_top - slack]
    if past_page:
        problems.append(f"{len(past_page)} lines past the bottom of the page (max {max(past_page):.1f} > {height})")
    if under_bar:
        problems.append(f"{len(under_bar)} lines under a system bar")
    if result["paged"]:
        col_top, col_bottom = result["col"]
        if abs(col_top - (20 + env[0])) > 0.6 or abs(col_bottom - (height - 16 - env[1])) > 0.6:
            problems.append(f"columns {col_top:.1f}..{col_bottom:.1f}, expected {20 + env[0]}..{height - 16 - env[1]}")
        past_column = [b for b in bottoms if b > col_bottom + slack]
        if past_column:
            problems.append(f"{len(past_column)} lines past the column box")
        tall = [r for r in result["images"] if r[1] > col_bottom + 1 or r[0] < col_top - 1]
        if tall:
            problems.append(f"{len(tall)} illustrations outside the column box")
    else:
        if max(bottoms) > height - max(cover_bottom, env[1]) + slack:
            problems.append(f"the chapter's last line ends at {max(bottoms):.1f} (page {height})")
    return problems


def _run_matrix(chrome, pages, width: int, height: int, layouts: tuple, typography: list) -> tuple:
    failures, stats = [], {"docs": 0, "pages": 0, "lines": 0}
    for name, status, nav, box, env in SCENARIOS:
        view_height = height - status - nav if box == "safe" else height
        covered = (0, 0) if box == "safe" else (status, nav)
        _set_screen(chrome, width, view_height, env)
        for layout in layouts:
            for size, spacing, family in typography:
                for chapter in (0, 1):
                    result = _measure(chrome, pages.publish(chapter, layout, size, spacing, family))
                    stats["docs"] += 1
                    stats["pages"] += int(result.get("pages") or 1)
                    stats["lines"] += len(result["lineBottoms"])
                    assert result["H"] == view_height and result["W"] == width
                    for problem in _problems(result, covered=covered, env=env):
                        failures.append(f"{width}x{height} {name}, {layout} {size}pt x{spacing} {family}, "
                                        f"chapter {chapter + 1}: {problem}")
    return failures, stats


@needs_cores
@pytest.mark.parametrize("width,height", SCREENS, ids=[f"{w}x{h}" for w, h in SCREENS])
def test_owner_report_10_no_reader_line_is_cut_at_the_bottom_of_a_page(chrome, pages, width, height):
    """The owner's complaint, on the pages the phone loads: with the page in its SafeArea (and when a
    WebView reports insets anyway, or a page fills the screen and reports the bars) no line of any page
    ends past the bottom of the page or under a system bar, in paged and scroll layouts."""
    rm = pages.rm
    failures, stats = _run_matrix(chrome, pages, width, height, (rm.LAYOUT_SINGLE,), TYPOGRAPHY)
    scroll_typography = TYPOGRAPHY if FULL else TYPOGRAPHY[:2]
    more, scroll_stats = _run_matrix(chrome, pages, width, height, (rm.LAYOUT_SCROLL,), scroll_typography)
    failures += more
    print(f"[reader-layout] {width}x{height}: paged {stats}, scroll {scroll_stats}")
    assert stats["pages"] > 10 * len(SCENARIOS) * len(TYPOGRAPHY), stats  # long chapters: many pages each
    assert chrome.exceptions == [], chrome.exceptions
    assert failures == [], "\n".join(failures[:40])


@needs_cores
def test_without_the_page_safe_area_the_navigation_bar_covers_text(chrome, pages):
    """The owner's U8 phone: the page filled the screen and its WebView reported no inset, so the
    3-button navigation bar covered the last lines of the pages. No CSS can see that bar: the page's
    SafeArea is what removes it (``test_page_sits_in_a_safe_area_and_the_chrome_stays_edge_to_edge``).
    This control shows the measurement above sees the owner's defect."""
    rm = pages.rm
    _set_screen(chrome, 412, 915, (0, 0))
    covered = []
    for chapter in (0, 1):
        result = _measure(chrome, pages.publish(chapter, rm.LAYOUT_SINGLE, 14, 1.8, "Embedded CSS"))
        problems = _problems(result, covered=(STATUS, NAV_3BUTTON), env=(0, 0))
        covered += [p for p in problems if "under a system bar" in p]
    assert covered


@needs_cores
def test_page_geometry_without_insets_is_the_desktop_geometry(chrome, pages):
    """With no inset the page keeps the desktop geometry the paging was built for: the columns run from
    20 px to innerHeight - 16 px (``_setupColumns``' ``innerHeight - 36``), so page counts do not change."""
    rm = pages.rm
    for width, height in SCREENS:
        _set_screen(chrome, width, height, (0, 0))
        result = _measure(chrome, pages.publish(0, rm.LAYOUT_SINGLE, 14, 1.8, "Embedded CSS"))
        assert result["col"] == pytest.approx([20.0, height - 16.0], abs=0.6)
        assert result["count"] >= 3
