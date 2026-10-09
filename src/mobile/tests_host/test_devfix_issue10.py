"""Owner's device report #10 on the U8 APK (2026-10-08): "the epub reader text clips out at the bottom".

What the owner saw: on an Android 15+ phone (the APK targets SDK 36, so the app runs edge to edge) the last
line or two of a Reader page were cut at the bottom of the screen or lay under the navigation bar. Two
defects stacked up (diagnosis wf_89d52c08-c74, skeptic-checked): the Reader's own full-screen View put the
WebView straight into its Stack (no SafeArea, unlike every other screen), so the page lay under the status
and navigation bars; and the mobile paged page shell padded ``html`` AND ``body`` with the safe-area insets
(the top inset counted twice) while ``#columns`` stayed ``innerHeight - 36`` tall, so with any reported
inset the column box ran past the bottom of the screen.

This acceptance test replays it on the real objects: the real app (``GlossarionApp`` on the fake Flet
session as Android, ``test_devfix_issue1.phone``) opens a long two-chapter book (a Completed book with its
raw EPUB: Translated, Original and Bilingual) through its real ReaderFeature, ReaderServer and
ReaderScreen; test_devfix_issue1's WebView stand-in loads every page the Reader publishes into headless
Chrome (DevTools over node). Then, for 412x915 and 360x800 phones:

* Flutter's layout of the Reader's View is resolved from the real control tree with the system insets the
  phone reports (``page.media.padding``: none, a 24 dp status bar + 48 dp 3-button navigation bar, and a
  32 dp status bar around a camera cutout + 24 dp gesture navigation bar): the
  View, the app shell's wrapper, the Reader's Stack and the page's ``SafeArea`` give the WebView's box on
  the screen. The WebView in Chrome gets exactly that size; the status and navigation bars are modelled on
  the screen around it (whatever part of the WebView lies under them is hidden).
* The WebView may or may not report the bars through ``env(safe-area-inset-*)`` (it depends on the device's
  WebView): none, the bars, or only the cutout are run.
* Several Aa settings (live ``applyStyle`` and fresh page loads), both chapters, Original and Bilingual,
  and the scroll layout. On every paged page every text line (and the illustration) is measured: each one
  must be fully visible (inside the WebView, the column box and clear of the bars) on exactly one page; no
  line is cut, hidden, skipped or shown twice. In the scroll layout every line can be scrolled fully into
  view. The column box is where the desktop geometry puts it (20 px .. innerHeight - 16 px) less the
  reported insets.
* A paragraph leading straight into a portrait illustration (where novels put them): the shared page
  shell pulls that paragraph into the full-page illustration box, which must still fit the phone's column.

Controls show the measurement sees the owner's defect (the page outside its SafeArea under the bars; the
U8 page shell when the base commit is in this clone), and the desktop page bytes are unchanged (identical
to the batch base's ``reader_doc`` when that commit is in this clone; never dependent on the mobile shell).

Run from src/mobile (mobile venv; headless Chrome + node >= 22 needed for the page measurements):
    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_devfix_issue10.py
"""

from __future__ import annotations

import asyncio
import collections
import importlib.util
import itertools
import shutil
import subprocess
import sys
import types
from pathlib import Path
from typing import Any, Optional

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = MOBILE_DIR / "app"
SRC_DIR = MOBILE_DIR.parent
REPO_DIR = SRC_DIR.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

# test_reader_layout's helpers (its DevTools ``cdp`` command, the PNG maker, the control-path walk) and,
# through it, test_devfix_issue1's app fixture, WebView stand-in and Chrome engine (one copy of each).
_SPEC = importlib.util.spec_from_file_location("_glossarion_devfix10_layout_helpers",
                                               Path(__file__).with_name("test_reader_layout.py"))
LAYOUT = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(LAYOUT)
D1 = LAYOUT.D1

phone = D1.phone  # the real app on the fake Flet session as Android (isolated storage), from devfix issue 1

needs_msgpack = pytest.mark.skipif(not D1._has("msgpack"), reason="msgpack not installed")

ORIGINAL, TRANSLATED, BILINGUAL = "original", "translated", "bilingual"
SIDES = ("left", "top", "right", "bottom")
SCREENS = [(412, 915, 2.625), (360, 800, 3.0)]  # dp and device pixel ratio
STATUS_BAR, NAV_BAR = 24, 48  # dp: status bar, 3-button navigation bar
CUTOUT_BAR, GESTURE_BAR = 32, 24  # dp: a status bar around a camera cutout, the gesture navigation handle
# (name, the insets Flutter reports (page.media.padding: left, top, right, bottom),
#  the insets the WebView reports through env(safe-area-inset-top / -bottom))
SCENARIOS = [
    ("no system bar over the app", (0, 0, 0, 0), (0, 0)),
    ("edge to edge, 24 dp status + 48 dp navigation bar, the WebView reports no inset",
     (0, STATUS_BAR, 0, NAV_BAR), (0, 0)),
    ("edge to edge, 24 dp status + 48 dp navigation bar, the WebView reports the bars too",
     (0, STATUS_BAR, 0, NAV_BAR), (STATUS_BAR, NAV_BAR)),
    ("edge to edge, 32 dp cutout status bar + 24 dp gesture bar, the WebView reports the cutout only",
     (0, CUTOUT_BAR, 0, GESTURE_BAR), (CUTOUT_BAR, 0)),
]
DEFAULT_TYPE = (14, 1.8, "Embedded CSS")
# (font size pt, line spacing, family): live applyStyle changes and, across the family switch, a reload
TYPOGRAPHY = [(12, 1.2, "Embedded CSS"), (24, 2.4, "Embedded CSS"), (18, 1.0, "Serif"), (14, 1.8, "Serif"),
              DEFAULT_TYPE]
SLACK = 1.0  # px: sub-pixel rounding of line boxes at a fractional device pixel ratio
BASES = ("55c46555", "96da1ec6")  # device-fix batch 2 base, and the owner's U8 APK (same reader_doc)

_EN = ("The rain kept falling on the old harbour city while the night watch walked its rounds, and the "
       "lanterns swung in the wind above the narrow streets. ")
_KO = "비가 오래된 항구 도시에 계속 내렸고 야경꾼들은 밤새 순찰을 돌았다. 바람에 등불이 흔들렸다. "


# =====================================================================================
# The owner's book: long chapters, the book's own stylesheet, an illustration, its raw EPUB
# =====================================================================================


def _write_book(path: Path, *, marker: str, lang: str, text: str, plate: tuple = (300, 1300),
                lead_in: Optional[str] = None) -> Path:
    """Two long chapters; chapter 2 has a full-page illustration on its own (after a rule), or (``lead_in``)
    both chapters have one right after that paragraph, the way novels place them."""
    from ebooklib import epub

    book = epub.EpubBook()
    book.set_identifier(f"glossarion-devfix-issue10-{marker.lower()}")
    book.set_title("Owner Book")
    book.set_language(lang)
    style = epub.EpubItem(uid="style", file_name="style/book.css", media_type="text/css",
                          content=b"p { text-indent: 1.5em; margin: 0 0 0.7em 0; } h1 { font-size: 1.5em; }"
                                  b" blockquote { margin: 1em 2em; font-style: italic; }")
    book.add_item(style)
    book.add_item(epub.EpubItem(uid="plate", file_name="images/plate.png", media_type="image/png",
                                content=LAYOUT._png(*plate)))
    items = []
    for number in (1, 2):
        mark = f"{marker}-MARK-{number:02d}"
        body = [f"<h1>{marker} chapter {number}</h1>", f"<p>{mark}</p>"]
        for n in range(30):
            body.append(f"<p>{mark} {n + 1}. {text * (1 + (n * 7) % 5)}</p>")
            if n == 9:
                body.append(f"<blockquote><p>{text * 2}</p></blockquote>")
            if n == 15 and lead_in is not None:
                body.append(f'<p>{lead_in}</p><p><img src="images/plate.png" alt="plate"/></p>')
            elif n == 15 and number == 2:  # a full-page illustration on its own page
                body.append('<hr/><p><img src="images/plate.png" alt="plate"/></p>')
        chapter = epub.EpubHtml(title=f"{marker} chapter {number}", file_name=f"chapter{number:04d}.xhtml", lang=lang)
        chapter.content = (f"<html><head><title>{marker} chapter {number}</title>"
                           f'<link rel="stylesheet" href="style/book.css" type="text/css"/></head>'
                           f"<body>{''.join(body)}</body></html>")
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


def _owner_book(folder: Path, **illustration: Any) -> dict:
    raw = _write_book(folder / "Owner Book.epub", marker="RAW", lang="ko", text=_KO, **illustration)
    translated = _write_book(folder / "Owner Book_translated.epub", marker="EN", lang="en", text=_EN, **illustration)
    return {"name": "Owner Book", "path": str(translated), "raw_source_path": str(raw)}


# =====================================================================================
# Flutter's layout of the Reader's View: where the WebView is on the screen
# =====================================================================================


def _sides(value: Any) -> tuple:
    if value is None:
        return 0.0, 0.0, 0.0, 0.0
    if isinstance(value, (int, float)):
        return (float(value),) * 4
    return tuple(float(getattr(value, side, 0) or 0) for side in SIDES)


def _main_size(control: Any, axis: str) -> float:
    """A Column / Row sibling's size along the main axis (it must be fixed or empty to be modelled)."""
    import flet as ft

    if not getattr(control, "visible", True):
        return 0.0
    fixed = getattr(control, axis, None)
    if fixed is not None:
        return float(fixed)
    empty = (isinstance(control, ft.Container) and control.content is None and control.padding is None
             and control.margin is None and not control.expand)
    assert empty, f"a {type(control).__name__} beside the page takes room the model cannot size"
    return 0.0


def _webview_box(view: Any, target: Any, screen: tuple, insets: tuple, *, safe_areas: bool = True) -> tuple:
    """(x, y, width, height) of ``target`` on the screen as Flutter lays out ``view`` (Flet 1.0.3: the View
    is a Scaffold whose body keeps MediaQuery's padding when there is no app bar or bottom bar; a SafeArea
    pads each side it avoids by the padding still there and removes it for its descendants; Containers
    and the View pad; a Column / Row child that expands gets what its siblings leave; a non-positioned
    Stack child fills the Stack). ``safe_areas=False``: the page laid out as if no SafeArea existed (U8)."""
    import flet as ft

    path = LAYOUT._path_to(view, target)
    assert path is not None, f"the {type(target).__name__} is not in the Reader's View"
    assert view.appbar is None and view.bottom_appbar is None and view.navigation_bar is None
    assert view.scroll is None and view.floating_action_button is None
    pad = dict(zip(SIDES, (float(v) for v in insets)))
    x0, y0, x1, y1 = 0.0, 0.0, float(screen[0]), float(screen[1])

    def shrink(sides: tuple) -> None:
        nonlocal x0, y0, x1, y1
        x0, y0, x1, y1 = x0 + sides[0], y0 + sides[1], x1 - sides[2], y1 - sides[3]

    for depth, node in enumerate(path):
        if node is not view:
            shrink(_sides(getattr(node, "margin", None)))
        if node is target:
            break
        child = path[depth + 1]
        if isinstance(node, ft.View):
            shrink(_sides(node.padding))
            others = [c for c in node.controls if c is not child]
            assert all(_main_size(c, "height") == 0 for c in others)
        elif isinstance(node, ft.SafeArea):
            minimum = _sides(node.minimum_padding)  # Flutter: max(avoided padding, minimum) per side
            if safe_areas:
                avoided = tuple(max(pad[side] if getattr(node, f"avoid_intrusions_{side}") else 0.0, least)
                                for side, least in zip(SIDES, minimum))
                shrink(avoided)
                for side in SIDES:
                    if getattr(node, f"avoid_intrusions_{side}"):
                        pad[side] = 0.0
        elif isinstance(node, ft.Container):
            shrink(_sides(node.padding))
        elif isinstance(node, (ft.Column, ft.Row)):
            axis = "height" if isinstance(node, ft.Column) else "width"
            assert child.expand, f"the page's {type(child).__name__} does not expand in its {type(node).__name__}"
            shown = [c for c in node.controls if getattr(c, "visible", True)]
            at = shown.index(child)
            gap = float(node.spacing or 0)
            before = sum(_main_size(c, axis) + gap for c in shown[:at])
            after = sum(_main_size(c, axis) + gap for c in shown[at + 1:])
            if axis == "height":
                y0, y1 = y0 + before, y1 - after
            else:
                x0, x1 = x0 + before, x1 - after
        elif isinstance(node, ft.Stack):
            edges = [getattr(child, side, None) for side in SIDES]
            if any(edge is not None for edge in edges):
                left, top, right, bottom = edges
                width, height = getattr(child, "width", None), getattr(child, "height", None)
                nx0 = x0 + left if left is not None else (x1 - right - width if right is not None and width else x0)
                ny0 = y0 + top if top is not None else (y1 - bottom - height if bottom is not None and height else y0)
                nx1 = x1 - right if right is not None else (nx0 + width if width else x1)
                ny1 = y1 - bottom if bottom is not None else (ny0 + height if height else y1)
                x0, y0, x1, y1 = nx0, ny0, nx1, ny1
        # any other control (GestureDetector, Semantics, ...) passes its box on
    return x0, y0, x1 - x0, y1 - y0


def _bars_over(box: tuple, screen: tuple, insets: tuple) -> tuple:
    """How much of the WebView's top and bottom edge the status / navigation bar covers (WebView px)."""
    _x, y, _w, h = box
    return max(0.0, float(insets[1]) - y), max(0.0, (y + h) - (float(screen[1]) - float(insets[3])))


def _assert_isolated(phone) -> None:
    """The running app's storage (data dir, Library, output, HOME / profile folders) is the test's tmp."""
    import os

    tmp = os.path.normcase(os.path.abspath(str(phone.tmp)))
    for name in ("GLOSSARION_DATA_DIR", "GLOSSARION_LIBRARY_DIR", "OUTPUT_DIRECTORY", "HOME", "USERPROFILE", "APPDATA"):
        value = os.environ.get(name)
        assert value and os.path.normcase(os.path.abspath(value)).startswith(tmp), f"{name}={value!r} is not under tmp"


def _media(insets: tuple) -> Any:
    import flet as ft

    padding = ft.Padding(left=insets[0], top=insets[1], right=insets[2], bottom=insets[3])
    return ft.PageMediaData(padding=padding, view_padding=padding,
                            view_insets=ft.Padding(left=0, top=0, right=0, bottom=0), device_pixel_ratio=2.625,
                            orientation=ft.Orientation.PORTRAIT, always_use_24_hour_format=False)


def _assert_page_layers(screen: Any, view: Any) -> None:
    """Only the page is in a SafeArea that avoids every intrusion; the chrome bars overlay edge to edge,
    each with its own SafeArea; nothing above the Reader's Stack (View, shell wrapper, body) is one."""
    import flet as ft

    area = screen.page_area
    assert isinstance(area, ft.SafeArea) and area.content is screen.page_slot
    assert all(getattr(area, f"avoid_intrusions_{side}") for side in SIDES)
    assert screen.stack.controls[0] is area
    path = LAYOUT._path_to(view, screen.stack)
    assert path is not None and not any(isinstance(c, ft.SafeArea) for c in path), [type(c).__name__ for c in path]
    assert (screen.top_slot.top, screen.top_slot.left, screen.top_slot.right) == (0, 0, 0)
    assert (screen.bottom_slot.bottom, screen.bottom_slot.left, screen.bottom_slot.right) == (0, 0, 0)
    assert isinstance(screen.chrome.top.content, ft.SafeArea) and isinstance(screen.chrome.bottom.content, ft.SafeArea)


# =====================================================================================
# The pages, measured in Chrome
# =====================================================================================

_MEASURE_JS = r"""
(async () => {
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  for (let i = 0; i < 300 && !(window.GLRDR && document.readyState === 'complete'); i++) { await sleep(20); }
  await sleep(200);  // an Aa change re-measures the columns 30 ms after applyStyle
  for (let i = 0; i < 150 && Array.prototype.some.call(document.images, (im) => !im.complete); i++) { await sleep(20); }
  const H = window.innerHeight, W = window.innerWidth;
  const probe = document.createElement('div');
  probe.style.cssText = 'position:fixed;left:0;top:0;width:1px;visibility:hidden;' +
    'padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)';
  document.documentElement.appendChild(probe);
  const ps = getComputedStyle(probe);
  const envTop = parseFloat(ps.paddingTop) || 0, envBottom = parseFloat(ps.paddingBottom) || 0;
  probe.remove();
  const root = document.getElementById('content') || document.body;
  const nodes = [];
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, null);
  let node;
  while ((node = walker.nextNode())) {
    if (node.nodeValue.trim() && node.parentElement) { nodes.push(node); }
  }
  const imgs = Array.prototype.slice.call(root.querySelectorAll('img'));
  // Every text line (one rect per line of each text node) as its line box: Range.getClientRects() gives
  // the font's content area, which overhangs the line box when the line spacing is tighter than the font.
  function items() {
    const lines = [];
    for (const n of nodes) {
      const range = document.createRange();
      range.selectNodeContents(n);
      const lh = parseFloat(getComputedStyle(n.parentElement).lineHeight);
      for (const q of range.getClientRects()) {
        if (q.width <= 0.5 || q.height <= 0.5) { continue; }
        const over = lh > 0 ? Math.max(0, (q.height - lh) / 2) : 0;
        lines.push([q.left, q.right, q.top + over, q.bottom - over]);
      }
    }
    const pics = imgs.map((im) => { const r = im.getBoundingClientRect(); return [r.left, r.right, r.top, r.bottom]; });
    return [lines, pics];
  }
  const res = {H: H, W: W, envTop: envTop, envBottom: envBottom, text: (document.body.innerText || '').slice(0, 400),
               broken: imgs.filter((im) => !(im.complete && im.naturalWidth > 0)).map((im) => im.getAttribute('src'))};
  const c = document.getElementById('columns');
  res.paged = !!c;
  const typeEl = c || document.body;
  res.fontSize = parseFloat(getComputedStyle(typeEl).fontSize);
  res.lineHeight = parseFloat(getComputedStyle(typeEl).lineHeight);
  if (c) {
    const start = GLRDR.page();
    res.count = GLRDR.count();
    res.pages = [];
    const onPage = (q) => q[2] > 0.5 && q[1] < W - 0.5;
    for (let p = 0; p < res.count; p++) {
      GLRDR.goTo(p);
      await sleep(0);
      const cr = c.getBoundingClientRect();
      const [lines, pics] = items();
      res.pages.push({page: GLRDR.page(), count: GLRDR.count(), col: [cr.top, cr.bottom, cr.left, cr.right],
                      total: [lines.length, pics.length],
                      lines: lines.map((q, i) => [i].concat(q)).filter(onPage),
                      pics: pics.map((q, i) => [i].concat(q)).filter(onPage)});
    }
    GLRDR.goTo(start);
  } else {
    window.scrollTo(0, 0);
    await sleep(60);
    const [lines, pics] = items();
    res.lines = lines;
    res.pics = pics;
    window.scrollTo(0, 1e9);
    await sleep(60);
    res.maxScroll = window.scrollY;
    window.scrollTo(0, 0);
  }
  return res;
})()
"""


def _paged_problems(result: dict, *, covered: tuple, env: tuple) -> list:
    """Each line / picture fully visible (inside the WebView, the column box and clear of the bars) on
    exactly one page; the column box where the desktop geometry less the reported insets puts it."""
    problems = []
    height, width = float(result["H"]), float(result["W"])
    pages = result["pages"]
    if not pages:
        return ["no page measured"]
    totals = {tuple(p["total"]) for p in pages}
    counts = {p["count"] for p in pages}
    if len(totals) != 1 or counts != {len(pages)} or [p["page"] for p in pages] != list(range(len(pages))):
        problems.append(f"the page set changed while paging: totals {totals}, counts {counts}")
    line_total, pic_total = pages[0]["total"]
    if line_total < 20:
        problems.append(f"only {line_total} lines measured")
    col_top, col_bottom, col_left, col_right = pages[0]["col"]
    if abs(col_top - (20 + env[0])) > 0.6 or abs(col_bottom - (height - 16 - env[1])) > 0.6:
        problems.append(f"column box {col_top:.1f}..{col_bottom:.1f}, expected {20 + env[0]}..{height - 16 - env[1]}")
    top = max(0.0, covered[0], col_top)
    bottom = min(height, height - covered[1], col_bottom)
    left, right = max(0.0, col_left), min(width, col_right)
    for kind, key, total in (("line", "lines", line_total), ("picture", "pics", pic_total)):
        full: collections.Counter = collections.Counter()
        cut = []
        for page in pages:
            for ident, x0, x1, y0, y1 in page[key]:
                if x0 >= left - 0.5 and x1 <= right + 0.5 and y0 >= top - SLACK and y1 <= bottom + SLACK:
                    full[ident] += 1
                else:
                    cut.append(f"page {page['page'] + 1}: {kind} {ident} at x {x0:.1f}..{x1:.1f}, y {y0:.1f}..{y1:.1f}")
        never = [i for i in range(total) if full[i] == 0]
        twice = [i for i in range(total) if full[i] > 1]
        if cut:
            problems.append(f"{len(cut)} {kind}s cut or hidden (visible {top:.1f}..{bottom:.1f}): " + "; ".join(cut[:4]))
        if never:
            problems.append(f"{len(never)} of {total} {kind}s never fully shown on any page: {never[:8]}")
        if twice:
            problems.append(f"{len(twice)} {kind}s shown on two pages: {twice[:8]}")
    return problems


def _scroll_problems(result: dict, *, covered: tuple) -> list:
    """Every line can be scrolled fully into view clear of the bars; so can every picture, or (a picture
    taller than the screen, which the scroll layout does not shrink, as on desktop) both of its ends."""
    problems = []
    height, width = float(result["H"]), float(result["W"])
    top, bottom = covered[0], height - covered[1]
    maximum = float(result["maxScroll"])
    if len(result["lines"]) < 20:
        problems.append(f"only {len(result['lines'])} lines measured")
    for kind, key in (("line", "lines"), ("picture", "pics")):
        bad = []
        for ident, (x0, x1, y0, y1) in enumerate(result[key]):
            if kind == "picture" and y1 - y0 > bottom - top:  # its top edge (scroll 0), its bottom edge (the end)
                reachable = y0 >= top - SLACK and y1 - maximum <= bottom + SLACK
            else:  # a scroll offset in 0..maximum shows it whole
                reachable = max(0.0, y1 - bottom) <= min(maximum, y0 - top) + SLACK
            if x0 < -0.5 or x1 > width + 0.5 or not reachable:
                bad.append(f"{kind} {ident} at y {y0:.1f}..{y1:.1f} (scroll 0..{maximum:.0f}, visible {top}..{bottom})")
        if bad:
            problems.append(f"{len(bad)} {kind}s never fully visible: " + "; ".join(bad[:4]))
    return problems


def _check(result: dict, *, box: tuple, covered: tuple, env: tuple, typography: Optional[tuple]) -> list:
    problems = []
    if (float(result["W"]), float(result["H"])) != (box[2], box[3]):
        problems.append(f"the page's viewport {result['W']}x{result['H']} is not the WebView's box {box[2]}x{box[3]}")
    if abs(result["envTop"] - env[0]) > 0.5 or abs(result["envBottom"] - env[1]) > 0.5:
        problems.append(f"the page reports insets {result['envTop']}/{result['envBottom']}, expected {env}")
    if typography is not None:
        size, spacing, _family = typography
        px = int(round(size * 96 / 72))
        if abs(result["fontSize"] - px) > 0.5 or abs(result["lineHeight"] - px * spacing) > 1.0:
            problems.append(f"typography {result['fontSize']}px / {result['lineHeight']}px is not {size}pt x{spacing}")
    if result["broken"]:
        problems.append(f"pictures that did not load: {result['broken']}")
    if result["paged"]:
        problems += _paged_problems(result, covered=covered, env=env)
    else:
        problems += _scroll_problems(result, covered=covered)
    return problems


# =====================================================================================
# Driving the app
# =====================================================================================


def _chrome_engine(phone) -> Any:
    """test_devfix_issue1's Chrome engine with test_reader_layout's ``cdp`` command, as the WebView's
    web engine (the page events reach the app as console messages)."""
    found = D1.ChromeEngine.locate()
    if found is None:
        pytest.skip("headless Chrome and node >= 22 are needed for the page measurements")
    phone.monkeypatch.setattr(D1, "_CDP_BRIDGE_JS", LAYOUT._bridge_with_cdp(D1._CDP_BRIDGE_JS))
    engine = D1.ChromeEngine(phone.tmp / "chrome", *found)
    try:
        engine.start()
    except RuntimeError as exc:
        pytest.skip(str(exc))
    phone.closers.append(engine.stop)
    engine.cdp = lambda method, params: engine.call("cdp", method=method, params=params)
    engine.measure = lambda: engine.call("eval", timeout=120.0, expr=_MEASURE_JS)
    return engine


async def _size_webview(client, box: tuple, env: tuple, dpr: float) -> None:
    """The WebView's viewport is its box (Chrome device metrics); ``env`` is what it reports as insets."""
    await client.do("cdp", "Emulation.setDeviceMetricsOverride",
                    {"width": int(box[2]), "height": int(box[3]), "deviceScaleFactor": dpr, "mobile": True})
    top, bottom = env
    await client.do("cdp", "Emulation.setSafeAreaInsetsOverride", {"insets": {
        "top": top, "topMax": top, "bottom": bottom, "bottomMax": bottom, "left": 0, "leftMax": 0,
        "right": 0, "rightMax": 0}})


async def _turn(turns, client, action, index: int, **kwargs: Any) -> dict:
    """``Turns.turn`` (exactly one ``load_request``, the page's ready event handled, the page on screen
    is chapter ``index``) with what the WebView and the Reader saw when it fails."""
    try:
        return await turns.turn(action, index, **kwargs)
    except AssertionError as exc:
        screen = turns.screen
        events = [(e.type, e.chapter, e.page, e.count, e.doc, e.seq) for e in screen.events[-12:]]
        raise AssertionError(
            f"{exc}\n  loads: {client.loads[-3:]}\n  console: {[t[:160] for _i, t in client.console[-12:]]}"
            f"\n  events: {events}\n  doc {screen.current_doc} chapter {screen.deduper.chapter} dropped "
            f"{screen.deduper.dropped} flavor {screen.session.flavor} layout {screen.layout} state {screen.state}"
        ) from exc


async def _press(phone, screen: Any, mode: str) -> None:
    """A SegmentedButton tap as Flutter sends it: ``selected`` patched first, then the ``change`` event."""
    buttons = screen.chrome.mode_buttons
    phone.session.apply_patch(buttons._i, {"selected": [mode]})
    await phone.session.dispatch_event(buttons._i, "change", [mode])


class _HeldEvents(list):
    """The Reader keeps its last 200 events and ``Turns`` tells new events from old ones by ``id()``,
    which Python reuses once a trimmed event is freed (measuring pages through ``GLRDR.goTo`` posts many
    page events): every trimmed event stays alive here."""

    def __init__(self, items: Any = ()) -> None:
        super().__init__(items)
        self.trimmed: list = []

    def __delitem__(self, key: Any) -> None:
        self.trimmed.extend(self[key] if isinstance(key, slice) else [self[key]])
        super().__delitem__(key)


async def _open_on(phone, client, size: tuple, insets: tuple, previous: Any = None) -> tuple:
    """The phone's size and system insets, then the Reader on chapter 1: (screen, View, Turns)."""
    if previous is not None:
        await D1._leave(phone, previous)
    phone.session.apply_page_patch({"width": size[0], "height": size[1]})
    phone.page.media = _media(insets)
    screen = await D1._open(phone, chapter=0)
    screen.events = _HeldEvents(screen.events)
    view = phone.page.views[-1]
    assert view is screen.view and screen.edge_to_edge and screen.renderer == "webview"
    turns = D1.Turns(phone, client, screen)
    await turns.page_ready(set(), 0)
    return screen, view, turns


async def _apply_type(phone, client, screen, turns, typography: tuple, index: int) -> None:
    """Aa › font size / line spacing / family: a live ``applyStyle`` on the page on screen, or (switching
    to or from the book's own CSS) a reload of the chapter; either way, what the owner then sees."""
    from glossarion_mobile.ui.reader.aa_sheet import SCOPE_ALL

    size, spacing, family = typography
    changes = {"font_size": size, "line_spacing": spacing, "font_family": family}
    reload = (family.strip() == "Embedded CSS") != screen.settings.embedded_css
    if reload:
        await _turn(turns, client, lambda: screen._apply_settings(changes, SCOPE_ALL, save=False), index)
        return
    scripts = len(client.scripts)
    screen._apply_settings(changes, SCOPE_ALL, save=False)
    assert await D1._wait(lambda: any("applyStyle" in s for s in client.scripts[scripts:]), 10), \
        "the Aa change never reached the page"


async def _run_screen(phone, client, width: int, height: int, dpr: float) -> tuple:
    """Every scenario on one phone size: (failures, stats)."""
    from glossarion_mobile.ui.reader import model as rm
    from glossarion_mobile.ui.reader.aa_sheet import SCOPE_ALL

    failures: list = []
    stats = collections.Counter()
    screen = None
    for name, insets, env in SCENARIOS:
        screen, view, turns = await _open_on(phone, client, (width, height), insets, screen)
        _assert_page_layers(screen, view)
        box = _webview_box(view, screen.webview, (width, height), insets)
        expected = (float(insets[0]), float(insets[1]), float(width - insets[0] - insets[2]),
                    float(height - insets[1] - insets[3]))
        assert box == expected, (name, box, expected)
        assert screen._page_size() == box[2:], "the Reader's page area (native page, selection chips)"
        covered = _bars_over(box, (width, height), insets)
        assert covered == (0.0, 0.0), (name, covered)
        await _size_webview(client, box, env, dpr)
        index = 0

        async def measure(label: str, typography: Optional[tuple]) -> None:
            result = await client.do("measure")
            stats["documents"] += 1
            stats["pages"] += len(result.get("pages") or [1])
            stats["lines"] += (result["pages"][0]["total"][0] if result["paged"] and result["pages"]
                               else len(result.get("lines") or []))
            for problem in _check(result, box=box, covered=covered, env=env, typography=typography):
                failures.append(f"{width}x{height} {name}: {label}: {problem}")

        for typography in TYPOGRAPHY:
            label = f"{typography[0]}pt x{typography[1]} {typography[2]}"
            await _apply_type(phone, client, screen, turns, typography, index)
            await measure(f"chapter {index + 1}, {label} (Aa)", typography)
            index = 1 - index
            await _turn(turns, client, lambda: screen.go_chapter(index), index)
            await measure(f"chapter {index + 1}, {label} (loaded)", typography)
        assert screen.settings.font_size == DEFAULT_TYPE[0] and screen.settings.font_family == DEFAULT_TYPE[2]
        for mode, kind in ((ORIGINAL, "RAW"), (BILINGUAL, "EN"), (TRANSLATED, "EN")):
            shown = await _turn(turns, client, lambda: _press(phone, screen, mode), index, mode=kind)
            assert screen.session.flavor == mode
            if mode == BILINGUAL:
                assert ("RAW", index) in D1._marks(shown["text"]) and ("EN", index) in D1._marks(shown["text"])
            if mode != TRANSLATED:
                await measure(f"chapter {index + 1}, {mode}", DEFAULT_TYPE)
        await _turn(turns, client, lambda: screen._apply_settings({"layout": rm.LAYOUT_SCROLL}, SCOPE_ALL, save=False), index)
        await measure(f"chapter {index + 1}, scroll layout", DEFAULT_TYPE)
        index = 1 - index
        await _turn(turns, client, lambda: screen.go_chapter(index), index)
        await measure(f"chapter {index + 1}, scroll layout", DEFAULT_TYPE)
        await _turn(turns, client, lambda: screen._apply_settings({"layout": rm.LAYOUT_SINGLE}, SCOPE_ALL, save=False), index)
    if screen is not None:
        await D1._leave(phone, screen)
    return failures, stats


@needs_msgpack
@pytest.mark.parametrize("width,height,dpr", SCREENS, ids=[f"{w}x{h}" for w, h, _ in SCREENS])
def test_owner_report_10_every_reader_line_is_fully_visible_exactly_once(phone, width, height, dpr):
    """The owner's complaint on the real app: whatever the phone's bars and the WebView's inset reporting,
    every line of every Reader page (Aa settings, both chapters, Original, Bilingual, scroll) is fully
    visible, clear of the status and navigation bars, on exactly one page."""
    async def scenario():
        phone.book = _owner_book(phone.tmp / "books")
        await phone.start("android")
        assert str(getattr(phone.page.platform, "value", phone.page.platform)) == "android"
        _assert_isolated(phone)
        engine = _chrome_engine(phone)
        client = D1._attach_webview(phone, engine)
        try:
            failures, stats = await _run_screen(phone, client, width, height, dpr)
            print(f"[devfix-issue10] {width}x{height}: {dict(stats)}")
            assert stats["documents"] == len(SCENARIOS) * (2 * len(TYPOGRAPHY) + 4)
            assert stats["pages"] > 10 * stats["documents"]
            assert engine.exceptions == [], engine.exceptions
            assert failures == [], "\n".join(failures[:40])
        finally:
            await phone.stop()

    asyncio.run(scenario())


@needs_msgpack
def test_owner_report_10_text_leading_into_an_illustration_is_not_cut(phone):
    """Novels put an illustration right after a paragraph. The shared page shell pulls that paragraph into
    the full-page illustration box (``reader_doc`` ``.full-page-img``: break-inside avoid, overflow hidden,
    the picture up to the column height less 64 px), so on a phone-width column a paragraph of a few lines
    plus a portrait illustration is taller than the column: its lines must still be fully visible.

    The box predates device-fix batch 2: the U8 page shell, and the desktop paged page at desktop window
    sizes, push the picture past the column bottom the same way. On a phone the narrow column also reaches
    the text (the paragraph's last line at 360x800 18 pt), so this case belongs to "never clipped"."""
    async def scenario():
        phone.book = _owner_book(phone.tmp / "books", plate=(1000, 1400), lead_in=_EN * 3)
        await phone.start("android")
        _assert_isolated(phone)
        engine = _chrome_engine(phone)
        client = D1._attach_webview(phone, engine)
        failures: list = []
        documents = 0
        insets = (0, STATUS_BAR, 0, NAV_BAR)
        try:
            screen = None
            for width, height, dpr in SCREENS:
                screen, view, turns = await _open_on(phone, client, (width, height), insets, screen)
                box = _webview_box(view, screen.webview, (width, height), insets)
                covered = _bars_over(box, (width, height), insets)
                await _size_webview(client, box, (0, 0), dpr)
                index = 0
                for typography in (DEFAULT_TYPE, (18, 1.8, "Embedded CSS")):
                    for _chapter in (0, 1):
                        index = 1 - index
                        await _turn(turns, client, lambda: screen.go_chapter(index), index)
                        await _apply_type(phone, client, screen, turns, typography, index)
                        result = await client.do("measure")
                        documents += 1
                        assert sum(len(p["pics"]) for p in result["pages"]) >= 1
                        for problem in _check(result, box=box, covered=covered, env=(0, 0), typography=typography):
                            failures.append(f"{width}x{height}, chapter {index + 1}, {typography[0]}pt x{typography[1]}: "
                                            f"{problem}")
            assert documents == 2 * len(SCREENS) * 2
            assert failures == [], "\n".join(failures[:20])
        finally:
            await phone.stop()

    asyncio.run(scenario())


# =====================================================================================
# Controls: the measurement sees what the owner saw on the U8 build
# =====================================================================================


def _base_reader_doc(tmp_path: Path, base: str) -> Optional[types.ModuleType]:
    """``src/reader_doc.py`` as it was at ``base`` (None when git or that commit is not in this clone)."""
    git = shutil.which("git")
    if git is None:
        return None
    try:
        source = subprocess.run([git, "-C", str(REPO_DIR), "show", f"{base}:src/reader_doc.py"],
                                capture_output=True, timeout=60)
    except Exception:
        return None
    if source.returncode != 0 or not source.stdout:
        return None
    path = tmp_path / f"reader_doc_{base}.py"
    path.write_bytes(source.stdout)
    spec = importlib.util.spec_from_file_location(f"_glossarion_reader_doc_{base}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@needs_msgpack
def test_the_measurement_sees_the_owners_u8_clipping(phone):
    """Without the page's SafeArea (U8) the 3-button navigation bar covers the last lines; with the U8 page
    shell a WebView that reports insets cuts lines off at the bottom of the screen, also inside the
    SafeArea. The same measurement passes on the fixed Reader (the test above)."""
    async def scenario():
        phone.book = _owner_book(phone.tmp / "books")
        await phone.start("android")
        _assert_isolated(phone)
        engine = _chrome_engine(phone)
        client = D1._attach_webview(phone, engine)
        width, height, dpr = SCREENS[0]
        insets = (0, STATUS_BAR, 0, NAV_BAR)
        try:
            screen, view, turns = await _open_on(phone, client, (width, height), insets)
            # U8: the page filled the screen under the bars, its WebView reporting no inset
            full = _webview_box(view, screen.webview, (width, height), insets, safe_areas=False)
            assert full == (0.0, 0.0, float(width), float(height))
            covered = _bars_over(full, (width, height), insets)
            assert covered == (float(STATUS_BAR), float(NAV_BAR))
            await _size_webview(client, full, (0, 0), dpr)
            await _turn(turns, client, lambda: screen.go_chapter(1), 1)
            problems = _check(await client.do("measure"), box=full, covered=covered, env=(0, 0), typography=None)
            assert any("cut or hidden" in p for p in problems), problems
            print("[devfix-issue10] U8 page under the bars:", problems[:2])
            base = next((m for m in (_base_reader_doc(phone.tmp, b) for b in BASES) if m is not None), None)
            if base is None:
                print("[devfix-issue10] the base commit is not in this clone: U8 page shell control skipped")
                return
            import reader_doc

            assert base._MOBILE_PAGED_CSS != reader_doc._MOBILE_PAGED_CSS
            phone.monkeypatch.setattr(reader_doc, "_MOBILE_PAGED_CSS", base._MOBILE_PAGED_CSS)
            index = 1
            for box in (full, _webview_box(view, screen.webview, (width, height), insets)):
                await _size_webview(client, box, (STATUS_BAR, NAV_BAR), dpr)
                covered = _bars_over(box, (width, height), insets)
                problems = []
                for typography in TYPOGRAPHY[:2] + [DEFAULT_TYPE]:  # a few pages of each (where lines fall varies)
                    index = 1 - index
                    await _turn(turns, client, lambda: screen.go_chapter(index), index)
                    await _apply_type(phone, client, screen, turns, typography, index)
                    problems += _check(await client.do("measure"), box=box, covered=covered,
                                       env=(STATUS_BAR, NAV_BAR), typography=typography)
                assert any(p.startswith("column box") for p in problems), problems
                assert any("cut or hidden" in p for p in problems), problems
                print(f"[devfix-issue10] U8 page shell, WebView {box[2]:.0f}x{box[3]:.0f}:",
                      [p[:150] for p in problems if "cut or hidden" in p][:2])
        finally:
            await phone.stop()

    asyncio.run(scenario())


# =====================================================================================
# The desktop page is unchanged
# =====================================================================================

_DESKTOP_BODY = (
    "<h1>Chapter 1</h1><p>The rain kept falling. <em>Lanterns</em> swung.</p>"
    "<blockquote><p>A quote.</p></blockquote>"
    '<div class="full-page-img"><img src="/img/plate.png" alt=""/></div>'
    "<p>비가 계속 내렸다.</p><p><a href='chapter0002.xhtml'>next</a> <code>x</code></p>"
)


def _desktop_pages(module: Any) -> list:
    themes = range(len(module.READER_THEMES))
    combos = itertools.product(themes, ("Embedded CSS", "Serif", "Sans-serif"), (12, 14, 24), (1.0, 1.8),
                               ((True, 1), (True, 2), (False, 1)), ("", "p { color: red; }"), (False, True))
    pages = []
    for theme, family, size, spacing, (paginated, spread), css, flag in combos:
        pages.append(module.wrap_reader_html(_DESKTOP_BODY, theme, font_family=family, font_size=size,
                                             line_spacing=spacing, paginated=paginated, spread_pages=spread,
                                             embedded_css=css, show_raw=flag, workspace_mode=flag))
    return pages


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    for name in ("GLOSSARION_LIBRARY_DIR", "OUTPUT_DIRECTORY", "HOME", "USERPROFILE", "APPDATA", "GLOSSARION_DATA_DIR"):
        folder = tmp_path / "env" / name.lower()
        folder.mkdir(parents=True)
        monkeypatch.setenv(name, str(folder))
    monkeypatch.setenv("GLOSSARION_HTTP_LOG", "0")
    config = SRC_DIR / "config.json"
    before = D1._md5(config)
    yield tmp_path
    assert D1._md5(config) == before, "src/config.json changed"


@pytest.mark.skipif(not D1._has("reader_doc"), reason="the shared reader core is not importable here")
def test_desktop_reader_page_never_depends_on_the_mobile_shell(isolated, monkeypatch):
    """The fix lives in the mobile-only shell constants: replacing every one of them changes no desktop
    page (paged / spread / scroll, every theme, family, size and spacing)."""
    import reader_doc

    desktop = _desktop_pages(reader_doc)
    for name in ("_MOBILE_PAGED_CSS", "_MOBILE_SCROLL_CSS", "_MOBILE_COMMON_CSS", "_MOBILE_VIEWPORT", "_MOBILE_BRIDGE_JS"):
        monkeypatch.setattr(reader_doc, name, f"/* {name} replaced */")
    assert _desktop_pages(reader_doc) == desktop
    for page in desktop:
        assert "env(" not in page and "100dvh" not in page and "GLRDR" not in page
    paged = [p for p in desktop if "id='columns'" in p]
    assert paged and all("padding: 10px 0 26px 0" in p and "(window.innerHeight - 36) + 'px'" in p for p in paged)


@pytest.mark.skipif(not D1._has("reader_doc"), reason="the shared reader core is not importable here")
def test_desktop_reader_page_bytes_are_those_of_the_batch_base(isolated):
    """Byte for byte the desktop pages the batch base (and the owner's U8 build) produced."""
    import reader_doc

    bases = [(b, m) for b, m in ((b, _base_reader_doc(isolated, b)) for b in BASES) if m is not None]
    if not bases:
        pytest.skip(f"none of the base commits {BASES} is in this clone")
    current = _desktop_pages(reader_doc)
    assert len(current) > 500
    for base, module in bases:
        assert _desktop_pages(module) == current, f"the desktop reader page differs from {base}'s"
