"""Owner report (2026-10-20): "epub reader is very slow to load, and the pages lack transition when
switching between them. I want the page turn transition".

Load speed
  Opening a book parsed every chapter with BeautifulSoup only to read its title
  (``reader_doc.EpubLoaderMixin``): seconds on a desktop for a 300-chapter novel, far more on a
  phone, holding the GIL the UI loop needs. ``_fast_chapter_title`` / ``_fast_heading_title`` read a
  plain ``<title>`` (or, as ebooklib re-serialises chapters with an empty ``<head/>``, the first plain
  h1-h3) and answer None whenever the markup could make the soup see something else. Pinned here:
  the same titles as the soup, no soup for plain chapters, only the first chapter converted before
  the first page shows, the neighbour chapter prepared in the background (``DocumentBuilder.prefetch``
  and the native page's blocks) and used by the turn.

Page turns
  The paged page slides: the shared shell's ``goTo`` scrolls ``#columns`` to the next / previous page
  over ``turn.ms`` (rAF, ease-out); at a chapter edge the page leaves to the other side when a
  neighbour exists, and the neighbour chapter's page enters from the side it was turned from. The
  native reader slides ``page_slot`` on chapter turns. Aa > Layout "Page turn animation" and Reduce
  motion switch it off (``turn.ms`` 0; the page also honours prefers-reduced-motion).

Run from src/mobile (mobile venv):
    python -m pytest -p no:cacheprovider -W ignore tests_host/test_reader_turns.py
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import re
import subprocess
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location("_glossarion_reader_tests_turns",
                                               Path(__file__).with_name("test_reader.py"))
R = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(R)

cores = R.cores  # the real reader cores, away from the user's Library / caches
novel = R.novel  # raw EPUB + an in-progress workspace (test_reader)
rm = R.rm
bridge = R.bridge
needs_flet = R.needs_flet
needs_node = R.needs_node


# =====================================================================================
# Load speed
# =====================================================================================


def _soup_title(content: str):
    """What the loader's BeautifulSoup pass decides: (title before truncation, page has text)."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(content, "html.parser")
    text = soup.get_text(strip=True)
    title = None
    tag = soup.find("title")
    if tag and tag.string:
        title = tag.string.strip()
    if not title:
        for heading in soup.find_all(["h1", "h2", "h3"]):
            found = heading.get_text(strip=True)
            if found:
                title = found
                break
    return title, bool(text)


_HEAD = "<?xml version='1.0' encoding='utf-8'?>\n<!DOCTYPE html>\n<html xmlns='http://www.w3.org/1999/xhtml'>"
TITLE_CASES = [
    # (markup, fast answer expected: a title or None = "parse instead")
    (_HEAD + "<head><title>제1화 은빛 숲</title></head><body><p>x</p></body></html>", "제1화 은빛 숲"),
    (_HEAD + "<head><title>\n  Tom &amp; Jerry&#33; </title></head><body></body></html>", "Tom & Jerry!"),
    (_HEAD + "<head/><body><h1>제2화 아르덴</h1><p>x</p></body></html>", "제2화 아르덴"),
    (_HEAD + "<head/><body><h2 class=\"c\"><span>Chapter 3</span><br/> <b>Fire</b></h2></body></html>",
     "Chapter 3Fire"),
    (_HEAD + "<head/><body><p>intro</p><h3 id='x'>Third &lt;level&gt;</h3><h1>later</h1></body></html>",
     "Third <level>"),
    # the soup decides: a comment / script / style before, tags or comments inside, empty or odd titles
    (_HEAD + "<head><!-- <title>Old</title> --><title>New</title></head><body/></html>", None),
    (_HEAD + "<head><title>A<!--x-->B</title></head><body/></html>", None),
    (_HEAD + "<head><title><b>Bold</b></title></head><body/></html>", None),
    (_HEAD + "<head><title>   </title></head><body><h1>Heading</h1></body></html>", None),
    (_HEAD + "<head><title/></head><body><h1>Heading</h1></body></html>", None),
    (_HEAD + "<head><script>var t='<title>x</title>';</script><title>T</title></head></html>", None),
    (_HEAD + "<head><style>p{}</style></head><body><h1>Styled</h1></body></html>", None),
    (_HEAD + "<head/><body><h1></h1><h2>Second</h2></body></html>", None),
    (_HEAD + "<head/><body><h1><!-- c -->Commented</h1></body></html>", None),
    (_HEAD + "<head/><body><h1><ruby>漢<rt>kan</rt></ruby></h1></body></html>", None),
    (_HEAD + "<head/><body><![CDATA[<h1>no</h1>]]><h1>Yes</h1></body></html>", None),
    (_HEAD + "<head/><body><p>No heading at all</p></body></html>", None),
    (_HEAD + "<head/><body><h1 title='a>b'>Quoted</h1></body></html>", None),
]


@pytest.mark.parametrize("markup, expected", TITLE_CASES)
def test_fast_titles_agree_with_the_soup_or_defer_to_it(cores, markup, expected):
    fast = cores._fast_chapter_title(markup) or cores._fast_heading_title(markup)
    assert fast == expected
    if fast is not None:
        assert _soup_title(markup) == (fast, True)


def _count_soups(monkeypatch) -> list:
    import bs4

    made: list = []
    real = bs4.BeautifulSoup

    class CountingSoup(real):
        def __init__(self, *args, **kwargs):
            made.append(1)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(bs4, "BeautifulSoup", CountingSoup)
    return made


def test_opening_an_epub_reads_titles_without_a_soup_per_chapter(cores, novel, monkeypatch, tmp_path):
    """The chapter list (titles, kept pages) is the soup's, without parsing a single plain chapter."""
    made = _count_soups(monkeypatch)
    fast = cores.load_epub_chapters(str(novel.raw), use_cache=False)
    assert fast[0] and made == []
    monkeypatch.setattr(cores, "_fast_chapter_title", lambda content: None)
    monkeypatch.setattr(cores, "_fast_heading_title", lambda content: None)
    monkeypatch.setattr(cores, "_EPUB_CACHE_DIR_OVERRIDE", str(tmp_path / "slowcache"))
    (tmp_path / "slowcache").mkdir()
    slow = cores.load_epub_chapters(str(novel.raw), use_cache=False)
    assert len(made) >= len(slow[0])  # the soup pass parses every chapter
    assert [title for title, _html in fast[0]] == [title for title, _html in slow[0]]
    assert fast[0] == slow[0] and fast[2] == slow[2]


@needs_flet
def test_first_page_converts_only_its_chapter_and_prefetches_the_next(novel, monkeypatch):
    """The first page shows after converting one chapter; the next one is prepared in the background
    and the turn to it uses that (no second conversion)."""
    from glossarion_mobile.ui.reader import reader_view as rv

    converted: list = []
    real_blocks = rv.build_native_blocks

    def counting_blocks(session, index):
        converted.append(index)
        return real_blocks(session, index)

    monkeypatch.setattr(rv, "build_native_blocks", counting_blocks)
    monkeypatch.setattr(rv, "PREFETCH_DELAY", 0.01)

    async def scenario():
        screen = R._screen(novel, route="/reader/ab12cd34ef56?ch=1")
        await screen.open()
        assert screen.index == 1 and converted == [1]  # one chapter, not the book
        assert screen.session.count > 2
        for _ in range(100):
            if screen._native_prefetch:
                break
            await asyncio.sleep(0.02)
        assert converted == [1, 2] and list(screen._native_prefetch)[0][0] == 2
        await screen.go_chapter(2, enter=1)
        assert screen.index == 2 and converted.count(2) == 1  # the turn used the prefetched chapter
        screen.dispose()

    asyncio.run(scenario())


def test_document_builder_prefetch_warms_the_next_chapter(novel):
    from glossarion_mobile.ui.reader.document import DocumentBuilder

    session = R._overlay_session(novel)
    builder = DocumentBuilder(session, lambda key, source=None: "/t/img/x")
    doc = builder.document(session.flavor)

    def processed() -> int:
        return len(getattr(doc, "_processed_html_cache", None) or {})

    before = processed()
    assert builder.prefetch(1, rm.LAYOUT_SINGLE) and builder.prefetched == [1] and processed() == before + 1
    builder.build(1, settings=rm.ReaderSettings(), layout=rm.LAYOUT_SINGLE, theme=R.DESKTOP_THEMES[0],
                  doc_id="d1", event_url="/t/__ev")
    assert processed() == before + 1  # the page wrapped the cached body (no second conversion)
    assert not builder.prefetch(1, rm.LAYOUT_ALL) and not builder.prefetch(session.count, rm.LAYOUT_SINGLE)
    builder.close()
    session.close()


# =====================================================================================
# Page turns: the page (shared shell + extras) under node
# =====================================================================================

_NODE_TURN_HARNESS = r"""
const vm = require('vm');
const SCRIPTS = %(scripts)s;
function makeEnv(opts) {
  const handlers = {}, winHandlers = {}, posted = [], timers = [], frames = new Map(), bodyLog = [];
  let fid = 0;
  const columns = {clientWidth: 400, scrollWidth: opts.scrollWidth, scrollLeft: 0, style: {},
                   getBoundingClientRect() { return {left: 0, top: 0, width: 400}; },
                   querySelectorAll() { return []; }};
  const bodyStyle = new Proxy({}, {set(t, k, v) { t[k] = v; bodyLog.push(k + '=' + v); return true; }});
  const body = {scrollHeight: 1000, offsetWidth: 400, style: bodyStyle};
  const elements = {};
  const document = {
    readyState: 'complete', head: {appendChild() {}}, body, documentElement: {scrollHeight: 1000, clientWidth: 400},
    getElementById(id) { return id === 'columns' ? columns : (elements[id] || null); },
    createElement(tag) { const el = {tagName: tag.toUpperCase(), id: '', textContent: ''};
      return new Proxy(el, {set(t, k, v) { t[k] = v; if (k === 'id') { elements[v] = t; } return true; }}); },
    querySelectorAll() { return []; }, getElementsByName() { return []; },
    addEventListener(name, fn) { (handlers[name] = handlers[name] || []).push(fn); },
  };
  const sandbox = {document, location: {hash: opts.hash || ''}, innerWidth: 400, innerHeight: 800, scrollY: 0,
    console: {log(s) { if (String(s).startsWith('GLRDR:')) { posted.push(JSON.parse(String(s).slice(6))); } }},
    setTimeout(fn) { timers.push(fn); return timers.length; }, clearTimeout() {},
    requestAnimationFrame(fn) { fid += 1; frames.set(fid, fn); return fid; },
    cancelAnimationFrame(id) { frames.delete(id); },
    getSelection() { return {toString() { return ''; }, rangeCount: 0, removeAllRanges() {}}; },
    scrollTo() {}, scrollBy() {},
    addEventListener(name, fn, capture) { (winHandlers[name] = winHandlers[name] || []).push({fn, capture: !!capture}); }};
  if (opts.reduce) { sandbox.matchMedia = (q) => ({matches: q.indexOf('reduce') >= 0}); }
  sandbox.window = sandbox;
  vm.createContext(sandbox);
  for (const code of SCRIPTS) { vm.runInContext(code, sandbox); }
  const run = () => { let guard = 0; while (timers.length && guard++ < 200) { timers.shift()(); } };
  const frame = (ts) => { const fs = [...frames.values()]; frames.clear(); fs.forEach(f => f(ts)); };
  const click = (x) => {
    const ev = {clientX: x, clientY: 100, target: {nodeType: 1, tagName: 'P', parentNode: null, getAttribute() { return null; }},
                stopped: false, preventDefault() {}, stopPropagation() { this.stopped = true; }};
    for (const h of (winHandlers.click || [])) { if (h.capture) { h.fn(ev); } }
    if (!ev.stopped) { for (const fn of (handlers.click || [])) { fn(ev); } }
  };
  const key = (k) => { for (const fn of (handlers.keydown || [])) { fn({key: k, preventDefault() {}}); } };
  return {G: sandbox.GLRDR, posted, columns, run, frame, frames, click, key, bodyLog};
}
const out = {};
// 9 pages of 400 px + the 1 px gap; this page was turned to from the previous chapter (enter 1)
let env = makeEnv({scrollWidth: 3600});
env.run();
out.enterLog = env.bodyLog.slice();
out.turn = env.G.turn();
env.click(390);                       // next: the page index moves at once, the columns slide
out.tap = {page: env.G.page(), left: env.columns.scrollLeft, pending: env.frames.size,
           event: env.posted[env.posted.length - 1]};
env.frame(1000); out.start = env.columns.scrollLeft;
env.frame(1130); out.mid = env.columns.scrollLeft;
env.frame(1260); out.end = env.columns.scrollLeft; out.endPending = env.frames.size;
env.click(5);                         // prev slides back
env.frame(2000); env.frame(2130); out.prevMid = env.columns.scrollLeft; env.frame(2400);
out.prevEnd = {page: env.G.page(), left: env.columns.scrollLeft};
env.click(390); env.frame(3000); env.frame(3050);  // a second turn while the first still slides
out.quickFrom = env.columns.scrollLeft;
env.click(390); out.quickPage = env.G.page();
env.frame(3100); env.frame(3400); out.quickEnd = env.columns.scrollLeft;
env.key('ArrowRight'); env.frame(4000); env.frame(4300);
out.key = {page: env.G.page(), left: env.columns.scrollLeft};
env.key('ArrowLeft'); env.frame(5000); env.frame(5300); out.keyBack = env.G.page();
env.G.setTurn({ms: 0});              // the switch off: pages change at once
env.click(390); out.off = {page: env.G.page(), left: env.columns.scrollLeft, pending: env.frames.size};
env.G.setTurn({ms: 260});
env.G.go(8); env.bodyLog.length = 0;
env.click(390);                      // past the last page with a next chapter: the page leaves to the left
out.leave = {log: env.bodyLog.slice(), event: env.posted[env.posted.length - 1]};
env.G.setTurn({next: false}); env.bodyLog.length = 0;
env.click(390); out.lastChapter = env.bodyLog.slice();
// prefers-reduced-motion: no slide, no entrance
env = makeEnv({scrollWidth: 3600, reduce: true});
env.run(); out.reduceEnter = env.bodyLog.slice();
env.click(390); out.reduce = {page: env.G.page(), left: env.columns.scrollLeft, pending: env.frames.size};
console.error(JSON.stringify(out));
"""


def _page_scripts(built) -> list:
    return [m.group(1) for m in re.finditer(r'<script nonce="[^"]+">(.*?)</script>', built.html, re.S)]


@needs_node
def test_paged_page_slides_on_turns_and_enters_from_the_turned_side(tmp_path, novel):
    from glossarion_mobile.ui.reader.document import DocumentBuilder

    session = R._overlay_session(novel)
    builder = DocumentBuilder(session, lambda key, source=None: "/t/img/x")
    turn = bridge.turn_config(rm.PAGE_TURN_MS, enter=1, has_next=True, has_prev=True)
    built = builder.build(1, settings=rm.ReaderSettings(), layout=rm.LAYOUT_SINGLE, theme=R.DESKTOP_THEMES[0],
                          doc_id="d1", event_url="/t/__ev", turn=turn)
    assert '"turn":{"ms":260,"enter":1,"next":true,"prev":true}' in built.html
    path = tmp_path / "turns.js"
    path.write_text(_NODE_TURN_HARNESS % {"scripts": json.dumps(_page_scripts(built))}, encoding="utf-8")
    proc = subprocess.run([R.NODE, str(path)], capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stderr.strip().splitlines()[-1])
    # the chapter page enters from the right (it was turned to from the previous chapter)
    assert out["turn"] == {"ms": 260, "enter": 1, "next": True, "prev": True}
    log = out["enterLog"]
    assert "transform=translateX(24%)" in log and "opacity=0" in log
    assert log.index("transform=translateX(24%)") < log.index("transform=translateX(0)")
    assert any(entry.startswith("transition=transform 260ms") for entry in log)
    # next: page 1 at once (the event too), the columns slide from 0 to 401 with an ease-out
    assert out["tap"]["page"] == 1 and out["tap"]["left"] == 0 and out["tap"]["pending"] == 1
    assert out["tap"]["event"]["type"] == "page" and out["tap"]["event"]["reason"] == "next"
    assert out["start"] == 0 and out["mid"] == round(401 * (1 - 0.5 ** 3)) and out["end"] == 401
    assert out["endPending"] == 0
    # prev slides back the other way
    assert 0 < out["prevMid"] < 401 and out["prevEnd"] == {"page": 0, "left": 0}
    # a second turn while sliding continues from where the columns are, no jump back
    assert 0 < out["quickFrom"] < 401 and out["quickPage"] == 2 and out["quickEnd"] == 802
    # the keyboard turns pages like the tap zones
    assert out["key"] == {"page": 3, "left": 1203} and out["keyBack"] == 2
    # the switch off: the page changes at once, no frame pending
    assert out["off"] == {"page": 3, "left": 1203, "pending": 0}
    # the last page with a next chapter: the page slides out to the left and fades, then the edge event
    assert "transform=translateX(-24%)" in out["leave"]["log"] and "opacity=0" in out["leave"]["log"]
    assert out["leave"]["event"]["type"] == "edge" and out["leave"]["event"]["edge"] == "end"
    assert out["lastChapter"] == []  # the book's last chapter: nothing to leave for
    # prefers-reduced-motion: no entrance, no slide
    assert out["reduceEnter"] == [] and out["reduce"] == {"page": 1, "left": 401, "pending": 0}
    builder.close()
    session.close()


def test_turn_config_and_bridge_config():
    assert bridge.turn_config(260, enter=-5, has_next=True) == {"ms": 260, "enter": -1, "next": True, "prev": False}
    assert bridge.turn_config(0)["ms"] == 0 and bridge.turn_config(-3)["ms"] == 0
    cfg = bridge.bridge_config(doc="d1", layout=rm.LAYOUT_SINGLE, turn=bridge.turn_config(260, enter=1))
    assert cfg["turn"]["enter"] == 1
    assert "turn" not in bridge.bridge_config(doc="d1", layout=rm.LAYOUT_SINGLE)
    assert rm.ReaderSettings().page_turn is True and rm.MOBILE_DEFAULTS["page_turn"] is True
    assert rm.resolve_settings(lambda k, d: d, None, {"page_turn": False}).page_turn is False


# =====================================================================================
# Page turns: the Reader (Python side)
# =====================================================================================


@needs_flet
def test_reader_turn_directions_setting_and_reduce_motion(novel, monkeypatch):
    from glossarion_mobile.ui import motion
    from glossarion_mobile.ui.reader import reader_view as rv

    monkeypatch.setattr(rv, "PREFETCH_DELAY", 0.01)
    renders: list = []
    real_render = rv.ReaderScreen.render

    async def recording_render(self, index, **kwargs):
        renders.append((index, kwargs.get("enter", 0)))
        return await real_render(self, index, **kwargs)

    monkeypatch.setattr(rv.ReaderScreen, "render", recording_render)

    async def scenario():
        screen = R._screen(novel, route="/reader/ab12cd34ef56?ch=1")
        await screen.open()
        assert renders[-1] == (1, 0)  # opening: no turn
        assert screen.turn_ms() == rm.PAGE_TURN_MS
        # past the last page: the next chapter enters from the right; before the first: the previous
        # chapter's last page enters from the left
        screen.handle_payload({"type": "edge", "seq": 1, "edge": "end", "chapter": 1})
        await asyncio.sleep(0.3)
        assert renders[-1] == (2, 1) and screen.index == 2
        screen.handle_payload({"type": "edge", "seq": 2, "edge": "start", "chapter": 2})
        await asyncio.sleep(0.3)
        assert renders[-1] == (1, -1) and screen.last_page
        # the chrome's ◀ ▶ turn chapters with the same directions; the chapters list does not slide
        screen.chrome.next_button.on_click(None)
        await asyncio.sleep(0.3)
        assert renders[-1] == (2, 1)
        screen.chrome.prev_button.on_click(None)
        await asyncio.sleep(0.3)
        assert renders[-1] == (1, -1)
        await screen.go_chapter(0)
        assert renders[-1] == (0, 0)
        # the webview page gets its motion in the config: direction, neighbours, duration
        screen.layout = rm.LAYOUT_SINGLE
        assert screen._turn_config(0, 1) == {"ms": 260, "enter": 1, "next": True, "prev": False}
        last = screen.session.count - 1
        assert screen._turn_config(last, -1)["next"] is False and screen._turn_config(last, -1)["prev"] is True
        screen.layout = rm.LAYOUT_ALL
        assert screen._turn_config(0, 1) is None  # Scroll all never turns pages
        screen.layout = rm.LAYOUT_SINGLE
        # Aa > Layout "Page turn animation" off: no motion, no re-render, the page is told
        scripts: list = []

        async def fake_js(script):
            scripts.append(script)
            return True

        screen._js = fake_js
        screen.renderer = "webview"
        count = len(renders)
        screen._apply_settings({"page_turn": False}, "all")
        await asyncio.sleep(0.05)
        assert screen.settings.page_turn is False and screen.turn_ms() == 0
        assert len(renders) == count and scripts == [bridge.js_call("setTurn", {"ms": 0})]
        screen._flush_style()
        assert screen.deps.prefs.get("reader_prefs", {}).get("page_turn") is False
        screen._apply_settings({"page_turn": True}, "all")
        await asyncio.sleep(0.05)
        assert scripts[-1] == bridge.js_call("setTurn", {"ms": 260})
        # Reduce motion: the page on screen follows the switch
        screen.did_show()
        try:
            motion.set_reduce_motion(True)
            await asyncio.sleep(0.05)
            assert screen.turn_ms() == 0 and scripts[-1] == bridge.js_call("setTurn", {"ms": 0})
        finally:
            motion.set_reduce_motion(False)
        await asyncio.sleep(0.05)
        assert scripts[-1] == bridge.js_call("setTurn", {"ms": 260})
        screen.renderer = "native"
        screen.dispose()

    asyncio.run(scenario())


@needs_flet
def test_native_chapter_turn_slides_the_page_in(novel, monkeypatch):
    import flet as ft

    from glossarion_mobile.ui.reader import reader_view as rv

    monkeypatch.setattr(rv, "PREFETCH_DELAY", 0.01)

    async def scenario():
        screen = R._screen(novel, route="/reader/ab12cd34ef56?ch=1")
        await screen.open()
        slot = screen.page_slot
        assert slot.offset == ft.Offset(0, 0) and slot.opacity == 1.0  # shown in place, no motion pending
        states: list = []
        real_set = screen._set_slot_motion

        def recording(native, *, enter=0):
            moved = real_set(native, enter=enter)
            states.append((slot.offset.x if slot.offset else None, slot.opacity,
                           slot.animate_offset.duration if slot.animate_offset else None))
            return moved

        screen._set_slot_motion = recording
        await screen.go_chapter(2, enter=1)
        # off to the right and hidden at once, then slid in over the turn's duration
        assert states == [(rv.SLIDE_SHIFT, 0.0, 1), (0, 1.0, rm.PAGE_TURN_MS)]
        states.clear()
        await screen.go_chapter(1, last=True, enter=-1)
        assert states == [(-rv.SLIDE_SHIFT, 0.0, 1), (0, 1.0, rm.PAGE_TURN_MS)]
        states.clear()
        screen._apply_settings({"page_turn": False}, "all")
        await screen.go_chapter(2, enter=1)
        assert states == [(0, 1.0, 1)]  # off: in place at once
        screen.dispose()

    asyncio.run(scenario())


@needs_flet
def test_aa_sheet_has_the_page_turn_switch():
    from glossarion_mobile.ui.reader.aa_sheet import AaSheet

    emitted: list = []
    sheet = AaSheet.__new__(AaSheet)
    sheet.settings = rm.ReaderSettings(page_turn=False)
    sheet.double_allowed = False
    sheet._emit = emitted.append
    sheet._layout_tab()
    assert sheet.turn_switch.label == "Page turn animation" and sheet.turn_switch.value is False
    sheet.turn_switch.on_change(type("E", (), {"control": type("C", (), {"value": True})()})())
    assert emitted == [{"page_turn": True}]
