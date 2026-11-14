"""The Reader's page images (owner report 2026-11-14: "When tapping an image the first time nothing
happens. And the stupid menu doesnt hide itself, so i have to tap a 2nd time. Also when i try to zoom in
to the image, it creates a silly pop up instead of a proper image view").

Cause: a double tap on a picture reached the page as click, click, dblclick. The shared shell's click
handler (``reader_doc._MOBILE_BRIDGE_JS``) made each click a ``tap`` (the chrome toggled twice: hidden,
then back up), and the extras' dblclick listener then posted ``img``, which opened a BottomSheet
(``sheets.ImageViewer``) over the chrome that was showing again.

Now:

* the page script (``bridge.READER_EXTRAS_JS``, capture phase before the shell) holds a click on a
  picture for the double tap's 300 ms: a second click on it is ``img`` alone, else the held click does
  once what a tap there does on text (tap zones or the chrome). Two fingers that start on a picture and
  spread post ``img`` and never reach the shell (no font size change); a pinch on text still does;
* ``img`` opens the app's full-screen ``MediaViewer`` (black ``ft.View`` on the shell's overlay stack,
  ✕ Close, pinch zoom and pan, double tap to zoom and back) through ``screens.files.open_media_viewer``;
  ✕ and Android / iOS back close it and the Reader keeps its page and chrome;
* the native fallback page opens the same viewer on a double-tapped image.

The page script runs under node against a DOM stand-in (like test_reader's harness); a WebView on a
phone is not run here.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import re
import shutil
import struct
import subprocess
import sys
import types
import zlib
from pathlib import Path

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = MOBILE_DIR / "app"
SRC_DIR = MOBILE_DIR.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from glossarion_mobile.ui.reader import bridge  # noqa: E402


def _has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


needs_flet = pytest.mark.skipif(not _has("flet"), reason="flet not installed")
needs_cores = pytest.mark.skipif(not _has("reader_doc"), reason="the shared reader_doc core is not importable")
NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="node is not installed")
PLATFORMS = ["android", "ios"]


def _png(width: int = 4, height: int = 6) -> bytes:
    row = b"\x00" + bytes((200, 120, 40)) * width
    data = zlib.compress(row * height)

    def chunk(kind: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", data) + chunk(b"IEND", b""))


PNG = _png()
SVG = b'<svg xmlns="http://www.w3.org/2000/svg" width="4" height="4"><rect width="4" height="4"/></svg>'


# =====================================================================================
# The page script
# =====================================================================================


def test_extras_hold_image_clicks_for_the_double_tap_and_take_image_pinches():
    js = bridge.READER_EXTRAS_JS
    # no dblclick listener: the double tap is two clicks on the same picture, decided here
    assert "addEventListener('dblclick'" not in js
    assert "var DOUBLE_MS = 300" in js and "function imageAt(el)" in js and "' full-page-img '" in js
    # one capture-phase click handler (before the shell's): pictures first, then tap edges off
    assert js.count("window.addEventListener('click'") == 1
    assert "if (imgTap !== tap) { return; }" in js  # a cancelled hold never fires late
    for name in ("touchstart", "touchmove", "touchend", "touchcancel"):
        assert f"window.addEventListener('{name}'" in js
    assert js.count("{capture: true, passive: true}") == 4  # before the shell's touch handlers


_HARNESS = r"""
const vm = require('vm');
const SCRIPTS = %(scripts)s;
function makeEnv(cfgScript) {
  const handlers = {}, winHandlers = {}, posted = [];
  let timers = [], nextTimer = 0;
  const columns = {clientWidth: 400, scrollWidth: 1200, scrollLeft: 0, style: {},
                   getBoundingClientRect() { return {left: 0, top: 0, width: 400}; },
                   querySelectorAll() { return []; }};
  const head = {children: [], appendChild(el) { this.children.push(el); }};
  const elements = {};
  const body = {scrollHeight: 1000, nodeType: 1, tagName: 'BODY'};
  const document = {
    readyState: 'complete', head, body, documentElement: {scrollHeight: 1000, clientWidth: 400},
    getElementById(id) { return id === 'columns' ? columns : (elements[id] || null); },
    createElement(tag) { const el = {tagName: tag.toUpperCase(), id: '', textContent: ''};
      return new Proxy(el, {set(t, k, v) { t[k] = v; if (k === 'id') { elements[v] = t; } return true; }}); },
    querySelectorAll() { return []; }, getElementsByName() { return []; },
    addEventListener(name, fn) { (handlers[name] = handlers[name] || []).push(fn); },
  };
  const box = {nodeType: 1, tagName: 'DIV', className: 'full-page-img full-page-img-first', parentNode: body,
               getAttribute() { return null; }, querySelector() { return img; }};
  const img = {nodeType: 1, tagName: 'IMG', className: '', parentNode: box,
               getAttribute(k) { return k === 'src' ? '/t/img/abc' : null; }};
  const para = {nodeType: 1, tagName: 'P', className: '', parentNode: body, getAttribute() { return null; }};
  const sandbox = {document, location: {hash: ''}, innerWidth: 400, innerHeight: 800, scrollY: 0,
    console: {log(s) { if (String(s).startsWith('GLRDR:')) { posted.push(JSON.parse(String(s).slice(6))); } }},
    setTimeout(fn) { const id = ++nextTimer; timers.push({id, fn}); return id; },
    clearTimeout(id) { timers = timers.filter((t) => t.id !== id); },
    getSelection() { return {toString() { return ''; }, rangeCount: 0, removeAllRanges() {}}; },
    scrollTo() {}, scrollBy() {},
    addEventListener(name, fn, opts) {
      const capture = opts === true || !!(opts && opts.capture);
      (winHandlers[name] = winHandlers[name] || []).push({fn, capture});
    }};
  sandbox.window = sandbox;
  vm.createContext(sandbox);
  for (const code of SCRIPTS) { vm.runInContext(code === '@CFG' ? cfgScript : code, sandbox); }
  const run = () => { let guard = 0; while (timers.length && guard++ < 200) { timers.shift().fn(); } };
  const fire = (name, ev) => {
    ev.stopped = false;
    ev.stopPropagation = function () { this.stopped = true; };
    ev.preventDefault = function () {};
    for (const h of (winHandlers[name] || [])) { if (h.capture && !ev.stopped) { h.fn(ev); } }
    if (!ev.stopped) { for (const fn of (handlers[name] || [])) { fn(ev); } }
    return ev;
  };
  const click = (x, target) => fire('click', {clientX: x, clientY: 300, target});
  const t = (target, x) => ({target, clientX: x, clientY: 300});
  return {G: sandbox.GLRDR, posted, run, click, fire, t, img, box, para, timers: () => timers.length};
}
const types = (env) => env.posted.map((e) => e.type);
const out = {};
let env = makeEnv(%(cfg_zones)s);
env.run(); env.posted.length = 0;
// a single tap on a picture: nothing at once, the chrome tap once the double tap's time is over
env.click(200, env.img);
out.heldBefore = types(env);
env.run();
out.single = env.posted.slice();
env.posted.length = 0;
// a double tap: the viewer only, no tap (the chrome stays as it was)
env.click(200, env.img); env.click(205, env.img); env.run();
out.double = env.posted.slice();
env.posted.length = 0;
// the margin of a full-page illustration counts as the picture
env.click(20 + 180, env.box); env.click(200, env.box); env.run();
out.boxDouble = types(env);
env.posted.length = 0;
// a single tap in the right third of a picture turns the page like a tap on text there
const before = env.G.page();
env.click(390, env.img); env.run();
out.zoneTurn = {before, after: env.G.page(), types: types(env)};
env.posted.length = 0;
// text: the shell's tap at once (no hold)
env.click(200, env.para);
out.textTap = types(env); out.textTimers = env.timers();
env.posted.length = 0;
// two fingers that start on the picture and spread: the viewer once; the shell never sees the pinch
let ev = env.fire('touchstart', {touches: [env.t(env.img, 150), env.t(env.img, 250)]});
out.pinchStartStopped = ev.stopped;
env.fire('touchmove', {touches: [env.t(env.img, 130), env.t(env.img, 270)]});
env.fire('touchmove', {touches: [env.t(env.img, 100), env.t(env.img, 300)]});
env.fire('touchend', {touches: [env.t(env.img, 300)], changedTouches: [env.t(env.img, 100)]});
env.fire('touchend', {touches: [], changedTouches: [env.t(env.img, 300)]});
out.imgPinch = env.posted.slice();
env.posted.length = 0;
// a pinch on text still changes the font size (the shell's 'scale')
env.fire('touchstart', {touches: [env.t(env.para, 150), env.t(env.para, 250)]});
env.fire('touchmove', {touches: [env.t(env.para, 100), env.t(env.para, 300)]});
env.fire('touchend', {touches: [], changedTouches: [env.t(env.para, 300)]});
out.textPinch = env.posted.slice();
// tap edges off: a held side tap on a picture is a centre tap, no page turn
env = makeEnv(%(cfg_no_zones)s);
env.run(); env.posted.length = 0;
const page0 = env.G.page();
env.click(390, env.img); env.run();
out.zonesOff = {page: page0 === env.G.page(), posted: env.posted.slice()};
console.error(JSON.stringify(out));
"""


def _page_scripts() -> list:
    """The real page: ``reader_doc.wrap_reader_html(mobile=True)`` + the extras, its scripts in order;
    the config script is a placeholder the harness fills per environment."""
    import reader_doc

    html = reader_doc.wrap_reader_html("<p>x</p>", 0, paginated=True, embedded_css="", mobile=True, chapter=0)
    html = bridge.inject_extras(html, bridge.bridge_config(doc="d1", layout="single_page"))
    scripts = re.findall(r"<script[^>]*>(.*?)</script>", html, re.S)
    return ["@CFG" if s.startswith("window.__GLRDR_CFG=") else s for s in scripts]


def _cfg_script(tap_zones: bool) -> str:
    cfg = bridge.bridge_config(doc="d1", layout="single_page", tap_zones=tap_zones, chapter=0)
    return f"window.__GLRDR_CFG={json.dumps(cfg)};"


@needs_node
@needs_cores
def test_image_taps_and_pinches_on_the_real_page_under_node(tmp_path):
    harness = _HARNESS % {"scripts": json.dumps(_page_scripts()), "cfg_zones": json.dumps(_cfg_script(True)),
                          "cfg_no_zones": json.dumps(_cfg_script(False))}
    path = tmp_path / "harness.js"
    path.write_text(harness, encoding="utf-8")
    proc = subprocess.run([NODE, str(path)], capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stderr.strip().splitlines()[-1])
    assert out["heldBefore"] == []  # held for the double tap
    assert [(e["type"], e.get("zone")) for e in out["single"]] == [("tap", "center")]  # and then it counts, once
    assert [(e["type"], e.get("src")) for e in out["double"]] == [("img", "/t/img/abc")]  # no tap at all
    assert out["boxDouble"] == ["img"]
    assert out["zoneTurn"]["after"] == out["zoneTurn"]["before"] + 1 and "tap" not in out["zoneTurn"]["types"]
    assert out["textTap"] == ["tap"] and out["textTimers"] == 0
    assert out["pinchStartStopped"] is True
    assert [e["type"] for e in out["imgPinch"]] == ["img"]  # once, and no 'scale' (font size)
    assert [e["type"] for e in out["textPinch"]] == ["scale"] and out["textPinch"][0]["scale"] == pytest.approx(2.0)
    assert out["zonesOff"]["page"] and [(e["type"], e["zone"]) for e in out["zonesOff"]["posted"]] == [
        ("tap", "center")]
    for event in out["single"] + out["double"] + out["imgPinch"] + out["textPinch"]:
        normalized = bridge.normalize_event(event)
        assert normalized is not None
    assert bridge.normalize_event(out["double"][0]).href == "/t/img/abc"


@needs_node
def test_extras_javascript_still_parses(tmp_path):
    path = tmp_path / "extras.js"
    path.write_text(bridge.READER_EXTRAS_JS, encoding="utf-8")
    proc = subprocess.run([NODE, "--check", str(path)], capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr


# =====================================================================================
# The full-screen viewer (Flet layer)
# =====================================================================================


class _Server:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.asked = []

    def image(self, image_id):
        self.asked.append(image_id)
        return (self.data, "image/png")


def _reader(tmp_path, platform: str, *, server=None):
    from glossarion_mobile.ui.reader.reader_view import ReaderDeps, ReaderScreen
    from glossarion_mobile.ui.router import parse_route

    page = types.SimpleNamespace(width=412, height=860, views=[], platform=platform, web=False,
                                 show_dialog=lambda d: dialogs.append(d), pop_dialog=lambda: None,
                                 update=lambda *a: None)
    dialogs, notes, pops = [], [], []

    def pop_overlay():  # the app's back: the shell pops the top overlay
        pops.append(1)
        if len(page.views) > 1:
            page.views.pop()

    deps = ReaderDeps(page=page, server=server, notify=lambda *a: notes.append(a), webview_ok=lambda: False,
                      cache_dir=str(tmp_path / "cache"), push_overlay=page.views.append, pop_overlay=pop_overlay,
                      files=None)
    screen = ReaderScreen(parse_route("/reader/ab12cd34ef56"), deps)
    page.views.append(types.SimpleNamespace(route="/reader/ab12cd34ef56"))  # the Reader's own View
    screen.dialogs, screen.notes, screen.pops = dialogs, notes, pops
    return screen


def _walk(control):
    yield control
    for name in ("content", "controls", "appbar", "leading", "actions"):
        child = getattr(control, name, None)
        if isinstance(child, list):
            for item in child:
                yield from _walk(item)
        elif child is not None and hasattr(child, "_c"):
            yield from _walk(child)


@needs_flet
@pytest.mark.parametrize("platform", PLATFORMS)
def test_a_page_image_opens_full_screen_over_the_reader_and_closes_back(tmp_path, platform):
    import flet as ft

    from glossarion_mobile.ui.components.media_viewer import MAX_ZOOM
    from glossarion_mobile.ui.theme import HIT_TARGET

    async def scenario():
        server = _Server(PNG)
        screen = _reader(tmp_path, platform, server=server)
        screen.chrome.set_visible(False)  # the reader had hidden the chrome
        screen.handle_payload({"type": "img", "src": "http://127.0.0.1:1/t0k/img/abc123?v=1", "seq": 1})
        for _ in range(50):
            await asyncio.sleep(0.02)
            if screen.image_viewer is not None:
                break
        viewer = screen.image_viewer
        assert viewer is not None and server.asked == ["abc123"]
        view = viewer.view
        assert screen.page.views[-1] is view and len(screen.page.views) == 2  # a pushed View, not a sheet
        assert screen.dialogs == []  # no BottomSheet, no "Close Menu" strip
        controls = list(_walk(view))
        assert not any(isinstance(c, ft.BottomSheet) for c in controls)
        assert view.bgcolor == ft.Colors.BLACK and view.padding == 0
        close = view.appbar.leading
        assert close.tooltip == "Close" and close.size_constraints == HIT_TARGET and close.key == "viewer-close"
        assert viewer.viewers and viewer.viewers[0].max_scale == MAX_ZOOM
        image = viewer.viewers[0].content
        assert isinstance(image, ft.Image) and image.fit == ft.BoxFit.CONTAIN
        assert image.src.endswith(".png") and Path(image.src).read_bytes() == PNG
        assert any(isinstance(c, ft.GestureDetector) and c.on_double_tap is not None for c in controls)
        # a second request while it is open (a double tap right after the pinch) opens nothing more
        assert await screen.view_image(PNG) is None and len(screen.page.views) == 2
        # ✕ Close: the overlay is popped; the Reader keeps its chrome as it was
        close.on_click(None)
        assert screen.pops == [1] and screen.page.views[-1] is not view and not screen.chrome.visible
        # Android / iOS back already popped it: ✕ (a late tap) does not pop the Reader
        viewer.close()
        assert screen.pops == [1] and len(screen.page.views) == 1
        # it opens again afterwards (same cached file)
        again = await screen.view_image(PNG)
        assert again is not None and screen.page.views[-1] is again.view
        assert again.viewers[0].content.src == image.src
        screen.dispose()

    asyncio.run(scenario())


@needs_flet
def test_an_image_the_viewer_cannot_show_says_so(tmp_path):
    from glossarion_mobile.ui.reader.reader_view import IMAGE_UNAVAILABLE, image_cache_file

    async def scenario():
        screen = _reader(tmp_path, "android")
        assert await screen.view_image(SVG) is None
        assert len(screen.page.views) == 1 and [n[0] for n in screen.notes] == [IMAGE_UNAVAILABLE]
        screen.dispose()

    asyncio.run(scenario())
    assert image_cache_file(b"", str(tmp_path)) is None
    path = image_cache_file(PNG, None)  # no app cache folder: the system temp folder
    assert path and Path(path).read_bytes() == PNG


@needs_flet
def test_a_tap_event_still_toggles_the_chrome_once(tmp_path):
    screen = _reader(tmp_path, "android")
    assert screen.chrome.visible
    screen.handle_payload({"type": "tap", "zone": "center", "seq": 1})
    assert not screen.chrome.visible
    screen.handle_payload({"type": "tap", "zone": "center", "seq": 2})
    assert screen.chrome.visible
    screen.dispose()


@needs_flet
def test_the_native_fallback_page_opens_the_same_viewer_on_a_double_tap(tmp_path):
    import flet as ft

    from glossarion_mobile.ui.reader import blocks as rb
    from glossarion_mobile.ui.reader import model as rm
    from glossarion_mobile.ui.reader.fallback_view import FallbackPage

    got = []
    page = FallbackPage(on_tap_zone=lambda z: None, on_pinch=lambda s: None, on_pinch_end=lambda: None,
                        on_paragraph=lambda t: None, on_scroll=lambda f: None, on_image=got.append)
    page.render(rb.html_to_blocks("<p>a</p><img src='x.png' alt='pic'/>"), theme={"bg": "#000000"},
                settings=rm.ReaderSettings(), images={"x.png": PNG})
    detector = page.list_view.controls[1]
    assert isinstance(detector, ft.GestureDetector) and detector.on_double_tap is not None
    assert detector.on_tap_up is None  # a single tap goes on to the page's tap zones
    detector.on_double_tap(None)
    assert got == [PNG]

    async def scenario():
        screen = _reader(tmp_path, "android")
        screen.fallback.on_image(PNG)  # the Reader's own fallback page is wired to the viewer
        for _ in range(50):
            await asyncio.sleep(0.02)
            if screen.image_viewer is not None:
                break
        assert screen.image_viewer is not None and screen.page.views[-1] is screen.image_viewer.view
        screen.dispose()

    asyncio.run(scenario())


@needs_flet
def test_media_viewer_double_tap_zooms_in_and_fits_again():
    from glossarion_mobile.ui.chat.media_model import MediaItem
    from glossarion_mobile.ui.components.media_viewer import DOUBLE_TAP_ZOOM, MAX_ZOOM, MediaViewer

    viewer = MediaViewer([MediaItem("image", "a.png", True)])
    calls = []

    async def zoom(factor):
        calls.append(("zoom", factor))

    async def reset(animation_duration=None):
        calls.append(("reset", animation_duration.milliseconds if animation_duration else None))

    interactive = viewer.viewers[0]
    object.__setattr__(interactive, "zoom", zoom)
    object.__setattr__(interactive, "reset", reset)

    def gesture(*scales):
        """One touch gesture on the image: its scale updates, then its end."""
        for scale in scales:
            interactive.on_interaction_update(types.SimpleNamespace(scale=scale, pointer_count=2))
        interactive.on_interaction_end(types.SimpleNamespace(pointer_count=0))

    async def scenario():
        assert await viewer.toggle_zoom(0) is True
        assert await viewer.toggle_zoom(0) is False
        gesture(1.0, 1.0)  # a pan: still fitted
        assert await viewer.toggle_zoom(0) is True
        assert await viewer.toggle_zoom(0) is False
        gesture(1.4, 2.2)  # a pinch zoomed it
        assert viewer.is_zoomed(0)
        assert await viewer.toggle_zoom(0) is False  # the double tap fits it again
        # pinched in and then back to the fitted size: it is fitted, so ONE double tap zooms in
        # (a flag set by any pinch made this double tap a reset that changed nothing)
        gesture(1.5, 3.0)
        gesture(0.6, 0.2)  # further out than fitted: the viewer stops at 1x
        assert not viewer.is_zoomed(0) and viewer.scales[0] == 1.0
        assert await viewer.toggle_zoom(0) is True
        gesture(4.0)  # past the viewer's own limit: held to it
        assert viewer.scales[0] == MAX_ZOOM
        assert await viewer.toggle_zoom(0) is False
        gesture()  # an end with no update (a tap) changes nothing
        assert viewer.scales[0] == 1.0
        assert await viewer.toggle_zoom(5) is False  # no such image

    asyncio.run(scenario())
    assert calls == [("zoom", DOUBLE_TAP_ZOOM), ("reset", 200), ("zoom", DOUBLE_TAP_ZOOM), ("reset", 200),
                     ("reset", 200), ("zoom", DOUBLE_TAP_ZOOM), ("reset", 200)]


@needs_flet
def test_reader_feature_hands_the_reader_the_overlay_stack():
    from glossarion_mobile.ui.reader.feature import ReaderFeature

    pushed, backs, updates = [], [], []
    page = types.SimpleNamespace(platform=None, web=False, update=lambda: updates.append(1))
    shell = types.SimpleNamespace(push_overlay=pushed.append)
    files = object()
    app = types.SimpleNamespace(page=page, shell=shell, back=lambda: backs.append(1), files=files, paths=None,
                                state=None)
    feature = ReaderFeature(app)
    deps = feature._deps()
    assert deps.files is files
    view = object()
    deps.push_overlay(view)
    assert pushed == [view] and updates == [1]
    deps.pop_overlay()
    assert backs == [1]
