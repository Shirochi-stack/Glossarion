"""Host tests for the U5 Reader (UI_SPEC §3.11).

* ``services/reader_server.py``: loopback-only binding, token-protected document /
  image paths, the ``/__ev`` event endpoint (token in path, cookie, header or query),
  the Host-header guard, the document CSP and version handling;
* ``ui/reader/bridge.py``: console-event parsing, normalisation of the shared mobile
  shell's events (``reader_doc.wrap_reader_html(mobile=True)``) and the Reader extras'
  events, de-duplication across the console and fetch channels and stale pages, the
  start position, the extras injection; the real page (shell + extras) runs under node
  (when installed) against a small DOM stand-in: paging, edges, aliases, live restyle
  with the proportional position, tap edges off, cleared selections;
* ``ui/reader/model.py``: settings resolution and clamps, scopes -> config keys,
  Double page only on tablets in landscape, theme/typography -> CSS, the desktop
  page-hint and resize formulas, saved positions, labels, mode availability;
* ``state/prefs.py``: reader positions with chapter/pages/last, per-book settings;
* ``ui/reader/blocks.py`` (fallback renderer / live panel content) and
  ``ui/reader/live.py`` over the shared ``live_stream`` classifier (checked against the
  desktop ``_classify_live_line`` rules) and its outcome strings;
* ``ui/reader/session.py`` / ``document.py`` over the real shared cores
  (``library_core.plan_open_reader``, ``reader_doc``, ``reader_overlay``,
  ``workspace_reader``) on the self-test EPUB: open plans (plain / overlay / dual /
  workspace), the overlay refresh, flavours incl. Bilingual, the Translate rules,
  search batches, pages with images served by the server, Scroll all, native blocks,
  the native TOC;
* the Flet layer (skipped without Flet): ReaderScreen event handling, settings scopes
  and Resume, the live translation flow incl. the cleanup after a Stop, the retranslate
  confirmation, the native fallback page, LivePanel, AaSheet, the chapters drawer,
  ReaderFeature routing, the ``single_chapter`` job kind, and the Reader in the real
  app shell on the fake Flet session (WebView + server + both event channels).

Run from src/mobile with the mobile venv:
    .venv/Scripts/python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_reader.py
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import types
import urllib.error
import urllib.request
from pathlib import Path

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = MOBILE_DIR / "app"
SRC_DIR = MOBILE_DIR.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from glossarion_mobile.services.reader_server import COOKIE_NAME, ReaderServer, image_id_for  # noqa: E402
from glossarion_mobile.state.prefs import Prefs  # noqa: E402
from glossarion_mobile.ui.reader import blocks as rb  # noqa: E402
from glossarion_mobile.ui.reader import bridge  # noqa: E402
from glossarion_mobile.ui.reader import live as rl  # noqa: E402
from glossarion_mobile.ui.reader import model as rm  # noqa: E402


def _has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


needs_flet = pytest.mark.skipif(not _has("flet"), reason="flet not installed")
NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="node is not installed")

DESKTOP_THEMES = [
    {"name": "Dark", "bg": "#1e1e1e", "fg": "#d4d4d4", "heading": "#c8c8f0", "link": "#6c9bd2", "code_bg": "#252530",
     "border": "#333333"},
    {"name": "Light", "bg": "#faf9f6", "fg": "#2c2c2c", "heading": "#333333", "link": "#1a73e8", "code_bg": "#eeeeee",
     "border": "#dddddd"},
    {"name": "Sepia", "bg": "#f4ecd8", "fg": "#5b4636", "heading": "#3e2c1c", "link": "#8b5e3c", "code_bg": "#ece0c8",
     "border": "#d4c8a8"},
    {"name": "Midnight", "bg": "#0d1117", "fg": "#c9d1d9", "heading": "#58a6ff", "link": "#58a6ff",
     "code_bg": "#161b22", "border": "#21262d"},
    {"name": "Forest", "bg": "#1a2e1a", "fg": "#c8d8c8", "heading": "#7ec87e", "link": "#5dbd5d", "code_bg": "#1e3a1e",
     "border": "#2a4a2a"},
    {"name": "Rose", "bg": "#2e1a2e", "fg": "#e0c8e0", "heading": "#d89ad8", "link": "#c074c0", "code_bg": "#3a1e3a",
     "border": "#4a2a4a"},
]


# =====================================================================================
# ReaderServer
# =====================================================================================


def _get(url: str, *, headers: dict | None = None, data: bytes | None = None, method: str | None = None):
    request = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, dict(response.headers), response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers or {}), exc.read()


@pytest.fixture
def server():
    events = []
    srv = ReaderServer(on_event=events.append)
    srv.start()
    srv.events = events
    yield srv
    srv.stop()


def test_server_keeps_dropped_connections_off_stderr(server, capsys, caplog):
    """The WebView drops keep-alive connections on every chapter turn: a debug line, never
    socketserver's traceback on stderr; another error is still logged."""
    import logging

    srv = server._server
    with caplog.at_level(logging.DEBUG, logger="glossarion.reader.server"):
        for error in (ConnectionResetError(10054, "reset"), BrokenPipeError(32, "pipe"), TimeoutError()):
            try:
                raise error
            except OSError:
                srv.handle_error(None, ("127.0.0.1", 5555))
        try:
            raise ValueError("boom")
        except ValueError:
            srv.handle_error(None, ("127.0.0.1", 5555))
    assert capsys.readouterr().err == ""
    records = [r for r in caplog.records if r.name == "glossarion.reader.server"]
    assert [r.levelno for r in records] == [logging.DEBUG] * 3 + [logging.ERROR]
    assert records[-1].exc_info and records[-1].exc_info[0] is ValueError


def test_server_binds_loopback_only():
    with pytest.raises(ValueError):
        ReaderServer(host="0.0.0.0")
    srv = ReaderServer()
    try:
        port = srv.start()
        assert port > 0 and srv.port == port and srv.origin == f"http://127.0.0.1:{port}"
        assert srv._server.server_address[0] == "127.0.0.1"
        assert len(srv.token) >= 32 and srv.base_path == f"/{srv.token}/"
    finally:
        srv.stop()
    assert not srv.running


def test_server_serves_document_with_token_csp_and_cookie(server):
    url = server.publish("<html><body>héllo</body></html>")
    assert url.startswith(server.base_url) and "reader.html?v=" in url
    status, headers, body = _get(url)
    assert status == 200 and body.decode("utf-8") == "<html><body>héllo</body></html>"
    assert headers["Content-Type"].startswith("text/html")
    assert headers["Cache-Control"] == "no-store"
    assert "connect-src 'self'" in headers["Content-Security-Policy"]
    assert "img-src 'self' data: blob:" in headers["Content-Security-Policy"]
    assert headers["Set-Cookie"].startswith(f"{COOKIE_NAME}={server.token};") and "HttpOnly" in headers["Set-Cookie"]
    assert headers["X-Content-Type-Options"] == "nosniff"
    # wrong / missing token, unknown document, traversal: bare 404s
    for bad in (url.replace(server.token, "x" * len(server.token)), f"{server.origin}/reader.html",
                f"{server.base_url}other.html", f"{server.base_url}..%2F..%2Fetc%2Fpasswd", f"{server.origin}/"):
        assert _get(bad)[0] == 404, bad
    # POST is for events only
    assert _get(url, data=b"{}", method="POST")[0] == 404


def test_server_rejects_foreign_host_header(server):
    url = server.publish("<p>x</p>")
    assert _get(url, headers={"Host": f"evil.example:{server.port}"})[0] == 404
    assert _get(url, headers={"Host": f"localhost:{server.port}"})[0] == 200


def test_server_keeps_recent_document_versions(server):
    first = server.publish("one")
    second = server.publish("two")
    assert _get(first)[2] == b"one" and _get(second)[2] == b"two"
    assert _get(server.base_url + "reader.html?v=999")[2] == b"two"  # unknown version -> latest
    for index in range(6):
        server.publish(f"n{index}")
    assert _get(first)[2] == b"n5"  # evicted -> latest
    with pytest.raises(ValueError):
        server.publish("x", name="../evil.html")


def test_server_images_bytes_file_and_loader(server, tmp_path):
    png = b"\x89PNG\r\n\x1a\nfake"
    picture = tmp_path / "pic.jpg"
    picture.write_bytes(b"\xff\xd8\xff\xe0jpegdata")
    calls = []
    url_bytes = server.register_image("OEBPS/images/cover.png", png)
    url_file = server.register_image(str(picture), str(picture))
    url_lazy = server.register_image("lazy.gif", lambda: calls.append(1) or b"GIF89a")
    url_none = server.register_image("missing.png", lambda: None)
    assert url_bytes.startswith(server.base_path + "img/") and url_bytes.endswith(".png")
    assert url_bytes.rsplit("/", 1)[1] == image_id_for("OEBPS/images/cover.png")
    status, headers, body = _get(server.origin + url_bytes)
    assert status == 200 and body == png and headers["Content-Type"] == "image/png"
    assert "sandbox" in headers["Content-Security-Policy"]
    status, headers, body = _get(server.origin + url_file)
    assert body == b"\xff\xd8\xff\xe0jpegdata" and headers["Content-Type"] == "image/jpeg"
    assert _get(server.origin + url_lazy)[2] == b"GIF89a" and calls == [1]
    assert _get(server.origin + url_none)[0] == 404
    assert _get(server.origin + server.base_path + "img/0000000000000000000a.png")[0] == 404
    assert _get(server.origin + url_bytes.replace(server.token, "nope"))[0] == 404
    assert server.image_count == 4
    server.clear_images()
    assert _get(server.origin + url_bytes)[0] == 404


def test_server_serves_only_images_by_content(server, tmp_path):
    """U5 review: an <img src="../../…"> that resolved to an app-private file (OAuth tokens,
    settings) is never served, whatever its registered name or extension says."""
    secret = tmp_path / "authgpt_tokens.json"
    secret.write_text('{"access_token": "sk-secret"}', encoding="utf-8")
    script = tmp_path / "evil.js"
    script.write_text("alert(1)", encoding="utf-8")
    svg = b'<?xml version="1.0"?><svg xmlns="http://www.w3.org/2000/svg" width="4" height="4"/>'
    for url in (server.register_image(str(secret), str(secret)), server.register_image("x.png", str(secret)),
                server.register_image("evil.js", str(script)), server.register_image("blob.png", b"hello")):
        assert _get(server.origin + url)[0] == 404, url
    status, headers, body = _get(server.origin + server.register_image("pic.svg", svg))
    assert status == 200 and headers["Content-Type"] == "image/svg+xml" and body == svg
    webp = b"RIFF\x10\x00\x00\x00WEBPVP8 data"
    assert _get(server.origin + server.register_image("a.bin", webp))[1]["Content-Type"] == "image/webp"


def test_server_event_endpoint_token_paths(server):
    payload = json.dumps({"t": "page", "page": 2, "count": 9, "doc": "d1", "seq": 1}).encode()
    as_json = {"Content-Type": "application/json"}
    status, _h, _b = _get(server.origin + server.event_path, data=payload, method="POST", headers=as_json)
    assert status == 204 and server.events[-1]["page"] == 2 and server.events[-1]["_channel"] == "http"
    # /__ev with the cookie (what the page's fetch sends) or the header
    assert _get(server.origin + "/__ev", data=payload, method="POST",
                headers={"Cookie": f"other=1; {COOKIE_NAME}={server.token}", **as_json})[0] == 204
    assert _get(server.origin + "/__ev", data=payload, method="POST",
                headers={"X-GLRDR-Token": server.token, **as_json})[0] == 204
    count = len(server.events)
    # U5 review: never from a URL a book can embed (<img src>, CSS url() carry the cookie) or a form
    assert _get(server.origin + f"/__ev?t={server.token}&d=" + urllib.request.quote('{"t":"tap"}'))[0] == 404
    assert _get(server.origin + "/__ev?d=" + urllib.request.quote('{"type":"link","href":"https://x"}'),
                headers={"Cookie": f"{COOKIE_NAME}={server.token}"})[0] == 404
    assert _get(server.origin + server.event_path, data=payload, method="POST")[0] == 404  # form content type
    assert _get(server.origin + server.event_path, data=payload, method="POST",
                headers={"Content-Type": "text/plain"})[0] == 404
    assert len(server.events) == count
    # no / wrong token: 404 and nothing delivered
    assert _get(server.origin + "/__ev", data=payload, method="POST", headers=as_json)[0] == 404
    assert _get(server.origin + "/__ev", data=payload, method="POST",
                headers={"Cookie": f"{COOKIE_NAME}=bad", **as_json})[0] == 404
    # malformed JSON is acknowledged but not delivered; oversized bodies are refused
    assert _get(server.origin + server.event_path, data=b"{not json", method="POST", headers=as_json)[0] == 204
    assert _get(server.origin + server.event_path, data=b"x" * (70 * 1024), method="POST", headers=as_json)[0] == 413
    assert len(server.events) == count


def test_server_document_csp_allows_only_the_page_nonce(server):
    from glossarion_mobile.services.reader_server import DOCUMENT_CSP, document_csp

    status, headers, _b = _get(server.publish("<p>x</p>", script_nonce="abcDEF123_-"))
    csp = headers["Content-Security-Policy"]
    assert status == 200 and "script-src 'nonce-abcDEF123_-'" in csp and "unsafe-inline'; img" not in csp
    assert "'unsafe-inline'" not in csp.split("script-src", 1)[1].split(";", 1)[0]
    assert csp == document_csp("abcDEF123_-")
    # no nonce: no script at all
    assert _get(server.publish("<p>y</p>"))[1]["Content-Security-Policy"] == DOCUMENT_CSP
    assert "script-src 'none'" in DOCUMENT_CSP
    with pytest.raises(ValueError):
        server.publish("<p>z</p>", script_nonce="bad' nonce")


# =====================================================================================
# bridge (the shared reader_doc mobile shell + the Reader's extras)
# =====================================================================================


def test_parse_console_message():
    assert bridge.parse_console_message('GLRDR:{"type":"tap","seq":3}') == {"type": "tap", "seq": 3}
    assert bridge.parse_console_message('app.js:1 GLRDR:{"type":"tap"}') == {"type": "tap"}  # source prefix
    assert bridge.parse_console_message("hello") is None
    assert bridge.parse_console_message("GLRDR:{bad json") is None
    assert bridge.parse_console_message('GLRDR:[1,2]') is None
    assert bridge.parse_console_message("x" * 40 + 'GLRDR:{"type":"tap"}') is None


def test_normalize_event_shell_and_extras_schemas():
    ev = bridge.normalize_event({"type": "page", "page": 12, "count": 9, "seq": "4", "reason": "next", "chapter": 3})
    assert (ev.type, ev.page, ev.count, ev.seq, ev.why, ev.chapter) == ("page", 8, 9, 4, "next", 3)
    end = bridge.normalize_event({"type": "edge", "edge": "end", "page": 3, "count": 4})
    start = bridge.normalize_event({"type": "edge", "edge": "start"})
    assert end.direction == 1 and start.direction == -1
    tap = bridge.normalize_event({"type": "tap", "zone": "center"})
    assert tap.type == "tap" and tap.zone == "center"
    sel = bridge.normalize_event({"type": "selection", "text": " word "})
    assert sel.type == "sel" and sel.text == " word "
    rect = bridge.normalize_event({"type": "selrect", "x": 0.1, "y": 1.7, "w": 0.2, "h": 0.05, "doc": "d1"})
    assert rect.rect == (0.1, 1.0, 0.2, 0.05) and rect.doc == "d1"
    assert bridge.normalize_event({"type": "selrect", "x": "nan"}).rect is None
    pinch = bridge.normalize_event({"type": "scale", "scale": 1.3})
    assert pinch.type == "pinch" and pinch.step == 2 and pinch.ended
    assert bridge.normalize_event({"type": "scale", "scale": 0.8}).step == -2
    assert bridge.scale_steps(9) == 6 and bridge.scale_steps(1.0) == 0 and bridge.scale_steps("x") == 0
    scroll = bridge.normalize_event({"type": "scroll", "fraction": 1.4, "ch": 3, "chf": -1})
    assert scroll.fraction == 1.0 and scroll.chapter == 3 and scroll.chapter_fraction == 0.0
    link = bridge.normalize_event({"type": "link", "href": "ch2.xhtml#a"})
    assert link.href == "ch2.xhtml#a" and link.external is False
    assert bridge.normalize_event({"type": "link", "href": "https://x.test/"}).external
    assert bridge.normalize_event({"type": "img", "src": "/x/img/1"}).href == "/x/img/1"
    assert bridge.normalize_event({"t": "found", "ok": False}).ok is False  # extras' own key
    assert bridge.normalize_event({"type": "evil"}) is None and bridge.normalize_event(None) is None
    assert len(bridge.normalize_event({"type": "selection", "text": "x" * 10000}).text) == 4000


def test_event_deduper_channels_documents_and_chapters():
    dedupe = bridge.EventDeduper()
    dedupe.set_document("d1", 4)
    page = bridge.normalize_event({"type": "page", "page": 1, "count": 3, "seq": 1, "chapter": 4})
    assert dedupe.accept(page) and not dedupe.accept(page)  # console + fetch copy
    other_same_seq = bridge.normalize_event({"type": "page", "page": 2, "count": 3, "seq": 1, "chapter": 4})
    assert dedupe.accept(other_same_seq)  # a different event (an older page reusing a sequence number)
    assert not dedupe.accept(bridge.normalize_event({"type": "page", "seq": 9, "chapter": 3}))  # stale chapter
    assert not dedupe.accept(bridge.normalize_event({"type": "selrect", "seq": 9, "doc": "d0"}))  # stale page
    assert dedupe.accept(bridge.normalize_event({"type": "scroll", "seq": 10, "doc": "d1", "ch": 7}))
    dedupe.set_document("d2", None)  # Scroll all: shell events carry no chapter
    assert dedupe.accept(bridge.normalize_event({"type": "ready", "seq": 1}))
    assert dedupe.dropped == 3


def test_position_fragment_and_config():
    assert bridge.position_fragment({"last": True}) == (10 ** 6, "")
    assert bridge.position_fragment({"fraction": 0.5}) == (None, "f=0.5")
    assert bridge.position_fragment({"fraction": 0.123456789}) == (None, "f=0.123457")
    assert bridge.position_fragment({"fraction": 0}) == (0, "")
    assert bridge.position_fragment({"page": "7"}) == (7, "") and bridge.position_fragment(None) == (0, "")
    cfg = bridge.bridge_config(doc="d3", layout=rm.LAYOUT_ALL, tap_zones=False, chapter=4, scroll_all=True,
                               hint={"fraction": 0.25, "page": 3}, find={"text": "검", "occurrence": 2}, anchor="s1")
    assert cfg == {"doc": "d3", "layout": "all_scroll", "zones": False, "ch": 4, "all": True,
                   "hint": {"fraction": 0.25}, "find": {"text": "검", "occurrence": 2}, "anchor": "s1"}


def test_inject_extras_live_style_and_escaped_config():
    cfg = bridge.bridge_config(doc="d7", layout=rm.LAYOUT_SINGLE, find={"text": "</script><b>", "occurrence": 2})
    html = bridge.inject_extras("<html><head><title>t</title></head><body><p>x</p></body></html>", cfg,
                                live_css="body{color:red}")
    assert html.index('<style id="glrdr-live">body{color:red}</style>') < html.index("</body>")
    assert bridge.READER_EXTRAS_JS.strip()[:20] in html
    raw_cfg = html.split("window.__GLRDR_CFG=", 1)[1].split(";</script>", 1)[0]
    assert "</script>" not in raw_cfg and json.loads(raw_cfg)["find"]["text"] == "</script><b>"
    assert bridge.inject_extras("<p>fragment</p>", cfg).startswith("<p>fragment</p><style")
    assert not bridge.has_shell_bridge(html)


def test_js_call_quotes_arguments():
    assert bridge.js_call("go", 3) == "window.GLRDR&&GLRDR.go&&GLRDR.go(3);"
    script = bridge.js_call("find", "a'b\"</script>", 0)
    assert "</script>" not in script and json.loads('"a\'b\\"\\u003c/script\\u003e"') == "a'b\"</script>"
    with pytest.raises(ValueError):
        bridge.js_call("go(1);alert", 1)


# The real page (reader_doc.wrap_reader_html(mobile=True) + the extras) runs under node in a
# small DOM stand-in: every <script> of the page shares one global, like in the WebView.
_NODE_HARNESS = r"""
const vm = require('vm');
const SCRIPTS = %(scripts)s;
function makeEnv(opts) {
  const handlers = {}, winHandlers = {}, posted = [], timers = [];
  const columns = {clientWidth: 400, scrollWidth: opts.scrollWidth, scrollLeft: 0, style: {},
                   getBoundingClientRect() { return {left: 0, top: 0, width: 400}; },
                   querySelectorAll() { return []; }};
  const head = {children: [], appendChild(el) { this.children.push(el); }};
  const elements = {};
  const document = {
    readyState: 'complete', head, body: {scrollHeight: 1000}, documentElement: {scrollHeight: 1000, clientWidth: 400},
    getElementById(id) { return id === 'columns' ? (opts.paged ? columns : null) : (elements[id] || null); },
    createElement(tag) { const el = {tagName: tag.toUpperCase(), id: '', textContent: ''};
      return new Proxy(el, {set(t, k, v) { t[k] = v; if (k === 'id') { elements[v] = t; } return true; }}); },
    querySelectorAll() { return []; }, getElementsByName() { return []; },
    addEventListener(name, fn) { (handlers[name] = handlers[name] || []).push(fn); },
  };
  let selection = '';
  const sandbox = {document, location: {hash: opts.hash || ''}, innerWidth: 400, innerHeight: 800, scrollY: 0,
    console: {log(s) { if (String(s).startsWith('GLRDR:')) { posted.push(JSON.parse(String(s).slice(6))); } }},
    setTimeout(fn) { timers.push(fn); return timers.length; }, clearTimeout() {},
    getSelection() { return {toString() { return selection; }, rangeCount: 0, removeAllRanges() { selection = ''; }}; },
    scrollTo() {}, scrollBy() {},
    addEventListener(name, fn, capture) { (winHandlers[name] = winHandlers[name] || []).push({fn, capture: !!capture}); }};
  sandbox.window = sandbox;
  vm.createContext(sandbox);
  for (const code of SCRIPTS) { vm.runInContext(code, sandbox); }
  const run = () => { let guard = 0; while (timers.length && guard++ < 200) { timers.shift()(); } };
  const click = (x, target) => {
    const ev = {clientX: x, clientY: 100, target: target || {nodeType: 1, tagName: 'P', parentNode: null, getAttribute() { return null; }},
                stopped: false, preventDefault() {}, stopPropagation() { this.stopped = true; }};
    for (const h of (winHandlers.click || [])) { if (h.capture) { h.fn(ev); } }
    if (!ev.stopped) { for (const fn of (handlers.click || [])) { fn(ev); } }
  };
  return {G: sandbox.GLRDR, posted, columns, run, click, handlers, winHandlers, sandbox,
          select(text) { selection = text; (handlers.selectionchange || []).forEach(f => f()); run(); }};
}
const out = {};
// 9 pages of 400 px; the page starts at #f=0.5 -> page round(8 * 0.5) = 4
let env = makeEnv({paged: true, scrollWidth: 3600, hash: '#f=0.5'});
env.run();
out.ready = env.posted.filter(e => e.type === 'ready')[0];
out.scrollLeft = env.columns.scrollLeft;
env.click(390); env.click(390);
out.afterTaps = env.G.page();
env.G.go(100); out.goClamped = env.G.page();
env.click(390); out.edgeEnd = env.posted[env.posted.length - 1];
env.G.go(0); env.click(5); out.edgeStart = env.posted[env.posted.length - 1];
env.click(200); out.centre = env.posted[env.posted.length - 1];
env.G.goLast(); out.goLast = env.G.page();
env.G.httpOff();
// applyStyle keeps the proportional position: page 6 of 9 -> 18 pages -> round(6/8*17) = 13 (half-even: 12.75 -> 13)
env.G.go(6); env.columns.scrollWidth = 7200; env.G.applyStyle('body{}'); env.run();
out.styled = env.G.page();
out.styleElement = !!env.sandbox.document.getElementById('glrdr-live');
// selection: the shell posts the text; the extras post the cleared selection
env.select('word'); env.select('');
out.selections = env.posted.filter(e => e.type === 'selection').map(e => e.text);
out.seqs = env.posted.map(e => e.seq);
// tap edges off: side taps become centre taps (no page turn)
env = makeEnv({paged: true, scrollWidth: 1200, hash: '', zones: false});
env.run(); env.click(390);
out.zonesOff = {page: env.G.page(), last: env.posted[env.posted.length - 1]};
console.error(JSON.stringify(out));
"""


@needs_node
def test_shared_page_bridge_and_extras_under_node(tmp_path, cores, novel):
    from glossarion_mobile.ui.reader.document import DocumentBuilder

    session = _overlay_session(novel)
    builder = DocumentBuilder(session, lambda key, source: "/t/img/x")

    def scripts(tap_zones: bool) -> list:
        built = builder.build(0, settings=rm.ReaderSettings(tap_zones=tap_zones), layout=rm.LAYOUT_SINGLE,
                              theme=DESKTOP_THEMES[0], doc_id="d1", event_url="/t/__ev", hint={"fraction": 0.5})
        assert built.has_page_bridge and built.fragment == "f=0.5"
        # every script of the page carries this page's CSP nonce (U5 review: nothing else runs)
        assert built.nonce and built.html.count("<script") == built.html.count(f'<script nonce="{built.nonce}">')
        return [m.group(1) for m in re.finditer(r'<script nonce="[^"]+">(.*?)</script>', built.html, re.S)]

    with_zones = scripts(True)
    without_zones = scripts(False)
    assert without_zones[-2] != with_zones[-2] and '"zones":false' in without_zones[-2]
    harness = _NODE_HARNESS % {"scripts": json.dumps(with_zones)}
    # the second environment needs the zones-off config: swap its CFG script in
    harness = harness.replace("env = makeEnv({paged: true, scrollWidth: 1200, hash: '', zones: false});",
                              "SCRIPTS.splice(0, SCRIPTS.length, ...%s);\n"
                              "env = makeEnv({paged: true, scrollWidth: 1200, hash: '', zones: false});"
                              % json.dumps(without_zones))
    path = tmp_path / "harness.js"
    path.write_text(harness, encoding="utf-8")
    proc = subprocess.run([NODE, str(path)], capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stderr.strip().splitlines()[-1])
    assert out["ready"]["page"] == 4 and out["ready"]["count"] == 9 and out["ready"]["chapter"] == 0
    assert out["scrollLeft"] == 4 * 401 and out["afterTaps"] == 6  # page width 400 + the 1 px column gap and out["goClamped"] == 8
    assert out["edgeEnd"]["type"] == "edge" and out["edgeEnd"]["edge"] == "end"
    assert out["edgeStart"]["edge"] == "start" and out["centre"] == {**out["centre"], "type": "tap", "zone": "center"}
    assert out["goLast"] == 8 and out["styleElement"]
    assert out["styled"] == rm.page_from_hint(rm.capture_hint(6, 9), 18) == 13
    assert out["selections"] == ["word", ""]
    assert out["seqs"] == sorted(out["seqs"]) and len(set(out["seqs"])) == len(out["seqs"])
    assert out["zonesOff"]["page"] == 0 and out["zonesOff"]["last"]["type"] == "tap"
    assert out["zonesOff"]["last"]["doc"] == "d1"
    # every event the shell and the extras post normalises
    for event in (out["ready"], out["edgeEnd"], out["centre"], out["zonesOff"]["last"]):
        assert bridge.normalize_event(event) is not None


@needs_node
def test_extras_javascript_syntax(tmp_path):
    path = tmp_path / "extras.js"
    path.write_text(bridge.READER_EXTRAS_JS, encoding="utf-8")
    proc = subprocess.run([NODE, "--check", str(path)], capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr


# =====================================================================================
# model
# =====================================================================================


def test_resolve_settings_defaults_overrides_and_clamps():
    config = {}
    settings = rm.resolve_settings(config.get)
    assert (settings.font_size, settings.line_spacing, settings.theme, settings.font_family, settings.layout) == \
        (14, 1.8, 0, "Embedded CSS", "single_page")
    assert settings.embedded_css and not settings.show_raw and settings.tap_zones and settings.margins == 16
    config = {"epub_reader_font_size": 40, "epub_reader_line_spacing": 0.2, "epub_reader_theme": 9,
              "epub_reader_font_family": "Sans", "epub_reader_layout": "weird", "epub_reader_native_toc": 1}
    settings = rm.resolve_settings(config.get)
    assert (settings.font_size, settings.line_spacing, settings.theme, settings.font_family, settings.layout,
            settings.native_toc) == (32, 1.0, 0, "Sans", "single_page", True)
    book = rm.resolve_settings(config.get, {"font_size": 11, "theme": 2, "margins": 99}, {"tap_zones": False})
    assert book.font_size == 11 and book.theme == 2 and book.margins == 40 and not book.tap_zones
    assert book.overridden == frozenset({"font_size", "theme", "margins"})
    assert rm.config_updates({"font_size": 99, "theme": 3, "margins": 20, "layout": "scroll"}) == {
        "epub_reader_font_size": 32, "epub_reader_theme": 3, "epub_reader_layout": "scroll"}


def test_double_page_only_on_tablet_landscape():
    assert rm.double_page_allowed(1280, 800) and not rm.double_page_allowed(800, 1280)
    assert not rm.double_page_allowed(899, 500) and not rm.double_page_allowed(None, None)
    assert rm.effective_layout(rm.LAYOUT_DOUBLE, 412, 860) == rm.LAYOUT_SINGLE
    assert rm.effective_layout(rm.LAYOUT_DOUBLE, 1280, 800) == rm.LAYOUT_DOUBLE
    assert rm.effective_layout(rm.LAYOUT_ALL, 412, 860) == rm.LAYOUT_ALL
    assert rm.is_paged(rm.LAYOUT_SINGLE) and not rm.is_paged(rm.LAYOUT_SCROLL) and rm.spread_for(rm.LAYOUT_DOUBLE) == 2


def test_override_css_theme_and_typography():
    settings = rm.ReaderSettings(font_size=14, line_spacing=2.0, theme=2, font_family="Sans", margins=20)
    css = rm.override_css(DESKTOP_THEMES[2], settings)
    assert "background: #f4ecd8 !important" in css and "color: #5b4636 !important" in css
    assert "#columns { font-size: 19px !important; line-height: 2.0 !important; font-family: -apple-system" in css
    assert "h1, h2, h3, h4, h5, h6 { color: #3e2c1c !important; }" in css and "a { color: #8b5e3c" in css
    assert ("#content { padding-left: max(20px, env(safe-area-inset-left)) !important; "
            "padding-right: max(20px, env(safe-area-inset-right)) !important; }") in css
    scroll = rm.override_css(DESKTOP_THEMES[0], rm.ReaderSettings(font_size=12), layout=rm.LAYOUT_SCROLL)
    assert "body { font-size: 16px !important; line-height: 1.8 !important; }" in scroll  # Embedded CSS: no family
    assert "#content" not in scroll and "body { padding-left: max(16px, env(safe-area-inset-left))" in scroll
    hostile = rm.override_css({"bg": "red;}</style>", "fg": "#abc"}, rm.ReaderSettings(font_family="X'};body{"))
    assert "</style>" not in hostile and "background: #1e1e1e" in hostile and "'X;body', Georgia" not in hostile
    assert rm.font_stack("Embedded CSS") is None and "monospace" in rm.font_stack("Mono")


def test_theme_for_follow_app_theme():
    settings = rm.ReaderSettings(theme=3)
    assert rm.theme_for(DESKTOP_THEMES, settings)["name"] == "Midnight"
    follow = settings.with_changes(follow_app_theme=True)
    assert rm.theme_for(DESKTOP_THEMES, follow, app_dark=False)["name"] == "Light"
    assert rm.theme_for(DESKTOP_THEMES, follow, app_dark=True)["name"] == "Dark"
    assert rm.theme_for([], settings)["bg"] == "#1e1e1e"


def _desktop_capture(page, pages):
    # epub_library EpubReaderDialog._capture_position_hint
    was_last = pages > 0 and page >= pages - 1
    proportion = max(0.0, min(1.0, page / (pages - 1))) if pages > 1 else 0.0
    return {"was_last_page": bool(was_last), "proportion": float(proportion)}


def _desktop_apply(hint, count):
    # epub_library EpubReaderDialog._apply_pending_page_hint (single page)
    c = max(1, int(count))
    if hint.get("was_last_page"):
        return c - 1
    target = round(float(hint.get("proportion") or 0.0) * (c - 1))
    return max(0, min(int(target), c - 1))


def test_page_hint_matches_desktop_formulas():
    for pages in range(0, 14):
        for page in range(0, max(1, pages)):
            mine = rm.capture_hint(page, pages)
            theirs = _desktop_capture(page, pages)
            assert mine == {"last": theirs["was_last_page"], "fraction": theirs["proportion"]}
            for count in (1, 2, 3, 7, 9, 20):
                assert rm.page_from_hint(mine, count) == _desktop_apply(theirs, count)
    assert rm.py_round(2.5) == 2 and rm.py_round(3.5) == 4
    assert rm.clamp_page(5, 6, spread=2) == 4 and rm.clamp_page(7, 7, spread=2) == 6 and rm.clamp_page(-3, 4) == 0
    assert rm.page_from_hint({"last": True}, 7, spread=2) == 6


def test_position_from_pref_and_labels():
    files = ["cover.xhtml", "ch001.xhtml", "ch002.xhtml"]
    pos = rm.Position.from_pref({"href": "OEBPS/CH002.XHTML", "fraction": 0.43, "page": 3, "mode": "bilingual",
                                 "pages": 8}, files)
    assert (pos.chapter, pos.href, pos.fraction, pos.page, pos.pages, pos.mode) == (2, "ch002.xhtml", 0.43, 3, 8,
                                                                                    "bilingual")
    by_index = rm.Position.from_pref({"href": "gone.xhtml", "chapter": 1, "fraction": "x", "mode": "evil"}, files)
    assert by_index.chapter == 1 and by_index.fraction == 0.0 and by_index.mode == "translated"
    assert rm.Position.from_pref({"href": "gone.xhtml", "chapter": 9}, files) is None
    assert rm.Position.from_pref(None, files) is None
    assert rm.Position().at_start and not rm.Position(chapter=0, page=2).at_start
    assert rm.book_percent(11, 0.43, 48) == 23 and rm.book_percent(47, 1.0, 48) == 100 and rm.book_percent(0, 0, 0) == 0
    assert rm.progress_label(12, 48, 43) == "Ch 12/48 · 43%"
    assert rm.page_label(2, 9) == "Page 3/9" and rm.page_label(0, 0) == "Page 1/1"
    assert rm.resume_label(12, 43) == "Resume at Ch 12 · 43%"
    assert rm.chapter_label(3, "") == "Chapter 3" and rm.chapter_label(3, "Rain") == "3. Rain"


def test_mode_availability():
    plain = rm.mode_availability(has_alternate=False, chapter_has_raw=True, chapter_has_translation=True)
    assert plain == {"original": False, "translated": True, "bilingual": False}
    overlay = rm.mode_availability(has_alternate=True, chapter_has_raw=True, chapter_has_translation=False)
    assert overlay == {"original": True, "translated": True, "bilingual": False}
    both = rm.mode_availability(has_alternate=True, chapter_has_raw=True, chapter_has_translation=True)
    assert both["bilingual"]


# =====================================================================================
# Prefs (reader positions, per-book settings)
# =====================================================================================


def test_prefs_reader_position_extra_fields_and_book_settings(tmp_path):
    clock = iter(range(100, 200))
    prefs = Prefs(tmp_path / "mobile_state.json", debounce=0.01, clock=lambda: next(clock))
    prefs.load()
    saved = prefs.set_reader_position("ab12cd34ef56", "ch012.xhtml", 0.43, page=3, mode="translated", chapter=12,
                                      pages=8, last=False)
    assert saved["chapter"] == 12 and saved["pages"] == 8 and saved["last"] is False
    legacy = prefs.set_reader_position("0123456789ab", "a.xhtml", 2.0)
    assert "chapter" not in legacy and legacy["fraction"] == 1.0
    assert prefs.reader_book_settings("ab12cd34ef56") == {}
    prefs.set_reader_book_settings("ab12cd34ef56", {"font_size": 18, "theme": 2})
    prefs.flush()
    again = Prefs(tmp_path / "mobile_state.json")
    data = again.load()
    assert data["reader_positions"]["ab12cd34ef56"]["chapter"] == 12
    assert again.reader_book_settings("ab12cd34ef56") == {"font_size": 18, "theme": 2}
    again.set_reader_book_settings("ab12cd34ef56", {})
    assert again.reader_book_settings("ab12cd34ef56") == {} and "ab12cd34ef56" not in again.get("reader_book_settings")
    assert again.clear_reader_position("ab12cd34ef56") and not again.clear_reader_position("ab12cd34ef56")
    with pytest.raises(ValueError):
        again.set_reader_book_settings("", {"font_size": 1})


# =====================================================================================
# blocks (fallback renderer content)
# =====================================================================================


def test_html_to_blocks_structure():
    html = ("<html><head><title>T</title><style>p{}</style></head><body><h2>Chapter 1</h2>"
            "<p>Hello <b>bold</b> and <i>it</i>.</p><p><img src='../images/a.png' alt='Map'/></p><hr/>"
            "<blockquote><p>quoted</p></blockquote><ul><li>one</li></ul><p>line<br/>break</p>"
            "<p><ruby>漢<rt>kan</rt></ruby>字</p><script>x()</script></body></html>")
    blocks = rb.html_to_blocks(html)
    kinds = [b.kind for b in blocks]
    assert kinds == ["heading", "para", "image", "rule", "quote", "item", "para", "para"]
    assert blocks[0].text == "Chapter 1" and blocks[0].level == 2
    assert blocks[1].text == "Hello bold and it." and ("bold", frozenset({"bold"})) in blocks[1].spans
    assert blocks[2].src == "../images/a.png" and blocks[2].alt == "Map"
    assert blocks[6].text == "line\nbreak" and blocks[7].text == "漢字"
    md = rb.blocks_to_markdown(blocks)
    assert md.startswith("## Chapter 1") and "**bold**" in md and "*it*" in md and "> quoted" in md and "- one" in md


def test_html_to_blocks_partial_stream_and_plain_text():
    partial = rb.html_to_blocks("<p>First para</p><p>Second, still stream")
    assert [b.text for b in partial] == ["First para", "Second, still stream"]
    cut = rb.html_to_blocks("<p>Done</p><p cla")
    assert [b.text for b in cut] == ["Done"]
    plain = rb.html_to_blocks("Line one\n\nLine &amp; two\n")
    assert [b.text for b in plain] == ["Line one", "Line & two"]
    assert rb.html_to_markdown("<h1>T*</h1><p>a_b</p>") == "# T\\*\n\na\\_b"
    shared = rb.html_to_blocks("<p>x</p>", shared=lambda html: [{"kind": "para", "text": "from core"}])
    assert [b.text for b in shared] == ["from core"]
    assert [b.text for b in rb.html_to_blocks("<p>x</p>", shared=lambda html: None)] == ["x"]


# =====================================================================================
# live panel classification (the shared live_stream classifier vs the desktop rules)
# =====================================================================================

_LIVE_STATUS_CHARS = set(
    "🚀📄📃📜📋✅⚠❌📚📦🔧📊🔍💾🖼🔄📌📸🧠🛰📡⏱⏳🟢🟡🟠🔴🎯📑📖🌐⚡🧪✨🎨💡"
    "🔠🗑🧹📂📁🔁🔂📝🔑🗝🔒🔓🚫⛔💬🌍🌏🌎🐛📈📉🤖🆗═─=[#"
)


class DesktopLiveClassifier:
    """epub_library ``EpubReaderDialog._classify_live_line`` at the parent commit (the oracle)."""

    def __init__(self):
        self.in_thinking = False
        self.streaming_text = False

    def classify(self, line):
        s = line.rstrip("\n")
        stripped = s.strip()
        low = stripped.lower()
        if "thinking complete" in low:
            self.in_thinking = False
            return "log"
        if stripped.startswith("\U0001f9e0") or " thinking..." in low:
            self.in_thinking = "thinking..." in low
            return "log"
        if self.in_thinking and (s.startswith("    ") or stripped == "​"):
            return "thinking"
        if "text streaming" in low or "first text token" in low:
            self.streaming_text = True
            return "log"
        if "stream complete" in low or "translation completed" in low:
            self.streaming_text = False
            return "log"
        if not stripped:
            return "content" if self.streaming_text else "log"
        first = stripped[0]
        if first in _LIVE_STATUS_CHARS:
            return "log"
        if stripped.startswith(("Traceback", 'File "', "[DEBUG]", "[INFO]", "[WARN", "[ERROR")):
            return "log"
        if first == "<":
            self.streaming_text = True
            return "content"
        return "content" if self.streaming_text else "log"


LIVE_LINES = [
    "🚀 Starting translation",
    "🧠 [gpt] Thinking...",
    "    planning the chapter",
    "    keeping honorifics",
    "🧠 [gpt] Thinking complete",
    "📡 text streaming started",
    "<h1>Chapter 3</h1>",
    "<p>The rain fell.</p>",
    "plain continuation",
    "[DEBUG] chunk 1/1",
    "✅ stream complete",
    "after the stream",
]

needs_live = pytest.mark.skipif(not _has("live_stream"), reason="shared live_stream module not importable")


@needs_live
def test_shared_live_classifier_matches_desktop_rules():
    import live_stream

    shared, desktop = live_stream.LiveLineClassifier(), DesktopLiveClassifier()
    lines = LIVE_LINES + ["", "    indented after thinking", "Traceback (most recent call last):", "x"]
    assert [shared.classify(line) for line in lines] == [desktop.classify(line) for line in lines]


@needs_live
def test_live_feed_routes_like_the_desktop_drain():
    feed = rl.LiveFeed("ch3.xhtml")
    # one drain per batch: the desktop pane gets the batch's thinking text, then its log lines
    assert not feed.feed(LIVE_LINES[:3]) and feed.side_count == 3
    assert feed.side_text.splitlines() == ["planning the chapter", "🚀 Starting translation", "🧠 [gpt] Thinking..."]
    assert feed.feed(LIVE_LINES[3:])
    assert feed.content == "<h1>Chapter 3</h1>\n<p>The rain fell.</p>\nplain continuation\n"
    side = feed.side_text.splitlines()
    assert side[3:] == ["keeping honorifics", "🧠 [gpt] Thinking complete", "📡 text streaming started",
                        "[DEBUG] chunk 1/1", "✅ stream complete", "after the stream"]
    assert feed.side_count == len(side) == 9
    assert feed.markdown().startswith("# Chapter 3\n\nThe rain fell.")
    feed2 = rl.LiveFeed("ch3.xhtml")
    feed2.feed([types.SimpleNamespace(text="<p>a</p>\n<p>b</p>")])  # a whole message, split like the desktop
    assert feed2.content == "<p>a</p>\n<p>b</p>\n" and feed2.content_version == 1
    assert rl.thinking_label(0) == "🧠 Thinking" and rl.thinking_label(4) == "🧠 Thinking (4)"


@needs_live
def test_live_outcome_strings_and_cleanup():
    calls = []
    done = rl.finish_outcome("/ws", "ch3.xhtml", stopped=False, chapter_completed=lambda f, c: True,
                             cleanup_incomplete=lambda f, c: calls.append((f, c)))
    assert done.completed and done.text == "✅ Translation finished — loading the translated chapter…" and calls == []
    stopped = rl.finish_outcome("/ws", "ch3.xhtml", stopped=True, chapter_completed=lambda f, c: False,
                                cleanup_incomplete=lambda f, c: calls.append((f, c)) or True)
    assert stopped.text == "⏹ Translation stopped — incomplete output cleared." and stopped.cleaned
    failed = rl.finish_outcome("/ws", "ch3.xhtml", stopped=False, chapter_completed=None, cleanup_incomplete=None)
    assert failed.text == "⚠️ Translation did not complete — incomplete output cleared." and not failed.completed
    assert calls == [("/ws", "ch3.xhtml")]
    assert rl.status_waiting("OEBPS/ch3.xhtml") == "\U0001f6f0️ Translating “ch3.xhtml” — waiting for stream…"
    assert rl.status_stopping() == "⏹ Stopping…" and rl.return_delay(True) == 2.2 and rl.return_delay(False) == 1.2
    assert rl.retranslate_question("Rain") == ("“Rain” is already translated.\n\nDelete its current translation and "
                                               "retranslate it live?")


# =====================================================================================
# session + document over the real shared cores
# =====================================================================================

needs_cores = pytest.mark.skipif(not all(_has(m) for m in ("reader_doc", "library_core", "reader_overlay",
                                                             "workspace_reader", "live_stream", "ebooklib")),
                                 reason="the shared reader cores are not importable here")


@pytest.fixture
def cores(tmp_path, monkeypatch):
    """The real reader cores, kept away from the user's Library / caches."""
    if not all(_has(m) for m in ("reader_doc", "library_core", "reader_overlay", "workspace_reader", "ebooklib")):
        pytest.skip("the shared reader cores are not importable here")
    import reader_doc

    monkeypatch.setenv("GLOSSARION_LIBRARY_DIR", str(tmp_path / "Library"))
    monkeypatch.setenv("OUTPUT_DIRECTORY", str(tmp_path / "Output"))
    monkeypatch.setattr(reader_doc, "_EPUB_CACHE_DIR_OVERRIDE", str(tmp_path / "epubcache"))
    (tmp_path / "epubcache").mkdir(exist_ok=True)
    return reader_doc


def _raw_epub(path: Path) -> Path:
    """The 12-chapter Korean self-test EPUB (prepare_assets), else the 3-chapter fixture builder."""
    from glossarion_mobile.diagnostics.fixtures import build_tiny_epub

    source = APP_DIR / "assets" / "selftest" / "selftest_ko_12ch.epub"
    if source.is_file():
        shutil.copy(source, path)
        return path
    return build_tiny_epub(path, chapters=3)


@pytest.fixture
def novel(tmp_path, cores):
    """Raw EPUB + an in-progress workspace: chapter 1 translated (with an image), chapter 2 pending."""
    raw = _raw_epub(tmp_path / "Novel.epub")
    workspace = tmp_path / "Output" / "Novel"
    workspace.mkdir(parents=True)
    (workspace / "source_epub.txt").write_text(str(raw), encoding="utf-8")
    (workspace / "images").mkdir()
    (workspace / "images" / "emblem.png").write_bytes(b"\x89PNG\r\n\x1a\nworkspace")
    (workspace / "response_chapter0001.html").write_text(
        "<html><head><title>Ch One EN</title></head><body><h1>Ch One EN</h1><p>The sword sang. rain</p>"
        "<p><img src='../images/emblem.png'/></p></body></html>", encoding="utf-8")
    progress = {"chapters": {"1": {"status": "completed", "original_basename": "chapter0001.xhtml",
                                   "output_file": "response_chapter0001.html"},
                             "2": {"status": "pending", "original_basename": "chapter0002.xhtml"}},
                "chapter_chunks": {}, "version": "2.1"}
    (workspace / "translation_progress.json").write_text(json.dumps(progress), encoding="utf-8")
    book = {"name": "Novel", "path": str(workspace), "output_folder": str(workspace), "is_in_progress": True}
    return types.SimpleNamespace(raw=raw, workspace=workspace, book=book, tmp=tmp_path)


def _overlay_session(novel):
    from glossarion_mobile.ui.reader import session as rs

    engine = rs.DocEngine()
    session = rs.ReaderSession(rs.plan_open(novel.book, engine=engine), engine=engine,
                               cache_dir=str(novel.tmp / "cache"))
    session.load()
    return session


def test_plan_open_maps_the_shared_decision(novel, tmp_path):
    from glossarion_mobile.diagnostics.fixtures import build_tiny_epub
    from glossarion_mobile.ui.reader import session as rs

    engine = rs.DocEngine()
    overlay = rs.plan_open(novel.book, engine=engine)
    assert overlay.mode == rs.MODE_OVERLAY and overlay.epub_path == str(novel.raw) == overlay.raw_path
    assert overlay.output_folder == str(novel.workspace) and overlay.toc_dir == str(novel.workspace)
    assert overlay.css_dirs and overlay.css_dirs[0].endswith("css")
    raw_only = rs.plan_open(novel.book, raw_only=True, engine=engine)
    assert raw_only.mode == rs.MODE_PLAIN and raw_only.initial_raw and raw_only.epub_path == str(novel.raw)
    compiled = build_tiny_epub(tmp_path / "Novel_translated.epub", chapters=3)
    dual = rs.plan_open({"name": "Novel", "path": str(compiled), "raw_source_path": str(novel.raw)}, engine=engine)
    assert dual.mode == rs.MODE_DUAL and dual.translated_path == str(compiled) and dual.raw_path == str(novel.raw)
    plain = rs.plan_open({"name": "Novel", "path": str(compiled)}, engine=engine)
    assert plain.mode == rs.MODE_PLAIN and plain.epub_path == str(compiled)
    pdf = tmp_path / "Doc.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    pdf_ws = tmp_path / "Output" / "Doc"
    pdf_ws.mkdir()
    (pdf_ws / "translation_progress.json").write_text('{"chapters": {}}', encoding="utf-8")
    ws_plan = rs.plan_open({"name": "Doc", "path": str(pdf_ws), "output_folder": str(pdf_ws),
                            "raw_source_path": str(pdf)}, engine=engine)
    assert ws_plan.mode == rs.MODE_WORKSPACE and ws_plan.source_path == str(pdf) and not ws_plan.initial_raw
    txt_ws = tmp_path / "Output" / "Story"
    txt_ws.mkdir()
    (txt_ws / "translation_progress.json").write_text('{"chapters": {}}', encoding="utf-8")
    story = rs.plan_open({"name": "Story", "path": str(txt_ws), "output_folder": str(txt_ws)}, engine=engine)
    assert story.mode == rs.MODE_WORKSPACE  # the desktop's OS-viewer case, read in-app
    missing = rs.plan_open({"name": "Gone", "path": str(tmp_path / "gone.pdf")}, engine=engine)
    assert missing.error and "Reader opens EPUB" in missing.error
    from glossarion_mobile.services.library import SharedCore

    with pytest.raises(rs.CoreMissing):
        rs.plan_open(novel.book, engine=rs.DocEngine(SharedCore({"library_core": types.ModuleType("library_core")})))


def test_overlay_session_over_the_real_cores(novel):
    session = _overlay_session(novel)
    assert session.plan.mode == "overlay" and session.count >= 3
    assert session.filenames[:3] == ["chapter0001.xhtml", "chapter0002.xhtml", "chapter0003.xhtml"]
    assert session.overlay_applied and session.has_alternate
    titles = session.titles()
    assert titles[0] == "Ch One EN" and titles[1] != "Ch One EN"
    info = session.chapter_info(0)
    assert info.status == "completed" and info.has_translation and info.number == 1
    assert session.chapter_info(1).status in ("", "pending") and not session.chapter_info(1).has_translation
    assert "The sword sang." in session.chapter_html(0) and "The sword sang." not in session.chapter_html(0, rm.ORIGINAL)
    assert session.available_modes(0) == {"original": True, "translated": True, "bilingual": True}
    assert session.available_modes(1)["bilingual"] is False
    assert "glr-bi" in session.chapter_html(0, rm.BILINGUAL)
    # Translate button / target (desktop _update_translate_btn_visibility, _on_translate_current_chapter)
    assert not session.translate_visible(0) and session.translate_visible(1)
    assert session.translate_target(1) == (str(novel.raw), "chapter0002.xhtml", "")
    assert session.live_output_folder(0) == str(novel.workspace)
    # a new response lands: only chapter 2 changes
    assert session.refresh_overlay() == set()
    (novel.workspace / "response_chapter0002.html").write_text(
        "<html><head><title>Ch Two EN</title></head><body><h1>Ch Two EN</h1></body></html>", encoding="utf-8")
    progress = json.loads((novel.workspace / "translation_progress.json").read_text(encoding="utf-8"))
    progress["chapters"]["2"].update(status="completed", output_file="response_chapter0002.html")
    (novel.workspace / "translation_progress.json").write_text(json.dumps(progress), encoding="utf-8")
    generation = session.generation
    assert session.refresh_overlay() == {1}
    assert session.titles()[1] == "Ch Two EN" and session.generation > generation
    assert session.set_flavor(rm.ORIGINAL) and not session.set_flavor(rm.ORIGINAL)
    assert session.titles()[0] != "Ch One EN"
    batches = []
    rows = session.search("의", on_batch=lambda rows, done: batches.append((len(rows), done)))
    assert rows and batches and batches[-1][1] is True and sum(n for n, _ in batches) == len(rows)
    assert {"chapter_idx", "local_occurrence", "excerpt", "title"} <= set(rows[0])
    assert session.image_bytes("../images/emblem.png")  # from the EPUB (or the workspace images)
    session.close()


def test_dual_session_swaps_epubs(novel, tmp_path):
    from glossarion_mobile.diagnostics.fixtures import build_tiny_epub
    from glossarion_mobile.ui.reader import session as rs

    compiled = build_tiny_epub(tmp_path / "Novel_out.epub", chapters=3)
    engine = rs.DocEngine()
    session = rs.ReaderSession(rs.plan_open({"name": "Novel", "path": str(compiled),
                                             "raw_source_path": str(novel.raw)}, engine=engine), engine=engine)
    session.load()
    translated_titles = session.titles()
    assert session.has_alternate and session.count == 3
    assert not session.translate_visible(0)  # the compiled view: every chapter is translated
    session.set_flavor(rm.ORIGINAL)
    assert session.titles() != translated_titles and session.translate_visible(0)
    assert session.translate_target(0)[0] == str(novel.raw)
    session.set_flavor(rm.BILINGUAL)
    assert "glr-bi" in session.chapter_html(0)
    session.set_flavor(rm.TRANSLATED)
    assert session.titles() == translated_titles


def test_workspace_session(cores, tmp_path):
    from glossarion_mobile.ui.reader import session as rs

    ws = tmp_path / "Output" / "Story"
    ws.mkdir(parents=True)
    (ws / "response_001.html").write_text("<h1>Translated 1</h1><p>text</p>", encoding="utf-8")
    progress = {"chapters": {"1": {"status": "completed", "original_basename": "001.txt",
                                   "output_file": "response_001.html", "actual_num": 1}}}
    (ws / "translation_progress.json").write_text(json.dumps(progress), encoding="utf-8")
    engine = rs.DocEngine()
    session = rs.ReaderSession(rs.plan_open({"name": "Story", "path": str(ws), "output_folder": str(ws)},
                                            engine=engine), engine=engine)
    session.load()
    assert session.plan.mode == rs.MODE_WORKSPACE and session.count == 1 and not session.has_alternate
    assert "Translated 1" in session.chapter_html(0)
    assert session.translate_target(0)[2] and not session.translate_visible(0)


def test_document_builder_pages_images_and_scroll_all(novel, server):
    from glossarion_mobile.ui.reader.document import DocumentBuilder, build_native_blocks

    session = _overlay_session(novel)
    builder = DocumentBuilder(session, server.register_image)
    settings = rm.ReaderSettings(font_size=16, theme=3, font_family="Serif")
    built = builder.build(0, settings=settings, layout=rm.LAYOUT_SINGLE, theme=DESKTOP_THEMES[3], doc_id="d9",
                          event_url=server.event_path, hint={"last": True})
    assert built.paged and built.chapter == 0 and built.has_page_bridge and built.fragment == ""
    assert "maximum-scale=1" in built.html and "100dvh" in built.html  # the shared mobile shell
    assert f'"{server.event_path}"' in built.html and "var INITIAL_PAGE = 1000000;" in built.html
    cfg = json.loads(built.html.split("window.__GLRDR_CFG=", 1)[1].split(";</script>", 1)[0])
    assert cfg == {"doc": "d9", "layout": "single_page", "zones": True}
    assert "font-family: Georgia, 'Noto Serif'" in built.html and "#0d1117" in built.html
    images = re.findall(r'src="(/[^"]+/img/[^"]+)"', built.html)
    assert images and all(i.startswith(server.base_path + "img/") for i in images)
    assert _get(server.origin + images[0])[2].startswith(b"\x89PNG")
    url = server.publish(built.html)
    status, _h, body = _get(url)
    assert status == 200 and b"window.GLRDR" in body and b"G.applyStyle" in body
    fraction = builder.build(1, settings=settings, layout=rm.LAYOUT_SCROLL, theme=DESKTOP_THEMES[0], doc_id="d10",
                             event_url=server.event_path, hint={"fraction": 0.25})
    assert not fraction.paged and fraction.fragment == "f=0.25"
    every = builder.build(0, settings=settings, layout=rm.LAYOUT_ALL, theme=DESKTOP_THEMES[0], doc_id="d11",
                          event_url=server.event_path, hint={"fraction": 0.5})
    assert every.html.count("data-glrdr-ch=") == session.count and "Chapter 2:" in every.html
    assert json.loads(every.html.split("window.__GLRDR_CFG=", 1)[1].split(";</script>", 1)[0])["all"] is True
    blocks = build_native_blocks(session, 0)
    assert blocks[0].kind == "heading" and blocks[0].text == "Ch One EN"
    builder.close()
    session.close()


def test_native_toc_rows(novel):
    from glossarion_mobile.ui.reader.model import toc_rows

    session = _overlay_session(novel)
    if not session.native_toc:
        pytest.skip("the fixture EPUB has no toc.ncx")
    first = session.native_toc[0]
    assert first["chapter_index"] == 0 and first["title"]
    rows = toc_rows(session.titles(), session.display_numbers, session.statuses(), session.native_toc)
    assert rows[0].chapter == 0 and rows[0].number is None
    plain_rows = toc_rows(session.titles(), session.display_numbers, session.statuses())
    assert [(r.chapter, r.number) for r in plain_rows[:2]] == [(0, 1), (1, 2)] and plain_rows[0].status == "completed"
    session.close()


# =====================================================================================
# Flet layer
# =====================================================================================


class FakeConfig:
    def __init__(self, data=None):
        self.data = dict(data or {})
        self.writes = []

    def get(self, key, default=None):
        return self.data.get(key, default)

    def set_many(self, values):
        self.writes.append(dict(values))
        self.data.update(values)

    def snapshot(self):
        return dict(self.data)


class FakeJobs:
    def __init__(self):
        self.submitted = []
        self.transitions = []
        self.stops = []
        self.busy = False
        self.buffers = {}

    def has_kind(self, kind):
        return kind == "single_chapter"

    def submit(self, spec):
        self.submitted.append(spec)
        return "job1"

    def on_transition(self, callback):
        self.transitions.append(callback)
        return lambda: self.transitions.remove(callback)

    def log_buffer(self, job_id):
        return self.buffers.get(job_id)

    def request_stop(self, job_id=None, **kwargs):
        self.stops.append(job_id)
        return "graceful"


def _screen(novel, *, config=None, route="/reader/ab12cd34ef56"):
    from glossarion_mobile.services.library import SharedCore
    from glossarion_mobile.ui.reader.reader_view import ReaderDeps, ReaderScreen
    from glossarion_mobile.ui.router import parse_route

    prefs = Prefs(novel.tmp / "mobile_state.json", debounce=0.01)
    prefs.load()
    page = types.SimpleNamespace(width=412, height=860, views=[], platform=None, web=False,
                                 show_dialog=lambda d: None, pop_dialog=lambda: None)
    notes = []
    deps = ReaderDeps(page=page, prefs=prefs, config=FakeConfig(config), jobs=FakeJobs(),
                      resolve=lambda bid: {"book": dict(novel.book)}, server=None, core=SharedCore(),
                      notify=lambda *a: notes.append(a), webview_ok=lambda: False, cache_dir=str(novel.tmp / "cache"))
    screen = ReaderScreen(parse_route(route), deps)
    screen.notes = notes
    return screen


@needs_flet
def test_reader_screen_opens_natively_and_handles_page_events(novel):
    async def scenario():
        screen = _screen(novel, route="/reader/ab12cd34ef56?ch=1")
        await screen.open()
        assert screen.state == "ready" and screen.renderer == "native" and screen.index == 1
        assert screen.session.plan.mode == "overlay" and len(screen.themes) == 6
        assert screen.chrome.book_title.value == "Novel" and screen.chrome.mode_buttons.visible
        assert screen.chrome.translate_button.visible  # chapter 2 is not translated yet
        total = screen.session.count
        assert screen.chrome.progress_text.value == f"Ch 2/{total} · {rm.book_percent(1, 0, total)}%"
        assert screen.fallback.list_view.controls  # the native renderer drew the chapter
        # the page's events (shared shell schema), as if the WebView had posted them
        screen.layout = rm.LAYOUT_SINGLE
        screen.deduper.set_document("d1", 1)
        screen.handle_payload({"type": "ready", "seq": 1, "page": 2, "count": 5, "chapter": 1})
        screen.handle_payload({"type": "page", "seq": 2, "page": 4, "count": 5, "chapter": 1, "reason": "next"})
        assert (screen.page_no, screen.page_count, screen.last_page, screen.fraction) == (4, 5, True, 1.0)
        assert screen.chrome.page_text.value == "Page 5/5"
        assert screen.handle_payload({"type": "page", "seq": 2, "page": 4, "count": 5, "chapter": 1,
                                      "reason": "next"}) is None  # the fetch copy
        assert screen.handle_payload({"type": "page", "seq": 3, "page": 0, "chapter": 0}) is None  # stale page
        screen.handle_payload({"type": "tap", "seq": 4, "zone": "center", "chapter": 1})
        assert not screen.chrome.visible
        screen.handle_payload({"type": "selection", "seq": 5, "text": "rain", "chapter": 1})
        assert screen.selection.visible and screen.selection.text == "rain"
        assert screen.selection.web_chip.label.value == "Define on web"
        screen.handle_payload({"type": "selrect", "seq": 6, "doc": "d1", "x": 0.2, "y": 0.5, "w": 0.1, "h": 0.03})
        assert screen.selection.container.top == pytest.approx(0.5 * 860 - 60)
        screen.handle_payload({"type": "selection", "seq": 7, "text": "", "doc": "d1"})
        assert not screen.selection.visible
        screen.handle_payload({"type": "scale", "seq": 8, "scale": 1.3, "chapter": 1})  # +2 pt, saved
        assert screen.settings.font_size == 16
        screen._flush_style()
        assert screen.deps.config.data["epub_reader_font_size"] == 16
        # edge: past the last page -> next chapter; before the first -> previous chapter's last page
        screen.handle_payload({"type": "edge", "seq": 9, "edge": "end", "chapter": 1})
        await asyncio.sleep(0.3)
        assert screen.index == 2
        saved = screen.save_position()
        assert saved["href"] == "chapter0003.xhtml" and saved["chapter"] == 2 and saved["mode"] == "translated"
        screen.handle_payload({"type": "edge", "seq": 1, "edge": "start", "chapter": 2})
        await asyncio.sleep(0.3)
        assert screen.index == 1 and screen.last_page
        screen._on_link("chapter0001.xhtml#top", False)
        await asyncio.sleep(0.3)
        assert screen.index == 0
        screen.dispose()
        assert screen.deps.prefs.reader_position("ab12cd34ef56")["href"] == "chapter0001.xhtml"

    asyncio.run(scenario())


@needs_flet
def test_reader_opens_at_the_callers_chapter_file(novel):
    from glossarion_mobile.ui.reader.reader_view import ReaderScreen

    async def scenario():
        base = _screen(novel, route="/reader/ab12cd34ef56?ch=5")
        # the Book page passes its spine row as ?ch= and the chapter's file name in-process: the file wins
        screen = ReaderScreen(base.match, base.deps, args={"book": dict(novel.book),
                                                           "chapter_filename": "OEBPS/Chapter0003.xhtml"})
        await screen.open()
        assert screen.index == 2
        screen.dispose()
        missing = ReaderScreen(base.match, base.deps, args={"book": dict(novel.book), "chapter_filename": "gone.xhtml"})
        await missing.open()
        assert missing.index == 5  # unknown file: the route's index
        missing.dispose()

    asyncio.run(scenario())


@needs_flet
def test_reader_settings_scopes_and_resume(novel):
    async def scenario():
        screen = _screen(novel, config={"epub_reader_theme": 1})
        screen.deps.prefs.set_reader_position("ab12cd34ef56", "chapter0002.xhtml", 0.5, page=2, mode="original")
        await screen.open()
        assert screen.session.flavor == rm.ORIGINAL and screen.index == 0  # saved mode, opened at the start
        assert screen.resume_offer is not None and screen.resume_offer.chapter == 1
        total = screen.session.count
        assert screen.notes[-1][0] == f"Resume at Ch 2 · {rm.book_percent(1, 0.5, total)}%"
        screen._apply_settings({"theme": 4, "line_spacing": 2.4}, "book")
        screen._flush_style()
        assert screen.deps.prefs.reader_book_settings("ab12cd34ef56") == {"theme": 4, "line_spacing": 2.4}
        assert not any("epub_reader_theme" in w for w in screen.deps.config.writes)
        assert screen.theme["name"] == "Forest" and screen.settings.overridden == frozenset({"theme", "line_spacing"})
        screen._apply_settings({"theme": 2}, "all")
        screen._flush_style()
        assert screen.deps.config.data["epub_reader_theme"] == 2
        assert screen.deps.prefs.reader_book_settings("ab12cd34ef56") == {"line_spacing": 2.4}
        screen._apply_settings({"tap_zones": False, "keep_screen_on": False}, "all")
        screen._flush_style()
        assert screen.deps.prefs.get("reader_prefs") == {"tap_zones": False, "keep_screen_on": False}
        await screen.set_mode(rm.TRANSLATED)
        assert screen.session.flavor == rm.TRANSLATED and screen.index == 0
        screen.dispose()

    asyncio.run(scenario())


@needs_flet
def test_reader_live_translation_flow(novel):
    from glossarion_mobile.services.dispatcher import LogBuffer

    async def scenario():
        screen = _screen(novel)
        await screen.open()
        await screen.go_chapter(1)
        await screen.translate_chapter()
        jobs = screen.deps.jobs
        spec = jobs.submitted[-1]
        assert spec.kind == "single_chapter"
        assert spec.params == {"chapter_file": "chapter0002.xhtml", "force_stream_all": True}
        assert spec.inputs == (str(novel.raw),) and spec.origin["bid"] == "ab12cd34ef56"
        live = screen.live
        assert live.active and screen.chrome.translate_button.content == "🛰️ Live view"
        buffer = LogBuffer(100)
        jobs.buffers["job1"] = buffer
        buffer.extend(LIVE_LINES)
        for callback in list(jobs.transitions):
            callback(types.SimpleNamespace(id="job1", is_terminal=False), None)
        assert live.feed.content.startswith("<h1>Chapter 3</h1>") and live.panel.state == "streaming"
        live.panel._on_stop()
        assert jobs.stops == ["job1"] and live.panel.status.value == "⏹ Stopping…"
        # the run stopped before the chapter completed: the partial response is cleaned up
        (novel.workspace / "response_chapter0002.html").write_text("<p>half", encoding="utf-8")
        progress = json.loads((novel.workspace / "translation_progress.json").read_text(encoding="utf-8"))
        progress["chapters"]["2"].update(status="in_progress", output_file="response_chapter0002.html")
        (novel.workspace / "translation_progress.json").write_text(json.dumps(progress), encoding="utf-8")
        stopped = types.SimpleNamespace(id="job1", is_terminal=True, stopped=True, output_dir=str(novel.workspace),
                                        output_dirs={})
        for callback in list(jobs.transitions):
            callback(stopped, None)
        for _ in range(60):
            await asyncio.sleep(0.05)
            if live.panel.state != "streaming":
                break
        assert not live.active and live.panel.state == "stopped"
        assert live.panel.status.value == "⏹ Translation stopped — incomplete output cleared."
        assert not (novel.workspace / "response_chapter0002.html").exists()
        progress = json.loads((novel.workspace / "translation_progress.json").read_text(encoding="utf-8"))
        assert progress["chapters"]["2"]["status"] == "pending"
        screen.dispose()

    asyncio.run(scenario())


@needs_flet
def test_reader_live_panel_does_not_wait_for_a_stream_when_streaming_is_off(novel):
    """devfix4 #14: with the mobile Streaming switch off the live job does not stream (job_kinds.single_chapter
    drops force_stream_all), so the panel says the chapter appears when it is done; absent keys mean on."""
    off = {"enable_streaming": False, "stream_thinking_logs": False, "allow_batch_stream_logs": False,
           "allow_authgpt_batch_stream_logs": False}

    async def scenario(config, expected):
        screen = _screen(novel, config=config)
        await screen.open()
        await screen.go_chapter(1)
        await screen.translate_chapter()
        assert screen.deps.jobs.submitted[-1].params == {"chapter_file": "chapter0002.xhtml", "force_stream_all": True}
        assert screen.live.panel.status.value == expected
        screen.dispose()

    asyncio.run(scenario(off, "\U0001f6f0️ Translating “chapter0002.xhtml” — Streaming is off: the chapter appears "
                              "when it is done"))
    asyncio.run(scenario(None, "\U0001f6f0️ Translating “chapter0002.xhtml” — waiting for stream…"))


@needs_flet
def test_completed_chapter_asks_before_retranslating(novel):
    async def scenario():
        screen = _screen(novel)
        await screen.open()
        shown = []
        screen.page.show_dialog = shown.append
        task = asyncio.ensure_future(screen.translate_chapter())  # chapter 1 is translated
        await asyncio.sleep(0.1)
        dialog = shown[-1]
        assert dialog.title.value == "Retranslate chapter"
        assert dialog.content.content.controls[0].value.startswith("“Ch One EN” is already translated.")
        dialog.actions[0].on_click(None)  # No
        await task
        assert screen.deps.jobs.submitted == [] and (novel.workspace / "response_chapter0001.html").exists()
        screen.deps.jobs.busy = True
        await screen.go_chapter(1)
        await screen.translate_chapter()
        assert shown[-1].content.content.controls[0].value.startswith("A translation is already running.")
        screen.dispose()

    asyncio.run(scenario())


@needs_flet
@needs_live
def test_fallback_page_live_panel_aa_sheet_and_drawer():
    import flet as ft

    from glossarion_mobile.ui.reader.aa_sheet import AaSheet
    from glossarion_mobile.ui.reader.fallback_view import FallbackPage
    from glossarion_mobile.ui.reader.live_panel import LivePanel
    from glossarion_mobile.ui.reader.toc_drawer import ChaptersDrawer, TocRow

    zones, steps, paragraphs = [], [], []
    page = FallbackPage(on_tap_zone=zones.append, on_pinch=steps.append, on_pinch_end=lambda: None,
                        on_paragraph=paragraphs.append, on_scroll=lambda f: None)
    page.set_size(300, 600)
    blocks = rb.html_to_blocks("<h1>T</h1><p>a <b>b</b></p><img src='x.png' alt='pic'/><hr/>")
    page.render(blocks, theme=DESKTOP_THEMES[2], settings=rm.ReaderSettings(font_size=12), images={"x.png": b"img"})
    controls = page.list_view.controls
    assert len(controls) == 4 and page.container.bgcolor == "#f4ecd8"
    assert controls[0].content.size == round(16 * 1.6) and controls[0].content.color == "#3e2c1c"
    assert isinstance(controls[2].content, ft.Image) and isinstance(controls[3], ft.Divider)
    para = controls[1].content.content
    assert para.size == 16 and [s.text for s in para.spans] == ["a ", "b"]
    controls[1].on_long_press(None)
    assert paragraphs == ["a b"]
    for x in (10, 150, 290):
        page._on_tap(types.SimpleNamespace(local_position=types.SimpleNamespace(x=x)))
    assert zones == ["prev", "centre", "next"]
    page._on_scale_update(types.SimpleNamespace(pointer_count=2, scale=1.2))
    page._on_scale_update(types.SimpleNamespace(pointer_count=2, scale=1.25))
    page._on_scale_update(types.SimpleNamespace(pointer_count=2, scale=0.9))
    assert steps == [1, -1]

    feed = rl.LiveFeed("ch3.xhtml")
    stops = []
    panel = LivePanel(chapter_file="ch3.xhtml", feed=feed, on_stop=lambda: stops.append(1), clock=lambda: 100.0)
    assert panel.state == "waiting" and panel.status.value.startswith("\U0001f6f0️ Translating “ch3.xhtml”")
    panel.add_lines(LIVE_LINES)
    panel.render(force=True)
    assert panel.state == "streaming" and panel.content_md.value.startswith("# Chapter 3")
    assert panel.thinking_tile.title.value == f"🧠 Thinking ({feed.side_count})"
    assert "planning the chapter" in panel.side_text.value
    panel.finish("⏹ Translation stopped — incomplete output cleared.", stopped=True)
    assert panel.state == "stopped" and panel.stop_button.disabled
    panel._on_stop()
    assert stops == []

    changes = []
    sheet = AaSheet(rm.ReaderSettings(), DESKTOP_THEMES, double_allowed=False,
                    on_change=lambda c, scope: changes.append((c, scope)))
    sheet._set_size(40)
    sheet._pick_theme(5)
    assert changes == [({"font_size": 32}, "all"), ({"theme": 5, "follow_app_theme": False}, "all")]
    assert [s.value for s in sheet.layout_buttons.segments] == ["single_page", "scroll", "all_scroll"]
    assert len(sheet.swatch_row.controls) == 6 and sheet.swatch_row.controls[5].bgcolor == "#2e1a2e"
    tablet = AaSheet(rm.ReaderSettings(), DESKTOP_THEMES, double_allowed=True)
    assert "double_page" in [s.value for s in tablet.layout_buttons.segments]

    opened = []
    drawer = ChaptersDrawer(on_open=opened.append)
    drawer.set_rows([TocRow(0, "One", 1, "completed"), TocRow(1, "Two", 2, "")], 1, native_available=False,
                    native_on=True, show_special=True)
    assert drawer.native_switch.disabled and not drawer.native_switch.value and drawer.native_reason.visible
    tile = drawer.list_view.controls[1]
    assert tile.selected and tile.title.value == "2. Two" and drawer.list_view.controls[0].leading is not None
    tile.on_click(None)
    assert opened and opened[0].chapter == 1
    drawer.set_current(0)
    assert drawer.list_view.controls[0].selected and not drawer.list_view.controls[1].selected


@needs_flet
def test_reader_feature_routes_and_opens(tmp_path):
    from glossarion_mobile.ui.reader.feature import IMPLEMENTED_ROUTES, ReaderFeature
    from glossarion_mobile.ui.reader.reader_view import ReaderScreen
    from glossarion_mobile.ui.router import parse_route

    prefs = Prefs(tmp_path / "mobile_state.json", debounce=0.01)
    prefs.load()
    navigated = []
    fallback_calls = []
    shell = types.SimpleNamespace(screen_factory=lambda match: fallback_calls.append(match.name) or "fallback")
    app = types.SimpleNamespace(page=types.SimpleNamespace(width=412, height=860, platform=None, web=False, views=[]),
                                shell=shell, prefs=prefs, navigate_to=lambda *a: navigated.append(a),
                                notify=lambda *a: None, paths=None, state=None)

    async def scenario():
        feature = await ReaderFeature.install(app)
        assert app.reader is feature and IMPLEMENTED_ROUTES == {"reader"}
        epub = tmp_path / "Shared.epub"
        epub.write_bytes(b"PK")
        bid = feature.open_file(str(epub))
        assert navigated[-1] == ("reader", {"bid": bid}, None) and len(bid) == 12
        assert feature.resolve(bid) == {"path": str(epub)}
        other = tmp_path / "Other.epub"
        other.write_bytes(b"PK")
        book = {"name": "B", "path": str(other)}
        bid2 = feature.open_book(book, chapter=3, mode="original")
        assert bid2 != bid and navigated[-1] == ("reader", {"bid": bid2}, {"ch": 3, "mode": "original"})
        assert feature.resolve(bid2) == {"book": book}
        screen = shell.screen_factory(parse_route(f"/reader/{bid}"))
        assert isinstance(screen, ReaderScreen) and screen.args["path"] == str(epub)
        assert screen.deps.server is None  # no WebView on this platform -> no server started
        assert shell.screen_factory(parse_route("/jobs")) == "fallback" and fallback_calls == ["jobs"]
        assert feature.resolve("ffffffffffff") is None

    asyncio.run(scenario())


def test_single_chapter_job_kind_sets_owner_flags(tmp_path, monkeypatch):
    from glossarion_mobile.job_kinds import single_chapter
    from glossarion_mobile.services.jobs import JobError

    epub = tmp_path / "Book.epub"
    epub.write_bytes(b"PK")
    seen = {}

    def fake_run(ctx, files):
        seen["filter"] = ctx.owner._single_chapter_filter
        seen["stream"] = ctx.owner._force_stream_all
        seen["files"] = files
        return {"ok": True, "outputs": []}

    monkeypatch.setattr(single_chapter, "run_translation", fake_run)
    owner = types.SimpleNamespace()
    logs = []
    ctx = types.SimpleNamespace(inputs=(str(epub),), params={"chapter_file": "OEBPS/chapter0001.xhtml",
                                                             "force_stream_all": True}, owner=owner, log=logs.append)
    assert single_chapter.run(ctx) == {"ok": True, "outputs": []}
    assert seen == {"filter": "chapter0001.xhtml", "stream": True, "files": [str(epub)]}
    assert owner._single_chapter_filter is None and owner._force_stream_all is False
    assert logs == ["🎯 Queued single-chapter translation: chapter0001.xhtml (Book.epub)"]
    with pytest.raises(JobError):
        single_chapter.run(types.SimpleNamespace(inputs=(str(epub),), params={}, owner=owner, log=logs.append))
    assert single_chapter.KINDS["single_chapter"]["stop_kind"] == "translation"


# =====================================================================================
# The real app shell on the fake Flet session (ReaderFeature installed like Integrate will)
# =====================================================================================

_UF = None


def _ui_helpers():
    """test_ui_foundations' app starter (one copy of the fake session helpers)."""
    global _UF
    if _UF is None:
        spec = importlib.util.spec_from_file_location("_glossarion_uf_helpers_reader",
                                                      Path(__file__).with_name("test_ui_foundations.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _UF = module
    return _UF


@pytest.fixture
def android_shell(novel, monkeypatch):
    """The real app on the fake Flet session as Android (storage under tmp), with the Reader
    feature installed over ``novel``'s book: ``await android_shell.open(route)`` gives the
    ``ReaderScreen`` once its first page is up."""
    if not (_has("flet") and _has("msgpack")):
        pytest.skip("flet / msgpack not installed")
    helpers = _ui_helpers()
    tb = helpers._TB
    storage_dirs = {}
    for name in ("data", "cache", "temp"):
        folder = novel.tmp / "storage" / name
        folder.mkdir(parents=True)
        storage_dirs[name] = folder
        monkeypatch.setenv(f"FLET_APP_STORAGE_{name.upper()}", str(folder))
    for key in tb._ENV_INPUTS:
        if key not in ("OUTPUT_DIRECTORY", "GLOSSARION_LIBRARY_DIR"):
            monkeypatch.delenv(key, raising=False)
    tb.rb.reset(restore_env=True)
    tb.secure_keys.reset()
    tb.rb.bootstrap(app_dir=APP_DIR, force=True)
    monkeypatch.setattr(tb.rb, "start_warm_import", lambda modules=None, *, on_done=None: on_done and on_done(
        {"ok": True, "modules": 0, "secs": 0.0, "failed": {}, "requested": []}))
    (storage_dirs["data"] / "mobile_state.json").write_text(json.dumps({"welcome_completed": True}), encoding="utf-8")
    book = dict(novel.book)
    shell = types.SimpleNamespace(helpers=helpers, conn=None, page=None, app=None, feature=None)

    async def open_reader(route="/reader/ab12cd34ef56?ch=1"):
        from glossarion_mobile.ui.reader.feature import ReaderFeature
        from glossarion_mobile.ui.reader.reader_view import ReaderScreen

        _main, shell.conn, _session, shell.page, shell.app = await helpers._start("android")
        shell.app.library = types.SimpleNamespace(
            book_for_bid=lambda bid: dict(book) if bid == "ab12cd34ef56" else None, bid_for=lambda b: "ab12cd34ef56")
        shell.feature = await ReaderFeature.install(shell.app)
        await shell.app.navigate(route)
        screen = shell.feature.active
        assert isinstance(screen, ReaderScreen)
        assert await helpers._wait(lambda: screen.state == "ready" and screen.webview is not None, 20)
        return screen

    async def close():
        if shell.feature is not None:
            shell.feature.detach()
        if shell.app is not None:
            await helpers._stop(shell.app)

    shell.open, shell.close = open_reader, close
    yield shell
    tb._join_app_io_threads()
    tb.secure_keys.reset()
    tb.rb.reset(restore_env=True)


def _load_requests(conn) -> list:
    """The URLs of every WebView ``load_request`` the app sent to the (fake) client, in order."""
    from flet.messaging.protocol import MessageAction

    return [(m.body.args or {}).get("url") for m in conn.messages
            if m.action == MessageAction.INVOKE_METHOD and m.body.name == "load_request"]


@needs_flet
@pytest.mark.skipif(not _has("msgpack"), reason="msgpack not installed")
def test_reader_in_app_shell_with_webview_and_server(android_shell, monkeypatch):
    from glossarion_mobile.ui.reader.reader_view import ReaderScreen

    built = []  # (url passed in, the control's url right after it was built)
    make_webview = ReaderScreen._make_webview

    def recording_make_webview(self, url):
        control = make_webview(self, url)
        built.append((url, control.url, control))
        return control

    monkeypatch.setattr(ReaderScreen, "_make_webview", recording_make_webview)

    async def scenario():
        try:
            screen = await android_shell.open()
            conn, page, feature, app = android_shell.conn, android_shell.page, android_shell.feature, android_shell.app
            view = page.views[-1]
            assert view.route == "/reader/ab12cd34ef56?ch=1" and view.appbar is None
            assert view.end_drawer is screen.toc.drawer
            assert screen.renderer == "webview" and feature.server is not None and feature.server.running
            url = screen.webview.url
            assert url.startswith(feature.server.base_url)
            # the first page: the WebView is built with its URL (flet-webview loads it in initState), no load_request
            assert [(u, c) for u, c, _ in built] == [(url, url)] and built[0][2] is screen.webview
            assert screen.page_slot.content is screen.webview and _load_requests(conn) == []
            status, _h, body = await asyncio.to_thread(_get, url)
            assert status == 200 and b"window.GLRDR" in body and b"__GLRDR_CFG" in body
            # console bridge: the first event narrows the shell to the console transport
            screen._on_console(types.SimpleNamespace(message="GLRDR:" + json.dumps(
                {"type": "ready", "seq": 1, "page": 0, "count": 4, "paginated": True, "chapter": 1})))
            await asyncio.sleep(0.1)
            assert screen.page_count == 4 and screen.console_ok
            assert "run_javascript" in conn.invoked()
            # the fetch fallback reaches the same handler (server thread -> dispatcher)
            payload = json.dumps({"type": "page", "seq": 2, "page": 3, "count": 4, "chapter": 1}).encode()
            assert (await asyncio.to_thread(_get, feature.server.origin + feature.server.event_path, data=payload,
                                            method="POST", headers={"Content-Type": "application/json"}))[0] == 204
            assert await android_shell.helpers._wait(lambda: screen.page_no == 3, 5)
            assert screen.chrome.page_text.value == "Page 4/4"
            doc, webview = screen.current_doc, screen.webview
            # owner's device report #1: past the last page the next chapter must really load in the WebView
            screen.handle_payload({"type": "edge", "seq": 3, "edge": "end", "chapter": 1})
            assert await android_shell.helpers._wait(lambda: screen.index == 2 and screen.current_doc != doc, 10)
            assert await android_shell.helpers._wait(lambda: len(_load_requests(conn)) == 1, 5)
            (loaded,) = _load_requests(conn)
            assert loaded == screen.webview.url != url and "?v=" in loaded
            assert screen.webview is webview and len(built) == 1  # navigated, not rebuilt
            status, _h, body = await asyncio.to_thread(_get, loaded)
            cfg = json.loads(body.decode("utf-8").split("window.__GLRDR_CFG=", 1)[1].split(";</script>", 1)[0])
            assert status == 200 and cfg["doc"] == screen.current_doc
            screen.chrome.set_visible(True)
            await screen._on_confirm_pop()
            assert not screen.chrome.visible
            screen.dispose()
            assert app.prefs.reader_position("ab12cd34ef56")["href"] == "chapter0003.xhtml"
        finally:
            await android_shell.close()

    asyncio.run(scenario())


@needs_flet
@pytest.mark.skipif(not _has("msgpack"), reason="msgpack not installed")
def test_every_chapter_change_navigates_the_webview(android_shell):
    """Owner's device report #1: flet-webview reads ``url`` only when the control is built, so every
    chapter change (◀ / ▶, slider, Chapters, mode, Aa layout, search, a page edge) must send exactly
    one ``load_request`` with the newly published page to the same WebView; the outgoing page's late
    events are dropped and the new page's are handled."""
    from glossarion_mobile.ui.reader.aa_sheet import SCOPE_ALL

    async def scenario():
        try:
            screen = await android_shell.open("/reader/ab12cd34ef56?ch=1")
            conn, wait = android_shell.conn, android_shell.helpers._wait
            webview = screen.webview
            seen = [webview.url]

            async def navigated(action, index):
                before = len(_load_requests(conn))
                doc = screen.current_doc
                result = action()
                if asyncio.iscoroutine(result):
                    await result
                assert await wait(lambda: len(_load_requests(conn)) > before and screen.current_doc != doc, 10)
                await asyncio.sleep(0.2)  # nothing else follows
                loads = _load_requests(conn)[before:]
                assert len(loads) == 1, loads
                assert loads[0] == screen.webview.url and "?v=" in loads[0] and loads[0] not in seen
                assert screen.webview is webview and screen.page_slot.content is webview
                assert screen.index == index
                seen.append(loads[0])

            await navigated(lambda: screen.chrome.next_button.on_click(None), 2)            # ▶
            # the outgoing chapter's late tap is dropped; the new page's tap toggles the chrome
            screen.chrome.set_visible(True)
            assert screen.handle_payload({"type": "tap", "seq": 40, "zone": "center", "chapter": 1}) is None
            assert screen.chrome.visible
            assert screen.handle_payload({"type": "tap", "seq": 41, "zone": "center", "chapter": 2}) is not None
            assert not screen.chrome.visible
            await navigated(lambda: screen.chrome.prev_button.on_click(None), 1)            # ◀
            screen.chrome.slider.value = 0
            await navigated(lambda: screen.chrome._on_slider(types.SimpleNamespace(control=screen.chrome.slider)), 0)
            await navigated(lambda: screen._on_toc_row(rm.TocRow(2, "Three", 3)), 2)         # Chapters
            await navigated(lambda: screen.set_mode(rm.ORIGINAL), 2)                         # Original
            await navigated(lambda: screen._apply_settings({"layout": "scroll"}, SCOPE_ALL, save=False), 2)
            await navigated(lambda: screen._on_search_pick({"chapter_idx": 0, "text": "의",
                                                            "local_occurrence": 0}), 0)       # search hit
            await navigated(lambda: screen.handle_payload({"type": "edge", "seq": 90, "edge": "end",
                                                           "chapter": 0}), 1)                # page edge
            screen.dispose()
        finally:
            await android_shell.close()

    asyncio.run(scenario())


@needs_flet
@pytest.mark.skipif(not _has("msgpack"), reason="msgpack not installed")
def test_webview_navigation_failure_races_and_dispose(android_shell, monkeypatch):
    """A failed ``load_request`` builds a new WebView with the page URL; two quick chapter changes
    load only the last page; nothing navigates after the Reader is gone."""
    import flet_webview as fwv

    async def scenario():
        try:
            screen = await android_shell.open("/reader/ab12cd34ef56?ch=1")
            conn = android_shell.conn
            old, first_url = screen.webview, screen.webview.url

            async def refused(self, url, method=None):
                raise RuntimeError("WebView must be added to page first.")

            with monkeypatch.context() as patch:
                patch.setattr(fwv.WebView, "load_request", refused)
                await screen.go_chapter(0)
            assert screen.index == 0 and screen.webview is not old
            assert screen.webview.key != old.key  # a per-build key: Flet would skip an equal control
            assert screen.page_slot.content is screen.webview
            assert screen.webview.url.startswith(android_shell.feature.server.base_url) and "?v=" in screen.webview.url
            assert screen.webview.url != first_url and _load_requests(conn) == []
            # two quick changes: the first build is superseded before it publishes
            before = len(_load_requests(conn))
            await asyncio.gather(screen.go_chapter(1), screen.go_chapter(2))
            loads = _load_requests(conn)[before:]
            assert loads == [screen.webview.url] and screen.index == 2
            # Back while the next chapter builds: no navigation for a disposed Reader
            before = len(_load_requests(conn))
            current = screen.webview.url
            task = asyncio.ensure_future(screen.go_chapter(1))
            await asyncio.sleep(0)
            screen.dispose()
            await task
            await asyncio.sleep(0.2)
            assert len(_load_requests(conn)) == before and screen.webview.url == current
        finally:
            await android_shell.close()

    asyncio.run(scenario())


def test_only_navigate_webview_sets_a_webview_url():
    """Guard: a ``url`` property change never navigates a built flet-webview WebView, so the
    Reader and the WebViewBridge assign ``.url`` and call ``load_request`` only inside
    ``navigate_webview`` (the first page goes into ``_make_webview``'s constructor)."""
    import ast

    class Finder(ast.NodeVisitor):
        def __init__(self, name):
            self.name, self.stack, self.hits = name, [], []

        def visit_FunctionDef(self, node):
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        visit_AsyncFunctionDef = visit_FunctionDef

        def _hit(self, node, what):
            self.hits.append((self.name, self.stack[-1] if self.stack else "<module>", what))

        def visit_Assign(self, node):
            for target in node.targets:
                if isinstance(target, ast.Attribute) and target.attr == "url":
                    self._hit(node, "url =")
            self.generic_visit(node)

        def visit_AugAssign(self, node):
            if isinstance(node.target, ast.Attribute) and node.target.attr == "url":
                self._hit(node, "url =")
            self.generic_visit(node)

        visit_AnnAssign = visit_AugAssign

        def visit_Call(self, node):
            if isinstance(node.func, ast.Attribute) and node.func.attr == "load_request":
                self._hit(node, "load_request")
            if isinstance(node.func, ast.Name) and node.func.id == "setattr" and len(node.args) > 1 \
                    and isinstance(node.args[1], ast.Constant) and node.args[1].value == "url":
                self._hit(node, "url =")
            self.generic_visit(node)

    files = sorted((APP_DIR / "glossarion_mobile" / "ui" / "reader").glob("*.py"))
    files.append(APP_DIR / "glossarion_mobile" / "services" / "webview_bridge.py")
    hits = []
    for path in files:
        finder = Finder(path.name)
        finder.visit(ast.parse(path.read_text(encoding="utf-8")))
        hits.extend(finder.hits)
    assert sorted(hits) == [("reader_view.py", "navigate_webview", "load_request"),
                            ("reader_view.py", "navigate_webview", "url =")]


@needs_flet
def test_plan_routes_reader_files_through_plan_for_file(novel, monkeypatch):
    """Open-with / Converter / Files hand the Reader a file path: EPUB and TXT (C6:
    ``intents.READER_EXTENSIONS``) go through ``session.plan_for_file``; other files are refused."""
    from glossarion_mobile.services import intents
    from glossarion_mobile.ui.reader import reader_view
    from glossarion_mobile.ui.reader.reader_view import ReaderScreen

    base = _screen(novel)
    planned = []
    monkeypatch.setattr(reader_view, "plan_for_file", lambda path: planned.append(path) or ("plan", path))
    txt = novel.tmp / "Story.txt"
    txt.write_text("첫 문단\n\n둘째 문단", encoding="utf-8")
    pdf = novel.tmp / "Doc.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    assert ".txt" in intents.READER_EXTENSIONS and ".epub" in intents.READER_EXTENSIONS
    for path in (txt, novel.raw):
        screen = ReaderScreen(base.match, base.deps, args={"path": str(path)})
        assert screen._plan() == ("plan", str(path))
    assert planned == [str(txt), str(novel.raw)]
    for gone in (pdf, novel.tmp / "missing.txt"):
        with pytest.raises(LookupError):
            ReaderScreen(base.match, base.deps, args={"path": str(gone)})._plan()


@needs_flet
def test_reader_opens_a_txt_file_from_a_path(novel):
    """Owner's device report #2 at the Reader: a .txt handed over by Open-with / the Converter / Files
    opens (C6: ``plan_for_file`` reads TXT) and its text is on the page."""
    from glossarion_mobile.ui.reader.reader_view import ReaderScreen

    async def scenario():
        base = _screen(novel)
        story = novel.tmp / "Story.txt"
        story.write_text("첫 문단입니다.\n\n둘째 문단입니다.\n", encoding="utf-8")
        screen = ReaderScreen(base.match, base.deps, args={"path": str(story)})
        await screen.open()
        assert screen.state == "ready" and screen.session.count >= 1 and screen.renderer == "native"
        texts = [t for c in screen.fallback.list_view.controls for t in _all_texts(c)]
        shown = " ".join(str(t.value or "") + "".join(s.text for s in (t.spans or [])) for t in texts)
        assert "첫 문단입니다." in shown and "둘째 문단입니다." in shown
        screen.dispose()

    asyncio.run(scenario())


def _all_texts(control) -> list:
    import flet as ft

    found = [control] if isinstance(control, ft.Text) else []
    for name in ("content", "controls"):
        child = getattr(control, name, None)
        for item in (child if isinstance(child, list) else [child] if child is not None else []):
            if hasattr(item, "__dict__"):
                found.extend(_all_texts(item))
    return found


@needs_flet
def test_fallback_text_is_not_selectable_so_taps_and_long_press_arrive():
    """A SelectableText wins the gesture arena: with it the tap zones and the paragraph long-press
    sheet never fire. Every fallback text is plain, with tap zones on or off."""
    from glossarion_mobile.ui.reader.fallback_view import FallbackPage

    paragraphs = []
    page = FallbackPage(on_tap_zone=lambda z: None, on_pinch=lambda s: None, on_pinch_end=lambda: None,
                        on_paragraph=paragraphs.append, on_scroll=lambda f: None)
    blocks = rb.html_to_blocks("<h1>Title</h1><p>one <i>two</i></p><blockquote>q</blockquote><ul><li>i</li></ul>")
    for zones in (True, False):
        page.render(blocks, theme=DESKTOP_THEMES[0], settings=rm.ReaderSettings(tap_zones=zones))
        texts = [t for c in page.list_view.controls for t in _all_texts(c)]
        assert len(texts) >= 4 and all(t.selectable is False for t in texts)
    page.list_view.controls[1].on_long_press(None)
    assert paragraphs == ["one two"]


@needs_flet
def test_fallback_page_by_reports_the_chapter_edges(monkeypatch):
    import flet as ft

    from glossarion_mobile.ui.reader import fallback_view as fv

    monkeypatch.setattr(fv, "PAGE_SCROLL_MS", 5)
    monkeypatch.setattr(fv, "PAGE_SETTLE", 0.01)
    page = fv.FallbackPage(on_tap_zone=lambda z: None, on_pinch=lambda s: None, on_pinch_end=lambda: None,
                           on_paragraph=lambda t: None, on_scroll=lambda f: None)
    page.set_size(400, 800)
    page.render(rb.html_to_blocks("<p>a</p>"), theme=DESKTOP_THEMES[0], settings=rm.ReaderSettings())
    list_state = {"extent": 0.0, "pixels": 0.0, "fail": False, "calls": []}

    async def scroll_to(self, offset=None, delta=None, scroll_key=None, duration=0, curve=None):
        list_state["calls"].append((offset, delta))
        if list_state["fail"]:
            raise RuntimeError("ListView Control must be added to the page first")
        extent = list_state["extent"]
        if not extent:
            return  # shorter than the screen: Flutter cannot scroll it, no notification
        target = list_state["pixels"] + delta if delta is not None else (extent + offset + 1 if offset < 0 else offset)
        list_state["pixels"] = max(0.0, min(extent, target))  # clamped: at a bound it only overscrolls
        page._on_scroll(types.SimpleNamespace(pixels=list_state["pixels"], min_scroll_extent=0.0,
                                              max_scroll_extent=extent))

    monkeypatch.setattr(ft.ListView, "scroll_to", scroll_to)

    async def scenario():
        assert page.at_edge(1) is None and page.at_edge(-1) is None   # nothing known after a render
        assert await page.page_by(1) == fv.PAGE_EDGE                   # short chapter: never moves
        assert await page.page_by(-1) == fv.PAGE_EDGE
        list_state["extent"] = 3000.0
        page.render(rb.html_to_blocks("<p>long</p>"), theme=DESKTOP_THEMES[0], settings=rm.ReaderSettings())
        assert await page.page_by(1) == fv.PAGE_SCROLLED and list_state["pixels"] == 720.0
        list_state["pixels"] = 3000.0
        page._on_scroll(types.SimpleNamespace(pixels=3000.0, min_scroll_extent=0.0, max_scroll_extent=3000.0))
        calls = len(list_state["calls"])
        assert page.at_edge(1) is True and await page.page_by(1) == fv.PAGE_EDGE
        assert len(list_state["calls"]) == calls                      # known end: no scroll at all
        page.max_extent = 4000.0                                      # stale extent at the real end:
        assert page.at_edge(1) is False                               # the scroll only overscrolls,
        assert await page.page_by(1) == fv.PAGE_EDGE                   # the list stays put
        assert len(list_state["calls"]) == calls + 1
        assert await page.page_by(-1) == fv.PAGE_SCROLLED and list_state["pixels"] == 2280.0
        # the same chapter drawn again (an Aa change) at its end: the list kept its offset, so the
        # first right tap turns (it only overscrolls); the extent is unknown until the next event
        list_state["pixels"] = 3000.0
        page._on_scroll(types.SimpleNamespace(pixels=3000.0, min_scroll_extent=0.0, max_scroll_extent=3000.0))
        page.render(rb.html_to_blocks("<p>long</p>"), theme=DESKTOP_THEMES[0], settings=rm.ReaderSettings(),
                    keep_offset=True)
        assert page.pixels == 3000.0 and page.at_edge(1) is None
        assert await page.page_by(1) == fv.PAGE_EDGE
        # known metrics in the middle of a chapter but the scroll events are late (a busy phone):
        # never a chapter turn that would skip the rest of the chapter
        list_state["pixels"] = 1000.0
        page._on_scroll(types.SimpleNamespace(pixels=1000.0, min_scroll_extent=0.0, max_scroll_extent=3000.0))
        real_on_scroll = page._on_scroll
        page._on_scroll = lambda e: None  # the events of this turn arrive after the settle time
        assert await page.page_by(1) == fv.PAGE_SCROLLED
        page._on_scroll = real_on_scroll
        list_state["fail"] = True
        assert await page.page_by(1) == fv.PAGE_SCROLLED               # a failed scroll never turns
        list_state.update(fail=False, extent=0.0)
        page.render(rb.html_to_blocks("<p>short</p>"), theme=DESKTOP_THEMES[0], settings=rm.ReaderSettings())
        later = asyncio.get_running_loop().call_later(
            0.001, lambda: page.render(rb.html_to_blocks("<p>next</p>"), theme=DESKTOP_THEMES[0],
                                       settings=rm.ReaderSettings()))
        assert await page.page_by(1) == fv.PAGE_SCROLLED               # another chapter was drawn meanwhile
        later.cancel()

    asyncio.run(scenario())


@needs_flet
def test_native_reader_edge_taps_turn_chapters(novel, monkeypatch):
    """Owner's device report #1, the Lightweight reader / WebView-failure path: an edge tap scrolls a
    screen and at the end (start) of the chapter opens the next (previous: its last page) chapter;
    a short chapter turns on the first tap, a long one scrolls first; the centre toggles the chrome."""
    import flet as ft

    from glossarion_mobile.ui.reader import fallback_view as fv

    monkeypatch.setattr(fv, "PAGE_SCROLL_MS", 5)
    monkeypatch.setattr(fv, "PAGE_SETTLE", 0.01)
    list_state = {"extent": 0.0, "pixels": 0.0, "calls": []}
    holder = {}

    async def scroll_to(self, offset=None, delta=None, scroll_key=None, duration=0, curve=None):
        list_state["calls"].append((offset, delta))
        extent = list_state["extent"]
        if not extent:
            return
        target = list_state["pixels"] + delta if delta is not None else (extent + offset + 1 if offset < 0 else offset)
        list_state["pixels"] = max(0.0, min(extent, target))
        holder["screen"].fallback._on_scroll(types.SimpleNamespace(pixels=list_state["pixels"], min_scroll_extent=0.0,
                                                                   max_scroll_extent=extent))

    monkeypatch.setattr(ft.ListView, "scroll_to", scroll_to)

    async def settle(predicate):
        for _ in range(100):
            if predicate():
                return True
            await asyncio.sleep(0.02)
        return predicate()

    def shown(index):  # that chapter is drawn (render finished)
        screen = holder["screen"]
        return lambda: screen.index == index and screen.fallback_generation == screen.render_generation

    async def scenario():
        screen = _screen(novel, route="/reader/ab12cd34ef56?ch=0")
        holder["screen"] = screen
        await screen.open()
        assert screen.renderer == "native" and screen.index == 0 and screen.settings.tap_zones
        screen._on_fallback_tap("next")                     # a short chapter: the first tap turns it
        assert await settle(shown(1))
        assert (None, 0.9 * 860) in list_state["calls"]
        # a second quick tap while the next chapter is still being drawn never skips that chapter
        task = asyncio.ensure_future(screen.go_chapter(2))
        await asyncio.sleep(0)
        assert screen.index == 2 and screen.fallback_generation != screen.render_generation
        calls = len(list_state["calls"])
        await screen._fallback_page(1)
        assert len(list_state["calls"]) == calls            # ignored: no scroll, no turn
        await task
        assert await settle(shown(2))
        await screen.go_chapter(1)
        assert await settle(shown(1))
        list_state["extent"] = 5000.0                       # a long chapter: the tap scrolls a screen
        screen._on_fallback_tap("next")
        await asyncio.sleep(0.2)
        assert screen.index == 1 and list_state["pixels"] == pytest.approx(774.0)
        list_state["pixels"] = 5000.0                       # the reader reached its end
        screen.fallback._on_scroll(types.SimpleNamespace(pixels=5000.0, min_scroll_extent=0.0,
                                                         max_scroll_extent=5000.0))
        screen._on_fallback_tap("next")
        assert await settle(shown(2))
        assert list_state["calls"][-1] == (0, None)         # another chapter opens at its top
        screen._on_fallback_tap("prev")                     # at its top: the previous chapter's end
        assert await settle(shown(1))
        assert screen.last_page and list_state["calls"][-1] == (-1, None)
        screen.chrome.set_visible(True)
        screen._on_fallback_tap("centre")
        assert not screen.chrome.visible
        screen.settings = screen.settings.with_changes(tap_zones=False)
        calls = len(list_state["calls"])
        screen._on_fallback_tap("next")                     # tap zones off: every tap is the chrome
        assert screen.chrome.visible and screen.index == 1 and len(list_state["calls"]) == calls
        screen.dispose()

    asyncio.run(scenario())


# =====================================================================================
# U5 review fixes: book content cannot run code, resume, one-shot open args, wakelock,
# overlay polling only while a job runs
# =====================================================================================


def test_sanitize_book_html_and_inert_css():
    from glossarion_mobile.ui.reader.document import inert_css, sanitize_book_html, stamp_script_nonce

    plain = "<html><head><meta http-equiv='Content-Type' content='text/html'/></head><body><p>a</p></body></html>"
    assert sanitize_book_html(plain) == plain  # untouched, byte for byte
    evil = ('<p onclick="x()">a<script>steal()</script><img src="a.png" onerror="y()"/>'
            '<a href=" JaVa\tScript:alert(1)">j</a><a href="https://example.org/">ok</a>'
            '<iframe src="/x"></iframe><base href="https://evil/"/><object data="x"></object>'
            '<meta http-equiv="refresh" content="0;url=https://evil/"/><svg><script>z()</script></svg></p>')
    clean = sanitize_book_html(evil)
    for gone in ("<script", "steal", "onclick", "onerror", "avaScript", "<iframe", "<base", "<object", "refresh"):
        assert gone not in clean, gone
    assert 'href="https://example.org/"' in clean and 'src="a.png"' in clean and ">j</a>" in clean
    assert "script" not in sanitize_book_html('<a href="java	script:x()">l</a>')  # tab inside the scheme
    assert inert_css("p{}</style><script>x()</script>") == "p{}\\3C /style>\\3C script>x()\\3C /script>"
    assert inert_css("p{color:red}") == "p{color:red}"
    assert stamp_script_nonce("<script>a</script><SCRIPT type=x>b</script><scripts>", "N") == \
        '<script nonce="N">a</script><script nonce="N" type=x>b</script><scripts>'


def test_style_raw_text_in_svg_and_math_never_becomes_a_nonced_script(novel, server, monkeypatch):
    """U5 verification: html.parser keeps <style> contents as raw text, but a browser parses markup
    inside an SVG / MathML <style>; such content must never reach the page as a nonce'd script."""
    from glossarion_mobile.ui.reader.document import (DocumentBuilder, defang_book_scripts,
                                                      sanitize_book_html)

    samples = ['<p>a</p><svg><style><script>one()</script></style></svg>',
               '<math><style><script>two()</script></style></math>',
               '<svg><style><img src="x" onerror="three()"></style></svg>',
               '<svg><style>p{color:red}</style><rect width="1"/></svg>']
    for sample in samples:
        clean = sanitize_book_html(sample)
        assert "<script" not in clean.lower() and "<img" not in clean.lower(), clean
    # CSS with no markup keeps its bytes (the fast path)
    assert sanitize_book_html(samples[3]) == samples[3]
    assert sanitize_book_html("<style>a&b{}</style><p>x</p>") == "<style>a&b{}</style><p>x</p>"
    # style text keeps "&" verbatim when it has to be rewritten
    assert "a&b" in sanitize_book_html("<svg><style>a&b{}<i></i></style></svg>")
    assert defang_book_scripts("<p>x</p><style></script><SCRIPT>y()</script></style>") == \
        "<p>x</p><style>&lt;/script>&lt;SCRIPT>y()&lt;/script></style>"
    assert defang_book_scripts("<p>scripts</p>") == "<p>scripts</p>"

    session = _overlay_session(novel)
    crafted = ('<html><body><p>Hello</p><svg><style><script>styleHidden()</script></style></svg>'
               '<math><style><script>mathHidden()</script></style></math></body></html>')
    monkeypatch.setattr(session, "chapter_html", lambda index, flavor=None: crafted)
    builder = DocumentBuilder(session, server.register_image)
    for layout in (rm.LAYOUT_SINGLE, rm.LAYOUT_ALL):
        built = builder.build(0, settings=rm.ReaderSettings(), layout=layout, theme=DESKTOP_THEMES[0],
                              doc_id="d9", event_url=server.event_path)
        nonced = re.findall(r'<script nonce="[^"]+">(.*?)</script>', built.html, re.S)
        assert built.nonce and nonced
        assert not any("styleHidden" in body or "mathHidden" in body for body in nonced)
        assert "<script>styleHidden" not in built.html and "<script>mathHidden" not in built.html
        assert built.html.count("<script") == built.html.count(f'<script nonce="{built.nonce}">')


def test_book_scripts_css_breakout_and_private_files_never_reach_the_page(novel, server, monkeypatch):
    """U5 review: a crafted EPUB cannot run script in the reader page, break out of its CSS, or
    have an app-private file served through <img src> traversal."""
    import reader_doc
    from glossarion_mobile.ui.reader.document import DocumentBuilder

    secret = novel.tmp / "private" / "authgpt_tokens.json"
    secret.parent.mkdir()
    secret.write_text('{"access_token": "sk-secret"}', encoding="utf-8")
    monkeypatch.setattr(reader_doc.ReaderDocument, "_get_embedded_css",
                        lambda self: "p{color:red}</style><script>cssBreakout()</script><style>")
    session = _overlay_session(novel)
    crafted = (f'<html><body><p>Hello</p><img src="{secret}"/>'
               '<script>fetch("/x")</script><p onmouseover="steal()">x</p></body></html>')
    monkeypatch.setattr(session, "chapter_html", lambda index, flavor=None: crafted)
    builder = DocumentBuilder(session, server.register_image)
    built = builder.build(0, settings=rm.ReaderSettings(font_family="Embedded CSS"), layout=rm.LAYOUT_SINGLE,
                          theme=DESKTOP_THEMES[0], doc_id="d1", event_url=server.event_path)
    html = built.html
    assert "fetch(\"/x\")" not in html and "steal()" not in html and "cssBreakout()</script>" not in html
    assert "\\3C /style>" in html  # the book CSS stayed inside its <style>
    assert built.nonce and html.count("<script") == html.count(f'<script nonce="{built.nonce}">') >= 2
    status, headers, _body = _get(server.publish(html, script_nonce=built.nonce))
    assert status == 200 and f"'nonce-{built.nonce}'" in headers["Content-Security-Policy"]
    # the traversal still materialises the file (desktop-shared resolver), but it is never served
    images = re.findall(r'src="(/[^"]+/img/[^"]+)"', html)
    assert images and all(_get(server.origin + url)[0] == 404 for url in images)
    builder.close()
    session.close()


@needs_flet
def test_external_book_links_open_only_after_confirmation(novel):
    async def scenario():
        screen = _screen(novel)
        await screen.open()
        opened, shown = [], []
        screen.deps.open_url = opened.append
        screen.page.show_dialog = shown.append
        screen._on_link("javascript:alert(1)", True)
        await asyncio.sleep(0.05)
        assert opened == [] and shown == [] and screen.notes[-1][0] == "This link cannot be opened"
        screen._on_link("https://example.org/a", True)
        await asyncio.sleep(0.05)
        assert opened == [] and shown and shown[-1].title.value == "Open link?"
        assert shown[-1].content.content.controls[0].value == "https://example.org/a"
        await shown[-1].actions[-1].on_click(None)  # Yes
        await asyncio.sleep(0.05)
        assert opened == ["https://example.org/a"]
        screen.dispose()

    asyncio.run(scenario())


@needs_flet
def test_continue_opens_at_the_saved_position_and_a_plain_open_keeps_it(novel, monkeypatch):
    """U5 review: ▶ Continue opens at the saved position; a plain open offers it and does not
    overwrite it with the untouched start (1 s debounce, dispose) until the reader moves."""
    from glossarion_mobile.ui.reader import reader_view
    from glossarion_mobile.ui.reader.reader_view import ReaderScreen

    monkeypatch.setattr(reader_view, "POSITION_SAVE_DELAY", 0.05)

    async def scenario():
        base = _screen(novel)
        prefs = base.deps.prefs
        prefs.set_reader_position("ab12cd34ef56", "chapter0002.xhtml", 0.5, page=2, mode="translated", chapter=1,
                                  pages=5)
        resumed = ReaderScreen(base.match, base.deps, args={"book": dict(novel.book), "resume": True})
        await resumed.open()
        assert resumed.index == 1 and resumed.fraction == 0.5 and resumed.resume_offer is None
        resumed.dispose()
        prefs.set_reader_position("ab12cd34ef56", "chapter0002.xhtml", 0.5, page=2, mode="translated", chapter=1,
                                  pages=5)
        plain = ReaderScreen(base.match, base.deps, args={"book": dict(novel.book)})
        await plain.open()
        assert plain.index == 0 and plain.resume_offer is not None and plain.resume_offer.chapter == 1
        await asyncio.sleep(0.2)  # the debounced save of the start fired
        assert prefs.reader_position("ab12cd34ef56")["href"] == "chapter0002.xhtml"
        plain.handle_payload({"type": "ready", "seq": 1, "page": 0, "count": 3, "chapter": 0})
        await asyncio.sleep(0.2)
        assert prefs.reader_position("ab12cd34ef56")["href"] == "chapter0002.xhtml"
        await plain.go_chapter(2)  # the reader moved: from now on its position is saved
        await asyncio.sleep(0.2)
        assert prefs.reader_position("ab12cd34ef56")["href"] == "chapter0003.xhtml"
        plain.dispose()
        # dispose with the offer still pending keeps the saved position too
        prefs.set_reader_position("ab12cd34ef56", "chapter0002.xhtml", 0.5, page=2, mode="translated", chapter=1)
        untouched = ReaderScreen(base.match, base.deps, args={"book": dict(novel.book)})
        await untouched.open()
        untouched.dispose()
        assert prefs.reader_position("ab12cd34ef56")["href"] == "chapter0002.xhtml"

    asyncio.run(scenario())


@needs_flet
def test_reader_feature_open_args_are_one_shot():
    from glossarion_mobile.ui.reader.feature import ReaderFeature
    from glossarion_mobile.ui.router import parse_route

    navigated = []
    app = types.SimpleNamespace(page=None, navigate_to=lambda *a: navigated.append(a), prefs=None,
                                library=types.SimpleNamespace(book_for_bid=lambda bid: None,
                                                              bid_for=lambda b: "ab12cd34ef56"))
    feature = ReaderFeature(app)
    built = []
    feature._deps = lambda: types.SimpleNamespace()
    import glossarion_mobile.ui.reader.reader_view as reader_view

    class Probe:
        def __init__(self, match, deps, *, args=None):
            built.append(dict(args or {}))

    original = reader_view.ReaderScreen
    reader_view.ReaderScreen = Probe
    try:
        book = {"name": "Novel", "path": "/x/Novel.epub"}
        assert feature.open_book(book, chapter_filename="chapter0003.xhtml", raw_only=True, resume=True) == \
            "ab12cd34ef56"
        match = parse_route("/reader/ab12cd34ef56")
        feature.make_screen(match)
        feature.make_screen(parse_route("/reader/ab12cd34ef56?ch=12"))  # a later deep link / notification
    finally:
        reader_view.ReaderScreen = original
    assert built[0] == {"book": book, "raw_only": True, "chapter_filename": "chapter0003.xhtml", "resume": True}
    assert built[1] == {"book": book}  # the id still resolves; the open request's arguments are gone
    assert feature.resolve("ab12cd34ef56") == {"book": book}


def test_shared_wakelock_is_reference_counted():
    from glossarion_mobile.services.wakelock import SharedWakelock

    class Lock:
        def __init__(self):
            self.calls = []

        async def enable(self):
            self.calls.append("enable")

        async def disable(self):
            self.calls.append("disable")

    async def scenario():
        raw = Lock()
        owner = SharedWakelock(raw)
        jobs, reader = owner.holder("jobs"), owner.holder("reader")
        await jobs.enable()
        await reader.enable()
        await reader.disable()  # the Reader closes while a job keeps the screen on
        assert raw.calls == ["enable"] and owner.on and jobs.held and not reader.held
        await reader.enable()
        await jobs.disable()  # the job ends while the Reader is open
        assert raw.calls == ["enable"] and owner.on
        await reader.disable()
        await reader.disable()
        assert raw.calls == ["enable", "disable"] and not owner.on

    asyncio.run(scenario())


@needs_flet
def test_overlay_polls_only_while_a_job_for_the_book_runs(novel):
    """U5 review: an in-progress book is re-read every 3 s only while one of its jobs runs; the
    job's end triggers one last refresh."""
    async def scenario():
        screen = _screen(novel)
        listeners = []
        jobs = screen.deps.jobs
        jobs.subscribe = lambda callback: listeners.append(callback) or (lambda: listeners.remove(callback))
        jobs.view = lambda: types.SimpleNamespace(active=None, queue=())
        refreshed = []
        await screen.open()
        assert screen.session.plan.mode == "overlay" and listeners and screen._poll_task is None
        screen.refresh_overlay = lambda: refreshed.append(1) or asyncio.sleep(0)
        other = types.SimpleNamespace(spec=types.SimpleNamespace(origin={"bid": "ffffffffffff"}, inputs=()),
                                      output_dir="", output_dirs={})
        listeners[0](types.SimpleNamespace(active=other, queue=()))
        assert screen._poll_task is None  # someone else's job
        mine = types.SimpleNamespace(spec=types.SimpleNamespace(origin={}, inputs=(str(novel.raw),)),
                                     output_dir=str(novel.workspace), output_dirs={})
        listeners[0](types.SimpleNamespace(active=mine, queue=()))
        assert screen._poll_task is not None
        listeners[0](types.SimpleNamespace(active=None, queue=()))
        await asyncio.sleep(0.05)
        assert screen._poll_task is None and refreshed == [1]
        screen.dispose()
        assert listeners == []

    asyncio.run(scenario())


# =====================================================================================
# U5 second review round: wakelock after an early back, a failed platform enable, queued
# jobs, oversized event bodies
# =====================================================================================


@needs_flet
def test_leaving_during_the_first_render_holds_no_wakelock(novel):
    """Back while the first chapter is still rendering: open() must not acquire the shared
    wakelock (nothing would release it), offer "Resume" or watch jobs for the disposed screen."""
    import dataclasses

    from glossarion_mobile.services.wakelock import SharedWakelock

    class Lock:
        def __init__(self):
            self.calls = []

        async def enable(self):
            self.calls.append("enable")

        async def disable(self):
            self.calls.append("disable")

    async def scenario():
        raw = Lock()
        owner = SharedWakelock(raw)
        screen = _screen(novel)
        listeners = []
        jobs = screen.deps.jobs
        jobs.subscribe = lambda callback: listeners.append(callback) or (lambda: listeners.remove(callback))
        jobs.view = lambda: types.SimpleNamespace(active=None, queue=())
        screen.deps.wakelock = owner.holder("reader")
        screen.settings = dataclasses.replace(screen.settings, keep_screen_on=True)
        real_render = screen.render

        async def slow_render(index, **kwargs):
            await asyncio.sleep(0.2)  # the first chapter builds on the io pool
            await real_render(index, **kwargs)

        screen.render = slow_render
        task = asyncio.ensure_future(screen.open())
        for _ in range(400):
            if screen.state == "ready":
                break
            await asyncio.sleep(0.005)
        assert screen.state == "ready" and screen.loading.visible
        screen.dispose()  # Android back while "Loading…"
        await task
        await asyncio.sleep(0.05)
        assert owner.holders == set() and not owner.on and raw.calls == []
        assert listeners == [] and screen._jobs_unsub is None and screen.resume_offer is None

    asyncio.run(scenario())


def test_shared_wakelock_counts_a_holder_only_after_the_platform_call():
    """A failed platform enable must not leave the holder counted: BackgroundExecution sees the
    failure and never releases, so a stale "jobs" holder kept the screen on after a Reader with
    "Keep screen on" closed."""
    from glossarion_mobile.services.wakelock import SharedWakelock

    class FlakyLock:
        def __init__(self):
            self.calls = []
            self.fail_next = True

        async def enable(self):
            if self.fail_next:
                self.fail_next = False
                self.calls.append("enable-failed")
                raise TimeoutError("invoke_method timed out")
            self.calls.append("enable")

        async def disable(self):
            self.calls.append("disable")

    async def scenario():
        raw = FlakyLock()
        owner = SharedWakelock(raw)
        jobs, reader = owner.holder("jobs"), owner.holder("reader")
        with pytest.raises(TimeoutError):
            await jobs.enable()
        assert not jobs.held and not owner.on and owner.holders == set()
        await reader.enable()  # a Reader with "Keep screen on" opens ...
        await reader.disable()  # ... and closes
        assert raw.calls == ["enable-failed", "enable", "disable"] and not owner.on and owner.holders == set()

    asyncio.run(scenario())


@needs_flet
def test_overlay_is_not_polled_for_a_queued_job(novel):
    """UI_SPEC §3.11: the overlay is refreshed only while a job for this book runs; while it waits
    in the queue behind another book's job nothing writes its chapters."""
    async def scenario():
        screen = _screen(novel)
        listeners = []
        jobs = screen.deps.jobs
        jobs.subscribe = lambda callback: listeners.append(callback) or (lambda: listeners.remove(callback))
        jobs.view = lambda: types.SimpleNamespace(active=None, queue=())
        await screen.open()
        other = types.SimpleNamespace(spec=types.SimpleNamespace(origin={"bid": "ffffffffffff"}, inputs=()),
                                      output_dir="", output_dirs={})
        mine = types.SimpleNamespace(spec=types.SimpleNamespace(origin={}, inputs=(str(novel.raw),)),
                                     output_dir=str(novel.workspace), output_dirs={})
        listeners[0](types.SimpleNamespace(active=other, queue=(mine,)))
        assert screen._poll_task is None
        listeners[0](types.SimpleNamespace(active=mine, queue=()))  # its turn: polled
        assert screen._poll_task is not None
        screen.dispose()

    asyncio.run(scenario())


def test_server_answers_oversized_events_with_a_readable_413(server):
    """An oversized event body is read and dropped before the 413: closing the socket with request
    bytes unread sends a TCP reset, and on Windows the client then got ConnectionResetError /
    ConnectionAbortedError instead of the 413 (about 5% of requests)."""
    as_json = {"Content-Type": "application/json"}
    for _ in range(40):
        status, _headers, body = _get(server.origin + server.event_path, data=b"x" * (70 * 1024), method="POST",
                                      headers=as_json)
        assert status == 413 and body == b"too large"
    assert server.events == []
