"""WebViewBridge (services/webview_bridge.py, ui/components/hidden_webview.py; U9).

* ``page_script`` wraps a route's JavaScript expression so the page logs ``GLWVB:`` answers
  (run under node when it is installed: values, errors, ``undefined``, trailing ``;`` / ``//``
  comments, chunked answers), and wraps every real route script into valid JavaScript;
* the answers come back through the Reader's console parser (``parse_console_message`` with
  the bridge's prefix) and are reassembled from chunks;
* end to end with a real asyncio loop thread, the real ``UiDispatcher`` and a fake WebView:
  the AuthND token flow and Gemini Free requests of the real backend modules run on worker
  threads (``GLOSSARION_MOBILE=1``, no helper processes), the hidden pages mount and unmount
  in the host, ``cancel_stream`` ends the waits at once, a JavaScript timeout gives None (late
  answers are dropped), the loop thread is refused and at most ``MAX_PAGES`` pages are open;
* the hidden host keeps each WebView laid out at its viewport size, clipped to one pixel,
  nearly transparent, ignoring input and outside the accessibility tree;
* Accounts › Experimental: tappable rows while the bridge is registered, disabled rows with a
  ReasonChip without flet-webview; ``WebViewBridgeFeature.install`` registers only where
  flet-webview runs.

Real data stays isolated (HOME / USERPROFILE point at tmp_path). Run from src/mobile:
    python -m pytest -p no:cacheprovider -W ignore tests_host/test_webview_bridge.py
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = MOBILE_DIR / "app"
SRC_DIR = MOBILE_DIR.parent
for entry in (str(APP_DIR), str(SRC_DIR)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import authnd_auth  # noqa: E402
import browser_driver  # noqa: E402
import gemini_free  # noqa: E402
from glossarion_mobile.services import webview_bridge as wb  # noqa: E402
from glossarion_mobile.services.dispatcher import UiDispatcher  # noqa: E402
from glossarion_mobile.state.store import LoopGuard  # noqa: E402
from glossarion_mobile.ui.reader.bridge import parse_console_message  # noqa: E402

NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="node is not installed")
needs_flet = pytest.mark.skipif(importlib.util.find_spec("flet") is None, reason="flet not installed")
PAGE_URL = "https://build.nvidia.com/z-ai/glm-5.1"
BLOCK = object()  # a site answer that never comes


# ============================================================================ page_script
def _run_node(source: str) -> list:
    """Run ``source`` under node with a console that records ``GLWVB:`` lines."""
    harness = ("const lines = [];\n"
               "globalThis.console = {log: (s) => lines.push(String(s))};\n"
               "const result = eval(" + json.dumps(source) + ");\n"
               "process.stdout.write(JSON.stringify({lines, result}));\n")
    proc = subprocess.run([NODE, "-e", harness], capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout)
    assert out["result"] is True  # the source ends with ``true`` (a WKWebView-safe result)
    return out["lines"]


@needs_node
def test_page_script_answers_through_the_console():
    lines = _run_node(wb.page_script("r1", "({a: 1, b: [2, 'x']})"))
    assert [parse_console_message(line, prefix=wb.CONSOLE_PREFIX) for line in lines] == [
        {"id": "r1", "ok": True, "value": {"a": 1, "b": [2, "x"]}}]
    assert parse_console_message(lines[0]) is None  # not a Reader (GLRDR:) line
    for expression, value in (("JSON.stringify({x: 1});", '{"x":1}'),       # trailing ;
                              ("1 + 1 // a comment", 2),                     # trailing comment
                              ("undefined", None),
                              ("(() => 'marker')();\n", "marker")):
        (line,) = _run_node(wb.page_script("r2", expression))
        assert parse_console_message(line, prefix=wb.CONSOLE_PREFIX) == {"id": "r2", "ok": True, "value": value}
    (line,) = _run_node(wb.page_script("r3", "(() => { throw new Error('nope'); })()"))
    assert parse_console_message(line, prefix=wb.CONSOLE_PREFIX) == {"id": "r3", "ok": False, "error": "nope"}


@needs_node
def test_page_script_chunks_long_answers():
    lines = _run_node(wb.page_script("big", "'x'.repeat(1000) + 'é'", chunk_chars=256))
    payloads = [parse_console_message(line, prefix=wb.CONSOLE_PREFIX) for line in lines]
    assert len(payloads) > 1 and all(p["id"] == "big" and p["parts"] == len(payloads) for p in payloads)
    page = wb.WebViewPage(types.SimpleNamespace(), "p", "authnd", (10, 10))
    future: concurrent.futures.Future = concurrent.futures.Future()
    page._pending["big"] = future
    for payload in reversed(payloads):  # order does not matter
        page.deliver(payload)
    assert future.result(0) == {"id": "big", "ok": True, "value": "x" * 1000 + "é"}


@needs_node
def test_page_script_wraps_every_route_script(tmp_path):
    scripts = [
        authnd_auth._captcha_injection_script("m:1:2", authnd_auth.DEFAULT_HCAPTCHA_SITEKEY, "__cb", 30000),
        authnd_auth.CAPTCHA_STATE_SCRIPT,
        gemini_free._ai_mode_set_prompt_script("User:\nhello \"quoted\" `tick` ${x}"),
        gemini_free.AI_MODE_CLICK_SEND_SCRIPT,
        gemini_free._page_snapshot_script("User:\nhello"),
    ]
    for index, script in enumerate(scripts):
        path = tmp_path / f"script{index}.js"
        path.write_text(wb.page_script(f"s{index}", script), encoding="utf-8")
        proc = subprocess.run([NODE, "--check", str(path)], capture_output=True, text=True, timeout=60)
        assert proc.returncode == 0, (index, proc.stderr)


# ============================================================================ end to end
class FakeHost:
    def __init__(self):
        self.entries = []
        self.added = 0

    def add(self, control, *, size, key):
        entry = types.SimpleNamespace(control=control, size=size, key=key)
        self.entries.append(entry)
        self.added += 1
        return entry

    def remove(self, entry):
        if entry in self.entries:
            self.entries.remove(entry)
            return True
        return False


class Site:
    """A website model answering the routes' scripts (Python stand-in for the page)."""

    title = "Site"

    def answer(self, expression):
        return None


class NvidiaSite(Site):
    title = "NVIDIA NIM"

    def __init__(self, block=False):
        self.block = block
        self.marker = ""

    def answer(self, expression):
        if self.block:
            return BLOCK
        if "const marker =" in expression:
            self.marker = json.loads(re.search(r"const marker = (\".*?\");", expression).group(1))
            return self.marker
        if "result: window.__authndResult" in expression:
            return json.dumps({"marker": self.marker, "readyState": "complete", "result": {
                "marker": self.marker, "pending": False, "step": "complete", "token": "wv-token", "error": None}})
        return None


class AiModeSite(Site):
    title = "Google Search"

    def __init__(self, answer_text="Bonjour", big=False):
        self.answer_text = "Bonjour " * 4000 if big else answer_text

    def answer(self, expression):
        if "AI Mode textarea not found" in expression:
            return json.dumps({"ok": True, "textareaCount": 1, "valueLength": 3})
        if "AI Mode send button not ready" in expression:
            return json.dumps({"ok": True, "label": "Send", "buttonCount": 1})
        if "answerMarkdown" in expression:
            prompt = json.loads(re.search(r"const promptText = (\".*?\");", expression).group(1))
            return json.dumps({"url": "https://www.google.com/search?udm=50", "title": "AI Mode", "busy": False,
                               "text": f"You said:\n{prompt}\n{self.answer_text}\nAI can make mistakes"})
        return None


class FakeWebView:
    """Stands in for ``flet_webview.WebView``: navigations fire the page events and
    ``run_javascript`` answers through ``on_console`` (asynchronously, on the loop)."""

    def __init__(self, page, url, site, *, chunk_chars=wb.CHUNK_CHARS):
        self.page, self.site, self.chunk_chars = page, site, chunk_chars
        self.sources, self.loads = [], []
        self.title = ""
        self._navigate(url)

    def _navigate(self, url):
        self.loads.append(url)
        loop = asyncio.get_running_loop()
        loop.call_soon(self.page.on_page_started, types.SimpleNamespace(data=url))

        def ended():
            self.title = self.site.title
            self.page.on_page_ended(types.SimpleNamespace(data=url))

        loop.call_later(0.01, ended)

    async def load_request(self, url):
        self._navigate(url)

    async def get_title(self):
        return self.title

    async def run_javascript(self, source):
        self.sources.append(source)
        request_id = json.loads(re.search(r"var id = (\"[0-9a-f]+\")", source).group(1))
        expression = source.split("var value = (\n", 1)[1].rsplit("\n    );\n    envelope", 1)[0]
        value = self.site.answer(expression)
        if value is BLOCK:
            return
        text = json.dumps({"id": request_id, "ok": True, "value": value})
        if len(text) <= self.chunk_chars:
            lines = [text]
        else:
            parts = -(-len(text) // self.chunk_chars)
            lines = [json.dumps({"id": request_id, "part": i, "parts": parts,
                                 "chunk": text[i * self.chunk_chars:(i + 1) * self.chunk_chars]})
                     for i in range(parts)]
        loop = asyncio.get_running_loop()
        for line in lines:
            loop.call_soon(self.page.on_console, types.SimpleNamespace(message="app.js:1 " + wb.CONSOLE_PREFIX + line))


@pytest.fixture
def ui_loop():
    """A running asyncio loop on its own thread with a bound UiDispatcher (the Flet loop)."""
    loop = asyncio.new_event_loop()
    ready = threading.Event()
    dispatcher = UiDispatcher(None, guard=LoopGuard())

    def run():
        asyncio.set_event_loop(loop)
        dispatcher.bind(loop)
        ready.set()
        loop.run_forever()

    thread = threading.Thread(target=run, name="fake-flet-loop", daemon=True)
    thread.start()
    assert ready.wait(5)
    yield types.SimpleNamespace(loop=loop, dispatcher=dispatcher)
    loop.call_soon_threadsafe(loop.stop)
    thread.join(5)
    loop.close()


@pytest.fixture
def mobile_env(monkeypatch, tmp_path):
    for name in ("AUTHND_TOKEN_MODE", "GEMINI_FREE_MODE", "TRANSLATION_CANCELLED"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GLOSSARION_MOBILE", "1")
    monkeypatch.setenv("GLOSSARION_NO_PROCESSES", "1")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("GEMINI_FREE_WAIT_AFTER_LOAD_MS", "0")
    monkeypatch.setenv("GEMINI_FREE_STABLE_MS", "0")
    monkeypatch.setattr(authnd_auth, "CAPTCHA_DOCUMENT_STABLE_SECONDS", 0.05)

    def no_popen(*args, **kwargs):
        raise AssertionError("a helper subprocess was started on mobile")

    monkeypatch.setattr(subprocess, "Popen", no_popen)
    browser_driver.unregister_driver()
    authnd_auth.reset_cancel()
    gemini_free.reset_cancel()
    yield monkeypatch
    browser_driver.unregister_driver()
    wb._set_current(None)
    authnd_auth.reset_cancel()
    gemini_free.reset_cancel()


def _bridge(ui_loop, site_factory, **kwargs):
    host = FakeHost()
    views = []
    chunk_chars = kwargs.pop("chunk_chars", wb.CHUNK_CHARS)

    def factory(page, url):
        view = FakeWebView(page, url, site_factory(), chunk_chars=chunk_chars)
        views.append(view)
        return view

    bridge = wb.WebViewBridge(ui_loop.dispatcher, host, webview_factory=factory, **kwargs).register()
    return bridge, host, views


def _in_worker(fn, *args, **kwargs):
    box = {}

    def run():
        try:
            box["result"] = fn(*args, **kwargs)
        except BaseException as exc:  # noqa: BLE001 - recorded for the assertion
            box["error"] = exc

    thread = threading.Thread(target=run, name="api-worker", daemon=True)
    thread.start()
    return thread, box


def _wait_for(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_authnd_token_through_the_bridge(ui_loop, mobile_env):
    bridge, host, views = _bridge(ui_loop, NvidiaSite)
    assert browser_driver.get_driver() is bridge and wb.availability() == (True, "")
    thread, box = _in_worker(authnd_auth.get_captcha_token, PAGE_URL, 60)
    thread.join(20)
    assert box.get("result") == "wv-token", box.get("error")
    (view,) = views
    assert view.loads == [PAGE_URL]
    assert len(view.sources) == 2 and authnd_auth.CAPTCHA_STATE_SCRIPT in view.sources[1]
    page_state = view.page.load_state
    assert page_state["generation"] == page_state["finished_generation"] == 1 and page_state["ok"]
    assert view.page.url() == PAGE_URL and view.page.owner == authnd_auth.BROWSER_OWNER
    assert _wait_for(lambda: not host.entries) and host.added == 1           # unmounted after use
    assert bridge.open_pages == 0 and bridge.status()["opened"] == 1


def test_gemini_request_through_the_bridge_with_chunked_answers(ui_loop, mobile_env):
    bridge, host, views = _bridge(ui_loop, lambda: AiModeSite(big=True), chunk_chars=1000)
    thread, box = _in_worker(gemini_free.send_chat_completion,
                             messages=[{"role": "user", "content": "hi"}], model="search/gemini")
    thread.join(30)
    result = box.get("result")
    assert result is not None, box.get("error")
    assert result["content"] == ("Bonjour " * 4000).strip()
    (view,) = views
    assert view.loads == [gemini_free._build_search_base_url()]
    assert view.page.viewport == gemini_free._default_viewport_size()
    assert _wait_for(lambda: not host.entries)


def test_cancel_stream_ends_the_wait_at_once(ui_loop, mobile_env):
    bridge, host, views = _bridge(ui_loop, lambda: NvidiaSite(block=True))
    thread, box = _in_worker(authnd_auth.get_captcha_token, PAGE_URL, 120)
    assert _wait_for(lambda: views and views[0].sources)
    started = time.monotonic()
    authnd_auth.cancel_stream()
    thread.join(5)
    assert not thread.is_alive() and time.monotonic() - started < 2
    assert "stream cancelled" in str(box.get("error"))
    assert _wait_for(lambda: not host.entries) and bridge.open_pages == 0


def test_page_calls_time_out_refuse_the_loop_and_cap_pages(ui_loop, mobile_env):
    bridge, host, views = _bridge(ui_loop, lambda: NvidiaSite(block=True), max_pages=2)
    page = bridge.open_page(owner="authnd")
    page.load(PAGE_URL)
    assert _wait_for(lambda: page.load_state["finished_generation"] == 1)
    assert page.title() == "NVIDIA NIM"
    started = time.monotonic()
    assert page.run_js("1 + 1", js_timeout_ms=200) is None                 # no answer: None, like Qt
    assert 0.15 <= time.monotonic() - started < 2
    late = types.SimpleNamespace(message=wb.CONSOLE_PREFIX + json.dumps({"id": "gone", "ok": True, "value": 1}))
    ui_loop.loop.call_soon_threadsafe(page.on_console, late)                 # unknown id: dropped
    with pytest.raises(ValueError):
        page.load("file:///etc/passwd")
    # the UI loop must never block on a page
    on_loop = asyncio.run_coroutine_threadsafe(_call_on_loop(page), ui_loop.loop).result(5)
    assert "UI thread" in on_loop
    # at most max_pages pages: the third open waits, a cancel check releases it
    second = bridge.open_page(owner="gemini_free")
    flag = threading.Event()
    threading.Timer(0.2, flag.set).start()
    with pytest.raises(browser_driver.BrowserCancelled):
        bridge.open_page(owner="authnd", cancel_check=flag.is_set)
    second.close()
    third = bridge.open_page(owner="authnd")                                   # a slot is free again
    for item in (page, third):
        item.close()
    page.close()                                                              # idempotent
    assert bridge.open_pages == 0
    with pytest.raises(RuntimeError, match="closed"):
        page.run_js("1")


@needs_flet  # navigate_webview lives in the Reader module (it imports flet)
def test_later_loads_navigate_through_load_request_and_failures_raise(ui_loop, mobile_env):
    """The page's first load builds its WebView with the URL; later loads go through the Reader's
    ``navigate_webview`` (``load_request``; the url property follows), and a refused navigation
    still raises out of ``WebViewPage.load`` (authnd/ and search/gemini report it)."""
    bridge, host, views = _bridge(ui_loop, NvidiaSite)
    page = bridge.open_page(owner="authnd")
    page.load(PAGE_URL)
    second = PAGE_URL + "?again=1"
    page.load(second)
    (view,) = views
    assert view.loads == [PAGE_URL, second] and view.url == second and host.added == 1
    page.close()

    class RefusingWebView(FakeWebView):
        async def load_request(self, url):
            raise RuntimeError("the platform view refused the navigation")

    refusing = []

    def factory(page_, url):
        view_ = RefusingWebView(page_, url, NvidiaSite())
        refusing.append(view_)
        return view_

    bridge.webview_factory = factory
    other = bridge.open_page(owner="gemini_free")
    other.load(PAGE_URL)  # the first load builds the WebView: no load_request
    with pytest.raises(RuntimeError, match="refused the navigation"):
        other.load(second)
    assert refusing[0].loads == [PAGE_URL]
    other.close()
    assert _wait_for(lambda: not host.entries) and bridge.open_pages == 0


async def _call_on_loop(page):
    try:
        page.run_js("1")
    except RuntimeError as exc:
        return str(exc)
    return "no error"


def test_cancel_pages_by_owner(ui_loop, mobile_env):
    bridge, host, views = _bridge(ui_loop, lambda: NvidiaSite(block=True))
    nd = bridge.open_page(owner="authnd")
    gf = bridge.open_page(owner="gemini_free")
    for page in (nd, gf):
        page.load(PAGE_URL)
    thread, box = _in_worker(gf.run_js, "1", 30000)
    assert _wait_for(lambda: len(views) == 2 and views[1].sources)
    browser_driver.cancel_pages("authnd")                                     # not gemini's page
    time.sleep(0.2)
    assert thread.is_alive()
    gemini_free.cancel_stream()
    thread.join(5)
    assert isinstance(box.get("error"), browser_driver.BrowserCancelled)
    with pytest.raises(browser_driver.BrowserCancelled):
        nd.wait(10)
    for page in (nd, gf):
        page.close()
    assert _wait_for(lambda: not host.entries)


# ============================================================================ hidden host / UI
@needs_flet
def test_hidden_host_keeps_the_webview_painted_but_unseen():
    import flet as ft

    from glossarion_mobile.ui.components.hidden_webview import HIDDEN_OPACITY, HiddenWebViewHost

    page = types.SimpleNamespace(overlay=[], updates=0)
    page.update = lambda: setattr(page, "updates", page.updates + 1)
    host = HiddenWebViewHost(page)
    control = ft.Text("web view stand-in")
    entry = host.add(control, size=(1280, 900), key="glwvb-1")
    assert page.overlay == [entry] and len(host) == 1 and page.updates == 1
    assert (entry.left, entry.top, entry.width, entry.height) == (0, 0, 1, 1)
    assert entry.opacity == HIDDEN_OPACITY > 0 and entry.ignore_interactions is True
    semantics = entry.content
    assert isinstance(semantics, ft.Semantics) and semantics.exclude_semantics is True
    stack = semantics.content
    assert isinstance(stack, ft.Stack) and stack.clip_behavior == ft.ClipBehavior.HARD_EDGE
    (frame,) = stack.controls
    assert (frame.left, frame.top, frame.width, frame.height) == (0, 0, 1280, 900) and frame.content is control
    assert host.remove(entry) is True and page.overlay == [] and host.remove(entry) is False
    host.add(ft.Text("a"), size=(10, 10), key="a")
    host.add(ft.Text("b"), size=(10, 10), key="b")
    assert host.clear() == 2 and page.overlay == []


@needs_flet
def test_accounts_experimental_rows_follow_the_bridge(monkeypatch):
    import flet as ft

    from glossarion_mobile.ui.router import parse_route
    from glossarion_mobile.ui.screens import accounts

    screen = accounts.AccountsScreen(parse_route("/settings/accounts"), oauth=None)
    monkeypatch.setattr(wb, "availability", lambda: (False, wb.UNSUPPORTED_REASON))
    tile = screen._experimental_tile()
    assert tile.key == "accounts-experimental" and tile.title == f"Experimental ({len(accounts.EXPERIMENTAL_ACCOUNTS)})"
    # U9: enabled rows with no tap action (a disabled ListTile would disable the ReasonChip)
    assert all(not row.disabled and row.on_click is None for row in tile.controls)
    assert {row.trailing.reason for row in tile.controls} == {accounts.NEEDS_WEBVIEW_CHIP}
    assert all(row.trailing.detail == wb.UNSUPPORTED_REASON for row in tile.controls)
    monkeypatch.setattr(wb, "availability", lambda: (True, ""))
    tile = screen._experimental_tile()
    assert [row.key for row in tile.controls] == ["experimental-authnd/", "experimental-search/gemini"]
    assert not any(row.disabled for row in tile.controls)
    assert {row.trailing.reason for row in tile.controls} == {accounts.EXPERIMENTAL_CHIP}
    sheet = screen.show_experimental("AuthND (NVIDIA Build)", accounts.EXPERIMENTAL_ACCOUNTS[0][2])
    assert "hCaptcha" in sheet.body and "user agent" in sheet.body
    assert isinstance(tile.controls[0].subtitle, ft.Text) and "no sign-in" in tile.controls[0].subtitle.value


@needs_flet
def test_install_registers_only_where_flet_webview_runs(ui_loop, mobile_env, monkeypatch):
    app = types.SimpleNamespace(page=types.SimpleNamespace(platform=types.SimpleNamespace(value="windows"), web=False),
                                dispatcher=ui_loop.dispatcher)
    assert asyncio.run(wb.WebViewBridgeFeature.install(app)) is None
    assert app.webview_bridge is None and browser_driver.get_driver() is None
    assert wb.availability() == (False, wb.UNSUPPORTED_REASON)
    monkeypatch.setattr(wb, "platform_supported", lambda page: True)
    bridge = asyncio.run(wb.WebViewBridgeFeature.install(app))
    assert app.webview_bridge is bridge and browser_driver.get_driver() is bridge and wb.current() is bridge
    assert wb.availability() == (True, "")
    bridge.shutdown()
    assert browser_driver.get_driver() is None and wb.availability()[0] is False


def test_new_files_keep_uniform_line_endings():
    for path in (APP_DIR / "glossarion_mobile" / "services" / "webview_bridge.py",
                 APP_DIR / "glossarion_mobile" / "ui" / "components" / "hidden_webview.py",
                 SRC_DIR / "browser_driver.py"):
        data = path.read_bytes()
        assert data.count(b"\r\n") in (0, data.count(b"\n")), path
