"""Hotfix 3, the native fix: the in-app browser routes on pages the extension owns.

The flet-webview pages lived in ``page.overlay``, which Flet 1.0.3 draws only in the top View: a
screen push / pop made Flutter build a new WebView that loaded its URL again without the page
hook, so an authchatgpt/ reply in progress was lost (``_PageLost`` -> resend), and Flutter builds
nothing while the app is hidden (hotfix 1 parks WebViews for jobs). ``services/native_webview.py``
drives ``android.webkit.WebView`` / ``WKWebView`` pages the extension's Kotlin / Swift code creates
outside the Flutter widget tree, through the ``GlossarionNative`` service (``web_open`` ...
``web_close`` + ``web`` events).

The fake extension below has the native semantics of ``HiddenWebPages`` (Kotlin / Swift): page
ids from Python, ``exists`` / ``no_page`` / ``gone`` / ``closed`` answers, pending scripts
answered when their page is destroyed, a renderer crash keeping the id answering ``gone``, and
``started`` / ``finished`` / ``gone`` events through the real ``NativeBridge`` ``web`` handler.
It runs on a real asyncio loop thread with the real ``UiDispatcher``; the routes are the real
backend modules on worker threads (``GLOSSARION_MOBILE=1``, no helper subprocess).

* authnd/, search/gemini (a 32 000-character answer in one piece) and authchatgpt/ (logged out,
  and logged in through the hooked same-origin chat iframe) run unchanged on the native driver;
* a screen push / pop and a sheet (``webview_check.screen_churn`` on a fake app) and the app
  being hidden while a reply is on its way change nothing: no new page, no new document, no
  resend; pages open and load while the app is hidden (no claim, no wait, no Flet control);
* Stop ends a held script at once and closes the native page; a JavaScript timeout gives None
  without cancelling the loop call; a renderer crash answers pending scripts None, sets
  ``renderer_gone`` and authchatgpt/ sends the message again on a new page, also when it happens
  before the reply (the chat frame, the composer, Send: no deadline waited out, the loading lock
  freed at once); a resend the native page did not cause by a stopped renderer is logged as
  chatgpt.com reloading itself, not as a screen change; iOS reports a loss as reason unknown;
* the page cap, the loop-thread refusal, a close during ``web_open`` (no orphan page);
* selection: the native driver when the extension reports ``web_pages``, else the flet-webview
  bridge as before (capability missing, stub, ``GLOSSARION_NATIVE_WEBVIEW=0``, the handshake
  raising or timing out); ``jobs_running`` / one-job-at-a-time lifecycle are no-ops for it;
  shutdown unregisters, cancels and resets; ``limitations()``;
* the on-device self-test check (``diagnostics.webview_check``) against a real ReaderServer page; it
  waits only so long for a page slot that browser-route requests hold (a skip in the smoke suite).

Real data stays isolated (HOME / USERPROFILE -> tmp_path; GLOSSARION_HTTP_LOG=0). Run from src/mobile:
    python -m pytest -p no:cacheprovider -W ignore tests_host/test_hotfix3_native_browser.py
"""

from __future__ import annotations

import ast
import asyncio
import dataclasses
import importlib.util
import json
import logging
import re
import shutil
import subprocess
import sys
import threading
import time
import types
import urllib.request
from pathlib import Path
from typing import Any, Optional

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = MOBILE_DIR / "app"
SRC_DIR = MOBILE_DIR.parent
for entry in (str(APP_DIR), str(SRC_DIR)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import authchatgpt_auth as gpt  # noqa: E402
import authnd_auth  # noqa: E402
import browser_driver  # noqa: E402
import gemini_free  # noqa: E402
from glossarion_mobile.diagnostics import selftest, webview_check  # noqa: E402
from glossarion_mobile.services import native as native_mod  # noqa: E402
from glossarion_mobile.services import native_webview as nw  # noqa: E402
from glossarion_mobile.services import webview_bridge as wb  # noqa: E402
from glossarion_mobile.services.native import EVENTS, NativeBridge, NativeStub  # noqa: E402

# The flet-webview suites' helpers (those files stay unchanged): fixtures, site models, threads.
from test_hotfix1_authchatgpt_bg import MESSAGES, REPLY, ChatGPTSite  # noqa: E402
from test_webview_bridge import (  # noqa: E402,F401
    BLOCK,
    PAGE_URL,
    AiModeSite,
    NvidiaSite,
    _in_worker,
    _wait_for,
    mobile_env,
    ui_loop,
)

NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="node is not installed")
needs_flet = pytest.mark.skipif(importlib.util.find_spec("flet") is None, reason="flet not installed")
CAPABILITY = {"version": 1, "max_pages": 8, "provider": "fake WebView"}
HOST_URL = gpt.BASE_URL + "/robots.txt"
FRAME_PREFIX = "((window, document) => ("
FRAME_SUFFIX = f"))({gpt._FRAME}.contentWindow, {gpt._FRAME}.contentWindow.document)"


def _clean(expression):
    return wb._clean_expression(expression)


# ============================================================================ the fake extension
@dataclasses.dataclass
class WebEvent:
    """Shape of ``flet_glossarion_native.WebPageEvent`` (an ``ft.Event``: ``name`` / ``data`` /
    ``control`` plus the contract's keys); ``NativeBridge.to_plain`` drops ``control``."""

    name: str = "web"
    data: Any = None
    control: Any = None
    type: str = ""
    page_id: Optional[str] = None
    url: Optional[str] = None
    ok: Optional[bool] = None
    error: Optional[str] = None
    code: Optional[int] = None
    main_frame: Optional[bool] = None
    crashed: Optional[bool] = None


class NativePage:
    def __init__(self, page_id, site, width, height, prevent_links):
        self.id, self.site = page_id, site
        self.width, self.height, self.prevent_links = width, height, list(prevent_links)
        self.url = None
        self.gone = False
        self.loads, self.evals, self.timeouts = [], [], []
        self.held = []  # asyncio futures of scripts the site holds (BLOCK)


class FakeNative:
    """``GlossarionNative``'s in-app page methods with ``HiddenWebPages``' semantics (Kotlin /
    Swift); ``site_factory()`` gives each new native page its website model (``answer(expr)``,
    ``title``, optional ``navigated(url)``)."""

    is_stub = False

    def __init__(self, site_factory, *, capability=CAPABILITY, max_pages=8):
        self.site_factory = site_factory
        self.capability = capability
        self.max_pages = max_pages
        self.pages: dict = {}
        self.created: list = []
        self.calls: list = []
        self.info_calls = 0
        self.info_error: Optional[BaseException] = None
        self.info_hang = False
        self.open_gate: Optional[threading.Event] = None  # web_open waits for it (a slow WebView build)
        self.on_web = None  # NativeBridge sets every on_* handler

    def methods(self, name):
        return [page_id for method, page_id in self.calls if method == name]

    # ---- platform facts
    async def get_platform_info(self):
        self.info_calls += 1
        if self.info_error is not None:
            raise self.info_error
        if self.info_hang:
            await asyncio.Event().wait()
        info = {"platform": "android", "native": True, "documents": True}
        if self.capability is not None:
            info["web_pages"] = dict(self.capability)
        return info

    # ---- the channel contract
    @staticmethod
    def _fail(code, message=""):
        return {"ok": False, "error": code, "message": message}

    def _emit(self, page, kind, **extra):
        handler = self.on_web
        if handler is None:
            return
        event = WebEvent(type=kind, page_id=page.id, **extra)
        asyncio.get_running_loop().call_soon(lambda: asyncio.ensure_future(handler(event)))

    def _destroy(self, page, code):
        for future in page.held:
            if not future.done():
                future.set_result(self._fail(code, "the page is gone"))
        page.held.clear()

    async def web_open(self, page_id, *, width, height, prevent_links=()):
        self.calls.append(("web_open", page_id))
        if self.open_gate is not None:
            while not self.open_gate.is_set():
                await asyncio.sleep(0.005)
        if not page_id:
            return self._fail("bad_args", "page_id is required")
        if page_id in self.pages:
            return self._fail("exists", "the page is already open")
        if len(self.pages) >= self.max_pages:
            return self._fail("too_many", "too many in-app pages")
        page = NativePage(page_id, self.site_factory(), int(width), int(height), prevent_links)
        self.pages[page_id] = page
        self.created.append(page)
        return {"ok": True, "page_id": page_id, "attached": True}

    def _page(self, page_id):
        page = self.pages.get(page_id)
        if page is None:
            return None, self._fail("no_page", "no such page")
        if page.gone:
            return None, self._fail("gone", "the page's renderer stopped")
        return page, None

    async def web_load(self, page_id, url):
        self.calls.append(("web_load", page_id))
        page, failure = self._page(page_id)
        if failure:
            return failure
        if not str(url).startswith(("http://", "https://")):
            return self._fail("bad_args", "only http(s) pages")
        page.loads.append(url)
        page.url = url
        navigated = getattr(page.site, "navigated", None)
        if callable(navigated):
            navigated(url)
        self._emit(page, "started", url=url)
        self._emit(page, "finished", url=url, ok=True, error=None)
        return {"ok": True}

    async def web_eval(self, page_id, script, *, timeout=30.0):
        self.calls.append(("web_eval", page_id))
        page, failure = self._page(page_id)
        if failure:
            return failure
        head, tail = "var value = (\n", "\n    );\n    envelope = {ok: true"
        expression = script.split(head, 1)[1].rsplit(tail, 1)[0]
        page.evals.append(expression)
        page.timeouts.append(timeout)
        value = page.site.answer(expression)
        if value is BLOCK:
            future = asyncio.get_running_loop().create_future()
            page.held.append(future)
            try:  # the extension's own invoke timeout (the script's + 5 s)
                return await asyncio.wait_for(future, float(timeout) + 5.0)
            except asyncio.TimeoutError:
                return self._fail("timeout", "web_eval did not answer in time")
        return {"ok": True, "result": json.dumps({"ok": True, "value": value}), "url": page.url}

    async def web_title(self, page_id):
        self.calls.append(("web_title", page_id))
        page, failure = self._page(page_id)
        if failure:
            return failure
        return {"ok": True, "title": page.site.title if page.url else None, "url": page.url}

    async def web_close(self, page_id):
        self.calls.append(("web_close", page_id))
        page = self.pages.pop(page_id, None)
        if page is not None:
            self._destroy(page, "closed")
        return {"ok": True, "closed": page is not None}

    async def web_reset(self):
        self.calls.append(("web_reset", None))
        pages = list(self.pages.values())
        self.pages.clear()
        for page in pages:
            self._destroy(page, "closed")
        return {"ok": True, "closed": len(pages)}

    async def web_status(self):
        self.calls.append(("web_status", None))
        return {"ok": True, "attached": bool(self.pages), "activity": True, "density": 2.75,
                "pages": [{"page_id": p.id, "url": p.url, "gone": p.gone, "width": p.width, "height": p.height}
                          for p in self.pages.values()]}

    # ---- test hooks (loop thread)
    def crash(self, page_id, crashed=True):
        """``onRenderProcessGone``: the page reports ``gone``, pending scripts answer ``gone`` and the
        id keeps answering ``gone`` until it is closed."""
        page = self.pages[page_id]
        page.gone = True
        self._emit(page, "gone", crashed=crashed)
        self._destroy(page, "gone")

    def release(self, page_id, value):
        page = self.pages[page_id]
        held, page.held = page.held, []
        for future in held:
            if not future.done():
                future.set_result({"ok": True, "result": json.dumps({"ok": True, "value": value})})


class ChatGPTFrameSite:
    """chatgpt.com as the logged-in authchatgpt/ flow meets it on a phone (memory:
    authchatgpt-web-chat-route): the small host page chatgpt.com/robots.txt, the same-origin chat
    iframe hooked at insertion, every later script run against the frame's window. ``hold``
    (threading.Event): polls report a reply in progress until it is set."""

    title = "chatgpt.com"

    def __init__(self, hold=None):
        self.hold = hold
        self.url = ""
        self.mark = None
        self.frame = self.hooked = self.requested = self.polled = False
        self.frame_inserts = self.polls = self.prompts = self.probes = 0
        self.framed_scripts = 0

    def navigated(self, url):
        self.url, self.mark = url, None
        self.frame = self.hooked = self.requested = self.polled = False

    def answer(self, expression):
        if expression.startswith(FRAME_PREFIX) and expression.endswith(FRAME_SUFFIX):
            if not self.frame:
                return None  # contentWindow of a missing frame: the script throws
            self.framed_scripts += 1
            return self._in_frame(expression[len(FRAME_PREFIX):-len(FRAME_SUFFIX)])
        on_chatgpt = self.url.startswith(gpt.BASE_URL + "/")
        if expression.startswith("(window.__glossarionNav = '"):
            self.mark = expression.split("'")[1]
            return True
        if "window.__glossarionNav !== '" in expression:
            return on_chatgpt and self.mark != expression.split("window.__glossarionNav !== '")[1].split("'")[0]
        if expression == "document.cookie.indexOf('oai-did=') >= 0":
            return True  # a returning visitor: the device cookie is there
        if expression == _clean(gpt._APP_COOKIE_JS):
            return True
        if expression.startswith(f"(() => {{const old = {gpt._FRAME};"):  # _frame_script: insert + hook
            self.frame = self.hooked = on_chatgpt
            self.frame_inserts += 1
            return True
        if expression == f"!{gpt._FRAME}":
            return not self.frame
        if "f.contentWindow.__glossarionGPT" in expression:  # hook_lost: asks the top page
            self.probes += 1
            return not (self.frame and self.hooked)
        return None

    def _in_frame(self, inner):
        if inner == _clean(gpt._READY):
            return self.hooked
        if inner == _clean(gpt._FRAME_LOST):
            return False
        if inner == _clean(gpt._COMPOSER):
            return {"text": ".", "send": True, "requested": self.requested}
        if inner == _clean(gpt._SEND):
            self.requested = self.hooked
            self.prompts += 1
            return True
        if inner == _clean(gpt._POLL):
            self.polls += 1
            if not self.hooked:
                return None
            state = {"chunks": [], "streams": 1, "open": 1, "idle": 0, "status": 200, "error": None, "kind": None,
                     "sentinel": 0, "resolved": None, "final": None, "finalImages": None}
            if self.hold is not None and not self.hold.is_set():
                return state  # the reply is on its way
            chunks, self.polled = ([] if self.polled else [REPLY]), True
            return dict(state, chunks=chunks, open=0)
        return True  # watch(), typing


class SelfTestSite:
    """The self-test's loopback page (``webview_check.SELFTEST_HTML`` from a real ReaderServer):
    the title is read from the served document; the check's scripts are answered per document.
    ``lose`` forgets the page's JavaScript state (what a rebuilt WebView did)."""

    def __init__(self):
        self.url = ""
        self.globals: dict = {}
        self.lose = False
        self._title = None

    def navigated(self, url):
        self.url, self.globals, self._title = url, {}, None

    @property
    def title(self):
        if self._title is None and self.url:
            with urllib.request.urlopen(self.url, timeout=10) as response:  # 127.0.0.1 only
                html = response.read().decode("utf-8")
            match = re.search(r"<title>(.*?)</title>", html)
            self._title = match.group(1) if match else ""
        return self._title

    def answer(self, expression):
        set_head = _clean(webview_check.SET_TOKEN_SCRIPT).split("{token}")[0]
        if expression.startswith(set_head):
            self.globals["token"] = json.loads(re.search(r"= (\"[0-9a-f]+\")", expression).group(1))
            return self.globals["token"]
        if expression == webview_check.GET_TOKEN_SCRIPT:
            return None if self.lose else self.globals.get("token")
        if expression == webview_check.MATH_SCRIPT:
            return 42
        if expression == _clean(webview_check.TYPE_SCRIPT):
            return "ok"
        if expression == webview_check.BIG_SCRIPT:
            return "x" * webview_check.BIG_SIZE
        return None


# ============================================================================ fixtures
@pytest.fixture
def native_env(mobile_env, monkeypatch):
    for name in ("AUTHCHATGPT_MODE", nw.ENV_SWITCH, "GLOSSARION_SIMULATE_PLATFORM"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GLOSSARION_HTTP_LOG", "0")
    monkeypatch.setattr(gpt, "_access_token", lambda log_fn: None)  # logged out unless a test logs in

    def no_flet_webview(slot, url):
        raise AssertionError("a flet-webview control was built")

    monkeypatch.setattr(wb, "_flet_webview", no_flet_webview)
    gpt.reset_cancel()
    yield monkeypatch
    gpt.reset_cancel()


class Harness:
    def __init__(self, ui_loop, site_factory, **kwargs):
        self.loop = ui_loop.loop
        self.fake = FakeNative(site_factory, **kwargs)
        self.bridge = NativeBridge(native=self.fake)
        self.driver = nw.NativeWebDriver(ui_loop.dispatcher, self.bridge, capability=CAPABILITY).register()

    def on_loop(self, fn, *args):
        """Run ``fn(*args)`` on the loop thread (a native callback) and wait for it."""
        done = threading.Event()
        box = {}

        def run():
            try:
                box["result"] = fn(*args)
            finally:
                done.set()

        self.loop.call_soon_threadsafe(run)
        assert done.wait(5)
        return box.get("result")

    def run(self, coro_fn, *args, timeout=10):
        return asyncio.run_coroutine_threadsafe(coro_fn(*args), self.loop).result(timeout)


@pytest.fixture
def harness(ui_loop, native_env):
    made = []

    def make(site_factory=NvidiaSite, **kwargs):
        made.append(Harness(ui_loop, site_factory, **kwargs))
        return made[-1]

    yield make
    for item in made:  # resolve held scripts, leave no native page and no registered driver
        resets = len(item.fake.methods("web_reset"))
        item.driver.shutdown()
        assert _wait_for(lambda h=item: len(h.fake.methods("web_reset")) > resets and not h.fake.pages
                         and not any(p.held for p in h.fake.created))


def _chat(logs, *, timeout=60):
    return _in_worker(gpt.send_chat_completion, messages=MESSAGES, model="authchatgpt/auto", timeout=timeout,
                      log_fn=logs.append)


# ============================================================================ the routes, unchanged
def test_authnd_token_through_native_pages(harness):
    h = harness(NvidiaSite)
    assert browser_driver.get_driver() is h.driver and wb.availability() == (True, "") and wb.current() is h.driver
    thread, box = _in_worker(authnd_auth.get_captcha_token, PAGE_URL, 60)
    thread.join(20)
    assert box.get("result") == "wv-token", box.get("error")
    (page,) = h.fake.created
    assert page.loads == [PAGE_URL] and page.prevent_links == list(wb.PREVENT_LINKS)
    assert (page.width, page.height) == wb.DEFAULT_VIEWPORT
    assert authnd_auth.CAPTCHA_STATE_SCRIPT in page.evals[1]
    assert _wait_for(lambda: h.fake.methods("web_close") == [page.id] and not h.fake.pages)
    assert h.driver.open_pages == 0 and h.driver.status()["opened"] == 1


def test_gemini_answer_arrives_in_one_piece(harness):
    h = harness(lambda: AiModeSite(big=True))
    thread, box = _in_worker(gemini_free.send_chat_completion,
                             messages=[{"role": "user", "content": "hi"}], model="search/gemini")
    thread.join(30)
    result = box.get("result")
    assert result is not None, box.get("error")
    assert result["content"] == ("Bonjour " * 4000).strip()  # ~32 000 characters, no chunking
    (page,) = h.fake.created
    assert page.loads == [gemini_free._build_search_base_url()]
    assert (page.width, page.height) == tuple(gemini_free._default_viewport_size())
    assert _wait_for(lambda: not h.fake.pages) and h.driver.open_pages == 0


def test_authchatgpt_logged_out_runs_on_the_native_page(harness):
    h = harness(ChatGPTSite)
    logs = []
    thread, box = _chat(logs)
    thread.join(30)
    assert (box.get("result") or {}).get("content") == "Title", box.get("error")
    (page,) = h.fake.created
    assert page.loads == [gpt.BASE_URL + "/"] and page.site.prompts == 1
    assert not any("again" in line for line in logs)
    assert _wait_for(lambda: not h.fake.pages)


class FakeShell:
    def __init__(self, app, tablet):
        self.app, self.tablet, self.stack = app, tablet, []

    @property
    def current_route(self):
        return self.stack[-1] if self.stack else "/"


class FakeApp:
    """What ``webview_check.screen_churn`` uses of the App: ``navigate`` pushes a View on phones
    (Flet then re-parents ``page.overlay``: the old bug's trigger), ``back`` pops it. ``welcome``:
    back on the chat home reopens an unfinished first-run Welcome (``App._resume_welcome``)."""

    def __init__(self, *, tablet=False, welcome=False):
        self.welcome = welcome
        self.page = types.SimpleNamespace(views=["/"], overlay=[], dialogs=[], width=412, height=860,
                                          app_visible=True, updates=0)
        self.page.show_dialog = self._show_dialog
        self.page.update = lambda: setattr(self.page, "updates", self.page.updates + 1)
        self.shell = FakeShell(self, tablet)
        self.routes: list = []

    def _show_dialog(self, dialog):
        dialog.open = True
        self.page.dialogs.append(dialog)

    async def navigate(self, route, **kwargs):
        self.routes.append(route)
        if route == "/":
            self.shell.stack = []
            del self.page.views[1:]
            return
        self.shell.stack.append(route)
        if not self.shell.tablet:
            self.page.views.append(route)

    def back(self):
        if self.shell.stack:
            self.shell.stack.pop()
            if not self.shell.tablet:
                self.page.views.pop()
        if self.welcome and not self.shell.stack:
            self.welcome = False
            self.shell.stack.append("/welcome")
            self.page.views.append("/welcome")


@needs_flet  # the churn's InfoSheet is a Flet BottomSheet
def test_logged_in_chat_frame_survives_screen_changes_and_the_app_hiding(harness, caplog):
    """The owner's case: a logged-in authchatgpt/ reply is on its way in the hooked chat iframe while
    the app pushes and pops Settings, opens a sheet and goes to the background. The native page
    never notices: one page, one document, no rebuild probe, no resend."""
    hold = threading.Event()
    h = harness(lambda: ChatGPTFrameSite(hold))
    app = FakeApp()
    app.page.app_visible = False  # Glossarion is in the background the whole time
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(gpt, "_access_token", lambda log_fn: "header.e30.signature")  # logged in: the iframe
        caplog.set_level(logging.INFO, logger="glossarion.webview_bridge")
        logs = []
        thread, box = _chat(logs)
        assert _wait_for(lambda: h.fake.created and h.fake.created[0].site.polls >= 2, timeout=30)
        (page,) = h.fake.created
        driver_page = h.driver.pages()[0]
        generation = driver_page.load_state["generation"]
        churn = asyncio.run_coroutine_threadsafe(webview_check.screen_churn(app, settle=0), h.loop).result(10)
        polls = page.site.polls
        assert _wait_for(lambda: page.site.polls > polls)  # still answering, still the same page
        hold.set()
        thread.join(30)
    assert (box.get("result") or {}).get("content") == "Title", box.get("error")
    assert churn["pushed"] == "/settings" and churn["views_pushed"] == churn["views_before"] + 1
    assert churn["after"] == churn["before"] == "/" and app.page.dialogs and not app.page.dialogs[0].open
    assert len(h.fake.created) == 1 and h.fake.methods("web_open") == [page.id]
    assert page.loads == [HOST_URL] and page.site.frame_inserts == 1 and page.site.framed_scripts > 3
    assert driver_page.load_state["generation"] == generation == 1  # no new document
    assert page.site.probes == 0 and page.site.prompts == 1
    assert not any("again" in line for line in logs)  # no _PageLost, no resend
    assert app.page.overlay == [] and not hasattr(h.driver, "host")  # no Flet control anywhere
    assert wb.HIDDEN_WAIT_MESSAGE not in caplog.text
    assert _wait_for(lambda: not h.fake.pages)


def test_a_renderer_crash_mid_reply_sends_the_message_again_on_a_new_page(harness):
    hold = threading.Event()
    sites = [ChatGPTFrameSite(hold), ChatGPTFrameSite()]
    h = harness(lambda: sites.pop(0))
    logs = []
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(gpt, "_access_token", lambda log_fn: "header.e30.signature")
        thread, box = _chat(logs)
        assert _wait_for(lambda: h.fake.created and h.fake.created[0].site.polls >= 1, timeout=30)
        first = h.fake.created[0]
        h.on_loop(h.fake.crash, first.id)
        thread.join(30)
    assert (box.get("result") or {}).get("content") == "Title", box.get("error")
    assert len(h.fake.created) == 2 and h.fake.created[1].loads == [HOST_URL]
    resent = [line for line in logs if "sending the message again" in line]
    assert len(resent) == 1 and resent[0].endswith(f"(1/{gpt.MAX_PAGE_RESENDS}).")
    assert "its renderer stopped" in resent[0] and "screen change" not in resent[0]
    assert first.site.probes == 0  # renderer_gone answered hook_lost without a script
    assert h.fake.created[1].site.prompts == 1 and first.site.prompts == 1
    assert _wait_for(lambda: set(h.fake.methods("web_close")) >= {first.id, h.fake.created[1].id})
    assert h.driver.status()["renderer_losses"] == 1


class _StuckFrameSite(ChatGPTFrameSite):
    """Logged in: the chat frame's app never starts (``_READY`` stays False, ``_driver_frame_ready``)."""

    def __init__(self):
        super().__init__()
        self.waits = 0

    def _in_frame(self, inner):
        if inner == _clean(gpt._READY):
            self.waits += 1
            return False
        return super()._in_frame(inner)


class _StuckComposerSite(ChatGPTSite):
    """Logged out: chatgpt.com never shows its composer (the ``_driver_wait`` in ``_driver_tab``)."""

    def __init__(self):
        super().__init__()
        self.waits = 0

    def answer(self, expression):
        if expression == "!!document.querySelector('#prompt-textarea')":
            self.waits += 1
            return False
        return super().answer(expression)


class _StuckSendSite(ChatGPTSite):
    """Logged out: the app started but Send never turns on (``_submit`` keeps waiting)."""

    def __init__(self):
        super().__init__()
        self.waits = 0

    def answer(self, expression):
        if expression == gpt._COMPOSER:
            self.waits += 1
            return {"text": ".", "send": False, "requested": False}
        return super().answer(expression)


STUCK_SITES = {"frame": (_StuckFrameSite, ChatGPTFrameSite), "composer": (_StuckComposerSite, ChatGPTSite),
               "send": (_StuckSendSite, ChatGPTSite)}


@pytest.mark.parametrize("phase", sorted(STUCK_SITES))
def test_a_renderer_crash_before_the_reply_sends_the_message_again_at_once(harness, phase):
    """A renderer that stops while the page is still set up (the chat frame starting, the composer
    appearing, Send turning on) makes every script answer None. Those waits used to poll the dead
    page until their deadlines (twice 90 s for the frame, 45 s for Send) and fail the request with
    "did not finish loading", never resent; now it goes again on a new page at once (1/3)."""
    stuck_class, healthy = STUCK_SITES[phase]
    stuck = stuck_class()
    sites = [stuck]
    h = harness(lambda: sites.pop(0) if sites else healthy())
    logs = []
    with pytest.MonkeyPatch.context() as patch:
        if phase == "frame":
            patch.setattr(gpt, "_access_token", lambda log_fn: "header.e30.signature")  # logged in: the iframe
        thread, box = _chat(logs)
        assert _wait_for(lambda: stuck.waits >= 2, timeout=30)
        first = h.fake.created[0]
        h.on_loop(h.fake.crash, first.id)
        thread.join(30)  # the old waits needed 45 to 180 s more
    assert not thread.is_alive()
    assert (box.get("result") or {}).get("content") == "Title", box.get("error")
    assert len(h.fake.created) == 2 and h.fake.created[1].site.prompts == 1 and stuck.prompts == 0
    resent = [line for line in logs if "sending the message again" in line]
    assert len(resent) == 1 and resent[0].endswith(f"(1/{gpt.MAX_PAGE_RESENDS}).")
    assert "its renderer stopped" in resent[0] and "screen change" not in resent[0]
    assert _wait_for(lambda: first.id in h.fake.methods("web_close") and not h.fake.pages)


def test_a_renderer_crash_during_setup_frees_the_loading_lock_at_once(harness):
    """Parallel requests (glossary extraction, batch translation) load their pages one at a time
    (``_driver_loading``). A page whose renderer stopped while it held that lock kept it until both
    frame deadlines ran out (3 minutes) and every other request waited behind it."""
    stuck = _StuckFrameSite()
    sites = [stuck]
    h = harness(lambda: sites.pop(0) if sites else ChatGPTFrameSite())
    first_logs, second_logs = [], []
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(gpt, "_access_token", lambda log_fn: "header.e30.signature")
        first, first_box = _chat(first_logs)
        assert _wait_for(lambda: stuck.waits >= 2, timeout=30)
        second, second_box = _chat(second_logs)  # waits for the loading lock the stuck page holds
        h.on_loop(h.fake.crash, h.fake.created[0].id)
        second.join(30)
        first.join(30)
    for box in (first_box, second_box):
        assert (box.get("result") or {}).get("content") == "Title", box.get("error")
    assert len(h.fake.created) == 3 and not any("again" in line for line in second_logs)
    assert len([line for line in first_logs if "sending the message again" in line]) == 1
    assert _wait_for(lambda: not h.fake.pages)


class _SelfReloadingSite(ChatGPTSite):
    """Logged-out chatgpt.com as the top page; the poll reports a reply on its way until ``hold`` is
    set, and the hook probe (``hook_lost``) asks the top page itself."""

    def __init__(self, hold):
        super().__init__()
        self.hold = hold
        self.polls = 0

    def answer(self, expression):
        if expression == gpt._POLL and self.hooked:
            self.polls += 1
            if not self.hold.is_set():
                return {"chunks": [], "streams": 1, "open": 1, "idle": 0, "status": 200, "error": None,
                        "kind": None, "sentinel": 0, "resolved": None, "final": None, "finalImages": None}
        if "f.contentWindow.__glossarionGPT" in expression:
            return not self.hooked
        return super().answer(expression)


def test_a_native_page_chatgpt_reloads_is_not_blamed_on_a_screen_change(harness):
    """Native pages survive screen changes: a resend without a stopped renderer is chatgpt.com
    replacing its own document (a new build), and the log says so instead of "a screen change"."""
    hold, done = threading.Event(), threading.Event()
    done.set()
    sites = [_SelfReloadingSite(hold), _SelfReloadingSite(done)]
    h = harness(lambda: sites.pop(0))
    logs = []
    thread, box = _chat(logs)
    assert _wait_for(lambda: h.fake.created and h.fake.created[0].site.polls >= 2, timeout=30)
    first = h.fake.created[0]
    first.site.navigated(first.url)  # a new document without the hook, on the same native page
    thread.join(30)
    assert (box.get("result") or {}).get("content") == "Title", box.get("error")
    resent = [line for line in logs if "sending the message again" in line]
    assert len(resent) == 1 and "chatgpt.com reloaded its page while it was answering" in resent[0]
    assert "screen change" not in resent[0] and resent[0].endswith(f"(1/{gpt.MAX_PAGE_RESENDS}).")
    assert h.driver.status()["renderer_losses"] == 0 and len(h.fake.created) == 2
    assert _wait_for(lambda: not h.fake.pages)


def test_the_resend_line_names_what_took_the_document():
    native = types.SimpleNamespace(name=nw.NATIVE_NAME)
    assert gpt._NATIVE_DRIVER == nw.NATIVE_NAME
    gone = gpt._page_lost_message(gpt._DriverPage(types.SimpleNamespace(driver=native, renderer_gone=True)), 1)
    reload = gpt._page_lost_message(gpt._DriverPage(types.SimpleNamespace(driver=native, renderer_gone=False)), 2)
    rebuilt = gpt._page_lost_message(gpt._DriverPage(types.SimpleNamespace(driver=wb.WebViewBridge)), 3)
    assert "its renderer stopped" in gone and gone.endswith(f"(1/{gpt.MAX_PAGE_RESENDS}).")
    assert "chatgpt.com reloaded its page" in reload and "screen change" not in reload
    assert "(a screen change)" in rebuilt and rebuilt.endswith(f"(3/{gpt.MAX_PAGE_RESENDS}).")
    # A page lost before the reply (renderer gone) is the same _PageLost the reply poll raises.
    with pytest.raises(gpt._PageLost) as info:
        gpt._check_page(gpt._DriverPage(types.SimpleNamespace(renderer_gone=True)))
    assert info.value.error_type == "browser_error" and info.value.page.page.renderer_gone is True
    gpt._check_page(gpt._DriverPage(types.SimpleNamespace(renderer_gone=False)))  # no error
    gpt._check_page(object())  # a desktop page: no renderer_gone at all


@pytest.mark.parametrize("route", ["authnd", "authchatgpt"])
def test_stop_ends_a_held_script_at_once_and_closes_the_native_page(harness, route):
    if route == "authnd":
        h = harness(lambda: NvidiaSite(block=True))
        thread, box = _in_worker(authnd_auth.get_captcha_token, PAGE_URL, 120)
        cancel = authnd_auth.cancel_stream
    else:
        class Stuck(ChatGPTSite):
            def answer(self, expression):
                return BLOCK if expression == gpt._POLL else super().answer(expression)

        h = harness(Stuck)
        thread, box = _chat([])
        cancel = gpt.cancel_stream
    assert _wait_for(lambda: h.fake.created and h.fake.created[0].held, timeout=30)
    (page,) = h.fake.created
    started = time.monotonic()
    cancel()
    thread.join(5)
    assert not thread.is_alive() and time.monotonic() - started < 2
    error = box.get("error")
    assert error is not None and ("stream cancelled" in str(error) or getattr(error, "error_type", "") == "cancelled")
    assert _wait_for(lambda: page.id in h.fake.methods("web_close") and not page.held)
    assert _wait_for(lambda: h.driver.open_pages == 0)


# ============================================================================ page semantics
def _open_loaded(h, owner="authnd", url=PAGE_URL, **kwargs):
    page = h.driver.open_page(owner=owner, **kwargs)
    page.load(url)
    assert browser_driver.wait_until_loaded(page, 5)
    return page


def test_events_drive_load_state_like_the_flet_bridge(harness):
    h = harness(NvidiaSite)
    page = _open_loaded(h)
    state = page.load_state
    assert state["generation"] == state["finished_generation"] == 1 and state["ok"] and state["error"] == ""
    assert page.url() == PAGE_URL and page.title() == "NVIDIA NIM"
    (native,) = h.fake.created
    payloads = [
        {"type": "started", "url": PAGE_URL + "?next=1"},
        {"type": "error", "url": PAGE_URL, "code": -2, "error": "net::ERR_NAME_NOT_RESOLVED", "main_frame": True},
        {"type": "finished", "url": PAGE_URL + "?next=1", "ok": False, "error": "net::ERR_NAME_NOT_RESOLVED"},
        {"type": "url", "url": PAGE_URL + "#pushed"},
        {"type": "error", "url": "https://ads.example/x.js", "code": -6, "error": "tracker blocked", "main_frame": False},
    ]
    for payload in payloads:
        h.run(h.bridge.native.on_web, WebEvent(page_id=native.id, **payload))
    assert state["generation"] == state["finished_generation"] == 2
    assert state["ok"] is True  # parity: finished always counts as loaded; the failure is in "error"
    assert state["error"] == "net::ERR_NAME_NOT_RESOLVED" and state["last_event"] == "resource error: tracker blocked"
    assert page.url() == PAGE_URL + "#pushed"
    h.run(h.bridge.native.on_web, WebEvent(page_id="not-ours", type="started", url="https://x.example/"))
    assert state["generation"] == 2 and h.driver.status()["events"] >= 7
    page.close()


def test_page_calls_time_out_refuse_the_loop_and_cap_pages(harness):
    h = harness(lambda: NvidiaSite(block=True))
    errors = []
    h.loop.call_soon_threadsafe(h.loop.set_exception_handler, lambda loop, context: errors.append(context))
    page = _open_loaded(h)
    started = time.monotonic()
    assert page.run_js("1 + 1", js_timeout_ms=200) is None  # no answer: None, like Qt
    assert 0.15 <= time.monotonic() - started < 5
    (native,) = h.fake.created
    assert native.timeouts[-1] == pytest.approx(0.2) and len(native.held) == 1  # the loop call is still on
    h.on_loop(h.fake.release, native.id, 2)  # the late answer is dropped quietly
    assert _wait_for(lambda: not native.held)
    with pytest.raises(ValueError):
        page.load("file:///etc/passwd")
    on_loop = h.run(_call_on_loop, page)
    assert "UI thread" in on_loop
    # at most MAX_PAGES (4) pages: the fifth open waits, its cancel check releases it
    others = [h.driver.open_page(owner="gemini_free") for _ in range(wb.MAX_PAGES - 1)]
    flag = threading.Event()
    threading.Timer(0.2, flag.set).start()
    with pytest.raises(browser_driver.BrowserCancelled):
        h.driver.open_page(owner="authnd", cancel_check=flag.is_set)
    others.pop().close()
    extra = h.driver.open_page(owner="authnd")  # a slot is free again
    for item in [page, extra] + others:
        item.close()
    page.close()  # idempotent
    assert h.driver.open_pages == 0
    with pytest.raises(RuntimeError, match="closed"):
        page.run_js("1")
    assert _wait_for(lambda: not h.fake.pages)
    assert errors == []  # nothing failed on the loop


async def _call_on_loop(page):
    try:
        page.run_js("1")
    except RuntimeError as exc:
        return str(exc)
    return "no error"


def test_a_close_during_web_open_leaves_no_native_page(harness):
    h = harness(NvidiaSite)
    h.fake.open_gate = threading.Event()  # the WebView build takes a while
    page = h.driver.open_page(owner="authnd")
    thread, box = _in_worker(page.run_js, "1", 10000)
    assert _wait_for(lambda: h.fake.methods("web_open"))
    page.close()
    thread.join(5)
    assert not thread.is_alive() and isinstance(box.get("error"), browser_driver.BrowserCancelled)
    h.fake.open_gate.set()
    assert _wait_for(lambda: h.fake.created and h.fake.methods("web_close") and not h.fake.pages)
    assert h.driver.open_pages == 0


@pytest.mark.parametrize("stop", ["cancel_pages", "cancel_check"])
def test_a_stop_during_web_open_closes_the_page_it_opened(harness, stop):
    """Stop while the WebView is still being built: the worker ends at once; the open that lands
    afterwards closes its native page again (``cancel_pages``: at once; a request's own cancel
    check: when the route closes its page on the way out)."""
    h = harness(NvidiaSite)
    h.fake.open_gate = threading.Event()
    flag = threading.Event()
    page = h.driver.open_page(owner="authnd", cancel_check=flag.is_set)
    thread, box = _in_worker(page.load, PAGE_URL)
    assert _wait_for(lambda: h.fake.methods("web_open"))
    if stop == "cancel_pages":
        authnd_auth.cancel_stream()
    else:
        flag.set()
    thread.join(5)
    assert not thread.is_alive() and isinstance(box.get("error"), browser_driver.BrowserCancelled)
    if stop == "cancel_check":
        page.close()  # the route's finally
    h.fake.open_gate.set()
    assert _wait_for(lambda: h.fake.created and h.fake.methods("web_close")[-1:] == [page.id] and not h.fake.pages)
    assert h.fake.created[0].loads == []  # opened, never navigated, closed again
    page.close()


def test_renderer_crash_answers_pending_scripts_none_and_the_page_stays_lost(harness):
    h = harness(lambda: NvidiaSite(block=True))
    page = _open_loaded(h)
    (native,) = h.fake.created
    thread, box = _in_worker(page.run_js, "document.title", 30000)
    assert _wait_for(lambda: native.held)
    h.on_loop(h.fake.crash, native.id)
    thread.join(5)
    assert not thread.is_alive() and "error" not in box and box.get("result") is None  # not cancelled
    assert _wait_for(lambda: page.crashed is True)
    assert page.renderer_gone and page.load_state["last_event"] == "renderer gone (crashed)"
    assert page.run_js("1") is None and page.title() == ""
    with pytest.raises(RuntimeError, match="lost"):
        page.load(PAGE_URL)
    assert gpt._DriverPage(page).hook_lost() is True
    assert h.driver.status()["renderer_gone"] == 1
    page.close()
    assert _wait_for(lambda: not h.fake.pages)


def test_a_page_the_extension_no_longer_has_counts_as_lost(harness):
    """``no_page`` for a page that was open (a web_reset, a new engine) is a lost document: the route
    starts over instead of failing on a dead id."""
    h = harness(NvidiaSite)
    page = _open_loaded(h)
    h.run(h.fake.web_reset)
    assert page.run_js("1") is None and page.renderer_gone and page.crashed is None
    fresh = h.driver.open_page(owner="authnd")
    assert fresh.run_js("1") is None and not fresh.renderer_gone  # not open yet: no lost flag
    for item in (page, fresh):
        item.close()


def test_hook_lost_needs_no_script_once_the_renderer_is_gone():
    calls = []
    page = types.SimpleNamespace(renderer_gone=True, run_js=lambda *a: calls.append(a))
    assert gpt._DriverPage(page).hook_lost() is True and calls == []
    page.renderer_gone = False
    page.run_js = lambda script, ms: calls.append(ms) or False
    assert gpt._DriverPage(page).hook_lost() is False and calls == [10000]


def test_pages_open_and_load_whatever_the_app_shows(harness, caplog):
    """No claim, no wait, no parked WebView, no Flet control: the native page is created on demand,
    also while the app is hidden (the driver has no notion of app visibility at all)."""
    h = harness(NvidiaSite)
    caplog.set_level(logging.INFO)
    page = _open_loaded(h)
    assert page.run_js(authnd_auth.CAPTCHA_STATE_SCRIPT) is not None
    for name in ("claim", "host", "app_visible", "prewarm", "expect_parking"):
        assert not hasattr(h.driver, name), name
    h.driver.job_started(["authchatgpt/auto"])  # PageDriver's no-ops
    h.driver.keep_warm(True)
    assert wb.HIDDEN_WAIT_MESSAGE not in caplog.text and "hidden_waits" not in h.driver.status()
    assert [method for method, _ in h.fake.calls].count("web_open") == 1
    page.close()


# ============================================================================ selection / lifecycle
@pytest.fixture
def ext_capability(monkeypatch):
    """``flet_glossarion_native.web_pages.capability`` (the extension's contract module). Until that
    module exists in this checkout a shim with the contract's rule stands in for it."""
    try:
        from flet_glossarion_native import web_pages  # noqa: F401
        return
    except Exception:
        pass
    shim = types.ModuleType("flet_glossarion_native.web_pages")

    def capability(info):
        if not isinstance(info, dict) or info.get("native") is not True or not isinstance(info.get("web_pages"), dict):
            return None
        try:
            return dict(info["web_pages"]) if int(info["web_pages"].get("version") or 0) >= 1 else None
        except (TypeError, ValueError):
            return None

    shim.capability = capability
    monkeypatch.setitem(sys.modules, "flet_glossarion_native.web_pages", shim)
    try:
        import flet_glossarion_native as package

        monkeypatch.setattr(package, "web_pages", shim, raising=False)
    except Exception:
        pkg = types.ModuleType("flet_glossarion_native")
        pkg.web_pages = shim
        monkeypatch.setitem(sys.modules, "flet_glossarion_native", pkg)


def _hide_ext_web_pages(monkeypatch):
    """An extension from before hotfix 3: ``flet_glossarion_native.web_pages`` does not import."""
    monkeypatch.setitem(sys.modules, "flet_glossarion_native.web_pages", None)
    package = sys.modules.get("flet_glossarion_native")
    if package is not None:
        monkeypatch.delattr(package, "web_pages", raising=False)


def _app(ui_loop, native):
    return types.SimpleNamespace(native=native, dispatcher=ui_loop.dispatcher, webview_bridge="unset",
                                 page=types.SimpleNamespace(platform=types.SimpleNamespace(value="android"), web=False))


def _install(ui_loop, app):
    return asyncio.run_coroutine_threadsafe(wb.WebViewBridgeFeature.install(app), ui_loop.loop).result(15)


def test_install_prefers_the_native_pages_when_the_extension_offers_them(ui_loop, native_env, ext_capability):
    fake = FakeNative(NvidiaSite)
    bridge = NativeBridge(native=fake)
    app = _app(ui_loop, bridge)
    driver = _install(ui_loop, app)
    try:
        assert isinstance(driver, nw.NativeWebDriver) and app.webview_bridge is driver
        assert browser_driver.get_driver() is driver and wb.current() is driver and wb.availability() == (True, "")
        assert driver.capability == CAPABILITY and driver.max_pages == wb.MAX_PAGES and fake.info_calls == 1
        assert _wait_for(lambda: fake.methods("web_reset") == [None])  # orphans of an earlier session
        assert callable(driver.screen_churn) and wb.limitations() == wb.NATIVE_LIMITATIONS
        assert driver.on_web_event in bridge._listeners["web"]
    finally:
        driver.shutdown()
    assert browser_driver.get_driver() is None and wb.availability()[0] is False
    assert driver.on_web_event not in bridge._listeners["web"]
    assert _wait_for(lambda: len(fake.methods("web_reset")) == 2) and wb.limitations() == wb.LIMITATIONS


@needs_flet  # the fallback builds the flet-webview host
@pytest.mark.parametrize("case", ["no_capability", "stub", "switched_off", "handshake_raises", "handshake_hangs",
                                  "old_extension", "version_0"])
def test_install_falls_back_to_flet_webview(ui_loop, native_env, ext_capability, case):
    fake = FakeNative(NvidiaSite, capability=None if case == "no_capability" else CAPABILITY)
    native: Any = NativeBridge(native=fake)
    if case == "stub":
        native = NativeBridge(native=NativeStub("host test"))
    elif case == "switched_off":
        native_env.setenv(nw.ENV_SWITCH, "0")
    elif case == "handshake_raises":
        fake.info_error = RuntimeError("TimeoutException: get_platform_info")
    elif case == "handshake_hangs":
        fake.info_hang = True
        native_env.setattr(nw, "CAPABILITY_TIMEOUT", 0.05)
    elif case == "old_extension":
        _hide_ext_web_pages(native_env)
    elif case == "version_0":
        fake.capability = {"version": 0}
    native_env.setattr(wb, "platform_supported", lambda page: True)
    app = _app(ui_loop, native)
    driver = _install(ui_loop, app)
    try:
        assert isinstance(driver, wb.WebViewBridge) and app.webview_bridge is driver
        assert browser_driver.get_driver() is driver and wb.limitations() == wb.LIMITATIONS
        assert fake.methods("web_reset") == [] and fake.methods("web_open") == []
        assert fake.info_calls == (0 if case in ("stub", "switched_off") else 1)
    finally:
        driver.shutdown()


def test_install_without_any_in_app_browser_reports_why(ui_loop, native_env, ext_capability):
    app = _app(ui_loop, NativeBridge(native=FakeNative(NvidiaSite, capability=None)))
    native_env.setattr(wb, "platform_supported", lambda page: False)
    assert _install(ui_loop, app) is None and app.webview_bridge is None
    assert wb.availability() == (False, wb.UNSUPPORTED_REASON)


def test_capability_reads_the_extension_contract(native_env, ext_capability):
    assert nw.capability({"native": True, "web_pages": CAPABILITY}) == CAPABILITY
    for info in ({}, {"native": False, "web_pages": CAPABILITY}, {"native": True}, {"native": True, "web_pages": 1},
                 {"native": True, "web_pages": {"version": 0}}, None, "x"):
        assert nw.capability(info) is None, info
    _hide_ext_web_pages(native_env)
    assert nw.capability({"native": True, "web_pages": CAPABILITY}) is None  # an extension from before hotfix 3


def test_jobs_and_the_one_job_queue_need_nothing_from_the_native_driver(harness):
    """``BackgroundExecution`` tells the in-app browser when a job starts and when the queue drains
    (hotfix 1 parks / prewarms flet WebViews then); the native driver keeps nothing for jobs and its
    pages keep working across one job after another."""
    from glossarion_mobile.services.background import BackgroundExecution
    from glossarion_mobile.services.jobs import JobSnapshot, JobSpec, JobState

    h = harness(NvidiaSite)

    class Native:
        async def call(self, method, *args, default=None, **kwargs):
            return {"start_job_service": True, "begin_background_task": 3}.get(method, default)

    queue: list = []
    jobs = types.SimpleNamespace(view=lambda: types.SimpleNamespace(queue=tuple(queue), paused=False), snapshot=lambda: None)
    background = BackgroundExecution(Native(), platform="android", jobs=jobs, permissions=types.SimpleNamespace())

    def snap(state, job_id):
        return JobSnapshot(id=job_id, spec=JobSpec(kind="translate", title="Book", params={"model": "authchatgpt/auto"}),
                           state=state, created=0.0)

    async def one_job(job_id, more_queued):
        await background.job_started(snap(JobState.STARTING, job_id))
        if more_queued:
            queue.append("next")
        await background.job_finished(snap(JobState.DONE, job_id))
        queue.clear()

    before = list(h.fake.calls)
    for index in range(2):
        asyncio.run(one_job(f"j{index}", more_queued=index == 0))
        thread, box = _in_worker(authnd_auth.get_captcha_token, PAGE_URL, 60)
        thread.join(20)
        assert box.get("result") == "wv-token", box.get("error")
    background.close()
    opened = [method for method, _ in h.fake.calls[len(before):] if method in ("web_open", "web_reset")]
    assert opened == ["web_open", "web_open"]  # one page per request, nothing prepared or reset for jobs
    wb.jobs_running(True, snap(JobState.RUNNING, "j9"))
    wb.jobs_running(False)
    assert h.driver.status()["registered"] and _wait_for(lambda: not h.fake.pages)


def test_app_lifecycle_keeps_the_pages_until_detach(harness):
    from glossarion_mobile.app import GlossarionApp

    h = harness(lambda: NvidiaSite(block=True))
    page = _open_loaded(h)
    fake_app = types.SimpleNamespace(webview_bridge=h.driver, lifecycle=[], spike=None, shell=None)
    for name in ("inactive", "hide", "pause", "resume"):
        h.run(GlossarionApp.on_lifecycle, fake_app, types.SimpleNamespace(state=types.SimpleNamespace(value=name)))
    assert h.driver.registered and page.run_js("1", 100) is None and not page.renderer_gone
    assert h.fake.methods("web_close") == [] and h.fake.pages
    h.run(GlossarionApp.on_lifecycle, fake_app, types.SimpleNamespace(state=types.SimpleNamespace(value="detach")))
    assert not h.driver.registered and browser_driver.get_driver() is None
    with pytest.raises(browser_driver.BrowserCancelled, match="closing"):
        page.run_js("1")
    assert _wait_for(lambda: h.fake.methods("web_reset") and not h.fake.pages)
    page.close()


# ============================================================================ the on-device self-test
async def _churn_report(**overrides):
    report = {"before": "/", "pushed": "/settings", "after": "/", "restored": False, "tablet": False,
              "views_before": 1, "views_pushed": 2, "sheet": True}
    report.update(overrides)
    return report


def _selftest_harness(harness, *, churn=None, site=SelfTestSite):
    h = harness(site)
    sites = []
    original = h.fake.site_factory

    def factory():
        sites.append(original())
        return sites[-1]

    h.fake.site_factory = factory

    async def default_churn():
        return await _churn_report()

    h.driver.screen_churn = churn or default_churn
    return h, sites


def test_selftest_check_passes_on_a_native_page(harness):
    h, sites = _selftest_harness(harness)
    detail = webview_check.run(h.driver)
    assert detail["driver"] == nw.NATIVE_NAME and detail["capability"] == CAPABILITY
    assert detail["title"] == webview_check.TITLE and detail["token_survived"] and detail["typed"]
    assert detail["big_result"] == webview_check.BIG_SIZE and detail["native_closed"] and detail["generation"] == 1
    assert detail["churn"]["proved_view_change"] is True
    (native,) = h.fake.created
    assert (native.width, native.height) == webview_check.VIEWPORT and native.loads[0].startswith("http://127.0.0.1:")
    assert native.id in h.fake.methods("web_close") and h.fake.methods("web_status")
    assert h.driver.open_pages == 0 and h.driver.pages(webview_check.OWNER) == []


@pytest.mark.parametrize("failure", ["lossy", "reloaded", "title", "no_push", "no_status"])
def test_selftest_check_fails_with_the_reason(harness, failure):
    holder: dict = {}

    async def churn():
        if failure == "lossy":
            holder["sites"][0].lose = True
        if failure == "reloaded":  # what a rebuilt flet WebView did: a new document
            native = holder["h"].fake.created[0]
            await holder["h"].fake.web_load(native.id, native.url)
        return await _churn_report(views_pushed=1 if failure == "no_push" else 2)

    class OtherTitle(SelfTestSite):
        title = "Some other page"

    h, sites = _selftest_harness(harness, churn=churn, site=OtherTitle if failure == "title" else SelfTestSite)
    holder.update(h=h, sites=sites)
    if failure == "no_status":
        async def silent():
            return {"ok": False, "error": "unavailable"}

        h.fake.web_status = silent
    expected = {"lossy": "lost its state across a screen change", "reloaded": r"generation 1 -> 2",
                "title": "reports the title", "no_push": "screen change did not happen",
                "no_status": "no page status"}[failure]
    with pytest.raises(AssertionError, match=expected):
        webview_check.run(h.driver)
    assert _wait_for(lambda: h.driver.open_pages == 0 and not h.fake.pages)  # closed in every case


def test_selftest_check_proves_less_on_a_tablet_but_passes(harness):
    async def churn():
        return await _churn_report(tablet=True, views_pushed=1)

    h, _sites = _selftest_harness(harness, churn=churn)
    assert webview_check.run(h.driver)["churn"]["proved_view_change"] is False


@needs_flet
@pytest.mark.parametrize("welcome", [False, True])
def test_screen_churn_pushes_pops_and_shows_a_sheet(ui_loop, welcome):
    app = FakeApp(welcome=welcome)
    report = asyncio.run_coroutine_threadsafe(webview_check.screen_churn(app, settle=0), ui_loop.loop).result(10)
    assert report["before"] == "/" and report["pushed"] == "/settings" and report["sheet"]
    assert (report["views_before"], report["views_pushed"]) == (1, 2)
    assert report["after"] == "/" and report["restored"] is welcome and app.page.views == ["/"]
    assert app.routes == (["/settings", "/"] if welcome else ["/settings"])
    (dialog,) = app.page.dialogs
    assert dialog.open is False


def test_smoke_suite_registers_the_check_after_thread_stack():
    names = [name for name, _ in selftest.SUITES["smoke"]]
    assert names[names.index("thread_stack") + 1] == "native_webview"
    assert dict(selftest.SUITES["smoke"])["native_webview"] is selftest.check_native_webview


def _ctx(*, strict, device=True, platform="android"):
    paths = types.SimpleNamespace(is_device=device, platform=platform if device else "desktop")
    return types.SimpleNamespace(strict=strict, paths=paths)


def test_check_native_webview_skips_off_device(native_env):
    for ctx in (types.SimpleNamespace(strict=True, paths=None), _ctx(strict=True, device=False)):
        with pytest.raises(selftest.CheckSkipped, match="devices only"):
            selftest.check_native_webview(ctx)
    native_env.setenv("GLOSSARION_SIMULATE_PLATFORM", "android")  # host smoke runs (strict, simulated)
    with pytest.raises(selftest.CheckSkipped, match="devices only"):
        selftest.check_native_webview(_ctx(strict=True))


def test_check_native_webview_on_a_device(harness, native_env, monkeypatch):
    monkeypatch.setattr(webview_check, "DRIVER_WAIT_SECONDS", 0.0)
    with pytest.raises(AssertionError, match="native in-app browser is not installed"):
        selftest.check_native_webview(_ctx(strict=True, platform="ios"))
    with pytest.raises(selftest.CheckSkipped, match="no native in-app browser"):
        selftest.check_native_webview(_ctx(strict=False))
    native_env.setenv(nw.ENV_SWITCH, "off")
    with pytest.raises(selftest.CheckSkipped, match="switched off"):
        selftest.check_native_webview(_ctx(strict=True))
    native_env.delenv(nw.ENV_SWITCH)
    h = harness(NvidiaSite)
    ran = []
    monkeypatch.setattr(webview_check, "run", lambda driver: ran.append(driver) or {"ok": True})
    assert selftest.check_native_webview(_ctx(strict=True)) == {"ok": True} and ran == [h.driver]
    h.driver.unregister()
    other = types.SimpleNamespace(name="flet-webview")
    browser_driver.register_driver(other)
    with pytest.raises(AssertionError, match="flet-webview in-app browser is installed instead"):
        selftest.check_native_webview(_ctx(strict=True))
    browser_driver.unregister_driver(other)


def test_selftest_check_waits_for_a_page_slot_only_so_long(harness, monkeypatch):
    """Browser-route requests holding every page slot blocked "Run self-test" (and the self-test
    lock) until one of them closed its page. The check now waits at most SLOT_WAIT_SECONDS for a
    slot and says why; the deadline bounds only that wait, never the page the check then uses."""
    h, _sites = _selftest_harness(harness)
    monkeypatch.setattr(webview_check, "SLOT_WAIT_SECONDS", 0.0)
    held = [h.driver.open_page(owner="authchatgpt") for _ in range(h.driver.max_pages)]
    try:
        thread, box = _in_worker(webview_check.run, h.driver)
        thread.join(30)
        assert not thread.is_alive()
        assert isinstance(box.get("error"), webview_check.PagesBusy), box
        assert "in use by browser-route requests" in str(box["error"]) and h.fake.created == []
    finally:
        for page in held:
            page.close()
    detail = webview_check.run(h.driver)  # a free slot: the whole check runs past the 0 s deadline
    assert detail["token_survived"] and detail["typed"] and detail["native_closed"]
    assert h.driver.open_pages == 0


def test_check_native_webview_skips_while_route_pages_hold_every_slot(harness, native_env, monkeypatch):
    h = harness(NvidiaSite)

    def busy(driver):
        assert driver is h.driver
        raise webview_check.PagesBusy("every in-app page (4) stayed in use by browser-route requests")

    monkeypatch.setattr(webview_check, "run", busy)
    with pytest.raises(selftest.CheckSkipped, match="stayed in use by browser-route requests"):
        selftest.check_native_webview(_ctx(strict=True))


# ============================================================================ the eval wrapper
def _run_node(source):
    harness_js = "process.stdout.write(JSON.stringify(eval(" + json.dumps(source) + ")));\n"
    proc = subprocess.run([NODE, "-e", harness_js], capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


@needs_node
def test_eval_script_returns_one_json_string():
    for expression, value in (("({a: 1, b: [2, 'x']})", {"a": 1, "b": [2, "x"]}),
                              ("JSON.stringify({x: 1});", '{"x":1}'),
                              ("1 + 1 // a comment", 2),
                              ("undefined", None),
                              ("", None),
                              ("(() => 'marker')();\n", "marker")):
        text = _run_node(nw.eval_script(expression))
        assert isinstance(text, str) and nw._envelope_value(text) == value, expression
    assert json.loads(_run_node(nw.eval_script("(() => { throw new Error('nope'); })()"))) == {"ok": False, "error": "nope"}
    assert json.loads(_run_node(nw.eval_script("(() => { const o = {}; o.o = o; return o; })()")))["ok"] is False


@needs_node
def test_eval_script_wraps_every_route_script(tmp_path):
    scripts = [
        authnd_auth._captcha_injection_script("m:1:2", authnd_auth.DEFAULT_HCAPTCHA_SITEKEY, "__cb", 30000),
        authnd_auth.CAPTCHA_STATE_SCRIPT,
        gemini_free._ai_mode_set_prompt_script("User:\nhello \"quoted\" `tick` ${x}"),
        gemini_free.AI_MODE_CLICK_SEND_SCRIPT,
        gemini_free._page_snapshot_script("User:\nhello"),
        gpt._frame_script({"token": "t", "prompt": "p", "model": "auto", "images": 0, "temporary": True,
                           "imageOutput": False}),
        gpt._POLL, gpt._COMPOSER, gpt._TYPE, gpt._SEND, webview_check.TYPE_SCRIPT,
    ]
    for index, script in enumerate(scripts):
        path = tmp_path / f"script{index}.js"
        path.write_text(nw.eval_script(script), encoding="utf-8")
        proc = subprocess.run([NODE, "--check", str(path)], capture_output=True, text=True, timeout=60)
        assert proc.returncode == 0, (index, proc.stderr)


def test_envelope_decoding_is_strict():
    assert nw._envelope_value(json.dumps({"ok": True, "value": [1, "é"]})) == [1, "é"]
    for text in (None, "", "not json", "[1]", json.dumps({"ok": False, "error": "x"}), 5):
        assert nw._envelope_value(text) is None
    assert nw.eval_script("1;").startswith(nw.EVAL_HEAD) and nw.eval_script("1;").endswith(nw.EVAL_TAIL)
    # the extension's fakes cut the expression out on this marker (keep the wrapper text stable)
    assert nw.eval_script("6 * 7").split("var value = (\n", 1)[1].rsplit(
        "\n    );\n    envelope = {ok: true", 1)[0] == "6 * 7"


# ============================================================================ services/native.py
def test_native_bridge_carries_web_events_and_stub_answers():
    assert "web" in EVENTS
    stub = NativeStub("host", on_web=lambda e: None)
    assert stub.on_web is not None
    for method in native_mod.WEB_METHODS:
        assert asyncio.run(getattr(stub, method)("p")) == {"ok": False, "error": "unavailable", "message": "host"}
    bridge = NativeBridge(native=NativeStub("host"))
    seen = []
    bridge.add_listener("web", seen.append)
    asyncio.run(bridge.native.on_web(WebEvent(type="started", page_id="p1", url="https://x.example/")))
    assert seen and seen[0]["type"] == "started" and seen[0]["page_id"] == "p1" and seen[0]["event"] == "web"
    assert "control" not in seen[0]
    assert bridge.remove_listener("web", seen.append) is True and bridge.remove_listener("web", seen.append) is False
    asyncio.run(bridge.native.on_web(WebEvent(type="finished", page_id="p1")))
    assert len(seen) == 1


def test_create_native_passes_only_the_handlers_the_extension_knows(monkeypatch):
    @dataclasses.dataclass
    class OldGlossarionNative:  # an extension from before hotfix 3: no on_web field
        on_share: Any = None
        on_document: Any = None

    module = types.SimpleNamespace(GlossarionNative=OldGlossarionNative)
    monkeypatch.setattr(native_mod, "load_extension", lambda: module)
    monkeypatch.delenv("GLOSSARION_FORCE_NATIVE_STUB", raising=False)
    page = types.SimpleNamespace(platform=types.SimpleNamespace(value="android"), web=False)
    handlers = {f"on_{name}": (lambda e: None) for name in EVENTS}
    native = native_mod.create_native(page, handlers)
    assert isinstance(native, OldGlossarionNative) and native.on_share is handlers["on_share"]


def test_contract_names_match_the_extension():
    try:
        from flet_glossarion_native import web_pages
    except Exception:
        pytest.skip("the extension's web_pages module is not in this checkout yet")
    codes = {getattr(item, "value", item) for item in web_pages.WebPageError}
    assert {nw._GONE, nw._NO_PAGE, nw._EXISTS, nw._TIMEOUT, nw._NATIVE_ERROR} <= codes
    assert tuple(web_pages.WEB_METHODS) == native_mod.WEB_METHODS
    assert web_pages.WEB_PAGES_VERSION == 1 and web_pages.NATIVE_MAX_PAGES >= wb.MAX_PAGES
    fields = {f.name for f in dataclasses.fields(WebEvent)}
    from flet_glossarion_native.types import WebPageEvent

    assert {f.name for f in dataclasses.fields(WebPageEvent)} == fields


# ============================================================================ the extension's own fake
def _ext_fake_module():
    try:
        from flet_glossarion_native import web_pages_fake
    except Exception:
        return None
    return web_pages_fake


needs_ext_fake = pytest.mark.skipif(_ext_fake_module() is None,
                                    reason="the extension's FakeWebPagesNative is not importable here")


@pytest.fixture
def ext_driver(ui_loop, native_env):
    """The native driver over ``flet_glossarion_native.web_pages_fake.FakeWebPagesNative`` (the native
    builder's model of HiddenWebPages): the same driver code against the contract's own fake."""
    module = _ext_fake_module()
    made = []

    def make(responder, **kwargs):
        fake = module.FakeWebPagesNative(responder, **kwargs)
        driver = nw.NativeWebDriver(ui_loop.dispatcher, NativeBridge(native=fake),
                                    capability=module.DEFAULT_CAPABILITY).register()
        made.append((fake, driver))
        return fake, driver

    yield make
    for fake, driver in made:
        resets = fake.resets
        driver.shutdown()
        assert _wait_for(lambda f=fake: f.resets > resets and not f.pages)


@needs_ext_fake
def test_the_extension_fake_runs_authnd_and_crashes_like_the_platform(ext_driver):
    sites: dict = {}
    fake, driver = ext_driver(lambda page_id, expression: sites.setdefault(page_id, NvidiaSite()).answer(expression),
                              title_of=lambda url: "NVIDIA NIM")
    thread, box = _in_worker(authnd_auth.get_captcha_token, PAGE_URL, 60)
    thread.join(20)
    assert box.get("result") == "wv-token", box.get("error")
    (opened,) = fake.calls_of("web_open")
    assert opened["prevent_links"] == list(wb.PREVENT_LINKS) and (opened["width"], opened["height"]) == wb.DEFAULT_VIEWPORT
    assert _wait_for(lambda: fake.closed_ids == [opened["page_id"]] and not fake.pages)
    # a renderer crash while a script is held: None (not a Stop), renderer_gone, hook_lost
    page = _open_loaded(types.SimpleNamespace(driver=driver))
    assert page.title() == "NVIDIA NIM"
    fake.hold_evals(page.id)
    thread, box = _in_worker(page.run_js, "document.title", 30000)
    assert fake.wait_held(1)
    fake.crash(page.id).result(5)
    thread.join(5)
    assert not thread.is_alive() and "error" not in box and box.get("result") is None
    assert _wait_for(lambda: page.crashed is True) and page.renderer_gone
    assert gpt._DriverPage(page).hook_lost() is True and page.run_js("1") is None
    with pytest.raises(RuntimeError, match="lost"):
        page.load(PAGE_URL)
    page.close()
    assert _wait_for(lambda: page.id in fake.closed_ids)


@needs_ext_fake
def test_an_ios_renderer_loss_is_reported_as_unknown_not_as_a_crash(ext_driver, monkeypatch):
    """iOS: ``webViewWebContentProcessDidTerminate`` gives no reason (a crash, or iOS reclaiming a
    suspended app), so HiddenWebPages.swift sends ``crashed`` null: the page is lost, the reason unknown."""
    lines = []
    monkeypatch.setattr(nw.log, "info", lambda message, *args: lines.append(message % args))
    fake, driver = ext_driver(lambda page_id, expression: None, title_of=lambda url: "NVIDIA NIM")
    page = _open_loaded(types.SimpleNamespace(driver=driver))
    assert fake.crash(page.id, crashed=None).result(5) is True
    assert fake.wait_events(lambda e: e["type"] == "gone" and e["page_id"] == page.id and e["crashed"] is None)
    assert _wait_for(lambda: page.renderer_gone and any("killed or unknown" in line for line in lines))
    assert page.crashed is None and page.load_state["last_event"] == "renderer gone"
    assert not any("(crashed)" in line for line in lines)
    page.close()
    assert _wait_for(lambda: page.id in fake.closed_ids)


@needs_ext_fake
@pytest.mark.parametrize("reload_in_churn", [False, True])
def test_the_extension_fake_runs_the_self_test(ext_driver, reload_in_churn):
    """The on-device check against the contract's fake; a page that reloads during the screen change
    (what a rebuilt flet-webview did) fails it."""
    set_head = _clean(webview_check.SET_TOKEN_SCRIPT).split("{token}")[0]
    holder: dict = {}

    def responder(page_id, expression):
        state = holder["fake"].page(page_id).state
        if expression.startswith(set_head):
            state["token"] = json.loads(re.search(r"= (\"[0-9a-f]+\")", expression).group(1))
            return state["token"]
        answers = {webview_check.GET_TOKEN_SCRIPT: lambda: state.get("token"), webview_check.MATH_SCRIPT: lambda: 42,
                   _clean(webview_check.TYPE_SCRIPT): lambda: "ok",
                   webview_check.BIG_SCRIPT: lambda: "x" * webview_check.BIG_SIZE}
        return answers.get(expression, lambda: None)()

    def title_of(url):
        with urllib.request.urlopen(url, timeout=10) as response:  # the real ReaderServer on 127.0.0.1
            return re.search(r"<title>(.*?)</title>", response.read().decode("utf-8")).group(1)

    fake, driver = ext_driver(responder, title_of=title_of)
    holder["fake"] = fake

    async def churn():
        if reload_in_churn:
            (page_id,) = list(fake.pages)
            fake.navigate(page_id, fake.page(page_id).url)
        return await _churn_report()

    driver.screen_churn = churn
    if reload_in_churn:
        with pytest.raises(AssertionError, match="lost its state across a screen change"):
            webview_check.run(driver)
    else:
        detail = webview_check.run(driver)
        assert detail["token_survived"] and detail["big_result"] == webview_check.BIG_SIZE and detail["typed"]
    assert _wait_for(lambda: not fake.pages and driver.open_pages == 0)


# ============================================================================ hygiene
EDITED = [
    APP_DIR / "glossarion_mobile" / "services" / "native_webview.py",
    APP_DIR / "glossarion_mobile" / "services" / "webview_bridge.py",
    APP_DIR / "glossarion_mobile" / "services" / "native.py",
    APP_DIR / "glossarion_mobile" / "services" / "background.py",
    APP_DIR / "glossarion_mobile" / "diagnostics" / "webview_check.py",
    APP_DIR / "glossarion_mobile" / "diagnostics" / "selftest.py",
    APP_DIR / "glossarion_mobile" / "app.py",
    SRC_DIR / "browser_driver.py",
    SRC_DIR / "authchatgpt_auth.py",
    Path(__file__),
]


@pytest.mark.parametrize("path", EDITED, ids=lambda p: p.name)
def test_files_keep_uniform_line_endings_and_python_310_syntax(path):
    data = path.read_bytes()
    assert data.count(b"\r\n") in (0, data.count(b"\n")), path
    ast.parse(data.decode("utf-8"), feature_version=(3, 10))


def test_nothing_logs_scripts_results_or_titles():
    """Scripts carry prompts, results carry answers: no log call of the native modules takes one."""
    forbidden = {"script", "source", "expression", "text", "result", "answer", "title", "value", "payload", "url"}
    for path in (APP_DIR / "glossarion_mobile" / "services" / "native_webview.py",
                 APP_DIR / "glossarion_mobile" / "diagnostics" / "webview_check.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                 and node.func.value.id == "log"]
        assert calls or path.name == "webview_check.py"
        for call in calls:
            names = {n.id for arg in call.args[1:] for n in ast.walk(arg) if isinstance(n, ast.Name)}
            assert not names & forbidden, (path.name, call.lineno, names & forbidden)
