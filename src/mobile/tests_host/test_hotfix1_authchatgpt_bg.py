"""Hotfix 1, issue 3: authchatgpt/ (and the other in-app browser routes) keep working while the
app is in the background.

Root cause. Flutter builds widgets only in frames and draws no frames while the app is hidden
or paused (``SchedulerBinding.scheduleFrame`` returns at once unless the lifecycle state is
resumed or inactive). flet-webview 1.0.3 creates its native WebView, starts the first load and
subscribes to ``load_request`` / ``run_javascript`` in ``initState``
(``webview_mobile_and_mac.dart``), and Flet's ``Control.invokeMethod`` waits 10 s for that
listener ("Timeout waiting for invoke method listener"). Every authchatgpt/ request opened a
new hidden page, so in the background the new WebView was never built: its calls timed out,
the page waits (90 s each) ran out and the translation stood still until the app came back.
A WebView built while the app was on screen keeps working (method calls and events go over
the Flet connection); Android's WebView is never paused by the plugins (no onPause /
pauseTimers; RENDERER_PRIORITY_IMPORTANT, not waived, is AwContents' default).

The fix (``services/webview_bridge.py``): while jobs run, used WebViews are parked on
``about:blank`` and taken over by the next page; a job on a browser route gets one built at its
start (``BackgroundExecution.job_started`` -> ``webview_bridge.jobs_running``); a page that
needs a WebView while the app is hidden and finds none never builds one, it waits (Stop ends
the wait) for a parked one or for the app.

The fake WebView models that Flutter rule: it records whether the app was visible when it was
created (the bridge must never create one while hidden) and a WebView that lost its Flutter
State answers no method call. Real data stays isolated (HOME / USERPROFILE -> tmp_path).
Run from src/mobile:
    python -m pytest -p no:cacheprovider -W ignore tests_host/test_hotfix1_authchatgpt_bg.py
"""

from __future__ import annotations

import ast
import asyncio
import json
import re
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

import authchatgpt_auth as gpt  # noqa: E402
import authnd_auth  # noqa: E402
import browser_driver  # noqa: E402
import gemini_free  # noqa: E402
from glossarion_mobile.services import webview_bridge as wb  # noqa: E402
from glossarion_mobile.services.dispatcher import UiDispatcher  # noqa: E402
from glossarion_mobile.state.store import LoopGuard  # noqa: E402

CHATGPT = gpt.CHAT_URL
NIM_URL = "https://build.nvidia.com/z-ai/glm-5.1"
NO_LISTENER = ("TimeoutException after 0:00:10.000000: Timeout waiting for invoke method listener for "
               "WebView(7655).{}")
REPLY = ('data: {"v": {"message": {"author": {"role": "assistant"}, "recipient": "all", "channel": null,'
         ' "content": {"content_type": "text", "parts": ["Title"]}, "metadata": {"model_slug": "gpt-x"}}}}\n\n'
         'data: {"type": "message_stream_complete"}\n\n')
MESSAGES = [{"role": "user", "content": "Translate"}]


# ============================================================================ fakes
class FlutterHost:
    """``HiddenWebViewHost`` with the Flet page's ``app_visible`` (False while hidden / paused)."""

    def __init__(self):
        self.page = types.SimpleNamespace(app_visible=True)
        self.entries = []
        self.added = 0
        self.resized = []

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

    def resize(self, entry, size):
        entry.size = size
        self.resized.append(size)
        return True

    def hide(self):
        self.page.app_visible = False

    def show(self):
        self.page.app_visible = True


class ChatGPTSite:
    """chatgpt.com as the logged-out authchatgpt/ flow meets it (one document per navigation)."""

    def __init__(self):
        self.url = ""
        self.mark = None
        self.hooked = False
        self.requested = False
        self.polled = False
        self.prompts = 0

    def navigated(self, url):
        self.url, self.mark, self.hooked, self.requested, self.polled = url, None, False, False, False

    def answer(self, expression):
        on_chatgpt = self.url.startswith(CHATGPT)
        if expression.startswith("(window.__glossarionNav = '"):
            self.mark = expression.split("'")[1]
            return True
        if "window.__glossarionNav !== '" in expression:
            return on_chatgpt and self.mark != expression.split("window.__glossarionNav !== '")[1].split("'")[0]
        if expression == "document.cookie.indexOf('__Secure-next-auth.session-token=') >= 0":
            return True
        if expression == "!!document.querySelector('#prompt-textarea')":
            return on_chatgpt
        if expression.startswith("(((window) => {"):  # the hook, added once the app started
            self.hooked = on_chatgpt
            return True
        if expression == gpt._COMPOSER:
            return {"text": ".", "send": True, "requested": self.requested}
        if expression == gpt._SEND:
            self.requested = self.hooked
            self.prompts += 1
            return True
        if expression == gpt._POLL:
            if not self.hooked:
                return None
            chunks, self.polled = ([] if self.polled else [REPLY]), True
            return {"chunks": chunks, "streams": 1, "open": 0, "idle": 0, "status": 200, "error": None,
                    "kind": None, "sentinel": 0, "resolved": None, "final": None, "finalImages": None}
        return None


class NvidiaSite:
    """build.nvidia.com for the authnd/ hCaptcha token flow (as in test_webview_bridge)."""

    def __init__(self):
        self.marker = ""

    def navigated(self, url):
        pass

    def answer(self, expression):
        if "const marker =" in expression:
            self.marker = json.loads(re.search(r"const marker = (\".*?\");", expression).group(1))
            return self.marker
        if "result: window.__authndResult" in expression:
            return json.dumps({"marker": self.marker, "readyState": "complete", "result": {
                "marker": self.marker, "pending": False, "step": "complete", "token": "wv-token", "error": None}})
        return None


class FlutterWebView:
    """``flet_webview.WebView`` under Flutter's rule: created while the app is visible it is built at
    once (initState: first load + method listener); ``created_visible`` False means the bridge mounted
    one in the background, which Flutter would not build until the app shows again. ``deaf``: its
    Flutter State is gone, so method calls find no listener (Flet's 10 s timeout, shortened)."""

    def __init__(self, host, slot, url, site):
        self.host, self.page, self.site = host, slot, site
        self.loads, self.sources = [], []
        self.created_visible = bool(host.page.app_visible)
        self.deaf = not self.created_visible
        self.url = url
        if not self.deaf:
            self._navigate(url)

    def _navigate(self, url):
        self.loads.append(url)
        self.site.navigated(url)
        loop = asyncio.get_running_loop()
        loop.call_soon(self.page.on_page_started, types.SimpleNamespace(data=url))
        loop.call_later(0.01, self.page.on_page_ended, types.SimpleNamespace(data=url))

    async def _no_listener(self, method):
        await asyncio.sleep(0.02)
        raise RuntimeError(NO_LISTENER.format(method))

    async def load_request(self, url):
        if self.deaf:
            await self._no_listener("load_request")
        self._navigate(url)

    async def get_title(self):
        return "Site"

    async def run_javascript(self, source):
        if self.deaf:
            await self._no_listener("run_javascript")
        self.sources.append(source)
        request_id = json.loads(re.search(r"var id = (\"[0-9a-f]+\")", source).group(1))
        expression = source.split("var value = (\n", 1)[1].rsplit("\n    );\n    envelope", 1)[0]
        text = json.dumps({"id": request_id, "ok": True, "value": self.site.answer(expression)})
        asyncio.get_running_loop().call_soon(
            self.page.on_console, types.SimpleNamespace(message="app.js:1 " + wb.CONSOLE_PREFIX + text))


# ============================================================================ fixtures
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
    for name in ("AUTHND_TOKEN_MODE", "GEMINI_FREE_MODE", "AUTHCHATGPT_MODE", "TRANSLATION_CANCELLED"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GLOSSARION_MOBILE", "1")
    monkeypatch.setenv("GLOSSARION_NO_PROCESSES", "1")
    monkeypatch.setenv("GLOSSARION_HTTP_LOG", "0")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setattr(authnd_auth, "CAPTCHA_DOCUMENT_STABLE_SECONDS", 0.05)
    monkeypatch.setattr(gpt, "_access_token", lambda log_fn: None)  # logged out: no token store

    def no_popen(*args, **kwargs):
        raise AssertionError("a helper subprocess was started on mobile")

    monkeypatch.setattr(subprocess, "Popen", no_popen)
    browser_driver.unregister_driver()
    for module in (authnd_auth, gemini_free, gpt):
        module.reset_cancel()
    yield monkeypatch
    browser_driver.unregister_driver()
    wb._set_current(None)
    for module in (authnd_auth, gemini_free, gpt):
        module.reset_cancel()


def _bridge(ui_loop, site_factory=ChatGPTSite, **kwargs):
    host = FlutterHost()
    views = []

    def factory(slot, url):
        view = FlutterWebView(host, slot, url, site_factory())
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


def _wait_for(predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def _snap(model=None):
    return types.SimpleNamespace(spec=types.SimpleNamespace(params={"model": model} if model else {}))


def _chat(logs):
    return _in_worker(gpt.send_chat_completion, messages=MESSAGES, model="authchatgpt/auto", timeout=60,
                      log_fn=logs.append)


# ============================================================================ the owner's report
def test_authchatgpt_keeps_translating_while_the_app_is_hidden(ui_loop, mobile_env):
    """A job on authchatgpt/ gets its WebView built while the app is on screen; with the app
    hidden, request after request runs on that WebView and none is ever built in the background."""
    bridge, host, views = _bridge(ui_loop)
    wb.jobs_running(True, _snap("authchatgpt/gpt-6-instant"))  # BackgroundExecution.job_started
    assert _wait_for(lambda: bridge.status()["spares"] == 1)
    assert bridge.status()["warm"] and bridge.prewarmed == 1
    host.hide()
    logs = []
    for _ in range(3):
        thread, box = _chat(logs)
        thread.join(30)
        assert not thread.is_alive()
        assert (box.get("result") or {}).get("content") == "Title", box.get("error")
        assert _wait_for(lambda: bridge.status()["spares"] == 1)  # parked again for the next request
    (view,) = views
    assert view.created_visible and host.added == 1
    assert view.loads == [wb.BLANK_URL] + [CHATGPT, wb.BLANK_URL] * 3
    assert bridge.hidden_waits == 0 and bridge.open_pages == 0
    assert not any("again" in line for line in logs)  # no page was lost / resent
    wb.jobs_running(False)  # the queue drained: the parked WebView goes
    assert _wait_for(lambda: not host.entries) and bridge.status()["spares"] == 0


def test_a_request_without_a_built_webview_waits_for_the_app_instead_of_stalling(ui_loop, mobile_env):
    """Hidden with nothing parked (a job whose model is not a browser route, then a request on
    one): no WebView is mounted in the background; the request waits and runs once the app shows."""
    bridge, host, views = _bridge(ui_loop)
    wb.jobs_running(True, _snap("gpt-5"))
    time.sleep(0.2)
    assert views == [] and bridge.prewarmed == 0  # no warm-up for a non-browser model
    host.hide()
    thread, box = _chat([])
    assert _wait_for(lambda: bridge.hidden_waits == 1)
    time.sleep(0.4)
    assert thread.is_alive() and views == [] and host.entries == []
    host.show()
    thread.join(30)
    assert (box.get("result") or {}).get("content") == "Title", box.get("error")
    assert _wait_for(lambda: bridge.status()["spares"] == 1)  # kept for the next request of the job
    (view,) = views
    assert view.created_visible and view.loads == [CHATGPT, wb.BLANK_URL]
    # the next request in the background takes it over
    host.hide()
    thread, box = _chat([])
    thread.join(30)
    assert (box.get("result") or {}).get("content") == "Title", box.get("error")
    assert len(views) == 1 and bridge.hidden_waits == 1


def test_stop_ends_a_background_wait_at_once(ui_loop, mobile_env):
    bridge, host, views = _bridge(ui_loop)
    host.hide()
    thread, box = _chat([])
    assert _wait_for(lambda: bridge.hidden_waits == 1)
    started = time.monotonic()
    gpt.cancel_stream()
    thread.join(5)
    assert not thread.is_alive() and time.monotonic() - started < 2
    error = box.get("error")
    assert isinstance(error, gpt.AuthChatGPTError) and error.error_type == "cancelled", error
    assert views == [] and _wait_for(lambda: bridge.open_pages == 0)


def test_other_browser_routes_share_the_parked_webview(ui_loop, mobile_env):
    """authnd/ (hCaptcha token) runs on the same bridge: a parked WebView serves it in the
    background too; a page with another viewport resizes the WebView it takes over."""
    bridge, host, views = _bridge(ui_loop, NvidiaSite)
    bridge.job_started(["authnd/z-ai/glm-5.1"])
    assert _wait_for(lambda: bridge.status()["spares"] == 1)
    host.hide()
    thread, box = _in_worker(authnd_auth.get_captcha_token, NIM_URL, 60)
    thread.join(20)
    assert box.get("result") == "wv-token", box.get("error")
    assert _wait_for(lambda: bridge.status()["spares"] == 1)
    page = bridge.open_page(owner="gemini_free", viewport=(1280, 900))
    page.load(NIM_URL)
    assert _wait_for(lambda: page.load_state["finished_generation"] == page.load_state["generation"] == 1)
    assert page.reused == 1 and page.builds == 0 and host.resized == [(1280, 900)]
    page.close()
    assert _wait_for(lambda: bridge.status()["spares"] == 1)
    (view,) = views
    assert view.created_visible and view.loads == [wb.BLANK_URL, NIM_URL, wb.BLANK_URL, NIM_URL, wb.BLANK_URL]
    bridge.keep_warm(False)
    assert _wait_for(lambda: not host.entries)


def test_a_parked_webview_that_lost_its_state_in_the_background_is_replaced_on_return(ui_loop, mobile_env):
    """A parked WebView that answers no listener while the app is hidden is dropped, never rebuilt
    in the background; the page waits and builds a new WebView once the app is visible."""
    bridge, host, views = _bridge(ui_loop, NvidiaSite)
    bridge.job_started(["search/gemini"])
    assert _wait_for(lambda: bridge.status()["spares"] == 1)
    host.hide()
    views[0].deaf = True
    page = bridge.open_page(owner="authnd")
    thread, box = _in_worker(page.load, NIM_URL)
    assert _wait_for(lambda: bridge.hidden_waits == 1 and not host.entries)
    assert thread.is_alive() and len(views) == 1
    host.show()
    thread.join(10)
    assert "error" not in box, box.get("error")
    assert len(views) == 2 and views[1].created_visible and views[1].loads == [NIM_URL] and page.builds == 1
    page.close()
    assert _wait_for(lambda: bridge.status()["spares"] == 1)
    bridge.shutdown()
    assert _wait_for(lambda: not host.entries)


def test_without_jobs_pages_keep_unmounting_after_use(ui_loop, mobile_env):
    """Outside jobs (Model Manager refresh, the desktop-like default) nothing is parked."""
    bridge, host, views = _bridge(ui_loop)
    thread, box = _chat([])
    thread.join(30)
    assert (box.get("result") or {}).get("content") == "Title", box.get("error")
    assert _wait_for(lambda: not host.entries)
    assert views[0].loads == [CHATGPT] and bridge.status()["spares"] == 0 and not bridge.status()["warm"]


def test_cancelled_pages_are_not_parked(ui_loop, mobile_env):
    bridge, host, views = _bridge(ui_loop, NvidiaSite)
    bridge.keep_warm(True)
    page = bridge.open_page(owner="authnd")
    page.load(NIM_URL)
    assert _wait_for(lambda: page.load_state["finished_generation"] == 1)
    browser_driver.cancel_pages("authnd")
    page.close()
    assert _wait_for(lambda: not host.entries) and bridge.status()["spares"] == 0


# ============================================================================ who warms up
def test_browser_route_detection_and_configured_models():
    for model in ("authchatgpt/auto", "AuthChatGPT/gpt-6-pro", "authnd/z-ai/glm-5.1", "authds/deepseek",
                  "search/gemini", "search/gemini-pro"):
        assert wb.uses_browser_route(model), model
    for model in ("authgpt/gpt-6-luna", "gpt-5", "search/opera", "", None, "authza/x"):
        assert not wb.uses_browser_route(model), model
    assert wb.job_models(_snap("authchatgpt/auto")) == ["authchatgpt/auto"]
    overrides = types.SimpleNamespace(spec=types.SimpleNamespace(params={"config_overrides": {"model": "authnd/x"}}))
    assert wb.job_models(overrides) == ["authnd/x"] and wb.job_models(None) == []
    store = {"model": "gpt-5", "multi_api_keys": [{"model": "authchatgpt/auto"}, "junk"],
             "fallback_keys": [{"model": "search/gemini"}, {}]}
    assert wb.configured_models(types.SimpleNamespace(get=store.get)) == ["gpt-5", "authchatgpt/auto", "search/gemini"]
    assert wb.configured_models(object()) == []


def test_configured_browser_model_warms_a_webview(ui_loop, mobile_env):
    bridge, host, views = _bridge(ui_loop)
    settings = {"model": "authchatgpt/gpt-6-instant"}
    bridge.config_models = lambda: wb.configured_models(types.SimpleNamespace(get=settings.get))
    wb.jobs_running(True, _snap())  # a book job: its model comes from the settings
    assert _wait_for(lambda: bridge.status()["spares"] == 1) and len(views) == 1
    wb.jobs_running(True, _snap())  # the next queued job: one WebView is enough
    time.sleep(0.2)
    assert len(views) == 1
    host.hide()
    wb.jobs_running(False)
    assert _wait_for(lambda: not host.entries)
    wb.jobs_running(True, _snap())  # a job that starts while the app is hidden builds nothing
    time.sleep(0.2)
    assert len(views) == 1 and not host.entries


def test_a_failed_warm_up_leaves_nothing_to_wait_for(ui_loop, mobile_env):
    bridge, host, views = _bridge(ui_loop)

    def broken(slot, url):
        raise RuntimeError("flet_webview is missing")

    bridge.webview_factory = broken
    bridge.job_started(["authchatgpt/auto"])
    assert _wait_for(lambda: bridge.prewarmed == 1)
    assert _wait_for(lambda: bridge.status()["parking"] == 0) and not host.entries
    page = bridge.open_page(owner="authchatgpt")
    started = time.monotonic()
    bridge.claim(page)  # visible, nothing parking: a page may build at once
    assert time.monotonic() - started < 1 and page._claimed is None
    page.close()


@pytest.mark.parametrize("platform", ["android", "ios"])
def test_background_execution_warms_while_jobs_run(platform, monkeypatch):
    from glossarion_mobile.services.background import BackgroundExecution
    from glossarion_mobile.services.jobs import JobSnapshot, JobSpec, JobState

    calls = []
    fake = types.SimpleNamespace(registered=True, job_started=lambda models: calls.append(("start", list(models))),
                                 keep_warm=lambda on: calls.append(("warm", on)))
    monkeypatch.setattr(wb, "current", lambda: fake)

    class Native:
        async def call(self, method, *args, default=None, **kwargs):
            return {"start_job_service": True, "begin_background_task": 3}.get(method, default)

    queue = []
    jobs = types.SimpleNamespace(view=lambda: types.SimpleNamespace(queue=tuple(queue), paused=False),
                                 snapshot=lambda: None)
    background = BackgroundExecution(Native(), platform=platform, jobs=jobs, permissions=types.SimpleNamespace())

    def snap(state, model="authchatgpt/auto"):
        return JobSnapshot(id="j1", spec=JobSpec(kind="translate", title="Book", params={"model": model}),
                           state=state, created=0.0)

    async def scenario():
        await background.job_started(snap(JobState.STARTING))
        queue.append("j2")
        await background.job_finished(snap(JobState.DONE))  # the next job inherits: still warm
        queue.clear()
        await background.job_finished(snap(JobState.DONE))  # the queue drained
        background.close()

    asyncio.run(scenario())
    assert calls == [("start", ["authchatgpt/auto"]), ("warm", False)]


# ============================================================================ hygiene
def test_hidden_host_resize_changes_only_the_frame():
    pytest.importorskip("flet")
    import flet as ft

    from glossarion_mobile.ui.components.hidden_webview import HiddenWebViewHost

    page = types.SimpleNamespace(overlay=[], updates=0)
    page.update = lambda: setattr(page, "updates", page.updates + 1)
    host = HiddenWebViewHost(page)
    entry = host.add(ft.Text("web view stand-in"), size=(480, 800), key="glwvb-1")
    assert host.resize(entry, (1280, 900)) is True and page.updates == 2
    frame = entry.content.content.controls[0]
    assert (frame.width, frame.height) == (1280, 900) and (entry.width, entry.height) == (1, 1)
    assert host.resize(entry, (1280, 900)) is False and host.resize(ft.Text("x"), (1, 1)) is False


def test_files_keep_uniform_line_endings_and_python_310_syntax():
    for path in (APP_DIR / "glossarion_mobile" / "services" / "webview_bridge.py",
                 APP_DIR / "glossarion_mobile" / "services" / "background.py",
                 APP_DIR / "glossarion_mobile" / "ui" / "components" / "hidden_webview.py",
                 Path(__file__)):
        data = path.read_bytes()
        assert data.count(b"\r\n") in (0, data.count(b"\n")), path
        ast.parse(data.decode("utf-8"), feature_version=(3, 10))
