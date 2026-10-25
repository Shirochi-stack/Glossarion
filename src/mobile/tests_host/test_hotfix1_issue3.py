"""Acceptance test for the owner's hotfix-1 issue 3: "Authchatgpt/ is reliant on the program being on
during translation. It doesn't work in the background like other provider".

What the owner saw on the phone: an authchatgpt/ (chatgpt.com web chat, the mobile default model) book
translation stood still as soon as Glossarion went to the background and only moved on once the app was
opened again. Every request opened a new hidden WebView, and Flutter builds no widget while the app is
hidden or paused (no frames), so the new WebView never got its native view, its first load or its method
listener: its calls ended with Flet's "Timeout waiting for invoke method listener" and the route's page
waits ran out. The fix (services/webview_bridge.py, services/background.py, ui/components/hidden_webview.py)
builds one WebView at the job's start while the app is on screen, parks used WebViews on about:blank
while jobs run and hands them to the next request; a request never mounts a new WebView in the background.

This file proves it end to end on the REAL app objects: the app started on a fake Flet session with the
Android platform (``test_ui_foundations._start`` through ``test_ui_flows._host_driver``), so the real
``WebViewBridgeFeature`` registers the real ``WebViewBridge`` with the real ``HiddenWebViewHost`` and the real
``flet_webview.WebView`` factory as the backend's ``browser_driver``; the real JobService, ChatFeature /
ChatRuns / ChatView, ``BackgroundExecution`` (``job_started`` -> ``webview_bridge.jobs_running``), the
shared translation pipeline and ``authchatgpt_auth`` (logged out, as on a fresh phone). The owner's taps go
through the UI (New chat, ＋ › Files, Send, Start, ✓ Yes) and Home / back to the app are Flet's
``app_lifecycle_state_change`` events dispatched through the session (``page.app_visible`` follows them,
exactly as on the phone).

What the fake session lacks is the phone's Flutter side of flet-webview. :class:`Phone` supplies it at
the protocol level: it intercepts the session's INVOKE_METHOD messages for WebView controls and sends
WebView events back through ``session.dispatch_event``, under Flutter's rule:

* a WebView control mounted while the app is visible is built at once (``initState``: native view, first
  load of its ``url``, the ``load_request`` / ``run_javascript`` listener);
* one mounted while the app is hidden is built only when the app shows again; until then its method
  calls end with Flet's "Timeout waiting for invoke method listener" (10 s on a phone, shortened here);
* a built WebView keeps working in the background (method calls and events travel over the Flet
  connection, not through frames).

Its documents are chatgpt.com as the logged-out authchatgpt/ flow drives it (the same expressions as
test_hotfix1_authchatgpt_bg's ChatGPTSite); the conversation itself is answered by the offline fake OpenAI
server on 127.0.0.1 (``diagnostics.fake_llm_server``: the prompt's ``[system]`` / ``[user]`` parts go to it
as chat messages), so its ``on_response`` hook drives the hold gate (``HoldGate`` from
test_devfix_issue12) and no step depends on the runner's speed. The E2E network guard refuses every
non-loopback connection for the whole test.

1. ``test_book_keeps_translating_with_the_app_in_the_background``: Balanced glossary off; the fake model
   holds after the third translation reply, so a chapter request is in flight on the hidden page when
   Home is pressed. Released in the background, every remaining chapter is sent and answered with the
   app hidden, the job reaches Done while still hidden, no WebView is ever mounted or built in the
   background, no call meets a missing listener, no request is sent twice, and back in the app the turn's
   card reads "Done · 6/6 chapters". Once the queue drained the parked WebView is unmounted.
2. ``test_glossary_gate_yes_then_home_translates_in_the_background``: the Balanced glossary runs on screen,
   the owner taps ✓ Yes on the approval card and goes Home at once: the whole translation runs in the
   background on the WebView built while the app was on screen.
3. ``test_start_then_home_at_once_runs_the_whole_job_in_the_background``: Start, then Home before the job's
   first request has a page: the WebView the job built at its start (``BackgroundExecution.job_started``
   -> ``WebViewBridge.prewarm``) serves the glossary in the background, the job waits at the gate; back in
   the app, ✓ Yes, Home again: the translation finishes in the background on that same WebView.

With the three issue-3 modules swapped back to their 01ae89d6 versions (a scratch import hook; not part
of this file) scenarios 1 and 2 fail with "the translation stood still in the background": the next
request mounts a new WebView while the app is hidden, it is never built, and its ``run_javascript``
calls keep meeting no listener - what the owner saw.

What only a device can prove (see the hotfix-1 report): that Android keeps the built WebView's renderer
and chatgpt.com's own page code (Cloudflare / Turnstile / sentinel, the streaming fetch) running while
the activity is stopped under the foreground service, and that the platform fires page events for
``about:blank``. iOS suspends WebKit's web content process in the background (documented limitation).

Real data stays untouched: the app runs in temp FLET_APP_STORAGE_* dirs (the bootstrap points HOME,
OUTPUT_DIRECTORY, GLOSSARION_LIBRARY_DIR, GLOSSARION_DATA_DIR and CONFIG_FILE there; asserted),
USERPROFILE / APPDATA / LOCALAPPDATA point into tmp_path and HTTP logging is off.

Run from src/mobile with the mobile env (on the Windows dev box ``.venv``, Python 3.12; keep its PYTHONPATH)::

    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_hotfix1_issue3.py
"""

from __future__ import annotations

import ast
import asyncio
import importlib.util
import json
import os
import re
import sys
import threading
import time
import types
import urllib.request
from pathlib import Path

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = MOBILE_DIR / "app"
SRC_DIR = MOBILE_DIR.parent
TESTS_DIR = MOBILE_DIR / "tests"
for entry in (str(APP_DIR), str(SRC_DIR), str(TESTS_DIR)):
    if entry not in sys.path:
        sys.path.insert(0, entry)


def _has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


_NEEDED = ("flet", "flet_webview", "msgpack", "ebooklib", "openai", "httpx", "tiktoken", "bs4", "lxml")
pytestmark = pytest.mark.skipif(not all(_has(m) for m in _NEEDED),
                                reason=f"needs {', '.join(_NEEDED)} (the project venv)")


def _load(alias: str, file_name: str):
    spec = importlib.util.spec_from_file_location(alias, Path(__file__).with_name(file_name))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# The device-report harness of test_devfix_issue12 (fixtures, the hold gate, the probe of the chat).
_D12 = _load("_glossarion_devfix12_hotfix1_issue3", "test_devfix_issue12.py")
_UF = _D12._UF
storage = _UF.storage
app_env = _UF.app_env
iso = _D12.iso
offline = _D12.offline
HoldGate = _D12.HoldGate
Probe = _D12.Probe
LEAVE = _D12.LEAVE
COME_BACK = _D12.COME_BACK
_until = _D12._until
_settled = _D12._settled
_finished = _D12._finished
_end_job = _D12._end_job
_assert_isolated = _D12._assert_isolated
_assert_offline = _D12._assert_offline

MODEL = "authchatgpt/auto"  # the mobile default model (setting_writes.MOBILE_DEFAULT_MODEL)
CHAPTERS = 6
EPUB_NAME = "background_novel.epub"
CHATGPT = "https://chatgpt.com/"
#: Flet's client waits this long for a WebView's method listener (10 s on a phone; shortened).
LISTENER_TIMEOUT = 1.0
#: Seconds a document takes to load in the fake WebView.
LOAD_SECONDS = 0.03
RUN_TIMEOUT = 240.0
#: Seconds the job may take with the app hidden (a few with the fix; the old code stood still).
HIDDEN_TIMEOUT = 60.0
#: Translation replies before the fake model holds (the run's EPUB metadata request, then chapters).
HOLD_AFTER = 3
NO_LISTENER = ("TimeoutException after 0:00:10.000000: Timeout waiting for invoke method listener for "
               "WebView({}).{}")
#: Job log lines that mean a request lost its page, waited for the app or was sent again.
BAD_LOG = ("sending the message again", "did not finish loading", "in-app browser could not",
           "invoke method listener", "no in-app page is ready", "page was reloaded")


# ==========================================================================
# The phone's Flutter side of flet-webview
# ==========================================================================


def _sse(content: str) -> str:
    """chatgpt.com's conversation stream for one visible assistant answer (``StreamParser``)."""
    message = {"v": {"message": {"author": {"role": "assistant"}, "recipient": "all", "channel": None,
                                 "content": {"content_type": "text", "parts": [content]},
                                 "metadata": {"model_slug": "gpt-fake-web"}}}}
    return "data: " + json.dumps(message) + "\n\n" + 'data: {"type": "message_stream_complete"}\n\n'


def _messages(prompt: str) -> list:
    """The chat messages of a web-chat prompt (``authds_auth.build_prompt``: ``[role]`` headers)."""
    parts = re.split(r"(?:^|\n\n)\[(system|user|assistant)\]\n", str(prompt or ""))
    if len(parts) < 3:
        return [{"role": "user", "content": str(prompt or "")}]
    return [{"role": role, "content": text} for role, text in zip(parts[1::2], parts[2::2])]


class ChatGPTDocument:
    """One document of a WebView (each navigation brings a new one): chatgpt.com as the logged-out
    authchatgpt/ flow drives it. The conversation is answered by the fake OpenAI server."""

    def __init__(self, phone: "Phone", view: "PhoneWebView", url: str) -> None:
        self.phone, self.view, self.url = phone, view, str(url or "")
        self.started = time.monotonic()
        self.mark = None
        self.hooked = False
        self.config: dict = {}
        self.requested = False
        self.streams = 0
        self.open = 0
        self.closed_at = 0.0
        self.error = None
        self._chunks: list = []
        self._lock = threading.Lock()

    @property
    def on_chatgpt(self) -> bool:
        return self.url.startswith(CHATGPT)

    @property
    def loaded(self) -> bool:
        return time.monotonic() - self.started >= LOAD_SECONDS

    def answer(self, expression: str):
        import authchatgpt_auth as gpt

        if expression.startswith("(window.__glossarionNav = '"):
            self.mark = expression.split("'")[1]
            return True
        if "window.__glossarionNav !== '" in expression:
            mark = expression.split("window.__glossarionNav !== '")[1].split("'")[0]
            return self.on_chatgpt and self.loaded and self.mark != mark
        if expression == "document.cookie.indexOf('__Secure-next-auth.session-token=') >= 0":
            return "session-token" in self.phone.cookies
        if expression.startswith("(document.cookie = '__Secure-next-auth.session-token="):
            self.phone.cookies.add("session-token")  # one cookie jar for every WebView of the app
            return True
        if expression == "!!document.querySelector('#prompt-textarea')":
            return self.on_chatgpt and self.loaded
        if expression.startswith("(((window) => {"):  # the hook, added once the app started
            self.hooked = self.on_chatgpt
            config = expression.split("const cfg = ", 1)[1]
            self.config = json.JSONDecoder().raw_decode(config)[0]
            return True
        if expression.startswith("(() => {const f = document.getElementById("):  # _DriverPage.hook_lost
            return not self.hooked
        if expression == gpt._COMPOSER:
            return {"text": ".", "send": True, "requested": self.requested}
        if expression == gpt._TYPE:
            return True
        if expression == gpt._SEND:
            if self.hooked and not self.requested:
                self._send()
            return True
        if expression == gpt._POLL:
            if not self.hooked:
                return None
            with self._lock:
                chunks, self._chunks = self._chunks, []
            idle = int((time.monotonic() - self.closed_at) * 1000) if self.closed_at else 0
            return {"chunks": chunks, "streams": self.streams, "open": self.open, "idle": idle, "status": 200,
                    "error": self.error, "kind": None, "sentinel": 0, "resolved": None, "final": None,
                    "finalImages": None}
        if expression == gpt._LOAD_CATALOG:
            return True
        if expression == gpt._CATALOG:  # the model catalog's auto-poll reads the picker here
            return {"catalog": {"names": {"gpt-6-instant": "GPT-6 Instant"}}, "error": None} if self.hooked else None
        return None

    def _send(self) -> None:
        """The page's conversation request (the hook swapped Glossarion's prompt in)."""
        self.requested = True
        self.streams, self.open = 1, 1
        prompt = str(self.config.get("prompt") or "")
        record = {"view": self.view.id, "sent_visible": self.phone.visible(), "answered_visible": None,
                  "chapters": sorted({int(n) for n in re.findall(r"제\s*(\d+)\s*장", prompt)}), "ok": False}
        self.phone.conversations.append(record)
        threading.Thread(target=self._converse, args=(prompt, record), name="chatgpt-conversation",
                         daemon=True).start()

    def _converse(self, prompt: str, record: dict) -> None:
        try:
            content = self.phone.backend(prompt)
            with self._lock:
                self._chunks.append(_sse(content))
            record["ok"] = True
        except Exception as exc:  # the request fails like a broken stream
            self.error = f"fake chatgpt.com backend failed: {exc}"
        finally:
            record["answered_visible"] = self.phone.visible()
            self.closed_at = time.monotonic()
            self.open = 0


class PhoneWebView:
    """The Flutter State + native WebView behind one mounted ``flet_webview.WebView`` control."""

    def __init__(self, phone: "Phone", control) -> None:
        self.phone, self.control = phone, control
        self.id = control._i
        self.mounted_visible = phone.visible()
        self.built = False
        self.built_visible = None
        self.disposed = False
        self.document = None
        self.loads: list = []  # (url, app visible)
        self.listening = asyncio.Event()

    def build(self) -> None:
        """``initState``: the native WebView, its listener and the first load of the control's ``url``."""
        if self.built or self.disposed:
            return
        self.built, self.built_visible = True, self.phone.visible()
        self.listening.set()
        self.navigate(str(self.control.url or ""))

    def dispose(self) -> None:
        self.disposed = True
        self.built = False
        self.listening.clear()

    def navigate(self, url: str) -> None:
        self.loads.append((url, self.phone.visible()))
        document = self.document = ChatGPTDocument(self.phone, self, url)
        self.phone.emit(self, "page_started", url)
        self.phone.emit(self, "url_change", url)
        loop = asyncio.get_running_loop()
        loop.call_later(LOAD_SECONDS, lambda: self.document is document and self.phone.emit(self, "page_ended", url))

    def run_javascript(self, source: str) -> None:
        request_id = json.loads(re.search(r"var id = (\"[0-9a-f]+\")", source).group(1))
        size = int(re.search(r"size = (\d+);", source).group(1))
        expression = source.split("var value = (\n", 1)[1].rsplit("\n    );\n    envelope", 1)[0]
        try:
            value = self.document.answer(expression) if self.document is not None else None
            envelope = {"id": request_id, "ok": True, "value": value}
        except Exception as exc:  # the page's script threw
            envelope = {"id": request_id, "ok": False, "error": str(exc)}
        text = json.dumps(envelope)
        from glossarion_mobile.services.webview_bridge import CONSOLE_PREFIX

        if len(text) <= size:
            lines = [text]
        else:
            parts = -(-len(text) // size)
            lines = [json.dumps({"id": request_id, "part": i, "parts": parts, "chunk": text[i * size:(i + 1) * size]})
                     for i in range(parts)]
        for line in lines:
            self.phone.emit(self, "console_message", {"message": CONSOLE_PREFIX + line, "severity_level": "log"})

    async def call(self, name: str, args: dict):
        if not self.built:
            # Flet's client waits for the State's method listener and fails without one.
            try:
                await asyncio.wait_for(self.listening.wait(), self.phone.listener_timeout)
            except asyncio.TimeoutError:
                self.phone.no_listener.append((self.id, name, self.phone.visible()))
                raise RuntimeError(NO_LISTENER.format(self.id, name)) from None
        if name == "load_request":
            self.navigate(str(args.get("url") or ""))
            return None
        if name == "run_javascript":
            self.run_javascript(str(args.get("value") or ""))
            return None
        if name == "get_title":
            return "ChatGPT" if self.document is not None and self.document.on_chatgpt else ""
        if name == "get_current_url":
            return self.document.url if self.document is not None else None
        return None


class Phone:
    """Flutter + the platform WebView for the fake Flet session (see the module docstring)."""

    def __init__(self, server, *, listener_timeout: float = LISTENER_TIMEOUT) -> None:
        self.server = server
        self.listener_timeout = float(listener_timeout)
        self.page = None
        self.session = None
        self.known: dict = {}  # control id -> PhoneWebView (every WebView ever mounted)
        self.views: list = []  # in mount order
        self.pending: list = []  # mounted while hidden: built once the app shows again
        self.cookies: set = set()
        self.no_listener: list = []
        self.lost_replies: list = []
        self.conversations: list = []
        self._tasks: set = set()
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    # ---- install --------------------------------------------------------------------------

    def install(self, monkeypatch) -> None:
        import flet_webview as fwv

        phone = self
        mount, unmount = fwv.WebView.did_mount, fwv.WebView.will_unmount

        def did_mount(control):
            mount(control)
            phone.mounted(control)

        def will_unmount(control):
            phone.unmounted(control)
            unmount(control)

        monkeypatch.setattr(fwv.WebView, "did_mount", did_mount)
        monkeypatch.setattr(fwv.WebView, "will_unmount", will_unmount)

    def visible(self) -> bool:
        return self.page is None or getattr(self.page, "app_visible", True) is not False

    def mounted(self, control) -> None:
        """A WebView control reached the client (session loop thread)."""
        session = control.page.session
        if self.session is None:
            self.session, self.page = session, session.page
            self._wrap(session)
        view = PhoneWebView(self, control)
        self.known[view.id] = view
        self.views.append(view)
        if self.visible():
            asyncio.get_running_loop().call_soon(view.build)  # the next frame
        else:
            self.pending.append(view)  # no frames while the app is hidden

    def unmounted(self, control) -> None:
        view = self.known.get(control._i)
        if view is not None:
            view.dispose()

    def frames(self) -> None:
        """The app is visible again: Flutter draws, so WebViews mounted meanwhile are built now."""
        if not self.visible():
            return
        pending, self.pending = self.pending, []
        for view in pending:
            view.build()

    def _wrap(self, session) -> None:
        from flet.messaging.protocol import MessageAction

        conn = session.connection
        send = conn.send_message

        def send_message(message):
            body = message.body
            if message.action == MessageAction.INVOKE_METHOD and getattr(body, "control_id", None) in self.known:
                conn.messages.append(message)
                conn.loop.call_soon_threadsafe(self._invoke, body)
                return
            send(message)

        conn.send_message = send_message

    def _invoke(self, body) -> None:
        view = self.known[body.control_id]

        async def run():
            result, error = None, None
            try:
                result = await view.call(body.name, dict(body.args or {}))
            except Exception as exc:
                error = str(exc)
            try:
                self.session.handle_invoke_method_results(body.control_id, body.call_id, result, error)
            except RuntimeError as exc:  # the control was unmounted meanwhile
                self.lost_replies.append((body.control_id, body.name, str(exc)[:120]))

        self._spawn(run())

    def emit(self, view: PhoneWebView, name: str, data) -> None:
        """A WebView event to Python, over the Flet connection (frames are not needed for it)."""
        if view.disposed or self.session is None:
            return
        self._spawn(self.session.dispatch_event(view.id, name, data))

    def _spawn(self, coro) -> None:
        task = asyncio.ensure_future(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    # ---- chatgpt.com's backend -------------------------------------------------------------

    def backend(self, prompt: str) -> str:
        """The conversation's answer: the fake OpenAI server answers the prompt's messages."""
        body = json.dumps({"model": "chatgpt-web", "messages": _messages(prompt)}).encode("utf-8")
        request = urllib.request.Request(self.server.url + "/chat/completions", data=body, method="POST",
                                         headers={"Content-Type": "application/json"})
        with self._opener.open(request, timeout=300) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return str(payload["choices"][0]["message"]["content"])

    # ---- what happened -----------------------------------------------------------------------

    def describe(self) -> str:
        rows = [f"view {v.id}: mounted_visible={v.mounted_visible} built={v.built} built_visible={v.built_visible} "
                f"disposed={v.disposed} loads={[(u[:30], vis) for u, vis in v.loads][-8:]}" for v in self.views]
        rows.append(f"no_listener={self.no_listener} lost_replies={self.lost_replies}")
        rows.append(f"conversations={self.conversations[-8:]}")
        return "\n".join(rows)


# ==========================================================================
# Fixtures and the owner's flow
# ==========================================================================


@pytest.fixture
def phone_factory(monkeypatch):
    """A :class:`Phone` for a fake server; the browser driver and the routes' cancel flags are reset
    around the test (the app registers its bridge as the process-wide ``browser_driver``)."""
    import authchatgpt_auth as gpt
    import browser_driver
    from glossarion_mobile.services import webview_bridge as wb

    gpt.reset_cancel()
    made = []

    def make(server):
        phone = Phone(server)
        phone.install(monkeypatch)
        made.append(phone)
        return phone

    yield make
    browser_driver.unregister_driver()
    wb._set_current(None)
    gpt.reset_cancel()


def _configure(app, glossary: str) -> None:
    """The owner's settings: authchatgpt/ (keyless, logged out) with the batch size choosing it sets on
    the phone (1), no request spacing, English; the Balanced glossary (approval asked) or none."""
    from glossarion_mobile.state.setting_writes import default_batch_size
    from glossarion_mobile.ui.chat.direct_text_rules import AUTO_ACCEPT_GLOSSARY_PREF
    from glossarion_mobile.ui.screens.welcome_flow import welcome_glossary_updates

    values = {"model": MODEL, "output_language": "English", "delay": 0, "glossary_mode_dialog_shown": True,
              "batch_size": default_batch_size(MODEL)}
    values.update(welcome_glossary_updates(glossary) if glossary != "off"
                  else {"auto_glossary_mode": "off", "enable_auto_glossary": False})
    store = app.config_store
    store.set_many(values)
    store.flush()
    assert store.save_error is None, store.save_error
    assert store.get("model") == MODEL and int(store.get("batch_size")) == 1
    assert bool(store.get("enable_auto_glossary")) is (glossary == "balanced")
    prefs = getattr(app, "prefs", None)
    assert prefs is None or not prefs.get(AUTO_ACCEPT_GLOSSARY_PREF, False)


async def _lifecycle(probe, phone: Phone, *states: str) -> None:
    """Home / back to the app through the session; Flutter draws again once the app is visible."""
    await probe.lifecycle(*states)
    phone.frames()


def _job(probe) -> dict:
    """The job's snapshot state (no progress counts: they lag behind held requests)."""
    run = probe.run()
    job_id = getattr(run, "job_id", None)
    snap = probe.service.snapshot(job_id) if job_id else None
    return {"id": job_id, "state": getattr(getattr(snap, "state", None), "value", None)}


def _describe(ctx) -> str:
    bridge = ctx.app.webview_bridge
    lines = [f"job={_job(ctx.probe)} visible={ctx.phone.visible()} bridge={bridge.status() if bridge else None}",
             ctx.phone.describe()]
    run = ctx.probe.run()
    if run is not None and run.job_id:
        lines.extend(ctx.app.job_service.read_log_tail(run.job_id, 40))
    return "\n".join(lines)


async def _settle_catalog_poll(app, timeout: float = 30.0) -> None:
    """The selected model's 24 h catalog auto-poll (after a model change; Model manager) reads
    chatgpt.com's picker in a hidden page of its own. On the phone the model was chosen long before the
    book: let that poll finish before the job starts (a poll overlapping the job's start shares the
    job's kept WebViews, and which one is built first becomes a race). The process keeps the catalog in
    memory, so after the first test of a run no poll is due: no wait then."""
    catalog = getattr(getattr(app, "models_keys", None), "catalog", None)
    bridge = app.webview_bridge
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        due = await asyncio.to_thread(catalog.due_provider, MODEL) if catalog is not None else None
        if not due and bridge.status()["open_pages"] == 0 and not bridge.host.entries:
            return
        await asyncio.sleep(0.2)
    raise AssertionError(f"the model catalog poll did not finish: due={due!r} bridge={bridge.status()}")


async def _owner_flow(iso, server, phone: Phone, body, *, glossary: str = "off", before_start=None) -> None:
    """Start the app on the Android session, apply the owner's settings, and make the owner's taps: New
    chat, ＋ › Files (the book), Send (the Plan card), Start ("Keep translations running": Not now, as the
    device flow answers it). ``before_start(app)`` runs right before Start; ``body(ctx)`` then drives
    and checks the run."""
    import browser_driver
    import flows
    from glossarion_mobile.diagnostics import fixtures

    epub = fixtures.build_tiny_epub(iso.picks / EPUB_NAME, chapters=CHAPTERS)
    tf = _UF._foundations()
    app, tester, driver = await _UF._host_driver(tf, {EPUB_NAME: epub})
    app.job_service.watcher_interval = 0.2
    run = None
    try:
        await flows.wait_home(driver)
        _assert_isolated(iso.tmp)
        bridge = app.webview_bridge
        # the real in-app browser of the Android build: flet-webview's hidden pages are the backend's driver
        assert bridge is not None and browser_driver.get_driver() is bridge, browser_driver.get_driver()
        assert type(bridge.host).__name__ == "HiddenWebViewHost" and bridge.host.page is tester.page
        _configure(app, glossary)
        await _settle_catalog_poll(app)
        await flows.go_home(driver)
        await driver.tap(tooltip="New chat")
        await driver.tap(tooltip=flows.ATTACH_TOOLTIP)
        await driver.pick_file(epub.name, lambda: driver.tap(key="attach-files"))
        await driver.wait(contains=epub.stem, timeout=60)  # the composer pill
        await driver.tap(key="send-idle_ready", timeout=60)
        await driver.wait(text="Ready to translate", timeout=60)  # the Plan card
        if before_start is not None:
            before_start(app)
        await driver.tap(text="Start")
        cid = app.chat_view.cid
        runs = app.chat_feature.runs
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if await driver.count(text="Keep translations running"):
                await driver.tap(text="Not now")
            candidate = runs.run_for(cid)
            if candidate is not None and candidate.job_id is not None:
                run = candidate
                break
            await asyncio.sleep(0.05)
        assert run is not None, "Start did not submit the chat's job"
        probe = Probe(app, tester, cid)
        ctx = types.SimpleNamespace(app=app, tester=tester, driver=driver, server=server, phone=phone, run=run,
                                    probe=probe, page=tester.page)
        await body(ctx)
    finally:
        try:
            if getattr(tester.page, "app_visible", True) is False:
                for state in COME_BACK:
                    await tester.session.dispatch_event(tester.page._i, "app_lifecycle_state_change", {"state": state})
                phone.frames()
            await _end_job(app, server, run)
        finally:
            try:
                bridge = getattr(app, "webview_bridge", None)
                if bridge is not None:
                    bridge.shutdown()  # what the app's "detach" does: unregister, cancel pages, unmount spares
                    await asyncio.sleep(0.2)
                app.jobs.close()
            finally:
                await tf._stop(app)


def _check_hidden_phase(ctx, *, views_before: int, hidden_from: int, on_screen: int = 1) -> None:
    """The requests from ``hidden_from`` on ran while the app was hidden (``on_screen`` of them were sent
    before Home), on WebViews built while it was on screen."""
    phone, bridge = ctx.phone, ctx.app.webview_bridge
    assert ctx.page.app_visible is False, "the app came back too early"
    # no WebView was mounted or built in the background, and no call met a missing listener
    assert len(phone.views) == views_before, phone.describe()
    assert all(v.mounted_visible and v.built_visible for v in phone.views), phone.describe()
    assert not phone.pending and not phone.no_listener, phone.describe()
    assert bridge.status()["hidden_waits"] == 0, bridge.status()
    # the requests after Home were sent and answered in the background, through the hidden pages
    hidden = phone.conversations[hidden_from:]
    assert len(hidden) > on_screen and all(c["ok"] for c in hidden), phone.describe()
    assert all(c["answered_visible"] is False for c in hidden), phone.describe()
    assert sum(1 for c in hidden if c["sent_visible"] is False) >= len(hidden) - on_screen, phone.describe()
    hidden_loads = [url for v in phone.views for url, visible in v.loads if not visible]
    assert hidden_loads.count(CHATGPT) >= len(hidden) - on_screen, phone.describe()


def _check_background_run(ctx, *, views_before: int, hidden_from: int, on_screen: int = 1) -> None:
    """The job finished while the app stayed hidden, on WebViews built while it was on screen; every
    request of the job went through chatgpt.com's page once."""
    _check_hidden_phase(ctx, views_before=views_before, hidden_from=hidden_from, on_screen=on_screen)
    phone, server = ctx.phone, ctx.server
    # every request went through chatgpt.com's page (none straight to an API), and none was sent twice
    records = server.records(status="ok")
    assert records and all(r.model == "chatgpt-web" for r in records), [r.model for r in records]
    assert len(records) == len(phone.conversations), (len(records), len(phone.conversations))
    # a chapter's own request names one chapter heading (the end-of-book header batch names them all)
    sent = sorted(c["chapters"][0] for c in phone.conversations if len(c["chapters"]) == 1)
    assert sent == list(range(1, CHAPTERS + 1)), f"chapters sent: {sent}"
    log = "\n".join(ctx.app.job_service.read_log_tail(ctx.run.job_id, 2000)).lower()
    found = [bad for bad in BAD_LOG if bad in log]
    assert not found, f"{found}\n{_describe(ctx)}"


async def _check_result(ctx) -> None:
    """Back in the app: the turn's card reads Done for the whole book; the parked WebView is gone once
    the queue drained."""
    await _finished(ctx.app, ctx.run)
    run, probe = ctx.run, ctx.probe
    assert run.state == "done" and run.error is None, (run.state, run.error, _describe(ctx))
    cards = probe.job_cards()
    assert len(cards) == 1, [probe.card_state(c) for c in cards]
    status = f"Done · {CHAPTERS}/{CHAPTERS} chapters"
    assert str(cards[0].state_text.value or "") == status, probe.card_state(cards[0])
    assert status in probe.visible_texts() and EPUB_NAME in probe.visible_texts()
    bridge = ctx.app.webview_bridge
    assert await _until(lambda: not bridge.host.entries and bridge.status()["spares"] == 0, 10), bridge.status()
    assert not bridge.status()["warm"] and all(v.disposed for v in ctx.phone.views), ctx.phone.describe()
    if os.environ.get("GLOSSARION_ISSUE3_DUMP"):  # debug aid: what the phone saw
        print(_describe(ctx))


# ==========================================================================
# 1. Home pressed mid-translation: the book keeps translating in the background
# ==========================================================================


def test_book_keeps_translating_with_the_app_in_the_background(iso, offline, phone_factory):
    async def body(ctx):
        app, server, phone, probe, page = ctx.app, ctx.server, ctx.phone, ctx.probe, ctx.page
        bridge = app.webview_bridge

        # ---- on screen: the job prepared its hidden page; a chapter request is in flight -----------
        assert await _until(lambda: _settled(server), RUN_TIMEOUT), _describe(ctx)
        assert _job(probe)["state"] == "RUNNING", _describe(ctx)
        assert bridge.status()["open_pages"] == 1, bridge.status()
        on_screen = dict(bridge.status())
        views_before = len(phone.views)
        assert views_before >= 1 and all(v.built_visible for v in phone.views), phone.describe()
        assert [c["ok"] for c in phone.conversations] == [True] * HOLD_AFTER + [False], phone.describe()

        # ---- Home: the app goes to the background, then the model answers ---------------------------
        await _lifecycle(probe, phone, *LEAVE)
        assert page.app_visible is False
        hidden_from = len(phone.conversations) - 1  # the held request was sent while the app was visible
        server.release()

        # ---- the rest of the book is translated with the app in the background ------------------------
        assert await _until(lambda: _job(probe)["state"] == "DONE", HIDDEN_TIMEOUT), \
            f"the translation stood still in the background\n{_describe(ctx)}"
        _check_background_run(ctx, views_before=views_before, hidden_from=hidden_from)
        assert len(phone.conversations) - hidden_from >= CHAPTERS + 1 - HOLD_AFTER, phone.describe()
        # the job kept used WebViews (BackgroundExecution.job_started -> jobs_running); its first one was
        # built on screen by the job's warm-up or, when the first request was quicker, by that request
        assert on_screen.get("warm") is True and on_screen.get("prewarmed", 0) <= 1, on_screen

        # ---- back in the app ----------------------------------------------------------------------------
        await _lifecycle(probe, phone, *COME_BACK)
        await _check_result(ctx)

    from glossarion_mobile.diagnostics.fake_llm_server import FakeLLMServer

    with FakeLLMServer() as server:
        HoldGate(server, HOLD_AFTER)
        phone = phone_factory(server)
        asyncio.run(_owner_flow(iso, server, phone, body))
    _assert_offline(offline)


# ==========================================================================
# 2. The glossary gate: ✓ Yes, then Home at once
# ==========================================================================


def test_glossary_gate_yes_then_home_translates_in_the_background(iso, offline, phone_factory):
    async def body(ctx):
        app, server, phone, probe, page = ctx.app, ctx.server, ctx.phone, ctx.probe, ctx.page
        run = ctx.run

        # ---- the Balanced glossary runs on screen through the hidden page -----------------------------
        assert await _until(lambda: run.awaiting_glossary, RUN_TIMEOUT), _describe(ctx)
        assert await _until(lambda: app.chat_view.approval_card is not None, 30), _describe(ctx)
        assert server.count("glossary") >= 1 and server.count("translation") == 0, _describe(ctx)
        assert _job(probe)["state"] == "RUNNING"
        # the last glossary page closes; its WebView is parked while the job runs (checked below, after
        # the owner-visible outcome)
        await _until(lambda: app.webview_bridge.status().get("spares", 0) >= 1, 10)
        on_screen = dict(app.webview_bridge.status())
        views_before = len(phone.views)
        hidden_from = len(phone.conversations)

        # ---- ✓ Yes on the approval card, then Home at once --------------------------------------------
        await _D12._tap(probe, app.chat_view.approval_card.yes_button)
        await _lifecycle(probe, phone, *LEAVE)
        assert page.app_visible is False

        # ---- the whole translation runs in the background -------------------------------------------------
        assert await _until(lambda: _job(probe)["state"] == "DONE", HIDDEN_TIMEOUT), \
            f"the translation stood still in the background\n{_describe(ctx)}"
        _check_background_run(ctx, views_before=views_before, hidden_from=hidden_from)
        translated = [c["chapters"][0] for c in phone.conversations[hidden_from:] if len(c["chapters"]) == 1]
        assert sorted(translated) == list(range(1, CHAPTERS + 1)), phone.describe()
        # at the gate the glossary's WebView was kept for the translation (parked on the blank page)
        assert on_screen.get("warm") is True and on_screen.get("spares", 0) >= 1, on_screen

        await _lifecycle(probe, phone, *COME_BACK)
        await _check_result(ctx)

    from glossarion_mobile.diagnostics.fake_llm_server import FakeLLMServer

    with FakeLLMServer() as server:
        phone = phone_factory(server)
        asyncio.run(_owner_flow(iso, server, phone, body, glossary="balanced"))
    _assert_offline(offline)


# ==========================================================================
# 3. Start, then Home at once: the job's own WebView serves it in the background
# ==========================================================================


def test_start_then_home_at_once_runs_the_whole_job_in_the_background(iso, offline, phone_factory):
    """The owner taps Start and leaves at once, before the job's first request has a page: the WebView the
    job built at its start (``BackgroundExecution.job_started`` -> ``WebViewBridge.prewarm``, on screen) serves
    the Balanced glossary in the background and the job waits at the approval gate; back in the app,
    ✓ Yes, Home again: the translation finishes in the background on the same WebView.

    The first request's page waits for chatgpt.com's one-at-a-time load lock
    (``authchatgpt_auth._DRIVER_LOADING``), which the test holds from Start until the app is hidden."""
    import authchatgpt_auth as gpt

    loading = gpt._DRIVER_LOADING
    held = []

    def hold_loading(app):
        assert loading.acquire(timeout=30), "chatgpt.com's load lock is busy before Start"
        held.append(True)

    def release_loading():
        if held:
            held.pop()
            loading.release()

    async def body(ctx):
        app, server, phone, probe, page = ctx.app, ctx.server, ctx.phone, ctx.probe, ctx.page
        run, bridge = ctx.run, app.webview_bridge

        # ---- on screen: the job started and built its WebView; its first request waits for a load --------
        assert await _until(lambda: bridge.status().get("spares", 0) == 1 and bridge.status()["open_pages"] == 1,
                            RUN_TIMEOUT), _describe(ctx)
        assert _job(probe)["state"] == "RUNNING" and not phone.conversations, _describe(ctx)
        on_screen = dict(bridge.status())
        views_before = len(phone.views)

        # ---- Home at once: everything from the first request on runs in the background ---------------------
        await _lifecycle(probe, phone, *LEAVE)
        assert page.app_visible is False
        release_loading()
        assert await _until(lambda: run.awaiting_glossary, HIDDEN_TIMEOUT), \
            f"the glossary stood still in the background\n{_describe(ctx)}"
        assert _job(probe)["state"] == "RUNNING" and server.count("glossary") >= 1, _describe(ctx)
        assert server.count("translation", status=None) == 0, "chapters were sent before the approval"
        _check_hidden_phase(ctx, views_before=views_before, hidden_from=0, on_screen=0)
        assert on_screen["prewarmed"] == 1 and on_screen["warm"] is True, on_screen
        (warmed,) = [v for v in phone.views if v.loads and v.loads[0][0] == "about:blank"]
        assert warmed.mounted_visible and warmed.built_visible, phone.describe()
        assert {c["view"] for c in phone.conversations} == {warmed.id}, phone.describe()

        # ---- back in the app at the gate: ✓ Yes, then Home again -----------------------------------------
        await _lifecycle(probe, phone, *COME_BACK)
        assert await _until(lambda: app.chat_view.approval_card is not None
                            and probe.on_screen(app.chat_view.approval_card.yes_button), 30), _describe(ctx)
        views_before = len(phone.views)
        hidden_from = len(phone.conversations)
        await _D12._tap(probe, app.chat_view.approval_card.yes_button)
        await _lifecycle(probe, phone, *LEAVE)
        assert page.app_visible is False

        # ---- the translation finishes in the background ----------------------------------------------------
        assert await _until(lambda: _job(probe)["state"] == "DONE", HIDDEN_TIMEOUT), \
            f"the translation stood still in the background\n{_describe(ctx)}"
        _check_background_run(ctx, views_before=views_before, hidden_from=hidden_from)
        assert {c["view"] for c in phone.conversations} == {warmed.id}, phone.describe()

        await _lifecycle(probe, phone, *COME_BACK)
        await _check_result(ctx)

    from glossarion_mobile.diagnostics.fake_llm_server import FakeLLMServer

    try:
        with FakeLLMServer() as server:
            phone = phone_factory(server)
            asyncio.run(_owner_flow(iso, server, phone, body, glossary="balanced", before_start=hold_loading))
    finally:
        release_loading()
    _assert_offline(offline)


# ==========================================================================
# Hygiene
# ==========================================================================


def test_file_keeps_uniform_line_endings_and_python_310_syntax():
    data = Path(__file__).read_bytes()
    assert data.count(b"\r\n") in (0, data.count(b"\n"))
    ast.parse(data.decode("utf-8"), feature_version=(3, 10))
