"""Regression tests for "ChatGPT / Claude sign-in hangs on the phone after choosing the organization".

The owner's report: after picking the account / organization the in-app browser page just
spins; no error in the browser or the app (ChatGPT and Claude). The causes and what each test
reproduces:

* **The sign-in foreground service died as soon as the browser opened** (root cause).
  ``ForegroundTaskOptions(stopWithTask: true)`` makes flutter_foreground_task 11.x stop the
  service whenever no activity of the app is resumed (``TrackVisibilityUtils``); opening the
  Custom Tab pauses MainActivity. Android then caches and freezes the app behind the browser:
  the kernel still accepts the browser's connection to the loopback listener, nobody answers,
  the page spins. ``test_job_service_options_leave_stop_with_task_to_the_manifest`` guards the
  Dart option; ``test_sign_in_service_lost_when_the_browser_opens_*`` model the old build and
  check the app notices (``SignInState.notice`` + paste field), the fixed build keeps it.
* **A job ending during the sign-in stopped the service** (and a sign-in ending stopped a job's
  service): ``test_a_job_and_a_sign_in_share_the_foreground_service``.
* **Claude lost a callback whose tab was closed before the (late) answer**: the handler wrote
  the page before recording the code. ``test_a_tab_closed_before_the_late_answer_still_finishes``
  holds the handler (the frozen app), resets the client connection, then lets it run.
* **One idle browser connection blocked the single-threaded listener**:
  ``test_an_idle_connection_does_not_hold_back_the_callback`` (mobile gets a threaded
  listener with a read timeout; desktop keeps ``HTTPServer``).
* **Nothing may wait for the glossarion:// deep link**: ``test_the_callback_page_needs_no_deep_link``.
* **Back in the app with the sign-in still waiting**: ``test_back_in_the_app_while_waiting_*``.

Fake servers only: the token exchange is faked, the "browser" is a socket / http.client
talking to the real loopback listeners of ``authgpt_auth`` / ``authcd_auth``; token stores
live in pytest's tmp_path.

Run from src/mobile:
    python -m pytest -p no:cacheprovider -W ignore tests_host/test_oauth_signin_hang.py
"""

from __future__ import annotations

import asyncio
import base64
import http.client
import importlib.util
import json
import logging
import os
import re
import socket
import struct
import sys
import threading
import time
import types
import urllib.parse
import xml.etree.ElementTree as ET
from http.server import HTTPServer
from pathlib import Path

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = MOBILE_DIR / "app"
SRC_DIR = MOBILE_DIR.parent
for entry in (str(APP_DIR), str(SRC_DIR)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from glossarion_mobile.services.native import NativeBridge, ServiceHolds  # noqa: E402
from glossarion_mobile.services.oauth import SIGN_IN_HOLD, OAuthBridge, SignInState  # noqa: E402

TIMEOUT = 10.0
RETURN_URL = "glossarion://app/oauth/return"
EXTENSION = MOBILE_DIR / "extensions" / "flet_glossarion_native"
FLUTTER_PKG = EXTENSION / "src" / "flutter" / "flet_glossarion_native"
ANDROID_NS = "{http://schemas.android.com/apk/res/android}"


# ==========================================================================
# helpers
# ==========================================================================


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _jwt(payload: dict) -> str:
    def part(data: dict) -> str:
        return base64.urlsafe_b64encode(json.dumps(data).encode()).decode().rstrip("=")

    return f"{part({'alg': 'none'})}.{part(payload)}.sig"


async def _wait_for(predicate, timeout=TIMEOUT) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.02)
    return predicate()


def _get(port: int, path: str, timeout: float = 5.0):
    """The browser following the redirect to the loopback (no redirects followed)."""
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        conn.request("GET", path)
        response = conn.getresponse()
        return response.status, response.read().decode("utf-8")
    finally:
        conn.close()


class Opener:
    def __init__(self, on_open=None) -> None:
        self.opened: list = []
        self.on_open = on_open

    def __call__(self, url: str) -> bool:
        self.opened.append(url)
        if self.on_open is not None:
            self.on_open(url)
        return True


@pytest.fixture
def mobile_auth(tmp_path, monkeypatch):
    """The real authgpt / authcd modules in the app's mobile mode, with faked token exchanges."""
    authgpt_auth = pytest.importorskip("authgpt_auth")
    authcd_auth = pytest.importorskip("authcd_auth")
    import token_encryption

    monkeypatch.setenv("GLOSSARION_MOBILE", "1")  # bootstrap env contract
    monkeypatch.setenv("GLOSSARION_OAUTH_RETURN_URL", RETURN_URL)
    monkeypatch.setattr(authgpt_auth, "CALLBACK_PORT", _free_port())
    exchanged: list = []

    def gpt_exchange(code, verifier, redirect_uri):
        exchanged.append(("authgpt", code, redirect_uri))
        return {"access_token": f"gpt-{code}", "refresh_token": "r", "expires_in": 3600,
                "expires_at": time.time() + 3600,
                "id_token": _jwt({"email": "reader@example.com",
                                  "https://api.openai.com/auth": {"chatgpt_plan_type": "plus"}})}

    def cd_exchange(code, verifier, redirect_uri=None, state=None):
        exchanged.append(("authcd", code, redirect_uri))
        return {"access_token": f"cd-{code}", "refresh_token": "r", "expires_at": time.time() + 3600,
                "account": {"email_address": "claude@example.com"}}

    monkeypatch.setattr(authgpt_auth, "exchange_code_for_tokens", gpt_exchange)
    monkeypatch.setattr(authcd_auth, "exchange_code_for_tokens", cd_exchange)
    token_encryption.set_symmetric_key(os.urandom(32))

    def namespace(module, store_cls, provider):
        stores: dict = {}

        def get_store(account_id=None):
            key = int(account_id or 0)
            if key not in stores:
                name = f"{provider}_tokens.json" if not key else f"{provider}_tokens_{key}.json"
                stores[key] = store_cls(token_file=str(tmp_path / name), account_id=key)
            return stores[key]

        return types.SimpleNamespace(begin_oauth=module.begin_oauth, complete_from_redirect=module.complete_from_redirect,
                                     get_store=get_store, _DEFAULT_TOKEN_DIR=str(tmp_path), stores=stores)

    modules = {
        "authgpt": namespace(authgpt_auth, authgpt_auth.AuthGPTTokenStore, "authgpt"),
        "authcd": namespace(authcd_auth, authcd_auth.AuthCDTokenStore, "authcd"),
    }
    try:
        yield types.SimpleNamespace(modules=modules, exchanged=exchanged, authgpt=authgpt_auth, authcd=authcd_auth,
                                    tmp=tmp_path)
    finally:
        token_encryption.set_symmetric_key(None)


def _bridge(env, opener, **kwargs) -> OAuthBridge:
    kwargs.setdefault("timeout", 20)
    kwargs.setdefault("safe_root", None)
    return OAuthBridge(auth_modules=env.modules, opener=opener, **kwargs)


def _port(session) -> int:
    """The loopback port of a sign-in (authgpt's own OAuthSession has no ``port``)."""
    return urllib.parse.urlsplit(session.redirect_uri).port


def _callback_path(session, code: str = "CODE", state: str = None) -> str:
    redirect = urllib.parse.urlsplit(session.redirect_uri)
    return f"{redirect.path}?code={code}&state={urllib.parse.quote(state or session.state)}"


class FakeServiceSide:
    """The Dart / flutter_foreground_task side of GlossarionNative's job service (one service per app).

    ``stop_when_paused`` models the old build (``ForegroundTaskOptions(stopWithTask: true)``): the
    library stops the service as soon as the app's activity pauses, which opening the in-app
    browser does, and the task handler reports ``destroyed``.
    """

    def __init__(self, stop_when_paused: bool = False) -> None:
        self.stop_when_paused = stop_when_paused
        self.running = False
        self.title = self.text = None
        self.calls: list = []

    async def start_job_service(self, title, text, **kwargs):
        self.calls.append(("update" if self.running else "start", title, text))
        self.running = True
        self.title, self.text = title, text
        return True

    async def update_job_service(self, title=None, text=None):
        self.calls.append(("update", title, text))
        if self.running:
            self.title = title or self.title
            self.text = text or self.text

    async def stop_job_service(self):
        self.calls.append(("stop",))
        self.running = False

    async def is_job_service_running(self):
        return self.running

    async def activity_paused(self):
        """MainActivity.onPause (the Custom Tab opened over it)."""
        if self.stop_when_paused and self.running:
            self.running = False
            self.calls.append(("stopped-by-library",))
            await self.on_foreground({"type": "destroyed", "is_timeout": False})


def _android_native(stop_when_paused: bool = False):
    side = FakeServiceSide(stop_when_paused)
    return NativeBridge(native=side), side


# ==========================================================================
# 1. the foreground service options (root cause)
# ==========================================================================


def _strip_dart_comments(source: str) -> str:
    return re.sub(r"//[^\n]*", "", source)


def test_job_service_options_leave_stop_with_task_to_the_manifest():
    """flutter_foreground_task 11.0.3: ``stopWithTask: true`` stores the STOP_WITH_TASK pref, and
    ForegroundService.onStartCommand then installs TrackVisibilityUtils, whose onActivityPaused stops the
    service once no activity is resumed - the sign-in Custom Tab pauses MainActivity, so the "Signing
    in…" service died the moment the browser appeared. Unset, the library falls back to the manifest's
    android:stopWithTask (stop only when the app is swiped away)."""
    dart = (FLUTTER_PKG / "lib" / "src" / "native_service.dart").read_text(encoding="utf-8")
    code = _strip_dart_comments(dart)
    start = code.index("foregroundTaskOptions: ForegroundTaskOptions(")
    options = code[start:code.index(")", code.index("allowAutoRestart", start))]
    assert "allowAutoRestart: false" in options
    assert "stopWithTask" not in code  # never set: not here, not in an update

    manifest = ET.parse(FLUTTER_PKG / "android" / "src" / "main" / "AndroidManifest.xml").getroot()
    service = manifest.find("application").find("service")
    assert service.get(ANDROID_NS + "name") == "com.pravera.flutter_foreground_task.service.ForegroundService"
    assert service.get(ANDROID_NS + "stopWithTask") == "true"  # swiping the app away still stops it

    readme = (EXTENSION / "README.md").read_text(encoding="utf-8")
    assert "flip the manifest flag and the" not in readme  # the old advice that paired the two switches


# ==========================================================================
# 2. the sign-in service while the browser is in front
# ==========================================================================


def test_sign_in_service_lost_when_the_browser_opens_shows_the_paste(mobile_auth, caplog):
    """The old build: the service stops when the browser opens (no error anywhere, the page spins).
    The bridge now notices (``destroyed`` event), warns, and the LoginPanel opens the paste field;
    the redirect URL pasted from the browser finishes the sign-in."""
    native, side = _android_native(stop_when_paused=True)
    opener = Opener(on_open=lambda url: asyncio.ensure_future(side.activity_paused()))
    bridge = _bridge(mobile_auth, opener, native=native, is_android=True, fgs_check_delay=30)

    async def scenario():
        task = asyncio.ensure_future(bridge.sign_in("authgpt"))
        assert await _wait_for(lambda: bridge.state.notice)
        assert bridge.state.step == "waiting" and "sign-in service is not running" in bridge.state.notice
        assert ("stopped-by-library",) in side.calls and SIGN_IN_HOLD not in native.service_holds
        session = bridge._session
        url = f"http://localhost:{_port(session)}{_callback_path(session, 'PASTED')}"
        status = await bridge.complete_with_paste(url)
        assert await asyncio.wait_for(task, TIMEOUT) == {}
        return status

    with caplog.at_level(logging.WARNING, logger="glossarion.oauth"):
        status = asyncio.run(scenario())
    assert status["signed_in"] and mobile_auth.exchanged[-1][:2] == ("authgpt", "PASTED")
    assert any("sign-in foreground service is gone" in r.getMessage() for r in caplog.records)
    assert side.calls.count(("stop",)) == 0  # nothing left to stop


def test_sign_in_service_lost_without_an_event_is_found_by_the_check(mobile_auth):
    """No ``destroyed`` event (or it was missed): the check after opening the browser finds it."""
    native, side = _android_native()
    opener = Opener(on_open=lambda url: setattr(side, "running", False))  # gone, silently
    bridge = _bridge(mobile_auth, opener, native=native, is_android=True, fgs_check_delay=0.05)

    async def scenario():
        task = asyncio.ensure_future(bridge.sign_in("authcd"))
        assert await _wait_for(lambda: bridge.state.notice)
        notice = bridge.state.notice
        bridge.cancel()
        await asyncio.wait_for(task, TIMEOUT)
        return notice

    notice = asyncio.run(scenario())
    assert "sign-in service is not running" in notice and "Get a code to paste instead" in notice  # Claude's code page


def test_reopen_browser_restarts_a_lost_sign_in_service_first(mobile_auth):
    """Back in the app after the service was lost, "Reopen browser" starts it again (the app is in front,
    so Android allows it) before the browser covers the app once more."""
    native, side = _android_native()
    opener = Opener()
    bridge = _bridge(mobile_auth, opener, native=native, is_android=True, fgs_check_delay=0.05)

    async def scenario():
        task = asyncio.ensure_future(bridge.sign_in("authgpt"))
        assert await _wait_for(lambda: opener.opened)
        side.running = False  # killed while the browser was in front
        assert await _wait_for(lambda: bridge.state.notice)
        assert bridge.reopen_browser()
        assert await _wait_for(lambda: len(opener.opened) == 2)
        assert side.running and not bridge.state.notice and SIGN_IN_HOLD in native.service_holds
        assert [c[0] for c in side.calls].count("start") == 2
        session = bridge._session
        await asyncio.to_thread(_get, _port(session), _callback_path(session))
        return await asyncio.wait_for(task, TIMEOUT)

    assert asyncio.run(scenario())["signed_in"]
    assert side.calls[-1] == ("stop",) and not side.running and len(native.service_holds) == 0


def test_sign_in_service_that_cannot_start_shows_the_paste_at_once(mobile_auth, caplog):
    native, side = _android_native()

    async def refuse(title, text, **kwargs):
        side.calls.append(("refused", title))
        return False

    side.start_job_service = refuse
    opener = Opener()
    bridge = _bridge(mobile_auth, opener, native=native, is_android=True, fgs_check_delay=0.01)

    async def scenario():
        task = asyncio.ensure_future(bridge.sign_in("authgpt"))
        assert await _wait_for(lambda: opener.opened)
        notice = bridge.state.notice
        session = bridge._session
        await asyncio.to_thread(_get, _port(session), _callback_path(session))
        return notice, await asyncio.wait_for(task, TIMEOUT)

    with caplog.at_level(logging.WARNING, logger="glossarion.oauth"):
        notice, status = asyncio.run(scenario())
    assert "sign-in service is not running" in notice and status["signed_in"]
    assert any("did not start" in r.getMessage() for r in caplog.records)
    assert ("stop",) not in side.calls  # nothing was started, nothing to stop


def test_sign_in_service_survives_the_browser_and_stops_when_done(mobile_auth):
    """The fixed build: opening the browser leaves the service running; the sign-in ends it."""
    native, side = _android_native(stop_when_paused=False)
    opener = Opener(on_open=lambda url: asyncio.ensure_future(side.activity_paused()))
    bridge = _bridge(mobile_auth, opener, native=native, is_android=True, fgs_check_delay=0.05)

    async def scenario():
        task = asyncio.ensure_future(bridge.sign_in("authgpt"))
        assert await _wait_for(lambda: opener.opened)
        await asyncio.sleep(0.2)  # past the check
        assert side.running and not bridge.state.notice and SIGN_IN_HOLD in native.service_holds
        session = bridge._session
        status, _page = await asyncio.to_thread(_get, _port(session), _callback_path(session))
        assert status == 200
        return await asyncio.wait_for(task, TIMEOUT)

    status = asyncio.run(scenario())
    assert status["signed_in"] and side.calls[0][0] == "start" and side.calls[-1] == ("stop",)
    assert not side.running and len(native.service_holds) == 0


def test_notification_stop_cancels_a_sign_in_that_owns_the_service(mobile_auth):
    native, side = _android_native()
    opener = Opener()
    bridge = _bridge(mobile_auth, opener, native=native, is_android=True, fgs_check_delay=30)

    async def scenario():
        task = asyncio.ensure_future(bridge.sign_in("authcd"))
        assert await _wait_for(lambda: opener.opened)
        await side.on_foreground({"type": "button", "button_id": "stop"})
        return await asyncio.wait_for(task, TIMEOUT)

    assert asyncio.run(scenario()) == {}
    assert bridge.state.step == "cancelled" and side.calls[-1] == ("stop",) and not side.running


def test_a_job_and_a_sign_in_share_the_foreground_service(mobile_auth):
    """A job ending while the browser is open must not stop the sign-in's service, and a sign-in
    ending must not stop a running job's service (it gets its notification back)."""
    from glossarion_mobile.services.background import BackgroundExecution
    from glossarion_mobile.services.jobs import JobSnapshot, JobSpec, JobState

    class Idle:
        def view(self):
            return types.SimpleNamespace(queue=(), paused=False)

        def snapshot(self, job_id=None):
            return None

    def snap(state):
        return JobSnapshot(id="job1", spec=JobSpec("translate", "Book.epub", ("a",)), state=state, created=0.0,
                           started=1.0)

    native, side = _android_native()
    background = BackgroundExecution(native, platform="android", jobs=Idle())

    async def scenario():
        # (1) job first, sign-in joins, job ends during the sign-in, sign-in ends.
        await background.job_started(snap(JobState.STARTING))
        assert side.running and side.calls[-1][0] == "start"
        opener = Opener()
        bridge = _bridge(mobile_auth, opener, native=native, is_android=True, fgs_check_delay=30)
        task = asyncio.ensure_future(bridge.sign_in("authgpt"))
        assert await _wait_for(lambda: opener.opened)
        assert sorted(native.service_holds.names()) == ["jobs", SIGN_IN_HOLD]
        await background.job_finished(snap(JobState.DONE))
        assert side.running, "the job's end stopped the service of the waiting sign-in"
        assert side.title == "Signing in to ChatGPT"  # its notification is back
        session = bridge._session
        await asyncio.to_thread(_get, _port(session), _callback_path(session, "ONE"))
        assert (await asyncio.wait_for(task, TIMEOUT))["signed_in"]
        assert not side.running and len(native.service_holds) == 0  # the last one out stopped it

        # (2) sign-in first, a job starts during it, sign-in ends, job ends.
        opener2 = Opener()
        bridge2 = _bridge(mobile_auth, opener2, native=native, is_android=True, fgs_check_delay=30)
        task = asyncio.ensure_future(bridge2.sign_in("authgpt"))
        assert await _wait_for(lambda: opener2.opened)
        assert side.running and side.title == "Signing in to ChatGPT"
        await background.job_started(snap(JobState.STARTING))
        session = bridge2._session
        await asyncio.to_thread(_get, _port(session), _callback_path(session, "TWO"))
        assert (await asyncio.wait_for(task, TIMEOUT))["signed_in"]
        assert side.running, "the sign-in's end stopped the running job's service"
        assert side.title == "Glossarion" and "Book.epub" in side.text  # the job's notification is back
        await background.job_finished(snap(JobState.DONE))
        assert not side.running and len(native.service_holds) == 0

    asyncio.run(scenario())


def test_a_late_destroyed_event_does_not_count_against_a_running_service(mobile_auth):
    """Every stop ends with a ``destroyed`` event; one that arrives after the service runs again (an earlier
    sign-in's or job's) must neither flag the sign-in nor drop a hold."""
    from glossarion_mobile.services.background import BackgroundExecution
    from glossarion_mobile.services.jobs import JobSnapshot, JobSpec, JobState

    native, side = _android_native()
    opener = Opener()
    bridge = _bridge(mobile_auth, opener, native=native, is_android=True, fgs_check_delay=30)
    background = BackgroundExecution(native, platform="android", jobs=None)
    job = JobSnapshot(id="j", spec=JobSpec("translate", "Book.epub", ("a",)), state=JobState.STARTING, created=0.0)

    async def scenario():
        task = asyncio.ensure_future(bridge.sign_in("authgpt"))
        assert await _wait_for(lambda: opener.opened)
        await background.job_started(job)
        await side.on_foreground({"type": "destroyed", "is_timeout": False})  # stale
        await background.on_foreground_event({"type": "destroyed", "is_timeout": False})
        assert not bridge.state.notice and sorted(native.service_holds.names()) == ["jobs", SIGN_IN_HOLD]
        side.running = False  # now it is really gone
        await side.on_foreground({"type": "destroyed", "is_timeout": False})
        await background.on_foreground_event({"type": "destroyed", "is_timeout": False})
        assert "sign-in service is not running" in bridge.state.notice and len(native.service_holds) == 0
        session = bridge._session
        await asyncio.to_thread(_get, _port(session), _callback_path(session))
        return await asyncio.wait_for(task, TIMEOUT)

    assert asyncio.run(scenario())["signed_in"]


def test_service_holds_hand_back_the_remaining_notification():
    holds = ServiceHolds()
    assert holds.release("jobs") is None
    holds.hold("jobs", "Glossarion", "Translating A")
    holds.hold(SIGN_IN_HOLD, "Signing in to Claude", "Waiting for sign-in…")
    holds.hold("jobs", "Glossarion", "Translating A: 2/9 chapters")  # refreshed
    assert holds.release(SIGN_IN_HOLD) == ("Glossarion", "Translating A: 2/9 chapters")
    assert holds.release("jobs") is None and len(holds) == 0


# ==========================================================================
# 3. the loopback listener
# ==========================================================================


def _gate(monkeypatch, handler_cls):
    """Hold every request in its handler (the app is frozen with the request already queued)."""
    entered, release = threading.Event(), threading.Event()
    original = handler_cls.do_GET

    def gated(self):
        entered.set()
        release.wait(TIMEOUT)
        return original(self)

    monkeypatch.setattr(handler_cls, "do_GET", gated)
    return entered, release


def _send_request(port: int, path: str) -> socket.socket:
    sock = socket.create_connection(("127.0.0.1", port), timeout=5)
    sock.sendall(f"GET {path} HTTP/1.1\r\nHost: localhost:{port}\r\n\r\n".encode("ascii"))
    return sock


def _reset(sock: socket.socket) -> None:
    """Close the way a killed tab does: RST, nothing left to receive the answer."""
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
    sock.close()


@pytest.mark.parametrize("provider", ["authgpt", "authcd"])
def test_a_tab_closed_before_the_late_answer_still_finishes(provider, mobile_auth, monkeypatch):
    """The app was frozen with the browser's callback queued; the user closed the stuck tab and went
    back to the app. On unfreezing, the handler's answer fails - the code must still be recorded and the
    sign-in finish (Claude used to write first and lose it; the sheet stayed on "Waiting" until 300 s)."""
    handler = {"authgpt": mobile_auth.authgpt._OAuthCallbackHandler,
               "authcd": mobile_auth.authcd._AutomaticLoginCallbackHandler}[provider]
    entered, release = _gate(monkeypatch, handler)
    opener = Opener()
    bridge = _bridge(mobile_auth, opener, timeout=8)

    async def scenario():
        task = asyncio.ensure_future(bridge.sign_in(provider))
        assert await _wait_for(lambda: opener.opened)
        session = bridge._session
        sock = await asyncio.to_thread(_send_request, _port(session), _callback_path(session, "LATE"))
        assert await asyncio.to_thread(entered.wait, 5)
        _reset(sock)  # the tab is gone before the frozen app answers
        await asyncio.sleep(0.2)
        release.set()  # the app is running again
        return await asyncio.wait_for(task, TIMEOUT)

    try:
        status = asyncio.run(scenario())
    finally:
        release.set()
    assert status["signed_in"] and mobile_auth.exchanged[-1][:2] == (provider, "LATE")
    assert bridge.state.step == "done"


@pytest.mark.parametrize("provider", ["authgpt", "authcd"])
def test_an_idle_connection_does_not_hold_back_the_callback(provider, mobile_auth):
    """A connection that never sends a request (a browser preconnect, an aborted load) used to block
    the single-threaded listener in readline() with no timeout: the real redirect waited behind it."""
    opener = Opener()
    bridge = _bridge(mobile_auth, opener)

    async def scenario():
        task = asyncio.ensure_future(bridge.sign_in(provider))
        assert await _wait_for(lambda: opener.opened)
        session = bridge._session
        idle = socket.create_connection(("127.0.0.1", _port(session)), timeout=5)
        try:
            await asyncio.sleep(0.2)  # the listener has accepted it
            status, page = await asyncio.to_thread(_get, _port(session), _callback_path(session, "AFTER-IDLE"), 4.0)
        finally:
            idle.close()
        assert status == 200 and "Return to Glossarion" in page
        return await asyncio.wait_for(task, TIMEOUT)

    status = asyncio.run(scenario())
    assert status["signed_in"] and mobile_auth.exchanged[-1][:2] == (provider, "AFTER-IDLE")


def test_desktop_keeps_the_single_threaded_listeners(mobile_auth, monkeypatch, tmp_path):
    """Desktop (no GLOSSARION_MOBILE) still gets plain ``HTTPServer`` listeners."""
    monkeypatch.delenv("GLOSSARION_MOBILE", raising=False)
    monkeypatch.delenv("FLET_PLATFORM", raising=False)
    monkeypatch.delenv("GLOSSARION_OAUTH_RETURN_URL", raising=False)
    gpt = mobile_auth.authgpt.begin_oauth(timeout=30, persist=False, dual_stack=False, auto_close=False)
    cd = mobile_auth.authcd.begin_oauth(timeout=30, persist=False, dual_stack=False, auto_close=False, serve=False)
    try:
        assert [type(s) for s in gpt._servers] == [HTTPServer]
        assert [type(s) for s in cd._servers] == [HTTPServer]
    finally:
        gpt.close()
        cd.close()
    monkeypatch.setenv("GLOSSARION_MOBILE", "1")
    mobile = mobile_auth.authcd.begin_oauth(timeout=30, persist=False, dual_stack=False, serve=False)
    try:
        assert [type(s) for s in mobile._servers] == [HTTPServer]  # caller-driven (desktop loop) stays plain
    finally:
        mobile.close()


@pytest.mark.parametrize("provider", ["authgpt", "authcd"])
def test_the_callback_page_needs_no_deep_link(provider, mobile_auth):
    """One request, answered with the page itself; the sign-in finishes without the glossarion://
    return link (Custom Tabs may block a script-started jump to the app), and the page says the
    tab can be closed."""
    returned: list = []
    opener = Opener()
    bridge = _bridge(mobile_auth, opener, close_browser=lambda: returned.append(True))

    async def scenario():
        task = asyncio.ensure_future(bridge.sign_in(provider))
        assert await _wait_for(lambda: opener.opened)
        session = bridge._session
        status, page = await asyncio.to_thread(_get, _port(session), _callback_path(session, "PAGE"))
        return status, page, await asyncio.wait_for(task, TIMEOUT)

    status, page, result = asyncio.run(scenario())
    assert status == 200  # no 302 to a second URL the app would also have to serve
    assert f"{RETURN_URL}?p={provider}" in page and "close this page" in page
    assert result["signed_in"] and bridge.state.step == "done" and returned == []


def test_a_jobs_browser_login_still_ends_on_the_callback(mobile_auth, monkeypatch):
    """A job whose ChatGPT login expired falls back to ``authgpt_auth.run_oauth_flow`` (it waits with
    ``session.wait()``). On mobile the callback is answered with the page itself - no /success request
    follows - so the callback has to end that wait, as the /success page does on desktop."""
    authgpt = mobile_auth.authgpt

    def browser(url, *args, **kwargs):
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
        port = urllib.parse.urlsplit(query["redirect_uri"][0]).port
        path = f"/auth/callback?code=JOB&state={urllib.parse.quote(query['state'][0])}"
        threading.Thread(target=_get, args=(port, path), daemon=True).start()
        return True

    monkeypatch.setattr(authgpt.webbrowser, "open", browser)
    started = time.monotonic()
    tokens = authgpt.run_oauth_flow(timeout=20)
    assert tokens["access_token"] == "gpt-JOB" and time.monotonic() - started < 10


def test_a_bad_chatgpt_callback_gets_an_error_page_on_mobile(mobile_auth):
    opener = Opener()
    bridge = _bridge(mobile_auth, opener)

    async def scenario():
        task = asyncio.ensure_future(bridge.sign_in("authgpt"))
        assert await _wait_for(lambda: opener.opened)
        session = bridge._session
        status, page = await asyncio.to_thread(_get, _port(session), _callback_path(session, "X", state="forged"))
        with pytest.raises(RuntimeError, match="state mismatch"):
            await asyncio.wait_for(task, TIMEOUT)
        return status, page

    status, page = asyncio.run(scenario())
    assert status == 400 and "state mismatch" in page and bridge.state.step == "error"


# ==========================================================================
# 4. back in the app while the sign-in still waits
# ==========================================================================


@pytest.mark.skipif(importlib.util.find_spec("flet") is None, reason="flet not installed")
def test_back_in_the_app_while_waiting_opens_the_paste(mobile_auth):
    from glossarion_mobile.ui.screens.accounts import LoginPanel

    opener = Opener()
    bridge = _bridge(mobile_auth, opener, resume_check_delay=0.01)
    panel = LoginPanel(bridge, provider="authcd")
    assert not panel.paste_field.visible and not panel.notice_text.visible

    async def scenario():
        task = asyncio.ensure_future(bridge.sign_in("authcd"))
        assert await _wait_for(lambda: opener.opened)
        bridge.on_lifecycle("inactive")  # leaving: nothing to do
        await asyncio.sleep(0.05)
        assert not bridge.state.notice
        bridge.on_lifecycle("resume")
        assert await _wait_for(lambda: bridge.state.notice)
        state = bridge.state
        session = bridge._session
        status = await bridge.complete_with_paste(f"PASTED#{session.state}", provider="authcd")
        assert await asyncio.wait_for(task, TIMEOUT) == {}
        return state, status

    state, status = asyncio.run(scenario())
    assert state.notice.startswith("Still waiting for the browser") and "Get a code to paste instead" in state.notice
    panel.apply(state)
    assert panel.notice_text.visible and panel.notice_text.value == state.notice
    assert panel.paste_field.visible and panel.finish_button.visible and panel.manual_button.visible
    assert status["signed_in"]
    panel.apply(bridge.state)  # done
    assert not panel.notice_text.visible and not panel.paste_field.visible


@pytest.mark.skipif(importlib.util.find_spec("flet") is None, reason="flet not installed")
def test_the_chat_feature_passes_app_lifecycle_to_the_sign_in():
    from glossarion_mobile.ui.chat.integration import ChatFeature

    seen: list = []
    oauth = types.SimpleNamespace(on_lifecycle=seen.append)
    page = types.SimpleNamespace(platform=None, on_app_lifecycle_state_change=None)
    app = types.SimpleNamespace(page=page, dispatcher=None, paths=None)
    feature = ChatFeature(app, chats=types.SimpleNamespace(flush=lambda: None), jobs=object(), oauth=oauth)
    feature._hook_lifecycle()
    event = types.SimpleNamespace(state=types.SimpleNamespace(value="resume"))  # Flet AppLifecycleStateChangeEvent
    asyncio.run(page.on_app_lifecycle_state_change(event))
    assert seen == ["resume"]


def test_back_in_the_app_after_the_callback_needs_no_notice(mobile_auth):
    opener = Opener()
    bridge = _bridge(mobile_auth, opener, resume_check_delay=0.2)

    async def scenario():
        task = asyncio.ensure_future(bridge.sign_in("authgpt"))
        assert await _wait_for(lambda: opener.opened)
        bridge.on_lifecycle("resume")  # the app thaws and its queued callback is answered right away
        session = bridge._session
        await asyncio.to_thread(_get, _port(session), _callback_path(session))
        result = await asyncio.wait_for(task, TIMEOUT)
        await asyncio.sleep(0.3)
        return result

    assert asyncio.run(scenario())["signed_in"]
    assert bridge.state.step == "done" and not bridge.state.notice


def test_listener_probe_and_the_listener_gone_notice(mobile_auth, monkeypatch):
    opener = Opener()
    bridge = _bridge(mobile_auth, opener, resume_check_delay=0.01)
    dead = types.SimpleNamespace(server_running=True, _servers=[
        types.SimpleNamespace(address_family=socket.AF_INET, server_address=("127.0.0.1", _free_port()))])
    assert OAuthBridge._listener_alive(dead) is False
    assert OAuthBridge._listener_alive(types.SimpleNamespace()) is True  # nothing to probe

    async def scenario():
        task = asyncio.ensure_future(bridge.sign_in("authgpt"))
        assert await _wait_for(lambda: opener.opened)
        assert await asyncio.to_thread(OAuthBridge._listener_alive, bridge._session)
        monkeypatch.setattr(OAuthBridge, "_listener_alive", staticmethod(lambda session: False))
        bridge.on_lifecycle(types.SimpleNamespace(value="resume"))  # Flet's AppLifecycleState
        assert await _wait_for(lambda: bridge.state.notice)
        notice = bridge.state.notice
        bridge.cancel()
        await asyncio.wait_for(task, TIMEOUT)
        return notice

    notice = asyncio.run(scenario())
    assert notice.startswith("Glossarion stopped listening") and "Get a code" not in notice
    closed = types.SimpleNamespace(server_running=False, _servers=dead._servers)
    assert OAuthBridge._listener_alive(closed) is False


def test_stall_notices_only_while_a_loopback_sign_in_waits():
    bridge = OAuthBridge(auth_modules={}, opener=Opener(), safe_root=None)
    for state in (SignInState(step="exchanging"), SignInState(step="done"),
                  SignInState(provider="authgrok", step="waiting", user_code="AB-12")):
        bridge.state = state
        bridge._flag_stalled("waiting")
        assert bridge.state.notice == ""
    bridge.state = SignInState(provider="authcd", step="waiting", manual_url="https://claude.example/code")
    bridge._flag_stalled("waiting")
    assert bridge.state.notice.startswith("Still waiting") and "Get a code to paste instead" in bridge.state.notice
