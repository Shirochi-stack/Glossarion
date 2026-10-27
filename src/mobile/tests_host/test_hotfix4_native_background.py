"""Hotfix 4: the in-app browser routes keep translating while Glossarion is in the background.

The owner's report on hotfix 3: an authchatgpt/ job "just freezes the request entirely when running in
the background. It has not moved from 7/107 in the jobs menu for several hours."

What happened (reproduced on the local emulator): Android's Chromium hides a WebView whose window is
not visible, and the Activity window is not visible while the app is in the background; Blink then
freezes the hidden page a minute or five later (timers and fetch streams stop, scripts still answer).
The reply poll (``authchatgpt_auth._complete``) treated "stream open, nothing new" and "no answer" as
"keep waiting", so the request waited out its whole timeout (10 h with Retry Timeout off) without a
log line, and the sequential job never got past that chapter.

The fix, pinned here:

* the cause (Android): every native page is a ``PageWebView`` that reports its window as visible to
  Chromium (``HiddenWebPages.kt``), so the page is never hidden and never frozen; the CI emulator
  proves it with the self-test suite ``background`` (``ci/android_smoke.sh`` presses HOME);
* the bound (every platform, mobile pages only): ``browser_driver.ReplyWatch`` fails a reply whose page
  answered nothing for ``PAGE_STALL_SECONDS``, ran no timer for as long (the poll's 0 ms probe timer), or
  whose open stream brought nothing for ``STREAM_QUIET_SECONDS``, with a clear error and one log line
  while it waits, so the job's own retry / next chapter runs; time the app itself was paused is not
  counted; desktop CDP pages keep their poll unchanged; ``authds/`` gets the same bound;
* the next requests of the job keep going after a frozen page (no unbounded wait anywhere on the way:
  a Stop while a request waits for a free page ends as the route's "cancelled", and a long wait for a
  page is logged).

Review round (findings R1-R3):

* R1, iOS: WebKit suspends the pages by design while Glossarion is off screen, even while iOS 26+
  continued processing keeps Python running. That time does not count (the page's ``content_suspended``,
  ``PageDriver.pages_suspended``: iOS and the app hidden), so a suspended reply waits for the app, logged
  once with "⏸", instead of failing every attempt of a chapter while the user is away; Android keeps its
  bound;
* R2, authds/: its reply poll logs the "⏳" notice too and treats an unanswered poll like authchatgpt/
  (``watch.unanswered``), reporting "reloaded" only when the page says its state is gone;
* R3, authchatgpt/ handed-off answers (Pro) and generated images: an in-app page's read of the
  conversation is aborted after ``_HANDOFF_READ_MS`` and ``_HANDOFF_READ_FAILURES`` failed reads in a row
  fail the request with the reason (desktop pages read as before).

No sleeps decide anything: the bounds run on a fake clock, or on shortened limits with polling waits.
Real data stays isolated (HOME / USERPROFILE -> tmp_path; GLOSSARION_HTTP_LOG=0). Run from src/mobile:
    python -m pytest -p no:cacheprovider -W ignore tests_host/test_hotfix4_native_background.py
"""

from __future__ import annotations

import ast
import json
import logging
import re
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = MOBILE_DIR / "app"
SRC_DIR = MOBILE_DIR.parent
REPO_DIR = MOBILE_DIR.parents[1]
for entry in (str(APP_DIR), str(SRC_DIR)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import authchatgpt_auth as gpt  # noqa: E402
import authds_auth as ds  # noqa: E402
import browser_driver  # noqa: E402
from glossarion_mobile.diagnostics import selftest, webview_check  # noqa: E402
from glossarion_mobile.services import native_webview as nw  # noqa: E402
from glossarion_mobile.services import webview_bridge as wb  # noqa: E402
from glossarion_mobile.services.native import NativeBridge  # noqa: E402
from glossarion_mobile.services.reader_server import ReaderServer  # noqa: E402

# The hotfix 1 / 3 suites' fakes and fixtures (those files stay unchanged).
from test_hotfix1_authchatgpt_bg import MESSAGES, REPLY, ChatGPTSite  # noqa: E402
from test_hotfix3_native_browser import (  # noqa: E402,F401
    CAPABILITY,
    ChatGPTFrameSite,
    FakeNative,
    _app,
    _chat,
    _clean,
    _install,
    ext_capability,
    harness,
    native_env,
)
from test_webview_bridge import _in_worker, _wait_for, mobile_env, ui_loop  # noqa: E402,F401

EXT_DIR = MOBILE_DIR / "extensions" / "flet_glossarion_native"
KOTLIN_WEB = (EXT_DIR / "src" / "flutter" / "flet_glossarion_native" / "android" / "src" / "main" / "kotlin" / "com"
              / "glossarion" / "flet_glossarion_native" / "HiddenWebPages.kt")
ANDROID_SMOKE = MOBILE_DIR / "ci" / "android_smoke.sh"
BUILD_MOBILE = REPO_DIR / ".github" / "workflows" / "build-mobile.yml"
#: Hotfix 3's own reply poll timeout: 10 h, what ``UnifiedClient.request_timeout`` is with Retry Timeout off.
TEN_HOURS = 36000.0


# ============================================================================ a fake clock for the bounds
class Clock:
    """``time`` for ``authchatgpt_auth`` / ``authds_auth`` / ``browser_driver``: ``sleep`` moves it on."""

    def __init__(self):
        self.now = 1000.0

    def monotonic(self):
        return self.now

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.now += max(0.0, float(seconds))

    def advance(self, seconds):
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    fake = Clock()
    module = types.SimpleNamespace(monotonic=fake.monotonic, time=fake.time, sleep=fake.sleep)
    for target in (gpt, ds, browser_driver):
        monkeypatch.setattr(target, "time", module)
    gpt.reset_cancel()
    yield fake
    gpt.reset_cancel()


class ReplyPage:
    """An in-app chatgpt.com page as ``_complete`` meets it (``driver``: a ``_DriverPage``; without it a
    desktop CDP page). ``poll(n, now)`` gives the n-th ``_POLL`` answer and how many seconds it took."""

    def __init__(self, clock, poll, *, driver=True, probe_cost=10.0):
        self.clock, self.poll, self.probe_cost = clock, poll, probe_cost
        self.polls = self.probes = 0
        if driver:
            self.driver = True
            self.hook_lost = self._hook_lost

    def evaluate(self, expression):
        if expression == gpt._COMPOSER:
            return {"text": ".", "send": True, "requested": True}
        if expression == gpt._POLL:
            self.polls += 1
            state, cost = self.poll(self.polls, self.clock.now)
            self.clock.advance(cost)
            return state
        return True

    def _hook_lost(self):
        self.probes += 1
        self.clock.advance(self.probe_cost)
        return None  # no answer either: not proof that the hook is gone


def state(chunks=(), *, open_=1, paused=0, streams=1):
    return {"chunks": list(chunks), "streams": streams, "open": open_, "idle": 0, "paused": paused, "status": 200,
            "error": None, "kind": None, "sentinel": 0, "resolved": None, "final": None, "finalImages": None}


PART = REPLY.split("\n\n")[0] + "\n\n"  # the reply's first frame: its text starts streaming


def _complete(page, logs, timeout=TEN_HOURS):
    return gpt._complete(page, timeout, None, None, None, log_fn=logs.append)


# ============================================================================ the bound (authchatgpt/)
def test_a_page_frozen_mid_reply_fails_the_request_instead_of_waiting_ten_hours(clock):
    """The owner's request: the reply started, then Android froze the page. Its scripts still answer
    ("stream open, nothing new") but its timers stopped: the poll's probe timer stays pending."""
    frozen_at = []

    def poll(n, now):
        if n <= 3:
            return state([PART] if n == 1 else []), 0.01
        frozen_at.append(frozen_at[0] if frozen_at else now)
        return state(paused=(now - frozen_at[0]) * 1000 + 3000), 0.01

    page, logs = ReplyPage(clock, poll), []
    with pytest.raises(gpt.AuthChatGPTError) as info:
        _complete(page, logs)
    waited = clock.now - frozen_at[0]
    assert info.value.error_type == "browser_error" and not isinstance(info.value, gpt._PageLost)
    assert str(info.value) == ("AuthChatGPT: the in-app ChatGPT page stopped running for 90 s (the system paused it "
                               "while Glossarion was in the background). Partial translation discarded.")
    assert browser_driver.PAGE_STALL_SECONDS <= waited < browser_driver.PAGE_STALL_SECONDS + 2
    notices = [line for line in logs if line.startswith("⏳")]
    assert len(notices) == 1 and "has not run (paused by the system?)" in notices[0] and "up to 90 s" in notices[0]


def test_a_page_that_never_answers_fails_within_the_stall_limit(clock):
    """A frozen renderer answers no script: every poll runs out its 30 s and the hook probe its 10 s."""
    def poll(n, now):
        return (state([PART]), 0.01) if n == 1 else (None, 30.0)

    page, logs = ReplyPage(clock, poll), []
    started = clock.now
    with pytest.raises(gpt.AuthChatGPTError) as info:
        _complete(page, logs)
    assert info.value.error_type == "browser_error" and not isinstance(info.value, gpt._PageLost)
    assert "the in-app ChatGPT page gave no answer for 90 s" in str(info.value)
    assert clock.now - started < 3 * 40.3 + 1 and page.probes == 3  # three silent polls, not 10 h of them
    assert [line for line in logs if line.startswith("⏳")] == [
        "⏳ AuthChatGPT: the in-app ChatGPT page has not answered for 40 s; giving it up to 90 s."]


def test_an_open_stream_that_brings_nothing_fails_after_the_quiet_limit(clock):
    def poll(n, now):
        return state([PART] if n == 1 else []), 0.01  # the page runs (no pause), the server sends nothing

    page, logs = ReplyPage(clock, poll), []
    started = clock.now
    with pytest.raises(gpt.AuthChatGPTError) as info:
        _complete(page, logs)
    assert info.value.error_type == "timeout"
    assert str(info.value) == "AuthChatGPT: chatgpt.com sent nothing for 600 s while answering. Partial translation discarded."
    assert browser_driver.STREAM_QUIET_SECONDS <= clock.now - started < browser_driver.STREAM_QUIET_SECONDS + 2


def test_a_long_reasoning_reply_and_a_busy_page_are_never_cut(clock):
    """Forty minutes of reasoning with a byte every five minutes, and now and then a poll whose probe timer
    waited a few seconds (a busy page): no bound fires, the reply arrives."""
    def poll(n, now):
        minute = (now - 1000.0) / 60.0
        if minute >= 40:
            return state([REPLY], open_=0), 0.01
        data = [":\n\n"] if n % 3000 == 0 else []  # 3000 polls x 0.1 s: every 5 minutes
        return state(data, paused=5000 if n % 700 == 0 else 0), 0.01

    page, logs = ReplyPage(clock, poll), []
    assert _complete(page, logs)["content"] == "Title"
    assert not [line for line in logs if line.startswith("⏳")]


def test_time_the_app_itself_was_paused_is_not_counted(clock):
    """iOS suspends the app, Android may freeze its process: one poll then "took" ten minutes. The page
    answers right after: no stall."""
    def poll(n, now):
        if n == 1:
            return state([PART]), 0.01
        if n == 2:
            return None, 600.0  # suspended while it waited
        if n == 3:
            return state(paused=600000), 0.01  # its probe timer runs right after resuming
        if n < 6:
            return state(), 0.01
        return state([REPLY], open_=0), 0.01

    page, logs = ReplyPage(clock, poll), []
    assert _complete(page, logs)["content"] == "Title" and page.polls == 6


def test_desktop_pages_keep_their_poll(clock):
    """CDP pages (no ``driver`` flag): the same frozen answers wait for the request timeout as before, and
    a missing hook state is a new document at once (``_PageLost``)."""
    def frozen(n, now):
        return state([PART] if n == 1 else [], paused=10 ** 7), 0.01

    page = ReplyPage(clock, frozen, driver=False)
    started = clock.now
    with pytest.raises(gpt.AuthChatGPTError) as info:
        gpt._complete(page, 500.0, None, None, None)
    assert info.value.error_type == "timeout" and "translation timed out" in str(info.value)
    assert clock.now - started >= 500.0
    page = ReplyPage(clock, lambda n, now: (None, 0.01), driver=False)
    with pytest.raises(gpt._PageLost):
        gpt._complete(page, 500.0, None, None, None)
    assert page.polls == 1


def test_a_stall_needs_the_driver_flag_exactly():
    watch = browser_driver.ReplyWatch(False, route="X", site="Y", host="z", error=RuntimeError)
    watch.unanswered(0.0)
    watch.answered(0.0, paused=10 ** 9, streaming=True)
    assert not watch.paused() and watch.silent_since is None
    watch = browser_driver.ReplyWatch(True, route="X", site="Y", host="z", error=RuntimeError)
    watch.answered(watch.poll_started(), paused=True)  # a bool is no milliseconds count
    assert not watch.paused()


def test_authds_gets_the_same_bound(clock):
    """authds/ (DeepSeek) polls its reply on the same in-app pages: a frozen one fails the same way."""
    class DSPage:
        def __init__(self, driver):
            if driver:
                self.driver = True
            self.polls = 0

        def evaluate(self, script):
            if "const cfg =" in script:
                return True
            if script == ds._POLL:
                self.polls += 1
                clock.advance(0.01)
                return {"chunks": [], "done": False, "error": None, "status": 200, "paused": 10 ** 7}
            return None

    started = clock.now
    with pytest.raises(ds.AuthDSError) as info:
        ds._complete(DSPage(True), "authds/flash", MESSAGES, TEN_HOURS, None, None, None)
    assert info.value.error_type == "browser_error" and "the in-app DeepSeek page stopped running for 90 s" in str(info.value)
    assert clock.now - started < browser_driver.PAGE_STALL_SECONDS + 2
    started = clock.now
    with pytest.raises(ds.AuthDSError) as info:  # desktop: the request timeout, as before
        ds._complete(DSPage(False), "authds/flash", MESSAGES, 300.0, None, None, None)
    assert info.value.error_type == "timeout" and clock.now - started >= 300.0


@pytest.mark.parametrize("script", [gpt._POLL, ds._POLL])
def test_the_poll_reports_its_probe_timer(script):
    """``paused`` is the probe timer's wait: scheduled once per run (no pile of timers), 0 once it ran."""
    assert "setTimeout(" in script and script.count("setTimeout(") == 1 and "paused:" in script
    assert "probeAt" in script and "splice(0)" in script


# ============================================================================ through the real native driver
class FrozenAfterReplyStarts(ChatGPTSite):
    """Logged-out chatgpt.com on a native page Android freezes once the reply started (``mode``
    ``paused``: scripts answer, timers stopped; ``silent``: no script answers)."""

    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.reply_polls = 0

    def answer(self, expression):
        if expression == gpt._POLL and self.hooked:
            self.reply_polls += 1
            if self.mode == "silent" and self.reply_polls > 1:
                return None
            return {"chunks": [PART] if self.reply_polls == 1 else [], "streams": 1, "open": 1, "idle": 0,
                    "paused": 0 if self.reply_polls == 1 else 10 ** 7, "status": 200, "error": None, "kind": None,
                    "sentinel": 0, "resolved": None, "final": None, "finalImages": None}
        return super().answer(expression)


@pytest.fixture
def short_stalls(monkeypatch):
    monkeypatch.setattr(browser_driver, "PAGE_STALL_SECONDS", 1.5)
    monkeypatch.setattr(browser_driver, "STALL_NOTICE_SECONDS", 0.2)


@pytest.mark.parametrize("mode", ["paused", "silent"])
def test_a_background_job_keeps_progressing_past_a_frozen_page(harness, short_stalls, mode):
    """A job's chapters, one after another, on native pages: chapter 2's page freezes mid-reply. That
    request fails with the reason (no 10 h wait, no silent spin), its native page is closed, the job's
    retry of chapter 2 and chapter 3 run on new pages and arrive."""
    plan = [ChatGPTSite(), FrozenAfterReplyStarts(mode), ChatGPTSite(), ChatGPTSite()]
    h = harness(lambda: plan.pop(0))
    outcomes, logs = [], []
    for chapter in (1, 2, 2, 3):  # the job retries chapter 2 once, as its normal retry handling does
        thread, box = _in_worker(gpt.send_chat_completion, messages=MESSAGES, model="authchatgpt/auto",
                                 timeout=TEN_HOURS, log_fn=logs.append)
        thread.join(30)
        assert not thread.is_alive(), f"chapter {chapter} is still waiting"
        outcomes.append(box.get("result", {}).get("content") if "result" in box else box["error"])
    assert outcomes[0] == outcomes[2] == outcomes[3] == "Title"
    error = outcomes[1]
    assert isinstance(error, gpt.AuthChatGPTError) and error.error_type == "browser_error"
    assert ("stopped running for 1 s" if mode == "paused" else "gave no answer for 1 s") in str(error)
    assert len(h.fake.created) == 4 and not any("sending the message again" in line for line in logs)
    assert any(line.startswith("⏳ AuthChatGPT: the in-app ChatGPT page") for line in logs)
    frozen = h.fake.created[1]
    assert _wait_for(lambda: frozen.id in h.fake.methods("web_close") and not h.fake.pages)


def test_a_logged_in_reply_in_the_chat_frame_is_bounded_too(harness, short_stalls):
    class FrozenFrame(ChatGPTFrameSite):
        def _in_frame(self, inner):
            if inner == _clean(gpt._POLL) and self.hooked:
                self.polls += 1
                return {"chunks": [], "streams": 1, "open": 1, "idle": 0, "paused": 10 ** 7, "status": 200,
                        "error": None, "kind": None, "sentinel": 0, "resolved": None, "final": None,
                        "finalImages": None}
            return super()._in_frame(inner)

    h = harness(FrozenFrame)
    logs = []
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(gpt, "_access_token", lambda log_fn: "header.e30.signature")
        thread, box = _chat(logs, timeout=TEN_HOURS)
        thread.join(30)
    assert not thread.is_alive()
    assert isinstance(box.get("error"), gpt.AuthChatGPTError) and "stopped running" in str(box["error"])
    assert _wait_for(lambda: not h.fake.pages)


def test_a_stop_while_waiting_for_a_free_page_is_the_routes_cancel(ui_loop, native_env, monkeypatch, caplog):
    """Every page slot in use (batch translation behind a stalled page): the wait is logged once, and a
    Stop ends it as AuthChatGPT's "cancelled", not as a bare BrowserCancelled from the driver."""
    monkeypatch.setattr(wb, "SLOT_NOTICE_SECONDS", 0.05)
    caplog.set_level(logging.INFO, logger="glossarion.webview_bridge")
    fake = FakeNative(ChatGPTSite)
    driver = nw.NativeWebDriver(ui_loop.dispatcher, NativeBridge(native=fake), capability=CAPABILITY,
                                max_pages=1).register()
    try:
        held = driver.open_page(owner="other")
        thread, box = _in_worker(gpt.send_chat_completion, messages=MESSAGES, model="authchatgpt/auto",
                                 timeout=TEN_HOURS, log_fn=lambda line: None)
        assert _wait_for(lambda: "waits for a free page (1 of 1 in use)" in caplog.text)
        gpt.cancel_stream()
        thread.join(10)
        assert not thread.is_alive()
        error = box.get("error")
        assert isinstance(error, gpt.AuthChatGPTError) and error.error_type == "cancelled"
        assert caplog.text.count("waits for a free page") == 1
        held.close()
    finally:
        gpt.reset_cancel()
        driver.shutdown()


# ============================================================================ R1: iOS suspends the pages by design
class InAppPage:
    """A driver page as the routes' ``_DriverPage`` wraps it (``run_js``, ``renderer_gone``,
    ``content_suspended``). ``answer(script, now)`` gives ``(value, seconds it took)``; ``suspends``:
    the page's ``content_suspended`` follows ``hidden`` (iOS), else it is always False (Android)."""

    def __init__(self, clock, answer, *, suspends=True):
        self.clock, self.answer, self.suspends = clock, answer, suspends
        self.hidden = False
        self.renderer_gone = False
        self.scripts = []

    def run_js(self, script, js_timeout_ms=15000):
        self.scripts.append(script)
        value, cost = self.answer(script, self.clock.now)
        self.clock.advance(cost)
        return value

    def content_suspended(self):
        return self.suspends and self.hidden


def _gpt_site(reply_poll):
    """``answer`` for an in-app chatgpt.com page: the composer sends at once, ``reply_poll(n, now)``
    answers the n-th ``_POLL``; the hook probe says "no answer" (None, 10 s)."""
    polls = [0]

    def answer(script, now):
        if script == gpt._COMPOSER:
            return {"text": ".", "send": True, "requested": True}, 0.01
        if script == gpt._POLL:
            polls[0] += 1
            return reply_poll(polls[0], now)
        if "f.contentWindow.__glossarionGPT" in script:  # _DriverPage.hook_lost
            return None, 10.0
        return True, 0.01

    return answer


SUSPENDED_LINE = ("⏸ AuthChatGPT: the system paused the in-app ChatGPT page while Glossarion is in the background; "
                  "the reply continues when you open the app.")


@pytest.mark.parametrize("frozen", ["silent", "paused"])
def test_ios_time_in_the_background_does_not_count(clock, frozen):
    """iOS 26+ keeps the job running in the background, WebKit keeps the page suspended: three hours of
    unanswered (or timer-less) polls fail nothing; the reply arrives once Glossarion is opened again."""
    box = {}

    def reply_poll(n, now):
        page = box["page"]
        if n == 1:
            page.hidden = True  # Glossarion left the screen right after the reply started
            box["hidden_at"] = now
            return state([PART]), 0.01
        if page.hidden and now - box["hidden_at"] >= 3 * 3600:
            page.hidden = False  # the user opens the app
            box["shown_at"] = now
        if page.hidden:
            return (None, 30.0) if frozen == "silent" else (state(paused=(now - box["hidden_at"]) * 1000 + 3000), 0.01)
        return state([REPLY], open_=0), 0.01

    page = box["page"] = InAppPage(clock, _gpt_site(reply_poll))
    logs = []
    result = gpt._complete(gpt._DriverPage(page), TEN_HOURS, None, None, None, log_fn=logs.append)
    assert result["content"] == "Title" and "shown_at" in box
    assert logs == [SUSPENDED_LINE]  # once, and no stall notice: the suspended time never counted


def test_ios_a_page_still_stopped_after_the_app_returns_fails_within_the_bound(clock):
    box = {}

    def reply_poll(n, now):
        page = box["page"]
        if n == 1:
            page.hidden = True
            box["hidden_at"] = now
            return state([PART]), 0.01
        if page.hidden and now - box["hidden_at"] >= 3600:
            page.hidden = False
            box.setdefault("shown_at", now)
        return state(paused=(now - box["hidden_at"]) * 1000 + 3000), 0.01  # it never runs again

    page = box["page"] = InAppPage(clock, _gpt_site(reply_poll))
    logs = []
    with pytest.raises(gpt.AuthChatGPTError) as info:
        gpt._complete(gpt._DriverPage(page), TEN_HOURS, None, None, None, log_fn=logs.append)
    assert "stopped running for 90 s" in str(info.value) and info.value.error_type == "browser_error"
    after = clock.now - box["shown_at"]
    assert browser_driver.PAGE_STALL_SECONDS <= after < browser_driver.PAGE_STALL_SECONDS + 2
    assert logs[0] == SUSPENDED_LINE and len([line for line in logs if line.startswith("⏳")]) == 1


def test_android_keeps_its_bound_while_the_app_is_hidden(clock):
    """The same frozen page on Android (no suspension by design: a frozen page is a fault)."""
    box = {}

    def reply_poll(n, now):
        if n == 1:
            box["page"].hidden = True
            box["hidden_at"] = now
            return state([PART]), 0.01
        return state(paused=(now - box["hidden_at"]) * 1000 + 3000), 0.01

    page = box["page"] = InAppPage(clock, _gpt_site(reply_poll), suspends=False)
    logs = []
    with pytest.raises(gpt.AuthChatGPTError) as info:
        gpt._complete(gpt._DriverPage(page), TEN_HOURS, None, None, None, log_fn=logs.append)
    assert "stopped running for 90 s" in str(info.value)
    assert clock.now - box["hidden_at"] < browser_driver.PAGE_STALL_SECONDS + 2
    assert SUSPENDED_LINE not in logs


def test_ios_a_reply_that_had_not_started_is_no_usage_limit_after_the_app_returns(clock):
    """The message went out, WebKit suspended the page before the reply stream started; ten minutes later
    the app is opened and the stream starts a few seconds after: no "did not start a reply" verdict
    counted from before the suspension."""
    box = {}

    def reply_poll(n, now):
        page = box["page"]
        if n == 1:
            page.hidden = True
            box["hidden_at"] = now
        if page.hidden and now - box["hidden_at"] >= 600:
            page.hidden = False
            box["shown_at"] = now
        if page.hidden:
            return None, 30.0
        if now - box["shown_at"] < 5:
            return state(open_=0, streams=0), 0.01  # the fetch is only starting
        return state([REPLY], open_=0), 0.01

    page = box["page"] = InAppPage(clock, _gpt_site(reply_poll))
    assert gpt._complete(gpt._DriverPage(page), TEN_HOURS, None, None, None, log_fn=lambda line: None)["content"] == "Title"


def test_authds_waits_for_the_app_on_ios_too(clock):
    box = {}

    def answer(script, now):
        page = box["page"]
        if script == ds._POLL:
            box["polls"] = box.get("polls", 0) + 1
            if box["polls"] == 1:
                page.hidden = True
                box["hidden_at"] = now
            if page.hidden and now - box["hidden_at"] >= 2 * 3600:
                page.hidden = False
            if page.hidden:
                return None, 30.0
            return {"chunks": [DS_REPLY], "done": True, "error": None, "status": 200, "paused": 0}, 0.01
        if script == "!window.__glossarionDS":
            return None, 10.0  # no answer: no proof the state is gone
        return True, 0.01

    page = box["page"] = InAppPage(clock, answer)
    logs = []
    result = ds._complete(ds._DriverPage(page), "authds/flash", MESSAGES, TEN_HOURS, None, None, None,
                          log_fn=logs.append)
    assert result["content"] == "Hello"
    assert logs == ["⏸ AuthDS: the system paused the in-app DeepSeek page while Glossarion is in the background; "
                    "the reply continues when you open the app."]


@pytest.mark.parametrize("ios", [True, False])
def test_a_generated_image_download_waits_for_the_app_on_ios_only(clock, ios):
    """The in-app image download cap (``_DRIVER_IMAGE_SECONDS``) does not count suspended time either."""
    box = {"start": clock.now}

    def answer(script, now):
        if "s.download(" in script:
            page = box["page"]
            page.hidden = now - box["start"] < 3600  # an hour in the background, then the app is opened
            return ({"pending": True} if page.hidden else {"b64": "aGk=", "mime": "image/png"}), 1.0
        return True, 0.01

    page = box["page"] = InAppPage(clock, answer, suspends=ios)
    wrapped = gpt._DriverPage(page)
    if ios:
        assert gpt._download_image(wrapped, "sediment://file_1", clock.now + TEN_HOURS, None) == {
            "mime": "image/png", "data": b"hi"}
        return
    with pytest.raises(gpt.AuthChatGPTError, match="timed out downloading the generated image"):
        gpt._download_image(wrapped, "sediment://file_1", clock.now + TEN_HOURS, None)
    assert gpt._DRIVER_IMAGE_SECONDS <= clock.now - box["start"] < gpt._DRIVER_IMAGE_SECONDS + 3


def test_the_watch_restarts_its_clocks_only_while_hidden(clock):
    hidden = [False]
    seen = []
    watch = browser_driver.ReplyWatch(True, route="R", site="S", host="h", error=lambda m, k: RuntimeError(m),
                                      log_fn=seen.append, hidden=lambda: hidden[0])
    started = clock.now
    for _ in range(10):  # 10 x 30 s with no answer while hidden: nothing fails
        hidden[0] = True
        asked = watch.poll_started()
        clock.advance(30)
        watch.unanswered(asked)
    assert watch.running_since(started) == 0 and len(seen) == 1 and seen[0].startswith("⏸ R:")
    hidden[0] = False
    with pytest.raises(RuntimeError, match="gave no answer for 90 s"):
        for _ in range(10):
            asked = watch.poll_started()
            clock.advance(30)
            watch.unanswered(asked)
    broken = browser_driver.ReplyWatch(True, route="R", site="S", host="h", error=RuntimeError,
                                       hidden=lambda: 1 / 0)  # a failing probe is "not suspended"
    assert broken._suspended() is False
    desktop = browser_driver.ReplyWatch(False, route="R", site="S", host="h", error=RuntimeError,
                                        hidden=lambda: True)
    assert desktop.poll_started() == clock.now and desktop.resumed is None
    clock.advance(70)
    assert desktop.running_since(started) == clock.now - started


def test_pages_are_suspended_only_on_ios_with_the_app_hidden(harness):
    h = harness(lambda: ChatGPTSite())
    on_screen = [True]
    h.driver.app_on_screen = lambda: on_screen[0]
    page = h.driver.open_page(owner="probe")
    try:
        wrapped_gpt, wrapped_ds = gpt._DriverPage(page), ds._DriverPage(page)
        for ios in (False, True):
            h.driver.suspends_hidden_pages = ios
            for shown in (True, False):
                on_screen[0] = shown
                expected = ios and not shown
                assert h.driver.pages_suspended() is expected and page.content_suspended() is expected
                assert wrapped_gpt.content_suspended() is expected and wrapped_ds.content_suspended() is expected
        h.driver.app_on_screen = lambda: 1 / 0  # never raises out of a page
        assert page.content_suspended() is False
        h.driver.app_on_screen = None  # no app (self-test, host): never suspended
        assert h.driver.pages_suspended() is False
    finally:
        page.close()
    bridge = wb.WebViewBridge(None, types.SimpleNamespace(page=types.SimpleNamespace(app_visible=False)))
    assert bridge.pages_suspended() is False  # Android / desktop
    bridge.suspends_hidden_pages = True
    assert bridge.pages_suspended() is True
    bridge.host.page.app_visible = True
    assert bridge.pages_suspended() is False
    assert browser_driver.content_suspended(object()) is False  # a desktop page has no such method


@pytest.mark.parametrize("platform,expected", [("ios", True), ("android", False), ("", False)])
def test_install_marks_ios_pages_as_suspended_in_the_background(ui_loop, native_env, ext_capability,
                                                                 platform, expected):
    if platform:
        native_env.setenv("GLOSSARION_SIMULATE_PLATFORM", platform)
    assert wb.suspends_hidden_pages() is expected
    fake = FakeNative(ChatGPTSite)
    app = _app(ui_loop, NativeBridge(native=fake))
    driver = _install(ui_loop, app)
    try:
        assert isinstance(driver, nw.NativeWebDriver) and driver.suspends_hidden_pages is expected
    finally:
        driver.shutdown()
    source = (APP_DIR / "glossarion_mobile" / "services" / "webview_bridge.py").read_text(encoding="utf-8")
    assert "bridge.suspends_hidden_pages = suspends_hidden_pages()" in source  # the flet-webview fallback too


# ============================================================================ R2: authds/ like authchatgpt/
DS_REPLY = ('data: {"p": "response/fragments", "o": "APPEND", "v": [{"type": "RESPONSE", "content": "Hello"}]}\n\n'
            'data: {"p": "response/status", "o": "SET", "v": "FINISHED"}\n\n')


def _ds_page(clock, *, gone=None, renderer_gone=False):
    """An in-app DeepSeek page whose reply polls never answer; the state probe answers ``gone``."""
    probes = []

    def answer(script, now):
        if script == ds._POLL:
            return None, 30.0
        if script == "!window.__glossarionDS":
            probes.append(now)
            return gone, 10.0
        return True, 0.01

    page = InAppPage(clock, answer, suspends=False)
    page.renderer_gone = renderer_gone
    return page, probes


def test_authds_an_unanswered_poll_is_bounded_and_logged(clock):
    page, probes = _ds_page(clock)
    logs, started = [], clock.now
    with pytest.raises(ds.AuthDSError) as info:
        ds._complete(ds._DriverPage(page), "authds/flash", MESSAGES, TEN_HOURS, None, None, None, log_fn=logs.append)
    assert info.value.error_type == "browser_error" and "the in-app DeepSeek page gave no answer for 90 s" in str(info.value)
    assert clock.now - started < 3 * 40.3 + 1 and len(probes) == 3
    assert logs == ["⏳ AuthDS: the in-app DeepSeek page has not answered for 40 s; giving it up to 90 s."]


@pytest.mark.parametrize("case", ["state_gone", "renderer_gone"])
def test_authds_a_page_that_lost_its_state_is_still_reported_as_reloaded(clock, case):
    page, probes = _ds_page(clock, gone=True, renderer_gone=case == "renderer_gone")
    with pytest.raises(ds.AuthDSError) as info:
        ds._complete(ds._DriverPage(page), "authds/flash", MESSAGES, TEN_HOURS, None, None, None)
    assert str(info.value) == "AuthDS page was reloaded during translation." and info.value.error_type == "browser_error"
    assert len(probes) == (0 if case == "renderer_gone" else 1)


def test_authds_desktop_page_without_state_is_reloaded_at_once(clock):
    class DesktopPage:
        def evaluate(self, script):
            return None if script == ds._POLL else True

    with pytest.raises(ds.AuthDSError, match="reloaded during translation"):
        ds._complete(DesktopPage(), "authds/flash", MESSAGES, 300.0, None, None, None)


@pytest.mark.parametrize("driver", [True, False])
def test_authds_passes_its_log_to_the_reply_watch_on_in_app_pages_only(monkeypatch, driver):
    from contextlib import contextmanager

    class Page:
        def evaluate(self, script):
            return True

    page = Page()
    if driver:
        page.driver = True
    calls = []

    @contextmanager
    def browser(**kwargs):
        yield page

    def complete(*args, **kwargs):
        calls.append(kwargs)
        return {"content": "ok"}

    monkeypatch.setattr(ds, "_open_browser", browser)
    monkeypatch.setattr(ds, "has_session", lambda: True)
    monkeypatch.setattr(ds, "_complete", complete)
    log_fn = [].append
    assert ds.send_chat_completion(messages=MESSAGES, model="authds/flash", log_fn=log_fn)["content"] == "ok"
    assert calls == ([{"log_fn": log_fn}] if driver else [{}])  # desktop: the call as before


# ============================================================================ R3: reads of a handed-off answer
def _frame(value, event=None):
    return (f"event: {event}\n" if event else "") + "data: " + (value if isinstance(value, str) else json.dumps(value)) + "\n\n"


HANDOFF = (_frame('"v1"', "delta_encoding")
           + _frame({"v": {"message": {"id": "t", "author": {"role": "tool"}, "content": {"content_type": "text", "parts": [""]},
                                       "recipient": "all", "status": "in_progress", "metadata": {}}}}, "delta")
           + _frame({"type": "stream_handoff", "conversation_id": "c", "options": []}) + _frame("[DONE]"))


class HandoffPage:
    """A page whose reply was handed off (Pro): every read of the conversation fails with ``failure``
    (``HTTP 404``, ``no answer in time``, ...), as the hook counts it."""

    def __init__(self, clock, failure, *, driver=True, image=False):
        self.clock, self.failure, self.image = clock, failure, image
        self.reads, self.polls = [], 0
        if driver:
            self.driver = True

    def evaluate(self, script):
        if script == gpt._COMPOSER:
            return {"text": ".", "send": True, "requested": True}
        if script == gpt._POLL:
            self.polls += 1
            self.clock.advance(0.01)
            status = int(self.failure[5:]) if self.failure.startswith("HTTP ") else 0
            return dict(state([HANDOFF] if self.polls == 1 else [], open_=0), pollFailures=len(self.reads),
                        pollFailure=self.failure if self.reads else None, pollStatus=status)
        if script in (gpt._POLL_ANSWER, gpt._POLL_ANSWER_TIMED):
            self.reads.append(script)
            return True
        return True


def test_the_handoff_stream_is_a_handoff():
    parser = gpt.StreamParser()
    parser.feed(HANDOFF)
    assert parser.handoff and not parser.finished


@pytest.mark.parametrize("failure,kind", [("HTTP 404", "api_error"), ("HTTP 401", "auth_error"),
                                          ("HTTP 429", "rate_limit"), ("no answer in time", "timeout"),
                                          ("network error", "browser_error"), ("<script>", "browser_error")])
def test_failed_reads_of_a_handed_off_answer_fail_the_request(clock, failure, kind):
    page, notes = HandoffPage(clock, failure), []
    started = clock.now
    with pytest.raises(gpt.AuthChatGPTError) as info:
        gpt._complete(page, TEN_HOURS, None, None, None, notes.append, log_fn=lambda line: None)
    shown = failure if failure != "<script>" else "read failed"  # only the hook's own words are shown
    assert str(info.value) == f"AuthChatGPT: ChatGPT's answer could not be read ({shown}, 6 times in a row)."
    assert info.value.error_type == kind and not isinstance(info.value, gpt._PageLost)
    assert len(page.reads) == gpt._HANDOFF_READ_FAILURES and set(page.reads) == {gpt._POLL_ANSWER_TIMED}
    assert clock.now - started < gpt._HANDOFF_READ_FAILURES * gpt._HANDOFF_POLL_SECONDS + 2
    assert len(notes) == 1 and "background" in notes[0]


def test_failed_reads_of_a_generated_image_fail_the_request(clock):
    page = HandoffPage(clock, "no conversation")
    with pytest.raises(gpt.AuthChatGPTError, match=r"could not be read \(no conversation, 6 times in a row\)"):
        gpt._complete(page, TEN_HOURS, None, None, None, None, image_output=True, log_fn=lambda line: None)
    assert set(page.reads) == {gpt._POLL_ANSWER_TIMED}


def test_desktop_reads_of_a_handed_off_answer_are_unchanged(clock):
    page = HandoffPage(clock, "HTTP 404", driver=False)
    started = clock.now
    with pytest.raises(gpt.AuthChatGPTError) as info:
        gpt._complete(page, 300.0, None, None, None, None)
    assert info.value.error_type == "timeout" and clock.now - started >= 300.0
    assert set(page.reads) == {gpt._POLL_ANSWER} and len(page.reads) >= 29


def test_the_hook_bounds_a_read_and_counts_failed_reads(tmp_path):
    """The real hook in Node: a read that never answers is aborted after ``limitMs`` (``polling`` does not
    stick), HTTP errors and a missing conversation count up, a good read sets the count back to 0, and a
    read without a limit (desktop) is not aborted. The new scripts are single expressions."""
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is needed to run the page hook")
    script = tmp_path / "hook.js"
    script.write_text(r"""
global.location = {href:'https://chatgpt.com/', origin:'https://chatgpt.com'};
global.window = global;
let mode = 'hang', aborted = 0;
global.fetch = (url, init) => {
 if (!String(url).includes('/conversation/')) return Promise.resolve(new Response('{}', {status:200}));
 if (mode === 'hang') return new Promise((resolve, reject) => {
  if (init && init.signal) init.signal.addEventListener('abort', () => { aborted += 1; reject(new DOMException('aborted', 'AbortError')); });
 });
 if (mode === '404') return Promise.resolve(new Response('{}', {status:404}));
 return Promise.resolve(new Response(JSON.stringify({mapping:{a:{message:{author:{role:'assistant'}, recipient:'all',
  status:'finished_successfully', content:{content_type:'text', parts:['Done']}, metadata:{}}}}}), {status:200}));
};
%s;
const s = window.__glossarionGPT;
const settle = async () => { for (let i = 0; i < 400 && s.polling; i++) await new Promise(r => setTimeout(r, 5)); };
(async () => {
 const out = {};
 out.noConversation = [s.poll(50), s.pollFailures, s.pollFailure];
 s.conversationId = 'c';
 out.started = s.poll(50); out.again = s.poll(50);
 await settle();
 out.hang = [s.polling, s.pollFailures, s.pollFailure, aborted];
 mode = '404'; s.poll(50); await settle();
 out.http = [s.pollFailures, s.pollFailure, s.pollStatus];
 mode = 'ok'; s.poll(50); await settle();
 out.ok = [s.pollFailures, s.pollFailure, s.final && s.final.text];
 mode = 'hang'; s.poll(); await new Promise(r => setTimeout(r, 120));
 out.unbounded = [s.polling, aborted];
 const scripts = %s;
 for (const x of scripts) new Function('var value = (' + String.fromCharCode(10) + x + String.fromCharCode(10) + ');');
 out.scripts = scripts.length;
 console.log(JSON.stringify(out));
 process.exit(0);
})();
""" % (gpt._hook({"token": "tok", "prompt": "p", "model": "auto", "temporary": True}),
       json.dumps([gpt._POLL, gpt._POLL_ANSWER, gpt._POLL_ANSWER_TIMED])), encoding="utf-8")
    out = subprocess.run([node, str(script)], capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr[-2000:]
    result = json.loads(out.stdout)
    assert result["noConversation"] == [False, 1, "no conversation"]
    assert result["started"] is True and result["again"] is False  # one read at a time
    assert result["hang"] == [False, 2, "no answer in time", 1]
    assert result["http"] == [3, "HTTP 404", 404]
    assert result["ok"] == [0, None, "Done"]
    assert result["unbounded"] == [True, 1]  # desktop's read has no limit, as before
    assert result["scripts"] == 3


# ============================================================================ the cause (Android)
def _kotlin_code():
    text = KOTLIN_WEB.read_bytes().decode("utf-8").replace("\r\n", "\n")
    return text, re.sub(r"/\*.*?\*/|//[^\n]*", "", text, flags=re.S)


def test_every_native_page_reports_a_visible_window_to_chromium():
    """AwContents hides a page whose window is not visible, and Blink freezes a hidden page: the native
    pages are PageWebViews whose window visibility is always VISIBLE, also when they are attached to a
    stopped Activity (View.dispatchAttachedToWindow sends no visibility for a window that is not shown)."""
    text, code = _kotlin_code()
    cls = code[code.index("private class PageWebView(context: Context) : WebView(context) {"):]
    cls = cls[:cls.index("\n    }\n") + 6]
    assert '@SuppressLint("ViewConstructor")\n    private class PageWebView' in text
    attached = cls[cls.index("override fun onAttachedToWindow() {"):cls.index("override fun onWindowVisibilityChanged")]
    assert attached.index("super.onAttachedToWindow()") < attached.index("super.onWindowVisibilityChanged(View.VISIBLE)")
    visibility = cls[cls.index("override fun onWindowVisibilityChanged(visibility: Int) {"):]
    assert "super.onWindowVisibilityChanged(View.VISIBLE)" in visibility
    assert "super.onWindowVisibilityChanged(visibility)" not in cls
    open_fn = code[code.index("private fun openPage("):code.index("private fun configure(")]
    assert "PageWebView(wrapper)" in open_fn and re.search(r"\bWebView\(wrapper\)", open_fn.replace("PageWebView", "")) is None
    assert "onPause" not in code and "pauseTimers" not in code  # nothing else pauses the pages
    # The class sits before the companion object (the per-instance registry rule of the critic tests).
    assert code.index("private class PageWebView") < code.index("companion object")


# ============================================================================ the CI emulator check (suite ``background``)
class BackgroundSite:
    """The background self-test's loopback page with a simulated page clock (3 s per look). ``fixed``
    False: Chromium hides the page while the app is hidden and its timers stop (hotfix 3)."""

    title = webview_check.BG_TITLE

    def __init__(self, app, fixed=True):
        self.app, self.fixed = app, fixed
        self.clock = 5_000_000.0
        self.started = None
        self.frozen_at = None

    def navigated(self, url):
        self.started = self.frozen_at = None

    def _series(self, step, until):
        out, t = [], self.started + step
        while t <= until:
            out.append(t)
            t += step
        return out

    def answer(self, expression):
        if expression == _clean(webview_check.BG_START_SCRIPT):
            self.started = self.clock
            return True
        if expression == _clean(webview_check.BG_STATE_SCRIPT):
            if self.started is None:
                return None
            self.clock += 3000.0
            hidden = not self.app.page.app_visible
            if hidden and not self.fixed and self.frozen_at is None:
                self.frozen_at = self.clock
            until = self.frozen_at if self.frozen_at is not None else self.clock
            return {"now": self.clock, "visibility": "hidden" if (hidden and not self.fixed) else "visible",
                    "beats": self._series(1000.0, until), "fetches": self._series(2000.0, until), "failures": 0,
                    "hidden": 0}
        return None


class ServiceState:
    """The job foreground service the check holds (``NativeBridge.*_job_service`` on the fake)."""

    def __init__(self, fake, *, starts=True):
        self.running, self.starts, self.stops, self.can_start = False, 0, 0, starts
        fake.is_job_service_running = self.is_running
        fake.start_job_service = self.start
        fake.stop_job_service = self.stop
        fake.update_job_service = self.update

    async def is_running(self):
        return self.running

    async def start(self, title, text, buttons=None):
        self.starts += 1
        self.running = self.can_start
        return self.can_start

    async def stop(self):
        self.stops += 1
        self.running = False

    async def update(self, title=None, text=None):
        return None


def _background_run(h, *, fixed=True, starts=True, home=True, **kwargs):
    app = types.SimpleNamespace(page=types.SimpleNamespace(app_visible=True))
    h.driver.app_on_screen = lambda: app.page.app_visible
    h.fake.site_factory = lambda: BackgroundSite(app, fixed)
    service = ServiceState(h.fake, starts=starts)
    markers = []

    def emit(name, payload):
        markers.append((name, payload))
        if name == webview_check.MARKER_BG_READY and home:
            app.page.app_visible = False  # the smoke script presses HOME

    kwargs.setdefault("seconds", 0)
    kwargs.setdefault("hide_wait", 2)
    kwargs.setdefault("poll", 0)
    box = {}
    try:
        box["detail"] = webview_check.run_background(h.driver, emit=emit, server_factory=ReaderServer,
                                                     sleep=lambda seconds: None, **kwargs)
    except AssertionError as exc:
        box["error"] = str(exc)
    return box, markers, service


def test_background_check_passes_while_the_page_keeps_running(harness):
    h = harness(lambda: None)
    box, markers, service = _background_run(h)
    assert "error" not in box, box.get("error")
    assert [name for name, _ in markers] == [webview_check.MARKER_BG_READY, webview_check.MARKER_BG]
    result = markers[1][1]
    assert result["visibility"] == ["visible"] and result["beats"] >= 2 and result["max_gap_ms"] == 1000
    assert result["fetches"] >= 1 and result["came_back"] is False
    assert service.starts == 1 and service.stops == 1 and not service.running  # held like a job, then released
    assert _wait_for(lambda: not h.fake.pages)


def test_background_check_fails_when_chromium_hides_and_freezes_the_page(harness):
    """Hotfix 3's pages: hidden in the background, timers stopped. The check fails with the reason, and
    the measurement line still goes out for the CI log."""
    h = harness(lambda: None)
    box, markers, service = _background_run(h, fixed=False)
    assert "Chromium treated the hidden page as hidden (document.visibilityState)" in box["error"]
    assert markers[-1][0] == webview_check.MARKER_BG and markers[-1][1]["visibility"] == ["hidden"]
    assert service.stops == 1 and _wait_for(lambda: not h.fake.pages)


@pytest.mark.parametrize("case,expected", [
    ("never_hidden", "Glossarion did not leave the screen within 0 s"),
    ("no_service", "the foreground service did not start"),
    ("started_hidden", "start the background check with Glossarion on screen"),
])
def test_background_check_refuses_what_proves_nothing(harness, case, expected):
    h = harness(lambda: None)
    if case == "started_hidden":
        h.driver.app_on_screen = lambda: False
        with pytest.raises(AssertionError) as info:
            webview_check.run_background(h.driver, emit=lambda *a: None, server_factory=ReaderServer,
                                         sleep=lambda seconds: None)
        assert expected in str(info.value)
        return
    box, markers, service = _background_run(h, home=case != "never_hidden", starts=case != "no_service",
                                            hide_wait=0)
    assert expected in box["error"]
    assert not service.running and _wait_for(lambda: not h.fake.pages)


@pytest.mark.parametrize("result,problem", [
    ({"visibility": ["visible"], "max_gap_ms": 1000, "fetches": 30, "seconds": 75.0, "came_back": False}, None),
    ({"visibility": ["visible"], "max_gap_ms": 61000, "fetches": 30, "seconds": 75.0, "came_back": False},
     "timers stopped for 61 s"),
    ({"visibility": ["visible"], "max_gap_ms": 1000, "fetches": 3, "seconds": 75.0, "came_back": False},
     "fetches stopped"),
    ({"visibility": ["hidden", "visible"], "max_gap_ms": 1000, "fetches": 30, "seconds": 75.0, "came_back": False},
     "as hidden, visible"),
    ({"visibility": ["visible"], "max_gap_ms": 1000, "fetches": 30, "seconds": 75.0, "came_back": True},
     "came back on screen"),
])
def test_background_result_rules(result, problem):
    if problem is None:
        webview_check.check_background_result(result)
        return
    with pytest.raises(AssertionError) as info:
        webview_check.check_background_result(result)
    assert problem in str(info.value)


def test_background_result_measures_inside_the_hidden_window():
    start = {"now": 10_000.0, "failures": 1}
    end = {"now": 20_000.0, "failures": 3, "beats": [9_000.0, 11_000.0, 12_000.0, 19_000.0],
           "fetches": [8_000.0, 12_000.0, 18_000.0]}
    result = webview_check.background_result(start, end, ["visible", "visible"], seconds=10.0, came_back=False)
    assert result == {"seconds": 10.0, "page_ms": 10000, "beats": 3, "fetches": 2, "max_gap_ms": 7000,
                      "fetch_failures": 2, "visibility": ["visible"], "came_back": False}


def test_the_background_suite_runs_on_android_devices_only(monkeypatch, tmp_path):
    assert selftest.SUITES["background"] == (("native_webview_background", selftest.check_native_webview_background),)
    assert "native_webview_background" not in dict(selftest.SUITES["smoke"])  # needs HOME: CI's own suite
    monkeypatch.delenv("GLOSSARION_SIMULATE_PLATFORM", raising=False)
    monkeypatch.delenv(nw.ENV_SWITCH, raising=False)
    calls = []
    monkeypatch.setattr(webview_check, "run_background", lambda driver, emit: calls.append(driver) or {"ok": 1})
    for platform, is_device, status in (("desktop", False, "skip"), ("ios", True, "skip"), ("android", True, "pass")):
        paths = types.SimpleNamespace(platform=platform, is_device=is_device, logs=tmp_path, home=tmp_path, data=tmp_path)
        monkeypatch.setattr(selftest.rb, "get_state", lambda paths=paths: types.SimpleNamespace(paths=paths))
        driver = types.SimpleNamespace(name=nw.NATIVE_NAME)
        monkeypatch.setattr(webview_check, "wait_for_native_driver", lambda timeout=None, d=driver: (d, ""))
        result = selftest.run_selftest("background", emit=False, write_report=False)
        assert result["checks"][0]["status"] == status, (platform, result["checks"][0])
    assert len(calls) == 1
    monkeypatch.setattr(webview_check, "wait_for_native_driver", lambda timeout=None: (None, "no in-app browser"))
    result = selftest.run_selftest("background", emit=False, write_report=False)
    assert result["ok"] is False and "native in-app browser is not installed" in result["checks"][0]["error"]


def test_install_gives_the_driver_the_apps_visibility(monkeypatch):
    app = types.SimpleNamespace(page=types.SimpleNamespace(app_visible=True))
    assert nw._app_on_screen(app) is True
    app.page.app_visible = False
    assert nw._app_on_screen(app) is False
    assert nw._app_on_screen(types.SimpleNamespace()) is True  # no page: never "hidden"
    source = (APP_DIR / "glossarion_mobile" / "services" / "native_webview.py").read_text(encoding="utf-8")
    assert "driver.app_on_screen = functools.partial(_app_on_screen, app)" in source


def _smoke():
    return ANDROID_SMOKE.read_bytes().decode("utf-8")


def test_the_android_smoke_presses_home_during_the_background_suite():
    script = _smoke()
    assert "\r\n" not in script
    assert 'RUN_BG_WEB="${RUN_BG_WEB:-$RUN_E2E}"' in script and 'BG_WEB_TIMEOUT="${BG_WEB_TIMEOUT:-360}"' in script
    assert 'BG_WEB_URL="${BG_WEB_URL:-glossarion://app/__selftest__?suite=background}"' in script
    body = script[script.index("run_background_suite() {"):]
    body = body[:body.index("\n}\n")]
    steps = [body.index(s) for s in ('start_suite background "$BG_WEB_URL"', "wait_for_marker GLOSSARION_BGWEB_READY",
                                     "adb shell input keyevent KEYCODE_HOME", 'wait_suite background "$BG_WEB_TIMEOUT"',
                                     "bg_web_result.txt", "monkey -p \"$PKG\"")]
    assert steps == sorted(steps)
    assert webview_check.MARKER_BG_READY == "GLOSSARION_BGWEB_READY" and webview_check.MARKER_BG == "GLOSSARION_BGWEB"
    # after the end-to-end suite, before the llama suite, and only when asked
    order = [script.index(s) for s in ('run_suite e2e "$E2E_URL"', 'if [ "$RUN_BG_WEB" = "1" ]; then\n  run_background_suite',
                                       'run_suite llama "$LLAMA_URL"')]
    assert order == sorted(order)
    # every suite still blocks on its own PASS line (the hotfix 3 contract test reads run_suite smoke)
    assert re.search(r'^run_suite smoke "\$SELFTEST_URL"', script, re.M)
    assert 'pass_line="GLOSSARION_SELFTEST PASS {\\"suite\\":\\"$suite\\""' in script


def test_the_emulator_job_runs_the_background_suite_and_the_llama_step_does_not():
    if not BUILD_MOBILE.exists():
        pytest.skip("no .github in this tree")
    yaml = pytest.importorskip("yaml")
    workflow = yaml.safe_load(BUILD_MOBILE.read_text(encoding="utf-8"))
    scripts = [str(s.get("with", {}).get("script", "")) for s in workflow["jobs"]["android-smoke"]["steps"]
               if "android_smoke.sh" in str(s.get("with", {}).get("script", ""))]
    main = [s for s in scripts if "RUN_LLAMA=1" not in s]
    llama = [s for s in scripts if "RUN_LLAMA=1" in s]
    assert len(main) == 1 and "RUN_E2E=0" not in main[0] and "RUN_BG_WEB=0" not in main[0]
    assert all("RUN_E2E=0" in s and "RUN_BG_WEB=1" not in s for s in llama)


# ============================================================================ hygiene
CHANGED = [
    SRC_DIR / "authchatgpt_auth.py", SRC_DIR / "authds_auth.py", SRC_DIR / "browser_driver.py",
    APP_DIR / "glossarion_mobile" / "services" / "native_webview.py",
    APP_DIR / "glossarion_mobile" / "services" / "webview_bridge.py",
    APP_DIR / "glossarion_mobile" / "diagnostics" / "webview_check.py",
    APP_DIR / "glossarion_mobile" / "diagnostics" / "selftest.py",
    APP_DIR / "glossarion_mobile" / "services" / "native.py",
    APP_DIR / "glossarion_mobile" / "ui" / "shell" / "app_shell.py",
    APP_DIR / "glossarion_mobile" / "app.py",
    EXT_DIR / "src" / "flet_glossarion_native" / "native.py",
    Path(__file__),
    Path(__file__).with_name("test_hotfix4_back_keeps_jobs.py"),
]


@pytest.mark.parametrize("path", CHANGED, ids=lambda p: p.name)
def test_uniform_line_endings_and_python_310_syntax(path):
    data = path.read_bytes()
    assert data.count(b"\r\n") in (0, data.count(b"\n")) and not data.startswith(b"\xef\xbb\xbf")
    ast.parse(data.decode("utf-8"), filename=path.name, feature_version=(3, 10))


def test_kotlin_and_smoke_script_keep_their_line_endings():
    kotlin_plugin = KOTLIN_WEB.with_name("GlossarionNativePlugin.kt")
    dart_service = EXT_DIR / "src" / "flutter" / "flet_glossarion_native" / "lib" / "src" / "native_service.dart"
    for path in (KOTLIN_WEB, ANDROID_SMOKE, kotlin_plugin, dart_service):
        data = path.read_bytes()
        assert data.count(b"\r\n") in (0, data.count(b"\n")) and not data.startswith(b"\xef\xbb\xbf"), path.name


def test_the_stall_lines_carry_no_page_content():
    """Errors and notices name the route, the site and seconds only (never a script, a prompt or a URL)."""
    seen = []
    watch = browser_driver.ReplyWatch(True, route="AuthChatGPT", site="ChatGPT", host="chatgpt.com",
                                      error=lambda message, kind: RuntimeError(message), log_fn=seen.append)
    for kind in ("silent", "paused", "quiet"):
        text = str(watch.error(kind, 90))
        assert re.fullmatch(r"[\w .:,()?/-]+", text), text
        assert "http" not in text and "prompt" not in text
