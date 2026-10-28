"""Host tests for the hotfix 3 in-app browser page API of flet-glossarion-native.

* The Python service over a fake Flet channel: safe defaults, argument marshalling, timeouts, result
  normalisation, and privacy (no script text in answers or logs).
* ``web_pages.py`` helpers and the ``WebPageEvent`` payloads (decoded with Flet's own ``from_dict``).
* ``web_pages_fake.FakeWebPagesNative``: the native semantics the app's host tests rely on.
* Static checks of the Dart / Kotlin / Swift sources (no Flutter SDK, Xcode or Gradle here): balance,
  method / code / event names across the four languages, constants, the critic's rules for the hidden
  pages (C1-C12), build files unchanged, uniform line endings, Python 3.10 parse.

Run: python -m pytest -p no:cacheprovider -W ignore src/mobile/extensions/flet_glossarion_native/tests
"""

import ast
import asyncio
import json
import logging
import re
import sys
import threading
import xml.etree.ElementTree as ET
from dataclasses import fields
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

ft = pytest.importorskip("flet")

from flet.controls.control_event import get_event_field_type  # noqa: E402
from flet.utils.from_dict import from_dict  # noqa: E402

import flet_glossarion_native as gn  # noqa: E402
from flet_glossarion_native import web_pages as wp  # noqa: E402
from flet_glossarion_native.web_pages_fake import (  # noqa: E402
    DEFAULT_CAPABILITY,
    FakeJsError,
    FakeWebPagesNative,
    expression_of,
)
from test_document_destinations import _code_only  # noqa: E402  (shared balance helper)

FLUTTER_PKG = ROOT / "src" / "flutter" / "flet_glossarion_native"
DART_SERVICE = FLUTTER_PKG / "lib" / "src" / "native_service.dart"
KOTLIN_DIR = FLUTTER_PKG / "android" / "src" / "main" / "kotlin" / "com" / "glossarion" / "flet_glossarion_native"
KOTLIN_PLUGIN = KOTLIN_DIR / "GlossarionNativePlugin.kt"
KOTLIN_WEB = KOTLIN_DIR / "HiddenWebPages.kt"
SWIFT_DIR = FLUTTER_PKG / "ios" / "flet_glossarion_native" / "Sources" / "flet_glossarion_native"
SWIFT_PLUGIN = SWIFT_DIR / "GlossarionNativePlugin.swift"
SWIFT_WEB = SWIFT_DIR / "HiddenWebPages.swift"
SWIFT_DOCS = SWIFT_DIR / "DocumentDestinations.swift"
PODSPEC = FLUTTER_PKG / "ios" / "flet_glossarion_native.podspec"
PACKAGE_SWIFT = FLUTTER_PKG / "ios" / "flet_glossarion_native" / "Package.swift"
GRADLE = FLUTTER_PKG / "android" / "build.gradle.kts"
PUBSPEC = FLUTTER_PKG / "pubspec.yaml"
MANIFEST = FLUTTER_PKG / "android" / "src" / "main" / "AndroidManifest.xml"
PY_PKG = ROOT / "src" / "flet_glossarion_native"
ANDROID_NS = "{http://schemas.android.com/apk/res/android}"

WEB_METHODS = set(wp.WEB_METHODS)
PLATFORM_CODES = {e.value for e in wp.WebPageError} - {"timeout"}

SCRIPT = "document.title + 'SECRET-PROMPT-1234'"

# The app's eval wrapper (services/native_webview.eval_script, design section 3.6).
WRAPPER = """(function () {
  var envelope;
  try {
    var value = (
%s
    );
    envelope = {ok: true, value: value === undefined ? null : value};
  } catch (error) {
    envelope = {ok: false, error: String(error && (error.message || error))};
  }
  try { return JSON.stringify(envelope); }
  catch (error) { return JSON.stringify({ok: false, error: 'the value is not JSON'}); }
})()"""


def wrap(expression):
    return WRAPPER % expression


def run(coro):
    return asyncio.run(coro)


class FakePage:
    def __init__(self, platform, web=False):
        self.platform = platform
        self.web = web


def attach(native, platform, web=False):
    page = FakePage(platform, web=web)
    native._page_or_none = lambda: page
    return page


def recording_client(native, results=None, error=None):
    calls = []

    async def fake_invoke(method_name, arguments=None, timeout=None):
        calls.append((method_name, arguments, timeout))
        if error is not None:
            raise error
        value = (results or {}).get(method_name)
        return value(arguments) if callable(value) else value

    native._invoke_method = fake_invoke
    return calls


def forbid_client(native):
    async def fail(*args, **kwargs):
        raise AssertionError("native client must not be called")

    native._invoke_method = fail


def call_all(native):
    return {
        "web_open": run(native.web_open("p1", width=360, height=640, prevent_links=("intent:",))),
        "web_load": run(native.web_load("p1", "https://example.com/")),
        "web_eval": run(native.web_eval("p1", SCRIPT, timeout=12)),
        "web_title": run(native.web_title("p1")),
        "web_close": run(native.web_close("p1")),
        "web_reset": run(native.web_reset()),
        "web_status": run(native.web_status()),
    }


def _source(path):
    return path.read_text(encoding="utf-8")


# ============================================================ the Python service


def test_every_web_method_is_a_coroutine_and_on_web_decodes_web_page_events():
    for name in WEB_METHODS:
        assert asyncio.iscoroutinefunction(getattr(gn.GlossarionNative, name)), name
    native = gn.GlossarionNative()
    assert get_event_field_type(native, "on_web") is gn.WebPageEvent
    assert {f.name for f in fields(gn.WebPageEvent)} - {"name", "data", "control"} == set(wp.WEB_EVENT_KEYS)
    assert {t.value for t in gn.WebPageEventType} == set(wp.WEB_EVENT_TYPES)
    assert gn.WEB_PAGES_VERSION == wp.WEB_PAGES_VERSION == 1 and gn.NATIVE_MAX_PAGES == 8
    assert gn.WebPageError is wp.WebPageError


@pytest.mark.parametrize("platform", [None, ft.PagePlatform.WINDOWS, ft.PagePlatform.MACOS, ft.PagePlatform.LINUX])
def test_safe_defaults_off_device(platform):
    native = gn.GlossarionNative()
    if platform is not None:
        attach(native, platform)
    forbid_client(native)
    for name, answer in call_all(native).items():
        assert answer["ok"] is False and answer["error"] == "unavailable", name
        assert isinstance(answer["message"], str)


def test_safe_defaults_on_web_and_when_disabled(monkeypatch):
    native = gn.GlossarionNative()
    attach(native, ft.PagePlatform.ANDROID, web=True)
    forbid_client(native)
    assert all(a["error"] == "unavailable" for a in call_all(native).values())
    monkeypatch.setenv(gn.native.DISABLE_ENV, "1")
    native = gn.GlossarionNative()
    attach(native, ft.PagePlatform.IOS)
    forbid_client(native)
    answers = call_all(native)
    assert all(a["error"] == "unavailable" and "disabled" in a["message"] for a in answers.values())


@pytest.mark.parametrize("platform", [ft.PagePlatform.ANDROID, ft.PagePlatform.IOS])
def test_marshalling_and_timeouts(platform):
    native = gn.GlossarionNative()
    attach(native, platform)
    calls = recording_client(native, results={
        "web_open": {"ok": True, "page_id": "p1", "attached": True},
        "web_load": {"ok": True},
        "web_eval": {"ok": True, "result": '{"ok":true,"value":1}', "url": "https://example.com/"},
        "web_title": {"ok": True, "title": "Example", "url": "https://example.com/"},
        "web_close": {"ok": True, "closed": True},
        "web_reset": {"ok": True, "closed": 2},
        "web_status": {"ok": True, "pages": [{"page_id": "p1", "gone": False}, "junk"], "attached": True},
    })
    answers = call_all(native)
    by_name = {name: (args, timeout) for name, args, timeout in calls}
    assert set(by_name) == WEB_METHODS
    assert by_name["web_open"] == ({"page_id": "p1", "width": 360, "height": 640, "prevent_links": ["intent:"]},
                                   wp.OPEN_TIMEOUT)
    assert by_name["web_load"] == ({"page_id": "p1", "url": "https://example.com/"}, wp.LOAD_TIMEOUT)
    assert by_name["web_eval"] == ({"page_id": "p1", "script": SCRIPT}, 12 + wp.EVAL_SLACK)
    for name in ("web_title", "web_close"):
        assert by_name[name] == ({"page_id": "p1"}, wp.QUERY_TIMEOUT)
    assert by_name["web_reset"] == ({}, wp.QUERY_TIMEOUT) and by_name["web_status"] == ({}, wp.QUERY_TIMEOUT)
    assert answers["web_open"] == {"ok": True, "page_id": "p1", "attached": True, "error": None, "message": None}
    assert answers["web_eval"]["result"] == '{"ok":true,"value":1}' and answers["web_eval"]["url"] == "https://example.com/"
    assert answers["web_title"]["title"] == "Example"
    assert answers["web_close"]["closed"] is True and answers["web_reset"]["closed"] == 2
    assert answers["web_status"]["pages"] == [{"page_id": "p1", "gone": False}]
    # The eval budget is the caller's; the invoke waits EVAL_SLACK longer (35 s < NativeBridge's 45 s guard).
    run(native.web_eval("p1", "1"))
    assert calls[-1][2] == wp.DEFAULT_EVAL_TIMEOUT + wp.EVAL_SLACK < 45.0


def test_viewport_is_clamped_and_bad_arguments_never_reach_the_platform():
    native = gn.GlossarionNative()
    attach(native, ft.PagePlatform.ANDROID)
    calls = recording_client(native, results={"web_open": {"ok": True}})
    run(native.web_open("p", width=0, height=99999, prevent_links=["", "market:"]))
    assert calls[-1][1] == {"page_id": "p", "width": 1, "height": wp.MAX_SIDE, "prevent_links": ["market:"]}
    run(native.web_open("p", width="wide", height=float("nan")))
    assert calls[-1][1]["width"] == 480 and calls[-1][1]["height"] == 800
    before = len(calls)
    assert run(native.web_open("", width=1, height=1))["error"] == "bad_args"
    for url in ("file:///data/x.html", "content://x/y", "javascript:alert(1)", "about:blank", "https://", None):
        assert run(native.web_load("p", url))["error"] == "bad_args", url
    assert run(native.web_eval("p", None))["error"] == "bad_args"
    assert len(calls) == before


def test_timeouts_and_errors_never_expose_the_script(caplog):
    native = gn.GlossarionNative()
    attach(native, ft.PagePlatform.ANDROID)
    caplog.set_level(logging.DEBUG, logger="flet_glossarion_native")
    # Flet's own TimeoutError text embeds the call arguments (flet/messaging/session.py invoke_method).
    recording_client(native, error=TimeoutError(f"Timeout waiting for invokeMethod web_eval({{'script': {SCRIPT!r}}}) call"))
    answer = run(native.web_eval("p1", SCRIPT))
    assert answer == {"ok": False, "error": "timeout", "message": "web_eval did not answer in time"}
    recording_client(native, error=asyncio.TimeoutError())
    assert run(native.web_title("p1"))["error"] == "timeout"
    recording_client(native, error=RuntimeError(f"GlossarionNative.web_eval failed: native_error: {SCRIPT}"))
    answer = run(native.web_eval("p1", SCRIPT))
    assert answer == {"ok": False, "error": "native_error", "message": "RuntimeError"}
    assert "SECRET-PROMPT" not in caplog.text and "SECRET-PROMPT" not in json.dumps(answer)
    assert native.native_available is True  # still trying


def test_missing_dart_service_marks_unavailable_for_the_session():
    native = gn.GlossarionNative()
    attach(native, ft.PagePlatform.IOS)
    calls = recording_client(native, error=RuntimeError(
        "Timeout waiting for invoke method listener for GlossarionNative(9).web_open"))
    assert run(native.web_open("p", width=10, height=10))["error"] == "unavailable"
    assert run(native.web_status())["error"] == "unavailable"
    assert len(calls) == 1 and native.native_available is False


def test_cancellation_is_never_swallowed():
    native = gn.GlossarionNative()
    attach(native, ft.PagePlatform.ANDROID)
    recording_client(native, error=asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        run(native.web_eval("p1", "1"))


def test_answers_are_normalised():
    native = gn.GlossarionNative()
    attach(native, ft.PagePlatform.ANDROID)
    recording_client(native, results={
        "web_open": {"ok": False, "error": "weird", "message": 3},
        "web_load": "not a map",
        "web_eval": {"ok": True, "result": 42, "url": 7},
        "web_title": {"ok": True, "title": None},
        "web_close": {"ok": True, "closed": "yes"},
        "web_reset": {"ok": True, "closed": "x"},
        "web_status": {"ok": True, "pages": "x"},
    })
    answers = call_all(native)
    assert answers["web_open"] == {"ok": False, "error": "native_error", "message": None}
    assert answers["web_load"]["error"] == "native_error"
    assert answers["web_eval"]["result"] is None and answers["web_eval"]["url"] is None
    assert answers["web_title"] == {"ok": True, "title": None, "url": None, "error": None, "message": None}
    assert answers["web_close"]["closed"] is False and answers["web_reset"]["closed"] == 0
    assert answers["web_status"]["pages"] == []
    recording_client(native, results={"web_eval": {"ok": False, "error": "gone", "message": "the renderer stopped"}})
    assert run(native.web_eval("p1", "1")) == {"ok": False, "error": "gone", "message": "the renderer stopped"}


# ============================================================ contract helpers and events


def test_web_page_event_payloads_decode_with_flet_from_dict():
    native = gn.GlossarionNative()
    base = {"control": native, "name": "web"}
    started = from_dict(gn.WebPageEvent, {**base, "page_id": "p", "type": "started", "url": "https://a/"})
    assert (started.type, started.page_id, started.url, started.ok) == ("started", "p", "https://a/", None)
    finished = from_dict(gn.WebPageEvent, {**base, "page_id": "p", "type": "finished", "url": "https://a/",
                                           "ok": False, "error": "net::ERR_NAME_NOT_RESOLVED"})
    assert finished.ok is False and finished.error == "net::ERR_NAME_NOT_RESOLVED"
    error = from_dict(gn.WebPageEvent, {**base, "page_id": "p", "type": "error", "url": None, "code": -2,
                                        "error": "x", "main_frame": True, "future_key": 1})
    assert error.code == -2 and error.main_frame is True
    gone = from_dict(gn.WebPageEvent, {**base, "page_id": "p", "type": "gone", "crashed": False})
    assert gone.type == gn.WebPageEventType.GONE and gone.crashed is False and gone.name == "web"
    url = from_dict(gn.WebPageEvent, {**base, "page_id": "p", "type": "url", "url": "https://a/#x"})
    assert url.url == "https://a/#x" and url.control is native


@pytest.mark.parametrize("info,expected", [
    (None, None),
    ({"native": False, "web_pages": {"version": 1}}, None),
    ({"native": True}, None),
    ({"native": True, "web_pages": "yes"}, None),
    ({"native": True, "web_pages": {"version": 0}}, None),
    ({"native": True, "web_pages": {"version": True}}, None),
    ({"native": True, "web_pages": {"version": "x"}}, None),
    ({"native": True, "web_pages": {"version": 1, "max_pages": 8, "provider": "com.google.android.webview 140"}},
     {"version": 1, "max_pages": 8, "provider": "com.google.android.webview 140"}),
    ({"native": True, "web_pages": {"version": "2", "provider": None}}, {"version": 2, "max_pages": 8, "provider": None}),
])
def test_capability(info, expected):
    assert wp.capability(info) == expected


def test_contract_helpers():
    assert wp.clamp_side(360.4, 1) == 360 and wp.clamp_side(-5, 1) == 1 and wp.clamp_side(10 ** 9, 1) == wp.MAX_SIDE
    assert wp.clamp_side(True, 7) == 7 and wp.clamp_side(None, 7) == 7
    assert wp.is_http_url("https://chatgpt.com/robots.txt") and wp.is_http_url("HTTP://127.0.0.1:8080/t/")
    assert not wp.is_http_url("https:/x") and not wp.is_http_url("intent://x") and not wp.is_http_url(3)
    assert wp.error_result("nope") == {"ok": False, "error": "native_error", "message": None}
    assert wp.error_result(wp.WebPageError.GONE, "m", page_id="p") == {"ok": False, "error": "gone", "message": "m",
                                                                       "page_id": "p"}
    assert wp.normalize_result({"ok": True, "x": 1}) == {"ok": True, "x": 1, "error": None, "message": None}


def test_expression_of_reads_the_wrapper():
    assert expression_of(wrap("6 * 7")) == "6 * 7"
    multi = "(() => {\n  // a comment\n  return 1;\n})()"
    assert expression_of(wrap(multi)) == multi
    assert expression_of(wrap("x").replace("\n", "\r\n")) == "x"
    assert expression_of("6 * 7") is None and expression_of(None) is None


# ============================================================ the fake (native semantics)


async def _open(fake, page_id="p1", **kwargs):
    return await fake.web_open(page_id, width=kwargs.pop("width", 360), height=kwargs.pop("height", 640), **kwargs)


def test_fake_open_cap_and_errors():
    async def scenario():
        fake = FakeWebPagesNative()
        assert (await _open(fake, prevent_links=["intent:", ""]))["ok"]
        assert fake.page("p1").prevent_links == ["intent:"] and fake.page("p1").width == 360
        assert (await _open(fake))["error"] == "exists"
        assert (await fake.web_open("", width=1, height=1))["error"] == "bad_args"
        for index in range(2, 9):
            assert (await _open(fake, f"p{index}"))["ok"]
        assert (await _open(fake, "p9"))["error"] == "too_many"
        assert (await fake.web_load("nope", "https://a/"))["error"] == "no_page"
        assert (await fake.web_eval("nope", "1"))["error"] == "no_page"
        assert (await fake.web_title("nope"))["error"] == "no_page"
        assert (await fake.web_load("p1", "file:///x"))["error"] == "bad_args"
        assert (await fake.web_close("nope"))["closed"] is False
        assert (await fake.web_close("p1"))["closed"] is True and fake.closed_ids == ["p1"]
        assert (await fake.web_reset())["closed"] == 7 and fake.pages == {} and fake.resets == 1
    run(scenario())


def test_fake_load_sends_events_after_the_answer_and_starts_a_new_document():
    async def scenario():
        received = []
        fake = FakeWebPagesNative(emit=received.append, title_of=lambda url: "Title of " + url)
        await _open(fake)
        fake.page("p1").state["token"] = "old"
        answer = await fake.web_load("p1", "https://example.com/a")
        assert answer == {"ok": True, "error": None, "message": None} and received == []
        await asyncio.sleep(0)
        assert [e["type"] for e in received] == ["started", "url", "finished"]
        assert received[-1] == {"page_id": "p1", "type": "finished", "url": "https://example.com/a", "ok": True,
                                "error": None}
        page = fake.page("p1")
        assert page.generation == 1 and page.state == {} and page.loads == ["https://example.com/a"]
        assert (await fake.web_title("p1"))["title"] == "Title of https://example.com/a"
        status = await fake.web_status()
        assert status["pages"] == [{"page_id": "p1", "url": "https://example.com/a", "gone": False, "width": 360,
                                    "height": 640, "pending": 0}] and status["attached"] is True
        fake.auto_finish = False
        await fake.web_load("p1", "https://example.com/b")
        await asyncio.sleep(0)
        assert received[-1]["type"] == "url"
        fake.finish("p1", ok=False, error="net::ERR_FAILED").result()
        assert [e["type"] for e in received[-2:]] == ["error", "finished"] and received[-1]["ok"] is False
    run(scenario())


def test_fake_eval_envelopes():
    def responder(page_id, expression):
        if expression == "boom()":
            raise FakeJsError("ReferenceError: boom is not defined")
        if expression == "window.huge":
            return "x" * 200_000
        if expression == "6 * 7":
            return 42
        if expression == "'plain'":
            return "plain"
        return None

    async def scenario():
        fake = FakeWebPagesNative(responder)
        await _open(fake)
        await fake.web_load("p1", "https://example.com/")
        answer = await fake.web_eval("p1", wrap("6 * 7"))
        assert json.loads(answer["result"]) == {"ok": True, "value": 42} and answer["url"] == "https://example.com/"
        assert json.loads((await fake.web_eval("p1", wrap("boom()")))["result"]) == {
            "ok": False, "error": "ReferenceError: boom is not defined"}
        big = json.loads((await fake.web_eval("p1", wrap("window.huge")))["result"])
        assert len(big["value"]) == 200_000
        assert json.loads((await fake.web_eval("p1", wrap("undefined")))["result"]) == {"ok": True, "value": None}
        # Not wrapped: the completion value when it is a string (the native rule), else null.
        assert (await fake.web_eval("p1", "'plain'"))["result"] == "plain"
        assert (await fake.web_eval("p1", "6 * 7"))["result"] is None
        assert fake.calls_of("web_eval")[0]["script"] == wrap("6 * 7")
    run(scenario())


def test_fake_events_reach_on_web_as_flet_events():
    async def scenario():
        fake = FakeWebPagesNative()
        seen = []

        async def on_web(event):
            seen.append(event)

        fake.on_web = on_web
        await _open(fake)
        await fake.web_load("p1", "https://example.com/")
        for _ in range(5):
            await asyncio.sleep(0)
        assert [e.type for e in seen] == ["started", "url", "finished"]
        assert all(isinstance(e, gn.WebPageEvent) and e.name == "web" and e.page_id == "p1" for e in seen)
        assert seen[-1].ok is True
    run(scenario())


@pytest.fixture
def loop_thread():
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    try:
        yield loop
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(10)
        loop.close()


def submit(loop, coro):
    return asyncio.run_coroutine_threadsafe(coro, loop)


def test_fake_held_evals_answer_once_on_release_close_or_crash(loop_thread):
    state = {"value": "before"}
    fake = FakeWebPagesNative(lambda page_id, expression: state["value"])
    for page_id in ("a", "b", "c"):
        assert submit(loop_thread, _open(fake, page_id)).result(10)["ok"]
    fake.hold_evals()
    held = [submit(loop_thread, fake.web_eval(page_id, wrap("x"))) for page_id in ("a", "b", "c")]
    assert fake.wait_held(3)
    status = submit(loop_thread, fake.web_status()).result(10)
    assert {p["page_id"]: p["pending"] for p in status["pages"]} == {"a": 1, "b": 1, "c": 1}
    assert not any(f.done() for f in held)
    # close answers its pending eval "closed"; a crash answers "gone" and sends the event.
    assert submit(loop_thread, fake.web_close("a")).result(10)["closed"] is True
    assert held[0].result(10) == {"ok": False, "error": "closed", "message": "the in-app page was closed"}
    assert fake.crash("b", crashed=False).result(10) is True
    assert held[1].result(10)["error"] == "gone"
    assert fake.wait_events(lambda e: e["type"] == "gone" and e["page_id"] == "b" and e["crashed"] is False)
    assert submit(loop_thread, fake.web_eval("b", "1")).result(10)["error"] == "gone"
    assert submit(loop_thread, fake.web_load("b", "https://x/")).result(10)["error"] == "gone"
    assert fake.crash("b").result(10) is False  # once
    # release computes the value at release time.
    state["value"] = "after"
    assert fake.release_evals().result(10) == 1
    assert json.loads(held[2].result(10)["result"]) == {"ok": True, "value": "after"}
    # A gone page keeps its id until it is closed; it counts towards the cap.
    assert "b" in fake.pages and fake.open_pages == ["c"]
    assert submit(loop_thread, fake.web_close("b")).result(10)["closed"] is True


def test_fake_hold_on_open_applies_the_call_and_delays_the_answer(loop_thread):
    fake = FakeWebPagesNative()
    fake.hold("web_open")
    pending = submit(loop_thread, _open(fake, "slow"))
    assert fake.wait_held(1, method="web_open")
    assert "slow" in fake.pages and not pending.done()  # the platform has the page; the answer travels
    assert submit(loop_thread, fake.web_close("slow")).result(10)["closed"] is True
    fake.release("web_open").result(10)
    assert pending.result(10)["ok"] is True and fake.pages == {}


def test_fake_reset_answers_every_held_eval_closed(loop_thread):
    fake = FakeWebPagesNative()
    submit(loop_thread, _open(fake, "a")).result(10)
    fake.hold_evals("a")
    held = submit(loop_thread, fake.web_eval("a", "1"))
    assert fake.wait_held(1)
    assert submit(loop_thread, fake.web_reset()).result(10)["closed"] == 1
    assert held.result(10)["error"] == "closed"


def test_fake_platform_info():
    async def scenario():
        info = await FakeWebPagesNative().get_platform_info()
        assert info["native"] is True and info["web_pages"] == DEFAULT_CAPABILITY
        assert wp.capability(info)["version"] == 1
        bare = await FakeWebPagesNative(capability=None, platform="ios").get_platform_info()
        assert "web_pages" not in bare and bare["platform"] == "ios" and wp.capability(bare) is None
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(FakeWebPagesNative(hang_platform_info=True).get_platform_info(), 0.01)
        assert FakeWebPagesNative.is_stub is False
    run(scenario())


# ============================================================ static cross-language checks


@pytest.mark.parametrize("path,lang", [
    (KOTLIN_WEB, "kotlin"), (KOTLIN_PLUGIN, "kotlin"), (SWIFT_WEB, "swift"), (SWIFT_PLUGIN, "swift"),
    (SWIFT_DOCS, "swift"), (DART_SERVICE, "dart"),
])
def test_sources_are_balanced(path, lang):
    code = _code_only(_source(path), lang)
    stack = []
    pairs = {")": "(", "]": "[", "}": "{"}
    for index, ch in enumerate(code):
        if ch in "([{":
            stack.append((ch, index))
        elif ch in ")]}":
            assert stack and stack[-1][0] == pairs[ch], f"{path.name}: unbalanced {ch!r} near {code[max(0, index - 80):index]!r}"
            stack.pop()
    assert not stack, f"{path.name}: unclosed {stack[-1][0]!r} near {code[stack[-1][1]:stack[-1][1] + 80]!r}"


def _dart_web_block(dart):
    start = dart.index("const Set<String> _kWebMethods")
    return dart[start:dart.index("};", start)]


def test_method_names_match_across_languages():
    dart = _source(DART_SERVICE)
    assert set(re.findall(r"'([a-z_]+)'", _dart_web_block(dart))) == WEB_METHODS
    invoke = dart[dart.index("Future<dynamic> _invokeMethod"):dart.index("dynamic _defaultResult")]
    assert WEB_METHODS <= set(re.findall(r"case '([a-z_]+)':", invoke))
    kotlin = _source(KOTLIN_WEB)
    kotlin_methods = set(re.findall(r'"([a-z_]+)"', re.search(r"val METHODS = setOf\((.*?)\)", kotlin, re.S).group(1)))
    assert kotlin_methods == WEB_METHODS
    assert WEB_METHODS <= set(re.findall(r'"([a-z_]+)" ->', kotlin))
    swift = _source(SWIFT_WEB)
    swift_methods = set(re.findall(r'"([a-z_]+)"', re.search(
        r"static let methods: Set<String> = \[(.*?)\]", swift, re.S).group(1)))
    assert swift_methods == WEB_METHODS
    assert WEB_METHODS <= set(re.findall(r'case "([a-z_]+)":', swift))
    python = {name for name in dir(gn.GlossarionNative) if not name.startswith("_")}
    assert WEB_METHODS <= python
    # No overlap with the document destinations: each plugin routes by its own set.
    assert not WEB_METHODS & set(re.findall(r"'([a-z_]+)'", dart[dart.index("_kDocumentMethods"):dart.index("};")]))


def test_error_codes_event_types_and_constants_match_python():
    kotlin = _source(KOTLIN_WEB)
    swift = _source(SWIFT_WEB)
    assert set(re.findall(r'const val ERR_[A-Z_]+ = "([a-z_]+)"', kotlin)) == PLATFORM_CODES
    assert set(re.findall(r'static let err[A-Za-z]+ = "([a-z_]+)"', swift)) == PLATFORM_CODES
    assert set(re.findall(r'const val EVENT_[A-Z_]+ = "([a-z_]+)"', kotlin)) == set(wp.WEB_EVENT_TYPES)
    assert set(re.findall(r'static let event[A-Za-z]+ = "([a-z_]+)"', swift)) == set(wp.WEB_EVENT_TYPES)
    assert re.search(r"const val VERSION = (\d+)", kotlin).group(1) == str(wp.WEB_PAGES_VERSION)
    assert re.search(r"const val MAX_PAGES = (\d+)", kotlin).group(1) == str(wp.NATIVE_MAX_PAGES)
    assert re.search(r"MAX_SUBRESOURCE_ERRORS = (\d+)", kotlin).group(1) == str(wp.MAX_SUBRESOURCE_ERRORS)
    assert re.search(r"MAX_SIDE_DP = (\d+)", kotlin).group(1) == str(wp.MAX_SIDE)
    assert re.search(r"static let version = (\d+)", swift).group(1) == str(wp.WEB_PAGES_VERSION)
    assert re.search(r"static let maxPages = (\d+)", swift).group(1) == str(wp.NATIVE_MAX_PAGES)
    assert re.search(r"private static let maxSide = (\d+)", swift).group(1) == str(wp.MAX_SIDE)
    dart = _source(DART_SERVICE)
    default = dart[dart.index("dynamic _defaultResult"):dart.index("Future<dynamic> _platform(")]
    assert "_kWebMethods.contains(name)" in default and "'error': 'unavailable'" in default


def test_event_keys_match_the_dataclass():
    allowed = set(wp.WEB_EVENT_KEYS)
    kotlin = _source(KOTLIN_WEB)
    keys = set()
    for call in re.findall(r"emit\(page, EVENT_[A-Z]+,(.*?)\)\n", kotlin):
        keys |= set(re.findall(r'"([a-z_]+)" to', call))
    emit = kotlin[kotlin.index("private fun emit("):kotlin.index("// ----------------------------------------------------------------- container")]
    keys |= set(re.findall(r'event\["([a-z_]+)"\]', emit))
    assert {"page_id", "type", "url", "ok", "error", "code", "main_frame", "crashed"} == keys
    swift = _source(SWIFT_WEB)
    delegate = swift[swift.index("// MARK: - WKNavigationDelegate"):]
    swift_keys = set(re.findall(r'"([a-z_]+)":', delegate)) | set(re.findall(r'(?:extras|failure|finished)\["([a-z_]+)"\]', delegate))
    swift_emit = swift[swift.index("private func emit("):swift.index("private func livePage(")]
    swift_keys |= set(re.findall(r'"([a-z_]+)":', swift_emit))
    assert swift_keys == allowed
    dart = _source(DART_SERVICE)
    assert "case 'web':" in dart and "control.triggerEvent('web', _stringMap(args));" in dart


def test_dart_service_rules():
    dart = _source(DART_SERVICE)
    # An existing test slices _kDocumentMethods up to the first "};": the web set must come after it.
    assert dart.index("_kDocumentMethods") < dart.index("};") < dart.index("_kWebMethods")
    invoke = dart[dart.index("Future<dynamic> _invokeMethod"):dart.index("dynamic _defaultResult")]
    block = invoke[invoke.index("case 'web_open':"):]
    block = block[:block.index("default:")]
    assert "debugPrint" not in _code_only(block, "dart") and "_waitReady()" not in _code_only(block, "dart")
    assert "if (!_isMobile) return _defaultResult(name);" in block and "is Map ?" in block
    platform_call = dart[dart.index("Future<dynamic> _onPlatformCall"):dart.index("void _onTaskData")]
    assert "case 'web':" in platform_call
    assert "const String kGlossarionNativeVersion = '0.1.0';" in dart


def test_kotlin_follows_the_critic_rules():
    kotlin = _source(KOTLIN_WEB)
    code = _code_only(kotlin, "kotlin")
    for forbidden in ("pauseTimers", "onPause(", "resumeTimers", "setWebContentsDebuggingEnabled",
                      "addJavascriptInterface", "result.error(", "dispatchTouchEvent", "onInterceptTouchEvent",
                      "androidx.webkit", "INVISIBLE", "View.GONE", "alpha ="):
        assert forbidden not in code, forbidden
    # The registry is per instance: the companion object holds constants and pure helpers only.
    companion = code[code.index("companion object"):]
    assert "Page" not in companion.replace("Pages", "") and "LinkedHashMap" not in companion
    assert "private val pages = LinkedHashMap<String, Page>()" in kotlin
    # C1: the activity setter only stores it and posts; the content view comes from peekDecorView.
    setter = kotlin[kotlin.index("var activity: Activity? = null"):kotlin.index("private val pages")]
    assert "mainHandler.post { if (activity === value) attachContainer() }" in setter
    assert "addView" not in setter and "findViewById" not in setter
    attach = kotlin[kotlin.index("private fun attachContainer()"):kotlin.index("private fun detachContainer()")]
    assert "peekDecorView()" in attach and "if (!hasLivePage())" in attach
    assert "content.addView(c, 0, FrameLayout.LayoutParams(1, 1, Gravity.TOP or Gravity.START))" in attach
    assert re.search(r"act\.findViewById|activity\.findViewById", code) is None
    # C10: the container is made lazily, never in a constructor / init of HiddenWebPages.
    assert "private var container: HiddenContainer? = null" in kotlin and "HiddenContainer(appContext).also" in kotlin
    assert '@SuppressLint("ViewConstructor")' in kotlin and "FOCUS_BLOCK_DESCENDANTS" in kotlin
    assert "override fun hasOverlappingRendering(): Boolean = false" in kotlin
    # C2 / C11: laid out by hand after addView (and after re-parenting); registered only afterwards.
    open_fn = kotlin[kotlin.index("private fun openPage("):kotlin.index("/** webview_flutter_android")]
    assert open_fn.index(".addView(webView") < open_fn.index("layoutPage(page)") < open_fn.index("pages[id] = page")
    assert "webView.destroy()" in open_fn[open_fn.index("catch (e: Exception)"):]
    layout = kotlin[kotlin.index("private fun layoutPage("):kotlin.index("private fun flushCookies(")]
    assert "page.webView.measure(exactly(page.widthPx), exactly(page.heightPx))" in layout
    assert "page.webView.layout(0, 0, page.widthPx, page.heightPx)" in layout
    assert "View.MeasureSpec.makeMeasureSpec(size, View.MeasureSpec.EXACTLY)" in kotlin
    assert "layoutPage(page)" in attach
    # C3: the renderer callback returns true and loses the page synchronously.
    gone = kotlin[kotlin.index("override fun onRenderProcessGone("):kotlin.index("private class PageChrome")]
    assert "return true" in gone and "lost(page, " in gone and "post" not in _code_only(gone, "kotlin")
    destroy = kotlin[kotlin.index("private fun destroy("):kotlin.index("/** The renderer is gone")]
    assert destroy.index("if (!page.gone)") < destroy.index("webView.stopLoading()")
    assert "page.destroyed = true" in destroy and "pending.send(fail(answer, message))" in destroy
    # web_reset / dispose: CookieManager.flush() blocks the main thread, so once for all pages, not per page.
    assert "private fun destroy(page: Page, answer: String, settle: Boolean = true)" in kotlin
    assert "if (settle && !hasLivePage())" in destroy
    reset = kotlin[kotlin.index("private fun reset(): Int"):kotlin.index("private fun status()")]
    assert "destroy(page, ERR_CLOSED, settle = false)" in reset and _code_only(reset, "kotlin").count("flushCookies()") == 1
    assert reset.index("destroy(page, ERR_CLOSED, settle = false)") < reset.index("removeContainerIfEmpty()") < reset.index("flushCookies()")
    # API 26+ calls behind SDK checks.
    for call in ("setRendererPriorityPolicy(", "getCurrentWebViewPackage()", "importantForAutofill ="):
        before = code[:code.index(call)]
        assert before.rfind("Build.VERSION.SDK_INT >= Build.VERSION_CODES.O") > before.rfind("}"), call
    assert "@TargetApi(Build.VERSION_CODES.O)" in kotlin
    # Settings.
    for setting in ("settings.javaScriptEnabled = true", "settings.domStorageEnabled = true",
                    "settings.setSupportMultipleWindows(true)", "settings.allowFileAccess = false",
                    "settings.allowContentAccess = false", "settings.useWideViewPort = true",
                    "settings.loadWithOverviewMode = true", "CookieManager.getInstance().setAcceptCookie(true)"):
        assert setting in kotlin, setting
    assert "userAgentString" not in code and "setAcceptThirdPartyCookies" not in code and "onCreateWindow" not in code
    assert "override fun onConsoleMessage(consoleMessage: ConsoleMessage?): Boolean = true" in kotlin
    # Exactly-once replies: result.success only inside Reply.send, behind the replied flag.
    assert code.count("result.success(") == 1
    send = kotlin[kotlin.index("fun send(answer: Map<String, Any?>)"):]
    send = send[:send.index("result.success(answer)")]
    assert "if (replied) return" in send and "replied = true" in send
    assert "reply.send(" in kotlin[kotlin.index("page.webView.evaluateJavascript(script)"):]
    # C12: the capability is computed once per instance.
    assert "by lazy { computeCapability() }" in kotlin
    # Logs carry exception type names only.
    assert "e.message" not in code and "e.toString()" not in code


def test_kotlin_plugin_wiring():
    plugin = _source(KOTLIN_PLUGIN)
    on_call = plugin[plugin.index("override fun onMethodCall("):plugin.index("// ------------------------------------------------------------ platform info")]
    assert on_call.index("docs.handles(call.method)") < on_call.index("web.handles(call.method)") < on_call.index("when (call.method)")
    assert 'sendToDart = { event -> channel?.invokeMethod("web", event) }' in plugin
    detach = plugin[plugin.index("override fun onDetachedFromEngine("):plugin.index("// ------------------------------------------------------------- ActivityAware")]
    assert "webPages?.dispose()" in detach and "webPages = null" in detach
    assert plugin.count("webPages?.activity = binding.activity") == 2 and plugin.count("webPages?.activity = null") == 2
    assert 'webPages?.capability()?.let { info["web_pages"] = it }' in plugin
    assert "companion object" not in plugin[:plugin.index("private var webPages")]


DELEGATE_SIGNATURES = (
    "func webView(_ webView: WKWebView, didStartProvisionalNavigation navigation: WKNavigation!)",
    "func webView(_ webView: WKWebView, didCommit navigation: WKNavigation!)",
    "func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!)",
    "func webView(_ webView: WKWebView, didFail navigation: WKNavigation!, withError error: Error)",
    "_ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!, withError error: Error",
    "func webViewWebContentProcessDidTerminate(_ webView: WKWebView)",
)


def test_swift_follows_the_critic_rules():
    swift = _source(SWIFT_WEB)
    code = _code_only(swift, "swift")
    assert "final class HiddenWebPages: NSObject, WKNavigationDelegate {" in swift
    assert "import WebKit" in swift and "configuration.websiteDataStore = WKWebsiteDataStore.default()" in swift
    for forbidden in ("decidePolicyFor", "@MainActor", "async", "await", "Task {", "@Sendable", "processPool",
                      "WKProcessPool", "isScrollEnabled", "uiDelegate", "WKUserScript", "WKContentWorld",
                      "callAsyncJavaScript", "isInspectable", "UTType", "WKWebpagePreferences",
                      "GlossarionNativePlugin.intValue", "?? NSNull()", "FlutterError"):
        assert forbidden not in code, forbidden
    for signature in DELEGATE_SIGNATURES:
        assert signature in swift, signature
    # C4: result is called once, with a typed [String: Any] variable, inside Reply.send.
    assert code.count("result(") == 1 and "result(answer)" in code
    send = swift[swift.index("func send(_ answer: [String: Any])"):]
    send = send[:send.index("result(answer)")]
    assert "if replied {" in send and "replied = true" in send
    assert "static func ok(_ values: [String: Any] = [:]) -> [String: Any]" in swift
    assert "static func fail(_ code: String, _ message: String) -> [String: Any]" in swift
    assert 'var event: [String: Any] = ["page_id": page.id, "type": type]' in swift
    assert "DocumentDestinations.orNull(value as? String)" in swift and "NSNull()" in swift
    # C5: numbers through NSNumber.
    assert "value as? NSNumber" in swift
    # Ignored failures: cancelled navigations and WebKit's frame-load-interrupted.
    assert "ns.domain == NSURLErrorDomain && ns.code == NSURLErrorCancelled" in swift
    assert 'ns.domain == "WebKitErrorDomain" && ns.code == 102' in swift and "WKErrorDomain" not in code
    # The window lookup is shared with the pickers; the container sits behind the root view.
    assert "DocumentDestinations.keyWindow()" in swift and "window.insertSubview(view, at: 0)" in swift
    assert "view.clipsToBounds = true" in swift and "view.isUserInteractionEnabled = false" in swift
    assert "isHidden" not in code and "alpha" not in code
    gone = swift[swift.index("func webViewWebContentProcessDidTerminate"):swift.index("/// A main-frame failure")]
    assert gone.index("page.gone = true") < gone.index("emit(page, HiddenWebPages.eventGone") < gone.index("destroy(page, HiddenWebPages.errGone)")
    # WebKit gives no reason (a crash, or iOS reclaiming a suspended app): crashed is unknown (null).
    assert 'emit(page, HiddenWebPages.eventGone, ["crashed": NSNull()])' in gone
    assert '"crashed": true' not in code and '"crashed": false' not in code
    destroy = swift[swift.index("private func destroy("):swift.index("private func emit(")]
    assert destroy.index("if !page.gone {") < destroy.index("webView.stopLoading()")
    assert "pending.send(HiddenWebPages.fail(answer, message))" in destroy


def test_swift_plugin_wiring_and_shared_key_window():
    plugin = _source(SWIFT_PLUGIN)
    handle = plugin[plugin.index("public func handle(_ call: FlutterMethodCall"):plugin.index("switch call.method {")]
    assert handle.index("DocumentDestinations.methods.contains(call.method)") < handle.index("HiddenWebPages.methods.contains(call.method)")
    assert "webPages.handle(call.method, args, result: result)" in handle
    # C6: web_pages is in `info` before it is copied into `base` for the async answer.
    info = plugin[plugin.index("private func platformInfo(result:"):]
    assert info.index('info["web_pages"] = webPages.capability()') < info.index("let base = info")
    assert "public func detachFromEngine(for registrar: FlutterPluginRegistrar)" in plugin
    assert 'self?.channel?.invokeMethod("web", arguments: event)' in plugin
    assert "private lazy var webPages = HiddenWebPages(" in plugin
    docs = _source(SWIFT_DOCS)
    assert "static func keyWindow() -> UIWindow? {" in docs
    top = docs[docs.index("static func topViewController()"):]
    assert "var root = keyWindow()?.rootViewController" in top
    # C9: the delegate-window fallback for a window without a root view controller stays.
    assert "root = UIApplication.shared.delegate?.window??.rootViewController" in top


def test_build_files_are_unchanged():
    gradle = _source(GRADLE)
    assert re.search(r"minSdk = 24\b", gradle) and re.search(r"compileSdk = 36\b", gradle)
    deps = re.search(r"^dependencies \{(.*?)^\}", gradle, re.S | re.M).group(1)
    assert re.findall(r'implementation\("([^"]+)"\)', deps) == ["androidx.core:core:1.15.0"]
    assert "androidx.webkit" not in gradle
    package = _source(PACKAGE_SWIFT)
    assert '.iOS("13.0")' in package and "// swift-tools-version: 5.9" in package and "WebKit" not in package
    podspec = _source(PODSPEC)
    assert "s.platform         = :ios, '13.0'" in podspec and "s.swift_version = '5.0'" in podspec
    assert "s.frameworks       = 'BackgroundTasks', 'UserNotifications', 'WebKit'" in podspec
    assert "flet_glossarion_native/Sources/flet_glossarion_native/**/*.swift" in podspec
    assert "s.version          = '0.1.0'" in podspec
    pubspec = _source(PUBSPEC)
    deps = re.search(r"^dependencies:\n(.*?)^flutter:", pubspec, re.S | re.M).group(1)
    # image_picker: hotfix 5's ＋ › Camera (the phone's camera app); the in-app browser itself adds nothing
    assert set(re.findall(r"^  ([a-z_]+):", deps, re.M)) == {"flutter", "flet", "flutter_foreground_task",
                                                             "image_picker"}
    assert re.search(r"^version: 0\.1\.0$", pubspec, re.M)
    root = ET.parse(MANIFEST).getroot()
    permissions = {p.get(ANDROID_NS + "name") for p in root.findall("uses-permission")}
    # REQUEST_INSTALL_PACKAGES: About › Updates opens the system installer for the downloaded APK (v9.15.7 batch)
    assert permissions == {
        "android.permission.FOREGROUND_SERVICE", "android.permission.FOREGROUND_SERVICE_DATA_SYNC",
        "android.permission.POST_NOTIFICATIONS", "android.permission.WAKE_LOCK",
        "android.permission.REQUEST_INSTALL_PACKAGES",
    }
    assert gn.EXTENSION_VERSION == "0.1.0"


def test_python_modules_parse_as_python_310_and_stay_flet_free():
    for name in ("web_pages.py", "web_pages_fake.py", "native.py", "types.py", "__init__.py"):
        ast.parse((PY_PKG / name).read_text(encoding="utf-8"), filename=name, feature_version=(3, 10))
    ast.parse(Path(__file__).read_text(encoding="utf-8"), filename="test_web_pages.py", feature_version=(3, 10))
    for name in ("web_pages.py", "web_pages_fake.py"):
        tree = ast.parse((PY_PKG / name).read_text(encoding="utf-8"))
        top = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
        modules = {alias.name for node in top if isinstance(node, ast.Import) for alias in node.names}
        modules |= {node.module for node in top if isinstance(node, ast.ImportFrom)}
        assert not any(str(m).split(".")[0] == "flet" for m in modules), name


def test_sources_have_uniform_line_endings():
    paths = [KOTLIN_WEB, KOTLIN_PLUGIN, SWIFT_WEB, SWIFT_PLUGIN, SWIFT_DOCS, DART_SERVICE, PODSPEC,
             ROOT / "README.md", Path(__file__)]
    paths += [PY_PKG / name for name in ("web_pages.py", "web_pages_fake.py", "native.py", "types.py", "__init__.py")]
    for path in paths:
        data = path.read_bytes()
        assert data.count(b"\r\n") in (0, data.count(b"\n")), path.name
        assert not data.startswith(b"\xef\xbb\xbf"), path.name
