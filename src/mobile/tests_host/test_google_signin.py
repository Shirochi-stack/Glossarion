"""authgem/'s Google sign-in on the phone (services/google_signin.py) and the extension's sign-in family
(web_pages.SIGN_IN_METHODS: web_sign_in, web_cookies, web_clear_cookies)."""
from __future__ import annotations

import asyncio
import re
import sys
import threading
from pathlib import Path

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
for _entry in (str(MOBILE_DIR / "app"), str(MOBILE_DIR.parent)):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from flet_glossarion_native import web_pages as WP  # noqa: E402
from flet_glossarion_native.web_pages_fake import DEFAULT_CAPABILITY, FakeWebPagesNative  # noqa: E402
from glossarion_mobile.services import google_signin as gs  # noqa: E402
from glossarion_mobile.services.native import NativeBridge  # noqa: E402

EXT = MOBILE_DIR / "extensions" / "flet_glossarion_native" / "src"
KOTLIN_DIR = EXT / "flutter/flet_glossarion_native/android/src/main/kotlin/com/glossarion/flet_glossarion_native"
SWIFT_DIR = EXT / "flutter/flet_glossarion_native/ios/flet_glossarion_native/Sources/flet_glossarion_native"


class LoopDispatcher:
    """UiDispatcher.submit on a loop running in its own thread (the app's Flet loop)."""

    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()

    def submit(self, coro_fn, *args):
        return asyncio.run_coroutine_threadsafe(coro_fn(*args), self.loop)

    def run(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(10)

    def close(self):
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(5)


@pytest.fixture
def phone(tmp_path):
    fake = FakeWebPagesNative(capability=dict(DEFAULT_CAPABILITY, sign_in=True))
    bridge = NativeBridge(native=fake)
    dispatcher = LoopDispatcher()
    service = gs.GoogleSignIn(bridge, dispatcher, marker_dir=str(tmp_path), page_driver=lambda: None)
    yield fake, bridge, dispatcher, service
    dispatcher.close()
    import authgem_auth

    authgem_auth.set_web_cookie_provider(None)
    gs._current = None


def test_cookie_header_becomes_google_cookies():
    assert gs.parse_cookie_header("SID=a; __Secure-1PSID=b=c ; junk; SID=dup") == [
        {"name": "SID", "value": "a", "domain": ".google.com", "path": "/"},
        {"name": "__Secure-1PSID", "value": "b=c", "domain": ".google.com", "path": "/"},
    ]
    assert gs.parse_cookie_header(None) == []


def test_sign_in_reads_and_clears_the_shared_jar(phone):
    fake, _bridge, dispatcher, service = phone
    fake.sign_in_cookies = None  # the user closes the page
    assert dispatcher.run(service.sign_in()) is False and not service.has_session()
    fake.sign_in_cookies = {"__Secure-1PSID": "psid", "SID": "sid", "NID": "n"}
    assert dispatcher.run(service.sign_in()) is True and service.has_session()
    (call,) = fake.calls_of("web_sign_in")[-1:]
    assert call["url"] == gs.SIGN_IN_URL and call["done_url"] == gs.DONE_URL
    assert call["done_cookie"] == "__Secure-1PSID" and call["cookie_url"] == gs.COOKIE_URL
    cookies = {c["name"]: c["value"] for c in service.cookies()}  # worker thread
    assert cookies == {"__Secure-1PSID": "psid", "SID": "sid", "NID": "n"}
    dispatcher.run(service.sign_out())
    assert not service.has_session() and service.cookies() == []
    assert fake.calls_of("web_clear_cookies")[-1]["domain"] == ".google.com"


def test_a_cookie_read_corrects_the_marker(phone):
    fake, _bridge, dispatcher, service = phone
    fake.cookie_jar = {"google.com": {"__Secure-1PSID": "x"}}  # signed in elsewhere (search/gemini page)
    assert not service.has_session()
    service.cookies()
    assert service.has_session()


def test_install_registers_the_cookie_provider(phone, tmp_path):
    fake, bridge, dispatcher, _service = phone
    import authgem_auth
    import types

    app = types.SimpleNamespace(native=bridge, dispatcher=dispatcher)
    service = dispatcher.run(gs.install(app))
    assert service is gs.current() is app.google_signin
    service.marker = str(tmp_path / "marker")
    assert authgem_auth.has_web_session() is False
    fake.sign_in_cookies = {"__Secure-1PSID": "psid"}
    assert dispatcher.run(service.sign_in()) is True
    assert authgem_auth.has_web_session() is True
    assert [c["name"] for c in authgem_auth._web_cookies()] == ["__Secure-1PSID"]


def test_install_skips_a_build_without_the_sign_in(phone):
    _fake, _bridge, dispatcher, _service = phone
    import types

    old = NativeBridge(native=FakeWebPagesNative())  # web pages without "sign_in"
    assert dispatcher.run(gs.install(types.SimpleNamespace(native=old, dispatcher=dispatcher))) is None


# ---------------------------------------------------------------- the extension's sign-in family

def _text(path: Path) -> str:
    return path.read_bytes().decode("utf-8").replace("\r\n", "\n")


def test_every_layer_offers_the_sign_in_family():
    names = set(WP.SIGN_IN_METHODS)
    assert names == {"web_sign_in", "web_cookies", "web_clear_cookies"} and not names & set(WP.WEB_METHODS)
    kotlin = _text(KOTLIN_DIR / "WebSignIn.kt")
    assert set(re.findall(r'"(web_[a-z_]+)"', re.search(r"val METHODS = setOf\((.*?)\)", kotlin).group(1))) == names
    swift = _text(SWIFT_DIR / "WebSignIn.swift")
    assert set(re.findall(r'"(web_[a-z_]+)"', re.search(r"static let methods: Set<String> = \[(.*?)\]", swift).group(1))) == names
    dart = _text(EXT / "flutter/flet_glossarion_native/lib/src/native_service.dart")
    block = re.search(r"const Set<String> _kSignInMethods = <String>\{(.*?)\};", dart, re.S).group(1)
    assert set(re.findall(r"'([a-z_]+)'", block)) == names
    assert all(f"case '{name}':" in dart for name in names) and "_kSignInMethods.contains(name)" in dart
    assert "WebSignIn.methods.contains(call.method)" in _text(SWIFT_DIR / "GlossarionNativePlugin.swift")
    assert "method in WebSignIn.METHODS" in _text(KOTLIN_DIR / "HiddenWebPages.kt")
    import flet_glossarion_native.native as gn

    for name in names:
        assert asyncio.iscoroutinefunction(getattr(gn.GlossarionNative, name)), name


def test_platforms_read_the_arguments_python_sends():
    sent = {"web_sign_in": {"url", "done_url", "done_cookie", "cookie_url", "user_agent", "title"},
            "web_cookies": {"url"}, "web_clear_cookies": {"url", "domain"}}
    native_py = _text(EXT / "flet_glossarion_native/native.py")
    for method, keys in sent.items():
        body = native_py[native_py.index(f"async def {method}("):]
        body = body[:body.index("\n    async def ") if "\n    async def " in body else len(body)]
        assert keys <= set(re.findall(r'"([a-z_]+)":', body)), method
    kotlin = _text(KOTLIN_DIR / "HiddenWebPages.kt")
    for method, keys in sent.items():
        branch = kotlin[kotlin.index(f'"{method}" ->'):]
        branch = branch[:branch.index("\n                \"web_", 1) if "\n                \"web_" in branch[1:] else 600]
        assert keys <= set(re.findall(r'call\.argument<String>\("([a-z_]+)"\)', branch)), (method, branch[:200])
    swift = _text(SWIFT_DIR / "WebSignIn.swift")
    read = set(re.findall(r'args\["([a-z_]+)"\]', swift))
    assert sent["web_sign_in"] | {"domain"} <= read | {"url"}
    assert 'cookies(args["url"]' in swift and 'clearCookies(args["domain"]' in swift


def test_the_sign_in_announces_a_browser_agent_not_a_web_view():
    kotlin = _text(KOTLIN_DIR / "WebSignIn.kt")
    assert 'replace("; wv)", ")")' in kotlin and "Version/" in kotlin
    assert "customUserAgent" in _text(SWIFT_DIR / "WebSignIn.swift")


def test_capability_announces_the_sign_in():
    assert '"sign_in" to true' in _text(KOTLIN_DIR / "HiddenWebPages.kt")
    assert '"sign_in": true' in _text(SWIFT_DIR / "HiddenWebPages.swift")
    assert WP.capability({"native": True, "web_pages": {"version": 1, "sign_in": True}})["sign_in"] is True


# ---------------------------------------------------------------- the model sheet route row

def test_route_row_shows_the_google_chip(phone, monkeypatch):
    ft = pytest.importorskip("flet")
    from glossarion_mobile.services import model_catalog as mc
    from glossarion_mobile.ui.sheets.model_sheet import ModelSheet, SheetEnv

    _fake, _bridge, _dispatcher, service = phone
    gs._current = service
    sheet = ModelSheet(env=SheetEnv(), current_model="authgem/gemini-4-neon")
    info = mc.RouteInfo(model="authgem/gemini-4-neon", logins=("google",), needs_key=False)
    rows = sheet._route_controls(info)
    chip = rows[0].controls[0]
    assert isinstance(chip, ft.Chip) and chip.key == "route-login-google"
    assert chip.label.value == "Sign in with Google"
    service._mark(True)
    chip = sheet._route_controls(info)[0].controls[0]
    assert chip.label.value == "Google ✓"
    gs._current = None
    assert sheet._route_controls(info)[0].controls[0].key != "route-login-google"
