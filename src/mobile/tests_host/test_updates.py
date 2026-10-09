"""Host tests for About › Updates (U9): ``services/updates.py`` and ``ui/screens/updates.py``.

* asset classification and the per-device choice (APK for this ABI, release-signed first; the
  IPAs on iOS; desktop-only releases offer nothing, which is not an error). Mobile builds are
  never published (owner's rule, 2026-10-09), so the files only the removed release job wrote
  (``altstore-source.json``, ``<prefix>_mobile_SHA256SUMS.txt``) are not mobile assets any more;
* ``UpdateService`` runs the desktop checker (``update_core.HeadlessUpdateChecker``) on a real
  ``MobileConfigStore``: the shared keys ``last_update_check_time`` / ``skipped_versions`` /
  ``auto_update_check`` are the only ones written, the 30-minute cache and the skip behave as
  on the desktop, network and GitHub errors become messages;
* the screen (fake context): Check now, Skip this version, Check on startup, link opening;
* the feature: the route goes to the screen, the Settings home marks it implemented, the
  startup check runs only on a phone, not under automation, and only notifies when a newer
  release has a file for this device; on the real app (fake Flet session) the page renders.

No network: ``requests.get`` is scripted.

Run from src/mobile:
    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_updates.py
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = MOBILE_DIR / "app"
SRC_DIR = MOBILE_DIR.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from glossarion_mobile.services import updates as up  # noqa: E402


def _has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


needs_flet = pytest.mark.skipif(not _has("flet"), reason="flet not installed")
needs_requests = pytest.mark.skipif(not (_has("requests") and _has("packaging")), reason="requests/packaging missing")

PREFIX = "Glossarion_v9.15.0"
GH = "https://github.com/Shirochi-stack/Glossarion/releases/download/v9.15.0/"


def _asset(name: str, size: int = 150 * 1024 * 1024) -> dict:
    return {"name": name, "size": size, "browser_download_url": GH + name}


def _mobile_release(tag: str = "v9.15.0", *, debug_only: bool = False) -> dict:
    names = [f"{PREFIX}_Android_arm64-v8a_debugsigned.apk", f"{PREFIX}_Android_x86_64_debugsigned.apk"]
    if not debug_only:
        names += [f"{PREFIX}_Android_arm64-v8a.apk", f"{PREFIX}_Android_x86_64.apk", f"{PREFIX}_Android.aab"]
    names += [f"{PREFIX}_iOS_unsigned.ipa", f"{PREFIX}_iOS.ipa", "Glossarion v9.15.0.exe",
              "L_Glossarion_Lite v9.15.0.exe", "Glossarion_v9.15.0_MAC.dmg", "Glossarion-9.15.0-Linux.zip"]
    return {"tag_name": tag, "body": "## What's new\n\n* faster", "html_url": f"https://github.com/x/y/releases/tag/{tag}",
            "published_at": "2026-10-09T10:00:00Z", "assets": [_asset(n) for n in names]}


def _desktop_release(tag: str = "v9.15.0") -> dict:
    release = _mobile_release(tag)
    release["assets"] = [a for a in release["assets"] if up.classify_asset(a["name"]) is None]
    return release


# ---------------------------------------------------------------------------
# asset choice
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,kind", [
    (f"{PREFIX}_Android_arm64-v8a.apk", "apk"),
    (f"{PREFIX}_Android_x86_64_debugsigned.apk", "apk"),
    (f"{PREFIX}_Android.aab", "aab"),
    (f"{PREFIX}_iOS_unsigned.ipa", "ipa"),
    (f"{PREFIX}_iOS.ipa", "ipa_signed"),
    # written only by the removed release job: never a mobile asset now
    ("altstore-source.json", None),
    (f"{PREFIX}_mobile_SHA256SUMS.txt", None),
    ("Glossarion v9.15.0.exe", None),
    ("Glossarion_v9.15.0_MAC.dmg", None),
    ("random_Android_arm64-v8a.apk", None),
    ("", None),
])
def test_classify_asset(name, kind):
    assert up.classify_asset(name) == kind


@needs_requests
def test_android_picks_this_abi_release_signed_first():
    downloads = up.select_downloads(_mobile_release(), "android", "arm64")
    assert downloads.available and downloads.apk.name == f"{PREFIX}_Android_arm64-v8a.apk"
    assert not downloads.apk.debug_signed and downloads.apk.abi == "arm64-v8a" and downloads.apk.size_mb == 150
    assert {a.name for a in downloads.other_apks} == {
        f"{PREFIX}_Android_arm64-v8a_debugsigned.apk", f"{PREFIX}_Android_x86_64_debugsigned.apk",
        f"{PREFIX}_Android_x86_64.apk"}
    x64 = up.select_downloads(_mobile_release(debug_only=True), "android", "x64")
    assert x64.apk.name == f"{PREFIX}_Android_x86_64_debugsigned.apk" and x64.apk.debug_signed
    unknown = up.select_downloads(_mobile_release(), "android", "unknown")
    assert unknown.apk is None and len(unknown.other_apks) == 4 and unknown.available  # listed, never picked


@needs_requests
def test_ios_offers_the_ipas():
    downloads = up.select_downloads(_mobile_release(), "ios", "arm64")
    assert downloads.available and downloads.apk is None and downloads.other_apks == []
    assert downloads.ipa.name == f"{PREFIX}_iOS_unsigned.ipa" and downloads.ipa.kind == "ipa"
    assert downloads.ipa_signed.name == f"{PREFIX}_iOS.ipa" and downloads.ipa_signed.kind == "ipa_signed"
    release = _mobile_release()
    release["assets"] = [a for a in release["assets"] if a["name"] != f"{PREFIX}_iOS.ipa"]
    unsigned_only = up.select_downloads(release, "ios", "arm64")
    assert unsigned_only.available and unsigned_only.ipa_signed is None


@needs_requests
def test_no_altstore_source_or_checksums_support():
    """The AltStore source and the checksums file came only from the removed release job."""
    assert not hasattr(up, "altstore_links") and not hasattr(up, "ALTSTORE_SOURCE_ASSET")
    fields = set(up.MobileDownloads.__dataclass_fields__)
    assert fields == {"platform", "apk", "other_apks", "ipa", "ipa_signed"}
    # an old release that still carries those files next to the desktop ones offers nothing
    release = _desktop_release()
    release["assets"] += [_asset("altstore-source.json", 900), _asset(f"{PREFIX}_mobile_SHA256SUMS.txt", 400)]
    for platform in ("android", "ios"):
        assert not up.select_downloads(release, platform, "arm64").available


@needs_requests
def test_desktop_only_release_offers_nothing():
    for platform in ("android", "ios", "windows"):
        downloads = up.select_downloads(_desktop_release(), platform, "arm64")
        assert not downloads.available and downloads.apk is None and downloads.ipa is None
    assert not up.select_downloads(None, "android", "arm64").available
    assert not up.select_downloads({"tag_name": "v9.15.0"}, "ios", "arm64").available  # no assets key
    # a forged asset URL is never offered
    release = _mobile_release()
    release["assets"] = [dict(a, browser_download_url="http://evil.invalid/x") for a in release["assets"]]
    assert not up.select_downloads(release, "android", "arm64").available


@needs_requests
def test_releases_page_comes_from_the_desktop_api_url():
    assert up.releases_page() == "https://github.com/Shirochi-stack/Glossarion/releases"


# ---------------------------------------------------------------------------
# UpdateService on a real MobileConfigStore
# ---------------------------------------------------------------------------

class _Response:
    def __init__(self, payload=None, status=200):
        self.status_code = status
        self._payload = payload

    def raise_for_status(self):
        import requests

        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}", response=self)

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


@pytest.fixture
def github(monkeypatch):
    """Scripted GitHub: ``github.latest`` is the /releases/latest payload (or an exception)."""
    import requests

    state = types.SimpleNamespace(latest=_mobile_release(), calls=[])

    def get(url, headers=None, timeout=None):
        state.calls.append(url)
        if isinstance(state.latest, Exception):
            raise state.latest
        if url.endswith("/releases/latest"):
            return state.latest if isinstance(state.latest, _Response) else _Response(state.latest)
        if "per_page=" in url:
            return _Response([state.latest] if isinstance(state.latest, dict) else [])
        raise AssertionError(url)

    monkeypatch.setattr(requests, "get", get)
    return state


def _store(tmp_path, config=None):
    pytest.importorskip("cryptography")
    from glossarion_mobile.state.config_store import MobileConfigStore

    path = tmp_path / "config.json"
    if config is not None:
        path.write_text(json.dumps(config), encoding="utf-8")
    store = MobileConfigStore(str(path), debounce=30)
    store.load()
    return store


@needs_requests
def test_service_check_update_skip_and_cache(tmp_path, github, capsys):
    store = _store(tmp_path, {"model": "authgpt/gpt-6-luna", "auto_update_check": False})
    before = {k: store.get(k) for k in store.keys()}
    service = up.UpdateService(store, "9.14.0", platform="android", arch="arm64", clock=lambda: 42.0)
    assert not service.startup_enabled() and service.last_checked() == 0

    result = service.check(manual=True)
    assert result.status == "update" and result.tag == "v9.15.0" and result.checked_at == 42.0
    assert result.message == "Glossarion v9.15.0 is available." and result.published == "2026-10-09"
    assert result.downloads.apk.name.endswith("_Android_arm64-v8a.apk")
    assert len(github.calls) == 2  # /releases/latest + history, as on the desktop
    assert store.get("last_update_check_time") > 0 and service.last_checked() == store.get("last_update_check_time")

    # the startup check honours the 30-minute window: no request
    cached = service.check(manual=False)
    assert cached.status == "cached" and len(github.calls) == 2 and service.last is result

    # Skip this version, then the startup check (after the window) stays quiet; Check now still shows it
    assert service.skip() == "v9.15.0" and service.skipped_versions() == ["v9.15.0"]
    store.set("last_update_check_time", 0)
    skipped = service.check(manual=False)
    assert skipped.status == "skipped" and "skipped" in skipped.message
    assert service.check(manual=True).status == "update"

    # only the shared update keys changed
    after = {k: store.get(k) for k in store.keys()}
    changed = {k for k in set(before) | set(after) if before.get(k) != after.get(k)}
    assert changed <= set(up.UPDATE_CONFIG_KEYS)
    service.set_startup(True)
    assert store.get("auto_update_check") is True and service.startup_enabled()
    store.close()
    capsys.readouterr()


@needs_requests
def test_service_current_desktop_only_and_errors(tmp_path, github, capsys):
    import requests

    store = _store(tmp_path)
    service = up.UpdateService(store, "9.15.0", platform="android", arch="arm64")
    assert service.startup_enabled()  # the desktop default (auto_update_check missing = on)
    current = service.check(manual=True)
    assert current.status == "current" and current.message == "You are up to date (9.15.0)."

    service = up.UpdateService(store, "9.14.0", platform="android", arch="arm64")
    github.latest = _desktop_release()
    desktop_only = service.check(manual=True)
    assert desktop_only.status == "update" and not desktop_only.downloads.available
    assert desktop_only.message == "Glossarion v9.15.0 is out, but it has no Android build. See the release page."
    ios = up.UpdateService(store, "9.14.0", platform="ios", arch="arm64").check(manual=True)
    assert "no iOS build" in ios.message

    for exc, text in ((requests.Timeout("t"), "timed out"), (requests.ConnectionError("c"), "Cannot reach GitHub"),
                      (_Response({}, 403), "rate limit"), (_Response({}, 502), "error: 502"),
                      (_Response(None), "Invalid response")):
        github.latest = exc
        result = service.check(manual=True)
        assert result.status == "error" and text in result.message, (exc, result.message)
    store.close()
    capsys.readouterr()


# ---------------------------------------------------------------------------
# the screen (fake context) and the feature
# ---------------------------------------------------------------------------

class _Ctx:
    def __init__(self):
        self.page = None
        self.messages: list = []

    def say(self, message, *_a):
        self.messages.append(message)

    async def run_io(self, fn, *args):
        return fn(*args)

    def spawn(self, coro):
        return asyncio.ensure_future(coro)

    def push(self, *controls):
        pass


class _FakeService:
    def __init__(self, result=None, *, startup=True):
        self.current_version = "9.14.0"
        self.last = None
        self._startup = startup
        self.result = result
        self.checks: list = []
        self.skipped: list = []
        self.startup_values: list = []

    def last_checked(self):
        return 0.0

    def startup_enabled(self):
        return self._startup

    def set_startup(self, value):
        self.startup_values.append(value)

    def check(self, *, manual):
        self.checks.append(manual)
        return self.result

    def skip(self):
        self.skipped.append(self.result.tag)
        return self.result.tag


def _result(platform="android", release=None, status="update"):
    release = release or _mobile_release()
    downloads = up.select_downloads(release, platform, "arm64")
    return up.UpdateResult(status=status, message=f"Glossarion {release['tag_name']} is available.",
                           tag=release["tag_name"], notes=release["body"], html_url=release["html_url"],
                           published="2026-10-09", downloads=downloads)


def _texts(control) -> list:
    out = []

    def walk(node):
        if node is None:
            return
        for attr in ("value", "content", "title", "label", "tooltip"):
            value = getattr(node, attr, None)
            if isinstance(value, str):
                out.append(value)
        for attr in ("content", "title", "leading", "trailing"):
            value = getattr(node, attr, None)
            if value is not None and not isinstance(value, str):
                walk(value)
        for child in getattr(node, "controls", None) or []:
            walk(child)

    walk(control)
    return out


def _keys(control) -> list:
    out = []

    def walk(node):
        if node is None or isinstance(node, str):
            return
        key = getattr(node, "key", None)
        if key is not None:
            out.append(str(key))
        for attr in ("content", "title", "leading", "trailing"):
            walk(getattr(node, attr, None))
        for child in getattr(node, "controls", None) or []:
            walk(child)

    walk(control)
    return out


@needs_flet
@needs_requests
def test_screen_check_skip_startup_and_links():
    from glossarion_mobile.ui.router import parse_route
    from glossarion_mobile.ui.screens.updates import UpdatesScreen

    opened: list = []
    service = _FakeService(_result())
    screen = UpdatesScreen(parse_route("/settings/updates"), _Ctx(), service=service, open_url=opened.append)
    body = screen.build_body()
    keys = _keys(body)
    assert {"updates-status", "updates-check", "updates-startup", "updates-release", "updates-note"} <= set(keys)
    assert screen.release_holder.content is None and "Check GitHub for a newer Glossarion release." in _texts(body)

    result = asyncio.run(screen.check_now())
    assert service.checks == [True] and result is service.result
    card = screen.release_holder.content
    texts, card_keys = _texts(card), _keys(card)
    assert "Glossarion v9.15.0" in texts and "Released 2026-10-09" in texts
    assert any(t.startswith("Download APK (arm64-v8a, 150 MB)") for t in texts)
    assert any(k.startswith("updates-skip-") for k in card_keys) and any(k.startswith("updates-notes-") for k in card_keys)
    apk_button = next(c for c in card.content.controls if str(getattr(c, "key", "")).startswith("updates-apk-"))
    apk_button.on_click(None)
    assert opened == [GH + f"{PREFIX}_Android_arm64-v8a.apk"]

    # a second render uses new keys (Flet 1.0.3 freezes a subtree re-created under the same key)
    asyncio.run(screen.check_now())
    assert set(_keys(screen.release_holder.content)).isdisjoint(card_keys)

    assert asyncio.run(screen.skip()) == "v9.15.0" and service.skipped == ["v9.15.0"]
    assert screen.result.status == "skipped" and not any(k.startswith("updates-skip-") for k in _keys(screen.release_holder.content))
    screen._on_startup(types.SimpleNamespace(control=types.SimpleNamespace(value=False)))
    assert service.startup_values == [False]


@needs_flet
@needs_requests
def test_screen_ios_and_desktop_only_rows():
    from glossarion_mobile.ui.router import parse_route
    from glossarion_mobile.ui.screens.updates import UpdatesScreen

    opened: list = []
    screen = UpdatesScreen(parse_route("/settings/updates"), _Ctx(), service=_FakeService(_result("ios")),
                           open_url=opened.append)
    body = screen.build_body()
    assert not any("AltStore" in t or "SideStore" in t for t in _texts(body))
    asyncio.run(screen.check_now())
    texts = _texts(screen.release_holder.content)
    assert any(t.startswith("Unsigned IPA (") for t in texts) and any(t.startswith("Signed IPA (") for t in texts)
    assert not any("AltStore" in t or "SideStore" in t for t in texts)
    assert "This release has no file for this device." not in texts
    ipa_tile = next(c for c in screen.release_holder.content.content.controls
                    if str(getattr(c, "key", "")).startswith("updates-ipa-") and "signed" not in str(c.key))
    ipa_tile.on_click(None)
    assert opened == [GH + f"{PREFIX}_iOS_unsigned.ipa"]

    for platform in ("android", "ios"):
        desktop = _result(platform, _desktop_release())
        screen = UpdatesScreen(parse_route("/settings/updates"), _Ctx(), service=_FakeService(desktop))
        screen.build_body()
        asyncio.run(screen.check_now())
        texts = _texts(screen.release_holder.content)
        assert "This release has no file for this device." in texts and "Release page" in texts
        assert not any(t.startswith(("Download APK", "Unsigned IPA", "Signed IPA")) for t in texts)


def _fake_app(platform="android", *, is_mobile=True, test=False, runtime="android"):
    from glossarion_mobile.state.store import Signal

    class Home:
        def __init__(self):
            self.implemented = frozenset({"settings.logs"})

    def fallback(match):
        return Home() if match.name == "settings" else types.SimpleNamespace(name=match.name)

    notes: list = []
    app = types.SimpleNamespace(
        page=types.SimpleNamespace(platform=types.SimpleNamespace(value=platform), test=test),
        paths=types.SimpleNamespace(platform=runtime),
        shell=types.SimpleNamespace(screen_factory=fallback), is_mobile=is_mobile, config_store=None,
        state=types.SimpleNamespace(backend=Signal(None), selftest_running=Signal(False)),
        notify=lambda *a: notes.append(a), navigate_to=lambda *a, **k: notes.append(("nav",) + a),
        pages_feature=types.SimpleNamespace(ctx=_Ctx()), dispatcher=None, url_launcher=None,
    )
    return app, notes


@needs_flet
@needs_requests
def test_feature_routes_and_startup_check(monkeypatch):
    from glossarion_mobile.ui.router import parse_route
    from glossarion_mobile.ui.screens import updates as screens

    monkeypatch.delenv(screens.DISABLE_ENV, raising=False)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)  # the guard that keeps host tests offline

    async def scenario():
        app, notes = _fake_app()
        service = _FakeService(_result())
        feature = await screens.UpdatesFeature.install(app, service=service, startup_delay=0)
        factory = app.shell.screen_factory
        assert isinstance(factory(parse_route("/settings/updates")), screens.UpdatesScreen)
        assert screens.ROUTE in factory(parse_route("/settings")).implemented
        assert factory(parse_route("/jobs")).name == "jobs"
        assert feature.startup_task is None
        app.state.backend.set({"ok": True})
        assert feature.startup_task is not None
        result = await feature.startup_task
        assert result is service.result and service.checks == [False]
        message, label, action = notes[0]
        assert message == "Glossarion v9.15.0 is available." and label == "View"
        action()
        assert notes[-1] == ("nav", screens.ROUTE)

        # no notification when the release has no file for this device, or the user turned it off
        for result_, startup in ((_result("android", _desktop_release()), True), (_result(), False)):
            app, notes = _fake_app()
            service = _FakeService(result_, startup=startup)
            feature = await screens.UpdatesFeature.install(app, service=service, startup_delay=0)
            app.state.backend.set({"ok": True})
            await feature.startup_task
            assert notes == [] and service.checks == ([False] if startup else [])

        # never in a desktop window, a phone-sized session on a desktop Python, under `flet test`,
        # under pytest, with GLOSSARION_UPDATE_CHECK=0 or a failed backend
        for kwargs in ({"is_mobile": False}, {"runtime": "desktop"}, {"test": True}):
            app, _notes = _fake_app(**kwargs)
            feature = await screens.UpdatesFeature.install(app, service=_FakeService(_result()), startup_delay=0)
            app.state.backend.set({"ok": True})
            assert feature.startup_task is False
        monkeypatch.setenv("PYTEST_CURRENT_TEST", "x")
        app, _notes = _fake_app()
        feature = await screens.UpdatesFeature.install(app, service=_FakeService(_result()), startup_delay=0)
        app.state.backend.set({"ok": True})
        assert feature.startup_task is False
        monkeypatch.delenv("PYTEST_CURRENT_TEST")
        monkeypatch.setenv(screens.DISABLE_ENV, "0")
        app, _notes = _fake_app()
        feature = await screens.UpdatesFeature.install(app, service=_FakeService(_result()), startup_delay=0)
        app.state.backend.set({"ok": True})
        assert feature.startup_task is False
        monkeypatch.delenv(screens.DISABLE_ENV)
        app, _notes = _fake_app()
        feature = await screens.UpdatesFeature.install(app, service=_FakeService(_result()), startup_delay=0)
        app.state.backend.set({"ok": False})
        assert feature.startup_task is None

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# the real app on a fake Flet session
# ---------------------------------------------------------------------------

_TB_SPEC = importlib.util.spec_from_file_location("_glossarion_tb_helpers_u9upd", Path(__file__).with_name("test_bootstrap.py"))
_TB = importlib.util.module_from_spec(_TB_SPEC)
_TB_SPEC.loader.exec_module(_TB)
storage = _TB.storage
app_env = _TB.app_env


def _foundations():
    spec = importlib.util.spec_from_file_location("_glossarion_tf_helpers_u9upd",
                                                  Path(__file__).with_name("test_ui_foundations.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(not (_has("flet") and _has("msgpack")), reason="flet/msgpack not installed")
@needs_requests
def test_updates_page_renders_on_the_real_app(app_env, caplog):
    from glossarion_mobile.ui.router import build_route
    from glossarion_mobile.ui.screens.updates import UpdatesFeature, UpdatesScreen

    tf = _foundations()

    async def scenario():
        _m, _conn, _session, _page, app = await tf._start("android")
        try:
            service = _FakeService(_result())
            feature = await UpdatesFeature.install(app, service=service, startup_delay=3600)
            match = await app.navigate(build_route("settings.updates"))
            assert match is not None and match.name == "settings.updates"
            screen = app.shell.top_screen
            assert isinstance(screen, UpdatesScreen) and feature.screens_built == ["settings.updates"]
            await screen.check_now()  # the session serialises the re-rendered card
            assert service.checks == [True] and screen.release_holder.content is not None
            await asyncio.sleep(0.2)
            errors = [r for r in caplog.records if r.levelno >= 40 and r.name.startswith("glossarion")]
            assert not errors, [r.getMessage() for r in errors]
            await app.navigate(build_route("settings"))
            assert "settings.updates" in app.shell.top_screen.implemented
        finally:
            jobs = getattr(app, "jobs", None)
            if jobs is not None:
                jobs.close()
            await tf._stop(app)

    asyncio.run(scenario())


@needs_flet
@needs_requests
def test_release_note_links_open_web_pages_only():
    from glossarion_mobile.ui.router import parse_route
    from glossarion_mobile.ui.screens.updates import UpdatesScreen

    opened: list = []
    screen = UpdatesScreen(parse_route("/settings/updates"), _Ctx(), service=_FakeService(_result()),
                           open_url=opened.append)
    screen.build_body()
    for url in ("https://github.com/x/y/pull/1", "intent://evil#Intent;end", "file:///data/x", "javascript:alert(1)",
                "altstore://source?url=x", "http://plain.example"):
        screen._open_web(url)
    assert opened == ["https://github.com/x/y/pull/1"]
    release = _mobile_release()
    release["html_url"] = "javascript:alert(1)"
    service = up.UpdateService(None, "9.14.0", platform="android", arch="arm64")
    assert service._describe(release).html_url == "https://github.com/Shirochi-stack/Glossarion/releases"
