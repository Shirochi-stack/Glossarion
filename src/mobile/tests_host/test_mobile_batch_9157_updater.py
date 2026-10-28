"""v9.15.7 owner batch, issue 6: "Auto updater in mobile is installing the apk from github for the user,
requiring extra manual work from the user. Can this be done automatically?"

Android now installs in the app (``services.updates.ApkInstaller``): the APK for this ABI (the newest
``_hotfixN`` of the release) is downloaded with the desktop's verified downloader (exact size + SHA-256),
then handed to the system installer through the extension (``install_package``; FileProvider + ACTION_VIEW).
The first time, one tap opens "Install unknown apps" for Glossarion and coming back installs. A fully
silent install is impossible for a sideloaded app; iOS keeps the IPA links and says why.

Covered here (host only; the Kotlin/Dart side is checked by reading its sources):
* the newest hotfix wins, the installed build is recognised by size + SHA-256 of base.apk, and a newer
  hotfix of the running release is offered (status ``hotfix``);
* the verified download (scripted GitHub), the ready file reused on a later visit, a bad digest refused;
* the installer flow with a fake native bridge: download -> permission -> settings -> resume -> installer;
* the screen and the feature (fake context, and the real app on a fake Flet session);
* every layer names the three native methods the same way, the manifest declares the permission and the
  provider, and the stub / bridge answer safely off-device.

Run from src/mobile:
    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic -q tests_host/test_mobile_batch_9157_updater.py
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import re
import sys
import types
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = MOBILE_DIR / "app"
SRC_DIR = MOBILE_DIR.parent
for entry in (str(APP_DIR),):
    if entry not in sys.path:
        sys.path.insert(0, entry)
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from glossarion_mobile.services import native as native_mod  # noqa: E402
from glossarion_mobile.services import updates as up  # noqa: E402

pytest.importorskip("requests")
pytest.importorskip("packaging")

EXT = MOBILE_DIR / "extensions" / "flet_glossarion_native"
FLUTTER = EXT / "src" / "flutter" / "flet_glossarion_native"
DART = FLUTTER / "lib" / "src" / "native_service.dart"
KOTLIN_DIR = FLUTTER / "android" / "src" / "main" / "kotlin" / "com" / "glossarion" / "flet_glossarion_native"
MANIFEST = FLUTTER / "android" / "src" / "main" / "AndroidManifest.xml"
PATHS_XML = FLUTTER / "android" / "src" / "main" / "res" / "xml" / "glossarion_native_update_paths.xml"
ANDROID_NS = "{http://schemas.android.com/apk/res/android}"
METHODS = ("can_install_packages", "open_install_settings", "install_package")

TAG = "v9.15.7"
GH = f"https://github.com/Shirochi-stack/Glossarion/releases/download/{TAG}/"
BASE_ARM = f"X_Glossarion_{TAG}_Android_arm64-v8a.apk"
HOTFIX_ARM = f"X_Glossarion_{TAG}_hotfix1_Android_arm64-v8a.apk"


def _has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


needs_flet = pytest.mark.skipif(not _has("flet"), reason="flet not installed")


def _text(path: Path) -> str:
    return path.read_bytes().decode("utf-8").replace("\r\n", "\n")


def _payload(seed: str, size: int = 4096) -> bytes:
    return (seed.encode() * (size // len(seed) + 1))[:size]


PAYLOADS = {
    BASE_ARM: _payload("base-arm"),
    HOTFIX_ARM: _payload("hotfix1-arm", 4100),
    f"X_Glossarion_{TAG}_Android_x86_64.apk": _payload("base-x64", 4200),
    f"X_Glossarion_{TAG}_hotfix1_Android_x86_64.apk": _payload("hotfix1-x64", 4300),
}


def _release() -> dict:
    """v9.15.7 as GitHub lists it (names from the real release; small payloads, real digests)."""
    assets = [{"name": name, "size": len(data), "browser_download_url": GH + name,
               "digest": "sha256:" + hashlib.sha256(data).hexdigest()} for name, data in PAYLOADS.items()]
    assets += [{"name": f"X_Glossarion_{TAG}_hotfix1_iOS_unsigned.ipa", "size": 9000,
                "browser_download_url": GH + f"X_Glossarion_{TAG}_hotfix1_iOS_unsigned.ipa"},
               {"name": f"Glossarion.{TAG}.exe", "size": 383233237, "browser_download_url": GH + "Glossarion.exe"}]
    return {"tag_name": TAG, "body": "notes", "html_url": f"https://github.com/x/y/releases/tag/{TAG}",
            "published_at": "2026-10-27T10:00:00Z", "assets": assets}


class _Response:
    def __init__(self, payload=None, *, raw: bytes = b"", status=200):
        self.status_code = status
        self._payload = payload
        self._raw = raw
        self.headers = {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def close(self):
        pass

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        data = self._raw if self._payload is None else json.dumps(self._payload).encode()
        for start in range(0, len(data), 1000):
            yield data[start:start + 1000]


@pytest.fixture
def github(monkeypatch):
    import requests

    state = types.SimpleNamespace(release=_release(), calls=[], payloads=dict(PAYLOADS))

    def get(url, headers=None, timeout=None, **kwargs):
        state.calls.append(url)
        if url.endswith("/releases/latest"):
            return _Response(state.release)
        if "per_page=" in url:
            return _Response([state.release])
        if url.startswith(GH):
            return _Response(raw=state.payloads[url[len(GH):]])
        raise AssertionError(url)

    monkeypatch.setattr(requests, "get", get)
    return state


def _service(tmp_path, *, version="9.15.7", installed: bytes | None = None, platform="android"):
    apk = None
    if installed is not None:
        apk = tmp_path / "data_app" / "base.apk"
        apk.parent.mkdir(parents=True, exist_ok=True)
        apk.write_bytes(installed)
    service = up.UpdateService(None, version, platform=platform, arch="arm64",
                               installed_apk=lambda: str(apk) if apk else None)
    service.checker.MIN_UPDATE_SIZE = 1000  # the real APKs are ~300 MB; these payloads are 4 kB
    return service


# ---------------------------------------------------------------------------
# hotfix choice and the installed build
# ---------------------------------------------------------------------------

def test_newest_hotfix_of_this_abi_is_picked():
    assert up.hotfix_number(HOTFIX_ARM) == 1 and up.hotfix_number(BASE_ARM) == 0
    assert up.hotfix_number("X_Glossarion_v9.15.3_hotfix12_Android_x86_64.apk") == 12
    downloads = up.select_downloads(_release(), "android", "arm64")
    assert downloads.apk.name == HOTFIX_ARM and downloads.apk.hotfix == 1
    assert downloads.apk.sha256 == hashlib.sha256(PAYLOADS[HOTFIX_ARM]).hexdigest()
    assert up.select_downloads(_release(), "android", "x64").apk.name.endswith("_hotfix1_Android_x86_64.apk")
    assert BASE_ARM in {a.name for a in downloads.other_apks}


def test_own_apk_path_reads_the_process_maps(tmp_path):
    maps = tmp_path / "maps"
    maps.write_text(
        "7f00-7f10 r--p 00000000 fd:01 1 /system/lib64/libc.so\n"
        "7f20-7f30 r--s 00012000 fd:05 9 /data/app/~~Ab==/com.glossarion.app-Cd==/base.apk\n"
        "7f40-7f50 rw-p 00000000 00:00 0 \n", encoding="utf-8")
    assert up.own_apk_path(str(maps)) == "/data/app/~~Ab==/com.glossarion.app-Cd==/base.apk"
    assert up.own_apk_path(str(tmp_path / "missing")) is None


def test_running_release_build_is_offered_its_hotfix(tmp_path, github, capsys):
    result = _service(tmp_path, installed=PAYLOADS[BASE_ARM]).check(manual=True)
    assert result.status == "hotfix" and result.installed == BASE_ARM
    assert result.message == f"Hotfix 1 for Glossarion {TAG} is available."
    assert result.install_asset.name == HOTFIX_ARM

    on_hotfix = _service(tmp_path / "b", installed=PAYLOADS[HOTFIX_ARM]).check(manual=True)
    assert on_hotfix.status == "current" and on_hotfix.installed == HOTFIX_ARM and on_hotfix.install_asset is None
    assert on_hotfix.message == "You are up to date (9.15.7)."

    # a CI build (bytes match no release file): the hotfix is offered, and the screen says it cannot tell
    unknown = _service(tmp_path / "c", installed=b"x" * len(PAYLOADS[BASE_ARM])).check(manual=True)
    assert unknown.status == "current" and unknown.installed == "" and unknown.install_asset.name == HOTFIX_ARM
    assert "cannot tell" in unknown.message

    newer = _service(tmp_path / "d", version="9.15.6", installed=PAYLOADS[BASE_ARM]).check(manual=True)
    assert newer.status == "update" and newer.install_asset.name == HOTFIX_ARM and newer.installed == ""
    ios = _service(tmp_path / "e", platform="ios").check(manual=True)
    assert ios.install_asset is None and ios.downloads.ipa is not None
    capsys.readouterr()


# ---------------------------------------------------------------------------
# the verified download
# ---------------------------------------------------------------------------

def test_download_is_verified_and_reused(tmp_path, github, capsys):
    service = _service(tmp_path, installed=PAYLOADS[BASE_ARM])
    result = service.check(manual=True)
    folder = tmp_path / "cache" / up.UPDATES_SUBDIR
    folder.mkdir(parents=True)
    (folder / "X_Glossarion_v9.15.6_Android_arm64-v8a.apk").write_bytes(b"old")  # an earlier update's file
    ticks: list = []
    path = service.download_apk(result.install_asset, str(folder), lambda *a: ticks.append(a), lambda: False)
    assert Path(path).read_bytes() == PAYLOADS[HOTFIX_ARM] and ticks[-1][0] == 100
    assert sorted(p.name for p in folder.iterdir()) == [HOTFIX_ARM, HOTFIX_ARM + up.VERIFIED_SUFFIX]
    assert service.ready_apk(result.install_asset, str(folder)) == path

    # a tampered file (or digest) is never offered as ready, and a bad download leaves nothing behind
    Path(path).write_bytes(PAYLOADS[HOTFIX_ARM][:-1])
    assert service.ready_apk(result.install_asset, str(folder)) is None
    github.payloads[HOTFIX_ARM] = _payload("evil", len(PAYLOADS[HOTFIX_ARM]))
    with pytest.raises(Exception, match="SHA-256"):
        service.download_apk(result.install_asset, str(folder), lambda *a: None, lambda: False)
    assert not any(p.suffix == ".apk" for p in folder.iterdir())
    service.prune_downloads(str(folder))
    assert list(folder.iterdir()) == []
    capsys.readouterr()


# ---------------------------------------------------------------------------
# the installer with a fake native bridge
# ---------------------------------------------------------------------------

class _FakeNative:
    is_stub = False

    def __init__(self, cache_dir: str, *, allowed: bool = False):
        self.cache_dir = cache_dir
        self.allowed = allowed
        self.calls: list = []
        self.answer = {"ok": True, "error": None, "message": ""}

    async def platform_info(self):
        return {"platform": "android", "native": True, "cache_dir": self.cache_dir}

    async def can_install_packages(self):
        self.calls.append("can_install_packages")
        return self.allowed

    async def open_install_settings(self):
        self.calls.append("open_install_settings")
        return True

    async def install_package(self, path):
        self.calls.append(("install_package", path))
        return dict(self.answer)


def test_installer_downloads_asks_once_and_opens_the_system_installer(tmp_path, github, capsys):
    async def scenario():
        service = _service(tmp_path, installed=PAYLOADS[BASE_ARM])
        result = service.check(manual=True)
        native = _FakeNative(str(tmp_path / "cache"))
        installer = up.ApkInstaller(service, native)
        seen: list = []
        installer.subscribe(lambda progress_only: seen.append((installer.state, progress_only)))

        outcome = await installer.download(result.install_asset)
        assert outcome == "permission" and installer.state == "permission"
        assert installer.path == str(tmp_path / "cache" / "updates" / HOTFIX_ARM)
        assert ("downloading", False) in seen
        assert await installer.resumed() is None  # back without allowing: nothing happens
        assert await installer.allow_installs() and "open_install_settings" in native.calls
        native.allowed = True
        assert await installer.resumed() == "installer" and installer.state == "installing"
        assert native.calls[-1] == ("install_package", installer.path)

        # a later visit finds the verified file and installs it without a new download
        github.calls.clear()
        again = up.ApkInstaller(service, native)
        await again.adopt(result.install_asset)
        assert again.state == "ready" and again.path == installer.path
        assert await again.download(result.install_asset) == "installer" and not github.calls

        # the cache was cleared under it
        native.answer = {"ok": False, "error": "missing", "message": "gone"}
        assert await again.install() == "missing" and again.state == "error" and "Download it again" in again.error

        # once the update is installed, the next check drops its APK
        await again.adopt(None)
        assert list((tmp_path / "cache" / "updates").iterdir()) == []

        # off-device: no cache folder from the extension -> a clear error, the browser link stays
        stub = up.ApkInstaller(service, native_mod.NativeBridge(native=native_mod.NativeStub()))
        assert await stub.download(result.install_asset) == "error" and "Android app" in stub.error

    asyncio.run(scenario())
    capsys.readouterr()


def test_installer_cancel(tmp_path, github, capsys):
    async def scenario():
        service = _service(tmp_path, installed=PAYLOADS[BASE_ARM])
        result = service.check(manual=True)
        installer = up.ApkInstaller(service, _FakeNative(str(tmp_path / "cache"), allowed=True))
        original = service.download_apk

        def cancelling(asset, folder, progress, cancelled):
            installer.cancel()
            return original(asset, folder, progress, cancelled)

        service.download_apk = cancelling
        assert await installer.download(result.install_asset) == "cancelled" and installer.state == "idle"
        assert not any(p.suffix == ".apk" for p in (tmp_path / "cache" / "updates").iterdir())

    asyncio.run(scenario())
    capsys.readouterr()


# ---------------------------------------------------------------------------
# the screen and the feature
# ---------------------------------------------------------------------------

class _Ctx:
    page = None

    def __init__(self):
        self.messages: list = []
        self.tasks: list = []

    def say(self, message, *_a):
        self.messages.append(message)

    async def run_io(self, fn, *args):
        return fn(*args)

    def spawn(self, coro):
        task = asyncio.ensure_future(coro)
        self.tasks.append(task)
        return task


def _walk(control, out):
    if control is None or isinstance(control, str):
        return
    out.append(control)
    for attr in ("content", "title", "leading", "trailing"):
        _walk(getattr(control, attr, None), out)
    for child in getattr(control, "controls", None) or []:
        _walk(child, out)


def _find(root, prefix):
    nodes: list = []
    _walk(root, nodes)
    return [n for n in nodes if str(getattr(n, "key", "") or "").startswith(prefix)]


def _labels(root) -> list:
    nodes: list = []
    _walk(root, nodes)
    return [n.content if isinstance(getattr(n, "content", None), str) else getattr(n, "value", None) for n in nodes]


@needs_flet
def test_screen_downloads_and_installs_in_the_app(tmp_path, github, capsys):
    from glossarion_mobile.ui.router import parse_route
    from glossarion_mobile.ui.screens.updates import UpdatesScreen

    async def scenario():
        service = _service(tmp_path, installed=PAYLOADS[BASE_ARM])
        native = _FakeNative(str(tmp_path / "cache"))
        installer = up.ApkInstaller(service, native)
        ctx = _Ctx()
        opened: list = []
        screen = UpdatesScreen(parse_route("/settings/updates"), ctx, service=service, open_url=opened.append,
                               installer=installer)
        body = screen.build_body()
        note = _find(body, "updates-note")[0].value
        assert "Android always asks you to confirm" in note and "first time" in note
        await screen.check_now()
        await asyncio.gather(*ctx.tasks)
        card = screen.release_holder.content
        assert f"Installed: {BASE_ARM}" in _labels(card)
        start = _find(card, "updates-install-start-")
        assert len(start) == 1 and start[0].content == "Download and install hotfix 1 (0 MB)"
        fallback = _find(card, "updates-apk-")[0]
        assert fallback.title.value.startswith("Download in the browser instead")
        fallback.on_click(None)
        assert opened == [GH + HOTFIX_ARM]

        start[0].on_click(None)  # Download and install
        await asyncio.gather(*ctx.tasks)
        assert installer.state == "permission"
        allow = _find(screen.install_box, "updates-install-allow-")
        assert len(allow) == 1 and allow[0].content == "Allow installs from Glossarion"
        allow[0].on_click(None)
        await asyncio.gather(*ctx.tasks)
        assert native.calls.count("open_install_settings") == 1
        native.allowed = True
        screen.app_resumed()  # back from Android's settings page
        await asyncio.gather(*ctx.tasks)
        assert installer.state == "installing" and native.calls[-1] == ("install_package", installer.path)
        again = _find(screen.install_box, "updates-install-go-")
        assert again and again[0].content == "Install again"
        screen.dispose()
        assert installer._listeners == []

    asyncio.run(scenario())
    capsys.readouterr()


@needs_flet
def test_ios_says_why_it_cannot_self_update(tmp_path, github, capsys):
    from glossarion_mobile.ui.router import parse_route
    from glossarion_mobile.ui.screens.updates import UpdatesScreen

    service = _service(tmp_path, platform="ios")
    screen = UpdatesScreen(parse_route("/settings/updates"), _Ctx(), service=service)
    body = screen.build_body()
    note = _find(body, "updates-note")[0].value
    assert "iOS does not let an app replace itself" in note and "AltStore" in note
    asyncio.run(screen.check_now())
    assert not _find(screen.release_holder.content, "updates-install-")
    assert _find(screen.release_holder.content, "updates-ipa-")
    capsys.readouterr()


@needs_flet
def test_feature_builds_the_installer_on_the_android_app_only():
    from glossarion_mobile.ui.screens import updates as screens

    def app(platform, native):
        return types.SimpleNamespace(page=types.SimpleNamespace(platform=types.SimpleNamespace(value=platform)),
                                     native=native, dispatcher=None)

    service = types.SimpleNamespace()
    assert screens.UpdatesFeature(app("android", _FakeNative("/c")), service=service).installer is not None
    stub = native_mod.NativeBridge(native=native_mod.NativeStub())
    assert screens.UpdatesFeature(app("android", stub), service=service).installer is None
    assert screens.UpdatesFeature(app("ios", _FakeNative("/c")), service=service).installer is None
    assert screens.UpdatesFeature(app("android", None), service=service).installer is None


_TB_SPEC = importlib.util.spec_from_file_location("_glossarion_tb_helpers_b9157u", Path(__file__).with_name("test_bootstrap.py"))
_TB = importlib.util.module_from_spec(_TB_SPEC)
_TB_SPEC.loader.exec_module(_TB)
storage = _TB.storage
app_env = _TB.app_env


@pytest.mark.skipif(not (_has("flet") and _has("msgpack")), reason="flet/msgpack not installed")
def test_install_step_renders_on_the_real_app(app_env, tmp_path, github, caplog, capsys):
    """The real app on a fake Flet session serialises the install step through every state."""
    from glossarion_mobile.ui.router import build_route
    from glossarion_mobile.ui.screens.updates import UpdatesFeature, UpdatesScreen

    spec = importlib.util.spec_from_file_location("_glossarion_tf_helpers_b9157u",
                                                  Path(__file__).with_name("test_ui_foundations.py"))
    tf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tf)

    async def scenario():
        _m, _conn, _session, _page, app = await tf._start("android")
        try:
            service = _service(tmp_path, installed=PAYLOADS[BASE_ARM])
            native = _FakeNative(str(tmp_path / "cache"), allowed=True)
            installer = up.ApkInstaller(service, native)
            await UpdatesFeature.install(app, service=service, installer=installer, startup_delay=3600)
            await app.navigate(build_route("settings.updates"))
            screen = app.shell.top_screen
            assert isinstance(screen, UpdatesScreen) and screen.installer is installer
            await screen.check_now()
            await asyncio.sleep(0.1)
            assert await screen.download_and_install(screen.result.install_asset) == "installer"
            await asyncio.sleep(0.2)
            assert installer.state == "installing" and _find(screen.install_box, "updates-install-go-")
            errors = [r for r in caplog.records if r.levelno >= 40 and r.name.startswith("glossarion")]
            assert not errors, [r.getMessage() for r in errors]
        finally:
            jobs = getattr(app, "jobs", None)
            if jobs is not None:
                jobs.close()
            await tf._stop(app)

    asyncio.run(scenario())
    capsys.readouterr()


# ---------------------------------------------------------------------------
# native layers agree (sources read, nothing compiled)
# ---------------------------------------------------------------------------

def test_native_layers_name_the_install_methods_alike():
    dart = _text(DART)
    invoke = dart[dart.index("Future<dynamic> _invokeMethod"):dart.index("dynamic _defaultResult")]
    defaults = dart[dart.index("dynamic _defaultResult"):dart.index("Future<dynamic> _platform(")]
    kotlin = _text(KOTLIN_DIR / "GlossarionNativePlugin.kt")
    updates_kt = _text(KOTLIN_DIR / "AppUpdates.kt")
    ext_py = _text(EXT / "src" / "flet_glossarion_native" / "native.py")
    for name in METHODS:
        assert f"case '{name}':" in invoke and f"case '{name}':" in defaults, name
        assert f'"{name}" ->' in kotlin, name
        assert f'"{name}"' in ext_py, name
        assert hasattr(native_mod.NativeBridge, name) and hasattr(native_mod.NativeStub, name), name
    assert "_isAndroid ? (await _platform(name, args)) == true : false" in invoke  # never on iOS
    assert 'call.argument<String>("path")' in kotlin
    # only <cache>/updates, through Glossarion's own FileProvider subclass
    assert "class GlossarionUpdateFileProvider : FileProvider()" in updates_kt
    assert 'const val UPDATES_DIR = "updates"' in updates_kt and up.UPDATES_SUBDIR == "updates"
    assert "file.parentFile != root" in updates_kt and "FLAG_GRANT_READ_URI_PERMISSION" in updates_kt
    assert "ACTION_MANAGE_UNKNOWN_APP_SOURCES" in updates_kt and "canRequestPackageInstalls()" in updates_kt


def test_manifest_permission_provider_and_paths():
    root = ET.parse(MANIFEST).getroot()
    permissions = {p.get(ANDROID_NS + "name") for p in root.findall("uses-permission")}
    assert "android.permission.REQUEST_INSTALL_PACKAGES" in permissions
    provider = root.find("application").find("provider")
    assert provider.get(ANDROID_NS + "name") == "com.glossarion.flet_glossarion_native.GlossarionUpdateFileProvider"
    assert provider.get(ANDROID_NS + "authorities") == "${applicationId}.glossarion_native.updates"
    assert provider.get(ANDROID_NS + "exported") == "false" and provider.get(ANDROID_NS + "grantUriPermissions") == "true"
    meta = provider.find("meta-data")
    assert meta.get(ANDROID_NS + "resource") == "@xml/glossarion_native_update_paths"
    paths = ET.parse(PATHS_XML).getroot()
    assert [(p.tag, p.get("path")) for p in paths] == [("cache-path", "updates/")]
    assert 'AUTHORITY_SUFFIX = ".glossarion_native.updates"' in _text(KOTLIN_DIR / "AppUpdates.kt")
    pyproject = _text(MOBILE_DIR / "pyproject.toml")
    assert re.search(r'^"android\.permission\.REQUEST_INSTALL_PACKAGES" = true', pyproject, re.M)


def test_stub_and_bridge_answer_safely_off_device():
    async def scenario():
        bridge = native_mod.NativeBridge(native=native_mod.NativeStub())
        assert await bridge.can_install_packages() is False
        assert await bridge.open_install_settings() is False
        answer = await bridge.install_package("/x.apk")
        assert answer["ok"] is False and answer["error"] == "unavailable"

    asyncio.run(scenario())
