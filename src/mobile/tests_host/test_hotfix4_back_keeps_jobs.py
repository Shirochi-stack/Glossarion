"""Hotfix 4 review finding R4: Android back on the chat while a job runs keeps the job's session.

The emulator repro of the owner report ("It has not moved from 7/107 in the jobs menu for several hours"):
back on the chat (the root View) finished MainActivity, which destroyed the Flutter engine. The foreground
service kept the job running, but the app's ``detach`` lifecycle unregistered the in-app browser
(every authchatgpt/ request then failed "needs the in-app browser") and the next session's Jobs page showed
the live job frozen as "Interrupted · N/M" with a Resume button.

The fix, pinned here on the REAL app (``main.main`` on the fake Flet session of tests_host, the shared
``test_ui_foundations`` helpers):

* while a job runs or waits in the queue (the Jobs badge), the root View cannot pop (``can_pop=False``),
  so back reaches ``on_confirm_pop``; on the chat the app then calls the extension's
  ``move_task_to_back`` (Android ``Activity.moveTaskToBack(true)``: the task goes to the background like
  Home, the Activity, the engine and this session stay) and refuses the pop;
* the drawer the menu button opened closes first (Flutter cannot close it through a held root route);
  on a tablet a screen in the main area pops first, as before;
* when the app could not be moved (an old extension, no answer) back leaves the app as before, and with
  no job the root View pops again (the system default);
* only Android: iOS has no system back, desktop none of this;
* the extension contract: Python ``GlossarionNative.move_task_to_back`` -> Dart ``move_task_to_back``
  (Android only, false elsewhere and when the platform lacks it) -> Kotlin ``moveTaskToBack(true)``.

The fake client answers ``move_task_to_back`` like the device does; nothing waits on a sleep. Real data
stays untouched (FLET_APP_STORAGE_*, HOME, OUTPUT_DIRECTORY, GLOSSARION_LIBRARY_DIR and CONFIG_FILE come
from ``app_env`` under ``tmp_path``; USERPROFILE / APPDATA / LOCALAPPDATA too; GLOSSARION_HTTP_LOG=0).

Run from src/mobile:
    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_hotfix4_back_keeps_jobs.py
"""

from __future__ import annotations

import asyncio
import importlib.util
import re
import sys
from pathlib import Path

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = MOBILE_DIR / "app"
EXT_DIR = MOBILE_DIR / "extensions" / "flet_glossarion_native"
for entry in (str(APP_DIR), str(EXT_DIR / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)


def _has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


pytestmark = pytest.mark.skipif(not (_has("flet") and _has("msgpack")), reason="flet / msgpack not installed")

# The real-app helpers (fake Flet session, _start / _wait / _stop) live in test_ui_foundations.py.
_UF_SPEC = importlib.util.spec_from_file_location("_glossarion_uf_helpers_hotfix4_back",
                                                  Path(__file__).with_name("test_ui_foundations.py"))
uf = importlib.util.module_from_spec(_UF_SPEC)
_UF_SPEC.loader.exec_module(uf)
storage = uf.storage
app_env = uf.app_env

FLUTTER_PKG = EXT_DIR / "src" / "flutter" / "flet_glossarion_native"
DART_SERVICE = FLUTTER_PKG / "lib" / "src" / "native_service.dart"
KOTLIN_PLUGIN = (FLUTTER_PKG / "android" / "src" / "main" / "kotlin" / "com" / "glossarion" / "flet_glossarion_native"
                 / "GlossarionNativePlugin.kt")
SWIFT_PLUGIN = (FLUTTER_PKG / "ios" / "flet_glossarion_native" / "Sources" / "flet_glossarion_native"
                / "GlossarionNativePlugin.swift")


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    home = tmp_path / "winhome"
    for sub in ("", "AppData/Roaming", "AppData/Local"):
        (home / sub).mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("APPDATA", str(home / "AppData" / "Roaming"))
    monkeypatch.setenv("LOCALAPPDATA", str(home / "AppData" / "Local"))
    monkeypatch.setenv("GLOSSARION_HTTP_LOG", "0")


def _device(conn, session, answers):
    """The fake client answers ``move_task_to_back`` like the device (``answers["move"]``; None: an
    extension without the method answers its default). Every other invoke keeps the fake's None."""
    from flet.messaging.protocol import MessageAction

    send = conn.send_message

    def send_message(message):
        body = message.body
        if message.action == MessageAction.INVOKE_METHOD and getattr(body, "name", None) == "move_task_to_back":
            conn.messages.append(message)
            conn.loop.call_soon_threadsafe(session.handle_invoke_method_results, body.control_id, body.call_id,
                                           answers["move"], None)
            return
        send(message)

    conn.send_message = send_message


def _invoked(conn, name):
    from flet.messaging.protocol import MessageAction

    return [m.body for m in conn.messages if m.action == MessageAction.INVOKE_METHOD and m.body.name == name]


def _patched(conn, view):
    from flet.messaging.protocol import MessageAction

    return [m for m in conn.messages if m.action == MessageAction.PATCH_CONTROL and getattr(m.body, "id", None) == view._i]


def _confirms(conn):
    return [(body.args or {}).get("should_pop") for body in _invoked(conn, "confirm_pop")]


async def _back(conn, session, view):
    """Android back reaching a View that cannot pop: Flet sends ``confirm_pop`` to Python and waits for
    ``view.confirm_pop(should_pop)``; returns that answer."""
    conn.messages.clear()
    await session.dispatch_event(view._i, "confirm_pop", None)
    assert await uf._wait(lambda: len(_confirms(conn)) == 1), "the app never answered confirm_pop"
    return _confirms(conn)[0]


def _jobs(app, running=0, queued=0):
    from glossarion_mobile.state.app_state import JobsBadge

    app.state.jobs_badge.set(JobsBadge(running=running, queued=queued))


def test_phone_back_on_the_chat_while_a_job_runs_moves_the_app_to_the_background(app_env):
    async def scenario():
        _m, conn, session, page, app = await uf._start("android")
        answers = {"move": True}
        _device(conn, session, answers)
        try:
            shell, root = app.shell, page.views[0]
            assert shell.move_to_background == app.native.move_task_to_back  # Android only (see the iOS test)
            assert root is shell.root_view and root.can_pop is True  # no job: the system default (leave)

            conn.messages.clear()
            _jobs(app, running=1)
            assert root.can_pop is False and callable(root.on_confirm_pop)
            assert _patched(conn, root)  # the client was told
            conn.messages.clear()
            _jobs(app, running=1, queued=2)  # same answer: the root View is not sent again
            assert not _patched(conn, root)

            assert await _back(conn, session, root) is False  # the pop is refused ...
            assert len(_invoked(conn, "move_task_to_back")) == 1  # ... and the app went to the background
            assert page.views[0] is root and not shell.stack  # the chat and this session stay

            answers["move"] = None  # an extension without the method: back leaves the app as before
            assert await _back(conn, session, root) is True

            answers["move"] = True
            _jobs(app, queued=1)  # a queued job keeps it too
            assert root.can_pop is False and await _back(conn, session, root) is False

            await shell.open_drawer()  # the menu button: back closes the drawer first
            assert shell.drawer_open
            assert await _back(conn, session, root) is False
            assert _invoked(conn, "close_drawer") and not _invoked(conn, "move_task_to_back")
            assert not shell.drawer_open
            await shell.open_drawer()
            await session.dispatch_event(shell.nav_drawer._i, "dismiss", None)  # closed by a swipe
            assert await uf._wait(lambda: not shell.drawer_open)

            _jobs(app)  # the queue drained: back leaves the app again
            assert root.can_pop is True
            assert await _back(conn, session, root) is True and not _invoked(conn, "move_task_to_back")
        finally:
            await uf._stop(app)

    asyncio.run(scenario())


def test_phone_a_pushed_screen_still_pops_while_a_job_runs(app_env):
    async def scenario():
        _m, conn, session, page, app = await uf._start("android")
        _device(conn, session, {"move": True})
        try:
            _jobs(app, running=1)
            app.navigate_to("jobs")
            assert await uf._wait(lambda: len(page.views) == 2)
            top = page.views[-1]
            assert top.can_pop is not False  # the screen's own View: back pops it (view_pop), not the app
            assert page.views[0].can_pop is False
            assert not _invoked(conn, "move_task_to_back")
        finally:
            await uf._stop(app)

    asyncio.run(scenario())


def test_tablet_back_pops_the_main_area_first_then_moves_the_app_to_the_background(app_env):
    from glossarion_mobile.ui.screens.keys import KeysScreen

    async def scenario():
        _m, conn, session, page, app = await uf._start("android", width=1000)
        _device(conn, session, {"move": True})
        try:
            shell, root = app.shell, page.views[0]
            assert shell.tablet and root.can_pop is True
            _jobs(app, running=1)
            assert root.can_pop is False
            await session.dispatch_event(app.drawer.keys_button._i, "click", None)
            assert await uf._wait(lambda: isinstance(shell.top_screen, KeysScreen))
            assert await _back(conn, session, root) is False  # the main area pops, as before
            assert shell.stack == [] and not _invoked(conn, "move_task_to_back")
            assert await _back(conn, session, root) is False  # the chat: to the background
            assert len(_invoked(conn, "move_task_to_back")) == 1
            _jobs(app)
            assert root.can_pop is True
        finally:
            await uf._stop(app)

    asyncio.run(scenario())


@pytest.mark.parametrize("platform", ["ios", "windows"])
def test_only_android_holds_back_for_jobs(app_env, platform):
    async def scenario():
        _m, conn, session, page, app = await uf._start(platform)
        try:
            assert app.shell.move_to_background is None
            _jobs(app, running=1)
            assert page.views[0].can_pop is True
        finally:
            await uf._stop(app)

    asyncio.run(scenario())


# ============================================================================ the extension contract
def test_native_bridge_and_stub():
    import types

    from glossarion_mobile.services.native import NativeBridge, NativeStub

    class Native:
        def __init__(self, answer):
            self.answer = answer

        async def move_task_to_back(self):
            return self.answer

    assert asyncio.run(NativeBridge(native=NativeStub("host")).move_task_to_back()) is False
    assert asyncio.run(NativeBridge(native=Native(True)).move_task_to_back()) is True
    assert asyncio.run(NativeBridge(native=Native("yes")).move_task_to_back()) is False  # never a truthy non-bool
    assert asyncio.run(NativeBridge(native=types.SimpleNamespace()).move_task_to_back()) is False  # without it


def test_extension_forwards_on_android_only():
    import flet as ft
    from flet_glossarion_native import GlossarionNative

    def attached(platform, answer):
        native = GlossarionNative()
        page = type("Page", (), {"platform": platform, "web": False})()
        native._page_or_none = lambda: page
        calls = []

        async def invoke(method_name, arguments=None, timeout=None):
            calls.append((method_name, arguments))
            return answer

        native._invoke_method = invoke
        return native, calls

    native, calls = attached(ft.PagePlatform.ANDROID, True)
    assert asyncio.run(native.move_task_to_back()) is True and calls == [("move_task_to_back", None)]
    native, calls = attached(ft.PagePlatform.ANDROID, "yes")
    assert asyncio.run(native.move_task_to_back()) is False
    native, calls = attached(ft.PagePlatform.WINDOWS, True)
    assert asyncio.run(native.move_task_to_back()) is False and calls == []  # desktop: never forwarded
    assert asyncio.run(GlossarionNative().move_task_to_back()) is False  # not attached


def _code(path):
    text = path.read_bytes().decode("utf-8").replace("\r\n", "\n")
    return re.sub(r"/\*.*?\*/|//[^\n]*", "", text, flags=re.S)


def test_dart_kotlin_and_swift_sides():
    dart = _code(DART_SERVICE)
    invoke = dart[dart.index("Future<dynamic> _invokeMethod"):dart.index("dynamic _defaultResult")]
    case = invoke[invoke.index("case 'move_task_to_back':"):]
    case = case[:case.index("case '", len("case 'move_task_to_back':"))]
    assert "_isAndroid" in case and "_platform('move_task_to_back', args)) == true" in case and ": false;" in case
    defaults = dart[dart.index("dynamic _defaultResult"):]
    assert re.search(r"case 'move_task_to_back':\s*return false;", defaults)

    kotlin = _code(KOTLIN_PLUGIN)
    branch = kotlin[kotlin.index('"move_task_to_back" -> {'):]
    branch = branch[:branch.index("else -> result.notImplemented()")]
    assert "activityBinding?.activity" in branch and "moveTaskToBack(true)" in branch
    assert "result.success(current != null && current.moveTaskToBack(true))" in branch
    assert "finish()" not in branch

    # iOS has no system back: Dart never asks Swift, so Swift has no such method.
    assert "move_task_to_back" not in SWIFT_PLUGIN.read_text(encoding="utf-8")


def test_app_wires_the_shell_on_android_only():
    source = (APP_DIR / "glossarion_mobile" / "app.py").read_text(encoding="utf-8")
    assert re.search(r'if self\.is_mobile and getattr\(getattr\(page, "platform", None\), "value", None\) == "android":'
                     r"\s+#[^\n]*\n\s+#[^\n]*\n\s+self\.shell\.move_to_background = self\.native\.move_task_to_back",
                     source)
