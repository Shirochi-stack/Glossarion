"""Host runs of the device UI flows (``src/mobile/tests/flows.py``; U9 ``flet test`` layer).

The flows ``flet test android`` runs on the emulator (``src/mobile/tests/test_ui_*.py``) run here
against the real app on a fake Flet session with ``tests/host_tester.PyTester``: every key,
tooltip and label they use must exist, and the screens they walk through must behave. The
Android file picker is replaced by ``HostPicker`` (the device run drives DocumentsUI through adb).
Like the CI emulator (320x640 mdpi), the flows run at 320 dp, and on every pump the tree is
checked for a Row whose Text asks for an ellipsis it can never get (no flex: on a device the row
overflows, which fails the Flutter test and clips the text for users).

The chat flow is the full one: Settings › Import from desktop (a config.json pointing at the
fake OpenAI server), a new chat, ＋ › Files with the 12-chapter self-test EPUB, Send, the job card
reaches Done, the book moves into the Library by itself ("Added to the Library", no Migrate tap),
Library › the book › Chapters (12 completed chapters).

The device side of the first Android UI test run (Build Mobile run 37800059580, both tests
failed) is pinned on models of the device: the Welcome race behind "not found: key='dest-library'",
the dropped swipes behind "not found: key='hub-settings.import'", the driver patch that lets the
swipes through (and makes a missed tap an error the driver retries), int-millisecond pumps over
Flet's RemoteTester, and the chat header subtitle that overflowed the app bar on 320-470 dp
phones ("A RenderFlex overflowed by 148 pixels").

Run from src/mobile:
    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_ui_flows.py
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = MOBILE_DIR / "app"
TESTS_DIR = MOBILE_DIR / "tests"
for path in (APP_DIR, TESTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

#: the CI emulator's width (API 35 default AVD: 320x640 mdpi), the strictest phone layout
DEVICE_WIDTH = 320


def _has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


needs_flet = pytest.mark.skipif(not (_has("flet") and _has("msgpack")), reason="flet/msgpack not installed")

_TB_SPEC = importlib.util.spec_from_file_location("_glossarion_tb_helpers_u9ui", Path(__file__).with_name("test_bootstrap.py"))
_TB = importlib.util.module_from_spec(_TB_SPEC)
_TB_SPEC.loader.exec_module(_TB)
storage = _TB.storage
app_env = _TB.app_env


def _foundations():
    spec = importlib.util.spec_from_file_location("_glossarion_tf_helpers_u9ui",
                                                  Path(__file__).with_name("test_ui_foundations.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---- the unflexed-ellipsis check ------------------------------------------------------------


def unflexed_ellipsis_rows(page) -> dict:
    """``{path: text}`` of every Row (not wrapping, not scrolling) with a direct Text child that
    asks for an ellipsis (``max_lines`` or ``overflow=ELLIPSIS``) but has neither ``expand`` nor a
    fixed ``width``. Flutter lays a non-flex Row child out at its natural width, so such a text
    never ellipsizes and a long value overflows the row. Walks every view, the overlay and the
    dialogs, hidden controls included (they overflow once shown)."""
    import flet as ft
    from host_tester import _children

    found: dict = {}
    roots = list(page.views or []) + list(getattr(page, "overlay", None) or [])
    roots += list(getattr(getattr(page, "_dialogs", None), "controls", None) or [])
    stack = [(root, "") for root in roots]
    seen: set = set()
    while stack:
        control, path = stack.pop()
        if id(control) in seen:
            continue
        seen.add(id(control))
        key = getattr(getattr(control, "key", None), "value", getattr(control, "key", None))
        here = f"{path}/{type(control).__name__}" + (f"[{key}]" if isinstance(key, str) else "")
        if isinstance(control, ft.Row) and not control.wrap and control.scroll is None:
            for child in control.controls or []:
                if (isinstance(child, ft.Text) and (child.max_lines or child.overflow == ft.TextOverflow.ELLIPSIS)
                        and not child.expand and child.width is None):
                    text = child.value if isinstance(child.value, str) and child.value else "".join(
                        getattr(span, "text", "") or "" for span in (child.spans or []))
                    found.setdefault(here[-200:], text[:80])
        for child in _children(control):
            stack.append((child, here[-300:]))
    return found


def _scanning_tester():
    from host_tester import PyTester

    class ScanningTester(PyTester):
        """``PyTester`` that runs ``unflexed_ellipsis_rows`` on every pump (every screen a flow waits on)."""

        def __init__(self, session, page) -> None:
            super().__init__(session, page)
            self.unflexed: dict = {}

        async def pump(self, duration=None) -> None:
            self.unflexed.update(unflexed_ellipsis_rows(self.page))
            await super().pump(duration)

    return ScanningTester


async def _host_driver(tf, files: dict, *, first_run: bool = False, width: int = 412, tester_cls=None):
    from host_tester import HostPicker, PyTester
    from ui_driver import UiDriver

    _m, conn, session, page, app = await tf._start("android", width=width, first_run=first_run)
    picker = HostPicker(files)
    bridge = getattr(app, "files", None)
    if bridge is not None:
        bridge._get_picker = lambda: picker  # what the app's FilePicker service would answer

    async def back():
        views = list(page.views or [])
        if len(views) > 1:
            await session.dispatch_event(page._i, "view_pop", {"route": views[-1].route})

    tester = (tester_cls or PyTester)(session, page)
    driver = UiDriver(tester, picker=picker, back=back, poll_ms=100, log=lambda *_a: None)
    return app, tester, driver


@needs_flet
def test_dump_home_tree_when_asked(app_env, capsys):
    """Debug aid: GLOSSARION_UI_DUMP=1 prints the home screen's keys / tooltips / texts."""
    if not os.environ.get("GLOSSARION_UI_DUMP"):
        pytest.skip("set GLOSSARION_UI_DUMP=1 to print the tree")
    tf = _foundations()

    async def scenario():
        app, tester, driver = await _host_driver(tf, {})
        try:
            await asyncio.sleep(0.5)
            with capsys.disabled():
                for row in tester.dump():
                    print(row)
        finally:
            app.jobs.close()
            await tf._stop(app)

    asyncio.run(scenario())


@needs_flet
@pytest.mark.parametrize("first_run", [False, True], ids=["returning-user", "first-run"])
def test_smoke_navigation_and_selftest_flows(app_env, first_run):
    """The device smoke test's flow at 320 dp: as a returning user, and as the device tests run it
    (a fresh install: ``dismiss_welcome(first_run=True)`` taps Skip)."""
    import flows

    tf = _foundations()

    async def scenario():
        app, tester, driver = await _host_driver(tf, {}, first_run=first_run, width=DEVICE_WIDTH,
                                                 tester_cls=_scanning_tester())
        try:
            assert await flows.dismiss_welcome(driver, timeout=30, first_run=first_run) is first_run
            await flows.smoke_navigation(driver)
            assert await flows.run_selftest(driver, timeout=30) == "PASS"
            assert tester.unflexed == {}, f"Row texts that overflow instead of ellipsizing: {tester.unflexed}"
        finally:
            app.jobs.close()
            await tf._stop(app)

    asyncio.run(scenario())


@needs_flet
@pytest.mark.skipif(not (_has("ebooklib") and _has("openai") and _has("tiktoken")), reason="backend packages missing")
def test_chat_epub_migrate_library_flow(app_env, tmp_path, monkeypatch):
    import shutil

    import flows
    from glossarion_mobile.diagnostics.fake_llm_server import FAKE_MODEL, FakeLLMServer

    epub_src = APP_DIR / "assets" / "selftest" / "selftest_ko_12ch.epub"
    if not epub_src.is_file():
        pytest.skip("app/assets/selftest is generated by tools/prepare_assets.py")
    monkeypatch.setenv("GLOSSARION_HTTP_LOG", "0")
    tf = _foundations()
    picks = tmp_path / "picks"
    picks.mkdir()
    epub = picks / flows.EPUB_NAME
    shutil.copyfile(epub_src, epub)

    async def scenario(server):
        config = flows.write_ui_config(picks / flows.CONFIG_NAME, server.url, FAKE_MODEL)
        app, tester, driver = await _host_driver(tf, {flows.CONFIG_NAME: config, flows.EPUB_NAME: epub},
                                                 width=DEVICE_WIDTH, tester_cls=_scanning_tester())
        try:
            await flows.wait_home(driver)
            await flows.import_desktop_config(driver)
            assert app.config_store.get("openai_base_url") == server.url
            await flows.chat_translate_and_migrate(driver, timeout=300)
            # the owner's device report #4: no Migrate step - the workspace left the chat's Attachments
            # for the output folder by itself
            chats = app.chat_feature.chats
            cid = app.chat_view.cid
            assert chats.attachment_folders(cid) == []
            stem = Path(flows.EPUB_NAME).stem
            assert any(Path(m[4]).name == stem for m in chats.messages(cid) if m[0] == "assistant" and len(m) > 4
                       and m[4] and "Attachments" not in Path(m[4]).parts)
            await flows.library_book_chapters(driver)
            assert tester.unflexed == {}, f"Row texts that overflow instead of ellipsizing: {tester.unflexed}"
            if os.environ.get("GLOSSARION_UI_DUMP"):
                for row in tester.dump(600):
                    print(row)
        except Exception:
            for row in tester.dump(600):
                print(row)
            raise
        finally:
            app.jobs.close()
            await tf._stop(app)

    with FakeLLMServer() as server:
        asyncio.run(scenario(server))
        assert any(r.kind == "translate" for r in server.requests) or server.requests


def _ui_xml(*nodes) -> str:
    rows = "".join(
        f'<node text="{text}" content-desc="{desc}" package="{pkg}" bounds="[{x},{y}][{x + 100},{y + 50}]" />'
        for text, desc, pkg, x, y in nodes)
    return f'<?xml version="1.0"?><hierarchy rotation="0"><node package="{nodes[0][2] if nodes else ""}">{rows}</node></hierarchy>'


def test_documents_picker_finds_the_file_through_the_roots():
    """The DocumentsUI driver on canned uiautomator dumps: the app first (picker not up yet), then
    Recent without the file, the roots drawer with Downloads, then the Downloads listing."""
    import xml.etree.ElementTree as ET

    from android_device import DocumentsPicker

    docs = "com.google.android.documentsui"
    screens = [
        _ui_xml(("", "", "com.glossarion.app", 0, 0)),
        _ui_xml(("Recent", "", docs, 0, 0), ("", "Show roots", docs, 10, 10)),
        _ui_xml(("Downloads", "", docs, 200, 300), ("sdk_gphone64_x86_64", "", docs, 200, 400)),
        _ui_xml(("Downloads", "", docs, 0, 0), ("glossarion-ui-selftest.epub", "", docs, 300, 600)),
    ]

    class FakeAdb:
        def __init__(self):
            self.taps = []
            self.dumps = 0

        def dump_ui(self, label="dump"):
            self.dumps += 1
            return ET.fromstring(screens[min(self.dumps - 1, len(screens) - 1)])

        def tap_xy(self, x, y):
            self.taps.append((x, y))

    adb = FakeAdb()
    picker = DocumentsPicker(adb, timeout=60)

    async def no_sleep(_s):
        return None

    import android_device

    real_sleep = android_device.asyncio.sleep
    android_device.asyncio.sleep = no_sleep
    try:
        asyncio.run(picker.choose("glossarion-ui-selftest.epub"))
    finally:
        android_device.asyncio.sleep = real_sleep
    # Show roots, then Downloads in the drawer, then the file
    assert adb.taps == [(60, 35), (250, 325), (350, 625)]

    adb = FakeAdb()
    screens[:] = [_ui_xml(("Recent", "", docs, 0, 0))]
    picker = DocumentsPicker(adb, timeout=0.5)
    android_device.asyncio.sleep = no_sleep
    try:
        with pytest.raises(AssertionError, match="never listed"):
            asyncio.run(picker.choose("missing.epub"))
    finally:
        android_device.asyncio.sleep = real_sleep


@needs_flet
def test_first_run_welcome_is_skipped(app_env):
    """A fresh install (the device tests' first launch): the Welcome flow, then Skip, then the home."""
    import flows

    tf = _foundations()

    async def scenario():
        app, tester, driver = await _host_driver(tf, {}, first_run=True)
        try:
            assert await flows.dismiss_welcome(driver, timeout=30) is True
            await flows.wait_home(driver, timeout=30)
        finally:
            app.jobs.close()
            await tf._stop(app)

    asyncio.run(scenario())


# ---- models of the device (Build Mobile run 37800059580) --------------------------------------


class _Clock:
    """``ui_driver.time`` for the device models: pumps and swipes advance it, nothing sleeps."""

    def __init__(self) -> None:
        self.t = 1000.0

    def monotonic(self) -> float:
        return self.t


class _Finder:
    def __init__(self, fid: int, count: int, index: int = 0) -> None:
        self.id, self.count, self.index = fid, count, index

    @property
    def first(self) -> "_Finder":
        return _Finder(self.id, 1, 0)

    def at(self, index: int) -> "_Finder":
        return _Finder(self.id, 1, index)


class _ModelTester:
    """The RemoteTester calls the flows use, on a scripted device."""

    def __init__(self, clock: _Clock) -> None:
        self.clock = clock
        self.finders: dict = {}
        self.pumped: list = []

    def visible(self) -> set:
        raise NotImplementedError

    def _find(self, kind: str, value) -> _Finder:
        fid = len(self.finders) + 1
        self.finders[fid] = (kind, value)
        return _Finder(fid, 1 if (kind, value) in self.visible() else 0)

    async def find_by_key(self, key):
        return self._find("key", key)

    async def find_by_text(self, text):
        return self._find("text", text)

    async def find_by_text_containing(self, pattern):
        return self._find("contains", pattern)

    async def find_by_tooltip(self, value):
        return self._find("tooltip", value)

    async def pump(self, duration=None) -> None:
        assert isinstance(duration, int), f"pump({duration!r}): Flet's RemoteTester needs int milliseconds"
        self.pumped.append(duration)
        self.clock.t += duration / 1000.0

    async def take_screenshot(self, name):
        return b""


class _StartupDevice(_ModelTester):
    """A fresh install under ``flet test``: the chat home is on screen when the tester connects and
    the app pushes the Welcome one frame later (after its feature installs). A covered route is
    offstage, so Flutter's finders no longer see the home, nor a drawer opened on it."""

    def __init__(self, clock: _Clock, *, welcome_after_pumps: int = 1) -> None:
        super().__init__(clock)
        self.welcome_after = welcome_after_pumps
        self.welcome_skipped = False
        self.drawer_open_on_root = False

    @property
    def welcome_up(self) -> bool:
        return len(self.pumped) >= self.welcome_after and not self.welcome_skipped

    def visible(self) -> set:
        import flows

        if self.welcome_up:
            return {("text", "Skip")}
        out = {("tooltip", flows.ATTACH_TOOLTIP), ("tooltip", "Open navigation"), ("key", "mode-chip")}
        if self.drawer_open_on_root:
            out |= {("key", "dest-library"), ("tooltip", "Settings")}
        return out

    async def tap(self, finder) -> None:
        kind, value = self.finders[finder.id]
        if (kind, value) == ("text", "Skip"):
            self.welcome_skipped = True
        elif (kind, value) == ("tooltip", "Open navigation"):
            self.drawer_open_on_root = True  # View.show_drawer on the root view, as in the CI log


def test_ci_dest_library_timeout_was_the_welcome_race(monkeypatch):
    """test_ui_smoke on the emulator: "UiTimeout: not found after 15s: key='dest-library'". The
    device tests settled for the chat home, which the Welcome covered a frame later, so the drawer
    opened under it. With ``first_run=True`` they wait for the Welcome and tap Skip."""
    import flows
    import ui_driver

    clock = _Clock()
    monkeypatch.setattr(ui_driver, "time", clock)

    async def run(first_run: bool):
        device = _StartupDevice(clock)
        driver = ui_driver.UiDriver(device, log=lambda *_a: None)
        skipped = await flows.dismiss_welcome(driver, timeout=180, first_run=first_run)
        await flows.wait_home(driver)
        await flows.open_drawer(driver)
        return skipped, device

    with pytest.raises(ui_driver.UiTimeout, match=r"not found after 15s: key='dest-library'"):
        asyncio.run(run(False))  # what the device tests did (the either/or path)
    skipped, device = asyncio.run(run(True))
    assert skipped and device.welcome_skipped and device.drawer_open_on_root


class _MissedHitTest(RuntimeError):
    """What Flet's RemoteTester raises for Flutter's fatal hit-test check (the driver patch)."""

    def __init__(self) -> None:
        super().__init__("Finder specifies a widget that would not receive pointer events.\n"
                         "A call to tap() with finder ... derived an Offset (160.0, 702.0) that would not hit test "
                         "on the specified widget.")


class _SettingsListDevice(_ModelTester):
    """The Settings home on a device: a lower group's rows are built once its card scrolls into the
    viewport (``found_after`` swipes), but a row's centre stays below the fold for a few more swipes
    (``hittable_after``), so the fatal hit test rejects a tap. ``propagate=False``: the test binding
    drops the adb swipes (Flet's driver before ``driver_patch``)."""

    def __init__(self, clock: _Clock, *, found_after: int = 3, hittable_after: int = 5, propagate: bool = True,
                 transient: int = 0, error: Exception = None) -> None:
        super().__init__(clock)
        self.found_after, self.hittable_after = found_after, hittable_after
        self.propagate = propagate
        self.transient = transient  # taps that first meet a rebuilt tree ("could not find any matching widgets")
        self.error = error
        self.swipes = 0
        self.scrolled = 0
        self.taps = 0
        self.tapped = False

    def swipe(self) -> None:
        self.swipes += 1
        self.clock.t += 0.9  # adb input swipe (700 ms) plus adb's own start-up
        if self.propagate:
            self.scrolled += 1

    def visible(self) -> set:
        return {("key", "hub-settings.import")} if self.scrolled >= self.found_after else set()

    async def tap(self, finder) -> None:
        self.taps += 1
        if self.error is not None:
            raise self.error
        if self.transient:
            self.transient -= 1
            raise RuntimeError('The finder "Found 1 widget with key [<\'hub-settings.import\'>]" (used in a call to '
                               '"tap()") could not find any matching widgets.')
        if self.scrolled < self.hittable_after:
            raise _MissedHitTest()
        self.tapped = True


def _list_driver(monkeypatch, **kwargs):
    import ui_driver

    clock = _Clock()
    monkeypatch.setattr(ui_driver, "time", clock)
    device = _SettingsListDevice(clock, **kwargs)

    async def scroll():
        device.swipe()

    return device, ui_driver.UiDriver(device, scroll=scroll, log=lambda *_a: None)


def test_ci_hub_settings_import_timeout_was_dropped_swipes(monkeypatch):
    """test_ui_chat_epub on the emulator: "UiTimeout: not found after 30s: key='hub-settings.import'"
    after 64 swipes that never moved the list (the integration-test binding dropped them)."""
    import ui_driver

    device, driver = _list_driver(monkeypatch, propagate=False)
    with pytest.raises(ui_driver.UiTimeout, match=r"not found after 30s: key='hub-settings.import'"):
        asyncio.run(driver.tap(key="hub-settings.import", timeout=30, scroll=True))
    assert device.swipes > 10 and device.scrolled == 0 and device.taps == 0
    # once the swipes reach the app (driver_patch) the same tap lands
    device, driver = _list_driver(monkeypatch)
    asyncio.run(driver.tap(key="hub-settings.import", timeout=30, scroll=True))
    assert device.tapped


def test_tap_scrolls_and_retries_a_row_found_below_the_fold(monkeypatch):
    """The lazy unit is a whole card: the row is found while its centre is still below the fold. The
    fatal hit test rejects the tap; the driver swipes one step and retries until the tap lands."""
    device, driver = _list_driver(monkeypatch, found_after=3, hittable_after=5)
    asyncio.run(driver.tap(key="hub-settings.import", timeout=30, scroll=True))
    assert device.tapped and device.scrolled == device.hittable_after
    assert device.taps == 1 + (device.hittable_after - device.found_after)
    missed = [s for s in driver.steps if s.startswith("tap missed")]
    assert len(missed) == device.taps - 1 and "would not receive pointer events" in missed[0]
    # every poll of a scrolling wait swipes (not every second one)
    assert [s for s in driver.steps if s.startswith("scroll for")] == ["scroll for key='hub-settings.import'"] * 3
    # a tap that meets a rebuilt tree ("could not find any matching widgets") is retried as well
    device, driver = _list_driver(monkeypatch, found_after=0, hittable_after=0, transient=2)
    asyncio.run(driver.tap(key="hub-settings.import", timeout=30))
    assert device.tapped and device.taps == 3 and device.swipes == 0  # no scroll asked: pump and retry


def test_tap_gives_up_on_a_target_that_never_becomes_hittable(monkeypatch):
    import ui_driver

    device, driver = _list_driver(monkeypatch, found_after=0, hittable_after=10 ** 6)
    with pytest.raises(ui_driver.UiTimeout, match=r"tap kept missing for 20s: key='hub-settings.import': "
                                                   r"Finder specifies a widget that would not receive pointer events"):
        asyncio.run(driver.tap(key="hub-settings.import", timeout=20, scroll=True))
    assert not device.tapped and device.taps > 5
    # any other error is the flow's problem: raised at once, no retry
    device, driver = _list_driver(monkeypatch, found_after=0, error=ValueError("Finder with id 7 is not registered."))
    with pytest.raises(ValueError, match="not registered"):
        asyncio.run(driver.tap(key="hub-settings.import", timeout=20))
    assert device.taps == 1


async def _fake_flutter_side(port: int, log: list):
    """The on-device RemoteWidgetTester's end of the socket (4-byte length-prefixed JSON frames)."""
    reader, writer = await asyncio.open_connection("127.0.0.1", port)

    async def serve():
        try:
            while True:
                header = await reader.readexactly(4)
                message = json.loads(await reader.readexactly(int.from_bytes(header, "big")))
                log.append((message["method"], message.get("params")))
                data = json.dumps({"id": message["id"], "result": None}).encode()
                writer.write(len(data).to_bytes(4, "big") + data)
                await writer.drain()
        except (asyncio.IncompleteReadError, ConnectionError):
            pass

    return asyncio.create_task(serve()), writer


def _flet_remote_tester():
    """Flet's ``RemoteTester``. ``flet.testing``'s package init imports scikit-image (the
    ``flet[test]`` extra the device job installs); without it, load the module on its own."""
    try:
        from flet.testing.remote_tester import RemoteTester

        return RemoteTester
    except ImportError:
        pass
    import flet

    package = types.ModuleType("flet.testing")
    package.__path__ = [str(Path(flet.__file__).resolve().parent / "testing")]
    saved = {name: sys.modules.get(name) for name in ("flet.testing", "flet.testing.finder", "flet.testing.remote_tester")}
    sys.modules["flet.testing"] = package
    try:
        module = importlib.import_module("flet.testing.remote_tester")
        return module.RemoteTester
    finally:
        for name, value in saved.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


@needs_flet
def test_pump_sends_int_milliseconds_over_flets_remote_tester():
    """``UiDriver.pump`` used to send a ``timedelta``: json.dumps raised after the RemoteTester had
    registered the reply future, so every poll leaked a future (976 "Future exception was never
    retrieved" at teardown) and pumped with no delay. Int milliseconds reach the device."""
    from ui_driver import UiDriver

    RemoteTester = _flet_remote_tester()

    async def scenario():
        remote = RemoteTester()
        port = await remote.start()
        log: list = []
        task, writer = await _fake_flutter_side(port, log)
        try:
            await remote.wait_for_connection(5)
            driver = UiDriver(remote, poll_ms=250, log=lambda *_a: None)
            await driver.pump(400)
            await driver.pump()
            leaked = [f for f in remote._pending.values() if not f.done()]
        finally:
            writer.close()
            await remote.stop()
            task.cancel()
        return log, leaked

    log, leaked = asyncio.run(scenario())
    assert log == [("pump", {"duration": 400}), ("pump", {"duration": 250})]
    assert leaked == []


@needs_flet
def test_host_tester_pump_takes_flet_durations():
    import flet as ft
    from host_tester import PyTester

    slept = []

    async def scenario():
        tester = PyTester(None, None)
        real_sleep = asyncio.sleep

        async def fake_sleep(seconds):
            slept.append(round(seconds, 3))
            await real_sleep(0)

        asyncio.sleep = fake_sleep
        try:
            await tester.pump(400)
            await tester.pump(ft.Duration(milliseconds=250))
            await tester.pump()
        finally:
            asyncio.sleep = real_sleep

    asyncio.run(scenario())
    assert slept == [0.4, 0.25, 0.05]


# ---- the `flet test` driver patch -----------------------------------------------------------------

#: integration_test/app_test.dart as Flet 1.0.3's build template renders it for this app
FLET_103_DRIVER = """import 'package:flet_integration_test/flet_integration_test.dart';
import 'package:glossarion/main.dart' as app;

// Device-mode integration test entry point. The app under test runs on-device
// with embedded Python over dart_bridge; a RemoteWidgetTester drives it over a
// raw socket connected to the pytest RemoteTester server (FLET_TEST_SERVER_URL).
void main() => runFletDeviceTest(appMain: app.main);
"""


@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["lf", "crlf"])
def test_device_driver_patch(tmp_path, newline):
    import driver_patch

    driver = driver_patch.driver_path(tmp_path)
    driver.parent.mkdir()
    driver.write_bytes(FLET_103_DRIVER.replace("\n", newline).encode("utf-8"))
    assert driver_patch.patch_device_driver(tmp_path) is True
    data = driver.read_bytes()
    assert data.count(b"\r\n") in (0, data.count(b"\n")) and (b"\r\n" in data) == (newline == "\r\n")
    text = data.decode("utf-8").replace("\r\n", "\n")
    body = text[text.index("void main() {"):]
    # the flags are set right after runFletDeviceTest created the binding, before any test body runs
    assert body.splitlines()[:5] == [
        "void main() {",
        "  runFletDeviceTest(appMain: app.main);",
        "  LiveTestWidgetsFlutterBinding.instance.shouldPropagateDevicePointerEvents = true;",
        "  WidgetController.hitTestWarningShouldBeFatal = true;",
        "}",
    ]
    assert driver_patch.DRIVER_LINE not in text and text.count("void main()") == 1
    # flutter_test only (a declared dev dependency of a `flet test` host), imported before any code
    assert text.splitlines()[0] == ("import 'package:flutter_test/flutter_test.dart' "
                                    "show LiveTestWidgetsFlutterBinding, WidgetController;")
    assert "package:integration_test/" not in text and text.splitlines()[1:3] == FLET_103_DRIVER.splitlines()[:2]
    assert driver_patch.patch_device_driver(tmp_path) is False and driver.read_bytes() == data  # once only
    driver.write_text("void main() => somethingElse();\n", encoding="utf-8")
    with pytest.raises(driver_patch.DriverPatchError, match="Flet template changed"):
        driver_patch.patch_device_driver(tmp_path)
    with pytest.raises(driver_patch.DriverPatchError, match="was not generated"):
        driver_patch.patch_device_driver(tmp_path / "missing")


def _device_conftest():
    spec = importlib.util.spec_from_file_location("_glossarion_device_conftest", TESTS_DIR / "conftest.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_device_conftest_patches_the_driver_and_starts_fresh(tmp_path, monkeypatch):
    """``pytest_configure`` of src/mobile/tests: on a `flet test android` run only, the provisioned
    driver is patched before any ``flutter test`` starts and a leftover app is uninstalled (every
    device test is a first run: ``dismiss_welcome(first_run=True)``)."""
    import android_device
    import driver_patch

    conftest = _device_conftest()
    driver = driver_patch.driver_path(tmp_path)
    driver.parent.mkdir()
    driver.write_text(FLET_103_DRIVER, encoding="utf-8")
    calls = []

    class FakeAdb:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def run(self, *args, **kwargs):
            calls.append((args, kwargs))
            return ""

    monkeypatch.setattr(android_device, "Adb", FakeAdb)
    monkeypatch.setattr(android_device, "adb_available", lambda: "adb")
    monkeypatch.setenv("FLET_TEST_FLUTTER_APP_DIR", str(tmp_path))
    monkeypatch.delenv("FLET_TEST_PLATFORM", raising=False)
    monkeypatch.delenv("GLOSSARION_PACKAGE", raising=False)
    conftest.pytest_configure(None)  # plain pytest / a desktop `flet test`: nothing happens
    assert driver.read_text(encoding="utf-8") == FLET_103_DRIVER and calls == []

    monkeypatch.setenv("FLET_TEST_PLATFORM", "android")
    conftest.pytest_configure(None)
    assert driver_patch.MARK in driver.read_text(encoding="utf-8")
    assert calls == [(("uninstall", "com.glossarion.app"), {"check": False, "timeout": 120})]

    def broken(*_a, **_k):
        raise RuntimeError("adb: device offline")

    monkeypatch.setattr(FakeAdb, "run", broken)
    conftest.pytest_configure(None)  # the uninstall is best effort; the patch is already in
    driver.write_text("void main() => somethingElse();\n", encoding="utf-8")
    with pytest.raises(pytest.UsageError, match="Flet template changed"):
        conftest.pytest_configure(None)


# ---- the chat header on narrow phones -----------------------------------------------------------

#: Roboto label-small width per character on the device, from the CI log: at 320 dp the 44-character
#: subtitle "authgpt/gpt-6-luna · Universal · → English ▾" overflowed its 88 dp title slot by 148 px
LABEL_SMALL_PX_PER_CHAR = (88 + 148) / 44


def _natural_px(control) -> float:
    import flet as ft

    if isinstance(control, ft.Text):
        text = control.value if isinstance(control.value, str) and control.value else "".join(
            span.text or "" for span in (control.spans or []))
        assert control.theme_style == ft.TextThemeStyle.LABEL_SMALL, "the width model knows label-small text only"
        return len(text) * LABEL_SMALL_PX_PER_CHAR
    if isinstance(control, ft.Container):
        padding = control.padding
        return _natural_px(control.content) + (padding.left + padding.right if padding is not None else 0)
    raise AssertionError(f"no width model for {type(control).__name__}")


def _row_layout(row, max_width: float, *, rigid: tuple = ()) -> tuple:
    """Flutter's RenderFlex for a width-bounded Row as Flet builds it: a child with ``expand`` is
    ``Expanded``, with ``expand`` + ``expand_loose`` a loose ``Flexible``; other children (and
    ``rigid`` ones) get their natural width. Returns (overflow px, {id(child): (natural, width)})."""
    children = [c for c in row.controls if c.visible is not False]
    spacing = (10 if row.spacing is None else row.spacing) * max(0, len(children) - 1)
    rigid_ids = {id(c) for c in rigid}  # by identity: Flet controls compare equal field by field
    flexible = [c for c in children if c.expand and id(c) not in rigid_ids]
    flexible_ids = {id(c) for c in flexible}
    sizes = {id(c): (_natural_px(c), _natural_px(c)) for c in children if id(c) not in flexible_ids}
    free = max(0.0, max_width - spacing - sum(width for _, width in sizes.values()))
    total_flex = sum(int(c.expand) for c in flexible)
    for child in flexible:
        share = free * int(child.expand) / total_flex
        natural = _natural_px(child)
        sizes[id(child)] = (natural, min(natural, share) if child.expand_loose else share)
    laid_out = spacing + sum(width for _, width in sizes.values())
    return max(0.0, laid_out - max_width), sizes


def _app_bar_title_px(bar, screen_width: float) -> float:
    """NavigationToolbar: the title slot is the bar minus the leading slot (56 dp by default), the
    visible actions (48 dp hit targets here) and the title spacing on both sides (16 dp)."""
    import flet as ft

    leading = (bar.leading_width or 56) if bar.leading is not None and bar.leading.visible is not False else 0
    actions = [a for a in (bar.actions or []) if a.visible is not False]
    for action in actions:
        assert isinstance(action, (ft.IconButton, ft.PopupMenuButton)) and action.size_constraints.min_width == 48
    spacing = 16 if bar.title_spacing is None else bar.title_spacing
    return screen_width - leading - 48 * len(actions) - 2 * spacing


@needs_flet
def test_header_subtitle_ellipsizes_in_the_app_bar_on_narrow_phones(app_env):
    """Build Mobile run 37800059580: "A RenderFlex overflowed by 148 pixels on the right" in the chat
    header's subtitle row (BoxConstraints w<=88 at 320 dp). The subtitle kept its natural width in
    a tight Row, so on phones narrower than ~470 dp it overflowed the app bar's title slot instead
    of ellipsizing, clipping the text and hiding the "custom" chip. At 320, 360 and 411 dp it now
    ellipsizes inside the title slot and the chip stays whole."""
    import flet as ft
    from glossarion_mobile.state.app_state import ChatContext

    tf = _foundations()
    context = dict(model="authgpt/gpt-6-luna", profile="Universal", target_language="English")

    async def scenario():
        _m, conn, session, page, app = await tf._start("android", width=DEVICE_WIDTH)
        try:
            header = app.chat_view.header
            row = header.title_column.controls[1]
            assert isinstance(row, ft.Row) and [id(c) for c in row.controls] == [id(header.subtitle),
                                                                                 id(header.custom_badge)]
            assert header.subtitle.max_lines == 1 and header.subtitle.overflow == ft.TextOverflow.ELLIPSIS
            for width in (320, 360, 411):
                await session.dispatch_event(page._i, "resize", {"width": width, "height": 860})
                bar = page.views[0].appbar
                assert isinstance(bar, ft.AppBar) and bar is header.wrapper and bar.title is header.title_slot
                assert header.title_slot.content is header.title_column
                slot = _app_bar_title_px(bar, width)
                for custom in (False, True):
                    header.set_context(ChatContext(custom=custom, **context))
                    assert header.subtitle_text == "authgpt/gpt-6-luna · Universal · → English ▾"
                    assert header.custom_badge.visible is custom
                    overflow, sizes = _row_layout(row, slot)
                    assert overflow == 0, (width, custom, overflow)
                    natural, laid_out = sizes[id(header.subtitle)]
                    assert 0 < laid_out < natural, (width, custom, laid_out, natural)  # ellipsized, still shown
                    if custom:
                        badge_natural, badge = sizes[id(header.custom_badge)]
                        assert badge == badge_natural > 0  # the chip is whole, right after the text
                        assert laid_out + row.spacing + badge <= slot + 1e-6
                    # the old row (the subtitle at its natural width) overflowed by what the device reported
                    old_overflow, _ = _row_layout(row, slot, rigid=(header.subtitle,))
                    assert old_overflow == pytest.approx(natural + (row.spacing + badge if custom else 0) - slot)
                    if width == 320 and not custom:
                        assert slot == 88 and old_overflow == pytest.approx(148)
            assert header.subtitle.expand and header.subtitle.expand_loose  # a loose Flexible
            assert not header.custom_badge.expand  # the chip keeps its own width
            # a short subtitle keeps its own width (loose): nothing is stretched or cut
            header.set_context(ChatContext(model="gpt-x", profile="P", target_language="EN", custom=True))
            overflow, sizes = _row_layout(row, _app_bar_title_px(page.views[0].appbar, 411))
            natural, laid_out = sizes[id(header.subtitle)]
            assert overflow == 0 and laid_out == natural
        finally:
            app.jobs.close()
            await tf._stop(app)

    asyncio.run(scenario())
