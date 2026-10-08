"""Fixtures of the ``flet test`` UI layer (U9; run by ``ci/android_ui_tests.sh`` on the emulator).

``flet test android --device-id <serial>`` provisions a debug test host with the app embedded and
runs pytest here; Flet's pytest plugin gives each test the ``flet_app`` fixture, whose ``tester``
drives the on-device Flutter ``WidgetTester``. Before any test starts, ``pytest_configure`` sets
two Flutter test settings in that host's driver (``driver_patch``: real ``adb`` swipes reach the
app; a tap that would miss its target fails instead of tapping empty space) and uninstalls a copy
of the app an aborted earlier run left behind, so every test starts as a fresh install (``flutter
test`` installs the app per test and uninstalls it afterwards). These fixtures add the device side:

* ``device``: ``android_device.Adb`` for the test's emulator (skips the test elsewhere);
* ``ui``: a ``UiDriver`` on ``flet_app.tester`` with the system Back key, list scrolling (an
  ``adb`` swipe) and the DocumentsUI picker; screenshots, uiautomator dumps and the step log go
  to ``GLOSSARION_UI_ARTIFACTS``;
* ``fake_server``: the app's own fake OpenAI server (``diagnostics/fake_llm_server.py``) run here
  on 127.0.0.1 and ``adb reverse``-d, so the app reaches it at the same URL;
* ``device_files``: the desktop-style config.json for that server and the 12-chapter self-test
  EPUB, pushed to the device's Download folder for the system picker.

These tests need a device: run them with ``python tools/build.py ui-tests --device-id …`` (or the
CI job); plain ``pytest`` skips them. ``tests_host/test_ui_flows.py`` runs the same flows on the
host without a device.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
MOBILE_DIR = TESTS_DIR.parent
APP_DIR = MOBILE_DIR / "app"
for _path in (str(TESTS_DIR), str(APP_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

ARTIFACTS = Path(os.environ.get("GLOSSARION_UI_ARTIFACTS") or (MOBILE_DIR / "build" / "ui-tests"))


def _device_run() -> bool:
    """A `flet test android` run (Flet sets these for the pytest it starts)."""
    return (os.environ.get("FLET_TEST_PLATFORM") == "android" and bool(os.environ.get("FLET_TEST_FLUTTER_APP_DIR")))


def _package() -> str:
    return os.environ.get("GLOSSARION_PACKAGE", "com.glossarion.app")


def pytest_configure(config):
    """Device runs only: patch the provisioned driver before any ``flutter test`` starts, then make
    the first test a fresh install (``flows.dismiss_welcome(first_run=True)`` relies on it)."""
    if not _device_run():
        return
    from driver_patch import DriverPatchError, patch_device_driver

    try:
        patched = patch_device_driver(os.environ["FLET_TEST_FLUTTER_APP_DIR"])
    except DriverPatchError as exc:
        raise pytest.UsageError(str(exc)) from None
    print(f"[ui] flet test driver: {'patched' if patched else 'already patched'} "
          "(device pointer events propagate, missed taps are fatal)")
    try:
        from android_device import Adb, adb_available

        if adb_available():
            Adb().run("uninstall", _package(), check=False, timeout=120)
    except Exception as exc:  # best effort: CI's emulator never has the app yet
        print(f"[ui] pre-run uninstall of {_package()} failed: {exc}")


def pytest_collection_modifyitems(config, items):
    if _device_run():
        return
    skip = pytest.mark.skip(reason="device UI tests: run them with `flet test android` (tools/build.py ui-tests)")
    for item in items:
        # only this directory's tests (a run that also collects tests_host/ keeps those)
        if TESTS_DIR in Path(str(getattr(item, "path", None) or item.fspath)).resolve().parents:
            item.add_marker(skip)


@pytest.fixture
def device(request):
    from android_device import Adb, adb_available

    if not adb_available():
        pytest.skip("adb is not on PATH")
    artifacts = ARTIFACTS / request.node.name
    return Adb(artifacts=artifacts)


@pytest.fixture
def ui(flet_app, device, request):
    from android_device import DocumentsPicker
    from ui_driver import UiDriver

    artifacts = ARTIFACTS / request.node.name
    artifacts.mkdir(parents=True, exist_ok=True)
    log_path = artifacts / "steps.log"

    def log(line: str) -> None:
        print(line)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    async def back() -> None:
        device.key("KEYCODE_BACK")

    # The app asks for the notification permission before its first job (a system dialog the
    # Flutter tester cannot answer): grant it up front, as android_smoke.sh installs with -g.
    device.shell(f"pm grant {_package()} android.permission.POST_NOTIFICATIONS", check=False)

    async def scroll() -> None:
        # a real swipe (the patched driver lets it through); off the event loop, so the
        # RemoteTester socket keeps being served while adb runs
        await asyncio.to_thread(device.swipe_up)

    driver = UiDriver(flet_app.tester, artifacts=artifacts, picker=DocumentsPicker(device), back=back,
                      scroll=scroll, log=log)
    yield driver
    try:
        device.screenshot(artifacts / "last_screen.png")
    except Exception:
        pass


@pytest.fixture
def fake_server(device):
    from glossarion_mobile.diagnostics.fake_llm_server import FakeLLMServer

    server = FakeLLMServer().start()
    device.reverse(server.port)
    try:
        yield server
    finally:
        device.remove_reverse(server.port)
        server.stop()


@pytest.fixture
def device_files(device, fake_server, tmp_path):
    import flows
    from glossarion_mobile.diagnostics.fake_llm_server import FAKE_MODEL

    epub = APP_DIR / "assets" / "selftest" / "selftest_ko_12ch.epub"
    if not epub.is_file():
        pytest.fail("app/assets/selftest/selftest_ko_12ch.epub is missing (tools/prepare_assets.py)")
    local_epub = tmp_path / flows.EPUB_NAME
    shutil.copyfile(epub, local_epub)
    config = flows.write_ui_config(tmp_path / flows.CONFIG_NAME, fake_server.url, FAKE_MODEL)
    names = (flows.CONFIG_NAME, flows.EPUB_NAME)
    device.push_download(config, flows.CONFIG_NAME)
    device.push_download(local_epub, flows.EPUB_NAME)
    try:
        yield names
    finally:
        for name in names:
            device.remove_download(name)
