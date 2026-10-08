"""adb helpers for the device UI tests (``flet test android``; host side of the test).

* ``Adb``: the emulator/device the tests run on (``ANDROID_SERIAL`` / ``FLET_TEST_DEVICE``),
  ``adb reverse`` (the app reaches the test's fake model server on 127.0.0.1), files pushed to
  ``/sdcard/Download`` (MediaStore scan requested), the system Back key, ``uiautomator`` dumps.
* ``DocumentsPicker``: chooses a file in the system picker (DocumentsUI), which is a native
  activity the Flutter tester cannot see: it reads ``uiautomator dump`` and taps with
  ``input tap`` — the file when it is listed, else the Downloads root (via "Show roots"), else
  the device storage root and its ``Download`` folder. Each attempt's dump is kept in the
  artifacts directory, so a failure shows exactly what the picker displayed.
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

__all__ = ["Adb", "DocumentsPicker", "adb_available"]

DOCUMENTS_UI = ("com.google.android.documentsui", "com.android.documentsui")
_BOUNDS = re.compile(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]")


def adb_available() -> Optional[str]:
    return os.environ.get("GLOSSARION_UI_ADB") or shutil.which("adb")


class Adb:
    def __init__(self, serial: Optional[str] = None, adb: Optional[str] = None,
                 artifacts: Optional[os.PathLike] = None) -> None:
        self.adb = adb or adb_available()
        if not self.adb:
            raise RuntimeError("adb is not on PATH")
        self.serial = serial or os.environ.get("ANDROID_SERIAL") or os.environ.get("FLET_TEST_DEVICE") or None
        self.artifacts = Path(artifacts) if artifacts else None
        self._dumps = 0

    def run(self, *args: str, check: bool = True, timeout: float = 60, binary: bool = False):
        cmd = [self.adb] + (["-s", self.serial] if self.serial else []) + list(args)
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout)
        if check and proc.returncode != 0:
            raise RuntimeError(f"{' '.join(cmd)} failed ({proc.returncode}): "
                               f"{proc.stderr.decode('utf-8', 'replace')[-400:]}")
        return proc.stdout if binary else proc.stdout.decode("utf-8", "replace")

    def shell(self, command: str, *, check: bool = True, timeout: float = 60) -> str:
        return self.run("shell", command, check=check, timeout=timeout)

    def reverse(self, port: int) -> None:
        """The device's 127.0.0.1:<port> reaches this machine's 127.0.0.1:<port>."""
        self.run("reverse", f"tcp:{port}", f"tcp:{port}")

    def remove_reverse(self, port: int) -> None:
        self.run("reverse", "--remove", f"tcp:{port}", check=False)

    def push_download(self, local: os.PathLike, name: str) -> str:
        remote = f"/sdcard/Download/{name}"
        self.run("push", str(local), remote, timeout=120)
        # The picker's Downloads root lists MediaStore rows: ask for a scan (both forms; best effort).
        self.shell(f"am broadcast -a android.intent.action.MEDIA_SCANNER_SCAN_FILE -d file://{remote}", check=False)
        self.shell("content call --uri content://media --method scan_volume --arg external_primary", check=False)
        return remote

    def remove_download(self, name: str) -> None:
        self.shell(f"rm -f /sdcard/Download/{name}", check=False)

    def key(self, code: str) -> None:
        self.shell(f"input keyevent {code}", check=False)

    def tap_xy(self, x: int, y: int) -> None:
        self.shell(f"input tap {x} {y}", check=False)

    def screen_size(self) -> tuple:
        if getattr(self, "_size", None) is None:
            text = self.shell("wm size", check=False)
            match = re.findall(r"(\d+)x(\d+)", text)
            self._size = tuple(int(v) for v in match[-1]) if match else (1080, 1920)
        return self._size

    def swipe_up(self) -> None:
        """Scroll the list under the finger down by about a third of the screen."""
        width, height = self.screen_size()
        x = width // 2
        self.shell(f"input swipe {x} {int(height * 0.72)} {x} {int(height * 0.38)} 400", check=False)

    def foreground_package(self) -> str:
        text = self.shell("dumpsys activity activities | grep -m 1 -E 'topResumedActivity|mResumedActivity'",
                          check=False)
        match = re.search(r"\s([a-zA-Z0-9_.]+)/", text)
        return match.group(1) if match else ""

    def dump_ui(self, label: str = "dump") -> Optional[ET.Element]:
        self.shell("uiautomator dump /sdcard/glossarion_ui_dump.xml", check=False, timeout=60)
        data = self.run("exec-out", "cat /sdcard/glossarion_ui_dump.xml", check=False, binary=True)
        if self.artifacts is not None and data:
            self._dumps += 1
            self.artifacts.mkdir(parents=True, exist_ok=True)
            (self.artifacts / f"uiautomator_{self._dumps:02d}_{label}.xml").write_bytes(data)
        try:
            return ET.fromstring(data.decode("utf-8", "replace")) if data else None
        except ET.ParseError:
            return None

    def screenshot(self, path: os.PathLike) -> None:
        data = self.run("exec-out", "screencap -p", check=False, binary=True)
        if data:
            Path(path).write_bytes(data)


def _center(node: ET.Element) -> Optional[tuple]:
    match = _BOUNDS.match(node.get("bounds", ""))
    if not match:
        return None
    x1, y1, x2, y2 = (int(v) for v in match.groups())
    return (x1 + x2) // 2, (y1 + y2) // 2


def _nodes(root: Optional[ET.Element]) -> list:
    return list(root.iter("node")) if root is not None else []


class DocumentsPicker:
    """Chooses files in DocumentsUI for ``UiDriver.pick_file``."""

    def __init__(self, adb: Adb, *, timeout: float = 120.0) -> None:
        self.adb = adb
        self.timeout = timeout

    async def arm(self, name: str) -> None:
        return None  # the files were pushed to /sdcard/Download by the fixture

    async def choose(self, name: str) -> None:
        """Tap ``name`` in DocumentsUI. Per round (one uiautomator dump): the file when it is
        listed; else the "Downloads" entry when one is on screen; else open the roots drawer and
        take Downloads, or the device's storage root (whose ``Download`` folder the next round
        opens)."""
        deadline = time.monotonic() + self.timeout
        attempt = 0
        stem = Path(name).stem
        while time.monotonic() < deadline:
            attempt += 1
            await asyncio.sleep(1.5)
            nodes = _nodes(await asyncio.to_thread(self.adb.dump_ui, f"picker_{attempt}"))
            if not {n.get("package", "") for n in nodes} & set(DOCUMENTS_UI):
                continue  # the picker is not on screen yet
            target = next((n for n in nodes if n.get("text") in (name, stem) and _center(n)), None)
            if target is not None:
                await asyncio.to_thread(self.adb.tap_xy, *_center(target))
                await asyncio.sleep(2.0)
                return
            folder = next((n for n in nodes if n.get("text") in ("Downloads", "Download") and _center(n)), None)
            if folder is not None and attempt % 4 != 0:  # every 4th round re-opens the roots instead
                await asyncio.to_thread(self.adb.tap_xy, *_center(folder))
                continue
            roots = next((n for n in nodes if n.get("content-desc") in ("Show roots", "Open navigation drawer")
                          and _center(n)), None)
            if roots is None:
                continue
            await asyncio.to_thread(self.adb.tap_xy, *_center(roots))
            await asyncio.sleep(1.0)
            drawer = _nodes(await asyncio.to_thread(self.adb.dump_ui, f"roots_{attempt}"))
            pick = next((n for n in drawer if n.get("text") == "Downloads" and _center(n)), None)
            if pick is None or attempt % 4 == 0:
                storage = re.compile(r"sdk_gphone|Android SDK|Internal storage|GB free")
                pick = next((n for n in drawer if _center(n) and storage.search(
                    (n.get("text") or "") + " " + (n.get("content-desc") or ""))), pick)
            if pick is not None:
                await asyncio.to_thread(self.adb.tap_xy, *_center(pick))
        raise AssertionError(f"the system file picker never listed {name!r} (see the uiautomator dumps)")
