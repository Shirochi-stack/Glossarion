"""Host tests for the U10 phone folder (Android ``Downloads/Glossarion``) and the copy-once mirror.

U10 folds the pre-U10 "Mirror outputs to Downloads/Glossarion" switch into cloud sync: books (EPUB,
PDF, TXT, HTML) reach the phone folder only as the cloud sync's phone-folder destination (one entry
per output, overwritten in place: ``services/cloud_sync``), while the Storage switch ``mirror_outputs``
copies the other outputs once. Covered here:

* ``mirror_output(skip_books=True)`` + ``LibraryService.on_job_finished``: once the cloud sync is
  installed every output goes to exactly one place (the cloud sync's ``output_kind`` decides);
  chat runs are never copied from their temporary run root; without the cloud sync nothing changes.
* ``FileBridge.save_to_downloads`` keeps the copy-once call; ``FileBridge.phone_folder_reason``
  (Android only / Android 10 or later / no native service; the facts read once).
* Settings › Storage: where books go (the cloud sync ``ui_state``), the link and the "My cloud app
  isn't listed" help of Settings › Cloud sync & sharing, the copy-once switch and its ReasonChips.
* The native contract: ``save_to_downloads(replace_uri=)`` through the extension and Kotlin, and the
  NativeBridge waiting as long as the extension for a large copy (Integrate patch).

No network, no real Downloads folder.

Run from src/mobile with the mobile venv:
    .venv/Scripts/python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_phone_folder.py
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = MOBILE_DIR / "app"
SRC_DIR = MOBILE_DIR.parent
EXT_DIR = MOBILE_DIR / "extensions" / "flet_glossarion_native"
KOTLIN_DIR = (EXT_DIR / "src" / "flutter" / "flet_glossarion_native" / "android" / "src" / "main" / "kotlin" / "com"
              / "glossarion" / "flet_glossarion_native")
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from glossarion_mobile.services import native as native_mod  # noqa: E402
from glossarion_mobile.services.files import (  # noqa: E402
    DOWNLOADS_SUBDIR,
    MIRROR_PREF,
    PHONE_FOLDER_LABEL,
    PHONE_FOLDER_NEEDS_ANDROID_10,
    PHONE_FOLDER_NOT_IN_BUILD,
    PHONE_FOLDER_ONLY_ANDROID,
    FileBridge,
)
from glossarion_mobile.services.native import NativeBridge, NativeStub  # noqa: E402


def _has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


needs_flet = pytest.mark.skipif(not _has("flet"), reason="flet not installed")


def _run(coro):
    return asyncio.run(coro)


class Prefs(dict):
    def set(self, key, value):
        self[key] = value


class FakeNative:
    """The GlossarionNative surface ``phone_folder_reason`` reads (wrapped by the real NativeBridge)."""

    def __init__(self, sdk_int: int = 34) -> None:
        self.sdk_int = sdk_int
        self.info_reads = 0

    async def get_platform_info(self) -> dict:
        self.info_reads += 1
        return {"platform": "android", "native": True, "sdk_int": self.sdk_int,
                "save_to_downloads": self.sdk_int >= 29}


def _bridge(tmp_path, *, platform="android", native=None):
    return FileBridge(inbox_dir=str(tmp_path / "Inbox"), platform=platform,
                      native=native if native is not None else NativeBridge(native=FakeNative()))


# ==========================================================================
# FileBridge
# ==========================================================================


def test_save_to_downloads_stays_the_copy_once_call(tmp_path):
    book = tmp_path / "Book.epub"
    book.write_bytes(b"PK1")
    calls = []

    class Recording:
        async def call(self, method, *args, default=None, **kwargs):
            calls.append((method, args, kwargs))
            return "content://media/external/downloads/1"

    files = FileBridge(inbox_dir=str(tmp_path / "Inbox"), platform="android", native=Recording())
    assert _run(files.save_to_downloads(str(book))) == "content://media/external/downloads/1"
    # no replace flag: a new entry in the flat Downloads/Glossarion (books in place are the cloud sync's job)
    assert calls == [("save_to_downloads", (str(book), "Book.epub", "application/epub+zip", DOWNLOADS_SUBDIR), {})]
    assert DOWNLOADS_SUBDIR == "Glossarion" and PHONE_FOLDER_LABEL == "Downloads/Glossarion"
    ios = FileBridge(inbox_dir=str(tmp_path / "Inbox"), platform="ios", native=Recording())
    assert _run(ios.save_to_downloads(str(book))) is None and len(calls) == 1


def test_phone_folder_reason_says_why_and_reads_the_facts_once(tmp_path):
    async def scenario():
        assert await _bridge(tmp_path, platform="ios").phone_folder_reason() == PHONE_FOLDER_ONLY_ANDROID
        assert await _bridge(tmp_path, platform="desktop").phone_folder_reason() == PHONE_FOLDER_ONLY_ANDROID
        old = _bridge(tmp_path, native=NativeBridge(native=FakeNative(sdk_int=28)))
        assert await old.phone_folder_reason() == PHONE_FOLDER_NEEDS_ANDROID_10
        stub = _bridge(tmp_path, native=NativeBridge(native=NativeStub("no extension")))
        assert await stub.phone_folder_reason() == PHONE_FOLDER_NOT_IN_BUILD
        none = FileBridge(inbox_dir=str(tmp_path / "Inbox"), platform="android", native=None)
        assert await none.phone_folder_reason() == PHONE_FOLDER_NOT_IN_BUILD
        fake = FakeNative()
        files = _bridge(tmp_path, native=NativeBridge(native=fake))
        for _ in range(3):
            assert await files.phone_folder_reason() is None
        assert fake.info_reads == 1

    _run(scenario())


# ==========================================================================
# Copy-once mirror of the other outputs + the Library's job-end hook
# ==========================================================================


BOOKS = ("Book.epub", "New Title.pdf", "Book_translated.txt", "Page_translated.html")
OTHERS = ("Scan_translated.html", "cover.png", "ep1_translated.srt", "glossary.csv")


def _outputs_folder(tmp_path):
    out = tmp_path / "Output" / "Book"
    out.mkdir(parents=True)
    # Scan_translated.html is the debug companion of Scan_translated.pdf: not a book output, so copied once
    for name in BOOKS + OTHERS + ("Scan_translated.pdf",):
        (out / name).write_bytes(b"x")
    return out


class CopyOnce:
    platform = "android"

    def __init__(self) -> None:
        self.saved: list = []
        self.io_calls = 0

    async def run_io(self, fn, *args):
        self.io_calls += 1
        return fn(*args)

    async def save_to_downloads(self, path):
        self.saved.append(os.path.basename(path))
        return "content://" + os.path.basename(path)


@needs_flet
def test_mirror_output_leaves_the_cloud_sync_outputs_alone(tmp_path):
    from glossarion_mobile.services.cloud_sync import output_kind
    from glossarion_mobile.ui.screens.storage import mirror_output

    out = _outputs_folder(tmp_path)
    files = CopyOnce()
    prefs = Prefs({MIRROR_PREF: True})
    uris = _run(mirror_output(files, str(out), prefs, skip_books=True))
    assert sorted(files.saved) == sorted(OTHERS) and len(uris) == len(OTHERS)
    assert files.io_calls == 1  # listed (and the companion-HTML check made) off the loop
    # every output goes to exactly one place: copied once here, or synced by the cloud sync
    for name in os.listdir(out):
        assert bool(output_kind(str(out / name))) == (name not in files.saved), name
    files.saved.clear()
    assert len(_run(mirror_output(files, str(out), prefs))) == len(os.listdir(out))  # pre-U10: everything
    assert _run(mirror_output(files, str(out / "Book.epub"), prefs, skip_books=True)) == []
    assert _run(mirror_output(files, str(out / "cover.png"), Prefs({MIRROR_PREF: False}), skip_books=True)) == []


@needs_flet
def test_job_end_mirror_copies_only_other_outputs_once_cloud_sync_is_installed(tmp_path):
    from glossarion_mobile.services.library import LibraryService

    out = _outputs_folder(tmp_path)
    files = CopyOnce()
    service = LibraryService(prefs=Prefs({MIRROR_PREF: True}), files=files, config={})
    targets = tuple(str(out / name) for name in sorted(os.listdir(out)))

    def job(kind, state="DONE"):
        spec = types.SimpleNamespace(kind=kind, inputs=(str(out),), params={"folder": str(out)})
        return types.SimpleNamespace(spec=spec, state=types.SimpleNamespace(value=state), outputs=targets)

    assert len(_run(service.on_job_finished(job("compile_epub")))) == len(targets)  # no cloud sync: as before
    assert service.cloud_sync is None
    service.cloud_sync = object()  # the app sets it when it installs the cloud sync
    files.saved.clear()
    uris = _run(service.on_job_finished(job("translate")))
    assert sorted(files.saved) == sorted(OTHERS) and len(uris) == len(OTHERS)
    files.saved.clear()
    # a chat run's outputs are in a run root the chat moves away: never copied from there
    assert _run(service.on_job_finished(job("direct_text"))) == [] and files.saved == []
    assert _run(service.on_job_finished(job("translate", "FAILED"))) == [] and files.saved == []
    service.prefs[MIRROR_PREF] = False
    assert _run(service.on_job_finished(job("manga"))) == [] and files.saved == []


# ==========================================================================
# Settings › Storage
# ==========================================================================


class Ctx:
    def __init__(self, prefs=None) -> None:
        self.prefs = prefs
        self.page = None
        self.store = None
        self.routes: list = []
        self.messages: list = []

    def say(self, message, *_a):
        self.messages.append(message)

    async def run_io(self, fn, *args):
        return fn(*args)

    def spawn(self, coro):
        return asyncio.ensure_future(coro)

    def go(self, name, params=None, *, fragment=None):
        self.routes.append(name)
        return name


def _walk(control):
    yield control
    for attr in ("controls", "content"):
        child = getattr(control, attr, None)
        if isinstance(child, list):
            for item in child:
                yield from _walk(item)
        elif child is not None and not isinstance(child, (str, int, float)):
            yield from _walk(child)


def _by_key(root, key):
    return next((c for c in _walk(root) if getattr(c, "key", None) == key), None)


def _texts(root):
    return [c.value for c in _walk(root) if type(c).__name__ == "Text" and isinstance(getattr(c, "value", None), str)]


class FakeFacade:
    """``CloudFacade`` of the Cloud sync screen (its ``snapshot`` dict is the contract this page reads)."""

    def __init__(self, service) -> None:
        self.service = service() if callable(service) else service

    @property
    def available(self) -> bool:
        return self.service is not None

    def snapshot(self) -> dict:
        return dict(self.service.state)


@needs_flet
def test_storage_page_shows_where_books_go_and_the_copy_switch(tmp_path, monkeypatch):
    from glossarion_mobile.ui.router import ROUTES_BY_NAME, parse_route
    from glossarion_mobile.ui.screens import cloud_sync as cloud_screen
    from glossarion_mobile.ui.screens.storage import (
        NOT_LISTED_TITLE,
        OTHER_OUTPUTS_LABEL,
        StorageScreen,
        phone_folder_books_text,
    )

    prefs = Prefs()
    ctx = Ctx(prefs)
    files = _bridge(tmp_path)
    screen = StorageScreen(parse_route("/settings/storage"), ctx, platform="android", files=files)
    body = screen.get_body()
    section = _by_key(body, "storage-phone-folder")
    assert section is not None and _by_key(section, "storage-mirror") is screen.mirror_switch
    assert OTHER_OUTPUTS_LABEL in _texts(section) and not screen.mirror_switch.disabled
    assert _by_key(section, "storage-mirror-reason") is None
    assert "not copied" in screen.books_text.value  # no cloud sync in this session
    screen.set_mirror(True)
    assert prefs[MIRROR_PREF] is True
    # the help is the Cloud sync screen's own "My cloud app isn't listed" text (TeraBox folder backup)
    assert _by_key(section, "storage-phone-help") is not None
    sheet = screen.show_not_listed_help()
    assert sheet.title == NOT_LISTED_TITLE and sheet.body == cloud_screen.NOT_LISTED_HELP_ANDROID
    assert "TeraBox" in sheet.body and "Download/Glossarion" in sheet.body
    # the link exists only once the router has the Cloud sync route (never a broken navigation)
    has_route = cloud_screen.ROUTE_NAME in ROUTES_BY_NAME
    assert (_by_key(section, "storage-phone-cloud") is not None) == has_route
    assert screen.open_cloud_settings() == (cloud_screen.ROUTE_NAME if has_route else None)
    assert ctx.routes == ([cloud_screen.ROUTE_NAME] if has_route else [])

    async def native_checks():
        assert await screen.check_phone_folder() is None and not screen.mirror_switch.disabled
        old = _bridge(tmp_path, native=NativeBridge(native=FakeNative(sdk_int=28)))
        nine = StorageScreen(parse_route("/settings/storage"), Ctx(Prefs()), platform="android", files=old)
        nine.get_body()
        assert await nine.check_phone_folder() == PHONE_FOLDER_NEEDS_ANDROID_10
        assert nine.mirror_switch.disabled and nine.mirror_reason_row.visible
        chip = _by_key(nine.mirror_reason_row, "storage-mirror-reason")
        assert chip is not None and chip.reason == PHONE_FOLDER_NEEDS_ANDROID_10
        assert await nine.check_phone_folder() == PHONE_FOLDER_NEEDS_ANDROID_10
        assert nine.mirror_reason_row.controls[0] is chip  # a keyed chip is not rebuilt

    _run(native_checks())

    ios = StorageScreen(parse_route("/settings/storage"), Ctx(Prefs()), platform="ios")
    ios_body = ios.get_body()
    assert ios.mirror_switch.disabled and _by_key(ios_body, "storage-mirror-reason").reason == PHONE_FOLDER_ONLY_ANDROID
    assert "Files" in ios.books_text.value and _by_key(ios_body, "storage-phone-help") is None

    # where books go, from the cloud sync's ui_state
    phone_on = {"destination": {"mode": "phone", "label": "Downloads/Glossarion"}, "enabled": True}
    assert "replaced in place" in phone_folder_books_text(phone_on)
    assert "automatic copies are off" in phone_folder_books_text({**phone_on, "enabled": False})
    drive = {"destination": {"mode": "folder", "label": "Glossarion", "provider_label": "Drive"}, "enabled": True}
    assert "Drive › Glossarion" in phone_folder_books_text(drive)  # the Cloud sync screen's wording
    assert "not to the phone folder" in phone_folder_books_text(drive)
    assert "choose Phone folder" in phone_folder_books_text({"destination": None, "enabled": False})

    monkeypatch.setattr(cloud_screen, "CloudFacade", FakeFacade)

    async def with_cloud():
        cloud = types.SimpleNamespace(state=phone_on)
        live = StorageScreen(parse_route("/settings/storage"), Ctx(Prefs()), platform="android", files=files,
                             cloud=lambda: cloud)
        live.get_body()
        state = await live.refresh_cloud_state()
        assert state and state["destination"]["mode"] == "phone"
        assert "replaced in place" in live.books_text.value
        cloud.state = drive
        live.app_resumed()  # the user changed the destination meanwhile
        for _ in range(50):
            await asyncio.sleep(0.01)
            if "Drive" in live.books_text.value:
                break
        assert "Drive › Glossarion" in live.books_text.value
        gone = StorageScreen(parse_route("/settings/storage"), Ctx(Prefs()), platform="android", files=files,
                             cloud=lambda: None)
        gone.get_body()
        assert await gone.refresh_cloud_state() is None and "not copied" in gone.books_text.value

    _run(with_cloud())


# ==========================================================================
# Native contract
# ==========================================================================


@needs_flet
def test_extension_forwards_the_replace_flag_and_kotlin_overwrites_in_place():
    import flet as ft

    sys.path.insert(0, str(EXT_DIR / "src"))
    try:
        from flet_glossarion_native import GlossarionNative
    finally:
        sys.path.remove(str(EXT_DIR / "src"))
    native = GlossarionNative()
    page = types.SimpleNamespace(platform=ft.PagePlatform.ANDROID, web=False)
    native._page_or_none = lambda: page
    calls = []

    async def fake_invoke(method_name, arguments=None, timeout=None):
        calls.append((method_name, arguments, timeout))
        return "content://media/external/downloads/3"

    native._invoke_method = fake_invoke
    uri = _run(native.save_to_downloads("/x/a.epub", "a.epub", "application/epub+zip", "Glossarion/Book",
                                        replace_uri="content://media/external/downloads/3"))
    assert uri == "content://media/external/downloads/3"
    assert calls[-1][1]["replace_uri"] == "content://media/external/downloads/3" and calls[-1][2] >= 600
    _run(native.save_to_downloads("/x/a.epub", "a.epub", "application/epub+zip"))
    assert not calls[-1][1].get("replace_uri")
    kotlin = "\n".join(path.read_text(encoding="utf-8") for path in sorted(KOTLIN_DIR.glob("*.kt")))
    assert 'argument<String>("replace_uri")' in kotlin
    assert '"wt"' in kotlin  # the entry is truncated and rewritten in place


def test_bridge_does_not_cut_a_large_copy_short():
    class Slow(FakeNative):
        async def save_to_downloads(self, *args, **kwargs):
            await asyncio.sleep(0.2)
            return "content://media/external/downloads/9"

    bridge = NativeBridge(native=Slow(), timeout=0.05)
    assert native_mod.METHOD_TIMEOUTS.get("save_to_downloads", 0) >= 900  # the extension's own limit
    original = dict(native_mod.METHOD_TIMEOUTS)
    native_mod.METHOD_TIMEOUTS["save_to_downloads"] = 1.0
    try:
        assert _run(bridge.call("save_to_downloads", "/x", "x", "text/plain", "Glossarion")) == (
            "content://media/external/downloads/9")
        assert _run(bridge.call("get_platform_info", default={})).get("native") is True  # others keep 45 s
    finally:
        native_mod.METHOD_TIMEOUTS.clear()
        native_mod.METHOD_TIMEOUTS.update(original)
