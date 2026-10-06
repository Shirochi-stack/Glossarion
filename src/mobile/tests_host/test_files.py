"""Host tests for U3 files: FileBridge import/export, IntentRouter, the minimal file browser.

Run from src/mobile with the mobile venv:
    .venv/Scripts/python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_files.py

Pickers, Share and the native bridge are fakes; ``library_core`` is replaced by
an injected Library/Raw folder and recorder. The browser test builds the
screen in the fake Flet session and is skipped without Flet.
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
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from glossarion_mobile.services import files as files_mod  # noqa: E402
from glossarion_mobile.services.files import (  # noqa: E402
    FileBridge,
    FolderPickUnavailable,
    mime_type_for,
    safe_name,
    same_content,
    unique_destination,
)
from glossarion_mobile.services.intents import (  # noqa: E402
    ACTION_ADD_TO_LIBRARY,
    ACTION_OPEN_IN_READER,
    ACTION_TRANSLATE_NEW_CHAT,
    IntentRouter,
)

try:
    import flet  # noqa: F401

    HAVE_FLET = True
except ImportError:
    HAVE_FLET = False

needs_flet = pytest.mark.skipif(not HAVE_FLET, reason="Flet is not installed")


class FakeNative:
    def __init__(self) -> None:
        self.calls = []

    async def call(self, method, *args, default=None, **kwargs):
        self.calls.append((method, args, kwargs))
        return "content://media/external/downloads/1" if method == "save_to_downloads" else default

    async def clear_shared(self, delete_files=False):
        self.calls.append(("clear_shared", (delete_files,), {}))


class FakePicker:
    def __init__(self, picked=None, directory=None, save_result="saved") -> None:
        self.picked = picked or []
        self.directory = directory
        self.save_result = save_result
        self.calls = []

    async def pick_files(self, **kwargs):
        self.calls.append(("pick_files", kwargs))
        return self.picked

    async def get_directory_path(self, **kwargs):
        self.calls.append(("get_directory_path", kwargs))
        if isinstance(self.directory, Exception):
            raise self.directory
        return self.directory

    async def save_file(self, **kwargs):
        self.calls.append(("save_file", kwargs))
        return self.save_result


class FakeShare:
    def __init__(self) -> None:
        self.calls = []

    async def share_files(self, items, **kwargs):
        self.calls.append((items, kwargs))


def bridge(tmp_path, platform="android", **kwargs):
    recorded = kwargs.pop("recorded", [])
    cache = tmp_path / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    return FileBridge(
        inbox_dir=str(tmp_path / "data" / "Inbox"),
        library_raw_dir=str(tmp_path / "Library" / "Raw"),
        record_library_input=recorded.append,
        cache_dirs=[str(cache), str(tmp_path / "temp")],
        platform=platform,
        **kwargs,
    ), cache, recorded


def write(path: Path, data: bytes = b"PK book") -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return str(path)


# ==========================================================================
# Import
# ==========================================================================


def test_import_collisions_dedupe_and_cache_copy_cleanup(tmp_path):
    files, cache, _ = bridge(tmp_path)
    first = write(cache / "file_picker" / "Book.epub", b"v1")
    [a] = files.import_paths([first])
    assert a.path == str(tmp_path / "data" / "Inbox" / "Book.epub") and not a.reused and a.target == "inbox"
    assert not os.path.exists(first)  # the picker's cache copy is removed on Android/iOS
    same = write(cache / "file_picker" / "Book.epub", b"v1")
    [b] = files.import_paths([same])
    assert b.path == a.path and b.reused  # re-importing the same book keeps its name (resume works)
    other = write(cache / "x" / "Book.epub", b"v2")
    [c] = files.import_paths([other])
    assert os.path.basename(c.path) == "Book (2).epub" and not c.reused
    third = write(cache / "y" / "Book.epub", b"v3")
    assert os.path.basename(files.import_paths([third])[0].path) == "Book (3).epub"
    # display names from the picker win over the cache file name; unsafe names are cleaned
    named = write(cache / "tmp123.bin", b"n")
    [d] = files.import_paths([named], names=["../My:Novel?.epub"])
    assert d.name == "My_Novel_.epub" and d.extension == ".epub"
    assert files.import_paths([str(tmp_path / "missing.epub"), None]) == []


def test_desktop_never_removes_the_source_file(tmp_path):
    files, cache, _ = bridge(tmp_path, platform="desktop")
    source = write(cache / "Book.epub")
    files.import_paths([source])
    assert os.path.exists(source)
    files2, cache2, _ = bridge(tmp_path / "b", platform="android")
    outside = write(tmp_path / "Downloads" / "Book.epub")
    files2.import_paths([outside])
    assert os.path.exists(outside)  # outside the app cache: never touched


def test_add_to_library_copies_into_raw_and_registers(tmp_path):
    files, cache, recorded = bridge(tmp_path)
    [imported] = files.import_paths([write(cache / "Book.epub")])
    added = files.add_to_library(imported.path)
    assert added.path == str(tmp_path / "Library" / "Raw" / "Book.epub") and added.target == "library"
    assert recorded == [added.path] and os.path.exists(imported.path)
    again = files.add_to_library(imported.path)
    assert again.reused and recorded == [added.path, added.path]  # registry dedupes by path
    zipped = files.import_paths([write(cache / "Pack.zip")])[0]
    with pytest.raises(ValueError):
        files.add_to_library(zipped.path)


def test_library_dir_defaults_to_the_shared_library_core(tmp_path, monkeypatch):
    fake = types.ModuleType("library_core")
    fake.get_library_raw_dir = lambda: str(tmp_path / "Lib" / "Raw")
    fake.record_library_raw_input = lambda path: fake.recorded.append(path)
    fake.recorded = []
    monkeypatch.setitem(sys.modules, "library_core", fake)
    files = FileBridge(inbox_dir=str(tmp_path / "Inbox"), platform="desktop")
    added = files.import_paths([write(tmp_path / "src" / "Book.epub")], target="library")[0]
    assert added.path == str(tmp_path / "Lib" / "Raw" / "Book.epub") and fake.recorded == [added.path]


def test_folder_import_and_android_fallback(tmp_path):
    files, cache, _ = bridge(tmp_path)
    folder = tmp_path / "Game"
    write(folder / "data" / "Map001.json", b"{}")
    write(folder / "img" / "a.png", b"png")
    write(folder / ".hidden" / "x", b"x")
    first = files.import_folder(str(folder))
    assert first.name == "Game" and first.files == 2 and os.path.exists(os.path.join(first.path, "data", "Map001.json"))
    assert not os.path.exists(os.path.join(first.path, ".hidden"))
    second = files.import_folder(str(folder))
    assert second.name == "Game (2)"
    with pytest.raises(FolderPickUnavailable) as info:
        files.import_folder(str(tmp_path / "nope"))
    assert info.value.fallbacks == ("files", "zip")

    async def scenario():
        files.picker = None
        files._picker = FakePicker(directory="content://com.android.externalstorage/tree/primary%3AGame")
        with pytest.raises(FolderPickUnavailable):
            await files.pick_folder()
        files._picker = FakePicker(directory=None)
        with pytest.raises(FolderPickUnavailable):
            await files.pick_folder()
        files._picker = FakePicker(directory=str(folder))
        imported = await files.pick_folder()
        assert imported.name == "Game (3)"

    asyncio.run(scenario())


def test_pick_files_imports_the_selection(tmp_path):
    files, cache, recorded = bridge(tmp_path)
    picked = [
        types.SimpleNamespace(path=write(cache / "a.tmp", b"a"), name="Novel.epub"),
        types.SimpleNamespace(path=None, name="web-only.epub"),
    ]
    files._picker = FakePicker(picked=picked)

    async def scenario():
        result = await files.pick_files(target="library")
        assert [f.name for f in result] == ["Novel.epub"] and recorded == [result[0].path]
        kwargs = files._picker.calls[0][1]
        assert kwargs["allow_multiple"] and "allowed_extensions" not in kwargs  # mobile: every file
        desktop, _c, _r = bridge(tmp_path / "desk", platform="desktop")
        desktop._picker = FakePicker(picked=[])
        await desktop.pick_files()
        assert "epub" in desktop._picker.calls[0][1]["allowed_extensions"]  # the desktop filter list
        files._picker = FakePicker(picked=[])
        assert await files.pick_files() == []

    asyncio.run(scenario())


def test_helpers():
    assert safe_name("a/b\\c.txt") == "c.txt" and safe_name("") == "file" and safe_name("x" * 300 + ".epub").endswith(
        ".epub")
    assert len(safe_name("x" * 300 + ".epub")) <= 180
    assert mime_type_for("a.epub") == "application/epub+zip" and mime_type_for("a.unknownext").startswith(
        "application/")


def test_unique_destination_and_same_content(tmp_path):
    write(tmp_path / "a.txt", b"1")
    write(tmp_path / "b.txt", b"1")
    write(tmp_path / "c.txt", b"2")
    assert same_content(str(tmp_path / "a.txt"), str(tmp_path / "b.txt"))
    assert not same_content(str(tmp_path / "a.txt"), str(tmp_path / "c.txt"))
    assert unique_destination(str(tmp_path), "a.txt") == str(tmp_path / "a (2).txt")
    assert unique_destination(str(tmp_path), "new.txt") == str(tmp_path / "new.txt")


# ==========================================================================
# Export
# ==========================================================================


def test_export_share_save_downloads_and_options(tmp_path, monkeypatch):
    native = FakeNative()
    share = FakeShare()
    files, cache, _ = bridge(tmp_path, native=native, share_factory=lambda: share)
    files._picker = FakePicker(save_result="content://saved")
    book = write(tmp_path / "out" / "Book" / "Book.epub", b"0123456789")

    async def scenario():
        assert await files.share([book, str(tmp_path / "missing.epub")])
        items, kwargs = share.calls[0]
        assert len(items) == 1 and getattr(items[0], "path", items[0]) == book
        assert not await files.share([str(tmp_path / "missing.epub")])
        result = await files.save_as(book)
        assert result.ok and result.location == "content://saved" and result.size == 10
        assert files._picker.calls[-1][1]["src_bytes"] == b"0123456789"
        assert files._picker.calls[-1][1]["file_name"] == "Book.epub"
        monkeypatch.setattr(files_mod, "SAVE_CONFIRM_BYTES", 5)
        big = await files.save_as(book)
        assert big.needs_confirm and not big.ok and len(files._picker.calls) == 1
        assert (await files.save_as(book, confirmed=True)).ok
        files._picker = FakePicker(save_result=None)
        assert not (await files.save_as(book, confirmed=True)).ok  # cancelled
        uri = await files.save_to_downloads(book)
        assert uri.startswith("content://") and native.calls[-1] == (
            "save_to_downloads", (book, "Book.epub", "application/epub+zip", "Glossarion"), {})
        assert (await files.save_as(str(tmp_path / "missing"))).error

    asyncio.run(scenario())
    ids = {o.id: o for o in files.export_options(book)}
    assert ids["downloads"].disabled_reason is None and "files" not in ids

    desktop, _cache, _ = bridge(tmp_path / "d", platform="desktop")
    target = tmp_path / "picked" / "Copy.epub"
    target.parent.mkdir(parents=True)
    desktop._picker = FakePicker(save_result=str(target))

    async def desktop_save():
        result = await desktop.save_as(book, confirmed=True)
        assert result.ok and target.read_bytes() == b"0123456789"  # desktop dev writes the bytes itself
        assert await desktop.save_to_downloads(book) is None

    asyncio.run(desktop_save())
    assert {o.id: o for o in desktop.export_options(book)}["downloads"].disabled_reason == "Android only"

    launched = []

    class Launcher:
        async def launch_url(self, url):
            launched.append(url)

    docs = tmp_path / "Documents" / "Glossarion"
    ios, _c, _ = bridge(tmp_path / "i", platform="ios", url_launcher=Launcher(), files_visible_root=str(docs))
    inside = write(docs / "Output" / "Book" / "Book.epub")
    options = {o.id: o for o in ios.export_options(inside)}
    assert options["files"].disabled_reason is None
    assert {o.id: o for o in ios.export_options(book)}["files"].disabled_reason

    async def ios_files():
        assert await ios.export("files", inside)
        assert launched == ["shareddocuments://" + os.path.dirname(inside)]

    asyncio.run(ios_files())


# ==========================================================================
# IntentRouter
# ==========================================================================


def test_intent_router_imports_shared_files_and_offers_actions(tmp_path):
    native = FakeNative()
    files, cache, recorded = bridge(tmp_path, native=native)
    presented, translated, notes = [], [], []
    router = IntentRouter(files=files, native=native, present=presented.append,
                          handlers={ACTION_TRANSLATE_NEW_CHAT: translated.append}, notify=notes.append)
    shared = write(cache / "shared" / "Book.epub")
    pack = write(cache / "shared" / "Pack.zip")
    items = [
        {"id": "1", "kind": "file", "path": shared, "name": "Book.epub", "uri": "content://x/1"},
        {"id": "2", "kind": "file", "path": pack, "name": "Pack.zip"},
        {"id": "3", "kind": "text", "text": "번역해 주세요"},
        {"id": "4", "kind": "url", "text": "glossarion://app/library", "source": "launch"},
        {"id": "5", "kind": "text", "text": "content://evil/route"},
        {"id": "6", "kind": "url", "text": "https://example.com/novel"},
        {"id": "7", "kind": "file", "path": None, "error": "copy failed"},
    ]

    async def scenario():
        imports = await router.handle(items)
        assert [i.label for i in imports] == ["번역해 주세요", "https://example.com/novel", "Book.epub", "Pack.zip"]
        assert presented == [imports]
        book, zipped = imports[2], imports[3]
        assert book.imported.path == str(tmp_path / "data" / "Inbox" / "Book.epub") and not os.path.exists(shared)
        actions = {a.id: a for a in book.actions}
        assert actions[ACTION_TRANSLATE_NEW_CHAT].disabled_reason is None
        assert actions[ACTION_ADD_TO_LIBRARY].disabled_reason is None
        assert actions[ACTION_OPEN_IN_READER].disabled_reason  # Reader: U5
        assert {a.id: a for a in zipped.actions}[ACTION_ADD_TO_LIBRARY].disabled_reason
        assert [a.id for a in imports[0].actions] == ["compose"]
        assert ("clear_shared", (False,), {}) in native.calls
        # the same items again (initial + event) are ignored
        assert await router.handle(items) == []
        await router.perform(ACTION_TRANSLATE_NEW_CHAT, book)
        assert translated == [book]
        added = await router.perform(ACTION_ADD_TO_LIBRARY, book)
        assert added.path == str(tmp_path / "Library" / "Raw" / "Book.epub") and recorded == [added.path]
        assert notes == ["Added to Library: Book.epub"]
        await router.perform(ACTION_OPEN_IN_READER, book)
        assert notes[-1] == "Not available yet"

    asyncio.run(scenario())
    bare = IntentRouter(files=files)
    imp = types.SimpleNamespace(imported=types.SimpleNamespace(extension=".epub"))
    assert {a.id: a for a in bare.actions_for(imp)}[ACTION_TRANSLATE_NEW_CHAT].disabled_reason


# ==========================================================================
# File browser (Flet)
# ==========================================================================


_TB_SPEC = importlib.util.spec_from_file_location("_glossarion_tb_helpers_files", Path(__file__).with_name(
    "test_bootstrap.py"))
_TB = importlib.util.module_from_spec(_TB_SPEC)
_TB_SPEC.loader.exec_module(_TB)


@needs_flet
def test_file_browser_lists_guards_roots_and_exports(tmp_path):
    pytest.importorskip("msgpack")
    from glossarion_mobile.state.prefs import Prefs
    from glossarion_mobile.ui.router import parse_route
    from glossarion_mobile.ui.screens.files import FileBrowserScreen, describe_size, resolve_location

    output = tmp_path / "Output"
    write(output / "Book" / "Book.epub", b"x" * 2048)
    write(output / "Book" / "translation_progress.json", b"{}")
    write(output / "Alpha.txt", b"a")
    write(output / "Book" / "chunk.part", b"")
    (output / "Empty").mkdir()
    prefs = Prefs(tmp_path / "mobile_state.json")
    prefs.load()
    roots = {"output": str(output), "library": "", "inbox": str(tmp_path / "Inbox"), "chats": ""}
    outside = prefs.file_ref(str(tmp_path))
    assert resolve_location(roots, "output", outside, prefs.resolve_file_ref)[2] == \
        "This folder is outside the app's storage"
    assert resolve_location(roots, "library", None, prefs.resolve_file_ref)[2] == "This location is not available"
    assert resolve_location(roots, "output", "abcdefabcdef", prefs.resolve_file_ref)[2] == "This folder link has expired"
    assert describe_size(2048) == "2.0 KB" and describe_size(10) == "10 B"

    async def scenario():
        conn, session = _TB._fake_session("android")
        page = session.page
        navigated = []
        files, _cache, _ = bridge(tmp_path)
        screen = FileBrowserScreen(parse_route("/tools/files/output"), roots=roots, files=files, page=page,
                                   navigate=lambda name, params=None: navigated.append((name, params)),
                                   file_ref=prefs.file_ref, resolve_ref=prefs.resolve_file_ref)
        page.views[0].controls.append(screen.get_body())
        page.update()
        screen.did_show()
        assert [e.name for e in screen.entries] == ["Book", "Empty", "Alpha.txt"]
        assert screen.status.value == "2 folders · 1 files"
        screen._on_entry(screen.entries[0])
        name, params = navigated[-1]
        assert name == "tools.files.folder" and params["root"] == "output"
        assert prefs.resolve_file_ref(params["fid"]) == str(output / "Book")

        folder = FileBrowserScreen(parse_route(f"/tools/files/output/{params['fid']}"), roots=roots, files=files,
                                   page=page, navigate=lambda name, params=None: navigated.append((name, params)),
                                   file_ref=prefs.file_ref, resolve_ref=prefs.resolve_file_ref)
        page.views[0].controls.append(folder.get_body())
        page.update()
        folder.did_show()
        assert [e.name for e in folder.entries] == ["Book.epub", "translation_progress.json"]
        assert [getattr(c, "content", None) for c in folder.crumbs.controls if isinstance(c, flet.TextButton)] == [
            "Output", "Book"]
        sheet = folder._on_entry(folder.entries[0])
        labels = [item.label for item in sheet.items]
        assert labels[:3] == ["Share…", "Save to…", "Save to Downloads"] and "Delete" in labels
        assert sheet.item("Delete").disabled_reason is None  # U7 file tools
        blocked = FileBrowserScreen(parse_route(f"/tools/files/output/{outside}"), roots=roots,
                                    resolve_ref=prefs.resolve_file_ref)
        body = blocked.get_body()
        assert blocked.error and blocked.list_view.controls[0].key == "files-error" and body is not None
        prefs.close()

    asyncio.run(scenario())
