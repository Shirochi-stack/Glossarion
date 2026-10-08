"""Host tests for U5 Open-with / Share routing: "Open in Reader" and "Add to Library" through the Library.

* IntentRouter offers "Open in Reader" for EPUBs once a handler is registered (the Library
  feature registers one); other files say why.
* ``LibraryFeature`` registers the handlers and ``FileBridge.library_import``: a shared file
  (an Inbox copy) goes through ``library_core.import_paths(copy_into_library=True,
  record_origins=True)`` (copy into Library/Raw + registry + workspace scaffold + origins),
  a picker cache copy is copied by FileBridge under its display name and registered in place.
* "Open in Reader" opens ``ReaderFeature.open_book`` when installed, else routes
  ``/reader/<bid>`` with an id ``LibraryService.book_for_bid`` resolves back to the file.
* ``FileBridge.unique_destination`` uses the shared ``library_core.unique_destination``.

``library_core`` is a fake module here (the real one would write the developer's Library).

Run from src/mobile with the mobile venv:
    .venv/Scripts/python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_intents.py
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import types
from pathlib import Path

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = MOBILE_DIR / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from glossarion_mobile.services import files as files_mod  # noqa: E402
from glossarion_mobile.services.files import FileBridge, unique_destination  # noqa: E402
from glossarion_mobile.services.intents import (  # noqa: E402
    ACTION_ADD_TO_LIBRARY,
    ACTION_OPEN_IN_READER,
    ACTION_TRANSLATE_NEW_CHAT,
    READER_EXTENSIONS,
    READER_REASON,
    READER_TYPES_REASON,
    IntentRouter,
)
from glossarion_mobile.services.library import LibraryService, SharedCore  # noqa: E402

try:
    import flet  # noqa: F401

    HAVE_FLET = True
except ImportError:
    HAVE_FLET = False

needs_flet = pytest.mark.skipif(not HAVE_FLET, reason="Flet is not installed")


def write(path: Path, data: bytes = b"PK book") -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return str(path)


class FakePrefs:
    def __init__(self) -> None:
        self.refs: dict = {}
        self.data: dict = {}

    def file_ref(self, path, *, kind=None):
        from glossarion_mobile.state.prefs import file_ref_id

        fid = file_ref_id(os.fspath(path))
        self.refs[fid] = os.path.abspath(os.fspath(path))
        return fid

    def resolve_file_ref(self, fid, *, touch=True):
        return self.refs.get(fid)

    def get(self, key, default=None):
        return self.data.get(key, default)

    def set(self, key, value):
        self.data[key] = value

    def reader_position(self, bid):
        return None


def fake_library_core(tmp_path: Path) -> types.ModuleType:
    """The ``library_core`` calls the mobile import path makes (desktop semantics, tmp folders)."""
    module = types.ModuleType("library_core")
    library = tmp_path / "Library"
    module.calls = []

    def raw_dir():
        (library / "Raw").mkdir(parents=True, exist_ok=True)
        return str(library / "Raw")

    def translated_dir():
        (library / "Translated").mkdir(parents=True, exist_ok=True)
        return str(library / "Translated")

    def unique_dest(directory, name, include_dirs=False):
        module.calls.append(("unique_destination", name, include_dirs))
        candidate = os.path.join(directory, name)
        stem, ext = (name, "") if include_dirs and os.path.isdir(candidate) else os.path.splitext(name)
        counter = 2
        while os.path.isfile(candidate) or (include_dirs and os.path.exists(candidate)):
            candidate = os.path.join(directory, f"{stem} ({counter}){ext}")
            counter += 1
        return candidate

    def import_paths(paths, target="raw", config=None, *, copy_into_library=False, record_origins=False):
        module.calls.append(("import_paths", list(paths), target, copy_into_library, record_origins))
        dest_dir = translated_dir() if target == "translated" else raw_dir()
        copied, imported = [], []
        for src in paths:
            if os.path.normcase(os.path.dirname(os.path.abspath(src))) == os.path.normcase(dest_dir):
                imported.append(src)
                continue
            dest = unique_dest(dest_dir, os.path.basename(src))
            shutil.copy2(src, dest)
            copied.append({"source": src, "path": dest, "reused": False})
            imported.append(dest)
        return {"imported": imported, "skipped": [], "errors": [], "copied": copied}

    module.get_library_raw_dir = raw_dir
    module.get_library_translated_dir = translated_dir
    module.unique_destination = unique_dest
    module.import_paths = import_paths
    module.record_library_raw_input = lambda path: module.calls.append(("record_raw", path))
    module.scan_library = lambda config=None: ([], [])
    module.library_root_path = lambda: str(library)
    return module


@pytest.fixture
def lib(tmp_path, monkeypatch):
    module = fake_library_core(tmp_path)
    monkeypatch.setitem(sys.modules, "library_core", module)
    return module


def bridge(tmp_path, platform="android"):
    cache = tmp_path / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    return FileBridge(inbox_dir=str(tmp_path / "data" / "Inbox"), cache_dirs=[str(cache)], platform=platform), cache


# ==========================================================================
# IntentRouter actions
# ==========================================================================


def test_open_in_reader_needs_a_handler_and_an_epub_or_txt(tmp_path, lib):
    files, cache = bridge(tmp_path)
    bare = IntentRouter(files=files)
    epub = types.SimpleNamespace(imported=types.SimpleNamespace(extension=".epub"))
    txt = types.SimpleNamespace(imported=types.SimpleNamespace(extension=".txt"))
    pdf = types.SimpleNamespace(imported=types.SimpleNamespace(extension=".pdf"))
    assert {a.id: a for a in bare.actions_for(epub)}[ACTION_OPEN_IN_READER].disabled_reason == READER_REASON
    assert {a.id: a for a in bare.actions_for(txt)}[ACTION_OPEN_IN_READER].disabled_reason == READER_REASON
    router = IntentRouter(files=files, handlers={ACTION_OPEN_IN_READER: lambda imp: "bid"})
    actions = {a.id: a for a in router.actions_for(epub)}
    assert actions[ACTION_OPEN_IN_READER].disabled_reason is None
    assert actions[ACTION_ADD_TO_LIBRARY].disabled_reason is None
    txt_actions = {a.id: a for a in router.actions_for(txt)}
    assert txt_actions[ACTION_OPEN_IN_READER].disabled_reason is None  # device report: the Reader reads TXT
    assert txt_actions[ACTION_ADD_TO_LIBRARY].disabled_reason is None
    assert READER_EXTENSIONS == (".epub", ".txt")
    assert {a.id: a for a in router.actions_for(pdf)}[ACTION_OPEN_IN_READER].disabled_reason == \
        READER_TYPES_REASON == "The Reader opens EPUB and TXT files"


# ==========================================================================
# FileBridge -> library_core
# ==========================================================================


def test_unique_destination_uses_the_shared_library_rule(tmp_path, lib):
    write(tmp_path / "a.txt", b"1")
    (tmp_path / "Game").mkdir()
    assert unique_destination(str(tmp_path), "a.txt") == str(tmp_path / "a (2).txt")
    assert unique_destination(str(tmp_path), "Game", folder=True) == str(tmp_path / "Game (2)")
    assert ("unique_destination", "a.txt", False) in lib.calls and ("unique_destination", "Game", True) in lib.calls


def test_add_to_library_from_the_inbox_records_origins(tmp_path, lib):
    files, cache = bridge(tmp_path)
    service = LibraryService(core=SharedCore({"library_core": lib}), prefs=FakePrefs())
    files.library_import = lambda paths, target, record: service.import_paths_blocking(paths, target, record)
    [inbox] = files.import_paths([write(cache / "share" / "Novel.epub", b"novel")], names=["Novel.epub"])
    added = files.add_to_library(inbox.path)
    assert added.path == str(tmp_path / "Library" / "Raw" / "Novel.epub") and added.target == "library"
    assert os.path.exists(inbox.path)  # the Inbox copy stays (it is the origin Undo returns to)
    call = [c for c in lib.calls if c[0] == "import_paths"][-1]
    assert call == ("import_paths", [inbox.path], "raw", True, True)
    assert service.dirty


def test_picker_copies_are_copied_here_and_registered_in_place(tmp_path, lib):
    files, cache = bridge(tmp_path)
    service = LibraryService(core=SharedCore({"library_core": lib}), prefs=FakePrefs())
    files.library_import = lambda paths, target, record: service.import_paths_blocking(paths, target, record)
    picked = write(cache / "file_picker" / "tmp-123.bin", b"raw")
    [imported] = files.import_paths([picked], target="library", names=["My Novel.txt"])
    assert imported.path == str(tmp_path / "Library" / "Raw" / "My Novel.txt")
    assert not os.path.exists(picked)  # the picker's cache copy is removed
    call = [c for c in lib.calls if c[0] == "import_paths"][-1]
    assert call == ("import_paths", [imported.path], "raw", True, False)  # already in Raw: registered in place
    before = len([c for c in lib.calls if c[0] == "import_paths"])
    translated = files.import_paths([write(cache / "fp" / "Done.epub", b"t")], target="translated")[0]
    assert translated.path == str(tmp_path / "Library" / "Translated" / "Done.epub")
    # Library/Translated needs no registry entry: the shelf scan lists the folder
    assert len([c for c in lib.calls if c[0] == "import_paths"]) == before


# ==========================================================================
# LibraryFeature handlers
# ==========================================================================


def _app(tmp_path, files, intents, *, reader=None):
    navigated = []
    notes = []
    app = types.SimpleNamespace(
        page=None, dispatcher=None, paths=None, config_store=None, prefs=FakePrefs(), files=files, jobs=None,
        intents=intents, shell=None, state=None, haptics=None,
        navigate_to=lambda name, params=None, query=None: navigated.append((name, params, query)),
        notify=lambda message, action=None, on_action=None: notes.append((message, action, on_action)),
    )
    if reader is not None:
        app.reader = reader
    return app, navigated, notes


@needs_flet
def test_library_feature_registers_reader_and_library_handlers(tmp_path, lib):
    from glossarion_mobile.ui.library.feature import IMPLEMENTED_ROUTES, LibraryFeature

    files, cache = bridge(tmp_path)
    intents = IntentRouter(files=files, handlers={ACTION_TRANSLATE_NEW_CHAT: lambda imp: None})
    app, navigated, notes = _app(tmp_path, files, intents)
    feature = asyncio.run(LibraryFeature.install(app, service=LibraryService(
        core=SharedCore({"library_core": lib}), prefs=app.prefs, files=files)))
    assert app.library is feature.service and app.library_feature is feature
    assert ACTION_OPEN_IN_READER in intents.handlers and ACTION_ADD_TO_LIBRARY in intents.handlers
    assert files.library_import is not None
    assert {"library", "library.book", "library.book.metadata", "library.scan_raw", "tools.progress"} <= \
        IMPLEMENTED_ROUTES

    shared = write(cache / "shared" / "Book.epub", b"epub!")

    async def scenario():
        [imp] = await intents.handle([{"id": "1", "kind": "file", "path": shared, "name": "Book.epub"}])
        assert {a.id: a for a in imp.actions}[ACTION_OPEN_IN_READER].disabled_reason is None
        bid = await intents.perform(ACTION_OPEN_IN_READER, imp)
        assert navigated[-1][0] == "reader" and navigated[-1][1] == {"bid": bid}
        assert feature.service.book_for_bid(bid)["path"] == os.path.abspath(imp.imported.path)
        added = await intents.perform(ACTION_ADD_TO_LIBRARY, imp)
        assert added.path == str(tmp_path / "Library" / "Raw" / "Book.epub")
        assert ("import_paths", [imp.imported.path], "raw", True, True) in lib.calls
        assert notes[-1][0] == "Added to Library: Book.epub"

    asyncio.run(scenario())


@needs_flet
def test_open_in_reader_prefers_the_reader_feature(tmp_path, lib):
    from glossarion_mobile.ui.library.feature import LibraryFeature

    opened = []
    reader = types.SimpleNamespace(open_book=lambda book=None, **kw: opened.append((book, kw)) or "abcdef123456")
    files, cache = bridge(tmp_path)
    intents = IntentRouter(files=files)
    app, navigated, notes = _app(tmp_path, files, intents, reader=reader)
    asyncio.run(LibraryFeature.install(app, service=LibraryService(core=SharedCore({"library_core": lib}),
                                                                     prefs=app.prefs, files=files)))
    path = write(tmp_path / "data" / "Inbox" / "Book.epub")
    imp = types.SimpleNamespace(imported=types.SimpleNamespace(path=path, name="Book.epub", extension=".epub"))
    assert asyncio.run(intents.perform(ACTION_OPEN_IN_READER, imp)) == "abcdef123456"
    assert opened[-1][0] is None and opened[-1][1]["path"] == path and not navigated


@needs_flet
def test_shared_txt_opens_in_the_reader(tmp_path, lib):
    """Device report issue 2: a .txt shared / opened with Glossarion gets an enabled "Open in Reader"
    routed through ``LibraryFeature.open_shared_in_reader`` to ``/reader/<bid>`` with the Inbox copy."""
    from glossarion_mobile.ui.library.feature import LibraryFeature

    files, cache = bridge(tmp_path)
    intents = IntentRouter(files=files, handlers={ACTION_TRANSLATE_NEW_CHAT: lambda imp: None})
    app, navigated, notes = _app(tmp_path, files, intents)
    feature = asyncio.run(LibraryFeature.install(app, service=LibraryService(
        core=SharedCore({"library_core": lib}), prefs=app.prefs, files=files)))
    assert intents.handlers[ACTION_OPEN_IN_READER] == feature.open_shared_in_reader
    shared = write(cache / "shared" / "Story.txt", "첫 문단.\n\n둘째 문단.".encode("utf-8"))

    async def scenario():
        [imp] = await intents.handle([{"id": "t1", "kind": "file", "path": shared, "name": "Story.txt"}])
        assert imp.imported.extension == ".txt"
        assert {a.id: a for a in imp.actions}[ACTION_OPEN_IN_READER].disabled_reason is None
        bid = await intents.perform(ACTION_OPEN_IN_READER, imp)
        assert bid and navigated[-1][0] == "reader" and navigated[-1][1] == {"bid": bid}
        assert feature.service.book_for_bid(bid)["path"] == os.path.abspath(imp.imported.path)
        return imp

    imp = asyncio.run(scenario())
    from glossarion_mobile.ui.reader.session import SOURCE_TXT, plan_for_file

    plan = plan_for_file(imp.imported.path)  # what the Reader opens for a path-only target
    assert plan.source_kind == SOURCE_TXT and plan.mode == "plain" and plan.title == "Story"
