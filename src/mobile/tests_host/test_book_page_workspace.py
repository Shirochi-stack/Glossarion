"""Device report issue 5: tapping a Completed book shows its EPUB details (Book page, UI_SPEC §3.10).

Two kinds of Completed books used to open a stripped Book page, because every workspace action keyed
off the card's ``output_folder``:

* an **organized** book (Organize moved the compiled EPUB to Library/Translated; its workspace stays in
  Output/<book>): ⋯ Compile EPUB / PDF, Edit metadata.json and Files were disabled, the metadata editor
  said "No output workspace", the Output tab saw no workspace, and the Chapters tab derived (and could
  create) ``Output/<raw stem>``;
* an **"Add translation" EPUB** (no workspace at all): Chapters and At a glance showed "This book has no
  output workspace yet" although ``load_book_details`` had already read the EPUB's chapters.

The Book page now resolves the workspace at each use (``LibraryService.workspace_for``, the desktop
``_resolve_book_output_folder``; the row and its id stay the card's) and lists the EPUB's own chapters
when there is no workspace. ⋯ › ↗ Share is ``LibraryContext.share_books`` (the card tap no longer shares).

Real data is never touched: GLOSSARION_LIBRARY_DIR, OUTPUT_DIRECTORY, HOME, USERPROFILE, APPDATA and
LOCALAPPDATA point into ``tmp_path``; GLOSSARION_HTTP_LOG=0.

Run from src/mobile:
    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_book_page_workspace.py
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
SRC_DIR = MOBILE_DIR.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

# The Library host-test fixtures (EPUB builder, FakePrefs, the fake Flet session, the context helper).
_TLU_SPEC = importlib.util.spec_from_file_location("_glossarion_tlu_helpers_book_workspace",
                                                   Path(__file__).with_name("test_library_ui.py"))
TLU = importlib.util.module_from_spec(_TLU_SPEC)
_TLU_SPEC.loader.exec_module(TLU)

from glossarion_mobile.services.library import LibraryService  # noqa: E402
from glossarion_mobile.ui.library import progress_model as pm  # noqa: E402

needs_flet = TLU.needs_flet
NO_WORKSPACE = "This book has no output workspace yet"


def _norm(path) -> str:
    return os.path.normcase(os.path.normpath(os.path.abspath(str(path))))


def _tree(root: Path) -> list:
    """Every folder under ``root`` (relative), to prove nothing was created."""
    out = []
    for folder, dirs, _files in os.walk(root):
        for name in dirs:
            out.append(os.path.relpath(os.path.join(folder, name), root))
    return sorted(out)


class FakeFiles:
    def __init__(self) -> None:
        self.shared: list = []

    async def share(self, paths):
        self.shared.append([str(p) for p in paths])
        return True

    def export_options(self, path):
        return []


class FakeReader:
    """``ReaderFeature.open_book`` (the Book page opens the Reader in-process)."""

    def __init__(self) -> None:
        self.opened: list = []

    def open_book(self, book, **kwargs):
        self.opened.append((dict(book or {}), kwargs))
        return "rid"


class FakeJobs:
    def __init__(self) -> None:
        self.submitted: list = []

    def has_kind(self, kind):
        return kind in ("compile_epub", "compile_pdf", "translate", "metadata")

    def submit(self, spec):
        self.submitted.append(spec)
        return f"job{len(self.submitted)}"


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A fixture Library with an organized book and an "Add translation" EPUB (real shared cores)."""
    lc = TLU._core("library_core", "install_library_env", "scan_library", "LibraryShelf", "BookDetailsModel",
                   "resolve_book_output_folder", "load_book_details")
    if lc is None:
        pytest.skip("library_core (U5 API) not importable")
    if TLU._core("progress_core", "build_book_progress") is None:
        pytest.skip("progress_core not importable")
    home = tmp_path / "home"
    for sub in ("", "AppData/Roaming", "AppData/Local"):
        (home / sub).mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("APPDATA", str(home / "AppData" / "Roaming"))
    monkeypatch.setenv("LOCALAPPDATA", str(home / "AppData" / "Local"))
    monkeypatch.setenv("GLOSSARION_HTTP_LOG", "0")
    library = tmp_path / "Library"
    output = tmp_path / "Output"
    # Organized book: the raw's stem differs from the workspace name, so a derived Output/<raw stem>
    # would be a NEW folder (the regression this pins).
    raw = TLU.make_epub(library / "Raw" / "Fin Source.epub", TLU.CHAPTERS[:2], title="Fin Raw")
    ws = output / "Fin"
    ws.mkdir(parents=True)
    (ws / "source_epub.txt").write_text(raw, encoding="utf-8")
    for name in ("ch001", "ch002"):
        (ws / f"response_{name}.html").write_text(f"<html><head><title>T {name}</title></head><body>{name}</body>"
                                                  "</html>", encoding="utf-8")
    prog = {"version": "2.1", "chapters": {
        "1": {"actual_num": 1, "status": "completed", "output_file": "response_ch001.html",
              "original_basename": "ch001.xhtml", "content_hash": "a"},
        "2": {"actual_num": 2, "status": "completed", "output_file": "response_ch002.html",
              "original_basename": "ch002.xhtml", "content_hash": "b"}}}
    (ws / "translation_progress.json").write_text(json.dumps(prog), encoding="utf-8")
    (ws / "metadata.json").write_text(json.dumps({"title": "Fin Meta", "creator": "Writer"}), encoding="utf-8")
    TLU.make_epub(ws / "Fin.epub", TLU.CHAPTERS[:2], title="Fin Translated")
    # "Add translation": a translated EPUB filed in the Library, no workspace anywhere
    shelf = TLU.make_epub(library / "Translated" / "Shelf.epub", TLU.CHAPTERS, title="Shelf Translated")
    monkeypatch.setenv("GLOSSARION_LIBRARY_DIR", str(library))
    monkeypatch.setenv("OUTPUT_DIRECTORY", str(output))
    paths = types.SimpleNamespace(library=library, output=output, cache=tmp_path / "cache")
    service = LibraryService(paths=paths, config={}, prefs=TLU.FakePrefs())
    service.ensure_env()
    try:
        yield {"service": service, "library": library, "output": output, "ws": ws, "raw": raw, "shelf": shelf,
               "tmp": tmp_path}
    finally:
        try:
            lc.uninstall_library_env()
        except Exception:
            pass


async def _organize(env) -> dict:
    """Desktop Organize (``LibraryShelf.execute_organize``): the compiled Fin.epub goes to Library/Translated;
    returns the Completed Library row."""
    service = env["service"]
    await service.refresh()
    plan = service.plan_organize_blocking()
    assert [os.path.basename(path) for _book, path in plan.get("translated_moves") or ()] == ["Fin.epub"]
    service.execute_organize_blocking(plan, "keep_both")
    await service.refresh()
    library_epub = env["library"] / "Translated" / "Fin.epub"
    row = next(b for b in service.snapshot.completed if _norm(b.get("path") or "") == _norm(library_epub))
    assert row.get("in_library") and not row.get("output_folder")
    return row


def _menu(screen) -> dict:
    return {str(item.content): item.disabled for item in screen.menu.items}


async def _open(env, bid: str, tab: str = "overview", **ctx_kwargs):
    from glossarion_mobile.ui.library.book_page import BookPageScreen
    from glossarion_mobile.ui.router import parse_route

    conn, session = TLU._TB._fake_session("android")
    page = session.page
    ctx = TLU._ctx(page, env["service"], **ctx_kwargs)
    screen = BookPageScreen(parse_route(f"/library/book/{bid}?tab={tab}"), ctx)
    screen.actions()
    TLU._mount(page, screen.get_body())
    await screen.load()
    page.update()
    return screen, ctx, page


# ==========================================================================
# An organized book keeps its workspace
# ==========================================================================


@needs_flet
def test_organized_book_keeps_every_workspace_action(env):
    service, ws, output = env["service"], env["ws"], env["output"]

    async def scenario():
        row = await _organize(env)
        assert _norm(service.workspace_for(row)) == _norm(ws)
        bid = service.bid_for(row)
        folders = _tree(output)
        files, jobs = FakeFiles(), FakeJobs()
        service.jobs = jobs
        screen, ctx, page = await _open(env, bid, "chapters", files=files, jobs=jobs)
        # the ⋯ menu: every workspace action is enabled, and ↗ Share is there
        menu = _menu(screen)
        for label in ("Compile EPUB", "Compile PDF", "Edit metadata.json", "Files", "↗ Share"):
            assert menu[label] is False, label
        # the page keeps the card's row and id
        assert screen.bid == bid and service.bid_for(screen.book) == bid
        assert not screen.book.get("output_folder") and _norm(screen.workspace) == _norm(ws)
        # Chapters: the workspace's Progress Manager rows
        chapters = screen.chapters
        assert not screen.progress.no_workspace and _norm(screen.progress.output_dir) == _norm(ws)
        assert [r.status for r in chapters.visible if r.kind == "chapter"] == ["completed", "completed"]
        assert NO_WORKSPACE not in TLU._texts(chapters.root)
        # Overview: ✏️ Edit, Compile and Files are live
        overview = screen.overview
        assert not overview.edit_button.disabled
        icons = {c.key: c.disabled for c in overview.icon_row.controls}
        assert icons["ov-compile"] is False and icons["ov-files"] is False
        # Compile writes into the resolved workspace
        assert await screen.compile("compile_epub") == "job1"
        assert [_norm(p) for p in jobs.submitted[-1].inputs] == [_norm(ws)]
        assert jobs.submitted[-1].origin["bid"] == bid
        # Files opens the workspace
        screen.open_files()
        assert ctx.navigated[-1] == ("tools.files.folder", {"root": "output", "fid": service.prefs.file_ref(str(ws))},
                                     None)
        # Output tab: the workspace (size, groups) and the Library EPUB itself
        await screen.output.reload()
        page.update()
        assert screen.output.size is not None and "metadata" in screen.output.groups
        assert _norm(env["library"] / "Translated" / "Fin.epub") in {_norm(p) for p, _k in screen.output.outputs}
        assert all(not b.disabled for b in screen.output.compile_row.controls)
        # a fresh Library scan keeps the workspace actions (the row stays the card's)
        await service.refresh()
        screen._on_library(service.snapshot)
        assert all(_menu(screen)[label] is False for label in ("Compile EPUB", "Compile PDF", "Edit metadata.json",
                                                                 "Files"))
        assert screen.bid == bid and _norm(screen.workspace) == _norm(ws)
        # opening the page created no folder (no derived Output/<raw stem>)
        assert _tree(output) == folders and not (output / "Fin Source").exists()
        screen.dispose()

    asyncio.run(scenario())


@needs_flet
def test_metadata_editor_of_an_organized_book_edits_the_workspace_metadata(env):
    from glossarion_mobile.ui.library.metadata_editor import MetadataEditorScreen
    from glossarion_mobile.ui.router import parse_route

    service, ws = env["service"], env["ws"]

    async def scenario():
        row = await _organize(env)
        bid = service.bid_for(row)
        conn, session = TLU._TB._fake_session("android")
        page = session.page
        ctx = TLU._ctx(page, service)
        editor = MetadataEditorScreen(parse_route(f"/library/book/{bid}/metadata"), ctx)
        editor.actions()
        body = editor.get_body()
        TLU._mount(page, body)
        assert getattr(body, "key", None) != "meta-none" and "No output workspace" not in TLU._texts(body)
        await editor.load()
        page.update()
        assert json.loads(editor.json_text)["title"] == "Fin Meta"  # <workspace>/metadata.json
        assert editor.fields["title"].value == "Fin Meta" and editor.fields["creator"].value == "Writer"
        editor.fields["title"].value = "Fin Edited"
        editor._on_change()
        assert await editor.save() is not None
        saved = json.loads((ws / "metadata.json").read_text(encoding="utf-8"))
        assert saved["title"] == "Fin Edited" and saved["title_translated"] is True
        # the metadata.json segment writes the same file
        editor.mode_buttons.selected = ["json"]
        editor._on_mode()
        editor.json_field.value = json.dumps(dict(saved, publisher="House"))
        editor._on_change()
        await editor.save()
        assert json.loads((ws / "metadata.json").read_text(encoding="utf-8"))["publisher"] == "House"
        assert editor.bid == bid and not editor.book.get("output_folder")

    asyncio.run(scenario())


@needs_flet
def test_a_scan_that_resolves_the_workspace_later_enables_the_menu(env):
    """A deep-linked Book page opened before any scan (the row is synthesised from the route id) creates
    nothing, then picks up the organized book's workspace from the next Library scan: the ⋯ menu is rebuilt
    and the Chapters tab loads the workspace's rows."""
    service, output = env["service"], env["output"]

    async def scenario():
        row = await _organize(env)
        bid = service.bid_for(row)
        folders = _tree(output)
        fresh = LibraryService(paths=service.paths, config={}, prefs=service.prefs)  # no scan yet
        fresh.ensure_env()
        env_fresh = dict(env, service=fresh)
        screen, ctx, page = await _open(env_fresh, bid, "chapters")
        assert screen.book.get("synthesised") and screen.workspace == ""
        assert _menu(screen)["Compile EPUB"] is True and screen.progress.no_workspace
        assert _tree(output) == folders  # no Output/<raw stem> for the unscanned file row
        await fresh.refresh()
        screen._on_library(fresh.snapshot)
        assert _norm(screen.workspace) == _norm(env["ws"])
        assert all(_menu(screen)[label] is False for label in ("Compile EPUB", "Compile PDF", "Edit metadata.json",
                                                                 "Files"))
        for _ in range(300):  # _sync_workspace reloads the Chapters on the loop
            if screen.progress is not None and not screen.progress.no_workspace:
                break
            await asyncio.sleep(0.01)
        assert _norm(screen.progress.output_dir) == _norm(env["ws"])
        assert [r.status for r in screen.chapters.visible if r.kind == "chapter"] == ["completed", "completed"]
        assert screen.bid == bid and _tree(output) == folders
        screen.dispose()

    asyncio.run(scenario())


# ==========================================================================
# An "Add translation" EPUB lists its own chapters
# ==========================================================================


@needs_flet
def test_add_translation_epub_lists_its_own_chapters(env):
    service, output = env["service"], env["output"]

    async def scenario():
        await service.refresh()
        row = next(b for b in service.snapshot.completed if _norm(b.get("path") or "") == _norm(env["shelf"]))
        assert service.workspace_for(row) == ""
        bid = service.bid_for(row)
        folders = _tree(output)
        reader = FakeReader()
        screen, ctx, page = await _open(env, bid, "chapters", reader=lambda: reader, files=FakeFiles())
        assert screen.progress.no_workspace
        chapters = screen.chapters
        rows = list(chapters.visible)
        assert [r.kind for r in rows] == ["spine"] * 4
        assert [r.filename for r in rows] == TLU.CHAPTERS
        assert rows[0].title.startswith("Ch.001 · ") and rows[3].title.startswith("Ch.004 · ")
        texts = TLU._texts(chapters.root)
        assert NO_WORKSPACE not in texts
        assert "\U0001f4d6 Chapters in this EPUB · 4" in texts
        assert not chapters.folder_chip.visible and not chapters.banner.visible
        # At a glance: "4 chapters" instead of the error
        glance = TLU._texts(screen.overview.glance)
        assert any("4 chapters" in text for text in glance) and NO_WORKSPACE not in glance
        # a row tap opens the Reader at that chapter
        chapters.on_row_tap(rows[2])
        book, kwargs = reader.opened[-1]
        assert kwargs["chapter"] == 2 and kwargs["chapter_filename"] == "ch003.xhtml"
        assert _norm(book.get("path")) == _norm(env["shelf"])
        # no selection or bulk actions without a workspace
        chapters.on_row_long_press(rows[0])
        assert not chapters.selecting and not chapters.selected
        # the search filters the EPUB rows
        chapters.search.value = "ch004"
        chapters._on_search()
        assert [r.filename for r in chapters.visible] == ["ch004.xhtml"]
        chapters.search.value = ""
        chapters._on_search()
        # the raw-titles toggle rebuilds the rows (no progress reload)
        chapters.toggle_raw_titles()
        assert len(chapters.visible) == 4 and all(r.kind == "spine" for r in chapters.visible)
        chapters.toggle_raw_titles()
        page.update()  # the EPUB rows serialise
        # the workspace actions stay off; Share does not
        menu = _menu(screen)
        assert menu["Compile EPUB"] is True and menu["Files"] is True and menu["↗ Share"] is False
        # the page created no workspace
        assert _tree(output) == folders and not (output / "Shelf").exists()
        screen.dispose()

    asyncio.run(scenario())


def test_library_file_without_workspace_never_builds_a_progress_view(tmp_path):
    """``load_progress_view`` does not open (and create) a Progress Manager workspace for a Library-filed
    book without one, even when a raw source resolves for it; a workspace row still opens normally."""
    built: list = []

    class Core:
        STATUS_GROUPS = None

        @staticmethod
        def ProgressOwner(config, save_config=None):  # noqa: N802 - the shared class name
            return types.SimpleNamespace(config=config, take_notices=lambda: [])

        @staticmethod
        def build_book_progress(source, config, **kwargs):
            built.append((source, kwargs.get("fixed_output_dir")))
            return None

    class Service:
        core = types.SimpleNamespace(module=lambda name: Core if name == "progress_core" else None)

        @staticmethod
        def config_snapshot():
            return {}

        @staticmethod
        def save_owner_config(config):
            return []

        @staticmethod
        def workspace_for(book):
            return ""

        @staticmethod
        def raw_source(book):
            return str(tmp_path / "Raw" / "Shelf.epub")

    library_row = {"name": "Shelf", "path": str(tmp_path / "Translated" / "Shelf.epub"), "type": "epub",
                   "in_library": True, "is_in_progress": False}
    view = pm.load_progress_view(Service(), library_row)
    assert view.no_workspace and view.error == NO_WORKSPACE and built == []
    # a translation workspace row is still opened through the Progress Manager core
    workspace_row = {"name": "Book", "type": "in_progress", "is_in_progress": True}
    pm.load_progress_view(Service(), workspace_row)
    assert built == [(str(tmp_path / "Raw" / "Shelf.epub"), None)]


def test_book_workspace_and_workspace_row(tmp_path):
    ws = tmp_path / "Output" / "Fin"
    ws.mkdir(parents=True)
    row = {"name": "Fin", "path": str(tmp_path / "Library" / "Translated" / "Fin.epub"), "in_library": True}
    service = types.SimpleNamespace(workspace_for=lambda book: str(ws) if book.get("name") == "Fin" else "")
    assert pm.book_workspace(service, row) == str(ws)
    copy = pm.workspace_row(service, row)
    assert copy["output_folder"] == str(ws) and "output_folder" not in row  # the card's row is unchanged
    # a service without the resolver (host fakes) reads the row
    assert pm.book_workspace(types.SimpleNamespace(), {"output_folder": "/out/Book"}) == "/out/Book"
    assert pm.workspace_row(types.SimpleNamespace(), {"name": "x"}) == {"name": "x"}


def test_job_for_book_matches_the_resolved_workspace(tmp_path):
    from glossarion_mobile.ui.library.book_page import job_for_book

    ws = str(tmp_path / "Output" / "Fin")
    snap = types.SimpleNamespace(spec=types.SimpleNamespace(origin={"bid": "other"}, inputs=()), output_dir=ws,
                                 output_dirs={})
    view = types.SimpleNamespace(active=snap, queue=())
    organized = {"name": "Fin", "path": str(tmp_path / "Library" / "Translated" / "Fin.epub")}
    assert job_for_book(view, "bid", organized) is None
    assert job_for_book(view, "bid", organized, folder=ws) is snap


# ==========================================================================
# ⋯ › ↗ Share
# ==========================================================================


@needs_flet
def test_book_page_share_sends_the_book_through_the_library_share(env):
    service = env["service"]

    async def scenario():
        await service.refresh()
        row = next(b for b in service.snapshot.completed if _norm(b.get("path") or "") == _norm(env["shelf"]))
        files = FakeFiles()
        screen, ctx, page = await _open(env, service.bid_for(row), files=files)
        item = next(i for i in screen.menu.items if str(i.content) == "↗ Share")
        assert not item.disabled
        calls: list = []
        share_books = ctx.share_books

        async def recording(books):
            calls.append([dict(b) for b in books])
            return await share_books(books)

        ctx.share_books = recording
        assert await screen.share() is True
        assert len(calls) == 1 and _norm(calls[0][0]["path"]) == _norm(env["shelf"])
        assert [[_norm(p) for p in paths] for paths in files.shared] == [[_norm(env["shelf"])]]
        screen.dispose()

    asyncio.run(scenario())
