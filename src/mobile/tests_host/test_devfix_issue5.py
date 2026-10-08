"""Acceptance test for the owner's device report #5 on the U8 APK:

    "Clicking on a complete library entry should behave the same as in progress. It should show my
    epub details."

What the owner saw on the phone: a finished TXT book (and a PDF without a workspace) opened the Android
share sheet on tap instead of the Book page, and a Completed EPUB filed in Library/Translated (Organize,
"Add translation") opened a stripped Book page: no chapters ("This book has no output workspace yet"),
Compile / Edit metadata.json / Files disabled.

This test drives the REAL app (``main.main`` on the fake Flet session of tests_host, the real
LibraryService over the real shared ``library_core`` / ``progress_core`` / ``glossary_progress_core``)
like the phone does: drawer › Library › "Completed (N)" › tap the card (the click event goes through the
Flet session to the card's handler) › the Book page. For every kind of Completed book

* a compiled EPUB workspace (Output/<book>/<book>.epub),
* a finished TXT workspace (Output/<book>/<book>_translated.txt; the shelf shows it as a TXT card),
* a finished PDF workspace (Output/<book>/<book>_translated.pdf) and a Library PDF without a workspace,
* an organized EPUB (Organize moved it to Library/Translated; its workspace stays in Output/<book>),
* an "Add translation" EPUB with no raw and no workspace,

the tap opens ``/library/book/<bid>`` (never the share sheet, never another app), and the page shows the
book: Overview (title, glance), Chapters (the workspace's Progress Manager rows, or the EPUB's own
spine), Glossary (the glossary progress when one exists), Output (the compiled file / the Library file and
the workspace). The workspace is the one ``library_core.resolve_book_output_folder`` gives (the desktop
``_resolve_book_output_folder``), so ⋯ Compile EPUB / PDF, Edit metadata.json and Files are enabled
exactly when a workspace exists; ⋯ › ↗ Share is always there and does open the share sheet. An In-progress
card is tapped too, for the "behave the same" comparison.

Real data is never touched: FLET_APP_STORAGE_* (so HOME, OUTPUT_DIRECTORY, GLOSSARION_LIBRARY_DIR,
CONFIG_FILE) come from ``app_env`` under ``tmp_path``; USERPROFILE, APPDATA and LOCALAPPDATA are pointed
there too, and GLOSSARION_HTTP_LOG=0.

Run from src/mobile:
    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_devfix_issue5.py
"""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import json
import os
import sys
import zipfile
from pathlib import Path

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = MOBILE_DIR / "app"
SRC_DIR = MOBILE_DIR.parent
TESTS_DIR = MOBILE_DIR / "tests"
for _path in (APP_DIR, TESTS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))


def _has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


pytestmark = pytest.mark.skipif(not (_has("flet") and _has("msgpack")), reason="flet / msgpack not installed")

_TB_SPEC = importlib.util.spec_from_file_location("_glossarion_tb_helpers_devfix5",
                                                  Path(__file__).with_name("test_bootstrap.py"))
_TB = importlib.util.module_from_spec(_TB_SPEC)
_TB_SPEC.loader.exec_module(_TB)
storage = _TB.storage
app_env = _TB.app_env

NO_WORKSPACE = "This book has no output workspace yet"
WORKSPACE_ITEMS = ("Compile EPUB", "Compile PDF", "Edit metadata.json", "Files")
SHARE_ITEM = "↗ Share"
#: Flet invoke methods that would put another app in front of Glossarion (share sheet, viewer, browser)
EXTERNAL_CALLS = ("share_files", "share_text", "share_uri", "launch_url", "open_file", "open_external")
CHAPTERS = ["ch001.xhtml", "ch002.xhtml", "ch003.xhtml", "ch004.xhtml"]


def _foundations():
    spec = importlib.util.spec_from_file_location("_glossarion_uf_helpers_devfix5",
                                                  Path(__file__).with_name("test_ui_foundations.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _core(name: str, *attrs: str):
    if not _has(name):
        return None
    try:
        module = importlib.import_module(name)
    except Exception:
        return None
    return module if all(hasattr(module, a) for a in attrs) else None


def _norm(path) -> str:
    return os.path.normcase(os.path.normpath(os.path.abspath(str(path))))


# ==========================================================================
# Fixture books (real files the shared scanners read)
# ==========================================================================


def make_epub(path: Path, chapters, title: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("mimetype", "application/epub+zip")
        zf.writestr("META-INF/container.xml",
                    '<?xml version="1.0"?><container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:'
                    'container"><rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-'
                    'package+xml"/></rootfiles></container>')
        manifest = "".join(f'<item id="c{i}" href="{n}" media-type="application/xhtml+xml"/>'
                           for i, n in enumerate(chapters))
        spine = "".join(f'<itemref idref="c{i}"/>' for i in range(len(chapters)))
        zf.writestr("OEBPS/content.opf",
                    '<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf" version="3.0"><metadata '
                    f'xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>{title}</dc:title><dc:creator>Author'
                    '</dc:creator><dc:language>ko</dc:language><dc:subject>Fantasy</dc:subject></metadata>'
                    f'<manifest>{manifest}</manifest><spine>{spine}</spine></package>')
        for n in chapters:
            zf.writestr(f"OEBPS/{n}", f"<html><head><title>{n}</title></head><body><h1>{n}</h1><p>text of {n}</p>"
                        "</body></html>")
    return str(path)


def make_pdf(path: Path, pages: int = 2, label: str = "Page") -> str:
    """A real PDF (PyMuPDF when installed, else a minimal hand-written one)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import fitz  # PyMuPDF

        doc = fitz.open()
        for index in range(pages):
            page = doc.new_page()
            page.insert_text((72, 72), f"{label} {index + 1}: some text on this page.")
        doc.save(str(path))
        doc.close()
        return str(path)
    except Exception:
        pass
    objects = ["<</Type/Catalog/Pages 2 0 R>>", "<</Type/Pages/Kids[3 0 R]/Count 1>>",
               "<</Type/Page/Parent 2 0 R/MediaBox[0 0 300 300]/Contents 4 0 R"
               "/Resources<</Font<</F1 5 0 R>>>>>>"]
    stream = f"BT /F1 12 Tf 20 150 Td ({label} 1) Tj ET"
    objects.append(f"<</Length {len(stream)}>>stream\n{stream}\nendstream")
    objects.append("<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>")
    out = b"%PDF-1.4\n"
    offsets = []
    for number, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n{body}\nendobj\n".encode("latin-1")
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode("latin-1")
    out += "".join(f"{o:010d} 00000 n \n" for o in offsets).encode("latin-1")
    out += f"trailer<</Size {len(objects) + 1}/Root 1 0 R>>\nstartxref\n{xref}\n%%EOF\n".encode("latin-1")
    path.write_bytes(out)
    return str(path)


def _progress(entries: dict) -> str:
    return json.dumps({"version": "2.1", "chapters": entries, "chapter_chunks": {}})


def _epub_workspace(output: Path, library: Path, name: str, *, raw_stem: str = "", chapters=CHAPTERS,
                    done=None, glossary: bool = False) -> Path:
    """Output/<name>: the raw in Library/Raw, response files, translation_progress.json (``done`` chapters
    completed; all by default), metadata.json and, when everything is done, the compiled <name>.epub."""
    raw_stem = raw_stem or name
    raw = make_epub(library / "Raw" / f"{raw_stem}.epub", chapters, title=f"{name} Raw")
    ws = output / name
    ws.mkdir(parents=True)
    (ws / "source_epub.txt").write_text(raw, encoding="utf-8")
    done = len(chapters) if done is None else done
    entries = {}
    for index, filename in enumerate(chapters[:done], 1):
        stem = os.path.splitext(filename)[0]
        (ws / f"response_{stem}.html").write_text(
            f"<html><head><title>{name} {stem}</title></head><body><p>{name} {stem}</p></body></html>",
            encoding="utf-8")
        entries[str(index)] = {"actual_num": index, "status": "completed", "output_file": f"response_{stem}.html",
                               "original_basename": filename, "content_hash": f"{name}-{index}"}
    (ws / "translation_progress.json").write_text(_progress(entries), encoding="utf-8")
    (ws / "metadata.json").write_text(json.dumps({"title": name, "creator": f"{name} Writer"}), encoding="utf-8")
    if done == len(chapters):
        make_epub(ws / f"{name}.epub", chapters, title=f"{name} Translated")
    if glossary:
        _glossary_progress(output, raw_stem, chapters)
    return ws


def _glossary_progress(output: Path, raw_stem: str, chapters) -> Path:
    gdir = output / "Glossary" / raw_stem
    gdir.mkdir(parents=True, exist_ok=True)
    data = {"progress_schema_version": "2.2", "indexing": "chapter_index_zero_based", "chapter_count": len(chapters),
            "chapters": {str(i): {"chapter_index": i, "status": "completed", "model_name": "gpt-x"}
                         for i in range(len(chapters))},
            "completed": list(range(len(chapters))), "failed": [],
            "chapter_filenames": {str(i): n for i, n in enumerate(chapters)},
            "minimal_pass": {"status": "completed", "model_name": "gpt-x", "entry_count": 7}, "book_title": raw_stem}
    path = gdir / f"{raw_stem}_glossary_progress.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _txt_workspace(output: Path, library: Path, name: str, sections: int = 3) -> Path:
    """A finished TXT workspace the way txt_processor + the translator leave it (word_count/ chunks,
    response_section_*.txt, progress entries numbered 1.0, 1.1, ...) plus the compiled <name>_translated.txt."""
    raw = library / "Raw" / f"{name}.txt"
    raw.parent.mkdir(parents=True, exist_ok=True)
    ws = output / name
    (ws / "word_count").mkdir(parents=True)
    entries, bodies, translated = {}, [], []
    for k in range(sections):
        body = f"원문 {k + 1}번째 조각입니다.\n\n둘째 문단 {k + 1}."
        bodies.append(body)
        (ws / "word_count" / f"section_1_{k}.txt").write_text(body, encoding="utf-8")
        text = f"Translated section {k + 1}.\n\nSecond paragraph {k + 1}."
        translated.append(text)
        (ws / f"response_section_1_{k}.txt").write_text(text, encoding="utf-8")
        num = round(1 + k * 0.1, 1)
        entries[str(num)] = {"actual_num": num, "output_file": f"response_section_1_{k}.txt", "status": "completed",
                             "content_hash": f"{name}-{k}"}
    raw.write_text("\n\n".join(bodies), encoding="utf-8")
    (ws / "source_epub.txt").write_text(str(raw), encoding="utf-8")
    (ws / "translation_progress.json").write_text(_progress(entries), encoding="utf-8")
    (ws / "metadata.json").write_text(json.dumps({"title": name}), encoding="utf-8")
    (ws / f"{name}_translated.txt").write_text("\n\n".join(translated), encoding="utf-8")
    return ws


def _pdf_workspace(output: Path, library: Path, name: str, pages: int = 2) -> Path:
    raw = make_pdf(library / "Raw" / f"{name}.pdf", pages, label=f"{name} raw")
    ws = output / name
    ws.mkdir(parents=True)
    (ws / "source_epub.txt").write_text(raw, encoding="utf-8")
    entries = {}
    for index in range(1, pages + 1):
        out = f"response_page_{index:03d}.html"
        (ws / out).write_text(f"<html><body><p>{name} page {index}</p></body></html>", encoding="utf-8")
        entries[str(index)] = {"actual_num": index, "status": "completed", "output_file": out,
                               "original_basename": f"page_{index:03d}.html", "content_hash": f"{name}-{index}"}
    (ws / "translation_progress.json").write_text(_progress(entries), encoding="utf-8")
    (ws / "metadata.json").write_text(json.dumps({"title": name}), encoding="utf-8")
    make_pdf(ws / f"{name}_translated.pdf", pages, label=f"{name} translated")
    return ws


def seed_library(paths) -> dict:
    """Every Completed kind of the owner's report (plus one In-progress book) in the app's own folders."""
    library, output = Path(paths.library), Path(paths.output)
    seeds = {
        "epub_ws": _epub_workspace(output, library, "Alpha", glossary=True),
        "txt_ws": _txt_workspace(output, library, "Bravo"),
        "pdf_ws": _pdf_workspace(output, library, "Charlie"),
        "pdf_lib": Path(make_pdf(library / "Translated" / "Delta.pdf", 3, label="Delta")),
        # organized below (LibraryService.execute_organize_blocking); the raw's stem differs from the
        # workspace name, so a derived Output/<raw stem> would be a NEW folder
        "organized": _epub_workspace(output, library, "Echo", raw_stem="Echo Source", chapters=CHAPTERS[:3],
                                     glossary=True),
        "add_translation": Path(make_epub(library / "Translated" / "Foxtrot.epub", CHAPTERS, title="Foxtrot")),
        "in_progress": _epub_workspace(output, library, "Golf", done=2),
    }
    return seeds


# ==========================================================================
# Driving the running app
# ==========================================================================


class ShareSpy:
    """Wraps the app's FileBridge.share (the call still goes through to Flet's Share service)."""

    def __init__(self, files) -> None:
        self.calls: list = []
        self._real = files.share
        files.share = self

    async def __call__(self, paths, **kwargs):
        self.calls.append([str(p) for p in paths])
        return await self._real(paths, **kwargs)


def _external_calls(conn) -> list:
    return [name for name in conn.invoked() if name in EXTERNAL_CALLS]


def _texts(control, out=None) -> list:
    out = [] if out is None else out
    if control is None:
        return out
    from host_tester import _children, _texts as node_texts

    if getattr(control, "visible", True) is False:
        return out
    out.extend(node_texts(control))
    for child in _children(control):
        _texts(child, out)
    return out


def _page_workspace(book_page) -> str:
    """The workspace the Book page works on (``BookPageScreen.workspace``; the row's own folder on a build
    without it, so this test reports the owner-visible failure there instead of an AttributeError)."""
    workspace = getattr(book_page, "workspace", None)
    return str(book_page.book.get("output_folder") or "") if workspace is None else str(workspace)


def _menu(book_page) -> dict:
    return {str(item.content): bool(item.disabled) for item in book_page.menu.items}


def _stack_routes(app) -> list:
    return [entry.route for entry in app.shell.stack]


async def _fresh_scan(service):
    """A rescan that really ran: ``LibraryService.refresh`` returns None (and scans nothing) while
    another scan runs, e.g. the one Library home starts when it opens."""
    for _ in range(600):
        if not service.scanning and await service.refresh(reason="test") is not None:
            return service.snapshot
        await asyncio.sleep(0.05)
    raise AssertionError("the Library never finished scanning")


async def _back_once(app, session, page) -> None:
    views = list(page.views or [])
    if len(views) > 1:
        await session.dispatch_event(page._i, "view_pop", {"route": views[-1].route})
    else:
        app.shell._on_tablet_back()
    await asyncio.sleep(0.05)


async def _back_to_library(app, session, page, uf) -> None:
    """System back until the Library home is on top: a phone pops the top View (``view_pop``, what the
    client sends); a wide window shows the screens in one View, so back is the shell's own back."""
    from glossarion_mobile.ui.library.home import LibraryScreen

    for _ in range(6):
        if isinstance(app.shell.top_screen, LibraryScreen):
            return
        await _back_once(app, session, page)
    assert isinstance(app.shell.top_screen, LibraryScreen), _stack_routes(app)


async def _tap_card(app, driver, tester, conn, page, uf, row, shelf_label: str):
    """Library › shelf › tap the card (a click event through the Flet session); returns the Book page."""
    from glossarion_mobile.ui.library.book_page import BookPageScreen

    service = app.library
    bid = service.bid_for(row)
    await driver.tap(contains=shelf_label, timeout=30)
    finder = await driver.wait(key=f"book-{bid}", timeout=30)
    external_before = len(_external_calls(conn))
    share_before = len(app.files.share.calls)
    await tester.tap(finder.first)
    assert await uf._wait(lambda: isinstance(app.shell.top_screen, BookPageScreen)
                          and app.shell.top_screen.bid == bid, timeout=15), (row.get("name"), _stack_routes(app))
    # the tap opened the Book page (/library/book/<bid> over the Library): no share sheet, no other app
    assert _stack_routes(app)[-2:] == ["/library", f"/library/book/{bid}"], _stack_routes(app)
    if not app.shell.tablet:
        assert uf._routes(page)[-2:] == ["/library", f"/library/book/{bid}"], uf._routes(page)
    assert _external_calls(conn)[external_before:] == [], _external_calls(conn)
    assert app.files.share.calls[share_before:] == []
    book_page = app.shell.top_screen
    assert await uf._wait(lambda: book_page.details_phase == "full" and book_page.progress is not None, timeout=30)
    await asyncio.sleep(0.2)  # the scan-driven _sync_workspace / chapter reload settle
    return book_page


async def _select_tab(driver, book_page, uf, name: str, label: str) -> None:
    await driver.tap(text=label, timeout=15)
    assert await uf._wait(lambda: book_page.current_tab == name, timeout=10), book_page.current_tab


async def _check_book_page(app, driver, tester, conn, page, uf, row, expect: dict) -> dict:
    """The Book page of ``row``: tabs populated, workspace actions, ↗ Share. Returns what it saw."""
    lc = importlib.import_module("library_core")
    book_page = await _tap_card(app, driver, tester, conn, page, uf, row, expect["shelf"])
    service = app.library
    seen: dict = {"name": row.get("name"), "type": row.get("type"), "wide": book_page.wide}

    # ---- the workspace: the row's own folder, else library_core.resolve_book_output_folder -------------
    resolved = lc.resolve_book_output_folder(dict(row)) or ""
    if resolved and not os.path.isfile(os.path.join(resolved, "translation_progress.json")):
        resolved = ""
    workspace = expect.get("workspace")
    assert _norm(resolved) == _norm(workspace) if workspace else resolved == "", (row.get("name"), resolved)
    page_workspace = _page_workspace(book_page)
    assert (_norm(page_workspace) == _norm(workspace)) if workspace else page_workspace == "", \
        (row.get("name"), page_workspace)
    assert book_page.bid == service.bid_for(row) and service.bid_for(book_page.book) == book_page.bid

    # ---- ⋯ menu ---------------------------------------------------------------------------------------
    menu = _menu(book_page)
    seen["menu"] = menu
    for label in WORKSPACE_ITEMS:
        assert menu[label] is (not workspace), (row.get("name"), label, menu)
    assert menu[SHARE_ITEM] is False, (row.get("name"), menu)

    # ---- Overview -------------------------------------------------------------------------------------
    overview = book_page.overview
    assert overview.hero.visible and not overview.skeleton.visible
    assert overview.title_text.value, row.get("name")
    assert not overview.error_text.visible, overview.error_text.value
    glance = _texts(overview.glance)
    seen["glance"] = glance
    assert glance, row.get("name")
    if expect.get("chapter_rows") or expect.get("spine_rows"):
        assert NO_WORKSPACE not in glance, (row.get("name"), glance)
    if expect.get("spine_rows"):
        assert any(f"{expect['spine_rows']} chapters" in text for text in glance), glance
    assert overview.edit_button.disabled is (not workspace)
    icons = {c.key: bool(c.disabled) for c in overview.icon_row.controls}
    assert icons["ov-compile"] is (not workspace) and icons["ov-files"] is (not workspace)
    assert icons["ov-share"] is False

    # ---- Overview › Start reading: the in-app Reader (never the share sheet / another app) --------------
    from glossarion_mobile.ui.reader.reader_view import ReaderScreen

    external_before = len(_external_calls(conn))
    share_before = len(app.files.share.calls)
    seen["read_label"] = str(overview.read_button.content)
    if expect.get("read") == "disabled":
        # the Reader cannot show it: the card menus' rule (no Reader item, the reason names ↗ Share)
        assert overview.read_button.disabled and "↗ Share" in str(overview.read_button.tooltip or ""), (
            row.get("name"), overview.read_button.tooltip)
        seen["reader"] = ("disabled", None, None, [])
    else:
        assert not overview.read_button.disabled, row.get("name")
        await driver.tap(key="ov-read", timeout=10)
        assert await uf._wait(lambda: isinstance(app.shell.top_screen, ReaderScreen), timeout=10), _stack_routes(app)
        reader = app.shell.top_screen
        assert await uf._wait(lambda: reader.state in ("ready", "error"), timeout=30), reader.state
        plan = getattr(getattr(reader, "session", None), "plan", None)
        seen["reader"] = (reader.state, getattr(plan, "mode", None), getattr(plan, "source_kind", None),
                          _texts(reader.loading)[:3] if reader.state == "error" else [])
        assert reader.state == expect.get("read", "ready"), (row.get("name"), seen["reader"])
        await _back_once(app, conn.session, page)
        assert await uf._wait(lambda: app.shell.top_screen is book_page, timeout=10), _stack_routes(app)
    assert _external_calls(conn)[external_before:] == [] and app.files.share.calls[share_before:] == []

    # ---- Chapters -------------------------------------------------------------------------------------
    await _select_tab(driver, book_page, uf, "chapters", "Chapters")
    chapters = book_page.chapters
    rows = list(chapters.visible)
    seen["chapters"] = [(r.kind, getattr(r, "status", None)) for r in rows]
    chapter_texts = _texts(chapters.root)
    seen["chapter_texts"] = chapter_texts[:12]
    details = book_page.details or {}
    seen["details"] = {"chapters_info": None if details.get("chapters_info") is None else len(details["chapters_info"]),
                       "error": details.get("error"), "keys": sorted(details)[:20]}
    seen["chapters_state"] = {"total": chapters.total_text.value, "banner": bool(chapters.banner.visible),
                              "banner_texts": _texts(chapters.banner),
                              "holder": type(getattr(chapters.list_holder, "content", None)).__name__,
                              "holder_key": getattr(getattr(chapters.list_holder, "content", None), "key", None),
                              "spine_mode": chapters.spine_mode}
    if expect.get("chapter_rows"):
        assert not book_page.progress.no_workspace
        assert _norm(book_page.progress.output_dir) == _norm(workspace)
        # chapter rows (EPUB / PDF spine) or the progress-only rows of a TXT workspace ("fallback"), as the
        # desktop Progress Manager lists them; the special-file rows (metadata, artifacts) are not counted
        statuses = [r.status for r in rows if r.kind in ("chapter", "fallback")]
        assert len(statuses) == expect["chapter_rows"] and set(statuses) <= set(expect.get("statuses", ("completed",))), \
            (row.get("name"), seen["chapters"])
        assert NO_WORKSPACE not in chapter_texts
    elif expect.get("spine_rows"):
        assert book_page.progress.no_workspace
        assert [r.kind for r in rows] == ["spine"] * expect["spine_rows"], seen["chapters"]
        assert f"\U0001f4d6 Chapters in this EPUB · {expect['spine_rows']}" in chapter_texts
        assert NO_WORKSPACE not in chapter_texts
    else:
        # nothing to list (a PDF has no EPUB spine and no workspace): the tab says so instead of loading forever
        assert book_page.progress.no_workspace and rows == []
        assert chapters.banner.visible and NO_WORKSPACE in _texts(chapters.banner), seen["chapters_state"]
        assert "Loading chapters…" not in chapter_texts
    # ---- Glossary -------------------------------------------------------------------------------------
    await _select_tab(driver, book_page, uf, "glossary", "Glossary")
    assert await uf._wait(lambda: book_page.glossary is not None, timeout=20)
    glossary = book_page.glossary
    seen["glossary"] = (glossary.path, len(glossary.rows), glossary.error, glossary.empty)
    assert not glossary.error, (row.get("name"), glossary.error)
    if expect.get("glossary_rows"):
        assert glossary.path and len(glossary.rows) >= expect["glossary_rows"], seen["glossary"]
    # ---- Output ---------------------------------------------------------------------------------------
    await _select_tab(driver, book_page, uf, "output", "Output")
    output = book_page.output
    assert await uf._wait(lambda: not output.stale, timeout=20)
    outputs = {_norm(p) for p, _kind in output.outputs}
    seen["outputs"] = sorted(os.path.basename(p) for p, _kind in output.outputs)
    for path in expect.get("outputs", ()):
        assert _norm(path) in outputs, (row.get("name"), seen["outputs"])
    if workspace:
        assert output.size is not None and all(not b.disabled for b in output.compile_row.controls)
    else:
        assert output.size is None and all(b.disabled for b in output.compile_row.controls)
    seen["output_texts"] = _texts(output.outputs_column)
    # Compile from this page targets the resolved workspace (the job spec, not started here)
    if workspace:
        spec = service.compile_spec(book_page.book, "compile_epub", folder=_page_workspace(book_page) or None)
        assert [_norm(p) for p in spec.inputs] == [_norm(workspace)] and spec.origin["bid"] == book_page.bid

    # ---- ⋯ › ↗ Share does open the share sheet (with this book's file) -----------------------------------
    external_before = len(_external_calls(conn))
    share_before = len(app.files.share.calls)
    await driver.tap(text=SHARE_ITEM, timeout=10)
    assert await uf._wait(lambda: len(app.files.share.calls) > share_before, timeout=10)
    shared = app.files.share.calls[-1]
    seen["shared"] = [os.path.basename(p) for p in shared]
    assert [_norm(p) for p in shared] == [_norm(expect["share"])], (row.get("name"), shared)
    assert await uf._wait(lambda: "share_files" in _external_calls(conn)[external_before:], timeout=10)
    book_page_bid = book_page.bid
    await _back_to_library(app, conn.session, page, uf)
    seen["bid"] = book_page_bid
    return seen


def _row(rows, path) -> dict:
    found = [b for b in rows if _norm(b.get("path") or "") == _norm(path)]
    assert len(found) == 1, (os.path.basename(str(path)), [(b.get("name"), b.get("type"), os.path.basename(
        str(b.get("path") or "")), bool(b.get("in_library"))) for b in rows])
    return found[0]


def _isolate(monkeypatch, tmp_path) -> None:
    home = tmp_path / "winhome"
    for sub in ("", "AppData/Roaming", "AppData/Local"):
        (home / sub).mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("APPDATA", str(home / "AppData" / "Roaming"))
    monkeypatch.setenv("LOCALAPPDATA", str(home / "AppData" / "Local"))
    monkeypatch.setenv("GLOSSARION_HTTP_LOG", "0")


def _folders(root: Path) -> list:
    out = []
    for folder, dirs, _files in os.walk(root):
        out.extend(os.path.relpath(os.path.join(folder, d), root) for d in dirs)
    return sorted(out)


# ==========================================================================
# The owner's scenario
# ==========================================================================


@pytest.mark.parametrize("width", [412, 1280], ids=["phone", "wide"])
def test_tapping_every_completed_book_opens_its_book_page(app_env, tmp_path, monkeypatch, width):
    """Phone (412 dp) and a wide tablet (1280 dp: the Book page is master-detail, the Overview in its own
    pane beside Chapters · Glossary · Output)."""
    lc = _core("library_core", "install_library_env", "uninstall_library_env", "current_library_env",
               "resolve_book_output_folder", "scan_library", "load_book_details", "BookDetailsModel")
    if lc is None:
        pytest.skip("library_core (U5 API) not importable")
    if _core("progress_core", "build_book_progress") is None or _core("glossary_progress_core") is None:
        pytest.skip("progress cores not importable")
    _isolate(monkeypatch, tmp_path)
    uf = _foundations()
    import flows
    import ui_driver
    from host_tester import PyTester
    from ui_driver import UiDriver

    from glossarion_mobile.ui.library.book_page import BookPageScreen
    from glossarion_mobile.ui.library.home import LibraryScreen

    report: dict = {}

    async def scenario():
        _m, conn, session, page, app = await uf._start("android", width=width)
        try:
            paths = app.paths
            # the app's Library / Output are inside this test's storage (never the developer's)
            assert _norm(tmp_path) in _norm(paths.library) and _norm(tmp_path) in _norm(paths.output)
            assert await uf._wait(lambda: lc.current_library_env() is not None, timeout=15)
            seeds = seed_library(paths)
            service = app.library
            share = ShareSpy(app.files)
            # The shared scanners list only EPUBs in Library/Translated, so no real scan yields a Completed
            # PDF card without a workspace - the case the old tap rule sent to the share sheet
            # (home._opens_in_another_app). The scanner reports one here, after the real scan's rows.
            library_pdf = seeds["pdf_lib"]
            real_scan = lc.scan_library

            def scan_with_a_library_pdf(config=None):
                in_progress_rows, completed_rows = real_scan(config)
                if library_pdf.is_file():
                    stat = library_pdf.stat()
                    completed_rows = list(completed_rows) + [{
                        "name": "Delta", "path": str(library_pdf), "type": "pdf", "in_library": True,
                        "size": stat.st_size, "mtime": stat.st_mtime, "translation_state": "completed",
                        "is_in_progress": False, "raw_source_path": "", "missing_raw_file": False}]
                return in_progress_rows, completed_rows

            monkeypatch.setattr(lc, "scan_library", scan_with_a_library_pdf)
            # Organize the finished Echo workspace into Library/Translated (desktop Organize; only that book)
            await _fresh_scan(service)
            echo_ws = seeds["organized"]
            echo_row = next(b for b in service.snapshot.all_books() if _norm(b.get("output_folder") or "") == _norm(echo_ws))
            plan = service.plan_organize_blocking([echo_row])
            assert [os.path.basename(p) for _b, p in plan.get("translated_moves") or ()] == ["Echo.epub"], plan
            service.execute_organize_blocking(plan, "keep_both")
            organized_epub = Path(paths.library) / "Translated" / "Echo.epub"
            assert organized_epub.is_file() and not (echo_ws / "Echo.epub").exists() and echo_ws.is_dir()

            # drawer › Library, like the owner
            tester = PyTester(session, page)
            driver = UiDriver(tester, poll_ms=50, log=lambda *_a: None)
            if not await driver.exists(key="dest-library", timeout=0.5):  # a wide window keeps the rail open
                await flows.open_drawer(driver)
            await driver.tap(key="dest-library", timeout=30)
            assert await uf._wait(lambda: isinstance(app.shell.top_screen, LibraryScreen), timeout=15)
            home = app.shell.top_screen
            assert await uf._wait(lambda: home.snapshot.scanned_at and home.cards, timeout=30)
            await _fresh_scan(service)
            output_folders = _folders(Path(paths.output))
            completed = list(service.snapshot.completed)
            in_progress = list(service.snapshot.in_progress)
            output = Path(paths.output)

            alpha = _row(completed, seeds["epub_ws"] / "Alpha.epub")
            bravo = _row(completed, seeds["txt_ws"] / "Bravo_translated.txt")
            charlie = _row(completed, seeds["pdf_ws"] / "Charlie_translated.pdf")
            delta = _row(completed, library_pdf)
            echo = _row(completed, organized_epub)
            foxtrot = _row(completed, seeds["add_translation"])
            golf = _row(in_progress, seeds["in_progress"])
            # the shelf kinds the owner's phone showed
            assert (alpha["type"], bravo["type"], charlie["type"], delta["type"], echo["type"], foxtrot["type"]) == \
                ("epub", "txt", "pdf", "pdf", "epub", "epub")
            assert not delta.get("output_folder")
            assert echo.get("in_library") and not echo.get("output_folder")
            assert foxtrot.get("in_library") and not foxtrot.get("output_folder")

            completed_label = "Completed ("
            cases = [
                ("compiled EPUB workspace", alpha,
                 {"shelf": completed_label, "workspace": seeds["epub_ws"], "chapter_rows": 4, "glossary_rows": 1,
                  "outputs": [seeds["epub_ws"] / "Alpha.epub"], "share": seeds["epub_ws"] / "Alpha.epub"}),
                ("finished TXT workspace (_translated.txt)", bravo,
                 {"shelf": completed_label, "workspace": seeds["txt_ws"], "chapter_rows": 3,
                  "outputs": [seeds["txt_ws"] / "Bravo_translated.txt"],
                  "share": seeds["txt_ws"] / "Bravo_translated.txt"}),
                ("finished PDF workspace", charlie,
                 {"shelf": completed_label, "workspace": seeds["pdf_ws"], "chapter_rows": 2,
                  "outputs": [seeds["pdf_ws"] / "Charlie_translated.pdf"],
                  "share": seeds["pdf_ws"] / "Charlie_translated.pdf"}),
                # the Reader cannot show a PDF without a workspace: Start reading is disabled with the
                # card menus' reason (the file goes to another app only through ⋯ › ↗ Share)
                ("Library PDF without a workspace", delta,
                 {"shelf": completed_label, "workspace": None, "share": seeds["pdf_lib"], "read": "disabled",
                  "outputs": [seeds["pdf_lib"]]}),
                ("organized Library/Translated EPUB", echo,
                 {"shelf": completed_label, "workspace": echo_ws, "chapter_rows": 3, "glossary_rows": 1,
                  "outputs": [organized_epub], "share": organized_epub}),
                ("Add-translation EPUB (no raw, no workspace)", foxtrot,
                 {"shelf": completed_label, "workspace": None, "spine_rows": 4,
                  "outputs": [seeds["add_translation"]], "share": seeds["add_translation"]}),
                ("In-progress EPUB workspace (comparison)", golf,
                 {"shelf": "In progress (", "workspace": seeds["in_progress"], "chapter_rows": 4,
                  "statuses": ("completed", "not_translated"),
                  "share": golf.get("raw_source_path") or golf.get("path")}),
            ]
            failures: list = []  # every kind is checked; the test fails at the end with all of them
            for label, row, expect in cases:
                try:
                    report[label] = await _check_book_page(app, driver, tester, conn, page, uf, row, expect)
                    # >= 1200 dp: master-detail (the Overview pane beside the tabs); a phone: the four tabs
                    assert report[label]["wide"] is (width >= 1200), report[label]["wide"]
                except (AssertionError, ui_driver.UiTimeout, AttributeError, KeyError) as exc:
                    failures.append(f"{label} [{row.get('type')}]: {type(exc).__name__}: {exc}")
                    try:
                        await _back_to_library(app, session, page, uf)
                    except AssertionError:
                        pass
            assert not failures, "\n".join(failures)

            # the card ⋯ sheet of every Completed card still offers ↗ Share (and Open Book Details)
            for row in (alpha, bravo, charlie, delta, echo, foxtrot):
                items = {item.label: item.disabled_reason for item in home.card_actions(row).items}
                assert items.get("↗ Share", "missing") is None and \
                    items.get("\U0001f4d1 Open Book Details", "missing") is None, (row.get("name"), items)

            # the organized book: ⋯ › ⟳ Refresh (a full reconcile + Library rescan) keeps the workspace actions
            book_page = await _tap_card(app, driver, tester, conn, page, uf, echo, completed_label)
            await driver.tap(text="⟳ Refresh", timeout=10)
            await asyncio.sleep(0.1)
            assert await uf._wait(lambda: not book_page._full_refreshing and not service.scanning, timeout=30)
            menu = _menu(book_page)
            assert all(menu[label] is False for label in WORKSPACE_ITEMS), menu
            assert _norm(_page_workspace(book_page)) == _norm(echo_ws) and not book_page.progress.no_workspace
            # ⋯ › Edit metadata.json edits <workspace>/metadata.json
            await driver.tap(text="Edit metadata.json", timeout=10)
            from glossarion_mobile.ui.library.metadata_editor import MetadataEditorScreen

            assert await uf._wait(lambda: isinstance(app.shell.top_screen, MetadataEditorScreen), timeout=10)
            editor = app.shell.top_screen
            assert await uf._wait(lambda: bool(getattr(editor, "json_text", "")), timeout=10)
            assert json.loads(editor.json_text)["title"] == "Echo"
            assert "No output workspace" not in _texts(editor.get_body())
            await _back_to_library(app, session, page, uf)
            # ⋯ › Files opens the workspace folder
            book_page = await _tap_card(app, driver, tester, conn, page, uf, echo, completed_label)
            await driver.tap(text="Files", timeout=10)
            assert await uf._wait(lambda: app.shell.stack[-1].match.name == "tools.files.folder", timeout=10)
            assert _norm(app.prefs.resolve_file_ref(app.shell.stack[-1].match.params.get("fid"))) == _norm(echo_ws)
            await _back_to_library(app, session, page, uf)
            assert isinstance(book_page, BookPageScreen)
            # opening the pages created no folder (no derived Output/<raw stem>)
            assert _folders(output) == output_folders and not (output / "Echo Source").exists()
            assert not (output / "Foxtrot").exists() and not (output / "Delta").exists()
            # the card taps never reached the share sheet: only the seven ⋯ › Share taps did
            assert len(share.calls) == len(cases)
            assert _external_calls(conn).count("share_files") == len(cases)
        finally:
            if os.environ.get("GLOSSARION_ACCEPT_DEBUG"):
                for key, value in report.items():
                    print("REPORT", key, json.dumps(value, ensure_ascii=False, default=str))
            try:
                app.jobs.close()
            except Exception:
                pass
            try:
                lc.uninstall_library_env()
            except Exception:
                pass
            await uf._stop(app)

    asyncio.run(scenario())
