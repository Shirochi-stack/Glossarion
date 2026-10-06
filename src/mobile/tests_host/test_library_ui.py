"""Host tests for the U5 Library (UI_SPEC §3): cards, filters, selection, Book page tabs, the service.

* BookCard strings for every ``translation_state`` (ribbons, pills, floored %, warnings, type
  badges, sizes) - from the local presentation and, when importable, compared with the shared
  ``library_core.card_progress_view`` / ``card_type_badge`` / ``card_size_text``.
* Density presets (all 11 desktop card sizes), filters (format / tri-state states / query) and
  sort through the shared query helpers.
* ``LibraryService`` over a fake ``library_core``: scan snapshot, route ids, the delete plan
  and execution through ``LibraryShelf``, ``on_job_finished`` (rescan + Android mirror), the
  2 s poller gating.
* The real shared cores on a fixture Library (isolated with ``install_library_env``): the
  scan, book details, the Chapters view (chunked chapter children, every status group),
  a Progress Manager action through ``mutate_progress``, the Glossary view with the pinned
  Minimal-pass and refinement rows. Skipped when a core is missing.
* Flet screens in the fake session (``page.update`` serialises every control): Library home
  (shelves, scan chip, selection bar labels, both delete confirmation levels, filter sheet
  persistence), Book page tabs, metadata editor, Scan for raw, TranslateSheet.

Run from src/mobile with the mobile venv:
    .venv/Scripts/python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_library_ui.py
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import sys
import types
import zipfile
from pathlib import Path

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = MOBILE_DIR / "app"
SRC_DIR = MOBILE_DIR.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from glossarion_mobile.services.library import (  # noqa: E402
    DeletePlan,
    DeleteTarget,
    LibraryService,
    Poller,
    ScanSnapshot,
    SharedCore,
    book_key,
)
from glossarion_mobile.ui.library import models  # noqa: E402
from glossarion_mobile.ui.library.models import (  # noqa: E402
    DENSITY_LABELS,
    DENSITY_ORDER,
    FilterState,
    build_card,
    card_progress,
    density_preset,
    effective_density,
    page_size_value,
    visible_books,
)


def _has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


needs_flet = pytest.mark.skipif(not (_has("flet") and _has("msgpack")), reason="flet / msgpack not installed")


def _core(name: str, *attrs: str):
    """The real shared module when it has every attribute, else None."""
    if not _has(name):
        return None
    try:
        module = importlib.import_module(name)
    except Exception:
        return None
    return module if all(hasattr(module, a) for a in attrs) else None


# ==========================================================================
# Fixtures
# ==========================================================================


def make_epub(path: Path, chapters, title="Raw Book") -> str:
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
            zf.writestr(f"OEBPS/{n}", f"<html><head><title>{n}</title></head><body><h1>{n}</h1><p>text</p>"
                        "</body></html>")
    return str(path)


CHAPTERS = ["ch001.xhtml", "ch002.xhtml", "ch003.xhtml", "ch004.xhtml"]


def make_workspace(tmp_path: Path) -> dict:
    """Library/Raw/Book.epub + Output/Book with completed / qa_failed / chunked pending / untranslated rows."""
    library = tmp_path / "Library"
    output = tmp_path / "Output"
    raw = make_epub(library / "Raw" / "Book.epub", CHAPTERS)
    ws = output / "Book"
    ws.mkdir(parents=True)
    (ws / "source_epub.txt").write_text(raw, encoding="utf-8")
    for name in ("ch001", "ch002"):
        (ws / f"response_{name}.html").write_text(f"<html><body>{name}</body></html>", encoding="utf-8")
    prog = {
        "version": "2.1",
        "chapters": {
            "1": {"actual_num": 1, "status": "completed", "output_file": "response_ch001.html",
                  "original_basename": "ch001.xhtml", "content_hash": "h1", "model_name": "gpt-x"},
            "2": {"actual_num": 2, "status": "qa_failed", "output_file": "response_ch002.html",
                  "original_basename": "ch002.xhtml", "content_hash": "h2", "qa_issues_found": ["missing_images"]},
            "3": {"actual_num": 3, "status": "pending", "output_file": "response_ch003.html",
                  "original_basename": "ch003.xhtml", "content_hash": "h3"},
        },
        "chapter_chunks": {
            "h3": {"schema_version": 2, "total": 2, "completed": [1],
                   "entries": {"1": {"index": 1, "status": "completed"}, "2": {"index": 2, "status": "pending"}},
                   "chunks": {"1": "<p>a</p>"}},
        },
    }
    (ws / "translation_progress.json").write_text(json.dumps(prog), encoding="utf-8")
    gdir = output / "Glossary" / "Book"
    gdir.mkdir(parents=True)
    gp = {"progress_schema_version": "2.2", "indexing": "chapter_index_zero_based", "chapter_count": 4,
          "chapters": {"0": {"chapter_index": 0, "status": "completed", "model_name": "gpt-x"}},
          "completed": [0], "failed": [2], "chapter_filenames": {str(i): n for i, n in enumerate(CHAPTERS)},
          "minimal_pass": {"status": "completed", "model_name": "gpt-x", "entry_count": 12}, "book_title": "Book"}
    (gdir / "Book_glossary_progress.json").write_text(json.dumps(gp), encoding="utf-8")
    return {"library": library, "output": output, "raw": raw, "ws": ws, "cache": tmp_path / "cache",
            "gp": gdir / "Book_glossary_progress.json"}


class FakePrefs:
    def __init__(self) -> None:
        self.refs: dict = {}
        self.data: dict = {}
        self.positions: dict = {}

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
        return self.positions.get(bid)


@pytest.fixture
def real_env(tmp_path, monkeypatch):
    """The real shared cores pinned to a fixture Library (the developer's Library is never read)."""
    lc = _core("library_core", "install_library_env", "scan_library", "LibraryShelf")
    if lc is None:
        pytest.skip("library_core (U5 API) not importable")
    fixture = make_workspace(tmp_path)
    monkeypatch.setenv("GLOSSARION_LIBRARY_DIR", str(fixture["library"]))
    monkeypatch.setenv("OUTPUT_DIRECTORY", str(fixture["output"]))
    paths = types.SimpleNamespace(library=fixture["library"], output=fixture["output"], cache=fixture["cache"])
    service = LibraryService(paths=paths, config={}, prefs=FakePrefs())
    service.ensure_env()
    fixture["service"] = service
    try:
        yield fixture
    finally:
        try:
            lc.uninstall_library_env()
        except Exception:
            pass


# ==========================================================================
# Cards
# ==========================================================================


def _book(state, done=0, total=0, **extra):
    book = {"name": "Book", "type": "in_progress", "workspace_kind": "epub", "size": 1536 * 1024,
            "translation_state": state, "is_in_progress": state != "completed", "completed_chapters": done,
            "total_chapters": total, "output_folder": "/out/Book", "path": "/out/Book"}
    book.update(extra)
    return book


CARD_CASES = [
    # state, done, total, pill, pct text, ribbon
    ("not_started", 0, 12, "\U0001f195 Not started", "", "NOT STARTED"),
    ("in_progress", 12, 48, "⏳ 12/48", "25%", "IN PROGRESS"),
    ("in_progress", 216, 217, "⏳ 216/217", "99%", "IN PROGRESS"),
    ("in_progress", 0, 0, "⏳ In progress", "", "IN PROGRESS"),
    ("ready_to_compile", 48, 48, "✨ Ready to compile (48/48)", "", "READY TO COMPILE"),
    ("ready_to_compile", 0, 0, "✨ Ready to compile", "", "READY TO COMPILE"),
    ("outdated_progress", 3, 10, "⚠ Outdated Progress file", "", "OUTDATED PROGRESS"),
]


@pytest.mark.parametrize("state,done,total,pill,pct,ribbon", CARD_CASES)
def test_card_strings_for_every_translation_state(state, done, total, pill, pct, ribbon):
    book = _book(state, done, total)
    progress = card_progress(book)
    assert (progress.pill_text, progress.pct_text, progress.ribbon_text) == (pill, pct, ribbon)
    card = build_card(book, key="k", bid="abcdef012345")
    assert (card.pill_text, card.pct_text, card.ribbon_text) == (pill, pct, ribbon)
    assert card.type_label == "EPUB" and card.type_emoji == "\U0001f4d5" and card.size_text == "1.5 MB"
    if state == "in_progress" and total:
        assert card.progress == pytest.approx(done / total)
    lc = _core("library_core", "card_progress_view")
    if lc is not None:  # the shared presentation says exactly the same
        view = lc.card_progress_view(book)
        assert view["pill_text"] == pill and view["ribbon_text"] == ribbon
        assert (view["pct_text"] if view["show_pct"] else "") == pct
        shared = build_card(book, key="k", bid="abcdef012345", view=view)
        assert (shared.pill_text, shared.pct_text, shared.ribbon_text) == (pill, pct, ribbon)


def test_completed_and_compiling_cards():
    done = _book("completed", 10, 10, is_in_progress=False, type="epub", size=900 * 1024)
    assert card_progress(done) is None
    card = build_card(done, key="k", bid="abcdef012345")
    assert card.pill_text is None and card.ribbon_text is None and card.size_text == "900 KB"
    compiling = build_card(_book("ready_to_compile", 4, 4), key="k", bid="b", compiling=True)
    assert compiling.ribbon_text == "⚙ COMPILING…" and compiling.ribbon_state == "compiling"
    assert compiling.pill_text == "✨ Ready to compile (4/4)"


def test_card_warnings_badges_and_type_kinds():
    book = _book("in_progress", 1, 4, missing_raw_file=True, failed_chapters=3,
                 compiled_conflicts=[("a.epub", "epub"), ("b_translated.html", "html")])
    card = build_card(book, key="k", bid="b")
    texts = [w.text for w in card.warnings]
    assert texts == ["⚠ missing raw", "⚠ +2", "QA ⚠ 3"]
    assert "a.epub (EPUB)" in card.warnings[1].tooltip and "b_translated.html (HTML)" in card.warnings[1].tooltip
    kinds = {"pdf": ("\U0001f4c4", "PDF"), "txt": ("\U0001f4d7", "TXT"), "html": ("\U0001f310", "HTML"),
             "image": ("\U0001f5bc️", "IMG"), "other": ("\U0001f4c1", "FOLDER")}
    for kind, (emoji, label) in kinds.items():
        card = build_card(_book("not_started", workspace_kind=kind), key="k", bid="b")
        assert (card.type_emoji, card.type_label) == (emoji, label), kind
    lc = _core("library_core", "card_type_badge", "card_size_text")
    if lc is not None:
        for kind, (emoji, label) in kinds.items():
            assert lc.card_type_badge(_book("not_started", workspace_kind=kind))[0] == emoji + label
        assert lc.card_size_text(1536 * 1024) == "1.5 MB" and lc.card_size_text(900 * 1024) == "900 KB"
        shared = build_card(book, key="k", bid="b", badge_text="\U0001f4d5EPUB", size_label="1.5 MB")
        assert shared.type_label == "\U0001f4d5EPUB" and shared.type_emoji == ""


def test_density_presets_cover_every_desktop_size():
    spec = {"2xs": 78, "xs": 92, "compact": 110, "normal": 140, "large": 180, "xl": 230, "2xl": 290, "3xl": 360,
            "4xl": 440, "5xl": 530, "6xl": 630}
    assert list(DENSITY_ORDER) == list(spec) and [DENSITY_LABELS[k] for k in DENSITY_ORDER] == [
        "2XS", "XS", "S", "M", "L", "XL", "2XL", "3XL", "4XL", "5XL", "6XL"]
    for key, card_w in spec.items():
        assert density_preset(key)[0] == card_w
    lc = _core("library_core", "_SIZE_PRESETS")
    if lc is not None:
        for key, card_w in spec.items():
            assert density_preset(key, lc._SIZE_PRESETS) == (card_w, lc._SIZE_PRESETS[key]["cover_h"])
    assert effective_density("normal", 1.6) == "large" and effective_density("6xl", 2.0) == "6xl"
    assert effective_density("bogus") == "compact"
    assert page_size_value("all") == 0 and page_size_value(50) == 50 and page_size_value(None) == 20


def test_filters_sort_and_tristate_states():
    books = [
        _book("in_progress", 1, 4, name="Alpha", mtime=3, size=10),
        _book("ready_to_compile", 4, 4, name="beta", mtime=1, size=30, missing_raw_file=True),
        _book("not_started", name="Gamma", mtime=2, size=20, workspace_kind="txt", failed_chapters=2),
    ]
    matches = lambda b, q: q.casefold() in b["name"].casefold()  # noqa: E731
    format_of = lambda b: b.get("workspace_kind") or b.get("type")  # noqa: E731
    sort = lambda bs, mode, rev: sorted(bs, key=lambda b: b["name"].lower(), reverse=rev)  # noqa: E731
    state = FilterState()
    assert [b["name"] for b in visible_books(books, state, matches=matches, format_of=format_of, sort=sort)] == [
        "Alpha", "beta", "Gamma"]
    state.fmt = "txt"
    assert [b["name"] for b in visible_books(books, state, matches=matches, format_of=format_of, sort=sort)] == [
        "Gamma"]
    state = FilterState(states={"missing_raw": True})
    assert [b["name"] for b in visible_books(books, state, matches=matches, format_of=format_of, sort=sort)] == [
        "beta"]
    state = FilterState(states={"qa": False, "ready_to_compile": False}, reverse=True)
    assert [b["name"] for b in visible_books(books, state, matches=matches, format_of=format_of, sort=sort)] == [
        "Alpha"]
    assert models.next_tristate(None) is True and models.next_tristate(True) is False and \
        models.next_tristate(False) is None
    lc = _core("library_core", "sort_books", "book_matches_query", "format_of_book")
    if lc is not None:  # the shared desktop rules through the service
        service = LibraryService(core=SharedCore({"library_core": lc}))
        assert [b["name"] for b in service.sort_books(books, "date")] == ["Alpha", "Gamma", "beta"]
        assert [b["name"] for b in service.sort_books(books, "size")] == ["beta", "Gamma", "Alpha"]
        assert [b["name"] for b in service.sort_books(books, "name", reverse=True)] == ["Gamma", "beta", "Alpha"]
        assert service.matches_query(books[1], "BETA") and not service.matches_query(books[1], "zzz")
        assert service.format_of(books[2]) == "txt"


# ==========================================================================
# Service (fake library_core)
# ==========================================================================


class FakeShelf:
    calls: list = []
    plan: dict = {}

    def __init__(self, in_progress, completed, config):
        self.in_progress, self.completed, self.config = in_progress, completed, config

    def counts(self):
        return {"raw_count": 1, "trans_count": 2, "raw_undo": 3, "trans_undo": 0, "missing_raw": 1}

    def plan_delete(self, books):
        FakeShelf.calls.append(("plan_delete", [b["name"] for b in books]))
        return FakeShelf.plan

    def execute_delete(self, plan, targets=None, on_progress=None):
        FakeShelf.calls.append(("execute_delete", [t[1] for t in targets or []]))
        if on_progress is not None:
            on_progress(1, len(targets or []), "x")
        return {"deleted": len(targets or []), "errors": [], "summary": f"Deleted {len(targets or [])} of "
                f"{len(targets or [])} items.", "unregistered": len(plan.get("unregister") or [])}

    def plan_clear_raw_link(self, books):
        return {"targets": [("ws", "raw", books[0], True)], "prompt": "Remove the saved raw-source pointer?"}

    def execute_clear_raw_link(self, plan):
        FakeShelf.calls.append(("clear_raw_link", len(plan["targets"])))
        return 1


def fake_core(tmp_path, rows=None):
    module = types.ModuleType("library_core")
    rows = rows if rows is not None else ([], [])
    module.scan_library = lambda config=None: (list(rows[0]), list(rows[1]))
    module.card_signature = lambda book: (book.get("name"), book.get("completed_chapters"),
                                          book.get("translation_state"))
    module.card_progress_view = lambda book: (
        None if card_progress(book) is None else {
            "state": card_progress(book).state, "pill_text": card_progress(book).pill_text,
            "ribbon_text": card_progress(book).ribbon_text, "pct_text": card_progress(book).pct_text or "0%",
            "show_pct": bool(card_progress(book).pct_text), "pct": card_progress(book).pct})
    module.LibraryShelf = FakeShelf
    module.summarize_folder_contents = lambda folder: ["    · 2 translated chapter HTML files",
                                                       "    · total on disk: 12 KB"]
    module.DELETE_KEYWORDS = ("halgakos", "delete")
    module.library_root_path = lambda: str(tmp_path / "Library")
    return module


def _rows(tmp_path):
    ws_a = tmp_path / "Output" / "Alpha"
    ws_b = tmp_path / "Output" / "Beta"
    for ws in (ws_a, ws_b):
        ws.mkdir(parents=True, exist_ok=True)
    in_progress = [
        _book("in_progress", 3, 10, name="Alpha", output_folder=str(ws_a), path=str(ws_a), mtime=2),
        _book("not_started", 0, 5, name="Beta", output_folder=str(ws_b), path=str(ws_b), mtime=1,
              missing_raw_file=True),
    ]
    epub = tmp_path / "Library" / "Translated" / "Done.epub"
    epub.parent.mkdir(parents=True, exist_ok=True)
    epub.write_bytes(b"PK")
    completed = [{"name": "Done", "type": "epub", "path": str(epub), "in_library": True, "size": 2048,
                  "mtime": 5, "translation_state": "completed", "is_in_progress": False}]
    return in_progress, completed


def test_service_scan_snapshot_ids_and_synthesised_rows(tmp_path):
    rows = _rows(tmp_path)
    prefs = FakePrefs()
    service = LibraryService(core=SharedCore({"library_core": fake_core(tmp_path, rows)}), prefs=prefs, config={})
    seen = []
    service.subscribe(seen.append)
    snap = asyncio.run(service.refresh())
    assert snap.ok and [b["name"] for b in snap.in_progress] == ["Alpha", "Beta"] and len(snap.completed) == 1
    assert seen == [snap] and not service.dirty
    assert snap.organize_count == 3 and snap.undo_count == 3 and snap.missing_raw == 1
    alpha = snap.in_progress[0]
    assert snap.views[book_key(alpha)]["pill_text"] == "⏳ 3/10"
    bid = service.bid_for(alpha)
    assert len(bid) == 12 and prefs.resolve_file_ref(bid) == os.path.abspath(alpha["output_folder"])
    assert service.book_for_bid(bid)["name"] == "Alpha"
    # a deep link to a workspace the scan has not seen yet resolves to a synthesised row
    other = tmp_path / "Output" / "Gamma"
    other.mkdir()
    gid = prefs.file_ref(str(other))
    synthesised = service.book_for_bid(gid)
    assert synthesised["output_folder"] == str(other) and synthesised["type"] == "in_progress"
    assert service.book_for_bid("ffffffffffff") is None
    # a scan that fails keeps the last shelves and reports the error
    service.core = SharedCore({"library_core": types.ModuleType("library_core")})
    failed = asyncio.run(service.refresh())
    assert failed.error and "scan_library" in failed.error and failed.in_progress == snap.in_progress


def test_delete_plan_and_execute_through_the_shared_shelf(tmp_path):
    rows = _rows(tmp_path)
    core = fake_core(tmp_path, rows)
    service = LibraryService(core=SharedCore({"library_core": core}), config={})
    asyncio.run(service.refresh())
    alpha, beta = rows[0]
    FakeShelf.calls = []
    FakeShelf.plan = {"targets": [("Alpha", alpha["output_folder"], True, alpha),
                                  ("Beta", beta["output_folder"], True, beta)],
                      "unregister": [], "needs_keyword": True, "simple_prompt": ""}
    plan = service.plan_delete_blocking([alpha, beta])
    assert isinstance(plan, DeletePlan) and plan.needs_keyword and plan.keywords == ("halgakos", "delete")
    assert [t.label for t in plan.targets] == ["Alpha", "Beta"] and plan.targets[0].contents[0].endswith(
        "translated chapter HTML files")
    progress = []
    report = service.execute_delete_blocking(plan, [beta["output_folder"]], lambda *a: progress.append(a))
    assert FakeShelf.calls == [("plan_delete", ["Alpha", "Beta"]), ("execute_delete", [beta["output_folder"]])]
    assert report.deleted == 1 and report.summary == "Deleted 1 of 1 items." and progress and service.dirty
    assert not service.deleting


def test_on_job_finished_rescans_and_mirrors_android_outputs(tmp_path, monkeypatch):
    pytest.importorskip("flet")
    rows = _rows(tmp_path)
    saved = []

    class Files:
        platform = "android"

        async def save_to_downloads(self, path):
            saved.append(path)
            return f"content://downloads/{len(saved)}"

    prefs = FakePrefs()
    prefs.data["mirror_outputs"] = True
    service = LibraryService(core=SharedCore({"library_core": fake_core(tmp_path, rows)}), prefs=prefs,
                             files=Files(), config={})
    service.subscribe(lambda snap: None)
    out = tmp_path / "Output" / "Alpha" / "Alpha.epub"
    out.write_bytes(b"PK")
    service.compiling.add(os.path.normcase(os.path.normpath(os.path.abspath(rows[0][0]["output_folder"]))))
    spec = types.SimpleNamespace(inputs=(rows[0][0]["output_folder"],), params={"folder": rows[0][0]["output_folder"]})
    done = types.SimpleNamespace(spec=spec, state=types.SimpleNamespace(value="DONE"), outputs=(str(out),))
    uris = asyncio.run(service.on_job_finished(done))
    assert uris == ["content://downloads/1"] and saved == [str(out)] and not service.compiling
    assert service.snapshot.scanned_at and not service.dirty  # a listener was mounted: rescanned at once
    failed = types.SimpleNamespace(spec=spec, state=types.SimpleNamespace(value="FAILED"), outputs=(str(out),))
    assert asyncio.run(service.on_job_finished(failed)) == [] and len(saved) == 1
    prefs.data["mirror_outputs"] = False
    assert asyncio.run(service.on_job_finished(done)) == [] and len(saved) == 1


def test_poller_ticks_only_while_visible_and_in_the_foreground():
    state = {"visible": False, "foreground": True, "ticks": 0, "sleeps": 0}

    async def tick():
        state["ticks"] += 1

    async def scenario():
        async def sleep(seconds):
            state["sleeps"] += 1
            if state["sleeps"] == 2:
                state["visible"] = True
            if state["sleeps"] == 4:
                state["foreground"] = False
            if state["sleeps"] >= 6:
                poller.stop()
            await asyncio.sleep(0)

        poller = Poller(tick, interval=2.0, visible=lambda: state["visible"], foreground=lambda: state["foreground"],
                        sleep=sleep)
        poller.start()
        for _ in range(50):
            await asyncio.sleep(0)
            if not poller.running:
                break
        assert state["ticks"] == 2  # sleeps 2 and 3: visible + foreground
        assert poller.poke() is None  # not visible/foreground: no immediate tick
        state["foreground"] = True
        await poller.run_once()
        assert state["ticks"] == 3

    asyncio.run(scenario())


# ==========================================================================
# Real shared cores on a fixture Library
# ==========================================================================


def test_real_library_scan_card_and_details(real_env):
    service = real_env["service"]
    snap = asyncio.run(service.refresh())
    assert snap.ok, snap.error
    assert [b["name"] for b in snap.in_progress] == ["Book"] and not snap.completed
    book = snap.in_progress[0]
    assert book["translation_state"] == "in_progress" and book["raw_source_path"] == real_env["raw"]
    view = snap.views[book_key(book)]
    card = build_card(book, key=book_key(book), bid=service.bid_for(book), view=view,
                      badge_text=service.card_badge(book)[0], size_label=service.card_badge(book)[1])
    assert card.ribbon_text == "IN PROGRESS" and card.pill_text.startswith("⏳ ")
    assert card.type_label == "\U0001f4d5EPUB"
    preview = service.load_details_blocking(book, "preview")
    assert preview["details"]["title"] == "Raw Book"
    full = service.load_details_blocking(book, "full")
    assert len(full["chapters_info"]) == 4
    model = service.details_model(book, full)
    assert model.title() == "Raw Book" and model.tags() == ["Fantasy"]
    from glossarion_mobile.ui.library.overview_tab import hero_values, primary_read_label, progress_strip_text

    hero = hero_values(book, full, model)
    assert hero["author"] == "Author" and hero["language"] == "ko"
    assert primary_read_label(book, full["chapters_info"])[0] == "\U0001f4d6  Read translated"
    strip = progress_strip_text(book, model)
    assert strip is None or strip.startswith("⏳  Translation in progress")
    # the service resolved the raw file through the shared resolver too
    assert service.raw_source(dict(book, raw_source_path="")) == real_env["raw"]


def test_real_progress_view_rows_chunks_and_status_groups(real_env):
    if _core("progress_core", "build_book_progress", "refresh_book_progress") is None:
        pytest.skip("progress_core (U5 API) not importable")
    from glossarion_mobile.ui.library import progress_model as pm

    service = real_env["service"]
    asyncio.run(service.refresh())
    book = service.snapshot.in_progress[0]
    view = pm.load_progress_view(service, book, show_special=False, show_model_info=True)
    assert view.error is None and view.state is not None
    chapters = [r for r in view.rows if r.kind == "chapter"]
    assert [r.title for r in chapters] == [f"Ch.00{i} · ch00{i}.xhtml" for i in range(1, 5)]
    assert [r.status for r in chapters] == ["completed", "qa_failed", "pending", "not_translated"]
    chunked = chapters[2]
    assert [c.title for c in chunked.children] == ["↳ Ch.003 · Chunk 1/2", "↳ Ch.003 · Chunk 2/2"]
    assert chunked.chunk_text.startswith("Chunks") and chapters[1].qa_lines == ("missing_images",)
    assert chapters[0].subtitle == "gpt-x" and chapters[0].icon == "✅" and chapters[0].label == "Completed"
    chips = {c.group: c for c in view.chips}
    assert chips["completed"].text == "✅ Completed 2" and chips["failed"].text == "❌ Failed 1"
    assert chips["missing"].text == "⬜ Not Translated 1" and chips["missing"].pinned
    assert not chips["merged"].visible and view.total_text.startswith("Total: ")
    assert view.groups["failed"] == ("failed", "qa_failed", "refine_failed")
    for group, members in view.groups.items():  # every status group filters to its own members
        statuses = {r.status for r in pm.filter_rows(view.rows, group, view.groups)}
        assert statuses <= set(members), group
    sig = pm.progress_signature(service, book, view)
    again = pm.load_progress_view(service, book, show_special=False, show_model_info=True, previous=view)
    assert again.state is not None and [r.key for r in again.rows] == [r.key for r in view.rows]
    assert pm.progress_signature(service, book, again) == sig


def test_real_progress_action_writes_through_mutate_progress(real_env):
    if _core("progress_actions", "plan_remove_qa_marks", "remove_qa_marks") is None:
        pytest.skip("progress_actions not importable")
    from glossarion_mobile.ui.library import progress_model as pm

    service = real_env["service"]
    asyncio.run(service.refresh())
    book = service.snapshot.in_progress[0]
    view = pm.load_progress_view(service, book)
    qa_row = next(r for r in view.rows if r.status == "qa_failed")
    done_row = next(r for r in view.rows if r.status == "completed")
    refused = pm.plan_action(service, view, "restore_in_progress", [done_row])
    assert refused.refusal == "None of the selected chapters have 'in_progress' status."
    plan = pm.plan_action(service, view, "remove_qa", [qa_row, done_row])
    assert plan.count == 1 and plan.refusal is None
    message = pm.apply_action(service, view, plan)
    assert message == "Removed failed mark from 1 chapters."
    prog = json.loads((real_env["ws"] / "translation_progress.json").read_text(encoding="utf-8"))
    assert prog["chapters"]["2"]["status"] == "completed" and "qa_issues_found" not in prog["chapters"]["2"]
    refreshed = pm.load_progress_view(service, book, previous=view, full=True)
    assert [r.status for r in refreshed.rows if r.kind == "chapter"][1] == "completed"


def test_real_glossary_view_pins_minimal_and_refinement_rows(real_env):
    if _core("glossary_progress_core", "open_glossary_progress", "glossary_rows") is None:
        pytest.skip("glossary_progress_core not importable")
    from glossarion_mobile.ui.library import progress_model as pm

    service = real_env["service"]
    asyncio.run(service.refresh())
    book = service.snapshot.in_progress[0]
    view = pm.load_glossary_view(service, book)
    assert view.path == str(real_env["gp"]) and not view.empty and not view.error
    kinds = [r.kind for r in view.rows]
    first_chapter = kinds.index("chapter")
    assert kinds[0] == "minimal" and all(k in ("minimal", "refinement") for k in kinds[:first_chapter])
    assert all(k == "chapter" for k in kinds[first_chapter:])
    minimal = view.rows[0]
    assert minimal.title == "Minimal Pass" and minimal.label == "Completed" and "12 entries" in minimal.subtitle
    refinement = [r for r in view.rows if r.kind == "refinement"]
    assert refinement and refinement[0].title.startswith("Refinement · ")
    chapters = [r for r in view.rows if r.kind == "chapter"]
    assert [r.status for r in chapters] == ["completed", "not_completed", "failed", "not_completed"]
    chips = {c.group: c for c in view.chips}
    assert chips["completed"].count == 2 and chips["failed"].count == 1 and chips["remaining"].count == 2
    assert not chips["not_refined"].visible and not chips["merged"].visible
    real_env["gp"].unlink()  # the file disappears: the deleted banner, last rows kept
    gone = pm.load_glossary_view(service, book, previous=view)
    assert gone.deleted and gone.rows == view.rows


# ==========================================================================
# Flet screens
# ==========================================================================


_TB_SPEC = importlib.util.spec_from_file_location("_glossarion_tb_helpers_library",
                                                  Path(__file__).with_name("test_bootstrap.py"))
_TB = importlib.util.module_from_spec(_TB_SPEC)
_TB_SPEC.loader.exec_module(_TB)


def _ctx(page, service, **kwargs):
    from glossarion_mobile.ui.library.common import LibraryContext

    navigated = []
    notes = []
    overlays = []
    ctx = LibraryContext(service=service, page=page,
                         navigate=lambda name, params=None, query=None: navigated.append((name, params, query)),
                         notify=lambda message, action=None, on_action=None: notes.append(message),
                         prefs=service.prefs, push_overlay=overlays.append, pop_overlay=lambda: overlays.pop(),
                         platform="android", **kwargs)
    ctx.navigated, ctx.notes, ctx.overlays = navigated, notes, overlays
    return ctx


def _mount(page, body):
    page.views[0].controls.append(body)
    page.update()


def _texts(control, out=None):
    """Every ``Text.value`` / string label under a control (dataclass walk)."""
    import flet as ft

    out = out if out is not None else []
    if control is None:
        return out
    if isinstance(control, ft.Text) and control.value:
        out.append(str(control.value))
    for name in ("content", "controls", "label", "leading", "title", "subtitle", "trailing", "tabs"):
        value = getattr(control, name, None)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, ft.BaseControl):
                    _texts(item, out)
        elif isinstance(value, ft.BaseControl):
            _texts(value, out)
        elif isinstance(value, str) and name in ("content", "label"):
            out.append(value)
    return out


@needs_flet
def test_library_home_shelves_selection_and_delete_levels(tmp_path):
    from glossarion_mobile.ui.library.delete_confirm import DeleteConfirmView
    from glossarion_mobile.ui.library.home import LibraryScreen
    from glossarion_mobile.ui.router import parse_route

    rows = _rows(tmp_path)
    config = {"epub_library_card_size": "normal", "epub_library_page_size": 20}
    service = LibraryService(core=SharedCore({"library_core": fake_core(tmp_path, rows)}), prefs=FakePrefs(),
                             config=config)

    async def scenario():
        conn, session = _TB._fake_session("android")
        page = session.page
        ctx = _ctx(page, service)
        screen = LibraryScreen(parse_route("/library"), ctx)
        actions = screen.actions()
        _mount(page, screen.get_body())
        screen.did_show()  # subscribes, starts the 2 s poller and runs the first full scan
        for _ in range(200):
            if screen.snapshot.scanned_at and screen.cards:
                break
            await asyncio.sleep(0.01)
        page.update()
        assert [s.label.value for s in screen.shelf_buttons.segments] == ["In progress (2)", "Completed (1)"]
        assert screen.scan_chip.visible and screen.scan_chip.label.value == "Scan for raw (1)"
        assert screen.menu_items["organize"].content == "Organize (3)" and screen.menu_items["undo"].content == "Undo (3)"
        assert list(screen.cards) == [book_key(b) for b in rows[0]] and screen.count_text.value == "2 novels"
        alpha_card = screen.cards[book_key(rows[0][0])]
        assert "⏳ 3/10" in _texts(alpha_card.control) and "IN PROGRESS" in _texts(alpha_card.control)
        assert "30%" in _texts(alpha_card.control)
        # tap opens the Book page; long-press enters selection with the bulk bar
        screen._on_card_tap(alpha_card.model)
        assert ctx.navigated[-1] == ("library.book", {"bid": alpha_card.model.bid}, None)
        screen._on_card_long_press(alpha_card.model)
        screen.toggle(book_key(rows[0][1]))
        page.update()
        assert screen.selection_bar.count_text.value == "2 selected" and screen.bulk_bar.control.visible
        labels = [a.label for a in screen.bulk_bar.actions]
        assert labels[:4] == ["Load 2 for translation", "Translate Metadata for 0 EPUBs", "Compile", "Delete 2"]
        assert "Delete glossary files (2)" in labels and "Restore glossary backup" in labels
        assert "Clear saved raw link (for 2 items)" in labels and "Share" in labels
        assert screen.bulk_bar.action("glossary_delete").disabled_reason
        # level 1: every target Not started -> a simple confirmation with the shared text
        beta = rows[0][1]
        FakeShelf.plan = {"targets": [("Beta", beta["output_folder"], True, beta)], "unregister": [],
                          "needs_keyword": False, "simple_prompt": "Permanently delete 1 Not Started item?"}
        dialog = await screen.delete_books([beta])
        assert dialog.dialog.content.content.controls[0].value == "Permanently delete 1 Not Started item?"
        await dialog._on_confirm()
        assert FakeShelf.calls[-1] == ("execute_delete", [beta["output_folder"]]) and not screen.selecting
        # level 2: the typed keyword unlocks Delete
        alpha = rows[0][0]
        FakeShelf.plan = {"targets": [("Alpha", alpha["output_folder"], True, alpha),
                                      ("Beta", beta["output_folder"], True, beta)],
                          "unregister": [], "needs_keyword": True, "simple_prompt": ""}
        view = await screen.delete_books([alpha, beta])
        assert isinstance(view, DeleteConfirmView) and ctx.overlays == [view.view]
        page.views.append(view.view)
        page.update()  # the full-screen confirmation serialises
        assert view.headline.value == "⚠  Permanent delete — 2 items"
        assert not view.can_delete
        view.field.value = "  HalGakos "
        view.checks[0].value = False
        view._sync()
        assert view.can_delete and view.selected_paths() == [beta["output_folder"]]
        view.field.value = "nope"
        assert not view.can_delete
        view.field.value = "delete"
        await view._on_delete()
        assert FakeShelf.calls[-1] == ("execute_delete", [beta["output_folder"]]) and ctx.overlays == []
        assert ctx.notes[-1] == "Deleted 1 of 1 items."
        # Completed shelf, list view, filters persisted in config (sparse desktop keys)
        screen.set_shelf("completed")
        assert config["epub_library_tab"] == 1 and screen.fab.content == "Add translation"
        screen.apply_filter_change("view", "list")
        screen.apply_filter_change("density", "xl")
        screen.apply_filter_change("sort", "name")
        screen.apply_filter_change("page_size", "all")
        page.update()
        assert config["epub_library_card_size"] == "xl" and config["epub_library_sort"] == "name"
        assert config["epub_library_page_size"] == "all" and service.prefs.data["library_view"] == "list"
        assert screen.count_text.value == "1 book"
        sheet = screen.open_filter_sheet()
        sheet._on_state("missing_raw")
        assert sheet.state.states == {"missing_raw": True} and "✓ Missing raw" in _texts(sheet.state_chips[
            "missing_raw"])
        assert actions and screen.dispose() is None

    asyncio.run(scenario())


@needs_flet
def test_book_page_tabs_render_progress_glossary_and_overview(real_env):
    if _core("progress_core", "build_book_progress") is None or _core("glossary_progress_core",
                                                                        "open_glossary_progress") is None:
        pytest.skip("progress cores not importable")
    from glossarion_mobile.ui.library.book_page import BookPageScreen
    from glossarion_mobile.ui.router import parse_route

    service = real_env["service"]

    async def scenario():
        conn, session = _TB._fake_session("android")
        page = session.page
        await service.refresh()
        book = service.snapshot.in_progress[0]
        bid = service.bid_for(book)
        ctx = _ctx(page, service)
        screen = BookPageScreen(parse_route(f"/library/book/{bid}?tab=chapters"), ctx)
        assert screen.title == "Book" and screen.initial_tab == "chapters"
        screen.actions()
        _mount(page, screen.get_body())
        await screen.load()
        await screen.reload_glossary()
        page.update()
        chapters = screen.chapters
        assert [r.status for r in chapters.visible if r.kind == "chapter"] == [
            "completed", "qa_failed", "pending", "not_translated"]
        assert screen.strip_mode.content.value == "Mode: Text" and screen.strip_bar.visible
        chunked = next(r for r in chapters.visible if r.children)
        chapters.toggle_expand(chunked.key)
        page.update()
        assert any("Chunk 2/2" in t for t in _texts(chapters.controls[chunked.key]))
        first = next(r for r in chapters.visible if r.kind == "chapter")
        chapters.set_titles([{"filename": "OEBPS/ch001.xhtml", "raw_title": "Raw One", "translated_title": "One"}])
        assert chapters.display_title(first) == "Ch.001 · One"
        chapters.toggle_raw_titles()
        assert chapters.display_title(first) == "Ch.001 · Raw One"
        assert service.config["epub_details_show_raw_titles"] is True
        chapters.toggle_raw_titles()
        chapters.set_filter("failed")
        assert [r.status for r in chapters.visible] == ["qa_failed"]
        chapters.set_filter(None)
        key = await chapters.jump_next("missing")
        assert key == next(r.key for r in chapters.visible if r.status == "not_translated")
        chapters.on_row_long_press(chapters.visible[0])
        primary, more = chapters.bulk_actions(chapters.selected_rows())
        assert [a.label for a in primary] == ["Retranslate", "Remove QA mark"]
        assert [a.id for a in more][-1] == "edit_translation" and more[-1].disabled_reason
        sheet = await chapters.show_row_sheet(next(r for r in chapters.visible if r.status == "qa_failed"))
        enabled = {i.label for i in sheet.items if i.disabled_reason is None}
        assert "\U0001f9f9 Remove QA Failed Mark" in enabled and "\U0001f4d6 Open in reader" in enabled
        chapters.exit_selection()
        chapters.on_row_tap(next(r for r in chapters.visible if r.status == "not_translated"))
        assert ctx.navigated[-1][0] == "reader" and ctx.navigated[-1][2]["mode"] == "original"
        glossary = screen.glossary_tab
        assert glossary.view is not None and glossary.visible_rows()[0].kind == "minimal"
        assert glossary.title_text.value == "\U0001f4d6 Book" and glossary.file_chip.visible
        overview = screen.overview
        assert overview.title_text.value == "Raw Book" and overview.read_button.content == "\U0001f4d6  Read translated"
        assert overview.strip.visible and overview.strip.content.value.startswith("⏳  Translation in progress")
        screen.set_tab("output")
        await screen.output.reload()
        page.update()
        assert screen.output.raw == real_env["raw"]
        screen.dispose()

    asyncio.run(scenario())


@needs_flet
def test_metadata_editor_and_scan_raw_screens(real_env):
    from glossarion_mobile.ui.library.metadata_editor import MetadataEditorScreen
    from glossarion_mobile.ui.library.scan_raw import ScanRawScreen, normalize_matches, status_line
    from glossarion_mobile.ui.router import parse_route

    service = real_env["service"]

    async def scenario():
        conn, session = _TB._fake_session("android")
        page = session.page
        await service.refresh()
        book = service.snapshot.in_progress[0]
        bid = service.bid_for(book)
        ctx = _ctx(page, service)
        editor = MetadataEditorScreen(parse_route(f"/library/book/{bid}/metadata"), ctx)
        editor.actions()
        _mount(page, editor.get_body())
        await editor.load()
        assert editor.fields["title"].value == "Raw Book" and editor.save_button.disabled
        editor.fields["title"].value = "Translated Title"
        editor._on_change()
        assert not editor.save_button.disabled and editor.edits() == {"title": "Translated Title"}
        saved = await editor.save()
        metadata = json.loads((real_env["ws"] / "metadata.json").read_text(encoding="utf-8"))
        assert metadata["title"] == "Translated Title" and metadata["title_translated"] is True
        assert metadata["original_title"] == "Raw Book" and saved is not None
        scan = ScanRawScreen(parse_route("/library/scan-raw"), ctx, inbox_dir=str(real_env["library"]))
        _mount(page, scan.get_body())
        matches = await scan.scan()
        assert matches == [] and scan.status.value == "ⓘ No missing-raw workspaces to pair."

    asyncio.run(scenario())
    rows = normalize_matches({"/o/A": {"book": {"name": "A"}, "path": "/r/A.epub", "ratio": 0.82, "accepted": True},
                              "/o/B": {"book": {"name": "B"}, "path": "", "ratio": 0, "accepted": False}})
    assert [(m.label, m.matched, m.accepted) for m in rows] == [("A", True, True), ("B", False, False)]
    assert status_line(rows, 5, "fuzzy", 70) == ("✔ 1 of 2 workspaces matched (5 candidate files scanned, "
                                                 "fuzzy ≥ 70%).")
    assert status_line(rows, 0, "exact", 70).startswith("⚠ No candidate files found in this folder (exact)")


@needs_flet
def test_translate_sheet_submits_a_library_translate_job(tmp_path):
    from glossarion_mobile.ui.library.translate_sheet import TranslateSheet

    raw = tmp_path / "Raw" / "Book.epub"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"PK")
    submitted = []

    class Jobs:
        def has_kind(self, kind):
            return kind == "translate"

        async def submit(self, spec):
            submitted.append(spec)
            return "job1"

    service = LibraryService(core=SharedCore({"library_core": types.ModuleType("library_core")}),
                             prefs=FakePrefs(), jobs=Jobs(), config={"model": "authgpt/gpt-6-luna",
                                                                     "output_language": "English"})
    book = _book("not_started", name="Book", raw_source_path=str(raw))

    async def scenario():
        conn, session = _TB._fake_session("android")
        ctx = _ctx(session.page, service)
        sheet = TranslateSheet(ctx, [book], [str(raw)])
        sheet.show(session.page)
        texts = _texts(sheet.sheet.content)
        assert "Model: authgpt/gpt-6-luna" in texts and "→ English" in texts
        assert sheet.review_switch.disabled and sheet.start_reason is None
        job = await sheet.start()
        assert job == "job1" and submitted[0].kind == "translate" and submitted[0].inputs == (str(raw),)
        assert submitted[0].origin["type"] == "library" and len(submitted[0].origin["bid"]) == 12
        missing = TranslateSheet(ctx, [book], [""])
        assert missing.start_reason == "No raw source file resolves for the selection" and missing.start_button.disabled

    asyncio.run(scenario())


# ==========================================================================
# The running app (real shell; Integrate adds the install call to app.py)
# ==========================================================================


storage = _TB.storage
app_env = _TB.app_env


@needs_flet
def test_library_feature_routes_in_the_running_app(app_env):
    lc = _core("library_core", "install_library_env", "scan_library")
    if lc is None:
        pytest.skip("library_core (U5 API) not importable")
    uf_spec = importlib.util.spec_from_file_location("_glossarion_uf_helpers_library",
                                                     Path(__file__).with_name("test_ui_foundations.py"))
    uf = importlib.util.module_from_spec(uf_spec)
    uf_spec.loader.exec_module(uf)
    from glossarion_mobile.ui.library.book_page import BookPageScreen
    from glossarion_mobile.ui.library.feature import LibraryFeature
    from glossarion_mobile.ui.library.home import LibraryScreen
    from glossarion_mobile.ui.library.scan_raw import ScanRawScreen

    async def scenario():
        _m, conn, session, page, app = await uf._start("android")
        try:
            paths = app.paths
            raw = make_epub(Path(paths.library) / "Raw" / "Novel.epub", CHAPTERS, title="Novel")
            ws = Path(paths.output) / "Novel"
            ws.mkdir(parents=True, exist_ok=True)
            (ws / "source_epub.txt").write_text(raw, encoding="utf-8")
            (ws / "translation_progress.json").write_text(json.dumps({"version": "2.1", "chapters": {},
                                                                      "chapter_chunks": {}}), encoding="utf-8")
            feature = await LibraryFeature.install(app)
            assert app.library is feature.service
            await app.navigate("/library")
            assert uf._routes(page)[-1] == "/library" and isinstance(app.shell.top_screen, LibraryScreen)
            home = app.shell.top_screen
            assert await uf._wait(lambda: bool(home.cards))
            book = feature.service.snapshot.in_progress[0]
            assert book["name"] == "Novel" and book["translation_state"] == "not_started"
            card = next(iter(home.cards.values()))
            assert card.model.pill_text == "\U0001f195 Not started" and card.model.ribbon_text == "NOT STARTED"
            home._on_card_tap(card.model)
            assert await uf._wait(lambda: isinstance(app.shell.top_screen, BookPageScreen))
            assert uf._routes(page)[-2:] == ["/library", f"/library/book/{card.model.bid}"]
            book_page = app.shell.top_screen
            assert await uf._wait(lambda: book_page.progress is not None and book_page.details is not None)
            assert book_page.overview.title_text.value == "Novel"
            await app.navigate("/library/scan-raw")
            assert isinstance(app.shell.top_screen, ScanRawScreen)
            await app.navigate(f"/tools/progress?out={card.model.bid}")
            assert isinstance(app.shell.top_screen, BookPageScreen)
            assert app.shell.top_screen.initial_tab == "chapters" and app.shell.top_screen.title == "Progress manager"
            assert {"library", "library.book", "library.scan_raw", "tools.progress"} <= set(feature.screens_built)
        finally:
            try:
                lc.uninstall_library_env()
            except Exception:
                pass
            await uf._stop(app)

    asyncio.run(scenario())


@needs_flet
def test_app_start_installs_the_library_and_the_reader(app_env):
    """Integrate (U5): GlossarionApp.start installs LibraryFeature then ReaderFeature, pins the Library
    folders, registers both IntentRouter actions and refreshes the Library when a job ends."""
    lc = _core("library_core", "install_library_env", "current_library_env", "uninstall_library_env")
    if lc is None:
        pytest.skip("library_core (U5 API) not importable")
    uf_spec = importlib.util.spec_from_file_location("_glossarion_uf_helpers_library_app",
                                                     Path(__file__).with_name("test_ui_foundations.py"))
    uf = importlib.util.module_from_spec(uf_spec)
    uf_spec.loader.exec_module(uf)
    from glossarion_mobile.services.intents import ACTION_ADD_TO_LIBRARY, ACTION_OPEN_IN_READER
    from glossarion_mobile.services.jobs import JobState
    from glossarion_mobile.ui.library.feature import LibraryFeature
    from glossarion_mobile.ui.library.home import LibraryScreen
    from glossarion_mobile.ui.reader.feature import ReaderFeature
    from glossarion_mobile.ui.screens.base import SHIPPED_MILESTONES

    async def scenario():
        _m, conn, session, page, app = await uf._start("android")
        try:
            feature = app.library_feature
            assert isinstance(feature, LibraryFeature) and app.library is feature.service
            assert isinstance(app.reader, ReaderFeature) and app.reader.library is app.library
            assert "U5" in SHIPPED_MILESTONES
            handlers = app.intents.handlers
            assert handlers[ACTION_OPEN_IN_READER] == feature.open_shared_in_reader
            assert handlers[ACTION_ADD_TO_LIBRARY] == feature.add_shared_to_library
            assert app.files.library_import == feature._library_import
            # the Library folders are pinned on the io pool right after the install
            assert await uf._wait(lambda: lc.current_library_env() is not None)
            assert os.path.normcase(lc.current_library_env().library_root) == os.path.normcase(str(app.paths.library))
            # a job that ends refreshes the Library (and mirrors outputs on Android) once
            finished = []

            async def on_job_finished(snap, outputs=None):
                finished.append(snap.id)
                return []

            feature.service.on_job_finished = on_job_finished
            snap = types.SimpleNamespace(id="job-1", is_terminal=True, state=JobState.DONE)
            app.job_service._deliver_transition(snap, JobState.RUNNING)
            app.job_service._deliver_transition(snap, JobState.RUNNING)
            assert await uf._wait(lambda: finished == ["job-1"])
            await app.navigate("/library")
            assert isinstance(app.shell.top_screen, LibraryScreen)
            await app.navigate("/tools")
            assert "tools.progress" in app.shell.top_screen.tiles
            assert app.shell.top_screen.tiles["tools.progress"].trailing is None  # no "Arrives in" chip
        finally:
            try:
                lc.uninstall_library_env()
            except Exception:
                pass
            await uf._stop(app)

    asyncio.run(scenario())


# ==========================================================================
# U5 review fixes
# ==========================================================================


@needs_flet
def test_pushed_screens_pop_back_to_the_screen_they_were_opened_from(app_env):
    """U5 review: the Reader, the metadata editor and Scan for raw opened from the Book page are
    pushed on top of it; back returns to /library/book/<bid> with the Book page still alive."""
    lc = _core("library_core", "install_library_env", "scan_library")
    if lc is None:
        pytest.skip("library_core (U5 API) not importable")
    uf_spec = importlib.util.spec_from_file_location("_glossarion_uf_helpers_library_push",
                                                     Path(__file__).with_name("test_ui_foundations.py"))
    uf = importlib.util.module_from_spec(uf_spec)
    uf_spec.loader.exec_module(uf)
    from glossarion_mobile.ui.library.book_page import BookPageScreen
    from glossarion_mobile.ui.library.home import LibraryScreen

    async def scenario():
        _m, conn, session, page, app = await uf._start("android")
        try:
            paths = app.paths
            raw = make_epub(Path(paths.library) / "Raw" / "Novel.epub", CHAPTERS, title="Novel")
            ws = Path(paths.output) / "Novel"
            ws.mkdir(parents=True, exist_ok=True)
            (ws / "source_epub.txt").write_text(raw, encoding="utf-8")
            (ws / "translation_progress.json").write_text(json.dumps({"version": "2.1", "chapters": {},
                                                                      "chapter_chunks": {}}), encoding="utf-8")
            await app.navigate("/library")
            home = app.shell.top_screen
            assert isinstance(home, LibraryScreen) and await uf._wait(lambda: bool(home.cards))
            card = next(iter(home.cards.values()))
            home._on_card_tap(card.model)
            assert await uf._wait(lambda: isinstance(app.shell.top_screen, BookPageScreen))
            book_page = app.shell.top_screen
            book_route = f"/library/book/{card.model.bid}"
            for route in (f"/reader/{card.model.bid}", f"{book_route}/metadata", "/library/scan-raw"):
                await app.navigate(route)
                assert uf._routes(page)[-3:] == ["/library", book_route, route], route
                await session.dispatch_event(page._i, "view_pop", {"route": route})
                assert uf._routes(page)[-2:] == ["/library", book_route], route
                assert app.shell.top_screen is book_page and not book_page.poller._stopped
            # a second book replaces the first at the same depth; a drawer destination resets the stack
            await app.navigate("/library/book/0123456789ab")
            assert uf._routes(page)[-2:] == ["/library", "/library/book/0123456789ab"]
            await app.navigate("/jobs")
            assert uf._routes(page)[-1] == "/jobs" and "/library" not in uf._routes(page)
        finally:
            try:
                lc.uninstall_library_env()
            except Exception:
                pass
            await uf._stop(app)

    asyncio.run(scenario())


@needs_flet
def test_retranslate_row_resets_a_completed_chapter_before_queueing(real_env):
    """U5 review: "Retranslate this chapter" on a completed row asks first and resets the progress
    entry to pending before the single-chapter job is queued (else the pipeline skips it)."""
    if _core("progress_core", "build_book_progress") is None:
        pytest.skip("progress cores not importable")
    from glossarion_mobile.ui.library.book_page import BookPageScreen
    from glossarion_mobile.ui.router import parse_route

    service = real_env["service"]
    progress_file = real_env["ws"] / "translation_progress.json"
    queued = []

    class Jobs:
        busy = False

        def has_kind(self, kind):
            return kind == "single_chapter"

        def submit(self, spec):
            queued.append((spec, json.loads(progress_file.read_text(encoding="utf-8"))["chapters"]["1"]["status"]))
            return "job-1"

    async def scenario():
        conn, session = _TB._fake_session("android")
        page = session.page
        await service.refresh()
        book = service.snapshot.in_progress[0]
        ctx = _ctx(page, service)
        screen = BookPageScreen(parse_route(f"/library/book/{service.bid_for(book)}?tab=chapters"), ctx)
        screen.actions()
        _mount(page, screen.get_body())
        await screen.load()
        chapters = screen.chapters
        completed = next(r for r in chapters.visible if r.status == "completed")
        service.jobs = Jobs()
        asked = []

        async def answer(title, body, **_kw):
            asked.append((title, body))
            return answer.value

        chapters._confirm = answer
        answer.value = False
        assert await chapters.translate_chapter(completed) is None and queued == []
        assert asked[-1][0] == "Retranslate chapter" and "retranslate it now?" in asked[-1][1]
        answer.value = True
        assert await chapters.translate_chapter(completed) == "job-1"
        spec, status_at_submit = queued[-1]
        assert spec.kind == "single_chapter" and spec.params["chapter_file"] == completed.filename
        assert status_at_submit == "pending"
        # a running translation refuses, like the desktop Book Details
        service.jobs.busy = True
        assert await chapters.translate_chapter(completed) is None and len(queued) == 1
        assert ctx.notes[-1].startswith("A translation is already running.")
        screen.dispose()

    asyncio.run(scenario())


def test_glossary_signature_sees_a_new_file_and_settles_on_a_deleted_one(real_env):
    if _core("glossary_progress_core", "open_glossary_progress") is None:
        pytest.skip("glossary progress core not importable")
    from glossarion_mobile.ui.library import progress_model as pm

    service = real_env["service"]
    asyncio.run(service.refresh())
    book = service.snapshot.in_progress[0]
    gp = real_env["gp"]
    moved = gp.with_name("moved.bak")
    gp.rename(moved)
    empty = pm.load_glossary_view(service, book)
    assert empty.empty and empty.signature is None
    assert pm.glossary_signature(empty, service, book) is None  # nothing yet: no reload
    moved.rename(gp)  # "Extract glossary" wrote the progress file
    assert pm.glossary_signature(empty, service, book) not in (None, empty.signature)
    view = pm.load_glossary_view(service, book)
    assert view.signature is not None and pm.glossary_signature(view, service, book) == view.signature
    gp.unlink()
    assert pm.glossary_signature(view, service, book) != view.signature  # -> reload
    deleted = pm.load_glossary_view(service, book, previous=view)
    assert deleted.deleted and pm.glossary_signature(deleted, service, book) == deleted.signature  # settled


def test_book_page_full_refresh_runs_one_at_a_time():
    from glossarion_mobile.ui.library.book_page import BookPageScreen

    screen = BookPageScreen.__new__(BookPageScreen)
    calls = []

    async def reload_progress(**_kw):
        calls.append("progress")
        await asyncio.sleep(0.05)

    async def nothing(*_a, **_kw):
        return None

    screen._full_refreshing = False
    screen.reload_progress = reload_progress
    screen.reload_details = nothing
    screen.glossary = None
    screen.service = types.SimpleNamespace(refresh=nothing)

    async def scenario():
        await asyncio.gather(*(screen.full_refresh() for _ in range(6)))  # one pull, many overscrolls
        await screen.full_refresh()

    asyncio.run(scenario())
    assert calls == ["progress", "progress"]


@needs_flet
def test_library_home_quiet_scans_compile_ribbon_back_and_txt_cards(tmp_path):
    """U5 review: an unchanged quiet scan pushes nothing; the COMPILING ribbon clears when the
    compile ends without changing the workspace; back leaves selection mode first; a TXT card
    is shared to another app instead of opening the Reader on an error."""
    from glossarion_mobile.services.library import _norm
    from glossarion_mobile.ui.library.home import LibraryScreen
    from glossarion_mobile.ui.router import parse_route
    from glossarion_mobile.ui.screens.base import build_screen_view

    rows = _rows(tmp_path)
    notes_txt = tmp_path / "notes.txt"
    notes_txt.write_text("plain text", encoding="utf-8")
    rows[1].append({"name": "notes", "type": "txt", "path": str(notes_txt), "size": 10, "mtime": 9,
                    "translation_state": "completed", "is_in_progress": False})
    service = LibraryService(core=SharedCore({"library_core": fake_core(tmp_path, rows)}), prefs=FakePrefs(),
                             config={"epub_library_card_size": "normal", "epub_library_page_size": 20})
    shared = []

    async def share(paths):
        shared.append(list(paths))
        return True

    async def scenario():
        conn, session = _TB._fake_session("android")
        page = session.page
        ctx = _ctx(page, service, files=types.SimpleNamespace(share=share))
        screen = LibraryScreen(parse_route("/library"), ctx)
        screen.actions()
        _mount(page, screen.get_body())
        screen.did_show()
        for _ in range(200):
            if screen.snapshot.scanned_at and screen.cards:
                break
            await asyncio.sleep(0.01)
        pushes = []
        ctx.push = lambda *controls: pushes.append(controls)
        screen._on_snapshot(screen.snapshot)
        assert pushes == []  # nothing changed: nothing is sent
        alpha = rows[0][0]
        key = book_key(alpha)
        service.compiling.add(_norm(alpha["output_folder"]))
        screen._on_snapshot(screen.snapshot)
        assert screen.cards[key].model.ribbon_state == "compiling"
        service.compiling.clear()  # cancelled while queued: the workspace signature never changed
        screen._on_snapshot(screen.snapshot)
        assert screen.cards[key].model.ribbon_state != "compiling"
        ctx.push = LibraryContext_push
        # Android back leaves selection mode before it leaves the Library
        screen._on_card_long_press(screen.cards[key].model)
        assert screen.selecting and screen.handle_back() is True and not screen.selecting
        assert screen.handle_back() is False
        view = build_screen_view(screen, "/library")
        assert view.can_pop is False and callable(view.on_confirm_pop)
        # a TXT card goes to the share sheet
        screen.set_shelf("completed")
        before = list(ctx.navigated)
        screen._on_card_tap(screen.cards[book_key(rows[1][-1])].model)
        await asyncio.sleep(0.05)
        assert shared == [[str(notes_txt)]] and ctx.navigated == before
        screen.dispose()

    from glossarion_mobile.ui.library.common import LibraryContext

    LibraryContext_push = LibraryContext.push
    asyncio.run(scenario())


@needs_flet
def test_delete_view_dismissed_by_back_and_a_failed_delete(tmp_path):
    """U5 review: a keyword view already popped by Android back is not popped again (that would
    pop the Library); a failed delete re-enables Cancel and the checkboxes."""
    from glossarion_mobile.ui.library.delete_confirm import DeleteFlow

    rows = _rows(tmp_path)
    service = LibraryService(core=SharedCore({"library_core": fake_core(tmp_path, rows)}), prefs=FakePrefs(),
                             config={})
    alpha, beta = rows[0]
    FakeShelf.plan = {"targets": [("Alpha", alpha["output_folder"], True, alpha),
                                  ("Beta", beta["output_folder"], True, beta)],
                      "unregister": [], "needs_keyword": True, "simple_prompt": ""}

    async def scenario():
        conn, session = _TB._fake_session("android")
        shell = types.SimpleNamespace(overlays=[])
        ctx = _ctx(session.page, service, shell=shell)
        flow = DeleteFlow(ctx)
        view = await flow.start([alpha, beta])
        assert ctx.overlays == [view.view] and view.view not in shell.overlays  # back already popped it
        view.field.value = "delete"
        await view._on_delete()
        assert ctx.overlays == [view.view]  # pop_overlay was not called a second time
        # a failure leaves the view usable
        flow2 = DeleteFlow(ctx)
        view2 = await flow2.start([alpha, beta])
        shell.overlays.append(view2.view)
        original = FakeShelf.execute_delete

        def boom(self, plan, targets=None, on_progress=None):
            raise OSError("disk full")

        FakeShelf.execute_delete = boom
        try:
            view2.field.value = "delete"
            await view2._on_delete()
        finally:
            FakeShelf.execute_delete = original
        assert not view2.cancel_button.disabled and not any(box.disabled for box in view2.checks)
        assert view2.status.value == "Delete failed: disk full" and not view2.delete_button.disabled

    asyncio.run(scenario())


@needs_flet
def test_chapters_list_is_windowed_beyond_1500_rows(real_env):
    if _core("progress_core", "build_book_progress") is None:
        pytest.skip("progress cores not importable")
    import flet as ft

    from glossarion_mobile.ui.library import chapters_tab as ct
    from glossarion_mobile.ui.library import progress_model as pm
    from glossarion_mobile.ui.library.book_page import BookPageScreen
    from glossarion_mobile.ui.router import parse_route

    service = real_env["service"]

    async def scenario():
        conn, session = _TB._fake_session("android")
        page = session.page
        await service.refresh()
        book = service.snapshot.in_progress[0]
        screen = BookPageScreen(parse_route(f"/library/book/{service.bid_for(book)}?tab=chapters"),
                                _ctx(page, service))
        screen.actions()
        _mount(page, screen.get_body())
        await screen.load()
        chapters = screen.chapters
        many = [pm.RowVM(key=f"opf:{i}:c{i}.xhtml", kind="chapter", status="completed", icon="✅",
                         label="Completed", title=f"Chapter {i}", filename=f"c{i}.xhtml") for i in range(3200)]
        chapters.page_size = 0  # Rows per page: All
        chapters._render_list(many)
        assert chapters.windowed and chapters.window_start == 0
        assert len(chapters.list_view.controls) == ct.WINDOW_STEP  # appended per scroll step, not all at once
        assert isinstance(chapters.list_holder.content, ft.Column)
        assert chapters.window_text.value == "Rows 1–1,500 of 3,200"
        chapters._extend_to(5000)
        assert len(chapters.list_view.controls) == ct.WINDOW_ROWS  # never more than one window mounted
        chapters.set_window(3100)
        assert chapters.window_start == 3000 and chapters.window_text.value == "Rows 3,001–3,200 of 3,200"
        assert chapters.list_view.controls and chapters.rendered <= 3200
        chapters._render_list(many[:40])
        assert not chapters.windowed and chapters.list_holder.content is chapters.list_view
        assert len(chapters.list_view.controls) == 40
        screen.dispose()

    asyncio.run(scenario())


def test_translate_sheet_reuses_the_sources_resolved_on_the_io_pool(tmp_path, monkeypatch):
    if not _has("flet"):
        pytest.skip("flet not installed")
    from glossarion_mobile.ui.library.translate_sheet import TranslateSheet

    raw = tmp_path / "Raw" / "Book.epub"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"PK")
    submitted = []

    class Jobs:
        def has_kind(self, kind):
            return kind == "translate"

        async def submit(self, spec):
            submitted.append(spec)
            return "job1"

    service = LibraryService(core=SharedCore({"library_core": types.ModuleType("library_core")}),
                             prefs=FakePrefs(), jobs=Jobs(), config={})
    book = _book("not_started", name="Book", raw_source_path=str(raw))
    ctx = _ctx(None, service)
    sheet = TranslateSheet(ctx, [book], [str(raw)])

    def no_ui_loop_scan(_book):
        raise AssertionError("raw_source() re-ran on the UI loop")

    monkeypatch.setattr(service, "raw_source", no_ui_loop_scan)
    assert asyncio.run(sheet.start()) == "job1" and submitted[0].inputs == (str(raw),)


# ==========================================================================
# U5 second review round
# ==========================================================================


def _foundations(tag):
    spec = importlib.util.spec_from_file_location(f"_glossarion_uf_helpers_{tag}",
                                                  Path(__file__).with_name("test_ui_foundations.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def _open_novel(app, uf):
    """The running app with one workspace (Novel): Library -> Book page; returns (home, page, bid)."""
    from glossarion_mobile.ui.library.book_page import BookPageScreen
    from glossarion_mobile.ui.library.home import LibraryScreen

    paths = app.paths
    raw = make_epub(Path(paths.library) / "Raw" / "Novel.epub", CHAPTERS, title="Novel")
    ws = Path(paths.output) / "Novel"
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "source_epub.txt").write_text(raw, encoding="utf-8")
    (ws / "translation_progress.json").write_text(json.dumps({"version": "2.1", "chapters": {},
                                                              "chapter_chunks": {}}), encoding="utf-8")
    await app.navigate("/library")
    home = app.shell.top_screen
    assert isinstance(home, LibraryScreen) and await uf._wait(lambda: bool(home.cards))
    card = next(iter(home.cards.values()))
    home._on_card_tap(card.model)
    assert await uf._wait(lambda: isinstance(app.shell.top_screen, BookPageScreen))
    book_page = app.shell.top_screen
    assert await uf._wait(lambda: book_page.details_phase == "full" and book_page.progress is not None)
    return home, book_page, card.model.bid


@needs_flet
def test_screens_opened_over_the_book_page_pop_back_to_it(app_env):
    """U5 second review: Files (Book page, Library card, Output tab) and the Overview "Last job"
    link are pushed on top of the Book page like the Reader, so back returns to it; the hidden
    Book page does not reload while a screen is on top of it; drawer navigation starts over."""
    lc = _core("library_core", "install_library_env", "scan_library")
    if lc is None:
        pytest.skip("library_core (U5 API) not importable")
    uf = _foundations("push2")

    async def scenario():
        _m, conn, session, page, app = await uf._start("android")
        try:
            _home, book_page, bid = await _open_novel(app, uf)
            book_route = f"/library/book/{bid}"
            loads = []
            real_load = app.library.load_details_blocking

            def counting(book, phase="full"):
                loads.append(phase)
                return real_load(book, phase)

            app.library.load_details_blocking = counting
            book_page.open_files()
            assert await uf._wait(lambda: app.shell.stack[-1].match.name == "tools.files.folder")
            routes = uf._routes(page)
            assert routes[:3] == ["/", "/library", book_route] and routes[-1].startswith("/tools/files/output/")
            await session.dispatch_event(page._i, "view_pop", {"route": routes[-1]})
            assert uf._routes(page) == ["/", "/library", book_route]
            assert app.shell.top_screen is book_page and not book_page.poller._stopped
            app.navigate_to("jobs.detail", {"jid": "abc123"})  # Overview "Last job: …"
            assert await uf._wait(lambda: uf._routes(page)[-1] == "/jobs/abc123")
            assert uf._routes(page) == ["/", "/library", book_route, "/jobs/abc123"]
            await session.dispatch_event(page._i, "view_pop", {"route": "/jobs/abc123"})
            assert app.shell.top_screen is book_page and not book_page.poller._stopped
            await asyncio.sleep(0.3)
            loads.clear()
            await app.navigate(f"/reader/{bid}")
            assert uf._routes(page)[-2:] == [book_route, f"/reader/{bid}"]
            await asyncio.sleep(0.8)
            assert loads == []  # the Book page under the Reader is not shown (reloaded) again
            await session.dispatch_event(page._i, "view_pop", {"route": f"/reader/{bid}"})
            assert app.shell.top_screen is book_page
            # drawer navigation rebuilds the stack from the route's static parents
            app._drawer_navigate("settings.logs")
            assert await uf._wait(lambda: uf._routes(page) == ["/", "/settings", "/settings/logs"])
            assert book_page.poller._stopped
        finally:
            try:
                lc.uninstall_library_env()
            except Exception:
                pass
            await uf._stop(app)

    asyncio.run(scenario())


@needs_flet
def test_android_back_on_the_keyword_delete_view_pops_only_that_view(app_env):
    """U5 second review: the keyword delete view has a View route of its own. Android back (the
    client sends the top View's route; Flet resolves it to the first View with that route) pops
    the delete view, not the Library or the Book page under it. An overlay whose route collides
    with another View's gets a suffix."""
    lc = _core("library_core", "install_library_env", "scan_library")
    if lc is None:
        pytest.skip("library_core (U5 API) not importable")
    import flet as ft

    from glossarion_mobile.ui.library.delete_confirm import DELETE_VIEW_ROUTE

    uf = _foundations("delete2")

    def keyword_plan(books):
        book = dict(books[0])
        return DeletePlan(targets=(DeleteTarget(label=book.get("name"), path=book.get("output_folder"),
                                                is_folder=True, book=book),), needs_keyword=True)

    async def scenario():
        _m, conn, session, page, app = await uf._start("android")
        try:
            home, book_page, bid = await _open_novel(app, uf)
            book_route = f"/library/book/{bid}"
            app.library.plan_delete_blocking = keyword_plan
            await book_page.delete()
            top = page.views[-1]
            assert top is book_page.delete_flow.view.view and top.route == DELETE_VIEW_ROUTE
            assert uf._routes(page) == ["/", "/library", book_route, DELETE_VIEW_ROUTE]
            await session.dispatch_event(page._i, "view_pop", {"route": top.route})
            assert uf._routes(page) == ["/", "/library", book_route] and app.shell.overlays == []
            assert app.shell.top_screen is book_page and not book_page.poller._stopped
            assert not home.poller._stopped
            # the same from the Library home
            await session.dispatch_event(page._i, "view_pop", {"route": book_route})
            assert app.shell.top_screen is home
            await home.delete_books([next(iter(home.books_by_key.values()))])
            assert uf._routes(page) == ["/", "/library", DELETE_VIEW_ROUTE]
            await session.dispatch_event(page._i, "view_pop", {"route": DELETE_VIEW_ROUTE})
            assert uf._routes(page) == ["/", "/library"] and app.shell.top_screen is home
            # any overlay reusing another View's route is renamed (the Model manager sub-screens
            # are built with the current route)
            clash = ft.View(route="/library")
            app.shell.push_overlay(clash)
            assert clash.route == "/library/overlay-1" and page.views[-1] is clash
            await session.dispatch_event(page._i, "view_pop", {"route": clash.route})
            assert app.shell.overlays == [] and app.shell.top_screen is home and not home.poller._stopped
        finally:
            try:
                lc.uninstall_library_env()
            except Exception:
                pass
            await uf._stop(app)

    asyncio.run(scenario())


@needs_flet
def test_tablet_system_back_leaves_selection_then_pops_the_main_area(app_env):
    """U5 second review: on tablets the Library lives in the root View's main area, so the system
    back has no second View to pop. The root cannot pop while a screen shows: back reaches its
    on_confirm_pop, which leaves selection mode first, then pops the main-area stack; with the
    chat showing, the root can pop (the system default leaves the app)."""
    lc = _core("library_core", "install_library_env", "scan_library")
    if lc is None:
        pytest.skip("library_core (U5 API) not importable")
    from flet.messaging.protocol import MessageAction

    from glossarion_mobile.ui.library.home import LibraryScreen

    uf = _foundations("tablet2")

    def confirm_calls(conn):
        return [(m.body.args or {}).get("should_pop") for m in conn.messages
                if m.action == MessageAction.INVOKE_METHOD and m.body.name == "confirm_pop"]

    async def scenario():
        _m, conn, session, page, app = await uf._start("android", width=1000)
        try:
            assert app.shell.tablet and page.views[0].can_pop is True  # the chat: system default
            paths = app.paths
            raw = make_epub(Path(paths.library) / "Raw" / "Novel.epub", CHAPTERS, title="Novel")
            ws = Path(paths.output) / "Novel"
            ws.mkdir(parents=True, exist_ok=True)
            (ws / "source_epub.txt").write_text(raw, encoding="utf-8")
            await app.navigate("/library")
            home = app.shell.top_screen
            assert isinstance(home, LibraryScreen) and await uf._wait(lambda: bool(home.cards))
            root = page.views[0]
            assert uf._routes(page) == ["/"] and root.can_pop is False and callable(root.on_confirm_pop)
            home._on_card_long_press(next(iter(home.cards.values())).model)
            assert home.selecting
            conn.messages.clear()
            await session.dispatch_event(root._i, "confirm_pop", None)
            assert await uf._wait(lambda: confirm_calls(conn) == [False])
            assert not home.selecting and app.shell.top_screen is home  # rule 2: selection first
            await session.dispatch_event(root._i, "confirm_pop", None)
            assert await uf._wait(lambda: confirm_calls(conn) == [False, False])
            assert app.shell.stack == [] and home.poller._stopped  # rule 5: the main area pops
            assert page.views[0].can_pop is True
        finally:
            try:
                lc.uninstall_library_env()
            except Exception:
                pass
            await uf._stop(app)

    asyncio.run(scenario())


@needs_flet
def test_chapters_selection_keeps_the_mounted_window(real_env):
    """U5 second review: in a windowed Chapters list, selecting / leaving selection keeps the window."""
    if _core("progress_core", "build_book_progress") is None:
        pytest.skip("progress cores not importable")
    from glossarion_mobile.ui.library import progress_model as pm
    from glossarion_mobile.ui.library.book_page import BookPageScreen
    from glossarion_mobile.ui.router import parse_route

    service = real_env["service"]

    async def scenario():
        conn, session = _TB._fake_session("android")
        page = session.page
        await service.refresh()
        book = service.snapshot.in_progress[0]
        screen = BookPageScreen(parse_route(f"/library/book/{service.bid_for(book)}?tab=chapters"),
                                _ctx(page, service))
        screen.actions()
        _mount(page, screen.get_body())
        await screen.load()
        chapters = screen.chapters
        many = [pm.RowVM(key=f"opf:{i}:c{i}.xhtml", kind="chapter", status="completed", icon="✅",
                         label="Completed", title=f"Chapter {i}", filename=f"c{i}.xhtml") for i in range(3200)]
        chapters.page_size = 0
        chapters._render_list(many)
        chapters.set_window(1600)
        assert chapters.window_start == 1500
        chapters.on_row_long_press(many[1700])
        chapters.select_all()
        assert chapters.window_start == 1500 and len(chapters.selected) == 3200
        assert screen.handle_back() is True and not chapters.selecting  # Android back leaves selection
        assert chapters.window_start == 1500 and chapters.window_text.value.startswith("Rows 1,501")
        chapters.on_row_long_press(many[1700])
        chapters.select_status("completed")
        assert chapters.window_start == 1500
        chapters.select_group("completed")
        assert chapters.window_start == 1500
        chapters.exit_selection()  # Close / after a bulk action
        assert chapters.window_start == 1500 and not chapters.selecting
        screen.dispose()

    asyncio.run(scenario())


@needs_flet
def test_pdf_without_a_workspace_is_not_offered_to_the_reader(tmp_path):
    """U5 second review: the card ⋯ "Open in Reader" follows the card tap: a PDF without a
    translation workspace opens in another app (library_core's "system" decision)."""
    from glossarion_mobile.ui.library.home import LibraryScreen
    from glossarion_mobile.ui.router import parse_route

    rows = _rows(tmp_path)
    pdf = tmp_path / "Manual.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")
    row = {"name": "Manual", "type": "pdf", "path": str(pdf), "size": 15, "mtime": 9,
           "translation_state": "completed", "is_in_progress": False}
    rows[1].append(row)
    service = LibraryService(core=SharedCore({"library_core": fake_core(tmp_path, rows)}), prefs=FakePrefs(),
                             config={})

    async def scenario():
        conn, session = _TB._fake_session("android")
        ctx = _ctx(session.page, service)
        screen = LibraryScreen(parse_route("/library"), ctx)

        def reader_item(book):
            return next(i for i in screen.card_actions(book).items if "Open in Reader" in i.label)

        assert "another app" in (reader_item(row).disabled_reason or "")
        workspace = dict(row, output_folder=str(tmp_path / "Output" / "Manual"))
        assert reader_item(workspace).disabled_reason is None
        epub = rows[1][0]  # the completed Done.epub
        assert reader_item(epub).disabled_reason is None
        screen.dispose()

    asyncio.run(scenario())
