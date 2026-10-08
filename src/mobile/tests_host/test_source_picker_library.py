"""Host tests for the in-chat Library picker (owner device report #8, package "picker").

The owner: "I should be able to just directly [use] my epub library from direct chat, not manually
do clicks". The chat's ＋ › From Library / the empty-chat chip / ``/library`` open the Tools
``SourcePicker`` in its opt-in Library mode (``segments=("library",)``, ``searchable``,
``book_rows``, ``include_unresolved``, ``query``, ``header_action``): one segment (no
SegmentedButton), the Library's "Filter title or tag…" search (shared ``book_matches_query``),
newest first (the Library's Date sort through ``visible_books`` + ``LibraryService.sort_books``),
the Library's own list rows (``card_model_for`` + ``BookListRow(show_more=False)``, covers through
``CoverQueue``) and one tap that hands the book to the chat. Tools pickers keep the defaults: the
four segments, plain rows in snapshot order, and ``library_targets`` output unchanged.

A real ``LibraryService`` over a fake ``library_core`` that carries the REAL shared
``sort_books`` / ``book_matches_query`` (no scan: the snapshot is set directly) and a fake
``library_covers``; the picker runs in the in-memory Flet session from ``test_bootstrap``.
Real data is never touched: HOME / USERPROFILE / APPDATA / GLOSSARION_LIBRARY_DIR /
OUTPUT_DIRECTORY point at the pytest tmp dir and GLOSSARION_HTTP_LOG=0.

Run from src/mobile with the mobile venv:
    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_source_picker_library.py
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
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from glossarion_mobile.services.library import LibraryService, ScanSnapshot, SharedCore, book_key  # noqa: E402
from glossarion_mobile.ui.tools import targets as tg  # noqa: E402


def _has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


needs_flet = pytest.mark.skipif(not (_has("flet") and _has("msgpack")), reason="flet / msgpack not installed")

NO_RAW = "No raw file on this device"


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    for name in ("HOME", "USERPROFILE", "APPDATA"):
        monkeypatch.setenv(name, str(home))
    monkeypatch.setenv("GLOSSARION_LIBRARY_DIR", str(tmp_path / "Library"))
    monkeypatch.setenv("OUTPUT_DIRECTORY", str(tmp_path / "Output"))
    monkeypatch.setenv("GLOSSARION_HTTP_LOG", "0")
    qa = types.ModuleType("qa_scan_runtime")
    qa.is_direct_text_qa_path = lambda path: bool(path) and "Direct Text" in str(path)
    monkeypatch.setitem(sys.modules, "qa_scan_runtime", qa)


def _shared_lc():
    try:
        import library_core
    except Exception as exc:  # pragma: no cover - backend not importable
        pytest.skip(f"library_core not importable: {exc}")
    if not (hasattr(library_core, "sort_books") and hasattr(library_core, "book_matches_query")):
        pytest.skip("library_core has no sort_books / book_matches_query")
    return library_core


def _file(path: Path, data: bytes = b"PK") -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return str(path)


def library(tmp_path):
    """Four Library rows: an in-progress workspace with a raw (mtime 100), a Completed book tagged
    "Fantasy" with a raw (300, the newest; cached cover), two Completed Library/Translated EPUBs with
    neither a workspace nor a raw (200 and 50: the unresolved rows)."""
    real = _shared_lc()
    alpha_ws = tmp_path / "Output" / "Alpha Novel"
    alpha_ws.mkdir(parents=True)
    (alpha_ws / "translation_progress.json").write_text("{}", encoding="utf-8")
    alpha = {"name": "Alpha Novel", "type": "in_progress", "workspace_kind": "epub", "output_folder": str(alpha_ws),
             "path": str(alpha_ws), "raw_source_path": _file(tmp_path / "Library" / "Raw" / "Alpha Novel.epub"),
             "mtime": 100.0, "size": 2048, "translation_state": "in_progress", "is_in_progress": True,
             "completed_chapters": 3, "total_chapters": 10}
    beta_ws = tmp_path / "Output" / "Beta Saga"
    beta_ws.mkdir(parents=True)
    beta = {"name": "Beta Saga", "type": "epub", "output_folder": str(beta_ws),
            "path": _file(tmp_path / "Output" / "Beta Saga" / "Beta Saga_translated.epub"),
            "raw_source_path": _file(tmp_path / "Library" / "Raw" / "Beta Saga.epub"), "subjects": ["Fantasy"],
            "mtime": 300.0, "size": 4096, "translation_state": "completed", "is_in_progress": False}
    gamma = {"name": "Gamma", "type": "epub", "in_library": True,
             "path": _file(tmp_path / "Library" / "Translated" / "Gamma.epub"), "mtime": 200.0, "size": 1024,
             "translation_state": "completed", "is_in_progress": False}
    delta = {"name": "Delta", "type": "epub", "in_library": True,
             "path": _file(tmp_path / "Library" / "Translated" / "Delta.epub"), "mtime": 50.0, "size": 1024,
             "translation_state": "completed", "is_in_progress": False}
    cover_calls: list = []
    alpha_cover = _file(tmp_path / "covers" / "alpha.png", b"\x89PNG\r\n\x1a\n")
    beta_cover = _file(tmp_path / "covers" / "beta.png", b"\x89PNG\r\n\x1a\n")

    core = types.ModuleType("library_core")
    core.sort_books = real.sort_books  # the desktop toolbar's Date / A-Z / Size order
    core.book_matches_query = real.book_matches_query  # titles, raw titles and tags
    covers = types.ModuleType("library_covers")

    def resolve_card_cover(book, config):
        cover_calls.append(book.get("name"))
        return alpha_cover if book.get("name") == "Alpha Novel" else None

    covers.resolve_card_cover = resolve_card_cover
    service = LibraryService(core=SharedCore({"library_core": core, "library_covers": covers}), config={})
    service.set_snapshot(ScanSnapshot(in_progress=(alpha,), completed=(beta, gamma, delta), scanned_at=1.0))
    service.covers[book_key(beta)] = beta_cover  # already in the Library's cover cache
    return types.SimpleNamespace(service=service, alpha=alpha, beta=beta, gamma=gamma, delta=delta,
                                 cover_calls=cover_calls, alpha_cover=alpha_cover, beta_cover=beta_cover)


def chat_rule(counter: list):
    """The chat's eligibility (ChatView.open_library_picker): the raw file must be on this device."""

    def eligible(target):
        counter.append(target.key)
        source = str(target.source or "")
        if not source or not os.path.isfile(source):
            return NO_RAW
        return None

    return eligible


def _tb():
    spec = importlib.util.spec_from_file_location("_glossarion_picker_tb_helpers",
                                                  Path(__file__).with_name("test_bootstrap.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ctx(page, service):
    from glossarion_mobile.ui.tools.common import ToolsContext

    navigated: list = []
    notes: list = []
    ctx = ToolsContext(service=service, page=page,
                       navigate=lambda name, params=None, query=None: navigated.append((name, params, query)),
                       notify=lambda message, action=None, on_action=None: notes.append(message),
                       platform="android", store={})
    ctx.navigated, ctx.notes = navigated, notes
    return ctx


async def _until(predicate, timeout=5.0):
    loop = asyncio.get_running_loop()
    end = loop.time() + timeout
    while not predicate():
        if loop.time() > end:
            raise AssertionError("timed out")
        await asyncio.sleep(0.01)


def _walk(control):
    from flet.controls.base_control import BaseControl

    yield control
    for name in ("content", "title", "subtitle", "leading", "trailing", "suffix"):
        child = getattr(control, name, None)
        if isinstance(child, BaseControl):
            yield from _walk(child)
    for name in ("controls", "actions"):
        for child in getattr(control, name, None) or ():
            if isinstance(child, BaseControl):
                yield from _walk(child)


def _shown(picker) -> list:
    """The titles the list shows, top to bottom."""
    by_control = {id(control): row for row, control in picker.book_cards.values()}
    out = []
    for control in picker.list_view.controls:
        row = by_control.get(id(control))
        out.append(row.model.title if row is not None else getattr(control, "key", None))
    return out


def _target(picker, title):
    return next(t for t in picker.rows["library"] if t.title == title)


def _library_picker(ctx, eligible, done, **kwargs):
    from glossarion_mobile.ui.tools.source_picker import SourcePicker

    options = dict(title="Attach from Library", multi=False, eligible=eligible, on_done=done.append,
                   segment="library", segments=("library",), searchable=True, book_rows=True,
                   include_unresolved=True, find_folder=False)
    options.update(kwargs)
    return SourcePicker(ctx, **options)


# ==========================================================================
# targets.library_targets
# ==========================================================================


def test_unresolved_rows_keep_distinct_keys_and_the_default_output_is_unchanged(tmp_path):
    lib = library(tmp_path)
    service = lib.service
    default = tg.library_targets(service)
    # the pre-change rows, field by field: In progress then Completed, unresolved rows dropped, the
    # folder / src: keys, the shared kinds; ``extra`` only gained the scanned row
    expected = [
        ("Alpha Novel", lib.alpha["output_folder"], lib.alpha["raw_source_path"], "library",
         service.bid_for(lib.alpha), 100.0, False, "in_progress",
         os.path.normcase(os.path.abspath(lib.alpha["output_folder"]))),
        ("Beta Saga", lib.beta["output_folder"], lib.beta["raw_source_path"], "library",
         service.bid_for(lib.beta), 300.0, False, "epub",
         os.path.normcase(os.path.abspath(lib.beta["output_folder"]))),
    ]
    assert [(t.title, t.folder, t.source, t.origin, t.bid, t.mtime, t.direct_text, t.extra["type"], t.key)
            for t in default] == expected
    assert [t.kind for t in default] == [tg.workspace_kind(t.folder, t.source, b)
                                         for t, b in zip(default, (lib.alpha, lib.beta))]
    assert all(set(t.extra) == {"type", "book"} for t in default)
    assert default[1].extra["book"] == lib.beta and default[1].extra["book"] is not lib.beta  # a copy
    assert tg.target_for_book(service, lib.gamma) is None  # the Tools deep links keep "no target"

    full = tg.library_targets(service, include_unresolved=True)
    assert [t.title for t in full] == ["Alpha Novel", "Beta Saga", "Gamma", "Delta"]
    gamma, delta = full[2], full[3]
    # both unresolved rows are listed: before, both keyed "src:" + cwd and the second was dropped
    assert gamma.key == "bid:" + service.bid_for(lib.gamma) and delta.key == "bid:" + service.bid_for(lib.delta)
    assert gamma.key != delta.key and not gamma.folder and not gamma.source
    assert gamma.kind == "epub" and gamma.mtime == 200.0 and gamma.extra["book"]["path"] == lib.gamma["path"]
    # without a route id the row's own file keys it; a target with nothing keeps the old key
    assert tg.target_key(tg.ToolTarget("x", extra={"book": {"path": lib.gamma["path"]}})) == \
        "path:" + os.path.normcase(os.path.abspath(lib.gamma["path"]))
    assert tg.target_key(tg.ToolTarget("x")) == "src:" + os.path.normcase(os.path.abspath(""))
    # Recent outputs (folders only) never see them
    assert [t.title for t in tg.recent_output_targets(service, library_rows=full)] == ["Alpha Novel"]


def test_the_sort_adapter_takes_reverse_by_keyword(tmp_path):
    """visible_books passes ``reverse`` positionally; LibraryService.sort_books takes it by keyword
    only, so the service method cannot be passed as is (skeptic probe: TypeError)."""
    from glossarion_mobile.ui.library.models import FilterState, visible_books

    lib = library(tmp_path)
    service = lib.service
    books = [lib.alpha, lib.beta, lib.gamma]
    with pytest.raises(TypeError):
        visible_books(books, FilterState(), matches=service.matches_query, format_of=service.format_of,
                      sort=service.sort_books)
    rows = tg.library_targets(service, include_unresolved=True)
    assert tg.LIBRARY_SORT == "date"
    assert [t.title for t in tg.order_library_rows(service, rows)] == ["Beta Saga", "Gamma", "Alpha Novel", "Delta"]
    assert [t.title for t in tg.order_library_rows(service, rows, "gam")] == ["Gamma"]
    assert [t.title for t in tg.order_library_rows(service, rows, "FANTASY")] == ["Beta Saga"]  # a tag
    assert [t.title for t in tg.order_library_rows(service, rows, "", sort="name")] == [
        "Alpha Novel", "Beta Saga", "Delta", "Gamma"]
    # a row without a scanned book (Browse) follows the books when its name matches
    loose = tg.ToolTarget("Picked File", source=str(tmp_path / "Picked File.epub"), origin="browse")
    assert [t.title for t in tg.order_library_rows(service, rows + [loose], "picked")] == ["Picked File"]
    assert tg.target_matches(service, loose, "file") and not tg.target_matches(service, loose, "saga")
    assert tg.target_matches(service, rows[1], "fantasy") and tg.target_matches(None, rows[1], "")


# ==========================================================================
# SourcePicker: Library mode
# ==========================================================================


@needs_flet
def test_library_picker_is_one_searchable_newest_first_list_of_library_rows(tmp_path):
    from glossarion_mobile.ui.components.reason_chip import ReasonChip
    from glossarion_mobile.ui.library.book_card import BookListRow

    lib = library(tmp_path)
    service = lib.service
    tb = _tb()

    async def scenario():
        _conn, session = tb._fake_session("android")
        page = session.page
        ctx = _ctx(page, service)
        checked: list = []
        done: list = []
        picker = _library_picker(ctx, chat_rule(checked), done).show(page)
        await _until(lambda: picker.loaded)
        page.update()  # every row serialises
        # one segment: no SegmentedButton; the Library's search field
        assert not picker.segment_row.visible and [s.value for s in picker.segmented.segments] == ["library"]
        assert picker.search_field is not None and picker.search_field.hint_text == "Filter title or tag…"
        assert any(c is picker.search_field for c in _walk(picker.sheet.content))
        # newest first (Date sort), the unresolved Completed rows included
        assert _shown(picker) == ["Beta Saga", "Gamma", "Alpha Novel", "Delta"]
        assert picker.status.value == "4 items"
        # the Library's own list rows: keyed book-<bid>, the cached cover, no ⋯
        beta_t = _target(picker, "Beta Saga")
        beta_row, beta_control = picker.book_cards[beta_t.key]
        assert isinstance(beta_row, BookListRow) and beta_control is beta_row.control
        assert beta_control.key == f"book-{service.bid_for(lib.beta)}" and beta_row.model.bid == beta_t.bid
        assert beta_row.cover_src == lib.beta_cover and beta_row.show_more is False
        assert not [c for c in _walk(beta_control) if getattr(c, "key", None) == "more"]
        alpha_row = picker.book_cards[_target(picker, "Alpha Novel").key][0]
        assert alpha_row.model.pill_text == "⏳ 3/10" and alpha_row.model.pct_text == "30%"
        # missing covers load through the Library's CoverQueue (io), the cached one is never looked up
        await _until(lambda: alpha_row.cover_src == lib.alpha_cover)
        await _until(lambda: not picker._covers.running)
        assert sorted(lib.cover_calls) == ["Alpha Novel", "Delta", "Gamma"]
        # the unresolved row: visible, the reason chip, muted, a tap does nothing
        gamma_t = _target(picker, "Gamma")
        gamma_row, gamma_control = picker.book_cards[gamma_t.key]
        chips = [c for c in _walk(gamma_control) if isinstance(c, ReasonChip)]
        assert gamma_control is not gamma_row.control and gamma_control in picker.list_view.controls
        assert [c.reason for c in chips] == [NO_RAW] and not gamma_control.disabled
        assert gamma_row.control.on_click is None and gamma_row.control.opacity < 1
        gamma_row.on_open(gamma_row.model)
        picker.toggle(gamma_t)
        assert done == [] and picker.result is None and picker.sheet.open
        # search: the shared query (tags too); the row objects are reused, so covers update in place
        picker.search_field.value = "fantasy"
        picker._on_query(types.SimpleNamespace(control=picker.search_field))
        assert _shown(picker) == ["Beta Saga"] and picker.list_view.controls[0] is beta_control
        assert picker.status.value == "1 of 4 items"
        picker.set_query("nothing like it")
        assert picker.list_view.controls == [] and picker.status.value == "No Library book matches “nothing like it”"
        picker._clear_query()
        assert picker.search_field.value == "" and len(picker.list_view.controls) == 4
        page.update()
        # eligibility ran once per row, in gather (io) - not again on render, search or tap
        assert sorted(checked) == sorted(t.key for t in picker.rows["library"])
        # one tap attaches: the picker closes and hands the book over (no navigation away)
        beta_row.on_open(beta_row.model)
        assert [t.key for t in done[0]] == [beta_t.key] and done[0][0].source == lib.beta["raw_source_path"]
        assert done[0][0].extra["book"]["name"] == "Beta Saga" and not picker.sheet.open
        assert ctx.navigated == []
        assert sorted(checked) == sorted(t.key for t in picker.rows["library"])

    asyncio.run(scenario())


@needs_flet
def test_library_picker_query_prefill_and_header_action(tmp_path):
    lib = library(tmp_path)
    tb = _tb()

    async def scenario():
        _conn, session = tb._fake_session("android")
        page = session.page
        ctx = _ctx(page, lib.service)
        opened: list = []
        picker = _library_picker(ctx, chat_rule([]), [], query="saga",
                                 header_action=("Open Library", lambda: opened.append("library"))).show(page)
        assert picker.search_field.value == "saga"
        await _until(lambda: picker.loaded)
        assert _shown(picker) == ["Beta Saga"]  # the first render is already filtered
        assert picker.header_button.content == "Open Library"
        assert any(c is picker.header_button for c in _walk(picker.sheet.content))
        picker.header_button.on_click(types.SimpleNamespace(control=picker.header_button))
        assert opened == ["library"] and not picker.sheet.open

    asyncio.run(scenario())


@needs_flet
def test_library_picker_multi_mode_updates_book_rows_in_place(tmp_path):
    lib = library(tmp_path)
    tb = _tb()

    async def scenario():
        _conn, session = tb._fake_session("android")
        page = session.page
        done: list = []
        picker = _library_picker(_ctx(page, lib.service), chat_rule([]), done, multi=True).show(page)
        await _until(lambda: picker.loaded)
        alpha_t = _target(picker, "Alpha Novel")
        row, control = picker.book_cards[alpha_t.key]
        picker.toggle(alpha_t)
        assert row.model.selected and picker.book_cards[alpha_t.key][1] is control
        assert picker.done_button.content == "Use 1" and done == []
        picker._on_done()
        assert [t.key for t in done[0]] == [alpha_t.key]

    asyncio.run(scenario())


@needs_flet
def test_large_library_picker_pages_its_rows_and_long_press_selects(tmp_path):
    """A large Library: book rows are built a page at a time (the Library's page size, Library home's
    paging helpers), never all on the UI loop; a long-press starts a multi-selection that keeps the
    built rows (the user's scroll position); a search starts again at its first page."""
    from glossarion_mobile.ui.library.models import DEFAULT_PAGE_SIZE

    lib = library(tmp_path)
    service = lib.service
    many = [{"name": f"Book {n:03d}", "type": "epub", "output_folder": "", "path": "",
             "raw_source_path": _file(tmp_path / "Library" / "Raw" / f"Book {n:03d}.epub"),
             "mtime": 1000.0 + n, "size": 1024, "translation_state": "not_started", "is_in_progress": False}
            for n in range(45)]
    service.set_snapshot(ScanSnapshot(in_progress=tuple(many), completed=(), scanned_at=1.0))
    tb = _tb()

    async def scenario():
        _conn, session = tb._fake_session("android")
        page = session.page
        done: list = []
        picker = _library_picker(_ctx(page, service), chat_rule([]), done, long_press_selects=True).show(page)
        await _until(lambda: picker.loaded)
        assert picker.page_size() == DEFAULT_PAGE_SIZE == 20
        assert _shown(picker) == [f"Book {n:03d}" for n in range(44, 24, -1)]  # newest first, one page
        assert len(picker.book_cards) == 20 and picker.status.value == "45 items"
        # the scroll nears the end: the next page (far from the end: nothing)
        picker._on_scroll(types.SimpleNamespace(pixels=0.0, max_scroll_extent=5000.0))
        assert len(picker.list_view.controls) == 20
        picker._on_scroll(types.SimpleNamespace(pixels=4800.0, max_scroll_extent=5000.0))
        assert len(picker.list_view.controls) == 40 and picker.rendered == 40
        # a long-press on a row of the second page: multi-selection, the 40 built rows stay
        deep = _target(picker, "Book 010")
        row = picker.book_cards[deep.key][0]
        assert not picker.multi and not picker.done_button.visible
        row.on_long_press(row.model)
        assert picker.multi and list(picker.selected) == [deep.key] and row.model.selected
        assert picker.done_button.visible and picker.done_button.content == "Use 1" and picker.sheet.open
        assert len(picker.list_view.controls) == 40
        other = _target(picker, "Book 030")
        picker.book_cards[other.key][0].on_open(picker.book_cards[other.key][0].model)  # a tap adds it
        assert sorted(picker.selected) == sorted([deep.key, other.key]) and picker.done_button.content == "Use 2"
        assert len(picker.list_view.controls) == 40 and picker.sheet.open and done == []
        # a search starts at its own first page; clearing it builds one page again
        picker.set_query("Book 0")
        assert len(picker.list_view.controls) == 20 and picker.status.value == "45 items"
        picker.set_query("Book 01")
        assert _shown(picker) == [f"Book {n:03d}" for n in range(19, 9, -1)]
        picker._clear_query()
        assert len(picker.list_view.controls) == 20
        picker._on_done()
        assert sorted(t.key for t in done[0]) == sorted([deep.key, other.key]) and not picker.sheet.open
        # "All" builds every row (the Library's own "All" page size)
        service.config = {"epub_library_page_size": "all"}
        everything = _library_picker(_ctx(page, service), chat_rule([]), []).show(page)
        await _until(lambda: everything.loaded)
        assert len(everything.list_view.controls) == 45

    asyncio.run(scenario())


# ==========================================================================
# SourcePicker: the Tools defaults
# ==========================================================================


@needs_flet
def test_tools_picker_defaults_are_unchanged(tmp_path):
    lib = library(tmp_path)
    tb = _tb()

    async def scenario():
        from glossarion_mobile.ui.tools.source_picker import SEGMENTS, SourcePicker

        _conn, session = tb._fake_session("android")
        page = session.page
        checked: list = []
        picker = SourcePicker(_ctx(page, lib.service), title="Choose folders to scan", multi=True,
                              eligible=chat_rule(checked), segment="library").show(page)
        await _until(lambda: picker.loaded)
        page.update()
        assert picker.segment_row.visible and [s.value for s in picker.segmented.segments] == [v for v, _l in SEGMENTS]
        assert picker.search_field is None and picker.header_button is None and picker.book_cards == {}
        # plain rows in snapshot order (In progress, then Completed); unresolved rows stay out
        assert [c.key for c in picker.list_view.controls] == [f"pick-{t.key}" for t in tg.library_targets(lib.service)]
        assert [t.title for t in picker.rows["library"]] == ["Alpha Novel", "Beta Saga"]
        assert [t.title for t in picker.rows["recent"]] == ["Alpha Novel"]
        assert picker.status.value == "2 items"
        assert len(checked) == len(set(checked)) == 2  # once per row, in gather
        picker.set_segment("recent")
        picker.set_segment("library")
        assert len(checked) == 2
        picker.set_segment("nonsense")
        assert picker.segment == "recent"
        # an invalid segment list keeps every segment; an unknown first segment falls back to the first
        fallback = SourcePicker(_ctx(page, lib.service), title="x", segments=("nope",), segment="browse")
        assert fallback.segments == tuple(v for v, _l in SEGMENTS) and fallback.segment == "browse"
        assert SourcePicker(_ctx(page, lib.service), title="x", segments=("library", "browse"),
                            segment="recent").segment == "library"

    asyncio.run(scenario())
