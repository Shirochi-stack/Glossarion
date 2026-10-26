"""Hotfix 3 issue 6 (owner, Android): "I want be able to jump from the epub reader directly to epub details,
but I don't see the option for that".

What the owner had: the Reader's only way to the Book page (the desktop Book Details) was "Open Book page", the
third row of the top bar's ⋯ sheet (the top bar fades out 2.5 s after a book opens), and it was disabled
("Not a Library book") for every Reader opened by path - the card ⋯ "📖 Open Translated EPUB", Book page ›
Output, Files › Reader, a chat's raw attachment, Open with - even when that file is a Library book's compiled
EPUB, raw EPUB or the Library EPUB itself (``reader_view.open_more``: ``not self.args.get("path")`` and
``ReaderFeature.has_book_page``).

Now the top bar has ⓘ Book Details (key ``reader-details``) next to search, and ⋯ starts with "Open Book
Details" (the Library card ⋯ / desktop label). Both open the book's Book page over the Reader, so Back returns
to the same page: the Library row the Reader was opened on, else the row a file opened by path belongs to
(``library.feature.library_book_for_path``: its own file / workspace, compiled output, a file in its workspace,
its raw), else an EPUB / TXT / workspace's own Book page (the Library synthesises its row: the EPUB's
details). A file with no Book page (a single HTML / Markdown file) keeps ⓘ visible, dimmed; the tap and the
⋯ row say why. A Library that was never scanned is scanned quietly first. On a tablet the Book page is a
main-area screen, so the full-screen Reader closes first.

The real-app test drives the app on the fake Flet session (real LibraryService, Library scan, Reader, Book
page) like the phone: Reader on a compiled EPUB by path › tap ⓘ › the book's Book page › Back › the same
Reader at the same chapter; a loose EPUB's ⓘ shows that EPUB's details. Real data is never touched (storage,
Library, Output, HOME / USERPROFILE / APPDATA / LOCALAPPDATA under tmp_path; GLOSSARION_HTTP_LOG=0).

Run from src/mobile (mobile venv):
    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_hotfix3_reader_details.py
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
import sys
import time
import types
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


needs_flet = pytest.mark.skipif(not _has("flet"), reason="flet not installed")
needs_session = pytest.mark.skipif(not (_has("flet") and _has("msgpack")), reason="flet / msgpack not installed")

_TB_SPEC = importlib.util.spec_from_file_location("_glossarion_tb_helpers_hotfix3_reader",
                                                  Path(__file__).with_name("test_bootstrap.py"))
_TB = importlib.util.module_from_spec(_TB_SPEC)
_TB_SPEC.loader.exec_module(_TB)
storage = _TB.storage
app_env = _TB.app_env


def _norm(path) -> str:
    return os.path.normcase(os.path.normpath(os.path.abspath(str(path))))


async def _wait(predicate, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.02)
    return bool(predicate())


def _isolate(monkeypatch, tmp_path: Path) -> None:
    home = tmp_path / "winhome"
    for sub in ("", "AppData/Roaming", "AppData/Local"):
        (home / sub).mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("APPDATA", str(home / "AppData" / "Roaming"))
    monkeypatch.setenv("LOCALAPPDATA", str(home / "AppData" / "Local"))
    monkeypatch.setenv("GLOSSARION_HTTP_LOG", "0")


# =====================================================================================
# Books on disk: a Library with an in-progress workspace (raw + compiled EPUB inside it), a Completed compiled
# workspace, an organized Library/Translated EPUB, a look-alike sibling folder, and loose files
# =====================================================================================


def _books(root: Path) -> types.SimpleNamespace:
    from glossarion_mobile.diagnostics.fixtures import build_tiny_epub

    library, output, inbox = root / "Library", root / "Output", root / "Inbox"
    golf_raw = Path(build_tiny_epub(library / "Raw" / "Golf.epub", chapters=3))
    golf_ws = output / "Golf"
    golf_ws.mkdir(parents=True)
    (golf_ws / "translation_progress.json").write_text('{"chapters": {}, "chapter_chunks": {}}', encoding="utf-8")
    golf_compiled = Path(build_tiny_epub(golf_ws / "Golf.epub", chapters=3))
    golf_chapter = golf_ws / "response_chapter0001.html"
    golf_chapter.write_text("<html><body><p>One</p></body></html>", encoding="utf-8")
    sibling = output / "Golf 2"  # shares Golf's name prefix: not inside Golf's workspace
    sibling.mkdir(parents=True)
    sibling_epub = Path(build_tiny_epub(sibling / "Golf 2.epub", chapters=2))
    alpha_ws = output / "Alpha"
    alpha_ws.mkdir(parents=True)
    (alpha_ws / "translation_progress.json").write_text('{"chapters": {}, "chapter_chunks": {}}', encoding="utf-8")
    alpha_epub = Path(build_tiny_epub(alpha_ws / "Alpha.epub", chapters=2))
    echo_epub = Path(build_tiny_epub(library / "Translated" / "Echo.epub", chapters=2))
    echo_ws = output / "Echo"
    echo_ws.mkdir(parents=True)
    echo_chapter = echo_ws / "response_c1.html"
    echo_chapter.write_text("<html><body><p>E</p></body></html>", encoding="utf-8")
    loose_epub = Path(build_tiny_epub(inbox / "Loose.epub", chapters=2))
    loose_txt = inbox / "Loose notes.txt"
    loose_txt.write_text("First line.\n\nSecond paragraph.\n", encoding="utf-8")
    loose_md = inbox / "notes.md"
    loose_md.write_text("# Notes\n\nSome text.\n", encoding="utf-8")
    golf = {"name": "Golf", "path": str(golf_ws), "output_folder": str(golf_ws), "type": "in_progress",
            "is_in_progress": True, "raw_source_path": str(golf_raw), "compiled_output_path": str(golf_compiled)}
    alpha = {"name": "Alpha", "path": str(alpha_epub), "output_folder": str(alpha_ws), "type": "epub",
             "is_in_progress": False}
    echo = {"name": "Echo", "path": str(echo_epub), "type": "epub", "in_library": True, "is_in_progress": False}
    return types.SimpleNamespace(**{k: v for k, v in locals().items() if k not in ("root", "build_tiny_epub")})


def _service(tmp_path: Path, books, *, scanned: bool = True):
    from glossarion_mobile.services.library import LibraryService, ScanSnapshot, book_key
    from glossarion_mobile.state.prefs import Prefs

    prefs = Prefs(tmp_path / "mobile_state.json", debounce=0.01)
    prefs.load()
    service = LibraryService(prefs=prefs)
    snapshot = ScanSnapshot(in_progress=(books.golf,), completed=(books.alpha, books.echo),
                            workspaces={book_key(books.echo): str(books.echo_ws)}, scanned_at=1.0)
    if scanned:
        service.set_snapshot(snapshot)
    service.test_snapshot = snapshot
    return service, prefs


def test_library_book_for_path_finds_the_book_a_file_belongs_to(tmp_path):
    from glossarion_mobile.ui.library.feature import library_book_for_path

    books = _books(tmp_path)
    service, _prefs = _service(tmp_path, books)

    def name(path):
        found = library_book_for_path(service, str(path))
        return found.get("name") if found else None

    assert name(books.golf_ws) == "Golf"  # its own workspace
    assert name(books.golf_compiled) == "Golf"  # its compiled EPUB (Library card ⋯ "📖 Open Translated EPUB")
    assert name(books.golf_chapter) == "Golf"  # a file inside its workspace
    assert name(books.golf_raw) == "Golf"  # its raw EPUB (a chat's raw attachment)
    assert name(books.alpha_epub) == "Alpha"  # a Completed row's own file wins over the workspace match
    assert name(books.echo_epub) == "Echo"  # a Library/Translated EPUB
    assert name(books.echo_chapter) == "Echo"  # inside the workspace the scan resolved for it
    assert name(books.sibling_epub) is None  # "Golf 2" only shares Golf's name
    assert name(books.loose_epub) is None and name("") is None
    assert library_book_for_path(None, str(books.golf_ws)) is None


def test_reader_feature_finds_the_book_page_of_every_reader_target(tmp_path):
    from glossarion_mobile.ui.reader.feature import (DETAILS_NO_LIBRARY, DETAILS_NOT_A_BOOK, ReaderFeature)

    books = _books(tmp_path)
    service, prefs = _service(tmp_path, books)
    app = types.SimpleNamespace(page=None, shell=None, prefs=prefs, library=service, navigate_to=None)
    feature = ReaderFeature(app)
    golf_bid = service.bid_for(books.golf)

    def details(target, bid="ab12cd34ef56"):
        return feature.book_details_blocking(bid, target)

    assert details({"book": books.golf}) == (golf_bid, "")
    for path in (books.golf_compiled, books.golf_raw, books.golf_chapter, books.golf_ws):
        assert details({"path": str(path)}) == (golf_bid, ""), path
    assert details({"path": str(books.alpha_epub)}) == (service.bid_for(books.alpha), "")
    # not in the Library: an EPUB / TXT has its own Book page (the row the Library makes up for its route id)
    for path, kind in ((books.loose_epub, "epub"), (books.loose_txt, "txt")):
        bid, reason = details({"path": str(path)})
        assert bid and reason == ""
        assert bid == prefs.file_ref(str(path), kind="book")  # the Reader's own route id for that file
        row = service.book_for_bid(bid)
        assert row["type"] == kind and _norm(row["path"]) == _norm(path) and not row.get("in_library")
    # a row the Library synthesised from a file's route id (a deep link) still finds the file's book
    synthesised = service._synthesise(str(books.golf_compiled))
    assert synthesised.get("synthesised") and details({"book": synthesised}) == (golf_bid, "")
    assert details({"path": str(books.loose_md)}) == (None, DETAILS_NOT_A_BOOK)
    assert details({"path": str(tmp_path / "gone.epub")})[0] is None
    # the route id alone (no target): the Reader's resolve
    bid = feature.open_book(path=str(books.golf_compiled))
    assert feature.book_details_blocking(bid, None) == (golf_bid, "")
    assert ReaderFeature(types.SimpleNamespace(page=None, shell=None)).book_details_blocking(
        "ab12cd34ef56", {"path": str(books.golf_compiled)}) == (None, DETAILS_NO_LIBRARY)


def test_a_library_that_was_never_scanned_is_scanned_quietly_once(tmp_path):
    from glossarion_mobile.ui.reader.feature import ReaderFeature

    books = _books(tmp_path)
    service, prefs = _service(tmp_path, books, scanned=False)
    scans = []

    async def refresh(*, quiet=False, reason=""):
        scans.append((quiet, reason))
        service.set_snapshot(service.test_snapshot)
        return service.snapshot

    service.refresh = refresh
    feature = ReaderFeature(types.SimpleNamespace(page=None, shell=None, prefs=prefs, library=service))
    golf_bid = service.bid_for(books.golf)

    async def scenario():
        assert await feature.book_details("ab12cd34ef56", {"book": books.golf}) == (golf_bid, "")
        assert scans == []  # a Library row needs no scan
        assert await feature.book_details("ab12cd34ef56", {"path": str(books.golf_compiled)}) == (golf_bid, "")
        assert await feature.book_details("ab12cd34ef56", {"path": str(books.golf_raw)}) == (golf_bid, "")
        assert scans == [(True, "reader")]

    asyncio.run(scenario())


def test_a_scan_another_screen_is_running_is_waited_for(tmp_path):
    """The Library home (or the cloud sync) is scanning when the Reader looks its file up: LibraryService.refresh
    returns None at once, so the lookup reads the running scan's result instead of an empty Library."""
    from glossarion_mobile.ui.reader.feature import ReaderFeature

    books = _books(tmp_path)
    service, prefs = _service(tmp_path, books, scanned=False)
    feature = ReaderFeature(types.SimpleNamespace(page=None, shell=None, prefs=prefs, library=service))
    golf_bid = service.bid_for(books.golf)

    async def scenario():
        service.scanning = True  # the other screen's scan

        async def other_scan_finishes():
            await asyncio.sleep(0.3)
            service.scanning = False
            service.set_snapshot(service.test_snapshot)

        finisher = asyncio.ensure_future(other_scan_finishes())
        started = time.monotonic()
        assert await feature.book_details("ab12cd34ef56", {"path": str(books.golf_compiled)}) == (golf_bid, "")
        assert finisher.done() and time.monotonic() - started < 10

    asyncio.run(scenario())


def test_open_book_details_pushes_the_book_page_and_closes_the_reader_first_on_a_tablet():
    from glossarion_mobile.ui.reader.feature import ReaderFeature

    calls = []
    for tablet, expected in ((False, [("navigate", "library.book", {"bid": "ab12cd34ef56"})]),
                             (True, [("back",), ("navigate", "library.book", {"bid": "ab12cd34ef56"})])):
        calls.clear()
        shell = types.SimpleNamespace(tablet=tablet, stack=[types.SimpleNamespace(fullscreen=True)],
                                      screen_factory=lambda match: None)
        app = types.SimpleNamespace(page=None, shell=shell, back=lambda: calls.append(("back",)),
                                    navigate_to=lambda *a: calls.append(("navigate",) + a))
        ReaderFeature(app).open_book_details("ab12cd34ef56")
        assert calls == expected, tablet


# =====================================================================================
# The Reader screen: ⓘ in the top bar and ⋯ › Open Book Details
# =====================================================================================


@pytest.fixture
def reader_cores(tmp_path, monkeypatch):
    if not all(_has(m) for m in ("reader_doc", "library_core", "reader_overlay", "workspace_reader", "ebooklib")):
        pytest.skip("the shared reader cores are not importable here")
    import reader_doc

    _isolate(monkeypatch, tmp_path)
    monkeypatch.setenv("GLOSSARION_LIBRARY_DIR", str(tmp_path / "Library"))
    monkeypatch.setenv("OUTPUT_DIRECTORY", str(tmp_path / "Output"))
    (tmp_path / "epubcache").mkdir(exist_ok=True)
    monkeypatch.setattr(reader_doc, "_EPUB_CACHE_DIR_OVERRIDE", str(tmp_path / "epubcache"))
    return reader_doc


class _Page(types.SimpleNamespace):
    def __init__(self):
        super().__init__(width=412, height=860, views=[], platform=None, web=False, dialogs=[])

    def show_dialog(self, dialog):
        self.dialogs.append(dialog)

    def pop_dialog(self):
        return None


def _reader_app(tmp_path, books, *, scanned=True):
    from glossarion_mobile.ui.reader.feature import ReaderFeature

    service, prefs = _service(tmp_path, books, scanned=scanned)
    navigated = []
    app = types.SimpleNamespace(page=_Page(), shell=None, prefs=prefs, library=service, paths=None, state=None,
                                navigate_to=lambda *a: navigated.append(a), notify=lambda *a, **k: None)
    feature = ReaderFeature(app)
    feature.attach()
    return types.SimpleNamespace(app=app, feature=feature, service=service, navigated=navigated)


def _top_keys(screen) -> list:
    row = screen.chrome.top.content.content.controls[0]
    return [getattr(c, "key", None) for c in row.controls]


async def _open_screen(env, book=None, path=None):
    from glossarion_mobile.ui.router import parse_route

    bid = env.feature.open_book(book, path=path)
    screen = env.feature.make_screen(parse_route(f"/reader/{bid}"))
    await screen.open()
    assert screen.state == "ready", screen.state
    assert await _wait(lambda: screen.details is not None, 15), "the Book page lookup never finished"
    assert env.navigated[-1] == ("reader", {"bid": bid}, None)
    env.navigated.clear()
    return screen


@needs_flet
def test_reader_top_bar_and_menu_open_the_book_details_of_a_file_opened_by_path(tmp_path, reader_cores):
    """The owner's case: the Reader on a Library book's compiled EPUB, opened by path (the card ⋯ "📖 Open
    Translated EPUB", Book page › Output). Before the fix: no top-bar control, and ⋯ › "Open Book page" was
    disabled "Not a Library book"."""
    from glossarion_mobile.ui.reader.chrome import DETAILS_TOOLTIP
    from glossarion_mobile.ui.reader.reader_view import DETAILS_LABEL

    books = _books(tmp_path)
    env = _reader_app(tmp_path, books)
    golf_bid = env.service.bid_for(books.golf)

    async def scenario():
        screen = await _open_screen(env, path=str(books.golf_compiled))
        assert screen.bid != golf_bid and screen.details == (golf_bid, "")
        # ⓘ sits in the top bar, between the titles and search, at full strength
        keys = _top_keys(screen)
        assert keys[0] == "reader-back" and keys[2:] == ["reader-details", "reader-search", "reader-more"], keys
        button = screen.chrome.details_button
        assert button.tooltip == DETAILS_TOOLTIP and button.icon_color is None and not button.disabled
        # the tap: the book's Book page (its Overview: no tab), the reading position saved first
        env.navigated.clear()
        button.on_click(None)
        assert await _wait(lambda: env.navigated, 5)
        assert env.navigated == [("library.book", {"bid": golf_bid})]
        assert env.app.prefs.reader_position(screen.bid) is not None
        # ⋯ starts with "Open Book Details" (the Library card ⋯ label), available
        menu = screen.open_more()
        first = menu.items[0]
        assert first.label == DETAILS_LABEL and first.disabled_reason is None
        assert "Open Book page" not in [item.label for item in menu.items]
        env.navigated.clear()
        menu._on_select(None, first)
        assert await _wait(lambda: env.navigated, 5)
        assert env.navigated == [("library.book", {"bid": golf_bid})]
        screen.dispose()

        # a Library row (the card's ▶ Continue, the Book page's Start reading): its own Book page
        screen = await _open_screen(env, book=books.golf)
        assert screen.bid == golf_bid and screen.details == (golf_bid, "")
        screen.dispose()

        # a loose EPUB (Open with / Files): its own Book page, the EPUB's details
        screen = await _open_screen(env, path=str(books.loose_epub))
        assert screen.details == (screen.bid, "")
        assert env.service.book_for_bid(screen.bid)["name"] == "Loose"
        screen.dispose()

    asyncio.run(scenario())


@needs_flet
def test_a_file_without_a_book_page_keeps_the_button_and_says_why(tmp_path, reader_cores):
    from glossarion_mobile.ui.reader.chrome import DETAILS_TOOLTIP
    from glossarion_mobile.ui.reader.feature import DETAILS_NOT_A_BOOK

    books = _books(tmp_path)
    env = _reader_app(tmp_path, books)

    async def scenario():
        screen = await _open_screen(env, path=str(books.loose_md))
        assert screen.details == (None, DETAILS_NOT_A_BOOK)
        button = screen.chrome.details_button
        assert "reader-details" in _top_keys(screen) and not button.disabled  # visible, tappable
        assert button.tooltip == DETAILS_NOT_A_BOOK and button.icon_color is not None  # dimmed
        menu = screen.open_more()
        assert menu.items[0].disabled_reason == DETAILS_NOT_A_BOOK
        shown = len(env.app.page.dialogs)
        assert await screen.open_book_details() is None
        assert env.navigated == []
        sheet = env.app.page.dialogs[-1]
        assert len(env.app.page.dialogs) == shown + 1 and DETAILS_TOOLTIP in str(_texts(sheet))
        assert DETAILS_NOT_A_BOOK in str(_texts(sheet))
        screen.dispose()

    asyncio.run(scenario())


@needs_flet
def test_a_tap_before_the_lookup_finished_waits_for_it(tmp_path, reader_cores):
    """ⓘ right after the book opened, while an unscanned Library is still being scanned: the tap waits for
    the lookup (no second scan) and opens the book's Book page."""
    books = _books(tmp_path)
    env = _reader_app(tmp_path, books, scanned=False)
    gate = asyncio.Event
    golf_bid = env.service.bid_for(books.golf)
    state = {"scans": 0}

    async def scenario():
        release = gate()

        async def refresh(*, quiet=False, reason=""):
            state["scans"] += 1
            await release.wait()  # the scan is held until the tap is waiting
            env.service.set_snapshot(env.service.test_snapshot)
            return env.service.snapshot

        env.service.refresh = refresh
        from glossarion_mobile.ui.router import parse_route

        bid = env.feature.open_book(path=str(books.golf_compiled))
        screen = env.feature.make_screen(parse_route(f"/reader/{bid}"))
        await screen.open()
        env.navigated.clear()  # the Reader's own route
        assert await _wait(lambda: state["scans"] == 1, 5)
        assert screen.details is None  # still looking: the ⋯ row stays available
        assert screen.open_more().items[0].disabled_reason is None
        tap = asyncio.ensure_future(screen.open_book_details())
        await asyncio.sleep(0.1)
        assert not tap.done() and env.navigated == []
        release.set()
        assert await asyncio.wait_for(tap, 10) == golf_bid
        assert env.navigated == [("library.book", {"bid": golf_bid})] and state["scans"] == 1
        screen.dispose()

    asyncio.run(scenario())


def _texts(control, out=None) -> list:
    out = [] if out is None else out
    if control is None:
        return out
    for name in ("value", "title", "content"):
        value = getattr(control, name, None)
        if isinstance(value, str):
            out.append(value)
    for name in ("content", "controls", "title", "actions"):
        child = getattr(control, name, None)
        for item in (child if isinstance(child, list) else [child] if child is not None else []):
            if hasattr(item, "__dict__") and not isinstance(item, str):
                _texts(item, out)
    return out


# =====================================================================================
# The real app on the fake Flet session
# =====================================================================================


async def _show_chrome(uf, reader) -> None:
    """The bars after the Reader's auto-hide (patched to 0 s, so it cannot fire between this and a tap): a
    centre tap shows them again; past the fade's late hide."""
    assert await uf._wait(lambda: not reader.chrome.visible, timeout=10), "the bars never auto-hid"
    reader.chrome.set_visible(True)
    await asyncio.sleep(0.25)
    assert reader.chrome.visible and reader.chrome.top.visible


def _no_auto_hide_delay(monkeypatch) -> None:
    from glossarion_mobile.ui.reader import reader_view

    monkeypatch.setattr(reader_view, "CHROME_AUTO_HIDE", 0.0)


def _foundations():
    spec = importlib.util.spec_from_file_location("_glossarion_uf_helpers_hotfix3_reader",
                                                  Path(__file__).with_name("test_ui_foundations.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _devfix5():
    """test_devfix_issue5's Library fixtures (real EPUB workspaces the shared scanner reads)."""
    spec = importlib.util.spec_from_file_location("_glossarion_devfix5_helpers_hotfix3_reader",
                                                  Path(__file__).with_name("test_devfix_issue5.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@needs_session
def test_reader_book_details_in_the_real_app_and_back_returns_to_the_same_page(app_env, tmp_path, monkeypatch):
    if not all(_has(m) for m in ("library_core", "reader_doc", "progress_core", "ebooklib")):
        pytest.skip("the shared Library / Reader cores are not importable here")
    import library_core as lc

    _isolate(monkeypatch, tmp_path)
    _no_auto_hide_delay(monkeypatch)
    uf = _foundations()
    d5 = _devfix5()
    from host_tester import PyTester
    from ui_driver import UiDriver

    from glossarion_mobile.ui.library.book_page import BookPageScreen
    from glossarion_mobile.ui.reader.reader_view import DETAILS_LABEL, ReaderScreen

    async def scenario():
        _m, conn, session, page, app = await uf._start("android", width=412)
        try:
            paths = app.paths
            assert _norm(tmp_path) in _norm(paths.library) and _norm(tmp_path) in _norm(paths.output)
            assert await uf._wait(lambda: lc.current_library_env() is not None, timeout=15)
            output, library = Path(paths.output), Path(paths.library)
            alpha_ws = d5._epub_workspace(output, library, "Alpha")  # finished: Output/Alpha/Alpha.epub
            loose = Path(d5.make_epub(tmp_path / "Shared" / "Loose Story.epub", d5.CHAPTERS[:2],
                                      title="Loose Story Title"))
            service = app.library  # the Library screen was never opened: the Reader's lookup scans it (first use)
            tester = PyTester(session, page)
            driver = UiDriver(tester, poll_ms=50, log=lambda *_a: None)

            async def open_reader(path) -> ReaderScreen:
                app.reader.open_book(path=str(path))  # Files › Reader / Book page › Output / Open with
                assert await uf._wait(lambda: isinstance(app.shell.top_screen, ReaderScreen), timeout=15)
                reader = app.shell.top_screen
                assert await uf._wait(lambda: reader.state == "ready", timeout=30), reader.state
                assert await uf._wait(lambda: reader.details is not None, timeout=30), "no Book page lookup"
                await _show_chrome(uf, reader)
                return reader

            # ---- the compiled EPUB of a Library book, opened by path -------------------------------------
            reader = await open_reader(alpha_ws / "Alpha.epub")
            alpha = next(b for b in service.snapshot.all_books() if _norm(b.get("output_folder") or "") == _norm(alpha_ws))
            alpha_bid = service.bid_for(alpha)
            assert reader.details == (alpha_bid, "")
            await reader.go_chapter(1)
            assert await uf._wait(lambda: reader.index == 1, timeout=10)
            reader_routes = [e.route for e in app.shell.stack]
            await driver.tap(key="reader-details", timeout=15)  # a real click through the Flet session
            assert await uf._wait(lambda: isinstance(app.shell.top_screen, BookPageScreen), timeout=15), \
                [e.route for e in app.shell.stack]
            book_page = app.shell.top_screen
            assert book_page.bid == alpha_bid and book_page.initial_tab == "overview"
            assert [e.route for e in app.shell.stack] == reader_routes + [f"/library/book/{alpha_bid}"]
            assert await uf._wait(lambda: book_page.details_phase == "full", timeout=30)
            assert not reader.disposed  # kept under the Book page
            # Back: the same Reader, at the same chapter
            await d5._back_once(app, session, page)
            assert await uf._wait(lambda: app.shell.top_screen is reader, timeout=10), [e.route for e in app.shell.stack]
            assert not reader.disposed and reader.index == 1 and reader.state == "ready"
            # ⋯ › Open Book Details (the sheet's first row) does the same
            assert reader.chrome.visible
            await driver.tap(key="reader-more", timeout=15)
            await driver.tap(text=DETAILS_LABEL, timeout=15)
            assert await uf._wait(lambda: isinstance(app.shell.top_screen, BookPageScreen)
                                  and app.shell.top_screen.bid == alpha_bid, timeout=15), \
                [e.route for e in app.shell.stack]
            assert [e.route for e in app.shell.stack] == reader_routes + [f"/library/book/{alpha_bid}"]
            await d5._back_once(app, session, page)
            assert await uf._wait(lambda: app.shell.top_screen is reader, timeout=10), [e.route for e in app.shell.stack]
            assert not reader.disposed and reader.index == 1
            await d5._back_once(app, session, page)
            assert await uf._wait(lambda: reader.disposed, timeout=10)

            # ---- a loose EPUB (no Library book): its own Book page shows that EPUB's details ---------------
            reader = await open_reader(loose)
            assert reader.details == (reader.bid, "")
            await driver.tap(key="reader-details", timeout=15)
            assert await uf._wait(lambda: isinstance(app.shell.top_screen, BookPageScreen), timeout=15)
            book_page = app.shell.top_screen
            assert book_page.bid == reader.bid and _norm(book_page.book["path"]) == _norm(loose)
            assert await uf._wait(lambda: book_page.details_phase == "full", timeout=30)
            assert "Loose Story Title" in str(book_page.overview.title_text.value or ""), \
                book_page.overview.title_text.value
            await d5._back_once(app, session, page)
            assert await uf._wait(lambda: app.shell.top_screen is reader and not reader.disposed, timeout=10)
            await d5._back_once(app, session, page)
            assert await uf._wait(lambda: reader.disposed, timeout=10)

            # ---- Book page › Start reading › ⓘ: the Book page under the Reader is the one shown ----------------
            # (the shell returns to a screen already on the stack, UI_SPEC §1.2; the position is saved)
            app.navigate_to("library.book", {"bid": alpha_bid})
            assert await uf._wait(lambda: isinstance(app.shell.top_screen, BookPageScreen), timeout=15)
            book_page = app.shell.top_screen
            assert await uf._wait(lambda: book_page.details_phase == "full", timeout=30)
            await driver.tap(key="ov-read", timeout=15)
            assert await uf._wait(lambda: isinstance(app.shell.top_screen, ReaderScreen), timeout=15)
            reader = app.shell.top_screen
            assert await uf._wait(lambda: reader.state == "ready" and reader.details is not None, timeout=30)
            assert reader.details == (alpha_bid, "")
            await _show_chrome(uf, reader)
            await driver.tap(key="reader-details", timeout=15)
            assert await uf._wait(lambda: app.shell.top_screen is book_page and reader.disposed, timeout=15), \
                [e.route for e in app.shell.stack]
            assert [e.route for e in app.shell.stack].count(f"/library/book/{alpha_bid}") == 1
            assert app.prefs.reader_position(reader.bid) is not None
        finally:
            await uf._stop(app)

    asyncio.run(scenario())


@needs_session
def test_on_a_tablet_the_reader_makes_way_for_the_book_page(app_env, tmp_path, monkeypatch):
    """1280 dp: the Book page is a main-area screen and the Reader a full-screen View over it, so ⓘ closes the
    Reader (its position saved) and the Book page is what the reader sees - not a page behind the Reader."""
    if not all(_has(m) for m in ("library_core", "reader_doc", "progress_core", "ebooklib")):
        pytest.skip("the shared Library / Reader cores are not importable here")
    import library_core as lc

    _isolate(monkeypatch, tmp_path)
    _no_auto_hide_delay(monkeypatch)
    uf = _foundations()
    d5 = _devfix5()
    from host_tester import PyTester
    from ui_driver import UiDriver

    from glossarion_mobile.ui.library.book_page import BookPageScreen
    from glossarion_mobile.ui.reader.reader_view import ReaderScreen

    async def scenario():
        _m, conn, session, page, app = await uf._start("android", width=1280)
        try:
            assert app.shell.tablet
            assert await uf._wait(lambda: lc.current_library_env() is not None, timeout=15)
            alpha_ws = d5._epub_workspace(Path(app.paths.output), Path(app.paths.library), "Alpha")
            app.reader.open_book(path=str(alpha_ws / "Alpha.epub"))
            assert await uf._wait(lambda: isinstance(app.shell.top_screen, ReaderScreen), timeout=15)
            reader = app.shell.top_screen
            assert await uf._wait(lambda: reader.state == "ready" and reader.details is not None, timeout=30)
            await _show_chrome(uf, reader)
            service = app.library
            alpha = next(b for b in service.snapshot.all_books() if _norm(b.get("output_folder") or "") == _norm(alpha_ws))
            driver = UiDriver(PyTester(session, page), poll_ms=50, log=lambda *_a: None)
            await driver.tap(key="reader-details", timeout=15)
            assert await uf._wait(lambda: isinstance(app.shell.top_screen, BookPageScreen), timeout=15)
            book_page = app.shell.top_screen
            assert book_page.bid == service.bid_for(alpha) and reader.disposed
            assert not any(entry.fullscreen for entry in app.shell.stack)  # nothing covers the main area
            assert app.shell._top_panel_entry().screen is book_page
            assert app.prefs.reader_position(reader.bid) is not None  # Continue reopens where the reader was
        finally:
            await uf._stop(app)

    asyncio.run(scenario())
