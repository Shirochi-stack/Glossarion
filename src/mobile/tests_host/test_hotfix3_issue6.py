"""Acceptance test for hotfix 3, owner issue 6 (Android phone):

    "I want be able to jump from the epub reader directly to epub details, but I don't see the option for that"

What the owner had: the Reader's only way to the Book page (the desktop Book Details) was "Open Book page", a
row in the middle of the top bar's ⋯ sheet, and it was disabled ("Not a Library book") for every Reader opened
by path - a chat attachment, Files › Open with › Reader, the card ⋯ "Open Translated EPUB", a shared EPUB.

This test drives the REAL app (``main.main`` on the fake Flet session of tests_host: the real LibraryService
scanning real workspaces through the shared ``library_core``, the real ReaderFeature / ReaderScreen with its
loopback page server and WebView page, the real Book page, the real chat view) the way the phone does, from
every place the owner can open an EPUB in the Reader:

* the Library: Library › In progress › card ⋯ › "📖 Open in Reader", the card's ▶ Continue, and Library ›
  Completed › card ⋯ › "📖 Open in Reader";
* a completed book's Book page: Library › Completed › tap the card › "Start reading";
* the chat: a picked copy of a Library book's raw EPUB (the chat attaches it as that Library book) › the
  composer's "Open in Reader" chip; the chat's "Open in Reader" for a run's workspace (inside the Library and
  outside it); a picked EPUB that is not in the Library;
* a file path: Tools › Files › Output › <book> › <book>.epub › Open with… › Reader; a shared EPUB / TXT that
  is not in the Library (Open with: ``ReaderFeature.open_file``); a Markdown file.

For each: the Reader is moved to another chapter and page (a page event from the page script, as the WebView
sends it), the bars come back with a centre tap (they fade out after the book opens), and ONE tap on the top
bar's ⓘ "Book Details" (a real click event through the Flet session) shows that book's Book page (Overview,
fully loaded, no error). Android Back (``view_pop`` of the Book page's View) then returns to the SAME Reader
object at the same chapter, page and fraction, with the page not reloaded (no new WebView ``load_request``:
a reload would restart the chapter on the phone) and the position saved. ⋯ › "Open Book Details" does the
same. A Reader opened from the Book page itself returns to that Book page (the shell never stacks a second
copy of a screen; UI_SPEC §1.2), its position saved, and the Book page's Continue reopens it at that place.
A file with no Book page of its own (Markdown) keeps ⓘ visible, dimmed, and its tap says why. Last, on
``test_hotfix3_sheets.FlutterModel`` (the Flet client's view / sheet behaviour, dismiss events included, as on
the phone): ⋯ › Open Book Details leaves no sheet over or under the Book page, Back brings no sheet back, and
the reason sheet closes with Close Menu ✕ and with a swipe.

Real data is never touched: FLET_APP_STORAGE_* (and so HOME, OUTPUT_DIRECTORY, GLOSSARION_LIBRARY_DIR,
GLOSSARION_DATA_DIR, CONFIG_FILE) come from ``app_env`` under ``tmp_path``; USERPROFILE, APPDATA and
LOCALAPPDATA are pointed there too, GLOSSARION_HTTP_LOG=0, and src/config.json must keep its bytes.

Run from src/mobile (mobile venv, Python 3.12):
    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_hotfix3_issue6.py
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import itertools
import os
import sys
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

CONFIG_JSON = SRC_DIR / "config.json"


def _has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


_NEEDED = ("flet", "msgpack", "flet_webview", "library_core", "reader_doc", "progress_core", "workspace_reader",
           "ebooklib", "bs4")
pytestmark = pytest.mark.skipif(not all(_has(m) for m in _NEEDED),
                                reason="flet / flet_webview / the shared Library and Reader cores are not importable")


def _load(alias: str, file_name: str):
    spec = importlib.util.spec_from_file_location(alias, Path(__file__).with_name(file_name))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_TB = _load("_glossarion_tb_hotfix3_issue6", "test_bootstrap.py")
storage = _TB.storage
app_env = _TB.app_env

_HELPERS: dict = {}


def _helpers():
    """test_ui_foundations (app start / stop), test_devfix_issue2 (Run: the app + driver + WebView loads),
    test_devfix_issue5 (real Library workspaces)."""
    if not _HELPERS:
        _HELPERS["uf"] = _load("_glossarion_uf_hotfix3_issue6", "test_ui_foundations.py")
        _HELPERS["d2"] = _load("_glossarion_d2_hotfix3_issue6", "test_devfix_issue2.py")
        _HELPERS["d5"] = _load("_glossarion_d5_hotfix3_issue6", "test_devfix_issue5.py")
    return _HELPERS["uf"], _HELPERS["d2"], _HELPERS["d5"]


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest() if path.is_file() else ""


def _norm(path) -> str:
    return os.path.normcase(os.path.normpath(os.path.abspath(str(path))))


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """Every user folder in tmp_path (the app's bootstrap then moves HOME, the Library, Output and its data
    into its own tmp storage); the Reader's bars hide at once (CHROME_AUTO_HIDE 0: no 2.5 s wait per open, and
    the hide cannot land between the centre tap and the ⓘ tap); src/config.json is never written."""
    before = _md5(CONFIG_JSON)
    for key in ("HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA"):
        folder = tmp_path / "env" / key.lower()
        folder.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv(key, str(folder))
    for key, name in (("GLOSSARION_LIBRARY_DIR", "Library"), ("OUTPUT_DIRECTORY", "Output"),
                      ("GLOSSARION_DATA_DIR", "Data")):
        (tmp_path / "pre" / name).mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv(key, str(tmp_path / "pre" / name))
    monkeypatch.setenv("GLOSSARION_HTTP_LOG", "0")
    from glossarion_mobile.ui.reader import reader_view

    monkeypatch.setattr(reader_view, "CHROME_AUTO_HIDE", 0.0)
    # the app's start patches flet classes for the whole process (the sheets' Close Menu strip, the views
    # guard): undone after the test, as in test_hotfix3_sheets
    import flet as ft
    from flet.controls.base_page import BasePage

    hooks = [(ft.BottomSheet, "before_update", ft.BottomSheet.__dict__.get("before_update")),
             (BasePage, "show_dialog", BasePage.__dict__.get("show_dialog")),
             (BasePage, "_remove_dialog", BasePage.__dict__.get("_remove_dialog"))]
    try:
        yield tmp_path
    finally:
        for cls, name, value in hooks:
            if value is None:
                if name in cls.__dict__:
                    delattr(cls, name)
            else:
                setattr(cls, name, value)
        if _has("library_core"):
            try:
                import library_core

                library_core.uninstall_library_env()
            except Exception:
                pass
        assert _md5(CONFIG_JSON) == before, "src/config.json changed"


# =====================================================================================
# Driving the Reader like the phone
# =====================================================================================

_SEQ = itertools.count(1000)
READ_PAGE, READ_PAGES = 3, 7  # the page the reader turned to in a chapter (paged: fraction 3/6 = 0.5)


def _stack(run) -> list:
    return [entry.route for entry in run.app.shell.stack]


def _reader_loads(run) -> list:
    """Every WebView ``load_request`` of a Reader page the app sent to the client."""
    base = str(run.app.reader.server.base_url)
    return [url for url in run.load_requests() if str(url or "").startswith(base)]


async def _new_reader(run, previous=None):
    """The Reader that the last action opened, once its first page is up (WebView page, ``state == ready``)."""
    assert await run.until(lambda: getattr(run.app.reader, "active", None) is not None
                           and run.app.reader.active is not previous, 30), f"no Reader opened (route {run.route()})"
    reader = await run.reader()
    assert reader is not previous and reader.renderer == "webview", reader.renderer
    assert await run.until(lambda: reader.details is not None, 30), "the Reader never looked its Book page up"
    return reader


async def _read_to(run, reader, chapter: int) -> None:
    """Turn to ``chapter`` (▶ / the slider) and page READ_PAGE there (the page script's ``page`` event)."""
    if reader.index != chapter:
        await reader.go_chapter(chapter)
        assert await run.until(lambda: reader.index == chapter and reader.current_doc, 15), reader.index
    event = reader.handle_payload({"type": "page", "seq": next(_SEQ), "page": READ_PAGE, "count": READ_PAGES,
                                   "fraction": 0.5, "chapter": chapter, "doc": reader.current_doc})
    assert event is not None, "the page event was dropped"
    assert (reader.index, reader.page_no, reader.page_count) == (chapter, READ_PAGE, READ_PAGES)
    assert abs(reader.fraction - 0.5) < 1e-6, reader.fraction


async def _show_bars(run, reader) -> None:
    """The bars fade out once the book is up; a tap in the middle of the page brings them back (the page
    script's ``tap`` event, what the WebView sends for a centre tap)."""
    assert await run.until(lambda: not reader.chrome.visible, 10), "the Reader's bars never auto-hid"
    assert await run.until(lambda: not reader.chrome.top.visible, 5)
    assert await run.driver.count(key="reader-details") == 0  # hidden with the bars, like search and ⋯
    reader.handle_payload({"type": "tap", "seq": next(_SEQ), "zone": "center", "chapter": reader.index,
                           "doc": reader.current_doc})
    assert reader.chrome.visible and reader.chrome.top.visible


def _spot(reader) -> tuple:
    """Where the Reader is and what its page shows (a reload would start a new document)."""
    return (reader.index, reader.page_no, reader.page_count, round(float(reader.fraction), 6), reader.current_doc,
            id(reader.webview), reader.webview_serial, reader.render_generation, reader.session.flavor,
            reader.bid, reader.state)


def _saved(run, reader) -> tuple:
    position = run.app.prefs.reader_position(reader.bid) or {}
    return position.get("chapter"), position.get("page"), round(float(position.get("fraction") or 0.0), 6)


class Visit:
    """One trip Reader › Book page › Back."""

    def __init__(self, run, reader, book_page, spot, loads, stack) -> None:
        self.run, self.reader, self.book_page = run, reader, book_page
        self.spot, self.loads, self.stack = spot, loads, stack


async def _open_details(run, reader, bid: str, *, via: str = "button") -> Visit:
    """ONE tap (``via`` "button": ⓘ in the top bar; "menu": ⋯ › Open Book Details) shows the Book page ``bid``
    over the Reader, fully loaded, on its Overview."""
    from glossarion_mobile.ui.library.book_page import BookPageScreen
    from glossarion_mobile.ui.reader.chrome import DETAILS_TOOLTIP
    from glossarion_mobile.ui.reader.reader_view import DETAILS_LABEL

    assert reader.details == (bid, ""), (reader.details, bid)
    stack, loads, spot = _stack(run), len(_reader_loads(run)), _spot(reader)
    button = reader.chrome.details_button
    assert button is not None and button.tooltip == DETAILS_TOOLTIP and button.icon_color is None
    assert await run.driver.count(key="reader-details") == 1  # on screen, in the top bar
    assert await run.driver.count(tooltip=DETAILS_TOOLTIP) == 1  # labelled "Book Details"
    if via == "button":
        await run.driver.tap(key="reader-details", timeout=15)
    else:
        await run.driver.tap(key="reader-more", timeout=15)
        await run.driver.wait(text=DETAILS_LABEL, timeout=15)
        await run.driver.tap(key="reader-more-details", timeout=15)
    assert await run.until(lambda: isinstance(run.app.shell.top_screen, BookPageScreen)
                           and run.app.shell.top_screen.bid == bid, 15), (_stack(run), bid)
    book_page = run.app.shell.top_screen
    assert _stack(run) == stack + [f"/library/book/{bid}"], _stack(run)  # pushed over the Reader
    assert run.route() == f"/library/book/{bid}"  # the client shows it (phone: its own View)
    assert book_page.initial_tab == "overview"
    assert await run.until(lambda: book_page.details is not None and book_page.details_phase == "full", 30)
    assert not book_page.overview.error_text.visible, book_page.overview.error_text.value
    assert not reader.disposed and _spot(reader) == spot  # the Reader waits under it, untouched
    assert _saved(run, reader) == (spot[0], READ_PAGE, 0.5)  # the reading position was saved first
    return Visit(run, reader, book_page, spot, loads, stack)


async def _back_to_reader(visit: Visit) -> None:
    """Android Back on the Book page: the same Reader, at the same chapter / page, its page not reloaded."""
    run, reader = visit.run, visit.reader
    await run.back()
    assert await run.until(lambda: run.app.shell.top_screen is reader, 10), _stack(run)
    assert _stack(run) == visit.stack and run.route() == visit.stack[-1]
    assert not reader.disposed and _spot(reader) == visit.spot, (_spot(reader), visit.spot)
    assert len(_reader_loads(run)) == visit.loads, "the Reader's page was reloaded (the chapter would restart)"
    assert _saved(run, reader) == (visit.spot[0], READ_PAGE, 0.5)
    assert reader.chrome.visible  # the bars as they were: ⋯ / ⓘ one tap away again


async def _round_trip(run, reader, bid: str, chapter: int, *, both: bool = True):
    """Read on, ⓘ › the Book page › Back; then ⋯ › Open Book Details › Back. Returns the last Book page."""
    await _read_to(run, reader, chapter)
    await _show_bars(run, reader)
    visit = await _open_details(run, reader, bid)
    book_page = visit.book_page
    await _back_to_reader(visit)
    if both:
        visit = await _open_details(run, reader, bid, via="menu")
        await _back_to_reader(visit)
    return book_page


async def _leave(run, reader) -> None:
    await run.back()
    assert await run.until(lambda: reader.disposed, 10), _stack(run)


def _row(service, folder) -> dict:
    found = [b for b in service.snapshot.all_books() if _norm(b.get("output_folder") or "") == _norm(folder)]
    assert len(found) == 1, [(b.get("name"), b.get("type"), b.get("output_folder")) for b in service.snapshot.all_books()]
    return dict(found[0])


async def _seed(run, d5) -> dict:
    """A Completed book (Output/Alpha with its compiled Alpha.epub) and an In-progress one (Output/Golf: 2 of
    4 chapters), both with their raw EPUB in Library/Raw, scanned by the app's Library."""
    import library_core as lc

    app = run.app
    assert await run.until(lambda: lc.current_library_env() is not None, 15)
    output, library = Path(app.paths.output), Path(app.paths.library)
    tmp = _norm(os.environ.get("FLET_APP_STORAGE_DATA", ""))
    assert tmp and _norm(output).startswith(os.path.dirname(tmp)) and _norm(library).startswith(os.path.dirname(tmp))
    alpha_ws = d5._epub_workspace(output, library, "Alpha")
    golf_ws = d5._epub_workspace(output, library, "Golf", done=2)
    service = app.library
    await d5._fresh_scan(service)
    alpha, golf = _row(service, alpha_ws), _row(service, golf_ws)
    assert alpha in [dict(b) for b in service.snapshot.completed], alpha
    assert golf in [dict(b) for b in service.snapshot.in_progress], golf
    return {"service": service, "alpha_ws": alpha_ws, "golf_ws": golf_ws, "alpha": alpha, "golf": golf,
            "alpha_bid": service.bid_for(alpha), "golf_bid": service.bid_for(golf),
            "alpha_raw": library / "Raw" / "Alpha.epub", "golf_raw": library / "Raw" / "Golf.epub"}


def _run_app(scenario) -> None:
    uf, d2, d5 = _helpers()

    async def main():
        run = await d2.start_app(uf, {})
        try:
            await scenario(run, d2, d5)
        finally:
            await d2.stop_app(run)

    asyncio.run(main())


# =====================================================================================
# 1. From the Library and from a completed book
# =====================================================================================


def test_library_and_completed_book_readers_jump_to_the_book_page_and_back(app_env, tmp_path):
    from glossarion_mobile.services.library import book_key
    from glossarion_mobile.ui.library.book_page import BookPageScreen
    from glossarion_mobile.ui.library.home import LibraryScreen

    async def scenario(run, d2, d5):
        app = run.app
        seeds = await _seed(run, d5)
        golf, alpha, golf_bid, alpha_bid = seeds["golf"], seeds["alpha"], seeds["golf_bid"], seeds["alpha_bid"]

        # ---- Library › In progress › Golf ⋯ › 📖 Open in Reader (the Library home stays under it) ----------
        await d2.open_library(run)
        home = app.shell.top_screen
        assert isinstance(home, LibraryScreen)
        home.set_shelf("progress")
        items = {item.label: item for item in home.card_actions(golf).items}
        assert items["\U0001f4d1 Open Book Details"].disabled_reason is None  # the card's own way there
        items["\U0001f4d6 Open in Reader"].on_select()
        reader = await _new_reader(run)
        assert reader.bid == golf_bid and _stack(run) == ["/library", f"/reader/{golf_bid}"], _stack(run)
        book_page = await _round_trip(run, reader, golf_bid, 2)
        assert _norm(book_page.workspace) == _norm(seeds["golf_ws"])
        await _leave(run, reader)
        assert app.shell.top_screen is home

        # ---- the card's ▶ Continue (the saved position): the Reader at Ch 3, page 3/7 ------------------------
        await d5._fresh_scan(seeds["service"])
        assert await run.until(lambda: book_key(golf) in home.cards and home.cards[book_key(golf)].model.has_continue,
                               15), "the Golf card offers no ▶ Continue"
        previous = reader
        home._on_continue(home.cards[book_key(golf)].model)  # the ▶ button's handler
        reader = await _new_reader(run, previous)
        assert reader.index == 2 and str(reader.webview.url).endswith("#f=0.5"), (reader.index, reader.webview.url)
        await _round_trip(run, reader, golf_bid, 3, both=False)
        await _leave(run, reader)

        # ---- Library › Completed › Alpha ⋯ › 📖 Open in Reader --------------------------------------------
        home.set_shelf("completed")
        previous = reader
        {item.label: item for item in home.card_actions(alpha).items}["\U0001f4d6 Open in Reader"].on_select()
        reader = await _new_reader(run, previous)
        assert reader.bid == alpha_bid and _stack(run) == ["/library", f"/reader/{alpha_bid}"], _stack(run)
        book_page = await _round_trip(run, reader, alpha_bid, 1)
        assert _norm(book_page.workspace) == _norm(seeds["alpha_ws"])
        await _leave(run, reader)

        # ---- a completed book's Book page › Start reading › ⓘ: back on THAT Book page --------------------
        book_page, bid = await d2.open_book_page(run, alpha, shelf="completed")
        assert bid == alpha_bid
        previous = reader
        await run.driver.tap(key="ov-read", timeout=15)
        reader = await _new_reader(run, previous)
        assert reader.bid == alpha_bid and reader.index == 1  # where the last visit left the book
        assert _stack(run) == ["/library", f"/library/book/{alpha_bid}", f"/reader/{alpha_bid}?mode=translated"]
        await _read_to(run, reader, 3)
        await _show_bars(run, reader)
        await run.driver.tap(key="reader-details", timeout=15)
        assert await run.until(lambda: app.shell.top_screen is book_page and reader.disposed, 15), _stack(run)
        assert _stack(run) == ["/library", f"/library/book/{alpha_bid}"]  # one Book page, never two copies
        assert _saved(run, reader) == (3, READ_PAGE, 0.5)
        # its Continue reopens the Reader where it was
        assert await run.until(lambda: book_page.overview.continue_button.visible
                               and "Ch 4" in str(book_page.overview.continue_button.content or ""), 15), \
            book_page.overview.continue_button.content
        previous = reader
        await run.driver.tap(key="ov-continue", timeout=15)
        reader = await _new_reader(run, previous)
        assert reader.index == 3 and str(reader.webview.url).endswith("#f=0.5"), (reader.index, reader.webview.url)
        assert isinstance(app.shell.stack[-2].screen, BookPageScreen)
        await _leave(run, reader)

    _run_app(scenario)


# =====================================================================================
# 2. From the chat
# =====================================================================================


def test_chat_attachment_readers_jump_to_the_book_page_and_back(app_env, tmp_path):
    from glossarion_mobile.ui.chat.chat_view import LIBRARY_ATTACHMENT_META

    async def scenario(run, d2, d5):
        app = run.app
        assert await run.until(lambda: getattr(app, "chat_feature", None) is not None
                               and getattr(app, "chat_view", None) is not None, 30)
        seeds = await _seed(run, d5)
        service, golf_bid = seeds["service"], seeds["golf_bid"]
        chats = app.chat_feature.chats
        cid = chats.new_chat()
        app._open_chat(cid)
        assert await run.until(lambda: app.chat_view.cid == cid, 10)
        view = app.chat_view

        async def chip_reader(previous):
            """The composer's "Open in Reader" chip (a click through the session)."""
            path = str((view.composer.attachment or {}).get("path") or "")
            assert await run.until(lambda: view.quick_chips.signature and view.quick_chips.signature[0] == path
                                   and "open_reader" in view.quick_chips.ids, 10), view.quick_chips.signature
            await run.driver.tap(key=f"quick-open_reader-{view.quick_chips._gen}", timeout=15)
            return await _new_reader(run, previous)

        # ---- ＋ › Files picks a copy of Golf's raw EPUB: the chat attaches it as the Library book ----------
        picked = app.files.import_paths([str(seeds["golf_raw"])], names=["Golf.epub"])[0].path
        assert _norm(picked) != _norm(seeds["golf_raw"])
        assert view.attach_file(picked)
        assert await run.until(lambda: _norm((view.composer.attachment or {}).get("path") or "")
                               == _norm(seeds["golf_raw"]), 30), view.composer.attachment
        assert (chats.meta(cid).get(LIBRARY_ATTACHMENT_META) or {}).get("bid") == golf_bid
        reader = await chip_reader(None)
        assert _norm(reader._target_path) == _norm(seeds["golf_raw"]) and reader.bid != golf_bid  # by path
        assert _stack(run) == [f"/reader/{reader.bid}"]  # over the chat
        await _round_trip(run, reader, golf_bid, 2)
        await _leave(run, reader)
        assert app.shell.stack == [] and app.chat_view.cid == cid  # back in the chat

        # ---- the chat's "Open in Reader" of a run's workspace (a Library book's) ------------------------------
        previous = reader
        await app.chat_feature.open_reader(str(seeds["golf_ws"]), str(seeds["golf_raw"]))
        reader = await _new_reader(run, previous)
        assert reader.bid == golf_bid
        book_page = await _round_trip(run, reader, golf_bid, 1, both=False)
        assert _norm(book_page.book.get("raw_source_path") or "") == _norm(seeds["golf_raw"])  # the Library's row
        await _leave(run, reader)

        # ---- a chat run's workspace outside the Library: that workspace's own Book page ---------------------
        kilo_ws = d5._epub_workspace(tmp_path / "ChatRuns", tmp_path / "ChatPicks", "Kilo", done=2)
        kilo_raw = tmp_path / "ChatPicks" / "Raw" / "Kilo.epub"
        previous = reader
        await app.chat_feature.open_reader(str(kilo_ws), str(kilo_raw))
        reader = await _new_reader(run, previous)
        kilo_bid = reader.details[0]
        assert kilo_bid and reader.details == (kilo_bid, "") and kilo_bid == reader.bid
        book_page = await _round_trip(run, reader, kilo_bid, 1, both=False)
        assert _norm(book_page.book.get("output_folder") or "") == _norm(kilo_ws) and book_page.title == "Kilo"
        await _leave(run, reader)

        # ---- a picked EPUB that is no Library book: the attachment's own Book page (its EPUB details) --------
        stranger = Path(d5.make_epub(tmp_path / "Picks" / "Stranger.epub", d5.CHAPTERS[:3], title="Stranger Title"))
        picked = app.files.import_paths([str(stranger)], names=["Stranger.epub"])[0].path
        cid2 = chats.new_chat()
        app._open_chat(cid2)
        assert await run.until(lambda: app.chat_view.cid == cid2, 10)
        assert view.attach_file(picked)
        lookup = view._library_lookup
        if lookup is not None:
            await lookup[1]
        assert not chats.meta(cid2).get(LIBRARY_ATTACHMENT_META)
        assert _norm((view.composer.attachment or {}).get("path") or "") == _norm(picked)
        previous = reader
        reader = await chip_reader(previous)
        assert reader.details == (reader.bid, "")  # the file's own Book page (its route id is the Reader's)
        book_page = await _round_trip(run, reader, reader.bid, 1, both=False)
        assert _norm(book_page.book["path"]) == _norm(picked) and not book_page.book.get("in_library")
        assert "Stranger Title" in str(book_page.overview.title_text.value or ""), book_page.overview.title_text.value
        assert service.snapshot.all_books() and all(_norm(b.get("path") or "") != _norm(picked)
                                                    for b in service.snapshot.all_books())
        await _leave(run, reader)

    _run_app(scenario)


# =====================================================================================
# 3. By file path (Files, Open with) and the fallback
# =====================================================================================


def test_file_path_readers_jump_to_the_book_page_and_the_fallbacks(app_env, tmp_path):
    from glossarion_mobile.ui.components.dialogs import close_dialog
    from glossarion_mobile.ui.reader.chrome import DETAILS_TOOLTIP
    from glossarion_mobile.ui.reader.feature import DETAILS_NOT_A_BOOK
    from glossarion_mobile.ui.reader.reader_view import DETAILS_LABEL
    from glossarion_mobile.ui.screens.files import FileBrowserScreen

    async def scenario(run, d2, d5):
        app = run.app
        seeds = await _seed(run, d5)
        alpha_ws, alpha_bid = seeds["alpha_ws"], seeds["alpha_bid"]

        # ---- Tools › Files › Output › Alpha › Alpha.epub › Open with… › Reader ---------------------------
        fid = app.prefs.file_ref(str(alpha_ws))
        await app.navigate(f"/tools/files/output/{fid}")
        files_screen = await run.until(lambda: app.shell.top_screen if isinstance(app.shell.top_screen, FileBrowserScreen)
                                       and app.shell.top_screen.entries else None, 30)
        assert files_screen, run.route()
        row = next(i for i, e in enumerate(files_screen.entries) if e.name == "Alpha.epub")
        await run.driver.tap(key=f"files-row-{row}", timeout=20)
        await run.driver.tap(key="file-open", timeout=20)
        await run.driver.tap(key="open-reader", timeout=20)
        reader = await _new_reader(run)
        assert _norm(reader._target_path) == _norm(alpha_ws / "Alpha.epub") and reader.bid != alpha_bid
        assert _stack(run)[-1] == f"/reader/{reader.bid}" and _stack(run)[-2] == files_screen.route
        await _round_trip(run, reader, alpha_bid, 2)
        await _leave(run, reader)
        assert app.shell.top_screen is files_screen

        # ---- Open with (a shared EPUB that is no Library book): its own Book page, the EPUB's details --------
        loose = Path(d5.make_epub(tmp_path / "Shared" / "Loose Story.epub", d5.CHAPTERS[:3], title="Loose Story Title"))
        previous = reader
        app.reader.open_file(str(loose))
        reader = await _new_reader(run, previous)
        assert reader.details == (reader.bid, "")
        book_page = await _round_trip(run, reader, reader.bid, 1)
        assert _norm(book_page.book["path"]) == _norm(loose) and book_page.book.get("type") == "epub"
        assert "Loose Story Title" in str(book_page.overview.title_text.value or ""), book_page.overview.title_text.value
        await _leave(run, reader)

        # ---- a TXT that is no Library book: its own Book page --------------------------------------------
        text = tmp_path / "Shared" / "Loose notes.txt"
        text.write_text("\n\n".join(f"Chapter {i}\n\n" + "Some words here. " * 400 for i in range(1, 5)),
                        encoding="utf-8")
        previous = reader
        app.reader.open_file(str(text))
        reader = await _new_reader(run, previous)
        assert reader.session.count >= 2, reader.session.count
        assert reader.details == (reader.bid, "")
        book_page = await _round_trip(run, reader, reader.bid, 1, both=False)
        assert _norm(book_page.book["path"]) == _norm(text) and book_page.book.get("type") == "txt"
        await _leave(run, reader)

        # ---- a Markdown file (Files › Reader): no Book page; ⓘ stays, dimmed, and says why ------------------
        notes = tmp_path / "Shared" / "notes.md"
        notes.write_text("# Notes\n\nSome text.\n", encoding="utf-8")
        previous = reader
        app.reader.open_book(path=str(notes))
        reader = await _new_reader(run, previous)
        assert reader.details == (None, DETAILS_NOT_A_BOOK)
        await _show_bars(run, reader)
        button = reader.chrome.details_button
        assert await run.driver.count(key="reader-details") == 1 and not button.disabled
        assert button.tooltip == DETAILS_NOT_A_BOOK and button.icon_color is not None  # dimmed, reason as tooltip
        stack = _stack(run)
        await run.driver.tap(key="reader-details", timeout=15)
        assert await run.until(lambda: any(getattr(d, "open", False) for d in run.page._dialogs.controls), 10)
        assert await run.driver.count(text=DETAILS_TOOLTIP) >= 1 and await run.driver.count(text=DETAILS_NOT_A_BOOK) >= 1
        assert _stack(run) == stack and app.shell.top_screen is reader  # nothing opened behind the sheet
        for dialog in [d for d in run.page._dialogs.controls if getattr(d, "open", False)]:
            close_dialog(run.page, dialog)
        sheet = reader.open_more()
        assert sheet.items[0].label == DETAILS_LABEL and sheet.items[0].disabled_reason == DETAILS_NOT_A_BOOK
        sheet.close()
        await _leave(run, reader)

    _run_app(scenario)


# =====================================================================================
# 4. On a client that reports every dismissed sheet (Flet's own Flutter client, the phone)
# =====================================================================================


def test_menu_row_and_reason_sheet_on_the_phone_client_model(app_env, tmp_path):
    """``test_hotfix3_sheets.FlutterModel`` (what the Flet 1.0.3 client does with views and sheets, frame by
    frame, dismiss events included): ⋯ › Open Book Details closes the ⋯ sheet and the Book page comes up with
    no sheet left over it or stranded under it; Back shows the Reader at the same place, no sheet coming back.
    The "Book Details" reason sheet of a Markdown file closes with Close Menu ✕ and with a swipe down."""
    sheets = _load("_glossarion_sheets_hotfix3_issue6", "test_hotfix3_sheets.py")
    from glossarion_mobile.ui.library.book_page import BookPageScreen
    from glossarion_mobile.ui.reader.feature import DETAILS_NOT_A_BOOK

    async def scenario(run, d2, d5):
        app = run.app
        model = sheets.FlutterModel(run.session, run.conn)
        await model.settle()
        seeds = await _seed(run, d5)
        alpha_bid = seeds["alpha_bid"]
        await model.settle()

        # ---- the Reader on Alpha's compiled EPUB by path (Book page › Output, Files, Open with) ---------
        app.reader.open_book(path=str(seeds["alpha_ws"] / "Alpha.epub"))
        reader = await _new_reader(run)
        await model.settle()
        reader_route = run.route()
        assert model.top() == ("page", reader_route) and model.modals() == []
        await _read_to(run, reader, 2)
        await _show_bars(run, reader)
        spot, loads = _spot(reader), len(_reader_loads(run))

        # ⋯ › Open Book Details
        await model.tap(reader.chrome.more_button)
        assert await model.wait_until(lambda: model.top()[0] == "modal"), model.top()
        menu = model.top()[1]
        row = sheets._find(menu.content, lambda c: getattr(c, "key", None) == "reader-more-details")
        assert row is not None, "the ⋯ sheet has no Open Book Details row"
        await model.tap(row)
        book_route = f"/library/book/{alpha_bid}"
        assert await model.wait_until(lambda: model.top() == ("page", book_route), 6), (model.top(), _stack(run))
        assert model.modals() == [] and menu not in run.page._dialogs.controls  # nothing left over or under it
        book_page = app.shell.top_screen
        assert isinstance(book_page, BookPageScreen) and book_page.bid == alpha_bid
        await model.back()  # Android Back
        assert await model.wait_until(lambda: model.top() == ("page", reader_route), 6), model.top()
        assert model.modals() == [] and app.shell.top_screen is reader
        assert _spot(reader) == spot and len(_reader_loads(run)) == loads

        # ⓘ the same way
        await model.tap(reader.chrome.details_button)
        assert await model.wait_until(lambda: model.top() == ("page", book_route), 6), model.top()
        await model.back()
        assert await model.wait_until(lambda: model.top() == ("page", reader_route), 6), model.top()
        assert model.modals() == [] and _spot(reader) == spot and len(_reader_loads(run)) == loads
        await _leave(run, reader)
        await model.settle()

        # ---- a Markdown file: the reason sheet closes with Close Menu ✕ and with a swipe ------------------
        notes = tmp_path / "Shared" / "notes.md"
        notes.parent.mkdir(parents=True, exist_ok=True)
        notes.write_text("# Notes\n\nSome text.\n", encoding="utf-8")
        previous = reader
        app.reader.open_book(path=str(notes))
        reader = await _new_reader(run, previous)
        await model.settle()
        reader_route = run.route()
        assert reader.details == (None, DETAILS_NOT_A_BOOK)
        await _show_bars(run, reader)
        for close in ("button", "swipe"):
            await model.tap(reader.chrome.details_button)
            assert await model.wait_until(lambda: model.top()[0] == "modal"), model.top()
            info = model.top()[1]
            assert DETAILS_NOT_A_BOOK in str(sheets._find(
                info.content, lambda c: getattr(c, "value", None) == DETAILS_NOT_A_BOOK).value)
            if close == "button":
                await model.tap(sheets.close_button(info))
            else:
                await model.drag_down(info)
            assert await model.wait_until(lambda: model.top() == ("page", reader_route), 6), (close, model.top())
            assert model.modals() == [] and app.shell.top_screen is reader and not reader.disposed
        await _leave(run, reader)

    _run_app(scenario)
