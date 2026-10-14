"""Acceptance for the owner's device report #2 (2026-10-08): "the Reader must open .txt".

What the owner saw on the U8 APK: a TXT book in the Library could not be read in the app (a card tap
handed the file to another app, the card's Reader item was disabled, Open-with offered no Reader and
a TXT translation workspace had no chapters). This file walks the owner's taps on the REAL app on a
fake Flet session as Android (``test_ui_flows`` pattern: ``host_tester.PyTester`` +
``ui_driver.UiDriver``; the Reader on its WebView + loopback server, every ``load_request`` the
app sends to the client recorded), with every folder in ``tmp_path``:

* Library › Import EPUB (the FAB, the picker returns the file) takes a 70k-character UTF-8 Korean
  novel with "제N화" headings and a CP949 (non-UTF-8) Korean novel: both become "Not started" cards;
* a card tap opens the Book page; its read button opens the Reader on the TXT: several ~20k
  sections ("Section N" in the Chapters drawer, ``section_0001.txt``...), the page the WebView
  loads shows the decoded text (no replacement characters), in-page paging events, the page edge,
  ▶ and a Chapters row turn sections (one ``load_request`` each), search finds the last section,
  Back saves the position and the Book page's Continue reopens there;
* the CP949 book through the card ⋯ "📖 Open in Reader": Korean text incl. CP949-only syllables;
* Open-with (warm share and cold start) of a GBK Chinese ``.txt``: the share sheet's
  "Open in Reader" is enabled and opens it;
* after a translation run (the real ``txt_processor.TextFileProcessor`` split +
  ``TransateKRtoEN.ProgressManager`` progress, one section left untranslated), the in-progress
  card's Book page "Read translated" opens the TXT workspace: Translated / Original / Bilingual,
  the untranslated section shows the placeholder over its raw text, the text-mode position maps
  to the same place in the workspace layout; once compiled (``create_output_structure``) the
  Completed card opens the Book page and "Start reading" reads the workspace;
* Japanese Shift-JIS / CP932 and traditional-Chinese Big5 ``.txt`` files decode as their own
  text (the decode order lists those code pages).

Run from src/mobile with the mobile venv:
    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_devfix_issue2.py
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = MOBILE_DIR / "app"
TESTS_DIR = MOBILE_DIR / "tests"
SRC_DIR = MOBILE_DIR.parent
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


def _tiktoken_assets() -> bool:
    folder = APP_DIR / "assets" / "tiktoken"
    return folder.is_dir() and any(p.is_file() and p.suffix == "" for p in folder.iterdir())


_APP_MARKS = (
    pytest.mark.skipif(not (_has("flet") and _has("msgpack") and _has("flet_webview")),
                       reason="flet / msgpack / flet_webview not installed"),
    pytest.mark.skipif(not all(_has(m) for m in ("library_core", "reader_doc", "workspace_reader", "txt_processor",
                                                 "ebooklib", "bs4")),
                       reason="the shared Library / Reader cores are not importable here"),
)


def needs_app(test):
    for mark in _APP_MARKS:
        test = mark(test)
    return test

_TB_SPEC = importlib.util.spec_from_file_location("_glossarion_tb_helpers_devfix2",
                                                  Path(__file__).with_name("test_bootstrap.py"))
_TB = importlib.util.module_from_spec(_TB_SPEC)
_TB_SPEC.loader.exec_module(_TB)
storage = _TB.storage
app_env = _TB.app_env

CONFIG_JSON = SRC_DIR / "config.json"
REPLACEMENT = "�"


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest() if path.is_file() else ""


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """HOME / USERPROFILE / APPDATA / the Library and output roots in tmp_path (the app's bootstrap then
    moves HOME, the Library and Output into its own tmp storage); the developer's src/config.json is
    never written."""
    before = _md5(CONFIG_JSON)
    for key in ("HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA"):
        folder = tmp_path / "env" / key.lower()
        folder.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv(key, str(folder))
    for key, name in (("GLOSSARION_LIBRARY_DIR", "Library"), ("OUTPUT_DIRECTORY", "Output")):
        (tmp_path / "pre" / name).mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv(key, str(tmp_path / "pre" / name))
    monkeypatch.setenv("GLOSSARION_HTTP_LOG", "0")
    for key in ("RETAIN_SOURCE_EXTENSION", "retain_source_extension", "MODEL", "MAX_OUTPUT_TOKENS"):
        monkeypatch.delenv(key, raising=False)
    try:
        yield tmp_path
    finally:
        if _has("library_core"):
            try:
                import library_core

                library_core.uninstall_library_env()
            except Exception:
                pass
        assert _md5(CONFIG_JSON) == before, "src/config.json changed"


# =====================================================================================
# Fixture books
# =====================================================================================

_KO_WORDS = ("바람이", "불었다", "그는", "검을", "들었다", "하늘은", "맑았다", "성벽", "너머로", "해가",
             "저물고", "있었다", "기사단은", "조용히", "길을", "떠났다")


def _korean_paragraph(chapter: int, index: int, words: int = 60) -> str:
    body = " ".join(_KO_WORDS[(chapter * 7 + index + k) % len(_KO_WORDS)] for k in range(words))
    return f"[{chapter}-{index}] {body}."


def korean_novel(chapters: int = 7, paragraphs: int = 40, *, marker: str = "") -> str:
    """A web-novel TXT: "제N화 ..." heading lines, blank-line paragraphs (about 10k characters a chapter)."""
    parts = []
    for chapter in range(1, chapters + 1):
        parts.append(f"제{chapter}화 기사의 귀환 ({chapter})")
        parts.extend(_korean_paragraph(chapter, index) for index in range(1, paragraphs + 1))
    if marker:
        parts[-1] += f" {marker}"
    return "\n\n".join(parts) + "\n"


def cp949_novel() -> str:
    """Korean with CP949-only syllables (똠, 쀍, 햏 are not in EUC-KR), about 30k characters."""
    parts = []
    for chapter in range(1, 4):
        parts.append(f"제{chapter}장 똠방각하의 하루")
        parts.extend(f"{_korean_paragraph(chapter, index)} 쀍 햏 똠방각하." for index in range(1, 40))
    return "\n\n".join(parts) + "\n"


def chinese_novel() -> str:
    """Simplified Chinese, about 27k characters."""
    parts = []
    for chapter in range(1, 7):
        parts.append(f"第{chapter}章 重生归来")
        for index in range(1, 30):
            parts.append(f"【{chapter}-{index}】林凡睁开眼睛，发现自己回到了十年前。窗外的阳光洒在桌上，"
                         "他的心中充满了复杂的情绪。“这一次，我绝不会再让悲剧重演！”他握紧了拳头。" * 3)
    return "\n\n".join(parts) + "\n"


def _plain(markup: str) -> str:
    """Page markup as text (tags dropped, entities left alone: the fixtures need none)."""
    return re.sub(r"<[^>]+>", " ", markup)


def _norm(path) -> str:
    return os.path.normcase(os.path.abspath(str(path)))


# =====================================================================================
# The running app
# =====================================================================================


def _foundations():
    spec = importlib.util.spec_from_file_location("_glossarion_tf_helpers_devfix2",
                                                  Path(__file__).with_name("test_ui_foundations.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Run:
    """The app, its fake client connection, the tester / driver and small helpers."""

    def __init__(self, tf, app, conn, session, page, tester, driver, picker) -> None:
        self.tf, self.app, self.conn, self.session, self.page = tf, app, conn, session, page
        self.tester, self.driver, self.picker = tester, driver, picker

    def route(self) -> str:
        views = list(self.page.views or [])
        return str(views[-1].route) if views else ""

    def load_requests(self) -> list:
        """The URL of every WebView ``load_request`` the app sent to the client, in order."""
        from flet.messaging.protocol import MessageAction

        return [(m.body.args or {}).get("url") for m in self.conn.messages
                if m.action == MessageAction.INVOKE_METHOD and m.body.name == "load_request"]

    async def until(self, predicate, timeout: float = 30.0, interval: float = 0.05):
        deadline = time.monotonic() + timeout
        while True:
            value = predicate()
            if value:
                return value
            if time.monotonic() >= deadline:
                return value
            await asyncio.sleep(interval)

    async def back(self) -> None:
        views = list(self.page.views or [])
        if len(views) > 1:
            await self.session.dispatch_event(self.page._i, "view_pop", {"route": views[-1].route})
        await asyncio.sleep(0.2)

    async def reader(self, *, timeout: float = 60.0):
        """The Reader once its first page is up."""
        from glossarion_mobile.ui.reader.reader_view import ReaderScreen

        def ready():
            screen = getattr(self.app.reader, "active", None)
            if isinstance(screen, ReaderScreen) and not screen.disposed and screen.state in ("ready", "error"):
                return screen
            return None

        screen = await self.until(ready, timeout)
        assert screen is not None, f"the Reader did not open (route {self.route()})"
        error = getattr(getattr(screen, "error_text", None), "value", "") or ""
        assert screen.state == "ready", f"the Reader shows an error: {error!r}"
        assert await self.until(lambda: screen.webview is not None and screen.current_doc, 20)
        return screen

    def page_html(self, url: str) -> str:
        """The document the WebView would load (the loopback server; no proxy)."""
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(url, timeout=10) as response:
            assert response.status == 200
            return response.read().decode("utf-8")

    async def shown_text(self, screen) -> str:
        url = screen.webview.url
        assert url and url.startswith(self.app.reader.server.base_url), url
        return _plain(await asyncio.to_thread(self.page_html, url))

    async def turn(self, screen, action, index: int) -> str:
        """Run a chapter-changing action: exactly one ``load_request`` with the new section's page."""
        before = len(self.load_requests())
        doc = screen.current_doc
        result = action()
        if asyncio.iscoroutine(result):
            await result
        assert await self.until(lambda: len(self.load_requests()) > before and screen.current_doc != doc, 15), \
            f"no WebView navigation (index {screen.index}, wanted {index})"
        await asyncio.sleep(0.15)
        loads = self.load_requests()[before:]
        assert len(loads) == 1 and loads[0] == screen.webview.url, loads
        assert screen.index == index
        return await self.shown_text(screen)

    def books(self) -> dict:
        return {str(b.get("folder_name") or b.get("name")): dict(b) for b in self.app.library.snapshot.all_books()}


async def start_app(tf, files: dict) -> Run:
    from host_tester import HostPicker, PyTester
    from ui_driver import UiDriver

    _m, conn, session, page, app = await tf._start("android")
    picker = HostPicker(files)
    bridge = getattr(app, "files", None)
    assert bridge is not None, "the app has no FileBridge"
    bridge._get_picker = lambda: picker  # what the app's FilePicker service would answer

    run = Run(tf, app, conn, session, page, PyTester(session, page), None, picker)
    run.driver = UiDriver(run.tester, picker=picker, back=run.back, poll_ms=100, log=lambda *_a: None)
    return run


async def stop_app(run: Run) -> None:
    reader = getattr(run.app, "reader", None)
    screen = getattr(reader, "active", None)
    if screen is not None and not getattr(screen, "disposed", True):
        screen.dispose()
    jobs = getattr(run.app, "jobs", None)
    if jobs is not None and hasattr(jobs, "close"):
        try:
            jobs.close()
        except Exception:
            pass
    await run.tf._stop(run.app)


async def open_library(run: Run) -> None:
    import flows

    await flows.open_drawer(run.driver)
    await run.driver.tap(key="dest-library")
    await run.driver.wait(key="lib-search", timeout=60)


async def import_txt(run: Run, name: str) -> dict:
    """Library › Import EPUB (the FAB) with ``name`` picked: the Library's own row for it."""
    await run.driver.pick_file(name, lambda: run.driver.tap(key="lib-fab"))
    await run.driver.wait(contains=f"Added to Library: {name}", timeout=60)
    stem = os.path.splitext(name)[0]
    books = await run.until(lambda: run.books() if stem in run.books() else None, 60)
    assert books, f"{name} did not reach the Library: {list(run.books())}"
    return books[stem]


async def open_book_page(run: Run, book: dict, *, shelf: str = "progress"):
    """Tap the book's card on its shelf: the Book page (UI_SPEC §3.3)."""
    from glossarion_mobile.ui.library.book_page import BookPageScreen
    from glossarion_mobile.ui.library.home import LibraryScreen

    await open_library(run)
    home = run.app.shell.top_screen
    assert isinstance(home, LibraryScreen)
    if home.shelf != shelf:
        home.set_shelf(shelf)
    bid = run.app.library.bid_for(book)
    await run.driver.tap(key=f"book-{bid}", timeout=60)
    assert await run.until(lambda: isinstance(run.app.shell.top_screen, BookPageScreen), 20), run.route()
    assert run.route() == f"/library/book/{bid}"
    book_page = run.app.shell.top_screen
    await run.driver.wait(key="ov-read", timeout=60)
    assert await run.until(lambda: book_page.details is not None and book_page.details_phase == "full", 60)
    error = book_page.overview.error_text
    assert not error.visible, f"Book page error: {error.value!r}"
    return book_page, bid


# =====================================================================================
# Library TXT: Import, card -> Book page -> Reader (text mode), Open in Reader, Open-with
# =====================================================================================


@needs_app
def test_library_txt_books_open_in_the_reader(app_env, tmp_path):
    from glossarion_mobile.ui.reader import model as rm

    tf = _foundations()
    picks = tmp_path / "picks"
    picks.mkdir()
    utf8_name, cp949_name = "기사의 귀환.txt", "똠방각하.txt"
    utf8_text = korean_novel(marker="SAPPHIRE-END")
    assert len(utf8_text) > 60000
    (picks / utf8_name).write_bytes(utf8_text.encode("utf-8"))
    cp949_text = cp949_novel()
    cp949_bytes = cp949_text.encode("cp949")
    with pytest.raises(UnicodeDecodeError):
        cp949_bytes.decode("utf-8")  # a real non-UTF-8 file
    (picks / cp949_name).write_bytes(cp949_bytes)
    files = {utf8_name: picks / utf8_name, cp949_name: picks / cp949_name}

    async def scenario():
        run = await start_app(tf, files)
        try:
            import flows

            await flows.wait_home(run.driver)
            await open_library(run)
            # ---- Import (the FAB): two "Not started" TXT books in Library/Raw -------------------
            utf8_book = await import_txt(run, utf8_name)
            cp949_book = await import_txt(run, cp949_name)
            library_root = Path(run.app.paths.library)
            for book, name in ((utf8_book, utf8_name), (cp949_book, cp949_name)):
                raw = str(book.get("raw_source_path") or "")
                assert raw and Path(raw).is_file() and _norm(Path(raw).parent) == _norm(library_root / "Raw"), book
                assert Path(raw).read_bytes() == files[name].read_bytes()
                assert book.get("translation_state") == "not_started" and book.get("is_in_progress"), book

            # ---- the UTF-8 card -> Book page -> read button -> Reader -----------------------------
            book_page, bid = await open_book_page(run, utf8_book)
            label = str(book_page.overview.read_button.content or "")
            assert "Read" in label or "reading" in label, label
            await run.driver.tap(key="ov-read")
            screen = await run.reader()
            assert run.route().startswith(f"/reader/{bid}")
            session = screen.session
            assert session.plan.source_kind == "txt" and not session.plan.error
            assert screen.renderer == "webview"
            count = session.count
            assert count >= 4, count  # 70k characters in ~20k sections
            assert session.filenames == [f"section_{n:04d}.txt" for n in range(1, count + 1)]
            assert session.titles() == [f"Section {n}" for n in range(1, count + 1)]
            for index in range(count):  # every section is a page of at most ~20k characters
                text = _plain(session.chapter_html(index))
                assert 0 < len(text.strip()) <= 22000, (index, len(text))
            # the whole novel, in order, nothing lost or garbled
            joined = " ".join(_plain(session.chapter_html(i)) for i in range(count))
            assert REPLACEMENT not in joined
            assert joined.count("제1화 기사의 귀환") == 1 and "제7화 기사의 귀환 (7)" in joined
            assert all(f"[{c}-{p}]" in joined for c in (1, 4, 7) for p in (1, 20, 40))
            # the Chapters drawer: one row per section, the current one selected
            assert [r.title for r in screen.toc.rows] == session.titles()
            assert len(screen.toc.list_view.controls) == count
            assert screen.chrome.book_title.value and screen.chrome.chapter_title.value == "Section 1"
            assert screen.chrome.progress_text.value.startswith(f"Ch 1/{count}")
            # the page the WebView loaded is the first section, decoded
            shown = await run.shown_text(screen)
            assert "제1화 기사의 귀환 (1)" in shown and "[1-1]" in shown and REPLACEMENT not in shown
            first_url = screen.webview.url
            assert run.load_requests() == []  # the first page is built with its URL
            # ---- paging: the page's own events (shared shell schema) --------------------------------
            screen.deduper.set_document(screen.current_doc, 0)
            screen.handle_payload({"type": "ready", "seq": 1, "page": 0, "count": 6, "chapter": 0,
                                   "paginated": True})
            screen.handle_payload({"type": "page", "seq": 2, "page": 3, "count": 6, "chapter": 0, "reason": "next"})
            assert (screen.page_no, screen.page_count) == (3, 6)
            assert screen.chrome.page_text.value == "Page 4/6"
            # past the last page: the next section loads into the same WebView
            webview = screen.webview
            shown = await run.turn(screen, lambda: screen.handle_payload(
                {"type": "edge", "seq": 3, "edge": "end", "chapter": 0}), 1)
            assert screen.webview is webview and screen.webview.url != first_url
            assert "[1-1]" not in shown and screen.chrome.chapter_title.value == "Section 2"
            second = _plain(session.chapter_html(1))
            assert second.split()[0] in shown
            # ▶ (the bottom bar) -> Section 3
            screen.chrome.set_visible(True)
            await run.turn(screen, lambda: run.driver.tap(key="reader-next"), 2)
            # a Chapters row -> the last section
            last = count - 1
            await run.turn(screen, lambda: screen._on_toc_row(screen.toc.rows[last]), last)
            assert "SAPPHIRE-END" in await run.shown_text(screen)
            assert screen.toc.current == last
            # search: the marker is in the last section only; a hit in section 1 opens it
            rows = await asyncio.to_thread(session.search, "SAPPHIRE-END")
            assert [r["chapter_idx"] for r in rows] == [last]
            hits = await asyncio.to_thread(session.search, "[1-3]")
            assert hits and hits[0]["chapter_idx"] == 0
            await run.turn(screen, lambda: screen._on_search_pick(hits[0]), 0)
            # the start edge of section 2 opens section 1 at its last page
            await run.turn(screen, lambda: screen.go_chapter(2), 2)
            shown = await run.turn(screen, lambda: screen.handle_payload(
                {"type": "edge", "seq": 1, "edge": "start", "chapter": 2}), 1)
            assert screen.last_page
            # reading in section 3, page 3/6; Back leaves the Reader and saves the position
            await run.turn(screen, lambda: screen.go_chapter(2), 2)
            screen.deduper.set_document(screen.current_doc, 2)
            screen.handle_payload({"type": "ready", "seq": 1, "page": 0, "count": 6, "chapter": 2})
            screen.handle_payload({"type": "page", "seq": 2, "page": 2, "count": 6, "chapter": 2, "reason": "next"})
            screen.chrome.set_visible(False)
            await run.back()
            assert await run.until(lambda: screen.disposed, 10), run.route()
            saved = run.app.prefs.reader_position(bid)
            assert saved and saved["href"] == "section_0003.txt" and saved["chapter"] == 2, saved
            assert 0.0 < float(saved.get("fraction") or 0.0) < 1.0, saved
            # the Book page offers Continue; it reopens the Reader at that section
            assert run.route() == f"/library/book/{bid}"
            await run.driver.wait(key="ov-continue", timeout=30)
            await run.driver.tap(key="ov-continue")
            again = await run.reader()
            assert again is not screen and again.index == 2, again.index
            assert again.chrome.chapter_title.value == "Section 3"
            assert "[" in await run.shown_text(again)
            again.dispose()
            await run.back()

            # ---- the CP949 book: card ⋯ "📖 Open in Reader" -----------------------------------------
            from glossarion_mobile.ui.library.home import LibraryScreen

            await open_library(run)
            home = run.app.shell.top_screen
            assert isinstance(home, LibraryScreen)
            cp949_row = run.books()["똠방각하"]
            sheet = home.card_actions(cp949_row)
            items = {item.label: item for item in sheet.items}
            reader_item = items.get("\U0001f4d6 Open in Reader")
            assert reader_item is not None and reader_item.disabled_reason is None, list(items)
            reader_item.on_select()
            cp_screen = await run.reader()
            cp_session = cp_screen.session
            assert cp_session.plan.source_kind == "txt" and cp_session.count >= 2
            cp_all = " ".join(_plain(cp_session.chapter_html(i)) for i in range(cp_session.count))
            assert REPLACEMENT not in cp_all and "똠방각하의 하루" in cp_all and "쀍 햏" in cp_all
            assert "똠방각하의 하루" in await run.shown_text(cp_screen)
            assert cp_screen.chrome.mode_buttons.visible is False  # plain text: no Original / Bilingual
            assert cp_session.available_modes(0)[rm.TRANSLATED]
            cp_screen.dispose()
            await run.back()
        finally:
            await stop_app(run)

    asyncio.run(scenario())


@needs_app
def test_open_with_a_txt_file_offers_and_opens_the_reader(app_env, tmp_path):
    """Open-with / Share of a ``.txt`` (warm share event and cold start): the share sheet's
    "Open in Reader" is enabled and opens the shared copy in the Reader (GBK Chinese here)."""
    from glossarion_mobile.services.intents import ACTION_OPEN_IN_READER

    tf = _foundations()
    text = chinese_novel()
    data = text.encode("gbk")
    with pytest.raises(UnicodeDecodeError):
        data.decode("utf-8")

    async def scenario():
        run = await start_app(tf, {})
        try:
            import flows

            await flows.wait_home(run.driver)
            jobs = run.app.jobs
            for source, deliver in (("share", lambda items: jobs._on_share_event({"items": items})),
                                    ("initial", jobs.handle_initial_shared)):
                # Android hands the app a copy in its cache folder (removed once imported)
                shared = Path(run.app.paths.cache) / "shared" / source / "重生归来.txt"
                shared.parent.mkdir(parents=True, exist_ok=True)
                shared.write_bytes(data)
                item = {"kind": "file", "path": str(shared), "name": shared.name, "id": f"{source}-1"}
                await deliver([item])
                imports = run.app.intents.history[-1:]
                assert imports and imports[0].imported is not None, run.app.intents.history
                imported = imports[0].imported
                assert Path(imported.path).is_file() and Path(imported.path).read_bytes() == data
                actions = {a.id: a for a in imports[0].actions}
                assert actions[ACTION_OPEN_IN_READER].disabled_reason is None, actions
                await run.driver.tap(key=f"intent-{ACTION_OPEN_IN_READER}", timeout=20)
                screen = await run.reader()
                session = screen.session
                assert session.plan.source_kind == "txt" and session.count >= 2
                assert _norm(session.plan.epub_path) == _norm(imported.path)
                assert screen.chrome.book_title.value == "重生归来"
                shown = await run.shown_text(screen)
                assert "第1章 重生归来" in shown and "林凡睁开眼睛" in shown and REPLACEMENT not in shown
                every = " ".join(_plain(session.chapter_html(i)) for i in range(session.count))
                assert "第3章 重生归来" in every and "【3-29】" in every
                await run.turn(screen, lambda: screen.handle_payload(
                    {"type": "edge", "seq": 5, "edge": "end", "chapter": 0}), 1)
                bid = screen.bid
                screen.dispose()
                assert run.app.prefs.reader_position(bid)["href"] == "section_0002.txt"
                await run.back()
        finally:
            await stop_app(run)

    asyncio.run(scenario())


# =====================================================================================
# TXT translation workspace: in progress, then compiled (Completed shelf)
# =====================================================================================


def _translate_workspace(raw: str, workspace: str, *, skip: tuple = ()) -> list:
    """What a TXT translation run leaves behind, from the pipeline's own code: the
    ``TextFileProcessor`` split (``word_count/`` + ``.cache/split.cache``), one translated file per
    section under the pipeline's output name and its ``ProgressManager`` entry (keyed by number, with
    the section's ``content_hash``); sections in ``skip`` stay untranslated. Returns the sections."""
    from TransateKRtoEN import FileUtilities, ProgressManager
    from txt_processor import TextFileProcessor

    sections = TextFileProcessor(raw, workspace).extract_chapters()
    progress = ProgressManager(workspace)
    for index, section in enumerate(sections):
        if index in skip:
            continue
        output = FileUtilities.create_chapter_filename(section, section["num"])
        Path(workspace, output).write_text(f"Translated section {index + 1}.\n\nThe knight returned "
                                           f"home ({index + 1}).", encoding="utf-8")
        progress.update(index, section["num"], section["content_hash"], output, status="completed",
                        chapter_obj=section)
    progress.save()
    return sections


@needs_app
def test_txt_translation_workspace_reads_translated_original_and_bilingual(app_env, tmp_path):
    if not (_has("TransateKRtoEN") and _has("tiktoken") and _tiktoken_assets()):
        pytest.skip("the translation pipeline (TransateKRtoEN, tiktoken assets) is not available here")
    from glossarion_mobile.ui.reader import model as rm
    from glossarion_mobile.ui.reader import text_book

    tf = _foundations()
    picks = tmp_path / "picks"
    picks.mkdir()
    name = "귀환록.txt"
    (picks / name).write_bytes(korean_novel(chapters=8, paragraphs=45).encode("utf-8"))

    async def scenario():
        run = await start_app(tf, {name: picks / name})
        try:
            import flows

            await flows.wait_home(run.driver)
            await open_library(run)
            book = await import_txt(run, name)
            raw, workspace = str(book["raw_source_path"]), str(book["output_folder"])
            assert Path(workspace, "translation_progress.json").is_file()
            # text mode first: read into section 2 (the Reader's own layout), then leave
            book_page, bid = await open_book_page(run, book)
            await run.driver.tap(key="ov-read")
            screen = await run.reader()
            text_count = screen.session.count
            await run.turn(screen, lambda: screen.go_chapter(1), 1)
            screen.deduper.set_document(screen.current_doc, 1)
            screen.handle_payload({"type": "ready", "seq": 1, "page": 0, "count": 4, "chapter": 1})
            screen.handle_payload({"type": "page", "seq": 2, "page": 2, "count": 4, "chapter": 1})
            screen.dispose()
            saved = run.app.prefs.reader_position(bid)
            assert saved["href"] == "section_0002.txt"
            await run.back()
            await run.back()

            # ---- a translation run: every section but the second ------------------------------------
            sections = await asyncio.to_thread(_translate_workspace, raw, workspace, skip=(1,))
            assert len(sections) >= 3, len(sections)
            assert text_book.has_text_workspace(workspace)
            run.app.library.mark_dirty()
            await run.app.library.refresh(reason="devfix2 translated")
            book = run.books()[os.path.splitext(name)[0]]
            assert book.get("is_in_progress") and run.app.library.bid_for(book) == bid, book
            book_page, _bid = await open_book_page(run, book)
            label = str(book_page.overview.read_button.content or "")
            assert "Read translated" in label, label
            await run.driver.tap(key="ov-read")
            screen = await run.reader()
            session = screen.session
            assert session.plan.mode == "workspace" and session.plan.source_kind == "txt"
            assert session.count == len(sections)
            assert session.filenames == [s["filename"] for s in sections]
            assert session.has_alternate and screen.chrome.mode_buttons.visible
            # the text-mode position (Section 2 of the Reader's own layout) maps to the same place
            position = await asyncio.to_thread(session.position_from_pref, saved)
            wanted = rm.book_percent(1, float(saved["fraction"]), text_count)
            got = rm.book_percent(position.chapter, position.fraction, session.count)
            assert abs(got - wanted) <= 2, (got, wanted, position)
            # Translated (default): the response text
            await run.turn(screen, lambda: screen.go_chapter(0), 0) if screen.index != 0 else None
            shown = await run.shown_text(screen)
            assert "Translated section 1." in shown and "[1-1]" not in shown
            # Original: the raw word_count section
            await run.turn(screen, lambda: screen.set_mode(rm.ORIGINAL), 0)
            shown = await run.shown_text(screen)
            assert "제1화 기사의 귀환 (1)" in shown and "Translated section 1." not in shown
            # Bilingual: both, paired
            await run.turn(screen, lambda: screen.set_mode(rm.BILINGUAL), 0)
            shown = await run.shown_text(screen)
            assert "Translated section 1." in shown and "제1화" in shown
            assert "glr-bi" in session.chapter_html(0, rm.BILINGUAL)
            # the untranslated section: the placeholder over its raw text
            await run.turn(screen, lambda: screen.set_mode(rm.TRANSLATED), 0)
            shown = await run.turn(screen, lambda: screen.go_chapter(1), 1)
            assert text_book.UNTRANSLATED_TEXT in shown
            assert _plain(session.chapter_html(1, rm.ORIGINAL)).split()[0] in shown
            assert session.available_modes(1)[rm.BILINGUAL] is False
            screen.dispose()
            await run.back()
            await run.back()

            # ---- the last section translated + compiled: the Completed shelf ----------------------
            await asyncio.to_thread(_translate_workspace, raw, workspace)
            from txt_processor import TextFileProcessor

            processor = TextFileProcessor(raw, workspace)
            compiled = [(Path(workspace, entry["output_file"]).name,
                         Path(workspace, entry["output_file"]).read_text(encoding="utf-8"))
                        for entry in json.loads(Path(workspace, "translation_progress.json").read_text(
                            encoding="utf-8"))["chapters"].values()]
            await asyncio.to_thread(processor.create_output_structure, compiled)
            assert any(p.name.endswith("_translated.txt") for p in Path(workspace).iterdir())
            run.app.library.mark_dirty()
            await run.app.library.refresh(reason="devfix2 compiled")
            done = [b for b in run.app.library.snapshot.completed
                    if _norm(b.get("output_folder") or b.get("path") or "").startswith(_norm(workspace))]
            assert done, [(b.get("name"), b.get("type")) for b in run.app.library.snapshot.all_books()]
            completed = done[0]
            assert completed.get("type") == "txt", completed
            book_page, cbid = await open_book_page(run, completed, shelf="completed")
            assert str(book_page.overview.read_button.content or "").strip().endswith("Start reading")
            await run.driver.tap(key="ov-read")
            screen = await run.reader()
            session = screen.session
            assert session.plan.source_kind == "txt" and session.plan.mode == "workspace"
            assert session.count == len(sections) and session.has_alternate
            shown = await run.shown_text(screen)
            assert "Translated section" in shown and text_book.UNTRANSLATED_TEXT not in " ".join(
                _plain(session.chapter_html(i)) for i in range(session.count))
            await run.turn(screen, lambda: screen.set_mode(rm.ORIGINAL), screen.index)
            assert "기사의 귀환" in await run.shown_text(screen)
            screen.dispose()
            await run.back()
            await run.back()
            # the Completed card's ⋯ "📖 Open in Reader" reads the same workspace
            await open_library(run)
            home = run.app.shell.top_screen
            home.set_shelf("completed")
            items = {item.label: item for item in home.card_actions(completed).items}
            reader_item = items.get("\U0001f4d6 Open in Reader")
            assert reader_item is not None and reader_item.disabled_reason is None, list(items)
            reader_item.on_select()
            screen = await run.reader()
            assert screen.session.plan.mode == "workspace" and screen.session.count == len(sections)
            assert "Translated section" in await run.shown_text(screen)
            screen.dispose()
            await run.back()

            # ---- Tools › Files › the compiled <stem>_translated.txt › Open with… › Reader -----------
            from glossarion_mobile.ui.screens.files import FileBrowserScreen

            compiled_txt = next(p for p in Path(workspace).iterdir() if p.name.endswith("_translated.txt"))
            fid = run.app.prefs.file_ref(workspace)
            await run.app.navigate(f"/tools/files/output/{fid}")
            files_screen = await run.until(lambda: run.app.shell.top_screen if isinstance(
                run.app.shell.top_screen, FileBrowserScreen) and run.app.shell.top_screen.entries else None, 30)
            assert files_screen, run.route()
            row = next(i for i, e in enumerate(files_screen.entries) if e.name == compiled_txt.name)
            await run.driver.tap(key=f"files-row-{row}", timeout=20)
            await run.driver.tap(key="file-open", timeout=20)
            await run.driver.tap(key="open-reader", timeout=20)
            screen = await run.reader()
            session = screen.session
            assert session.plan.source_kind == "txt" and _norm(session.plan.epub_path) == _norm(compiled_txt)
            assert session.count == len(sections)  # one section per translated section (the separator)
            assert "Translated section 1." in await run.shown_text(screen)
            every = " ".join(_plain(session.chapter_html(i)) for i in range(session.count))
            assert "=" * 50 not in every and f"Translated section {len(sections)}." in every
            screen.dispose()
        finally:
            await stop_app(run)

    asyncio.run(scenario())


# =====================================================================================
# Code pages the decode order names
# =====================================================================================

_JAPANESE = ("第一章　転生\n\n目を覚ますと、そこは見知らぬ森の中だった。「ここはどこだ？」と彼は呟いた。"
             "木々の間から差し込む光が、地面に模様を描いている。\n\n") * 40
_TRADITIONAL = ("第一章　重生\n\n林凡睜開眼睛，發現自己回到了十年前。窗外的陽光灑在桌上，"
                "他的心中充滿了複雜的情緒。\n\n") * 40


@pytest.mark.parametrize("label,text,codec", [
    ("shift_jis", _JAPANESE, "shift_jis"),
    ("cp932", "吾輩は猫である。名前はまだ無い。どこで生れたかとんと見当がつかぬ。\n\n" * 60, "cp932"),
    ("big5", _TRADITIONAL, "big5"),
    ("gbk", chinese_novel(), "gbk"),
    ("cp949", cp949_novel(), "cp949"),
], ids=["shift_jis", "cp932", "big5", "gbk", "cp949"])
def test_cjk_code_page_txt_files_read_as_their_own_text(tmp_path, label, text, codec):
    """A ``.txt`` in a CJK code page reads as its own text in the Reader session (the decode order
    in ``text_book`` lists CP949, GB18030, Shift-JIS and Big5)."""
    if not all(_has(m) for m in ("library_core", "reader_doc", "workspace_reader", "txt_processor", "bs4")):
        pytest.skip("the shared Reader cores are not importable here")
    from glossarion_mobile.ui.reader import session as rs

    path = tmp_path / f"{label}.txt"
    path.write_bytes(text.encode(codec))
    session = rs.ReaderSession(rs.plan_for_file(str(path)), engine=rs.DocEngine())
    session.load()
    shown = " ".join(_plain(session.chapter_html(i)) for i in range(session.count))
    first = text.strip().split("\n")[0]
    assert REPLACEMENT not in shown and first in shown, (label, shown[:60])
