"""Acceptance test for the owner's device report #11 on the U8 APK (2026-10-08):

    "The Translated, Bilingual and Original buttons are all just the Translated button."

What the owner saw on the phone: whichever segment was pressed, the Reader kept showing the translated
text (after a reopen in Original: the Korean text, whatever was pressed). The Python side switched the
flavour and published the right page, but the Reader only assigned the WebView's ``url``; flet-webview
1.0.3 reads ``url`` once (``initState``) and only ``load_request`` navigates, so the page on screen stayed
the one the WebView was built with while the segment's highlight moved.

This test replays the owner's session headlessly on the REAL app objects: ``main.main`` on the fake Flet
session as Android (test_devfix_issue1's ``phone`` fixture: isolated storage, src/config.json unchanged),
the real LibraryService scanning the app's own Library / Output folders, the real Book page and its
"Start reading" / "Continue" buttons, the real ReaderFeature, ReaderServer and ReaderScreen. The Flutter
side is test_devfix_issue1's ``WebViewClient`` (a WebView loads its url once when it is built; later
pages only through ``load_request``) running the page in headless Chrome when installed (else its
``ShellModel``); "what the owner sees" is the text of the page that WebView shows.

For both kinds of book the owner reads with the segments,

* an in-progress book (Library "In progress": the raw EPUB in Library/Raw, chapters 1-3 of 4 translated
  in Output/<book>), which the Reader opens as an ``overlay`` over the raw EPUB, and
* a Completed book (Output/<book>/<book>.epub compiled, its raw EPUB linked), opened as ``dual``,

every press of a segment (what Flutter sends: the ``selected`` patch, then ``change``) sends exactly ONE
``load_request`` (on the wire, to the same WebView; no new WebView is built) and the page then on screen
is the pressed version: Original = Korean only, Bilingual = each Korean paragraph followed by its English
one, Translated = English only. The segment, the session and the saved position agree. Chrome taps keep
working after every switch: a centre tap on the new page shows / hides the bars, ▶ / ◀ change the
chapter in the chosen version (again one ``load_request`` each), a right-edge tap turns a page of the
long chapter. A book left in Original and reopened with "Continue" starts in Korean and still switches.

Not pinned here: the page a switch lands on follows the desktop ``_capture_position_hint`` (a chapter
that fits one page counts as "on its last page", so Bilingual, about twice as long, opens on its last page).

Run from src/mobile (mobile venv; ``GLOSSARION_TEST_NO_BROWSER=1`` uses ShellModel instead of Chrome):
    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_devfix_issue11.py
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import re
import urllib.request
from pathlib import Path
from typing import Any

import pytest

_SPEC = importlib.util.spec_from_file_location("_glossarion_devfix1_helpers_issue11",
                                               Path(__file__).with_name("test_devfix_issue1.py"))
D1 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(D1)

phone = D1.phone  # the real app on the fake Flet session as Android, isolated storage (devfix issue 1)

pytestmark = pytest.mark.skipif(not D1._has("msgpack"), reason="msgpack not installed")

ORIGINAL, TRANSLATED, BILINGUAL = "original", "translated", "bilingual"
CHAPTERS = 4
LONG = 1  # chapter 2 (index 1) spans several pages
TRANSLATED_IN_PROGRESS = 3  # the in-progress book has chapters 1-3 translated, chapter 4 not yet
_KO = "비가 오래된 항구 도시에 계속 내렸고 야경꾼들은 밤새 순찰을 돌았다. "
_EN = "The rain kept falling on the old harbour city while the night watch walked its rounds. "
HANGUL = re.compile(r"[가-힣]")
PARAGRAPH = re.compile(r"\b(EN|RAW)-MARK-(\d\d) (\d+)\.")
BOOKS = {"in_progress": "Moonlit Harbor", "completed": "Silver Lantern"}


# =====================================================================================
# The owner's two books, in the app's own Library / Output folders
# =====================================================================================


def _paragraphs(number: int) -> int:
    return 12 if number - 1 == LONG else 2


def _chapter_body(marker: str, number: int, filler: str) -> str:
    mark = f"{marker}-MARK-{number:02d}"
    body = [f"<h1>{marker} chapter {number}</h1>", f"<p>{mark}</p>"]
    body += [f"<p>{mark} {n}. {filler * 3}</p>" for n in range(1, _paragraphs(number) + 1)]
    return "".join(body)


def _write_epub(path: Path, *, title: str, marker: str, lang: str, filler: str) -> Path:
    from ebooklib import epub

    book = epub.EpubBook()
    book.set_identifier(f"glossarion-devfix-issue11-{title.lower().replace(' ', '-')}-{marker.lower()}")
    book.set_title(title)
    book.set_language(lang)
    book.add_author("Glossarion tests")
    items = []
    for number in range(1, CHAPTERS + 1):
        chapter = epub.EpubHtml(title=f"{marker} chapter {number}", file_name=f"chapter{number:04d}.xhtml", lang=lang)
        chapter.content = (f"<html><head><title>{marker} chapter {number}</title></head>"
                           f"<body>{_chapter_body(marker, number, filler)}</body></html>")
        book.add_item(chapter)
        items.append(chapter)
    book.toc = tuple(items)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav", *items]
    path.parent.mkdir(parents=True, exist_ok=True)
    epub.write_epub(str(path), book)
    return path


def _workspace(output: Path, library: Path, name: str, done: int) -> Path:
    """Output/<name> the way a translation run leaves it: source_epub.txt -> Library/Raw/<name>.epub,
    response files + translation_progress.json for chapters 1..``done``, metadata.json; when every chapter
    is done, the compiled Output/<name>/<name>.epub (same chapter file names, English)."""
    raw = _write_epub(library / "Raw" / f"{name}.epub", title=name, marker="RAW", lang="ko", filler=_KO)
    workspace = output / name
    workspace.mkdir(parents=True)
    (workspace / "source_epub.txt").write_text(str(raw), encoding="utf-8")
    entries = {}
    for number in range(1, done + 1):
        response = f"response_chapter{number:04d}.html"
        (workspace / response).write_text(
            f"<html><head><title>EN chapter {number}</title></head><body>{_chapter_body('EN', number, _EN)}"
            "</body></html>", encoding="utf-8")
        entries[str(number)] = {"actual_num": number, "status": "completed", "output_file": response,
                                "original_basename": f"chapter{number:04d}.xhtml", "content_hash": f"{name}-{number}"}
    (workspace / "translation_progress.json").write_text(
        json.dumps({"version": "2.1", "chapters": entries, "chapter_chunks": {}}), encoding="utf-8")
    (workspace / "metadata.json").write_text(json.dumps({"title": name}), encoding="utf-8")
    if done == CHAPTERS:
        _write_epub(workspace / f"{name}.epub", title=name, marker="EN", lang="en", filler=_EN)
    return workspace


def _seed(phone) -> dict:
    paths = phone.helpers._TB.rb.get_paths()
    assert paths is not None
    library, output = Path(paths.library), Path(paths.output)
    tmp = os.path.normcase(str(phone.tmp))
    assert os.path.normcase(str(library)).startswith(tmp) and os.path.normcase(str(output)).startswith(tmp)
    return {"in_progress": _workspace(output, library, BOOKS["in_progress"], TRANSLATED_IN_PROGRESS),
            "completed": _workspace(output, library, BOOKS["completed"], CHAPTERS)}


async def _library_row(phone, kind: str) -> dict:
    """The book's row from a real Library scan (the shelf the owner taps it on)."""
    service = phone.app.library
    snapshot = None
    for _ in range(600):
        if not service.scanning:
            snapshot = await service.refresh(reason="test")
            if snapshot is not None:
                break
        await asyncio.sleep(0.05)
    assert snapshot is not None, "the Library never finished scanning"
    shelf = snapshot.in_progress if kind == "in_progress" else snapshot.completed
    rows = [dict(b) for b in shelf if str(b.get("name") or "").startswith(BOOKS[kind])]
    assert len(rows) == 1, (kind, [(b.get("name"), b.get("path")) for b in snapshot.all_books()])
    return rows[0]


# =====================================================================================
# Driving the app like the owner
# =====================================================================================


async def _book_page(phone, row: dict) -> Any:
    from glossarion_mobile.ui.library.book_page import BookPageScreen

    app = phone.app
    bid = app.library.bid_for(row)
    app.navigate_to("library.book", {"bid": bid})
    assert await D1._wait(lambda: isinstance(app.shell.top_screen, BookPageScreen) and app.shell.top_screen.bid == bid,
                          15), [e.route for e in app.shell.stack]
    page = app.shell.top_screen
    assert await D1._wait(lambda: page.details_phase == "full", 30), page.details_phase
    return page


async def _reader_from(phone, button: Any) -> Any:
    """Tap a Book page button that opens the Reader; returns the ReaderScreen once its page is up."""
    from glossarion_mobile.ui.reader.reader_view import ReaderScreen

    app = phone.app
    previous = app.reader.active
    assert button.visible is not False and not button.disabled, str(button.content)
    await D1._fire(phone, button, "click")

    def shown() -> bool:
        screen = app.reader.active
        return (isinstance(screen, ReaderScreen) and screen is not previous and screen.state == "ready"
                and screen.webview is not None and screen.page_slot.content is screen.webview)
    assert await D1._wait(shown, 30), "the Reader did not open"
    return app.reader.active


async def _tap_segment(phone, screen: Any, mode: str) -> None:
    """The owner's tap on a segment: only possible while the bars are up and the segment is enabled; Flutter
    sends the ``selected`` patch, then the ``change`` event."""
    chrome = screen.chrome
    assert chrome.visible and chrome.top.visible and not chrome.top.ignore_interactions, "the bars are hidden"
    buttons = chrome.mode_buttons
    assert buttons.visible and chrome.segments[mode].disabled is False, (mode, buttons.visible)
    assert phone.session.index.get(buttons._i) is buttons, "the segments are not on screen"
    phone.session.apply_patch(buttons._i, {"selected": [mode]})
    await phone.session.dispatch_event(buttons._i, "change", [mode])


async def _slide(phone, screen: Any, index: int) -> None:
    """The chapter slider dragged to ``index`` (Flutter patches ``value``, then ``change_end``)."""
    slider = screen.chrome.slider
    phone.session.apply_patch(slider._i, {"value": index})
    await D1._fire(phone, slider, "change_end", index)


def _wire_loads(phone, webview: Any) -> list:
    """The ``load_request`` calls sent to ``webview`` over the Flet connection (what the phone receives)."""
    return [str((m.body.args or {}).get("url")) for m in D1._invokes(phone.conn, "load_request")
            if m.body.control_id == webview._i]


def _served(url: str) -> str:
    with urllib.request.urlopen(url.split("#", 1)[0], timeout=10) as response:
        return response.read().decode("utf-8")


def _expect_version(text: str, index: int, mode: str) -> None:
    """The page text is chapter ``index`` in ``mode``: Korean only / Korean + English interleaved paragraph
    by paragraph / English only."""
    number = index + 1
    marks = {kind for kind, chapter in D1._marks(text) if chapter == index}
    paragraphs = [(kind, int(n)) for kind, chapter, n in PARAGRAPH.findall(text) if int(chapter) == number]
    count = _paragraphs(number)
    korean, english = bool(HANGUL.search(text)), "harbour" in text
    if mode == ORIGINAL:
        assert marks == {"RAW"} and korean and not english, (mode, index, marks, korean, english)
        assert paragraphs == [("RAW", n) for n in range(1, count + 1)], paragraphs
    elif mode == TRANSLATED:
        assert marks == {"EN"} and english and not korean, (mode, index, marks, korean, english)
        assert paragraphs == [("EN", n) for n in range(1, count + 1)], paragraphs
    else:
        assert marks == {"RAW", "EN"} and korean and english, (mode, index, marks, korean, english)
        assert paragraphs == [p for n in range(1, count + 1) for p in (("RAW", n), ("EN", n))], paragraphs


class Owner:
    """One Reader screen as the owner uses it (``D1.Turns`` checks each page change)."""

    def __init__(self, phone, client: Any, screen: Any) -> None:
        self.phone, self.client, self.screen = phone, client, screen
        self.turns = D1.Turns(phone, client, screen)
        self.webview = screen.webview
        self.presses = 0
        self.log: list = []  # what the owner saw after each page change (printed with -s)

    def note(self, what: str, index: int, mode: str, shown: dict) -> None:
        text = shown["text"]
        self.log.append(f"{what:<12} ch{index + 1} {mode:<10} page {shown['page'] + 1}/{shown['count']} "
                        f"korean={bool(HANGUL.search(text))} english={'harbour' in text} "
                        f"paragraphs={''.join(k[0] for k, c, _n in PARAGRAPH.findall(text) if int(c) == index + 1)} "
                        f"load_requests={len(_wire_loads(self.phone, self.webview))}")

    async def first_page(self, index: int, mode: str) -> None:
        screen = self.screen
        assert self.client.loads[-1] == (self.webview._i, self.webview.url, "init")
        assert _wire_loads(self.phone, self.webview) == []
        await self.turns.page_ready(set(), index)
        shown = await self.turns.check_shown(index, mode="RAW" if mode == ORIGINAL else "EN")
        _expect_version(shown["text"], index, mode)
        assert screen.session.flavor == mode and screen.chrome.mode_buttons.selected == [mode]
        self.note("opened", index, mode, shown)
        assert await D1._wait(lambda: not screen.chrome.visible, 10)  # the first-open fade (shortened)

    async def change(self, action: Any, index: int, mode: str, what: str = "chapter") -> dict:
        """``action`` changes the page: exactly one ``load_request`` (on the wire, same WebView) and the
        page then shown is chapter ``index`` in ``mode``; the segment, session and saved position agree."""
        screen = self.screen
        wire = len(_wire_loads(self.phone, self.webview))
        shown = await self.turns.turn(action, index, mode="RAW" if mode == ORIGINAL else "EN")
        sent = _wire_loads(self.phone, self.webview)[wire:]
        assert sent == [self.client.shown_url] == [screen.webview.url], sent
        assert screen.webview is self.webview and self.client.loads[-1][2] == "load_request"
        _expect_version(shown["text"], index, mode)
        assert ("glr-bi" in _served(self.client.shown_url)) == (mode == BILINGUAL)
        assert screen.session.flavor == mode and screen.chrome.mode_buttons.selected == [mode]
        assert not screen._mode_busy
        self.note(what, index, mode, shown)
        return shown

    async def press(self, mode: str) -> dict:
        self.presses += 1
        return await self.change(lambda: _tap_segment(self.phone, self.screen, mode), self.screen.index, mode,
                                 f"press {mode}")

    async def centre_tap(self, visible: bool) -> None:
        """A centre tap on the page on screen toggles the bars (its events are handled)."""
        chrome = self.screen.chrome
        assert chrome.visible is not visible
        await self.turns.tap(D1.CENTRE)
        assert await D1._wait(lambda: chrome.visible is visible, 10), \
            f"a centre tap did not {'show' if visible else 'hide'} the bars"
        assert self.screen.events[-1].type == "tap" and self.screen.events[-1].chapter == self.screen.index

    async def show_bars(self) -> None:
        if not self.screen.chrome.visible:
            await self.centre_tap(True)

    async def chrome_round_trip(self, mode: str) -> None:
        """▶ to the long chapter, a right-edge tap turns its page, ◀ back: all in ``mode``."""
        screen, chrome = self.screen, self.screen.chrome
        start = screen.index
        await self.show_bars()
        await self.change(lambda: D1._fire(self.phone, chrome.next_button, "click"), start + 1, mode, "chrome next")
        assert start + 1 == LONG
        await self.centre_tap(False)
        assert (await self.turns.shown())["count"] > 1, "the long chapter should span several pages"
        await self.turns.tap(D1.RIGHT)
        assert await D1._wait(lambda: screen.page_no == 1, 10), "a right-edge tap did not turn the page"
        shown = await self.turns.check_shown(LONG, mode="RAW" if mode == ORIGINAL else "EN")
        _expect_version(shown["text"], LONG, mode)
        assert chrome.page_text.value.startswith("Page 2/")
        self.note("page tap", LONG, mode, shown)
        await self.centre_tap(True)
        await self.change(lambda: D1._fire(self.phone, chrome.prev_button, "click"), start, mode, "chrome prev")


# =====================================================================================
# The owner's scenario
# =====================================================================================


@pytest.mark.parametrize("kind", ["in_progress", "completed"])
def test_owner_report_11_each_segment_shows_its_own_text(phone, kind):
    """Library › the book's page › Start reading, then Original, Bilingual, Translated (and back through
    every other transition): one ``load_request`` per press and the page shows the pressed version; the
    bars, ▶ / ◀ and page taps keep working after each switch. Then Original, back, Continue: the book
    reopens in Korean and still switches."""
    async def scenario():
        seeded = _seed(phone)
        await phone.start()
        engine = D1._engine(phone)
        print(f"[devfix-issue11] page engine: {engine.name}")
        client = D1._attach_webview(phone, engine)
        try:
            row = await _library_row(phone, kind)
            book_page = await _book_page(phone, row)
            assert os.path.normcase(str(book_page.workspace)) == os.path.normcase(str(seeded[kind]))
            screen = await _reader_from(phone, book_page.overview.read_button)
            session = screen.session
            assert screen.renderer == "webview" and session.count == CHAPTERS
            assert session.plan.mode == ("overlay" if kind == "in_progress" else "dual"), session.plan.mode
            assert session.has_alternate and screen.chrome.mode_buttons.visible
            for index in range(CHAPTERS):
                translated = kind == "completed" or index < TRANSLATED_IN_PROGRESS
                assert session.available_modes(index) == {ORIGINAL: True, TRANSLATED: True, BILINGUAL: translated}
            owner = Owner(phone, client, screen)
            await owner.first_page(0, TRANSLATED)

            # the owner's complaint: each segment, and every transition between them
            for mode in (ORIGINAL, BILINGUAL, TRANSLATED):
                await owner.show_bars()
                await owner.press(mode)
                await owner.centre_tap(False)   # the new page's taps reach the Reader
                await owner.chrome_round_trip(mode)
            for mode in (BILINGUAL, ORIGINAL, TRANSLATED):
                await owner.show_bars()
                await owner.press(mode)
                await owner.centre_tap(False)
            assert owner.presses == 6

            if kind == "in_progress":
                # chapter 4 is not translated yet: Translated shows the raw there (the overlay falls back, as on
                # the desktop), Bilingual is off; Original still loads its page
                last = CHAPTERS - 1
                await owner.show_bars()
                shown = await owner.turns.turn(lambda: _slide(phone, screen, last), last, mode="RAW")
                _expect_version(shown["text"], last, ORIGINAL)
                owner.note("slider", last, TRANSLATED, shown)
                assert screen.chrome.segments[BILINGUAL].disabled is True
                assert screen.chrome.mode_buttons.selected == [TRANSLATED] and session.flavor == TRANSLATED
                await owner.press(ORIGINAL)
                await owner.change(lambda: _slide(phone, screen, 0), 0, ORIGINAL, "slider")

            # left in Original: the saved position says so, and Continue reopens the book in Korean
            if session.flavor != ORIGINAL:
                await owner.show_bars()
                await owner.press(ORIGINAL)
            print(f"[devfix-issue11] {kind}:", *owner.log, sep="\n  ")
            await D1._leave(phone, screen)
            saved = phone.app.prefs.reader_position(screen.bid)
            assert saved["mode"] == ORIGINAL and saved["chapter"] == 0, saved
            assert await D1._wait(lambda: phone.app.shell.top_screen is book_page, 10)
            overview = book_page.overview
            assert await D1._wait(lambda: overview.continue_button.visible is True, 15), "no Continue button"
            screen = await _reader_from(phone, overview.continue_button)
            assert screen.session.flavor == ORIGINAL
            owner = Owner(phone, client, screen)
            await owner.first_page(0, ORIGINAL)
            for mode in (TRANSLATED, BILINGUAL, ORIGINAL, TRANSLATED):
                await owner.show_bars()
                await owner.press(mode)
                await owner.centre_tap(False)
            print(f"[devfix-issue11] {kind} reopened with Continue:", *owner.log, sep="\n  ")
            assert engine.exceptions == [], engine.exceptions  # the page scripts never threw
        finally:
            await phone.stop()

    asyncio.run(scenario())
