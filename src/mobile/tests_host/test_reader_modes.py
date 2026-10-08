"""Owner's device report #11 on the U8 APK (2026-10-08): "The Translated, Bilingual and Original buttons are
all just the Translated button".

What the owner saw: whichever segment was pressed, the Reader kept showing the translated text. The
session switched the flavour and published the right page (Korean only, interleaved, English only),
but the Reader only assigned the WebView's ``url``, which flet-webview 1.0.3 reads once (``initState``),
so the page on screen never changed while the segment's highlight moved. The devfix3 Reader navigation
(``navigate_webview``: ``load_request`` on the same WebView) fixes the transport; this file pins every
segment on both kinds of book the owner has, and the hardening around a switch:

* a Completed (compiled) book with its raw EPUB linked (``dual``), and an in-progress book read over its
  raw EPUB (``overlay``): each press of Original, Bilingual and Translated sends exactly one
  ``load_request`` to the same WebView, and the page the WebView shows is Korean only, Korean and English
  interleaved (``glr-bi``), then English only; the segment and the session agree;
* a chapter of the in-progress book with no translation yet: Bilingual is disabled, Original shows the raw;
* a switch that cannot load the raw EPUB keeps the translation on screen and the segment on Translated
  (desktop ``_restore_raw_toggle_value``), says "Could not switch", and the next chapter is English; a book
  saved in Original whose raw EPUB no longer loads opens on its translation;
* one switch at a time (desktop ``_raw_toggle_in_flight``): a second tap while the raw EPUB loads snaps
  back and never mixes one switch's EPUB with the other's flavour;
* the native ("Lightweight") reader shows the three versions too.

Every press is what Flutter sends for a SegmentedButton tap: the ``selected`` property patch, then the
``change`` event. The real app runs on the fake Flet session as Android with test_devfix_issue1's WebView
client stand-in (headless Chrome when installed, else its ShellModel), storage under tmp.

Run from src/mobile (mobile venv):
    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_reader_modes.py
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import re
import time
import urllib.request
from pathlib import Path
from typing import Any

import pytest

_SPEC = importlib.util.spec_from_file_location("_glossarion_devfix1_helpers_modes",
                                               Path(__file__).with_name("test_devfix_issue1.py"))
D1 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(D1)

phone = D1.phone  # the real app on the fake Flet session as Android (isolated storage), from devfix issue 1

needs_msgpack = pytest.mark.skipif(not D1._has("msgpack"), reason="msgpack not installed")

ORIGINAL, TRANSLATED, BILINGUAL = "original", "translated", "bilingual"
TRANSLATED_CHAPTERS = (1, 2, 3)  # the in-progress book's finished chapters (1-based)


def _in_progress_book(folder: Path) -> dict:
    """The owner's raw EPUB plus a translation workspace with chapters 1-3 done (Library "In progress")."""
    raw = D1._write_epub(folder / "Owner Book.epub", marker="RAW", lang="ko", filler=D1._KO, link=False)
    workspace = folder / "Output" / "Owner Book"
    workspace.mkdir(parents=True)
    (workspace / "source_epub.txt").write_text(str(raw), encoding="utf-8")
    chapters: dict = {}
    for number in range(1, D1.CHAPTERS + 1):
        entry = {"status": "pending", "original_basename": f"chapter{number:04d}.xhtml"}
        if number in TRANSLATED_CHAPTERS:
            mark = f"EN-MARK-{number:02d}"
            paragraphs = "".join(f"<p>{mark} {n + 1}. {D1._EN * 6}</p>"
                                 for n in range(16 if number - 1 == D1.LONG else 1))
            (workspace / f"response_chapter{number:04d}.html").write_text(
                f"<html><head><title>EN chapter {number}</title></head><body><h1>EN chapter {number}</h1>"
                f"<p>{mark}</p>{paragraphs}</body></html>", encoding="utf-8")
            entry.update(status="completed", output_file=f"response_chapter{number:04d}.html")
        chapters[str(number)] = entry
    (workspace / "translation_progress.json").write_text(
        json.dumps({"chapters": chapters, "chapter_chunks": {}, "version": "2.1"}), encoding="utf-8")
    return {"name": "Owner Book", "path": str(workspace), "output_folder": str(workspace), "is_in_progress": True}


async def _press(phone, screen: Any, mode: str) -> None:
    """A SegmentedButton tap as Flutter sends it: ``selected`` patched first, then the ``change`` event."""
    buttons = screen.chrome.mode_buttons
    assert phone.session.index.get(buttons._i) is buttons, "the mode segments are not on screen"
    phone.session.apply_patch(buttons._i, {"selected": [mode]})
    await phone.session.dispatch_event(buttons._i, "change", [mode])


def _page_html(url: str) -> str:
    with urllib.request.urlopen(url.split("#", 1)[0], timeout=10) as response:
        return response.read().decode("utf-8")


def _kinds(text: str, index: int) -> set:
    """Which versions of chapter ``index`` (0-based) a text shows: {"RAW", "EN"}."""
    return {kind for kind, chapter in D1._marks(text) if chapter == index}


async def _switch(phone, client, screen, turns, mode: str, index: int) -> dict:
    """Press ``mode`` on chapter ``index``: exactly one ``load_request`` on the same WebView, and the
    page the WebView then shows (and the page served at its URL) is that version of the chapter."""
    expected = {ORIGINAL: {"RAW"}, TRANSLATED: {"EN"}, BILINGUAL: {"RAW", "EN"}}[mode]
    shown = await turns.turn(lambda: _press(phone, screen, mode), index, mode="RAW" if mode == ORIGINAL else "EN")
    assert _kinds(shown["text"], index) == expected, (mode, sorted(D1._marks(shown["text"])))
    html = _page_html(client.shown_url)
    assert ("glr-bi" in html) == (mode == BILINGUAL)
    assert screen.chrome.mode_buttons.selected == [mode] and screen.session.flavor == mode
    assert not screen._mode_busy
    return shown


@needs_msgpack
@pytest.mark.parametrize("kind", ["completed", "in_progress"])
def test_owner_report_11_every_segment_shows_its_own_text(phone, kind):
    """The owner's complaint: on a Completed book with its raw EPUB and on an in-progress book, Original,
    Bilingual and Translated each load their own page into the WebView (one ``load_request`` per press)."""
    async def scenario():
        if kind == "in_progress":
            phone.book = _in_progress_book(phone.tmp / "progress")
        await phone.start()
        engine = D1._engine(phone)
        client = D1._attach_webview(phone, engine)
        try:
            screen = await D1._open(phone, chapter=0)
            session = screen.session
            assert screen.renderer == "webview"
            assert session.plan.mode == ("dual" if kind == "completed" else "overlay")
            assert screen.chrome.mode_buttons.visible and session.flavor == TRANSLATED
            assert session.available_modes(0) == {ORIGINAL: True, TRANSLATED: True, BILINGUAL: True}
            turns = D1.Turns(phone, client, screen)
            await turns.page_ready(set(), 0)
            assert _kinds((await turns.check_shown(0))["text"], 0) == {"EN"}
            await _switch(phone, client, screen, turns, ORIGINAL, 0)
            await _switch(phone, client, screen, turns, BILINGUAL, 0)
            await _switch(phone, client, screen, turns, TRANSLATED, 0)
            await _switch(phone, client, screen, turns, BILINGUAL, 0)
            await _switch(phone, client, screen, turns, ORIGINAL, 0)
            await _switch(phone, client, screen, turns, TRANSLATED, 0)
            if kind == "in_progress":
                # chapter 4 has no translation yet: Bilingual is off, Original shows the raw text
                untranslated = len(TRANSLATED_CHAPTERS)

                async def slide() -> None:
                    screen.chrome.slider.value = untranslated
                    await D1._fire(phone, screen.chrome.slider, "change_end", untranslated)

                shown = await turns.turn(slide, untranslated, mode="RAW")
                assert _kinds(shown["text"], untranslated) == {"RAW"}
                assert screen.chrome.segments[BILINGUAL].disabled is True
                assert session.available_modes(untranslated)[BILINGUAL] is False
                await _switch(phone, client, screen, turns, ORIGINAL, untranslated)
                # Translated without a translation shows the raw text too (the overlay falls back, desktop parity)
                shown = await turns.turn(lambda: _press(phone, screen, TRANSLATED), untranslated, mode="RAW")
                assert _kinds(shown["text"], untranslated) == {"RAW"} and session.flavor == TRANSLATED
                assert screen.chrome.mode_buttons.selected == [TRANSLATED]
            assert engine.exceptions == [], engine.exceptions
        finally:
            await phone.stop()

    asyncio.run(scenario())


@needs_msgpack
def test_a_switch_that_cannot_load_the_original_keeps_the_translation(phone, monkeypatch):
    """The raw EPUB of a Completed book does not load: Original and Bilingual say "Could not switch", the
    translation stays on screen, the segment snaps back to Translated (no ``load_request``), and the next
    chapter is English. Reopened in Original, the book opens on its translation."""
    from glossarion_mobile.ui.reader import session as rs

    raw = os.path.normcase(os.path.abspath(phone.book["raw_source_path"]))
    load_epub = rs.DocEngine.load_epub
    broken = {"raw": False}

    def flaky_load_epub(self, path, **kwargs):
        if broken["raw"] and os.path.normcase(os.path.abspath(path)) == raw:
            raise OSError("the original EPUB is unreadable")
        return load_epub(self, path, **kwargs)

    monkeypatch.setattr(rs.DocEngine, "load_epub", flaky_load_epub)

    async def scenario():
        await phone.start()
        engine = D1._engine(phone)
        client = D1._attach_webview(phone, engine)
        try:
            screen = await D1._open(phone, chapter=0)
            session = screen.session
            notes: list = []
            notify = screen.deps.notify or (lambda *args: None)
            screen.deps.notify = lambda message, *rest: (notes.append(message), notify(message, *rest))
            turns = D1.Turns(phone, client, screen)
            await turns.page_ready(set(), 0)
            broken["raw"] = True
            for mode in (ORIGINAL, BILINGUAL):
                loads, doc, before = len(client.loads), screen.current_doc, len(notes)
                await _press(phone, screen, mode)
                assert await D1._wait(lambda: len(notes) > before and not screen._mode_busy, 10), notes
                assert notes[-1].startswith("Could not switch")
                await asyncio.sleep(0.3)
                assert len(client.loads) == loads and screen.current_doc == doc  # the translation stays
                assert session.flavor == TRANSLATED and screen.chrome.mode_buttons.selected == [TRANSLATED]
                assert os.path.normcase(os.path.abspath(session.active_path)) != raw
                assert _kinds((await turns.shown())["text"], 0) == {"EN"}
            shown = await turns.turn(lambda: D1._fire(phone, screen.chrome.next_button, "click"), 1)
            assert _kinds(shown["text"], 1) == {"EN"} and screen.chrome.mode_buttons.selected == [TRANSLATED]
            await D1._leave(phone, screen)

            # a book saved / opened in Original whose raw EPUB no longer loads opens on its translation
            from glossarion_mobile.ui.reader.reader_view import ReaderScreen

            said: list = []
            screen_notify = ReaderScreen.notify

            def recording(self, message, *args, **kwargs):
                said.append(message)
                return screen_notify(self, message, *args, **kwargs)

            monkeypatch.setattr(ReaderScreen, "notify", recording)
            reopened = await D1._open(phone, chapter=1, mode=ORIGINAL)
            assert "The original could not be opened; showing the translation" in said
            assert reopened.state == "ready" and reopened.session.flavor == TRANSLATED
            assert reopened.chrome.mode_buttons.selected == [TRANSLATED]
            turns = D1.Turns(phone, client, reopened)
            await turns.page_ready(set(), 1)
            assert _kinds((await turns.check_shown(1))["text"], 1) == {"EN"}
        finally:
            await phone.stop()

    asyncio.run(scenario())


@needs_msgpack
def test_one_switch_at_a_time_while_the_original_loads(phone, monkeypatch):
    """Original, then Translated 0.2 s later while the raw EPUB still loads (a big book on a phone): the
    second tap snaps back instead of starting a second switch, so the segment, the session and the page
    agree on Original at the end, with one ``load_request``."""
    from glossarion_mobile.ui.reader import session as rs

    raw = os.path.normcase(os.path.abspath(phone.book["raw_source_path"]))
    load_epub = rs.DocEngine.load_epub

    def slow_load_epub(self, path, **kwargs):
        if os.path.normcase(os.path.abspath(path)) == raw:
            time.sleep(1.2)
        return load_epub(self, path, **kwargs)

    monkeypatch.setattr(rs.DocEngine, "load_epub", slow_load_epub)

    async def scenario():
        await phone.start()
        engine = D1._engine(phone)
        client = D1._attach_webview(phone, engine)
        try:
            screen = await D1._open(phone, chapter=0)
            turns = D1.Turns(phone, client, screen)
            await turns.page_ready(set(), 0)

            async def two_taps():
                await _press(phone, screen, ORIGINAL)
                await asyncio.sleep(0.2)
                assert screen._mode_busy
                await _press(phone, screen, TRANSLATED)
                # snapped back to the version that is loading (no second switch starts)
                assert await D1._wait(lambda: screen.chrome.mode_buttons.selected == [ORIGINAL], 5)
                assert screen._mode_busy and screen.session.flavor == ORIGINAL

            shown = await turns.turn(two_taps, 0, mode="RAW")
            assert _kinds(shown["text"], 0) == {"RAW"}
            assert screen.session.flavor == ORIGINAL and screen.chrome.mode_buttons.selected == [ORIGINAL]
            assert os.path.normcase(os.path.abspath(screen.session.active_path)) == raw
            await _switch(phone, client, screen, turns, TRANSLATED, 0)
        finally:
            await phone.stop()

    asyncio.run(scenario())


@needs_msgpack
def test_lightweight_reader_shows_each_version(phone):
    """The native renderer (⋯ › Lightweight reader) re-renders its blocks for each segment."""
    async def scenario():
        phone.book = _in_progress_book(phone.tmp / "progress")
        await phone.start()
        phone.app.prefs.set("reader_prefs", {"lightweight": True})
        try:
            screen = await D1._open(phone, chapter=0)
            assert screen.renderer == "native" and screen.webview is None

            def shows(expected: set) -> bool:
                body = " ".join(D1.ListViewClient._text(screen.fallback.list_view))
                return screen.fallback_generation == screen.render_generation and _kinds(body, 0) == expected

            assert await D1._wait(lambda: shows({"EN"}), 10)
            for mode, expected in ((ORIGINAL, {"RAW"}), (BILINGUAL, {"RAW", "EN"}), (TRANSLATED, {"EN"})):
                await _press(phone, screen, mode)
                assert await D1._wait(lambda: shows(expected) and not screen._mode_busy, 10), mode
                assert screen.chrome.mode_buttons.selected == [mode] and screen.session.flavor == mode
        finally:
            await phone.stop()

    asyncio.run(scenario())


def test_mode_switch_guard_is_in_the_reader_screen():
    """Guard: the Reader keeps one switch at a time and restores the flavour on a failed switch (both in
    ``ReaderScreen``; the chrome only reports taps)."""
    source = (Path(__file__).resolve().parents[1] / "app" / "glossarion_mobile" / "ui" / "reader"
              / "reader_view.py").read_text(encoding="utf-8")
    body = source.split("async def set_mode", 1)[1].split("\n    async def ", 1)[0]
    assert "self._mode_busy" in body and "_switch_flavor" in body
    assert re.search(r"async def _switch_flavor\(.*?\n(?:.*\n)*?.*session\.set_flavor, previous", source)


@needs_msgpack
def test_bilingual_survives_a_reload_and_never_doubles_a_chapter_without_both_versions(phone):
    """Skeptic findings B + C (issue 11): a dual book in Bilingual keeps the raw text after a reload
    (Show special files), and Bilingual carried onto a chapter without a translation shows it once."""
    from glossarion_mobile.ui.reader import model as rm
    from glossarion_mobile.ui.reader import session as rs

    hangul = re.compile(r"[가-힣]")
    engine = rs.DocEngine()
    overlay = rs.ReaderSession(rs.plan_open(_in_progress_book(phone.tmp / "progress"), engine=engine), engine=engine,
                               cache_dir=str(phone.tmp / "cache"))
    overlay.load()
    overlay.set_flavor(rm.BILINGUAL)
    untranslated = len(TRANSLATED_CHAPTERS)
    assert overlay.effective_flavor(untranslated) == rm.TRANSLATED and overlay.effective_flavor(0) == rm.BILINGUAL
    assert len(hangul.findall(overlay.chapter_html(untranslated))) == \
        len(hangul.findall(overlay.chapter_html(untranslated, rm.ORIGINAL)))
    assert "glr-bi" in overlay.chapter_html(0)
    overlay.close()
    dual = rs.ReaderSession(rs.plan_open(phone.book, engine=engine), engine=engine, cache_dir=str(phone.tmp / "cache2"))
    dual.load()
    dual.set_flavor(rm.BILINGUAL)
    dual.dual_cache.clear()
    dual.load()  # what _on_special_files does
    assert _kinds(re.sub(r"<[^>]+>", " ", dual.chapter_html(0)), 0) == {"RAW", "EN"}
    dual.close()
