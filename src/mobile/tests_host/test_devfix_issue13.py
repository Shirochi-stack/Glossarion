"""Acceptance test for the owner's device report item 13 (devfix4, 2026-10-08): "'↑ Scroll for earlier
messages (N hidden)' does nothing, and a book chat shows only one request card".

What the owner saw on the phone, and why (diagnosis + skeptic re-check): the render re-tailed the window
whenever it ended near the tail, so every tap was undone in the same call (book chats with big chapters,
pasted-chapter text chats, chats of limit + 1 messages); a book turn whose file card fell out of the
message window collapsed into one orphan 'Attachment' card holding a few request rows; jump-to and
search were snapped back the same way; a scroll to the top loaded nothing; a live run's next tick pulled
the view back to the end.

Every scenario runs the REAL app objects headless: ``ChatView`` + ``Transcript`` + ``JobCard`` over
``ChatStoreAdapter`` on the desktop-format store (``direct_text_store.ChatStore``), ``ChatRuns`` with
test_chat's ``FakeJobService`` (one shared ``direct_text_stream.DirectTextStream`` per job), the real
Jump-to sheet and search paths, and - for the client side - test_bootstrap's in-memory Flet session
(client "click" / "scroll" events dispatched by control id through Flet's own event decoding; the patches
and ``invoke_method`` calls recorded as the socket transport sends them):

1. many-card chats (a 60-message text chat, a pasted-chapters chat whose window the 120k budget cuts, a
   chat of limit + 1 messages, 20 book turns with 12k-character chapters, a mixed chat of text turns and
   40-chapter books, the owner's single 40-chapter book): every tap on the loader row renders earlier
   cards, "N hidden" drops to 0, the rendered cards are exactly the window's, every book keeps its own
   titled JobCard (never 'Attachment'), and the slid window survives every kind of re-render;
2. scroll to top: a user scroll near the top (or a pull past it) loads one page per gesture and puts the
   card that was at the top back in view; the mirror at the bottom loads newer cards;
3. jump-to / search: ``jump_to(0)``, the Jump-to sheet and a search hit render the first card (and a
   request row inside the first book's JobCard) and stay there;
4. appends: a live text run and a live book job (through its glossary gate) keep a window the user slid
   up - no scroll to the end, the ↓ button shows, "N newer" grows - and ↓ brings the newest back; a window
   that ended at the tail follows the new cards (the desktop ``_update_history_window_after_append``);
   timing included: a tap that lands while a scroll to the end is settling (a stream repaint's, a job
   snapshot's follow render) and a jump during a live run are not undone by that scroll;
5. the owner's flow end to end: a real book job (the 12-chapter self-test EPUB, 11-17k-character chapters)
   through the real JobService, ChatRuns and shared pipeline against the offline fake OpenAI server
   (``diagnostics.e2e.E2ESession``), in a chat with 60 earlier messages, in a child process: while it
   runs the user loads earlier cards (tap and scroll to the top) and nothing pulls them back; after it
   the book is its file card + its own JobCard ("Done · 12/12 chapters", every request), the taps reach
   the first card and jump-to reaches it and a chapter's request row.

Real data is never touched: test_chat's ``isolated_env`` points HOME / USERPROFILE / APPDATA /
GLOSSARION_LIBRARY_DIR / GLOSSARION_DATA_DIR / OUTPUT_DIRECTORY at tmp_path and turns HTTP logging off;
the child process of 5. gets the same, its own sandbox (``E2ESession``: config.json redirected, process
and network guards, write audit) and fails on any write outside tmp_path.

Run from src/mobile with the mobile venv (Windows: ``src/mobile/.venv``, which borrows the 3.12 user site,
so keep PYTHONPATH; CI: the uv env)::

    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_devfix_issue13.py
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
import types
from pathlib import Path

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = MOBILE_DIR / "app"
SRC_DIR = MOBILE_DIR.parent
for _entry in (str(APP_DIR), str(SRC_DIR)):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)


def _load(filename: str, alias: str):
    spec = importlib.util.spec_from_file_location(alias, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# the issue-12/13 chat fixtures and helpers (test_chat's ChatView harness, the chat builders, ``_quiet``)
_TW = _load("test_chat_transcript_window.py", "_glossarion_tw_helpers_issue13")
_TC = _TW._TC
needs_backend = _TW.needs_backend
needs_flet = _TW.needs_flet
if _TC is not None:
    desktop_store_cls = _TC.desktop_store_cls  # noqa: F811  (pytest fixtures)
    isolated_env = _TC.isolated_env

CID = "5"  # test_chat's empty chat in its desktop-format history
LIMIT = 20  # the desktop default rendered-card limit
PAGE = 6  # the desktop page: limit // 3

pytestmark = [needs_backend, needs_flet]


# ==========================================================================
# Helpers
# ==========================================================================


def _shown(view) -> list:
    """(kind, message index) of every rendered card, read from the card controls themselves."""
    from glossarion_mobile.ui.chat.cards import JobCard
    from glossarion_mobile.ui.chat.messages import AssistantMessage, UserBubble, UserFileCard
    from glossarion_mobile.ui.chat.transcript import CardSlot

    out = []
    for control in view.transcript.messages:
        if not isinstance(control, CardSlot):
            continue
        card = control.card
        if isinstance(card, JobCard):
            out.append(("job", int(str(control.slot_key).split("-", 1)[1])))
        elif isinstance(card, UserFileCard):
            out.append(("user_file", card.index))
        elif isinstance(card, UserBubble):
            out.append(("user", card.index))
        elif isinstance(card, AssistantMessage):
            out.append(("assistant", card.index))
        else:  # pragma: no cover - a card kind this test does not build
            out.append((type(card).__name__, None))
    return out


def _window_cards(view) -> list:
    start, end = view.window
    return [(item.kind, item.index) for item in view.items[start:end]]


def _check_render(view) -> None:
    """The rendered cards are the window's, and the loader rows say how many cards are hidden."""
    from glossarion_mobile.ui.chat.transcript import EARLIER_TEMPLATE, LATER_TEMPLATE

    transcript = view.transcript
    start, end = view.window
    assert _shown(view) == _window_cards(view)
    assert transcript.hidden_before == start and transcript.hidden_after == len(view.items) - end
    controls = list(transcript.controls)
    if start:
        assert controls[0] is transcript.earlier_box
        assert transcript.earlier_row.content == EARLIER_TEMPLATE.format(n=start)
    else:
        assert transcript.earlier_box not in controls
    if end < len(view.items):
        assert transcript.later_box in controls
        assert transcript.later_row.content == LATER_TEMPLATE.format(n=len(view.items) - end)
    else:
        assert transcript.later_box not in controls


def _check_book_cards(view, messages) -> None:
    """Every rendered JobCard is its own book's (title = the turn's file), never the orphan 'Attachment'."""
    from glossarion_mobile.ui.chat.cards import JobCard
    from glossarion_mobile.ui.chat.transcript import CardSlot

    for control in view.transcript.messages:
        if isinstance(control, CardSlot) and isinstance(control.card, JobCard):
            index = int(str(control.slot_key).split("-", 1)[1])
            assert messages[index][0] == "user_file", control.slot_key
            assert control.card.title_text.value == messages[index][1] != "Attachment"


async def _rerender_everything(view) -> None:
    """Every re-render a chat sees without the user moving: a render with follow (job finishes, sends of
    other chats), a run notification, a store change. None may move the window."""
    window, shown = view.window, _shown(view)
    view.render_transcript(follow=True)
    view._on_run_changed(CID)
    view._on_store_changed()
    view.render_transcript()
    assert view.window == window and _shown(view) == shown


async def _tap_earlier_until_top(view, calls, messages=None) -> list:
    """Tap "↑ Scroll for earlier messages (N hidden)" (the TextButton's own handler) until nothing is
    hidden; every tap must render earlier cards. Returns the hidden counts, opening one first."""
    transcript = view.transcript
    hidden = [transcript.hidden_before]
    while transcript.hidden_before:
        assert len(hidden) < 60, "the taps never reach the first card"
        before = _shown(view)
        first = view.window[0]
        transcript.earlier_row.on_click(None)
        await _TW._quiet(calls)
        _check_render(view)
        assert transcript.hidden_before < hidden[-1], "the tap loaded earlier cards"
        assert view.window[0] < first and _shown(view)[0] not in before, "an earlier card is now the first one"
        assert not transcript.follow_tail, "reading earlier cards: nothing pulls the view back to the end"
        if messages is not None:
            _check_book_cards(view, messages)
        await _rerender_everything(view)
        hidden.append(transcript.hidden_before)
    return hidden


def _recorder(view) -> list:
    """Record ``Transcript.scroll_to`` calls (the client's scroll commands)."""
    scrolls: list = []

    async def scroll_to(**kwargs):
        scrolls.append(kwargs)

    view.transcript.scroll_to = scroll_to
    return scrolls


def _to_end(scrolls) -> list:
    return [s for s in scrolls if s.get("offset") == -1]


def _scroll_event(pixels, delta=None, kind="update", overscroll=None, maximum=9000.0):
    return types.SimpleNamespace(event_type=kind, pixels=float(pixels), min_scroll_extent=0.0,
                                 max_scroll_extent=float(maximum), viewport_dimension=800.0, scroll_delta=delta,
                                 overscroll=overscroll)


def _books(tmp_path: Path, count: int, chapters: int, chars: int, first: int = 1) -> list:
    messages: list = []
    for number in range(first, first + count):
        book = tmp_path / f"book{number:02d}.epub"
        book.write_bytes(b"PK\x03\x04fake epub")
        messages += _TW._book_turn(str(book), chapters, chars, str(tmp_path / "Output" / "Attachments" / book.stem))
    return messages


def _row_labels(card) -> list:
    return [row.content.controls[0].controls[0].value for row in card.requests_column.controls]


def _run(scenario_factory, desktop_store_cls, tmp_path, **kwargs):
    """Run ``scenario_factory(chat)`` on a fresh real ChatView in its own event loop."""

    async def main():
        chat = _TC._chat_view(desktop_store_cls, tmp_path, cid=CID, **kwargs)
        try:
            await scenario_factory(chat)
        finally:
            for task in chat.calls.tasks:
                task.cancel()
            _TC._close(chat)

    asyncio.run(main())


@pytest.fixture
def no_highlight(monkeypatch):
    from glossarion_mobile.ui.chat import chat_view as cv

    monkeypatch.setattr(cv, "HIGHLIGHT_SECONDS", 0)


# ==========================================================================
# 1. Many-card chats: the loader tap loads earlier cards, N drops to 0
# ==========================================================================


def test_60_message_text_chat_each_tap_loads_a_page_of_earlier_cards(desktop_store_cls, isolated_env):
    async def scenario(chat):
        view, adapter, calls = chat.view, chat.adapter, chat.calls
        scrolls = _recorder(view)
        adapter.append_messages(CID, _TW._text_turns(30, 200))
        view.load_chat(CID)
        await _TW._quiet(calls)
        assert len(view.items) == 60 and view.window == (40, 60)
        _check_render(view)
        assert view.transcript.earlier_row.content == "↑ Scroll for earlier messages (40 hidden)"
        opened = len(scrolls)

        hidden = await _tap_earlier_until_top(view, calls)
        assert hidden == [40, 34, 28, 22, 16, 10, 4, 0]
        assert view.window == (0, LIMIT) and _shown(view)[0] == ("user", 0)
        assert view.transcript.later_row.content == "Scroll for newer messages (40 hidden) ↓"
        assert _to_end(scrolls[opened:]) == [], "no tap scrolled the view to the end"

        # the mirror row: newer cards, one page at a time
        view.transcript.later_row.on_click(None)
        await _TW._quiet(calls)
        assert view.window == (PAGE, LIMIT + PAGE)
        _check_render(view)
        # ↓ : the newest cards again, following them
        await view._jump_to_end()
        assert view.window == (40, 60) and view.transcript.follow_tail
        _check_render(view)

    _run(scenario, desktop_store_cls, isolated_env)


def test_pasted_chapter_chat_and_limit_plus_one_chat_slide(desktop_store_cls, isolated_env):
    """The skeptic's stuck text chats: a window the 120k budget cuts (12 turns of 5k source + 15k
    translation opened at "12 hidden" and stayed there) and a chat of exactly limit + 1 messages."""

    async def scenario(chat):
        view, adapter, calls = chat.view, chat.adapter, chat.calls
        _recorder(view)
        adapter.append_messages(CID, _TW._text_turns(12, 15_000, source=5_000))
        view.load_chat(CID)
        await _TW._quiet(calls)
        assert view.window == (12, 24)
        assert await _tap_earlier_until_top(view, calls) == [12, 6, 0]
        assert view.window == (0, LIMIT) and view.transcript.hidden_after == 4

        # one more card appended while the user reads the start: the window stays (it did not end at the tail)
        adapter.append_messages(CID, [("user", "one more question")])
        view._on_run_changed(CID)
        assert view.window == (0, LIMIT) and view.transcript.hidden_after == 5
        _check_render(view)

    _run(scenario, desktop_store_cls, isolated_env)


def test_limit_plus_one_chat_tap_reaches_the_first_card(desktop_store_cls, isolated_env):
    async def scenario(chat):
        view, adapter, calls = chat.view, chat.adapter, chat.calls
        _recorder(view)
        adapter.append_messages(CID, _TW._text_turns(10, 200) + [("user", "the 21st message")])
        view.load_chat(CID)
        await _TW._quiet(calls)
        assert view.window == (1, 21) and view.transcript.earlier_row.content.endswith("(1 hidden)")
        assert await _tap_earlier_until_top(view, calls) == [1, 0]
        assert view.window == (0, 20) and view.transcript.later_row.content == "Scroll for newer messages (1 hidden) ↓"

    _run(scenario, desktop_store_cls, isolated_env)


BOOK_CHATS = {
    # the owner's chat A / chat B of the diagnosis: one book, big chapters (a dead "33 hidden" loader and one
    # orphan 'Attachment' card before the fix)
    "one book, 40 x 12k chapters": lambda tmp: _books(tmp, 1, 40, 12_000),
    "one book, 12 x 15k chapters": lambda tmp: _books(tmp, 1, 12, 15_000),
    # many book turns + a text question: 41 cards, the window opens on a JobCard whose file card is hidden
    "20 books of 5 x 12k chapters + a question": lambda tmp: _books(tmp, 20, 5, 12_000)
    + [("user", "What should I translate next?")],
    # text turns of pasted chapters, then 40-chapter books (the budget and the card limit both matter)
    "10 pasted-chapter turns + 3 books of 40 x 12k": lambda tmp: _TW._text_turns(10, 15_000, source=5_000)
    + _books(tmp, 3, 40, 12_000),
}


@pytest.mark.parametrize("shape", list(BOOK_CHATS))
def test_book_chats_show_every_book_and_tap_to_the_first_card(desktop_store_cls, isolated_env, shape):
    from glossarion_mobile.ui.chat.cards import EARLIER_REQUESTS_TEMPLATE, JobCard

    async def scenario(chat):
        view, adapter, calls = chat.view, chat.adapter, chat.calls
        scrolls = _recorder(view)
        adapter.append_messages(CID, BOOK_CHATS[shape](isolated_env))
        messages = adapter.messages(CID)
        view.load_chat(CID)
        await _TW._quiet(calls)
        _check_render(view)
        _check_book_cards(view, messages)
        books = [i for i, m in enumerate(messages) if m[0] == "user_file"]
        # one file card + one JobCard per book turn, whatever the size of its chapters
        assert [item.index for item in view.items if item.kind == "job"] == books
        assert [item.index for item in view.items if item.kind == "user_file"] == books
        for control in view.transcript.cards:
            if isinstance(control, JobCard):
                assert control.state_text.value == "Done"
                chapters = sum(1 for m in messages if m[0] == "assistant" and str(m[5]).startswith("Chapter")
                               and str(m[4]).endswith(Path(control.title_text.value).stem))
                assert control.requests_tile.title == f"Requests ({chapters + 1})"  # + its glossary request
        opened = len(scrolls)

        hidden = await _tap_earlier_until_top(view, calls, messages)
        assert hidden[-1] == 0 and view.window[0] == 0
        assert _shown(view)[0] == (view.items[0].kind, view.items[0].index)
        assert _to_end(scrolls[opened:]) == []
        if len(view.items) <= 2:  # a single book: both its cards are always shown, no loader at all
            assert hidden == [0] and view.transcript.earlier_box not in view.transcript.controls
            job = view.transcript.cards[-1]
            total = len(messages) - 3  # every assistant card but the report and the actions card
            assert isinstance(job, JobCard) and job.requests_tile.title == f"Requests ({total})"
            assert len(job.requests_column.controls) == min(LIMIT, total)
            assert job.earlier_requests_button.visible == (total > LIMIT)
            if total > LIMIT:
                assert job.earlier_requests_button.content == EARLIER_REQUESTS_TEMPLATE.format(
                    n=min(LIMIT, total - LIMIT), hidden=total - LIMIT)
            while job.earlier_requests_button.visible:
                job.earlier_requests_button.on_click(None)
            assert len(job.requests_column.controls) == total
            labels = _row_labels(job)
            assert labels[0].startswith("Glossary") and labels[-1].startswith(f"Chapter {total - 1} (chunk")
        else:
            assert len(hidden) > 2 and all(a > b for a, b in zip(hidden, hidden[1:]))

    _run(scenario, desktop_store_cls, isolated_env)


# ==========================================================================
# 2. Scroll to top: a user scroll near the top loads earlier cards
# ==========================================================================


def test_scrolling_to_the_top_loads_earlier_cards_once_per_gesture(desktop_store_cls, isolated_env):
    from glossarion_mobile.ui.chat.transcript import EDGE_HOLD_SECONDS, EDGE_LOAD_PX, CardSlot

    async def scenario(chat):
        view, adapter, calls = chat.view, chat.adapter, chat.calls
        scrolls = _recorder(view)
        adapter.append_messages(CID, _books(isolated_env, 20, 5, 12_000) + [("user", "What next?")])
        messages = adapter.messages(CID)
        view.load_chat(CID)
        await _TW._quiet(calls)
        transcript = view.transcript
        assert view.window == (21, 41)
        opened = len(scrolls)
        transcript.on_scroll(_scroll_event(3000.0, -40.0))  # scrolling up, far from the top: nothing
        transcript.on_scroll(_scroll_event(EDGE_LOAD_PX - 1, 30.0))  # near the top, going down: nothing
        assert view.window == (21, 41) and transcript.edge_loads == 0

        hidden = [transcript.hidden_before]
        pulls = 0
        while transcript.hidden_before:
            top = next(c for c in transcript.messages if isinstance(c, CardSlot)).slot_key
            loads = transcript.edge_loads
            if pulls % 2:  # a pull past the top (an overscroll at pixels 0) is a scroll to the top too
                event = _scroll_event(0.0, kind="overscroll", overscroll=-14.0)
            else:
                event = _scroll_event(150.0, -35.0)
            transcript.on_scroll(event)
            transcript.on_scroll(_scroll_event(40.0, -35.0))  # the same gesture goes on: no second load
            assert transcript.edge_loads == loads + 1
            await _TW._quiet(calls)  # the viewport restore (keep_in_view)
            assert scrolls[-1]["duration"] == 0 and getattr(scrolls[-1]["scroll_key"], "value", None) == top, \
                "the card that was at the top is put back in view"
            _check_render(view)
            _check_book_cards(view, messages)
            assert transcript.hidden_before < hidden[-1] and transcript.slot_for(top) is not None
            hidden.append(transcript.hidden_before)
            pulls += 1
            await asyncio.sleep(EDGE_HOLD_SECONDS + 0.05)  # the next gesture
        assert hidden == [21, 15, 9, 3, 0] and _shown(view)[0] == ("user_file", 0)
        assert _to_end(scrolls[opened:]) == [], "no edge load scrolled the view to the end"

        # at the top nothing more loads; at the bottom a scroll down loads the newer cards
        loads = transcript.edge_loads
        transcript.on_scroll(_scroll_event(0.0, -30.0))
        assert transcript.edge_loads == loads and view.window == (0, LIMIT)
        transcript.on_scroll(_scroll_event(8800.0, 40.0))
        await _TW._quiet(calls)
        assert view.window == (PAGE, LIMIT + PAGE)
        _check_render(view)

    _run(scenario, desktop_store_cls, isolated_env)


def test_client_tap_and_scroll_through_the_flet_session(desktop_store_cls, isolated_env):
    """The phone's own events: the client's "click" on the loader row and its "scroll" notification on the
    transcript, by control id through Flet 1.0.3's event decoding (``OnScrollEvent``), on the mounted
    controls; the client receives the patched loader text and the ``scroll_to`` of the old top card."""
    from flet.messaging.protocol import MessageAction

    from glossarion_mobile.ui.chat.transcript import EARLIER_TEMPLATE, CardSlot

    async def scenario(chat):
        view, adapter, calls = chat.view, chat.adapter, chat.calls
        conn, session = _TC._TB._fake_session("android")
        page = session.page
        page.views[0].controls.append(view.transcript)
        page.update()
        await asyncio.sleep(0.05)
        adapter.append_messages(CID, _TW._text_turns(30, 200))
        view.load_chat(CID)
        await _TW._quiet(calls)
        transcript = view.transcript
        assert view.window == (40, 60) and transcript.earlier_row._i in session.index

        button = transcript.earlier_row._i
        mark = len(conn.messages)
        await session.dispatch_event(button, "click", None)  # the tap
        await _TW._quiet(calls)
        assert view.window == (34, 54) and transcript.earlier_row._i == button
        patches = [m for m in conn.messages[mark:] if m.action == MessageAction.PATCH_CONTROL]
        assert any(EARLIER_TEMPLATE.format(n=34) in str(m.body) for m in patches), "the client got the new count"
        _check_render(view)

        top = next(c for c in transcript.messages if isinstance(c, CardSlot)).slot_key
        await asyncio.sleep(0.4)  # the tap's own hold on edge loads (the client lays the cards out)
        mark = len(conn.messages)
        await session.dispatch_event(transcript._i, "scroll", {
            "event_type": "update", "pixels": 120.0, "min_scroll_extent": 0.0, "max_scroll_extent": 6000.0,
            "viewport_dimension": 800.0, "scroll_delta": -30.0})
        await _TW._quiet(calls)
        assert view.window == (28, 48) and transcript.edge_loads == 1
        invoked = [m.body for m in conn.messages[mark:] if m.action == MessageAction.INVOKE_METHOD]
        restores = [b for b in invoked if b.name == "scroll_to"
                    and getattr((b.args or {}).get("scroll_key"), "value", (b.args or {}).get("scroll_key")) == top]
        assert restores, "the client is told to keep the old top card in view"
        assert any(EARLIER_TEMPLATE.format(n=28) in str(m.body) for m in conn.messages[mark:]
                   if m.action == MessageAction.PATCH_CONTROL)
        _check_render(view)

    _run(scenario, desktop_store_cls, isolated_env)


# ==========================================================================
# 3. Jump-to / search reach the earliest cards and stay there
# ==========================================================================


def test_jump_to_sheet_and_search_reach_the_first_cards(desktop_store_cls, isolated_env, no_highlight):
    from glossarion_mobile.ui.chat.cards import JobCard
    from glossarion_mobile.ui.chat.messages import UserBubble, UserFileCard

    async def scenario(chat):
        view, adapter, calls = chat.view, chat.adapter, chat.calls
        scrolls = _recorder(view)
        # the mixed chat: 10 pasted-chapter turns (messages 0-19), then 3 books of 40 x 12k chapters
        adapter.append_messages(CID, _TW._text_turns(10, 15_000, source=5_000) + _books(isolated_env, 3, 40, 12_000))
        messages = adapter.messages(CID)
        view.load_chat(CID)
        await _TW._quiet(calls)
        assert view.window[0] > 0

        # jump_to(0): the first card is rendered, scrolled to, and stays through every re-render
        await view.jump_to(0)
        await _TW._quiet(calls)
        assert view.window[0] == 0 and _shown(view)[0] == ("user", 0)
        assert isinstance(view.transcript.slot_for(view._scroll_key_for(0)).card, UserBubble)
        assert scrolls[-1]["scroll_key"].value == view._scroll_key_for(0).value
        _check_render(view)
        await _rerender_everything(view)

        # the Jump-to sheet (Inputs tab, first row), from the newest cards
        view.show_newest()
        await _TW._quiet(calls)  # its scroll to the end lands before the jump
        assert view.window[0] > 0
        sheet = view.open_jump_to()
        assert sheet.inputs[0].index == 0
        sheet.jump(sheet.inputs[0].index)
        await _TW._quiet(calls)
        assert view.window[0] == 0 and view.transcript.slot_for(view._scroll_key_for(0)) is not None
        await _rerender_everything(view)

        # a request inside the first book's JobCard: its card is rendered, the row listed and its list open
        first_book = next(i for i, m in enumerate(messages) if m[0] == "user_file")
        chapter_3 = first_book + 4  # file, glossary, chapter 1, chapter 2, chapter 3
        assert str(messages[chapter_3][5]).startswith("Chapter 3 (chunk")
        view.show_newest()
        await _TW._quiet(calls)  # its scroll to the end lands before the jump
        await view.jump_to(chapter_3)
        await _TW._quiet(calls)
        slot = view.transcript.slot_for(view._scroll_key_for(chapter_3))
        assert slot is not None and isinstance(slot.card, JobCard) and slot.slot_key == f"job-{first_book}"
        assert slot.card.title_text.value == "book01.epub" and slot.card.requests_tile.expanded
        assert any(label.startswith("Chapter 3 (chunk") for label in _row_labels(slot.card))
        assert scrolls[-1]["scroll_key"].value == f"job-{first_book}"
        await _rerender_everything(view)
        # the book's own file card, too
        await view.jump_to(first_book)
        assert isinstance(view.transcript.slot_for(view._scroll_key_for(first_book)).card, UserFileCard)

        # search: the only hit is the first message
        view.show_newest()
        await _TW._quiet(calls)  # its scroll to the end lands before the jump
        view.open_search()
        view._on_search_query("[src0]")
        await _TW._quiet(calls)
        assert view.search_hits == [0]
        assert view.window[0] == 0 and view.transcript.slot_for(view._scroll_key_for(0)) is not None
        await _rerender_everything(view)
        view.close_search()

    _run(scenario, desktop_store_cls, isolated_env)


# ==========================================================================
# 4. Appends while scrolled up keep the window; at the tail they follow
# ==========================================================================


@pytest.fixture
def fast_stream(monkeypatch):
    """The stream loop repaints every 20 ms (the desktop cadence is 280-900 ms): the test waits less."""
    from glossarion_mobile.ui.chat.stream_bridge import RunStream

    monkeypatch.setattr(RunStream, "render_interval_ms", lambda self: 20)


STREAM_SCROLL = {"offset": -1, "duration": 0}  # the stream loop's ``scroll_to_end(0)``


def _watch_scroll_to_end(view) -> list:
    """Record every ``Transcript.scroll_to_end`` as it starts (its 0.15 s settle begins); it still runs."""
    starts: list = []
    original = view.transcript.scroll_to_end

    async def scroll_to_end(*args, **kwargs):
        starts.append((args, kwargs))
        await original(*args, **kwargs)

    view.transcript.scroll_to_end = scroll_to_end
    return starts


async def _until(predicate, what: str, timeout: float = 5.0) -> None:
    loop = asyncio.get_running_loop()
    end = loop.time() + timeout
    while not predicate():
        assert loop.time() < end, f"timed out waiting for {what}"
        await asyncio.sleep(0.01)


async def _stream_settled(chat, scrolls: list, since: int) -> None:
    """The stream loop repainted what was fed after ``scrolls[since]`` (its own scroll to the end landed)
    and nothing is moving any more: the chat is idle, as when a user starts reading."""
    await _until(lambda: STREAM_SCROLL in scrolls[since:], "the stream loop's repaint")
    count = -1
    while count != len(scrolls):
        count = len(scrolls)
        await asyncio.sleep(0.3)
    await _TW._quiet(chat.calls)


async def _send_text(chat, text: str, scrolls: list):
    """The composer's Send (``on_send_action``) of a text turn; the job streams one request card."""
    from glossarion_mobile.ui.chat.send_state import SendAction

    view, runs, jobs, calls = chat.view, chat.runs, chat.jobs, chat.calls
    view.composer.set_text(text)
    view.on_send_action(SendAction.SEND)
    await _TW._quiet(calls)
    run = runs.live_run(CID)
    assert run is not None
    since = len(scrolls)
    jobs.publish("RUNNING", last_line="📤 sending api call now")
    for line in _TC.CARD_LINES:
        jobs.line(line)
    await _stream_settled(chat, scrolls, since)
    return run


async def _finish_text(chat, run, text: str) -> None:
    """The job wrote its output and ends: ``ChatStore.finish_run`` appends the answer."""
    Path(run.run.expected_output).parent.mkdir(parents=True, exist_ok=True)
    Path(run.run.expected_output).write_text(text, encoding="utf-8")
    chat.jobs.publish("DONE")
    _TC._finish_all(chat.runs)
    await asyncio.sleep(0.05)  # the finish thread's notification runs on the loop
    await _TW._quiet(chat.calls)
    assert not run.live


def test_a_live_text_run_never_yanks_a_window_the_user_slid_up(desktop_store_cls, isolated_env, fast_stream):
    from glossarion_mobile.ui.chat.messages import AssistantMessage

    async def scenario(chat):
        view, adapter, jobs, calls = chat.view, chat.adapter, chat.jobs, chat.calls
        _TW._post_to_loop(view)
        scrolls = _recorder(view)
        adapter.append_messages(CID, _TW._text_turns(30, 200))
        view.load_chat(CID)
        await _TW._quiet(calls)

        # a send while reading the tail: the window shows the newest cards (the question at the end)
        run = await _send_text(chat, "a new question", scrolls)
        assert len(view.items) == 61 and view.window == (41, 61) and view.transcript.hidden_after == 0
        # the user taps "↑ earlier" twice while it streams
        mark = len(scrolls)
        view.transcript.earlier_row.on_click(None)
        view.transcript.earlier_row.on_click(None)
        await _TW._quiet(calls)
        slid = view.window
        assert slid == (29, 49) and not view.transcript.follow_tail
        for k in range(5):
            jobs.line(f"more streamed text {k}", "Thread-2 (api_call)")
            await asyncio.sleep(0.1)
        await _until(lambda: "more streamed text 4" in str(run.stream.segments()), "the streamed text")
        await asyncio.sleep(0.3)
        assert view.window == slid and _to_end(scrolls[mark:]) == [], "the stream never scrolls to the end"
        assert view.new_fab.visible, "↓ shows instead"
        # the run ends: its answer is appended after the window, which stays where the user reads
        await _finish_text(chat, run, "Hello there, translated")
        assert len(view.items) == 62 and view.window == slid and view.transcript.hidden_after == 13
        assert _to_end(scrolls[mark:]) == []
        _check_render(view)
        # ↓ : the newest cards with the answer
        await view._jump_to_end()
        assert view.window[1] == 62 and view.transcript.hidden_after == 0 and view.transcript.follow_tail
        assert _shown(view)[-1] == ("assistant", 61)
        assert isinstance(view.transcript.cards[-1], AssistantMessage)

        # at the tail: the next answer is followed (window moves to the new tail, the view scrolls to it)
        run = await _send_text(chat, "another question", scrolls)
        mark = len(scrolls)
        await _finish_text(chat, run, "Another translation")
        assert len(view.items) == 64 and view.window[1] == 64 and view.transcript.hidden_after == 0
        assert _shown(view)[-1] == ("assistant", 63) and _to_end(scrolls[mark:])
        _check_render(view)

    _run(scenario, desktop_store_cls, isolated_env)


def test_a_tap_while_a_scroll_to_the_end_is_settling_keeps_the_earlier_cards(desktop_store_cls, isolated_env,
                                                                            fast_stream):
    """The skeptic's device-side cause (live runs), timing included. Every scroll to the end waits
    ``settle`` (0.15 s, the client lays the update out first) before ``scroll_to(offset=-1)``; while a run
    streams at the tail one is pending after every repaint tick (desktop cadence 280-900 ms) and after
    every job snapshot (``ChatRuns._emit`` -> ``render_transcript(follow=True)``). A tap on "↑ earlier"
    that lands inside that wait must win: the view stays on the earlier cards the tap just loaded.

    On the phone this is the owner's case: a transcript that fits the screen sends no scroll events, so
    ``follow_tail`` stays True and the loader row's tap is the only way up; the tap prepends a page of
    cards and the scroll to the end that was already settling then lands on the bottom of the now longer
    list, away from the cards just loaded (a 150 ms window after every repaint / snapshot while a run
    streams). Here the harness has no client, so ``follow_tail`` stays True the same way."""

    async def scenario(chat):
        view, adapter, jobs, calls = chat.view, chat.adapter, chat.jobs, chat.calls
        _TW._post_to_loop(view)
        scrolls = _recorder(view)
        starts = _watch_scroll_to_end(view)
        adapter.append_messages(CID, _TW._text_turns(30, 200))
        view.load_chat(CID)
        await _TW._quiet(calls)
        await _send_text(chat, "a new question", scrolls)
        assert view.transcript.follow_tail and view.transcript.hidden_after == 0 and view.window == (41, 61)

        # 1. the stream loop: its repaint of new text is waiting to scroll to the end when the user taps
        count = len(starts)
        jobs.line("more streamed text", "Thread-2 (api_call)")
        await _until(lambda: len(starts) > count, "the stream loop's scroll to the end")
        mark = len(scrolls)
        view.transcript.earlier_row.on_click(None)  # the tap, inside the 0.15 s settle
        await asyncio.sleep(0.3)
        assert view.window == (35, 55) and not view.transcript.follow_tail
        stream_yanks = _to_end(scrolls[mark:])

        # 2. a job snapshot while following: render_transcript(follow=True) spawned a scroll to the end
        await view._jump_to_end()
        await asyncio.sleep(0.3)
        assert view.transcript.follow_tail and view.window == (41, 61)
        count = len(starts)
        jobs.publish("RUNNING", progress={"total": 1, "completed": 0}, in_flight=1, last_line="📤 request 1")
        await _until(lambda: len(starts) > count, "the snapshot's follow scroll")
        mark = len(scrolls)
        view.transcript.earlier_row.on_click(None)  # the tap, inside the 0.15 s settle
        await asyncio.sleep(0.3)
        assert view.window == (35, 55) and not view.transcript.follow_tail
        render_yanks = _to_end(scrolls[mark:])

        assert (stream_yanks, render_yanks) == ([], []), (
            "scrolls to the end after the tap that loaded earlier cards - "
            f"from a stream repaint: {stream_yanks}; from a job snapshot's follow render: {render_yanks}")

    _run(scenario, desktop_store_cls, isolated_env)


def test_a_jump_during_a_live_run_is_not_undone_by_the_next_repaint(desktop_store_cls, isolated_env, fast_stream,
                                                                    no_highlight):
    """Jump-to while a run streams at the end of the chat: the jumped-to card (17 cards above the end,
    inside the newest window, so no re-render) is where the user reads now. The next repaint of the
    streaming card must not scroll the view back to the end - ``slide`` stops following the tail for this
    reason; ``jump_to`` must as well.

    The client is modelled: the jump's own scroll reports its first position one ``scroll_interval``
    (100 ms) later, far from the end (``follow_tail`` False from then on). A repaint before that - or one
    whose scroll to the end is settling when the jump starts - must not undo the jump. The stream here
    repaints every 20 ms, so it always comes first; on the phone (280-900 ms) it does now and then."""

    async def scenario(chat):
        view, adapter, jobs, calls = chat.view, chat.adapter, chat.jobs, chat.calls
        _TW._post_to_loop(view)
        scrolls = _recorder(view)
        adapter.append_messages(CID, _TW._text_turns(30, 200))
        view.load_chat(CID)
        await _TW._quiet(calls)
        run = await _send_text(chat, "a new question", scrolls)
        assert view.window == (41, 61)
        mark = len(scrolls)
        await view.jump_to(44)
        await _TW._quiet(calls)
        assert view.window == (41, 61) and scrolls[mark]["scroll_key"].value == view._scroll_key_for(44).value

        async def client_reports_the_jump():
            await asyncio.sleep(view.transcript.scroll_interval / 1000.0)
            view.transcript.on_scroll(_scroll_event(2000.0, -40.0, maximum=6000.0))

        client = asyncio.ensure_future(client_reports_the_jump())
        jobs.line("more streamed text", "Thread-2 (api_call)")
        await _until(lambda: "more streamed text" in str(run.stream.segments()), "the streamed text")
        await asyncio.sleep(0.3)
        await client
        assert not view.transcript.follow_tail
        assert _to_end(scrolls[mark:]) == [], (
            f"the next stream repaint scrolled the jump away to the end: {scrolls[mark:]}")

    _run(scenario, desktop_store_cls, isolated_env)


def test_a_window_slid_up_but_still_ending_at_the_tail_follows_like_the_desktop(desktop_store_cls, isolated_env):
    """The desktop rule (``_update_history_window_after_append``): a window that still ends at the old tail
    follows new cards; one that does not is kept."""

    async def scenario(chat):
        view, adapter, calls = chat.view, chat.adapter, chat.calls
        _recorder(view)
        adapter.append_messages(CID, _TW._text_turns(12, 15_000, source=5_000))
        view.load_chat(CID)
        await _TW._quiet(calls)
        view.transcript.earlier_row.on_click(None)
        await _TW._quiet(calls)
        assert view.window == (6, 24) and view.transcript.hidden_after == 0
        adapter.append_messages(CID, [("user", "a later question")])
        view._on_run_changed(CID)
        assert view.window[1] == 25 and _shown(view)[-1] == ("user", 24)
        _check_render(view)

    _run(scenario, desktop_store_cls, isolated_env)


def test_a_live_book_job_never_yanks_a_window_the_user_slid_up(desktop_store_cls, isolated_env, fast_stream):
    """A book translating in a long chat (the owner's flow): the user reads earlier cards while the job
    streams, reaches its glossary gate (the gate's request cards are frozen into the chat) and goes on;
    the window stays, ↓ brings back the running card - the book's own, live, with every request."""
    from glossarion_mobile.ui.chat.cards import JobCard

    async def scenario(chat):
        view, adapter, runs, jobs, calls = chat.view, chat.adapter, chat.runs, chat.jobs, chat.calls
        _TW._post_to_loop(view)
        scrolls = _recorder(view)
        adapter.append_messages(CID, _TW._text_turns(12, 200))
        view.load_chat(CID)
        await _TW._quiet(calls)
        book = isolated_env / "novel.epub"
        book.write_bytes(b"PK\x03\x04fake epub")
        record = {"path": str(book), "name": "novel.epub", "extension": ".epub", "size": 15}
        run = await runs.send(CID, text="", attachment=record, settings=view.settings(CID), output_mode="text")
        view._on_run_changed(CID)
        since = len(scrolls)
        jobs.publish("RUNNING", last_line="📑 Running glossary extraction before translation...")
        for k in range(22):
            thread, lines = _TW._glossary_lines(k, 1_500)
            for line in lines:
                jobs.line(line, thread)
        await _stream_settled(chat, scrolls, since)
        assert len(view.items) == 26 and view.window == (6, 26)
        live = view.live_job_card
        assert live is not None and live.title_text.value == "novel.epub" and live in view.transcript.cards
        assert live.requests_tile.title == "Requests (22)"

        # the user reads the earliest cards
        hidden = await _tap_earlier_until_top(view, calls)
        assert hidden == [6, 0] and view.window == (0, 20) and view.live_job_card is None
        mark = len(scrolls)
        # the gate: the job asks, ChatRuns.commit_gate freezes the 22 glossary cards into the chat
        jobs.publish("RUNNING", question={"id": "q1", "kind": "direct_text_glossary_approval",
                                          "data": {"path": str(isolated_env / "glossary.csv")}})
        await _TW._quiet(calls)
        assert run.awaiting_glossary and len(adapter.messages(CID)) == 24 + 1 + 22
        assert view.window == (0, 20) and len(view.items) == 26
        _check_render(view)
        # ✓ Yes (from the approval card in the tail); the translation streams
        view._answer_glossary(True)
        for k in range(1, 4):
            thread, lines = _TW._chapter_lines(k)
            for line in lines:
                jobs.line(line, thread)
        jobs.publish("RUNNING", progress={"total": 12, "completed": 3}, in_flight=1, last_line="📤 chapter 3")
        await _until(lambda: "chapter 3 translated" in str(run.stream.segments()), "the streamed chapters")
        await asyncio.sleep(0.3)
        assert view.window == (0, 20) and _to_end(scrolls[mark:]) == [] and view.new_fab.visible
        _check_render(view)

        # ↓ : the running card of the book, with the gate's cards and the live ones
        await view._jump_to_end()
        await asyncio.sleep(0.1)
        card = view.live_job_card
        assert view.window[1] == 26 and card is not None and card in view.transcript.cards
        assert [c for c in view.transcript.cards if isinstance(c, JobCard)] == [card]
        assert card.title_text.value == "novel.epub" and card.phase.name == "running"
        assert card.requests_tile.title == "Requests (25)"

        jobs.publish("CANCELLED", progress={"total": 12, "completed": 3})
        _TC._finish_all(runs)
        await asyncio.sleep(0.05)
        await _TW._quiet(calls)
        assert not run.live

    _run(scenario, desktop_store_cls, isolated_env, jobs=_TW._started_jobs())


# ==========================================================================
# 5. The owner's flow end to end: a real book job through the real pipeline
# ==========================================================================

E2E_DEPENDENCIES = ("ebooklib", "openai", "httpx", "lxml", "bs4", "tiktoken")
SELFTEST_EPUB = APP_DIR / "assets" / "selftest" / "selftest_ko_12ch.epub"  # tools/prepare_assets.py (CI: prepare)
TIKTOKEN_ASSETS = APP_DIR / "assets" / "tiktoken"  # the bundled BPE files (no download in the sandbox)
LIVE_CHILD = "--issue13-live-child"
#: chapters answered before the fake model stops answering (the live checks run while it is held)
LIVE_HELD_AFTER = 3
#: the fake model's marker: every Hangul run becomes ~900 characters, so chapters are 11-17k characters
#: (the diagnosis's live repro: the 120k budget cut such a book turn into one orphan 'Attachment' card)
BIG_MARKER = ("[FAKE-EN " + "lorem ipsum " * 200)[:900] + "]"


def _missing_modules(*modules: str) -> list:
    return [m for m in modules if importlib.util.find_spec(m) is None]


def _child_env(tmp_path: Path) -> dict:
    """The child's environment: every data, output and home folder under ``tmp_path``, the mobile runtime
    (no worker processes), the bundled tiktoken files, no HTTP log; the parent's user site stays found."""
    import site

    env = {k: v for k, v in os.environ.items() if not k.startswith(("FLET_", "GLOSSARION_"))}
    folders = {}
    for name in ("home", "appdata", "Library", "data", "Output", "work", "tiktoken"):
        folders[name] = tmp_path / name
        folders[name].mkdir(parents=True, exist_ok=True)
    for blob in TIKTOKEN_ASSETS.iterdir():
        if blob.is_file():
            shutil.copy2(blob, folders["tiktoken"] / blob.name)
    if "PYTHONUSERBASE" not in env:
        try:
            env["PYTHONUSERBASE"] = site.getuserbase()  # HOME / APPDATA move: the user site must not
        except Exception:
            pass
    env.update(
        HOME=str(folders["home"]), USERPROFILE=str(folders["home"]), APPDATA=str(folders["appdata"]),
        GLOSSARION_LIBRARY_DIR=str(folders["Library"]), GLOSSARION_DATA_DIR=str(folders["data"]),
        OUTPUT_DIRECTORY=str(folders["Output"]), GLOSSARION_HTTP_LOG="0", GLOSSARION_MOBILE="1",
        GLOSSARION_NO_PROCESSES="1", TIKTOKEN_CACHE_DIR=str(folders["tiktoken"]), TMPDIR=str(folders["work"]),
        TEMP=str(folders["work"]), TMP=str(folders["work"]), PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1",
    )
    return env


@pytest.mark.skipif(not SELFTEST_EPUB.is_file() or not TIKTOKEN_ASSETS.is_dir(),
                    reason="run tools/prepare_assets.py first")
@pytest.mark.skipif(bool(_missing_modules(*E2E_DEPENDENCIES)),
                    reason=f"backend dependencies missing: {_missing_modules(*E2E_DEPENDENCIES)}")
def test_the_owners_book_chat_through_the_real_pipeline(tmp_path):
    """What the owner saw on the phone, through the real job: a book translating in a chat with earlier
    messages. Before the fix the finished book was one orphan JobCard['Attachment' | 'Done' | Requests (8)]
    under a dead "↑ Scroll for earlier messages (8 hidden)" (diagnosis live_u8b)."""
    result_file = tmp_path / "issue13_live.json"
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), LIVE_CHILD, str(result_file), str(tmp_path)],
        cwd=str(tmp_path / "work") if (tmp_path / "work").is_dir() else str(tmp_path),
        env=_child_env(tmp_path), capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=900,
    )
    tail = (proc.stdout or "")[-3000:] + "\n" + (proc.stderr or "")[-6000:]
    assert result_file.is_file(), f"no result (exit {proc.returncode}):\n{tail}"
    r = json.loads(result_file.read_text(encoding="utf-8"))
    assert r.get("error") is None, f"{r.get('error')}\n{tail}"
    assert proc.returncode == 0, tail

    # the chat opens on its newest cards; the book's send keeps following them
    assert r["opened"] == {"window": [40, 60], "earlier_row": "↑ Scroll for earlier messages (40 hidden)"}
    live = r["live"]
    assert live["window"] == [42, 62] and live["items_tail"] == [["user_file", 60], ["job", 60]]
    assert live["card"]["title"] == "selftest_ko_12ch.epub" and live["card"]["live"]
    assert live["card"]["requests"] >= LIVE_HELD_AFTER
    # the tap while it runs: earlier cards, N drops by one page, nothing follows the run any more
    tap = r["live_tap"]
    assert tap["window"] == [36, 56] and tap["earlier_row"] == "↑ Scroll for earlier messages (36 hidden)"
    assert tap["first_card"] == ["user", 36] and not tap["follow_tail"] and tap["live_card_shown"] is False
    assert tap["scrolls_to_end_after"] == [], "a repaint of the running book scrolled the view to the end"
    # a scroll to the top while it runs: one more page, the card that was at the top put back in view
    edge = r["live_edge"]
    assert edge["window"] == [30, 50] and edge["edge_loads"] == 1 and edge["restored"] == edge["old_top"]
    # the run finishes while the user reads: the window stays, nothing scrolls to the end, ↓ shows
    done = r["finished"]
    assert done["run_state"] == "done" and done["window"] == [30, 50] and done["scrolls_to_end"] == []
    assert done["new_fab"] is True
    # ↓ : the book is its file card + its own JobCard with every request (never 'Attachment')
    newest = r["newest"]
    assert newest["window"] == [42, 62] and newest["cards_tail"][-2:] == ["UserFileCard", "JobCard"]
    card = newest["card"]
    assert card["title"] == "selftest_ko_12ch.epub" and card["state"] == "Done · 12/12 chapters"
    assert not card["live"] and card["requests_title"] == f"Requests ({newest['turn_requests']})"
    assert newest["turn_requests"] >= 13 and newest["chapters_in_store"] == 12
    assert newest["big_chapters"], "the fake chapters are 11-17k characters (the budget case)"
    assert r["job_titles_seen"] == ["selftest_ko_12ch.epub"], "an orphan 'Attachment' card was rendered"
    # the taps reach the first card, N dropping one page each time
    assert r["taps"] == [42, 36, 30, 24, 18, 12, 6, 0] and r["first_card"] == ["user", 0]
    # jump-to: the first card, and chapter 3's request row inside the book's JobCard
    assert r["jump_first"] == {"window_start": 0, "slot": "UserBubble", "kept": True}
    jump = r["jump_chapter"]
    assert jump["slot"] == "job-60" and jump["expanded"] and jump["row"].startswith("Chapter 3 (chunk")
    assert r["writes_outside"] == [], "the run wrote outside its sandbox"


async def _live_child_scenario(tmp_root: Path) -> dict:
    """The child process of ``test_the_owners_book_chat_through_the_real_pipeline`` (see its asserts)."""
    from glossarion_mobile.diagnostics.e2e import E2ESession
    from glossarion_mobile.services.dispatcher import UiDispatcher
    from glossarion_mobile.state.app_state import AppState
    from glossarion_mobile.state.chat_store_adapter import ChatStoreAdapter, ChatStoreBinding
    from glossarion_mobile.ui.chat.cards import JobCard
    from glossarion_mobile.ui.chat.chat_view import ChatView
    from glossarion_mobile.ui.chat.context import ChatEnv
    from glossarion_mobile.ui.chat.job_binding import JobsAdapter
    from glossarion_mobile.ui.chat.messages import AssistantMessage, UserBubble, UserFileCard
    from glossarion_mobile.ui.chat.run_controller import ChatRuns
    from glossarion_mobile.ui.chat.run_request import attachment_record
    from glossarion_mobile.ui.chat.transcript import EDGE_HOLD_SECONDS, CardSlot
    from glossarion_mobile.ui.settings.schema_access import SchemaAccess

    work = tmp_root / "work"
    out: dict = {"error": None}
    paths = types.SimpleNamespace(assets_dir=str(APP_DIR / "assets"), temp=str(work),
                                  writable_dirs=lambda: {"tmp": str(tmp_root)}, backend_dir=str(SRC_DIR))
    session = E2ESession(paths, root=str(work / "e2e"), keep=False, isolated=True, log=lambda line: None)
    session.setup()
    session.configure("balanced")
    server = session.server
    server.marker = BIG_MARKER
    arrived: list = []

    def hold_after_three(record) -> None:  # the model stops answering after LIVE_HELD_AFTER chapters
        if record.kind == "translation" and len(record.chapters) == 1:
            arrived.append(record.chapters[0])
            if len(arrived) == LIVE_HELD_AFTER + 1:
                server.hold()

    server.on_request.append(hold_after_three)
    history = str(session.root / "direct_text_chats.json")
    chats = ChatStoreAdapter(ChatStoreBinding(history_path=history, output_root=str(session.root / "Output")),
                             history_path=history, save_delay=0.05)
    assert chats.load(), chats.load_error
    jobs = JobsAdapter(session.service)
    runs = ChatRuns(chats, jobs, temp_dir=str(session.root / "runs"), model_name=lambda: session.store.get("model"))
    runs.attach()
    loop = asyncio.get_running_loop()
    tasks: list = []

    def spawn(coro):
        task = asyncio.ensure_future(coro)
        tasks.append(task)
        return task

    page = types.SimpleNamespace(width=412, height=900, show_dialog=lambda *a: None, update=lambda *a: None)
    env = ChatEnv(page=page, store=session.store, schema=SchemaAccess(), dispatcher=UiDispatcher().bind(loop),
                  chats=chats, runs=runs, jobs=jobs)
    env.spawn = spawn
    cid = chats.new_chat()
    chats.append_messages(cid, _TW._text_turns(30, 200))  # 60 earlier messages
    state = AppState()
    state.current_chat.set(cid)
    view = ChatView(page, state=state, navigate=lambda *a, **k: None, notify=lambda *a, **k: None, env=env)
    transcript = view.transcript
    scrolls: list = []

    async def scroll_to(**kwargs):
        scrolls.append(kwargs)

    transcript.scroll_to = scroll_to
    titles: list = []

    def watch() -> None:
        for card in transcript.cards:
            if isinstance(card, JobCard) and card.title_text.value not in titles:
                titles.append(card.title_text.value)

    def first_card() -> list:
        for control in transcript.messages:
            if isinstance(control, CardSlot):
                card = control.card
                kind = {UserBubble: "user", UserFileCard: "user_file", AssistantMessage: "assistant"}.get(type(card))
                return [kind or type(card).__name__, getattr(card, "index", None)]
        return []

    def to_end(since: int) -> list:
        return [s for s in scrolls[since:] if s.get("offset") == -1]

    async def until(predicate, what: str, timeout: float = 240.0) -> None:
        end = time.monotonic() + timeout
        while not predicate():
            if view.approval_card is not None and runs.live_run(cid) is not None and not view.approval_card.answered:
                view.approval_card.yes_button.on_click(None)  # ✓ Yes on the glossary card, if asked
            watch()
            if time.monotonic() > end:
                raise AssertionError(f"timed out waiting for {what}")
            await asyncio.sleep(0.05)
        watch()

    async def quiet(seconds: float = 0.6, timeout: float = 30.0) -> None:
        """No scroll command for ``seconds``: nothing is settling when the user acts."""
        end = time.monotonic() + timeout
        count, since = len(scrolls), time.monotonic()
        while time.monotonic() - since < seconds and time.monotonic() < end:
            await asyncio.sleep(0.05)
            if len(scrolls) != count:
                count, since = len(scrolls), time.monotonic()

    try:
        out["opened"] = {"window": list(view.window), "earlier_row": transcript.earlier_row.content}
        record = attachment_record(session.import_epub(session.epub.name))
        run = await runs.send(cid, text="", attachment=record, settings=view.settings(cid), output_mode="text")
        # the model holds the 4th chapter: the book runs, its card lists the answered requests
        await until(lambda: server.holding and server.parked >= 1, "the held chapter request")
        await until(lambda: view.live_job_card is not None
                    and len(view.live_job_card.requests_column.controls) >= LIVE_HELD_AFTER, "the live card's rows")
        await quiet()
        card = view.live_job_card
        out["live"] = {"window": list(view.window), "items_tail": [[i.kind, i.index] for i in view.items[-2:]],
                       "card": {"title": card.title_text.value, "live": bool(card.phase.live),
                                "requests": len(card.requests_column.controls)}}
        # the tap on "↑ Scroll for earlier messages" while the book runs
        transcript.earlier_row.on_click(None)
        await asyncio.sleep(0.3)  # a scroll already settling before the tap has landed
        mark = len(scrolls)
        for _ in range(3):  # the job goes on streaming status: repaints and snapshots
            view._on_run_changed(cid)
            await asyncio.sleep(0.4)
        out["live_tap"] = {"window": list(view.window), "earlier_row": transcript.earlier_row.content,
                           "first_card": first_card(), "follow_tail": transcript.follow_tail,
                           "live_card_shown": view.live_job_card is not None,
                           "scrolls_to_end_after": to_end(mark)}
        # a scroll to the top while it runs (the next gesture: after the tap's hold)
        await asyncio.sleep(EDGE_HOLD_SECONDS + 0.1)
        top = next(c for c in transcript.messages if isinstance(c, CardSlot)).slot_key
        loads = transcript.edge_loads
        transcript.on_scroll(types.SimpleNamespace(event_type="update", pixels=150.0, min_scroll_extent=0.0,
                                                   max_scroll_extent=9000.0, viewport_dimension=800.0,
                                                   scroll_delta=-35.0, overscroll=None))
        await asyncio.sleep(0.5)  # keep_in_view's settle + restore
        keyed = [s for s in scrolls[mark:] if s.get("scroll_key") is not None]
        out["live_edge"] = {"window": list(view.window), "edge_loads": transcript.edge_loads - loads,
                            "old_top": top, "restored": getattr(keyed[-1]["scroll_key"], "value", None) if keyed else None}
        # the model answers again; the book finishes while the user reads the earlier cards
        mark = len(scrolls)
        server.release()
        await until(lambda: not run.live, "the end of the run", timeout=600)
        for thread in list(runs.finish_threads):
            thread.join(120)
        await asyncio.sleep(1.0)
        out["finished"] = {"run_state": run.state, "window": list(view.window), "scrolls_to_end": to_end(mark),
                           "new_fab": bool(view.new_fab.visible)}
        # ↓ : the newest cards
        await view._jump_to_end()
        await asyncio.sleep(0.3)
        watch()
        messages = chats.messages(cid)
        turn = [m for m in messages[61:] if m[0] == "assistant"
                and str(m[5]) not in ("Extraction report", "Attachment actions")]
        chapters = [i for i, m in enumerate(messages) if m[0] == "assistant" and str(m[5]).startswith("Chapter ")]
        sizes = [len(chats.message_text(cid, i, "content")) for i in chapters]
        job = transcript.cards[-1]
        out["newest"] = {
            "window": list(view.window), "cards_tail": [type(c).__name__ for c in transcript.cards[-2:]],
            "card": {"title": job.title_text.value, "state": job.state_text.value, "live": bool(job.phase.live),
                     "requests_title": job.requests_tile.title} if isinstance(job, JobCard) else None,
            "turn_requests": len(turn), "chapters_in_store": len(chapters),
            "big_chapters": bool(sizes) and min(sizes) > 8_000,
        }
        # the taps reach the first card
        taps = [transcript.hidden_before]
        while transcript.hidden_before and len(taps) < 30:
            transcript.earlier_row.on_click(None)
            await asyncio.sleep(0.05)
            watch()
            taps.append(transcript.hidden_before)
        out["taps"], out["first_card"] = taps, first_card()
        # jump-to: the first card, from the newest cards (and it stays through re-renders)
        view.show_newest()
        await asyncio.sleep(0.3)
        await view.jump_to(0)
        await asyncio.sleep(0.1)
        slot = transcript.slot_for(view._scroll_key_for(0))
        window = view.window
        view.render_transcript(follow=True)
        view._on_run_changed(cid)
        out["jump_first"] = {"window_start": view.window[0], "slot": type(getattr(slot, "card", None)).__name__,
                             "kept": view.window == window and transcript.slot_for(view._scroll_key_for(0)) is not None}
        # jump-to a chapter's request: the book's JobCard, its row listed and its Requests list open
        chapter_3 = next(i for i in chapters if str(messages[i][5]).startswith("Chapter 3 (chunk"))
        view.show_newest()
        await asyncio.sleep(0.3)
        await view.jump_to(chapter_3)
        await asyncio.sleep(0.1)
        slot = transcript.slot_for(view._scroll_key_for(chapter_3))
        rows = []
        if slot is not None and isinstance(slot.card, JobCard):
            rows = [row.content.controls[0].controls[0].value for row in slot.card.requests_column.controls]
        out["jump_chapter"] = {"slot": getattr(slot, "slot_key", None),
                               "expanded": bool(slot is not None and getattr(slot.card, "requests_tile", None)
                                                and slot.card.requests_tile.expanded),
                               "row": next((r for r in rows if str(r).startswith("Chapter 3 (chunk")), "")}
        out["job_titles_seen"] = titles
    finally:
        try:
            server.release()
        except Exception:
            pass
        for task in tasks:
            task.cancel()
        out["writes_outside"] = [v.get("path") for v in (session.audit.violations if session.audit else [])][:10]
        runs.detach()
        chats.close()
        session.close()
    return out


def _live_child_main(argv: list) -> int:
    result_file, tmp_root = Path(argv[0]), Path(argv[1])
    try:
        result = asyncio.run(_live_child_scenario(tmp_root))
    except BaseException as exc:  # the parent shows it with the child's output
        import traceback

        result = {"error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"}
    result_file.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0 if result.get("error") is None else 1


if __name__ == "__main__" and LIVE_CHILD in sys.argv:
    sys.exit(_live_child_main(sys.argv[sys.argv.index(LIVE_CHILD) + 1:]))
