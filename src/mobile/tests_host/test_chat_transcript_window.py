"""Owner device report issues 12 + 13 (devfix4): the chat transcript window and the book turn's JobCard.

Run from src/mobile with the mobile venv:
    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_chat_transcript_window.py

* 12: "a chat attachment job shows Done right after the glossary step, and translation request cards do
  not appear in the chat until the Jobs page is opened". The glossary gate freezes its request cards into
  the chat (``ChatStore.commit_request_phase``); the desktop message window then started after the turn's
  user_file, ``build_items`` gave the JobCard the index -1, and the running card was drawn as a static
  "Done" / "Attachment" card while every live repaint went to a card off screen.
* 13: "'↑ Scroll for earlier messages (N hidden)' does nothing" and "only one request card": the render
  re-tailed the window whenever it ended near the tail, undoing every slide and jump; a book turn's
  request messages are rows of one JobCard, so a message window never added a card.

Fix (mobile only, ``ui/chat/transcript_model`` + ``chat_view`` + ``transcript`` + ``cards``): the chat
windows over its rendered cards (the whole chat's ``build_items``; a JobCard always carries its turn), the
window moves with the desktop arithmetic (``_shift_history_window`` / ``_update_history_window_after_append``,
compared with the dialog source below), the loader tap and a scroll near an edge load a page of cards,
jumps stay where they land, and the running JobCard lists the turn's committed cards + its live ones with
"↑ Show … earlier requests" paging. Every test runs with HOME / USERPROFILE / APPDATA /
GLOSSARION_LIBRARY_DIR / GLOSSARION_DATA_DIR / OUTPUT_DIRECTORY in tmp_path (test_chat's ``isolated_env``).
"""

from __future__ import annotations

import asyncio
import importlib.util
import random
import sys
import threading
import types
from pathlib import Path

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = MOBILE_DIR / "app"
SRC_DIR = MOBILE_DIR.parent
for _entry in (str(APP_DIR), str(SRC_DIR)):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from glossarion_mobile.ui.chat.transcript_model import (  # noqa: E402
    ACTIONS_LABEL,
    REPORT_LABEL,
    ROW_PREVIEW_CHARS,
    TranscriptItem,
    build_items,
    item_position,
    item_sizes,
    message_size,
    shift_window,
    slide_window,
    tail_start,
    tail_window,
    window_after_append,
)


def _has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _backend_error() -> str:
    try:
        import direct_text_stream  # noqa: F401  (the chat's shared Direct Text code)
    except Exception as exc:  # pragma: no cover - a venv without the backend dependencies
        return f"the shared backend is not importable here ({exc}); use the project venv (uv sync)"
    return ""


_BACKEND_ERROR = _backend_error()
needs_backend = pytest.mark.skipif(bool(_BACKEND_ERROR), reason=_BACKEND_ERROR or "backend importable")
needs_flet = pytest.mark.skipif(not (_has("flet") and _has("msgpack")), reason="flet / msgpack not installed")
needs_desktop_source = pytest.mark.skipif(not (SRC_DIR / "translator_gui.py").is_file(),
                                          reason="src/translator_gui.py not present")


def _load(filename: str, alias: str):
    spec = importlib.util.spec_from_file_location(alias, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_TC = _load("test_chat.py", "_glossarion_tc_helpers_transcript_window") if not _BACKEND_ERROR else None
if _TC is not None:
    desktop_store_cls = _TC.desktop_store_cls  # noqa: F811  (pytest fixtures)
    isolated_env = _TC.isolated_env


# ==========================================================================
# Fixtures: chats as the store holds them
# ==========================================================================

PAD = "The knight walked through the silver forest. "  # 45 characters


def _body(chars: int, tag: str) -> str:
    return (f"[{tag}] " + PAD * (chars // len(PAD) + 1))[:chars]


def _assistant(content: str, label: str, folder: str = "") -> tuple:
    return ("assistant", content, "", "Token summary  ·  Thinking 0  ·  Text 900", folder, label,
            {"created_at": "2026-10-01T10:12:00+09:00"})


def _book_turn(path: str, chapters: int, chars: int, folder: str = "/w/Attachments/book") -> list:
    """A finished book turn: the file, one glossary card, ``chapters`` chapter cards, report, actions."""
    name = Path(path).name
    turn = [("user_file", name, path, 15, "", "user"),
            _assistant("type,raw_name,translated_name\n" + _body(3000, "glossary"),
                       "Glossary extraction · Request 1", folder)]
    for chapter in range(1, chapters + 1):
        label = f"Chapter {chapter} (chunk 1/1) · ch{chapter:03d}.xhtml · Request {chapter + 1}"
        turn.append(_assistant(_body(chars, f"ch{chapter}"), label, folder))
    turn.append(_assistant("## Extraction report\n- Chapter payloads: ready", REPORT_LABEL, folder))
    turn.append(_assistant("Attachment actions", ACTIONS_LABEL, folder))
    return turn


def _text_turns(turns: int, chars: int, source: int = 0) -> list:
    out: list = []
    for turn in range(turns):
        out.append(("user", _body(source, f"src{turn}") if source else f"turn {turn} source"))
        out.append(_assistant(_body(chars, f"t{turn}"), f"Request {turn + 1}"))
    return out


# ==========================================================================
# Pure: grouping, card sizes, the desktop window arithmetic
# ==========================================================================


def test_a_job_item_always_carries_its_turn():
    messages = [("user", "a"), _assistant("b", "Request 1"),
                ("user_file", "book.epub", "/x/book.epub", 1, "", "user"),
                _assistant("c", "Chapter 1 · Request 2"), _assistant("d", REPORT_LABEL),
                _assistant("e", ACTIONS_LABEL),
                ("user", "next")]
    whole = build_items(messages)
    job = whole[3]
    assert (job.kind, job.index, job.requests, job.report, job.actions, job.detached) == ("job", 2, [3], 4, 5, False)
    partial = build_items(messages, 4, 7)  # a range starting inside the turn: the job still knows its turn
    assert [(i.kind, i.index, i.detached) for i in partial] == [("job", 2, True), ("user", 6, False)]
    assert partial[0].key == "job-2" and partial[0].report == 4
    # the card that shows a message: the file card for the user_file, its JobCard for the turn's cards
    assert [item_position(whole, i) for i in range(7)] == [0, 1, 2, 3, 3, 3, 4]
    assert TranscriptItem("job", 2, requests=[3]).shows(3) and not TranscriptItem("job", 2).shows(2)


@pytest.mark.parametrize("cards, chars, limit", [(20, 1_500, 20), (3, 45_000, 20), (4, 600, 4), (2, 100_000, 20)])
def test_the_message_window_detached_the_turn_the_card_window_never_does(cards, chars, limit):
    """The issue-12 trigger: ``cards`` committed glossary cards push the turn's user_file out of the
    desktop message window (two cards never do: it always keeps 3 messages). Over the chat's cards the
    turn is one file card + one JobCard, both shown, and a sub-range still names the turn."""
    messages = [("user_file", "novel.epub", "/x/novel.epub", 15, "", "user")]
    messages += [_assistant(_body(chars, f"g{k}"), f"Glossary · Request {k + 1}") for k in range(cards)]
    start, _end = tail_window(messages, limit)
    assert (start > 0) == (cards > 2)
    detached = build_items(messages, start)
    assert [(i.kind, i.index) for i in detached] == ([("job", 0)] if start else [("user_file", 0), ("job", 0)])
    items = build_items(messages)
    assert [(i.kind, i.index) for i in items] == [("user_file", 0), ("job", 0)]
    sizes = item_sizes(items, messages, (), limit)
    assert sizes[1] == min(cards, limit) * min(chars, ROW_PREVIEW_CHARS)  # the rows a JobCard shows
    assert tail_start(sizes, limit) == 0


def test_card_sizes_follow_the_desktop_rule():
    messages = _text_turns(3, 9_000, source=500) + _book_turn("/x/b.epub", 30, 12_000)
    expanded = {1}
    items = build_items(messages)
    sizes = item_sizes(items, messages, expanded, 20)
    for item, size in zip(items, sizes):
        if item.kind != "job":
            assert size == message_size(messages[item.index], item.index, expanded)
    job = items[-1]
    report = message_size(messages[job.report], job.report, expanded)
    assert sizes[-1] == report + 20 * ROW_PREVIEW_CHARS  # the newest 20 of its 31 request rows
    # a text chat's cards are its messages: the card window is the desktop window
    text = _text_turns(30, 8_000, source=300)
    text_items = build_items(text)
    assert tail_start(item_sizes(text_items, text, (), 20), 20) == tail_window(text, 20)[0]


def test_shift_and_append_rules():
    # the owner's stuck cases: a budget-cut tail and a chat of limit + 1 messages now move
    assert shift_window((12, 24), 24, 20, -1) == (6, 24)
    assert shift_window((6, 24), 24, 20, -1) == (0, 20)
    assert shift_window((0, 20), 24, 20, -1) is None
    assert shift_window((1, 21), 21, 20, -1) == (0, 20)
    assert shift_window((0, 20), 21, 20, 1) == (1, 21)
    assert shift_window((80, 100), 100, 20, 1) is None
    assert slide_window((80, 100), 100, 20, -1) == (74, 94) and slide_window((74, 94), 100, 20, 1) == (80, 100)
    assert slide_window((0, 20), 100, 20, -1) == (0, 20)  # cannot move: unchanged
    # appends: a window at the old tail follows (None = re-tail), a slid one stays (clamped)
    assert window_after_append((10, 30), 30, 31) is None
    assert window_after_append((4, 24), 30, 31) == (4, 24)
    assert window_after_append((4, 24), 30, 10) == (4, 10)


@needs_backend
@needs_desktop_source
def test_window_arithmetic_is_the_dialogs():
    """``shift_window`` / ``window_after_append`` / ``tail_start`` against the Direct Text dialog's own
    ``_shift_history_window`` / ``_update_history_window_after_append`` / ``_reset_history_window``."""
    namespace: dict = {}
    exec("class D:\n    _DEFAULT_RENDERED_CARD_LIMIT = 20\n    _MIN_RENDERED_CARD_LIMIT = 4\n"
         "    _MAX_RENDERED_CARD_LIMIT = 200\n"
         + "\n".join(_TC._method_source(name) for name in (
             "_normalize_rendered_card_limit", "_assistant_message_char_count", "_assistant_storage_for",
             "_reset_history_window", "_shift_history_window", "_update_history_window_after_append")),
         namespace)
    dialog = namespace["D"]
    rng = random.Random(1213)

    def host(messages, limit, start, end):
        h = dialog()
        h._chat_messages = messages
        h._history_card_limit = limit
        h._history_character_budget = 120000
        h._expanded_processing_messages = set()
        h._history_visible_start, h._history_visible_end = start, end
        return h

    def random_messages(count):
        out = []
        for _ in range(count):
            if rng.random() < 0.4:
                out.append(("user", "u" * rng.randint(0, 9000)))
            else:
                out.append(("assistant", "", "", "", "", "", {"content_chars": rng.randint(0, 60000)}))
        return out

    for _ in range(2500):
        limit = rng.choice([4, 5, 6, 7, 20, 21, 40, 200])
        total = rng.randint(0, 70)
        start, end = rng.randint(0, total + 2), rng.randint(0, total + 2)
        h = host([("user", "x")] * total, limit, start, end)
        direction = rng.choice([-1, 1])
        moved = dialog._shift_history_window(h, direction)
        ours = shift_window((start, end), total, limit, direction)
        assert ours == ((h._history_visible_start, h._history_visible_end) if moved else None), \
            (start, end, total, limit, direction)

        previous, total = rng.randint(0, 70), rng.randint(0, 70)
        messages = random_messages(total)
        start, end = rng.randint(0, previous + 2), rng.randint(0, previous + 2)
        h = host(messages, limit, start, end)
        followed = dialog._update_history_window_after_append(h, previous)
        ours = window_after_append((start, end), previous, total)
        if followed:
            sizes = [message_size(m, i) for i, m in enumerate(messages)]
            assert ours is None and (tail_start(sizes, limit), total) == (h._history_visible_start,
                                                                         h._history_visible_end)
        else:
            assert ours == (h._history_visible_start, h._history_visible_end), (start, end, previous, total)


# ==========================================================================
# Flet: JobCard request paging, the Transcript's loader rows and edge loads
# ==========================================================================


def _segments(count: int, start: int = 0) -> list:
    return [{"label": f"Chapter {k} · Request {k}", "content": f"text {k}", "phase": "processing",
             "complete": True, "index": k} for k in range(start, start + count)]


@needs_flet
def test_job_card_pages_its_request_rows():
    from glossarion_mobile.ui.chat.cards import EARLIER_REQUESTS_TEMPLATE, JobCard
    from glossarion_mobile.ui.chat.job_binding import CardPhase

    card = JobCard(attachment={"name": "novel.epub", "extension": ".epub", "size": 15}, phase=CardPhase("done"),
                   row_page=20)
    segments = _segments(45)
    card.set_requests(segments)
    assert card.requests_tile.title == "Requests (45)" and not card.requests_tile.expanded
    assert len(card.requests_column.controls) == 20 and card.earlier_requests_button.visible
    assert card.earlier_requests_button.content == EARLIER_REQUESTS_TEMPLATE.format(n=20, hidden=25)
    first_rows = list(card.requests_column.controls)
    card.set_requests(segments)  # a repaint with nothing new keeps every row control
    assert all(a is b for a, b in zip(first_rows, card.requests_column.controls))
    card.show_earlier_requests()
    assert len(card.requests_column.controls) == 40
    assert card.earlier_requests_button.content == EARLIER_REQUESTS_TEMPLATE.format(n=5, hidden=5)
    assert card.reveal_request(2) and card.requests_tile.expanded
    assert len(card.requests_column.controls) == 45 and not card.earlier_requests_button.visible
    assert not card.reveal_request(999)
    live = JobCard(attachment={"name": "novel.epub"}, phase=CardPhase("running"), requests_expanded=True)
    assert live.requests_tile.expanded and live.saved_requests == []
    live._on_requests_toggle(types.SimpleNamespace(data="false"))  # the user closes it: it stays closed
    assert live.requests_tile.expanded is False


@needs_flet
def test_transcript_loads_at_its_edges_once_per_scroll():
    import flet as ft

    from glossarion_mobile.ui.chat.transcript import EDGE_LOAD_PX, EARLIER_TEMPLATE, Transcript

    loads: list = []
    transcript = Transcript(on_load_earlier=lambda auto=False: loads.append(("earlier", auto)) or True,
                            on_load_later=lambda auto=False: loads.append(("later", auto)) or True)
    transcript.set_messages([ft.Text("a"), ft.Text("b")], hidden_before=6, hidden_after=0)
    box = transcript.earlier_box
    assert transcript.controls[0] is box and box.controls[0].content == EARLIER_TEMPLATE.format(n=6)
    transcript.set_messages([ft.Text("c")], hidden_before=3, hidden_after=2)
    assert transcript.controls[0] is box and transcript.earlier_row.content == EARLIER_TEMPLATE.format(n=3)
    assert transcript.controls[-1] is transcript.later_box  # built once, never re-wrapped

    def scroll(pixels, delta=None, kind="update", overscroll=None, maximum=5000.0):
        return types.SimpleNamespace(event_type=kind, pixels=pixels, min_scroll_extent=0.0,
                                     max_scroll_extent=maximum, viewport_dimension=800.0,
                                     scroll_delta=delta, overscroll=overscroll)

    transcript.on_scroll(scroll(2000.0, -50.0))  # far from the top: nothing
    transcript.on_scroll(scroll(EDGE_LOAD_PX - 1, 30.0))  # near the top but going down: nothing
    assert loads == []
    transcript.on_scroll(scroll(EDGE_LOAD_PX - 1, -30.0))
    assert loads == [("earlier", True)]
    transcript.on_scroll(scroll(10.0, -30.0))  # the load is still restoring its viewport: no cascade
    assert loads == [("earlier", True)] and transcript.edge_loads == 1
    transcript.release_edge_load(hold=0)
    transcript.on_scroll(scroll(0.0, kind="overscroll", overscroll=-12.0))  # a pull past the top
    assert loads[-1] == ("earlier", True)
    transcript.release_edge_load(hold=0)
    transcript.on_scroll(scroll(4800.0, 40.0))  # towards newer cards near the end
    assert loads[-1] == ("later", True)
    transcript.release_edge_load(hold=0)
    transcript.hold_edge_loads(5.0)  # a jump's scroll: nothing loads meanwhile
    transcript.on_scroll(scroll(0.0, -40.0))
    assert len(loads) == 3
    transcript.earlier_row.on_click(None)  # the tap always works (a short transcript sends no scroll events)
    assert loads[-1] == ("earlier", False)


# ==========================================================================
# The chat view: a real ChatView on the desktop-format store, test_chat's FakeJobService
# ==========================================================================


def _cards(view, cls):
    return [card for card in view.transcript.cards if isinstance(card, cls)]


async def _quiet(calls, timeout: float = 5.0) -> None:
    """Wait for the view's spawned tasks, except the live run's stream loop (it runs while the run does)."""
    loop = asyncio.get_running_loop()
    end = loop.time() + timeout
    while loop.time() < end:
        pending = [t for t in calls.tasks if not t.done() and getattr(t.get_coro(), "__name__", "") != "_stream_loop"]
        if not pending:
            return
        await asyncio.wait(pending, timeout=0.1)


def _post_to_loop(view) -> None:
    """``ChatEnv.on_ui`` like the app's dispatcher: worker-thread notifications run on this loop."""
    loop = asyncio.get_running_loop()
    loop_thread = threading.get_ident()

    def on_ui(fn, *args):
        if threading.get_ident() == loop_thread:
            fn(*args)
        else:
            loop.call_soon_threadsafe(fn, *args)

    view.env.on_ui = on_ui


def _started_jobs():
    """test_chat's FakeJobService whose snapshots say the job ran (a Stop commits its cards)."""

    class Jobs(_TC.FakeJobService):
        def publish(self, state, **kwargs):
            signal = self.snapshot
            original = signal.set

            def set_started(snap):
                snap.started = 1.0
                original(snap)

            signal.set = set_started
            try:
                return super().publish(state, **kwargs)
            finally:
                signal.set = original

    return Jobs()


def _glossary_lines(k: int, chars: int) -> tuple:
    thread = f"Thread-{10 + k} (api_call)"
    row = "character,김서연,Seo-yeon Kim,female,a recurring character with a long description\n"
    body = "type,raw_name,translated_name,gender,description\n" + row * max(1, chars // len(row))
    return thread, [f"🚀 [{thread}] Sending API call now", f"📡 [{thread}] Text streaming...", body]


def _chapter_lines(k: int) -> tuple:
    thread = f"Thread-{40 + k} (api_call)"
    return thread, [f"🚀 [{thread}] Sending API call now", f"📡 [{thread}] Text streaming...",
                    f"GLFAKE chapter {k} translated text"]


GATE_CASES = [(3, 45_000, 20), (20, 1_500, 20), (4, 600, 4)]


@needs_backend
@needs_flet
@pytest.mark.parametrize("cards, chars, limit", GATE_CASES)
def test_owner_issue12_book_card_stays_live_through_the_glossary_gate(desktop_store_cls, isolated_env,
                                                                     cards, chars, limit):
    """Owner issue 12, the exact complaint: at the glossary gate the attachment's card said "Done" and the
    translation's request cards never appeared in the chat (only in Jobs). Now the turn's JobCard stays the
    live card through the gate - "Waiting for your glossary decision", the book's title, the gate's cards
    listed - then lists the translation's live requests with the progress, and a Stop shows Stopped +
    Resume. After a relaunch the card reads the job's end from JobService for the real turn."""
    from glossarion_mobile.services.jobs import JobSnapshot, JobSpec, JobState, Progress
    from glossarion_mobile.ui.chat.cards import JobCard
    from glossarion_mobile.ui.chat.job_binding import JobsAdapter
    from glossarion_mobile.ui.chat.run_controller import ChatRuns

    tmp_path = isolated_env

    async def scenario():
        jobs = _started_jobs()
        chat = _TC._chat_view(desktop_store_cls, tmp_path, cid="5", jobs=jobs)
        view, adapter, runs, calls = chat.view, chat.adapter, chat.runs, chat.calls
        _post_to_loop(view)
        if limit != 20:
            chat.store.set("direct_text_rendered_card_limit", limit)
        book = tmp_path / "novel.epub"
        book.write_bytes(b"PK\x03\x04fake epub")
        record = {"path": str(book), "name": "novel.epub", "extension": ".epub", "size": 15}
        try:
            run = await runs.send("5", text="", attachment=record, settings=view.settings("5"), output_mode="text")
            view._on_run_changed("5")
            jobs.publish("RUNNING", last_line="📑 Running glossary extraction before translation...")
            for k in range(cards):
                thread, lines = _glossary_lines(k, chars)
                for line in lines:
                    jobs.line(line, thread)
            run.stream.drain(final=True)  # the stream loop's ticks, all at once
            view._update_live_job_card(run)
            live = view.live_job_card
            assert live is not None and live.requests_tile.title == f"Requests ({cards})"

            # the gate: the job asks, ChatRuns.commit_gate freezes the glossary cards into the chat
            jobs.publish("RUNNING", question={"id": "q1", "kind": "direct_text_glossary_approval",
                                              "data": {"path": str(tmp_path / "glossary.csv")}})
            await _quiet(calls)
            messages = adapter.messages("5")
            assert len(messages) == 1 + cards and run.awaiting_glossary
            assert tail_window(messages, limit)[0] > run.user_index  # the old window cut the turn's file off
            job_cards = _cards(view, JobCard)
            assert len(job_cards) == 1, "one card for the turn"
            card = job_cards[0]
            assert card is view.live_job_card is live, "the running card stays the live card at the gate"
            assert card.phase.name == "running" and card.state_text.value == "Waiting for your glossary decision"
            assert card.title_text.value == "novel.epub" and card.requests_tile.title == f"Requests ({cards})"
            assert len(card.requests_column.controls) == min(cards, limit) and card.requests_tile.expanded
            assert not any(c.state_text.value == "Done" for c in job_cards)
            assert view.approval_card is not None and view.approval_card in view.transcript.tail

            # ✓ Yes, the translation streams its requests and reports progress
            view._answer_glossary(True)
            jobs.publish("RUNNING", last_line="📑 Glossary extraction complete, proceeding to translation...")
            for k in range(1, 4):
                thread, lines = _chapter_lines(k)
                for line in lines:
                    jobs.line(line, thread)
            jobs.publish("RUNNING", progress={"total": 12, "completed": 3}, in_flight=1, last_line="📤 chapter 3")
            run.stream.drain(final=True)  # the stream loop's ticks
            view._update_live_job_card(run)
            assert _cards(view, JobCard) == [card] and card is view.live_job_card
            assert card.phase.name == "running" and card.state_text.value == "Translating"
            assert card.requests_tile.title == f"Requests ({cards + 3})"
            assert len(card.requests_column.controls) == min(cards + 3, limit)
            assert "Chapter 3/12" in card.line_text.value
            labels = [row.content.controls[0].controls[0].value for row in card.requests_column.controls]
            assert labels[-1] == view._live_segments(run)[-1][1]["label"]  # the live request is listed

            # ■ Stop: the run commits; the Result card keeps its turn
            jobs.publish("CANCELLED", progress={"total": 12, "completed": 3})
            _TC._finish_all(runs)
            await asyncio.sleep(0.05)  # the finish thread's notification runs on the loop
            await _quiet(calls)
            assert not run.live and view.live_job_card is None
            ended = _cards(view, JobCard)
            assert len(ended) == 1 and ended[0].title_text.value == "novel.epub"
            assert ended[0].phase.name == "stopped" and ended[0].state_text.value.startswith("Stopped")
            assert "resume" in ended[0].action_buttons
            assert view._turn_source(view.items[-1]) == str(book)  # Reader / Library / QA know the turn's file

            # relaunch: no run in this session; JobService remembers how the turn's job ended
            asked: list = []

            def persisted(cid, index):
                asked.append(int(index))
                if int(index) != run.user_index:
                    return None
                return JobSnapshot(id="killed", spec=JobSpec("direct_text", "novel.epub"), state=JobState.INTERRUPTED,
                                   created=1.0, started=2.0, progress=Progress(total=12, completed=3))

            relaunched = ChatRuns(adapter, JobsAdapter(_TC.FakeJobService()), temp_dir=str(tmp_path))
            relaunched.persisted_job = persisted
            view.env.runs = relaunched
            view.load_chat("5")
            await _quiet(calls)
            card = _cards(view, JobCard)[0]
            assert card.state_text.value == "Interrupted · 3/12 chapters" and "resume" in card.action_buttons
            assert run.user_index in asked and -1 not in asked
        finally:
            for task in calls.tasks:
                task.cancel()
            _TC._close(chat)

    asyncio.run(scenario())


@needs_backend
@needs_flet
def test_an_orphan_run_never_takes_over_another_turns_card(desktop_store_cls, isolated_env):
    """A recovered job whose turn was deleted has ``user_index`` -1: no card is its live card."""
    from glossarion_mobile.ui.chat.cards import JobCard
    from glossarion_mobile.ui.chat.run_controller import ChatRun
    from glossarion_mobile.ui.chat.run_request import DirectTextRun
    from glossarion_mobile.ui.chat.stream_bridge import RunStream

    async def scenario():
        chat = _TC._chat_view(desktop_store_cls, isolated_env, cid="5")
        view, runs, adapter = chat.view, chat.runs, chat.adapter
        try:
            # a long finished book turn: the old message window cut its file card off (index -1, too)
            _history_chat(adapter, "5", _book_turn(str(isolated_env / "long.epub"), 25, 500))
            orphan = ChatRun(cid="5", run=DirectTextRun(temp_root="", source_path="", source_extension=".epub",
                                                        is_attachment=True, expected_output=""),
                             stream=RunStream(), user_index=-1, job_id="j-orphan", state="running")
            runs.runs["5"] = orphan
            view.load_chat("5")
            cards = _cards(view, JobCard)
            assert len(cards) == 1 and cards[0].title_text.value == "long.epub"
            assert view.live_job_card is None and cards[0].phase.name == "done"
        finally:
            runs.runs.pop("5", None)
            for task in chat.calls.tasks:
                task.cancel()
            _TC._close(chat)

    asyncio.run(scenario())


def _history_chat(adapter, cid: str, messages: list) -> None:
    adapter.append_messages(cid, messages)


@needs_backend
@needs_flet
def test_owner_issue13_scroll_for_earlier_messages_loads_earlier_cards(desktop_store_cls, isolated_env, monkeypatch):
    """Owner issue 13, the exact complaint: tapping "↑ Scroll for earlier messages (N hidden)" did nothing
    and a book turn showed one 'Attachment' card with a few requests. Now each tap renders a page of
    earlier cards (N drops, the cards change) and the window stays where it was slid through every kind
    of re-render; the book turn is its file card + its own JobCard listing every request (newest first
    page, "↑ Show … earlier requests" for the rest)."""
    from glossarion_mobile.ui.chat import chat_view as cv
    from glossarion_mobile.ui.chat.cards import JobCard
    from glossarion_mobile.ui.chat.messages import AssistantMessage, UserBubble, UserFileCard
    from glossarion_mobile.ui.chat.transcript import EARLIER_TEMPLATE

    monkeypatch.setattr(cv, "HIGHLIGHT_SECONDS", 0)
    tmp_path = isolated_env
    book = tmp_path / "saga.epub"
    book.write_bytes(b"PK\x03\x04fake epub")
    folder = str(tmp_path / "Output" / "Attachments" / "saga")

    async def scenario():
        chat = _TC._chat_view(desktop_store_cls, tmp_path, cid="5")
        view, adapter, calls = chat.view, chat.adapter, chat.calls
        scrolls: list = []

        async def scroll_to(**kwargs):
            scrolls.append(kwargs)

        view.transcript.scroll_to = scroll_to
        try:
            # 12 text turns of pasted chapters (5k source + 15k translation) + a 40-chapter book turn
            messages = _text_turns(12, 15_000, source=5_000) + _book_turn(str(book), 40, 12_000, folder)
            _history_chat(adapter, "5", messages)
            view.load_chat("5")
            await _quiet(calls)
            hidden = view.transcript.hidden_before
            assert hidden > 0 and view.transcript.earlier_row.content == EARLIER_TEMPLATE.format(n=hidden)

            # the tap: a page (limit // 3 cards) of earlier cards, every time, until none is hidden
            seen = [hidden]
            while view.transcript.hidden_before:
                first = view.transcript.cards[0]
                view.transcript.earlier_row.on_click(None)
                await _quiet(calls)
                assert view.transcript.hidden_before < seen[-1], "the tap loads earlier cards"
                assert view.transcript.cards[0] is not first
                seen.append(view.transcript.hidden_before)
                # the slid window survives every re-render (live snapshots, store changes, finishes)
                window = view.window
                view.render_transcript(follow=True)
                view._on_run_changed("5")
                view._on_store_changed()
                assert view.window == window
            assert seen[0] - seen[1] == 6 and seen[-1] == 0
            assert isinstance(view.transcript.cards[0], UserBubble) and view.transcript.hidden_after > 0
            assert not view.transcript.follow_tail, "reading history: a live run would not pull the view down"

            # the book turn: its file card + its own JobCard (title, state, every request, paged)
            view.load_chat("5")
            await _quiet(calls)
            total = len(view.items)
            assert total == 24 + 2 and view.window == (hidden, total) and view.transcript.hidden_after == 0
            shown = view.transcript.cards
            assert isinstance(shown[-2], UserFileCard) and isinstance(shown[-1], JobCard)
            job = shown[-1]
            assert job.title_text.value == "saga.epub" and job.state_text.value == "Done"
            assert job.requests_tile.title == "Requests (41)" and len(job.requests_column.controls) == 20
            assert job.earlier_requests_button.visible
            job.earlier_requests_button.on_click(None)
            assert len(job.requests_column.controls) == 40
            view.slide(-1)
            view.slide(-1)
            view.slide(-1)

            # an append while the user reads history keeps the window; ↓ brings the newest back
            window = view.window
            adapter.append_messages("5", [("user", "a later question")])
            view.render_transcript()
            assert view.window == window and view.transcript.hidden_after > 0
            await view._jump_to_end()
            assert view.window[1] == len(view.items) and view.transcript.hidden_after == 0
            assert view.transcript.follow_tail

            # jump-to / search reach the earliest card and a request inside the book's card
            await view.jump_to(0)
            assert view.window[0] == 0 and view.transcript.slot_for(view._scroll_key_for(0)) is not None
            view.render_transcript(follow=True)
            assert view.window[0] == 0, "the jump is not snapped back to the tail"
            chapter_two = 24 + 3  # user_file 24, glossary 25, chapter 1 at 26
            await view.jump_to(chapter_two)
            key = view._scroll_key_for(chapter_two)
            assert key.value == "job-24"
            slot = view.transcript.slot_for(key)
            assert slot is not None and slot.card.requests_tile.expanded
            labels = [row.content.controls[0].controls[0].value for row in slot.card.requests_column.controls]
            assert any(label.startswith("Chapter 2 (chunk") for label in labels)
            assert scrolls and scrolls[-1].get("scroll_key").value == "job-24"
            # a text card the window had dropped
            await view.jump_to(3)
            assert view.transcript.slot_for(view._scroll_key_for(3)) is not None
            assert isinstance(view.transcript.slot_for(view._scroll_key_for(3)).card, AssistantMessage)
        finally:
            for task in calls.tasks:
                task.cancel()
            _TC._close(chat)

    asyncio.run(scenario())


@needs_backend
@needs_flet
def test_scrolling_near_the_top_loads_earlier_cards_and_keeps_the_viewport(desktop_store_cls, isolated_env):
    """UI_SPEC §2.8: a user scroll near the top prepends a page of cards and puts the card that was at the
    top back in view (``scroll_to(scroll_key=…, duration=0)``); one load per gesture."""
    from glossarion_mobile.ui.chat.transcript import CardSlot

    async def scenario():
        chat = _TC._chat_view(desktop_store_cls, isolated_env, cid="5")
        view, adapter, calls = chat.view, chat.adapter, chat.calls
        scrolls: list = []

        async def scroll_to(**kwargs):
            scrolls.append(kwargs)

        view.transcript.scroll_to = scroll_to
        try:
            _history_chat(adapter, "5", _text_turns(30, 200))
            view.load_chat("5")
            await _quiet(calls)
            assert view.window == (40, 60)
            first_key = next(c for c in view.transcript.messages if isinstance(c, CardSlot)).slot_key
            event = types.SimpleNamespace(event_type="update", pixels=120.0, min_scroll_extent=0.0,
                                          max_scroll_extent=9000.0, viewport_dimension=800.0, scroll_delta=-35.0)
            view.transcript.on_scroll(event)
            view.transcript.on_scroll(event)  # the same gesture: still restoring, nothing more
            assert view.window == (34, 54) and view.transcript.edge_loads == 1
            await _quiet(calls)
            assert scrolls[-1] == {"scroll_key": scrolls[-1]["scroll_key"], "duration": 0}
            assert scrolls[-1]["scroll_key"].value == first_key  # the old top card is back in view
            view.transcript.release_edge_load(hold=0)
            view.transcript.on_scroll(event)
            await _quiet(calls)
            assert view.window == (28, 48)
        finally:
            for task in calls.tasks:
                task.cancel()
            _TC._close(chat)

    asyncio.run(scenario())


@needs_backend
@needs_flet
def test_actions_that_add_a_card_show_the_newest_cards(desktop_store_cls, isolated_env):
    """A slid window never hides the card the user just made (a batch plan, a cancelled plan, a send)."""

    async def scenario():
        chat = _TC._chat_view(desktop_store_cls, isolated_env, cid="5")
        view, adapter, calls = chat.view, chat.adapter, chat.calls
        try:
            _history_chat(adapter, "5", _text_turns(30, 200))
            view.load_chat("5")
            view.slide(-1)
            view.slide(-1)
            assert view.transcript.hidden_after and not view.transcript.follow_tail
            book = isolated_env / "a.epub"
            book.write_bytes(b"PK\x03\x04")
            view.set_batch([str(book)])
            assert view.transcript.hidden_after == 0 and view.window[1] == len(view.items)
            assert view.transcript.follow_tail
        finally:
            for task in calls.tasks:
                task.cancel()
            _TC._close(chat)

    asyncio.run(scenario())
