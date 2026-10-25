"""Hotfix 1 (owner device report on the current APK), the chat cards' issues 2, 4 and 8.

2. "Clicking refine doesn't hide the response card, which results in an issue where the user clicks yes during
   refinement, so we end up having translation and refinement requests take place at the same time. The
   refinement is meant for the translation, so it doesn't make sense for it to start before it. Solve this
   hiding the response card during glossary refinement."
   ✨ Refine on the glossary approval card hides Edit / Refine / ✓ Yes / ■ No / Always accept for the whole
   refinement and shows the refining state (a ring and "Refining the glossary…") instead; an answer that still
   reaches the chat meanwhile is refused (nothing goes to JobService), the composer and the running card say
   so, and the card stays that way on a repaint, a card rebuild and a chat switch. When the refinement ends the
   buttons come back ("✨ Refined", or the error). ``JobSnapshot.refining`` (``JobService.refine_at_gate``)
   is followed too, whichever card is on screen.
4. "Clicking resume doesn't update the response card, so the translation starts, but the response card still
   says resume." Resume (and Retry failed) turns the Result card into the running card at the tap, before its
   run is submitted (Stop available, no Resume to tap twice), the run's live card is that same object, it
   reads "Queued · starts after …" only while another job runs, Stop before the run exists cancels it, and an
   older turn's card never resubmits the chat's latest job in its place.
8. "Response card run options are missing the api call delay variable." The Run options panel (Plan card,
   Result card for Resume / Retry failed, batch plan) has the desktop main window's "API call delay (s, or
   range 5-7)" tile (config ``delay`` + ``delay_max``); its per-run value reaches the job's config and the
   SEND_INTERVAL_SECONDS / SEND_INTERVAL_MAX_SECONDS it exports.

Run from src/mobile (mobile venv):
    python -m pytest -p no:cacheprovider -W ignore tests_host/test_hotfix1_chat_cards.py -q
"""

from __future__ import annotations

import asyncio
import importlib.util
import inspect
import json
import sys
import threading
import types
from pathlib import Path

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
for _path in (MOBILE_DIR / "app", MOBILE_DIR / "tests", MOBILE_DIR.parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


def _load(name: str, file_name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file_name))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # its dataclasses look their module up
    spec.loader.exec_module(module)
    return module


RO = _load("_glossarion_hotfix1_chat_cards_ro", "test_run_options_resume.py")
TC = RO.TC
desktop_store_cls = TC.desktop_store_cls
isolated_env = TC.isolated_env
needs_flet = TC.needs_flet

QUESTION = {"id": "q1", "kind": "direct_text_glossary_approval", "data": {}}
GLOSSARY = "type,raw_name,translated_name\ncharacter,김철수,Kim Cheolsu\nterm,마왕성,Demon King's Castle\n"


class RefineJobs(TC.FakeJobService):
    """The fake JobService with the gate's ``refine_at_gate`` (blocking, like the real one: it runs off the loop)
    held until ``release``; ``fail`` makes it raise."""

    def __init__(self):
        super().__init__()
        self.release = threading.Event()
        self.started = threading.Event()
        self.refines: list = []
        self.fail = ""

    def refine_at_gate(self, job_id, glossary_path):
        self.refines.append((job_id, glossary_path))
        self.started.set()
        assert self.release.wait(20), "the test never released the refinement"
        if self.fail:
            raise RuntimeError(self.fail)
        Path(glossary_path).write_text(GLOSSARY + "term,용사,Hero\n", encoding="utf-8")
        return {"ok": True, "glossary_path": glossary_path}

    def publish_refining(self, refining: bool):
        """The job's next snapshot with ``JobSnapshot.refining`` (``_suspend_question`` / ``_resume_question``)."""
        snap = types.SimpleNamespace(**vars(self.snapshot.value))
        snap.refining = refining
        self.snapshot.set(snap)
        return snap


def _hidden(card) -> bool:
    """The approval card shows the refining state, not its buttons."""
    return (not card.edit_row.visible and not card.answer_row.visible and not card.always_button.visible
            and card.refining_row.visible and card.refining)


def _answerable(card) -> bool:
    return (card.edit_row.visible and card.answer_row.visible and not card.refining_row.visible
            and not card.refining and not card.yes_button.disabled and not card.no_button.disabled)


async def _until(predicate, timeout: float = 10.0) -> bool:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.02)
    return bool(predicate())


async def _gate(chat, tmp_path):
    """An attachment run of chat 5 waiting at the glossary gate (the approval card in the tail)."""
    from glossarion_mobile.ui.chat.direct_text_rules import DirectTextSettings

    glossary = tmp_path / "book_glossary.csv"
    glossary.write_text(GLOSSARY, encoding="utf-8")
    book = tmp_path / "book.epub"
    book.write_bytes(b"PK\x03\x04")
    record = {"path": str(book), "name": "book.epub", "extension": ".epub", "size": 4}
    run = await chat.runs.send("5", text="", attachment=record, settings=DirectTextSettings(), output_mode="text")
    chat.jobs.publish("RUNNING", question=dict(QUESTION, data={"path": str(glossary)}))
    assert run.awaiting_glossary
    chat.view.render_transcript()
    card = chat.view.approval_card
    assert card is not None and card in chat.view.transcript.tail
    return run, card, glossary


# ==========================================================================
# 2. ✨ Refine hides the approval card's answers until the refinement ended
# ==========================================================================


@needs_flet
def test_owner_2_yes_during_refinement_never_starts_the_translation(desktop_store_cls, isolated_env):
    """The owner's complaint: ✓ Yes tapped while ✨ Refine runs started the translation next to the refinement.
    Now the card hides its answers while it refines, every way an answer could still reach the chat is refused
    (JobService gets nothing), and Yes works once the refinement is done - with the refined glossary."""
    from glossarion_mobile.services.jobs import GLOSSARY_REFINING_TEXT
    from glossarion_mobile.ui.chat import chat_view as cv
    from glossarion_mobile.ui.chat.cards import ACTION_REQUIRED_LABEL, REFINING_LABEL, JobCard

    tmp_path = isolated_env
    prefs = TC._Prefs()

    async def scenario():
        chat = TC._chat_view(desktop_store_cls, tmp_path, cid="5", prefs=prefs, jobs=RefineJobs())
        view, jobs = chat.view, chat.jobs
        view.state.send_block = lambda: None  # the engine is ready and the model needs no sign-in
        try:
            await RO._settle(chat.calls)
            run, card, glossary = await _gate(chat, tmp_path)
            assert not card.yes_button.disabled and card.refine_button.visible and not card.refine_button.disabled
            assert view.send_inputs().block is cv.BLOCK_GLOSSARY_PENDING

            card.refine_button.on_click(None)  # ✨ Refine
            assert await _until(jobs.started.is_set), "the refinement did not start"
            assert jobs.refines == [(run.job_id, str(glossary))]
            card.yes_button.on_click(None)  # the owner's tap: the translation must not start next to the refinement
            assert jobs.answers == [] and run.awaiting_glossary, "✓ Yes was sent while the glossary was refined"
            assert _hidden(card) and card.action_label.value == REFINING_LABEL

            # every way an answer could still be made goes nowhere
            card.yes_button.on_click(None)
            card.no_button.on_click(None)
            card.always_button.on_click(None)
            card.always_button.on_click(None)
            card.edit_button.on_click(None)
            view._answer_glossary(True)          # a card that had not shown the refinement yet
            view._always_accept_glossary()
            assert jobs.answers == [] and run.awaiting_glossary and jobs.stops == []
            assert prefs.values == {}  # "Always accept" was not switched on either
            assert any(note[0] == cv.GLOSSARY_REFINING_WAIT for note in chat.calls.notes)
            assert _hidden(card) and card.answered is None  # the refused tap is not spent
            # the composer and the running card say what is happening
            assert view.send_inputs().block is cv.BLOCK_GLOSSARY_REFINING
            assert cv.BLOCK_GLOSSARY_REFINING.message == GLOSSARY_REFINING_TEXT
            live = [c for c in TC._cards(view, JobCard) if c is view.live_job_card]
            assert live and live[0].state_text.value == GLOSSARY_REFINING_TEXT

            # repaints, a rebuilt card and a chat switch while it refines: still no buttons
            view.render_transcript()
            assert view.approval_card is card and _hidden(card)
            view.load_chat("2")
            view.load_chat("5")
            rebuilt = view.approval_card
            assert rebuilt is not None and rebuilt is not card and rebuilt in view.transcript.tail
            assert _hidden(rebuilt)
            rebuilt.yes_button.on_click(None)
            assert jobs.answers == []

            # the refinement ends: the answers come back on the card that is on screen, with the refined file
            jobs.release.set()
            assert await _until(lambda: _answerable(rebuilt)), "the card did not come back after the refinement"
            await RO._settle(chat.calls)
            assert rebuilt.refine_button.content == "✨ Refined" and rebuilt.refine_button.disabled
            assert rebuilt.action_label.value == ACTION_REQUIRED_LABEL and not rebuilt.refine_error.visible
            assert rebuilt.info["entries"] == 3  # the preview shows the refined glossary
            assert view.send_inputs().block is cv.BLOCK_GLOSSARY_PENDING
            view.render_transcript()
            assert _answerable(view.approval_card) and view.approval_card.refine_outcome == "refined"
            view.approval_card.yes_button.on_click(None)
            assert jobs.answers == [("q1", True)] and not run.awaiting_glossary
            jobs.publish("CANCELLED")
            TC._finish_all(chat.runs)
        finally:
            jobs.release.set()
            TC._close(chat)

    asyncio.run(scenario())


@needs_flet
def test_refine_follows_the_job_snapshot_and_shows_a_failure(desktop_store_cls, isolated_env):
    """``JobSnapshot.refining`` alone (the job service's word: a card rebuilt after the app came back, any card)
    hides the answers until it goes off; a refinement that fails brings them back with its error and Refine
    can be tried again."""
    tmp_path = isolated_env

    async def scenario():
        chat = TC._chat_view(desktop_store_cls, tmp_path, cid="5", jobs=RefineJobs())
        view, jobs = chat.view, chat.jobs
        try:
            await RO._settle(chat.calls)
            run, card, _glossary = await _gate(chat, tmp_path)
            jobs.publish_refining(True)
            assert _hidden(view.approval_card)
            view.approval_card = None  # rebuilt (app back from the background, a new window)
            view.render_transcript()
            assert _hidden(view.approval_card)
            view.approval_card.yes_button.on_click(None)
            view._answer_glossary(True)
            assert jobs.answers == []
            jobs.publish_refining(False)
            assert _answerable(view.approval_card)

            # ✨ Refine that fails: the error on the card, the answers and Refine back
            card = view.approval_card
            jobs.fail = "The selected entry type(s) contain no glossary entries"
            assert card.refine()
            assert _hidden(card)
            jobs.release.set()
            assert await _until(lambda: not card.refining)
            await RO._settle(chat.calls)
            assert _answerable(card) and card.refine_outcome == "failed"
            assert card.refine_error.visible and jobs.fail in card.refine_error.value
            assert card.refine_button.content == "✨ Refine" and not card.refine_button.disabled
            assert not [n for n in chat.calls.notes if "Could not refine" in n[0]]  # said on the card, no snackbar
            card.no_button.on_click(None)
            assert jobs.answers == [("q1", False)]
            jobs.publish("CANCELLED")
            TC._finish_all(chat.runs)
        finally:
            jobs.release.set()
            TC._close(chat)

    asyncio.run(scenario())


@needs_flet
def test_approval_card_refining_state_unit():
    """The card on its own: ``set_refining`` hides and refuses, ``refined`` restores, ``reopen`` un-spends a tap."""
    from glossarion_mobile.ui.chat.cards import GlossaryApprovalCard

    answered: list = []
    edits: list = []
    refines: list = []
    card = GlossaryApprovalCard(path="", info={"exists": True, "name": "g.csv", "entries": 2, "preview": []},
                                on_answer=answered.append, on_edit=edits.append, on_always=lambda: answered.append("always"),
                                on_refine=lambda: refines.append(1))
    card.path = "g.csv"
    assert card.refine() and refines == [1] and _hidden(card)
    assert not card.refine() and refines == [1]  # one refinement at a time
    assert not card.answer(True) and not card.always() and not card.always_tap()
    card._edit()
    assert answered == [] and edits == []
    card.refined(True, None)
    assert _answerable(card) and card.always_button.visible and card.refine_button.disabled
    assert card.answer(True) and answered == [True]
    card.reopen()
    assert card.answered is None and not card.yes_button.disabled and card.refine_button.disabled  # refined once
    assert card.set_refining(True) and not card.set_refining(True)
    assert card.set_refining(False) and _answerable(card)


def test_running_label_while_refining():
    from glossarion_mobile.services.jobs import GLOSSARY_REFINING_TEXT
    from glossarion_mobile.ui.chat.job_binding import GLOSSARY_REFINING_LABEL, STARTING_LABEL, running_label

    assert GLOSSARY_REFINING_LABEL == GLOSSARY_REFINING_TEXT  # the job's own words
    assert STARTING_LABEL == "Starting translation…" == running_label(types.SimpleNamespace(state="QUEUED"))
    running = types.SimpleNamespace(state="RUNNING", last_line="")
    assert running_label(running, waiting_for_glossary=True) == "Waiting for your glossary decision"
    assert running_label(running, waiting_for_glossary=True, refining=True) == GLOSSARY_REFINING_TEXT
    refining = types.SimpleNamespace(state="RUNNING", last_line="", refining=True)
    assert running_label(refining, waiting_for_glossary=True) == GLOSSARY_REFINING_TEXT
    stopping = types.SimpleNamespace(state="STOPPING", last_line="", refining=True)
    assert running_label(stopping, waiting_for_glossary=True) == "Stopping after current request…"


# ==========================================================================
# 4. Resume / Retry failed: the running card at once
# ==========================================================================


def _turn_cards(view):
    from glossarion_mobile.ui.chat.cards import JobCard

    return [c for c in TC._cards(view, JobCard)]


class HistoryJobs(TC.FakeJobService):
    """The fake JobService that remembers its ended jobs (``JobService.view().history``), so a chat knows how each
    of its turns' jobs ended (``ChatRuns.persisted_job``) and which job a turn ran (``_chat_jobs``)."""

    def __init__(self):
        super().__init__()
        self.history: list = []
        self.clock = 0.0

    def view(self):
        return types.SimpleNamespace(active=None, queue=[], interrupted=[], history=list(self.history))

    def publish(self, state, **kwargs):
        snap = super().publish(state, **kwargs)
        if state in ("DONE", "FAILED", "CANCELLED"):
            self.clock += 1.0
            snap.finished = self.clock
            self.history.insert(0, snap)
        return snap


async def _stop_run(chat) -> None:
    """Stop the chat's live run (no output yet) and render its Result."""
    chat.jobs.publish("RUNNING")
    chat.runs.request_stop(chat.view.cid)
    chat.jobs.publish("CANCELLED")
    TC._finish_all(chat.runs)
    chat.view.render_transcript()


async def _stopped_book(chat, tmp_path, name="novel.epub", stop=RO._stop):
    """Attach, Start, stop: the turn's stopped Result card (with Resume)."""
    view = chat.view
    book = tmp_path / name
    book.write_bytes(b"PK\x03\x04")
    assert view.attach_file(str(book))
    view.begin_send()
    view.start_plan()
    await RO._settle(chat.calls)
    await stop(chat)
    card = next(c for c in _turn_cards(view) if c.title_text.value == name)
    assert "resume" in card.action_buttons and card.phase.name == "stopped"
    return card


@needs_flet
def test_owner_4_resume_turns_the_card_into_the_running_card_at_once(desktop_store_cls, isolated_env):
    """The owner's complaint: after Resume the translation started but the card still said Resume. The tap itself
    turns the card into the running card (no Resume, Stop available) before the run is even submitted; a second
    tap on the old button sends nothing; the run's live card is the same object through the job's events and
    the end of the run."""
    STARTING_LABEL = "Starting translation…"  # job_binding.STARTING_LABEL (the words, so this also runs on old code)

    tmp_path = isolated_env

    async def scenario():
        chat = TC._chat_view(desktop_store_cls, tmp_path)
        view, jobs = chat.view, chat.jobs
        try:
            await RO._settle(chat.calls)
            ended = await _stopped_book(chat, tmp_path)
            before = len(jobs.submitted)
            ended.action_buttons["resume"].on_click(None)
            # at once, before anything was awaited: the running card
            live = view.live_job_card
            assert live is not None and live is not ended and live in _turn_cards(view)
            assert ended not in _turn_cards(view)
            assert live.phase.name == "running" and live.state_text.value == STARTING_LABEL
            assert "stop" in live.action_buttons and "resume" not in live.action_buttons
            assert live.title_text.value == "novel.epub"
            ended.action_buttons["resume"].on_click(None)  # the stale button once more: nothing more is sent
            view.render_transcript()
            assert view.live_job_card is live
            await RO._settle(chat.calls)
            assert len(jobs.submitted) == before + 1, chat.calls.notes
            run = chat.runs.live_run(view.cid)
            assert run is not None and run.job_id is not None
            # submitted, its job not reported yet: still the same running card ("Starting translation…", Stop)
            assert view.live_job_card is live and live.phase.name == "running" and "stop" in live.action_buttons
            assert live.state_text.value == STARTING_LABEL
            jobs.publish("QUEUED")
            assert view.live_job_card is live and live.phase.name == "running"  # nothing ahead: it starts now
            jobs.publish("RUNNING")
            assert view.live_job_card is live and live.state_text.value == "Translating"
            # the app goes to the background and comes back: the card is rebuilt from the run, still live
            view.load_chat("5")
            view.load_chat("2")
            assert view.live_job_card is not None and view.live_job_card.phase.name == "running"
            assert "resume" not in view.live_job_card.action_buttons
            # the run ends: the Result again, which can be resumed again
            jobs.publish("CANCELLED")
            TC._finish_all(chat.runs)
            view.render_transcript()
            ended = next(c for c in _turn_cards(view) if c.title_text.value == "novel.epub")
            assert view.live_job_card is None and ended.phase.name == "stopped" and "resume" in ended.action_buttons
        finally:
            TC._close(chat)

    asyncio.run(scenario())


@needs_flet
def test_resume_behind_another_job_reads_queued_then_running(desktop_store_cls, isolated_env):
    """A resumed run waiting behind another chat's job says so (and can be cancelled), then runs on the same card."""
    tmp_path = isolated_env

    async def scenario():
        chat = TC._chat_view(desktop_store_cls, tmp_path)
        view, jobs = chat.view, chat.jobs
        try:
            await RO._settle(chat.calls)
            ended = await _stopped_book(chat, tmp_path)
            other = types.SimpleNamespace(id="other", spec=types.SimpleNamespace(kind="translate", title="Other book",
                                                                                 params={}),
                                          state="RUNNING", question=None, progress=None, in_flight=0, last_line="")
            jobs.snapshot.value = other  # another job runs (the Library's)
            ended.action_buttons["resume"].on_click(None)
            live = view.live_job_card
            await RO._settle(chat.calls)
            view.render_transcript()
            assert view.live_job_card is live and live.phase.name == "queued"
            assert live.state_text.value == "Queued · starts after Other book"
            assert "cancel_queued" in live.action_buttons and "resume" not in live.action_buttons
            jobs.publish("RUNNING")
            assert view.live_job_card is live and live.phase.name == "running" and "stop" in live.action_buttons
            jobs.publish("CANCELLED")
            TC._finish_all(chat.runs)
        finally:
            TC._close(chat)

    asyncio.run(scenario())


@needs_flet
def test_stop_before_the_resumed_run_exists_cancels_it(desktop_store_cls, isolated_env):
    """Stop on the card right after Resume (its run not submitted yet): nothing is submitted, the Result is back."""
    tmp_path = isolated_env

    async def scenario():
        chat = TC._chat_view(desktop_store_cls, tmp_path)
        view, jobs = chat.view, chat.jobs
        try:
            await RO._settle(chat.calls)
            ended = await _stopped_book(chat, tmp_path)
            before = len(jobs.submitted)
            ended.action_buttons["resume"].on_click(None)
            live = view.live_job_card
            live.action_buttons["stop"].on_click(None)
            assert view.live_job_card is live and live.phase.name == "stopping"
            await RO._settle(chat.calls)
            assert len(jobs.submitted) == before and chat.runs.live_run(view.cid) is None
            card = next(c for c in _turn_cards(view) if c.title_text.value == "novel.epub")
            assert card.phase.name == "stopped" and "resume" in card.action_buttons and view.live_job_card is None
        finally:
            TC._close(chat)

    asyncio.run(scenario())


@needs_flet
def test_retry_failed_turns_the_card_into_the_running_card_at_once(desktop_store_cls, isolated_env):
    """"Finished with issues" (done with failed chapters) › Retry failed: the same as Resume."""
    tmp_path = isolated_env

    async def scenario():
        chat = TC._chat_view(desktop_store_cls, tmp_path)
        view, jobs = chat.view, chat.jobs
        try:
            await RO._settle(chat.calls)
            book = tmp_path / "novel.epub"
            book.write_bytes(b"PK\x03\x04")
            assert view.attach_file(str(book))
            view.begin_send()
            view.start_plan()
            await RO._settle(chat.calls)
            jobs.publish("RUNNING")
            jobs.publish("DONE", progress={"total": 10, "completed": 8, "failed": 2})
            TC._finish_all(chat.runs)
            view.render_transcript()
            ended = next(c for c in _turn_cards(view) if c.title_text.value == "novel.epub")
            assert ended.state_text.value.startswith("Finished with issues") and "retry" in ended.action_buttons
            before = len(jobs.submitted)
            ended.action_buttons["retry"].on_click(None)
            live = view.live_job_card
            assert live is not None and live.phase.name == "running" and "retry" not in live.action_buttons
            await RO._settle(chat.calls)
            assert len(jobs.submitted) == before + 1 and view.live_job_card is live
            jobs.publish("RUNNING")
            assert view.live_job_card is live and "stop" in live.action_buttons
            jobs.publish("CANCELLED")
            TC._finish_all(chat.runs)
        finally:
            TC._close(chat)

    asyncio.run(scenario())


@needs_flet
def test_an_older_turns_resume_never_runs_the_latest_turns_job(desktop_store_cls, isolated_env):
    """Two stopped books in one chat: Resume on the first one's card used to resubmit the chat's LATEST job (the
    second book) while the first card kept saying Resume. It resumes its own turn's job where ``ChatRuns.resubmit``
    can be told the turn, and is refused otherwise; the second card's Resume runs its own job."""
    from glossarion_mobile.ui.chat import chat_view as cv

    tmp_path = isolated_env

    async def scenario():
        chat = TC._chat_view(desktop_store_cls, tmp_path, jobs=HistoryJobs())
        view, jobs = chat.view, chat.jobs
        try:
            await RO._settle(chat.calls)
            await _stopped_book(chat, tmp_path, "first.epub", stop=_stop_run)
            first_root = jobs.submitted[-1].params["run"]["temp_root"]
            await _stopped_book(chat, tmp_path, "second.epub", stop=_stop_run)
            second_root = jobs.submitted[-1].params["run"]["temp_root"]
            first = next(c for c in _turn_cards(view) if c.title_text.value == "first.epub")
            assert first.phase.name == "stopped" and "resume" in first.action_buttons
            before = len(jobs.submitted)
            first.action_buttons["resume"].on_click(None)
            await RO._settle(chat.calls)
            sent = [spec.params["run"]["temp_root"] for spec in jobs.submitted[before:]]
            assert second_root not in sent, "the first card resubmitted the second turn's job"
            if "user_index" in inspect.signature(chat.runs.resubmit).parameters:
                assert sent == [first_root]  # its own turn's job
                assert view.live_job_card is not None and view.live_job_card.title_text.value == "first.epub"
                jobs.publish("CANCELLED")
                TC._finish_all(chat.runs)
            else:
                assert sent == [] and any(n[0] == cv.OLDER_RUN_RESUME for n in chat.calls.notes)
                assert view.live_job_card is None
                first = next(c for c in _turn_cards(view) if c.title_text.value == "first.epub")
                assert first.phase.name == "stopped" and "resume" in first.action_buttons
            view.render_transcript()
            second = next(c for c in _turn_cards(view) if c.title_text.value == "second.epub")
            before = len(jobs.submitted)
            second.action_buttons["resume"].on_click(None)
            assert view.live_job_card is not None and view.live_job_card.title_text.value == "second.epub"
            await RO._settle(chat.calls)
            assert len(jobs.submitted) == before + 1
            assert jobs.submitted[-1].params["run"]["temp_root"] == second_root
            jobs.publish("CANCELLED")
            TC._finish_all(chat.runs)
        finally:
            TC._close(chat)

    asyncio.run(scenario())


@needs_flet
def test_a_book_in_the_library_still_continues_in_its_translate_sheet(desktop_store_cls, isolated_env):
    """Retry failed / Resume of a turn whose workspace moved into the Library opens the Library translate sheet
    (its own progress file), whichever job is the chat's latest: the older-run check is the chat run's only.
    The card is the running card until the sheet is up, then the Result again (the Library runs the book)."""
    from glossarion_mobile.ui.chat import chat_view as cv

    tmp_path = isolated_env
    handed: list = []

    async def library_translate(folder, source="", **extra):
        handed.append((folder, extra))
        return "sheet"

    async def scenario():
        chat = TC._chat_view(desktop_store_cls, tmp_path, library_translate=library_translate)
        view = chat.view
        try:
            await RO._settle(chat.calls)
            item = TC._job_item(chat.adapter)
            moved = str(tmp_path / "Output" / "book")
            io = view.env.run_io

            async def state(fn, *args):
                return (moved, False) if fn == view._job_workspace_state else await io(fn, *args)

            view.env.run_io = state
            view._resume_turn_kwargs = lambda cid, it: None  # another turn's job is the chat's latest
            view.begin_resume(item)
            assert view.live_job_card is not None and view._resume_pending(item.index) is not None
            await RO._settle(chat.calls)
            assert handed == [(moved, {})] and not any(n[0] == cv.OLDER_RUN_RESUME for n in chat.calls.notes)
            assert view._resume_pending(item.index) is None and view.live_job_card is None
            assert chat.jobs.submitted == []
        finally:
            TC._close(chat)

    asyncio.run(scenario())


# ==========================================================================
# 8. The API call delay in the Run options
# ==========================================================================


def _delay_env(config: dict, tmp_path: Path) -> tuple:
    """SEND_INTERVAL_SECONDS / SEND_INTERVAL_MAX_SECONDS a translation job's owner exports for ``config`` (the
    shared HeadlessOwner + run_env, as the job adapter builds them)."""
    headless_owner = pytest.importorskip("headless_owner")
    run_env = pytest.importorskip("run_env")
    glossary_paths = pytest.importorskip("glossary_paths")
    import job_runner
    from glossarion_mobile.state.setting_writes import with_mobile_streaming_defaults

    original = glossary_paths.migrate_all_legacy_glossary_files
    glossary_paths.migrate_all_legacy_glossary_files = lambda *a, **k: None
    try:
        with job_runner.scoped_process_state(keywise=True, isolate_key_pools=True):
            owner = headless_owner.HeadlessOwner(with_mobile_streaming_defaults(json.loads(json.dumps(config))))
            env = run_env.build_translation_env(owner, str(tmp_path / "in" / "Book.epub"), "")
            return env["SEND_INTERVAL_SECONDS"], env["SEND_INTERVAL_MAX_SECONDS"]
    finally:
        glossary_paths.migrate_all_legacy_glossary_files = original


@needs_flet
def test_owner_8_run_options_have_the_api_call_delay(desktop_store_cls, isolated_env):
    """The owner's complaint: the Run options had no API call delay. The panel has the desktop's tile, next to the
    chunk size; "Only for this run" keeps the value in the run (the settings keep theirs), off it is the setting;
    the collapsed line shows it."""
    from glossarion_mobile.ui.chat.plan_card import PANEL_KEYS, RunOptionsPanel
    from glossarion_mobile.ui.chat.plan_model import RUN_OPTION_KEYS
    from glossarion_mobile.ui.settings.tiles import DelayRangeTile

    assert PANEL_KEYS.index("delay") == PANEL_KEYS.index("manual_chunk_size") + 1
    assert [k for k in PANEL_KEYS if k != "delay"] == list(RUN_OPTION_KEYS)
    store, schema = TC._config_store(isolated_env)
    ctx = RO._SettingsCtx(store, schema)
    try:
        panel = RunOptionsPanel(ctx)
        tile = panel.tiles["delay"]
        assert isinstance(tile, DelayRangeTile) and tile.label == "API call delay (s, or range 5-7)"
        assert tile.field.value == "3-7" and "Delay 3-7 s" in panel.summary_text.value  # a new user's range
        assert tile.apply("4")
        assert (store.get("delay"), store.get("delay_max")) == (4.0, "") and panel.config_overrides() == {}
        assert "Delay 4 s" in panel.summary_text.value
        panel.switch.value = True
        panel._on_switch()
        assert panel.tiles["delay"].apply("5-8")
        assert panel.config_overrides() == {"delay": 5.0, "delay_max": 8.0}
        assert (store.get("delay"), store.get("delay_max")) == (4.0, "")  # the setting keeps its value
        assert "Delay 5-8 s" in panel.summary_text.value
        # typed and Start tapped before the field lost focus: committed like the other numbers
        panel.tiles["delay"].field.value = "6"
        panel.commit()
        assert panel.config_overrides() == {"delay": 6.0, "delay_max": ""}
        panel.release()
    finally:
        store._saver.close()


@needs_flet
def test_plan_and_resume_run_with_the_cards_api_call_delay(desktop_store_cls, isolated_env):
    """Start (Plan card) and Resume (Result card) run with the delay their Run options show: the job's config and the
    SEND_INTERVAL_SECONDS / SEND_INTERVAL_MAX_SECONDS its owner exports; the batch plan's panel has it too."""
    tmp_path = isolated_env

    async def scenario():
        chat = TC._chat_view(desktop_store_cls, tmp_path)
        view, store = chat.view, chat.store
        try:
            await RO._settle(chat.calls)
            store.set_many({"delay": 2.0})
            book = tmp_path / "novel.epub"
            book.write_bytes(b"PK\x03\x04")
            assert view.attach_file(str(book))
            view.begin_send()
            panel = view._run_panel(chat.adapter.meta(view.cid)["pending_plan"])
            assert panel.tiles["delay"].field.value == "2"
            panel.switch.value = True
            panel._on_switch()
            assert panel.tiles["delay"].apply("5-8")
            view.start_plan()
            await RO._settle(chat.calls)
            config = RO._job_config(store, chat.jobs.submitted[-1])
            assert (config["delay"], config["delay_max"]) == (5.0, 8.0) and store.get("delay") == 2.0
            assert _delay_env(config, tmp_path) == ("5.0", "8.0")

            panel = await RO._stop(chat)  # the stopped card's Run options start from the job's values
            assert panel.tiles["delay"].field.value in ("2", "5-8")
            panel.switch.value = True
            panel._on_switch()
            assert panel.tiles["delay"].apply("1.5")
            config = await RO._resume(chat)
            assert (config["delay"], config["delay_max"]) == (1.5, "")
            assert _delay_env(config, tmp_path) == ("1.5", "")
            chat.jobs.publish("CANCELLED")
            TC._finish_all(chat.runs)

            batch = view._run_panel({"created": 1.0}, batch=True)
            assert "delay" in batch.tiles
            batch.release()
        finally:
            TC._close(chat)

    asyncio.run(scenario())
