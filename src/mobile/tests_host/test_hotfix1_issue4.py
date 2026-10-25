"""Acceptance test for hotfix 1, owner issue 4 (device report on the current APK):
"There is a bug where clicking resume doesn't update the response card, so the translation starts, but
the response card still says resume."

What the owner saw is reproduced end to end on the REAL app objects, as the other tests_host acceptance
tests do (``test_devfix_issue12._owner_flow``): the app started on a fake Flet session, the real JobService
on the app's ``UiDispatcher``, ``HeadlessOwner`` and the shared translation pipeline, the real
``ChatFeature`` / ``ChatRuns`` / ``ChatView`` and the desktop ``direct_text_store.ChatStore``. Only the model
is the offline fake OpenAI server on 127.0.0.1 (``diagnostics.fake_llm_server``); the E2E's network guard
refuses every non-loopback connection for the whole test. Every tap goes through the UI tree the client
would show (New chat, ＋ › Files, Send, Start, the running card's Stop, the Result card's Resume / Retry
failed, the drawer's chat rows, Android Home / back through ``app_lifecycle_state_change``).

Every step that needs a run to be in a given state waits on the fake model's own state: a reply hook
(``test_devfix_issue12.HoldGate``, or ``ChapterGate`` below) parks the model's requests, so the job cannot
end, and no step depends on the runner's speed. A sampler records, every 40 ms on the UI loop, what the
chat shows; from the Resume / Retry tap until the resumed run ends, NO sample may show a card that offers
Resume / Retry failed or has an ended look (the owner's "still says Resume").

1. ``test_resume_turns_the_result_card_into_the_running_card``: a 40-chapter book, ■ Stop mid-book (graceful,
   while the model holds the sent requests), the Result reads "Stopped · n/40 chapters" with Resume. Resume:
   at the tap the turn's one card is the live card (mounted, on screen, Running, ■ Stop on screen, no Resume),
   before its run even exists; once the resumed job translates (held) the same card object shows
   "Translating", "Chapter n/40" (n >= the stopped count, never ahead of the job) and Stop. Android Home and
   back while it translates: still the live card, no Resume. Exactly one job is submitted for the Resume.
   The end: "Done · 40/40 chapters".
2. ``test_resume_queued_behind_another_chat_job``: the same book stopped in chat A; chat B starts another
   book, whose job the model holds (RUNNING). Back in chat A (drawer), Resume: the card is the live card at
   once and reads "Queued · starts after <chat B's job>" with Cancel (no Resume) while chat B's job runs;
   also after Android Home and back. Chat B's job is let go: chat A's resumed job starts on the same card,
   Running with Stop and "Chapter n/40". The end: Done.
3. ``test_retry_failed_turns_the_result_card_into_the_running_card``: a 12-chapter book; the model refuses one
   chapter (HTTP 400, every attempt) in the first run, which ends "Finished with issues · 1 failed" with Retry
   failed. Retry failed: the live card at once (no Retry failed / Resume, Stop on screen); the retry run (held)
   shows Running, "Chapter n/12" and Stop; one job for the tap; the end: "Done · 12/12 chapters". The retry run
   sends only the failed chapter. Found here, outside issue 4 and the same on the base 01ae89d6: that retry run
   translated the whole book again, because a chat run that ended DONE with failed chapters had its run root
   removed (``run_controller._finish_run_root``), so the resubmitted run had no ``translation_progress.json``;
   such a run now keeps its run root, like a stopped one.
4. ``test_resume_after_a_relaunch_turns_the_card_into_the_running_card``: ■ Stop mid-book, Android Home, the app
   closed and started again on the same storage (a phone reclaims a backgrounded app): the Result card comes from
   JobService's history (no run in the new process) and reads "Stopped · n/40 chapters" with Resume. Resume: the
   live card at once, then Running with Stop and "Chapter n/40" (n >= the stopped count), one job, Done.

Real data stays untouched: the app runs in temp FLET_APP_STORAGE_* dirs (the bootstrap points HOME,
OUTPUT_DIRECTORY, GLOSSARION_LIBRARY_DIR, GLOSSARION_DATA_DIR and CONFIG_FILE there; asserted by
``_owner_flow``), USERPROFILE / APPDATA / LOCALAPPDATA point into tmp_path and HTTP logging is off.

Run from src/mobile with the mobile env (on the Windows dev box: ``.venv``, Python 3.12, which borrows the
3.12 user site, so keep that ``PYTHONPATH``)::

    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_hotfix1_issue4.py
"""

from __future__ import annotations

import asyncio
import importlib.util
import re
import sys
import threading
from pathlib import Path

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = MOBILE_DIR / "app"
SRC_DIR = MOBILE_DIR.parent
TESTS_DIR = MOBILE_DIR / "tests"
for _entry in (str(APP_DIR), str(SRC_DIR), str(TESTS_DIR)):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)


def _load(alias: str, file_name: str):
    spec = importlib.util.spec_from_file_location(alias, Path(__file__).with_name(file_name))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


#: The issue-12 acceptance harness (the owner's flow on the real app, the probe, the hold gate), loaded under
#: a private name: its ``Probe`` global is swapped for ``ResumeProbe`` below in this copy only.
I12 = _load("_glossarion_hotfix1_issue4_i12", "test_devfix_issue12.py")
pytestmark = I12.pytestmark
storage = I12.storage
app_env = I12.app_env
iso = I12.iso
offline = I12.offline

EPUB_NAME = I12.EPUB_NAME
RUN_TIMEOUT = I12.RUN_TIMEOUT
LEAVE = I12.LEAVE
COME_BACK = I12.COME_BACK
CHAPTER_RE = I12.CHAPTER_RE
_until = I12._until
_tap = I12._tap
_settled = I12._settled
_held_state = I12._held_state
_next_run = I12._next_run
_finished = I12._finished
_stop_sampling = I12._stop_sampling
HoldGate = I12.HoldGate

CHAPTERS = 40
OTHER_BOOK = "other_book.epub"
OTHER_CHAPTERS = 6
#: What a resumed run's card may look like while it lives (never an ended one, never Resume / Retry failed)
LIVE_PHASES = ("running", "queued", "stopping", "force_stopping")
RESUME_BUTTONS = ("resume", "retry")
RESUME_TEXTS = ("Resume", "Retry failed")
_HEADING = re.compile(r"제\s*(\d+)\s*장")


# ==========================================================================
# The probe: what the chat shows, with the card's buttons and the chat on screen
# ==========================================================================


class ResumeProbe(I12.Probe):
    """``test_devfix_issue12.Probe`` plus, per sample, the card's action buttons, the chat on screen and the run
    ``run_for`` returns (the stopped run until the resumed one exists)."""

    def card_state(self, card) -> dict:
        state = super().card_state(card)
        state["buttons"] = sorted(card.action_buttons)
        return state

    def sample(self, note: str = "") -> dict:
        entry = super().sample(note)
        run = self.run()
        entry["cid"] = self.view.cid
        entry["run_id"] = id(run) if run is not None else None
        return entry


I12.Probe = ResumeProbe  # ``_owner_flow`` builds its probe from this module's global


class ChapterGate:
    """The fake model's reply hook: ``hold()`` once a translation reply for a chapter >= ``chapter`` went out
    (armed by ``arm``). Only the 40-chapter book has such chapters, so another book's requests never trip it."""

    def __init__(self, server) -> None:
        self.server = server
        self._lock = threading.Lock()
        self.chapter = None
        server.on_response.append(self)

    def arm(self, chapter: int) -> None:
        with self._lock:
            self.chapter = int(chapter)

    def __call__(self, record) -> None:
        if record.kind != "translation":
            return
        numbers = {int(n) for n in _HEADING.findall(record.preview or "")}
        with self._lock:
            if self.chapter is None or len(numbers) != 1 or min(numbers) < self.chapter:
                return
            self.chapter = None
        self.server.hold()


def _refusing_server(translation_delay: float):
    """The fake model, answering every request for a chapter in ``refuse`` with HTTP 400 (an invalid-request
    error): the API client retries it and gives up, and the pipeline marks the chapter failed, so the run ends
    "Finished with issues"."""
    from glossarion_mobile.diagnostics import fake_llm_server as fls

    class RefusingServer(fls.FakeLLMServer):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.refuse: set = set()
            self.refused: list = []

        def _serve_chat(self, handler, payload, path) -> None:
            refuse = set(self.refuse)
            if refuse and self._classify(payload) == "translation":
                messages = [m for m in (payload.get("messages") or []) if isinstance(m, dict)]
                last_user = next((fls._content_text(m.get("content")) for m in reversed(messages)
                                  if m.get("role") == "user"), "")
                numbers = {int(n) for n in _HEADING.findall(last_user)}
                if len(numbers) == 1 and numbers <= refuse:
                    with self._lock:
                        self.refused.append(min(numbers))
                    handler._error(400, "The fake model refuses this chapter (test)")
                    return
            super()._serve_chat(handler, payload, path)

    server = RefusingServer()
    server.set_delay("translation", translation_delay)  # timing only: the holds decide when each step happens
    return server


# ==========================================================================
# Helpers
# ==========================================================================


def _chat_jobs(service, cid: str) -> list:
    """Ids of every job JobService knows for chat ``cid`` (active, queued, history), oldest first."""
    view = service.view()
    snaps = ([view.active] if view.active is not None else []) + list(view.queue) + list(view.history)
    ids = []
    for snap in snaps:
        origin = getattr(getattr(snap, "spec", None), "origin", None) or {}
        if str(origin.get("cid") or "") == str(cid) and snap.id not in ids:
            ids.append(snap.id)
    return sorted(ids, key=lambda job_id: getattr(service.snapshot(job_id), "created", 0) or 0)


def _card_buttons_text(probe, card) -> str:
    return f"{probe.card_state(card)} on screen: {sorted(t for t in probe.visible_texts() if len(t) < 40)}"


def _assert_live_look(probe, card, *, phases=("running",), note: str = "") -> dict:
    """``card`` is the chat's one JobCard for the turn and it is the running card: the live card, mounted and on
    screen, in a live phase, no Resume / Retry failed anywhere in the transcript, its Stop (running) or Cancel
    (queued) on screen. No await inside: the caller checks one moment."""
    cards = probe.job_cards()
    assert len(cards) == 1 and cards[0] is card, f"{note}: {[probe.card_state(c) for c in cards]}"
    state = probe.card_state(card)
    assert card is probe.view.live_job_card, f"{note}: not the live card: {state}"
    assert state["mounted"] and probe.on_screen(card.state_text), f"{note}: not on the client / screen: {state}"
    assert state["phase"] in phases, f"{note}: {state}"
    assert state["title"] == EPUB_NAME, f"{note}: {state}"
    assert not set(RESUME_BUTTONS) & set(card.action_buttons), f"{note}: the card still offers {state['buttons']}"
    texts = probe.visible_texts()
    assert not set(RESUME_TEXTS) & texts, f"{note}: {_card_buttons_text(probe, card)}"
    wanted = "stop" if state["phase"] == "running" else ("cancel_queued" if state["phase"] == "queued" else "force_stop")
    button = card.action_buttons.get(wanted)
    assert button is not None and probe.on_screen(button), f"{note}: no {wanted} on screen: {state}"
    return state


def _progress_shown(card, floor: int, total: int) -> bool:
    """``card`` shows "Chapter n/total" with n >= floor."""
    if card is None:
        return False
    match = CHAPTER_RE.search(str(card.line_text.value or ""))
    return bool(match and card.line_text.visible and int(match.group(1)) >= floor and int(match.group(2)) == total)


def _assert_translating(probe, card, *, floor: int, total: int, note: str) -> dict:
    """One moment of the resumed run while the model holds it: the job RUNNING, the same live card Running
    ("Translating"), Stop on screen, "Chapter n/total" on screen with floor <= n <= the job's count."""
    sample = probe.sample(note)
    assert sample["job"]["state"] == "RUNNING", f"{note}: {sample['job']}"
    state = _assert_live_look(probe, card, phases=("running",), note=note)
    assert state["state"] == "Translating", f"{note}: {state}"
    assert state["chapter"] is not None and state["chapter"][1] == total, f"{note}: {state}"
    assert floor <= state["chapter"][0] <= sample["job"]["completed"], f"{note}: {state} job {sample['job']}"
    texts = probe.visible_texts()
    assert state["line"] in texts and "Stop" in texts and "Translating" in texts, f"{note}: {_card_buttons_text(probe, card)}"
    return state


def _assert_samples_live_since(probe, start: int, run_ended, cid: str) -> int:
    """Every sample from index ``start`` (``len(probe.samples)`` taken right before the Resume / Retry tap: the
    sampler and the tap share the UI loop, so the index is exact where a ~16 ms Windows clock tick is not) until
    the resumed run ``run_ended`` ended, while chat ``cid`` was on screen: one card for the turn, the live card, a
    live phase, never Resume / Retry failed, never an ended look ("Stopped" / "Done" / "Finished with issues").
    Returns how many samples were checked."""
    bad = []
    checked = 0
    for s in probe.samples[start:]:
        if s["cid"] != cid:
            continue
        if s["run_id"] == id(run_ended) and not s["run_live"]:
            continue  # the resumed run has ended: its Result
        checked += 1
        cards = s["cards"]
        if len(cards) != 1:
            bad.append(("cards", s))
            continue
        c = cards[0]
        if not c["live"] or not c["mounted"]:
            bad.append(("not the live card", s))
        elif c["phase"] not in LIVE_PHASES:
            bad.append(("an ended look", s))
        elif set(RESUME_BUTTONS) & set(c["buttons"]):
            bad.append(("still offers Resume / Retry failed", s))
        elif c["state"].startswith(("Stopped", "Done", "Finished with issues", "Failed", "Interrupted")):
            bad.append(("an ended state line", s))
        elif c["title"] != EPUB_NAME:
            bad.append(("title", s))
    assert not bad, "\n".join(f"{why}: {s}" for why, s in bad[:5])
    return checked


async def _stop_mid_book(ctx) -> dict:
    """■ Stop on the running card while the model holds the requests the run already sent (``HoldGate``): a
    graceful Stop waits for them, they answer, the run ends Stopped. Returns the turn's Result card, the
    stopped chapter count and the committed request labels."""
    app, run, probe, server = ctx.app, ctx.run, ctx.probe, ctx.server
    view = app.chat_view
    assert await _until(lambda: _settled(server), RUN_TIMEOUT), _held_state(probe, server)
    await _tap(probe, view.live_job_card.action_buttons["stop"])
    assert await _until(lambda: run.stop_requested, 5)
    assert await _until(lambda: _settled(server) and probe.job()["state"] == "STOPPING", 10), \
        (probe.job(), _held_state(probe, server))
    server.release()
    await _finished(app, run)
    assert run.state == "stopped", run.state
    job = probe.job()
    assert job["state"] == "CANCELLED" and 0 < job["completed"] <= CHAPTERS - 20, job
    cards = probe.job_cards()
    assert len(cards) == 1, [probe.card_state(c) for c in cards]
    ended = cards[0]
    state = probe.card_state(ended)
    assert state["phase"] == "stopped" and state["state"] == f"Stopped · {job['completed']}/{CHAPTERS} chapters", state
    assert "resume" in ended.action_buttons and probe.on_screen(ended.action_buttons["resume"]), state
    assert ended is not view.live_job_card
    return {"card": ended, "completed": job["completed"], "committed": probe.committed_labels(),
            "jobs": _chat_jobs(app.job_service, probe.cid)}


async def _open_chat(driver, app, cid: str) -> None:
    """The drawer's chat row (what the owner taps to go back to a chat)."""
    import flows

    await flows.open_drawer(driver)
    await driver.tap(key=f"chat-{cid}", timeout=15)
    assert await _until(lambda: app.chat_view.cid == cid, 15), app.chat_view.cid
    await flows.go_home(driver)


async def _start_other_chat(ctx, picks: Path) -> tuple:
    """New chat › ＋ › Files (another book) › Send › Start, as the owner taps them; returns (cid, its run)."""
    import flows
    from glossarion_mobile.diagnostics import fixtures

    driver, app = ctx.driver, ctx.app
    book = fixtures.build_tiny_epub(picks / OTHER_BOOK, chapters=OTHER_CHAPTERS)
    driver.picker.files[OTHER_BOOK] = book
    await flows.go_home(driver)
    await driver.tap(tooltip="New chat")
    assert await _until(lambda: app.chat_view.cid != ctx.probe.cid, 15)
    await driver.tap(tooltip=flows.ATTACH_TOOLTIP)
    await driver.pick_file(OTHER_BOOK, lambda: driver.tap(key="attach-files"))
    await driver.wait(contains=book.stem, timeout=60)
    await driver.tap(key="send-idle_ready", timeout=60)
    await driver.wait(text="Ready to translate", timeout=60)
    await driver.tap(text="Start")
    cid = app.chat_view.cid
    runs = app.chat_feature.runs
    assert await _until(lambda: getattr(runs.run_for(cid), "job_id", None) is not None, 60), "chat B's Start sent nothing"
    return cid, runs.run_for(cid)


# ==========================================================================
# 1. Resume: the running card at the tap, through the run, after Home and back
# ==========================================================================


def test_resume_turns_the_result_card_into_the_running_card(iso, offline):
    seen: dict = {}

    async def body(ctx):
        app, run, probe, server = ctx.app, ctx.run, ctx.probe, ctx.server
        view, service = app.chat_view, app.job_service
        stopped = await _stop_mid_book(ctx)
        ended, floor = stopped["card"], stopped["completed"]

        # ---- Resume on the Result card ----------------------------------------------------------------------
        gate.hold_after(6)  # the model holds again after 6 replies of the resumed run
        tap_i = len(probe.samples)  # the samples from here on are after the tap
        await _tap(probe, ended.action_buttons["resume"])
        # at the tap (no await since): the running card, before the resumed run exists
        card = view.live_job_card
        _assert_live_look(probe, card, note="at the Resume tap")
        assert ended not in probe.job_cards()
        at_tap = probe.sample("at the Resume tap")
        seen["at_tap"] = at_tap["cards"][0]

        assert await _until(lambda: _next_run(probe, run) is not None, 30), probe.describe()
        resumed = _next_run(probe, run)
        try:
            assert await _until(lambda: _settled(server), RUN_TIMEOUT), _held_state(probe, server)
            # the job's progress reaches the card without any tap (its snapshots / the stream ticks)
            assert await _until(lambda: _progress_shown(view.live_job_card, floor, CHAPTERS), 30), probe.describe()
            assert view.live_job_card is card, "the resumed run's live card is not the card Resume turned live"
            _assert_translating(probe, card, floor=floor, total=CHAPTERS, note="resumed run (held)")

            # ---- Android Home while it translates, then back ----------------------------------------------
            await probe.lifecycle(*LEAVE)
            assert probe.page.app_visible is False
            hidden = probe.sample("hidden")
            assert len(hidden["cards"]) == 1 and hidden["cards"][0]["live"], hidden
            assert not set(RESUME_BUTTONS) & set(hidden["cards"][0]["buttons"]), hidden
            await probe.lifecycle(*COME_BACK)
            assert probe.page.app_visible is not False
            _assert_translating(probe, view.live_job_card, floor=floor, total=CHAPTERS, note="back in the app")
            await asyncio.sleep(1.0)  # the stream loop's catch-up tick (the model still holds: nothing can end)
            _assert_translating(probe, view.live_job_card, floor=floor, total=CHAPTERS, note="after the catch-up")
            seen["back"] = probe.card_state(view.live_job_card)
        finally:
            server.release()

        # ---- the end -----------------------------------------------------------------------------------------
        await _finished(app, resumed)
        await _stop_sampling(ctx)
        assert resumed.state == "done" and resumed.error is None, (resumed.state, resumed.error)
        seen["checked"] = _assert_samples_live_since(probe, tap_i, resumed, probe.cid)
        # one tap, one job: exactly one job was submitted for the Resume (the same run root)
        jobs = _chat_jobs(service, probe.cid)
        assert jobs[:len(stopped["jobs"])] == stopped["jobs"] and len(jobs) == len(stopped["jobs"]) + 1, \
            (stopped["jobs"], jobs)
        I12._check_result(probe, stopped["committed"], "done", f"Done · {CHAPTERS}/{CHAPTERS} chapters")

    with I12._server(0.08) as server:  # timing only: the holds decide when each step happens
        gate = HoldGate(server, 9)  # the EPUB metadata request + 8 chapters, then ■ Stop
        asyncio.run(I12._owner_flow(iso, server, body, chapters=CHAPTERS, glossary="off"))
    assert seen.get("checked"), f"the scenario did not reach the resumed run's end: {seen}"
    I12._assert_offline(offline)


# ==========================================================================
# 2. Resume behind another chat's job: Queued on the same card, then Running
# ==========================================================================


def test_resume_queued_behind_another_chat_job(iso, offline):
    seen: dict = {}

    async def body(ctx):
        app, run, probe, server, driver = ctx.app, ctx.run, ctx.probe, ctx.server, ctx.driver
        view, service = app.chat_view, app.job_service
        cid = probe.cid
        stopped = await _stop_mid_book(ctx)
        floor = stopped["completed"]

        # ---- chat B: another book, its job held by the model (RUNNING) -----------------------------------------
        gate.hold_after(2)  # chat B's metadata request + its first chapter, then the model holds
        other_cid, other = await _start_other_chat(ctx, iso.picks)
        assert await _until(lambda: _settled(server), RUN_TIMEOUT), _held_state(probe, server)
        other_job = service.snapshot(other.job_id)
        assert other_job.state.value == "RUNNING", other_job.state

        # ---- back in chat A (the drawer), Resume ---------------------------------------------------------------
        await _open_chat(driver, app, cid)
        cards = probe.job_cards()
        assert len(cards) == 1 and cards[0].phase.name == "stopped" and "resume" in cards[0].action_buttons, \
            [probe.card_state(c) for c in cards]
        ended = cards[0]
        chapters.arm(20)  # chat A's resumed run: the model holds once chapter 20 answered (chat B has 6)
        tap_i = len(probe.samples)  # the samples from here on are after the tap
        await _tap(probe, ended.action_buttons["resume"])
        card = view.live_job_card
        _assert_live_look(probe, card, phases=("running", "queued"), note="at the Resume tap")

        assert await _until(lambda: _next_run(probe, run) is not None and _next_run(probe, run).job_id is not None, 30), \
            probe.describe()
        resumed = _next_run(probe, run)
        queued_line = f"Queued · starts after {other_job.spec.title}"
        try:
            # while chat B's job runs (held), chat A's resumed job waits in the queue, and its card says so
            assert service.snapshot(resumed.job_id).state.value == "QUEUED", service.snapshot(resumed.job_id).state
            assert await _until(lambda: probe.card_state(view.live_job_card)["state"] == queued_line, 10), \
                probe.card_state(view.live_job_card)
            state = _assert_live_look(probe, view.live_job_card, phases=("queued",), note="queued")
            assert view.live_job_card is card and queued_line in probe.visible_texts(), state
            seen["queued"] = state

            # ---- Android Home while it waits, then back ---------------------------------------------------------
            await probe.lifecycle(*LEAVE)
            await probe.lifecycle(*COME_BACK)
            state = _assert_live_look(probe, view.live_job_card, phases=("queued",), note="queued, back in the app")
            assert state["state"] == queued_line and service.snapshot(other.job_id).state.value == "RUNNING", state

            # ---- chat B's job is let go: chat A's resumed job starts on the same card --------------------------
            server.release()
            assert await _until(lambda: _settled(server) and service.snapshot(resumed.job_id).state.value == "RUNNING",
                                RUN_TIMEOUT), _held_state(probe, server)
            assert service.snapshot(other.job_id).is_terminal, service.snapshot(other.job_id).state
            assert await _until(lambda: _progress_shown(view.live_job_card, floor, CHAPTERS), 30), probe.describe()
            assert view.live_job_card is card, "the running card is not the card Resume turned live"
            seen["running"] = _assert_translating(probe, card, floor=floor, total=CHAPTERS, note="resumed run (held)")
        finally:
            server.release()

        await _finished(app, resumed)
        await _stop_sampling(ctx)
        assert resumed.state == "done" and resumed.error is None, (resumed.state, resumed.error)
        seen["checked"] = _assert_samples_live_since(probe, tap_i, resumed, cid)
        jobs = _chat_jobs(service, cid)
        assert len(jobs) == len(stopped["jobs"]) + 1, (stopped["jobs"], jobs)
        I12._check_result(probe, stopped["committed"], "done", f"Done · {CHAPTERS}/{CHAPTERS} chapters")
        other_run = app.chat_feature.runs.run_for(other_cid)
        assert await _until(lambda: not other_run.live, 60) and other_run.state == "done", other_run.state

    with I12._server(0.08) as server:
        gate = HoldGate(server, 9)
        chapters = ChapterGate(server)
        asyncio.run(I12._owner_flow(iso, server, body, chapters=CHAPTERS, glossary="off"))
    assert seen.get("checked"), f"the scenario did not reach the resumed run's end: {seen}"
    I12._assert_offline(offline)


# ==========================================================================
# 3. Retry failed: the running card at the tap
# ==========================================================================


def test_retry_failed_turns_the_result_card_into_the_running_card(iso, offline):
    total = 12
    refuse = {5}
    seen: dict = {}

    async def body(ctx):
        app, run, probe, server = ctx.app, ctx.run, ctx.probe, ctx.server
        view, service = app.chat_view, app.job_service
        await _finished(app, run)
        assert run.state == "done", (run.state, run.error)
        job = probe.job()
        assert job["state"] == "DONE" and job["total"] == total, job
        cards = probe.job_cards()
        assert len(cards) == 1, [probe.card_state(c) for c in cards]
        ended = cards[0]
        state = probe.card_state(ended)
        assert state["phase"] == "done" and state["state"] == f"Finished with issues · {len(refuse)} failed", \
            (state, sorted(set(server.refused)))
        assert "retry" in ended.action_buttons and probe.on_screen(ended.action_buttons["retry"]), state
        committed = probe.committed_labels()
        before = _chat_jobs(service, probe.cid)

        # ---- Retry failed (the model answers every chapter now and holds the retry run's requests: it sends only the
        # failed chapter, so the hold starts at the tap, not after a reply) -----------------------------------------
        server.refuse.clear()
        mark = server.mark()
        server.hold()
        tap_i = len(probe.samples)  # the samples from here on are after the tap
        await _tap(probe, ended.action_buttons["retry"])
        card = view.live_job_card
        _assert_live_look(probe, card, note="at the Retry failed tap")
        assert ended not in probe.job_cards()

        assert await _until(lambda: _next_run(probe, run) is not None, 30), probe.describe()
        retried = _next_run(probe, run)
        try:
            assert await _until(lambda: _settled(server), RUN_TIMEOUT), _held_state(probe, server)
            assert await _until(lambda: _progress_shown(view.live_job_card, 0, total), 30), probe.describe()
            assert view.live_job_card is card, "the retry run's live card is not the card Retry failed turned live"
            seen["running"] = _assert_translating(probe, card, floor=0, total=total, note="retry run (held)")
        finally:
            server.release()

        await _finished(app, retried)
        await _stop_sampling(ctx)
        assert retried.state == "done" and retried.error is None, (retried.state, retried.error)
        seen["checked"] = _assert_samples_live_since(probe, tap_i, retried, probe.cid)
        jobs = _chat_jobs(service, probe.cid)
        assert len(jobs) == len(before) + 1, (before, jobs)  # one tap, one job
        I12._check_result(probe, committed, "done", f"Done · {total}/{total} chapters")
        seen["resent"] = _chapter_requests(server.records("translation", since=mark))

    with _refusing_server(0.03) as server:
        server.refuse.update(refuse)
        gate = HoldGate(server, None)
        asyncio.run(I12._owner_flow(iso, server, body, chapters=total, glossary="off"))
    assert seen.get("checked"), f"the scenario did not reach the retry run's end: {seen}"
    I12._assert_offline(offline)
    # Found by this test, outside issue 4 (the card), the same on the base 01ae89d6: a chat run that ended DONE with
    # failed chapters lost its run root (``run_controller._finish_run_root``), so Retry failed resubmitted a run root
    # with no translation_progress.json and the whole book was translated again. Such a run now keeps its run root.
    assert seen["resent"] == sorted(refuse), \
        f"Retry failed sent chapters {seen['resent']} again, not only the failed {sorted(refuse)}"


def _chapter_requests(records) -> list:
    """The chapter of each one-chapter translation request in ``records`` (sorted, repeats kept)."""
    out = []
    for record in records:
        numbers = {int(n) for n in _HEADING.findall(record.preview or "")}
        if len(numbers) == 1:
            out.extend(numbers)
    return sorted(out)


# ==========================================================================
# 4. Resume after the app was killed and started again
# ==========================================================================


async def _relaunch(iso, server, cid: str, body) -> None:
    """The app started again on the same storage (Android reclaims a backgrounded app; the first launch flushed
    what its lifecycle flushes and closed), chat ``cid`` opened from the drawer; ``body(ctx)`` then drives it
    while a sampler runs, as ``test_devfix_issue12._owner_flow`` does for the first launch."""
    import types

    import flows

    tf = I12._UF._foundations()
    app, tester, driver = await I12._UF._host_driver(tf, {EPUB_NAME: iso.picks / EPUB_NAME})
    app.job_service.watcher_interval = 0.2  # as _owner_flow: the card's "Chapter n/m" follows the job closely
    stop = asyncio.Event()
    sampler = None
    try:
        await flows.wait_home(driver)
        I12._assert_isolated(iso.tmp)
        await _open_chat(driver, app, cid)
        probe = ResumeProbe(app, tester, cid)
        sampler = asyncio.ensure_future(probe.sampler(stop))
        await body(types.SimpleNamespace(app=app, tester=tester, driver=driver, server=server, run=None, probe=probe,
                                         stop_sampler=stop, sampler=sampler))
    finally:
        stop.set()
        if sampler is not None:
            try:
                await asyncio.wait_for(sampler, 5)
            except Exception:
                sampler.cancel()
        try:
            if getattr(tester.page, "app_visible", True) is False:
                for state in COME_BACK:
                    await tester.session.dispatch_event(tester.page._i, "app_lifecycle_state_change", {"state": state})
            await I12._end_job(app, server, app.chat_feature.runs.run_for(cid))
        finally:
            try:
                app.jobs.close()
            finally:
                await tf._stop(app)


def test_resume_after_a_relaunch_turns_the_card_into_the_running_card(iso, offline):
    """■ Stop mid-book, Android Home (the app flushes the chat history, settings and prefs), the app is closed
    and started again: the chat's Result card (from JobService's history, no run in this process) reads
    "Stopped · n/40 chapters" with Resume. Resume: the live card at once, then Running with Stop and
    "Chapter n/40" (n >= the stopped count: the run continues from its progress), one job for the tap, Done."""
    first: dict = {}
    seen: dict = {}

    async def stop_and_leave(ctx):
        app, probe = ctx.app, ctx.probe
        first.update(await _stop_mid_book(ctx))
        first["cid"] = probe.cid
        await probe.lifecycle(*LEAVE)  # Home: the chat history, config.json and prefs are flushed
        for name in ("config_store", "prefs"):
            target = getattr(app, name, None)
            if target is not None:
                target.flush()

    async def resume_after_relaunch(ctx):
        app, probe, server = ctx.app, ctx.probe, ctx.server
        view, service = app.chat_view, app.job_service
        cid, floor = first["cid"], first["completed"]
        assert app.chat_feature.runs.run_for(cid) is None, "the first launch's run is in this process"
        cards = probe.job_cards()
        assert len(cards) == 1, [probe.card_state(c) for c in cards]
        ended = cards[0]
        state = probe.card_state(ended)
        assert state["phase"] == "stopped" and state["state"] == f"Stopped · {floor}/{CHAPTERS} chapters", state
        assert "resume" in ended.action_buttons and probe.on_screen(ended.action_buttons["resume"]), state
        before = _chat_jobs(service, cid)
        assert before == first["jobs"], (before, first["jobs"])

        gate.hold_after(6)
        tap_i = len(probe.samples)  # the samples from here on are after the tap
        await _tap(probe, ended.action_buttons["resume"])
        card = view.live_job_card
        _assert_live_look(probe, card, note="at the Resume tap after a relaunch")
        assert ended not in probe.job_cards()

        assert await _until(lambda: _next_run(probe, None) is not None, 30), probe.describe()
        resumed = _next_run(probe, None)
        try:
            assert await _until(lambda: _settled(server), RUN_TIMEOUT), _held_state(probe, server)
            assert await _until(lambda: _progress_shown(view.live_job_card, floor, CHAPTERS), 30), probe.describe()
            assert view.live_job_card is card, "the resumed run's live card is not the card Resume turned live"
            seen["running"] = _assert_translating(probe, card, floor=floor, total=CHAPTERS,
                                                  note="resumed run after a relaunch (held)")
        finally:
            server.release()

        await _finished(app, resumed)
        await _stop_sampling(ctx)
        assert resumed.state == "done" and resumed.error is None, (resumed.state, resumed.error)
        seen["checked"] = _assert_samples_live_since(probe, tap_i, resumed, cid)
        jobs = _chat_jobs(service, cid)
        assert jobs[:len(before)] == before and len(jobs) == len(before) + 1, (before, jobs)
        I12._check_result(probe, first["committed"], "done", f"Done · {CHAPTERS}/{CHAPTERS} chapters")

    with I12._server(0.08) as server:
        gate = HoldGate(server, 9)
        asyncio.run(I12._owner_flow(iso, server, stop_and_leave, chapters=CHAPTERS, glossary="off"))
        assert first.get("cid"), "the first launch did not stop the run"
        asyncio.run(_relaunch(iso, server, first["cid"], resume_after_relaunch))
    assert seen.get("checked"), f"the scenario did not reach the resumed run's end: {seen}"
    I12._assert_offline(offline)
