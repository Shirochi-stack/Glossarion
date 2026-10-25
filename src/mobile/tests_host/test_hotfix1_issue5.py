"""Acceptance test for owner issue 5 of hotfix 1 (device report on the current APK): "Mobile absolutely has no
Job threading safety, so like I mentioned earlier, you can send multiple separate jobs assigned to the same
attachment, which is wrong. They should reuse the same job, so we don't sent the same request twice by
accident."

This file proves, end to end on the REAL app objects, that one file gets one job: the app started on a fake
Flet session (``test_ui_flows._host_driver`` through ``test_devfix_issue12``'s harness), the real JobService on
the app's UiDispatcher, ``HeadlessOwner`` and the shared translation pipeline (``TransateKRtoEN``, the Balanced
glossary pre-pass with the Direct Text approval question), the real ``ChatFeature`` / ``ChatRuns`` /
``ChatView``, ``LibraryService`` and the Book page's Chapters tab (the Progress Manager). Only the model is the
offline fake OpenAI server on 127.0.0.1; the E2E network guard refuses every non-loopback connection.

The owner's double / triple taps are click events that all reach the app before its first handler ran (the
client sends them before the patch disabling the control arrives), dispatched concurrently through the Flet
session; the other surfaces (Library "Run translation", the Jobs page and notification Resume, the glossary
notification's answer, another chat's send) call the same JobService API from real threads at the same moment
(a barrier). Every ``JobService.submit_job`` / ``resume_job`` call the app makes, from any surface and thread, is
recorded with its answer.

1. ``test_chat_taps_threads_and_a_second_chat_share_the_attachments_one_job`` (Balanced glossary): Send x3,
   Start x3 on the Plan card; at the glossary gate the Library's "Run translation" of the same file, the very
   same request again, the Jobs page Resume and another chat's attachment send, from threads; a second chat
   with the same file (Send, Start x2); the approval card's Yes x3 together with the notification's answer
   from threads; while the translation runs (the fake model holds) the same thread burst again. One job for
   the whole run, every call answers its id, every chapter request and the glossary pass reach the model once.
   The second chat is pointed at the job ("… is already running (Chat · …)" with Open: its page), whatever it
   offers next queues nothing, and its turn is not shown as a run of its own that ended: it keeps its Plan card
   (Start again once the book's job ended, as "Save to: Library" does) or follows the job - never an
   "Interrupted · no output yet" Result with Resume / Retry failed (checked last).
2. ``test_stop_then_resume_from_every_surface_resumes_once`` (glossary off): Stop mid-book (graceful: the held
   requests answer and are committed), then the Result card's Resume x3 together with the Jobs page /
   notification Resume of the stopped job from threads: one resumed job, every call answers its id; while it
   runs, more Resumes (of the stopped and of the resumed job) and the Library's run of the file reuse it. The
   model sees every chapter once over both runs.
3. ``test_library_run_translation_progress_manager_qa_and_a_chat_share_one_job`` (a Library book): Book page ›
   Chapters › Run translation x3 opens one sheet, its Start x3 queues one job; while it runs, Run translation /
   Run QA scan taps that still reach the app queue nothing, a chat with the same file ("Save to: Library",
   Start x2) is pointed at the book's job, and the Library / chat / Resume calls from threads reuse it; after
   it, Run QA scan x3 together with the same scan from threads queues one QA scan. Every chapter reaches the
   model once and the scan sends nothing.

Holds and waits follow the fake model's own state (``HoldGate``, ``_settled``), never sleeps. Real data stays
untouched: the app runs on temp FLET_APP_STORAGE_* dirs (HOME, OUTPUT_DIRECTORY, GLOSSARION_LIBRARY_DIR,
GLOSSARION_DATA_DIR and CONFIG_FILE there; asserted), USERPROFILE / APPDATA / LOCALAPPDATA point into tmp_path
and HTTP logging is off. A failing scenario force-stops its jobs before the app closes.

Run from src/mobile with the mobile env (on the dev box ``.venv``, Python 3.12 with the shell PYTHONPATH)::

    .venv/Scripts/python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic \\
        tests_host/test_hotfix1_issue5.py
"""

from __future__ import annotations

import asyncio
import collections
import importlib.util
import os
import re
import sys
import threading
import time
import types
from pathlib import Path

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
for entry in (str(MOBILE_DIR / "app"), str(MOBILE_DIR.parent), str(MOBILE_DIR / "tests")):
    if entry not in sys.path:
        sys.path.insert(0, entry)


def _load(alias: str, file_name: str):
    spec = importlib.util.spec_from_file_location(alias, Path(__file__).with_name(file_name))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


I12 = _load("_glossarion_devfix12_hotfix1_issue5", "test_devfix_issue12.py")  # the real-pipeline harness
pytestmark = I12.pytestmark
storage = I12.storage
app_env = I12.app_env
iso = I12.iso
offline = I12.offline

RUN_TIMEOUT = I12.RUN_TIMEOUT
THREAD_TIMEOUT = 60.0
#: timing only: the holds decide when each step happens
TRANSLATION_DELAY = 0.05
HEADING = re.compile(r"제\s*(\d+)\s*장")
ALREADY = ("already running", "already queued", "still stopping")


# ==========================================================================
# Instruments
# ==========================================================================


class Submissions:
    """Every ``JobService.submit_job`` / ``resume_job`` call of the app - Send / Start / Resume / the Library /
    the Progress Manager / the Jobs page, any thread - with its answer, and every job that was ever queued
    (its first transition)."""

    def __init__(self, service) -> None:
        self.service = service
        self.lock = threading.Lock()
        self.calls: list = []
        self.created: list = []
        submit_job, resume_job = service.submit_job, service.resume_job

        def recording_submit(spec, *, front=False):
            result = submit_job(spec, front=front)
            self._note("submit", str(getattr(spec, "kind", "")), result)
            return result

        def recording_resume(job_id, params=None):
            result = resume_job(job_id, params)
            self._note(f"resume:{job_id}", "", result)
            return result

        service.submit_job = recording_submit  # ``submit`` / ``resume`` call these through ``self``
        service.resume_job = recording_resume
        self._unsubscribe = service.on_transition(self._on_transition)

    def _note(self, what: str, kind: str, result) -> None:
        with self.lock:
            self.calls.append(types.SimpleNamespace(
                what=what, kind=kind, job_id=getattr(result, "job_id", None), reused=getattr(result, "reused", None),
                thread=threading.current_thread().name))

    def _on_transition(self, snap, previous) -> None:
        with self.lock:
            if previous is None and snap.id not in self.created:
                self.created.append(snap.id)

    def since(self, mark: int) -> list:
        with self.lock:
            return list(self.calls[mark:])

    def mark(self) -> int:
        with self.lock:
            return len(self.calls)

    def jobs(self) -> list:
        """Every job JobService knows (running, queued, interrupted, ended), oldest first."""
        view = self.service.view()
        snaps = ([view.active] if view.active is not None else []) + list(view.queue) + list(view.interrupted) \
            + list(view.history)
        return sorted({s.id: s for s in snaps}.values(), key=lambda s: s.created)

    def close(self) -> None:
        self._unsubscribe()
        for name in ("submit_job", "resume_job"):
            self.service.__dict__.pop(name, None)


def _describe_calls(calls) -> str:
    return "\n".join(f"{c.what} {c.kind} -> {c.job_id} reused={c.reused} [{c.thread}]" for c in calls)


@pytest.fixture
def snacks(iso, monkeypatch):
    """Every snackbar the app shows (``GlossarionApp.notify``: the chat, the Library, the Book page, Jobs)."""
    import glossarion_mobile.app as app_module

    original = app_module.show_snackbar
    seen: list = []

    def record(page, message, *, action_label=None, on_action=None, **kwargs):
        seen.append(types.SimpleNamespace(message=str(message), label=action_label, action=on_action))
        return original(page, message, action_label=action_label, on_action=on_action, **kwargs)

    monkeypatch.setattr(app_module, "show_snackbar", record)
    return seen


def _server():
    """The fake model; its glossary answers include the term every chapter of the test book has."""
    from glossarion_mobile.diagnostics.fake_llm_server import FakeLLMServer, GlossaryEntry

    server = FakeLLMServer()
    server.glossary = server.glossary + (GlossaryEntry("term", "마왕성", "Demon King's Castle", "", "The demon's keep"),)
    server.set_delay("translation", TRANSLATION_DELAY)
    return server


def _chapter_requests(server) -> collections.Counter:
    """How many requests the model got for each chapter (a chapter request names one "제N장" heading), whatever
    their outcome: an aborted request sent again counts twice."""
    counts: collections.Counter = collections.Counter()
    for record in server.records("translation"):
        found = {int(n) for n in HEADING.findall(record.preview or "")}
        if len(found) == 1:
            counts[found.pop()] += 1
    return counts


def _duplicate_requests(server, kinds=("glossary", "translation")) -> list:
    """Requests whose prompt the model got more than once (kind, preview, prompt length), with their count."""
    counts = collections.Counter((r.kind, r.preview, r.prompt_chars) for r in server.records() if r.kind in kinds)
    return [(key, n) for key, n in counts.items() if n > 1]


def _assert_each_chapter_once(server, chapters: int) -> None:
    counts = _chapter_requests(server)
    twice = {c: n for c, n in counts.items() if n != 1}
    assert not twice, f"chapters the model got more than once: {twice}"
    assert sorted(counts) == list(range(1, chapters + 1)), sorted(set(range(1, chapters + 1)) - set(counts))


# ==========================================================================
# Taps and threads
# ==========================================================================


def _diagnose(ctx, snacks) -> None:
    """What the scenario saw (printed; ``pytest -s`` shows it): the surface calls, the jobs, the snackbars and the
    model's requests."""
    lines = [f"jobs: {[(s.id, s.spec.kind, s.state.value) for s in ctx.subs.jobs()]}",
             f"calls ({len(ctx.subs.calls)}):", _describe_calls(ctx.subs.calls),
             f"snackbars: {[(s.message, s.label) for s in snacks]}",
             f"requests: {collections.Counter(r.kind for r in ctx.server.records())}",
             f"chapter requests: {dict(sorted(_chapter_requests(ctx.server).items()))}",
             "other translation requests: " + repr([r.preview[:60] for r in ctx.server.records("translation")
                                                   if len(set(HEADING.findall(r.preview or ""))) != 1])]
    print("\n".join(lines))


def _clickable(control):
    node = control
    while node is not None:
        if getattr(node, "on_click", None) is not None:
            return node
        node = getattr(node, "parent", None)
    raise AssertionError(f"{type(control).__name__} has no click handler")


#: the click handlers ``_burst`` started (each event in its own task): checked to end, cancelled at teardown
TAPS: list = []


async def _burst(tester, control, taps: int = 3) -> list:
    """``taps`` taps on ``control`` that all reach the app before its first handler ran: concurrent click events
    through the Flet session, each in its own task as the server dispatches them, whether the control is disabled
    by then or not (the client sent them before that patch arrived). A tap does not wait for its handler (an async
    one may wait for the user, e.g. the "Keep translations running" sheet): the handlers' tasks are returned (and
    kept in ``TAPS``) once they have started."""
    target = _clickable(control)
    tasks = [asyncio.ensure_future(tester.session.dispatch_event(target._i, "click", None)) for _ in range(taps)]
    TAPS.extend(tasks)
    await asyncio.sleep(0.05)
    return tasks


def _handlers_ended() -> bool:
    return all(task.done() for task in TAPS)


def _tap_errors() -> list:
    return [repr(task.exception()) for task in TAPS if task.done() and not task.cancelled() and task.exception()]


def _hammer(calls) -> list:
    """Each callable on its own thread, released together (a barrier); their results (or exceptions)."""
    barrier = threading.Barrier(len(calls))
    out: list = [None] * len(calls)

    def run(index, fn):
        try:
            barrier.wait(THREAD_TIMEOUT)
            out[index] = fn()
        except BaseException as exc:  # noqa: BLE001 - the test reports it
            out[index] = exc

    threads = [threading.Thread(target=run, args=(i, fn), name=f"issue5-surface-{i}", daemon=True)
               for i, fn in enumerate(calls)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(THREAD_TIMEOUT)
    assert not any(thread.is_alive() for thread in threads), "a surface call hung"
    return out


async def _control(driver, tester, **spec):
    finder = await driver.wait(timeout=60, **spec)
    return tester.control(finder.first)


async def _send_button(driver, tester):
    """The composer's Send: "send-idle_ready", or "send-queue" while another chat's job runs (it queues)."""
    specs = ({"key": "send-idle_ready"}, {"key": "send-queue"})
    index = await driver.wait_any(*specs, timeout=60)
    return tester.control((await driver.find(**specs[index])).first)


async def _answer_battery_sheet(driver) -> None:
    """The first long job on Android asks "Keep translations running" (``prepare_for_run``): "Not now", as the
    device flow answers it."""
    if await driver.count(text="Keep translations running"):
        await driver.tap(text="Not now")


async def _until(driver, predicate, timeout: float = 60.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        await _answer_battery_sheet(driver)
        if predicate():
            return True
        await asyncio.sleep(0.05)
    return bool(predicate())


async def _attach(driver, epub: Path) -> None:
    import flows

    await driver.tap(tooltip=flows.ATTACH_TOOLTIP)
    await driver.pick_file(epub.name, lambda: driver.tap(key="attach-files"))
    await driver.wait(contains=epub.stem, timeout=60)  # the composer pill


async def _new_chat(app, driver) -> str:
    import flows

    before = app.chat_view.cid
    await flows.go_home(driver)
    await driver.tap(tooltip="New chat")

    def fresh() -> bool:  # the desktop _new_chat rule: an empty current chat is reused
        cid = app.chat_view.cid
        return bool(cid) and (cid != before or not app.chat_feature.chats.messages(cid))

    assert await _until(driver, fresh, 30), (before, app.chat_view.cid)
    return app.chat_view.cid


def _user_files(app, cid) -> int:
    return sum(1 for m in app.chat_feature.chats.messages(cid) if m and m[0] == "user_file")


def _already(snacks, mark: int = 0) -> list:
    return [s for s in snacks[mark:] if any(words in s.message for words in ALREADY)]


async def _opens_job(app, driver, snack, job_id: str) -> None:
    """The snackbar's Open goes to that job's page (Jobs › job)."""
    assert snack.label == "Open" and callable(snack.action), snack
    snack.action()
    assert await _until(driver, lambda: str(app.shell.current_route or "").rstrip("/").endswith(f"/jobs/{job_id}"),
                        30), app.shell.current_route


def _clear_client_stop() -> None:
    """The teardown's force stop leaves the shared client's process-wide stop flags set; the next job's start clears
    them, but the tests after this one in the same process may start none (see test_hotfix1_glossary_editor_freeze)."""
    client = sys.modules.get("unified_api_client")
    if client is not None and hasattr(client, "set_stop_flag"):
        client.set_stop_flag(False)


async def _end_jobs(app, server) -> None:
    """Teardown: force-stop whatever still runs (a failed scenario), so no job thread outlives the test."""
    service = getattr(app, "job_service", None)
    if service is None:
        return
    server.release(abort=True)
    service.cancel_all_queued()
    for _attempt in range(3):
        active = service.snapshot()
        if active is None:
            break
        service.request_stop(active.id, force=True, reason="test teardown")
        if await I12._until(lambda: service.snapshot() is None, 30):
            break
    await asyncio.to_thread(service.wait_idle, 60)
    for thread in list(app.chat_feature.runs.finish_threads):
        await asyncio.to_thread(thread.join, 60)


async def _scenario(iso, server, body, *, chapters: int, glossary: str) -> None:
    """The app on its temp storage with the owner's settings (the fake endpoint, ``glossary``), the instruments
    installed, then ``body(ctx)``; the teardown ends every job and closes the app."""
    import flows

    epub = I12._book(iso.picks, chapters)
    tf = I12._UF._foundations()
    app, tester, driver = await I12._UF._host_driver(tf, {I12.EPUB_NAME: epub})
    service = app.job_service
    service.watcher_interval = 0.2
    subs = Submissions(service)
    TAPS.clear()
    try:
        await flows.wait_home(driver)
        I12._assert_isolated(iso.tmp)
        I12._configure(app, server, glossary)
        ctx = types.SimpleNamespace(app=app, tester=tester, driver=driver, server=server, service=service,
                                    subs=subs, epub=epub, chapters=chapters)
        await body(ctx)
    finally:
        try:
            for task in TAPS:
                if not task.done():
                    task.cancel()  # a failed scenario's handler still waiting
            TAPS.clear()
            subs.close()
            await _end_jobs(app, server)
        finally:
            try:
                app.jobs.close()
            finally:
                await tf._stop(app)
                _clear_client_stop()


async def _start_in_chat(ctx, *, sends: int = 3, starts: int = 3):
    """A new chat, ＋ › Files with the book, Send x``sends``, then Start x``starts`` on the Plan card; the chat's
    run once it has its job."""
    app, driver, tester = ctx.app, ctx.driver, ctx.tester
    cid = await _new_chat(app, driver)
    await _attach(driver, ctx.epub)
    send = await _send_button(driver, tester)
    await _burst(tester, send, sends)
    await driver.wait(text="Ready to translate", timeout=60)  # the Plan card
    assert _user_files(app, cid) == 1, app.chat_feature.chats.messages(cid)
    start = await _control(driver, tester, text="Start")
    await _burst(tester, start, starts)
    runs = app.chat_feature.runs
    assert await _until(driver, lambda: getattr(runs.run_for(cid), "job_id", None) is not None, 60), \
        "Start did not submit the chat's job"
    return cid, runs.run_for(cid)


# ==========================================================================
# 1. One chat's taps, the other surfaces and a second chat: one job
# ==========================================================================


def test_chat_taps_threads_and_a_second_chat_share_the_attachments_one_job(iso, offline, snacks):
    chapters = 24
    seen: dict = {}

    async def body(ctx):
        from glossarion_mobile.services.jobs import JOB_TARGET_PARAM, JobSpec, already_active_text

        app, driver, tester, server, service, subs = ctx.app, ctx.driver, ctx.tester, ctx.server, ctx.service, ctx.subs
        gate = I12.HoldGate(server)

        # ---- Send x3, Start x3 -------------------------------------------------------------------------------------
        cid, run = await _start_in_chat(ctx)
        job_id = run.job_id
        probe = I12.Probe(app, tester, cid)
        path = run.params[JOB_TARGET_PARAM][0]  # the file the chat works on (its Inbox copy)
        assert await I12._until(lambda: run.awaiting_glossary, RUN_TIMEOUT), probe.describe()
        assert subs.created == [job_id], (subs.created, _describe_calls(subs.calls))
        assert [s.id for s in subs.jobs()] == [job_id]
        assert _user_files(app, cid) == 1
        glossary_requests = server.count("glossary", status=None)
        assert glossary_requests >= 1

        # ---- at the gate: the other surfaces at once, from threads ---------------------------------------------------
        library_spec = app.library.translate_spec([{"name": I12.EPUB_NAME}], sources=[path])  # Library "Run translation"
        same_request = service.snapshot(job_id).spec  # the very same request again

        def other_chat_send(index):  # another chat's attachment send of the same file (its own run input)
            return JobSpec("direct_text", I12.EPUB_NAME, (str(iso.tmp / f"other{index}" / I12.EPUB_NAME),),
                           params={JOB_TARGET_PARAM: [path], "chat_id": 900 + index},
                           origin={"type": "chat", "cid": str(900 + index), "label": "Chat · other"})

        def surfaces():
            return ([lambda: service.submit(library_spec)] * 3 + [lambda: service.submit(same_request)] * 2
                    + [lambda: service.resume(job_id)] * 2
                    + [lambda i=i: service.submit(other_chat_send(i)) for i in range(2)])

        mark = subs.mark()
        results = await asyncio.to_thread(_hammer, surfaces())
        assert results == [job_id] * len(results), results
        calls = subs.since(mark)
        assert calls and all(c.job_id == job_id and c.reused for c in calls), _describe_calls(calls)
        assert subs.created == [job_id]

        # ---- a second chat with the same file: Send, Start x2 ----------------------------------------------------------
        mark, snack_mark = subs.mark(), len(snacks)
        cid2 = await _new_chat(app, driver)
        await _attach(driver, ctx.epub)
        assert app.chat_view.composer.attachment["path"] == path, "the same file is not the same attachment"
        send = await _send_button(driver, tester)
        await _burst(tester, send, 1)
        await driver.wait(text="Ready to translate", timeout=60)
        start = await _control(driver, tester, text="Start")
        await _burst(tester, start, 2)
        assert await _until(driver, lambda: _already(snacks, snack_mark), 30), [s.message for s in snacks[snack_mark:]]
        told = _already(snacks, snack_mark)
        assert len(told) == 1, [s.message for s in told]
        assert told[0].message == already_active_text(service.snapshot(job_id)), told[0].message
        assert subs.created == [job_id] and app.chat_feature.runs.live_run(cid2) is None
        assert all(c.job_id == job_id for c in subs.since(mark)), _describe_calls(subs.since(mark))
        # what the second chat's turn offers now (its Plan card's Start, a card's Resume / Retry failed) queues
        # nothing either while the book's job runs
        app.chat_view.render_transcript()
        probe2 = I12.Probe(app, tester, cid2)
        shows = sorted(probe2.visible_texts())
        offers = [c.action_buttons[key] for c in probe2.job_cards() for key in ("resume", "retry")
                  if key in c.action_buttons and probe2.on_screen(c.action_buttons[key])]
        plan_kept = isinstance(app.chat_feature.chats.meta(cid2).get("pending_plan"), dict)
        snack_mark2 = len(snacks)
        for control in offers:
            await _burst(tester, control, 2)
        if await driver.count(text="Start"):
            await _burst(tester, await _control(driver, tester, text="Start"), 2)
        assert await _until(driver, _handlers_ended, 30), [t for t in TAPS if not t.done()]
        assert subs.created == [job_id], _describe_calls(subs.since(mark))
        assert getattr(app.chat_feature.runs.live_run(cid2), "job_id", job_id) == job_id
        assert all(c.job_id == job_id for c in subs.since(mark)), _describe_calls(subs.since(mark))
        seen["second_chat"] = {"message": told[0].message, "messages": len(app.chat_feature.chats.messages(cid2)),
                               "plan_kept": plan_kept, "live_card": app.chat_view.live_job_card is not None,
                               "shows": shows, "offers": len(offers),
                               "taps_said": [s.message for s in snacks[snack_mark2:]]}
        # The owner's "they should reuse the same job": the second chat's turn is pointed at the book's running job,
        # never shown as a run of its own that ended. Checked at the end (the rest of the scenario runs first): it
        # keeps its Plan card (Start can be tapped again once the book's job ended, as "Save to: Library" keeps it)
        # or follows the book's job, and shows no "Interrupted · no output yet" Result with Resume / Retry failed.
        second_chat_ok = (plan_kept or app.chat_view.live_job_card is not None) and not offers \
            and not any(t.startswith("Interrupted") for t in shows)
        await _opens_job(app, driver, told[0], job_id)

        # ---- back in the first chat: ✓ Yes x3 with the notification's answer from threads ------------------------------
        await app.navigate(f"/chat/{cid}")
        assert await _until(driver, lambda: app.chat_view.cid == cid and app.chat_view.approval_card is not None, 30)
        question = service.pending_question(job_id)
        assert question is not None
        gate.hold_after(6)  # the model holds after 6 translation replies: the job is still running at the next checks
        yes = app.chat_view.approval_card.yes_button
        assert probe.on_screen(yes)
        _taps, answers = await asyncio.gather(
            _burst(tester, yes, 3),
            asyncio.to_thread(_hammer, [lambda: service.answer(question["id"], True)] * 4))
        assert answers == [True] * 4, answers
        assert await I12._until(lambda: I12._settled(server), RUN_TIMEOUT), I12._held_state(probe, server)
        assert service.snapshot(job_id).state.value == "RUNNING" and service.pending_question(job_id) is None
        assert server.count("glossary", status=None) == glossary_requests, "the glossary pass ran again"

        # ---- while it translates: the same surfaces again ---------------------------------------------------------------
        mark = subs.mark()
        results = await asyncio.to_thread(_hammer, surfaces())
        assert results == [job_id] * len(results), results
        assert all(c.job_id == job_id and c.reused for c in subs.since(mark)), _describe_calls(subs.since(mark))
        assert len(_chapter_requests(server)) < chapters, "the run is not held mid-book"
        server.release()

        # ---- the end: one job did all of it, each request once -----------------------------------------------------------
        await I12._finished(app, run, RUN_TIMEOUT)
        assert run.state == "done" and run.error is None, (run.state, run.error)
        assert subs.created == [job_id] and [s.id for s in subs.jobs()] == [job_id], \
            (subs.created, [(s.id, s.spec.kind, s.spec.origin) for s in subs.jobs()])
        assert all(c.job_id == job_id for c in subs.calls), _describe_calls(subs.calls)
        assert sum(1 for c in subs.calls if c.what == "submit" and not c.reused) == 1, _describe_calls(subs.calls)
        assert server.count("glossary", status=None) == glossary_requests
        _assert_each_chapter_once(server, chapters)
        assert not _duplicate_requests(server), _duplicate_requests(server)
        assert not [s.message for s in snacks if s.message.startswith("Could not")], [s.message for s in snacks]
        assert await _until(driver, _handlers_ended, 30), [t for t in TAPS if not t.done()]
        assert not _tap_errors(), _tap_errors()
        _diagnose(ctx, snacks)
        seen["calls"] = len(subs.calls)
        print(f"second chat: {seen['second_chat']}")
        assert second_chat_ok, (
            "the second chat with the same file is not pointed at the book's job: after the refused Start its turn "
            "lost its Plan card and shows a Result of a run it never had", seen["second_chat"])

    with _server() as server:
        asyncio.run(_scenario(iso, server, body, chapters=chapters, glossary="balanced"))
    I12._assert_offline(offline)
    assert seen.get("calls"), "the scenario did not reach its end"


# ==========================================================================
# 2. Stop, then Resume from every surface at once: one resumed job
# ==========================================================================


def test_stop_then_resume_from_every_surface_resumes_once(iso, offline, snacks):
    chapters = 30
    seen: dict = {}

    async def body(ctx):
        from glossarion_mobile.services.jobs import JOB_TARGET_PARAM

        app, driver, tester, server, service, subs = ctx.app, ctx.driver, ctx.tester, ctx.server, ctx.service, ctx.subs
        gate = I12.HoldGate(server, 9)  # the EPUB metadata request + 8 chapters, then ■ Stop
        cid, run = await _start_in_chat(ctx, sends=2, starts=2)
        stopped = run.job_id
        probe = I12.Probe(app, tester, cid)
        view = app.chat_view
        path = run.params[JOB_TARGET_PARAM][0]

        # ---- ■ Stop mid-book (graceful: the held requests answer, the run commits them) -----------------------------
        assert await I12._until(lambda: I12._settled(server), RUN_TIMEOUT), I12._held_state(probe, server)
        await I12._tap(probe, view.live_job_card.action_buttons["stop"])
        assert await I12._until(lambda: run.stop_requested, 5)
        assert await I12._until(lambda: probe.job()["state"] == "STOPPING", 10), probe.job()
        server.release()
        await I12._finished(app, run, RUN_TIMEOUT)
        assert run.state == "stopped", run.state
        done_before = set(_chapter_requests(server))
        assert done_before and len(done_before) < chapters - 10, done_before
        assert subs.created == [stopped]

        # ---- Resume x3 on the Result card + the Jobs page / notification Resume from threads, at once --------------------
        ended = probe.job_cards()[0]
        assert "resume" in ended.action_buttons and probe.on_screen(ended.action_buttons["resume"])
        gate.hold_after(6)
        mark = subs.mark()
        _taps, results = await asyncio.gather(
            _burst(tester, ended.action_buttons["resume"], 3),
            asyncio.to_thread(_hammer, [lambda: service.resume(stopped)] * 4))
        assert await _until(driver, lambda: I12._next_run(probe, run) is not None, 30), _describe_calls(subs.calls)
        resumed = I12._next_run(probe, run)
        resumed_id = resumed.job_id
        assert resumed_id and resumed_id != stopped
        assert results == [resumed_id] * 4, (results, _describe_calls(subs.since(mark)))
        assert subs.created == [stopped, resumed_id], (subs.created, _describe_calls(subs.since(mark)))
        assert await I12._until(lambda: I12._settled(server), RUN_TIMEOUT), I12._held_state(probe, server)
        assert service.snapshot(resumed_id).state.value == "RUNNING"
        assert view.live_job_card is not None and "resume" not in view.live_job_card.action_buttons

        # ---- while it runs: Resume again (both jobs), the Library's run of the file, the chat's Resume path -------------
        library_spec = app.library.translate_spec([{"name": I12.EPUB_NAME}], sources=[path])
        mark = subs.mark()
        results = await asyncio.to_thread(_hammer, [lambda: service.resume(stopped)] * 2
                                          + [lambda: service.resume(resumed_id)] * 2
                                          + [lambda: service.submit(library_spec)] * 2)
        assert results == [resumed_id] * 6, results
        again = await app.chat_feature.runs.resubmit(cid)
        assert again is resumed, again
        assert all(c.job_id == resumed_id and c.reused for c in subs.since(mark)), _describe_calls(subs.since(mark))
        server.release()

        # ---- the end: Done, every chapter once over both runs ------------------------------------------------------------
        await I12._finished(app, resumed, RUN_TIMEOUT)
        assert resumed.state == "done" and resumed.error is None, (resumed.state, resumed.error)
        assert subs.created == [stopped, resumed_id], subs.created
        assert [s.id for s in subs.jobs()] == [stopped, resumed_id]
        # the first Start queued the stopped job; every call after it answered the resumed job, which one of them
        # queued (a resume that queues is recorded with its own submit)
        first, later = subs.calls[0], subs.calls[1:]
        assert first.job_id == stopped and not first.reused, _describe_calls(subs.calls)
        assert later and all(c.job_id == resumed_id for c in later), _describe_calls(subs.calls)
        assert sum(1 for c in later if c.what == "submit" and not c.reused) == 1, _describe_calls(subs.calls)
        _assert_each_chapter_once(server, chapters)
        assert not [s.message for s in snacks if s.message.startswith("Could not")], [s.message for s in snacks]
        assert await _until(driver, _handlers_ended, 30), [t for t in TAPS if not t.done()]
        assert not _tap_errors(), _tap_errors()
        _diagnose(ctx, snacks)
        seen["resumed"] = resumed_id

    with _server() as server:
        asyncio.run(_scenario(iso, server, body, chapters=chapters, glossary="off"))
    I12._assert_offline(offline)
    assert seen.get("resumed"), "the scenario did not reach its end"


# ==========================================================================
# 3. The Library book: Run translation, Run QA scan and a chat with the same file
# ==========================================================================


def test_library_run_translation_progress_manager_qa_and_a_chat_share_one_job(iso, offline, snacks):
    chapters = 20
    seen: dict = {}

    async def body(ctx):
        from glossarion_mobile.services.jobs import JOB_TARGET_PARAM, JobSpec, already_active_text
        from glossarion_mobile.ui.tools import qa_model as qm
        from glossarion_mobile.ui.tools import targets as tg

        app, driver, tester, server, service, subs = ctx.app, ctx.driver, ctx.tester, ctx.server, ctx.service, ctx.subs
        library = app.library
        gate = I12.HoldGate(server)

        # ---- the book in the Library, its Book page › Chapters ---------------------------------------------------------
        report = await asyncio.to_thread(library.import_paths_blocking, [str(ctx.epub)], "raw")
        assert not report.errors, report
        await library.refresh()
        book = next(b for b in library.snapshot.all_books()
                    if os.path.basename(str(b.get("raw_source_path") or "")) == I12.EPUB_NAME)
        bid = library.bid_for(book)
        raw = await asyncio.to_thread(library.raw_source, book)
        assert raw and os.path.isfile(raw)

        async def chapters_tab():
            await app.navigate(f"/library/book/{bid}?tab=chapters")
            assert await _until(driver, lambda: getattr(getattr(app.shell.top_screen, "chapters", None),
                                                        "run_button", None) is not None, 60)
            tab = app.shell.top_screen.chapters
            assert await _until(driver, lambda: tab.view is not None and not tab.run_button.disabled, 60)
            return tab

        chapters_view = await chapters_tab()

        # ---- Run translation x3: one sheet; its Start x3: one job ------------------------------------------------------
        await _burst(tester, chapters_view.run_button, 3)
        start = await _control(driver, tester, key="start")
        sheets = [d for d in (getattr(getattr(app.page, "_dialogs", None), "controls", None) or [])
                  if getattr(d, "open", False) and type(d).__name__ == "BottomSheet"]
        assert len(sheets) == 1, f"{len(sheets)} translate sheets opened"
        gate.hold_after(5)
        mark = subs.mark()
        await _burst(tester, start, 3)
        assert await _until(driver, lambda: len(subs.since(mark)) >= 3 and service.snapshot() is not None, 60), \
            _describe_calls(subs.since(mark))
        job_id = service.snapshot().id
        calls = subs.since(mark)
        assert len(calls) == 3 and all(c.job_id == job_id for c in calls), _describe_calls(calls)
        assert sum(1 for c in calls if not c.reused) == 1 and subs.created == [job_id], _describe_calls(calls)
        assert service.snapshot(job_id).spec.kind == "translate"
        assert await I12._until(lambda: I12._settled(server), RUN_TIMEOUT), \
            (server.holding, server.parked, server.in_flight())

        # ---- while it runs: Run translation / Run QA scan taps that still reach the app queue nothing ------------------
        assert await _until(driver, lambda: chapters_view.run_button.disabled and chapters_view.qa_button.disabled, 30)
        mark, snack_mark = subs.mark(), len(snacks)
        reasons = {chapters_view.run_button.tooltip, chapters_view.qa_button.tooltip}
        await _burst(tester, chapters_view.qa_button, 2)
        await _burst(tester, chapters_view.run_button, 2)
        # the handlers (spawned by the clicks) say why the buttons wait; the final job list also covers a late one
        assert await _until(driver, lambda: sum(1 for s in snacks[snack_mark:] if s.message in reasons) >= 2, 30), \
            [s.message for s in snacks[snack_mark:]]
        assert subs.since(mark) == [] and subs.created == [job_id], _describe_calls(subs.since(mark))

        # ---- a chat with the same file: "Save to: Library", Start x2 ---------------------------------------------------
        snack_mark = len(snacks)
        cid = await _new_chat(app, driver)
        await _attach(driver, ctx.epub)
        send = await _send_button(driver, tester)
        await _burst(tester, send, 1)
        await driver.wait(text="Ready to translate", timeout=60)
        plan = app.chat_feature.chats.meta(cid).get("pending_plan") or {}
        assert plan.get("destination") == "library" and plan.get("library_bid") == bid, plan
        start = await _control(driver, tester, text="Start")
        await _burst(tester, start, 2)
        assert await _until(driver, lambda: _already(snacks, snack_mark), 30), [s.message for s in snacks[snack_mark:]]
        told = _already(snacks, snack_mark)
        assert len(told) == 1 and told[0].message == already_active_text(service.snapshot(job_id)), \
            [s.message for s in told]
        assert subs.created == [job_id]
        seen["chat"] = told[0].message
        await _opens_job(app, driver, told[0], job_id)

        # ---- the other surfaces at once, from threads ------------------------------------------------------------------
        library_spec = library.translate_spec([book], sources=[raw])
        chat_send = JobSpec("direct_text", I12.EPUB_NAME, (str(iso.tmp / "other" / I12.EPUB_NAME),),
                            params={JOB_TARGET_PARAM: [raw], "chat_id": 901},
                            origin={"type": "chat", "cid": "901", "label": "Chat · other"})
        mark = subs.mark()
        results = await asyncio.to_thread(_hammer, [lambda: service.submit(library_spec)] * 4
                                          + [lambda: service.resume(job_id)] * 2
                                          + [lambda: service.submit(chat_send)] * 2)
        assert results == [job_id] * 8, results
        assert all(c.job_id == job_id and c.reused for c in subs.since(mark)), _describe_calls(subs.since(mark))
        server.release()
        assert await _until(driver, lambda: service.snapshot(job_id).is_terminal, RUN_TIMEOUT), service.snapshot(job_id)
        assert service.snapshot(job_id).state.value == "DONE", service.snapshot(job_id)
        _assert_each_chapter_once(server, chapters)
        translations = server.count("translation", status=None)

        # ---- Run QA scan x3 + the same scan from threads: one QA scan -------------------------------------------------
        chapters_view = await chapters_tab()
        assert await _until(driver, lambda: not chapters_view.qa_button.disabled, 60), chapters_view.qa_button.tooltip
        book_now = library.book_for_bid(bid) or book
        target = await asyncio.to_thread(tg.target_for_book, library, book_now)
        qa_spec = qm.qa_spec([target], qm.CHAT_QA_MODE, disable_word_count=not target.source,
                             origin=library.origin_for(book_now))
        mark = subs.mark()
        _taps, results = await asyncio.gather(_burst(tester, chapters_view.qa_button, 3),
                                              asyncio.to_thread(_hammer, [lambda: service.submit(qa_spec)] * 4))
        assert await _until(driver, lambda: len(subs.created) == 2, 30), subs.created
        qa_id = subs.created[1]
        assert service.snapshot(qa_id).spec.kind == "qa_scan"
        assert results == [qa_id] * 4, results
        assert await _until(driver, lambda: len(subs.since(mark)) >= 5, 30), _describe_calls(subs.since(mark))
        calls = subs.since(mark)
        assert all(c.job_id == qa_id for c in calls) and sum(1 for c in calls if not c.reused) == 1, \
            _describe_calls(calls)
        assert await _until(driver, lambda: service.snapshot(qa_id).is_terminal, RUN_TIMEOUT), service.snapshot(qa_id)
        assert service.snapshot(qa_id).state.value == "DONE", service.snapshot(qa_id)
        assert [s.id for s in subs.jobs()] == [job_id, qa_id], [(s.id, s.spec.kind) for s in subs.jobs()]
        assert server.count("translation", status=None) == translations, "the QA scan sent requests"
        assert not _duplicate_requests(server), _duplicate_requests(server)
        assert not [s.message for s in snacks if s.message.startswith("Could not")], [s.message for s in snacks]
        assert await _until(driver, _handlers_ended, 30), [t for t in TAPS if not t.done()]
        assert not _tap_errors(), _tap_errors()
        _diagnose(ctx, snacks)
        seen["qa"] = qa_id

    with _server() as server:
        asyncio.run(_scenario(iso, server, body, chapters=chapters, glossary="off"))
    I12._assert_offline(offline)
    assert seen.get("qa"), "the scenario did not reach its end"


# ==========================================================================
# 4. ChatView: a send / resume that started nothing (review findings)
# ==========================================================================

TC = _load("_glossarion_chat_helpers_hotfix1_issue5", "test_chat.py")  # the ChatView harness (``_chat_view``)
desktop_store_cls = TC.desktop_store_cls
isolated_env = TC.isolated_env


def test_a_refused_send_without_a_plan_card_puts_its_attachment_back(desktop_store_cls, isolated_env):
    """A Send that needs no Plan card (a short TXT) clears the composer first; when job safety then refuses it
    (another chat or the Library already runs that file) nothing was recorded, so its text and attachment go back
    into the composer instead of being lost; the snackbar says where the file runs."""
    from glossarion_mobile.services.jobs import JobAlreadyActive

    async def scenario():
        chat = TC._chat_view(desktop_store_cls, isolated_env)
        view, calls = chat.view, chat.calls
        try:
            await TC._settle(calls)
            path = isolated_env / "short.txt"
            path.write_text("짧은 글입니다.", encoding="utf-8")
            record = {"name": "short.txt", "path": str(path), "extension": ".txt", "size": path.stat().st_size}
            before = len(chat.adapter.messages(view.cid))
            sent: list = []

            async def refuse(cid, **kwargs):
                sent.append(dict(kwargs.get("attachment") or {}))
                raise JobAlreadyActive(None, "“short.txt” is already starting in another chat")

            view.env.runs.send = refuse
            view.composer.set_attachment(record)
            view.composer.set_text("please translate")
            view.begin_send()
            await TC._settle(calls)
            assert sent and sent[0].get("path") == str(path), sent
            assert len(chat.adapter.messages(view.cid)) == before, "the refused send recorded a turn"
            assert view.composer.attachment == record, view.composer.attachment
            assert view.composer.send_text() == "please translate"
            assert any("already starting" in str(note[0]) for note in calls.notes), calls.notes
            # something typed meanwhile is never overwritten
            view.composer.clear()
            view.composer.set_text("newer")
            view._give_back(view.cid, None, ("please translate", record), before)
            assert view.composer.send_text() == "newer" and view.composer.attachment is None
        finally:
            TC._close(chat)

    asyncio.run(scenario())


def test_a_resume_the_preflight_refused_does_not_also_say_nothing_to_resume(desktop_store_cls, isolated_env):
    """``ChatRuns.resubmit`` returns None both when the app's preflight refused the run (it already said why, e.g.
    "… · Choose model": ``take_refused``) and when there is nothing to resume; only the latter says "Nothing to
    resume in this chat"."""

    async def scenario():
        chat = TC._chat_view(desktop_store_cls, isolated_env)
        view, calls, runs = chat.view, chat.calls, chat.runs
        try:
            await TC._settle(calls)

            async def refused(cid, **kwargs):
                runs._mark_refused(str(cid))
                return None

            async def nothing(cid, **kwargs):
                return None

            runs.resubmit = refused
            assert await view._resubmit(view.cid) is None
            assert not any("Nothing to resume" in str(note[0]) for note in calls.notes), calls.notes
            runs.resubmit = nothing
            assert await view._resubmit(view.cid) is None
            assert any("Nothing to resume in this chat" in str(note[0]) for note in calls.notes), calls.notes
        finally:
            TC._close(chat)

    asyncio.run(scenario())
