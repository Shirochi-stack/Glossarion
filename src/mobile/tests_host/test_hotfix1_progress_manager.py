"""Hotfix 1, issues 7 and 9: Book page › Chapters (the Progress Manager).

Issue 7 (owner, device): next to Run translation there is a Run QA scan, and both buttons are disabled
during a glossary or translation run.

* Run QA scan submits the chat's one-tap scan for this book (``qa_scan`` job, ``qa_model.CHAT_QA_MODE``
  Quick Scan; the job applies the mobile sample size 0) on the book's workspace and raw source, with the
  book's Library origin; a double tap queues one scan; a finished scan reloads the rows and offers its report.
* Both buttons wait (disabled, the reason as tooltip) while a translation or glossary run is active (any book:
  jobs run one at a time; also at its glossary gate) or queued for this book; Run QA scan also while this
  book's own scan is queued or running. They follow every JobsView delivery live - checked with a scripted
  jobs fake and with the real ``JobService`` (submit -> disabled, run ends -> enabled, no reopen).

Issue 9 (owner, device): "Lack of feedback when interacting with progress manager, like setting an entry to in
progress just happens with no refresh". Root cause: ``ChaptersTab._apply_planned`` wrote the progress and
reloaded silently (5f74999e dropped the result snackbar, nothing showed work in progress), so an action looked
like nothing happened. Now the list's progress bar runs while the action writes and the rows reload, the rows,
the stats chips and the strip refresh at once (desktop ``_refresh_retranslation_data``), then the desktop
result text follows; failures say why. Glossary Progress actions do the same (a forced reload: a poll load in
flight no longer swallows the action's refresh).

Run from src/mobile with the mobile venv (Python 3.12):
    .venv/Scripts/python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic \
        tests_host/test_hotfix1_progress_manager.py
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import json
import os
import sys
import threading
import time
import types
from pathlib import Path

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = MOBILE_DIR / "app"
SRC_DIR = MOBILE_DIR.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

# The Library fixtures (fixture Library + workspace with completed / qa_failed / chunked pending rows, the
# real shared cores pinned to it) and the Flet fake session live in test_library_ui.
_TLU_SPEC = importlib.util.spec_from_file_location("_glossarion_tlu_hotfix1_pm",
                                                   Path(__file__).with_name("test_library_ui.py"))
_TLU = importlib.util.module_from_spec(_TLU_SPEC)
_TLU_SPEC.loader.exec_module(_TLU)

real_env = _TLU.real_env  # noqa: F811 - the fixture, re-exported for this module

needs_flet = _TLU.needs_flet
TIMEOUT = 20.0


def _needs_cores():
    if _TLU._core("progress_core", "build_book_progress") is None or _TLU._core(
            "progress_actions", "restore_in_progress", "remove_qa_marks") is None:
        pytest.skip("progress cores not importable")


def _ctx(page, service, jobs=None):
    """A LibraryContext whose snackbars record (message, action label, handler)."""
    from glossarion_mobile.ui.library.common import LibraryContext

    navigated, notes = [], []
    ctx = LibraryContext(service=service, page=page, jobs=jobs, prefs=service.prefs, platform="android",
                         navigate=lambda name, params=None, query=None: navigated.append((name, params, query)),
                         notify=lambda message, action=None, on_action=None: notes.append((message, action, on_action)))
    ctx.navigated, ctx.notes = navigated, notes
    return ctx


def _messages(ctx):
    return [note[0] for note in ctx.notes]


def _set_status(real_env, key, status):
    path = real_env["ws"] / "translation_progress.json"
    prog = json.loads(path.read_text(encoding="utf-8"))
    prog["chapters"][key]["status"] = status
    path.write_text(json.dumps(prog), encoding="utf-8")


async def _open_book(real_env, jobs=None):
    from glossarion_mobile.ui.library.book_page import BookPageScreen
    from glossarion_mobile.ui.router import parse_route

    service = real_env["service"]
    conn, session = _TLU._TB._fake_session("android")
    page = session.page
    await service.refresh()
    book = service.snapshot.in_progress[0]
    if jobs is not None:
        service.jobs = jobs
    ctx = _ctx(page, service, jobs)
    screen = BookPageScreen(parse_route(f"/library/book/{service.bid_for(book)}?tab=chapters"), ctx)
    screen.actions()
    _TLU._mount(page, screen.get_body())
    await screen.load()
    return conn, screen, ctx, book


def _sent(conn, start):
    """Every message sent to the client since ``start``, msgpack-encoded like the socket transport."""
    import msgpack
    from flet.controls.base_control import BaseControl
    from flet.messaging.protocol import configure_encode_object_for_msgpack

    encode = configure_encode_object_for_msgpack(BaseControl)
    return b"".join(msgpack.packb([m.action, m.body], default=encode) for m in conn.messages[start:])


def _chips(chapters):
    return {chip.group: chip.count for chip in chapters.view.chips}


async def _until(predicate, timeout=TIMEOUT):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.02)
    return predicate()


# ==========================================================================
# Issue 9: feedback after every Progress Manager action
# ==========================================================================


@needs_flet
def test_restore_in_progress_shows_work_refreshes_rows_and_counts_and_confirms(real_env, monkeypatch):
    """The owner's case: an In progress row's action. While it writes, the list's progress bar runs; the row,
    the stats chips and the strip refresh at once (and reach the client); then the desktop result text
    ("In Progress Restored") follows as a snackbar."""
    _needs_cores()
    from glossarion_mobile.ui.library import progress_model as pm

    _set_status(real_env, "1", "in_progress")
    seen = {}
    real_apply = pm.apply_action

    def apply(service, view, plan, **kwargs):
        seen["bar_while_writing"] = seen["screen"].chapters.loading.visible
        return real_apply(service, view, plan, **kwargs)

    monkeypatch.setattr(pm, "apply_action", apply)

    async def scenario():
        conn, scr, ctx, _book = await _open_book(real_env)
        seen["screen"] = scr
        chapters = scr.chapters
        row = next(r for r in chapters.visible if r.status == "in_progress")
        before = _chips(chapters)
        assert before["in_progress"] == 1 and before["failed"] == 1 and not chapters.loading.visible
        start = len(conn.messages)
        result = await chapters.run_action("restore_in_progress", [row])
        assert result == "Successfully marked 1 as failed."
        assert seen["bar_while_writing"] is True and not chapters.loading.visible
        after = next(r for r in chapters.visible if r.key == row.key)
        assert after.status == "failed"
        assert any("Failed" in t for t in _TLU._texts(chapters.controls[row.key]))
        chips = _chips(chapters)
        assert chips["in_progress"] == 0 and chips["failed"] == 2
        assert any(c.text.endswith("2") for c in chapters.view.chips if c.group == "failed")
        assert b"Failed" in _sent(conn, start)  # the refreshed row was pushed to the client
        assert _messages(ctx)[-1] == "Successfully marked 1 as failed."
        assert chapters.last_message == ("In Progress Restored", "Successfully marked 1 as failed.")
        scr.dispose()

    asyncio.run(scenario())


@needs_flet
def test_bulk_remove_qa_confirms_and_a_failed_write_says_why(real_env, monkeypatch):
    """A bulk selection action (Remove QA mark): the confirmation, then refreshed rows/chips + the desktop text;
    a write that fails says why and the progress bar stops."""
    _needs_cores()
    from glossarion_mobile.ui.library import progress_model as pm

    async def scenario():
        conn, screen, ctx, _book = await _open_book(real_env)
        chapters = screen.chapters
        asked = []

        async def confirm(title, body, **_kw):
            asked.append((title, body))
            return True

        chapters._confirm = confirm
        qa_row = next(r for r in chapters.visible if r.status == "qa_failed")
        done_row = next(r for r in chapters.visible if r.status == "completed")
        chapters.on_row_long_press(qa_row)
        chapters.toggle_select(done_row.key, force=True)
        primary, _more = chapters.bulk_actions(chapters.selected_rows())
        remove = next(a for a in primary if a.id == "remove_qa")
        assert remove.disabled_reason is None
        before = _chips(chapters)
        message = await chapters.run_action("remove_qa", chapters.selected_rows())
        assert asked[-1] == ("Confirm Remove Failed Mark", "Remove failed mark from 1 chapters?")
        assert message == "Removed failed mark from 1 chapters."
        assert _messages(ctx)[-1] == "Removed failed mark from 1 chapters."
        after = _chips(chapters)
        assert not chapters.selecting and before["failed"] == 1 and after["failed"] == 0
        assert after["completed"] == before["completed"] + 1
        assert next(r for r in chapters.visible if r.key == qa_row.key).status == "completed"

        def broken(*_a, **_kw):
            raise OSError("disk full")

        monkeypatch.setattr(pm, "apply_action", broken)
        _set_status(real_env, "1", "in_progress")
        await screen.reload_progress(force=True)
        row = next(r for r in chapters.visible if r.status == "in_progress")
        assert await chapters.run_action("restore_in_progress", [row]) is None
        assert _messages(ctx)[-1] == "Could not update progress: disk full"
        assert not chapters.loading.visible
        screen.dispose()

    asyncio.run(scenario())


@needs_flet
def test_glossary_progress_action_refreshes_even_with_a_poll_load_in_flight(real_env):
    """Glossary Progress › Mark as completed: the rows refresh (a forced reload waits for a poll load in flight
    instead of being dropped) and a snackbar says what changed."""
    if _TLU._core("glossary_progress_core", "open_glossary_progress", "mark_glossary_completed") is None:
        pytest.skip("glossary_progress_core not importable")

    async def scenario():
        conn, screen, ctx, _book = await _open_book(real_env)
        await screen.reload_glossary()
        tab = screen.glossary_tab
        row = next(r for r in tab.visible_rows() if r.kind == "chapter" and r.status == "not_completed")
        screen._loading_glossary = True  # a 2 s poll load is reading the file right now
        result = await tab.run_action("mark_completed", [row])
        assert result is not None
        assert next(r for r in tab.view.rows if r.key == row.key).status == "completed"
        assert _messages(ctx)[-1] == "Marked 1 chapter as completed"
        assert not tab.loading.visible
        screen.dispose()

    asyncio.run(scenario())
    from glossarion_mobile.ui.library.glossary_tab import action_done_text

    one = types.SimpleNamespace(kind="chapter", title="Ch.4 · ch004.xhtml")
    two = types.SimpleNamespace(kind="minimal", title="Minimal Pass")
    assert action_done_text("remove", [one]) == "Removed Ch.4 from progress"
    assert action_done_text("remove", [one, two]) == "Removed 2 rows from progress"
    assert action_done_text("mark_completed", [one, one]) == "Marked 2 chapters as completed"


# ==========================================================================
# Issue 7: Run QA scan, and both run buttons waiting for translation / glossary runs
# ==========================================================================


class FakeJobs:
    """The JobService surface the Book page uses, with real JobSnapshot / JobsView objects; ``publish``
    delivers a new JobsView to the subscribers like ``JobService._deliver_view``."""

    def __init__(self):
        self.listeners, self.transitions, self.submitted = [], [], []
        self.active = None
        self.queue = ()

    def has_kind(self, kind):
        return True

    @property
    def busy(self):
        return self.active is not None

    def subscribe(self, callback, *, immediate=False):
        self.listeners.append(callback)
        return lambda: self.listeners.remove(callback)

    def on_transition(self, callback):
        self.transitions.append(callback)
        return lambda: self.transitions.remove(callback)

    def view(self):
        from glossarion_mobile.services.jobs import JobsView

        return JobsView(active=self.active, queue=tuple(self.queue))

    def publish(self, active=None, queue=()):
        self.active, self.queue = active, tuple(queue)
        view = self.view()
        for callback in list(self.listeners):
            callback(view)

    def submit(self, spec):
        from glossarion_mobile.services.jobs import JobSnapshot, JobState

        job_id = f"job{len(self.submitted) + 1}"
        self.submitted.append((job_id, spec))
        self.publish(self.active, tuple(self.queue) + (JobSnapshot(id=job_id, spec=spec, state=JobState.QUEUED,
                                                                   created=1.0),))
        return job_id

    def end(self, job_id, state, outputs=(), error=None):
        from glossarion_mobile.services.jobs import JobSnapshot, JobState

        spec = next(s for jid, s in self.submitted if jid == job_id)
        snap = JobSnapshot(id=job_id, spec=spec, state=state, created=1.0, started=2.0, finished=3.0,
                           outputs=tuple(outputs), error=error)
        active = None if self.active is not None and self.active.id == job_id else self.active
        self.publish(active, tuple(s for s in self.queue if s.id != job_id))
        for callback in list(self.transitions):
            callback(snap, JobState.RUNNING)


def _snap(job_id, kind, state, *, origin=None, inputs=(), question=None, refining=False):
    from glossarion_mobile.services.jobs import JobSnapshot, JobSpec

    spec = JobSpec(kind=kind, title=job_id, inputs=inputs, origin=origin or {"type": "library", "bid": "other"})
    return JobSnapshot(id=job_id, spec=spec, state=state, created=1.0, question=question, refining=refining)


@needs_flet
def test_run_qa_scan_queues_the_books_quick_scan_once_and_offers_the_report(real_env):
    _needs_cores()
    from glossarion_mobile.job_kinds.qa import report_path_for
    from glossarion_mobile.services.jobs import JobState
    from glossarion_mobile.ui.library.chapters_tab import QA_QUEUED, QA_RUNNING
    from glossarion_mobile.ui.tools import qa_model as qm

    jobs = FakeJobs()

    async def scenario():
        conn, screen, ctx, book = await _open_book(real_env, jobs)
        screen._unsubs.append(jobs.subscribe(screen._on_jobs))  # did_show's jobs wiring
        screen._on_jobs(jobs.view())
        chapters = screen.chapters
        header = next(c for c in chapters.root.controls[0].content.controls if hasattr(c, "controls"))
        keys = [getattr(c, "key", None) for c in header.controls]
        assert keys.index("ch-qa") == keys.index("ch-run") + 1  # next to Run translation
        assert chapters.qa_button.content == "Run QA scan" and not chapters.qa_button.disabled
        # a double tap queues one scan
        first, second = await asyncio.gather(chapters.run_qa_scan(), chapters.run_qa_scan())
        assert first == "job1" and second is None and len(jobs.submitted) == 1
        spec = jobs.submitted[0][1]
        assert spec.kind == "qa_scan" and spec.params["mode"] == qm.CHAT_QA_MODE == "quick-scan"
        assert spec.params["targets"] == [{"folder": str(real_env["ws"]), "source": real_env["raw"]}]
        assert "disable_word_count" not in spec.params  # the raw source is known
        assert spec.origin["bid"] == real_env["service"].bid_for(book)
        assert not any(m == QA_QUEUED for m in _messages(ctx))  # the strip shows it ("· queued")
        assert screen.job is not None and screen.job.id == "job1"
        assert chapters.qa_button.disabled and chapters.qa_button.tooltip == QA_RUNNING
        assert not chapters.run_button.disabled  # a QA scan does not block Run translation
        assert await chapters.run_qa_scan() is None and _messages(ctx)[-1] == QA_RUNNING
        # the scan ends: the rows reload, the report is offered, the button comes back
        report = report_path_for(str(real_env["ws"]))
        jobs.end("job1", JobState.DONE, outputs=(report,))
        assert await _until(lambda: any(n[0] == "QA scan finished" for n in ctx.notes))
        note = next(n for n in ctx.notes if n[0] == "QA scan finished")
        assert note[1] == "Report" and callable(note[2])
        assert not chapters.qa_button.disabled and chapters.qa_button.tooltip is None
        # a failed scan says why
        assert await chapters.run_qa_scan() == "job2"
        jobs.end("job2", JobState.FAILED, error="boom")
        assert await _until(lambda: _messages(ctx)[-1] == "QA scan failed: boom")
        screen.dispose()

    asyncio.run(scenario())


@needs_flet
def test_run_buttons_wait_for_translation_and_glossary_runs_and_follow_job_events(real_env):
    _needs_cores()
    from glossarion_mobile.services.jobs import JobState
    from glossarion_mobile.ui.library import book_page as bp
    from glossarion_mobile.ui.library.chapters_tab import QA_RUNNING

    jobs = FakeJobs()

    async def scenario():
        conn, screen, ctx, book = await _open_book(real_env, jobs)
        screen._unsubs.append(jobs.subscribe(screen._on_jobs))
        screen._on_jobs(jobs.view())
        chapters = screen.chapters
        bid = real_env["service"].bid_for(book)
        mine = {"type": "library", "bid": bid}

        def state():
            return (chapters.run_button.disabled, chapters.run_button.tooltip,
                    chapters.qa_button.disabled, chapters.qa_button.tooltip)

        assert state() == (False, None, False, None)
        cases = [
            # another book's translation runs: jobs run one at a time
            (_snap("t", "translate", JobState.RUNNING), (), bp.RUN_WAIT_TRANSLATION, bp.RUN_WAIT_TRANSLATION),
            # a glossary extraction runs
            (_snap("g", "extract_glossary", JobState.RUNNING), (), bp.RUN_WAIT_GLOSSARY, bp.RUN_WAIT_GLOSSARY),
            # this book's translation waits at its glossary gate (also while the gate's glossary is refined)
            (_snap("gate", "translate", JobState.RUNNING, origin=mine, refining=True,
                   question={"id": "q1", "kind": "glossary_approval", "data": {}}), (),
             bp.RUN_WAIT_TRANSLATION, bp.RUN_WAIT_TRANSLATION),
            # this book's translation is queued behind another book's compile
            (_snap("c", "compile_epub", JobState.RUNNING), (_snap("q", "translate", JobState.QUEUED, origin=mine),),
             bp.RUN_WAIT_QUEUED_TRANSLATION, bp.RUN_WAIT_QUEUED_TRANSLATION),
            # this book's glossary run is queued (matched by its raw input)
            (None, (_snap("qg", "extract_glossary", JobState.QUEUED, inputs=(real_env["raw"],)),),
             bp.RUN_WAIT_QUEUED_GLOSSARY, bp.RUN_WAIT_QUEUED_GLOSSARY),
            # another book's translation queued, a compile running: nothing to wait for
            (_snap("c2", "compile_epub", JobState.RUNNING), (_snap("o", "translate", JobState.QUEUED),), None, None),
            # this book's QA scan is queued: only Run QA scan waits
            (None, (_snap("qa", "qa_scan", JobState.QUEUED, origin=mine),), None, QA_RUNNING),
        ]
        for active, queue, run_reason, qa_reason in cases:
            jobs.publish(active, queue)  # a jobs event: no reopen, no reload
            assert state() == (run_reason is not None, run_reason, qa_reason is not None, qa_reason), active
        # tapping a waiting Run translation explains, opens nothing
        jobs.publish(_snap("t", "translate", JobState.RUNNING), ())
        opened = []

        async def open_translate():
            opened.append(True)

        screen.open_translate = open_translate
        assert await chapters.run_translation() is None and opened == []
        assert _messages(ctx)[-1] == bp.RUN_WAIT_TRANSLATION
        # the run ends: both come back at once; a QA scan running does not block Run translation
        jobs.publish(None, ())
        assert state() == (False, None, False, None)
        jobs.publish(_snap("other-qa", "qa_scan", JobState.RUNNING), ())
        assert await chapters.run_translation() is None and opened == [True]
        screen.dispose()

    asyncio.run(scenario())


class _Backend:
    """Minimal ``services.jobs.JobBackend`` stand-in (no shared pipeline)."""

    def __init__(self):
        self.job_lock = threading.RLock()

    def scoped_process_state(self, *, capture_stdout=None, lock, **_kw):
        @contextlib.contextmanager
        def scope():
            with lock:
                yield

        return scope()

    def make_owner(self, config, *, host, **_kw):
        return types.SimpleNamespace(stop_requested=False, graceful_stop_var=True, wait_for_chunks_var=True,
                                     config=config, host=host)

    def reset_for_new_run(self, kind="translation", **_kw):
        return 1

    def request_stop(self, *, graceful, wait_for_chunks, force, set_stop_requested, log=None, **_kw):
        set_stop_requested()

    def progress_watcher(self, resolver, host, interval=2.0, **_kw):
        return None


@needs_flet
def test_real_job_service_disables_while_the_books_translation_runs_and_re_enables_when_it_ends(real_env, tmp_path):
    """The live path with the real JobService: the Book page's own jobs subscription (did_show) disables both
    buttons when this book's translation starts and enables them again the moment it ends."""
    _needs_cores()
    from glossarion_mobile import job_kinds
    from glossarion_mobile.services.jobs import JobService, JobSpec
    from glossarion_mobile.ui.library import book_page as bp

    release = threading.Event()
    entered = threading.Event()

    def run(ctx):
        entered.set()
        release.wait(TIMEOUT)

    def kinds(kind):
        return job_kinds.KindInfo(kind=kind, verb="Translating", icon="TRANSLATE", stop_kind="translation", run=run)

    jobs = JobService(jobs_dir=str(tmp_path / "jobs"), backend=_Backend(), kinds=kinds, config_loader=lambda: {})

    async def scenario():
        conn, screen, ctx, book = await _open_book(real_env, jobs)
        screen.did_show()
        chapters = screen.chapters
        try:
            assert not chapters.run_button.disabled and not chapters.qa_button.disabled
            service = real_env["service"]
            jobs.submit(JobSpec(kind="translate", title="Book", inputs=(real_env["raw"],),
                                origin=service.origin_for(book)))
            assert await _until(entered.is_set)
            assert await _until(lambda: chapters.run_button.disabled and chapters.qa_button.disabled)
            assert chapters.run_button.tooltip == bp.RUN_WAIT_TRANSLATION
            release.set()
            assert await _until(lambda: not chapters.run_button.disabled and not chapters.qa_button.disabled)
            assert chapters.run_button.tooltip is None and chapters.qa_button.tooltip is None
        finally:
            release.set()
            screen.dispose()
            jobs.close()

    asyncio.run(scenario())


def test_job_matching_keeps_job_for_book_and_lists_every_job_of_the_book():
    from glossarion_mobile.services.jobs import JobsView, JobState
    from glossarion_mobile.ui.library import book_page as bp

    book = {"name": "B", "output_folder": os.path.abspath("ws-b"), "raw_source_path": os.path.abspath("b.epub")}
    active = _snap("a", "compile_epub", JobState.RUNNING)
    by_raw = _snap("r", "translate", JobState.QUEUED, inputs=(os.path.abspath("b.epub"),))
    by_bid = _snap("i", "qa_scan", JobState.QUEUED, origin={"type": "library", "bid": "B1"})
    view = JobsView(active=active, queue=(by_raw, _snap("x", "translate", JobState.QUEUED), by_bid))
    assert [s.id for s in bp.jobs_for_book(view, "B1", book)] == ["r", "i"]
    assert bp.job_for_book(view, "B1", book).id == "r"
    assert bp.job_for_book(view, "B1", book, include_queued=False) is None
    assert bp.job_for_book(None, "B1", book) is None and bp.jobs_for_book(None, "B1", book) == []
    assert bp.run_wait_reason(view, "B1", book) == bp.RUN_WAIT_QUEUED_TRANSLATION
    assert bp.run_wait_reason(JobsView(active=active, queue=(by_bid,)), "B1", book) is None


def test_resolve_qa_and_insert_image_results_carry_the_desktop_box_titles():
    """Review finding: the result sheet's title is the desktop box's for that outcome - Resolve QA's "QA Issue
    Resolved" / "QA Issue Not Resolved" / "QA Issue Not Fully Resolved" (Retranslation_GUI), Insert Missing Image
    the shared ``insert_missing_images`` title (desktop ``_show_message(kind, title, message)``) - never a menu label."""
    from glossarion_mobile.ui.library import chapters_tab
    from glossarion_mobile.ui.library import progress_model as pm

    class Actions:
        outcome: dict = {}
        inserted: tuple = ()

        def resolve_llm_token_qa(self, progress_file, target, output_path):
            return self.outcome

        def llm_token_repair_summary(self, outcome, output_path):
            return "No malformed empty-attribute tags remain; the stale LLM token QA mark was cleared.\n\n" \
                   "The QA issue has been resolved."

        def insert_missing_images(self, data, target, restore_fn):
            return self.inserted

    actions = Actions()
    service = types.SimpleNamespace(core=types.SimpleNamespace(module=lambda name: actions), mark_dirty=lambda: None)
    view = types.SimpleNamespace(state=types.SimpleNamespace(data={"progress_file": "p.json", "output_dir": "out"},
                                                             owner=None))
    resolve = pm.ActionPlan("resolve_qa", [{"key": "1"}], 1, extra={"output_path": "out/response_1.html"})
    cases = [
        ({"repair": {"resolved": False, "error": "still there"}}, "QA Issue Not Resolved", "still there"),
        ({"repair": {"resolved": True}, "error": "disk full"}, "QA Issue Not Fully Resolved",
         "The malformed tags were repaired, but the progress file could not be updated:\ndisk full"),
        ({"repair": {"resolved": True, "repaired": 0}}, "QA Issue Resolved", None),
    ]
    for outcome, title, text in cases:
        actions.outcome = outcome
        result = pm.apply_action(service, view, resolve)
        assert isinstance(result, pm.TitledResult) and result.box_title == title, (outcome, result)
        if text is not None:
            assert str(result) == text
    for shared in (("error", "Error", "Failed to restore images: boom", False), ("info", "Success", "Restored 2", True)):
        actions.inserted = shared
        result = pm.apply_action(service, view, pm.ActionPlan("insert_image", [{"key": "1"}], 1))
        assert (result.box_title, str(result)) == (shared[1], shared[2])
    # the table holds only real desktop titles; these two take theirs from the result
    assert "resolve_qa" not in chapters_tab.ACTION_RESULT_TITLES and "insert_image" not in chapters_tab.ACTION_RESULT_TITLES
