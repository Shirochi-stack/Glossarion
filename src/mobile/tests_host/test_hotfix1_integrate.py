"""Hotfix 1 integration: the chat's UI meets the job-safety JobService API (owner issues 4 and 5).

* "Save to: Library" (Plan card Start) of a book that already has an active job queues nothing, says where it
  runs ("Translating “Book.epub” is already queued (…)" · Open) and keeps the plan, so Start can be tapped again
  - on the real JobService, through the chat's ``JobsAdapter``;
* the chat's QA scan of a workspace whose scan is already active finds that scan before a second "QA scan" card
  is recorded (``active_job_of`` with the chat's own ``qa_model.chat_qa_job`` parts);
* a Send / Resume refused with ``JobAlreadyActive`` (another chat or the Library runs the book) says so with Open
  (that job's page); any other error keeps "Could not start: …";
* ``ChatRuns.resubmit`` takes the Result card's turn (``user_index``), which ``ChatView`` feature-detects.

Run from src/mobile with the mobile venv:
    .venv/Scripts/python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic \\
        tests_host/test_hotfix1_integrate.py
"""

from __future__ import annotations

import asyncio
import importlib.util
import inspect
import sys
import types
from pathlib import Path

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = MOBILE_DIR / "app"
SRC_DIR = MOBILE_DIR.parent
for entry in (str(APP_DIR), str(SRC_DIR)):
    if entry not in sys.path:
        sys.path.insert(0, entry)


def _has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


needs_flet = pytest.mark.skipif(not (_has("flet") and _has("msgpack")), reason="flet / msgpack not installed")

_JS_SPEC = importlib.util.spec_from_file_location("_glossarion_hotfix1_integrate_js",
                                                  Path(__file__).with_name("test_hotfix1_job_safety.py"))
JS = importlib.util.module_from_spec(_JS_SPEC)
_JS_SPEC.loader.exec_module(JS)

isolated = JS.isolated  # fixture: HOME / Library / output / data in tmp_path


def _fake_view(jobs, notes, navigated, metas):
    """The parts of ``ChatView`` that ``_submit_library`` uses (as test_u9_gap_closure's duck-typed view)."""
    chats = types.SimpleNamespace(session=lambda cid: {"title": "Novel"}, append_messages=lambda *a, **k: None,
                                  set_meta=lambda cid, key, value: metas.append((cid, key, value)))
    return types.SimpleNamespace(env=types.SimpleNamespace(jobs=jobs, chats=chats), render_transcript=lambda **k: None,
                                 notify=lambda *a: notes.append(a),
                                 navigate=lambda *a: navigated.append(a), run_overrides=lambda cid: {})


@needs_flet
def test_save_to_library_of_an_active_book_queues_nothing_and_says_where_it_runs(isolated):
    from glossarion_mobile.ui.chat.chat_view import ChatView
    from glossarion_mobile.ui.chat.job_binding import JobsAdapter

    service, kinds = JS.make_service(isolated)
    try:
        path = JS.book(isolated)
        notes, navigated, metas = [], [], []
        view = _fake_view(JobsAdapter(service), notes, navigated, metas)
        record = {"path": path, "name": "Book.epub"}
        first = asyncio.run(ChatView._submit_library(view, "c1", {"output_mode": "text"}, dict(record), {}))
        assert first and JS.wait_for(lambda: kinds.started.is_set())
        assert not notes  # the first Start says nothing (the strip shows the job)
        plan = {"output_mode": "text", "created": 2}
        again = asyncio.run(ChatView._submit_library(view, "c1", plan, dict(record), {}))
        assert again == first
        assert len(JS._live_jobs(service)) == 1  # nothing new was queued
        text, label, open_job = notes[-1]
        assert "Book.epub" in text and "already" in text and label == "Open"
        open_job()
        assert navigated[-1] == ("jobs.detail", {"jid": str(first)})
        assert ("c1", "pending_plan", plan) in metas  # the plan stays: Start can be tapped again
    finally:
        kinds.release.set()
        service.wait_idle(JS.TIMEOUT)
        service.close()


def test_the_chat_qa_scan_finds_the_workspaces_active_scan(isolated):
    from glossarion_mobile.ui.chat.chat_view import active_job_of
    from glossarion_mobile.ui.chat.job_binding import JobsAdapter
    from glossarion_mobile.ui.tools import qa_model

    workspace = isolated / "output" / "Book"
    workspace.mkdir(parents=True)
    job = qa_model.chat_qa_job(str(workspace), "", cid="c1", chat_title="Novel")
    if isinstance(job, str):
        pytest.skip(job)  # the QA scanner is not importable here
    service, kinds = JS.make_service(isolated)
    try:
        adapter = JobsAdapter(service)
        assert active_job_of(adapter, *job) is None
        job_id = asyncio.run(adapter.submit(*job))
        assert JS.wait_for(lambda: kinds.started.is_set())
        found = active_job_of(adapter, *job)
        assert found is not None and found.id == job_id
        # a jobs object without the service API (an older adapter, a test fake) cannot tell: None
        assert active_job_of(types.SimpleNamespace(), *job) is None
    finally:
        kinds.release.set()
        service.wait_idle(JS.TIMEOUT)
        service.close()


def test_a_refused_send_or_resume_names_the_running_job():
    from glossarion_mobile.services.jobs import JobAlreadyActive
    from glossarion_mobile.ui.chat.chat_view import notify_not_started

    notes, navigated = [], []
    snapshot = types.SimpleNamespace(id="j9", state="RUNNING",
                                     spec=types.SimpleNamespace(kind="translate", title="Book.epub",
                                                                origin={"label": "Library · Book"}))
    notify_not_started(lambda *a: notes.append(a), lambda *a: navigated.append(a), JobAlreadyActive(snapshot), "x")
    text, label, open_job = notes[-1]
    assert text == str(JobAlreadyActive(snapshot)) and "Book.epub" in text and label == "Open"
    open_job()
    assert navigated == [("jobs.detail", {"jid": "j9"})]
    # another chat is still preparing a send of the file: no job to open yet
    notify_not_started(lambda *a: notes.append(a), lambda *a: None, JobAlreadyActive(None, "Busy elsewhere"), "x")
    assert notes[-1] == ("Busy elsewhere",)
    notify_not_started(lambda *a: notes.append(a), lambda *a: None, RuntimeError("boom"), "Could not start")
    assert notes[-1] == ("Could not start: boom",)


def test_resubmit_takes_the_cards_turn():
    from glossarion_mobile.ui.chat.run_controller import ChatRuns

    assert "user_index" in inspect.signature(ChatRuns.resubmit).parameters
    assert "user_index" in inspect.signature(ChatRuns._resubmit).parameters
    assert "user_index" in inspect.signature(ChatRuns._last_job).parameters
