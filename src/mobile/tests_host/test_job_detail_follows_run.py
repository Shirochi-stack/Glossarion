"""Owner report: after stopping a translation and starting a new run of the same book, Back to that job's page (Job
detail, ``/jobs/<jid>``) still showed the stopped run (its log ending in "Stopped", its requests) while the job bar
showed the new run.

A Job page that is open (shown, or kept under another page) follows the run that carries its job on
(``JobService.carried_into``: a run of the book that does that work, or its Resume, followed to the newest): snapshot,
title, requests and the live log. A QA scan or a compile of the book does not take it over, and a page opened by id on
an entry that was already carried on still shows that entry (``test_m11_job_book_entries``).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("GLOSSARION_HTTP_LOG", "0")
HERE = Path(__file__).resolve().parent
for path in (HERE.parent / "app", HERE.parent.parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from glossarion_mobile.services.jobs import JobSnapshot, JobSpec, JobState, Progress  # noqa: E402
from test_jobs import TIMEOUT, Gate, epub, make_service, wait_for  # noqa: E402

pytest.importorskip("flet")

TITLE = "A Fallen Necromancer Returns as a Hero"


@pytest.fixture(autouse=True)
def scratch_env(tmp_path, monkeypatch):
    for name in ("HOME", "USERPROFILE", "OUTPUT_DIRECTORY", "GLOSSARION_LIBRARY_DIR"):
        monkeypatch.setenv(name, str(tmp_path / "scratch_home"))


class LogDispatcher:
    """``subscribe_log`` with a pump the test drives (the app's pump runs on the UI loop)."""

    bound = False

    def __init__(self) -> None:
        self.subs: list = []

    def on_loop_thread(self) -> bool:
        return True

    def mark_dirty(self, *controls) -> None:
        pass

    def subscribe_log(self, buffer, callback, *, backlog=0, batch=None):
        sub = [buffer, callback, max(0, buffer.last_seq - max(0, backlog))]
        self.subs.append(sub)

        def unsubscribe():
            if sub in self.subs:
                self.subs.remove(sub)

        return unsubscribe

    def pump(self) -> None:
        for sub in list(self.subs):
            lines, gap = sub[0].since(sub[2])
            if lines or gap:
                sub[2] = lines[-1].seq if lines else sub[0].last_seq
                sub[1](lines, gap)


def _library(source, bid="b1"):
    return JobSpec("translate", TITLE, (source,), origin={"type": "library", "bid": bid, "label": f"Library · {TITLE}"})


def _ended(job_id, spec, state=JobState.CANCELLED, **fields):
    return JobSnapshot(id=job_id, spec=spec, state=state, created=1.0, started=1.0, finished=2.0,
                       progress=Progress(total=10, completed=4), **fields)


def _store(service, *entries):
    with service._lock:
        service._history = list(entries)
        service._write_history_locked()


def _open(service, job_id, dispatcher=None):
    from glossarion_mobile.ui.router import parse_route
    from glossarion_mobile.ui.screens.job_detail import JobDetailScreen

    detail = JobDetailScreen(parse_route(f"/jobs/{job_id}"), service=service, dispatcher=dispatcher)
    detail.get_body()
    detail.console.attach()  # what mounting it does
    detail.did_show()
    return detail


def _requests_by_job(service):
    """Every job has one request card naming it."""
    service.request_segments = lambda job_id: [{"label": f"request of {job_id}", "text": f"text of {job_id}"}]


def test_carried_into_follows_the_newest_live_run_of_the_same_work(tmp_path):
    source = epub(tmp_path)
    service, _ = make_service(tmp_path)
    try:
        _store(service, _ended("old", _library(source)))
        assert service.carried_into("old") is None
        service.pause_queue()
        new = service.submit(_library(source))
        assert service.carried_into("old") == new
        assert service.carried_into(new) is None
        service.cancel_queued(new)  # it never ran: the old entry is the book's entry again
        assert service.carried_into("old") is None
        assert service.carried_into("never-existed") is None
    finally:
        service.close()


def test_a_stopped_runs_page_follows_the_new_run_of_its_book(tmp_path):
    source = epub(tmp_path)
    service, backend = make_service(tmp_path)
    dispatcher = LogDispatcher()
    _requests_by_job(service)
    try:
        gate = Gate()
        backend.behavior = gate
        first = service.submit(_library(source))
        assert gate.entered.wait(TIMEOUT * 24)  # a first run in the process imports the shared translator
        service.log_buffer(first).append("line of the first run")
        detail = _open(service, first, dispatcher)
        dispatcher.pump()
        assert "line of the first run" in detail.console.visible_text()
        detail._on_stop()  # Stop, then Force stop (the graceful stop would wait for in-flight chunks)
        assert wait_for(lambda: service.snapshot(first).state is JobState.STOPPING)
        detail._apply(service.snapshot(first))
        detail._on_stop()
        assert wait_for(lambda: service.snapshot(first).state is JobState.CANCELLED)
        assert wait_for(lambda: detail.snap.state is JobState.CANCELLED)

        service.pause_queue()  # the new run waits: the page follows it from its queued state on
        second = service.submit(_library(source))
        assert wait_for(lambda: detail.job_id == second)
        assert detail.snap.id == second and detail.title_text.value == TITLE
        assert detail.snap.state is JobState.QUEUED and detail.stop_button.content == "Cancel"
        assert "line of the first run" not in detail.console.visible_text()
        assert detail.requests_title.value == "Requests (1)"
        assert f"request of {second}" in str(detail.requests_column.controls[0].content.controls[0].controls[0].value)

        service.resume_queue()  # it starts: the page shows its log buffer
        assert wait_for(lambda: service.log_buffer(second) is not None)
        assert wait_for(lambda: detail.console.buffer is service.log_buffer(second))
        service.log_buffer(second).append("line of the second run")
        dispatcher.pump()
        text = detail.console.visible_text()
        assert "line of the second run" in text and "line of the first run" not in text
        # one console and one requests subscription, both on the new run's buffer
        assert len(dispatcher.subs) == 2 and all(sub[0] is service.log_buffer(second) for sub in dispatcher.subs)

        service.log_buffer(first).append("late line of the first run")
        dispatcher.pump()
        assert "late line of the first run" not in detail.console.visible_text()
        detail.dispose()
        detail.console.detach()
        assert dispatcher.subs == []
        gate.release.set()
        assert service.wait_idle(TIMEOUT * 4)
    finally:
        gate.release.set()
        service.close()


def test_the_page_follows_a_chain_of_runs(tmp_path):
    source = epub(tmp_path)
    service, _ = make_service(tmp_path)
    try:
        _store(service, _ended("old", _library(source)))
        detail = _open(service, "old")
        service.pause_queue()
        second = service.submit(_library(source))
        assert wait_for(lambda: detail.job_id == second)
        service.cancel_queued(second)  # cancelled while queued: it never ran
        assert wait_for(lambda: service.snapshot(second).is_terminal)
        third = service.submit(_library(source))
        assert wait_for(lambda: detail.job_id == third), detail.job_id
        assert detail.snap.id == third
        detail.dispose()
        service.cancel_queued(third)
    finally:
        service.close()


@pytest.mark.parametrize("start", ["qa", "compile"])
def test_a_qa_scan_or_compile_of_the_book_does_not_take_the_page_over(tmp_path, start):
    source, workspace = epub(tmp_path), str(tmp_path / "out" / "Book")
    service, _ = make_service(tmp_path)
    try:
        _store(service, _ended("old", _library(source), output_dir=workspace))
        detail = _open(service, "old")
        service.pause_queue()
        spec = {
            "qa": JobSpec("qa_scan", TITLE, (workspace,), params={"mode": "quick-scan",
                                                                 "targets": [{"folder": workspace, "source": source}]}),
            "compile": JobSpec("compile_epub", TITLE, params={"folder": workspace}),
        }[start]
        other = service.submit(spec)
        assert "old" not in [s.id for s in service.view().history]  # the book's entry was carried on
        assert service.carried_into("old") is None
        service._changed()
        assert detail.job_id == "old" and detail.snap.id == "old"
        detail.dispose()
        service.cancel_queued(other)
    finally:
        service.close()


def test_a_new_run_takes_the_entry_over_from_a_qa_scan_of_the_book(tmp_path):
    """Review finding: a QA scan of the book carried the stopped entry first (it does not do its work); a translation
    started while the scan is queued or running still takes the entry (and its open page) over."""
    source, workspace = epub(tmp_path), str(tmp_path / "out" / "Book")
    service, _ = make_service(tmp_path)
    try:
        _store(service, _ended("old", _library(source), output_dir=workspace))
        detail = _open(service, "old")
        service.pause_queue()
        scan = service.submit(JobSpec("qa_scan", TITLE, (workspace,), params={
            "mode": "quick-scan", "targets": [{"folder": workspace, "source": source}]}))
        assert service.carried_into("old") is None
        new = service.submit(_library(source))
        assert service.carried_into("old") == new
        assert wait_for(lambda: detail.job_id == new)
        detail.dispose()
        service.cancel_queued(scan)  # the scan ends: the entry stays with the run that does its work
        assert service.carried_into("old") == new
        assert "old" not in [s.id for s in service.view().history]
        service.cancel_queued(new)
    finally:
        service.close()


def test_a_finished_runs_page_stays_on_it(tmp_path):
    """A Done run's page keeps showing it while a new run fixes its chapters (``test_m11_qa_count_live``)."""
    source = epub(tmp_path)
    service, _ = make_service(tmp_path)
    try:
        _store(service, _ended("old", _library(source), JobState.DONE))
        detail = _open(service, "old")
        service.pause_queue()
        new = service.submit(_library(source))
        assert service.carried_into("old") == new
        assert detail.job_id == "old" and detail.snap.id == "old"
        detail.dispose()
        service.cancel_queued(new)
    finally:
        service.close()


def test_a_page_opened_by_id_on_a_carried_on_entry_stays_on_it(tmp_path):
    source = epub(tmp_path)
    service, _ = make_service(tmp_path)
    try:
        _store(service, _ended("old", _library(source)))
        service.pause_queue()
        new = service.submit(_library(source))
        detail = _open(service, "old")
        service._changed()
        assert detail.job_id == "old" and detail.snap.id == "old" and not detail.resume_button.visible
        detail.dispose()
        service.cancel_queued(new)
    finally:
        service.close()
