"""Owner issue 2 (mobile batch 9.15.11): "The QA failed labels in jobs doesn't update when said failed chapters are in
progress."

The number comes from one place, ``library_core._read_progress_summary`` over the book's ``translation_progress.json``
(a chapter ``in_progress`` is not counted as failed). A running job follows it through its ``ProgressWatcher``, but a
finished job's saved counts only followed when another job ended, or when its Job page was opened, and a job without
an output folder of its own (``retranslate``) never moved them at all. These tests run the real watcher and the real
summary over a real progress file.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
from pathlib import Path

import pytest

os.environ.setdefault("GLOSSARION_HTTP_LOG", "0")
HERE = Path(__file__).resolve().parent
for path in (HERE.parent / "app", HERE.parent.parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from glossarion_mobile.services.jobs import JobSpec, JobState, progress_line  # noqa: E402
from test_jobs import TIMEOUT, epub, make_service, wait_for  # noqa: E402


@pytest.fixture(autouse=True)
def scratch_env(tmp_path, monkeypatch):
    for name in ("HOME", "USERPROFILE", "OUTPUT_DIRECTORY", "GLOSSARION_LIBRARY_DIR"):
        monkeypatch.setenv(name, str(tmp_path / "scratch_home"))


def _marks(folder, statuses) -> None:
    """A real progress file; ``statuses``: one status per chapter."""
    os.makedirs(folder, exist_ok=True)
    rows = {}
    for index, status in enumerate(statuses, 1):
        output = f"response_{index:03d}.html"
        Path(folder, output).write_text("<p>done</p>", encoding="utf-8")
        rows[str(index)] = {"actual_num": index, "output_file": output, "content_hash": f"h{index}", "status": status}
    Path(folder, "translation_progress.json").write_text(json.dumps({"chapters": rows, "chapter_chunks": {}}),
                                                         encoding="utf-8")


def _live_service(tmp_path, **kwargs):
    """A service whose jobs run the real ``ProgressWatcher`` and summary, polling fast."""
    import job_runner

    service, backend = make_service(tmp_path, **kwargs)
    backend.progress_summary = job_runner._default_summary_reader
    backend.progress_watcher = lambda resolver, host, interval: job_runner.ProgressWatcher(
        resolver, host, interval=0.02, summary_reader=job_runner._default_summary_reader,
        watchdog_state=lambda: None)
    return service, backend


class Step:
    """A worker that runs each of ``steps`` (progress file changes) only when the test says so."""

    def __init__(self, folder_seen) -> None:
        self.folder_seen = folder_seen
        self.ready = threading.Event()
        self.go = threading.Event()
        self.done = threading.Event()
        self.release = threading.Event()

    def __call__(self, owner, request):
        self.folder_seen["path"] = owner._resolve_translation_output_dir(request["files"][0])
        self.ready.set()
        assert self.go.wait(TIMEOUT)
        self.done.set()
        assert self.release.wait(TIMEOUT)


def _first_job(service, backend, source, folder):
    """A finished translation: 2 completed, 4 QA failed."""

    def translate(owner, request):
        folder["path"] = owner._resolve_translation_output_dir(request["files"][0])
        _marks(folder["path"], ["completed"] * 2 + ["qa_failed"] * 4)

    backend.behavior = translate
    job_id = service.submit(JobSpec("translate", "Book", (source,)))
    assert service.wait_idle(TIMEOUT) and service.snapshot(job_id).state is JobState.DONE
    assert wait_for(lambda: service.snapshot(job_id).progress.failed == 4), service.snapshot(job_id).progress
    return job_id


def test_the_pipeline_taking_a_chapter_up_stops_counting_it_in_the_summary(tmp_path):
    """The source of truth: the real ProgressManager's in-progress mark drops the chapter from the failed count."""
    from library_core import _read_progress_summary
    from TransateKRtoEN import ProgressManager

    _marks(tmp_path, ["completed", "qa_failed", "qa_failed"])
    assert _read_progress_summary(str(tmp_path / "translation_progress.json"))["failed"] == 2
    manager = ProgressManager(str(tmp_path))
    manager.update(1, 2, "h2", "response_002.html", status="in_progress")
    manager.save()
    summary = _read_progress_summary(str(tmp_path / "translation_progress.json"))
    assert (summary["failed"], summary["in_progress"]) == (1, 1), summary


def test_the_running_jobs_line_follows_the_chapters_it_takes_up(tmp_path):
    service, backend = _live_service(tmp_path)
    source = epub(tmp_path)
    folder: dict = {}
    _first_job(service, backend, source, folder)

    step = Step(folder)
    backend.behavior = step
    second = service.submit(JobSpec("translate", "Book", (source,)))
    assert step.ready.wait(TIMEOUT)
    assert wait_for(lambda: service.snapshot(second).progress.failed == 4), service.snapshot(second).progress
    _marks(folder["path"], ["completed"] * 2 + ["in_progress"] + ["qa_failed"] * 3)
    assert wait_for(lambda: service.snapshot(second).progress.failed == 3), service.snapshot(second).progress
    assert "3 failed" in progress_line(service.snapshot(second), verb=True)
    step.go.set()
    step.release.set()
    assert service.wait_idle(TIMEOUT)
    service.close()


def test_a_finished_jobs_count_follows_while_a_new_job_takes_its_chapters_up(tmp_path):
    """The reported bug: "4 QA failed" on the finished job stayed until the new job ended."""
    from glossarion_mobile.ui.screens.job_detail import JobDetailScreen
    from glossarion_mobile.ui.router import parse_route

    service, backend = _live_service(tmp_path)
    source = epub(tmp_path)
    folder: dict = {}
    first = _first_job(service, backend, source, folder)
    detail = JobDetailScreen(parse_route(f"/jobs/{first}"), service=service)
    detail.get_body()
    assert detail.qa_chip.label.value == "4 QA failed"

    step = Step(folder)
    backend.behavior = step
    service.submit(JobSpec("translate", "Book", (source,)))
    assert step.ready.wait(TIMEOUT)
    _marks(folder["path"], ["completed"] * 2 + ["in_progress"] * 2 + ["qa_failed"] * 2)
    assert wait_for(lambda: service.snapshot(first).progress.failed == 2), service.snapshot(first).progress
    # the Job page still open on the old job shows it: its view delivery reaches it
    detail._on_view(service.view())
    assert detail.qa_chip.label.value == "2 QA failed" and detail.qa_chip.visible
    assert "2 failed" in detail.progress_text.value
    # and a chapter that finishes moves it on, down to none
    _marks(folder["path"], ["completed"] * 6)
    assert wait_for(lambda: service.snapshot(first).progress.failed == 0), service.snapshot(first).progress
    detail._on_view(service.view())
    assert not detail.qa_chip.visible
    step.go.set()
    step.release.set()
    assert service.wait_idle(TIMEOUT)
    service.close()
    reloaded, _ = make_service(tmp_path)
    assert reloaded.snapshot(first).progress.failed == 0  # what it showed last survives a restart
    reloaded.close()


def test_a_reader_retranslation_moves_the_same_number(tmp_path):
    """The Reader's and the Book page's "Translate this chapter": a ``single_chapter`` job on the same source."""
    service, backend = _live_service(tmp_path)
    source = epub(tmp_path)
    folder: dict = {}
    first = _first_job(service, backend, source, folder)
    step = Step(folder)
    backend.behavior = step
    chapter = JobSpec("single_chapter", "Book · chapter0003.xhtml", (source,),
                      params={"chapter_file": "chapter0003.xhtml", "force_stream_all": True},
                      origin={"type": "library", "bid": "b1", "label": "Reader · Book"})
    service.submit(chapter)
    assert step.ready.wait(TIMEOUT)
    _marks(folder["path"], ["completed"] * 2 + ["in_progress"] + ["qa_failed"] * 3)
    assert wait_for(lambda: service.snapshot(first).progress.failed == 3), service.snapshot(first).progress
    step.go.set()
    step.release.set()
    assert service.wait_idle(TIMEOUT)
    service.close()


def test_a_retranslate_job_without_an_output_folder_moves_the_number(tmp_path):
    """Retranslate Selected (Book page, Progress manager) names its workspace only in its params: when it ended,
    nothing refreshed the book's finished entries."""
    from glossarion_mobile import job_kinds

    folder: dict = {}

    def reset(ctx):
        _marks(folder["path"], ["completed"] * 2 + ["pending"] * 4)
        return {"ok": True}

    def kinds(kind):
        if kind == "retranslate":
            return job_kinds.KindInfo(kind=kind, verb="Resetting", icon="TRANSLATE", stop_kind="translation",
                                      run=reset, resumable=False)
        return job_kinds.get_kind(kind)

    service, backend = _live_service(tmp_path, kinds=kinds)
    source = epub(tmp_path)
    first = _first_job(service, backend, source, folder)
    second = service.submit(JobSpec("retranslate", "Book · 4 selected", params={"output_dir": folder["path"]},
                                    resumable=False))
    assert service.wait_idle(TIMEOUT) and service.snapshot(second).state is JobState.DONE
    assert wait_for(lambda: service.snapshot(first).progress.failed == 0), service.snapshot(first).progress
    assert service.snapshot(first).progress.in_progress == 4
    service.close()


def test_opening_the_jobs_page_reads_marks_changed_without_a_job(tmp_path):
    """A Progress manager edit runs no job: the rows follow when the Jobs page shows, off the UI loop."""
    from glossarion_mobile.ui.router import parse_route
    from glossarion_mobile.ui.screens.jobs import JobsScreen

    service, backend = _live_service(tmp_path)
    source = epub(tmp_path)
    folder: dict = {}
    first = _first_job(service, backend, source, folder)
    _marks(folder["path"], ["completed"] * 5 + ["qa_failed"])

    async def scenario():
        screen = JobsScreen(parse_route("/jobs"), service=service)
        screen.get_body()
        on_loop = threading.get_ident()
        threads = []
        reader = backend.progress_summary
        backend.progress_summary = lambda path: threads.append(threading.get_ident()) or reader(path)
        screen.did_show()
        assert await asyncio.to_thread(wait_for, lambda: service.snapshot(first).progress.failed == 1)
        assert threads and on_loop not in threads
        await asyncio.sleep(0)
        screen._on_view(service.view())
        screen.dispose()

    asyncio.run(scenario())
    assert service.view().find(first).progress.completed == 5
    service.close()
