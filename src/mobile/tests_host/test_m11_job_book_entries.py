"""Owner issue 3 (mobile batch 9.15.11): "I found some Jobs using the title of the html file instead of the book title
for the label. This happens when I clicked run translation after QA failed. I somehow have 3 separate finished jobs.
Old finished jobs for the same attachment should also be auto cleared when another in progress job for it is
present."

(a) A job's label is its book's title, never a chapter file: ``JobSpec`` drops one, also from entries stored by an
    older build.
(b) A book has one entry under Finished: while a job for it is queued or running none of the book's ended entries is
    listed. The book is told by its source, workspace or Library id (``job_books``), never by the title.
(c) What was listed stays readable by its id (``JobsView.carried_on``): chat cards, open-job links and notifications
    keep working, also after a restart.
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

from glossarion_mobile.services.jobs import (  # noqa: E402
    FILE_TITLE_EXTENSIONS, JOB_TARGET_PARAM, JobService, JobSnapshot, JobSpec, JobState, Progress, job_books,
    same_books,
)
from test_jobs import TIMEOUT, Gate, epub, make_service, wait_for  # noqa: E402
from test_reader import _screen, cores, novel  # noqa: E402,F401  (the Reader's own fixtures)

TITLE = "A Fallen Necromancer Returns as a Hero"


@pytest.fixture(autouse=True)
def scratch_env(tmp_path, monkeypatch):
    for name in ("HOME", "USERPROFILE", "OUTPUT_DIRECTORY", "GLOSSARION_LIBRARY_DIR"):
        monkeypatch.setenv(name, str(tmp_path / "scratch_home"))


def _spec(kind, title, *inputs, **fields):
    return JobSpec(kind, title, tuple(inputs), **fields)


def _ended(job_id, spec, state=JobState.DONE, created=1.0, **fields):
    fields.setdefault("progress", Progress(total=177, completed=168, failed=9))
    return JobSnapshot(id=job_id, spec=spec, state=state, created=created, started=created, finished=created + 1,
                       **fields)


def _library(source, bid="b1"):
    return _spec("translate", TITLE, source, origin={"type": "library", "bid": bid, "label": f"Library · {TITLE}"})


def _reader(source, chapter="chapter0004.xhtml", bid="b1"):
    return _spec("single_chapter", f"{TITLE} · {chapter}", source, params={"chapter_file": chapter},
                 origin={"type": "library", "bid": bid, "label": f"Reader · {TITLE}"})


def _store(service, *entries):
    with service._lock:
        service._history = list(entries)
        service._write_history_locked()


def _ids(service):
    return [s.id for s in service.view().history]


# ---- (a) the label --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("title, kind, inputs, params, expected", [
    # the Reader's and the Book page's chapter job
    (f"{TITLE} · chapter0004.xhtml", "single_chapter", ("/b/raw.epub",), {"chapter_file": "chapter0004.xhtml"}, TITLE),
    (f"{TITLE} · OEBPS/ch7.html", "single_chapter", ("/b/raw.epub",), {"chapter_file": "OEBPS/ch7.html"}, TITLE),
    # a book without a name: its source file's name
    (" · chapter0004.xhtml", "single_chapter", ("/b/raw.epub",), {"chapter_file": "chapter0004.xhtml"}, "raw"),
    ("chapter0004.xhtml", "single_chapter", ("/b/raw.epub",), {"chapter_file": "chapter0004.xhtml"}, "raw"),
    # Resolve QA of a chapter row: its output file was the label
    (f"{TITLE} · response_0004.html", "resolve_qa", ("/b/raw.epub",), {}, TITLE),
    ("response_0004.html", "resolve_qa", ("/b/raw.epub",), {}, "raw"),
    # other labels stay
    (f"{TITLE} · Table of contents", "resolve_qa", ("/b/raw.epub",), {}, f"{TITLE} · Table of contents"),
    (f"{TITLE} · 3 selected", "retranslate", (), {}, f"{TITLE} · 3 selected"),
    # a chat's attachment is its file until the metadata translation names it
    ("novel.html", "translate", ("/b/novel.html",), {}, "novel.html"),
])
def test_the_label_never_names_a_chapter_file(title, kind, inputs, params, expected):
    assert JobSpec(kind, title, inputs, params=params).title == expected


def test_entries_stored_by_an_older_build_read_the_books_title(tmp_path):
    source = epub(tmp_path)
    service, _ = make_service(tmp_path)
    stored = _ended("r1", _spec("single_chapter", "x", source, params={"chapter_file": "chapter0004.xhtml"})).to_dict()
    stored["spec"]["title"] = f"{TITLE} · chapter0004.xhtml"  # as the older build wrote it
    Path(service.history_state_path).write_text(json.dumps({"version": 1, "jobs": [stored]}), encoding="utf-8")
    service.close()
    reloaded, _ = make_service(tmp_path)
    assert reloaded.view().history[0].title == TITLE
    reloaded.close()


def test_a_file_named_job_takes_its_translated_title_also_for_xhtml(tmp_path):
    assert ".xhtml" in FILE_TITLE_EXTENSIONS
    (tmp_path / "metadata.json").write_text(json.dumps({"title": TITLE, "title_translated": True}), encoding="utf-8")
    job = type("J", (), {"output_dir": str(tmp_path), "metadata_mtime": 0.0, "id": "j",
                         "spec": JobSpec("translate", "novel.xhtml", ("/b/novel.xhtml",))})()
    fake = type("S", (), {"_lock": threading.RLock()})()
    assert JobService._take_translated_title(fake, job) is True and job.spec.title == TITLE


def test_the_readers_chapter_job_is_labelled_with_the_book(novel):
    """The real Reader's Translate this chapter."""
    pytest.importorskip("flet")

    async def scenario():
        screen = _screen(novel)
        await screen.open()
        await screen.go_chapter(1)
        await screen.translate_chapter()
        spec = screen.deps.jobs.submitted[-1]
        assert spec.params["chapter_file"] == "chapter0002.xhtml"
        assert spec.title == "Novel" == screen.session.plan.title
        screen.dispose()

    asyncio.run(scenario())


# ---- (b) one entry per book ----------------------------------------------------------------------------------


def test_the_owners_three_finished_jobs_leave_when_a_new_job_is_queued(tmp_path):
    """The screenshot: Finished (3), one book, two Library entries and the Reader's, stored by the older build."""
    source, other = epub(tmp_path), epub(tmp_path, "Other.epub")
    service, backend = make_service(tmp_path)
    _store(service,
           _ended("lib1", _library(source), created=3.0),
           _ended("lib2", _library(source, bid="b-from-the-qa-card"), created=2.0),
           _ended("read1", _reader(source), created=1.0),
           _ended("other1", _library(other, bid="b2"), JobState.FAILED, created=0.5))
    service.close()

    service, backend = make_service(tmp_path)  # restart: they come back from the stored state
    assert _ids(service) == ["lib1", "lib2", "read1", "other1"]
    service.pause_queue()
    job_id = service.submit(_library(source))
    assert service.view().queue[0].id == job_id
    assert _ids(service) == ["other1"]  # queued is enough; another book's entry stays
    saved = json.loads(Path(service.history_state_path).read_text(encoding="utf-8"))
    assert [j["id"] for j in saved["jobs"]] == ["other1"]
    service.cancel_queued(job_id)
    service.close()

    service, backend = make_service(tmp_path)  # stays gone after a restart
    assert "lib1" not in _ids(service) and "read1" not in _ids(service) and "other1" in _ids(service)
    service.close()


def test_at_most_one_finished_entry_remains_once_the_new_job_ended(tmp_path):
    source = epub(tmp_path)
    service, backend = make_service(tmp_path)
    _store(service, _ended("lib1", _library(source)), _ended("read1", _reader(source)))
    gate = Gate()
    backend.behavior = gate
    job_id = service.submit(_library(source))
    assert gate.entered.wait(TIMEOUT)
    assert _ids(service) == []  # running
    gate.release.set()
    assert service.wait_idle(TIMEOUT)
    assert _ids(service) == [job_id]
    service.close()


@pytest.mark.parametrize("state", [JobState.DONE, JobState.CANCELLED, JobState.FAILED])
@pytest.mark.parametrize("start", ["library", "reader", "chat", "qa", "compile", "retranslate"])
def test_every_start_replaces_every_ended_state(tmp_path, state, start):
    source, workspace = epub(tmp_path), str(tmp_path / "out" / "Book")
    service, _ = make_service(tmp_path)
    _store(service, _ended("old", _library(source), state, output_dir=workspace),
           _ended("same-title-other-book", _library(epub(tmp_path, "Other.epub"), bid="b2")))
    service.pause_queue()
    spec = {
        "library": _library(source),
        "reader": _reader(source),
        "chat": _spec("direct_text", "Chat · Novel", str(tmp_path / "run_root" / "input.txt"),
                      params={JOB_TARGET_PARAM: source}, origin={"type": "chat", "cid": "3"}),
        "qa": _spec("qa_scan", TITLE, workspace, params={"mode": "quick-scan",
                                                        "targets": [{"folder": workspace, "source": source}]}),
        "compile": _spec("compile_epub", TITLE, params={"folder": workspace}, origin={"type": "chat", "cid": "3"}),
        "retranslate": _spec("retranslate", f"{TITLE} · 9 selected", params={"output_dir": workspace}, resumable=False),
    }[start]
    job_id = service.submit(spec)
    assert _ids(service) == ["same-title-other-book"]
    service.cancel_queued(job_id)
    service.close()


def test_a_chat_attachments_entry_and_the_librarys_are_one_book(tmp_path):
    source = epub(tmp_path)
    service, _ = make_service(tmp_path)
    chat = _spec("direct_text", "novel.epub", str(tmp_path / "run" / "x.txt"), params={JOB_TARGET_PARAM: source},
                 origin={"type": "chat", "cid": "3"})
    _store(service, _ended("chat1", chat))
    service.pause_queue()
    first = service.submit(_library(source))
    assert _ids(service) == []
    service.cancel_queued(first)  # now its Cancelled entry is the book's one entry
    assert _ids(service) == [first]
    second = service.submit(chat)
    assert _ids(service) == []
    service.cancel_queued(second)
    service.close()


def test_a_typed_chat_and_a_job_over_several_books_are_not_one_book(tmp_path):
    a, b = epub(tmp_path, "A.epub"), epub(tmp_path, "B.epub")
    service, _ = make_service(tmp_path)
    typed = _spec("direct_text", "Typed text", str(tmp_path / "run" / "x.txt"), origin={"type": "chat", "cid": "3"})
    both = _spec("translate", "A +1", a, b, origin={"type": "library", "label": "Library"})
    _store(service, _ended("typed1", typed), _ended("both1", both))
    service.pause_queue()
    only_a = service.submit(_library(a))
    assert _ids(service) == ["typed1", "both1"]  # B's work is in it too
    service.cancel_queued(only_a)
    again = service.submit(both)
    assert "both1" not in _ids(service) and only_a not in _ids(service) and "typed1" in _ids(service)
    service.cancel_queued(again)
    service.close()


def test_a_job_that_ends_while_another_for_its_book_waits_is_not_listed(tmp_path):
    source = epub(tmp_path)
    service, backend = make_service(tmp_path)
    gate = Gate()
    backend.behavior = gate
    first = service.submit(_reader(source))
    assert gate.entered.wait(TIMEOUT)
    service.pause_queue()
    queued = service.submit(_library(source))
    gate.release.set()
    assert wait_for(lambda: service.snapshot(first).is_terminal)
    assert wait_for(lambda: first not in _ids(service) and service.snapshot(first) is not None), _ids(service)
    service.cancel_queued(queued)
    service.close()


def test_a_book_known_only_by_its_workspace_leaves_when_the_new_job_names_it(tmp_path):
    """A scan picked in Tools names the folder alone; a translation of the book only says its folder once it runs."""
    source = epub(tmp_path)
    service, backend = make_service(tmp_path)
    workspace = str(Path(backend.out_root) / "Book")
    scan = _spec("qa_scan", TITLE, workspace, params={"mode": "quick-scan", "targets": [{"folder": workspace}]})
    _store(service, _ended("scan1", scan))
    gate = Gate()
    backend.behavior = gate
    service.submit(_spec("translate", TITLE, source))
    assert gate.entered.wait(TIMEOUT)
    assert wait_for(lambda: _ids(service) == []), _ids(service)
    gate.release.set()
    assert service.wait_idle(TIMEOUT)
    service.close()


def test_the_book_is_told_by_its_paths_not_its_title(tmp_path):
    a, b = epub(tmp_path, "A.epub"), epub(tmp_path, "B.epub")
    assert not same_books(job_books(_spec("translate", TITLE, a)), job_books(_spec("translate", TITLE, b)))
    assert same_books(job_books(_library(a)), job_books(_reader(a, bid="other")))
    assert job_books(_spec("manga", TITLE, a)) == ()  # not a book's job
    assert job_books(_spec("direct_text", "Typed", a)) == ()  # its input is a file of its run root


# ---- (c) what referred to the entry ----------------------------------------------------------------------------


def test_an_entry_that_left_stays_readable_by_its_id_after_a_restart(tmp_path):
    source = epub(tmp_path)
    service, _ = make_service(tmp_path)
    _store(service, _ended("old", _library(source), JobState.CANCELLED, outputs=("/b/x.epub",)))
    service.pause_queue()
    new = service.submit(_library(source))
    assert service.snapshot("old").spec.title == TITLE and service.view().find("old") is not None
    assert service.resume("old") == new  # a stale Resume reaches the job that carries it on
    service.cancel_queued(new)
    service.close()

    service, _ = make_service(tmp_path)
    snap = service.snapshot("old")
    assert snap is not None and snap.state is JobState.CANCELLED and snap.outputs == ("/b/x.epub",)
    assert "old" not in _ids(service)
    assert service.resume("old") is None  # nothing to resume through it now: refused, no crash
    assert service.latest_job_for(source).id == new  # the chat's Result card follows the book's job
    service.clear_finished()
    assert service.snapshot("old") is None
    service.close()


def test_the_job_page_of_an_entry_that_left_opens_and_offers_no_resume(tmp_path):
    pytest.importorskip("flet")
    from glossarion_mobile.ui.router import parse_route
    from glossarion_mobile.ui.screens.job_detail import JobDetailScreen

    source = epub(tmp_path)
    service, _ = make_service(tmp_path)
    _store(service, _ended("old", _library(source), JobState.CANCELLED))
    service.pause_queue()
    new = service.submit(_library(source))
    detail = JobDetailScreen(parse_route("/jobs/old"), service=service)
    detail.get_body()
    assert detail.snap.id == "old" and not detail.resume_button.visible
    unknown = JobDetailScreen(parse_route("/jobs/never-existed"), service=service)
    unknown.get_body()  # the page it always was: "Job not found"
    assert unknown.snap is None
    service.cancel_queued(new)
    service.close()


def test_the_jobs_page_lists_the_book_once(tmp_path):
    pytest.importorskip("flet")
    from glossarion_mobile.ui.router import parse_route
    from glossarion_mobile.ui.screens.jobs import JobsScreen

    source = epub(tmp_path)
    service, _ = make_service(tmp_path)
    _store(service, _ended("lib1", _library(source)), _ended("read1", _reader(source)))
    screen = JobsScreen(parse_route("/jobs"), service=service)
    screen.get_body()

    def rows():
        return [str(c.key) for c in screen.list_view.controls if str(getattr(c, "key", "")).startswith("job-row-")]

    assert rows() == ["job-row-lib1", "job-row-read1"]
    service.pause_queue()
    new = service.submit(_library(source))
    screen._render(service.view())
    assert rows() == []  # the queued job is in its reorderable list, the book's old entries are gone
    service.cancel_queued(new)
    service.close()
