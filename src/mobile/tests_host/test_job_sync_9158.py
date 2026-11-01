"""Owner 2026-10-30, mobile on v9.15.8:

1. "The mobile version of glossarion still shows dupe jobs if one job started in the chat and one was resumed in
   the jobs menu. We need the chat and the jobs menu in sync." A book has one entry under Finished / Interrupted:
   a new job for it carries its Stopped / Failed / Interrupted work on (``JobService._drop_superseded``; since
   mobile batch 9.15.11 its Done entries too, ``test_m11_job_book_entries``), a resumed Interrupted job is not
   listed again, and the chat's Result card of that book follows the book's
   newer job (``ChatView._book_job_elsewhere`` over ``JobService.latest_job_for``).
2. "The translating chapters label also is still wrong ... it's 182 spines in mobile, so then log really should
   have been Translating • [17/182] Ch. 13 ... (1/2)" in the Jobs page and the notification. A job started from
   the Library / Jobs page named its requests "Request N", so it had no position at all; it now names chapters,
   and the position carries the spine total.
3. "In the notifications, the actual title of the novel, should be a much smaller font and truncated": the
   headline is the job line, the book's title the smaller text under it, cut to one short line.
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path

os.environ.setdefault("GLOSSARION_HTTP_LOG", "0")  # unified_api_client (via TransateKRtoEN) logs into src/ otherwise
HERE = Path(__file__).resolve().parent
for path in (HERE.parent / "app", HERE.parent.parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from glossarion_mobile.services.jobs import (  # noqa: E402
    NOTIFICATION_TITLE_CHARS, JobService, JobSnapshot, JobSpec, JobState, Progress, notification_lines,
    position_line, progress_line, request_position, short_title,
)
from test_jobs import TIMEOUT, Gate, epub, make_service, wait_for  # noqa: E402


def _seg(label, complete=False):
    return {"label": label, "complete": complete}


def _snap(title="Book.epub", **kwargs):
    kwargs.setdefault("progress", Progress(total=15, completed=14))
    return JobSnapshot(id="j", spec=JobSpec("translate", title), state=JobState.RUNNING, created=0.0, started=0.0,
                       **kwargs)


# ---- 2. the job line ------------------------------------------------------------------------------------------


def test_the_position_names_the_spine_place_out_of_the_spine():
    segments = [_seg("Chapter 13 (chunk 1/2) · chapter0013.xhtml")]
    assert request_position(segments, lambda name: 17, 182) == "[17/182] Ch. 13 (1/2)"
    assert request_position([_seg("Chapter 13 · chapter0013.xhtml")], lambda name: 17, 182) == "[17/182] Ch. 13"
    # no spine read (yet): the place alone; not in the spine: neither
    assert request_position(segments, lambda name: 17) == "[17] Ch. 13 (1/2)"
    assert request_position(segments, lambda name: 0, 182) == "Ch. 13 (1/2)"


def test_the_job_line_is_verb_spine_chapter_and_chunk():
    running = _snap(position="[17/182] Ch. 13 (1/2)")
    assert position_line(running) == "Translating • [17/182] Ch. 13 (1/2)"
    assert progress_line(running, now=2.0, verb=True).startswith("Translating • [17/182] Ch. 13 (1/2)")
    # the JobStrip names the kind in its title already: its subtitle starts with the place
    assert progress_line(running, now=2.0).startswith("[17/182] Ch. 13 (1/2)")
    # before a request names a chapter: the counts
    assert position_line(_snap()) == "Translating • 14/15"
    ended = JobSnapshot(id="j", spec=JobSpec("translate", "Book"), state=JobState.DONE, created=0.0,
                        position="[17/182] Ch. 13", progress=Progress(total=15, completed=15))
    assert position_line(ended) == "Translating • 15/15"


def test_the_service_reads_the_spine_once_and_counts_it(tmp_path, monkeypatch):
    import TransateKRtoEN

    reads = []
    spine = [f"chapter{n:04d}.xhtml" for n in range(182)]

    def files(output_dir, epub_path=None):
        reads.append(output_dir)
        return spine

    monkeypatch.setattr(TransateKRtoEN, "_single_pass_spine_files", files)
    job = types.SimpleNamespace(stream=types.SimpleNamespace(segments=lambda drain=False: [
        _seg("Chapter 13 (chunk 1/2) · chapter0016.xhtml")]), position="", spine={}, spine_files=None,
        output_dir=str(tmp_path))
    service = types.SimpleNamespace(_lock=__import__("threading").RLock(), _changed=lambda: None,
                                    _spine_position=lambda j, name: JobService._spine_position(j, name))
    JobService._track_position(service, job)
    assert job.position == "[17/182] Ch. 13 (1/2)"
    job.stream = types.SimpleNamespace(segments=lambda drain=False: [_seg("Chapter 14 · chapter0017.html")])
    JobService._track_position(service, job)
    assert job.position == "[18/182] Ch. 14" and reads == [str(tmp_path)]  # same stem, other extension


def test_the_spine_helper_reads_the_workspace_opf(tmp_path):
    from TransateKRtoEN import _single_pass_spine_files, _single_pass_spine_ref_for_file

    (tmp_path / "META-INF").mkdir()
    (tmp_path / "META-INF" / "container.xml").write_text(
        '<?xml version="1.0"?><container xmlns="urn:oasis:names:tc:opendocument:xmlns:container" version="1.0">'
        '<rootfiles><rootfile full-path="content.opf" media-type="application/oebps-package+xml"/></rootfiles>'
        '</container>', encoding="utf-8")
    items = "".join(f'<item id="c{n}" href="Text/ch{n}.xhtml" media-type="application/xhtml+xml"/>' for n in range(3))
    refs = "".join(f'<itemref idref="c{n}"/>' for n in (2, 0, 1))
    (tmp_path / "content.opf").write_text(
        '<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf" version="3.0">'
        f'<manifest>{items}<item id="css" href="s.css" media-type="text/css"/></manifest>'
        f'<spine>{refs}</spine></package>', encoding="utf-8")
    assert _single_pass_spine_files(str(tmp_path)) == ["ch2.xhtml", "ch0.xhtml", "ch1.xhtml"]
    assert _single_pass_spine_ref_for_file(str(tmp_path), "ch0.xhtml") == (1, 2)
    assert _single_pass_spine_ref_for_file(str(tmp_path), "ch1.html") == (2, 3)  # matched by stem
    assert _single_pass_spine_files(str(tmp_path / "missing")) == []


def test_a_library_run_names_chapters_in_its_request_cards(monkeypatch):
    """Its requests were "Request 1", "Request 2": ``request_position`` found no chapter and the line fell back
    to "14/15"."""
    import direct_text_stream

    from glossarion_mobile.services import jobs as jobs_mod

    made = []
    monkeypatch.setattr(direct_text_stream, "make_stream", lambda **kw: made.append(kw) or object())
    for kind in ("translate", "single_chapter"):
        jobs_mod._make_request_stream(JobSpec(kind, "Book", ("/b/Book.epub",)))
    jobs_mod._make_request_stream(JobSpec("direct_text", "Chat", (), params={"is_attachment": False}))
    assert [m["source_is_attachment"] for m in made] == [True, True, False]


# ---- 3. the notification ----------------------------------------------------------------------------------


def test_the_notification_headline_is_the_job_line_and_the_title_is_cut_short():
    long_title = "The Seiura of the Hero World: a very long light novel title that never ends"
    headline, book = notification_lines(_snap(long_title, position="[17/182] Ch. 13 (1/2)", in_flight=1))
    assert headline == "Translating • [17/182] Ch. 13 (1/2) · 1 in flight"
    assert book == short_title(long_title) and len(book) == NOTIFICATION_TITLE_CHARS and book.endswith("…")
    assert short_title("Short\n title") == "Short title"


def test_the_service_notification_sends_headline_and_book(tmp_path):
    import asyncio

    from glossarion_mobile.services.background import BackgroundExecution

    calls = []

    class Native:
        async def call(self, method, *args, default=None, **kwargs):
            calls.append((method, args, kwargs))
            return True

    background = BackgroundExecution(native=Native(), platform="android")
    background.service_running = True
    snap = _snap(position="[17/182] Ch. 13 (1/2)")
    asyncio.run(background.job_progress(snap, force=True))
    update = [c for c in calls if c[0] == "update_job_service"][-1]
    assert update[2] == {"title": "Translating • [17/182] Ch. 13 (1/2)", "text": "Book.epub"}


# ---- 1. one job per book ----------------------------------------------------------------------------------


def _ended(job_id, path, state, created, *, stopped=False):
    return JobSnapshot(id=job_id, spec=JobSpec("translate", "Book", (path,)), state=state, created=created,
                       started=created if stopped or state is not JobState.CANCELLED else None,
                       finished=created + 1, progress=Progress(total=15, completed=14))


def test_a_new_job_for_a_book_carries_on_its_unfinished_entries(tmp_path):
    service, backend = make_service(tmp_path)
    book, other = epub(tmp_path), epub(tmp_path, "Other.epub")
    with service._lock:
        service._history = [_ended("stopped1", book, JobState.CANCELLED, 3.0, stopped=True),
                            _ended("done1", book, JobState.DONE, 2.0),
                            _ended("other1", other, JobState.FAILED, 1.0)]
        service._interrupted = [_ended("killed1", book, JobState.RUNNING, 0.5)]
    gate = Gate()
    backend.behavior = gate
    job_id = service.submit(JobSpec("translate", "Book", (book,), origin={"type": "library"}))
    assert gate.entered.wait(TIMEOUT)
    # the book's Stopped, Done and Interrupted entries leave (owner 2026-11-01); the other book's stays
    assert [s.id for s in service.view().history] == ["other1"]
    assert service.interrupted == ()
    assert service.latest_job_for(book).id == job_id
    gate.release.set()
    assert service.wait_idle(TIMEOUT)
    assert service.latest_job_for(book).id == job_id  # ended: still the book's newest
    assert service.latest_job_for(other).id == "other1" and service.latest_job_for(tmp_path / "x.epub") is None
    service.close()


def test_a_resumed_interrupted_job_is_listed_once(tmp_path):
    service, backend = make_service(tmp_path)
    spec = JobSpec("direct_text", "Typed text", ())  # no target: only the resume itself can drop it
    with service._lock:
        service._interrupted = [JobSnapshot(id="killed2", spec=spec, state=JobState.RUNNING, created=1.0,
                                            started=1.0)]
    resumed = service.resume("killed2")
    assert resumed and service.interrupted == ()
    assert service.wait_idle(TIMEOUT)
    assert "killed2" not in [s.id for s in service.view().history]
    assert service.resume("killed2") is None  # and a second Resume is still refused
    service.close()


def test_the_chat_card_follows_the_books_newer_job():
    from glossarion_mobile.ui.chat.chat_view import ChatView

    own = types.SimpleNamespace(id="chat1", created=5.0, is_active=False)
    newer = types.SimpleNamespace(id="lib1", created=9.0, is_active=False,
                                  spec=types.SimpleNamespace(origin={"type": "library"}))
    found = {"job": newer}
    jobs = types.SimpleNamespace(snapshot_of=lambda jid: None, latest_job_for=lambda path, family: found["job"])
    live = {"run": None}
    view = types.SimpleNamespace(env=types.SimpleNamespace(
        jobs=jobs, runs=types.SimpleNamespace(live_run=lambda cid: live["run"])), cid="c1", _handed_off={},
        _turn_source=lambda item: "/books/hero.epub")
    item = types.SimpleNamespace(index=3)
    assert ChatView._book_job_elsewhere(view, item, own) is newer  # resumed in the Jobs page after the chat's run
    found["job"] = types.SimpleNamespace(id="old", created=1.0, is_active=False, spec=newer.spec)
    assert ChatView._book_job_elsewhere(view, item, own) is None  # older than the turn's own job
    found["job"] = own
    assert ChatView._book_job_elsewhere(view, item, own) is None  # the turn's own job
    running = types.SimpleNamespace(id="lib2", created=0.0, is_active=True)
    found["job"] = running
    assert ChatView._book_job_elsewhere(view, item, own) is running  # active: always shown
    live["run"] = types.SimpleNamespace(job_id="lib2")
    assert ChatView._book_job_elsewhere(view, item, own) is None  # unless it is this chat's live run
    live["run"] = None
    # the turn's own job left the lists (a later job carried it on): the newest job of the book, not this chat's
    found["job"] = newer
    assert ChatView._book_job_elsewhere(view, item, None) is newer
    found["job"] = types.SimpleNamespace(id="c2", created=9.0, is_active=False,
                                         spec=types.SimpleNamespace(origin={"type": "chat", "cid": "c1"}))
    assert ChatView._book_job_elsewhere(view, item, None) is None


def test_the_chat_cards_stop_and_resume_act_on_the_books_job():
    from glossarion_mobile.ui.chat.chat_view import ChatView

    stops, resumes, routes, notes, renders = [], [], [], [], []
    jobs = types.SimpleNamespace(request_stop=lambda force=False, job_id=None: stops.append((job_id, force)),
                                 resume=lambda jid: resumes.append(jid) or "new1")
    view = types.SimpleNamespace(env=types.SimpleNamespace(jobs=jobs), navigate=lambda r, p: routes.append((r, p)),
                                 notify=notes.append, transcript=types.SimpleNamespace(follow_tail=True),
                                 render_transcript=lambda follow=True: renders.append(follow))
    assert ChatView._book_job_action(view, "stop", "lib1") and stops == [("lib1", False)]
    assert ChatView._book_job_action(view, "force_stop", "lib1") and stops[-1] == ("lib1", True)
    assert ChatView._book_job_action(view, "log", "lib1") and routes == [("jobs.detail", {"jid": "lib1"})]
    assert ChatView._book_job_action(view, "resume", "lib1") and resumes == ["lib1"] and notes == []
    assert not ChatView._book_job_action(view, "read", "lib1")  # the card's own actions


def test_the_chat_follows_the_snapshots_the_adapter_delivers():
    """A book job started in the Library repaints the chat once, its line changes repaint the card that shows it,
    and its end too."""
    from glossarion_mobile.ui.chat.chat_view import ChatView

    def running(position, state=JobState.RUNNING):
        return JobSnapshot(id="lib1", spec=JobSpec("translate", "Book", ("/b/Book.epub",)), state=state,
                           created=1.0, started=1.0, position=position, progress=Progress(total=15, completed=14))

    live = {"run": None}
    view = types.SimpleNamespace(env=types.SimpleNamespace(runs=types.SimpleNamespace(
        live_run=lambda cid: live["run"])), cid="c1", _book_jobs={}, _book_seen={}, _handed_off={})
    changed = lambda snap: ChatView._book_job_changed(view, snap)  # noqa: E731
    assert not changed(None)  # the view without an active job
    assert changed(running("[17/182] Ch. 13 (1/2)"))  # a book job started: repaint once
    assert not changed(running("[17/182] Ch. 13 (2/2)"))  # no card shows it
    view._book_jobs[("c1", 3)] = "lib1"
    assert changed(running("[18/182] Ch. 14"))  # the card shows it: its line follows
    assert not changed(running("[18/182] Ch. 14"))
    assert changed(running("[18/182] Ch. 14", JobState.CANCELLED))  # it ended
    view._handed_off[("c1", 4)] = "lib3"
    handed = JobSnapshot(id="lib3", spec=JobSpec("translate", "Book"), state=JobState.DONE, created=2.0)
    assert changed(handed) and view._handed_off == {}  # the Resume's Library job ended
    live["run"] = types.SimpleNamespace(job_id="lib2")
    other = JobSnapshot(id="lib2", spec=JobSpec("translate", "Book", ()), state=JobState.RUNNING, created=2.0)
    assert not changed(other)  # the chat's own run: its live card shows it
    assert not changed(JobSnapshot(id="qa1", spec=JobSpec("qa_scan", "QA"), state=JobState.RUNNING,
                                   created=3.0))  # not a translation


def test_the_chat_reaches_the_books_job_through_the_real_adapter(tmp_path):
    """The chat's ``env.jobs`` is a ``JobsAdapter``: hotfix 8's ``jobs.snapshot(id)`` raised TypeError on it and its
    ``active_job_for`` lookup found nothing, so a Result card never saw the book's other job on a phone."""
    from glossarion_mobile.ui.chat.chat_view import ChatView
    from glossarion_mobile.ui.chat.job_binding import JobsAdapter

    service, backend = make_service(tmp_path)
    gate = Gate()
    backend.behavior = gate
    book = epub(tmp_path)
    job_id = service.submit(JobSpec("translate", "Book", (book,), origin={"type": "library"}))
    assert gate.entered.wait(TIMEOUT)
    view = types.SimpleNamespace(env=types.SimpleNamespace(jobs=JobsAdapter(service), runs=None), cid="c1",
                                 _handed_off={}, _turn_source=lambda item: book)
    item = types.SimpleNamespace(index=3)
    assert ChatView._book_job_elsewhere(view, item).id == job_id
    view._handed_off[("c1", 3)] = job_id
    assert ChatView._book_job_elsewhere(view, item).id == job_id
    assert JobsAdapter(service).request_stop(force=True, job_id=job_id)
    gate.release.set()
    assert service.wait_idle(TIMEOUT)
    service.close()
