"""v9.15.7 hotfix 8, owner (2026-10-30, phone): the jobs notification and menu "have a chapter tracker label,
but it's useless. It's not tracking the actual remaining chapters. I want it to track spine number, chapter
number and chunk number. And be compact. Like [5] Ch. 3 (1/4)".

Fix: ``JobService._track_position`` reads the job's newest in-flight request card (``request_position``) after
each request-stream drain; the spine number is the file's place in the workspace OPF (the Progress manager's
"[005]"). The job line and the notification show it while the job runs, "Ch x/y" only without one.

Run from src/mobile with the mobile venv::

    .venv/Scripts/python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_hotfix_9157_8.py
"""

from __future__ import annotations

import sys
import threading
import types
from pathlib import Path

MOBILE_DIR = Path(__file__).resolve().parents[1]
for entry in (str(MOBILE_DIR / "app"), str(MOBILE_DIR.parent)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from glossarion_mobile.services.jobs import (  # noqa: E402
    JobService, JobSnapshot, JobSpec, JobState, Progress, notification_text, progress_line, request_position,
)


def _seg(label, complete=False):
    return {"label": label, "complete": complete}


def test_the_newest_in_flight_request_names_spine_chapter_and_chunk():
    spine = {"chapter0003.xhtml": 5}
    segments = [_seg("Chapter 1 (chunk 3/3) · chapter0001.xhtml", complete=True),
                _seg("Chapter 2 (chunk 1/8) · chapter0002.xhtml", complete=True),
                _seg("Chapter 3 (chunk 1/4) · chapter0003.xhtml")]
    assert request_position(segments, lambda name: spine.get(name, 0)) == "[5] Ch. 3 (1/4)"
    # a request that names no chapter (metadata, glossary) is passed over; nothing at all: ''
    assert request_position(segments + [_seg("Metadata translation")], lambda name: spine.get(name, 0)) == \
        "[5] Ch. 3 (1/4)"
    assert request_position([_seg("Metadata translation")]) == ""
    assert request_position([_seg("Chapter 3 (chunk 1/4)", complete=True)]) == ""
    # no file (or one the OPF does not list): no spine number; a text source's sections
    assert request_position([_seg("Chapter 12 (chunk 2/2)")]) == "Ch. 12 (2/2)"
    assert request_position([_seg("Section 4 (chunk 1/1) · Section_4.txt")], lambda name: 0) == "Sec. 4 (1/1)"


def _snap(**kwargs):
    return JobSnapshot(id="j", spec=JobSpec("translate", "Book.epub"), state=JobState.RUNNING, created=0.0,
                       started=0.0, progress=Progress(total=9, completed=8), in_flight=1, **kwargs)


def test_the_job_line_and_notification_show_the_position():
    line = progress_line(_snap(position="[5] Ch. 3 (1/4)"), now=7323.0)
    assert line.startswith("[5] Ch. 3 (1/4) · 1 in flight") and "Ch 8/9" not in line
    assert notification_text(_snap(position="[5] Ch. 3 (1/4)")).endswith("Book.epub: [5] Ch. 3 (1/4) · 1 in flight")
    # without a position (no chapter request yet): the counts, as before
    assert progress_line(_snap(), now=10.0).startswith("Ch 8/9")
    assert "8/9 chapters" in notification_text(_snap())


def test_the_service_tracks_the_position_after_each_drain(tmp_path):
    segments = [_seg("Chapter 3 (chunk 1/4) · chapter0003.xhtml")]
    changed = []
    job = types.SimpleNamespace(stream=types.SimpleNamespace(segments=lambda drain=False: list(segments)),
                                position="", spine={"chapter0003.xhtml": 5}, output_dir=str(tmp_path))
    service = types.SimpleNamespace(_lock=threading.RLock(), _changed=lambda: changed.append(1),
                                    _spine_position=lambda j, name: JobService._spine_position(j, name))
    JobService._track_position(service, job)
    assert job.position == "[5] Ch. 3 (1/4)" and changed == [1]
    JobService._track_position(service, job)  # unchanged: no repaint
    assert changed == [1]
    segments.append(_seg("Chapter 4 (chunk 2/2) · chapter0004.xhtml"))
    JobService._track_position(service, job)
    # chapter0004.xhtml is not in a workspace OPF here: no spine number, remembered as unknown
    assert job.position == "Ch. 4 (2/2)" and job.spine["chapter0004.xhtml"] == 0 and changed == [1, 1]


def test_a_result_whose_book_runs_in_another_job_stops_offering_resume():
    """Owner 2026-10-30: "I clicked resume, and the resume button worked, but the response card did not update
    ... It still says resume" - the Resume of a book that moved into the Library starts a Library job, which the
    chat's own run never sees."""
    from glossarion_mobile.ui.chat.chat_view import RESUMED_ELSEWHERE, ChatView
    from glossarion_mobile.ui.chat.job_binding import CardPhase
    from glossarion_mobile.ui.chat.cards import offers_resume

    running = types.SimpleNamespace(id="lib1", is_active=True)
    asked = []

    def active_job_for(path, family):
        asked.append((path, family))
        return running if path == "/books/hero.epub" else None

    jobs = types.SimpleNamespace(snapshot=lambda jid: running if jid == "lib1" else None, active_job_for=active_job_for)
    view = types.SimpleNamespace(env=types.SimpleNamespace(jobs=jobs), cid="c1", _handed_off={},
                                 _turn_source=lambda item: item.source)
    item = types.SimpleNamespace(index=3, source="/books/hero.epub")
    assert ChatView._book_job_elsewhere(view, item) is running and asked == [("/books/hero.epub", "translation")]
    other = types.SimpleNamespace(index=4, source="/books/other.epub")
    assert ChatView._book_job_elsewhere(view, other) is None
    # the job the card's Resume handed to the Library is found by its id, whatever its source
    view._handed_off[("c1", 4)] = "lib1"
    assert ChatView._book_job_elsewhere(view, other) is running
    running.is_active = False
    assert ChatView._book_job_elsewhere(view, other) is None
    # the card then shows a Result without Resume / Retry
    assert not offers_resume(CardPhase("done"), RESUMED_ELSEWHERE)
