"""The job notification shows the streamed text of up to five in-flight requests, then "+N more".

Owner 2026-11-17: "why do my notifications all only display the thinking/text streaming log of 1 in flight
request. It should be at least 5, with a +N when there is more not shown". The job kept one combined tail
of streamed text (``stream_tail``), so requests running at once shared a single row. The rows now come from
the job's request cards (``DirectTextStream.segments``): one short row per in-flight request that streams.
"""

from __future__ import annotations

import asyncio
import sys
import threading
import types
from pathlib import Path

APP = Path(__file__).resolve().parents[1] / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from glossarion_mobile.services import jobs as jb  # noqa: E402
from glossarion_mobile.services.background import BackgroundExecution  # noqa: E402
from glossarion_mobile.services.jobs import (  # noqa: E402
    INSTANCE_PARAM, JobSnapshot, JobSpec, JobState, JobsView, Progress, live_rows, request_streams)
from glossarion_mobile.services.notifications import JobNotifications  # noqa: E402


def _segment(content="", thinking="", phase="text", complete=False, status_only=False):
    return {"content": content, "thinking": thinking, "phase": phase, "complete": complete,
            "status_only": status_only}


def _snap(job_id="a1", instance=1, title="Book one", question=None, state=JobState.RUNNING):
    return JobSnapshot(id=job_id, spec=JobSpec("translate", title, ("x",), params={INSTANCE_PARAM: instance}),
                       state=state, created=1.0, started=1.0, progress=Progress(total=10, completed=1),
                       question=question)


def test_request_streams_takes_each_in_flight_request_that_streams():
    segments = [
        _segment(content="<p>done long ago</p>", complete=True),            # finished: no row
        _segment(phase="processing", status_only=True),                     # a status card: no row
        _segment(thinking="weighing the honorific\n", phase="thinking"),    # thinking
        _segment(content="<p>She smiled.</p>\n", thinking="earlier thought\n", phase="text"),  # answering
        _segment(phase="queue"),                                            # nothing streamed yet: no row
        _segment(thinking="only thoughts so far\n", phase="processing"),    # thoughts, no answer yet
    ]
    assert request_streams(segments) == [("weighing the honorific\n", True), ("<p>She smiled.</p>\n", False),
                                         ("only thoughts so far\n", True)]
    assert request_streams(None) == [] and request_streams(["not a card"]) == []
    long = _segment(content="x" * (jb.STREAM_TAIL_KEEP + 500))
    assert len(request_streams([long])[0][0]) == jb.STREAM_TAIL_KEEP      # the end of the text is enough


def test_request_streams_reads_the_real_request_cards():
    """``DirectTextStream.segments(drain=False)``: the keys the rows read are the stream's own."""
    import direct_text_stream

    stream = direct_text_stream.make_stream()
    stream.load_segments([
        {"request_number": 1, "label": "Chapter 1", "content": "<p>first</p>", "thinking": "", "phase": "text",
         "complete": True, "status_only": False},
        {"request_number": 2, "label": "Chapter 2", "content": "<p>second</p>", "thinking": "", "phase": "text",
         "complete": False, "status_only": False},
        {"request_number": 3, "label": "Chapter 3", "content": "", "thinking": "hmm", "phase": "thinking",
         "complete": False, "status_only": False},
    ])
    assert request_streams(stream.segments(drain=False)) == [("<p>second</p>", False), ("hmm", True)]
    job = types.SimpleNamespace(stream=stream)
    service = types.SimpleNamespace(_lock=threading.RLock(), _active=job, _find_live=lambda job_id: job)
    assert jb.JobService.request_streams(service) == [("<p>second</p>", False), ("hmm", True)]
    assert jb.JobService.request_streams(service, "j1") == [("<p>second</p>", False), ("hmm", True)]
    service._active = None
    assert jb.JobService.request_streams(service) == []
    service._active = types.SimpleNamespace(stream=None)
    assert jb.JobService.request_streams(service) == []


def test_live_rows_show_five_requests_and_count_the_rest(monkeypatch):
    monkeypatch.setattr(jb, "on_device_row", lambda: "")
    snap = _snap()
    streams = [(f"<p>request {n} writes its answer here</p>", n % 3 == 0) for n in range(1, 8)]  # 7 streaming
    rows = live_rows(snap, "combined tail", False, streams=streams)
    assert rows == [
        "✍️ request 1 writes its answer here",
        "✍️ request 2 writes its answer here",
        "🧠 request 3 writes its answer here",
        "✍️ request 4 writes its answer here",
        "✍️ request 5 writes its answer here",
        "+2 more",
    ]
    assert jb.NOTIFICATION_STREAM_ROWS == 5
    # exactly five: all shown, no "+N"
    assert live_rows(snap, "", False, streams=streams[:5]) == rows[:5]
    # each row is short while several stream: about one line each
    long = [("word " * 80, False), ("other " * 80, True)]
    two = live_rows(snap, "", False, streams=long)
    assert len(two) == 2 and all(len(row) <= jb.NOTIFICATION_MULTI_STREAM_CHARS + 4 for row in two)
    assert two[0].startswith("✍️ …") and two[1].startswith("🧠 …")
    # a share of the rows (several jobs in one notification)
    assert live_rows(snap, "", False, streams=streams, max_rows=2) == rows[:2] + ["+5 more"]


def test_one_request_keeps_its_single_longer_row(monkeypatch):
    monkeypatch.setattr(jb, "on_device_row", lambda: "")
    snap = _snap()
    tail = "<p>" + "the long sentence goes on " * 12 + "</p>"
    single = live_rows(snap, tail, False)
    assert live_rows(snap, tail, False, streams=[(tail, False)]) == single       # one request: as before
    assert live_rows(snap, tail, True, streams=[]) == live_rows(snap, tail, True)
    assert len(single) == 1 and len(single[0]) > jb.NOTIFICATION_MULTI_STREAM_CHARS + 4
    # entries with nothing to show do not count as a request that streams
    assert live_rows(snap, tail, False, streams=[("<p></p>", False), (" \n", True)]) == single
    # a job that asks, or is not running, shows no rows
    assert live_rows(_snap(question={"kind": "glossary"}), tail, False, streams=[("a", False), ("b", False)]) == []
    assert live_rows(None, tail, False, streams=[("a", False), ("b", False)]) == []
    # the on-device row stays first
    assert live_rows(snap, "", False, device="📱 12 tokens written", streams=[("a", False), ("b", True)]) == [
        "📱 12 tokens written", "✍️ a", "🧠 b"]


class Native:
    def __init__(self) -> None:
        self.calls = []

    async def call(self, method, *args, default=None, **kwargs):
        self.calls.append((method, args, kwargs))
        return {"start_job_service": True, "show_notification": True}.get(method, default)

    async def init_notifications(self):
        return True

    async def cancel_notification(self, notification_id):
        self.calls.append(("cancel_notification", (notification_id,), {}))


class Jobs:
    def __init__(self, running, streams) -> None:
        self.running = list(running)
        self.streams = dict(streams)

    def snapshot(self, job_id=None):
        return next((s for s in self.running if s.instance == 1), None)

    def running_snapshots(self):
        return list(self.running)

    def stream_tail(self, job_id=None):
        return "the combined tail of every request"

    def stream_thinking(self, job_id=None):
        return False

    def request_streams(self, job_id=None):
        return list(self.streams.get(job_id, ()))

    def device_row(self, job_id=None, threads=False):
        return ""

    def view(self):
        return JobsView(running=tuple(self.running))


def _texts(native):
    return [c[2]["text"] for c in native.calls if c[0] == "update_job_service"]


def test_the_android_notification_lists_the_requests(monkeypatch):
    monkeypatch.setattr(jb, "on_device_row", lambda: "")
    one = _snap()
    jobs = Jobs([one], {"a1": [(f"<p>chapter {n} text</p>", n == 2) for n in range(1, 9)]})  # 8 streaming
    native = Native()
    background = BackgroundExecution(native, platform="android", jobs=jobs, notifications=JobNotifications(
        native, platform="android"), clock=lambda: 1000.0)

    async def scenario():
        await background.job_started(one)
        await background.job_progress(one, force=True)
        assert _texts(native)[-1].split("\n") == [
            "Book one", "✍️ chapter 1 text", "🧠 chapter 2 text", "✍️ chapter 3 text", "✍️ chapter 4 text",
            "✍️ chapter 5 text", "+3 more"]
        jobs.streams["a1"] = [("<p>chapter 9 text</p>", False)]     # one request left: its single row again
        await background.job_progress(one, force=True)
        assert _texts(native)[-1].split("\n") == ["Book one", "✍️ the combined tail of every request"]
        jobs.streams["a1"] = []
        await background.job_progress(one, force=True)
        assert _texts(native)[-1].split("\n") == ["Book one", "✍️ the combined tail of every request"]
        background.close()

    asyncio.run(scenario())


def test_jobs_of_two_instances_share_the_rows(monkeypatch):
    monkeypatch.setattr(jb, "on_device_row", lambda: "")
    one, two = _snap("a1", 1, "Book one"), _snap("b2", 2, "Book two")
    jobs = Jobs([one, two], {"a1": [(f"one {n}", False) for n in range(1, 5)],     # 4 streaming
                             "b2": [("two 1", True), ("two 2", False)]})
    native = Native()
    background = BackgroundExecution(native, platform="android", jobs=jobs, notifications=JobNotifications(
        native, platform="android"), clock=lambda: 1000.0)

    async def scenario():
        await background.job_started(one)
        await background.job_progress(two, force=True)
        lines = _texts(native)[-1].split("\n")
        first = lines.index(next(line for line in lines if line.startswith("[1] ")))
        second = lines.index(next(line for line in lines if line.startswith("[2] ")))
        assert lines[first + 1:second] == ["    ✍️ one 1", "    ✍️ one 2", "    +2 more"]   # 5 rows over 2 jobs
        assert lines[second + 1:] == ["    🧠 two 1", "    ✍️ two 2"]
        background.close()

    asyncio.run(scenario())


def test_ios_keeps_one_line_per_job(monkeypatch):
    monkeypatch.setattr(jb, "on_device_row", lambda: "")
    one = _snap()
    jobs = Jobs([one], {"a1": [("a", False), ("b", False), ("c", True)]})
    background = BackgroundExecution(Native(), platform="ios", jobs=jobs, notifications=JobNotifications(
        Native(), platform="ios"), clock=lambda: 1000.0)
    assert background._live_rows(one) == [] and background._rows_of(one) == []
