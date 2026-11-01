"""Owner issues 2 and 3 (mobile batch 9.15.11), from the chat's side.

A book's ended entries leave Finished while a new job for that book runs (``JobService._drop_superseded``). A chat turn
reads how its job ended from those lists, so it must still find an entry that left. The Result card's "N QA failed"
chip caches the workspace's count under a key a job started elsewhere does not change, so it must read it again when
such a job ends.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from dataclasses import replace
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
    JOB_TARGET_PARAM, JobSnapshot, JobSpec, JobState, Progress,
)
from glossarion_mobile.ui.chat.job_binding import JobsAdapter, ended_card, ended_kind  # noqa: E402
from glossarion_mobile.ui.chat.run_controller import ChatRuns  # noqa: E402
from test_chat import (  # noqa: E402,F401  (the chat's own fixtures)
    _adapter, _cards, _chat_view, _close, _desktop_history, _settle, desktop_store_cls, isolated_env, needs_flet,
)
from test_jobs import TIMEOUT, Gate, epub, make_service  # noqa: E402


def _stopped_chat_job(tmp_path, source):
    """A chat's EPUB turn that was stopped at 4/10 chapters."""
    spec = JobSpec("direct_text", "book.epub", (str(tmp_path / "run" / "x.txt"),),
                   params={"chat_id": "2", "user_index": 2, JOB_TARGET_PARAM: source},
                   origin={"type": "chat", "cid": "2"})
    return JobSnapshot(id="chat1", spec=spec, state=JobState.CANCELLED, created=1.0, started=1.0, finished=2.0,
                       progress=Progress(total=10, completed=4))


def _chat_runs(desktop_store_cls, tmp_path, service):
    adapter = _adapter(desktop_store_cls, tmp_path, save_delay=0.01)
    runs = ChatRuns(adapter, JobsAdapter(service), temp_dir=str(tmp_path))
    runs.attach()
    return adapter, runs


def test_a_chat_turn_keeps_its_ending_after_its_entry_left_finished(desktop_store_cls, isolated_env):
    """The same book started from the Library: the turn's job leaves Finished, and the turn still reads "Stopped"
    (with its Resume), in this run and after a relaunch."""
    tmp_path = isolated_env
    _desktop_history(tmp_path)
    service, backend = make_service(tmp_path)
    book = epub(tmp_path)
    with service._lock:
        service._history = [_stopped_chat_job(tmp_path, book)]
    adapter, runs = _chat_runs(desktop_store_cls, tmp_path, service)
    gate = Gate()
    try:
        assert runs.persisted_job("2", 2).id == "chat1"
        backend.behavior = gate
        service.submit(JobSpec("translate", "Book", (book,), origin={"type": "library"}))
        assert gate.entered.wait(TIMEOUT)
        assert "chat1" not in [s.id for s in service.view().history]  # it left Finished

        remembered = runs.persisted_job("2", 2)
        assert remembered is not None and remembered.id == "chat1"
        assert ended_card(ended_kind(remembered), remembered) == ("stopped", "Stopped · 4/10 chapters")
        assert runs.last_job_ending("2")[0] == "stopped"
        assert [index for _key, index in runs.chat_job_turns("2")] == [2]  # Delete message still moves its turn
        gate.release.set()
        assert service.wait_idle(TIMEOUT)
    finally:
        gate.release.set()
        adapter.close()
        service.close()

    relaunched, _backend = make_service(tmp_path)
    adapter, runs = _chat_runs(desktop_store_cls, tmp_path, relaunched)
    try:
        remembered = runs.persisted_job("2", 2)
        assert remembered is not None and ended_kind(remembered) == "stopped"
    finally:
        adapter.close()
        relaunched.close()


def _marks(folder, *statuses):
    (Path(folder) / "translation_progress.json").write_text(json.dumps({"chapters": {
        str(i): {"status": status} for i, status in enumerate(statuses, 1)}}), encoding="utf-8")


def _book_job(kind, source, state=JobState.RUNNING):
    spec = JobSpec(kind, "Book", (source,), origin={"type": "library", "bid": "b1"})
    return JobSnapshot(id=f"lib-{kind}", spec=spec, state=state, created=time.time(), started=time.time())


@needs_flet
@pytest.mark.parametrize("kind", ["translate", "retranslate"])
def test_the_result_cards_chip_rereads_when_a_job_elsewhere_ends(desktop_store_cls, isolated_env, kind):
    """A Library or Reader job fixed the book's QA-failed chapters: the chat's Result card reads "2 QA failed" until
    the chat is rebuilt, because neither its cache key nor the chat's own runs saw that job."""
    from glossarion_mobile.ui.chat.cards import JobCard

    tmp_path = isolated_env

    def result_card(view):
        return next(c for c in _cards(view, JobCard) if c.title_text.value == "book.epub")

    async def scenario():
        chat = _chat_view(desktop_store_cls, tmp_path)
        view, adapter, calls = chat.view, chat.adapter, chat.calls
        try:
            await _settle(calls)
            workspace = Path(adapter.messages("2")[3][4])
            source = adapter.messages("2")[2][2]
            _marks(workspace, "qa_failed", "qa_failed", "completed")
            view._drop_extras("2")
            view.render_transcript()
            await _settle(calls)
            assert result_card(view).failed_chip.label.value == "2 QA failed"
            assert result_card(view).failed_chip.visible

            running = _book_job(kind, source)
            view._on_job_snapshot(running)
            await _settle(calls)
            _marks(workspace, "completed", "completed", "completed")
            done = replace(running, state=JobState.DONE, finished=time.time())
            view._on_job_snapshot(done)
            await _settle(calls)
            assert not result_card(view).failed_chip.visible

            # the same ending delivered again (the view's subscribers) reads nothing and repaints nothing
            _marks(workspace, "qa_failed", "completed", "completed")
            view._on_job_snapshot(done)
            await _settle(calls)
            assert not result_card(view).failed_chip.visible
        finally:
            _close(chat)

    asyncio.run(scenario())
