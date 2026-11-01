"""What the review of mobile batch 9.15.11 found (issues 2 to 5).

(a) The QA scan a translation's Multipass refinement runs gets the higher worker default, so it also needs the turns
    and the short switch interval of a QA scan job.
(b) An entry a newer job of its book took out of Finished still counts as what the book's last translation was: the
    chat's Result card and the Book page's "Last job" read it.
(c) That job only replaces the entry while it does the work. A QA scan or a compile does not resume a Stopped
    translation: Resume starts the translation, and the entry and its Resume return when the job ended.
(d) A job cancelled before it ran replaced nothing: the book's real result returns, with its Open and Share.
(e) The QA screen stays under Settings (its AI Hunter button): Start must not write back what it was built with.
(f) Android names one folder /data/user/0/... and /data/data/...: the live counts compare the real folder.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("GLOSSARION_HTTP_LOG", "0")
HERE = Path(__file__).resolve().parent
for path in (HERE.parent / "app", HERE.parent.parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from glossarion_mobile.services.jobs import FAMILY_TRANSLATION, JobState, Progress  # noqa: E402
from test_jobs import TIMEOUT, epub, make_service, wait_for  # noqa: E402
from test_m11_job_book_entries import TITLE, _ended, _ids, _library, _spec, _store  # noqa: E402
from test_m11_job_book_entries import scratch_env  # noqa: E402,F401  (the fixture)
from test_m11_qa_scan_lag import _TB, needs_backend, needs_flet, phone  # noqa: E402,F401


def _qa_of(source, workspace):
    return _spec("qa_scan", TITLE, workspace, params={"mode": "quick-scan",
                                                     "targets": [{"folder": workspace, "source": source}]},
                 origin={"type": "chat", "cid": "3"})


def _chat_stopped(job_id, source, tmp_path, created=1.0):
    from glossarion_mobile.services.jobs import JOB_TARGET_PARAM

    spec = _spec("direct_text", "book.epub", str(tmp_path / "run" / "x.txt"), params={JOB_TARGET_PARAM: source},
                 origin={"type": "chat", "cid": "3"})
    return _ended(job_id, spec, JobState.CANCELLED, created=created, progress=Progress(total=10, completed=5))


# ---- (a) the Multipass scan ----------------------------------------------------------------------------------


@needs_backend
def test_the_multipass_scan_of_a_translation_job_takes_turns(phone, monkeypatch):
    """The real ``run_translation`` around a worker that calls the backend's real Multipass function, down to a
    recording ``run_qa_scan_path`` (it imports it per call)."""
    import qa_scan_runtime as qsr
    import scan_html_folder as shf
    import TransateKRtoEN as backend

    from glossarion_mobile.job_kinds import qa as qa_kind
    from glossarion_mobile.job_kinds import translate as translate_kind

    real_submit, real_multipass = shf._submit_worker, backend._run_multipass_refinement_qa_scan
    seen: dict = {}

    def recording_scan(*_args, **_kwargs):
        seen["interval"] = sys.getswitchinterval()
        seen["submit"] = shf._submit_worker
        return {"ok": True}

    monkeypatch.setattr(qsr, "run_qa_scan_path", recording_scan)
    monkeypatch.setattr(translate_kind, "record_output_dirs", lambda _ctx, _files: {})
    monkeypatch.setattr(translate_kind, "indexing_phase", lambda _ctx: _no_scope())
    monkeypatch.setattr(translate_kind, "post_translation_qa_trigger", lambda _ctx: _no_scope([]))
    outside = []

    def worker(_request):
        outside.append(sys.getswitchinterval())  # the translation itself runs as it did
        backend._run_multipass_refinement_qa_scan(str(phone), stop_flag=lambda: False, mode="quick-scan", qa_settings={},
                                                  epub_path="", progress_path="", config=None)
        return {"ok": True, "outputs": []}

    owner = SimpleNamespace(_prepare_translation_run=lambda _files: {"files": []}, _translation_worker=worker)
    ctx = SimpleNamespace(owner=owner, phase=lambda _label: None, stop_requested=lambda: False, log=lambda _text: None,
                          add_outputs=lambda _paths: None, output_dir=str(phone))
    previous = sys.getswitchinterval()
    sys.setswitchinterval(0.005)
    try:
        translate_kind.run_translation(ctx, ["/b/x.epub"])
        assert seen["interval"] == pytest.approx(qa_kind.SCAN_SWITCH_INTERVAL)
        assert seen["submit"] is not real_submit and outside == [pytest.approx(0.005)]
        # all restored, the interval after the scan too
        assert shf._submit_worker is real_submit and backend._run_multipass_refinement_qa_scan is real_multipass
        assert sys.getswitchinterval() == pytest.approx(0.005)
    finally:
        sys.setswitchinterval(previous)


class _no_scope:
    def __init__(self, value=None):
        self.value = value

    def __enter__(self):
        return self.value

    def __exit__(self, *_exc):
        return False


# ---- (b) the book's latest job -----------------------------------------------------------------------------


def test_a_result_card_keeps_the_books_newest_job_through_a_qa_scan(tmp_path):
    """The chat's job stopped at 5/10, the Library's finished the book, then a QA scan of the book."""
    source, workspace = epub(tmp_path), str(tmp_path / "out" / "Book")
    service, _ = make_service(tmp_path)
    _store(service, _ended("lib", _library(source), created=2.0, output_dir=workspace, progress=Progress(total=10, completed=10)),
           _chat_stopped("chat", source, tmp_path))
    with service._lock:  # as the Library's job left it: the chat's entry replaced by the job that did its work
        service._carry_on_locked({"chat": ("lib", True)})
    assert service.latest_job_for(source, FAMILY_TRANSLATION).id == "lib"
    service.pause_queue()
    scan = service.submit(_qa_of(source, workspace))
    assert _ids(service) == []
    assert service.latest_job_for(source, FAMILY_TRANSLATION).id == "lib"  # was None: the card fell back to Stopped
    service.cancel_queued(scan)
    assert service.latest_job_for(source, FAMILY_TRANSLATION).id == "lib"
    service.close()


@needs_flet
def test_the_book_pages_last_job_survives_a_chats_qa_scan(tmp_path):
    from glossarion_mobile.ui.library.overview_tab import OverviewTab

    source, workspace = epub(tmp_path), str(tmp_path / "out" / "Book")
    service, _ = make_service(tmp_path)
    _store(service, _ended("lib", _library(source, bid="b1"), output_dir=workspace))
    service.pause_queue()
    scan = service.submit(_qa_of(source, workspace))
    view = service.view()
    assert not view.history
    tab = SimpleNamespace(page=SimpleNamespace(job_view=view, bid="b1"))
    assert OverviewTab._last_job(tab).id == "lib"
    service.cancel_queued(scan)
    service.close()


# ---- (c) Resume of an entry another kind of job replaced ----------------------------------------------------


def test_resume_of_a_stopped_translation_is_not_answered_by_a_qa_scan(tmp_path):
    source, workspace = epub(tmp_path), str(tmp_path / "out" / "Book")
    service, backend = make_service(tmp_path)
    _store(service, _ended("stopped", _library(source), JobState.CANCELLED, output_dir=workspace))
    service.pause_queue()
    scan = service.submit(_qa_of(source, workspace))
    assert _ids(service) == []
    result = service.resume_job("stopped")
    assert result is not None and not result.reused and result.job_id != scan  # was the QA scan, reused
    assert result.snapshot.spec.kind == "translate"
    assert [s.id for s in service.view().queue] == [scan, result.job_id]
    assert service.resume_job("stopped").job_id == result.job_id  # a double tap
    service.cancel_queued(result.job_id)
    service.cancel_queued(scan)
    service.close()


def test_the_entry_and_its_resume_return_when_a_scan_of_the_book_ended(tmp_path, monkeypatch):
    from dataclasses import replace

    from glossarion_mobile import job_kinds

    source, workspace = epub(tmp_path), str(tmp_path / "out" / "Book")
    service, backend = make_service(tmp_path)
    _store(service, _ended("stopped", _library(source), JobState.CANCELLED, output_dir=workspace),
           _ended("done", _library(source), created=0.5, output_dir=workspace))
    entered, release = threading.Event(), threading.Event()

    def scan_run(_ctx):
        entered.set()
        release.wait(TIMEOUT)
        return {"ok": True, "outputs": []}

    monkeypatch.setitem(job_kinds._CACHE, "qa_scan", replace(job_kinds.get_kind("qa_scan"), run=scan_run))
    scan = service.submit(_qa_of(source, workspace))
    assert entered.wait(TIMEOUT)
    assert _ids(service) == []
    release.set()
    assert service.wait_idle(TIMEOUT)
    assert wait_for(lambda: _ids(service) == [scan, "stopped"]), _ids(service)  # a Done entry stays replaced
    assert service.snapshot("stopped").resolution is None and service.snapshot("done") is not None
    service.pause_queue()
    assert service.resume_job("stopped").snapshot.spec.kind == "translate"
    service.resume_queue()
    assert service.wait_idle(TIMEOUT)
    service.close()


def test_a_job_that_does_the_work_keeps_the_entry_out_for_good(tmp_path):
    source = epub(tmp_path)
    service, _ = make_service(tmp_path)
    _store(service, _ended("stopped", _library(source), JobState.CANCELLED))
    new = service.submit(_library(source))
    assert service.wait_idle(TIMEOUT)
    assert _ids(service) == [new]
    assert service.resume_job("stopped") is None  # as on main: its work went on under the new job
    service.close()


def test_a_scan_killed_with_the_app_gives_the_entry_back_at_the_next_launch(tmp_path):
    source, workspace = epub(tmp_path), str(tmp_path / "out" / "Book")
    service, _ = make_service(tmp_path)
    _store(service, _ended("stopped", _library(source), JobState.CANCELLED, output_dir=workspace))
    service.pause_queue()
    service.submit(_qa_of(source, workspace))
    assert _ids(service) == []
    saved = json.loads(Path(service.history_state_path).read_text(encoding="utf-8"))
    interrupted_state = Path(service.active_state_path).read_bytes()
    service.close()
    Path(service.active_state_path).write_bytes(interrupted_state)  # close() cleaned up what a kill would leave
    assert saved["carried_by"]["stopped"]["covers"] is False

    again, _ = make_service(tmp_path)
    assert _ids(again) == ["stopped"] and again.snapshot("stopped").resolution is None
    assert len(again.interrupted) == 1
    again.close()


# ---- (d) a job cancelled before it ran ----------------------------------------------------------------------


def test_cancelling_a_queued_job_gives_the_books_result_back(tmp_path):
    source = epub(tmp_path)
    service, _ = make_service(tmp_path)
    _store(service, _ended("done", _library(source), outputs=("/b/x.epub",)))
    service.pause_queue()
    new = service.submit(_library(source))
    assert _ids(service) == []
    assert service.cancel_queued(new)
    assert _ids(service) == [new, "done"]
    assert service.view().find("done").outputs == ("/b/x.epub",) and service.snapshot("done").resolution is None
    service.close()

    service, _ = make_service(tmp_path)  # also after a restart
    assert _ids(service) == [new, "done"]
    service.close()


def test_cancelling_one_of_two_queued_jobs_keeps_the_entry_out_for_the_other(tmp_path):
    source, workspace = epub(tmp_path), str(tmp_path / "out" / "Book")
    service, _ = make_service(tmp_path)
    _store(service, _ended("done", _library(source), output_dir=workspace))
    service.pause_queue()
    first = service.submit(_qa_of(source, workspace))
    second = service.submit(_library(source))
    assert service.cancel_queued(first)
    assert _ids(service) == []
    assert service.cancel_queued(second)
    assert set(_ids(service)) == {first, second, "done"}
    service.close()


# ---- (e) the QA screen under Settings -----------------------------------------------------------------------


@needs_flet
@needs_backend
def test_back_from_settings_the_qa_screen_does_not_write_its_old_numbers_back(phone, monkeypatch):
    from glossarion_mobile.ui.router import parse_route
    from glossarion_mobile.ui.tools.qa_screen import QaScannerScreen

    monkeypatch.setattr(os, "cpu_count", lambda: 8)
    ui = _TB_tools()
    target = ui.tool_target(phone, "Alpha")

    async def scenario():
        store = {"qa_scanner_settings": {"check_word_count_ratio": False, "quick_scan_sample_size": 500},
                 "ai_hunter_config": {"ai_hunter_max_workers": 4}}
        _conn, session = _TB._fake_session("android")
        ctx = ui._ctx(session.page, store=store)
        screen = QaScannerScreen(parse_route("/tools/qa"), ctx)
        ui._mount(session.page, screen.get_body())
        screen.set_targets([target])
        assert screen.workers_field.value == "4" and screen.sample_field.value == "500"
        # Settings, on top of this screen
        store["ai_hunter_config"]["ai_hunter_max_workers"] = 2
        store["qa_scanner_settings"]["quick_scan_sample_size"] = 700
        screen.did_show()
        assert screen.workers_field.value == "2" and screen.sample_field.value == "700"
        assert await screen.start()
        assert store["ai_hunter_config"]["ai_hunter_max_workers"] == 2
        assert store["qa_scanner_settings"]["quick_scan_sample_size"] == 700
        # a text the field refused stays for the owner to fix
        screen.workers_field.value = "99"
        assert not screen._on_workers()
        screen.did_show()
        assert screen.workers_field.value == "99"
        screen.dispose()

    asyncio.run(scenario())


def _TB_tools():
    from test_m11_qa_scan_lag import _load

    return _load(HERE / "test_tools_ui.py", "_glossarion_tools_ui_helpers_m11_review")


# ---- (f) one folder under two names -------------------------------------------------------------------------


def test_the_live_counts_follow_a_folder_under_its_other_android_name(tmp_path, monkeypatch):
    real_realpath = os.path.realpath
    long_name, short_name = str(tmp_path / "data_user_0" / "Book"), str(tmp_path / "data_data" / "Book")

    def realpath(path, **kwargs):  # /data/user/0 is a link to /data/data
        return real_realpath(str(path).replace(str(tmp_path / "data_user_0"), str(tmp_path / "data_data")), **kwargs)

    monkeypatch.setattr(os.path, "realpath", realpath)
    source = epub(tmp_path)
    service, backend = make_service(tmp_path)
    _store(service, _ended("old", _spec("translate", TITLE, source), output_dir=long_name, created=0.5,
                           output_dirs={source: long_name}, progress=Progress(total=10, completed=6, failed=4)))
    service._follow_progress({"path": os.path.join(short_name, "translation_progress.json"), "total": 10, "completed": 6,
                              "in_progress": 3, "failed": 1})
    snap = service.snapshot("old")
    assert (snap.progress.in_progress, snap.progress.failed) == (3, 1)
    service.close()
