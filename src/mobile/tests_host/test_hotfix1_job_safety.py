"""Hotfix 1 issue 5 (+ the JobService half of issue 2): job thread safety.

Owner (device report): several separate jobs could be sent for the same attachment; they must reuse the
same job so the same request is never sent twice. And ✨ Refine on the glossary approval card must keep the
translation from starting until the refinement ended.

* ``JobService.submit`` / ``submit_job`` reuse an ACTIVE job (queued, starting, running, waiting at the
  glossary gate, refining, stopping) for the same request (``spec_key``) or, in the same job family, a
  target it covers (``job_target``) - check and insert under one lock (real threads hammer it here);
* ``active_job_for`` finds that job before anything is recorded;
* ``resume`` / ``resume_job`` of an active job (or of one already resumed into an active job) return it;
* gate answers are idempotent (the first one counts) and refused while ``refine_at_gate`` runs; the job
  waits at the gate until the refinement ended, also when stopped meanwhile;
* ``ChatRuns``: a second Send / Resume of a chat binds to its live run, a second chat (or the Library)
  never starts a second job for the same file, ``answer_glossary`` is refused while refining.

Run from src/mobile with the mobile venv:
    .venv/Scripts/python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic \
        tests_host/test_hotfix1_job_safety.py
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import sys
import tempfile
import threading
import time
import types
from pathlib import Path

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = MOBILE_DIR / "app"
SRC_DIR = MOBILE_DIR.parent  # the shared GUI-free modules (direct_text_store for the chat's run options)
for entry in (str(APP_DIR), str(SRC_DIR)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from glossarion_mobile.services import jobs as jobs_mod  # noqa: E402
from glossarion_mobile.services.jobs import (  # noqa: E402
    FAMILY_CHAPTER,
    FAMILY_QA,
    FAMILY_TRANSLATION,
    JOB_TARGET_PARAM,
    JobAlreadyActive,
    JobService,
    JobSpec,
    JobState,
    already_active_text,
    job_target,
    spec_key,
)

TIMEOUT = 10.0


# ==========================================================================
# Fakes: a JobService backend and job kinds without the shared pipeline
# ==========================================================================


class Backend:
    """The ``JobBackend`` surface JobService uses (job_runner / stop_control contracts, no env writes)."""

    def __init__(self) -> None:
        self.job_lock = threading.RLock()
        self.stops: list = []

    @contextlib.contextmanager
    def scoped_process_state(self, *, capture_stdout, lock):
        with lock:
            yield

    def make_owner(self, config, *, host):
        return types.SimpleNamespace(config=config, host=host, stop_requested=False, graceful_stop_var=True,
                                     wait_for_chunks_var=True)

    def reset_for_new_run(self, kind):
        return 1

    def request_stop(self, *, graceful, set_stop_requested, force=False, **kwargs):
        self.stops.append("force" if force else ("graceful" if graceful else "immediate"))
        set_stop_requested()

    def progress_watcher(self, resolver, host, interval):
        return None


class Kinds:
    """Job kinds whose run waits on ``release`` (or a Stop), asks the glossary gate when the spec says so and
    records what ran, in order."""

    def __init__(self) -> None:
        self.release = threading.Event()
        self.started = threading.Event()
        self.events: list = []
        self.answers: list = []
        self.lock = threading.Lock()

    def note(self, what: str) -> None:
        with self.lock:
            self.events.append(what)

    def run(self, ctx):
        self.started.set()
        self.note(f"start:{ctx.job_id}")
        if ctx.params.get("ask"):
            answer = ctx.ask("glossary_approval", path=ctx.params.get("glossary") or "g.csv", default=None)
            self.answers.append(answer)
            if not answer:
                return None
            self.note(f"translate:{ctx.job_id}")
        while not self.release.wait(0.01):
            if ctx.stop_requested():
                break
        self.note(f"end:{ctx.job_id}")
        return None

    def __call__(self, kind):
        return types.SimpleNamespace(run=self.run, stop_kind="translation", verb="Translating")


def make_service(tmp_path, kinds=None):
    kinds = kinds or Kinds()
    service = JobService(jobs_dir=tmp_path / "jobs", logs_dir=tmp_path / "logs", backend=Backend(), kinds=kinds)
    return service, kinds


def book(tmp_path, name="Book.epub") -> str:
    path = tmp_path / "in" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"PK fake epub")
    return str(path)


def wait_for(predicate, timeout=TIMEOUT) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def translate(path, title="Book", **params):
    return JobSpec("translate", title, (path,), params=params, origin={"type": "library", "label": "Library · Book"})


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """Every probe stays in tmp_path (no Library, output root or home of the real user)."""
    for name in ("HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA"):
        monkeypatch.setenv(name, str(tmp_path / "home"))
    monkeypatch.setenv("GLOSSARION_LIBRARY_DIR", str(tmp_path / "library"))
    monkeypatch.setenv("OUTPUT_DIRECTORY", str(tmp_path / "output"))
    monkeypatch.setenv("GLOSSARION_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("GLOSSARION_HTTP_LOG", "0")
    return tmp_path


# ==========================================================================
# Targets and keys (pure)
# ==========================================================================


def test_job_targets_and_families(isolated):
    tmp = isolated
    a, b = book(tmp, "A.epub"), book(tmp, "B.epub")
    target = job_target(translate(a))
    assert target.family == FAMILY_TRANSLATION and len(target.keys) == 1
    # a chat attachment send names its file; a typed-text send (fresh run input) has no target
    chat = JobSpec("direct_text", "Chat", (str(tmp / "run" / "A.txt"),), params={JOB_TARGET_PARAM: [a]})
    assert job_target(chat) == target
    assert job_target(JobSpec("direct_text", "Chat", (str(tmp / "run" / "typed.txt"),))) is None
    # the same file under another spelling is the same target
    assert job_target(translate(os.path.join(os.path.dirname(a), ".", "A.epub"))) == target
    # a two-book run covers each book, a one-book run does not cover the pair
    pair = job_target(JobSpec("translate", "A +1", (a, b)))
    assert pair.covers(target) and not target.covers(pair)
    # other families and kinds
    assert job_target(JobSpec("qa_scan", "QA", (str(tmp / "ws"),))).family == FAMILY_QA
    chapter = job_target(JobSpec("single_chapter", "A · c1", (a,), params={"chapter_file": "c1.xhtml"}))
    assert chapter.family == FAMILY_CHAPTER and not chapter.covers(target) and not target.covers(chapter)
    assert job_target(JobSpec("compile_epub", "A", (str(tmp / "ws"),))) is None
    # the very same request, wherever it was tapped (title and origin do not count)
    assert spec_key(translate(a, "x")) == spec_key(JobSpec("translate", "y", (a,), origin={"type": "chat"}))
    assert spec_key(translate(a)) != spec_key(translate(a, review_glossary=True))


# ==========================================================================
# JobService: reuse, under real threads
# ==========================================================================


def test_concurrent_submits_for_one_book_make_one_job(isolated):
    service, kinds = make_service(isolated)
    path = book(isolated)
    threads, results = [], []
    barrier = threading.Barrier(16)

    def hammer(index):
        barrier.wait()
        if index % 3 == 0:  # a chat attachment send of the same file
            spec = JobSpec("direct_text", "Book.epub", (str(isolated / "run" / f"{index}.epub"),),
                           params={JOB_TARGET_PARAM: [path], "chat_id": index})
        elif index % 3 == 1:  # Library / Progress Manager / QA card "Run translation"
            spec = translate(path)
        else:  # the chat's "Save to: Library" plan
            spec = JobSpec("translate", "Book.epub", (path,), params={"config_overrides": {"x": index}},
                           origin={"type": "chat", "cid": str(index)})
        results.append(service.submit_job(spec))

    for index in range(16):
        thread = threading.Thread(target=hammer, args=(index,))
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join(TIMEOUT)
    assert len(results) == 16
    assert len({r.job_id for r in results}) == 1, sorted({r.job_id for r in results})
    assert sum(1 for r in results if not r.reused) == 1  # exactly one submission queued a job
    assert all(r.snapshot is not None and r.snapshot.id == results[0].job_id for r in results)
    view = service.view()
    assert len(view.queue) + (1 if view.active is not None else 0) == 1
    kinds.release.set()
    assert service.wait_idle(TIMEOUT)
    assert [e for e in kinds.events if e.startswith("start:")] == [f"start:{results[0].job_id}"]
    service.close()


@pytest.mark.parametrize("phase", ["queued", "running", "gate", "refining", "stopping"])
def test_an_active_job_is_reused_in_every_phase(isolated, phase):
    service, kinds = make_service(isolated)
    service.on_question(lambda snap, question: None)  # the chat's card listens: the gate waits
    path = book(isolated)
    params = {"ask": True} if phase in ("gate", "refining") else {}
    if phase == "queued":
        service.pause_queue()
    first = service.submit(translate(path, **params))
    if phase != "queued":
        assert kinds.started.wait(TIMEOUT)
    if phase in ("gate", "refining"):
        assert wait_for(lambda: service.pending_question(first) is not None)
    if phase == "refining":
        service._find_live(first).refining = True  # what refine_at_gate holds while it refines
    if phase == "stopping":
        assert service.request_stop(first) == "graceful"
        assert service.snapshot(first).state is JobState.STOPPING
    again = service.submit_job(translate(path, **params))
    assert again.reused and again.job_id == first
    # the same book from the chat (an attachment send) and the "Save to: Library" plan
    chat = service.submit_job(JobSpec("direct_text", "Book.epub", (str(isolated / "x.epub"),),
                                      params={JOB_TARGET_PARAM: [path]}))
    assert chat.reused and chat.job_id == first
    assert service.active_job_for(path).id == first
    assert service.active_job_for(path, family=FAMILY_QA) is None
    assert service.active_job_for(translate(path)).id == first
    # another book, another family: queued after it
    other = service.submit_job(translate(book(isolated, "Other.epub")))
    glossary = service.submit_job(JobSpec("extract_glossary", "Book", (path,)))
    assert not other.reused and not glossary.reused and len({first, other.job_id, glossary.job_id}) == 3
    # nothing ran twice: one job per book and family
    service.cancel_all_queued()
    if phase == "refining":
        service._find_live(first).refining = False
    if phase == "queued":
        service.resume_queue()
    service.request_stop(first, force=True)
    kinds.release.set()
    assert service.wait_idle(TIMEOUT)
    # ended: a new submission queues a new job
    assert service.active_job_for(path) is None
    fresh = service.submit_job(translate(path))
    assert not fresh.reused and fresh.job_id != first
    assert service.wait_idle(TIMEOUT)
    service.close()


def test_identical_requests_of_any_kind_are_reused(isolated):
    service, kinds = make_service(isolated)
    folder = str(isolated / "ws")
    blocker = service.submit(translate(book(isolated, "Blocker.epub")))
    assert kinds.started.wait(TIMEOUT)
    first = service.submit_job(JobSpec("compile_epub", "Book", (folder,), params={"folder": folder}))
    twice = service.submit_job(JobSpec("compile_epub", "Book (again)", (folder,), params={"folder": folder},
                                       origin={"type": "chat"}))
    pdf = service.submit_job(JobSpec("compile_pdf", "Book", (folder,), params={"folder": folder}))
    assert not first.reused and twice.reused and twice.job_id == first.job_id and not pdf.reused
    # typed-text chat sends never collapse (each has its own run input and turn)
    typed = [service.submit_job(JobSpec("direct_text", "Chat", (str(isolated / f"t{i}.txt"),),
                                        params={"user_index": i})) for i in range(2)]
    assert not any(r.reused for r in typed) and typed[0].job_id != typed[1].job_id
    service.cancel_all_queued()
    service.request_stop(blocker, force=True)
    assert service.wait_idle(TIMEOUT)
    service.close()


def test_resume_of_an_active_job_returns_it(isolated):
    service, kinds = make_service(isolated)
    path = book(isolated)
    first = service.submit(translate(path))
    assert kinds.started.wait(TIMEOUT)
    assert service.resume(first) == first
    result = service.resume_job(first)
    assert result.reused and result.job_id == first
    assert service.request_stop(first) == "graceful"
    assert service.resume(first) == first  # still stopping: the same job
    assert service.wait_idle(TIMEOUT)
    assert service.snapshot(first).state is JobState.CANCELLED

    # a double Resume of the stopped job from several surfaces at once: one new job
    kinds.started.clear()
    barrier = threading.Barrier(8)
    ids = []

    def resume():
        barrier.wait()
        ids.append(service.resume(first))

    threads = [threading.Thread(target=resume) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(TIMEOUT)
    assert len(ids) == 8 and len(set(ids)) == 1 and ids[0] != first
    resumed = ids[0]
    # a Library "Run translation" of the same book while the resumed run is active: that run again
    assert service.submit(translate(path)) == resumed
    kinds.release.set()
    assert service.wait_idle(TIMEOUT)
    starts = [e for e in kinds.events if e.startswith("start:")]
    assert starts == [f"start:{first}", f"start:{resumed}"]
    service.close()


# ==========================================================================
# The glossary gate: idempotent answers, refused while refining (issue 2)
# ==========================================================================


def test_gate_answers_are_idempotent_under_threads(isolated):
    service, kinds = make_service(isolated)
    service.on_question(lambda snap, question: None)
    kinds.release.set()
    job_id = service.submit(translate(book(isolated), ask=True))
    assert wait_for(lambda: service.pending_question(job_id) is not None)
    question = service.pending_question(job_id)
    barrier = threading.Barrier(12)
    results = []

    def tap(value):
        barrier.wait()
        results.append((value, service.answer(question["id"], value)))

    threads = [threading.Thread(target=tap, args=(index % 2 == 0,)) for index in range(12)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(TIMEOUT)
    assert service.wait_idle(TIMEOUT)
    assert len(kinds.answers) == 1
    got = kinds.answers[0]
    # every tap with the answer the job got is a success (the repeats are no-ops), every other one refused
    assert all(ok == (value == got) for value, ok in results), (got, results)
    # later taps on a stale card / notification: the same answer stays a no-op, the other one is refused
    assert service.answer(question["id"], got) is True and service.answer(question["id"], not got) is False
    assert service.answer("no-such-question", True) is False
    service.close()


def _fake_refinement(monkeypatch, kinds, hold: threading.Event, honour_stop: bool = True):
    module = types.ModuleType("glossary_progress_core")

    def plan_manual_glossary_refinement(owner, glossary_path=None, progress_path=None, source_path=None,
                                        selected_types=None, target_chunk_count=None, log=None):
        return {"types": list(selected_types or ())}, ["plan"]

    def run_manual_glossary_refinement(owner, glossary_path, progress_path, options, plan):
        kinds.note("refine:start")
        while not hold.wait(0.01):
            if honour_stop and getattr(owner, "stop_requested", False):
                break  # the shared refinement polls the owner's stop flag
        kinds.note("refine:end")

    module.plan_manual_glossary_refinement = plan_manual_glossary_refinement
    module.run_manual_glossary_refinement = run_manual_glossary_refinement
    monkeypatch.setitem(sys.modules, "glossary_progress_core", module)


def test_yes_during_refinement_is_refused_and_the_translation_waits_for_it(isolated, monkeypatch):
    service, kinds = make_service(isolated)
    hold = threading.Event()
    _fake_refinement(monkeypatch, kinds, hold)
    events = []
    service.on_event(lambda job_id, kind, data: events.append(kind))
    service.on_question(lambda snap, question: None)
    kinds.release.set()
    glossary = isolated / "Book_glossary.csv"
    glossary.write_text("type,raw_name,translated_name\n", encoding="utf-8")
    job_id = service.submit(translate(book(isolated), ask=True, glossary=str(glossary)))
    assert wait_for(lambda: service.pending_question(job_id) is not None)
    question = service.pending_question(job_id)

    refined = {}
    worker = threading.Thread(target=lambda: refined.setdefault("result", service.refine_at_gate(job_id, str(glossary))))
    worker.start()
    assert wait_for(lambda: "refine:start" in kinds.events)
    snap = service.snapshot(job_id)
    assert snap.refining and service.is_refining(job_id)
    assert jobs_mod.QUESTION_SUSPENDED_EVENT in events  # the card and its alerts are withdrawn
    # ✓ Yes (card, notification, Jobs sheet) while it refines: refused, from any thread, any number of times
    taps = [threading.Thread(target=lambda: refined.setdefault("taps", []).append(service.answer(question["id"], True)))
            for _ in range(6)]
    for tap in taps:
        tap.start()
    for tap in taps:
        tap.join(TIMEOUT)
    assert refined["taps"] == [False] * 6
    with pytest.raises(RuntimeError, match="already being refined"):
        service.refine_at_gate(job_id, str(glossary))
    # a second submission of the book meanwhile reuses the waiting job
    assert service.submit(translate(book(isolated), ask=True, glossary=str(glossary))) == job_id
    time.sleep(0.3)
    assert not any(e.startswith("translate:") for e in kinds.events)  # nothing started
    hold.set()
    worker.join(TIMEOUT)
    assert refined["result"]["ok"] and not service.snapshot(job_id).refining
    assert service.pending_question(job_id) is not None  # the gate still waits, now for the refined file
    assert service.answer(question["id"], True)
    assert service.answer(question["id"], True)  # a repeated Yes is a no-op
    assert service.wait_idle(TIMEOUT)
    order = kinds.events
    assert order.index("refine:end") < order.index(f"translate:{job_id}"), order
    assert kinds.answers == [True]
    # the gate was answered: Refine is refused from now on
    with pytest.raises(RuntimeError):
        service.refine_at_gate(job_id, str(glossary))
    service.close()


def test_a_stop_during_refinement_waits_for_it_before_the_job_ends(isolated, monkeypatch):
    service, kinds = make_service(isolated)
    hold = threading.Event()
    _fake_refinement(monkeypatch, kinds, hold, honour_stop=False)
    service.on_question(lambda snap, question: None)
    glossary = isolated / "Book_glossary.csv"
    glossary.write_text("type,raw_name,translated_name\n", encoding="utf-8")
    job_id = service.submit(translate(book(isolated), ask=True, glossary=str(glossary)))
    assert wait_for(lambda: service.pending_question(job_id) is not None)
    worker = threading.Thread(target=lambda: service.refine_at_gate(job_id, str(glossary)))
    worker.start()
    assert wait_for(lambda: "refine:start" in kinds.events)
    # Stop while the refinement still sends its requests: the job keeps its owner and process state (and the
    # next job waits) until the refinement ended
    assert service.request_stop(job_id) == "graceful"
    time.sleep(0.4)
    snap = service.snapshot(job_id)
    assert snap.state is JobState.STOPPING and snap.refining and snap.finished is None
    assert "refine:end" not in kinds.events
    hold.set()
    worker.join(TIMEOUT)
    assert service.wait_idle(TIMEOUT)
    assert service.snapshot(job_id).state is JobState.CANCELLED
    assert "refine:end" in kinds.events and not any(e.startswith("translate:") for e in kinds.events)
    service.close()


def test_already_active_text():
    snap = jobs_mod.JobSnapshot(id="j1", spec=JobSpec("translate", "Book.epub", origin={"label": "Chat · Novel"}),
                                state=JobState.QUEUED, created=0.0)
    assert already_active_text(snap) == "Translating “Book.epub” is already queued (Chat · Novel)"
    error = JobAlreadyActive(snap)
    assert error.job_id == "j1" and str(error) == already_active_text(snap)
    assert str(JobAlreadyActive(None, "custom")) == "custom" and JobAlreadyActive().job_id is None


# ==========================================================================
# ChatRuns over a real JobService
# ==========================================================================


class ChatStore:
    """The ``ChatStoreAdapter`` calls ChatRuns makes (in memory)."""

    def __init__(self) -> None:
        self.chats: dict = {}
        self.running: dict = {}
        self.lock = threading.Lock()

    def _chat(self, cid):
        return self.chats.setdefault(str(cid), [])

    def record_user_turn(self, cid, turn, display):
        with self.lock:
            messages = self._chat(cid)
            messages.append(tuple(turn))
            return len(messages) - 1

    def messages(self, cid):
        return list(self._chat(cid))

    def append_messages(self, cid, messages):
        with self.lock:
            self._chat(cid).extend(tuple(m) for m in messages)

    def session(self, cid):
        return {"id": cid, "title": f"Chat {cid}"}

    def request_count(self, cid):
        return sum(1 for m in self._chat(cid) if m and m[0] == "assistant")

    def set_running(self, cid, value):
        self.running[str(cid)] = bool(value)

    def overrides(self, cid):
        return {}

    def flush(self):
        return None

    def schedule_save(self):
        return None

    def forget_bodies(self, cid):
        return None

    def refresh_attachments(self, cid):
        return None

    def job_turns(self, cid):
        return {}

    def fingerprints(self, cid):
        return [repr(m) for m in self._chat(cid)]


@pytest.fixture
def chat(isolated, monkeypatch):
    from glossarion_mobile.ui.chat import run_controller
    from glossarion_mobile.ui.chat.job_binding import JobsAdapter
    from glossarion_mobile.ui.chat.run_request import DirectTextRun

    gates: dict = {"prepare": None, "prepared": []}

    def prepare_direct_text_run(*, text, attachment, output_mode="text", attachment_prompt_role="user",
                                manual_glossary=None, temp_dir=None):
        gate = gates["prepare"]
        if gate is not None:
            assert gate.wait(TIMEOUT)
        root = tempfile.mkdtemp(prefix="glossarion_input_output_", dir=temp_dir)
        gates["prepared"].append(root)
        source = str((attachment or {}).get("path") or os.path.join(root, "typed.txt"))
        return DirectTextRun(temp_root=root, source_path=source, source_extension=os.path.splitext(source)[1],
                             is_attachment=bool(attachment), expected_output=os.path.join(root, "out.txt"),
                             display_name=os.path.basename(source) if attachment else "")

    monkeypatch.setattr(run_controller, "prepare_direct_text_run", prepare_direct_text_run)
    service, kinds = make_service(isolated)
    store = ChatStore()
    runs_dir = isolated / "runs"
    runs_dir.mkdir()

    async def run_io(fn, *args):
        return await asyncio.to_thread(fn, *args)

    runs = run_controller.ChatRuns(store, JobsAdapter(service), run_io=run_io, temp_dir=str(runs_dir))
    runs.attach()
    ns = types.SimpleNamespace(service=service, kinds=kinds, store=store, runs=runs, gates=gates, tmp=isolated)
    yield ns
    service.cancel_all_queued()
    active = service.snapshot()
    if active is not None:
        service.request_stop(active.id, force=True)
    kinds.release.set()
    service.wait_idle(TIMEOUT)
    for thread in list(runs.finish_threads):
        thread.join(TIMEOUT)
    service.close()


def _record(path):
    return {"path": str(path), "name": os.path.basename(str(path)), "extension": ".epub", "size": 12}


def _send(runs, cid, record, **kwargs):
    from glossarion_mobile.ui.chat.direct_text_rules import DirectTextSettings

    return runs.send(cid, text="", attachment=record, settings=DirectTextSettings(), output_mode="text", **kwargs)


def _live_jobs(service):
    view = service.view()
    return ([view.active] if view.active is not None else []) + list(view.queue)


def test_one_chat_one_run_and_one_job_per_file(chat):
    path = book(chat.tmp)
    record = _record(path)
    run = asyncio.run(_send(chat.runs, "1", record))
    assert run is not None and run.job_id and run.params[JOB_TARGET_PARAM] == [os.path.abspath(path)]
    assert chat.kinds.started.wait(TIMEOUT)
    turns = len(chat.store.messages("1"))
    # Send / Start again in the same chat (double tap, a second Plan Start): the live run, nothing recorded
    again = asyncio.run(_send(chat.runs, "1", record))
    assert again is run and len(chat.store.messages("1")) == turns
    # Resume while it runs: the same run
    assert asyncio.run(chat.runs.resubmit("1")) is run
    # another chat with the same file: refused before anything is recorded, naming where it runs
    with pytest.raises(JobAlreadyActive) as refused:
        asyncio.run(_send(chat.runs, "2", record))
    assert refused.value.job_id == run.job_id and "Chat 1" in str(refused.value)
    assert chat.store.messages("2") == [] and chat.runs.live_run("2") is None
    # the Library ("Run translation", QA card, Progress Manager) gets the chat's job
    assert chat.service.submit(translate(path)) == run.job_id
    # a different book in the second chat is its own job (queued after)
    other = asyncio.run(_send(chat.runs, "2", _record(book(chat.tmp, "Other.epub"))))
    assert other is not None and other.job_id != run.job_id
    assert len(_live_jobs(chat.service)) == 2


def test_concurrent_sends_of_one_file_start_one_job(chat):
    path = book(chat.tmp)
    record = _record(path)
    gate = threading.Event()
    chat.gates["prepare"] = gate

    async def both():
        first = asyncio.ensure_future(_send(chat.runs, "1", record))
        await asyncio.sleep(0.05)  # chat 1 is preparing its run
        second = asyncio.ensure_future(_send(chat.runs, "2", record))
        double = asyncio.ensure_future(_send(chat.runs, "1", record))  # chat 1's double tap
        await asyncio.sleep(0.05)
        gate.set()
        return await asyncio.gather(first, second, double, return_exceptions=True)

    first, second, double = asyncio.run(both())
    assert getattr(first, "job_id", None), first
    assert isinstance(second, JobAlreadyActive), second
    assert double is None  # chat 1's first send was still preparing: nothing else happened
    assert len(_live_jobs(chat.service)) == 1 and len(chat.gates["prepared"]) == 1
    assert len(chat.store.messages("1")) == 1 and chat.store.messages("2") == []


def test_a_job_queued_elsewhere_while_the_send_prepares_is_not_duplicated(chat):
    path = book(chat.tmp)
    blocker = chat.service.submit(translate(book(chat.tmp, "Blocker.epub")))
    assert chat.kinds.started.wait(TIMEOUT)
    gate = threading.Event()
    chat.gates["prepare"] = gate

    async def race():
        send = asyncio.ensure_future(_send(chat.runs, "3", _record(path)))
        await asyncio.sleep(0.05)
        library = chat.service.submit(translate(path))  # the Library queues the book meanwhile
        gate.set()
        try:
            await send
        except JobAlreadyActive as exc:
            return library, exc
        return library, None

    library, refused = asyncio.run(race())
    assert refused is not None and refused.job_id == library
    ids = [s.id for s in _live_jobs(chat.service)]
    assert ids == [blocker, library]  # no second job for the book
    # the turn says why nothing started; the chat is idle again and the unused run root is gone
    messages = chat.store.messages("3")
    assert messages[0][0] == "user_file" and messages[-1][0] == "assistant"
    assert "could not be started" in messages[-1][1] and "already queued" in messages[-1][1]
    assert chat.runs.live_run("3") is None and chat.store.running.get("3") is False
    assert not os.path.isdir(chat.gates["prepared"][0])


def test_answer_glossary_is_refused_while_the_gate_glossary_is_refined(chat, monkeypatch):
    hold = threading.Event()
    _fake_refinement(monkeypatch, chat.kinds, hold)
    chat.kinds.release.set()
    glossary = chat.tmp / "Book_glossary.csv"
    glossary.write_text("type,raw_name,translated_name\n", encoding="utf-8")

    from glossarion_mobile.ui.chat import run_controller

    original = run_controller.job_params

    def with_gate(**kwargs):
        params = original(**kwargs)
        params["ask"] = True
        params["glossary"] = str(glossary)
        return params

    monkeypatch.setattr(run_controller, "job_params", with_gate)
    run = asyncio.run(_send(chat.runs, "1", _record(book(chat.tmp))))
    assert wait_for(lambda: run.awaiting_glossary)
    worker = threading.Thread(target=lambda: chat.service.refine_at_gate(run.job_id, str(glossary)))
    worker.start()
    assert wait_for(lambda: run.refining)
    assert run.awaiting_glossary and not run.gate_open and chat.runs.caption("1") == run_controller.STATUS["refining"]
    assert chat.runs.answer_glossary("1", True) is False and run.awaiting_glossary
    # the snapshot is a moment late: JobService itself refuses, and the run learns it is refining
    run.refining = False
    assert chat.runs.answer_glossary("1", True) is False and run.refining and run.question
    assert not any(e.startswith("translate:") for e in chat.kinds.events)
    hold.set()
    worker.join(TIMEOUT)
    assert wait_for(lambda: not run.refining)
    assert chat.runs.answer_glossary("1", True)
    assert chat.service.wait_idle(TIMEOUT)
    events = chat.kinds.events
    assert events.index("refine:end") < events.index(f"translate:{run.job_id}")


def test_qa_card_run_translation_reuses_the_books_job_and_says_so(isolated):
    """The QA card's "Run translation" (``ChatFeature.run_translation_for``) twice, or while the Library runs
    the book: one job, and the second tap says where it runs with the way to it."""
    from glossarion_mobile.ui.chat.integration import ChatFeature
    from glossarion_mobile.ui.chat.job_binding import JobsAdapter

    service, kinds = make_service(isolated)
    raw = book(isolated)
    workspace = isolated / "Output" / "Book"
    workspace.mkdir(parents=True)
    (workspace / "translation_progress.json").write_text("{}", encoding="utf-8")

    class Library:
        def translate_spec(self, books, *, sources=None, config_overrides=None):
            return JobSpec("translate", books[0]["name"], tuple(sources), params={"config_overrides": dict(config_overrides or {})},
                           origin={"type": "library", "label": f"Library · {books[0]['name']}"})

        async def submit(self, spec):
            return service.submit(spec)

    said, went = [], []
    feature = ChatFeature.__new__(ChatFeature)
    feature.dispatcher = None
    feature.jobs = JobsAdapter(service)
    feature.app = types.SimpleNamespace(library=Library(), notify=lambda *args: said.append(args),
                                        navigate_to=lambda name, params=None: went.append((name, params)))
    first = asyncio.run(feature.run_translation_for(str(workspace), raw))
    assert first and said == []
    again = asyncio.run(feature.run_translation_for(str(workspace), raw))
    assert again == first and len(said) == 1
    message, label, action = said[0]
    assert message == already_active_text(service.snapshot(first)) and label == "Open"
    action()
    assert went == [("jobs.detail", {"jid": first})]
    assert len(_live_jobs(service)) == 1
    service.request_stop(first, force=True)
    kinds.release.set()
    assert service.wait_idle(TIMEOUT)
    service.close()


def test_chat_sends_keep_the_apps_submit_path(chat):
    """The chat submits through the app's ``JobsFeature.submit`` (model preflight + the background preparation
    on the user's tap: the notification permission, the battery sheet, iOS BGContinuedProcessing) - never
    straight to ``JobService.submit_job``, which the facade's ``__getattr__`` would also reach - and still
    tells a reused job from its own."""
    from glossarion_mobile.ui.chat.job_binding import JobsAdapter

    service = chat.service
    prepared = []

    class Facade:  # ui.screens.jobs.JobsFeature: submit adds the preparation, the rest is the JobService API
        async def submit(self, spec, *, long_job=True):
            prepared.append(spec.kind)
            return service.submit(spec)

        def __getattr__(self, name):
            return getattr(service, name)

    chat.runs.jobs = JobsAdapter(Facade())
    run = asyncio.run(_send(chat.runs, "1", _record(book(chat.tmp))))
    assert run is not None and prepared == ["direct_text"]

    class Refusing(Facade):  # the preflight refused the run (an excluded model route): it says why itself
        async def submit(self, spec, *, long_job=True):
            prepared.append("refused")
            return None

    chat.runs.jobs = JobsAdapter(Refusing())
    assert asyncio.run(_send(chat.runs, "2", _record(book(chat.tmp, "Other.epub")))) is None
    # no run without a job is left behind: the chat's next Send is not bound to a dead run
    assert chat.runs.live_run("2") is None and chat.store.running.get("2") is False
    assert prepared == ["direct_text", "refused"]


def test_a_resume_the_preflight_refuses_is_told_apart_from_nothing_to_resume(chat):
    """The app's submit refuses a resubmitted run (``JobsFeature.submit``: the model preflight notifies "… Choose
    model" and returns None): ``resubmit`` returns None, and ``take_refused`` says the app already told the user
    why (ChatView then does not add a false "Nothing to resume in this chat"); a resubmit with nothing to resume
    leaves no such mark."""
    record = _record(book(chat.tmp))
    run = asyncio.run(_send(chat.runs, "1", record))
    assert run is not None and chat.kinds.started.wait(TIMEOUT)
    chat.service.request_stop(run.job_id)
    assert wait_for(lambda: not run.live)
    adapter = chat.runs.jobs
    original = adapter.submit

    async def preflight_refuses(*args, **kwargs):
        return None

    adapter.submit = preflight_refuses
    try:
        assert asyncio.run(chat.runs.resubmit("1")) is None
    finally:
        adapter.submit = original
    assert chat.runs.take_refused("1") is True
    assert chat.runs.take_refused("1") is False  # told once
    assert chat.runs.live_run("1") is None and chat.store.running.get("1") is False
    # a chat with nothing to resume: None without the mark
    assert asyncio.run(chat.runs.resubmit("9")) is None
    assert chat.runs.take_refused("9") is False
