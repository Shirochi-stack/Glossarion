"""Owner bug: "it refuses to switch on device models after a job has been clearly stopped".

A graceful Stop lets the answer being generated run to its end, and the client's stop flags (``GRACEFUL_STOP``,
the immediate-stop flags) stay set until the next run. When the job ended before that answer did, nobody read
it any more, yet a model switch waited for all of it (minutes on a phone: the Use button stayed "Loading…"),
and a switch that found a stopped answer in the parallel extra context gave up with "still stopping" after 10 s.
Now the answer of a job that is over is told to end at its next stop check and the switch follows by itself; a
running job's live answer still refuses. The fakes generate a long answer token by token; nothing native runs.
"""

from __future__ import annotations

import asyncio
import threading
import time
import types

import local_llm_fakes as fakes  # noqa: F401  (sys.path for the app and src)
from glossarion_mobile.services.local_llm import engine as eng
from glossarion_mobile.services.local_llm import service as svc
from glossarion_mobile.services.local_llm.server import LocalLLMError
from test_devfix7_force_stop_switch import HeldPrompt
from test_local_llm_server import make_service

MESSAGES = {"messages": [{"role": "user", "content": "translate this"}]}
LONG = 50000  # tokens: about 100 s at the fakes' pace, so only a stop ends it within a test


def _wait(predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def _long_engine():
    return eng.FakeEngine(delay=0.002, reply=lambda messages: "w " * LONG)


class _Run:
    """A job's request on a thread, as the endpoint's handler runs it (the client never hangs up here)."""

    def __init__(self, service, model="local/a") -> None:
        self.error = None
        self.text = ""
        self.thread = threading.Thread(target=self._go, args=(service, model), daemon=True)
        self.thread.start()

    def _go(self, service, model) -> None:
        try:
            with service.chat(model, MESSAGES, lambda: False) as run:
                for delta in run.deltas:
                    self.text += delta
        except LocalLLMError as exc:
            self.error = exc
        except Exception as exc:  # noqa: BLE001  (an engine failure mid-answer)
            self.error = exc


class _Load:
    def __init__(self, service, stem="b") -> None:
        self.result = None
        self.thread = threading.Thread(target=lambda: setattr(self, "result", service.load_blocking(stem)), daemon=True)
        self.thread.start()

    def finished(self, timeout=5.0):
        self.thread.join(timeout)
        return not self.thread.is_alive()


def _job_on_a(tmp_path, engine=None):
    """A running job on local/a generating a long answer; ``flags`` stands for the client's stop flags."""
    engine = engine or _long_engine()
    job = {"model": "local/a"}
    flags = {"state": None}
    service, _engine = make_service(tmp_path, stems=("a", "b"), engine=engine, stop_state=lambda: flags["state"],
                                    active_job_model=lambda: job["model"])
    run = _Run(service)
    assert _wait(lambda: engine.generated > 0 or getattr(engine, "entered", threading.Event()).is_set())
    return service, engine, run, job, flags


def test_a_running_job_still_blocks_the_switch(tmp_path):
    service, engine, run, job, flags = _job_on_a(tmp_path)
    try:
        assert service.load_blocking("b") == (False, svc.BUSY_FOR_JOB)
        assert not service.orphaned() and not service.load_must_wait()
        assert service.loaded_stem == "a" and engine.unloads == 0
    finally:
        service.stop_generation()
        run.thread.join(10)


def test_graceful_stop_then_the_job_ends_the_switch_goes_ahead(tmp_path):
    """The reproduction: the graceful answer would have run on for ~100 s, holding the switch all along."""
    service, engine, run, job, flags = _job_on_a(tmp_path)
    flags["state"] = "graceful"  # GRACEFUL_STOP=1; it stays set after the job (the next run resets it)
    job["model"] = None  # the job is over for the user (Stopped)
    assert service.orphaned() and service.load_must_wait()
    load = _Load(service)
    assert load.finished(), "the switch still waits for the stopped job's whole answer"
    assert load.result == (True, None) and service.loaded_stem == "b"
    run.thread.join(10)
    assert not run.thread.is_alive() and engine.generated < LONG
    assert not service.gate.busy and service._holder_token is None and service._cut is None


def test_force_stop_then_switch(tmp_path):
    service, engine, run, job, flags = _job_on_a(tmp_path)
    service.stop_generation()  # JobBackend.request_stop(force=True)
    flags["state"] = "force"
    job["model"] = None
    load = _Load(service)
    assert load.finished() and load.result == (True, None) and service.loaded_stem == "b"
    run.thread.join(10)
    assert not run.thread.is_alive() and not service.gate.busy


def test_force_stop_while_native_code_runs_waits_for_it_then_switches(tmp_path):
    engine = HeldPrompt()
    service, engine, run, job, flags = _job_on_a(tmp_path, engine=engine)
    service.stop_generation()
    flags["state"] = "force"
    job["model"] = None
    load = _Load(service)
    assert _wait(lambda: service.gate.waiting() == 1)
    assert load.result is None and engine.unloads == 0  # never freed under the native call
    engine.release_prompt.set()
    assert load.finished() and load.result == (True, None) and service.loaded_stem == "b"
    assert not engine.freed_while_held and engine.generated == 0


def test_an_error_in_the_request_gives_the_engine_back(tmp_path):
    class Failing(eng.FakeEngine):
        def chat(self, messages, params, should_stop):
            yield "w "
            self.generated += 1
            raise RuntimeError("decode failed")

    service, engine, run, job, flags = _job_on_a(tmp_path, engine=Failing())
    run.thread.join(10)
    assert isinstance(run.error, RuntimeError)
    job["model"] = None
    assert not service.gate.busy and service.load_blocking("b") == (True, None)


def test_a_failure_while_releasing_still_gives_the_engine_back(tmp_path, monkeypatch):
    """The release used to free the gate last: a failing step before it (the GPU marker's Prefs write) kept the
    engine taken for good, and every switch waited for an answer that had already ended."""
    service, _engine = make_service(tmp_path, stems=("a", "b"), stop_state=lambda: None)
    run = service.chat("local/a", MESSAGES, lambda: False)
    service._gpu_attempt = "a"

    def broken():
        raise OSError("prefs write failed")

    monkeypatch.setattr(service, "_clear_gpu_attempt", broken)
    try:
        run.close()
    except OSError:
        pass
    assert not service.gate.busy and service._holder_token is None
    monkeypatch.undo()
    assert service.load_blocking("b") == (True, None)


def test_a_job_that_ends_during_the_wait_hands_over_by_itself(tmp_path):
    """Graceful Stop while the job still runs: its answer is wanted, the switch waits; once the job is over the
    answer ends at its next stop check and the model switches without another tap."""
    service, engine, run, job, flags = _job_on_a(tmp_path)
    flags["state"] = "graceful"  # the job is Stopping: it still waits for this answer
    assert service.load_must_wait() and not service.orphaned()
    load = _Load(service)
    assert _wait(lambda: service.gate.waiting() == 1)
    time.sleep(0.1)
    assert load.result is None and service._cut is None  # not cut while the job waits for it
    job["model"] = None  # the job ended
    assert load.finished() and load.result == (True, None) and service.loaded_stem == "b"
    run.thread.join(10)
    assert engine.generated < LONG


def test_an_answer_without_a_job_is_not_cut(tmp_path):
    engine = _long_engine()
    flags = {"state": None}
    service, _engine = make_service(tmp_path, stems=("a", "b"), engine=engine, stop_state=lambda: flags["state"],
                                    active_job_model=lambda: None)
    run = _Run(service)  # the endpoint test, no job behind it
    assert _wait(lambda: engine.generated > 0)
    flags["state"] = "graceful"  # even under a Stop's flags
    try:
        assert service.stopping() and service.load_must_wait()
        assert not service.orphaned() and not service.end_orphan()
    finally:
        service.stop_generation()
        run.thread.join(10)


def test_a_chat_during_a_cloud_job_is_not_cut_when_that_job_ends_or_stops(tmp_path):
    """Review: "a job ran when the answer arrived" also held for a cloud job, so ending or stopping that job made
    the user's own on-device chat orphaned and a model pick cut it short."""
    for ending in ("ended", "stopped"):
        engine = _long_engine()
        job = {"model": "openai/gpt-x"}
        flags = {"state": None}
        service, _engine = make_service(tmp_path / ending, stems=("a", "b"), engine=engine,
                                        stop_state=lambda: flags["state"], active_job_model=lambda: job["model"])
        run = _Run(service)  # the user's chat on local/a during the cloud job
        assert _wait(lambda: engine.generated > 0)
        if ending == "ended":
            job["model"] = None
        else:
            flags["state"] = "graceful"
        try:
            assert not service.orphaned()
            assert service.selected_model_changed("local/b") == "deferred"
            time.sleep(0.1)
            assert run.thread.is_alive() and service._cut is None, f"cloud job {ending}: the chat was cut"
        finally:
            service.stop_generation()
            run.thread.join(10)


def test_a_stopped_job_followed_by_a_cloud_job_still_hands_over(tmp_path):
    """Review: the next run clears the stop flags and a cloud job makes a job active again; the stopped
    on-device job's answer must still end for the switch instead of running out first."""
    service, engine, run, job, flags = _job_on_a(tmp_path)
    flags["state"] = "graceful"
    job["model"] = "openai/gpt-x"  # the stopped job ended; the next queued job runs a cloud model
    flags["state"] = None  # its reset_for_new_run cleared the stop flags
    assert service.orphaned() and service.load_must_wait()
    load = _Load(service)
    assert load.finished() and load.result == (True, None) and service.loaded_stem == "b"
    run.thread.join(10)
    assert engine.generated < LONG


def test_selecting_another_model_ends_the_stopped_answer_and_unloads_after_it(tmp_path):
    service, engine, run, job, flags = _job_on_a(tmp_path)
    flags["state"] = "graceful"
    job["model"] = None
    assert service.selected_model_changed("local/b") == "deferred"
    run.thread.join(5)
    assert not run.thread.is_alive() and engine.generated < LONG
    assert _wait(lambda: service.loaded_stem is None)


def test_parallel_extra_context_stopped_answer_is_waited_for_not_refused(tmp_path, monkeypatch):
    """It used to wait FORCE_UNLOAD_WAIT, then refuse with "The on-device model is still stopping"."""
    monkeypatch.setattr(svc, "FORCE_UNLOAD_WAIT", 0.05)
    held = HeldPrompt()
    made = []
    service, _engine = make_service(tmp_path, stems=("a", "b"), stop_state=lambda: None)

    def factory():
        engine = eng.FakeEngine() if not made else held
        made.append(engine)
        return engine

    service._engine_factory = factory
    service.save_parallel_requests(True)
    first = service.chat("local/a", MESSAGES, lambda: False)  # the main context
    second = _Run(service)  # the extra context, held in "native code"
    assert held.entered.wait(10)
    first.close()
    service.stop_generation()  # an immediate Stop
    load = _Load(service)
    assert _wait(lambda: service._parallel.secondary.gate.waiting() == 1)
    assert load.result is None
    held.release_prompt.set()
    assert load.finished() and load.result == (True, None)
    second.thread.join(10)
    assert service.loaded_stem == "b" and not held.loaded and not held.freed_while_held


def test_use_after_a_stopped_job_says_it_waits_and_selects_the_model(tmp_path):
    from glossarion_mobile.ui.screens.local_models_panel import LocalModelsPanel

    service, engine, run, job, flags = _job_on_a(tmp_path)
    flags["state"] = "graceful"
    job["model"] = None
    notes, used = [], []
    panel = types.SimpleNamespace(service=service, context_changed=lambda stem: None, say=notes.append,
                                  add_model=None, use_model=used.append, render=lambda: None)

    async def go():
        return await asyncio.wait_for(LocalModelsPanel.activate(panel, "b"), 5)

    assert asyncio.run(go()) is True
    run.thread.join(10)
    assert notes == ["Waiting for the current answer to end, then switching the model…"]
    assert used == ["local/b"] and service.loaded_stem == "b"
