"""Owner issue: after a double-tap force stop the model could not be switched ("still processing").

A force stop while the on-device engine evaluates a prompt (or loads a model) found no stop check: native
code ran on holding the generation gate and the switch was refused at once. These tests hold a generation
inside the engine with an event, as a native call would be, then stop and switch. No sleeps decide anything.
"""

from __future__ import annotations

import asyncio
import sys
import threading
import time
import types

import local_llm_fakes as fakes  # noqa: F401  (sys.path for the app and src)
from glossarion_mobile.services.jobs import JobBackend
from glossarion_mobile.services.local_llm import engine as eng
from glossarion_mobile.services.local_llm import service as svc
from glossarion_mobile.services.local_llm.server import LocalLLMError
from test_local_llm_server import make_service

BUSY = "The on-device model is busy; try again when the current answer is done"
MESSAGES = {"messages": [{"role": "user", "content": "one two three"}]}


def _wait(predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


class HeldPrompt(eng.FakeEngine):
    """The first chat stays "in native code" until released: no stop check runs while it is held."""

    def __init__(self) -> None:
        super().__init__()
        self.entered = threading.Event()
        self.release_prompt = threading.Event()
        self.freed_while_held = False

    def chat(self, messages, params, should_stop):
        self.calls.append({"messages": list(messages), "params": dict(params), "gpu": self.on_gpu})
        self.entered.set()
        assert self.release_prompt.wait(30)
        for index in range(100000):  # a long answer: only a stop check can end it
            if index % eng.STOP_CHECK_EVERY == 0 and should_stop():
                return
            self.generated += 1
            yield "w "

    def unload(self) -> None:
        if self.entered.is_set() and not self.release_prompt.is_set():
            self.freed_while_held = True
        super().unload()


class _Run:
    """A request on a thread, like the endpoint's handler (the client side never hangs up here)."""

    def __init__(self, service, model="local/a") -> None:
        self.error = None
        self.text = ""
        self.thread = threading.Thread(target=self._go, args=(service, model), daemon=True)
        self.thread.start()

    def _go(self, service, model) -> None:
        try:
            with service.chat(model, MESSAGES, lambda: False) as run:
                self.text = "".join(run.deltas)
        except LocalLLMError as exc:
            self.error = exc


class _Load:
    def __init__(self, service, stem="b") -> None:
        self.result = None
        self.thread = threading.Thread(target=lambda: setattr(self, "result", service.load_blocking(stem)), daemon=True)
        self.thread.start()


def _held(tmp_path, **kwargs):
    engine = HeldPrompt()
    service, _engine = make_service(tmp_path, stems=("a", "b"), engine=engine, stop_state=lambda: None, **kwargs)
    run = _Run(service)
    assert engine.entered.wait(10)
    return service, engine, run


def test_force_stop_during_the_prompt_then_switch_waits_and_loads_by_itself(tmp_path):
    """The reproduction: refused before the fix, now it waits, never frees under the call, and proceeds."""
    job = {"model": "local/a"}
    service, engine, run = _held(tmp_path, active_job_model=lambda: job["model"])
    assert service.load_blocking("b") == (False, BUSY)  # the job's live answer still refuses
    assert not service.load_must_wait()

    service.stop_generation()  # the flags are gone at once here: stop_state stays None
    assert service.stopping() and service.load_must_wait()
    load = _Load(service)
    assert _wait(lambda: service.gate.waiting() == 1)
    assert load.result is None and service.loaded_stem == "a" and engine.unloads == 0

    engine.release_prompt.set()
    load.thread.join(10)
    run.thread.join(10)
    assert load.result == (True, None) and service.loaded_stem == "b"
    assert not engine.freed_while_held and engine.generated == 0
    assert not service.gate.busy and not service.stopping()


def test_the_stop_outlives_the_flags_of_the_next_run(tmp_path):
    """A stop seen only through the client's flags is lost when the next run clears them during the prompt."""
    flags = {"state": None}
    engine = HeldPrompt()
    service, _engine = make_service(tmp_path, stems=("a",), engine=engine, stop_state=lambda: flags["state"])
    run = _Run(service)
    assert engine.entered.wait(10)
    service.stop_generation()
    flags["state"] = "force"
    flags["state"] = None  # reset_for_new_run, while the prompt is still being evaluated
    engine.release_prompt.set()
    run.thread.join(10)
    assert not run.thread.is_alive() and engine.generated == 0 and not service.gate.busy
    # a request of the new run is not touched by the old stop
    with service.chat("local/a", MESSAGES, lambda: False) as second:
        assert next(iter(second.deltas)) == "w "


def test_a_request_queued_at_the_stop_ends_and_the_gate_is_free_for_the_switch(tmp_path):
    service, engine, run = _held(tmp_path)
    queued = _Run(service)
    assert _wait(lambda: service.gate.waiting() == 1)
    service.stop_generation()
    queued.thread.join(10)
    assert queued.error is not None and queued.error.status == 499
    engine.release_prompt.set()
    run.thread.join(10)
    assert service.load_blocking("b") == (True, None) and service.loaded_stem == "b"


def test_without_a_job_the_switch_waits_for_the_answer_and_does_not_cut_it(tmp_path):
    service, engine, run = _held(tmp_path)
    assert service.load_must_wait() and not service.stopping()
    load = _Load(service)
    assert _wait(lambda: service.gate.waiting() == 1)
    service.stop_generation()  # only to end this test's endless answer
    engine.release_prompt.set()
    load.thread.join(10)
    run.thread.join(10)
    assert load.result == (True, None)


def test_a_switch_during_the_client_flags_force_stop_is_not_pinned_by_the_stopping_job(tmp_path):
    flags = {"state": None}
    engine = HeldPrompt()
    job = {"model": "local/a"}
    service, _engine = make_service(tmp_path, stems=("a", "b"), engine=engine, stop_state=lambda: flags["state"],
                                    active_job_model=lambda: job["model"])
    run = _Run(service)
    assert engine.entered.wait(10)
    assert service.selected_model_changed("local/b") is None  # pinned by the live job
    flags["state"] = "force"
    assert service.selected_model_changed("local/b") == "deferred"
    engine.release_prompt.set()
    run.thread.join(10)
    assert service.loaded_stem is None and engine.unloads == 1 and not engine.freed_while_held


def test_a_forced_unload_that_timed_out_leaves_the_generation_stopped(tmp_path, monkeypatch):
    """It used to clear its abort when it gave up, so the generation went on to the end of its answer."""
    monkeypatch.setattr(svc, "FORCE_UNLOAD_WAIT", 0.01)
    service, engine, run = _held(tmp_path)
    assert service.unload(force=True) is False and engine.loaded and engine.unloads == 0
    engine.release_prompt.set()
    run.thread.join(10)
    assert not run.thread.is_alive() and engine.generated == 0 and not service.gate.busy


def test_a_stop_during_the_load_evaluates_no_prompt(tmp_path):
    class HeldLoad(eng.FakeEngine):
        entered = threading.Event()
        go = threading.Event()

        def load(self, *args, **kwargs):
            self.entered.set()
            assert self.go.wait(30)
            super().load(*args, **kwargs)

    engine = HeldLoad()
    service, _engine = make_service(tmp_path, stems=("a", "b"), engine=engine, stop_state=lambda: None)
    run = _Run(service)
    assert engine.entered.wait(10)
    service.stop_generation()
    engine.go.set()
    run.thread.join(10)
    assert run.error is not None and run.error.status == 499 and not engine.calls and not service.gate.busy


# ---- LlamaEngine: the stop reaches a prompt being evaluated --------------------------------------------


class _Context:
    def __init__(self) -> None:
        self.decoded = 0

    def decode(self, batch):
        self.decoded += 1


class _Llama:
    """``Llama.eval``'s shape: one ``_ctx.decode`` per batch, the first chunk after the last one."""

    def __init__(self, batches=4) -> None:
        self._ctx = _Context()
        self.batches = batches

    def create_chat_completion(self, **kwargs):
        for batch in range(self.batches):
            self._ctx.decode(batch)
        yield {"choices": [{"delta": {"content": "first "}, "finish_reason": None}]}
        self._ctx.decode("token")
        yield {"choices": [{"delta": {"content": "second"}, "finish_reason": "stop"}]}

    def tokenize(self, text, add_bos=False, special=True):
        return list(range(len(text.split())))

    def close(self):
        pass


def _engine(llm):
    engine = eng.LlamaEngine()
    engine.llm, engine.path, engine.n_ctx = llm, "x.gguf", 4096
    return engine


def test_llama_prompt_stops_between_decodes_and_the_context_is_restored():
    llm = _Llama()
    answers = iter([False, True])
    stream = _engine(llm).chat([{"role": "user", "content": "hi"}], {"max_tokens": 8}, lambda: next(answers))
    try:
        next(stream)
    except eng.GenerationStopped:
        pass
    else:
        raise AssertionError("the stop did not reach the prompt evaluation")
    assert llm._ctx.decoded == 1  # the second decode never started
    assert "decode" not in vars(llm._ctx)


def test_llama_prompt_checks_end_with_the_first_chunk():
    llm = _Llama()
    checks = []

    def should_stop():
        checks.append(llm._ctx.decoded)
        return False

    out = "".join(_engine(llm).chat([{"role": "user", "content": "hi"}], {"max_tokens": 8}, should_stop))
    assert out == "first second" and llm._ctx.decoded == 5
    assert checks[:4] == [0, 1, 2, 3] and "decode" not in vars(llm._ctx)
    assert llm._ctx.decoded == 5 and len(checks) == 5  # 4 decodes + the first chunk's check; no per-token one


def test_llama_engine_without_the_seam_still_answers():
    class Bare(_Llama):
        def create_chat_completion(self, **kwargs):
            yield {"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]}

    llm = Bare()
    del llm._ctx
    assert "".join(_engine(llm).chat([{"role": "user", "content": "hi"}], {"max_tokens": 8}, lambda: False)) == "ok"


def test_a_stopped_prompt_is_a_499_and_never_a_gpu_failure(tmp_path):
    prefs = fakes.FakePrefs()
    prefs.set(svc.GPU_KEY, True)

    class Stopped(eng.FakeEngine):
        def chat(self, messages, params, should_stop):
            self.calls.append(1)
            raise eng.GenerationStopped("stop")
            yield ""  # pragma: no cover

    engine = Stopped()
    service, _engine = make_service(tmp_path, stems=("a",), engine=engine, prefs=prefs, gpu_backend="OpenCL",
                                    stop_state=lambda: None)
    try:
        service.chat("local/a", MESSAGES, lambda: False)
    except LocalLLMError as exc:
        assert exc.status == 499 and exc.code == "client_closed"
    else:
        raise AssertionError("expected the cancelled answer")
    assert service.gpu_fallback() is None and not service.gate.busy and len(engine.calls) == 1


# ---- the job's Stop reaches the service -----------------------------------------------------------------


def test_only_an_immediate_job_stop_stops_the_generation(tmp_path, monkeypatch):
    service, _engine = make_service(tmp_path)
    svc.set_default_service(service)
    fake = types.SimpleNamespace(apply_force_stop_flags=lambda: None, reset_api_watchdog=lambda **k: None,
                                 request_stop=lambda **k: None, announce_stop=lambda *a: None)
    monkeypatch.setitem(sys.modules, "stop_control", fake)
    try:
        backend = JobBackend()
        args = {"wait_for_chunks": False, "set_stop_requested": lambda: None, "log": lambda *a: None}
        backend.request_stop(graceful=True, force=False, **args)
        assert service._stops == 0
        backend.request_stop(graceful=True, force=True, **args)
        assert service._stops == 1
        backend.request_stop(graceful=False, force=False, **args)
        assert service._stops == 2
    finally:
        svc.set_default_service(None)


# ---- the screen says so in plain words ------------------------------------------------------------------


def test_use_says_it_waits_and_then_selects_the_model(tmp_path):
    from glossarion_mobile.ui.screens.local_models_panel import LocalModelsPanel

    service, engine, run = _held(tmp_path)
    notes, used = [], []

    def say(message):
        notes.append(message)
        service.stop_generation()  # the answer ends only after the screen has said it waits
        engine.release_prompt.set()

    panel = types.SimpleNamespace(service=service, context_changed=lambda stem: None, say=say, add_model=None,
                                  use_model=used.append, render=lambda: None)
    assert asyncio.run(LocalModelsPanel.activate(panel, "b")) is True
    run.thread.join(10)
    assert notes == ["Waiting for the current answer to end, then switching the model…"]
    assert used == ["local/b"] and service.loaded_stem == "b" and not engine.freed_while_held
