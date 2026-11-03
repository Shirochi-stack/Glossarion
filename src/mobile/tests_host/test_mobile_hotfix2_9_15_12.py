"""Mobile v9.15.12 hotfix 2 (owner's device report):

1. An on-device translation stopped while Glossarion was in the background (Home). Leaving the app moves it to
   another cpuset; with fewer cores than llama.cpp threads, ggml's spin-waiting threads all but stop (2 threads
   on 1 core ran 10x slower on the emulator). The engine now follows the CPUs the app may use.
2. The saved on-device Context (16,384) was reset when a model was chosen (see test_local_context).

Run from src/mobile (mobile venv):
    python -m pytest -p no:cacheprovider -W ignore tests_host/test_mobile_hotfix2_9_15_12.py -q
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

MOBILE_DIR = Path(__file__).resolve().parents[1]
if str(MOBILE_DIR / "app") not in sys.path:
    sys.path.insert(0, str(MOBILE_DIR / "app"))


class _Ctx:
    def __init__(self) -> None:
        self.ctx = "ctx-pointer"
        self.decoded = 0

    def decode(self, batch):
        self.decoded += 1
        return 0


class _Llama:
    def __init__(self) -> None:
        self._ctx = _Ctx()
        self.events: list = []

    def tokenize(self, text, add_bos=False, special=True):
        return list(range(len(text.split())))

    def create_chat_completion(self, **kwargs):
        for _ in range(int(kwargs.get("decodes", 0)) or 6):
            self._ctx.decode(None)  # what Llama.eval does per batch / token
            yield {"choices": [{"delta": {"content": "x"}, "finish_reason": None}]}
        yield {"choices": [{"delta": {}, "finish_reason": "stop"}]}


def test_engine_threads_follow_the_cpus_the_app_may_use(monkeypatch):
    from glossarion_mobile.services.local_llm import engine as eng

    calls: list = []
    monkeypatch.setitem(sys.modules, "llama_cpp", types.SimpleNamespace(
        llama_set_n_threads=lambda ctx, n, nb: calls.append((ctx, n, nb)), LogitsProcessorList=list))
    cpus = {"set": {0, 1, 2, 3}}
    monkeypatch.setattr(eng.os, "sched_getaffinity", lambda pid: set(cpus["set"]), raising=False)
    clock = {"t": 100.0}
    monkeypatch.setattr(eng.time, "monotonic", lambda: clock["t"])

    engine = eng.LlamaEngine()
    engine.llm, engine.path, engine.n_ctx, engine.n_threads = _Llama(), "m.gguf", 4096, 4
    llm = engine.llm
    messages = [{"role": "user", "content": "hi"}]
    real_decode = llm._ctx.decode

    def tick(n_cpus):
        cpus["set"] = set(range(n_cpus))
        clock["t"] += eng.CPU_CHECK_SECONDS

    # four cores: nothing changes
    out = engine.chat(messages, {"max_tokens": 8}, lambda: False)
    next(out)
    assert calls == []
    # Home: the app's cpuset shrinks to one core, then two; the threads follow (the decode right after a check)
    tick(1)
    next(out)
    assert calls[-1] == ("ctx-pointer", 1, 1)
    tick(2)
    next(out)
    assert calls[-1] == ("ctx-pointer", 2, 2)
    "".join(out)
    # the request's end restores the loaded count and the decode wrappers are gone
    assert calls[-1] == ("ctx-pointer", 4, 4)
    assert "decode" not in vars(llm._ctx) and llm._ctx.decode == real_decode

    # never more threads than loaded, and one thread or no affinity API: nothing to follow
    calls.clear()
    cpus["set"] = set(range(8))
    "".join(engine.chat(messages, {"max_tokens": 8}, lambda: False))
    assert calls == []
    engine.n_threads = 1
    assert eng._follow_cpus(llm, 1)() is None and "decode" not in vars(llm._ctx)
    monkeypatch.delattr(eng.os, "sched_getaffinity", raising=False)
    assert eng._follow_cpus(llm, 4)() is None and "decode" not in vars(llm._ctx)


def test_a_stop_during_the_prompt_keeps_the_cpu_follower_in_place(monkeypatch):
    """``_stop_between_decodes`` wraps the follower; ending it puts the follower back, not the class method."""
    from glossarion_mobile.services.local_llm import engine as eng

    monkeypatch.setitem(sys.modules, "llama_cpp", types.SimpleNamespace(llama_set_n_threads=lambda *a: None))
    monkeypatch.setattr(eng.os, "sched_getaffinity", lambda pid: {0, 1}, raising=False)
    llm = _Llama()
    cpus_done = eng._follow_cpus(llm, 4)
    follower = vars(llm._ctx)["decode"]
    prompt_done = eng._stop_between_decodes(llm, lambda: False)
    assert vars(llm._ctx)["decode"] is not follower
    prompt_done()
    assert vars(llm._ctx)["decode"] is follower
    cpus_done()
    assert "decode" not in vars(llm._ctx)
