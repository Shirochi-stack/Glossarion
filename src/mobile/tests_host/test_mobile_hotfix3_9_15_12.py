"""Mobile v9.15.12 hotfix 3: the on-device progress line says what happens before the first output token.

Owner: Gemma 4 reads a long prompt for minutes ("SDK stream opened in 147.5s"), and hotfix 2's line showed
"0 tokens, 0.0 tok/s" all that time. Until the first output token the line is the prompt's progress
(``reading the prompt, 1,024 of ~1,536 tokens``), then the tokens written.

Run from src/mobile (mobile venv):
    python -m pytest -p no:cacheprovider -W ignore tests_host/test_mobile_hotfix3_9_15_12.py -q
"""

from __future__ import annotations

import logging
import sys
import threading
import time
import types
from pathlib import Path

MOBILE_DIR = Path(__file__).resolve().parents[1]
if str(MOBILE_DIR / "app") not in sys.path:
    sys.path.insert(0, str(MOBILE_DIR / "app"))

BATCH = 512


class _Ctx:
    ctx = "ctx-pointer"

    def decode(self, batch):
        return 0


class _Llama:
    """Reads a prompt in three 512-token batches (slowly), then writes four tokens."""

    def __init__(self) -> None:
        self._ctx = _Ctx()

    def tokenize(self, text, add_bos=False, special=True):
        return list(range(3 * BATCH))  # count_prompt: ~1,536

    def create_chat_completion(self, **kwargs):
        for _ in range(3):
            time.sleep(0.12)
            self._ctx.decode(types.SimpleNamespace(batch=types.SimpleNamespace(n_tokens=BATCH)))
        for _ in range(4):
            time.sleep(0.03)
            yield {"choices": [{"delta": {"content": "x"}, "finish_reason": None}]}
        time.sleep(0.12)
        yield {"choices": [{"delta": {}, "finish_reason": "stop"}]}


def test_the_progress_line_shows_the_prompt_being_read_then_the_tokens_written(monkeypatch):
    from glossarion_mobile.services.local_llm import engine as eng

    monkeypatch.setitem(sys.modules, "llama_cpp", types.SimpleNamespace(
        llama_set_n_threads=lambda *a: None, llama_n_threads=lambda ctx: 4, LogitsProcessorList=list))
    monkeypatch.setattr(eng.os, "sched_getaffinity", lambda pid: set(range(8)), raising=False)
    monkeypatch.setattr(eng, "HEARTBEAT_SECONDS", 0.05)
    monkeypatch.setattr(eng, "_cpuset", lambda: "/moderate")

    lines: list = []

    class Grab(logging.Handler):
        def emit(self, record):
            lines.append(record.getMessage())

    parent = logging.getLogger("unified_api_client")
    handler, level = Grab(), parent.level
    parent.addHandler(handler)
    parent.setLevel(logging.INFO)
    try:
        engine = eng.LlamaEngine()
        engine.llm, engine.path, engine.n_ctx, engine.n_threads = _Llama(), "m.gguf", 8192, 4
        assert "".join(engine.chat([{"role": "user", "content": "hi"}], {"max_tokens": 8}, lambda: False)) == "xxxx"
    finally:
        parent.removeHandler(handler)
        parent.setLevel(level)
    beats = [line for line in lines if line.startswith("📱 on-device: ") and "threads, CPUs" in line]
    reading = [line for line in beats if "reading the prompt" in line]
    written = [line for line in beats if "tokens written" in line]
    assert reading and written and beats.index(reading[-1]) < beats.index(written[0])
    assert not any("0 tokens" in line or "0.0 tok/s" in line for line in reading)
    assert any("of ~1,536 tokens" in line for line in reading)  # the prompt's progress, not "0 tokens"
    assert all(line.endswith("4 threads, CPUs 0-7, cpuset /moderate") for line in beats)
    assert "4 tokens written" in written[-1]


def test_line_texts():
    from glossarion_mobile.services.local_llm import engine as eng

    beat = eng._Heartbeat(types.SimpleNamespace(llm=None, n_threads=4))
    beat.prompt_total = 5896
    assert beat.line(15, 0).startswith("📱 on-device: reading the prompt (~5,896 tokens), 4 threads")
    beat.read(512)
    beat.read(512)
    beat.started = time.monotonic() - 32  # the average since reading began, not the last interval
    assert "reading the prompt, 1,024 of ~5,896 tokens (32.0 tok/s)" in beat.line(15, 0)
    beat.reading, beat.tokens = False, 274
    assert "274 tokens written, 4.4 tok/s (last 15 s)" in beat.line(15, 66)
    unknown = eng._Heartbeat(types.SimpleNamespace(llm=None, n_threads=2))
    assert "reading the prompt (? tokens)" in unknown.line(15, 0)
    assert eng._batch_tokens(types.SimpleNamespace(batch=types.SimpleNamespace(n_tokens=64))) == 64
    assert eng._batch_tokens(None) == 0 and threading.active_count() >= 1
