"""The inference engines behind the on-device endpoint.

``Engine`` protocol: ``load(path, n_ctx, n_threads)``, ``unload()``, ``chat(messages, params,
should_stop) -> iterator of text deltas`` (``last_usage`` / ``last_finish`` describe the finished
generation), ``count_prompt(messages) -> tokens``, ``loaded``.

* ``LlamaEngine``: llama-cpp-python (``ENGINE_VERSION``; on the device the pypi.flet.dev cp313 wheels).
  ``llama_cpp`` is imported only when a model loads, so the app starts without it and the screen says
  the engine is unavailable instead of failing.
* ``FakeEngine``: a deterministic echo with a configurable per-token delay (host tests and the
  on-device offline e2e check).

Qwen3.5 thinks by default; the chat template is rendered with ``enable_thinking=False`` unless the
request asks otherwise (``chat_template_kwargs``), and the server strips a leading think block anyway.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import os
import re
import threading
import time
from typing import Any, Callable, Iterator, Optional, Sequence

log = logging.getLogger("glossarion.local_llm")

__all__ = ["ENGINE_VERSION", "EngineUnavailable", "FakeEngine", "LlamaEngine", "default_threads", "engine_status",
           "system_info"]

#: The llama-cpp-python pin of ``src/mobile/pyproject.toml`` (``[tool.flet.android/ios].dependencies``).
ENGINE_VERSION = "0.3.32"
#: Qwen3.5 non-thinking sampling defaults (used when a request leaves them out).
DEFAULT_SAMPLING = {"top_k": 20, "top_p": 0.8, "min_p": 0.0}
STOP_CHECK_EVERY = 8  # tokens between two "is the client still there?" checks
UNAVAILABLE = "The on-device engine is not available in this build (llama-cpp-python is not installed)"


class EngineUnavailable(RuntimeError):
    pass


def engine_status() -> tuple:
    """``(available, reason)`` without importing llama_cpp (that loads the native libraries)."""
    try:
        found = importlib.util.find_spec("llama_cpp") is not None
    except (ImportError, ValueError):
        found = False
    return (True, None) if found else (False, UNAVAILABLE)


def system_info() -> Optional[str]:
    """``llama_print_system_info()`` (loads all four libraries), or None without llama_cpp."""
    try:
        llama_cpp = importlib.import_module("llama_cpp")
    except Exception:
        return None
    try:
        raw = llama_cpp.llama_print_system_info()
        return raw.decode("utf-8", "replace").strip() if isinstance(raw, bytes) else str(raw).strip()
    except Exception as exc:  # pragma: no cover - device only
        return f"unavailable ({exc})"


def default_threads() -> int:
    """``max(2, min(4, cpus - 2))``: 4 on 8- and 6-core phones, 2 on 4-core ones."""
    cpus = 0
    affinity = getattr(os, "sched_getaffinity", None)
    if affinity is not None:
        try:
            cpus = len(affinity(0))
        except OSError:
            cpus = 0
    cpus = cpus or os.cpu_count() or 4
    return max(2, min(4, cpus - 2))


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(p.get("text") or "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    return str(content or "")


class _Base:
    def __init__(self) -> None:
        self.path: Optional[str] = None
        self.n_ctx = 0
        self.n_threads = 0
        self.last_usage: dict = {}
        self.last_finish = "stop"

    @property
    def loaded(self) -> bool:
        return self.path is not None


class FakeEngine(_Base):
    """Deterministic: replies ``Echo: <last user text>`` one word per token (``think=True`` puts a
    ``<think>…</think>`` block first). Records what it saw for the tests."""

    name = "fake"

    def __init__(self, *, delay: float = 0.0, think: bool = False, load_delay: float = 0.0,
                 reply: Optional[Callable[[list], str]] = None) -> None:
        super().__init__()
        self.delay = delay
        self.think = think
        self.load_delay = load_delay
        self.reply = reply
        self.loads: list = []
        self.unloads = 0
        self.generated = 0  # tokens produced since the last chat started
        self.calls: list = []

    def load(self, path: str, n_ctx: int, n_threads: int) -> None:
        if self.load_delay:
            time.sleep(self.load_delay)
        self.path, self.n_ctx, self.n_threads = path, int(n_ctx), int(n_threads)
        self.loads.append((path, int(n_ctx), int(n_threads)))

    def unload(self) -> None:
        if self.path is not None:
            self.unloads += 1
        self.path = None

    def count_prompt(self, messages: Sequence[dict]) -> int:
        return sum(len(_text_of(m.get("content")).split()) + 4 for m in messages)

    def chat(self, messages: Sequence[dict], params: dict, should_stop: Callable[[], bool]) -> Iterator[str]:
        self.calls.append({"messages": list(messages), "params": dict(params)})
        self.generated = 0
        last = next((m for m in reversed(list(messages)) if m.get("role") == "user"), {})
        text = self.reply(list(messages)) if self.reply else f"Echo: {_text_of(last.get('content'))}"
        if self.think:
            text = "<think>\nplanning the answer\n</think>\n\n" + text
        tokens = re.findall(r"\S+\s*|\s+", text)
        limit = int(params.get("max_tokens") or 0) or len(tokens)
        self.last_finish = "stop"
        for index, token in enumerate(tokens):
            if index >= limit:
                self.last_finish = "length"
                break
            if index % STOP_CHECK_EVERY == 0 and should_stop():
                break
            if self.delay:
                time.sleep(self.delay)
            self.generated += 1
            yield token
        prompt = self.count_prompt(messages)
        self.last_usage = {"prompt_tokens": prompt, "completion_tokens": self.generated,
                           "total_tokens": prompt + self.generated}


class LlamaEngine(_Base):
    """llama-cpp-python, one model at a time; the caller (``LocalEngineService``) serialises calls."""

    name = "llama.cpp"

    def __init__(self) -> None:
        super().__init__()
        self.llm: Any = None
        self.formatter: Any = None
        self._lock = threading.RLock()

    def _module(self) -> Any:
        try:
            return importlib.import_module("llama_cpp")
        except Exception as exc:
            raise EngineUnavailable(f"{UNAVAILABLE}: {exc}") from exc

    def load(self, path: str, n_ctx: int, n_threads: int) -> None:
        llama_cpp = self._module()
        with self._lock:
            self.unload()
            llm = llama_cpp.Llama(model_path=str(path), n_ctx=int(n_ctx), n_threads=int(n_threads),
                                  n_threads_batch=int(n_threads), n_batch=512, n_ubatch=512, n_gpu_layers=0,
                                  use_mmap=True, use_mlock=False, verbose=False)
            self.formatter = _make_formatter(llm)
            if self.formatter is not None:
                llm.chat_handler = self.formatter.to_chat_handler()
            self.llm = llm
            self.path, self.n_ctx, self.n_threads = str(path), int(n_ctx), int(n_threads)

    def unload(self) -> None:
        with self._lock:
            llm, self.llm = self.llm, None
            self.formatter = None
            self.path = None
            if llm is not None:
                try:
                    llm.close()
                except Exception:
                    log.debug("closing the model failed", exc_info=True)
                del llm

    def _prompt(self, messages: Sequence[dict]) -> str:
        if self.formatter is not None:
            return self.formatter(messages=list(messages)).prompt
        return "\n".join(f"{m.get('role')}: {_text_of(m.get('content'))}" for m in messages)

    def count_prompt(self, messages: Sequence[dict]) -> int:
        llm = self.llm
        if llm is None:
            raise EngineUnavailable("No model is loaded")
        prompt = self._prompt(messages)
        return len(llm.tokenize(prompt.encode("utf-8"), add_bos=False, special=True))

    def chat(self, messages: Sequence[dict], params: dict, should_stop: Callable[[], bool]) -> Iterator[str]:
        llm = self.llm
        if llm is None:
            raise EngineUnavailable("No model is loaded")
        if self.formatter is not None:
            self.formatter.enable_thinking = bool(params.get("enable_thinking", False))
        kwargs: dict = {
            "messages": [{"role": m.get("role"), "content": _text_of(m.get("content"))} for m in messages],
            "stream": True,
            "max_tokens": int(params.get("max_tokens") or 0) or None,
            "temperature": float(params.get("temperature", 0.7)),
            "top_p": float(params.get("top_p", DEFAULT_SAMPLING["top_p"])),
            "top_k": int(params.get("top_k", DEFAULT_SAMPLING["top_k"])),
            "min_p": float(params.get("min_p", DEFAULT_SAMPLING["min_p"])),
            "presence_penalty": float(params.get("presence_penalty") or 0.0),
            "frequency_penalty": float(params.get("frequency_penalty") or 0.0),
            "stop": params.get("stop") or None,
        }
        if params.get("seed") is not None:
            kwargs["seed"] = int(params["seed"])
        self.last_finish = "stop"
        completion = 0
        # 0.3.32's create_chat_completion takes no stopping_criteria: the stream is checked every
        # STOP_CHECK_EVERY chunks and closing the generator stops llama.cpp's sampling loop.
        chunks = llm.create_chat_completion(**kwargs)
        try:
            for index, chunk in enumerate(chunks):
                choice = (chunk.get("choices") or [{}])[0]
                delta = choice.get("delta") or {}
                text = delta.get("content")
                if text:
                    completion += 1
                    yield text
                if choice.get("finish_reason"):
                    self.last_finish = choice["finish_reason"]
                if index % STOP_CHECK_EVERY == 0 and should_stop():
                    break
        finally:
            close = getattr(chunks, "close", None)
            if close is not None:
                close()
        prompt = 0
        try:
            prompt = self.count_prompt(messages)
        except Exception:
            pass
        self.last_usage = {"prompt_tokens": prompt, "completion_tokens": completion,
                           "total_tokens": prompt + completion}


def _make_formatter(llm: Any) -> Any:
    """``NoThinkFormatter`` over the GGUF's ``tokenizer.chat_template`` (eos/bos from the model's token
    ids, as llama.py builds its own Jinja2ChatFormatter); None when the GGUF has no template."""
    try:
        from llama_cpp import llama_chat_format
    except Exception:  # pragma: no cover
        return None
    template = (getattr(llm, "metadata", None) or {}).get("tokenizer.chat_template")
    if not template:
        return None
    eos_id, bos_id = llm.token_eos(), llm.token_bos()
    eos = llm._model.token_get_text(eos_id) if eos_id != -1 else ""
    bos = llm._model.token_get_text(bos_id) if bos_id != -1 else ""

    class NoThinkFormatter(llama_chat_format.Jinja2ChatFormatter):
        enable_thinking = False

        def __call__(self, **kwargs: Any) -> Any:
            kwargs.setdefault("enable_thinking", self.enable_thinking)
            return super().__call__(**kwargs)

    return NoThinkFormatter(template=template, eos_token=eos, bos_token=bos, stop_token_ids=[eos_id])
