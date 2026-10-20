"""The inference engines behind the on-device endpoint.

``Engine`` protocol: ``load(path, n_ctx, n_threads, mmproj=None)``, ``unload()``, ``chat(messages, params,
should_stop) -> iterator of text deltas`` (``last_usage`` / ``last_finish`` describe the finished
generation), ``count_prompt(messages) -> tokens``, ``loaded``, ``vision`` (a projector was given).

Vision: a model loaded with its ``mmproj`` projector answers requests with ``image_url`` parts through
llama-cpp-python's ``MTMDChatHandler`` (libmtmd; the model's own chat template, so Qwen3.5 works). The
projector is loaded on the first image only (text-only use costs no extra RAM) and freed with the model.
The service has already scaled the images down (``vision.prepare_messages``).

* ``LlamaEngine``: llama-cpp-python (``ENGINE_VERSION``; on the device the pypi.flet.dev cp313 wheels).
  ``llama_cpp`` is imported only when a model loads, so the app starts without it and the screen says
  the engine is unavailable instead of failing.
* ``FakeEngine``: a deterministic echo with a configurable per-token delay (host tests and the
  on-device offline e2e check).

Qwen3.5 thinks by default; the chat template is rendered with ``enable_thinking=False`` unless the
request asks otherwise (``chat_template_kwargs``), and the server strips a leading think block anyway.

CJK ban (``params["ban_cjk"]``, decided by the service, ``script_ban``): ``LlamaEngine`` passes a logits
processor that sets the model's Han / kana / Hangul token ids to -inf. The ids are computed from the
vocabulary on the first such request after a load (about 0.6 s for Qwen3.5's 248k tokens on a desktop
CPU; logged) and kept until the model is unloaded; a request without ``ban_cjk`` gets no processor.
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

__all__ = ["ENGINE_VERSION", "ContextOverflow", "EngineUnavailable", "FakeEngine", "LlamaEngine", "cpu_variant",
           "default_threads", "engine_status", "system_info", "vision_status"]

#: The llama-cpp-python pin of ``src/mobile/pyproject.toml`` (``[tool.flet.android/ios].dependencies``).
ENGINE_VERSION = "0.3.32"
#: Qwen3.5 non-thinking sampling defaults (used when a request leaves them out).
DEFAULT_SAMPLING = {"top_k": 20, "top_p": 0.8, "min_p": 0.0}
STOP_CHECK_EVERY = 8  # tokens between two "is the client still there?" checks
UNAVAILABLE = "The on-device engine is not available in this build (llama-cpp-python is not installed)"
NO_MTMD = "This build's on-device engine cannot read images (llama-cpp-python without libmtmd)"
_VISION: list = []  # cached vision_status() result


class EngineUnavailable(RuntimeError):
    pass


def engine_status() -> tuple:
    """``(available, reason)`` without importing llama_cpp (that loads the native libraries)."""
    try:
        found = importlib.util.find_spec("llama_cpp") is not None
    except (ImportError, ValueError):
        found = False
    return (True, None) if found else (False, UNAVAILABLE)


def cpu_variant() -> Optional[str]:
    """The CPU backend our llama-cpp-python build registered at import: a runtime-selected variant on
    Android arm64 (``android_armv8.6_1`` = dotprod + fp16 + i8mm ... ``android_armv8.0_1`` = baseline),
    ``builtin`` where libggml links the backend (x86_64, iOS), ``none``; None without that loader."""
    try:
        ext = importlib.import_module("llama_cpp._ctypes_extensions")
    except Exception:
        return None
    value = getattr(ext, "CPU_BACKEND_VARIANT", None)
    return str(value) if value else None


def system_info() -> Optional[str]:
    """``llama_print_system_info()`` (loads all the libraries; the CPU line lists the features the
    registered backend was built with) plus the CPU variant, or None without llama_cpp."""
    try:
        llama_cpp = importlib.import_module("llama_cpp")
    except Exception:
        return None
    try:
        raw = llama_cpp.llama_print_system_info()
        text = raw.decode("utf-8", "replace").strip() if isinstance(raw, bytes) else str(raw).strip()
    except Exception as exc:  # pragma: no cover - device only
        return f"unavailable ({exc})"
    variant = cpu_variant()
    if variant and variant != "builtin":
        text = f"{text} (CPU variant {variant})".strip()
    return text


def vision_status() -> tuple:
    """``(available, reason)``: libmtmd loads (``llama_cpp.mtmd_cpp`` dlopens it at import). Cached."""
    if not _VISION:
        try:
            importlib.import_module("llama_cpp.mtmd_cpp")
            _VISION.append((True, None))
        except Exception as exc:
            log.info("on-device vision unavailable: %s", exc)
            _VISION.append((False, f"{NO_MTMD}: {exc}"))
    return _VISION[0]


def image_count(messages: Sequence[dict]) -> int:
    from glossarion_mobile.services.local_llm.vision import image_count as count

    return count(messages)


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
        self.mmproj: Optional[str] = None
        self.n_ctx = 0
        self.n_threads = 0
        self.last_usage: dict = {}
        self.last_finish = "stop"

    @property
    def loaded(self) -> bool:
        return self.path is not None

    @property
    def vision(self) -> bool:
        """The loaded model has a vision projector (in memory once ``vision_loaded``)."""
        return self.path is not None and bool(self.mmproj)

    @property
    def vision_loaded(self) -> bool:
        return False

    def _image_tokens(self, messages: Sequence[dict]) -> int:
        from glossarion_mobile.services.local_llm.vision import IMAGE_TOKENS

        return image_count(messages) * IMAGE_TOKENS


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
        self.projector_loads = 0
        self._projector = False

    def load(self, path: str, n_ctx: int, n_threads: int, mmproj: Optional[str] = None) -> None:
        if self.load_delay:
            time.sleep(self.load_delay)
        self.path, self.n_ctx, self.n_threads = path, int(n_ctx), int(n_threads)
        self.mmproj = mmproj
        self._projector = False
        self.loads.append((path, int(n_ctx), int(n_threads)))

    def unload(self) -> None:
        if self.path is not None:
            self.unloads += 1
        self.path = None
        self.mmproj = None
        self._projector = False

    @property
    def vision_loaded(self) -> bool:
        return self._projector

    def count_prompt(self, messages: Sequence[dict]) -> int:
        return sum(len(_text_of(m.get("content")).split()) + 4 for m in messages) + self._image_tokens(messages)

    def chat(self, messages: Sequence[dict], params: dict, should_stop: Callable[[], bool]) -> Iterator[str]:
        self.calls.append({"messages": list(messages), "params": dict(params)})
        self.generated = 0
        images = image_count(messages)
        if images:
            if not self.mmproj:
                raise EngineUnavailable("This on-device model has no vision support")
            if not self._projector:
                self._projector = True
                self.projector_loads += 1
        last = next((m for m in reversed(list(messages)) if m.get("role") == "user"), {})
        text = self.reply(list(messages)) if self.reply else f"Echo: {_text_of(last.get('content'))}"
        if images and not self.reply:
            text += f" [{images} image{'s' if images != 1 else ''}]"
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
        self.vision_handler: Any = None
        self._ban: Any = None  # the loaded model's script_ban.BanLogitsProcessor (computed on first use)
        self._lock = threading.RLock()

    def _module(self) -> Any:
        try:
            return importlib.import_module("llama_cpp")
        except Exception as exc:
            raise EngineUnavailable(f"{UNAVAILABLE}: {exc}") from exc

    def load(self, path: str, n_ctx: int, n_threads: int, mmproj: Optional[str] = None) -> None:
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
            self.mmproj = str(mmproj) if mmproj else None

    @property
    def vision_loaded(self) -> bool:
        return self.vision_handler is not None and getattr(self.vision_handler, "mtmd_ctx", None) is not None

    def _vision(self) -> Any:
        """The model's MTMD chat handler (created on the first image; the projector loads in its first call)."""
        if self.vision_handler is None:
            if not self.mmproj:
                raise EngineUnavailable("This on-device model has no vision support")
            available, reason = vision_status()
            if not available:
                raise EngineUnavailable(reason or NO_MTMD)
            self.vision_handler = _make_vision_handler(self.mmproj)
        return self.vision_handler

    def unload(self) -> None:
        with self._lock:
            llm, self.llm = self.llm, None
            handler, self.vision_handler = self.vision_handler, None
            self.formatter = None
            self._ban = None
            self.path = None
            self.mmproj = None
            if handler is not None:
                try:
                    handler._exit_stack.close()  # mtmd_free: the projector, before its model goes
                except Exception:
                    log.debug("freeing the vision projector failed", exc_info=True)
            if llm is not None:
                try:
                    llm.close()
                except Exception:
                    log.debug("closing the model failed", exc_info=True)
                del llm

    def cjk_ban(self) -> Any:
        """The loaded model's CJK-ban logits processor (``script_ban``), computed from its vocabulary on
        the first call after a load and kept until the model is unloaded."""
        llm = self.llm
        if llm is None:
            raise EngineUnavailable("No model is loaded")
        with self._lock:
            if self._ban is None:
                from glossarion_mobile.services.local_llm import script_ban

                started = time.monotonic()
                n_vocab = int(llm.n_vocab())
                pieces = (llm.detokenize([token], special=False) for token in range(n_vocab))
                ids = script_ban.banned_token_ids(pieces)
                self._ban = script_ban.BanLogitsProcessor(ids, n_vocab)
                log.info("on-device CJK ban: %d of %d tokens in %.2f s", len(ids), n_vocab,
                         time.monotonic() - started)
            return self._ban

    def _prompt(self, messages: Sequence[dict]) -> str:
        if self.formatter is not None:
            return self.formatter(messages=list(messages)).prompt
        return "\n".join(f"{m.get('role')}: {_text_of(m.get('content'))}" for m in messages)

    def count_prompt(self, messages: Sequence[dict]) -> int:
        llm = self.llm
        if llm is None:
            raise EngineUnavailable("No model is loaded")
        text_only = [{"role": m.get("role"), "content": _text_of(m.get("content"))} for m in messages]
        prompt = self._prompt(text_only)
        return len(llm.tokenize(prompt.encode("utf-8"), add_bos=False, special=True)) + self._image_tokens(messages)

    def chat(self, messages: Sequence[dict], params: dict, should_stop: Callable[[], bool]) -> Iterator[str]:
        llm = self.llm
        if llm is None:
            raise EngineUnavailable("No model is loaded")
        if self.formatter is not None:
            self.formatter.enable_thinking = bool(params.get("enable_thinking", False))
        images = image_count(messages)
        handler = self._vision() if images else None
        if handler is not None:
            handler.enable_thinking = bool(params.get("enable_thinking", False))
            chat_messages = [_vision_message(m) for m in messages]
        else:
            chat_messages = [{"role": m.get("role"), "content": _text_of(m.get("content"))} for m in messages]
        kwargs: dict = {
            "messages": chat_messages,
            "stream": True,
            "max_tokens": int(params.get("max_tokens") or 0) or None,
            "temperature": float(params.get("temperature", 0.7)),
            "top_p": float(params.get("top_p", DEFAULT_SAMPLING["top_p"])),
            "top_k": int(params.get("top_k", DEFAULT_SAMPLING["top_k"])),
            "min_p": float(params.get("min_p", DEFAULT_SAMPLING["min_p"])),
            "presence_penalty": float(params.get("presence_penalty") or 0.0),
            "frequency_penalty": float(params.get("frequency_penalty") or 0.0),
            "repeat_penalty": float(params.get("repeat_penalty") or 1.0),
            "stop": params.get("stop") or None,
        }
        if params.get("seed") is not None:
            kwargs["seed"] = int(params["seed"])
        if params.get("ban_cjk"):
            try:
                processors = importlib.import_module("llama_cpp").LogitsProcessorList
            except Exception:  # a test's fake model: any list of processors works the same
                processors = list
            kwargs["logits_processor"] = processors([self.cjk_ban()])
        self.last_finish = "stop"
        completion = 0
        # 0.3.32's create_chat_completion takes no stopping_criteria: the stream is checked every
        # STOP_CHECK_EVERY chunks and closing the generator stops llama.cpp's sampling loop.
        if handler is not None:
            # the MTMD handler evaluates the prompt (text and image chunks) in this call and returns the
            # lazy token stream; the text handler is back in place for the next request right after
            text_handler = llm.chat_handler
            llm.chat_handler = handler
            try:
                chunks = llm.create_chat_completion(**kwargs)
            except ValueError as exc:
                if "n_ctx" in str(exc):
                    raise ContextOverflow(str(exc)) from exc
                raise
            finally:
                llm.chat_handler = text_handler
        else:
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


class ContextOverflow(ValueError):
    """The prompt with its images does not fit ``n_ctx`` (libmtmd's own count, after the estimate)."""


def _vision_message(message: dict) -> dict:
    """A message for the MTMD handler: a user turn keeps its text and image_url parts; the others are text."""
    content = message.get("content")
    if message.get("role") == "user" and isinstance(content, list):
        parts = [p for p in content if isinstance(p, dict) and p.get("type") in ("text", "image_url")]
        return {"role": "user", "content": parts}
    return {"role": message.get("role"), "content": _text_of(content)}


def _make_vision_handler(mmproj: str) -> Any:
    """``MTMDChatHandler`` (libmtmd; the GGUF's own chat template) rendering with ``enable_thinking``
    like ``NoThinkFormatter``. CPU only (``use_gpu=False``), quiet."""
    from llama_cpp import llama_chat_format

    class NoThinkVisionHandler(llama_chat_format.MTMDChatHandler):
        enable_thinking = False

        def __call__(self, **kwargs: Any) -> Any:
            kwargs.setdefault("enable_thinking", self.enable_thinking)
            return super().__call__(**kwargs)

    return NoThinkVisionHandler(clip_model_path=str(mmproj), verbose=False, use_gpu=False)


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
