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

GPU (optional, ``load(..., gpu=True)``; the service decides): Metal on iOS, OpenCL (Qualcomm Adreno) on
Android arm64 (``gpu_offer``). Our llama-cpp-python build registers the GPU backend only on the first
GPU load (``llama_cpp._ctypes_extensions.register_gpu_backend``), so with the GPU off no GPU code runs.
A GPU load offloads every layer (``n_gpu_layers=-1``, one device), then evaluates a short prompt and
one more token and checks the logits are finite; when the backend has no device, the model does not
load or that first token fails, it raises ``GpuUnavailable`` (the service reloads on the CPU). A CPU
load uses ``split_mode`` NONE with ``main_gpu`` -1: no offload device at all, even after a GPU backend
was registered earlier in the process. ``device`` names what runs the model. The vision projector
always stays on the CPU.

Script ban (``params["ban_scripts"]``, the ``script_ban.SCRIPTS`` names decided by the service): ``LlamaEngine``
passes a logits processor that sets the token ids of those scripts (Han / kana / Hangul, Devanagari,
Thai, Arabic, Cyrillic ...) to -inf. The vocabulary is decoded once, on the first such request after a
load (``script_ban.VocabScripts``; logged), and one processor per set of banned scripts is kept until the
model is unloaded; a request without ``ban_scripts`` gets no processor. ``params["ban_allow_kaomoji"]``
(the opt-in emoticon exemption) unbans the emoticon punctuation and the emoticon characters of the
request's user messages.
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
#: lines a job's Log shows too (job logs listen to ``unified_api_client`` and its children)
job_log = logging.getLogger("unified_api_client.on_device")

__all__ = ["ENGINE_VERSION", "ContextOverflow", "EngineUnavailable", "FakeEngine", "GenerationStopped",
           "GpuUnavailable", "LlamaEngine", "cpu_variant", "default_threads", "engine_status", "gpu_offer", "gpu_state", "system_info", "vision_status"]

#: The llama-cpp-python pin of ``src/mobile/pyproject.toml`` (``[tool.flet.android/ios].dependencies``).
ENGINE_VERSION = "0.3.32"
#: Qwen3.5 non-thinking sampling defaults (used when a request leaves them out).
DEFAULT_SAMPLING = {"top_k": 20, "top_p": 0.8, "min_p": 0.0}
#: llama.cpp's default DRY sequence breakers: a repeat never counts across a line break, a colon, a
#: quote or an asterisk, so a chapter's paragraph tags and dialogue lines stay free
DRY_SEQUENCE_BREAKERS = ("\n", ":", '"', "*")
STOP_CHECK_EVERY = 8  # tokens between two "is the client still there?" checks
#: seconds between two checks of the CPUs Android lets the app use (``_follow_cpus``)
CPU_CHECK_SECONDS = 1.0
#: seconds between two progress lines while an on-device request runs (``_Heartbeat``)
HEARTBEAT_SECONDS = 15.0
UNAVAILABLE = "The on-device engine is not available in this build (llama-cpp-python is not installed)"
NO_MTMD = "This build's on-device engine cannot read images (llama-cpp-python without libmtmd)"
_VISION: list = []  # cached vision_status() result


class EngineUnavailable(RuntimeError):
    pass


class GenerationStopped(RuntimeError):
    """A Stop arrived while the prompt was being evaluated: no token was generated."""


class GpuUnavailable(RuntimeError):
    """The GPU path failed (no GPU backend / device, the model did not load on it, or its first token
    failed); the message says why. The engine holds no model afterwards."""


#: llama.cpp's LLAMA_SPLIT_MODE_NONE: the whole model on ``main_gpu`` (-1: on no device = the CPU).
SPLIT_MODE_NONE = 0
#: what the GPU check evaluates (a batch, then one more token)
GPU_CHECK_PROMPT = ("The lighthouse keeper climbed the stairs every evening, counted each step and wrote the "
                    "weather into a book that nobody had read for forty years.")


def gpu_offer(platform: Optional[str] = None, machine: Optional[str] = None) -> Optional[str]:
    """The GPU backend this app build offers on the device, without loading anything: ``"Metal"`` on iOS,
    ``"OpenCL"`` on Android arm64 (Qualcomm Adreno only; other GPUs are refused when it loads), None
    elsewhere (x86_64 Android, desktop)."""
    name = (platform or os.environ.get("GLOSSARION_PLATFORM") or "").strip().lower()
    if name == "ios":
        # ggml-metal calls an iOS 15 API without an availability check: no GPU option on iOS 13 / 14
        return "Metal" if ios_major() >= 15 else None
    if name == "android":
        import platform as _platform

        arch = (machine if machine is not None else _platform.machine()).strip().lower()
        return "OpenCL" if arch in ("aarch64", "arm64", "arm64-v8a") else None
    return None


def ios_major() -> int:
    """The running iOS major version (``platform.ios_ver``, Python 3.13+); 99 when unknown (off iOS)."""
    import platform as _platform

    try:
        release = _platform.ios_ver().release  # type: ignore[attr-defined]
        return int(str(release).split(".")[0])
    except Exception:
        return 99


def gpu_state() -> dict:
    """``register_gpu_backend()`` of our llama-cpp-python build (registers the GPU backend once per
    process): ``{"backend", "device", "error"}``. A build without it gives an error."""
    try:
        ext = importlib.import_module("llama_cpp._ctypes_extensions")
    except Exception as exc:
        return {"backend": None, "device": None, "error": f"{UNAVAILABLE}: {exc}"}
    register = getattr(ext, "register_gpu_backend", None)
    if register is None:
        return {"backend": None, "device": None, "error": "this engine build has no GPU support"}
    try:
        return dict(register())
    except Exception as exc:  # pragma: no cover - the loader never raises
        return {"backend": None, "device": None, "error": f"{type(exc).__name__}: {exc}"}


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


def cpu_budget(cpus: int) -> int:
    """Threads that fit ``cpus`` allowed CPUs while the app's own threads (the loop, Flutter, the stream)
    keep a core or two: like ``default_threads``, ``cpus - 2`` from four CPUs, ``cpus - 1`` below, at least 1."""
    cpus = int(cpus or 0)
    return max(1, cpus - 2 if cpus >= 4 else cpus - 1)


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
        self.on_gpu = False  # the loaded model runs on the GPU
        self.device: Optional[str] = None  # what runs the loaded model ("GPU: … (OpenCL)", "CPU …")

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
                 reply: Optional[Callable[[list], str]] = None, gpu_device: str = "Fake GPU (Test)",
                 gpu_error: Optional[str] = None, gpu_chat_error: Optional[str] = None) -> None:
        super().__init__()
        self.delay = delay
        self.think = think
        self.load_delay = load_delay
        self.reply = reply
        self.gpu_device = gpu_device
        self.gpu_error = gpu_error  # a GPU load raises GpuUnavailable(gpu_error)
        self.gpu_chat_error = gpu_chat_error  # a chat on the GPU raises RuntimeError(gpu_chat_error)
        self.loads: list = []
        self.gpu_loads: list = []  # the gpu flag of every load attempt
        self.unloads = 0
        self.generated = 0  # tokens produced since the last chat started
        self.calls: list = []
        self.projector_loads = 0
        self._projector = False

    def load(self, path: str, n_ctx: int, n_threads: int, mmproj: Optional[str] = None, gpu: bool = False) -> None:
        if self.load_delay:
            time.sleep(self.load_delay)
        self.gpu_loads.append(bool(gpu))
        if gpu and self.gpu_error:
            self.unload()
            raise GpuUnavailable(self.gpu_error)
        self.path, self.n_ctx, self.n_threads = path, int(n_ctx), int(n_threads)
        self.mmproj = mmproj
        self._projector = False
        self.on_gpu = bool(gpu)
        self.device = f"GPU: {self.gpu_device}" if gpu else "CPU"
        self.loads.append((path, int(n_ctx), int(n_threads)))

    def unload(self) -> None:
        if self.path is not None:
            self.unloads += 1
        self.path = None
        self.mmproj = None
        self._projector = False
        self.on_gpu = False
        self.device = None

    @property
    def vision_loaded(self) -> bool:
        return self._projector

    def count_prompt(self, messages: Sequence[dict]) -> int:
        return sum(len(_text_of(m.get("content")).split()) + 4 for m in messages) + self._image_tokens(messages)

    def chat(self, messages: Sequence[dict], params: dict, should_stop: Callable[[], bool]) -> Iterator[str]:
        self.calls.append({"messages": list(messages), "params": dict(params), "gpu": self.on_gpu})
        self.generated = 0
        if self.on_gpu and self.gpu_chat_error:
            raise RuntimeError(self.gpu_chat_error)
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
        self._scripts: Any = None  # the loaded model's script_ban.VocabScripts (computed on first use)
        self.sliding_window = 0  # the loaded model's SWA window in tokens (0: none)
        self._lock = threading.RLock()

    def _module(self) -> Any:
        try:
            return importlib.import_module("llama_cpp")
        except Exception as exc:
            raise EngineUnavailable(f"{UNAVAILABLE}: {exc}") from exc

    def load(self, path: str, n_ctx: int, n_threads: int, mmproj: Optional[str] = None, gpu: bool = False) -> None:
        llama_cpp = self._module()
        with self._lock:
            self.unload()
            device = None
            if gpu:
                state = gpu_state()
                if not state.get("device"):
                    raise GpuUnavailable(state.get("error") or "no GPU device")
                device = f"GPU: {state['device']} ({state.get('backend')})"
                # every layer on the one GPU device (index 0: the only GPU backend this build registers)
                placement = {"n_gpu_layers": -1, "split_mode": SPLIT_MODE_NONE, "main_gpu": 0}
            else:
                placement = {"n_gpu_layers": 0, "split_mode": SPLIT_MODE_NONE, "main_gpu": -1}
            try:
                # mobile hotfix 1: a sliding-window model (Gemma 3 / 4) keeps only its window in the SWA layers'
                # cache, not the whole context (llama.cpp's default swa_full=true), and the CPU runs flash
                # attention (llama.cpp's "auto" picks it there; the GPU keeps it off as before), so Gemma 4's
                # mixed V sizes are not padded. ``chat`` then starts every request on an empty cache.
                llm = llama_cpp.Llama(model_path=str(path), n_ctx=int(n_ctx), n_threads=int(n_threads),
                                      n_threads_batch=int(n_threads), n_batch=512, n_ubatch=512, use_mmap=True,
                                      use_mlock=False, verbose=False, swa_full=False, flash_attn=not gpu,
                                      **placement)
            except Exception as exc:
                if gpu:
                    raise GpuUnavailable(f"the model did not load on the GPU ({exc})") from exc
                raise
            if gpu:
                try:
                    _gpu_check(llm)
                except Exception as exc:
                    _close(llm)
                    raise GpuUnavailable(f"the first token on the GPU failed ({exc})") from exc
            self.formatter = _make_formatter(llm)
            if self.formatter is not None:
                llm.chat_handler = self.formatter.to_chat_handler()
            self.llm = llm
            self.sliding_window = _sliding_window(llama_cpp, llm)
            self.path, self.n_ctx, self.n_threads = str(path), int(n_ctx), int(n_threads)
            self.mmproj = str(mmproj) if mmproj else None
            self.on_gpu = bool(gpu)
            if device is None:
                variant = cpu_variant()
                device = f"CPU ({variant})" if variant and variant not in ("builtin", "none") else "CPU"
            self.device = device

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
            self._scripts = None
            self.sliding_window = 0
            self.path = None
            self.mmproj = None
            self.on_gpu = False
            self.device = None
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

    def script_ban(self, scripts: Sequence[str], allow_kaomoji: bool = False, source: str = "",
                   whitelist: bool = False) -> Any:
        """The loaded model's logits processor banning ``scripts`` (``script_ban``): the vocabulary is
        decoded on the first call after a load, each set of scripts' processor on its first use; both are
        kept until the model is unloaded. ``allow_kaomoji``: emoticons allowed, with the emoticon
        characters of ``source`` (the request's user text) let through; ``whitelist``: all its characters."""
        llm = self.llm
        if llm is None:
            raise EngineUnavailable("No model is loaded")
        with self._lock:
            started = time.monotonic()
            if self._scripts is None:
                from glossarion_mobile.services.local_llm import script_ban

                n_vocab = int(llm.n_vocab())
                pieces = (llm.detokenize([token], special=False) for token in range(n_vocab))
                self._scripts = script_ban.VocabScripts(pieces)
                log.info("on-device script ban: %d of %d tokens are non-Latin script, scanned in %.2f s",
                         len(self._scripts), n_vocab, time.monotonic() - started)
            processor = self._scripts.processor(scripts, allow_kaomoji=allow_kaomoji, source=source, whitelist=whitelist)
            log.debug("on-device script ban: %d tokens of %s%s (%.2f s)", len(processor), ",".join(scripts),
                      " (emoticons allowed)" if allow_kaomoji else "", time.monotonic() - started)
            return processor

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
            "temperature": float(params.get("temperature", 0.4)),
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
        # create_chat_completion takes no DRY settings: the build's patched _init_sampler reads them from
        # the model (patches/mobile.patch 9.; an unpatched build ignores the attribute)
        llm.dry_sampling = {"multiplier": float(params.get("dry_multiplier") or 0.0),
                            "base": float(params.get("dry_base") or 1.75),
                            "allowed_length": int(params.get("dry_allowed_length") or 2),
                            "penalty_last_n": -1, "seq_breakers": list(DRY_SEQUENCE_BREAKERS)}
        if params.get("ban_scripts"):
            try:
                processors = importlib.import_module("llama_cpp").LogitsProcessorList
            except Exception:  # a test's fake model: any list of processors works the same
                processors = list
            allow = bool(params.get("ban_allow_kaomoji"))
            whitelist = bool(params.get("ban_source_whitelist"))
            source = ("\n".join(_text_of(m.get("content")) for m in messages if m.get("role") == "user")
                      if allow or whitelist else "")
            kwargs["logits_processor"] = processors([self.script_ban(tuple(params["ban_scripts"]), allow, source, whitelist)])
        if self.sliding_window:
            _empty_cache(llm)
        self.last_finish = "stop"
        completion = 0
        # 0.3.32's create_chat_completion takes no stopping_criteria: the stream is checked every
        # STOP_CHECK_EVERY chunks and closing the generator stops llama.cpp's sampling loop; until the first
        # chunk (the prompt, which can take minutes on a phone) a Stop is seen between two decodes.
        cpus_done = _follow_cpus(llm, self.n_threads)
        heartbeat = _Heartbeat(self)
        try:
            heartbeat.prompt_total = self.count_prompt(messages)
        except Exception:
            heartbeat.prompt_total = 0
        heartbeat.start()
        prompt_done = _stop_between_decodes(llm, should_stop, on_batch=heartbeat.read)
        try:
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
            repeat_watch = None
            if params.get("forced_truncation_seconds"):
                from forced_truncation import RepetitionWatch

                repeat_watch = RepetitionWatch("\n".join(_text_of(m.get("content")) for m in messages if m.get("role") == "user"),
                                               params["forced_truncation_seconds"])
            try:
                for index, chunk in enumerate(chunks):
                    prompt_done()
                    heartbeat.reading = False
                    choice = (chunk.get("choices") or [{}])[0]
                    delta = choice.get("delta") or {}
                    text = delta.get("content")
                    if text:
                        completion += 1
                        heartbeat.tokens = completion
                        yield text
                        if repeat_watch is not None and repeat_watch.feed(text):
                            # Forced Truncation: closing the generator stops sampling; the client sees length, or
                            # stop for a glossary request under "Mark glossary forced truncation as complete"
                            self.last_finish = params.get("forced_truncation_finish") or "length"
                            break
                    if choice.get("finish_reason"):
                        self.last_finish = choice["finish_reason"]
                    if index % STOP_CHECK_EVERY == 0 and should_stop():
                        break
            finally:
                close = getattr(chunks, "close", None)
                if close is not None:
                    close()
        finally:
            heartbeat.stop()
            prompt_done()
            cpus_done()
        prompt = 0
        try:
            prompt = self.count_prompt(messages)
        except Exception:
            pass
        self.last_usage = {"prompt_tokens": prompt, "completion_tokens": completion,
                           "total_tokens": prompt + completion}


class ContextOverflow(ValueError):
    """The prompt with its images does not fit ``n_ctx`` (libmtmd's own count, after the estimate)."""


def _batch_tokens(batch: Any) -> int:
    """The tokens of a ``LlamaBatch`` (its native ``llama_batch.n_tokens``), 0 when unknown."""
    try:
        native = getattr(batch, "batch", batch)
        return max(0, int(getattr(native, "n_tokens", 0) or 0))
    except Exception:
        return 0


def _stop_between_decodes(llm: Any, should_stop: Callable[[], bool],
                          on_batch: Optional[Callable[[int], Any]] = None) -> Callable[[], None]:
    """While a prompt is evaluated ``Llama.eval`` runs one native ``llama_decode`` per ``n_batch`` tokens
    with no stop check: raise ``GenerationStopped`` before the next one on a Stop (``Llama.eval`` keeps its
    token count and the context's memory consistent for the next request). The decode running then ends
    first: native code is never interrupted. ``on_batch(tokens)`` follows each decoded batch (the prompt's
    progress). Returns the function that ends the checks (safe to call again). The MTMD handler's image
    chunks do not go through here."""
    context = getattr(llm, "_ctx", None)
    decode = getattr(context, "decode", None)
    if decode is None:
        return lambda: None
    previous = vars(context).get("decode")  # another wrapper (``_follow_cpus``) stays in place

    def checked(batch: Any) -> Any:
        if should_stop():
            raise GenerationStopped("A Stop arrived while the prompt was being evaluated")
        result = decode(batch)
        if on_batch is not None:
            on_batch(_batch_tokens(batch))
        return result

    context.decode = checked

    def done() -> None:
        if vars(context).get("decode") is checked:
            if previous is not None:
                context.decode = previous
            else:
                del context.decode  # the class's own method again

    return done


def _cpuset() -> str:
    """This process's cpuset group (``/top-app``, ``/foreground``, ``/background``…), '' where unreadable."""
    try:
        with open("/proc/self/cpuset", encoding="ascii") as handle:
            return handle.read().strip()
    except OSError:
        return ""


def _cpu_list(cpus: Any) -> str:
    """``{0, 1, 2, 5}`` -> ``0-2,5``."""
    ranges: list = []
    for cpu in sorted(cpus):
        if ranges and cpu == ranges[-1][1] + 1:
            ranges[-1][1] = cpu
        else:
            ranges.append([cpu, cpu])
    return ",".join(str(a) if a == b else f"{a}-{b}" for a, b in ranges)


class _Heartbeat:
    """Mobile hotfix 2: a progress line every ``HEARTBEAT_SECONDS`` while an on-device request runs (tokens,
    speed, threads, the CPUs and cpuset Android gives the app), from its own thread. The owner's phone
    stopped on-device jobs while Glossarion was in the background; a frozen process leaves a gap in these
    timestamps, a throttled one keeps logging at a low speed, a stuck request logs 0 tok/s. Until the first
    output token the line is the prompt's progress (hotfix 3: Gemma 4 reads a long prompt for minutes, which
    read as "0 tokens, 0.0 tok/s")."""

    def __init__(self, engine: "LlamaEngine") -> None:
        self.engine = engine
        self.tokens = 0
        self.reading = True  # no output yet: the prompt is evaluated
        self.prompt_total = 0  # the prompt's estimated tokens (``count_prompt``), 0 when unknown
        self.prompt_read = 0  # prompt tokens decoded so far (``_stop_between_decodes`` batches)
        self.started = time.monotonic()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="gl-llm-heartbeat", daemon=True)

    def start(self) -> "_Heartbeat":
        self.started = time.monotonic()
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()

    def read(self, tokens: int) -> None:
        self.prompt_read += int(tokens or 0)

    def line(self, seconds: float, new_tokens: int) -> str:
        """The progress since the last line: ``new_tokens`` output tokens, or prompt tokens while reading."""
        affinity = getattr(os, "sched_getaffinity", None)
        try:
            cpus = _cpu_list(affinity(0)) if affinity is not None else "?"
        except OSError:
            cpus = "?"
        llm = self.engine.llm
        threads = self.engine.n_threads
        try:
            threads = int(importlib.import_module("llama_cpp").llama_n_threads(llm._ctx.ctx))  # after _follow_cpus
        except Exception:
            pass
        cpuset = _cpuset()
        rate = f"{new_tokens / max(seconds, 0.001):.1f} tok/s"
        total = f"~{self.prompt_total:,}" if self.prompt_total else "?"
        if not self.reading:
            progress = f"{self.tokens:,} tokens written, {rate} (last {seconds:.0f} s)"
        elif self.prompt_read:
            # the average since the start: a 512-token batch can take longer than one line's interval
            average = self.prompt_read / max(time.monotonic() - self.started, 0.001)
            progress = (f"reading the prompt, {min(self.prompt_read, self.prompt_total or self.prompt_read):,} "
                        f"of {total} tokens ({average:.1f} tok/s)")
        else:
            progress = f"reading the prompt ({total} tokens)"
        return (f"📱 on-device: {progress}, {threads} threads, CPUs {cpus}" + (f", cpuset {cpuset}" if cpuset else ""))

    def _count(self) -> int:
        """What the speed counts: prompt tokens while reading, output tokens after."""
        return self.prompt_read if self.reading else self.tokens

    def _run(self) -> None:
        last_at, last_tokens, was_reading = time.monotonic(), 0, True
        while not self._stop.wait(HEARTBEAT_SECONDS):
            if was_reading and not self.reading:  # the first output token: the speed counts tokens from 0
                last_tokens, was_reading = 0, False
            now, tokens = time.monotonic(), self._count()
            try:
                job_log.info("%s", self.line(now - last_at, tokens - last_tokens))
            except Exception:
                log.debug("the on-device progress line failed", exc_info=True)
            last_at, last_tokens = now, tokens


def _follow_cpus(llm: Any, threads: int) -> Callable[[], None]:
    """Keep llama.cpp's threads within the CPUs Android lets the app use (mobile hotfix 2). Leaving the app
    moves it to another cpuset; a phone can give that fewer cores than the threads, and ggml's threads
    spin-wait for each other on every step: with more threads than cores generation all but stops (2
    threads on 1 core ran 10x slower on the emulator). About once a second, before a decode, the thread
    count follows the allowed CPUs (``cpu_budget``, never above the loaded count; ``llama_set_n_threads``)
    and the change is logged; the end of the
    request restores the loaded count. Returns the function that ends it (safe to call again)."""
    context = getattr(llm, "_ctx", None)
    decode = getattr(context, "decode", None)
    affinity = getattr(os, "sched_getaffinity", None)
    try:
        setter = importlib.import_module("llama_cpp").llama_set_n_threads
    except Exception:
        setter = None
    if decode is None or affinity is None or setter is None or int(threads or 0) < 2:
        return lambda: None
    threads = int(threads)
    state = {"n": threads, "at": 0.0}

    def apply(count: int, cpus: int) -> None:
        try:
            setter(context.ctx, count, count)
        except Exception:
            log.debug("changing the thread count failed", exc_info=True)
            return
        state["n"] = count
        job_log.info("📱 on-device: %d of %d threads, %d CPU%s available to the app", count, threads, cpus,
                     "" if cpus == 1 else "s")

    def fitted(batch: Any) -> Any:
        now = time.monotonic()
        if now - state["at"] >= CPU_CHECK_SECONDS:
            state["at"] = now
            try:
                cpus = len(affinity(0))
            except OSError:
                cpus = threads
            want = min(threads, cpu_budget(cpus))
            if want != state["n"]:
                apply(want, cpus)
        return decode(batch)

    context.decode = fitted

    def done() -> None:
        if vars(context).get("decode") is fitted:
            del context.decode
        if state["n"] != threads:
            try:
                setter(context.ctx, threads, threads)  # the next request starts at the loaded count again
            except Exception:
                log.debug("restoring the thread count failed", exc_info=True)
            state["n"] = threads

    return done


def _sliding_window(llama_cpp: Any, llm: Any) -> int:
    """The model's sliding attention window (``llama_model_n_swa``), 0 for a model without one."""
    try:
        return max(0, int(llama_cpp.llama_model_n_swa(llm._model.model)))
    except Exception:
        log.debug("reading the sliding window failed", exc_info=True)
        return 0


def _empty_cache(llm: Any) -> None:
    """Start the next request on an empty cache. With ``swa_full=False`` the SWA layers have already dropped
    the tokens behind their window, but 0.3.32's prompt-prefix reuse (``Llama.generate``) only checks that
    ``seq_rm`` succeeds, and it always does: a reused prefix would attend to tokens that are gone."""
    try:
        llm.reset()
        llm._ctx.kv_cache_clear()
    except Exception:
        log.debug("clearing the cache failed", exc_info=True)


def _close(llm: Any) -> None:
    try:
        llm.close()
    except Exception:
        log.debug("closing the model failed", exc_info=True)


def _gpu_check(llm: Any) -> None:
    """The GPU's first tokens: evaluate ``GPU_CHECK_PROMPT`` (a batch) and one more token; the last
    logits must be finite and not all equal (a broken kernel gives NaN / inf / zeros). Leaves the
    model reset."""
    import math

    tokens = list(llm.tokenize(GPU_CHECK_PROMPT.encode("utf-8"), add_bos=True, special=False))[:64]
    if not tokens:
        raise RuntimeError("the check prompt has no tokens")
    try:
        llm.reset()
        llm.eval(tokens)
        llm.eval(tokens[-1:])
        n_vocab = int(llm.n_vocab())
        logits = llm._ctx.get_logits()
        values = [float(logits[i]) for i in range(n_vocab)]
    finally:
        llm.reset()
    bad = sum(1 for v in values if not math.isfinite(v))
    if bad:
        raise RuntimeError(f"{bad} of {n_vocab} logits are NaN or infinite")
    if not values or max(values) == min(values):
        raise RuntimeError("the logits are all equal")


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
