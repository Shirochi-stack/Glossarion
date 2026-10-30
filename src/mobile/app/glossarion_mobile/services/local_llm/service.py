"""LocalEngineService: the on-device engine's lifecycle (one per app; Models & keys feature install).

* At start it runs ``LocalLLMServer`` (cheap while idle), publishes ``GLOSSARION_LOCAL_LLM_URL`` for the
  shared ``local/`` route, lists unfinished downloads and refreshes the ``local`` catalog. No model is
  loaded at start.
* **One model loaded at a time.** A request for a model that is not loaded loads it on the server thread
  under the generation gate (state ``loading`` then ``ready``); a request for another model unloads the
  current one first, unless the running job uses it (409, no thrash). Generations run one at a time in
  arrival order (FIFO, at most ``MAX_WAITERS`` waiting, then 503).
* **RAM gate** before every load: ``need = size * 1.15 + n_ctx * 64 KiB + 400 MiB``. Android refuses
  above total RAM; above available RAM it only warns ("confirm": the load goes ahead and the message is
  logged and shown, owner 2026-10-27: Use always loads, the warning is cosmetic); iOS refuses above
  half of the RAM (the jetsam limit of an app without the increased-memory entitlement).
  The first image request of a vision model checks again with its projector added
  (``projector_need``): the projector loads only then.
* **Vision:** a model downloaded with its ``mmproj`` projector takes OpenAI ``image_url`` parts
  (``data:`` URLs); the images are decoded with ``safe_image``, scaled to at most 896 px and re-encoded
  before the engine sees them (``vision.prepare_messages``). Text-only models keep the clear 400.
* **Unload policy:** after ``IDLE_UNLOAD`` seconds without a request or running job; at once when iOS
  hides the app with no job running; after ``BACKGROUND_UNLOAD`` seconds in the background on Android
  unless a job runs (its foreground service keeps the process); when the selected model moves off the
  loaded one (``selected_model_changed``; after the current answer, never under a job using it);
  ``detach`` (Android destroys the activity, the process and its job may go on) unloads without a job
  but keeps the endpoint up: a job still running would otherwise get "Connection error" on every retry.
  The endpoint's ``base_url`` never changes while the process lives (``LocalLLMServer`` rebinds its port).
* **Stop:** a queued request gives up on the client closing its connection or any job Stop
  (``process_stop_state``: graceful or immediate); the answer being generated stops within a few tokens
  on an immediate Stop, a disconnect or a forced unload, and runs to its end on a graceful one.
* **Adaptive script ban** ("Block other scripts in translations", Prefs ``BLOCK_CJK_KEY``, on by default):
  a request that writes target-language text bans the tokens of every script the target does not use
  (English: Han / kana / Hangul, Devanagari, Thai, Arabic, Cyrillic ...; a CJK target keeps all three CJK
  scripts); Extract Glossary and a single-pass auto glossary's translations ban them too, except the tokens
  of their source text's characters (``params["ban_source_whitelist"]``, owner 2026-10-30; an image has no
  text and is not banned); other glossary requests keep the default logits. The
  request kind and target come from the body fields ``unified_api_client`` adds for this endpoint only
  (``glossarion_task``, ``target_language``, ``glossarion_single_pass``); ``script_ban.banned_scripts``
  decides and the engine gets ``params["ban_scripts"]``. The opt-in "Allow emoticons and symbols from
  other scripts (kaomoji)" (config ``script_ban_allow_kaomoji``, shared with the API models' ban; shown
  under the switch too) arrives as ``glossarion_allow_kaomoji`` and becomes ``params["ban_allow_kaomoji"]``.
* **GPU (experimental, off by default;** Prefs ``GPU_KEY``, Model manager › On-device › engine card): offered
  where the build has a GPU backend (``engine.gpu_offer``: Metal on iOS, OpenCL / Adreno on Android
  arm64). With it on, a load first tries the GPU (``engine.load(..., gpu=True)``); when that fails
  (``GpuUnavailable``: no backend / device, the model or its first token failed) or a GPU answer fails,
  the reason is logged and kept (Prefs ``GPU_FALLBACK_KEY``), the model loads on the CPU, and that model
  stays on the CPU until the setting or the model changes; the engine card says "GPU unavailable: …".
  A GPU load leaves a marker (``GPU_ATTEMPT_KEY``) until the model's first answer ends: when the app
  died in between (a driver crash aborts the process), the next start records the fallback instead of
  crashing again. The RAM gate adds ``gpu_extra`` (GPU memory is shared with the CPU on phones).
"""

from __future__ import annotations

import asyncio
import collections
import dataclasses
import logging
import os
import sys
import threading
import time
from typing import Any, Callable, Optional

from glossarion_mobile.services.local_llm import downloads as dl
from glossarion_mobile.services.local_llm import engine as eng
from glossarion_mobile.services.local_llm import hf, presets, script_ban, stem_of
from glossarion_mobile.services.local_llm import vision as vis
from glossarion_mobile.services.local_llm.server import ChatRun, LocalLLMError, LocalLLMServer, wants_thinking

log = logging.getLogger("glossarion.local_llm")

__all__ = ["EngineSnapshot", "LocalEngineService", "default_service", "model_block", "projector_need", "ram_need",
           "set_default_service"]

URL_ENV = "GLOSSARION_LOCAL_LLM_URL"
#: each on-device model's context window while a job runs (``local/<stem>=<tokens>;...``), read by the
#: shared ``unified_api_client.model_context_window`` to size glossary chunks (jobs.local_context_scope)
CONTEXT_ENV = "GLOSSARION_LOCAL_LLM_CONTEXT"
SETTINGS_KEY = "local_llm_settings"
#: Prefs key of the on-device sampling settings (Model manager > On-device > engine card). They replace
#: the request's top_p / top_k / min_p / penalties for every on-device request, so the global
#: anti-duplicate settings keep applying to cloud models only; the temperature stays the request's.
SAMPLING_KEY = "local_llm_sampling"
#: Qwen's recommended non-thinking sampling for its small models (owner 2026-10-18: no penalty let a 0.8B
#: model loop on "<p><br/>"); presence_penalty 1.5 counters endless repetition. The
#: temperature is the global one (an on-device model sets 0.4: setting_writes.ON_DEVICE_TEMPERATURE).
#: The DRY sampler (owner 2026-10-21: a 2B model looped on backticks through the presence penalty)
#: penalizes continuing a repeated sequence, more the longer it gets; 0.8 / 1.75 / 2 are llama.cpp's
#: usual values and DRY multiplier 0 switches it off (engine.DRY_SEQUENCE_BREAKERS reset a match).
DEFAULT_ONDEVICE_SAMPLING = {"top_p": 0.8, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5,
                             "frequency_penalty": 0.0, "repeat_penalty": 1.0, "dry_multiplier": 0.8,
                             "dry_base": 1.75, "dry_allowed_length": 2}
#: (low, high, type) of each sampling setting
SAMPLING_LIMITS = {"top_p": (0.0, 1.0, float), "top_k": (0, 200, int),
                   "min_p": (0.0, 1.0, float), "presence_penalty": (-2.0, 2.0, float),
                   "frequency_penalty": (-2.0, 2.0, float), "repeat_penalty": (0.5, 2.0, float),
                   "dry_multiplier": (0.0, 5.0, float), "dry_base": (1.0, 4.0, float),
                   "dry_allowed_length": (1, 20, int)}
#: field labels (Model manager) and error messages
SAMPLING_LABELS = {"top_p": "Top P", "top_k": "Top K", "min_p": "Min P",
                   "presence_penalty": "Presence penalty", "frequency_penalty": "Frequency penalty",
                   "repeat_penalty": "Repeat penalty", "dry_multiplier": "DRY multiplier",
                   "dry_base": "DRY base", "dry_allowed_length": "DRY allowed length"}
#: Prefs key of "Block other scripts in translations (adaptive)" (Model manager > On-device > engine card;
#: a bool, on when unset; named from when it banned CJK only, kept so the saved choice stays): see
#: ``script_ban``
BLOCK_CJK_KEY = "local_llm_block_cjk"
#: Prefs key of "Use GPU (experimental)" (a bool, off when unset)
GPU_KEY = "local_llm_gpu"
#: Prefs key of the GPU fallback: ``{"stem", "reason"}`` (that model runs on the CPU until the setting or
#: the model changes)
GPU_FALLBACK_KEY = "local_llm_gpu_fallback"
#: Prefs key of a GPU load whose first answer has not ended yet: ``{"stem"}`` (crash detection)
GPU_ATTEMPT_KEY = "local_llm_gpu_attempt"
CRASH_REASON = "Glossarion closed while the GPU was starting this model"
MAX_WAITERS = 8
IDLE_UNLOAD = 600.0
BACKGROUND_UNLOAD = 60.0
WATCH_INTERVAL = 15.0
MIB = 1024 * 1024
GIB = 1024 * MIB
NOT_DOWNLOADED = "Download it in Model manager › On-device"
NO_VISION = "This on-device model has no vision support"
#: seconds a forced unload / delete waits for a running generation to stop (it checks every 8 tokens)
FORCE_UNLOAD_WAIT = 10.0

_DEFAULT: Optional["LocalEngineService"] = None


def set_default_service(service: Optional["LocalEngineService"]) -> None:
    global _DEFAULT
    _DEFAULT = service


def default_service() -> Optional["LocalEngineService"]:
    return _DEFAULT


def model_block(model: Any) -> Optional[tuple]:
    """``(reason, detail)`` for a ``local/`` model that cannot run now (not downloaded, no engine), else
    None. Only while the app's service runs (elsewhere ``local/`` is a user's own llama-server)."""
    service = _DEFAULT
    text = str(model or "").strip()
    if service is None or not text.lower().startswith("local/"):
        return None
    if service.registry.resolve(text) is None:
        return f"{text} is not downloaded", f"{text} is not on this device yet. {NOT_DOWNLOADED}."
    available, reason = service.engine_status()
    if not available:
        return "On-device engine unavailable", reason or eng.UNAVAILABLE
    return None


class _TokenRedactor(logging.Filter):
    """Replaces the endpoint's path token in a logger's records (the job log shows httpx's lines)."""

    def __init__(self, token: str) -> None:
        super().__init__()
        self.token = token

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            text = record.getMessage()
        except Exception:
            return True
        if self.token and self.token in text:
            record.msg, record.args = text.replace(self.token, "…"), None
        return True


def _redact_token_in(logger_name: str, token: str) -> None:
    logger = logging.getLogger(logger_name)
    if not any(isinstance(f, _TokenRedactor) and f.token == token for f in logger.filters):
        logger.addFilter(_TokenRedactor(token))


def process_stop_state() -> Optional[str]:
    """The app's job Stop as the endpoint sees it: ``"force"`` while the shared client's immediate stop
    flags are set (``unified_api_client.global_stop_flag`` / ``UnifiedClient._global_cancelled``, the
    ones its own ``_is_stop_requested`` checks), ``"graceful"`` while ``GRACEFUL_STOP=1``, else None.
    Only peeks at an already-imported ``unified_api_client`` (never imports it)."""
    client = sys.modules.get("unified_api_client")
    if client is not None:
        try:
            if getattr(client, "global_stop_flag", False) or \
                    getattr(getattr(client, "UnifiedClient", None), "_global_cancelled", False):
                return "force"
        except Exception:
            pass
    if os.environ.get("GRACEFUL_STOP") == "1":
        return "graceful"
    return None


#: Context of a model without its own / a saved one (tokens): smaller models get more. Owner 2026-10-19:
#: halved from 8192 / 4096 — shorter requests (the output limit is half the context) keep small
#: on-device models on task.
DEFAULT_CONTEXT_SMALL = 4096
DEFAULT_CONTEXT_LARGE = 2048  # models of 4 GiB or more


def default_context(size: int) -> int:
    """The default context for a model file of ``size`` bytes."""
    return DEFAULT_CONTEXT_SMALL if int(size or 0) < 4 * GIB else DEFAULT_CONTEXT_LARGE


def ram_need(size: int, n_ctx: int) -> int:
    return int(int(size) * 1.15 + int(n_ctx) * 64 * 1024 + 400 * MIB)


def projector_need(size: int) -> int:
    """What a vision projector adds once it is loaded: its weights plus the image encoder's buffers."""
    return int(int(size) * 1.2 + 256 * MIB) if size else 0


def gpu_extra(size: int) -> int:
    """What a GPU load adds on a phone (shared memory): the driver, the compiled kernels and the GPU's
    own copies / repacked weights where it cannot map the file (OpenCL)."""
    return int(int(size) * 0.10 + 256 * MIB)


def _gb(count: float) -> str:
    return f"{count / 1e9:.1f} GB"


def _platform() -> str:
    return (os.environ.get("GLOSSARION_PLATFORM") or "desktop").strip().lower()


def memory_info() -> tuple:
    """``(total, available)`` bytes (``available`` None when unknown)."""
    if _platform() == "ios":
        try:
            total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except (ValueError, OSError, AttributeError):
            total = 0
        return int(total), None
    try:
        import psutil

        vm = psutil.virtual_memory()
        return int(vm.total), int(vm.available)
    except Exception:
        pass
    try:  # Linux / Android without psutil
        values: dict = {}
        with open("/proc/meminfo", encoding="ascii") as handle:
            for line in handle:
                key, _, rest = line.partition(":")
                values[key] = int(rest.split()[0]) * 1024
        return values.get("MemTotal", 0), values.get("MemAvailable")
    except (OSError, ValueError, IndexError):
        return 0, None


class _Gate:
    """FIFO admission: one holder, at most ``max_waiters`` waiting in arrival order."""

    def __init__(self, max_waiters: int = MAX_WAITERS) -> None:
        self._cond = threading.Condition()
        self._queue: collections.deque = collections.deque()
        self._busy = False
        self.max_waiters = max_waiters

    @property
    def busy(self) -> bool:
        return self._busy

    def waiting(self) -> int:
        return len(self._queue)

    def try_acquire(self) -> bool:
        with self._cond:
            if self._busy or self._queue:
                return False
            self._busy = True
            return True

    def acquire(self, should_stop: Optional[Callable[[], bool]] = None, on_wait: Optional[Callable[[], None]] = None) -> bool:
        with self._cond:
            if not self._busy and not self._queue:
                self._busy = True
                return True
            if len(self._queue) >= self.max_waiters:
                raise LocalLLMError(503, "server_busy", "On-device model busy: too many requests are waiting",
                                    "server_error")
            ticket = object()
            self._queue.append(ticket)
        if on_wait is not None:
            on_wait()
        with self._cond:
            try:
                while self._busy or self._queue[0] is not ticket:
                    self._cond.wait(0.25)
                    if should_stop is not None and should_stop():
                        self._queue.remove(ticket)
                        self._cond.notify_all()
                        return False
                self._queue.popleft()
                self._busy = True
                return True
            except BaseException:
                if ticket in self._queue:
                    self._queue.remove(ticket)
                    self._cond.notify_all()
                raise

    def release(self) -> None:
        with self._cond:
            self._busy = False
            self._cond.notify_all()

    def take_over(self, abort: threading.Event, timeout: float) -> bool:
        """Forced unload / delete: set ``abort`` (the running generation's stop check sees it), wait up
        to ``timeout`` s for the holder to let go and take the gate ahead of the waiters. False when the
        holder did not let go (the caller must then leave the engine alone)."""
        abort.set()
        deadline = time.monotonic() + timeout
        try:
            with self._cond:
                while self._busy:
                    left = deadline - time.monotonic()
                    if left <= 0:
                        return False
                    self._cond.wait(min(left, 0.25))
                self._busy = True
                return True
        finally:
            abort.clear()


@dataclasses.dataclass(frozen=True)
class EngineSnapshot:
    state: str = "idle"  # idle | loading | ready | generating
    model: Optional[str] = None  # the loaded (or loading) stem
    queue: int = 0
    error: Optional[str] = None
    system_info: Optional[str] = None
    available: bool = True
    unavailable_reason: Optional[str] = None
    load_seconds: Optional[float] = None
    running: bool = False  # the endpoint is up
    vision: bool = False  # the loaded model has a vision projector
    device: Optional[str] = None  # what runs the loaded model ("GPU: Adreno (TM) 830 (OpenCL)", "CPU (…)")
    gpu_note: Optional[str] = None  # "GPU unavailable: <reason> — using CPU"


class LocalEngineService:
    def __init__(self, *, prefs: Any = None, registry: Optional[dl.Registry] = None,
                 engine_factory: Optional[Callable[[], Any]] = None, run_io: Optional[Callable[..., Any]] = None,
                 post: Optional[Callable[..., Any]] = None, refresh_catalog: Optional[Callable[[], Any]] = None,
                 active_job_model: Optional[Callable[[], Optional[str]]] = None, platform: Optional[str] = None,
                 clock: Callable[[], float] = time.monotonic, memory: Optional[Callable[[], tuple]] = None,
                 watch: bool = True, downloads: Optional[dl.DownloadManager] = None,
                 stop_state: Callable[[], Optional[str]] = process_stop_state,
                 gpu_backend: Optional[str] = None) -> None:
        self.prefs = prefs
        self.stop_state = stop_state
        self.registry = registry or dl.Registry()
        self.downloads = downloads or dl.DownloadManager(registry=self.registry, run_io=run_io)
        self._engine_factory = engine_factory
        self.engine: Any = None
        self.run_io = run_io
        self.post = post
        self.refresh_catalog = refresh_catalog
        self.active_job_model = active_job_model
        self.platform = (platform or _platform()).lower()
        # the GPU backend offered here ("Metal" / "OpenCL"; "" or None: no GPU option)
        self.gpu_backend = (gpu_backend if gpu_backend is not None else eng.gpu_offer(self.platform)) or None
        self._loaded_gpu = False  # the loaded model was asked for on the GPU (whatever it got)
        self._gpu_attempt: Optional[str] = None  # stem of a GPU load whose first answer has not ended
        self.clock = clock
        self.memory = memory or memory_info
        self.server = LocalLLMServer(self)
        self.gate = _Gate()
        self._state = "idle"
        self._stem: Optional[str] = None
        self._error: Optional[str] = None
        self._system_info: Optional[str] = None
        self._load_seconds: Optional[float] = None
        self.last_used = clock()
        self.background_since: Optional[float] = None
        self.listeners: list = []
        self._lock = threading.RLock()
        self._watch = watch
        self._watcher: Optional[threading.Thread] = None
        self._stop_watch = threading.Event()
        self._abort = threading.Event()  # a forced unload / delete stops the running generation
        self._unload_when_idle = False  # the selection moved off the loaded model while it answered
        self.log: list = []  # (stem, load seconds) of every load (diagnostics)

    # ---- engine ----------------------------------------------------------------------------------

    def engine_status(self) -> tuple:
        if self._engine_factory is not None:
            return True, None
        return eng.engine_status()

    def vision_status(self) -> tuple:
        """``(available, reason)`` of image input (libmtmd in this build's llama-cpp-python)."""
        if self._engine_factory is not None:
            return True, None
        available, reason = self.engine_status()
        return eng.vision_status() if available else (False, reason)

    def is_vision(self, stem: str) -> bool:
        """``stem`` was downloaded with its vision projector."""
        return bool((self.registry.load().get(stem) or {}).get("mmproj"))

    def _engine(self) -> Any:
        if self.engine is None:
            self.engine = self._engine_factory() if self._engine_factory is not None else eng.LlamaEngine()
        return self.engine

    @property
    def loaded_stem(self) -> Optional[str]:
        engine = self.engine
        return self._stem if engine is not None and engine.loaded else None

    def snapshot(self) -> EngineSnapshot:
        available, reason = self.engine_status()
        loaded = self.loaded_stem
        return EngineSnapshot(self._state if (loaded or self._state == "loading") else "idle",
                              self._stem if (loaded or self._state == "loading") else None,
                              self.gate.waiting(), self._error, self._system_info, available, reason,
                              self._load_seconds, self.server.running,
                              bool(loaded and getattr(self.engine, "vision", False)),
                              getattr(self.engine, "device", None) if loaded else None, self.gpu_note())

    def health(self, model: Optional[str] = None) -> dict:
        """``{"state", "model", "queue"}`` (plus ``busy`` and ``load_seconds`` of the last load); with
        ``model`` also ``target``: ``{"stem", "size", "loaded"}`` of that model (``stem`` None when it is
        not downloaded). The shared client reads it before a request to log loading / waiting lines."""
        snap = self.snapshot()
        out = {"state": snap.state, "model": snap.model, "queue": snap.queue, "busy": bool(self.gate.busy),
               "load_seconds": self._load_seconds}
        if model:
            stem = self.registry.resolve(model)
            size = int((self.registry.load().get(stem) or {}).get("size") or 0) if stem else 0
            out["target"] = {"stem": stem, "size": size,
                             "loaded": bool(stem and snap.model == stem and snap.state != "loading")}
        return out

    def subscribe(self, listener: Callable[[EngineSnapshot], Any]) -> Callable[[], None]:
        self.listeners.append(listener)

        def unsubscribe() -> None:
            if listener in self.listeners:
                self.listeners.remove(listener)

        return unsubscribe

    def _notify(self) -> None:
        snap = self.snapshot()
        for listener in list(self.listeners):
            try:
                if self.post is not None:
                    self.post(listener, snap)
                else:
                    listener(snap)
            except Exception:
                log.exception("engine listener failed")

    def _set(self, state: str, stem: Optional[str] = None, error: Optional[str] = None) -> None:
        with self._lock:
            self._state = state
            if stem is not None or state == "idle":
                self._stem = stem
            self._error = error
        self._notify()

    # ---- start / stop ------------------------------------------------------------------------------

    def start(self) -> str:
        self.server.start()
        _redact_token_in("httpx", self.server.token)  # "HTTP Request: POST http://127.0.0.1:<port>/<token>/v1/…"
        os.environ[URL_ENV] = self.server.base_url
        set_default_service(self)
        self._recover_gpu_attempt()
        self.downloads.scan()
        if self._watch and self._watcher is None:
            self._stop_watch.clear()
            self._watcher = threading.Thread(target=self._watch_loop, name="gl-llm-idle", daemon=True)
            self._watcher.start()
        if self.registry.load():  # nothing to list before the first download
            self.request_catalog_refresh()
        return self.server.base_url

    def stop(self) -> None:
        self._stop_watch.set()
        self._watcher = None
        self.unload(force=True)
        url = self.server.base_url if self.server.running else None
        self.server.stop()
        logger = logging.getLogger("httpx")
        for item in [f for f in logger.filters if isinstance(f, _TokenRedactor) and f.token == self.server.token]:
            logger.removeFilter(item)
        if url is not None and os.environ.get(URL_ENV) == url:
            os.environ.pop(URL_ENV, None)
        if _DEFAULT is self:
            set_default_service(None)

    def request_catalog_refresh(self) -> None:
        if self.refresh_catalog is None:
            return
        try:
            if self.post is not None:
                self.post(self.refresh_catalog)
            else:
                self.refresh_catalog()
        except Exception:
            log.debug("catalog refresh failed", exc_info=True)

    # ---- settings ------------------------------------------------------------------------------------

    def settings(self) -> dict:
        raw = None
        try:
            raw = self.prefs.get(SETTINGS_KEY, None) if self.prefs is not None else None
        except Exception:
            raw = None
        raw = raw if isinstance(raw, dict) else {}
        out = {"n_ctx": 0, "n_threads": 0}
        for key, low, high in (("n_ctx", 512, presets.MAX_CTX), ("n_threads", 1, 64)):
            try:
                value = int(raw.get(key) or 0)
            except (TypeError, ValueError):
                value = 0
            out[key] = value if low <= value <= high else 0
        return out

    def save_settings(self, n_ctx: Any = 0, n_threads: Any = 0) -> tuple:
        """(settings, None) or (None, message)."""
        values = {}
        for key, value, low, high in (("n_ctx", n_ctx, 512, presets.MAX_CTX), ("n_threads", n_threads, 1, 64)):
            text = str(value or "").strip()
            if not text or text == "0":
                values[key] = 0
                continue
            try:
                number = int(text)
            except ValueError:
                return None, f"{'Context' if key == 'n_ctx' else 'Threads'} must be a whole number"
            if not low <= number <= high:
                return None, f"{'Context' if key == 'n_ctx' else 'Threads'} must be between {low} and {high:,}"
            values[key] = number
        if self.prefs is not None:
            self.prefs.set(SETTINGS_KEY, values)
        return values, None

    def save_context(self, n_ctx: Any) -> tuple:
        """Save only the Context default (0 or blank: each model's own, ``context_for``); Threads stay."""
        return self.save_settings(n_ctx, self.settings()["n_threads"])

    def sampling(self) -> dict:
        """The on-device sampling settings: the saved ones, Qwen's recommended value for the rest."""
        try:
            raw = self.prefs.get(SAMPLING_KEY, None) if self.prefs is not None else None
        except Exception:
            raw = None
        raw = raw if isinstance(raw, dict) else {}
        out = dict(DEFAULT_ONDEVICE_SAMPLING)
        for key, (low, high, kind) in SAMPLING_LIMITS.items():
            try:
                value = kind(raw[key])
            except (KeyError, TypeError, ValueError):
                continue
            if low <= value <= high:
                out[key] = value
        return out

    def save_sampling(self, values: dict) -> tuple:
        """(sampling, None) or (None, message); a blank value goes back to Qwen's recommendation."""
        out = {}
        for key, (low, high, kind) in SAMPLING_LIMITS.items():
            text = str((values or {}).get(key, "") if (values or {}).get(key) is not None else "").strip()
            if not text:
                out[key] = DEFAULT_ONDEVICE_SAMPLING[key]
                continue
            try:
                number = kind(float(text)) if kind is int else kind(text)
            except ValueError:
                return None, f"{SAMPLING_LABELS[key]} must be a number"
            if kind is int and float(text) != number:
                return None, f"{SAMPLING_LABELS[key]} must be a whole number"
            if not low <= number <= high:
                return None, f"{SAMPLING_LABELS[key]} must be between {low:g} and {high:g}"
            out[key] = number
        if self.prefs is not None:
            self.prefs.set(SAMPLING_KEY, dict(out))
        return out, None

    def block_cjk(self) -> bool:
        """"Block other scripts in translations (adaptive)": on unless switched off."""
        try:
            raw = self.prefs.get(BLOCK_CJK_KEY, None) if self.prefs is not None else None
        except Exception:
            raw = None
        return raw is None or bool(raw)

    def save_block_cjk(self, enabled: Any) -> bool:
        value = bool(enabled)
        if self.prefs is not None:
            self.prefs.set(BLOCK_CJK_KEY, value)
        return value

    # ---- GPU -------------------------------------------------------------------------------------------

    def _pref(self, key: str) -> Any:
        try:
            return self.prefs.get(key, None) if self.prefs is not None else None
        except Exception:
            return None

    def _set_pref(self, key: str, value: Any) -> None:
        if self.prefs is None:
            return
        try:
            self.prefs.set(key, value)
        except Exception:
            log.debug("saving %s failed", key, exc_info=True)

    def use_gpu(self) -> bool:
        """"Use GPU (experimental)": off unless switched on, and only where a GPU backend is offered."""
        return bool(self.gpu_backend) and bool(self._pref(GPU_KEY))

    def gpu_fallback(self) -> Optional[dict]:
        """``{"stem", "reason"}`` of the model kept on the CPU after a GPU failure, else None."""
        raw = self._pref(GPU_FALLBACK_KEY)
        if isinstance(raw, dict) and raw.get("stem"):
            return {"stem": str(raw["stem"]), "reason": str(raw.get("reason") or "unknown reason")}
        return None

    def gpu_note(self) -> Optional[str]:
        """The engine card's GPU line while the GPU is on: why a model runs on the CPU instead."""
        if not self.use_gpu():
            return None
        fallback = self.gpu_fallback()
        if fallback is None:
            return None
        return f"GPU unavailable for {fallback['stem']}: {fallback['reason']} — using CPU"

    def save_gpu(self, enabled: Any) -> bool:
        """Switch the GPU on or off: forgets any GPU fallback and frees an idle loaded model, so the next
        request loads it on the new device (a model that is answering reloads on its next request)."""
        value = bool(enabled) and bool(self.gpu_backend)
        self._set_pref(GPU_KEY, value)
        self._set_pref(GPU_FALLBACK_KEY, None)
        self._clear_gpu_attempt()
        if self.loaded_stem is not None and self._loaded_gpu != value:
            self.unload()
        self._notify()
        return value

    def _gpu_wanted(self, stem: str) -> bool:
        if not self.use_gpu():
            return False
        fallback = self.gpu_fallback()
        return not (fallback and fallback["stem"] == stem)

    def _gpu_failed(self, stem: str, reason: str) -> None:
        """Keep ``stem`` on the CPU (until the setting or the model changes) and say why."""
        reason = " ".join(str(reason or "unknown reason").split())[:300]
        log.warning("on-device GPU unavailable for %s: %s; using the CPU", stem, reason)
        self._set_pref(GPU_FALLBACK_KEY, {"stem": stem, "reason": reason})
        self._clear_gpu_attempt()

    def _mark_gpu_attempt(self, stem: str) -> None:
        self._gpu_attempt = stem
        self._set_pref(GPU_ATTEMPT_KEY, {"stem": stem, "at": int(time.time())})

    def _clear_gpu_attempt(self) -> None:
        if self._gpu_attempt is not None or self._pref(GPU_ATTEMPT_KEY):
            self._gpu_attempt = None
            self._set_pref(GPU_ATTEMPT_KEY, None)

    def _recover_gpu_attempt(self) -> Optional[str]:
        """At start: a GPU load whose first answer never ended means the process died on the GPU (a
        driver abort cannot be caught): keep that model on the CPU. Returns its stem."""
        raw = self._pref(GPU_ATTEMPT_KEY)
        if not isinstance(raw, dict) or not raw.get("stem"):
            return None
        stem = str(raw["stem"])
        self._set_pref(GPU_ATTEMPT_KEY, None)
        if self.use_gpu():
            self._gpu_failed(stem, CRASH_REASON)
        return stem

    def preset_for(self, stem: str) -> Optional[dict]:
        """The preset that brought ``stem`` (same repo and file; else a file-less preset of that repo)."""
        entry = self.registry.load().get(stem) or {}
        repo = str(entry.get("repo") or "").casefold()
        fallback = None
        for row in presets.load_presets(self.prefs):
            if str(row.get("repo") or "").casefold() != repo:
                continue
            if row.get("file") and row.get("file") == entry.get("file"):
                return row
            if not row.get("file") and fallback is None:
                fallback = row
        return fallback

    def threads(self) -> int:
        return self.settings()["n_threads"] or eng.default_threads()

    def context_for(self, stem: str) -> int:
        row = self.preset_for(stem) or {}
        if row.get("n_ctx"):
            return int(row["n_ctx"])
        if self.settings()["n_ctx"]:
            return self.settings()["n_ctx"]
        size = int((self.registry.load().get(stem) or {}).get("size") or 0)
        return default_context(size)

    def output_limit_for(self, stem: str) -> int:
        """The output-token limit of a job on ``local/<stem>``: half its context, so the prompt keeps the
        other half."""
        return max(256, self.context_for(stem) // 2)

    # ---- RAM gate --------------------------------------------------------------------------------

    def ram_check(self, stem: str, n_ctx: Optional[int] = None, *, vision: bool = False, gpu: bool = False) -> tuple:
        """``("ok"|"confirm"|"refuse", message)`` for loading ``stem`` (``vision``: with its projector;
        ``gpu``: on the GPU, ``gpu_extra`` more)."""
        entry = self.registry.load().get(stem) or {}
        size = int(entry.get("size") or 0)
        if not size:
            try:
                size = os.path.getsize(self.registry.file_path(stem))
            except OSError:
                size = 0
        need = ram_need(size, n_ctx or self.context_for(stem))
        if gpu:
            need += gpu_extra(size)
        projector = entry.get("mmproj") if isinstance(entry.get("mmproj"), dict) else {}
        if vision and projector:
            need += projector_need(int(projector.get("size") or 0))
        name = (self.preset_for(stem) or {}).get("name") or stem
        return self.ram_verdict(need, f"{name} with its image support" if vision and projector else name)

    def preset_ram_check(self, preset: dict) -> tuple:
        """``ram_check`` for a preset that may not be downloaded yet (its known size; the Welcome and
        Model manager hints): ``("ok"|"confirm"|"refuse"|"unknown", message)``."""
        size = int(preset.get("size") or 0)
        if not size or not self.memory()[0]:
            return "unknown", ""
        n_ctx = int(preset.get("n_ctx") or 0) or self.settings()["n_ctx"] or default_context(size)
        return self.ram_verdict(ram_need(size, n_ctx), str(preset.get("name") or "This model"))

    def ram_verdict(self, need: int, name: str) -> tuple:
        """The RAM gate for ``need`` bytes (see the module docstring)."""
        total, available = self.memory()
        if not total:
            return "ok", ""
        if self.platform == "ios":
            if need > 0.5 * total:
                return "refuse", (f"This device has {_gb(total)} RAM; iOS stops apps that use more than about half "
                                  f"of it, and {name} needs about {_gb(need)}. Pick a smaller model.")
            return "ok", ""
        if need > total:
            return "refuse", f"This phone has {_gb(total)} RAM; {name} needs about {_gb(need)}. Pick a smaller model."
        if available is not None and need > available and self.platform == "android":
            return "confirm", (f"{name} needs about {_gb(need)} and {_gb(available)} is free. Android may close "
                               "Glossarion while this model runs.")
        return "ok", ""

    # ---- load / unload (callers hold the gate) ---------------------------------------------------------

    def _ensure_loaded(self, stem: str, *, vision: bool = False) -> None:
        engine = self._engine()
        n_ctx = self.context_for(stem)
        fallback = self.gpu_fallback()
        if fallback and fallback["stem"] != stem:
            self._set_pref(GPU_FALLBACK_KEY, None)  # another model: the GPU gets a new try
        gpu = self._gpu_wanted(stem)
        # a changed Context (Model manager or Welcome slider) or GPU setting reloads the model on the next request
        if self._stem == stem and engine.loaded and (not engine.n_ctx or engine.n_ctx == n_ctx) \
                and self._loaded_gpu == gpu:
            if vision and not engine.vision_loaded:  # the projector loads with the first image
                self._gate_ram(stem, engine.n_ctx or self.context_for(stem), vision=True,
                               gpu=bool(getattr(engine, "on_gpu", False)))
            return
        available, reason = self.engine_status()
        if not available:
            raise LocalLLMError(503, "engine_unavailable", reason or eng.UNAVAILABLE, "server_error")
        path = self.registry.file_path(stem)
        if not os.path.isfile(path):
            raise LocalLLMError(404, "model_not_found", f"local/{stem} is not downloaded. {NOT_DOWNLOADED}.")
        self._gate_ram(stem, n_ctx, vision=vision, gpu=gpu)
        if engine.loaded:
            self._unload_locked()
        self._unload_when_idle = False
        self._set("loading", stem)
        size = int((self.registry.load().get(stem) or {}).get("size") or 0)
        log.info("loading on-device model %s (%s, n_ctx %d%s)", stem, _gb(size) if size else "size unknown", n_ctx,
                 f", GPU {self.gpu_backend}" if gpu else "")
        started = time.monotonic()
        mmproj = self.registry.projector_path(stem)
        try:
            if gpu:
                self._mark_gpu_attempt(stem)
                try:
                    engine.load(path, n_ctx, self.threads(), mmproj=mmproj, gpu=True)
                except Exception as exc:  # GpuUnavailable, or anything else the GPU path raised
                    self._gpu_failed(stem, str(exc) or type(exc).__name__)
                    engine.load(path, n_ctx, self.threads(), mmproj=mmproj)
            else:
                engine.load(path, n_ctx, self.threads(), mmproj=mmproj)
        except Exception as exc:
            self._clear_gpu_attempt()
            self._loaded_gpu = False
            self._set("idle", None, f"Could not load {stem}: {exc}")
            log.exception("loading %s failed", stem)
            raise LocalLLMError(500, "load_failed", f"Could not load local/{stem}: {exc}", "server_error") from exc
        self._loaded_gpu = self._gpu_wanted(stem)  # False after a fallback: no reload loop
        self._load_seconds = round(time.monotonic() - started, 2)
        self.log.append((stem, self._load_seconds))
        log.info("on-device model %s loaded in %.1f s (n_ctx %d, %d threads, %s)", stem, self._load_seconds, n_ctx,
                 self.threads(), getattr(engine, "device", None) or "CPU")
        if self._system_info is None and not isinstance(engine, eng.FakeEngine):
            self._system_info = eng.system_info()
        self.last_used = self.clock()
        self._set("ready", stem)

    def _gate_ram(self, stem: str, n_ctx: int, *, vision: bool, gpu: bool = False) -> None:
        """Only "refuse" stops a load; "confirm" (tight on free RAM) is a warning in the log, the load goes
        ahead (owner 2026-10-27; the screens show the same message as a note)."""
        verdict, message = self.ram_check(stem, n_ctx, vision=vision, gpu=gpu)
        if verdict == "refuse":
            raise LocalLLMError(507, "insufficient_memory", message, "server_error")
        if verdict == "confirm":
            log.warning("loading anyway: %s", message)

    def _unload_locked(self) -> None:
        engine = self.engine
        if engine is not None and engine.loaded:
            stem = self._stem
            engine.unload()
            log.info("on-device model %s unloaded", stem)
        self._loaded_gpu = False
        self._clear_gpu_attempt()  # unloaded normally: the GPU did not take the app down
        self._set("idle", None)

    def load_blocking(self, stem: str) -> tuple:
        """Explicit Load (Model manager): ``(True, None)`` or ``(False, message)``; never waits for a
        running generation."""
        if not self.gate.try_acquire():
            return False, "The on-device model is busy; try again when the current answer is done"
        try:
            self._ensure_loaded(stem)
            return True, None
        except LocalLLMError as exc:
            return False, exc.message
        finally:
            self.gate.release()

    async def load(self, stem: str) -> tuple:
        return await self._io(lambda: self.load_blocking(stem))

    def unload(self, *, force: bool = False) -> bool:
        """Unload now (False when a generation holds the engine, unless ``force``: then the running
        generation is stopped first and the model is unloaded once it has let go; never while native
        code still uses it)."""
        if self.engine is None or not self.engine.loaded:
            return False
        if not self.gate.try_acquire():
            if not force or not self.gate.take_over(self._abort, FORCE_UNLOAD_WAIT):
                return False
        try:
            self._unload_locked()
            return True
        finally:
            self.gate.release()

    def selected_model_changed(self, model: Any) -> Optional[str]:
        """The app's selected model changed (``local/A`` -> ``local/B`` or a cloud model): the loaded model
        is freed at once, or right after the answer it is giving (``"deferred"``), unless the new selection
        is that model or a running job uses it. Returns ``"unloaded"`` / ``"deferred"`` / None."""
        loaded = self.loaded_stem
        text = str(model or "").strip()
        stem = self.registry.resolve(text) if text.lower().startswith("local/") else None
        if loaded is None or (stem and stem.casefold() == loaded.casefold()):
            self._unload_when_idle = False
            return None
        pinned = self._pinned_stem()
        if pinned and pinned.casefold() == loaded.casefold():
            return None
        if self.unload():
            log.info("on-device model %s unloaded: the selected model is now %s", loaded, text or "none")
            return "unloaded"
        self._unload_when_idle = True
        return "deferred"

    def _deferred_unload(self) -> None:
        """``selected_model_changed`` while the model answered: unload once nothing waits for it."""
        if self.gate.busy or self.gate.waiting():
            return
        pinned = self._pinned_stem()
        loaded = self.loaded_stem
        if loaded is None or (pinned and pinned.casefold() == loaded.casefold()):
            self._unload_when_idle = False
            return
        if self.unload():
            self._unload_when_idle = False
            log.info("on-device model %s unloaded after its answer: another model is selected", loaded)

    async def _io(self, fn: Callable[[], Any]) -> Any:
        if self.run_io is not None:
            return await self.run_io(fn)
        return await asyncio.to_thread(fn)

    # ---- the endpoint backend ------------------------------------------------------------------------

    def list_models(self) -> list:
        loaded = self.loaded_stem
        out = []
        for stem, entry in sorted(self.registry.load().items(), key=lambda kv: kv[0].casefold()):
            out.append({"id": stem, "created": int(entry.get("downloaded_at") or 0), "size": int(entry.get("size") or 0),
                        "repo": entry.get("repo"), "file": entry.get("file"), "loaded": stem == loaded,
                        "vision": bool(entry.get("mmproj"))})
        return out

    def _pinned_stem(self) -> Optional[str]:
        if self.active_job_model is None:
            return None
        try:
            model = self.active_job_model()
        except Exception:
            return None
        if not model or not str(model).strip().lower().startswith("local/"):
            return None
        return self.registry.resolve(model) or stem_of(model)

    def job_active(self) -> bool:
        if self.active_job_model is None:
            return False
        try:
            return self.active_job_model() is not None
        except Exception:
            return False

    def chat(self, model: str, body: dict, should_stop: Callable[[], bool]) -> ChatRun:
        stem = self.registry.resolve(model)
        if stem is None:
            shown = model if str(model).lower().startswith("local/") else f"local/{model}"
            raise LocalLLMError(404, "model_not_found", f"{shown} is not downloaded. {NOT_DOWNLOADED}.")
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise LocalLLMError(400, "invalid_request", "messages must be a non-empty list")
        images = vis.image_count(messages)
        problem = vis.check_parts(messages)
        if images and not self.is_vision(stem):
            raise LocalLLMError(400, "unsupported_content", NO_VISION)
        if problem:
            raise LocalLLMError(400, "unsupported_content", problem)
        if images:
            available, reason = self.vision_status()
            if not available:
                raise LocalLLMError(400, "unsupported_content", reason or eng.NO_MTMD)
            try:
                messages = vis.prepare_messages(messages)  # before the gate: decoding never holds the engine
            except vis.VisionInputError as exc:
                raise LocalLLMError(400, "invalid_image", str(exc)) from exc
        pinned = self._pinned_stem()
        if pinned and pinned.casefold() != stem.casefold():
            raise LocalLLMError(409, "engine_busy", f"The on-device engine is busy with {pinned} for a running job")
        stop_state = self.stop_state

        def stop_waiting() -> bool:
            # a queued request gives up on any Stop (graceful lets only the answer being generated finish)
            return bool(should_stop() or stop_state())

        def stop_generating() -> bool:
            return bool(self._abort.is_set() or should_stop() or stop_state() == "force")

        if stop_waiting():
            raise LocalLLMError(499, "client_closed", "The request was cancelled (Stop)")
        if not self.gate.acquire(stop_waiting, on_wait=self._notify):
            raise LocalLLMError(499, "client_closed", "The request was cancelled while waiting")
        released = [False]

        def release() -> None:
            if released[0]:
                return
            released[0] = True
            self.last_used = self.clock()
            if self._gpu_attempt is not None:
                self._clear_gpu_attempt()  # the GPU's first answer ended without taking the app down
            if self._state == "generating":
                self._set("ready", self._stem)
            self.gate.release()
            self._notify()
            if self._unload_when_idle:
                self._deferred_unload()

        try:
            self._ensure_loaded(stem, vision=bool(images))
            engine = self.engine
            n_ctx = int(engine.n_ctx or self.context_for(stem))
            prompt_tokens = int(engine.count_prompt(messages))
            if prompt_tokens >= n_ctx:
                raise LocalLLMError(400, "context_length_exceeded",
                                    f"This model's maximum context length is {n_ctx} tokens. However, your messages "
                                    f"resulted in {prompt_tokens} tokens. Please reduce the length of the messages.")
            params = self._params(body, n_ctx - prompt_tokens)
            self._set("generating", stem)
            try:
                deltas = _primed(engine.chat(messages, params, stop_generating), n_ctx)
            except (LocalLLMError, KeyboardInterrupt, SystemExit):
                raise
            except Exception as exc:
                if not getattr(engine, "on_gpu", False):
                    raise
                # the GPU failed on this prompt: the same request again on the CPU
                self._gpu_failed(stem, f"the GPU failed while answering ({exc})")
                self._ensure_loaded(stem, vision=bool(images))
                engine = self.engine
                self._set("generating", stem)
                deltas = _primed(engine.chat(messages, params, stop_generating), n_ctx)
        except BaseException:
            release()
            raise
        return ChatRun(stem, deltas, usage=lambda: getattr(engine, "last_usage", {}),
                       finish=lambda: getattr(engine, "last_finish", "stop"), release=release)

    def _params(self, body: dict, room: int) -> dict:
        wanted = body.get("max_completion_tokens") or body.get("max_tokens")
        try:
            wanted = int(wanted) if wanted else 0
        except (TypeError, ValueError):
            wanted = 0
        params: dict = {"max_tokens": max(1, min(wanted, room) if wanted > 0 else room)}
        for key in ("temperature", "seed"):
            if body.get(key) is not None:
                params[key] = body[key]
        # the on-device sampling settings, never the request's (the global anti-duplicate values are for
        # cloud models)
        params.update(self.sampling())
        stop = body.get("stop")
        if isinstance(stop, str):
            stop = [stop]
        if isinstance(stop, list) and stop:
            params["stop"] = [str(s) for s in stop][:8]
        params["enable_thinking"] = wants_thinking(body)
        scripts = script_ban.banned_scripts(body.get("glossarion_task"), body.get("target_language"),
                                            single_pass=body.get("glossarion_single_pass"), enabled=self.block_cjk())
        whitelist = script_ban.source_whitelisted(body.get("glossarion_task"), single_pass=body.get("glossarion_single_pass"))
        if whitelist and any(isinstance(m, dict) and isinstance(m.get("content"), list)
                             and any(isinstance(p, dict) and p.get("type") != "text" for p in m["content"])
                             for m in body.get("messages") or []):
            scripts = ()  # an image: no text to tell the source's characters by
        if scripts:
            params["ban_scripts"] = scripts
            if whitelist:
                params["ban_source_whitelist"] = True
            # "Allow emoticons and symbols from other scripts (kaomoji)": the shared config setting
            # script_ban_allow_kaomoji, sent by unified_api_client like the request kind and target
            if script_ban._truthy(body.get("glossarion_allow_kaomoji")):
                params["ban_allow_kaomoji"] = True
        return params

    # ---- downloads -----------------------------------------------------------------------------------

    async def resolve(self, ref: hf.HfRef) -> hf.RepoFiles:
        """The files and vision projectors of ``ref`` (Hugging Face, off the loop)."""
        return await self._io(lambda: hf.list_repo(ref, session=self.downloads.session))

    def spec_for(self, repo: str, commit: str, file: hf.HfFile) -> dl.DownloadSpec:
        return dl.DownloadSpec(repo, file.path, commit, int(file.size or 0), file.sha256,
                               self.registry.stem_for(repo, file.path))

    def projector_spec_for(self, stem: str, repo: str, commit: str, file: hf.HfFile) -> dl.DownloadSpec:
        """The download of ``stem``'s vision projector ``file`` (same repository and commit)."""
        return dl.DownloadSpec(repo, file.path, commit, int(file.size or 0), file.sha256,
                               self.registry.projector_key(stem), kind="mmproj", owner=stem)

    async def download(self, spec: dl.DownloadSpec) -> dl.DownloadState:
        state = await self.downloads.download(spec)
        if state.done:
            if spec.kind == "mmproj" and self.loaded_stem == spec.owner and self.engine is not None \
                    and not self.engine.vision:
                self.unload()  # loaded text-only: the next request loads it again with its projector
            self.request_catalog_refresh()
        return state

    def delete(self, stem: str) -> bool:
        """Delete a downloaded model (unloads it first). False without deleting anything while the model
        is loading or answering (``busy_with(stem)``): its file must not go away under native code."""
        if self.busy_with(stem):
            return False
        if self._stem == stem and self.engine is not None and self.engine.loaded and not self.unload():
            return False
        removed = self.registry.remove(stem)
        self.downloads.cancel(self.registry.projector_key(stem))  # its image support's running / paused part
        self.request_catalog_refresh()
        return removed

    def busy_with(self, stem: str) -> bool:
        """True while ``stem`` is loading or generating (a delete must wait)."""
        return self._stem == stem and (self.gate.busy or self._state in ("loading", "generating"))

    # ---- unload policy -------------------------------------------------------------------------------

    def tick(self, now: Optional[float] = None) -> Optional[str]:
        """Apply the unload policy; returns why it unloaded (``"idle"`` / ``"background"``) or None."""
        now = self.clock() if now is None else now
        if self.engine is None or not self.engine.loaded or self.gate.busy:
            return None
        if self._unload_when_idle:
            self._deferred_unload()
            if self.loaded_stem is None:
                return "switched"
        if self.job_active():
            return None
        if self.background_since is not None and self.platform == "android" \
                and now - self.background_since >= BACKGROUND_UNLOAD:
            return "background" if self.unload() else None
        if now - self.last_used >= IDLE_UNLOAD:
            return "idle" if self.unload() else None
        return None

    def on_lifecycle(self, name: str) -> Optional[str]:
        """``GlossarionApp.on_lifecycle``: returns what it did (tests)."""
        name = str(name or "").lower()
        if name in ("hide", "pause", "inactive"):
            if name == "inactive":
                return None
            if self.background_since is None:
                self.background_since = self.clock()
            if self.platform == "ios" and not self.job_active():
                return "unloaded" if self.unload() else None
            return None
        if name in ("show", "resume", "restart"):
            self.background_since = None
            return None
        if name == "detach":
            # Android destroys the activity (Back, or memory pressure in the background) while the process
            # and a job under its foreground service go on: that job's requests still need the endpoint, so
            # it stays up (an idle endpoint costs nothing; the process' end closes it). Without a job the
            # model is freed now.
            if not self.job_active():
                self.unload()
            return "detached"
        return None

    def _watch_loop(self) -> None:
        while not self._stop_watch.wait(WATCH_INTERVAL):
            try:
                self.tick()
            except Exception:
                log.debug("unload policy failed", exc_info=True)


def _primed(deltas: Any, n_ctx: int) -> Any:
    """Run the generation up to its first delta here, so loading the projector, evaluating the prompt and
    their errors happen before the endpoint answers (a 400 instead of a broken stream)."""
    import itertools

    try:
        first = next(deltas)
    except StopIteration:
        return iter(())
    except eng.ContextOverflow as exc:
        deltas.close()
        raise LocalLLMError(400, "context_length_exceeded",
                            f"This model's maximum context length is {n_ctx} tokens. However, your messages with "
                            f"their images did not fit ({exc}). Please reduce the length of the messages.") from exc
    except eng.EngineUnavailable as exc:
        deltas.close()
        raise LocalLLMError(400, "unsupported_content", str(exc)) from exc
    rest = itertools.chain([first], deltas)

    class _Run:
        def __iter__(self) -> Any:
            return rest

        def __next__(self) -> Any:
            return next(rest)

        def close(self) -> None:
            deltas.close()

    return _Run()
