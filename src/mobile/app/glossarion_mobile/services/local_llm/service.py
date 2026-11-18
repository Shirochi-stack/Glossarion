"""LocalEngineService: the on-device engine's lifecycle (one per app; Models & keys feature install).

* At start it runs ``LocalLLMServer`` (cheap while idle), publishes ``GLOSSARION_LOCAL_LLM_URL`` for the
  shared ``local/`` route, lists unfinished downloads and refreshes the ``local`` catalog. No model is
  loaded at start.
* **One model loaded at a time.** A request for a model that is not loaded loads it on the server thread
  under the generation gate (state ``loading`` then ``ready``); a request for another model unloads the
  current one first, unless the running job uses it (409, no thrash). Generations run one at a time in
  arrival order (FIFO, at most ``MAX_WAITERS`` waiting, then 503). Opt-in parallel requests use two
  independent contexts; the extra context is freed after its answer (``parallel.RequestSlots``).
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
  on an immediate Stop, a disconnect or a forced unload, and runs to its end on a graceful one. An
  immediate Stop reaches a prompt being evaluated too (between two decodes, ``engine.GenerationStopped``)
  and stays in force for the requests that had arrived (``stop_generation``), also after the next run
  clears the client's stop flags or when a socket closed under a blocked read never tells the endpoint.
  Native code is never interrupted or freed under: a switch (``load_blocking``) right after a Stop waits
  for that generation to end and then loads by itself; only a running job's live answer refuses it. The
  answer of a job that is over (stopped, or ended, while its answer still runs: ``orphaned``) is told to end
  at its next stop check first, so the switch waits a few tokens, not the rest of an answer nobody reads.
* **Adaptive script ban** ("Block other scripts in translations", Prefs ``BLOCK_CJK_KEY``, on by default):
  a request that writes target-language text bans the tokens of every script the target does not use
  (English: Han / kana / Hangul, Devanagari, Thai, Arabic, Cyrillic ...; a CJK target keeps all three CJK
  scripts); Extract Glossary, refinement and a single-pass auto glossary's translations ban them too, except
  the tokens of their source text's characters (``params["ban_source_whitelist"]``, owner 2026-10-30; an image
  has no text and is not banned), and "Whitelist source text" (config ``script_ban_whitelist_source``, body
  field ``glossarion_whitelist_source``) does that for every banned request. The
  request kind and target come from the body fields ``unified_api_client`` adds for this endpoint only
  (``glossarion_task``, ``target_language``, ``glossarion_single_pass``); ``script_ban.banned_scripts``
  decides and the engine gets ``params["ban_scripts"]``. The opt-in "Allow emoticons and symbols from
  other scripts (kaomoji)" (config ``script_ban_allow_kaomoji``, shared with the API models' ban; shown
  under the switch too) arrives as ``glossarion_allow_kaomoji`` and becomes ``params["ban_allow_kaomoji"]``.
* **GPU (experimental, off by default;** Prefs ``GPU_KEY``, Model manager › On-device › engine card): offered
  where the build has a GPU backend (``engine.gpu_offer``: Metal on iOS, OpenCL / Adreno on Android
  arm64). With it on, a load offloads the saved layer count (eight by default; all is opt-in); when that fails
  (``GpuUnavailable``: no backend / device, the model or its first token failed) or a GPU answer fails,
  the reason is logged and kept (Prefs ``GPU_FALLBACK_KEY``), the model loads on the CPU, and that model
  stays on the CPU until the setting or the model changes; the engine card says "GPU unavailable: …".
  A GPU load leaves a marker (``GPU_ATTEMPT_KEY``) until the model's first answer ends: when the app
  died in between (a driver crash aborts the process), the next start records the fallback instead of
  crashing again. The RAM gate adds ``gpu_extra`` (GPU memory is shared with the CPU on phones).
* **NPU (Snapdragon Hexagon):** a third backend, chosen per model at load time (``_placement``). Two stages on
  Android arm64. Stage 1 (``run_npu_check``: ``engine.npu_precheck``, cheap: the build carries the backend,
  libcdsprpc opens, the cDSP reports its arch) runs once per app version, in the background from ``start`` on the
  version's first start (``NPU_CHECK_KEY`` keeps its answer for later starts). Stage 2 (``_npu_session``: the
  wheel's ``register_npu_backend`` through ``engine.npu_state``, which opens a real DSP session) runs at most once
  per process, on the loading thread before the first NPU load, and only once the phone's HTP skeleton is on disk:
  the APK carries none (the Hexagon SDK licence), so that load first downloads ``engine.npu_skeleton_source()``'s
  llama.cpp release archive (``downloads.fetch_blocking``: one plain GET, https redirects only, no retries) and
  takes only its pinned skeletons (``downloads.extract_pinned``) into ``npu_skel_dir(<release>)``; folders of
  other releases go once it is complete. A failed download is not a model fallback: that load runs on the GPU /
  CPU, the NPU line says why, and it is tried again at the next start or "Try the NPU again". A Stop (or the
  service stopping) during the download ends that load instead (nothing loads, nothing is recorded); the next NPU
  load resumes the part. Either
  stage still running after ``NPU_WATCHDOG_SECONDS``, or one the app died in (marker ``NPU_CHECK_ATTEMPT_KEY``
  with its stage), and a failing stage 2 (``NPU_SESSION_KEY``) keep the NPU off until the app version changes.
  The NPU is offered (``npu_offered``) when stage 1 passed and stage 2 has not failed. A model's ``backend`` (its
  preset row; "auto" for every other model, Add from Hugging Face included): "auto" runs on the NPU when it is
  offered, "Use NPU" (``NPU_KEY``, on when unset) is on, the model has no NPU fallback and its quant is one
  the backend accepts (``presets.npu_quant_kind`` not "cpu"), else today's GPU / CPU rule; "npu" / "gpu" / "cpu"
  name the device. An NPU load that fails (or its first token), a failing NPU answer, a first NPU load still
  running after ``NPU_WATCHDOG_SECONDS`` (the restart message; native code is never killed) or a marker still set
  at the next start (``NPU_ATTEMPT_KEY`` around an NPU load and its first token, ``NPU_INFLIGHT_KEY`` around
  every NPU answer while the app is in the foreground: the backend aborts the process on DSP failures; both are
  written to disk before the native call) records an NPU fallback for that model
  (``NPU_FALLBACK_KEY``): it runs on the GPU / CPU until its file, its backend choice or the app version changes,
  or "Try the NPU again" (``retry_npu``). No retry loop. The second parallel context and the vision projector
  never use the NPU. Power mode: ``NPU_POWER_KEY`` (upstream's modes, default burst), applied when the backend
  registers (a change applies after a restart). The screens only read this state (``npu_progress``,
  ``npu_status_line``): no stage, download or NPU lock runs on the UI thread.
"""

from __future__ import annotations

import asyncio
import collections
import dataclasses
import logging
import os
import re
import sys
import threading
import time
from typing import Any, Callable, Optional

from glossarion_mobile.services.local_llm import downloads as dl
from glossarion_mobile.services.local_llm import engine as eng
from glossarion_mobile.services.local_llm import hf, presets, script_ban, stem_of
from glossarion_mobile.services.local_llm import vision as vis
from glossarion_mobile.services.local_llm.parallel import PARALLEL_KEY, RequestSlots
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
# Match the pinned llama-cpp-python defaults, without the small-model penalties.
# Temperature continues to come from the request's normal translation settings.
SIGMA_SAMPLING_KEY = "sigma_q8_sampling"
DEFAULT_SIGMA_SAMPLING = {**DEFAULT_ONDEVICE_SAMPLING, "top_p": 0.95, "top_k": 40,
                          "min_p": 0.05, "presence_penalty": 0.0, "dry_multiplier": 0.0}
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
#: "Forced Truncation" of the on-device engine (on by default) and its seconds (``forced_truncation``)
FORCED_TRUNCATION_KEY = "local_llm_forced_truncation"
FORCED_TRUNCATION_SECONDS_KEY = "local_llm_forced_truncation_seconds"
#: "Mark glossary forced truncation as complete" of the on-device engine (on by default)
MARK_GLOSSARY_COMPLETE_KEY = "local_llm_mark_glossary_complete"
#: Prefs key of "Use GPU (experimental)" (a bool, off when unset)
GPU_KEY = "local_llm_gpu"
GPU_LAYERS_KEY = "local_llm_gpu_layers"
MAX_GPU_LAYERS = 128
#: Prefs key of the GPU fallback: ``{"stem", "reason"}`` (that model runs on the CPU until the setting or
#: the model changes)
GPU_FALLBACK_KEY = "local_llm_gpu_fallback"
#: Prefs key of a GPU load whose first answer has not ended yet: ``{"stem"}`` (crash detection)
GPU_ATTEMPT_KEY = "local_llm_gpu_attempt"
CRASH_REASON = "Glossarion closed while the GPU was starting this model"
#: Prefs key of "Use NPU" (a bool, on when unset; shown only where the NPU check passed)
NPU_KEY = "local_llm_npu"
#: Prefs key of the NPU power mode (``engine.NPU_POWER_MODES``; unset: upstream's default)
NPU_POWER_KEY = "local_llm_npu_power"
#: Prefs key of the NPU fallbacks: ``{stem: {"reason", "file", "backend", "version", "at"}}``
NPU_FALLBACK_KEY = "local_llm_npu_fallback"
#: Prefs key of an NPU load running now (its first token included): ``{"stem", "at"}``
NPU_ATTEMPT_KEY = "local_llm_npu_attempt"
#: Prefs key of an NPU answer running now: ``{"stem", "at", "background"}`` (set before every NPU answer, cleared
#: after; ``background``: the app is in the background, where an ordinary kill ends it too, so it records nothing)
NPU_INFLIGHT_KEY = "local_llm_npu_inflight"
#: Prefs key of the last NPU check (stage 1, ``engine.npu_precheck``): ``{"version", "state", "blocked", "seconds",
#: "source"}`` (blocked: it hung or the app died in it; source: ``engine.npu_skeleton_source()``)
NPU_CHECK_KEY = "local_llm_npu_check"
#: Prefs key of the last NPU start (stage 2, ``register_npu_backend``): ``{"version", "state", "blocked", "seconds"}``
NPU_SESSION_KEY = "local_llm_npu_session"
#: Prefs key of an NPU stage running now: ``{"version", "at", "stage"}`` (stage 1 or 2)
NPU_CHECK_ATTEMPT_KEY = "local_llm_npu_check_attempt"
NPU_CRASH_REASON = "Glossarion closed while the NPU was running this model"
NPU_CHECK_CRASH_REASON = "Glossarion closed while the NPU check was running"
NPU_START_CRASH_REASON = "Glossarion closed while the NPU was starting"
#: seconds to connect for the NPU files' download (an offline phone falls through to the GPU / CPU quickly)
NPU_FETCH_TIMEOUT = (10.0, 30.0)
NPU_CHECKING_TEXT = "Checking the NPU…"
NPU_STARTING_TEXT = "Starting the NPU…"
NPU_FILES_ERROR_TEXT = "NPU files not downloaded: {reason}"
NPU_HANG_REASON = "the NPU did not answer within 120 s"
NPU_CHECK_HANG_NOTE = "The NPU check did not answer. The NPU stays off until the next app update."
#: seconds the NPU check and a first NPU load get before the NPU counts as not answering
NPU_WATCHDOG_SECONDS = 120.0
#: seconds stage 1 runs before the screens show "Checking the NPU…": a build without the backend and a phone
#: without a Hexagon DSP answer at once and never show it (they look as they did before the NPU)
NPU_CHECK_QUIET_SECONDS = 0.5
MAX_WAITERS = 8
IDLE_UNLOAD = 600.0
BACKGROUND_UNLOAD = 60.0
WATCH_INTERVAL = 15.0
MIB = 1024 * 1024
GIB = 1024 * MIB
NOT_DOWNLOADED = "Download it in Model manager › On-device"
NO_VISION = "This on-device model has no vision support"
#: ``load_blocking`` refused: the engine gives the live answer of a running job (a Stop lets the switch go ahead)
BUSY_FOR_JOB = "The on-device model is answering for a running job; stop the job to switch models"
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


#: Shared on-device default. Saved lower contexts and per-model limits take precedence.
DEFAULT_CONTEXT_SMALL = presets.DEFAULT_CONTEXT
DEFAULT_CONTEXT_LARGE = presets.DEFAULT_CONTEXT
#: What the Welcome screen saves for every on-device model when no Context is saved yet (owner 2026-10-31).
WELCOME_CONTEXT = presets.DEFAULT_CONTEXT


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


def current_app_version() -> str:
    """This app build and its engine (``runtime_bootstrap``'s version and bundle, ``ENGINE_VERSION``): an NPU
    fallback or a blocked NPU check lasts until it changes."""
    version: dict = {}
    try:
        from glossarion_mobile import runtime_bootstrap as rb

        state = rb.get_state()
        version = dict(state.version or {}) if state is not None else {}
    except Exception:
        version = {}
    app = "+".join(str(part) for part in (version.get("version") or "dev", version.get("bundle")) if part)
    return f"{app}/llama-{eng.ENGINE_VERSION}"


def npu_htp_root() -> str:
    """``<app data>/htp``: the HTP skeleton folders (the app's private files directory on Android; FastRPC
    cannot read them from inside the APK)."""
    base = ""
    try:
        from glossarion_mobile import runtime_bootstrap as rb

        paths = rb.get_paths()
        base = str(paths.data) if paths is not None else ""
    except Exception:
        base = ""
    base = base or os.environ.get("GLOSSARION_DATA_DIR", "").strip() or os.path.join(os.path.expanduser("~"), ".glossarion")
    return os.path.join(base, "htp")


def npu_skel_dir(release: str, root: Optional[str] = None) -> str:
    """Where the downloaded HTP skeletons live: ``<root>/<skeleton release>`` (``npu_skeleton_source()["release"]``,
    not the app version: an app update with the same engine keeps them)."""
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", str(release or "dev")).strip("_.") or "dev"
    return os.path.join(root or npu_htp_root(), name)


def _npu_blocked_state(reason: str) -> dict:
    return {"backend": None, "device": "", "arch": None, "error": reason,
            "steps": [{"name": "session", "ok": False, "detail": reason}]}


#: one NPU files' download at a time in the process (the self-check's services share the folder)
_NPU_FETCH_LOCK = threading.Lock()
#: ``_fetch_npu_files``' answer when a Stop or the service stopping ended the download
NPU_FETCH_CANCELLED = "cancelled"
#: the release archive's name in the skeleton folder while it downloads (``.part``) and until it is extracted
NPU_ARCHIVE_NAME = "skeletons.tar.gz"


def _npu_bytes_text(done: int, total: int) -> str:
    """"Getting NPU files · 12 / 26 MB"."""
    if total:
        return f"Getting NPU files · {done / 1e6:.0f} / {total / 1e6:.0f} MB"
    return f"Getting NPU files · {done / 1e6:.0f} MB"


def _anonymous_session() -> Any:
    """A ``requests`` session that sends nothing of the phone's: no netrc credentials, no environment proxies or
    cookies (``trust_env`` off)."""
    import requests

    session = requests.Session()
    session.trust_env = False
    return session


def _remove_other_npu_folders(current: str) -> None:
    """Once ``current`` holds the phone's skeleton: every other folder in its parent goes (other skeleton releases,
    and the per-app-version folders of the earlier test build)."""
    import shutil

    root = os.path.dirname(os.path.abspath(current))
    keep = os.path.basename(os.path.abspath(current))
    try:
        names = os.listdir(root)
    except OSError:
        return
    for name in names:
        path = os.path.join(root, name)
        if name == keep or os.path.islink(path) or not os.path.isdir(path):
            continue
        shutil.rmtree(path, ignore_errors=True)
        log.info("on-device NPU: removed the old NPU files folder %s", name)


def _tidy_npu_folder(current: str) -> None:
    """Once ``current`` holds the phone's skeleton: what a download the app died in left there goes (the archive,
    its part and sidecar, unfinished ``.tmp`` skeletons). Under the fetch lock: never under a running extraction."""
    with _NPU_FETCH_LOCK:
        try:
            names = os.listdir(current)
        except OSError:
            return
        for name in names:
            if name.startswith(NPU_ARCHIVE_NAME) or name.endswith(".tmp"):
                dl._remove(os.path.join(current, name))
                log.info("on-device NPU: removed the leftover %s", name)


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

    def take_over(self, timeout: float) -> bool:
        """Forced unload / delete of a generation already told to stop: wait up to ``timeout`` s for the
        holder to let go and take the gate ahead of the waiters. False when the holder did not let go
        (the caller must then leave the engine alone; the holder stays stopped and lets go later)."""
        deadline = time.monotonic() + timeout
        with self._cond:
            while self._busy:
                left = deadline - time.monotonic()
                if left <= 0:
                    return False
                self._cond.wait(min(left, 0.25))
            self._busy = True
            return True


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
    backend: Optional[str] = None  # "NPU" / "GPU" / "CPU": what the loaded model actually runs on
    npu_note: Optional[str] = None  # "NPU unavailable for <model>: <reason>. Using CPU." / the restart message
    #: what a load is doing before the model itself loads ("Getting NPU files · 12 / 26 MB", "Starting the NPU…")
    phase: Optional[str] = None


class LocalEngineService:
    def __init__(self, *, prefs: Any = None, registry: Optional[dl.Registry] = None,
                 engine_factory: Optional[Callable[[], Any]] = None, run_io: Optional[Callable[..., Any]] = None,
                 post: Optional[Callable[..., Any]] = None, refresh_catalog: Optional[Callable[[], Any]] = None,
                 active_job_model: Optional[Callable[[], Optional[str]]] = None, platform: Optional[str] = None,
                 clock: Callable[[], float] = time.monotonic, memory: Optional[Callable[[], tuple]] = None,
                 watch: bool = True, downloads: Optional[dl.DownloadManager] = None,
                 stop_state: Callable[[], Optional[str]] = process_stop_state,
                 gpu_backend: Optional[str] = None, npu_probe: Optional[Callable[[str, dict], dict]] = None,
                 app_version: Optional[str] = None, recover_npu: bool = True,
                 npu_precheck: Optional[Callable[[], dict]] = None,
                 npu_source: Optional[Callable[[], Optional[dict]]] = None, npu_root: Optional[str] = None,
                 npu_session: Any = None, _parallel_child: bool = False) -> None:
        self.prefs = prefs
        #: ``start`` turns NPU markers left by a dead process into fallbacks (the self-check's own services do
        #: not: they share the app's markers while the app's own answers may be running)
        self._recover_npu_at_start = recover_npu
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
        self._loaded_gpu_layers = 0
        self._gpu_attempt_key = GPU_ATTEMPT_KEY + ("_parallel" if _parallel_child else "")
        self._gpu_attempt: Optional[str] = None  # stem of a GPU load whose first answer has not ended
        # NPU, on Android arm64 (tests pass fakes): stage 1 ``npu_precheck()`` (``engine.npu_precheck``), the
        # skeleton archive ``npu_source()`` (``engine.npu_skeleton_source``), stage 2 ``npu_probe(skel_dir,
        # options)`` (``engine.npu_state``); ``npu_root`` the skeleton folders' parent, ``npu_session`` the HTTP
        # client of the files' download (``requests``). The second parallel context never uses the NPU.
        self._is_child = bool(_parallel_child)
        platform_npu = eng.npu_platform(self.platform)
        self._npu_probe = npu_probe if npu_probe is not None else (eng.npu_state if platform_npu else None)
        self._npu_precheck = npu_precheck if npu_precheck is not None else (
            eng.npu_precheck if platform_npu else None)
        self._npu_source_fn = npu_source if npu_source is not None else eng.npu_skeleton_source
        if self._is_child or self._npu_precheck is None:
            self._npu_probe = None
        self._npu_root = npu_root
        self._npu_http = npu_session
        self.app_version = app_version or current_app_version()
        self._npu_state: Optional[dict] = None  # this process's stage-1 answer (None: not run yet)
        self._npu_session_state: Optional[dict] = None  # this process's stage-2 answer (None: not run yet)
        self._npu_source: Optional[dict] = None  # ``npu_skeleton_source()`` as stage 1 read it
        self._npu_lock = threading.Lock()  # stage 1 (never taken on the UI thread)
        self._npu_prepare_lock = threading.Lock()  # the files and stage 2 (loading threads only)
        self._npu_checking = False  # stage 1 has run longer than ``npu_quiet_seconds`` (the screens show it)
        self._npu_phase: Optional[str] = None  # "downloading" | "starting" while a load prepares the NPU
        self._npu_bytes = (0, 0)  # the files' download: (done, total)
        self._npu_fetch_tried = False  # this app start downloaded (or tried to download) the NPU files
        self.npu_download_error: Optional[str] = None  # why the last download failed (this process)
        self._npu_files_present: Optional[bool] = None  # the phone's skeleton is on disk (cached stat)
        self._npu_cancel = threading.Event()  # the service stops: a running download ends
        self.npu_hung = False  # the check or a first NPU load did not answer in time
        self._npu_hung_stem: Optional[str] = None  # the model whose first NPU load hung (its restart message)
        self._npu_inflight: Optional[str] = None  # stem of the NPU answer running now (its in-flight marker)
        self._npu_marker_lock = threading.Lock()
        self.npu_watchdog_seconds = NPU_WATCHDOG_SECONDS
        self.npu_quiet_seconds = NPU_CHECK_QUIET_SECONDS
        self._npu_loaded_once = False  # a first NPU load returned in this process (the watchdog is for the first)
        self._npu_attempt: Optional[str] = None  # stem of an NPU load whose first answer has not ended
        self._loaded_npu = False
        #: what ran the last load (diagnostics): ``{"stem", "backend", "device", "seconds"}``
        self.last_load: Optional[dict] = None
        #: a backend the self-check forces for its timing runs ("npu" / "gpu" / "cpu"; None: the usual choice)
        self.backend_override: Optional[str] = None
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
        # immediate Stops so far: a request that arrived before the latest one is stopped for good
        self._stops = 0
        self._holder_stops: Optional[int] = None  # ``_stops`` when the generation holding the gate arrived
        self._holder_job = False  # a running job pinned this model when the generation holding the gate arrived
        self._holder_token: Optional[object] = None  # the generation holding the gate
        self._cut: Optional[object] = None  # ``_holder_token`` of an orphaned generation told to end (``end_orphan``)
        self._unload_when_idle = False  # the selection moved off the loaded model while it answered
        self.log: list = []  # (stem, load seconds) of every load (diagnostics)
        self._parallel = None if _parallel_child else RequestSlots(self)
        self._migrate_sigma_defaults()
        #: ``Registry.repair_projector_models`` at start: projectors an earlier version downloaded as models
        self.repaired: list = [] if _parallel_child else self._repair_projector_models()

    def _repair_projector_models(self) -> list:
        try:
            return self.registry.repair_projector_models()
        except Exception:
            log.exception("repairing projector downloads failed")
            return []

    def _migrate_sigma_defaults(self) -> None:
        if self.prefs is None or self._pref("sigma_defaults_v1"):
            return
        settings = self.settings()
        if settings["n_ctx"] > presets.DEFAULT_CONTEXT:
            self.save_context(presets.DEFAULT_CONTEXT)
        rows = presets.load_presets(self.prefs)
        changed = False
        for row in rows:
            if row.get("n_ctx", 0) > presets.DEFAULT_CONTEXT:
                row["n_ctx"] = presets.DEFAULT_CONTEXT
                changed = True
        # Old first-run installations may never have saved the stock Heavy row.
        for entry in self.registry.load().values():
            if len(rows) >= presets.MAX_PRESETS:
                break  # downloaded models remain available in the separate downloaded list
            if str(entry.get("repo") or "").casefold() != "unsloth/qwen3.5-9b-gguf":
                continue
            if any(row.get("repo", "").casefold() == entry["repo"].casefold()
                   and row.get("file") == entry.get("file") for row in rows):
                continue
            label = f"Qwen 3.5 9B · {presets.quant_label(entry.get('file'))}"
            rows.append({"name": label, "note": f"Custom · {label}", "repo": entry["repo"],
                         "file": entry["file"], "quant": "", "n_ctx": 0, "size": entry.get("size", 0)})
            changed = True
        if changed:
            presets.save_presets(self.prefs, rows)
        self._set_pref("sigma_defaults_v1", True)

    def parallel_requests(self) -> bool:
        return bool(self._pref(PARALLEL_KEY))

    def save_parallel_requests(self, enabled: Any) -> bool:
        value = bool(enabled)
        self._set_pref(PARALLEL_KEY, value)
        if self._parallel is not None:
            self._parallel.changed()
        self._notify()
        return value

    def uses_normal_settings(self, stem: Optional[str] = None) -> bool:
        stem = str(stem or self._stem or "").removeprefix("local/")
        return presets.uses_normal_settings(self.registry.load().get(stem) or {})

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
        snap = EngineSnapshot(self._state if (loaded or self._state == "loading") else "idle",
                              self._stem if (loaded or self._state == "loading") else None,
                              self.gate.waiting(), self._error, self._system_info, available, reason,
                              self._load_seconds, self.server.running,
                              bool(loaded and getattr(self.engine, "vision", False)),
                              getattr(self.engine, "device", None) if loaded else None, self.gpu_note(),
                              self._engine_backend() if loaded else None, self.npu_note(),
                              self._npu_load_phase() if self._state == "loading" else None)
        pool = self._parallel
        if pool is not None:
            secondary = pool.secondary
            if secondary is not None and secondary.gate.busy and not self.gate.busy:
                snap = dataclasses.replace(secondary.snapshot(), running=self.server.running)
            snap = dataclasses.replace(snap, queue=snap.queue + len(pool.queue))
        return snap

    def health(self, model: Optional[str] = None) -> dict:
        """``{"state", "model", "queue"}`` (plus ``busy`` and ``load_seconds`` of the last load); with
        ``model`` also ``target``: ``{"stem", "size", "loaded"}`` of that model (``stem`` None when it is
        not downloaded). The shared client reads it before a request to log loading / waiting lines."""
        snap = self.snapshot()
        out = {"state": snap.state, "model": snap.model, "queue": snap.queue, "busy": bool(self.gate.busy or (self._parallel and self._parallel.active)),
               "load_seconds": self._load_seconds}
        if self.parallel_requests() or (self._parallel and self._parallel.used):
            out.update(parallel_requests=self.parallel_requests(), request_slots=2 if self.parallel_requests() else 1)
        if model:
            stem = self.registry.resolve(model)
            size = int((self.registry.load().get(stem) or {}).get("size") or 0) if stem else 0
            out["target"] = {"stem": stem, "size": size,
                             "loaded": bool(stem and snap.model == stem and snap.state != "loading")}
            if out["target"]["loaded"] and snap.backend and (self.npu_offered() or snap.npu_note):
                out["backend"] = snap.backend  # the job log's "loaded … · runs on NPU" (phones with the NPU)
                if snap.npu_note:
                    out["npu_note"] = snap.npu_note
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
        if self._npu_probe is not None:
            self._npu_cancel.clear()
            if self._recover_npu_at_start:
                self._recover_npu()
            self._start_npu_check()
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
        self._npu_cancel.set()  # a running NPU files' download ends (its part stays for the next start)
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
        for key, low, high in (("n_ctx", 512, presets.MAX_CTX), ("n_threads", 1, self.thread_limit())):
            try:
                value = int(raw.get(key) or 0)
            except (TypeError, ValueError):
                value = 0
            out[key] = min(value, high) if key == "n_threads" and value > 0 else value if low <= value <= high else 0
        return out

    def save_settings(self, n_ctx: Any = 0, n_threads: Any = 0) -> tuple:
        """(settings, None) or (None, message)."""
        values = {}
        for key, value, low, high in (("n_ctx", n_ctx, 512, presets.MAX_CTX), ("n_threads", n_threads, 1, self.thread_limit())):
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

    def sampling(self, stem: Optional[str] = None) -> dict:
        """Editable defaults for the selected model, with Q8 choices saved separately."""
        normal = self.uses_normal_settings(stem)
        try:
            raw = self.prefs.get(SIGMA_SAMPLING_KEY if normal else SAMPLING_KEY, None) if self.prefs is not None else None
        except Exception:
            raw = None
        raw = raw if isinstance(raw, dict) else {}
        out = dict(DEFAULT_SIGMA_SAMPLING if normal else DEFAULT_ONDEVICE_SAMPLING)
        for key, (low, high, kind) in SAMPLING_LIMITS.items():
            try:
                value = kind(raw[key])
            except (KeyError, TypeError, ValueError):
                continue
            if low <= value <= high:
                out[key] = value
        return out

    def save_sampling(self, values: dict, *, stem: Optional[str] = None) -> tuple:
        """(sampling, None) or (None, message); blanks restore this model's defaults."""
        normal = self.uses_normal_settings(stem)
        defaults = DEFAULT_SIGMA_SAMPLING if normal else DEFAULT_ONDEVICE_SAMPLING
        out = {}
        for key, (low, high, kind) in SAMPLING_LIMITS.items():
            text = str((values or {}).get(key, "") if (values or {}).get(key) is not None else "").strip()
            if not text:
                out[key] = defaults[key]
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
            self.prefs.set(SIGMA_SAMPLING_KEY if normal else SAMPLING_KEY, dict(out))
        return out, None

    def block_cjk(self, stem: Optional[str] = None) -> bool:
        """"Block other scripts in translations (adaptive)": on unless switched off."""
        if self.uses_normal_settings(stem):
            return bool(self._pref("sigma_q8_block_cjk"))
        try:
            raw = self.prefs.get(BLOCK_CJK_KEY, None) if self.prefs is not None else None
        except Exception:
            raw = None
        return raw is None or bool(raw)

    def forced_truncation(self, stem: Optional[str] = None) -> bool:
        """"Forced Truncation": on unless switched off."""
        if self.uses_normal_settings(stem):
            return bool(self._pref("sigma_q8_forced_truncation"))
        raw = self._pref(FORCED_TRUNCATION_KEY)
        return raw is None or bool(raw)

    def forced_truncation_seconds(self) -> int:
        try:
            return max(1, min(600, int(self._pref(FORCED_TRUNCATION_SECONDS_KEY))))
        except (TypeError, ValueError):
            return 3

    def mark_glossary_complete(self, stem: Optional[str] = None) -> bool:
        """"Mark glossary forced truncation as complete": on unless switched off."""
        if self.uses_normal_settings(stem):
            return bool(self._pref("sigma_q8_mark_glossary_complete"))
        raw = self._pref(MARK_GLOSSARY_COMPLETE_KEY)
        return raw is None or bool(raw)

    def save_mark_glossary_complete(self, enabled: Any, *, stem: Optional[str] = None) -> bool:
        value = bool(enabled)
        if self.prefs is not None:
            key = "sigma_q8_mark_glossary_complete" if self.uses_normal_settings(stem) else MARK_GLOSSARY_COMPLETE_KEY
            self.prefs.set(key, value)
        return value

    def save_forced_truncation(self, enabled: Any, seconds: Any = None, *, stem: Optional[str] = None) -> tuple:
        """Save the switch and, when given, its seconds; the values now in force."""
        if self.prefs is not None:
            key = "sigma_q8_forced_truncation" if self.uses_normal_settings(stem) else FORCED_TRUNCATION_KEY
            self.prefs.set(key, bool(enabled))
            if seconds is not None:
                try:
                    self.prefs.set(FORCED_TRUNCATION_SECONDS_KEY, max(1, min(600, int(float(seconds)))))
                except (TypeError, ValueError):
                    pass
        return self.forced_truncation(stem), self.forced_truncation_seconds()

    def save_block_cjk(self, enabled: Any, *, stem: Optional[str] = None) -> bool:
        value = bool(enabled)
        if self.prefs is not None:
            self.prefs.set("sigma_q8_block_cjk" if self.uses_normal_settings(stem) else BLOCK_CJK_KEY, value)
        return value

    # ---- GPU -------------------------------------------------------------------------------------------

    def _pref(self, key: str) -> Any:
        try:
            return self.prefs.get(key, None) if self.prefs is not None else None
        except Exception:
            return None

    def _set_pref(self, key: str, value: Any, *, durable: bool = False) -> None:
        """``durable``: on disk before this returns (``Prefs.set`` only schedules a debounced save), for the NPU
        crash markers set right before native code that may take the process down."""
        if self.prefs is None:
            return
        try:
            self.prefs.set(key, value)
            flush = getattr(self.prefs, "flush", None) if durable else None
            if callable(flush):
                flush()
        except Exception:
            log.debug("saving %s failed", key, exc_info=True)

    def use_gpu(self) -> bool:
        """"Use GPU (experimental)": off unless switched on, and only where a GPU backend is offered."""
        return bool(self.gpu_backend) and bool(self._pref(GPU_KEY))

    def gpu_layer_limit(self, stem: Optional[str] = None) -> Optional[int]:
        """Transformer blocks plus the output layer of the selected downloaded GGUF."""
        from .model_limits import gpu_layer_limit
        stem = self.registry.resolve(stem) if stem else self.loaded_stem
        return gpu_layer_limit(self.registry.file_path(stem)) if stem else None

    def gpu_layers(self, stem: Optional[str] = None) -> int:
        """Eight layers by default; zero uses CPU, -1 is an explicit all-layers choice."""
        try:
            value = int(self._pref(GPU_LAYERS_KEY))
        except (TypeError, ValueError, OverflowError):
            value = eng.DEFAULT_GPU_LAYERS
        limit = self.gpu_layer_limit(stem)
        if value < -1 or value > 10_000:
            value = eng.DEFAULT_GPU_LAYERS
        return min(value, limit) if limit is not None and value >= 0 else value

    def save_gpu_layers(self, layers: Any, stem: Optional[str] = None) -> int:
        try:
            value = int(layers)
        except (TypeError, ValueError, OverflowError):
            return self.gpu_layers()
        if not -1 <= value <= (self.gpu_layer_limit(stem) or MAX_GPU_LAYERS):
            return self.gpu_layers()
        if value != self._pref(GPU_LAYERS_KEY):
            self._set_pref(GPU_LAYERS_KEY, value)
            self._set_pref(GPU_FALLBACK_KEY, None)
            if self.loaded_stem is not None and self._loaded_gpu:
                self.unload()  # busy generations finish; the next request uses the new count
            self._notify()
        return value

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
        if self.loaded_stem is not None and self._loaded_gpu != value and not self._loaded_npu:
            self.unload()
        self._notify()
        return value

    def _gpu_wanted(self, stem: str) -> bool:
        if not self.use_gpu() or self.gpu_layers() == 0:
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
        self._set_pref(self._gpu_attempt_key, {"stem": stem, "at": int(time.time())})

    def _clear_gpu_attempt(self) -> None:
        if self._gpu_attempt is not None or self._pref(self._gpu_attempt_key):
            self._gpu_attempt = None
            self._set_pref(self._gpu_attempt_key, None)

    def _recover_gpu_attempt(self) -> Optional[str]:
        """At start: a GPU load whose first answer never ended means the process died on the GPU (a
        driver abort cannot be caught): keep that model on the CPU. Returns its stem."""
        raw = self._pref(GPU_ATTEMPT_KEY) or self._pref(GPU_ATTEMPT_KEY + "_parallel")
        if not isinstance(raw, dict) or not raw.get("stem"):
            return None
        stem = str(raw["stem"])
        self._set_pref(GPU_ATTEMPT_KEY, None)
        self._set_pref(GPU_ATTEMPT_KEY + "_parallel", None)
        if self.use_gpu():
            self._gpu_failed(stem, CRASH_REASON)
        return stem

    # ---- NPU -------------------------------------------------------------------------------------------

    def _engine_backend(self) -> Optional[str]:
        engine = self.engine
        if engine is None or not engine.loaded:
            return None
        return "NPU" if getattr(engine, "on_npu", False) else "GPU" if getattr(engine, "on_gpu", False) else "CPU"

    def npu_possible(self) -> bool:
        """This build could have an NPU (Android arm64, or a test's fakes): the check runs here."""
        return self._npu_probe is not None

    def _saved_npu(self, key: str) -> Optional[dict]:
        """This app version's saved stage answer (``NPU_CHECK_KEY`` / ``NPU_SESSION_KEY``), else None."""
        saved = self._pref(key)
        if isinstance(saved, dict) and saved.get("version") == self.app_version and isinstance(saved.get("state"), dict):
            return saved
        return None

    def _stage1(self) -> Optional[dict]:
        """Stage 1's answer: this process's, else this app version's saved one (None: never checked)."""
        if self._npu_state is not None:
            return self._npu_state
        saved = self._saved_npu(NPU_CHECK_KEY)
        return saved["state"] if saved else None

    def _stage2(self) -> Optional[dict]:
        """Stage 2's answer: this process's, else this app version's saved one (None: not run for this version)."""
        if self._npu_session_state is not None:
            return self._npu_session_state
        saved = self._saved_npu(NPU_SESSION_KEY)
        return saved["state"] if saved else None

    def npu_state(self) -> Optional[dict]:
        """The NPU's latest answer: stage 2 (``register_npu_backend``) once it ran for this app version, else stage 1
        (None: never checked)."""
        return self._stage2() or self._stage1()

    def npu_offered(self) -> bool:
        """Stage 1 passed and stage 2 has not failed for this app version (this process's answers, or the saved
        ones until they have run): "Use NPU", the power mode and the Custom / Sigma quants' NPU marks show. Never on
        a build without the NPU functions. Reads saved state only (the screens call it)."""
        return self.npu_possible() and self.npu_failure() is None

    def npu_failure(self) -> Optional[tuple]:
        """``(step, reason)`` of why the NPU is not offered (the diagnostics line), None when it is."""
        if not self.npu_possible():
            return "shipped", "this build has no NPU (Android arm64 only)"
        state = self._stage1()
        if state is None:
            return "check", "not run yet"
        failure = eng.npu_precheck_failure(state)
        if failure is not None:
            return failure
        session = self._stage2()
        return eng.npu_failure(session) if session is not None else None

    def npu_options(self) -> dict:
        """The GGML_HEXAGON_* options ``register_npu_backend`` sets before the backend loads."""
        return {eng.NPU_POWER_ENV: self.npu_power()}

    def npu_skel_dir(self) -> Optional[str]:
        """This engine's skeleton folder (``npu_skel_dir(<release>)``), None before stage 1 read the release."""
        source = self.npu_source()
        return npu_skel_dir(source["release"], self._npu_root) if source else None

    def npu_source(self) -> Optional[dict]:
        """``engine.npu_skeleton_source()`` as stage 1 read it (saved with its answer; no native call here)."""
        if self._npu_source is not None:
            return self._npu_source
        saved = self._saved_npu(NPU_CHECK_KEY)
        source = saved.get("source") if saved else None
        return source if isinstance(source, dict) and source.get("files") else None

    def npu_skeleton(self) -> Optional[dict]:
        """The ``files`` entry of the phone's skeleton (its arch from stage 1), None when unknown."""
        state = self._stage1() or {}
        return eng.npu_skeleton_file(self.npu_source(), state.get("arch"))

    def npu_files_ready(self, refresh: bool = False) -> bool:
        """The phone's skeleton is in its folder: a regular file of the pinned size (a cached stat; ``refresh``
        looks again)."""
        if self._npu_files_present is None or refresh:
            folder, spec = self.npu_skel_dir(), self.npu_skeleton()
            present = False
            if folder and spec:
                path = os.path.join(folder, spec["name"])
                try:
                    present = os.path.isfile(path) and not os.path.islink(path) \
                        and os.path.getsize(path) == int(spec["size"])
                except OSError:
                    present = False
            if not (folder and spec) and not refresh:
                return False  # stage 1 has not named it yet: ask again later
            self._npu_files_present = present
        return bool(self._npu_files_present)

    def npu_seconds(self) -> dict:
        """``{"check": s, "start": s}`` of the stages that ran (this process's, else this version's saved ones)."""
        out = {}
        for name, key in (("check", NPU_CHECK_KEY), ("start", NPU_SESSION_KEY)):
            saved = self._saved_npu(key)
            if saved and isinstance(saved.get("seconds"), (int, float)):
                out[name] = float(saved["seconds"])
        return out

    def _mark_npu_stage(self, stage: int) -> None:
        self._set_pref(NPU_CHECK_ATTEMPT_KEY, {"version": self.app_version, "at": int(time.time()), "stage": stage},
                       durable=True)

    def _watched(self, name: str, call: Callable[[], dict], slow: Optional[Callable[[], None]] = None) -> tuple:
        """``(state, blocked, seconds)`` of ``call()`` on its own thread, given ``npu_watchdog_seconds``: a call
        still running then counts as failed (the restart message; native code cannot be stopped). ``slow`` runs
        once the call is still running after ``npu_quiet_seconds``."""
        answer: list = []

        def run() -> None:
            try:
                answer.append(dict(call() or {}))
            except Exception as exc:  # the loader never raises; a fake may
                answer.append(_npu_blocked_state(f"{type(exc).__name__}: {exc}"))

        started = time.monotonic()
        worker = threading.Thread(target=run, name=name, daemon=True)
        worker.start()
        if slow is not None:
            worker.join(min(self.npu_quiet_seconds, self.npu_watchdog_seconds))
            if not answer:
                slow()
        worker.join(max(0.0, self.npu_watchdog_seconds - (time.monotonic() - started)))
        seconds = round(time.monotonic() - started, 1)
        if not answer:
            self.npu_hung = True
            log.warning("on-device NPU %s did not answer within %.0f s: NPU off until the app version changes",
                        "check" if name.endswith("check") else "start", self.npu_watchdog_seconds)
            return _npu_blocked_state(NPU_HANG_REASON), True, seconds
        return answer[0], False, seconds

    def run_npu_check(self) -> dict:
        """Stage 1 (``engine.npu_precheck``: the build carries the backend, libcdsprpc opens, the cDSP reports its
        arch), once per app version: a saved answer of this version is used as it is (later calls answer the
        same). It also reads the skeleton archive the engine names. One that does not answer within
        ``npu_watchdog_seconds`` counts as failed. Background thread or a loading thread, never the UI."""
        with self._npu_lock:
            if self._npu_state is not None:
                return self._npu_state
            precheck = self._npu_precheck if self._npu_probe is not None else None
            if precheck is None:
                self._npu_state = {"backend": None, "device": "", "arch": None, "error": eng.NO_NPU,
                                   "steps": [{"name": "shipped", "ok": False, "detail": eng.NO_NPU}]}
                return self._npu_state
            saved = self._saved_npu(NPU_CHECK_KEY)
            if saved is not None:
                self._npu_state = saved["state"]
                return self._npu_state
            def slow() -> None:
                self._npu_checking = True
                self._notify()  # the screens show "Checking the NPU…"

            self._mark_npu_stage(1)
            state, blocked, seconds = self._watched("gl-llm-npu-check", precheck, slow)
            if not blocked:
                state = eng.npu_precheck_state(state)
            source = None
            if not blocked and eng.npu_precheck_failure(state) is None:
                try:
                    source = self._npu_source_fn()
                except Exception:
                    log.exception("npu_skeleton_source failed")
                if not source or not eng.npu_skeleton_file(source, state.get("arch")):
                    state = dict(state, error="this engine build names no NPU files for this phone")
                    state["steps"] = [*state.get("steps", []),
                                      {"name": "arch", "ok": False, "detail": state["error"]}]
                    source = None
            failure = eng.npu_precheck_failure(state)
            log.info("on-device NPU check in %.1f s: %s", seconds,
                     f"Hexagon v{state.get('arch')}, NPU files from llama.cpp {source['release']}" if failure is None
                     else f"not used ({failure[0]}: {failure[1]})")
            self._set_pref(NPU_CHECK_KEY, {"version": self.app_version, "state": state, "blocked": blocked,
                                           "seconds": seconds, "source": source})
            self._set_pref(NPU_CHECK_ATTEMPT_KEY, None, durable=True)
            self._npu_source = source
            self._npu_files_present = None
            self._npu_state = state
            self._npu_checking = False
        self._notify()
        return state

    def _start_npu_check(self) -> None:
        """The first start of an app version runs stage 1 in the background (the rows need its answer; the NPU
        section shows its progress once it takes longer than ``npu_quiet_seconds``); later starts show the saved
        answer. Stage 2 only runs before an NPU load."""
        if self._npu_probe is None or self._stage1() is not None:
            return
        threading.Thread(target=self._npu_check_quietly, name="gl-llm-npu-start", daemon=True).start()

    def _npu_check_quietly(self) -> None:
        try:
            self.run_npu_check()
        except Exception:
            log.exception("the NPU check failed")

    def npu_progress(self) -> Optional[dict]:
        """``{"phase", "text", "fraction"}`` while stage 1 ("checking", once it has run ``npu_quiet_seconds``),
        the NPU files' download ("downloading", a fraction) or stage 2 ("starting") runs, else None. Attributes
        only: the screens call it."""
        phase = self._npu_phase
        if phase is None and self._npu_checking and self.npu_possible():
            phase = "checking"
        if phase == "checking":
            return {"phase": phase, "text": NPU_CHECKING_TEXT, "fraction": None}
        if phase == "downloading":
            done, total = self._npu_bytes
            return {"phase": phase, "text": _npu_bytes_text(done, total),
                    "fraction": max(0.0, min(1.0, done / total)) if total else None}
        if phase == "starting":
            return {"phase": phase, "text": NPU_STARTING_TEXT, "fraction": None}
        return None

    def _npu_load_phase(self) -> Optional[str]:
        """The loading model's extra phase (``EngineSnapshot.phase``) while it fetches the files or starts the NPU."""
        progress = self.npu_progress() if self._npu_phase is not None else None
        return progress["text"] if progress else None

    def npu_status_line(self) -> Optional[str]:
        """The NPU section's one line (where the NPU is offered): why the files are not on the phone, that they
        download at the first model load, or the ready NPU ("Hexagon v81 (HTP0) ready"); the progress text while
        something runs. Saved state and a cached stat only."""
        progress = self.npu_progress()
        if progress is not None:
            return progress["text"]
        if not self.npu_offered():
            return None
        if self.npu_download_error:
            return NPU_FILES_ERROR_TEXT.format(reason=self.npu_download_error)
        if not self.npu_files_ready():
            source = self.npu_source() or {}
            size = f" ({float(source['size']) / 1e6:.0f} MB)" if source.get("size") else ""
            return f"NPU files{size} download from llama.cpp's release at the first model load."
        return f"{eng.npu_label(self.npu_state() or {})} ready"

    def npu_files_note(self, stem: Optional[str]) -> Optional[str]:
        """The model row's warning while this start's NPU files download failed and ``stem`` would run on the NPU."""
        if not stem or not self.npu_download_error or not self.npu_offered():
            return None
        if not self._npu_choice_wants(stem, self.backend_choice(stem)):
            return None
        return NPU_FILES_ERROR_TEXT.format(reason=self.npu_download_error)

    def _recover_npu(self) -> Optional[str]:
        """At start: an NPU stage the app died in (this version) keeps the NPU off; an NPU load or answer the app
        died in records that model's NPU fallback. Returns the stem of that model, else None."""
        attempt = self._pref(NPU_CHECK_ATTEMPT_KEY)
        if isinstance(attempt, dict):
            self._set_pref(NPU_CHECK_ATTEMPT_KEY, None)
            if attempt.get("version") == self.app_version:
                start = attempt.get("stage") == 2
                reason = NPU_START_CRASH_REASON if start else NPU_CHECK_CRASH_REASON
                saved = self._saved_npu(NPU_CHECK_KEY) or {}
                entry = {"version": self.app_version, "state": _npu_blocked_state(reason), "blocked": True}
                if start:
                    self._set_pref(NPU_SESSION_KEY, entry)
                else:
                    self._set_pref(NPU_CHECK_KEY, {**entry, "source": saved.get("source")})
                log.warning("on-device NPU: %s; the NPU stays off until the app version changes", reason)
        inflight = self._pref(NPU_INFLIGHT_KEY)
        if isinstance(inflight, dict) and inflight.get("background"):
            # the app was in the background during that answer: Android's memory killer or a swipe from recents
            # ends a process there too, and nothing tells those apart from a DSP crash, so it records nothing
            log.info("on-device NPU: the app closed in the background during an answer of %s; no NPU fallback",
                     inflight.get("stem"))
            inflight = None
        raw = inflight or self._pref(NPU_ATTEMPT_KEY)
        self._set_pref(NPU_ATTEMPT_KEY, None)
        if not isinstance(raw, dict) or not raw.get("stem"):
            self._set_pref(NPU_INFLIGHT_KEY, None, durable=True)
            return None
        self._set_pref(NPU_INFLIGHT_KEY, None)
        stem = str(raw["stem"])
        self._npu_failed(stem, NPU_CRASH_REASON)  # saved together with the cleared markers
        return stem

    # ---- NPU: the files and stage 2 (a loading thread, holding the gate) -------------------------------------

    def _prepare_npu(self) -> Optional[str]:
        """Before an NPU load: the phone's skeleton (downloaded once per app start when it is missing), then stage 2
        (once per process). None when the NPU is registered, else why this load does not use it (no fallback is
        recorded for the model). A Stop during the download raises ``LocalLLMError`` (499): the load ends, and
        nothing counts as a failed download (the next NPU load resumes the part)."""
        with self._npu_prepare_lock:
            session = self._npu_session_state
            if session is not None:
                failure = eng.npu_failure(session)
                return None if failure is None else f"{failure[0]}: {failure[1]}"
            failure = self.npu_failure()
            if failure is not None:
                return f"{failure[0]}: {failure[1]}"
            folder, spec, source = self.npu_skel_dir(), self.npu_skeleton(), self.npu_source()
            if not (folder and spec and source):
                return "this engine build names no NPU files for this phone"
            if not self.npu_files_ready(refresh=True):
                if self._npu_fetch_tried:
                    return NPU_FILES_ERROR_TEXT.format(reason=self.npu_download_error or "not tried again yet")
                self._npu_fetch_tried = True
                error = self._fetch_npu_files(source, folder)
                if error == NPU_FETCH_CANCELLED:
                    self._npu_fetch_tried = False  # a Stop is not a refusal of the NPU
                    raise LocalLLMError(499, "client_closed", "The request was cancelled (Stop)")
                if error is not None or not self.npu_files_ready(refresh=True):
                    self.npu_download_error = error or "the phone's file is missing from the download"
                    log.warning("on-device NPU: %s; this load uses the GPU / CPU",
                                NPU_FILES_ERROR_TEXT.format(reason=self.npu_download_error))
                    self._notify()
                    return NPU_FILES_ERROR_TEXT.format(reason=self.npu_download_error)
            self.npu_download_error = None
            _remove_other_npu_folders(folder)
            _tidy_npu_folder(folder)
            state = self._npu_session(folder)
            failure = eng.npu_failure(state)
            return None if failure is None else f"{failure[0]}: {failure[1]}"

    def _fetch_npu_files(self, source: dict, folder: str) -> Optional[str]:
        """Download ``source``'s archive (``downloads.fetch_blocking``: resumable part in ``folder``, size and
        sha256 checked, one plain anonymous GET, https redirects only, no retries) and take only its pinned
        skeletons (``downloads.extract_pinned``); the archive is deleted afterwards. An archive already in place (it
        is only renamed there once verified: the app died before extracting it) is extracted without a download.
        None when done, else why."""
        archive = os.path.join(folder, NPU_ARCHIVE_NAME)
        stops = self._stops
        self._npu_bytes = (0, int(source.get("size") or 0))
        self._npu_phase = "downloading"
        self._notify()
        last = [0.0]

        def progress(update: dict) -> None:
            if update.get("phase") != "downloading":
                return
            self._npu_bytes = (int(update.get("bytes_done") or 0), int(update.get("bytes_total") or 0))
            now = time.monotonic()
            if now - last[0] >= 0.5:
                last[0] = now
                self._notify()

        def should_stop() -> Optional[str]:
            return "cancel" if self._npu_cancel.is_set() or self._stops != stops else None

        meta = {"url": source["url"], "file": source["url"], "repo": "", "size": source["size"],
                "sha256": source["sha256"], "release": source["release"]}
        started = time.monotonic()
        fetched = False
        try:
            with _NPU_FETCH_LOCK:  # the self-check's services share the folder
                if self.npu_files_ready(refresh=True):
                    return None
                if not (os.path.isfile(archive) and dl._size(archive) == int(source["size"])):
                    session = self._npu_http if self._npu_http is not None else _anonymous_session()
                    dl.fetch_blocking(source["url"], archive, size=int(source["size"]), sha256=source["sha256"],
                                      meta=meta, progress=progress, should_stop=should_stop, session=session,
                                      backoff=(), timeout=NPU_FETCH_TIMEOUT, host="GitHub", https_only=True)
                fetched = True
                dl.extract_pinned(archive, source["files"], folder)
                dl._remove(archive)
            log.info("on-device NPU: files of llama.cpp %s downloaded in %.1f s", source["release"],
                     time.monotonic() - started)
            return None
        except dl.DownloadCancelled:
            return NPU_FETCH_CANCELLED  # the part stays: the next try resumes it
        except dl.DownloadError as exc:
            if fetched:
                dl._remove(archive)  # verified but without the pinned files: never kept
            return str(exc) or type(exc).__name__
        except Exception as exc:
            log.exception("the NPU files' download failed")
            return f"{type(exc).__name__}: {exc}"
        finally:
            self._npu_phase = None
            self._npu_files_present = None
            self._notify()

    def _npu_session(self, folder: str) -> dict:
        """Stage 2 (``register_npu_backend`` with the skeleton folder and options), watched and marked like stage 1;
        its answer is saved for this app version (a failure keeps the NPU off until the version changes)."""
        probe = self._npu_probe
        options = self.npu_options()
        self._npu_power_started = options.get(eng.NPU_POWER_ENV)
        self._npu_phase = "starting"
        self._notify()
        try:
            self._mark_npu_stage(2)
            state, blocked, seconds = self._watched("gl-llm-npu-session", lambda: probe(folder, options))
            failure = eng.npu_failure(state)
            log.info("on-device NPU start in %.1f s: %s", seconds,
                     f"{eng.npu_label(state)} ready" if failure is None else f"not used ({failure[0]}: {failure[1]})")
            if failure is not None and failure[0] == "skeletons" and eng.NPU_FILES_MISSING in failure[1]:
                # nothing was registered: the files, not the NPU (a later start downloads them again)
                self.npu_download_error = failure[1]
                self._npu_files_present = False
            else:
                self._set_pref(NPU_SESSION_KEY, {"version": self.app_version, "state": state, "blocked": blocked,
                                                 "seconds": seconds})
                self._npu_session_state = state
            self._set_pref(NPU_CHECK_ATTEMPT_KEY, None, durable=True)
            return state
        finally:
            self._npu_phase = None
            self._notify()

    def use_npu(self) -> bool:
        """"Use NPU": on unless switched off (it only matters where the NPU check passed)."""
        raw = self._pref(NPU_KEY)
        return raw is None or bool(raw)

    def save_use_npu(self, enabled: Any) -> bool:
        """Switch the NPU on or off for "Auto" models; an idle loaded model that would now run elsewhere is
        freed, so the next request loads it on the new device."""
        value = bool(enabled)
        changed = value != self.use_npu()
        self._set_pref(NPU_KEY, value)
        stem = self.loaded_stem
        if changed and stem is not None and (self._loaded_npu or self.npu_offered()) \
                and self.backend_choice(stem) == "auto":
            self.unload()  # busy: the next load after the answer uses the new choice
        self._notify()
        return value

    def npu_power(self) -> str:
        raw = str(self._pref(NPU_POWER_KEY) or "").strip()
        return raw if raw in eng.NPU_POWER_MODES else eng.NPU_DEFAULT_POWER

    def save_npu_power(self, mode: Any) -> str:
        """The power mode for the next time the backend registers (after a restart when it already has)."""
        value = str(mode or "").strip()
        if value not in eng.NPU_POWER_MODES:
            return self.npu_power()
        self._set_pref(NPU_POWER_KEY, value)
        self._notify()
        return value

    def npu_power_pending(self) -> bool:
        """The saved power mode differs from the one the registered backend started with (needs a restart)."""
        if self._npu_session_state is None or not eng.npu_passed(self._npu_session_state):
            return False
        return getattr(self, "_npu_power_started", self.npu_power()) != self.npu_power()

    def backend_choice(self, stem: str) -> str:
        """The model's backend: its preset row's, "auto" for every other model."""
        return presets.backend_of(self.preset_for(stem))

    def npu_layers(self, stem: str) -> int:
        """The preset row's NPU layers, -1 (all) by default."""
        try:
            value = int((self.preset_for(stem) or {}).get("npu_layers") or 0)
        except (TypeError, ValueError):
            value = 0
        return value if value > 0 else -1

    def _npu_fallbacks(self) -> dict:
        raw = self._pref(NPU_FALLBACK_KEY)
        return dict(raw) if isinstance(raw, dict) else {}

    def npu_fallback(self, stem: Optional[str]) -> Optional[dict]:
        """``{"stem", "reason"}`` while ``stem`` stays off the NPU: until its file, its backend choice or the app
        version changes, or "Try the NPU again"."""
        if not stem:
            return None
        entry = self._npu_fallbacks().get(stem)
        if not isinstance(entry, dict):
            return None
        file = (self.registry.load().get(stem) or {}).get("file")
        if entry.get("file") != file or entry.get("backend") != self.backend_choice(stem) \
                or entry.get("version") != self.app_version:
            return None
        return {"stem": stem, "reason": str(entry.get("reason") or "unknown reason")}

    def _npu_failed(self, stem: str, reason: str) -> None:
        """Keep ``stem`` off the NPU (see ``npu_fallback``) and say why."""
        reason = " ".join(str(reason or "unknown reason").split())[:300]
        log.warning("on-device NPU unavailable for %s: %s; using the %s", stem, reason,
                    "GPU" if self._gpu_wanted(stem) else "CPU")
        fallbacks = self._npu_fallbacks()
        fallbacks[stem] = {"reason": reason, "file": (self.registry.load().get(stem) or {}).get("file"),
                           "backend": self.backend_choice(stem), "version": self.app_version, "at": int(time.time())}
        self._set_pref(NPU_FALLBACK_KEY, fallbacks, durable=True)
        self._clear_npu_attempt()

    def retry_npu(self, stem: str) -> bool:
        """"Try the NPU again": forget the model's NPU fallback, and a failed NPU files download of this start (the
        next NPU load downloads once more); an idle loaded model reloads on its next request."""
        fallbacks = self._npu_fallbacks()
        files = bool(self.npu_files_note(stem))
        if files:
            self._npu_fetch_tried = False
            self.npu_download_error = None
        if fallbacks.pop(stem, None) is None and not files:
            return False
        self._set_pref(NPU_FALLBACK_KEY, fallbacks)
        if self._npu_hung_stem == stem:
            self.npu_hung = False
            self._npu_hung_stem = None
        if self.loaded_stem == stem and not self._loaded_npu:
            self.unload()
        self._notify()
        return True

    def npu_note(self, stem: Optional[str] = None) -> Optional[str]:
        """The NPU line of the engine card and the model's row: the restart message for the model whose first
        NPU load hung in this process, else why ``stem`` (default: the loaded or loading model) runs elsewhere."""
        stem = stem or self._stem
        fallback = self.npu_fallback(stem) if stem else None
        target = "GPU" if stem and self._gpu_wanted(stem) else "CPU"
        if self.npu_hung and stem and stem == self._npu_hung_stem:
            return f"The NPU did not answer. Restart Glossarion; this model will use the {target}."
        if fallback is None:
            return None
        return f"NPU unavailable for {stem}: {fallback['reason']}. Using {target}."

    def _write_npu_inflight(self, stem: Any = "") -> None:
        """Set (a stem) or clear (None) the in-flight marker of the NPU answer running now, or rewrite it ("",
        ``on_lifecycle``), on disk before returning; ``background``: the app is in the background now."""
        with self._npu_marker_lock:
            if stem != "":
                self._npu_inflight = stem
            elif self._npu_inflight is None:
                return
            current = self._npu_inflight
            self._set_pref(NPU_INFLIGHT_KEY, None if current is None else {
                "stem": current, "at": int(time.time()), "background": self.background_since is not None},
                durable=True)

    def npu_hang_note(self) -> Optional[str]:
        """The diagnostics line after a hang in this process: the hung model's restart message, or for a hung
        check that the NPU stays off until the app version changes (a restart does not run the check again)."""
        if not self.npu_hung:
            return None
        if self._npu_hung_stem:
            return self.npu_note(self._npu_hung_stem)
        return NPU_CHECK_HANG_NOTE

    def _mark_npu_attempt(self, stem: str) -> None:
        self._npu_attempt = stem
        self._set_pref(NPU_ATTEMPT_KEY, {"stem": stem, "at": int(time.time())}, durable=True)

    def _clear_npu_attempt(self) -> None:
        # only this service's own marker: the parallel secondary and the self-check services share the key
        if self._npu_attempt is not None:
            self._npu_attempt = None
            self._set_pref(NPU_ATTEMPT_KEY, None, durable=True)

    def _npu_choice_wants(self, stem: str, choice: str) -> bool:
        """``stem``'s own rule asks for the NPU: its choice is "npu", or "auto" with "Use NPU" on and a quant the
        backend accepts, and the model has no NPU fallback (prefs only)."""
        if self._is_child or self._npu_probe is None or choice not in ("auto", "npu"):
            return False
        if choice == "auto":
            if not self.use_npu():
                return False
            if presets.npu_quant_kind((self.registry.load().get(stem) or {}).get("file")) == "cpu":
                return False
        return not self.npu_fallback(stem)

    def _npu_wanted(self, stem: str, choice: str) -> bool:
        """The NPU for ``stem`` now: its own rule asks for it, the NPU is offered (stage 1 runs here first when the
        start's background check has not answered yet) and this start's NPU files download has not failed."""
        if not self._npu_choice_wants(stem, choice):
            return False
        if self._stage1() is None:
            self.run_npu_check()  # a load that comes before the start's background check waits for it
        return self.npu_offered() and not (self._npu_fetch_tried and self.npu_download_error)

    def _placement(self, stem: str) -> tuple:
        """``(backend, layers)`` of the next load of ``stem``: ("npu", NPU layers), ("gpu", GPU layers) or
        ("cpu", 0)."""
        choice = self.backend_override or self.backend_choice(stem)
        if choice == "cpu":
            return "cpu", 0
        if self._npu_wanted(stem, choice):
            return "npu", self.npu_layers(stem)
        if choice == "gpu":
            fallback = self.gpu_fallback()
            layers = self.gpu_layers(stem)
            if self.gpu_backend and layers != 0 and not (fallback and fallback["stem"] == stem):
                return "gpu", layers
            return "cpu", 0
        if self._gpu_wanted(stem):
            return "gpu", self.gpu_layers(stem)
        return "cpu", 0

    def _load_on_npu(self, engine: Any, stem: str, path: str, n_ctx: int, mmproj: Optional[str], layers: int) -> bool:
        """One NPU load (watched by the attempt marker until it returns); True when the model runs on the NPU.
        Any failure, or a first load still running after ``npu_watchdog_seconds``, records the NPU fallback."""
        self._mark_npu_attempt(stem)
        fired = threading.Event()
        timer = None
        if not self._npu_loaded_once:
            def hang() -> None:
                fired.set()
                self.npu_hung = True
                self._npu_hung_stem = stem
                self._npu_failed(stem, NPU_HANG_REASON)
                self._notify()

            timer = threading.Timer(self.npu_watchdog_seconds, hang)
            timer.daemon = True
            timer.start()
        try:
            engine.load(path, n_ctx, self.threads(), mmproj=mmproj, npu=True, n_gpu_layers=layers)
            if fired.is_set():
                raise eng.NpuUnavailable(NPU_HANG_REASON)
            if not getattr(engine, "on_npu", False):
                raise eng.NpuUnavailable("the engine did not load the model on the NPU")
            self._npu_loaded_once = True
            # the load ran its first token on the DSP (``engine.load``'s check); from here every NPU answer has
            # its own in-flight marker, so an idle loaded model leaves no marker behind
            self._clear_npu_attempt()
            return True
        except Exception as exc:  # NpuUnavailable, or anything else the NPU path raised
            if timer is not None:
                timer.cancel()
            if engine.loaded:
                engine.unload()
            if not fired.is_set():
                self._npu_failed(stem, str(exc) or type(exc).__name__)
            self._clear_npu_attempt()
            return False
        finally:
            if timer is not None:
                timer.cancel()

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

    def thread_limit(self) -> int:
        return eng.available_cpus()

    def threads(self) -> int:
        return min(self.thread_limit(), self.settings()["n_threads"] or eng.default_threads())

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
        backend, layers = self._placement(stem)
        npu = backend == "npu"
        gpu = backend == "gpu"
        gpu_layers = layers if gpu or npu else 0
        # a changed Context (Model manager or Welcome slider), GPU / NPU setting or backend reloads the model on the
        # next request
        if self._stem == stem and engine.loaded and (not engine.n_ctx or engine.n_ctx == n_ctx) \
                and self._loaded_gpu == gpu and self._loaded_npu == npu and self._loaded_gpu_layers == gpu_layers \
                and engine.n_threads == self.threads():
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
                 ", NPU" if npu else f", GPU {self.gpu_backend}" if gpu else "")
        started = time.monotonic()
        mmproj = self.registry.projector_path(stem)
        if npu:
            # the NPU files (downloaded at the first NPU load) and stage 2; while they run the loading state says so
            # (``EngineSnapshot.phase``). Neither keeps the model from loading: today's GPU / CPU rule, no fallback;
            # only a Stop during the download ends the load.
            try:
                reason = self._prepare_npu()
            except LocalLLMError:
                self._set("idle", None)
                raise
            except Exception as exc:  # pragma: no cover - the stages never raise
                log.exception("preparing the NPU failed")
                reason = f"{type(exc).__name__}: {exc}"
            if reason is not None:
                log.info("on-device model %s: not on the NPU this time (%s)", stem, reason)
                npu = False
                gpu = self._gpu_wanted(stem)
                gpu_layers = self.gpu_layers(stem) if gpu else 0
        try:
            if npu and not self._load_on_npu(engine, stem, path, n_ctx, mmproj, gpu_layers):
                # the NPU fallback is recorded: today's GPU / CPU rule for this load
                npu = False
                gpu = self._gpu_wanted(stem)
                gpu_layers = self.gpu_layers(stem) if gpu else 0
            if npu:
                pass  # loaded on the NPU
            elif gpu:
                self._mark_gpu_attempt(stem)
                try:
                    engine.load(path, n_ctx, self.threads(), mmproj=mmproj, gpu=True, n_gpu_layers=gpu_layers)
                except Exception as exc:  # GpuUnavailable, or anything else the GPU path raised
                    self._gpu_failed(stem, str(exc) or type(exc).__name__)
                    engine.load(path, n_ctx, self.threads(), mmproj=mmproj)
            else:
                engine.load(path, n_ctx, self.threads(), mmproj=mmproj)
        except Exception as exc:
            self._clear_gpu_attempt()
            self._clear_npu_attempt()
            self._loaded_gpu = False
            self._loaded_npu = False
            self._set("idle", None, f"Could not load {stem}: {exc}")
            log.exception("loading %s failed", stem)
            raise LocalLLMError(500, "load_failed", f"Could not load local/{stem}: {exc}", "server_error") from exc
        self._loaded_gpu = bool(getattr(engine, "on_gpu", False))  # record what loaded, not a setting changed during load
        self._loaded_npu = bool(getattr(engine, "on_npu", False))
        self._loaded_gpu_layers = gpu_layers if self._loaded_gpu or self._loaded_npu else 0
        self._load_seconds = round(time.monotonic() - started, 2)
        self.log.append((stem, self._load_seconds))
        self.last_load = {"stem": stem, "backend": self._engine_backend() or "CPU",
                          "device": getattr(engine, "device", None) or "CPU", "seconds": self._load_seconds}
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
        self._loaded_npu = False
        self._clear_gpu_attempt()  # unloaded normally: the GPU did not take the app down
        self._clear_npu_attempt()  # nor the NPU
        self._set("idle", None)

    def stop_generation(self) -> None:
        """An immediate Stop (``JobBackend.request_stop``): every request that has arrived so far ends,
        whatever happens to the client's stop flags and connections from here on. The one generating lets
        go at its next stop check; nothing here touches the engine."""
        with self._lock:
            self._stops += 1
        if self._parallel is not None:
            self._parallel.stop_generation()

    def stopping(self) -> bool:
        """A generation that was told to stop, or one under any job Stop, still holds the engine."""
        holder = self._holder_stops
        return self.gate.busy and ((holder is not None and holder != self._stops) or bool(self.stop_state()))

    def orphaned(self) -> bool:
        """The generation holding the engine came from an on-device job that is over for the user: that job
        pinned this model when the answer arrived, and no running job pins it any more (it ended, was force
        stopped, or the next job runs a cloud model or none; a job's model is fixed, so this needs no stop flag
        the next run clears). Nobody reads that answer (a graceful Stop's answer runs to its end, minutes on a
        phone): an explicit switch ends it (``end_orphan``). An answer that arrived with no on-device job behind
        it (endpoint test, self-check, a chat during a cloud job) is never orphaned."""
        return bool(self.gate.busy and self._holder_job and self._holder_token is not None
                    and self._pinned_stem() is None)

    def end_orphan(self) -> bool:
        """Tell an ``orphaned`` generation to end at its next stop check (only that one: the requests queued
        behind it and the Stop counter are left alone; native code is never interrupted). True when it did."""
        token = self._holder_token
        if token is None or self._cut is token or not self.orphaned():
            return False
        self._cut = token
        log.info("on-device model %s: ending the answer of a stopped job for a model switch", self._stem)
        return True

    def load_must_wait(self) -> bool:
        """``load_blocking`` would wait for the engine: something holds it, and it is not the live answer of
        a running job on an on-device model (that one refuses)."""
        return bool(self.gate.busy or self.gate.waiting()) and (self._pinned_stem() is None or self.stopping())

    def _wait_for_engine(self) -> None:
        """Take the gate behind its holder (FIFO), ending the holder's answer as soon as it is ``orphaned``
        (also when its job ends during the wait). Raises ``LocalLLMError`` (503) on a full queue."""
        def keep_waiting() -> bool:
            self.end_orphan()  # a job that ended during the wait left its answer to nobody
            return False

        self.end_orphan()
        self.gate.acquire(keep_waiting, on_wait=self._notify)

    def load_blocking(self, stem: str) -> tuple:
        """Explicit Load (Model manager, Welcome): ``(True, None)`` or ``(False, message)``. Refused during
        a running job's live answer; an answer that was stopped or has no job behind it is waited for (it
        cannot be cut off under native code), then the model loads by itself. The answer of a job that is
        over (``orphaned``) is told to end first, so the wait is a few tokens, not the rest of that answer."""
        pool = self._parallel
        secondary = pool.secondary if pool is not None else None
        if secondary is not None and secondary.gate.busy:
            if not (secondary.stopping() or secondary.orphaned()):
                return False, "The on-device model is busy; try again when the current answers are done"
            secondary.stop_generation()  # a stopped answer in the extra context ends at its next stop check
            log.info("on-device model %s: waiting for the extra context's answer to end", stem)
            try:
                secondary.gate.acquire(on_wait=self._notify)
            except LocalLLMError as exc:
                return False, exc.message
            try:
                secondary._unload_locked()
            finally:
                secondary.gate.release()
        if not self.gate.try_acquire():
            if not self.load_must_wait():
                return False, BUSY_FOR_JOB
            log.info("on-device model %s: waiting for the current answer to end", stem)
            try:
                self._wait_for_engine()
            except LocalLLMError as exc:
                return False, exc.message
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
        if force:
            self.stop_generation()
        extra = self._parallel.unload(force=force) if self._parallel is not None else False
        if self.engine is None or not self.engine.loaded:
            return extra
        if not self.gate.try_acquire():
            if not force:
                return False
            self.stop_generation()
            if not self.gate.take_over(FORCE_UNLOAD_WAIT):
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
        if self._parallel is not None and self._parallel.secondary is not None:
            self._parallel.secondary.selected_model_changed(model)
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
        self.end_orphan()  # a stopped job's answer nobody reads: the unload follows its next stop check
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
        # a job being force stopped keeps nothing loaded: its requests are over, the switch may go ahead
        if self.active_job_model is None or self.stop_state() == "force":
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
        if self._parallel is not None:
            return self._parallel.chat(model, body, should_stop)
        return self._chat_serial(model, body, should_stop)

    def _chat_serial(self, model: str, body: dict, should_stop: Callable[[], bool]) -> ChatRun:
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
        arrived = self._stops
        under_job = bool(pinned)  # a running job pins this on-device model (any other pin was refused above)
        token = object()

        def stop_waiting() -> bool:
            # a queued request gives up on any Stop (graceful lets only the answer being generated finish)
            return bool(self._stops != arrived or should_stop() or stop_state())

        def stop_generating() -> bool:
            # ``self._cut``: a model switch ended this answer of a job that is over (``end_orphan``)
            return bool(self._stops != arrived or should_stop() or stop_state() == "force" or self._cut is token)

        if stop_waiting():
            raise LocalLLMError(499, "client_closed", "The request was cancelled (Stop)")
        if not self.gate.acquire(stop_waiting, on_wait=self._notify):
            raise LocalLLMError(499, "client_closed", "The request was cancelled while waiting")
        self._holder_stops = arrived
        self._holder_job = under_job
        self._holder_token = token
        released = [False]
        inflight = [False]  # this answer set the NPU in-flight marker (cleared in ``release``)

        def mark_inflight(engine: Any) -> None:
            if getattr(engine, "on_npu", False) and not inflight[0]:
                inflight[0] = True
                self._write_npu_inflight(stem)

        def release() -> None:
            if released[0]:
                return
            released[0] = True
            try:
                self.last_used = self.clock()
                if inflight[0]:
                    inflight[0] = False
                    self._write_npu_inflight(None)  # the NPU answer ended without taking the app down
                if self._gpu_attempt is not None:
                    self._clear_gpu_attempt()  # the GPU's first answer ended without taking the app down
                if self._npu_attempt is not None:
                    self._clear_npu_attempt()
                if self._state == "generating":
                    self._set("ready", self._stem)
            finally:  # whatever failed above, the engine goes back to the next request
                self._holder_stops = None
                self._holder_job = False
                self._holder_token = None
                if self._cut is token:
                    self._cut = None
                self.gate.release()
            self._notify()
            if self._unload_when_idle:
                self._deferred_unload()

        try:
            self._ensure_loaded(stem, vision=bool(images))
            if stop_generating():  # a Stop during the load (native, it ran to its end): no prompt evaluation
                raise LocalLLMError(499, "client_closed", "The request was cancelled (Stop)")
            engine = self.engine
            n_ctx = int(engine.n_ctx or self.context_for(stem))
            prompt_tokens = int(engine.count_prompt(messages))
            if prompt_tokens >= n_ctx:
                raise LocalLLMError(400, "context_length_exceeded",
                                    f"This model's maximum context length is {n_ctx} tokens. However, your messages "
                                    f"resulted in {prompt_tokens} tokens. Please reduce the length of the messages.")
            params = self._params(body, n_ctx - prompt_tokens, stem=stem)
            self._set("generating", stem)
            mark_inflight(engine)
            try:
                deltas = _primed(engine.chat(messages, params, stop_generating), n_ctx)
            except (LocalLLMError, KeyboardInterrupt, SystemExit):
                raise
            except Exception as exc:
                if getattr(engine, "on_npu", False):
                    # the NPU failed on this prompt: the same request again on the GPU / CPU (once)
                    self._npu_failed(stem, f"the NPU failed while answering ({exc})")
                elif getattr(engine, "on_gpu", False):
                    # the GPU failed on this prompt: the same request again on the CPU
                    self._gpu_failed(stem, f"the GPU failed while answering ({exc})")
                else:
                    raise
                self._ensure_loaded(stem, vision=bool(images))
                engine = self.engine
                self._set("generating", stem)
                deltas = _primed(engine.chat(messages, params, stop_generating), n_ctx)
        except BaseException:
            release()
            raise
        return ChatRun(stem, deltas, usage=lambda: getattr(engine, "last_usage", {}),
                       finish=lambda: getattr(engine, "last_finish", "stop"), release=release)

    def _params(self, body: dict, room: int, *, stem: Optional[str] = None) -> dict:
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
        params.update(self.sampling(stem))
        stop = body.get("stop")
        if isinstance(stop, str):
            stop = [stop]
        if isinstance(stop, list) and stop:
            params["stop"] = [str(s) for s in stop][:8]
        params["enable_thinking"] = wants_thinking(body)
        if self.forced_truncation(stem):
            params["forced_truncation_seconds"] = self.forced_truncation_seconds()
            if self.mark_glossary_complete(stem) and "glossary" in str(body.get("glossarion_task") or "").lower():
                params["forced_truncation_finish"] = "stop"
        scripts = script_ban.banned_scripts(body.get("glossarion_task"), body.get("target_language"),
                                            single_pass=body.get("glossarion_single_pass"), enabled=self.block_cjk(stem))
        always = script_ban.source_whitelisted(body.get("glossarion_task"), single_pass=body.get("glossarion_single_pass"))
        whitelist = always or script_ban._truthy(body.get("glossarion_whitelist_source"))
        if whitelist and any(isinstance(m, dict) and isinstance(m.get("content"), list)
                             and any(isinstance(p, dict) and p.get("type") != "text" for p in m["content"])
                             for m in body.get("messages") or []):
            # an image: no text to tell the source's characters by
            whitelist = False
            if always:
                scripts = ()
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

    def delete_blocker(self, stem: str) -> Optional[str]:
        """Why ``stem`` cannot be deleted now, else None: a running job uses it (also between two of its
        requests), or it is loading or answering (``busy_with``). Loaded and idle is no reason: ``delete``
        unloads it first."""
        pinned = self._pinned_stem()
        if pinned and pinned.casefold() == stem.casefold():
            return f"local/{stem} is in use by a running job: stop the job, then delete it"
        if self.busy_with(stem):
            return f"local/{stem} is loading or answering a request: delete it when that is done"
        return None

    def delete(self, stem: str) -> bool:
        """Delete a model and everything of it on the phone: the file, its vision projector and the unfinished
        parts of both (a paused or failed download; a running one is cancelled). Unloads it first. False
        without deleting anything while ``delete_blocker(stem)`` names a reason: its file must not go away
        under native code or under a running job."""
        if self.delete_blocker(stem):
            return False
        if self._stem == stem and self.engine is not None and self.engine.loaded and not self.unload():
            return False
        removed = self.registry.remove(stem)
        for key in (stem, self.registry.projector_key(stem)):  # running / paused / failed parts
            removed = self.downloads.cancel(key) or removed
        self.request_catalog_refresh()
        return removed

    def busy_with(self, stem: str) -> bool:
        """True while ``stem`` is loading or generating (a delete must wait)."""
        return bool(self._parallel and self._parallel.busy_with(stem)) or (
            self._stem == stem and (self.gate.busy or self._state in ("loading", "generating")))

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
        if self._parallel is not None and self._parallel.secondary is not None:
            self._parallel.secondary.on_lifecycle(name)
        name = str(name or "").lower()
        if name in ("hide", "pause", "inactive"):
            if name == "inactive":
                return None
            if self.background_since is None:
                self.background_since = self.clock()
                self._write_npu_inflight()  # a kill from here on is not counted as an NPU crash
            if self.platform == "ios" and not self.job_active():
                return "unloaded" if self.unload() else None
            return None
        if name in ("show", "resume", "restart"):
            if self.background_since is not None:
                self.background_since = None
                self._write_npu_inflight()
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
    except eng.GenerationStopped as exc:
        deltas.close()
        raise LocalLLMError(499, "client_closed", "The request was cancelled (Stop)") from exc
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
