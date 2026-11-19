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
A GPU load offloads the configured layers (eight by default, one device), then evaluates a short prompt and
one more token and checks the logits are finite; when the backend has no device, the model does not
load or that first token fails, it raises ``GpuUnavailable`` (the service reloads on the CPU). A CPU
load uses ``split_mode`` NONE with ``main_gpu`` -1: no offload device at all, even after a GPU backend
was registered earlier in the process. ``device`` names what runs the model. The vision projector
always stays on the CPU.

NPU (Qualcomm Hexagon, ``load(..., npu=True)``; the service decides, per model): Android arm64 builds whose
llama-cpp-python wheel carries libggml-hexagon (``npu_backend_shipped``). Two stages: ``npu_precheck()`` (stage 1,
cheap: the build carries the backend, the phone's libcdsprpc.so opens, the cDSP reports its HTP arch; nothing is
loaded or registered) and ``npu_state(skel_dir, options)`` (stage 2: the wheel's ``register_npu_backend``, which
needs the phone's HTP skeleton ``libggml-htp-v<arch>.so`` already in ``skel_dir``, sets the GGML_HEXAGON_* options,
loads the backend by soname and opens and closes a real session) and keeps its answer; ``npu_passed`` says
whether every step passed. The APK carries no skeletons (the Hexagon SDK licence): ``npu_skeleton_source()`` names
the llama.cpp release archive of the same commit that holds them, which the service downloads at the first NPU
load. A wheel without these functions (older builds, iOS, x86_64) gives a clean "not shipped" state and the NPU
is simply not offered. The device is chosen with ``main_gpu`` =
``device_index("HTP")``; a GPU load uses ``device_index`` of its own backend too, so a GPU load never lands on
the NPU or the reverse (a wheel without ``device_index`` keeps ``main_gpu`` 0 for the GPU). An NPU load runs
the same first-token check as the GPU (``_gpu_check``) and raises ``NpuUnavailable`` on any failure. The
"ggml-hex:" lines of the native log are kept (``hex_log_lines``) for the diagnostics.

NPU + GPU together (``load(..., npu=True, split="layer" | "tensor")``; the service's "Run on" modes "NPU + GPU"
(layer) and "NPU + GPU, tensor split", and the device checks' timing): the model on exactly two devices, the NPU first
and the GPU second, half of it each
(``tensor_split`` 1:1), all layers offloaded. llama-cpp-python 0.3.37's ``Llama()`` has ``split_mode``,
``main_gpu``, ``tensor_split``, ``flash_attn``, ``type_k`` / ``type_v`` but no device list, and builds its
``llama_model_params`` (llama.py: ``llama_model_default_params()``, then the load in the same ``__init__``); the
struct's ``devices`` field (a NULL-terminated ``ggml_backend_dev_t`` list) is filled by ``_model_devices``, which
wraps ``llama_model_default_params`` for the loading thread only, under ``_NATIVE_LOAD_LOCK``. With a list, llama.cpp
uses those devices in that order (``llama_prepare_model_devices``); ``main_gpu`` only matters for split mode NONE.
"layer": each device holds a slice of the layers (split mode LAYER). "tensor": every layer split across both (split
mode TENSOR, experimental upstream); it needs flash attention on (every load already has it) and an unquantised KV
cache, so this load alone passes ``type_k`` / ``type_v`` f16. A refusal (the architecture, flash attention, memory, a
device missing) raises ``SplitUnavailable`` with llama.cpp's own error line (``native_errors``).

NPU reads, CPU writes (``load(..., npu=True, handoff=True)``; the device checks' timing only): two copies of the model,
``reader`` on the NPU (every layer, split mode NONE on the HTP) and ``llm`` on the CPU, with the same context settings.
Each answer's prompt (the exact tokens ``Llama.generate`` gets, through a wrapper set on the CPU model for that
``chat`` call only) is read by the NPU up to its last token but one, the sequence's memory is copied to the CPU
context (``llama_state_seq_get_data`` / ``llama_state_seq_set_data``, sequence 0; never ``Llama.save_state``, which
copies the scores), and the CPU model reads the last token itself and writes the answer: llama-cpp-python replays one
token after a restored state, and a hybrid model's recurrent part cannot roll back a restored last token. The load runs
the same hand-off once on ``GPU_CHECK_PROMPT`` (a refused state, a failed copy or bad logits raise
``SplitUnavailable``); it is also the warm-up of both models. ``last_timing`` gains ``handoff_s`` and
``handoff_bytes``; no projector (no vision in this mode).

Native aborts (``capture_aborts``): a GGML_ABORT / GGML_ASSERT ends the process and Android keeps no stderr, so before
the first native work (``npu_precheck``, ``npu_state``, every ``LlamaEngine.load``) ggml's abort callback is registered
once per process (``ggml_set_abort_callback`` of the loader's libggml); it writes ggml's "file:line: message", the
time, the process id and ``RUN_ID`` to ``native_abort.txt`` in the app's data folder and returns (ggml aborts right
after). The service's crash recovery attaches it (``with_abort_reason``) to the reason of a marker of the same run.

Script ban (``params["ban_scripts"]``, the ``script_ban.SCRIPTS`` names decided by the service): ``LlamaEngine``
passes a logits processor that sets the token ids of those scripts (Han / kana / Hangul, Devanagari,
Thai, Arabic, Cyrillic ...) to -inf. The vocabulary is decoded once, on the first such request after a
load (``script_ban.VocabScripts``; logged), and one processor per set of banned scripts is kept until the
model is unloaded; a request without ``ban_scripts`` gets no processor. ``params["ban_allow_kaomoji"]``
(the opt-in emoticon exemption) unbans the emoticon punctuation and the emoticon characters of the
request's user messages.
"""

from __future__ import annotations

import collections
import contextlib
import importlib
import importlib.util
import logging
import os
import re
import threading
import time
from typing import Any, Callable, Iterator, Optional, Sequence

log = logging.getLogger("glossarion.local_llm")
#: the progress and thread lines (hotfix 2). Hotfix 4: the diagnostics log only, no longer a child of
#: ``unified_api_client`` (a job's Log); the job screen shows the live row beside its Log title instead
progress_log = logging.getLogger("glossarion.local_llm.progress")

__all__ = ["ENGINE_VERSION", "ContextOverflow", "EngineUnavailable", "FakeEngine", "GenerationStopped",
           "GpuUnavailable", "LlamaEngine", "NpuUnavailable", "SPLITS", "SplitUnavailable", "cpu_variant",
           "default_threads", "device_index",
           "engine_status", "gpu_offer", "gpu_state", "hex_log_lines", "npu_failure", "npu_passed", "npu_platform",
           "npu_precheck", "npu_precheck_failure", "npu_skeleton_file", "npu_skeleton_source", "npu_state",
           "system_info", "vision_status"]

#: The llama-cpp-python pin of ``src/mobile/pyproject.toml`` (``[tool.flet.android/ios].dependencies``).
ENGINE_VERSION = "0.3.37"
#: Qwen3.5 non-thinking sampling defaults (used when a request leaves them out).
DEFAULT_SAMPLING = {"top_k": 20, "top_p": 0.8, "min_p": 0.0}
DEFAULT_GPU_LAYERS = 8
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


class NpuUnavailable(RuntimeError):
    """The NPU path failed (backend not registered, the model did not load on it, or its first token failed);
    the message says why. The engine holds no model afterwards."""


class SplitUnavailable(RuntimeError):
    """The NPU + GPU load failed (a device missing, llama.cpp refused the split, or its first token failed); the
    message says why. The engine holds no model afterwards."""


#: llama.cpp's LLAMA_SPLIT_MODE_NONE: the whole model on ``main_gpu`` (-1: on no device = the CPU).
SPLIT_MODE_NONE = 0
#: llama.h's LLAMA_SPLIT_MODE_LAYER and LLAMA_SPLIT_MODE_TENSOR: the NPU + GPU loads (``load(split=...)``)
SPLITS = {"layer": 1, "tensor": 3}
#: ggml's GGML_TYPE_F16: the unquantised KV cache a tensor split needs
GGML_TYPE_F16 = 1
#: ggml's GGML_LOG_LEVEL_ERROR: the native log lines kept for a refused load's reason
NATIVE_ERROR_LEVEL = 4
NATIVE_ERROR_LINES = 5
_NATIVE_ERRORS: collections.deque = collections.deque(maxlen=NATIVE_ERROR_LINES)
#: the registry name of llama.cpp's Hexagon backend (ggml-hexagon at the pinned commit)
NPU_BACKEND = "HTP"
#: the oldest HTP arch the backend has skeletons for; a lower or unknown arch is never offered (upstream would
#: silently assume v73 when its own query fails)
NPU_MIN_ARCH = 73
#: the steps of ``register_npu_backend``, in order
NPU_STEPS = ("shipped", "libcdsprpc", "arch", "skeletons", "registry", "session")
#: the steps of ``npu_precheck`` (stage 1), the cheap ones before the backend needs its skeleton
NPU_PRECHECK_STEPS = NPU_STEPS[:3]
#: ``register_npu_backend``'s "skeletons" detail when the phone's skeleton is not in its folder
NPU_FILES_MISSING = "NPU files not downloaded"
#: GGML_HEXAGON_POWER_MODE and the modes ggml-hexagon.cpp documents at the pin (its default: burst)
NPU_POWER_ENV = "GGML_HEXAGON_POWER_MODE"
NPU_POWER_MODES = ("low_power_saver", "power_saver", "high_power_saver", "low_balanced", "balanced",
                   "high_performance", "sustained_high_performance", "burst")
NPU_DEFAULT_POWER = "burst"
NO_NPU = "this engine build has no NPU support"
#: how many "ggml-hex:" log lines the diagnostics keep
HEX_LOG_LINES = 20
_HEX_LOG: collections.deque = collections.deque(maxlen=HEX_LOG_LINES)
_HEX_PENDING: list = [""]
_HEX_HOOK: list = []  # the installed log callback (kept alive: ctypes calls it from native code)
_NPU_STATE: dict = {}  # the last ``npu_state`` answer of this process


def npu_platform(platform: Optional[str] = None, machine: Optional[str] = None) -> bool:
    """Android arm64: the only builds that can carry the Hexagon backend (nothing is imported)."""
    name = (platform or os.environ.get("GLOSSARION_PLATFORM") or "").strip().lower()
    if name != "android":
        return False
    import platform as _platform

    arch = (machine if machine is not None else _platform.machine()).strip().lower()
    return arch in ("aarch64", "arm64", "arm64-v8a")


def _extensions() -> Any:
    try:
        return importlib.import_module("llama_cpp._ctypes_extensions")
    except Exception:
        return None


def _npu_missing(step: str, detail: str) -> dict:
    return {"backend": None, "device": "", "arch": None, "error": detail,
            "steps": [{"name": step, "ok": False, "detail": detail}]}


def _arch_number(value: Any) -> Optional[int]:
    """81 of an HTP arch given as 81, "81" or "v81"; None when unknown."""
    if value is None or isinstance(value, bool):
        return None
    text = str(value).strip().lower().removeprefix("v")
    try:
        return int(text)
    except ValueError:
        return None


def _steps(raw: Any) -> list:
    return [{"name": str(s.get("name") or "?"), "ok": bool(s.get("ok")), "detail": str(s.get("detail") or "")}
            for s in (raw or []) if isinstance(s, dict)]


def _npu_functions(*names: str) -> tuple:
    """``(functions, None)`` of the wheel's NPU functions, or ``(None, failed state)``."""
    ext = _extensions()
    if ext is None:
        return None, _npu_missing("shipped", UNAVAILABLE)
    functions = [getattr(ext, name, None) for name in names]
    if any(function is None for function in functions):
        return None, _npu_missing("shipped", NO_NPU)
    return functions, None


def npu_precheck() -> dict:
    """Stage 1, ``npu_precheck()`` of our build (cheap, repeatable, never raises; nothing is loaded or registered):
    ``{"backend": None, "device": "", "arch": 81 | None, "error", "steps": [shipped, libcdsprpc, arch]}``. A build
    without the function answers a failed "shipped" step."""
    functions, missing = _npu_functions("npu_precheck")
    if missing is not None:
        return missing
    capture_aborts()
    try:
        raw = functions[0]()
    except Exception as exc:  # pragma: no cover - the wheel's function never raises
        return _npu_missing("shipped", f"{type(exc).__name__}: {exc}")
    return npu_precheck_state(raw)


def npu_precheck_state(raw: Any) -> dict:
    """``npu_precheck``'s answer in the service's shape (arch 81 for "v81")."""
    state = dict(raw) if isinstance(raw, dict) else {}
    return {"backend": None, "device": "", "arch": _arch_number(state.get("arch")),
            "error": str(state.get("error") or ""), "steps": _steps(state.get("steps"))}


def npu_precheck_failure(state: Optional[dict]) -> Optional[tuple]:
    """``(step, reason)`` of the first stage-1 step that keeps the NPU from being offered, else None."""
    state = state or {}
    steps = list(state.get("steps") or [])
    for step in steps:  # a blocked check (it hung, or the app died in it) fails its own step
        if not step.get("ok"):
            return str(step.get("name") or "?"), str(step.get("detail") or state.get("error") or "failed")
    names = {step.get("name") for step in steps}
    for name in NPU_PRECHECK_STEPS:
        if name not in names:
            return name, str(state.get("error") or "not checked")
    arch = state.get("arch")
    if not isinstance(arch, int) or arch < NPU_MIN_ARCH:
        return "arch", f"unsupported HTP arch {arch!r}" if arch is not None else "the HTP arch is unknown"
    return None


def npu_skeleton_source() -> Optional[dict]:
    """``npu_skeleton_source()`` of our build: the llama.cpp release archive that holds the HTP skeletons of this
    engine's commit (``{"release", "commit", "url", "size", "sha256", "files": {"v81": {"name", "member", "size",
    "sha256"}, ...}}``), None on a build without it or with an answer that is not a pinned https archive."""
    functions, missing = _npu_functions("npu_skeleton_source")
    if missing is not None:
        return None
    try:
        raw = functions[0]()
    except Exception:
        log.debug("npu_skeleton_source failed", exc_info=True)
        return None
    if not isinstance(raw, dict) or not str(raw.get("url") or "").startswith("https://") \
            or not raw.get("sha256") or not isinstance(raw.get("files"), dict) or not raw.get("release"):
        return None
    files = {}
    for key, spec in raw["files"].items():
        if not isinstance(spec, dict) or not all(spec.get(k) for k in ("name", "member", "size", "sha256")):
            return None
        files[str(key)] = {"name": str(spec["name"]), "member": str(spec["member"]), "size": int(spec["size"]),
                           "sha256": str(spec["sha256"]).lower()}
    return {"release": str(raw["release"]), "commit": str(raw.get("commit") or ""), "url": str(raw["url"]),
            "size": int(raw.get("size") or 0), "sha256": str(raw["sha256"]).lower(), "files": files}


def npu_skeleton_file(source: Optional[dict], arch: Optional[int]) -> Optional[dict]:
    """The ``files`` entry of ``source`` for HTP arch ``arch`` (81 -> "v81"), None when it has none."""
    if not source or not isinstance(arch, int):
        return None
    spec = (source.get("files") or {}).get(f"v{arch}")
    return spec if isinstance(spec, dict) else None


def npu_state(skel_dir: str, options: Optional[dict] = None) -> dict:
    """Stage 2, ``register_npu_backend(skel_dir, options)`` of our llama-cpp-python build (idempotent; it never
    raises): ``{"backend": "HTP", "device", "arch", "error", "steps": [{"name", "ok", "detail"}]}``. It copies
    nothing: its "skeletons" step fails ("NPU files not downloaded") unless ``skel_dir`` already holds the phone's
    skeleton, and then nothing is registered. A build without the function, or one that does not ship the
    backend, answers a failed "shipped" step."""
    functions, missing = _npu_functions("npu_backend_shipped", "register_npu_backend")
    if missing is not None:
        return missing
    shipped, register = functions
    try:
        if not shipped():
            return _npu_missing("shipped", "this engine build does not carry the NPU backend")
    except Exception as exc:
        return _npu_missing("shipped", f"{type(exc).__name__}: {exc}")
    capture_hex_log()
    capture_aborts()
    try:
        raw = register(str(skel_dir), dict(options or {}))
        state = dict(raw) if isinstance(raw, dict) else {}
    except Exception as exc:  # pragma: no cover - the loader never raises
        state = {"error": f"{type(exc).__name__}: {exc}"}
    out = {"backend": state.get("backend"), "device": str(state.get("device") or ""),
           "arch": _arch_number(state.get("arch")), "error": str(state.get("error") or ""),
           "steps": _steps(state.get("steps"))}
    _NPU_STATE.clear()
    _NPU_STATE.update(out)
    return out


def npu_failure(state: Optional[dict]) -> Optional[tuple]:
    """``(step, reason)`` of the first thing that keeps the NPU from being offered, else None (it passed)."""
    state = state or {}
    steps = list(state.get("steps") or [])
    for step in steps:
        if not step.get("ok"):
            return str(step.get("name") or "?"), str(step.get("detail") or state.get("error") or "failed")
    names = {step.get("name") for step in steps}
    for name in NPU_STEPS:
        if name not in names:
            return name, str(state.get("error") or "not checked")
    arch = state.get("arch")
    if not isinstance(arch, int) or arch < NPU_MIN_ARCH:
        return "arch", f"unsupported HTP arch {arch!r}" if arch is not None else "the HTP arch is unknown"
    if state.get("error"):
        return "session", str(state["error"])
    if not state.get("device"):
        return "registry", "no NPU device"
    return None


def npu_passed(state: Optional[dict]) -> bool:
    """Every step of ``register_npu_backend`` passed, with a known arch the backend supports."""
    return bool(state) and npu_failure(state) is None


def npu_label(state: Optional[dict] = None) -> str:
    """"Hexagon v81 (HTP0)" of the registered NPU."""
    state = state if state is not None else _NPU_STATE
    arch = state.get("arch")
    device = str(state.get("device") or "") or NPU_BACKEND
    return f"Hexagon v{arch} ({device})" if arch else f"Hexagon ({device})"


def soc_label() -> str:
    """``ro.soc.manufacturer`` + ``ro.soc.model`` (Android 12+; a label for the diagnostics only), '' elsewhere."""
    if not npu_platform():
        return ""
    import subprocess

    parts = []
    for prop in ("ro.soc.manufacturer", "ro.soc.model"):
        try:
            done = subprocess.run(["getprop", prop], capture_output=True, text=True, timeout=3)
            value = (done.stdout or "").strip()
        except Exception:
            value = ""
        if value:
            parts.append(value)
    return " ".join(parts)


def device_index(name: str, missing: int = -1) -> int:
    """``device_index(name)`` of our build: the ``main_gpu`` of the registered backend called ``name``
    ("HTP", "OpenCL"), following llama.cpp's own device order; -1 when it is not registered, ``missing`` on a
    build without the function."""
    ext = _extensions()
    function = getattr(ext, "device_index", None) if ext is not None else None
    if function is None:
        return missing
    try:
        return int(function(str(name)))
    except Exception:
        log.debug("device_index(%s) failed", name, exc_info=True)
        return -1


def _remember_hex(text: str, level: Optional[int] = None) -> None:
    if level == NATIVE_ERROR_LEVEL and text.strip():
        _NATIVE_ERRORS.append(text.strip()[:400])
    pending = _HEX_PENDING[0] + text
    *lines, rest = pending.split("\n")
    _HEX_PENDING[0] = rest[-2000:]
    for line in lines:
        line = line.strip()
        if line.startswith("ggml-hex:"):
            _HEX_LOG.append(line[:400])


def hex_log_lines() -> list:
    """The last ``HEX_LOG_LINES`` "ggml-hex:" lines of the native log (since ``capture_hex_log``)."""
    lines = list(_HEX_LOG)
    if _HEX_PENDING[0].strip().startswith("ggml-hex:"):
        lines = (lines + [_HEX_PENDING[0].strip()[:400]])[-HEX_LOG_LINES:]
    return lines


def native_errors() -> list:
    """The last ``NATIVE_ERROR_LINES`` error lines of the native log (since ``capture_hex_log``): why llama.cpp
    refused a load (``llama_model_load_from_file`` only returns NULL)."""
    return list(_NATIVE_ERRORS)


#: the error line llama.cpp adds after every failed model load (llama.cpp's ``llama_model_load_from_file_impl``),
#: after the line that says why: never the reason itself
NATIVE_GENERIC_ERRORS = ("failed to load model",)


def native_reason() -> Optional[str]:
    """The last native error line that says why a load failed (``native_errors`` without llama.cpp's generic
    "failed to load model" that always follows it), or None."""
    for line in reversed(native_errors()):
        if not any(line.endswith(f": {generic}") or line == generic for generic in NATIVE_GENERIC_ERRORS):
            return line
    return None


def _device_handles(names: Sequence[str]) -> list:
    """The ``ggml_backend_dev_t`` of device 0 of each registered backend in ``names`` ("HTP", "OpenCL"), in that
    order (``ggml_backend_reg_by_name`` / ``ggml_backend_reg_dev_get`` of the loader's libggml). Raises
    ``SplitUnavailable`` naming the first backend that is not registered or has no device."""
    import ctypes

    ext = _extensions()
    libggml = getattr(ext, "_LIBGGML", None) if ext is not None else None
    if libggml is None:
        raise SplitUnavailable("this engine build cannot choose the devices")
    by_name = libggml.ggml_backend_reg_by_name
    by_name.restype, by_name.argtypes = ctypes.c_void_p, [ctypes.c_char_p]
    count = libggml.ggml_backend_reg_dev_count
    count.restype, count.argtypes = ctypes.c_size_t, [ctypes.c_void_p]
    get = libggml.ggml_backend_reg_dev_get
    get.restype, get.argtypes = ctypes.c_void_p, [ctypes.c_void_p, ctypes.c_size_t]
    handles = []
    for name in names:
        reg = by_name(str(name).encode("ascii"))
        if not reg:
            raise SplitUnavailable(f"the {name} backend is not registered")
        if int(count(reg)) < 1:
            raise SplitUnavailable(f"the {name} backend has no device")
        handles.append(int(get(reg, 0)))
    return handles


@contextlib.contextmanager
def _model_devices(handles: Sequence[int]) -> Iterator[None]:
    """While a ``Llama()`` is built on this thread: its ``llama_model_params.devices`` is the NULL-terminated list
    ``handles`` (llama-cpp-python 0.3.37 has no device argument; llama.py fills the struct it gets from
    ``llama_cpp.llama_cpp.llama_model_default_params()`` and loads the model in the same ``__init__``). Other
    threads get the plain defaults. Call under ``_NATIVE_LOAD_LOCK``."""
    import ctypes

    module = importlib.import_module("llama_cpp.llama_cpp")
    original = module.llama_model_default_params
    devices = (ctypes.c_void_p * (len(handles) + 1))(*handles, None)
    thread = threading.get_ident()

    def with_devices() -> Any:
        params = original()
        if threading.get_ident() == thread:
            params.devices = ctypes.addressof(devices)
        return params

    module.llama_model_default_params = with_devices
    try:
        yield
    finally:
        module.llama_model_default_params = original


def capture_hex_log() -> bool:
    """Wrap llama-cpp-python's log callback (``llama_cpp._logger``): every line still goes where it went (the
    original callback prints by the "llama-cpp-python" logger's level); the "ggml-hex:" lines are kept as well, and
    the error lines (``native_errors``). Once per process; False when this build has no log hook."""
    if _HEX_HOOK:
        return True
    try:
        llama_cpp = importlib.import_module("llama_cpp")
        logger_module = importlib.import_module("llama_cpp._logger")
    except Exception:
        return False
    callback_type = getattr(llama_cpp, "llama_log_callback", None)
    setter = getattr(llama_cpp, "llama_log_set", None)
    if callback_type is None or setter is None:
        return False
    original = getattr(logger_module, "llama_log_callback", None)

    def hook(level: int, text: Any, user_data: Any) -> None:
        try:
            _remember_hex(text.decode("utf-8", "replace") if isinstance(text, bytes) else str(text or ""), int(level))
        except Exception:
            pass
        if original is not None:
            try:
                original(level, text, user_data)
            except Exception:
                pass

    try:
        import ctypes

        callback = callback_type(hook)
        setter(callback, ctypes.c_void_p(0))
    except Exception:
        log.debug("installing the native log hook failed", exc_info=True)
        return False
    _HEX_HOOK.append(callback)
    return True


#: the native abort note in the app's data folder (``capture_aborts``): the last ggml_abort message of a run
ABORT_NOTE_FILE = "native_abort.txt"
#: the longest native message kept (characters after the build folder is stripped)
ABORT_NOTE_CHARS = 240
#: this process: its id and start time (the note and the crash markers carry it; a note matches a marker of its run)
RUN_ID = f"{os.getpid()}-{int(time.time())}"
_ABORT_HOOK: list = []  # the registered abort callback (kept alive: ggml calls it from native code)
_ABORT_NOTES: dict = {}  # path -> the note as read once per process (``abort_note``)


def app_data_dir() -> str:
    """The app's data folder (the app's private files directory on Android): runtime_bootstrap's, else
    ``GLOSSARION_DATA_DIR``, else ``~/.glossarion``."""
    base = ""
    try:
        from glossarion_mobile import runtime_bootstrap as rb

        paths = rb.get_paths()
        base = str(paths.data) if paths is not None else ""
    except Exception:
        base = ""
    return base or os.environ.get("GLOSSARION_DATA_DIR", "").strip() or os.path.join(os.path.expanduser("~"), ".glossarion")


def abort_note_path() -> str:
    return os.path.join(app_data_dir(), ABORT_NOTE_FILE)


def _write_abort_note(target: bytes, prefix: bytes, message: Any) -> None:
    """The abort callback's whole work: ``<run>\\t<pid>\\t<epoch s>\\t<message>`` into ``target`` with os-level
    calls, on disk before it returns (ggml calls abort() right after)."""
    try:
        text = message if isinstance(message, bytes) else b""
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_BINARY", 0), 0o600)
        try:
            os.write(fd, prefix + str(int(time.time())).encode() + b"\t" + text[:2048])
            os.fsync(fd)
        finally:
            os.close(fd)
    except BaseException:
        pass


def capture_aborts(path: Optional[str] = None) -> bool:
    """Register ``ggml_set_abort_callback`` (the loader's libggml, like ``_device_handles``) once per process, before
    native work: a GGML_ABORT / GGML_ASSERT then writes its "file:line: message" to the note (``abort_note_path``)
    before the process aborts. False, and nothing else, on a build without the symbol or any failure."""
    if _ABORT_HOOK:
        return True
    try:
        import ctypes

        ext = _extensions()
        libggml = getattr(ext, "_LIBGGML", None) if ext is not None else None
        if libggml is None:
            return False
        setter = libggml.ggml_set_abort_callback
        callback_type = ctypes.CFUNCTYPE(None, ctypes.c_char_p)
        setter.restype, setter.argtypes = ctypes.c_void_p, [callback_type]
        target = path or abort_note_path()
        with contextlib.suppress(OSError):
            os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
        encoded, prefix = os.fsencode(target), f"{RUN_ID}\t{os.getpid()}\t".encode()

        def on_abort(message: Any) -> None:
            _write_abort_note(encoded, prefix, message)

        callback = callback_type(on_abort)
        setter(callback)
    except Exception:
        log.debug("registering the native abort callback failed", exc_info=True)
        return False
    _ABORT_HOOK.append(callback)
    return True


def _abort_message(raw: str) -> str:
    """The note's message on one line, without the build machine's folder before "file.cpp:635:", bounded."""
    text = " ".join(raw.split())
    text = re.sub(r"^\S*[/\\]([^/\\\s]+:\d+:)", r"\1", text)
    return text if len(text) <= ABORT_NOTE_CHARS else text[:ABORT_NOTE_CHARS - 1] + "…"


def abort_note(path: Optional[str] = None) -> Optional[dict]:
    """The last native abort's note: ``{"run", "pid", "at", "message"}``, None without one (read once per process: only
    a dead run writes it)."""
    target = path or abort_note_path()
    if target not in _ABORT_NOTES:
        note = None
        try:
            with open(target, "rb") as handle:
                parts = handle.read(4096).decode("utf-8", "replace").split("\t", 3)
            if len(parts) == 4 and parts[0] and parts[3].strip():
                note = {"run": parts[0], "pid": int(parts[1]), "at": int(parts[2]), "message": _abort_message(parts[3])}
        except (OSError, ValueError):
            note = None
        _ABORT_NOTES[target] = note
    return _ABORT_NOTES[target]


def abort_reason(marker: Any, path: Optional[str] = None) -> Optional[str]:
    """The native abort message of the run that left ``marker`` (a crash marker's ``"run"``), else None: a note from
    any other run is never attached."""
    run = marker.get("run") if isinstance(marker, dict) else None
    note = abort_note(path) if run else None
    return note["message"] if note and note["run"] == run else None


def with_abort_reason(reason: str, marker: Any, limit: int = 300) -> str:
    """``reason`` followed by the native abort message of ``marker``'s run, when there is one, within ``limit``
    characters (the recorded reasons' cap), so a cut message keeps its closing ellipsis."""
    native = abort_reason(marker)
    if not native:
        return reason
    room = limit - len(reason) - 2
    if room < 2:
        return reason
    if len(native) > room:
        native = native[:room - 1].rstrip() + "…"
    return f"{reason}: {native}"


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


def available_cpus() -> int:
    """Logical cores available to this process, including Android cpuset restrictions."""
    cpus = 0
    affinity = getattr(os, "sched_getaffinity", None)
    if affinity is not None:
        try:
            cpus = len(affinity(0))
        except OSError:
            cpus = 0
    cpus = cpus or os.cpu_count() or 4
    return max(1, cpus)


def default_threads() -> int:
    """Leave CPU room for the UI, with at most four automatic inference threads."""
    return min(4, cpu_budget(available_cpus()))


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


def _split_of(npu: Any, split: Optional[str]) -> Optional[str]:
    """``load``'s NPU + GPU split ("layer" / "tensor"), None for every other load; a split needs ``npu``."""
    if split is None:
        return None
    if split not in SPLITS:
        raise ValueError(f"unknown NPU + GPU split {split!r} (known: {', '.join(SPLITS)})")
    if not npu:
        raise ValueError("an NPU + GPU split is an NPU load (npu=True)")
    return split


def _handoff_of(npu: Any, split: Optional[str], handoff: Any) -> bool:
    """``load``'s NPU-reads / CPU-writes mode: an NPU load without a split."""
    if not handoff:
        return False
    if not npu or split is not None:
        raise ValueError("NPU reads, CPU writes is an NPU load without a split (npu=True, split=None)")
    return True


def _timing(prompt_tokens: int, started: float, first_at: float, ended: float, tokens: int) -> dict:
    """``last_timing`` of an answer: the prompt from ``started`` to the first output token, the generation from
    there to ``ended`` (no output: all of it was the prompt)."""
    first_at = first_at or ended
    return {"prompt_tokens": int(prompt_tokens), "prompt_s": max(first_at - started, 0.0), "tokens": int(tokens),
            "gen_s": max(ended - first_at, 0.0)}


def _with_handoff(timing: dict, handoff: Optional[dict]) -> dict:
    """``timing`` plus ``handoff_s`` / ``handoff_bytes`` of the answer's hand-off (NPU reads, CPU writes), when one ran.
    The prompt figures already include it: the prompt runs to the first output token."""
    if handoff:
        timing.update(handoff_s=float(handoff.get("seconds") or 0.0), handoff_bytes=int(handoff.get("bytes") or 0))
    return timing


class _Base:
    def __init__(self) -> None:
        self.path: Optional[str] = None
        self.mmproj: Optional[str] = None
        self.n_ctx = 0
        self.n_threads = 0
        self.last_usage: dict = {}
        self.last_finish = "stop"
        self.on_gpu = False  # the loaded model runs on the GPU
        self.on_npu = False  # the loaded model runs on the NPU
        #: "layer" / "tensor": the loaded model is split over the NPU and the GPU (``on_npu`` / ``on_gpu`` stay
        #: False: no NPU or GPU marker or fallback is about such a load)
        self.split: Optional[str] = None
        #: NPU reads, CPU writes (``load(handoff=True)``): the NPU reads each prompt, the CPU writes the answer
        self.handoff = False
        #: the last answer's hand-off: ``{"seconds", "bytes", "read_s"}`` (None: no hand-off ran)
        self.last_handoff: Optional[dict] = None
        self.device: Optional[str] = None  # what runs the loaded model ("GPU: … (OpenCL)", "NPU: …", "CPU …")
        #: the last answer's speed: ``{"prompt_tokens", "prompt_s", "tokens", "gen_s"}`` (``gen_s``: from the first
        #: output token to the last, over ``tokens - 1`` tokens)
        self.last_timing: dict = {}

    @property
    def backend(self) -> Optional[str]:
        """"NPU", "GPU", "CPU", "NPU + GPU" or "NPU + CPU" (NPU reads, CPU writes) of the loaded model, None without
        one."""
        if self.path is None:
            return None
        if self.split:
            return "NPU + GPU"
        if self.handoff:
            return "NPU + CPU"
        return "NPU" if self.on_npu else "GPU" if self.on_gpu else "CPU"

    @property
    def loaded(self) -> bool:
        return self.path is not None

    def warm(self) -> None:
        """Run the loaded model once the way an NPU / GPU load's first-token check does (``_gpu_check``) when its
        load did not (a CPU load), so a timed answer starts on a model that has run before. The device checks'
        timing only; normal loads never call it."""

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
                 gpu_error: Optional[str] = None, gpu_chat_error: Optional[str] = None,
                 npu_device: str = "Hexagon v81 (HTP0)", npu_error: Optional[str] = None,
                 npu_chat_error: Optional[str] = None, npu_load_delay: float = 0.0,
                 split_error: Optional[dict] = None, split_chat_error: Optional[str] = None,
                 handoff_error: Optional[str] = None, handoff_bytes: int = 3_200_000) -> None:
        super().__init__()
        self.delay = delay
        self.think = think
        self.load_delay = load_delay
        self.reply = reply
        self.gpu_device = gpu_device
        self.gpu_error = gpu_error  # a GPU load raises GpuUnavailable(gpu_error)
        self.gpu_chat_error = gpu_chat_error  # a chat on the GPU raises RuntimeError(gpu_chat_error)
        self.npu_device = npu_device
        self.npu_error = npu_error  # an NPU load raises NpuUnavailable(npu_error)
        self.npu_chat_error = npu_chat_error  # a chat on the NPU raises RuntimeError(npu_chat_error)
        self.npu_load_delay = npu_load_delay  # an NPU load takes this long (the watchdog tests)
        self.split_error = dict(split_error or {})  # {split: why}: that NPU + GPU load raises SplitUnavailable(why)
        self.split_chat_error = split_chat_error  # a chat on the NPU + GPU raises RuntimeError(split_chat_error)
        self.handoff_error = handoff_error  # an NPU-reads / CPU-writes load raises SplitUnavailable(handoff_error)
        self.handoff_bytes = int(handoff_bytes)  # the sequence state each hand-off copies
        self.loads: list = []
        self.gpu_loads: list = []  # the gpu flag of every load attempt
        #: "npu" / "gpu" / "cpu" / "npu+gpu:layer" / "npu+gpu:tensor" / "npu+cpu:handoff" of every load attempt
        self.backends: list = []
        self.npu_layers: list = []  # the layer count of every NPU load attempt
        self.unloads = 0
        self.generated = 0  # tokens produced since the last chat started
        self.calls: list = []
        self.projector_loads = 0
        self._projector = False
        self.warms: list = []  # the backend ("npu" / "gpu" / "cpu" / "npu+gpu:…") of every ``warm`` that ran

    def warm(self) -> None:
        if self.path is not None and self.backends and self.backends[-1] == "cpu":
            self.warms.append(self.backends[-1])

    def load(self, path: str, n_ctx: int, n_threads: int, mmproj: Optional[str] = None, gpu: bool = False,
             n_gpu_layers: int = DEFAULT_GPU_LAYERS, npu: bool = False, split: Optional[str] = None,
             handoff: bool = False) -> None:
        split = _split_of(npu, split)
        handoff = _handoff_of(npu, split, handoff)
        npu = bool(npu) and split is None and not handoff
        gpu = bool(gpu and n_gpu_layers != 0) and not npu and split is None and not handoff
        self.n_gpu_layers = -1 if split or handoff else int(n_gpu_layers) if gpu or npu else 0
        if self.load_delay:
            time.sleep(self.load_delay)
        self.gpu_loads.append(bool(gpu))
        self.backends.append(f"npu+gpu:{split}" if split else "npu+cpu:handoff" if handoff else "npu" if npu
                             else "gpu" if gpu else "cpu")
        if handoff:
            if self.npu_load_delay:
                time.sleep(self.npu_load_delay)
            if self.handoff_error:
                self.unload()
                raise SplitUnavailable(self.handoff_error)
            mmproj = None  # no vision in this mode
        if split:
            if self.npu_load_delay:
                time.sleep(self.npu_load_delay)
            if self.split_error.get(split):
                self.unload()
                raise SplitUnavailable(self.split_error[split])
        if npu:
            self.npu_layers.append(int(n_gpu_layers))
            if self.npu_load_delay:
                time.sleep(self.npu_load_delay)
            if self.npu_error:
                self.unload()
                raise NpuUnavailable(self.npu_error)
        if gpu and self.gpu_error:
            self.unload()
            raise GpuUnavailable(self.gpu_error)
        self.path, self.n_ctx, self.n_threads = path, int(n_ctx), int(n_threads)
        self.mmproj = mmproj
        self._projector = False
        self.on_gpu = bool(gpu)
        self.on_npu = npu
        self.split = split
        self.handoff = handoff
        self.device = (f"NPU + GPU ({split} split): {self.npu_device} + {self.gpu_device}" if split
                       else f"NPU + CPU: {self.npu_device} reads, CPU writes" if handoff
                       else f"NPU: {self.npu_device}" if npu else f"GPU: {self.gpu_device}" if gpu else "CPU")
        self.loads.append((path, int(n_ctx), int(n_threads)))

    def unload(self) -> None:
        if self.path is not None:
            self.unloads += 1
        self.path = None
        self.mmproj = None
        self._projector = False
        self.on_gpu = False
        self.on_npu = False
        self.split = None
        self.handoff = False
        self.device = None

    @property
    def vision_loaded(self) -> bool:
        return self._projector

    def count_prompt(self, messages: Sequence[dict]) -> int:
        return sum(len(_text_of(m.get("content")).split()) + 4 for m in messages) + self._image_tokens(messages)

    def chat(self, messages: Sequence[dict], params: dict, should_stop: Callable[[], bool]) -> Iterator[str]:
        self.calls.append({"messages": list(messages), "params": dict(params), "gpu": self.on_gpu, "npu": self.on_npu,
                           "split": self.split, "handoff": self.handoff})
        self.last_handoff = None
        self.generated = 0
        if self.on_gpu and self.gpu_chat_error:
            raise RuntimeError(self.gpu_chat_error)
        if self.on_npu and self.npu_chat_error:
            raise RuntimeError(self.npu_chat_error)
        if self.split and self.split_chat_error:
            raise RuntimeError(self.split_chat_error)
        started = time.monotonic()
        first_at = 0.0
        images = image_count(messages)
        if images:
            if not self.mmproj:
                raise EngineUnavailable("This on-device model has no vision support")
            if not self._projector:
                self._projector = True
                self.projector_loads += 1
        if self.handoff:
            # the NPU's read and the copy of its state (the real engine does both before the CPU's first token)
            self.last_handoff = {"seconds": 0.01, "bytes": self.handoff_bytes, "read_s": 0.0}
            if should_stop():
                raise GenerationStopped("A Stop arrived while the prompt was being evaluated")
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
            first_at = first_at or time.monotonic()
            yield token
        prompt = self.count_prompt(messages)
        self.last_usage = {"prompt_tokens": prompt, "completion_tokens": self.generated,
                           "total_tokens": prompt + self.generated}
        self.last_timing = _with_handoff(_timing(prompt, started, first_at, time.monotonic(), self.generated),
                                         self.last_handoff)


# Backend registration and llama-cpp-python's first initialization are process-global.
# Serialize model construction; independent contexts can decode concurrently afterwards.
_NATIVE_LOAD_LOCK = threading.RLock()


class LlamaEngine(_Base):
    """llama-cpp-python, one model at a time; the caller (``LocalEngineService``) serialises calls."""

    name = "llama.cpp"

    def __init__(self) -> None:
        super().__init__()
        self.llm: Any = None
        self.reader: Any = None  # NPU reads, CPU writes: the NPU copy of the model (``llm`` is the CPU one)
        self.formatter: Any = None
        self.vision_handler: Any = None
        self._scripts: Any = None  # the loaded model's script_ban.VocabScripts (computed on first use)
        self.sliding_window = 0  # the loaded model's SWA window in tokens (0: none)
        self.live: Optional["_Heartbeat"] = None  # the request running now (``chat``), for the notification
        self._lock = threading.RLock()
        self._checked = False  # the loaded model's first-token check ran (an NPU / GPU load)

    def warm(self) -> None:
        with self._lock:
            llm = self.llm
            if llm is None or self._checked:
                return
            self._checked = True
            try:
                _gpu_check(llm)
            except Exception:
                log.debug("warming the model failed", exc_info=True)

    def _module(self) -> Any:
        try:
            return importlib.import_module("llama_cpp")
        except Exception as exc:
            raise EngineUnavailable(f"{UNAVAILABLE}: {exc}") from exc

    def load(self, path: str, n_ctx: int, n_threads: int, mmproj: Optional[str] = None, gpu: bool = False,
             n_gpu_layers: int = DEFAULT_GPU_LAYERS, npu: bool = False, split: Optional[str] = None,
             handoff: bool = False) -> None:
        split = _split_of(npu, split)
        if _handoff_of(npu, split, handoff):
            self._load_handoff(path, n_ctx, n_threads)
            return
        npu = bool(npu) and split is None
        gpu = bool(gpu and n_gpu_layers != 0) and not npu and split is None
        llama_cpp = self._module()
        capture_aborts()  # before the GPU backend registers and the model loads
        with self._lock, _NATIVE_LOAD_LOCK:
            self.unload()
            device = None
            handles: Optional[list] = None
            extra: dict = {}
            if split:
                # exactly the NPU and then the GPU (``_model_devices``), half the model each, every layer offloaded
                if device_index(NPU_BACKEND) < 0:
                    raise SplitUnavailable(_NPU_STATE.get("error") or "the NPU backend is not registered")
                state = gpu_state()
                if not state.get("device"):
                    raise SplitUnavailable(state.get("error") or "no GPU device")
                backend = str(state.get("backend") or "")
                handles = _device_handles((NPU_BACKEND, backend))
                device = f"NPU + GPU ({split} split): {npu_label()} + {state['device']} ({backend})"
                placement = {"n_gpu_layers": -1, "split_mode": SPLITS[split], "main_gpu": 0,
                             "tensor_split": [1.0, 1.0]}
                if split == "tensor":
                    # an unquantised KV cache for this load only (flash attention is on for every load)
                    extra = {"type_k": GGML_TYPE_F16, "type_v": GGML_TYPE_F16}
                _NATIVE_ERRORS.clear()
            elif npu:
                # registered by the service's NPU check (``npu_state``) before any NPU load
                index = device_index(NPU_BACKEND)
                if index < 0:
                    raise NpuUnavailable(_NPU_STATE.get("error") or "the NPU backend is not registered")
                device = f"NPU: {npu_label()}"
                placement = {"n_gpu_layers": int(n_gpu_layers or -1), "split_mode": SPLIT_MODE_NONE,
                             "main_gpu": index}
            elif gpu:
                state = gpu_state()
                if not state.get("device"):
                    raise GpuUnavailable(state.get("error") or "no GPU device")
                device = f"GPU: {state['device']} ({state.get('backend')})"
                # the GPU backend's own device: never the NPU, whatever else is registered (a build without
                # ``device_index`` has the GPU as its only device: 0)
                backend = str(state.get("backend") or "")
                index = device_index(backend, missing=0)
                if index < 0:
                    if backend == "OpenCL":
                        raise GpuUnavailable("the OpenCL backend is not in llama.cpp's device list")
                    index = 0
                # The UI chooses the count; -1 offloads all layers only when explicitly selected.
                placement = {"n_gpu_layers": int(n_gpu_layers), "split_mode": SPLIT_MODE_NONE, "main_gpu": index}
            else:
                placement = {"n_gpu_layers": 0, "split_mode": SPLIT_MODE_NONE, "main_gpu": -1}
            try:
                # Keep SWA caches window-sized and enable Flash Attention on both CPU and GPU.
                # The mobile OpenCL wheel includes the upstream DK512 Gemma 4 kernels; disabling FA
                # pads Sigma's mixed-size V cache and prevents the fused GPU attention path.
                # ``chat`` starts every sliding-window request on an empty cache.
                with _model_devices(handles) if handles else contextlib.nullcontext():
                    llm = llama_cpp.Llama(model_path=str(path), n_ctx=int(n_ctx), n_threads=int(n_threads),
                                          n_threads_batch=int(n_threads), n_batch=512, n_ubatch=512, use_mmap=True,
                                          use_mlock=False, verbose=False, swa_full=False, flash_attn=True,
                                          **placement, **extra)
            except Exception as exc:
                if split:
                    native = native_reason()
                    raise SplitUnavailable(f"the model did not load on the NPU + GPU ({exc})"
                                           + (f": {native}" if native else "")) from exc
                if npu:
                    raise NpuUnavailable(f"the model did not load on the NPU ({exc})") from exc
                if gpu:
                    raise GpuUnavailable(f"the model did not load on the GPU ({exc})") from exc
                raise
            if gpu or npu or split:
                try:
                    _gpu_check(llm)
                except Exception as exc:
                    _close(llm)
                    if split:
                        raise SplitUnavailable(f"the first token on the NPU + GPU failed ({exc})") from exc
                    if npu:
                        raise NpuUnavailable(f"the first token on the NPU failed ({exc})") from exc
                    raise GpuUnavailable(f"the first token on the GPU failed ({exc})") from exc
            self.formatter = _make_formatter(llm)
            if self.formatter is not None:
                llm.chat_handler = self.formatter.to_chat_handler()
            self.llm = llm
            self.sliding_window = _sliding_window(llama_cpp, llm)
            self.path, self.n_ctx, self.n_threads = str(path), int(n_ctx), int(n_threads)
            self.mmproj = str(mmproj) if mmproj else None
            self.on_gpu = bool(gpu)
            self.on_npu = npu
            self.split = split
            self._checked = bool(gpu or npu or split)  # ``_gpu_check`` ran on this load
            if device is None:
                variant = cpu_variant()
                device = f"CPU ({variant})" if variant and variant not in ("builtin", "none") else "CPU"
            self.device = device

    def _load_handoff(self, path: str, n_ctx: int, n_threads: int) -> None:
        """NPU reads, CPU writes: the NPU copy (``reader``: every layer on the HTP) and the CPU copy (``llm``) of the
        model with the same context settings, then one hand-off on ``GPU_CHECK_PROMPT`` (the self-test, and the
        warm-up of both). Any failure frees both and raises ``SplitUnavailable`` with llama.cpp's reason."""
        llama_cpp = self._module()
        capture_aborts()  # before the models load
        with self._lock, _NATIVE_LOAD_LOCK:
            self.unload()
            index = device_index(NPU_BACKEND)
            if index < 0:
                raise SplitUnavailable(_NPU_STATE.get("error") or "the NPU backend is not registered")
            common = {"model_path": str(path), "n_ctx": int(n_ctx), "n_threads": int(n_threads),
                      "n_threads_batch": int(n_threads), "n_batch": 512, "n_ubatch": 512, "use_mmap": True,
                      "use_mlock": False, "verbose": False, "swa_full": False, "flash_attn": True}
            _NATIVE_ERRORS.clear()
            reader = llm = None
            try:
                try:
                    reader = llama_cpp.Llama(**common, n_gpu_layers=-1, split_mode=SPLIT_MODE_NONE, main_gpu=index)
                except Exception as exc:
                    raise SplitUnavailable(f"the model did not load on the NPU ({exc})") from exc
                try:
                    _gpu_check(reader)
                except Exception as exc:
                    raise SplitUnavailable(f"the first token on the NPU failed ({exc})") from exc
                try:
                    llm = llama_cpp.Llama(**common, n_gpu_layers=0, split_mode=SPLIT_MODE_NONE, main_gpu=-1)
                except Exception as exc:
                    raise SplitUnavailable(f"the model did not load on the CPU ({exc})") from exc
                try:
                    _handoff_check(llama_cpp, reader, llm)
                except Exception as exc:
                    raise SplitUnavailable(f"the CPU could not continue from the NPU ({exc})") from exc
            except SplitUnavailable as exc:
                for model in (llm, reader):  # the CPU copy first, as ``unload``
                    if model is not None:
                        _close(model)
                native = native_reason()
                if native:
                    raise SplitUnavailable(f"{exc}: {native}") from exc
                raise
            self.formatter = _make_formatter(llm)
            if self.formatter is not None:
                llm.chat_handler = self.formatter.to_chat_handler()
            self.llm, self.reader = llm, reader
            self.sliding_window = _sliding_window(llama_cpp, llm)
            self.path, self.n_ctx, self.n_threads = str(path), int(n_ctx), int(n_threads)
            self.mmproj = None  # no vision in this mode
            self.handoff = True
            self._checked = True  # the hand-off check ran both models
            self.device = f"NPU + CPU: {npu_label()} reads, CPU writes"

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
            reader, self.reader = self.reader, None
            handler, self.vision_handler = self.vision_handler, None
            self.formatter = None
            self._scripts = None
            self.sliding_window = 0
            self.path = None
            self.mmproj = None
            self.on_gpu = False
            self.on_npu = False
            self.split = None
            self.handoff = False
            self.last_handoff = None
            self.device = None
            # The projector goes before its model. llama-cpp-python 0.3.37 registers mtmd_free on the
            # model's own exit stack, after the context and the model, so ``llm.close()`` below runs it
            # first; up to 0.3.35 the handler had its own stack, closed here.
            stack = getattr(handler, "_exit_stack", None) if handler is not None else None
            if stack is not None:
                try:
                    stack.close()
                except Exception:
                    log.debug("freeing the vision projector failed", exc_info=True)
            if llm is not None:
                try:
                    llm.close()
                except Exception:
                    log.debug("closing the model failed", exc_info=True)
                del llm
            if reader is not None:  # NPU reads, CPU writes: the NPU copy after the CPU one
                _close(reader)
                del reader

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
        if params.get("normal_settings"):
            # Sigma Q8 uses the caller's settings; absent values use llama.cpp's defaults,
            # never the Qwen/small-model defaults above.
            for key in ("temperature", "top_p", "top_k", "min_p", "presence_penalty", "frequency_penalty", "repeat_penalty"):
                if key not in params:
                    kwargs.pop(key, None)
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
        # 0.3.37's create_chat_completion takes no stopping_criteria: the stream is checked every
        # STOP_CHECK_EVERY chunks and closing the generator stops llama.cpp's sampling loop; until the first
        # chunk (the prompt, which can take minutes on a phone) a Stop is seen between two decodes.
        cpus_done = _follow_cpus(llm, self.n_threads)
        heartbeat = _Heartbeat(self)
        try:
            heartbeat.prompt_total = self.count_prompt(messages)
        except Exception:
            heartbeat.prompt_total = 0
        heartbeat.start()
        self.live = heartbeat  # the job notification's on-device row (``notification_row``)
        prompt_done = _stop_between_decodes(llm, should_stop, on_batch=heartbeat.read)
        self.last_handoff = None
        handoff_done = self._handoff_for_this_call(llm, should_stop, heartbeat) if self.reader is not None \
            else (lambda: None)
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
                    if heartbeat.reading:
                        heartbeat.reading, heartbeat.first_token_at = False, time.monotonic()
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
            if self.live is heartbeat:
                self.live = None
            prompt_done()
            handoff_done()
            cpus_done()
            # the prompt's evaluated tokens (a reused prefix is not evaluated again), else its estimate
            self.last_timing = _with_handoff(_timing(heartbeat.prompt_read or heartbeat.prompt_total, heartbeat.started,
                                                     heartbeat.first_token_at, time.monotonic(), completion),
                                             self.last_handoff)
        prompt = 0
        try:
            prompt = self.count_prompt(messages)
        except Exception:
            pass
        self.last_usage = {"prompt_tokens": prompt, "completion_tokens": completion,
                           "total_tokens": prompt + completion}

    def _handoff_for_this_call(self, llm: Any, should_stop: Callable[[], bool], heartbeat: "_Heartbeat") -> Callable[[], None]:
        """NPU reads, CPU writes, for one ``chat`` call: ``llm.generate`` (an instance attribute over the class's) gets
        the exact prompt tokens ``_create_completion`` built; the NPU reads all but the last (a Stop is seen between
        two of its decodes, its batches count as the prompt's progress), its sequence state goes to the CPU context,
        and the CPU's own ``generate`` reads the last token and writes the answer. Returns the function that removes
        the wrapper (safe to call again)."""
        reader = self.reader
        llama_cpp = self._module()
        original = llm.generate

        def generate(tokens: Any, *args: Any, **kwargs: Any) -> Any:
            tokens = list(tokens)
            if len(tokens) > 1:
                started = time.monotonic()
                reader.reset()
                reader._ctx.kv_cache_clear()
                done = _stop_between_decodes(reader, should_stop, on_batch=heartbeat.read)
                try:
                    reader.eval(tokens[:-1])
                finally:
                    done()
                read_at = time.monotonic()
                copied = _handoff_state(llama_cpp, reader, llm, tokens)
                self.last_handoff = {"seconds": time.monotonic() - read_at, "bytes": copied,
                                     "read_s": read_at - started}
            return original(tokens, *args, **kwargs)

        llm.generate = generate

        def done() -> None:
            if vars(llm).get("generate") is generate:
                del llm.generate  # the class's own method again

        return done


def _handoff_state(llama_cpp: Any, reader: Any, llm: Any, tokens: Sequence[int]) -> int:
    """Copy sequence 0's memory from the NPU context (``reader``, which has read ``tokens[:-1]``) to the CPU context
    (``llm``, emptied first) and set the CPU model's token record to match, so its ``generate`` replays only the last
    token. Returns the bytes copied; raises ``RuntimeError`` when the state is empty or refused (llama.cpp returns 0
    on any mismatch: layer count, KV types, V transposition, free cells)."""
    import ctypes

    size = int(llama_cpp.llama_state_seq_get_size(reader.ctx, 0))
    if size <= 0:
        raise RuntimeError("the NPU's state is empty")
    buffer = (ctypes.c_uint8 * size)()
    copied = int(llama_cpp.llama_state_seq_get_data(reader.ctx, buffer, size, 0))
    llm.reset()
    llm._ctx.kv_cache_clear()
    if copied <= 0 or not int(llama_cpp.llama_state_seq_set_data(llm.ctx, buffer, copied, 0)):
        raise RuntimeError("the CPU could not read the NPU's state")
    count = len(tokens) - 1
    llm.input_ids[:count] = list(tokens[:count])
    llm.n_tokens = count
    llm._requires_eval = True
    return copied


def _handoff_check(llama_cpp: Any, reader: Any, llm: Any) -> None:
    """The hand-off once at load: the NPU reads ``GPU_CHECK_PROMPT`` but its last token, the CPU takes its state and
    reads the last token; the CPU's logits must be finite and not all equal. Leaves both models reset."""
    tokens = list(llm.tokenize(GPU_CHECK_PROMPT.encode("utf-8"), add_bos=True, special=False))[:64]
    if len(tokens) < 2:
        raise RuntimeError("the check prompt has too few tokens")
    try:
        reader.reset()
        reader._ctx.kv_cache_clear()
        reader.eval(tokens[:-1])
        _handoff_state(llama_cpp, reader, llm, tokens)
        llm.eval(tokens[-1:])
        _check_logits(llm)
    finally:
        reader.reset()
        llm.reset()


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
        self.first_token_at = 0.0
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

    def notification_row(self, threads: bool = False) -> str:
        """The job notification's on-device row (hotfix 3): "📱 Processing · 2,048 / ~5,896 tokens ·
        32.0 tok/s" until the first output token, then "📱 412 tokens written · 6.5 tok/s" (averages: the
        notification updates every second, a per-second speed would jump). ``threads`` (the job screen's
        status beside its Log title, hotfix 4) adds " · 4 threads"."""
        now = time.monotonic()
        if self.reading:
            total = f"~{self.prompt_total:,}" if self.prompt_total else "?"
            if not self.prompt_read:
                row = f"📱 Processing · {total} tokens"
            else:
                read = min(self.prompt_read, self.prompt_total or self.prompt_read)
                speed = self.prompt_read / max(now - self.started, 0.001)
                row = f"📱 Processing · {read:,} / {total} tokens · {speed:.1f} tok/s"
        else:
            speed = self.tokens / max(now - (self.first_token_at or self.started), 0.001)
            row = f"📱 {self.tokens:,} tokens written · {speed:.1f} tok/s"
        return f"{row} · {self.threads()} threads" if threads else row

    def threads(self) -> int:
        """The threads llama.cpp runs now (``_follow_cpus`` changes them), else the loaded count."""
        try:
            return int(importlib.import_module("llama_cpp").llama_n_threads(self.engine.llm._ctx.ctx))
        except Exception:
            return self.engine.n_threads

    def line(self, seconds: float, new_tokens: int) -> str:
        """The progress since the last line: ``new_tokens`` output tokens, or prompt tokens while reading."""
        affinity = getattr(os, "sched_getaffinity", None)
        try:
            cpus = _cpu_list(affinity(0)) if affinity is not None else "?"
        except OSError:
            cpus = "?"
        threads = self.threads()
        cpuset = _cpuset()
        rate = f"{new_tokens / max(seconds, 0.001):.1f} tok/s"
        total = f"~{self.prompt_total:,}" if self.prompt_total else "?"
        if not self.reading:
            progress = f"{self.tokens:,} tokens written, {rate} (last {seconds:.0f} s)"
        elif self.prompt_read:
            # the average since the start: a 512-token batch can take longer than one line's interval
            average = self.prompt_read / max(time.monotonic() - self.started, 0.001)
            progress = (f"processing, {min(self.prompt_read, self.prompt_total or self.prompt_read):,} "
                        f"of {total} tokens ({average:.1f} tok/s)")
        else:
            progress = f"processing ({total} tokens)"
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
                progress_log.info("%s", self.line(now - last_at, tokens - last_tokens))
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
        progress_log.info("📱 on-device: %d of %d threads, %d CPU%s available to the app", count, threads, cpus,
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
    the tokens behind their window, but 0.3.37's prompt-prefix reuse (``Llama.generate``) only checks that
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
    """The GPU's or the NPU's first tokens: evaluate ``GPU_CHECK_PROMPT`` (a batch) and one more token; the last
    logits must be finite and not all equal (a broken kernel gives NaN / inf / zeros). Leaves the
    model reset."""
    tokens = list(llm.tokenize(GPU_CHECK_PROMPT.encode("utf-8"), add_bos=True, special=False))[:64]
    if not tokens:
        raise RuntimeError("the check prompt has no tokens")
    try:
        llm.reset()
        llm.eval(tokens)
        llm.eval(tokens[-1:])
        _check_logits(llm)
    finally:
        llm.reset()


def _check_logits(llm: Any) -> None:
    """The last decode's logits are finite and not all equal (a broken kernel or state gives NaN / inf / zeros)."""
    import math

    n_vocab = int(llm.n_vocab())
    logits = llm._ctx.get_logits()
    values = [float(logits[i]) for i in range(n_vocab)]
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
