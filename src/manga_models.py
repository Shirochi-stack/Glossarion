"""On-demand ONNX model manager for manga translation (desktop and Glossarion Mobile).

The manga pipeline runs three kinds of local ONNX models: the RT-DETR comic
bubble/text detector (bubble_detector.BubbleDetector), and the AOT / LaMa
inpainters (local_inpainter.LocalInpainter). None of them is bundled with the
app; they are downloaded from Hugging Face the first time they are needed.

This module is the registry and download manager for those files:

* ``REGISTRY``: every model with its pinned Hugging Face revision, file name,
  sha256 and size. The pins are the LFS object ids Hugging Face reports
  (``X-Linked-ETag``), so a download is verified byte for byte.
* ``download()``: resumable (HTTP ``Range``) download into ``<file>.partial``,
  size + sha256 verification, then an atomic ``os.replace`` into place. It
  reports progress, can be cancelled (the partial file is kept for resuming),
  retries transient network errors and checks free space first.
* ``status()`` / ``statuses()`` / ``disk_usage()`` / ``delete()`` / ``cancel()`` for
  the model manager UI, and ``load()`` / ``unload()`` to keep a downloaded RT-DETR
  detector's session warm (the settings "Load" action).
* ``required_models(config)`` / ``missing_models(config)``: the models a run with that
  config loads, so a job can download them (with progress) before it starts.
* Paths: a model lands exactly where the backend modules look for it, i.e.
  ``<BUBBLE_CACHE_DIR>/<file>`` for detectors (bubble_detector) and
  ``<MODEL_CACHE_DIR>/<file>`` for inpainters (local_inpainter.download_model).
  Desktop defaults are unchanged ('models', '~/.cache/inpainting'). On mobile
  the bootstrap points the three cache variables at ``<data>/models/...``
  (``cache_env()``), and ``default_cache_dir()`` gives the same answer when
  the variables are unset.
* Phone defaults: ``MOBILE_MANGA_SETTINGS_OVERRIDES`` / ``MOBILE_TOP_LEVEL_OVERRIDES``
  (small INT8 detector, AOT inpainting, a lower HD resize limit, one worker),
  applied by ``apply_mobile_defaults()`` only when ``mobile_runtime.is_mobile()``.
  Desktop defaults are never changed.

Stdlib only (plus mobile_runtime); importable on Python 3.10 without the
manga stack (no numpy / cv2 / onnxruntime), so the settings UI can show model
status cheaply.
"""

from __future__ import annotations

import copy
import errno
import hashlib
import http.client
import os
import shutil
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import mobile_runtime

__all__ = [
    "KIND_DETECTOR",
    "KIND_INPAINTER",
    "ModelSpec",
    "ModelStatus",
    "DownloadProgress",
    "ModelDownloadError",
    "ChecksumMismatch",
    "DownloadCancelled",
    "InsufficientSpace",
    "REGISTRY",
    "CACHE_ENV_SUBDIRS",
    "DESKTOP_CACHE_DEFAULTS",
    "MOBILE_DETECTOR_KEY",
    "MOBILE_INPAINTER_KEY",
    "MOBILE_MANGA_SETTINGS_OVERRIDES",
    "MOBILE_TOP_LEVEL_OVERRIDES",
    "get_spec",
    "specs",
    "find_spec",
    "spec_for_detector_variant",
    "spec_for_inpaint_method",
    "required_models",
    "missing_models",
    "mobile_models_root",
    "cache_env",
    "default_cache_dir",
    "cache_dir",
    "model_path",
    "partial_path",
    "is_installed",
    "verify",
    "status",
    "statuses",
    "active_downloads",
    "download",
    "ensure",
    "cancel",
    "delete",
    "disk_usage",
    "DiskUsage",
    "load",
    "unload",
    "loaded",
    "legacy_progress",
    "format_size",
    "mobile_default_overrides",
    "apply_mobile_defaults",
    "apply_mobile_top_level_defaults",
]

KIND_DETECTOR = "detector"
KIND_INPAINTER = "inpaint"

HF_ENDPOINT_DEFAULT = "https://huggingface.co"
USER_AGENT = "Glossarion"
PARTIAL_SUFFIX = ".partial"

# Env variable -> sub-directory of ``<data>/models`` on mobile. bubble_detector reads
# BUBBLE_CACHE_DIR (RT-DETR downloads), local_inpainter reads MODEL_CACHE_DIR
# (inpainting model downloads) and ONNX_CACHE_DIR (ONNX conversions / path fixes),
# all three when they are imported. runtime_bootstrap sets them from cache_env().
CACHE_ENV_SUBDIRS = {
    "BUBBLE_CACHE_DIR": "detector",
    "MODEL_CACHE_DIR": "inpainting",
    "ONNX_CACHE_DIR": "onnx",
}
# The defaults bubble_detector.py / local_inpainter.py use when the variable is unset
# (desktop; expanduser applied at use).
DESKTOP_CACHE_DEFAULTS = {
    "BUBBLE_CACHE_DIR": "models",
    "MODEL_CACHE_DIR": "~/.cache/inpainting",
    "ONNX_CACHE_DIR": "models",
}
MODELS_DIRNAME = "models"


# =========================================================================== registry
@dataclass(frozen=True)
class ModelSpec:
    """One downloadable model file.

    ``selector`` is the settings value that picks the model: the RT-DETR ONNX file
    name (``manga_settings.ocr.rtdetr_onnx_variant``) for detectors, the local
    inpainting method (``manga_local_inpaint_model``) for inpainters.
    """

    key: str
    kind: str
    title: str
    repo_id: str
    filename: str
    revision: str
    sha256: str
    size: int
    cache_env: str
    selector: str
    description: str = ""
    phone_default: bool = False

    @property
    def size_mb(self) -> float:
        return self.size / (1024 * 1024)

    def url(self, endpoint: Optional[str] = None) -> str:
        """``{HF_ENDPOINT}/{repo}/resolve/{revision}/{filename}`` (HF_ENDPOINT honoured like
        bubble_detector.hf_urllib_download)."""
        base = endpoint
        if base is None:
            base = os.environ.get("HF_ENDPOINT", "").strip() or HF_ENDPOINT_DEFAULT
        return "{}/{}/resolve/{}/{}".format(
            base.rstrip("/"),
            urllib.parse.quote(self.repo_id, safe="/"),
            urllib.parse.quote(self.revision, safe=""),
            urllib.parse.quote(self.filename, safe="/"),
        )


_RTDETR_REPO = "ogkalu/comic-text-and-bubble-detector"
_RTDETR_REVISION = "16e8a622f91fabc6b5b65c96d32d1183f8843546"

# Sizes and sha256 are the Hugging Face LFS metadata (api/models/<repo>/tree/<revision>)
# of the pinned revisions; the repo ids and file names match
# BubbleDetector.RTDETR_ONNX_FILENAMES / rtdetr_onnx_repo and local_inpainter.LAMA_JIT_MODELS.
_SPECS = (
    ModelSpec(
        key="rtdetr_v4_s_int8",
        kind=KIND_DETECTOR,
        title="RT-DETR v4-S INT8",
        repo_id=_RTDETR_REPO,
        filename="detector-v4-s_int8.onnx",
        revision=_RTDETR_REVISION,
        sha256="5fe9e4f576e49d4e7e8b0e029d6d3cdc252abd4694113e1cae120e62c931ea79",
        size=11120765,
        cache_env="BUBBLE_CACHE_DIR",
        selector="detector-v4-s_int8.onnx",
        description="Small quantized bubble/text detector. Recommended for phones.",
        phone_default=True,
    ),
    ModelSpec(
        key="rtdetr_int8",
        kind=KIND_DETECTOR,
        title="RT-DETR INT8",
        repo_id=_RTDETR_REPO,
        filename="detector_int8.onnx",
        revision=_RTDETR_REVISION,
        sha256="b5022ad46416b6fe4f88b0cc082cfd2ff5b1cfc624088c2f19879485493f5913",
        size=43838857,
        cache_env="BUBBLE_CACHE_DIR",
        selector="detector_int8.onnx",
        description="Quantized full-size detector.",
    ),
    ModelSpec(
        key="rtdetr",
        kind=KIND_DETECTOR,
        title="RT-DETR (full precision)",
        repo_id=_RTDETR_REPO,
        filename="detector.onnx",
        revision=_RTDETR_REVISION,
        sha256="065744e91c0594ad8663aa8b870ce3fb27222942eded5a3cc388ce23421bd195",
        size=168481531,
        cache_env="BUBBLE_CACHE_DIR",
        selector="detector.onnx",
        description="Full-precision detector (desktop default). Large and slower on phones.",
    ),
    ModelSpec(
        key="aot_onnx",
        kind=KIND_INPAINTER,
        title="AOT ONNX",
        repo_id="ogkalu/aot-inpainting",
        filename="aot.onnx",
        revision="42ffc84ff1bd46dd95f1c5a41e83ee7e98f39189",
        sha256="ffd39ed8e2a275869d3b49180d030f0d8b8b9c2c20ed0e099ecd207201f0eada",
        size=23068213,
        cache_env="MODEL_CACHE_DIR",
        selector="aot_onnx",
        description="Fast, light inpainting model. Recommended for phones.",
        phone_default=True,
    ),
    ModelSpec(
        key="anime_onnx",
        kind=KIND_INPAINTER,
        title="Anime/Manga LaMa ONNX",
        repo_id="ogkalu/lama-manga-onnx-dynamic",
        filename="lama-manga-dynamic.onnx",
        revision="ee4ed4a8447b6730fc41d34f90876b6c48af925a",
        sha256="de31ffa5ba26916b8ea35319f6c12151ff9654d4261bccf0583a69bb095315f9",
        size=206291843,
        cache_env="MODEL_CACHE_DIR",
        selector="anime_onnx",
        description="LaMa tuned for manga (desktop default). Large; needs much more memory than AOT.",
    ),
    ModelSpec(
        key="lama_onnx",
        kind=KIND_INPAINTER,
        title="LaMa ONNX (Carve)",
        repo_id="Carve/LaMa-ONNX",
        filename="lama_fp32.onnx",
        revision="c3c0c9e468934d62e79c329e35d82dd09ff8c444",
        sha256="1faef5301d78db7dda502fe59966957ec4b79dd64e16f03ed96913c7a4eb68d6",
        size=208044816,
        cache_env="MODEL_CACHE_DIR",
        selector="lama_onnx",
        description="General LaMa inpainting model. Large; needs much more memory than AOT.",
    ),
)

REGISTRY: Dict[str, ModelSpec] = {spec.key: spec for spec in _SPECS}

MOBILE_DETECTOR_KEY = "rtdetr_v4_s_int8"
MOBILE_INPAINTER_KEY = "aot_onnx"

ModelRef = Union[str, ModelSpec]


def get_spec(model: ModelRef) -> ModelSpec:
    """The ModelSpec for a registry key (or the spec itself). KeyError when unknown."""
    if isinstance(model, ModelSpec):
        return model
    try:
        return REGISTRY[str(model)]
    except KeyError:
        raise KeyError(f"unknown manga model {model!r}") from None


def specs(kind: Optional[str] = None) -> List[ModelSpec]:
    """Registered models in display order, optionally only one kind."""
    return [spec for spec in _SPECS if kind is None or spec.kind == kind]


def find_spec(repo_id: str, filename: str) -> Optional[ModelSpec]:
    """The registered model for a Hugging Face (repo, file) pair, or None."""
    repo = str(repo_id or "").strip()
    name = str(filename or "").strip()
    for spec in _SPECS:
        if spec.repo_id == repo and spec.filename == name:
            return spec
    return None


def spec_for_detector_variant(onnx_filename: str) -> Optional[ModelSpec]:
    """The detector for an ``rtdetr_onnx_variant`` value (a file name), or None."""
    name = os.path.basename(str(onnx_filename or "").strip())
    for spec in specs(KIND_DETECTOR):
        if spec.selector == name:
            return spec
    return None


def spec_for_inpaint_method(method: str) -> Optional[ModelSpec]:
    """The inpainter for a local inpainting method (aot_onnx / anime_onnx / lama_onnx), or None
    (torch JIT methods, custom-image-edit, ... have no downloadable ONNX file here)."""
    name = str(method or "").strip().lower()
    for spec in specs(KIND_INPAINTER):
        if spec.selector == name:
            return spec
    return None


# =========================================================================== phone defaults
# Applied on top of the canonical manga_settings defaults (manga_settings_dialog
# default_settings / manga_settings_defaults) only when mobile_runtime.is_mobile().
# A user's stored value always wins; these only replace *defaults*.
#  * detector: the 11 MB INT8 v4-S export instead of the 168 MB detector.onnx;
#  * inpainting: AOT ONNX (23 MB) instead of the 206 MB anime LaMa;
#  * HD resize limit 1024 (desktop 1536): bounds inpainting memory on 2-4 GB phones;
#  * single-worker concurrency for on-device compute (region workers, and panel workers even
#    when parallel panels are switched on; RT-DETR runs one page at a time then, since
#    bubble_detector ties its session semaphore to panel_max_workers or defaults it to 1),
#    and models unloaded after a batch to give the memory back.
#  * ocr.rtdetr_max_concurrency is deliberately NOT overridden: manga_translator uses it as
#    the number of parallel cloud OCR calls per page (Google Vision region OCR, default 12),
#    which is network-bound, so lowering it would only slow the phone's default (cloud) OCR.
MOBILE_MANGA_SETTINGS_OVERRIDES: Dict[str, Any] = {
    "ocr": {
        "rtdetr_onnx_variant": REGISTRY[MOBILE_DETECTOR_KEY].selector,
    },
    "advanced": {
        "hd_strategy_resize_limit": 1024,
        "max_workers": 1,
        "panel_max_workers": 1,
        "unload_models_after_translation": True,
    },
    "inpainting": {
        "local_method": REGISTRY[MOBILE_INPAINTER_KEY].selector,
    },
}
MOBILE_TOP_LEVEL_OVERRIDES: Dict[str, Any] = {
    "manga_local_inpaint_model": REGISTRY[MOBILE_INPAINTER_KEY].selector,
}


def _flatten(prefix: str, value: Any, out: Dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _flatten(f"{prefix}.{key}" if prefix else str(key), item, out)
    else:
        out[prefix] = value


def mobile_default_overrides() -> Dict[str, Any]:
    """Every phone default as a dotted settings key -> value
    (``manga_settings.advanced.hd_strategy_resize_limit`` -> 1024, ``manga_local_inpaint_model`` -> 'aot_onnx')."""
    out: Dict[str, Any] = {}
    _flatten("manga_settings", MOBILE_MANGA_SETTINGS_OVERRIDES, out)
    out.update(MOBILE_TOP_LEVEL_OVERRIDES)
    return out


def _deep_merge(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def apply_mobile_defaults(manga_settings_defaults: Dict[str, Any], *, force: bool = False) -> Dict[str, Any]:
    """A deep copy of a canonical ``manga_settings`` *defaults* dict with the phone
    defaults layered on, when ``mobile_runtime.is_mobile()`` (or ``force``). On desktop
    the copy is returned unchanged. Merge the user's stored settings over the result."""
    result = copy.deepcopy(dict(manga_settings_defaults or {}))
    if force or mobile_runtime.is_mobile():
        _deep_merge(result, MOBILE_MANGA_SETTINGS_OVERRIDES)
    return result


def apply_mobile_top_level_defaults(top_level_defaults: Dict[str, Any], *, force: bool = False) -> Dict[str, Any]:
    """Same as apply_mobile_defaults for the top-level ``manga_*`` defaults."""
    result = copy.deepcopy(dict(top_level_defaults or {}))
    if force or mobile_runtime.is_mobile():
        result.update(copy.deepcopy(MOBILE_TOP_LEVEL_OVERRIDES))
    return result


def _default_for(path: tuple, desktop_default: Any) -> Any:
    if mobile_runtime.is_mobile():
        node: Any = MOBILE_MANGA_SETTINGS_OVERRIDES
        for part in path:
            if not isinstance(node, dict) or part not in node:
                return desktop_default
            node = node[part]
        return node
    return desktop_default


def required_models(config: Optional[Dict[str, Any]]) -> List[ModelSpec]:
    """The registered models a manga run with ``config`` loads.

    Mirrors the lookups of the run path: the RT-DETR ONNX detector when bubble
    detection is on with ``detector_type`` 'rtdetr_onnx'; the local ONNX inpainter
    when inpainting is not skipped, the method is 'local' and no model file of the
    user's own is configured (``manga_<method>_model_path``). Missing values take the
    desktop GUI's defaults, or the phone defaults on mobile.
    """
    config = config if isinstance(config, dict) else {}
    manga = config.get("manga_settings") if isinstance(config.get("manga_settings"), dict) else {}
    out: List[ModelSpec] = []

    ocr = manga.get("ocr") if isinstance(manga.get("ocr"), dict) else {}
    detection_on = ocr.get("bubble_detection_enabled", True)
    detector_type = ocr.get("detector_type", _default_for(("ocr", "detector_type"), "rtdetr_onnx"))
    if detection_on and str(detector_type or "") == "rtdetr_onnx":
        variant = ocr.get("rtdetr_onnx_variant") or _default_for(("ocr", "rtdetr_onnx_variant"), "detector.onnx")
        spec = spec_for_detector_variant(variant)
        if spec is not None:
            out.append(spec)

    if not config.get("manga_skip_inpainting", False):
        inpainting = manga.get("inpainting") if isinstance(manga.get("inpainting"), dict) else {}
        method = config.get("manga_inpaint_method") or inpainting.get("method") or "local"
        if str(method) == "local":
            default_local = (MOBILE_TOP_LEVEL_OVERRIDES["manga_local_inpaint_model"]
                             if mobile_runtime.is_mobile() else "anime_onnx")
            local = config.get("manga_local_inpaint_model") or inpainting.get("local_method") or default_local
            spec = spec_for_inpaint_method(local)
            custom = str(config.get(f"manga_{local}_model_path") or config.get(f"{local}_model_path") or "")
            if spec is not None and not (custom and os.path.isfile(custom)):
                out.append(spec)
    return out


def missing_models(config: Optional[Dict[str, Any]]) -> List[ModelSpec]:
    """required_models(config) that are not downloaded yet."""
    return [spec for spec in required_models(config) if not is_installed(spec)]


# =========================================================================== paths
def mobile_models_root(data_dir: Optional[str] = None) -> Optional[str]:
    """``<data>/models`` (data = ``data_dir`` or GLOSSARION_DATA_DIR), or None without a data dir."""
    data = str(data_dir if data_dir is not None else os.environ.get("GLOSSARION_DATA_DIR", "")).strip()
    return os.path.join(data, MODELS_DIRNAME) if data else None


def cache_env(models_root: Union[str, "os.PathLike[str]"]) -> Dict[str, str]:
    """BUBBLE_CACHE_DIR / MODEL_CACHE_DIR / ONNX_CACHE_DIR under ``models_root`` (the mobile
    bootstrap sets these before any manga module is imported)."""
    root = os.fspath(models_root)
    return {name: os.path.join(root, sub) for name, sub in CACHE_ENV_SUBDIRS.items()}


def default_cache_dir(env_name: str, desktop_default: Optional[str] = None) -> str:
    """The directory a backend module uses for ``env_name`` when the variable is unset.

    Desktop: ``desktop_default`` (or DESKTOP_CACHE_DEFAULTS), unchanged. Mobile with a
    data dir: ``<data>/models/<sub>`` so a model never lands in the process cwd.
    """
    if env_name not in CACHE_ENV_SUBDIRS:
        raise KeyError(f"not a model cache variable: {env_name!r}")
    if desktop_default is None:
        desktop_default = os.path.expanduser(DESKTOP_CACHE_DEFAULTS[env_name])
    if mobile_runtime.is_mobile():
        root = mobile_models_root()
        if root:
            return os.path.join(root, CACHE_ENV_SUBDIRS[env_name])
    return desktop_default


def cache_dir(env_name: str) -> str:
    """The cache directory in effect: ``os.environ.get(env_name, default_cache_dir(env_name))``,
    the same expression bubble_detector / local_inpainter evaluate."""
    return os.environ.get(env_name, default_cache_dir(env_name))


def model_path(model: ModelRef, dest_dir: Optional[str] = None) -> str:
    """Where the model file lives (and where the backend module looks for it)."""
    spec = get_spec(model)
    base = dest_dir if dest_dir else cache_dir(spec.cache_env)
    return os.path.join(base, *spec.filename.split("/"))


def partial_path(model: ModelRef, dest_dir: Optional[str] = None) -> str:
    """The resumable partial download next to ``model_path`` (``<file>.partial``)."""
    return model_path(model, dest_dir) + PARTIAL_SUFFIX


def _size(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def is_installed(model: ModelRef, dest_dir: Optional[str] = None, *, verify_hash: bool = False) -> bool:
    """True when the model file exists with the expected size (and sha256 when ``verify_hash``)."""
    spec = get_spec(model)
    path = model_path(spec, dest_dir)
    if not os.path.isfile(path) or _size(path) != spec.size:
        return False
    return verify(spec, dest_dir) if verify_hash else True


def _sha256_file(path: str, cancel: Any = None, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            if _cancelled(cancel):
                raise DownloadCancelled("verification cancelled")
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def verify(model: ModelRef, dest_dir: Optional[str] = None) -> bool:
    """Full sha256 check of the installed file (False when missing or different)."""
    spec = get_spec(model)
    path = model_path(spec, dest_dir)
    if not os.path.isfile(path) or _size(path) != spec.size:
        return False
    try:
        return _sha256_file(path) == spec.sha256
    except OSError:
        return False


# =========================================================================== status
@dataclass(frozen=True)
class DownloadProgress:
    """A progress report. ``phase`` is 'download', 'verify' or 'done'."""

    key: str
    downloaded: int
    total: int
    bytes_per_sec: float = 0.0
    resumed_from: int = 0
    phase: str = "download"

    @property
    def fraction(self) -> float:
        return min(1.0, self.downloaded / self.total) if self.total > 0 else 0.0

    @property
    def percent(self) -> int:
        return int(self.fraction * 100)


@dataclass(frozen=True)
class ModelStatus:
    key: str
    kind: str
    title: str
    path: str
    size: int                     # expected size in bytes
    installed: bool
    bytes_on_disk: int            # installed file + partial download
    partial_bytes: int
    downloading: Optional[DownloadProgress] = None
    phone_default: bool = False

    @property
    def state(self) -> str:
        """'downloading', 'installed', 'partial' (resumable) or 'missing'."""
        if self.downloading is not None:
            return "downloading"
        if self.installed:
            return "installed"
        if self.partial_bytes:
            return "partial"
        return "missing"


_STATE_LOCK = threading.Lock()
_PATH_LOCKS: Dict[str, threading.Lock] = {}
_ACTIVE: Dict[str, DownloadProgress] = {}
# key -> the cancel events of every download() call for that model (a second call waits on the
# path lock while the first downloads; cancel(key) stops both)
_CANCELS: Dict[str, List[threading.Event]] = {}
_LOADED: Dict[str, Any] = {}


def _path_lock(path: str) -> threading.Lock:
    key = os.path.normcase(os.path.abspath(path))
    with _STATE_LOCK:
        lock = _PATH_LOCKS.get(key)
        if lock is None:
            lock = _PATH_LOCKS[key] = threading.Lock()
        return lock


def _set_active(key: str, progress: Optional[DownloadProgress]) -> None:
    with _STATE_LOCK:
        if progress is None:
            _ACTIVE.pop(key, None)
        else:
            _ACTIVE[key] = progress


def active_downloads() -> Dict[str, DownloadProgress]:
    """Downloads running in this process: key -> latest progress."""
    with _STATE_LOCK:
        return dict(_ACTIVE)


def status(model: ModelRef, dest_dir: Optional[str] = None) -> ModelStatus:
    spec = get_spec(model)
    path = model_path(spec, dest_dir)
    installed = is_installed(spec, dest_dir)
    partial = _size(path + PARTIAL_SUFFIX)
    with _STATE_LOCK:
        running = _ACTIVE.get(spec.key)
    return ModelStatus(
        key=spec.key,
        kind=spec.kind,
        title=spec.title,
        path=path,
        size=spec.size,
        installed=installed,
        bytes_on_disk=(_size(path) if os.path.isfile(path) else 0) + partial,
        partial_bytes=partial,
        downloading=running,
        phone_default=spec.phone_default,
    )


def statuses(kind: Optional[str] = None) -> List[ModelStatus]:
    return [status(spec) for spec in specs(kind)]


def format_size(num_bytes: int) -> str:
    """'11 MB', '206 MB', '1.2 GB', '512 KB'."""
    value = float(max(0, int(num_bytes or 0)))
    for unit, step in (("GB", 1024 ** 3), ("MB", 1024 ** 2), ("KB", 1024)):
        if value >= step:
            scaled = value / step
            return f"{scaled:.1f} {unit}" if scaled < 10 else f"{scaled:.0f} {unit}"
    return f"{int(value)} B"


# =========================================================================== download
class ModelDownloadError(Exception):
    """A model could not be downloaded (network, HTTP status, disk)."""


class ChecksumMismatch(ModelDownloadError):
    """The downloaded bytes are not the pinned file (size or sha256); nothing is kept."""


class DownloadCancelled(ModelDownloadError):
    """The download was cancelled; the partial file is kept for resuming."""


class InsufficientSpace(ModelDownloadError):
    """Not enough free storage for the remaining bytes."""


class _IncompleteDownload(OSError):
    """The connection ended before the whole file arrived (retried with Range)."""


class _RangeMismatch(Exception):
    """The server answered a Range request with another range (restart from zero)."""


def _cancelled(cancel: Any) -> bool:
    if cancel is None:
        return False
    try:
        if hasattr(cancel, "is_set"):
            return bool(cancel.is_set())
        return bool(cancel())
    except Exception:
        return False


def _wait(cancel: Any, seconds: float) -> None:
    deadline = time.monotonic() + max(0.0, seconds)
    while time.monotonic() < deadline:
        if _cancelled(cancel):
            raise DownloadCancelled("download cancelled")
        time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))


def _parse_content_range(value: Optional[str]):
    """'bytes 100-199/1000' -> (100, 199, 1000); total None for '*'."""
    text = str(value or "").strip()
    if not text.lower().startswith("bytes "):
        return None
    try:
        span, _, total = text[6:].partition("/")
        start, _, end = span.partition("-")
        return int(start), int(end), (None if total.strip() == "*" else int(total))
    except ValueError:
        return None


def _remove(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def _retryable_http(code: int) -> bool:
    return code in (408, 425, 429) or code >= 500


def _report(progress: Optional[Callable[[DownloadProgress], Any]], report: DownloadProgress) -> None:
    _set_active(report.key, report)
    if progress is None:
        return
    try:
        progress(report)
    except Exception:
        pass


def _check_space(directory: str, needed: int, margin: int = 32 * 1024 * 1024) -> None:
    try:
        free = shutil.disk_usage(directory).free
    except OSError:
        return
    if free < needed + margin:
        raise InsufficientSpace(
            f"Not enough free storage: {format_size(needed)} needed, {format_size(free)} free"
        )


def _fetch(spec: ModelSpec, partial: str, url: str, progress, cancel, timeout: float,
           chunk_size: int, throttle: float) -> None:
    """One HTTP attempt: resume ``partial`` with Range when it has bytes, else start over."""
    offset = _size(partial) if os.path.isfile(partial) else 0
    if offset > spec.size:
        _remove(partial)
        offset = 0
    if offset == spec.size:
        return  # complete on disk; the caller verifies it
    headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "identity"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    try:
        response = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and offset:
            raise _RangeMismatch(f"HTTP 416 for bytes={offset}-") from exc
        raise
    with response:
        status_code = getattr(response, "status", None) or response.getcode()
        mode = "wb"
        if offset and status_code == 206:
            parsed = _parse_content_range(response.headers.get("Content-Range"))
            if parsed is None or parsed[0] != offset:
                raise _RangeMismatch(f"Content-Range {response.headers.get('Content-Range')!r} for bytes={offset}-")
            if parsed[2] is not None and parsed[2] != spec.size:
                _remove(partial)
                raise ChecksumMismatch(
                    f"{spec.filename}: the server has {parsed[2]} bytes, expected {spec.size} (file changed upstream?)"
                )
            mode = "ab"
        else:
            offset = 0  # 200: the server sent the whole file (Range ignored)
        length = (response.headers.get("Content-Length") or "").strip()
        if length.isdigit() and offset + int(length) != spec.size:
            _remove(partial)
            raise ChecksumMismatch(
                f"{spec.filename}: the server has {offset + int(length)} bytes, expected {spec.size}"
            )
        downloaded = offset
        started = time.monotonic()
        last = 0.0
        _report(progress, DownloadProgress(spec.key, downloaded, spec.size, 0.0, offset))
        with open(partial, mode) as fh:
            while True:
                if _cancelled(cancel):
                    raise DownloadCancelled(f"{spec.filename}: download cancelled at {downloaded} bytes")
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                fh.write(chunk)
                downloaded += len(chunk)
                if downloaded > spec.size:
                    fh.close()
                    _remove(partial)
                    raise ChecksumMismatch(f"{spec.filename}: received more than {spec.size} bytes")
                now = time.monotonic()
                if now - last >= throttle or downloaded == spec.size:
                    last = now
                    speed = (downloaded - offset) / max(now - started, 1e-6)
                    _report(progress, DownloadProgress(spec.key, downloaded, spec.size, speed, offset))
        if downloaded < spec.size:
            raise _IncompleteDownload(f"{spec.filename}: connection closed at {downloaded} of {spec.size} bytes")


def download(
    model: ModelRef,
    *,
    dest_dir: Optional[str] = None,
    progress: Optional[Callable[[DownloadProgress], Any]] = None,
    cancel: Any = None,
    url: Optional[str] = None,
    timeout: float = 60.0,
    retries: int = 3,
    chunk_size: int = 256 * 1024,
    throttle: float = 0.1,
    backoff: float = 1.0,
) -> str:
    """Download (or resume) a model and return its verified path.

    An installed file of the right size is returned without network access. Bytes go
    to ``<file>.partial``; an interrupted or cancelled download keeps it and the next
    call resumes with ``Range``. After the last byte the size and sha256 are checked
    and the file is renamed into place atomically. ``cancel`` is a threading.Event or
    a callable; ``progress`` receives DownloadProgress reports (throttled).

    Raises DownloadCancelled (partial kept), ChecksumMismatch (partial deleted),
    InsufficientSpace, or ModelDownloadError (network/HTTP after ``retries`` consecutive
    failed attempts; an attempt that received bytes before the connection dropped resets
    the count, so a slow, flaky phone connection still finishes a large model).
    """
    spec = get_spec(model)
    target = model_path(spec, dest_dir)
    partial = target + PARTIAL_SUFFIX
    source = url or spec.url()
    own_cancel = threading.Event()  # cancel(key) from another thread (e.g. a job's Stop)

    def stop_requested() -> bool:
        return own_cancel.is_set() or _cancelled(cancel)

    with _STATE_LOCK:
        _CANCELS.setdefault(spec.key, []).append(own_cancel)
    try:
        with _path_lock(target):
            if is_installed(spec, dest_dir):
                _report(progress, DownloadProgress(spec.key, spec.size, spec.size, 0.0, spec.size, "done"))
                return target
            return _download_locked(spec, target, partial, source, progress, stop_requested, timeout,
                                    retries, chunk_size, throttle, backoff)
    finally:
        with _STATE_LOCK:
            events = _CANCELS.get(spec.key)
            if events is not None and own_cancel in events:
                events.remove(own_cancel)
                if not events:
                    del _CANCELS[spec.key]
            _ACTIVE.pop(spec.key, None)


def _download_locked(spec: ModelSpec, target: str, partial: str, source: str, progress, cancel,
                     timeout: float, retries: int, chunk_size: int, throttle: float, backoff: float) -> str:
    """download() body; the caller holds the path lock."""
    directory = os.path.dirname(target) or "."
    os.makedirs(directory, exist_ok=True)
    if _cancelled(cancel):
        raise DownloadCancelled(f"{spec.filename}: download cancelled")
    _check_space(directory, max(0, spec.size - (_size(partial) if os.path.isfile(partial) else 0)))
    failures = 0
    restarted = False
    while True:
        before = _size(partial) if os.path.isfile(partial) else 0
        try:
            _fetch(spec, partial, source, progress, cancel, timeout, chunk_size, throttle)
            break
        except (DownloadCancelled, ChecksumMismatch):
            raise
        except _RangeMismatch:
            _remove(partial)  # the partial does not match what the server serves now
            if restarted:
                raise ModelDownloadError(f"{spec.filename}: the server does not resume consistently")
            restarted = True
        except urllib.error.HTTPError as exc:
            if not _retryable_http(exc.code):
                raise ModelDownloadError(f"{spec.filename}: HTTP {exc.code} {exc.reason}") from exc
            failures += 1
            if failures > retries:
                raise ModelDownloadError(f"{spec.filename}: HTTP {exc.code} {exc.reason}") from exc
            _wait(cancel, backoff * failures)
        except (urllib.error.URLError, http.client.HTTPException, socket.timeout, OSError) as exc:
            if getattr(exc, "errno", None) == errno.ENOSPC:
                raise InsufficientSpace(f"{spec.filename}: storage is full") from exc
            if (_size(partial) if os.path.isfile(partial) else 0) > before:
                failures = 0  # the attempt made progress: only consecutive stalls count
            failures += 1
            if failures > retries:
                reason = getattr(exc, "reason", None) or exc
                raise ModelDownloadError(f"{spec.filename}: download failed: {reason}") from exc
            _wait(cancel, backoff * failures)
    _report(progress, DownloadProgress(spec.key, spec.size, spec.size, 0.0, 0, "verify"))
    received = _size(partial)
    if received != spec.size:
        _remove(partial)
        raise ChecksumMismatch(f"{spec.filename}: size {received} != {spec.size}")
    digest = _sha256_file(partial, cancel)
    if digest != spec.sha256:
        _remove(partial)
        raise ChecksumMismatch(f"{spec.filename}: sha256 {digest} != {spec.sha256}")
    os.replace(partial, target)
    _report(progress, DownloadProgress(spec.key, spec.size, spec.size, 0.0, 0, "done"))
    return target


def cancel(model: ModelRef) -> bool:
    """Cancel every download() call for ``model`` in this process, running or waiting for
    the running one (the partial file is kept). Returns False when none is running."""
    spec = get_spec(model)
    with _STATE_LOCK:
        events = list(_CANCELS.get(spec.key) or ())
    for event in events:
        event.set()
    return bool(events)


def ensure(models: Iterable[ModelRef], **kwargs: Any) -> List[str]:
    """download() every model that is not installed yet; returns their paths in order."""
    return [download(model, **kwargs) for model in models]


def legacy_progress(callback: Optional[Callable[..., Any]]) -> Optional[Callable[[DownloadProgress], None]]:
    """Adapt a ``callback(percent, downloaded_mb, total_mb, speed_mb)`` (local_inpainter /
    bubble_detector download signature) to download()'s progress reports."""
    if callback is None:
        return None
    mb = 1024 * 1024

    def adapter(report: DownloadProgress) -> None:
        callback(report.percent, report.downloaded / mb, report.total / mb, report.bytes_per_sec / mb)

    return adapter


# =========================================================================== delete / disk usage
def delete(model: ModelRef, dest_dir: Optional[str] = None, *, include_partial: bool = True) -> int:
    """Remove a downloaded model (and its partial download); returns the bytes freed.
    Refuses while the model is downloading in this process."""
    spec = get_spec(model)
    with _STATE_LOCK:
        if spec.key in _ACTIVE:
            raise ModelDownloadError(f"{spec.title} is downloading; cancel the download first")
    target = model_path(spec, dest_dir)
    freed = 0
    paths = [target] + ([target + PARTIAL_SUFFIX] if include_partial else [])
    with _path_lock(target):
        for path in paths:
            if os.path.isfile(path):
                size = _size(path)
                try:
                    os.remove(path)
                except OSError as exc:
                    raise ModelDownloadError(f"Could not delete {os.path.basename(path)}: {exc}") from exc
                freed += size
    return freed


def _tree_size(path: str) -> int:
    total = 0
    if os.path.isfile(path):
        return _size(path)
    for root, _dirs, files in os.walk(path):
        for name in files:
            total += _size(os.path.join(root, name))
    return total


class DiskUsage(dict):
    """disk_usage() result: a dict that also converts to its ``total_bytes`` with int()."""

    def __int__(self) -> int:
        return int(self.get("total_bytes") or 0)


def disk_usage() -> "DiskUsage":
    """Storage used by the model caches.

    ``{"total_bytes": n, "models": {key: bytes}, "dirs": {ENV: {"path": p, "bytes": n}}}``
    (``int(disk_usage())`` is the total); ``dirs`` counts every file in each cache dir
    (conversions, config.json, partials), and ``total_bytes`` counts a directory shared by
    two variables once.
    """
    models = {spec.key: status(spec).bytes_on_disk for spec in _SPECS}
    dirs: Dict[str, Dict[str, Any]] = {}
    seen: Dict[str, int] = {}
    for env_name in CACHE_ENV_SUBDIRS:
        path = cache_dir(env_name)
        real = os.path.normcase(os.path.realpath(path))
        if real not in seen:
            seen[real] = _tree_size(path) if os.path.isdir(path) else 0
        dirs[env_name] = {"path": path, "bytes": seen[real]}
    # a cache dir nested in another (e.g. 'models' and 'models/x') is counted once
    roots = sorted(seen)
    total = 0
    for index, real in enumerate(roots):
        if any(real.startswith(other.rstrip(os.sep) + os.sep) for other in roots[:index]):
            continue
        total += seen[real]
    return DiskUsage(total_bytes=total, models=models, dirs=dirs)


# =========================================================================== load / unload (detectors)
def load(model: ModelRef, *, config_path: Optional[str] = None) -> bool:
    """Warm up a downloaded RT-DETR detector (the settings "Load" action).

    Opens its onnxruntime session through bubble_detector.BubbleDetector, whose
    class-level shared session the next run reuses (C++ backend on desktop, Python
    onnxruntime on mobile). Inpainting models are loaded by the run's inpainter pool,
    so they cannot be preloaded here. Raises ModelDownloadError when the model is not
    downloaded or fails to load.
    """
    spec = get_spec(model)
    if spec.kind != KIND_DETECTOR:
        raise ModelDownloadError(f"{spec.title} is loaded by the inpainter when a run starts")
    if not is_installed(spec):
        raise ModelDownloadError(f"{spec.title} is not downloaded")
    import bubble_detector  # heavy (numpy / cv2 / onnxruntime): only when asked

    detector = bubble_detector.BubbleDetector(config_path=config_path or os.environ.get("CONFIG_FILE") or "config.json")
    if not detector.load_rtdetr_onnx_model(model_id=spec.repo_id, onnx_filename=spec.filename):
        raise ModelDownloadError(f"{spec.title} could not be loaded")
    with _STATE_LOCK:
        _LOADED[spec.key] = detector
    return True


def unload(model: ModelRef) -> bool:
    """Release a detector loaded by load() (its shared session included). False when not loaded."""
    spec = get_spec(model)
    with _STATE_LOCK:
        detector = _LOADED.pop(spec.key, None)
    if detector is None:
        return False
    try:
        detector.unload(release_shared=True)
    except Exception:
        pass
    return True


def loaded() -> List[str]:
    """Keys of the detectors load() keeps warm."""
    with _STATE_LOCK:
        return list(_LOADED)
