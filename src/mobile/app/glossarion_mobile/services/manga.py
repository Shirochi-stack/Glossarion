"""MangaService: the UI-free side of Tools › Manga (UI_SPEC §4.6) and of the manga job adapters.

A binding layer only. Every desktop rule comes from the U8 shared cores, imported lazily
(``core(name)``), so the app starts without them and a missing core turns into a disabled
action with a reason (``MISSING_CORE``) instead of an ImportError:

* ``manga_files_core`` — the Files tab model moved verbatim out of ``MangaTranslationTab``
  (natural sort, skip keys, image range, process groups, CBZ extract / create, the
  ``manga_selected_files`` / ``manga_skipped_processing_files`` / ``manga_selected_folder_roots``
  persistence). Its methods take the tab as ``self``; ``MangaFileList`` is that ``self`` on
  mobile (the attributes the moved methods read, no-op GUI hooks);
* ``manga_settings_defaults`` — canonical ``manga_settings`` + top-level ``manga_*`` defaults
  (``merge_manga_settings(config)``);
* ``manga_models`` — the on-demand ONNX model registry / downloader (RT-DETR, AOT / anime /
  LaMa ONNX inpainting) into ``BUBBLE_CACHE_DIR`` / ``ONNX_CACHE_DIR`` / ``MODEL_CACHE_DIR``;
* ``manga_editor_core`` — the editor pipeline (detect / clean / recognize / translate / render)
  and the per-image state (``ImageStateManager`` without its worker process on mobile);
* ``manga_ocr_io`` — the portable ``glossarion-manga-ocr`` JSON (import / export);
* ``settings_schema`` — mobile availability of the manga combo values (torch-only OCR
  providers, detectors and inpainters show disabled with the reason).

The status chips (OCR provider, inpainting method) restate the desktop status labels of
``MangaTranslationTab._check_provider_status`` / the inpainting group (which keys must be set,
which file must exist); they are display only. Runs never read them: the job builds its
OCR config and environment from the config snapshot with ``manga_env`` on the job thread.

Pure Python (no Flet), Python 3.10 compatible.
"""

from __future__ import annotations

import copy
import importlib
import json
import logging
import os
import re
import threading
import time
import zipfile
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

__all__ = [
    "BOX_STEPS",
    "DETECTORS",
    "FileGroup",
    "IMAGE_EXTENSIONS",
    "INPAINT_METHODS",
    "KIND_BATCH",
    "KIND_STEP",
    "K_AZURE_ENDPOINT",
    "K_AZURE_KEY",
    "K_CONSOLIDATE",
    "K_CREATE_CBZ",
    "K_DOCINTEL_ENDPOINT",
    "K_DOCINTEL_KEY",
    "K_EDIT_ENDPOINT",
    "K_FOLDER_ROOTS",
    "K_GOOGLE_CLOUD_CREDS",
    "K_GOOGLE_CREDS",
    "K_INPAINT_METHOD",
    "K_LOCAL_MODEL",
    "K_PROVIDER",
    "K_REPLICATE_KEY",
    "K_SELECTED",
    "K_SKIPPED",
    "K_SKIP_INPAINT",
    "K_SPLIT_SUBFOLDERS",
    "K_USE_EDIT_ENDPOINT",
    "LOCAL_INPAINT_MODELS",
    "MISSING_CORE",
    "MangaFileList",
    "ModelEntry",
    "ModelManager",
    "NOT_IN_BUILD",
    "OCR_PROVIDERS",
    "ONNX_INPAINT_MODELS",
    "OptionRow",
    "PresetsBusy",
    "P_BUBBLE_DETECTION",
    "P_DETECTOR",
    "P_INPAINT_METHOD",
    "P_LOCAL_METHOD",
    "P_RTDETR_VARIANT",
    "STEPS",
    "STEP_LABELS",
    "TOOL_ROUTE",
    "azure_docintel_backend",
    "batch_spec",
    "cached_font_preset_updates",
    "chip_text",
    "core",
    "core_attr",
    "current_inpaint_choice",
    "default_manga_settings",
    "detector_rows",
    "display_copy",
    "editor_session",
    "effective_setting",
    "font_preset_updates",
    "google_backend",
    "importable",
    "inpaint_choice_updates",
    "inpaint_method_rows",
    "inpaint_status",
    "list_ocr_files",
    "local_model_rows",
    "merged_manga_settings",
    "natural_sort_key",
    "new_editor_session",
    "now_stamp",
    "ocr_provider_rows",
    "presets_available",
    "rapidocr_reason",
    "register_editor_session",
    "rendering_reset_updates",
    "step_spec",
    "test_image_edit_endpoint",
    "top_level_defaults",
    "value_reason",
    "zip_images",
]

log = logging.getLogger("glossarion.manga")

KIND_BATCH = "manga"
KIND_STEP = "manga_step"
TOOL_ROUTE = "tools.manga"

#: Desktop "Select Manga Images or CBZ" filter (``_add_files``) and the CBZ / ZIP archives.
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp")

MISSING_CORE = "The shared manga module is not in this build"
NOT_IN_BUILD = "Not in this build"

# ---------------------------------------------------------------------------
# Shared cores
# ---------------------------------------------------------------------------


def core(name: str) -> Any:
    """The shared module ``name`` or None (never raises; a broken import is logged once)."""
    try:
        return importlib.import_module(name)
    except Exception as exc:  # ImportError, or a core that fails at import on this platform
        _note_missing(name, exc)
        return None


_MISSING_LOGGED: set = set()


def _note_missing(name: str, exc: BaseException) -> None:
    if name not in _MISSING_LOGGED:
        _MISSING_LOGGED.add(name)
        log.info("shared module %s unavailable: %s", name, exc)


def core_attr(module_name: str, *names: str) -> Any:
    """The first attribute of ``names`` the shared module defines (None when absent)."""
    module = core(module_name)
    if module is None:
        return None
    for name in names:
        value = getattr(module, name, None)
        if value is not None:
            return value
    return None


def importable(name: str) -> bool:
    """Whether ``import name`` works (a ``sys.modules[name] = None`` block counts as missing)."""
    import sys

    if name in sys.modules:
        return sys.modules[name] is not None
    try:
        import importlib.util

        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _is_mobile() -> bool:
    runtime = core("mobile_runtime")
    try:
        return bool(runtime.is_mobile()) if runtime is not None else False
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Settings access (config snapshot + canonical defaults)
# ---------------------------------------------------------------------------

#: Config keys (desktop names; the Settings tab writes them like ``_save_rendering_settings``).
K_PROVIDER = "manga_ocr_provider"
K_GOOGLE_CREDS = "google_vision_credentials"
K_GOOGLE_CLOUD_CREDS = "google_cloud_credentials"
K_AZURE_KEY = "azure_vision_key"
K_AZURE_ENDPOINT = "azure_vision_endpoint"
K_DOCINTEL_KEY = "azure_document_intelligence_key"
K_DOCINTEL_ENDPOINT = "azure_document_intelligence_endpoint"
K_INPAINT_METHOD = "manga_inpaint_method"
K_LOCAL_MODEL = "manga_local_inpaint_model"
K_SKIP_INPAINT = "manga_skip_inpainting"
K_REPLICATE_KEY = "replicate_api_key"
K_EDIT_ENDPOINT = "custom_image_edit_endpoint"
K_USE_EDIT_ENDPOINT = "use_custom_image_edit_endpoint"
K_SELECTED = "manga_selected_files"
K_SKIPPED = "manga_skipped_processing_files"
K_FOLDER_ROOTS = "manga_selected_folder_roots"
K_SPLIT_SUBFOLDERS = "manga_split_first_level_subfolders"
K_CREATE_CBZ = "manga_create_cbz_at_end"
K_CONSOLIDATE = "manga_auto_consolidate_images"
P_DETECTOR = ("manga_settings", "ocr", "detector_type")
P_RTDETR_VARIANT = ("manga_settings", "ocr", "rtdetr_onnx_variant")
P_BUBBLE_DETECTION = ("manga_settings", "ocr", "bubble_detection_enabled")
P_INPAINT_METHOD = ("manga_settings", "inpainting", "method")
P_LOCAL_METHOD = ("manga_settings", "inpainting", "local_method")


def _lookup(config: Mapping[str, Any], path: Sequence[str]) -> Any:
    node: Any = config
    for part in path:
        if not isinstance(node, Mapping) or part not in node:
            return _MISSING
        node = node[part]
    return node


class _MissingType:
    def __repr__(self) -> str:
        return "MISSING"

    def __bool__(self) -> bool:
        return False


_MISSING: Any = _MissingType()


def default_manga_settings() -> dict:
    """The canonical ``manga_settings`` defaults (``manga_settings_defaults``) with the phone
    defaults of ``manga_models`` layered on when the app runs on mobile (desktop: unchanged)."""
    defaults_fn = core_attr("manga_settings_defaults", "default_manga_settings")
    defaults = defaults_fn() if callable(defaults_fn) else {}
    apply_mobile = core_attr("manga_models", "apply_mobile_defaults")
    if callable(apply_mobile):
        try:
            defaults = apply_mobile(defaults)
        except Exception:
            log.debug("apply_mobile_defaults failed", exc_info=True)
    return dict(defaults or {})


def top_level_defaults() -> dict:
    """The top-level ``manga_*`` defaults the manga tab runs with (+ the phone defaults)."""
    defaults = core_attr("manga_settings_defaults", "MANGA_TOP_LEVEL_DEFAULTS")
    defaults = copy.deepcopy(dict(defaults)) if isinstance(defaults, Mapping) else {}
    apply_mobile = core_attr("manga_models", "apply_mobile_top_level_defaults")
    if callable(apply_mobile):
        try:
            defaults = dict(apply_mobile(defaults))
        except Exception:
            log.debug("apply_mobile_top_level_defaults failed", exc_info=True)
    return defaults


def _deep_merge(base: dict, update: Mapping) -> dict:
    deep_merge = core_attr("manga_settings_defaults", "deep_merge")
    if callable(deep_merge):
        return deep_merge(base, copy.deepcopy(dict(update)))
    for key, value in update.items():  # pragma: no cover - without the shared module
        if isinstance(base.get(key), dict) and isinstance(value, Mapping):
            _deep_merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def merged_manga_settings(config: Mapping[str, Any]) -> dict:
    """``config['manga_settings']`` over ``default_manga_settings()`` (``merge_manga_settings`` order)."""
    stored = (config or {}).get("manga_settings")
    return _deep_merge(default_manga_settings(), stored if isinstance(stored, Mapping) else {})


def effective_setting(config: Mapping[str, Any], key: Any, default: Any = None) -> Any:
    """A stored value, else the canonical manga default, else ``default``.

    ``key`` is a top-level key or a path tuple (``("manga_settings", "ocr", "detector_type")``).
    """
    path = tuple(key) if isinstance(key, (tuple, list)) else (str(key),)
    value = _lookup(config or {}, path)
    if value is not _MISSING:
        return value
    if path[0] == "manga_settings" and len(path) > 1:
        value = _lookup(merged_manga_settings(config or {}), path[1:])
        if value is not _MISSING:
            return value
    if len(path) == 1:
        defaults = top_level_defaults()
        if path[0] in defaults and defaults[path[0]] is not None:
            return copy.deepcopy(defaults[path[0]])
    return default


# ---------------------------------------------------------------------------
# Option catalogs (desktop combo values and labels) and their mobile availability
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OptionRow:
    value: str
    label: str
    status: str = "ready"  # ready | needs_key | not_downloaded | downloading | unavailable | off
    chip: str = ""  # status chip text ("Ready", "Needs key", "Not in this build", ...)
    reason: Optional[str] = None  # set when the row is disabled (ReasonChip)
    detail: str = ""  # sub-line (backend, what is missing)
    progress: Optional[float] = None

    @property
    def disabled(self) -> bool:
        return self.reason is not None


#: ``MangaTranslationTab`` OCR provider combo (value, label) in desktop order; PaddleOCR is
#: commented out on desktop and listed here (UI_SPEC §4.6: listed, never hidden).
OCR_PROVIDERS = (
    ("custom-api", "Your Own key"),
    ("google", "Google Cloud Vision"),
    ("azure", "Azure Computer Vision"),
    ("azure-document-intelligence", "📋 Azure Document Intelligence (successor to Azure AI Vision)"),
    ("rapidocr", "⚡ RapidOCR (Fast & Local)"),
    ("manga-ocr", "🇯🇵 Manga OCR (Japanese)"),
    ("Qwen2-VL", "🇰🇷 Qwen2-VL (Korean)"),
    ("easyocr", "🌏 EasyOCR (Multi-lang)"),
    ("paddleocr", "🐼 PaddleOCR"),
    ("doctr", "📄 DocTR"),
)

#: Inpainting method radio buttons (+ the Skip Inpainter toggle shown as the first choice).
INPAINT_METHODS = (
    ("skip", "Skip"),
    ("local", "Local / API Model"),
    ("cloud", "Replicate API"),
    ("hybrid", "Hybrid"),
)
#: Local model combo (desktop order).
LOCAL_INPAINT_MODELS = (
    ("aot", "AOT (Torch JIT)"),
    ("aot_onnx", "AOT ONNX (Optimized)"),
    ("lama", "LaMa (Torch JIT)"),
    ("lama_onnx", "LaMa ONNX"),
    ("anime", "Anime (Torch JIT)"),
    ("anime_onnx", "Anime ONNX"),
    ("custom-image-edit", "Custom image edit endpoint"),
    ("mat", "MAT (Torch JIT)"),
    ("ollama", "Ollama"),
    ("sd_local", "Stable Diffusion (local)"),
)
ONNX_INPAINT_MODELS = frozenset({"aot_onnx", "lama_onnx", "anime_onnx"})

#: MangaSettingsDialog detector types (``manga_settings.ocr.detector_type``).
DETECTORS = (
    ("rtdetr_onnx", "RT-DETR (ONNX)"),
    ("rtdetr", "RT-DETR (PyTorch)"),
    ("yolo", "YOLOv8"),
    ("custom", "Custom Model"),
)

def chip_text(reason: Optional[str]) -> str:
    """The short ReasonChip text of a reason ("Needs PyTorch · not available on mobile." -> "Needs PyTorch")."""
    text = str(reason or "").strip()
    for sep in (" · ", " (", ". ", "; "):
        if sep in text:
            text = text.split(sep, 1)[0]
    text = text.rstrip(".")
    return text if len(text) <= 32 else text[:30].rstrip() + "…"


def value_reason(key: str, value: Any, *, mobile: Optional[bool] = None) -> Optional[str]:
    """Why ``value`` of the combo ``key`` cannot run here (None when it can).

    The rules are ``settings_schema``'s (``is_value_available``: torch-only OCR providers,
    detectors and inpainters, the non-functional ollama / sd_local, Hybrid); the desktop never
    disables anything. The value stays listed (disabled, with the reason) and a stored value
    round-trips untouched.
    """
    if mobile is None:
        mobile = _is_mobile()
    if not mobile:
        return None
    check = core_attr("settings_schema", "is_value_available")
    if not callable(check):
        return None
    try:
        ok, reason = check(key, value, "mobile")
    except Exception:
        log.debug("settings_schema.is_value_available failed", exc_info=True)
        return None
    return None if ok else (str(reason or "") or "Not available on mobile")


def _kept_note(reason: str) -> str:
    return f"{reason}\n\nYour saved choice is kept for the desktop app."


# ---------------------------------------------------------------------------
# OCR provider status (desktop _check_provider_status rules; display only)
# ---------------------------------------------------------------------------


def _google_credentials_problem(path: str) -> Optional[str]:
    """``_validate_google_credentials``: a service-account JSON file."""
    if not path:
        return "Credentials needed"
    if not os.path.exists(path):
        return "Credentials file not found"
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return "Invalid credentials JSON"
    if not isinstance(data, Mapping):
        return "Invalid credentials JSON"
    if any(k not in data for k in ("type", "project_id", "private_key", "client_email")):
        return "Invalid credentials JSON"
    if str(data.get("type", "")).lower() != "service_account":
        return "Invalid credentials JSON"
    return None


def google_backend() -> Optional[str]:
    """``"SDK"`` (google-cloud-vision), ``"REST"`` (google_vision_rest) or None."""
    if importable("google.cloud.vision"):
        return "SDK"
    if core("google_vision_rest") is not None:
        return "REST"
    return None


def azure_docintel_backend() -> Optional[str]:
    """``"SDK"`` (azure-ai-formrecognizer, what ``ocr_manager`` imports), ``"REST"``
    (``azure_document_intelligence_rest``, mobile) or None."""
    if importable("azure.ai.formrecognizer"):
        return "SDK"
    if core("azure_document_intelligence_rest") is not None:
        return "REST"
    return None


def rapidocr_reason() -> Optional[str]:
    missing = [name for name in ("rapidocr_onnxruntime", "pyclipper", "shapely") if not importable(name)]
    if missing:
        return f"{NOT_IN_BUILD} (no Android / iOS wheels: {', '.join(missing)})"
    return None


def ocr_provider_rows(config: Mapping[str, Any], *, mobile: Optional[bool] = None,
                      model_status: Optional[Callable[[str], Any]] = None) -> list:
    """One ``OptionRow`` per desktop OCR provider with its status chip (UI_SPEC §4.6)."""
    if mobile is None:
        mobile = _is_mobile()
    cfg = config or {}
    rows: list = []
    for value, label in OCR_PROVIDERS:
        reason = value_reason(K_PROVIDER, value, mobile=mobile)
        if reason:
            rows.append(OptionRow(value, label, "unavailable", chip_text(reason), reason, _kept_note(reason)))
            continue
        if value == "custom-api":
            detection = effective_setting(cfg, P_BUBBLE_DETECTION, True)
            detail = "Uses the translation model, its API key and the Vision key pool"
            if not detection:
                rows.append(OptionRow(value, label, "ready", "Ready", None,
                                      "Enable AI bubble detection for best results"))
            else:
                rows.append(OptionRow(value, label, "ready", "Ready", None, detail))
        elif value == "google":
            backend = google_backend()
            if backend is None:
                rows.append(OptionRow(value, label, "unavailable", NOT_IN_BUILD,
                                      "Neither google-cloud-vision nor its REST fallback is in this build"))
                continue
            problem = _google_credentials_problem(str(cfg.get(K_GOOGLE_CREDS) or cfg.get(K_GOOGLE_CLOUD_CREDS) or ""))
            detail = f"Google Vision {backend}"
            rows.append(OptionRow(value, label, "needs_key" if problem else "ready",
                                  "Needs key" if problem else "Ready", None,
                                  f"{detail} · {problem}" if problem else detail))
        elif value == "azure":
            if not importable("azure.ai.vision.imageanalysis"):
                rows.append(OptionRow(value, label, "unavailable", NOT_IN_BUILD,
                                      "azure-ai-vision-imageanalysis is not in this build"))
                continue
            ok = bool(str(cfg.get(K_AZURE_KEY) or "").strip())
            rows.append(OptionRow(value, label, "ready" if ok else "needs_key", "Ready" if ok else "Needs key", None,
                                  "Azure AI Vision SDK" if ok else "Key needed"))
        elif value == "azure-document-intelligence":
            backend = azure_docintel_backend()
            if backend is None:
                rows.append(OptionRow(value, label, "unavailable", NOT_IN_BUILD,
                                      "Neither the Document Intelligence SDK nor its REST client is in this build"))
                continue
            key = str(cfg.get(K_AZURE_KEY) or cfg.get(K_DOCINTEL_KEY) or "").strip()
            endpoint = str(cfg.get(K_AZURE_ENDPOINT) or cfg.get(K_DOCINTEL_ENDPOINT) or "").strip()
            if key and endpoint:
                rows.append(OptionRow(value, label, "ready", "Ready", None, f"Document Intelligence {backend}"))
            else:
                rows.append(OptionRow(value, label, "needs_key", "Needs key", None,
                                      "Endpoint needed" if key else "Key & Endpoint needed"))
        elif value == "rapidocr":
            missing = rapidocr_reason()
            if missing:
                rows.append(OptionRow(value, label, "unavailable", NOT_IN_BUILD, missing))
                continue
            status = model_status("rapidocr") if model_status is not None else None
            rows.append(_model_backed_row(value, label, status, "RapidOCR ONNX models"))
        else:
            rows.append(OptionRow(value, label, "ready", "Ready"))
    return rows


def _model_backed_row(value: str, label: str, status: Any, what: str) -> "OptionRow":
    if status is None:
        return OptionRow(value, label, "ready", "Ready", None, what)
    state = getattr(status, "status", "ready")
    if state == "downloading":
        pct = int(round(float(getattr(status, "progress", 0.0) or 0.0) * 100))
        return OptionRow(value, label, "downloading", f"Downloading {pct}%", None, what,
                         progress=getattr(status, "progress", None))
    if state in ("missing", "error"):
        return OptionRow(value, label, "not_downloaded", "Model not downloaded", None, what)
    if state == "unavailable":
        reason = getattr(status, "reason", None) or NOT_IN_BUILD
        return OptionRow(value, label, "unavailable", NOT_IN_BUILD, reason)
    return OptionRow(value, label, "ready", "Ready", None, what)


def detector_rows(*, mobile: Optional[bool] = None) -> list:
    key = ".".join(P_DETECTOR)
    rows = []
    for value, label in DETECTORS:
        reason = value_reason(key, value, mobile=mobile)
        rows.append(OptionRow(value, label, "unavailable" if reason else "ready",
                              chip_text(reason), reason))
    return rows


def inpaint_method_rows(*, mobile: Optional[bool] = None) -> list:
    rows = []
    for value, label in INPAINT_METHODS:
        reason = value_reason(K_INPAINT_METHOD, value, mobile=mobile) if value != "skip" else None
        rows.append(OptionRow(value, label, "unavailable" if reason else "ready",
                              chip_text(reason), reason))
    return rows


def local_model_rows(*, mobile: Optional[bool] = None) -> list:
    rows = []
    for value, label in LOCAL_INPAINT_MODELS:
        reason = value_reason(K_LOCAL_MODEL, value, mobile=mobile)
        rows.append(OptionRow(value, label, "unavailable" if reason else "ready", chip_text(reason), reason,
                              _kept_note(reason) if reason else ""))
    return rows


def current_inpaint_choice(config: Mapping[str, Any]) -> tuple:
    """(method, local model) as the inpainting group shows them: Skip wins (``manga_skip_inpainting``)."""
    cfg = config or {}
    if bool(effective_setting(cfg, K_SKIP_INPAINT, False)):
        method = "skip"
    else:
        method = str(cfg.get(K_INPAINT_METHOD) or effective_setting(cfg, P_INPAINT_METHOD, "local") or "local")
    # the tab reads manga_local_inpaint_model (default 'anime_onnx'; the phone default on mobile)
    local = str(effective_setting(cfg, K_LOCAL_MODEL, "anime_onnx") or "anime_onnx")
    if local == "qwen_image_edit":  # desktop migration: qwen_image_edit became custom-image-edit
        local = "custom-image-edit"
    return method, local


def inpaint_choice_updates(method: str, local: Optional[str] = None) -> dict:
    """The config writes of picking ``method`` (and ``local``) — the keys ``_save_rendering_settings``
    writes for the inpainting group (nested + top-level copies, the Skip toggle)."""
    updates: dict = {}
    if method == "skip":
        updates[K_SKIP_INPAINT] = True
        return updates
    updates[K_SKIP_INPAINT] = False
    updates[K_INPAINT_METHOD] = method
    updates[P_INPAINT_METHOD] = method
    if local:
        updates[K_LOCAL_MODEL] = local
        updates[P_LOCAL_METHOD] = local
    return updates


def inpaint_status(config: Mapping[str, Any], *, model_status: Optional[Callable[[str], Any]] = None,
                   mobile: Optional[bool] = None) -> OptionRow:
    """The inpainting method row's chip: Preloaded · Loading · Not downloaded · Needs key."""
    method, local = current_inpaint_choice(config)
    cfg = config or {}
    if method == "skip":
        return OptionRow("skip", "Skip", "off", "Off", None, "Text is drawn over the original image")
    reason = value_reason(K_INPAINT_METHOD, method, mobile=mobile)
    if reason:
        return OptionRow(method, dict(INPAINT_METHODS).get(method, method), "unavailable",
                         chip_text(reason), reason)
    if method == "cloud":
        ok = bool(str(cfg.get(K_REPLICATE_KEY) or "").strip())
        return OptionRow("cloud", "Replicate API", "ready" if ok else "needs_key", "Ready" if ok else "Needs key",
                         None, "Cloud API configured" if ok else "Cloud API not configured")
    reason = value_reason(K_LOCAL_MODEL, local, mobile=mobile)
    label = dict(LOCAL_INPAINT_MODELS).get(local, local)
    if reason:
        return OptionRow(local, label, "unavailable", chip_text(reason), reason)
    if local == "custom-image-edit":
        endpoint = str(cfg.get(K_EDIT_ENDPOINT) or cfg.get("manga_custom-image-edit_model_path") or "").strip()
        return OptionRow(local, label, "ready" if endpoint else "needs_key", "Ready" if endpoint else "Needs key",
                         None, endpoint or "Endpoint URL needed")
    status = model_status(local) if model_status is not None else None
    if status is None:
        return OptionRow(local, label, "ready", "Ready")
    state = getattr(status, "status", "ready")
    if state == "loaded":
        return OptionRow(local, label, "ready", "Preloaded", None, getattr(status, "path", "") or "")
    if state == "loading":
        return OptionRow(local, label, "downloading", "Loading", None)
    if state == "downloading":
        pct = int(round(float(getattr(status, "progress", 0.0) or 0.0) * 100))
        return OptionRow(local, label, "downloading", f"Downloading {pct}%", None,
                         progress=getattr(status, "progress", None))
    if state in ("missing", "error"):
        return OptionRow(local, label, "not_downloaded", "Not downloaded", None,
                         getattr(status, "error", "") or "Download the model in Settings › Inpainting")
    if state == "unavailable":
        return OptionRow(local, label, "unavailable", NOT_IN_BUILD, getattr(status, "reason", None) or NOT_IN_BUILD)
    return OptionRow(local, label, "ready", "Ready", None, getattr(status, "path", "") or "")


# ---------------------------------------------------------------------------
# Rendering presets / reset / image-edit endpoint test (shared helpers only)
# ---------------------------------------------------------------------------


def _config_updates(value: Any) -> dict:
    """``{key or path tuple: value}`` from a shared helper's result (dotted keys become paths)."""
    out: dict = {}
    for key, item in dict(value or {}).items():
        if isinstance(key, str) and key.startswith("manga_settings."):
            out[tuple(key.split("."))] = item
        else:
            out[tuple(key) if isinstance(key, list) else key] = item
    return out


#: preset -> its config writes. The desktop presets set constants (``_set_font_preset``), so the
#: measured writes do not depend on the config and are computed once per session.
_PRESET_CACHE: dict = {}
_PRESET_LOCK = threading.Lock()


class PresetsBusy(RuntimeError):
    """The presets are measured on scratch manga tabs, which write ``os.environ``: that waits
    for the running job (it owns the process state, ``job_runner.JOB_LOCK``)."""


def presets_available() -> bool:
    """Whether this build has the shared presets (cheap: manga_settings_defaults is stdlib only)."""
    if callable(core_attr("manga_settings_defaults", "font_preset_updates", "font_preset_config_updates")):
        return True
    return isinstance(core_attr("manga_settings_defaults", "FONT_PRESET_UPDATES", "FONT_PRESETS"), Mapping)


def cached_font_preset_updates(preset: str) -> Optional[dict]:
    """The preset's config writes when already measured in this session (UI loop safe), else None."""
    with _PRESET_LOCK:
        cached = _PRESET_CACHE.get(preset)
    return copy.deepcopy(cached) if cached is not None else None


def font_preset_updates(preset: str) -> dict:
    """Blocking (io pool): config writes of the Manga / Manhwa / Large Text presets (desktop
    ``_set_font_preset`` + ``_save_rendering_settings``), from ``manga_settings_defaults``;
    empty without it. The shared helper runs the moved desktop method on scratch headless tabs
    whose set-up writes ``os.environ`` (and puts it back), so it runs under
    ``job_runner.JOB_LOCK`` like anything else that builds an owner, never while a job owns the
    process state (``PresetsBusy``); the result is cached for the session."""
    cached = cached_font_preset_updates(preset)
    if cached is not None:
        return cached
    func = core_attr("manga_settings_defaults", "font_preset_updates", "font_preset_config_updates")
    if callable(func):
        job_lock = core_attr("job_runner", "JOB_LOCK")
        if job_lock is not None and not job_lock.acquire(blocking=False):
            raise PresetsBusy("Presets can be applied once the running job has finished")
        try:
            updates = _config_updates(func(preset))
        except Exception:
            log.exception("font_preset_updates(%s) failed", preset)
            return {}
        finally:
            if job_lock is not None:
                job_lock.release()
        if updates:
            with _PRESET_LOCK:
                _PRESET_CACHE[preset] = copy.deepcopy(updates)
        return updates
    presets = core_attr("manga_settings_defaults", "FONT_PRESET_UPDATES", "FONT_PRESETS")
    if isinstance(presets, Mapping) and isinstance(presets.get(preset), Mapping):
        return _config_updates(presets[preset])
    return {}


def rendering_reset_updates() -> dict:
    """Config writes of Rendering › Reset (desktop ``_reset_rendering_to_defaults``), shared only."""
    value = core_attr("manga_settings_defaults", "rendering_reset_updates", "RENDERING_RESET_UPDATES",
                      "RENDERING_RESET_DEFAULTS")
    if callable(value):
        try:
            value = value()
        except Exception:
            log.exception("rendering_reset_updates failed")
            return {}
    return _config_updates(value) if isinstance(value, Mapping) else {}


def test_image_edit_endpoint(config: Mapping[str, Any]) -> str:
    """Blocking: the custom image-edit endpoint check (desktop ``_test_custom_image_edit_endpoint``,
    shared helper); returns the message the desktop shows."""
    func = core_attr("manga_env", "test_custom_image_edit_endpoint", "check_custom_image_edit_endpoint")
    if not callable(func):
        raise RuntimeError("The endpoint test needs the shared manga module (manga_env)")
    result = func(dict(config or {}))
    if isinstance(result, tuple):
        return str(result[-1])
    return str(result or "")


# ---------------------------------------------------------------------------
# Files tab model (manga_files_core)
# ---------------------------------------------------------------------------


class _MainGuiConfig:
    """``main_gui`` for the moved Files methods: ``config`` + ``save_config`` (sparse writes back)."""

    def __init__(self, config: dict, on_save: Optional[Callable[[dict], Any]] = None) -> None:
        self.config = config
        self._on_save = on_save
        self.saves = 0

    def save_config(self, *args: Any, **kwargs: Any) -> None:
        self.saves += 1
        if self._on_save is not None:
            self._on_save(self.config)


class _HostLog:
    """First in the host's MRO: the moved code's log lines go to the Files tab (not stdout)."""

    def _log(self, message: Any = "", level: str = "info", *args: Any, **kwargs: Any) -> None:
        self.logs.append((str(message), level))
        if self._log_fn is not None:
            try:
                self._log_fn(str(message))
            except Exception:
                pass


class _FilesHostBase:
    """The ``MangaTranslationTab`` attributes the moved Files / glossary methods read."""

    def __init__(self, config: Optional[dict] = None, *, on_save: Optional[Callable[[dict], Any]] = None,
                 temp_root: Optional[str] = None, log_fn: Optional[Callable[[str], Any]] = None) -> None:
        self.main_gui = _MainGuiConfig(dict(config or {}), on_save)
        self.selected_files: list = []
        self.skipped_processing_files: set = set()
        self.manga_image_range_value = ""
        self.manga_selected_folder_roots: list = []
        self.manga_split_first_level_subfolders_value = bool((config or {}).get(K_SPLIT_SUBFOLDERS, False))
        self.cbz_jobs: dict = {}
        self.cbz_image_to_job: dict = {}
        self.cbz_temp_root = temp_root
        self.manga_process_group_index = 0
        self._manga_processing_files = None
        self._current_image_path = None
        self.logs: list = []
        self._log_fn = log_fn
        files_core = core("manga_files_core")
        shim = getattr(files_core, "FileListShim", None) if files_core is not None else None
        self.file_listbox = shim(self) if shim is not None else None
        self.sync_config(self.main_gui.config)

    def sync_config(self, config: Mapping[str, Any]) -> None:
        """Take the current config (the store is the source of truth; the tab keeps copies)."""
        self.main_gui.config = dict(config or {})
        cfg = self.main_gui.config
        self.manga_custom_glossary_path = str(cfg.get("manga_custom_glossary_path") or "")
        self.manga_generated_glossary_path = str(cfg.get("manga_generated_glossary_path") or "")
        self.manga_glossary_auto_load_suppressed = bool(cfg.get("manga_glossary_auto_load_suppressed", False))
        self.manga_glossary_auto_load_suppressed_root = str(cfg.get("manga_glossary_auto_load_suppressed_root") or "")


_HOST_CLASS: dict = {}


def _files_mixins() -> tuple:
    """``MangaOcrSessionMixin`` (OCR export folder / names) + ``MangaEnvMixin`` (the glossary
    auto-load the selection triggers) + ``MangaFilesMixin`` + ``MangaHooksMixin``: the classes
    ``MangaTranslationTab`` inherits the moved methods from."""
    files_core = core("manga_files_core")
    if files_core is None:
        return ()
    found = []
    env = core("manga_env")
    for name in ("MangaOcrSessionMixin", "MangaEnvMixin"):
        cls = getattr(env, name, None) if env is not None else None
        if cls is not None:
            found.append(cls)
    for name in ("MangaFilesMixin", "MangaHooksMixin"):
        cls = getattr(files_core, name, None)
        if cls is not None:
            found.append(cls)
    return tuple(found)


def _files_host_class() -> type:
    mixins = _files_mixins()
    key = tuple(id(m) for m in mixins)
    cls = _HOST_CLASS.get(key)
    if cls is None:
        cls = type("MangaFilesHost", (_HostLog, *mixins, _FilesHostBase), {})
        _HOST_CLASS.clear()
        _HOST_CLASS[key] = cls
    return cls


def natural_sort_key(text: str) -> Any:
    func = core_attr("manga_files_core", "_natural_sort_key")
    if callable(func):
        return func(text)
    return str(text).lower()


@dataclass(frozen=True)
class FileGroup:
    root: str
    name: str
    files: tuple


#: Config keys the moved Files / glossary code writes on ``main_gui.config`` (written back).
_HOST_SAVED_KEYS = (K_SELECTED, K_SKIPPED, K_FOLDER_ROOTS, "manga_generated_glossary_path",
                    "manga_glossary_auto_load_suppressed", "manga_glossary_auto_load_suppressed_root")
_SKIPPED_FOLDERS = ("glossary", "ocr text", "mangaglossary_backup", "__macosx")


def _safe_extract(archive: str, target: str) -> str:
    """Extract a ZIP into ``target`` (no member may escape it)."""
    os.makedirs(target, exist_ok=True)
    root = os.path.realpath(target)
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            dest = os.path.realpath(os.path.join(target, member.filename))
            if dest != root and not dest.startswith(root + os.sep):
                raise ValueError(f"Unsafe path in the archive: {member.filename}")
        zf.extractall(target)
    return target


class MangaFileList:
    """The Files tab state (UI_SPEC §4.6 Files) over the moved ``MangaFilesMixin`` methods.

    ``host`` carries the ``MangaTranslationTab`` attributes (selected files in visible order,
    skip keys, image range, folder roots, CBZ jobs) and runs the desktop methods unchanged:
    ``_add_dropped_manga_paths`` (images, CBZ archives, folders), ``_sort_files``,
    ``_toggle_skip_processing_for_path``, ``_parse_manga_image_range``,
    ``_manga_range_filtered_files``, ``_manga_process_groups_for_paths``,
    ``_persist_selected_files`` / ``_load_persisted_files`` (the desktop keys, written back
    through ``save(updates)``) and ``_create_cbz_from_isolated_folders``. Every mutation holds
    a lock (the tab runs them on the io pool).
    """

    SORTS = ("name", "numeric", "date", "reverse")

    def __init__(self, config: Optional[Mapping[str, Any]] = None, *,
                 save: Optional[Callable[[dict], Any]] = None, temp_root: Optional[str] = None,
                 log_fn: Optional[Callable[[str], Any]] = None,
                 config_source: Optional[Callable[[], Mapping[str, Any]]] = None,
                 folders_root: Optional[str] = None) -> None:
        self._save = save
        self._config_source = config_source
        self.folders_root = folders_root or (os.path.join(os.path.dirname(temp_root), "folders") if temp_root else "")
        self.lock = threading.RLock()
        cls = _files_host_class()
        self.host = cls(dict(config or {}), on_save=self._on_host_save, temp_root=temp_root, log_fn=log_fn)

    @property
    def available(self) -> bool:
        return bool(_files_mixins())

    def _require(self) -> None:
        if not self.available:
            raise RuntimeError(MISSING_CORE + " (manga_files_core)")

    def _sync(self) -> None:
        if self._config_source is not None:
            try:
                self.host.sync_config(self._config_source())
            except Exception:
                log.debug("reading the config for the manga files failed", exc_info=True)

    # ---- state -------------------------------------------------------------------------------

    @property
    def files(self) -> list:
        return list(self.host.selected_files)

    @property
    def image_range(self) -> str:
        return str(self.host.manga_image_range_value or "")

    @property
    def split_first_level(self) -> bool:
        return bool(self.host.manga_split_first_level_subfolders_value)

    @property
    def folder_roots(self) -> list:
        return list(self.host.manga_selected_folder_roots or [])

    @property
    def cbz_jobs(self) -> dict:
        return dict(getattr(self.host, "cbz_jobs", {}) or {})

    @property
    def cbz_image_to_job(self) -> dict:
        return dict(getattr(self.host, "cbz_image_to_job", {}) or {})

    def skip_key(self, path: str) -> str:
        method = getattr(self.host, "_skip_key_for_path", None)
        if callable(method):
            return method(path)
        return os.path.normcase(os.path.abspath(os.path.normpath(path)))

    def is_skipped(self, path: str) -> bool:
        method = getattr(self.host, "_is_manually_skipped_processing_file", None)
        if callable(method):
            return bool(method(path))
        return False

    def range_skipped(self, path: str) -> bool:
        """Outside the image range (shown dimmed; not processed)."""
        method = getattr(self.host, "_visible_range_skipped_keys", None)
        if not callable(method):
            return False
        try:
            return self.skip_key(path) in method()
        except Exception:
            return False

    def parse_range(self) -> tuple:
        """(indices or None, error) of the image range, 1-based visible rows (desktop rule)."""
        method = getattr(self.host, "_parse_manga_image_range", None)
        if not callable(method):
            return None, (MISSING_CORE if self.image_range else None)
        return method(len(self.host.selected_files))

    def range_status(self) -> str:
        """The line under the range field: "All images" / "N in range · R of T will run" / the error."""
        indices, error = self.parse_range()
        total = len(self.host.selected_files)
        if error:
            return str(error)
        run = len(self.run_files()[0])
        if indices is None:
            return "All images" if not total else f"All images · {run} of {total} will run"
        return f"{len(indices)} in range · {run} of {total} will run"

    def run_files(self) -> tuple:
        """(files the run processes, error): the range and the per-file skips (``_manga_range_filtered_files``)."""
        method = getattr(self.host, "_manga_range_filtered_files", None)
        if callable(method):
            return method()
        return list(self.host.selected_files), None

    def groups(self) -> list:
        """Process groups (first-level subfolders when "split" is on): ``_manga_process_groups_for_paths``."""
        method = getattr(self.host, "_manga_process_groups_for_paths", None)
        files = self.files
        if not callable(method) or not files:
            return [FileGroup(root="", name="All images", files=tuple(files))] if files else []
        return [FileGroup(root=str(g.get("root") or ""), name=str(g.get("name") or ""),
                          files=tuple(g.get("files") or ())) for g in method(files) or []]

    def output_path_for(self, path: str) -> str:
        """Where the run writes ``path``'s translated page (``_get_manga_output_path_for_file``).

        The desktop method creates the page's output folder; a folder it had to create for this
        lookup only (still empty) is removed again, so viewing pages leaves no empty folders."""
        method = getattr(self.host, "_get_manga_output_path_for_file", None)
        if not callable(method):
            return ""
        try:
            output = str(method(path))
        except Exception:
            return ""
        folder = os.path.dirname(output)
        try:
            if folder and os.path.isdir(folder) and not os.listdir(folder):
                os.rmdir(folder)
        except OSError:
            pass
        return output

    def existing_outputs(self, paths: Optional[Sequence[str]] = None) -> list:
        """Translated pages already on disk for ``paths`` (default: the run files)."""
        found: list = []
        for path in (paths if paths is not None else self.run_files()[0]):
            output = self.output_path_for(path)
            if output and os.path.isfile(output) and output not in found:
                found.append(output)
        return found

    # ---- edits ---------------------------------------------------------------------------------

    def add_paths(self, paths: Iterable[str]) -> int:
        """Images, CBZ archives and folders (``_add_dropped_manga_paths``); a ``.zip`` is a folder
        that could not be picked directly (Android): it is extracted next to the session, then added
        as that folder, so its subfolders keep the process grouping."""
        self._require()
        wanted: list = []
        for raw in paths:
            path = os.path.abspath(os.fspath(raw))
            if path.lower().endswith(".zip") and os.path.isfile(path):
                stem = os.path.splitext(os.path.basename(path))[0]
                target = os.path.join(self.folders_root or os.path.dirname(path), stem)
                if not os.path.isdir(target):
                    _safe_extract(path, target)
                wanted.append(target)
            else:
                wanted.append(path)
        with self.lock:
            self._sync()
            before = len(self.host.selected_files)
            self.host._add_dropped_manga_paths(wanted)
            return len(self.host.selected_files) - before

    def remove(self, paths: Iterable[str]) -> int:
        self._require()
        drop = {self.skip_key(p) for p in paths}
        with self.lock:
            self._sync()
            before = len(self.host.selected_files)
            self.host.selected_files = [p for p in self.host.selected_files if self.skip_key(p) not in drop]
            self.host._prune_skipped_processing_files()
            self.host._persist_selected_files()
            return before - len(self.host.selected_files)

    def clear(self) -> None:
        self._require()
        with self.lock:
            self._sync()
            self.host.selected_files = []
            self.host.skipped_processing_files = set()
            self.host.manga_selected_folder_roots = []
            self.host.cbz_jobs = {}
            self.host.cbz_image_to_job = {}
            self.host.manga_image_range_value = ""
            self.host._persist_selected_files()

    def move(self, old_index: int, new_index: int) -> bool:
        """Drag reorder (desktop ``_on_files_reordered``: the new order, no active sort, persisted)."""
        self._require()
        with self.lock:
            files = self.host.selected_files
            if not (0 <= old_index < len(files)) or not (0 <= new_index < len(files)) or old_index == new_index:
                return False
            self._sync()
            item = files.pop(old_index)
            files.insert(new_index, item)
            self.host._manga_file_sort = None
            self.host._persist_selected_files()
            return True

    def sort(self, sort_type: str, reverse: bool = False) -> None:
        """Name / Number (natural) / Date / Reverse (``_sort_files``)."""
        self._require()
        if sort_type not in self.SORTS:
            raise ValueError(sort_type)
        with self.lock:
            self._sync()
            self.host._sort_files(sort_type, reverse)

    def toggle_skip(self, path: str) -> bool:
        """Per-file "Process this image" (``_toggle_skip_processing_for_path``); True when now skipped."""
        self._require()
        with self.lock:
            self._sync()
            self.host._toggle_skip_processing_for_path(path)
            return self.is_skipped(path)

    def set_range(self, text: str) -> None:
        self.host.manga_image_range_value = str(text or "").strip()

    def set_split_first_level(self, on: bool) -> None:
        self.host.manga_split_first_level_subfolders_value = bool(on)
        if self._save is not None:
            self._save({K_SPLIT_SUBFOLDERS: bool(on)})

    # ---- persistence ---------------------------------------------------------------------------

    def _on_host_save(self, config: Mapping[str, Any]) -> None:
        if self._save is None:
            return
        updates = {key: copy.deepcopy(config[key]) for key in _HOST_SAVED_KEYS if key in config}
        if updates:
            self._save(updates)

    def load(self, config: Mapping[str, Any]) -> int:
        """Restore the persisted selection (``_load_persisted_files``; missing files dropped)."""
        if not self.available:
            return 0
        with self.lock:
            self.host.sync_config(config or {})
            self.host.selected_files = []
            self.host._load_persisted_files()
            return len(self.host.selected_files)

    # ---- CBZ / output ---------------------------------------------------------------------------

    def cbz_job_for(self, path: str) -> Optional[str]:
        return (getattr(self.host, "cbz_image_to_job", {}) or {}).get(path)

    def create_cbz(self, files: Optional[Sequence[str]] = None) -> str:
        """Create CBZ (desktop button): ``_create_cbz_from_isolated_folders`` over the run files;
        returns the archive it wrote (``<output folder>/<folder name>_translated.cbz``)."""
        self._require()
        with self.lock:
            self._sync()
            run = list(files) if files is not None else self.run_files()[0]
            self.host._manga_processing_files = run
            try:
                self.host._create_cbz_from_isolated_folders()
            finally:
                self.host._manga_processing_files = None
            override = str(self.host.main_gui.config.get("output_directory") or "") or os.environ.get(
                "OUTPUT_DIRECTORY", "")
            parent = override if override and os.path.isdir(override) else os.path.dirname(run[0])
            return os.path.join(parent, f"{os.path.basename(parent)}_translated.cbz")

    def ocr_dir(self) -> str:
        """The OCR Text folder of the active output root (``_manga_ocr_output_dir``)."""
        method = getattr(self.host, "_manga_ocr_output_dir", None)
        if not callable(method):
            return ""
        with self.lock:
            self._sync()
            return str(method())

    def ocr_export_path(self) -> str:
        """A timestamped export file in the OCR folder (``_manga_ocr_timestamped_export_filename``)."""
        folder = self.ocr_dir()
        method = getattr(self.host, "_manga_ocr_timestamped_export_filename", None)
        if not folder or not callable(method):
            return ""
        with self.lock:
            return os.path.join(folder, str(method()))

    def run_params(self) -> dict:
        """What ``manga_runner.HeadlessMangaRunner`` rebuilds the selection from (JSON-safe): every
        file in visible order + the range and skips it applies itself."""
        run_files, error = self.run_files()
        return {
            "files": self.files,
            "run_files": list(run_files),
            "range_error": error,
            "folder_roots": self.folder_roots,
            "split_first_level": self.split_first_level,
            "image_range": self.image_range,
            "skipped": sorted(self.skip_key(p) for p in self.host.skipped_processing_files if p),
            "cbz_jobs": {src: dict(job) for src, job in self.cbz_jobs.items()},
            "cbz_image_to_job": self.cbz_image_to_job,
        }


def zip_images(paths: Sequence[str], archive_path: str) -> str:
    """"Download images": the translated images as one ZIP for Save / Share (the system pickers
    save one file at a time)."""
    os.makedirs(os.path.dirname(os.path.abspath(archive_path)) or ".", exist_ok=True)
    tmp = archive_path + ".part"
    used: set = set()
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in paths:
            name = os.path.basename(path)
            stem, ext = os.path.splitext(name)
            n = 2
            while name.lower() in used:
                name = f"{stem} ({n}){ext}"
                n += 1
            used.add(name.lower())
            zf.write(path, name)
    os.replace(tmp, archive_path)
    return archive_path


# ---------------------------------------------------------------------------
# Models (manga_models)
# ---------------------------------------------------------------------------


@dataclass
class ModelEntry:
    id: str
    label: str
    kind: str = "inpaint"  # detector | inpaint | ocr
    size: Optional[int] = None  # bytes
    status: str = "missing"  # missing | downloading | ready | loading | loaded | error | unavailable
    progress: Optional[float] = None
    path: str = ""
    error: str = ""
    reason: Optional[str] = None

    @property
    def size_label(self) -> str:
        if not self.size:
            return ""
        fmt = core_attr("manga_models", "format_size")
        if callable(fmt):
            return str(fmt(self.size))
        size = float(self.size)
        for unit in ("B", "KB", "MB", "GB"):
            if size < 1024 or unit == "GB":
                return f"{size:.0f} {unit}" if unit in ("B", "KB") else f"{size:.1f} {unit}"
            size /= 1024.0
        return ""

    @property
    def chip(self) -> str:
        if self.status == "downloading":
            return f"Downloading {int(round((self.progress or 0.0) * 100))}%"
        return {"missing": "Not downloaded", "ready": "Downloaded", "loading": "Loading", "loaded": "Loaded",
                "error": "Failed", "unavailable": NOT_IN_BUILD}.get(self.status, self.status)


class ModelManager:
    """``manga_models`` for the screens: registry rows, download (resume + sha256 in the core)
    with progress, cancel, delete, disk usage and Load / Unload (detectors). Blocking calls run
    on the io pool; progress callbacks arrive on the download thread (the screen posts them to
    the UI loop)."""

    def __init__(self, module: Any = None) -> None:
        self._module = module
        self._lock = threading.Lock()
        self._errors: dict = {}
        self._loading: set = set()

    @property
    def module(self) -> Any:
        return self._module if self._module is not None else core("manga_models")

    @property
    def available(self) -> bool:
        return self.module is not None

    def _spec(self, model_id: str) -> Any:
        module = self.module
        if module is None:
            return None
        try:
            return module.get_spec(model_id)
        except Exception:
            return None

    def entries(self, kind: Optional[str] = None) -> list:
        module = self.module
        if module is None:
            return []
        return [self.status(spec.key) for spec in module.specs(kind)]

    def status(self, model_id: str) -> ModelEntry:
        module = self.module
        spec = self._spec(model_id)
        if spec is None:
            return ModelEntry(model_id, model_id, status="unavailable",
                              reason=MISSING_CORE + " (manga_models)" if module is None
                              else f"No model '{model_id}' in the registry")
        entry = ModelEntry(spec.key, spec.title, kind=spec.kind, size=int(spec.size or 0) or None)
        try:
            state = module.status(spec.key)
        except Exception as exc:
            entry.status, entry.error = "error", str(exc)
            return entry
        entry.path = str(state.path or "")
        with self._lock:
            error = self._errors.get(spec.key)
            loading = spec.key in self._loading
        loaded = spec.key in set(getattr(module, "loaded", lambda: [])() or ())
        if state.downloading is not None:
            entry.status, entry.progress = "downloading", float(state.downloading.fraction)
        elif loading:
            entry.status = "loading"
        elif state.installed:
            entry.status = "loaded" if loaded else "ready"
        elif error:
            entry.status, entry.error = "error", error
        else:
            entry.status = "missing"
            if state.partial_bytes and spec.size:
                entry.progress = min(1.0, float(state.partial_bytes) / float(spec.size))
                entry.error = f"Paused at {int(entry.progress * 100)}% · Download resumes"
        return entry

    def download(self, model_id: str, on_progress: Optional[Callable[[ModelEntry], Any]] = None) -> ModelEntry:
        """Blocking download (io pool): ``manga_models.download`` (resume, size + sha256, atomic)."""
        module = self.module
        if module is None:
            return ModelEntry(model_id, model_id, status="unavailable", reason=MISSING_CORE + " (manga_models)")
        with self._lock:
            self._errors.pop(model_id, None)

        def progress(report: Any) -> None:
            if on_progress is None:
                return
            entry = self.status(model_id)
            fraction = getattr(report, "fraction", None)
            if fraction is not None and getattr(report, "phase", "download") != "done":
                entry.status, entry.progress = "downloading", float(fraction)
            try:
                on_progress(entry)
            except Exception:
                log.debug("model progress callback failed", exc_info=True)

        try:
            module.download(model_id, progress=progress)
        except Exception as exc:
            cancelled = type(exc).__name__ == "DownloadCancelled"
            with self._lock:
                self._errors[model_id] = "Cancelled · Download resumes" if cancelled else (str(exc) or
                                                                                           type(exc).__name__)
            log.info("model %s download stopped: %s", model_id, exc)
        return self.status(model_id)

    def cancel(self, model_id: str) -> bool:
        module = self.module
        try:
            return bool(module.cancel(model_id)) if module is not None else False
        except Exception:
            return False

    def delete(self, model_id: str) -> ModelEntry:
        module = self.module
        if module is not None:
            try:
                self.unload(model_id)
                module.delete(model_id)
                with self._lock:
                    self._errors.pop(model_id, None)
            except Exception as exc:
                with self._lock:
                    self._errors[model_id] = str(exc)
        return self.status(model_id)

    def can_load(self, model_id: Optional[str] = None) -> bool:
        """Load warms the RT-DETR detector session (inpainters load when a run starts)."""
        module = self.module
        if module is None or not callable(getattr(module, "load", None)):
            return False
        if model_id is None:
            return True
        spec = self._spec(model_id)
        return spec is not None and spec.kind == getattr(module, "KIND_DETECTOR", "detector")

    def load(self, model_id: str) -> ModelEntry:
        module = self.module
        with self._lock:
            self._loading.add(model_id)
        try:
            module.load(model_id)
            with self._lock:
                self._errors.pop(model_id, None)
        except Exception as exc:
            with self._lock:
                self._errors[model_id] = str(exc) or type(exc).__name__
        finally:
            with self._lock:
                self._loading.discard(model_id)
        entry = self.status(model_id)
        with self._lock:
            error = self._errors.get(model_id)
        if error and entry.status in ("ready", "loaded"):
            entry.error = error
        return entry

    def unload(self, model_id: str) -> ModelEntry:
        module = self.module
        try:
            if module is not None and callable(getattr(module, "unload", None)):
                module.unload(model_id)
        except Exception:
            log.debug("unloading %s failed", model_id, exc_info=True)
        return self.status(model_id)

    def disk_usage(self) -> int:
        module = self.module
        try:
            usage = module.disk_usage() if module is not None else {}
            return int((usage or {}).get("total_bytes") or 0)
        except Exception:
            return 0

    def for_local_method(self, method: str) -> Optional[str]:
        """The registry key of a local inpainting method (``anime_onnx`` -> ``anime_onnx``)."""
        finder = getattr(self.module, "spec_for_inpaint_method", None) if self.module is not None else None
        spec = finder(method) if callable(finder) else None
        return spec.key if spec is not None else None

    def detector_id(self, config: Mapping[str, Any]) -> Optional[str]:
        """The registry key of the configured RT-DETR ONNX variant (``rtdetr_onnx_variant``; the
        phone default on mobile when unset)."""
        module = self.module
        if module is None:
            return None
        variant = str(effective_setting(config or {}, P_RTDETR_VARIANT, "") or "")
        try:
            spec = module.spec_for_detector_variant(variant) if variant else None
            if spec is None:
                spec = (module.get_spec(getattr(module, "MOBILE_DETECTOR_KEY", "rtdetr_v4_s_int8")) if _is_mobile()
                        else module.spec_for_detector_variant("detector.onnx"))
        except (KeyError, AttributeError):
            return None
        return spec.key if spec is not None else None

    def detector_variants(self) -> list:
        """``(selector, title, description, size label)`` of the registry's RT-DETR ONNX exports: the
        values of the desktop dialog's "ONNX Export" combo (``manga_settings.ocr.rtdetr_onnx_variant``)."""
        module = self.module
        if module is None:
            return []
        try:
            found = module.specs(getattr(module, "KIND_DETECTOR", "detector"))
        except Exception:
            return []
        out = []
        for spec in found:
            size = ModelEntry(spec.key, spec.title, size=int(spec.size or 0) or None).size_label
            out.append((str(spec.selector), str(spec.title), str(getattr(spec, "description", "") or ""), size))
        return out

    def required(self, config: Mapping[str, Any]) -> list:
        """The models a run with ``config`` needs that are not downloaded yet (``missing_models``)."""
        module = self.module
        try:
            return [spec.key for spec in module.missing_models(dict(config or {}))] if module is not None else []
        except Exception:
            return []


# ---------------------------------------------------------------------------
# Editor (manga_editor_core.MangaEditorSession) and its jobs
# ---------------------------------------------------------------------------

#: Editor steps that run as MANGA_STEP jobs (the session method each one calls on the job thread).
STEPS = ("detect", "clean", "recognize", "translate", "translate_all", "render", "ocr_box", "translate_box",
         "clean_box", "import_ocr")
STEP_LABELS = {
    "detect": "Detect",
    "clean": "Clean",
    "recognize": "Recognize",
    "translate": "Translate",
    "translate_all": "Translate all",
    "render": "Save & Update Overlay",
    "ocr_box": "OCR this text",
    "translate_box": "Translate this text",
    "clean_box": "Clean this box",
    "import_ocr": "Import OCR",
}
BOX_STEPS = frozenset({"ocr_box", "translate_box", "clean_box"})

_EDITOR_SESSIONS: dict = {}
_EDITOR_LOCK = threading.Lock()


def new_editor_session(*, image_paths: Sequence[str] = (), state_file: Optional[str] = None,
                       log_callback: Optional[Callable[..., Any]] = None,
                       event_callback: Optional[Callable[..., Any]] = None) -> Any:
    """Blocking: ``manga_editor_core.MangaEditorSession`` for the Files pages (its state file is the
    desktop ``image_state.json`` layout under the app data folder; no worker process on mobile).
    The session lives as long as the app; jobs hand it their HeadlessOwner (``owner=``)."""
    module = core("manga_editor_core")
    cls = getattr(module, "MangaEditorSession", None) if module is not None else None
    if cls is None:
        raise RuntimeError(MISSING_CORE + " (manga_editor_core)")
    if not state_file:
        default = getattr(module, "default_state_file", None)
        state_file = default() if callable(default) else None
    return cls(None, state_file=state_file, image_paths=list(image_paths or ()), log_callback=log_callback,
               event_callback=event_callback)


def register_editor_session(session: Any) -> str:
    """An opaque token for ``session`` (job params must stay JSON: the step finds it again)."""
    token = f"es{id(session):x}"
    with _EDITOR_LOCK:
        _EDITOR_SESSIONS[token] = session
    return token


def editor_session(token: Optional[str]) -> Any:
    with _EDITOR_LOCK:
        return _EDITOR_SESSIONS.get(str(token or ""))


def display_copy(path: Optional[str], revision: int, cache_dir: str) -> str:
    """A versioned copy of a rendered page for display (``page_<id>_v3.png``): the editor rewrites
    the same output file on every render, and Flet's ``Image`` cache would keep showing the first
    one. Older copies of the same page are removed. Blocking (io pool)."""
    if not path or not os.path.isfile(path) or not cache_dir:
        return path or ""
    import hashlib

    stem, ext = os.path.splitext(os.path.basename(path))
    key = hashlib.sha1(os.path.abspath(path).encode("utf-8", "surrogatepass")).hexdigest()[:8]
    try:
        stamp = int(os.path.getmtime(path))
    except OSError:
        stamp = 0
    target = os.path.join(cache_dir, f"{stem}_{key}_v{int(revision)}_{stamp}{ext or '.png'}")
    if os.path.isfile(target):
        return target
    os.makedirs(cache_dir, exist_ok=True)
    tmp = target + ".part"
    import shutil

    shutil.copyfile(path, tmp)
    os.replace(tmp, target)
    prefix = f"{stem}_{key}_v"
    for name in os.listdir(cache_dir):
        full = os.path.join(cache_dir, name)
        if name.startswith(prefix) and full != target:
            try:
                os.remove(full)
            except OSError:
                pass
    return target


def list_ocr_files(folder: str) -> list:
    """Auto-saved OCR JSON files in ``folder`` (newest first)."""
    if not folder or not os.path.isdir(folder):
        return []
    found = []
    for name in os.listdir(folder):
        if name.lower().endswith(".json"):
            full = os.path.join(folder, name)
            try:
                found.append((os.path.getmtime(full), full))
            except OSError:
                continue
    return [path for _mtime, path in sorted(found, reverse=True)]


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------


def batch_spec(files: MangaFileList, *, title: str = "", output_root: str = "", glossary_only: bool = False,
               editor_session: str = "", imported_ocr: str = "") -> Any:
    """The MANGA job for the current Files selection (Start / Generate glossary): every file in
    visible order with the range, skips, folder roots and CBZ jobs (the runner applies them like
    the desktop Start; all process groups run). ``imported_ocr``: the last imported OCR JSON, which
    the run reuses like the desktop's (``manga_env.import_ocr_session`` before Start)."""
    from glossarion_mobile.services.jobs import JobSpec

    params = files.run_params()
    if params.get("range_error"):
        raise ValueError(params["range_error"])
    run_files = list(params["run_files"])
    if not run_files:
        raise ValueError("No images to process (all skipped or outside the image range)")
    params["output_root"] = output_root
    params["glossary_only"] = bool(glossary_only)
    params["editor_session"] = editor_session  # the runner updates that session's page state
    params["imported_ocr"] = str(imported_ocr or "")
    groups = files.groups()
    name = title or (groups[0].name if len(groups) == 1 and groups[0].name else "") or (
        os.path.basename(os.path.dirname(run_files[0])) if len(run_files) > 1 else os.path.basename(run_files[0]))
    verb = "Glossary · " if glossary_only else ""
    return JobSpec(kind=KIND_BATCH,
                   title=f"{verb}{name} · {len(run_files)} image{'s' if len(run_files) != 1 else ''}",
                   inputs=tuple(run_files), params=params,
                   origin={"type": "tool", "route": TOOL_ROUTE, "label": "Tools · Manga"}, resumable=False)


def step_spec(step: str, session_token: str, image_path: str, *, images: Optional[Sequence[str]] = None,
              index: Optional[int] = None, extra: Optional[Mapping] = None) -> Any:
    """A MANGA_STEP job (editor workflow button / per-box action) on the editor session ``session_token``
    for ``image_path`` (``images``: Translate all / Import OCR; ``index``: the box of a per-box step)."""
    from glossarion_mobile.services.jobs import JobSpec

    if step not in STEPS:
        raise ValueError(step)
    if step in BOX_STEPS and index is None:
        raise ValueError(f"{step} needs a box")
    params: dict = {"step": step, "session": session_token, "image": os.path.abspath(image_path)}
    if images:
        params["images"] = [os.path.abspath(p) for p in images]
    if index is not None:
        params["index"] = int(index)
    if extra:
        params.update(dict(extra))
    inputs = tuple(params.get("images") or (params["image"],))
    label = STEP_LABELS[step]
    if step == "translate_all":
        title = f"{label} · {len(inputs)} pages"
    elif step in BOX_STEPS:
        title = f"{label} · box {int(index) + 1} · {os.path.basename(image_path)}"
    else:
        title = f"{label} · {os.path.basename(image_path)}"
    return JobSpec(kind=KIND_STEP, title=title, inputs=inputs, params=params,
                   origin={"type": "tool", "route": TOOL_ROUTE, "label": "Tools · Manga editor"}, resumable=False)


def now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")
