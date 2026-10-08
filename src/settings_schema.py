"""Canonical Glossarion settings schema: desktop tables + a curated overlay.

GUI-free and Python 3.10-compatible (never imports PySide6, translator_gui or
dpi_setup). The data comes from ``settings_schema_data.py``, which
``src/mobile/tools/schema_extract.py`` generates from the desktop sources
(``save_config``'s settings_map, the ``_init_variables`` tables, the startup
assignments, the env builders and the dialog modules). This module adds what the
desktop code cannot express: sections (mirroring the desktop dialogs), labels where
the generator found none, platform availability with reasons, and the fresh-install
values measured by the U0 desktop oracle.

Rules (plan section 2 / design critique):

* Defaults are **display-only**. Nothing here writes ``config.json``;
  ``config_store.load_config`` only reads and decrypts, and the mobile app writes keys
  sparsely, so an imported desktop config behaves exactly like desktop.
* Desktop default disagreements are recorded in ``SettingSpec.discrepancies`` and
  never "fixed" here (each fix is a separate, user-approved desktop change).
* Settings that a platform cannot run are never hidden: ``is_available`` returns
  ``(False, reason)`` and the UI shows a disabled row with the reason.

Public API: ``SettingSpec``, ``Section``, ``EnvBinding``, ``MISSING``,
``all_specs()``, ``spec(key)``, ``sections()``, ``section(id)``, ``effective_default(key)``,
``coerce(key, value)``, ``apply_converter(conv, value)``, ``search(query)``,
``is_available(key, platform='mobile')``, ``env_names(key)``, ``choice_values(key)``,
``evaluate_rule(rule_id, config)``.

Desktop tables (U9 P5b): ``desktop_settings_map(owner)``, ``desktop_bool_vars(owner)`` and
``desktop_str_vars(owner)`` return save_config's ``settings_map`` and ``_init_variables``'
``bool_vars`` / ``str_vars`` exactly as the desktop literals built them (same rows, order,
sources, defaults and converter behaviour), from the generated ``DESKTOP_*`` rows. The desktop
methods call them, so the schema is the single source of those tables; the literals live on in
``src/mobile/tools/frozen_desktop_tables.py`` (the generator's input and the oracle of
tests/test_schema_p5b.py). They never build the SettingSpec index.

``visible_if`` / ``locked_if`` (U4) are rule ids that ``settings_rules.evaluate`` answers:
``lock:<key>`` gives the lock reason from ``settings_rules.evaluate_locks`` ('' when the
setting is free); the visibility ids (``thinking:*``, ``output_mode:*``, ``glossary:*``) are
False when the desktop disables or hides the control for the current configuration. Choices
are plain values or ``(value, label)`` pairs (desktop combos whose items show other text).
"""
from __future__ import annotations

import copy
import importlib
import math
import re
import threading
from dataclasses import dataclass
from typing import Any, NamedTuple, Optional, Tuple

__all__ = [
    "MISSING", "EnvBinding", "SettingSpec", "Section", "all_specs", "spec", "has_spec",
    "sections", "section", "section_of", "effective_default", "coerce", "apply_converter",
    "search", "is_available", "is_value_available", "unavailable_values", "env_names", "keys",
    "choice_values", "evaluate_rule", "LOCKED_IF", "VISIBLE_IF",
    "desktop_settings_map", "desktop_bool_vars", "desktop_str_vars",
]


class _Missing:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self):
        return "MISSING"

    def __bool__(self):
        return False

    def __reduce__(self):
        return (_Missing, ())


MISSING: Any = _Missing()
_NO_DEFAULT = "<none>"     # generator marker: env site without a call-site default


class EnvBinding(NamedTuple):
    """One env variable a setting feeds. ``site`` is translation, glossary, epub, pdf,
    startup (initialize_environment_variables / __init__), save (save_config) or qa."""
    name: str
    site: str
    call_site_default: Any = MISSING


@dataclass(frozen=True)
class SettingSpec:
    key: str
    type: str                         # bool|int|float|str|choice|list|dict|json|path|secret
    default: Any = MISSING            # effective fresh-install default (display only)
    save_default: Any = MISSING       # settings_map default when there is one
    converter: Optional[tuple] = None # ConvSpec, see apply_converter
    var_names: Tuple[str, ...] = ()
    widget_sources: Tuple[str, ...] = ()
    env: Tuple[EnvBinding, ...] = ()
    section: str = ""
    label: str = ""
    tooltip: str = ""
    choices: Optional[tuple] = None
    minimum: Any = None
    maximum: Any = None
    visible_if: Optional[str] = None  # rule ids evaluated by settings_rules (U4)
    locked_if: Optional[str] = None
    platforms: frozenset = frozenset({"desktop", "mobile"})
    discrepancies: Tuple[str, ...] = ()
    # provenance (generated)
    default_source: str = ""          # init | save | nested | dialog | env:<site> | read | oracle
    init_default: Any = MISSING
    parent: Optional[str] = None      # nested settings: qa_scanner_settings / manga_settings / ai_hunter_config
    ui_sites: Tuple[str, ...] = ()
    group: str = ""                   # desktop group box / tab title
    origins: Tuple[str, ...] = ()
    flags: Tuple[str, ...] = ()
    unavailable: Tuple[Tuple[str, str], ...] = ()   # ((platform, reason), ...)
    unavailable_values: Tuple[Tuple[str, str, str], ...] = ()   # ((value, platform, reason), ...)
    readonly: str = ""                # U9: reason a plain tile only shows the value (READONLY_REASONS)

    @property
    def nested(self) -> bool:
        return self.parent is not None

    @property
    def path(self) -> Tuple[str, ...]:
        return tuple(self.key.split("."))

    def env_names(self) -> Tuple[str, ...]:
        return tuple(dict.fromkeys(binding.name for binding in self.env))


class Section(NamedTuple):
    id: str
    title: str
    keys: Tuple[str, ...]
    group: str
    description: str = ""


# =========================================================================== converters
def _safe_int(value, default):
    # save_config.safe_int
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def _safe_float(value, default):
    # save_config.safe_float
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


# Named converter functions referenced by settings_map; resolved lazily from the
# GUI-free modules that own them (never from translator_gui).
_CALL_CONVERTERS = {
    "_format_plain_decimal_setting": ("run_env",),
}


def _resolve_call(name):
    for module_name in _CALL_CONVERTERS.get(name, ()):
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        func = getattr(module, name, None)
        if callable(func):
            return func
    raise LookupError(f"settings_schema: converter {name!r} is not importable from a GUI-free module")


def apply_converter(conv, value):
    """Apply a ConvSpec exactly like the desktop settings_map lambda it came from.

    ConvSpec tuples (generated):
      ('bool',) ('str',) ('int',) ('float',) ('list',) ('dict',)   builtin call
      ('none',)                                                      value unchanged
      ('safe_int'|'safe_float', d, lo, hi, order)                    safe_x(v, d) clamped:
          order '' -> max(lo, x) / min(hi, x) (one bound), 'min_max' -> min(hi, max(lo, x)),
          'max_min' -> max(lo, min(hi, x))
      ('int_if_digits', strip, d)    int(v) if str(v)[.lstrip(strip)].isdigit() else d
      ('choice', allowed, d)         str(v).strip().lower() if that is in allowed else d
      ('str_or', blank, d)           (str(v).strip() if v is not None else blank) or d
      ('call', name)                 a named function (e.g. _format_plain_decimal_setting)
    Exceptions propagate exactly like the lambda's (e.g. ('int',) on 'abc').
    """
    if conv is None:
        return value
    kind = conv[0]
    if kind == "bool":
        return bool(value)
    if kind == "str":
        return str(value)
    if kind == "int":
        return int(value)
    if kind == "float":
        return float(value)
    if kind == "list":
        return list(value)
    if kind == "dict":
        return dict(value)
    if kind == "none":
        return value
    if kind in ("safe_int", "safe_float"):
        _kind, default, lo, hi, order = conv
        result = _safe_int(value, default) if kind == "safe_int" else _safe_float(value, default)
        if order == "min_max":
            return min(hi, max(lo, result))
        if order == "max_min":
            return max(lo, min(hi, result))
        if lo is not None:
            result = max(lo, result)
        if hi is not None:
            result = min(hi, result)
        return result
    if kind == "int_if_digits":
        _kind, strip, default = conv
        text = str(value)
        if strip is not None:
            text = text.lstrip(strip)
        return int(value) if text.isdigit() else default
    if kind == "choice":
        _kind, allowed, default = conv
        return str(value).strip().lower() if str(value).strip().lower() in allowed else default
    if kind == "str_or":
        _kind, blank, default = conv
        return (str(value).strip() if value is not None else blank) or default
    if kind == "call":
        return _resolve_call(conv[1])(value)
    raise ValueError(f"settings_schema: unknown converter {conv!r}")


# =========================================================================== overlay
# Desktop Other Settings group order (the 2-column grid, other_settings.py), then the
# main window, Glossary Manager tabs, QA Scanner, Manga, Library/Reader, Progress
# Manager and Direct Text. Each tuple: (section id, title, group, description).
SECTION_DEFS = (
    ("main.model", "Model & API", "main", "Main window: model, API key and endpoint fields."),
    ("main.prompt", "Profile & System Prompt", "main", "Main window: prompt profile and the system prompt."),
    ("main.run", "Run Settings", "main", "Main window run settings (delays, chapter range, temperature, history, glossary mode, batching)."),
    ("main.file", "Input & Output", "main", "Main window input/output options."),
    ("other.meta_data", "Meta Data", "other_settings", ""),
    ("other.context", "Context Management & Memory", "other_settings", ""),
    ("other.response", "Response Handling & Retry Logic", "other_settings", ""),
    ("other.processing", "Processing Options", "other_settings", ""),
    ("other.processing.extraction", "Chapter Extraction Settings", "other_settings", "Part of Processing Options."),
    ("other.image", "Image Translation & Vision API", "other_settings", ""),
    ("other.anti_duplicate.core", "Anti-Duplicate Parameters · Core Parameters", "other_settings", ""),
    ("other.anti_duplicate.advanced", "Anti-Duplicate Parameters · Advanced", "other_settings", ""),
    ("other.anti_duplicate.stop", "Anti-Duplicate Parameters · Stop Sequences", "other_settings", ""),
    ("other.anti_duplicate.logit_bias", "Anti-Duplicate Parameters · Logit Bias", "other_settings", ""),
    ("other.endpoints", "Custom API Endpoints", "other_settings", ""),
    ("other.debug", "Debug Controls", "other_settings", ""),
    ("other.output", "Output Settings", "other_settings", ""),
    ("other.danger", "Danger Zone", "other_settings", ""),
    ("glossary.general", "General", "glossary", "Glossary Manager: General Settings tab."),
    ("glossary.balanced_full", "Balanced / Full Generation", "glossary", "Glossary Manager: Balanced/Full Generation tab."),
    ("glossary.minimal", "Minimal Generation", "glossary", "Glossary Manager: Minimal Glossary Generation tab."),
    ("glossary.refinement", "Refinement", "glossary", "Glossary Manager: Glossary Refinement tab."),
    ("glossary.unified", "Unified Glossary", "glossary", "Glossary Manager: unified glossary settings dialog."),
    ("glossary.anti_duplicate", "Glossary Anti-Duplicate Parameters", "glossary", "Glossary Manager: anti-duplicate parameters for glossary requests."),
    ("glossary.editor", "Glossary Editor", "glossary", "Glossary Manager: editor preferences."),
    ("qa.settings", "QA Scanner Settings", "qa", "qa_scanner_settings.* and the QA Scanner dialog."),
    ("qa.ai_hunter", "AI Hunter", "qa", "ai_hunter_config.* (duplicate detection)."),
    ("manga.settings", "Manga Translator", "manga", "manga_settings.* and the manga translator."),
    ("library.library", "Library", "library", "EPUB Library shelves."),
    ("library.reader", "Reader", "library", "EPUB Reader preferences."),
    ("progress.manager", "Progress Manager", "progress", "Translation / glossary progress managers."),
    ("glossary.parallel_epub", "Parallel EPUB Glossary", "glossary", "Raw/translated EPUB pair glossary extraction."),
    ("keys.pools", "API Key Pools", "api_keys", "Multi-Key Manager pools (main rotation, fallback and per-feature keys)."),
    ("direct_text.settings", "Direct Text", "direct_text", "Direct Text (chat) dialog settings."),
    ("tools.review", "Review Generator", "tools", "Review generator prompts and modes."),
    ("tools.async", "Async Batch", "tools", "Async (batch API) processing."),
    ("other.stored", "Other Stored Settings", "other_settings",
     "Settings stored in config.json that have no dedicated desktop dialog section."),
    ("internal.state", "Internal State", "internal",
     "Window geometry, last-used paths, caches and bookkeeping values. Not user settings; "
     "kept so a desktop config round-trips untouched."),
)

# Generated desktop UI site -> section id.
SITE_TO_SECTION = {
    "other.meta_data": "other.meta_data",
    "other.context": "other.context",
    "other.response": "other.response",
    "other.processing": "other.processing",
    "other.image": "other.image",
    "other.anti_duplicate": "other.anti_duplicate.core",
    "other.endpoints": "other.endpoints",
    "other.debug": "other.debug",
    "other.output": "other.output",
    "other.danger": "other.danger",
    "other.dialog": "other.stored",
    "main.run": "main.run",
    "main.model": "main.model",
    "main.prompt": "main.prompt",
    "main.file": "main.file",
    "direct_text": "direct_text.settings",
    "glossary.general": "glossary.general",
    "glossary.balanced_full": "glossary.balanced_full",
    "glossary.minimal": "glossary.minimal",
    "glossary.refinement": "glossary.refinement",
    "glossary.unified": "glossary.unified",
    "glossary.anti_duplicate": "glossary.anti_duplicate",
    "glossary.editor": "glossary.editor",
    "glossary.other": "glossary.general",
    "qa": "qa.settings",
    "manga": "manga.settings",
    "library": "library.library",
    "progress": "progress.manager",
    "keys": "keys.pools",
}
ANTI_DUPLICATE_TABS = {
    "Core Parameters": "other.anti_duplicate.core",
    "Advanced": "other.anti_duplicate.advanced",
    "Stop Sequences": "other.anti_duplicate.stop",
    "Logit Bias": "other.anti_duplicate.logit_bias",
}
GROUP_SECTIONS = {
    ("other.processing", "Chapter Extraction Settings"): "other.processing.extraction",
}
# Key patterns -> section, checked before the generated UI sites (first match wins).
SECTION_PATTERNS = (
    (r"^(context_window_size)$", None),
    (r"^qa_scanner_settings\.", "qa.settings"),
    (r"^ai_hunter_config\.", "qa.ai_hunter"),
    (r"^manga_settings\.", "manga.settings"),
    (r"^epub_reader_", "library.reader"),
    (r"^epub_library_", "library.library"),
    (r"^(direct_text_|input_output_)", "direct_text.settings"),
    (r"(^|_)(geometry|splitter_sizes|splitter_state|window_state|col_widths|column_widths)($|_)", "internal.state"),
    (r"^last_|_last_(path|dir|folder|file|used|check|raw_epub|translated_epub)$", "internal.state"),
    (r"_dialog_(size|pos)$|^(main_)?window_(size|pos)$|_tree_heights$|_column_order$", "internal.state"),
    (r"^(_|config_version$|migrated_|.*_migrated$|.*_migration_done$|.*_dialog_shown$)", "internal.state"),
    (r"^(use_)?(multi_api|fallback|glossary|glossary_refinement|metadata|qa_scan|rolling_summary|"
     r"truncation_retry|ai_truncation_detection|inpainter|tts)_keys$|^use_main_key_fallback$|"
     r"^fallback_key_shuffle$|^force_key_rotation$|^rotation_frequency$", "keys.pools"),
    (r"^review_", "tools.review"),
    (r"^async_", "tools.async"),
    (r"^(book_title_|batch_header_|metadata_|translate_metadata)", "other.meta_data"),
    (r"^(custom_image_edit_|vision_ocr_)", "other.image"),
    (r"^parallel_epub_", "glossary.parallel_epub"),
    (r"^(manga_|rapidocr_|qwen2vl_)", "manga.settings"),
    (r"^(active_)?assistant_prompt", "main.prompt"),
)
# Fallback placements for keys the generator saw no desktop dialog for.
FALLBACK_PATTERNS = (
    (r"^glossary_", "glossary.general"),
    (r"(^|_)(tts|audio)(_|$)", "other.image"),
    (r"_endpoint$|_base_url$|^openai_|^azure_", "other.endpoints"),
    (r"^(enable_)?(pdf|epub)_", "other.output"),
)
# Explicit key -> section placements (curated; win over everything else).
SECTION_OVERRIDES = {
    "model": "main.model",
    "api_key": "main.model",
    "active_profile": "main.prompt",
    "prompt_profiles": "main.prompt",
    "chapter_range": "main.run",
    "use_spine_order": "main.run",
    "delay": "main.run",
    "api_queue": "main.run",
    "thread_submission_delay": "main.run",
    "translation_temperature": "main.run",
    "disable_temperature": "main.run",
    "translation_history_limit": "main.run",
    "translation_history_rolling": "main.run",
    "contextual": "main.run",
    "token_limit": "main.run",
    "token_limit_disabled": "main.run",
    "max_output_tokens": "main.run",
    "auto_glossary_mode": "main.run",
    "output_language": "main.run",
    "REMOVE_AI_ARTIFACTS": "main.run",
    "auto_update_check": "other.context",
    "vertex_ai_location": "main.model",
    "google_cloud_credentials": "main.model",
    "replicate_api_key": "other.endpoints",
    "use_thread_pool_extraction": "other.processing.extraction",
    # U9: the Glossary Refinement tab's "Request mode" combo (GlossaryManager_GUI refinement tab)
    "glossary_refinement_chunking_mode": "glossary.refinement",
    # U9: bookkeeping of the Custom Fields editor (removing the default 'description' field), not a setting
    "custom_field_description_removed": "internal.state",
    # legacy keys that desktop startup migrates or removes (kept so old configs round-trip)
    "conservative_batching": "internal.state",
    "conservative_batch_multiplier": "internal.state",
    "max_images_per_chapter": "internal.state",
    "compress_glossary_strict_gender_matching": "internal.state",
    # U9 gap audit: not top-level settings (a field of every custom_entry_types entry; the old EPUB
    # layout flag the desktop Other Settings dialog migrates to epub_layout_mode and removes)
    "has_gender": "internal.state",
    "selected_files": "internal.state",  # U9 gap audit: the desktop's runtime file selection, not a setting
    "legacy_structure": "internal.state",
    # U9: the TTS endpoint override sits with the other endpoint overrides (Endpoints › Text-to-speech)
    "openai_tts_endpoint": "other.endpoints",
    # U9: the "Configure All" dialog's ⚙️ Advanced tab (metadata_batch_translator), with its prompts
    "lang_prompt_behavior": "other.meta_data",
    "forced_source_lang": "other.meta_data",
    # U9: the Multi-Key Manager footer's refusal pattern length limit, next to Disable refusal checks
    "refusal_pattern_length_limit": "other.response",
    # U9: the Glossary Manager's Balanced/Full tab shows the Single Pass header prompt
    "single_pass_glossary_header_prompt": "glossary.balanced_full",
}

# Platform availability: (regex on key, platforms it is unavailable on, reason). The
# setting stays visible (disabled, with the reason) and its stored value round-trips.
UNAVAILABLE_RULES = (
    (r"^tor_|_tor_|tor_proxy", ("mobile",),
     "Tor routing is not available on mobile (no bundled Tor client)."),
    (r"^(auto_dpi_scale|dpi_scale\w*|gui_scale\w*|gui_font_scale|ui_scale\w*)$", ("mobile",),
     "Desktop DPI / GUI scaling. The mobile app follows the system display and text size."),
    (r"^(authza_|glm_)|_authza_|authza_use_general_api|glm_access_mode|glm_proxy", ("mobile",),
     "AuthZA / GLM access modes use desktop-only routes (excluded on mobile)."),
    # U9: AuthND / Gemini Free run in the hidden in-app browser (browser_driver + the mobile
    # WebViewBridge), so their NIM / AuthND token helper and browser chunking settings apply on
    # mobile; only the helper-subprocess limit has nothing to limit there (FEATURE_MAP
    # auth-routes #21, other-settings #57).
    (r"^authnd_token_subprocess_concurrency$", ("mobile",),
     "No helper subprocesses on mobile: AuthND tokens come from the hidden in-app browser, "
     "limited by Token concurrency."),
    (r"(^|_)(auto_install_update|update_install|install_updates|auto_download_update|update_channel_install)($|_)", ("mobile",),
     "Self-installing updates are desktop-only. Mobile shows a 'new version' link instead."),
    (r"mousewheel|mouse_wheel|wheel_lock|wheel_locked", ("mobile",),
     "Mouse-wheel lock has no meaning on touch screens."),
    (r"key_tree.*(zoom|font|height)|tree_zoom|multi_key_tree_font", ("mobile",),
     "Key-tree zoom belongs to the desktop Multi-Key Manager tree view."),
    (r"^enable_gui_yield$|gui_yield", ("mobile",),
     "GUI responsiveness yielding is a desktop Qt event-loop setting; the mobile UI runs on its own loop."),
    (r"^(antigravity|ocagy|ocz|autharena|ollamapull)_|_(antigravity|ocagy|autharena|ollamapull)_", ("mobile",),
     "This route is excluded on mobile (npm/bun or desktop binary)."),
    (r"(claude_code|grok_cli|cli_credential|import_cli)", ("mobile",),
     "Claude Code / Grok CLI credential import reads desktop CLI stores (excluded on mobile)."),
    (r"(process_priority|cpu_affinity|priority_class)", ("mobile",),
     "Process priority / CPU affinity cannot be changed by mobile apps."),
    # Worker-process settings the backend ignores when mobile_runtime.processes_available()
    # is False: pdf_fast_extractor forces 1 worker, TransateKRtoEN skips async PDF extraction,
    # local_inpainter never starts its worker process.
    (r"^(pdf_extraction_workers|pdf_async_page_threshold)$", ("mobile",),
     "PDF extraction runs single-process on mobile (no worker processes), so this setting has no effect."),
    (r"^manga_settings\.inpainting\.disable_worker_process$", ("mobile",),
     "Local inpainting always runs in-process on mobile (no worker processes), so this setting has no effect."),
    # U8 manga: torch-only model tooling. Mobile downloads ready-made ONNX models (manga_models).
    (r"^manga_settings\.advanced\.(auto_convert_to_onnx|auto_convert_to_onnx_background|quantize_models|"
     r"onnx_quantize|torch_precision)$", ("mobile",),
     "ONNX conversion / quantization needs PyTorch and the onnx package (desktop only). "
     "Mobile downloads ready-made ONNX models."),
    (r"^qwen2vl_", ("mobile",),
     "Qwen2-VL OCR needs PyTorch + transformers · not available on mobile."),
    (r"^manga_settings\.ocr\.(bubble_model_path|bubble_max_detections_yolo|custom_model_path)$", ("mobile",),
     "YOLOv8 / custom detector models need PyTorch + ultralytics · mobile uses RT-DETR ONNX."),
    # the Manga Settings "experimental editing tools" switch only enables the desktop preview's
    # Brush / Eraser (EXPERIMENTAL_TRANSLATE_ALL is read by manga_image_preview alone)
    (r"^experimental_translate_all$", ("mobile",),
     "Brush and eraser are desktop only (experimental mask painting there; not planned for mobile)."),
    # U9 Tier B (plan dependency rule; src/mobile/pyproject.toml, tests_host/test_tier_b.py):
    # sentence-transformers has no Android/iOS build (PyTorch, hf-xet), so the embeddings stage of
    # silent-truncation detection never runs there (scan_html_folder falls back to the heuristic
    # score, as desktop does without the package).
    (r"^qa_scanner_settings\.truncation_embed_threshold$", ("mobile",),
     "The embeddings stage of silent-truncation detection needs sentence-transformers (PyTorch) · "
     "not available on mobile. Borderline chapters are judged by the heuristic score alone."),
    # U9 feature-map audit: the mobile output root is app storage (Android; it can mirror to
    # Downloads/Glossarion) or the iOS Documents folder; runtime_bootstrap exports OUTPUT_DIRECTORY,
    # which run_env reads before this key (Settings › Data › Storage says the same).
    (r"^output_directory$", ("mobile",),
     "Not on mobile: outputs live in the app's Output folder (Data › Storage; iOS Files › Glossarion, "
     "Android can mirror to Downloads/Glossarion). The desktop value is kept untouched."),
    # qa_scan_runtime always scans with threads on mobile (no worker processes).
    (r"^qa_scanner_settings\.use_thread_executor$", ("mobile",),
     "Locked on mobile: the QA scan always runs in threads (no worker processes)."),
)

# Per-value availability for choice settings whose options include desktop-only backends
# (U8 manga combos): (regex on key, values (case-insensitive), platforms, reason). The value
# stays listed (disabled, with the reason) and a stored value round-trips untouched.
_MANGA_OCR_KEYS = r"^(manga_ocr_provider|ocr_provider)$"
_MANGA_LOCAL_INPAINT_KEYS = r"^(manga_local_inpaint_model|manga_settings\.inpainting\.local_method)$"
UNAVAILABLE_VALUE_RULES = (
    (_MANGA_OCR_KEYS, ("manga-ocr", "qwen2-vl", "easyocr", "doctr"), ("mobile",),
     "Needs PyTorch · not available on mobile."),
    (_MANGA_OCR_KEYS, ("paddleocr",), ("mobile",),
     "Needs PaddlePaddle · not available on mobile."),
    (r"^manga_settings\.ocr\.detector_type$", ("rtdetr",), ("mobile",),
     "RT-DETR (PyTorch) needs PyTorch + transformers · use RT-DETR ONNX on mobile."),
    (r"^manga_settings\.ocr\.detector_type$", ("yolo", "custom"), ("mobile",),
     "YOLOv8 / custom detectors need PyTorch + ultralytics · use RT-DETR ONNX on mobile."),
    (_MANGA_LOCAL_INPAINT_KEYS, ("aot", "lama", "anime", "lama_official", "mat", "qwen_image_edit"), ("mobile",),
     "Torch JIT / checkpoint model · needs PyTorch. Use the ONNX models (aot_onnx, anime_onnx, lama_onnx)."),
    (_MANGA_LOCAL_INPAINT_KEYS, ("ollama", "sd_local"), ("mobile",),
     "Not functional on desktop either (no inpainting backend for this choice)."),
    (r"^(manga_inpaint_method|manga_settings\.inpainting\.method)$", ("hybrid",), ("mobile",),
     "Hybrid runs an ensemble of local models · needs PyTorch models (experimental on desktop)."),
    (r"^manga_settings\.advanced\.ram_cap_mode$", ("hard",), ("mobile",),
     "The hard RAM cap uses a Windows Job Object (desktop only); mobile uses the soft cap."),
    # U9 Tier B: argostranslate needs ctranslate2 + sentencepiece (no Android/iOS builds). The
    # SDLXLIFF reviewer's provider choice (sdlxliff_review_core MACHINE_TRANSLATION_PROVIDER_CONFIG_KEY).
    (r"^sdlxliff_machine_translation_provider$", ("argos", "argos-translate", "argostranslate"), ("mobile",),
     "Argos Translate (offline) needs ctranslate2 + sentencepiece, which have no Android/iOS builds · "
     "not available on mobile."),
)

# Fresh-install values measured by the U0 desktop oracle (tests/parity golden
# 'fresh_install', captured at 4c825d81 and 96af9adb): desktop startup runs GUI handlers and
# save_config, so the effective value differs from the static init default. Each entry:
# key -> (value, reason). Never edit to "fix" a default; record what desktop does.
FRESH_INSTALL_OVERRIDES = {
    "auto_glossary_mode": (
        "off",
        "the startup glossary-shortcut handler selects index 0 ('Off') on a fresh config "
        "(enable_auto_glossary False) and save_config persists it; auto_glossary_mode_var "
        "init default is 'balanced'",
    ),
    "enable_auto_glossary": (
        False,
        "the startup glossary-shortcut handler applies mode 'off', which clears enable_auto_glossary",
    ),
    "append_glossary_auto_load": (
        True,
        "TranslatorGUI.__init__ force-syncs auto-mapping from config auto_glossary_mode "
        "(missing -> 'off' -> auto-mapping on)",
    ),
    "custom_glossary_fields": (
        ["description"],
        "TranslatorGUI.__init__ seeds 'description' on first run unless "
        "custom_field_description_removed is set",
    ),
    "translate_metadata_fields": (
        {"description": True, "subject": True, "title": True},
        "TranslatorGUI.__init__ seeds description + subject when no field is configured, then title",
    ),
    "active_profile": (
        "Universal",
        "_init_variables selects the first prompt profile; the built-in order starts with Universal",
    ),
    "token_limit": (
        200000,
        "the main-window input token limit field shows 200,000 when unset and the startup "
        "save_config stores it",
    ),
}

# Keys whose fresh-install config value differs from the effective (displayed) default
# on purpose: the displayed value is what runs use. Reason shown with the setting.
FRESH_INSTALL_NOTES = {
    key: ("fresh install (U0 oracle): config.json stores '' because the startup save_config "
          "(_glossary_env_mappings) blanks missing glossary prompt keys; runs use the built-in text")
    for key in ("append_glossary_prompt", "unified_auto_glosary_prompt3", "glossary_refinement_system_prompt",
                "glossary_translation_prompt", "glossary_format_instructions",
                # U9: same _glossary_env_mappings list; its built-in text is a DEFAULT_REFS overlay
                "single_pass_glossary_header_prompt")
}

# U9: how a setting behaves differently on Glossarion Mobile (shown with its help, like the
# desktop discrepancies; the value itself is untouched).
MOBILE_NOTES = {
    "ai_hunter_config.ai_hunter_max_workers": (
        "mobile: QA scans run on at most 2 worker threads (qa_scan_runtime.MOBILE_QA_MAX_WORKERS); "
        "a smaller value here still applies"),
    "allow_authgpt_batch_stream_logs": (
        "mobile: applies to AuthGPT / AuthGrok / AuthGem / AuthCD; AuthZA, Arena, Antigravity and OcAgy "
        "are excluded on mobile"),
    # FEATURE_MAP other-settings #139 (Adapted: FFT may be slow; warning note)
    "advanced_watermark_removal": (
        "mobile: the Advanced FFT pass is slow on a phone (several seconds per page); the value stays editable"),
    # FEATURE_MAP manga #52 / #65 (Adapted: capped for phones); services/manga.prepare_run lowers the job's copy
    "manga_settings.advanced.panel_max_workers": (
        "mobile: a manga job runs at most 2 panels at once (each loads its own detector / inpainter); "
        "a higher stored value is lowered for the run only and kept in config.json"),
    "manga_settings.advanced.max_workers": (
        "mobile: a manga job uses at most 2 workers; a higher stored value is lowered for the run only "
        "and kept in config.json"),
}

# U9: settings that work on mobile for only part of what their desktop label names: key -> the short
# text of a ReasonChip on the tile (the full sentence is the MOBILE_NOTES entry). Display only.
MOBILE_PARTIAL_REASONS = {
    "allow_authgpt_batch_stream_logs": "AuthZA / Arena / Antigravity / OcAgy excluded on mobile",
    "advanced_watermark_removal": "Advanced FFT is slow on phones",
    "manga_settings.advanced.panel_max_workers": "Capped to 2 on phones",
    "manga_settings.advanced.max_workers": "Capped to 2 on phones",
}

# U9: generated defaults that are empty although the desktop editor shows, and runs use, a built-in
# text while the stored value is empty: key -> '$ref' target (resolved lazily like other $ref
# defaults). The stored value is untouched; "Default:" / Reset show the real text.
DEFAULT_REFS = {
    # GlossaryManager_GUI._default_single_pass_header_prompt == this constant; extract_glossary_from_epub
    # falls back to it when SINGLE_PASS_GLOSSARY_HEADER_PROMPT is empty
    "single_pass_glossary_header_prompt": "extract_glossary_from_epub:DEFAULT_SINGLE_PASS_GLOSSARY_HEADER_PROMPT",
}

# Defaults computed at runtime on desktop (effective_default returns None for them).
COMPUTED_DEFAULTS = {
    "prompt_profiles": "the built-in prompt profiles (prompt defaults + translator_gui default_prompts)",
    "extraction_workers": "min(8, max(2, cpu_count // 2))",
    "ai_hunter_config.ai_hunter_max_workers": "max(1, cpu_count // 2)",
}

# Allowed values the desktop code checks without a settings_map choice converter or a
# combo box the generator reads choices from (translator_gui / owner_state source noted).
# Display only; coerce() falls back to the default for other values only for keys without
# a settings_map converter. (REMOVE_AI_ARTIFACTS and auto_glossary_mode now come from their
# desktop combos, with labels: settings_schema_data.)
CHOICES_OVERRIDES = {
    "translation_chunk_prompt_role": ("system", "assistant", "user"),     # _init_variables
    "rolling_summary_mode": ("replace", "append"),                         # _on_context_mode_changed
    # U9: desktop radio groups / fixed combos the generator saw no item list for; (value, label)
    # pairs copied from the desktop controls (other_settings / GlossaryManager_GUI /
    # QA_Scanner_GUI / ai_hunter_enhanced). A stored value outside the list stays shown as custom.
    "text_extraction_method": (("standard", "Standard (BeautifulSoup)"),   # other_settings Text Extraction Method
                               ("enhanced", "🚀 Enhanced (html2text)")),
    "file_filtering_level": (("smart", "Smart (Aggressive Filtering)"),    # File Filtering Level
                             ("comprehensive", "Comprehensive (Moderate Filtering)"),
                             ("full", "No Filtering")),
    "enhanced_filtering": (("smart", "Smart (Aggressive Filtering)"),      # mirrors file_filtering_level
                           ("comprehensive", "Comprehensive (Moderate Filtering)"),
                           ("full", "No Filtering")),
    "extraction_mode": (("smart", "Smart (Aggressive Filtering)"),         # save_config: 'enhanced' or the level
                        ("comprehensive", "Comprehensive (Moderate Filtering)"),
                        ("full", "No Filtering"),
                        ("enhanced", "🚀 Enhanced (html2text)")),
    "batching_mode": (("conservative", "Conservative batching"),           # other_settings batching radios
                      ("direct", "Direct batching"),
                      ("aggressive", "No batching")),
    "gemini_service_tier": ("off", "standard", "flex", "fast", "priority"),  # gemini_service_tier_combo
    "epub_layout_mode": (("auto", "Auto"), ("epub2", "EPUB2"), ("epub3", "EPUB3")),  # layout_combo / Converter
    "summary_role": ("user", "system", "both"),                            # rolling summary role_combo
    "duplicate_detection_mode": (("basic", "Basic (Fast) - Original 85% threshold, 1000 chars"),
                                 ("ai-hunter", "AI Hunter - Multi-method semantic analysis"),
                                 ("cascading", "Cascading - Basic first, then AI Hunter")),
    "emergency_glossary_compliance_mode": (("characters", "Characters"), ("all", "All"), ("custom", "Custom")),
    "image_compression_format": (("auto", "Auto (Best quality/size ratio)"), ("webp", "WebP (Best compression)"),
                                 ("jpeg", "JPEG (Wide compatibility)"), ("png", "PNG (Lossless)")),
    "compress_glossary_strict_matching_mode": (("all", "All"), ("gender", "Gender Entries"),   # "Whole term for"
                                               ("custom", "Custom"), ("none", "None")),
    "glossary_entry_type_filter_mode": (("strict", "Strict"), ("loose", "Loose"), ("none", "No Filtering")),
    "glossary_duplicate_key_mode": (("fuzzy", "Fuzzy"), ("auto", "Auto"), ("skip", "Skip")),  # no desktop control
    "glossary_duplicate_algorithm": (("auto", "Auto - Uses all algorithms"),
                                     ("strict", "Strict - High precision, minimal merging"),
                                     ("balanced", "Balanced - Token + Partial matching"),
                                     ("aggressive", "Aggressive - Maximum duplicate detection"),
                                     ("basic", "Basic Only - Simple Levenshtein distance")),
    "glossary_filter_mode": (("all", "All names & terms"), ("only_with_honorifics", "Names with honorifics only"),
                             ("only_without_honorifics", "Names without honorifics & terms")),
    "glossary_refinement_type_mode": (("all", "All Active Entry Types"), ("selected", "Selected Entry Types")),
    "glossary_refinement_chunking_mode": (("separate", "Send each entry type in a separate request"),  # Request mode
                                          ("all", "Send all entry types")),
    "unified_glossary_source_language": tuple(
        (name.lower(), name) for name in (
            "Auto", "Korean", "Japanese", "Chinese", "English", "Spanish", "French", "German", "Italian",
            "Portuguese", "Russian", "Arabic", "Hindi", "Turkish", "Hebrew", "Thai", "Other")),
    "qa_scanner_settings.report_format": (("summary", "Summary only"), ("detailed", "Detailed (recommended)"),
                                          ("verbose", "Verbose (all data)")),
    "qa_scanner_settings.counting_mode": (("sampled", "Character count (sampled) - Fastest"),
                                          ("exact", "Character count (exact) - Default"),
                                          ("word", "Word count (legacy)")),
    "qa_scanner_settings.ai_truncation_prompt_role": ("system", "user"),
    "ai_hunter_config.detection_mode": (("single_method", "Single Method"),
                                        ("multi_method", "Multi-Method Agreement"),
                                        ("weighted_average", "Weighted Average")),
    # metadata_batch_translator "Configure All" › ⚙️ Advanced › Language Detection radios
    "lang_prompt_behavior": (("auto", "Auto-detect and include language (e.g., 'Translate this Korean text')"),
                             ("never", "Never include language (e.g., 'Translate this text')"),
                             ("always", "Always specify language:")),
    # Tools › Headers & metadata translation mode radios (headers_model.METADATA_MODES)
    "metadata_translation_mode": (("together", "Translate together (single API call)"),
                                  ("metadata_separate", "Translate Metadata separately (2 API calls)"),
                                  ("parallel", "Translate separately (parallel API calls)")),
    # U9 gap audit: no desktop control; TransateKRtoEN._vision_ocr_source_prepass_enabled_for_mode reads
    # auto / 1|true|on|source|qa|new / 0|false|off|direct|old (anything else counts as auto)
    "vision_ocr_source_prepass": (("auto", "Auto"), ("on", "On (source prepass)"), ("off", "Off (direct OCR)")),
    # U9 gap audit, Settings › Reader & Library: the desktop Library toolbar / reader values
    # (library_core._ALL_SIZES with the toolbar labels, SORT_DATE / SORT_NAME / SORT_SIZE,
    # reader_doc.READER_LAYOUTS / READER_THEME_NAMES; pinned by tests/parity/test_u9_gap_round4.py)
    "epub_library_card_size": (("2xs", "2XS"), ("xs", "XS"), ("compact", "S"), ("normal", "M"), ("large", "L"),
                               ("xl", "XL"), ("2xl", "2XL"), ("3xl", "3XL"), ("4xl", "4XL"), ("5xl", "5XL"),
                               ("6xl", "6XL")),
    "epub_library_sort": (("date", "Date"), ("name", "A-Z"), ("size", "Size")),
    "epub_reader_layout": (("single_page", "Single page"), ("scroll", "Scroll"), ("all_scroll", "Scroll all"),
                           ("double_page", "Double page")),
    "epub_reader_theme": ((0, "Dark"), (1, "Light"), (2, "Sepia"), (3, "Midnight"), (4, "Forest"), (5, "Rose")),
}

# U9: types the generator inferred from a text widget although the stored value is structured
# (a dict written by the profile bar / the Headers metadata-fields sheet) or numeric (the desktop
# spin boxes store float / int). Display only; desktop tables never read SettingSpec.type.
TYPE_OVERRIDES = {
    "glossary_prompt_profiles": "dict",
    "active_glossary_prompt_profiles": "dict",
    "glossary_prompt_profile_defaults": "dict",
    "translate_metadata_fields": "dict",
    "glossary_compression_factor": "float",
    "glossary_request_merge_count": "int",
    # the Glossary Manager's output token limit entry (QLineEdit, saved as int; -1 = the main limit)
    "glossary_max_output_tokens": "int",
    # the Multi-Key Manager tree's font size (int) and column heights (dict); typed 'secret' by name
    "multi_api_key_tree_font_size": "int",
    "multi_api_key_tree_heights": "dict",
    # U9 gap audit, Settings › Reader & Library / Progress Manager: checkboxes stored as booleans and the
    # Glossary Progress refinement temperature (glossary_progress_core: config.get('temperature', 0.1))
    "epub_details_show_special_files": "bool",
    "retranslation_manual_editing": "bool",
    "retranslation_show_model_info": "bool",
    "sdlxliff_one_column_layout": "bool",
    "sdlxliff_one_row_layout": "bool",
    "sdlxliff_two_column_layout": "bool",
    "temperature": "float",
}

# U9: settings mobile shows but never edits in a plain tile: structured stores a dedicated surface
# edits, and mirrors the backend derives from another key (run_env exports USE_TITLE /
# TRANSLATE_TOC_NCX from skip_title_tag_translation / use_toc_ncx; the glossary mode drives
# enable_auto_glossary). key -> reason shown on the tile.
READONLY_REASONS = {
    "glossary_prompt_profiles": "Edited by the profile bar (Glossary › Balanced/Full, Minimal, Refinement)",
    "active_glossary_prompt_profiles": "Edited by the profile bar (Glossary › Balanced/Full, Minimal, Refinement)",
    "glossary_prompt_profile_defaults": "Edited by the profile bar (Glossary › Balanced/Full, Minimal, Refinement)",
    "translate_metadata_fields": "Edited in Tools › Headers & metadata › Metadata fields",
    "use_title": "Follows Skip title tag translation",
    "translate_toc_ncx": "Follows Use & Translate TOC / PDF bookmarks",
    "enable_auto_glossary": "Follows Glossary mode",
    # the desktop Output Mode selector writes these legacy flags (settings_rules.output_mode_flags)
    "enable_image_translation": "Follows Output mode",
    "enable_image_output_mode": "Follows Output mode",
    "enable_video_output_mode": "Follows Output mode",
    "enable_audio_output_mode": "Follows Output mode",
    "enable_refinement_output_mode": "Follows Output mode",
    # U9 gap audit: the desktop removed this option (other_settings and owner_state force it off and
    # run_env exports USE_HEADER_AS_OUTPUT='0'), so a mobile switch would do nothing
    "use_header_as_output": "Disabled on desktop too: translated titles made surprising file names",
    # not settings: a field of every custom_entry_types entry; the legacy EPUB layout flag
    "has_gender": "A field of each entry type (Glossary › Entry types), not a setting of its own",
    "legacy_structure": ("Legacy EPUB layout flag: the desktop Other Settings dialog migrates it to EPUB layout "
                         "mode and removes it"),
    "selected_files": "The desktop's current file selection, not a setting",
    # U9 gap audit: the desktop Context Mode combo (owner_state._on_context_mode_changed) writes these three as
    # one choice (settings_rules.apply_context_mode); mobile shows the same Context mode selector
    "contextual": "Follows Context mode",
    "use_rolling_summary": "Follows Context mode",
    "rolling_summary_mode": "Follows Context mode",
    # U9 gap audit: the Direct Text dialog's settings; Chat settings › All chats edits them with their real
    # types (the generator saw them as text, so a plain tile would store 'false' as a string)
    **{key: "Edited in Chat settings › All chats" for key in (
        "direct_text_attachment_prompt_role", "direct_text_disable_auto_scroll", "direct_text_disable_thinking",
        "direct_text_force_multipass_off", "direct_text_force_no_glossary", "direct_text_force_simple_mode",
        "direct_text_glossary_override_mode", "direct_text_manual_glossary", "direct_text_output_mode",
        "direct_text_rendered_card_limit", "direct_text_skip_prompt_profile",
        "direct_text_skip_system_prompt_profile", "direct_text_skip_user_prompt_profile",
    )},
}

# U9: curated labels that replace the generated widget text where the generator picked a
# neighbouring label (several settings sharing "Mode" / "Output Resolution" / "Character"), a
# help sentence or a template placeholder, or the text of an inverted desktop checkbox
# (glossary_use_smart_filter is shown non-inverted here). Display only.
LABEL_FIXES = {
    "max_output_tokens": "Max output tokens",  # the desktop row label is "Budget" (a sub-label)
    "translation_chunk_prompt": "Chunk prompt",
    "image_chunk_prompt": "Image chunk prompt",
    "gemini_safety_threshold": "Gemini safety threshold",
    "openrouter_preferred_provider": "Preferred OpenRouter provider",
    "pdf_output_format": "PDF output format",
    "pdf_render_mode": "PDF render mode",
    "scan_phase_mode": "Post-translation scan mode",
    "nanogpt_video_duration": "Video duration",
    "nanogpt_video_resolution": "Video resolution",
    "use_toc_ncx": "Use & Translate TOC / PDF bookmarks",
    "translate_toc_ncx": "Translate TOC / PDF bookmarks",
    "use_title": "Use title tag",
    "auto_glossary_mode": "Glossary mode",
    "enable_auto_glossary": "Automatic glossary generation",
    "gtool_filter_user_prompt": "GTool image scan system prompt",
    "gtool_scan_user_prompt": "GTool image scan user prompt",
    "rolling_summary_system_prompt": "Memory summary system prompt",
    "rolling_summary_user_prompt": "Memory summary user prompt",
    "manual_glossary_prompt3": "Balanced/Full extraction prompt",
    "unified_auto_glosary_prompt3": "Minimal glossary extraction prompt",
    "glossary_use_smart_filter": "Smart filtering (off: send the full text)",
    "glossary_refinement_chunking_mode": "Request mode",
    "openai_tts_endpoint": "TTS endpoint override (/audio/speech)",
    "vision_ocr_source_prepass": "Vision OCR source prepass",
    # U9 gap audit: Settings › Reader & Library / Progress Manager / Direct Text (humanized keys before)
    "epub_details_show_special_files": "Show special files in book details",
    "epub_library_card_size": "Card size",
    "epub_library_sort": "Sort by",
    "epub_reader_font_size": "Reader font size",
    "epub_reader_layout": "Reader layout",
    "epub_reader_line_spacing": "Reader line spacing",
    "epub_reader_native_toc": "Use the book's own table of contents",
    "epub_reader_theme": "Reader theme",
    "retranslation_manual_editing": "Manual editing (Chapters ⋯)",
    "retranslation_show_model_info": "Show model info in the chapter list",
    "sdlxliff_one_column_layout": "SDLXLIFF reviewer: one-column layout",
    "sdlxliff_one_row_layout": "SDLXLIFF reviewer: one-row layout",
    "sdlxliff_two_column_layout": "SDLXLIFF reviewer: two-column layout",
    "temperature": "Glossary Progress refinement temperature (fallback)",
    "direct_text_force_simple_mode": "Force simple mode",
    "direct_text_skip_system_prompt_profile": "Skip prompt profile (system prompt)",
    "direct_text_skip_user_prompt_profile": "Skip prompt profile (user prompt)",
    "lang_prompt_behavior": "Source language in prompts",
    "forced_source_lang": "Language to use (Always specify)",
    "qa_scanner_settings.word_count_multipliers": "Word-count multipliers (per language)",
    "qa_scanner_settings.sdlxliff_tag_retention_threshold": "Minimum source tags retained (0-1, 1 = strict)",
    "qa_scanner_settings.sdlxliff_tag_surplus_tolerance": "Maximum surplus tags allowed (0-1)",
    "qa_scanner_settings.ai_truncation_prompt_role": "AI truncation prompt role",
    "ai_hunter_config.enabled": "AI Hunter enabled",
    "ai_hunter_config.language_detection.enabled": "Language detection enabled",
    **{f"ai_hunter_config.thresholds.{name}": f"{name.title()} threshold (%)"
       for name in ("character", "exact", "pattern", "semantic", "structural", "text")},
    **{f"ai_hunter_config.weights.{name}": f"{name.title()} weight"
       for name in ("character", "exact", "pattern", "semantic", "structural", "text")},
}

# U9: tooltips of desktop controls the generator saw no tooltip for (copied from the control). Display only.
TOOLTIP_OVERRIDES = {
    "glossary_refinement_chunking_mode": (
        "Send all entry types combines them within the token budget. If splitting is needed, "
        "characters and surnames are grouped first, followed by enabled gendered types, then other types. "
        "A type that exceeds the budget on its own is split into smaller requests."),
}

# Rule ids evaluated by settings_rules.evaluate (U4). locked_if: the keys a registered
# settings_rules lock rule covers (context batching, disable temperature, the stream-thinking
# lock of Enable thoughts, the Glossary Manager mode locks). visible_if: the control is
# disabled / hidden on desktop unless the rule holds (it stays visible on mobile, disabled).
LOCKED_IF = {key: "lock:" + key for key in (
    "batching_mode", "translation_temperature", "enable_thoughts",
    "append_glossary", "append_glossary_auto_load", "fuzzy_auto_mapping", "fuzzy_auto_mapping_threshold",
    # U9: the manual factors while their Auto box is on (other_settings / Glossary Manager disable them)
    "compression_factor", "glossary_compression_factor",
)}
VISIBLE_IF = {
    # Other Settings > Response Handling: thinking controls follow their enable toggles
    "thinking_budget": "thinking:gemini",
    "thinking_level": "thinking:gemini",
    "gpt_effort": "thinking:gpt",
    "openrouter_use_reasoning_tokens": "thinking:gpt",
    "gpt_reasoning_tokens": "thinking:gpt_budget",
    "anthropic_effort": "thinking:anthropic",
    "anthropic_force_adaptive": "thinking:anthropic",
    "anthropic_thinking_budget": "thinking:anthropic_budget",
    # Image Translation & Vision API: the output mode's sub-settings
    "image_output_resolution": "output_mode:image",
    "nanogpt_video_duration": "output_mode:video",
    "nanogpt_video_resolution": "output_mode:video",
    "vision_ocr_batch_translation": "output_mode:vision_request",
    "vision_ocr_batch_size": "output_mode:vision_request",
    "vision_ocr_skip_translation": "output_mode:vision",
    "vision_ocr_keep_images": "output_mode:vision",
    # Glossary Manager: the extraction prompt (modes with extraction), the append format
    # (Append Glossary on) and the Minimal tab's Targeted Extraction Settings (Minimal mode)
    "unified_auto_glosary_prompt3": "glossary:extraction_prompt",
    "append_glossary_prompt": "glossary:append_prompt",
    # Other Settings › Chapter extraction (on_extraction_method_change): the BeautifulSoup option shows only
    # for Text Extraction Method Standard, the html2text options frame only for Enhanced
    "fix_stray_p_gt_bs": "extraction:standard",
    **{key: "extraction:enhanced" for key in (
        "enhanced_preserve_structure", "skip_markdown_to_html", "allow_ai_markdown_headers",
        "enhanced_single_line_break", "convert_br_to_paragraphs", "html2text_escape_snob",
        "preserve_asterisk_separator_lines", "use_markdown2_converter",
    )},
    **{key: "glossary:targeted_extraction" for key in (
        "glossary_min_frequency", "glossary_max_names", "glossary_max_titles", "glossary_context_window",
        "glossary_max_text_size", "glossary_max_sentences", "glossary_include_all_characters",
        "glossary_chapter_split_threshold", "glossary_filter_mode", "strip_honorifics",
    )},
}

# Labels for settings the generator found no widget text for (curated, short).
LABEL_OVERRIDES = {
    "model": "Model",
    "api_key": "API key",
    "active_profile": "Prompt profile",
    "auto_glossary_mode": "Glossary mode",
    "output_language": "Target language",
    "contextual": "Contextual translation",
    "token_limit": "Input token limit",
    "token_limit_disabled": "Disable input token limit",
    "max_output_tokens": "Max output tokens",
}


# =========================================================================== building
_LOCK = threading.Lock()
_STATE = {}


def _data():
    import settings_schema_data
    return settings_schema_data


def _humanize(key: str) -> str:
    leaf = key.rsplit(".", 1)[-1]
    text = leaf.replace("_", " ").strip()
    return text[:1].upper() + text[1:] if text else key


def _section_for(key: str, entry: dict) -> str:
    if key in SECTION_OVERRIDES:
        return SECTION_OVERRIDES[key]
    for pattern, section_id in SECTION_PATTERNS:
        if re.search(pattern, key):
            if section_id is None:
                break          # explicit "use the generated UI site"
            return section_id
    sites = entry.get("ui_sites") or ()
    for site in sites:
        section_id = SITE_TO_SECTION.get(site)
        if not section_id:
            continue
        if section_id == "other.anti_duplicate.core":
            return ANTI_DUPLICATE_TABS.get(entry.get("ui_tab") or "", section_id)
        return GROUP_SECTIONS.get((section_id, entry.get("ui_group") or ""), section_id)
    for pattern, section_id in FALLBACK_PATTERNS:
        if re.search(pattern, key):
            return section_id
    if entry.get("origins") and set(entry["origins"]) <= {"read"}:
        return "internal.state"
    return "other.stored"


def _unavailable_for(key: str):
    out = []
    for pattern, platforms, reason in UNAVAILABLE_RULES:
        if re.search(pattern, key):
            for platform in platforms:
                if platform not in dict(out):
                    out.append((platform, reason))
    return tuple(out)


def _unavailable_values_for(key: str):
    out = []
    seen = set()
    for pattern, values, platforms, reason in UNAVAILABLE_VALUE_RULES:
        if re.search(pattern, key):
            for value in values:
                for platform in platforms:
                    if (value, platform) not in seen:
                        seen.add((value, platform))
                        out.append((value, platform, reason))
    return tuple(out)


def _env_bindings(raw):
    out = []
    for item in raw or ():
        name, site, default = item
        out.append(EnvBinding(name, site, MISSING if default == _NO_DEFAULT else default))
    return tuple(out)


def _build_spec(key: str, entry: dict) -> SettingSpec:
    default = entry.get("default", MISSING)
    default_source = entry.get("default_source", "")
    discrepancies = tuple(entry.get("discrepancies", ()))
    if key in FRESH_INSTALL_OVERRIDES:
        value, reason = FRESH_INSTALL_OVERRIDES[key]
        discrepancies = discrepancies + (
            f"fresh install (U0 oracle): {value!r}; {reason}; generated default {default!r} ({default_source})",)
        default, default_source = value, "oracle"
    if key in FRESH_INSTALL_NOTES:
        discrepancies = discrepancies + (FRESH_INSTALL_NOTES[key],)
    if key in COMPUTED_DEFAULTS:
        discrepancies = discrepancies + (f"computed at runtime: {COMPUTED_DEFAULTS[key]}",)
    if key in MOBILE_NOTES:
        discrepancies = discrepancies + (MOBILE_NOTES[key],)
    if key in DEFAULT_REFS and default in ("", MISSING):
        default = {"$ref": DEFAULT_REFS[key]}  # the stored '' is explained by FRESH_INSTALL_NOTES
    unavailable = _unavailable_for(key)
    platforms = frozenset({"desktop", "mobile"}) - {p for p, _r in unavailable}
    label = LABEL_FIXES.get(key) or entry.get("label") or LABEL_OVERRIDES.get(key) or _humanize(key)
    label = label.replace("&&", "&").strip().rstrip(":").strip()  # Qt mnemonic escape
    choices = CHOICES_OVERRIDES.get(key) or (tuple(entry["choices"]) if entry.get("choices") else None)
    setting_type = TYPE_OVERRIDES.get(key) or entry.get("type", "str")
    if choices and setting_type == "str" and "editable_choices" not in entry.get("flags", ()):
        setting_type = "choice"     # editable desktop combos keep free text (the items are suggestions)
    return SettingSpec(
        key=key,
        type=setting_type,
        default=default,
        save_default=entry.get("save_default", MISSING),
        converter=entry.get("converter"),
        var_names=tuple(entry.get("var_names", ())),
        widget_sources=tuple(entry.get("widget_sources", ())),
        env=_env_bindings(entry.get("env")),
        section=_section_for(key, entry),
        label=label,
        tooltip=TOOLTIP_OVERRIDES.get(key) or entry.get("tooltip", ""),
        choices=choices,
        minimum=entry.get("minimum"),
        maximum=entry.get("maximum"),
        visible_if=VISIBLE_IF.get(key),
        locked_if=LOCKED_IF.get(key),
        platforms=platforms,
        discrepancies=discrepancies,
        default_source=default_source,
        init_default=entry.get("init_default", MISSING),
        parent=entry.get("parent"),
        ui_sites=tuple(entry.get("ui_sites", ())),
        group=entry.get("ui_tab") or entry.get("ui_group") or "",
        origins=tuple(entry.get("origins", ())),
        flags=tuple(entry.get("flags", ())),
        unavailable=unavailable,
        unavailable_values=_unavailable_values_for(key),
        readonly=READONLY_REASONS.get(key, ""),
    )


def _state():
    state = _STATE.get("built")
    if state is not None:
        return state
    with _LOCK:
        state = _STATE.get("built")
        if state is not None:
            return state
        data = _data()
        specs = {key: _build_spec(key, entry) for key, entry in data.SETTINGS.items()}
        by_section = {sid: [] for sid, *_rest in SECTION_DEFS}
        for key in sorted(specs):
            by_section.setdefault(specs[key].section, []).append(key)
        secs = []
        for sid, title, group, description in SECTION_DEFS:
            secs.append(Section(sid, title, tuple(by_section.get(sid, ())), group, description))
        known = {sid for sid, *_rest in SECTION_DEFS}
        for sid in sorted(set(by_section) - known):     # defensive: unknown overlay target
            secs.append(Section(sid, _humanize(sid), tuple(by_section[sid]), "other_settings", ""))
        state = {"specs": specs, "sections": tuple(secs), "order": tuple(data.SETTINGS_MAP_ORDER)}
        _STATE["built"] = state
        return state


def _reset_cache():
    """Drop the built schema (tests regenerate the data module)."""
    with _LOCK:
        _STATE.clear()


# =========================================================================== API
def all_specs():
    """Every setting, sorted by key."""
    specs = _state()["specs"]
    return [specs[key] for key in sorted(specs)]


def keys():
    return sorted(_state()["specs"])


def has_spec(key: str) -> bool:
    return key in _state()["specs"]


def spec(key: str) -> SettingSpec:
    try:
        return _state()["specs"][key]
    except KeyError:
        raise KeyError(f"unknown setting {key!r}") from None


def sections(include_empty: bool = False):
    """Sections in desktop order. ``internal.state`` is included (marked by group
    'internal') so every key belongs to exactly one section."""
    secs = _state()["sections"]
    return [s for s in secs if include_empty or s.keys]


def section(section_id: str) -> Section:
    for item in _state()["sections"]:
        if item.id == section_id:
            return item
    raise KeyError(f"unknown section {section_id!r}")


def section_of(key: str) -> Section:
    return section(spec(key).section)


def env_names(key: str):
    return spec(key).env_names()


_UNRESOLVED = object()
_DENY_IMPORT = re.compile(r"(^|\.)(translator_gui|other_settings|GlossaryManager_GUI|QA_Scanner_GUI|"
                          r"manga_settings_dialog|epub_library|Retranslation_GUI|dpi_setup|PySide6)($|\.)")


def _is_marker(value):
    if not isinstance(value, dict) or not value:
        return False
    return set(value) in ({"$ref"}, {"$expr"}, {"$attr"}, {"$ref", "as"})


def _resolve_marker(value):
    """Resolve a generated default marker: {'$ref': 'module:NAME'[, 'as': 'list'|'tuple']}
    is imported lazily from a GUI-free module; $expr / $attr stay unresolved."""
    if not _is_marker(value):
        return value
    ref = value.get("$ref")
    if not (isinstance(ref, str) and ":" in ref):
        return _UNRESOLVED
    module_name, name = ref.split(":", 1)
    if _DENY_IMPORT.search(module_name):
        return _UNRESOLVED
    try:
        module = importlib.import_module(module_name)
        resolved = getattr(module, name)
    except Exception:
        return _UNRESOLVED
    if value.get("as") == "list":
        return list(resolved)
    if value.get("as") == "tuple":
        return tuple(resolved)
    return resolved


def effective_default(key: str):
    """The value desktop uses for ``key`` on a fresh install (display only).

    ``$ref`` defaults (e.g. built-in prompt texts) are imported lazily from their
    GUI-free module. Returns None when the default is computed at runtime ($expr) or
    there is none."""
    item = spec(key)
    value = item.default
    if value is MISSING or key in COMPUTED_DEFAULTS:
        return None
    resolved = _resolve_marker(value)
    if resolved is _UNRESOLVED:
        return None
    if item.type == "list" and isinstance(resolved, tuple):
        return list(resolved)
    if isinstance(resolved, (list, dict, set)):
        return copy.deepcopy(resolved)
    return resolved


def _truthy(value, default=False):
    # save_config._config_bool semantics
    if value is None:
        return bool(default)
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def coerce(key: str, value):
    """Convert ``value`` like desktop does when it saves ``key``.

    settings_map keys use their ConvSpec (identical to the desktop lambda, including the
    exceptions it raises). Other keys are converted by type: bool like
    save_config._config_bool, int/float like _config_int/_config_float (clamped to
    minimum/maximum, falling back to the default), choice falls back to the default when
    the value is not allowed; everything else is returned unchanged."""
    item = spec(key)
    if item.converter is not None:
        return apply_converter(item.converter, value)
    default = effective_default(key)
    if item.type == "bool":
        return _truthy(value, default)
    if item.type in ("int", "float"):
        cast = int if item.type == "int" else float
        try:
            number = cast(float(value)) if item.type == "int" else float(value)
            if isinstance(number, float) and math.isnan(number):
                raise ValueError
        except (TypeError, ValueError, OverflowError):
            if default is None:
                return value
            try:
                number = cast(default)
            except (TypeError, ValueError, OverflowError):
                return value
        if item.minimum is not None:
            number = max(cast(item.minimum), number)
        if item.maximum is not None:
            number = min(cast(item.maximum), number)
        return number
    if item.type == "choice" and item.choices:
        allowed = choice_values(key)
        text = str(value if value is not None else "").strip()
        if text in allowed:
            return text
        lowered = text.lower()
        if lowered in allowed:
            return lowered
        return default if default is not None else value
    return value


def choice_values(key: str):
    """The stored values of a setting's choices (labels dropped), or None."""
    choices = spec(key).choices
    if not choices:
        return None
    return tuple(c[0] if isinstance(c, tuple) and len(c) == 2 else c for c in choices)


def evaluate_rule(rule_id: str, config):
    """Evaluate a ``visible_if`` / ``locked_if`` rule id (settings_rules.evaluate, lazy import)."""
    import settings_rules
    return settings_rules.evaluate(rule_id, config)


def is_available(key: str, platform: str = "mobile"):
    """(True, '') or (False, reason). Unavailable settings stay visible (disabled)."""
    item = spec(key)
    for plat, reason in item.unavailable:
        if plat == platform:
            return False, reason
    return True, ""


def is_value_available(key: str, value, platform: str = "mobile"):
    """(True, '') or (False, reason) for one option of a choice setting (e.g. the
    'manga-ocr' OCR provider on mobile). Works for keys without a generated spec too
    (``manga_ocr_provider``); unavailable options stay listed, disabled."""
    wanted = str(value if value is not None else "").strip().lower()
    for item_value, plat, reason in _unavailable_values_for(key):
        if plat == platform and item_value == wanted:
            return False, reason
    return True, ""


def unavailable_values(key: str, platform: str = "mobile"):
    """{value: reason} for the options of ``key`` that cannot run on ``platform``."""
    return {value: reason for value, plat, reason in _unavailable_values_for(key) if plat == platform}


def search(query: str, *, limit: Optional[int] = None):
    """Settings matching every word of ``query`` (key, label, tooltip, section title,
    env names, group), best matches first: exact key, key prefix, label prefix, label
    word, key substring, tooltip/env/section. Unavailable settings are included (the UI
    shows them disabled with the is_available reason)."""
    words = [w for w in re.split(r"\s+", (query or "").strip().lower()) if w]
    if not words:
        return []
    state = _state()
    titles = {s.id: s.title.lower() for s in state["sections"]}
    scored = []
    phrase = " ".join(words)
    for key, item in state["specs"].items():
        key_l = key.lower()
        label_l = item.label.lower()
        haystack = " ".join((key_l, key_l.replace("_", " ").replace(".", " "), label_l,
                             item.tooltip.lower(), titles.get(item.section, ""), item.group.lower(),
                             " ".join(item.env_names()).lower()))
        if not all(word in haystack for word in words):
            continue
        if key_l == phrase or key_l == phrase.replace(" ", "_"):
            score = 0
        elif key_l.startswith(phrase.replace(" ", "_")):
            score = 1
        elif label_l.startswith(phrase):
            score = 2
        elif phrase in label_l:
            score = 3
        elif phrase.replace(" ", "_") in key_l:
            score = 4
        elif all(w in label_l or w in key_l for w in words):
            score = 5
        else:
            score = 6
        scored.append((score, len(key), key, item))
    scored.sort(key=lambda t: t[:3])
    result = [t[3] for t in scored]
    return result[:limit] if limit else result


# =========================================================================== desktop tables (U9 P5b)
# settings_map / bool_vars / str_vars as the desktop literals built them, from the generated
# DESKTOP_* rows (see the settings_schema_data header for the row encoding).
_BUILTIN_CONVERTERS = {"bool": bool, "str": str, "int": int, "float": float, "list": list, "dict": dict}
_SPEC_CONVERTERS = ("safe_int", "safe_float", "int_if_digits", "choice", "str_or")
_TABLE_CONVERTERS = {}


def _table_converter(conv):
    """The callable the desktop literal held: the builtin itself, the named function itself,
    None, or a function that applies the lambda's ConvSpec (tier C proves them equal)."""
    kind = conv[0]
    if kind in _BUILTIN_CONVERTERS and len(conv) == 1:
        return _BUILTIN_CONVERTERS[kind]
    if kind == "none":
        return None
    if kind == "call":
        return _resolve_call(conv[1])
    if kind not in _SPEC_CONVERTERS:
        raise ValueError(f"settings_schema: settings_map converter {conv!r} cannot be rebuilt")
    convert = _TABLE_CONVERTERS.get(conv)
    if convert is None:
        def convert(value, _conv=conv):
            return apply_converter(_conv, value)
        convert.conv_spec = conv
        _TABLE_CONVERTERS[conv] = convert
    return convert


def _table_value(spec, owner):
    """A table default: evaluated for every build, like the literal (lists / dicts are fresh)."""
    tag = spec[0]
    if tag == "value":
        value = spec[1]
        return copy.deepcopy(value) if isinstance(value, (list, dict, set)) else value
    if tag == "ref":
        value = _resolve_marker({"$ref": spec[1]})
        if value is _UNRESOLVED:
            raise LookupError(f"settings_schema: desktop table default {spec[1]!r} is not importable")
        return value
    if tag == "attr":
        return getattr(owner, spec[1], _table_value(spec[2], owner))
    if tag == "config":
        return owner.config.get(spec[1], _table_value(spec[2], owner))
    raise ValueError(f"settings_schema: unknown desktop table default {spec!r}")


def desktop_settings_map(owner):
    """save_config's ``settings_map``: ``[(key, [sources...], default, converter), ...]``.

    ``owner`` is the desktop owner (``self`` of ``_apply_live_settings_to_config``): the
    ``getattr(self, 'default_*', '')`` defaults read it."""
    return [(key, list(sources), _table_value(default, owner), _table_converter(conv))
            for key, sources, default, conv in _data().DESKTOP_SETTINGS_MAP]


def desktop_bool_vars(owner):
    """``_init_variables``' ``bool_vars``: ``[(attribute, key, default), ...]``; one default
    reads ``owner.config`` (``missing_finish_as_prohibited`` falls back to the legacy key)."""
    return [(var, key, _table_value(default, owner)) for var, key, default in _data().DESKTOP_BOOL_VARS]


def desktop_str_vars(owner=None):
    """``_init_variables``' ``str_vars``: ``[(attribute, key, default), ...]``."""
    return [(var, key, _table_value(default, owner)) for var, key, default in _data().DESKTOP_STR_VARS]
