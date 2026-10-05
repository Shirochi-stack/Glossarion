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
    "search", "is_available", "env_names", "keys", "choice_values", "evaluate_rule",
    "LOCKED_IF", "VISIBLE_IF",
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
    # legacy keys that desktop startup migrates or removes (kept so old configs round-trip)
    "conservative_batching": "internal.state",
    "conservative_batch_multiplier": "internal.state",
    "max_images_per_chapter": "internal.state",
    "compress_glossary_strict_gender_matching": "internal.state",
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
    (r"^authnd_|_authnd_|^nim_token|nim_auto_token", ("mobile",),
     "The NIM / AuthND browser token helper needs the mobile WebView bridge (planned for U9)."),
    (r"^gemini_free_", ("mobile",),
     "Gemini Free browser chunking needs the mobile WebView bridge (planned for U9)."),
    (r"(^|_)(auto_install_update|update_install|install_updates|auto_download_update|update_channel_install)($|_)", ("mobile",),
     "Self-installing updates are desktop-only. Mobile shows a 'new version' link instead."),
    (r"mousewheel|mouse_wheel|wheel_lock|wheel_locked", ("mobile",),
     "Mouse-wheel lock has no meaning on touch screens."),
    (r"key_tree.*(zoom|font)|tree_zoom|multi_key_tree_font", ("mobile",),
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
                "glossary_translation_prompt", "glossary_format_instructions")
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
}

# Rule ids evaluated by settings_rules.evaluate (U4). locked_if: the keys a registered
# settings_rules lock rule covers (context batching, disable temperature, the stream-thinking
# lock of Enable thoughts, the Glossary Manager mode locks). visible_if: the control is
# disabled / hidden on desktop unless the rule holds (it stays visible on mobile, disabled).
LOCKED_IF = {key: "lock:" + key for key in (
    "batching_mode", "translation_temperature", "enable_thoughts",
    "append_glossary", "append_glossary_auto_load", "fuzzy_auto_mapping", "fuzzy_auto_mapping_threshold",
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
    unavailable = _unavailable_for(key)
    platforms = frozenset({"desktop", "mobile"}) - {p for p, _r in unavailable}
    label = entry.get("label") or LABEL_OVERRIDES.get(key) or _humanize(key)
    label = label.strip().rstrip(":").strip()
    choices = CHOICES_OVERRIDES.get(key) or (tuple(entry["choices"]) if entry.get("choices") else None)
    setting_type = entry.get("type", "str")
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
        tooltip=entry.get("tooltip", ""),
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
