# -*- coding: utf-8 -*-
"""Shared runtime path for QA scans launched from GUI and worker code."""

import json
import os
import re
import sys

import mobile_runtime
from emoticon_patterns import DEFAULT_EMOTICON_PATTERNS


CANONICAL_WORD_COUNT_MULTIPLIERS = {
    "english": 1.0,
    "spanish": 1.10,
    "french": 1.10,
    "german": 1.05,
    "italian": 1.05,
    "portuguese": 1.10,
    "russian": 1.15,
    "arabic": 1.15,
    "hindi": 1.10,
    "turkish": 1.05,
    "chinese": 2.50,
    "chinese (simplified)": 2.50,
    "chinese (traditional)": 2.50,
    "japanese": 2.20,
    "korean": 2.30,
    "hebrew": 1.05,
    "thai": 1.10,
    "other": 1.0,
}

DEFAULT_AI_ARTIFACT_PATTERNS = [
    'Sure',
    'Okay',
    'Understood',
    'Of course',
    'Got it',
    'Alright',
    'Certainly',
    "Here's",
    'Here is',
    "I'll translate",
    'I will translate',
    'Let me translate',
    'System:',
    'Assistant:',
    'AI:',
    'User:',
    'Human:',
    'Model:',
    'Translation note',
    'Note',
    "Here's the translation",
    "I've translated",
]

HTML_LIKE_EXTENSIONS = frozenset({".htm", ".html", ".xhtml"})


def is_html_like_path(path):
    """Return whether *path* ends in one or more HTML-family extensions.

    Treat compound output names such as ``chapter.htm.xhtml`` and
    ``chapter.html.xhtml`` as HTML documents.  Splitting repeatedly also
    makes the intended compound-extension behavior explicit instead of
    depending on scattered string-suffix checks.
    """
    if not isinstance(path, (str, os.PathLike)):
        return False

    basename = os.path.basename(os.fspath(path)).casefold()
    stem, extension = os.path.splitext(basename)
    if extension not in HTML_LIKE_EXTENSIONS:
        return False

    # Consume any additional HTML-family suffixes. The final suffix is what
    # identifies the file; this loop deliberately supports double/triple
    # variants without accepting unrelated trailing suffixes such as .bak.
    while stem:
        next_stem, next_extension = os.path.splitext(stem)
        if next_extension not in HTML_LIKE_EXTENSIONS:
            break
        stem = next_stem
    return True


def normalize_target_language(display_text):
    if not display_text:
        return "english"
    s = str(display_text).strip().lower()
    mapping = {
        "english": "english",
        "en": "english",
        "spanish": "spanish",
        "es": "spanish",
        "french": "french",
        "fr": "french",
        "german": "german",
        "de": "german",
        "portuguese": "portuguese",
        "pt": "portuguese",
        "italian": "italian",
        "it": "italian",
        "russian": "russian",
        "ru": "russian",
        "japanese": "japanese",
        "ja": "japanese",
        "korean": "korean",
        "ko": "korean",
        "chinese": "chinese",
        "chinese (simplified)": "chinese (simplified)",
        "chinese (traditional)": "chinese (traditional)",
        "zh": "chinese",
        "zh-cn": "chinese (simplified)",
        "zh-tw": "chinese (traditional)",
        "arabic": "arabic",
        "ar": "arabic",
        "hebrew": "hebrew",
        "he": "hebrew",
        "thai": "thai",
        "th": "thai",
    }
    if s in mapping:
        return mapping[s]
    first = s.split()[0] if s.split() else "english"
    return mapping.get(first, first)


def default_qa_scan_settings():
    return {
        "foreign_char_threshold": 0,
        "excluded_characters": "",
        "whitelist_emoticon_patterns": False,
        "emoticon_patterns": list(DEFAULT_EMOTICON_PATTERNS),
        "emoticon_patterns_are_regex": False,
        "target_language": "english",
        "source_language": "auto",
        "check_encoding_issues": False,
        "check_repetition": True,
        "check_translation_artifacts": True,
        "check_ai_artifacts": False,
        "ai_artifact_patterns": list(DEFAULT_AI_ARTIFACT_PATTERNS),
        "ai_artifact_patterns_are_regex": False,
        "check_ai_thinking_preamble": False,
        "ai_thinking_preamble_patterns": list(DEFAULT_AI_THINKING_PREAMBLE_PATTERNS),
        "ai_thinking_preamble_patterns_are_regex": False,
        "ai_thinking_preamble_sample_size": 500,
        "check_punctuation_mismatch": False,
        "check_quotation_mismatch": False,
        "ignore_excess_quotation_marks": False,
        "only_check_incomplete_quotations": False,
        "ignore_consecutive_missing_quotations": False,
        "skip_stylistic_single_quotes": False,
        "include_square_brackets_as_quotations": False,
        "punctuation_loss_threshold": 49,
        "flag_excess_punctuation": False,
        "excess_punctuation_threshold": 49,
        "check_glossary_leakage": True,
        "check_potential_truncation": False,
        "check_missing_images": True,
        "min_file_length": 0,
        "min_duplicate_word_count": 500,
        "report_format": "detailed",
        "auto_save_report": True,
        "check_missing_html_tag": True,
        "check_missing_header_tags": True,
        "check_missing_beautifulsoup_tags": False,
        "sdlxliff_tag_retention_threshold": 0.9,
        "sdlxliff_tag_surplus_tolerance": 0.05,
        "sdlxliff_min_source_paragraph_tags": 20,
        "check_all_text_in_header": True,
        "check_invalid_tag_mismatch": False,
        "check_invalid_nesting": False,
        "check_silent_truncation": False,
        "truncation_cheap_threshold": 12,
        "truncation_borderline_score": 40,
        "truncation_length_threshold": 30,
        "truncation_embed_threshold": 30,
        "check_ai_truncation_detection": False,
        "ai_truncation_tail_chars": 400,
        "check_word_count_ratio": True,
        "check_multiple_headers": True,
        "warn_name_mismatch": True,
        "quick_scan_sample_size": 1000,
        "cache_enabled": True,
        "cache_auto_size": False,
        "cache_show_stats": False,
        "cache_normalize_text": 10000,
        "cache_similarity_ratio": 20000,
        "cache_content_hashes": 5000,
        "cache_semantic_fingerprint": 2000,
        "cache_structural_signature": 2000,
        "cache_translation_artifacts": 1000,
        "word_count_multipliers": dict(CANONICAL_WORD_COUNT_MULTIPLIERS),
    }


def normalize_qa_scan_settings(settings=None, target_language=None):
    normalized = default_qa_scan_settings()
    if isinstance(settings, dict):
        normalized.update(settings)
    normalized["word_count_multipliers"] = dict(CANONICAL_WORD_COUNT_MULTIPLIERS)
    language = target_language or normalized.get("target_language") or os.getenv("OUTPUT_LANGUAGE", "")
    if language:
        normalized["target_language"] = normalize_target_language(language)
    return normalized


def _env_bool(name, default=False):
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def _env_int(name, default):
    try:
        return int(os.getenv(name, default))
    except Exception:
        return default


def _env_float(name, default):
    try:
        return float(os.getenv(name, default))
    except Exception:
        return default


def _json_list_from_env(name):
    raw = os.getenv(name, "").strip()
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


#: Glossarion Mobile runs QA scans on a thread pool (Android / iOS have no process pools) with at
#: most this many workers (``AI_HUNTER_MAX_WORKERS``). The desktop never applies either.
MOBILE_QA_MAX_WORKERS = 2


def mobile_qa_forcing_active():
    """True on Glossarion Mobile (wherever ``mobile_runtime`` reports no process pools)."""
    return not mobile_runtime.processes_available()


def mobile_qa_max_workers():
    """Worker cap of a mobile scan: a smaller positive ``AI_HUNTER_MAX_WORKERS`` wins."""
    try:
        configured = int(str(os.getenv("AI_HUNTER_MAX_WORKERS", "") or "").strip() or 0)
    except (TypeError, ValueError):
        configured = 0
    if 0 < configured < MOBILE_QA_MAX_WORKERS:
        return configured
    return MOBILE_QA_MAX_WORKERS


def mobile_qa_env_overrides():
    """QA env forced on top of the settings mirror on mobile (empty on desktop)."""
    if not mobile_qa_forcing_active():
        return {}
    return {
        "QA_USE_THREAD_EXECUTOR": "1",
        "AI_HUNTER_MAX_WORKERS": str(mobile_qa_max_workers()),
    }


def qa_scan_cache_config_from_settings(qa_settings):
    settings = qa_settings if isinstance(qa_settings, dict) else {}
    cache_config = {
        "enabled": settings.get("cache_enabled", True),
        "auto_size": settings.get("cache_auto_size", False),
        "show_stats": settings.get("cache_show_stats", False),
        "sizes": {},
    }
    for cache_name in (
        "normalize_text",
        "similarity_ratio",
        "content_hashes",
        "semantic_fingerprint",
        "structural_signature",
        "translation_artifacts",
    ):
        size = settings.get(f"cache_{cache_name}", None)
        if size is not None:
            cache_config["sizes"][cache_name] = None if size == -1 else size
    return cache_config


def apply_qa_scan_env_from_settings(qa_settings):
    settings = qa_settings if isinstance(qa_settings, dict) else {}
    counting_mode = str(settings.get("counting_mode", "") or "").strip().lower()
    mappings = {
        "QA_FOREIGN_CHAR_THRESHOLD": str(settings.get("foreign_char_threshold", 0)),
        "QA_TARGET_LANGUAGE": str(settings.get("target_language", "english")),
        "QA_WHITELIST_EMOTICON_PATTERNS": "1" if settings.get(
            "whitelist_emoticon_patterns", False
        ) else "0",
        "QA_EMOTICON_PATTERNS_JSON": json.dumps(
            settings.get("emoticon_patterns", DEFAULT_EMOTICON_PATTERNS),
            ensure_ascii=False,
        ),
        "QA_EMOTICON_PATTERNS_ARE_REGEX": "1" if settings.get(
            "emoticon_patterns_are_regex", False
        ) else "0",
        "QA_CHECK_ENCODING": "1" if settings.get("check_encoding_issues", False) else "0",
        "QA_CHECK_REPETITION": "1" if settings.get("check_repetition", True) else "0",
        "QA_CHECK_ARTIFACTS": "1" if settings.get("check_translation_artifacts", False) else "0",
        "QA_CHECK_AI_ARTIFACTS": "1" if settings.get("check_ai_artifacts", False) else "0",
        "QA_AI_ARTIFACT_PATTERNS_JSON": json.dumps(
            settings.get("ai_artifact_patterns", DEFAULT_AI_ARTIFACT_PATTERNS)
        ),
        "QA_AI_ARTIFACT_PATTERNS_ARE_REGEX": "1" if settings.get(
            "ai_artifact_patterns_are_regex", False
        ) else "0",
        "QA_CHECK_AI_THINKING_PREAMBLE": "1" if settings.get("check_ai_thinking_preamble", False) else "0",
        "QA_AI_THINKING_PREAMBLE_PATTERNS_JSON": json.dumps(
            settings.get("ai_thinking_preamble_patterns", DEFAULT_AI_THINKING_PREAMBLE_PATTERNS)
        ),
        "QA_AI_THINKING_PREAMBLE_PATTERNS_ARE_REGEX": "1" if settings.get(
            "ai_thinking_preamble_patterns_are_regex", False
        ) else "0",
        "QA_AI_THINKING_PREAMBLE_SAMPLE_SIZE": str(
            settings.get("ai_thinking_preamble_sample_size", 500)
        ),
        "QA_CHECK_GLOSSARY_LEAKAGE": "1" if settings.get("check_glossary_leakage", True) else "0",
        "QA_CHECK_MISSING_IMAGES": "1" if settings.get("check_missing_images", True) else "0",
        "QA_MIN_FILE_LENGTH": str(settings.get("min_file_length", 0)),
        "QA_MIN_DUPLICATE_WORD_COUNT": str(settings.get("min_duplicate_word_count", 500)),
        "QA_REPORT_FORMAT": str(settings.get("report_format", "detailed")),
        "QA_AUTO_SAVE_REPORT": "1" if settings.get("auto_save_report", True) else "0",
        "QA_CACHE_ENABLED": "1" if settings.get("cache_enabled", True) else "0",
        "QA_SDLXLIFF_TAG_RETENTION_THRESHOLD": str(
            settings.get("sdlxliff_tag_retention_threshold", 0.9)
        ),
        "QA_SDLXLIFF_TAG_SURPLUS_TOLERANCE": str(
            settings.get("sdlxliff_tag_surplus_tolerance", 0.05)
        ),
        "QA_SDLXLIFF_MIN_SOURCE_PARAGRAPH_TAGS": str(
            settings.get("sdlxliff_min_source_paragraph_tags", 20)
        ),
        "QA_USE_THREAD_EXECUTOR": "1" if settings.get("use_thread_executor", False) else "0",
        "QA_USE_WORD_COUNT": "1" if counting_mode == "word" else "0",
        "QA_EXACT_CHAR_COUNT": "1" if counting_mode == "exact" else "0",
        "QA_CHECK_MISSING_HTML_TAG": "1" if settings.get("check_missing_html_tag", True) else "0",
        "QA_CHECK_BODY_TAG": "1" if settings.get("check_body_tag", False) else "0",
        "QA_CHECK_MISSING_HEADER_TAGS": "1" if settings.get("check_missing_header_tags", False) else "0",
        "QA_CHECK_MISSING_BEAUTIFULSOUP_TAGS": "1" if settings.get("check_missing_beautifulsoup_tags", False) else "0",
        "QA_CHECK_INVALID_NESTING": "1" if settings.get("check_invalid_nesting", False) else "0",
        "QA_CHECK_SILENT_TRUNCATION": "1" if settings.get("check_silent_truncation", False) else "0",
        "QA_CHECK_POTENTIAL_TRUNCATION": "1" if settings.get("check_potential_truncation", False) else "0",
        "QA_CHECK_AI_TRUNCATION_DETECTION": "1" if settings.get("check_ai_truncation_detection", False) else "0",
        "QA_CHECK_WORD_COUNT_RATIO": "1" if settings.get("check_word_count_ratio", False) else "0",
        "QA_CHECK_MULTIPLE_HEADERS": "1" if settings.get("check_multiple_headers", True) else "0",
        "QA_CHECK_ALL_TEXT_IN_HEADER": "1" if settings.get("check_all_text_in_header", True) else "0",
        "QA_CHECK_INVALID_TAG_MISMATCH": "1" if settings.get("check_invalid_tag_mismatch", False) else "0",
        "QA_CHECK_PUNCTUATION_MISMATCH": "1" if settings.get("check_punctuation_mismatch", False) else "0",
        "QA_CHECK_QUOTATION_MISMATCH": "1" if settings.get("check_quotation_mismatch", False) else "0",
        "QA_IGNORE_EXCESS_QUOTATION_MARKS": "1" if settings.get("ignore_excess_quotation_marks", False) else "0",
        "QA_ONLY_CHECK_INCOMPLETE_QUOTATIONS": "1" if settings.get("only_check_incomplete_quotations", False) else "0",
        "QA_IGNORE_CONSECUTIVE_MISSING_QUOTATIONS": "1" if settings.get("ignore_consecutive_missing_quotations", False) else "0",
        "QA_SKIP_STYLISTIC_SINGLE_QUOTES": "1" if settings.get("skip_stylistic_single_quotes", False) else "0",
        "QA_INCLUDE_SQUARE_BRACKETS_AS_QUOTATIONS": "1" if settings.get("include_square_brackets_as_quotations", False) else "0",
        "QA_FLAG_EXCESS_PUNCTUATION": "1" if settings.get("flag_excess_punctuation", False) else "0",
        "QA_PUNCTUATION_LOSS_THRESHOLD": str(settings.get("punctuation_loss_threshold", 50)),
        "QA_EXCESS_PUNCTUATION_THRESHOLD": str(settings.get("excess_punctuation_threshold", 49)),
        "QA_SOURCE_LANGUAGE": str(settings.get("source_language", "auto")),
    }
    mappings.update(mobile_qa_env_overrides())
    previous = {key: os.environ.get(key) for key in mappings}
    for key, value in mappings.items():
        os.environ[key] = value
    return previous


def active_qa_output_folder_for_source(source_path):
    """Return the translator's exact active output folder for ``source_path``.

    ``EPUB_OUTPUT_DIR`` points at the translated book folder itself, unlike
    ``OUTPUT_DIRECTORY`` which points at the parent output root.  Only accept
    it when its folder name matches the source stem so a stale value from a
    different book cannot redirect a QA scan.
    """
    if not source_path or is_direct_text_qa_path(source_path):
        return None

    active_output = str(os.getenv("EPUB_OUTPUT_DIR", "") or "").strip()
    if not active_output:
        return None

    source_stem = os.path.splitext(os.path.basename(os.path.abspath(source_path)))[0]
    output_path = os.path.abspath(active_output)
    if is_direct_text_qa_path(output_path):
        return None
    output_stem = os.path.basename(output_path.rstrip("/\\"))
    if os.path.normcase(output_stem) != os.path.normcase(source_stem):
        return None
    if not os.path.isdir(output_path):
        return None
    return output_path


def is_direct_text_qa_path(path):
    """Return whether a path belongs to Direct Text's persistent or temp output."""
    if not path:
        return False

    try:
        normalized = os.path.abspath(os.path.expanduser(str(path))).replace('\\', '/')
    except Exception:
        normalized = str(path).replace('\\', '/')

    for component in (part.strip().casefold() for part in normalized.split('/') if part.strip()):
        underscored = component.replace('-', '_').replace(' ', '_')
        if underscored == 'direct_text' or underscored.startswith('direct_text_'):
            return True
        if underscored.startswith('glossarion_direct_text_'):
            return True
        if underscored.startswith('glossarion_input_output_'):
            return True
    return False


def automatic_qa_output_candidates(
    source_path,
    *,
    current_dir,
    script_dir,
    output_root=None,
    platform_name=None,
):
    """Return safe automatic output candidates in translator write priority.

    On Windows/Linux, an unconfigured translation is written beneath the
    process working directory. A same-named directory beside the input EPUB is
    commonly the raw EPUB extraction and must not be auto-scanned. macOS is the
    exception: the translator deliberately writes relative output beside the
    input because frozen apps may start with ``/`` as their working directory.
    """
    if not source_path or is_direct_text_qa_path(source_path):
        return []

    source_path = os.path.abspath(str(source_path))
    source_stem = os.path.splitext(os.path.basename(source_path))[0]
    platform_name = str(platform_name or sys.platform).lower()
    candidates = []

    active_output = active_qa_output_folder_for_source(source_path)
    if active_output:
        candidates.append(active_output)

    if output_root:
        candidates.append(os.path.join(os.path.abspath(str(output_root)), source_stem))

    if platform_name == 'darwin':
        candidates.append(os.path.join(os.path.dirname(source_path), source_stem))

    candidates.extend((
        os.path.join(os.path.abspath(str(current_dir)), source_stem),
        os.path.join(os.path.abspath(str(script_dir)), source_stem),
        os.path.join(os.path.abspath(str(current_dir)), 'src', source_stem),
    ))

    safe_candidates = []
    seen = set()
    for candidate in candidates:
        normalized = os.path.normpath(candidate)
        identity = os.path.normcase(os.path.abspath(normalized))
        if identity in seen or is_direct_text_qa_path(normalized):
            continue
        seen.add(identity)
        safe_candidates.append(normalized)
    return safe_candidates


def restore_env(previous):
    for key, value in (previous or {}).items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _widget_text(widget, default=""):
    try:
        if hasattr(widget, "text"):
            return widget.text()
    except Exception:
        pass
    return default


def _checked_value(value, default=False):
    if value is None:
        return bool(default)
    if hasattr(value, "isChecked"):
        try:
            return bool(value.isChecked())
        except Exception:
            return bool(default)
    if hasattr(value, "get"):
        try:
            return _checked_value(value.get(), default)
        except Exception:
            return bool(default)
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def _live_config_from_owner(owner):
    cfg = dict(getattr(owner, "config", {}) or {})
    api_key = _widget_text(getattr(owner, "api_key_entry", None), cfg.get("api_key", "")).strip()
    model = getattr(owner, "model_var", cfg.get("model", "")) or cfg.get("model", "")
    if hasattr(owner, "batch_translation_var"):
        cfg["batch_translation"] = _checked_value(getattr(owner, "batch_translation_var", None))
    if hasattr(owner, "batch_size_entry"):
        try:
            cfg["batch_size"] = int(_widget_text(getattr(owner, "batch_size_entry", None), "3") or 3)
        except Exception:
            pass
    cfg["api_key"] = api_key or cfg.get("api_key", "")
    cfg["model"] = model or cfg.get("model", "")
    cfg["use_qa_scan_keys"] = _checked_value(
        getattr(owner, "use_qa_scan_keys_var", None),
        cfg.get("use_qa_scan_keys", False),
    )
    cfg["skip_title_tag_translation"] = _checked_value(
        getattr(owner, "skip_title_tag_translation_var", None),
        cfg.get("skip_title_tag_translation", False),
    )
    cfg.setdefault("qa_scan_keys", cfg.get("qa_scan_keys", []))
    cfg["use_ai_truncation_detection_keys"] = _checked_value(
        getattr(owner, "use_ai_truncation_detection_keys_var", None),
        cfg.get("use_ai_truncation_detection_keys", False),
    )
    cfg.setdefault("ai_truncation_detection_keys", cfg.get("ai_truncation_detection_keys", []))
    cfg.setdefault("force_key_rotation", cfg.get("force_key_rotation", True))
    cfg.setdefault("rotation_frequency", cfg.get("rotation_frequency", 1))
    return cfg, api_key, model


def _live_config_from_worker(config):
    api_key = (
        getattr(config, "API_KEY", None)
        or os.getenv("API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("OPENAI_OR_Gemini_API_KEY")
        or os.getenv("GEMINI_API_KEY")
        or ""
    )
    model = getattr(config, "MODEL", None) or os.getenv("MODEL", "")
    live_config = {
        "api_key": api_key,
        "model": model,
        "output_language": os.getenv("OUTPUT_LANGUAGE", "").strip(),
        "translation_temperature": getattr(config, "TEMP", _env_float("TRANSLATION_TEMPERATURE", 0.3)),
        "max_output_tokens": getattr(config, "MAX_OUTPUT_TOKENS", _env_int("MAX_OUTPUT_TOKENS", 8192)),
        "batch_translation": bool(getattr(config, "BATCH_TRANSLATION", _env_bool("BATCH_TRANSLATION", False))),
        "batch_size": int(getattr(config, "BATCH_SIZE", _env_int("BATCH_SIZE", 10)) or 10),
        "use_qa_scan_keys": os.getenv("USE_QA_SCAN_KEYS", "0") == "1",
        "qa_scan_keys": _json_list_from_env("QA_SCAN_API_KEYS") or _json_list_from_env("VISION_API_KEYS"),
        "use_ai_truncation_detection_keys": os.getenv("USE_AI_TRUNCATION_DETECTION_KEYS", "0") == "1",
        "ai_truncation_detection_keys": _json_list_from_env("AI_TRUNCATION_DETECTION_API_KEYS"),
        "force_key_rotation": os.getenv("FORCE_KEY_ROTATION", "1") == "1",
        "rotation_frequency": _env_int("ROTATION_FREQUENCY", 1),
        "skip_title_tag_translation": _env_bool(
            "SKIP_TITLE_TAG_TRANSLATION",
            False,
        ),
        "use_custom_openai_endpoint": os.getenv("USE_CUSTOM_OPENAI_ENDPOINT", "0") == "1",
        "openai_base_url": os.getenv("OPENAI_CUSTOM_BASE_URL", ""),
        "use_gemini_openai_endpoint": os.getenv("USE_GEMINI_OPENAI_ENDPOINT", "0") == "1",
        "gemini_openai_endpoint": os.getenv("GEMINI_OPENAI_ENDPOINT", ""),
    }
    return live_config, api_key, model


def prepare_qa_scan_settings(qa_settings, owner=None, config=None, output_mode=None):
    if owner is not None:
        live_config, api_key, model = _live_config_from_owner(owner)
        if output_mode is None:
            try:
                output_mode = owner._get_output_mode() if hasattr(owner, "_get_output_mode") else live_config.get("output_mode", "text")
            except Exception:
                output_mode = live_config.get("output_mode", "text")
    else:
        live_config, api_key, model = _live_config_from_worker(config)
        if output_mode is None:
            output_mode = getattr(config, "OUTPUT_MODE", os.getenv("OUTPUT_MODE", "text")) if config is not None else os.getenv("OUTPUT_MODE", "text")

    settings = normalize_qa_scan_settings(qa_settings, target_language=live_config.get("output_language"))
    # The owner's SAVED config is the single source of truth for the AI
    # truncation toggle — it must beat any stale settings snapshot the
    # caller passed in (e.g. a dict captured before the user turned the
    # check off).
    if owner is not None:
        try:
            live_qa = (getattr(owner, "config", {}) or {}).get("qa_scanner_settings", {}) or {}
            if "check_ai_truncation_detection" in live_qa:
                settings["check_ai_truncation_detection"] = bool(live_qa["check_ai_truncation_detection"])
        except Exception:
            pass
    settings["_live_api_key"] = api_key
    settings["_live_model"] = model
    settings["_live_config"] = live_config
    settings["_output_mode"] = output_mode or "text"
    settings["skip_title_tag_translation"] = bool(
        live_config.get(
            "skip_title_tag_translation",
            _env_bool("SKIP_TITLE_TAG_TRANSLATION", False),
        )
    )

    output_language = live_config.get("output_language") or live_config.get("output_language_var")
    if output_language and not settings.get("target_language"):
        settings["target_language"] = str(output_language).strip().lower()
    if mobile_qa_forcing_active():
        settings["use_thread_executor"] = True
    return settings


def run_qa_scan_path(
    folder_path,
    log=print,
    stop_flag=None,
    mode="quick-scan",
    qa_settings=None,
    epub_path=None,
    selected_files=None,
    text_file_mode=None,
    progress_path=None,
    owner=None,
    config=None,
):
    """Run the same configured QA scanner path for GUI and translation-worker callers."""
    if is_direct_text_qa_path(folder_path) or is_direct_text_qa_path(epub_path):
        log(
            "⏭️ QA scan skipped: Direct Text folders and temporary Direct Text "
            "outputs are excluded from automatic QA scanning."
        )
        return None

    current_settings = prepare_qa_scan_settings(qa_settings, owner=owner, config=config)
    previous_env = apply_qa_scan_env_from_settings(current_settings)
    try:
        from scan_html_folder import configure_qa_cache, scan_html_folder

        configure_qa_cache(qa_scan_cache_config_from_settings(current_settings))
        return scan_html_folder(
            folder_path,
            log=log,
            stop_flag=stop_flag,
            mode=mode,
            qa_settings=current_settings,
            epub_path=epub_path,
            selected_files=selected_files,
            text_file_mode=text_file_mode,
            progress_path=progress_path,
        )
    finally:
        restore_env(previous_env)
DEFAULT_AI_THINKING_PREAMBLE_PATTERNS = [
    'The user wants a ',
    'The user wants me to ',
    'I need to translate',
    'I need to analyze',
    'I need to verify',
    'Let me translate',
    'Let me analyze',
    'Let me verify',
]


# ---------------------------------------------------------------------------------------------
# Moved from QA_Scanner_GUI in U6 (Glossarion Mobile shares them; QA_Scanner_GUI imports every
# name below): the AI-truncation prompt and Custom-mode defaults of its dialogs, its pure
# owner / path / name helpers (verbatim) and the latest-report search of open_latest_qa_report.
# ---------------------------------------------------------------------------------------------

#: Default prompt of AI Truncation Detection (QA Scanner settings, "Edit Prompt"). The scanner's
#: built-in fallback in scan_html_folder is the same text.
DEFAULT_AI_TRUNCATION_PROMPT = (
    "You are a strict translation quality analyst. Your ONLY job is to determine if "
    "a translated text has been accidentally TRUNCATED (cut off abruptly mid-sentence, "
    "or completely missing the final paragraphs/sentences present in the source).\n"
    "You must be forgiving of minor structural changes, combined paragraphs, or paraphrasing. "
    "Only evaluate the final sentences of the provided texts. Ignore mismatches occurring at the beginning "
    "of the provided tail segment, as it may have been cleanly cut from a larger document.\n"
    "Only answer YES if there is a glaring, obvious failure where the translation explicitly ends prematurely "
    "compared to the source text. If it is a complete, well-formed ending that conveys the general final message, answer NO.\n"
    "Respond with ONLY the word YES or NO. Do not explain."
)

#: Custom scan mode defaults (the Custom mode dialog; thresholds in percent).
DEFAULT_CUSTOM_MODE_SETTINGS = {
    'similarity': 85,
    'semantic': 80,
    'structural': 90,
    'word_overlap': 75,
    'minhash_threshold': 80,
    'consecutive_chapters': 2,
    'check_all_pairs': False,
    'sample_size': 3000,
    'min_text_length': 500,
    'min_duplicate_word_count': 500,
}


def _qa_owner_output_mode(owner):
    try:
        if hasattr(owner, '_get_output_mode'):
            return str(owner._get_output_mode() or '').strip().lower()
    except Exception:
        pass
    try:
        return str(getattr(owner, 'config', {}).get('output_mode', '') or '').strip().lower()
    except Exception:
        return ''


def _qa_owner_uses_truncation_context(owner):
    try:
        settings = getattr(owner, 'config', {}).get('qa_scanner_settings', {}) or {}
        context = (
            settings.get('_qa_context')
            or settings.get('qa_context')
            or settings.get('context')
            or settings.get('_context')
            or ''
        )
        if str(context).strip().lower() in ('truncation', 'qa_truncation'):
            return True
        return bool(settings.get('check_ai_truncation_detection', False))
    except Exception:
        return False


def _qa_vision_ocr_source_path(path, owner=None):
    """Return this book output's OCR-source EPUB used for Vision/truncation QA, when available."""
    if not path or not str(path).lower().endswith('.epub'):
        return path
    if _qa_owner_output_mode(owner) != 'vision' and not _qa_owner_uses_truncation_context(owner):
        return path
    try:
        abs_path = os.path.abspath(path)
        stem, ext = os.path.splitext(os.path.basename(abs_path))
        if stem.lower().endswith('_ocr'):
            stem = stem[:-4]
        candidates = []
        env_candidate = os.getenv('QA_VISION_OCR_SOURCE_EPUB', '').strip() or os.getenv('VISION_OCR_SOURCE_EPUB', '').strip()
        if env_candidate:
            candidates.append(env_candidate)
        output_dir = os.getenv('EPUB_OUTPUT_DIR', '').strip()
        if output_dir:
            output_abs = os.path.abspath(output_dir)
            if os.path.basename(output_abs) == stem:
                candidates.append(os.path.join(output_abs, "OCR", f"{stem}_OCR{ext or '.epub'}"))
            else:
                candidates.append(os.path.join(output_abs, stem, "OCR", f"{stem}_OCR{ext or '.epub'}"))
        override_dir = os.getenv('OUTPUT_DIRECTORY') or os.getenv('OUTPUT_DIR')
        if override_dir:
            candidates.append(os.path.join(os.path.abspath(override_dir), stem, "OCR", f"{stem}_OCR{ext or '.epub'}"))
        try:
            owner_base_dir = getattr(owner, 'base_dir', '') if owner is not None else ''
            if owner_base_dir:
                candidates.append(os.path.join(os.path.abspath(owner_base_dir), stem, "OCR", f"{stem}_OCR{ext or '.epub'}"))
        except Exception:
            pass
        candidates.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), stem, "OCR", f"{stem}_OCR{ext or '.epub'}"))
        for candidate in candidates:
            try:
                candidate_abs = os.path.abspath(candidate)
                candidate_stem = os.path.splitext(os.path.basename(candidate_abs))[0]
                if candidate_stem.lower().endswith('_ocr'):
                    candidate_stem = candidate_stem[:-4]
                if candidate_stem != stem:
                    continue
                if os.path.basename(os.path.dirname(candidate_abs)).lower() != 'ocr':
                    continue
                if os.path.isfile(candidate_abs):
                    return candidate_abs
            except Exception:
                continue
    except Exception:
        pass
    return path


def _normalize_target_language(display_text):
    """Normalize a human-facing target language label to a canonical value.

    The QA pipeline expects simple lowercase identifiers like "english",
    "korean", or "chinese". This helper maps common dropdown labels to
    those canonical forms so detection logic stays stable even if the
    UI wording changes (e.g. "Chinese (Simplified)").
    """
    if not display_text:
        return "english"

    s = display_text.strip().lower()

    mapping = {
        # Core languages
        "english": "english",
        "en": "english",
        "spanish": "spanish",
        "es": "spanish",
        "french": "french",
        "fr": "french",
        "german": "german",
        "de": "german",
        "portuguese": "portuguese",
        "pt": "portuguese",
        "italian": "italian",
        "it": "italian",
        "russian": "russian",
        "ru": "russian",
        "japanese": "japanese",
        "ja": "japanese",
        "korean": "korean",
        "ko": "korean",
        # Chinese variants (keep distinct)
        "chinese": "chinese",
        "chinese (simplified)": "chinese (simplified)",
        "chinese (traditional)": "chinese (traditional)",
        "zh": "chinese",
        "zh-cn": "chinese (simplified)",
        "zh-tw": "chinese (traditional)",
        # RTL / other scripts
        "arabic": "arabic",
        "ar": "arabic",
        "hebrew": "hebrew",
        "he": "hebrew",
        "thai": "thai",
        "th": "thai",
    }

    if s in mapping:
        return mapping[s]

    # Fallback: use the first word (e.g. "english (us)" → "english")
    first = s.split()[0]
    return mapping.get(first, first)


def _normalize_source_language(display_text):
    """
    Normalize source language without collapsing Chinese variants.
    Returns lowercase labels that align with word_count_multipliers keys.
    """
    if not display_text:
        return 'auto'
    s = display_text.strip().lower()
    if s == 'auto':
        return 'auto'
    # Keep distinct variants for Chinese
    if 'chinese' in s:
        if 'traditional' in s:
            return 'chinese (traditional)'
        if 'simplified' in s:
            return 'chinese (simplified)'
        return 'chinese'
    return s



def check_epub_folder_match(epub_name, folder_name, custom_suffixes=''):
    """
    Check if EPUB name and folder name likely refer to the same content
    Uses strict matching to avoid false positives with similar numbered titles
    """
    # Normalize names for comparison
    epub_norm = normalize_name_for_comparison(epub_name)
    folder_norm = normalize_name_for_comparison(folder_name)

    # Direct match
    if epub_norm == folder_norm:
        return True

    # Check if folder has common output suffixes that should be ignored
    output_suffixes = ['_output', '_translated', '_trans', '_en', '_english', '_done', '_complete', '_final']
    if custom_suffixes:
        custom_list = [s.strip() for s in custom_suffixes.split(',') if s.strip()]
        output_suffixes.extend(custom_list)

    for suffix in output_suffixes:
        if folder_norm.endswith(suffix):
            folder_base = folder_norm[:-len(suffix)]
            if folder_base == epub_norm:
                return True
        if epub_norm.endswith(suffix):
            epub_base = epub_norm[:-len(suffix)]
            if epub_base == folder_norm:
                return True

    # Check for exact match with version numbers removed
    version_pattern = r'[\s_-]v\d+$'
    epub_no_version = re.sub(version_pattern, '', epub_norm)
    folder_no_version = re.sub(version_pattern, '', folder_norm)

    if epub_no_version == folder_no_version and (epub_no_version != epub_norm or folder_no_version != folder_norm):
        return True

    # STRICT NUMBER CHECK - all numbers must match exactly
    epub_numbers = re.findall(r'\d+', epub_name)
    folder_numbers = re.findall(r'\d+', folder_name)

    if epub_numbers != folder_numbers:
        return False

    # If we get here, numbers match, so check if the text parts are similar enough
    epub_text_only = re.sub(r'\d+', '', epub_norm).strip()
    folder_text_only = re.sub(r'\d+', '', folder_norm).strip()

    if epub_numbers and folder_numbers:
        return epub_text_only == folder_text_only

    return False


def normalize_name_for_comparison(name):
    """Normalize a filename for comparison - preserving number positions"""
    name = name.lower()
    name = re.sub(r'\.(epub|txt|html?)$', '', name)
    name = re.sub(r'[-_\s]+', ' ', name)
    name = re.sub(r'\[(?![^\]]*\d)[^\]]*\]', '', name)
    name = re.sub(r'\((?![^)]*\d)[^)]*\)', '', name)
    name = re.sub(r'[^\w\s\-]', ' ', name)
    name = ' '.join(name.split())
    return name.strip()


def find_latest_qa_report(override_dir=None, last_report_path=None):
    """Return the newest QA report (``validation_results.html``), or None when there is none.

    The search of ``QAScannerMixin.open_latest_qa_report`` (moved in U6): walk
    ``override_dir`` (OUTPUT_DIRECTORY / the ``output_directory`` setting) when it is a
    directory, else the current working directory, skipping Direct Text folders. The last
    scan's report (``last_report_path``) is the fallback only when the walk finds nothing.
    """
    newest = None
    newest_mtime = -1

    search_roots = []
    if override_dir and os.path.isdir(override_dir):
        search_roots.append(os.path.normpath(override_dir))
    else:
        search_roots.append(os.getcwd())

    for root_dir in search_roots:
        if is_direct_text_qa_path(root_dir):
            continue
        for root, dirs, files in os.walk(root_dir):
            dirs[:] = [
                dirname for dirname in dirs
                if not is_direct_text_qa_path(os.path.join(root, dirname))
            ]
            for fname in files:
                if fname.lower() == "validation_results.html":
                    candidate = os.path.join(root, fname)
                    try:
                        mtime = os.path.getmtime(candidate)
                    except Exception:
                        mtime = 0
                    if mtime > newest_mtime:
                        newest_mtime = mtime
                        newest = candidate

    # Fallback to cached path only if nothing found in current search
    if not newest and last_report_path and os.path.exists(last_report_path):
        newest = last_report_path

    if not newest or not os.path.exists(newest):
        return None
    return newest
