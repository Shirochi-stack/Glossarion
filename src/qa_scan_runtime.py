# -*- coding: utf-8 -*-
"""Shared runtime path for QA scans launched from GUI and worker code."""

import json
import os


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
        "foreign_char_threshold": 10,
        "excluded_characters": "",
        "target_language": "english",
        "source_language": "auto",
        "check_encoding_issues": False,
        "check_repetition": True,
        "check_translation_artifacts": True,
        "check_ai_artifacts": True,
        "check_punctuation_mismatch": False,
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
        "QA_FOREIGN_CHAR_THRESHOLD": str(settings.get("foreign_char_threshold", 10)),
        "QA_TARGET_LANGUAGE": str(settings.get("target_language", "english")),
        "QA_CHECK_ENCODING": "1" if settings.get("check_encoding_issues", False) else "0",
        "QA_CHECK_REPETITION": "1" if settings.get("check_repetition", True) else "0",
        "QA_CHECK_ARTIFACTS": "1" if settings.get("check_translation_artifacts", False) else "0",
        "QA_CHECK_AI_ARTIFACTS": "1" if settings.get("check_ai_artifacts", False) else "0",
        "QA_CHECK_GLOSSARY_LEAKAGE": "1" if settings.get("check_glossary_leakage", True) else "0",
        "QA_CHECK_MISSING_IMAGES": "1" if settings.get("check_missing_images", True) else "0",
        "QA_MIN_FILE_LENGTH": str(settings.get("min_file_length", 0)),
        "QA_MIN_DUPLICATE_WORD_COUNT": str(settings.get("min_duplicate_word_count", 500)),
        "QA_REPORT_FORMAT": str(settings.get("report_format", "detailed")),
        "QA_AUTO_SAVE_REPORT": "1" if settings.get("auto_save_report", True) else "0",
        "QA_CACHE_ENABLED": "1" if settings.get("cache_enabled", True) else "0",
        "QA_PARAGRAPH_THRESHOLD": str(settings.get("paragraph_threshold", 0.3)),
        "QA_USE_THREAD_EXECUTOR": "1" if settings.get("use_thread_executor", False) else "0",
        "QA_USE_WORD_COUNT": "1" if counting_mode == "word" else "0",
        "QA_EXACT_CHAR_COUNT": "1" if counting_mode == "exact" else "0",
        "QA_CHECK_MISSING_HTML_TAG": "1" if settings.get("check_missing_html_tag", True) else "0",
        "QA_CHECK_BODY_TAG": "1" if settings.get("check_body_tag", False) else "0",
        "QA_CHECK_MISSING_HEADER_TAGS": "1" if settings.get("check_missing_header_tags", False) else "0",
        "QA_CHECK_PARAGRAPH_STRUCTURE": "1" if settings.get("check_paragraph_structure", True) else "0",
        "QA_CHECK_INVALID_NESTING": "1" if settings.get("check_invalid_nesting", False) else "0",
        "QA_CHECK_SILENT_TRUNCATION": "1" if settings.get("check_silent_truncation", False) else "0",
        "QA_CHECK_POTENTIAL_TRUNCATION": "1" if settings.get("check_potential_truncation", False) else "0",
        "QA_CHECK_AI_TRUNCATION_DETECTION": "1" if settings.get("check_ai_truncation_detection", False) else "0",
        "QA_CHECK_WORD_COUNT_RATIO": "1" if settings.get("check_word_count_ratio", False) else "0",
        "QA_CHECK_MULTIPLE_HEADERS": "1" if settings.get("check_multiple_headers", True) else "0",
        "QA_CHECK_ALL_TEXT_IN_HEADER": "1" if settings.get("check_all_text_in_header", True) else "0",
        "QA_CHECK_INVALID_TAG_MISMATCH": "1" if settings.get("check_invalid_tag_mismatch", False) else "0",
        "QA_CHECK_PUNCTUATION_MISMATCH": "1" if settings.get("check_punctuation_mismatch", False) else "0",
        "QA_FLAG_EXCESS_PUNCTUATION": "1" if settings.get("flag_excess_punctuation", False) else "0",
        "QA_PUNCTUATION_LOSS_THRESHOLD": str(settings.get("punctuation_loss_threshold", 50)),
        "QA_EXCESS_PUNCTUATION_THRESHOLD": str(settings.get("excess_punctuation_threshold", 49)),
        "QA_SOURCE_LANGUAGE": str(settings.get("source_language", "auto")),
    }
    previous = {key: os.environ.get(key) for key in mappings}
    for key, value in mappings.items():
        os.environ[key] = value
    return previous


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
    settings["_live_api_key"] = api_key
    settings["_live_model"] = model
    settings["_live_config"] = live_config
    settings["_output_mode"] = output_mode or "text"

    output_language = live_config.get("output_language") or live_config.get("output_language_var")
    if output_language and not settings.get("target_language"):
        settings["target_language"] = str(output_language).strip().lower()
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
