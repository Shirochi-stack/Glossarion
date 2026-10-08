"""Pure helpers behind the settings screens (no Flet import; host-testable).

* ``tile_kind(spec)`` maps a ``settings_schema.SettingSpec`` to a tile:
  switch (bool) · number / slider (int, float; slider when both bounds are
  set and the range is small) · segmented / dropdown (choices; segmented for
  at most 4 short labels) · text (str) · prompt (multi-line prompt texts) ·
  secret · path · list (list of strings) · json (dicts and structured lists).
* value summaries (secrets masked like ``sk-…a1B2``), labels, help lines,
  env names, the group order of Settings home and the windowing math the
  SectionPage uses to keep a jump target inside the built window (Flet's
  ``scroll_to(scroll_key=)`` only reaches built items).

Every accessor is tolerant of schema fields that are missing or shaped
differently (``choices`` as values or ``(value, label)`` pairs, ``env`` as
names or ``EnvBinding`` objects), because the schema module is generated and
grows milestone by milestone.
"""

from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

__all__ = [
    "CURATED_REMNANT_GROUPS",
    "CURATED_SECTIONS",
    "CURATED_SOURCE_PREFIXES",
    "GROUP_ORDER",
    "GROUP_TITLES",
    "HEADING_OVERRIDES",
    "SECTION_GROUPS",
    "SECTION_ORDER",
    "SECTION_TITLES",
    "TILE_KINDS",
    "VIRTUAL_SPECS",
    "VirtualSpec",
    "WINDOW_SIZE",
    "choice_options",
    "config_path",
    "env_names",
    "group_title",
    "grouped_specs",
    "sub_heading",
    "help_line",
    "humanize_key",
    "is_prompt",
    "is_secret",
    "label_for",
    "mask_secret",
    "ordered_groups",
    "plain_text",
    "preview_text",
    "sample_value",
    "slider_params",
    "spec_attr",
    "spec_type",
    "summarize",
    "tile_kind",
    "window_bounds",
]

TILE_KINDS = (
    "switch",
    "number",
    "slider",
    "segmented",
    "dropdown",
    "text",
    "prompt",
    "secret",
    "path",
    "list",
    "json",
)

# Tiles rendered at once in a SectionPage; larger sections slide this window. (64: Response handling &
# retries, 61 tiles since U9 curated the refusal pattern limit in, stays one page with its links.)
WINDOW_SIZE = 64

# Settings home group order (UI_SPEC §4.15); unknown groups follow in first-seen order.
GROUP_ORDER = (
    "General",
    "Translation",
    "Models & keys",
    "Glossary",
    "QA",
    "Manga",
    "Reader & Library",
    "Data",
    "About",
    "Advanced",
)

# settings_schema Section.group ids -> Settings home groups.
GROUP_TITLES = {
    "main": "Translation",
    "other_settings": "Translation",
    "direct_text": "Translation",
    "glossary": "Glossary",
    "qa": "QA",
    "manga": "Manga",
    "library": "Reader & Library",
    "progress": "Reader & Library",
    "api_keys": "Models & keys",
    "tools": "Translation",
    "internal": "Advanced",
}
# Sections whose home group differs from their schema group.
SECTION_GROUPS = {
    "main.model": "Models & keys",
    "other.endpoints": "Models & keys",
    "other.debug": "Data",
    "other.danger": "About",
    "other.stored": "Advanced",
}

#: UI_SPEC §4.15 curated sections (U9): ``(id, title, home group, ((sub-heading, keys), ...))``.
#: Each listed key moves out of its desktop schema section into the curated one (the schema stays the
#: source of the tiles; the desktop dialogs are untouched); keys a build does not have are skipped. A
#: curated id equal to a schema id (``other.response``) keeps that section and only regroups it: its
#: remaining keys follow under "Other settings". A desktop section emptied by the move
#: (``main.run``, ``other.output``) resolves to the curated section that took most of its keys.
CURATED_SECTIONS: tuple = (
    ("translation_defaults", "Translation defaults", "Translation", (
        ("Language & output mode", ("output_language", "output_mode")),
        ("Sampling & limits", ("translation_temperature", "disable_temperature", "max_output_tokens", "token_limit",
                               "token_limit_disabled", "manual_chunk_size")),
        ("Pacing", ("delay", "thread_submission_delay", "api_queue")),
        ("Batch translation", ("batch_translation", "batch_size")),
        ("Glossary", ("auto_glossary_mode", "enable_auto_glossary", "append_glossary", "append_glossary_auto_load",
                      "fuzzy_auto_mapping", "manual_glossary_path", "manual_glossary_map", "title_trim_count",
                      "group_affiliation_trim_count", "traits_trim_count", "refer_trim_count",
                      "locations_trim_count")),
        ("Chapters & cleanup", ("chapter_range", "use_spine_order", "REMOVE_AI_ARTIFACTS")),
        ("Multipass", ("multipass_mode", "multipass_refinement_mode", "partial_b2_entries_per_request",
                       "refinement_system_prompt", "refinement_user_prompt", "refinement_failed_system_prompt",
                       "refinement_failed_user_prompt", "refinement_full_with_raw_system_prompt",
                       "refinement_full_with_raw_user_prompt", "refinement_full_with_raw_raw_header",
                       "refinement_full_with_raw_raw_footer", "refinement_full_with_raw_raw_role",
                       "refinement_partial_system_prompt", "refinement_partial_user_prompt",
                       "refinement_partial_b_system_prompt", "refinement_partial_b_user_prompt",
                       "refinement_partial_b2_system_prompt", "refinement_partial_b2_user_prompt")),
    )),
    ("context_memory", "Context & memory", "Translation", (
        # context_mode: the desktop Context Mode combo (VIRTUAL_SPECS); the three flags follow it
        ("Context mode", ("context_mode", "contextual", "translation_history_limit", "translation_history_rolling",
                          "include_source_in_history")),
        ("Rolling summary", ("use_rolling_summary", "rolling_summary_mode", "rolling_summary_exchanges",
                             "rolling_summary_max_entries", "rolling_summary_max_tokens", "summary_role",
                             "rolling_summary_system_prompt", "rolling_summary_user_prompt")),
    )),
    ("other.response", "Response handling & retries", "Translation", (
        ("Streaming", ("enable_streaming", "stream_thinking_logs", "allow_batch_stream_logs",
                       "allow_authgpt_batch_stream_logs")),
        ("Retries", ("max_retries", "max_retry_tokens", "retry_timeout", "timeout_retry_attempts", "chunk_timeout",
                     "indefinite_rate_limit_retry", "ignore_retry_after")),
        ("Truncation", ("retry_truncated", "truncation_retry_attempts", "char_ratio_truncation_enabled",
                        "char_ratio_truncation_percent", "char_ratio_truncation_attempts",
                        "char_ratio_min_output_chars", "missing_finish_as_prohibited",
                        "unknown_finish_as_prohibited")),
        ("Duplicates", ("duplicate_detection_mode", "duplicate_lookback_chapters", "retry_duplicate_bodies")),
        ("Failure saving", ("save_partial_results", "save_prohibited_results", "preserve_original_text_on_failure")),
        ("Safety checks", ("disable_refusal_checks", "refusal_pattern_length_limit", "disable_empty_safety_heuristic",
                           "disable_qa_marker_checks", "qa_marker_length_limit")),
        ("Stop logic", ("graceful_stop", "wait_for_chunks", "dispatch_order_timeout", "enable_chunk_progress")),
        ("HTTP", ("enable_http_tuning", "connect_timeout", "read_timeout", "http_pool_connections",
                  "http_pool_maxsize", "tor_proxy_enabled")),
        ("Compression factor", ("auto_compression_factor", "compression_factor")),
        ("Parallel extraction", ("enable_parallel_extraction", "extraction_workers")),
        ("NIM / AuthND helpers", ("authnd_token_concurrency", "authnd_token_concurrency_auto",
                                  "authnd_token_subprocess_concurrency", "authnd_token_timeout")),
        ("Gemini Free chunking", ("gemini_free_adaptive_split", "gemini_free_html_splitter",
                                  "gemini_free_html_text_node_transport", "gemini_free_min_subchunk_body_chars",
                                  "gemini_free_subchunk_balancer", "gemini_free_subchunk_concurrency",
                                  "gemini_free_subchunk_payload_format", "gemini_free_subchunk_prompt_chars",
                                  "gemini_free_subchunk_safety_chars", "gemini_free_subchunk_start_delay",
                                  "gemini_free_subchunk_timeout", "gemini_free_subchunk_url_chars")),
    )),
    ("epub_output", "EPUB output", "Translation", (
        ("Layout & navigation", ("epub_layout_mode", "force_ncx_only", "epub_use_html_method")),
        # FEATURE_MAP other-settings #111-#114 "EPUB output" (the desktop shows them with chapter extraction)
        ("Contents", ("disable_epub_gallery", "disable_automatic_cover_creation", "skip_non_spine_special_files",
                      "skip_unreferenced_epub_images")),
        ("Styles", ("attach_css_to_chapters", "epub_css_override_path")),
        ("File names", ("retain_source_extension",)),
        ("Remote images", ("download_remote_image_urls", "remote_image_download_workers",
                           "remote_image_download_interval")),
        # enable_image_compression is the master switch the Vision compression dialog shares (Image & vision
        # links here); the quality controls below only apply while it is on (FEATURE_MAP other-settings #170)
        ("Image compression", ("enable_image_compression", "image_compression_quality", "exclude_cover_compression",
                               "exclude_gif_compression")),
        ("Sidecars", ("output_md", "output_txt", "output_sdlxliff")),
    )),
    ("pdf", "PDF", "Translation", (
        ("Input", ("pdf_output_format", "pdf_async_page_threshold", "pdf_extraction_workers", "pdf_use_toc_sections",
                   "pdf_render_mode")),
        ("Input layout", ("pdf_paragraph_alignment", "pdf_header_alignment", "pdf_paragraph_justification",
                          "pdf_rtl_paragraph_layout")),
        ("Output", ("enable_pdf_output", "pdf_fast_rendering", "pdf_render_batch_size",
                        "pdf_use_rapid_workspace_compiler", "pdf_generate_toc", "pdf_toc_page_numbers",
                        "pdf_page_numbers", "pdf_page_number_alignment")),
    )),
    ("thinking", "Thinking & reasoning", "Models & keys", (
        ("Thoughts", ("enable_thoughts",)),
        ("Gemini", ("enable_gemini_thinking", "thinking_budget", "thinking_level")),
        ("OpenAI / OpenRouter", ("enable_gpt_thinking", "gpt_effort", "gpt_reasoning_tokens",
                                 "openrouter_use_reasoning_tokens", "pass_thinking_all_openai")),
        ("Anthropic", ("enable_anthropic_thinking", "anthropic_effort", "anthropic_force_adaptive",
                       "anthropic_thinking_budget")),
        ("DeepSeek", ("enable_deepseek_thinking", "deepseek_effort", "deepseek_use_responses_api")),
        ("Metadata & TOC requests", ("lightweight_thinking_level", "skip_book_title_thinking",
                                     "skip_metadata_thinking", "skip_toc_thinking")),
    )),
    ("provider_options", "Provider options & safety", "Models & keys", (
        ("Safety filters", ("disable_gemini_safety", "gemini_safety_threshold")),
        ("OpenRouter", ("openrouter_preferred_provider", "openrouter_accept_identity", "openrouter_use_http_only")),
        ("Service tier", ("gemini_service_tier", "force_service_tier_unknown_routes")),
    )),
)

@dataclass(frozen=True)
class VirtualSpec:
    """A settings control that is not one config key (``SchemaAccess.spec`` falls back to these): its tile
    reads and writes through ``state.setting_writes`` (U9: the desktop Context Mode combo)."""

    key: str
    label: str
    type: str = "choice"
    choices: tuple = ()
    default: Any = None
    tooltip: str = ""
    section: str = ""
    group: str = ""
    readonly: str = ""
    virtual: str = ""
    discrepancies: tuple = ()


def _context_mode_spec() -> VirtualSpec:
    from glossarion_mobile.state.setting_writes import CONTEXT_MODE_CHOICES, CONTEXT_MODE_KEY

    return VirtualSpec(
        key=CONTEXT_MODE_KEY, label="Context mode", choices=CONTEXT_MODE_CHOICES, default="off",
        section="context_memory", virtual="context_mode",
        tooltip=("Off · Contextual History · Rolling Summary (Replace / Append): the desktop Context Mode. It sets "
                 "Contextual translation, Use rolling summary and Rolling summary mode together, and Off turns "
                 "batching to No batching (settings_rules.apply_context_mode)."))


#: Settings controls without a config key of their own (key -> VirtualSpec).
VIRTUAL_SPECS: dict = {"context_mode": _context_mode_spec()}


#: Desktop sections the curated map takes keys from (a schema without them is left as it is).
CURATED_SOURCE_PREFIXES = ("main.", "other.")

#: Home group of a desktop section once the curated map moved keys out of it (what Context
#: Management & Memory keeps is the desktop display scaling and the update check).
CURATED_REMNANT_GROUPS = {"other.context": "Advanced"}

#: Mobile titles of desktop sections the curated map regroups (UI_SPEC §4.15 names).
SECTION_TITLES = {
    "other.meta_data": "Metadata, TOC & headers",
    "other.processing": "Processing & extraction",
    "other.processing.extraction": "Chapter extraction",
    "other.image": "Image & vision",
    "other.context": "Desktop display scaling & update check",
}

#: Settings home order of the sections inside their groups (unlisted ones follow in schema order).
SECTION_ORDER = (
    "translation_defaults", "direct_text.settings", "context_memory", "other.response", "other.processing",
    "other.processing.extraction", "other.meta_data", "epub_output", "pdf", "other.image",
    "main.model", "other.endpoints", "thinking", "provider_options", "keys.pools",
)

_SECRET_TYPES = ("secret", "password")
_PROMPT_TYPES = ("prompt", "text", "multiline", "textarea")
_LIST_TYPES = ("list", "tuple", "array")
_JSON_TYPES = ("dict", "json", "object", "map", "mapping")
_SECRET_KEY_HINTS = ("api_key", "apikey", "_secret", "password", "_token", "cookie")
#: Count / size settings whose names contain a secret hint ("_token" in glossary_max_output_tokens,
#: the key-tree font size / heights): never masked (UI_SPEC §5.7 SecretTile is for credentials only).
_NOT_SECRET_KEY = re.compile(r"(_tokens$|(^|_)max_|output_tokens|font_size|_heights?$|_count$|_limit$)")
_PROMPT_KEY_HINTS = ("prompt", "instruction", "template", "system_message")
_SEGMENT_LABEL_MAX = 12


def _is_missing(value: Any) -> bool:
    return value is None or type(value).__name__ == "_Missing"  # settings_schema.MISSING


def spec_attr(spec: Any, name: str, default: Any = None) -> Any:
    value = getattr(spec, name, default)
    return default if _is_missing(value) else value


def spec_type(spec: Any) -> str:
    return str(spec_attr(spec, "type", "str") or "str").strip().lower()


def config_path(spec: Any) -> tuple:
    """Where the value lives in config.json: nested settings (``spec.parent``) are dotted paths."""
    key = str(spec_attr(spec, "key", ""))
    if spec_attr(spec, "parent", None):
        return tuple(key.split("."))
    return (key,)


def sample_value(spec: Any, value: Any = None) -> Any:
    """The stored value, else the literal schema default (``$ref``/``$expr`` markers and MISSING skipped)."""
    if value is not None:
        return value
    default = spec_attr(spec, "default", None)
    if _is_reference(default):
        return None
    return default


def _is_reference(value: Any) -> bool:
    """A generated lazy default (``{'$ref': ...}``, ``{'$expr': ...}``, ``{'$ref': ..., 'as': 'list'}``)."""
    return (isinstance(value, dict) and bool(value)
            and all(str(k).startswith("$") or k == "as" for k in value)
            and any(str(k).startswith("$") for k in value))


_TAG = re.compile(r"</?[A-Za-z][A-Za-z0-9]*(?:\s[^<>]*)?/?>")
_BREAK = re.compile(r"(?i)<(?:br\s*/?|/p|/li|/div|/tr|/h[1-6]|/ul|/ol)\s*>")
_ITEM = re.compile(r"(?i)<li\b[^>]*>")


def plain_text(text: Any) -> str:
    """Qt rich-text tooltips (``<qt><p>…<br>…``) as plain text; plain text passes through."""
    value = str(text or "")
    if _TAG.search(value):
        value = _BREAK.sub("\n", value)
        value = _ITEM.sub("• ", value)
        value = _TAG.sub("", value)
        value = html.unescape(value)
    lines = [re.sub(r"[ \t ]+", " ", line).strip() for line in value.splitlines()]
    out: list[str] = []
    for line in lines:
        if line or (out and out[-1]):
            out.append(line)
    return "\n".join(out).strip()


def group_title(section_id: str, raw_group: Any) -> str:
    """Settings home group of a schema section (schema group ids mapped to UI_SPEC §4.15 groups)."""
    group = str(raw_group or "").strip()
    return SECTION_GROUPS.get(section_id) or GROUP_TITLES.get(group) or group or "Settings"


def humanize_key(key: str) -> str:
    text = str(key).replace("_", " ").replace(".", " ").strip()
    return text[:1].upper() + text[1:] if text else str(key)


def label_for(spec: Any) -> str:
    label = str(spec_attr(spec, "label", "") or "").strip()
    if label:
        return label
    key = str(spec_attr(spec, "key", ""))
    return humanize_key(key.rsplit(".", 1)[-1] if spec_attr(spec, "parent", None) else key)


def help_line(spec: Any, limit: int = 140) -> str:
    text = plain_text(spec_attr(spec, "tooltip", ""))
    if not text:
        return ""
    first = text.splitlines()[0].strip()
    return first if len(first) <= limit else first[: limit - 1].rstrip() + "…"


def env_names(spec: Any) -> list[str]:
    out: list[str] = []
    for binding in spec_attr(spec, "env", ()) or ():
        if isinstance(binding, str):
            name = binding
        elif isinstance(binding, (tuple, list)) and binding:
            name = str(binding[0])
        else:
            name = getattr(binding, "name", None) or getattr(binding, "env", None) or ""
        name = str(name).strip()
        if name and name not in out:
            out.append(name)
    return out


def choice_options(spec: Any) -> list[tuple[Any, str]]:
    """``[(value, label), ...]`` from ``spec.choices`` (values, pairs or a mapping)."""
    raw = spec_attr(spec, "choices", None)
    if not raw:
        return []
    out: list[tuple[Any, str]] = []
    items: Iterable[Any] = raw.items() if isinstance(raw, dict) else raw
    for item in items:
        if isinstance(item, (tuple, list)) and len(item) == 2:
            value, label = item[0], item[1]
        else:
            value, label = item, item
        out.append((value, str(label) if label not in (None, "") else str(value)))
    return out


def is_secret(spec: Any) -> bool:
    kind = spec_type(spec)
    key = str(spec_attr(spec, "key", "")).lower()
    if key != "api_key" and _NOT_SECRET_KEY.search(key.rsplit(".", 1)[-1]):
        return False
    if kind in _SECRET_TYPES:
        return True
    if kind not in ("str", "string"):
        return False
    return key == "api_key" or any(hint in key for hint in _SECRET_KEY_HINTS)


def is_prompt(spec: Any) -> bool:
    kind = spec_type(spec)
    if kind in _PROMPT_TYPES:
        return True
    if kind not in ("str", "string"):
        return False
    key = str(spec_attr(spec, "key", "")).lower()
    if any(hint in key for hint in _PROMPT_KEY_HINTS):
        return True
    default = spec_attr(spec, "default", "")
    return isinstance(default, str) and ("\n" in default or len(default) > 160)


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def slider_params(spec: Any) -> Optional[tuple[float, float, int]]:
    """(min, max, divisions) when a bounded number fits a slider, else None."""
    kind = spec_type(spec)
    if kind not in ("int", "float"):
        return None
    low, high = _number(spec_attr(spec, "minimum", None)), _number(spec_attr(spec, "maximum", None))
    if low is None or high is None or high <= low:
        return None
    span = high - low
    if kind == "int":
        if span > 100:
            return None
        return low, high, int(span)
    if span > 10:
        return None
    step = 0.01 if span <= 1 else 0.1
    return low, high, max(1, int(round(span / step)))


def tile_kind(spec: Any, value: Any = None) -> str:
    """Tile for ``spec``; ``value`` (the stored value, when known) settles mis-typed specs.

    The generator types some non-string settings as ``secret`` by name (pool lists such
    as ``multi_api_keys``, flags such as ``use_multi_api_keys``); those get the tile of
    their actual value type instead of a masked text field.
    """
    kind = spec_type(spec)
    if kind in _SECRET_TYPES:
        sample = sample_value(spec, value)
        if isinstance(sample, bool):
            return "switch"
        if isinstance(sample, (list, tuple, dict)):
            return "json"
    if kind in ("bool", "boolean"):
        return "switch"
    options = choice_options(spec)
    if options or kind in ("choice", "enum", "combo"):
        if options and len(options) <= 4 and all(len(label) <= _SEGMENT_LABEL_MAX for _v, label in options):
            return "segmented"
        return "dropdown"
    if is_secret(spec):
        return "secret"
    if kind in ("int", "float", "number"):
        return "slider" if slider_params(spec) is not None else "number"
    if kind == "path":
        return "path"
    if kind in _LIST_TYPES:
        sample = sample_value(spec, value)  # unknown contents -> JSON, never stringify dict items
        if isinstance(sample, (list, tuple)) and all(isinstance(v, str) for v in sample):
            return "list"
        default = spec_attr(spec, "default", None)
        if sample is None and _is_reference(default) and default.get("as") == "list":
            return "list"  # a list constant of a backend module (QA emoticon patterns)
        return "json"
    if kind in _JSON_TYPES:
        return "json"
    if is_prompt(spec):
        return "prompt"
    return "text"


def mask_secret(value: Any) -> str:
    """``sk-…a1B2`` style mask; never shows more than 3 leading and 4 trailing characters."""
    text = "" if value is None else str(value)
    if not text:
        return "Not set"
    if text.startswith("ENC:"):
        return "Encrypted (key unavailable)"
    if len(text) <= 10:
        return "•" * min(len(text), 8)
    return f"{text[:3]}…{text[-4:]}"


def preview_text(value: Any, lines: int = 2, limit: int = 160) -> str:
    text = "" if value is None else str(value)
    parts = [line.strip() for line in text.strip().splitlines() if line.strip()][:lines]
    joined = " · ".join(parts)
    return joined if len(joined) <= limit else joined[: limit - 1].rstrip() + "…"


def _format_number(value: Any) -> str:
    if isinstance(value, float):
        text = f"{value:.4f}".rstrip("0").rstrip(".")
        return text if text not in ("", "-0") else "0"
    return str(value)


def summarize(value: Any, kind: str, spec: Any = None) -> str:
    """One-line description of ``value`` for a tile subtitle."""
    if kind == "secret":
        return mask_secret(value)
    if value is None:
        return "Not set"
    if kind == "switch":
        return "On" if bool(value) else "Off"
    if kind in ("segmented", "dropdown"):
        for option, label in choice_options(spec) if spec is not None else ():
            if option == value or str(option) == str(value):
                return label
        return str(value) if value != "" else "Not set"
    if kind in ("number", "slider"):
        return _format_number(value)
    if kind == "prompt":
        text = preview_text(value)
        return text or "Empty"
    if kind == "path":
        text = str(value)
        if not text:
            return "None"
        return text.replace("\\", "/").rsplit("/", 1)[-1] or text
    if kind == "list":
        items = list(value) if isinstance(value, (list, tuple)) else []
        if not items:
            return "Empty"
        shown = ", ".join(str(v) for v in items[:3])
        return f"{len(items)} item{'s' if len(items) != 1 else ''}: {shown}{'…' if len(items) > 3 else ''}"
    if kind == "json":
        if isinstance(value, dict):
            return f"{len(value)} entr{'ies' if len(value) != 1 else 'y'}"
        if isinstance(value, (list, tuple)):
            return f"{len(value)} item{'s' if len(value) != 1 else ''}"
        try:
            return preview_text(json.dumps(value, ensure_ascii=False), 1, 80)
        except (TypeError, ValueError):
            return str(value)[:80]
    text = str(value)
    return text if text else "Empty"


def window_bounds(index: int, total: int, size: int = WINDOW_SIZE) -> tuple[int, int]:
    """``[start, end)`` of a window of ``size`` items with ``index`` near its centre."""
    if total <= size:
        return 0, total
    index = min(max(0, index), total - 1)
    start = max(0, min(index - size // 2, total - size))
    return start, start + size


def ordered_groups(names: Sequence[str]) -> list[str]:
    seen: list[str] = []
    for name in names:
        if name not in seen:
            seen.append(name)
    known = [g for g in GROUP_ORDER if g in seen]
    return known + [g for g in seen if g not in GROUP_ORDER]


#: Sub-headings that replace the generator's desktop group where it picked a neighbouring group box (the
#: Custom API Endpoints dialog reads as one "AuthZA / GLM Access Mode" group; output_directory sits
#: under "Context Management & Memory") or where mobile shows a key next to its partner. Display only.
HEADING_OVERRIDES = {
    "use_custom_openai_endpoint": "Custom OpenAI endpoint",
    "openai_base_url": "Custom OpenAI endpoint",
    "azure_api_version": "Custom OpenAI endpoint",
    "override_gemma_for_custom_endpoint": "Custom OpenAI endpoint",
    "use_gemini_openai_endpoint": "Gemini custom endpoint",
    "gemini_openai_endpoint": "Gemini custom endpoint",
    "force_native_anthropic": "Anthropic custom endpoint",
    "anthropic_base_url": "Anthropic custom endpoint",
    "use_custom_image_edit_endpoint": "Custom image edit endpoint",
    "custom_image_edit_endpoint": "Custom image edit endpoint",
    "groq_base_url": "Provider base URLs",
    "fireworks_base_url": "Provider base URLs",
    "tts_voice": "Text-to-speech",
    "openai_tts_endpoint": "Text-to-speech",
    "output_directory": "Output folder",
    "lang_prompt_behavior": "Configure All › Advanced",
    "forced_source_lang": "Configure All › Advanced",
    "metadata_translation_mode": "Custom metadata",
    "translate_metadata_fields": "Custom metadata",
    "metadata_field_prompts": "Custom metadata",
    # Glossary Manager › Balanced/Full: the factor / limit beside their Auto box, the Single Pass header
    # prompt beside the extraction prompt
    "glossary_compression_factor": "Balanced/Full Extraction Settings",
    "glossary_max_output_tokens": "Balanced/Full Extraction Settings",
    "single_pass_glossary_header_prompt": "Balanced/Full Extraction Prompt",
}


def sub_heading(spec: Any) -> str:
    """Heading of a setting inside its section page: ``HEADING_OVERRIDES``, else the desktop group box
    (``spec.group``, e.g. "Foreign Character Detection", "Word Count Analysis"), else the nested path
    between the parent object and the leaf ("Thresholds", "Language detection › Languages"); "" when flat."""
    override = HEADING_OVERRIDES.get(str(spec_attr(spec, "key", "")))
    if override:
        return override
    group = str(spec_attr(spec, "group", "") or "").strip()
    if group:
        return group
    key = str(spec_attr(spec, "key", ""))
    parts = key.split(".")
    if spec_attr(spec, "parent", None) and len(parts) > 2:
        return " › ".join(humanize_key(part) for part in parts[1:-1])
    return ""


def grouped_specs(specs: Sequence[Any]) -> tuple[list, dict]:
    """``(specs, headings)``: the section's specs grouped by ``sub_heading`` (groups in the order they
    first appear, the ungrouped ones last under "Other settings" when the section has groups), and
    ``{key: heading}``. A section without any sub-heading keeps its order and has no headings."""
    headings = [sub_heading(spec) for spec in specs]
    if not any(headings):
        return list(specs), {}
    order: list = []
    for heading in headings:
        if heading and heading not in order:
            order.append(heading)
    buckets: dict = {heading: [] for heading in order}
    rest: list = []
    for spec, heading in zip(specs, headings):
        (buckets[heading] if heading else rest).append(spec)
    out: list = []
    labels: dict = {}
    for heading in order:
        for spec in buckets[heading]:
            out.append(spec)
            labels[str(spec_attr(spec, "key", ""))] = heading
    for spec in rest:
        out.append(spec)
        labels[str(spec_attr(spec, "key", ""))] = "Other settings"
    return out, labels
