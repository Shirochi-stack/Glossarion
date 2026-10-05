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
from typing import Any, Iterable, Optional, Sequence

__all__ = [
    "GROUP_ORDER",
    "GROUP_TITLES",
    "SECTION_GROUPS",
    "TILE_KINDS",
    "WINDOW_SIZE",
    "choice_options",
    "config_path",
    "env_names",
    "group_title",
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

# Tiles rendered at once in a SectionPage; larger sections slide this window.
WINDOW_SIZE = 60

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

_SECRET_TYPES = ("secret", "password")
_PROMPT_TYPES = ("prompt", "text", "multiline", "textarea")
_LIST_TYPES = ("list", "tuple", "array")
_JSON_TYPES = ("dict", "json", "object", "map", "mapping")
_SECRET_KEY_HINTS = ("api_key", "apikey", "_secret", "password", "_token", "cookie")
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
    if isinstance(default, dict) and len(default) == 1 and str(next(iter(default))).startswith("$"):
        return None
    return default


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
    if kind in _SECRET_TYPES:
        return True
    if kind not in ("str", "string"):
        return False
    key = str(spec_attr(spec, "key", "")).lower()
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
