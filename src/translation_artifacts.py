"""Shared handling for translated non-HTML EPUB workspace artifacts."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Callable, Dict, Iterable, List, Mapping, Tuple


TRANSLATION_ARTIFACT_PROGRESS_PREFIX = "__translation_artifact__"

TRANSLATION_ARTIFACT_SPECS: Tuple[Dict[str, Any], ...] = (
    {
        "kind": "toc",
        "filename": "TOC.txt",
        "label": "Table of Contents",
        "progress_key": f"{TRANSLATION_ARTIFACT_PROGRESS_PREFIX}:toc",
        "actual_num": -2,
        "toggle_attr": "use_toc_ncx_var",
        "toggle_config": "use_toc_ncx",
        "toggle_fallback_config": "translate_toc_ncx",
        "toggle_env": "USE_TOC_NCX",
        "default_enabled": True,
    },
    {
        "kind": "headers",
        "filename": "translated_headers.txt",
        "label": "Chapter Headers",
        "progress_key": f"{TRANSLATION_ARTIFACT_PROGRESS_PREFIX}:headers",
        "actual_num": -3,
        "toggle_attr": "batch_translate_headers_var",
        "toggle_config": "batch_translate_headers",
        "toggle_env": "BATCH_TRANSLATE_HEADERS",
        "default_enabled": True,
    },
)

QA_TRANSLATION_ARTIFACT_FILENAMES = (
    "metadata.json",
    "TOC.txt",
    "translated_headers.txt",
)


def translation_artifact_spec_for_filename(filename: Any) -> Dict[str, Any] | None:
    basename = os.path.basename(str(filename or "")).casefold()
    for spec in TRANSLATION_ARTIFACT_SPECS:
        if spec["filename"].casefold() == basename:
            return spec
    return None


def translation_artifact_spec_for_kind(kind: Any) -> Dict[str, Any] | None:
    normalized = str(kind or "").strip().casefold()
    for spec in TRANSLATION_ARTIFACT_SPECS:
        if spec["kind"].casefold() == normalized:
            return spec
    return None


def is_translation_artifact_progress_entry(key: Any, entry: Any = None) -> bool:
    if str(key or "").startswith(TRANSLATION_ARTIFACT_PROGRESS_PREFIX):
        return True
    if not isinstance(entry, Mapping):
        return False
    if entry.get("translation_artifact_progress_key"):
        return True
    return translation_artifact_spec_for_filename(entry.get("output_file")) is not None


def _iter_string_values(value: Any, path: Tuple[Any, ...] = ()) -> Iterable[Tuple[Tuple[Any, ...], str]]:
    if isinstance(value, str):
        if value.strip():
            yield path, value
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            yield from _iter_string_values(item, path + (index,))
        return
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _iter_string_values(item, path + (key,))


def metadata_translated_string_values(metadata: Any) -> List[Tuple[Tuple[Any, ...], str]]:
    """Return only translated metadata values, excluding stored source fields."""
    if not isinstance(metadata, Mapping):
        return []
    values: List[Tuple[Tuple[Any, ...], str]] = []
    for field, value in metadata.items():
        field_name = str(field)
        if field_name.startswith("original_") or field_name.endswith("_translated"):
            continue
        translated_flag = (
            "title_translated"
            if field_name == "title"
            else f"{field_name}_translated"
        )
        if metadata.get(translated_flag) is not True:
            continue
        values.extend(_iter_string_values(value, (field,)))
    return values


_TRANSLATED_CACHE_LINE_RE = re.compile(
    r"^(?P<prefix>\s*Translated:\s*)(?P<value>.*?)(?P<ending>\r\n|\n|\r)?$"
)
_CACHE_STATUS_LINE_RE = re.compile(
    r"^\s*Status:\s*(?P<value>.*?)(?P<ending>\r\n|\n|\r)?$",
    re.IGNORECASE,
)
_CACHE_BLOCK_BOUNDARY_RE = re.compile(r"^(?:\s*-{3,}\s*|\s*Chapter\s+\d+\s*:)")


def translated_cache_string_values(content: Any) -> List[str]:
    values = []
    for line in str(content or "").splitlines():
        match = _TRANSLATED_CACHE_LINE_RE.match(line)
        if match and match.group("value").strip():
            values.append(match.group("value").strip())
    return values


def translation_artifact_qa_text(filename: Any, content: Any) -> str:
    """Extract translated payload text while omitting intentional source text."""
    basename = os.path.basename(str(filename or "")).casefold()
    if basename == "metadata.json":
        try:
            metadata = json.loads(str(content or ""))
        except Exception:
            return ""
        return "\n".join(value for _path, value in metadata_translated_string_values(metadata))
    if basename in {"toc.txt", "translated_headers.txt"}:
        return "\n".join(translated_cache_string_values(content))
    return ""


def collect_translation_artifact_partial_targets(
    filename: Any,
    content: Any,
    has_foreign_characters: Callable[[str], bool],
):
    """Build editable targets containing only translated artifact values."""
    basename = os.path.basename(str(filename or "")).casefold()
    raw_content = str(content or "")

    if basename == "metadata.json":
        try:
            metadata = json.loads(raw_content)
        except Exception:
            return {"kind": "artifact_json", "data": {}, "trailing_newline": False}, []
        targets = [
            {"kind": "artifact_json_value", "path": path}
            for path, value in metadata_translated_string_values(metadata)
            if has_foreign_characters(value)
        ]
        return {
            "kind": "artifact_json",
            "data": metadata,
            "trailing_newline": raw_content.endswith(("\n", "\r")),
        }, targets

    if basename in {"toc.txt", "translated_headers.txt"}:
        lines = raw_content.splitlines(keepends=True)
        if not lines and raw_content:
            lines = [raw_content]
        targets = []
        for index, line in enumerate(lines):
            match = _TRANSLATED_CACHE_LINE_RE.match(line)
            if not match:
                continue
            value = match.group("value")
            if value.strip() and has_foreign_characters(value):
                targets.append({
                    "kind": "artifact_line_value",
                    "index": index,
                    "prefix": match.group("prefix"),
                    "ending": match.group("ending") or "",
                })
        return {"kind": "artifact_lines", "lines": lines}, targets

    return {"kind": "artifact_lines", "lines": [raw_content]}, []


def _get_path_value(data: Any, path: Iterable[Any]) -> Any:
    current = data
    for part in path:
        current = current[part]
    return current


def _set_path_value(data: Any, path: Iterable[Any], value: Any) -> None:
    parts = list(path)
    current = data
    for part in parts[:-1]:
        current = current[part]
    current[parts[-1]] = value


def translation_artifact_target_fragment(document: Mapping[str, Any], target: Mapping[str, Any]) -> str:
    kind = target.get("kind")
    if kind == "artifact_json_value":
        return str(_get_path_value(document.get("data", {}), target.get("path", ())))
    if kind == "artifact_line_value":
        line = document.get("lines", [])[int(target["index"])]
        match = _TRANSLATED_CACHE_LINE_RE.match(line)
        return match.group("value") if match else ""
    return ""


def apply_translation_artifact_response(
    document: Dict[str, Any],
    target: Mapping[str, Any],
    refined_value: Any,
) -> None:
    """Replace one translated value without modifying source/audit fields."""
    cleaned = str(refined_value or "").strip()
    if not cleaned:
        raise ValueError("Artifact refinement returned an empty translated value")

    kind = target.get("kind")
    if kind == "artifact_json_value":
        _set_path_value(document.get("data", {}), target.get("path", ()), cleaned)
        return
    if kind == "artifact_line_value":
        if "\n" in cleaned or "\r" in cleaned:
            raise ValueError(
                "Artifact refinement returned multiple lines for one cache value"
            )
        index = int(target["index"])
        lines = document.get("lines", [])
        lines[index] = (
            f"{target.get('prefix', '')}{cleaned}{target.get('ending', '')}"
        )

        # Failed cache entries are deliberately ignored by the normal cache
        # loader. Once Partial.b/b2 repairs their Translated value, remove the
        # failure marker (while retaining its newline so later target indices
        # remain stable) or the repaired value would still never be consumed.
        for status_index in range(index + 1, len(lines)):
            line = lines[status_index]
            if _CACHE_BLOCK_BOUNDARY_RE.match(line):
                break
            status_match = _CACHE_STATUS_LINE_RE.match(line)
            if not status_match:
                continue
            if re.search(
                r"\btranslation\s+failed\b",
                status_match.group("value"),
                flags=re.IGNORECASE,
            ):
                lines[status_index] = status_match.group("ending") or ""
            break
        return
    raise ValueError(f"Unsupported translation artifact target: {kind}")


def render_translation_artifact_document(document: Mapping[str, Any]) -> str:
    kind = document.get("kind")
    if kind == "artifact_json":
        rendered = json.dumps(document.get("data", {}), ensure_ascii=False, indent=2)
        if document.get("trailing_newline"):
            rendered += "\n"
        return rendered
    if kind == "artifact_lines":
        return "".join(document.get("lines", []))
    return ""
