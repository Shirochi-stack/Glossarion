"""Pure helpers for mode-aware metadata translation progress tracking."""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Mapping
from urllib.parse import quote


METADATA_PROGRESS_KEY = "__metadata__"
METADATA_PROGRESS_PREFIX = "__metadata__"
VALID_METADATA_MODES = {"together", "metadata_separate", "parallel"}


def normalize_metadata_mode(mode: Any) -> str:
    value = str(mode or "together").strip().lower()
    return value if value in VALID_METADATA_MODES else "together"


def resolve_metadata_field_settings(
    settings: Any,
    source_path: str | None = None,
) -> Dict[str, bool]:
    """Resolve flat or per-EPUB field settings to a simple boolean mapping."""
    if not isinstance(settings, Mapping):
        return {"title": True}

    resolved: Mapping[str, Any] = settings
    per_epub = settings.get("_per_epub")
    basename = os.path.basename(str(source_path or ""))
    if basename and isinstance(per_epub, Mapping):
        match = per_epub.get(basename)
        if not isinstance(match, Mapping):
            basename_lower = basename.lower()
            match = next(
                (
                    value for key, value in per_epub.items()
                    if str(key).lower() == basename_lower and isinstance(value, Mapping)
                ),
                None,
            )
        if isinstance(match, Mapping):
            resolved = match

    fields = {
        str(field): bool(enabled)
        for field, enabled in resolved.items()
        if str(field) != "_per_epub"
    }
    fields.setdefault("title", True)
    return fields


def metadata_value_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True


def metadata_field_complete(metadata: Mapping[str, Any], field: str) -> bool:
    flag = "title_translated" if field == "title" else f"{field}_translated"
    return bool(metadata.get(flag, False))


def metadata_field_label(field: str) -> str:
    label = (
        str(field or "metadata")
        .replace("_", " ")
        .replace("-", " ")
        .replace(":", " ")
    )
    return " ".join(part.capitalize() for part in label.split()) or "Metadata"


def metadata_progress_key_for_field(field: str) -> str:
    return f"{METADATA_PROGRESS_PREFIX}:field:{quote(str(field), safe='')}"


def build_metadata_progress_plan(
    mode: Any,
    metadata: Any,
    field_settings: Any,
    *,
    title_allowed: bool = True,
    source_path: str | None = None,
) -> List[Dict[str, Any]]:
    """Return the progress rows matching the API-call shape for *mode*.

    Together mode has one row. Metadata-separate mode has a title row and one
    grouped row for the remaining fields. Parallel mode has one row per enabled
    field that is actually present in the source metadata.
    """
    if not isinstance(metadata, Mapping):
        metadata = {}
    fields = resolve_metadata_field_settings(field_settings, source_path)
    selected = [
        field for field in metadata
        if fields.get(str(field), False)
        and (str(field) != "title" or title_allowed)
        and metadata_value_present(metadata.get(field))
    ]
    selected = [str(field) for field in selected]
    if "title" in selected:
        selected.remove("title")
        selected.insert(0, "title")
    if not selected:
        return []

    normalized_mode = normalize_metadata_mode(mode)
    phases: List[Dict[str, Any]] = []

    def add(key: str, phase: str, phase_fields: Iterable[str], label: str) -> None:
        field_list = list(phase_fields)
        if not field_list:
            return
        phases.append({
            "key": key,
            "mode": normalized_mode,
            "phase": phase,
            "fields": field_list,
            "label": label,
            "index": len(phases),
        })

    if normalized_mode == "together":
        add(METADATA_PROGRESS_KEY, "combined", selected, "Metadata")
    elif normalized_mode == "metadata_separate":
        if "title" in selected:
            add(
                f"{METADATA_PROGRESS_PREFIX}:title",
                "title",
                ["title"],
                "Metadata: Book Title",
            )
        add(
            f"{METADATA_PROGRESS_PREFIX}:fields",
            "metadata",
            [field for field in selected if field != "title"],
            "Metadata: Other Fields",
        )
    else:
        for field in selected:
            add(
                metadata_progress_key_for_field(field),
                "field",
                [field],
                f"Metadata: {metadata_field_label(field)}",
            )
    return phases


def is_metadata_progress_entry(key: Any, entry: Any = None) -> bool:
    if str(key or "").startswith(METADATA_PROGRESS_PREFIX):
        return True
    return isinstance(entry, Mapping) and entry.get("special_type") == "metadata"
