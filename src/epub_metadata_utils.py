"""Shared helpers for extracting repeatable EPUB metadata fields."""

from copy import deepcopy
from typing import Any, Dict, Set


DC_ELEMENTS = (
    "title",
    "creator",
    "subject",
    "description",
    "publisher",
    "contributor",
    "date",
    "type",
    "format",
    "identifier",
    "source",
    "language",
    "relation",
    "coverage",
    "rights",
)

# These fields may occur more than once in an OPF package. Subject is the
# repeatable field currently supported end-to-end by the EPUB compiler.
REPEATABLE_DC_ELEMENTS = frozenset({"subject"})


def extract_dc_metadata(soup) -> Dict[str, Any]:
    """Extract Dublin Core metadata while preserving repeatable fields.

    A single subject remains a string for compatibility with existing
    metadata.json files. Multiple subjects are returned as an ordered list.
    """
    metadata: Dict[str, Any] = {}

    for element in DC_ELEMENTS:
        tags = soup.find_all(element)
        values = [tag.get_text(strip=True) for tag in tags]
        values = [value for value in values if value]
        if not values:
            continue

        if element in REPEATABLE_DC_ELEMENTS and len(values) > 1:
            metadata[element] = values
        else:
            metadata[element] = values[0]

    return metadata


def _as_nonempty_list(value: Any) -> list:
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    if value is None:
        return []
    text = str(value).strip()
    return [text] if text else []


def restore_truncated_repeatable_metadata(
    existing: Dict[str, Any], source: Dict[str, Any]
) -> Set[str]:
    """Restore repeatable values omitted by older single-tag extraction.

    Only fields for which the source contains more values than the cached
    source value are repaired. If a repaired field had been translated, its
    stale translation marker is removed so the complete list is translated on
    the next metadata pass.
    """
    restored: Set[str] = set()

    for field in REPEATABLE_DC_ELEMENTS:
        source_values = _as_nonempty_list(source.get(field))
        if len(source_values) <= 1:
            continue

        translated_key = f"{field}_translated"
        original_key = f"original_{field}"
        cached_source = (
            existing.get(original_key)
            if existing.get(translated_key)
            else existing.get(field)
        )
        if len(_as_nonempty_list(cached_source)) >= len(source_values):
            continue

        source_value = deepcopy(source[field])
        existing[field] = source_value
        if existing.get(translated_key) or original_key in existing:
            existing[original_key] = deepcopy(source_value)
        existing.pop(translated_key, None)
        restored.add(field)

    return restored
