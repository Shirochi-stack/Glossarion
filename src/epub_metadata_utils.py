"""Shared helpers for extracting repeatable EPUB metadata fields."""

from copy import deepcopy
from typing import Any, Dict, Set
import zipfile
from xml.etree import ElementTree

from epub_package import find_epub_opf_member


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


def _local_name(value: str) -> str:
    return str(value or "").rsplit("}", 1)[-1].split(":")[-1]


def extract_epub_metadata_file(source_path: str) -> Dict[str, Any]:
    """Read Dublin Core and custom OPF metadata directly from an EPUB."""
    with zipfile.ZipFile(source_path, "r") as archive:
        opf_name = find_epub_opf_member(archive)
        if not opf_name:
            raise ValueError("The EPUB does not contain an OPF package file")
        root = ElementTree.fromstring(archive.read(opf_name))

    metadata: Dict[str, Any] = {}
    dc_values: Dict[str, list[str]] = {
        field: [] for field in DC_ELEMENTS
    }
    meta_elements = []
    for element in root.iter():
        local_name = _local_name(element.tag).lower()
        if local_name in dc_values:
            text = "".join(element.itertext()).strip()
            if text:
                dc_values[local_name].append(text)
        if local_name == "meta":
            meta_elements.append(element)

    for field, values in dc_values.items():
        if not values:
            continue
        if field in REPEATABLE_DC_ELEMENTS and len(values) > 1:
            metadata[field] = values
        else:
            metadata[field] = values[0]

    for element in meta_elements:
        name = element.attrib.get("name") or element.attrib.get(
            "property", ""
        )
        content = element.attrib.get("content", "")
        if not name or not content:
            continue
        cleaned_name = str(name)
        for prefix in ("calibre:", "dc:", "opf:"):
            if cleaned_name.startswith(prefix):
                cleaned_name = cleaned_name[len(prefix):]
                break
        cleaned_name = cleaned_name.replace("-", "_")
        metadata.setdefault(cleaned_name, content)

    if "series" not in metadata:
        for element in meta_elements:
            name = element.attrib.get("name") or element.attrib.get(
                "property", ""
            )
            if "series" in str(name).lower():
                series_name = element.attrib.get("content", "")
                if series_name:
                    metadata["series"] = series_name
                    break

    for element in meta_elements:
        if "refines" not in element.attrib:
            continue
        property_name = element.attrib.get("property", "")
        content = (
            "".join(element.itertext()).strip()
            or element.attrib.get("content", "")
        )
        if not property_name or not content:
            continue
        property_name = _local_name(property_name).replace("-", "_")
        metadata.setdefault(property_name, content)

    return metadata


def merge_source_epub_metadata(
    workspace_metadata: Any,
    source_metadata: Any,
) -> tuple[Dict[str, Any], Set[str]]:
    """Restore source OPF fields without replacing translated workspace data.

    Extraction caches may leave ``metadata.json`` containing only structural
    chapter fields. The source OPF remains authoritative for missing source
    values, while an existing ``*_translated`` marker keeps its translated
    value and receives the source value as ``original_*`` instead.
    """
    metadata = (
        deepcopy(workspace_metadata)
        if isinstance(workspace_metadata, dict)
        else {}
    )
    if not isinstance(source_metadata, dict):
        return metadata, set()

    restored_fields: Set[str] = set()
    for field, value in source_metadata.items():
        translated_key = f"{field}_translated"
        original_key = f"original_{field}"
        if metadata.get(translated_key):
            if original_key not in metadata:
                metadata[original_key] = deepcopy(value)
                restored_fields.add(original_key)
            continue
        if field not in metadata:
            metadata[field] = deepcopy(value)
            restored_fields.add(field)

    restored_fields.update(
        restore_truncated_repeatable_metadata(metadata, source_metadata)
    )
    return metadata, restored_fields


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
