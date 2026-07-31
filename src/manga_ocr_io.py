"""Portable import/export helpers for manga OCR and editor mappings."""

from __future__ import annotations

import copy
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional


FORMAT_ID = "glossarion-manga-ocr"
FORMAT_VERSION = 1
EDITOR_STATE_KEYS = (
    "detection_regions",
    "viewer_rectangles",
    "recognized_texts",
    "translated_texts",
    "overlay_rects",
    "last_render_positions",
)


class MangaOcrFormatError(ValueError):
    """Raised when a selected file is not a supported manga OCR export."""


def _json_safe(value: Any) -> Any:
    """Return a detached, JSON-compatible copy of a value."""
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _normalized_path(path: str) -> str:
    return os.path.normcase(os.path.abspath(os.path.expanduser(str(path or ""))))


def _relative_path(image_path: str, source_root: Optional[str]) -> str:
    if not source_root:
        return os.path.basename(image_path)
    try:
        relative = os.path.relpath(image_path, source_root)
        if relative != ".." and not relative.startswith(".." + os.sep):
            return relative.replace("\\", "/")
    except (OSError, ValueError):
        pass
    return os.path.basename(image_path)


def serialize_region(region: Any) -> Dict[str, Any]:
    """Serialize either a TextRegion object or a region dictionary."""
    if isinstance(region, Mapping):
        record = dict(region)
    elif hasattr(region, "to_dict"):
        record = dict(region.to_dict())
    else:
        record = {
            key: getattr(region, key)
            for key in (
                "text",
                "vertices",
                "bounding_box",
                "confidence",
                "region_type",
                "translated_text",
            )
            if hasattr(region, key)
        }

    # Renderer records use bbox/coords while MangaTranslator uses
    # bounding_box/vertices. Keep both spellings for round-trip compatibility.
    if "bounding_box" not in record and record.get("bbox") is not None:
        record["bounding_box"] = record.get("bbox")
    if "bbox" not in record and record.get("bounding_box") is not None:
        record["bbox"] = record.get("bounding_box")
    if "vertices" not in record and record.get("coords") is not None:
        record["vertices"] = record.get("coords")
    if "coords" not in record and record.get("vertices") is not None:
        record["coords"] = record.get("vertices")

    for key in (
        "bubble_bounds",
        "bubble_type",
        "should_inpaint",
        "shape",
        "polygon",
        "rect_index",
    ):
        if key not in record and hasattr(region, key):
            value = getattr(region, key)
            if value is not None:
                record[key] = value

    record.setdefault("text", "")
    record.setdefault("confidence", 1.0)
    record.setdefault("region_type", record.get("bubble_type") or "text_block")
    record["translated_text"] = record.get("translated_text")
    return _json_safe(record)


def make_page(
    image_path: str,
    regions: Iterable[Any],
    *,
    index: int = 0,
    source_root: Optional[str] = None,
    editor_state: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    absolute_path = os.path.abspath(image_path)
    page: Dict[str, Any] = {
        "index": int(index),
        "source_path": absolute_path.replace("\\", "/"),
        "relative_path": _relative_path(absolute_path, source_root),
        "file_name": os.path.basename(absolute_path),
        "regions": [serialize_region(region) for region in (regions or [])],
    }
    try:
        page["file_size"] = os.path.getsize(absolute_path)
    except OSError:
        pass
    if editor_state is not None:
        page["editor_state"] = {
            key: _json_safe(editor_state[key])
            for key in EDITOR_STATE_KEYS
            if key in editor_state
        }
    return page


def create_document(
    pages: Iterable[Mapping[str, Any]],
    *,
    workflow: str,
    source_root: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "format": FORMAT_ID,
        "version": FORMAT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "workflow": str(workflow or "manga"),
        "source_root": (
            os.path.abspath(source_root).replace("\\", "/") if source_root else None
        ),
        "pages": [_json_safe(dict(page)) for page in pages],
    }


def validate_document(document: Any) -> Dict[str, Any]:
    if not isinstance(document, dict):
        raise MangaOcrFormatError("The OCR file must contain a JSON object.")
    if document.get("format") != FORMAT_ID:
        raise MangaOcrFormatError("This is not a Glossarion manga OCR file.")
    version = document.get("version")
    if version != FORMAT_VERSION:
        raise MangaOcrFormatError(
            f"Unsupported manga OCR version {version!r}; expected {FORMAT_VERSION}."
        )
    pages = document.get("pages")
    if not isinstance(pages, list):
        raise MangaOcrFormatError("The OCR file has no valid pages list.")
    for page_index, page in enumerate(pages, start=1):
        if not isinstance(page, dict):
            raise MangaOcrFormatError(f"Page {page_index} is not a JSON object.")
        if not isinstance(page.get("regions", []), list):
            raise MangaOcrFormatError(f"Page {page_index} has invalid region data.")
    return document


def load_document(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8-sig") as handle:
            document = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise MangaOcrFormatError(f"Could not read the OCR file: {exc}") from exc
    return validate_document(document)


def write_document(path: str, document: Mapping[str, Any]) -> str:
    """Validate and atomically write an OCR document."""
    detached = validate_document(_json_safe(dict(document)))
    absolute_path = os.path.abspath(path)
    parent = os.path.dirname(absolute_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    temp_path = absolute_path + ".tmp"
    with open(temp_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(detached, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temp_path, absolute_path)
    return absolute_path


def match_document_pages(
    document: Mapping[str, Any], image_paths: Iterable[str]
) -> Dict[str, Dict[str, Any]]:
    """Match exported pages to current images without relying on one machine's paths."""
    validate_document(dict(document))
    images = [os.path.abspath(path) for path in image_paths]
    pages = list(document.get("pages") or [])
    matched: Dict[str, Dict[str, Any]] = {}
    used_page_ids = set()

    exact = {}
    for page in pages:
        source_path = page.get("source_path")
        if source_path:
            exact.setdefault(_normalized_path(source_path), []).append(page)
    for image_path in images:
        candidates = exact.get(_normalized_path(image_path), [])
        if len(candidates) == 1:
            matched[image_path] = candidates[0]
            used_page_ids.add(id(candidates[0]))

    # Relative paths preserve duplicate filenames in nested manga folders.
    current_root = None
    if images:
        try:
            current_root = os.path.commonpath([os.path.dirname(path) for path in images])
        except ValueError:
            current_root = None
    relative_pages: Dict[str, List[Dict[str, Any]]] = {}
    for page in pages:
        if id(page) in used_page_ids:
            continue
        relative = str(page.get("relative_path") or "").replace("\\", "/").casefold()
        if relative:
            relative_pages.setdefault(relative, []).append(page)
    for image_path in images:
        if image_path in matched:
            continue
        relative = _relative_path(image_path, current_root).replace("\\", "/").casefold()
        candidates = relative_pages.get(relative, [])
        if len(candidates) == 1 and id(candidates[0]) not in used_page_ids:
            matched[image_path] = candidates[0]
            used_page_ids.add(id(candidates[0]))

    # A unique basename is a safe fallback when the manga folder was moved.
    by_name: Dict[str, List[Dict[str, Any]]] = {}
    for page in pages:
        if id(page) in used_page_ids:
            continue
        name = str(page.get("file_name") or os.path.basename(page.get("source_path") or ""))
        by_name.setdefault(name.casefold(), []).append(page)
    image_name_counts: Dict[str, int] = {}
    for image_path in images:
        key = os.path.basename(image_path).casefold()
        image_name_counts[key] = image_name_counts.get(key, 0) + 1
    for image_path in images:
        if image_path in matched:
            continue
        key = os.path.basename(image_path).casefold()
        candidates = by_name.get(key, [])
        if image_name_counts.get(key) == 1 and len(candidates) == 1:
            matched[image_path] = candidates[0]
            used_page_ids.add(id(candidates[0]))

    return matched


def canonical_regions_from_editor_state(state: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Merge editor rectangles, OCR text, and translations into pipeline regions."""
    # Manual rectangle positions are the newest mapping and intentionally take
    # precedence over the original detector boxes.
    bases = list(state.get("viewer_rectangles") or [])
    if not bases:
        bases = list(state.get("detection_regions") or [])

    recognized_by_index: Dict[int, Any] = {}
    for fallback_index, entry in enumerate(state.get("recognized_texts") or []):
        index = entry.get("region_index", fallback_index) if isinstance(entry, dict) else fallback_index
        try:
            recognized_by_index[int(index)] = entry
        except (TypeError, ValueError):
            continue

    translated_by_index: Dict[int, Any] = {}
    for fallback_index, entry in enumerate(state.get("translated_texts") or []):
        index = fallback_index
        if isinstance(entry, dict):
            index = (entry.get("original") or {}).get("region_index", fallback_index)
        try:
            translated_by_index[int(index)] = entry
        except (TypeError, ValueError):
            continue

    count = max(
        len(bases),
        (max(recognized_by_index) + 1) if recognized_by_index else 0,
        (max(translated_by_index) + 1) if translated_by_index else 0,
    )
    regions: List[Dict[str, Any]] = []
    for index in range(count):
        base = dict(bases[index]) if index < len(bases) and isinstance(bases[index], dict) else {}
        recognized = recognized_by_index.get(index)
        translated = translated_by_index.get(index)

        if "bbox" not in base and all(key in base for key in ("x", "y", "width", "height")):
            base["bbox"] = [base["x"], base["y"], base["width"], base["height"]]
        if isinstance(recognized, dict):
            base.setdefault("bbox", recognized.get("bbox"))
            base["text"] = recognized.get("text", "")
            for key in ("confidence", "bubble_type", "region_type", "bubble_bounds"):
                if recognized.get(key) is not None:
                    base[key] = recognized.get(key)
        elif isinstance(recognized, str):
            base["text"] = recognized
        else:
            base.setdefault("text", "")

        if isinstance(translated, dict):
            base["translated_text"] = translated.get("translation")
        base.setdefault("rect_index", index)
        regions.append(serialize_region(base))
    return regions


def editor_state_from_page(page: Mapping[str, Any]) -> Dict[str, Any]:
    """Return saved editor state, or synthesize it from pipeline region records."""
    saved_state: Dict[str, Any] = {}
    if isinstance(page.get("editor_state"), dict):
        saved_state = {
            key: copy.deepcopy(page["editor_state"][key])
            for key in EDITOR_STATE_KEYS
            if key in page["editor_state"]
        }

    detection_regions = []
    viewer_rectangles = []
    recognized_texts = []
    translated_texts = []
    for index, raw_region in enumerate(page.get("regions") or []):
        region = serialize_region(raw_region)
        bbox = list(region.get("bounding_box") or region.get("bbox") or [0, 0, 1, 1])
        if len(bbox) < 4:
            bbox = [0, 0, 1, 1]
        x, y, width, height = bbox[:4]
        detection = {
            "bbox": bbox[:4],
            "coords": region.get("vertices") or region.get("coords") or [
                [x, y], [x + width, y], [x + width, y + height], [x, y + height]
            ],
            "confidence": region.get("confidence", 1.0),
            "shape": region.get("shape", "rect"),
            "bubble_type": region.get("bubble_type"),
            "region_type": region.get("region_type"),
            "bubble_bounds": region.get("bubble_bounds", bbox[:4]),
        }
        detection_regions.append(detection)
        viewer_rectangles.append({
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "shape": region.get("shape", "rect"),
            **{
                key: region[key]
                for key in ("bubble_type", "region_type", "bubble_bounds", "polygon")
                if region.get(key) is not None
            },
        })
        if str(region.get("text") or "").strip():
            recognized_texts.append({
                "region_index": index,
                "bbox": bbox[:4],
                "text": region.get("text", ""),
                "confidence": region.get("confidence", 1.0),
                "bubble_type": region.get("bubble_type"),
                "region_type": region.get("region_type"),
                "bubble_bounds": region.get("bubble_bounds", bbox[:4]),
            })
        if str(region.get("translated_text") or "").strip():
            translated_texts.append({
                "original": {
                    "region_index": index,
                    "text": region.get("text", ""),
                },
                "translation": region.get("translated_text", ""),
                "bbox": bbox[:4],
            })

    state: Dict[str, Any] = {
        "detection_regions": detection_regions,
        "viewer_rectangles": viewer_rectangles,
        "recognized_texts": recognized_texts,
    }
    if translated_texts:
        state["translated_texts"] = translated_texts

    if saved_state:
        # Keep the editor's richer geometry/overlay metadata, while treating the
        # canonical region records as the source of truth for text. Older/manual
        # snapshots may contain OCR only even though regions include translations.
        merged_state = {**state, **saved_state}
        if recognized_texts:
            merged_state["recognized_texts"] = recognized_texts
        if translated_texts:
            merged_state["translated_texts"] = translated_texts
        return merged_state
    return state


def region_record_to_text_region(record: Mapping[str, Any], text_region_class: Any) -> Any:
    """Create a MangaTranslator TextRegion while restoring optional detector metadata."""
    normalized = serialize_region(record)
    bbox = normalized.get("bounding_box") or normalized.get("bbox") or [0, 0, 1, 1]
    vertices = normalized.get("vertices") or normalized.get("coords")
    if not vertices:
        x, y, width, height = bbox[:4]
        vertices = [[x, y], [x + width, y], [x + width, y + height], [x, y + height]]
    region = text_region_class(
        text=str(normalized.get("text") or ""),
        vertices=[tuple(vertex[:2]) for vertex in vertices],
        bounding_box=tuple(bbox[:4]),
        confidence=float(normalized.get("confidence", 1.0) or 1.0),
        region_type=str(normalized.get("region_type") or "text_block"),
        translated_text=normalized.get("translated_text"),
        bubble_bounds=(
            tuple(normalized["bubble_bounds"][:4])
            if normalized.get("bubble_bounds") is not None
            else None
        ),
    )
    for key in ("bubble_type", "should_inpaint", "shape", "polygon", "rect_index"):
        if normalized.get(key) is not None:
            setattr(region, key, copy.deepcopy(normalized[key]))
    return region
