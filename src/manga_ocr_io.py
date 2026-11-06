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
    "region_styles",
)
#: Per-box text style (the manual editor's Format toolbar), stored in a page's editor state as
#: ``region_styles`` = {"<region index>": style} and carried by an exported region as ``style``.
REGION_STYLE_KEYS = ("font", "font_size", "bold", "italic", "underline", "align", "color")
REGION_STYLE_MAX_FONT_SIZE = 512
#: The font size each box was last drawn with (auto fit or its own size), stored in a page's editor
#: state as ``region_font_sizes`` = {"<region index>": px} by every render, and carried by a
#: rendered region (``TextRegion.rendered_font_size``) as ``rendered_font_size``.
REGION_FONT_SIZES_KEY = "region_font_sizes"


class MangaOcrFormatError(ValueError):
    """Raised when a selected file is not a supported manga OCR export."""


def normalize_font_size(value: Any) -> Optional[Any]:
    """A px font size in steps of 0.5 (28, 28.5), capped; None when it is not a positive number.

    Whole sizes come back as ``int`` (as before fractional sizes), the others as ``float``."""
    if value in (None, "") or isinstance(value, bool):
        return None
    try:
        size = round(float(value) * 2) / 2
    except (TypeError, ValueError, OverflowError):
        return None
    if not size > 0:
        return None
    size = min(size, REGION_STYLE_MAX_FONT_SIZE)
    return int(size) if float(size).is_integer() else size


def normalize_region_style(style: Any) -> Dict[str, Any]:
    """A box's text style with only valid, non-default entries ({} = render with Rendering Settings).

    ``font``: a font file path; ``font_size``: px in steps of 0.5 (absent = Auto fit); ``bold`` /
    ``italic`` / ``underline``: True; ``align``: "left" or "right" (absent = centered); ``color``:
    [r, g, b].
    """
    if not isinstance(style, Mapping):
        return {}
    out: Dict[str, Any] = {}
    font = style.get("font")
    if isinstance(font, str) and font.strip():
        out["font"] = font.strip()
    size = normalize_font_size(style.get("font_size"))
    if size is not None:
        out["font_size"] = size
    for key in ("bold", "italic", "underline"):
        if style.get(key) is True:
            out[key] = True
    align = str(style.get("align") or "").strip().lower()
    if align in ("left", "right"):
        out["align"] = align
    color = style.get("color")
    if isinstance(color, (list, tuple)) and len(color) >= 3:
        try:
            rgb = [int(channel) for channel in color[:3]]
        except (TypeError, ValueError):
            rgb = None
        if rgb is not None and all(0 <= channel <= 255 for channel in rgb):
            out["color"] = rgb
    return out


def region_styles_from_state(state: Any) -> Dict[int, Dict[str, Any]]:
    """The page's per-box styles by region index (empty / invalid entries dropped)."""
    styles: Dict[int, Dict[str, Any]] = {}
    raw = state.get("region_styles") if isinstance(state, Mapping) else None
    if not isinstance(raw, Mapping):
        return styles
    for key, value in raw.items():
        try:
            index = int(key)
        except (TypeError, ValueError):
            continue
        style = normalize_region_style(value)
        if style and index >= 0:
            styles[index] = style
    return styles


def region_font_sizes_from_state(state: Any) -> Dict[int, Any]:
    """The size each box of the page was last rendered at, by region index (``region_font_sizes``)."""
    sizes: Dict[int, Any] = {}
    raw = state.get(REGION_FONT_SIZES_KEY) if isinstance(state, Mapping) else None
    if not isinstance(raw, Mapping):
        return sizes
    for key, value in raw.items():
        try:
            index = int(key)
        except (TypeError, ValueError):
            continue
        size = normalize_rendered_font_size(value)
        if size is not None and index >= 0:
            sizes[index] = size
    return sizes


def normalize_rendered_font_size(value: Any) -> Optional[Any]:
    """A rendered font size as stored (px, one decimal; whole sizes as ``int``); None if invalid."""
    if value in (None, "") or isinstance(value, bool):
        return None
    try:
        size = round(float(value), 1)
    except (TypeError, ValueError, OverflowError):
        return None
    if not 0 < size < float("inf"):
        return None
    return int(size) if size.is_integer() else size


def set_region_style(state: Dict[str, Any], index: int, style: Any) -> Dict[str, Any]:
    """Store box ``index``'s style in ``state`` (an empty style removes it); returns the stored style."""
    styles = {str(key): value for key, value in region_styles_from_state(state).items()}
    normalized = normalize_region_style(style)
    if normalized:
        styles[str(int(index))] = normalized
    else:
        styles.pop(str(int(index)), None)
    if styles:
        state["region_styles"] = styles
    else:
        state.pop("region_styles", None)
    return normalized


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

    styles = region_styles_from_state(state)

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
        if styles.get(index):
            base["style"] = styles[index]
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
    region_styles: Dict[str, Any] = {}
    region_font_sizes: Dict[str, Any] = {}
    for index, raw_region in enumerate(page.get("regions") or []):
        region = serialize_region(raw_region)
        style = normalize_region_style(region.get("style"))
        if style:
            region_styles[str(index)] = style
        rendered_size = normalize_rendered_font_size(region.get("rendered_font_size"))
        if rendered_size is not None:
            region_font_sizes[str(index)] = rendered_size
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
    if region_styles:
        state["region_styles"] = region_styles
    if region_font_sizes:
        state[REGION_FONT_SIZES_KEY] = region_font_sizes

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
    style = normalize_region_style(normalized.get("style"))
    if style:
        region.style = style
    return region


# ---------------------------------------------------------------------------
# Import OCR "merge": the imported regions join the page's own (detected / existing) regions
# ---------------------------------------------------------------------------

#: An existing region is replaced by an imported one when their boxes overlap by at least this
#: share of the smaller box's area (the rule that merged the webtoon's SFX-only OCR files into
#: the detected dialogue: every replaced box was the detector finding the same sound effect).
MERGE_OVERLAP_OF_SMALLER = 0.5


def region_box(region: Any) -> List[float]:
    """``[x, y, w, h]`` of a region record (``bounding_box`` / ``bbox``) or a TextRegion."""
    if isinstance(region, Mapping):
        box = region.get("bounding_box") or region.get("bbox")
    else:
        box = getattr(region, "bounding_box", None)
    try:
        values = [float(value) for value in list(box or [])[:4]]
    except (TypeError, ValueError):
        values = []
    return values if len(values) == 4 else [0.0, 0.0, 0.0, 0.0]


def overlap_of_smaller(a: Iterable[float], b: Iterable[float]) -> float:
    """Intersection area of two ``[x, y, w, h]`` boxes over the area of the smaller one."""
    ax, ay, aw, ah = list(a)[:4]
    bx, by, bw, bh = list(b)[:4]
    width = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    height = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    smaller = max(1.0, min(aw * ah, bw * bh))
    return (width * height) / smaller


def merge_regions(
    existing: Iterable[Any],
    imported: Iterable[Any],
    threshold: float = MERGE_OVERLAP_OF_SMALLER,
):
    """Merge imported regions into a page's own regions: ``(merged, replaced)``.

    Every imported region is kept. An existing region that overlaps an imported one by
    ``threshold`` of the smaller box is replaced by it (the imported text is the correction) and
    listed in ``replaced``; the other existing regions stay. The merged list keeps the existing
    reading order: an imported region takes the place of the first region it replaces, one that
    replaces nothing goes before the first kept existing region below its top edge (imported
    regions keep their own order otherwise). Works on region records and TextRegion objects."""
    existing = list(existing or [])
    imported = list(imported or [])
    imported_boxes = [region_box(region) for region in imported]
    replaced: List[Any] = []
    merged: List[Any] = []
    placed = set()
    for region in existing:
        box = region_box(region)
        hits = [index for index, other in enumerate(imported_boxes)
                if overlap_of_smaller(box, other) >= threshold]
        if not hits:
            merged.append(region)
            continue
        replaced.append(region)
        for index in hits:
            if index not in placed:
                placed.add(index)
                merged.append(imported[index])
    kept = {id(region) for region in existing} - {id(region) for region in replaced}
    for index, region in enumerate(imported):
        if index in placed:
            continue
        top = imported_boxes[index][1]
        position = next((i for i, other in enumerate(merged)
                         if id(other) in kept and region_box(other)[1] > top), len(merged))
        merged.insert(position, region)
    return merged, replaced


def merged_import_page(page: Mapping[str, Any], existing_state: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """The imported *page* merged with a page's editor state (Import OCR with merge on).

    Its regions are the state's regions (``canonical_regions_from_editor_state``) merged with the
    page's (``merge_regions``), numbered again in their new order; the imported page's own
    ``editor_state`` is dropped because its indices no longer match. Without existing regions the
    page comes back as it is."""
    existing = canonical_regions_from_editor_state(existing_state) if existing_state else []
    if not existing:
        return dict(page)
    imported = [dict(region) for region in (page.get("regions") or []) if isinstance(region, Mapping)]
    merged, _replaced = merge_regions(existing, imported)
    out = {key: value for key, value in page.items() if key != "editor_state"}
    out["regions"] = [dict(serialize_region(region), rect_index=index) for index, region in enumerate(merged)]
    return out


def editor_state_for_import(
    page: Mapping[str, Any],
    existing_state: Optional[Mapping[str, Any]] = None,
    *,
    merge: bool = False,
) -> Dict[str, Any]:
    """The editor state an Import OCR writes for a matched page: the imported page's own state
    (replace, the default) or, with ``merge``, the page's existing regions merged with it.

    A merge numbers the boxes again, so every per-box key of the existing state (its rendered
    overlays and positions, translations, styles, font sizes) is written too: one the merged
    page has no value for comes back empty instead of leaving the old indices on other boxes."""
    if not merge:
        return editor_state_from_page(page)
    renumbered = bool(existing_state) and bool(canonical_regions_from_editor_state(existing_state))
    state = editor_state_from_page(merged_import_page(page, existing_state))
    if renumbered:
        for key in EDITOR_STATE_KEYS + (REGION_FONT_SIZES_KEY,):
            if key not in state and key in existing_state:
                state[key] = {} if isinstance(existing_state.get(key), Mapping) else []
    return state
