"""SDLXLIFF extraction helpers.

This module treats SDLXLIFF as a round-trippable XML container. It extracts
eligible source segments into JSON batch chapters while protecting inline XML
structure with placeholders that the converter can restore.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from lxml import etree


XLIFF_NS = "urn:oasis:names:tc:xliff:document:1.2"
SDL_NS = "http://sdl.com/FileTypes/SdlXliff/1.0"
PLACEHOLDER_RE = re.compile(r"\[\[XLIFF_TAG_\d{6}_\d{4}\]\]")
CONFIRMED_STATES = {
    "approvedsignoff",
    "approvedtranslation",
    "signedoff",
    "translated",
    "translationapproved",
    "confirmed",
}


def _local_name(tag: Any) -> str:
    if not isinstance(tag, str):
        return ""
    return etree.QName(tag).localname if tag.startswith("{") else tag


def _ns_uri(tag: Any) -> str:
    if not isinstance(tag, str) or not tag.startswith("{"):
        return ""
    return etree.QName(tag).namespace or ""


def _qname_like(parent: etree._Element, local: str) -> str:
    ns = _ns_uri(parent.tag)
    return f"{{{ns}}}{local}" if ns else local


def _iter_children_named(parent: etree._Element, local: str) -> Iterable[etree._Element]:
    for child in parent:
        if _local_name(child.tag) == local:
            yield child


def _first_child_named(parent: etree._Element, local: str) -> Optional[etree._Element]:
    return next(_iter_children_named(parent, local), None)


def _iter_descendants_named(parent: etree._Element, local: str) -> Iterable[etree._Element]:
    for elem in parent.iter():
        if elem is not parent and _local_name(elem.tag) == local:
            yield elem


def _element_visible_text(elem: Optional[etree._Element]) -> str:
    if elem is None:
        return ""
    return "".join(elem.itertext()).strip()


def _segment_conf_state(trans_unit: etree._Element, mid: Optional[str]) -> str:
    for seg in trans_unit.iter():
        if _local_name(seg.tag) != "seg" or _ns_uri(seg.tag) != SDL_NS:
            continue
        seg_id = seg.get("id")
        if mid is not None and seg_id is not None and str(seg_id) != str(mid):
            continue
        conf = (seg.get("conf") or seg.get("confirmationLevel") or "").strip()
        if conf:
            return conf
    return ""


def _segment_locked(trans_unit: etree._Element, mid: Optional[str]) -> bool:
    truthy = {"1", "true", "yes", "on"}
    if str(trans_unit.get("locked", "")).strip().lower() in truthy:
        return True
    for seg in trans_unit.iter():
        if _local_name(seg.tag) != "seg" or _ns_uri(seg.tag) != SDL_NS:
            continue
        seg_id = seg.get("id")
        if mid is not None and seg_id is not None and str(seg_id) != str(mid):
            continue
        if str(seg.get("locked", "")).strip().lower() in truthy:
            return True
    return False


def _is_confirmed_state(conf: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "", str(conf or "").lower())
    return normalized in CONFIRMED_STATES or "approved" in normalized


def _translate_disabled(elem: etree._Element) -> bool:
    current = elem
    while current is not None:
        if str(current.get("translate", "")).strip().lower() == "no":
            return True
        current = current.getparent()
    return False


def _find_target_segment(trans_unit: etree._Element, mid: Optional[str]) -> Optional[etree._Element]:
    target = _first_child_named(trans_unit, "target")
    if target is None:
        return None
    if mid is None:
        mrks = [m for m in _iter_descendants_named(target, "mrk") if m.get("mtype") == "seg"]
        return mrks[0] if len(mrks) == 1 else target
    for mrk in _iter_descendants_named(target, "mrk"):
        if mrk.get("mtype") == "seg" and str(mrk.get("mid", "")) == str(mid):
            return mrk
    return None


def _source_segments(trans_unit: etree._Element) -> List[Tuple[Optional[str], etree._Element, str]]:
    seg_source = _first_child_named(trans_unit, "seg-source")
    segments: List[Tuple[Optional[str], etree._Element, str]] = []
    if seg_source is not None:
        for idx, mrk in enumerate(_iter_descendants_named(seg_source, "mrk")):
            if mrk.get("mtype") == "seg":
                segments.append((mrk.get("mid") or str(idx + 1), mrk, "seg-source"))
    if segments:
        return segments

    source = _first_child_named(trans_unit, "source")
    if source is not None:
        segments.append((None, source, "source"))
    return segments


def _nsmap_entries(elem: etree._Element) -> List[Dict[str, str]]:
    entries = []
    for prefix, uri in (elem.nsmap or {}).items():
        if uri:
            entries.append({"prefix": "" if prefix is None else str(prefix), "uri": str(uri)})
    return entries


def _element_template(elem: etree._Element, kind: str) -> Dict[str, Any]:
    return {
        "kind": kind,
        "tag": elem.tag,
        "attrib": dict(elem.attrib),
        "nsmap": _nsmap_entries(elem),
    }


def _protect_inline_xml(segment_elem: etree._Element, segment_num: int) -> Tuple[str, Dict[str, Dict[str, Any]], List[str]]:
    parts: List[str] = []
    tag_map: Dict[str, Dict[str, Any]] = {}
    tag_sequence: List[str] = []
    counter = 0

    def next_placeholder() -> str:
        nonlocal counter
        placeholder = f"[[XLIFF_TAG_{segment_num:06d}_{counter:04d}]]"
        counter += 1
        tag_sequence.append(placeholder)
        return placeholder

    def walk(elem: etree._Element) -> None:
        nonlocal counter
        if elem.text:
            parts.append(elem.text)
        for child in elem:
            has_content = bool(child.text) or len(child) > 0
            if has_content:
                start_placeholder = next_placeholder()
                tag_map[start_placeholder] = _element_template(child, "start")
                parts.append(start_placeholder)
                walk(child)
                end_placeholder = next_placeholder()
                tag_map[end_placeholder] = {"kind": "end"}
                parts.append(end_placeholder)
            else:
                placeholder = next_placeholder()
                tag_map[placeholder] = _element_template(child, "empty")
                parts.append(placeholder)
            if child.tail:
                parts.append(child.tail)

    walk(segment_elem)
    return "".join(parts), tag_map, tag_sequence


def is_placeholder_only_text(text: str) -> bool:
    text = str(text or "").strip()
    if not text or not PLACEHOLDER_RE.search(text):
        return False
    return not PLACEHOLDER_RE.sub("", text).strip()


def parse_sdlxliff(path: str) -> etree._ElementTree:
    parser = etree.XMLParser(remove_blank_text=False, recover=False, huge_tree=True)
    return etree.parse(path, parser)


def iter_eligible_segments(path: str) -> List[Dict[str, Any]]:
    tree = parse_sdlxliff(path)
    root = tree.getroot()
    eligible: List[Dict[str, Any]] = []
    segment_num = 0

    for unit_index, trans_unit in enumerate(
        elem for elem in root.iter() if _local_name(elem.tag) == "trans-unit"
    ):
        if _translate_disabled(trans_unit):
            continue
        unit_id = trans_unit.get("id") or str(unit_index + 1)
        for mid, src_elem, source_kind in _source_segments(trans_unit):
            if _segment_locked(trans_unit, mid):
                continue
            target_elem = _find_target_segment(trans_unit, mid)
            target_text = _element_visible_text(target_elem)
            conf = _segment_conf_state(trans_unit, mid)
            if target_text and _is_confirmed_state(conf):
                continue
            segment_num += 1
            source_text, tag_map, tag_sequence = _protect_inline_xml(src_elem, segment_num)
            if not source_text.strip():
                continue
            segment_id = str(segment_num)
            eligible.append(
                {
                    "id": segment_id,
                    "segment_num": segment_num,
                    "unit_index": unit_index,
                    "unit_id": unit_id,
                    "mid": mid,
                    "source_kind": source_kind,
                    "source_text": source_text,
                    "target_text": target_text,
                    "conf": conf,
                    "tag_map": tag_map,
                    "tag_sequence": tag_sequence,
                }
            )
    return eligible


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _available_tokens_from_env() -> int:
    explicit = _safe_int(os.getenv("SDLXLIFF_AVAILABLE_TOKENS"), 0)
    if explicit > 0:
        return max(1000, explicit)
    max_output_tokens = _safe_int(os.getenv("MAX_OUTPUT_TOKENS", "8192"), 8192)
    compression_factor = _safe_float(os.getenv("COMPRESSION_FACTOR", "2.0"), 2.0)
    if compression_factor <= 0:
        compression_factor = 0.000000000001
    return max(1000, int((max_output_tokens - 500) / compression_factor))


def _segment_source_hash(segments: List[Dict[str, Any]]) -> str:
    payload = [
        {
            "id": segment.get("id"),
            "unit_index": segment.get("unit_index"),
            "mid": segment.get("mid"),
            "source": segment.get("source_text"),
            "tag_sequence": segment.get("tag_sequence") or [],
        }
        for segment in segments
    ]
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _count_tokens(text: str) -> int:
    try:
        from chapter_splitter import ChapterSplitter

        return ChapterSplitter(model_name=os.getenv("MODEL", "gpt-3.5-turbo")).count_tokens(text)
    except Exception:
        return max(1, len(text) // 4)


def _batch_body(segments: List[Dict[str, Any]]) -> str:
    records = [{"id": str(segment["id"]), "source": segment["source_text"]} for segment in segments]
    return json.dumps(records, ensure_ascii=False, indent=2)


def _pack_segment_batches(segments: List[Dict[str, Any]], available_tokens: int) -> List[List[str]]:
    batches: List[List[str]] = []
    current: List[Dict[str, Any]] = []

    for segment in segments:
        candidate = current + [segment]
        if current and _count_tokens(_batch_body(candidate)) > available_tokens:
            batches.append([str(item["id"]) for item in current])
            current = [segment]
        else:
            current = candidate

    if current:
        batches.append([str(item["id"]) for item in current])
    return batches


def _cache_path(output_dir: str) -> str:
    cache_dir = os.path.join(output_dir, ".cache")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, "sdlxliff_batches.cache")


def _load_batch_cache(output_dir: str, source_hash: str, segment_ids: set) -> Optional[List[List[str]]]:
    path = _cache_path(output_dir)
    try:
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            cache = json.load(f)
        if cache.get("version") != 2:
            return None
        if cache.get("source_hash") != source_hash:
            print("SDLXLIFF batch cache invalidated: source segments changed")
            return None
        batches = []
        for batch in cache.get("batches", []):
            ids = [str(item) for item in batch.get("segment_ids", [])]
            if not ids or any(segment_id not in segment_ids for segment_id in ids):
                return None
            batches.append(ids)
        return batches or None
    except Exception as exc:
        print(f"Could not load SDLXLIFF batch cache: {exc}")
        return None


def _save_batch_cache(output_dir: str, source_hash: str, source_file: str, batches: List[List[str]]) -> None:
    path = _cache_path(output_dir)
    cache = {
        "version": 2,
        "source_hash": source_hash,
        "source_file": os.path.basename(source_file),
        "batch_count": len(batches),
        "batches": [
            {
                "num": idx + 1,
                "filename": f"section_{idx + 1}.txt",
                "segment_ids": ids,
            }
            for idx, ids in enumerate(batches)
        ],
    }
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
        print(f"Saved SDLXLIFF batch cache ({len(batches)} batch(es))")
    except Exception as exc:
        print(f"Could not save SDLXLIFF batch cache: {exc}")


def _chapter_from_batch(batch_num: int, batch_segments: List[Dict[str, Any]], source_path: str) -> Dict[str, Any]:
    body = _batch_body(batch_segments)
    segment_ids = [str(segment["id"]) for segment in batch_segments]
    return {
        "num": batch_num,
        "title": f"SDLXLIFF Batch {batch_num}",
        "body": body,
        "filename": f"section_{batch_num}.txt",
        "source_file": source_path,
        "content_hash": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        "file_size": len(body),
        "has_images": False,
        "image_count": 0,
        "is_chunk": False,
        "sdlxliff_batch": True,
        "sdlxliff_segment_ids": segment_ids,
        "sdlxliff_placeholder_only": all(is_placeholder_only_text(segment["source_text"]) for segment in batch_segments),
    }


def _manifest_segment(segment: Dict[str, Any], batch_num: Optional[int]) -> Dict[str, Any]:
    segment_copy = dict(segment)
    placeholder_only = is_placeholder_only_text(segment.get("source_text") or "")
    if placeholder_only:
        segment_copy["auto_insert"] = True
        segment_copy["auto_target_text"] = segment.get("source_text") or ""
        segment_copy["filename"] = None
    else:
        segment_copy.pop("source_text", None)
        segment_copy["auto_insert"] = False
        segment_copy["filename"] = f"section_{batch_num}.txt"
    segment_copy["batch_num"] = batch_num
    segment_copy["placeholder_only"] = placeholder_only
    return segment_copy


def extract_sdlxliff_to_chapters(path: str, output_dir: str) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    segments = iter_eligible_segments(path)
    source_hash = _segment_source_hash(segments)
    translatable_segments = [
        segment for segment in segments if not is_placeholder_only_text(segment.get("source_text") or "")
    ]
    auto_insert_segments = [
        segment for segment in segments if is_placeholder_only_text(segment.get("source_text") or "")
    ]
    translatable_by_id = {str(segment["id"]): segment for segment in translatable_segments}
    translatable_ids = set(translatable_by_id.keys())

    batches = _load_batch_cache(output_dir, source_hash, translatable_ids)
    if batches is not None:
        print(f"Loaded SDLXLIFF batch cache ({len(batches)} batch(es))")
    else:
        available_tokens = _available_tokens_from_env()
        print(f"SDLXLIFF JSON batch size: {available_tokens:,} tokens")
        batches = _pack_segment_batches(translatable_segments, available_tokens)
        _save_batch_cache(output_dir, source_hash, path, batches)

    chapters = []
    manifest_batches = []
    batch_assignments = {}
    for batch_num, ids in enumerate(batches, start=1):
        batch_segments = [translatable_by_id[segment_id] for segment_id in ids if segment_id in translatable_by_id]
        if not batch_segments:
            continue
        chapters.append(_chapter_from_batch(batch_num, batch_segments, path))
        manifest_batches.append(
            {
                "num": batch_num,
                "filename": f"section_{batch_num}.txt",
                "segment_ids": [str(segment["id"]) for segment in batch_segments],
            }
        )
        for segment in batch_segments:
            batch_assignments[str(segment["id"])] = batch_num

    manifest_segments = [
        _manifest_segment(segment, batch_assignments.get(str(segment["id"])))
        for segment in segments
    ]

    manifest = {
        "version": 2,
        "type": "sdlxliff_json_batches",
        "source_file": os.path.abspath(path),
        "source_basename": os.path.basename(path),
        "source_hash": source_hash,
        "segment_count": len(segments),
        "translatable_segment_count": len(translatable_segments),
        "auto_insert_segment_count": len(auto_insert_segments),
        "batch_count": len(chapters),
        "batches": manifest_batches,
        "segments": manifest_segments,
    }

    manifest_path = os.path.join(output_dir, "sdlxliff_manifest.json")
    chapters_path = os.path.join(output_dir, "chapters_full.json")
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    with open(chapters_path, "w", encoding="utf-8") as f:
        json.dump(chapters, f, ensure_ascii=False)
    metadata = {
        "title": os.path.splitext(os.path.basename(path))[0],
        "type": "sdlxliff",
        "source_file": os.path.abspath(path),
        "chapter_count": len(chapters),
        "segment_count": len(segments),
        "translatable_segment_count": len(translatable_segments),
        "auto_insert_segment_count": len(auto_insert_segments),
        "batch_count": len(chapters),
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return {
        "success": True,
        "chapters": len(chapters),
        "segments": len(segments),
        "translatable_segments": len(translatable_segments),
        "auto_insert_segments": len(auto_insert_segments),
        "batches": len(chapters),
        "manifest_path": manifest_path,
        "chapters_path": chapters_path,
        "metadata": metadata,
    }


def extract_sdlxliff_texts(path: str) -> List[str]:
    return [
        segment["source_text"]
        for segment in iter_eligible_segments(path)
        if not is_placeholder_only_text(segment.get("source_text") or "")
    ]


def copy_element_without_tail(elem: etree._Element) -> etree._Element:
    cloned = copy.deepcopy(elem)
    cloned.tail = None
    return cloned
