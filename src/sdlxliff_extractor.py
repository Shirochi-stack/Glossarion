"""SDLXLIFF extraction helpers.

This module treats SDLXLIFF as a round-trippable XML container. It extracts
eligible source segments into Glossarion-style chapter dictionaries while
protecting inline XML tags with placeholders that the converter can restore.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
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
    for elem in (trans_unit,):
        if str(elem.get("locked", "")).strip().lower() in truthy:
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


def _protect_inline_xml(segment_elem: etree._Element, segment_num: int) -> Tuple[str, Dict[str, str]]:
    parts: List[str] = []
    tag_map: Dict[str, str] = {}
    if segment_elem.text:
        parts.append(segment_elem.text)
    for tag_idx, child in enumerate(segment_elem):
        placeholder = f"[[XLIFF_TAG_{segment_num:06d}_{tag_idx:04d}]]"
        tag_map[placeholder] = etree.tostring(child, encoding="unicode", with_tail=False)
        parts.append(placeholder)
        if child.tail:
            parts.append(child.tail)
    return "".join(parts).strip(), tag_map


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
            source_text, tag_map = _protect_inline_xml(src_elem, segment_num)
            if not source_text:
                continue
            eligible.append(
                {
                    "segment_num": segment_num,
                    "unit_index": unit_index,
                    "unit_id": unit_id,
                    "mid": mid,
                    "source_kind": source_kind,
                    "source_text": source_text,
                    "target_text": target_text,
                    "conf": conf,
                    "tag_map": tag_map,
                }
            )
    return eligible


def _chapter_from_segment(segment: Dict[str, Any], source_path: str) -> Dict[str, Any]:
    num = int(segment["segment_num"])
    body = segment["source_text"]
    return {
        "num": num,
        "title": f"SDLXLIFF Segment {num}",
        "body": body,
        "filename": f"section_{num}.txt",
        "source_file": source_path,
        "content_hash": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        "file_size": len(body),
        "has_images": False,
        "image_count": 0,
        "is_chunk": False,
        "sdlxliff_segment": True,
    }


def _worker_count() -> int:
    try:
        workers = int(os.getenv("EXTRACTION_WORKERS", "2"))
    except Exception:
        workers = 2
    return max(1, workers)


def extract_sdlxliff_to_chapters(path: str, output_dir: str) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    segments = iter_eligible_segments(path)
    workers = min(_worker_count(), max(1, len(segments)))
    if workers > 1 and len(segments) > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            chapters = list(pool.map(lambda seg: _chapter_from_segment(seg, path), segments))
    else:
        chapters = [_chapter_from_segment(seg, path) for seg in segments]

    manifest_segments = []
    for segment in segments:
        segment_copy = dict(segment)
        segment_copy.pop("source_text", None)
        segment_copy["filename"] = f"section_{segment['segment_num']}.txt"
        manifest_segments.append(segment_copy)

    manifest = {
        "version": 1,
        "source_file": os.path.abspath(path),
        "source_basename": os.path.basename(path),
        "segment_count": len(segments),
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
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return {
        "success": True,
        "chapters": len(chapters),
        "segments": len(segments),
        "manifest_path": manifest_path,
        "chapters_path": chapters_path,
        "metadata": metadata,
    }


def extract_sdlxliff_texts(path: str) -> List[str]:
    return [segment["source_text"] for segment in iter_eligible_segments(path)]


def copy_element_without_tail(elem: etree._Element) -> etree._Element:
    cloned = copy.deepcopy(elem)
    cloned.tail = None
    return cloned
