"""SDLXLIFF JSON batch round-trip writer."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from lxml import etree

from sdlxliff_extractor import (
    PLACEHOLDER_RE,
    _first_child_named,
    _iter_descendants_named,
    normalize_target_language_code,
)


def _local_name(tag) -> str:
    if not isinstance(tag, str):
        return ""
    return etree.QName(tag).localname if tag.startswith("{") else tag


def _ns_uri(tag) -> str:
    if not isinstance(tag, str) or not tag.startswith("{"):
        return ""
    return etree.QName(tag).namespace or ""


def _qname_like(parent: etree._Element, local: str) -> str:
    ns = _ns_uri(parent.tag)
    return f"{{{ns}}}{local}" if ns else local


def _iter_trans_units(root: etree._Element):
    for elem in root.iter():
        if _local_name(elem.tag) == "trans-unit":
            yield elem


def _iter_file_elements(root: etree._Element):
    for elem in root.iter():
        if _local_name(elem.tag) == "file":
            yield elem


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


def _ensure_target_segment(trans_unit: etree._Element, mid: Optional[str]) -> etree._Element:
    target = _first_child_named(trans_unit, "target")
    if target is None:
        target = etree.Element(_qname_like(trans_unit, "target"), nsmap=trans_unit.nsmap)
        seg_source = _first_child_named(trans_unit, "seg-source")
        source = _first_child_named(trans_unit, "source")
        insert_after = seg_source if seg_source is not None else source
        if insert_after is not None:
            trans_unit.insert(trans_unit.index(insert_after) + 1, target)
        else:
            trans_unit.insert(0, target)
    if mid is None:
        return target
    existing = _find_target_segment(trans_unit, mid)
    if existing is not None:
        return existing
    mrk = etree.SubElement(target, _qname_like(target, "mrk"))
    mrk.set("mtype", "seg")
    mrk.set("mid", str(mid))
    return mrk


def _batch_output_candidates(output_dir: str, filename: str) -> Iterable[str]:
    base = filename or ""
    candidates = [base]
    if base and not os.path.basename(base).startswith("response_"):
        candidates.append(f"response_{base}")
    seen = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        yield os.path.join(output_dir, candidate)


def _normalize_output_name(value: str) -> str:
    return os.path.basename(str(value or "")).lower()


def _completed_outputs(output_dir: str) -> Optional[set]:
    progress_path = os.path.join(output_dir, "translation_progress.json")
    if not os.path.exists(progress_path):
        return None
    try:
        with open(progress_path, "r", encoding="utf-8") as f:
            progress = json.load(f)
    except Exception:
        return set()
    completed = set()
    for info in (progress.get("chapters") or {}).values():
        if not isinstance(info, dict):
            continue
        if str(info.get("status") or "").lower() != "completed":
            continue
        output_file = info.get("output_file")
        if output_file:
            completed.add(_normalize_output_name(output_file))
    return completed


def _read_completed_batch(output_dir: str, batch: Dict[str, Any], completed_outputs: Optional[set]) -> Tuple[Optional[str], Optional[str]]:
    found_uncompleted = False
    for candidate in _batch_output_candidates(output_dir, str(batch.get("filename") or "")):
        if not os.path.exists(candidate):
            continue
        if completed_outputs is not None and _normalize_output_name(candidate) not in completed_outputs:
            found_uncompleted = True
            continue
        with open(candidate, "r", encoding="utf-8") as f:
            return f.read(), None
    return None, "invalid_status" if found_uncompleted else "missing"


def _validate_placeholders(text: str, segment: Dict[str, Any]) -> bool:
    expected = list(segment.get("tag_sequence") or [])
    found = PLACEHOLDER_RE.findall(text or "")
    return found == expected


def _nsmap_from_entries(entries: List[Dict[str, str]]) -> Dict[Optional[str], str]:
    nsmap: Dict[Optional[str], str] = {}
    for entry in entries or []:
        if not isinstance(entry, dict):
            continue
        uri = entry.get("uri")
        if not uri:
            continue
        prefix = entry.get("prefix")
        nsmap[None if prefix in (None, "") else str(prefix)] = str(uri)
    return nsmap


def _make_element(info: Dict[str, Any]) -> etree._Element:
    tag = info.get("tag")
    if not tag:
        raise ValueError("Missing inline tag name")
    nsmap = _nsmap_from_entries(info.get("nsmap") or [])
    attrib = dict(info.get("attrib") or {})
    return etree.Element(tag, attrib=attrib, nsmap=nsmap or None)


def _set_mixed_content(elem: etree._Element, text: str, tag_map: Dict[str, Dict[str, Any]]) -> None:
    for child in list(elem):
        elem.remove(child)
    elem.text = None

    stack = [{"elem": elem, "last_child": None}]

    def append_text(value: str) -> None:
        if not value:
            return
        context = stack[-1]
        last_child = context["last_child"]
        if last_child is None:
            context["elem"].text = (context["elem"].text or "") + value
        else:
            last_child.tail = (last_child.tail or "") + value

    for part in re.split(f"({PLACEHOLDER_RE.pattern})", text or ""):
        if part == "":
            continue
        info = tag_map.get(part)
        if info is None:
            append_text(part)
            continue

        kind = info.get("kind")
        if kind == "empty":
            child = _make_element(info)
            stack[-1]["elem"].append(child)
            stack[-1]["last_child"] = child
        elif kind == "start":
            child = _make_element(info)
            stack[-1]["elem"].append(child)
            stack[-1]["last_child"] = child
            stack.append({"elem": child, "last_child": None})
        elif kind == "end":
            if len(stack) <= 1:
                raise ValueError("Unexpected inline end placeholder")
            stack.pop()
        else:
            raise ValueError(f"Unknown inline placeholder kind: {kind}")

    if len(stack) != 1:
        raise ValueError("Unclosed inline placeholder")


def _parse_batch_translation(text: str, expected_ids: List[str]) -> Tuple[Dict[str, str], List[str], List[str]]:
    try:
        payload = json.loads(text)
    except Exception as exc:
        raise ValueError(f"Malformed JSON batch output: {exc}") from exc
    if not isinstance(payload, list):
        raise ValueError("SDLXLIFF batch output must be a JSON array")

    expected_set = set(expected_ids)
    seen = set()
    translations: Dict[str, str] = {}
    seen_order: List[str] = []

    for item in payload:
        if not isinstance(item, dict) or set(item.keys()) != {"id", "target"}:
            raise ValueError("SDLXLIFF batch records must contain exactly id and target")
        segment_id = str(item.get("id"))
        if segment_id in seen:
            raise ValueError(f"Duplicate SDLXLIFF batch id: {segment_id}")
        seen.add(segment_id)
        if segment_id not in expected_set:
            raise ValueError(f"Extra SDLXLIFF batch id: {segment_id}")
        target = item.get("target")
        if not isinstance(target, str):
            raise ValueError(f"SDLXLIFF batch target must be a string: {segment_id}")
        translations[segment_id] = target
        seen_order.append(segment_id)

    missing = [segment_id for segment_id in expected_ids if segment_id not in translations]
    if missing:
        raise ValueError(f"Missing SDLXLIFF batch ids: {', '.join(missing)}")
    if seen_order != expected_ids:
        raise ValueError("SDLXLIFF batch ids are out of order")
    return translations, [], []


def _apply_segment_translation(segment: Dict[str, Any], translation: str, trans_units: List[etree._Element]) -> bool:
    if not _validate_placeholders(translation, segment):
        return False
    unit_index = int(segment.get("unit_index", -1))
    if unit_index < 0 or unit_index >= len(trans_units):
        return False
    target_elem = _ensure_target_segment(trans_units[unit_index], segment.get("mid"))
    try:
        _set_mixed_content(target_elem, translation, segment.get("tag_map") or {})
    except Exception:
        return False
    return True


def _target_language_code(manifest: Dict[str, Any]) -> Optional[str]:
    return (
        normalize_target_language_code(manifest.get("target_language_code"))
        or normalize_target_language_code(manifest.get("target_language"))
        or normalize_target_language_code(os.getenv("OUTPUT_LANGUAGE"))
        or normalize_target_language_code(os.getenv("GLOSSARY_TARGET_LANGUAGE"))
    )


def _apply_target_language(root: etree._Element, manifest: Dict[str, Any]) -> Optional[str]:
    code = _target_language_code(manifest)
    if not code:
        return None
    for file_elem in _iter_file_elements(root):
        file_elem.set("target-language", code)
    return code


def convert_sdlxliff(output_dir: str, manifest_path: Optional[str] = None, output_path: Optional[str] = None) -> Dict:
    manifest_path = manifest_path or os.path.join(output_dir, "sdlxliff_manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    source_file = manifest["source_file"]
    parser = etree.XMLParser(remove_blank_text=False, recover=False, huge_tree=True)
    tree = etree.parse(source_file, parser)
    root = tree.getroot()
    target_language_code = _apply_target_language(root, manifest)
    trans_units = list(_iter_trans_units(root))

    segments_by_id = {str(segment.get("id")): segment for segment in manifest.get("segments", [])}
    completed_outputs = _completed_outputs(output_dir)
    updated = 0
    skipped = 0
    missing = 0
    invalid_batches = 0

    for segment in manifest.get("segments", []):
        if not segment.get("auto_insert"):
            continue
        if _apply_segment_translation(segment, str(segment.get("auto_target_text") or ""), trans_units):
            updated += 1
        else:
            skipped += 1

    for batch in manifest.get("batches", []):
        expected_ids = [str(segment_id) for segment_id in batch.get("segment_ids", [])]
        batch_text, read_error = _read_completed_batch(output_dir, batch, completed_outputs)
        if batch_text is None:
            if read_error == "missing":
                missing += len(expected_ids)
            else:
                skipped += len(expected_ids)
            invalid_batches += 1
            continue

        try:
            translations, skipped_ids, missing_ids = _parse_batch_translation(batch_text, expected_ids)
        except Exception:
            skipped += len(expected_ids)
            invalid_batches += 1
            continue

        skipped += len(skipped_ids)
        missing += len(missing_ids)

        for segment_id, translation in translations.items():
            segment = segments_by_id.get(segment_id)
            if not segment:
                skipped += 1
                continue
            if _apply_segment_translation(segment, translation, trans_units):
                updated += 1
            else:
                skipped += 1

    if output_path is None:
        stem, ext = os.path.splitext(os.path.basename(source_file))
        output_path = os.path.join(output_dir, f"{stem}_translated{ext or '.sdlxliff'}")
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return {
        "success": True,
        "output_path": output_path,
        "updated": updated,
        "skipped": skipped,
        "missing": missing,
        "invalid_batches": invalid_batches,
        "target_language_code": target_language_code,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Write translated SDLXLIFF from Glossarion JSON batch output files.")
    parser.add_argument("output_dir")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    result = convert_sdlxliff(args.output_dir, args.manifest, args.output)
    print(json.dumps(result, ensure_ascii=False))
