"""SDLXLIFF round-trip writer."""

from __future__ import annotations

import copy
import json
import os
import re
from typing import Dict, Optional

from lxml import etree

from sdlxliff_extractor import PLACEHOLDER_RE, _first_child_named, _iter_descendants_named


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


def _translated_candidates(output_dir: str, segment_num: int, filename: str):
    base = filename or f"section_{segment_num}.txt"
    stem, ext = os.path.splitext(base)
    candidates = [
        base,
        f"response_{base}",
        f"{stem}{ext}",
        f"response_{stem}{ext}",
        f"section_{segment_num}.txt",
        f"response_section_{segment_num}.txt",
    ]
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        yield os.path.join(output_dir, candidate)


def _read_translation(output_dir: str, segment: Dict) -> Optional[str]:
    for candidate in _translated_candidates(
        output_dir,
        int(segment.get("segment_num", 0) or 0),
        str(segment.get("filename") or ""),
    ):
        if os.path.exists(candidate):
            with open(candidate, "r", encoding="utf-8") as f:
                return f.read().strip()
    return None


def _validate_placeholders(text: str, tag_map: Dict[str, str]) -> bool:
    expected = set(tag_map.keys())
    found = set(PLACEHOLDER_RE.findall(text or ""))
    return expected == found


def _parse_fragment(xml_text: str) -> etree._Element:
    parser = etree.XMLParser(remove_blank_text=False, recover=False, huge_tree=True)
    return etree.fromstring(xml_text.encode("utf-8"), parser=parser)


def _set_mixed_content(elem: etree._Element, text: str, tag_map: Dict[str, str]) -> None:
    for child in list(elem):
        elem.remove(child)
    elem.text = None
    parts = re.split(f"({PLACEHOLDER_RE.pattern})", text)
    current_parent = elem
    previous_child = None
    for part in parts:
        if not part:
            continue
        if part in tag_map:
            child = _parse_fragment(tag_map[part])
            child = copy.deepcopy(child)
            child.tail = None
            elem.append(child)
            previous_child = child
            current_parent = elem
            continue
        if previous_child is None:
            elem.text = (elem.text or "") + part
        else:
            previous_child.tail = (previous_child.tail or "") + part


def convert_sdlxliff(output_dir: str, manifest_path: Optional[str] = None, output_path: Optional[str] = None) -> Dict:
    manifest_path = manifest_path or os.path.join(output_dir, "sdlxliff_manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    source_file = manifest["source_file"]
    parser = etree.XMLParser(remove_blank_text=False, recover=False, huge_tree=True)
    tree = etree.parse(source_file, parser)
    root = tree.getroot()
    trans_units = list(_iter_trans_units(root))

    updated = 0
    skipped = 0
    missing = 0
    for segment in manifest.get("segments", []):
        translation = _read_translation(output_dir, segment)
        if translation is None:
            missing += 1
            continue
        tag_map = segment.get("tag_map") or {}
        if not _validate_placeholders(translation, tag_map):
            skipped += 1
            continue
        unit_index = int(segment.get("unit_index", -1))
        if unit_index < 0 or unit_index >= len(trans_units):
            skipped += 1
            continue
        target_elem = _ensure_target_segment(trans_units[unit_index], segment.get("mid"))
        try:
            _set_mixed_content(target_elem, translation, tag_map)
        except Exception:
            skipped += 1
            continue
        updated += 1

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
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Write translated SDLXLIFF from Glossarion output files.")
    parser.add_argument("output_dir")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    result = convert_sdlxliff(args.output_dir, args.manifest, args.output)
    print(json.dumps(result, ensure_ascii=False))
