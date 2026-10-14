"""Shared HTML SDLXLIFF sidecar writer.

This module intentionally stays small so both translation and review UI paths can
use the exact same writer in source runs and frozen builds.
"""

import hashlib
import json
import os
import threading
import time


GLOSSARION_SDLXLIFF_NS = "urn:glossarion:sdlxliff"
MANUAL_UNTRANSLATED_ATTRIBUTE = f"{{{GLOSSARION_SDLXLIFF_NS}}}manual-untranslated"
MANUAL_EDITING_ATTRIBUTE = f"{{{GLOSSARION_SDLXLIFF_NS}}}manual-editing"
USER_ADDED_TARGET_INDEXES_ATTRIBUTE = (
    f"{{{GLOSSARION_SDLXLIFF_NS}}}user-added-target-indexes"
)
USER_ADDED_BREAK_POSITIONS_ATTRIBUTE = (
    f"{{{GLOSSARION_SDLXLIFF_NS}}}user-added-break-positions"
)
_SIDECAR_FRESHNESS_MANIFEST_TYPE = "html_sdlxliff_sidecar_freshness"
_SIDECAR_FRESHNESS_MANIFEST_LOCK = threading.RLock()
_SIDECAR_MUTATION_LOCKS_GUARD = threading.Lock()
_SIDECAR_MUTATION_LOCKS = {}


def _sdlxliff_mutation_lock(path):
    """Return one in-process lock for mutations of a specific sidecar."""
    key = os.path.normcase(os.path.abspath(os.fspath(path)))
    with _SIDECAR_MUTATION_LOCKS_GUARD:
        lock = _SIDECAR_MUTATION_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _SIDECAR_MUTATION_LOCKS[key] = lock
        return lock


def _write_sdlxliff_tree_atomically(tree, path):
    """Replace one sidecar from a unique same-directory temporary file."""
    path = os.fspath(path)
    temp_path = (
        f"{path}.{os.getpid()}.{threading.get_ident()}."
        f"{time.time_ns()}.tmp"
    )
    try:
        tree.write(temp_path, encoding="utf-8", xml_declaration=True)
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def _sidecar_freshness_logical_key(output_name):
    name = os.path.basename(str(output_name or "").replace("\\", "/"))
    if name.casefold().endswith(".sdlxliff"):
        name = name[:-len(".sdlxliff")]
    if name.startswith("response_"):
        name = name[len("response_"):]
    while True:
        stem, extension = os.path.splitext(name)
        if not extension:
            break
        name = stem
    return name.casefold()


def _sidecar_freshness_file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        while True:
            block = source.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _sidecar_freshness_file_stat(path):
    stat = os.stat(path)
    return {
        "size": int(stat.st_size),
        "mtime_ns": int(getattr(
            stat,
            "st_mtime_ns",
            int(stat.st_mtime * 1000000000),
        )),
    }


def _record_html_sdlxliff_freshness(
    output_dir,
    output_name,
    sidecar_path,
    source_payload,
):
    """Record a sidecar created by the live translation pipeline.

    The final HTML is already on disk when the shared writer is called.  If a
    caller uses the writer only to create a manual/source-only sidecar, there
    is no output file and no freshness record is needed.
    """
    output_path = os.path.join(output_dir, output_name)
    if not os.path.isfile(output_path):
        return False
    logical_key = _sidecar_freshness_logical_key(output_name)
    if not logical_key:
        return False
    manifest_dir = os.path.join(output_dir, "SDLXLIFF")
    manifest_path = os.path.join(manifest_dir, "sdlxliff_manifest.json")
    output_stat = _sidecar_freshness_file_stat(output_path)
    record = {
        "output_name": output_name,
        "sidecar_name": os.path.basename(sidecar_path),
        "output_sha256": _sidecar_freshness_file_hash(output_path),
        "output_size": output_stat["size"],
        "output_mtime_ns": output_stat["mtime_ns"],
        "source_sha256": hashlib.sha256(
            str(source_payload or "").encode("utf-8")
        ).hexdigest(),
        "source": None,
    }
    with _SIDECAR_FRESHNESS_MANIFEST_LOCK:
        manifest = None
        try:
            if os.path.isfile(manifest_path):
                with open(manifest_path, "r", encoding="utf-8") as source:
                    candidate = json.load(source)
                if (
                    isinstance(candidate, dict)
                    and candidate.get("type") == _SIDECAR_FRESHNESS_MANIFEST_TYPE
                    and isinstance(candidate.get("entries"), dict)
                ):
                    manifest = candidate
        except Exception:
            manifest = None
        if manifest is None:
            manifest = {
                "version": 1,
                "type": _SIDECAR_FRESHNESS_MANIFEST_TYPE,
                "hash_algorithm": "sha256",
                "entries": {},
            }
        manifest["entries"][logical_key] = record
        manifest["version"] = 1
        manifest["type"] = _SIDECAR_FRESHNESS_MANIFEST_TYPE
        manifest["hash_algorithm"] = "sha256"
        temp_path = (
            f"{manifest_path}.{os.getpid()}.{threading.get_ident()}."
            f"{time.time_ns()}.tmp"
        )
        try:
            with open(temp_path, "w", encoding="utf-8") as target:
                json.dump(manifest, target, ensure_ascii=False, indent=2)
            os.replace(temp_path, manifest_path)
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
    return True


def _html_sdlxliff_enabled():
    return str(os.getenv("OUTPUT_SDLXLIFF", "1")).strip().lower() in {"1", "true", "yes", "on"}


def _html_sdlxliff_lang_code(value, default="und"):
    value = str(value or "").strip()
    if not value:
        return default
    try:
        from sdlxliff_extractor import normalize_target_language_code
        return normalize_target_language_code(value) or default
    except Exception:
        return value or default


def _html_sdlxliff_source_text(chapter, fallback=""):
    chapter = chapter if isinstance(chapter, dict) else {}
    for key in ("original_html", "source_html", "raw_html", "body", "content"):
        value = chapter.get(key)
        if isinstance(value, str) and value:
            return value
    return fallback if isinstance(fallback, str) else str(fallback or "")


def _html_sdlxliff_blank_manual_target(source_html):
    """Keep the source document structure while blanking editable body text."""
    source_html = source_html if isinstance(source_html, str) else str(source_html or "")
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(source_html, "html.parser")
        container = soup.body or soup
        for text_node in list(container.find_all(string=True)):
            parent_name = str(getattr(getattr(text_node, "parent", None), "name", "") or "").lower()
            if parent_name in {"script", "style"}:
                continue
            text_node.replace_with("")
        return str(soup)
    except Exception:
        return ""


def _is_manual_untranslated_sdlxliff(path_or_root):
    """Return whether a sidecar is an untouched manual-editing placeholder."""
    try:
        import xml.etree.ElementTree as ET

        root = path_or_root
        if isinstance(path_or_root, (str, bytes, os.PathLike)):
            root = ET.parse(path_or_root).getroot()
        for element in root.iter():
            if str(element.tag).rsplit("}", 1)[-1] != "file":
                continue
            return str(element.attrib.get(MANUAL_UNTRANSLATED_ATTRIBUTE, "")).lower() in {
                "1", "true", "yes", "on"
            }
    except Exception:
        pass
    return False


def _is_manual_editing_sdlxliff(path_or_root):
    """Return whether a sidecar uses sparse, structurally indexed manual targets."""
    try:
        import xml.etree.ElementTree as ET

        root = path_or_root
        if isinstance(path_or_root, (str, bytes, os.PathLike)):
            root = ET.parse(path_or_root).getroot()
        for element in root.iter():
            if str(element.tag).rsplit("}", 1)[-1] != "file":
                continue
            return str(element.attrib.get(MANUAL_EDITING_ATTRIBUTE, "")).lower() in {
                "1", "true", "yes", "on"
            }
    except Exception:
        pass
    return False


def _clear_manual_untranslated_sdlxliff(root):
    """Promote a manual placeholder to an edited SDLXLIFF document."""
    changed = False
    for element in root.iter():
        local_name = str(element.tag).rsplit("}", 1)[-1]
        if local_name == "file" and MANUAL_UNTRANSLATED_ATTRIBUTE in element.attrib:
            element.attrib.pop(MANUAL_UNTRANSLATED_ATTRIBUTE, None)
            changed = True
        elif local_name == "target" and element.attrib.get("state") == "new":
            element.set("state", "translated")
            changed = True
    return changed


def _blank_manual_untranslated_sdlxliff_target(path):
    """Blank a legacy raw-filled manual target while retaining its HTML skeleton."""
    try:
        import xml.etree.ElementTree as ET

        with _sdlxliff_mutation_lock(path):
            tree = ET.parse(path)
            root = tree.getroot()
            if not _is_manual_untranslated_sdlxliff(root):
                return False
            source_element = None
            target_element = None
            for element in root.iter():
                local_name = str(element.tag).rsplit("}", 1)[-1]
                if local_name == "source" and source_element is None:
                    source_element = element
                elif local_name == "target" and target_element is None:
                    target_element = element
            if source_element is None or target_element is None:
                return False
            blank_target = _html_sdlxliff_blank_manual_target(source_element.text or "")
            if not list(target_element) and (target_element.text or "") == blank_target:
                return False
            for child in list(target_element):
                target_element.remove(child)
            target_element.text = blank_target
            target_element.set("state", "new")
            ET.register_namespace("", "urn:oasis:names:tc:xliff:document:1.2")
            ET.register_namespace("sdl", "http://sdl.com/FileTypes/SdlXliff/1.0")
            ET.register_namespace("glossarion", GLOSSARION_SDLXLIFF_NS)
            _write_sdlxliff_tree_atomically(tree, path)
            return True
    except Exception:
        return False


def _reset_sdlxliff_target_for_manual_retranslation(path):
    """Turn a translated HTML sidecar into a source-only manual placeholder.

    Every target keeps the corresponding source document's tag skeleton, but
    all editable text is removed.  A path-scoped lock plus atomic replacement
    makes repeated or concurrent resets idempotent and prevents partial XML.
    """
    import xml.etree.ElementTree as ET

    if not path or not os.path.isfile(path):
        return False

    with _sdlxliff_mutation_lock(path):
        # Recheck after acquiring the lock in case another reset replaced it.
        if not os.path.isfile(path):
            return False
        tree = ET.parse(path)
        root = tree.getroot()
        file_elements = [
            element
            for element in root.iter()
            if str(element.tag).rsplit("}", 1)[-1] == "file"
        ]
        if not file_elements:
            raise ValueError("SDLXLIFF file element not found")
        for file_element in file_elements:
            file_element.set(MANUAL_UNTRANSLATED_ATTRIBUTE, "true")
            file_element.set(MANUAL_EDITING_ATTRIBUTE, "true")
            file_element.attrib.pop(USER_ADDED_TARGET_INDEXES_ATTRIBUTE, None)
            file_element.attrib.pop(USER_ADDED_BREAK_POSITIONS_ATTRIBUTE, None)

        reset_count = 0
        for trans_unit in root.iter():
            if str(trans_unit.tag).rsplit("}", 1)[-1] != "trans-unit":
                continue
            source_element = None
            target_element = None
            for child in list(trans_unit):
                local_name = str(child.tag).rsplit("}", 1)[-1]
                if local_name == "source" and source_element is None:
                    source_element = child
                elif local_name == "target" and target_element is None:
                    target_element = child
            if target_element is None:
                continue
            blank_target = _html_sdlxliff_blank_manual_target(
                source_element.text if source_element is not None else ""
            )
            for child in list(target_element):
                target_element.remove(child)
            target_element.text = blank_target
            target_element.set("state", "new")
            reset_count += 1

        if not reset_count:
            raise ValueError("SDLXLIFF target element not found")

        ET.register_namespace("", "urn:oasis:names:tc:xliff:document:1.2")
        ET.register_namespace("sdl", "http://sdl.com/FileTypes/SdlXliff/1.0")
        ET.register_namespace("glossarion", GLOSSARION_SDLXLIFF_NS)
        _write_sdlxliff_tree_atomically(tree, path)
        return True


def _write_html_sdlxliff_sidecar(
    output_dir,
    output_filename,
    chapter,
    source_html,
    target_html,
    raise_errors=False,
    manual_untranslated=False,
    record_freshness=True,
    preserve_review_metadata=False,
):
    if not _html_sdlxliff_enabled():
        return None
    if not output_dir or not output_filename:
        return None
    if not isinstance(target_html, str):
        return None

    chapter = chapter if isinstance(chapter, dict) else {}
    if chapter.get("sdlxliff_batch") or chapter.get("sdlxliff_segment"):
        return None

    output_name = os.path.basename(str(output_filename).replace("\\", "/"))
    if not output_name.lower().endswith((".html", ".htm", ".xhtml")):
        return None

    sidecar_dir = os.path.join(output_dir, "SDLXLIFF")
    sidecar_path = os.path.join(sidecar_dir, f"{output_name}.sdlxliff")

    try:
        import xml.etree.ElementTree as ET

        xliff_ns = "urn:oasis:names:tc:xliff:document:1.2"
        sdl_ns = "http://sdl.com/FileTypes/SdlXliff/1.0"
        ET.register_namespace("", xliff_ns)
        ET.register_namespace("sdl", sdl_ns)
        ET.register_namespace("glossarion", GLOSSARION_SDLXLIFF_NS)

        source_name = (
            chapter.get("original_basename")
            or chapter.get("original_filename")
            or chapter.get("filename")
            or output_name
        )
        source_lang = _html_sdlxliff_lang_code(os.getenv("SOURCE_LANGUAGE") or os.getenv("SOURCE_LANGUAGE_CODE"), "und")
        target_lang = _html_sdlxliff_lang_code(os.getenv("OUTPUT_LANGUAGE"), "und")

        source_payload = _html_sdlxliff_source_text(chapter, source_html)
        target_payload = (
            _html_sdlxliff_blank_manual_target(source_payload)
            if manual_untranslated
            else target_html
        )

        preserved_review_attributes = {}
        if preserve_review_metadata and os.path.isfile(sidecar_path):
            try:
                existing_root = ET.parse(sidecar_path).getroot()
                review_prefix = f"{{{GLOSSARION_SDLXLIFF_NS}}}"
                for existing_element in existing_root.iter():
                    if str(existing_element.tag).rsplit("}", 1)[-1] != "file":
                        continue
                    preserved_review_attributes = {
                        key: value
                        for key, value in existing_element.attrib.items()
                        if key.startswith(review_prefix)
                        and key != MANUAL_UNTRANSLATED_ATTRIBUTE
                    }
                    break
            except Exception:
                preserved_review_attributes = {}

        root = ET.Element(f"{{{xliff_ns}}}xliff", {"version": "1.2"})
        file_attributes = {
            "original": str(source_name),
            "datatype": "html",
            "source-language": source_lang,
            "target-language": target_lang,
        }
        file_attributes.update(preserved_review_attributes)
        if manual_untranslated:
            file_attributes[MANUAL_UNTRANSLATED_ATTRIBUTE] = "true"
            file_attributes[MANUAL_EDITING_ATTRIBUTE] = "true"
        file_el = ET.SubElement(root, f"{{{xliff_ns}}}file", file_attributes)
        body_el = ET.SubElement(file_el, f"{{{xliff_ns}}}body")
        trans_unit = ET.SubElement(body_el, f"{{{xliff_ns}}}trans-unit", {"id": "1"})
        ET.SubElement(trans_unit, f"{{{xliff_ns}}}source").text = source_payload
        target_attributes = {"state": "new"} if manual_untranslated else {}
        ET.SubElement(trans_unit, f"{{{xliff_ns}}}target", target_attributes).text = target_payload

        os.makedirs(sidecar_dir, exist_ok=True)
        ET.ElementTree(root).write(sidecar_path, encoding="utf-8", xml_declaration=True)
        if not manual_untranslated and record_freshness:
            try:
                _record_html_sdlxliff_freshness(
                    output_dir,
                    output_name,
                    sidecar_path,
                    source_payload,
                )
            except Exception:
                # Hash tracking must never turn a valid translation sidecar
                # into a failed chapter save.  Without a record, the reviewer
                # intentionally falls back to the legacy mtime check.
                pass
        return sidecar_path
    except Exception as exc:
        if raise_errors:
            raise
        print(f"WARNING: Failed to write SDLXLIFF sidecar for {output_filename}: {exc}")
        return None
