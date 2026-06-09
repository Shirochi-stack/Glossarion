"""Shared HTML SDLXLIFF sidecar writer.

This module intentionally stays small so both translation and review UI paths can
use the exact same writer in source runs and frozen builds.
"""

import os


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


def _write_html_sdlxliff_sidecar(output_dir, output_filename, chapter, source_html, target_html, raise_errors=False):
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

    try:
        import xml.etree.ElementTree as ET

        xliff_ns = "urn:oasis:names:tc:xliff:document:1.2"
        sdl_ns = "http://sdl.com/FileTypes/SdlXliff/1.0"
        ET.register_namespace("", xliff_ns)
        ET.register_namespace("sdl", sdl_ns)

        source_name = (
            chapter.get("original_basename")
            or chapter.get("original_filename")
            or chapter.get("filename")
            or output_name
        )
        source_lang = _html_sdlxliff_lang_code(os.getenv("SOURCE_LANGUAGE") or os.getenv("SOURCE_LANGUAGE_CODE"), "und")
        target_lang = _html_sdlxliff_lang_code(os.getenv("OUTPUT_LANGUAGE"), "und")

        root = ET.Element(f"{{{xliff_ns}}}xliff", {"version": "1.2"})
        file_el = ET.SubElement(root, f"{{{xliff_ns}}}file", {
            "original": str(source_name),
            "datatype": "html",
            "source-language": source_lang,
            "target-language": target_lang,
        })
        body_el = ET.SubElement(file_el, f"{{{xliff_ns}}}body")
        trans_unit = ET.SubElement(body_el, f"{{{xliff_ns}}}trans-unit", {"id": "1"})
        ET.SubElement(trans_unit, f"{{{xliff_ns}}}source").text = _html_sdlxliff_source_text(chapter, source_html)
        ET.SubElement(trans_unit, f"{{{xliff_ns}}}target").text = target_html

        sidecar_dir = os.path.join(output_dir, "SDLXLIFF")
        os.makedirs(sidecar_dir, exist_ok=True)
        sidecar_path = os.path.join(sidecar_dir, f"{output_name}.sdlxliff")
        ET.ElementTree(root).write(sidecar_path, encoding="utf-8", xml_declaration=True)
        return sidecar_path
    except Exception as exc:
        if raise_errors:
            raise
        print(f"WARNING: Failed to write SDLXLIFF sidecar for {output_filename}: {exc}")
        return None
