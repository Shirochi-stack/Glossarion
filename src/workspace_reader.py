"""HTML-workspace support for the integrated EPUB reader.

The reader UI is useful for any ordered set of HTML documents, not only for
files stored inside an EPUB zip.  This module turns a translation output
folder into that neutral chapter manifest and provides the deliberately lazy
PDF raw-section cache used by the reader's Raw toggle.
"""

from __future__ import annotations

import hashlib
import html
import json
import os
import re
import threading
from pathlib import Path
from typing import Dict, List

from output_workspace import read_workspace_source_path, source_format_label


_HTML_EXTENSIONS = {".html", ".htm", ".xhtml"}


def _load_json(path: Path, default=None):
    try:
        with path.open("r", encoding="utf-8-sig", errors="replace") as stream:
            value = json.load(stream)
        return value
    except (OSError, UnicodeError, json.JSONDecodeError):
        return default


def _entry_number(item) -> tuple:
    key, entry = item
    for field in ("actual_num", "chapter_num"):
        try:
            return (0, int(entry.get(field)), str(key))
        except (TypeError, ValueError):
            pass
    match = re.search(r"(\d+)(?!.*\d)", str(key))
    if match:
        return (1, int(match.group(1)), str(key))
    return (2, 0, str(key))


def _is_reader_entry(key: str, entry: dict) -> bool:
    if not isinstance(entry, dict):
        return False
    output_file = str(entry.get("output_file") or "").strip()
    original = str(entry.get("original_basename") or "").strip()
    if os.path.basename(output_file).casefold() == "source_epub.txt":
        return False
    if os.path.basename(original).casefold() == "source_epub.txt":
        return False
    if entry.get("special_type") or entry.get("translation_artifact_progress_key"):
        return False
    if entry.get("auto_discovered") and not entry.get("pdf_toc_section"):
        return False
    extensions = {
        os.path.splitext(output_file)[1].lower(),
        os.path.splitext(original)[1].lower(),
    }
    return bool(extensions & _HTML_EXTENSIONS or entry.get("pdf_toc_section"))


def _display_title(entry: dict, fallback: str) -> str:
    for field in (
        "pdf_toc_title",
        "pdf_section_title",
        "translated_title",
        "title",
    ):
        value = " ".join(str(entry.get(field) or "").split())
        if value:
            return value
    stem = os.path.splitext(os.path.basename(fallback))[0]
    return stem.replace("_", " ").replace("-", " ").strip() or fallback


def build_workspace_reader_manifest(
    output_dir: str,
    *,
    source_path: str | None = None,
) -> Dict:
    """Build an ordered, format-aware reader manifest for *output_dir*."""
    workspace = os.path.abspath(os.path.normpath(str(output_dir or "")))
    if not os.path.isdir(workspace):
        raise FileNotFoundError(f"Translation workspace not found: {workspace}")

    source = str(source_path or read_workspace_source_path(workspace) or "").strip()
    if source and not os.path.isabs(source):
        source = os.path.abspath(os.path.join(workspace, source))
    source_format = source_format_label(source).lower()

    progress_path = Path(workspace) / "translation_progress.json"
    progress = _load_json(progress_path, {}) or {}
    chapters = progress.get("chapters") if isinstance(progress, dict) else {}
    chapters = chapters if isinstance(chapters, dict) else {}

    entries: List[Dict] = []
    for key, entry in sorted(chapters.items(), key=_entry_number):
        if not _is_reader_entry(str(key), entry):
            continue
        output_file = str(entry.get("output_file") or "").strip()
        original = str(entry.get("original_basename") or "").strip()
        translated_path = ""
        if output_file:
            candidate = output_file if os.path.isabs(output_file) else os.path.join(
                workspace, output_file
            )
            candidate = os.path.normpath(candidate)
            if os.path.isfile(candidate):
                translated_path = candidate
        filename = os.path.basename(output_file or original or str(key))
        entries.append(
            {
                "key": str(key),
                "filename": filename,
                "original_filename": os.path.basename(original or filename),
                "title": _display_title(entry, original or output_file or str(key)),
                "translated_path": translated_path,
                "status": str(entry.get("status") or ""),
                "pdf_toc_section": bool(entry.get("pdf_toc_section")),
                "pdf_section_id": str(entry.get("pdf_section_id") or ""),
                "pdf_start_page": entry.get("pdf_start_page"),
                "pdf_end_page": entry.get("pdf_end_page"),
            }
        )

    metadata = _load_json(Path(workspace) / "metadata.json", {}) or {}
    title = str(metadata.get("title") or os.path.basename(workspace)).strip()
    image_dirs = [
        path
        for path in (
            os.path.join(workspace, "images"),
            os.path.join(workspace, "translated_images"),
        )
        if os.path.isdir(path)
    ]
    css_dirs = []
    css_subdir = os.path.join(workspace, "css")
    if os.path.isdir(css_subdir):
        css_dirs.append(css_subdir)
    try:
        if any(
            item.is_file() and item.name.lower().endswith(".css")
            for item in os.scandir(workspace)
        ):
            css_dirs.append(workspace)
    except OSError:
        pass

    return {
        "workspace": workspace,
        "source_path": source,
        "source_format": source_format,
        "title": title,
        "entries": entries,
        "image_dirs": image_dirs,
        "css_dirs": css_dirs,
    }


def _section_cache_key(entry: dict) -> str:
    identity = str(entry.get("pdf_section_id") or entry.get("key") or "").strip()
    if not identity:
        identity = f"{entry.get('pdf_start_page')}-{entry.get('pdf_end_page')}"
    return hashlib.sha256(identity.encode("utf-8", "replace")).hexdigest()[:20]


def ensure_pdf_raw_section(
    manifest: Dict,
    entry: Dict,
    *,
    mode: str = "fast_semantic",
    extract_images: bool = True,
) -> str:
    """Return cached raw HTML for one PDF bookmark section.

    Only the requested page range is inspected/extracted.  The cache is tied
    to the source PDF's size and nanosecond mtime, while the page extractor
    also compares per-page source signatures before reusing any old artifacts.
    This means a replaced/extended PDF invalidates the section without forcing
    an eager walk across the rest of the document.
    """
    source = os.path.abspath(str(manifest.get("source_path") or ""))
    workspace = os.path.abspath(str(manifest.get("workspace") or ""))
    if not source.lower().endswith(".pdf") or not os.path.isfile(source):
        raise FileNotFoundError("The raw PDF source is not available.")
    if not os.path.isdir(workspace):
        raise FileNotFoundError("The translation workspace is not available.")

    try:
        start_page = int(entry.get("pdf_start_page"))
        end_page = int(entry.get("pdf_end_page"))
    except (TypeError, ValueError) as exc:
        raise ValueError("This PDF entry has no bookmark page range.") from exc
    if start_page < 1 or end_page < start_page:
        raise ValueError(f"Invalid PDF bookmark range: {start_page}-{end_page}")

    from pdf_fast_extractor import extract_pdf_page_range_for_reader

    if mode not in ("fast_semantic", "fast_layout"):
        mode = "fast_semantic"
    section_key = _section_cache_key(entry)
    cache_dir = Path(workspace) / ".pdf_reader_cache" / mode
    cache_dir.mkdir(parents=True, exist_ok=True)
    html_path = cache_dir / f"section_{section_key}.html"
    meta_path = cache_dir / f"section_{section_key}.json"
    source_stat = os.stat(source)
    expected = {
        "version": 2,
        "source_path": os.path.normcase(source),
        "source_size": int(source_stat.st_size),
        "source_mtime_ns": int(source_stat.st_mtime_ns),
        "start_page": start_page,
        "end_page": end_page,
        "title": str(entry.get("title") or ""),
        "mode": mode,
        "extract_images": bool(extract_images),
    }
    cached = _load_json(meta_path, {}) or {}
    if html_path.is_file() and all(cached.get(key) == value for key, value in expected.items()):
        return str(html_path)

    page_items = extract_pdf_page_range_for_reader(
        source,
        workspace,
        start_page=start_page,
        end_page=end_page,
        mode=mode,
        extract_images=extract_images,
        section_title=str(entry.get("title") or ""),
    )
    head_parts = []
    body_parts = []
    for page_number, page_html in page_items:
        head_match = re.search(r"<head[^>]*>(.*?)</head>", page_html or "", re.I | re.S)
        body_match = re.search(r"<body[^>]*>(.*?)</body>", page_html or "", re.I | re.S)
        if head_match and head_match.group(1) not in head_parts:
            head_parts.append(head_match.group(1))
        page_body = body_match.group(1) if body_match else str(page_html or "")
        body_parts.append(
            f'<section class="pdf-reader-source-page" data-pdf-page="{page_number}">'
            f"{page_body}</section>"
        )
    title = html.escape(str(entry.get("title") or "PDF section"), quote=True)
    document = (
        '<!DOCTYPE html><html><head><meta charset="utf-8">'
        f"<title>{title}</title>"
        + "\n".join(head_parts)
        + "</head><body>"
        + "\n".join(body_parts)
        + "</body></html>"
    )
    write_token = f"{os.getpid()}.{threading.get_ident()}"
    temporary = html_path.with_name(f"{html_path.name}.{write_token}.tmp")
    temporary.write_text(document, encoding="utf-8")
    os.replace(temporary, html_path)
    metadata = dict(expected)
    metadata["html_path"] = str(html_path)
    temporary_meta = meta_path.with_name(f"{meta_path.name}.{write_token}.tmp")
    temporary_meta.write_text(
        json.dumps(metadata, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    os.replace(temporary_meta, meta_path)
    return str(html_path)
