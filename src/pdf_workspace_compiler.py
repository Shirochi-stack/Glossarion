"""Compile translated response files in a PDF workspace into one PDF."""

from __future__ import annotations

import html
import json
import os
import re
import unicodedata
from typing import Callable

from bs4 import BeautifulSoup


LogCallback = Callable[[str], None]


def _log(callback: LogCallback | None, message: str) -> None:
    if callback:
        callback(message)
    else:
        try:
            print(message)
        except UnicodeEncodeError:
            print(message.encode("ascii", errors="backslashreplace").decode("ascii"))


def _natural_key(value: str) -> list:
    return [
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", str(value or ""))
    ]


def _numeric_order(value, fallback: int) -> tuple:
    try:
        return (0, float(value), fallback)
    except (TypeError, ValueError):
        return (1, fallback, fallback)


def _workspace_response_entries(folder: str) -> list[tuple[str, str]]:
    """Return existing translated response files as ``(path, title)``."""
    progress_path = os.path.join(folder, "translation_progress.json")
    entries: list[tuple[tuple, str, str]] = []
    seen: set[str] = set()
    try:
        with open(progress_path, "r", encoding="utf-8") as handle:
            progress = json.load(handle)
        chapters = progress.get("chapters", {}) if isinstance(progress, dict) else {}
    except (OSError, ValueError, TypeError):
        chapters = {}

    if isinstance(chapters, dict):
        for index, (key, info) in enumerate(chapters.items()):
            if not isinstance(info, dict):
                continue
            output_name = str(info.get("output_file") or "").strip()
            if not output_name.lower().endswith((".html", ".xhtml", ".htm")):
                continue
            output_path = (
                output_name
                if os.path.isabs(output_name)
                else os.path.join(folder, output_name)
            )
            if not os.path.isfile(output_path):
                continue
            normalized = os.path.normcase(os.path.normpath(output_path))
            if normalized in seen:
                continue
            seen.add(normalized)
            title = str(
                info.get("pdf_section_title")
                or info.get("pdf_toc_title")
                or info.get("title")
                or f"Section {info.get('actual_num') or key}"
            ).strip()
            entries.append((
                _numeric_order(info.get("actual_num", key), index),
                output_path,
                title,
            ))

    if entries:
        entries.sort(key=lambda item: item[0])
        return [(path, title) for _order, path, title in entries]

    fallback = []
    try:
        for entry in os.scandir(folder):
            if not entry.is_file(follow_symlinks=False):
                continue
            name = entry.name.lower()
            if name.startswith("response_") and name.endswith(
                (".html", ".xhtml", ".htm")
            ):
                fallback.append(entry.path)
    except (OSError, PermissionError):
        pass
    fallback.sort(key=lambda path: _natural_key(os.path.basename(path)))
    return [
        (path, f"Section {index}")
        for index, path in enumerate(fallback, 1)
    ]


def _fragment_body(content: str) -> str:
    soup = BeautifulSoup(content or "", "html.parser")
    container = soup.body if soup.body is not None else soup
    return "".join(str(child) for child in container.contents)


def _source_pdf_stem(folder: str) -> str:
    try:
        from output_workspace import read_workspace_source_path

        source = read_workspace_source_path(folder)
    except Exception:
        source = ""
    if source.lower().endswith(".pdf"):
        return os.path.splitext(os.path.basename(source))[0]
    leaf = os.path.basename(os.path.normpath(folder)) or "translated"
    return re.sub(r"_PDF(?:_\d+)?$", "", leaf, flags=re.IGNORECASE)


def _outline_title_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or ""))
    return " ".join(normalized.split()).casefold()


def _keep_only_section_bookmarks(pdf_path: str, titles: list[str]) -> None:
    """Normalize the generated outline to exactly one entry per response."""
    import fitz

    document = fitz.open(pdf_path)
    temp_path = f"{pdf_path}.outline.tmp"
    try:
        generated = document.get_toc(simple=True) or []
        cursor = 0
        cleaned = []
        for title in titles:
            wanted = _outline_title_key(title)
            matched_page = None
            for position in range(cursor, len(generated)):
                row = generated[position]
                if len(row) >= 3 and _outline_title_key(row[1]) == wanted:
                    matched_page = max(1, int(row[2]))
                    cursor = position + 1
                    break
            if matched_page is None:
                # WeasyPrint exposes the explicit bookmark anchors in its
                # outline. This fallback keeps the result valid if another
                # renderer dropped them, without reintroducing sentence-level
                # heading bookmarks.
                matched_page = cleaned[-1][2] if cleaned else 1
            cleaned.append([1, title, matched_page])
        document.set_toc(cleaned)
        document.save(temp_path, garbage=4, deflate=True)
    finally:
        document.close()
    os.replace(temp_path, pdf_path)


def compile_pdf_workspace(folder: str, log_callback: LogCallback | None = None) -> str:
    """Build a translated PDF from the current response HTML files."""
    if not folder or not os.path.isdir(folder):
        raise ValueError("PDF output workspace does not exist.")
    entries = _workspace_response_entries(folder)
    if not entries:
        raise ValueError("No translated response HTML files were found.")

    _log(log_callback, f"📄 Compiling PDF from {len(entries)} translated section(s)…")
    sections = []
    titles = []
    for index, (path, title) in enumerate(entries, 1):
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            content = handle.read()
        title = title or f"Section {index}"
        titles.append(title)
        sections.append(
            f'<section class="compiled-pdf-section" data-section="{index}">'
            f'<div id="pdf-section-{index}" class="pdf-bookmark-anchor">'
            f"{html.escape(title)}</div>"
            f"{_fragment_body(content)}</section>"
        )

    stem = _source_pdf_stem(folder)
    html_path = os.path.join(folder, f"{stem}_translated.html")
    pdf_path = os.path.join(folder, f"{stem}_translated.pdf")
    document_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{html.escape(stem)} - Translated</title>
  <link rel="stylesheet" href="styles.css">
  <style>
    h1, h2, h3, h4, h5, h6 {{ bookmark-level: none !important; }}
    .pdf-bookmark-anchor {{
      bookmark-level: 1 !important;
      bookmark-label: content(text);
      height: 0; margin: 0; padding: 0; overflow: hidden;
      color: transparent; font-size: 0; line-height: 0;
    }}
    .compiled-pdf-section {{ margin: 0; padding: 0; }}
  </style>
</head>
<body>
{''.join(sections)}
</body>
</html>"""
    with open(html_path, "w", encoding="utf-8") as handle:
        handle.write(document_html)

    from pdf_extractor import create_pdf_from_html

    css_path = os.path.join(folder, "styles.css")
    images_dir = os.path.join(folder, "images")
    success = create_pdf_from_html(
        document_html,
        pdf_path,
        css_path=css_path if os.path.isfile(css_path) else None,
        images_dir=images_dir if os.path.isdir(images_dir) else None,
    )
    if not success or not os.path.isfile(pdf_path):
        raise RuntimeError("The PDF renderer did not create an output file.")
    _keep_only_section_bookmarks(pdf_path, titles)
    _log(log_callback, f"✅ PDF compilation complete: {pdf_path}")
    return pdf_path
