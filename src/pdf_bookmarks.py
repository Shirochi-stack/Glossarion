"""Deterministic bookmark construction for rendered PDF chapter pages."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import re
from typing import Any


_PDF_SOURCE_PAGE_BREAK_RE = re.compile(
    r'<(?:div|span)\b[^>]*\bclass\s*=\s*["\'][^"\']*'
    r'\bpdf-toc-page-break\b[^"\']*["\'][^>]*>\s*'
    r'</(?:div|span)\s*>',
    flags=re.IGNORECASE,
)


def remove_pdf_source_page_break_markers(html: str) -> str:
    """Remove obsolete per-source-page breaks from bookmark-grouped HTML.

    Older TOC extraction output contains empty ``pdf-toc-page-break`` divs
    between every original PDF page. Keeping them when translated text is
    reflowed produces large blank regions in the compiled PDF. Section-level
    breaks are added separately by the compiler, so these inner markers are
    safe to remove without joining different bookmarks together.
    """
    return _PDF_SOURCE_PAGE_BREAK_RE.sub("", str(html or ""))


def replace_with_chapter_bookmarks(
    pages: Sequence[Any],
    chapters: Iterable[tuple[str, int, str]],
) -> int:
    """Keep exactly one bookmark for each titled source HTML chapter.

    WeasyPrint derives outlines from the CSS ``bookmark-level`` property.
    Imported EPUB styles can apply that property to arbitrary headings,
    paragraphs, or inline elements, so CSS suppression alone cannot guarantee
    a clean outline. Clear every derived bookmark after rendering and add one
    level-1 entry at each wrapper anchor inserted by the PDF compiler.

    ``chapters`` contains ``(source_filename, chapter_number, title)`` tuples.
    The filename is intentionally retained in the contract even though the
    rendered anchor is keyed by chapter number: it makes the one-entry-per-file
    invariant explicit and keeps callers from accidentally passing TOC rows or
    individual text nodes.
    """
    rendered_pages = list(pages)
    for page in rendered_pages:
        page.bookmarks[:] = []

    added = 0
    seen_files: set[str] = set()
    for source_filename, chapter_number, title in chapters:
        source_key = str(source_filename or "")
        if source_key in seen_files:
            continue
        seen_files.add(source_key)

        label = str(title or "").strip()
        if not label:
            continue
        anchor_name = f"chapter-{chapter_number}"
        for page in rendered_pages:
            target = page.anchors.get(anchor_name)
            if target is None:
                continue
            # WeasyPrint 68 stores anchor hit areas as (x1, y1, x2, y2),
            # while bookmark destinations remain points. Older releases used
            # (x, y) for anchors, so taking the leading pair supports both.
            destination = tuple(target[:2])
            if len(destination) != 2:
                continue
            page.bookmarks.append((1, label, destination, "open"))
            added += 1
            break
    return added
