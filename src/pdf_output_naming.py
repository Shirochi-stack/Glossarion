"""Safe output names for bookmark-grouped PDF sections.

PDF bookmark IDs are deliberately stable across outline insertions and belong in
``translation_progress.json``. Bookmark titles are display metadata and can be
arbitrarily long, so they do not belong in Windows filenames either. This module
keeps both values separate from short, ordinary output filenames.
"""

from __future__ import annotations

import os
import re
import unicodedata


_INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_WHITESPACE = re.compile(r"\s+")
_UNDERSCORES = re.compile(r"_+")
_WINDOWS_RESERVED = {
    "con", "prn", "aux", "nul",
    *(f"com{number}" for number in range(1, 10)),
    *(f"lpt{number}" for number in range(1, 10)),
}


def _safe_title(value, fallback="section", max_length=88):
    title = unicodedata.normalize("NFC", str(value or "").strip())
    title = _INVALID_FILENAME_CHARS.sub("_", title)
    title = _WHITESPACE.sub("_", title)
    title = _UNDERSCORES.sub("_", title).strip(" ._")
    if not title:
        title = fallback
    if title.casefold() in _WINDOWS_RESERVED:
        title = f"_{title}"
    title = title[:max_length].rstrip(" ._")
    return title or fallback


def _number_token(actual_num):
    try:
        numeric = float(actual_num)
    except (TypeError, ValueError):
        return _safe_title(actual_num, "000", max_length=24)
    major = int(numeric)
    fractional = int(round((numeric - major) * 1000))
    if fractional:
        return f"{major:03d}_{fractional:03d}"
    return f"{major:03d}"


def _truncate_utf16(value, max_units):
    """Truncate to a Windows filename-component budget (UTF-16 code units)."""
    result = []
    used = 0
    for character in str(value or ""):
        units = max(1, len(character.encode("utf-16-le")) // 2)
        if used + units > max_units:
            break
        result.append(character)
        used += units
    return "".join(result)


def safe_pdf_book_filename_stem(value, fallback="translated", max_units=180):
    """Return a Windows-safe, bounded book-title stem while preserving spaces."""
    title = unicodedata.normalize("NFC", str(value or "").strip())
    title = _INVALID_FILENAME_CHARS.sub("_", title)
    title = _WHITESPACE.sub(" ", title).strip(" .")
    if not title:
        title = fallback
    if title.casefold() in _WINDOWS_RESERVED:
        title = f"_{title}"
    title = _truncate_utf16(title, max(24, int(max_units))).rstrip(" .")
    return title or fallback


def readable_pdf_section_filename(chapter, actual_num=None, retain=False):
    """Return a short numbered filename without hashes or bookmark titles."""
    chapter = chapter if isinstance(chapter, dict) else {}
    mapped = str(chapter.get("_pdf_mapped_output_file") or "").strip()
    if mapped:
        return os.path.basename(mapped)

    if actual_num is None:
        actual_num = chapter.get("actual_chapter_num", chapter.get("num", 0))
    stem = f"pdf_section_{_number_token(actual_num)}"

    if chapter.get("is_chunk"):
        chunk_info = chapter.get("chunk_info") or {}
        chunk_index = chunk_info.get("chunk_idx")
        if chunk_index is not None:
            stem = f"{stem}_part_{chunk_index}"

    return f"{stem}.html" if retain else f"response_{stem}.html"


def allocate_readable_pdf_filename(output_dir, preferred, occupied=(), current=""):
    """Choose a readable, non-destructive filename, using numeric suffixes."""
    preferred = os.path.basename(str(preferred or ""))
    current = os.path.basename(str(current or ""))
    occupied_names = {
        os.path.basename(str(name)).casefold()
        for name in occupied
        if str(name or "").strip()
    }
    occupied_names.discard(current.casefold())

    stem, extension = os.path.splitext(preferred)
    candidate = preferred
    suffix = 2
    while True:
        candidate_path = os.path.join(output_dir, candidate)
        collision = candidate.casefold() in occupied_names
        if os.path.isfile(candidate_path):
            collision = os.path.normcase(os.path.abspath(candidate_path)) != os.path.normcase(
                os.path.abspath(os.path.join(output_dir, current))
            )
        if not collision:
            return candidate
        candidate = f"{stem}_{suffix}{extension}"
        suffix += 1


def move_pdf_output_to_readable_name(output_dir, current, preferred, occupied=()):
    """Rename an existing output without overwriting anything.

    Returns ``(filename, moved)``.  Missing files still receive the preferred
    mapping so the next translation writes a readable name.
    """
    current = os.path.basename(str(current or ""))
    preferred = os.path.basename(str(preferred or ""))
    if not preferred:
        return current, False
    destination = allocate_readable_pdf_filename(
        output_dir,
        preferred,
        occupied=occupied,
        current=current,
    )
    if current.casefold() == destination.casefold():
        return current or destination, False

    source_path = os.path.join(output_dir, current) if current else ""
    destination_path = os.path.join(output_dir, destination)
    if source_path and os.path.isfile(source_path):
        os.replace(source_path, destination_path)
        return destination, True
    return destination, False
