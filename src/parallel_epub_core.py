"""parallel_epub_core: the Parallel EPUB Pair mapper without Qt (Glossarion mobile rewrite, U6).

Shared by the desktop (``ParallelEpubPairDialog`` in parallel_epub_glossary.py and the pair
handlers of ``TranslatorGUI``) and by the mobile Parallel EPUB pair screen and its glossary
job. Everything here moved verbatim (frozen source: ``git show U6_BASE_SHA``, see
tests/test_parallel_epub_core.py, which pins every block and its documented edits):

* parallel_epub_glossary.py 52-512: the constants, the chapter accessors, compact / restore
  of a saved selection, auto-mapping, the wrapper prompt and the paired-EPUB writer. They
  are unchanged; parallel_epub_glossary re-exports every name.
* The pure halves of ``ParallelEpubPairDialog`` methods: the background loader
  (``_start_epub_load``), restoring a saved selection (``restore_persisted_selection``,
  ``_apply_pending_persisted_mapping``), the offset stepper (``_apply_mapping_offset``),
  bulk unmapping (``_set_rows_unmapped``), the selected mapping and its status / unpaired
  counts / warning (``_selected_mapping``, ``_update_mapping_status``,
  ``_unpaired_file_counts``, ``_unpaired_warning_text``), the Use Mapped Pair checks and
  pairs (``_accept_pair``) and the prompt profiles (``__init__``,
  ``_persist_prompt_settings``). The dialog keeps its widgets and calls these.
* ``TranslatorGUI`` pair helpers, with ``config`` in place of ``self.config``: the chapter
  loader (``_load_parallel_epub_chapters``), the disposable working EPUB
  (``_build_parallel_epub_pair_artifact``), the raw book's glossary folder and the mapping
  sidecar (``_resolve_parallel_epub_glossary_output_dir``,
  ``_parallel_epub_mapping_sidecar_path``, ``_write/_read_parallel_epub_mapping_sidecar``),
  the record of an activated pair (``_activate_parallel_epub_pair_source``) and the rebuild
  of a saved pair (``_start_parallel_epub_pair_restore``'s worker).

Python 3.10 compatible; never imports PySide6, translator_gui or dpi_setup. ebooklib is
imported at module level, as parallel_epub_glossary did; extract_glossary_from_epub only
inside the functions that need it.
"""

from __future__ import annotations

import html
import json
import os
import re
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

from ebooklib import epub
from epub_special_files import special_file_flags

__all__ = [
    "DEFAULT_PARALLEL_EPUB_PROFILE",
    "DEFAULT_PARALLEL_EPUB_WRAPPER_PROMPT",
    "PARALLEL_EPUB_SELECTION_CONFIG_KEY",
    "PARALLEL_EPUB_SYSTEM_INSTRUCTIONS",
    "active_parallel_epub_profile",
    "apply_parallel_epub_wrapper",
    "auto_map_epub_chapters",
    "build_parallel_epub_pair_artifact",
    "build_parallel_epub_pairs",
    "chapter_filename",
    "chapter_special_flags",
    "chapter_text",
    "compact_parallel_epub_selection",
    "default_parallel_epub_system_prompt",
    "load_parallel_epub_chapters",
    "load_parallel_epub_documents",
    "offset_parallel_epub_mapping",
    "parallel_epub_mapping_sidecar_path",
    "parallel_epub_mapping_status",
    "parallel_epub_pair_source_state",
    "parallel_epub_profiles",
    "parallel_epub_prompt_settings",
    "parallel_epub_selection_matches",
    "parallel_epub_working_filename",
    "persisted_parallel_epub_rows",
    "prepare_persisted_parallel_epub_selection",
    "read_parallel_epub_mapping_sidecar",
    "rebuild_parallel_epub_pair_result",
    "render_parallel_epub_wrapper",
    "resolve_parallel_epub_glossary_output_dir",
    "restore_parallel_epub_pairs",
    "selected_parallel_epub_mapping",
    "translated_mapping_label",
    "unpaired_file_counts",
    "unpaired_warning_text",
    "valid_parallel_epub_rows",
    "validate_parallel_epub_pair",
    "write_parallel_epub",
    "write_parallel_epub_mapping_sidecar",
]


DEFAULT_PARALLEL_EPUB_PROFILE = "Parallel EPUB Glossary"
PARALLEL_EPUB_SELECTION_CONFIG_KEY = "parallel_epub_pair_selection"

DEFAULT_PARALLEL_EPUB_WRAPPER_PROMPT = """\
[RAW EPUB START — {raw_filename}]
{raw_text}
[RAW EPUB END]

[TRANSLATED EPUB START — {translated_filename}]
{translated_text}
[TRANSLATED EPUB END]"""

PARALLEL_EPUB_SYSTEM_INSTRUCTIONS = """\
PAIR-SPECIFIC INSTRUCTIONS:
- You are cross-checking aligned HTML chapters from a raw/source-language EPUB and an existing translated EPUB.
- Every user input contains one or more mapped raw/translated chapter pairs. Treat each RAW EPUB section as the authority for raw_name and its matching TRANSLATED EPUB section as the authority for established translated_name spellings.
- Cross-check both sections before creating each entry. Only output an entry when its matching established rendering is present in the translated section, and copy that rendering exactly. If the translated section does not provide a verifiable matching rendering, skip the entry entirely; never invent one yourself.
- Use the paired context to recover entries that one edition makes implicit, but never invent an entry or translation unsupported by either section.
- The pair-specific rules above take priority if the general glossary rules below would otherwise make you ignore the supplied translated edition."""


def parallel_epub_working_filename(raw_path: str) -> str:
    """Keep the raw EPUB basename so glossary output uses its normal folder."""

    source_name = os.path.basename(str(raw_path or "").strip())
    stem, extension = os.path.splitext(source_name)
    if not stem:
        stem = "raw_epub"
    if extension.lower() != ".epub":
        return f"{stem}.epub"
    return source_name


def default_parallel_epub_system_prompt() -> str:
    """Return pair instructions followed by the canonical prompt verbatim."""
    from extract_glossary_from_epub import DEFAULT_GLOSSARY_PROMPT

    return f"{PARALLEL_EPUB_SYSTEM_INSTRUCTIONS}\n\n{DEFAULT_GLOSSARY_PROMPT}"


def chapter_filename(chapter) -> str:
    """Read a chapter filename from extractor tuples or dialog dictionaries."""
    if isinstance(chapter, dict):
        return str(chapter.get("filename") or "")
    if isinstance(chapter, (tuple, list)) and len(chapter) >= 2:
        return str(chapter[1] or "")
    return ""


def chapter_text(chapter) -> str:
    """Read chapter text from extractor tuples or dialog dictionaries."""
    if isinstance(chapter, dict):
        return str(chapter.get("text") or "")
    if isinstance(chapter, (tuple, list)) and chapter:
        return str(chapter[0] or "")
    return str(chapter or "")


def compact_parallel_epub_selection(result: dict) -> dict:
    """Return the persistent, text-free representation of a mapped pair."""

    mappings = []
    for ordinal, pair in enumerate(result.get("pairs") or []):
        if not isinstance(pair, dict):
            continue
        raw_filename = str(pair.get("raw_filename") or "")
        translated_filename = str(pair.get("translated_filename") or "")
        if not raw_filename or not translated_filename:
            continue
        try:
            raw_index = int(pair.get("raw_index", ordinal))
        except (TypeError, ValueError):
            raw_index = ordinal
        try:
            translated_index = int(pair.get("translated_index", ordinal))
        except (TypeError, ValueError):
            translated_index = ordinal
        mappings.append(
            {
                "raw_index": raw_index,
                "translated_index": translated_index,
                "raw_filename": raw_filename,
                "translated_filename": translated_filename,
            }
        )

    return {
        "version": 1,
        "raw_path": os.path.abspath(str(result.get("raw_path") or "")),
        "translated_path": os.path.abspath(
            str(result.get("translated_path") or "")
        ),
        "mapping": mappings,
        "wrapper_prompt": str(result.get("wrapper_prompt") or ""),
        "system_prompt": str(result.get("system_prompt") or ""),
        "profile_name": str(
            result.get("profile_name") or DEFAULT_PARALLEL_EPUB_PROFILE
        ),
    }


def restore_parallel_epub_pairs(
    raw_chapters: Sequence,
    translated_chapters: Sequence,
    stored_mapping: Sequence,
) -> tuple[List[Dict[str, object]], int]:
    """Reattach saved filename mappings to freshly extracted chapter text.

    Stored indexes are used only when the filename at that index still agrees.
    Filename lookup is the fallback, so a harmless EPUB reading-order change does
    not destroy the saved mapping. Missing or duplicate references are skipped
    and returned as the second value.
    """

    raw_used = set()
    translated_used = set()

    def resolve_index(chapters, saved_index, saved_filename, used):
        expected = str(saved_filename or "")
        expected_key = expected.replace("\\", "/").casefold()
        try:
            candidate = int(saved_index)
        except (TypeError, ValueError):
            candidate = -1
        if (
            0 <= candidate < len(chapters)
            and candidate not in used
            and chapter_filename(chapters[candidate])
            .replace("\\", "/")
            .casefold()
            == expected_key
        ):
            return candidate
        for index, chapter in enumerate(chapters):
            if index in used:
                continue
            if (
                chapter_filename(chapter).replace("\\", "/").casefold()
                == expected_key
            ):
                return index
        return None

    restored = []
    skipped = 0
    for entry in stored_mapping or []:
        if not isinstance(entry, dict):
            skipped += 1
            continue
        raw_index = resolve_index(
            raw_chapters,
            entry.get("raw_index"),
            entry.get("raw_filename"),
            raw_used,
        )
        translated_index = resolve_index(
            translated_chapters,
            entry.get("translated_index"),
            entry.get("translated_filename"),
            translated_used,
        )
        if raw_index is None or translated_index is None:
            skipped += 1
            continue
        raw_used.add(raw_index)
        translated_used.add(translated_index)
        restored.append(
            {
                "raw_index": raw_index,
                "translated_index": translated_index,
                "raw_filename": chapter_filename(raw_chapters[raw_index]),
                "raw_text": chapter_text(raw_chapters[raw_index]),
                "translated_filename": chapter_filename(
                    translated_chapters[translated_index]
                ),
                "translated_text": chapter_text(
                    translated_chapters[translated_index]
                ),
            }
        )
    return restored, skipped


def _normalized_member_stem(filename: str) -> str:
    stem = Path(str(filename or "")).stem.casefold()
    return re.sub(r"[^a-z0-9]+", "", stem)


def _member_number_signature(filename: str) -> tuple:
    numbers = tuple(
        int(part) for part in re.findall(r"\d+", Path(str(filename or "")).stem)
    )
    # Zero-only names such as 0000_Information are front-matter offset
    # candidates, not chapter-number anchors.
    return numbers if any(number > 0 for number in numbers) else ()


def _has_positive_member_number(filename: str) -> bool:
    """Return whether a filename contains any numeric value greater than zero."""

    return bool(_member_number_signature(filename))


def _nonpositive_member_layout(chapters: Sequence) -> tuple:
    """Describe where no-number/zero-only files occur among numbered files."""

    positive_members_seen = 0
    layout = []
    for chapter in chapters:
        if _has_positive_member_number(chapter_filename(chapter)):
            positive_members_seen += 1
        else:
            layout.append(positive_members_seen)
    return tuple(layout)


def _chapter_special_flags(
    chapters: Sequence,
    predicate: Callable[[str], bool],
    *,
    protect_interior: bool = False,
    reading_order: Optional[Sequence[str]] = None,
) -> List[bool]:
    """Keep textless EPUB documents in the special-file boundary context."""
    filenames = [chapter_filename(chapter) for chapter in chapters]
    if not protect_interior or not reading_order:
        return special_file_flags(
            filenames, predicate, protect_interior=protect_interior,
        )
    ordered_flags = special_file_flags(
        reading_order, predicate, protect_interior=True,
    )
    by_filename = dict(zip(reading_order, ordered_flags))
    return [
        by_filename[filename] if filename in by_filename else bool(predicate(filename))
        for filename in filenames
    ]


def auto_map_epub_chapters(
    raw_chapters: Sequence,
    translated_chapters: Sequence,
    *,
    enable_auto_offset: bool = True,
    special_file_predicate: Optional[Callable[[str], bool]] = None,
    protect_interior_special_files: bool = True,
    raw_reading_order: Optional[Sequence[str]] = None,
    translated_reading_order: Optional[Sequence[str]] = None,
) -> List[Dict[str, object]]:
    """Map names/numbers, leaving special files available for manual pairing."""
    mappings: List[Dict[str, object]] = [
        {
            "raw_index": index,
            "translated_index": None,
            "strategy": "Unmatched",
            "auto_offset": 0,
        }
        for index in range(len(raw_chapters))
    ]
    available = set(range(len(translated_chapters)))

    def assign_unique(key_func: Callable[[str], object], strategy: str) -> None:
        raw_keys: Dict[object, List[int]] = {}
        translated_keys: Dict[object, List[int]] = {}
        for raw_index, mapping in enumerate(mappings):
            if mapping["translated_index"] is not None:
                continue
            key = key_func(chapter_filename(raw_chapters[raw_index]))
            if key:
                raw_keys.setdefault(key, []).append(raw_index)
        for translated_index in sorted(available):
            key = key_func(chapter_filename(translated_chapters[translated_index]))
            if key:
                translated_keys.setdefault(key, []).append(translated_index)
        for key, raw_indexes in raw_keys.items():
            translated_indexes = translated_keys.get(key, [])
            if len(raw_indexes) != 1 or len(translated_indexes) != 1:
                continue
            raw_index = raw_indexes[0]
            translated_index = translated_indexes[0]
            mappings[raw_index]["translated_index"] = translated_index
            mappings[raw_index]["strategy"] = strategy
            available.discard(translated_index)

    if enable_auto_offset:
        # No-number and zero-only documents deliberately stay out of every
        # automatic assignment, even when their stems match. They remain in
        # the UI for explicit manual selection.
        assign_unique(
            lambda filename: (
                _normalized_member_stem(filename)
                if _has_positive_member_number(filename)
                else ""
            ),
            "Exact filename",
        )
    else:
        assign_unique(_normalized_member_stem, "Exact filename")

    def assign_reading_group(numbered: bool, strategy: str) -> None:
        raw_indexes = [
            index
            for index, mapping in enumerate(mappings)
            if mapping["translated_index"] is None
            and _has_positive_member_number(chapter_filename(raw_chapters[index]))
            is numbered
        ]
        translated_indexes = [
            index
            for index in sorted(available)
            if _has_positive_member_number(
                chapter_filename(translated_chapters[index])
            )
            is numbered
        ]
        for raw_index, translated_index in zip(raw_indexes, translated_indexes):
            visual_offset = raw_index - translated_index if numbered else 0
            mappings[raw_index]["translated_index"] = translated_index
            mappings[raw_index]["auto_offset"] = visual_offset
            mappings[raw_index]["strategy"] = (
                f"Auto offset {visual_offset:+d}" if visual_offset else strategy
            )
            available.discard(translated_index)

    if enable_auto_offset:
        # When zero-only/unnumbered files occur at different positions, they
        # are the offset signal. Align positive-numbered reading sequences
        # first; raw_0002 can legitimately correspond to translated_0001.
        has_nonpositive_offset = _nonpositive_member_layout(
            raw_chapters
        ) != _nonpositive_member_layout(translated_chapters)
        if has_nonpositive_offset:
            assign_reading_group(True, "Numbered order")
        assign_unique(_member_number_signature, "Chapter number")
        if not has_nonpositive_offset:
            assign_reading_group(True, "Numbered order")

        # No-number/zero-only raw rows stay Unmatched and their translated
        # counterparts stay unused. This prevents front matter from entering
        # the paired glossary unless the user selects it manually.
    else:
        assign_unique(_member_number_signature, "Chapter number")
        unmatched_raw = [
            index
            for index, mapping in enumerate(mappings)
            if mapping["translated_index"] is None
        ]
        for raw_index, translated_index in zip(unmatched_raw, sorted(available)):
            mappings[raw_index]["translated_index"] = translated_index
            mappings[raw_index]["strategy"] = "Reading order"
            available.discard(translated_index)

    if special_file_predicate:
        raw_special = _chapter_special_flags(
            raw_chapters,
            special_file_predicate, protect_interior=protect_interior_special_files,
            reading_order=raw_reading_order,
        )
        translated_special = _chapter_special_flags(
            translated_chapters,
            special_file_predicate, protect_interior=protect_interior_special_files,
            reading_order=translated_reading_order,
        )
        # Keep special files in the candidate sequences until alignment is
        # complete. Removing one first would shift every following positional
        # pair (for example, raw 56 would receive translated 57).
        for mapping in mappings:
            raw_index = mapping["raw_index"]
            translated_index = mapping["translated_index"]
            if raw_special[raw_index] or (
                translated_index is not None
                and translated_special[translated_index]
            ):
                mapping["translated_index"] = None
                mapping["strategy"] = "Special file — Unmapped"
                mapping["auto_offset"] = 0

    return mappings


def apply_parallel_epub_wrapper(
    template: str,
    *,
    raw_text: str,
    translated_text: str,
    raw_filename: str,
    translated_filename: str,
) -> str:
    """Expand only supported placeholders, leaving unrelated braces intact."""
    result = str(template or "")
    replacements = {
        "{raw_text}": str(raw_text or ""),
        "{translated_text}": str(translated_text or ""),
        "{raw_filename}": str(raw_filename or ""),
        "{translated_filename}": str(translated_filename or ""),
    }
    for placeholder, value in replacements.items():
        result = result.replace(placeholder, value)
    return result


def write_parallel_epub(
    output_path: str,
    pairs: Iterable[Dict[str, str]],
    wrapper_prompt: str,
    *,
    title: str = "Parallel EPUB Pair",
) -> str:
    """Write mapped chapter pairs as one valid EPUB for the shared extractor."""
    pair_list = list(pairs)
    if not pair_list:
        raise ValueError("At least one mapped HTML pair is required.")
    if "{raw_text}" not in wrapper_prompt or "{translated_text}" not in wrapper_prompt:
        raise ValueError(
            "The wrapper prompt must contain {raw_text} and {translated_text}."
        )

    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    book = epub.EpubBook()
    book.set_identifier(f"glossarion-parallel-{uuid.uuid4().hex}")
    book.set_title(str(title or "Parallel EPUB Pair"))
    book.set_language("und")

    epub_chapters = []
    for index, pair in enumerate(pair_list, start=1):
        wrapped = apply_parallel_epub_wrapper(
            wrapper_prompt,
            raw_text=pair.get("raw_text", ""),
            translated_text=pair.get("translated_text", ""),
            raw_filename=pair.get("raw_filename", ""),
            translated_filename=pair.get("translated_filename", ""),
        )
        # A preformatted element preserves wrapper boundaries while escaping any
        # markup that appeared in source prose. The established EPUB extractor
        # will turn it back into plain text before sending it to the model.
        content = (
            "<html xmlns=\"http://www.w3.org/1999/xhtml\"><head>"
            f"<title>Mapped pair {index}</title></head><body>"
            f"<pre style=\"white-space: pre-wrap\">{html.escape(wrapped)}</pre>"
            "</body></html>"
        )
        chapter = epub.EpubHtml(
            title=f"Mapped pair {index}",
            file_name=f"pair_{index:04d}.xhtml",
            lang="und",
        )
        chapter.content = content
        book.add_item(chapter)
        epub_chapters.append(chapter)

    book.toc = tuple(epub_chapters)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    # Keep the required EPUB navigation document out of the reading spine so a
    # user who enables "translate special files" still sends only mapped pairs
    # through glossary extraction.
    book.spine = list(epub_chapters)
    epub.write_epub(output_path, book, {})
    return output_path


#: Public spellings of two moved names (the originals keep working).
chapter_special_flags = _chapter_special_flags
render_parallel_epub_wrapper = apply_parallel_epub_wrapper


# ---------------------------------------------------------------------------
# ParallelEpubPairDialog: the pure halves of its methods
# ---------------------------------------------------------------------------


def load_parallel_epub_documents(chapter_loader, path):
    """Read one side of a pair like the dialog's background loader (``_start_epub_load``).

    Returns ``(chapters, reading_order, error)``: the documents with readable text as
    ``{"text", "filename"}``, every document's filename in reading order (textless ones
    included, for the special-file boundary context), and the error text the dialog shows
    ("" on success).
    """
    chapters = []
    reading_order = []
    error = ""
    try:
        extracted = list(chapter_loader(path) or [])
        for index, item in enumerate(extracted, start=1):
            filename = chapter_filename(item) or f"HTML {index}"
            reading_order.append(filename)
            text = chapter_text(item)
            if not text.strip():
                continue
            chapters.append(
                {
                    "text": text,
                    "filename": filename,
                }
            )
        if not chapters:
            error = "The EPUB has no eligible HTML files with readable text."
    except Exception as exc:
        error = str(exc)
    return chapters, reading_order, error


def prepare_persisted_parallel_epub_selection(selection):
    """A saved selection ready to restore, or None (``restore_persisted_selection``).

    Both saved paths must still be existing ``.epub`` files; the result has absolute paths
    and only the dict entries of the saved mapping.
    """
    if not isinstance(selection, dict) or not selection.get("mapping"):
        return None
    raw_path = os.path.abspath(str(selection.get("raw_path") or ""))
    translated_path = os.path.abspath(
        str(selection.get("translated_path") or "")
    )
    if not (
        os.path.isfile(raw_path)
        and raw_path.lower().endswith(".epub")
        and os.path.isfile(translated_path)
        and translated_path.lower().endswith(".epub")
    ):
        return None

    return {
        **selection,
        "raw_path": raw_path,
        "translated_path": translated_path,
        "mapping": [
            dict(item)
            for item in selection.get("mapping") or []
            if isinstance(item, dict)
        ],
    }


def parallel_epub_selection_matches(selection, raw_path, translated_path):
    """Whether a saved selection belongs to the loaded EPUB pair (``_apply_pending_persisted_mapping``)."""
    current_paths = (
        os.path.normcase(os.path.abspath(raw_path)),
        os.path.normcase(os.path.abspath(translated_path)),
    )
    saved_paths = (
        os.path.normcase(os.path.abspath(str(selection.get("raw_path") or ""))),
        os.path.normcase(
            os.path.abspath(str(selection.get("translated_path") or ""))
        ),
    )
    return current_paths == saved_paths


def persisted_parallel_epub_rows(row_count, restored):
    """Each raw row's ``(translated index or -1, Match text)`` after restoring a saved mapping.

    ``_apply_pending_persisted_mapping``: missing entries in the compact mapping are
    intentional unmapped rows, so every row starts as "Saved — Unmapped" and the restored
    pairs (:func:`restore_parallel_epub_pairs`) become "Saved Mapping".
    """
    rows = [(-1, "Saved — Unmapped") for _row in range(row_count)]
    for pair in restored:
        row = int(pair["raw_index"])
        translated_index = int(pair["translated_index"])
        if 0 <= row < row_count:
            rows[row] = (translated_index, "Saved Mapping")
    return rows


def translated_mapping_label(translated_chapters, translated_index):
    """The Translated HTML cell text of a translated index (``_translated_mapping_label``)."""
    if 0 <= translated_index < len(translated_chapters):
        return str(translated_chapters[translated_index]["filename"])
    return "— Unmapped —"


def offset_parallel_epub_mapping(
    auto_mapping,
    offset,
    translated_chapters,
    special_file_predicate=None,
    *,
    protect_interior=True,
    reading_order=None,
):
    """Each raw row's ``(translated index or -1, Match text)`` under a cumulative offset.

    ``_apply_mapping_offset``: the automatic assignments shifted by ``offset`` (the dialog's
    running ``_mapping_offset``); indexes past either end and special translated files
    become unmapped. With offset 0 the automatic Match text comes back.
    """
    translated_count = len(translated_chapters)
    translated_special = _chapter_special_flags(
        translated_chapters,
        special_file_predicate or (lambda _filename: False),
        protect_interior=bool(protect_interior),
        reading_order=reading_order,
    )
    # Offset direction follows what the user sees in the raw-row table:
    # +1 moves the existing assignments down one raw row, so each row must
    # select the translated index that was previously one row above it.
    rows = []
    for automatic in auto_mapping:
        base_index = automatic.get("translated_index")
        shifted_index = (
            None
            if base_index is None
            else int(base_index) - offset
        )
        if shifted_index is None or not 0 <= shifted_index < translated_count:
            translated_index = -1
            strategy = f"Offset {offset:+d} (unmapped)"
        elif translated_special[shifted_index]:
            translated_index = -1
            strategy = "Special file — Unmapped"
        else:
            translated_index = shifted_index
            strategy = f"Offset {offset:+d}"
        if not offset:
            strategy = str(automatic.get("strategy") or "Unmatched")
        rows.append((translated_index, strategy))
    return rows


def valid_parallel_epub_rows(rows, row_count):
    """The distinct in-range rows of a bulk "Set as Unmapped", ascending (``_set_rows_unmapped``)."""
    return sorted(
        {
            int(row)
            for row in rows
            if 0 <= int(row) < row_count
        }
    )


def selected_parallel_epub_mapping(translated_indexes):
    """``[{"raw_index", "translated_index"}]`` of the mapped rows (``_selected_mapping``).

    ``translated_indexes`` holds each raw row's translated index in row order; a value that
    is not an integer, or is negative, means the row is unmapped.
    """
    selected = []
    for row, translated_index in enumerate(translated_indexes):
        try:
            translated_index = int(translated_index)
        except (TypeError, ValueError):
            translated_index = -1
        if translated_index >= 0:
            selected.append({"raw_index": row, "translated_index": translated_index})
    return selected


def unpaired_file_counts(mapping, raw_count, translated_count):
    """Return unmatched raw and unused translated document counts."""

    used_translated = {item["translated_index"] for item in mapping}
    unmatched_raw = raw_count - len(mapping)
    unused_translated = translated_count - len(used_translated)
    return unmatched_raw, unused_translated


def unpaired_warning_text(mapping, raw_count, translated_count):
    """Explain every individual HTML document excluded from the pair."""

    unmatched_raw, unused_translated = unpaired_file_counts(
        mapping, raw_count, translated_count
    )
    excluded_total = unmatched_raw + unused_translated
    return (
        f"{excluded_total} HTML file(s) are not part of a mapped pair and "
        "will be skipped.\n\n"
        f"Mapped raw/translated pairs: {len(mapping)}\n"
        f"Unmatched raw HTML files: {unmatched_raw}\n"
        f"Unused translated HTML files: {unused_translated}\n\n"
        "Unused translated files are the extra files that overflow beyond "
        "the available raw rows, or files that no raw row currently selects.\n\n"
        "Continue with only the mapped pairs?"
    )


def parallel_epub_mapping_status(mapping, offset, auto_mapping, raw_count, translated_count):
    """``(status text, duplicate assignment count)`` of the mapping status line (``_update_mapping_status``)."""
    used = [item["translated_index"] for item in mapping]
    duplicate_count = len(used) - len(set(used))
    unmatched_raw, unused_translated = unpaired_file_counts(
        mapping, raw_count, translated_count
    )
    parts = [f"{len(mapping)} mapped"]
    if offset:
        parts.append(f"offset {offset:+d}")
    else:
        automatic_offsets = sorted(
            {
                int(item.get("auto_offset") or 0)
                for item in auto_mapping
                if int(item.get("auto_offset") or 0)
            }
        )
        if len(automatic_offsets) == 1:
            parts.append(f"auto offset {automatic_offsets[0]:+d}")
        elif automatic_offsets:
            parts.append("automatic numbering offsets")
    if unmatched_raw:
        parts.append(f"{unmatched_raw} raw unmatched")
    if unused_translated:
        parts.append(f"{unused_translated} translated unused")
    if duplicate_count:
        parts.append(f"{duplicate_count} duplicate assignment(s)")
    status_text = " • ".join(parts)
    return status_text, duplicate_count


def validate_parallel_epub_pair(
    *,
    loading,
    raw_path,
    translated_path,
    raw_chapters,
    translated_chapters,
    wrapper_prompt,
    system_prompt,
    mapping,
):
    """Why "Use Mapped Pair" cannot go ahead, or None (``_accept_pair``'s checks, in order).

    Returns ``(kind, title, text)`` where kind is ``"information"`` or ``"warning"`` (the
    dialog's message box). ``system_prompt`` is the stripped prompt text and ``mapping`` the
    selected rows. The unpaired-files confirmation comes after these checks
    (:func:`unpaired_file_counts`, :func:`unpaired_warning_text`).
    """
    if loading:
        return (
            "information",
            "EPUB Still Loading",
            "Wait for both EPUBs to finish loading before using the mapped pair.",
        )
    if not raw_chapters or not translated_chapters:
        return ("warning", "EPUBs Required", "Load both the raw and translated EPUB.")
    if os.path.normcase(os.path.abspath(raw_path)) == os.path.normcase(
        os.path.abspath(translated_path)
    ):
        return (
            "warning",
            "Two EPUBs Required",
            "Choose the source EPUB on the left and its translated edition on the right.",
        )
    wrapper = wrapper_prompt
    if "{raw_text}" not in wrapper or "{translated_text}" not in wrapper:
        return (
            "warning",
            "Wrapper Placeholders Required",
            "The wrapper prompt must contain both {raw_text} and {translated_text}.",
        )
    if not system_prompt:
        return ("warning", "System Prompt Required", "The system prompt cannot be empty.")
    if not mapping:
        return ("warning", "Mapping Required", "Map at least one HTML file pair.")
    translated_indexes = [item["translated_index"] for item in mapping]
    if len(translated_indexes) != len(set(translated_indexes)):
        return (
            "warning",
            "Duplicate Mapping",
            "Each translated HTML file can only be assigned once.",
        )
    return None


def build_parallel_epub_pairs(mapping, raw_chapters, translated_chapters):
    """The accepted pairs with their chapter text (``_accept_pair``)."""
    pairs = []
    for item in mapping:
        raw = raw_chapters[item["raw_index"]]
        translated = translated_chapters[item["translated_index"]]
        pairs.append(
            {
                "raw_index": item["raw_index"],
                "translated_index": item["translated_index"],
                "raw_filename": raw["filename"],
                "raw_text": raw["text"],
                "translated_filename": translated["filename"],
                "translated_text": translated["text"],
            }
        )
    return pairs


def parallel_epub_profiles(config):
    """The system-prompt profiles the dialog starts with: the saved ones plus the built-in default."""
    saved_profiles = config.get("parallel_epub_glossary_profiles", {})
    profiles = dict(saved_profiles) if isinstance(saved_profiles, dict) else {}
    default_prompt = default_parallel_epub_system_prompt()
    if DEFAULT_PARALLEL_EPUB_PROFILE not in profiles:
        profiles[DEFAULT_PARALLEL_EPUB_PROFILE] = default_prompt
    return profiles


def active_parallel_epub_profile(config, profiles):
    """The profile the dialog selects on open: the saved active one while it still exists."""
    active = str(
        config.get("parallel_epub_glossary_active_profile")
        or DEFAULT_PARALLEL_EPUB_PROFILE
    )
    if active not in profiles:
        active = DEFAULT_PARALLEL_EPUB_PROFILE
    return active


def parallel_epub_prompt_settings(profiles, active_profile, wrapper_prompt):
    """The config values ``_persist_prompt_settings`` writes, in its order."""
    return {
        "parallel_epub_glossary_profiles": dict(profiles),
        "parallel_epub_glossary_active_profile": (
            str(active_profile or "").strip() or DEFAULT_PARALLEL_EPUB_PROFILE
        ),
        "parallel_epub_glossary_wrapper_prompt": wrapper_prompt,
    }


# ---------------------------------------------------------------------------
# TranslatorGUI pair helpers (``config`` in place of ``self.config``)
# ---------------------------------------------------------------------------


def load_parallel_epub_chapters(epub_path):
    """Keep all EPUB documents for pairing and reading-order boundaries."""
    from extract_glossary_from_epub import extract_chapters_from_epub

    # Special-file rules only veto automatic pairs in the dialog. Applying
    # them during extraction would hide files from manual selection and
    # prevent saved manual mappings from being restored.
    return extract_chapters_from_epub(
        epub_path, return_document_metadata=True, include_special_files=True,
    )


def build_parallel_epub_pair_artifact(result):
    """Build the disposable working EPUB without changing GUI state.

    Returns ``(TemporaryDirectory, working EPUB path)``; the caller owns the folder.
    """

    raw_path = str(result.get("raw_path") or "")
    raw_stem = os.path.splitext(os.path.basename(raw_path))[0]
    pair_temp_dir = tempfile.TemporaryDirectory(
        prefix="glossarion_parallel_epub_"
    )
    generated_path = os.path.join(
        pair_temp_dir.name,
        parallel_epub_working_filename(raw_path),
    )
    try:
        write_parallel_epub(
            generated_path,
            result.get("pairs") or [],
            str(result.get("wrapper_prompt") or ""),
            title=raw_stem,
        )
    except Exception:
        pair_temp_dir.cleanup()
        raise
    return pair_temp_dir, generated_path


def resolve_parallel_epub_glossary_output_dir(raw_path, config, *, create=False):
    """Return the normal Glossary/<raw EPUB>/ directory for a pair ("" without a raw path)."""

    raw_path = str(raw_path or "")
    if not raw_path:
        return ""

    raw_base = os.path.splitext(os.path.basename(raw_path))[0]
    override_dir = os.environ.get("OUTPUT_DIRECTORY") or config.get(
        "output_directory"
    )
    if override_dir:
        shared_glossary_dir = os.path.join(
            os.path.abspath(str(override_dir)), "Glossary"
        )
    else:
        shared_glossary_dir = "Glossary"
    # Match the glossary extractor's writable macOS fallback exactly.
    if sys.platform == "darwin" and not os.path.isabs(shared_glossary_dir):
        shared_glossary_dir = os.path.join(
            os.path.dirname(os.path.abspath(raw_path)), shared_glossary_dir
        )

    from glossary_paths import get_book_glossary_dir

    return get_book_glossary_dir(
        shared_glossary_dir,
        raw_base,
        create=bool(create),
        fallback_base=raw_path,
    )


def parallel_epub_mapping_sidecar_path(raw_path, config, *, create_parent=False):
    """Return the durable mapping path inside the raw book's glossary folder."""

    raw_path = str(raw_path or "").strip()
    if not raw_path:
        return ""
    raw_path = os.path.abspath(raw_path)
    folder = resolve_parallel_epub_glossary_output_dir(
        raw_path, config, create=bool(create_parent)
    )
    if not folder:
        return ""
    from glossary_paths import sanitize_glossary_folder_name

    raw_base = os.path.splitext(os.path.basename(raw_path))[0]
    safe_base = sanitize_glossary_folder_name(raw_base)
    return os.path.join(folder, f"{safe_base}_parallel_epub_mapping.json")


def write_parallel_epub_mapping_sidecar(selection, config):
    """Atomically persist a compact, chapter-text-free pair selection."""

    if not isinstance(selection, dict) or not selection.get("mapping"):
        raise ValueError("The Parallel EPUB mapping is empty.")
    path = parallel_epub_mapping_sidecar_path(
        str(selection.get("raw_path") or ""), config, create_parent=True
    )
    if not path:
        raise ValueError("The Parallel EPUB mapping path could not be resolved.")
    from app_paths import _atomic_json_write

    _atomic_json_write(path, selection)
    return path


def read_parallel_epub_mapping_sidecar(raw_path, config):
    """Read and validate a compact mapping stored with glossary artifacts."""

    path = parallel_epub_mapping_sidecar_path(raw_path, config)
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            selection = json.load(handle)
        if not isinstance(selection, dict) or not isinstance(
            selection.get("mapping"), list
        ):
            return None
        saved_raw_path = os.path.abspath(
            str(selection.get("raw_path") or "")
        )
        if os.path.normcase(saved_raw_path) != os.path.normcase(
            os.path.abspath(str(raw_path or ""))
        ):
            return None
        return selection
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def parallel_epub_pair_source_state(
    result,
    *,
    persistent_selection,
    mapping_sidecar_path="",
    generated_path="",
    pair_temp_dir=None,
):
    """The record of an activated pair (``TranslatorGUI._parallel_epub_pair_source``).

    ``_activate_parallel_epub_pair_source``: the real raw / translated paths, the pair's
    system prompt and profile (read back by ``RunEnvMixin._parallel_epub_system_prompt_for_file``),
    the mapped raw filenames, the compact selection, its sidecar, the working EPUB and the
    temporary folder that holds it.
    """
    raw_path = os.path.abspath(str(result.get("raw_path") or ""))
    translated_path = os.path.abspath(
        str(result.get("translated_path") or "")
    )
    pairs = list(result.get("pairs") or [])
    pair_count = len(pairs)
    return {
        "raw_path": raw_path,
        "translated_path": translated_path,
        "system_prompt": str(result.get("system_prompt") or ""),
        "profile_name": str(result.get("profile_name") or ""),
        "pair_count": pair_count,
        "raw_filenames": [
            str(pair.get("raw_filename") or "") for pair in pairs
        ],
        "persistent_selection": persistent_selection,
        "mapping_sidecar_path": mapping_sidecar_path,
        "generated_path": generated_path,
        "temporary_directory": pair_temp_dir,
    }


def rebuild_parallel_epub_pair_result(selection, config, chapter_loader=None):
    """Rebuild a saved pair's accepted result from the two real EPUBs.

    ``_start_parallel_epub_pair_restore``'s worker up to the working EPUB: reload both EPUBs
    (``chapter_loader``, default :func:`load_parallel_epub_chapters`), reattach the saved
    mapping, and take the prompts from the selection, then the config (wrapper prompt /
    the profile's system prompt), then the defaults. Returns ``(result, skipped mappings)``;
    raises ValueError when none of the saved mappings still exist.
    """
    if chapter_loader is None:
        chapter_loader = load_parallel_epub_chapters
    raw_path = str(selection.get("raw_path") or "")
    translated_path = str(selection.get("translated_path") or "")
    raw_chapters = chapter_loader(raw_path)
    translated_chapters = chapter_loader(
        translated_path
    )
    pairs, skipped = restore_parallel_epub_pairs(
        raw_chapters,
        translated_chapters,
        selection.get("mapping") or [],
    )
    if not pairs:
        raise ValueError(
            "None of the saved HTML filename mappings still exist."
        )
    profile_name = str(
        selection.get("profile_name")
        or DEFAULT_PARALLEL_EPUB_PROFILE
    )
    profiles = config.get("parallel_epub_glossary_profiles", {})
    profile_prompt = (
        profiles.get(profile_name, "")
        if isinstance(profiles, dict)
        else ""
    )
    result = {
        "raw_path": raw_path,
        "translated_path": translated_path,
        "pairs": pairs,
        "wrapper_prompt": str(
            selection.get("wrapper_prompt")
            or config.get("parallel_epub_glossary_wrapper_prompt")
            or DEFAULT_PARALLEL_EPUB_WRAPPER_PROMPT
        ),
        "system_prompt": str(
            selection.get("system_prompt")
            or profile_prompt
            or default_parallel_epub_system_prompt()
        ),
        "profile_name": profile_name,
    }
    return result, skipped
