"""Source-pointer helpers for EPUB, PDF, and TXT translation workspaces.

``source_epub.txt`` is a legacy filename, but it is the durable source pointer
for all three formats. A same-stem format collision is resolved at input
selection time by renaming the new source file, after which the application's
ordinary stem-based output logic remains the single source of truth.
"""

from __future__ import annotations

import os


SOURCE_REFERENCE_FILENAME = "source_epub.txt"

_FORMAT_LABELS = {
    ".epub": "EPUB",
    ".pdf": "PDF",
    ".txt": "TXT",
}


def source_format_label(path: str) -> str:
    """Return ``EPUB``, ``PDF``, or ``TXT`` for a supported source path."""
    try:
        extension = os.path.splitext(str(path or "").strip())[1].lower()
    except (TypeError, ValueError):
        return ""
    return _FORMAT_LABELS.get(extension, "")


def read_workspace_source_path(workspace: str) -> str:
    """Read a workspace's legacy source pointer without requiring it to exist."""
    pointer = os.path.join(workspace, SOURCE_REFERENCE_FILENAME)
    try:
        with open(pointer, "r", encoding="utf-8-sig", errors="replace") as handle:
            return handle.read().strip().strip('"')
    except (OSError, UnicodeError):
        return ""


def workspace_source_format(workspace: str) -> str:
    """Return the format recorded by ``source_epub.txt`` in *workspace*."""
    return source_format_label(read_workspace_source_path(workspace))


def rename_input_for_workspace_collision(input_path: str, workspace: str) -> str:
    """Rename a selected source when *workspace* belongs to another format.

    Selecting ``Novel.epub`` while the existing ``Novel`` workspace points to
    ``Novel.pdf`` renames the source itself to ``Novel_EPUB.epub``. Normal
    stem-based output creation will then naturally use ``Novel_EPUB``.

    Existing files are never overwritten. A numbered suffix is used when the
    preferred target already exists. Unsupported formats, missing sources,
    matching workspaces, and already-suffixed names are left untouched.
    """
    raw_input = str(input_path or "").strip()
    raw_workspace = str(workspace or "").strip()
    incoming_format = source_format_label(raw_input)
    if not raw_input or not raw_workspace or not incoming_format:
        return raw_input

    source = os.path.abspath(os.path.expanduser(raw_input))
    if not os.path.isfile(source):
        return raw_input

    existing_format = workspace_source_format(os.path.normpath(raw_workspace))
    if not existing_format or existing_format == incoming_format:
        return raw_input

    parent = os.path.dirname(source)
    stem, extension = os.path.splitext(os.path.basename(source))
    suffix = f"_{incoming_format}"
    if stem.casefold().endswith(suffix.casefold()):
        return raw_input

    # Leave margin under the common Windows 255-character component limit.
    max_stem_length = max(1, 240 - len(extension) - len(suffix))
    target_stem = f"{stem[:max_stem_length]}{suffix}"
    candidate = os.path.join(parent, f"{target_stem}{extension}")
    index = 2
    while os.path.exists(candidate):
        numbered_suffix = f"_{index}"
        numbered_stem = target_stem[
            : max(1, 240 - len(extension) - len(numbered_suffix))
        ]
        candidate = os.path.join(
            parent, f"{numbered_stem}{numbered_suffix}{extension}"
        )
        index += 1

    os.rename(source, candidate)
    return candidate


def write_workspace_source_reference(workspace: str, input_path: str) -> str:
    """Persist the absolute raw-source path and return the pointer filename."""
    os.makedirs(workspace, exist_ok=True)
    pointer = os.path.join(workspace, SOURCE_REFERENCE_FILENAME)
    source_path = os.path.abspath(os.path.expanduser(str(input_path or "")))
    with open(pointer, "w", encoding="utf-8") as handle:
        handle.write(source_path)
    return pointer
