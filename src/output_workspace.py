"""Resolve translation workspaces without mixing same-named source formats.

``source_epub.txt`` is a legacy filename, but it is the durable source pointer
for EPUB, PDF, and TXT translation workspaces.  The helpers here keep that
compatibility while ensuring that two inputs such as ``Novel.epub`` and
``Novel.pdf`` cannot share one output directory.
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


def resolve_source_aware_workspace(input_path: str, default_workspace: str) -> str:
    """Return a collision-safe output directory for *input_path*.

    The normal unsuffixed directory remains the first choice.  If it already
    records another supported source format, the input is routed to a sibling
    named ``<stem>_<FORMAT>``.  An existing matching suffixed workspace is
    reused, allowing subsequent runs of an updated PDF/EPUB/TXT to retain its
    progress.  Numeric fallbacks only matter if a manually-created suffixed
    folder itself points at a different format.
    """
    raw_workspace = str(default_workspace or "").strip()
    if not raw_workspace:
        return ""
    workspace = os.path.normpath(raw_workspace)
    incoming_format = source_format_label(input_path)
    if not incoming_format:
        return workspace

    existing_format = workspace_source_format(workspace)
    if not existing_format or existing_format == incoming_format:
        return workspace

    parent = os.path.dirname(workspace)
    leaf = os.path.basename(workspace)
    suffix = f"_{incoming_format}"
    suffixed_leaf = leaf if leaf.casefold().endswith(suffix.casefold()) else f"{leaf}{suffix}"
    suffixed = os.path.join(parent, suffixed_leaf) if parent else suffixed_leaf

    candidate = suffixed
    index = 2
    while os.path.exists(candidate):
        candidate_format = workspace_source_format(candidate)
        if not candidate_format or candidate_format == incoming_format:
            return candidate
        candidate = f"{suffixed}_{index}"
        index += 1
    return candidate


def write_workspace_source_reference(workspace: str, input_path: str) -> str:
    """Persist the absolute raw-source path and return the pointer filename."""
    os.makedirs(workspace, exist_ok=True)
    pointer = os.path.join(workspace, SOURCE_REFERENCE_FILENAME)
    source_path = os.path.abspath(os.path.expanduser(str(input_path or "")))
    with open(pointer, "w", encoding="utf-8") as handle:
        handle.write(source_path)
    return pointer
