"""Helpers for locating translation workspaces outside the normal output root.

Direct Text keeps attachment runs under::

    <output root>/Direct Text/<conversation>/Attachments/<source stem>

Those workspaces are deliberately isolated from ordinary composer messages, but
they still contain regular translation artifacts such as
``translation_progress.json``.  This module provides a small, strict scanner so
callers can include them without relying on ambiguous basename matching.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator


SOURCE_INPUT_SIDECAR = "source_input.txt"
SOURCE_EPUB_SIDECAR = "source_epub.txt"


def _path_key(path: str) -> str:
    """Return a case/platform-normalized absolute key for *path*."""
    if not path:
        return ""
    try:
        return os.path.normcase(os.path.realpath(os.path.abspath(path)))
    except (OSError, TypeError, ValueError):
        return ""


def _read_path_sidecar(workspace: str, filename: str) -> str:
    sidecar = os.path.join(workspace, filename)
    if not os.path.isfile(sidecar):
        return ""
    try:
        with open(sidecar, "r", encoding="utf-8-sig", errors="replace") as handle:
            value = handle.read().strip()
    except OSError:
        return ""
    if not value:
        return ""
    if not os.path.isabs(value):
        value = os.path.join(workspace, value)
    return os.path.normpath(value)


def workspace_source_paths(workspace: str) -> list[str]:
    """Return source paths explicitly recorded by one output workspace.

    Sidecars are authoritative.  Progress metadata is only consulted for
    explicit source/input fields; chapter names and workspace basenames are
    intentionally never treated as proof of identity.
    """
    paths: list[str] = []
    seen: set[str] = set()

    def _add(value) -> None:
        if not isinstance(value, str) or not value.strip():
            return
        candidate = value.strip()
        if not os.path.isabs(candidate):
            candidate = os.path.join(workspace, candidate)
        key = _path_key(candidate)
        if not key or key in seen:
            return
        seen.add(key)
        paths.append(os.path.normpath(candidate))

    _add(_read_path_sidecar(workspace, SOURCE_INPUT_SIDECAR))
    _add(_read_path_sidecar(workspace, SOURCE_EPUB_SIDECAR))

    progress_path = os.path.join(workspace, "translation_progress.json")
    if not os.path.isfile(progress_path):
        return paths
    try:
        with open(progress_path, "r", encoding="utf-8-sig") as handle:
            progress = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return paths
    if not isinstance(progress, dict):
        return paths

    source_keys = (
        "source_input",
        "source_file",
        "source_path",
        "input_file",
        "input_path",
        "epub_path",
    )
    for key in source_keys:
        _add(progress.get(key))
    return paths


def workspace_matches_input(workspace: str, input_path: str) -> bool:
    """Return True only for an explicit, exact source-path match."""
    wanted = _path_key(input_path)
    if not wanted:
        return False
    return any(_path_key(path) == wanted for path in workspace_source_paths(workspace))


def write_workspace_source_input(workspace: str, input_path: str) -> str:
    """Persist the original input path used by an attachment workspace."""
    if not workspace or not input_path:
        return ""
    os.makedirs(workspace, exist_ok=True)
    sidecar = os.path.join(workspace, SOURCE_INPUT_SIDECAR)
    with open(sidecar, "w", encoding="utf-8") as handle:
        handle.write(os.path.abspath(input_path))
    return sidecar


def _casefold_child(parent: str, wanted_name: str) -> str:
    try:
        with os.scandir(parent) as entries:
            for entry in entries:
                if (
                    entry.is_dir(follow_symlinks=False)
                    and entry.name.casefold() == wanted_name.casefold()
                ):
                    return entry.path
    except (OSError, PermissionError):
        pass
    return ""


def iter_direct_text_attachment_workspaces(output_root: str) -> Iterator[str]:
    """Yield Direct Text attachment workspaces below *output_root*.

    The traversal is intentionally bounded to the documented three directory
    levels instead of recursively scanning an arbitrary output tree.
    """
    if not output_root or not os.path.isdir(output_root):
        return
    root = os.path.abspath(output_root)
    if os.path.basename(os.path.normpath(root)).casefold() == "direct text":
        direct_text_root = root
    else:
        direct_text_root = _casefold_child(root, "Direct Text")
    if not direct_text_root:
        return

    try:
        conversations = os.scandir(direct_text_root)
    except (OSError, PermissionError):
        return
    with conversations:
        for conversation in conversations:
            if not conversation.is_dir(follow_symlinks=False):
                continue
            if conversation.name.casefold() == "attachments":
                # Compatibility with the early Direct Text layout that did
                # not place attachment workspaces below a conversation.
                attachments = conversation.path
            else:
                attachments = _casefold_child(conversation.path, "Attachments")
            if not attachments:
                continue
            try:
                workspaces = os.scandir(attachments)
            except (OSError, PermissionError):
                continue
            with workspaces:
                for workspace in workspaces:
                    if workspace.is_dir(follow_symlinks=False):
                        yield workspace.path


def find_direct_text_attachment_workspace(
    output_root: str,
    input_path: str,
    *,
    require_progress: bool = False,
) -> str:
    """Return the newest exact source match in Direct Text attachments."""
    matches: list[tuple[float, str]] = []
    for workspace in iter_direct_text_attachment_workspaces(output_root):
        if require_progress and not os.path.isfile(
            os.path.join(workspace, "translation_progress.json")
        ):
            continue
        if not workspace_matches_input(workspace, input_path):
            continue
        progress_path = os.path.join(workspace, "translation_progress.json")
        try:
            mtime = os.path.getmtime(
                progress_path if os.path.isfile(progress_path) else workspace
            )
        except OSError:
            mtime = 0.0
        matches.append((mtime, workspace))
    if not matches:
        return ""
    matches.sort(key=lambda item: item[0], reverse=True)
    return matches[0][1]
