"""Keep a book's workspace and its glossary folder together under the Library's output root.

The translation pipeline keeps two folders per book under its output root (``OUTPUT_DIRECTORY``):
the workspace ``<root>/<book>/`` (``translation_progress.json``, the chapter extraction cache
``chapters_full.json`` + ``.chapter_extraction_complete``) and the glossary folder
``<root>/Glossary/<book>/`` (``<book>_glossary.csv``, ``<book>_glossary_progress.json`` and the
glossary's EPUB text cache ``.cache/epub_text.json``). A run "continues" only when both are found
under the root it runs in.

Two mobile paths used to split them:

* a chat run translates in its run root (``<run root>/<book>`` + ``<run root>/Glossary/<book>``), and
  at its end only the workspace tree is kept (``Attachments/<book>``, later moved into the Library);
  the run root - with the glossary folder - is deleted. :func:`stash_run_glossary` keeps the glossary
  inside the workspace (``<workspace>/Glossary/``, a layout the glossary progress lookup already
  reads) and :func:`carry_glossary` files it under ``<root>/Glossary/<book>/`` when the workspace
  moves into the Library (issues 19 / 20: Continue re-extracted the glossary and the EPUB text);
* a Library book whose workspace sits under another folder than the Library's output root asked
  "Output Folder Mismatch" and, on Yes, pointed the whole app at that folder. :func:`adopt_workspace`
  moves (or merges, for the same book) the workspace and its glossary into the Library's root
  instead, so every book stays in the Library's folder (issue 18).

All functions are blocking file work: call them off the UI loop, while no job writes the folders.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from typing import Any, Optional

log = logging.getLogger("glossarion.workspace_sync")

__all__ = [
    "STASH_DIR",
    "adopt_workspace",
    "book_glossary_dir",
    "carry_glossary",
    "chapter_hashes",
    "merge_tree",
    "same_book",
    "same_chapters",
    "same_dir",
    "stash_run_glossary",
]

#: The workspace's own glossary folder (``glossary_progress_core`` searches ``<root>/<book>/Glossary``).
STASH_DIR = "Glossary"


def _norm(path: Any) -> str:
    try:
        return os.path.normcase(os.path.normpath(os.path.abspath(str(path))))
    except Exception:
        return str(path or "")


def same_dir(a: Any, b: Any) -> bool:
    """The same folder, also through a symlinked spelling (Android ``/data/user/0`` = ``/data/data``)."""
    if not a or not b:
        return False
    if _norm(a) == _norm(b):
        return True
    try:
        return os.path.normcase(os.path.realpath(str(a))) == os.path.normcase(os.path.realpath(str(b)))
    except Exception:
        return False


def book_glossary_dir(root: str, book: str) -> str:
    """``<root>/Glossary/<book>/`` as the extractor names it (``glossary_paths.get_book_glossary_dir``)."""
    shared = os.path.join(os.path.abspath(str(root)), "Glossary")
    try:
        from glossary_paths import sanitize_glossary_folder_name

        return os.path.join(shared, sanitize_glossary_folder_name(book))
    except Exception:
        return os.path.join(shared, str(book))


def merge_tree(src: str, dst: str) -> int:
    """Copy every file of ``src`` into ``dst`` (same relative paths); a file both hold keeps the newer copy.
    Returns the number of files copied."""
    copied = 0
    if not src or not os.path.isdir(src):
        return 0
    for folder, _dirs, files in os.walk(src):
        rel = os.path.relpath(folder, src)
        target_dir = dst if rel == "." else os.path.join(dst, rel)
        for name in files:
            source = os.path.join(folder, name)
            target = os.path.join(target_dir, name)
            try:
                if os.path.exists(target) and os.path.getmtime(target) >= os.path.getmtime(source):
                    continue
                os.makedirs(target_dir, exist_ok=True)
                shutil.copy2(source, target)
                copied += 1
            except OSError:
                log.debug("copying %s failed", source, exc_info=True)
    return copied


def stash_run_glossary(run_root: str, book: str, workspace: str) -> str:
    """A chat run ended: its glossary folder ``<run root>/Glossary/<book>/`` goes with the kept workspace
    (``<workspace>/Glossary/``) before the run root is deleted. Returns the stash ('' when there was none)."""
    if not run_root or not workspace or not os.path.isdir(workspace):
        return ""
    source = book_glossary_dir(run_root, book)
    if not os.path.isdir(source):
        return ""
    stash = os.path.join(workspace, STASH_DIR)
    merge_tree(source, stash)
    return stash


def carry_glossary(workspace: str, target_root: str, old_root: str = "") -> str:
    """The workspace now lives under ``target_root``: its glossary - the stash ``<workspace>/Glossary/`` and
    the old root's ``<old root>/Glossary/<book>/`` - joins ``<target root>/Glossary/<book>/`` (newer files
    win), where the next run of the book reads and continues it. The stash is removed once copied; the old
    root's folder is left in place. Returns the glossary folder ('' when there was nothing to carry)."""
    if not workspace or not target_root:
        return ""
    book = os.path.basename(os.path.normpath(str(workspace)))
    target = book_glossary_dir(target_root, book)
    carried = False
    if old_root and not same_dir(old_root, target_root):
        old = book_glossary_dir(old_root, book)
        if os.path.isdir(old) and not same_dir(old, target):
            merge_tree(old, target)
            carried = True
    stash = os.path.join(str(workspace), STASH_DIR)
    if os.path.isdir(stash) and not same_dir(stash, target):
        merge_tree(stash, target)
        shutil.rmtree(stash, ignore_errors=True)
        carried = True
    return target if carried else ""


def chapter_hashes(folder: Any) -> set:
    """The ``content_hash`` of every chapter in ``<folder>/translation_progress.json`` (the pipeline
    hashes each chapter's source text); empty when there is no readable progress."""
    try:
        with open(os.path.join(str(folder), "translation_progress.json"), encoding="utf-8") as stream:
            progress = json.load(stream)
    except (OSError, ValueError):
        return set()
    chapters = progress.get("chapters") if isinstance(progress, dict) else None
    if not isinstance(chapters, dict):
        return set()
    return {str(entry["content_hash"]) for entry in chapters.values()
            if isinstance(entry, dict) and entry.get("content_hash")}


def same_chapters(folder: str, target: str) -> bool:
    """``target`` holds no translation yet, or at least half of the chapters either workspace translated
    carry the same pipeline ``content_hash``: the two translations are of the same book."""
    theirs = chapter_hashes(target)
    if not theirs:
        return True
    mine = chapter_hashes(folder)
    shared = len(mine & theirs)
    return bool(mine) and shared > 0 and shared * 2 >= min(len(mine), len(theirs))


def same_book(target: str, folder: str, source: str = "") -> bool:
    """The folder ``target`` is the same book as the workspace ``folder``: its ``source_epub.txt`` points at
    the same raw file (or the same content) as ``source`` / ``folder``'s own pointer, and any translation it
    holds is of the same chapters (:func:`same_chapters`). A path alone proves nothing (FileBridge reuses a
    freed Inbox name)."""
    try:
        import library_core  # shared (U5)
    except Exception:
        return False
    read_pointer = getattr(library_core, "_read_source_epub_pointer", None)
    same_content = getattr(library_core, "_same_file_content", None)
    if read_pointer is None:
        return False
    pointed = read_pointer(target)
    if not pointed:
        return False
    for raw in (source, read_pointer(folder)):
        if not raw or not os.path.isfile(raw):
            continue
        if _norm(raw) == _norm(pointed) or (same_content is not None and same_content(raw, pointed)):
            return same_chapters(folder, target)
    return False


def adopt_workspace(folder: str, target_root: str, source: str = "") -> dict:
    """Move the workspace ``folder`` (and its glossary) into ``target_root`` - the Library's output root.

    Returns ``{"status", "target", "reason"}``: ``"same"`` (already there, maybe under another spelling of
    the path), ``"moved"``, ``"merged"`` (``<target root>/<book>`` existed and is the same book: newer files
    win, then ``folder`` is removed) or ``"kept"`` (nothing moved: ``reason`` says why - another book has the
    name, the folder is gone, the move failed)."""
    folder = os.path.abspath(str(folder or ""))
    if not folder or not os.path.isdir(folder):
        return {"status": "kept", "target": "", "reason": "the workspace folder is missing"}
    if not target_root:
        return {"status": "kept", "target": "", "reason": "no Library output folder"}
    old_root = os.path.dirname(folder)
    if same_dir(old_root, target_root):
        return {"status": "same", "target": folder, "reason": ""}
    try:
        nested = os.path.commonpath([_norm(folder), _norm(target_root)]) == _norm(folder)
    except ValueError:  # another drive
        nested = False
    if nested:
        return {"status": "kept", "target": "", "reason": "the output folder lies inside this folder"}
    target = os.path.join(os.path.abspath(str(target_root)), os.path.basename(folder))
    try:
        os.makedirs(target_root, exist_ok=True)
        if os.path.exists(target):
            if not os.path.isdir(target) or not same_book(target, folder, source):
                return {"status": "kept", "target": target, "reason": "a different book has this name"}
            merge_tree(folder, target)
            shutil.rmtree(folder, ignore_errors=True)
            status = "merged"
        else:
            shutil.move(folder, target)
            status = "moved"
    except Exception as exc:
        log.warning("moving the workspace %s into %s failed: %s", folder, target_root, exc)
        return {"status": "kept", "target": target, "reason": str(exc)}
    try:
        carry_glossary(target, target_root, old_root)
    except Exception:
        log.exception("moving the glossary of %s failed", target)
    return {"status": status, "target": target, "reason": ""}
