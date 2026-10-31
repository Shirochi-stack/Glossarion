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

A third one split the workspace itself: the pipeline names it after the raw file's whole stem
(``<root>/X.`` for ``X..epub``, the 120+ characters of a long name), a chat names its copy with
``direct_text_store.attachment_workspace_name`` (``Attachments/X``), and that copy used to move into the
Library under the chat's spelling. The book then had two folders: the Library card, the Reader and the
Progress manager of ``<root>/X`` showed one translation while every run of it went on in ``<root>/X.``
(the owner's "the progress file is lost after an update": the sweep at the next start made the move).
:func:`split_workspaces` finds such folders and :func:`join_workspace` makes them one; a merge keeps the
chapters both folders finished (:func:`merge_progress`).

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
    "chat_spelling",
    "join_workspace",
    "keep_book_state",
    "merge_progress",
    "merge_tree",
    "named_after",
    "prepare_merge",
    "read_progress",
    "same_book",
    "same_chapters",
    "same_dir",
    "split_workspaces",
    "stash_run_glossary",
]

PROGRESS_FILE = "translation_progress.json"
METADATA_FILE = "metadata.json"

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


def merge_tree(src: str, dst: str, *, keep_existing: bool = False) -> int:
    """Copy every file of ``src`` into ``dst`` (same relative paths); a file both hold keeps the newer copy
    (``keep_existing``: ``dst``'s copy). Returns the number of files copied."""
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
                if os.path.exists(target) and (keep_existing or os.path.getmtime(target) >= os.path.getmtime(source)):
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


def chat_spelling(stem: Any) -> str:
    """The ``Attachments/`` folder name a chat gives the file stem ``stem`` (not always the stem)."""
    from direct_text_store import attachment_workspace_name  # shared (U3)

    return attachment_workspace_name(stem)


def named_after(folder_name: Any, stem: Any) -> bool:
    """The folder ``folder_name`` is named after a raw file with this stem: the stem itself, or a chat's
    spelling of it (any case)."""
    if not folder_name or not stem:
        return False
    name = str(folder_name).casefold()
    return name == str(stem).casefold() or name == chat_spelling(stem).casefold()


def read_progress(folder: Any) -> Optional[dict]:
    """``<folder>/translation_progress.json``, None when it is missing or unreadable."""
    try:
        with open(os.path.join(str(folder), PROGRESS_FILE), encoding="utf-8") as stream:
            progress = json.load(stream)
    except (OSError, ValueError):
        return None
    return progress if isinstance(progress, dict) else None


def _write_progress(folder: str, progress: dict) -> None:
    path = os.path.join(folder, PROGRESS_FILE)
    temp = path + ".merge"
    with open(temp, "w", encoding="utf-8") as stream:
        json.dump(progress, stream, ensure_ascii=False, indent=2)
    os.replace(temp, path)


def _row_rank(entry: Any) -> int:
    """How far a progress row got: 2 finished, 1 ended with an issue (a QA failure, a failed request), 0 no
    further than started (the rows an interrupted run leaves)."""
    status = str(entry.get("status") or "") if isinstance(entry, dict) else ""
    if status.startswith("completed"):
        return 2
    return 0 if status in ("", "in_progress", "pending") else 1


def merge_progress(own: Any, other: Any) -> tuple:
    """Two progress files of one book as one: ``(merged, rows taken from other)``. ``own`` stays as it is, but
    for the chapters ``other`` got further with (a row is matched by its source file, its key otherwise, since
    the key holds a content hash). A row of ``other`` that only started is dropped: nothing is running it."""
    merged = dict(own) if isinstance(own, dict) else {}
    other = other if isinstance(other, dict) else {}
    chapters = dict(merged.get("chapters") or {})
    chunks = dict(merged.get("chapter_chunks") or {})
    other_chunks = other.get("chapter_chunks") or {}

    def chapter_of(key: Any, entry: Any) -> str:
        return str((entry.get("original_basename") if isinstance(entry, dict) else "") or key)

    rows = {chapter_of(key, entry): key for key, entry in chapters.items()}
    taken: list = []
    for key, entry in (other.get("chapters") or {}).items():
        rank = _row_rank(entry)
        if not rank:
            continue
        chapter = chapter_of(key, entry)
        found = rows.get(chapter)
        if found is not None:
            if _row_rank(chapters[found]) >= rank:
                continue
            replaced = chapters.pop(found)
            if isinstance(replaced, dict):
                chunks.pop(str(replaced.get("content_hash") or found), None)
        chapters[key] = entry
        rows[chapter] = key
        chunk_key = str(entry.get("content_hash") or key)
        if chunk_key in other_chunks:
            chunks[chunk_key] = other_chunks[chunk_key]
        taken.append(entry)
    merged["chapters"] = chapters
    if chunks or "chapter_chunks" in merged:
        merged["chapter_chunks"] = chunks
    return merged, taken


def prepare_merge(folder: str, target: str) -> dict:
    """The shared Migrate is about to copy the chat's workspace ``folder`` over the Library folder ``target``,
    replacing files of the same name. The partial text of a chapter the chat only started must not replace a
    chapter ``target`` holds (nothing is running it: the chat's workspace is idle). Returns what
    :func:`keep_book_state` puts back."""
    for entry in ((read_progress(folder) or {}).get("chapters") or {}).values():
        name = str(entry.get("output_file") or "") if isinstance(entry, dict) else ""
        if name and not os.path.isabs(name) and not _row_rank(entry) and os.path.isfile(os.path.join(target, name)):
            try:
                os.remove(os.path.join(folder, name))
            except OSError:
                log.debug("removing the partial chapter %s failed", name, exc_info=True)
    metadata = None
    try:
        with open(os.path.join(target, METADATA_FILE), "rb") as stream:
            metadata = stream.read()
    except OSError:
        pass
    return {"progress": read_progress(target), "metadata": metadata}


def keep_book_state(folder: str, state: Any) -> None:
    """The Migrate merged a chat's workspace into the Library folder ``folder``. The book keeps its title and
    the chapters only its old progress file knew, whose files are all still there (``state``:
    :func:`prepare_merge`); nothing to do without one."""
    if not isinstance(state, dict):
        return
    try:
        if state.get("metadata") is not None:
            with open(os.path.join(folder, METADATA_FILE), "wb") as stream:
                stream.write(state["metadata"])
        if isinstance(state.get("progress"), dict):
            merged, taken = merge_progress(read_progress(folder), state["progress"])
            if taken:
                _write_progress(folder, merged)
    except OSError:
        log.exception("restoring the Library book's state in %s failed", folder)


def join_workspace(folder: str, target: str) -> str:
    """The folder ``folder`` is the book's workspace under another name: make ``target`` hold it. No ``target``
    yet: ``folder`` is renamed. Otherwise what ``target`` lacks comes over (:func:`merge_progress`) and
    ``folder`` is removed. Returns "moved", "merged" or "" (nothing changed: a progress file cannot be read,
    so nothing could be kept for sure, or a file operation failed)."""
    try:
        if not os.path.exists(target):
            os.rename(folder, target)
            return "moved"
        mine, theirs = os.path.join(folder, PROGRESS_FILE), os.path.join(target, PROGRESS_FILE)
        own, other = read_progress(target), read_progress(folder)
        if (os.path.exists(theirs) and own is None) or (os.path.exists(mine) and other is None):
            log.warning("not joining %s and %s: a progress file cannot be read", folder, target)
            return ""
        merge_tree(folder, target, keep_existing=True)
        if own is not None and other is not None:
            merged, taken = merge_progress(own, other)
            for entry in taken:  # the chapter's text goes with its row, over what an unfinished row left there
                name = str(entry.get("output_file") or "")
                path = os.path.join(folder, name)
                if name and not os.path.isabs(name) and os.path.isfile(path):
                    shutil.copy2(path, os.path.join(target, name))
            _write_progress(target, merged)
        shutil.rmtree(folder)
    except OSError as exc:
        log.warning("joining the workspace %s to %s failed: %s", folder, target, exc)
        return ""
    return "merged"


def split_workspaces(root: Any) -> list:
    """``(folder, target, raw)`` for each first-level folder of the output ``root`` that holds a book under a
    chat's spelling of its raw file's stem, while the pipeline's folder for that raw file is ``target``
    (``<root>/<stem>``, which may not exist yet). A ``target`` that belongs to another book, or that is the same
    folder (a file system that drops trailing dots), is left alone."""
    try:
        import library_core  # shared (U5)
    except Exception:
        return []
    read_pointer = getattr(library_core, "_read_source_epub_pointer", None)
    same_content = getattr(library_core, "_same_file_content", None)
    if read_pointer is None:
        return []
    root = str(root or "")
    try:
        entries = sorted((e for e in os.scandir(root) if e.is_dir(follow_symlinks=False)), key=lambda e: e.name)
    except OSError:
        return []
    found: list = []
    for entry in entries:
        raw = read_pointer(entry.path) if os.path.isfile(os.path.join(entry.path, "source_epub.txt")) else None
        if not raw:
            continue
        stem = os.path.splitext(os.path.basename(raw))[0]
        if stem == entry.name or chat_spelling(stem) != entry.name:
            continue
        target = os.path.join(root, stem)
        if os.path.exists(target):
            if not os.path.isdir(target) or os.path.samefile(entry.path, target):
                continue
            pointed = read_pointer(target)
            if pointed and _norm(pointed) != _norm(raw) and not (same_content is not None and same_content(pointed, raw)):
                continue
        found.append((entry.path, target, raw))
    return found


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
