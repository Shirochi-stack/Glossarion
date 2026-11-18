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

A Retranslate is a decision, not an older state: the mobile retranslate job marks each chapter row it reset
(:func:`mark_reset_rows`, ``retranslate_reset_at``), and every merge keeps that reset over a copy of the row
that is not newer than it (the old completed row, its chapter file and its chunk results do not come back).
Rows an interrupted run left still lose to a finished copy.

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
    "DIFFERENT_BOOK",
    "RESET_MARK",
    "RESET_PARTIAL",
    "RUN_ROOTS_DIR",
    "STASH_DIR",
    "adopt_workspace",
    "book_glossary_dir",
    "carry_glossary",
    "chapter_hashes",
    "chat_spelling",
    "join_workspace",
    "keep_book_state",
    "mark_reset_rows",
    "merge_progress",
    "merge_tree",
    "named_after",
    "points_elsewhere",
    "prepare_merge",
    "read_progress",
    "reset_outputs",
    "run_root_of",
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

#: The folder (under the app's data folder) that holds the chat runs' run roots
#: (``ChatFeature``: ``<data>/direct_text_runs/glossarion_input_output_XXXX``).
RUN_ROOTS_DIR = "direct_text_runs"

#: A chapter row's Retranslate time (the mobile retranslate job, :func:`mark_reset_rows`).
RESET_MARK = "retranslate_reset_at"

#: On a row whose Retranslate reset only some of its chunks: the time that reset started. Its chapter file still
#: holds the other chunks (the reset rewrote it then), so only a copy older than that is the old text.
RESET_PARTIAL = "retranslate_reset_partial_since"

#: :func:`adopt_workspace`'s reason when ``<target root>/<book>`` belongs to another book.
DIFFERENT_BOOK = "a different book has this name"


def _norm(path: Any) -> str:
    try:
        return os.path.normcase(os.path.normpath(os.path.abspath(str(path))))
    except Exception:
        return str(path or "")


def run_root_of(path: Any) -> str:
    """The chat run root ``path`` lies in (``<data>/direct_text_runs/<run root>``, the run root itself
    included), '' when it lies in none. A run root is a chat run's temporary output folder, never a Library one."""
    if not path:
        return ""
    try:
        current = os.path.normpath(os.path.abspath(str(path)))
    except Exception:
        return ""
    while True:
        parent = os.path.dirname(current)
        if not parent or parent == current:
            return ""
        if os.path.basename(parent).casefold() == RUN_ROOTS_DIR:
            return current
        current = parent


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


def carry_glossary(workspace: str, target_root: str, old_root: str = "", *, keep_existing: bool = False) -> str:
    """The workspace now lives under ``target_root``: its glossary - the stash ``<workspace>/Glossary/`` and
    the old root's ``<old root>/Glossary/<book>/`` - joins ``<target root>/Glossary/<book>/`` (newer files
    win; ``keep_existing``: a file the target folder already holds stays, as when the workspace merged into a
    book the Library already had), where the next run of the book reads and continues it. The stash is removed
    once copied; the old root's folder is left in place. Returns the glossary folder ('' when there was nothing
    to carry)."""
    if not workspace or not target_root:
        return ""
    book = os.path.basename(os.path.normpath(str(workspace)))
    target = book_glossary_dir(target_root, book)
    carried = False
    if old_root and not same_dir(old_root, target_root):
        old = book_glossary_dir(old_root, book)
        if os.path.isdir(old) and not same_dir(old, target):
            merge_tree(old, target, keep_existing=keep_existing)
            carried = True
    stash = os.path.join(str(workspace), STASH_DIR)
    if os.path.isdir(stash) and not same_dir(stash, target):
        merge_tree(stash, target, keep_existing=keep_existing)
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


def points_elsewhere(target: Any, raw: Any) -> bool:
    """The folder ``target``'s ``source_epub.txt`` names a raw file other than ``raw`` (not the same path, not
    the same content): ``target`` is another book's workspace. False when either is unknown."""
    if not target or not raw:
        return False
    try:
        import library_core  # shared (U5)
    except Exception:
        return False
    read_pointer = getattr(library_core, "_read_source_epub_pointer", None)
    same_content = getattr(library_core, "_same_file_content", None)
    if read_pointer is None or not os.path.isfile(os.path.join(str(target), "source_epub.txt")):
        return False
    pointed = read_pointer(str(target))
    if not pointed or _norm(pointed) == _norm(raw):
        return False
    if same_content is not None and os.path.isfile(str(raw)) and os.path.isfile(str(pointed)):
        try:
            return not same_content(str(raw), str(pointed))
        except Exception:
            return True
    return True


def _same_book_unpointed(target: str, folder: str, source: str = "") -> bool:
    """:func:`same_book` for a ``target`` without a ``source_epub.txt`` (the pipeline writes it only when its
    translation stage creates the folder: a run stopped in the glossary stage leaves none): ``target`` holds no
    translation yet, or it is named after the raw file's stem and its translation is of the same chapters.
    A ``target`` with a pointer is left to :func:`same_book`."""
    try:
        import library_core  # shared (U5)

        read_pointer = getattr(library_core, "_read_source_epub_pointer", None)
    except Exception:
        read_pointer = None
    if os.path.isfile(os.path.join(target, "source_epub.txt")) and read_pointer is not None and read_pointer(target):
        return False
    if not os.path.isfile(os.path.join(target, PROGRESS_FILE)):
        return True
    raw = source or (read_pointer(folder) if read_pointer is not None else "") or ""
    stem = os.path.splitext(os.path.basename(str(raw)))[0] if raw else ""
    return bool(stem) and named_after(os.path.basename(os.path.normpath(target)), stem) and same_chapters(folder, target)


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


def _partial_outputs(progress: Any) -> list:
    """The chapter files of the rows in ``progress`` that only started: partial text. The pipeline takes a chapter
    file without a row for a finished chapter, so it must never arrive without its row."""
    chapters = (progress.get("chapters") if isinstance(progress, dict) else None) or {}
    files = {True: set(), False: set()}
    for entry in chapters.values():
        name = str(entry.get("output_file") or "") if isinstance(entry, dict) else ""
        if name and not os.path.isabs(name):
            files[bool(_row_rank(entry))].add(name)
    return sorted(files[False] - files[True])


def _compiled_documents(folder: str) -> list:
    """The compiled EPUB / PDF files of a workspace (never its raw file): built from its own chapters."""
    try:
        from output_workspace import read_workspace_source_path

        raw = read_workspace_source_path(folder)
        names = [entry.name for entry in os.scandir(folder) if entry.is_file()]
    except (ImportError, OSError):
        return []
    return [name for name in names if os.path.splitext(name)[1].lower() in (".epub", ".pdf")
            and not (raw and _norm(os.path.join(folder, name)) == _norm(raw))]


def _stamp(entry: Any, key: str) -> float:
    try:
        return float(entry.get(key) or 0) if isinstance(entry, dict) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _reset_wins(row: Any, rival: Any) -> bool:
    """``row`` carries a Retranslate mark (:data:`RESET_MARK`) no older than ``rival``'s last change (None: no
    rival): the user's reset stands over that copy of the chapter."""
    mark = _stamp(row, RESET_MARK)
    return bool(mark) and (rival is None or _stamp(rival, "last_updated") <= mark)


def _live_reset(entry: Any) -> bool:
    """A row the Retranslate reset that nothing has translated since (marked, not changed after the mark)."""
    mark = _stamp(entry, RESET_MARK)
    return bool(mark) and not _row_rank(entry) and _stamp(entry, "last_updated") <= mark


def _reset_file_kept(entry: Any, path: str) -> bool:
    """``path`` is what a chunk-level Retranslate left of the row ``entry``'s chapter (the other chunks' text,
    rewritten by that reset or later: not older than :data:`RESET_PARTIAL`, with a little slack for coarse file
    times), not an older copy of the whole old chapter."""
    since = _stamp(entry, RESET_PARTIAL)
    if not since or not path:
        return False
    try:
        return os.path.getmtime(path) >= since - 2.0
    except OSError:
        return False


def reset_outputs(progress: Any, folder: str = "") -> list:
    """The chapter files of the rows in ``progress`` the Retranslate reset and nothing translated since: the old
    translation, which must not sit beside the reset row (the pipeline would take it for the chapter's text).
    A row whose reset took only some of its chunks keeps the file that reset left (the other chunks); an older
    copy of the chapter in ``folder`` is old text (no ``folder``: such a file is kept)."""
    chapters = (progress.get("chapters") if isinstance(progress, dict) else None) or {}
    files = {True: set(), False: set()}
    for entry in chapters.values():
        name = str(entry.get("output_file") or "") if isinstance(entry, dict) else ""
        if name and not os.path.isabs(name):
            drop = _live_reset(entry)
            if drop and entry.get(RESET_PARTIAL) and (not folder or _reset_file_kept(entry, os.path.join(folder, name))):
                drop = False
            files[drop].add(name)
    return sorted(files[True] - files[False])


def _drop_reset_outputs(folder: str, progress: Any) -> None:
    for name in reset_outputs(progress, folder):
        path = os.path.join(folder, name)
        if os.path.isfile(path):
            try:
                os.remove(path)
            except OSError:
                log.warning("removing the old text of the retranslated chapter %s failed", path, exc_info=True)


def mark_reset_rows(progress_file: str, before: Any, when: Optional[float] = None, *, rows: Any = (),
                    partial: Any = (), since: Optional[float] = None) -> int:
    """The mobile retranslate job reset chapters of ``progress_file`` (``before``: the file as it was): each row it
    reset - the plan's chapter rows (``rows``: progress keys) and chunk parents (``partial``), else each row it
    turned to ``pending`` - carries :data:`RESET_MARK` (and ``last_updated``) = ``when``, so a merge with an older
    copy of the book keeps the reset. A row only some of whose chunks were reset (in ``partial``, not in ``rows``)
    also carries :data:`RESET_PARTIAL` = ``since`` (when the reset started): its file still holds the other
    chunks. Returns the number of rows marked."""
    import time

    if not progress_file:
        return 0
    try:
        with open(str(progress_file), encoding="utf-8") as stream:
            progress = json.load(stream)
    except (OSError, ValueError):
        return 0
    chapters = progress.get("chapters") if isinstance(progress, dict) else None
    if not isinstance(chapters, dict):
        return 0
    old_rows = ((before.get("chapters") if isinstance(before, dict) else None) or {})
    stamp = float(when if when is not None else time.time())
    whole = {str(key) for key in rows or ()}
    chunked = {str(key) for key in partial or ()} - whole
    marked = 0
    for key, entry in chapters.items():
        if not isinstance(entry, dict) or str(entry.get("status") or "") != "pending":
            continue
        previous = old_rows.get(key)
        if (str(key) not in whole and str(key) not in chunked and isinstance(previous, dict)
                and str(previous.get("status") or "") == "pending"):
            continue  # pending before and not in the plan (an interrupted run's row): not this reset's row
        entry[RESET_MARK] = stamp
        entry["last_updated"] = stamp
        if str(key) in chunked:
            # still reset by an earlier chunk reset (pending since): its file dates from then
            earlier = _stamp(previous, RESET_PARTIAL) if _live_reset(previous) else 0.0
            entry[RESET_PARTIAL] = earlier or float(since if since is not None else stamp)
        else:
            entry.pop(RESET_PARTIAL, None)
        marked += 1
    if marked:
        path = str(progress_file)
        temp = path + ".merge"
        with open(temp, "w", encoding="utf-8") as stream:
            json.dump(progress, stream, ensure_ascii=False, indent=2)
        os.replace(temp, path)
    return marked


def merge_progress(own: Any, other: Any) -> tuple:
    """Two progress files of one book as one: ``(merged, rows taken from other)``. ``own`` stays as it is, but
    for the chapters ``other`` got further with (a row is matched by its source file, its key otherwise, since
    the key holds a content hash). A row of ``other`` that only started is dropped: nothing is running it.
    A Retranslate reset (:data:`RESET_MARK`) wins over a copy of the row that is not newer than it, on either
    side: its chunk results come with it, the old ones are dropped."""
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
        chapter = chapter_of(key, entry)
        found = rows.get(chapter)
        mine = chapters.get(found) if found is not None else None
        if mine is not None and _reset_wins(mine, entry):
            continue  # own's Retranslate is newer than other's copy of the chapter
        if not _reset_wins(entry, mine):
            rank = _row_rank(entry)
            if not rank or (mine is not None and _row_rank(mine) >= rank):
                continue
        if found is not None:
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
    for name in _partial_outputs(read_progress(folder)):
        if os.path.isfile(os.path.join(target, name)):
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
    :func:`prepare_merge`); nothing to do without one. A chapter the Library book had reset for retranslation
    stays reset: the chat's older text the Migrate copied in for it is removed."""
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
            _drop_reset_outputs(folder, merged)
    except OSError:
        log.exception("restoring the Library book's state in %s failed", folder)


def join_workspace(folder: str, target: str) -> str:
    """The folder ``folder`` is the book's workspace under another name: make ``target`` hold it. No ``target``
    yet: ``folder`` is renamed. Otherwise what ``target`` lacks comes over (:func:`merge_progress`) and
    ``folder`` is removed. A chapter either folder reset for retranslation (and nothing translated since) stays
    reset, without its old text. Returns "moved", "merged" or "" (nothing changed: a progress file cannot be
    read, so nothing could be kept for sure, or a file operation failed)."""
    try:
        if not os.path.exists(target):
            os.rename(folder, target)
            return "moved"
        mine, theirs = os.path.join(folder, PROGRESS_FILE), os.path.join(target, PROGRESS_FILE)
        own, other = read_progress(target), read_progress(folder)
        if (os.path.exists(theirs) and own is None) or (os.path.exists(mine) and other is None):
            log.warning("not joining %s and %s: a progress file cannot be read", folder, target)
            return ""
        # what stays behind: an unfinished row's text (its row is dropped) and the outputs built from this folder;
        # not what a chunk-level Retranslate left of a chapter (its other chunks: the merge decides on it below)
        chunk_resets = {str(entry.get("output_file") or "") for entry in ((other or {}).get("chapters") or {}).values()
                        if isinstance(entry, dict) and entry.get(RESET_PARTIAL) and _live_reset(entry)}
        for name in [n for n in _partial_outputs(other) if n not in chunk_resets] + _compiled_documents(folder):
            path = os.path.join(folder, name)
            if os.path.isfile(path):
                os.remove(path)
        merge_tree(folder, target, keep_existing=True)
        if own is not None and other is not None:
            merged, taken = merge_progress(own, other)
            for entry in taken:  # the chapter's text goes with its row, over what an unfinished row left there
                name = str(entry.get("output_file") or "")
                path = os.path.join(folder, name)
                if not name or os.path.isabs(name):
                    continue
                if _live_reset(entry):
                    # a reset row: only what a chunk-level reset left of the chapter comes along; target's older
                    # copy of the whole chapter is old text
                    kept = _reset_file_kept(entry, path)
                    if kept:
                        shutil.copy2(path, os.path.join(target, name))
                    elif entry.get(RESET_PARTIAL) and os.path.isfile(os.path.join(target, name)):
                        os.remove(os.path.join(target, name))
                elif os.path.isfile(path):
                    shutil.copy2(path, os.path.join(target, name))
            _write_progress(target, merged)
        _drop_reset_outputs(target, read_progress(target))
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
    the path), ``"moved"``, ``"merged"`` (``<target root>/<book>`` existed and is the same book - its
    ``source_epub.txt`` names the same raw (:func:`same_book`), or it has none and holds no other translation
    - : the two progress files become one (:func:`join_workspace`: the chapters either finished, a Retranslate
    reset kept), then ``folder`` is removed) or ``"kept"`` (nothing moved: ``reason`` says why - another book
    has the name, the folder is gone, the move failed)."""
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
            if not os.path.isdir(target) or not (same_book(target, folder, source)
                                                 or _same_book_unpointed(target, folder, source)):
                return {"status": "kept", "target": target, "reason": DIFFERENT_BOOK}
            status = join_workspace(folder, target)
            if not status:
                return {"status": "kept", "target": target,
                        "reason": "the two copies could not be joined (a progress file cannot be read, or a file "
                                  "operation failed)"}
        else:
            shutil.move(folder, target)
            status = "moved"
    except Exception as exc:
        log.warning("moving the workspace %s into %s failed: %s", folder, target_root, exc)
        return {"status": "kept", "target": target, "reason": str(exc)}
    try:
        # merged into a book the Library already had: its glossary stays, the run's only adds what it lacks (a
        # newer but shorter glossary of a stopped run must not replace a finished one)
        carry_glossary(target, target_root, old_root, keep_existing=status == "merged")
    except Exception:
        log.exception("moving the glossary of %s failed", target)
    return {"status": status, "target": target, "reason": ""}
