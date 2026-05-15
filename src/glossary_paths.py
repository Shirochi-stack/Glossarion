# -*- coding: utf-8 -*-
"""Shared helpers for per-book glossary storage."""

import os
import shutil


_WINDOWS_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
    "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
}


def glossary_book_base_from_output(output_path_or_name: str) -> str:
    """Return the book base name from a glossary output filename/path."""
    name = os.path.splitext(os.path.basename(str(output_path_or_name or "")))[0]
    for suffix in ("_glossary_progress", "_glossary_history", "_gender_tracker", "_glossary"):
        if name.lower().endswith(suffix):
            return name[:-len(suffix)] or "book"
    return name or "book"


def sanitize_glossary_folder_name(name: str) -> str:
    """Make a book title safe for use as a folder name while preserving Unicode."""
    raw = str(name or "").strip()
    cleaned = []
    for ch in raw:
        if ord(ch) < 32 or ch in '<>:"/\\|?*':
            cleaned.append("_")
        else:
            cleaned.append(ch)
    folder = "".join(cleaned).strip(" .")
    while "__" in folder:
        folder = folder.replace("__", "_")
    if not folder:
        folder = "book"
    if folder.upper() in _WINDOWS_RESERVED_NAMES:
        folder = f"{folder}_book"
    return folder[:150].rstrip(" .") or "book"


def get_book_glossary_dir(shared_glossary_dir: str, book_name: str, create: bool = True) -> str:
    """Return Glossary/<book>/, creating it by default."""
    root = os.path.abspath(shared_glossary_dir or os.path.join(os.getcwd(), "Glossary"))
    folder = sanitize_glossary_folder_name(book_name)
    path = os.path.join(root, folder)
    if create:
        os.makedirs(path, exist_ok=True)
    return path


def get_book_glossary_path(shared_glossary_dir: str, book_name: str, filename: str, create: bool = True) -> str:
    return os.path.join(get_book_glossary_dir(shared_glossary_dir, book_name, create=create), os.path.basename(filename))


def migrate_legacy_glossary_files(shared_glossary_dir: str, book_name: str, logger=None):
    """Move old flat Glossary/<book>_* files into Glossary/<book>/.

    Existing destination files are left untouched to avoid overwriting user edits.
    Returns a list of ``(source, destination)`` pairs that were moved.
    """
    root = os.path.abspath(shared_glossary_dir or os.path.join(os.getcwd(), "Glossary"))
    if not os.path.isdir(root):
        return []

    base = glossary_book_base_from_output(book_name)
    book_dir = get_book_glossary_dir(root, base, create=True)
    filenames = [
        f"{base}_glossary.json",
        f"{base}_glossary.csv",
        f"{base}_glossary.txt",
        f"{base}_glossary.md",
        f"{base}_glossary_progress.json",
        f"{base}_gender_tracker.json",
        f"{base}_glossary_history.json",
    ]

    moved = []
    for filename in filenames:
        src = os.path.join(root, filename)
        dst = os.path.join(book_dir, filename)
        if not os.path.isfile(src):
            continue
        if os.path.abspath(src) == os.path.abspath(dst):
            continue
        if os.path.exists(dst):
            if logger:
                logger(f"Legacy glossary migration skipped existing file: {dst}")
            continue
        try:
            shutil.move(src, dst)
            moved.append((src, dst))
            if logger:
                logger(f"Moved legacy glossary file: {os.path.basename(src)} -> {os.path.basename(book_dir)}/")
        except Exception as exc:
            if logger:
                logger(f"Legacy glossary migration failed for {src}: {exc}")
    return moved


def repair_nested_glossary_folder(shared_glossary_dir: str, book_name: str, logger=None):
    """Repair accidental Glossary/<book>/Glossary/<book>/ nesting."""
    root = os.path.abspath(shared_glossary_dir or os.path.join(os.getcwd(), "Glossary"))
    base = glossary_book_base_from_output(book_name)
    book_dir = get_book_glossary_dir(root, base, create=True)
    nested_dir = os.path.join(book_dir, "Glossary", sanitize_glossary_folder_name(base))
    if not os.path.isdir(nested_dir):
        return []

    moved = []
    for filename in os.listdir(nested_dir):
        src = os.path.join(nested_dir, filename)
        dst = os.path.join(book_dir, filename)
        if not os.path.isfile(src):
            continue
        if os.path.exists(dst):
            if logger:
                logger(f"Nested glossary repair skipped existing file: {dst}")
            continue
        try:
            shutil.move(src, dst)
            moved.append((src, dst))
            if logger:
                logger(f"Moved nested glossary file: {filename} -> {os.path.basename(book_dir)}/")
        except Exception as exc:
            if logger:
                logger(f"Nested glossary repair failed for {src}: {exc}")

    for empty_dir in (nested_dir, os.path.dirname(nested_dir)):
        try:
            if os.path.isdir(empty_dir) and not os.listdir(empty_dir):
                os.rmdir(empty_dir)
        except Exception:
            pass
    return moved


def repair_misplaced_glossary_backup(shared_glossary_dir: str, book_name: str, backup_root: str = None, logger=None):
    """Move accidental Glossary/<book>/Glossary_Backup files back to the shared backup folder."""
    root = os.path.abspath(shared_glossary_dir or os.path.join(os.getcwd(), "Glossary"))
    base = glossary_book_base_from_output(book_name)
    book_dir = get_book_glossary_dir(root, base, create=True)
    misplaced_dir = os.path.join(book_dir, "Glossary_Backup")
    if not os.path.isdir(misplaced_dir):
        return []

    target_dir = os.path.abspath(backup_root or os.path.join(os.path.dirname(root), "Glossary_Backup"))
    os.makedirs(target_dir, exist_ok=True)

    moved = []
    for filename in os.listdir(misplaced_dir):
        src = os.path.join(misplaced_dir, filename)
        dst = os.path.join(target_dir, filename)
        if not os.path.isfile(src):
            continue
        if os.path.exists(dst):
            try:
                if os.path.getsize(src) == os.path.getsize(dst):
                    with open(src, "rb") as src_f, open(dst, "rb") as dst_f:
                        if src_f.read() == dst_f.read():
                            os.remove(src)
                            moved.append((src, dst))
                            if logger:
                                logger(f"Removed duplicate misplaced glossary backup: {filename}")
                            continue
            except Exception:
                pass
            if logger:
                logger(f"Misplaced glossary backup repair skipped existing file: {dst}")
            continue
        try:
            shutil.move(src, dst)
            moved.append((src, dst))
            if logger:
                logger(f"Moved misplaced glossary backup: {filename} -> {os.path.basename(target_dir)}/")
        except Exception as exc:
            if logger:
                logger(f"Misplaced glossary backup repair failed for {src}: {exc}")

    try:
        if os.path.isdir(misplaced_dir) and not os.listdir(misplaced_dir):
            os.rmdir(misplaced_dir)
    except Exception:
        pass
    return moved


def migrate_legacy_named_files(shared_dir: str, folder_name: str, filenames, logger=None):
    """Move specific flat files into ``shared_dir/<folder_name>/``.

    This is used by glossary-adjacent stores whose filenames do not use the
    standard ``*_glossary`` suffix, such as manga glossary backups.
    """
    root = os.path.abspath(shared_dir or os.getcwd())
    if not os.path.isdir(root):
        return []

    target_dir = get_book_glossary_dir(root, folder_name, create=True)
    moved = []
    for filename in filenames or []:
        filename = os.path.basename(str(filename or ""))
        if not filename:
            continue
        src = os.path.join(root, filename)
        dst = os.path.join(target_dir, filename)
        if not os.path.isfile(src):
            continue
        if os.path.exists(dst):
            if logger:
                logger(f"Legacy file migration skipped existing file: {dst}")
            continue
        try:
            shutil.move(src, dst)
            moved.append((src, dst))
            if logger:
                logger(f"Moved legacy file: {os.path.basename(src)} -> {os.path.basename(target_dir)}/")
        except Exception as exc:
            if logger:
                logger(f"Legacy file migration failed for {src}: {exc}")
    return moved
