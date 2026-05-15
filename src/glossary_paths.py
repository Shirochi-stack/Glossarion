# -*- coding: utf-8 -*-
"""Shared helpers for per-book glossary storage."""

import os
import shutil
import threading


_WINDOWS_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
    "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
}

_LEGACY_GLOSSARY_SUFFIXES = (
    "_glossary_progress.json",
    "_glossary_history.json",
    "_gender_tracker.json",
    "_glossary.json",
    "_glossary.csv",
    "_glossary.txt",
    "_glossary.md",
)

_BACKGROUND_MIGRATION_LOCK = threading.Lock()
_BACKGROUND_MIGRATION_THREADS = {}


def _safe_log(logger, message: str):
    if not logger:
        return
    try:
        logger(message)
    except Exception:
        pass


def _files_are_identical(path_a: str, path_b: str) -> bool:
    try:
        if os.path.getsize(path_a) != os.path.getsize(path_b):
            return False
        with open(path_a, "rb") as file_a, open(path_b, "rb") as file_b:
            return file_a.read() == file_b.read()
    except Exception:
        return False


def _unique_legacy_destination(dst: str) -> str:
    if not os.path.exists(dst):
        return dst
    stem, ext = os.path.splitext(dst)
    candidate = f"{stem}_legacy{ext}"
    if not os.path.exists(candidate):
        return candidate
    counter = 2
    while True:
        candidate = f"{stem}_legacy_{counter}{ext}"
        if not os.path.exists(candidate):
            return candidate
        counter += 1


def glossary_book_base_from_output(output_path_or_name: str) -> str:
    """Return the book base name from a glossary output filename/path."""
    raw_name = os.path.basename(str(output_path_or_name or ""))
    lower_name = raw_name.lower()
    for suffix in _LEGACY_GLOSSARY_SUFFIXES:
        if lower_name.endswith(suffix):
            return raw_name[:-len(suffix)] or "book"

    stem, ext = os.path.splitext(raw_name)
    name = stem if ext.lower() in {".json", ".csv", ".txt", ".md"} else raw_name
    for suffix in ("_glossary_progress", "_glossary_history", "_gender_tracker", "_glossary"):
        if name.lower().endswith(suffix):
            return name[:-len(suffix)] or "book"
    return name or "book"


def legacy_glossary_base_from_filename(filename: str) -> str:
    """Return the book base for a flat legacy glossary filename, or empty."""
    name = os.path.basename(str(filename or ""))
    lower_name = name.lower()
    for suffix in _LEGACY_GLOSSARY_SUFFIXES:
        if lower_name.endswith(suffix):
            base = name[:-len(suffix)]
            return base.strip() or ""
    return ""


def legacy_glossary_backup_base_from_filename(filename: str) -> str:
    """Return the book base for a legacy glossary backup filename, or empty."""
    base = legacy_glossary_base_from_filename(filename)
    if base:
        return base
    name = os.path.basename(str(filename or ""))
    marker = "_glossary_"
    idx = name.lower().find(marker)
    if idx > 0:
        return name[:idx].strip()
    return ""


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
            if _files_are_identical(src, dst):
                try:
                    os.remove(src)
                    moved.append((src, dst))
                    _safe_log(logger, f"Removed duplicate legacy glossary file: {os.path.basename(src)}")
                    continue
                except Exception as exc:
                    _safe_log(logger, f"Legacy glossary duplicate cleanup failed for {src}: {exc}")
                    continue
            dst = _unique_legacy_destination(dst)
            _safe_log(logger, f"Legacy glossary migration found existing destination; preserving old file as: {os.path.basename(dst)}")
        try:
            shutil.move(src, dst)
            moved.append((src, dst))
            _safe_log(logger, f"Moved legacy glossary file: {os.path.basename(src)} -> {os.path.basename(book_dir)}/")
        except Exception as exc:
            _safe_log(logger, f"Legacy glossary migration failed for {src}: {exc}")
    return moved


def discover_legacy_glossary_bases(shared_glossary_dir: str):
    """Return every book base that still has flat legacy files in Glossary/."""
    root = os.path.abspath(shared_glossary_dir or os.path.join(os.getcwd(), "Glossary"))
    if not os.path.isdir(root):
        return []

    bases = set()
    for filename in os.listdir(root):
        path = os.path.join(root, filename)
        if not os.path.isfile(path):
            continue
        base = legacy_glossary_base_from_filename(filename)
        if base:
            bases.add(glossary_book_base_from_output(base))
    return sorted(bases, key=lambda value: value.casefold())


def migrate_all_legacy_glossary_files(shared_glossary_dir: str, logger=None, backup_root: str = None):
    """Migrate every flat legacy glossary file and repair all known book folders."""
    root = os.path.abspath(shared_glossary_dir or os.path.join(os.getcwd(), "Glossary"))
    if not os.path.isdir(root):
        return []

    moved = []
    moved.extend(migrate_shared_glossary_backups(root, logger=logger))
    bases = discover_legacy_glossary_bases(root)
    for base in bases:
        moved.extend(migrate_legacy_glossary_files(root, base, logger=logger))

    skip_dirs = {
        "backups",
        "glossary_backup",
        "mangaglossary_backup",
        "truncation_logs",
        "__pycache__",
    }
    for dirname in list(os.listdir(root)):
        dir_path = os.path.join(root, dirname)
        if not os.path.isdir(dir_path):
            continue
        if dirname.strip().lower() in skip_dirs or dirname.startswith("."):
            continue
        moved.extend(repair_nested_glossary_folder(root, dirname, logger=logger))
        moved.extend(repair_misplaced_glossary_backup(root, dirname, logger=logger))

    if moved:
        _safe_log(logger, f"Glossary migration sweep completed: moved/repaired {len(moved)} file(s)")
    return moved


def migrate_shared_glossary_backups(shared_glossary_dir: str, logger=None):
    """Move legacy shared Glossary/Backups files into per-book Backups folders."""
    root = os.path.abspath(shared_glossary_dir or os.path.join(os.getcwd(), "Glossary"))
    backup_dir = os.path.join(root, "Backups")
    if not os.path.isdir(backup_dir):
        return []

    moved = []

    def _move_backup_file(src_path: str, base: str, timestamp_folder: str = None):
        if not base or not os.path.isfile(src_path):
            return
        book_backup_dir = os.path.join(get_book_glossary_dir(root, base, create=True), "Backups")
        if timestamp_folder:
            book_backup_dir = os.path.join(book_backup_dir, os.path.basename(timestamp_folder))
        os.makedirs(book_backup_dir, exist_ok=True)
        dst_path = os.path.join(book_backup_dir, os.path.basename(src_path))
        if os.path.exists(dst_path):
            if _files_are_identical(src_path, dst_path):
                try:
                    os.remove(src_path)
                    moved.append((src_path, dst_path))
                    _safe_log(logger, f"Removed duplicate shared glossary backup: {os.path.basename(src_path)}")
                    return
                except Exception as exc:
                    _safe_log(logger, f"Shared glossary backup duplicate cleanup failed for {src_path}: {exc}")
                    return
            dst_path = _unique_legacy_destination(dst_path)
        try:
            shutil.move(src_path, dst_path)
            moved.append((src_path, dst_path))
            _safe_log(logger, f"Moved shared glossary backup: {os.path.basename(src_path)} -> {base}/Backups/")
        except Exception as exc:
            _safe_log(logger, f"Shared glossary backup migration failed for {src_path}: {exc}")

    for name in list(os.listdir(backup_dir)):
        path = os.path.join(backup_dir, name)
        if os.path.isfile(path):
            base = legacy_glossary_backup_base_from_filename(name)
            _move_backup_file(path, base)
            continue
        if not os.path.isdir(path):
            continue
        for filename in list(os.listdir(path)):
            src_path = os.path.join(path, filename)
            if not os.path.isfile(src_path):
                continue
            base = legacy_glossary_backup_base_from_filename(filename)
            _move_backup_file(src_path, base, timestamp_folder=name)
        try:
            if os.path.isdir(path) and not os.listdir(path):
                os.rmdir(path)
        except Exception:
            pass

    try:
        if os.path.isdir(backup_dir) and not os.listdir(backup_dir):
            os.rmdir(backup_dir)
    except Exception:
        pass

    return moved


def start_background_glossary_migration(shared_glossary_dir: str, logger=None, backup_root: str = None):
    """Start a daemon-thread migration sweep for the shared Glossary folder."""
    root = os.path.abspath(shared_glossary_dir or os.path.join(os.getcwd(), "Glossary"))
    with _BACKGROUND_MIGRATION_LOCK:
        existing = _BACKGROUND_MIGRATION_THREADS.get(root)
        if existing and existing.is_alive():
            return existing

        def _run():
            try:
                migrate_all_legacy_glossary_files(root, logger=logger)
            except Exception as exc:
                _safe_log(logger, f"Glossary migration sweep failed: {exc}")

        thread = threading.Thread(target=_run, name="GlossaryMigrationSweep", daemon=True)
        _BACKGROUND_MIGRATION_THREADS[root] = thread
        thread.start()
        return thread


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
            _safe_log(logger, f"Nested glossary repair skipped existing file: {dst}")
            continue
        try:
            shutil.move(src, dst)
            moved.append((src, dst))
            _safe_log(logger, f"Moved nested glossary file: {filename} -> {os.path.basename(book_dir)}/")
        except Exception as exc:
            _safe_log(logger, f"Nested glossary repair failed for {src}: {exc}")

    for empty_dir in (nested_dir, os.path.dirname(nested_dir)):
        try:
            if os.path.isdir(empty_dir) and not os.listdir(empty_dir):
                os.rmdir(empty_dir)
        except Exception:
            pass
    return moved


def repair_misplaced_glossary_backup(shared_glossary_dir: str, book_name: str, backup_root: str = None, logger=None):
    """Move accidental Glossary/<book>/Glossary_Backup files into book-owned storage."""
    root = os.path.abspath(shared_glossary_dir or os.path.join(os.getcwd(), "Glossary"))
    base = glossary_book_base_from_output(book_name)
    book_dir = get_book_glossary_dir(root, base, create=True)
    misplaced_dir = os.path.join(book_dir, "Glossary_Backup")
    if not os.path.isdir(misplaced_dir):
        return []

    moved = []
    for filename in os.listdir(misplaced_dir):
        src = os.path.join(misplaced_dir, filename)
        if not os.path.isfile(src):
            continue
        file_base = legacy_glossary_base_from_filename(filename)
        if file_base and sanitize_glossary_folder_name(file_base).casefold() == sanitize_glossary_folder_name(base).casefold():
            target_dir = book_dir
        else:
            target_dir = os.path.join(book_dir, "Backups")
        os.makedirs(target_dir, exist_ok=True)
        dst = os.path.join(target_dir, filename)
        if os.path.exists(dst):
            try:
                if _files_are_identical(src, dst):
                    os.remove(src)
                    moved.append((src, dst))
                    _safe_log(logger, f"Removed duplicate misplaced glossary backup: {filename}")
                    continue
            except Exception:
                pass
            dst = _unique_legacy_destination(dst)
        try:
            shutil.move(src, dst)
            moved.append((src, dst))
            _safe_log(logger, f"Moved misplaced glossary backup: {filename} -> {os.path.relpath(target_dir, book_dir)}/")
        except Exception as exc:
            _safe_log(logger, f"Misplaced glossary backup repair failed for {src}: {exc}")

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
            if _files_are_identical(src, dst):
                try:
                    os.remove(src)
                    moved.append((src, dst))
                    _safe_log(logger, f"Removed duplicate legacy file: {os.path.basename(src)}")
                    continue
                except Exception as exc:
                    _safe_log(logger, f"Legacy duplicate cleanup failed for {src}: {exc}")
                    continue
            dst = _unique_legacy_destination(dst)
            _safe_log(logger, f"Legacy file migration found existing destination; preserving old file as: {os.path.basename(dst)}")
        try:
            shutil.move(src, dst)
            moved.append((src, dst))
            _safe_log(logger, f"Moved legacy file: {os.path.basename(src)} -> {os.path.basename(target_dir)}/")
        except Exception as exc:
            _safe_log(logger, f"Legacy file migration failed for {src}: {exc}")
    return moved
