"""glossary_files: the main window's glossary file actions without Qt (Glossarion mobile rewrite, U6).

Shared by ``TranslatorGUI`` (its methods and closures keep the widgets, dialogs and owner state
and call these) and the mobile Glossaries / Library screens. Moved verbatim out of
translator_gui.py (frozen source: ``git show U6_BASE_SHA``; tests/test_glossary_files.py pins
every block and its documented edits), with explicit parameters in place of ``self``:

* ``create_glossary_backup`` + ``_clean_old_backups``: the Glossary Editor's JSON backups
  (``<glossary folder>/Backups/<stem>_<operation>_<timestamp>.json``, pruned to
  ``glossary_max_backups``). The editor's file, its data and the "Continue anyway?" question
  are parameters.
* The 🗑️ / ↩️ closures of the auto-glossary row (``_delete_current_glossary``,
  ``_find_latest_backup``, ``_restore_glossary_backup``): the EPUBs they act on, every
  glossary artifact of a book, moving them to ``Backups/<timestamp>/``, the newest backup and
  its restore. The confirmation boxes, sounds and owner-state resets stay in the closures.
* The data steps of the Map Glossaries to EPUBs dialog (``_open_glossary_mapping_dialog``):
  the drop filter, the prefill lookup, Save's mapping / missing-file check and the Manual
  Glossary Only copy into each EPUB's output folder.
* ``_comprehensive_json_fix`` / ``_analyze_json_errors`` (JSON inputs and Load Glossary).

The auto-load / auto-mapping / output-sync helpers moved in U3 into
``translation_pipeline.GlossaryPipelineMixin``; :func:`guess_glossary_for_input_file` and
:func:`copy_glossary_to_output_folders` run those methods for a config without building a
HeadlessOwner (a bare owner object carries ``config`` / ``base_dir`` / ``append_log``).

Python 3.10 compatible; never imports PySide6, translator_gui or dpi_setup.
"""

import json
import os
import re
import time

from app_paths import _get_app_dir

__all__ = [
    "ALLOWED_GLOSSARY_EXTENSIONS",
    "analyze_json_errors",
    "build_glossary_mapping",
    "clean_old_backups",
    "collect_glossary_files_for_inputs",
    "comprehensive_json_fix",
    "copy_glossary_to_output_folders",
    "copy_mapped_glossaries_to_outputs",
    "create_glossary_backup",
    "delete_glossary_files",
    "find_latest_glossary_backup",
    "glossary_delete_display",
    "guess_glossary_for_input_file",
    "is_allowed_glossary_file",
    "mapped_glossary_for_input",
    "normalize_glossary_drop_path",
    "restore_glossary_backup",
    "selected_glossary_epubs",
]


# ---------------------------------------------------------------------------
# Glossary Editor backups (TranslatorGUI.create_glossary_backup / _clean_old_backups)
# ---------------------------------------------------------------------------


def create_glossary_backup(glossary_path, current_glossary_data, operation_name="manual", *,
                           config, append_log=print, ask_continue=None):
    """Create a backup of the current glossary if auto-backup is enabled

    ``glossary_path`` / ``current_glossary_data`` are the open glossary and its data. After a
    failed backup ``ask_continue("Backup Failed", text) -> bool`` decides whether the caller may
    go ahead (without it: no). Returns True when the caller may go ahead.
    """
    # For manual backups, always proceed. For automatic backups, check the setting.
    if operation_name != "manual" and not config.get('glossary_auto_backup', True):
        return True

    if not current_glossary_data or not glossary_path:
        return True

    try:
        # Get the original glossary file path
        original_path = glossary_path
        original_dir = os.path.dirname(original_path)
        original_name = os.path.basename(original_path)

        # Create backup directory
        backup_dir = os.path.join(original_dir, "Backups")

        # Create directory if it doesn't exist
        try:
            os.makedirs(backup_dir, exist_ok=True)
        except Exception as e:
            append_log(f"⚠️ Failed to create backup directory: {str(e)}")
            return False

        # Generate timestamp-based backup filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup_name = f"{os.path.splitext(original_name)[0]}_{operation_name}_{timestamp}.json"
        backup_path = os.path.join(backup_dir, backup_name)

        # Try to save backup
        with open(backup_path, 'w', encoding='utf-8') as f:
            json.dump(current_glossary_data, f, ensure_ascii=False, indent=2)

        append_log(f"💾 Backup created: {backup_name}")

        # Optional: Clean old backups if more than limit
        max_backups = config.get('glossary_max_backups', 50)
        if max_backups > 0:
            clean_old_backups(backup_dir, original_name, max_backups, append_log)

        return True

    except Exception as e:
        # Log the actual error
        append_log(f"⚠️ Backup failed: {str(e)}")
        # Ask user if they want to continue anyway
        if ask_continue is None:
            return False
        return bool(ask_continue("Backup Failed",
                                 f"Failed to create backup: {str(e)}\n\nContinue anyway?"))


def clean_old_backups(backup_dir, original_name, max_backups, append_log=print):
    """Remove old backups exceeding the limit"""
    try:
        # Find all backups for this glossary
        prefix = os.path.splitext(original_name)[0]
        backups = []

        for file in os.listdir(backup_dir):
            if file.startswith(prefix) and file.endswith('.json'):
                file_path = os.path.join(backup_dir, file)
                backups.append((file_path, os.path.getmtime(file_path)))

        # Sort by modification time (oldest first)
        backups.sort(key=lambda x: x[1])

        # Remove oldest backups if exceeding limit
        while len(backups) > max_backups:
            old_backup = backups.pop(0)
            os.remove(old_backup[0])
            append_log(f"🗑️ Removed old backup: {os.path.basename(old_backup[0])}")

    except Exception as e:
        append_log(f"⚠️ Error cleaning old backups: {str(e)}")


# ---------------------------------------------------------------------------
# 🗑️ Delete / ↩️ Restore glossary files of the selected inputs
# ---------------------------------------------------------------------------


def selected_glossary_epubs(selected_files, get_current_epub_path=None):
    """The EPUBs the 🗑️ / ↩️ buttons act on: the selected EPUBs, else the current EPUB."""
    files = list(selected_files or [])
    epubs = [p for p in files if str(p).lower().endswith('.epub')]
    if not epubs and get_current_epub_path is not None:
        ep = get_current_epub_path()
        if ep:
            epubs = [ep]
    return epubs


def collect_glossary_files_for_inputs(epubs, *, config, guess_glossary=None,
                                      auto_loaded_glossary_path=None, manual_glossary_path=None,
                                      manually_loaded=False):
    """Every glossary artifact of the books, as deduplicated ``[(book base name, path)]``.

    ``_delete_current_glossary``: the shared ``Glossary/`` folder (``Glossary/<book>/`` and
    flat), the book's output folder (``glossary.*``, ``Glossary/<book>_glossary.*``, progress
    files), the glossary auto-mapping would pick (``guess_glossary(epub_path)``; default
    :func:`guess_glossary_for_input_file` for ``config``) and the active auto-mapped glossary
    (``auto_loaded_glossary_path or manual_glossary_path``, unless ``manually_loaded``).
    """
    if guess_glossary is None:
        def guess_glossary(path):
            return guess_glossary_for_input_file(path, config)

    override_dir = os.environ.get('OUTPUT_DIRECTORY') or config.get('output_directory')
    mode = config.get('auto_glossary_mode', 'off').lower()
    is_balanced_full = mode in ('balanced', 'full')  # noqa: F841 - unused, kept from the desktop

    all_files = []  # (book_base, file_path)
    for epub_path in epubs:
        base = os.path.splitext(os.path.basename(epub_path))[0]

        if override_dir:
            _root = os.path.abspath(override_dir)
        else:
            _root = _get_app_dir()

        # 1. Shared Glossary/ folder at root: <root>/Glossary/<base>_glossary.*
        gdir = os.path.join(_root, 'Glossary')
        if os.path.isdir(gdir):
            nested_gdir = os.path.join(gdir, base)
            if os.path.isdir(nested_gdir):
                for ext in ['.csv', '.json', '.txt', '.md']:
                    f = os.path.join(nested_gdir, f"{base}_glossary{ext}")
                    if os.path.exists(f):
                        all_files.append((base, f))
                for fname in [
                    f"{base}_glossary_progress.json",
                    f"{base}_gender_tracker.json",
                    f"{base}_glossary_history.json",
                ]:
                    f = os.path.join(nested_gdir, fname)
                    if os.path.exists(f):
                        all_files.append((base, f))
            for ext in ['.csv', '.json', '.txt', '.md']:
                f = os.path.join(gdir, f"{base}_glossary{ext}")
                if os.path.exists(f):
                    all_files.append((base, f))
            for fname in [
                f"{base}_glossary_progress.json",
                f"{base}_gender_tracker.json",
                f"{base}_glossary_history.json",
            ]:
                pf = os.path.join(gdir, fname)
                if os.path.exists(pf):
                    all_files.append((base, pf))

        # 2. Per-book folder: <root>/<base>/glossary.* and <root>/<base>/Glossary/<base>_glossary.*
        out_dir = os.path.join(_root, base)
        if os.path.isdir(out_dir):
            for ext in ['.csv', '.json', '.txt', '.md']:
                f = os.path.join(out_dir, f"glossary{ext}")
                if os.path.exists(f):
                    all_files.append((base, f))
            # Per-book Glossary subfolder
            book_gdir = os.path.join(out_dir, 'Glossary')
            if os.path.isdir(book_gdir):
                for ext in ['.csv', '.json', '.txt', '.md']:
                    f = os.path.join(book_gdir, f"{base}_glossary{ext}")
                    if os.path.exists(f):
                        all_files.append((base, f))
            # Progress files
            for pname in ['glossary_progress.json', f'{base}_glossary_progress.json']:
                pf = os.path.join(out_dir, pname)
                if os.path.exists(pf):
                    all_files.append((base, pf))

        # 3. Include whatever _guess_glossary_for_input_file would map
        #    (handles fuzzy matching when enabled)
        try:
            _guessed = guess_glossary(epub_path)
            if _guessed and os.path.exists(_guessed):
                _g_norm = os.path.normpath(os.path.abspath(_guessed))
                if _g_norm not in {os.path.normpath(os.path.abspath(fp)) for _, fp in all_files}:
                    all_files.append((base, _guessed))
        except Exception:
            pass

        # 4. Also include the currently active auto-mapped glossary
        _auto_gp = auto_loaded_glossary_path or manual_glossary_path
        if _auto_gp and os.path.exists(_auto_gp) and not manually_loaded:
            _auto_norm = os.path.normpath(os.path.abspath(_auto_gp))
            if _auto_norm not in {os.path.normpath(os.path.abspath(fp)) for _, fp in all_files}:
                all_files.append((base, _auto_gp))

    # Deduplicate by normalized path
    _seen = set()
    _deduped = []
    for bk, fp in all_files:
        _norm = os.path.normpath(os.path.abspath(fp))
        if _norm not in _seen:
            _seen.add(_norm)
            _deduped.append((bk, fp))
    all_files = _deduped
    return all_files


def glossary_delete_display(all_files):
    """The "Delete Glossary" confirmation lines: ``[book]``, then that book's file names."""
    # Group by book for display
    from collections import OrderedDict
    grouped = OrderedDict()
    for bk, fp in all_files:
        grouped.setdefault(bk, []).append(fp)
    display = []
    for bk, fps in grouped.items():
        display.append(f"[{bk}]")
        for fp in fps:
            display.append(f"  {os.path.basename(fp)}")
    return display


def delete_glossary_files(all_files, append_log=print):
    """Move each file into ``<its folder>/Backups/<timestamp>/``; returns the "book/file" names moved."""
    import shutil
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    deleted = []
    for bk, fp in all_files:
        try:
            backup_root = os.path.join(os.path.dirname(fp), 'Backups')
            backup_dir = os.path.join(backup_root, timestamp)
            os.makedirs(backup_dir, exist_ok=True)
            shutil.move(fp, os.path.join(backup_dir, os.path.basename(fp)))
            deleted.append(f"{bk}/{os.path.basename(fp)}")
        except Exception as e:
            append_log(f"⚠️ Failed to delete {fp}: {e}")
    return deleted


def find_latest_glossary_backup(epubs, *, config):
    """Find the latest backup subfolder across the books: ``(folder, file names)`` or ``(None, [])``."""
    try:
        override_dir = os.environ.get('OUTPUT_DIRECTORY') or config.get('output_directory')
        latest_dir = None
        latest_time = ''
        for epub_path in epubs:
            base = os.path.splitext(os.path.basename(epub_path))[0]
            backup_dirs_to_check = []
            if override_dir:
                backup_dirs_to_check.append(os.path.join(os.path.abspath(override_dir), 'Glossary', base, 'Backups'))
                backup_dirs_to_check.append(os.path.join(os.path.abspath(override_dir), 'Glossary', 'Backups'))
                backup_dirs_to_check.append(os.path.join(os.path.abspath(override_dir), base, 'Backups'))
            else:
                backup_dirs_to_check.append(os.path.join('Glossary', base, 'Backups'))
                backup_dirs_to_check.append(os.path.join(_get_app_dir(), 'Glossary', base, 'Backups'))
                backup_dirs_to_check.append(os.path.join('Glossary', 'Backups'))
                backup_dirs_to_check.append(os.path.join(_get_app_dir(), base, 'Backups'))
            for bdir in backup_dirs_to_check:
                if not os.path.isdir(bdir):
                    continue
                for sub in os.listdir(bdir):
                    sub_path = os.path.join(bdir, sub)
                    if os.path.isdir(sub_path) and sub > latest_time:
                        backup_files = [f for f in os.listdir(sub_path) if os.path.isfile(os.path.join(sub_path, f))]
                        if backup_files:
                            latest_time = sub
                            latest_dir = sub_path
        return latest_dir, os.listdir(latest_dir) if latest_dir else []
    except Exception:
        return None, []


def restore_glossary_backup(backup_dir, backup_files, append_log=print):
    """Copy a backup folder's files back beside its ``Backups`` folder; returns the restored names."""
    import shutil
    # Determine where to restore to (parent of Backups dir)
    restore_dir = os.path.dirname(os.path.dirname(backup_dir))
    restored = []
    for fname in backup_files:
        src = os.path.join(backup_dir, fname)
        dst = os.path.join(restore_dir, fname)
        try:
            shutil.copy2(src, dst)
            restored.append(fname)
        except Exception as e:
            append_log(f"⚠️ Failed to restore {fname}: {e}")
    return restored


# ---------------------------------------------------------------------------
# Map Glossaries to EPUBs (TranslatorGUI._open_glossary_mapping_dialog)
# ---------------------------------------------------------------------------

#: Glossary files the mapping dialog accepts (picked or dropped).
ALLOWED_GLOSSARY_EXTENSIONS = {'.json', '.csv', '.txt', '.md'}


def normalize_glossary_drop_path(p: str) -> str:
    """A picked / dropped path, unquoted and absolute ('' when empty)."""
    try:
        p = (p or '').strip().strip('"')
        if not p:
            return ''
        return os.path.normpath(os.path.abspath(p))
    except Exception:
        return (p or '').strip()


def is_allowed_glossary_file(p: str) -> bool:
    """Whether an existing file is a glossary the mapping dialog accepts."""
    try:
        if not p or not os.path.exists(p):
            return False
        return os.path.splitext(p)[1].lower() in ALLOWED_GLOSSARY_EXTENSIONS
    except Exception:
        return False


def mapped_glossary_for_input(existing_map, epub_path):
    """The glossary a saved mapping holds for an EPUB (exact, absolute or normalised key), or None."""
    gp = None
    try:
        key = os.path.normpath(os.path.abspath(epub_path))
        gp = existing_map.get(epub_path) or existing_map.get(key) or existing_map.get(os.path.normpath(epub_path))
    except Exception:
        gp = None
    return gp


def build_glossary_mapping(rows):
    """Save's mapping of the dialog rows ``[(epub path, glossary path text)]``.

    Returns ``(mapping, missing)``: ``{absolute EPUB: absolute glossary}`` for the filled rows
    whose glossary exists, and "book.epub → path" lines for rows that point at missing files
    (the dialog then refuses to save).
    """
    mapping = {}
    missing = []
    for _ep, _text in rows:
        p = _text.strip()
        if not p:
            continue
        if not os.path.exists(p):
            missing.append(f"{os.path.basename(_ep)} → {p}")
            continue
        mapping[os.path.normpath(os.path.abspath(_ep))] = os.path.normpath(os.path.abspath(p))
    return mapping, missing


def copy_mapped_glossaries_to_outputs(mapping, *, config=None, append_log=print):
    """Manual Glossary Only: copy each mapped glossary into its EPUB's output folder.

    The output folder follows the translator / retranslation rules (``OUTPUT_DIRECTORY``,
    ``OUTPUT_DIR``, ``config['output_directory']``, else the book name; created with an empty
    ``translation_progress.json``) and the file becomes ``glossary.csv`` / ``.md`` / ``.json``.
    Returns ``(copied, already in place, failed)``, or None when the copy step itself failed.
    """
    try:
        import shutil as _shutil
        import sys as _sys

        def _resolve_out_dir(_epub_path):
            """Same rules as translator / retranslation GUIs."""
            if not _epub_path:
                return None
            # RPG Maker .exe uses its own GTool_Translation folder; skip
            if _epub_path.lower().endswith('.exe'):
                return None
            _base = os.path.splitext(os.path.basename(_epub_path))[0]
            _override = None
            for _c in (
                os.environ.get('OUTPUT_DIRECTORY'),
                os.environ.get('OUTPUT_DIR'),
                config.get('output_directory') if config is not None else None,
            ):
                if _c is None:
                    continue
                _c = str(_c).strip().strip('"')
                if _c:
                    _override = _c
                    break
            _out = os.path.join(os.path.abspath(_override), _base) if _override else _base
            if _sys.platform == 'darwin' and not os.path.isabs(_out):
                _out = os.path.join(os.path.dirname(os.path.abspath(_epub_path)), _out)
            if not os.path.exists(_out):
                try:
                    os.makedirs(_out, exist_ok=True)
                    _pf = os.path.join(_out, "translation_progress.json")
                    if not os.path.exists(_pf):
                        with open(_pf, 'w', encoding='utf-8') as _f:
                            json.dump(
                                {"chapters": {}, "chapter_chunks": {}, "version": "2.1"},
                                _f, ensure_ascii=False, indent=2,
                            )
                    append_log(f"\U0001F4C1 Created output folder: {_out}")
                except Exception as _e:
                    append_log(f"\u26a0\ufe0f Failed to create output folder for {os.path.basename(_epub_path)}: {_e}")
                    return None
            return _out

        def _target_name_for(_p):
            _ext = os.path.splitext(_p)[1].lower()
            if _ext in ('.csv', '.txt'):
                return 'glossary.csv'
            if _ext == '.md':
                return 'glossary.md'
            if _ext == '.json':
                return 'glossary.json'
            return 'glossary.csv'

        copied = 0
        skipped = 0
        failed = 0
        for _epub_path, _glossary_path in mapping.items():
            try:
                _out = _resolve_out_dir(_epub_path)
                if not _out:
                    failed += 1
                    continue
                _dest = os.path.join(_out, _target_name_for(_glossary_path))
                if os.path.abspath(_glossary_path) == os.path.abspath(_dest):
                    append_log(f"\U0001F4CE Glossary already at output path: {_dest}")
                    skipped += 1
                    continue
                _shutil.copy2(_glossary_path, _dest)
                append_log(f"\U0001F4CE Copied glossary to EPUB output: {_dest}")
                copied += 1
            except Exception as _e:
                failed += 1
                append_log(
                    f"\u26a0\ufe0f Copy failed for {os.path.basename(_epub_path)}: {_e}"
                )
        _parts = []
        if copied:
            _parts.append(f"{copied} copied")
        if skipped:
            _parts.append(f"{skipped} already in place")
        if failed:
            _parts.append(f"{failed} failed")
        _summary = ", ".join(_parts) if _parts else "no changes"
        append_log(
            f"\U0001F4DA Manual Glossary Only: mapping applied to {len(mapping)} EPUB output folder(s): {_summary}"
        )
        return copied, skipped, failed
    except Exception as e:
        append_log(f"\u26a0\ufe0f Failed to copy mapped glossaries to output folders: {e}")
        return None


# ---------------------------------------------------------------------------
# JSON repair (TranslatorGUI._comprehensive_json_fix / _analyze_json_errors)
# ---------------------------------------------------------------------------

def comprehensive_json_fix(content):
    """Apply comprehensive JSON fixes."""
    import re

    # Store original for comparison
    fixed = content

    # 1. Remove BOM if present
    if fixed.startswith('\ufeff'):
        fixed = fixed[1:]

    # 2. Fix common Unicode issues first
    replacements = {
        '"': '"',  # Left smart quote
        '"': '"',  # Right smart quote
        ''': "'",  # Left smart apostrophe
            ''': "'",  # Right smart apostrophe
        '–': '-',  # En dash
        '—': '-',  # Em dash
        '…': '...',  # Ellipsis
        '\u200b': '',  # Zero-width space
        '\u00a0': ' ',  # Non-breaking space
    }
    for old, new in replacements.items():
        fixed = fixed.replace(old, new)

    # 3. Fix trailing commas in objects and arrays
    fixed = re.sub(r',\s*}', '}', fixed)
    fixed = re.sub(r',\s*]', ']', fixed)

    # 4. Fix multiple commas
    fixed = re.sub(r',\s*,+', ',', fixed)

    # 5. Fix missing commas between array/object elements
    # Between closing and opening braces/brackets
    fixed = re.sub(r'}\s*{', '},{', fixed)
    fixed = re.sub(r']\s*\[', '],[', fixed)
    fixed = re.sub(r'}\s*\[', '},[', fixed)
    fixed = re.sub(r']\s*{', '],{', fixed)

    # Between string values (but not inside strings)
    # This is tricky, so we'll be conservative
    fixed = re.sub(r'"\s+"(?=[^:]*":)', '","', fixed)

    # 6. Fix unquoted keys (simple cases)
    # Match unquoted keys that are followed by a colon
    fixed = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', fixed)

    # 7. Fix single quotes to double quotes for keys and simple string values
    # Keys
    fixed = re.sub(r"([{,]\s*)'([^']+)'(\s*:)", r'\1"\2"\3', fixed)
    # Simple string values (be conservative)
    fixed = re.sub(r"(:\s*)'([^'\"]*)'(\s*[,}])", r'\1"\2"\3', fixed)

    # 8. Fix common escape issues
    # Replace single backslashes with double backslashes (except for valid escapes)
    # This is complex, so we'll only fix obvious cases
    fixed = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', fixed)

    # 9. Ensure proper brackets/braces balance
    # Count opening and closing brackets
    open_braces = fixed.count('{')
    close_braces = fixed.count('}')
    open_brackets = fixed.count('[')
    close_brackets = fixed.count(']')

    # Add missing closing braces/brackets at the end
    if open_braces > close_braces:
        fixed += '}' * (open_braces - close_braces)
    if open_brackets > close_brackets:
        fixed += ']' * (open_brackets - close_brackets)

    # 10. Remove trailing comma before EOF
    fixed = re.sub(r',\s*$', '', fixed.strip())

    # 11. Fix unescaped newlines in strings (conservative approach)
    # This is very tricky to do with regex without a proper parser
    # We'll skip this for safety

    # 12. Remove comments (JSON doesn't support comments)
    # Remove // style comments
    fixed = re.sub(r'//.*$', '', fixed, flags=re.MULTILINE)
    # Remove /* */ style comments
    fixed = re.sub(r'/\*.*?\*/', '', fixed, flags=re.DOTALL)

    return fixed


def analyze_json_errors(original, fixed, original_error, fixed_error):
    """Analyze JSON errors and provide helpful information."""
    analysis = []

    # Check for common issues
    if '{' in original and original.count('{') != original.count('}'):
        analysis.append(f"• Mismatched braces: {original.count('{')} opening, {original.count('}')} closing")

    if '[' in original and original.count('[') != original.count(']'):
        analysis.append(f"• Mismatched brackets: {original.count('[')} opening, {original.count(']')} closing")

    if original.count('"') % 2 != 0:
        analysis.append("• Odd number of quotes (possible unclosed string)")

    # Check for BOM
    if original.startswith('\ufeff'):
        analysis.append("• File starts with BOM (Byte Order Mark)")

    # Check for common problematic patterns
    if re.search(r'[''""…]', original):
        analysis.append("• Contains smart quotes or special Unicode characters")

    if re.search(r':\s*[a-zA-Z_][a-zA-Z0-9_]*\s*[,}]', original):
        analysis.append("• Possible unquoted string values")

    if re.search(r'[{,]\s*[a-zA-Z_][a-zA-Z0-9_]*\s*:', original):
        analysis.append("• Possible unquoted keys")

    if '//' in original or '/*' in original:
        analysis.append("• Contains comments (not valid in JSON)")

    # Try to find the approximate error location
    if hasattr(original_error, 'lineno'):
        lines = original.split('\n')
        if 0 < original_error.lineno <= len(lines):
            error_line = lines[original_error.lineno - 1]
            analysis.append(f"\nError near line {original_error.lineno}:")
            analysis.append(f"  {error_line.strip()}")

    return "\n".join(analysis) if analysis else "Unable to determine specific issues."


# ---------------------------------------------------------------------------
# The U3 glossary auto-mapping helpers (translation_pipeline.GlossaryPipelineMixin), owner-free
# ---------------------------------------------------------------------------

_LOOKUP_OWNER_CLASS = None


def _lookup_owner(config, *, base_dir="", append_log=None):
    """A bare owner (no HeadlessOwner init, no env writes) for the mixins' glossary helpers."""
    global _LOOKUP_OWNER_CLASS
    if _LOOKUP_OWNER_CLASS is None:
        from run_env import RunEnvMixin
        from translation_pipeline import GlossaryPipelineMixin

        class _GlossaryFilesOwner(GlossaryPipelineMixin, RunEnvMixin):
            """``config`` / ``base_dir`` / ``append_log`` and the mixin methods, nothing else."""

        _LOOKUP_OWNER_CLASS = _GlossaryFilesOwner
    owner = object.__new__(_LOOKUP_OWNER_CLASS)
    owner.config = config if config is not None else {}
    owner.base_dir = base_dir
    owner.append_log = append_log if append_log is not None else (lambda _message: None)
    return owner


def guess_glossary_for_input_file(input_path, config=None, *, base_dir=""):
    """The glossary auto-mapping picks for an input, or None (``_guess_glossary_for_input_file``).

    Searches ``<output override>/Glossary``, ``<base_dir>/Glossary`` and the app folder's
    ``Glossary``; fuzzy matching follows ``FUZZY_AUTO_MAPPING`` /
    ``FUZZY_AUTO_MAPPING_THRESHOLD`` in the environment, as on the desktop.
    """
    return _lookup_owner(config, base_dir=base_dir)._guess_glossary_for_input_file(input_path)


def copy_glossary_to_output_folders(glossary_path, input_files, *, config=None, append_log=print):
    """Copy a glossary into each input's translation output folder (``_copy_glossary_to_output_folders``).

    Returns the number of folders it was copied into.
    """
    owner = _lookup_owner(config, append_log=append_log)
    return owner._copy_glossary_to_output_folders(glossary_path, input_files=list(input_files or []))
