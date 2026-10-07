"""Manga translator file selection, CBZ handling and the tab's GUI-free callbacks.

Shared GUI-free core (Glossarion mobile rewrite, milestone U8). ``MangaTranslationTab``
(manga_integration.py) inherits these classes; the methods were moved there verbatim, so
the desktop and the mobile ``manga_runner.HeadlessMangaRunner`` run the same code:

* :class:`MangaFilesMixin`: the Files tab logic (source roots, process groups split by
  first-level subfolder, 1-based image range, per-file skip keys, add paths / folders /
  CBZ archives, sort, selected-file persistence in ``config['manga_selected_files']``,
  CBZ packaging at the end of a run, per-image output paths).
* :class:`MangaHooksMixin`: the tab callbacks the moved code calls. Four are moved
  verbatim (``_update_progress``, ``_update_current_file``, ``_stop_startup_heartbeat``,
  ``_update_manga_preview_image_list_for_range``: GUI-free or widget-guarded); the rest are
  GUI-free defaults of methods ``MangaTranslationTab`` keeps (Qt) and overrides
  (tests/test_manga_env.py checks every default is overridden on the desktop).
* the module helpers ``_natural_sort_key``, ``_MANGA_SKIP_PREFIX``, ... (manga_integration
  re-exports them); ``_get_app_dir`` honours ``GLOSSARION_DATA_DIR`` through
  ``mobile_runtime.data_dir`` on non-Windows (desktop never sets it).

Rules: Python 3.10 compatible; never import PySide6, translator_gui or dpi_setup (the
``ImageRenderer`` name below imports that Qt module lazily, only where the desktop calls it).
"""

import os
import platform
import re
import sys
from typing import Any, Dict, List, Optional

from mobile_runtime import data_dir

def _get_app_dir() -> str:
    """Return the application's base directory (Windows-safe)."""
    if platform.system() == 'Windows':
        if getattr(sys, 'frozen', False):
            return os.path.dirname(sys.executable)
        return os.path.dirname(os.path.abspath(__file__))
    return data_dir(os.getcwd())

def _manga_cmd_debug_logging_enabled() -> bool:
    return (
        os.environ.get('DEBUG_MODE', '0') == '1'
        or os.environ.get('SHOW_DEBUG_BUTTONS', '0') == '1'
        or os.environ.get('MANGA_DEBUG_MODE', '0') == '1'
        or os.environ.get('DEBUG_SAVE_REQUEST_PAYLOADS_VERBOSE', '0') == '1'
    )

def _manga_cmd_debug_print(*args, **kwargs):
    try:
        first_arg = str(args[0]) if args else ''
        important = any(term in first_arg.lower() for term in ('error', 'failed', 'failure', 'exception', 'warning', 'critical'))
        debug_prefixes = (
            '[FILE_SELECTION]',
            '[FILE_PERSIST]',
            '[SYNC_SELECTION]',
            '[STATE DEBUG]',
            '[STATE]',
        )
        if first_arg.startswith(debug_prefixes) and not important and not _manga_cmd_debug_logging_enabled():
            return
    except Exception:
        pass
    print(*args, **kwargs)

def _translation_run_token_matches(current_token, event_token) -> bool:
    """Return whether a lifecycle event belongs to the active manga run."""
    return event_token is None or event_token == current_token

def _natural_sort_key(text):
    """Generate a key for natural/numerical sorting.
    Converts numeric portions to integers for proper ordering.
    Examples: '1.png', '2.png', '10.png' -> sorted correctly as 1, 2, 10
    """
    def convert(part):
        return (0, int(part)) if part.isdigit() else (1, part.lower())
    return [convert(c) for c in re.split('([0-9]+)', text)]

_MANGA_SKIP_PREFIX = "⏭️ "

def _manga_filename_without_skip_prefix(text: str) -> str:
    """Remove current or legacy skip markers from a displayed filename."""
    value = str(text or '')
    for prefix in (_MANGA_SKIP_PREFIX, "[SKIP] "):
        if value.startswith(prefix):
            return value[len(prefix):]
    return value



class _LazyImageRenderer:
    """``ImageRenderer`` as the moved methods reference it: the desktop's Qt module, imported on
    first attribute access (it is already loaded whenever the desktop tab exists). Without
    PySide6 every attribute raises AttributeError, which the moved call sites already catch."""

    def __getattr__(self, attr):
        try:
            import ImageRenderer as _module
        except ImportError as exc:
            raise AttributeError(f"ImageRenderer is unavailable ({exc}): {attr}") from exc
        return getattr(_module, attr)

    def __repr__(self):
        return "<lazy ImageRenderer>"


ImageRenderer = _LazyImageRenderer()


class FileListShim:
    """The ``QListWidget`` surface the moved file code touches (``file_listbox``), GUI-free.

    Rows are ``owner.selected_files``; only the current row is state.
    """

    def __init__(self, owner):
        self._owner = owner
        self._row = -1
        self._enabled = True

    def count(self):
        return len(getattr(self._owner, 'selected_files', []) or [])

    def currentRow(self):
        return self._row

    def setCurrentRow(self, row):
        self._row = int(row)

    def clear(self):
        self._row = -1

    def blockSignals(self, value):
        return False

    def setEnabled(self, value):
        self._enabled = bool(value)

    def isEnabled(self):
        return self._enabled

    def current_path(self):
        files = list(getattr(self._owner, 'selected_files', []) or [])
        return files[self._row] if 0 <= self._row < len(files) else None


class MangaHooksMixin:
    """Tab callbacks the moved manga code calls (see the module docstring).

    The defaults below are GUI-free; ``MangaTranslationTab`` defines each of them itself
    (Qt widgets) and its own definition wins. ``HeadlessMangaRunner`` overrides ``_log``
    and ``_update_*`` to report to its job host.
    """

    #: Methods MangaTranslationTab overrides with its Qt version (checked by tests).
    GUI_HOOKS = (
        '_log', '_reset_ui_state', '_monitor_translation_output', '_update_manga_image_range_display',
        '_add_manga_file_item', '_rebuild_manga_file_listbox',
    )

    def _log(self, message, level="info"):
        print(message)

    def _reset_ui_state(self, expected_start_token=None):
        """GUI-free half of ``MangaTranslationTab._reset_ui_state``: the run-state flags."""
        if not _translation_run_token_matches(
            getattr(self, '_translation_start_token', None),
            expected_start_token,
        ):
            return False
        try:
            self._stop_startup_heartbeat()
        except Exception:
            pass
        self.is_running = False
        self._graceful_stop_pending = False
        self._translation_startup_pending = False
        self._translation_start_cancel_requested = False
        self._stop_click_times = []
        return True

    def _monitor_translation_output(self, image_path):
        """Desktop: watches the output folder to refresh the preview widget. No-op here."""
        return None

    def _update_manga_image_range_display(self):
        """Desktop: greys out skipped rows in the file list. No-op here."""
        return None

    def _add_manga_file_item(self, filepath):
        """Desktop: appends a QListWidget row. No-op here (rows are ``selected_files``)."""
        return None

    def _rebuild_manga_file_listbox(self, current_path=None):
        """Desktop: rebuilds the QListWidget rows. Keeps the current row on *current_path*."""
        listbox = getattr(self, 'file_listbox', None)
        files = list(getattr(self, 'selected_files', []) or [])
        if listbox is not None:
            if current_path and current_path in files:
                listbox.setCurrentRow(files.index(current_path))
            elif files:
                listbox.setCurrentRow(0)
        self._update_manga_image_range_display()

    def _update_progress(self, current: int, total: int, status: str):
        """Thread-safe progress update"""
        self.update_queue.put(('progress', current, total, status))

    def _update_current_file(self, filename: str):
        """Thread-safe current file update"""
        self.update_queue.put(('current_file', filename))

    def _stop_startup_heartbeat(self):
        """Stop the startup heartbeat spinner"""
        try:
            self._startup_heartbeat_running = False
            # Clear the spinner text immediately
            if hasattr(self, 'progress_label') and self.progress_label:
                self.progress_label.setText("Initializing...")
                self.progress_label.setStyleSheet("color: white;")
        except Exception:
            pass

    def _update_manga_preview_image_list_for_range(self) -> None:
        """Keep the preview thumbnails aligned with the active visible-order range."""
        if not hasattr(self, 'image_preview_widget') or not self.image_preview_widget:
            return
        all_files = list(getattr(self, 'selected_files', []) or [])
        files = list(all_files)
        groups = self._manga_current_process_groups()
        if len(groups) > 1:
            index = self._manga_selected_process_group_index()
            group_files = set(groups[index].get('files', []) or [])
            files = [path for path in files if path in group_files]

        current_path = getattr(self.image_preview_widget, 'current_image_path', None)
        self.image_preview_widget.set_image_list(files)
        try:
            if hasattr(self.image_preview_widget, 'set_skipped_processing_paths'):
                self.image_preview_widget.set_skipped_processing_paths(self._skipped_processing_keys_for_preview())
        except Exception:
            pass

        if not files:
            return
        skipped_keys = self._skipped_processing_keys_for_preview()
        if current_path in files and self._skip_key_for_path(current_path) not in skipped_keys:
            try:
                if hasattr(self.image_preview_widget, '_update_thumbnail_selection'):
                    self.image_preview_widget._update_thumbnail_selection(current_path)
            except Exception:
                pass
            return

        first_path = next((path for path in files if self._skip_key_for_path(path) not in skipped_keys), None)
        if not first_path:
            return
        try:
            if first_path in self.selected_files and hasattr(self, 'file_listbox') and self.file_listbox:
                target_row = self.selected_files.index(first_path)
                if self.file_listbox.currentRow() != target_row:
                    self.file_listbox.setCurrentRow(target_row)
                elif os.path.exists(first_path):
                    self._current_image_path = first_path
                    self.image_preview_widget.load_image(first_path)
            elif os.path.exists(first_path):
                self._current_image_path = first_path
                self.image_preview_widget.load_image(first_path)
        except Exception as exc:
            print(f"[IMAGE_RANGE_PREVIEW] Failed to sync preview: {exc}")


class MangaFilesMixin:
    """MangaTranslationTab's Files-tab logic (moved verbatim, U8)."""

    def _manga_source_root_for_path(self, path: str) -> str:
        """Return the user-facing manga source folder for an image or CBZ-derived image."""
        if not path:
            return ""
        try:
            mapped_cbz = getattr(self, 'cbz_image_to_job', {}).get(path)
            if mapped_cbz:
                path = mapped_cbz
            abs_path = os.path.abspath(path)
            if os.path.isdir(abs_path):
                return abs_path
            folder_hints = list(getattr(self, 'manga_selected_folder_roots', []) or [])
            folder_hints.sort(key=lambda value: len(os.path.abspath(value)), reverse=True)
            for folder in folder_hints:
                try:
                    folder_abs = os.path.abspath(folder)
                    if os.path.commonpath([folder_abs, abs_path]) == folder_abs:
                        return folder_abs
                except Exception:
                    continue
            return os.path.dirname(abs_path)
        except Exception:
            return ""

    def _manga_selected_source_roots(self) -> List[str]:
        """Return unique source roots currently represented by the file list."""
        roots = []
        seen = set()
        for path in getattr(self, 'selected_files', []) or []:
            if not path or not os.path.exists(path):
                continue
            root = self._manga_source_root_for_path(path)
            if not root:
                continue
            norm = os.path.normcase(os.path.abspath(root))
            if norm not in seen:
                seen.add(norm)
                roots.append(root)
        return roots

    def _manga_source_roots_for_paths(self, paths: List[str]) -> List[str]:
        """Return unique source roots for paths being added."""
        roots = []
        seen = set()
        for path in paths or []:
            root = self._manga_source_root_for_path(path)
            if not root:
                continue
            norm = os.path.normcase(os.path.abspath(root))
            if norm not in seen:
                seen.add(norm)
                roots.append(root)
        return roots

    def _manga_split_first_level_subfolders_enabled(self) -> bool:
        try:
            if hasattr(self, 'manga_split_first_level_subfolders_checkbox'):
                return bool(self.manga_split_first_level_subfolders_checkbox.isChecked())
        except Exception:
            pass
        return bool(getattr(
            self,
            'manga_split_first_level_subfolders_value',
            self.main_gui.config.get('manga_split_first_level_subfolders', False)
        ))

    def _same_manga_source_root(self, left: str, right: str) -> bool:
        try:
            return os.path.normcase(os.path.abspath(left)) == os.path.normcase(os.path.abspath(right))
        except Exception:
            return False

    def _manga_process_groups_for_paths(self, paths: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Group selected images into manga runs."""
        paths = list(paths if paths is not None else (getattr(self, 'selected_files', []) or []))
        folder_hints = [
            os.path.abspath(folder)
            for folder in getattr(self, 'manga_selected_folder_roots', []) or []
            if folder and os.path.isdir(folder)
        ]
        if folder_hints:
            folder_hints.sort(key=lambda value: len(os.path.abspath(value)), reverse=True)
            hint_order: List[str] = []
            hint_lookup: Dict[str, str] = {}
            hint_to_files: Dict[str, List[str]] = {}
            leftovers: List[str] = []

            for path in paths:
                abs_path = os.path.abspath(path)
                matched_hint = ""
                for hint in folder_hints:
                    try:
                        if os.path.commonpath([hint, abs_path]) == hint:
                            matched_hint = hint
                            break
                    except Exception:
                        continue
                if not matched_hint:
                    leftovers.append(path)
                    continue
                hint_key = os.path.normcase(matched_hint)
                if hint_key not in hint_to_files:
                    hint_to_files[hint_key] = []
                    hint_lookup[hint_key] = matched_hint
                    hint_order.append(hint_key)
                hint_to_files[hint_key].append(path)

            groups: List[Dict[str, Any]] = []
            split_first_level = self._manga_split_first_level_subfolders_enabled()
            if split_first_level:
                for hint_key in hint_order:
                    hint_root = hint_lookup[hint_key]
                    child_order: List[str] = []
                    child_to_files: Dict[str, List[str]] = {}
                    child_lookup: Dict[str, str] = {}
                    for path in hint_to_files[hint_key]:
                        abs_path = os.path.abspath(path)
                        try:
                            rel = os.path.relpath(abs_path, hint_root)
                            parts = [part for part in rel.split(os.sep) if part and part != '.']
                        except Exception:
                            parts = []
                        child_root = hint_root
                        if len(parts) > 1:
                            child_root = os.path.join(hint_root, parts[0])
                        child_key = os.path.normcase(os.path.abspath(child_root))
                        if child_key not in child_to_files:
                            child_to_files[child_key] = []
                            child_lookup[child_key] = os.path.abspath(child_root)
                            child_order.append(child_key)
                        child_to_files[child_key].append(path)

                    for child_key in child_order:
                        root = child_lookup[child_key]
                        groups.append({
                            'root': root,
                            'name': os.path.basename(os.path.normpath(root)) or root,
                            'files': child_to_files[child_key],
                            'source_roots': [root],
                        })
            else:
                same_parent = False
                if len(hint_order) > 1:
                    try:
                        parents = {
                            os.path.normcase(os.path.abspath(os.path.dirname(os.path.normpath(hint_lookup[key]))))
                            for key in hint_order
                        }
                        same_parent = len(parents) == 1
                    except Exception:
                        same_parent = False

                all_hints_are_leaf_page_folders = bool(hint_order)
                for hint_key in hint_order:
                    hint_root = hint_lookup[hint_key]
                    for path in hint_to_files[hint_key]:
                        try:
                            rel = os.path.relpath(os.path.abspath(path), hint_root)
                            parts = [part for part in rel.split(os.sep) if part and part != '.']
                        except Exception:
                            parts = []
                        if len(parts) > 1:
                            all_hints_are_leaf_page_folders = False
                            break
                    if not all_hints_are_leaf_page_folders:
                        break

                if len(hint_order) > 1 and same_parent and all_hints_are_leaf_page_folders:
                    parent_root = os.path.abspath(os.path.dirname(os.path.normpath(hint_lookup[hint_order[0]])))
                    files = []
                    for hint_key in hint_order:
                        files.extend(hint_to_files[hint_key])
                    groups.append({
                        'root': parent_root,
                        'name': os.path.basename(os.path.normpath(parent_root)) or parent_root,
                        'files': files,
                        'source_roots': [hint_lookup[key] for key in hint_order],
                    })
                else:
                    for hint_key in hint_order:
                        root = hint_lookup[hint_key]
                        groups.append({
                            'root': root,
                            'name': os.path.basename(os.path.normpath(root)) or root,
                            'files': hint_to_files[hint_key],
                            'source_roots': [root],
                        })

            if leftovers:
                leftover_groups: Dict[str, Dict[str, Any]] = {}
                for path in leftovers:
                    root = os.path.dirname(os.path.abspath(path))
                    key = os.path.normcase(root)
                    if key not in leftover_groups:
                        leftover_groups[key] = {
                            'root': root,
                            'name': os.path.basename(os.path.normpath(root)) or root,
                            'files': [],
                            'source_roots': [root],
                        }
                    leftover_groups[key]['files'].append(path)
                groups.extend(leftover_groups.values())
            return groups

        root_order: List[str] = []
        root_to_files: Dict[str, List[str]] = {}
        root_lookup: Dict[str, str] = {}

        for path in paths:
            if not path:
                continue
            root = self._manga_source_root_for_path(path)
            if not root:
                continue
            try:
                root = os.path.abspath(root)
                root_key = os.path.normcase(root)
            except Exception:
                continue
            if root_key not in root_to_files:
                root_to_files[root_key] = []
                root_lookup[root_key] = root
                root_order.append(root_key)
            root_to_files[root_key].append(path)

        if not root_order:
            return []
        if len(root_order) == 1:
            root = root_lookup[root_order[0]]
            return [{
                'root': root,
                'name': os.path.basename(os.path.normpath(root)) or root,
                'files': root_to_files[root_order[0]],
                'source_roots': [root],
            }]

        ancestor_key = None
        for candidate_key in root_order:
            candidate = root_lookup[candidate_key]
            try:
                candidate_abs = os.path.abspath(candidate)
                if all(os.path.commonpath([candidate_abs, root_lookup[key]]) == candidate_abs for key in root_order):
                    ancestor_key = candidate_key
                    break
            except Exception:
                ancestor_key = None
        if ancestor_key:
            root = root_lookup[ancestor_key]
            files = []
            for key in root_order:
                files.extend(root_to_files[key])
            return [{
                'root': root,
                'name': os.path.basename(os.path.normpath(root)) or root,
                'files': files,
                'source_roots': [root_lookup[key] for key in root_order],
            }]

        parent_order: List[str] = []
        parent_to_roots: Dict[str, List[str]] = {}
        parent_lookup: Dict[str, str] = {}
        for root_key in root_order:
            root = root_lookup[root_key]
            parent = os.path.dirname(os.path.normpath(root)) or root
            parent_key = os.path.normcase(os.path.abspath(parent))
            if parent_key not in parent_to_roots:
                parent_to_roots[parent_key] = []
                parent_lookup[parent_key] = os.path.abspath(parent)
                parent_order.append(parent_key)
            parent_to_roots[parent_key].append(root_key)

        groups: List[Dict[str, Any]] = []
        for parent_key in parent_order:
            child_root_keys = parent_to_roots[parent_key]
            if len(child_root_keys) > 1:
                group_root = parent_lookup[parent_key]
            else:
                group_root = root_lookup[child_root_keys[0]]
            group_files: List[str] = []
            for root_key in child_root_keys:
                group_files.extend(root_to_files[root_key])
            groups.append({
                'root': group_root,
                'name': os.path.basename(os.path.normpath(group_root)) or group_root,
                'files': group_files,
                'source_roots': [root_lookup[key] for key in child_root_keys],
            })
        return groups

    def _manga_current_process_groups(self) -> List[Dict[str, Any]]:
        return self._manga_process_groups_for_paths(getattr(self, 'selected_files', []) or [])

    def _manga_selected_process_group_index(self) -> int:
        groups = self._manga_current_process_groups()
        if not groups:
            return 0
        index = int(getattr(self, 'manga_process_group_index', 0) or 0)
        return max(0, min(index, len(groups) - 1))

    def _update_manga_process_group_nav(self) -> None:
        widget = getattr(self, 'manga_process_nav_widget', None)
        combo = getattr(self, 'manga_process_combo', None)
        if not widget or not combo:
            return
        groups = self._manga_current_process_groups()
        show_nav = len(groups) > 1
        widget.setVisible(show_nav)
        index = self._manga_selected_process_group_index()

        combo.blockSignals(True)
        try:
            combo.clear()
            for group in groups:
                count = len(group.get('files', []) or [])
                combo.addItem(f"{group.get('name') or 'Manga'} ({count})", group.get('root', ''))
            if groups:
                combo.setCurrentIndex(index)
        finally:
            combo.blockSignals(False)

        counter = getattr(self, 'manga_process_counter_label', None)
        if counter:
            counter.setText(f"{index + 1} / {len(groups)}" if groups else "0 / 0")
        if hasattr(self, 'manga_process_prev_btn'):
            self.manga_process_prev_btn.setEnabled(show_nav and index > 0)
        if hasattr(self, 'manga_process_next_btn'):
            self.manga_process_next_btn.setEnabled(show_nav and index < len(groups) - 1)
        combo.setToolTip("\n".join(group.get('root', '') for group in groups[:25]))

    def _can_add_manga_paths_from_single_source(self, paths: List[str]) -> bool:
        """Compatibility shim: multi-folder manga selections are supported."""
        return True

    def _filter_manga_paths_to_single_source(self, paths: List[str]) -> List[str]:
        """Return restored paths unchanged; mixed manga roots are valid now."""
        return list(paths or [])

    def _current_manga_source_dir(self) -> str:
        """Return the active manga process root."""
        active_group = getattr(self, '_manga_active_process_group', None)
        if isinstance(active_group, dict) and active_group.get('root'):
            return active_group.get('root', '')
        groups = self._manga_current_process_groups()
        if groups:
            index = self._manga_selected_process_group_index()
            return groups[index].get('root', '')
        return ""

    def _update_manga_loaded_directory_label(self) -> None:
        """Refresh the file-list directory status label."""
        label = getattr(self, 'manga_loaded_directory_label', None)
        if not label:
            return
        groups = self._manga_current_process_groups()
        if not groups:
            label.setText("Loaded directory: none")
            label.setToolTip("")
            return

        index = self._manga_selected_process_group_index()
        source_dir = groups[index].get('root', '')
        prefix = "Loaded directory"
        if len(groups) > 1:
            prefix = f"Loaded directory ({index + 1}/{len(groups)})"
        label.setText(f"{prefix}: {source_dir}")
        label.setToolTip("\n".join(group.get('root', '') for group in groups[:25]))

    def _refresh_manga_selection_status(self, *, allow_autoload: bool = True) -> None:
        """Refresh folder and glossary labels after the selected manga files change."""
        self._update_manga_process_group_nav()
        self._update_manga_loaded_directory_label()
        self._update_manga_image_range_display()
        if allow_autoload:
            self._autoload_manga_glossary_for_selection()
        self._update_manga_glossary_status_label()

    def _parse_manga_image_range(self, total: Optional[int] = None):
        """Parse the 1-based visible image range. Blank means all rows."""
        text = str(getattr(self, 'manga_image_range_value', '') or '').strip()
        if not text and hasattr(self, 'manga_image_range_entry'):
            try:
                text = self.manga_image_range_entry.text().strip()
            except Exception:
                text = ""
        if not text:
            return None, None

        text = re.sub(r'\s*-\s*', '-', text)
        indices = set()
        tokens = [part for part in re.split(r'[\s,]+', text) if part]
        if not tokens:
            return None, None

        for token in tokens:
            match = re.fullmatch(r'(\d*)\s*-\s*(\d*)', token)
            if match:
                start_text, end_text = match.groups()
                if not start_text and not end_text:
                    return None, f"Invalid image range token: {token}"
                start = int(start_text) if start_text else 1
                if end_text:
                    end = int(end_text)
                elif total is not None:
                    end = int(total)
                else:
                    return None, f"Open-ended range needs a loaded file count: {token}"
            else:
                if not token.isdigit():
                    return None, f"Invalid image range token: {token}"
                start = end = int(token)

            if start < 1 or end < 1:
                return None, "Image range uses 1-based row numbers."
            if start > end:
                return None, f"Image range start is after end: {token}"

            if total is not None:
                if start > total:
                    continue
                end = min(end, int(total))
            indices.update(range(start, end + 1))

        if total is not None:
            indices = {idx for idx in indices if 1 <= idx <= int(total)}
        return indices, None

    def _manga_range_filtered_files(self):
        """Return the files selected by the current image range, in visible order."""
        files = list(getattr(self, 'selected_files', []) or [])
        indices, error = self._parse_manga_image_range(len(files))
        if error:
            return [], error
        manual_skipped = self._manual_skipped_processing_keys()
        if indices is None:
            return [path for path in files if self._skip_key_for_path(path) not in manual_skipped], None
        return [
            path for idx, path in enumerate(files, start=1)
            if idx in indices and self._skip_key_for_path(path) not in manual_skipped
        ], None

    def _current_manga_processing_files(self) -> List[str]:
        """Files for the current run; falls back to the full visible list outside a run."""
        run_files = getattr(self, '_manga_processing_files', None)
        if run_files is not None:
            return list(run_files)
        return list(getattr(self, 'selected_files', []) or [])

    def _skip_key_for_path(self, path: str) -> str:
        try:
            return os.path.normcase(os.path.abspath(os.path.normpath(path)))
        except Exception:
            return str(path or '')

    def _manual_skipped_processing_keys(self) -> set:
        skipped = getattr(self, 'skipped_processing_files', set()) or set()
        return {self._skip_key_for_path(path) for path in skipped if path}

    def _is_manually_skipped_processing_file(self, path: str) -> bool:
        return self._skip_key_for_path(path) in self._manual_skipped_processing_keys()

    def _visible_range_skipped_keys(self) -> set:
        files = list(getattr(self, 'selected_files', []) or [])
        indices, error = self._parse_manga_image_range(len(files))
        if error or indices is None:
            return set()
        return {
            self._skip_key_for_path(path)
            for idx, path in enumerate(files, start=1)
            if idx not in indices
        }

    def _skipped_processing_keys_for_preview(self) -> set:
        return self._manual_skipped_processing_keys() | self._visible_range_skipped_keys()

    def _prune_skipped_processing_files(self) -> None:
        valid_keys = {self._skip_key_for_path(path) for path in getattr(self, 'selected_files', []) or []}
        current = self._manual_skipped_processing_keys()
        self.skipped_processing_files = {path for path in current if path in valid_keys}

    def _toggle_skip_processing_for_path(self, path: str) -> None:
        try:
            key = self._skip_key_for_path(path)
            skipped = self._manual_skipped_processing_keys()
            if key in skipped:
                skipped.remove(key)
                label = "processing enabled"
            else:
                skipped.add(key)
                label = "processing skipped"
            self.skipped_processing_files = skipped
            self._update_manga_image_range_display()
            self._update_manga_preview_image_list_for_range()
            self._persist_selected_files()
            self._log(f"{os.path.basename(path)}: {label}", "info")
        except Exception as e:
            print(f"[FILE_SKIP] Failed to toggle skip state: {e}")

    def _move_manga_file_entry(self, path: str, direction: str) -> bool:
        """Move one file in the visible processing order."""
        files = list(getattr(self, 'selected_files', []) or [])
        path_key = self._skip_key_for_path(path)
        source_row = next(
            (
                index for index, candidate in enumerate(files)
                if self._skip_key_for_path(candidate) == path_key
            ),
            -1,
        )
        if source_row < 0 or len(files) < 2:
            return False

        if direction == 'up':
            destination_row = max(0, source_row - 1)
        elif direction == 'down':
            destination_row = min(len(files) - 1, source_row + 1)
        elif direction == 'top':
            destination_row = 0
        elif direction == 'bottom':
            destination_row = len(files) - 1
        else:
            return False
        if destination_row == source_row:
            return False

        moved_path = files.pop(source_row)
        files.insert(destination_row, moved_path)
        self.selected_files = files
        self._manga_file_sort = None
        self._rebuild_manga_file_listbox(current_path=moved_path)
        if hasattr(self, 'image_preview_widget'):
            self._update_manga_preview_image_list_for_range()
        self._persist_selected_files()
        self._log(
            f"Moved {os.path.basename(moved_path)} from row {source_row + 1} "
            f"to row {destination_row + 1}",
            "info",
        )
        return True

    def _add_dropped_manga_paths(self, paths: List[str]) -> None:
        """Add dropped images, CBZ archives, and recursively scanned folders."""
        if not paths or not self._can_add_manga_paths_from_single_source(paths):
            return

        image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'}
        supported_file_extensions = image_extensions | {'.cbz'}
        if not hasattr(self, 'manga_selected_folder_roots'):
            self.manga_selected_folder_roots = []

        existing_folder_keys = {
            os.path.normcase(os.path.abspath(folder))
            for folder in self.manga_selected_folder_roots
            if folder
        }
        files_to_add = []
        seen_file_keys = set()
        dropped_folders = []

        def queue_file(path: str) -> None:
            absolute = os.path.abspath(path)
            if os.path.splitext(absolute)[1].lower() not in supported_file_extensions:
                return
            key = os.path.normcase(absolute)
            if key not in seen_file_keys:
                seen_file_keys.add(key)
                files_to_add.append(absolute)

        for raw_path in paths:
            path = os.path.abspath(raw_path)
            if os.path.isdir(path):
                dropped_folders.append(path)
                folder_key = os.path.normcase(path)
                if folder_key not in existing_folder_keys:
                    existing_folder_keys.add(folder_key)
                    self.manga_selected_folder_roots.append(path)
                for root, dirnames, filenames in os.walk(path):
                    dirnames[:] = [
                        dirname for dirname in dirnames
                        if not dirname.lower().endswith('_translated')
                        and dirname.lower() not in {
                            'glossary', 'ocr text', 'mangaglossary_backup', '__macosx'
                        }
                    ]
                    dirnames.sort(key=_natural_sort_key)
                    for filename in sorted(filenames, key=_natural_sort_key):
                        queue_file(os.path.join(root, filename))
            elif os.path.isfile(path):
                queue_file(path)

        added_count = 0
        first_added_path = None
        for path in files_to_add:
            extension = os.path.splitext(path)[1].lower()
            if extension == '.cbz':
                previous_count = len(self.selected_files)
                added_count += self._add_cbz_archive_images(path, image_extensions)
                if first_added_path is None and len(self.selected_files) > previous_count:
                    first_added_path = self.selected_files[previous_count]
            elif path not in self.selected_files:
                self.selected_files.append(path)
                self._add_manga_file_item(path)
                added_count += 1
                if first_added_path is None:
                    first_added_path = path

        self._apply_manga_file_sort()
        if first_added_path and self.file_listbox.currentRow() < 0:
            try:
                self.file_listbox.setCurrentRow(self.selected_files.index(first_added_path))
            except ValueError:
                pass

        self._update_manga_image_range_display()
        if hasattr(self, 'image_preview_widget'):
            self._update_manga_preview_image_list_for_range()
        self._persist_selected_files()

        source_count = len(paths)
        folder_count = len(dropped_folders)
        if added_count:
            self._log(
                f"📥 Added {added_count} manga images from {source_count} dropped item(s)"
                + (f" ({folder_count} folder(s))" if folder_count else ""),
                "success",
            )
        else:
            self._log("No new supported manga images were found in the dropped items", "warning")

    def _ensure_cbz_temp_root(self) -> Optional[str]:
        cbz_temp_root = getattr(self, 'cbz_temp_root', None)
        if cbz_temp_root is None:
            try:
                import tempfile
                cbz_temp_root = tempfile.mkdtemp(prefix='glossarion_cbz_')
                self.cbz_temp_root = cbz_temp_root
            except Exception:
                cbz_temp_root = None
        return cbz_temp_root

    def _add_cbz_archive_images(self, path: str, image_extensions: set) -> int:
        """Extract a CBZ and append its images to the current selection."""
        try:
            import zipfile
            cbz_temp_root = self._ensure_cbz_temp_root()
            base = os.path.splitext(os.path.basename(path))[0]
            extract_dir = os.path.join(cbz_temp_root or os.path.dirname(path), base)
            os.makedirs(extract_dir, exist_ok=True)
            with zipfile.ZipFile(path, 'r') as zf:
                zf.extractall(extract_dir)
            if not hasattr(self, 'cbz_jobs'):
                self.cbz_jobs = {}
            if not hasattr(self, 'cbz_image_to_job'):
                self.cbz_image_to_job = {}
            out_dir = os.path.join(os.path.dirname(path), f"{base}_translated")
            self.cbz_jobs[path] = {
                'extract_dir': extract_dir,
                'out_dir': out_dir,
            }
            added = 0
            for root, _, files_in_dir in os.walk(extract_dir):
                for fn in sorted(files_in_dir, key=_natural_sort_key):
                    if os.path.splitext(fn)[1].lower() in image_extensions:
                        target_path = os.path.join(root, fn)
                        if target_path not in self.selected_files:
                            self.selected_files.append(target_path)
                            self._add_manga_file_item(target_path)
                            added += 1
                        self.cbz_image_to_job[target_path] = path
            self._log(f"📦 Added {added} images from CBZ: {os.path.basename(path)}", "info")
            return added
        except Exception as e:
            self._log(f"❌ Failed to read CBZ {os.path.basename(path)}: {e}", "error")
            return 0

    def _apply_manga_file_sort(self):
        """Apply the active sort after loading images without logging a manual action."""
        sort_state = getattr(self, '_manga_file_sort', ('numeric', False))
        if not sort_state or not self.selected_files:
            return
        sort_type, reverse = sort_state
        if sort_type == 'numeric':
            key = lambda path: _natural_sort_key(os.path.basename(path))
        elif sort_type == 'name':
            key = lambda path: os.path.basename(path).lower()
        elif sort_type == 'date':
            key = lambda path: os.path.getmtime(path) if os.path.exists(path) else 0
        else:
            return
        previous_order = list(self.selected_files)
        self.selected_files.sort(key=key, reverse=reverse)
        if self.selected_files != previous_order:
            self._rebuild_manga_file_listbox()

    def _sort_files(self, sort_type, reverse=False):
        """Sort the file list according to the specified type.
        
        Args:
            sort_type: One of 'name', 'numeric', 'date', 'reverse'
            reverse: If True, reverse the sort order (descending)
        """
        if not self.selected_files:
            return
        
        # Remember current selection
        current_row = self.file_listbox.currentRow()
        current_path = None
        if 0 <= current_row < len(self.selected_files):
            current_path = self.selected_files[current_row]
        
        # Persist current image state before sorting
        try:
            if hasattr(self, '_current_image_path') and self._current_image_path:
                ImageRenderer._persist_current_image_state(self)
        except Exception:
            pass
        
        # Sort based on type
        if sort_type == 'reverse':
            self.selected_files.reverse()
        elif sort_type == 'name':
            # Alphabetical sort by filename
            self.selected_files.sort(key=lambda x: os.path.basename(x).lower(), reverse=reverse)
        elif sort_type == 'numeric':
            # Natural/numerical sort
            self.selected_files.sort(key=lambda x: _natural_sort_key(os.path.basename(x)), reverse=reverse)
        elif sort_type == 'date':
            # Sort by file modification date
            self.selected_files.sort(key=lambda x: os.path.getmtime(x) if os.path.exists(x) else 0, reverse=reverse)

        self._manga_file_sort = None if sort_type == 'reverse' else (sort_type, reverse)
        
        # Rebuild the listbox and restore selection to the same file if possible.
        self._rebuild_manga_file_listbox(current_path=current_path)
        
        # Update thumbnail preview list
        if hasattr(self, 'image_preview_widget'):
            self._update_manga_preview_image_list_for_range()
        
        # Log the action
        direction = 'descending' if reverse else 'ascending'
        sort_names = {'name': 'alphabetically', 'numeric': 'numerically', 'date': 'by date', 'reverse': 'in reverse'}
        if sort_type != 'reverse':
            self._log(f"📋 Sorted {len(self.selected_files)} files {sort_names.get(sort_type, sort_type)} ({direction})", "info")
        else:
            self._log(f"📋 Sorted {len(self.selected_files)} files {sort_names.get(sort_type, sort_type)}", "info")
        
        # Persist the new order
        self._persist_selected_files()

    def _persist_selected_files(self):
        """Save the current selected_files list to config for persistence across restarts."""
        print = _manga_cmd_debug_print
        try:
            if not hasattr(self, 'main_gui') or not self.main_gui:
                return
            # Only save files that still exist
            valid_files = [f for f in self.selected_files if os.path.exists(f)]
            self.main_gui.config['manga_selected_files'] = valid_files
            self._prune_skipped_processing_files()
            valid_keys = {self._skip_key_for_path(path) for path in valid_files}
            self.main_gui.config['manga_skipped_processing_files'] = sorted(
                key for key in self._manual_skipped_processing_keys()
                if key in valid_keys
            )
            folder_roots = [
                folder for folder in getattr(self, 'manga_selected_folder_roots', []) or []
                if folder and os.path.isdir(folder)
            ]
            self.main_gui.config['manga_selected_folder_roots'] = folder_roots
            self._refresh_manga_selection_status()
            if hasattr(self.main_gui, 'save_config'):
                self.main_gui.save_config(show_message=False)
            print(f"[FILE_PERSIST] Saved {len(valid_files)} files to config")
        except Exception as e:
            print(f"[FILE_PERSIST] Error saving files: {e}")

    def _load_persisted_files(self):
        """Load the persisted selected_files list from config on startup."""
        print = _manga_cmd_debug_print
        try:
            if not hasattr(self, 'main_gui') or not self.main_gui:
                return
            saved_files = self.main_gui.config.get('manga_selected_files', [])
            if not saved_files:
                return
            
            # Filter to only files that still exist
            valid_files = [f for f in saved_files if os.path.exists(f)]
            valid_files = self._filter_manga_paths_to_single_source(valid_files)
            if not valid_files:
                return
            valid_keys = {self._skip_key_for_path(path) for path in valid_files}
            saved_skipped = self.main_gui.config.get('manga_skipped_processing_files', [])
            if isinstance(saved_skipped, list):
                self.skipped_processing_files = {
                    self._skip_key_for_path(path)
                    for path in saved_skipped
                    if self._skip_key_for_path(path) in valid_keys
                }
            saved_folder_roots = self.main_gui.config.get('manga_selected_folder_roots', [])
            if isinstance(saved_folder_roots, list):
                self.manga_selected_folder_roots = [
                    os.path.abspath(folder)
                    for folder in saved_folder_roots
                    if folder and os.path.isdir(folder)
                ]
            
            print(f"[FILE_PERSIST] Loading {len(valid_files)} persisted files")
            
            # Add files to the list
            for filepath in valid_files:
                if filepath not in self.selected_files:
                    self.selected_files.append(filepath)
                    self._add_manga_file_item(filepath)
            self._apply_manga_file_sort()
            
            # Auto-select first image
            if self.file_listbox.count() > 0:
                self.file_listbox.setCurrentRow(0)
            
            # Update thumbnail preview list
            if hasattr(self, 'image_preview_widget'):
                self._update_manga_preview_image_list_for_range()
            self._update_manga_image_range_display()
            self._refresh_manga_selection_status()
            
            self._log(f"📂 Restored {len(valid_files)} images from previous session", "info")
        except Exception as e:
            print(f"[FILE_PERSIST] Error loading files: {e}")

    def _create_cbz_from_isolated_folders(self):
        """Create a single CBZ file from all isolated *_translated folders"""
        import zipfile
        
        source_files = self._current_manga_processing_files()
        if not source_files:
            raise FileNotFoundError("No images loaded. Please load some images first.")
        
        try:
            
            # Get parent directory - respect OUTPUT_DIRECTORY override
            first_file = source_files[0]
            
            # Check for output directory override
            override_dir = None
            if hasattr(self, 'main_gui') and self.main_gui and hasattr(self.main_gui, 'config'):
                override_dir = self.main_gui.config.get('output_directory', '')
            if not override_dir:
                override_dir = os.environ.get('OUTPUT_DIRECTORY', '')
            
            if override_dir and os.path.isdir(override_dir):
                parent_dir = override_dir
            else:
                parent_dir = os.path.dirname(first_file)
            
            # Find all *_translated folders
            translated_folders = []
            allowed_folders = {
                f"{os.path.splitext(os.path.basename(path))[0]}_translated"
                for path in source_files
            }
            for item in os.listdir(parent_dir):
                item_path = os.path.join(parent_dir, item)
                if os.path.isdir(item_path) and item.endswith('_translated') and item in allowed_folders:
                    translated_folders.append(item_path)
            
            if not translated_folders:
                self._log("⚠️ No translated folders found for CBZ creation", "warning")
                raise FileNotFoundError("No translated images found. Please translate some images first.")
            
            # Create CBZ filename based on parent folder name
            parent_folder_name = os.path.basename(parent_dir)
            cbz_filename = f"{parent_folder_name}_translated.cbz"
            cbz_path = os.path.join(parent_dir, cbz_filename)
            
            # Counter for images
            image_count = 0
            image_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif')
            
            # Create CBZ (ZIP) file - only include the main translated images, not _cleaned versions
            with zipfile.ZipFile(cbz_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                # Sort folders to maintain order
                for folder in sorted(translated_folders):
                    # Get all images from this folder
                    for filename in sorted(os.listdir(folder)):
                        if filename.lower().endswith(image_extensions):
                            # Skip _cleaned versions - only include the main translated images
                            if '_cleaned' in filename.lower():
                                continue
                            src_path = os.path.join(folder, filename)
                            # Add to CBZ with just the filename (flat structure)
                            zf.write(src_path, filename)
                            image_count += 1
            
            self._log(f"📦 Created CBZ file with {image_count} images: {cbz_filename}", "success")
            
        except Exception as e:
            self._log(f"❌ Error creating CBZ file: {str(e)}", "error")
            import traceback
            self._log(traceback.format_exc(), "debug")
            # Re-raise so button handler can show error state
            raise

    def _finalize_cbz_jobs(self):
        """Package translated outputs back into .cbz for each imported CBZ.
        - Always creates a CLEAN archive with only final translated pages.
        - If save_intermediate is enabled in settings, also creates a DEBUG archive that
          contains the same final pages at root plus debug/raw artifacts under subfolders.
        """
        try:
            if not hasattr(self, 'cbz_jobs') or not self.cbz_jobs:
                return
            import zipfile
            active_run_files = set(self._current_manga_processing_files())
            # Read debug flag from settings
            save_debug = False
            try:
                save_debug = bool(self.main_gui.config.get('manga_settings', {}).get('advanced', {}).get('save_intermediate', False))
            except Exception:
                save_debug = False
            image_exts = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif')
            text_exts = ('.txt', '.json', '.csv', '.log')
            excluded_patterns = ('_mask', '_overlay', '_debug', '_raw', '_ocr', '_regions', '_chunk', '_clean', '_cleaned', '_inpaint', '_inpainted')

            for cbz_path, job in self.cbz_jobs.items():
                out_dir = job.get('out_dir')
                if not out_dir or not os.path.isdir(out_dir):
                    continue
                parent = os.path.dirname(cbz_path)
                base = os.path.splitext(os.path.basename(cbz_path))[0]

                # Compute original basenames from extracted images mapping
                original_basenames = set()
                try:
                    if hasattr(self, 'cbz_image_to_job'):
                        for img_path, job_path in self.cbz_image_to_job.items():
                            if active_run_files and img_path not in active_run_files:
                                continue
                            if job_path == cbz_path:
                                original_basenames.add(os.path.basename(img_path))
                except Exception:
                    pass

                # 1) CLEAN ARCHIVE: only final translated images from *_translated folders
                clean_zip = os.path.join(parent, f"{base}_translated.cbz")
                clean_count = 0
                
                # Look specifically in *_translated subfolders for final images
                translated_images = []
                if os.path.isdir(out_dir):
                    for item in os.listdir(out_dir):
                        item_path = os.path.join(out_dir, item)
                        if os.path.isdir(item_path) and item.endswith('_translated'):
                            # This is a *_translated folder, collect its images
                            for fn in os.listdir(item_path):
                                if fn.lower().endswith(image_exts):
                                    fp = os.path.join(item_path, fn)
                                    fn_lower = fn.lower()
                                    # Skip debug artifacts
                                    if any(p in fn_lower for p in excluded_patterns):
                                        continue
                                    if original_basenames and fn not in original_basenames:
                                        continue
                                    translated_images.append((fp, fn))
                
                # If no *_translated folders found, fall back to root level images in out_dir
                if not translated_images:
                    for fn in os.listdir(out_dir) if os.path.isdir(out_dir) else []:
                        fp = os.path.join(out_dir, fn)
                        if os.path.isfile(fp) and fn.lower().endswith(image_exts):
                            fn_lower = fn.lower()
                            # Skip debug artifacts and only include if matches original basenames (if available)
                            if any(p in fn_lower for p in excluded_patterns):
                                continue
                            if original_basenames and fn not in original_basenames:
                                continue
                            translated_images.append((fp, fn))
                
                with zipfile.ZipFile(clean_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
                    for fp, fn in sorted(translated_images, key=lambda x: x[1]):
                        zf.write(fp, fn)  # place at root with page filename
                        clean_count += 1
                self._log(f"📦 Compiled CLEAN {clean_count} pages into {os.path.basename(clean_zip)}", "success")

                # 2) DEBUG ARCHIVE: include final pages + extras under subfolders
                if save_debug:
                    debug_zip = os.path.join(parent, f"{base}_translated_debug.cbz")
                    dbg_count = 0
                    raw_count = 0
                    page_count = 0
                    
                    # Helper to iterate all files in out_dir for debug archive
                    all_files = []
                    if os.path.isdir(out_dir):
                        for root, _, files in os.walk(out_dir):
                            for fn in files:
                                fp = os.path.join(root, fn)
                                rel = os.path.relpath(fp, out_dir)
                                all_files.append((fp, rel, fn))
                    
                    with zipfile.ZipFile(debug_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
                        # Add final translated pages at root (reuse the same translated_images list)
                        for fp, fn in translated_images:
                            zf.write(fp, fn)
                            page_count += 1
                        
                        # Add all other files under appropriate subfolders
                        for fp, rel, fn in all_files:
                            fn_lower = fn.lower()
                            
                            # Skip files already added as final pages
                            if any(fp == tfp for tfp, _ in translated_images):
                                continue
                                
                            # Raw text/logs
                            if fn_lower.endswith(text_exts):
                                zf.write(fp, os.path.join('raw', rel))
                                raw_count += 1
                                continue
                            # Other images or artifacts -> debug/
                            zf.write(fp, os.path.join('debug', rel))
                            dbg_count += 1
                    self._log(f"📦 Compiled DEBUG archive: pages={page_count}, debug_files={dbg_count}, raw={raw_count} -> {os.path.basename(debug_zip)}", "info")
        except Exception as e:
            self._log(f"⚠️ Failed to compile CBZ packages: {e}", "warning")

    def _get_manga_output_path_for_file(self, filepath: str) -> str:
        """Mirror the normal per-image output path routing used by the worker."""
        filename = os.path.basename(filepath)
        try:
            if hasattr(self, 'cbz_image_to_job') and filepath in self.cbz_image_to_job:
                cbz_file = self.cbz_image_to_job[filepath]
                job = getattr(self, 'cbz_jobs', {}).get(cbz_file)
                if job:
                    output_dir = job.get('out_dir')
                    os.makedirs(output_dir, exist_ok=True)
                    return os.path.join(output_dir, filename)
        except Exception:
            pass

        base_name = os.path.splitext(filename)[0]
        parent_dir = os.environ.get('OUTPUT_DIRECTORY') or os.path.dirname(filepath)
        output_dir = os.path.join(parent_dir, f"{base_name}_translated")
        os.makedirs(output_dir, exist_ok=True)
        return os.path.join(output_dir, filename)


__all__ = [
    "FileListShim",
    "ImageRenderer",
    "MangaFilesMixin",
    "MangaHooksMixin",
    "_MANGA_SKIP_PREFIX",
    "_get_app_dir",
    "_manga_cmd_debug_logging_enabled",
    "_manga_cmd_debug_print",
    "_manga_filename_without_skip_prefix",
    "_natural_sort_key",
    "_translation_run_token_matches",
]
