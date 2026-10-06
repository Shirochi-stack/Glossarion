"""Glossary Progress core: the GUI-free half of the Progress Manager's Glossary
Progress panel (mobile milestone U5).

Moved verbatim from ``git show 20b446b0:src/Retranslation_GUI.py`` (RG line numbers):

* module helpers RG 1403-1491 (filename keys, parallel-source chapter maps, zero-based
  index mapping) and 1875-2136, 2152-2169 (legend statistics, refinement rows);
* ``glossary_progress_locator``: the progress/glossary file lookup closures of
  ``_force_retranslation_epub_or_text`` (RG 22030-22110, 22136-22159, 22543-22649);
* ``make_glossary_progress_model``: the data closures of ``_build_gp_panel``
  (RG 23524-24546, 24872-24896, 24932-25139, 25287-25412, 25575-25799, 26413,
  26443-26481) plus the data halves of the usage-input loader (25243-25285), the
  footnote collection (25806-25842) and Remove from progress (26165-26236).

``_build_gp_panel`` builds the model and binds the same names, so the desktop panel
runs this code.  The two glossary-progress writes (Mark as Completed, Remove from
progress) now run under the extractor's progress-file lock and replace the file
atomically (``glossary_refinement.locked_progress_file``; DISCREPANCIES U5).

Rules: Python 3.10 compatible; never import PySide6, translator_gui or dpi_setup.
"""

import html as html_lib
import json
import os
import re
import tempfile
import threading
import time
import types
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field as dataclass_field
from typing import Any, List, Optional

from chapter_display_numbering import nonreset_chapter_display_numbers
from epub_package import find_epub_opf_member
from glossary_usage import (
    CHECK_PREFIX,
    WARNING_PREFIX,
    _is_special_basename,
    build_chapter_footnote,
    parse_glossary_file,
    read_translated_output_text,
    read_epub_spine_chapters,
)
from progress_core import (
    _format_qa_issue_for_progress_display,
    _progress_entry_model_for_display,
    _progress_entry_refined_for_display,
    _progress_entry_refinement_failed_for_display,
    _progress_status_hides_model_for_display,
    _select_progress_entry_for_display,
)


def _glossary_progress_filename_keys(name):
    """Return filename keys shared by glossary progress/index matching."""
    base = os.path.basename(str(name or ""))
    if not base:
        return set()
    stem = os.path.splitext(base)[0]
    keys = {base.lower(), stem.lower()}
    if stem.lower().startswith('response_'):
        keys.add(stem[9:].lower())
    return {key for key in keys if key}


def _filter_glossary_source_chapter_map(
    chapter_map, spine_index_map, source_filenames
):
    """Restrict a raw EPUB chapter map to the mapped parallel-source files."""

    requested = [str(name or "") for name in (source_filenames or []) if name]
    if not requested:
        return dict(chapter_map or {}), dict(spine_index_map or {})

    chapter_map = dict(chapter_map or {})
    spine_index_map = dict(spine_index_map or {})
    key_to_indices = {}
    for source_index, filename in sorted(chapter_map.items()):
        for key in _glossary_progress_filename_keys(filename):
            key_to_indices.setdefault(key, []).append(source_index)

    used_indices = set()
    filtered_map = {}
    filtered_spine_map = {}
    for mapped_index, requested_filename in enumerate(requested):
        source_index = None
        for key in _glossary_progress_filename_keys(requested_filename):
            source_index = next(
                (
                    index
                    for index in key_to_indices.get(key, [])
                    if index not in used_indices
                ),
                None,
            )
            if source_index is not None:
                break
        if source_index is None:
            filtered_map[mapped_index] = os.path.basename(requested_filename)
            filtered_spine_map[mapped_index] = mapped_index + 1
            continue
        used_indices.add(source_index)
        filtered_map[mapped_index] = chapter_map[source_index]
        filtered_spine_map[mapped_index] = spine_index_map.get(
            source_index, source_index + 1
        )

    return filtered_map, filtered_spine_map


def _parallel_glossary_progress_filename_aliases(source_filenames):
    """Map generated pair_NNNN chapter names onto displayed raw rows."""

    aliases = {}
    for mapped_index, source_filename in enumerate(source_filenames or []):
        if not source_filename:
            continue
        generated_filename = f"pair_{mapped_index + 1:04d}.xhtml"
        for key in _glossary_progress_filename_keys(generated_filename):
            aliases[key] = mapped_index
    return aliases


def _map_zero_based_glossary_progress_index(value, progress_data, filename_key_to_index):
    """Map an extraction-list index onto the full OPF progress-manager view."""
    try:
        index = int(value)
    except (TypeError, ValueError):
        return None
    progress_data = progress_data if isinstance(progress_data, dict) else {}
    chapter_filenames = progress_data.get('chapter_filenames', {})
    progress_filename = None
    if isinstance(chapter_filenames, dict):
        progress_filename = chapter_filenames.get(str(index))
        if progress_filename is None:
            progress_filename = chapter_filenames.get(index)
    if progress_filename:
        for filename_key in _glossary_progress_filename_keys(progress_filename):
            if filename_key in filename_key_to_index:
                return filename_key_to_index[filename_key]
        return None
    return index


def _combine_glossary_progress_legend_stats(
    chapter_stats,
    refinement_status_counts,
):
    """Combine chapter and refinement rows into the displayed legend totals."""
    chapter_stats = chapter_stats if isinstance(chapter_stats, dict) else {}
    refinement_status_counts = (
        refinement_status_counts
        if isinstance(refinement_status_counts, dict)
        else {}
    )

    def _count(mapping, key):
        try:
            return max(0, int(mapping.get(key, 0) or 0))
        except (TypeError, ValueError):
            return 0

    normalized_refinement = {}
    for raw_status, raw_count in refinement_status_counts.items():
        status = str(raw_status or 'unknown').strip().lower().replace(' ', '_')
        try:
            count = max(0, int(raw_count or 0))
        except (TypeError, ValueError):
            count = 0
        normalized_refinement[status] = (
            normalized_refinement.get(status, 0) + count
        )

    result = {
        key: _count(chapter_stats, key)
        for key in (
            'total',
            'completed',
            'skipped',
            'in_progress',
            'failed',
            'merged',
            'remaining',
        )
    }
    result['total'] += sum(normalized_refinement.values())
    result['completed'] += (
        normalized_refinement.get('completed', 0)
        + normalized_refinement.get('refined', 0)
    )
    result['skipped'] += normalized_refinement.get('skipped', 0)
    result['in_progress'] += (
        normalized_refinement.get('in_progress', 0)
        + normalized_refinement.get('partially_in_progress', 0)
    )
    # A non-chapter row that has not run yet (the Minimal pass before it
    # starts) is outstanding work, so it belongs in the same bucket the
    # legend shows as "Not Translated".
    result['remaining'] += normalized_refinement.get('not_completed', 0)
    result['not_refined'] = normalized_refinement.get('not_refined', 0)
    result['refine_failed'] = (
        normalized_refinement.get('refine_failed', 0)
        + normalized_refinement.get('failed', 0)
        + normalized_refinement.get('error', 0)
    )
    return result


def _glossary_refinement_type_key(value):
    """Return a comparison key that accepts section-heading plurals."""
    key = str(value or '').strip().casefold()
    if key in ('term', 'terms'):
        return 'terms'
    if len(key) > 3 and key.endswith('ies'):
        return key[:-3] + 'y'
    if key.endswith(('sses', 'xes', 'ches', 'shes', 'zes')):
        return key[:-2]
    if len(key) > 1 and key.endswith('s') and not key.endswith(('ss', 'us', 'is')):
        return key[:-1]
    return key


def _merge_glossary_refinement_row_info(expected_info, saved_info):
    """Reconcile persisted refinement state with the live glossary counts."""
    expected = dict(expected_info) if isinstance(expected_info, dict) else {}
    saved = dict(saved_info) if isinstance(saved_info, dict) else None
    represented = saved is not None
    info = dict(expected)
    if saved is not None:
        info.update(saved)

    expected_count = expected.get('entry_count_before')
    try:
        expected_count = int(expected_count or 0)
    except (TypeError, ValueError):
        expected_count = 0
    saved_status = str((saved or {}).get('status') or '').strip().lower()
    saved_reason = str((saved or {}).get('reason') or '').strip().lower()

    def _zero_count(value):
        try:
            return int(value or 0) == 0
        except (TypeError, ValueError):
            return False

    saved_no_entries = saved_reason == 'no_entries' or (
        saved_status == 'skipped'
        and _zero_count((saved or {}).get('entry_count_before'))
        and _zero_count((saved or {}).get('entry_count_after'))
    )

    if expected_count > 0 and saved_no_entries:
        # The former empty-state record is no longer a valid refinement result.
        info = dict(expected)
        represented = False
    elif str(expected.get('reason') or '').strip().lower() == 'no_entries':
        # Live emptiness always wins over a stale completed/in-progress record.
        info.update(expected)

    if 'current_entry_count' in expected:
        # This is live glossary state, never historical progress metadata.
        info['current_entry_count'] = expected.get('current_entry_count')
    info['_has_saved_progress'] = represented
    return info


def _glossary_refinement_row_detail(info, status):
    """Format live entry totals plus useful run-specific progress details."""
    info = info if isinstance(info, dict) else {}
    status = str(status or '').strip().lower()
    parts = []

    current_count = info.get('current_entry_count')
    try:
        current_count = int(current_count)
    except (TypeError, ValueError):
        current_count = None
    if current_count is not None:
        noun = 'entry' if current_count == 1 else 'entries'
        parts.append(f'{current_count:,} {noun}')

    if status == 'partially_in_progress':
        parts.append(
            f"{int(info.get('active_type_count') or 0)}/"
            f"{int(info.get('total_type_count') or 0)} types active"
        )

    total_chunks = info.get('total_chunks')
    if total_chunks and status in (
        'in_progress',
        'partially_in_progress',
        'failed',
        'refine_failed',
    ):
        parts.append(
            f"chunks {int(info.get('completed_chunks') or 0)}/"
            f"{int(total_chunks)}"
        )

    before = info.get('entry_count_before')
    after = info.get('entry_count_after')
    try:
        before = int(before) if before is not None else None
        after = int(after) if after is not None else None
    except (TypeError, ValueError):
        before = after = None
    if status == 'completed' and before is not None and after is not None and before != after:
        parts.append(f'refined {before:,} -> {after:,}')

    return ''.join(f' | {part}' for part in parts)


def _glossary_refinement_manual_completion_info(
    entry_type, entries, saved_info, config, *, output_path=None, completed_at=None,
):
    """Accept the current source entries as this type's completed refinement."""
    from glossary_refinement import (
        DEFAULT_GLOSSARY_REFINEMENT_SYSTEM_PROMPT,
        _IDENTITY_HASH_VERSION,
        _canonical_refinement_mode,
        _entry_hash,
        _entry_identity_hash,
        _prompt_requests_unit_separator,
    )

    info = dict(saved_info) if isinstance(saved_info, dict) else {}
    config = config if isinstance(config, dict) else {}
    entries = list(entries or [])
    mode = _canonical_refinement_mode(config.get('glossary_refinement_chunking_mode', 'all'))
    system_prompt = config.get('glossary_refinement_system_prompt') or DEFAULT_GLOSSARY_REFINEMENT_SYSTEM_PROMPT
    user_prompt = config.get('glossary_refinement_user_prompt', '')
    delimiter = 'unit_separator' if _prompt_requests_unit_separator(system_prompt, user_prompt) else 'comma'
    hash_mode = f'{mode}:{delimiter}'
    hash_type = 'all selected entry types' if info.get('is_aggregate') else entry_type
    content_hash = _entry_hash(hash_type, entries, hash_mode)
    identity_hash = _entry_identity_hash(hash_type, entries, hash_mode)
    now = time.time() if completed_at is None else completed_at
    info.update({
        'entry_type': entry_type,
        'status': 'completed',
        'manually_marked_completed': True,
        'completed_at': now,
        'last_updated': now,
        'entry_count_before': len(entries),
        'entry_count_after': len(entries),
        'current_entry_count': len(entries),
        'input_hash': content_hash,
        'output_hash': content_hash,
        'identity_hash_version': _IDENTITY_HASH_VERSION,
        'input_identity_hash': identity_hash,
        'output_identity_hash': identity_hash,
        'chunking_mode': mode,
        'payload_delimiter': delimiter,
    })
    if output_path:
        info['output_file'] = os.path.basename(output_path)
    if info.get('total_chunks') is not None:
        info['completed_chunks'] = info['total_chunks']
    for field in (
        'error', 'error_message', 'failure_reason', 'qa_issues_found', 'reason',
        'legacy_identity_hashes', 'legacy_identity_entry_type', 'legacy_identity_hash_mode',
    ):
        info.pop(field, None)
    return info


def _normalize_glossary_refinement_selection(refinement_keys, active_types):
    """Resolve selected row keys, with the aggregate row taking precedence."""
    keys = [str(key or '') for key in refinement_keys or [] if str(key or '')]
    active = [str(entry_type or '').strip() for entry_type in active_types or [] if str(entry_type or '').strip()]
    if any(key.startswith('all::') for key in keys):
        return active
    requested = {
        _glossary_refinement_type_key(key.split('::', 1)[1])
        for key in keys
        if key.startswith('type::') and '::' in key
    }
    return [
        entry_type for entry_type in active
        if _glossary_refinement_type_key(entry_type) in requested
    ]


def _find_matching_glossary_refinement_aggregate(refinement, selected_types):
    """Find an aggregate progress record regardless of saved type ordering."""
    if not isinstance(refinement, dict):
        return None
    expected_scope = {
        _glossary_refinement_type_key(entry_type)
        for entry_type in selected_types or []
        if str(entry_type or '').strip()
    }
    if not expected_scope:
        return None
    for raw_key, info in refinement.items():
        key = str(raw_key or '')
        if not key.startswith('all::') or not isinstance(info, dict):
            continue
        saved_scope = {
            _glossary_refinement_type_key(entry_type)
            for entry_type in key.split('::', 1)[1].split(',')
            if str(entry_type or '').strip()
        }
        if saved_scope == expected_scope:
            return info
    return None


def _derive_glossary_refinement_aggregate_status(statuses, represented=None):
    normalized = [str(status or 'not_refined').strip().lower() for status in statuses or []]
    if any(status in ('failed', 'error', 'refine_failed') for status in normalized):
        return 'failed'
    if any(status == 'in_progress' for status in normalized):
        actionable_statuses = [status for status in normalized if status != 'skipped']
        if actionable_statuses and all(status == 'in_progress' for status in actionable_statuses):
            return 'in_progress'
        return 'partially_in_progress'
    if any(status in ('not_refined', 'unknown', '') for status in normalized):
        return 'not_refined'
    if normalized and all(status == 'skipped' for status in normalized):
        return 'skipped'
    if represented is not None and any(not bool(value) for value in represented):
        # A newly enabled type has joined the aggregate scope but has not yet
        # acquired its own persisted refinement result.
        return 'not_refined'
    return 'completed'


# ---------------------------------------------------------------------------
# Writes: the extractor's progress-file lock + atomic replace
# ---------------------------------------------------------------------------


@contextmanager
def _locked_glossary_progress_write(path):
    """Serialize a glossary-progress mutation with the extractor and the refiner.

    ``glossary_refinement.locked_progress_file`` is the cross-process lock both
    writers take (``extract_glossary_from_epub`` saves and ``update_refinement_progress``).
    """
    from glossary_refinement import _progress_lock, locked_progress_file

    with _progress_lock, locked_progress_file(path):
        yield


def _write_glossary_progress_atomic(path, data):
    """Write a glossary progress file through a temp file + atomic replace."""
    progress_dir = os.path.dirname(path) or "."
    os.makedirs(progress_dir, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=progress_dir,
            delete=False,
            suffix=".tmp",
        ) as temp_f:
            temp_path = temp_f.name
            json.dump(data, temp_f, ensure_ascii=False, indent=2)
            temp_f.flush()
            try:
                os.fsync(temp_f.fileno())
            except OSError:
                pass
        os.replace(temp_path, path)
        temp_path = None
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Locator: progress / glossary file lookup (closures of _force_retranslation_epub_or_text)
# ---------------------------------------------------------------------------


def glossary_progress_locator(self, file_path, glossary_progress_source_path=None):
    """Glossary-progress lookup helpers for one Progress Manager source.

    ``self`` is the Progress Manager owner (``config``, ``custom_entry_types``,
    ``base_dir``).  Returns a namespace of the moved closures.
    """
    def _glossary_progress_search_dirs(base):
        """Return likely glossary progress locations, newest per-book layout first."""
        search_dirs = []
        seen = set()

        def _add(path):
            if not path:
                return
            path = os.path.abspath(path)
            key = os.path.normcase(path)
            if key not in seen:
                seen.add(key)
                search_dirs.append(path)

        def _add_root(root):
            if not root:
                return
            root = os.path.abspath(root)
            shared = os.path.join(root, 'Glossary')
            try:
                from glossary_paths import get_book_glossary_dir
                _add(get_book_glossary_dir(shared, base, create=False))
            except Exception:
                _add(os.path.join(shared, base))
            _add(shared)
            _add(os.path.join(root, base, 'Glossary'))
            _add(os.path.join(root, base))

        _override_dir = (os.environ.get('OUTPUT_DIRECTORY') or os.environ.get('OUTPUT_DIR'))
        if not _override_dir and hasattr(self, 'config'):
            _override_dir = self.config.get('output_directory')
        if _override_dir:
            _add_root(_override_dir)

        try:
            from app_paths import _get_app_dir
            _app_dir = _get_app_dir()
        except Exception:
            _app_dir = os.getcwd()
        _add_root(_app_dir)

        if hasattr(self, 'base_dir'):
            _add_root(self.base_dir)

        return search_dirs

    def _find_progress_in_dir(directory, progress_name):
        candidate = os.path.join(directory, progress_name)
        if os.path.isfile(candidate):
            return candidate
        generic = os.path.join(directory, 'glossary_progress.json')
        if os.path.isfile(generic):
            return generic
        if os.path.basename(directory).lower() != 'glossary':
            try:
                matches = [
                    os.path.join(directory, name)
                    for name in os.listdir(directory)
                    if name.lower().endswith('_glossary_progress.json')
                ]
                if matches:
                    return max(matches, key=lambda path: os.path.getmtime(path))
            except Exception:
                pass
        return None

    def _find_glossary_progress_file():
        """Locate the glossary progress file for the current EPUB."""
        try:
            lookup_path = glossary_progress_source_path or file_path
            base = os.path.splitext(os.path.basename(lookup_path))[0]
            progress_name = f"{base}_glossary_progress.json"
            for d in _glossary_progress_search_dirs(base):
                if not os.path.isdir(d):
                    continue
                found = _find_progress_in_dir(d, progress_name)
                if found:
                    return found
        except Exception:
            pass
        return None

    def _find_gp_for_file(fp):
        """Locate the glossary progress file for a given EPUB path."""
        try:
            lookup_path = fp
            try:
                if glossary_progress_source_path and (
                    os.path.normcase(os.path.abspath(fp))
                    == os.path.normcase(os.path.abspath(file_path))
                ):
                    lookup_path = glossary_progress_source_path
            except Exception:
                if str(fp) == str(file_path):
                    lookup_path = glossary_progress_source_path or fp
            base = os.path.splitext(os.path.basename(lookup_path))[0]
            progress_name = f"{base}_glossary_progress.json"
            for d in _glossary_progress_search_dirs(base):
                if not os.path.isdir(d):
                    continue
                found = _find_progress_in_dir(d, progress_name)
                if found:
                    return found
        except Exception:
            pass
        return None

    def _bool_setting(value):
        if isinstance(value, str):
            return value.strip().lower() in ('1', 'true', 'yes', 'on')
        return bool(value)

    def _refinement_type_key(value):
        return _glossary_refinement_type_key(value)

    def _active_glossary_refinement_types():
        cfg = getattr(self, 'config', {}) or {}
        raw_custom_types = (
            getattr(self, 'custom_entry_types', None)
            or cfg.get('custom_entry_types', {})
            or {}
        )
        custom_types = dict(raw_custom_types) if isinstance(raw_custom_types, dict) else {}
        if not custom_types:
            custom_types = {
                'character': {'enabled': True},
                'terms': {'enabled': True},
                'surnames': {'enabled': True},
                'titles': {'enabled': True},
                'locations': {'enabled': True},
                'nicknames': {'enabled': True},
            }
        if 'term' in custom_types and 'terms' not in custom_types:
            custom_types['terms'] = custom_types.pop('term')
        elif 'term' in custom_types:
            custom_types.pop('term', None)

        active_types = []
        for type_name, type_cfg in custom_types.items():
            if isinstance(type_cfg, dict) and not type_cfg.get('enabled', True):
                continue
            type_name = str(type_name or '').strip()
            if type_name:
                active_types.append(type_name)
        return sorted(
            active_types,
            key=lambda name: (name not in ('character', 'terms'), name),
        )

    def _glossary_refinement_expected_entries(glossary_entries=None):
        """Return the always-visible aggregate and active type row model."""
        active_types = _active_glossary_refinement_types()
        if not active_types:
            return {}
        counts = Counter(
            _refinement_type_key(entry.get('type'))
            for entry in (glossary_entries or [])
            if isinstance(entry, dict) and str(entry.get('type') or '').strip()
        )
        chunking_mode = str(
            (getattr(self, 'config', {}) or {}).get(
                'glossary_refinement_chunking_mode', 'all'
            ) or 'all'
        ).lower()
        chunking_mode = 'all' if chunking_mode in ('all', 'all_types', 'all_in_one') else 'separate'
        aggregate_key = f"all::{','.join(active_types)}"
        aggregate_count = sum(counts.get(_refinement_type_key(t), 0) for t in active_types)
        expected = {
            aggregate_key: {
                'entry_type': 'All Entry Types',
                'status': 'not_refined' if aggregate_count else 'skipped',
                'chunking_mode': chunking_mode,
                'entry_count_before': aggregate_count,
                'current_entry_count': aggregate_count,
                'is_aggregate': True,
                'selected_types': list(active_types),
                'reason': 'no_entries' if not aggregate_count else '',
            }
        }
        for entry_type in active_types:
            entry_count = counts.get(_refinement_type_key(entry_type), 0)
            expected[f"type::{entry_type}"] = {
                'entry_type': entry_type,
                'status': 'not_refined' if entry_count else 'skipped',
                'chunking_mode': chunking_mode,
                'entry_count_before': entry_count,
                'current_entry_count': entry_count,
                'entry_count_after': 0 if not entry_count else None,
                'reason': 'no_entries' if not entry_count else '',
            }
        return expected

    def _find_glossary_for_refinement(source_path, progress_path=None):
        base = os.path.splitext(os.path.basename(source_path or ''))[0]
        directories = []
        if progress_path:
            directories.append(os.path.dirname(os.path.abspath(progress_path)))
        directories.extend(_glossary_progress_search_dirs(base))
        seen = set()
        for directory in directories:
            directory_key = os.path.normcase(os.path.abspath(directory))
            if directory_key in seen:
                continue
            seen.add(directory_key)
            for extension in ('.csv', '.json', '.txt', '.md'):
                for filename in (
                    f'{base}_glossary{extension}',
                    f'{base}{extension}',
                    f'glossary{extension}',
                ):
                    candidate = os.path.join(directory, filename)
                    if os.path.isfile(candidate):
                        return candidate
        return None

    _namespace = dict(locals())
    for _name in ('self', 'file_path', 'glossary_progress_source_path'):
        _namespace.pop(_name, None)
    return types.SimpleNamespace(**_namespace)


# ---------------------------------------------------------------------------
# Model: the data closures of _build_gp_panel
# ---------------------------------------------------------------------------


def make_glossary_progress_model(
    self,
    fp,
    gp_path,
    *,
    pump_loading=None,
    glossary_progress_source_path=None,
    glossary_progress_source_filenames=None,
    output_dir=None,
    prog=None,
    find_gp_for_file=None,
    glossary_refinement_expected_entries=None,
    refinement_type_key=None,
    current_gp_data=None,
):
    """The Glossary Progress panel's data model for one source EPUB.

    ``self`` is the Progress Manager owner.  ``current_gp_data`` returns the panel's
    current progress dict (desktop: the panel's ``gp_data``, which its refresh
    rebinds); the moved closures fall back to it where they used ``gp_data``.
    Returns a namespace of the moved closures plus ``gp_data`` and ``panel_state``.
    """
    _find_gp_for_file = find_gp_for_file
    _glossary_refinement_expected_entries = glossary_refinement_expected_entries
    _refinement_type_key = refinement_type_key
    if _find_gp_for_file is None or _glossary_refinement_expected_entries is None:
        _locator = glossary_progress_locator(
            self, fp, glossary_progress_source_path
        )
        if _find_gp_for_file is None:
            _find_gp_for_file = _locator._find_gp_for_file
        if _glossary_refinement_expected_entries is None:
            _glossary_refinement_expected_entries = (
                _locator._glossary_refinement_expected_entries
            )
    if _refinement_type_key is None:
        _refinement_type_key = _glossary_refinement_type_key

    def _pump_loading_frame():
        if callable(pump_loading):
            try:
                pump_loading()
            except Exception:
                pass
    
    def _gp_load_progress_dict(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
        except Exception as e:
            print(f"⚠️ Could not load glossary progress file {path}: {e}")
            return {}
        if isinstance(loaded, dict):
            return loaded
        print(f"⚠️ Glossary progress file has legacy non-dict shape: {type(loaded).__name__}")
        return {}

    def _gp_int_list(values):
        if values is None:
            return []
        if isinstance(values, dict):
            values = values.keys()
        elif isinstance(values, (str, int, float)):
            values = [values]
        result = []
        seen = set()
        try:
            iterator = iter(values)
        except TypeError:
            iterator = iter([values])
        for value in iterator:
            if isinstance(value, dict):
                value = value.get('chapter_index', value.get('actual_num', value.get('chapter_num')))
            try:
                ivalue = int(value)
            except (TypeError, ValueError):
                continue
            if ivalue not in seen:
                seen.add(ivalue)
                result.append(ivalue)
        return result

    def _gp_safe_int(value, default=0):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    gp_data = _gp_load_progress_dict(gp_path)

    def _current_gp_data():
        if callable(current_gp_data):
            return current_gp_data()
        return gp_data

    _pump_loading_frame()

    completed_indices = _gp_int_list(gp_data.get('completed', []))
    skipped_indices = _gp_int_list(gp_data.get('skipped', []))
    failed_indices = _gp_int_list(gp_data.get('failed', []))
    merged_indices = _gp_int_list(gp_data.get('merged_indices', []))
    book_title = gp_data.get('book_title', '')

    def _gp_qa_issue_map(_d):
        if not isinstance(_d, dict):
            _d = {}
        issues = {}

        def _add(idx, values):
            try:
                key = int(idx)
            except (TypeError, ValueError):
                return
            if isinstance(values, str):
                values = [values]
            if not isinstance(values, list):
                return
            bucket = issues.setdefault(key, [])
            for value in values:
                text = str(value).strip()
                if text and text not in bucket:
                    bucket.append(text)

        raw_map = _d.get('qa_issues_found', {})
        if isinstance(raw_map, dict):
            for idx, values in raw_map.items():
                if isinstance(values, dict):
                    values = values.get('qa_issues_found') or values.get('issues') or []
                mapped_idx = _gp_index_for_progress_value(idx, _d)
                _add(mapped_idx if mapped_idx is not None else idx, values)

        chapters = _d.get('chapters', {})
        if isinstance(chapters, dict):
            for key, info in chapters.items():
                if not isinstance(info, dict):
                    continue
                idx = _gp_index_for_entry(info, key, _d)
                _add(idx, info.get('qa_issues_found', []))
        return issues

    def _gp_filename_chapter_num(fname):
        import re as _re_gp_num
        nums = _re_gp_num.findall(r'[0-9]+', os.path.splitext(str(fname or ""))[0])
        if nums:
            try:
                return int(nums[-1])
            except (TypeError, ValueError):
                return None
        return None

    def _gp_source_chapter_num(ci, fname):
        try:
            is_special = self._is_special_file(fname)
        except (AttributeError, TypeError):
            is_special = _is_special_basename(fname)
        if is_special:
            return 0
        ch_num = _gp_filename_chapter_num(fname)
        if ch_num is not None:
            return ch_num
        return 0

    def _gp_display_chapter_num(ci, fname):
        return (panel_state.get('_chapter_display_numbers') or {}).get(
            ci,
            _gp_source_chapter_num(ci, fname),
        )

    def _gp_filename_keys(name):
        """Normalize a filename into a set of lowercase lookup keys."""
        return _glossary_progress_filename_keys(name)

    def _rebuild_reverse_lookups():
        """Rebuild O(1) lookup dicts from chapter_map. Call after chapter_map changes."""
        cmap = panel_state.get('chapter_map') or {}
        spine_index_map = panel_state.get('spine_index_map') or {}
        ordered_indices = sorted(
            cmap,
            key=lambda index: (spine_index_map.get(index, index + 1), index),
        )
        display_numbers = nonreset_chapter_display_numbers(
            _gp_source_chapter_num(index, cmap.get(index, ''))
            for index in ordered_indices
        )
        panel_state['_chapter_display_numbers'] = dict(
            zip(ordered_indices, display_numbers)
        )
        # filename key -> chapter index (first wins)
        fk_to_ci = {}
        for lookup_idx, (ci, mapped_name) in enumerate(cmap.items()):
            for key in _gp_filename_keys(mapped_name):
                fk_to_ci.setdefault(key, ci)
            if lookup_idx and lookup_idx % 200 == 0:
                _pump_loading_frame()
        # Parallel extraction runs against a temporary EPUB whose
        # chapters are named pair_0001.xhtml, pair_0002.xhtml, etc.
        # The Progress Manager intentionally displays the mapped raw
        # filenames instead, so bridge those two identities by their
        # stable mapped order. This lets both structured chapter
        # entries and top-level completed/in-progress arrays resolve.
        fk_to_ci.update(
            _parallel_glossary_progress_filename_aliases(
                glossary_progress_source_filenames
            )
        )
        panel_state['_fk_to_ci'] = fk_to_ci
        # actual_num (from filename) -> list of chapter indices
        anum_to_ci = {}
        for lookup_idx, (ci, fname) in enumerate(cmap.items()):
            num = _gp_filename_chapter_num(fname)
            if num is not None:
                anum_to_ci.setdefault(num, []).append(ci)
            if lookup_idx and lookup_idx % 200 == 0:
                _pump_loading_frame()
        panel_state['_anum_to_ci'] = anum_to_ci
    def _gp_index_for_actual_num(actual_num, _d=None):
        try:
            actual_num = int(actual_num)
        except (TypeError, ValueError):
            return None
        # O(1) reverse lookup from cached dict
        matches = list(panel_state.get('_anum_to_ci', {}).get(actual_num, []))
        if not matches:
            chapter_numbers = (_d or _current_gp_data()).get('chapter_numbers', {})
            if isinstance(chapter_numbers, dict):
                for ci, num in chapter_numbers.items():
                    try:
                        if int(num) == actual_num:
                            matches.append(int(ci))
                    except (TypeError, ValueError):
                        pass
        if len(matches) == 1:
            return matches[0]
        return None

    def _gp_index_for_entry(info, key=None, _d=None):
        if not isinstance(info, dict):
            info = {}

        had_filename_anchor = False
        fk_to_ci = panel_state.get('_fk_to_ci') or {}
        for fname_key in ('output_file', 'original_basename', 'chapter_file', 'source_filename', 'filename'):
            fname = os.path.basename(str(info.get(fname_key, "") or ""))
            if not fname:
                continue
            had_filename_anchor = True
            # O(1) lookup via reverse dict instead of scanning chapter_map
            for k in _gp_filename_keys(fname):
                if k in fk_to_ci:
                    return fk_to_ci[k]
        if had_filename_anchor:
            return None

        for num_key in ('actual_num', 'chapter_num'):
            ci = _gp_index_for_actual_num(info.get(num_key), _d)
            if ci is not None:
                return ci

        if key is not None and 'chapter_index' not in info:
            ci = _gp_index_for_actual_num(key, _d)
            if ci is not None:
                return ci

        try:
            return int(info.get('chapter_index', key))
        except (TypeError, ValueError):
            return None

    def _gp_index_for_progress_value(value, _d):
        if not isinstance(_d, dict):
            _d = {}
        try:
            ivalue = int(value)
        except (TypeError, ValueError):
            return None
        if str(_d.get('indexing', '')).lower() == 'chapter_index_zero_based':
            # Glossary extraction indices only cover files that were
            # eligible for extraction. The progress-manager list uses
            # the full OPF spine, which can contain an earlier cover or
            # other non-requested file. Resolve through the persisted
            # filename map before treating the integer as a UI row.
            return _map_zero_based_glossary_progress_index(
                ivalue,
                _d,
                panel_state.get('_fk_to_ci') or {},
            )
        positions = _d.get('chapter_positions', {})
        if isinstance(positions, dict) and str(ivalue) in positions:
            return ivalue
        ci = _gp_index_for_actual_num(ivalue, _d)
        return ci if ci is not None else ivalue

    def _gp_sets(_d):
        if not isinstance(_d, dict):
            _d = {}

        def _index_set(values):
            result = set()
            for value in _gp_int_list(values):
                ci = _gp_index_for_progress_value(value, _d)
                if ci is not None:
                    result.add(ci)
            return result

        comp = set()
        fail = set()
        merg = set()
        chapters = _d.get('chapters', {})
        represented = set()
        if isinstance(chapters, dict):
            for key, info in chapters.items():
                if not isinstance(info, dict):
                    continue
                ci = _gp_index_for_entry(info, key, _d)
                if ci is None:
                    continue
                represented.add(ci)
                status = str(info.get('status', '')).lower()
                if status in ('failed', 'qa_failed', 'error'):
                    fail.add(ci)
                elif status == 'merged':
                    merg.add(ci)
                elif status == 'completed':
                    comp.add(ci)

        # A progress file can briefly contain a mixture of structured
        # chapter entries and legacy/top-level arrays while the
        # extractor is saving or after a row was removed manually.
        # Structured entries win for their own index, but must not make
        # unrelated top-level updates disappear from the live view.
        comp |= _index_set(_d.get('completed', [])) - represented
        fail |= _index_set(_d.get('failed', [])) - represented
        merg |= _index_set(_d.get('merged_indices', [])) - represented
        # Failed should win over completed in the UI.
        comp -= fail
        return comp, fail, merg

    def _gp_skipped_set(_d):
        if not isinstance(_d, dict):
            _d = {}
        skipped = set()
        represented = set()
        chapters = _d.get('chapters', {})
        if isinstance(chapters, dict):
            for key, info in chapters.items():
                if not isinstance(info, dict):
                    continue
                ci = _gp_index_for_entry(info, key, _d)
                if ci is None:
                    continue
                represented.add(ci)
                if str(info.get('status', '')).lower() in (
                    'skipped_empty',
                    'skipped_image_only',
                    'skipped_title_header_only',
                ):
                    skipped.add(ci)
        for value in _gp_int_list(_d.get('skipped', [])):
            ci = _gp_index_for_progress_value(value, _d)
            if ci is not None and ci not in represented:
                skipped.add(ci)
        return skipped

    def _gp_in_progress_set(_d, _precomputed_sets=None):
        if not isinstance(_d, dict):
            _d = {}
        result = set()
        chapters = _d.get('chapters', {})
        represented = set()
        if isinstance(chapters, dict):
            for key, info in chapters.items():
                if not isinstance(info, dict):
                    continue
                ci = _gp_index_for_entry(info, key, _d)
                if ci is None:
                    continue
                status = str(info.get('status', '')).lower()
                if status:
                    represented.add(ci)
                if status != 'in_progress':
                    continue
                result.add(ci)
        for value in _gp_int_list(_d.get('in_progress', [])):
            ci = _gp_index_for_progress_value(value, _d)
            if ci is not None and ci not in represented:
                result.add(ci)
        if _precomputed_sets:
            comp, fail, merg = _precomputed_sets
        else:
            comp, fail, merg = _gp_sets(_d)
        return result - comp - fail - merg - _gp_skipped_set(_d)

    def _gp_status_cache(_d):
        if not isinstance(_d, dict):
            _d = {}
        comp = set()
        skipped = set()
        fail = set()
        merg = set()
        in_prog = set()
        qa_failed = set()
        skipped_variants = {}
        entries_by_ci = {}
        represented = set()
        chapters = _d.get('chapters', {})
        if isinstance(chapters, dict):
            for key, info in chapters.items():
                if not isinstance(info, dict):
                    continue
                ci = _gp_index_for_entry(info, key, _d)
                if ci is None:
                    continue
                represented.add(ci)
                entries_by_ci.setdefault(ci, []).append(info)
                entry_status = str(info.get('status', '')).lower()
                if entry_status in ('failed', 'qa_failed', 'error'):
                    fail.add(ci)
                elif entry_status == 'merged':
                    merg.add(ci)
                elif entry_status == 'completed':
                    comp.add(ci)
                elif entry_status == 'in_progress':
                    in_prog.add(ci)
                elif entry_status in (
                    'skipped_empty',
                    'skipped_image_only',
                    'skipped_title_header_only',
                ):
                    skipped.add(ci)
                    skipped_variants[ci] = entry_status
                if entry_status == 'qa_failed':
                    qa_failed.add(ci)

        def _add_unrepresented(values, target):
            for value in _gp_int_list(values):
                ci = _gp_index_for_progress_value(value, _d)
                if ci is not None and ci not in represented:
                    target.add(ci)

        _add_unrepresented(_d.get('completed', []), comp)
        _add_unrepresented(_d.get('skipped', []), skipped)
        _add_unrepresented(_d.get('failed', []), fail)
        _add_unrepresented(_d.get('merged_indices', []), merg)
        _add_unrepresented(_d.get('in_progress', []), in_prog)

        # A chapter whose only entry is a stale failure counts as
        # "represented", so the plain helper above would drop the
        # top-level in_progress marker for exactly the chapter being
        # retried. Honour that marker when the entry representing the
        # chapter is itself a failure; a completed/skipped/merged entry
        # still wins, since those are final.
        for value in _gp_int_list(_d.get('in_progress', [])):
            ci = _gp_index_for_progress_value(value, _d)
            if ci is None or ci in comp or ci in skipped or ci in merg:
                continue
            if ci in fail:
                in_prog.add(ci)
        # Precedence between conflicting markers for the same chapter.
        #
        # A live in_progress supersedes a failure left over from an
        # earlier attempt: if the chapter is being retried right now,
        # the old failure is history. This used to read
        #   in_prog -= comp | skipped | fail | merg
        # which did the opposite -- a stale failed entry cancelled the
        # in_progress marker, so a chapter actively streaming showed as
        # Failed for the whole retry.
        #
        # Completions, skips and merges are final states, so they still
        # win over an in_progress marker stranded by an interrupted run.
        in_prog -= comp | skipped | merg
        fail -= in_prog
        comp -= fail
        skipped -= fail
        issues = _gp_qa_issue_map(_d)
        return {
            'completed': comp,
            'skipped': skipped,
            'failed': fail,
            'merged': merg,
            'in_progress': in_prog,
            'issues': issues,
            'qa_failed': qa_failed,
            'skipped_variants': skipped_variants,
            'entries_by_ci': entries_by_ci,
        }

    def _gp_status_for(ci, _d, cache=None):
        if not isinstance(_d, dict):
            _d = {}
        cache = cache or _gp_status_cache(_d)
        comp = cache['completed']
        fail = cache['failed']
        merg = cache['merged']
        in_prog = cache['in_progress']
        issues = cache['issues']
        if ci in fail:
            return ('qa_failed' if issues.get(ci) or ci in cache['qa_failed'] else 'failed'), issues.get(ci, [])
        if ci in merg:
            return 'merged', []
        if ci in cache['skipped_variants']:
            return cache['skipped_variants'][ci], []
        if ci in cache['skipped']:
            return 'skipped', []
        if ci in comp:
            return 'completed', []
        if ci in in_prog:
            return 'in_progress', []
        return 'not_completed', []

    def _gp_entries_for(ci, _d):
        if not isinstance(_d, dict):
            return []
        chapters = _d.get('chapters', {})
        entries = []
        if isinstance(chapters, dict):
            for key, info in chapters.items():
                if not isinstance(info, dict):
                    continue
                if _gp_index_for_entry(info, key, _d) != ci:
                    continue
                entries.append(info)
        return entries

    def _gp_entry_for(ci, _d, status=None):
        return _select_progress_entry_for_display(_gp_entries_for(ci, _d), status)

    def _gp_model_for(ci, _d, status=None, entry=None):
        if _progress_status_hides_model_for_display(status):
            return ''
        selected = entry if isinstance(entry, dict) and entry else _gp_entry_for(ci, _d, status)
        model_name = _progress_entry_model_for_display(selected)
        if model_name:
            return model_name
        return '(model unknown)'

    def _gp_display_for(ci, fname, _d, cache=None):
        opf_pos = (panel_state.get('spine_index_map') or {}).get(ci, ci + 1)
        ch_num = _gp_display_chapter_num(ci, fname)
        cache = cache or _gp_status_cache(_d)
        status, issues = _gp_status_for(ci, _d, cache)
        cached_entries = cache.get('entries_by_ci', {}).get(ci, [])
        entry = _select_progress_entry_for_display(cached_entries, status)
        hide_model = _progress_status_hides_model_for_display(status)
        model_name = (
            ''
            if hide_model
            else _progress_entry_model_for_display(entry) or '(model unknown)'
        )
        icons = {
            'completed': '\u2705',
            'skipped': '⏭️',
            'skipped_empty': '📄',
            'skipped_image_only': '📸',
            'skipped_title_header_only': '🏷️',
            'failed': '\u274c',
            'qa_failed': '\u274c',
            'merged': '\U0001f517',
            'in_progress': '\U0001f504',
            'not_completed': '\u2b1c',
        }
        icon = icons.get(status) or '\u2b1c'
        skipped_labels = {
            'skipped': 'Skipped',
            'skipped_empty': 'Empty (Skipped)',
            'skipped_image_only': 'Image Only (Skipped)',
            'skipped_title_header_only': 'Title/Header Only (Skipped)',
        }
        status_label = skipped_labels.get(
            status,
            status.replace('_', ' ').title(),
        )
        if status == 'completed' and _progress_entry_refinement_failed_for_display(entry):
            status_label = f"{status_label} 💀"
        elif status == 'completed' and _progress_entry_refined_for_display(entry):
            status_label = f"{status_label} ⭐"
        if status in skipped_labels or hide_model:
            display = f"[{opf_pos:03d}] Ch.{ch_num:03d} | {icon} {status_label:14s} | {fname}"
        else:
            display = f"[{opf_pos:03d}] Ch.{ch_num:03d} | {icon} {status_label:14s} | {fname} -> {model_name}"
        if issues:
            qa_issue_previews = entry.get('qa_issue_previews', {}) if isinstance(entry, dict) else {}
            if not isinstance(qa_issue_previews, dict):
                qa_issue_previews = {}

            issues_display = ', '.join(
                _format_qa_issue_for_progress_display(issue, qa_issue_previews)
                for issue in issues[:2]
            )
            if len(issues) > 2:
                issues_display += f' (+{len(issues)-2} more)'
            display += f" | {issues_display}"
        return display, status

    def _gp_glossary_entries(_d):
        """Load the canonical glossary associated with this progress file."""
        gp_dir = os.path.dirname(gp_path)
        candidates = []
        recorded = str((_d or {}).get('glossary_output_file') or '').strip()
        if recorded:
            recorded_path = recorded if os.path.isabs(recorded) else os.path.join(gp_dir, os.path.basename(recorded))
            candidates.extend([
                recorded_path,
                os.path.splitext(recorded_path)[0] + '.csv',
                os.path.splitext(recorded_path)[0] + '.json',
            ])
        source_bases = [
            os.path.splitext(os.path.basename(fp or ''))[0],
            os.path.splitext(os.path.basename(gp_path or ''))[0].replace('_glossary_progress', ''),
        ]
        for source_base in source_bases:
            if not source_base:
                continue
            for extension in ('.csv', '.json', '.txt', '.md'):
                candidates.extend([
                    os.path.join(gp_dir, f'{source_base}_glossary{extension}'),
                    os.path.join(gp_dir, f'{source_base}{extension}'),
                ])
        for extension in ('.csv', '.json', '.txt', '.md'):
            candidates.append(os.path.join(gp_dir, f'glossary{extension}'))

        seen_paths = set()
        for candidate in candidates:
            normalized = os.path.normcase(os.path.abspath(candidate))
            if normalized in seen_paths or not os.path.isfile(candidate):
                continue
            seen_paths.add(normalized)
            try:
                stat = os.stat(candidate)
                signature = (normalized, stat.st_mtime_ns, stat.st_size)
            except OSError:
                signature = (normalized,)
            if signature == panel_state.get('_glossary_entries_signature'):
                return list(panel_state.get('_glossary_entries_cache') or [])
            try:
                entries = parse_glossary_file(candidate)
            except Exception:
                continue
            panel_state['_glossary_entries_signature'] = signature
            panel_state['_glossary_entries_cache'] = list(entries or [])
            panel_state['_glossary_path'] = candidate
            return list(entries or [])
        panel_state['_glossary_entries_signature'] = None
        panel_state['_glossary_entries_cache'] = []
        panel_state['_glossary_path'] = None
        return []

    def _gp_minimal_pass_toggle_enabled():
        """Whether 'Add minimal glossary pass' is currently on.

        Read live rather than cached: the dialog can be open while the
        setting is flipped in Glossary Manager, and the row should
        appear or vanish on the next refresh.
        """
        try:
            checkbox = getattr(self, 'glossary_add_minimal_pass_checkbox', None)
            if checkbox is not None:
                return bool(checkbox.isChecked())
        except Exception:
            pass
        try:
            config = getattr(self, 'config', None)
            if isinstance(config, dict) and 'glossary_add_minimal_pass' in config:
                return bool(config.get('glossary_add_minimal_pass'))
        except Exception:
            pass
        return str(os.environ.get('GLOSSARY_ADD_MINIMAL_PASS', '0')).strip().lower() in (
            '1', 'true', 'yes', 'on'
        )

    def _gp_minimal_pass_row(_d):
        """The Minimal-pass row, or None when the pass never ran.

        extract_glossary_from_epub only writes the `minimal_pass` key
        when the toggle is on, so presence of the key is what keeps
        this row hidden otherwise -- and what makes it appear as soon
        as the pass starts, since the dialog re-reads the file on a
        timer.
        """
        if not isinstance(_d, dict):
            _d = {}
        info = _d.get('minimal_pass')
        if not isinstance(info, dict) or not info:
            # The pass has not run yet. Still show the row -- as an
            # outstanding item -- whenever the toggle is on, so the
            # work it represents is visible before it starts rather
            # than appearing from nowhere mid-run.
            if not _gp_minimal_pass_toggle_enabled():
                return None
            info = {'status': 'not_completed'}
        raw_status = str(info.get('status') or 'unknown').strip().lower()
        status = 'failed' if raw_status in ('failed', 'error') else raw_status
        icon_map = {
            'completed': '\u2705',
            'skipped': '\u23ed\ufe0f',
            'failed': '\u274c',
            'in_progress': '\U0001f504',
            'not_completed': '\u2b1c',
        }
        icon = icon_map.get(status, '\u2b1c')
        reason = str(info.get('reason') or '').strip().lower()
        if status == 'skipped' and reason == 'no_entries':
            status_label = 'Skipped - No Entries'
        elif status == 'skipped' and reason == 'stopped':
            status_label = 'Skipped - Stopped'
        elif status == 'not_completed':
            status_label = 'Not Translated'
        else:
            status_label = status.replace('_', ' ').title()
        # The "Minimal Pass" prefix already names the row, so nothing
        # else is needed ahead of the model -- chapter rows put a
        # filename there only because they need one to be told apart.
        parts = [f"Minimal Pass | {icon} {status_label:20s}"]
        if status in ("in_progress", "completed", "failed"):
            # Only states that reached an API call name a model.
            model_name = _progress_entry_model_for_display(info)
            parts.append(model_name or '(model unknown)')
        try:
            count = int(info.get('entry_count'))
        except (TypeError, ValueError):
            count = None
        if count:
            parts.append(f"{count} entries")
        elif status == 'failed':
            error = str(info.get('error') or '').strip()
            if error:
                parts.append(error[:80])
        display = " | ".join(parts)
        return ('minimal_pass', display, status)

    def _gp_minimal_pass_status_counts(_d):
        row = _gp_minimal_pass_row(_d)
        if not row:
            return {}
        status = str(row[2] or 'unknown').lower().replace(' ', '_')
        return {status: 1}

    def _gp_refinement_rows(_d):
        from glossary_refinement import _find_type_refinement_progress

        refinement = _d.get('refinement', {}) if isinstance(_d, dict) else {}
        if not isinstance(refinement, dict):
            refinement = {}
        expected = _glossary_refinement_expected_entries(_gp_glossary_entries(_d))
        merged_rows = {}
        for expected_key, expected_info in expected.items():
            saved_info = refinement.get(expected_key)
            if expected_info.get('is_aggregate') and not isinstance(saved_info, dict):
                saved_info = _find_matching_glossary_refinement_aggregate(
                    refinement,
                    expected_info.get('selected_types'),
                )
            if expected_key.startswith('type::'):
                _saved_key, saved_info = _find_type_refinement_progress(
                    refinement, expected_key.split('::', 1)[1],
                )
                saved_info = saved_info or None
            info = _merge_glossary_refinement_row_info(
                expected_info,
                saved_info,
            )
            if expected_info.get('is_aggregate'):
                info['entry_type'] = 'All Entry Types'
                info['is_aggregate'] = True
                info['selected_types'] = list(expected_info.get('selected_types') or [])
            merged_rows[expected_key] = info

        individual_infos = [
            info for key, info in merged_rows.items() if key.startswith('type::')
        ]
        aggregate_key = next((key for key in merged_rows if key.startswith('all::')), None)
        if aggregate_key:
            statuses = [str(info.get('status') or 'not_refined').lower() for info in individual_infos]
            aggregate_status = _derive_glossary_refinement_aggregate_status(
                statuses,
                [info.get('_has_saved_progress') for info in individual_infos],
            )
            aggregate_entry_count = sum(
                int(info.get('current_entry_count') or 0) for info in individual_infos
            )
            merged_rows[aggregate_key]['status'] = aggregate_status
            merged_rows[aggregate_key]['active_type_count'] = sum(
                status == 'in_progress' for status in statuses
            )
            merged_rows[aggregate_key]['total_type_count'] = len(statuses)
            merged_rows[aggregate_key]['current_entry_count'] = aggregate_entry_count
            if aggregate_status == 'skipped':
                merged_rows[aggregate_key]['reason'] = 'no_entries'
        rows = []
        for key, info in merged_rows.items():
            if not isinstance(info, dict):
                continue
            entry_type = str(info.get('entry_type') or key.replace('type::', '')).strip() or 'entry type'
            raw_status = str(info.get('status') or 'unknown').lower()
            status = 'refine_failed' if raw_status in ('failed', 'error') else raw_status
            model_name = str(info.get('model_name') or info.get('model') or '').strip() or '(model unknown)'
            detail = _glossary_refinement_row_detail(info, status)
            icon_map = {
                'completed': '\u2705',
                'skipped': '⏭️',
                'failed': '\u274c',
                'qa_failed': '\u274c',
                'refine_failed': '💀',
                'in_progress': '\U0001f504',
                'partially_in_progress': '\U0001f504',
                'not_refined': '\u2728',
            }
            icon = icon_map.get(status, '\u2b1c')
            if status == 'skipped' and str(info.get('reason') or '').lower() == 'no_entries':
                status_label = 'Skipped - No Entries'
                model_suffix = ''
            else:
                status_label = 'Refine Failed' if status == 'refine_failed' else status.replace('_', ' ').title()
                model_suffix = (
                    f" -> {model_name}"
                    if status not in ('skipped', 'not_refined') and not info.get('is_aggregate')
                    else ''
                )
            display = f"Refinement | {icon} {status_label:20s} | {entry_type}{model_suffix}{detail}"
            rows.append((key, display, status))
        return rows

    def _gp_refinement_status_counts(_d):
        counts = {}
        for _key, _display, status in _gp_refinement_rows(_d):
            status = str(status or 'unknown').lower().replace(' ', '_')
            counts[status] = counts.get(status, 0) + 1
        return counts

    def _gp_extra_row_status_counts(_d):
        """Non-chapter rows that still count toward the header totals.

        The legend's second input is "everything that is not a chapter",
        so the Minimal-pass row is tallied here alongside refinement and
        shows up in Total/Completed/In Progress like any other entry.
        """
        counts = dict(_gp_refinement_status_counts(_d))
        for status, count in _gp_minimal_pass_status_counts(_d).items():
            counts[status] = counts.get(status, 0) + count
        return counts

    def _gp_color_for(status):
        if status == 'completed':
            return '#27ae60'
        if status in (
            'skipped',
            'skipped_empty',
            'skipped_image_only',
            'skipped_title_header_only',
        ):
            return '#94a3b8'
        if status == 'merged':
            return '#17a2b8'
        if status in ('in_progress', 'partially_in_progress'):
            return '#f59e0b'
        if status == 'refine_failed':
            return '#7f5f00'
        if status in ('failed', 'qa_failed'):
            return '#e74c3c'
        return '#5a9fd4'

    def _gp_restore_in_progress_entry(info):
        if not isinstance(info, dict):
            return None
        previous_status = str(info.get('previous_status', '') or '').lower()
        previous_entry = info.get('previous_progress_entry')
        if isinstance(previous_entry, dict):
            restored = dict(previous_entry)
            restored_status = str(restored.get('status', previous_status) or previous_status).lower()
            if restored_status and restored_status not in ('in_progress', 'not_completed', 'not translated', 'not_translated'):
                restored.pop('previous_status', None)
                restored.pop('previous_progress_entry', None)
                return restored
        if previous_status in ('qa_failed', 'failed', 'error', 'pending', 'merged', 'completed'):
            restored = dict(info)
            restored['status'] = 'failed' if previous_status == 'error' else previous_status
            restored.pop('previous_status', None)
            restored.pop('previous_progress_entry', None)
            restored.pop('previous_status_unknown', None)
            return restored
        if info.get('previous_status_unknown'):
            restored = dict(info)
            restored['status'] = 'failed'
            restored.pop('previous_status', None)
            restored.pop('previous_progress_entry', None)
            restored.pop('previous_status_unknown', None)
            return restored
        if previous_status in ('not_completed', 'not translated', 'not_translated'):
            return None
        if info.get('output_file'):
            restored = dict(info)
            restored['status'] = 'failed'
            restored.pop('previous_status', None)
            restored.pop('previous_progress_entry', None)
            restored.pop('previous_status_unknown', None)
            return restored
        return None

    # Lightweight spine reader - returns (chapter_map, total_chapters, spine_index_map)
    def _read_spine_map(epub_path, translate_special):
        """Read OPF spine and return (chapter_map, total_chapters, spine_index_map)."""
        cmap = {}
        spine_index_map = {}
        if not (epub_path.lower().endswith('.epub') and os.path.exists(epub_path)):
            return cmap, 0, spine_index_map
        try:
            import zipfile
            from xml.etree import ElementTree as ET
            with zipfile.ZipFile(epub_path, 'r') as zf:
                opf_path = find_epub_opf_member(zf)
                
                if not opf_path:
                    return cmap, 0, spine_index_map
                
                opf_xml = ET.fromstring(zf.read(opf_path))
                opf_ns = {'opf': 'http://www.idpf.org/2007/opf'}
                
                id_to_href = {}
                html_types = {'application/xhtml+xml', 'text/html', 'application/html+xml'}
                for manifest_idx, item in enumerate(opf_xml.findall('.//opf:manifest/opf:item', opf_ns)):
                    mid = item.get('id', '')
                    mtype = item.get('media-type', '')
                    href = item.get('href', '')
                    if mtype in html_types:
                        id_to_href[mid] = href
                    if manifest_idx and manifest_idx % 200 == 0:
                        _pump_loading_frame()
                
                spine_hrefs = []
                for spine_ref_idx, itemref in enumerate(opf_xml.findall('.//opf:spine/opf:itemref', opf_ns)):
                    idref = itemref.get('idref', '')
                    if idref in id_to_href:
                        spine_hrefs.append(id_to_href[idref])
                    if spine_ref_idx and spine_ref_idx % 200 == 0:
                        _pump_loading_frame()
                
                _kw_env = os.environ.get('SPECIAL_FILE_KEYWORDS', '')
                special_keywords = [k.strip().lower() for k in _kw_env.split(',') if k.strip()] if _kw_env else [
                    'title', 'toc', 'copyright', 'preface', 'nav',
                    'message', 'notice', 'colophon', 'dedication', 'epigraph',
                    'foreword', 'acknowledgment', 'author', 'appendix',
                    'bibliography'
                ]
                _exact_env = os.environ.get('SPECIAL_FILE_EXACT', '')
                special_exact = [k.strip().lower() for k in _exact_env.split(',') if k.strip()] if _exact_env else ['index', 'glossary', 'glossary_extension', 'glossary_unified']
                import re as _re_spine
                ci = 0
                for opf_pos, href in enumerate(spine_hrefs, start=1):
                    basename = os.path.basename(href)
                    if not translate_special:
                        name_noext = os.path.splitext(basename)[0]
                        name_lower = name_noext.lower()
                        name_stripped = _re_spine.sub(r'\d+$', '', name_lower).rstrip('_- ')
                        # Exact match: these are special only when the basename matches exactly
                        if name_lower in special_exact:
                            continue
                        if any(kw in name_lower for kw in special_keywords):
                            has_digits = bool(_re_spine.search(r'\d', name_noext))
                            if not has_digits or any(kw == name_stripped or kw in name_stripped for kw in special_keywords):
                                continue
                    cmap[ci] = basename
                    spine_index_map[ci] = opf_pos
                    ci += 1
                    if opf_pos % 200 == 0:
                        _pump_loading_frame()
                filtered_map, filtered_spine_map = (
                    _filter_glossary_source_chapter_map(
                        cmap,
                        spine_index_map,
                        glossary_progress_source_filenames,
                    )
                )
                return (
                    filtered_map,
                    len(filtered_map),
                    filtered_spine_map,
                )
        except Exception:
            return cmap, 0, spine_index_map
    
    # Mutable state so refresh can update chapter_map when toggle changes
    _ts_init = os.getenv('TRANSLATE_SPECIAL_FILES', '0') == '1'
    _cmap_init, _total_init, _spine_idx_init = _read_spine_map(fp, _ts_init)
    _pump_loading_frame()
    
    if _total_init == 0:
        _total_init = _gp_safe_int(gp_data.get('chapter_count'), 0)
        if _total_init <= 0:
            chapter_filenames = gp_data.get('chapter_filenames', {})
            if isinstance(chapter_filenames, dict) and chapter_filenames:
                _total_init = max((int(k) for k in chapter_filenames.keys() if str(k).isdigit()), default=-1) + 1
        if _total_init <= 0:
            _idx_values = []
            for _values in (completed_indices, skipped_indices, failed_indices, merged_indices):
                _idx_values.extend(_gp_int_list(_values))
            _total_init = (max(_idx_values) + 1) if _idx_values else 1
    
    # Store in mutable dict so closures can update
    panel_state = {
        'chapter_map': _cmap_init,
        'spine_index_map': _spine_idx_init,
        'total': _total_init,
        'translate_special': _ts_init,
        'populate_generation': 0,
        'mark_completed_lock': threading.Lock(),
        '_gp_bg_running': False,
        '_refresh_generation': 0,
        '_chapter_items': {},
        '_row_fingerprints': {},
        '_refinement_fingerprint': None,
        '_full_rebuild_pending': False,
        '_refresh_after_running': False,
        '_refresh_callbacks': [],
    }
    if not panel_state['chapter_map']:
        chapter_filenames = gp_data.get('chapter_filenames', {})
        if isinstance(chapter_filenames, dict):
            try:
                panel_state['chapter_map'] = {
                    int(k): os.path.basename(str(v or ""))
                    for k, v in chapter_filenames.items()
                    if str(k).lstrip('-').isdigit() and v
                }
                panel_state['spine_index_map'] = {
                    int(k): int(k) + 1
                    for k in chapter_filenames.keys()
                    if str(k).lstrip('-').isdigit()
                }
            except Exception:
                panel_state['chapter_map'] = {}
                panel_state['spine_index_map'] = {}
    
    # Build O(1) reverse lookups from chapter_map
    _rebuild_reverse_lookups()
    _pump_loading_frame()
    
    def _gp_file_signature(path):
        """Return a precise signature for an atomically replaced progress file."""
        try:
            stat = os.stat(path)
            return (stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size)
        except OSError:
            return None

    # The extractor atomically replaces this file, sometimes several
    # times inside one second. A float mtime alone can miss those
    # transitions on Windows, so track nanosecond timestamps and size.
    panel_state['_last_signature'] = _gp_file_signature(gp_path)

    # Background results are generation-tagged so a manual deletion can
    # invalidate an already queued snapshot instead of letting it paint
    # stale statuses back over the list.
    _gp_pending_result = []

    def _invalidate_gp_refresh():
        panel_state['_refresh_generation'] = panel_state.get('_refresh_generation', 0) + 1
        panel_state['_last_signature'] = None
        _gp_pending_result.clear()

    def _finish_gp_refresh_callbacks():
        """Notify manual-refresh callers after the requested snapshot is applied."""
        callbacks = panel_state.get('_refresh_callbacks') or []
        panel_state['_refresh_callbacks'] = []
        for callback in callbacks:
            try:
                callback()
            except (RuntimeError, TypeError):
                pass

    def _gp_stats_for_dict(_d):
        _cache2 = _gp_status_cache(_d)
        _comp2 = _cache2['completed']
        _skip2 = _cache2['skipped']
        _fail2 = _cache2['failed']
        _merg2 = _cache2['merged']
        _prog2 = _cache2['in_progress']
        _total = panel_state['total']
        _ref_counts2 = _gp_extra_row_status_counts(_d)
        return _combine_glossary_progress_legend_stats(
            {
                'total': _total,
                'completed': len(_comp2 - _skip2 - _merg2),
                'skipped': len(_skip2),
                'in_progress': len(_prog2),
                'failed': len(_fail2),
                'merged': len(_merg2),
                'remaining': max(
                    0,
                    _total
                    - len(_comp2 | _skip2 | _fail2 | _merg2 | _prog2),
                ),
            },
            _ref_counts2,
        )

    def _gp_remove_mapped_indices_from_list(_d, key, indices_to_remove, match_raw=False):
        values = _d.get(key, [])
        if not isinstance(values, list):
            return False
        kept = []
        for value in values:
            mapped_ci = _gp_index_for_progress_value(value, _d)
            if mapped_ci is not None and mapped_ci in indices_to_remove:
                continue
            if match_raw:
                try:
                    if int(value) in indices_to_remove:
                        continue
                except (TypeError, ValueError):
                    pass
            kept.append(value)
        if len(kept) != len(values):
            _d[key] = kept
            return True
        return False

    def _gp_progress_uses_chapter_entries(_d):
        chapters = _d.get('chapters', {}) if isinstance(_d, dict) else {}
        if not isinstance(chapters, dict):
            return False
        return any(isinstance(info, dict) and str(info.get('status') or '').strip() for info in chapters.values())

    def _gp_progress_value_for_index(ci, _d):
        if str(_d.get('indexing', '')).lower() == 'chapter_index_zero_based':
            return ci
        positions = _d.get('chapter_positions', {})
        if isinstance(positions, dict) and str(ci) in positions:
            return ci
        chapter_numbers = _d.get('chapter_numbers', {})
        if isinstance(chapter_numbers, dict) and str(ci) in chapter_numbers:
            try:
                return int(chapter_numbers[str(ci)])
            except (TypeError, ValueError):
                return chapter_numbers[str(ci)]
        fname = (panel_state.get('chapter_map') or {}).get(ci, f'chapter {ci + 1}')
        ch_num = _gp_source_chapter_num(ci, fname)
        return ch_num if ch_num is not None else ci

    def _gp_chapter_entry_map(_d):
        chapters = _d.get('chapters', {}) if isinstance(_d, dict) else {}
        if not isinstance(chapters, dict):
            return {}
        entry_map = {}
        for key, info in chapters.items():
            if not isinstance(info, dict):
                continue
            ci = _gp_index_for_entry(info, key, _d)
            if ci is not None:
                entry_map.setdefault(ci, (key, info))
        return entry_map

    def _gp_row_updates_for_targets(_d, target_specs):
        updates = {}
        cache = _gp_status_cache(_d)
        refinement_rows = {row[0]: row for row in _gp_refinement_rows(_d)}
        _cmap = panel_state['chapter_map']
        for kind, value in target_specs:
            if kind == 'refinement':
                row = refinement_rows.get(value)
                if not row:
                    continue
                _rk, display, status = row
                updates[(kind, value)] = {
                    'display': display,
                    'status': status,
                    'color': _gp_color_for(status),
                }
            else:
                ci = value
                fname = _cmap.get(ci, f'chapter {ci + 1}')
                display, status = _gp_display_for(ci, fname, _d, cache)
                updates[(kind, value)] = {
                    'display': display,
                    'status': status,
                    'color': _gp_color_for(status),
                }
        return updates

    def _gp_apply_mark_completed_to_progress(_rp, target_specs):
        _d = _gp_load_progress_dict(_rp)
        if not isinstance(_d, dict):
            _d = {}

        chapter_indices = {value for kind, value in target_specs if kind == 'chapter'}
        refinement_keys = {value for kind, value in target_specs if kind == 'refinement'}
        changed = False
        now = time.time()

        if chapter_indices:
            for key in ('skipped', 'failed', 'merged_indices', 'in_progress', 'manual_removed_indices'):
                changed = _gp_remove_mapped_indices_from_list(
                    _d,
                    key,
                    chapter_indices,
                    match_raw=(key == 'manual_removed_indices'),
                ) or changed

            completed_values = _d.get('completed', [])
            if not isinstance(completed_values, list):
                completed_values = []
            completed_mapped = {
                mapped_ci
                for mapped_ci in (_gp_index_for_progress_value(value, _d) for value in completed_values)
                if mapped_ci is not None
            }
            for ci in sorted(chapter_indices):
                if ci not in completed_mapped:
                    completed_values.append(_gp_progress_value_for_index(ci, _d))
                    completed_mapped.add(ci)
                    changed = True
            _d['completed'] = completed_values

            qa_map = _d.get('qa_issues_found', {})
            if isinstance(qa_map, dict):
                new_qa_map = {}
                for key, value in qa_map.items():
                    mapped_ci = _gp_index_for_progress_value(key, _d)
                    keep = mapped_ci not in chapter_indices if mapped_ci is not None else True
                    if keep:
                        new_qa_map[key] = value
                if len(new_qa_map) != len(qa_map):
                    _d['qa_issues_found'] = new_qa_map
                    changed = True

            if _gp_progress_uses_chapter_entries(_d):
                chapters = _d.get('chapters')
                if not isinstance(chapters, dict):
                    chapters = {}
                    _d['chapters'] = chapters
                entry_map = _gp_chapter_entry_map(_d)
                for ci in sorted(chapter_indices):
                    entry_key, entry = entry_map.get(ci, (None, None))
                    fname = (panel_state.get('chapter_map') or {}).get(ci, f'chapter {ci + 1}')
                    if not isinstance(entry, dict):
                        entry_key = str(ci)
                        if entry_key in chapters:
                            entry_key = f"manual_completed_{ci}"
                        entry = {
                            'chapter_index': ci,
                            'actual_num': _gp_source_chapter_num(ci, fname),
                            'original_basename': fname,
                        }
                        chapters[entry_key] = entry
                        entry_map[ci] = (entry_key, entry)
                    entry['status'] = 'completed'
                    entry.setdefault('chapter_index', ci)
                    entry.setdefault('actual_num', _gp_source_chapter_num(ci, fname))
                    entry.setdefault('original_basename', fname)
                    entry['last_updated'] = now
                    entry['manually_marked_completed'] = True
                    for field in (
                        'qa_issues', 'qa_issues_found', 'qa_timestamp', 'failure_reason',
                        'error_message', 'previous_status', 'previous_progress_entry',
                        'previous_status_unknown', 'merged_parent_chapter', 'skip_reason'
                    ):
                        entry.pop(field, None)
                    if str(entry.get('model_name') or '').upper() == 'SKIPPED':
                        entry.pop('model_name', None)
                    changed = True

        if refinement_keys:
            refinement = _d.setdefault('refinement', {})
            if not isinstance(refinement, dict):
                refinement = {}
                _d['refinement'] = refinement
            glossary_entries = _gp_glossary_entries(_d)
            expected_refinement = _glossary_refinement_expected_entries(glossary_entries)
            if any(str(key).startswith('all::') for key in refinement_keys):
                refinement_keys.update(expected_refinement)
            for ref_key in sorted(refinement_keys):
                expected_info = expected_refinement.get(ref_key, {})
                ref_info = dict(expected_info)
                saved_info = refinement.get(ref_key)
                if isinstance(saved_info, dict):
                    ref_info.update(saved_info)
                entry_type = expected_info.get('entry_type') or str(ref_key).split('::', 1)[-1]
                target_types = expected_info.get('selected_types') if expected_info.get('is_aggregate') else [entry_type]
                target_types = {_refinement_type_key(value) for value in target_types or []}
                target_entries = [
                    entry for entry in glossary_entries
                    if isinstance(entry, dict) and _refinement_type_key(entry.get('type')) in target_types
                ]
                refinement[ref_key] = _glossary_refinement_manual_completion_info(
                    entry_type,
                    target_entries,
                    ref_info,
                    getattr(self, 'config', {}) or {},
                    output_path=panel_state.get('_glossary_path'),
                    completed_at=now,
                )
                changed = True

        if changed:
            _write_glossary_progress_atomic(_rp, _d)

        return {
            'changed': changed,
            'data': _d,
            'row_updates': _gp_row_updates_for_targets(_d, target_specs),
            'refresh_refinement_rows': bool(refinement_keys),
            'stats': _gp_stats_for_dict(_d),
        }

    _gp_apply_mark_completed_to_progress_unlocked = _gp_apply_mark_completed_to_progress

    def _gp_apply_mark_completed_to_progress(_rp, target_specs):
        """Mark as Completed under the extractor's progress-file lock."""
        with _locked_glossary_progress_write(_rp):
            return _gp_apply_mark_completed_to_progress_unlocked(_rp, target_specs)

    def _gp_load_usage_inputs(progress_callback=None):
        """Glossary entries, source chapters and progress for footnotes (RG 25243-25285).

        Returns ``((entries, chapters, progress_data), None)`` or ``(None, (kind, title,
        message))``; the desktop shows the message box, mobile a sheet.
        """
        if callable(progress_callback):
            progress_callback("Finding glossary file...")
        glossary_path = _find_glossary_file()
        if not glossary_path or not os.path.isfile(glossary_path):
            return None, (
                'information',
                "No Glossary Found",
                "No glossary file could be found for this progress file.",
            )
        try:
            if callable(progress_callback):
                progress_callback(f"Parsing glossary: {os.path.basename(glossary_path)}")
            entries = parse_glossary_file(glossary_path)
        except Exception as exc:
            return None, ('critical', "Glossary Footnote", f"Could not parse glossary:\n{exc}")
        if not entries:
            return None, ('information', "Glossary Footnote", "The glossary has no parseable entries.")
        try:
            translate_special = bool(panel_state.get('translate_special'))
            if callable(progress_callback):
                progress_callback("Reading EPUB source chapters...")
            chapters = read_epub_spine_chapters(fp, translate_special=translate_special, include_text=True)
        except Exception as exc:
            return None, ('critical', "Glossary Footnote", f"Could not read EPUB source chapters:\n{exc}")
        if not chapters:
            return None, ('information', "Glossary Footnote", "No source chapters could be read from this EPUB.")
        if callable(progress_callback):
            progress_callback(f"Loaded {len(entries)} glossary entries and {len(chapters)} source chapters.")
        progress_path = _find_gp_for_file(fp) or gp_path
        if callable(progress_callback):
            progress_callback("Loading latest glossary progress data...")
        progress_data = _gp_load_progress_dict(progress_path) if progress_path and os.path.isfile(progress_path) else dict(_current_gp_data() or {})
        return (entries, chapters, progress_data), None

    def _gp_footnote_markdown_to_html(markdown_text):
        def _split_entry_line(line):
            content = line[2:].strip()
            status = ""
            for marker, marker_status in ((WARNING_PREFIX, "warning"), (CHECK_PREFIX, "confirmed")):
                marker_prefix = f"{marker} "
                if content.startswith(marker_prefix):
                    status = marker_status
                    content = content[len(marker_prefix):].strip()
                    break
            label = content
            details = ""
            if content.endswith(")"):
                details_start = content.rfind(" (")
                if details_start > -1:
                    label = content[:details_start].strip()
                    details = content[details_start + 2:-1].strip()
            return status, label, details

        def _entry_details_html(details):
            if not details:
                return ""
            parts = [part.strip() for part in details.split(";") if part.strip()]
            tags = []
            desc_parts = []
            if parts:
                tags.append(parts[0])
            if len(parts) > 1 and parts[1].casefold() in {"male", "female", "unknown", "nonbinary", "non-binary", "other"}:
                tags.append(parts[1])
                desc_parts = parts[2:]
            else:
                desc_parts = parts[1:]
            blocks = ['<div class="entry-meta">']
            if tags:
                tag_text = " | ".join(html_lib.escape(part) for part in tags)
                blocks.append(f'<div class="entry-tags">{tag_text}</div>')
            if desc_parts:
                desc_text = html_lib.escape("; ".join(desc_parts))
                blocks.append(f'<div class="entry-desc">{desc_text}</div>')
            blocks.append("</div>")
            return "".join(blocks)

        html_parts = [
            "<html><head><style>",
            "body { background: #2d2d2d; color: #f2f2f2; font-family: Segoe UI, Arial, sans-serif; font-size: 15px; line-height: 1.35; }",
            "h1, h2 { margin: 0 0 12px 0; font-size: 26px; font-weight: 700; }",
            "h3 { margin: 18px 0 10px 0; font-size: 20px; font-weight: 700; }",
            "p { margin: 4px 0 10px 0; }",
            "ul.meta-list { margin: 0 0 14px 28px; }",
            "ul.entry-list { margin: 6px 0 0 28px; }",
            "li { margin: 5px 0; }",
            "li.entry { margin: 0 0 12px 0; }",
            ".entry-label { color: #ffffff; font-weight: 600; }",
            ".entry-warning { color: #ffca66; font-weight: 700; }",
            ".entry-confirmed { color: #8bdc81; font-weight: 700; }",
            ".entry-meta { margin-top: 2px; color: #c5c9d1; font-size: 13px; line-height: 1.35; }",
            ".entry-tags { color: #9ecbff; font-weight: 600; }",
            ".entry-desc { color: #c7cbd2; }",
            ".saved-path { color: #d8d8d8; font-family: Consolas, monospace; }",
            ".loading { color: #d8d8d8; font-size: 18px; font-style: italic; margin-top: 12px; }",
            "</style></head><body>",
        ]
        list_kind = None
        in_entries = False

        def close_list():
            nonlocal list_kind
            if list_kind:
                html_parts.append("</ul>")
                list_kind = None

        def open_list(kind):
            nonlocal list_kind
            if list_kind == kind:
                return
            close_list()
            class_name = "entry-list" if kind == "entries" else "meta-list"
            html_parts.append(f'<ul class="{class_name}">')
            list_kind = kind

        for raw_line in str(markdown_text or "").splitlines():
            line = raw_line.strip()
            if not line:
                close_list()
                continue
            if line.startswith("### "):
                close_list()
                heading = line[4:].strip()
                in_entries = heading.casefold() == "matched glossary entries"
                html_parts.append(f"<h3>{html_lib.escape(heading)}</h3>")
            elif line.startswith("## "):
                close_list()
                in_entries = False
                html_parts.append(f"<h2>{html_lib.escape(line[3:].strip())}</h2>")
            elif line.startswith("# "):
                close_list()
                in_entries = False
                html_parts.append(f"<h1>{html_lib.escape(line[2:].strip())}</h1>")
            elif line.startswith("- "):
                if in_entries:
                    open_list("entries")
                    status, label, details = _split_entry_line(line)
                    marker = ""
                    if status == "warning":
                        marker = f'<span class="entry-warning">{html_lib.escape(WARNING_PREFIX)}</span> '
                    elif status == "confirmed":
                        marker = f'<span class="entry-confirmed">{html_lib.escape(CHECK_PREFIX)}</span> '
                    html_parts.append(
                        '<li class="entry">'
                        f'<div class="entry-label">{marker}{html_lib.escape(label)}</div>'
                        f"{_entry_details_html(details)}"
                        "</li>"
                    )
                else:
                    open_list("meta")
                    html_parts.append(f"<li>{html_lib.escape(line[2:].strip())}</li>")
            else:
                close_list()
                escaped = html_lib.escape(line)
                if os.path.isabs(line):
                    html_parts.append(f'<p class="saved-path">{escaped}</p>')
                else:
                    html_parts.append(f"<p>{escaped}</p>")
        close_list()
        html_parts.append("</body></html>")
        return "".join(html_parts)

    def _gp_source_chapter_maps(chapters):
        chapter_by_index = {
            int(chapter.get("chapter_index", idx)): chapter
            for idx, chapter in enumerate(chapters)
        }
        chapter_by_filename_key = {}
        for idx, chapter in enumerate(chapters):
            for filename in (
                chapter.get("filename"),
                chapter.get("member_path"),
            ):
                for key in _gp_filename_keys(filename):
                    chapter_by_filename_key.setdefault(key, chapter)
        return chapter_by_index, chapter_by_filename_key

    def _gp_source_chapter_for_row(ci, chapter_by_index, chapter_by_filename_key):
        fname = (panel_state.get('chapter_map') or {}).get(ci, "")
        for key in _gp_filename_keys(fname):
            if key in chapter_by_filename_key:
                return chapter_by_filename_key[key]
        return chapter_by_index.get(ci)

    def _gp_output_text_for_row(ci, progress_entry_key, progress_entry, fname, progress_data):
        progress_entry = progress_entry if isinstance(progress_entry, dict) else {}
        output_file = progress_entry.get('output_file') or fname
        display_info = {
            'output_file': output_file,
            'original_filename': progress_entry.get('original_filename') or fname,
            'original_basename': progress_entry.get('original_basename') or fname,
            'num': _gp_source_chapter_num(ci, fname),
            'info': progress_entry,
            'progress_key': progress_entry_key,
        }
        resolved_output_file = None
        resolved_output_path = None
        resolver_progress = progress_data if isinstance(progress_data, dict) else prog
        try:
            resolved_output_file, resolved_output_path = self._resolve_existing_output_path(
                output_dir,
                output_file,
                display_info=display_info,
                prog=resolver_progress,
            )
        except Exception:
            resolved_output_file, resolved_output_path = None, None
        if not resolved_output_path:
            return None, False, resolved_output_file or output_file
        try:
            return read_translated_output_text(resolved_output_path), True, resolved_output_file or output_file
        except Exception:
            return None, False, resolved_output_file or output_file

    def _gp_chapter_for_footnote(ci, chapter, progress_entry, resolved_output_file, fname):
        progress_entry = progress_entry if isinstance(progress_entry, dict) else {}
        chapter_for_footnote = dict(chapter)
        chapter_for_footnote.update(
            {
                "chapter_index": ci,
                "progress_entry": progress_entry,
                "progress_chapter_num": _gp_display_chapter_num(ci, fname),
                "spine_number": (panel_state.get('spine_index_map') or {}).get(ci, chapter.get("spine_number", ci + 1)),
                "output_file": resolved_output_file or progress_entry.get('output_file') or fname,
            }
        )
        return chapter_for_footnote

    skip_unmatched_config_key = "glossary_progress_skip_unmatched_entries"

    def _gp_skip_unmatched_entries():
        try:
            return bool((getattr(self, "config", {}) or {}).get(skip_unmatched_config_key, True))
        except Exception:
            return True

    def _gp_set_skip_unmatched_entries(enabled):
        try:
            self.config[skip_unmatched_config_key] = bool(enabled)
            if hasattr(self, "save_config"):
                self.save_config(show_message=False)
        except Exception:
            pass

    def _gp_write_completed_summary(entries, chapters, progress_data, progress_callback=None, skip_unmatched_entries=None):
        progress_data = progress_data if isinstance(progress_data, dict) else {}
        if skip_unmatched_entries is None:
            skip_unmatched_entries = _gp_skip_unmatched_entries()
        chapter_by_index, chapter_by_filename_key = _gp_source_chapter_maps(chapters)
        progress_entries = _gp_chapter_entry_map(progress_data)
        completed, failed, merged = _gp_sets(progress_data)
        summary_indices = sorted((completed | merged) - failed)
        title = progress_data.get("book_title") or os.path.splitext(os.path.basename(fp))[0] or "Glossary Footnotes"
        lines = [
            f"# {title} - Completed Glossary Footnotes",
            "",
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
        ]
        missing_chapters = []
        missing_output = []
        total_sections = len(summary_indices)
        if not summary_indices:
            lines.append("No completed or merged glossary-progress chapters were found.")
        section_results = []

        def _build_summary_section(section_idx, ci):
            chapter = _gp_source_chapter_for_row(ci, chapter_by_index, chapter_by_filename_key)
            if not chapter:
                return {
                    "section_idx": section_idx,
                    "chapter_index": ci,
                    "chapter_num": _gp_display_chapter_num(ci, (panel_state.get('chapter_map') or {}).get(ci, '')),
                    "text": "",
                    "missing_chapter": ci,
                    "missing_output": None,
                }
            fname = (panel_state.get('chapter_map') or {}).get(ci, chapter.get("filename", ""))
            progress_entry_key, progress_entry = progress_entries.get(ci, (None, {}))
            output_text, output_available, resolved_output_file = _gp_output_text_for_row(
                ci,
                progress_entry_key,
                progress_entry,
                fname,
                progress_data,
            )
            missing_output_value = None
            if not output_available:
                missing_output_value = (
                    ci,
                    resolved_output_file
                    or (progress_entry.get('output_file') if isinstance(progress_entry, dict) else "")
                    or fname,
                )
            section_text = build_chapter_footnote(
                entries,
                _gp_chapter_for_footnote(ci, chapter, progress_entry, resolved_output_file, fname),
                progress_data,
                output_text=output_text,
                output_available=output_available,
                skip_unmatched_entries=skip_unmatched_entries,
            ).rstrip()
            return {
                "section_idx": section_idx,
                "chapter_index": ci,
                "chapter_num": _gp_display_chapter_num(ci, fname),
                "text": section_text,
                "missing_chapter": None,
                "missing_output": missing_output_value,
            }

        if summary_indices:
            from concurrent.futures import as_completed

            worker_count = min(total_sections, max(2, min(8, os.cpu_count() or 4)))
            if callable(progress_callback):
                progress_callback(
                    {
                        "current": 0,
                        "total": total_sections,
                        "chapter_num": "",
                        "status": f"starting {worker_count} threads",
                    }
                )
            completed_sections = 0
            with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="GlossarySummary") as executor:
                futures = [
                    executor.submit(_build_summary_section, section_idx, ci)
                    for section_idx, ci in enumerate(summary_indices, start=1)
                ]
                for future in as_completed(futures):
                    result = future.result()
                    section_results.append(result)
                    completed_sections += 1
                    if callable(progress_callback):
                        progress_callback(
                            {
                                "current": completed_sections,
                                "total": total_sections,
                                "chapter_num": result.get("chapter_num", ""),
                                "status": "generated",
                            }
                        )

        for result in sorted(section_results, key=lambda item: item.get("section_idx", 0)):
            if result.get("missing_chapter") is not None:
                missing_chapters.append(result["missing_chapter"])
                continue
            if result.get("missing_output"):
                missing_output.append(result["missing_output"])
            lines.append(result.get("text", ""))
            lines.append("")
        if missing_chapters:
            lines.extend(
                [
                    "## Summary warnings",
                    "",
                    "Completed or merged chapters that could not be mapped back to source text:",
                ]
            )
            lines.extend(
                f"- Ch.{_gp_display_chapter_num(ci, (panel_state.get('chapter_map') or {}).get(ci, ''))}"
                for ci in missing_chapters
            )
            lines.append("")
        if missing_output:
            lines.extend(
                [
                    "## Output-file warnings",
                    "",
                    "Completed or merged chapters whose translated output file could not be resolved:",
                ]
            )
            lines.extend(
                f"- Ch.{_gp_display_chapter_num(ci, (panel_state.get('chapter_map') or {}).get(ci, ''))}: {path}"
                for ci, path in missing_output
            )
            lines.append("")
        content = "\n".join(lines).rstrip() + "\n"
        book_base = os.path.splitext(os.path.basename(fp))[0]
        safe_base = re.sub(r'[<>:"/\\\\|?*]+', "_", str(book_base or "book")).strip(" ._") or "book"
        summary_dir = os.path.join(output_dir or os.getcwd(), "glossary_footnotes")
        os.makedirs(summary_dir, exist_ok=True)
        summary_path = os.path.join(summary_dir, f"{safe_base}_completed_glossary_footnotes.md")
        with open(summary_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(content)
        return summary_path, content

    def _gp_collect_footnotes(entries, chapters, progress_data, target_values):
        """Footnote markdown of selected chapter rows (RG 25806-25842).

        Returns ``(parts, missing, missing_output)``.
        """
        skip_unmatched_entries = _gp_skip_unmatched_entries()
        chapter_by_index, chapter_by_filename_key = _gp_source_chapter_maps(chapters)

        parts = []
        missing = []
        missing_output = []
        for value in target_values:
            try:
                ci = int(value)
            except (TypeError, ValueError):
                continue
            chapter = _gp_source_chapter_for_row(ci, chapter_by_index, chapter_by_filename_key)
            if not chapter:
                missing.append(ci)
                continue
            fname = (panel_state.get('chapter_map') or {}).get(ci, chapter.get("filename", ""))
            progress_entry_key, progress_entry = _gp_chapter_entry_map(progress_data).get(ci, (None, {}))
            output_text, output_available, resolved_output_file = _gp_output_text_for_row(
                ci,
                progress_entry_key,
                progress_entry,
                fname,
                progress_data,
            )
            if not output_available:
                missing_output.append((ci, resolved_output_file or (progress_entry.get('output_file') if isinstance(progress_entry, dict) else "") or fname))
                continue
            parts.append(
                build_chapter_footnote(
                    entries,
                    _gp_chapter_for_footnote(ci, chapter, progress_entry, resolved_output_file, fname),
                    progress_data,
                    output_text=output_text,
                    output_available=output_available,
                    skip_unmatched_entries=skip_unmatched_entries,
                ).rstrip()
            )
        return parts, missing, missing_output

    _gp_folder = os.path.dirname(gp_path)

    def _find_glossary_file(_gp_dir=_gp_folder, _epub_path=fp):
        """Find the glossary file (csv/json/txt) in the same directory as the progress file."""
        bases = []
        for candidate_path in (
            _epub_path,
            glossary_progress_source_path,
        ):
            base = os.path.splitext(os.path.basename(candidate_path or ""))[0]
            if base and base not in bases:
                bases.append(base)
        progress_stem = os.path.splitext(os.path.basename(gp_path))[0]
        if progress_stem.endswith("_glossary_progress"):
            progress_stem = progress_stem[: -len("_glossary_progress")]
        if progress_stem and progress_stem not in bases:
            bases.append(progress_stem)
        # Search priority: book-specific glossary > generic glossary
        for ext in ['.csv', '.json', '.txt', '.md']:
            for base in bases:
                for pattern in [
                    os.path.join(_gp_dir, f"{base}_glossary{ext}"),
                    os.path.join(_gp_dir, f"{base}{ext}"),
                ]:
                    if os.path.isfile(pattern):
                        return pattern
            generic = os.path.join(_gp_dir, f"glossary{ext}")
            if os.path.isfile(generic):
                return generic
        # Also check parent dir (if progress is in Glossary/ subfolder)
        parent = os.path.dirname(_gp_dir)
        if os.path.basename(_gp_dir).lower() == 'glossary':
            for ext in ['.csv', '.json', '.txt', '.md']:
                generic = os.path.join(parent, f"glossary{ext}")
                if os.path.isfile(generic):
                    return generic
                for base in bases:
                    pattern = os.path.join(parent, f"{base}_glossary{ext}")
                    if os.path.isfile(pattern):
                        return pattern
        return None

    def _gp_apply_remove_from_progress(_rp, removable_specs):
        """Remove from progress under the extractor's lock (RG 26165-26236).

        ``removable_specs`` are ``(kind, value)`` targets (``chapter`` index,
        ``refinement`` key, ``minimal_pass``).  Returns ``(changed, data,
        remove_minimal_pass)``.
        """
        removable_targets = [(None, spec) for spec in removable_specs]
        with _locked_glossary_progress_write(_rp):
            _d = _gp_load_progress_dict(_rp)
            
            indices_to_remove = set(value for _, (kind, value) in removable_targets if kind == 'chapter')
            refinement_keys_to_remove = set(value for _, (kind, value) in removable_targets if kind == 'refinement')
            remove_minimal_pass = any(kind == 'minimal_pass' for _, (kind, _value) in removable_targets)
            changed = False
            if remove_minimal_pass and 'minimal_pass' in _d:
                del _d['minimal_pass']
                changed = True
            if indices_to_remove:
                removed_indices = _d.get('manual_removed_indices', [])
                if not isinstance(removed_indices, list):
                    removed_indices = []
                removed_set = set()
                for value in removed_indices:
                    mapped_ci = _gp_index_for_progress_value(value, _d)
                    if mapped_ci is not None:
                        removed_set.add(mapped_ci)
                removed_set.update(indices_to_remove)
                _d['manual_removed_indices'] = sorted(removed_set)
                _d['manual_removed_session_id'] = _d.get('progress_session_id')
                changed = True

            for key in ('completed', 'skipped', 'failed', 'merged_indices', 'in_progress'):
                lst = _d.get(key, [])
                new_lst = []
                for v in lst:
                    mapped_ci = _gp_index_for_progress_value(v, _d)
                    if not indices_to_remove or mapped_ci not in indices_to_remove:
                        new_lst.append(v)
                if len(new_lst) != len(lst):
                    _d[key] = new_lst
                    changed = True

            qa_map = _d.get('qa_issues_found', {})
            if indices_to_remove and isinstance(qa_map, dict):
                new_qa_map = {}
                for k, v in qa_map.items():
                    mapped_ci = _gp_index_for_progress_value(k, _d)
                    keep = mapped_ci not in indices_to_remove if mapped_ci is not None else True
                    if keep:
                        new_qa_map[k] = v
                if len(new_qa_map) != len(qa_map):
                    _d['qa_issues_found'] = new_qa_map
                    changed = True

            chapters = _d.get('chapters', {})
            if indices_to_remove and isinstance(chapters, dict):
                new_chapters = {}
                chapters_changed = False
                for k, v in chapters.items():
                    ci = _gp_index_for_entry(v, k, _d) if isinstance(v, dict) else None
                    keep = ci not in indices_to_remove if ci is not None else True
                    if keep:
                        new_chapters[k] = v
                    else:
                        chapters_changed = True
                        changed = True
                if chapters_changed or len(new_chapters) != len(chapters):
                    _d['chapters'] = new_chapters
                    changed = True

            if refinement_keys_to_remove and isinstance(_d.get('refinement'), dict):
                for ref_key in refinement_keys_to_remove:
                    if ref_key in _d['refinement']:
                        del _d['refinement'][ref_key]
                        changed = True
            

            if changed:
                _write_glossary_progress_atomic(_rp, _d)
        return changed, _d, remove_minimal_pass

    _namespace = dict(locals())
    _namespace.pop('self', None)
    return types.SimpleNamespace(**_namespace)


# ---------------------------------------------------------------------------
# Public API (names of the U5 shared contract; the mobile Book page's Glossary tab)
# ---------------------------------------------------------------------------


def find_glossary_progress(owner, source_path, *, source_override=None):
    """Path of a source's ``<book>_glossary_progress.json`` (or None).

    ``source_override`` is the paired EPUB whose glossary progress is shown instead
    (desktop: Parallel EPUB Pair).  Search roots: the output override, the app folder,
    then ``owner.base_dir``; per root the per-book Glossary folder first.
    """
    locator = glossary_progress_locator(owner, source_path, source_override)
    return locator._find_gp_for_file(source_path)


def find_glossary_file(model):
    """The glossary CSV/JSON/TXT/MD belonging to a glossary progress file (or None)."""
    return model._find_glossary_file()


def glossary_refinement_expected(owner, glossary_entries, source_path=None):
    """Always-visible refinement rows for the active entry types (aggregate + per type)."""
    locator = glossary_progress_locator(owner, source_path or '', None)
    return locator._glossary_refinement_expected_entries(glossary_entries)


def open_glossary_progress(owner, source_path, gp_path=None, *, output_dir=None, prog=None,
                           source_override=None, source_filenames=None):
    """Load the Glossary Progress model of a source (None when no progress file exists).

    ``source_filenames`` restricts and orders the chapter rows (Parallel EPUB Pair).
    """
    locator = glossary_progress_locator(owner, source_path, source_override)
    if gp_path is None:
        gp_path = locator._find_gp_for_file(source_path)
    if not gp_path or not os.path.isfile(gp_path):
        return None
    return make_glossary_progress_model(
        owner,
        source_path,
        gp_path,
        glossary_progress_source_path=source_override,
        glossary_progress_source_filenames=source_filenames,
        output_dir=output_dir,
        prog=prog if isinstance(prog, dict) else {},
        find_gp_for_file=locator._find_gp_for_file,
        glossary_refinement_expected_entries=locator._glossary_refinement_expected_entries,
        refinement_type_key=locator._refinement_type_key,
    )


def reload_glossary_progress(model):
    """Re-read the model's progress file; returns the new dict ({} when it disappeared)."""
    path = model._find_gp_for_file(model.fp) or model.gp_path
    if not path or not os.path.isfile(path):
        return {}
    return model._gp_load_progress_dict(path)


def glossary_progress_signature(path):
    """(mtime_ns, ctime_ns, size) of a glossary progress file, None when missing."""
    try:
        stat = os.stat(path)
        return (stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size)
    except OSError:
        return None


def glossary_chapter_map(model):
    """``(chapter_map, total, spine_index_map)``: rows index -> filename / spine position."""
    state = model.panel_state
    return dict(state.get('chapter_map') or {}), state.get('total', 0), dict(state.get('spine_index_map') or {})


class GlossaryIndex:
    """Maps glossary-progress values (entries, list values, chapter numbers) to rows."""

    def __init__(self, model):
        self.model = model

    def index_for_entry(self, info, key=None, data=None):
        return self.model._gp_index_for_entry(info, key, data)

    def index_for_value(self, value, data):
        return self.model._gp_index_for_progress_value(value, data)

    def index_for_actual_num(self, actual_num, data=None):
        return self.model._gp_index_for_actual_num(actual_num, data)


def glossary_status_cache(model, data):
    """Status sets of every row (completed/skipped/failed/merged/in_progress/issues...)."""
    return model._gp_status_cache(data)


#: Icon per glossary-progress status (RG _gp_display_for / _gp_refinement_rows).
GLOSSARY_STATUS_ICONS = {
    'completed': '✅',
    'skipped': '⏭️',
    'skipped_empty': '📄',
    'skipped_image_only': '📸',
    'skipped_title_header_only': '🏷️',
    'failed': '❌',
    'qa_failed': '❌',
    'refine_failed': '💀',
    'merged': '\U0001f517',
    'in_progress': '\U0001f504',
    'partially_in_progress': '\U0001f504',
    'not_refined': '✨',
    'not_completed': '⬜',
}
#: Legend groups of the Glossary tab chips (RG _gp_make_cycle targets 26296-26311).
GLOSSARY_STATUS_GROUPS = {
    'completed': ('completed',),
    'skipped': ('skipped', 'skipped_empty', 'skipped_image_only', 'skipped_title_header_only'),
    'in_progress': ('in_progress', 'partially_in_progress'),
    'failed': ('failed', 'qa_failed'),
    'merged': ('merged',),
    'remaining': ('not_completed', 'not_translated', 'no_tts'),
    'not_refined': ('not_refined',),
    'refine_failed': ('refine_failed',),
}


@dataclass
class GlossaryRow:
    """One row of the Glossary Progress list (``display`` is the desktop line)."""

    kind: str                      # 'minimal_pass' | 'chapter' | 'refinement'
    key: Any                       # chapter index / refinement key / 'minimal_pass'
    display: str
    status: str
    color: str
    icon: str
    chapter_index: Optional[int] = None
    filename: str = ''
    issues: List[str] = dataclass_field(default_factory=list)


def glossary_rows(model, data=None):
    """Rows in desktop order: Minimal pass, chapters, then refinement rows."""
    data = data if isinstance(data, dict) else model._current_gp_data()
    rows = []
    minimal = model._gp_minimal_pass_row(data)
    if minimal:
        key, display, status = minimal
        rows.append(GlossaryRow('minimal_pass', key, display, status, model._gp_color_for(status),
                                GLOSSARY_STATUS_ICONS.get(status, '⬜')))
    cache = model._gp_status_cache(data)
    chapter_map = model.panel_state.get('chapter_map') or {}
    for ci in range(model.panel_state.get('total', 0)):
        fname = chapter_map.get(ci, f'chapter {ci + 1}')
        display, status = model._gp_display_for(ci, fname, data, cache)
        _status, issues = model._gp_status_for(ci, data, cache)
        rows.append(GlossaryRow('chapter', ci, display, status, model._gp_color_for(status),
                                GLOSSARY_STATUS_ICONS.get(status, '⬜'), ci, fname, list(issues or [])))
    for key, display, status in model._gp_refinement_rows(data):
        rows.append(GlossaryRow('refinement', key, display, status, model._gp_color_for(status),
                                GLOSSARY_STATUS_ICONS.get(status, '⬜')))
    return rows


def glossary_stats(model, data=None):
    """Legend totals (chapters + Minimal pass + refinement rows)."""
    data = data if isinstance(data, dict) else model._current_gp_data()
    return model._gp_stats_for_dict(data)


def _target_specs(targets):
    specs = []
    for target in targets or []:
        if isinstance(target, GlossaryRow):
            if target.kind == 'chapter':
                specs.append(('chapter', target.chapter_index))
            elif target.kind == 'refinement':
                specs.append(('refinement', target.key))
            else:
                specs.append(('minimal_pass', target.key))
        else:
            specs.append(tuple(target))
    return specs


def mark_glossary_completed(model, targets):
    """Mark chapter / refinement rows completed; returns the payload (data, row updates, stats)."""
    path = model._find_gp_for_file(model.fp)
    if not path or not os.path.isfile(path):
        return {'changed': False, 'data': {}}
    specs = [spec for spec in _target_specs(targets) if spec[0] in ('chapter', 'refinement')]
    return model._gp_apply_mark_completed_to_progress(path, specs)


def remove_glossary_progress(model, targets):
    """Remove rows from glossary progress (the next run extracts them again).

    Returns ``{'changed', 'data', 'remove_minimal_pass'}``.
    """
    path = model._find_gp_for_file(model.fp)
    if not path or not os.path.isfile(path):
        return {'changed': False, 'data': {}, 'remove_minimal_pass': False}
    changed, data, remove_minimal_pass = model._gp_apply_remove_from_progress(path, _target_specs(targets))
    return {'changed': changed, 'data': data, 'remove_minimal_pass': remove_minimal_pass}


def glossary_footnotes(model, chapter_indices, progress_callback=None):
    """Glossary footnote markdown of chapter rows.

    Returns ``{'markdown', 'missing', 'missing_output', 'error'}``; ``error`` is a
    ``(kind, title, message)`` when the glossary / source could not be loaded.
    """
    loaded, error = model._gp_load_usage_inputs(progress_callback)
    if error is not None:
        return {'markdown': '', 'missing': [], 'missing_output': [], 'error': error}
    entries, chapters, progress_data = loaded
    parts, missing, missing_output = model._gp_collect_footnotes(entries, chapters, progress_data, chapter_indices)
    return {
        'markdown': ("\n\n".join(parts) + "\n") if parts else '',
        'missing': missing,
        'missing_output': missing_output,
        'error': None,
    }


def write_glossary_summary(model, progress_callback=None):
    """Write ``glossary_footnotes/<book>_completed_glossary_footnotes.md``.

    Returns ``(summary_path, content)``; raises RuntimeError when inputs are missing.
    """
    loaded, error = model._gp_load_usage_inputs(
        (lambda message: progress_callback({"stage": "loading", "message": str(message)}))
        if callable(progress_callback) else None
    )
    if error is not None:
        raise RuntimeError("Could not load glossary entries, source chapters, or progress data.")
    entries, chapters, progress_data = loaded
    return model._gp_write_completed_summary(
        entries,
        chapters,
        progress_data,
        progress_callback=progress_callback,
        skip_unmatched_entries=model._gp_skip_unmatched_entries(),
    )


__all__ = [
    'GLOSSARY_STATUS_GROUPS',
    'GLOSSARY_STATUS_ICONS',
    'GlossaryIndex',
    'GlossaryRow',
    'build_chapter_footnote',
    'find_glossary_file',
    'find_glossary_progress',
    'glossary_chapter_map',
    'glossary_footnotes',
    'glossary_progress_locator',
    'glossary_progress_signature',
    'glossary_refinement_expected',
    'glossary_rows',
    'glossary_stats',
    'glossary_status_cache',
    'make_glossary_progress_model',
    'mark_glossary_completed',
    'open_glossary_progress',
    'reload_glossary_progress',
    'remove_glossary_progress',
    'write_glossary_summary',
]
