"""The Glossary Editor document: parse, save and edit glossary files without Qt (U6).

Shared by the desktop Glossary Editor (GlossaryManager_GUI: the ``_setup_glossary_editor_tab``
closure, ``_on_tree_double_click`` and ``convert_glossary_format`` keep their dialogs and
widgets and call these functions) and the mobile Glossary editor. Every function body is moved
verbatim from GlossaryManager_GUI.py (frozen at U6_BASE_SHA in tests/test_glossary_document.py)
with explicit parameters in place of the editor's ``self`` state; the documented edits are
listed there and in tests/parity/DISCREPANCIES.md ("U6 Glossary Editor document").

The editor state lives on a *document* object with the attribute names the desktop keeps on
TranslatorGUI, so the desktop passes ``self`` and GUI-free callers pass a
:class:`GlossaryDocument`:

* ``current_glossary_data`` / ``current_glossary_format`` ('token_csv', 'list' or 'dict') /
  ``current_glossary_sections``;
* ``current_gender_tracker_data`` / ``current_gender_tracker_path`` /
  ``_pending_gender_decisions`` / ``_gender_variants_pending_save``;
* ``glossary_column_fields`` and ``_original_translated_map`` (the last saved translated
  names, which drive "Update output files on save").

Rows of the editor view are addressed by a *source ref*: the entry index for list / token CSV
glossaries, the key for dict (JSON object) glossaries. Find/Replace works on a *row* object
with ``columnCount()``, ``text(column)``, ``setText(column, value)``, ``ref()`` and
``set_ref(value)``; column 0 is the row number and column ``i`` shows
``glossary_column_fields[i - 1]`` (:class:`EditorRow`).

GUI-free entry points (mobile):

* :class:`GlossaryDocument` -- one open glossary, driven like the desktop editor (load, rows,
  edit / resolve gender, save with "Update output files", delete, clean, remove duplicates,
  trim, filter, find / replace with the output-file fallback, undo / redo, hide-unused scan,
  export, Save As, Convert Format, backups), with :class:`EditorOwner` for the TranslatorGUI
  values the editor reads;
* :func:`resolve_editor_glossaries` / :func:`unified_glossary_editor_files` -- the files the
  editor lists for an input;
* :class:`GlossaryPromptProfiles` / :class:`RefinementPromptProfiles` -- the Glossary Manager
  prompt profiles on prompt_profiles' Default-plus-named engine.

Python 3.10 compatible; never imports PySide6, translator_gui or dpi_setup. The Balanced/Full
extractor (extract_glossary_from_epub) is imported lazily, as the desktop editor does, for
gender resolution on save and for Remove Duplicates.
"""

import copy as _copy
import json
import os
import re
import shutil as _shutil
import time

from gender_tracking import (
    BINARY_GENDERS,
    collapse_tracked_gender_variants,
    display_gender,
    editor_gender_status,
    normalize_bias,
    normalize_entries_gender,
    normalize_gender,
    normalize_threshold,
    occurrence_bounds,
    resolved_storage_gender,
    tracker_entry_for_raw,
    tracker_path_for_glossary,
)
from glossary_usage import (
    build_prepared_output_index,
    entry_matches_output_index,
    html_to_text,
)

#: GlossaryManager_GUI's directory on desktop (both modules live in src/); the editor also
#: looks for a shared ``Glossary`` folder next to it.
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

#: The desktop editor keeps at most this many undo snapshots (``self._undo_max``).
EDITOR_UNDO_MAX = 50

#: Glossary formats the editor works with (``current_glossary_format``).
LIST_FORMATS = ('list', 'token_csv')


def _noop_log(_message):
    return None


# ---------------------------------------------------------------------------
# Gender tracking in the editor (module-level helpers of GlossaryManager_GUI)
# ---------------------------------------------------------------------------

def gender_resolution_summary(tracker_entry, stored_gender, threshold, bias):
    """Build the resolution-dialog history model from shared tracker rules."""
    automatic_entry = dict(tracker_entry or {})
    automatic_entry["decision"] = "auto"
    status = editor_gender_status(automatic_entry, stored_gender, threshold, bias)
    stats = status.get("stats", {})
    total = sum(int(values.get("count", 0) or 0) for values in stats.values())
    genders = {}
    for gender in BINARY_GENDERS:
        values = stats.get(gender, {})
        first, last = occurrence_bounds(automatic_entry, gender)
        genders[gender] = {
            "count": int(values.get("count", 0) or 0),
            "ratio": float(values.get("ratio", 0.0) or 0.0),
            "first": first,
            "last": last,
        }
    changes = [change for change in automatic_entry.get("changes", []) if isinstance(change, dict)]
    return {
        "status": status,
        "calculated_auto_gender": resolved_storage_gender(automatic_entry, stored_gender),
        "total": total,
        "genders": genders,
        "flip_count": len(changes),
        "latest_flips": changes[-5:],
    }


def collect_glossary_filter_values(value_states, *, restrict_to_visible=False):
    """Return checked filter values, optionally limited to search matches.

    ``value_states`` contains ``(raw_value, is_visible, is_checked)`` tuples.
    Limiting an active search to visible values gives the popup the spreadsheet
    behavior users expect: typing a value and applying it filters to those
    search results instead of silently retaining every hidden checked value.
    """
    return {
        raw_value
        for raw_value, is_visible, is_checked in value_states
        if is_checked and (is_visible or not restrict_to_visible)
    }


def load_editor_gender_tracker(glossary_path):
    """Load an associated tracker without making editor-open mutate it."""
    tracker_path = tracker_path_for_glossary(glossary_path)
    if not tracker_path or not os.path.exists(tracker_path):
        return None, tracker_path
    try:
        with open(tracker_path, "r", encoding="utf-8") as tracker_file:
            tracker = json.load(tracker_file)
        if isinstance(tracker, dict) and isinstance(tracker.get("entries"), dict):
            return tracker, tracker_path
    except Exception:
        pass
    return None, tracker_path


def editor_entry_has_gender(entry, custom_types):
    if not isinstance(entry, dict):
        return False
    entry_type = str(entry.get("type", "character") or "character").strip()
    config = custom_types.get(entry_type) if isinstance(custom_types, dict) else None
    if isinstance(config, dict):
        return bool(config.get("has_gender", False))
    return entry_type.casefold() == "character"


def prepare_editor_gender_tracking(entries, glossary_path, custom_types, tracking_disabled=False):
    env_disabled = str(os.getenv('GLOSSARY_SKIP_GENDER_TRACKING', '0')).strip().lower() in {
        '1', 'true', 'yes', 'on'
    }
    if tracking_disabled or env_disabled:
        return list(entries or []), None, tracker_path_for_glossary(glossary_path), 0
    tracker, tracker_path = load_editor_gender_tracker(glossary_path)
    if not tracker:
        return list(entries or []), None, tracker_path, 0
    prepared, collapsed = collapse_tracked_gender_variants(
        entries,
        tracker,
        has_gender=lambda entry: editor_entry_has_gender(entry, custom_types),
        score_entry=lambda entry: sum(
            1 for key, value in entry.items()
            if not str(key).startswith("_") and value not in (None, "", [], {})
        ),
    )
    return prepared, tracker, tracker_path, collapsed


def editor_gender_settings(owner, config):
    threshold = getattr(
        owner,
        'glossary_gender_noise_threshold_var',
        config.get('glossary_gender_noise_threshold', 10),
    )
    bias = getattr(
        owner,
        'glossary_gender_tracking_bias_var',
        config.get('glossary_gender_tracking_bias', 'none'),
    )
    return threshold, bias


# ---------------------------------------------------------------------------
# Document state
# ---------------------------------------------------------------------------

def reset_document(doc):
    """Forget the loaded glossary (``_clear_glossary_editor`` and the editor tab set-up)."""
    doc.current_glossary_data = None
    doc.current_glossary_format = None
    doc.current_gender_tracker_data = None
    doc.current_gender_tracker_path = ''
    doc._pending_gender_decisions = {}
    doc._gender_variants_pending_save = 0


def apply_parse_payload(doc, payload):
    """Install a :func:`parse_glossary_file` result on ``doc`` (``apply_loaded_glossary_result``).

    Sets the data, format, sections, gender tracker, column fields and the translated-name
    baseline; returns ``(entries, column_fields)``.
    """
    entries = payload.get('entries', [])
    column_fields = payload.get('column_fields', [])
    doc.current_glossary_data = payload.get('current_data')
    doc.current_glossary_format = payload.get('current_format')
    doc.current_glossary_sections = payload.get('sections', [])
    doc.current_gender_tracker_data = payload.get('gender_tracker')
    doc.current_gender_tracker_path = payload.get('gender_tracker_path', '')
    doc._pending_gender_decisions = {}
    doc._gender_variants_pending_save = int(payload.get('gender_variants_collapsed', 0) or 0)
    doc.glossary_column_fields = list(column_fields)
    if doc.current_glossary_format in ['list', 'token_csv']:
        doc._original_translated_map = {
            idx: entry.get('translated_name', '') for idx, entry in enumerate(doc.current_glossary_data or [])
            if isinstance(entry, dict)
        }
    elif doc.current_glossary_format == 'dict':
        doc._original_translated_map = dict((doc.current_glossary_data or {}).get('entries', {}))
    else:
        doc._original_translated_map = {}
    return entries, column_fields


def configured_entry_types(config, custom_entry_types=None):
    """The entry-type configuration the editor status line ranks custom types by: the owner's
    ``custom_entry_types``, else ``config['custom_entry_types']``."""
    return custom_entry_types or (config or {}).get('custom_entry_types', {}) or {}


#: Entry types the Entry Type Configuration list puts first and offers no × for (the desktop list).
BUILTIN_ENTRY_TYPES = ('character', 'terms')


def normalize_legacy_entry_types(custom_entry_types):
    """Legacy ``term`` -> ``terms`` in *custom_entry_types* (in place; returned).

    Moved from the Glossary Manager's Entry Type Configuration (GlossaryManager_GUI)."""
    # Normalize legacy key "term" -> "terms"
    if 'term' in custom_entry_types and 'terms' not in custom_entry_types:
        custom_entry_types['terms'] = custom_entry_types.pop('term')
    # If both exist, prefer "terms" and drop legacy duplicate
    if 'term' in custom_entry_types and 'terms' in custom_entry_types:
        custom_entry_types.pop('term', None)
    return custom_entry_types


def sorted_entry_types(custom_entry_types):
    """``[(type, config), ...]``: built-in first, then custom alphabetically (the desktop list order)."""
    # Sort types: built-in first, then custom alphabetically
    return sorted(custom_entry_types.items(),
                  key=lambda x: (x[0] not in ['character', 'terms'], x[0]))


def add_entry_type(custom_entry_types, text, has_gender):
    """"Add Type": ``(type name, None)`` after adding it to *custom_entry_types* (in place), or
    ``(None, (title, message))`` - the desktop warning - for a blank or duplicate name."""
    type_name = text.strip().lower()
    if not type_name:
        return None, ("Invalid Input", "Please enter a type name")

    if type_name in custom_entry_types:
        return None, ("Duplicate Type", f"Type '{type_name}' already exists")

    # Add the new type
    custom_entry_types[type_name] = {
        'enabled': True,
        'has_gender': has_gender
    }
    return type_name, None


def entry_type_remove_warning(type_name):
    """The desktop warning ``(title, message)`` when *type_name* may not be removed, else None."""
    if type_name in ['character', 'term']:
        return ("Cannot Remove", "Built-in types cannot be removed")
    return None


def description_removed_flag(action, field):
    """Custom Fields: the ``custom_field_description_removed`` value after *action* ('add' /
    'remove') of *field*, or None when the flag stays (any field but "description")."""
    if field.lower() == 'description':
        # If user manually adds "description" back, clear the removal flag;
        # if user manually removes "description", set flag to prevent re-adding
        return action != 'add'
    return None


def custom_fields_flag_updates(old_fields, new_fields):
    """``{'custom_field_description_removed': bool}`` for an edit of the whole Custom Fields list
    (the removed fields, then the added ones, through ``description_removed_flag``), else {}."""
    old_fields = [str(f) for f in (old_fields or [])]
    new_fields = [str(f) for f in (new_fields or [])]
    flag = None
    for field in old_fields:
        if field not in new_fields:
            changed = description_removed_flag('remove', field)
            flag = changed if changed is not None else flag
    for field in new_fields:
        if field not in old_fields:
            changed = description_removed_flag('add', field)
            flag = changed if changed is not None else flag
    return {} if flag is None else {'custom_field_description_removed': flag}


class EditorOwner:
    """The TranslatorGUI state the desktop editor reads besides the document, for one config.

    TranslatorGUI start-up (owner_state.ConfigStateMixin._init_config_state) seeds
    ``custom_glossary_fields = ['description']`` unless the user removed it and sets the
    attributes below from the config; the editor code reads them with ``getattr(self, ...)``,
    so GUI-free callers pass this object where the desktop passes ``self``.
    tests/test_glossary_document.py compares every value with a HeadlessOwner.
    """

    def __init__(self, config=None):
        config = dict(config or {})
        if (
            not config.get('custom_glossary_fields', [])
            and not config.get('custom_field_description_removed', False)
        ):
            config['custom_glossary_fields'] = ['description']
        self.config = config
        self.custom_entry_types = config.get('custom_entry_types', {
            'character': {'enabled': True, 'has_gender': True},
            'term': {'enabled': True, 'has_gender': False},
            'surnames': {'enabled': True, 'has_gender': False},
            'titles': {'enabled': True, 'has_gender': True},
            'locations': {'enabled': True, 'has_gender': False},
            'nicknames': {'enabled': True, 'has_gender': True}
        })
        self.glossary_gender_noise_threshold_var = config.get('glossary_gender_noise_threshold', 10)
        self.glossary_gender_tracking_bias_var = config.get('glossary_gender_tracking_bias', 'none')
        self.enable_parallel_extraction_var = config.get('enable_parallel_extraction', True)
        self.extraction_workers_var = config.get(
            'extraction_workers',
            min(8, max(2, (os.cpu_count() or 4) // 2)),
        )


def document_entry_count(data, fmt):
    """Entries in the loaded glossary (the Trim / Filter dialogs' "Total entries")."""
    return len(data) if fmt in ['list', 'token_csv'] else len(data.get('entries', {}))


def is_new_format_data(data, fmt):
    """Whether the glossary has typed entries (``type`` / ``raw_name`` / ``translated_name``)."""
    return (fmt in ['list', 'token_csv'] and 
                   data and 
                   'type' in data[0])


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def display_glossary_path(path, roots=3):
    """Display the last few path parts for glossary selectors."""
    if not path:
        return ''
    parts = []
    current = os.path.normpath(path)
    while current:
        parent, name = os.path.split(current)
        if name:
            parts.append(name)
        if not parent or parent == current or len(parts) >= roots:
            break
        current = parent
    return '/'.join(reversed(parts)) if parts else os.path.basename(path)


def unified_glossary_folder_key(config):
    """The Unified Glossary subfolder the current settings resolve to."""
    try:
        import unified_glossary
        return unified_glossary.describe_folder_key(
            config.get('unified_glossary_source_language', 'auto'),
            bool(config.get('unified_glossary_combine_all_languages', False)),
            config.get('output_language') or os.environ.get('OUTPUT_LANGUAGE') or 'English',
        )
    except Exception:
        return 'auto'


def unified_glossary_editor_files(config, base_dir='', folder_key=None, module_dir=_MODULE_DIR):
    """Existing unified glossary files, this run's language pair first."""
    try:
        import unified_glossary
    except Exception:
        return []
    roots = []
    override_dir = os.environ.get('OUTPUT_DIRECTORY') or config.get('output_directory')
    if override_dir and str(override_dir).strip():
        roots.append(os.path.join(os.path.abspath(str(override_dir)), 'Glossary'))
    shared_env = str(os.environ.get('GLOSSARY_SHARED_DIR', '') or '').strip()
    if shared_env:
        roots.append(os.path.abspath(shared_env))
    for shared_root in (
        os.path.join(str(base_dir or ''), 'Glossary'),
        os.path.join(module_dir, 'Glossary'),
        os.path.join(os.getcwd(), 'Glossary'),
    ):
        roots.append(shared_root)

    preferred = str(
        os.environ.get('UNIFIED_GLOSSARY_RESOLVED_KEY', '')
        or (folder_key() if folder_key is not None else unified_glossary_folder_key(config))
    ).casefold()
    found, seen = [], set()
    for root in roots:
        try:
            unified_root = unified_glossary.unified_root(root)
            if not os.path.isdir(unified_root):
                continue
            keys = sorted(
                os.listdir(unified_root),
                key=lambda name: (name.casefold() != preferred, name.casefold()),
            )
        except Exception:
            continue
        for key in keys:
            csv_path = unified_glossary.unified_paths(root, key)[2]
            norm = os.path.normcase(os.path.abspath(csv_path))
            if norm in seen or not os.path.isfile(csv_path):
                continue
            seen.add(norm)
            found.append(csv_path)
    return found


def unified_glossary_shared_dir(config):
    """The shared Glossary/ folder, resolved the way a run resolves it."""
    override_dir = os.environ.get('OUTPUT_DIRECTORY') or config.get('output_directory')
    if override_dir and str(override_dir).strip():
        return os.path.join(os.path.abspath(str(override_dir)), 'Glossary')
    try:
        from app_paths import _get_app_dir
        return os.path.join(_get_app_dir(), 'Glossary')
    except Exception:
        return os.path.join(os.getcwd(), 'Glossary')


def unified_rebuild_settings(config, shared_dir):
    """The settings snapshot Unified Glossary "🔄 Rebuild Now" passes to ``unified_glossary.rebuild_now``
    (``GlossaryManagerMixin._rebuild_unified_glossary_now``)."""
    settings = {
        'OUTPUT_LANGUAGE': config.get('output_language') or os.environ.get('OUTPUT_LANGUAGE') or 'English',
        'UNIFIED_GLOSSARY_COMBINE_ALL_LANGUAGES': '1' if config.get('unified_glossary_combine_all_languages', False) else '0',
        'UNIFIED_GLOSSARY_EXCLUDE_GENDER_ENTRIES': '1' if config.get('unified_glossary_exclude_gender_entries', True) else '0',
        'GLOSSARY_SHARED_DIR': shared_dir,
    }
    return settings


def resolve_editor_glossaries(source_paths, *, config, override_dir=None,
                              auto_loaded_glossary_path=None, manual_glossary_path=None,
                              manual_glossary_map=None, output_base_getter=None, base_dir='',
                              module_dir=_MODULE_DIR):
    """The glossary files the editor lists for these input sources (``auto_select_current_glossary``).

    Already-resolved paths (auto-loaded, manual, mapped) come first, then the per-book and
    shared ``Glossary`` folders in the order the glossary mode implies. Returns
    ``(found_glossaries, found_glossary_sources)``: ``[(display, path)]`` and
    ``{path: source_path}``.
    """
    # Collect any paths that the main GUI has already resolved
    # (via auto-mapping or manual load) so we always try them FIRST
    # regardless of which branch we take below.
    _already_mapped = []
    try:
        for _p in (auto_loaded_glossary_path, manual_glossary_path):
            if _p and os.path.exists(_p):
                _already_mapped.append(_p)
        _mmap = manual_glossary_map or {}
        if isinstance(_mmap, dict):
            for _p in _mmap.values():
                if _p and os.path.exists(_p) and _p not in _already_mapped:
                    _already_mapped.append(_p)
    except Exception:
        _already_mapped = []

    ext_priority = ['.csv', '.json', '.txt', '.md']
    mode = str(config.get('auto_glossary_mode', 'off')).lower()
    auto_mapping_on = bool(config.get('append_glossary_auto_load', False))
    use_per_book = (mode == 'minimal') or (auto_mapping_on and mode not in ('balanced', 'full', 'single_pass'))

    found_glossaries = []  # list of (display_name, full_path)
    found_glossary_sources = {}

    for source_path in source_paths:
        if not source_path or not os.path.exists(source_path):
            continue
        base = os.path.splitext(os.path.basename(source_path))[0]

        candidates = []

        # Prefer any already-resolved path that looks associated
        # with THIS input source (same basename stem).
        _base_lc = base.lower()
        for _p in _already_mapped:
            try:
                _p_stem = os.path.splitext(os.path.basename(_p))[0].lower()
                _dir_name = os.path.basename(os.path.dirname(_p)).lower()
                if (_base_lc in _p_stem) or (_base_lc == _dir_name):
                    if _p not in candidates:
                        candidates.append(_p)
            except Exception:
                pass
        # Also try single-selection case: any resolved path when
        # there is exactly one input source selected.
        if len(source_paths) == 1:
            for _p in _already_mapped:
                if _p not in candidates:
                    candidates.append(_p)

        if override_dir and override_dir.strip():
            abs_override = os.path.abspath(override_dir)
            glossary_folder = os.path.join(abs_override, 'Glossary')
            book_dir = os.path.join(abs_override, base)

            if use_per_book:
                for ext in ext_priority:
                    candidates.append(os.path.join(book_dir, f"glossary{ext}"))
                for ext in ext_priority:
                    candidates.append(os.path.join(book_dir, 'Glossary', f"glossary{ext}"))
                for ext in ext_priority:
                    candidates.append(os.path.join(glossary_folder, base, f"{base}_glossary{ext}"))
                for ext in ext_priority:
                    candidates.append(os.path.join(glossary_folder, base, f"{base}{ext}"))
                for ext in ext_priority:
                    candidates.append(os.path.join(glossary_folder, f"{base}_glossary{ext}"))
                for ext in ext_priority:
                    candidates.append(os.path.join(glossary_folder, f"{base}{ext}"))
            else:
                for ext in ext_priority:
                    candidates.append(os.path.join(glossary_folder, base, f"{base}_glossary{ext}"))
                for ext in ext_priority:
                    candidates.append(os.path.join(glossary_folder, base, f"{base}{ext}"))
                for ext in ext_priority:
                    candidates.append(os.path.join(glossary_folder, f"{base}_glossary{ext}"))
                for ext in ext_priority:
                    candidates.append(os.path.join(glossary_folder, f"{base}{ext}"))
                for ext in ext_priority:
                    candidates.append(os.path.join(book_dir, f"glossary{ext}"))
                for ext in ext_priority:
                    candidates.append(os.path.join(book_dir, 'Glossary', f"glossary{ext}"))
        else:
            # No override — check auto/manual paths first, then EPUB parent dir
            auto_path = auto_loaded_glossary_path
            manual_path = manual_glossary_path
            if auto_path and os.path.exists(auto_path):
                candidates.append(auto_path)
            if manual_path and os.path.exists(manual_path):
                candidates.append(manual_path)
            source_parent = os.path.dirname(os.path.abspath(source_path))
            output_dirs = [os.path.join(source_parent, base)]
            try:
                if callable(output_base_getter):
                    output_dirs.append(
                        os.path.join(output_base_getter(source_path), base)
                    )
            except Exception:
                pass
            for out_dir in output_dirs:
                for ext in ext_priority:
                    candidates.append(os.path.join(out_dir, f"glossary{ext}"))
                for ext in ext_priority:
                    candidates.append(
                        os.path.join(out_dir, 'Glossary', f"glossary{ext}")
                    )

            shared_dirs = []
            for shared_root in (
                os.path.join(str(base_dir or ''), 'Glossary'),
                os.path.join(module_dir, 'Glossary'),
                os.path.join(os.getcwd(), 'Glossary'),
            ):
                if shared_root and shared_root not in shared_dirs:
                    shared_dirs.append(shared_root)
            for glossary_folder in shared_dirs:
                for ext in ext_priority:
                    candidates.append(
                        os.path.join(
                            glossary_folder,
                            base,
                            f"{base}_glossary{ext}",
                        )
                    )
                for ext in ext_priority:
                    candidates.append(
                        os.path.join(glossary_folder, base, f"{base}{ext}")
                    )
                for ext in ext_priority:
                    candidates.append(
                        os.path.join(glossary_folder, f"{base}_glossary{ext}")
                    )
                for ext in ext_priority:
                    candidates.append(
                        os.path.join(glossary_folder, f"{base}{ext}")
                    )

        # Pick first match for this book
        for cand in candidates:
            if os.path.exists(cand):
                display = display_glossary_path(cand)
                if not any(fp == cand for _, fp in found_glossaries):
                    found_glossaries.append((display, cand))
                    found_glossary_sources[cand] = source_path
                break
    return found_glossaries, found_glossary_sources


# ---------------------------------------------------------------------------
# Parse
# ---------------------------------------------------------------------------

def glossary_type_count_summary(entries, configured_types=None, max_custom_types=7):
    """Return built-in and prioritized custom entry-type counts for editor status."""
    counts = {}
    display_names = {}
    for entry in entries or []:
        if not isinstance(entry, dict):
            continue
        raw_type = str(entry.get('type') or '').strip()
        if not raw_type:
            continue
        normalized = raw_type.casefold()
        counts[normalized] = counts.get(normalized, 0) + 1
        display_names.setdefault(normalized, raw_type)

    characters = counts.get('character', 0)
    terms = counts.get('terms', 0) + counts.get('term', 0)
    parts = [f"Characters: {characters}", f"Terms: {terms}"]

    configured_types = configured_types or {}
    configured_meta = {}
    if isinstance(configured_types, dict):
        for order, (type_name, type_config) in enumerate(configured_types.items()):
            normalized = str(type_name or '').strip().casefold()
            if not normalized:
                continue
            type_config = type_config if isinstance(type_config, dict) else {}
            configured_meta[normalized] = {
                'has_gender': bool(type_config.get('has_gender', False)),
                'order': order,
                'display_name': str(type_name).strip(),
            }

    built_in_types = {'character', 'term', 'terms'}
    custom_types = [
        normalized
        for normalized, count in counts.items()
        if normalized not in built_in_types and count > 0
    ]
    custom_types.sort(
        key=lambda normalized: (
            not configured_meta.get(normalized, {}).get('has_gender', False),
            -counts[normalized],
            configured_meta.get(normalized, {}).get('order', float('inf')),
            normalized,
        )
    )

    try:
        custom_limit = max(0, int(max_custom_types))
    except (TypeError, ValueError):
        custom_limit = 7
    for normalized in custom_types[:custom_limit]:
        raw_label = (
            configured_meta.get(normalized, {}).get('display_name')
            or display_names.get(normalized)
            or normalized
        )
        label = str(raw_label).replace('_', ' ').strip().title()
        parts.append(f"{label}: {counts[normalized]}")

    return ", ".join(parts)

def parse_token_glossary(lines, custom_glossary_fields=(), custom_entry_types=None):
    """Parse token-efficient glossary CSV text without touching Qt widgets."""
    entries = []
    sections = []
    current_section = None
    header_columns = ['raw_name', 'translated_name', 'gender', 'description']
    default_extra_columns = []
    try:
        import PatternManager as _pm
        pf = getattr(_pm, 'PATTERN_ADDITIONAL_FIELDS', [])
        if isinstance(pf, (list, tuple)):
            default_extra_columns.extend(pf)
    except Exception:
        pass
    default_extra_columns.extend(custom_glossary_fields)
    extra_columns = list(default_extra_columns)
    custom_types = custom_entry_types or {
        'character': {'enabled': True, 'has_gender': True},
        'terms': {'enabled': True, 'has_gender': False},
        'surnames': {'enabled': True, 'has_gender': False},
        'titles': {'enabled': True, 'has_gender': True},
        'locations': {'enabled': True, 'has_gender': False},
        'nicknames': {'enabled': True, 'has_gender': True},
    }

    type_map = {}
    for t in custom_types.keys():
        t_lower = t.lower()
        type_map[t_lower] = t
        if not t_lower.endswith('s'):
            type_map[f"{t_lower}s"] = t

    import re

    def _parse_token_entry_line(line):
        body = line[2:].strip()

        def _split_head_desc(text):
            paren_depth = 0
            bracket_depth = 0
            for idx, ch in enumerate(text):
                if ch == '(' and bracket_depth == 0:
                    paren_depth += 1
                elif ch == ')' and bracket_depth == 0 and paren_depth > 0:
                    paren_depth -= 1
                elif ch == '[' and paren_depth == 0:
                    bracket_depth += 1
                elif ch == ']' and paren_depth == 0 and bracket_depth > 0:
                    bracket_depth -= 1
                elif ch == ':' and paren_depth == 0 and bracket_depth == 0:
                    return text[:idx].rstrip(), text[idx + 1:].strip()
            return text, ""

        head, desc = _split_head_desc(body)

        extra_values = {}

        def _pull_custom_tails(text):
            while True:
                tail = re.search(r'\s+\(([^()]*)\)\s*$', text)
                if not tail or ':' not in tail.group(1):
                    return text.rstrip()
                for paren_m in re.finditer(r'\(([^)]+)\)', tail.group(0)):
                    content = paren_m.group(1).strip()
                    if ':' in content:
                        k, v = content.split(':', 1)
                        extra_values[k.strip()] = v.strip()
                text = text[:tail.start()].rstrip()

        head = _pull_custom_tails(head)
        bracket = ""
        gender_match = re.search(r'\s*\[([^\]]*)\]\s*$', head)
        if gender_match:
            bracket = (gender_match.group(1) or '').strip()
            head = head[:gender_match.start()].rstrip()
            head = _pull_custom_tails(head)

        equal_match = re.match(r'^(?P<raw>.+?)\s*=\s*(?P<translated>.+?)\s*$', head)
        if equal_match:
            raw_name = (equal_match.group('raw') or '').strip()
            translated = (equal_match.group('translated') or '').strip()
            return translated, raw_name, bracket, desc, extra_values

        legacy_match = re.match(r'^(?P<translated>.*)\s+\((?P<raw>.*?)\)\s*$', head)
        if legacy_match:
            translated = (legacy_match.group('translated') or '').strip()
            raw_name = (legacy_match.group('raw') or '').strip()
            return translated, raw_name, bracket, desc, extra_values

        return None

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        if line.lower().startswith('glossary columns:'):
            cols_text = line.split(':', 1)[1]
            header_columns = [c.strip() for c in cols_text.split(',') if c.strip()]
            if len(header_columns) < 4:
                header_columns = ['raw_name', 'translated_name', 'gender', 'description']
            positional = {'translated_name', 'raw_name', 'gender'}
            extra_columns = [c for c in header_columns if c.lower() not in positional]
            if not extra_columns:
                extra_columns = list(default_extra_columns)
            continue
        if line.startswith('===') and line.endswith('==='):
            section_name = line.strip('=').strip()
            current_section = section_name
            sections.append(section_name)
            continue
        if not line.startswith('* '):
            continue

        parsed_line = _parse_token_entry_line(line)
        if not parsed_line:
            continue
        translated, raw_name, bracket, desc, extra_values = parsed_line
        if desc and ' | ' in desc:
            parts = desc.split(' | ')
            desc = parts[0].strip()
            for part in parts[1:]:
                if ':' in part:
                    k, v = part.split(':', 1)
                    extra_values[k.strip()] = v.strip()

        if desc and extra_columns:
            remaining_cols = [c for c in extra_columns if c not in extra_values]
            if remaining_cols:
                paren_match = re.search(r'\s*\((.+)\)\s*$', desc)
                if paren_match:
                    paren_content = paren_match.group(1).strip()
                    cols_in_paren = [
                        c for c in remaining_cols
                        if re.search(re.escape(c) + r'\s*:', paren_content, re.IGNORECASE)
                    ]
                    if cols_in_paren:
                        positions = []
                        for c in cols_in_paren:
                            col_match = re.search(re.escape(c) + r'\s*:\s*', paren_content, re.IGNORECASE)
                            if col_match:
                                positions.append((col_match.start(), col_match.end(), c))
                        positions.sort(key=lambda x: x[0])
                        for i, (_start, end, col) in enumerate(positions):
                            if i + 1 < len(positions):
                                val = paren_content[end:positions[i + 1][0]].strip().rstrip(',').strip()
                            else:
                                val = paren_content[end:].strip()
                            extra_values[col] = val
                        desc = desc[:paren_match.start()].strip().rstrip(',').strip()
                        remaining_cols = [c for c in remaining_cols if c not in extra_values]

                for col in remaining_cols:
                    if not desc:
                        break
                    col_esc = re.escape(col)
                    comma_match = re.compile(r',\s*' + col_esc + r'\s*:\s*', re.IGNORECASE).search(desc)
                    if comma_match:
                        extra_values[col] = desc[comma_match.end():].strip()
                        desc = desc[:comma_match.start()].strip()
                        continue
                    start_match = re.compile(r'^' + col_esc + r'\s*:\s*', re.IGNORECASE).search(desc)
                    if start_match:
                        extra_values[col] = desc[start_match.end():].strip()
                        desc = ''

        entry = {
            'type': type_map.get((current_section or 'terms').lower(), 'terms'),
            'raw_name': raw_name,
            'translated_name': translated,
            'gender': bracket,
        }
        if desc and 'description' not in extra_values:
            entry['description'] = desc
        if current_section:
            entry['_section'] = current_section
        for col in extra_columns:
            if col in extra_values:
                entry[col] = extra_values[col]
        entries.append(entry)

    return entries, sections

def parse_glossary_file(path, config, custom_entry_types=None):
    """Read and parse a glossary file for the editor without touching Qt widgets."""
    all_fields = set()
    entries = []
    current_data = None
    current_format = None
    sections = []

    if path.lower().endswith('.csv'):
        with open(path, 'r', encoding='utf-8') as f:
            raw_content = f.read()
        lines = raw_content.splitlines(True)

        token_style = False
        for line in lines:
            lstrip = line.lstrip()
            if lstrip.startswith('===') or lstrip.startswith('* '):
                token_style = True
                break
        if not token_style and lines and lines[0].lower().startswith('glossary columns:'):
            token_style = True

        if token_style:
            entries, sections = parse_token_glossary(
                lines, config.get('custom_glossary_fields', []), custom_entry_types
            )
            current_data = entries
            current_format = 'token_csv'
            for entry in entries:
                all_fields.update(entry.keys())
        else:
            import csv
            _GSEP = '\x1F'
            if _GSEP in raw_content:
                rows = []
                for line in raw_content.split('\n'):
                    line = line.strip()
                    if line:
                        rows.append([p.strip() for p in line.split(_GSEP)])
            else:
                from io import StringIO
                rows = list(csv.reader(StringIO(raw_content)))

            header_names = None
            data_start = 0
            if rows and rows[0] and rows[0][0].strip().lower() == 'type':
                header_names = [h.strip().lower() for h in rows[0]]
                data_start = 1

            if header_names:
                col_map = {name: i for i, name in enumerate(header_names)}
                expected_cols = len(header_names)
                desc_idx = col_map.get('description', -1)
                cols_after_desc = expected_cols - desc_idx - 1 if desc_idx >= 0 else 0

                for row in rows[data_start:]:
                    if not row or len(row) < 3:
                        continue
                    entry = {}
                    excess = len(row) - expected_cols
                    if excess > 0 and desc_idx >= 0:
                        for name, idx in col_map.items():
                            if idx < desc_idx:
                                entry[name] = row[idx] if idx < len(row) else ''
                        after_desc_names = [
                            n for n, i in sorted(col_map.items(), key=lambda x: x[1])
                            if i > desc_idx
                        ]
                        for offset, name in enumerate(after_desc_names):
                            tail_idx = len(row) - cols_after_desc + offset
                            entry[name] = row[tail_idx] if tail_idx < len(row) else ''
                        desc_end = len(row) - cols_after_desc
                        entry['description'] = ', '.join(row[desc_idx:desc_end])
                    else:
                        for name, idx in col_map.items():
                            entry[name] = row[idx] if idx < len(row) else ''
                    entries.append(entry)
            else:
                for row in rows[data_start:]:
                    if len(row) >= 3:
                        entry = {
                            'type': row[0],
                            'raw_name': row[1],
                            'translated_name': row[2],
                        }
                        if len(row) > 3:
                            entry['gender'] = row[3]
                        if len(row) > 4:
                            entry['description'] = ', '.join(row[4:])
                        entries.append(entry)

            current_data = entries
            current_format = 'list'
            for entry in entries:
                all_fields.update(entry.keys())
    else:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, dict):
            if 'entries' in data:
                current_data = data
                current_format = 'dict'
                for original, translated in data['entries'].items():
                    entry = {'original': original, 'translated': translated}
                    entries.append(entry)
                    all_fields.update(entry.keys())
            else:
                current_data = {'entries': data}
                current_format = 'dict'
                for original, translated in data.items():
                    entry = {'original': original, 'translated': translated}
                    entries.append(entry)
                    all_fields.update(entry.keys())
        elif isinstance(data, list):
            current_data = data
            current_format = 'list'
            for item in data:
                if isinstance(item, dict):
                    all_fields.update(item.keys())
                    entries.append(item)

    gender_tracker = None
    gender_tracker_path = tracker_path_for_glossary(path)
    gender_variants_collapsed = 0
    if current_format in ['list', 'token_csv']:
        custom_types = custom_entry_types or config.get('custom_entry_types', {})
        entries, gender_tracker, gender_tracker_path, gender_variants_collapsed = (
            prepare_editor_gender_tracking(
                entries,
                path,
                custom_types,
                tracking_disabled=bool(config.get('glossary_skip_gender_tracking', False)),
            )
        )
        current_data = entries
        all_fields = set()
        for entry in entries:
            if isinstance(entry, dict):
                all_fields.update(entry.keys())

    # One spelling per Gender column, whatever the file holds: `male`
    # and `Male` also have to match the same Gender filter.
    normalize_entries_gender(current_data if isinstance(current_data, list) else entries)
    if current_format in ['list', 'token_csv'] and entries and 'type' in entries[0]:
        column_fields = []
        if any('_section' in e for e in entries):
            column_fields.append('_section')
        column_fields.extend(['type', 'raw_name', 'translated_name', 'gender'])
        for entry in entries:
            for field in entry.keys():
                if field.startswith('_'):
                    continue
                if field not in column_fields:
                    column_fields.append(field)
    else:
        standard_fields = [
            'original_name', 'name', 'original', 'translated', 'gender',
            'title', 'group_affiliation', 'traits', 'how_they_refer_to_others',
            'locations',
        ]
        column_fields = [field for field in standard_fields if field in all_fields]
        column_fields.extend(sorted(all_fields - set(standard_fields)))

    stats = [f"Total entries: {len(entries)}"]
    if current_format in ['list', 'token_csv'] and entries and 'type' in entries[0]:
        stats.append(glossary_type_count_summary(
            entries, configured_entry_types(config, custom_entry_types)
        ))
    elif current_format == 'list':
        chars = sum(1 for e in entries if 'original_name' in e or 'name' in e)
        locs = sum(1 for e in entries if 'locations' in e and e['locations'])
        stats.append(f"Characters: {chars}, Locations: {locs}")
    if gender_variants_collapsed:
        stats.append(
            f"{gender_variants_collapsed} tracked gender duplicate"
            f"{'s' if gender_variants_collapsed != 1 else ''} pending save"
        )

    return {
        'path': path,
        'entries': entries,
        'current_data': current_data,
        'current_format': current_format,
        'sections': sections,
        'column_fields': column_fields,
        'stats_text': " | ".join(stats),
        'gender_tracker': gender_tracker,
        'gender_tracker_path': gender_tracker_path,
        'gender_variants_collapsed': gender_variants_collapsed,
    }


def reparse_glossary_file(doc, path, config, custom_entry_types=None):
    """Re-read ``path`` into ``doc`` synchronously (the editor's undo / redo restore path).

    Unlike :func:`parse_glossary_file` this path keeps the file's gender variants (no tracker
    collapse) and matches ``.csv`` case-sensitively, as the desktop editor does. Sets the data,
    format and (token CSV) sections on ``doc``; returns ``(entries, column_fields, stats_text)``.
    """
    def parse_token_efficient_glossary(lines):
        return parse_token_glossary(lines, config.get('custom_glossary_fields', []), custom_entry_types)

    # Prepare accumulator for field discovery
    all_fields = set()

    # Try CSV first
    if path.endswith('.csv'):
        # Peek to detect token-efficient format
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        token_style = False
        for l in lines:
            lstrip = l.lstrip()
            if lstrip.startswith('===') or lstrip.startswith('* '):
                token_style = True
                break
        if not token_style and lines and lines[0].lower().startswith('glossary columns:'):
            token_style = True

        if token_style:
            entries, sections = parse_token_efficient_glossary(lines)
            doc.current_glossary_data = entries
            doc.current_glossary_format = 'token_csv'
            doc.current_glossary_sections = sections
            for e in entries:
                all_fields.update(e.keys())
        else:
            import csv
            entries = []
            _GSEP = '\x1F'
            with open(path, 'r', encoding='utf-8') as f:
                raw_content = f.read()

            if _GSEP in raw_content:
                # New Unit Separator format
                rows = []
                for _line in raw_content.split('\n'):
                    _line = _line.strip()
                    if _line:
                        rows.append([p.strip() for p in _line.split(_GSEP)])
            else:
                # Legacy comma-separated format
                from io import StringIO
                reader = csv.reader(StringIO(raw_content))
                rows = list(reader)

            # Detect header row: first row whose first cell is 'type'
            header_names = None
            data_start = 0
            if rows and rows[0] and rows[0][0].strip().lower() == 'type':
                header_names = [h.strip().lower() for h in rows[0]]
                data_start = 1

            if header_names:
                # Build column map from header
                col_map = {}  # name -> index
                for i, name in enumerate(header_names):
                    col_map[name] = i
                expected_cols = len(header_names)

                # Find the description column index if it exists
                desc_idx = col_map.get('description', -1)
                # Count how many named columns come AFTER description
                cols_after_desc = 0
                if desc_idx >= 0:
                    cols_after_desc = expected_cols - desc_idx - 1

                for row in rows[data_start:]:
                    if not row or len(row) < 3:
                        continue
                    entry = {}
                    excess = len(row) - expected_cols

                    if excess > 0 and desc_idx >= 0:
                        # Row has more cells than expected — description has unquoted commas.
                        # Assign columns before description normally
                        for name, idx in col_map.items():
                            if idx < desc_idx:
                                entry[name] = row[idx] if idx < len(row) else ''
                        # Assign columns after description from the END of the row
                        after_desc_names = [n for n, i in sorted(col_map.items(), key=lambda x: x[1]) if i > desc_idx]
                        for offset, name in enumerate(after_desc_names):
                            # Read from the tail end of the row
                            tail_idx = len(row) - cols_after_desc + offset
                            entry[name] = row[tail_idx] if tail_idx < len(row) else ''
                        # Everything in the middle is the description
                        desc_end = len(row) - cols_after_desc
                        entry['description'] = ', '.join(row[desc_idx:desc_end])
                    else:
                        # Normal case — row has expected number of cells (or fewer)
                        for name, idx in col_map.items():
                            entry[name] = row[idx] if idx < len(row) else ''

                    entries.append(entry)
            else:
                # No header — fall back to positional parsing
                for row in rows[data_start:]:
                    if len(row) >= 3:
                        entry = {
                            'type': row[0],
                            'raw_name': row[1],
                            'translated_name': row[2]
                        }
                        if len(row) > 3:
                            entry['gender'] = row[3]
                        if len(row) > 4:
                            entry['description'] = ', '.join(row[4:])
                        entries.append(entry)

            doc.current_glossary_data = entries
            doc.current_glossary_format = 'list'
            for e in entries:
                all_fields.update(e.keys())
    else:
        # JSON format
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        entries = []

        if isinstance(data, dict):
            if 'entries' in data:
                doc.current_glossary_data = data
                doc.current_glossary_format = 'dict'
                for original, translated in data['entries'].items():
                    entry = {'original': original, 'translated': translated}
                    entries.append(entry)
                    all_fields.update(entry.keys())
            else:
                doc.current_glossary_data = {'entries': data}
                doc.current_glossary_format = 'dict'
                for original, translated in data.items():
                    entry = {'original': original, 'translated': translated}
                    entries.append(entry)
                    all_fields.update(entry.keys())

        elif isinstance(data, list):
            doc.current_glossary_data = data
            doc.current_glossary_format = 'list'
            for item in data:
                all_fields.update(item.keys())
                entries.append(item)

    normalize_entries_gender(entries)
    # Set up columns based on new format
    if doc.current_glossary_format in ['list', 'token_csv'] and entries and 'type' in entries[0]:
        # New simple format
        column_fields = []
        # Show section if present
        if any('_section' in e for e in entries):
            column_fields.append('_section')
        column_fields.extend(['type', 'raw_name', 'translated_name', 'gender'])

        # Include description/custom fields
        for entry in entries:
            for field in entry.keys():
                if field.startswith('_'):
                    continue
                if field not in column_fields:
                    column_fields.append(field)

        # Check for any custom fields
        for entry in entries:
            for field in entry.keys():
                if field.startswith('_'):
                    continue
                if field not in column_fields:
                    column_fields.append(field)
    else:
        # Old format compatibility
        standard_fields = ['original_name', 'name', 'original', 'translated', 'gender', 
                         'title', 'group_affiliation', 'traits', 'how_they_refer_to_others', 
                         'locations']

        column_fields = []
        for field in standard_fields:
            if field in all_fields:
                column_fields.append(field)

        custom_fields = sorted(all_fields - set(standard_fields))
        column_fields.extend(custom_fields)

    # Update stats
    stats = []
    stats.append(f"Total entries: {len(entries)}")

    if doc.current_glossary_format in ['list', 'token_csv'] and entries and 'type' in entries[0]:
        # New format stats
        stats.append(glossary_type_count_summary(
            entries, configured_entry_types(config, custom_entry_types)
        ))
    elif doc.current_glossary_format == 'list':
        # Old format stats
        chars = sum(1 for e in entries if 'original_name' in e or 'name' in e)
        locs = sum(1 for e in entries if 'locations' in e and e['locations'])
        stats.append(f"Characters: {chars}, Locations: {locs}")
    return entries, column_fields, " | ".join(stats)


# ---------------------------------------------------------------------------
# Editor view (rows, display values, column filters, status line)
# ---------------------------------------------------------------------------

def editor_row_specs(column_fields, data, fmt):
    specs = []
    fields = list(column_fields or [])
    if fmt in ['list', 'token_csv']:
        for source_idx, entry in enumerate(data or []):
            entry_dict = dict(entry) if isinstance(entry, dict) else {}
            specs.append((source_idx, source_idx, entry_dict))
    elif fmt == 'dict':
        data = data or {}
        entries = data.get('entries', data) if isinstance(data, dict) else {}
        if isinstance(entries, dict):
            for source_idx, (key, value) in enumerate(entries.items()):
                if isinstance(value, dict):
                    entry_dict = dict(value)
                    entry_dict.setdefault('original', key)
                    entry_dict.setdefault('raw_name', key)
                else:
                    entry_dict = {'original': key, 'translated': value}
                specs.append((source_idx, key, entry_dict))
    return fields, specs


def editor_display_value(entry, field):
    """Render a glossary value exactly as it appears in the editor tree."""
    value = entry.get(field, '') if isinstance(entry, dict) else ''
    if isinstance(value, list):
        value = ', '.join(str(v) for v in value)
    elif isinstance(value, dict):
        value = ', '.join(f"{k}: {v}" for k, v in value.items())
    elif value is None:
        value = ''
    return str(value)


def entry_for_ref(data, fmt, ref):
    """The entry dict a list / token CSV row refers to (``_editor_entry_for_item``)."""
    if fmt in ['list', 'token_csv']:
        try:
            source_index = int(ref)
        except (TypeError, ValueError):
            return None
        if 0 <= source_index < len(data or []):
            entry = data[source_index]
            return entry if isinstance(entry, dict) else None
    return None


def row_matches_column_filters(row_text, column_fields, filters):
    filters = filters if isinstance(filters, dict) else {}
    fields = list(column_fields or [])
    field_columns = {field: index + 1 for index, field in enumerate(fields)}
    for field, allowed_values in filters.items():
        column = field_columns.get(field)
        if column is None or row_text(column) not in allowed_values:
            return False
    return True


def prune_column_filters(filters, column_fields):
    """Drop column filters for fields the loaded glossary no longer has."""
    valid_fields = set(column_fields)
    return {
        field: allowed
        for field, allowed in (filters if isinstance(filters, dict) else {}).items()
        if field in valid_fields
    }


def column_filter_values(texts):
    """A column's distinct values in filter order: blanks first, then case-insensitive."""
    values = sorted(
        {
            value
            for value in texts
        },
        key=lambda value: (value != '', value.casefold()),
    )
    return values


def loaded_stats_text(entries, fmt, configured_types=None):
    stats = [f"Total entries: {len(entries)}"]
    if fmt in ['list', 'token_csv'] and entries and isinstance(entries[0], dict) and 'type' in entries[0]:
        stats.append(glossary_type_count_summary(entries, configured_types))
    elif fmt == 'list':
        chars = sum(1 for e in entries if isinstance(e, dict) and ('original_name' in e or 'name' in e))
        locs = sum(1 for e in entries if isinstance(e, dict) and 'locations' in e and e['locations'])
        stats.append(f"Characters: {chars}, Locations: {locs}")
    return " | ".join(stats)


def baseline_translated(original_translated_map, fmt, ref, col_key):
    if col_key not in ['translated_name', 'translated']:
        return None
    if fmt in ['list', 'token_csv']:
        try:
            idx = int(ref)
        except Exception:
            return None
        return original_translated_map.get(idx, '')
    elif fmt == 'dict':
        key = ref
        return original_translated_map.get(key, '')
    return None


def saved_translated_baseline(data, fmt):
    """The translated-name baseline after a save (``save_edited_glossary``); None for other formats."""
    if fmt in ['list', 'token_csv']:
        return {
            idx: entry.get('translated_name', '') for idx, entry in enumerate(data)
        }
    elif fmt == 'dict':
        return dict((data or {}).get('entries', {}))
    return None


def collect_translated_changes(data, fmt, original_translated_map):
    """Return list of (old, new) translated-name changes since last baseline."""
    changes = []
    if fmt in ['list', 'token_csv']:
        for idx, entry in enumerate(data or []):
            old = original_translated_map.get(idx, entry.get('translated_name', ''))
            new = entry.get('translated_name', '')
            if old != new:
                changes.append((old, new))
    elif fmt == 'dict':
        entries = (data or {}).get('entries', {})
        for key, new in entries.items():
            old = original_translated_map.get(key, new)
            if old != new:
                changes.append((old, new))
    return changes


def tracker_entry_for_entry(entry, tracker_data):
    if not isinstance(entry, dict):
        return None
    return tracker_entry_for_raw(
        tracker_data,
        entry.get('raw_name', ''),
    )


def entry_gender_status(entry, tracker_data, settings):
    tracker_entry = tracker_entry_for_entry(entry, tracker_data)
    if not tracker_entry:
        return None
    threshold, bias = settings()
    return editor_gender_status(
        tracker_entry,
        entry.get('gender', '') if isinstance(entry, dict) else '',
        threshold,
        bias,
    )


def can_resolve_gender(status):
    """Whether a row's gender status offers "Resolve Gender…" (a tracked conflict)."""
    return isinstance(status, dict) and bool(status.get('conflict'))


def format_tracker_location(occurrence):
    if not isinstance(occurrence, dict):
        return "—"
    chapter = occurrence.get('chapter_num')
    chapter_file = str(occurrence.get('chapter_file', '') or '').strip()
    parts = []
    if chapter not in (None, ''):
        parts.append(f"chapter {chapter}")
    if chapter_file:
        parts.append(chapter_file)
    return " · ".join(parts) or "—"


def current_gender_decision(tracker_entry):
    """The stored decision for the resolution dialog: 'auto', 'male' or 'female'."""
    current_decision = normalize_gender(tracker_entry.get('decision', 'auto'))
    if current_decision not in {'auto', *BINARY_GENDERS}:
        current_decision = 'auto'
    return current_decision


def gender_settings_labels(threshold, bias):
    """``(threshold_pct, bias_label)`` for the resolution dialog's overview line."""
    threshold_pct = normalize_threshold(threshold) * 100
    bias_label = normalize_bias(bias).replace('_', ' ').title()
    if bias_label == 'None':
        bias_label = 'No Bias'
    return threshold_pct, bias_label


def gender_history_line(gender, summary, total):
    """One "Tracking history" line of the resolution dialog."""
    values = summary['genders'][gender]
    count = values['count']
    ratio = values['ratio'] * 100
    return (
        f"{gender.title()}: {count}/{total} ({ratio:.1f}%)  ·  "
        f"first {format_tracker_location(values['first'])}  ·  "
        f"last {format_tracker_location(values['last'])}"
    )


def gender_flip_line(change):
    """One "Latest flips" line of the resolution dialog."""
    source = normalize_gender(change.get('from', '')).title() or '?'
    target = normalize_gender(change.get('to', '')).title() or '?'
    return f"• {source} → {target} · {format_tracker_location(change)}"


def apply_gender_decision(doc, tracker_entry, data_entry, raw_name, selected):
    """Record a Resolve Gender decision (applied to the tracker on the next save)."""
    tracker_entry['decision'] = selected
    doc._pending_gender_decisions[raw_name] = selected
    data_entry['gender'] = display_gender(resolved_storage_gender(
        tracker_entry,
        data_entry.get('gender', ''),
    ))


def reload_flash_rows(old_rows, new_row_keys):
    """Rows to flash after an external reload: ``[(index, colour)]``.

    ``old_rows`` maps the previous row index to all its column texts; ``new_row_keys`` holds,
    per new row, its texts without the row-number column (None for a missing row). Yellow
    marks a modified row, green an added one, red the position of a deleted one.
    """
    old_content = {}
    for i, cols in old_rows.items():
        key = cols[1:]  # skip row # column
        old_content[key] = i
    new_content = {}
    new_count = len(new_row_keys)
    for i in range(new_count):
        key = new_row_keys[i]
        if key is not None:
            new_content[key] = i

    old_keys = set(old_content.keys())
    new_keys = set(new_content.keys())
    added_keys = new_keys - old_keys    # new or modified entries
    deleted_keys = old_keys - new_keys  # removed entries

    flash_indices = []  # (index, color)
    _used_positions = set()

    # Greedy pair: match each deleted key to nearest added key = modification (orange)
    remaining_added = {k: new_content[k] for k in added_keys}
    remaining_deleted = {k: old_content[k] for k in deleted_keys}
    matched_added = set()
    matched_deleted = set()
    for d_key in sorted(remaining_deleted, key=lambda k: remaining_deleted[k]):
        d_pos = remaining_deleted[d_key]
        best_a, best_dist = None, float('inf')
        for a_key in remaining_added:
            if a_key in matched_added:
                continue
            dist = abs(remaining_added[a_key] - d_pos)
            if dist < best_dist:
                best_dist = dist
                best_a = a_key
        if best_a is not None:
            matched_added.add(best_a)
            matched_deleted.add(d_key)
            idx = remaining_added[best_a]
            flash_indices.append((idx, "#eab308"))  # yellow = modified
            _used_positions.add(idx)

    # Green for truly new rows (unmatched adds)
    for key in added_keys - matched_added:
        idx = new_content[key]
        flash_indices.append((idx, "#15803d"))
        _used_positions.add(idx)

    # Red for truly deleted rows (unmatched deletes)
    for key in deleted_keys - matched_deleted:
        old_idx = old_content[key]
        nearest = min(old_idx, new_count - 1) if new_count > 0 else -1
        if nearest >= 0 and nearest not in _used_positions:
            flash_indices.append((nearest, "#dc2626"))
            _used_positions.add(nearest)

    # Fallback: duplicate rows (content keys same but count changed)
    if not flash_indices and new_count != len(old_rows):
        if new_count > len(old_rows):
            for idx in range(len(old_rows), new_count):
                flash_indices.append((idx, "#15803d"))
        else:
            flash_indices.append((max(0, new_count - 1), "#dc2626"))
    return flash_indices


# ---------------------------------------------------------------------------
# Save, Save As, Export selection, Convert format
# ---------------------------------------------------------------------------

def write_token_csv(entries, path_out, sections, custom_glossary_fields):
    """Write ``entries`` in the token-efficient format (``save_current_glossary.save_token_csv``).

    A non-empty ``sections`` list is extended in place with new section names, as the editor
    extends ``current_glossary_sections``.
    """
    sections = sections or []
    if not sections:
        sections = ['CHARACTERS', 'TERMS', 'TITLES', 'ORGANIZATIONS', 'LOCATIONS', 'ITEMS', 'ABILITYS']

    grouped = {sec: [] for sec in sections}
    default_map = {'character': 'CHARACTERS', 'terms': 'TERMS'}
    for entry in entries:
        sec = entry.get('_section')
        if not sec:
            sec = default_map.get(entry.get('type', 'terms'), 'TITLES')
        if sec not in grouped:
            grouped[sec] = []
            sections.append(sec)
        grouped[sec].append(entry)

    # Build header columns: standard + pattern-manager fields + custom/additional fields
    standard_cols = ['raw_name', 'translated_name', 'gender', 'description']
    pattern_fields = []
    try:
        import PatternManager as _pm
        pf = getattr(_pm, 'PATTERN_ADDITIONAL_FIELDS', [])
        if isinstance(pf, (list, tuple)):
            pattern_fields = list(pf)
    except Exception:
        pattern_fields = []

    custom_fields = custom_glossary_fields
    # include any fields present in data that are not internal/standard
    data_fields = []
    for e in entries:
        for k in e.keys():
            if k.startswith('_') or k in ['type'] + standard_cols:
                continue
            if k not in custom_fields and k not in pattern_fields and k not in data_fields:
                data_fields.append(k)
    header_cols = standard_cols + pattern_fields + custom_fields + data_fields

    lines = [f"Glossary Columns: {', '.join(header_cols)}", ""]
    for sec in sections:
        sec_entries = grouped.get(sec, [])
        if not sec_entries:
            continue
        lines.append(f"=== {sec} ===")
        for e in sec_entries:
            translated = e.get('translated_name', '')
            raw_name = e.get('raw_name', '')
            gender = e.get('gender', '')
            desc = e.get('description', '')

            line = f"* {raw_name} = {translated}" if raw_name else f"* {translated}"
            if gender:
                line += f" [{gender}]"
            extra_tail = []
            for col in header_cols:
                if col in ['translated_name', 'raw_name', 'gender', 'description']:
                    continue
                val = e.get(col, '')
                if val:
                    extra_tail.append(f"{col}: {val}")
            if desc:
                line += f": {desc}"
            if extra_tail:
                tail_str = " | ".join(extra_tail)
                line += f" | {tail_str}" if desc else f": {tail_str}"
            lines.append(line)
        lines.append("")

    with open(path_out, 'w', encoding='utf-8', newline='') as f:
        f.write("\n".join(lines).rstrip() + "\n")


def save_document(doc, path, config):
    """Write ``doc`` to ``path`` in its format (``save_current_glossary``).

    List / token CSV glossaries first apply pending gender decisions and fold tracked gender
    variants (extract_glossary_from_epub), then sync the tracker with the saved glossary.
    Returns False when there is nothing to save; raises on I/O errors (the editor shows them).
    """
    if not path or doc.current_glossary_data is None:
        return False
    if doc.current_glossary_format in ['list', 'token_csv']:
        from extract_glossary_from_epub import (
            _load_gender_tracker,
            resolve_glossary_gender_variants,
            set_gender_tracker_decisions,
            sync_gender_tracker_with_glossary,
        )
        pending_decisions = dict(getattr(doc, '_pending_gender_decisions', {}) or {})
        if pending_decisions:
            set_gender_tracker_decisions(path, pending_decisions)
        tracker_path = tracker_path_for_glossary(path)
        tracker = _load_gender_tracker(tracker_path) if os.path.exists(tracker_path) else None
        if tracker:
            doc.current_gender_tracker_data = tracker
            doc.current_gender_tracker_path = tracker_path
            doc.current_glossary_data, _collapsed = resolve_glossary_gender_variants(
                doc.current_glossary_data,
                output_path=path,
                tracker=tracker,
            )
    if path.endswith('.csv'):
        if getattr(doc, 'current_glossary_format', '') == 'token_csv':
            write_token_csv(
                doc.current_glossary_data,
                path,
                getattr(doc, 'current_glossary_sections', []),
                config.get('custom_glossary_fields', []),
            )
        else:
            import csv
            standard_fields = ['type', 'raw_name', 'translated_name', 'gender']
            extra_fields = []
            for entry in doc.current_glossary_data:
                for k in entry.keys():
                    if k.startswith('_') or k in standard_fields:
                        continue
                    if k not in extra_fields:
                        extra_fields.append(k)
            with open(path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(standard_fields + extra_fields)
                for entry in doc.current_glossary_data:
                    row = [
                        entry.get('type', ''),
                        entry.get('raw_name', ''),
                        entry.get('translated_name', ''),
                        entry.get('gender', '')
                    ]
                    for field in extra_fields:
                        row.append(entry.get(field, ''))
                    writer.writerow(row)
    else:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(doc.current_glossary_data, f, ensure_ascii=False, indent=2)
    if doc.current_glossary_format in ['list', 'token_csv']:
        sync_gender_tracker_with_glossary(doc.current_glossary_data, path)
    doc._pending_gender_decisions = {}
    doc._gender_variants_pending_save = 0
    return True


def save_glossary_as(path, data, fmt, config):
    """Save As: CSV (type, raw, translated, gender for list glossaries) or JSON."""
    if path.endswith('.csv'):
        # Save as CSV
        import csv
        with open(path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            if fmt == 'list':
                _custom_types = config.get('custom_entry_types', {
                    'character': {'enabled': True, 'has_gender': True},
                    'terms': {'enabled': True, 'has_gender': False},
                    'surnames': {'enabled': True, 'has_gender': False},
                    'titles': {'enabled': True, 'has_gender': True},
                    'locations': {'enabled': True, 'has_gender': False},
                    'nicknames': {'enabled': True, 'has_gender': True}
                })
                for entry in data:
                    entry_cfg = _custom_types.get(entry.get('type', ''), {})
                    if entry_cfg.get('has_gender', False):
                        writer.writerow([entry.get('type', ''), entry.get('raw_name', ''), 
                                       entry.get('translated_name', ''), entry.get('gender', '')])
                    else:
                        writer.writerow([entry.get('type', ''), entry.get('raw_name', ''), 
                                       entry.get('translated_name', ''), ''])
    else:
        # Save as JSON
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def export_selected_entries(path, data, fmt, refs, config):
    """Export the selected rows (``refs``) to CSV or JSON (Export Selection)."""
    if fmt in ['list', 'token_csv']:
        exported = []
        for ref in refs:
            try:
                idx = int(ref)
            except (TypeError, ValueError):
                continue
            if 0 <= idx < len(data):
                exported.append(data[idx])

        if path.endswith('.csv'):
            # Export as CSV
            import csv
            with open(path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                _custom_types = config.get('custom_entry_types', {
                    'character': {'enabled': True, 'has_gender': True},
                    'terms': {'enabled': True, 'has_gender': False},
                    'surnames': {'enabled': True, 'has_gender': False},
                    'titles': {'enabled': True, 'has_gender': True},
                    'locations': {'enabled': True, 'has_gender': False},
                    'nicknames': {'enabled': True, 'has_gender': True}
                })
                for entry in exported:
                    entry_cfg = _custom_types.get(entry.get('type', ''), {})
                    if entry_cfg.get('has_gender', False):
                        writer.writerow([entry.get('type', ''), entry.get('raw_name', ''), 
                                       entry.get('translated_name', ''), entry.get('gender', '')])
                    else:
                        writer.writerow([entry.get('type', ''), entry.get('raw_name', ''), 
                                       entry.get('translated_name', ''), ''])
        else:
            # Export as JSON
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(exported, f, ensure_ascii=False, indent=2)

    else:
        exported = {}
        for ref in refs:
            key = ref
            if key in data.get('entries', {}):
                value = data['entries'][key]
                exported[key] = value

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(exported, f, ensure_ascii=False, indent=2)


def default_convert_path(current_path, config):
    """The Convert Format dialog's suggested CSV path."""
    if current_path:
        default_csv_path = current_path.replace('.json', '.csv')
    else:
        override_dir = os.environ.get("OUTPUT_DIRECTORY") or config.get("output_directory", "")
        default_csv_path = os.path.join(os.path.abspath(override_dir) if override_dir else os.getcwd(), "glossary.csv")
    return default_csv_path


def convert_to_csv(doc, csv_path, config):
    """Convert Format: write the loaded glossary as token-efficient or legacy CSV.

    Old JSON shapes are normalised to typed entries first. Returns the format label
    ("token-efficient" / "legacy CSV"), or None when there are no entries.
    """
    # Check whether to use legacy CSV or token-efficient format
    use_legacy = config.get('glossary_use_legacy_csv', False)

    # Get custom types for gender info
    custom_types = config.get('custom_entry_types', {
        'character': {'enabled': True, 'has_gender': True},
        'terms': {'enabled': True, 'has_gender': False},
        'surnames': {'enabled': True, 'has_gender': False},
        'titles': {'enabled': True, 'has_gender': True},
        'locations': {'enabled': True, 'has_gender': False},
        'nicknames': {'enabled': True, 'has_gender': True}
    })

    # Get custom fields
    custom_fields = config.get('custom_glossary_fields', [])

    # ── Normalise entries to new-format dicts (type/raw_name/translated_name/gender) ──
    entries = []
    if isinstance(doc.current_glossary_data, list) and doc.current_glossary_data:
        if 'type' in doc.current_glossary_data[0]:
            # Already new format
            entries = list(doc.current_glossary_data)
        else:
            # Old format → convert
            for entry in doc.current_glossary_data:
                is_location = False
                if 'locations' in entry and entry['locations']:
                    is_location = True
                elif 'title' in entry and any(term in str(entry.get('title', '')).lower()
                                              for term in ['location', 'place', 'city', 'region']):
                    is_location = True
                entry_type = 'terms' if is_location else 'character'
                type_config = custom_types.get(entry_type, {})
                new_entry = {
                    'type': entry_type,
                    'raw_name': entry.get('original_name', entry.get('original', '')),
                    'translated_name': entry.get('name', entry.get('translated', '')),
                    'gender': entry.get('gender', 'Unknown') if type_config.get('has_gender', False) else '',
                }
                desc = entry.get('description', '')
                if desc:
                    new_entry['description'] = desc
                entries.append(new_entry)
    elif isinstance(doc.current_glossary_data, dict):
        # Dict format (key→value pairs)
        src = doc.current_glossary_data.get('entries', doc.current_glossary_data)
        for original, translated in src.items():
            entries.append({
                'type': 'terms',
                'raw_name': original,
                'translated_name': translated,
                'gender': '',
            })

    if not entries:
        return None

    if use_legacy:
        # ── Legacy CSV format ────────────────────────────────────────
        import csv
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            header = ['type', 'raw_name', 'translated_name', 'gender']
            if custom_fields:
                header.extend(custom_fields)
            writer.writerow(header)
            for entry in entries:
                entry_type = entry.get('type', 'terms')
                type_config = custom_types.get(entry_type, {})
                row = [
                    entry_type,
                    entry.get('raw_name', ''),
                    entry.get('translated_name', ''),
                    entry.get('gender', '') if type_config.get('has_gender', False) else '',
                ]
                for field in custom_fields:
                    row.append(entry.get(field, ''))
                writer.writerow(row)
    else:
        # ── Token-efficient format (sectioned, bullet-style) ─────────
        sections = getattr(doc, 'current_glossary_sections', []) or []
        if not sections:
            # Build sections from entry types
            seen = set()
            for e in entries:
                sec = e.get('_section')
                if not sec:
                    t = e.get('type', 'terms').upper()
                    sec = t + 'S' if not t.endswith('S') else t
                if sec not in seen:
                    sections.append(sec)
                    seen.add(sec)

        grouped = {sec: [] for sec in sections}
        default_map = {'character': 'CHARACTERS', 'terms': 'TERMS'}
        for entry in entries:
            sec = entry.get('_section')
            if not sec:
                t = entry.get('type', 'terms')
                sec = default_map.get(t)
                if not sec:
                    sec = t.upper() + ('S' if not t.upper().endswith('S') else '')
            if sec not in grouped:
                grouped[sec] = []
                sections.append(sec)
            grouped[sec].append(entry)

        # Build header columns
        standard_cols = ['raw_name', 'translated_name', 'gender', 'description']
        pattern_fields = []
        try:
            import PatternManager as _pm
            pf = getattr(_pm, 'PATTERN_ADDITIONAL_FIELDS', [])
            if isinstance(pf, (list, tuple)):
                pattern_fields = list(pf)
        except Exception:
            pattern_fields = []

        # Include any fields present in data that are not internal/standard
        data_fields = []
        for e in entries:
            for k in e.keys():
                if k.startswith('_') or k in ['type'] + standard_cols:
                    continue
                if k not in custom_fields and k not in pattern_fields and k not in data_fields:
                    data_fields.append(k)
        header_cols = standard_cols + pattern_fields + custom_fields + data_fields

        lines = [f"Glossary Columns: {', '.join(header_cols)}", ""]
        for sec in sections:
            sec_entries = grouped.get(sec, [])
            if not sec_entries:
                continue
            lines.append(f"=== {sec} ===")
            for e in sec_entries:
                translated = e.get('translated_name', '')
                raw_name = e.get('raw_name', '')
                gender = e.get('gender', '')
                desc = e.get('description', '')

                line = f"* {raw_name} = {translated}" if raw_name else f"* {translated}"
                if gender:
                    line += f" [{gender}]"
                extra_tail = []
                for col in header_cols:
                    if col in ['translated_name', 'raw_name', 'gender', 'description']:
                        continue
                    val = e.get(col, '')
                    if val:
                        extra_tail.append(f"{col}: {val}")
                if desc:
                    line += f": {desc}"
                if extra_tail:
                    tail_str = " | ".join(extra_tail)
                    line += f" | {tail_str}" if desc else f": {tail_str}"
                lines.append(line)
            lines.append("")

        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            f.write("\n".join(lines).rstrip() + "\n")

    return "legacy CSV" if use_legacy else "token-efficient"


# ---------------------------------------------------------------------------
# Editing operations
# ---------------------------------------------------------------------------

def normalize_edit_value(col_key, new_value):
    """The value a cell edit stores (gender cells are written in display form)."""
    if str(col_key).strip().lower() == 'gender':
        new_value = display_gender(new_value)
    return new_value


def apply_entry_edit(doc, ref, col_key, new_value):
    """Store a cell edit (``_on_tree_double_click.save_edit``).

    Returns ``(row_idx, new_ref, ref_changed)``: renaming a dict glossary's key changes the
    row's ref.
    """
    new_ref, ref_changed = ref, False
    try:
        row_idx = int(ref)
    except Exception:
        row_idx = -1

    if doc.current_glossary_format in ['list', 'token_csv']:
        if 0 <= row_idx < len(doc.current_glossary_data):
            data_entry = doc.current_glossary_data[row_idx]
            # Standard fields must be kept as empty strings, not removed,
            # to avoid losing columns (e.g. gender) when saved.
            _standard_fields = {'type', 'raw_name', 'translated_name', 'gender', 'description'}
            if new_value:
                data_entry[col_key] = new_value
            elif col_key in _standard_fields:
                data_entry[col_key] = ''
            else:
                data_entry.pop(col_key, None)

    elif doc.current_glossary_format == 'dict':
        key = ref
        entries = doc.current_glossary_data.get('entries', {})
        if key in entries:
            if col_key == 'original':
                value = entries.pop(key)
                new_key = new_value or key
                entries[new_key] = value
                new_ref, ref_changed = new_key, True
            elif col_key == 'translated':
                entries[key] = new_value
    return row_idx, new_ref, ref_changed


def delete_entries(doc, refs):
    """Delete the rows ``refs``; returns ``(indices_to_delete, keys_to_delete)``."""
    indices_to_delete = []
    keys_to_delete = []
    for ref in refs:
       if doc.current_glossary_format in ['list', 'token_csv']:
           try:
               indices_to_delete.append(int(ref))
           except (TypeError, ValueError):
               pass
       elif doc.current_glossary_format == 'dict':
           key = ref
           if key is not None:
               keys_to_delete.append(key)

    indices_to_delete.sort(reverse=True)

    if doc.current_glossary_format in ['list', 'token_csv']:
       for idx in indices_to_delete:
           if 0 <= idx < len(doc.current_glossary_data):
               del doc.current_glossary_data[idx]

    elif doc.current_glossary_format == 'dict':
       for key in keys_to_delete:
           doc.current_glossary_data.get('entries', {}).pop(key, None)
    return indices_to_delete, keys_to_delete


def count_empty_fields(data):
    """``(empty_fields_found, {field: count})`` for Clean Empty Fields."""
    empty_fields_found = False
    fields_cleaned = {}

    # Count empty fields first
    for entry in data:
        for field in list(entry.keys()):
            value = entry[field]
            if value is None or value == "" or (isinstance(value, list) and len(value) == 0) or (isinstance(value, dict) and len(value) == 0):
                empty_fields_found = True
                fields_cleaned[field] = fields_cleaned.get(field, 0) + 1
    return empty_fields_found, fields_cleaned


def remove_empty_fields(data):
    """Drop empty values from every entry; returns how many were removed."""
    total_cleaned = 0
    for entry in data:
        for field in list(entry.keys()):
            value = entry[field]
            if value is None or value == "" or (isinstance(value, list) and len(value) == 0) or (isinstance(value, dict) and len(value) == 0):
                entry.pop(field)
                total_cleaned += 1
    return total_cleaned


def clean_empty_fields_message(total_cleaned, fields_cleaned):
    msg = f"Cleaned {total_cleaned} empty fields\n\n"
    msg += "Fields cleaned:\n"
    for field, count in sorted(fields_cleaned.items(), key=lambda x: x[1], reverse=True):
        msg += f"• {field}: {count} entries\n"
    return msg


def remove_duplicate_entries(data, glossary_path, gender_tracker, disable_honorifics_filter):
    """Remove Duplicates with the Balanced/Full extractor's dedup engine (skip_duplicate_entries).

    Sets GLOSSARY_DISABLE_HONORIFICS_FILTER like the desktop editor. Raises ImportError when
    the extractor is unavailable (callers fall back to
    :func:`remove_duplicate_entries_fallback`).
    """
    from extract_glossary_from_epub import skip_duplicate_entries, remove_honorifics

    # Set environment variable for honorifics toggle
    os.environ['GLOSSARY_DISABLE_HONORIFICS_FILTER'] = '1' if disable_honorifics_filter else '0'

    return skip_duplicate_entries(
        data,
        glossary_path=glossary_path,
        gender_tracker=gender_tracker,
    )


def remove_duplicate_entries_fallback(data):
    """First entry per raw name (case-insensitive); returns ``(unique_entries, duplicates)``."""
    seen_raw_names = set()
    unique_entries = []
    duplicates = 0

    for entry in data:
        raw_name = entry.get('raw_name', '').lower().strip()
        if raw_name and raw_name not in seen_raw_names:
            seen_raw_names.add(raw_name)
            unique_entries.append(entry)
        elif raw_name:
            duplicates += 1
    return unique_entries, duplicates


DUPLICATE_DETECTION_INFO_TITLE = "Duplicate Detection"
DUPLICATE_DETECTION_INFO_TEXT = (
    "Duplicate detection is based on the raw_name field.\n\n"
    "• Entries with identical raw_name values are considered duplicates\n"
    "• The first occurrence is kept, later ones are removed\n"
    "• Honorifics filtering can be toggled in the Manual Glossary tab\n\n"
    "When honorifics filtering is enabled, names are compared after removing honorifics."
)


def trim_preview_text(entry_count, top_n):
    entries_to_remove = max(0, entry_count - top_n)

    preview_text = f"Preview of changes:\n"
    preview_text += f"• Entries: {entry_count} → {top_n} ({entries_to_remove} removed)\n"
    return preview_text


def trim_entries(doc, top_n):
    """Keep the first ``top_n`` entries (Trim Entries)."""
    if doc.current_glossary_format in ['list', 'token_csv']:
        # Keep only top N entries
        if top_n < len(doc.current_glossary_data):
            doc.current_glossary_data = doc.current_glossary_data[:top_n]

    elif doc.current_glossary_format == 'dict':
        # For dict format, only support entry limit
        entries = list(doc.current_glossary_data['entries'].items())
        if top_n < len(entries):
            doc.current_glossary_data['entries'] = dict(entries[:top_n])


def filter_entry_types(config, data):
    """The entry types the Filter Entries dialog offers (enabled types, then types in the data)."""
    _custom_types_cfg = config.get('custom_entry_types', {
        'character': {'enabled': True, 'has_gender': True},
        'terms': {'enabled': True, 'has_gender': False},
        'surnames': {'enabled': True, 'has_gender': False},
        'titles': {'enabled': True, 'has_gender': True},
        'locations': {'enabled': True, 'has_gender': False},
        'nicknames': {'enabled': True, 'has_gender': True}
    })
    filter_types = [t for t, cfg in _custom_types_cfg.items() if cfg.get('enabled', True)]

    # Also include any types actually present in the glossary data
    for _entry in data:
        _t = _entry.get('type')
        if _t and _t not in filter_types:
            filter_types.append(_t)
    return filter_types


def parse_type_limit(text):
    """A "First N" limit field: None means no limit."""
    if text is None:
        return None
    text = text.strip()
    if not text:
        return None
    try:
        return max(0, int(text))
    except ValueError:
        return None


def entry_matches_filter(entry, type_counts=None, *, is_new_format, kept_types, search_text,
                         gender_value, config, type_limits):
    # Filter Entries. ``kept_types`` maps a type to its "Keep" checkbox state, ``type_limits``
    # a type to its "First N" text; ``type_counts`` (updated in place) applies those limits.
    """Check if an entry matches the filter conditions"""
    # Type filter
    if is_new_format and entry.get('type'):
        type_check = kept_types.get(entry['type'])
        if type_check is not None and not type_check:
            return False

    # Text filter
    search_text = search_text.strip().lower()
    if search_text:
        # Search in all text fields
        entry_text = ' '.join(str(v) for v in entry.values() if isinstance(v, str)).lower()
        if search_text not in entry_text:
            return False

    # Gender filter
    if is_new_format and gender_value != "all":
        # Apply gender filter to any type with has_gender
        _custom_types = config.get('custom_entry_types', {
            'character': {'enabled': True, 'has_gender': True},
            'terms': {'enabled': True, 'has_gender': False},
            'surnames': {'enabled': True, 'has_gender': False},
            'titles': {'enabled': True, 'has_gender': True},
            'locations': {'enabled': True, 'has_gender': False},
            'nicknames': {'enabled': True, 'has_gender': True}
        })
        entry_cfg = _custom_types.get(entry.get('type', ''), {})
        if entry_cfg.get('has_gender', False) and entry.get('gender') != gender_value:
            return False

    # Per-type "first N" limit (applied only to entries that passed all other filters)
    if is_new_format and type_counts is not None and entry.get('type'):
        _t = entry['type']
        limit = parse_type_limit(type_limits.get(_t))
        if limit is not None and type_counts.get(_t, 0) >= limit:
            return False
        type_counts[_t] = type_counts.get(_t, 0) + 1

    return True


def count_filter_matches(data, fmt, matches):
    """Preview Filter: how many entries ``matches(entry, type_counts)`` keeps."""
    matching = 0
    type_counts = {}

    if fmt in ['list', 'token_csv']:
        for entry in data:
            if matches(entry, type_counts):
                matching += 1
    else:
        for key, entry in data.get('entries', {}).items():
            if matches(entry, type_counts):
                matching += 1
    return matching


def filter_list_entries(data, matches):
    """Apply Filter: the entries ``matches(entry, type_counts)`` keeps, in order."""
    filtered = []
    type_counts = {}
    for entry in data:
        if matches(entry, type_counts):
            filtered.append(entry)
    return filtered


# ---------------------------------------------------------------------------
# Find / Replace
# ---------------------------------------------------------------------------

class EditorRow:
    """A row of the editor view for GUI-free callers (the protocol Find/Replace works on).

    ``texts[0]`` is the row number and ``texts[i]`` the display value of
    ``glossary_column_fields[i - 1]`` (:func:`editor_display_value`), as in the desktop tree;
    ``gender_status`` is the row's tracker status (:func:`entry_gender_status`) or None.
    """

    __slots__ = ('texts', 'source_ref', 'gender_status')

    def __init__(self, texts, source_ref, gender_status=None):
        self.texts = list(texts)
        self.source_ref = source_ref
        self.gender_status = gender_status

    @classmethod
    def from_spec(cls, display_idx, source_ref, entry, column_fields):
        return cls(
            [str(display_idx)] + [editor_display_value(entry, field) for field in column_fields],
            source_ref,
        )

    def columnCount(self):
        return len(self.texts)

    def text(self, column):
        return self.texts[column] if 0 <= column < len(self.texts) else ''

    def setText(self, column, value):
        self.texts[column] = value

    def ref(self):
        return self.source_ref

    def set_ref(self, value):
        self.source_ref = value


def editor_rows(column_fields, data, fmt, visible_source_indices=None, gender_status=None):
    """The editor view's rows (numbered from 1), optionally limited to some source indices.

    With ``gender_status(entry)``, a row with a tracker status shows the tracker's label in its
    gender column, as the desktop tree does (``_apply_editor_gender_presentation``).
    """
    fields, specs = editor_row_specs(column_fields, data, fmt)
    visible_set = None if visible_source_indices is None else set(visible_source_indices)
    gender_column = fields.index('gender') + 1 if 'gender' in fields else None
    rows = []
    for source_idx, source_ref, entry in specs:
        if visible_set is not None and source_idx not in visible_set:
            continue
        row = EditorRow.from_spec(len(rows) + 1, source_ref, entry, fields)
        if gender_status is not None:
            status = gender_status(entry)
            row.gender_status = status
            if isinstance(status, dict) and gender_column is not None:
                row.setText(gender_column, status.get("label", row.text(gender_column)))
        rows.append(row)
    return rows


def find_next_index(text, total, row_texts, last_find_pos=-1):
    """Find Next: the next row (after ``last_find_pos``, wrapping) whose texts contain ``text``.

    ``row_texts(index)`` returns all column texts of a row, the row number included.
    """
    text_lower = text.lower()
    start = (last_find_pos + 1) % total
    for offset in range(total):
        idx = (start + offset) % total
        cols = row_texts(idx)
        if any(text_lower in c.lower() for c in cols):
            return idx
    return None


def row_has_match(row, find_text):
    """Whether a row's columns (not the row number) contain ``find_text``, ignoring case."""
    return bool(find_text) and row is not None and any(
        re.search(re.escape(find_text), row.text(c), re.IGNORECASE)
        for c in range(1, row.columnCount())
    )


def replace_in_row(doc, row, column_fields, text, repl, on_replaced=None):
    """Replace ``text`` (case-insensitive, literal) in one row and keep ``doc`` in sync.

    ``on_replaced(col_key, after)`` runs after each changed column (the desktop re-highlights
    the row there). Returns the number of replacements.
    """
    pattern = re.compile(re.escape(text), re.IGNORECASE)
    replacements = 0

    for col_idx in range(1, row.columnCount()):
        before = row.text(col_idx)
        after, count = pattern.subn(repl, before)
        if count == 0:
            continue

        row.setText(col_idx, after)
        col_key = column_fields[col_idx - 1] if column_fields else None

        if doc.current_glossary_format in ['list', 'token_csv']:
            try:
                row_idx = int(row.ref())
            except Exception:
                row_idx = -1
            if 0 <= row_idx < len(doc.current_glossary_data) and col_key:
                entry = doc.current_glossary_data[row_idx]
                if col_key == '_section':
                    entry['_section'] = after
                elif after:
                    entry[col_key] = after
                elif col_key in {'type', 'raw_name', 'translated_name', 'gender', 'description'}:
                    entry[col_key] = ''
                else:
                    entry.pop(col_key, None)

        elif doc.current_glossary_format == 'dict':
            key = row.ref()
            entries = doc.current_glossary_data.get('entries', {})
            if col_key == 'original':
                value = entries.pop(key, None)
                new_key = after if after else key
                entries[new_key] = value
                row.set_ref(new_key)
            elif col_key == 'translated' and key in entries:
                entries[key] = after

        replacements += count
        if on_replaced is not None:
            on_replaced(col_key, after)

    return replacements


NO_GLOSSARY_MATCH_TITLE = "No glossary match"
NO_GLOSSARY_MATCH_TEXT = "No entry found in the glossary. Update output files directly?"


def update_output_files_prompt(changes):
    """``(title, text, examples)`` of the "Update output files" confirmation on save."""
    example_lines = "<br>".join(f"{old or '&lt;empty&gt;'} -> {new or '&lt;empty&gt;'}" for old, new in changes[:5])
    return (
        "Update output files",
        f"{len(changes)} entries have had their translated name field updated.\nNow matching output files will be updated to reflect the change.",
        example_lines,
    )


# ---------------------------------------------------------------------------
# Output files: Update output files on save, Find/Replace fallback, Hide unused entries
# ---------------------------------------------------------------------------

def configured_update_workers(enable_parallel, raw_workers):
    """Threads for updating output files: the extraction worker count when parallel is on."""
    enabled = bool(enable_parallel)
    if not enabled:
        return 1
    try:
        workers = int(raw_workers)
    except (TypeError, ValueError):
        workers = 1
    return max(1, workers)


def update_output_files(glossary_path, changes, log=_noop_log, enable_parallel=False, raw_workers=1):
    """Replace old translated names with new ones across output files."""
    if not changes:
        return 0, 0

    effective_changes = [(old, new) for old, new in changes if old and new and old != new]
    if not effective_changes:
        return 0, 0

    if not glossary_path or not os.path.exists(glossary_path):
        log("Cannot update output files: no glossary file loaded.")
        return 0, 0

    glossary_dir = os.path.dirname(glossary_path)
    glossary_fname = os.path.splitext(os.path.basename(glossary_path))[0]
    parent_of_glossary_dir = os.path.dirname(glossary_dir)
    is_shared_glossary_folder = os.path.basename(glossary_dir).lower() == 'glossary'
    is_book_glossary_subfolder = os.path.basename(parent_of_glossary_dir).lower() == 'glossary'

    book_output_dir = None
    if is_shared_glossary_folder:
        book_name = None
        for suffix in ('_glossary', '_Glossary'):
            if glossary_fname.endswith(suffix):
                book_name = glossary_fname[:-len(suffix)]
                break
        if not book_name:
            book_name = glossary_fname
        if book_name:
            candidate = os.path.join(parent_of_glossary_dir, book_name)
            if os.path.isdir(candidate):
                book_output_dir = candidate
        if not book_output_dir:
            book_output_dir = parent_of_glossary_dir
    elif is_book_glossary_subfolder:
        book_name = os.path.basename(glossary_dir)
        grandparent = os.path.dirname(parent_of_glossary_dir)
        candidate = os.path.join(grandparent, book_name)
        book_output_dir = candidate if os.path.isdir(candidate) else grandparent
    else:
        book_output_dir = glossary_dir

    if not os.path.isdir(book_output_dir):
        log(f"Cannot update output files: directory not found: {book_output_dir}")
        return 0, 0

    log(f"Scanning output files in: {book_output_dir}")

    excluded_extensions = {'.csv', '.json'}
    excluded_names = {'metadata.json', 'metadata.opf', 'metadata.xml', 'content.opf', 'toc.ncx'}
    candidate_paths = []
    for name in sorted(os.listdir(book_output_dir)):
        path = os.path.join(book_output_dir, name)
        if not os.path.isfile(path):
            continue
        lower_name = name.lower()
        if any(lower_name.endswith(ext) for ext in excluded_extensions):
            continue
        if lower_name in excluded_names or lower_name.endswith('.opf'):
            continue
        candidate_paths.append(path)

    if not candidate_paths:
        return 0, 0

    def _update_one_file(path):
        logs = []
        try:
            with open(path, 'rb') as f:
                raw_bytes = f.read()
            try:
                content = raw_bytes.decode('utf-8')
            except UnicodeDecodeError:
                content = raw_bytes.decode('utf-8', errors='replace')

            new_content = content
            per_pair_counts = []
            for old, new in effective_changes:
                before_count = new_content.count(old)
                if before_count == 0:
                    continue
                new_content = new_content.replace(old, new)
                per_pair_counts.append((old, new, before_count))

            if new_content == content:
                return {"path": path, "files_updated": 0, "replacements": 0, "logs": logs}

            try:
                stat_before = os.stat(path)
                mtime_before = stat_before.st_mtime
                size_before = stat_before.st_size
            except Exception:
                mtime_before, size_before = None, None

            new_bytes = new_content.encode('utf-8')
            with open(path, 'wb') as f:
                f.write(new_bytes)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except Exception:
                    pass

            try:
                with open(path, 'rb') as f:
                    verify_bytes = f.read()
                stat_after = os.stat(path)
                mtime_after = stat_after.st_mtime
                size_after = stat_after.st_size
            except Exception as ve:
                logs.append(f"Could not verify write of {path}: {ve}")
                verify_bytes = None
                mtime_after, size_after = None, None

            replaced_here = sum(c for _o, _n, c in per_pair_counts)
            write_landed = verify_bytes == new_bytes
            if write_landed:
                logs.append(
                    f"Updated file: {path} ({replaced_here} replacements, "
                    f"{size_before} -> {size_after} bytes)"
                )
                return {"path": path, "files_updated": 1, "replacements": replaced_here, "logs": logs}

            resolved = os.path.realpath(path)
            logs.append(
                f"Write did NOT persist for {path}"
                + (f" (realpath: {resolved})" if resolved != path else "")
            )
            logs.append(
                f"   expected {len(new_bytes)} bytes, found {size_after} bytes"
                + (f" (mtime {mtime_before} -> {mtime_after})" if mtime_before is not None else "")
            )
            if verify_bytes is not None:
                try:
                    verify_text = verify_bytes.decode('utf-8', errors='replace')
                    for old, new, _cnt in per_pair_counts:
                        still = verify_text.count(old)
                        if still:
                            logs.append(f"   - '{old}' still present {still} time(s) (expected 0)")
                except Exception:
                    pass
            logs.append(
                "   Likely causes: file locked by another app, OneDrive/antivirus revert, "
                "or read-only attribute. Try closing viewers and retry."
            )
            return {"path": path, "files_updated": 0, "replacements": replaced_here, "logs": logs}
        except Exception as e:
            return {
                "path": path,
                "files_updated": 0,
                "replacements": 0,
                "logs": [f"Failed to update {path}: {e}"],
            }

    workers = configured_update_workers(enable_parallel, raw_workers)
    if workers > 1 and len(candidate_paths) > 1:
        log(f"Updating {len(candidate_paths)} output files with {workers} worker threads")
        from concurrent.futures import ThreadPoolExecutor, as_completed
        results = []
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="GlossaryFileUpdate") as executor:
            future_to_path = {executor.submit(_update_one_file, path): path for path in candidate_paths}
            for future in as_completed(future_to_path):
                try:
                    results.append(future.result())
                except Exception as e:
                    path = future_to_path.get(future, "<unknown>")
                    results.append({
                        "path": path,
                        "files_updated": 0,
                        "replacements": 0,
                        "logs": [f"Failed to update {path}: {e}"],
                    })
        results.sort(key=lambda item: item.get("path", ""))
    else:
        results = [_update_one_file(path) for path in candidate_paths]

    files_updated = 0
    total_replacements = 0
    for result in results:
        files_updated += int(result.get("files_updated", 0) or 0)
        total_replacements += int(result.get("replacements", 0) or 0)
        for line in result.get("logs", []):
            log(line)
    return files_updated, total_replacements


def editor_associated_source_path(glossary_path, source_map=None, input_sources=lambda: [],
                                  epub_getter=None, file_path=None, config=None):
    """The input file a glossary belongs to (mapped source, single input, current EPUB)."""
    try:
        mapped = source_map or {}
        if glossary_path in mapped and os.path.exists(mapped[glossary_path]):
            return mapped[glossary_path]
    except Exception:
        pass
    try:
        source_paths = [
            path
            for path in input_sources()
            if os.path.exists(path)
        ]
        if len(source_paths) == 1:
            return source_paths[0]
    except Exception:
        pass
    for getter_name in ('get_current_epub_path',):
        try:
            getter = epub_getter
            if callable(getter):
                path = getter()
                if path and str(path).lower().endswith('.epub') and os.path.exists(path):
                    return path
        except Exception:
            pass
    for attr_name in ('file_path',):
        try:
            path = file_path
            if path and str(path).lower().endswith('.epub') and os.path.exists(path):
                return path
        except Exception:
            pass
    try:
        path = config.get('last_epub_path') if config is not None else None
        if path and str(path).lower().endswith('.epub') and os.path.exists(path):
            return path
    except Exception:
        pass
    return None


def editor_output_dir_from_source(source_path, config=None, resolver=None, base_dir_getter=None):
    """The translated output folder of an input file, if it exists."""
    if not source_path:
        return None
    try:
        if callable(resolver):
            resolved = resolver(source_path)
            if resolved and os.path.isdir(resolved):
                return resolved
    except Exception:
        pass
    file_base = os.path.splitext(os.path.basename(source_path))[0]
    candidates = []
    override_dir = None
    try:
        for value in (
            os.environ.get('OUTPUT_DIRECTORY'),
            os.environ.get('OUTPUT_DIR'),
            config.get('output_directory') if config is not None else None,
        ):
            value = str(value or '').strip().strip('"')
            if value:
                override_dir = value
                break
    except Exception:
        override_dir = None
    if override_dir:
        candidates.append(os.path.join(os.path.abspath(override_dir), file_base))
    try:
        if callable(base_dir_getter):
            candidates.append(os.path.join(base_dir_getter(source_path), file_base))
    except Exception:
        pass
    candidates.append(os.path.join(os.path.dirname(os.path.abspath(source_path)), file_base))
    candidates.append(os.path.abspath(file_base))
    for candidate in candidates:
        if candidate and os.path.isdir(candidate):
            return candidate
    return None


def editor_output_dir_from_glossary(glossary_path, source_map=None, from_source=editor_output_dir_from_source):
    """The translated output folder a glossary file belongs to, derived from its location."""
    if not glossary_path or not os.path.exists(glossary_path):
        return None
    try:
        mapped = source_map or {}
        source_path = mapped.get(glossary_path)
        if source_path:
            resolved = from_source(source_path)
            if resolved:
                return resolved
    except Exception:
        pass
    glossary_dir = os.path.dirname(os.path.abspath(glossary_path))
    glossary_fname = os.path.splitext(os.path.basename(glossary_path))[0]
    parent_of_glossary_dir = os.path.dirname(glossary_dir)
    is_shared_glossary_folder = os.path.basename(glossary_dir).lower() == 'glossary'
    is_book_glossary_subfolder = os.path.basename(parent_of_glossary_dir).lower() == 'glossary'

    if is_shared_glossary_folder:
        book_name = None
        for suffix in ('_glossary', '_Glossary'):
            if glossary_fname.endswith(suffix):
                book_name = glossary_fname[:-len(suffix)]
                break
        if not book_name:
            book_name = glossary_fname
        if book_name:
            candidate = os.path.join(parent_of_glossary_dir, book_name)
            if os.path.isdir(candidate):
                return candidate
        return parent_of_glossary_dir if os.path.isdir(parent_of_glossary_dir) else None

    if is_book_glossary_subfolder:
        book_name = os.path.basename(glossary_dir)
        grandparent = os.path.dirname(parent_of_glossary_dir)
        candidate = os.path.join(grandparent, book_name)
        return candidate if os.path.isdir(candidate) else grandparent

    return glossary_dir if os.path.isdir(glossary_dir) else None


def translated_output_files(output_dir):
    if not output_dir or not os.path.isdir(output_dir):
        return []
    readable_exts = {'.html', '.htm', '.xhtml', '.xml', '.txt', '.md'}
    excluded_names = {
        'metadata.json',
        'metadata.opf',
        'metadata.xml',
        'content.opf',
        'toc.ncx',
        'glossary.csv',
        'glossary.json',
        'translation_progress.json',
    }
    paths = []
    for name in sorted(os.listdir(output_dir)):
        path = os.path.join(output_dir, name)
        if not os.path.isfile(path):
            continue
        lower_name = name.lower()
        if lower_name in excluded_names:
            continue
        if os.path.splitext(lower_name)[1] not in readable_exts:
            continue
        paths.append(path)
    return paths


def read_translated_output_texts(paths, progress_callback=None):
    texts = []
    errors = []
    total_files = len(paths or [])
    last_progress_emit = 0.0
    for file_idx, path in enumerate(paths, start=1):
        try:
            with open(path, 'rb') as f:
                raw_bytes = f.read()
            try:
                content = raw_bytes.decode('utf-8')
            except UnicodeDecodeError:
                content = raw_bytes.decode('utf-8', errors='replace')
            if os.path.splitext(path)[1].lower() in {'.html', '.htm', '.xhtml', '.xml'}:
                content = html_to_text(content)
            texts.append(content)
        except Exception as exc:
            errors.append(f"Failed to read translated output file for hide-unused: {path}: {exc}")
        if callable(progress_callback):
            now = time.monotonic()
            if file_idx == total_files or now - last_progress_emit >= 0.5:
                progress_callback(file_idx, total_files, os.path.basename(path))
                last_progress_emit = now
    return texts, errors


def editor_usage_entries(fields, specs):
    """Rows as Hide-unused matches them: display values plus raw / translated fallbacks."""
    entries = []
    for source_idx, _source_ref, source_entry in specs:
        entry = {'source_index': source_idx}
        for field in fields:
            value = source_entry.get(field, '') if isinstance(source_entry, dict) else ''
            if isinstance(value, list):
                value = ', '.join(str(v) for v in value)
            elif isinstance(value, dict):
                value = ', '.join(f"{k}: {v}" for k, v in value.items())
            elif value is None:
                value = ''
            entry[field] = str(value)
        if not entry.get('raw_name'):
            for fallback in ('original_name', 'original'):
                if entry.get(fallback):
                    entry['raw_name'] = entry.get(fallback)
                    break
        if not entry.get('translated_name'):
            for fallback in ('name', 'translated'):
                if entry.get(fallback):
                    entry['translated_name'] = entry.get(fallback)
                    break
        entries.append(entry)
    return entries


def compute_used_rows(entries, output_dir, total, token=None, emit_progress=_noop_log):
    """Hide unused entries: which rows' terms occur in the translated output files.

    ``emit_progress(payload)`` receives 'reading' / 'starting' / 'matching' stages. Returns the
    result payload (``used_rows`` are source indices; ``no_files`` when the folder holds no
    readable output).
    """
    try:
        output_files = translated_output_files(output_dir)
        if not output_files:
            result = {
                'ok': True,
                'token': token,
                'output_dir': output_dir,
                'no_files': True,
                'total': total,
            }
        else:
            output_texts, errors = read_translated_output_texts(
                output_files,
                progress_callback=lambda current, total_files, _name: emit_progress(
                    {
                        'stage': 'reading',
                        'current': current,
                        'total_files': total_files,
                    }
                ),
            )
            emit_progress({'stage': 'starting', 'total': total, 'total_files': len(output_texts)})
            output_index = build_prepared_output_index(output_texts)
            used_rows = []

            def _source_idx_for_entry(entry, fallback_idx):
                source_idx = entry.get('source_index', fallback_idx)
                try:
                    return int(source_idx)
                except (TypeError, ValueError):
                    return fallback_idx

            if entries and output_index['outputs']:
                # Token-set matching is ~microseconds per entry, so a plain
                # loop in this worker thread keeps the UI responsive without
                # the overhead (and GIL ping-pong) of a thread pool.
                checked = 0
                last_emit = 0.0
                for fallback_idx, entry in enumerate(entries):
                    source_idx = _source_idx_for_entry(entry, fallback_idx)
                    if entry_matches_output_index(entry, output_index):
                        used_rows.append(source_idx)
                    checked += 1
                    now = time.monotonic()
                    if checked == total or now - last_emit >= 0.5:
                        emit_progress(
                            {
                                'stage': 'matching',
                                'checked': checked,
                                'total': total,
                                'used': len(used_rows),
                            }
                        )
                        last_emit = now
            else:
                emit_progress(
                    {
                        'stage': 'matching',
                        'checked': total,
                        'total': total,
                        'used': 0,
                    }
                )
            result = {
                'ok': True,
                'token': token,
                'output_dir': output_dir,
                'used_rows': sorted(set(used_rows)),
                'errors': errors,
                'total': total,
            }
    except Exception as exc:
        result = {
            'ok': False,
            'token': token,
            'output_dir': output_dir,
            'error': str(exc),
            'total': total,
        }
    return result


# ---------------------------------------------------------------------------
# Undo / Redo (glossary snapshots and output-file replacements)
# ---------------------------------------------------------------------------

def trim_undo_stack(undo_stack, undo_max=EDITOR_UNDO_MAX):
    if len(undo_stack) > undo_max:
        undo_stack.pop(0)


def push_undo_snapshot(doc, undo_stack, redo_stack, undo_max=EDITOR_UNDO_MAX):
    """Snapshot ``doc`` before a change; False (nothing pushed) when no glossary is loaded."""
    if doc.current_glossary_data is None:
        return False
    snap = {
        'kind': 'glossary',
        'data': _copy.deepcopy(doc.current_glossary_data),
        'gender_tracker': _copy.deepcopy(getattr(doc, 'current_gender_tracker_data', None)),
        'pending_gender_decisions': _copy.deepcopy(getattr(doc, '_pending_gender_decisions', {})),
    }
    undo_stack.append(snap)
    trim_undo_stack(undo_stack, undo_max)
    redo_stack.clear()
    return True


def push_html_undo_snapshot(undo_stack, redo_stack, undo_max, changes):
    """Record a direct output-file Find/Replace (``[(old, new)]``) so it can be undone."""
    if not changes:
        return False
    undo_stack.append({'kind': 'html', 'changes': list(changes)})
    trim_undo_stack(undo_stack, undo_max)
    redo_stack.clear()
    return True


def undo_entry_kind(entry):
    return entry.get('kind') if isinstance(entry, dict) and 'kind' in entry else 'glossary'


def undo_step(doc, undo_stack, redo_stack):
    """Pop the last undo entry.

    Returns ``('html', reverse_changes)`` for an output-file replacement (re-apply the pairs to
    the output files), ``('glossary', None)`` after restoring the snapshot into ``doc`` (save
    and reload it), ``('blocked', None)`` when no glossary is loaded and ``(None, None)`` when
    there is nothing to undo.
    """
    if not undo_stack:
        return None, None
    entry = undo_stack.pop()
    if undo_entry_kind(entry) == 'html':
        # Reverse the output-file replacements (new -> old)
        redo_stack.append(entry)
        reverse = [(new, old) for (old, new) in entry.get('changes', [])]
        return 'html', reverse
    # Glossary snapshot
    if doc.current_glossary_data is None:
        # Nothing to restore into; put it back and bail
        undo_stack.append(entry)
        return 'blocked', None
    redo_stack.append({
        'kind': 'glossary',
        'data': _copy.deepcopy(doc.current_glossary_data),
        'gender_tracker': _copy.deepcopy(getattr(doc, 'current_gender_tracker_data', None)),
        'pending_gender_decisions': _copy.deepcopy(getattr(doc, '_pending_gender_decisions', {})),
    })
    doc.current_glossary_data = entry.get('data') if isinstance(entry, dict) else entry
    if isinstance(entry, dict):
        doc.current_gender_tracker_data = entry.get('gender_tracker')
        restored_pending = dict(entry.get('pending_gender_decisions', {}) or {})
        tracker_entries = (doc.current_gender_tracker_data or {}).get('entries', {})
        for tracker_item in tracker_entries.values() if isinstance(tracker_entries, dict) else []:
            if isinstance(tracker_item, dict) and tracker_item.get('raw_name'):
                restored_pending.setdefault(
                    tracker_item['raw_name'],
                    tracker_item.get('decision', 'auto'),
                )
        doc._pending_gender_decisions = restored_pending
    return 'glossary', None


def redo_step(doc, undo_stack, redo_stack):
    """Pop the last redo entry: ``('html', entry)`` (re-apply ``entry['changes']``), otherwise
    as :func:`undo_step`."""
    if not redo_stack:
        return None, None
    entry = redo_stack.pop()
    if undo_entry_kind(entry) == 'html':
        # Re-apply the output-file replacements (old -> new)
        undo_stack.append(entry)
        return 'html', entry
    if doc.current_glossary_data is None:
        redo_stack.append(entry)
        return 'blocked', None
    undo_stack.append({
        'kind': 'glossary',
        'data': _copy.deepcopy(doc.current_glossary_data),
        'gender_tracker': _copy.deepcopy(getattr(doc, 'current_gender_tracker_data', None)),
        'pending_gender_decisions': _copy.deepcopy(getattr(doc, '_pending_gender_decisions', {})),
    })
    doc.current_glossary_data = entry.get('data') if isinstance(entry, dict) else entry
    if isinstance(entry, dict):
        doc.current_gender_tracker_data = entry.get('gender_tracker')
        restored_pending = dict(entry.get('pending_gender_decisions', {}) or {})
        tracker_entries = (doc.current_gender_tracker_data or {}).get('entries', {})
        for tracker_item in tracker_entries.values() if isinstance(tracker_entries, dict) else []:
            if isinstance(tracker_item, dict) and tracker_item.get('raw_name'):
                restored_pending.setdefault(
                    tracker_item['raw_name'],
                    tracker_item.get('decision', 'auto'),
                )
        doc._pending_gender_decisions = restored_pending
    return 'glossary', None


# ---------------------------------------------------------------------------
# Backups
# ---------------------------------------------------------------------------

def editor_backup_dir(glossary_path):
    """The editor's backup folder: ``Backups`` next to the glossary."""
    return os.path.join(os.path.dirname(glossary_path), "Backups")


def count_editor_backups(backup_dir):
    """The Backup Settings dialog's count of backups in ``backup_dir``."""
    return len([f for f in os.listdir(backup_dir) if f.endswith('.json')])


def list_editor_backups(glossary_path):
    """This glossary's backups, oldest first (the files create_glossary_backup writes and
    _clean_old_backups prunes: ``<stem>*.json`` in :func:`editor_backup_dir`)."""
    backup_dir = editor_backup_dir(glossary_path)
    if not os.path.isdir(backup_dir):
        return []
    prefix = os.path.splitext(os.path.basename(glossary_path))[0]
    backups = []
    for file in os.listdir(backup_dir):
        if file.startswith(prefix) and file.endswith('.json'):
            file_path = os.path.join(backup_dir, file)
            backups.append((file_path, os.path.getmtime(file_path)))
    backups.sort(key=lambda x: x[1])
    return [path for path, _mtime in backups]


def load_editor_backup(backup_path):
    """A backup's glossary data (create_glossary_backup dumps ``current_glossary_data`` as JSON)."""
    with open(backup_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def backup_settings_message(enabled, max_backups):
    status = "enabled" if enabled else "disabled"
    if enabled:
        limit = max_backups
        limit_text = "unlimited" if limit == 0 else f"max {limit}"
        msg = f"Automatic backups {status} ({limit_text})"
    else:
        msg = f"Automatic backups {status}"
    return msg


# ---------------------------------------------------------------------------
# "Load" as the manual glossary (Manual Glossary Only copies it into the EPUB output)
# ---------------------------------------------------------------------------

def resolve_epub_output_dir(epub_path, config=None, log=_noop_log):
    """``(output_dir, file_base)`` of an EPUB, creating the folder with an empty progress file."""
    if not epub_path:
        return None, None
    file_base = os.path.splitext(os.path.basename(epub_path))[0]
    override_dir = None
    try:
        # Check every path used across the codebase:
        #   - OUTPUT_DIRECTORY / OUTPUT_DIR env vars
        #   - config['output_directory']
        # Trim whitespace + treat empty strings as "no override".
        _candidates = [
            os.environ.get('OUTPUT_DIRECTORY'),
            os.environ.get('OUTPUT_DIR'),
        ]
        if config is not None:
            _candidates.append(config.get('output_directory'))
        for _c in _candidates:
            if _c is None:
                continue
            _c = str(_c).strip().strip('"')
            if _c:
                override_dir = _c
                break
    except Exception:
        override_dir = None
    if override_dir:
        output_dir = os.path.join(os.path.abspath(override_dir), file_base)
    else:
        output_dir = file_base
    # macOS .app bundles can run with cwd='/' (read-only). Anchor
    # relative output paths against the source EPUB's directory in
    # that case — matches Retranslation_GUI's behavior.
    try:
        import sys as _sys
        if _sys.platform == 'darwin' and not os.path.isabs(output_dir):
            output_dir = os.path.join(os.path.dirname(os.path.abspath(epub_path)), output_dir)
    except Exception:
        pass
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
            # Seed an empty progress file so the retranslation GUI
            # can discover this folder later without regenerating it.
            progress_file_path = os.path.join(output_dir, "translation_progress.json")
            if not os.path.exists(progress_file_path):
                empty_prog = {"chapters": {}, "chapter_chunks": {}, "version": "2.1"}
                with open(progress_file_path, 'w', encoding='utf-8') as _f:
                    json.dump(empty_prog, _f, ensure_ascii=False, indent=2)
            log(f"\U0001F4C1 Created output folder: {output_dir}")
        except Exception as _e:
            log(f"\u26a0\ufe0f Failed to create output folder: {_e}")
            return None, file_base
    return output_dir, file_base


def manual_only_epub_path(selected_files, config=None):
    """The EPUB Manual Glossary Only copies into: the one selected EPUB, $EPUB_PATH, last EPUB."""
    epub_path = None
    try:
        selected = list(selected_files or [])
        epubs = [p for p in selected if str(p).lower().endswith('.epub')]
        if len(epubs) == 1:
            epub_path = epubs[0]
        if not epub_path:
            _env_ep = os.environ.get('EPUB_PATH')
            if _env_ep and os.path.isfile(_env_ep) and _env_ep.lower().endswith('.epub'):
                epub_path = _env_ep
        if not epub_path:
            _last = config.get('last_epub_path') if config is not None else None
            if _last and os.path.isfile(_last) and _last.lower().endswith('.epub'):
                epub_path = _last
    except Exception:
        pass
    return epub_path


def copy_glossary_to_epub_output(path, epub_path, config=None, log=_noop_log):
    """Copy the glossary to ``<EPUB output>/glossary.csv`` (Manual Glossary Only)."""
    out_dir, _base = resolve_epub_output_dir(epub_path, config, log)
    if out_dir:
        dest = os.path.join(out_dir, "glossary.csv")
        if os.path.abspath(path) == os.path.abspath(dest):
            log(
                f"\U0001F4CE Glossary already at output path: {dest}"
            )
        else:
            _shutil.copy2(path, dest)
            log(
                f"\U0001F4CE Copied glossary to EPUB output: {dest}"
            )


def editor_entry_line_number(file_path, row):
    """The line "Edit in Notepad" opens a row at: first line containing one of its names."""
    _line_num = 1
    try:
        # Skip generic column values that would match section headers
        _GENERIC_VALS = {
            'character', 'characters', 'terms', 'term', 'title', 'titles',
            'organization', 'organizations', 'location', 'locations',
            'item', 'items', 'ability', 'abilitys', 'abilities',
            'male', 'female', 'unknown', '',
        }
        # Gather searchable text — prefer unique fields (names) over generic (type/gender)
        search_terms = []
        # Check UserRole data first (original key for dict format)
        user_data = row.ref()
        if isinstance(user_data, str) and user_data.strip():
            search_terms.append(user_data.strip())
        # Then check visible columns, skipping generic values
        for col_idx in range(1, row.columnCount()):
            val = (row.text(col_idx) or '').strip()
            if val and len(val) >= 2 and val.lower() not in _GENERIC_VALS:
                search_terms.append(val)

        if search_terms:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as _f:
                _lines = _f.readlines()
            # Find the first line containing any of the search terms
            for term in search_terms:
                for _i, _ln in enumerate(_lines, 1):
                    if term in _ln:
                        _line_num = _i
                        break
                if _line_num > 1:
                    break
    except Exception:
        pass
    return _line_num


# ---------------------------------------------------------------------------
# Glossary Manager prompt profiles: config state (Balanced/Full, Minimal, Refinement)
# ---------------------------------------------------------------------------

#: The unnamed profile every glossary prompt bucket has (GlossaryManagerMixin attribute).
GLOSSARY_PROMPT_DEFAULT_PROFILE = "Default"


def glossary_prompt_profile_meta(profile_key):
    """Metadata for user-defined glossary prompt profile buckets."""
    meta = {
        'balanced_full': {
            'config_key': 'manual_glossary_prompt3',
            'legacy_config_key': 'manual_glossary_prompt',
            'attr': 'manual_glossary_prompt',
            'label': 'Balanced/Full Profile:',
            'empty_tip': 'Create a profile for the Balanced/Full extraction prompt',
        },
        'minimal': {
            'config_key': 'unified_auto_glosary_prompt3',
            'legacy_config_key': None,
            'attr': 'unified_auto_glosary_prompt3',
            'label': 'Minimal Profile:',
            'empty_tip': 'Create a profile for the Minimal glossary prompt',
        },
    }
    return meta.get(profile_key, meta['balanced_full'])


def is_default_glossary_prompt_profile(profile_name):
    return str(profile_name or '').strip().casefold() == GLOSSARY_PROMPT_DEFAULT_PROFILE.casefold()


def ensure_glossary_prompt_profiles(owner):
    """Ensure glossary prompt profile containers exist without seeding defaults."""
    profiles = owner.config.get('glossary_prompt_profiles', {})
    if not isinstance(profiles, dict):
        profiles = {}
    for key in ('balanced_full', 'minimal'):
        if not isinstance(profiles.get(key), dict):
            profiles[key] = {}
    owner.config['glossary_prompt_profiles'] = profiles

    active = owner.config.get('active_glossary_prompt_profiles', {})
    if not isinstance(active, dict):
        active = {}
    owner.config['active_glossary_prompt_profiles'] = active

    defaults = owner.config.get('glossary_prompt_profile_defaults', {})
    if not isinstance(defaults, dict):
        defaults = {}
    for key in ('balanced_full', 'minimal'):
        if key not in defaults:
            meta = glossary_prompt_profile_meta(key)
            defaults[key] = str(
                owner.config.get(meta['config_key'])
                or getattr(owner, meta['attr'], '')
                or ''
            )
    owner.config['glossary_prompt_profile_defaults'] = defaults
    return profiles, active


def glossary_prompt_profiles_for(owner, profile_key):
    profiles, _active = ensure_glossary_prompt_profiles(owner)
    return profiles.setdefault(profile_key, {})


def active_glossary_prompt_profile_for(owner, profile_key):
    _profiles, active = ensure_glossary_prompt_profiles(owner)
    return str(active.get(profile_key, '') or '').strip()


def set_active_glossary_prompt_profile(owner, profile_key, profile_name):
    _profiles, active = ensure_glossary_prompt_profiles(owner)
    if profile_name and not is_default_glossary_prompt_profile(profile_name):
        active[profile_key] = profile_name
    else:
        active.pop(profile_key, None)
    owner.config['active_glossary_prompt_profiles'] = active


def default_glossary_prompt_profile_text(owner, profile_key):
    ensure_glossary_prompt_profiles(owner)
    defaults = owner.config.setdefault('glossary_prompt_profile_defaults', {})
    return str(defaults.get(profile_key, '') or '')


def set_default_glossary_prompt_profile_text(owner, profile_key, text):
    ensure_glossary_prompt_profiles(owner)
    defaults = owner.config.setdefault('glossary_prompt_profile_defaults', {})
    defaults[profile_key] = text or ''
    owner.config['glossary_prompt_profile_defaults'] = defaults


def store_glossary_prompt_current(owner, profile_key, text):
    """Make ``text`` the bucket's current prompt (owner attribute + config keys)."""
    meta = glossary_prompt_profile_meta(profile_key)
    setattr(owner, meta['attr'], text)
    owner.config[meta['config_key']] = text
    if meta.get('legacy_config_key'):
        owner.config[meta['legacy_config_key']] = text
    return text


def default_glossary_refinement_system_prompt():
    try:
        from glossary_refinement import DEFAULT_GLOSSARY_REFINEMENT_SYSTEM_PROMPT
        return DEFAULT_GLOSSARY_REFINEMENT_SYSTEM_PROMPT
    except Exception:
        return ""


def default_glossary_refinement_user_prompt():
    try:
        from glossary_refinement import DEFAULT_GLOSSARY_REFINEMENT_USER_PROMPT
        return DEFAULT_GLOSSARY_REFINEMENT_USER_PROMPT
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Glossary Manager prompt profiles without the widgets (mobile)
# ---------------------------------------------------------------------------

class GlossaryPromptProfiles:
    """The Balanced/Full or Minimal prompt profiles of the Glossary Manager, without widgets.

    The profile rules are prompt_profiles' Default-plus-named engine (``PrefillState`` and the
    ``prefill_*`` steps the desktop "Asst. Prompt" dialog uses), over the config buckets the
    desktop controls keep (``glossary_prompt_profiles[key]``,
    ``active_glossary_prompt_profiles[key]``, ``glossary_prompt_profile_defaults[key]``). Each
    action then writes the config the way the matching GlossaryManagerMixin action does
    (``_on_glossary_prompt_profile_selected``, ``_auto_save_...``, ``_new_...``, ``_save_...``,
    ``_delete_...``), including the current prompt keys (:func:`store_glossary_prompt_current`).
    ``text`` is what the desktop prompt editor shows. tests/test_glossary_document.py replays
    random action sequences on the desktop controls and on this class.

    ``owner`` is the TranslatorGUI stand-in (``config`` + the prompt attribute), ``persist()``
    the config saver (False = failed), ``log`` gets the desktop log lines. Actions that show a
    box on the desktop return ``(title, text)``.
    """

    def __init__(self, owner, profile_key, current_text='', *, persist=None, log=None):
        import prompt_profiles

        self._pp = prompt_profiles
        self.owner = owner
        self.key = profile_key
        self.persist = persist
        self.log = log or _noop_log
        ensure_glossary_prompt_profiles(owner)
        self.state = prompt_profiles.PrefillState()
        self._load()
        # Opening the tab: the editor gets the current prompt (staged into the selected
        # profile), then the active profile is applied.
        self.stage(self.selected_name(), current_text)
        self.select(self.selected_name())

    # ---- state <-> config ------------------------------------------------------------
    def _load(self):
        profiles = glossary_prompt_profiles_for(self.owner, self.key)
        active = active_glossary_prompt_profile_for(self.owner, self.key)
        self.state.profiles = profiles
        self.state.default_prompt = default_glossary_prompt_profile_text(self.owner, self.key)
        self.state.active_name = (
            active if active in profiles and not is_default_glossary_prompt_profile(active) else ''
        )

    def _store_profiles(self):
        self.owner.config['glossary_prompt_profiles'][self.key] = self.state.profiles

    def _store_current(self):
        return store_glossary_prompt_current(self.owner, self.key, str(self.state.text or '').strip())

    def _persist(self):
        if self.persist is None:
            return None
        try:
            return self.persist()
        except Exception:
            return False

    @property
    def text(self):
        return self.state.text

    @property
    def names(self):
        return [GLOSSARY_PROMPT_DEFAULT_PROFILE] + [
            name for name in self.state.profiles if not is_default_glossary_prompt_profile(name)
        ]

    def selected_name(self):
        return self.state.active_name or GLOSSARY_PROMPT_DEFAULT_PROFILE

    # ---- actions ---------------------------------------------------------------------
    def select(self, name):
        """Pick a profile (or Default); returns its text, or None for an unknown name."""
        name = str(name or '').strip()
        text = self._pp.prefill_select(self.state, name)
        if text is None:
            return None
        set_active_glossary_prompt_profile(self.owner, self.key, self.state.active_name)
        self._store_current()
        return text

    def stage(self, name, text):
        """An edit in the prompt editor (auto-saved into the active profile or Default)."""
        name = str(name or '').strip()
        staged_into_default = (
            self.state.active_name not in self.state.profiles
            and (not name or is_default_glossary_prompt_profile(name))
        )
        self._pp.prefill_stage(self.state, name, str(text or ''))
        if staged_into_default:
            set_default_glossary_prompt_profile_text(self.owner, self.key, self.state.default_prompt)
            set_active_glossary_prompt_profile(self.owner, self.key, '')
        self._store_current()

    def new(self):
        """+ New Profile: an empty profile, selected and saved."""
        name = self._pp.prefill_new(self.state)
        self._store_profiles()
        set_active_glossary_prompt_profile(self.owner, self.key, name)
        self._store_current()
        self._persist()
        self.log(f"✅ Created glossary prompt profile: '{name}'")
        return name

    def save(self, name, text):
        """Save Profile under ``name`` (the edited name; "Default" saves the Default text)."""
        name = str(name or '').strip()
        active = self.state.active_name
        previous_profiles = dict(self.state.profiles)
        outcome = self._pp.prefill_save(self.state, name, str(text or ''))
        if not outcome.ok:
            return outcome.title, outcome.error
        if is_default_glossary_prompt_profile(name):
            set_default_glossary_prompt_profile_text(self.owner, self.key, self.state.default_prompt)
            set_active_glossary_prompt_profile(self.owner, self.key, '')
            self._store_current()
            self._persist()
            self.log("✅ Saved default glossary prompt profile")
            return None
        self._store_profiles()
        set_active_glossary_prompt_profile(self.owner, self.key, name)
        self._store_current()
        if self._persist() is False:
            self.state.profiles = previous_profiles
            self.state.active_name = active
            self._store_profiles()
            set_active_glossary_prompt_profile(self.owner, self.key, active)
            return "Save Failed", "Could not save the glossary prompt profile. Please try again."
        self.log(f"✅ Saved glossary prompt profile: '{name}'")
        return None

    def delete(self, name, confirm=None):
        """Delete Profile ``name``; ``confirm(name)`` is the desktop's Yes/No question."""
        name = str(name or '').strip()
        if is_default_glossary_prompt_profile(name):
            return "Default Profile", "The Default glossary prompt profile cannot be deleted."
        outcome = self._pp.prefill_delete(self.state, name, confirm=confirm)
        if not outcome.ok:
            return (outcome.title, outcome.error) if outcome.error else None
        self._store_profiles()
        set_active_glossary_prompt_profile(self.owner, self.key, self.state.active_name)
        self._store_current()
        self._persist()
        self.log(f"\U0001f5d1️ Deleted glossary prompt profile: '{name}'")
        return None


class RefinementPromptProfiles:
    """The Glossary Refinement prompt profiles (a system + user prompt pair each), without widgets.

    The same Default-plus-named engine as :class:`GlossaryPromptProfiles`; a pair travels through
    it as one canonical text. Config keys as the desktop controls
    (``_create_refinement_prompt_profile_controls``) write them after every action:
    ``glossary_refinement_prompt_profiles`` / ``..._profile_default`` /
    ``active_glossary_refinement_prompt_profile`` and the runtime
    ``glossary_refinement_system_prompt`` / ``glossary_refinement_user_prompt`` (also set on
    ``owner``). ``system`` / ``user`` are what the desktop prompt editors show when it opens.
    """

    PROFILES_KEY = 'glossary_refinement_prompt_profiles'
    DEFAULT_KEY = 'glossary_refinement_prompt_profile_default'
    ACTIVE_KEY = 'active_glossary_refinement_prompt_profile'
    SYSTEM_KEY = 'glossary_refinement_system_prompt'
    USER_KEY = 'glossary_refinement_user_prompt'

    def __init__(self, owner, system='', user='', *, persist=None, log=None):
        import prompt_profiles

        self._pp = prompt_profiles
        self.owner = owner
        self.persist = persist
        self.log = log or _noop_log
        config = owner.config
        initial = {'system': str(system or '').strip(), 'user': str(user or '').strip()}

        def clean_pair(value, fallback):
            if not isinstance(value, dict):
                return dict(fallback)
            return {key: str(value.get(key, fallback[key]) or '') for key in ('system', 'user')}

        raw_profiles = config.get(self.PROFILES_KEY, {})
        profiles = {
            name: self._encode(clean_pair(pair, initial))
            for name, pair in (raw_profiles.items() if isinstance(raw_profiles, dict) else ())
            if isinstance(name, str) and name.strip() and name.strip().casefold() != 'default'
            and isinstance(pair, dict)
        }
        self.state = prompt_profiles.PrefillState()
        self._pp.prefill_load(self.state, profiles, self._encode(clean_pair(config.get(self.DEFAULT_KEY), initial)),
                              config.get(self.ACTIVE_KEY, ''))
        self._sync()

    @staticmethod
    def _encode(pair):
        return json.dumps([pair.get('system', ''), pair.get('user', '')], ensure_ascii=False)

    @staticmethod
    def _decode(text):
        if not text:
            return {'system': '', 'user': ''}
        system, user = json.loads(text)
        return {'system': system, 'user': user}

    @property
    def pair(self):
        """The system / user prompts the editors show."""
        return self._decode(self.state.text)

    @property
    def names(self):
        return [self._pp.PREFILL_DEFAULT_NAME] + list(self.state.profiles)

    def selected_name(self):
        return self.state.active_name or self._pp.PREFILL_DEFAULT_NAME

    def _clean(self, pair):
        return {'system': str(pair.get('system', '') or '').strip(), 'user': str(pair.get('user', '') or '').strip()}

    def _sync(self):
        config = self.owner.config
        pair = self._clean(self.pair)  # the editors' text, as the desktop reads it back
        config[self.PROFILES_KEY] = {name: self._decode(text) for name, text in self.state.profiles.items()}
        config[self.DEFAULT_KEY] = self._decode(self.state.default_prompt)
        config[self.ACTIVE_KEY] = self.state.active_name
        self.owner.glossary_refinement_system_prompt = pair['system'] or default_glossary_refinement_system_prompt()
        self.owner.glossary_refinement_user_prompt = pair['user']
        config[self.SYSTEM_KEY] = self.owner.glossary_refinement_system_prompt
        config[self.USER_KEY] = self.owner.glossary_refinement_user_prompt

    def _snapshot(self):
        keys = (self.PROFILES_KEY, self.DEFAULT_KEY, self.ACTIVE_KEY, self.SYSTEM_KEY, self.USER_KEY)
        return (_copy.deepcopy(self.state), {key: (key in self.owner.config, _copy.deepcopy(self.owner.config.get(key)))
                                             for key in keys})

    def _persist(self, before):
        self._sync()
        result = None
        if self.persist is not None:
            try:
                result = self.persist()
            except Exception:
                result = False
        if result is not False:
            return True
        self.state, saved_config = before
        self._sync()
        for key, (present, value) in saved_config.items():
            if present:
                self.owner.config[key] = value
            else:
                self.owner.config.pop(key, None)
        return False

    _SAVE_FAILED = ("Save Failed", "Could not save the refinement prompt profiles. Please try again.")

    def select(self, name):
        """Pick a profile (or Default); returns its pair, or None for an unknown name."""
        text = self._pp.prefill_select(self.state, str(name or '').strip())
        if text is None:
            return None
        self._sync()
        return self.pair

    def stage(self, name, pair):
        """An edit in either prompt editor."""
        self._pp.prefill_stage(self.state, str(name or ''), self._encode(self._clean(pair)))
        self._sync()

    def new(self):
        before = self._snapshot()
        name = self._pp.prefill_new(self.state)
        if not self._persist(before):
            return self._SAVE_FAILED
        self.log(f"✅ Created refinement prompt profile: '{name}'")
        return None

    def save(self, name, pair):
        before = self._snapshot()
        outcome = self._pp.prefill_save(self.state, str(name or ''), self._encode(self._clean(pair)))
        if not outcome.ok:
            return outcome.title, outcome.error
        if not self._persist(before):
            return self._SAVE_FAILED
        self.log(f"✅ Saved refinement prompt profile: '{self.state.active_name or 'Default'}'")
        return None

    def delete(self, name, confirm=None):
        name = str(name or '').strip()
        if name.casefold() == 'default':
            return "Default Profile", "The Default refinement prompt profile cannot be deleted."
        if name not in self.state.profiles:
            return "Profile Not Found", "Select an existing profile to delete."
        if confirm is not None and not confirm(name):
            return None
        before = self._snapshot()
        self._pp.prefill_delete(self.state, name)
        if not self._persist(before):
            return self._SAVE_FAILED
        self.log(f"\U0001f5d1️ Deleted refinement prompt profile: '{name}'")
        return None


# ---------------------------------------------------------------------------
# GlossaryDocument: the editor without Qt (mobile)
# ---------------------------------------------------------------------------

class GlossaryEditorError(Exception):
    """An editor action the desktop answers with an error or warning box (its text)."""


class GlossaryDocument:
    """One glossary file open in the editor, driven like the desktop Glossary Editor.

    Each action runs the same shared steps in the same order as the desktop action of the same
    name (the ``_setup_glossary_editor_tab`` closures), with the dialogs left to the caller:
    confirmations are asked before calling, error boxes raise :class:`GlossaryEditorError`
    and information boxes come back as ``(title, text)``. Files written are byte-identical to
    the desktop editor's (tests/test_glossary_document.py drives both on real glossaries).

    ``backup(doc, operation_name) -> bool`` stands for the desktop's create_glossary_backup
    (glossary_files); without it no backups are written. ``log`` receives the lines the desktop
    appends to its log. ``owner`` defaults to an :class:`EditorOwner` for ``config``.

    As on the desktop, every action that saves also reloads the file, which clears the undo
    history; cell edits, Replace and Resolve Gender stay undoable until the next save.
    """

    def __init__(self, config=None, *, owner=None, backup=None, log=None):
        self.owner = owner if owner is not None else EditorOwner(config)
        self.config = self.owner.config
        self.custom_entry_types = getattr(self.owner, 'custom_entry_types', None)
        self.backup = backup
        self.log = log or _noop_log
        self.path = ''
        reset_document(self)
        self.current_glossary_sections = []
        self.glossary_column_fields = []
        self._original_translated_map = {}
        self._undo_stack = []
        self._redo_stack = []
        self._undo_max = EDITOR_UNDO_MAX
        self.stats_text = "No glossary loaded"
        self._last_loaded_glossary_log = ''
        self._last_find_text = ""
        self._last_find_pos = -1

    # ---- helpers -------------------------------------------------------------------------
    def _create_backup(self, operation_name):
        if self.backup is None:
            return True
        return self.backup(self, operation_name)

    def _require_data(self):
        if not self.current_glossary_data:
            raise GlossaryEditorError("No glossary loaded")

    def _save_or_raise(self):
        """``save_current_glossary`` (an error box on failure)."""
        if not self.path or self.current_glossary_data is None:
            return False
        try:
            return save_document(self, self.path, self.config)
        except Exception as e:
            raise GlossaryEditorError(f"Failed to save: {e}") from e

    def gender_settings(self):
        return editor_gender_settings(self.owner, self.config)

    def entry_type_config(self):
        return configured_entry_types(self.config, self.custom_entry_types)

    # ---- load / view ---------------------------------------------------------------------
    @classmethod
    def open(cls, path, config=None, **kwargs):
        doc = cls(config, **kwargs)
        doc.load(path)
        return doc

    def load(self, path=None):
        """Read the file (the editor's background load and ``apply_loaded_glossary_result``)."""
        if path is not None:
            self.path = path
        path = self.path
        if not path or not os.path.exists(path):
            raise GlossaryEditorError("Please select a valid glossary file")
        try:
            payload = parse_glossary_file(path, self.config, self.custom_entry_types)
        except Exception as exc:
            self.stats_text = "Failed to load glossary"
            self.log(f"Failed to load glossary: {exc}")
            raise GlossaryEditorError(f"Failed to load glossary: {exc}") from exc
        entries, _column_fields = apply_parse_payload(self, payload)
        self.stats_text = payload.get('stats_text', f"Total entries: {len(entries)}")
        log_msg = f"Loaded {len(entries)} entries from glossary"
        if self._last_loaded_glossary_log != log_msg:
            self.log(log_msg)
            self._last_loaded_glossary_log = log_msg
        self._last_find_text = ""
        self._last_find_pos = -1
        self._undo_stack.clear()
        self._redo_stack.clear()
        return payload

    def reload_after_restore(self):
        """Save the in-memory data and re-read it (``load_glossary_for_editing(skip_file_read=True)``).

        Returns the save error text, or None.
        """
        path = self.path
        if not path or not os.path.exists(path):
            raise GlossaryEditorError("Please select a valid glossary file")
        save_error = None
        try:
            if self.current_glossary_data is not None:
                try:
                    save_document(self, path, self.config)
                except Exception as e:
                    save_error = f"Failed to save: {e}"
            entries, column_fields, stats_text = reparse_glossary_file(
                self, path, self.config, self.custom_entry_types
            )
            self.glossary_column_fields = list(column_fields)
            self.stats_text = stats_text
            _log_msg = f"✅ Loaded {len(entries)} entries from glossary"
            if self._last_loaded_glossary_log != _log_msg:
                self.log(_log_msg)
                self._last_loaded_glossary_log = _log_msg
            self._last_find_text = ""
            self._last_find_pos = -1
        except Exception as e:
            self.stats_text = "Failed to load glossary"
            self.log(f"❌ Failed to load glossary: {e}")
        return save_error

    def rows(self, visible_source_indices=None):
        """The editor view's rows (:class:`EditorRow`, numbered from 1, gender labels and status
        as the desktop tree shows them)."""
        return editor_rows(
            self.glossary_column_fields, self.current_glossary_data, self.current_glossary_format,
            visible_source_indices, self.gender_status,
        )

    def entry(self, ref):
        return entry_for_ref(self.current_glossary_data, self.current_glossary_format, ref)

    def gender_status(self, entry):
        """The row's tracker status (label, conflict, unresolved) or None."""
        return entry_gender_status(entry, self.current_gender_tracker_data, self.gender_settings)

    def is_changed(self, ref, col_key, value):
        """Whether a translated cell differs from the last saved value (the orange highlight)."""
        baseline = baseline_translated(
            self._original_translated_map, self.current_glossary_format, ref, col_key
        )
        return baseline is not None and value != baseline

    def translated_changes(self):
        """``[(old, new)]`` translated-name changes since the last save."""
        return collect_translated_changes(
            self.current_glossary_data, self.current_glossary_format, self._original_translated_map
        )

    # ---- undo / redo -------------------------------------------------------------------
    def push_undo(self):
        return push_undo_snapshot(self, self._undo_stack, self._redo_stack, self._undo_max)

    def can_undo(self):
        return bool(self._undo_stack)

    def can_redo(self):
        return bool(self._redo_stack)

    def undo(self):
        """``_undo_action``: ``('html', (files, replacements) or None)``,
        ``('glossary', save_error)`` or ``(None, None)``."""
        kind, reverse = undo_step(self, self._undo_stack, self._redo_stack)
        if kind == 'html':
            try:
                files_updated, total_repl = self.update_output_files(reverse)
                self.log(f"↶ Undo: reverted {total_repl} replacement(s) across {files_updated} output file(s).")
            except Exception as e:
                self.log(f"⚠️ Undo (output files) failed: {e}")
                return 'html', None
            return 'html', (files_updated, total_repl)
        if kind != 'glossary':
            return None, None
        return 'glossary', self.reload_after_restore()

    def redo(self):
        """``_redo_action`` (see :meth:`undo`)."""
        kind, entry = redo_step(self, self._undo_stack, self._redo_stack)
        if kind == 'html':
            try:
                files_updated, total_repl = self.update_output_files(list(entry.get('changes', [])))
                self.log(f"↷ Redo: re-applied {total_repl} replacement(s) across {files_updated} output file(s).")
            except Exception as e:
                self.log(f"⚠️ Redo (output files) failed: {e}")
                return 'html', None
            return 'html', (files_updated, total_repl)
        if kind != 'glossary':
            return None, None
        return 'glossary', self.reload_after_restore()

    # ---- save ----------------------------------------------------------------------------
    def save(self):
        """Write the file (``save_current_glossary``); True when written."""
        return self._save_or_raise()

    def update_output_files(self, changes):
        """Replace ``[(old, new)]`` in the book's output files (``update_html_files``)."""
        return update_output_files(
            self.path,
            changes,
            self.log,
            getattr(self.owner, 'enable_parallel_extraction_var', self.config.get('enable_parallel_extraction', False)),
            getattr(self.owner, 'extraction_workers_var', self.config.get('extraction_workers', os.environ.get('EXTRACTION_WORKERS', 1))),
        )

    def save_edits(self, update_output_files=None):
        """The Save button (``save_edited_glossary``), after the user confirmed.

        ``update_output_files`` is the "Update output files on save" switch (default: the
        config). When it is on and :meth:`translated_changes` is not empty the desktop first
        asks :func:`update_output_files_prompt` and saves nothing when the user declines.
        Returns ``{'saved', 'files_updated', 'replacements'}``.
        """
        if update_output_files is None:
            update_output_files = bool(self.config.get('update_html_on_save', True))
        changes = self.translated_changes()
        report = {'saved': False, 'files_updated': 0, 'replacements': 0}
        if not self._create_backup("before_save"):
            return report
        if not self._save_or_raise():
            return report
        report['saved'] = True
        if update_output_files and changes:
            files_updated, total_repl = self.update_output_files(changes)
            self.log(f"Updated {files_updated} files with translated-name changes ({total_repl} replacements).")
            report.update(files_updated=files_updated, replacements=total_repl)
        baseline = saved_translated_baseline(self.current_glossary_data, self.current_glossary_format)
        if baseline is not None:
            self._original_translated_map = baseline
        self.log(f"✅ Saved glossary to: {self.path}")
        return report

    def save_as(self, path):
        """Save As (the document then points at ``path``, as the desktop's file box does)."""
        if not self.current_glossary_data:
            raise GlossaryEditorError("No glossary loaded")
        try:
            save_glossary_as(path, self.current_glossary_data, self.current_glossary_format, self.config)
        except Exception as e:
            raise GlossaryEditorError(f"Failed to save: {e}") from e
        self.path = path
        self.log(f"✅ Saved glossary as: {path}")
        return "Success", f"Glossary saved to {os.path.basename(path)}"

    def export_selection(self, path, refs):
        """Export Selection of the rows ``refs``."""
        refs = list(refs)
        if not refs:
            raise GlossaryEditorError("No entries selected")
        try:
            export_selected_entries(
                path, self.current_glossary_data, self.current_glossary_format, refs, self.config
            )
        except Exception as e:
            raise GlossaryEditorError(f"Failed to export: {e}") from e
        return "Success", f"Exported {len(refs)} entries to {os.path.basename(path)}"

    def convert(self, csv_path):
        """Convert Format to ``csv_path`` (reloads when it overwrote this file)."""
        self._require_data()
        if not self._create_backup("before_export"):
            return None
        try:
            fmt_label = convert_to_csv(self, csv_path, self.config)
            if fmt_label is None:
                raise GlossaryEditorError("No entries to export")
            self.log(f"✅ Exported glossary to {fmt_label} format: {csv_path}")
            if self.path and os.path.normpath(csv_path) == os.path.normpath(self.path):
                self.load()
        except GlossaryEditorError:
            raise
        except Exception as e:
            self.log(f"❌ CSV export failed: {e}")
            raise GlossaryEditorError(f"Failed to export CSV: {e}") from e
        return "Success", f"Glossary exported to {fmt_label} format:\n{csv_path}"

    # ---- editing -------------------------------------------------------------------------
    def edit_cell(self, ref, col_key, value):
        """A cell edit (``save_edit``); returns ``(stored_value, new_ref)``."""
        new_value = normalize_edit_value(col_key, value)
        self.push_undo()
        _row_idx, new_ref, _ref_changed = apply_entry_edit(self, ref, col_key, new_value)
        return new_value, new_ref

    def resolve_gender(self, ref, decision):
        """Resolve Gender for a conflicted row: 'auto', 'male' or 'female' (saved on Save)."""
        data_entry = self.entry(ref)
        if not can_resolve_gender(self.gender_status(data_entry)):
            return False
        tracker_entry = tracker_entry_for_entry(data_entry, self.current_gender_tracker_data)
        if not data_entry or not tracker_entry:
            return False
        raw_name = str(data_entry.get('raw_name', '') or '').strip()
        self.push_undo()
        apply_gender_decision(self, tracker_entry, data_entry, raw_name, decision)
        return True

    def delete(self, refs):
        """Delete Selected after the confirmation; returns the success box or None."""
        refs = list(refs)
        if not refs:
            raise GlossaryEditorError("Please select entries to delete")
        count = len(refs)
        if not self._create_backup(f"before_delete_{count}"):
            return None
        self.push_undo()
        indices_to_delete, keys_to_delete = delete_entries(self, refs)
        if self._save_or_raise():
            self.load()
            return "Success", f"Deleted {len(indices_to_delete) + len(keys_to_delete)} entries"
        return None

    def clean_empty_fields(self):
        """Clean Empty Fields; returns the information box or None."""
        self._require_data()
        if self.current_glossary_format not in LIST_FORMATS:
            return None
        empty_fields_found, fields_cleaned = count_empty_fields(self.current_glossary_data)
        if not empty_fields_found:
            return "Info", "No empty fields found in glossary"
        if not self._create_backup("before_clean"):
            return None
        total_cleaned = remove_empty_fields(self.current_glossary_data)
        if self._save_or_raise():
            self.load()
            return "Success", clean_empty_fields_message(total_cleaned, fields_cleaned)
        return None

    def remove_duplicates(self):
        """Remove Duplicates (the Balanced/Full dedup engine); returns the information box.

        Like the desktop editor it sets GLOSSARY_DISABLE_HONORIFICS_FILTER in os.environ.
        """
        self._require_data()
        if self.current_glossary_format not in LIST_FORMATS:
            return None
        try:
            original_count = len(self.current_glossary_data)
            self.current_glossary_data = remove_duplicate_entries(
                self.current_glossary_data,
                self.path,
                getattr(self, 'current_gender_tracker_data', None),
                self.config.get('glossary_disable_honorifics_filter', False),
            )
            duplicates_removed = original_count - len(self.current_glossary_data)
            if duplicates_removed > 0:
                if self.config.get('glossary_auto_backup', False):
                    self._create_backup(f"before_remove_{duplicates_removed}_dupes")
                if self._save_or_raise():
                    self.load()
                    self.log(f"\U0001f5d1️ Removed {duplicates_removed} duplicates based on raw_name")
                    return "Success", f"Removed {duplicates_removed} duplicate entries"
                return None
            return "Info", "No duplicates found"
        except ImportError:
            unique_entries, duplicates = remove_duplicate_entries_fallback(self.current_glossary_data)
            if duplicates > 0:
                self.current_glossary_data = unique_entries
                if self._save_or_raise():
                    self.load()
                    return "Success", f"Removed {duplicates} duplicate entries"
                return None
            return "Info", "No duplicates found"

    def trim_preview(self, top_n):
        return trim_preview_text(
            document_entry_count(self.current_glossary_data, self.current_glossary_format), int(top_n)
        )

    def trim(self, top_n):
        """Trim Entries to the first ``top_n``; returns the success box or None."""
        self._require_data()
        try:
            top_n = int(top_n)
        except (TypeError, ValueError) as exc:
            raise GlossaryEditorError("Please enter valid numbers") from exc
        entries_to_remove = len(self.current_glossary_data) - top_n
        if entries_to_remove > 0:
            if not self._create_backup(f"before_trim_{entries_to_remove}"):
                return None
        trim_entries(self, top_n)
        if self._save_or_raise():
            self.load()
            return "Success", f"Trimmed glossary to {top_n} entries"
        return None

    def filter_types(self):
        """The types Filter Entries offers (empty for untyped glossaries)."""
        if not is_new_format_data(self.current_glossary_data, self.current_glossary_format):
            return []
        return filter_entry_types(self.config, self.current_glossary_data)

    def filter_matcher(self, *, kept_types=None, search_text='', gender_value='all', type_limits=None):
        """``matches(entry, type_counts)`` for the Filter Entries choices.

        ``kept_types``: {type: keep}; ``type_limits``: {type: "First N" text};
        ``gender_value``: 'all', 'Male', 'Female' or 'Unknown'.
        """
        is_new_format = is_new_format_data(self.current_glossary_data, self.current_glossary_format)
        kept_types = dict(kept_types or {})
        type_limits = dict(type_limits or {})

        def matches(entry, type_counts=None):
            return entry_matches_filter(
                entry, type_counts, is_new_format=is_new_format, kept_types=kept_types,
                search_text=search_text, gender_value=gender_value, config=self.config,
                type_limits=type_limits,
            )

        return matches

    def preview_filter(self, **choices):
        """Preview Filter: ``(matching, removed)``."""
        matching = count_filter_matches(
            self.current_glossary_data, self.current_glossary_format, self.filter_matcher(**choices)
        )
        entry_count = document_entry_count(self.current_glossary_data, self.current_glossary_format)
        return matching, entry_count - matching

    def apply_filter(self, **choices):
        """Apply Filter (list glossaries); returns the success box or None."""
        self._require_data()
        if self.current_glossary_format not in LIST_FORMATS:
            return None
        filtered = filter_list_entries(self.current_glossary_data, self.filter_matcher(**choices))
        removed = len(self.current_glossary_data) - len(filtered)
        if removed > 0:
            if not self._create_backup(f"before_filter_remove_{removed}"):
                return None
        self.current_glossary_data[:] = filtered
        if self._save_or_raise():
            self.load()
            return "Success", f"Filter applied!\n\nKept: {len(filtered)} entries\nRemoved: {removed} entries"
        return None

    # ---- find / replace ------------------------------------------------------------------
    def find_next(self, text, rows):
        """Find Next over ``rows`` (as shown); returns the row index or None."""
        if not text or not rows:
            return None
        if text != self._last_find_text:
            self._last_find_pos = -1
        idx = find_next_index(
            text, len(rows), lambda i: [rows[i].text(c) for c in range(rows[i].columnCount())],
            self._last_find_pos,
        )
        if idx is not None:
            self._last_find_text = text
            self._last_find_pos = idx
        return idx

    def replace_in(self, row, text, repl):
        """Replace in one row (snapshots for undo only when it matches)."""
        if row_has_match(row, text):
            self.push_undo()
        if not text or row is None:
            return 0
        return replace_in_row(self, row, self.glossary_column_fields, text, repl)

    def replace_all(self, text, repl, rows=None):
        """Replace All over ``rows`` (default: every row).

        The desktop replaces in the rows its view holds: all of them, or only the used ones
        while Hide unused entries is on. Returns the number of replacements; when it is 0 the
        desktop offers :meth:`replace_in_output_files` ("No glossary match").
        """
        if rows is None:
            rows = self.rows()
        if not rows or not text:
            return 0
        if any(row_has_match(row, text) for row in rows if row is not None):
            self.push_undo()
        total_repl = 0
        for row in rows:
            if row is None:
                continue
            total_repl += replace_in_row(self, row, self.glossary_column_fields, text, repl)
        self._last_find_text = text
        return total_repl

    def replace_in_output_files(self, old, new):
        """The "No glossary match" fallback: replace in the output files, undoably."""
        files_updated, total_file_repl = self.update_output_files([(old, new)])
        if total_file_repl > 0:
            push_html_undo_snapshot(self._undo_stack, self._redo_stack, self._undo_max, [(old, new)])
        self.log(f"Updated {files_updated} files directly from Find/Replace fallback ({total_file_repl} replacements).")
        return files_updated, total_file_repl

    # ---- hide unused / output folder -----------------------------------------------------
    def output_dir(self, source_path=None, source_map=None):
        """The translated output folder: from the input file, else from the glossary's place."""
        source_dir = editor_output_dir_from_source(source_path, self.config)
        if source_dir:
            return source_dir
        return editor_output_dir_from_glossary(
            self.path, source_map, lambda path: editor_output_dir_from_source(path, self.config)
        )

    def used_rows(self, output_dir, emit_progress=_noop_log):
        """Hide unused entries: :func:`compute_used_rows` for this document and ``output_dir``."""
        fields, specs = editor_row_specs(
            self.glossary_column_fields, self.current_glossary_data, self.current_glossary_format
        )
        entries = editor_usage_entries(fields, specs)
        return compute_used_rows(entries, output_dir, len(specs), None, emit_progress)

    # ---- backups -------------------------------------------------------------------------
    def backups(self):
        """This glossary's backups, oldest first."""
        return list_editor_backups(self.path) if self.path else []

    def restore_backup(self, backup_path):
        """Replace the entries with a backup's and save them in this file's format.

        The desktop has no restore button for editor backups (users copy them out of the
        Backups folder); this saves the backup's entries like an edit, after a backup of the
        current state.
        """
        self._require_data()
        data = load_editor_backup(backup_path)
        if isinstance(data, list) != (self.current_glossary_format in LIST_FORMATS):
            raise GlossaryEditorError("The backup does not match this glossary's format")
        if not self._create_backup("before_restore"):
            return None
        self.current_glossary_data = data
        if self._save_or_raise():
            self.load()
            return "Success", f"Restored {os.path.basename(backup_path)}"
        return None
