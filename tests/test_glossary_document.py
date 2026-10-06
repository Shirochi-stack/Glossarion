"""U6: glossary_document (the Glossary Editor document model) parity and API tests.

The pure inner functions of the desktop Glossary Editor (the ``_setup_glossary_editor_tab``
closure of GlossaryManager_GUI, ``_on_tree_double_click`` and ``convert_glossary_format``) and
the editor's parse helpers moved to ``glossary_document`` with explicit parameters; the
closures and methods stay as thin wrappers. The oracle is GlossaryManager_GUI.py at
``U6_BASE_SHA`` (``git show``), run against the live module globals.

Tiers:
* H (hygiene): glossary_document imports without PySide6 / translator_gui / dpi_setup, parses
  as Python 3.10, and both files keep uniform line endings;
* V (verbatim): every moved body equals the frozen source plus the documented edits
  (``MOVED_BLOCKS``), and GlossaryManager_GUI differs from the frozen file only inside the
  rewired spans (``REWIRE_SPANS``);
* D (differential fuzz, >= ``PARITY_U6_STATES`` = 500 states each): the frozen methods and
  closures vs the working-tree wrappers and the shared functions: token / legacy CSV and JSON
  parsing, saving (token CSV / legacy CSV / JSON, gender-variant resolution, tracker sync,
  section aliasing), Convert Format, the view helpers, Filter Entries, Find/Replace rows,
  Update output files, the editor's glossary auto-selection and the hide-unused helpers;
* F (file system, real glossaries): the offscreen frozen editor tab vs the working-tree tab,
  driven through their buttons and dialogs on copies of src/Glossary/<book>/ glossaries (plus
  JSON list / dict and legacy CSV variants) for every editor operation: glossary bytes, gender
  tracker, backups, output files, tree rows, editor state, undo stacks, boxes and logs;
* M (mobile API): GlossaryDocument runs the same operation sequences on a third copy and
  writes the same files and state as the frozen desktop editor;
* O (owner): EditorOwner reproduces the HeadlessOwner attributes the editor reads;
* S (smoke): the Glossary Manager dialog's editor tab loads a real glossary offscreen.

Run (repository root)::

    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic \\
        tests/test_glossary_document.py
"""

from __future__ import annotations

import ast
import contextlib
import copy
import json
import os
import random
import re
import shutil
import subprocess
import sys
import textwrap
import time
import types
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
TESTS = REPO_ROOT / "tests"
for _p in (str(TESTS), str(SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import glossary_document as gd  # noqa: E402

HAS_QT = __import__("importlib").util.find_spec("PySide6") is not None
needs_qt = pytest.mark.skipif(not HAS_QT, reason="needs PySide6")
#: Desktop handlers raising inside Qt slots are part of the behaviour under comparison.
pytestmark = pytest.mark.qt_no_exception_capture

STATES = max(1, int(os.environ.get("PARITY_U6_STATES", "500")))
SEED = int(os.environ.get("PARITY_U6_SEED", "6060"))
#: Real glossaries copied for tier F / M (folders under src/Glossary).
REAL_BOOKS = (
    "122279",
    "Welcome to the Ruin Salvation Gallery",
    "master swordman stream_unknown author",
    "[312577] 전생하자마자 뚜드려맞는 먼치킨 마녀 겸 용사가 전부 다 배신하고 죽인다",
    "What is rest defence",
)
#: A real JSON-list glossary (an editor backup of book 344608).
REAL_JSON_LIST = ("344608", "Backups/344608_glossary_before_save_20260531_094939.json")


#: U6 base: GlossaryManager_GUI.py at this commit is the frozen source of every moved body.
U6_BASE_SHA = "e28e3a0f8e2ef3bd2aea59c87049b336ea469ac0"

SELF_TO_DOC = ("re", r"\bself\b", "doc")
SELF_TO_OWNER = ("re", r"\bself\b", "owner")

#: name -> (first line, last line, strip, add, edits). Lines are 1-based in the frozen
#: GlossaryManager_GUI.py; ``strip`` columns of indentation are removed and ``add`` added;
#: whitespace-only lines become empty. ``edits`` are applied in order: (old, new) plain
#: replacements (``old`` must occur) or ("re", pattern, repl) regex substitutions.
MOVED_BLOCKS = {
    # ---- module-level helpers (re-exported by GlossaryManager_GUI under the old names)
    "gender_resolution_summary": (91, 116, 0, 0, [
        ("def _gender_resolution_summary(", "def gender_resolution_summary("),
    ]),
    "collect_glossary_filter_values": (119, 131, 0, 0, [
        ("def _collect_glossary_filter_values(", "def collect_glossary_filter_values("),
    ]),
    "load_editor_gender_tracker": (134, 146, 0, 0, [
        ("def _load_editor_gender_tracker(", "def load_editor_gender_tracker("),
    ]),
    "editor_entry_has_gender": (149, 156, 0, 0, [
        ("def _editor_entry_has_gender(", "def editor_entry_has_gender("),
    ]),
    "prepare_editor_gender_tracking": (159, 177, 0, 0, [
        ("def _prepare_editor_gender_tracking(", "def prepare_editor_gender_tracking("),
        ("_load_editor_gender_tracker(glossary_path)", "load_editor_gender_tracker(glossary_path)"),
        ("_editor_entry_has_gender(entry, custom_types)", "editor_entry_has_gender(entry, custom_types)"),
    ]),
    # ---- GlossaryManagerMixin methods (the mixin keeps thin wrappers)
    "display_glossary_path": (1032, 1045, 4, 0, [
        ("def _display_glossary_path(path, roots=3):", "def display_glossary_path(path, roots=3):"),
    ]),
    "unified_glossary_editor_files": (1047, 1089, 4, 0, [
        ("def _unified_glossary_editor_files(self):",
         "def unified_glossary_editor_files(config, base_dir='', folder_key=None, module_dir=_MODULE_DIR):"),
        ("os.environ.get('OUTPUT_DIRECTORY') or self.config.get('output_directory')",
         "os.environ.get('OUTPUT_DIRECTORY') or config.get('output_directory')"),
        ("os.path.join(str(getattr(self, 'base_dir', '') or ''), 'Glossary'),",
         "os.path.join(str(base_dir or ''), 'Glossary'),"),
        ("os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Glossary'),",
         "os.path.join(module_dir, 'Glossary'),"),
        ("os.environ.get('UNIFIED_GLOSSARY_RESOLVED_KEY', '') or self._unified_glossary_folder_key()",
         "os.environ.get('UNIFIED_GLOSSARY_RESOLVED_KEY', '')\n"
         "        or (folder_key() if folder_key is not None else unified_glossary_folder_key(config))"),
    ]),
    "unified_glossary_folder_key": (1800, 1810, 4, 0, [
        ("def _unified_glossary_folder_key(self):", "def unified_glossary_folder_key(config):"),
        ("self.config.get(", "config.get("),
    ]),
    "glossary_type_count_summary": (1121, 1185, 4, 0, [
        ("def _glossary_type_count_summary(self, entries, max_custom_types=7):",
         "def glossary_type_count_summary(entries, configured_types=None, max_custom_types=7):"),
        ("    configured_types = (\n"
         "        getattr(self, 'custom_entry_types', None)\n"
         "        or getattr(self, 'config', {}).get('custom_entry_types', {})\n"
         "        or {}\n"
         "    )\n",
         "    configured_types = configured_types or {}\n"),
    ]),
    "parse_token_glossary": (1187, 1367, 4, 0, [
        ("def _parse_editor_token_glossary_async(self, lines):",
         "def parse_token_glossary(lines, custom_glossary_fields=(), custom_entry_types=None):"),
        ("default_extra_columns.extend(self.config.get('custom_glossary_fields', []))",
         "default_extra_columns.extend(custom_glossary_fields)"),
        ("custom_types = getattr(self, 'custom_entry_types', {}) or {",
         "custom_types = custom_entry_types or {"),
    ]),
    "parse_glossary_file": (1369, 1555, 4, 0, [
        ("def _parse_glossary_file_for_editor_async(self, path):",
         "def parse_glossary_file(path, config, custom_entry_types=None):"),
        ("entries, sections = self._parse_editor_token_glossary_async(lines)",
         "entries, sections = parse_token_glossary(\n"
         "                lines, config.get('custom_glossary_fields', []), custom_entry_types\n"
         "            )"),
        ("custom_types = getattr(self, 'custom_entry_types', None) or self.config.get('custom_entry_types', {})",
         "custom_types = custom_entry_types or config.get('custom_entry_types', {})"),
        ("_prepare_editor_gender_tracking(", "prepare_editor_gender_tracking("),
        ("tracking_disabled=bool(self.config.get(", "tracking_disabled=bool(config.get("),
        ("stats.append(self._glossary_type_count_summary(entries))",
         "stats.append(glossary_type_count_summary(\n"
         "            entries, configured_entry_types(config, custom_entry_types)\n"
         "        ))"),
    ]),
    # ---- inner functions of the _setup_glossary_editor_tab closure
    "reset_document": (991, 996, 8, 4, [SELF_TO_DOC]),
    "editor_gender_settings": (7818, 7829, 8, 0, [
        ("def _gender_settings():", "def editor_gender_settings(owner, config):"),
        ("        self,\n", "        owner,\n"),
        ("self.config.get(", "config.get("),
    ]),
    "tracker_entry_for_entry": (7831, 7837, 8, 0, [
        ("def _gender_tracker_entry(entry):", "def tracker_entry_for_entry(entry, tracker_data):"),
        ("getattr(self, 'current_gender_tracker_data', None),", "tracker_data,"),
    ]),
    "entry_gender_status": (7839, 7849, 8, 0, [
        ("def _gender_status(entry):", "def entry_gender_status(entry, tracker_data, settings):"),
        ("tracker_entry = _gender_tracker_entry(entry)", "tracker_entry = tracker_entry_for_entry(entry, tracker_data)"),
        ("threshold, bias = _gender_settings()", "threshold, bias = settings()"),
    ]),
    "baseline_translated": (7867, 7879, 8, 0, [
        ("def get_baseline_translated(item, col_key):",
         "def baseline_translated(original_translated_map, fmt, ref, col_key):"),
        ("self.current_glossary_format", "fmt"),
        ("idx = int(item.data(0, Qt.UserRole))", "idx = int(ref)"),
        ("key = item.data(0, Qt.UserRole)", "key = ref"),
        ("self._original_translated_map", "original_translated_map"),
    ]),
    "editor_row_specs": (7889, 7908, 8, 0, [
        ("def _editor_data_row_specs():", "def editor_row_specs(column_fields, data, fmt):"),
        ("fields = list(getattr(self, 'glossary_column_fields', []) or [])", "fields = list(column_fields or [])"),
        ("self.current_glossary_format", "fmt"),
        ("enumerate(self.current_glossary_data or [])", "enumerate(data or [])"),
        ("data = self.current_glossary_data or {}", "data = data or {}"),
    ]),
    "editor_display_value": (7910, 7919, 8, 0, [
        ("def _editor_display_value(entry, field):", "def editor_display_value(entry, field):"),
    ]),
    "row_matches_column_filters": (7944, 7952, 8, 0, [
        ("def _tree_item_matches_glossary_filters(item):",
         "def row_matches_column_filters(row_text, column_fields, filters):"),
        ("filters = _active_glossary_column_filters()", "filters = filters if isinstance(filters, dict) else {}"),
        ("fields = list(getattr(self, 'glossary_column_fields', []) or [])", "fields = list(column_fields or [])"),
        ("item.text(column)", "row_text(column)"),
    ]),
    "prune_column_filters": (8312, 8317, 12, 4, [
        ("self._glossary_column_filters = {", "return {"),
        ("_active_glossary_column_filters().items()", "(filters if isinstance(filters, dict) else {}).items()"),
    ]),
    "column_filter_values": (8024, 8030, 12, 4, [
        ("            self.glossary_tree.topLevelItem(row).text(column)\n"
         "            for row in range(self.glossary_tree.topLevelItemCount())\n",
         "            value\n"
         "            for value in texts\n"),
    ]),
    "loaded_stats_text": (8420, 8428, 8, 0, [
        ("def _loaded_glossary_stats_text(entries):",
         "def loaded_stats_text(entries, fmt, configured_types=None):"),
        ("self.current_glossary_format", "fmt"),
        ("stats.append(self._glossary_type_count_summary(entries))",
         "stats.append(glossary_type_count_summary(entries, configured_types))"),
    ]),
    "apply_parse_payload": (8462, 8480, 12, 4, [SELF_TO_DOC]),
    "reparse_glossary_file.parse": (8768, 8947, 15, 4, [SELF_TO_DOC]),
    "reparse_glossary_file.stats": (8952, 8963, 15, 4, [
        SELF_TO_DOC,
        ("stats.append(doc._glossary_type_count_summary(entries))",
         "stats.append(glossary_type_count_summary(\n"
         "            entries, configured_entry_types(config, custom_entry_types)\n"
         "        ))"),
    ]),
    "save_document.head": (9037, 9038, 11, 4, [SELF_TO_DOC]),
    "save_document.resolve": (9040, 9061, 15, 4, [SELF_TO_DOC]),
    "write_token_csv": (9063, 9131, 27, 4, [
        ("sections = getattr(self, 'current_glossary_sections', []) or []", "sections = sections or []"),
        ("custom_fields = self.config.get('custom_glossary_fields', [])", "custom_fields = custom_glossary_fields"),
    ]),
    "save_document.tail": (9134, 9164, 15, 4, [SELF_TO_DOC]),
    "count_empty_fields": (9176, 9185, 16, 4, [("self.current_glossary_data", "data")]),
    "remove_empty_fields": (9197, 9203, 16, 4, [("self.current_glossary_data", "data")]),
    "clean_empty_fields_message": (9209, 9212, 20, 4, []),
    "delete_entries": (9234, 9256, 16, 4, [
        ("for item in selected:", "for ref in refs:"),
        ("int(item.data(0, Qt.UserRole))", "int(ref)"),
        ("key = item.data(0, Qt.UserRole)", "key = ref"),
        SELF_TO_DOC,
    ]),
    "remove_duplicate_entries.env": (9270, 9273, 20, 4, [
        ("self.config.get('glossary_disable_honorifics_filter', False)", "disable_honorifics_filter"),
    ]),
    "remove_duplicate_entries.call": (9276, 9280, 20, 4, [
        ("self.current_glossary_data = skip_duplicate_entries(", "return skip_duplicate_entries("),
        ("    self.current_glossary_data,\n", "    data,\n"),
        ("glossary_path=self.editor_file_entry.text(),", "glossary_path=glossary_path,"),
        ("gender_tracker=getattr(self, 'current_gender_tracker_data', None),", "gender_tracker=gender_tracker,"),
    ]),
    "remove_duplicate_entries_fallback": (9296, 9306, 20, 4, [("self.current_glossary_data", "data")]),
    "duplicate_detection_info_text": (9322, 9326, 16, 4, []),
    "backup_settings_message": (9457, 9463, 16, 4, [
        ("backup_checkbox.isChecked()", "enabled"),
        ("limit = max_backups_spinbox.value()", "limit = max_backups"),
    ]),
    "document_entry_count": (9537, 9537, 12, 4, [
        ("entry_count = len(", "return len("),
        ("self.current_glossary_data", "data"),
        ("self.current_glossary_format", "fmt"),
    ]),
    "trim_preview_text": (9582, 9585, 20, 4, []),
    "trim_entries": (9616, 9625, 20, 4, [SELF_TO_DOC]),
    "is_new_format_data": (9712, 9714, 12, 4, [
        ("is_new_format = (", "return ("),
        ("self.current_glossary_format", "fmt"),
        ("self.current_glossary_data", "data"),
    ]),
    "filter_entry_types": (9732, 9746, 16, 4, [
        ("self.config.get(", "config.get("),
        ("self.current_glossary_data", "data"),
    ]),
    "parse_type_limit": (9890, 9896, 16, 4, [
        ("text = edit.text().strip()", "text = text.strip()"),
    ]),
    "entry_matches_filter": (9899, 9937, 16, 4, [
        ("type_check = type_checks.get(entry['type'])", "type_check = kept_types.get(entry['type'])"),
        ("if type_check and not type_check.isChecked():", "if type_check is not None and not type_check:"),
        ("search_text = search_entry.text().strip().lower()", "search_text = search_text.strip().lower()"),
        ("_custom_types = self.config.get(", "_custom_types = config.get("),
        ("limit = get_type_limit(_t)", "limit = parse_type_limit(type_limits.get(_t))"),
    ]),
    "count_filter_matches": (9948, 9958, 16, 4, [
        ("self.current_glossary_format", "fmt"),
        ("self.current_glossary_data", "data"),
        ("check_entry_matches(", "matches("),
    ]),
    "filter_list_entries": (9984, 9988, 20, 4, [
        ("self.current_glossary_data", "data"),
        ("check_entry_matches(", "matches("),
    ]),
    "export_selected_entries": (10044, 10089, 15, 4, [
        ("self.current_glossary_format", "fmt"),
        ("for item in selected:", "for ref in refs:"),
        ("idx = int(item.data(0, Qt.UserRole))", "idx = int(ref)"),
        ("key = item.data(0, Qt.UserRole)", "key = ref"),
        ("self.current_glossary_data", "data"),
        ("self.config.get(", "config.get("),
    ]),
    "collect_translated_changes": (10097, 10111, 12, 4, [
        ("self.current_glossary_format", "fmt"),
        ("self.current_glossary_data", "data"),
        ("self._original_translated_map", "original_translated_map"),
    ]),
    "configured_update_workers": (10356, 10374, 16, 4, [
        ("    enabled = bool(\n"
         "        getattr(\n"
         "            self,\n"
         "            'enable_parallel_extraction_var',\n"
         "            self.config.get('enable_parallel_extraction', False),\n"
         "        )\n"
         "    )\n",
         "    enabled = bool(enable_parallel)\n"),
        ("    raw_workers = getattr(\n"
         "        self,\n"
         "        'extraction_workers_var',\n"
         "        self.config.get('extraction_workers', os.environ.get('EXTRACTION_WORKERS', 1)),\n"
         "    )\n",
         ""),
    ]),
    "update_output_files.head": (10290, 10353, 12, 4, [
        ("    glossary_path = self.editor_file_entry.text()\n", ""),
        ("self.append_log(", "log("),
    ]),
    "update_output_files.tail": (10376, 10494, 12, 4, [
        ("workers = _configured_update_workers()", "workers = configured_update_workers(enable_parallel, raw_workers)"),
        ("self.append_log(", "log("),
    ]),
    "update_output_files_prompt": (10499, 10499, 15, 4, []),
    "saved_translated_baseline": (10523, 10528, 15, 4, [
        ("self._original_translated_map = {", "return {"),
        ("self._original_translated_map = dict(", "return dict("),
        ("self.current_glossary_format", "fmt"),
        ("self.current_glossary_data", "data"),
    ]),
    "save_glossary_as": (10570, 10595, 15, 4, [
        ("self.current_glossary_format", "fmt"),
        ("self.config.get(", "config.get("),
        ("self.current_glossary_data", "data"),
    ]),
    "resolve_editor_glossaries": (10683, 10828, 16, 4, [
        ("        for _attr in ('auto_loaded_glossary_path', 'manual_glossary_path'):\n"
         "            _p = getattr(self, _attr, None)\n",
         "        for _p in (auto_loaded_glossary_path, manual_glossary_path):\n"),
        ("_mmap = getattr(self, 'manual_glossary_map', None) or {}", "_mmap = manual_glossary_map or {}"),
        ("self.config.get(", "config.get("),
        ("auto_path = getattr(self, 'auto_loaded_glossary_path', None)", "auto_path = auto_loaded_glossary_path"),
        ("manual_path = getattr(self, 'manual_glossary_path', None)", "manual_path = manual_glossary_path"),
        ("                output_base_getter = getattr(self, '_get_output_base_dir', None)\n", ""),
        ("os.path.join(str(getattr(self, 'base_dir', '') or ''), 'Glossary'),",
         "os.path.join(str(base_dir or ''), 'Glossary'),"),
        ("os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Glossary'),",
         "os.path.join(module_dir, 'Glossary'),"),
        ("display = self._display_glossary_path(cand)", "display = display_glossary_path(cand)"),
    ]),
    "trim_undo_stack": (10890, 10891, 12, 4, [
        ("self._undo_stack", "undo_stack"),
        ("self._undo_max", "undo_max"),
    ]),
    "push_undo_snapshot": (10895, 10905, 12, 4, [
        ("    if self.current_glossary_data is None:\n        return\n",
         "    if self.current_glossary_data is None:\n        return False\n"),
        ("self._undo_stack.append(snap)", "undo_stack.append(snap)"),
        ("_trim_undo_stack()", "trim_undo_stack(undo_stack, undo_max)"),
        ("self._redo_stack.clear()", "redo_stack.clear()"),
        SELF_TO_DOC,
    ]),
    "push_html_undo_snapshot": (10916, 10920, 12, 4, [
        ("    if not changes:\n        return\n", "    if not changes:\n        return False\n"),
        ("self._undo_stack.append(", "undo_stack.append("),
        ("_trim_undo_stack()", "trim_undo_stack(undo_stack, undo_max)"),
        ("self._redo_stack.clear()", "redo_stack.clear()"),
    ]),
    "undo_entry_kind": (10924, 10924, 12, 4, []),
    "undo_step.html": (10927, 10933, 12, 4, [
        ("    if not self._undo_stack:\n        return\n", "    if not undo_stack:\n        return None, None\n"),
        ("entry = self._undo_stack.pop()", "entry = undo_stack.pop()"),
        ("if _entry_kind(entry) == 'html':", "if undo_entry_kind(entry) == 'html':"),
        ("self._redo_stack.append(entry)", "redo_stack.append(entry)"),
    ]),
    "undo_step.glossary": (10941, 10963, 12, 4, [
        ("self._undo_stack.append(entry)\n        return\n", "undo_stack.append(entry)\n        return 'blocked', None\n"),
        ("self._redo_stack.append({", "redo_stack.append({"),
        SELF_TO_DOC,
    ]),
    "redo_step.html": (10968, 10973, 12, 4, [
        ("    if not self._redo_stack:\n        return\n", "    if not redo_stack:\n        return None, None\n"),
        ("entry = self._redo_stack.pop()", "entry = redo_stack.pop()"),
        ("if _entry_kind(entry) == 'html':", "if undo_entry_kind(entry) == 'html':"),
        ("self._undo_stack.append(entry)", "undo_stack.append(entry)"),
    ]),
    "redo_step.glossary": (10981, 11001, 12, 4, [
        ("self._redo_stack.append(entry)\n        return\n", "redo_stack.append(entry)\n        return 'blocked', None\n"),
        ("self._undo_stack.append({", "undo_stack.append({"),
        SELF_TO_DOC,
    ]),
    "editor_entry_line_number": (11096, 11128, 12, 0, [
        ("user_data = selected.data(0, Qt.UserRole)", "user_data = row.ref()"),
        ("selected.columnCount()", "row.columnCount()"),
        ("selected.text(col_idx)", "row.text(col_idx)"),
    ]),
    "entry_for_ref": (11165, 11173, 12, 4, [
        ("self.current_glossary_format", "fmt"),
        ("source_index = int(item.data(0, Qt.UserRole))", "source_index = int(ref)"),
        ("self.current_glossary_data", "data"),
    ]),
    "can_resolve_gender": (11177, 11177, 12, 4, []),
    "format_tracker_location": (11179, 11189, 8, 0, [
        ("def _format_tracker_location(occurrence):", "def format_tracker_location(occurrence):"),
    ]),
    "current_gender_decision": (11208, 11210, 12, 4, []),
    "gender_settings_labels": (11235, 11238, 12, 4, []),
    "gender_history_line": (11251, 11258, 16, 4, [
        ("history_layout.addWidget(QLabel(\n", "return (\n"),
        ("_format_tracker_location(", "format_tracker_location("),
        ("\n    ))", "\n    )"),
    ]),
    "gender_flip_line": (11264, 11266, 20, 4, [
        ("latest_lines.append(f\"• {source} → {target} · {_format_tracker_location(change)}\")",
         "return f\"• {source} → {target} · {format_tracker_location(change)}\""),
    ]),
    "apply_gender_decision": (11301, 11306, 16, 4, [SELF_TO_DOC]),
    "editor_associated_source_path": (11373, 11415, 12, 4, [
        ("        mapped = (\n"
         "            getattr(self, '_editor_glossary_source_map', None)\n"
         "            or getattr(self, '_editor_glossary_epub_map', {})\n"
         "            or {}\n"
         "        )\n",
         "        mapped = source_map or {}\n"),
        ("for path in self._glossary_editor_input_sources()", "for path in input_sources()"),
        ("getter = getattr(self, getter_name, None)", "getter = epub_getter"),
        ("path = getattr(self, attr_name, None)", "path = file_path"),
        ("path = self.config.get('last_epub_path') if hasattr(self, 'config') else None",
         "path = config.get('last_epub_path') if config is not None else None"),
    ]),
    "editor_usage_entries": (11418, 11442, 12, 4, [
        ("    fields, specs = _editor_data_row_specs()\n", ""),
    ]),
    "editor_output_dir_from_source": (11445, 11483, 12, 4, [
        ("        resolver = getattr(self, '_resolve_open_output_folder_for_file', None)\n", ""),
        ("self.config.get('output_directory') if hasattr(self, 'config') else None,",
         "config.get('output_directory') if config is not None else None,"),
        ("        base_dir_getter = getattr(self, '_get_output_base_dir', None)\n", ""),
    ]),
    "editor_output_dir_from_glossary": (11486, 11527, 12, 4, [
        ("        mapped = (\n"
         "            getattr(self, '_editor_glossary_source_map', None)\n"
         "            or getattr(self, '_editor_glossary_epub_map', {})\n"
         "            or {}\n"
         "        )\n",
         "        mapped = source_map or {}\n"),
        ("resolved = _editor_output_dir_from_source(source_path)", "resolved = from_source(source_path)"),
    ]),
    "translated_output_files": (11537, 11562, 8, 0, [
        ("def _editor_translated_output_files(output_dir):", "def translated_output_files(output_dir):"),
    ]),
    "read_translated_output_texts": (11564, 11587, 8, 0, [
        ("def _read_translated_output_texts(paths, progress_callback=None):",
         "def read_translated_output_texts(paths, progress_callback=None):"),
    ]),
    "compute_used_rows": (11691, 11769, 16, 4, [
        ("_editor_translated_output_files(output_dir)", "translated_output_files(output_dir)"),
        ("_read_translated_output_texts(", "read_translated_output_texts("),
    ]),
    "find_next_index": (11811, 11817, 16, 4, [
        ("start = (getattr(self, \"_last_find_pos\", -1) + 1) % total", "start = (last_find_pos + 1) % total"),
        ("    item = self.glossary_tree.topLevelItem(idx)\n"
         "        cols = [item.text(c) for c in range(item.columnCount())]\n",
         "    cols = row_texts(idx)\n"),
    ]),
    "replace_in_row": (11833, 11875, 16, 4, [
        ("item.columnCount()", "row.columnCount()"),
        ("before = item.text(col_idx)", "before = row.text(col_idx)"),
        ("item.setText(col_idx, after)", "row.setText(col_idx, after)"),
        ("col_key = self.glossary_column_fields[col_idx - 1] if self.glossary_column_fields else None",
         "col_key = column_fields[col_idx - 1] if column_fields else None"),
        ("row_idx = int(item.data(0, Qt.UserRole))", "row_idx = int(row.ref())"),
        ("key = item.data(0, Qt.UserRole)", "key = row.ref()"),
        ("item.setData(0, Qt.UserRole, new_key)", "row.set_ref(new_key)"),
        ("        update_row_highlight(item, col_key, after)\n",
         "        if on_replaced is not None:\n            on_replaced(col_key, after)\n"),
        SELF_TO_DOC,
    ]),
    "row_has_match": (11882, 11885, 16, 4, [
        ("will_change = bool(find_text) and item is not None and any(",
         "return bool(find_text) and row is not None and any("),
        ("item.text(c)", "row.text(c)"),
        ("item.columnCount()", "row.columnCount()"),
    ]),
    "resolve_epub_output_dir": (12077, 12128, 12, 4, [
        ("    if hasattr(self, 'config'):\n            _candidates.append(self.config.get('output_directory'))\n",
         "    if config is not None:\n            _candidates.append(config.get('output_directory'))\n"),
        ("self.append_log(", "log("),
    ]),
    "manual_only_epub_path": (12275, 12290, 20, 4, [
        ("selected = list(getattr(self, 'selected_files', []) or [])", "selected = list(selected_files or [])"),
        ("_last = self.config.get('last_epub_path') if hasattr(self, 'config') else None",
         "_last = config.get('last_epub_path') if config is not None else None"),
    ]),
    "copy_glossary_to_epub_output": (12296, 12307, 24, 4, [
        ("out_dir, _base = _resolve_epub_output_dir(epub_path)",
         "out_dir, _base = resolve_epub_output_dir(epub_path, config, log)"),
        ("self.append_log(", "log("),
    ]),
    "reload_flash_rows": (12467, 12528, 20, 4, [
        ("new_count = self.glossary_tree.topLevelItemCount()", "new_count = len(new_row_keys)"),
        ("    item = self.glossary_tree.topLevelItem(i)\n"
         "        if item:\n"
         "            new_content[_row_key(item)] = i\n",
         "    key = new_row_keys[i]\n"
         "        if key is not None:\n"
         "            new_content[key] = i\n"),
    ]),
    "normalize_edit_value": (12678, 12679, 11, 4, []),
    "apply_entry_edit": (12685, 12713, 11, 4, [
        ("row_idx = int(item.data(0, Qt.UserRole))", "row_idx = int(ref)"),
        ("key = item.data(0, Qt.UserRole)", "key = ref"),
        ("item.setData(0, Qt.UserRole, new_key)", "new_ref, ref_changed = new_key, True"),
        SELF_TO_DOC,
    ]),
    "default_convert_path": (12804, 12808, 8, 4, [
        ("self.config.get(", "config.get("),
    ]),
    "convert_to_csv": (12822, 12982, 12, 4, [
        ("        QMessageBox.critical(self.dialog, \"Error\", \"No entries to export\")\n        return\n",
         "        return None\n"),
        ("self.config.get(", "config.get("),
        SELF_TO_DOC,
    ]),
    # ---- Glossary Manager prompt profiles: config state (the mixin keeps wrappers)
    "glossary_prompt_profile_meta": (457, 475, 4, 0, [
        ("def _glossary_prompt_profile_meta(self, profile_key):", "def glossary_prompt_profile_meta(profile_key):"),
    ]),
    "is_default_glossary_prompt_profile": (477, 478, 4, 0, [
        ("def _is_default_glossary_prompt_profile(self, profile_name):",
         "def is_default_glossary_prompt_profile(profile_name):"),
        ("self.GLOSSARY_PROMPT_DEFAULT_PROFILE", "GLOSSARY_PROMPT_DEFAULT_PROFILE"),
    ]),
    "ensure_glossary_prompt_profiles": (480, 507, 4, 0, [
        ("def _ensure_glossary_prompt_profiles(self):", "def ensure_glossary_prompt_profiles(owner):"),
        ("self._glossary_prompt_profile_meta(key)", "glossary_prompt_profile_meta(key)"),
        SELF_TO_OWNER,
    ]),
    "glossary_prompt_profiles_for": (509, 511, 4, 0, [
        ("def _glossary_prompt_profiles_for(self, profile_key):", "def glossary_prompt_profiles_for(owner, profile_key):"),
        ("self._ensure_glossary_prompt_profiles()", "ensure_glossary_prompt_profiles(owner)"),
    ]),
    "active_glossary_prompt_profile_for": (513, 515, 4, 0, [
        ("def _active_glossary_prompt_profile_for(self, profile_key):",
         "def active_glossary_prompt_profile_for(owner, profile_key):"),
        ("self._ensure_glossary_prompt_profiles()", "ensure_glossary_prompt_profiles(owner)"),
    ]),
    "set_active_glossary_prompt_profile": (517, 523, 4, 0, [
        ("def _set_active_glossary_prompt_profile(self, profile_key, profile_name):",
         "def set_active_glossary_prompt_profile(owner, profile_key, profile_name):"),
        ("self._ensure_glossary_prompt_profiles()", "ensure_glossary_prompt_profiles(owner)"),
        ("self._is_default_glossary_prompt_profile(profile_name)", "is_default_glossary_prompt_profile(profile_name)"),
        SELF_TO_OWNER,
    ]),
    "default_glossary_prompt_profile_text": (525, 528, 4, 0, [
        ("def _default_glossary_prompt_profile_text(self, profile_key):",
         "def default_glossary_prompt_profile_text(owner, profile_key):"),
        ("self._ensure_glossary_prompt_profiles()", "ensure_glossary_prompt_profiles(owner)"),
        SELF_TO_OWNER,
    ]),
    "set_default_glossary_prompt_profile_text": (530, 534, 4, 0, [
        ("def _set_default_glossary_prompt_profile_text(self, profile_key, text):",
         "def set_default_glossary_prompt_profile_text(owner, profile_key, text):"),
        ("self._ensure_glossary_prompt_profiles()", "ensure_glossary_prompt_profiles(owner)"),
        SELF_TO_OWNER,
    ]),
    "store_glossary_prompt_current": (563, 568, 4, 0, [
        ("meta = self._glossary_prompt_profile_meta(profile_key)", "meta = glossary_prompt_profile_meta(profile_key)"),
        SELF_TO_OWNER,
    ]),
    "default_glossary_refinement_system_prompt": (4144, 4149, 4, 0, [
        ("def _default_glossary_refinement_system_prompt(self):", "def default_glossary_refinement_system_prompt():"),
    ]),
    "default_glossary_refinement_user_prompt": (4151, 4156, 4, 0, [
        ("def _default_glossary_refinement_user_prompt(self):", "def default_glossary_refinement_user_prompt():"),
    ]),
    # ---- Unified Glossary "Rebuild Now": shared folder + settings snapshot (Integrate U6; the
    # mobile unified_glossary job uses both). translator_gui's _get_app_dir is app_paths'.
    "unified_glossary_shared_dir": (5800, 5809, 4, 0, [
        ("def _unified_glossary_shared_dir(self):", "def unified_glossary_shared_dir(config):"),
        ("self.config.get('output_directory')", "config.get('output_directory')"),
        ("from translator_gui import _get_app_dir", "from app_paths import _get_app_dir"),
    ]),
    "unified_rebuild_settings": (5854, 5859, 4, 0, [
        ("re", r"self\.config\.get\(", "config.get("),
    ]),
}


#: GlossaryManager_GUI.py differs from the frozen file only in these spans:
#: (first frozen line, last frozen line, number of lines that replaced them).
REWIRE_SPANS = (
    (47, 47, 10), (91, 178, 4), (276, 277, 27), (457, 475, 3), (477, 478, 2), (480, 507, 3),
    (509, 511, 2), (513, 515, 2), (517, 523, 2), (525, 528, 2), (530, 534, 2), (563, 568, 1),
    (991, 996, 1), (1031, 1045, 2), (1047, 1089, 8), (1121, 1185, 13), (1187, 1367, 7), (1369, 1555, 5),
    (1800, 1810, 3), (4144, 4149, 2), (4151, 4156, 2), (5802, 5809, 1), (5854, 5859, 1),
    (7818, 7849, 12), (7867, 7879, 7),
    (7889, 7919, 8),
    (7944, 7952, 6), (8024, 8030, 4), (8312, 8317, 3), (8420, 8428, 4), (8462, 8480, 1), (8566, 8966, 9),
    (9035, 9167, 9), (9175, 9185, 4), (9196, 9203, 2), (9208, 9212, 2), (9234, 9256, 3), (9270, 9280, 8),
    (9296, 9306, 3), (9321, 9326, 2), (9422, 9422, 1), (9457, 9463, 3), (9537, 9537, 3), (9582, 9585, 1),
    (9616, 9625, 1), (9701, 9701, 3), (9712, 9714, 3), (9732, 9746, 1), (9885, 9937, 12), (9948, 9958, 3),
    (9984, 9988, 1), (10044, 10089, 7), (10096, 10111, 5), (10289, 10494, 17), (10499, 10503, 5), (10523, 10528, 5),
    (10570, 10595, 3), (10683, 10828, 13), (10887, 10892, 0), (10895, 10906, 2), (10916, 10925, 3), (10927, 10933, 2),
    (10941, 10963, 2), (10968, 10973, 2), (10981, 11001, 2), (11018, 11018, 1), (11096, 11128, 1), (11165, 11173, 3),
    (11177, 11190, 2), (11208, 11210, 1), (11235, 11238, 1), (11251, 11258, 3), (11264, 11266, 1), (11301, 11306, 1),
    (11371, 11415, 13), (11417, 11442, 3), (11444, 11483, 7), (11485, 11527, 10), (11537, 11588, 0), (11691, 11769, 1),
    (11811, 11823, 17), (11833, 11875, 8), (11882, 11885, 3), (11927, 11928, 2), (12072, 12129, 3), (12269, 12269, 0),
    (12275, 12290, 4), (12296, 12307, 6), (12467, 12528, 8), (12677, 12679, 1), (12685, 12713, 5), (12803, 12808, 2),
    (12822, 12984, 5),
)

#: Names GlossaryManager_GUI re-exports from glossary_document (tests import them from there).
REEXPORTS = {
    "_gender_resolution_summary": "gender_resolution_summary",
    "_collect_glossary_filter_values": "collect_glossary_filter_values",
    "_load_editor_gender_tracker": "load_editor_gender_tracker",
    "_editor_entry_has_gender": "editor_entry_has_gender",
    "_prepare_editor_gender_tracking": "prepare_editor_gender_tracking",
}


# =============================================================================================
# frozen source, closures, recorders
# =============================================================================================

_GIT_CACHE = {}


def _git_text(relpath, sha=U6_BASE_SHA):
    key = (relpath, sha)
    if key not in _GIT_CACHE:
        try:
            data = subprocess.run(["git", "show", f"{sha}:{relpath}"], cwd=str(REPO_ROOT),
                                  capture_output=True, check=True).stdout
        except (OSError, subprocess.CalledProcessError) as exc:
            pytest.skip(f"frozen source {relpath}@{sha[:8]} unavailable: {exc}")
        _GIT_CACHE[key] = data.decode("utf-8-sig").replace("\r\n", "\n")
    return _GIT_CACHE[key]


def _current_text(name):
    return (SRC / name).read_text(encoding="utf-8-sig").replace("\r\n", "\n")


def _apply_edits(text, edits, name):
    for edit in edits:
        if edit[0] == "re":
            _kind, pattern, repl = edit
            text = re.sub(pattern, repl, text)
        else:
            old, new = edit
            assert old in text, f"{name}: documented edit not found: {old!r}"
            text = text.replace(old, new)
    return text


def _legacy_block(name):
    start, end, strip, add, edits = MOVED_BLOCKS[name]
    out = []
    for line in _git_text("src/GlossaryManager_GUI.py").split("\n")[start - 1:end]:
        if not line.strip():
            out.append("")
            continue
        assert line.startswith(" " * strip), (name, start, line)
        out.append(" " * add + line[strip:])
    return _apply_edits("\n".join(out), edits, name)


_TREES = {}
_CODES = {}


def _tree_of(text):
    key = hash(text)
    if key not in _TREES:
        _TREES[key] = ast.parse(text)
    return _TREES[key]


def _find_scope(text, path, cls):
    node = next(n for n in _tree_of(text).body if isinstance(n, ast.ClassDef) and n.name == cls)
    for name in path:
        node = next(n for n in ast.walk(node)
                    if isinstance(n, ast.FunctionDef) and n.name == name and n is not node)
    return node


def _nested_source(text, *path, cls="GlossaryManagerMixin"):
    """Dedented source of a method (``path=(name,)``) or of a function nested in it."""
    node = _find_scope(text, path, cls)
    lines = text.split("\n")
    return textwrap.dedent("\n".join(lines[node.lineno - 1:node.end_lineno]))


def _alias_value(text, *path, cls="GlossaryManagerMixin"):
    """The expression of ``name = <expr>`` inside the enclosing function (a rewired alias), or None."""
    node = _find_scope(text, path[:-1], cls)
    for child in ast.walk(node):
        if (isinstance(child, ast.Assign) and len(child.targets) == 1
                and isinstance(child.targets[0], ast.Name) and child.targets[0].id == path[-1]):
            return ast.Expression(child.value)
    return None


def _compiled(text, path):
    """(kind, code) for an editor function: 'def' source code or an 'alias' expression."""
    key = (hash(text), path)
    if key not in _CODES:
        label = f"<{'.'.join(path)}>"
        try:
            _CODES[key] = ("def", compile(_nested_source(text, *path), label, "exec"))
        except StopIteration:
            alias = _alias_value(text, *path)
            assert alias is not None, f"{'.'.join(path)} is neither a function nor an alias"
            _CODES[key] = ("alias", compile(alias, label, "eval"))
    return _CODES[key]


def _closure(module, text, path, **free):
    """A nested editor function executed with its closure variables supplied as globals.

    A function the rewire replaced by an alias (``_name = glossary_document.name``) resolves to
    the alias target."""
    ns = dict(vars(module))
    ns.update(free)
    kind, code = _compiled(text, tuple(path))
    if kind == "alias":
        return eval(code, ns)
    exec(code, ns)
    return ns[path[-1]]


def _load_gm_module(text, name, file_path):
    module = types.ModuleType(name)
    module.__file__ = str(file_path)
    sys.modules[name] = module
    exec(compile(text, str(file_path), "exec"), module.__dict__)
    return module


@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def legacy_gm(qapp):
    return _load_gm_module(_git_text("src/GlossaryManager_GUI.py"), "_glossary_manager_u6_frozen",
                           SRC / "GlossaryManager_GUI.py")


@pytest.fixture(scope="module")
def new_gm(qapp):
    import GlossaryManager_GUI
    return GlossaryManager_GUI


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    """Never touch the real Library / output roots; restore the process env and cwd."""
    original = dict(os.environ)
    monkeypatch.setenv("GLOSSARION_LIBRARY_DIR", str(tmp_path / "_library"))
    for key in ("OUTPUT_DIRECTORY", "OUTPUT_DIR", "EPUB_PATH", "GLOSSARY_SHARED_DIR",
                "UNIFIED_GLOSSARY_RESOLVED_KEY", "GLOSSARY_SKIP_GENDER_TRACKING", "EXTRACTION_WORKERS"):
        monkeypatch.delenv(key, raising=False)
    cwd = os.getcwd()
    yield
    # monkeypatch.undo() first: its records may hold values a test set directly (the stop
    # protocol writes TRANSLATION_CANCELLED etc.), which its own teardown would re-apply after
    # this one and leak into later tests' subprocesses; then the environment from before the test.
    monkeypatch.undo()
    os.chdir(cwd)
    os.environ.clear()
    os.environ.update(original)


def _run(fn, *args, **kwargs):
    try:
        return ("ok", fn(*args, **kwargs))
    except Exception as exc:  # the frozen code may raise; the new code must raise the same
        return ("raise", type(exc).__name__, str(exc))


#: The gender tracker stamps every write (extract_glossary_from_epub._write_gender_tracker).
_UPDATED_AT = re.compile(r'(updated_at\\*"\s*:\s*\\*")\d{4}-\d\d-\d\d \d\d:\d\d:\d\d')


def _norm(obj, *roots):
    """JSON-comparable copy with the given folders replaced by <ROOT> (tracker stamps masked)."""
    text = json.dumps(obj, default=repr, ensure_ascii=False, sort_keys=False)
    for root in roots:
        raw = str(root)
        for variant in sorted({raw, raw.replace("\\", "\\\\"), raw.replace("\\", "/"),
                               raw.replace("\\", "\\\\\\\\")}, key=len, reverse=True):
            text = text.replace(variant, "<ROOT>")
    return json.loads(_UPDATED_AT.sub(r"\1<STAMP>", text))


def _tree_bytes(root):
    out = {}
    root = Path(root)
    if not root.exists():
        return out
    for path in sorted(root.rglob("*")):
        if path.is_file():
            out[path.relative_to(root).as_posix()] = path.read_bytes()
    return out


class FakeEntry:
    """``editor_file_entry`` (the combo shim) / a QLineEdit: ``text()`` / ``setText()``."""

    def __init__(self, text=""):
        self._text = text

    def text(self):
        return self._text

    def setText(self, text):
        self._text = text


class FakeCheck:
    def __init__(self, checked):
        self._checked = bool(checked)

    def isChecked(self):
        return self._checked


class FakeItem:
    """A glossary tree item: column texts, the UserRole source ref, recorded backgrounds."""

    def __init__(self, texts, ref):
        self.texts = list(texts)
        self.ref_value = ref
        self.hidden = False

    def columnCount(self):
        return len(self.texts)

    def text(self, column):
        return self.texts[column] if 0 <= column < len(self.texts) else ""

    def setText(self, column, value):
        self.texts[column] = value

    def data(self, column, role):
        from PySide6.QtCore import Qt
        return self.ref_value if (column, role) == (0, Qt.UserRole) else None

    def setData(self, column, role, value):
        from PySide6.QtCore import Qt
        if (column, role) == (0, Qt.UserRole):
            self.ref_value = value

    def state(self):
        return [list(self.texts), self.ref_value]


class FakeTree:
    def __init__(self, items, current=None):
        self.items = items
        self.current = current

    def topLevelItemCount(self):
        return len(self.items)

    def topLevelItem(self, index):
        return self.items[index] if 0 <= index < len(self.items) else None

    def currentItem(self):
        return self.current

    def setCurrentItem(self, item):
        self.current = item

    def scrollToItem(self, item):
        pass


class Recorder:
    def __init__(self, result=None):
        self.calls = []
        self.result = result

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.result(*args, **kwargs) if callable(self.result) else self.result


class BoxRecorder:
    """Stands in for QMessageBox inside an extracted closure (static boxes and instances)."""

    Yes, No, Ok, Cancel = 16384, 65536, 1024, 4194304
    Warning = Question = Information = Critical = 0

    def __init__(self, answers=()):
        self.log = []
        self.answers = list(answers)
        outer = self

        class _Box:
            def __init__(self, *args, **kwargs):
                self.fields = {}

            def __getattr__(self, name):
                def setter(*args):
                    self.fields[name] = args
                return setter

            def exec(self):
                answer = outer.answers.pop(0) if outer.answers else outer.No
                outer.log.append(("exec", sorted((k, [str(a) for a in v]) for k, v in self.fields.items()), answer))
                return answer

        self._box = _Box

    def __call__(self, *args, **kwargs):
        return self._box(*args, **kwargs)

    def _static(self, kind):
        def show(*args, **kwargs):
            self.log.append((kind, [a for a in args if isinstance(a, str)]))
            return self.Ok
        return show

    def __getattr__(self, name):
        if name in ("critical", "warning", "information", "question"):
            return self._static(name)
        raise AttributeError(name)


# =============================================================================================
# random inputs (seeded from the real glossaries)
# =============================================================================================

TYPES = ("character", "terms", "term", "titles", "locations", "surnames", "nicknames", "organization",
         "skills", "")
GENDERS = ("male", "Male", "female", "FEMALE", "unknown", "Unknown", "", "other")
EXTRA_FIELDS = ("description", "fun fact", "aliases", "rank", "notes")
TRICKY = ("Luna", "Kai Ren", "A (B)", "x: y", "[tag]", "a | b", "p, q", "r=s", "  pad  ", "", "Ünïcødé",
          "루나", "카이 (Kai)", "王", "(note: z)", "fun fact: f", "desc, rank: S", "=== X ===", "* z")


def _real_glossary_paths():
    out = []
    for book in REAL_BOOKS:
        folder = SRC / "Glossary" / book
        if folder.is_dir():
            out.extend(sorted(p for p in folder.glob("*.csv") if p.name.endswith("_glossary.csv")))
    return out


@pytest.fixture(scope="module")
def real_lines():
    lines = []
    for path in _real_glossary_paths():
        lines.extend(path.read_text(encoding="utf-8").splitlines(True))
    if not lines:
        lines = ["=== CHARACTERS ===\n", "* 루나 = Luna [female]: a witch\n", "* 카이 = Kai [male]\n"]
    return lines


@pytest.fixture(scope="module")
def real_entries(real_lines):
    entries, _sections = gd.parse_token_glossary(real_lines, ["description"], None)
    return entries or [{"type": "character", "raw_name": "루나", "translated_name": "Luna", "gender": "female"}]


def _rand_text(rng):
    return rng.choice(TRICKY) if rng.random() < 0.6 else "".join(
        rng.choice("abc XYZ가나다:()[]|,=-") for _ in range(rng.randint(0, 12)))


def _rand_entry(rng, real_entries):
    if rng.random() < 0.7:
        entry = copy.deepcopy(rng.choice(real_entries))
    else:
        entry = {"type": rng.choice(TYPES), "raw_name": _rand_text(rng), "translated_name": _rand_text(rng)}
    for _ in range(rng.randint(0, 3)):
        roll = rng.random()
        if roll < 0.25:
            entry["gender"] = rng.choice(GENDERS)
        elif roll < 0.45:
            entry[rng.choice(EXTRA_FIELDS)] = _rand_text(rng)
        elif roll < 0.55:
            entry.pop(rng.choice(list(entry) or ["x"]), None)
        elif roll < 0.65:
            entry["_section"] = rng.choice(("CHARACTERS", "TERMS", "TITLES", "SKILLS", "Misc"))
        elif roll < 0.75:
            entry["type"] = rng.choice(TYPES)
        elif roll < 0.8:
            entry[rng.choice(EXTRA_FIELDS)] = rng.choice(([], {}, None, ["a", "b"], {"k": "v"}, 3))
        elif roll < 0.9:
            entry["translated_name"] = _rand_text(rng)
    return entry


def _rand_custom_types(rng):
    roll = rng.random()
    if roll < 0.25:
        return None
    if roll < 0.35:
        return {}
    types_ = {}
    for name in rng.sample(("character", "term", "terms", "titles", "locations", "skills", "Nicknames"),
                           rng.randint(1, 5)):
        types_[name] = {"enabled": rng.random() < 0.8, "has_gender": rng.random() < 0.5}
    return types_


def _rand_config(rng, *, owner_effective=False):
    config = {}
    roll = rng.random()
    if roll < 0.6:
        config["custom_glossary_fields"] = rng.choice(([], ["description"], ["description", "fun fact"],
                                                       ["fun fact", "rank"], ["notes"]))
    elif roll < 0.65 and not owner_effective:
        config["custom_glossary_fields"] = "desc"
    custom_types = _rand_custom_types(rng)
    if custom_types is not None:
        config["custom_entry_types"] = custom_types
    for key in ("glossary_skip_gender_tracking", "glossary_use_legacy_csv", "glossary_disable_honorifics_filter",
                "custom_field_description_removed"):
        if rng.random() < 0.4:
            config[key] = rng.random() < 0.5
    if owner_effective:
        config = gd.EditorOwner(config).config
    return config


def _rand_token_lines(rng, real_lines):
    lines = []
    header = rng.random()
    if header < 0.3:
        cols = ["translated_name", "raw_name", "gender", "description"]
        cols += rng.sample(EXTRA_FIELDS[1:], rng.randint(0, 3))
        if rng.random() < 0.2:
            cols = cols[:rng.randint(0, 4)]
        lines.append(rng.choice(("Glossary Columns: ", "glossary columns:", "GLOSSARY COLUMNS: ")) + ", ".join(cols))
        lines.append("")
    for _ in range(rng.randint(0, 25)):
        roll = rng.random()
        if roll < 0.35:
            lines.append(rng.choice(real_lines).rstrip("\r\n"))
        elif roll < 0.45:
            lines.append(f"=== {rng.choice(('CHARACTERS', 'TERMS', 'TITLES', 'SKILLS', 'Locations', 'misc', ''))} ===")
        elif roll < 0.8:
            raw, tr = _rand_text(rng), _rand_text(rng)
            head = rng.choice((f"* {raw} = {tr}", f"* {tr} ({raw})", f"* {tr}", f"*  {raw}={tr}  "))
            if rng.random() < 0.5:
                head += f" [{rng.choice(GENDERS)}]"
            if rng.random() < 0.3:
                head += f" ({rng.choice(EXTRA_FIELDS)}: {_rand_text(rng)})"
            tail = ""
            if rng.random() < 0.6:
                tail = ": " + _rand_text(rng)
                if rng.random() < 0.4:
                    tail += " | " + f"{rng.choice(EXTRA_FIELDS)}: {_rand_text(rng)}"
                if rng.random() < 0.3:
                    tail += f", {rng.choice(EXTRA_FIELDS)}: {_rand_text(rng)}"
                if rng.random() < 0.3:
                    tail += f" ({rng.choice(EXTRA_FIELDS)}: {_rand_text(rng)}, {rng.choice(EXTRA_FIELDS)}: x)"
            lines.append(head + tail)
        else:
            lines.append(_rand_text(rng))
    return [line + rng.choice(("\n", "\r\n", "\n")) for line in lines]


def _real_pairs():
    """(glossary bytes, tracker bytes) of the real books that have a gender tracker."""
    pairs = []
    for book in REAL_BOOKS:
        glossary = SRC / "Glossary" / book / f"{book}_glossary.csv"
        tracker = Path(gd.tracker_path_for_glossary(str(glossary)))
        if glossary.is_file() and tracker.is_file():
            pairs.append((glossary.read_bytes(), tracker.read_bytes()))
    return pairs


_REAL_PAIRS = _real_pairs()


def _rand_glossary_file(rng, folder, real_lines, real_entries, real_trackers):
    """Write a random glossary (+ maybe a gender tracker); returns its path."""
    folder.mkdir(parents=True, exist_ok=True)
    if _REAL_PAIRS and rng.random() < 0.15:
        # a real book glossary with its tracker (tracked gender variants fold), lines dropped
        glossary, tracker = rng.choice(_REAL_PAIRS)
        lines = glossary.split(b"\n")
        kept = [line for line in lines if rng.random() < 0.9 or not line.startswith(b"* ")]
        path = folder / "Book_glossary.csv"
        path.write_bytes(b"\n".join(kept))
        Path(gd.tracker_path_for_glossary(str(path))).write_bytes(tracker)
        return path
    kind = rng.choice(("token", "token", "token", "legacy", "usep", "json_list", "json_dict", "json_flat",
                       "broken_json", "empty"))
    ext = ".json" if kind.startswith("json") or kind == "broken_json" else rng.choice((".csv", ".csv", ".CSV"))
    name = rng.choice(("Book_glossary", "glossary", "Book"))
    path = folder / f"{name}{ext}"
    entries = [_rand_entry(rng, real_entries) for _ in range(rng.randint(0, 12))]
    if kind == "token":
        text = "".join(_rand_token_lines(rng, real_lines))
    elif kind in ("legacy", "usep"):
        sep = "," if kind == "legacy" else "\x1f"
        rows = []
        if rng.random() < 0.6:
            rows.append(sep.join(rng.choice((["type", "raw_name", "translated_name", "gender", "description"],
                                             ["Type", "raw_name", "translated_name"],
                                             ["type", "raw_name", "translated_name", "description", "rank"]))))
        for entry in entries:
            cells = [str(entry.get(k, "")) for k in ("type", "raw_name", "translated_name", "gender")]
            cells += [_rand_text(rng) for _ in range(rng.randint(0, 3))]
            if kind == "legacy" and rng.random() < 0.3:
                cells = ['"' + c.replace('"', '""') + '"' for c in cells]
            rows.append(sep.join(cells[:rng.randint(1, len(cells))]))
        text = "\n".join(rows) + rng.choice(("", "\n", "\r\n"))
    elif kind == "json_list":
        data = list(entries)
        if rng.random() < 0.1:
            data.append(rng.choice(("str", 3, None)))
        text = json.dumps(data, ensure_ascii=rng.random() < 0.5, indent=rng.choice((None, 2)))
    elif kind == "json_dict":
        text = json.dumps({"entries": {e.get("raw_name", ""): e.get("translated_name", "") for e in entries}},
                          ensure_ascii=False)
    elif kind == "json_flat":
        text = json.dumps({e.get("raw_name", ""): e.get("translated_name", "") for e in entries}, ensure_ascii=False)
    elif kind == "broken_json":
        text = "{not json"
    else:
        text = ""
    data = text.encode("utf-8")
    if rng.random() < 0.08:
        data = b"\xef\xbb\xbf" + data
    path.write_bytes(data)
    tracker_path = Path(gd.tracker_path_for_glossary(str(path)))
    roll = rng.random()
    if roll < 0.35 and real_trackers:
        tracker_path.write_bytes(rng.choice(real_trackers))
    elif roll < 0.45:
        tracker_path.write_text("not json", encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def real_trackers():
    out = []
    for book in REAL_BOOKS:
        out.extend(p.read_bytes() for p in sorted((SRC / "Glossary" / book).glob("*_gender_tracker.json")))
    # CI has no src/Glossary: fall back to the synthetic 루나 female/male conflict tracker, so the
    # parser still collapses gender variants (real_lines' fallback has a 루나 character line).
    return out or [_synthetic_editor_fixtures()[0].tracker]


# =============================================================================================
# H: hygiene
# =============================================================================================

def test_module_is_gui_free_cheap_and_python_310():
    source = (SRC / "glossary_document.py").read_text(encoding="utf-8")
    ast.parse(source, feature_version=(3, 10))
    code = (
        "import sys; sys.modules['PySide6'] = None; sys.path.insert(0, %r); "
        "import glossary_document as gd; "
        "heavy = [m for m in ('translator_gui', 'dpi_setup', 'GlossaryManager_GUI', 'PySide6', "
        "'extract_glossary_from_epub', 'unified_api_client') if sys.modules.get(m)]; "
        "doc = gd.GlossaryDocument({}); "
        "print(heavy, sorted(doc.custom_entry_types)[0], doc.config['custom_glossary_fields'])" % str(SRC)
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == "[] character ['description']"


@pytest.mark.parametrize("name", ["glossary_document.py", "GlossaryManager_GUI.py"])
def test_line_endings_are_uniform_and_bom_unchanged(name):
    data = (SRC / name).read_bytes()
    assert data.count(b"\r\n") in (0, data.count(b"\n")), "mixed line endings"
    assert not data.startswith(b"\xef\xbb\xbf")


# =============================================================================================
# V: verbatim moves and the rewire-only diff
# =============================================================================================

def test_moved_blocks_are_verbatim():
    source = _current_text("glossary_document.py")
    missing = [name for name in MOVED_BLOCKS if _legacy_block(name) not in source]
    assert not missing, f"not found verbatim (+ documented edits): {missing}"


def test_glossary_manager_changed_only_in_rewired_spans():
    legacy = _git_text("src/GlossaryManager_GUI.py").split("\n")
    current = _current_text("GlossaryManager_GUI.py").split("\n")
    li = ci = 0
    for start, end, n_new in REWIRE_SPANS:
        segment = legacy[li:start - 1]
        assert current[ci:ci + len(segment)] == segment, f"unexpected change before frozen line {start}"
        ci += len(segment) + n_new
        li = end
    assert current[ci:] == legacy[li:], "unexpected change after the last rewired span"


@needs_qt
def test_reexports_are_the_shared_functions(new_gm):
    for old, new in REEXPORTS.items():
        assert getattr(new_gm, old) is getattr(gd, new), old
    assert new_gm.GlossaryManagerMixin._display_glossary_path is gd.display_glossary_path


# =============================================================================================
# D: differential fuzz (frozen methods / closures vs working tree vs shared functions)
# =============================================================================================

class _Owner:
    """Attribute bag standing in for TranslatorGUI in extracted methods / closures."""


def _owner(base, **attrs):
    cls = type("Owner", (base,), {})
    owner = cls()
    for key, value in attrs.items():
        setattr(owner, key, value)
    return owner


@needs_qt
def test_token_parsers_match_frozen(legacy_gm, new_gm, real_lines):
    legacy_text = _git_text("src/GlossaryManager_GUI.py")
    for state in range(STATES):
        rng = random.Random(SEED + state)
        lines = _rand_token_lines(rng, real_lines)
        config = _rand_config(rng)
        attrs = {"config": config}
        custom_types = _rand_custom_types(rng)
        if custom_types is not None or rng.random() < 0.5:
            attrs["custom_entry_types"] = custom_types
        legacy_owner = _owner(legacy_gm.GlossaryManagerMixin, **copy.deepcopy(attrs))
        new_owner = _owner(new_gm.GlossaryManagerMixin, **copy.deepcopy(attrs))
        inline = _closure(legacy_gm, legacy_text, ("_setup_glossary_editor_tab", "load_glossary_for_editing",
                                                   "parse_token_efficient_glossary"), self=legacy_owner)
        expected = _run(legacy_owner._parse_editor_token_glossary_async, list(lines))
        assert _run(inline, list(lines)) == expected, state
        assert _run(new_owner._parse_editor_token_glossary_async, list(lines)) == expected, state
        shared = _run(gd.parse_token_glossary, list(lines), config.get("custom_glossary_fields", []),
                      getattr(new_owner, "custom_entry_types", {}))
        assert shared == expected, state


@needs_qt
def test_file_parser_matches_frozen(tmp_path, legacy_gm, new_gm, real_lines, real_entries, real_trackers):
    parsed = collapsed = 0
    for state in range(STATES):
        rng = random.Random(SEED * 3 + state)
        folder = tmp_path / f"s{state}"
        path = _rand_glossary_file(rng, folder, real_lines, real_entries, real_trackers)
        config = _rand_config(rng)
        attrs = {"config": config}
        custom_types = _rand_custom_types(rng)
        if custom_types is not None or rng.random() < 0.5:
            attrs["custom_entry_types"] = custom_types
        env_skip = rng.random() < 0.1
        if env_skip:
            os.environ["GLOSSARY_SKIP_GENDER_TRACKING"] = "1"
        try:
            legacy_owner = _owner(legacy_gm.GlossaryManagerMixin, **copy.deepcopy(attrs))
            new_owner = _owner(new_gm.GlossaryManagerMixin, **copy.deepcopy(attrs))
            before = _tree_bytes(folder)
            expected = _run(legacy_owner._parse_glossary_file_for_editor_async, str(path))
            assert _run(new_owner._parse_glossary_file_for_editor_async, str(path)) == expected, state
            shared = _run(gd.parse_glossary_file, str(path), copy.deepcopy(config),
                          copy.deepcopy(attrs.get("custom_entry_types")))
            assert shared == expected, state
            assert _tree_bytes(folder) == before, "parsing never writes"
            if expected[0] == "ok" and expected[1]["entries"]:
                parsed += 1
                collapsed += bool(expected[1]["gender_variants_collapsed"])
        finally:
            os.environ.pop("GLOSSARY_SKIP_GENDER_TRACKING", None)
    assert parsed >= STATES // 3 and (collapsed or STATES < 100), (parsed, collapsed)


@needs_qt
def test_type_count_summary_matches_frozen(legacy_gm, new_gm, real_entries):
    for state in range(STATES):
        rng = random.Random(SEED * 5 + state)
        entries = [_rand_entry(rng, real_entries) for _ in range(rng.randint(0, 40))]
        if rng.random() < 0.1:
            entries.append("not a dict")
        attrs = {}
        if rng.random() < 0.8:
            attrs["config"] = _rand_config(rng)
        custom_types = _rand_custom_types(rng)
        if custom_types is not None:
            attrs["custom_entry_types"] = custom_types
        limit = rng.choice((7, 0, 2, -1, "3", "x", None))
        legacy_owner = _owner(legacy_gm.GlossaryManagerMixin, **copy.deepcopy(attrs))
        new_owner = _owner(new_gm.GlossaryManagerMixin, **copy.deepcopy(attrs))
        expected = _run(legacy_owner._glossary_type_count_summary, entries, limit)
        assert _run(new_owner._glossary_type_count_summary, entries, limit) == expected, state


# ---- save -----------------------------------------------------------------------------------

def _rand_document(rng, real_entries):
    fmt = rng.choice(("token_csv", "token_csv", "list", "dict"))
    if fmt == "dict":
        data = {"entries": {_rand_text(rng): _rand_text(rng) for _ in range(rng.randint(0, 8))}}
        if rng.random() < 0.3:
            data["meta"] = {"k": 1}
    else:
        data = [_rand_entry(rng, real_entries) for _ in range(rng.randint(0, 15))]
    sections = rng.choice(([], ["CHARACTERS", "TERMS"], ["TITLES"], ["CHARACTERS", "TERMS", "SKILLS", "Misc"], None))
    return fmt, data, sections


def _save_side(root, rng_state, scenario, runner):
    """Materialise a save scenario under ``root`` and run ``runner(owner)``; return the observation."""
    fmt, data, sections, name, tracker, pending, config = scenario
    root.mkdir(parents=True, exist_ok=True)
    path = root / name
    path.write_text("previous content\n", encoding="utf-8")
    if tracker is not None:
        Path(gd.tracker_path_for_glossary(str(path))).write_bytes(tracker)
    attrs = {
        "config": copy.deepcopy(config),
        "editor_file_entry": FakeEntry(str(path)),
        "current_glossary_data": copy.deepcopy(data),
        "current_glossary_format": fmt,
        "current_gender_tracker_data": None,
        "current_gender_tracker_path": "",
        "_pending_gender_decisions": copy.deepcopy(pending),
        "_gender_variants_pending_save": 3,
    }
    if sections is not None:
        attrs["current_glossary_sections"] = list(sections)
    result = runner(attrs, path)
    owner = result.pop("owner")
    state = {k: getattr(owner, k, "<absent>") for k in (
        "current_glossary_data", "current_glossary_format", "current_glossary_sections",
        "current_gender_tracker_data", "current_gender_tracker_path", "_pending_gender_decisions",
        "_gender_variants_pending_save")}
    return _norm({"result": result, "state": state,
                  "files": {k: v.decode("utf-8", "replace") for k, v in _tree_bytes(root).items()}}, root)


def _rand_save_scenario(rng, real_entries, real_trackers, *, owner_effective):
    fmt, data, sections = _rand_document(rng, real_entries)
    name = rng.choice(("Book_glossary.csv", "Book_glossary.json", "glossary.csv", "Book.CSV", "Book_glossary.txt"))
    tracker = rng.choice(real_trackers) if (real_trackers and rng.random() < 0.4) else None
    pending = {}
    if rng.random() < 0.3 and isinstance(data, list):
        for entry in rng.sample(data, min(len(data), 2)):
            if isinstance(entry, dict) and entry.get("raw_name"):
                pending[str(entry["raw_name"])] = rng.choice(("auto", "male", "female"))
    return fmt, data, sections, name, tracker, pending, _rand_config(rng, owner_effective=owner_effective)


@needs_qt
def test_save_matches_frozen_closure(tmp_path, monkeypatch, legacy_gm, new_gm, real_entries, real_trackers):
    monkeypatch.setattr(os, "fsync", lambda _fd: None)  # durability is not under test; keeps 500 states fast
    legacy_text = _git_text("src/GlossaryManager_GUI.py")
    new_text = _current_text("GlossaryManager_GUI.py")
    path_ = ("_setup_glossary_editor_tab", "save_current_glossary")
    for state in range(STATES):
        rng = random.Random(SEED * 7 + state)
        owner_effective = rng.random() < 0.7
        scenario = _rand_save_scenario(rng, real_entries, real_trackers, owner_effective=owner_effective)

        def closure_runner(module, text):
            def runner(attrs, path):
                owner = _owner(_Owner, **attrs)
                boxes = BoxRecorder()
                fn = _closure(module, text, path_, self=owner, parent=None, QMessageBox=boxes)
                return {"ret": _run(fn), "boxes": boxes.log, "owner": owner}
            return runner

        def document_runner(attrs, path):
            doc = gd.GlossaryDocument(attrs["config"])
            doc.path = str(path)
            for key, value in attrs.items():
                if key not in ("config", "editor_file_entry"):
                    setattr(doc, key, value)
            if "current_glossary_sections" not in attrs:
                del doc.current_glossary_sections
            outcome = _run(doc.save)
            boxes = []
            if outcome[0] == "raise":
                boxes = [("critical", ["Error", outcome[2]])]
                outcome = ("ok", False)
            return {"ret": outcome, "boxes": boxes, "owner": doc}

        legacy = _save_side(tmp_path / f"s{state}" / "legacy", state, scenario, closure_runner(legacy_gm, legacy_text))
        new = _save_side(tmp_path / f"s{state}" / "new", state, scenario, closure_runner(new_gm, new_text))
        assert new == legacy, state
        if owner_effective:
            mobile = _save_side(tmp_path / f"s{state}" / "doc", state, scenario, document_runner)
            assert mobile == legacy, state


# ---- Convert Format ------------------------------------------------------------------------

@needs_qt
def test_convert_format_matches_frozen(tmp_path, monkeypatch, legacy_gm, new_gm, real_entries):
    from PySide6.QtWidgets import QFileDialog, QMessageBox

    boxes = []
    target = {"path": ""}
    for kind in ("critical", "information", "warning"):
        monkeypatch.setattr(QMessageBox, kind, staticmethod(
            lambda *a, _k=kind, **kw: boxes.append((_k, [x for x in a if isinstance(x, str)]))))
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (boxes.append(("dialog", [x for x in a if isinstance(x, str)]))
                                                      or (target["path"], ""))))
    converted = 0
    for state in range(STATES):
        rng = random.Random(SEED * 11 + state)
        fmt, data, sections = _rand_document(rng, real_entries)
        if rng.random() < 0.3 and isinstance(data, list):
            for entry in data:  # old JSON shapes
                if isinstance(entry, dict):
                    entry.pop("type", None)
                    entry["original_name"] = entry.pop("raw_name", "")
                    entry["name"] = entry.pop("translated_name", "")
        config = _rand_config(rng, owner_effective=True)
        if rng.random() < 0.3:
            config["output_directory"] = "out"
        backup_ok = rng.random() < 0.9
        observations = []
        for side in ("legacy", "new", "doc"):
            root = tmp_path / f"s{state}" / side
            root.mkdir(parents=True)
            current = root / rng_choice_name(state)
            current.write_text("old", encoding="utf-8")
            choice = random.Random(state).choice(("same", "other", "cancel", "upper"))
            target["path"] = {"same": str(current), "other": str(root / "converted.csv"), "cancel": "",
                              "upper": str(root / "Converted.CSV")}[choice]
            boxes.clear()
            reloads = []
            attrs = dict(config=copy.deepcopy(config), current_glossary_data=copy.deepcopy(data),
                         current_glossary_format=fmt, editor_file_entry=FakeEntry(str(current)),
                         dialog=None, logs=[])
            if sections is not None:
                attrs["current_glossary_sections"] = list(sections)
            if side == "doc":
                doc = gd.GlossaryDocument(attrs["config"], backup=lambda _d, _op: backup_ok,
                                          log=lambda m, _a=attrs: _a["logs"].append(m))
                doc.path = str(current)
                doc.current_glossary_data = attrs["current_glossary_data"]
                doc.current_glossary_format = fmt
                if sections is None:
                    del doc.current_glossary_sections
                else:
                    doc.current_glossary_sections = attrs["current_glossary_sections"]
                if choice == "cancel" or not data:
                    outcome = ("skipped",)
                else:
                    doc.load = lambda *_a, **_k: reloads.append("reload")
                    outcome = _run(doc.convert, target["path"])
                owner = doc
                logs = attrs["logs"]
            else:
                module = legacy_gm if side == "legacy" else new_gm
                owner = _owner(module.GlossaryManagerMixin, **attrs)
                owner.append_log = owner.logs.append
                owner.create_glossary_backup = lambda _op: backup_ok
                outcome = _run(owner.convert_glossary_format, lambda: reloads.append("reload"))
                logs = owner.logs
            observations.append(_norm({
                "outcome": outcome if side != "doc" else None,
                "boxes": list(boxes) if side != "doc" else None,
                "files": {k: v.decode("utf-8") for k, v in _tree_bytes(root).items()},
                "sections": getattr(owner, "current_glossary_sections", "<absent>"),
                "data": owner.current_glossary_data,
                "reloads": reloads,
                "logs": [line for line in logs if "Exported" in line or "failed" in line],
            }, root))
        legacy, new, doc_obs = observations
        assert new == legacy, state
        assert doc_obs["files"] == legacy["files"], state
        assert doc_obs["sections"] == legacy["sections"] and doc_obs["data"] == legacy["data"], state
        assert doc_obs["reloads"] == legacy["reloads"], state
        assert doc_obs["logs"] == legacy["logs"], state
        converted += any(text != "old" for text in legacy["files"].values())
    assert converted >= STATES // 5, f"only {converted} conversions wrote a file"


def rng_choice_name(state):
    return random.Random(state * 13).choice(("Book_glossary.json", "Book_glossary.csv", "glossary.json", "g.csv"))


# ---- editor view helpers --------------------------------------------------------------------

def _rand_view_state(rng, real_entries):
    fmt, data, sections = _rand_document(rng, real_entries)
    fmt = rng.choice((fmt, fmt, None, "other"))
    if fmt in ("list", "token_csv") and isinstance(data, list):
        fields = ["type", "raw_name", "translated_name", "gender"] + rng.sample(
            ["description", "fun fact", "_section", "aliases"], rng.randint(0, 3))
    else:
        fields = rng.choice((["original", "translated"], ["original", "translated", "gender"], []))
    if rng.random() < 0.1 and isinstance(data, dict):
        data["entries"] = {k: {"translated": v, "gender": "male"} for k, v in data.get("entries", {}).items()}
    baseline = {}
    if isinstance(data, list):
        for idx, entry in enumerate(data):
            if isinstance(entry, dict) and rng.random() < 0.7:
                baseline[idx] = entry.get("translated_name", "") if rng.random() < 0.7 else _rand_text(rng)
    elif isinstance(data, dict):
        for key, value in data.get("entries", {}).items():
            if rng.random() < 0.7:
                baseline[key] = value if rng.random() < 0.7 else _rand_text(rng)
    filters = {}
    for field in rng.sample(fields, min(len(fields), rng.randint(0, 2))):
        filters[field] = {_rand_text(rng) for _ in range(rng.randint(0, 3))}
    return fmt, data, fields, baseline, filters


@needs_qt
def test_view_helpers_match_frozen_closures(legacy_gm, new_gm, real_entries):
    from PySide6.QtCore import Qt

    legacy_text = _git_text("src/GlossaryManager_GUI.py")
    new_text = _current_text("GlossaryManager_GUI.py")
    names = ("_editor_data_row_specs", "_editor_display_value", "collect_translated_changes",
             "_loaded_glossary_stats_text", "get_baseline_translated", "_tree_item_matches_glossary_filters")
    for state in range(STATES):
        rng = random.Random(SEED * 17 + state)
        fmt, data, fields, baseline, filters = _rand_view_state(rng, real_entries)
        config = _rand_config(rng)
        results = []
        for module, text in ((legacy_gm, legacy_text), (new_gm, new_text)):
            owner = _owner(module.GlossaryManagerMixin, config=copy.deepcopy(config),
                           current_glossary_data=copy.deepcopy(data), current_glossary_format=fmt,
                           glossary_column_fields=list(fields), _original_translated_map=dict(baseline),
                           _glossary_column_filters=copy.deepcopy(filters))
            active = _closure(module, text, ("_setup_glossary_editor_tab", "_active_glossary_column_filters"),
                              self=owner)
            fns = {name: _closure(module, text, ("_setup_glossary_editor_tab", name), self=owner, Qt=Qt,
                                  _active_glossary_column_filters=active) for name in names}
            specs = _run(fns["_editor_data_row_specs"])
            out = {"specs": specs, "changes": _run(fns["collect_translated_changes"]),
                   "stats": _run(fns["_loaded_glossary_stats_text"], data if isinstance(data, list) else [])}
            rows = []
            if specs[0] == "ok":
                for display_idx, (source_idx, ref, entry) in enumerate(specs[1][1], start=1):
                    texts = [str(display_idx)] + [fns["_editor_display_value"](entry, f) for f in specs[1][0]]
                    item = FakeItem(texts, ref)
                    rows.append([texts,
                                 [_run(fns["get_baseline_translated"], item, f) for f in specs[1][0] + ["translated"]],
                                 _run(fns["_tree_item_matches_glossary_filters"], item)])
            out["rows"] = rows
            results.append(_norm(out))
        assert results[1] == results[0], state
        # the GUI-free row model shows the same texts
        if results[0]["specs"][0] == "ok":
            doc_rows = gd.editor_rows(fields, copy.deepcopy(data), fmt)
            assert [r.texts for r in doc_rows] == [r[0] for r in results[0]["rows"]], state


# ---- Filter Entries ------------------------------------------------------------------------

@needs_qt
def test_filter_entries_match_frozen_closure(legacy_gm, new_gm, real_entries):
    legacy_text = _git_text("src/GlossaryManager_GUI.py")
    new_text = _current_text("GlossaryManager_GUI.py")
    base = ("_setup_glossary_editor_tab", "filter_entries_dialog")
    for state in range(STATES):
        rng = random.Random(SEED * 19 + state)
        data = [_rand_entry(rng, real_entries) for _ in range(rng.randint(0, 30))]
        config = _rand_config(rng)
        types_ = sorted({str(e.get("type", "")) for e in data if isinstance(e, dict)} | {"character", "terms"})
        checks = {t: rng.random() < 0.7 for t in types_ if rng.random() < 0.9}
        limits = {t: rng.choice(("", "", "2", "0", "x", " 3 ", "-1")) for t in types_ if rng.random() < 0.8}
        search = rng.choice(("", "", "a", "LUNA", " kai ", _rand_text(rng)))
        gender = rng.choice(("all", "all", "Male", "Female", "Unknown"))
        is_new = rng.random() < 0.8
        results = []
        for module, text in ((legacy_gm, legacy_text), (new_gm, new_text)):
            owner = _owner(_Owner, config=copy.deepcopy(config))
            free = dict(self=owner, is_new_format=is_new, gender_value=gender,
                        type_checks={t: FakeCheck(v) for t, v in checks.items()},
                        type_limits={t: FakeEntry(v) for t, v in limits.items()},
                        search_entry=FakeEntry(search))
            if module is legacy_gm:
                free["get_type_limit"] = _closure(module, text, base + ("get_type_limit",), **free)
            matches = _closure(module, text, base + ("check_entry_matches",), **free)
            counts = {}
            results.append(([_run(matches, copy.deepcopy(e), counts) for e in data],
                            [_run(matches, copy.deepcopy(e)) for e in data], counts))
        assert results[1] == results[0], state
        doc = gd.GlossaryDocument(config)
        doc.current_glossary_data, doc.current_glossary_format = copy.deepcopy(data), "token_csv"
        matcher = doc.filter_matcher(kept_types=checks, search_text=search, gender_value=gender,
                                     type_limits=limits)
        if is_new == gd.is_new_format_data(data, "token_csv"):
            counts = {}
            assert ([_run(matcher, copy.deepcopy(e), counts) for e in data], counts) == \
                (results[0][0], results[0][2]), state


# ---- Find / Replace -----------------------------------------------------------------------

def _rand_rows(rng, fmt, data, fields):
    rows = []
    for display_idx, (source_idx, ref, entry) in enumerate(gd.editor_row_specs(fields, data, fmt)[1], start=1):
        texts = [str(display_idx)] + [gd.editor_display_value(entry, f) for f in fields]
        rows.append((texts, ref if rng.random() < 0.95 else rng.choice(("x", None, 999))))
    return rows


@needs_qt
def test_find_replace_match_frozen_closures(legacy_gm, new_gm, real_entries):
    import re as re_module

    legacy_text = _git_text("src/GlossaryManager_GUI.py")
    new_text = _current_text("GlossaryManager_GUI.py")
    base = ("_setup_glossary_editor_tab", "find_in_tree")
    for state in range(STATES):
        rng = random.Random(SEED * 23 + state)
        fmt, data, fields, baseline, _filters = _rand_view_state(rng, real_entries)
        if fmt not in ("list", "token_csv", "dict"):
            fmt = "token_csv" if isinstance(data, list) else "dict"
        rows = _rand_rows(rng, fmt, data, fields)
        if rng.random() < 0.3:
            fields = fields[:-1] if fields else fields
        find = rng.choice(("a", "LUNA", "an", "", "(", "x: y", "카이", _rand_text(rng)))
        repl = rng.choice(("", "b", "Kai", "$1", "\\g<0>", _rand_text(rng)))
        action = rng.choice(("replace_all", "replace_current", "find_next", "find_next"))
        answers = [rng.choice((BoxRecorder.Yes, BoxRecorder.No))]
        last_find = rng.choice(("", find, "zz"))
        last_pos = rng.choice((-1, 0, 2, 50))
        current_idx = rng.choice(list(range(len(rows))) + [None])
        results = []
        for module, text in ((legacy_gm, legacy_text), (new_gm, new_text)):
            items = [FakeItem(t, r) for t, r in rows]
            tree = FakeTree(items, items[current_idx] if current_idx is not None else None)
            owner = _owner(_Owner, glossary_column_fields=list(fields), current_glossary_format=fmt,
                           current_glossary_data=copy.deepcopy(data), glossary_tree=tree, dialog=None,
                           _last_find_text=last_find, _last_find_pos=last_pos, logs=[])
            owner.append_log = owner.logs.append
            owner._push_undo_snapshot = Recorder()
            owner._push_html_undo_snapshot = Recorder()
            owner._apply_glossary_column_filters = Recorder()
            highlights = Recorder()
            status = FakeEntry()
            html = Recorder(result=lambda changes: (len(changes), 2))
            boxes = BoxRecorder(answers)
            free = dict(self=owner, find_edit=FakeEntry(find), replace_edit=FakeEntry(repl), re=re_module,
                        update_row_highlight=lambda item, col, val, _h=highlights: _h(item.state(), col, val),
                        status_label=status, update_html_files=html, QMessageBox=boxes)
            free["replace_in_item"] = _closure(module, text, base + ("replace_in_item",), **free)
            fn = _closure(module, text, base + (action,), **free)
            outcome = _run(fn)
            results.append(_norm({
                "outcome": outcome, "items": [i.state() for i in items], "data": owner.current_glossary_data,
                "status": status.text(), "undo": len(owner._push_undo_snapshot.calls),
                "html_undo": owner._push_html_undo_snapshot.calls, "filters": len(owner._apply_glossary_column_filters.calls),
                "highlights": highlights.calls, "html": html.calls, "boxes": boxes.log, "logs": owner.logs,
                "last": [getattr(owner, "_last_find_text", None), getattr(owner, "_last_find_pos", None),
                         getattr(owner, "_last_replace_text", None)],
                "current": tree.current.state() if tree.current is not None else None,
            }))
        assert results[1] == results[0], state
        # GlossaryDocument over EditorRows: same data, rows and replacement count
        if action == "replace_all" and rows:
            doc = gd.GlossaryDocument({})
            doc.current_glossary_data, doc.current_glossary_format = copy.deepcopy(data), fmt
            doc.glossary_column_fields = list(fields)
            doc_rows = [gd.EditorRow(t, r) for t, r in rows]
            outcome = _run(doc.replace_all, find, repl, doc_rows)
            legacy_outcome = results[0]["outcome"]
            assert outcome[0] == legacy_outcome[0], state
            if outcome[0] == "raise":
                assert outcome[1] == legacy_outcome[1], state
                count = None
            else:
                count = outcome[1]
            assert _norm(doc.current_glossary_data) == results[0]["data"], state
            assert _norm([[r.texts, r.source_ref] for r in doc_rows]) == results[0]["items"], state
            assert doc.can_undo() == bool(results[0]["undo"]), state
            status = results[0]["status"]
            if count is None:
                pass
            elif status.startswith("Replaced"):
                assert count == int(re.search(r"Replaced (\d+)", status).group(1)), state
            elif status.startswith("Updated"):
                assert count == 0, state


# ---- Update output files ----------------------------------------------------------------------

def _rand_output_layout(rng, root):
    book = rng.choice(("Book", "My Novel"))
    layout = rng.choice(("shared", "book_sub", "minimal", "minimal_sub", "shared_noname"))
    if layout == "shared":
        glossary = root / "Glossary" / f"{book}_glossary.csv"
        out_dir = root / book
    elif layout == "shared_noname":
        glossary = root / "Glossary" / f"{book}.csv"
        out_dir = root / book
    elif layout == "book_sub":
        glossary = root / "Glossary" / book / f"{book}_glossary.csv"
        out_dir = root / book
    elif layout == "minimal":
        glossary = root / book / "glossary.csv"
        out_dir = root / book
    else:
        glossary = root / book / "Glossary" / "glossary.csv"
        out_dir = root / book
    glossary.parent.mkdir(parents=True, exist_ok=True)
    if rng.random() < 0.9:
        glossary.write_text("* a = b\n", encoding="utf-8")
    if rng.random() < 0.85:
        out_dir.mkdir(parents=True, exist_ok=True)
        names = ["response_001_ch1.html", "response_002.xhtml", "notes.txt", "metadata.json", "content.opf",
                 "toc.ncx", "glossary.csv", "chapter.HTML", "image.png", "x.OPF"]
        for name in rng.sample(names, rng.randint(1, len(names))):
            words = [rng.choice(("Luna", "Kai", "Luna Luna", "Old Name", "Ren", "카이", "x")) for _ in range(6)]
            payload = ("<p>" + " ".join(words) + "</p>\r\n").encode("utf-8")
            if rng.random() < 0.1:
                payload += b"\xff\xfe broken"
            (out_dir / name).write_bytes(payload)
        if rng.random() < 0.3:
            (out_dir / "sub").mkdir(exist_ok=True)
            (out_dir / "sub" / "deep.html").write_text("Luna", encoding="utf-8")
    return glossary, out_dir


@needs_qt
def test_update_output_files_match_frozen_closure(tmp_path, monkeypatch, legacy_gm, new_gm):
    monkeypatch.setattr(os, "fsync", lambda _fd: None)
    legacy_text = _git_text("src/GlossaryManager_GUI.py")
    new_text = _current_text("GlossaryManager_GUI.py")
    path_ = ("_setup_glossary_editor_tab", "update_html_files")
    for state in range(STATES):
        rng = random.Random(SEED * 29 + state)
        changes = [(rng.choice(("Luna", "Kai", "Old Name", "", None, "x")), rng.choice(("Moon", "Kai", "", "New", None)))
                   for _ in range(rng.randint(0, 4))]
        attrs = {"config": {}}
        if rng.random() < 0.5:
            attrs["enable_parallel_extraction_var"] = rng.random() < 0.7
        if rng.random() < 0.5:
            attrs["extraction_workers_var"] = rng.choice((1, 3, "2", "x", 0))
        if rng.random() < 0.3:
            attrs["config"] = {"enable_parallel_extraction": True, "extraction_workers": rng.choice((2, 4))}
        layout_seed = rng.random()
        observations = []
        for side in ("legacy", "new", "doc"):
            root = tmp_path / f"s{state}" / side
            glossary, _out_dir = _rand_output_layout(random.Random(layout_seed), root)
            logs = []
            if side == "doc":
                owner = gd.EditorOwner(attrs["config"])
                for key in ("enable_parallel_extraction_var", "extraction_workers_var"):
                    if key in attrs:
                        setattr(owner, key, attrs[key])
                    else:
                        delattr(owner, key)
                doc = gd.GlossaryDocument(owner=owner, log=logs.append)
                doc.path = str(glossary)
                outcome = _run(doc.update_output_files, copy.deepcopy(changes))
            else:
                module, text = (legacy_gm, legacy_text) if side == "legacy" else (new_gm, new_text)
                owner = _owner(_Owner, editor_file_entry=FakeEntry(str(glossary)), **copy.deepcopy(attrs))
                owner.append_log = logs.append
                fn = _closure(module, text, path_, self=owner)
                outcome = _run(fn, copy.deepcopy(changes))
            observations.append(_norm({"outcome": outcome, "logs": logs, "files": {
                k: v.decode("utf-8", "replace") for k, v in _tree_bytes(root).items()}}, root))
        assert observations[1] == observations[0], state
        assert observations[2] == observations[0], state


# ---- auto-selection of the editor's glossary files ---------------------------------------------

class FakeCombo:
    def __init__(self):
        self.items = []
        self.index = -1
        self.blocked = False
        self.log = []

    def blockSignals(self, value):
        old, self.blocked = self.blocked, value
        return old

    def clear(self):
        self.items = []
        self.index = -1
        self.log.append("clear")

    def addItem(self, text, data=None):
        self.items.append([text, data])
        if self.index < 0:
            self.index = 0

    def setItemData(self, index, value, role=None):
        self.items[index].append(value)

    def count(self):
        return len(self.items)

    def itemData(self, index):
        return self.items[index][1]

    def setCurrentIndex(self, index):
        self.index = index
        self.log.append(("index", index, self.blocked))


def _rand_select_layout(rng, root):
    sources = []
    for i in range(rng.randint(0, 3)):
        base = rng.choice(("Book", "Novel Two", "x", f"b{i}"))
        src = root / "in" / f"{base}{rng.choice(('.epub', '.txt', '.zip'))}"
        src.parent.mkdir(parents=True, exist_ok=True)
        if rng.random() < 0.9:
            src.write_text("x", encoding="utf-8")
        sources.append(str(src))
    out = root / "out"
    candidates = []
    for src in sources:
        base = Path(src).stem
        for ext in (".csv", ".json", ".txt", ".md"):
            candidates += [out / base / f"glossary{ext}", out / base / "Glossary" / f"glossary{ext}",
                           out / "Glossary" / base / f"{base}_glossary{ext}", out / "Glossary" / base / f"{base}{ext}",
                           out / "Glossary" / f"{base}_glossary{ext}", out / "Glossary" / f"{base}{ext}",
                           Path(src).parent / base / f"glossary{ext}",
                           root / "basedir" / "Glossary" / base / f"{base}_glossary{ext}",
                           root / "Glossary" / f"{base}{ext}"]
    for cand in rng.sample(candidates, min(len(candidates), rng.randint(0, 6))):
        cand.parent.mkdir(parents=True, exist_ok=True)
        cand.write_text("g", encoding="utf-8")
    attrs = {"config": {"auto_glossary_mode": rng.choice(("off", "minimal", "balanced", "full", "single_pass",
                                                            "off_no_automap")),
                        "append_glossary_auto_load": rng.random() < 0.5,
                        "append_glossary": rng.random() < 0.5}}
    if rng.random() < 0.5:
        attrs["config"]["output_directory"] = str(out)
    elif rng.random() < 0.3:
        os.environ["OUTPUT_DIRECTORY"] = str(out)
    existing = [str(c) for c in candidates if c.exists()]
    for key in ("auto_loaded_glossary_path", "manual_glossary_path"):
        if rng.random() < 0.3:
            attrs[key] = rng.choice(existing + [str(root / "missing.csv"), None]) if existing else None
    if rng.random() < 0.2:
        attrs["manual_glossary_map"] = {s: rng.choice(existing + [None]) for s in sources} if existing else {}
    if rng.random() < 0.3:
        attrs["base_dir"] = str(root / "basedir")
    if rng.random() < 0.3:
        attrs["_get_output_base_dir"] = lambda path, _r=root: str(_r / "alt")
    if rng.random() < 0.2:
        attrs["_glossary_editor_manual_source"] = True
    return sources, attrs


@needs_qt
def test_auto_select_matches_frozen_closure(tmp_path, legacy_gm, new_gm, capsys):
    legacy_text = _git_text("src/GlossaryManager_GUI.py")
    new_text = _current_text("GlossaryManager_GUI.py")
    path_ = ("_setup_glossary_editor_tab", "auto_select_current_glossary")
    for state in range(STATES):
        layout_seed = SEED * 31 + state
        observations = []
        for module, text, side in ((legacy_gm, legacy_text, "legacy"), (new_gm, new_text, "new")):
            # the combo shows the last 3 path parts: keep them equal on both sides
            root = tmp_path / side / f"s{state}" / "ws"
            root.mkdir(parents=True)
            os.chdir(root)
            os.environ.pop("OUTPUT_DIRECTORY", None)
            sources, attrs = _rand_select_layout(random.Random(layout_seed), root)
            combo = FakeCombo()
            owner = _owner(module.GlossaryManagerMixin, editor_file_combo=combo, **attrs)
            owner._glossary_editor_input_sources = lambda _s=sources: list(_s)
            owner._clear_glossary_editor = Recorder()
            owner._append_unified_glossaries_to_editor_combo = Recorder()
            owner._update_editor_nav_buttons = Recorder()
            if random.Random(layout_seed).random() < 0.5:
                owner._autofill_glossary_for_current_selection = Recorder()
            load = Recorder()
            fn = _closure(module, text, path_, self=owner, load_glossary_for_editing=load)
            outcome = _run(fn)
            printed = capsys.readouterr().out
            observations.append(_norm({
                "outcome": outcome, "printed": printed, "items": combo.items, "log": combo.log,
                "loads": len(load.calls), "clears": len(owner._clear_glossary_editor.calls),
                "autofill": len(getattr(owner, "_autofill_glossary_for_current_selection", Recorder()).calls),
                "maps": [getattr(owner, "_editor_glossary_source_map", None),
                         getattr(owner, "_editor_glossary_epub_map", None)],
                "manual": getattr(owner, "_glossary_editor_manual_source", None),
            }, root))
        assert observations[1] == observations[0], state


# ---- Unified Glossary Rebuild Now helpers (Integrate U6) ------------------------------------

@needs_qt
def test_unified_rebuild_helpers_match_frozen(tmp_path, monkeypatch, legacy_gm, new_gm):
    """The frozen ``_unified_glossary_shared_dir`` and the settings snapshot of the frozen
    ``_rebuild_unified_glossary_now`` vs the working-tree wrapper and the shared functions."""
    frozen_lines = _git_text("src/GlossaryManager_GUI.py").split("\n")
    snapshot = "\n".join(line[8:] for line in frozen_lines[5853:5859])  # `settings = {...}` (5854-5859)
    assert snapshot.startswith("settings = {") and snapshot.endswith("}")
    import app_paths

    app_dir = tmp_path / "app"
    monkeypatch.setattr(app_paths, "_get_app_dir", lambda: str(app_dir))
    # translator_gui's _get_app_dir is app_paths' (``from app_paths import ... _get_app_dir``); the
    # frozen method imports it from there, so the stand-in carries the same patched function.
    monkeypatch.setitem(sys.modules, "translator_gui", types.SimpleNamespace(_get_app_dir=app_paths._get_app_dir))
    for state in range(STATES):
        rng = random.Random(SEED * 41 + state)
        config = {}
        for key, values in (("output_directory", (None, "", "  ", str(tmp_path / "out"), "rel/out")),
                            ("output_language", (None, "", "Korean", "English")),
                            ("unified_glossary_combine_all_languages", (True, False, 0, 1, None, "yes")),
                            ("unified_glossary_exclude_gender_entries", (True, False, 0, None, ""))):
            if rng.random() < 0.7:
                config[key] = rng.choice(values)
        for key, values in (("OUTPUT_DIRECTORY", ("", str(tmp_path / "env_out"))),
                            ("OUTPUT_LANGUAGE", ("", "Japanese"))):
            if rng.random() < 0.4:
                monkeypatch.setenv(key, rng.choice(values))
            else:
                monkeypatch.delenv(key, raising=False)
        legacy = _owner(legacy_gm.GlossaryManagerMixin, config=dict(config))
        new = _owner(new_gm.GlossaryManagerMixin, config=dict(config))
        shared_dir = legacy._unified_glossary_shared_dir()
        assert new._unified_glossary_shared_dir() == shared_dir == gd.unified_glossary_shared_dir(dict(config)), state
        scope = {"self": legacy, "os": os, "shared_dir": shared_dir}
        exec(snapshot, scope)
        assert gd.unified_rebuild_settings(dict(config), shared_dir) == scope["settings"], state


# ---- Hide unused entries helpers --------------------------------------------------------------

@needs_qt
def test_hide_unused_helpers_match_frozen_closures(tmp_path, monkeypatch, legacy_gm, new_gm, real_entries):
    legacy_text = _git_text("src/GlossaryManager_GUI.py")
    new_text = _current_text("GlossaryManager_GUI.py")
    base = ("_setup_glossary_editor_tab",)
    clock = {"t": 1000.0}

    def fake_monotonic():
        clock["t"] += 0.3
        return clock["t"]

    monkeypatch.setattr(time, "monotonic", fake_monotonic)
    used_states = 0
    for state in range(STATES):
        rng = random.Random(SEED * 37 + state)
        fmt, data, fields, _baseline, _filters = _rand_view_state(rng, real_entries)
        layout_seed = rng.random()
        observations = []
        for module, text, side in ((legacy_gm, legacy_text, "legacy"), (new_gm, new_text, "new")):
            root = tmp_path / f"s{state}" / side
            root.mkdir(parents=True)
            glossary, out_dir = _rand_output_layout(random.Random(layout_seed), root)
            if out_dir.is_dir() and isinstance(data, list):
                used = random.Random(layout_seed * 7)
                names = [str(e.get("translated_name", "")) for e in data if isinstance(e, dict) and used.random() < 0.4]
                (out_dir / "response_999_used.html").write_text(
                    "<html><body><p>" + "</p><p>".join(names) + "</p></body></html>", encoding="utf-8")
            source = root / "in" / "Book.epub"
            if random.Random(layout_seed).random() < 0.5:
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_text("x", encoding="utf-8")
            config = {"output_directory": str(root)} if random.Random(layout_seed * 3).random() < 0.5 else {}
            owner = _owner(_Owner, config=config, editor_file_entry=FakeEntry(str(glossary)),
                           current_glossary_data=copy.deepcopy(data), current_glossary_format=fmt,
                           glossary_column_fields=list(fields),
                           _editor_glossary_source_map={str(glossary): str(source)})
            owner._glossary_editor_input_sources = lambda _s=str(source): [_s]
            emitted = []

            class _Bridge:
                class _Sig:
                    def __init__(self, name):
                        self.name = name

                    def emit(self, payload):
                        emitted.append((self.name, dict(payload, token="<token>")))

                progress = _Sig("progress")
                finished = _Sig("finished")

            owner._hide_unused_filter_bridge = _Bridge()
            free = {"self": owner}
            for name in ("_editor_data_row_specs", "_editor_output_dir_from_source", "_editor_associated_source_path",
                         "_editor_output_dir_from_glossary", "_editor_translated_output_dir",
                         "_editor_tree_usage_entries"):
                free[name] = _closure(module, text, base + (name,), **free)
            if module is legacy_gm:
                for name in ("_editor_translated_output_files", "_read_translated_output_texts"):
                    free[name] = _closure(module, text, base + (name,), **free)
            output_dir = free["_editor_translated_output_dir"]()
            entries = free["_editor_tree_usage_entries"]()
            total = len(free["_editor_data_row_specs"]()[1])
            clock["t"] = 1000.0
            worker = _closure(module, text, base + ("_apply_hide_unused_entries_filter", "worker"),
                              token=object(), output_dir=output_dir, entries=entries, total=total, **free)
            outcome = _run(worker) if output_dir else ("no output dir",)
            observations.append(_norm({"output_dir": output_dir, "entries": entries, "outcome": outcome,
                                       "emitted": emitted}, root))
        assert observations[1] == observations[0], state
        legacy_obs = observations[0]
        if legacy_obs["output_dir"]:
            doc = gd.GlossaryDocument({})
            doc.path = str(glossary)
            doc.current_glossary_data, doc.current_glossary_format = copy.deepcopy(data), fmt
            doc.glossary_column_fields = list(fields)
            clock["t"] = 1000.0
            result = doc.used_rows(observations[1]["output_dir"].replace("<ROOT>", str(root)))
            finished = [p for n, p in observations[1]["emitted"] if n == "finished"]
            expected = dict(finished[-1])
            expected.pop("token")
            got = _norm(dict(result), root)
            got.pop("token")
            assert got == expected, state
            used_states += bool(expected.get("used_rows"))
    assert used_states >= STATES // 10, used_states


# =============================================================================================
# F / M: the offscreen editor tab (frozen vs working tree) and GlossaryDocument, real glossaries
# =============================================================================================

#: Operation sequences per glossary fixture (tier F / M); each runs ``EDITOR_STEPS`` operations.
EDITOR_RUNS = max(1, int(os.environ.get("PARITY_U6_EDITOR_RUNS", "2")))
EDITOR_STEPS = max(1, int(os.environ.get("PARITY_U6_EDITOR_STEPS", "16")))


def _frozen_backup_methods():
    """create_glossary_backup / _clean_old_backups of the frozen TranslatorGUI (glossary_files'
    move target), as plain functions over ``(owner, ...)`` with a deterministic clock."""
    text = _git_text("src/translator_gui.py")
    out = {}
    for name in ("create_glossary_backup", "_clean_old_backups"):
        start = text.index(f"\n    def {name}(")
        end = text.index("\n    def ", start + 10)
        out[name] = textwrap.dedent(text[start + 1:end])
    return out


class _Stamp:
    """``time`` stand-in for the backup names: one deterministic stamp per backup."""

    def __init__(self):
        self.count = 0

    def strftime(self, _fmt):
        self.count += 1
        return f"20260101_{self.count:06d}"


def _bind_backups(sources, stamp, boxes):
    ns = {"os": os, "json": json, "time": stamp, "QMessageBox": boxes}
    for source in sources.values():
        exec(compile(source, "<frozen translator_gui backups>", "exec"), ns)
    return ns["create_glossary_backup"], ns["_clean_old_backups"]


class _Script:
    """What the patched dialogs do during one editor operation (shared by every side)."""

    def __init__(self):
        self.reset({})
        self.root = None

    def reset(self, decision):
        self.decision = decision
        self.answers = list(decision.get("answers", ()))
        self.log = []


_SCRIPT = _Script()


def _qt():
    from PySide6 import QtCore, QtWidgets
    return QtCore, QtWidgets


def _button(widget, text):
    _QtCore, QtWidgets = _qt()
    for button in widget.findChildren(QtWidgets.QPushButton):
        if button.text() == text:
            return button
    raise AssertionError(f"no button {text!r}")


def _handle_dialog(dialog):
    QtCore, QtWidgets = _qt()
    title = dialog.windowTitle()
    d = _SCRIPT.decision
    _SCRIPT.log.append(("dialog", title))
    if title.startswith("Edit "):
        dialog.findChild(QtWidgets.QTextEdit).setPlainText(d["value"])
        _button(dialog, "Save").click()
    elif title == "Resolve Tracked Gender":
        for radio in dialog.findChildren(QtWidgets.QRadioButton):
            if radio.text().lower() == d["decision"]:
                radio.setChecked(True)
        _button(dialog, "Apply").click()
    elif title == "Smart Trim Glossary":
        dialog.findChild(QtWidgets.QLineEdit).setText(d["top_n"])
        _button(dialog, "Preview Changes").click()
        _button(dialog, "Apply Trim").click()
    elif title == "Filter Entries":
        for frame in dialog.findChildren(QtWidgets.QFrame, "typeRow"):
            check = frame.findChild(QtWidgets.QCheckBox)
            type_name = check.text()[len("Keep "):]
            check.setChecked(d["keep"].get(type_name, True))
            frame.findChild(QtWidgets.QLineEdit).setText(d["limits"].get(type_name, ""))
        for group in dialog.findChildren(QtWidgets.QGroupBox):
            if group.title() == "Text Content Filter":
                group.findChild(QtWidgets.QLineEdit).setText(d["search"])
        for radio in dialog.findChildren(QtWidgets.QRadioButton):
            if radio.text() == d["gender_label"]:
                radio.setChecked(True)
        _button(dialog, "Preview Filter").click()
        _button(dialog, "Apply Filter").click()
    elif title == "Find / Replace":
        find_edit, replace_edit = dialog.findChildren(QtWidgets.QLineEdit)[:2]
        find_edit.setText(d["find"])
        replace_edit.setText(d["replace"])
        for name in d["buttons"]:
            _button(dialog, name).click()
        dialog.accept()
    elif title == "Automatic Backup Settings":
        _button(dialog, "Backup Now").click()
        _button(dialog, "Cancel").click()
    elif title.startswith("Filter "):
        value_list = dialog.findChild(QtWidgets.QListWidget)
        for index in d.get("uncheck", ()):
            item = value_list.item(index)
            if item is not None:
                item.setCheckState(QtCore.Qt.Unchecked)
        _button(dialog, "Apply").click()
    else:
        _SCRIPT.log.append(("unexpected dialog", title))
        dialog.reject()
    return dialog.result()


def _patch_qt_dialogs(monkeypatch):
    QtCore, QtWidgets = _qt()
    QMessageBox = QtWidgets.QMessageBox

    def static(kind):
        def show(*args, **kwargs):
            _SCRIPT.log.append((kind, [a for a in args if isinstance(a, str)]))
            return QMessageBox.Ok
        return show

    def box_exec(box):
        answer = _SCRIPT.answers.pop(0) if _SCRIPT.answers else "yes"
        _SCRIPT.log.append(("box", box.windowTitle(), box.text(), box.informativeText(), answer))
        return QMessageBox.Yes if answer == "yes" else QMessageBox.No

    def save_name(*args, **kwargs):
        rel = _SCRIPT.decision.get("save_path", "")
        _SCRIPT.log.append(("save dialog", [a for a in args if isinstance(a, str)][1:2]))
        return (str(_SCRIPT.root / rel) if rel else "", "")

    for kind in ("critical", "warning", "information", "question"):
        monkeypatch.setattr(QMessageBox, kind, staticmethod(static(kind)))
    monkeypatch.setattr(QMessageBox, "exec", box_exec)
    monkeypatch.setattr(QtWidgets.QDialog, "exec", _handle_dialog)
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", staticmethod(save_name))
    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: ("", "")))


# ---- fixtures: real glossaries in a book layout -----------------------------------------------

class GlossaryFixture:
    """A glossary laid out as the desktop finds it: <root>/in/<book>.epub (the selected input),
    <root>/out/Glossary/<book>/<file> (+ tracker) and translated output in <root>/out/<book>/."""

    def __init__(self, name, book, filename, payload, tracker=None):
        self.name, self.book, self.filename = name, book, filename
        self.payload, self.tracker = payload, tracker

    def materialize(self, root, names):
        (root / "in").mkdir(parents=True, exist_ok=True)
        (root / "in" / f"{self.book}.epub").write_bytes(b"PK")
        folder = root / "out" / "Glossary" / self.book
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / self.filename
        path.write_bytes(self.payload)
        if self.tracker is not None:
            Path(gd.tracker_path_for_glossary(str(path))).write_bytes(self.tracker)
        out = root / "out" / self.book
        out.mkdir(parents=True, exist_ok=True)
        rng = random.Random(len(names))
        for idx in range(3):
            chosen = [n for n in names if rng.random() < 0.35]
            body = "</p>\r\n<p>".join(f"{n} said hello." for n in chosen)
            (out / f"response_{idx:03d}_chapter{idx}.html").write_text(
                f"<html><body><p>{body}</p></body></html>\r\n", encoding="utf-8")
        (out / "notes.txt").write_text(" ".join(names[:5]), encoding="utf-8")
        (out / "metadata.json").write_text(json.dumps({"title": names[:1]}), encoding="utf-8")
        (out / "content.opf").write_text("<package/>", encoding="utf-8")
        return path


def _editor_fixtures():
    fixtures = []
    for book in REAL_BOOKS:
        folder = SRC / "Glossary" / book
        glossary = folder / f"{book}_glossary.csv"
        if not glossary.is_file():
            continue
        tracker = Path(gd.tracker_path_for_glossary(str(glossary)))
        fixtures.append(GlossaryFixture(book, book, glossary.name, glossary.read_bytes(),
                                        tracker.read_bytes() if tracker.is_file() else None))
    book, rel = REAL_JSON_LIST
    source = SRC / "Glossary" / book / rel
    if source.is_file():
        entries = json.loads(source.read_text(encoding="utf-8"))
        fixtures.append(GlossaryFixture("json-list", book, f"{book}_glossary.json", source.read_bytes()))
        mapping = {"entries": {e["raw_name"]: e["translated_name"] for e in entries if isinstance(e, dict)}}
        fixtures.append(GlossaryFixture("json-dict", book, f"{book}_glossary.json",
                                        json.dumps(mapping, ensure_ascii=False, indent=2).encode("utf-8")))
        rows = [["type", "raw_name", "translated_name", "gender", "description"]]
        rows += [[e.get("type", ""), e.get("raw_name", ""), e.get("translated_name", ""), e.get("gender", ""),
                  e.get("description", "")] for e in entries if isinstance(e, dict)]
        import csv
        import io
        buffer = io.StringIO()
        csv.writer(buffer).writerows(rows)
        fixtures.append(GlossaryFixture("legacy-csv", book, f"{book}_glossary.csv", buffer.getvalue().encode("utf-8")))
        fixtures.append(GlossaryFixture("usep-csv", book, f"{book}_glossary.csv",
                                        "\n".join("\x1f".join(r) for r in rows).encode("utf-8")))
    if not fixtures or os.environ.get("PARITY_U6_SYNTHETIC") == "1":
        fixtures = _synthetic_editor_fixtures()
    return fixtures


#: CI has no src/Glossary (user data, gitignored): tier F then runs on a synthetic book glossary in
#: the same five layouts (token CSV + gender tracker, JSON list / dict, legacy and \x1f CSV).
SYNTHETIC_ENTRIES = (
    {"type": "character", "raw_name": "루나", "translated_name": "Luna", "gender": "female", "description": "a witch"},
    {"type": "character", "raw_name": "카이", "translated_name": "Kai", "gender": "male", "description": "a knight"},
    {"type": "character", "raw_name": "세라", "translated_name": "Sera", "gender": "", "description": ""},
    {"type": "character", "raw_name": "루나", "translated_name": "Luna", "gender": "female", "description": ""},
    {"type": "term", "raw_name": "마나", "translated_name": "Mana", "gender": "", "description": "magic power"},
    {"type": "term", "raw_name": "은빛 숲", "translated_name": "Silver Forest", "gender": "", "description": ""},
    {"type": "term", "raw_name": "검은 탑", "translated_name": "Black Tower", "gender": "", "description": "x: y"},
    {"type": "terms", "raw_name": "성검", "translated_name": "Holy Sword", "gender": "", "description": "[tag]"},
)


def _synthetic_editor_fixtures():
    import csv
    import io
    import tempfile

    book = "Synthetic Book"
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, f"{book}_glossary.csv")
        gd.write_token_csv([dict(e) for e in SYNTHETIC_ENTRIES], path, [], ["description"])
        token = Path(path).read_bytes()
    def occurrence(raw, name, gender, chapter):
        return {"raw_name": raw, "gender": gender, "chapter_index": chapter, "chapter_num": chapter,
                "chapter_file": f"chapter{chapter:04d}.html", "source_path": f"{book}.epub", "translated_name": name}

    tracked = {
        # 루나: two genders seen (a tracked conflict the editor shows); 카이: consistent
        "루나": ("Luna", [("female", 0), ("female", 2), ("male", 5)]),
        "카이": ("Kai", [("male", 1), ("male", 3)]),
    }
    tracker_entries = {}
    for raw, (name, seen) in tracked.items():
        genders = {}
        for gender, chapter in seen:
            genders.setdefault(gender, {"first_seen_chapter": chapter, "first_seen_file": f"chapter{chapter:04d}.html"})
        tracker_entries[raw] = {"raw_name": raw, "translated_name": name, "genders": genders,
                                "occurrences": [occurrence(raw, name, g, c) for g, c in seen]}
    tracker = json.dumps({"version": 1, "entries": tracker_entries}, ensure_ascii=False, indent=2).encode("utf-8")
    entries = [dict(e) for e in SYNTHETIC_ENTRIES]
    fixtures = [GlossaryFixture("synthetic-token", book, f"{book}_glossary.csv", token, tracker),
                GlossaryFixture("synthetic-json-list", book, f"{book}_glossary.json",
                                json.dumps(entries, ensure_ascii=False, indent=2).encode("utf-8"))]
    mapping = {"entries": {e["raw_name"]: e["translated_name"] for e in entries}}
    fixtures.append(GlossaryFixture("synthetic-json-dict", book, f"{book}_glossary.json",
                                    json.dumps(mapping, ensure_ascii=False, indent=2).encode("utf-8")))
    rows = [["type", "raw_name", "translated_name", "gender", "description"]]
    rows += [[e["type"], e["raw_name"], e["translated_name"], e["gender"], e["description"]] for e in entries]
    buffer = io.StringIO()
    csv.writer(buffer).writerows(rows)
    fixtures.append(GlossaryFixture("synthetic-legacy-csv", book, f"{book}_glossary.csv",
                                    buffer.getvalue().encode("utf-8")))
    fixtures.append(GlossaryFixture("synthetic-usep-csv", book, f"{book}_glossary.csv",
                                    "\n".join("\x1f".join(r) for r in rows).encode("utf-8")))
    return fixtures


EDITOR_FIXTURES = _editor_fixtures()


# ---- one editor side -----------------------------------------------------------------------------

#: Process environment the editor writes (Load as manual glossary, Remove Duplicates).
_SIDE_ENV = ("APPEND_GLOSSARY", "MANUAL_GLOSSARY", "GLOSSARY_DISABLE_HONORIFICS_FILTER")

class EditorSide:
    """The real editor tab of one GlossaryManager_GUI module on its own temp root."""

    def __init__(self, module, root, fixture, config, backups):
        QtCore, QtWidgets = _qt()
        self.module, self.root, self.fixture = module, root, fixture
        self.logs = []
        self.saved_configs = []
        names = [n for n in re.findall(r"= ([^\[\]:|\n\r]+?)(?: \[|:|\r?\n)", fixture.payload.decode("utf-8", "replace"))][:60]
        names += re.findall(r'"translated_name": "([^"]+)"', fixture.payload.decode("utf-8", "replace"))[:60]
        self.glossary = fixture.materialize(root, names or ["Luna"])
        self.source = root / "in" / f"{fixture.book}.epub"
        self.stamp = _Stamp()
        create_backup, clean_backups = _bind_backups(backups, self.stamp, _BoxesProxy())
        side = self
        owner_values = gd.EditorOwner(config)
        cls = type("EditorOwnerU6", (module.GlossaryManagerMixin,), {
            "create_glossary_backup": create_backup,
            "_clean_old_backups": clean_backups,
            "append_log": lambda self_, message: side.logs.append(message),
            "save_config": lambda self_, show_message=True: side.saved_configs.append(show_message),
            "_show_message": lambda self_, *a, **k: side.logs.append(("message",) + tuple(map(str, a))),
        })
        owner = cls()
        owner.config = owner_values.config
        for attr in ("custom_entry_types", "glossary_gender_noise_threshold_var",
                     "glossary_gender_tracking_bias_var"):
            setattr(owner, attr, getattr(owner_values, attr))
        owner.enable_parallel_extraction_var = False
        owner.extraction_workers_var = 1
        owner.selected_files = [str(self.source)]
        owner.dialog = QtWidgets.QWidget()
        self.parent = QtWidgets.QWidget()
        self.owner = owner
        self.env = {key: None for key in _SIDE_ENV}
        with self.active():
            owner._setup_glossary_editor_tab(self.parent)
            # The 500 ms external-change poll restarts after every Save and undo; the test
            # triggers it explicitly ("external") so reloads happen at the same step on every side.
            owner._editor_auto_reload_timer.stop()
            owner._editor_auto_reload_timer.start = lambda *args: None
            self.settle()

    @contextlib.contextmanager
    def active(self):
        """Run with this side's cwd, output root and process env (the editor writes os.environ)."""
        os.environ["OUTPUT_DIRECTORY"] = str(self.root / "out")
        for key, value in self.env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        cwd = os.getcwd()
        os.chdir(self.root)
        _SCRIPT.root = self.root
        try:
            yield self
        finally:
            os.chdir(cwd)
            self.env = {key: os.environ.get(key) for key in _SIDE_ENV}

    def settle(self, timeout=60.0):
        _QtCore, QtWidgets = _qt()
        owner = self.owner
        deadline = time.monotonic() + timeout
        quiet = 0
        busy_prefixes = ("Loading glossary", "Applying filtered", "Scanning translated", "Matching glossary",
                         "Reading translated")
        while time.monotonic() < deadline:
            QtWidgets.QApplication.processEvents()
            busy = (getattr(owner, "_editor_load_token", None) is not None
                    or getattr(owner, "_hide_unused_filter_token", None) is not None
                    or owner.stats_label.text().startswith(busy_prefixes))
            quiet = 0 if busy else quiet + 1
            if quiet >= 8:
                return
            time.sleep(0.003)
        raise AssertionError(f"editor did not settle: {owner.stats_label.text()!r}")

    def wait(self, seconds):
        _QtCore, QtWidgets = _qt()
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            QtWidgets.QApplication.processEvents()
            time.sleep(0.01)
        self.settle()

    # ---- observation ------------------------------------------------------------------
    def rows(self):
        QtCore, _QtWidgets = _qt()
        tree = self.owner.glossary_tree
        out = []
        for i in range(tree.topLevelItemCount()):
            item = tree.topLevelItem(i)
            brush = item.background(1) if item.columnCount() > 1 else None
            out.append([[item.text(c) for c in range(item.columnCount())], item.data(0, QtCore.Qt.UserRole),
                        item.isHidden(),
                        None if brush is None else [brush.style().name, brush.color().name()]])
        return out

    def snapshot(self):
        owner = self.owner
        combo = owner.editor_file_combo
        filters = getattr(owner, "_glossary_column_filters", {}) or {}
        return _norm({
            "files": {k: v.decode("utf-8", "replace") for k, v in _tree_bytes(self.root).items()},
            "data": owner.current_glossary_data,
            "format": owner.current_glossary_format,
            "sections": getattr(owner, "current_glossary_sections", "<absent>"),
            "tracker": owner.current_gender_tracker_data,
            "tracker_path": owner.current_gender_tracker_path,
            "pending": owner._pending_gender_decisions,
            "variants": owner._gender_variants_pending_save,
            "fields": owner.glossary_column_fields,
            "baseline": sorted((str(k), v) for k, v in owner._original_translated_map.items()),
            "undo": owner._undo_stack, "redo": owner._redo_stack,
            "undo_enabled": [owner._undo_btn.isEnabled(), owner._redo_btn.isEnabled()],
            "filters": {k: sorted(v) for k, v in filters.items()},
            "stats": owner.stats_label.text(),
            "rows": self.rows(),
            "combo": [[combo.itemText(i), combo.itemData(i)] for i in range(combo.count())],
            "current": owner.editor_file_entry.text(),
            "find": [getattr(owner, "_last_find_text", None), getattr(owner, "_last_find_pos", None)],
            "config": {k: owner.config.get(k) for k in ("append_glossary", "manual_glossary_path",
                                                        "update_html_on_save")},
            "manual": [getattr(owner, "manual_glossary_path", None),
                       getattr(owner, "_glossary_editor_manual_source", None)],
            "env": dict(self.env),
            "logs": self.logs,
            "script": _SCRIPT.log,
            "saved_configs": self.saved_configs,
        }, self.root)

    # ---- operations -------------------------------------------------------------------
    def apply(self, op, d):
        QtCore, QtWidgets = _qt()
        owner = self.owner
        tree = owner.glossary_tree
        _SCRIPT.reset(d)
        with self.active():
            if op == "edit":
                item = tree.topLevelItem(d["row"])
                if item is not None:
                    owner._on_tree_double_click(item, d["col"])
            elif op in ("delete", "export"):
                tree.clearSelection()
                for row in d["rows"]:
                    item = tree.topLevelItem(row)
                    if item is not None:
                        item.setSelected(True)
                _button(self.parent, "Delete Selected" if op == "delete" else "Export Selection").click()
            elif op in ("save", "save_as", "reload", "clean", "dedupe", "trim", "filter", "convert",
                        "backup_settings", "undo", "redo", "load_manual", "about"):
                if op == "convert":
                    owner.config["glossary_use_legacy_csv"] = d["legacy"]
                if op == "load_manual":
                    owner.config["auto_glossary_mode"] = d["mode"]
                label = {"save": "Save", "save_as": "Save As...", "reload": "Reload",
                         "clean": "Clean Empty Fields", "dedupe": "Remove Duplicates", "trim": "Trim Entries",
                         "filter": "Filter Entries", "convert": "Convert Format", "backup_settings": "Backup Settings",
                         "undo": "↶ Undo", "redo": "↷ Redo", "load_manual": "\U0001F4C4 Load",
                         "about": "About Format"}[op]
                button = _button(self.parent, label)
                if button.isEnabled():
                    button.click()
            elif op == "find":
                from PySide6.QtGui import QKeySequence, QShortcut
                tree.setCurrentItem(tree.topLevelItem(d.get("current", -1)))
                find_key = QKeySequence(QKeySequence.Find)
                shortcut = next(s for s in owner.dialog.findChildren(QShortcut) if s.key() == find_key)
                shortcut.activated.emit()
            elif op == "hide_unused":
                owner.hide_unused_entries_checkbox.setChecked(d["on"])
            elif op == "column_filter":
                owner._glossary_column_filter_dismissed = None
                tree.header().sectionClicked.emit(d["col"])
            elif op == "clear_filters":
                owner._glossary_column_filters = {}
                owner._apply_glossary_column_filters()
            elif op == "resolve":
                item = tree.topLevelItem(d["row"])
                if item is not None:
                    owner._open_gender_resolution_for_item(item)
            elif op == "external":
                owner._editor_auto_reload_timer.timeout.emit()   # records the mtime baseline
                path = Path(owner.editor_file_entry.text())
                if path.is_file():
                    data = path.read_bytes()
                    path.write_bytes(data.replace(d["old"].encode("utf-8"), d["new"].encode("utf-8"), 1))
                    stat = path.stat()
                    os.utime(path, (stat.st_atime, stat.st_mtime + 30))
                owner._editor_auto_reload_timer.timeout.emit()
            else:
                raise AssertionError(op)
            self.settle()
            if op == "external":
                self.wait(1.7)


class _BoxesProxy:
    """QMessageBox for the frozen create_glossary_backup (it only asks after a failed backup)."""

    Yes, No = 16384, 65536

    def question(self, *args, **kwargs):
        _SCRIPT.log.append(("question", [a for a in args if isinstance(a, str)]))
        return self.Yes


# ---- operation choice (from the frozen side's state, applied to every side) -----------------------

OPS = ("edit", "edit", "edit", "save", "save", "delete", "clean", "dedupe", "trim", "filter", "find", "find",
       "undo", "undo", "redo", "hide_unused", "column_filter", "clear_filters", "resolve", "export", "convert",
       "reload", "backup_settings", "load_manual", "external", "save_as", "about")


def _decide(rng, side, step):
    QtCore, _QtWidgets = _qt()
    owner = side.owner
    tree = owner.glossary_tree
    count = tree.topLevelItemCount()
    fields = list(owner.glossary_column_fields or [])
    op = rng.choice(OPS)
    # A side may pin its first ops (the rng draw above still happens, so later steps are unchanged).
    forced = getattr(side, "_forced_ops", ())
    if step < len(forced):
        op = forced[step]
    if op == "save_as" and step < 4:
        op = "save"
    d = {"answers": [rng.choice(("yes", "yes", "no")), "yes", "yes"]}
    pick_row = (lambda: rng.randrange(count)) if count else (lambda: -1)
    if op == "edit":
        visible = [c for c in range(1, len(fields) + 1) if fields[c - 1] != "_section"] or [1]
        d.update(row=pick_row(), col=rng.choice(visible))
        col_key = fields[d["col"] - 1] if 0 < d["col"] <= len(fields) else ""
        current = tree.topLevelItem(d["row"]).text(d["col"]) if count else ""
        if col_key == "gender":
            d["value"] = rng.choice(("male", "FEMALE", "", "Unknown", current))
        else:
            d["value"] = rng.choice((current + " II", "", current.upper(), "Luna", current))
        d["decision"] = rng.choice(("auto", "male", "female"))
    elif op in ("delete", "export"):
        d["rows"] = sorted({pick_row() for _ in range(rng.randint(1, 3))})
        d["save_path"] = rng.choice(("export.json", "export.csv"))
    elif op == "trim":
        d["top_n"] = rng.choice((str(max(0, count - rng.randint(0, 5))), "x", str(count + 3), "-2"))
    elif op == "filter":
        types_ = sorted({str(e.get("type")) for e in (owner.current_glossary_data or [])
                         if isinstance(e, dict) and e.get("type")}) if isinstance(owner.current_glossary_data, list) else []
        d["keep"] = {t: rng.random() < 0.8 for t in types_}
        d["limits"] = {t: rng.choice(("", "", "3", "x")) for t in types_}
        d["search"] = rng.choice(("", "", "a", "the", "Lu"))
        d["gender_label"] = rng.choice(("All genders", "All genders", "Male only", "Female only", "Unknown only"))
    elif op == "find":
        texts = [tree.topLevelItem(i).text(c) for i in range(count) for c in range(1, tree.topLevelItem(i).columnCount())]
        words = [w for t in texts for w in t.split() if len(w) > 2][:200]
        d["find"] = rng.choice(words + ["zzqx", "Luna", "the"]) if words else rng.choice(("zzqx", "Luna"))
        d["replace"] = rng.choice(("Moon", "", d["find"].upper(), "Ren"))
        d["buttons"] = rng.choice((["Replace All"], ["Find Next", "Replace"], ["Find Next", "Find Next"],
                                   ["Replace"], ["Replace All", "Replace All"]))
        d["current"] = pick_row() if rng.random() < 0.7 else -1
    elif op == "hide_unused":
        d["on"] = not owner.hide_unused_entries_checkbox.isChecked()
    elif op == "column_filter":
        choices = [c for c in range(1, len(fields) + 1) if fields[c - 1] != "_section"]
        d["col"] = rng.choice(choices) if choices else 1
        d["uncheck"] = sorted({rng.randrange(6) for _ in range(rng.randint(0, 3))})
    elif op == "resolve":
        role = QtCore.Qt.UserRole + 20
        conflicts = [i for i in range(count)
                     if isinstance(tree.topLevelItem(i).data(0, role), dict)
                     and tree.topLevelItem(i).data(0, role).get("conflict")]
        d["row"] = rng.choice(conflicts) if conflicts else pick_row()
        d["decision"] = rng.choice(("auto", "male", "female"))
    elif op == "convert":
        d["legacy"] = rng.random() < 0.4
        d["save_path"] = rng.choice(("converted.csv", "SAME"))
    elif op == "load_manual":
        d["mode"] = rng.choice(("off_no_automap", "balanced"))
    elif op == "external":
        text = Path(owner.editor_file_entry.text()).read_text(encoding="utf-8", errors="replace") \
            if Path(owner.editor_file_entry.text()).is_file() else ""
        words = [w for w in re.findall(r"[A-Za-z]{4,}", text)][:100]
        d["old"] = rng.choice(words) if words else "zzqx"
        d["new"] = d["old"] + "x"
    elif op == "save_as":
        d["save_path"] = rng.choice(("renamed.json", "renamed.csv"))
    return op, d


def _resolve_save_path(d, side):
    if d.get("save_path") == "SAME":
        d = dict(d)
        d["save_path"] = str(Path(side.owner.editor_file_entry.text()).relative_to(side.root))
    return d


# ---- the GlossaryDocument (mobile) replay of one operation -----------------------------------------

class DocumentSide:
    """GlossaryDocument on a third copy, driven with the decisions the desktop side took."""

    def __init__(self, root, fixture, config, backups):
        self.root, self.fixture = root, fixture
        self.logs = []
        names = re.findall(r"= ([^\[\]:|\n\r]+?)(?: \[|:|\r?\n)", fixture.payload.decode("utf-8", "replace"))[:60]
        names += re.findall(r'"translated_name": "([^"]+)"', fixture.payload.decode("utf-8", "replace"))[:60]
        self.glossary = fixture.materialize(root, names or ["Luna"])
        self.source = root / "in" / f"{fixture.book}.epub"
        self.stamp = _Stamp()
        create_backup, clean_backups = _bind_backups(backups, self.stamp, _BoxesProxy())
        logs = self.logs

        def backup(doc, operation_name):
            adapter = types.SimpleNamespace(config=doc.config, current_glossary_data=doc.current_glossary_data,
                                            editor_file_entry=FakeEntry(doc.path), append_log=logs.append)
            adapter._clean_old_backups = lambda *a: clean_backups(adapter, *a)
            return create_backup(adapter, operation_name)

        owner = gd.EditorOwner(config)
        owner.enable_parallel_extraction_var = False
        owner.extraction_workers_var = 1
        self.doc = gd.GlossaryDocument(owner=owner, backup=backup, log=logs.append)
        self.hide_unused = False
        with self.active():
            self.doc.open_result = self.doc.load(str(self.glossary))

    @contextlib.contextmanager
    def active(self):
        os.environ["OUTPUT_DIRECTORY"] = str(self.root / "out")
        cwd = os.getcwd()
        os.chdir(self.root)
        try:
            yield self
        finally:
            os.chdir(cwd)

    def snapshot(self):
        doc = self.doc
        return _norm({
            "files": {k: v.decode("utf-8", "replace") for k, v in _tree_bytes(self.root).items()},
            "data": doc.current_glossary_data, "format": doc.current_glossary_format,
            "sections": getattr(doc, "current_glossary_sections", "<absent>"),
            "tracker": doc.current_gender_tracker_data, "tracker_path": doc.current_gender_tracker_path,
            "pending": doc._pending_gender_decisions, "variants": doc._gender_variants_pending_save,
            "fields": doc.glossary_column_fields,
            "baseline": sorted((str(k), v) for k, v in doc._original_translated_map.items()),
            "undo": doc._undo_stack, "redo": doc._redo_stack,
            "current": doc.path,
        }, self.root)

    def apply(self, op, d, desktop):
        """Replay ``op`` as the mobile UI would, using what the desktop dialogs showed (``desktop``:
        the frozen side right before the operation: rows, refs, fields, conflict state)."""
        doc = self.doc
        answers = list(d.get("answers", ()))
        with self.active():
            try:
                self._apply(op, d, desktop, doc, answers)
            except gd.GlossaryEditorError:
                pass

    def _apply(self, op, d, desktop, doc, answers):
        rows = desktop["rows"]
        fields = desktop["fields"]

        def ref(row):
            return rows[row][1] if 0 <= row < len(rows) else None

        if op == "edit":
            if not (0 <= d["row"] < len(rows)) or not (0 < d["col"] <= len(fields)):
                return
            if desktop["conflict"].get(d["row"]) and fields[d["col"] - 1] == "gender":
                doc.resolve_gender(ref(d["row"]), d["decision"])
            else:
                doc.edit_cell(ref(d["row"]), fields[d["col"] - 1], d["value"])
        elif op == "save":
            update = bool(doc.config.get("update_html_on_save", True))
            if update and doc.translated_changes() and answers and answers[0] == "no":
                return
            doc.save_edits(update_output_files=update)
        elif op == "delete":
            refs = [ref(r) for r in d["rows"] if 0 <= r < len(rows)]
            if refs and (not answers or answers[0] == "yes"):
                doc.delete(refs)
        elif op == "export":
            refs = [ref(r) for r in d["rows"] if 0 <= r < len(rows)]
            if refs:
                doc.export_selection(str(self.root / d["save_path"]), refs)
        elif op == "clean":
            doc.clean_empty_fields()
        elif op == "dedupe":
            doc.remove_duplicates()
        elif op == "trim":
            try:
                int(d["top_n"])
            except ValueError:
                return
            if doc.current_glossary_data:
                doc.trim(d["top_n"])
        elif op == "filter":
            if doc.current_glossary_data:
                gender = {"All genders": "all", "Male only": "Male", "Female only": "Female",
                          "Unknown only": "Unknown"}[d["gender_label"]]
                doc.apply_filter(kept_types=d["keep"], search_text=d["search"], gender_value=gender,
                                 type_limits=d["limits"])
        elif op == "find":
            view = [gd.EditorRow(texts, r) for texts, r, _hidden, _bg in rows]
            current = view[d["current"]] if 0 <= d.get("current", -1) < len(view) else None
            for name in d["buttons"]:
                if name == "Replace All":
                    count = doc.replace_all(d["find"], d["replace"], view)
                    if count == 0 and d["find"] and view:
                        if (answers.pop(0) if answers else "yes") == "yes":
                            doc.replace_in_output_files(d["find"], d["replace"])
                elif name == "Replace":
                    if current is not None:
                        doc.replace_in(current, d["find"], d["replace"])
                elif name == "Find Next":
                    idx = doc.find_next(d["find"], view)
                    if idx is not None:
                        current = view[idx]
        elif op == "undo":
            doc.undo()
        elif op == "redo":
            doc.redo()
        elif op == "resolve":
            if 0 <= d["row"] < len(rows) and desktop["conflict"].get(d["row"]):
                doc.resolve_gender(ref(d["row"]), d["decision"])
        elif op == "convert":
            if doc.current_glossary_data:
                doc.config["glossary_use_legacy_csv"] = d["legacy"]
                target = doc.path if d["save_path"] == "SAME" else str(self.root / d["save_path"])
                doc.convert(target)
        elif op == "reload":
            doc.load()
        elif op == "external":
            path = Path(doc.path)
            if path.is_file():
                data = path.read_bytes()
                path.write_bytes(data.replace(d["old"].encode("utf-8"), d["new"].encode("utf-8"), 1))
                stat = path.stat()
                os.utime(path, (stat.st_atime, stat.st_mtime + 30))
                doc.load()
        elif op == "backup_settings":
            if doc.current_glossary_data:
                doc._create_backup("manual")
        elif op == "load_manual":
            if d["mode"] == "off_no_automap" and (not answers or answers[0] == "yes"):
                epub = gd.manual_only_epub_path([str(self.source)], doc.config)
                if epub:
                    gd.copy_glossary_to_epub_output(doc.path, epub, doc.config, doc.log)
        elif op == "save_as":
            if doc.current_glossary_data:
                doc.save_as(str(self.root / d["save_path"]))
        # hide_unused / column_filter / clear_filters / about change only the desktop view


def _desktop_view(side):
    QtCore, _QtWidgets = _qt()
    tree = side.owner.glossary_tree
    role = QtCore.Qt.UserRole + 20
    conflict = {}
    for i in range(tree.topLevelItemCount()):
        status = tree.topLevelItem(i).data(0, role)
        conflict[i] = bool(isinstance(status, dict) and status.get("conflict"))
    return {"rows": side.rows(), "fields": list(side.owner.glossary_column_fields or []), "conflict": conflict}


def _dump_on_mismatch(got, expected, label):
    """With $PARITY_U6_DUMP_DIR set, write both sides of a mismatch there (debugging aid)."""
    folder = os.environ.get("PARITY_U6_DUMP_DIR")
    if not folder or got == expected:
        return
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", label)
    Path(folder).mkdir(parents=True, exist_ok=True)
    for name, value in (("expected", expected), ("got", got)):
        Path(folder, f"{safe}.{name}.json").write_text(json.dumps(value, ensure_ascii=False, indent=1),
                                                      encoding="utf-8")


_DOC_KEYS = ("files", "data", "format", "sections", "tracker", "tracker_path", "pending", "variants", "fields",
             "baseline", "undo", "redo", "current")


@pytest.fixture(scope="module")
def isolated_gm(qapp, tmp_path_factory):
    """(frozen, working tree) GlossaryManager_GUI loaded with ``__file__`` in an empty temp folder,
    so the editor's shared ``Glossary`` folder next to the module is not the user's src/Glossary."""
    app_dir = tmp_path_factory.mktemp("gm_app")
    fake_file = app_dir / "GlossaryManager_GUI.py"
    legacy = _load_gm_module(_git_text("src/GlossaryManager_GUI.py"), "_glossary_manager_u6_frozen_iso", fake_file)
    new = _load_gm_module(_current_text("GlossaryManager_GUI.py"), "_glossary_manager_u6_new_iso", fake_file)
    return legacy, new


@needs_qt
@pytest.mark.parametrize("fixture", EDITOR_FIXTURES, ids=[f.name for f in EDITOR_FIXTURES])
def test_editor_tab_matches_frozen_on_real_glossaries(fixture, tmp_path, monkeypatch, isolated_gm):
    """F + M: every editor operation through the frozen and the working-tree editor tabs (their
    buttons and dialogs) and through GlossaryDocument, on copies of a real glossary."""
    monkeypatch.setattr(os, "fsync", lambda _fd: None)
    _patch_qt_dialogs(monkeypatch)
    legacy_gm, new_gm = isolated_gm
    backups = _frozen_backup_methods()
    for run in range(EDITOR_RUNS):
        rng = random.Random(f"{SEED}:{fixture.name}:{run}")
        config = _rand_config(rng, owner_effective=True)
        config["update_html_on_save"] = rng.random() < 0.7
        config["glossary_auto_backup"] = rng.random() < 0.8
        if rng.random() < 0.3:
            config["glossary_skip_gender_tracking"] = False
        # The same last path parts on every side: the editor shows paths by their last 3 parts.
        legacy = EditorSide(legacy_gm, tmp_path / "legacy" / f"run{run}" / "ws", fixture, copy.deepcopy(config), backups)
        new = EditorSide(new_gm, tmp_path / "new" / f"run{run}" / "ws", fixture, copy.deepcopy(config), backups)
        doc = DocumentSide(tmp_path / "doc" / f"run{run}" / "ws", fixture, copy.deepcopy(config), backups)
        if fixture.name == "synthetic-token":
            # CI (no src/Glossary): the random draws alone never give redo / resolve an effect on the
            # synthetic glossaries, so the one fixture with a gender tracker starts with them.
            legacy._forced_ops = ("edit", "edit", "undo", "redo", "resolve", "resolve")
        expected = legacy.snapshot()
        assert new.snapshot() == expected, (fixture.name, run, "load")
        assert {k: doc.snapshot()[k] for k in _DOC_KEYS} == {k: expected[k] for k in _DOC_KEYS}, \
            (fixture.name, run, "load (document)")
        assert [r[0] for r in expected["rows"]] == [r.texts for r in doc.doc.rows()]
        history = []
        previous = expected
        for step in range(EDITOR_STEPS):
            op, d = _decide(rng, legacy, step)
            view = _desktop_view(legacy)
            history.append((op, {k: v for k, v in d.items() if k != "answers"}))
            legacy.apply(op, _resolve_save_path(d, legacy))
            expected = legacy.snapshot()
            _OP_RUNS[op] = _OP_RUNS.get(op, 0) + 1
            if any(expected[k] != previous[k] for k in ("files", "data", "undo", "redo", "rows", "filters")):
                _OP_EFFECTS[op] = _OP_EFFECTS.get(op, 0) + 1
            previous = expected
            new.apply(op, _resolve_save_path(d, new))
            got = new.snapshot()
            _dump_on_mismatch(got, expected, f"{fixture.name}-{run}-{step}-new")
            assert got == expected, (fixture.name, run, step, history[-3:],
                                     {k: (expected[k], got[k]) for k in expected if expected[k] != got[k]})
            doc.apply(op, d, view)
            mobile = doc.snapshot()
            diff = {k: (expected[k], mobile[k]) for k in _DOC_KEYS if expected[k] != mobile[k]}
            _dump_on_mismatch({k: mobile[k] for k in _DOC_KEYS}, {k: expected[k] for k in _DOC_KEYS},
                              f"{fixture.name}-{run}-{step}-doc")
            assert not diff, (fixture.name, run, step, history[-3:], diff)
            if op == "hide_unused" and d["on"] and not legacy.owner.stats_label.text().startswith(("No ", "Hide")):
                with doc.active():
                    used = doc.doc.used_rows(doc.doc.output_dir(str(doc.source)))
                shown = [r[1] for r in legacy.rows()]
                refs = [spec[1] for spec in gd.editor_row_specs(
                    doc.doc.glossary_column_fields, doc.doc.current_glossary_data,
                    doc.doc.current_glossary_format)[1] if spec[0] in set(used.get("used_rows", []))]
                assert refs == shown, (fixture.name, run, step)


#: Operations run / operations that changed files, data, history or the view (tier F).
_OP_RUNS = {}
_OP_EFFECTS = {}


def test_editor_operations_were_exercised():
    """Every editor operation changed something at least once in the tier F runs above."""
    if not _OP_RUNS:
        pytest.skip("tier F did not run in this session")
    # "about" only shows a box; "clear_filters" only acts after a column filter
    idle = sorted(op for op in set(OPS) - {"about", "clear_filters"} if not _OP_EFFECTS.get(op))
    assert not idle, (idle, _OP_RUNS, _OP_EFFECTS)


# ---- O: the owner values the editor reads ------------------------------------------------------

def test_editor_owner_matches_headless_owner(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    from _headless_env import headless_owner

    configs = [{}, {"custom_glossary_fields": ["fun fact"], "glossary_gender_noise_threshold": 25},
               {"custom_entry_types": {"skills": {"enabled": True, "has_gender": False}},
                "custom_field_description_removed": True, "enable_parallel_extraction": False,
                "extraction_workers": 6, "glossary_gender_tracking_bias": "male"}]
    for index, config in enumerate(configs):
        owner = gd.EditorOwner(copy.deepcopy(config))
        with headless_owner(tmp_path / f"o{index}", monkeypatch, copy.deepcopy(config)) as headless:
            for attr in ("custom_entry_types", "glossary_gender_noise_threshold_var",
                         "glossary_gender_tracking_bias_var", "enable_parallel_extraction_var",
                         "extraction_workers_var"):
                assert getattr(owner, attr) == getattr(headless, attr), (index, attr)
            assert owner.config.get("custom_glossary_fields", []) == headless.config.get("custom_glossary_fields", []), index


# ---- S: the Glossary Manager dialog's editor tab, offscreen ---------------------------------------

@needs_qt
def test_glossary_manager_editor_tab_smoke(tmp_path, monkeypatch, isolated_gm):
    """Build the editor tab of the working-tree Glossary Manager on a real glossary and use it."""
    if not EDITOR_FIXTURES:
        pytest.skip("no real glossary fixtures")
    monkeypatch.setattr(os, "fsync", lambda _fd: None)
    _patch_qt_dialogs(monkeypatch)
    fixture = EDITOR_FIXTURES[0]
    side = EditorSide(isolated_gm[1], tmp_path / "smoke", fixture, {"update_html_on_save": True},
                      _frozen_backup_methods())
    owner = side.owner
    assert owner.glossary_tree.topLevelItemCount() > 0
    assert owner.stats_label.text().startswith("Total entries:")
    assert owner.editor_file_entry.text() == str(side.glossary)
    first = owner.glossary_tree.topLevelItem(0)
    col = owner.glossary_column_fields.index("translated_name") + 1
    side.apply("edit", {"row": 0, "col": col, "value": first.text(col) + " Smoke", "answers": ["yes"]})
    side.apply("save", {"answers": ["yes", "yes"]})
    reloaded = gd.GlossaryDocument.open(str(side.glossary), owner.config)
    assert any(" Smoke" in str(e.get("translated_name", "")) for e in reloaded.current_glossary_data)


# =============================================================================================
# P: Glossary Manager prompt profiles (Balanced/Full, Minimal, Refinement)
# =============================================================================================

PROFILE_STATES = max(1, int(os.environ.get("PARITY_U6_PROFILE_RUNS", "60")))
PROFILE_STEPS = max(1, int(os.environ.get("PARITY_U6_PROFILE_STEPS", "14")))
PROFILE_KEYS = ("glossary_prompt_profiles", "active_glossary_prompt_profiles", "glossary_prompt_profile_defaults",
                "manual_glossary_prompt3", "manual_glossary_prompt", "unified_auto_glosary_prompt3")
REFINEMENT_KEYS = ("glossary_refinement_prompt_profiles", "glossary_refinement_prompt_profile_default",
                   "active_glossary_refinement_prompt_profile", "glossary_refinement_system_prompt",
                   "glossary_refinement_user_prompt")
PROFILE_NAMES = ("Alpha", "Beta", "New Profile #1", "Default", "default", "", "  Gamma  ", "Delta")
SAVE_LABEL = "\U0001f4be Save Profile"
DELETE_LABEL = "\U0001f5d1 Delete Profile"
PROMPT_TEXTS = ("Extract {language} names", "", "  padded  ", "Line one\nLine two", "x\x1fy", "Use {entries}")


def _rand_profile_config(rng, *, reachable):
    config = {}
    if rng.random() < 0.8:
        buckets = {}
        for key in ("balanced_full", "minimal"):
            if rng.random() < 0.85:
                names = [n for n in ("Alpha", "Beta", "Gamma", "New Profile #1") if rng.random() < 0.5]
                buckets[key] = {n: rng.choice(PROMPT_TEXTS) for n in names}
            elif not reachable:
                buckets[key] = rng.choice(("bad", None, ["x"]))
        if not reachable and rng.random() < 0.2 and isinstance(buckets.get("balanced_full", {}), dict):
            buckets.setdefault("balanced_full", {})["Default"] = "shadow"
        config["glossary_prompt_profiles"] = buckets if (reachable or rng.random() < 0.9) else "bad"
    if rng.random() < 0.7:
        config["active_glossary_prompt_profiles"] = {
            key: rng.choice(("Alpha", "Beta", "Gone", "")) for key in ("balanced_full", "minimal") if rng.random() < 0.7}
    if rng.random() < 0.6:
        config["glossary_prompt_profile_defaults"] = {
            key: rng.choice(PROMPT_TEXTS) for key in ("balanced_full", "minimal") if rng.random() < 0.7}
    for key in ("manual_glossary_prompt3", "unified_auto_glosary_prompt3"):
        if rng.random() < 0.6:
            config[key] = rng.choice(PROMPT_TEXTS)
    if rng.random() < 0.6:
        config["glossary_refinement_prompt_profiles"] = {
            n: {"system": rng.choice(PROMPT_TEXTS), "user": rng.choice(PROMPT_TEXTS)}
            for n in ("Alpha", "Beta", "Gamma") if rng.random() < 0.5}
        if not reachable and rng.random() < 0.3:
            config["glossary_refinement_prompt_profiles"]["Bad"] = "not a pair"
    if rng.random() < 0.5:
        config["glossary_refinement_prompt_profile_default"] = {"system": rng.choice(PROMPT_TEXTS)}
    if rng.random() < 0.5:
        config["active_glossary_refinement_prompt_profile"] = rng.choice(("Alpha", "Gone", ""))
    return config


def _profile_owner_attrs(rng):
    attrs = {}
    if rng.random() < 0.5:
        attrs["manual_glossary_prompt"] = rng.choice(PROMPT_TEXTS)
    if rng.random() < 0.5:
        attrs["unified_auto_glosary_prompt3"] = rng.choice(PROMPT_TEXTS)
    return attrs


@needs_qt
def test_prompt_profile_state_helpers_match_frozen(legacy_gm, new_gm):
    """Tier D: the profile config helpers (frozen methods vs wrappers vs shared functions)."""
    calls = ("ensure", "profiles_for", "active_for", "set_active", "default_text", "set_default", "is_default",
             "meta", "sync")
    for state in range(STATES):
        rng = random.Random(SEED * 41 + state)
        config = _rand_profile_config(rng, reachable=False)
        attrs = _profile_owner_attrs(rng)
        script = [(rng.choice(calls), rng.choice(("balanced_full", "minimal", "other")), rng.choice(PROFILE_NAMES),
                   rng.choice(PROMPT_TEXTS)) for _ in range(rng.randint(1, 8))]
        results = []
        for side in ("legacy", "new", "shared"):
            module = legacy_gm if side == "legacy" else new_gm
            owner = _owner(module.GlossaryManagerMixin, config=copy.deepcopy(config), **copy.deepcopy(attrs))
            from PySide6.QtWidgets import QTextEdit
            out = []
            for call, key, name, text in script:
                if call == "sync":
                    editor = QTextEdit()
                    editor.setPlainText(text)
                    owner._glossary_prompt_profile_widgets = {key: {"editor": editor}}
                if side == "shared":
                    fn = {
                        "ensure": lambda: gd.ensure_glossary_prompt_profiles(owner),
                        "profiles_for": lambda: gd.glossary_prompt_profiles_for(owner, key),
                        "active_for": lambda: gd.active_glossary_prompt_profile_for(owner, key),
                        "set_active": lambda: gd.set_active_glossary_prompt_profile(owner, key, name),
                        "default_text": lambda: gd.default_glossary_prompt_profile_text(owner, key),
                        "set_default": lambda: gd.set_default_glossary_prompt_profile_text(owner, key, text),
                        "is_default": lambda: gd.is_default_glossary_prompt_profile(name),
                        "meta": lambda: gd.glossary_prompt_profile_meta(key),
                        "sync": lambda: gd.store_glossary_prompt_current(owner, key, text.strip()),
                    }[call]
                else:
                    fn = {
                        "ensure": owner._ensure_glossary_prompt_profiles,
                        "profiles_for": lambda: owner._glossary_prompt_profiles_for(key),
                        "active_for": lambda: owner._active_glossary_prompt_profile_for(key),
                        "set_active": lambda: owner._set_active_glossary_prompt_profile(key, name),
                        "default_text": lambda: owner._default_glossary_prompt_profile_text(key),
                        "set_default": lambda: owner._set_default_glossary_prompt_profile_text(key, text),
                        "is_default": lambda: owner._is_default_glossary_prompt_profile(name),
                        "meta": lambda: owner._glossary_prompt_profile_meta(key),
                        "sync": lambda: owner._sync_glossary_prompt_profile_current_prompt(key),
                    }[call]
                out.append(_run(fn))
            out.append({k: getattr(owner, k, "<absent>") for k in ("manual_glossary_prompt", "unified_auto_glosary_prompt3")})
            out.append(owner.config)
            results.append(_norm(out))
        assert results[1] == results[0], state
        assert results[2] == results[0], state
    for module in (legacy_gm, new_gm):
        owner = _owner(module.GlossaryManagerMixin, config={})
        assert owner._default_glossary_refinement_system_prompt() == gd.default_glossary_refinement_system_prompt()
        assert owner._default_glossary_refinement_user_prompt() == gd.default_glossary_refinement_user_prompt()


class _ProfileBoxes:
    """QMessageBox stand-in for the profile controls: records boxes, answers questions."""

    def __init__(self):
        self.log = []
        self.answers = []

    def install(self, monkeypatch):
        from PySide6.QtWidgets import QMessageBox
        boxes = self

        def static(kind):
            def show(*args, **kwargs):
                texts = [a for a in args if isinstance(a, str)]
                boxes.log.append((kind, texts[:2]))
                if kind == "question":
                    return QMessageBox.Yes if (boxes.answers.pop(0) if boxes.answers else True) else QMessageBox.No
                return QMessageBox.Ok
            return show

        def exec_(box):
            answer = boxes.answers.pop(0) if boxes.answers else True
            boxes.log.append(("question", [box.windowTitle(), box.text()]))
            return QMessageBox.Yes if answer else QMessageBox.No

        for kind in ("critical", "warning", "information", "question"):
            monkeypatch.setattr(QMessageBox, kind, staticmethod(static(kind)))
        monkeypatch.setattr(QMessageBox, "exec", exec_)


def _desktop_profile_owner(module, config, attrs, persist_results, logs):
    cls = type("ProfileOwner", (module.GlossaryManagerMixin,), {
        "save_config": lambda self, show_message=True: (persist_results.pop(0) if persist_results else None),
        "append_log": lambda self, message: logs.append(message),
    })
    owner = cls()
    owner.config = config
    for key, value in attrs.items():
        setattr(owner, key, value)
    return owner


def _button_in(widget, text):
    from PySide6.QtWidgets import QPushButton
    return next(b for b in widget.findChildren(QPushButton) if b.text() == text)


@needs_qt
@pytest.mark.parametrize("profile_key", ["balanced_full", "minimal"])
def test_glossary_prompt_profiles_match_frozen_controls(profile_key, monkeypatch, legacy_gm, new_gm):
    """Tier P: the desktop profile row (frozen and working tree, real widgets) vs
    GlossaryPromptProfiles over random action sequences."""
    from PySide6.QtWidgets import QTextEdit

    boxes = _ProfileBoxes()
    boxes.install(monkeypatch)
    seen = set()
    for state in range(PROFILE_STATES):
        rng = random.Random(f"{SEED}:{profile_key}:{state}")
        config = _rand_profile_config(rng, reachable=True)
        attrs = _profile_owner_attrs(rng)
        initial = rng.choice(PROMPT_TEXTS)
        ops = []
        for _ in range(PROFILE_STEPS):
            op = rng.choice(("select", "select", "edit", "edit", "name", "new", "save", "save", "delete", "fail_save"))
            ops.append((op, rng.choice(PROFILE_NAMES + ("Gone",)), rng.choice(PROMPT_TEXTS), rng.random() < 0.7))
        traces = []
        names_before = []
        for side in ("legacy", "new", "mobile"):
            persist, logs, trace = [], [], []
            boxes.log.clear()
            if side == "mobile":
                owner = types.SimpleNamespace(config=copy.deepcopy(config), **copy.deepcopy(attrs))
                profiles = gd.GlossaryPromptProfiles(
                    owner, profile_key, initial, persist=lambda: persist.pop(0) if persist else None, log=logs.append)
            else:
                module = legacy_gm if side == "legacy" else new_gm
                owner = _desktop_profile_owner(module, copy.deepcopy(config), copy.deepcopy(attrs), persist, logs)
                editor = QTextEdit()
                row = owner._create_glossary_prompt_profile_controls(profile_key, editor)
                editor.setPlainText(owner._sep_for_display(initial))
                owner._apply_active_glossary_prompt_profile(profile_key)
                combo = owner._glossary_prompt_profile_widgets[profile_key]["combo"]
            for step, (op, name, text, yes) in enumerate(ops):
                boxes.answers = [yes]
                if op == "fail_save":
                    persist.append(False)
                    op = "save"
                if side == "mobile":
                    # the name the desktop row held when the action ran (a mobile name field)
                    current = names_before[step]
                    if op == "select":
                        profiles.select(name)
                    elif op == "edit":
                        profiles.stage(current, text)
                    elif op == "new":
                        profiles.new()
                    elif op == "save":
                        box = profiles.save(current, profiles.text)
                        if box:
                            boxes.log.append(("warning", list(box)))
                    elif op == "delete":
                        box = profiles.delete(current, confirm=lambda _n: boxes.log.append(
                            ("question", ["Delete Profile", f"Delete glossary prompt profile '{_n}'?"])) or yes)
                        if box:
                            boxes.log.append(("warning", list(box)))
                    shown = str(profiles.text or "").strip()
                else:
                    if side == "legacy":
                        names_before.append(combo.currentText())
                    if op == "select":
                        # typing a name + Enter runs the row's select handler; Qt would also add
                        # the typed text to this combo's list (no NoInsert policy, DISCREPANCIES U6)
                        combo.setEditText(name)
                        owner._on_glossary_prompt_profile_selected(profile_key)
                    elif op == "edit":
                        editor.setPlainText(owner._sep_for_display(text))
                    elif op == "name":
                        combo.setEditText(name)
                    elif op == "new":
                        _button_in(row, "+ New Profile").click()
                    elif op == "save":
                        _button_in(row, SAVE_LABEL).click()
                    elif op == "delete":
                        _button_in(row, DELETE_LABEL).click()
                    shown = owner._glossary_prompt_text(editor)
                persist.clear()
                trace.append(_norm({
                    "op": op, "config": {k: owner.config.get(k, "<absent>") for k in PROFILE_KEYS},
                    "attrs": {k: getattr(owner, k, "<absent>") for k in ("manual_glossary_prompt",
                                                                      "unified_auto_glosary_prompt3")},
                    "shown": shown, "boxes": list(boxes.log), "logs": list(logs),
                    "selection": None if side == "mobile" else combo.currentText(),
                }))
                boxes.log.clear()
            traces.append(trace)
        for step, (legacy, new, mobile) in enumerate(zip(*traces)):
            assert new == legacy, (profile_key, state, step, ops[:step + 1])
            legacy = dict(legacy, selection=None)
            assert mobile == legacy, (profile_key, state, step, ops[:step + 1],
                                      {k: (legacy[k], mobile[k]) for k in legacy if legacy[k] != mobile[k]})
            seen.update(box[1][0] for box in legacy["boxes"] if box[1])
            seen.update(line.split(":")[0] for line in legacy["logs"])
    expected = {"Save Failed", "Default Profile", "Profile Not Found", "Delete Profile",
                "\u2705 Saved glossary prompt profile", "\u2705 Created glossary prompt profile",
                "\U0001f5d1\ufe0f Deleted glossary prompt profile", "\u2705 Saved default glossary prompt profile"}
    if PROFILE_STATES >= 30:
        assert expected <= seen, sorted(expected - seen)


@needs_qt
def test_refinement_prompt_profiles_match_frozen_controls(monkeypatch, legacy_gm, new_gm):
    """Tier P: the refinement profile row (system + user prompt pair) vs RefinementPromptProfiles."""
    from PySide6.QtWidgets import QComboBox, QTextEdit

    boxes = _ProfileBoxes()
    boxes.install(monkeypatch)
    seen = set()
    for state in range(PROFILE_STATES):
        rng = random.Random(f"{SEED}:refinement:{state}")
        config = _rand_profile_config(rng, reachable=False)
        initial = {"system": rng.choice(PROMPT_TEXTS), "user": rng.choice(PROMPT_TEXTS)}
        ops = []
        for _ in range(PROFILE_STEPS):
            op = rng.choice(("select", "select", "edit_system", "edit_user", "name", "new", "save", "save",
                             "delete", "fail_save", "fail_new"))
            ops.append((op, rng.choice(PROFILE_NAMES + ("Gone",)), rng.choice(PROMPT_TEXTS), rng.random() < 0.7))
        traces = []
        names_before = []
        for side in ("legacy", "new", "mobile"):
            persist, logs, trace = [], [], []
            boxes.log.clear()
            if side == "mobile":
                owner = types.SimpleNamespace(config=copy.deepcopy(config))
                profiles = gd.RefinementPromptProfiles(
                    owner, initial["system"], initial["user"],
                    persist=lambda: persist.pop(0) if persist else None, log=logs.append)
            else:
                module = legacy_gm if side == "legacy" else new_gm
                owner = _desktop_profile_owner(module, copy.deepcopy(config), {}, persist, logs)
                system_editor, user_editor = QTextEdit(), QTextEdit()
                system_editor.setPlainText(owner._sep_for_display(initial["system"]))
                user_editor.setPlainText(owner._sep_for_display(initial["user"]))
                row = owner._create_refinement_prompt_profile_controls(system_editor, user_editor)
                combo = row.findChild(QComboBox, "refinement_prompt_profile_combo")
            for step, (op, name, text, yes) in enumerate(ops):
                boxes.answers = [yes]
                if op in ("fail_save", "fail_new"):
                    persist.append(False)
                    op = op[len("fail_"):]
                if side == "mobile":
                    current = names_before[step]
                    if op == "select":
                        profiles.select(name)
                    elif op in ("edit_system", "edit_user"):
                        pair = dict(profiles.pair)
                        pair[op[len("edit_"):]] = text
                        profiles.stage(current, pair)
                    elif op == "new":
                        box = profiles.new()
                        if box:
                            boxes.log.append(("warning", list(box)))
                    elif op == "save":
                        box = profiles.save(current, profiles.pair)
                        if box:
                            boxes.log.append(("warning", list(box)))
                    elif op == "delete":
                        box = profiles.delete(current, confirm=lambda _n: boxes.log.append(
                            ("question", ["Delete Profile", f"Delete refinement prompt profile '{_n}'?"])) or yes)
                        if box:
                            boxes.log.append(("warning", list(box)))
                    shown = {k: v.strip() for k, v in profiles.pair.items()}
                else:
                    if side == "legacy":
                        names_before.append(combo.currentText())
                    if op == "select":
                        combo.setEditText(name)
                        combo.lineEdit().returnPressed.emit()
                    elif op == "edit_system":
                        system_editor.setPlainText(owner._sep_for_display(text))
                    elif op == "edit_user":
                        user_editor.setPlainText(owner._sep_for_display(text))
                    elif op == "name":
                        combo.setEditText(name)
                    elif op == "new":
                        _button_in(row, "+ New Profile").click()
                    elif op == "save":
                        _button_in(row, SAVE_LABEL).click()
                    elif op == "delete":
                        _button_in(row, DELETE_LABEL).click()
                    shown = {"system": owner._glossary_prompt_text(system_editor),
                             "user": owner._glossary_prompt_text(user_editor)}
                persist.clear()
                trace.append(_norm({
                    "op": op, "config": {k: owner.config.get(k, "<absent>") for k in REFINEMENT_KEYS},
                    "attrs": {k: getattr(owner, k, "<absent>") for k in ("glossary_refinement_system_prompt",
                                                                      "glossary_refinement_user_prompt")},
                    "shown": shown, "boxes": list(boxes.log), "logs": list(logs),
                    "selection": None if side == "mobile" else combo.currentText(),
                }))
                boxes.log.clear()
            traces.append(trace)
        for step, (legacy, new, mobile) in enumerate(zip(*traces)):
            assert new == legacy, (state, step, ops[:step + 1])
            legacy = dict(legacy, selection=None)
            assert mobile == legacy, (state, step, ops[:step + 1],
                                      {k: (legacy[k], mobile[k]) for k in legacy if legacy[k] != mobile[k]})
            seen.update(box[1][0] for box in legacy["boxes"] if box[1])
            seen.update(line.split(":")[0] for line in legacy["logs"])
    expected = {"Save Failed", "Default Profile", "Profile Not Found", "Delete Profile",
                "\u2705 Saved refinement prompt profile", "\u2705 Created refinement prompt profile",
                "\U0001f5d1\ufe0f Deleted refinement prompt profile"}
    if PROFILE_STATES >= 30:
        assert expected <= seen, sorted(expected - seen)


# =============================================================================================
# G: goldens (synthetic glossaries, no Qt): the frozen desktop editor's bytes, pinned
# =============================================================================================

def _golden_entries():
    return [
        {"type": "character", "raw_name": "루나", "translated_name": "Luna", "gender": "female",
         "description": "A witch (rank: S)", "_section": "CHARACTERS"},
        {"type": "character", "raw_name": "카이", "translated_name": "Kai", "gender": "Male",
         "fun fact": "x: y", "_section": "CHARACTERS"},
        {"type": "terms", "raw_name": "마나", "translated_name": "Mana", "gender": "", "description": "energy | flow"},
        {"type": "titles", "raw_name": "공작", "translated_name": "Duke", "gender": "male"},
        {"type": "skills", "raw_name": "검술", "translated_name": "Swordsmanship", "rank": "A", "_section": "SKILLS"},
        {"type": "character", "raw_name": "", "translated_name": "Nameless", "gender": "unknown"},
    ]


def _golden_cases():
    """(case id, format, data, sections or None, file name, config) for save / convert."""
    cases = []
    for fmt in ("token_csv", "list"):
        for name in ("Book_glossary.csv", "Book_glossary.json"):
            for sections in (None, [], ["CHARACTERS", "TERMS"]):
                for fields in ([], ["description", "fun fact"]):
                    for legacy_csv in (False, True):
                        cases.append((f"{fmt}-{name}-{sections}-{fields}-{legacy_csv}", fmt, _golden_entries(),
                                      sections, name, {"custom_glossary_fields": fields,
                                                       "glossary_use_legacy_csv": legacy_csv}))
    cases.append(("dict", "dict", {"entries": {"루나": "Luna", "카이": "Kai"}}, None, "Book_glossary.json",
                  {"custom_glossary_fields": ["description"]}))
    return cases


def _golden_outputs(tmp, case, *, save, convert):
    """sha256 of the bytes ``save(owner)`` / ``convert(owner, csv_path)`` write, per case."""
    import hashlib

    case_id, fmt, data, sections, name, config = case
    out = {}
    for action, run in (("save", save), ("convert", convert)):
        root = tmp / f"{abs(hash(case_id)) % 10 ** 8}-{action}"
        root.mkdir(parents=True, exist_ok=True)
        path = root / name
        path.write_text("old", encoding="utf-8")
        owner = gd.GlossaryDocument(config)
        owner.path = str(path)
        owner.current_glossary_data = copy.deepcopy(data)
        owner.current_glossary_format = fmt
        if sections is None:
            del owner.current_glossary_sections
        else:
            owner.current_glossary_sections = list(sections)
        target = path if action == "save" else root / "converted.csv"
        outcome = _run(run, owner, str(target))
        written = target.read_bytes() if target.exists() else None
        if written is not None and target.suffix == ".json":
            # JSON is written in text mode (json.dump, no newline=) on both sides: CRLF on Windows,
            # LF on Linux CI. The CSV writers pass newline='' and write the same bytes everywhere.
            written = written.replace(b"\r\n", b"\n")
        digest = hashlib.sha256(written).hexdigest()[:16] if written is not None else None
        out[action] = [outcome[0], digest, getattr(owner, "current_glossary_sections", None)]
    return out


def _document_save(owner, path):
    return gd.save_document(owner, path, owner.config)


def _document_convert(owner, path):
    return gd.convert_to_csv(owner, path, owner.config)


#: Pinned outputs of the frozen desktop editor (verified by test_goldens_are_the_frozen_desktop_output).
GOLDEN_SHA256 = {
    'token_csv-Book_glossary.csv-None-[]-False': {'save': ['ok', '6799f94260324fea', None], 'convert': ['ok', '6799f94260324fea', None]},
    'token_csv-Book_glossary.csv-None-[]-True': {'save': ['ok', '6799f94260324fea', None], 'convert': ['ok', 'cf3b17311b9bfb0b', None]},
    "token_csv-Book_glossary.csv-None-['description', 'fun fact']-False": {'save': ['ok', '6799f94260324fea', None], 'convert': ['ok', '6799f94260324fea', None]},
    "token_csv-Book_glossary.csv-None-['description', 'fun fact']-True": {'save': ['ok', '6799f94260324fea', None], 'convert': ['ok', '71314486a5ac698c', None]},
    'token_csv-Book_glossary.csv-[]-[]-False': {'save': ['ok', '6799f94260324fea', []], 'convert': ['ok', '6799f94260324fea', []]},
    'token_csv-Book_glossary.csv-[]-[]-True': {'save': ['ok', '6799f94260324fea', []], 'convert': ['ok', 'cf3b17311b9bfb0b', []]},
    "token_csv-Book_glossary.csv-[]-['description', 'fun fact']-False": {'save': ['ok', '6799f94260324fea', []], 'convert': ['ok', '6799f94260324fea', []]},
    "token_csv-Book_glossary.csv-[]-['description', 'fun fact']-True": {'save': ['ok', '6799f94260324fea', []], 'convert': ['ok', '71314486a5ac698c', []]},
    "token_csv-Book_glossary.csv-['CHARACTERS', 'TERMS']-[]-False": {'save': ['ok', '6799f94260324fea', ['CHARACTERS', 'TERMS', 'TITLES', 'SKILLS']], 'convert': ['ok', '6799f94260324fea', ['CHARACTERS', 'TERMS', 'TITLES', 'SKILLS']]},
    "token_csv-Book_glossary.csv-['CHARACTERS', 'TERMS']-[]-True": {'save': ['ok', '6799f94260324fea', ['CHARACTERS', 'TERMS', 'TITLES', 'SKILLS']], 'convert': ['ok', 'cf3b17311b9bfb0b', ['CHARACTERS', 'TERMS']]},
    "token_csv-Book_glossary.csv-['CHARACTERS', 'TERMS']-['description', 'fun fact']-False": {'save': ['ok', '6799f94260324fea', ['CHARACTERS', 'TERMS', 'TITLES', 'SKILLS']], 'convert': ['ok', '6799f94260324fea', ['CHARACTERS', 'TERMS', 'TITLES', 'SKILLS']]},
    "token_csv-Book_glossary.csv-['CHARACTERS', 'TERMS']-['description', 'fun fact']-True": {'save': ['ok', '6799f94260324fea', ['CHARACTERS', 'TERMS', 'TITLES', 'SKILLS']], 'convert': ['ok', '71314486a5ac698c', ['CHARACTERS', 'TERMS']]},
    'token_csv-Book_glossary.json-None-[]-False': {'save': ['ok', 'a1db622e844dc0d6', None], 'convert': ['ok', '6799f94260324fea', None]},
    'token_csv-Book_glossary.json-None-[]-True': {'save': ['ok', 'a1db622e844dc0d6', None], 'convert': ['ok', 'cf3b17311b9bfb0b', None]},
    "token_csv-Book_glossary.json-None-['description', 'fun fact']-False": {'save': ['ok', 'a1db622e844dc0d6', None], 'convert': ['ok', '6799f94260324fea', None]},
    "token_csv-Book_glossary.json-None-['description', 'fun fact']-True": {'save': ['ok', 'a1db622e844dc0d6', None], 'convert': ['ok', '71314486a5ac698c', None]},
    'token_csv-Book_glossary.json-[]-[]-False': {'save': ['ok', 'a1db622e844dc0d6', []], 'convert': ['ok', '6799f94260324fea', []]},
    'token_csv-Book_glossary.json-[]-[]-True': {'save': ['ok', 'a1db622e844dc0d6', []], 'convert': ['ok', 'cf3b17311b9bfb0b', []]},
    "token_csv-Book_glossary.json-[]-['description', 'fun fact']-False": {'save': ['ok', 'a1db622e844dc0d6', []], 'convert': ['ok', '6799f94260324fea', []]},
    "token_csv-Book_glossary.json-[]-['description', 'fun fact']-True": {'save': ['ok', 'a1db622e844dc0d6', []], 'convert': ['ok', '71314486a5ac698c', []]},
    "token_csv-Book_glossary.json-['CHARACTERS', 'TERMS']-[]-False": {'save': ['ok', 'a1db622e844dc0d6', ['CHARACTERS', 'TERMS']], 'convert': ['ok', '6799f94260324fea', ['CHARACTERS', 'TERMS', 'TITLES', 'SKILLS']]},
    "token_csv-Book_glossary.json-['CHARACTERS', 'TERMS']-[]-True": {'save': ['ok', 'a1db622e844dc0d6', ['CHARACTERS', 'TERMS']], 'convert': ['ok', 'cf3b17311b9bfb0b', ['CHARACTERS', 'TERMS']]},
    "token_csv-Book_glossary.json-['CHARACTERS', 'TERMS']-['description', 'fun fact']-False": {'save': ['ok', 'a1db622e844dc0d6', ['CHARACTERS', 'TERMS']], 'convert': ['ok', '6799f94260324fea', ['CHARACTERS', 'TERMS', 'TITLES', 'SKILLS']]},
    "token_csv-Book_glossary.json-['CHARACTERS', 'TERMS']-['description', 'fun fact']-True": {'save': ['ok', 'a1db622e844dc0d6', ['CHARACTERS', 'TERMS']], 'convert': ['ok', '71314486a5ac698c', ['CHARACTERS', 'TERMS']]},
    'list-Book_glossary.csv-None-[]-False': {'save': ['ok', '09533fa9951518be', None], 'convert': ['ok', '6799f94260324fea', None]},
    'list-Book_glossary.csv-None-[]-True': {'save': ['ok', '09533fa9951518be', None], 'convert': ['ok', 'cf3b17311b9bfb0b', None]},
    "list-Book_glossary.csv-None-['description', 'fun fact']-False": {'save': ['ok', '09533fa9951518be', None], 'convert': ['ok', '6799f94260324fea', None]},
    "list-Book_glossary.csv-None-['description', 'fun fact']-True": {'save': ['ok', '09533fa9951518be', None], 'convert': ['ok', '71314486a5ac698c', None]},
    'list-Book_glossary.csv-[]-[]-False': {'save': ['ok', '09533fa9951518be', []], 'convert': ['ok', '6799f94260324fea', []]},
    'list-Book_glossary.csv-[]-[]-True': {'save': ['ok', '09533fa9951518be', []], 'convert': ['ok', 'cf3b17311b9bfb0b', []]},
    "list-Book_glossary.csv-[]-['description', 'fun fact']-False": {'save': ['ok', '09533fa9951518be', []], 'convert': ['ok', '6799f94260324fea', []]},
    "list-Book_glossary.csv-[]-['description', 'fun fact']-True": {'save': ['ok', '09533fa9951518be', []], 'convert': ['ok', '71314486a5ac698c', []]},
    "list-Book_glossary.csv-['CHARACTERS', 'TERMS']-[]-False": {'save': ['ok', '09533fa9951518be', ['CHARACTERS', 'TERMS']], 'convert': ['ok', '6799f94260324fea', ['CHARACTERS', 'TERMS', 'TITLES', 'SKILLS']]},
    "list-Book_glossary.csv-['CHARACTERS', 'TERMS']-[]-True": {'save': ['ok', '09533fa9951518be', ['CHARACTERS', 'TERMS']], 'convert': ['ok', 'cf3b17311b9bfb0b', ['CHARACTERS', 'TERMS']]},
    "list-Book_glossary.csv-['CHARACTERS', 'TERMS']-['description', 'fun fact']-False": {'save': ['ok', '09533fa9951518be', ['CHARACTERS', 'TERMS']], 'convert': ['ok', '6799f94260324fea', ['CHARACTERS', 'TERMS', 'TITLES', 'SKILLS']]},
    "list-Book_glossary.csv-['CHARACTERS', 'TERMS']-['description', 'fun fact']-True": {'save': ['ok', '09533fa9951518be', ['CHARACTERS', 'TERMS']], 'convert': ['ok', '71314486a5ac698c', ['CHARACTERS', 'TERMS']]},
    'list-Book_glossary.json-None-[]-False': {'save': ['ok', 'a1db622e844dc0d6', None], 'convert': ['ok', '6799f94260324fea', None]},
    'list-Book_glossary.json-None-[]-True': {'save': ['ok', 'a1db622e844dc0d6', None], 'convert': ['ok', 'cf3b17311b9bfb0b', None]},
    "list-Book_glossary.json-None-['description', 'fun fact']-False": {'save': ['ok', 'a1db622e844dc0d6', None], 'convert': ['ok', '6799f94260324fea', None]},
    "list-Book_glossary.json-None-['description', 'fun fact']-True": {'save': ['ok', 'a1db622e844dc0d6', None], 'convert': ['ok', '71314486a5ac698c', None]},
    'list-Book_glossary.json-[]-[]-False': {'save': ['ok', 'a1db622e844dc0d6', []], 'convert': ['ok', '6799f94260324fea', []]},
    'list-Book_glossary.json-[]-[]-True': {'save': ['ok', 'a1db622e844dc0d6', []], 'convert': ['ok', 'cf3b17311b9bfb0b', []]},
    "list-Book_glossary.json-[]-['description', 'fun fact']-False": {'save': ['ok', 'a1db622e844dc0d6', []], 'convert': ['ok', '6799f94260324fea', []]},
    "list-Book_glossary.json-[]-['description', 'fun fact']-True": {'save': ['ok', 'a1db622e844dc0d6', []], 'convert': ['ok', '71314486a5ac698c', []]},
    "list-Book_glossary.json-['CHARACTERS', 'TERMS']-[]-False": {'save': ['ok', 'a1db622e844dc0d6', ['CHARACTERS', 'TERMS']], 'convert': ['ok', '6799f94260324fea', ['CHARACTERS', 'TERMS', 'TITLES', 'SKILLS']]},
    "list-Book_glossary.json-['CHARACTERS', 'TERMS']-[]-True": {'save': ['ok', 'a1db622e844dc0d6', ['CHARACTERS', 'TERMS']], 'convert': ['ok', 'cf3b17311b9bfb0b', ['CHARACTERS', 'TERMS']]},
    "list-Book_glossary.json-['CHARACTERS', 'TERMS']-['description', 'fun fact']-False": {'save': ['ok', 'a1db622e844dc0d6', ['CHARACTERS', 'TERMS']], 'convert': ['ok', '6799f94260324fea', ['CHARACTERS', 'TERMS', 'TITLES', 'SKILLS']]},
    "list-Book_glossary.json-['CHARACTERS', 'TERMS']-['description', 'fun fact']-True": {'save': ['ok', 'a1db622e844dc0d6', ['CHARACTERS', 'TERMS']], 'convert': ['ok', '71314486a5ac698c', ['CHARACTERS', 'TERMS']]},
    'dict': {'save': ['ok', '15fcace5a9cccc54', None], 'convert': ['ok', '77fbc3aac7e94c7a', None]},
}


def test_goldens_match_glossary_document(tmp_path):
    """Runs without Qt (CI): glossary_document writes the pinned desktop bytes."""
    pytest.importorskip("extract_glossary_from_epub")
    os.environ["GLOSSARY_SKIP_GENDER_TRACKING"] = "1"
    for case in _golden_cases():
        got = _golden_outputs(tmp_path, case, save=_document_save, convert=_document_convert)
        assert got == GOLDEN_SHA256[case[0]], case[0]


@needs_qt
def test_goldens_are_the_frozen_desktop_output(tmp_path, monkeypatch, legacy_gm):
    """The pinned hashes are what the frozen desktop closures write."""
    from PySide6.QtWidgets import QFileDialog, QMessageBox

    os.environ["GLOSSARY_SKIP_GENDER_TRACKING"] = "1"
    legacy_text = _git_text("src/GlossaryManager_GUI.py")
    target = {}
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (target["path"], "")))
    for kind in ("critical", "information", "warning"):
        monkeypatch.setattr(QMessageBox, kind, staticmethod(lambda *a, **k: QMessageBox.Ok))

    def frozen_save(owner, path):
        legacy_owner = _owner(_Owner, config=owner.config, editor_file_entry=FakeEntry(path),
                              current_glossary_data=owner.current_glossary_data,
                              current_glossary_format=owner.current_glossary_format,
                              current_gender_tracker_data=None, current_gender_tracker_path="",
                              _pending_gender_decisions={}, _gender_variants_pending_save=0)
        if hasattr(owner, "current_glossary_sections"):
            legacy_owner.current_glossary_sections = owner.current_glossary_sections
        fn = _closure(legacy_gm, legacy_text, ("_setup_glossary_editor_tab", "save_current_glossary"),
                      self=legacy_owner, parent=None, QMessageBox=BoxRecorder())
        result = fn()
        owner.current_glossary_data = legacy_owner.current_glossary_data
        return result

    def frozen_convert(owner, path):
        legacy_owner = _owner(legacy_gm.GlossaryManagerMixin, config=owner.config, dialog=None, logs=[],
                              current_glossary_data=owner.current_glossary_data,
                              current_glossary_format=owner.current_glossary_format,
                              editor_file_entry=FakeEntry(owner.path))
        if hasattr(owner, "current_glossary_sections"):
            legacy_owner.current_glossary_sections = owner.current_glossary_sections
        legacy_owner.append_log = legacy_owner.logs.append
        legacy_owner.create_glossary_backup = lambda _op: True
        target["path"] = path
        legacy_owner.convert_glossary_format(lambda: None)
        return "token-efficient" if not owner.config.get("glossary_use_legacy_csv") else "legacy CSV"

    mismatched = []
    for case in _golden_cases():
        got = _golden_outputs(tmp_path, case, save=frozen_save, convert=frozen_convert)
        if os.environ.get("PARITY_U6_PRINT_GOLDENS"):
            print(f"    {case[0]!r}: {got!r},")
        if got != GOLDEN_SHA256.get(case[0]):
            mismatched.append(case[0])
    assert not mismatched, mismatched
