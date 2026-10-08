#!/usr/bin/env python3
"""Generate ``src/settings_schema_data.py`` from the desktop sources.

Stdlib only, Python 3.10+. The desktop modules are parsed with :mod:`ast` and never
imported (``translator_gui.py`` starts with a UTF-8 BOM, so files are read through
:func:`tokenize.open`). The output is pure data that ``src/settings_schema.py`` turns
into ``SettingSpec`` objects; the curated overlay (sections, platform availability,
oracle-calibrated defaults) lives in ``settings_schema.py``, not here.

What is read (shared-core design section 2):

a. ``save_config``'s ``settings_map`` tuples ``(key, sources, save_default, converter)``.
   The converter lambdas are mapped to ConvSpec tuples through a pattern table; anything
   the table does not know is kept as ``('unknown', <source>)`` and listed in
   ``UNKNOWN_CONVERTERS``. Duplicated keys: the last occurrence wins, as in the loop.
b. Init defaults: the ``bool_vars`` / ``str_vars`` tables (``str_vars`` defaults are
   ``str(default)``, like ``create_var``), ``self.X = <expr of self.config.get('k', d)>``,
   ``if 'k' not in self.config: self.config['k'] = v``, ``self.config.setdefault('k', v)``
   (also in loops over literal key lists / dicts) and unconditional ``self.config['k'] = v``
   in the startup functions (``INIT_FUNCS``, ``metadata_defaults.ensure_metadata_prompt_defaults``
   and every owner method they call directly: the GUI handlers startup replays). Values
   are evaluated as on a fresh install (``config.get('k', d)`` is ``d``, ``a or b``,
   builtin conversions, ``self.<attr>`` literals); reads under ``if 'k' in self.config``
   and migrations that return early when a key is missing are skipped. The EFFECTIVE
   default of a settings_map key whose source exists at startup is its converter applied
   to the init value (startup runs save_config); otherwise the save default.
c. Env bindings: every env assignment (dict literal, ``x['ENV'] = v``, ``_update_env``,
   ``set_env``, ``('ENV', v)`` pairs) in the env builders, with the config keys its value
   reads (``config.get('k', d)``, ``getattr(self, 'k_var', d)``, ``self.k_var``, helper
   calls that pass the key as a string, locals assigned from those, and the return value
   of small owner helper methods such as ``self._current_auto_glossary_mode()``).
d. Labels and tooltips: widgets created with a text (``QCheckBox("...")``,
   ``self._create_styled_checkbox("...")``), the nearest preceding ``QLabel`` for widgets
   without one, helper rows (``_create_slider_row(layout, "Top-P:", self, 'top_p_var')``),
   ``setToolTip`` text, the enclosing ``QGroupBox`` title and ``addTab`` title (closures
   included); a widget is tied to keys through ``settings_map`` widget sources,
   ``setChecked`` / ``setText`` / ``setValue`` arguments (strong) and the handlers
   connected to its signals (weak). Keys passed as class constants
   (``self.API_KEY_TREE_FONT_SIZE_CONFIG``) are resolved.
   Choices come from the combo box construction: ``addItems([...])`` (literal, local or
   module constant lists), ``addItem(label, data)`` (also in a loop over a literal list of
   pairs) and the value -> index map of ``setCurrentIndex({...}.get(value))``. A combo is tied
   to its key strongly (settings_map widget source, ``setCurrentIndex`` / ``setCurrentText``
   arguments, ``findText`` / ``findData`` lookups), never through its change handler alone.
   Items shown with other text become ``(value, label)`` pairs; index-coded combos (the key
   stores the item index as '0', '1', ...) are ints; editable combos flag ``editable_choices``.
e. Nested dict settings: ``qa_scanner_settings.*`` (``qa_scan_runtime`` defaults, the
   ``save_config`` setdefault block, ``apply_qa_scan_env_from_settings`` env),
   ``manga_settings.*`` (``manga_settings_defaults.default_manga_settings()``, which
   ``MangaSettingsDialog.default_settings`` is built from since U8), ``ai_hunter_config.*``
   (``default_ai_hunter_config()``).

f. The desktop tables themselves (U9 P5b): ``settings_map``, ``bool_vars`` and ``str_vars``
   row for row (``DESKTOP_SETTINGS_MAP`` / ``DESKTOP_BOOL_VARS`` / ``DESKTOP_STR_VARS``), from
   which ``settings_schema.desktop_settings_map`` / ``desktop_bool_vars`` / ``desktop_str_vars``
   build the exact tuples the desktop iterates. Since the switch the desktop methods call those
   functions (``settings_map = desktop_settings_map(self)``); the literals live on in the frozen
   copy ``src/mobile/tools/frozen_desktop_tables.py`` and ``load_module`` splices each one back in
   place of its call, so every record above is read from the same code as before the switch.

The extraction works on both layouts of the shared-core move: code may live in
``translator_gui.py`` or in the GUI-free mixin modules (``owner_state.py``,
``run_env.py``, ``settings_persistence.py`` and the later pipeline modules). Every
module is searched, the mixins count as TranslatorGUI methods, functions are classified
by name (never by file), records are de-duplicated as sets and never carry file names or
line numbers, and ties are broken by method name, never by position: a verbatim move
does not change the output (tests_host/test_schema_extract.py proves it on a fake tree).
Non-verbatim edits (a closure turned into a helper method, env built through a new
helper) can change a few fields; regenerate after such a change.

Usage::

    python src/mobile/tools/schema_extract.py            # (re)write src/settings_schema_data.py
    python src/mobile/tools/schema_extract.py --check    # exit 1 when the committed file is stale
    python src/mobile/tools/schema_extract.py --stdout   # print instead of writing
    python src/mobile/tools/schema_extract.py --report   # short summary (unknown converters, counts)
"""
from __future__ import annotations

import argparse
import ast
import copy
import os
import pprint
import re
import sys
import tokenize
from dataclasses import dataclass, field
from pathlib import Path

GENERATOR_VERSION = 3   # 2: widget choices; typed defaults settle the type before name heuristics
                        # 3: the desktop tables row for row (DESKTOP_*, U9 P5b)

TOOLS_DIR = Path(__file__).resolve().parent
DEFAULT_SRC = TOOLS_DIR.parents[1]
OUTPUT_NAME = "settings_schema_data.py"

# Modules holding TranslatorGUI state code, in the order duplicates are resolved.
# The mixin modules come first: while a move is in flight both copies may exist,
# and records are sets, so identical copies collapse.
OWNER_MODULES = (
    "owner_state.py",
    "run_env.py",
    "settings_persistence.py",
    "translation_pipeline.py",
    "text_jobs.py",
    "input_preparation.py",
    # U4: GUI-free rules / catalog steps the TranslatorGUI handlers call (moved bodies)
    "settings_rules.py",
    "model_catalog_core.py",
    # U6: the main window's glossary file actions and Parallel EPUB pair helpers (moved bodies,
    # ``config`` in place of ``self.config``)
    "glossary_files.py",
    "parallel_epub_core.py",
    # U7: the image / generative and RPG Maker runners (TranslatorGUI mixins, moved verbatim)
    "image_job.py",
    "rpgmaker_job.py",
    "translator_gui.py",
)
OWNER_CLASSES = ("TranslatorGUI",)          # plus every *Mixin class in OWNER_MODULES
DIALOG_MODULES = (                           # label / tooltip / UI-site priority order
    "other_settings.py",
    "translator_gui.py",
    "direct_text_store.py",   # the Direct Text dialog's GUI-free halves (inherited by the dialog)
    "direct_text_stream.py",
    "GlossaryManager_GUI.py",
    "glossary_document.py",   # U6: the Glossary Editor + prompt-profile helpers GlossaryManager_GUI calls
    "QA_Scanner_GUI.py",
    "manga_settings_dialog.py",
    "epub_library.py",
    "library_core.py",      # epub_library's GUI-free half (re-imported by the Library)
    "library_covers.py",    # U5: the rest of epub_library's GUI-free half (covers, reader, live view)
    "reader_doc.py",
    "live_stream.py",
    "Retranslation_GUI.py",
    "progress_core.py",     # the Progress Manager's GUI-free half (Retranslation_GUI inherits it, U5)
    "progress_actions.py",
    "glossary_progress_core.py",
    "sdlxliff_review_core.py",   # U7: the SDLXLIFF reviewer / sidecar auto-generation (Retranslation_GUI)
    "multi_api_key_manager.py",
    # U7: moved dialog code whose functions are listed in UI_SITE_ROOTS (no "*" root): the QA
    # Scanner's bulk scan loop and the main window's "Translate Headers Now" worker
    "qa_scan_runtime.py",
    "translate_headers_standalone.py",
)
EXTRA_MODULES = ("qa_scan_runtime.py", "ai_hunter_enhanced.py", "metadata_defaults.py",
                 "manga_settings_defaults.py")
# Dialog modules scanned for a few moved functions only (the rest of the module is backend code
# that was never part of the dialog): module -> top-level names. Reach stays inside the list.
DIALOG_MODULE_FUNCTIONS = {
    "qa_scan_runtime.py": ("run_bulk_qa_scan", "load_current_qa_settings"),
    "translate_headers_standalone.py": ("run_translate_headers_now",),
}
# Dialog-handler rules moved into functions that RETURN the config writes as a dict literal
# (the handler applies them with ``config.update(...)``): the literal's keys are touched keys.
CONFIG_UPDATE_FUNCS = {
    ("direct_text_store.py", "glossary_override_config_updates"),   # U7: _on_glossary_override_toggled
}
# Top-level definitions a scanned module holds for the mobile app only (not desktop code moved out
# of a dialog): never scanned, so they add no records, origins or UI sites of their own.
SCAN_EXCLUDE = {
    "glossary_document.py": ("EditorOwner", "EditorRow", "editor_rows", "GlossaryPromptProfiles",
                             "RefinementPromptProfiles", "GlossaryEditorError", "GlossaryDocument"),
}
# (module, function, widget) -> config keys the widget shows although the dialog no longer
# names them in an expression bound to it: the U4 review fixes moved the "Assistant Prompt"
# dialog's drafts into prompt_profiles.PrefillState, so its profile combo reads
# ``state.active_name`` (the generator cannot follow that object to the config read).
# The label / tooltip texts still come from the dialog source.
WIDGET_KEY_PINS = {
    ("translator_gui.py", "show_assistant_prompt_dialog", "profile_combo"): ("active_assistant_prompt_profile",),
}
# U9 P5b: the desktop builds its three settings tables from the schema
# (``settings_map = desktop_settings_map(self)`` in settings_persistence, ``bool_vars =
# desktop_bool_vars(self)`` / ``str_vars = desktop_str_vars(self)`` in owner_state). The literals
# live on in this frozen copy (path relative to src/); load_module splices each one back in place
# of its call, so the generator reads the same code as before the switch.
FROZEN_TABLES = "mobile/tools/frozen_desktop_tables.py"
DESKTOP_TABLES = ("settings_map", "bool_vars", "str_vars")
# GUI-free helpers that TranslatorGUI.__init__ runs on self.config (startup writes).
EXTRA_INIT_FUNCS = (
    ("metadata_defaults.py", "ensure_metadata_prompt_defaults"),   # via MetadataBatchTranslatorUI / _hook_metadata_defaults
)

# Function names -> category. "init" = desktop startup (fresh-install defaults);
# the env sites follow the EnvBinding site vocabulary of settings_schema.
INIT_FUNCS = {
    "__init__",                      # TranslatorGUI only (see categories)
    "_init_config_state",
    "_init_variables",
    "_init_gui_backed_state",
    "_init_default_prompts",
    "_init_default_prompt_profiles",
    "_replay_gui_startup_handlers",
    "initialize_extraction_variables",
    "_setup_gui",
    "_create_settings_section",
    "_create_prompt_section",
    "_create_model_section",
    "_create_api_section",
    "_create_profile_section",
    "create_file_section",
}
SAVE_FUNCS = {"save_config", "_collect_live_settings", "_apply_live_settings_to_config", "_export_settings_env"}
ENV_SITES = {
    "translation": (
        "_get_environment_variables",
        "_apply_direct_text_runtime_environment",
        "_export_multipass_runtime_env",
        "_export_chapter_range_runtime_env",
        "_metadata_only_environment_for_file",
        "_apply_forced_streaming_environment",
        "_process_text_file",
        "_strict_matching_env_dict",
        "_unified_glossary_env_dict",
        "_current_glossary_request_env",
        "_sync_custom_prefix_routes_env",
        "run_translation_thread",
        "run_translation_direct",
        "_prepare_translation_run",
        "_translation_worker",
    ),
    "glossary": (
        "_build_glossary_extraction_env",
        "_extract_glossary_from_text_file",
        "run_glossary_extraction_direct",
    ),
    "epub": ("_build_epub_compile_env", "run_epub_converter_direct", "_run_epub_compile"),
    "pdf": ("_build_pdf_compile_env", "run_pdf_converter_direct", "_run_pdf_compile"),
    "startup": (
        "initialize_environment_variables",
        "__init__",
        "_init_config_state",
        "_init_variables",
        "_init_gui_backed_state",
    ),
    "save": ("save_config", "_export_settings_env", "_glossary_env_mappings"),
}
SITE_ORDER = ("translation", "glossary", "epub", "pdf", "startup", "save", "qa")

# Nested dict settings: parent key -> where its defaults come from.
NESTED_PARENTS = ("qa_scanner_settings", "manga_settings", "ai_hunter_config")
NESTED_DEFAULT_SOURCES = (
    # (parent, module, qualname, how): how = "return" (function returning a dict literal)
    # or "attr:<name>" (self.<name> = {...} inside the function).
    ("qa_scanner_settings", "qa_scan_runtime.py", "default_qa_scan_settings", "return"),
    # U8: MangaSettingsDialog.default_settings moved verbatim into manga_settings_defaults
    ("manga_settings", "manga_settings_defaults.py", "default_manga_settings", "return"),
    ("ai_hunter_config", "ai_hunter_enhanced.py", "default_ai_hunter_config", "return"),
)
# Nested dicts that are one value (user data keyed by language etc.), not a settings group.
NESTED_DATA_LEAVES = {"qa_scanner_settings.word_count_multipliers"}
# (module, function, local name) whose ``.get('k', d)`` reads the parent dict; site "qa".
NESTED_ENV_FUNCS = (
    ("qa_scan_runtime.py", "apply_qa_scan_env_from_settings", "settings", "qa_scanner_settings", "qa"),
)
# Local names that hold a nested dict in a module even when the assignment is opaque.
NESTED_NAME_HINTS = {
    "QA_Scanner_GUI.py": {
        "qa_settings": "qa_scanner_settings",
        "current_qa_settings": "qa_scanner_settings",
        "refreshed_qa_settings": "qa_scanner_settings",
    },
    # U7: QA_Scanner_GUI's run_scan loop (qa_scan_runtime.run_bulk_qa_scan, DIALOG_MODULE_FUNCTIONS)
    "qa_scan_runtime.py": {
        "qa_settings": "qa_scanner_settings",
        "current_qa_settings": "qa_scanner_settings",
        "refreshed_qa_settings": "qa_scanner_settings",
    },
}

# Desktop UI sites: (module, top-level function) -> site id. Reachable helpers
# (self.<name> references inside the same module) inherit the site.
UI_SITE_ROOTS = (
    ("other_settings.py", "_create_prompt_management_section", "other.meta_data"),
    ("other_settings.py", "_create_context_management_section", "other.context"),
    ("other_settings.py", "_create_response_handling_section", "other.response"),
    ("other_settings.py", "_create_processing_options_section", "other.processing"),
    ("other_settings.py", "_create_image_translation_section", "other.image"),
    ("other_settings.py", "_create_anti_duplicate_section", "other.anti_duplicate"),
    ("other_settings.py", "_create_custom_api_endpoints_section", "other.endpoints"),
    ("other_settings.py", "_create_debug_controls_section", "other.debug"),
    ("other_settings.py", "_create_output_settings_section", "other.output"),
    ("other_settings.py", "_create_danger_zone_section", "other.danger"),
    ("other_settings.py", "open_other_settings", "other.dialog"),
    ("translator_gui.py", "_create_settings_section", "main.run"),
    ("translator_gui.py", "_create_model_section", "main.model"),
    ("translator_gui.py", "_create_api_section", "main.model"),
    ("translator_gui.py", "_create_prompt_section", "main.prompt"),
    ("translator_gui.py", "_create_profile_section", "main.prompt"),
    ("translator_gui.py", "create_file_section", "main.file"),
    ("translator_gui.py", "_InputOutputDialog.*", "direct_text"),
    ("direct_text_store.py", "ChatStoreMixin.*", "direct_text"),
    # U7: the Direct Text dialog's handler rules moved out of translator_gui as module functions
    ("direct_text_store.py", "configured_glossary_override_mode", "direct_text"),
    ("direct_text_store.py", "glossary_override_config_updates", "direct_text"),
    ("direct_text_store.py", "chat_rename_title", "direct_text"),
    ("direct_text_store.py", "manual_glossary_source_record", "direct_text"),
    ("direct_text_store.py", "sniff_manual_glossary_extension", "direct_text"),
    ("direct_text_stream.py", "DirectTextStreamMixin.*", "direct_text"),
    ("GlossaryManager_GUI.py", "_setup_glossary_general_tab", "glossary.general"),
    ("GlossaryManager_GUI.py", "_setup_manual_glossary_tab", "glossary.balanced_full"),
    ("GlossaryManager_GUI.py", "_setup_auto_glossary_tab", "glossary.minimal"),
    ("GlossaryManager_GUI.py", "_setup_glossary_refinement_tab", "glossary.refinement"),
    ("GlossaryManager_GUI.py", "_open_unified_glossary_settings_dialog", "glossary.unified"),
    ("GlossaryManager_GUI.py", "_open_glossary_anti_duplicate_dialog", "glossary.anti_duplicate"),
    ("GlossaryManager_GUI.py", "_create_glossary_anti_duplicate_section", "glossary.anti_duplicate"),
    ("GlossaryManager_GUI.py", "_setup_glossary_editor_tab", "glossary.editor"),
    ("GlossaryManager_GUI.py", "*", "glossary.other"),
    # U6: glossary_document holds the Editor tab's inner functions and the Balanced/Full tab's
    # prompt-profile config helpers (moved out of GlossaryManager_GUI).
    ("glossary_document.py", "glossary_prompt_profile_meta", "glossary.balanced_full"),
    ("glossary_document.py", "ensure_glossary_prompt_profiles", "glossary.balanced_full"),
    ("glossary_document.py", "glossary_prompt_profiles_for", "glossary.balanced_full"),
    ("glossary_document.py", "active_glossary_prompt_profile_for", "glossary.balanced_full"),
    ("glossary_document.py", "set_active_glossary_prompt_profile", "glossary.balanced_full"),
    ("glossary_document.py", "default_glossary_prompt_profile_text", "glossary.balanced_full"),
    ("glossary_document.py", "set_default_glossary_prompt_profile_text", "glossary.balanced_full"),
    ("glossary_document.py", "store_glossary_prompt_current", "glossary.balanced_full"),
    ("glossary_document.py", "default_glossary_refinement_system_prompt", "glossary.refinement"),
    ("glossary_document.py", "default_glossary_refinement_user_prompt", "glossary.refinement"),
    ("glossary_document.py", "unified_glossary_shared_dir", "glossary.unified"),
    ("glossary_document.py", "unified_rebuild_settings", "glossary.unified"),
    ("glossary_document.py", "*", "glossary.editor"),
    ("QA_Scanner_GUI.py", "*", "qa"),
    # U7: QA_Scanner_GUI's run_scan worker and settings loader closure (moved verbatim; only the
    # DIALOG_MODULE_FUNCTIONS are scanned, at the "*" distance they had inside QA_Scanner_GUI)
    ("qa_scan_runtime.py", "*", "qa"),
    ("manga_settings_dialog.py", "*", "manga"),
    ("epub_library.py", "*", "library"),
    ("library_core.py", "*", "library"),
    ("library_covers.py", "*", "library"),
    ("reader_doc.py", "*", "library"),
    ("live_stream.py", "*", "library"),
    ("Retranslation_GUI.py", "*", "progress"),
    ("progress_core.py", "*", "progress"),
    ("progress_actions.py", "*", "progress"),
    ("glossary_progress_core.py", "*", "progress"),
    ("sdlxliff_review_core.py", "*", "progress"),
    ("multi_api_key_manager.py", "*", "keys"),
    # U7: other_settings' "Translate Headers Now" worker (prompt management section)
    ("translate_headers_standalone.py", "run_translate_headers_now", "other.meta_data"),
)
UI_REACH_DEPTH = 3
# Never walk into these from a UI root: they touch every setting and are not UI.
UI_REACH_EXCLUDE = (INIT_FUNCS | SAVE_FUNCS | {n for names in ENV_SITES.values() for n in names}
                    | {"_glossary_env_mappings", "initialize_environment_variables"})

ENV_RE = re.compile(r"^[A-Z][A-Z0-9_]{2,}$")
CONFIG_NAMES = {"config", "cfg", "_cfg", "conf", "_config", "live_config", "config_dict"}
ENV_SETTER_FUNCS = {"_update_env", "set_env", "putenv", "setdefault", "_set_env", "update_env"}
WIDGET_FACTORIES = {"_create_styled_checkbox", "_create_preview_pool_button"}
# Widgets whose text names the setting they edit (buttons and radios name actions / choices).
INPUT_FACTORIES = {"_create_styled_checkbox"}
INPUT_SUFFIXES = ("ComboBox", "LineEdit", "SpinBox", "CheckBox", "TextEdit", "Slider", "Combo")
WIDGET_SUFFIXES = ("ComboBox", "LineEdit", "SpinBox", "CheckBox", "TextEdit", "Slider", "Combo",
                   "RadioButton", "Button", "Label")
COMBO_SUFFIXES = ("ComboBox", "Combo")       # widgets whose items are choices
CHOICE_TIE_METHODS = {"findText", "findData"}  # combo lookups of the stored value
BIND_METHODS = {"setChecked", "setText", "setValue", "setCurrentText", "setCurrentIndex",
                "setPlainText", "setEditText"}
SIGNALS = {"toggled", "stateChanged", "textChanged", "valueChanged", "currentIndexChanged",
           "currentTextChanged", "editingFinished", "clicked", "textEdited", "activated"}
LABEL_WINDOW = 6
NONE = "<none>"           # no call-site default (EnvBinding / records)
_MISSING = object()


# --------------------------------------------------------------------------- reading
def read_source(path: Path) -> str:
    with tokenize.open(str(path)) as fh:       # BOM + PEP 263 aware
        return fh.read()


@dataclass
class Module:
    name: str
    source: str
    tree: ast.Module
    consts: dict = field(default_factory=dict)    # NAME -> literal value
    imports: dict = field(default_factory=dict)   # local NAME -> "module:NAME"
    class_consts: dict = field(default_factory=dict)  # class-body NAME -> str (config key constants)


def _table_call(node):
    """The table name when ``node`` is ``<table> = [module.]desktop_<table>(...)`` (U9 P5b)."""
    if not (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)):
        return None
    table = node.targets[0].id
    if table in DESKTOP_TABLES and isinstance(node.value, ast.Call) and _call_name(node.value) == "desktop_" + table:
        return table
    return None


def frozen_tables(src: Path) -> dict:
    """Table name -> source text of its literal, from the frozen copy (FROZEN_TABLES)."""
    path = src / FROZEN_TABLES
    if not path.is_file():
        raise SystemExit(f"schema_extract: the desktop builds its tables from settings_schema but the frozen "
                         f"copy {FROZEN_TABLES} (the save_config settings_map / _init_variables tables) is missing")
    source = read_source(path)
    tree = ast.parse(source, filename=FROZEN_TABLES)
    tables = {}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in DESKTOP_TABLES and isinstance(node.value, ast.List)):
            table = node.targets[0].id
            if table in tables:
                raise SystemExit(f"schema_extract: {FROZEN_TABLES} defines {table} twice")
            tables[table] = ast.get_source_segment(source, node.value)
    missing = [table for table in DESKTOP_TABLES if table not in tables]
    if missing:
        raise SystemExit(f"schema_extract: {', '.join(missing)} not found in {FROZEN_TABLES}")
    return tables


def splice_frozen_tables(src: Path, name: str, source: str) -> str:
    """Put the frozen literal back in place of each ``<table> = desktop_<table>(...)`` call.

    Positions are only used here: records never carry line numbers, so the spliced module
    yields exactly the records the literal produced before the switch (U9 P5b)."""
    if "desktop_" not in source:
        return source
    calls = [(node.value, table) for node in ast.walk(ast.parse(source, filename=name))
             for table in (_table_call(node),) if table]
    if not calls:
        return source
    literals = frozen_tables(src)
    lines = source.split("\n")
    for value, table in sorted(calls, key=lambda item: (item[0].lineno, item[0].col_offset), reverse=True):
        first = lines[value.lineno - 1].encode("utf-8")
        last = lines[value.end_lineno - 1].encode("utf-8")
        spliced = (first[:value.col_offset].decode("utf-8") + literals[table]
                   + last[value.end_col_offset:].decode("utf-8"))
        lines[value.lineno - 1:value.end_lineno] = [spliced]
    return "\n".join(lines)


def load_module(src: Path, name: str):
    path = src / name
    if not path.is_file():
        return None
    source = splice_frozen_tables(src, name, read_source(path))
    tree = ast.parse(source, filename=name)
    excluded = SCAN_EXCLUDE.get(name, ())
    if excluded:
        tree.body = [node for node in tree.body
                     if not (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                             and node.name in excluded)]
    mod = Module(name, source, tree)
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            ok, value = literal(node.value)
            if ok:
                mod.consts[node.targets[0].id] = value
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            for alias in node.names:
                mod.imports[alias.asname or alias.name] = f"{node.module}:{alias.name}"
    # class-level string constants used as config keys (e.g. API_KEY_TREE_FONT_SIZE_CONFIG)
    ambiguous = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if (isinstance(item, ast.Assign) and len(item.targets) == 1
                        and isinstance(item.targets[0], ast.Name) and const_str(item.value) is not None):
                    name = item.targets[0].id
                    if name in mod.class_consts and mod.class_consts[name] != item.value.value:
                        ambiguous.add(name)
                    mod.class_consts.setdefault(name, item.value.value)
    for name in ambiguous:
        mod.class_consts.pop(name, None)
    # imports inside functions / try blocks (translator_gui imports many names lazily)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            for alias in node.names:
                mod.imports.setdefault(alias.asname or alias.name, f"{node.module}:{alias.name}")
    return mod


def literal(node):
    try:
        return True, ast.literal_eval(node)
    except Exception:
        return False, None


def segment(mod: Module, node) -> str:
    """Source text of a node with whitespace collapsed (stable across Python versions)."""
    lines = mod.__dict__.get("_lines")
    if lines is None:
        lines = mod.__dict__["_lines"] = [line.encode("utf-8") for line in mod.source.splitlines(True)]
    start, end = node.lineno - 1, node.end_lineno - 1
    if start == end:
        raw = lines[start][node.col_offset:node.end_col_offset]
    else:
        raw = b"".join([lines[start][node.col_offset:]] + lines[start + 1:end] + [lines[end][:node.end_col_offset]])
    return " ".join(raw.decode("utf-8").split())


def const_str(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def key_str(mod, node):
    """A config key: a str literal, or self.NAME / Class.NAME bound to a str in a class body."""
    text = const_str(node)
    if text is not None:
        return text
    if mod is not None and isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        if node.value.id == "self" or node.value.id[:1].isupper():
            return mod.class_consts.get(node.attr)
    return None


def joined_text(node):
    """Text of a str constant, implicit concatenation, ``+`` of constants or f-string parts."""
    if node is None:
        return None
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = joined_text(node.left), joined_text(node.right)
        if left is not None and right is not None:
            return left + right
        return None
    if isinstance(node, ast.JoinedStr):
        parts = []
        for value in node.values:
            if isinstance(value, ast.Constant):
                parts.append(str(value.value))
            else:
                parts.append("{…}")
        return "".join(parts)
    if isinstance(node, ast.Call) and _call_name(node) in ("tr", "_tr", "_wrapped_tooltip_html") and node.args:
        return joined_text(node.args[0])
    return None


def _call_name(call: ast.Call):
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def is_self_attr(node, name=None):
    return (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
            and node.value.id == "self" and (name is None or node.attr == name))


def is_config_like(node) -> bool:
    if isinstance(node, ast.Name):
        return node.id in CONFIG_NAMES
    if isinstance(node, ast.Attribute):
        return node.attr in ("config",)
    return False


# --------------------------------------------------------------------------- functions
@dataclass
class Func:
    module: Module
    qualname: str          # "Class.method" or "function"
    node: ast.AST
    cls: str | None

    @property
    def name(self):
        return self.qualname.rsplit(".", 1)[-1]


def iter_functions(mod: Module):
    for node in mod.tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield Func(mod, node.name, node, None)
        elif isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    yield Func(mod, f"{node.name}.{item.name}", item, node.name)


def owner_functions(mods: dict):
    """TranslatorGUI methods, *Mixin methods and module functions of the owner modules."""
    out = []
    for name in OWNER_MODULES:
        mod = mods.get(name)
        if mod is None:
            continue
        for fn in iter_functions(mod):
            if fn.cls is None:
                if name == "translator_gui.py":
                    continue                   # module helpers of the GUI are not owner state
                out.append(fn)
            elif fn.cls in OWNER_CLASSES or fn.cls.endswith("Mixin") and name != "translator_gui.py":
                out.append(fn)
    other = mods.get("other_settings.py")
    if other is not None:
        for fn in iter_functions(other):
            if fn.qualname == "initialize_extraction_variables":
                out.append(fn)
    for mod_name, qualname in EXTRA_INIT_FUNCS:
        mod = mods.get(mod_name)
        if mod is not None:
            for fn in iter_functions(mod):
                if fn.qualname == qualname:
                    out.append(fn)
    return out


def _contains_settings_map(fn: Func) -> bool:
    for node in ast.walk(fn.node):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "settings_map"):
            return True
    return False


_INIT_CALLEES = set()      # owner functions called directly from an init function (set per run)


def categories(fn: Func):
    """Set of categories ('init', 'save', env sites) of an owner function."""
    cats = set()
    name = fn.name
    if fn.qualname in _INIT_CALLEES:
        cats.add("init")
    if _contains_settings_map(fn):
        cats.add("save")
    if name == "__init__" and fn.cls not in OWNER_CLASSES and not (fn.cls or "").endswith("Mixin"):
        return cats
    if name in INIT_FUNCS or any(fn.module.name == m and fn.qualname == q for m, q in EXTRA_INIT_FUNCS):
        cats.add("init")
    if name in SAVE_FUNCS:
        cats.add("save")
    for site, names in ENV_SITES.items():
        if name in names:
            cats.add("env:" + site)
    return cats


# --------------------------------------------------------------------------- refs
# Small owner helper methods whose return value a ref may come through
# (``combo.setCurrentIndex(self._saved_auto_glossary_shortcut_index())``,
# ``'AUTO_GLOSSARY_MODE': self._current_auto_glossary_mode()``). Set per run.
_METHODS = {"index": {}, "cache": {}}
METHOD_MAX_STATEMENTS = 30
METHOD_MAX_DEPTH = 2


def _method_size(node) -> int:
    return sum(1 for sub in ast.walk(node) if isinstance(sub, ast.stmt))


class RefContext:
    """Resolves which config keys an expression reads, inside one function."""

    def __init__(self, mod: Module, func_node, var_map: dict, known_keys: set,
                 nested_hints: dict | None = None, attr_paths: dict | None = None,
                 method_depth: int = 0):
        self.method_depth = method_depth
        self.mod = mod
        self.var_map = var_map
        self.known_keys = known_keys
        self.locals = {}
        self.paths = dict(nested_hints or {})     # local name -> nested path tuple
        self.attr_paths = attr_paths or {}        # self.<attr> -> nested path tuple
        self._busy = set()
        self._into_defs = False
        for node in ast.walk(func_node):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.locals.setdefault(target.id, []).append(node.value)
            elif isinstance(node, (ast.AnnAssign, ast.AugAssign)) and isinstance(node.target, ast.Name) and node.value is not None:
                self.locals.setdefault(node.target.id, []).append(node.value)
        for name, values in self.locals.items():
            if name in self.paths:
                continue
            for value in values:
                path = self.path_of(value)
                if path:
                    self.paths[name] = path
                    break

    # -- nested dict paths
    def path_of(self, node, depth=0):
        if depth > 6 or node is None:
            return None
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr in ("get", "setdefault") and node.args:
                key = const_str(node.args[0])
                if key is not None:
                    if is_config_like(func.value):
                        return (key,) if key in NESTED_PARENTS else None
                    base = self.path_of(func.value, depth + 1)
                    if base:
                        return base + (key,)
                return None
            if isinstance(func, ast.Attribute) and func.attr == "copy" and not node.args:
                return self.path_of(func.value, depth + 1)
            name = _call_name(node)
            if name in ("dict", "deepcopy", "_merge_settings", "copy") and node.args:
                return self.path_of(node.args[0], depth + 1)
            return None
        if isinstance(node, ast.Subscript):
            key = const_str(node.slice)
            if key is None:
                return None
            if is_config_like(node.value):
                return (key,) if key in NESTED_PARENTS else None
            base = self.path_of(node.value, depth + 1)
            return base + (key,) if base else None
        if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or):
            return self.path_of(node.values[0], depth + 1)
        if isinstance(node, ast.Name):
            return self.paths.get(node.id)
        if is_self_attr(node):
            return self.attr_paths.get(node.attr)
        return None

    # -- refs
    def refs(self, node, depth=0, into_defs=False):
        """Return (key, default) pairs; default is NONE when the site has none."""
        out = []
        self._into_defs = into_defs
        self._refs(node, out, depth)
        self._into_defs = False
        return out

    def _method_refs(self, name):
        cache = _METHODS["cache"]
        cache_key = (name, self.method_depth, id(self.var_map), len(self.var_map), len(self.known_keys))
        if cache_key in cache:
            return cache[cache_key]
        cache[cache_key] = []                  # recursion guard
        found = []
        for fn in _METHODS["index"].get(name, ()):
            if _method_size(fn.node) > METHOD_MAX_STATEMENTS:
                continue
            ctx = RefContext(fn.module, fn.node, self.var_map, self.known_keys,
                             method_depth=self.method_depth + 1)
            for node in ast.walk(fn.node):
                if isinstance(node, ast.Return) and node.value is not None:
                    found.extend(ctx.refs(node.value))
        result = sorted({repr(item): item for item in found}.values(), key=repr)
        cache[cache_key] = result
        return result

    def _default(self, node):
        if node is None:
            return NONE
        return value_repr(self.mod, node)

    def _refs(self, node, out, depth):
        if node is None or depth > 8:
            return
        if isinstance(node, ast.Call):
            func = node.func
            name = _call_name(node)
            handled_args = set()
            descend_func = isinstance(func, ast.Attribute)
            if isinstance(func, ast.Attribute) and func.attr in ("get", "setdefault", "pop") and node.args:
                key = key_str(self.mod, node.args[0])
                if key is not None:
                    dflt = node.args[1] if len(node.args) > 1 else None
                    if is_config_like(func.value):
                        out.append((key, self._default(dflt) if not _is_config_call(dflt) else NONE))
                        handled_args.add(0)
                        descend_func = False
                    else:
                        base = self.path_of(func.value)
                        if base:
                            out.append((".".join(base + (key,)), self._default(dflt) if not _is_config_call(dflt) else NONE))
                            handled_args.add(0)
                            descend_func = False
            elif name == "getattr" and len(node.args) >= 2 and isinstance(node.args[0], ast.Name) and node.args[0].id == "self":
                attr = const_str(node.args[1])
                if attr is not None and attr in self.var_map:
                    dflt = node.args[2] if len(node.args) > 2 else None
                    use = NONE if dflt is None or not literal(dflt)[0] else self._default(dflt)
                    for key in self.var_map[attr]:
                        out.append((key, use))
                handled_args.update((0, 1))
            elif name == "hasattr":
                return
            elif is_self_attr(func) and func.attr in _METHODS["index"] and self.method_depth < METHOD_MAX_DEPTH:
                out.extend(self._method_refs(func.attr))
            else:
                # helper(... 'key', default ...) / helper('x_var', 'key', default)
                args = node.args
                key_idx = None
                for i, arg in enumerate(args):
                    s = const_str(arg)
                    if s is not None and s in self.known_keys:
                        key_idx = i
                        break
                if key_idx is not None:
                    key = const_str(args[key_idx])
                    dflt = args[key_idx + 1] if key_idx + 1 < len(args) else None
                    if dflt is None:
                        for kw in node.keywords:
                            if kw.arg == "default":
                                dflt = kw.value
                    use = NONE if dflt is None or not literal(dflt)[0] else self._default(dflt)
                    out.append((key, use))
                    handled_args.add(key_idx)
                    for i, arg in enumerate(args):
                        s = const_str(arg)
                        if s is not None and i != key_idx:
                            handled_args.add(i)
                else:
                    for i, arg in enumerate(args):
                        s = const_str(arg)
                        if s is not None and s in self.var_map:
                            for key in self.var_map[s]:
                                out.append((key, NONE))
                            handled_args.add(i)
            if descend_func:
                self._refs(func.value, out, depth + 1)
            for i, arg in enumerate(node.args):
                if i not in handled_args:
                    self._refs(arg, out, depth + 1)
            for kw in node.keywords:
                self._refs(kw.value, out, depth + 1)
            return
        if isinstance(node, ast.Subscript):
            key = key_str(self.mod, node.slice)
            if key is not None:
                if is_config_like(node.value):
                    out.append((key, NONE))
                    return
                base = self.path_of(node.value)
                if base:
                    out.append((".".join(base + (key,)), NONE))
                    return
        if is_self_attr(node) and isinstance(node.ctx, ast.Load):
            if node.attr in self.var_map:
                for key in self.var_map[node.attr]:
                    out.append((key, NONE))
            return
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id in self.locals and node.id not in self._busy:
                self._busy.add(node.id)
                for value in self.locals[node.id]:
                    self._refs(value, out, depth + 1)
                self._busy.discard(node.id)
            return
        if isinstance(node, (ast.Lambda, ast.FunctionDef, ast.AsyncFunctionDef)) and not self._into_defs:
            return
        for child in ast.iter_child_nodes(node):
            self._refs(child, out, depth + 1)


def _is_config_call(node):
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get" and is_config_like(node.func.value))


# Imported constants are inlined when the defining src module binds them to a literal
# that is not a long string (long prompt texts stay {'$ref': ...} and are imported
# lazily by settings_schema.effective_default).
INLINE_STR_MAX = 160
_REF_SRC = {"dir": None, "cache": {}}


def _inline_ref(ref: str):
    module_name, _sep, name = ref.partition(":")
    src = _REF_SRC["dir"]
    if src is None or not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", module_name):
        return _MISSING
    cache = _REF_SRC["cache"]
    if module_name not in cache:
        path = Path(src) / (module_name + ".py")
        cache[module_name] = load_module(Path(src), module_name + ".py") if path.is_file() else None
    mod = cache[module_name]
    if mod is None or name not in mod.consts:
        return _MISSING
    value = mod.consts[name]
    if isinstance(value, str) and len(value) > INLINE_STR_MAX:
        return _MISSING
    return value


def value_repr(mod: Module, node):
    """Literal value, a module-constant/import reference, or the source expression."""
    ok, value = literal(node)
    if ok:
        return value
    if is_self_attr(node) and node.attr.startswith("default_"):
        return {"$attr": node.attr}
    if (isinstance(node, ast.Call) and _call_name(node) == "getattr" and len(node.args) >= 2
            and isinstance(node.args[0], ast.Name) and node.args[0].id == "self"
            and (const_str(node.args[1]) or "").startswith("default_")):
        return {"$attr": const_str(node.args[1])}
    if isinstance(node, ast.Name):
        if node.id in mod.consts:
            return mod.consts[node.id]
        if node.id in mod.imports:
            inline = _inline_ref(mod.imports[node.id])
            if inline is not _MISSING:
                return inline
            return {"$ref": mod.imports[node.id]}
    if isinstance(node, ast.Call) and _call_name(node) == "str" and len(node.args) == 1:
        inner = value_repr(mod, node.args[0])
        if not isinstance(inner, dict):
            return str(inner)
    return {"$expr": segment(mod, node)}


_FRESH_BUILTINS = {"str": str, "int": int, "float": float, "bool": bool, "list": list, "tuple": tuple,
                   "max": max, "min": min}


class Fresh:
    """Evaluate a startup expression the way it behaves on a fresh install (empty config):
    ``config.get('k', d)`` is ``d``, ``a or b`` picks the first truthy value, builtin
    conversions run on literals. Anything else is ``value_repr`` (a marker)."""

    def __init__(self, mod: Module, local_consts: dict, attr_literals: dict):
        self.mod = mod
        self.local_consts = local_consts
        self.attr_literals = attr_literals

    def __call__(self, node, depth=0):
        if node is None:
            return None
        if depth > 8:
            return value_repr(self.mod, node)
        ok, value = literal(node)
        if ok:
            return value
        if isinstance(node, ast.Name):
            if node.id in self.local_consts:
                return self.local_consts[node.id]
            return value_repr(self.mod, node)
        if is_self_attr(node):
            if node.attr.startswith("default_"):
                return {"$attr": node.attr}
            if node.attr in self.attr_literals:
                return self.attr_literals[node.attr]
            return value_repr(self.mod, node)
        if isinstance(node, ast.BoolOp):
            last = None
            for item in node.values:
                last = self(item, depth + 1)
                if _is_marker(last):
                    return value_repr(self.mod, node)
                if isinstance(node.op, ast.Or) and last:
                    return last
                if isinstance(node.op, ast.And) and not last:
                    return last
            return last
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "get" and is_config_like(func.value) and node.args:
                return self(node.args[1], depth + 1) if len(node.args) > 1 else None
            name = _call_name(node)
            if name == "getattr" and len(node.args) >= 2 and isinstance(node.args[0], ast.Name) and node.args[0].id == "self":
                attr = const_str(node.args[1]) or ""
                if attr.startswith("default_"):
                    return {"$attr": attr}
                if attr in self.attr_literals:
                    return self.attr_literals[attr]
                return self(node.args[2], depth + 1) if len(node.args) > 2 else value_repr(self.mod, node)
            if isinstance(func, ast.Name) and name in _FRESH_BUILTINS and node.args and not node.keywords:
                args = [self(arg, depth + 1) for arg in node.args]
                if len(args) == 1 and name in ("list", "tuple") and _is_marker(args[0]) and "$ref" in args[0]:
                    return {"$ref": args[0]["$ref"], "as": name}
                if any(_is_marker(arg) for arg in args):
                    return value_repr(self.mod, node)
                try:
                    return _FRESH_BUILTINS[name](*args)
                except Exception:
                    return value_repr(self.mod, node)
            if isinstance(func, ast.Attribute) and func.attr in ("strip", "lower", "upper") and not node.args:
                inner = self(func.value, depth + 1)
                if isinstance(inner, str):
                    return getattr(inner, func.attr)()
        return value_repr(self.mod, node)


def _fresh_install_noop(func_node) -> bool:
    """True when the function starts with ``if 'k' not in self.config: return`` (a
    migration that never runs on an empty config)."""
    body = list(func_node.body)
    if body and isinstance(body[0], ast.Expr) and const_str(body[0].value) is not None:
        body = body[1:]
    if not body or not isinstance(body[0], ast.If):
        return False
    test = body[0].test
    return (isinstance(test, ast.Compare) and len(test.ops) == 1 and isinstance(test.ops[0], ast.NotIn)
            and const_str(test.left) is not None and is_config_like(test.comparators[0])
            and len(body[0].body) == 1 and isinstance(body[0].body[0], ast.Return))


def local_constants(func_node):
    """Names assigned exactly once in a function, to a literal."""
    seen = {}
    for node in ast.walk(func_node):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    seen.setdefault(target.id, []).append(node.value)
    out = {}
    for name, values in seen.items():
        if len(values) == 1:
            ok, value = literal(values[0])
            if ok:
                out[name] = value
    return out


def source_order(func_node):
    return sorted((n for n in ast.walk(func_node) if hasattr(n, "lineno")),
                  key=lambda n: (n.lineno, n.col_offset))


# --------------------------------------------------------------------------- (a) settings_map
@dataclass
class SMEntry:
    key: str
    sources: tuple
    save_default: object
    converter: tuple
    index: int


def find_settings_map(mods: dict):
    for name in OWNER_MODULES:
        mod = mods.get(name)
        if mod is None:
            continue
        for node in ast.walk(mod.tree):
            if (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "settings_map"
                    and isinstance(node.value, ast.List)):
                return mod, node.value
    raise SystemExit("schema_extract: save_config settings_map not found in " + ", ".join(OWNER_MODULES))


def module_constant(mods: dict, name: str):
    for mod_name in OWNER_MODULES:
        mod = mods.get(mod_name)
        if mod is not None and name in mod.consts:
            return True, mod.consts[name]
    return False, None


def parse_settings_map(mods: dict):
    mod, lst = find_settings_map(mods)
    entries = []
    unknown = {}
    for index, elt in enumerate(lst.elts):
        if not isinstance(elt, ast.Tuple) or len(elt.elts) != 4:
            raise SystemExit(f"schema_extract: unexpected settings_map entry {segment(mod, elt)!r}")
        key = const_str(elt.elts[0])
        if key is None:
            raise SystemExit(f"schema_extract: non-literal settings_map key {segment(mod, elt.elts[0])!r}")
        sources = []
        for src_node in elt.elts[1].elts:
            if isinstance(src_node, ast.Tuple):
                ok, value = literal(src_node)
                sources.append(f"{value[0]}:{value[1]}" if ok else segment(mod, src_node))
            else:
                sources.append(const_str(src_node) or segment(mod, src_node))
        save_default = settings_default(mod, elt.elts[2])
        conv = converter_spec(mods, mod, elt.elts[3])
        if conv[0] == "unknown":
            unknown[key] = conv[1]
        entries.append(SMEntry(key, tuple(sources), save_default, conv, index))
    return entries, unknown


# --------------------------------------------------------------------------- (f) desktop tables
# Row encoding read by settings_schema (U9 P5b). A table default is a tagged tuple:
#   ('value', literal)                 the literal (lists / dicts are copied for every build)
#   ('ref', 'module:NAME')             an imported constant (settings_schema._resolve_marker)
#   ('attr', name, default)            getattr(self, name, <default>)
#   ('config', key, default)           self.config.get(key, <default>)
# Converters are ConvSpecs (converter_spec); sources keep their literal form ('attr' or
# ('config', key)). Anything else cannot be rebuilt and stops the generator.
def _desktop_unbuildable(mod: Module, node, what: str):
    raise SystemExit(f"schema_extract: desktop table {what} {segment(mod, node)!r} cannot be rebuilt by "
                     "settings_schema (use a literal, an imported constant, getattr(self, 'name', default) "
                     "or self.config.get('key', default))")


def table_default(mod: Module, node):
    ok, value = literal(node)
    if ok:
        return ("value", value)
    if isinstance(node, ast.Name):
        if node.id in mod.consts:
            return ("value", mod.consts[node.id])
        ref = mod.imports.get(node.id, "")
        if ":" in ref:
            return ("ref", ref)
    if isinstance(node, ast.Call) and not node.keywords:
        func = node.func
        if (isinstance(func, ast.Name) and func.id == "getattr" and len(node.args) == 3
                and isinstance(node.args[0], ast.Name) and node.args[0].id == "self" and const_str(node.args[1])):
            return ("attr", const_str(node.args[1]), table_default(mod, node.args[2]))
        if (isinstance(func, ast.Attribute) and func.attr == "get" and is_self_attr(func.value, "config")
                and len(node.args) == 2 and const_str(node.args[0])):
            return ("config", const_str(node.args[0]), table_default(mod, node.args[1]))
    _desktop_unbuildable(mod, node, "default")


def _table_rows(mod: Module, lst, width: int, table: str):
    if not isinstance(lst, ast.List):
        _desktop_unbuildable(mod, lst, table)
    for elt in lst.elts:
        if not (isinstance(elt, ast.Tuple) and len(elt.elts) == width):
            _desktop_unbuildable(mod, elt, f"{table} row")
        yield elt.elts


def _table_sources(mod: Module, node):
    if not isinstance(node, ast.List):
        _desktop_unbuildable(mod, node, "sources")
    out = []
    for elt in node.elts:
        ok, value = literal(elt)
        if not (ok and (isinstance(value, str)
                        or (isinstance(value, tuple) and all(isinstance(v, str) for v in value)))):
            _desktop_unbuildable(mod, elt, "source")
        out.append(value)
    return tuple(out)


def _single_table(mods: dict, table: str):
    """(module, list node) of the bool_vars / str_vars table; copies left by an in-flight
    move must be identical."""
    found = []
    for fn in owner_functions(mods):
        for node in ast.walk(fn.node):
            if (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == table and isinstance(node.value, ast.List)):
                found.append((fn.module, node.value))
    if not found:
        raise SystemExit(f"schema_extract: _init_variables {table} table not found")
    if len({ast.dump(node) for _mod, node in found}) > 1:
        raise SystemExit(f"schema_extract: the {table} copies differ")
    return found[0]


def desktop_tables(mods: dict) -> dict:
    """The three desktop tables, row for row (``DESKTOP_*`` in the generated data)."""
    mod, lst = find_settings_map(mods)
    settings_map = tuple(
        (const_str(key) or _desktop_unbuildable(mod, key, "key"), _table_sources(mod, sources),
         table_default(mod, default), converter_spec(mods, mod, conv))
        for key, sources, default, conv in _table_rows(mod, lst, 4, "settings_map"))
    out = {"settings_map": settings_map}
    for table in ("bool_vars", "str_vars"):
        mod, lst = _single_table(mods, table)
        out[table] = tuple(
            (const_str(var) or _desktop_unbuildable(mod, var, "attribute"),
             const_str(key) or _desktop_unbuildable(mod, key, "key"), table_default(mod, default))
            for var, key, default in _table_rows(mod, lst, 3, table))
    return out


def settings_default(mod: Module, node):
    if (isinstance(node, ast.Call) and _call_name(node) == "getattr" and len(node.args) >= 2
            and isinstance(node.args[0], ast.Name) and node.args[0].id == "self"):
        attr = const_str(node.args[1])
        if attr:
            return {"$attr": attr}
    return value_repr(mod, node)


_SAFE = ("safe_int", "safe_float")


def converter_spec(mods: dict, mod: Module, node):
    """Map a settings_map converter to a ConvSpec tuple (see settings_schema.coerce)."""
    if isinstance(node, ast.Name):
        if node.id in ("bool", "str", "int", "float", "list", "dict"):
            return (node.id,)
        return ("call", node.id)
    if isinstance(node, ast.Constant) and node.value is None:
        return ("none",)
    if isinstance(node, ast.Lambda) and len(node.args.args) == 1:
        arg = node.args.args[0].arg
        body = node.body
        spec = _safe_clamp(body, arg)
        if spec:
            return spec
        spec = _int_if_digits(body, arg)
        if spec:
            return spec
        spec = _choice(mods, body, arg)
        if spec:
            return spec
        spec = _str_or(body, arg)
        if spec:
            return spec
    return ("unknown", segment(mod, node))


def _is_arg(node, arg):
    return isinstance(node, ast.Name) and node.id == arg


def _num(node):
    ok, value = literal(node)
    if ok and isinstance(value, (int, float)) and not isinstance(value, bool):
        return True, value
    return False, None


def _safe_call(node, arg):
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _SAFE
            and len(node.args) == 2 and _is_arg(node.args[0], arg)):
        ok, default = _num(node.args[1])
        if ok:
            return node.func.id, default
    return None


def _minmax(node, which):
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == which
            and len(node.args) == 2 and not node.keywords):
        ok, bound = _num(node.args[0])
        if ok:
            return bound, node.args[1]
    return None


def _safe_clamp(body, arg):
    """safe_int(v, d) | max(lo, safe_int(v, d)) | min(hi, safe_int(v, d)) |
    min(hi, max(lo, safe_int(v, d))) | max(lo, min(hi, safe_int(v, d)))  (same for safe_float)."""
    direct = _safe_call(body, arg)
    if direct:
        return (direct[0], direct[1], None, None, "")
    outer_max = _minmax(body, "max")
    outer_min = _minmax(body, "min")
    if outer_max:
        lo, inner = outer_max
        direct = _safe_call(inner, arg)
        if direct:
            return (direct[0], direct[1], lo, None, "")
        inner_min = _minmax(inner, "min")
        if inner_min:
            hi, inner2 = inner_min
            direct = _safe_call(inner2, arg)
            if direct:
                return (direct[0], direct[1], lo, hi, "max_min")
    if outer_min:
        hi, inner = outer_min
        direct = _safe_call(inner, arg)
        if direct:
            return (direct[0], direct[1], None, hi, "")
        inner_max = _minmax(inner, "max")
        if inner_max:
            lo, inner2 = inner_max
            direct = _safe_call(inner2, arg)
            if direct:
                return (direct[0], direct[1], lo, hi, "min_max")
    return None


def _str_v(node, arg):
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "str"
            and len(node.args) == 1 and _is_arg(node.args[0], arg))


def _int_if_digits(body, arg):
    """int(v) if str(v).lstrip(chars).isdigit() else d"""
    if not isinstance(body, ast.IfExp):
        return None
    good = (isinstance(body.body, ast.Call) and isinstance(body.body.func, ast.Name)
            and body.body.func.id == "int" and len(body.body.args) == 1 and _is_arg(body.body.args[0], arg))
    test = body.test
    if not (good and isinstance(test, ast.Call) and isinstance(test.func, ast.Attribute)
            and test.func.attr == "isdigit" and not test.args):
        return None
    inner = test.func.value
    strip = None
    if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Attribute) and inner.func.attr == "lstrip":
        if len(inner.args) == 1 and const_str(inner.args[0]) is not None:
            strip = const_str(inner.args[0])
            inner = inner.func.value
        else:
            return None
    if not _str_v(inner, arg):
        return None
    ok, default = literal(body.orelse)
    if not ok:
        return None
    return ("int_if_digits", strip, default)


def _norm_lower(node, arg):
    """str(v).strip().lower()"""
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "lower"
            and not node.args and isinstance(node.func.value, ast.Call)
            and isinstance(node.func.value.func, ast.Attribute) and node.func.value.func.attr == "strip"
            and not node.func.value.args and _str_v(node.func.value.func.value, arg))


def _choice(mods, body, arg):
    """str(v).strip().lower() if str(v).strip().lower() in ALLOWED else d"""
    if not (isinstance(body, ast.IfExp) and _norm_lower(body.body, arg)):
        return None
    test = body.test
    if not (isinstance(test, ast.Compare) and len(test.ops) == 1 and isinstance(test.ops[0], ast.In)
            and _norm_lower(test.left, arg)):
        return None
    allowed_node = test.comparators[0]
    ok, allowed = literal(allowed_node)
    if not ok and isinstance(allowed_node, ast.Name):
        ok, allowed = module_constant(mods, allowed_node.id)
    if not ok:
        return None
    ok, default = literal(body.orelse)
    if not ok:
        return None
    return ("choice", tuple(allowed), default)


def _str_or(body, arg):
    """(str(v).strip() if v is not None else blank) or d"""
    if not (isinstance(body, ast.BoolOp) and isinstance(body.op, ast.Or) and len(body.values) == 2):
        return None
    first, second = body.values
    if not isinstance(first, ast.IfExp):
        return None
    strip_ok = (isinstance(first.body, ast.Call) and isinstance(first.body.func, ast.Attribute)
                and first.body.func.attr == "strip" and not first.body.args and _str_v(first.body.func.value, arg))
    test = first.test
    test_ok = (isinstance(test, ast.Compare) and len(test.ops) == 1 and isinstance(test.ops[0], ast.IsNot)
               and _is_arg(test.left, arg) and isinstance(test.comparators[0], ast.Constant)
               and test.comparators[0].value is None)
    blank_ok, blank = literal(first.orelse)
    default_ok, default = literal(second)
    if strip_ok and test_ok and blank_ok and default_ok:
        return ("str_or", blank, default)
    return None


# --------------------------------------------------------------------------- collection
@dataclass
class KeyInfo:
    key: str
    sm: list = field(default_factory=list)            # SMEntry, settings_map order
    init: set = field(default_factory=set)            # (kind, default)
    save: set = field(default_factory=set)            # (kind, default)
    dialog: set = field(default_factory=set)          # (default,)
    nested_defaults: set = field(default_factory=set) # (source, default)
    reads: set = field(default_factory=set)           # (default,) seen at any other read site
    env: set = field(default_factory=set)             # (ENV, site, default)
    var_names: set = field(default_factory=set)
    labels: list = field(default_factory=list)        # (priority, label, tooltip, group, tab)
    ui_sites: dict = field(default_factory=dict)      # site -> best distance
    choices: set = field(default_factory=set)
    bounds: set = field(default_factory=set)          # (lo, hi)
    widget_choices: set = field(default_factory=set)  # (priority, fn, ((value, label), ...), editable, index_coded)


class Collector:
    def __init__(self, src: Path):
        self.src = src
        self.mods = {}
        for name in dict.fromkeys(OWNER_MODULES + DIALOG_MODULES + EXTRA_MODULES):
            mod = load_module(src, name)
            if mod is not None:
                self.mods[name] = mod
        if "translator_gui.py" not in self.mods:
            raise SystemExit(f"schema_extract: {src / 'translator_gui.py'} not found")
        _REF_SRC["dir"] = str(src)
        _REF_SRC["cache"] = {}
        self.keys = {}
        self.var_map = {}          # var attr -> set(keys)
        self.widget_attrs = set()  # self.<attr> assigned a Qt widget anywhere
        self.unknown = {}
        self.sm_entries = []
        self.notes = []
        self.attr_values = {}      # self.default_* attribute -> value (for $attr defaults)
        self.attr_literals = {}    # self.<attr> = <literal> (first in startup order)
        self.startup_attrs = set() # self.<attr> assigned during desktop startup

    def info(self, key) -> KeyInfo:
        info = self.keys.get(key)
        if info is None:
            info = self.keys[key] = KeyInfo(key)
        return info

    # -- pass 0: widget attrs
    def collect_widget_attrs(self):
        for mod in self.mods.values():
            for node in ast.walk(mod.tree):
                if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                    if _is_widget_call(node.value):
                        for target in node.targets:
                            if is_self_attr(target):
                                self.widget_attrs.add(target.attr)

    # -- pass 1: settings_map + tables
    def collect_settings_map(self):
        self.sm_entries, self.unknown = parse_settings_map(self.mods)
        for entry in self.sm_entries:
            info = self.info(entry.key)
            info.sm.append(entry)
            for source in entry.sources:
                if ":" in source:
                    continue
                if source not in self.widget_attrs and not _looks_like_widget(source):
                    self.var_map.setdefault(source, set()).add(entry.key)
                    info.var_names.add(source)

    def collect_tables(self):
        found = set()
        for fn in owner_functions(self.mods):
            for node in ast.walk(fn.node):
                if (isinstance(node, ast.Assign) and len(node.targets) == 1
                        and isinstance(node.targets[0], ast.Name)
                        and node.targets[0].id in ("bool_vars", "str_vars")
                        and isinstance(node.value, ast.List)):
                    table = node.targets[0].id
                    found.add(table)
                    for elt in node.value.elts:
                        if not (isinstance(elt, ast.Tuple) and len(elt.elts) == 3
                                and const_str(elt.elts[0]) and const_str(elt.elts[1])):
                            self.notes.append(f"{table}: unexpected entry {segment(fn.module, elt)}")
                            continue
                        var, key = const_str(elt.elts[0]), const_str(elt.elts[1])
                        default = Fresh(fn.module, local_constants(fn.node), self.attr_literals)(elt.elts[2])
                        if table == "str_vars" and not _is_marker(default):
                            default = str(default)
                        info = self.info(key)
                        info.init.add(("table_" + table.split("_")[0], _hashable(default)))
                        info.var_names.add(var)
                        self.startup_attrs.add(var)
                        self.var_map.setdefault(var, set()).add(key)
        for table in ("bool_vars", "str_vars"):
            if table not in found:
                raise SystemExit(f"schema_extract: _init_variables {table} table not found")

    def collect_method_index(self):
        _METHODS["index"] = {}
        _METHODS["cache"] = {}
        for fn in owner_functions(self.mods):
            if fn.cls is not None:
                _METHODS["index"].setdefault(fn.name, []).append(fn)

    def collect_init_callees(self):
        """Owner functions called directly (self.m(...)) from a startup function: the GUI
        handlers desktop startup replays (_setup_gui at HEAD, _init_gui_backed_state after
        the shared-core move) run on a fresh install too."""
        _INIT_CALLEES.clear()
        funcs = owner_functions(self.mods)
        by_name = {}
        for fn in funcs:
            by_name.setdefault(fn.name, []).append(fn)
        excluded = SAVE_FUNCS | {n for names in ENV_SITES.values() for n in names if n not in INIT_FUNCS}
        roots = [fn for fn in funcs if "init" in categories(fn)]
        for fn in roots:
            for node in ast.walk(fn.node):
                if (isinstance(node, ast.Call) and is_self_attr(node.func)
                        and node.func.attr not in excluded):
                    for callee in by_name.get(node.func.attr, ()):
                        if not _contains_settings_map(callee):
                            _INIT_CALLEES.add(callee.qualname)

    def collect_startup_attrs(self):
        for fn in owner_functions(self.mods):
            if "init" not in categories(fn):
                continue
            for node in source_order(fn.node):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if is_self_attr(target):
                            self.startup_attrs.add(target.attr)
                            if target.attr.startswith("default_") and target.attr not in self.attr_values:
                                self.attr_values[target.attr] = value_repr(fn.module, node.value)
                            ok, value = literal(node.value)
                            if ok and target.attr not in self.attr_literals:
                                self.attr_literals[target.attr] = value
                elif isinstance(node, ast.Call) and _call_name(node) == "setattr" and len(node.args) >= 2:
                    attr = const_str(node.args[1])
                    if attr and isinstance(node.args[0], ast.Name) and node.args[0].id == "self":
                        self.startup_attrs.add(attr)

    def collect_init_var_assignments(self):
        """self.X = <expr reading exactly one config key> in startup functions -> var map."""
        for fn in owner_functions(self.mods):
            if "init" not in categories(fn):
                continue
            ctx = RefContext(fn.module, fn.node, {}, set())
            for node in ast.walk(fn.node):
                if isinstance(node, ast.Assign) and len(node.targets) == 1 and is_self_attr(node.targets[0]):
                    keys = {k for k, _d in ctx.refs(node.value) if "." not in k}
                    if len(keys) == 1:
                        attr = node.targets[0].attr
                        if attr not in self.widget_attrs and not attr.startswith("_"):
                            key = next(iter(keys))
                            self.var_map.setdefault(attr, set()).add(key)
                            self.info(key).var_names.add(attr)
        # Lazily created dialog vars: if not hasattr(self, 'x_var'): self.x_var = self.config.get('k', d)
        # (dialog modules only: translator_gui's own assignments are owner state, and they
        # move into the GUI-free mixins)
        for name in DIALOG_MODULES:
            mod = self.mods.get(name)
            if mod is None or name in OWNER_MODULES or name in DIALOG_MODULE_FUNCTIONS:
                continue
            for node in ast.walk(mod.tree):
                if isinstance(node, ast.Assign) and len(node.targets) == 1 and is_self_attr(node.targets[0]):
                    attr = node.targets[0].attr
                    if not attr.endswith("_var") or attr in self.widget_attrs:
                        continue
                    value = node.value
                    calls = [c for c in ast.walk(value) if _is_config_call(c) and c.args and const_str(c.args[0])]
                    keys = {const_str(c.args[0]) for c in calls}
                    if len(keys) == 1:
                        key = next(iter(keys))
                        if attr not in self.var_map:
                            self.var_map.setdefault(attr, set()).add(key)
                            self.info(key).var_names.add(attr)
                        if key in self.var_map.get(attr, ()):
                            for call in calls:
                                if len(call.args) > 1 and not _is_config_call(call.args[1]):
                                    self.info(key).dialog.add((_hashable(value_repr(mod, call.args[1])),))

    @property
    def known_keys(self):
        return set(self.keys)

    # -- pass 2: init / save / env records in owner functions
    def collect_owner_records(self):
        known = self.known_keys
        for fn in owner_functions(self.mods):
            cats = categories(fn)
            if not cats:
                continue
            ctx = RefContext(fn.module, fn.node, self.var_map, known)
            if "init" in cats:
                self._init_records(fn, ctx, "init")
            if "save" in cats:
                self._init_records(fn, ctx, "save")
            for cat in cats:
                if cat.startswith("env:"):
                    self._env_records(fn, ctx, cat[4:])

    def _init_records(self, fn: Func, ctx: RefContext, bucket: str):
        mod = fn.module
        if bucket == "init" and _fresh_install_noop(fn.node):
            return
        fresh = Fresh(mod, local_constants(fn.node), self.attr_literals)
        # reads under ``if 'k' in self.config:`` never run on a fresh install
        guarded = set()
        for node in ast.walk(fn.node):
            if isinstance(node, ast.If):
                test = node.test
                if (isinstance(test, ast.Compare) and len(test.ops) == 1 and isinstance(test.ops[0], ast.In)
                        and const_str(test.left) is not None and is_config_like(test.comparators[0])):
                    key = const_str(test.left)
                    for stmt in node.body:
                        for sub in ast.walk(stmt):
                            guarded.add((id(sub), key))
                    if bucket == "init":
                        for stmt in node.orelse:
                            if (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                                    and is_self_attr(stmt.targets[0])
                                    and key in self.var_map.get(stmt.targets[0].attr, ())):
                                value = fresh(stmt.value)
                                self.info(key).init.add(("var", _hashable(value)))

        def add(key, kind, default):
            if bucket == "save" and kind == "read" and default == NONE:
                return
            info = self.info(key)
            getattr(info, bucket).add((kind, _hashable(default)))

        top_level = set(id(s) for s in fn.node.body)
        for node in ast.walk(fn.node):
            # if 'k' not in self.config: self.config['k'] = v
            if isinstance(node, ast.If):
                test = node.test
                if (isinstance(test, ast.Compare) and len(test.ops) == 1 and isinstance(test.ops[0], ast.NotIn)
                        and const_str(test.left) is not None and is_config_like(test.comparators[0])):
                    key = const_str(test.left)
                    for stmt in node.body:
                        if (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                                and isinstance(stmt.targets[0], ast.Subscript)
                                and is_config_like(stmt.targets[0].value)
                                and const_str(stmt.targets[0].slice) == key):
                            add(key, "ifmissing", fresh(stmt.value))
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Subscript) and is_config_like(target.value):
                        key = const_str(target.slice)
                        if key is not None:
                            kind = "forced" if id(node) in top_level else "write"
                            add(key, kind, fresh(node.value))
            elif isinstance(node, ast.Call):
                func = node.func
                if (isinstance(func, ast.Attribute) and func.attr == "setdefault"
                        and is_config_like(func.value) and node.args and const_str(node.args[0]) is not None):
                    dflt = node.args[1] if len(node.args) > 1 else None
                    add(const_str(node.args[0]), "setdefault", fresh(dflt) if dflt is not None else None)
                elif (isinstance(func, ast.Attribute) and func.attr in ("get",)
                        and is_config_like(func.value) and node.args and const_str(node.args[0]) is not None):
                    key = const_str(node.args[0])
                    if (id(node), key) in guarded:
                        continue
                    dflt = node.args[1] if len(node.args) > 1 else None
                    add(key, "read", fresh(dflt) if dflt is not None else NONE)
        if bucket == "init":
            for node in ast.walk(fn.node):
                if isinstance(node, ast.Assign) and len(node.targets) == 1 and is_self_attr(node.targets[0]):
                    attr = node.targets[0].attr
                    if attr in self.var_map:
                        for call in ast.walk(node.value):
                            key = const_str(call.args[0]) if _is_config_call(call) and call.args else None
                            if key not in self.var_map[attr] or (id(call), key) in guarded:
                                continue
                            value = fresh(node.value)
                            if _is_expr_marker(value):
                                dflt = call.args[1] if len(call.args) > 1 else None
                                value = fresh(dflt) if dflt is not None else NONE
                            add(key, "var", value)
        # for k, v in DEFAULTS.items(): self.config.setdefault(k, v)
        # for k in [KEYS]: self.config[k] = <expr>
        literals = {}
        for node in ast.walk(fn.node):
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                if isinstance(node.value, (ast.Dict, ast.List, ast.Tuple)):
                    literals.setdefault(node.targets[0].id, node.value)
        for node in ast.walk(fn.node):
            if not isinstance(node, ast.For):
                continue
            iterable = node.iter
            if (isinstance(iterable, ast.Call) and isinstance(iterable.func, ast.Attribute)
                    and iterable.func.attr == "items" and not iterable.args):
                source = iterable.func.value
                source = literals.get(source.id) if isinstance(source, ast.Name) else source
                if not (isinstance(source, ast.Dict) and isinstance(node.target, ast.Tuple)
                        and len(node.target.elts) == 2 and all(isinstance(e, ast.Name) for e in node.target.elts)):
                    continue
                kname, vname = node.target.elts[0].id, node.target.elts[1].id
                for sub in ast.walk(node):
                    if (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute) and sub.func.attr == "setdefault"
                            and is_config_like(sub.func.value) and len(sub.args) == 2
                            and isinstance(sub.args[0], ast.Name) and sub.args[0].id == kname
                            and isinstance(sub.args[1], ast.Name) and sub.args[1].id == vname):
                        for key_node, value_node in zip(source.keys, source.values):
                            key = const_str(key_node)
                            if key is not None:
                                add(key, "setdefault", fresh(value_node))
            else:
                source = literals.get(iterable.id) if isinstance(iterable, ast.Name) else iterable
                if not (isinstance(source, (ast.List, ast.Tuple)) and isinstance(node.target, ast.Name)):
                    continue
                keys = [const_str(e) for e in source.elts]
                if not keys or any(k is None for k in keys):
                    continue
                kname = node.target.id
                for sub in node.body:
                    if (isinstance(sub, ast.Assign) and len(sub.targets) == 1
                            and isinstance(sub.targets[0], ast.Subscript) and is_config_like(sub.targets[0].value)
                            and isinstance(sub.targets[0].slice, ast.Name) and sub.targets[0].slice.id == kname):
                        for key in keys:
                            add(key, "write", fresh(sub.value))

        if bucket == "save":
            # _config_bool('k', d) / _config_int('k', d, lo, hi) / _config_float / _config_choice
            for node in ast.walk(fn.node):
                if isinstance(node, ast.Call) and _call_name(node) in ("_config_bool", "_config_int", "_config_float", "_config_choice") and node.args:
                    key = const_str(node.args[0])
                    if key is None:
                        continue
                    info = self.info(key)
                    name = _call_name(node)
                    dflt = value_repr(mod, node.args[1]) if len(node.args) > 1 else NONE
                    info.save.add(("read", _hashable(dflt)))
                    if name in ("_config_int", "_config_float"):
                        lo = literal(node.args[2])[1] if len(node.args) > 2 else None
                        hi = literal(node.args[3])[1] if len(node.args) > 3 else None
                        info.bounds.add((lo, hi))
                    if name == "_config_choice" and len(node.args) > 2:
                        ok, allowed = literal(node.args[2])
                        if ok:
                            info.choices.add(tuple(sorted(allowed)))

    def _env_records(self, fn: Func, ctx: RefContext, site: str):
        for env_name, value in env_assignments(fn.node):
            for key, default in ctx.refs(value):
                self.info(key).env.add((env_name, site, _hashable(default)))
        # bounded helpers inside env builders carry min/max
        for node in ast.walk(fn.node):
            if isinstance(node, ast.Call) and _call_name(node) in ("_bounded_config_int", "_bounded_config_float") and len(node.args) >= 3:
                key = const_str(node.args[1])
                if key is None:
                    continue
                lo = literal(node.args[3])[1] if len(node.args) > 3 else None
                hi = literal(node.args[4])[1] if len(node.args) > 4 else None
                self.info(key).bounds.add((lo, hi))
            elif isinstance(node, ast.Call) and _call_name(node) == "_choice_config_value" and len(node.args) >= 4:
                key = const_str(node.args[1])
                ok, allowed = literal(node.args[3])
                if key is not None and ok:
                    self.info(key).choices.add(tuple(sorted(allowed)))

    # -- (e) nested settings
    def collect_nested(self):
        for parent, mod_name, qualname, how in NESTED_DEFAULT_SOURCES:
            mod = self.mods.get(mod_name)
            if mod is None:
                self.notes.append(f"nested defaults: {mod_name} missing")
                continue
            fn = next((f for f in iter_functions(mod) if f.qualname == qualname), None)
            if fn is None:
                self.notes.append(f"nested defaults: {mod_name}:{qualname} missing")
                continue
            dict_node = None
            if how == "return":
                for node in ast.walk(fn.node):
                    if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict):
                        dict_node = node.value
                        break
            else:
                attr = how.split(":", 1)[1]
                for node in ast.walk(fn.node):
                    if (isinstance(node, ast.Assign) and len(node.targets) == 1
                            and is_self_attr(node.targets[0], attr) and isinstance(node.value, ast.Dict)):
                        dict_node = node.value
                        break
            if dict_node is None:
                self.notes.append(f"nested defaults: no dict literal in {mod_name}:{qualname}")
                continue
            for path, value in flatten_dict(mod, dict_node, (parent,)):
                self.info(".".join(path)).nested_defaults.add(("defaults", _hashable(value)))
        # save_config's qa setdefault block
        for fn in owner_functions(self.mods):
            if "save" not in categories(fn):
                continue
            for node in ast.walk(fn.node):
                if (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                        and node.targets[0].id == "default_qa_settings" and isinstance(node.value, ast.Dict)):
                    for path, value in flatten_dict(fn.module, node.value, ("qa_scanner_settings",)):
                        self.info(".".join(path)).nested_defaults.add(("save", _hashable(value)))
                elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "update"
                        and isinstance(node.func.value, ast.Name) and node.func.value.id == "default_qa_settings"
                        and node.args and isinstance(node.args[0], ast.Dict)):
                    for path, value in flatten_dict(fn.module, node.args[0], ("qa_scanner_settings",)):
                        self.info(".".join(path)).nested_defaults.add(("save", _hashable(value)))
        # qa env
        for mod_name, func_name, local, parent, site in NESTED_ENV_FUNCS:
            mod = self.mods.get(mod_name)
            fn = next((f for f in iter_functions(mod) if f.qualname == func_name), None) if mod else None
            if fn is None:
                self.notes.append(f"nested env: {mod_name}:{func_name} missing")
                continue
            ctx = RefContext(mod, fn.node, {}, set(), nested_hints={local: (parent,)})
            for env_name, value in env_assignments(fn.node):
                for key, default in ctx.refs(value):
                    if key.startswith(parent + "."):
                        self.info(key).env.add((env_name, site, _hashable(default)))

    # -- (d) labels / tooltips and UI sites
    def collect_ui(self):
        known = self.known_keys
        widget_key = {}
        for entry in self.sm_entries:
            for source in entry.sources:
                if ":" not in source and (source in self.widget_attrs or _looks_like_widget(source)):
                    widget_key.setdefault(source, set()).add(entry.key)
        for priority, name in enumerate(DIALOG_MODULES):
            mod = self.mods.get(name)
            if mod is None:
                continue
            attr_paths = _module_attr_paths(mod)
            hints = {k: (v,) for k, v in NESTED_NAME_HINTS.get(name, {}).items()}
            funcs = list(iter_functions(mod))
            if name in DIALOG_MODULE_FUNCTIONS:
                funcs = [fn for fn in funcs if fn.qualname.split(".", 1)[0] in DIALOG_MODULE_FUNCTIONS[name]]
            if name == "translator_gui.py":
                # methods moved into the GUI-free mixins are still TranslatorGUI methods
                funcs += [fn for fn in owner_functions(self.mods) if fn.module.name != name]
            roots = _ui_roots(name, funcs)
            reach = _reachability(mod, funcs, roots)
            for fn in funcs:
                ctx = RefContext(fn.module, fn.node, self.var_map, known, nested_hints=hints, attr_paths=attr_paths)
                site_info = reach.get(fn.qualname)
                # every key the function touches gets the function's UI site
                touched = set()
                for stmt in fn.node.body:
                    for key, _default in ctx.refs(stmt, into_defs=True):
                        touched.add(key)
                if (name, fn.qualname) in CONFIG_UPDATE_FUNCS:
                    for node in ast.walk(fn.node):
                        if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict):
                            touched.update(k for k in (const_str(item) for item in node.value.keys if item) if k)
                for node in ast.walk(fn.node):
                    if (isinstance(node, ast.Assign) and len(node.targets) == 1
                            and isinstance(node.targets[0], ast.Subscript)):
                        tgt = node.targets[0]
                        key = key_str(fn.module, tgt.slice)
                        if key and is_config_like(tgt.value):
                            touched.add(key)
                        elif key:
                            base = ctx.path_of(tgt.value)
                            if base:
                                touched.add(".".join(base + (key,)))
                for key in touched:
                    info = self.info(key)
                    if site_info:
                        site, dist = site_info
                        best = info.ui_sites.get(site)
                        if best is None or dist < best:
                            info.ui_sites[site] = dist
                labels, bind_defaults, choice_records = widget_labels(fn.module, fn, ctx, widget_key)
                for key, strength, nkeys, label, tooltip, group, tab in labels:
                    self.info(key).labels.append((priority, strength, nkeys, fn.name, label, tooltip, group, tab))
                for key, pairs, editable, index_coded in choice_records:
                    self.info(key).widget_choices.add((priority, fn.name, pairs, editable, index_coded))
                if name != "translator_gui.py":
                    for key, default in bind_defaults:
                        self.info(key).dialog.add((_hashable(default),))

    # -- generic reads (keys referenced elsewhere in owner modules)
    def collect_other_reads(self):
        """Keys read anywhere else in the owner code (and their call-site defaults)."""
        for fn in owner_functions(self.mods):
            if categories(fn):
                continue
            for node in ast.walk(fn.node):
                if _is_config_call(node) and node.args and const_str(node.args[0]) is not None:
                    key = const_str(node.args[0])
                    dflt = node.args[1] if len(node.args) > 1 else None
                    if _is_config_call(dflt):
                        dflt = None
                    self.info(key).reads.add((_hashable(value_repr(fn.module, dflt)) if dflt is not None else NONE,))
                elif (isinstance(node, ast.Subscript) and is_config_like(node.value)
                        and const_str(node.slice) is not None):
                    self.info(const_str(node.slice))

    def run(self):
        self.collect_widget_attrs()
        self.collect_method_index()
        self.collect_init_callees()
        self.collect_settings_map()
        self.collect_startup_attrs()
        self.collect_tables()
        self.collect_init_var_assignments()
        self.collect_owner_records()
        self.collect_other_reads()
        self.collect_nested()
        self.collect_ui()
        return self


def _is_widget_call(call: ast.Call) -> bool:
    name = _call_name(call)
    if not name:
        return False
    if name in WIDGET_FACTORIES:
        return True
    if re.match(r"^Q[A-Z]", name):
        return True
    return name.endswith(WIDGET_SUFFIXES)


def _is_input_call(call: ast.Call) -> bool:
    name = _call_name(call) or ""
    return name in INPUT_FACTORIES or name.endswith(INPUT_SUFFIXES)


def _looks_like_widget(attr: str) -> bool:
    return attr.endswith(("_checkbox", "_entry", "_combo", "_check", "_edit", "_spin", "_slider",
                          "_cb", "_button", "_text")) or attr in ("trans_temp", "trans_history")


# Records are sets, so defaults are stored as their repr (always hashable, and two equal
# literals from a verbatim move collapse); _VALUES maps the repr back to the value.
_VALUES = {}


def _hashable(value):
    if isinstance(value, str) and value == NONE:
        return NONE
    key = repr(value)
    _VALUES.setdefault(key, value)
    return key


def _unhash(key):
    if key == NONE:
        return NONE
    return _VALUES[key]


def _is_marker(value):
    """{'$ref': 'module:NAME'[, 'as': 'list'|'tuple']} | {'$expr': src} | {'$attr': name}"""
    if not isinstance(value, dict) or not value:
        return False
    keys = set(value)
    return keys in ({"$ref"}, {"$expr"}, {"$attr"}, {"$ref", "as"})


def _is_expr_marker(value):
    return isinstance(value, dict) and set(value) == {"$expr"}


def flatten_dict(mod: Module, node: ast.Dict, prefix: tuple):
    fresh = Fresh(mod, {}, {})
    for key_node, value_node in zip(node.keys, node.values):
        key = const_str(key_node)
        if key is None:
            continue
        path = prefix + (key,)
        if (isinstance(value_node, ast.Dict) and value_node.keys and all(const_str(k) for k in value_node.keys)
                and ".".join(path) not in NESTED_DATA_LEAVES):
            yield from flatten_dict(mod, value_node, path)
        else:
            yield path, fresh(value_node)


def env_assignments(func_node):
    """Yield (ENV_NAME, value node) for every env write in a function."""
    for node in ast.walk(func_node):
        if isinstance(node, ast.Dict):
            for key_node, value in zip(node.keys, node.values):
                name = const_str(key_node)
                if name and ENV_RE.match(name):
                    yield name, value
        elif isinstance(node, (ast.Assign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Subscript) and not is_config_like(target.value):
                    name = const_str(target.slice)
                    if name and ENV_RE.match(name):
                        yield name, node.value
        elif isinstance(node, ast.Call):
            name = _call_name(node)
            if name in ENV_SETTER_FUNCS and len(node.args) >= 2:
                if isinstance(node.func, ast.Attribute) and is_config_like(node.func.value):
                    continue
                env = const_str(node.args[0])
                if env and ENV_RE.match(env):
                    yield env, node.args[1]
        elif isinstance(node, ast.Tuple) and len(node.elts) == 2:
            env = const_str(node.elts[0])
            if env and ENV_RE.match(env) and not isinstance(node.ctx, ast.Store):
                yield env, node.elts[1]


def _module_attr_paths(mod: Module):
    """self.<attr> = <nested dict expr> anywhere in the module (e.g. MangaSettingsDialog.settings)."""
    out = {}
    for node in ast.walk(mod.tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and is_self_attr(node.targets[0]):
            ctx = RefContext.__new__(RefContext)
            ctx.paths, ctx.attr_paths = {}, {}
            path = RefContext.path_of(ctx, node.value)
            if path:
                out.setdefault(node.targets[0].attr, path)
    return out


def _ui_roots(mod_name: str, funcs):
    roots = []
    for name, func, site in UI_SITE_ROOTS:
        if name != mod_name:
            continue
        if func == "*":
            roots.append(("*", site))
        elif func.endswith(".*"):
            cls = func[:-2]
            for fn in funcs:
                if fn.cls == cls:
                    roots.append((fn.qualname, site))
        else:
            for fn in funcs:
                if fn.name == func:
                    roots.append((fn.qualname, site))
    return roots


def _reachability(mod: Module, funcs, roots):
    """qualname -> (site, distance). Explicit roots win; '*' covers the rest at distance 9."""
    by_name = {}
    for fn in funcs:
        by_name.setdefault(fn.name, []).append(fn)
    refs = {}
    for fn in funcs:
        names = set()
        for node in ast.walk(fn.node):
            if is_self_attr(node):
                names.add(node.attr)
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                names.add(node.func.id)
        refs[fn.qualname] = names
    result = {}
    root_names = {q for q, _s in roots if q != "*"}
    for qual, site in roots:
        if qual == "*":
            continue
        frontier = [qual]
        if qual not in result or result[qual][1] > 0:
            result[qual] = (site, 0)
        seen = {qual}
        for dist in range(1, UI_REACH_DEPTH + 1):
            nxt = []
            for q in frontier:
                for name in refs.get(q, ()):
                    for fn in by_name.get(name, ()):
                        if fn.qualname in seen or fn.qualname in root_names or fn.name in UI_REACH_EXCLUDE:
                            continue
                        seen.add(fn.qualname)
                        prev = result.get(fn.qualname)
                        if prev is None or prev[1] > dist:
                            result[fn.qualname] = (site, dist)
                        nxt.append(fn.qualname)
            frontier = nxt
    for qual, site in roots:
        if qual == "*":
            for fn in funcs:
                result.setdefault(fn.qualname, (site, 9))
    return result


def _str_items(value):
    if isinstance(value, (list, tuple)) and value and all(isinstance(v, str) for v in value):
        return list(value)
    return None


def _pair_items(value):
    if (isinstance(value, (list, tuple)) and value
            and all(isinstance(v, tuple) and len(v) == 2 and all(isinstance(x, str) for x in v) for v in value)):
        return [tuple(v) for v in value]
    return None


def _local_literals(func_node):
    """NAME -> literal value of the function's (and its closures') literal assignments."""
    out = {}
    for node in ast.walk(func_node):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            ok, value = literal(node.value)
            if ok:
                out.setdefault(node.targets[0].id, value)
    return out


def _items_value(node, mod: Module, local_lits: dict):
    """The literal list behind an addItems argument: a literal, a local / module constant,
    or list(...) / tuple(...) of one."""
    if isinstance(node, ast.Call) and _call_name(node) in ("list", "tuple") and len(node.args) == 1:
        node = node.args[0]
    ok, value = literal(node)
    if ok:
        return value
    if isinstance(node, ast.Name):
        if node.id in local_lits:
            return local_lits[node.id]
        if node.id in mod.consts:
            return mod.consts[node.id]
    return None


def _loop_item_pairs(func_node, mod: Module, local_lits: dict):
    """widget id -> [(data, label)] from ``for label, data in <pairs>: w.addItem(label, data)``."""
    out = {}
    for node in ast.walk(func_node):
        if not (isinstance(node, ast.For) and isinstance(node.target, ast.Tuple) and len(node.target.elts) == 2
                and all(isinstance(e, ast.Name) for e in node.target.elts)):
            continue
        pairs = _pair_items(_items_value(node.iter, mod, local_lits))
        if not pairs:
            continue
        names = [e.id for e in node.target.elts]
        for sub in ast.walk(node):
            if (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute) and sub.func.attr == "addItem"
                    and len(sub.args) == 2 and all(isinstance(a, ast.Name) and a.id in names for a in sub.args)):
                wid = _widget_id(sub.func.value)
                if wid:
                    li, di = names.index(sub.args[0].id), names.index(sub.args[1].id)
                    out.setdefault(wid, []).extend((p[di], p[li]) for p in pairs)
    return out


def _statement_depths(body, depth=0, out=None):
    """id(statement) -> nesting depth, over the statements _linear_statements yields."""
    out = {} if out is None else out
    for stmt in body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        out[id(stmt)] = depth
        for attr in ("body", "orelse", "finalbody"):
            inner = getattr(stmt, attr, None)
            if isinstance(inner, list) and inner and isinstance(inner[0], ast.stmt):
                _statement_depths(inner, depth + 1, out)
        for handler in getattr(stmt, "handlers", []) or []:
            _statement_depths(handler.body, depth + 1, out)
    return out


def _record_assign(last_assign: dict, name: str, value, depth: int):
    """Reaching assignments of a local: an assignment replaces those at its depth or deeper
    (``idx = ...`` reused per combo) and adds to shallower ones (``if x is None: x = ...``)."""
    kept = [(d, v) for d, v in last_assign.get(name, ()) if d < depth]
    kept.append((depth, value))
    last_assign[name] = kept


def _resolve_local(node, last_assign: dict, depth=0):
    """A local name -> the value of its latest reaching assignment."""
    while isinstance(node, ast.Name) and node.id in last_assign and depth < 4:
        node = last_assign[node.id][-1][1]
        depth += 1
    return node


def _index_kind(node, last_assign: dict, depth=0):
    """How ``setCurrentIndex(node)`` picks the item: ('map', {value: index}) for a literal
    ``{...}.get(value)``, ('lookup', None) for a ``findText`` / ``findData`` result, else
    ('unknown', None) (an index from a method or arithmetic: the items are not the values)."""
    if isinstance(node, ast.Name) and node.id in last_assign and depth < 4:
        kinds = [_index_kind(value, last_assign, depth + 1) for _d, value in last_assign[node.id]]
        for kind in kinds:
            if kind[0] == "map":
                return kind
        return kinds[-1]
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get"
            and isinstance(node.func.value, ast.Dict)):
        ok, mapping = literal(node.func.value)
        if ok and mapping and all(isinstance(k, str) for k in mapping) and all(
                isinstance(v, int) and not isinstance(v, bool) for v in mapping.values()):
            return "map", mapping
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in CHOICE_TIE_METHODS:
        return "lookup", None
    if isinstance(node, ast.Call) and _call_name(node) in ("max", "min", "int") and node.args:
        # setCurrentIndex(max(0, combo.findData(value))): still a lookup of the stored value
        for arg in node.args:
            kind = _index_kind(arg, last_assign, depth + 1)
            if kind[0] != "unknown":
                return kind
    return "unknown", None


def _precise_refs(node, ctx: RefContext, last_assign: dict):
    """Config keys an expression reads, resolving local names through their reaching
    assignments only (a function reusing ``idx`` for several combos must not tie them all)."""
    pctx = copy.copy(ctx)
    pctx.locals = {name: [value for _d, value in values] for name, values in last_assign.items()}
    pctx._busy = set()
    return [key for key, _default in pctx.refs(node)]


def _combo_choices(info):
    """``(((value, label), ...), index_coded)`` of one combo, or None."""
    items = info["items"]
    if not items or info["dynamic"]:
        return None
    index_map = info["index_map"]
    if info["index_unknown"] and not index_map:
        return None                     # positioned by a computed index: items are not the values
    if index_map:
        by_index = {}
        for value, index in index_map.items():
            by_index.setdefault(index, value)
        if not all(i in by_index for i in range(len(items))):
            return None
        pairs = [(by_index[i], items[i][1]) for i in range(len(items))]
        index_coded = all(isinstance(v, str) and v.isdigit() and int(v) == i for i, (v, _l) in enumerate(pairs))
    else:
        pairs = list(items)
        index_coded = False
    seen, unique = set(), []
    for value, label in pairs:
        if value not in seen:
            seen.add(value)
            unique.append((value, label))
    return tuple(unique), index_coded


def widget_labels(mod: Module, fn: Func, ctx: RefContext, widget_key: dict):
    """([(key, label, tooltip, group, tab)], [(key, default)], [(key, pairs, editable,
    index_coded)]) for widgets created in one function; the defaults come from the
    expressions that fill the widgets, the choices from the combo boxes' items."""
    local_defs = {}
    nested = []
    for node in ast.walk(fn.node):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node is not fn.node:
            local_defs.setdefault(node.name, node)
            nested.append(node)
    local_lits = _local_literals(fn.node)
    loop_pairs = _loop_item_pairs(fn.node, mod, local_lits)
    # the function body, then every nested def (dialogs are often built in closures);
    # None separates the segments and resets the label / group / tab context
    stmts = [None] + list(_linear_statements(fn.node.body))
    for node in sorted(nested, key=lambda n: (n.lineno, n.col_offset)):
        stmts.append(None)
        stmts.extend(_linear_statements(node.body))
    widgets = {}
    bind_defaults = set()
    helper_labels = []
    last_label = None
    group = None
    pending_tab = []
    current_tab = None
    containers = {}            # container widget id -> statement index it was created at
    depths = _statement_depths(fn.node.body)
    for node in nested:
        _statement_depths(node.body, 0, depths)
    last_assign = {}          # local name -> [(depth, value)] reaching assignments (choice ties)
    for idx, stmt in enumerate(stmts):
        if stmt is None:
            last_label, group, current_tab, pending_tab = None, None, None, []
            last_assign = {}
            continue
        if (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name)):
            _record_assign(last_assign, stmt.targets[0].id, stmt.value, depths.get(id(stmt), 0))
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.value, ast.Call):
            target = stmt.targets[0]
            wid = _widget_id(target)
            call = stmt.value
            if wid and _is_widget_call(call):
                cname = _call_name(call)
                text = joined_text(call.args[0]) if call.args else None
                if cname == "QGroupBox" and text:
                    group = text.strip()
                    continue
                if cname == "QLabel":
                    if text and text.strip():
                        last_label = (text.strip(), idx)
                    continue
                if cname in ("QWidget", "QFrame", "QScrollArea", "QHBoxLayout", "QVBoxLayout",
                             "QGridLayout", "QFormLayout", "QTabWidget", "QSplitter"):
                    containers[wid] = idx
                    continue
                if not _is_input_call(call):
                    continue
                near = None
                if not text and last_label and idx - last_label[1] <= LABEL_WINDOW:
                    near = last_label[0]
                info = {"text": (text or "").strip() or None, "near": near, "tooltip": None,
                        "keys": set(), "handler_keys": set(), "group": group, "tab": current_tab,
                        "combo": (cname or "").endswith(COMBO_SUFFIXES), "items": [], "dynamic": False,
                        "index_map": None, "index_unknown": False, "editable": False, "choice_keys": set(),
                        "source_keys": set()}
                attr = wid[5:] if wid.startswith("self.") else None
                if attr and attr in widget_key:
                    info["keys"].update(widget_key[attr])
                    info["source_keys"].update(widget_key[attr])
                info["keys"].update(WIDGET_KEY_PINS.get((mod.name, fn.name, wid), ()))
                widgets[wid] = info
                pending_tab.append(wid)
                continue
        for call in _calls_in(stmt):
            func = call.func
            if _call_name(call) == "QLabel" and call.args:
                text = joined_text(call.args[0])
                if text and text.strip():
                    last_label = (text.strip(), idx)
                continue
            if isinstance(func, ast.Name) and func.id in local_defs or is_self_attr(func):
                # helper rows: _create_slider_row(layout, "Top-P:", self, 'top_p_var', ...)
                label_text, keys = None, set()
                for arg in call.args:
                    text = const_str(arg)
                    if text is None:
                        continue
                    if text in ctx.var_map:
                        keys.update(ctx.var_map[text])
                    elif text in ctx.known_keys:
                        keys.add(text)
                    elif label_text is None and re.search(r"[A-Za-z]", text) and (" " in text or text.endswith(":")):
                        label_text = text.strip()
                if label_text and keys:
                    for key in sorted(keys):
                        helper_labels.append((key, 0, len(keys), label_text, None, group, current_tab))
                continue
            if not isinstance(func, ast.Attribute):
                continue
            owner = _widget_id(func.value)
            if func.attr == "setToolTip" and owner in widgets and call.args:
                text = joined_text(call.args[0])
                if text:
                    widgets[owner]["tooltip"] = text.strip()
            elif func.attr in BIND_METHODS and owner in widgets and call.args:
                for key, default in ctx.refs(call.args[0]):
                    widgets[owner]["keys"].add(key)
                    if default != NONE:
                        bind_defaults.add((key, _hashable(default)))
                if widgets[owner]["combo"] and func.attr in ("setCurrentIndex", "setCurrentText", "setEditText"):
                    target = _resolve_local(call.args[0], last_assign)
                    if func.attr == "setCurrentIndex":
                        kind, mapping = _index_kind(call.args[0], last_assign)
                        if kind == "map":
                            widgets[owner]["index_map"] = mapping
                        elif kind == "unknown":
                            widgets[owner]["index_unknown"] = True
                        if kind == "lookup":
                            target = call.args[0]       # the lookup's argument is in its refs
                    widgets[owner]["choice_keys"].update(_precise_refs(target, ctx, last_assign))
            elif func.attr == "addItems" and owner in widgets and call.args and widgets[owner]["combo"]:
                items = _str_items(_items_value(call.args[0], mod, local_lits))
                if items is None:
                    widgets[owner]["dynamic"] = True
                else:
                    widgets[owner]["items"].extend((s, s) for s in items)
            elif func.attr == "addItem" and owner in widgets and call.args and widgets[owner]["combo"]:
                label = const_str(call.args[0])
                data = const_str(call.args[1]) if len(call.args) > 1 else label
                if label is not None and data is not None:
                    widgets[owner]["items"].append((data, label))
                elif owner not in loop_pairs:
                    widgets[owner]["dynamic"] = True
            elif func.attr in CHOICE_TIE_METHODS and owner in widgets and call.args:
                widgets[owner]["choice_keys"].update(_precise_refs(call.args[0], ctx, last_assign))
            elif func.attr == "setEditable" and owner in widgets and call.args:
                ok, value = literal(call.args[0])
                if ok and value:
                    widgets[owner]["editable"] = True
            elif func.attr == "connect" and isinstance(func.value, ast.Attribute) and func.value.attr in SIGNALS:
                owner = _widget_id(func.value.value)
                if owner in widgets and call.args:
                    widgets[owner]["handler_keys"].update(_handler_keys(call.args[0], local_defs, ctx))
            elif func.attr == "addTab" and len(call.args) >= 2:
                title = joined_text(call.args[1])
                page = _widget_id(call.args[0])
                if title:
                    title = title.strip()
                    if page in containers and idx - containers[page] <= 4:
                        # page added right after it was created: the widgets follow
                        current_tab = title
                    else:
                        # page filled first, then added
                        for wid in pending_tab:
                            if wid in widgets and widgets[wid]["tab"] is None:
                                widgets[wid]["tab"] = title
                        current_tab = None
                    pending_tab = []
    choice_records = []
    for wid, info in widgets.items():
        if not info["combo"]:
            continue
        if wid in loop_pairs and not info["items"]:
            info["items"].extend(loop_pairs[wid])
        found = _combo_choices(info)
        if found is None:
            continue
        pairs, index_coded = found
        for key in sorted(info["source_keys"] | info["choice_keys"]):
            choice_records.append((key, pairs, info["editable"], index_coded))
    labels = list(helper_labels)
    for wid, info in widgets.items():
        label = info["text"] or info["near"]
        if not (label or info["tooltip"] or info["group"] or info["tab"]):
            continue
        # strength 0: the widget shows the key (settings_map source / setChecked etc.);
        # strength 1: its handler writes the key (weaker: one control often drives several)
        direct = info["keys"]
        indirect = info["handler_keys"] - direct
        for key in sorted(direct):
            labels.append((key, 0, len(direct), label, info["tooltip"], info["group"], info["tab"]))
        for key in sorted(indirect):
            labels.append((key, 1, len(indirect), label, info["tooltip"], info["group"], info["tab"]))
    return labels, [(k, _unhash(d)) for k, d in sorted(bind_defaults)], choice_records


def _widget_id(node):
    if isinstance(node, ast.Name):
        return node.id
    if is_self_attr(node):
        return "self." + node.attr
    return None


def _linear_statements(body):
    for stmt in body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        yield stmt
        for attr in ("body", "orelse", "finalbody"):
            inner = getattr(stmt, attr, None)
            if isinstance(inner, list) and inner and isinstance(inner[0], ast.stmt):
                yield from _linear_statements(inner)
        for handler in getattr(stmt, "handlers", []) or []:
            yield from _linear_statements(handler.body)


def _calls_in(stmt):
    if isinstance(stmt, (ast.Expr, ast.Assign, ast.AugAssign, ast.Return)):
        for node in ast.walk(stmt):
            if isinstance(node, ast.Call):
                yield node


def _handler_keys(node, local_defs, ctx: RefContext):
    keys = set()
    body = None
    if isinstance(node, ast.Lambda):
        body = node.body
    elif isinstance(node, ast.Name) and node.id in local_defs:
        body = local_defs[node.id]
    if body is None:
        return keys
    for sub in ast.walk(body):
        if isinstance(sub, ast.Assign):
            for target in sub.targets:
                if isinstance(target, ast.Subscript) and is_config_like(target.value):
                    key = const_str(target.slice)
                    if key:
                        keys.add(key)
                elif is_self_attr(target) and target.attr in ctx.var_map:
                    keys.update(ctx.var_map[target.attr])
        elif isinstance(sub, ast.Call) and _call_name(sub) == "setattr" and len(sub.args) >= 2:
            attr = const_str(sub.args[1])
            if attr and attr in ctx.var_map:
                keys.update(ctx.var_map[attr])
    return keys


# --------------------------------------------------------------------------- merge
def norm_value(value):
    """Comparable form of a default: 'True' == True == 1, '5' == 5.0."""
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if text.lower() in ("true", "false"):
            return 1.0 if text.lower() == "true" else 0.0
        try:
            return float(text)
        except ValueError:
            return text
    if isinstance(value, list):
        return ("$list",) + tuple(norm_value(v) for v in value)
    return value


INIT_PRIORITY = ("ifmissing", "setdefault", "forced", "table_bool", "table_str", "var", "read", "write")


def build_entry(info: KeyInfo, collector):
    unknown = collector.unknown
    entry = {}
    flags = []
    origins = []
    if info.sm:
        last = info.sm[-1]
        origins.append("settings_map")
        entry["converter"] = last.converter
        if last.save_default is not None:
            entry["save_default"] = last.save_default
        if len(info.sm) > 1:
            flags.append("settings_map_duplicate")
            variants = {(e.sources, repr(e.save_default), e.converter) for e in info.sm}
            if len(variants) > 1:
                flags.append("settings_map_duplicate_differs")
        widget_sources = []
        for e in info.sm:
            for s in e.sources:
                if s not in info.var_names and s not in widget_sources:
                    widget_sources.append(s)
        if widget_sources:
            entry["widget_sources"] = tuple(widget_sources)
        sm_vars = []
        for e in info.sm:
            for s in e.sources:
                if s in info.var_names and s not in sm_vars:
                    sm_vars.append(s)
        var_names = sm_vars + sorted(v for v in info.var_names if v not in sm_vars)
    else:
        var_names = sorted(info.var_names)
    if var_names:
        entry["var_names"] = tuple(var_names)
    if info.key in unknown:
        flags.append("unknown_converter")

    init_default = _pick(info.init, INIT_PRIORITY)
    if info.init:
        origins.append("init")
        if init_default is not _MISSING:
            entry["init_default"] = _unhash(init_default)
    save_extra = _pick(info.save, ("setdefault", "forced", "ifmissing", "read", "write"))
    if save_extra is not _MISSING and "save_default" not in entry:
        entry["save_default"] = _unhash(save_extra)
    if info.save:
        origins.append("save")
    nested = _pick(info.nested_defaults, ("defaults", "save"))
    if nested is not _MISSING:
        origins.append("nested")
        entry["nested_default"] = _unhash(nested)
    dialog_defaults = sorted({d[0] for d in info.dialog}, key=repr)
    if dialog_defaults:
        origins.append("dialog")
        entry["dialog_default"] = _unhash(dialog_defaults[0])
    read_defaults = sorted({d[0] for d in info.reads if d[0] != NONE}, key=repr)
    if info.env:
        origins.append("env")
        env = sorted({(name, site, d) for name, site, d in info.env},
                     key=lambda t: (SITE_ORDER.index(t[1]) if t[1] in SITE_ORDER else 99, t[0], repr(t[2])))
        entry["env"] = tuple((name, site, _unhash(d)) for name, site, d in env)
    if read_defaults:
        origins.append("read")

    for field_name in ("init_default", "save_default", "nested_default", "dialog_default"):
        if field_name in entry:
            entry[field_name] = resolve_attr(entry[field_name], collector.attr_values)

    # effective fresh-install default (display only; see settings_schema):
    # settings_map keys: startup save_config stores converter(<startup source value>)
    # when a source attribute exists at startup, else the settings_map default;
    # other keys: the init default, else the save default; then nested / dialog /
    # call-site defaults.
    candidates = []
    if info.sm:
        last = info.sm[-1]
        present = any(_source_present(src, collector) for src in last.sources)
        if present and "init_default" in entry:
            converted = _convert(last.converter, entry["init_default"])
            entry["default"] = converted
            entry["default_source"] = "init"
        if "save_default" in entry:
            candidates.append(("save", entry["save_default"]))
        if "init_default" in entry:
            candidates.append(("init", entry["init_default"]))
    else:
        # config writes at startup (init, then the startup save_config) beat var values
        init_write = _pick(info.init, ("ifmissing", "setdefault", "forced"))
        save_write = _pick(info.save, ("setdefault", "ifmissing"))
        if init_write is not _MISSING:
            candidates.append(("init", resolve_attr(_unhash(init_write), collector.attr_values)))
        if save_write is not _MISSING:
            candidates.append(("save", resolve_attr(_unhash(save_write), collector.attr_values)))
        if "init_default" in entry:
            candidates.append(("init", entry["init_default"]))
        if "save_default" in entry:
            candidates.append(("save", entry["save_default"]))
    if "nested_default" in entry:
        candidates.append(("nested", entry["nested_default"]))
    if "dialog_default" in entry:
        candidates.append(("dialog", entry["dialog_default"]))
    # run sites first (translation, glossary, ...), startup/save last
    env_defaults = [(site, _unhash(d)) for _n, site, d in
                    sorted(info.env, key=lambda t: (SITE_ORDER.index(t[1]) if t[1] in SITE_ORDER else 99, t[0], t[2]))
                    if d != NONE]
    for site, d in env_defaults:
        candidates.append(("env:" + site, d))
    for d in read_defaults:
        candidates.append(("read", _unhash(d)))
    if candidates and "default" not in entry:
        chosen = next((c for c in candidates if not _is_expr_marker(c[1])), candidates[0])
        entry["default"] = chosen[1]
        entry["default_source"] = chosen[0]

    # discrepancies: distinct defaults between sites (never resolved here)
    disc = []
    by_norm = {}
    for source, value in _discrepancy_candidates(info, entry):
        if _is_marker(value) or isinstance(value, dict):
            continue
        by_norm.setdefault(_freeze(norm_value(value)), []).append((source, value))
    if len(by_norm) > 1:
        parts = []
        for _norm, items in sorted(by_norm.items(), key=lambda kv: repr(kv[0])):
            sources = sorted({s for s, _v in items})
            representative = sorted(items, key=lambda item: (item[0], repr(item[1])))[0][1]
            parts.append(f"{'/'.join(sources)}={representative!r}")
        disc.append("defaults differ: " + ", ".join(parts))
    if disc:
        entry["discrepancies"] = tuple(disc)

    choices = set()
    for c in info.choices:
        choices.update(c)
    conv = entry.get("converter")
    # combo choices: the dialog module priority, then the function name (never positions)
    widget = sorted(info.widget_choices, key=lambda t: (t[0], t[1], repr(t[2])))
    if conv and conv[0] == "choice":
        entry["choices"] = tuple(conv[1])
        for _priority, _fn, pairs, _editable, _coded in widget:
            if {v for v, _l in pairs} == set(conv[1]) and any(v != label for v, label in pairs):
                entry["choices"] = pairs             # the same values, shown with their labels
                break
    elif widget and not isinstance(entry.get("default"), bool):
        # (a bool setting is never a combo value: a combo tied to it through a fallback read
        # such as the old enable_auto_glossary migration only names its sibling's values)
        _priority, _fn, pairs, editable, index_coded = widget[0]
        if index_coded:
            pairs = tuple((int(v), label) for v, label in pairs)
            flags.append("index_coded_choices")
        if any(v != label for v, label in pairs):
            entry["choices"] = pairs
        else:
            entry["choices"] = tuple(v for v, _l in pairs)
        if editable:
            flags.append("editable_choices")
    elif choices:
        entry["choices"] = tuple(sorted(choices))
    entry["type"] = infer_type(info.key, entry)
    if entry["type"] == "choice" and "editable_choices" in flags:
        entry["type"] = "str"                        # any text is allowed; the items are suggestions
    lo, hi = _bounds(info, conv)
    if lo is not None:
        entry["minimum"] = lo
    if hi is not None:
        entry["maximum"] = hi
    if info.labels:
        # (binding strength, module priority, keys on that widget, function name, label...):
        # positions in the file are not used, so a verbatim move keeps the choice
        best = sorted(info.labels, key=lambda t: (t[1], t[0], t[2], t[3], 0 if t[4] else 1,
                                                  t[4] or "", t[5] or "", t[6] or "", t[7] or ""))
        label = next((t[4] for t in best if t[4]), None)
        tooltip = next((t[5] for t in best if t[5]), None)
        group = next((t[6] for t in best if t[6]), None)
        tab = next((t[7] for t in best if t[7]), None)
        if label:
            entry["label"] = label
        if tooltip:
            entry["tooltip"] = tooltip
        if group:
            entry["ui_group"] = group
        if tab:
            entry["ui_tab"] = tab
    if info.ui_sites:
        # direct (distance 0-1: the builder or a handler it names) before indirect, then the
        # curated desktop order; a closure turned into a method stays in the same bucket
        entry["ui_sites"] = tuple(s for s, _d in sorted(info.ui_sites.items(),
                                                       key=lambda kv: (0 if kv[1] <= 1 else 1, _SITE_RANK.get(kv[0], 999), kv[0])))
    if "." in info.key:
        entry["parent"] = info.key.split(".", 1)[0]
    if origins:
        entry["origins"] = tuple(origins)
    if flags:
        entry["flags"] = tuple(flags)
    return entry


def resolve_attr(value, attr_values, depth=0):
    if isinstance(value, dict) and set(value) == {"$attr"} and depth < 4:
        target = attr_values.get(value["$attr"])
        if target is not None:
            return resolve_attr(target, attr_values, depth + 1)
    return value


def _source_present(source, collector) -> bool:
    if source.startswith("config:"):
        key = source.split(":", 1)[1]
        info = collector.keys.get(key)
        return bool(info and any(kind in ("ifmissing", "setdefault", "forced") for kind, _d in info.init))
    return source in collector.startup_attrs


_CONVERTER_IMPL = {}


def _apply_converter_impl():
    """settings_schema.apply_converter, loaded from the src tree being generated (or the
    repo's src/ for a tree without one) under a private name: the generator never imports
    from sys.path, so the output does not depend on what the caller imported before."""
    src = _REF_SRC["dir"]
    path = Path(src) / "settings_schema.py" if src else None
    if path is None or not path.is_file():
        path = DEFAULT_SRC / "settings_schema.py"
    key = str(path)
    if key not in _CONVERTER_IMPL:
        import importlib.util
        name = f"_schema_extract_settings_schema_{len(_CONVERTER_IMPL)}"
        spec = importlib.util.spec_from_file_location(name, key)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module          # dataclasses resolve annotations through sys.modules
        spec.loader.exec_module(module)
        _CONVERTER_IMPL[key] = module.apply_converter
    return _CONVERTER_IMPL[key]


def _convert(conv, value):
    """Apply a settings_map ConvSpec to a literal init default (startup save_config)."""
    if conv is None or _is_marker(value) or conv[0] in ("call", "unknown"):
        return value
    apply_converter = _apply_converter_impl()     # a broken settings_schema.py must fail loudly
    try:
        return apply_converter(conv, value)
    except Exception:                             # the desktop lambda raises too: keep the raw value
        return value


_SITE_RANK = {}
for _i, (_m, _f, _site) in enumerate(UI_SITE_ROOTS):
    _SITE_RANK.setdefault(_site, _i)


def _pick(records, priority):
    """First kind in priority order with a usable default. Kinds whose values are only
    runtime expressions ($expr) are skipped while another kind has a value; within a
    kind, literals beat markers."""
    by_kind = {}
    for kind, d in records:
        if d == NONE or d == "None" and kind == "setdefault":
            continue
        by_kind.setdefault(kind, set()).add(d)
    fallback = _MISSING
    for kind in priority:
        values = by_kind.get(kind)
        if not values:
            continue
        usable = [v for v in values if not _is_expr_marker(_unhash(v))]
        if not usable:
            if fallback is _MISSING:
                fallback = sorted(values)[0]
            continue
        literal_values = [v for v in usable if not _is_marker(_unhash(v))]
        return sorted(literal_values or usable)[0]
    return fallback


def _freeze(value):
    if isinstance(value, list):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, dict):
        return tuple(sorted((k, _freeze(v)) for k, v in value.items()))
    if isinstance(value, set):
        return tuple(sorted(map(repr, value)))
    return value


def _discrepancy_candidates(info: KeyInfo, entry):
    out = []
    for kind, d in info.init:
        if d != NONE and kind != "write":
            out.append((f"init:{kind}", _unhash(d)))
    if info.sm and info.sm[-1].save_default is not None:
        out.append(("save", info.sm[-1].save_default))
    for kind, d in info.save:
        if d != NONE and kind in ("setdefault", "ifmissing", "write", "forced"):
            out.append((f"save:{kind}", _unhash(d)))
    for name, site, d in info.env:
        if d != NONE:
            out.append((f"env:{site}:{name}", _unhash(d)))
    for source, d in info.nested_defaults:
        out.append((f"nested:{source}", _unhash(d)))
    return sorted(out, key=lambda item: (item[0], repr(item[1])))


def infer_type(key: str, entry) -> str:
    conv = entry.get("converter")
    if conv:
        kind = conv[0]
        mapping = {"bool": "bool", "int": "int", "float": "float", "list": "list", "dict": "dict",
                   "safe_int": "int", "safe_float": "float", "int_if_digits": "int", "choice": "choice",
                   "call": "str", "str_or": "str"}
        if kind in mapping:
            base = mapping[kind]
            if base != "str":
                return base
    # A typed default or an index-coded combo settles the type before the name heuristics
    # (use_multi_api_keys is a bool, multi_api_keys a list of key dicts, the number spacing
    # "token fix" an index code; none of them is a secret).
    default = entry.get("default", entry.get("save_default"))
    choices = entry.get("choices")
    if choices and all(isinstance(c, tuple) and isinstance(c[0], int) for c in choices):
        return "int"
    if isinstance(default, bool):
        return "bool"
    if isinstance(default, int):
        return "int"
    if isinstance(default, float):
        return "float"
    if isinstance(default, list):
        return "list"
    if isinstance(default, dict) and default.get("as") in ("list", "tuple"):
        return "list"
    if isinstance(default, dict) and not _is_marker(default):
        return "dict"
    lowered = key.lower().rsplit(".", 1)[-1]
    if (re.search(r"(^|_)(api_key|secret|password|token)s?($|_)", lowered)
            and not re.search(r"token(s)?_(limit|budget|count|threshold|concurrency|timeout|fix)|max_tokens|_tokens$|token_limit", lowered)):
        return "secret"
    if lowered in ("api_key",) or lowered.endswith("_api_key") or lowered.endswith("_api_keys"):
        return "secret"
    if re.search(r"(_path|_dir|_directory|_folder|_file)$", lowered) or lowered == "google_cloud_credentials":
        return "path"
    if choices:
        return "choice"
    return "str"


def _bounds(info: KeyInfo, conv):
    lo = hi = None
    if conv and conv[0] in ("safe_int", "safe_float"):
        lo, hi = conv[2], conv[3]
    if lo is None and hi is None and info.bounds:
        values = sorted(info.bounds, key=repr)
        lo, hi = values[0]
    return lo, hi


# --------------------------------------------------------------------------- output
HEADER = '''\
# GENERATED by src/mobile/tools/schema_extract.py - do not edit by hand.
# Regenerate: python src/mobile/tools/schema_extract.py   (tests/test_settings_schema.py
# fails when this file is stale). Pure data, Python 3.10; read through settings_schema.py.
#
# SETTINGS[key] fields (absent = unknown):
#   type, default (effective fresh-install default: init default, else save default,
#   else nested/dialog/call-site default; display only), default_source, init_default,
#   save_default (settings_map), nested_default, dialog_default, converter (ConvSpec),
#   var_names, widget_sources, env ((ENV, site, call-site default or '<none>'), ...),
#   label, tooltip, ui_group, ui_tab, ui_sites, choices, minimum, maximum, parent,
#   origins, flags, discrepancies.
# Non-literal defaults: {'$ref': 'module:NAME'}, {'$attr': 'self attribute'},
# {'$expr': 'source text'}.
#
# DESKTOP_SETTINGS_MAP / DESKTOP_BOOL_VARS / DESKTOP_STR_VARS (U9 P5b): the desktop tables row
# for row; the desktop builds its settings_map / bool_vars / str_vars from them through
# settings_schema.desktop_settings_map / desktop_bool_vars / desktop_str_vars. Rows:
#   settings_map (key, sources, default, ConvSpec); bool_vars / str_vars (attribute, key, default).
#   default: ('value', literal) | ('ref', 'module:NAME') | ('attr', name, default)
#            | ('config', key, default). The literals live in src/mobile/tools/frozen_desktop_tables.py.
'''


def generate(src: Path) -> str:
    _VALUES.clear()
    collector = Collector(src).run()
    unknown = collector.unknown
    settings = {}
    for key in sorted(collector.keys):
        info = collector.keys[key]
        if not _is_setting_key(key):
            continue
        settings[key] = build_entry(info, collector)
    # drop container nodes: a key that is a prefix of a nested key (the parent dict
    # itself, e.g. qa_scanner_settings, and intermediate dicts)
    nested = [k for k in settings if "." in k]
    containers = set()
    for key in nested:
        parts = key.split(".")
        for i in range(1, len(parts)):
            containers.add(".".join(parts[:i]))
    for key in containers:
        settings.pop(key, None)
    order = tuple(e.key for e in collector.sm_entries)
    lines = [HEADER.rstrip("\n"), "", f"GENERATOR_VERSION = {GENERATOR_VERSION}", ""]
    lines.append("# save_config settings_map keys in source order (duplicates kept; the last wins).")
    lines.append("SETTINGS_MAP_ORDER = " + pprint.pformat(order, width=100))
    lines.append("")
    lines.append("# settings_map converters the ConvSpec pattern table does not recognise.")
    lines.append("UNKNOWN_CONVERTERS = " + pprint.pformat(dict(sorted(unknown.items())), width=100))
    lines.append("")
    tables = desktop_tables(collector.mods)
    for table, comment in (
            ("settings_map", "save_config settings_map rows (key, sources, default, ConvSpec), in order."),
            ("bool_vars", "_init_variables bool_vars rows (attribute, key, default), in order."),
            ("str_vars", "_init_variables str_vars rows (attribute, key, default), in order.")):
        lines.append(f"# {comment}")
        lines.append(f"DESKTOP_{table.upper()} = " + pprint.pformat(tables[table], width=100))
        lines.append("")
    lines.append("SETTINGS = {")
    for key, entry in settings.items():
        body = pprint.pformat(entry, width=96, sort_dicts=True)
        body = body.replace("\n", "\n    ")
        lines.append(f"    {key!r}: {body},")
    lines.append("}")
    lines.append("")
    return "\n".join(lines)


def _is_setting_key(key: str) -> bool:
    """Config keys are identifiers (dotted for nested settings); a few legacy keys use '-'."""
    return bool(key) and re.match(r"^[A-Za-z_][A-Za-z0-9_.\-]*$", key) is not None


def output_path(src: Path) -> Path:
    return src / OUTPUT_NAME


def write_output(src: Path, text: str) -> Path:
    path = output_path(src)
    newline = "\r\n" if os.name == "nt" else "\n"
    if path.is_file():
        raw = path.read_bytes()
        newline = "\r\n" if b"\r\n" in raw[:4096] else "\n"
    with open(path, "w", encoding="utf-8", newline=newline) as fh:
        fh.write(text)
    return path


def read_committed(src: Path) -> str | None:
    path = output_path(src)
    if not path.is_file():
        return None
    with open(path, encoding="utf-8", newline="") as fh:
        return fh.read().replace("\r\n", "\n")


def report(src: Path) -> str:
    collector = Collector(src).run()
    lines = [f"modules: {', '.join(sorted(collector.mods))}",
             f"settings_map entries: {len(collector.sm_entries)} ({len({e.key for e in collector.sm_entries})} keys)",
             f"keys: {len(collector.keys)}",
             f"unknown converters: {len(collector.unknown)}"]
    for key, src_text in sorted(collector.unknown.items()):
        lines.append(f"  {key}: {src_text}")
    for note in collector.notes:
        lines.append(f"note: {note}")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC, help="src/ directory (default: repo src/)")
    parser.add_argument("--check", action="store_true", help="fail when the committed file is stale")
    parser.add_argument("--stdout", action="store_true", help="print the generated module")
    parser.add_argument("--report", action="store_true", help="print a summary")
    args = parser.parse_args(argv)
    src = args.src.resolve()
    if args.report:
        print(report(src))
        return 0
    text = generate(src)
    if args.stdout:
        sys.stdout.write(text)
        return 0
    if args.check:
        committed = read_committed(src)
        if committed != text:
            print(f"{output_path(src)} is stale; run python src/mobile/tools/schema_extract.py", file=sys.stderr)
            return 1
        print(f"{output_path(src)} is up to date")
        return 0
    path = write_output(src, text)
    print(f"wrote {path} ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
