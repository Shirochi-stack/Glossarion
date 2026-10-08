#!/usr/bin/env python3
"""Freeze the desktop code scheduled to move in U1-U3 into a legacy oracle.

The oracle is a byte-for-byte copy of the original method bodies, taken from
``git show <sha>:src/<file>`` (never from the working tree), written to::

    tests/parity/legacy/legacy_<sha12>.py               class LegacyMethods (TranslatorGUI)
    tests/parity/legacy/legacy_<sha12>__<module>.py     frozen helpers from other modules, and
                                                        (U2+) whole shared mixin modules
                                                        (owner_state, run_env, settings_persistence)

``load_legacy()`` executes those files into private namespaces whose globals
come from the frozen imports/constants/functions of the original module. Names
that cannot be frozen (computed module globals such as ``CONFIG_FILE``) are
listed in ``FREEZE_MANIFEST['runtime_globals']`` and resolved from the live
module at load time; the capture harness overrides the path-like ones with the
sandbox.

Usage (repository root)::

    python tests/parity/freeze_legacy.py [--sha REV]

Notes:
* translator_gui.py starts with a UTF-8 BOM, so sources are decoded with
  ``utf-8-sig`` and CRLF is normalised before ``ast.parse``.
* Methods are selected by name and closed over ``self.<name>`` /
  ``TranslatorGUI.<name>`` / ``cls.<name>`` references; GUI-only methods
  (logging, watchdog widgets, executor) are never frozen and are provided by
  ``fakes.FakeState`` recorders instead (``FREEZE_MANIFEST['recorded_methods']``).
* ``__init__`` is frozen as synthetic blocks (``legacy_pre_config_block``,
  ``legacy_init_block`` = the config block, ``legacy_default_prompts_block``,
  ``legacy_watchdog_block``) and the GUI-backed state that ``_setup_gui``
  produces is frozen as ``legacy_gui_state_block`` / ``legacy_gui_handlers_block``
  (statements copied verbatim from the create_* section builders).
"""

from __future__ import annotations

import argparse
import ast
import builtins
import hashlib
import os
import pprint
import subprocess
import symtable
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path

PARITY_DIR = Path(__file__).resolve().parent
TESTS_DIR = PARITY_DIR.parent
REPO_ROOT = TESTS_DIR.parent
SRC_DIR = REPO_ROOT / "src"
LEGACY_DIR = PARITY_DIR / "legacy"

FREEZER_VERSION = 2  # v2: shared mixin modules frozen too (manifest 'frozen_mixins')

TG_FILE = "translator_gui.py"
TG_CLASS = "TranslatorGUI"

# ---------------------------------------------------------------------------
# What to freeze
# ---------------------------------------------------------------------------

#: Methods exercised by the goldens; their self-reference closure is frozen too.
TG_ENTRY_METHODS = (
    # run_env (U2): main builder + helpers listed in plan section 2
    "_get_environment_variables",
    "_resolve_glossary_for_env",
    "_glossary_env_mappings",
    "_strict_matching_env_dict",
    "_unified_glossary_env_dict",
    "_custom_prefix_routes_env_json",
    "_ollama_settings_env_json",
    "_get_qa_scanner_settings_json",
    "_translation_batching_mode_for_env",
    "_glossary_batching_mode_for_env",
    "_export_multipass_runtime_env",
    "_export_chapter_range_runtime_env",
    "_apply_direct_text_runtime_environment",
    "_metadata_only_environment_for_file",
    "_apply_forced_streaming_environment",
    "_format_translation_anti_duplicate_settings",
    "_log_translation_anti_duplicate_settings",
    "_current_glossary_request_env",
    "_live_bool_setting",
    "_live_text_setting",
    "_current_auto_glossary_mode",
    "_glossary_contextual_env_value",
    "_glossary_skip_title_header_only_env_value",
    "_glossary_add_minimal_pass_env_value",
    "_glossary_match_engine_env_value",
    "_context_mode_from_flags",
    "_model_is_image_gen",
    "_model_is_video_gen",
    "_is_generative_output_mode",
    "_get_output_mode",
    "_get_allowed_image_output_mode",
    "_get_allowed_video_output_mode",
    "_active_translation_output_mode",
    "_get_output_base_dir",
    "_subtitle_zip_output_info",
    "_resolve_translation_output_dir",
    "_output_side_glossary_backup_dir_for_source",
    "_current_glossary_cjk_script_filter_enabled",
    "_resolve_max_retry_tokens",
    "_resolve_max_retries",
    "_compression_chunk_budget",
    "_get_multipass_refinement_mode",
    "_sync_multipass_refinement_mode_from_combo",
    "_live_chapter_range_settings",
    "_parse_chapter_range_text",
    "_get_scan_phase_mode",
    "_normalize_custom_prefix_endpoint_type",
    "_is_valid_custom_prefix_endpoint_type",
    "_normalize_custom_prefix_routes",
    "_sync_custom_prefix_routes_env",
    "initialize_environment_variables",
    # owner_state (U2)
    "_init_variables",
    "_init_default_prompts",
    "_sanitize_config_prompts",
    "_get_protected_prompt_profiles",
    "_migrate_strict_matching_config",
    "_upgrade_special_file_exact",
    "_coerce_live_bool",
    # text_jobs / compile runners (U3) - exercised with stubbed backend entry points
    "_process_text_file",
    "_extract_glossary_from_text_file",
    "run_epub_converter_direct",
    "run_pdf_converter_direct",
    "_run_parallel_metadata_files",
    # input_preparation (U3): ZIP / HTML / subtitle-ZIP input resolution
    "_convert_zip_input_to_epub_if_needed",
    "_resolve_zip_inputs_for_translation",
    "_extract_subtitle_zip_input_if_needed",
    "_has_epub_conversion_inputs",
    # stop_control (U3): the desktop stop protocol (tier T trace tests drive it)
    "stop_translation",
    # settings persistence (U2 _collect_live_settings/_export_settings_env); desktop
    # startup runs it once from the glossary shortcut handler
    "save_config",
)

#: U4: TranslatorGUI methods that stay desktop handlers but now call the shared GUI-free
#: rules (settings_rules: model-route control visibility, chunk-size / compression helpers)
#: and model catalog core (model_catalog_core: tombstones, poll markers, provider refresh,
#: Model Manager save order and custom prefix validation). Tier D fuzzes each of them
#: (moved_functions.REWIRED): frozen legacy vs the working-tree TranslatorGUI method.
TG_ENTRY_METHODS_U4 = (
    # model-route-driven login / key / Google-credential controls (settings_rules.route_*)
    "on_model_change",
    "_model_needs_google_creds",
    "_iter_enabled_key_pool_models",
    "_has_google_creds_model_in_key_pools",
    "_has_vertex_model_in_key_pools",
    "_has_authgpt_in_key_pools",
    "_has_authgrok_in_key_pools",
    "_authgpt_pool_route_requested",
    "_authgrok_pool_route_requested",
    "_has_authgem_in_key_pools",
    "_has_authgem_vertex_in_key_pools",
    "_has_authcd_in_key_pools",
    "_collect_auth_account_ids_from_pools",
    "_authgem_vertex_control_model",
    # main-window Chunk Size field (settings_rules compression helpers)
    "_on_chunk_size_edited",
    "_apply_chunk_size",
    "_remember_manual_chunk_size",
    "_hold_manual_chunk_size",
    # model catalog (model_catalog_core)
    "_restore_removed_model_choices",
    "_ensure_polled_model_marker_state",
    "_expire_polled_model_markers",
    "_apply_polled_model_icons",
    "_apply_provider_model_catalog_refresh",
    "_save_model_order",
    "_collect_custom_prefix_routes_from_table",
    "_save_model_manager_state",
)
TG_ENTRY_METHODS = TG_ENTRY_METHODS + TG_ENTRY_METHODS_U4

#: Frozen verbatim for later steps (U3 trace tier) but NOT closed over or exercised in U0.
TG_FREEZE_ONLY_METHODS = (
    "run_translation_thread",
    "run_translation_direct",
    "run_glossary_extraction_direct",
    "_reset_prompt_profile_to_default",
    # U3: run-start resets (stop_control) and the translation / glossary pipelines
    "run_glossary_extraction_thread",
    "stop_glossary_extraction",
    "_reset_stop_flags_if_idle",
    "_process_image_folder_for_glossary",
    "_collect_translation_qa_failures",
    "_filter_translation_qa_failures_to_current_range",
    "_qa_failure_matches_resolution_request",
    "_prepare_multipass_qa_refinement_run",
    "_clear_translation_run_overrides",
    "_log_translation_qa_failure_summary",
    "auto_load_glossary_for_file",
    "_run_generative_prompt_mode",
)

#: GUI-only TranslatorGUI methods: never frozen, provided by fakes.FakeState recorders.
TG_RECORDED_METHODS = frozenset({
    "append_log",
    "append_log_direct",
    "append_log_with_api_error_detection",
    "_queue_gui_log_message",
    "_schedule_log_autoscroll",
    "_scroll_log_after_append",
    "_should_log_autoscroll",
    "_reset_api_watchdog_progress",
    "_create_watchdog_snapshot",
    "_ensure_executor",
    "update_run_button",
    "_update_manual_glossary_status",
    "_update_compression_token_budget_label",
    # U4: GUI-only callees of the TG_ENTRY_METHODS_U4 handlers (login buttons and their
    # token-store snapshots, account-slot combos, model combo / completer / poll border,
    # the mouse-wheel guard, the consolidated poll log and the queued full refresh)
    "_update_target_lang_state",
    "_update_authgpt_login_status",
    "_update_authgrok_login_status",
    "_update_authcd_login_status",
    "_update_authgem_login_status",
    "_update_ocagy_login_status",
    "_update_authza_login_status",
    "_update_antigravity_login_status",
    "_auth_status_snapshot",
    "_fetch_authgem_projects",
    "_reposition_authgem_project_combo",
    "_refresh_auth_account_arrows",
    "_set_model_poll_border_active",
    "_refresh_model_combo_catalog",
    "_refresh_model_search_poll_state",
    "_apply_combobox_mousewheel_lock",
    "_log_provider_model_catalog_feedback",
    "_launch_pending_full_provider_catalog_refresh",
})

#: Shared GUI-free mixin modules (U2+). When they exist at the frozen SHA the
#: TranslatorGUI body no longer holds the moved methods: the freezer then seeds
#: its closure with the TranslatorGUI methods the mixins reference (hooks, GUI
#: helpers), saves each module's source at that SHA as
#: ``legacy_<sha12>__<module>.py`` (manifest ``frozen_mixins``) and the owner
#: factory adds those FROZEN mixin classes as bases. The live modules are never
#: used by the legacy side: imports of a frozen mixin module from frozen code
#: resolve to its frozen copy.
SHARED_MIXIN_MODULES = (
    # U3, in TranslatorGUI's base order (shared mixins first, pipelines before env/state)
    ("translation_pipeline", "TranslationPipelineMixin"),
    ("text_jobs", "TextJobsMixin"),
    ("input_preparation", "InputPreparationMixin"),
    # U2
    ("settings_persistence", "SettingsPersistenceMixin"),
    ("run_env", "RunEnvMixin"),
    ("owner_state", "ConfigStateMixin"),
)

#: GUI-free helper modules the moved code imports (U3+: job hooks, scoped process state,
#: the stop protocol). They hold no owner mixin of their own (TextJobsMixin and
#: InputPreparationMixin inherit job_runner.JobHooksMixin), but they are frozen whole like
#: the mixin modules so the legacy side never runs their live working-tree versions.
#: U7: the image / generative-only and RPG Maker runners (TranslationPipelineMixin inherits
#: image_job.ImageJobMixin and rpgmaker_job.RpgMakerJobMixin, like JobHooksMixin above).
#: U9 (P5b): save_config's settings_map and _init_variables' bool_vars / str_vars are built by
#: settings_schema.desktop_settings_map / desktop_bool_vars / desktop_str_vars from the generated
#: rows in settings_schema_data, so both are frozen whole too: an oracle frozen after P5b keeps the
#: tables of its SHA instead of the working tree's (settings_schema resolves its converter /
#: default modules with importlib.import_module, which _load_frozen_mixins routes the same way).
SHARED_HELPER_MODULES = ("job_runner", "stop_control", "image_job", "rpgmaker_job",
                         "settings_schema", "settings_schema_data")

#: (method, statement prefix) - GUI-backed *state* assignments made while _setup_gui builds widgets.
GUI_STATE_STATEMENTS = (
    ("create_file_section", "self.vertex_location_var = "),
    ("create_file_section", "self.deep_scan_var = "),
    ("_create_model_section", "default_model = "),
    ("_create_model_section", "self.model_var = default_model"),
    ("_create_settings_section", "self.context_mode_var = "),
    ("_create_settings_section", "self.translation_history_rolling_var = True"),
    ("_create_settings_section", "self.config['translation_history_rolling'] = True"),
)

#: (method, statement prefix) - startup handlers _setup_gui runs once the widgets exist (desktop order).
GUI_HANDLER_STATEMENTS = (
    ("_create_settings_section", "self._on_disable_temperature_toggle()"),
    # nested handler + the explicit startup call (26044 / 26254-26258): syncs the
    # glossary toggles to the shortcut combo AND runs save_config() at startup
    ("_create_settings_section", "def _on_auto_glossary_shortcut_changed(index):"),
    ("_create_settings_section", "try:\n    _on_auto_glossary_shortcut_changed("),
    ("_create_settings_section", "self._on_context_mode_changed()"),
    ("_create_prompt_section", "saved_lang = "),
    ("_create_prompt_section", "glossary_lang = "),
    ("_create_prompt_section", "if saved_lang and glossary_lang and"),
    ("_create_prompt_section", "self.update_target_language(final_lang)"),
    ("_setup_gui", "if hasattr(self, 'profile_var') and self.profile_var in self.prompt_profiles:"),
    ("_setup_gui", "self._update_auto_compression_factor()"),
)

#: U2 layout: the GUI-backed state is ConfigStateMixin._init_gui_backed_state()
#: (called by __init__ right before _setup_gui) and the startup handlers are
#: shared mixin methods the section builders call at the same points.
GUI_STATE_STATEMENTS_U2 = (
    ("__init__", "self._init_gui_backed_state()"),
)
GUI_HANDLER_STATEMENTS_U2 = (
    ("_create_settings_section", "self._on_disable_temperature_toggle()"),
    ("_create_settings_section", "try:\n    self._on_auto_glossary_shortcut_changed("),
    ("_create_settings_section", "self._on_context_mode_changed()"),
    ("_create_prompt_section", "final_lang = self._resolve_startup_target_language()"),
    ("_create_prompt_section", "self.update_target_language(final_lang)"),
    ("_setup_gui", "self._init_active_profile_prompt()"),
    ("_setup_gui", "self._update_auto_compression_factor()"),
)

#: Statement specs per layout, tried in order (the first that matches wins).
GUI_STATE_LAYOUTS = (("legacy", GUI_STATE_STATEMENTS), ("u2", GUI_STATE_STATEMENTS_U2))
GUI_HANDLER_LAYOUTS = (("legacy", GUI_HANDLER_STATEMENTS), ("u2", GUI_HANDLER_STATEMENTS_U2))


@dataclass(frozen=True)
class ExternalSpec:
    module: str
    functions: tuple = ()
    classes: dict = field(default_factory=dict)  # class -> methods (closure inside the class)
    patch: tuple = ()  # attributes patched into the live module while legacy code runs
    bind: tuple = ()  # functions bound onto the owner as methods (setup_other_settings_methods)


EXTERNALS = (
    # output_naming (U1) + owner_state GUI-backed init (U2)
    ExternalSpec(
        "other_settings",
        functions=("initialize_extraction_variables", "_rename_output_files_for_retain"),
        patch=("initialize_extraction_variables", "_rename_output_files_for_retain"),
    ),
    # ollama_settings (U1)
    ExternalSpec(
        "ollama_settings_dialog",
        functions=("ollama_settings_json", "normalize_ollama_settings"),
        patch=("ollama_settings_json",),
    ),
    # metadata_defaults (U1): MetadataBatchTranslatorUI(self) inside the __init__ config block
    ExternalSpec(
        "metadata_batch_translator",
        classes={"MetadataBatchTranslatorUI": ("__init__", "_initialize_default_prompts")},
        patch=("MetadataBatchTranslatorUI",),
    ),
    # config_store (U1): save_config() -> self._backup_config_file() (bound by
    # setup_other_settings_methods from other_settings, defined in config_backup).
    # _restore_config_from_backup stays a recorder: it opens a modal QMessageBox.
    ExternalSpec(
        "config_backup",
        functions=("_backup_config_file",),
        bind=("_backup_config_file",),
    ),
)

#: setup_other_settings_methods(gui) binds these other_settings functions onto the
#: desktop owner as instance methods; the legacy owner gets the frozen version for
#: names in an ExternalSpec.bind and a recorder for every other name.
OTHER_SETTINGS_SETUP_FUNCTION = "setup_other_settings_methods"


def other_settings_bound_methods(sha: str) -> list:
    src = ModuleSource("other_settings.py", git_show_text(sha, "src/other_settings.py"))
    fn = src.functions.get(OTHER_SETTINGS_SETUP_FUNCTION)
    if fn is None:
        raise SystemExit("freeze_legacy: other_settings.setup_other_settings_methods not found")
    names = None
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "methods_to_bind" for t in node.targets
        ):
            names = ast.literal_eval(node.value)
    if names is None:
        raise SystemExit("freeze_legacy: methods_to_bind list not found in setup_other_settings_methods")
    module_names = set(src.functions) | set(src.imports) | set(src.assigns)
    # setup binds only names that exist on the module and are callable
    return [n for n in names if n in module_names and n not in src.assigns]

# ---------------------------------------------------------------------------
# git / source helpers
# ---------------------------------------------------------------------------


def _git(*args: str) -> bytes:
    return subprocess.run(
        ["git", *args], cwd=str(REPO_ROOT), check=True, capture_output=True
    ).stdout


def resolve_sha(rev: str = "HEAD") -> str:
    return _git("rev-parse", "--verify", f"{rev}^{{commit}}").decode().strip()


def git_show_text(sha: str, relpath: str) -> str:
    """Return ``git show sha:relpath`` decoded with utf-8-sig and LF line endings."""
    data = _git("show", f"{sha}:{relpath}")
    text = data.decode("utf-8-sig")
    return text.replace("\r\n", "\n").replace("\r", "\n")


class ModuleSource:
    """Top-level binding index of one module's source."""

    def __init__(self, relpath: str, text: str):
        self.relpath = relpath
        self.text = text
        self.lines = text.split("\n")
        self.tree = ast.parse(text, filename=relpath)
        self.functions: dict[str, ast.AST] = {}
        self.classes: dict[str, ast.ClassDef] = {}
        self.assigns: dict[str, list[ast.AST]] = {}
        self.imports: dict[str, list[tuple[ast.AST, ast.alias]]] = {}
        self.nested_bindings: set[str] = set()
        self._index(self.tree.body, top=True)

    def _bind_assign_targets(self, target, out: set):
        if isinstance(target, ast.Name):
            out.add(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                self._bind_assign_targets(elt, out)
        elif isinstance(target, ast.Starred):
            self._bind_assign_targets(target.value, out)

    def _index(self, body, top: bool):
        for stmt in body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if top:
                    self.functions[stmt.name] = stmt
                else:
                    self.nested_bindings.add(stmt.name)
            elif isinstance(stmt, ast.ClassDef):
                if top:
                    self.classes[stmt.name] = stmt
                else:
                    self.nested_bindings.add(stmt.name)
            elif isinstance(stmt, (ast.Import, ast.ImportFrom)):
                for alias in stmt.names:
                    bound = alias.asname or alias.name.split(".")[0]
                    if alias.name == "*":
                        continue
                    self.imports.setdefault(bound, []).append((stmt, alias))
            elif isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                names: set = set()
                targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
                for t in targets:
                    self._bind_assign_targets(t, names)
                for n in names:
                    if top and isinstance(stmt, ast.Assign):
                        self.assigns.setdefault(n, []).append(stmt)
                    else:
                        self.nested_bindings.add(n)
            elif isinstance(stmt, (ast.If, ast.Try, ast.With, ast.For, ast.While)):
                if isinstance(stmt, ast.If) and ast.unparse(stmt.test).replace('"', "'") == "__name__ == '__main__'":
                    continue  # script-only block: never runs on import
                for attr in ("body", "orelse", "finalbody"):
                    self._index(getattr(stmt, attr, []) or [], top=False)
                for handler in getattr(stmt, "handlers", []) or []:
                    if handler.name:
                        self.nested_bindings.add(handler.name)
                    self._index(handler.body, top=False)
                if isinstance(stmt, ast.For):
                    names = set()
                    self._bind_assign_targets(stmt.target, names)
                    self.nested_bindings |= names

    def node_start(self, node) -> int:
        decos = getattr(node, "decorator_list", None) or []
        return min([node.lineno] + [d.lineno for d in decos])

    def segment(self, node) -> str:
        return "\n".join(self.lines[self.node_start(node) - 1:node.end_lineno])

    def line_span(self, node) -> list:
        return [self.node_start(node), node.end_lineno]


def class_methods(cls: ast.ClassDef) -> dict:
    out = {}
    for stmt in cls.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out[stmt.name] = stmt  # last definition wins, as in Python
    return out


def class_attr_statements(cls: ast.ClassDef) -> dict:
    out = {}
    for stmt in cls.body:
        if isinstance(stmt, ast.Assign):
            for t in stmt.targets:
                if isinstance(t, ast.Name):
                    out[t.id] = stmt
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.value is not None:
            out[stmt.target.id] = stmt
    return out


def _in_module_lineage(msrc: "ModuleSource", class_name: str) -> list:
    """[(name, ClassDef)] of *class_name* and its base classes defined in the same module
    (transitively, subclass first): the methods an owner inherits through that mixin."""
    out, queue, seen = [], [class_name], set()
    while queue:
        name = queue.pop(0)
        if name in seen or name not in msrc.classes:
            continue
        seen.add(name)
        node = msrc.classes[name]
        out.append((name, node))
        queue.extend(b.id for b in node.bases if isinstance(b, ast.Name))
    return out


def self_references(node, class_name: str) -> set:
    """Names referenced as ``self.X``, ``cls.X``, ``<ClassName>.X`` or getattr/hasattr(self, 'X')."""
    refs = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Attribute) and isinstance(sub.value, ast.Name):
            if sub.value.id in ("self", "cls", class_name):
                refs.add(sub.attr)
        elif isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name):
            if sub.func.id in ("getattr", "hasattr", "setattr", "delattr") and len(sub.args) >= 2:
                a0, a1 = sub.args[0], sub.args[1]
                if isinstance(a0, ast.Name) and a0.id == "self" and isinstance(a1, ast.Constant) and isinstance(a1.value, str):
                    refs.add(a1.value)
    return refs


def class_attribute_uses(node, class_name: str) -> set:
    out = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Attribute) and isinstance(sub.value, ast.Name) and sub.value.id == class_name:
            out.add(sub.attr)
    return out


def global_names_of(source: str) -> set:
    """Global (module-level) names referenced by *source* (precise, via symtable)."""
    table = symtable.symtable(source, "<frozen>", "exec")
    found = set()

    def walk(t, is_module):
        for sym in t.get_symbols():
            if not sym.is_referenced():
                continue
            if is_module:
                if not sym.is_assigned() and not sym.is_imported():
                    found.add(sym.get_name())
            elif sym.is_global() and not sym.is_declared_global():
                found.add(sym.get_name())
            elif sym.is_declared_global():
                found.add(sym.get_name())
        for child in t.get_children():
            walk(child, False)

    walk(table, True)
    return found


def _wrap_statements_as_function(stmts_src: str) -> str:
    return "def __frozen_block__(self):\n" + stmts_src + "\n"


def _is_literal_assign(stmt) -> bool:
    try:
        ast.literal_eval(stmt.value)
        return True
    except Exception:
        return False


def _uses_qt(node) -> bool:
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and sub.id[:1] == "Q" and sub.id[1:2].isupper():
            return True
    return False


# ---------------------------------------------------------------------------
# __init__ blocks
# ---------------------------------------------------------------------------


@dataclass
class Block:
    name: str
    doc: str
    stmts: list
    text: str
    spans: list


def _block_from_contiguous(src: ModuleSource, name: str, doc: str, stmts: list) -> Block:
    first, last = stmts[0], stmts[-1]
    text = "\n".join(src.lines[first.lineno - 1:last.end_lineno])
    return Block(name, doc, stmts, text, [[first.lineno, last.end_lineno]])


def _block_from_statements(src: ModuleSource, name: str, doc: str, stmts: list) -> Block:
    parts, spans = [], []
    for stmt in stmts:
        parts.append("\n".join(src.lines[stmt.lineno - 1:stmt.end_lineno]))
        spans.append([stmt.lineno, stmt.end_lineno])
    return Block(name, doc, stmts, "\n".join(parts), spans)


def _check_indent(stmts, expected=8):
    for stmt in stmts:
        if stmt.col_offset != expected:
            raise SystemExit(
                f"freeze_legacy: statement at line {stmt.lineno} has indentation "
                f"{stmt.col_offset}, expected {expected}; update the statement spec"
            )


def build_init_blocks(src: ModuleSource, init: ast.FunctionDef) -> list:
    body = list(init.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant):
        body = body[1:]

    def find(pred, start=0, what=""):
        for i in range(start, len(body)):
            if pred(body[i]):
                return i
        raise SystemExit(f"freeze_legacy: __init__ anchor not found: {what}")

    # pre-U1: ``with open(CONFIG_FILE ...)``; U1+: ``self.config = load_config(CONFIG_FILE)``
    config_load = find(
        lambda s: isinstance(s, ast.Try)
        and ("open(CONFIG_FILE" in ast.unparse(s) or "load_config(CONFIG_FILE" in ast.unparse(s)),
        what="config load try (open(CONFIG_FILE ...) / load_config(CONFIG_FILE))",
    )
    update_mgr = find(
        lambda s: isinstance(s, ast.Try) and "UpdateManager" in ast.unparse(s),
        start=config_load + 1,
        what="UpdateManager try",
    )
    # pre-U2: ``self.default_prompts = {...}``; U2+: ``self._init_default_prompt_profiles()``
    default_prompts = find(
        lambda s: (
            isinstance(s, ast.Assign)
            and any(ast.unparse(t) == "self.default_prompts" for t in s.targets)
        ) or (isinstance(s, ast.Expr) and ast.unparse(s) == "self._init_default_prompt_profiles()"),
        start=update_mgr + 1,
        what="self.default_prompts = {...} / self._init_default_prompt_profiles()",
    )
    init_vars = find(
        lambda s: isinstance(s, ast.Expr) and ast.unparse(s) == "self._init_variables()",
        start=default_prompts + 1,
        what="self._init_variables()",
    )
    # pre-U2: the watchdog-dir try inline; U2+: ``self._init_watchdog_dir()``
    watchdog = find(
        lambda s: (isinstance(s, ast.Try) and "glossarion_watchdog" in ast.unparse(s))
        or (isinstance(s, ast.Expr) and ast.unparse(s) == "self._init_watchdog_dir()"),
        start=init_vars + 1,
        what="watchdog dir try / self._init_watchdog_dir()",
    )
    startup_env = find(
        lambda s: isinstance(s, ast.Expr) and ast.unparse(s) == "self.initialize_environment_variables()",
        start=watchdog + 1,
        what="self.initialize_environment_variables()",
    )
    post_import = find(
        lambda s: isinstance(s, ast.ImportFrom) and s.module == "metadata_batch_translator",
        start=startup_env + 1,
        what="from metadata_batch_translator import MetadataBatchTranslatorUI (post startup)",
    )
    post_assign = find(
        lambda s: isinstance(s, ast.Assign)
        and any(ast.unparse(t) == "self.metadata_batch_ui" for t in s.targets),
        start=post_import + 1,
        what="self.metadata_batch_ui = MetadataBatchTranslatorUI(self)",
    )
    # pre-U2: the auto-encryption try inline; U2+: ``self._auto_encrypt_api_keys()``
    auto_encrypt = find(
        lambda s: (isinstance(s, ast.Try) and "needs_encryption" in ast.unparse(s))
        or (isinstance(s, ast.Expr) and ast.unparse(s) == "self._auto_encrypt_api_keys()"),
        start=post_assign + 1,
        what="auto-encryption save_config try / self._auto_encrypt_api_keys()",
    )

    # Pre-config: plain self-attribute assignments made before the config load
    # (no Qt calls), followed by the config-load try itself.
    pre = []
    for stmt in body[:config_load]:
        if not isinstance(stmt, ast.Assign):
            continue
        if not all(
            isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == "self"
            for t in stmt.targets
        ):
            continue
        if _uses_qt(stmt.value):
            continue
        pre.append(stmt)
    pre.append(body[config_load])
    init_block = body[config_load + 1:update_mgr]
    prompts_block = body[update_mgr + 1:default_prompts + 1]
    post = [body[post_import], body[post_assign], body[auto_encrypt]]
    for stmts in (pre, init_block, prompts_block, [body[watchdog]], post):
        _check_indent(stmts)
    last_init = ast.unparse(init_block[-1])
    moved_init = [ast.unparse(s) for s in init_block] == ["self._init_config_state()"]
    if "use_markdown2_converter_var" not in last_init and not moved_init:
        raise SystemExit(
            "freeze_legacy: the __init__ config block no longer ends at use_markdown2_converter_var "
            "(and is not the U2 self._init_config_state() call); re-check the anchors"
        )
    return [
        _block_from_statements(
            src, "legacy_pre_config_block",
            "__init__: plain attribute defaults before the config load + the config load/decrypt/sanitize try.",
            pre,
        ),
        _block_from_contiguous(
            src, "legacy_init_block",
            "__init__ config block (config-backed attributes and startup env exports).",
            init_block,
        ),
        _block_from_contiguous(
            src, "legacy_default_prompts_block",
            "__init__: second MetadataBatchTranslatorUI init + default_* prompt attributes + default_prompts.",
            prompts_block,
        ),
        _block_from_contiguous(
            src, "legacy_watchdog_block",
            "__init__: shared GLOSSARION_WATCHDOG_DIR export (after _setup_gui).",
            [body[watchdog]],
        ),
        _block_from_statements(
            src, "legacy_post_startup_block",
            "__init__ after initialize_environment_variables: third MetadataBatchTranslatorUI init and "
            "the auto-encryption save_config (runs when config holds a plain api_key / replicate_api_key; "
            "GUI log handlers are not replayed).",
            post,
        ),
    ]


def build_statement_block(src: ModuleSource, methods: dict, name: str, doc: str, spec) -> Block:
    picked = []
    for method_name, prefix in spec:
        method = methods.get(method_name)
        if method is None:
            raise SystemExit(f"freeze_legacy: method {method_name} not found for {prefix!r}")
        matches = [
            stmt for stmt in method.body
            if ast.unparse(stmt).startswith(prefix)
        ]
        if len(matches) != 1:
            raise SystemExit(
                f"freeze_legacy: expected exactly one top-level statement starting with {prefix!r} "
                f"in {method_name}, found {len(matches)}"
            )
        picked.append(matches[0])
    _check_indent(picked)
    return _block_from_statements(src, name, doc, picked)


def build_layout_block(src: ModuleSource, methods: dict, name: str, doc: str, layouts) -> tuple:
    """First layout whose statement spec matches completely -> (layout name, Block)."""
    errors = []
    for layout, spec in layouts:
        try:
            return layout, build_statement_block(src, methods, name, doc, spec)
        except SystemExit as exc:
            errors.append(f"{layout}: {exc}")
    raise SystemExit(f"freeze_legacy: no statement layout matched for {name}: " + " | ".join(errors))


# ---------------------------------------------------------------------------
# Module-level resolution (imports / constants / functions / shims)
# ---------------------------------------------------------------------------


@dataclass
class Resolution:
    imports: dict = field(default_factory=dict)      # name -> import statement text
    constants: dict = field(default_factory=dict)    # name -> stmt node
    functions: dict = field(default_factory=dict)    # name -> node
    shims: dict = field(default_factory=dict)        # class -> {attr: node}
    runtime: dict = field(default_factory=dict)      # name -> reason


def _import_text(stmt, alias) -> str:
    if isinstance(stmt, ast.Import):
        if alias.asname:
            return f"import {alias.name} as {alias.asname}"
        return f"import {alias.name}"
    module = "." * (stmt.level or 0) + (stmt.module or "")
    if alias.asname:
        return f"from {module} import {alias.name} as {alias.asname}"
    return f"from {module} import {alias.name}"


def resolve_globals(src: ModuleSource, seed_sources: list, *, own_class: str | None,
                    shim_uses: dict | None = None) -> Resolution:
    """Resolve every global name used by *seed_sources* against *src* top-level bindings."""
    res = Resolution()
    builtin_names = set(dir(builtins)) | {"__file__", "__name__", "__builtins__"}
    pending = set()
    for text in seed_sources:
        pending |= global_names_of(text)
    seen = set()
    shim_uses = dict(shim_uses or {})
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        if name in HARNESS_PROVIDED_GLOBALS:
            # sandbox paths: never import/compute them (config_backup imports the
            # live translator_gui.CONFIG_FILE, i.e. the developer's real config)
            res.runtime[name] = "harness-provided sandbox path"
            continue
        if name in builtin_names and name not in src.functions and name not in src.assigns and name not in src.imports:
            continue
        if own_class and name == own_class:
            continue  # aliased to the frozen class at the end of the file
        if name in src.functions:
            node = src.functions[name]
            res.functions[name] = node
            pending |= global_names_of(src.segment(node))
            continue
        if name in src.classes:
            if shim_uses and name in shim_uses:
                cls = src.classes[name]
                attrs = class_attr_statements(cls)
                meths = class_methods(cls)
                shim = {}
                for attr in sorted(shim_uses[name]):
                    if attr in attrs:
                        shim[attr] = attrs[attr]
                        pending |= global_names_of(ast.unparse(attrs[attr]))
                    elif attr in meths:
                        shim[attr] = meths[attr]
                        pending |= global_names_of(_dedent_method(src, meths[attr]))
                    else:
                        res.runtime[name] = f"class attribute {attr} not found in class body"
                        shim = None
                        break
                if shim is not None:
                    res.shims[name] = shim
                    continue
            res.runtime[name] = "module-level class used beyond constant attributes"
            continue
        assigns = src.assigns.get(name, [])
        if (
            len(assigns) == 1
            and name not in src.imports
            and name not in src.nested_bindings
            and name not in src.functions
            and _is_literal_assign(assigns[0])
        ):
            res.constants[name] = assigns[0]
            continue
        if name in src.imports and not assigns and name not in src.functions:
            stmt, alias = src.imports[name][0]
            res.imports[name] = _import_text(stmt, alias)
            continue
        if name in builtin_names:
            continue
        if assigns or name in src.nested_bindings or name in src.imports:
            res.runtime[name] = "computed or conditionally bound module global"
        else:
            res.runtime[name] = "not bound at module level"
    return res


def _dedent_method(src: ModuleSource, node) -> str:
    text = src.segment(node)
    indent = node.col_offset
    lines = []
    for line in text.split("\n"):
        lines.append(line[indent:] if line[:indent].strip() == "" else line)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Emission
# ---------------------------------------------------------------------------

_HEADER = '''# GENERATED by tests/parity/freeze_legacy.py - do not edit by hand.
# Frozen from `git show {sha}:src/{file}` (freezer v{version}).
# Load with tests/parity/freeze_legacy.load_legacy(); executing this file
# directly is not supported (globals are seeded by the loader).
'''


def _emit_imports(res: Resolution) -> list:
    out = ["# ---- frozen imports (each guarded; failures fall back to the live module) ----",
           "_FROZEN_IMPORT_ERRORS = {}"]
    for name in sorted(res.imports):
        out.append("try:")
        out.append(f"    {res.imports[name]}")
        out.append("except Exception as _frozen_import_exc:  # pragma: no cover")
        out.append(f"    _FROZEN_IMPORT_ERRORS[{name!r}] = repr(_frozen_import_exc)")
    return out


def _emit_constants(src: ModuleSource, res: Resolution) -> list:
    out = ["", "# ---- frozen module constants ----"]
    for stmt in sorted({id(s): s for s in res.constants.values()}.values(), key=lambda s: s.lineno):
        out.append(f"# {src.relpath}:{stmt.lineno}-{stmt.end_lineno}")
        out.append(src.segment(stmt))
    return out


def _emit_functions(src: ModuleSource, res: Resolution) -> list:
    out = ["", "# ---- frozen module functions ----"]
    for node in sorted(res.functions.values(), key=lambda n: n.lineno):
        start, end = src.line_span(node)
        out.append("")
        out.append(f"# {src.relpath}:{start}-{end}")
        out.append(src.segment(node))
    return out


def _emit_shims(src: ModuleSource, res: Resolution) -> list:
    out = []
    for cls_name in sorted(res.shims):
        shim = res.shims[cls_name]
        out.append("")
        out.append(f"class {cls_name}:  # shim: only the frozen attributes of {src.relpath}:{cls_name}")
        for attr, node in sorted(shim.items(), key=lambda kv: kv[1].lineno):
            start, end = src.line_span(node)
            out.append(f"    # {src.relpath}:{start}-{end}")
            out.append(src.segment(node))
    return out


def _emit_class(src: ModuleSource, class_name: str, attr_nodes: list, method_nodes: list,
                blocks: list, doc: str) -> list:
    out = ["", "", f"class {class_name}:", f"    {doc!r}"]
    for node in sorted(attr_nodes, key=lambda n: n.lineno):
        start, end = src.line_span(node)
        out.append(f"    # {src.relpath}:{start}-{end}")
        out.append(src.segment(node))
    for node in sorted(method_nodes, key=lambda n: src.node_start(n)):
        start, end = src.line_span(node)
        out.append("")
        out.append(f"    # {src.relpath}:{start}-{end}")
        out.append(src.segment(node))
    for block in blocks:
        out.append("")
        spans = ", ".join(f"{a}-{b}" for a, b in block.spans)
        out.append(f"    # synthetic block from {src.relpath} lines {spans}")
        out.append(f"    def {block.name}(self):")
        out.append(f"        {block.doc!r}")
        out.append(block.text)
    return out


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Audit helpers recorded in the manifest (for U2 HeadlessOwner work)
# ---------------------------------------------------------------------------


def _sync_call_closure(methods: dict, start: str) -> list:
    seen, queue = [], [start]
    while queue:
        name = queue.pop(0)
        if name in seen or name not in methods:
            continue
        seen.append(name)

        def visit(node, fn=methods[name]):
            if isinstance(node, ast.Lambda):
                return
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node is not fn:
                return
            if isinstance(node, ast.Call):
                f = node.func
                if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id == "self":
                    if f.attr in methods and f.attr not in seen:
                        queue.append(f.attr)
            for child in ast.iter_child_nodes(node):
                visit(child)

        visit(methods[name])
    return seen


def gui_backed_audit(methods: dict) -> tuple:
    """All ``self.*_var =`` / ``self.config[...] =`` writes and widget creations reachable from _setup_gui.

    Nested handler functions that a section builder calls directly at startup
    (like ``_on_auto_glossary_shortcut_changed``) are listed in *writes* as
    ``[method, line, 'nested-call:<name>', <call>]`` so U2 can review them.
    """
    reached = _sync_call_closure(methods, "_setup_gui")
    writes, widgets = [], set()
    for name in reached:
        fn = methods[name]
        def _writes_self_state(fdef) -> bool:
            for sub in ast.walk(fdef):
                if isinstance(sub, ast.Assign):
                    for t in sub.targets:
                        text = ast.unparse(t)
                        if text.startswith("self.") and not text.startswith("self._"):
                            return True
            return False

        nested = {
            n.name for n in ast.walk(fn)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n is not fn and _writes_self_state(n)
        }

        def visit(node):
            if isinstance(node, ast.Lambda):
                return
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node is not fn:
                return
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in nested
            ):
                writes.append([name, node.lineno, f"nested-call:{node.func.id}", ast.unparse(node)[:100]])
            if isinstance(node, ast.Assign):
                value = ast.unparse(node.value)
                for t in node.targets:
                    for x in (t.elts if isinstance(t, ast.Tuple) else [t]):
                        if isinstance(x, ast.Attribute) and isinstance(x.value, ast.Name) and x.value.id == "self":
                            is_widget = (
                                isinstance(node.value, ast.Call)
                                and (
                                    (isinstance(node.value.func, ast.Name) and node.value.func.id[:1].isupper())
                                    or "self._create_styled_checkbox" in value
                                )
                            )
                            if is_widget:
                                widgets.add(x.attr)
                            elif x.attr.endswith("_var"):
                                writes.append([name, node.lineno, f"self.{x.attr}", value[:100]])
                        elif isinstance(x, ast.Subscript) and ast.unparse(x.value) == "self.config":
                            writes.append([name, node.lineno, ast.unparse(x), value[:100]])
            for child in ast.iter_child_nodes(node):
                visit(child)

        visit(fn)
    return reached, writes, sorted(widgets)


# ---------------------------------------------------------------------------
# Freeze
# ---------------------------------------------------------------------------


def legacy_paths(sha: str) -> tuple:
    short = sha[:12]
    return LEGACY_DIR / f"legacy_{short}.py", short


def shared_mixin_sources(sha: str) -> dict:
    """{module: (class_name, ModuleSource)} for the shared mixin modules present at *sha*.

    Helper modules (``SHARED_HELPER_MODULES``) are listed with ``class_name None``.
    """
    out = {}
    specs = list(SHARED_MIXIN_MODULES) + [(module, None) for module in SHARED_HELPER_MODULES]
    for module, class_name in specs:
        try:
            text = git_show_text(sha, f"src/{module}.py")
        except (subprocess.CalledProcessError, OSError):
            continue  # not extracted yet at this SHA
        src = ModuleSource(f"{module}.py", text)
        if class_name is not None and class_name not in src.classes:
            raise SystemExit(f"freeze_legacy: {module}.py has no class {class_name}")
        out[module] = (class_name, src)
    return out


def freeze(rev: str = "HEAD", out_dir: Path = LEGACY_DIR, *, entry_methods=None) -> Path:
    """Freeze TranslatorGUI @ *rev*; *entry_methods* overrides TG_ENTRY_METHODS (closure seeds)."""
    sha = resolve_sha(rev)
    short = sha[:12]
    out_dir.mkdir(parents=True, exist_ok=True)
    tg_text = git_show_text(sha, f"src/{TG_FILE}")
    tg = ModuleSource(TG_FILE, tg_text)
    cls = tg.classes[TG_CLASS]
    methods = class_methods(cls)
    attrs = class_attr_statements(cls)
    dup = {}
    for stmt in cls.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            dup.setdefault(stmt.name, []).append(stmt.lineno)
    duplicates = {k: v for k, v in dup.items() if len(v) > 1}

    blocks = build_init_blocks(tg, methods["__init__"])
    state_layout, state_block = build_layout_block(
        tg, methods, "legacy_gui_state_block",
        "_setup_gui section builders: GUI-backed state assignments (widgets are installed after this).",
        GUI_STATE_LAYOUTS,
    )
    handler_layout, handler_block = build_layout_block(
        tg, methods, "legacy_gui_handlers_block",
        "_setup_gui startup handlers that run once widgets exist (desktop order).",
        GUI_HANDLER_LAYOUTS,
    )
    blocks += [state_block, handler_block]

    # ---- shared GUI-free mixins (U2+) ----
    mixins = shared_mixin_sources(sha)
    mixin_methods: dict[str, list] = {}
    mixin_refs: dict[str, set] = {}
    for module, (class_name, msrc) in mixins.items():
        if class_name is None:
            # helper module: only its *Mixin classes are owner code (job_runner.JobHooksMixin);
            # other classes (ProgressWatcher, StopClickTracker) use ``self`` for themselves
            owner_classes = [(name, cls) for name, cls in sorted(msrc.classes.items()) if name.endswith("Mixin")]
            mixin_methods[module] = sorted({n for _name, cls in owner_classes
                                            for n in list(class_methods(cls)) + list(class_attr_statements(cls))})
            refs = set()
            for name, cls in owner_classes:
                refs |= self_references(cls, name)
            mixin_refs[module] = refs
            continue
        mcls = msrc.classes[class_name]
        # the named mixin plus its base classes defined in the same module (U3:
        # TranslationPipelineMixin -> GlossaryPipelineMixin -> PipelineHooksMixin)
        owner_classes = _in_module_lineage(msrc, class_name)
        mixin_methods[module] = (sorted({n for _name, c in owner_classes for n in class_methods(c)})
                                 + sorted({n for _name, c in owner_classes for n in class_attr_statements(c)}))
        refs = set()
        for name, c in owner_classes:
            refs |= self_references(c, name)
        for fn in msrc.functions.values():  # module helpers taking the owner (owner/self param)
            refs |= self_references(fn, class_name)
        mixin_refs[module] = refs
    mixin_provided = {name for names in mixin_methods.values() for name in names}

    # ---- method closure ----
    frozen: dict[str, str] = {}
    recorded: set = set()
    missing: dict[str, str] = {}
    needed_attrs: set = set()
    entries = tuple(TG_ENTRY_METHODS if entry_methods is None else entry_methods)
    queue = [(m, "entry") for m in entries]
    for block in blocks:
        node = ast.parse(_wrap_statements_as_function(block.text))
        for ref in sorted(self_references(node, TG_CLASS)):
            queue.append((ref, f"block:{block.name}"))
    for module, refs in mixin_refs.items():
        # TranslatorGUI methods the shared mixins call (hook overrides, GUI helpers)
        for ref in sorted(refs):
            queue.append((ref, f"mixin:{module}"))
    while queue:
        name, why = queue.pop(0)
        if name in frozen or name in recorded:
            continue
        if name in TG_RECORDED_METHODS:
            recorded.add(name)
            continue
        if name not in methods:
            if name in attrs:
                needed_attrs.add(name)
            continue
        frozen[name] = why
        for ref in sorted(self_references(methods[name], TG_CLASS)):
            if ref in methods and ref not in frozen:
                queue.append((ref, f"closure:{name}"))
            elif ref in attrs:
                needed_attrs.add(ref)
    moved_to_mixins = sorted(
        name for name in set(entries) | set(TG_FREEZE_ONLY_METHODS)
        if name not in methods and name in mixin_provided
    )
    for name in entries:
        if name not in methods and name not in mixin_provided:
            missing[name] = "entry method not found in TranslatorGUI"
    unresolved_self_methods: dict[str, list] = {}
    for name in TG_FREEZE_ONLY_METHODS:
        if name not in methods:
            if name not in mixin_provided:
                missing[name] = "freeze-only method not found"
            continue
        if name not in frozen:
            frozen[name] = "freeze_only"
        refs = self_references(methods[name], TG_CLASS)
        for ref in refs:
            if ref in attrs:
                needed_attrs.add(ref)
            elif ref in methods and ref not in frozen and ref not in recorded:
                unresolved_self_methods.setdefault(ref, []).append(name)

    exercised = [m for m, why in frozen.items() if why != "freeze_only"]
    method_nodes = [methods[m] for m in frozen]
    attr_nodes = [attrs[a] for a in sorted(needed_attrs)]

    # Private (name-mangled) attributes would be mangled with the frozen class name.
    for node in method_nodes + [ast.parse(_wrap_statements_as_function(b.text)) for b in blocks]:
        for sub in ast.walk(node):
            if isinstance(sub, ast.Attribute) and sub.attr.startswith("__") and not sub.attr.endswith("__"):
                raise SystemExit(
                    f"freeze_legacy: name-mangled attribute {sub.attr!r} near line {getattr(sub, 'lineno', '?')} "
                    "cannot be frozen into another class verbatim"
                )

    # class attributes may reference other class attributes
    changed = True
    while changed:
        changed = False
        for node in list(attr_nodes):
            for ref in self_references(node, TG_CLASS) | global_names_of(ast.unparse(node)):
                if ref in attrs and attrs[ref] not in attr_nodes:
                    attr_nodes.append(attrs[ref])
                    needed_attrs.add(ref)
                    changed = True

    # ---- module globals ----
    seed = [_dedent_method(tg, n) for n in method_nodes]
    seed += [ast.unparse(n) for n in attr_nodes]
    seed += [_wrap_statements_as_function(b.text) for b in blocks]
    shim_uses: dict[str, set] = {}
    for cname in tg.classes:
        if cname == TG_CLASS:
            continue
        uses = set()
        for n in method_nodes:
            uses |= class_attribute_uses(n, cname)
        for b in blocks:
            uses |= class_attribute_uses(ast.parse(_wrap_statements_as_function(b.text)), cname)
        if uses:
            shim_uses[cname] = uses
    res = resolve_globals(tg, seed, own_class=TG_CLASS, shim_uses=shim_uses)

    reached, gui_writes, startup_widgets = gui_backed_audit(methods)
    exercised_refs = set()
    for m in exercised:
        exercised_refs |= self_references(methods[m], TG_CLASS)
    for b in blocks:
        exercised_refs |= self_references(ast.parse(_wrap_statements_as_function(b.text)), TG_CLASS)
    for refs in mixin_refs.values():
        exercised_refs |= refs
    recorded_methods = sorted(
        r for r in exercised_refs
        if r in methods and r not in frozen
    )
    # settings_map-style string sources (getattr(self, source_attr)) count as references too
    exercised_strings = set()
    for m in exercised:
        for sub in ast.walk(methods[m]):
            if isinstance(sub, ast.Constant) and isinstance(sub.value, str) and sub.value.isidentifier():
                exercised_strings.add(sub.value)
    for module, (class_name, msrc) in mixins.items():
        for sub in ast.walk(msrc.tree):
            if isinstance(sub, ast.Constant) and isinstance(sub.value, str) and sub.value.isidentifier():
                exercised_strings.add(sub.value)
    startup_widget_refs = sorted(set(startup_widgets) & (exercised_refs | exercised_strings))

    # ---- externals ----
    externals_manifest = {}
    external_files = []
    for spec in EXTERNALS:
        path, info = freeze_external(sha, short, spec, out_dir)
        external_files.append(path.name)
        externals_manifest[spec.module] = info
    # ---- shared mixin modules: whole-module verbatim copies at this SHA (U2+) ----
    frozen_mixins = {}
    for module, (class_name, msrc) in mixins.items():
        path = freeze_mixin_module(sha, module, msrc, out_dir)
        frozen_mixins[module] = {
            "class": class_name,
            "source_file": f"src/{module}.py",
            "source_sha256": _sha256(msrc.text),
            "file": path.name,
            "header_lines": len(_MIXIN_HEADER.format(sha=sha, file=f"{module}.py", version=FREEZER_VERSION)
                                .split("\n")) - 1,
        }
    bound = other_settings_bound_methods(sha)
    bound_frozen = {name: spec.module for spec in EXTERNALS for name in spec.bind}
    missing_bound = sorted(set(bound_frozen) - set(bound))
    if missing_bound:
        raise SystemExit(f"freeze_legacy: {missing_bound} are no longer bound by setup_other_settings_methods")
    bound_referenced = sorted(set(bound) & exercised_refs)

    manifest = {
        "freezer_version": FREEZER_VERSION,
        "sha": sha,
        "short_sha": short,
        "source_file": f"src/{TG_FILE}",
        "source_sha256": _sha256(tg_text),
        "class": TG_CLASS,
        "frozen_methods": {m: {"lines": tg.line_span(methods[m]), "reason": frozen[m]} for m in frozen},
        "exercised_methods": sorted(exercised),
        "freeze_only_methods": [m for m in TG_FREEZE_ONLY_METHODS if m in methods],
        "class_attrs": {a: tg.line_span(attrs[a]) for a in sorted(needed_attrs)},
        "synthetic_blocks": {b.name: b.spans for b in blocks},
        "module_functions": {n: tg.line_span(node) for n, node in sorted(res.functions.items())},
        "module_constants": {n: tg.line_span(node) for n, node in sorted(res.constants.items())},
        "imports": dict(sorted(res.imports.items())),
        "shims": {c: sorted(v) for c, v in sorted(res.shims.items())},
        "runtime_globals": dict(sorted(res.runtime.items())),
        "recorded_methods": recorded_methods,
        "unresolved_freeze_only_refs": {k: sorted(v) for k, v in sorted(unresolved_self_methods.items())},
        "missing": missing,
        "layouts": {
            "init_config_block": "u2" if any(
                ast.unparse(s) == "self._init_config_state()" for s in blocks[1].stmts
            ) else "legacy",
            "gui_state": state_layout,
            "gui_handlers": handler_layout,
        },
        "shared_mixins": [[module, mixins[module][0]] for module, _cls in SHARED_MIXIN_MODULES if module in mixins],
        "shared_helpers": [module for module in SHARED_HELPER_MODULES if module in mixins],
        "frozen_mixins": frozen_mixins,
        "mixin_methods": mixin_methods,
        "moved_to_mixins": moved_to_mixins,
        "duplicate_method_definitions": duplicates,
        "startup_widget_attrs": startup_widget_refs,
        "gui_backed_audit": {
            "methods_reached_from_setup_gui": len(reached),
            "writes": gui_writes,
            "replayed": [list(x) for x in dict(GUI_STATE_LAYOUTS)[state_layout] + dict(GUI_HANDLER_LAYOUTS)[handler_layout]],
        },
        "externals": externals_manifest,
        "external_files": external_files,
        "module_patches": [
            [spec.module, attr] for spec in EXTERNALS for attr in spec.patch
        ],
        "other_settings_bound_methods": bound,
        "bound_frozen": bound_frozen,
        "bound_referenced_by_exercised_code": bound_referenced,
    }

    body = [_HEADER.format(sha=sha, file=TG_FILE, version=FREEZER_VERSION)]
    body.append("FREEZE_MANIFEST = " + pprint.pformat(manifest, width=110, sort_dicts=False))
    body.append("")
    body += _emit_imports(res)
    body += _emit_constants(tg, res)
    body += _emit_functions(tg, res)
    body += _emit_shims(tg, res)
    body += _emit_class(
        tg, "LegacyMethods", attr_nodes, method_nodes, blocks,
        f"Frozen TranslatorGUI methods @ {short} (mixed into fakes.FakeState).",
    )
    body.append("")
    body.append("")
    body.append("# Frozen code calls TranslatorGUI.<method>(self, ...) explicitly; bind it to the frozen class.")
    body.append(f"{TG_CLASS} = LegacyMethods")
    body.append("")
    out_path = out_dir / f"legacy_{short}.py"
    text = "\n".join(body)
    compile(text, str(out_path), "exec")
    _verify_globals(text, res, extra_defined={TG_CLASS, "LegacyMethods"}, where=out_path.name)
    out_path.write_text(text, encoding="utf-8", newline="\n")
    (out_dir / "LATEST.txt").write_text(sha + "\n", encoding="utf-8")
    return out_path


def _verify_globals(text: str, res: Resolution, *, extra_defined: set, where: str) -> None:
    tree = ast.parse(text)
    defined = set(extra_defined)
    idx = ModuleSource(where, text)
    defined |= set(idx.functions) | set(idx.classes) | set(idx.assigns) | set(idx.imports) | idx.nested_bindings
    used = global_names_of(text)
    builtin_names = set(dir(builtins)) | {"__file__", "__name__", "__builtins__"}
    unresolved = sorted(n for n in used if n not in defined and n not in builtin_names and n not in res.runtime)
    if unresolved:
        raise SystemExit(f"freeze_legacy: {where} references unresolved globals: {unresolved}")
    del tree


_MIXIN_HEADER = '''# GENERATED by tests/parity/freeze_legacy.py - do not edit by hand.
# Frozen copy of `git show {sha}:src/{file}` (freezer v{version}): the whole shared
# mixin module as it was at the frozen commit, so the legacy oracle never runs the live
# module. Load with tests/parity/freeze_legacy.load_legacy(); imports of the other frozen
# mixin modules resolve to their frozen copies. The source follows this header verbatim.
'''


def freeze_mixin_module(sha: str, module: str, msrc: ModuleSource, out_dir: Path) -> Path:
    """Write ``legacy_<sha12>__<module>.py`` = header + the module source at *sha* (verbatim)."""
    out_path = out_dir / f"legacy_{sha[:12]}__{module}.py"
    text = _MIXIN_HEADER.format(sha=sha, file=f"{module}.py", version=FREEZER_VERSION) + msrc.text
    compile(text, str(out_path), "exec")
    out_path.write_text(text, encoding="utf-8", newline="\n")
    return out_path


def frozen_mixin_source(path: Path, info: dict) -> str:
    """The verbatim module source stored in a frozen mixin file (header stripped)."""
    lines = path.read_text(encoding="utf-8").split("\n")
    return "\n".join(lines[int(info["header_lines"]):])


def freeze_external(sha: str, short: str, spec: ExternalSpec, out_dir: Path) -> tuple:
    relpath = f"{spec.module}.py"
    text = git_show_text(sha, f"src/{relpath}")
    src = ModuleSource(relpath, text)
    seeds, fn_nodes, class_parts = [], {}, {}
    reexports = {}
    for name in spec.functions:
        node = src.functions.get(name)
        if node is None and name in src.imports and name not in src.assigns:
            # Moved to a shared GUI-free module and re-exported here (U1+): the
            # frozen namespace imports it like the original module does.
            stmt, alias = src.imports[name][0]
            reexports[name] = _import_text(stmt, alias)
            continue
        if node is None:
            raise SystemExit(f"freeze_legacy: {relpath}:{name} not found")
        fn_nodes[name] = node
        seeds.append(src.segment(node))
    for cname, wanted in spec.classes.items():
        cls = src.classes.get(cname)
        if cls is None:
            raise SystemExit(f"freeze_legacy: {relpath}:{cname} not found")
        meths = class_methods(cls)
        attrs = class_attr_statements(cls)
        picked, queue, attr_picked = {}, list(wanted), {}
        while queue:
            m = queue.pop(0)
            if m in picked:
                continue
            if m not in meths:
                if m in attrs:
                    attr_picked[m] = attrs[m]
                continue
            picked[m] = meths[m]
            for ref in self_references(meths[m], cname):
                if ref in meths and ref not in picked:
                    queue.append(ref)
                elif ref in attrs:
                    attr_picked[ref] = attrs[ref]
        class_parts[cname] = (picked, attr_picked)
        for node in list(picked.values()) + list(attr_picked.values()):
            seeds.append(_dedent_method(src, node))
    res = resolve_globals(src, seeds, own_class=None)
    # functions requested explicitly are emitted even when not referenced by each other
    for name, node in fn_nodes.items():
        res.functions[name] = node
    for name, import_text in reexports.items():
        res.imports[name] = import_text
    body = [_HEADER.format(sha=sha, file=relpath, version=FREEZER_VERSION)]
    info = {
        "source_file": f"src/{relpath}",
        "source_sha256": _sha256(text),
        "functions": {n: src.line_span(node) for n, node in sorted(res.functions.items())},
        "constants": {n: src.line_span(node) for n, node in sorted(res.constants.items())},
        "imports": dict(sorted(res.imports.items())),
        "classes": {
            c: {"methods": {m: src.line_span(n) for m, n in parts[0].items()},
                "attrs": {a: src.line_span(n) for a, n in parts[1].items()}}
            for c, parts in class_parts.items()
        },
        "runtime_globals": dict(sorted(res.runtime.items())),
        "reexports": dict(sorted(reexports.items())),
        "patch": list(spec.patch),
    }
    body.append("FREEZE_MANIFEST = " + pprint.pformat(info, width=110, sort_dicts=False))
    body.append("")
    body += _emit_imports(res)
    body += _emit_constants(src, res)
    body += _emit_functions(src, res)
    for cname, (picked, attr_picked) in class_parts.items():
        body += _emit_class(
            src, cname, list(attr_picked.values()), list(picked.values()), [],
            f"Frozen subset of {relpath}:{cname} @ {short}",
        )
    body.append("")
    out_path = out_dir / f"legacy_{short}__{spec.module}.py"
    out_text = "\n".join(body)
    compile(out_text, str(out_path), "exec")
    _verify_globals(out_text, res, extra_defined=set(spec.classes), where=out_path.name)
    out_path.write_text(out_text, encoding="utf-8", newline="\n")
    return out_path, info


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


@dataclass
class LegacyBundle:
    sha: str
    path: Path
    namespace: dict
    externals: dict  # module -> namespace
    manifest: dict
    mixins: dict = field(default_factory=dict)  # module -> frozen module object (U2+)

    @property
    def methods(self):
        return self.namespace["LegacyMethods"]

    def mixin_classes(self) -> tuple:
        """The FROZEN shared mixin classes, in desktop precedence order (empty before U2)."""
        out = []
        for module, class_name in self.manifest.get("shared_mixins", []) or []:
            if module not in self.mixins:
                raise RuntimeError(
                    f"legacy oracle @ {self.sha[:12]} lists the shared mixin {module} without a frozen "
                    "copy (frozen by freezer v1); re-run tests/parity/freeze_legacy.py --sha "
                    f"{self.sha[:12]} so the oracle never falls back to the live module"
                )
            out.append(getattr(self.mixins[module], class_name))
        return tuple(out)

    def mixin_namespaces(self) -> list:
        """Globals of the frozen mixin modules (for path/backend/Qt patches)."""
        return [vars(module) for module in self.mixins.values()]

    def module_patches(self) -> list:
        """[(live_module_name, attribute, frozen_object)] to patch while legacy code runs."""
        out = []
        for module, attr in self.manifest["module_patches"]:
            out.append((module, attr, self.externals[module][attr]))
        return out

    def external(self, module: str, name: str):
        return self.externals[module][name]

    def bound_methods(self) -> dict:
        """{name: frozen function | None} bound onto the owner by setup_other_settings_methods.

        ``None`` means "bind a recorder" (GUI dialogs, toggles, styling helpers).
        """
        frozen = self.manifest.get("bound_frozen", {})
        return {
            name: (self.externals[frozen[name]][name] if name in frozen else None)
            for name in self.manifest.get("other_settings_bound_methods", [])
        }


#: Runtime globals the capture harness always overrides (sandbox paths).
HARNESS_PROVIDED_GLOBALS = frozenset({"CONFIG_FILE", "_APP_DIR"})


def latest_sha() -> str:
    marker = LEGACY_DIR / "LATEST.txt"
    if not marker.exists():
        raise FileNotFoundError("no frozen legacy oracle; run tests/parity/freeze_legacy.py first")
    return marker.read_text(encoding="utf-8").strip()


class _FrozenBuiltins(dict):
    """``__builtins__`` of frozen code when the oracle has frozen mixin modules.

    Only ``__import__`` is stored: ``import owner_state`` / ``from run_env import X`` in
    frozen code return the frozen copies. Every other builtin is looked up live in
    :mod:`builtins` (``__missing__``), so harness patches such as ``builtins.open`` apply.
    """

    def __missing__(self, key):
        try:
            return getattr(builtins, key)
        except AttributeError:
            raise KeyError(key) from None


def _load_frozen_mixins(short: str, manifest: dict) -> tuple:
    """({module: frozen module object}, builtins for frozen code) from manifest ``frozen_mixins``."""
    frozen = manifest.get("frozen_mixins") or {}
    modules: dict = {}
    if not frozen:
        return modules, builtins
    if str(SRC_DIR) not in sys.path:
        # the frozen modules import their live src siblings (emoticon_patterns, app_paths, ...)
        sys.path.insert(0, str(SRC_DIR))
    real_import = builtins.__import__
    import importlib as real_importlib

    class _FrozenImportlib(types.ModuleType):
        """``importlib`` as frozen code sees it: ``import_module`` of a frozen module name returns the
        frozen copy (U9: settings_schema resolves converters / defaults that way), the rest is live."""

        def __getattr__(self, attr):
            return getattr(real_importlib, attr)

    def import_module(name, package=None):
        if package is None and name in frozen:
            return load(name)
        return real_importlib.import_module(name, package)

    frozen_importlib = _FrozenImportlib("importlib")
    frozen_importlib.import_module = import_module

    def frozen_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level == 0 and name in frozen:
            return load(name)
        if level == 0 and name == "importlib" and all(item == "import_module" for item in (fromlist or ())):
            return frozen_importlib
        return real_import(name, globals, locals, fromlist, level)

    frozen_builtins = _FrozenBuiltins(__import__=frozen_import)

    def load(module):
        if module in modules:
            return modules[module]
        info = frozen[module]
        path = LEGACY_DIR / info["file"]
        mod = types.ModuleType(f"parity_legacy_{short}__{module}")
        mod.__file__ = str(SRC_DIR / f"{module}.py")
        mod.__builtins__ = frozen_builtins
        mod.__frozen_path__ = str(path)
        modules[module] = mod  # registered first: import cycles resolve like sys.modules
        # A module with its own ``from __future__ import annotations`` (U9: settings_schema) has
        # string annotations, and @dataclass then looks the class's module up in sys.modules while
        # the class is created: the frozen copy is listed under its private name for the exec only
        # (the live module name is never touched).
        sys.modules[mod.__name__] = mod
        try:
            # dont_inherit: a whole module is compiled with ITS future flags, not this file's
            # ``from __future__ import annotations`` (string annotations break @dataclass in a
            # module that is not in sys.modules)
            exec(compile(path.read_text(encoding="utf-8"), str(path), "exec", dont_inherit=True), vars(mod))
        except BaseException:
            modules.pop(module, None)
            raise
        finally:
            if sys.modules.get(mod.__name__) is mod:
                del sys.modules[mod.__name__]
        return mod

    for module in frozen:
        load(module)
    return modules, frozen_builtins


def _exec_frozen(path: Path, module_name: str, live_module_name: str, builtins_ns=builtins) -> dict:
    text = path.read_text(encoding="utf-8")
    ns: dict = {
        "__name__": module_name,
        "__file__": str(SRC_DIR / f"{live_module_name}.py"),
        "__builtins__": builtins_ns,
        "__frozen_path__": str(path),
    }
    code = compile(text, str(path), "exec")
    exec(code, ns)
    manifest = ns.get("FREEZE_MANIFEST", {})
    missing = dict(manifest.get("runtime_globals", {}))
    for name in ns.get("_FROZEN_IMPORT_ERRORS", {}):
        missing[name] = "frozen import failed"
    for name in HARNESS_PROVIDED_GLOBALS & set(missing):
        # Path globals: the capture harness always points these at the sandbox,
        # so there is no need to import the live module (whose import has
        # side effects such as log-folder cleanup) just to read them.
        ns[name] = None
        missing.pop(name)
    if missing:
        if str(SRC_DIR) not in sys.path:
            sys.path.insert(0, str(SRC_DIR))
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        import importlib

        live = importlib.import_module(live_module_name)
        unresolved = []
        for name in missing:
            if hasattr(live, name):
                ns[name] = getattr(live, name)
            else:
                ns[name] = None
                unresolved.append(name)
        ns["__runtime_unresolved__"] = unresolved
    return ns


def load_legacy(sha: str | None = None) -> LegacyBundle:
    sha = sha or latest_sha()
    path, short = legacy_paths(sha)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run freeze_legacy.py --sha {sha}")
    manifest = _read_manifest(path)
    # frozen mixin modules first: frozen code importing them must get the frozen copies
    mixins, frozen_builtins = _load_frozen_mixins(short, manifest)
    ns = _exec_frozen(path, f"parity_legacy_{short}", "translator_gui", frozen_builtins)
    manifest = ns["FREEZE_MANIFEST"]
    externals = {}
    for module in manifest["externals"]:
        ext_path = LEGACY_DIR / f"legacy_{short}__{module}.py"
        externals[module] = _exec_frozen(ext_path, f"parity_legacy_{short}__{module}", module, frozen_builtins)
    return LegacyBundle(sha=manifest["sha"], path=path, namespace=ns, externals=externals, manifest=manifest,
                        mixins=mixins)


def _read_manifest(path: Path) -> dict:
    """``FREEZE_MANIFEST`` literal of a frozen file, without executing it."""
    text = path.read_text(encoding="utf-8")
    end = text.find("\n# ---- frozen imports")  # the manifest is emitted right before the imports
    try:
        tree = ast.parse(text[:end] if end > 0 else text, filename=str(path))
    except SyntaxError:
        tree = ast.parse(text, filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "FREEZE_MANIFEST" for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise KeyError(f"FREEZE_MANIFEST not found in {path}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sha", default="HEAD", help="commit to freeze (default HEAD = BASE_SHA)")
    args = parser.parse_args(argv)
    path = freeze(args.sha)
    bundle = load_legacy(resolve_sha(args.sha))
    m = bundle.manifest
    print(f"frozen {m['sha']} -> {path.relative_to(REPO_ROOT)}")
    print(f"  methods: {len(m['frozen_methods'])} ({len(m['exercised_methods'])} exercised, "
          f"{len(m['freeze_only_methods'])} freeze-only)")
    print(f"  synthetic blocks: {', '.join(m['synthetic_blocks'])}")
    print(f"  module functions: {len(m['module_functions'])}, constants: {len(m['module_constants'])}, "
          f"imports: {len(m['imports'])}, shims: {list(m['shims'])}")
    print(f"  runtime globals (live fallback): {sorted(m['runtime_globals'])}")
    print(f"  recorded GUI methods: {m['recorded_methods']}")
    print(f"  externals: {', '.join(m['external_files'])}")
    frozen_mixins = m.get("frozen_mixins") or {}
    if frozen_mixins:
        print(f"  frozen shared mixins: {', '.join(i['file'] for i in frozen_mixins.values())}")
    if m["missing"]:
        print(f"  MISSING: {m['missing']}")
    unresolved = bundle.namespace.get("__runtime_unresolved__", [])
    if unresolved:
        print(f"  runtime globals not found in live module: {unresolved}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
