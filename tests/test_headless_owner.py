"""HeadlessOwner and the U2 shared-core moves (Glossarion mobile rewrite, milestone U2).

TranslatorGUI's settings/state/env code moved verbatim into GUI-free mixins
(``owner_state.ConfigStateMixin``, ``run_env.RunEnvMixin``,
``settings_persistence.SettingsPersistenceMixin``) that TranslatorGUI inherits first,
and ``headless_owner.HeadlessOwner`` builds the same owner from a config dict.

What is checked:
* import hygiene (PySide6 blocked, no translator_gui / dpi_setup) and Python 3.10 syntax;
* MRO: shared mixins first, no GUI mixin / other_settings binding / TranslatorGUI body
  shadows a moved name (only declared hooks are overridden), mixins define no __init__;
* tier G: the working-tree desktop composition (frozen TranslatorGUI body + live
  mixins, driven by the U0 fakes) reproduces the goldens captured at BASE_SHA for all
  12 scenarios, and HeadlessOwner reproduces every env entry + the startup env/config;
* tier R: ``HeadlessOwner(desktop._collect_live_settings())`` equals a desktop restarted
  from that saved config; the first-run desktop differs from both only by the recorded
  desktop first-run/restart quirks (KNOWN_ROUNDTRIP_DIVERGENCES);
* the owner contract, the startup-handler replay order and widget shim sources,
  DirectTextRunOptions vs the original dialog code, side-effect freedom.

Run (repository root)::

    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic \
        tests/test_headless_owner.py
"""

from __future__ import annotations

import ast
import copy
import json
import os
import random
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
TESTS = REPO_ROOT / "tests"
for _p in (str(TESTS), str(SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _headless_env import scrubbed_env as _scrubbed_env  # noqa: E402

#: Parent commit of the U2 extraction (merged U1); the oracle/goldens are frozen here.
BASE_SHA = "96af9adb396e09cc8fee28df777884e6bbb544bd"
WORKTREE_SHA = "f" * 40

NEW_MODULES = ("owner_state", "run_env", "settings_persistence", "headless_owner", "library_core")
TOUCHED_FILES = NEW_MODULES + ("translator_gui", "other_settings", "epub_library")
MIXINS = (
    ("settings_persistence", "SettingsPersistenceMixin"),
    ("run_env", "RunEnvMixin"),
    ("owner_state", "ConfigStateMixin"),
)
GUI_MIXINS = (
    ("QA_Scanner_GUI", "QAScannerMixin"),
    ("Retranslation_GUI", "RetranslationMixin"),
    ("GlossaryManager_GUI", "GlossaryManagerMixin"),
)
#: Hooks: the mixin default is GUI-free, TranslatorGUI overrides with the original code.
HOOK_OVERRIDES = frozenset({
    "_can_read_widgets",
    "_hook_persist_sanitized_config",
    "_hook_save_default_config",
    "_hook_ensure_executor",
    "_hook_metadata_defaults",
    "_hook_context_mode_layout",
})
#: Env entries HeadlessOwner is compared on (the job runners arrive in U3).
HEADLESS_ENTRIES = (
    "translation_env", "glossary_env_mappings", "run_helpers", "multipass_runtime_env",
    "chapter_range_runtime_env", "metadata_only_env", "direct_text_env", "forced_streaming_env",
)
#: OWNER_CONTRACT names HeadlessOwner does not provide yet, with the reason.
#: (U3: auto_load_glossary_for_file arrived with translation_pipeline.GlossaryPipelineMixin.)
HEADLESS_DEFERRED: dict = {}
#: Env keys where a first-run desktop differs from the same desktop restarted from the
#: config its startup save_config wrote (HeadlessOwner from _collect_live_settings()
#: equals the restarted desktop). Desktop behaviour, recorded in DISCREPANCIES.md (U2).
KNOWN_ROUNDTRIP_DIVERGENCES = {
    "SEND_INTERVAL_SECONDS": "first run exports the delay_entry text ('5'); save_config stores safe_float -> '5.0' after a restart",
    "CONNECT_TIMEOUT": "startup env reads connect_timeout_var ('10'); save_config stores safe_float -> '10.0' after a restart",
    "READ_TIMEOUT": "startup env reads read_timeout_var ('180'); save_config stores safe_float -> '180.0' after a restart",
    "IMAGE_CHUNK_OVERLAP_PERCENT": "image_chunk_overlap_var '3' at first run; save_config stores safe_float -> '3.0' after a restart",
    "GLOSSARY_TRANSLATION_PROMPT": "_glossary_env_mappings writes '' for the missing key during the startup save, so the built-in default is lost after a restart",
    "GLOSSARY_FORMAT_INSTRUCTIONS": "same as GLOSSARY_TRANSLATION_PROMPT",
    "EXTRACTION_MODE": "save_config derives extraction_mode from file_filtering_level_var; the first-run startup env still uses the _init_variables extraction_mode_var",
}


def _src_text(module: str) -> str:
    return (SRC / f"{module}.py").read_bytes().decode("utf-8-sig").replace("\r\n", "\n")


def _tree(module: str) -> ast.Module:
    return ast.parse(_src_text(module))


def _class(module: str, name: str) -> ast.ClassDef:
    return next(n for n in _tree(module).body if isinstance(n, ast.ClassDef) and n.name == name)


def _class_names(cls: ast.ClassDef) -> set:
    out = set()
    for node in cls.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.add(node.name)
        elif isinstance(node, ast.Assign):
            out.update(t.id for t in node.targets if isinstance(t, ast.Name))
    return out


def _git_show(relpath: str) -> str:
    try:
        data = subprocess.run(["git", "show", f"{BASE_SHA}:{relpath}"], cwd=str(REPO_ROOT),
                              check=True, capture_output=True).stdout
    except Exception as exc:  # pragma: no cover - shallow clone
        message = f"git show {BASE_SHA}:{relpath} unavailable: {exc}"
        if os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"):
            # CI checks out full history (fetch-depth: 0); skipping there would silently
            # drop every desktop-parity check of the U2 moves
            pytest.fail(message + " (CI must fetch the base commit)")
        pytest.skip(message)
    return data.decode("utf-8-sig").replace("\r\n", "\n")


# ---------------------------------------------------------------------------
# import hygiene / Python 3.10
# ---------------------------------------------------------------------------

_HYGIENE_PROBE = r"""
import sys
sys.modules['PySide6'] = None
sys.path.insert(0, {src!r})
import importlib
importlib.import_module({module!r})
leaked = [n for n in ('translator_gui', 'dpi_setup', 'PySide6.QtCore', 'PySide6.QtWidgets')
          if sys.modules.get(n) is not None]
print('LEAKED=' + ','.join(leaked))
"""


@pytest.mark.parametrize("module", NEW_MODULES)
def test_new_module_imports_without_qt_or_translator_gui(module):
    env = {k: v for k, v in os.environ.items() if not k.startswith("GLOSSARION_")}
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run(
        [sys.executable, "-c", _HYGIENE_PROBE.format(src=str(SRC), module=module)],
        capture_output=True, text=True, encoding="utf-8", env=env, timeout=180,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().splitlines()[-1] == "LEAKED=", proc.stdout


@pytest.mark.parametrize("module", TOUCHED_FILES)
def test_touched_files_parse_as_python_310(module):
    ast.parse(_src_text(module), feature_version=(3, 10))


def test_new_modules_never_import_qt_or_the_gui():
    for module in NEW_MODULES:
        for node in ast.walk(_tree(module)):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            for name in names:
                assert not name.startswith(("PySide6", "translator_gui", "dpi_setup")), (module, name)


# ---------------------------------------------------------------------------
# MRO / no collisions
# ---------------------------------------------------------------------------

def _shared_names() -> dict:
    return {module: _class_names(_class(module, cls)) for module, cls in MIXINS}


def test_translator_gui_lists_the_shared_mixins_first():
    cls = _class("translator_gui", "TranslatorGUI")
    bases = [ast.unparse(b) for b in cls.bases]
    u2 = [
        "SettingsPersistenceMixin", "RunEnvMixin", "ConfigStateMixin",
        "QAScannerMixin", "RetranslationMixin", "GlossaryManagerMixin", "QMainWindow",
    ]
    # U3 shared mixins (job runners, input preparation, pipelines) precede the U2 ones
    assert bases[-len(u2):] == u2
    assert set(bases[:-len(u2)]) <= {"TranslationPipelineMixin", "TextJobsMixin", "InputPreparationMixin"}


def test_shared_mixins_do_not_overlap_or_define_init():
    names = _shared_names()
    seen = {}
    for module, provided in names.items():
        assert "__init__" not in provided, module
        for name in provided:
            assert name not in seen, f"{name} defined by both {seen[name]} and {module}"
            seen[name] = module


def test_no_gui_mixin_or_translator_gui_body_shadows_a_moved_name():
    shared = set().union(*_shared_names().values())
    for module, cls in GUI_MIXINS:
        overlap = shared & _class_names(_class(module, cls))
        assert not overlap, f"{cls} defines shared names {sorted(overlap)}"
    body = _class_names(_class("translator_gui", "TranslatorGUI"))
    assert shared & body == set(HOOK_OVERRIDES), sorted((shared & body) ^ set(HOOK_OVERRIDES))
    hooks_with_defaults = {name for name in HOOK_OVERRIDES if name in shared}
    assert hooks_with_defaults == set(HOOK_OVERRIDES)


def test_other_settings_bindings_do_not_shadow_moved_names():
    fn = next(n for n in _tree("other_settings").body
              if isinstance(n, ast.FunctionDef) and n.name == "setup_other_settings_methods")
    bound = None
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "methods_to_bind" for t in node.targets):
            bound = set(ast.literal_eval(node.value))
    assert bound
    shared = set().union(*_shared_names().values())
    assert not bound & shared, sorted(bound & shared)
    # initialize_extraction_variables moved to owner_state and is re-exported
    reexports = [n for n in _tree("other_settings").body if isinstance(n, ast.ImportFrom) and n.module == "owner_state"]
    assert [a.name for a in reexports[0].names] == ["initialize_extraction_variables"]


def test_moved_methods_left_translator_gui():
    """The bodies are deleted from TranslatorGUI (no duplicates) and exist once in a mixin."""
    legacy_cls = next(n for n in ast.parse(_git_show("src/translator_gui.py")).body
                      if isinstance(n, ast.ClassDef) and n.name == "TranslatorGUI")
    legacy = _class_names(legacy_cls)
    shared = _shared_names()
    body = _class_names(_class("translator_gui", "TranslatorGUI"))
    moved = {name for provided in shared.values() for name in provided if name in legacy}
    assert len(moved) >= 70
    assert not (moved & body) - HOOK_OVERRIDES


# ---------------------------------------------------------------------------
# verbatim moves: the mixin method == the BASE_SHA method (modulo listed hook edits)
# ---------------------------------------------------------------------------

#: name -> (old fragment, new fragment) source substitutions made while moving.
_MOVE_EDITS = {
    "_get_multipass_refinement_mode": [("QApplication.instance()", None)],
    "_export_multipass_runtime_env": [("QApplication.instance()", None)],
    "_live_chapter_range_settings": [("TranslatorGUI._parse_chapter_range_text", "RunEnvMixin._parse_chapter_range_text")],
    "_export_chapter_range_runtime_env": [("TranslatorGUI._live_chapter_range_settings", "RunEnvMixin._live_chapter_range_settings")],
    "_is_valid_custom_prefix_endpoint_type": [("TranslatorGUI.", "RunEnvMixin.")],
    "_apply_forced_streaming_environment": [("_InputOutputDialog._FORCED_STREAM_ENV_KEYS", "FORCED_STREAM_ENV_KEYS")],
    "_sanitize_config_prompts": [("_atomic_json_write", None)],
    "_on_context_mode_changed": [("self.frame.addWidget", None)],
    # U9 P5b: settings_schema builds the bool_vars / str_vars tables (desktop_bool_vars /
    # desktop_str_vars). The old fragments come from the frozen copy of the literals, so this
    # entry also pins src/mobile/tools/frozen_desktop_tables.py to the BASE_SHA tables.
    "_init_variables": [
        (ast.unparse(node), {
            "bool_vars": "from settings_schema import desktop_bool_vars, desktop_str_vars\n"
                         "    bool_vars = desktop_bool_vars(self)",
            "str_vars": "str_vars = desktop_str_vars(self)",
        }[node.targets[0].id])
        for node in ast.walk(ast.parse(
            (SRC / "mobile" / "tools" / "frozen_desktop_tables.py").read_text(encoding="utf-8")))
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in ("bool_vars", "str_vars")
    ],
}


def _method_nodes(tree, cls_name):
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls_name)
    return {n.name: n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


def test_moved_methods_are_verbatim():
    legacy = _method_nodes(ast.parse(_git_show("src/translator_gui.py")), "TranslatorGUI")
    checked = 0
    for module, cls_name in MIXINS:
        for name, node in _method_nodes(_tree(module), cls_name).items():
            if name not in legacy or name in HOOK_OVERRIDES:
                continue
            old = ast.unparse(legacy[name])
            new = ast.unparse(node)
            if name in _MOVE_EDITS:
                for old_frag, new_frag in _MOVE_EDITS[name]:
                    assert old_frag in old, (name, old_frag)
                    if new_frag is not None:
                        old = old.replace(old_frag, new_frag)
                if all(n is not None for _o, n in _MOVE_EDITS[name]):
                    assert new == old, name
                checked += 1
                continue
            assert new == old, f"{module}.{cls_name}.{name} differs from {BASE_SHA[:12]}"
            checked += 1
    assert checked >= 60


# ---------------------------------------------------------------------------
# owner contract / startup replay / shims
# ---------------------------------------------------------------------------

def test_owner_contract_is_derived_and_satisfied(tmp_path, monkeypatch):
    import app_paths
    import headless_owner

    contract = headless_owner.OWNER_CONTRACT
    assert contract == headless_owner.compute_owner_contract()
    assert {"api_key_entry", "prompt_text", "delay_entry", "token_limit_entry", "append_log"} <= set(contract)
    monkeypatch.setattr(app_paths, "CONFIG_FILE", str(tmp_path / "config.json"))
    monkeypatch.setattr(app_paths, "__file__", str(tmp_path / "app_paths.py"))
    with _scrubbed_env(tmp_path):
        owner = headless_owner.HeadlessOwner({}, host=types.SimpleNamespace(log=lambda _m: None))
    missing = sorted(n for n in contract if not hasattr(owner, n) and n not in HEADLESS_DEFERRED)
    assert not missing, missing
    assert set(HEADLESS_DEFERRED) <= set(contract)


def _handler_calls(method_name):
    calls = []
    fn = next(n for n in _class("translator_gui", "TranslatorGUI").body
              if isinstance(n, ast.FunctionDef) and n.name == method_name)
    replay = {"_restore_authgem_project_selection", "_on_disable_temperature_toggle",
              "_on_auto_glossary_shortcut_changed", "_on_context_mode_changed",
              "_resolve_startup_target_language", "update_target_language", "_init_active_profile_prompt",
              "_update_auto_compression_factor"}
    for stmt in fn.body:
        for node in ast.walk(stmt):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name) and node.func.value.id == "self"
                    and node.func.attr in replay):
                calls.append((node.lineno, node.func.attr))
    return [name for _line, name in sorted(calls)]


def test_startup_handler_replay_matches_desktop_order():
    setup = next(n for n in _class("translator_gui", "TranslatorGUI").body
                 if isinstance(n, ast.FunctionDef) and n.name == "_setup_gui")
    sections = [node.func.attr for stmt in setup.body for node in [getattr(stmt, "value", None)]
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr.startswith(("create_", "_create_"))]
    desktop = []
    for section in sections:
        desktop += _handler_calls(section)
    desktop += [c for c in _handler_calls("_setup_gui")]
    replay_fn = next(n for n in _class("owner_state", "ConfigStateMixin").body
                     if isinstance(n, ast.FunctionDef) and n.name == "_replay_gui_startup_handlers")
    replay = [attr for _line, _col, attr in sorted(
        (node.lineno, node.col_offset, node.func.attr) for node in ast.walk(replay_fn)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name) and node.func.value.id == "self"
    )]
    assert replay == [
        "_restore_authgem_project_selection",  # _create_model_section (before the settings section)
        "_on_disable_temperature_toggle", "_on_auto_glossary_shortcut_changed", "_on_context_mode_changed",
        "_resolve_startup_target_language", "update_target_language", "_init_active_profile_prompt",
        "_update_auto_compression_factor",
    ]
    assert [c for c in desktop if c in replay] == replay


def test_widget_shim_sources_match_the_desktop_builders():
    import headless_owner

    text = _src_text("translator_gui")
    tree = ast.parse(text)
    statements = {ast.unparse(n) for n in ast.walk(tree) if isinstance(n, ast.Expr)}
    for attr, expr in headless_owner.STARTUP_WIDGET_SOURCES.items():
        assert ast.unparse(ast.parse(expr).body[0]) in statements, (attr, expr)


def test_combo_items_match_the_parity_fakes():
    import headless_owner
    from parity import scenarios

    assert headless_owner.CONTEXT_MODE_ITEMS == scenarios.CONTEXT_MODE_ITEMS
    assert headless_owner.MULTIPASS_ITEMS == scenarios.MULTIPASS_ITEMS
    assert headless_owner.REMOVE_ARTIFACTS_ITEMS == scenarios.REMOVE_ARTIFACTS_ITEMS
    assert headless_owner.AUTO_GLOSSARY_SHORTCUT_ITEMS == scenarios.AUTO_GLOSSARY_SHORTCUT_ITEMS


def test_shims_answer_hasattr_like_qt():
    from headless_owner import CheckShim, ComboShim, PlainTextShim, TextShim

    assert hasattr(TextShim(), "text") and not hasattr(TextShim(), "isChecked")
    assert hasattr(CheckShim(), "isChecked") and hasattr(CheckShim(), "text")
    assert hasattr(PlainTextShim(), "toPlainText") and not hasattr(PlainTextShim(), "text")
    combo = ComboShim.with_data((("A", "a"), ("B", "b")), "zz")
    assert combo.currentIndex() == 0 and combo.currentData() == "a" and not hasattr(combo, "text")
    for shim in (TextShim(), CheckShim(), PlainTextShim(), combo):
        assert not hasattr(shim, "get")
    # PySide6: QLineEdit(None) / setText(None) / QTextEdit.setPlainText(None) read back as ''
    entry, editor = TextShim(None), PlainTextShim(None)
    assert entry.text() == "" and editor.toPlainText() == ""
    entry.setText("x"), entry.setText(None), editor.setText(None), editor.setPlainText(None)
    assert entry.text() == "" and editor.toPlainText() == ""
    assert TextShim(0).text() == "0"


# ---------------------------------------------------------------------------
# HeadlessOwner behaviour
# ---------------------------------------------------------------------------

def _owner(tmp_path, monkeypatch, config, **kwargs):
    """HeadlessOwner(config) in a scrubbed env; returns (owner, startup os.environ, env builder)."""
    import app_paths
    from headless_owner import HeadlessOwner

    monkeypatch.setattr(app_paths, "CONFIG_FILE", str(tmp_path / "config.json"))
    monkeypatch.setattr(app_paths, "__file__", str(tmp_path / "app_paths.py"))
    with _scrubbed_env(tmp_path):
        owner = HeadlessOwner(config, host=types.SimpleNamespace(log=lambda _m: None), **kwargs)
        startup = dict(os.environ)
        env = owner._get_environment_variables(str(tmp_path / "book.epub"), owner.api_key_entry.text())
    return owner, startup, env


def test_null_text_settings_read_back_empty_like_qt(tmp_path, monkeypatch):
    owner, _startup, env = _owner(tmp_path, monkeypatch, {"chapter_range": None, "vertex_ai_location": None})
    assert owner.chapter_range_entry.text() == "" and owner.vertex_location_entry.text() == ""
    assert env["CHAPTER_RANGE"] == ""
    collected = owner._collect_live_settings()
    assert collected["chapter_range"] == "" and collected["vertex_ai_location"] == ""


def _legacy_statements(method, predicate):
    tree = ast.parse(_git_show("src/translator_gui.py"))
    fn = _method_nodes(tree, "TranslatorGUI")[method]
    return [s for s in fn.body if predicate(s)]


def _mixin_body(name):
    fn = next(n for n in _class("owner_state", "ConfigStateMixin").body
              if isinstance(n, ast.FunctionDef) and n.name == name)
    body = fn.body[1:] if isinstance(fn.body[0], ast.Expr) and isinstance(fn.body[0].value, ast.Constant) else fn.body
    return [ast.unparse(s) for s in body]


def _desktop_method_source(name):
    fn = _method_nodes(_tree("translator_gui"), "TranslatorGUI")[name]
    return [ast.unparse(s) for s in fn.body]


def test_authgem_project_restore_is_moved_verbatim():
    legacy = _legacy_statements(
        "_create_model_section",
        lambda s: ast.unparse(s).startswith(("saved_project = self.config.get('authgem_project'",
                                             "if saved_project:\n    try:\n        import authgem_auth")))
    assert len(legacy) == 2
    assert _mixin_body("_restore_authgem_project_selection") == [ast.unparse(s) for s in legacy]
    model_section = _desktop_method_source("_create_model_section")
    assert "self._restore_authgem_project_selection()" in model_section
    assert not any(line.startswith("saved_project = ") for line in model_section)


def test_headless_owner_restores_the_authgem_project_like_the_desktop(tmp_path, monkeypatch):
    authgem_auth = pytest.importorskip("authgem_auth")
    monkeypatch.setattr(authgem_auth, "_cached_project_id", {})
    monkeypatch.setattr(authgem_auth, "_project_set_by_gui", {})
    _owner_obj, startup, _env = _owner(tmp_path, monkeypatch, {"authgem_project": "my-proj-123",
                                                               "model": "authgem/gemini-2.5-pro"})
    assert authgem_auth._cached_project_id == {0: "my-proj-123"}
    assert authgem_auth._project_set_by_gui == {0: True}
    assert startup["GOOGLE_CLOUD_PROJECT"] == "my-proj-123"
    monkeypatch.setattr(authgem_auth, "_cached_project_id", {})
    _owner(tmp_path, monkeypatch, {"model": "authgem/gemini-2.5-pro"})
    assert authgem_auth._cached_project_id == {}  # nothing saved -> nothing restored


def test_auto_encrypt_startup_save_is_moved_verbatim():
    legacy = _legacy_statements("__init__", lambda s: isinstance(s, ast.Try) and "needs_encryption" in ast.unparse(s))
    assert len(legacy) == 1
    assert _mixin_body("_auto_encrypt_api_keys") == [ast.unparse(legacy[0])]
    init = _desktop_method_source("__init__")
    assert init[-1] == "self._auto_encrypt_api_keys()"  # still the last step of __init__
    assert not any("needs_encryption" in line for line in init)


def test_headless_owner_replays_the_auto_encrypt_save(tmp_path, monkeypatch):
    # desktop: a decrypted config with a plain api_key re-saves after initialize_environment_variables,
    # which re-exports the saved settings (GLOSSARY_ENTRY_TYPE_FILTER_MODE: init 'none', save 'Loose')
    _o, with_key, _e = _owner(tmp_path, monkeypatch, {"api_key": "sk-test-plain"})
    _o, encrypted, _e = _owner(tmp_path, monkeypatch, {"api_key": "ENC:already"})
    _o, no_key, _e = _owner(tmp_path, monkeypatch, {})
    assert with_key["GLOSSARY_ENTRY_TYPE_FILTER_MODE"] == "Loose"
    assert encrypted["GLOSSARY_ENTRY_TYPE_FILTER_MODE"] == "none"
    assert no_key["GLOSSARY_ENTRY_TYPE_FILTER_MODE"] == "none"


_EPUB_COMPILE_PROBE = r"""
import json, os, sys, zipfile
for name in ("PySide6", "shiboken6", "translator_gui", "dpi_setup"):
    sys.modules[name] = None
sys.path.insert(0, {src!r})
root = {root!r}
os.environ.update({{"HOME": root, "USERPROFILE": root, "GLOSSARION_LIBRARY_DIR": os.path.join(root, "Library"),
                   "GLOSSARION_APP_DIR": os.path.join(root, "app")}})
book = os.path.join(root, "raw", "Some Book.epub")
os.makedirs(os.path.dirname(book))
with zipfile.ZipFile(book, "w") as zf:
    zf.writestr("OEBPS/chapter0001.xhtml", "<p>1</p>")
folder = os.path.join(root, "out", "Workspace")
os.makedirs(folder)
with open(os.path.join(folder, "source_epub.txt"), "w", encoding="utf-8") as fh:
    fh.write(book)
import app_paths
app_paths.CONFIG_FILE = os.path.join(root, "config.json")
import run_env
from headless_owner import HeadlessOwner
logs = []
owner = HeadlessOwner({{"model": "gpt-4o"}}, host=type("H", (), {{"log": lambda self, m: logs.append(str(m))}})())
delta = run_env.build_epub_compile_env(owner, folder)
print("RESULT=" + json.dumps({{"epub_path": delta["set"].get("EPUB_PATH"), "book": book, "logs": logs,
                              "epub_library": "epub_library" in sys.modules,
                              "qt": [n for n in ("PySide6.QtCore", "PySide6.QtWidgets") if sys.modules.get(n)]}}))
"""


def test_epub_compile_env_resolves_the_source_epub_without_qt(tmp_path):
    """run_env's EPUB compile env resolves the source EPUB through library_core with PySide6 blocked."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("GLOSSARION_")}
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run(
        [sys.executable, "-c", _EPUB_COMPILE_PROBE.format(src=str(SRC), root=str(tmp_path))],
        capture_output=True, text=True, encoding="utf-8", env=env, timeout=300, cwd=str(tmp_path),
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    line = next(l for l in proc.stdout.splitlines() if l.startswith("RESULT="))
    result = json.loads(line[len("RESULT="):])
    assert result["epub_path"] == result["book"], result["logs"]
    assert not result["epub_library"] and not result["qt"]
    assert not any("Could not resolve source EPUB" in l for l in result["logs"])


def test_library_core_is_moved_verbatim_and_reexported():
    import library_core

    legacy = ast.parse(_git_show("src/epub_library.py"))
    legacy_defs = {}
    for node in legacy.body:
        if isinstance(node, ast.FunctionDef):
            legacy_defs[node.name] = ast.unparse(node)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    legacy_defs[t.id] = ast.unparse(node)
    defined = {}
    for node in _tree("library_core").body:
        if isinstance(node, ast.FunctionDef):
            defined[node.name] = ast.unparse(node)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id not in ("logger", "__all__"):
                    defined[t.id] = ast.unparse(node)
    # U5 extended library_core with new shared APIs (LibraryEnv, plain shelf / details
    # objects, mixins); every name that epub_library defined at BASE_SHA is a move and
    # must stay verbatim, except the documented U5 seams (tests/parity/DISCREPANCIES.md).
    adapted = {"_default_output_root"}  # LibraryEnv output-root seam (desktop never installs one)
    # owner edits made after the move (in library_core, which desktop now uses): (old, new) fragments
    later_edits = {
        # 3646b1a3 "image reference": the image-reference sidecar is a progress sidecar too
        "_PROGRESS_SIDECAR_FILENAMES": ("'image_rename_map.json'})", "'image_rename_map.json', 'image_reference_map.json'})"),
    }
    moved = {name: text for name, text in defined.items() if name in legacy_defs}
    assert set(moved) <= set(library_core.__all__)
    for name, text in moved.items():
        if name in adapted:
            continue
        expected = legacy_defs[name]
        if name in later_edits:
            old, new = later_edits[name]
            assert old in expected, name
            expected = expected.replace(old, new)
        assert text == expected, f"library_core.{name} differs from epub_library @ {BASE_SHA[:12]}"
    # epub_library no longer defines them; it imports the same objects
    current = {n.name for n in _tree("epub_library").body if isinstance(n, ast.FunctionDef)}
    assert not current & set(moved)
    reexport = [n for n in _tree("epub_library").body if isinstance(n, ast.ImportFrom) and n.module == "library_core"]
    assert set(moved) <= {a.name for a in reexport[0].names}

def test_headless_owner_never_writes_config_json(tmp_path, monkeypatch):
    import app_paths
    from headless_owner import HeadlessOwner

    target = tmp_path / "config.json"
    monkeypatch.setattr(app_paths, "CONFIG_FILE", str(target))
    monkeypatch.setattr(app_paths, "__file__", str(tmp_path / "app_paths.py"))
    broken = "Keep original Korean quotation marks (, ' ', 「」, 『』) as-is"
    cfg = {"prompt_profiles": {"x": broken}, "model": "gpt-4o"}
    original = copy.deepcopy(cfg)
    with _scrubbed_env(tmp_path):
        owner = HeadlessOwner(cfg, host=types.SimpleNamespace(log=lambda _m: None))
    assert not target.exists()
    assert cfg == original  # the caller's dict is not mutated
    assert owner.config["sanitization_korean_quotes_fixed"] is True  # sanitizer ran in memory
    assert owner.config["auto_update_check"] is True


def test_headless_owner_records_library_raw_inputs_like_the_desktop(tmp_path, monkeypatch):
    """U5: the run set-up's registry hook goes through library_core (desktop: epub_library -> the same function)."""
    import library_core
    import translation_pipeline
    from headless_owner import HeadlessOwner

    library = tmp_path / "Library"
    monkeypatch.setenv("GLOSSARION_LIBRARY_DIR", str(library))
    book = tmp_path / "Book.epub"
    book.write_bytes(b"PK")
    owner = HeadlessOwner.__new__(HeadlessOwner)  # the hook needs no owner state
    assert owner._record_library_raw_inputs([str(book), str(tmp_path / "missing.epub"), "", None]) is None
    assert [os.path.normcase(p) for p in library_core.load_library_raw_inputs()] == [os.path.normcase(str(book))]
    # the mixin default stays GUI-free and inert for other owners; desktop keeps its epub_library hook
    assert translation_pipeline.PipelineHooksMixin._record_library_raw_inputs(owner, [str(book)]) is None
    tg_hook = " ".join(_desktop_method_source("_record_library_raw_inputs"))
    assert "from epub_library import record_library_raw_input" in tg_hook
    reexports = {a.name for n in _tree("epub_library").body
                 if isinstance(n, ast.ImportFrom) and n.module == "library_core" for a in n.names}
    assert "record_library_raw_input" in reexports  # desktop and mobile write through one function


def test_fresh_install_headless_owner_runs_like_the_desktop(tmp_path, monkeypatch):
    import app_paths
    from headless_owner import HeadlessOwner

    monkeypatch.setattr(app_paths, "CONFIG_FILE", str(tmp_path / "config.json"))
    monkeypatch.setattr(app_paths, "__file__", str(tmp_path / "app_paths.py"))
    with _scrubbed_env(tmp_path):
        owner = HeadlessOwner({}, host=types.SimpleNamespace(log=lambda _m: None))
        env = owner._get_environment_variables(str(tmp_path / "book.epub"), owner.api_key_entry.text())
        startup_glossary = os.environ.get("AUTO_GLOSSARY_MODE")
    assert owner.model_var == "authgpt/gpt-6-luna"
    assert env["MODEL"] == "authgpt/gpt-6-luna"
    # the startup glossary-mode shortcut handler overrides the _init_variables default 'balanced'
    assert owner.auto_glossary_mode_var == "off" and env["AUTO_GLOSSARY_MODE"] == "off"
    assert startup_glossary == "off"
    assert env["SEND_INTERVAL_SECONDS"] == "5" and env["CONNECT_TIMEOUT"] == "10.0"
    assert "PRESERVE_ORIGINAL_FORMATOPTIMIZE_FOR_OCR" in env  # desktop quirk preserved
    assert len(owner.config) > 400  # the startup save_config populated the settings map (in memory)


def test_model_and_api_key_arguments(tmp_path, monkeypatch):
    import app_paths
    from headless_owner import HeadlessOwner

    monkeypatch.setattr(app_paths, "CONFIG_FILE", str(tmp_path / "config.json"))
    monkeypatch.setattr(app_paths, "__file__", str(tmp_path / "app_paths.py"))
    with _scrubbed_env(tmp_path):
        owner = HeadlessOwner({"model": "gpt-4o", "api_key": "from-config"}, model="gemini-2.5-flash",
                              api_key="sk-from-keystore", host=types.SimpleNamespace(log=lambda _m: None))
        env = owner._get_environment_variables(str(tmp_path / "b.epub"), owner.api_key_entry.text())
    assert owner.model_var == "gemini-2.5-flash" and env["MODEL"] == "gemini-2.5-flash"
    assert env["API_KEY"] == "sk-from-keystore"


def test_collect_live_settings_is_side_effect_free(tmp_path, monkeypatch):
    import app_paths
    from headless_owner import HeadlessOwner

    monkeypatch.setattr(app_paths, "CONFIG_FILE", str(tmp_path / "config.json"))
    monkeypatch.setattr(app_paths, "__file__", str(tmp_path / "app_paths.py"))
    with _scrubbed_env(tmp_path):
        owner = HeadlessOwner({"custom_prefix_routes": [{"prefix": "lan", "routing": "http://h:1/v1"}],
                               "delay": 2}, host=types.SimpleNamespace(log=lambda _m: None))
        owner.custom_prefix_routes = [{"prefix": "x", "routing": "http://changed"}]  # unsaved edit
        owner.delay_entry.setText("7")
        cfg_before = copy.deepcopy(owner.config)
        attrs_before = {k: copy.deepcopy(v) for k, v in vars(owner).items() if k in (
            "custom_prefix_routes", "custom_entry_types", "use_header_as_output_var")}
        env_before = dict(os.environ)
        cfg_id = id(owner.config)
        collected = owner._collect_live_settings()
        assert dict(os.environ) == env_before
    assert id(owner.config) == cfg_id and owner.config == cfg_before
    assert {k: v for k, v in vars(owner).items() if k in attrs_before} == attrs_before
    assert "_sync_custom_prefix_routes_env" not in vars(owner)
    assert collected["delay"] == 7.0  # widget-first, save_config converter
    assert collected["custom_prefix_routes"][0]["prefix"] == "x/"


def test_build_functions_leave_the_environment_unchanged(tmp_path, monkeypatch):
    import app_paths
    import run_env
    from headless_owner import HeadlessOwner

    monkeypatch.setattr(app_paths, "CONFIG_FILE", str(tmp_path / "config.json"))
    monkeypatch.setattr(app_paths, "__file__", str(tmp_path / "app_paths.py"))
    folder = tmp_path / "out" / "Book"
    folder.mkdir(parents=True)
    with _scrubbed_env(tmp_path):
        owner = HeadlessOwner({"model": "gpt-4o"}, host=types.SimpleNamespace(log=lambda _m: None))
        for key in ("EPUB_LAYOUT_MODE", "USE_TOC_NCX", "TRANSLATE_TOC_NCX"):
            os.environ.pop(key, None)
        before = dict(os.environ)
        epub = run_env.build_epub_compile_env(owner, str(folder))
        pdf = run_env.build_pdf_compile_env(owner)
        startup = run_env.build_startup_env(owner)
        glossary = run_env.build_glossary_env(owner, str(tmp_path / "Book.epub"), "sk-x")
        after = dict(os.environ)
    assert before == after
    assert epub["set"]["EPUB_LAYOUT_MODE"] == "auto" and "EPUB_CSS_OVERRIDE_PATH" not in epub["set"]
    assert pdf["set"]["TRANSLATE_TOC_NCX"] == pdf["set"]["USE_TOC_NCX"]
    assert startup["set"]  # initialize_environment_variables re-exports the startup env
    assert glossary.env_updates["MODEL"] == "gpt-4o" and glossary.env_updates["API_KEY"] == "sk-x"
    assert glossary.output_path.endswith("Book_glossary.json")


# ---------------------------------------------------------------------------
# DirectTextRunOptions vs the original _InputOutputDialog code
# ---------------------------------------------------------------------------

def _legacy_direct_text_block():
    text = _git_show("src/translator_gui.py")
    tree = ast.parse(text)
    dlg = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "_InputOutputDialog")
    for node in ast.walk(dlg):
        body = getattr(node, "body", None)
        if not isinstance(body, list):
            continue
        for i, stmt in enumerate(body):
            if ast.unparse(stmt) == "gui.selected_files = [self._temp_input]":
                end = next(j for j in range(i, len(body))
                           if ast.unparse(body[j]).startswith("gui.enable_refinement_output_mode_var"))
                lines = text.split("\n")[stmt.lineno - 1:body[end].end_lineno]
                return textwrap.dedent("\n".join(lines))
    raise AssertionError("legacy direct text block not found")


def _new_direct_text_block():
    text = _src_text("translator_gui")
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and ast.unparse(node).startswith("DirectTextRunOptions("):
            lines = text.split("\n")[node.lineno - 1:node.end_lineno]
            return textwrap.dedent("\n".join(lines))
    raise AssertionError("DirectTextRunOptions call not found in translator_gui")


def test_direct_text_options_match_the_original_dialog_block():
    from headless_owner import DirectTextRunOptions

    legacy_src, new_src = _legacy_direct_text_block(), _new_direct_text_block()
    rng = random.Random(11)
    modes = ["text", "vision", "image", "video", "audio", "refinement", "refine", "", None]
    for _ in range(300):
        dialog = types.SimpleNamespace(
            _temp_input=f"in{rng.randint(0, 9)}.txt", _temp_root=f"root{rng.randint(0, 9)}",
            force_multipass_off_checkbox=types.SimpleNamespace(isChecked=lambda v=rng.random() < 0.5: v),
            skip_thinking_checkbox=types.SimpleNamespace(isChecked=lambda v=rng.random() < 0.5: v),
            skip_prompt_profile_checkbox=types.SimpleNamespace(isChecked=lambda v=rng.random() < 0.5: v),
            _run_source_is_attachment=rng.random() < 0.5, _run_output_mode=rng.choice(modes),
        )
        policy = rng.choice(["manual", "no_glossary", "none", "attachments_only"])
        dialog._selected_glossary_override_mode = lambda p=policy: p
        dialog._force_no_glossary_for_mode = lambda mode, att: str(mode or "attachments_only") == "no_glossary" or (
            str(mode or "attachments_only") == "attachments_only" and not bool(att))
        local = {
            "manual_glossary_path": rng.choice(["", None, "g.csv"]),
            "text": rng.choice(["hello", ""]), "attachment": rng.random() < 0.5,
            "attachment_prompt_role": rng.choice(["user", "system", "assistant"]),
        }
        results = []
        for src in (legacy_src, new_src):
            gui = types.SimpleNamespace(manual_glossary_map={"a": "b"})
            ns = {"self": dialog, "gui": gui, "os": os, "DirectTextRunOptions": DirectTextRunOptions, **local}
            exec(compile(src, "<direct text>", "exec"), ns)
            results.append(dict(vars(gui)))
        assert results[0] == results[1]
        assert set(results[1]) - {"manual_glossary_map"} <= set(DirectTextRunOptions.OWNER_ATTRS) | set(
            DirectTextRunOptions.MANUAL_GLOSSARY_ATTRS)


def test_direct_text_options_round_trip_and_scenario_attrs():
    from headless_owner import DirectTextRunOptions
    from parity import scenarios

    opts = DirectTextRunOptions(selected_files=["a.txt"], attachment_prompt="Do it", attachment_prompt_role="system",
                                output_mode="vision", manual_glossary_path="g.csv", skip_thinking=True)
    owner = opts.apply_to(types.SimpleNamespace())
    assert DirectTextRunOptions.from_owner(owner) == opts
    assert owner._direct_text_use_manual_glossary is True and owner.manual_glossary_map == {}
    run_attrs = scenarios.get("direct_text_attachment")["run_attrs"]
    expected = DirectTextRunOptions(
        attachment_prompt=run_attrs["_direct_text_attachment_prompt"],
        attachment_prompt_role=run_attrs["_direct_text_attachment_prompt_role"],
        skip_prompt_profile=run_attrs["_direct_text_skip_prompt_profile"],
        output_mode=run_attrs["_direct_text_output_mode"],
        force_multipass_off=run_attrs["_direct_text_force_multipass_off"],
        manual_glossary_path=run_attrs["_direct_text_manual_glossary_path"],
        force_no_glossary=run_attrs["_direct_text_force_no_glossary"],
        skip_thinking=run_attrs["_direct_text_skip_thinking"],
    ).apply_to(types.SimpleNamespace())
    for name, value in run_attrs.items():
        assert getattr(expected, name) == value, name


# ---------------------------------------------------------------------------
# tier G: goldens (desktop composition and HeadlessOwner) / tier R: round trip
# ---------------------------------------------------------------------------

def _worktree_text(_sha, relpath):
    path = REPO_ROOT / relpath
    if not path.exists():
        raise FileNotFoundError(relpath)
    return path.read_bytes().decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")


class _RecordingHost:
    def __init__(self, recorder):
        self.recorder = recorder

    def log(self, message):
        self.recorder.record("append_log", [message], {})


@pytest.fixture(scope="module")
def parity(tmp_path_factory):
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from parity import capture_golden as cg
    from parity import fakes, freeze_legacy

    _git_show("src/translator_gui.py")
    with pytest.MonkeyPatch.context() as mp:
        try:
            legacy_bundle = freeze_legacy.load_legacy(BASE_SHA)
        except FileNotFoundError:
            legacy_dir = tmp_path_factory.mktemp("legacy")
            mp.setattr(freeze_legacy, "LEGACY_DIR", legacy_dir)
            freeze_legacy.freeze(BASE_SHA, legacy_dir)
            legacy_bundle = freeze_legacy.load_legacy(BASE_SHA)
        out_dir = tmp_path_factory.mktemp("worktree")
        mp.setattr(freeze_legacy, "LEGACY_DIR", out_dir)
        mp.setattr(freeze_legacy, "resolve_sha", lambda rev="HEAD": WORKTREE_SHA)
        mp.setattr(freeze_legacy, "git_show_text", _worktree_text)
        freeze_legacy.freeze("WORKTREE", out_dir)
        new_bundle = freeze_legacy.load_legacy(WORKTREE_SHA)
    manifest = new_bundle.manifest
    assert manifest["layouts"] == {"init_config_block": "u2", "gui_state": "u2", "gui_handlers": "u2"}
    # U3+ worktrees also freeze the job-runner mixins (and helper modules) before the U2 ones
    assert [m for m, _c in manifest["shared_mixins"]][-3:] == ["settings_persistence", "run_env", "owner_state"]
    assert not manifest["missing"], manifest["missing"]

    desktop_factory = fakes.make_legacy_owner_factory(new_bundle)
    desktop_factory.kind = "worktree-u2"

    def headless_factory(scenario, ctx, config_override=None):
        from config_store import load_config
        from headless_owner import HeadlessOwner

        fakes.patch_shared_module_paths(ctx)
        if config_override is not None:
            config = ctx.sandbox.resolve(copy.deepcopy(config_override))
        else:
            try:
                config = load_config(str(ctx.sandbox.config_file))
            except Exception:
                config = {}
        return HeadlessOwner(config, host=_RecordingHost(ctx.recorder))

    headless_factory.kind = "headless"
    goldens = {}

    def golden(name):
        if name not in goldens:
            from parity import scenarios

            try:
                goldens[name] = cg.load_golden(BASE_SHA, name)
            except (FileNotFoundError, OSError):
                legacy_factory = fakes.make_legacy_owner_factory(legacy_bundle)
                goldens[name] = cg.roundtrip(cg.capture(legacy_factory, scenarios.get(name)))
        return goldens[name]

    return {"cg": cg, "fakes": fakes, "desktop": desktop_factory, "headless": headless_factory, "golden": golden,
            "worktree_bundle": new_bundle}


def _scenario_names():
    from parity import scenarios

    return list(scenarios.SCENARIO_NAMES)


def test_oracle_frozen_at_a_mixin_commit_never_runs_the_live_mixins(parity, monkeypatch):
    """Freezing a commit that already has the shared mixins (U3 freezes U2) keeps them frozen:
    whole-module copies, loaded privately, used by the legacy owner even if the live modules change."""
    import owner_state
    import run_env
    import settings_persistence
    from parity import freeze_legacy, roundtrip, scenarios

    bundle, cg, fakes = parity["worktree_bundle"], parity["cg"], parity["fakes"]
    live = {"owner_state": owner_state, "run_env": run_env, "settings_persistence": settings_persistence}
    frozen = bundle.manifest["frozen_mixins"]
    assert set(live) <= set(frozen)  # U3+: text_jobs / input_preparation / job_runner / stop_control too
    classes = fakes.shared_mixin_classes(bundle)
    assert [c.__name__ for c in classes][-3:] == ["SettingsPersistenceMixin", "RunEnvMixin", "ConfigStateMixin"]
    for module in live:
        info = frozen[module]
        assert info["source_sha256"] == freeze_legacy._sha256(_worktree_text(None, info["source_file"]))
        frozen_cls = getattr(bundle.mixins[module], info["class"])
        assert frozen_cls in classes and frozen_cls is not getattr(live[module], info["class"])
        assert frozen_cls.__module__.startswith("parity_legacy_")
    # imports between frozen mixin modules resolve to the frozen copies
    frozen_state, frozen_env = vars(bundle.mixins["owner_state"]), vars(bundle.mixins["run_env"])
    assert frozen_state["_format_plain_decimal_setting"] is frozen_env["_format_plain_decimal_setting"]
    assert frozen_state["_format_plain_decimal_setting"] is not run_env._format_plain_decimal_setting
    # break the live run_env: the legacy owner (frozen copies) still reproduces the golden env
    def broken(self, *args, **kwargs):
        raise AssertionError("the legacy oracle ran the live run_env")

    monkeypatch.setattr(run_env.RunEnvMixin, "_get_environment_variables", broken)
    scenario = scenarios.get("gemini_text_typical")
    result = cg.capture(parity["desktop"], scenario, entries=["translation_env"])
    entry = result["entries"]["translation_env"]
    assert not entry.get("exception") and "boot_error" not in entry, entry
    assert cg.roundtrip(entry["result"]) == parity["golden"]("gemini_text_typical")["entries"]["translation_env"]["result"]
    # tier R still runs the LIVE mixins on a frozen-mixin legacy owner
    owner = type("LegacyFake", (fakes.FakeState, bundle.methods) + classes, {})(fakes.CallRecorder())
    roundtrip._with_shared_mixins(owner)
    assert type(owner)._collect_live_settings is settings_persistence.SettingsPersistenceMixin._collect_live_settings
    assert type(owner)._get_environment_variables is broken


@pytest.mark.parametrize("name", _scenario_names())
def test_desktop_composition_reproduces_goldens(parity, name):
    from parity import scenarios

    cg = parity["cg"]
    result = cg.capture(parity["desktop"], scenarios.get(name))
    problems = cg.diff(parity["golden"](name), cg.roundtrip(result))
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("name", _scenario_names())
def test_headless_owner_reproduces_golden_env(parity, name):
    from parity import scenarios

    cg = parity["cg"]
    scenario = scenarios.get(name)
    entries = [e for e in scenario["entries"] if e in HEADLESS_ENTRIES] + ["boot"]
    result = cg.capture(parity["headless"], scenario, entries=entries)
    golden = parity["golden"](name)
    problems = []
    for entry in entries:
        got, exp = cg.roundtrip(result["entries"][entry]), golden["entries"][entry]
        if entry == "boot":  # GUI attrs differ by design (shims vs Qt widgets); env + config must not
            got = {"env": got["final"]["env"], "config": got["final"]["config"]}
            exp = {"env": exp["final"]["env"], "config": exp["final"]["config"]}
        problems += [f"[{entry}] {p}" for p in cg.diff(exp, got)]
    assert not problems, "\n".join(problems)
    if name == "fresh_install":
        env = result["entries"]["translation_env"]["result"]
        assert env["MODEL"] == "authgpt/gpt-6-luna" and env["AUTO_GLOSSARY_MODE"] == "off"


def _observe(owner, scenario, ctx):
    path = ctx.sandbox.path(scenario["input"])
    return ctx.norm({
        "translation_env": owner._get_environment_variables(path, owner.api_key_entry.text()),
        "glossary_env_mappings": dict(owner._glossary_env_mappings()),
        "glossary_request": owner._current_glossary_request_env(),
        "auto_glossary_mode": owner._current_auto_glossary_mode(),
        "multipass": owner._get_multipass_refinement_mode(),
        "chapter_range": owner._live_chapter_range_settings(),
    })


def _run_attrs(scenario, sandbox):
    attrs = scenario.get("run_attrs") or {}
    if callable(attrs):
        attrs = attrs(sandbox)
    return sandbox.resolve(attrs)


def _boot_and_observe(factory, scenario, *, config_file=None, config_override=None):
    from parity import normalize

    scen = dict(scenario)
    if config_file is not None or config_override is not None:
        scen["config"] = config_file  # None -> no config.json (owner built from the dict)
    with normalize.CaptureContext(scen, "roundtrip") as ctx:
        owner = factory(scen, ctx, config_override) if config_override is not None else factory(scen, ctx)
        for key, value in _run_attrs(scenario, ctx.sandbox).items():
            setattr(owner, key, value)
        boot_env = ctx.norm(normalize.env_delta(ctx.baseline_env, normalize.effective_env()))
        observed = _observe(owner, scenario, ctx)
        collected = owner._collect_live_settings() if hasattr(owner, "_collect_live_settings") else None
        api_key = owner.api_key_entry.text()
        unsandboxed = ctx.norm(collected) if collected is not None else None
    return {"observed": observed, "boot_env": boot_env, "collected": unsandboxed, "api_key": api_key}


def _env_keys(problems):
    keys = set()
    for p in problems:
        parts = p.split(":")[0].strip("/").split("/")
        keys.add(parts[-1].split("[")[0])
    return keys


@pytest.mark.parametrize("name", _scenario_names())
def test_roundtrip_headless_equals_desktop_after_save(parity, name):
    from parity import scenarios

    cg = parity["cg"]
    scenario = scenarios.get(name)
    live = _boot_and_observe(parity["desktop"], scenario)
    saved = live["collected"]  # '<SANDBOX>' tokens; re-resolved in each new sandbox
    assert saved and saved.get("model")
    # R1: HeadlessOwner(saved settings) == a desktop restarted from the same config.json
    restart = _boot_and_observe(parity["desktop"], scenario, config_file=saved)
    headless = _boot_and_observe(
        lambda scen, ctx, override: parity["headless"](scen, ctx, override), scenario, config_override=saved)
    r1 = cg.diff(cg.roundtrip(restart["observed"]), cg.roundtrip(headless["observed"]))
    r1 += [f"[boot env] {p}" for p in cg.diff(cg.roundtrip(restart["boot_env"]), cg.roundtrip(headless["boot_env"]))]
    assert not r1, "\n".join(r1)
    # R2: the first-run desktop differs from the restart only by the recorded desktop quirks
    r2 = cg.diff(cg.roundtrip(live["observed"]), cg.roundtrip(restart["observed"]), limit=200)
    r2 += cg.diff(cg.roundtrip(live["boot_env"]), cg.roundtrip(restart["boot_env"]), limit=200)
    unexpected = _env_keys(r2) - set(KNOWN_ROUNDTRIP_DIVERGENCES)
    assert not unexpected, "\n".join(r2)


# ---------------------------------------------------------------------------
# lifted handler: ConfigStateMixin._on_auto_glossary_shortcut_changed vs the legacy
# nested function of _create_settings_section (differential, seeded states)
# ---------------------------------------------------------------------------

def _legacy_shortcut_handler_factory():
    text = _git_show("src/translator_gui.py")
    tree = ast.parse(text)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TranslatorGUI")
    section = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_create_settings_section")
    nested = next(n for n in section.body
                  if isinstance(n, ast.FunctionDef) and n.name == "_on_auto_glossary_shortcut_changed")
    src = textwrap.dedent("\n".join(text.split("\n")[nested.lineno - 1:nested.end_lineno]))
    factory_src = "def _make(self):\n" + textwrap.indent(src, "    ") + "\n    return _on_auto_glossary_shortcut_changed\n"
    ns = {"os": os}
    exec(compile(factory_src, "<legacy shortcut handler>", "exec"), ns)
    return ns["_make"]


class _Calls(list):
    def method(self, name, result=None):
        def call(*args, **kwargs):
            self.append((name, args, tuple(sorted(kwargs.items()))))
            return result
        return call


class _StyledCheck:
    """Checkbox with the style()/polish surface the handler touches."""

    def __init__(self, calls, name, checked):
        self._calls, self._name, self._checked = calls, name, checked

    def isChecked(self):
        return self._checked

    def setChecked(self, value):
        self._calls.append((f"{self._name}.setChecked", (value,), ()))
        self._checked = bool(value)

    def blockSignals(self, value):
        self._calls.append((f"{self._name}.blockSignals", (value,), ()))

    def update(self):
        self._calls.append((f"{self._name}.update", (), ()))

    def style(self):
        calls, name = self._calls, self._name
        return types.SimpleNamespace(unpolish=lambda w: calls.append((f"{name}.unpolish", (), ())),
                                     polish=lambda w: calls.append((f"{name}.polish", (), ())))


def _random_shortcut_owner(rng, tmp_path):
    from headless_owner import ComboShim, TextShim

    calls = _Calls()
    owner = types.SimpleNamespace()
    owner.config = {k: rng.choice([True, False, "x"]) for k in rng.sample(
        ["auto_glossary_mode", "enable_auto_glossary", "append_glossary", "append_glossary_auto_load",
         "fuzzy_auto_mapping"], rng.randint(0, 5))}
    for name in ("append_glossary_checkbox", "append_glossary_auto_load_checkbox", "fuzzy_auto_mapping_checkbox"):
        if rng.random() < 0.5:
            setattr(owner, name, _StyledCheck(calls, name, rng.random() < 0.5))
    if rng.random() < 0.5:
        owner.auto_glossary_mode_combo = ComboShim([(str(i), i) for i in range(8)], index=0)
    glossary = tmp_path / "g.csv"
    glossary.write_text("x", encoding="utf-8")
    owner.selected_files = rng.choice([[], ["a.epub"], ["a.epub", "b.epub"], ["a.pdf"], ["A.EPUB", "x.txt"]])
    owner.save_config = calls.method("save_config", True)
    owner.append_log = calls.method("append_log")
    if rng.random() < 0.7:
        owner._autofill_glossary_for_current_selection = calls.method("_autofill")
    if rng.random() < 0.7:
        owner.auto_load_glossary_for_file = calls.method("auto_load_glossary_for_file")
    if rng.random() < 0.5:
        owner.editor_file_entry = TextShim("")
        owner.auto_loaded_glossary_path = rng.choice([None, str(glossary)])
    return owner, calls


def _owner_state(owner):
    out = {}
    for key, value in vars(owner).items():
        if callable(value) or key == "config":
            continue
        out[key] = getattr(value, "_checked", getattr(value, "_text", getattr(value, "_index", value)))
    return out


def test_glossary_shortcut_handler_matches_the_legacy_nested_function(tmp_path):
    from owner_state import ConfigStateMixin

    make_legacy = _legacy_shortcut_handler_factory()
    for seed in range(400):
        results = []
        for side in ("legacy", "new"):
            rng = random.Random(seed)
            owner, calls = _random_shortcut_owner(rng, tmp_path)
            index = rng.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, -1, None])
            error = None
            try:
                if side == "legacy":
                    make_legacy(owner)(index)
                else:
                    ConfigStateMixin._on_auto_glossary_shortcut_changed(owner, index)
            except Exception as exc:  # compared by type
                error = type(exc).__name__
            results.append((copy.deepcopy(owner.config), _owner_state(owner), list(calls), error))
        assert results[0] == results[1], seed
