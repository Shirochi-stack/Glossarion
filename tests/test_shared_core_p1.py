"""Shared GUI-free core, phase P1 (Glossarion mobile rewrite, milestone U1).

Code moved out of translator_gui / config_backup / metadata_batch_translator /
ollama_settings_dialog / other_settings into GUI-free modules (app_paths,
config_store, prompt_defaults, metadata_defaults, ollama_settings, key_pools,
output_naming), ai_hunter_enhanced defaults lifted to module level, and the
GLOSSARION_LIBRARY_DIR seam in epub_library.get_library_dir.

What is checked:
* import hygiene: every new module imports with PySide6 blocked and never pulls
  in translator_gui or dpi_setup; all touched files parse as Python 3.10;
* the moves are verbatim: AST equality with ``git show BASE_SHA:src/...``
  (``self.config`` -> ``config`` where a method body became a function);
* behaviour: differential runs of the legacy code (exec'd from ``git show``)
  against the moved functions, plus unit tests of the new config_store API;
* desktop parity: the working-tree TranslatorGUI methods, frozen with the U0
  oracle (tests/parity) and driven by the same fakes, reproduce the goldens
  captured at BASE_SHA for all 12 scenarios (boot/config load, startup env,
  translation env incl. key pools, save_config, compile runners), and the
  config.json written by the startup save_config is identical.

Run (repository root)::

    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic \
        tests/test_shared_core_p1.py
"""

from __future__ import annotations

import ast
import copy
import io
import json
import os
import random
import subprocess
import sys
import textwrap
import time
import types
from contextlib import redirect_stdout
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
TESTS = REPO_ROOT / "tests"
for _p in (str(TESTS), str(SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

#: Parent commit of the P1 extraction (the merged U0 commit); goldens are captured here.
BASE_SHA = "4c825d81f381be3e99c66032d914d3a676a0eae7"

NEW_MODULES = (
    "app_paths",
    "config_store",
    "prompt_defaults",
    "metadata_defaults",
    "ollama_settings",
    "key_pools",
    "output_naming",
)
TOUCHED_FILES = NEW_MODULES + (
    "translator_gui",
    "config_backup",
    "metadata_batch_translator",
    "ollama_settings_dialog",
    "other_settings",
    "ai_hunter_enhanced",
    "epub_library",
)


# ---------------------------------------------------------------------------
# helpers: legacy sources from git
# ---------------------------------------------------------------------------

_LEGACY_CACHE: dict = {}


def legacy_source(relpath: str) -> str:
    """``git show BASE_SHA:<relpath>`` (utf-8-sig, LF)."""
    if relpath not in _LEGACY_CACHE:
        try:
            data = subprocess.run(
                ["git", "show", f"{BASE_SHA}:{relpath}"], cwd=str(REPO_ROOT),
                check=True, capture_output=True,
            ).stdout
        except Exception as exc:  # pragma: no cover - environment dependent
            pytest.skip(f"git show {BASE_SHA}:{relpath} unavailable: {exc}")
        _LEGACY_CACHE[relpath] = data.decode("utf-8-sig").replace("\r\n", "\n")
    return _LEGACY_CACHE[relpath]


_TREE_CACHE: dict = {}


def legacy_tree(relpath: str) -> ast.Module:
    """Parsed BASE_SHA source (cached; callers must not mutate it, see ``renamed``)."""
    key = ("legacy", relpath)
    if key not in _TREE_CACHE:
        _TREE_CACHE[key] = ast.parse(legacy_source(relpath))
    return _TREE_CACHE[key]


def current_tree(module: str) -> ast.Module:
    """Parsed working-tree source of src/<module>.py (cached)."""
    key = ("current", module)
    if key not in _TREE_CACHE:
        _TREE_CACHE[key] = ast.parse((SRC / f"{module}.py").read_text(encoding="utf-8-sig"))
    return _TREE_CACHE[key]


def top_function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"function {name} not found")


def top_assign(tree: ast.Module, name: str) -> ast.Assign:
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return node
    raise AssertionError(f"assignment {name} not found")


def class_method(tree: ast.Module, cls: str, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == cls:
            found = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == name]
            if found:
                return found[-1]
    raise AssertionError(f"{cls}.{name} not found")


def segment(text: str, node) -> str:
    lines = text.split("\n")
    return textwrap.dedent("\n".join(lines[node.lineno - 1:node.end_lineno]))


def exec_legacy_function(relpath: str, node, namespace: dict, name: str = None):
    src = segment(legacy_source(relpath), node)
    exec(compile(src, f"<legacy {relpath}:{node.lineno}>", "exec"), namespace)
    return namespace[name or node.name]


def dump(node) -> str:
    return ast.dump(node, include_attributes=False)


class _Rename(ast.NodeTransformer):
    """``self.config`` / ``self.gui.config`` -> ``config`` (method body -> function body)."""

    def visit_Attribute(self, node):
        self.generic_visit(node)
        text = ast.unparse(node)
        if text in ("self.config", "self.gui.config"):
            return ast.copy_location(ast.Name(id="config", ctx=node.ctx), node)
        return node


def renamed(node):
    return _Rename().visit(copy.deepcopy(node))


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
        capture_output=True, text=True, encoding="utf-8", env=env, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().splitlines()[-1] == "LEAKED=", proc.stdout


@pytest.mark.parametrize("module", TOUCHED_FILES)
def test_touched_files_parse_as_python_310(module):
    ast.parse((SRC / f"{module}.py").read_text(encoding="utf-8-sig"), feature_version=(3, 10))


def test_ai_hunter_module_imports_without_qt():
    probe = (
        "import sys; sys.modules['PySide6'] = None; sys.path.insert(0, %r); "
        "import ai_hunter_enhanced as a; "
        "assert not a.HAS_GUI; assert a.default_ai_hunter_config()['enabled'] is True; print('ok')"
    ) % str(SRC)
    proc = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0 and proc.stdout.strip() == "ok", proc.stderr


def test_config_backup_no_longer_imports_translator_gui():
    tree = current_tree("config_backup")
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.module != "translator_gui"
        elif isinstance(node, ast.Import):
            assert all(alias.name != "translator_gui" for alias in node.names)


# ---------------------------------------------------------------------------
# verbatim moves (AST equality with BASE_SHA)
# ---------------------------------------------------------------------------

def test_app_paths_is_the_verbatim_translator_gui_block():
    old = legacy_tree("src/translator_gui.py")
    new = current_tree("app_paths")
    for name in ("_atomic_json_write", "_get_app_dir"):
        assert dump(top_function(new, name)) == dump(top_function(old, name)), name

    def app_dir_block(tree):
        body = tree.body
        start = next(i for i, n in enumerate(body) if isinstance(n, ast.If)
                     and ast.unparse(n.test) == "getattr(sys, 'frozen', False) and hasattr(sys, 'executable')")
        end = next(i for i, n in enumerate(body) if isinstance(n, ast.Assign)
                   and ast.unparse(n) == "CONFIG_FILE = os.path.join(_APP_DIR, 'config.json')")
        return [dump(n) for n in body[start:end + 1]]

    assert app_dir_block(new) == app_dir_block(old)
    # translator_gui no longer defines them; it re-imports the names
    current = current_tree("translator_gui")
    defined = {n.name for n in current.body if isinstance(n, ast.FunctionDef)}
    assert not {"_atomic_json_write", "_get_app_dir"} & defined
    imports = [ast.unparse(n) for n in current.body if isinstance(n, ast.ImportFrom) and n.module == "app_paths"]
    assert imports == ["from app_paths import CONFIG_FILE, _APP_DIR, _atomic_json_write, _get_app_dir"]


def _legacy_key_pool_tries():
    method = class_method(legacy_tree("src/translator_gui.py"), "TranslatorGUI", "_get_environment_variables")
    tries = [s for s in method.body if isinstance(s, ast.Try)]
    env_try = next(t for t in tries if "os.environ['USE_MULTI_KEYS'] = '1'" in ast.unparse(t))
    mem_try = next(t for t in tries if "set_in_memory_tts_keys" in ast.unparse(t))
    return env_try, mem_try


def test_key_pools_is_the_verbatim_get_environment_variables_section():
    import key_pools  # noqa: F401  (import check)

    env_try, mem_try = _legacy_key_pool_tries()
    new = current_tree("key_pools")
    export_fn = top_function(new, "export_key_pool_env")
    apply_fn = top_function(new, "apply_key_pools_to_runtime")
    new_env_try = [s for s in export_fn.body if isinstance(s, ast.Try)]
    new_mem_try = [s for s in apply_fn.body if isinstance(s, ast.Try)]
    assert len(new_env_try) == len(new_mem_try) == 1
    assert dump(new_env_try[0]) == dump(renamed(env_try))
    assert dump(new_mem_try[0]) == dump(renamed(mem_try))
    # the desktop builder now delegates (and no longer contains the section)
    method = class_method(current_tree("translator_gui"), "TranslatorGUI", "_get_environment_variables")
    text = ast.unparse(method)
    assert "apply_key_pools_to_runtime(self.config)" in text
    assert "set_in_memory_multi_keys" not in text and "USE_MULTI_KEYS" not in text


@pytest.mark.parametrize("name", [
    "is_ollamapull_route", "ollamapull_model_name", "normalize_ollama_settings", "ollama_settings_json",
    "parse_option", "_json_object", "_reported_parameter_defaults",
])
def test_ollama_settings_functions_are_verbatim(name):
    assert dump(top_function(current_tree("ollama_settings"), name)) == dump(
        top_function(legacy_tree("src/ollama_settings_dialog.py"), name))


@pytest.mark.parametrize("name", [
    "OLLAMAPULL_PREFIX", "OPTION_GROUPS", "COMMON_OPTION_KEYS", "RESERVED_REQUEST_KEYS", "SLIDER_OPTIONS",
])
def test_ollama_settings_constants_are_verbatim_and_reexported(name):
    assert dump(top_assign(current_tree("ollama_settings"), name)) == dump(
        top_assign(legacy_tree("src/ollama_settings_dialog.py"), name))
    reexports = [n for n in current_tree("ollama_settings_dialog").body
                 if isinstance(n, ast.ImportFrom) and n.module == "ollama_settings"]
    assert len(reexports) == 1 and name in {a.name for a in reexports[0].names}


@pytest.mark.parametrize("name", [
    "_library_origins_raw_sources_for_stem", "_library_raw_inputs_for_stem", "_rename_output_files_for_retain",
])
def test_output_naming_functions_are_verbatim_and_reexported(name):
    assert dump(top_function(current_tree("output_naming"), name)) == dump(
        top_function(legacy_tree("src/other_settings.py"), name))
    reexports = [n for n in current_tree("other_settings").body
                 if isinstance(n, ast.ImportFrom) and n.module == "output_naming"]
    assert len(reexports) == 1 and name in {a.name for a in reexports[0].names}


def test_metadata_defaults_is_the_verbatim_initialize_default_prompts_body():
    old = class_method(legacy_tree("src/metadata_batch_translator.py"),
                       "MetadataBatchTranslatorUI", "_initialize_default_prompts")
    new = top_function(current_tree("metadata_defaults"), "ensure_metadata_prompt_defaults")
    old_body = [dump(renamed(s)) for s in old.body[1:]]  # skip docstring
    new_body = [dump(s) for s in new.body[2:-1]]  # skip docstring + count_before ... return
    assert new_body == old_body
    caller = class_method(current_tree("metadata_batch_translator"),
                          "MetadataBatchTranslatorUI", "_initialize_default_prompts")
    assert "ensure_metadata_prompt_defaults(self.gui.config)" in ast.unparse(caller)


_PROMPT_CONSTANTS = {
    # (method, attribute) -> constant
    ("__init__", "default_translation_chunk_prompt"): "DEFAULT_TRANSLATION_CHUNK_PROMPT",
    ("__init__", "default_image_chunk_prompt"): "DEFAULT_IMAGE_CHUNK_PROMPT",
    ("__init__", "default_vision_ocr_prompt"): "DEFAULT_VISION_OCR_PROMPT",
    ("__init__", "default_vision_ocr_user_prompt"): "DEFAULT_VISION_OCR_USER_PROMPT",
    ("__init__", "default_vision_ocr_combined_context_prompt"): "DEFAULT_VISION_OCR_COMBINED_CONTEXT_PROMPT",
    ("__init__", "default_vision_ocr_translation_user_prompt"): "DEFAULT_VISION_OCR_TRANSLATION_USER_PROMPT",
    ("_init_default_prompts", "default_rolling_summary_system_prompt"): "DEFAULT_ROLLING_SUMMARY_SYSTEM_PROMPT",
    ("_init_default_prompts", "default_assistant_prompt"): "DEFAULT_ASSISTANT_PROMPT",
    ("_init_default_prompts", "default_rolling_summary_user_prompt"): "DEFAULT_ROLLING_SUMMARY_USER_PROMPT",
}


@pytest.mark.parametrize("key", list(_PROMPT_CONSTANTS), ids=lambda k: k[1])
def test_prompt_default_values_are_unchanged(key):
    import prompt_defaults

    method_name, attr = key
    method = class_method(legacy_tree("src/translator_gui.py"), "TranslatorGUI", method_name)
    assigns = [s for s in method.body if isinstance(s, ast.Assign)
               and any(ast.unparse(t) == f"self.{attr}" for t in s.targets)]
    assert len(assigns) == 1
    legacy_value = ast.literal_eval(assigns[0].value)
    assert getattr(prompt_defaults, _PROMPT_CONSTANTS[key]) == legacy_value
    # translator_gui assigns the constant to the same attribute in the same method
    new_method = class_method(current_tree("translator_gui"), "TranslatorGUI", method_name)
    new_assigns = [ast.unparse(s) for s in new_method.body if isinstance(s, ast.Assign)
                   and any(ast.unparse(t) == f"self.{attr}" for t in s.targets)]
    assert new_assigns == [f"self.{attr} = {_PROMPT_CONSTANTS[key]}"]


# ---------------------------------------------------------------------------
# behaviour: key pools (differential, >= 500 seeded states)
# ---------------------------------------------------------------------------

_POOL_FLAGS = (
    "use_multi_api_keys", "use_fallback_keys", "fallback_key_shuffle", "use_glossary_keys",
    "use_glossary_refinement_keys", "use_metadata_keys", "use_qa_scan_keys",
    "use_ai_truncation_detection_keys", "use_rolling_summary_keys", "use_truncation_retry_keys",
    "use_inpainter_keys", "use_tts_keys", "force_key_rotation",
)
_POOL_LISTS = (
    "multi_api_keys", "glossary_keys", "glossary_refinement_keys", "metadata_keys", "qa_scan_keys",
    "ai_truncation_detection_keys", "rolling_summary_keys", "truncation_retry_keys", "inpainter_keys",
    "tts_keys",
)
_POOL_ENV = (
    "USE_MULTI_KEYS", "USE_FALLBACK_KEYS", "FALLBACK_KEY_SHUFFLE", "USE_GLOSSARY_KEYS",
    "USE_GLOSSARY_REFINEMENT_KEYS", "GLOSSARY_REFINEMENT_API_KEYS", "USE_METADATA_KEYS", "METADATA_API_KEYS",
    "USE_VISION_KEYS", "USE_QA_SCAN_KEYS", "VISION_API_KEYS", "QA_SCAN_API_KEYS",
    "USE_AI_TRUNCATION_DETECTION_KEYS", "AI_TRUNCATION_DETECTION_API_KEYS", "USE_ROLLING_SUMMARY_KEYS",
    "ROLLING_SUMMARY_API_KEYS", "USE_TRUNCATION_RETRY_KEYS", "TRUNCATION_RETRY_API_KEYS",
)
_SCALARS = (True, False, "0", "1", "", "abc", 0, 5, -1, None)


def _random_pool_config(rng: random.Random) -> dict:
    cfg = {}
    for key in _POOL_FLAGS:
        if rng.random() < 0.75:
            cfg[key] = rng.choice(_SCALARS)
    for key in _POOL_LISTS:
        roll = rng.random()
        if roll < 0.25:
            continue
        if roll < 0.85:
            cfg[key] = [{"api_key": f"{key}-{i}", "model": "m"} for i in range(rng.randint(0, 3))]
        elif roll < 0.95:
            cfg[key] = rng.choice(_SCALARS)
        else:
            cfg[key] = [{1, 2}]  # not JSON serialisable -> exercises the except path
    if rng.random() < 0.5:
        cfg["rotation_frequency"] = rng.choice((1, 3, "2", None))
    return cfg


def _recording_client_module(calls: list) -> types.ModuleType:
    module = types.ModuleType("unified_api_client")

    class UnifiedClient:
        pass

    def _make(name):
        def method(cls, *args, **kwargs):
            calls.append((name, copy.deepcopy(args), dict(sorted(kwargs.items()))))
            return True
        return classmethod(method)

    for pool in ("multi", "glossary", "glossary_refinement", "vision", "rolling_summary",
                 "truncation_retry", "inpainter", "tts"):
        setattr(UnifiedClient, f"set_in_memory_{pool}_keys", _make(f"set_in_memory_{pool}_keys"))
        setattr(UnifiedClient, f"clear_in_memory_{pool}_keys", _make(f"clear_in_memory_{pool}_keys"))
    module.UnifiedClient = UnifiedClient
    return module


def _run_pools(fn, config, monkeypatch):
    calls: list = []
    monkeypatch.setitem(sys.modules, "unified_api_client", _recording_client_module(calls))
    for name in _POOL_ENV:
        monkeypatch.delenv(name, raising=False)
    error = None
    try:
        fn(config)
    except Exception as exc:  # pragma: no cover - both sides swallow everything
        error = type(exc).__name__
    env = {name: os.environ.get(name) for name in _POOL_ENV}
    return env, calls, error, config


def test_key_pools_match_legacy_on_seeded_states(monkeypatch):
    import key_pools

    env_try, mem_try = _legacy_key_pool_tries()
    text = legacy_source("src/translator_gui.py")
    lines = text.split("\n")
    body = textwrap.dedent("\n".join(lines[env_try.lineno - 1:env_try.end_lineno]
                                     + lines[mem_try.lineno - 1:mem_try.end_lineno]))
    ns = {"os": os, "json": json}
    exec("def legacy_key_pools(self):\n" + textwrap.indent(body, "    "), ns)
    legacy = ns["legacy_key_pools"]

    rng = random.Random(20261005)
    for i in range(600):
        cfg = _random_pool_config(rng)
        old = _run_pools(lambda c: legacy(types.SimpleNamespace(config=c)), copy.deepcopy(cfg), monkeypatch)
        new = _run_pools(key_pools.apply_key_pools_to_runtime, copy.deepcopy(cfg), monkeypatch)
        assert new == old, (i, cfg)


def test_key_pools_export_env_flag(monkeypatch):
    import key_pools

    cfg = {"use_multi_api_keys": True, "multi_api_keys": [{"api_key": "k", "model": "m"}],
           "use_tts_keys": True, "tts_keys": [{"api_key": "t", "model": "tts"}]}
    env, calls, error, _ = _run_pools(lambda c: key_pools.apply_key_pools_to_runtime(c, export_env=False),
                                      cfg, monkeypatch)
    assert error is None and all(v is None for v in env.values())
    names = [c[0] for c in calls]
    assert names[0] == "set_in_memory_multi_keys" and "set_in_memory_tts_keys" in names
    assert calls[0][2] == {"force_rotation": True, "rotation_frequency": 1}
    env, _calls, _e, _ = _run_pools(key_pools.export_key_pool_env, {"use_qa_scan_keys": 1}, monkeypatch)
    assert env["USE_VISION_KEYS"] == env["USE_QA_SCAN_KEYS"] == "1"
    assert env["VISION_API_KEYS"] == env["QA_SCAN_API_KEYS"] == "[]"
    assert env["USE_MULTI_KEYS"] == "0"


# ---------------------------------------------------------------------------
# behaviour: metadata defaults, ollama, ai hunter, prompt sanitizer
# ---------------------------------------------------------------------------

_METADATA_KEYS = ("book_title_system_prompt", "book_title_prompt", "batch_header_system_prompt",
                  "batch_header_prompt", "metadata_batch_prompt", "metadata_field_prompts")


def test_metadata_defaults_match_legacy():
    import metadata_defaults

    old = class_method(legacy_tree("src/metadata_batch_translator.py"),
                       "MetadataBatchTranslatorUI", "_initialize_default_prompts")
    legacy = exec_legacy_function("src/metadata_batch_translator.py", old, {})
    rng = random.Random(7)
    for _ in range(200):
        cfg = {k: f"user-{k}" for k in _METADATA_KEYS if rng.random() < 0.5}
        if rng.random() < 0.3:
            cfg["unrelated"] = 1
        old_cfg = copy.deepcopy(cfg)
        legacy(types.SimpleNamespace(gui=types.SimpleNamespace(config=old_cfg)))
        new_cfg = copy.deepcopy(cfg)
        added = metadata_defaults.ensure_metadata_prompt_defaults(new_cfg)
        assert new_cfg == old_cfg and list(new_cfg) == list(old_cfg)
        assert added == any(k not in cfg for k in _METADATA_KEYS)
    assert metadata_defaults.ensure_metadata_prompt_defaults({k: "x" for k in _METADATA_KEYS}) is False


def test_ollama_settings_match_legacy():
    pytest.importorskip("PySide6")
    import ollama_settings
    import ollama_settings_dialog

    tree = legacy_tree("src/ollama_settings_dialog.py")
    ns = {"copy": copy, "json": json}
    exec_legacy_function("src/ollama_settings_dialog.py", top_function(tree, "normalize_ollama_settings"), ns)
    legacy_json = exec_legacy_function("src/ollama_settings_dialog.py", top_function(tree, "ollama_settings_json"), ns)
    samples = [None, {}, [], "x", {"ollama_settings": None}, {"ollama_settings": {"auto_update": False}},
               {"ollama_settings": {"models": "bad", "extra": [1, 2]}},
               {"ollama_settings": {"models": {"llama3": {"options": {"num_ctx": 4096}}}, "auto_update": True}},
               {"ollama_settings": {"künftig": "ü", "models": {}}}]
    def outcome(fn, arg):
        try:
            return ("ok", fn(copy.deepcopy(arg)))
        except Exception as exc:
            return ("error", type(exc).__name__)

    for sample in samples:
        assert outcome(ollama_settings.ollama_settings_json, sample) == outcome(legacy_json, sample)
        value = sample.get("ollama_settings") if isinstance(sample, dict) else sample
        assert outcome(ollama_settings.normalize_ollama_settings, value) == outcome(
            ns["normalize_ollama_settings"], value)
    # the dialog module re-exports the very same objects
    for name in ("normalize_ollama_settings", "ollama_settings_json", "parse_option", "OPTION_GROUPS",
                 "SLIDER_OPTIONS", "is_ollamapull_route", "_reported_parameter_defaults"):
        assert getattr(ollama_settings_dialog, name) is getattr(ollama_settings, name)


def test_ollama_parse_option_contract():
    import ollama_settings

    assert ollama_settings.parse_option(" 8 ", "int") == 8
    assert ollama_settings.parse_option("0.5", "float") == 0.5
    assert ollama_settings.parse_option("Yes", "bool") is True
    assert ollama_settings.parse_option('["a"]', "array") == ["a"]
    for text, kind in (("nan", "float"), ("maybe", "bool"), ("[1]", "array"), ("1", "weird")):
        with pytest.raises(ValueError):
            ollama_settings.parse_option(text, kind)
    with pytest.raises(ValueError):
        ollama_settings._json_object("[]", "Extra")


def _legacy_ai_hunter_default_expr():
    tree = legacy_tree("src/ai_hunter_enhanced.py")
    init = class_method(tree, "AIHunterConfigGUI", "__init__")
    node = next(s for s in init.body if isinstance(s, ast.Assign)
                and ast.unparse(s.targets[0]) == "self.default_ai_hunter")
    return compile(ast.Expression(node.value), "<legacy ai hunter default>", "eval")


@pytest.mark.parametrize("cpus", [1, 8, 16, None])
def test_ai_hunter_defaults_match_legacy(monkeypatch, cpus):
    import ai_hunter_enhanced

    monkeypatch.setattr(os, "cpu_count", lambda: cpus)
    legacy = eval(_legacy_ai_hunter_default_expr(), {"os": os})
    first = ai_hunter_enhanced.default_ai_hunter_config()
    assert first == legacy
    # a fresh structure per call (callers mutate nested dicts)
    first["thresholds"]["text"] = 1
    assert ai_hunter_enhanced.default_ai_hunter_config()["thresholds"]["text"] == legacy["thresholds"]["text"]


def test_ai_hunter_merge_matches_legacy_and_gui_uses_it():
    import ai_hunter_enhanced

    tree = legacy_tree("src/ai_hunter_enhanced.py")
    legacy_merge = exec_legacy_function("src/ai_hunter_enhanced.py",
                                        class_method(tree, "AIHunterConfigGUI", "_merge_configs"), {})
    holder = types.SimpleNamespace()
    holder._merge_configs = lambda d, e: legacy_merge(holder, d, e)
    rng = random.Random(3)

    def rand_tree(depth=0):
        out = {}
        for key in rng.sample(["thresholds", "weights", "enabled", "text", "x", "languages"], rng.randint(0, 4)):
            out[key] = rand_tree(depth + 1) if depth < 2 and rng.random() < 0.5 else rng.choice([1, "a", None, [1]])
        return out

    for _ in range(300):
        default, existing = rand_tree(), rand_tree()
        assert ai_hunter_enhanced.merge_ai_hunter_config(copy.deepcopy(existing), copy.deepcopy(default)) == \
            legacy_merge(holder, copy.deepcopy(default), copy.deepcopy(existing))
    cfg = {"ai_hunter_config": {"thresholds": {"text": 50}, "custom": True}}
    gui = ai_hunter_enhanced.AIHunterConfigGUI(None, cfg)
    assert cfg["ai_hunter_config"]["thresholds"]["text"] == 50 and cfg["ai_hunter_config"]["custom"] is True
    assert cfg["ai_hunter_config"]["thresholds"]["exact"] == 90
    assert gui.default_ai_hunter == ai_hunter_enhanced.default_ai_hunter_config()


def test_prompt_sanitizer_core():
    from prompt_defaults import sanitize_prompt_profiles

    broken = "Keep original Korean quotation marks (, ' ', 「」, 『』) as-is"
    assert sanitize_prompt_profiles({}) is None
    assert sanitize_prompt_profiles({"prompt_profiles": {"a": broken}, "sanitization_korean_quotes_fixed": True}) is None
    cfg = {"prompt_profiles": {"a": broken, "b": {"prompt": broken}, "c": "fine"}}
    buf = io.StringIO()
    with redirect_stdout(buf):
        assert sanitize_prompt_profiles(cfg) is True
    assert cfg["prompt_profiles"]["a"] == broken.replace("(, ", '(" ", ')
    assert cfg["prompt_profiles"]["b"]["prompt"] == cfg["prompt_profiles"]["a"]
    assert cfg["sanitization_korean_quotes_fixed"] is True
    assert buf.getvalue().count("[Sanitizer] Fixed malformed Korean quotes") == 2
    clean = {"prompt_profiles": {"c": "fine"}}
    assert sanitize_prompt_profiles(clean) is False and clean["sanitization_korean_quotes_fixed"] is True


@pytest.mark.parametrize("case", ["no_profiles", "flagged", "broken_str", "broken_dict", "clean"])
def test_sanitize_config_prompts_matches_legacy(tmp_path, case):
    import api_key_encryption
    import app_paths
    import prompt_defaults

    broken = "Keep original Korean quotation marks (, ' ', 「」, 『』) as-is"
    configs = {
        "no_profiles": {"api_key": "sk-x"},
        "flagged": {"prompt_profiles": {"a": broken}, "sanitization_korean_quotes_fixed": True},
        "broken_str": {"prompt_profiles": {"a": broken, "b": "ok"}, "api_key": "sk-y"},
        "broken_dict": {"prompt_profiles": {"a": {"prompt": broken}}},
        "clean": {"prompt_profiles": {"a": "ok"}},
    }
    legacy_node = class_method(legacy_tree("src/translator_gui.py"), "TranslatorGUI", "_sanitize_config_prompts")
    new_node = class_method(current_tree("translator_gui"), "TranslatorGUI", "_sanitize_config_prompts")
    results = []
    for label, node, extra in (("old", legacy_node, {}),
                               ("new", new_node, {"sanitize_prompt_profiles": prompt_defaults.sanitize_prompt_profiles})):
        config_file = tmp_path / label / "config.json"
        config_file.parent.mkdir()
        ns = {"_atomic_json_write": app_paths._atomic_json_write, "CONFIG_FILE": str(config_file), **extra}
        src = segment(legacy_source("src/translator_gui.py") if label == "old"
                      else (SRC / "translator_gui.py").read_text(encoding="utf-8-sig").replace("\r\n", "\n"), node)
        exec(src, ns)
        owner = types.SimpleNamespace(config=copy.deepcopy(configs[case]))
        buf = io.StringIO()
        with redirect_stdout(buf):
            ns["_sanitize_config_prompts"](owner)
        written = None
        if config_file.exists():
            written = api_key_encryption.decrypt_config(json.loads(config_file.read_text(encoding="utf-8")))
        results.append((owner.config, written, buf.getvalue()))
    assert results[0] == results[1]


# ---------------------------------------------------------------------------
# behaviour: output naming (retain source extension rename)
# ---------------------------------------------------------------------------

_OPF = """<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0">
  <manifest>
    <item id="c1" href="Text/chapter001.xhtml" media-type="application/xhtml+xml"/>
    <item id="c2" href="chapter002.html" media-type="application/xhtml+xml"/>
    <item id="c3" href="chapter003.htm" media-type="text/html"/>
    <item id="css" href="style.css" media-type="text/css"/>
  </manifest>
</package>
"""


def _make_output_dir(root: Path) -> Path:
    out = root / "Book"
    out.mkdir(parents=True)
    (out / "content.opf").write_text(_OPF, encoding="utf-8")
    for name in ("response_chapter001.html", "chapter002.html.html", "response_chapter003.htm.xhtml",
                 "unrelated.html", "notes.txt"):
        (out / name).write_text(name, encoding="utf-8")
    progress = {"chapters": {
        "1": {"output_file": "response_chapter001.html"},
        "2": {"output_file": "chapter002.html.html"},
        "3": {"output_file": "response_chapter003.htm.xhtml"},
    }}
    (out / "translation_progress.json").write_text(json.dumps(progress), encoding="utf-8")
    return out


def _tree_state(folder: Path):
    files = {}
    for path in sorted(folder.iterdir()):
        files[path.name] = path.read_text(encoding="utf-8")
    return files


@pytest.mark.parametrize("retain", [True, False])
def test_rename_output_files_matches_legacy(tmp_path, monkeypatch, retain):
    import epub_package
    import output_naming

    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    for name in ("EPUB_PATH", "OUTPUT_DIR", "OUTPUT_DIRECTORY"):
        monkeypatch.delenv(name, raising=False)
    tree = legacy_tree("src/other_settings.py")
    ns = {"os": os, "json": json, "find_opf_path": epub_package.find_opf_path,
          "find_epub_opf_member": epub_package.find_epub_opf_member}
    for name in ("_library_origins_raw_sources_for_stem", "_library_raw_inputs_for_stem",
                 "_rename_output_files_for_retain"):
        exec_legacy_function("src/other_settings.py", top_function(tree, name), ns)
    outcomes = []
    for label, fn in (("old", ns["_rename_output_files_for_retain"]),
                      ("new", output_naming._rename_output_files_for_retain)):
        folder = _make_output_dir(tmp_path / label)
        logs = []
        gui = types.SimpleNamespace(append_log=logs.append, config={})
        result = fn(gui, retain, output_dir=str(folder))
        outcomes.append((result, logs, _tree_state(folder)))
        # a second call is idempotent
        outcomes.append((fn(gui, retain, output_dir=str(folder)), _tree_state(folder)))
    assert outcomes[0] == outcomes[2] and outcomes[1] == outcomes[3]
    assert outcomes[0][0][0] == "renamed"
    missing = types.SimpleNamespace(config={})
    assert output_naming._rename_output_files_for_retain(missing, retain, str(tmp_path / "nope")) == ("no_opf",)


# ---------------------------------------------------------------------------
# config_store
# ---------------------------------------------------------------------------

@pytest.fixture
def config_path(tmp_path, monkeypatch):
    import app_paths

    path = tmp_path / "cfg" / "config.json"
    path.parent.mkdir()
    monkeypatch.setattr(app_paths, "CONFIG_FILE", str(path))
    return path


def test_load_config_reads_and_decrypts_only(config_path):
    import api_key_encryption
    import config_store

    with pytest.raises(FileNotFoundError):
        config_store.load_config()
    plain = {"api_key": "sk-secret", "model": "m", "prompt_profiles": {"a": "Korean quotation marks (, ' ')"}}
    config_path.write_text(json.dumps(api_key_encryption.encrypt_config(copy.deepcopy(plain))), encoding="utf-8")
    before = config_path.read_bytes()
    assert config_store.load_config() == plain  # no sanitize / defaults
    assert config_store.load_config(str(config_path)) == plain
    assert config_store.load_config(decrypt=False) == json.loads(before)  # file content as-is
    assert config_path.read_bytes() == before  # never writes
    config_path.write_text("{broken", encoding="utf-8")
    with pytest.raises(ValueError):
        config_store.load_config()


def _legacy_final_write():
    method = class_method(legacy_tree("src/translator_gui.py"), "TranslatorGUI", "save_config")
    text = legacy_source("src/translator_gui.py").split("\n")
    start = next(i for i in range(method.lineno - 1, method.end_lineno)
                 if text[i].strip() == "# --- 5. Final Write to File ---")
    end = next(i for i in range(start, method.end_lineno)
               if text[i].strip() == "_atomic_json_write(CONFIG_FILE, encrypted_config)")
    body = textwrap.dedent("\n".join(text[start:end + 1]))
    return "def legacy_final_write(self, CONFIG_FILE):\n" + textwrap.indent(body, "    ")


@pytest.mark.parametrize("cfg", [
    {"api_key": "sk-1", "model": "m"},
    {"api_key": "", "google_cloud_credentials": "C:/creds/vertex.json", "multi_api_keys": [{"api_key": "k1"}]},
    {"replicate_api_key": "r8", "nested": {"x": [1, 2]}, "unicode": "번역"},
])
def test_save_config_file_matches_legacy_final_write(tmp_path, cfg):
    import api_key_encryption
    import app_paths
    import config_store

    ns = {"encrypt_config": api_key_encryption.encrypt_config, "json": json,
          "_atomic_json_write": app_paths._atomic_json_write}
    exec(_legacy_final_write(), ns)
    old_path, new_path = tmp_path / "old.json", tmp_path / "new.json"
    ns["legacy_final_write"](types.SimpleNamespace(config=copy.deepcopy(cfg)), str(old_path))
    config_store.save_config_file(copy.deepcopy(cfg), str(new_path), backup=False)
    old, new = (json.loads(p.read_text(encoding="utf-8")) for p in (old_path, new_path))
    assert list(old) == list(new)
    enc = lambda d: sorted(k for k, v in d.items() if isinstance(v, str) and v.startswith("ENC:"))  # noqa: E731
    assert enc(old) == enc(new)
    assert api_key_encryption.decrypt_config(old) == api_key_encryption.decrypt_config(new) == cfg
    assert not list(tmp_path.glob("*.tmp"))
    assert not (tmp_path / "config_backups").exists()


def test_save_config_file_backup_and_default_path(config_path):
    import config_store

    config_store.save_config_file({"model": "first"}, backup=True)  # nothing to back up yet
    assert not (config_path.parent / "config_backups").exists()
    config_store.save_config_file({"model": "second"})
    backups = config_store.list_config_backups()
    assert len(backups) == 1 and json.loads(Path(backups[0]["path"]).read_text(encoding="utf-8")) == {"model": "first"}
    assert json.loads(config_path.read_text(encoding="utf-8")) == {"model": "second"}


def test_backup_retention_latest_and_restore(config_path, capsys):
    import config_store

    assert config_store.backup_config_file() is None  # no config yet
    assert config_store.latest_backup() is None and config_store.list_config_backups() == []
    assert config_store.restore_latest_backup() is None
    config_path.write_text('{"v": 1}', encoding="utf-8")
    backup_dir = Path(config_store.config_backup_dir())
    assert backup_dir == config_path.parent / "config_backups"
    old = backup_dir / "config_20200101_000000.json.bak"
    backup_dir.mkdir()
    old.write_text('{"v": 0}', encoding="utf-8")
    stale = time.time() - 80 * 3600
    os.utime(old, (stale, stale))
    newer = backup_dir / "config_20200102_000000.json.bak"
    newer.write_text('{"v": "newer"}', encoding="utf-8")
    os.utime(newer, (time.time() - 3600, time.time() - 3600))
    created = config_store.backup_config_file()
    assert created and Path(created).read_text(encoding="utf-8") == '{"v": 1}'
    assert not old.exists() and newer.exists()  # 72 h retention
    names = [b["name"] for b in config_store.list_config_backups()]
    assert names[0] == Path(created).name and names[-1] == newer.name
    assert config_store.latest_backup() == created
    config_path.write_text('{"v": 2}', encoding="utf-8")
    assert config_store.restore_latest_backup() == created
    assert config_path.read_text(encoding="utf-8") == '{"v": 1}'
    # failures are printed, never raised
    assert config_store.backup_config_file(str(config_path), retention_hours="x") is None
    assert "Could not create config backup" in capsys.readouterr().out


def test_restore_config_backup_file_validates_and_replaces_atomically(config_path, tmp_path):
    import config_store

    config_path.write_bytes(b'{"current": true}')
    bad = tmp_path / "bad.bak"
    for payload in (b"{", b"[]"):
        bad.write_bytes(payload)
        with pytest.raises(ValueError):
            config_store.restore_config_backup_file(None, str(bad))
        assert config_path.read_bytes() == b'{"current": true}'
    good = tmp_path / "good.bak"
    good.write_bytes(b'{"restored": 1}')
    order = []
    config_store.restore_config_backup_file(
        str(config_path), str(good), safety_backup=lambda: order.append(config_path.read_bytes()))
    assert order == [b'{"current": true}'] and config_path.read_bytes() == b'{"restored": 1}'
    # default safety backup goes through backup_config_file
    good.write_bytes(b'{"restored": 2}')
    config_store.restore_config_backup_file(None, str(good))
    assert any(b["path"] and Path(b["path"]).read_bytes() == b'{"restored": 1}'
               for b in config_store.list_config_backups())
    assert not list(config_path.parent.glob(".config_restore_*"))


# ---------------------------------------------------------------------------
# epub_library seam / translator_gui re-exports
# ---------------------------------------------------------------------------

def test_library_dir_seam(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    import epub_library

    home = tmp_path / "home"
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("HOME", str(home))
    for value in (None, "", "   "):
        if value is None:
            monkeypatch.delenv("GLOSSARION_LIBRARY_DIR", raising=False)
        else:
            monkeypatch.setenv("GLOSSARION_LIBRARY_DIR", value)
        expected = Path.home() / "Documents" / "Glossarion" / "Library"
        assert epub_library.get_library_dir() == str(expected)
    override = tmp_path / "docs" / "Library"
    monkeypatch.setenv("GLOSSARION_LIBRARY_DIR", str(override))
    assert epub_library.get_library_dir() == str(override) and override.is_dir()
    assert epub_library.get_library_raw_dir() == str(override / "Raw")


def test_translator_gui_reexports_shared_names():
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    import app_paths
    import config_backup
    import other_settings
    import translator_gui

    assert translator_gui.CONFIG_FILE == app_paths.CONFIG_FILE == app_paths.config_file_path()
    assert translator_gui._APP_DIR == app_paths._APP_DIR
    assert translator_gui._get_app_dir is app_paths._get_app_dir
    assert translator_gui._atomic_json_write is app_paths._atomic_json_write
    assert translator_gui.decrypt_config  # still importable for old callers
    assert other_settings._rename_output_files_for_retain.__module__ == "output_naming"
    assert other_settings._backup_config_file is config_backup._backup_config_file


# ---------------------------------------------------------------------------
# desktop parity: working-tree methods vs the BASE_SHA goldens (U0 oracle)
# ---------------------------------------------------------------------------

WORKTREE_SHA = "f" * 40


class _AstProxy(types.ModuleType):
    """``ast`` for the freezer, whose __init__ anchor searches for ``open(CONFIG_FILE``
    (the pre-move config load); the moved load is ``load_config(CONFIG_FILE)``."""

    def __getattr__(self, name):
        return getattr(ast, name)

    @staticmethod
    def unparse(node):
        text = ast.unparse(node)
        if isinstance(node, ast.Try) and "load_config(CONFIG_FILE" in text:
            text += "\n# open(CONFIG_FILE"
        return text


def _worktree_text(_sha, relpath):
    return (REPO_ROOT / relpath).read_bytes().decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")


@pytest.fixture(scope="module")
def parity(tmp_path_factory):
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from parity import capture_golden as cg
    from parity import fakes, freeze_legacy

    legacy_source("src/translator_gui.py")  # skips when BASE_SHA is not in the clone
    with pytest.MonkeyPatch.context() as mp:
        # legacy oracle at BASE_SHA (frozen into a temp dir when not present locally)
        try:
            legacy_bundle = freeze_legacy.load_legacy(BASE_SHA)
        except FileNotFoundError:
            legacy_dir = tmp_path_factory.mktemp("legacy")
            mp.setattr(freeze_legacy, "LEGACY_DIR", legacy_dir)
            freeze_legacy.freeze(BASE_SHA, legacy_dir)
            legacy_bundle = freeze_legacy.load_legacy(BASE_SHA)
        # the same freezer over the WORKING TREE = the moved code, driven by the same fakes
        out_dir = tmp_path_factory.mktemp("worktree")
        mp.setattr(freeze_legacy, "LEGACY_DIR", out_dir)
        mp.setattr(freeze_legacy, "resolve_sha", lambda rev="HEAD": WORKTREE_SHA)
        mp.setattr(freeze_legacy, "git_show_text", _worktree_text)
        mp.setattr(freeze_legacy, "ast", _AstProxy("ast"))
        mp.setattr(freeze_legacy, "EXTERNALS", (
            # functions that moved to GUI-free modules are not frozen: the live
            # output_naming / ollama_settings are the code under test
            freeze_legacy.ExternalSpec("other_settings", functions=("initialize_extraction_variables",),
                                       patch=("initialize_extraction_variables",)),
            freeze_legacy.ExternalSpec(
                "metadata_batch_translator",
                classes={"MetadataBatchTranslatorUI": ("__init__", "_initialize_default_prompts")},
                patch=("MetadataBatchTranslatorUI",)),
            freeze_legacy.ExternalSpec("config_backup", functions=("_backup_config_file",),
                                       bind=("_backup_config_file",)),
        ))
        freeze_legacy.freeze("WORKTREE", out_dir)
        new_bundle = freeze_legacy.load_legacy(WORKTREE_SHA)

    imports = new_bundle.manifest["imports"]
    for name, module in (("load_config", "config_store"), ("save_config_file", "config_store"),
                         ("apply_key_pools_to_runtime", "key_pools"), ("_get_app_dir", "app_paths"),
                         ("sanitize_prompt_profiles", "prompt_defaults")):
        assert imports.get(name) == f"from {module} import {name}", (name, imports.get(name))

    base_new = fakes.make_legacy_owner_factory(new_bundle)

    def new_factory(scenario, ctx):
        import app_paths

        sb = ctx.sandbox
        # app_paths is the live module the moved code reads paths from
        ctx.patch_module_attr("app_paths", "_APP_DIR", str(sb.app_dir))
        ctx.patch_module_attr("app_paths", "CONFIG_FILE", str(sb.config_file))
        ctx.patch_module_attr("app_paths", "__file__", str(sb.src_file("app_paths.py")))
        if not app_paths.config_file_path().startswith(str(sb.root)):
            raise RuntimeError("refusing to run: app_paths.CONFIG_FILE is outside the sandbox")
        return base_new(scenario, ctx)

    new_factory.kind = "worktree"
    return {"cg": cg, "legacy_bundle": legacy_bundle,
            "legacy_factory": fakes.make_legacy_owner_factory(legacy_bundle), "new_factory": new_factory}


def _scenario_names():
    from parity import scenarios

    return list(scenarios.SCENARIO_NAMES)


@pytest.mark.parametrize("name", _scenario_names())
def test_moved_code_reproduces_desktop_goldens(parity, name):
    from parity import scenarios

    cg = parity["cg"]
    result = cg.capture(parity["new_factory"], scenarios.get(name))
    try:
        expected = cg.load_golden(BASE_SHA, name)
    except (FileNotFoundError, OSError):
        expected = cg.roundtrip(cg.capture(parity["legacy_factory"], scenarios.get(name)))
    problems = cg.diff(expected, cg.roundtrip(result))
    assert not problems, "\n".join(problems)


def _boot_config_file(cg, factory, scenario):
    seen = {}

    def wrapped(scen, ctx):
        owner = factory(scen, ctx)
        path = ctx.sandbox.config_file
        seen["raw"] = path.read_text(encoding="utf-8") if path.exists() else None
        return owner

    wrapped.kind = factory.kind
    record = cg.capture(wrapped, scenario, entries=["boot"])
    assert "boot_error" not in record["entries"]["boot"], record["entries"]["boot"].get("boot_error")
    return seen["raw"]


@pytest.mark.parametrize("name", _scenario_names())
def test_startup_save_config_writes_same_config_json(parity, name):
    import api_key_encryption
    from parity import scenarios

    cg = parity["cg"]
    old = _boot_config_file(cg, parity["legacy_factory"], scenarios.get(name))
    new = _boot_config_file(cg, parity["new_factory"], scenarios.get(name))
    assert (old is None) == (new is None)
    if old is None:
        return
    old, new = json.loads(old), json.loads(new)
    assert list(old) == list(new)
    enc = lambda d: sorted(k for k, v in d.items() if isinstance(v, str) and v.startswith("ENC:"))  # noqa: E731
    assert enc(old) == enc(new)
    assert api_key_encryption.decrypt_config(old) == api_key_encryption.decrypt_config(new)
