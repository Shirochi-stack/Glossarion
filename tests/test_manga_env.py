"""manga_settings_defaults / manga_env / manga_runner / manga_files_core (Glossarion mobile rewrite, U8).

``MangaTranslationTab`` (manga_integration.py) lost its GUI-free methods to shared mixins:
``manga_files_core.MangaFilesMixin`` / ``MangaHooksMixin`` (Files tab, CBZ, callbacks),
``manga_env.MangaEnvMixin`` / ``MangaOcrSessionMixin`` (settings state, run-start env, glossary
paths, automatic OCR export) and ``manga_runner.MangaRunMixin`` (start, worker, stop, glossary
workflow). The tab inherits them; mobile runs the same code through
``manga_runner.HeadlessMangaRunner`` with a ``HeadlessOwner`` as ``main_gui``.

What is checked here:

* verbatim: every moved method / module helper equals its text at ``LEGACY_SHA`` (git show);
  ``_start_translation_heavy`` and ``__init__`` equal the legacy text once the documented split-outs
  are inlined back; ``MangaSettingsDialog.default_settings`` comes from ``default_manga_settings()``;
* composition: the tab inherits the mixins, no longer defines a moved name, overrides every GUI
  hook default; manga_integration re-exports the moved module helpers (Qt tier);
* hygiene: the new modules import without PySide6 (manga_env does not import manga_translator),
  parse as Python 3.10 and have uniform line endings;
* settings: defaults / merge equal the dialog's; the top-level defaults table equals what
  ``_load_rendering_settings`` produces on an empty config;
* run-start parity: the LEGACY ``_start_translation_heavy`` and the new one (+ split-outs) run the
  same scenario matrix (custom-api / google / azure / Document Intelligence / Qwen2-VL, key pools,
  batch modes, own-auth, aborts, existing translator, inpainting modes) with recording
  ``UnifiedClient`` / ``MangaTranslator`` / ``OCRManager`` stubs: logs, calls, env delta, config
  and queue are equal; ``build_manga_run_env`` returns the same env values;
* worker parity: LEGACY ``_translation_worker`` vs the new one (sequential, failures, CBZ jobs,
  OUTPUT_DIRECTORY, parallel panels, stop);
* per-image pipeline trace: ``MangaTranslator.process_image`` of the LEGACY module vs the working
  tree on a fixture page with recording BubbleDetector / OCRManager / LocalInpainter /
  UnifiedClient stand-ins (calls + arguments, result, written pixels); mobile mode runs the same
  pipeline without the print hijack;
* HeadlessOwner is the duck-typed ``main_gui`` (``MANGA_OWNER_CONTRACT`` from an AST scan);
* ``process_image`` print hijack is never installed on mobile, unchanged on desktop; on mobile
  ``restore_print`` leaves unified_api_client's own print alone;
* the headless runner (start / wait / outputs / CBZ / stop / glossary-only) and the contract
  helpers (``apply_rendering_settings``, ``build_ocr_config``, glossary env, OCR import, the
  font-size preset writes).

Run (repository root)::

    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests/test_manga_env.py
"""

from __future__ import annotations

import ast
import copy
import functools
import gc
import json
import os
import random
import re
import shutil
import subprocess
import sys
import textwrap
import threading
import time
import types
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
TESTS = REPO_ROOT / "tests"
for _p in (str(TESTS), str(SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _headless_env import headless_owner  # noqa: E402
from parity import runner_parity as rp  # noqa: E402

#: The commit the U8 manga code was moved from (manga_integration.py etc. before the move). Re-frozen
#: at 9355bb5d (main with the owner's safe_image upgrade): manga_integration / manga_settings_dialog
#: are unchanged since 9a46f869, manga_translator gained the open_image reads.
LEGACY_SHA = "9355bb5d376ca9eb8149a36b35b6028a95f86009"

NEW_MODULES = ("manga_settings_defaults", "manga_files_core", "manga_env", "manga_runner", "google_vision_rest")
#: (module, class) of the moved MangaTranslationTab methods.
MIXINS = (
    ("manga_files_core", "MangaHooksMixin"),
    ("manga_files_core", "MangaFilesMixin"),
    ("manga_env", "MangaEnvMixin"),
    ("manga_env", "MangaOcrSessionMixin"),
    ("manga_runner", "MangaRunMixin"),
)
#: Methods written for U8 (not moved): GUI-free hook defaults, the __init__ blocks, the split-outs.
NEW_METHODS = {
    "MangaHooksMixin": {"_log", "_reset_ui_state", "_monitor_translation_output", "_update_manga_image_range_display",
                        "_add_manga_file_item", "_rebuild_manga_file_listbox"},
    "MangaEnvMixin": {"_init_manga_run_state", "_init_manga_prompt_state", "_reset_manga_graceful_stop_env",
                      "_prepare_manga_run_env", "_apply_manga_batch_env"},
}
#: Module-level names moved out of manga_integration (re-exported there).
MOVED_MODULE_NAMES = {
    "manga_files_core": ("_get_app_dir", "_manga_cmd_debug_logging_enabled", "_manga_cmd_debug_print",
                         "_translation_run_token_matches", "_natural_sort_key", "_MANGA_SKIP_PREFIX",
                         "_manga_filename_without_skip_prefix"),
    "manga_runner": ("_IS_WINDOWS", "_lower_current_thread_priority_and_affinity", "_demote_non_main_threads"),
}
#: The only edits inside a moved module helper: ``_get_app_dir`` honours GLOSSARION_DATA_DIR
#: (mobile_runtime.data_dir returns its argument unchanged on desktop, which never sets it),
#: except in a frozen Windows build.
GET_APP_DIR_EDITS = (
    ("    return os.getcwd()", "    return data_dir(os.getcwd())"),
    ("        return os.path.dirname(os.path.abspath(__file__))",
     "        return data_dir(os.path.dirname(os.path.abspath(__file__)))"),
)


# ---------------------------------------------------------------------------
# real-data isolation
# ---------------------------------------------------------------------------

#: Modules whose ``_get_app_dir`` decides where the automatic OCR export ("OCR Text") and the manga
#: glossary backups ("MangaGlossary_Backup") go when no output folder is set (src/ on Windows).
_APP_DIR_MODULES = ("manga_files_core", "manga_env", "manga_runner", "manga_integration", "manga_editor_core")


@pytest.fixture(autouse=True)
def _no_writes_into_src(tmp_path):
    """No test writes into the user's src/ folder: ``_get_app_dir`` of the shared modules points at a
    temp dir and the HTTP request log (unified_api_client patches requests at import) is off. Its own
    MonkeyPatch, because several tests call ``monkeypatch.undo()`` between the legacy and new runs."""
    app_dir = tmp_path / "_app_dir"
    app_dir.mkdir(exist_ok=True)

    def fake_app_dir():
        return str(app_dir)

    patch = pytest.MonkeyPatch()
    try:
        patch.setenv("GLOSSARION_HTTP_LOG", "0")  # before unified_api_client's import-time enable
        for name in ("manga_files_core", "manga_env", "manga_runner"):
            __import__(name)  # imported now, so a test importing them later still sees the patch
        for name in _APP_DIR_MODULES:
            module = sys.modules.get(name)
            if module is not None and callable(vars(module).get("_get_app_dir")):
                patch.setattr(module, "_get_app_dir", fake_app_dir)
        http_logger = sys.modules.get("http_logger")
        if http_logger is not None and getattr(http_logger, "_log_folder", None) is not None:
            patch.setattr(http_logger, "_log_folder", None)  # _save_http_log returns early
        yield
    finally:
        patch.undo()


# ---------------------------------------------------------------------------
# sources
# ---------------------------------------------------------------------------


def _legacy_text(relpath):
    try:
        return rp.git_show(LEGACY_SHA, relpath)
    except rp.Unavailable as exc:  # pragma: no cover - shallow clone
        if os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"):
            pytest.fail(str(exc))
        pytest.skip(str(exc))


def _module_text(module):
    return (SRC / f"{module}.py").read_bytes().decode("utf-8-sig").replace("\r\n", "\n")


@functools.lru_cache(maxsize=None)
def _cached_legacy(relpath):
    return rp.git_show(LEGACY_SHA, relpath)


def _legacy_mi():
    try:
        return _cached_legacy("src/manga_integration.py")
    except rp.Unavailable as exc:  # pragma: no cover - shallow clone
        if os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"):
            pytest.fail(str(exc))
        pytest.skip(str(exc))


def _class_nodes(text, class_name):
    tree = ast.parse(text)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    return cls, {n.name: n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


def _legacy_tab_methods():
    text = _legacy_mi()
    _cls, methods = _class_nodes(text, "MangaTranslationTab")
    return {name: rp.node_text(text, node) for name, node in methods.items()}


def _new_methods(module, class_name):
    text = _module_text(module)
    _cls, methods = _class_nodes(text, class_name)
    return {name: rp.node_text(text, node) for name, node in methods.items()}


def _moved_triples():
    out = []
    for module, class_name in MIXINS:
        text = _module_text(module)
        _cls, methods = _class_nodes(text, class_name)
        for name in methods:
            if name not in NEW_METHODS.get(class_name, set()):
                out.append((module, class_name, name))
    return out


def _strip_n(text, n):
    """Remove exactly *n* leading spaces from every line (shorter whitespace lines -> '')."""
    return "\n".join(line[n:] if line.startswith(" " * n) else line.lstrip(" ") for line in text.split("\n"))


def _indent(lines, n):
    return [(" " * n + line) if line else line for line in lines]


def _body_lines(method_text):
    """Body lines of a 4-space indented method (after its docstring), at 4-space indentation."""
    flat = _strip_n(method_text, 4)
    fn = ast.parse(flat).body[0]
    first = fn.body[0]
    lines = flat.split("\n")
    has_doc = isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str)
    if has_doc:
        start = first.end_lineno  # 0-based index of the line after the docstring (comments included)
    else:
        start = first.lineno - 1
        while start - 1 > fn.lineno - 1 and lines[start - 1].lstrip().startswith("#"):
            start -= 1
    return lines[start:fn.end_lineno]


def _rstrip_lines(text):
    return "\n".join(line.rstrip() for line in text.split("\n"))


# ---------------------------------------------------------------------------
# verbatim moves
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("module,class_name,name", _moved_triples(),
                         ids=[f"{c}.{n}" for _m, c, n in _moved_triples()])
def test_moved_method_is_the_legacy_text(module, class_name, name):
    legacy = _legacy_tab_methods()
    assert name in legacy, f"{name} was not a MangaTranslationTab method at {LEGACY_SHA[:12]}"
    new = _new_methods(module, class_name)[name]
    if name == "_start_translation_heavy":
        new = _inline_start_translation_heavy(new)
        assert _rstrip_lines(new) == _rstrip_lines(legacy[name])
    else:
        assert new == legacy[name]


def _inline_start_translation_heavy(text):
    env = _new_methods("manga_env", "MangaEnvMixin")
    lines = text.split("\n")

    def replace(marker_lines, body):
        idx = next(i for i in range(len(lines)) if lines[i:i + len(marker_lines)] == marker_lines)
        lines[idx:idx + len(marker_lines)] = body

    graceful = _body_lines(env["_reset_manga_graceful_stop_env"])
    replace(["            self._reset_manga_graceful_stop_env()"], _indent(graceful, 8))
    prep = _body_lines(env["_prepare_manga_run_env"])
    assert prep[0] == "    import os" and prep[-1] == "    return ocr_config, api_key, model, needs_new_client"
    prep = [re.sub(r"^(\s+)return None$", r"\1return", line) for line in prep[1:-1]]
    replace(["            prepared = self._prepare_manga_run_env(start_token)",
             "            if prepared is None:",
             "                return",
             "            ocr_config, api_key, model, needs_new_client = prepared"], _indent(prep, 8))
    batch = _body_lines(env["_apply_manga_batch_env"])
    replace(["            self._apply_manga_batch_env()"], _indent(batch, 8))
    return "\n".join(lines)


def test_split_out_methods_have_exactly_the_documented_edits():
    env = _new_methods("manga_env", "MangaEnvMixin")
    prep = _body_lines(env["_prepare_manga_run_env"])
    assert sum(1 for line in prep if re.fullmatch(r"\s+return None", line)) == 6
    assert sum(1 for line in prep if re.fullmatch(r"\s+return", line)) == 0
    assert "self._reset_ui_state(start_token)" in "\n".join(prep)
    runner = _new_methods("manga_runner", "MangaRunMixin")["_start_translation_heavy"]
    assert runner.count("self._prepare_manga_run_env(start_token)") == 1
    assert runner.count("self._reset_manga_graceful_stop_env()") == 1
    assert runner.count("self._apply_manga_batch_env()") == 1


def test_init_blocks_are_verbatim_and_desktop_init_calls_them():
    legacy = _legacy_tab_methods()["__init__"]
    mi_text = _module_text("manga_integration")
    _cls, methods = _class_nodes(mi_text, "MangaTranslationTab")
    new_init = rp.node_text(mi_text, methods["__init__"]).split("\n")
    env_text = _module_text("manga_env")
    env_tree = ast.parse(env_text)
    limits = next(n for n in env_tree.body if isinstance(n, ast.FunctionDef) and n.name == "apply_manga_startup_thread_limits")
    limits_body = _body_lines(_indent_text(rp.node_text(env_text, limits), 4))
    env = _new_methods("manga_env", "MangaEnvMixin")
    blocks = {
        "        apply_manga_startup_thread_limits(main_gui)  # moved to manga_env (U8)": _indent(limits_body, 4),
        "        self._init_manga_run_state()  # moved to manga_env.MangaEnvMixin (U8)":
            _indent(_body_lines(env["_init_manga_run_state"]), 4),
        "        self._init_manga_prompt_state()  # moved to manga_env.MangaEnvMixin (U8)":
            _indent(_body_lines(env["_init_manga_prompt_state"]), 4),
    }
    rebuilt = []
    for line in new_init:
        rebuilt.extend(blocks.pop(line) if line in blocks else [line])
    assert not blocks, f"call lines missing from MangaTranslationTab.__init__: {list(blocks)}"
    assert _rstrip_lines("\n".join(rebuilt)) == _rstrip_lines(legacy)


def _indent_text(text, n):
    return "\n".join(_indent(text.split("\n"), n))


def test_module_helpers_are_verbatim():
    legacy = _legacy_mi()
    legacy_tree = ast.parse(legacy)
    for module, names in MOVED_MODULE_NAMES.items():
        text = _module_text(module)
        tree = ast.parse(text)
        for name in names:
            def find(t, src):
                for node in t.body:
                    if isinstance(node, ast.FunctionDef) and node.name == name:
                        return rp.node_text(src, node)
                    if isinstance(node, ast.Assign) and any(isinstance(x, ast.Name) and x.id == name for x in node.targets):
                        return "\n".join(src.split("\n")[node.lineno - 1:node.end_lineno])
                return None
            new = find(tree, text)
            old = find(legacy_tree, legacy)
            if name in ("_lower_current_thread_priority_and_affinity", "_demote_non_main_threads", "_IS_WINDOWS"):
                continue  # inside the Windows block: checked below
            assert new is not None and old is not None, name
            if name == "_get_app_dir":
                for before, after in GET_APP_DIR_EDITS:
                    assert old.count(before) == 1, before
                    old = old.replace(before, after)
            assert new == old, name
    # the Windows thread-priority block moved as one piece
    runner = _module_text("manga_runner")
    start = legacy.index("# Windows thread priority constants")
    end = legacy.index("# Try to import UnifiedClient for API initialization")
    assert legacy[start:end].rstrip("\n") in runner
    assert "_IS_WINDOWS = platform.system().lower().startswith('win')" in runner


def test_desktop_tab_keeps_only_gui_methods_and_inherits_the_mixins():
    mi_text = _module_text("manga_integration")
    tree = ast.parse(mi_text)
    cls, methods = _class_nodes(mi_text, "MangaTranslationTab")
    assert [ast.unparse(b) for b in cls.bases] == [
        "MangaRunMixin", "MangaOcrSessionMixin", "MangaEnvMixin", "MangaFilesMixin", "MangaHooksMixin", "QObject"]
    moved = {name for _m, _c, name in _moved_triples()}
    assert not (moved & set(methods)), sorted(moved & set(methods))
    for name in ("_global_cancelled", "_global_cancel_lock"):
        assert not any(isinstance(s, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in s.targets)
                       for s in cls.body), name
    import manga_files_core
    for hook in manga_files_core.MangaHooksMixin.GUI_HOOKS:
        assert hook in methods, f"MangaTranslationTab must override the GUI-free default {hook}"
    module_defs = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    module_assigns = {t.id for n in tree.body if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}
    for names in MOVED_MODULE_NAMES.values():
        for name in names:
            assert name not in module_defs and name not in module_assigns, name
    # every method the tab still defines is unchanged, except __init__ (checked above)
    legacy = _legacy_tab_methods()
    changed = [name for name, node in methods.items()
               if name != "__init__" and rp.node_text(mi_text, node) != legacy.get(name)]
    assert not changed, changed


def test_settings_dialog_builds_its_defaults_from_the_shared_module():
    legacy_text = _cached_legacy("src/manga_settings_dialog.py")
    _cls, legacy_methods = _class_nodes(legacy_text, "MangaSettingsDialog")
    assign = next(n for n in ast.walk(legacy_methods["__init__"]) if isinstance(n, ast.Assign)
                  and ast.unparse(n.targets[0]) == "self.default_settings")
    import manga_settings_defaults as msd

    assert msd.default_manga_settings() == ast.literal_eval(assign.value)
    new_text = _module_text("manga_settings_dialog")
    _cls, methods = _class_nodes(new_text, "MangaSettingsDialog")
    new_assign = next(n for n in ast.walk(methods["__init__"]) if isinstance(n, ast.Assign)
                      and ast.unparse(n.targets[0]) == "self.default_settings")
    assert ast.unparse(new_assign.value) == "default_manga_settings()"
    # the literal itself moved verbatim (comments included)
    legacy_lines = legacy_text.split("\n")[assign.lineno:assign.end_lineno - 1]
    msd_text = _module_text("manga_settings_defaults")
    fn = next(n for n in ast.parse(msd_text).body if isinstance(n, ast.FunctionDef) and n.name == "default_manga_settings")
    ret = fn.body[-1]
    new_lines = msd_text.split("\n")[ret.lineno:ret.end_lineno - 1]
    assert new_lines == [_strip_n(line, 4) for line in legacy_lines]
    # every other dialog method is unchanged
    changed = [name for name, node in methods.items()
               if name != "__init__" and rp.node_text(new_text, node) != rp.node_text(legacy_text, legacy_methods[name])]
    assert not changed, changed


def test_schema_generator_reads_the_shared_manga_defaults():
    sys.path.insert(0, str(SRC / "mobile" / "tools"))
    try:
        import schema_extract as se
    finally:
        sys.path.remove(str(SRC / "mobile" / "tools"))
    assert ("manga_settings", "manga_settings_defaults.py", "default_manga_settings", "return") in se.NESTED_DEFAULT_SOURCES
    assert "manga_settings_defaults.py" in se.EXTRA_MODULES


# ---------------------------------------------------------------------------
# hygiene
# ---------------------------------------------------------------------------


def test_new_modules_parse_as_python_310_with_uniform_line_endings():
    for module in NEW_MODULES:
        data = (SRC / f"{module}.py").read_bytes()
        assert data.count(b"\r\n") in (0, data.count(b"\n")), f"{module}.py has mixed line endings"
        ast.parse(data.decode("utf-8"), feature_version=(3, 10))


def test_new_modules_import_without_qt():
    code = textwrap.dedent("""
        import sys
        for name in ('PySide6', 'PySide6.QtWidgets', 'PySide6.QtCore', 'PySide6.QtGui', 'shiboken6',
                     'translator_gui', 'dpi_setup', 'ImageRenderer', 'manga_integration', 'manga_image_preview'):
            sys.modules[name] = None
        import manga_settings_defaults, google_vision_rest, manga_files_core, manga_env
        assert 'manga_translator' not in sys.modules, 'manga_env must stay light'
        assert 'cv2' not in sys.modules
        import manga_runner
        assert not hasattr(manga_files_core.ImageRenderer, '_add_text_overlay_to_viewer')
        print('ok')
    """)
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONPATH=os.pathsep.join(
        [str(SRC)] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]))
    proc = subprocess.run([sys.executable, "-c", code], cwd=str(SRC), env=env, capture_output=True,
                          text=True, encoding="utf-8", errors="replace", timeout=600)
    assert proc.returncode == 0 and proc.stdout.strip().endswith("ok"), proc.stdout[-2000:] + proc.stderr[-4000:]


# ---------------------------------------------------------------------------
# settings defaults
# ---------------------------------------------------------------------------


def _legacy_merge(existing):
    """``MangaSettingsDialog._merge_settings`` at LEGACY_SHA with a fresh ``default_settings``."""
    legacy_text = _cached_legacy("src/manga_settings_dialog.py")
    _cls, methods = _class_nodes(legacy_text, "MangaSettingsDialog")
    assign = next(n for n in ast.walk(methods["__init__"]) if isinstance(n, ast.Assign)
                  and ast.unparse(n.targets[0]) == "self.default_settings")
    source = "class _D:\n" + rp.node_text(legacy_text, methods["_merge_settings"]) + "\n"
    ns = {"Dict": dict}
    exec(compile(source, "<legacy _merge_settings>", "exec"), ns)
    owner = types.SimpleNamespace(default_settings=ast.literal_eval(assign.value))
    return ns["_D"]._merge_settings(owner, copy.deepcopy(existing))


def _random_manga_settings(rng):
    import manga_settings_defaults as msd

    tree = msd.default_manga_settings()
    out = {}
    for key, value in tree.items():
        if rng.random() < 0.5:
            continue
        if isinstance(value, dict):
            out[key] = {k: (not v if isinstance(v, bool) else v) for k, v in value.items() if rng.random() < 0.4}
            if rng.random() < 0.2:
                out[key]["extra_user_key"] = rng.randint(0, 9)
        else:
            out[key] = value if rng.random() < 0.5 else "changed"
    if rng.random() < 0.3:
        out["unknown_section"] = {"a": 1}
    return out


def test_merge_manga_settings_equals_the_dialog_merge():
    import manga_settings_defaults as msd

    rng = random.Random(8)
    for _ in range(300):
        existing = _random_manga_settings(rng)
        config = {"manga_settings": copy.deepcopy(existing)}
        assert msd.merge_manga_settings(config) == _legacy_merge(existing)
        assert config == {"manga_settings": existing}, "merge must not modify the config"
    assert msd.merge_manga_settings({}) == msd.default_manga_settings()
    assert msd.merge_manga_settings({"ocr": {"detector_type": "yolo"}})["ocr"]["detector_type"] == "yolo"


#: MANGA_TOP_LEVEL_DEFAULTS key -> the tab attribute _load_rendering_settings sets from it.
_TOP_LEVEL_ATTRS = {
    'manga_bg_opacity': 'bg_opacity_value', 'manga_free_text_only_bg_opacity': 'free_text_only_bg_opacity_value',
    'manga_bg_style': 'bg_style_value', 'manga_bg_reduction': 'bg_reduction_value', 'manga_font_size': 'font_size_value',
    'manga_font_path': 'selected_font_path', 'manga_skip_inpainting': 'skip_inpainting_value',
    'manga_inpaint_quality': 'inpaint_quality_value', 'manga_inpaint_dilation': 'inpaint_dilation_value',
    'manga_inpaint_passes': 'inpaint_passes_value',
    'manga_disable_inpaint_performance_mode': 'disable_inpaint_performance_mode_value',
    'manga_font_size_mode': 'font_size_mode_value', 'manga_font_size_multiplier': 'font_size_multiplier_value',
    'manga_force_caps_lock': 'force_caps_lock_value', 'manga_constrain_to_bubble': 'constrain_to_bubble_value',
    'manga_max_font_size': 'max_font_size_value', 'manga_strict_text_wrapping': 'strict_text_wrapping_value',
    'manga_safe_area_enabled': 'safe_area_enabled_value', 'manga_safe_area_scale': 'safe_area_scale_value',
    'manga_shadow_enabled': 'shadow_enabled_value', 'manga_shadow_offset_x': 'shadow_offset_x_value',
    'manga_shadow_offset_y': 'shadow_offset_y_value', 'manga_shadow_blur': 'shadow_blur_value',
    'manga_font_style': 'font_style_value', 'manga_full_page_context': 'full_page_context_value',
    'manga_glossary_enabled': 'manga_glossary_enabled_value', 'manga_custom_glossary_path': 'manga_custom_glossary_path',
    'manga_generated_glossary_path': 'manga_generated_glossary_path',
    'manga_glossary_auto_load_suppressed': 'manga_glossary_auto_load_suppressed',
    'manga_glossary_auto_load_suppressed_root': 'manga_glossary_auto_load_suppressed_root',
    'manga_split_first_level_subfolders': 'manga_split_first_level_subfolders_value',
    'manga_glossary_debug_ocr_text': 'manga_glossary_debug_ocr_text_value',
    'manga_visual_context_enabled': 'visual_context_enabled_value', 'qwen2vl_model_size': 'qwen2vl_model_size',
    'rapidocr_use_recognition': 'rapidocr_use_recognition_value', 'rapidocr_language': 'rapidocr_language_value',
    'rapidocr_detection_mode': 'rapidocr_detection_mode_value',
    'manga_custom_api_ocr_batch_enabled': 'custom_api_ocr_batch_enabled_value',
    'manga_custom_api_ocr_batch_size': 'custom_api_ocr_batch_size_value',
    'manga_batch_image_requests_enabled': 'batch_image_requests_enabled_value',
    'manga_batch_image_requests_size': 'batch_image_requests_size_value',
    'manga_create_cbz_at_end': 'create_cbz_at_end_value', 'manga_auto_consolidate_images': 'auto_consolidate_images_value',
}


def test_top_level_defaults_are_what_the_tab_loads_from_an_empty_config(tmp_path, monkeypatch):
    import manga_env
    import manga_settings_defaults as msd

    with headless_owner(tmp_path, monkeypatch, {}) as owner:
        owner.config = {}
        state = manga_env.HeadlessMangaState(owner)
    for key, default in msd.MANGA_TOP_LEVEL_DEFAULTS.items():
        if key == 'manga_ocr_provider':
            assert state.ocr_provider_value == default
        elif key == 'manga_text_color':
            assert [state.text_color_r_value, state.text_color_g_value, state.text_color_b_value] == default
        elif key == 'manga_shadow_color':
            assert [state.shadow_color_r_value, state.shadow_color_g_value, state.shadow_color_b_value] == default
        else:
            assert getattr(state, _TOP_LEVEL_ATTRS[key]) == default, key
    assert set(msd.MANGA_TOP_LEVEL_DEFAULTS) == set(_TOP_LEVEL_ATTRS) | {'manga_ocr_provider', 'manga_text_color',
                                                                       'manga_shadow_color'}
    assert state.ocr_prompt == manga_env.default_manga_ocr_prompt()
    assert state.manga_glossary_prompt == manga_env.default_manga_glossary_prompt()
    assert state.full_page_context_prompt == manga_env.default_full_page_context_prompt()


# ---------------------------------------------------------------------------
# recording stubs
# ---------------------------------------------------------------------------


_TIME_RE = re.compile(r"\d+\.\d+s\b")


class Recorder:
    def __init__(self, root):
        self.root = str(root)
        self.events = []
        self.lock = threading.Lock()

    def norm(self, value):
        if isinstance(value, str):
            value = value.replace(self.root, "<ROOT>").replace(self.root.replace("\\", "/"), "<ROOT>")
            return _TIME_RE.sub("<t>s", value)
        if isinstance(value, dict):
            return {self.norm(k): self.norm(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self.norm(v) for v in value]
        if isinstance(value, (int, float, bool)) or value is None:
            return value
        return f"<{type(value).__name__}>"

    def add(self, *event, **data):
        with self.lock:
            self.events.append(self.norm(list(event) + ([data] if data else [])))

    @property
    def host(self):
        recorder = self

        class _Host:
            def log(self, message, **kw):
                recorder.add("log", kw.get("level", "info"), message)

            def emit(self, kind, **data):
                recorder.add("emit", kind, data)

        return _Host()


def make_fake_client(rec):
    import unified_api_client

    real = unified_api_client.UnifiedClient

    class FakeUnifiedClient:
        def __init__(self, model=None, api_key=None, **kwargs):
            rec.add("UnifiedClient", model, api_key, sorted(kwargs))
            self.model = model

        @classmethod
        def _model_needs_api_key(cls, model):
            return real._model_needs_api_key(model)

        @classmethod
        def set_global_cancellation(cls, value):
            rec.add("UnifiedClient.set_global_cancellation", value)

        @classmethod
        def is_globally_cancelled(cls):
            return False

        def __getattr__(self, name):
            raise AttributeError(name)

    for pool in ("multi", "inpainter", "vision", "glossary", "fallback"):
        def setter(cls, keys, _pool=pool, **kw):
            rec.add(f"set_in_memory_{_pool}_keys", keys, kw)

        def clearer(cls, _pool=pool):
            rec.add(f"clear_in_memory_{_pool}_keys")

        setattr(FakeUnifiedClient, f"set_in_memory_{pool}_keys", classmethod(setter))
        setattr(FakeUnifiedClient, f"clear_in_memory_{pool}_keys", classmethod(clearer))
    return FakeUnifiedClient


def make_fake_translator(rec, *, fail=(), stop_after=None, stop_flag_owner=None):
    """A recording MangaTranslator: process_image copies the page to its output path."""

    class FakeMangaTranslator:
        _globally_cancelled = False

        def __init__(self, ocr_config, client, main_gui, log_callback=None, **kwargs):
            object.__setattr__(self, "_rec", rec)
            rec.add("MangaTranslator", dict(ocr_config), getattr(client, "model", None), sorted(kwargs))
            object.__setattr__(self, "client", client)
            object.__setattr__(self, "main_gui", main_gui)
            object.__setattr__(self, "manga_settings", main_gui.config.setdefault('manga_settings', {}))

        def __setattr__(self, name, value):
            if callable(value) and not isinstance(value, (str, int, float, bool)):
                rec.add("translator.set", name, "<callable>")
            else:
                rec.add("translator.set", name, value)
            object.__setattr__(self, name, value)

        @classmethod
        def is_globally_cancelled(cls):
            return False

        @classmethod
        def set_global_cancellation(cls, value):
            rec.add("MangaTranslator.set_global_cancellation", value)

        @classmethod
        def reset_global_flags(cls):
            rec.add("MangaTranslator.reset_global_flags")

        @classmethod
        def force_release_all_pool_checkouts(cls, *args, **kwargs):
            rec.add("MangaTranslator.force_release_all_pool_checkouts")
            return 0, 0

        def set_stop_flag(self, flag):
            rec.add("translator.set_stop_flag")
            object.__setattr__(self, "stop_flag", flag)

        def set_full_page_context(self, enabled=False, custom_prompt=None):
            rec.add("translator.set_full_page_context", enabled, bool(custom_prompt))

        def update_text_rendering_settings(self, **kwargs):
            rec.add("translator.update_text_rendering_settings", kwargs)

        def reset_history_manager(self):
            rec.add("translator.reset_history_manager")

        def reset_for_new_image(self):
            rec.add("translator.reset_for_new_image")

        def process_image(self, image_path, output_path=None, batch_index=None, batch_total=None, **kwargs):
            name = os.path.basename(image_path)
            rec.add("process_image", name, output_path, batch_index, batch_total, sorted(kwargs))
            if stop_after is not None and stop_flag_owner is not None:
                done = sum(1 for e in rec.events if e and e[0] == "process_image")
                if done >= stop_after:
                    stop_flag_owner.stop_flag.set()
            if name in fail:
                return {"success": False, "output_path": output_path, "regions": [], "errors": ["boom 429"]}
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            shutil.copyfile(image_path, output_path)
            return {"success": True, "output_path": output_path,
                    "regions": [{"translated_text": "hello"}, {"translated_text": ""}]}

        def __getattr__(self, name):
            if name.startswith("__"):
                raise AttributeError(name)

            def _call(*args, **kwargs):
                rec.add(f"translator.{name}")
                return None

            return _call

    return FakeMangaTranslator


def make_ocr_stub(rec):
    module = types.ModuleType("ocr_manager")

    class OCRManager:
        def __init__(self, log_callback=None):
            rec.add("OCRManager")
            self.providers = {}

        def get_provider(self, name):
            rec.add("OCRManager.get_provider", name)
            return None

        def set_stop_flag(self, flag):
            rec.add("OCRManager.set_stop_flag")

    module.OCRManager = OCRManager
    return module


def make_image_renderer_stub(rec):
    module = types.ModuleType("ImageRenderer")
    module._reset_cancellation_flags = lambda tab: rec.add("ImageRenderer._reset_cancellation_flags")
    return module


def _effective_env():
    import large_env

    env = dict(os.environ)
    env.update(dict(getattr(large_env, "_store", {}) or {}))
    return env


def _write_pages(root, names=("1.png", "2.png", "3.png"), sub="manga"):
    from PIL import Image

    folder = Path(root) / sub
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, name in enumerate(names):
        path = folder / name
        Image.new("RGB", (32 + i, 32), (255, 255, 255)).save(path)
        paths.append(str(path))
    return paths


def _legacy_tab_function(name, globals_):
    source = "class _LegacyTab:\n" + _legacy_tab_methods()[name] + "\n"
    ns = dict(globals_)
    ns.setdefault("__builtins__", __builtins__)
    exec(compile(source, f"<manga_integration@{LEGACY_SHA[:12]}:{name}>", "exec"), ns)
    return vars(ns["_LegacyTab"])[name]


# ---------------------------------------------------------------------------
# run-start parity (legacy _start_translation_heavy vs new + split-outs)
# ---------------------------------------------------------------------------

_KEYS = [{"api_key": "sk-one", "model": "gpt-4o-mini"}, {"api_key": "sk-two", "model": "gpt-4o"}]
START_SCENARIOS = {
    "custom_api_defaults": {"config": {"model": "gpt-4o-mini", "api_key": "sk-main"}},
    "custom_api_key_pools_batch": {"config": {
        "model": "gpt-4o-mini", "api_key": "sk-main", "use_multi_api_keys": True, "multi_api_keys": _KEYS,
        "force_key_rotation": False, "rotation_frequency": 3, "use_inpainter_keys": True, "inpainter_keys": _KEYS[:1],
        "use_fallback_keys": True, "fallback_keys": _KEYS[1:], "use_qa_scan_keys": True, "qa_scan_keys": _KEYS,
        "batch_translation": True, "batch_size": 4, "batching_mode": "conservative", "batch_group_size": 2,
        "manga_settings": {"advanced": {"parallel_processing": True, "max_workers": 3}},
        "manga_settings_compression": True}},
    "multi_key_without_valid_keys_aborts": {"config": {
        "model": "gpt-4o-mini", "api_key": "sk-main", "use_multi_api_keys": True,
        "multi_api_keys": [{"api_key": "", "model": ""}]}},
    "google_with_credentials": {"config": {"model": "gpt-4o-mini", "api_key": "sk-main", "manga_ocr_provider": "google",
                                           "google_vision_credentials": "<CREDS>"}},
    "google_missing_credentials_aborts": {"config": {"model": "gpt-4o-mini", "api_key": "sk-main",
                                                     "manga_ocr_provider": "google",
                                                     "google_vision_credentials": "<ROOT>/missing.json"}},
    "azure_with_key": {"config": {"model": "gpt-4o-mini", "api_key": "sk-main", "manga_ocr_provider": "azure",
                                  "azure_vision_key": "az-key", "azure_vision_endpoint": "https://az.example/"}},
    "azure_without_key_aborts": {"config": {"model": "gpt-4o-mini", "api_key": "sk-main", "manga_ocr_provider": "azure"}},
    "document_intelligence_uses_cv_widgets": {"config": {
        "model": "gpt-4o-mini", "api_key": "sk-main", "manga_ocr_provider": "azure-document-intelligence",
        "azure_vision_key": "cv-key", "azure_vision_endpoint": "https://cv.example/",
        "azure_document_intelligence_key": "di-key", "azure_document_intelligence_endpoint": "https://di.example/"}},
    "qwen2vl_provider": {"config": {"model": "gpt-4o-mini", "api_key": "sk-main", "manga_ocr_provider": "Qwen2-VL"}},
    "own_auth_model_without_key": {"config": {"model": "authgpt/gpt-6-luna", "manga_ocr_provider": "azure",
                                              "azure_vision_key": "az", "azure_vision_endpoint": "https://e/"}},
    "missing_key_aborts": {"config": {"model": "gpt-4o-mini", "manga_ocr_provider": "azure",
                                      "azure_vision_key": "az", "azure_vision_endpoint": "https://e/"}},
    "bubble_detection_off_writes_detector_defaults": {"config": {
        "model": "gpt-4o-mini", "api_key": "sk-main",
        "manga_settings": {"ocr": {"bubble_detection_enabled": False}}}},
    "existing_translator_cloud_inpainting": {"config": {
        "model": "gpt-4o-mini", "api_key": "sk-main", "replicate_api_key": "r8-key",
        "manga_settings": {"inpainting": {"method": "cloud"}}}, "existing_translator": True},
    "existing_translator_skip_inpainting_model_change": {"config": {
        "model": "gpt-4o", "api_key": "sk-main", "manga_skip_inpainting": True,
        "batch_translation": False, "batching_mode": "aggressive"}, "existing_translator": True},
    "output_directory_override_and_range": {"config": {
        "model": "gpt-4o-mini", "api_key": "sk-main", "output_directory": "<ROOT>/out"}, "image_range": "2-3"},
}


def _materialize(value, root):
    if isinstance(value, str):
        return value.replace("<ROOT>", str(root))
    if isinstance(value, dict):
        return {k: _materialize(v, root) for k, v in value.items()}
    if isinstance(value, list):
        return [_materialize(v, root) for v in value]
    return value


def _run_start(tmp_path, monkeypatch, scenario, side):
    import manga_runner
    import unified_api_client

    root = Path(tmp_path) / side
    root.mkdir(parents=True, exist_ok=True)
    rec = Recorder(root)
    config = _materialize(copy.deepcopy(scenario["config"]), root)
    if config.get("google_vision_credentials") == "<CREDS>":
        creds = root / "creds.json"
        creds.write_text("{}", encoding="utf-8")
        config["google_vision_credentials"] = str(creds)
    if config.pop("manga_settings_compression", False):
        config.setdefault("manga_settings", {})["compression"] = {"enabled": True, "format": "webp"}
    pages = _write_pages(root)
    fake_client = make_fake_client(rec)
    fake_translator = make_fake_translator(rec)
    monkeypatch.setattr(unified_api_client, "UnifiedClient", fake_client)
    monkeypatch.setitem(sys.modules, "ImageRenderer", make_image_renderer_stub(rec))
    monkeypatch.setitem(sys.modules, "ocr_manager", make_ocr_stub(rec))
    monkeypatch.setattr(manga_runner, "MangaTranslator", fake_translator)
    with headless_owner(root, monkeypatch, config, host=rec.host) as owner:
        owner.save_config = lambda show_message=True: rec.add("save_config", show_message)
        runner = manga_runner.HeadlessMangaRunner(owner, host=rec.host, files=pages,
                                                  image_range=scenario.get("image_range", ""))
        if scenario.get("existing_translator"):
            owner.client = fake_client(model="gpt-4o-mini", api_key="sk-existing")
            runner.translator = fake_translator({"provider": "custom-api"}, owner.client, owner)
        runner._translation_worker = lambda token=None: rec.add("worker", token)
        processing, _err = runner._manga_range_filtered_files()
        runner._manga_processing_files = None
        runner._translation_start_token = 1
        runner._translation_startup_pending = True
        runner.is_running = True
        rec.add("--start--")
        before = _effective_env()
        if side == "legacy":
            heavy = _legacy_tab_function("_start_translation_heavy", {
                "os": os, "json": json, "time": time, "threading": threading, "MangaTranslator": fake_translator,
                "_lower_current_thread_priority_and_affinity": manga_runner._lower_current_thread_priority_and_affinity,
            })
        else:
            heavy = manga_runner.MangaRunMixin._start_translation_heavy
        thread = threading.Thread(target=heavy, args=(runner, None, None, 1), name="MangaStartHeavy")
        thread.start()
        thread.join(60)
        assert not thread.is_alive()
        worker = getattr(runner, "translation_thread", None)
        if worker is not None:
            worker.join(30)
        after = _effective_env()
        delta = {k: after.get(k) for k in sorted(set(before) | set(after)) if before.get(k) != after.get(k)}
        queue_items = []
        while not runner.update_queue.empty():
            queue_items.append(runner.update_queue.get_nowait())
        result = {
            "events": rec.events,
            "env": rec.norm(delta),
            "config": rec.norm(json.loads(json.dumps(owner.config, default=str))),
            "queue": rec.norm([list(item) for item in queue_items]),
            "state": rec.norm({
                "total": runner.total_files, "completed": runner.completed_files, "failed": runner.failed_files,
                "processing": runner._manga_processing_files, "startup_pending": runner._translation_startup_pending,
                "is_running": runner.is_running, "ocr_prompt": runner.ocr_prompt,
                "client_model": getattr(getattr(owner, "client", None), "model", None),
                "translator": type(getattr(runner, "translator", None)).__name__,
                "processing_files": len(processing),
            }),
        }
    return result


@pytest.mark.parametrize("name", sorted(START_SCENARIOS))
def test_run_start_matches_the_legacy_start(tmp_path, monkeypatch, name):
    scenario = START_SCENARIOS[name]
    legacy = _run_start(tmp_path, monkeypatch, scenario, "legacy")
    monkeypatch.undo()
    new = _run_start(tmp_path, monkeypatch, scenario, "new")
    assert new["events"] == legacy["events"]
    assert new["env"] == legacy["env"]
    assert new["config"] == legacy["config"]
    assert new["queue"] == legacy["queue"]
    assert new["state"] == legacy["state"]
    aborted = "aborts" in name
    assert any(e[0] == "worker" for e in new["events"]) == (not aborted)


@pytest.mark.parametrize("name", sorted(n for n in START_SCENARIOS if "existing" not in n))
def test_build_manga_run_env_is_the_start_env(tmp_path, monkeypatch, name):
    import manga_env
    import unified_api_client

    scenario = START_SCENARIOS[name]
    legacy = _run_start(tmp_path, monkeypatch, scenario, "legacy")
    monkeypatch.undo()
    root = Path(tmp_path) / "build"
    root.mkdir(parents=True, exist_ok=True)
    rec = Recorder(root)
    config = _materialize(copy.deepcopy(scenario["config"]), root)
    if config.get("google_vision_credentials") == "<CREDS>":
        creds = root / "creds.json"
        creds.write_text("{}", encoding="utf-8")
        config["google_vision_credentials"] = str(creds)
    if config.pop("manga_settings_compression", False):
        config.setdefault("manga_settings", {})["compression"] = {"enabled": True, "format": "webp"}
    monkeypatch.setattr(unified_api_client, "UnifiedClient", make_fake_client(rec))
    monkeypatch.setitem(sys.modules, "ImageRenderer", make_image_renderer_stub(rec))
    monkeypatch.setitem(sys.modules, "ocr_manager", make_ocr_stub(rec))
    with headless_owner(root, monkeypatch, config, host=rec.host) as owner:
        owner.save_config = lambda show_message=True: None
        if "aborts" in name:
            with pytest.raises(manga_env.MangaRunEnvError):
                manga_env.build_manga_run_env(owner, host=rec.host)
            return
        delta = manga_env.build_manga_run_env(owner, host=rec.host)
        built = rec.norm(dict(delta))
        assert delta.ocr_config.get("provider") == (config.get("manga_ocr_provider") or "custom-api")
    later_keys = {"OUTPUT_DIRECTORY", "EXTRACTION_WORKERS"}
    assert set(legacy["env"]) - set(built) <= later_keys, sorted(set(legacy["env"]) - set(built))
    for key, value in built.items():
        assert legacy["env"].get(key) == value, key


# ---------------------------------------------------------------------------
# worker parity (legacy _translation_worker vs new)
# ---------------------------------------------------------------------------

WORKER_SCENARIOS = {
    "sequential": {},
    "sequential_failure_and_cbz_at_end": {"fail": ("2.png",), "config": {"manga_create_cbz_at_end": True}},
    "output_directory_override": {"config": {"output_directory": "<ROOT>/out"}, "env": {"OUTPUT_DIRECTORY": "<ROOT>/out"}},
    "cbz_job_routing": {"cbz": True},
    "parallel_panels": {"config": {"manga_settings": {"advanced": {"parallel_panel_translation": True,
                                                                   "panel_max_workers": 2,
                                                                   "panel_start_stagger_ms": 0}}},
                        "unordered": True},
    "stop_after_first_page": {"stop_after": 1},
    "auto_cleanup_and_unload": {"config": {"manga_settings": {"advanced": {"auto_cleanup_models": True,
                                                                           "unload_models_after_translation": True}}}},
}


def _run_worker(tmp_path, monkeypatch, scenario, side):
    import manga_runner
    import manga_translator

    root = Path(tmp_path) / side
    root.mkdir(parents=True, exist_ok=True)
    rec = Recorder(root)
    config = _materialize(copy.deepcopy(scenario.get("config", {})), root)
    config.setdefault("model", "gpt-4o-mini")
    pages = _write_pages(root)
    with headless_owner(root, monkeypatch, config, host=rec.host) as owner:
        for key, value in scenario.get("env", {}).items():
            os.environ[key] = _materialize(value, root)
        owner.save_config = lambda show_message=True: None
        runner = manga_runner.HeadlessMangaRunner(owner, host=rec.host, files=pages)
        fake = make_fake_translator(rec, fail=scenario.get("fail", ()), stop_after=scenario.get("stop_after"),
                                    stop_flag_owner=runner)
        monkeypatch.setattr(manga_translator, "MangaTranslator", fake)
        if scenario.get("cbz"):
            cbz = Path(root) / "book.cbz"
            with zipfile.ZipFile(cbz, "w") as zf:
                for page in pages:
                    zf.write(page, os.path.basename(page))
            runner.selected_files = []
            runner.cbz_temp_root = str(Path(root) / "cbz_tmp")
            runner._add_dropped_manga_paths([str(cbz)])
        owner.client = types.SimpleNamespace(model="gpt-4o-mini")
        runner.translator = fake({"provider": "custom-api"}, owner.client, owner)
        runner._manga_processing_files, _err = runner._manga_range_filtered_files()
        runner.total_files = len(runner._manga_processing_files)
        runner._translation_start_token = 1
        runner.is_running = True
        rec.add("--worker--")
        if side == "legacy":
            worker = _legacy_tab_function("_translation_worker", {"os": os})
        else:
            worker = manga_runner.MangaRunMixin._translation_worker
        thread = threading.Thread(target=worker, args=(runner, 1))
        thread.start()
        thread.join(120)
        assert not thread.is_alive()
        queue_items = []
        while not runner.update_queue.empty():
            queue_items.append(runner.update_queue.get_nowait())
        tree = sorted(str(p.relative_to(root)).replace("\\", "/") for p in root.rglob("*") if p.is_file())
        archives = {}
        for path in root.rglob("*.cbz"):
            with zipfile.ZipFile(path) as zf:
                archives[str(path.relative_to(root)).replace("\\", "/")] = sorted(zf.namelist())
        events = rec.events
        queue = [rec.norm(list(q)) for q in queue_items]
        if scenario.get("unordered"):
            events = sorted(json.dumps(e, sort_keys=True, default=str) for e in events)
            # parallel panels: the started/done counters in progress items follow thread timing
            queue = [_unordered_queue_item(q) for q in queue]
        result = {
            "events": events,
            "queue": sorted(json.dumps(q, default=str) for q in queue),
            "tree": tree,
            "archives": archives,
            "counts": (runner.completed_files, runner.failed_files, runner.total_files),
            "translator_kept": runner.translator is not None,
        }
    return result


def _unordered_queue_item(item):
    """A queue item with the timing-dependent counters of a parallel run masked."""
    if item and item[0] == "progress" and len(item) == 4 and isinstance(item[3], str):
        return ["progress", "<n>", item[2], re.sub(r"\b\d+/(\d+)", r"<n>/\1", item[3])]
    return item


@pytest.mark.parametrize("name", sorted(WORKER_SCENARIOS))
def test_worker_matches_the_legacy_worker(tmp_path, monkeypatch, name):
    scenario = WORKER_SCENARIOS[name]
    legacy = _run_worker(tmp_path, monkeypatch, scenario, "legacy")
    monkeypatch.undo()
    new = _run_worker(tmp_path, monkeypatch, scenario, "new")
    assert new == legacy
    assert any(e[0] == "process_image" if isinstance(e, list) else '"process_image"' in e for e in new["events"])


# ---------------------------------------------------------------------------
# HeadlessOwner as main_gui
# ---------------------------------------------------------------------------

#: main_gui names TranslatorGUI has that a fresh HeadlessOwner lacks, and why that is the same run.
DESKTOP_ONLY_MAIN_GUI = {
    "_ensure_executor": "desktop runs the worker on its ThreadPoolExecutor; without it the tab starts a "
                        "dedicated thread (same code); EXTRACTION_WORKERS is set from extraction_workers_var "
                        "right after either way",
    "manga_translator": "set by TranslatorGUI.open_manga_translator to the tab; HeadlessMangaState sets it to itself",
    "graceful_stop_active": "written (never read first) by the run start / stop",
}


def test_headless_owner_is_the_duck_typed_main_gui(tmp_path, monkeypatch):
    import headless_owner as ho
    import manga_env

    contract = ho.MANGA_OWNER_CONTRACT
    assert contract == ho.compute_manga_owner_contract()
    assert {"config", "_get_environment_variables"} <= set(contract)
    assert "set_all_environment_variables" not in contract  # hasattr-guarded; only app.py defines it
    with headless_owner(tmp_path, monkeypatch, {}) as owner:
        missing = [name for name in contract if not hasattr(owner, name)]
        assert not missing, missing
        assert not hasattr(owner, "set_all_environment_variables")
        state = manga_env.HeadlessMangaState(owner)
        assert owner.manga_translator is state
    assert ho.OWNER_CONTRACT == ho.compute_owner_contract()


def test_headless_state_replays_the_widget_values_the_desktop_builds(tmp_path, monkeypatch):
    """``HeadlessMangaState`` fills ``ocr_provider_value`` / the Azure entries from the same config
    expressions ``_build_pyside6_interface`` uses (STARTUP_WIDGET_SOURCES, pinned to the desktop)."""
    import manga_env

    desktop = _module_text("manga_integration")
    for name, source in manga_env.STARTUP_WIDGET_SOURCES.items():
        assert desktop.count(source) == 1, f"the desktop no longer builds {name} like this: {source!r}"
    config = {"ocr_provider": "azure", "azure_vision_key": "k", "azure_vision_endpoint": "https://e/"}
    with headless_owner(tmp_path / "a", monkeypatch, copy.deepcopy(config)) as owner:
        state = manga_env.HeadlessMangaState(owner)
        assert (state.ocr_provider_value, state.azure_key_entry.text(), state.azure_endpoint_entry.text()) == (
            "azure", "k", "https://e/")
    monkeypatch.undo()
    with headless_owner(tmp_path / "b", monkeypatch, {"manga_ocr_provider": "google", "ocr_provider": "azure"}) as owner:
        state = manga_env.HeadlessMangaState(owner)
        assert state.ocr_provider_value == "google"
        assert state.azure_endpoint_entry.text() == "https://YOUR-RESOURCE.cognitiveservices.azure.com/"


def test_mobile_run_defaults_reach_the_runs_tab_state_and_detector(tmp_path, monkeypatch):
    """A fresh mobile config once the MANGA job put the phone defaults in
    (``manga_models.apply_mobile_run_defaults``) runs with the AOT ONNX inpainter, the v4-S INT8
    detector and the 1024 px limit Settings shows, not the desktop fallbacks the tab and the
    translator read for missing keys (anime_onnx, detector.onnx)."""
    import manga_env
    import manga_models
    from manga_translator import MangaTranslator

    config = manga_models.apply_mobile_run_defaults({}, force=True)
    with headless_owner(tmp_path / "phone", monkeypatch, copy.deepcopy(config)) as owner:
        state = manga_env.HeadlessMangaState(owner)
        assert state.local_model_type_value == "aot_onnx"
        manga = owner.config["manga_settings"]
        assert MangaTranslator._get_rtdetr_onnx_filename(None, manga["ocr"]) == "detector-v4-s_int8.onnx"
        assert manga["advanced"]["hd_strategy_resize_limit"] == 1024 and manga["advanced"]["max_workers"] == 1
    monkeypatch.undo()
    with headless_owner(tmp_path / "desktop", monkeypatch, {}) as owner:
        state = manga_env.HeadlessMangaState(owner)
        assert state.local_model_type_value == "anime_onnx"
        assert MangaTranslator._get_rtdetr_onnx_filename(None, {}) == "detector.onnx"


def test_every_main_gui_name_the_manga_code_reads_is_provided_like_the_desktop(tmp_path, monkeypatch):
    import headless_owner as ho

    names = set()
    for module, class_name, owners in ho.MANGA_OWNER_CONTRACT_MODULES:
        cls, _m = _class_nodes(_module_text(module), class_name)
        for node in ast.walk(cls):
            if isinstance(node, ast.Attribute) and isinstance(node.value, (ast.Name, ast.Attribute)) \
                    and ast.unparse(node.value) in owners and isinstance(node.ctx, ast.Load):
                names.add(node.attr)
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("getattr", "hasattr")
                    and len(node.args) >= 2 and ast.unparse(node.args[0]) in owners
                    and isinstance(node.args[1], ast.Constant)):
                names.add(node.args[1].value)
    tg = (SRC / "translator_gui.py").read_bytes().decode("utf-8-sig")
    shared = "".join(_module_text(m) for m in ("owner_state", "run_env", "settings_persistence",
                                                 "translation_pipeline", "text_jobs", "input_preparation"))
    with headless_owner(tmp_path, monkeypatch, {}) as owner:
        for name in sorted(names):
            desktop = bool(re.search(rf"self\.{re.escape(name)}\s*=|def {re.escape(name)}\(", tg + shared)) \
                or bool(re.search(rf"'{re.escape(name)}'", shared))
            if name in DESKTOP_ONLY_MAIN_GUI:
                assert desktop and not hasattr(owner, name), name
            elif hasattr(owner, name):
                continue
            else:
                assert not re.search(rf"self\.{re.escape(name)}\s*=\s*(?!None)", tg), \
                    f"TranslatorGUI sets main_gui.{name} at startup but HeadlessOwner has none"


# ---------------------------------------------------------------------------
# builtins.print hijack (process_image) and log routing
# ---------------------------------------------------------------------------


def _process_empty_page(tmp_path, monkeypatch, mobile):
    import builtins

    import manga_translator
    import unified_api_client

    if mobile:
        monkeypatch.setenv("GLOSSARION_MOBILE", "1")
    else:
        monkeypatch.delenv("GLOSSARION_MOBILE", raising=False)
    page = _write_pages(tmp_path, names=("p.png",))[0]
    original = builtins.print
    uc_print = vars(unified_api_client).get("print")
    with headless_owner(tmp_path / "o", monkeypatch, {"model": "gpt-4o-mini"}) as owner:
        if mobile:
            os.environ["GLOSSARION_MOBILE"] = "1"
        translator = manga_translator.MangaTranslator({"provider": "custom-api"}, None, owner,
                                                      log_callback=lambda *_a: None,
                                                      skip_inpainter_init=True, skip_ocr_init=True)
        out = str(Path(tmp_path) / "out" / "p.png")
        try:
            result = translator.process_image(page, out, precomputed_regions=[])
            hijacked = builtins.print is not original
            name = getattr(builtins.print, "__name__", "")
            uc_during = getattr(vars(unified_api_client).get("print"), "__name__", "")
        finally:
            translator.restore_print()
            restored = builtins.print is original
            uc_kept = vars(unified_api_client).get("print") is uc_print
            del translator  # __del__ restores again (desktop: in this env, before the reset below)
            gc.collect()
            builtins.print = original
            vars(unified_api_client)["print"] = uc_print
    return {"result": result, "hijacked": hijacked, "name": name, "restored": restored,
            "uc_during": uc_during, "uc_kept": uc_kept}


def test_print_hijack_is_desktop_only(tmp_path, monkeypatch):
    import unified_api_client

    assert getattr(vars(unified_api_client).get("print"), "__name__", "") == "_gui_print"
    desktop = _process_empty_page(tmp_path / "desktop", monkeypatch, mobile=False)
    assert desktop["result"]["success"] and desktop["hijacked"] and desktop["name"] == "manga_print"
    assert desktop["uc_during"] == "manga_print" and desktop["restored"]
    # desktop (unchanged): the restore leaves unified_api_client with the plain builtin print
    assert not desktop["uc_kept"]
    monkeypatch.undo()
    mobile = _process_empty_page(tmp_path / "mobile", monkeypatch, mobile=True)
    assert mobile["result"]["success"] and not mobile["hijacked"] and mobile["restored"]
    # mobile: unified_api_client keeps its own logger-routed print through and after the job
    assert mobile["uc_during"] == "_gui_print" and mobile["uc_kept"]


def test_mobile_restore_only_undoes_an_installed_manga_print(monkeypatch):
    import builtins

    import manga_translator

    monkeypatch.delenv("GLOSSARION_MOBILE", raising=False)
    assert manga_translator._manga_print_hijack_enabled() and manga_translator._manga_print_restore_enabled()
    monkeypatch.setenv("GLOSSARION_MOBILE", "1")
    assert not manga_translator._manga_print_hijack_enabled()
    assert not manga_translator._manga_print_restore_enabled()

    def manga_print(*args, **kwargs):  # what a desktop-style hijack would have installed
        return None

    monkeypatch.setattr(builtins, "print", manga_print)
    assert manga_translator._manga_print_restore_enabled()


# ---------------------------------------------------------------------------
# per-image pipeline trace: MangaTranslator.process_image at LEGACY_SHA vs the working tree
# ---------------------------------------------------------------------------
#
# Both modules run the same fixture page with recording stand-ins for the collaborators the
# pipeline reaches through module imports (bubble_detector.BubbleDetector, ocr_manager.OCRManager,
# local_inpainter.LocalInpainter) and a recording UnifiedClient. The collaborator calls (with
# their arguments), the result dict and the written pixels must be identical. The working-tree
# module is also run in mobile mode (GLOSSARION_MOBILE=1): same pipeline, print never hijacked.

PIPELINE_SCENARIOS = {
    "individual_batched": {"manga_full_page_context": False, "manga_visual_context_enabled": False},
    "full_page_context": {"manga_full_page_context": True, "manga_visual_context_enabled": False},
    "full_page_visual_context": {"manga_full_page_context": True, "manga_visual_context_enabled": True},
    "skip_inpainting": {"manga_full_page_context": False, "manga_visual_context_enabled": False,
                        "manga_skip_inpainting": True},
}
#: The fixture page's speech bubbles (x, y, w, h) and the text the stand-in OCR reads in each.
PIPELINE_BUBBLES = ((40, 60, 150, 100, "こんにちは", "Hello there"), (200, 300, 160, 120, "さようなら", "Goodbye now"))


@functools.lru_cache(maxsize=None)
def _legacy_manga_translator_file(directory):
    path = Path(directory) / f"manga_translator_legacy_{LEGACY_SHA[:12]}.py"
    if not path.exists():
        path.write_text(_cached_legacy("src/manga_translator.py"), encoding="utf-8")
    return str(path)


@pytest.fixture(scope="module")
def legacy_manga_translator(tmp_path_factory):
    """``manga_translator`` as of LEGACY_SHA, imported under its own module name."""
    import importlib.util

    try:
        _cached_legacy("src/manga_translator.py")
    except rp.Unavailable as exc:  # pragma: no cover - shallow clone
        if os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"):
            pytest.fail(str(exc))
        pytest.skip(str(exc))
    import manga_translator  # the live module first: both share TransateKRtoEN & co.

    name = f"manga_translator_legacy_{LEGACY_SHA[:12]}"
    path = _legacy_manga_translator_file(str(tmp_path_factory.mktemp("legacy_manga_translator")))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    module.__file__ = manga_translator.__file__  # project-relative lookups (fonts) see src/
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(name, None)


class PipelineTrace:
    """Thread-safe recorder of collaborator calls, normalised (tmp root, arrays -> shape/sum)."""

    def __init__(self, root):
        self.root = str(root)
        self.calls = {}
        self.lock = threading.Lock()

    def norm(self, value):
        import numpy as np

        if isinstance(value, str):
            return value.replace(self.root, "<ROOT>").replace(self.root.replace("\\", "/"), "<ROOT>")
        if isinstance(value, dict):
            return {str(k): self.norm(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self.norm(v) for v in value]
        if isinstance(value, np.ndarray):
            return f"<ndarray {list(value.shape)} {value.dtype} sum={int(value.astype('int64').sum())}>"
        if isinstance(value, (int, float, bool)) or value is None:
            return value
        return f"<{type(value).__name__}>"

    def add(self, who, *args, **kwargs):
        with self.lock:
            self.calls.setdefault(who, []).append(json.dumps(self.norm([list(args), kwargs]),
                                                             ensure_ascii=False, sort_keys=True))

    def snapshot(self):
        # calls from the pipeline's worker threads interleave freely; compare them per collaborator
        return {who: sorted(items) for who, items in self.calls.items()}


def make_pipeline_stubs(trace):
    """Recording ``bubble_detector`` / ``ocr_manager`` / ``local_inpainter`` modules."""
    bubble_detector = types.ModuleType("bubble_detector")

    class BubbleDetector:
        def __init__(self, *args, **kwargs):
            trace.add("BubbleDetector()", *args, **kwargs)
            self.rtdetr_onnx_loaded = False
            self.rtdetr_onnx_filename = None
            self.rtdetr_loaded = False
            self.model_loaded = False
            self.model = None

        def load_rtdetr_onnx_model(self, model_id=None, onnx_filename=None, **kwargs):
            trace.add("BubbleDetector.load_rtdetr_onnx_model", model_id=model_id, onnx_filename=onnx_filename, **kwargs)
            self.rtdetr_onnx_loaded = True
            self.rtdetr_onnx_filename = onnx_filename
            return True

        def detect_with_rtdetr_onnx(self, image_path=None, confidence=None, return_all_bubbles=False, **kwargs):
            trace.add("BubbleDetector.detect_with_rtdetr_onnx", image_path=image_path, confidence=confidence,
                      return_all_bubbles=return_all_bubbles, **kwargs)
            return {"bubbles": [], "text_bubbles": [tuple(b[:4]) for b in PIPELINE_BUBBLES], "text_free": []}

        def set_stop_flag(self, _flag):
            trace.add("BubbleDetector.set_stop_flag")

        def set_log_callback(self, _callback):
            trace.add("BubbleDetector.set_log_callback")

    bubble_detector.BubbleDetector = BubbleDetector

    ocr_manager = types.ModuleType("ocr_manager")

    class OCRResult:
        def __init__(self, text, bbox, confidence=1.0, vertices=None):
            self.text, self.bbox, self.confidence, self.vertices = text, bbox, confidence, vertices

    class OCRManager:
        def __init__(self, *args, **kwargs):
            trace.add("OCRManager()", *[a for a in args if not callable(a)])
            self.providers = {}

        def check_provider_status(self, name):
            trace.add("OCRManager.check_provider_status", name)
            return {"installed": True, "loaded": True}

        def load_provider(self, name, **kwargs):
            trace.add("OCRManager.load_provider", name, **kwargs)
            return True

        def get_provider(self, name):
            return self.providers.get(name)

        def detect_text(self, image, provider_name=None, **kwargs):
            trace.add("OCRManager.detect_text", image, provider_name, **kwargs)
            height, width = image.shape[:2]
            text = next(b[4] for b in PIPELINE_BUBBLES if (b[2], b[3]) == (width, height))
            return [OCRResult(text, (0, 0, width, height), 0.99, [(0, 0), (width, 0), (width, height), (0, height)])]

        def set_stop_flag(self, _flag):
            trace.add("OCRManager.set_stop_flag")

        def reset_stop_flags(self):
            trace.add("OCRManager.reset_stop_flags")

    ocr_manager.OCRManager = OCRManager
    ocr_manager.OCRResult = OCRResult

    local_inpainter = types.ModuleType("local_inpainter")

    class LocalInpainter:
        def __init__(self, *args, **kwargs):
            trace.add("LocalInpainter()", *args, **kwargs)
            self.model_loaded = False
            self.current_method = None

        def download_jit_model(self, method, *args, **kwargs):
            trace.add("LocalInpainter.download_jit_model", method, *args, **kwargs)
            path = os.path.join(trace.root, "models", f"{method}.onnx")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            open(path, "wb").close()
            return path

        def load_model(self, method, model_path=None, force_reload=False, **kwargs):
            trace.add("LocalInpainter.load_model", method, model_path, force_reload=force_reload, **kwargs)
            self.model_loaded = True
            self.current_method = method
            return True

        def load_model_with_retry(self, method, model_path=None, force_reload=False, **kwargs):
            return self.load_model(method, model_path, force_reload=force_reload, **kwargs)

        def inpaint(self, image, mask, refinement="normal", iterations=None, **kwargs):
            trace.add("LocalInpainter.inpaint", image, mask, refinement=refinement, iterations=iterations, **kwargs)
            cleaned = image.copy()
            cleaned[mask > 0] = 255
            return cleaned

        def set_stop_flag(self, _flag):
            trace.add("LocalInpainter.set_stop_flag")

        def set_log_callback(self, _callback):
            trace.add("LocalInpainter.set_log_callback")

        def unload(self):
            trace.add("LocalInpainter.unload")

    local_inpainter.LocalInpainter = LocalInpainter
    local_inpainter.HybridInpainter = type("HybridInpainter", (), {})
    local_inpainter.AnimeMangaInpaintModel = type("AnimeMangaInpaintModel", (), {})
    return {"bubble_detector": bubble_detector, "ocr_manager": ocr_manager, "local_inpainter": local_inpainter}


class PipelineClient:
    """A recording UnifiedClient: answers the batched / full-page / single-region prompts."""

    model = "gpt-4o-mini"
    api_key = "sk-test"

    def __init__(self, trace):
        self.trace = trace

    @staticmethod
    def _text(messages):
        parts = []
        for message in messages:
            content = message.get("content")
            if isinstance(content, list):
                content = " ".join(str(p.get("text", "")) for p in content if isinstance(p, dict))
            parts.append(f"{message.get('role')}: {content}")
        return "\n".join(parts)

    def _answer(self, messages):
        user = self._text([m for m in messages if m.get("role") == "user"])
        found = [b for b in PIPELINE_BUBBLES if b[4] in user]
        if len(found) > 1:
            return json.dumps({b[4]: b[5] for b in found}, ensure_ascii=False)
        return found[0][5] if found else ""

    def send(self, messages, temperature=None, max_tokens=None, **kwargs):
        self.trace.add("client.send", self._text(messages), temperature=temperature, max_tokens=max_tokens,
                       **{k: v for k, v in kwargs.items() if k != "context"})
        return self._answer(messages), "stop"

    def send_image(self, messages, image_data, temperature=None, max_tokens=None, **kwargs):
        self.trace.add("client.send_image", self._text(messages), len(image_data or b""), temperature=temperature,
                       max_tokens=max_tokens, **{k: v for k, v in kwargs.items() if k != "context"})
        return self._answer(messages), "stop"


def _write_pipeline_page(path):
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (400, 600), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    for x, y, w, h, _text, _translation in PIPELINE_BUBBLES:
        draw.ellipse((x, y, x + w, y + h), outline=(0, 0, 0), width=3)
        draw.rectangle((x + w // 4, y + h // 3, x + 3 * w // 4, y + 2 * h // 3), fill=(0, 0, 0))
    image.save(path)


def _pixels(path):
    from PIL import Image

    if not os.path.isfile(path):
        return None
    with Image.open(path) as image:
        return (image.mode, image.size, image.tobytes())


def _trace_process_image(tmp_path, monkeypatch, module, side, scenario, *, mobile=False):
    import builtins

    import unified_api_client

    root = Path(tmp_path) / side
    trace = PipelineTrace(root)
    for name, stub in make_pipeline_stubs(trace).items():
        monkeypatch.setitem(sys.modules, name, stub)
    cls = module.MangaTranslator
    with cls._inpaint_pool_lock:
        cls._inpaint_pool.clear()
    with cls._detector_pool_lock:
        cls._detector_pool.clear()
    cls._global_cancelled = False
    page = root / "pages" / "p1.png"
    page.parent.mkdir(parents=True, exist_ok=True)
    _write_pipeline_page(page)
    out = root / "pages" / "p1_translated" / "p1.png"
    config = {"model": "gpt-4o-mini", "api_key": "sk-test",
              "manga_settings": {"ocr": {"rtdetr_model_url": "ogkalu/comic-text-and-bubble-detector"}}}
    config.update(copy.deepcopy(scenario))
    original_print = builtins.print
    uc_print = vars(unified_api_client).get("print")
    hijacked = []
    with headless_owner(root / "owner", monkeypatch, config) as owner:
        if mobile:
            os.environ["GLOSSARION_MOBILE"] = "1"
        owner.save_config = lambda show_message=True: None

        def log(message, level="info"):
            if builtins.print is not original_print:
                hijacked.append(getattr(builtins.print, "__name__", "?"))

        translator = cls({"provider": "custom-api"}, PipelineClient(trace), owner, log_callback=log)
        try:
            result = translator.process_image(str(page), str(out))
        finally:
            translator.restore_print()
            del translator  # its __del__ restores once more; reset afterwards
            gc.collect()
            builtins.print = original_print
            vars(unified_api_client)["print"] = uc_print
    regions = result.get("regions") or []
    assert result.get("success") and [r.get("translated_text") for r in regions] == [b[5] for b in PIPELINE_BUBBLES], \
        result
    return {
        "calls": trace.snapshot(),
        "result": trace.norm({k: v for k, v in result.items() if not k.startswith("_")}),
        "output": _pixels(str(out)),
        "cleaned": _pixels(result.get("cleaned_image_path") or ""),
        "files": sorted(str(p.relative_to(root)).replace("\\", "/") for p in root.rglob("*")
                        if p.is_file() and "owner" not in p.relative_to(root).parts),
        "hijacked": sorted(set(hijacked)),
    }


@pytest.mark.parametrize("name", sorted(PIPELINE_SCENARIOS))
def test_process_image_trace_matches_the_legacy_pipeline(tmp_path, monkeypatch, legacy_manga_translator, name):
    import manga_translator

    scenario = PIPELINE_SCENARIOS[name]
    monkeypatch.delenv("GLOSSARION_MOBILE", raising=False)
    legacy = _trace_process_image(tmp_path, monkeypatch, legacy_manga_translator, "legacy", scenario)
    monkeypatch.undo()
    monkeypatch.delenv("GLOSSARION_MOBILE", raising=False)
    new = _trace_process_image(tmp_path, monkeypatch, manga_translator, "new", scenario)
    assert new["calls"] == legacy["calls"]
    assert new["result"] == legacy["result"]
    assert new["files"] == legacy["files"]
    assert new["output"] == legacy["output"] and new["output"] is not None
    assert new["cleaned"] == legacy["cleaned"]
    assert legacy["hijacked"] == new["hijacked"] == ["manga_print"]  # desktop routes print into the log
    called = set(new["calls"])
    assert {"BubbleDetector.detect_with_rtdetr_onnx", "OCRManager.detect_text"} <= called
    assert any(c.startswith("client.send") for c in called)
    assert ("LocalInpainter.inpaint" in called) == (not scenario.get("manga_skip_inpainting"))
    monkeypatch.undo()
    monkeypatch.delenv("GLOSSARION_MOBILE", raising=False)
    mobile = _trace_process_image(tmp_path, monkeypatch, manga_translator, "mobile", scenario, mobile=True)
    assert mobile["calls"] == new["calls"] and mobile["result"] == new["result"]
    assert mobile["output"] == new["output"] and mobile["cleaned"] == new["cleaned"]
    assert mobile["hijacked"] == []  # mobile never replaces builtins.print


# ---------------------------------------------------------------------------
# headless runner (the MANGA job entry point)
# ---------------------------------------------------------------------------


def _runner_env(tmp_path, monkeypatch, rec, translator_kwargs=None):
    import manga_runner
    import manga_translator
    import unified_api_client

    fake = make_fake_translator(rec, **(translator_kwargs or {}))
    monkeypatch.setattr(unified_api_client, "UnifiedClient", make_fake_client(rec))
    monkeypatch.setitem(sys.modules, "ImageRenderer", make_image_renderer_stub(rec))
    monkeypatch.setitem(sys.modules, "ocr_manager", make_ocr_stub(rec))
    monkeypatch.setattr(manga_runner, "MangaTranslator", fake)
    monkeypatch.setattr(manga_translator, "MangaTranslator", fake)
    return fake


def test_headless_runner_translates_pages_and_packages_cbz(tmp_path, monkeypatch):
    import manga_runner

    rec = Recorder(tmp_path)
    _runner_env(tmp_path, monkeypatch, rec)
    pages = _write_pages(tmp_path)
    progress = []
    with headless_owner(tmp_path / "o", monkeypatch, {"model": "gpt-4o-mini", "api_key": "sk",
                                                      "manga_create_cbz_at_end": True}) as owner:
        result = manga_runner.run_manga_batch(owner, host=rec.host, files=pages, image_range="1,3",
                                              progress=lambda *a, **k: progress.append((a, k)))
    assert result["ok"] and result["completed"] == 2 and result["failed"] == 0 and result["total"] == 2
    assert [os.path.basename(p) for p in result["outputs"]] == ["1.png", "3.png"]
    assert all(os.path.isfile(p) and p.endswith(os.path.join(os.path.basename(p)[:-4] + "_translated",
                                                             os.path.basename(p))) for p in result["outputs"])
    assert [os.path.basename(p) for p in result["cbz_paths"]] == ["manga_translated.cbz"]
    with zipfile.ZipFile(result["cbz_paths"][0]) as zf:
        assert len(zf.namelist()) == 2
    assert progress and progress[-1][1]["label"].startswith("Complete!")
    assert [e[1] for e in rec.events if e[0] == "process_image"] == ["1.png", "3.png"]


def test_headless_runner_refuses_like_the_start_button(tmp_path, monkeypatch):
    import manga_runner

    rec = Recorder(tmp_path)
    _runner_env(tmp_path, monkeypatch, rec)
    pages = _write_pages(tmp_path)
    with headless_owner(tmp_path / "o", monkeypatch, {"model": "gpt-4o-mini", "api_key": "sk"}) as owner:
        with pytest.raises(manga_runner.MangaRunError, match="select manga images"):
            manga_runner.HeadlessMangaRunner(owner).run()
        with pytest.raises(manga_runner.MangaRunError, match="Invalid image range"):
            manga_runner.HeadlessMangaRunner(owner, files=pages, image_range="x").run()
        with pytest.raises(manga_runner.MangaRunError, match="does not include"):
            manga_runner.HeadlessMangaRunner(owner, files=pages, skipped=pages).run()
        result = manga_runner.HeadlessMangaRunner(owner, files=pages).run()
        assert result["ok"]
    with headless_owner(tmp_path / "o2", monkeypatch, {"model": "gpt-4o-mini", "manga_ocr_provider": "azure"}) as owner:
        result = manga_runner.HeadlessMangaRunner(owner, host=rec.host, files=pages).run()
    assert not result["ok"] and "Azure credentials" in result["error"]


def test_headless_runner_stop_reaches_the_stop_protocol(tmp_path, monkeypatch):
    import manga_runner

    import unified_api_client

    rec = Recorder(tmp_path)
    pages = _write_pages(tmp_path, names=tuple(f"{i}.png" for i in range(1, 7)))
    monkeypatch.setenv("GRACEFUL_STOP", "x")
    monkeypatch.delenv("GRACEFUL_STOP")  # an earlier test's flag must not be the ambient value checked below
    monkeypatch.setitem(sys.modules, "psutil", None)  # never terminate the test process' children
    monkeypatch.setattr(unified_api_client, "hard_cancel_all", lambda: rec.add("hard_cancel_all"))
    with headless_owner(tmp_path / "o", monkeypatch, {"model": "gpt-4o-mini", "api_key": "sk",
                                                      "graceful_stop": False}) as owner:
        runner = manga_runner.HeadlessMangaRunner(owner, host=rec.host, files=pages)
        _runner_env(tmp_path, monkeypatch, rec, {"stop_after": 2, "stop_flag_owner": None})

        class StopAfterTwo(manga_runner.MangaTranslator):
            def process_image(self, image_path, output_path=None, **kwargs):
                result = super().process_image(image_path, output_path, **kwargs)
                if sum(1 for e in rec.events if e and e[0] == "process_image") == 2:
                    runner.request_stop()
                return result

        monkeypatch.setattr(manga_runner, "MangaTranslator", StopAfterTwo)
        result = runner.run()
    assert result["stopped"] and not result["ok"]
    assert len(result["outputs"]) == 2
    assert os.environ.get("GRACEFUL_STOP") != "1"
    assert runner.is_running is False


def test_headless_runner_glossary_only_flag_is_reported(tmp_path, monkeypatch):
    import manga_runner

    rec = Recorder(tmp_path)
    _runner_env(tmp_path, monkeypatch, rec)
    pages = _write_pages(tmp_path)
    calls = []
    with headless_owner(tmp_path / "o", monkeypatch, {"model": "gpt-4o-mini", "api_key": "sk"}) as owner:
        runner = manga_runner.HeadlessMangaRunner(owner, host=rec.host, files=pages, glossary_only=True)
        monkeypatch.setattr(runner, "_run_manga_glossary_workflow",
                            lambda glossary_only_run=False: calls.append(glossary_only_run))
        result = runner.run()
    assert calls == [True] and result["glossary_only"] is True
    assert runner._manga_glossary_only_run is False  # the worker's finally resets it, like desktop


# ---------------------------------------------------------------------------
# contract helpers
# ---------------------------------------------------------------------------


def test_apply_rendering_settings_matches_the_legacy_method(tmp_path, monkeypatch):
    import manga_env

    config = {"manga_font_size_mode": "multiplier", "manga_font_size_multiplier": 1.4, "manga_bg_opacity": 90,
              "manga_skip_inpainting": False, "manga_safe_area_enabled": True, "manga_safe_area_scale": 0.8,
              "replicate_api_key": "r8", "manga_settings": {"inpainting": {"method": "cloud"},
                                                            "advanced": {"concise_logs": True}},
              "output_directory": str(tmp_path / "out")}
    results = []
    for side in ("legacy", "new"):
        rec = Recorder(tmp_path)
        with headless_owner(tmp_path / side, monkeypatch, copy.deepcopy(config)) as owner:
            state = manga_env.HeadlessMangaState(owner, host=rec.host)
            translator = make_fake_translator(rec)({"provider": "custom-api"}, None, owner)
            state.translator = translator
            if side == "legacy":
                _legacy_tab_function("_apply_rendering_settings", {"os": os})(state)
            else:
                manga_env.apply_rendering_settings(translator, owner, state=state)
            results.append((rec.events, os.environ.get("OUTPUT_DIRECTORY")))
        monkeypatch.undo()
    assert results[0] == results[1]
    assert any(e[:2] == ["translator.set", "inpaint_mode"] and e[2] == "cloud" for e in results[1][0])


def test_build_ocr_config_and_glossary_env_match_the_tab(tmp_path, monkeypatch):
    import manga_env

    creds = tmp_path / "c.json"
    creds.write_text("{}", encoding="utf-8")
    assert manga_env.build_ocr_config({"manga_ocr_provider": "google", "google_vision_credentials": str(creds)}) == {
        "provider": "google", "google_credentials_path": str(creds)}
    assert manga_env.build_ocr_config({"manga_ocr_provider": "google", "google_cloud_credentials": "nope"}) == {
        "provider": "google"}
    assert manga_env.build_ocr_config({"azure_vision_key": "k", "azure_vision_endpoint": "e"}, "azure") == {
        "provider": "azure", "azure_key": "k", "azure_endpoint": "e"}
    assert manga_env.build_ocr_config({}) == {"provider": "custom-api"}
    # Without an explicit provider the helper resolves it like the desktop tab does before any
    # worker reads ocr_provider_value (STARTUP_WIDGET_SOURCES: manga_ocr_provider, then
    # ocr_provider, then custom-api), not with the method's never-used getattr fallback.
    only_generic = {"ocr_provider": "azure", "azure_vision_key": "k", "azure_vision_endpoint": "https://e/"}
    assert manga_env.build_ocr_config(only_generic) == {
        "provider": "azure", "azure_key": "k", "azure_endpoint": "https://e/"}
    assert manga_env.build_ocr_config({"manga_ocr_provider": "", "ocr_provider": "azure"}) == {"provider": "azure"}
    assert manga_env.build_ocr_config({"manga_ocr_provider": "google", "ocr_provider": "azure"}) == {
        "provider": "google"}
    for index, config in enumerate((only_generic, {"manga_ocr_provider": "", "ocr_provider": "azure"},
                                    {"manga_ocr_provider": "google", "ocr_provider": "azure"}, {})):
        with headless_owner(tmp_path / f"tab{index}", monkeypatch, copy.deepcopy(config)) as owner:
            state = manga_env.HeadlessMangaState(owner)  # the tab's widget values (desktop expressions)
            assert manga_env.build_ocr_config(copy.deepcopy(config)) == state._build_manga_worker_ocr_config()
        monkeypatch.undo()
    with headless_owner(tmp_path / "o", monkeypatch, {"glossary_target_language": "French",
                                                      "use_glossary_keys": False}) as owner:
        before = dict(os.environ)
        previous = manga_env.prepare_manga_glossary_env(owner)
        assert os.environ["GLOSSARY_TARGET_LANGUAGE"] == "French"
        assert os.environ["GLOSSARY_SYSTEM_PROMPT"] == manga_env.default_manga_glossary_prompt()
        assert os.environ["SAVE_GLOSSARY_IN_OUTPUT"] == "0"
        manga_env.restore_manga_glossary_env(previous)
        assert dict(os.environ) == before


#: A config whose font-sizing values all differ from every preset (the button must still set them).
_PRESET_CONFIG = {"model": "gpt-4o-mini", "manga_strict_text_wrapping": True, "manga_max_font_size": 30,
                  "manga_settings": {"font_sizing": {"algorithm": "smart", "line_spacing": 1.0, "prefer_larger": False,
                                                     "bubble_size_factor": True, "min_size": 5, "max_size": 30},
                                     "rendering": {"auto_fit_style": "balanced", "auto_min_size": 5,
                                                   "auto_max_size": 30}}}


def test_font_preset_updates_are_what_the_preset_button_saves(tmp_path, monkeypatch):
    import manga_env
    import manga_files_core
    import manga_settings_defaults as msd

    env_before = dict(os.environ)
    updates = {preset: msd.font_preset_updates(preset, copy.deepcopy(_PRESET_CONFIG)) for preset in manga_env.FONT_PRESETS}
    assert dict(os.environ) == env_before, "font_preset_updates must leave the process env as it was"
    assert manga_env.font_preset_updates("balanced", copy.deepcopy(_PRESET_CONFIG)) == updates["balanced"]
    # absolute: the same entries for every preset and whatever the current config holds
    keys = set(updates["small"])
    assert keys and all(set(u) == keys for u in updates.values())
    assert {k: v for k, v in msd.font_preset_updates("large").items()} == updates["large"]
    assert {"manga_settings.font_sizing.prefer_larger", "manga_settings.rendering.auto_fit_style",
            "manga_strict_text_wrapping", "manga_max_font_size"} <= keys
    assert manga_env.font_preset_updates("not-a-preset", _PRESET_CONFIG) == {}
    for preset in manga_env.FONT_PRESETS:
        # the desktop button: the LEGACY _set_font_preset on a tab over the same config, against a
        # plain save of the same tab
        saved = []
        for apply_preset in (False, True):
            with headless_owner(tmp_path / preset / str(int(apply_preset)), monkeypatch,
                                copy.deepcopy(_PRESET_CONFIG)) as owner:
                owner.save_config = lambda show_message=True: None
                state = manga_env.HeadlessMangaState(owner)
                if apply_preset:
                    _legacy_tab_function("_set_font_preset",
                                         {"ImageRenderer": manga_files_core.ImageRenderer})(state, preset)
                else:
                    state._save_rendering_settings()
                saved.append(manga_env._flatten_manga_config(json.loads(json.dumps(owner.config, default=str))))
            monkeypatch.undo()
        plain, pressed = saved
        changed = {k for k, v in pressed.items() if plain.get(k, object()) != v}
        assert changed <= keys, sorted(changed - keys)
        assert {k: pressed[k] for k in keys} == updates[preset], preset


def test_import_ocr_session_feeds_the_next_run(tmp_path, monkeypatch):
    import manga_env
    import manga_ocr_io

    pages = _write_pages(tmp_path)
    document = manga_ocr_io.create_document([manga_ocr_io.make_page(
        pages[1], [{"text": "src", "translated_text": "dst", "bounding_box": [1, 2, 3, 4]}])], workflow="automatic")
    path = tmp_path / "session.json"
    manga_ocr_io.write_document(str(path), document)
    with headless_owner(tmp_path / "o", monkeypatch, {}) as owner:
        state = manga_env.HeadlessMangaState(owner)
        state.selected_files = list(pages)
        matches = manga_env.import_ocr_session(state, str(path))
        # keys are the page map's normalised paths (os.path.normcase on Windows)
        assert [os.path.normcase(p) for p in matches] == [os.path.normcase(pages[1])]
        regions = state._resolve_imported_ocr_regions(pages[1])
        assert regions and regions[0].text == "src"
        assert state._resolve_imported_ocr_regions(pages[0]) is None
        assert manga_env.import_ocr_session(state, document=manga_ocr_io.create_document([], workflow="x")) == {}
        assert state._imported_ocr_document is None


def test_files_logic_runs_headless(tmp_path, monkeypatch):
    import manga_env

    root = tmp_path / "lib"
    a = _write_pages(root, names=("1.png", "10.png", "2.png"), sub="series/ch1")
    b = _write_pages(root, names=("1.png",), sub="series/ch2")
    with headless_owner(tmp_path / "o", monkeypatch, {"manga_split_first_level_subfolders": True}) as owner:
        state = manga_env.HeadlessMangaState(owner)
        state._add_dropped_manga_paths([str(root / "series")])
        # the folder walk queues ch1 then ch2; the default numeric sort is by file NAME (stable),
        # so the two 1.png pages come first (desktop behaviour)
        assert [os.path.relpath(p, root) for p in state.selected_files] == [
            os.path.join("series", "ch1", "1.png"), os.path.join("series", "ch2", "1.png"),
            os.path.join("series", "ch1", "2.png"), os.path.join("series", "ch1", "10.png")]
        groups = state._manga_process_groups_for_paths()
        assert [g["name"] for g in groups] == ["ch1", "ch2"]
        state.manga_image_range_value = "2-3"
        files, error = state._manga_range_filtered_files()
        assert error is None and files == [b[0], a[2]]
        state._toggle_skip_processing_for_path(a[2])
        assert state._manga_range_filtered_files()[0] == [b[0]]
        state._persist_selected_files()
        assert owner.config["manga_selected_files"] == state.selected_files
        assert owner.config["manga_skipped_processing_files"] == [state._skip_key_for_path(a[2])]
        fresh = manga_env.HeadlessMangaState(owner)
        fresh._load_persisted_files()
        assert fresh.selected_files == state.selected_files and fresh.file_listbox.currentRow() == 0
        assert b[0] in fresh.selected_files


# ---------------------------------------------------------------------------
# desktop smoke (Qt tier): the tab and the settings dialog still build
# ---------------------------------------------------------------------------


def test_desktop_manga_tab_and_settings_dialog_build_offscreen(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    code = textwrap.dedent(r"""
        import os, sys, types
        from PySide6.QtWidgets import QApplication, QDialog, QScrollArea, QWidget
        app = QApplication.instance() or QApplication([])
        import ImageRenderer, manga_integration, manga_files_core, manga_env, manga_runner
        ImageRenderer._preload_shared_bubble_detector = lambda *_a, **_k: None
        ImageRenderer._preload_shared_inpainter = lambda *_a, **_k: True
        class StateManager:
            def __init__(self, path): self.states = {}; self._timer_lock = None
            def get_state(self, p): return self.states.get(p, {})
            def set_state(self, p, s, save=True): self.states[p] = s
            def update_state(self, p, u, save=True): self.states.setdefault(p, {}).update(u)
            def flush(self): pass
            def flush_async(self): pass
        manga_integration.ImageStateManager = StateManager
        from headless_owner import HeadlessOwner
        owner = HeadlessOwner({'model': 'gpt-4o-mini', 'api_key': 'sk'}, host=types.SimpleNamespace(log=lambda *a, **k: None))
        owner.save_config = lambda show_message=True: True
        tab = manga_integration.MangaTranslationTab(QWidget(), owner, QDialog(), QScrollArea())
        for _ in range(5):
            app.processEvents()
        assert type(tab).__mro__[1:6] == (manga_runner.MangaRunMixin, manga_env.MangaOcrSessionMixin,
                                          manga_env.MangaEnvMixin, manga_files_core.MangaFilesMixin,
                                          manga_files_core.MangaHooksMixin)
        assert tab.ocr_provider_value == 'custom-api' and tab.inpaint_method_value == 'local'
        assert manga_integration._natural_sort_key is manga_files_core._natural_sort_key
        assert manga_integration._IS_WINDOWS is manga_runner._IS_WINDOWS
        from manga_settings_dialog import MangaSettingsDialog
        dialog = MangaSettingsDialog(None, owner, owner.config)
        assert dialog.default_settings['ocr']['detector_type'] == 'rtdetr_onnx'
        dialog.close()
        print('ok', flush=True)  # os._exit skips the stdio flush
        os._exit(0)
    """)
    env = dict(os.environ, PYTHONIOENCODING="utf-8", QT_QPA_PLATFORM="offscreen", HOME=str(tmp_path),
               USERPROFILE=str(tmp_path), GLOSSARION_LIBRARY_DIR=str(tmp_path / "library"),
               CONFIG_FILE=str(tmp_path / "config.json"), YOLO_CONFIG_DIR=str(tmp_path / "yolo"),
               GLOSSARION_HTTP_LOG="0")
    env["PYTHONPATH"] = os.pathsep.join([str(SRC)] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p])
    proc = subprocess.run([sys.executable, "-c", code], cwd=str(tmp_path), env=env, capture_output=True,
                          text=True, encoding="utf-8", errors="replace", timeout=600)
    built = proc.stdout.strip().endswith("ok")
    # Windows: torch / onnxruntime DLLs loaded by the tab rarely fault in DLL_PROCESS_DETACH while
    # os._exit tears the process down; that is after the checks above have printed 'ok'.
    teardown_fault = sys.platform == "win32" and proc.returncode in (3221225477, -1073741819)
    assert built and (proc.returncode == 0 or teardown_fault), proc.stdout[-3000:] + proc.stderr[-4000:]
