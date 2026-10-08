"""U6 parity for the QA Scanner pieces moved into ``qa_scan_runtime`` (QA_Scanner_GUI rewired).

What moved (milestone U6, plan section 2; shared-core design P6 "qa_scan_runtime additions"),
byte-for-byte from ``QA_Scanner_GUI`` at ``U6_BASE_SHA``:

* the pure helpers (QG 176-397): ``_qa_owner_output_mode``, ``_qa_owner_uses_truncation_context``,
  ``_qa_vision_ocr_source_path``, ``_normalize_target_language``, ``_normalize_source_language``,
  ``check_epub_folder_match``, ``normalize_name_for_comparison``;
* the report search of ``QAScannerMixin.open_latest_qa_report`` (QG 488-540) as
  ``find_latest_qa_report(override_dir, last_report_path)`` (the method keeps its dialogs);
* the default AI-truncation prompt (QG 4693-4703) as ``DEFAULT_AI_TRUNCATION_PROMPT`` and the
  Custom-mode defaults (QG 1377-1388) as ``DEFAULT_CUSTOM_MODE_SETTINGS``.

QA_Scanner_GUI imports every moved name (module globals, same objects). Also in U6:
``scan_html_folder.DEFAULT_REFUSAL_PATTERNS`` is key_pool_service's list (U4 carry-over), and
the mobile forcing: when ``mobile_runtime`` reports no process pools (Glossarion Mobile),
``prepare_qa_scan_settings`` sets ``use_thread_executor`` and ``apply_qa_scan_env_from_settings``
mirrors ``QA_USE_THREAD_EXECUTOR=1`` and a small ``AI_HUNTER_MAX_WORKERS``; desktop unchanged.

Checks (legacy = ``git show U6_BASE_SHA``; legacy functions/methods are executed from that
source, so no Qt is needed):

* verbatim moves, the GUI / runtime / scanner changed only in the documented places (since
  2026-10-08 also the owner-approved chat-QA opt-in ``allow_direct_text=False`` of
  ``run_qa_scan_path`` / ``run_bulk_qa_scan``: the default still refuses Direct Text, only the
  keyword lets it through, and no desktop caller passes it);
* differential fuzz (``PARITY_U6_QA_STATES``, default 500 seeded states per function) of the
  pure helpers, the OCR-source resolver and ``open_latest_qa_report`` on random trees
  (``PARITY_U6_QA_FS_STATES``, default 150), ``apply_qa_scan_env_from_settings`` /
  ``prepare_qa_scan_settings`` (desktop == legacy; mobile == legacy + the forcing);
* import hygiene (no PySide6 / translator_gui; Python 3.10 syntax);
* end to end: a quick scan of a U3-E2E-style output workspace (the 12-chapter self-test EPUB,
  translated by the E2E fake model) through ``run_qa_scan_path`` with a HeadlessOwner, once with
  the desktop environment (process pool) and once with the mobile environment (threads, forced
  workers): the report folder and ``translation_progress.json`` must be identical.
  ``PARITY_U6_QA_E2E_SANDBOX=<kept e2e sandbox>`` scans a real U3 E2E run (``e2e --keep``)
  instead of the generated copy; ``PARITY_U6_QA_E2E=0`` skips the test. langdetect is seeded
  in every process of both runs (an unseeded scan is not reproducible: DISCREPANCIES U6).

Run (repository root)::

    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic \\
        tests/test_qa_runtime_additions.py
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import random
import re
import shutil
import string
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
TESTS = REPO_ROOT / "tests"
MOBILE_APP = SRC / "mobile" / "app"
MOBILE_TOOLS = SRC / "mobile" / "tools"
for _p in (str(TESTS), str(SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

#: Parent commit of the U6 QA move (U5 on main); the legacy oracle is ``git show`` of it.
U6_BASE_SHA = "e28e3a0f8e2ef3bd2aea59c87049b336ea469ac0"
STATES = max(1, int(os.environ.get("PARITY_U6_QA_STATES", "500")))
FS_STATES = max(1, int(os.environ.get("PARITY_U6_QA_FS_STATES", "150")))
SEED = int(os.environ.get("PARITY_U6_QA_SEED", "6006"))

import key_pool_service  # noqa: E402
import qa_scan_runtime  # noqa: E402

MOVED_HELPERS = (
    "_qa_owner_output_mode",
    "_qa_owner_uses_truncation_context",
    "_qa_vision_ocr_source_path",
    "_normalize_target_language",
    "_normalize_source_language",
    "check_epub_folder_match",
    "normalize_name_for_comparison",
)
#: every name QA_Scanner_GUI imports from qa_scan_runtime since U6
GUI_REEXPORTS = MOVED_HELPERS + ("find_latest_qa_report", "DEFAULT_AI_TRUNCATION_PROMPT",
                                 "DEFAULT_CUSTOM_MODE_SETTINGS")
#: new top-level names of qa_scan_runtime (besides the moved helpers)
RUNTIME_NEW_NAMES = {"MOBILE_QA_MAX_WORKERS", "mobile_qa_forcing_active", "mobile_qa_max_workers",
                     "mobile_qa_env_overrides", "find_latest_qa_report", "DEFAULT_AI_TRUNCATION_PROMPT",
                     "DEFAULT_CUSTOM_MODE_SETTINGS"}
#: U7 additions (QA_Scanner_GUI's bulk loop / loader / reset and translator_gui's QA stop flags;
#: pinned against the frozen code by tests/test_u7_tool_cores.py)
U7_BASE_SHA = "41814faa95e273e956870bd3aac5a2c6fb7d66b1"
#: U9 on main, the last commit before the owner-approved chat-QA opt-in (2026-10-08):
#: ``allow_direct_text=False`` on run_qa_scan_path / run_bulk_qa_scan, passed only by Glossarion
#: Mobile's chat QA job (tests/parity/DISCREPANCIES.md "Chat QA of Direct Text workspaces").
DEVFIX_BASE_SHA = "cddd73a4f5d057a03810d00b5787f37c14017130"
RUNTIME_U7_NAMES = {"load_current_qa_settings", "reset_qa_cancel_flags", "next_qa_stop_phase",
                    "apply_qa_graceful_stop_flags", "apply_qa_force_stop_flags", "clear_qa_stop_flags",
                    "run_bulk_qa_scan"}


def _mask_u7_run_qa_scan(method):
    """``run_qa_scan`` with the three U7-edited spots blanked: the cancel-flag reset (try block before,
    import + call after), the ``_load_current_qa_settings`` body and the ``run_scan`` worker body."""
    import copy

    method = copy.deepcopy(method)
    for node in ast.walk(method):
        if isinstance(node, ast.FunctionDef) and node.name in ("_load_current_qa_settings", "run_scan"):
            node.body = [ast.Pass()]
        for field in ("body", "orelse", "finalbody"):
            stmts = getattr(node, field, None)
            if not isinstance(stmts, list):
                continue
            kept = []
            for stmt in stmts:
                src = ast.unparse(stmt) if isinstance(stmt, ast.stmt) else ""
                if isinstance(stmt, ast.Try) and any(
                        ast.unparse(s) == "os.environ['TRANSLATION_CANCELLED'] = '0'" for s in stmt.body):
                    kept.append(ast.Pass())
                    continue
                if src in ("from qa_scan_runtime import reset_qa_cancel_flags", "reset_qa_cancel_flags()"):
                    if src.startswith("from"):
                        kept.append(ast.Pass())
                    continue
                kept.append(stmt)
            setattr(node, field, kept)
    return ast.dump(method)
MOBILE_ENV_KEYS = ("GLOSSARION_MOBILE", "GLOSSARION_NO_PROCESSES", "FLET_PLATFORM")


@pytest.fixture(autouse=True)
def _desktop_env(monkeypatch, tmp_path):
    """Desktop process env by default; never let a scan or lookup leave tmp_path."""
    for key in MOBILE_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("GLOSSARION_LIBRARY_DIR", str(tmp_path / "_library"))
    saved = dict(os.environ)
    cwd = os.getcwd()
    yield
    os.chdir(cwd)
    os.environ.clear()
    os.environ.update(saved)


# =============================================================================================
# legacy oracle
# =============================================================================================

_GIT_CACHE: dict = {}


def git_text(relpath: str, sha: str = U6_BASE_SHA) -> str:
    key = (relpath, sha)
    if key not in _GIT_CACHE:
        try:
            data = subprocess.run(["git", "show", f"{sha}:{relpath}"], cwd=str(REPO_ROOT),
                                  capture_output=True, check=True).stdout
        except (OSError, subprocess.CalledProcessError) as exc:
            if os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"):
                raise
            pytest.skip(f"legacy source {relpath}@{sha[:8]} unavailable: {exc}")
        _GIT_CACHE[key] = data.decode("utf-8-sig").replace("\r\n", "\n")
    return _GIT_CACHE[key]


def src_text(name: str) -> str:
    return (SRC / name).read_text(encoding="utf-8-sig").replace("\r\n", "\n")


def _top_defs(tree) -> dict:
    out = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            out[node.name] = node
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    out[target.id] = node
    return out


def _class(tree, name):
    return next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)


def _member(cls_node, name):
    return next(n for n in cls_node.body if isinstance(n, ast.FunctionDef) and n.name == name)


def _exec_defs(nodes, filename: str, namespace: dict) -> dict:
    module = ast.Module(body=list(nodes), type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, filename, "exec"), namespace)
    return namespace


@pytest.fixture(scope="module")
def legacy_gui_tree():
    return ast.parse(git_text("src/QA_Scanner_GUI.py"))


@pytest.fixture(scope="module")
def new_gui_tree():
    return ast.parse(src_text("QA_Scanner_GUI.py"))


@pytest.fixture(scope="module")
def legacy_helpers(legacy_gui_tree):
    """The seven helpers executed from the legacy QA_Scanner_GUI source (``__file__`` = its path)."""
    defs = _top_defs(legacy_gui_tree)
    ns = {"os": os, "re": re, "__file__": str(SRC / "QA_Scanner_GUI.py"), "__name__": "legacy_qa_gui_u6"}
    return types.SimpleNamespace(**{k: v for k, v in _exec_defs(
        [defs[name] for name in MOVED_HELPERS], str(SRC / "QA_Scanner_GUI.py"), ns).items()
        if k in MOVED_HELPERS})


@pytest.fixture(scope="module")
def legacy_runtime(tmp_path_factory):
    """qa_scan_runtime at U6_BASE_SHA, imported as a separate module."""
    folder = tmp_path_factory.mktemp("legacy_qa_runtime")
    path = folder / "legacy_qa_scan_runtime_u6.py"
    path.write_text(git_text("src/qa_scan_runtime.py"), encoding="utf-8")
    spec = importlib.util.spec_from_file_location("legacy_qa_scan_runtime_u6", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# =============================================================================================
# 1. verbatim moves and documented edits
# =============================================================================================

def test_moved_helpers_are_verbatim_and_in_source_order(legacy_gui_tree):
    legacy = _top_defs(legacy_gui_tree)
    new_tree = ast.parse(src_text("qa_scan_runtime.py"))
    new = _top_defs(new_tree)
    for name in MOVED_HELPERS:
        assert ast.unparse(new[name]) == ast.unparse(legacy[name]), f"qa_scan_runtime.{name} differs"
    order = [n.name for n in new_tree.body if isinstance(n, ast.FunctionDef) and n.name in MOVED_HELPERS]
    assert order == list(MOVED_HELPERS)


def test_constants_equal_the_dialog_literals(legacy_gui_tree):
    mixin = _class(legacy_gui_tree, "QAScannerMixin")
    prompt = custom = None
    for method in mixin.body:
        if not isinstance(method, ast.FunctionDef):
            continue
        for node in ast.walk(method):
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                if node.targets[0].id == "_ai_trunc_default_prompt" and method.name == "show_qa_scanner_settings":
                    prompt = ast.literal_eval(node.value)
                if node.targets[0].id == "custom_settings" and method.name == "run_qa_scan" \
                        and isinstance(node.value, ast.Dict):
                    custom = ast.literal_eval(node.value)
    assert prompt and qa_scan_runtime.DEFAULT_AI_TRUNCATION_PROMPT == prompt
    assert custom and qa_scan_runtime.DEFAULT_CUSTOM_MODE_SETTINGS == custom
    assert list(qa_scan_runtime.DEFAULT_CUSTOM_MODE_SETTINGS) == list(custom)
    # the scanner's built-in fallback prompt is the same text (recorded duplicate)
    scan_constants = {n.value for n in ast.walk(ast.parse(src_text("scan_html_folder.py")))
                      if isinstance(n, ast.Constant) and isinstance(n.value, str)
                      and n.value.startswith("You are a strict translation quality analyst.")}
    assert scan_constants == {prompt}


def _replace_stmt(text: str, old_node, new_text: str) -> str:
    old_text = ast.unparse(old_node)
    assert text.count(old_text) == 1, old_text[:80]
    return text.replace(old_text, new_text)


def test_gui_changed_only_in_the_documented_places(legacy_gui_tree, new_gui_tree):
    legacy_body = [n for n in legacy_gui_tree.body
                   if not (isinstance(n, ast.FunctionDef) and n.name in MOVED_HELPERS)]
    new_body = list(new_gui_tree.body)
    # the one added import, right after the legacy ``from qa_scan_runtime import is_html_like_path``
    added = [n for n in new_body if isinstance(n, ast.ImportFrom) and n.module == "qa_scan_runtime"
             and any(a.name == "find_latest_qa_report" for a in n.names)]
    assert len(added) == 1
    assert sorted(a.name for a in added[0].names) == sorted(GUI_REEXPORTS)
    assert all(a.asname is None for a in added[0].names)
    new_body.remove(added[0])
    assert len(new_body) == len(legacy_body)
    for old, new in zip(legacy_body, new_body):
        if isinstance(old, ast.ClassDef) and old.name == "QAScannerMixin":
            continue
        assert ast.dump(old) == ast.dump(new), getattr(old, "name", ast.unparse(old)[:60])
    defined = {n.name for n in new_gui_tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    assert not defined & set(MOVED_HELPERS)

    old_cls, new_cls = _class(legacy_gui_tree, "QAScannerMixin"), _class(new_gui_tree, "QAScannerMixin")
    assert [getattr(n, "name", None) for n in old_cls.body] == [getattr(n, "name", None) for n in new_cls.body]
    edited = {"open_latest_qa_report", "run_qa_scan", "show_qa_scanner_settings"}
    for old, new in zip(old_cls.body, new_cls.body):
        if getattr(old, "name", None) not in edited:
            assert ast.dump(old) == ast.dump(new), getattr(old, "name", "?")

    def assign(method, target):
        return next(n for n in ast.walk(method) if isinstance(n, ast.Assign)
                    and isinstance(n.targets[0], ast.Name) and n.targets[0].id == target)

    old_run, new_run = _member(old_cls, "run_qa_scan"), _member(new_cls, "run_qa_scan")
    # U6 edit, checked on the U7 base (U7 moved three more spots of this method; see below)
    u7_base_cls = _class(ast.parse(git_text("src/QA_Scanner_GUI.py", U7_BASE_SHA)), "QAScannerMixin")
    u7_base_run = _member(u7_base_cls, "run_qa_scan")
    expected = _replace_stmt(ast.unparse(old_run), assign(old_run, "custom_settings"),
                             "custom_settings = dict(DEFAULT_CUSTOM_MODE_SETTINGS)")
    assert ast.unparse(u7_base_run) == expected
    # U7: only the documented spots changed since the U7 base
    assert _mask_u7_run_qa_scan(new_run) == _mask_u7_run_qa_scan(u7_base_run)
    old_set, new_set = _member(old_cls, "show_qa_scanner_settings"), _member(new_cls, "show_qa_scanner_settings")
    expected = _replace_stmt(ast.unparse(old_set), assign(old_set, "_ai_trunc_default_prompt"),
                             "_ai_trunc_default_prompt = DEFAULT_AI_TRUNCATION_PROMPT")
    assert ast.unparse(new_set) == expected
    opener = ast.unparse(_member(new_cls, "open_latest_qa_report"))
    assert "find_latest_qa_report(override_dir, getattr(self, 'last_qa_report_path', None))" in opener
    assert "os.walk" not in opener and "is_direct_text_qa_path" not in opener


def _without_direct_text_opt_in(fn):
    """``run_qa_scan_path`` with the owner-approved chat-QA opt-in (2026-10-08) taken out again: the
    trailing ``allow_direct_text=False`` parameter and the ``not allow_direct_text and (...)`` wrapper
    around the Direct Text guard. Asserts the opt-in has exactly that shape."""
    import copy

    fn = copy.deepcopy(fn)
    assert fn.args.args[-1].arg == "allow_direct_text"
    assert ast.unparse(fn.args.defaults[-1]) == "False"
    del fn.args.args[-1], fn.args.defaults[-1]
    guards = [s for s in fn.body if isinstance(s, ast.If) and "allow_direct_text" in ast.unparse(s.test)]
    assert len(guards) == 1
    test = guards[0].test
    assert isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And) and len(test.values) == 2
    assert ast.unparse(test.values[0]) == "not allow_direct_text"
    guards[0].test = test.values[1]
    assert "allow_direct_text" not in ast.unparse(fn)
    return fn


def test_runtime_changed_only_in_the_documented_places(legacy_runtime):
    legacy_tree = ast.parse(git_text("src/qa_scan_runtime.py"))
    new_tree = ast.parse(src_text("qa_scan_runtime.py"))
    legacy, new = _top_defs(legacy_tree), _top_defs(new_tree)
    assert set(new) - set(legacy) == RUNTIME_NEW_NAMES | set(MOVED_HELPERS) | RUNTIME_U7_NAMES
    assert not set(legacy) - set(new)
    edited = {"apply_qa_scan_env_from_settings", "prepare_qa_scan_settings", "run_qa_scan_path"}
    for name, node in legacy.items():
        if name not in edited:
            assert ast.dump(node) == ast.dump(new[name]), name
    # run_qa_scan_path: only the owner-approved chat-QA opt-in (2026-10-08, DISCREPANCIES)
    assert ast.dump(_without_direct_text_opt_in(new["run_qa_scan_path"])) == ast.dump(legacy["run_qa_scan_path"])
    # apply: one statement before the snapshot of the previous values
    old_apply, new_apply = legacy["apply_qa_scan_env_from_settings"], new["apply_qa_scan_env_from_settings"]
    inserted = [ast.unparse(s) for s in new_apply.body if ast.dump(s) not in {ast.dump(o) for o in old_apply.body}]
    assert inserted == ["mappings.update(mobile_qa_env_overrides())"]
    assert [ast.dump(s) for s in new_apply.body if ast.unparse(s) not in inserted] == \
        [ast.dump(s) for s in old_apply.body]
    assert ast.unparse(new_apply.body[new_apply.body.index(
        next(s for s in new_apply.body if ast.unparse(s) == inserted[0])) + 1]).startswith("previous = ")
    # prepare: the forcing right before ``return settings``
    old_prep, new_prep = legacy["prepare_qa_scan_settings"], new["prepare_qa_scan_settings"]
    assert [ast.dump(s) for s in new_prep.body[:-2]] == [ast.dump(s) for s in old_prep.body[:-1]]
    assert ast.unparse(new_prep.body[-2]) == "if mobile_qa_forcing_active():\n    settings['use_thread_executor'] = True"
    assert ast.dump(new_prep.body[-1]) == ast.dump(old_prep.body[-1])
    # imports: re (the moved helpers) and mobile_runtime (the forcing) only
    old_imports = [ast.unparse(n) for n in legacy_tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    new_imports = [ast.unparse(n) for n in new_tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert [i for i in new_imports if i not in old_imports] == ["import re", "import mobile_runtime"]
    assert all(i in new_imports for i in old_imports)


# ---- owner-approved desktop-shared edit (2026-10-08): the chat-QA Direct Text opt-in -------------

_DIRECT_TEXT_SKIP_LOG = ("⏭️ QA scan skipped: Direct Text folders and temporary Direct Text "
                         "outputs are excluded from automatic QA scanning.")


def test_runtime_changed_only_by_the_direct_text_opt_in_since_u9():
    """Against the last commit before it, only run_qa_scan_path (the parameter + the guard condition)
    and run_bulk_qa_scan (the keyword-only parameter, handed to its single run_qa_scan_path call)
    changed; every other statement of qa_scan_runtime is the same."""
    import copy

    base_tree = ast.parse(git_text("src/qa_scan_runtime.py", DEVFIX_BASE_SHA))
    new_tree = ast.parse(src_text("qa_scan_runtime.py"))
    edited = ("run_qa_scan_path", "run_bulk_qa_scan")

    def rest(tree):
        return [ast.dump(n) for n in tree.body if not (isinstance(n, ast.FunctionDef) and n.name in edited)]

    assert rest(new_tree) == rest(base_tree)
    base, new = _top_defs(base_tree), _top_defs(new_tree)
    assert ast.dump(_without_direct_text_opt_in(new["run_qa_scan_path"])) == ast.dump(base["run_qa_scan_path"])
    bulk = copy.deepcopy(new["run_bulk_qa_scan"])
    assert bulk.args.kwonlyargs[-1].arg == "allow_direct_text"
    assert ast.unparse(bulk.args.kw_defaults[-1]) == "False"
    del bulk.args.kwonlyargs[-1], bulk.args.kw_defaults[-1]
    calls = [n for n in ast.walk(bulk) if isinstance(n, ast.Call) and ast.unparse(n.func) == "run_qa_scan_path"]
    assert len(calls) == 1
    assert ast.unparse(calls[0].keywords[-1]) == "allow_direct_text=allow_direct_text"
    del calls[0].keywords[-1]
    assert ast.dump(bulk) == ast.dump(base["run_bulk_qa_scan"])


def test_direct_text_guard_refuses_by_default_and_scans_only_with_the_opt_in(tmp_path, monkeypatch):
    scanned = []
    fake = types.ModuleType("scan_html_folder")
    fake.configure_qa_cache = lambda config: None

    def scan_html_folder(folder, **kw):
        scanned.append((folder, kw.get("epub_path"), kw.get("mode")))
        return ["scanned"]

    fake.scan_html_folder = scan_html_folder
    monkeypatch.setitem(sys.modules, "scan_html_folder", fake)
    chat = tmp_path / "Output" / "Direct Text" / "Novel - chat" / "Attachments" / "Book"
    chat.mkdir(parents=True)
    plain = tmp_path / "Output" / "Book"
    plain.mkdir(parents=True)
    chat_source = str(chat.parent / "Book.epub")
    # a Direct Text folder, and a Library-style folder whose source is a Direct Text file
    for folder, epub in ((str(chat), None), (str(plain), chat_source)):
        logs = []
        assert qa_scan_runtime.run_qa_scan_path(folder, log=logs.append, epub_path=epub, config={}) is None
        assert logs == [_DIRECT_TEXT_SKIP_LOG] and not scanned
        logs = []
        assert qa_scan_runtime.run_qa_scan_path(folder, log=logs.append, epub_path=epub, config={},
                                                allow_direct_text=False) is None
        assert logs == [_DIRECT_TEXT_SKIP_LOG] and not scanned
        assert qa_scan_runtime.run_qa_scan_path(folder, log=logs.append, epub_path=epub, config={},
                                                allow_direct_text=True) == ["scanned"]
        assert scanned == [(folder, epub, "quick-scan")] and logs.count(_DIRECT_TEXT_SKIP_LOG) == 1
        scanned.clear()
    assert qa_scan_runtime.run_qa_scan_path(str(plain), log=lambda m: None, config={}) == ["scanned"]


def test_bulk_scan_hands_the_opt_in_to_each_folder_scan(tmp_path, monkeypatch):
    seen = []

    def recorder(folder, **kw):
        seen.append((os.path.basename(folder), kw["allow_direct_text"]))

    monkeypatch.setattr(qa_scan_runtime, "run_qa_scan_path", recorder)
    folders = []
    for name in ("Book", "Other"):
        folder = tmp_path / "Output" / "Direct Text" / "chat" / "Attachments" / name
        folder.mkdir(parents=True)
        folders.append(str(folder))
    common = dict(mode="quick-scan", epub_path=None, qa_settings={}, load_settings=dict,
                  selected_mode_value="quick-scan", disable_word_count_for_run=False, epub_basename_map={},
                  global_selected_files=None, log=lambda message: None, stop_flag=lambda: False)
    qa_scan_runtime.run_bulk_qa_scan(folders[:1], **common)
    qa_scan_runtime.run_bulk_qa_scan(folders, allow_direct_text=True, **common)
    assert seen == [("Book", False), ("Book", True), ("Other", True)]


def test_desktop_callers_never_pass_the_direct_text_opt_in():
    """Only Glossarion Mobile's chat QA job opts in: QA_Scanner_GUI's bulk scan and TransateKRtoEN's
    multipass scan keep the Direct Text refusal, and no other desktop module names the keyword."""
    for name in ("QA_Scanner_GUI.py", "TransateKRtoEN.py"):
        tree = ast.parse(src_text(name))
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and getattr(n.func, "id", getattr(n.func, "attr", "")) in ("run_bulk_qa_scan", "run_qa_scan_path")]
        assert calls, name
        for call in calls:
            assert all(kw.arg is not None and kw.arg != "allow_direct_text" for kw in call.keywords), name
    mentions = sorted(p.name for p in SRC.glob("*.py") if p.name != "qa_scan_runtime.py"
                      and "allow_direct_text" in p.read_text(encoding="utf-8", errors="replace"))
    assert mentions == []


def _drop_local_hashlib_import(tree):
    """The user-approved fix (2026-10-06): update_new_format_progress no longer re-imports hashlib
    locally (that import shadowed the module one and raised UnboundLocalError). Applied to the
    legacy tree so the comparison below still pins everything else."""
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "update_new_format_progress")
    removed = 0
    for node in ast.walk(fn):
        for field in ("body", "orelse", "finalbody"):
            stmts = getattr(node, field, None)
            if isinstance(stmts, list):
                kept = [s for s in stmts if not (isinstance(s, ast.Import) and [a.name for a in s.names] == ["hashlib"])]
                removed += len(stmts) - len(kept)
                setattr(node, field, kept)
    return removed


def test_scanner_refusal_defaults_are_key_pool_service_list():
    legacy_tree = ast.parse(git_text("src/scan_html_folder.py"))
    new_tree = ast.parse(src_text("scan_html_folder.py"))
    assert _drop_local_hashlib_import(legacy_tree) == 1
    assert _drop_local_hashlib_import(new_tree) == 0
    old = [n for n in legacy_tree.body if isinstance(n, ast.Assign)
           and getattr(n.targets[0], "id", "") == "DEFAULT_REFUSAL_PATTERNS"]
    assert len(old) == 1 and ast.literal_eval(old[0].value) == list(key_pool_service.DEFAULT_REFUSAL_PATTERNS)
    imports = [n for n in new_tree.body if isinstance(n, ast.ImportFrom) and n.module == "key_pool_service"]
    assert [[a.name for a in n.names] for n in imports] == [["DEFAULT_REFUSAL_PATTERNS"]]
    assert not [n for n in new_tree.body if isinstance(n, ast.Assign)
                and getattr(n.targets[0], "id", "") == "DEFAULT_REFUSAL_PATTERNS"]
    # the import stands where the list was; nothing else in the scanner changed (bar the hashlib fix)
    old_body = [ast.dump(n) for n in legacy_tree.body]
    new_body = [ast.dump(n) for n in new_tree.body]
    index = old_body.index(ast.dump(old[0]))
    assert new_body[:index] == old_body[:index] and new_body[index + 1:] == old_body[index + 1:]
    assert new_body[index] == ast.dump(imports[0])


def test_scanner_refusal_fallback_has_the_legacy_values(monkeypatch, tmp_path):
    scan_html_folder = pytest.importorskip("scan_html_folder")
    assert scan_html_folder.DEFAULT_REFUSAL_PATTERNS is key_pool_service.DEFAULT_REFUSAL_PATTERNS
    monkeypatch.setattr(scan_html_folder, "_config_json_in", lambda base: str(tmp_path / "missing.json"))
    legacy_tree = ast.parse(git_text("src/scan_html_folder.py"))
    legacy_list = next(ast.literal_eval(n.value) for n in legacy_tree.body if isinstance(n, ast.Assign)
                       and getattr(n.targets[0], "id", "") == "DEFAULT_REFUSAL_PATTERNS")
    assert list(scan_html_folder._get_refusal_patterns_for_scan()) == legacy_list


_HYGIENE_PROBE = r"""
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ("PySide6", "translator_gui", "dpi_setup", "shiboken6", "QA_Scanner_GUI"):
            raise ImportError("blocked: " + name)
        return None
sys.meta_path.insert(0, Block())
sys.path.insert(0, {src!r})
import qa_scan_runtime
for name in {names!r}:
    getattr(qa_scan_runtime, name)
print("OK", sorted(m for m in sys.modules if m.split(".")[0] in ("PySide6", "QA_Scanner_GUI", "translator_gui")))
"""


def test_qa_scan_runtime_imports_without_qt_and_parses_on_python310():
    proc = subprocess.run([sys.executable, "-c", _HYGIENE_PROBE.format(src=str(SRC), names=GUI_REEXPORTS)],
                          capture_output=True, text=True, encoding="utf-8", timeout=180)
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert proc.stdout.strip().splitlines()[-1] == "OK []"
    tree = ast.parse(src_text("qa_scan_runtime.py"), feature_version=(3, 10))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            imported.add(node.module.split(".")[0])
    assert not imported & {"PySide6", "shiboken6", "translator_gui", "dpi_setup", "QA_Scanner_GUI"}


def test_gui_module_reexports_the_moved_objects():
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    import QA_Scanner_GUI

    for name in GUI_REEXPORTS:
        assert getattr(QA_Scanner_GUI, name) is getattr(qa_scan_runtime, name), name


def test_source_files_keep_uniform_line_endings():
    for name in ("QA_Scanner_GUI.py", "qa_scan_runtime.py", "scan_html_folder.py"):
        data = (SRC / name).read_bytes()
        assert data.count(b"\r\n") in (0, data.count(b"\n")), name


# =============================================================================================
# 2. differential fuzz: pure helpers
# =============================================================================================

def _outcome(fn, *args):
    try:
        return ("ok", fn(*args))
    except Exception as exc:  # compared by type and message
        return ("raise", type(exc).__name__, str(exc))


NAME_PARTS = ["Novel", "novel", "My_Book", "my-book", "Book 2", "Vol.3", "[Translated]", "[Vol 2]",
              "(Complete)", "(2023)", "제1권", "은빛 숲", "Ω", "re:zero", "  spaced  ", "v2", "_v3", "-v10",
              "_output", "_translated", "_en", "_final", "_done", "_custom", ".epub", ".txt", ".html",
              ".htm", ".EPUB", "1", "12", "001", "!", "&", "'s", "Ch-07", "x", ""]
SUFFIX_FLAVOURS = ["", "_output", "_translated", "_trans", "_en", "_english", "_done", "_complete", "_final",
                   "_custom", " v2", "_v2", "-v3", " (1)", "_mine"]


def _rand_name(rng):
    parts = [rng.choice(NAME_PARTS) for _ in range(rng.randint(1, 5))]
    sep = rng.choice(["", " ", "_", "-", "  "])
    return sep.join(parts)


def _derived(rng, name):
    base = re.sub(r"\.(epub|txt|html?)$", "", name, flags=re.I) if rng.random() < 0.7 else name
    out = base + rng.choice(SUFFIX_FLAVOURS)
    if rng.random() < 0.3:
        out = out.upper() if rng.random() < 0.5 else out.lower()
    if rng.random() < 0.3:
        out = out.replace(" ", rng.choice(["_", "-", "  "]))
    if rng.random() < 0.2:
        out = re.sub(r"\d+", lambda m: str(int(m.group()) + rng.choice([0, 1])), out)
    return out


def test_name_matching_matches_legacy(legacy_helpers):
    rng = random.Random(SEED)
    for _ in range(STATES):
        name = _rand_name(rng)
        assert _outcome(qa_scan_runtime.normalize_name_for_comparison, name) == \
            _outcome(legacy_helpers.normalize_name_for_comparison, name), name
        epub = name
        folder = _derived(rng, epub) if rng.random() < 0.7 else _rand_name(rng)
        suffixes = rng.choice(["", "_custom", "_mine, _x", " , _final ,", ",,", "v2"])
        args = (epub, folder, suffixes) if rng.random() < 0.8 else (epub, folder)
        if rng.random() < 0.5:
            args = (folder, epub) + args[2:]
        assert _outcome(qa_scan_runtime.check_epub_folder_match, *args) == \
            _outcome(legacy_helpers.check_epub_folder_match, *args), args


LANG_LABELS = ["English", "english", "EN", "en", "English (US)", "Spanish", "es", "French", "fr", "German",
               "Portuguese", "pt-BR", "Italian", "Russian", "Japanese", "ja", "Korean", "ko", "Chinese",
               "Chinese (Simplified)", "Chinese (Traditional)", "zh", "zh-CN", "zh-TW", "chinese simplified",
               "Arabic", "Hebrew", "Thai", "auto", "Auto", "AUTO", "", None, " ", "   ", "\t", "Klingon",
               "日本語", "  Korean  ", "traditional chinese", 0, 5, ["en"]]


def test_language_normalizers_match_legacy(legacy_helpers):
    rng = random.Random(SEED + 1)
    for i in range(STATES):
        label = LANG_LABELS[i % len(LANG_LABELS)] if i < len(LANG_LABELS) else rng.choice(LANG_LABELS)
        if isinstance(label, str) and rng.random() < 0.3:
            label = rng.choice(["", " ", "x "]) + label + rng.choice(["", " (UK)", " ", "-x"])
        for name in ("_normalize_target_language", "_normalize_source_language"):
            assert _outcome(getattr(qa_scan_runtime, name), label) == \
                _outcome(getattr(legacy_helpers, name), label), (name, label)


class _Raises:
    def __get__(self, obj, objtype=None):
        raise RuntimeError("attribute failure")


def _rand_owner(rng):
    kind = rng.choice(["none", "plain", "getter", "getter_raises", "config_none", "config_list", "prop_raises"])
    if kind == "none":
        return None
    attrs = {}
    if kind == "getter":
        value = rng.choice(["vision", "Vision ", " VISION", "text", "", None, "image", 0])
        attrs["_get_output_mode"] = lambda self, v=value: v
    elif kind == "getter_raises":
        def boom(self):
            raise ValueError("no mode")
        attrs["_get_output_mode"] = boom
    elif kind == "prop_raises":
        attrs["config"] = _Raises()
    owner_cls = type("FuzzOwner", (), attrs)
    owner = owner_cls()
    if kind == "config_none":
        owner.config = None
    elif kind == "config_list":
        owner.config = ["not", "a", "dict"]
    elif kind != "prop_raises" and rng.random() < 0.9:
        settings_choice = rng.choice(["dict", "none", "missing", "list"])
        config = {"output_mode": rng.choice(["vision", "text", "", None, "Vision", 3])}
        if settings_choice == "dict":
            config["qa_scanner_settings"] = {
                key: rng.choice(["truncation", "qa_truncation", " Truncation ", "other", "", None, 1])
                for key in rng.sample(["_qa_context", "qa_context", "context", "_context"], rng.randint(0, 3))
            }
            if rng.random() < 0.6:
                config["qa_scanner_settings"]["check_ai_truncation_detection"] = rng.choice([True, False, 0, 1, "", "x"])
        elif settings_choice == "none":
            config["qa_scanner_settings"] = None
        elif settings_choice == "list":
            config["qa_scanner_settings"] = ["x"]
        owner.config = config
    if rng.random() < 0.5:
        owner.base_dir = rng.choice(["", None, "<SB>/base"])
    return owner


def test_owner_helpers_match_legacy(legacy_helpers):
    rng = random.Random(SEED + 2)
    for _ in range(STATES):
        owner = _rand_owner(rng)
        for name in ("_qa_owner_output_mode", "_qa_owner_uses_truncation_context"):
            assert _outcome(getattr(qa_scan_runtime, name), owner) == \
                _outcome(getattr(legacy_helpers, name), owner), (name, vars(owner) if owner else None)


OCR_ENV = ("QA_VISION_OCR_SOURCE_EPUB", "VISION_OCR_SOURCE_EPUB", "EPUB_OUTPUT_DIR", "OUTPUT_DIRECTORY", "OUTPUT_DIR")


def test_vision_ocr_source_resolver_matches_legacy(legacy_helpers, tmp_path):
    rng = random.Random(SEED + 3)
    for state in range(FS_STATES):
        sb = tmp_path / f"s{state}"
        sb.mkdir()
        stem = rng.choice(["Book", "book", "Novel_1", "은빛"])
        files = [
            sb / "out" / stem / "OCR" / f"{stem}_OCR.epub",
            sb / "out" / "OCR" / f"{stem}_OCR.epub",
            sb / "root" / stem / "OCR" / f"{stem}_OCR.epub",
            sb / "base" / stem / "OCR" / f"{stem}_OCR.epub",
            sb / "env" / "OCR" / f"{stem}_OCR.epub",
            sb / "env" / f"{stem}_OCR.epub",
            sb / "env" / "OCR" / f"Other_OCR.epub",
            sb / "env" / "ocr" / f"{stem}_ocr.EPUB",
        ]
        for path in files:
            if rng.random() < 0.4 and not path.exists():  # case-insensitive file systems: OCR == ocr
                path.parent.mkdir(parents=True, exist_ok=True)
                if rng.random() < 0.9:
                    path.write_bytes(b"PK")
                else:
                    path.mkdir()  # a directory, not a file
        values = {
            "QA_VISION_OCR_SOURCE_EPUB": [None, "", "  ", str(files[4]), str(files[5]), str(files[6]), str(files[7])],
            "VISION_OCR_SOURCE_EPUB": [None, "", str(files[4]), str(files[7])],
            "EPUB_OUTPUT_DIR": [None, "", str(sb / "out" / stem), str(sb / "out"), " "],
            "OUTPUT_DIRECTORY": [None, "", str(sb / "root")],
            "OUTPUT_DIR": [None, str(sb / "root"), str(sb / "nowhere")],
        }
        for key in OCR_ENV:
            value = rng.choice(values[key])
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        owner = _rand_owner(rng)
        if owner is not None and getattr(owner, "base_dir", None) == "<SB>/base":
            owner.base_dir = str(sb / "base")
        if owner is not None and rng.random() < 0.6:
            try:
                owner.config = dict(owner.config or {}) if isinstance(owner.config, dict) else owner.config
                if isinstance(owner.config, dict):
                    owner.config["output_mode"] = "vision"
            except Exception:
                pass
        path = rng.choice([None, "", "x.txt", str(sb / "in" / f"{stem}.epub"), str(sb / "in" / f"{stem}_OCR.epub"),
                           str(sb / "in" / f"{stem}.EPUB"), f"{stem}.epub", str(files[0])])
        os.chdir(str(sb))
        args = (path, owner) if rng.random() < 0.85 else (path,)
        assert _outcome(qa_scan_runtime._qa_vision_ocr_source_path, *args) == \
            _outcome(legacy_helpers._qa_vision_ocr_source_path, *args), (state, path)
        for key in OCR_ENV:
            os.environ.pop(key, None)


# =============================================================================================
# 3. open_latest_qa_report: legacy method vs the rewired method (+ find_latest_qa_report)
# =============================================================================================

class _QtRecorder:
    """Stand-ins for QMessageBox / QDesktopServices / QUrl recording what the method shows."""

    def __init__(self):
        self.events = []
        recorder = self

        class MessageBox:
            @staticmethod
            def information(parent, title, text):
                recorder.events.append(("information", title, text))

            @staticmethod
            def warning(parent, title, text):
                recorder.events.append(("warning", title, text))

        class DesktopServices:
            @staticmethod
            def openUrl(url):
                recorder.events.append(("open", url))

        class Url:
            @staticmethod
            def fromLocalFile(path):
                return ("file", path)

        self.ns = {"QMessageBox": MessageBox, "QDesktopServices": DesktopServices, "QUrl": Url}


def _opener(tree, extra_ns):
    method = _member(_class(tree, "QAScannerMixin"), "open_latest_qa_report")
    recorder = _QtRecorder()
    ns = {"os": os, "__name__": "qa_opener_u6", **recorder.ns, **extra_ns}
    _exec_defs([method], "QA_Scanner_GUI.py", ns)
    return ns["open_latest_qa_report"], recorder


REPORT = "validation_results.html"
TREE_DIRS = ["a", "a/b", "a/b/c", "Direct Text", "Direct Text/x", "direct-text_chat", "Glossarion_Direct_Text_1",
             "glossarion_input_output_2", "Book_Scan Report", "Book", "Book/Book_Scan Report", "z"]


def _owner_state(rng, sb):
    owner = types.SimpleNamespace()
    log = []
    if rng.random() < 0.9:
        owner.append_log = log.append
    config_kind = rng.choice(["dict", "dict", "none", "missing"])
    if config_kind == "dict":
        owner.config = {}
        if rng.random() < 0.6:
            owner.config["output_directory"] = rng.choice([str(sb / "tree"), str(sb / "tree" / "a"), str(sb / "nope"),
                                                           "", str(sb / "tree" / "Direct Text")])
    elif config_kind == "none":
        owner.config = None
    last = rng.choice(["missing", None, "", str(sb / "outside" / REPORT), str(sb / "gone" / REPORT)])
    if last != "missing":
        owner.last_qa_report_path = last
    return owner, log


def test_open_latest_qa_report_matches_legacy(legacy_gui_tree, new_gui_tree, tmp_path):
    legacy_open, legacy_rec = _opener(legacy_gui_tree, {})
    new_open, new_rec = _opener(new_gui_tree, {"find_latest_qa_report": qa_scan_runtime.find_latest_qa_report})
    rng = random.Random(SEED + 4)
    cwd = os.getcwd()
    try:
        for state in range(FS_STATES):
            sb = tmp_path / f"s{state}"
            (sb / "tree").mkdir(parents=True)
            (sb / "cwd").mkdir()
            (sb / "outside").mkdir()
            if rng.random() < 0.6:
                (sb / "outside" / REPORT).write_text("old", encoding="utf-8")
            mtimes = [1_600_000_000 + rng.choice([0, 0, 5, 10, 10, 99]) for _ in TREE_DIRS]
            for rel, mtime in zip(TREE_DIRS, mtimes):
                if rng.random() < 0.5:
                    d = sb / "tree" / rel
                    d.mkdir(parents=True, exist_ok=True)
                    name = rng.choice([REPORT, REPORT, "Validation_Results.HTML", "validation_results.htm", "x.html"])
                    f = d / name
                    f.write_text(rel, encoding="utf-8")
                    os.utime(f, (mtime, mtime))
            if rng.random() < 0.3:
                (sb / "cwd" / REPORT).write_text("cwd", encoding="utf-8")
            env_choice = rng.choice(["unset", "unset", "tree", "sub", "missing", "direct", "empty"])
            env_value = {"tree": str(sb / "tree"), "sub": str(sb / "tree" / "a"), "missing": str(sb / "nope"),
                         "direct": str(sb / "tree" / "Direct Text"), "empty": ""}.get(env_choice)
            if env_value is None:
                os.environ.pop("OUTPUT_DIRECTORY", None)
            else:
                os.environ["OUTPUT_DIRECTORY"] = env_value
            os.chdir(str(rng.choice([sb / "cwd", sb / "tree", sb / "tree" / "a" if (sb / "tree" / "a").is_dir() else sb])))
            seed_state = rng.random()
            outcomes = []
            for opener, rec in ((legacy_open, legacy_rec), (new_open, new_rec)):
                owner, log = _owner_state(random.Random(seed_state), sb)
                rec.events.clear()
                opener(owner)
                outcomes.append((list(rec.events), list(log), getattr(owner, "last_qa_report_path", "<unset>")))
            assert outcomes[0] == outcomes[1], (state, env_choice, outcomes)
            # the shared search alone: the report the legacy method opened (or None)
            owner, _log = _owner_state(random.Random(seed_state), sb)
            try:
                override = os.environ.get("OUTPUT_DIRECTORY") or owner.config.get("output_directory")
            except AttributeError:
                continue  # the method's own failure path (config is None / missing)
            found = qa_scan_runtime.find_latest_qa_report(override, getattr(owner, "last_qa_report_path", None))
            opened = [e[1][1] for e in outcomes[0][0] if e[0] == "open"]
            assert ([os.path.abspath(found)] if found else []) == opened, state
    finally:
        os.chdir(cwd)


# =============================================================================================
# 4. env helpers: desktop == legacy, mobile == legacy + the forcing
# =============================================================================================

def _rand_settings(rng, runtime):
    defaults = runtime.default_qa_scan_settings()
    settings = {}
    for key, value in defaults.items():
        roll = rng.random()
        if roll < 0.25:
            continue
        if roll < 0.6:
            settings[key] = value
        elif isinstance(value, bool):
            settings[key] = rng.choice([True, False, 0, 1, None, "yes"])
        elif isinstance(value, (int, float)):
            settings[key] = rng.choice([0, 1, 49, 50, 0.5, -1, None, "7"])
        elif isinstance(value, list):
            settings[key] = rng.choice([[], ["Sure"], ["ü", "。"], value])
        else:
            settings[key] = rng.choice(["", "english", "Korean", None])
    for key in ("counting_mode", "use_thread_executor", "check_body_tag", "exclude_ruby_tags"):
        if rng.random() < 0.5:
            settings[key] = rng.choice(["word", "exact", "", None, True, False, "WORD"])
    return settings if rng.random() < 0.95 else rng.choice([None, [], "x"])


ENV_PRESET = ("QA_USE_THREAD_EXECUTOR", "AI_HUNTER_MAX_WORKERS", "QA_TARGET_LANGUAGE", "QA_REPORT_FORMAT",
              "OUTPUT_LANGUAGE", "SKIP_TITLE_TAG_TRANSLATION", "API_KEY", "MODEL", "OUTPUT_MODE")


def _preset_env(rng):
    for key in ENV_PRESET:
        roll = rng.random()
        if roll < 0.5:
            os.environ.pop(key, None)
        else:
            os.environ[key] = rng.choice(["0", "1", "4", "16", "", " 1 ", "x", "korean", "gpt-x", "vision"])


def _apply_outcome(runtime, settings):
    before = dict(os.environ)
    previous = runtime.apply_qa_scan_env_from_settings(settings)
    after = {key: os.environ.get(key) for key in previous}
    runtime.restore_env(previous)
    restored = dict(os.environ) == before
    return previous, after, restored


@pytest.mark.parametrize("mobile_env", [None, "GLOSSARION_MOBILE", "GLOSSARION_NO_PROCESSES"])
def test_apply_env_desktop_is_legacy_and_mobile_adds_the_forcing(legacy_runtime, mobile_env, monkeypatch):
    if mobile_env:
        monkeypatch.setenv(mobile_env, "1")
    rng = random.Random(SEED + 5)
    for _ in range(STATES):
        settings = _rand_settings(rng, legacy_runtime)
        _preset_env(rng)
        old_prev, old_after, old_restored = _apply_outcome(legacy_runtime, settings)
        new_prev, new_after, new_restored = _apply_outcome(qa_scan_runtime, settings)
        assert old_restored and new_restored
        if not mobile_env:
            assert (new_prev, new_after) == (old_prev, old_after)
            continue
        assert qa_scan_runtime.mobile_qa_forcing_active()
        expected_workers = str(qa_scan_runtime.mobile_qa_max_workers())
        assert set(new_prev) == set(old_prev) | {"AI_HUNTER_MAX_WORKERS"}
        assert {k: new_prev[k] for k in old_prev} == old_prev
        assert new_prev["AI_HUNTER_MAX_WORKERS"] == os.environ.get("AI_HUNTER_MAX_WORKERS")
        assert new_after == {**old_after, "QA_USE_THREAD_EXECUTOR": "1", "AI_HUNTER_MAX_WORKERS": expected_workers}


def test_mobile_worker_cap(monkeypatch):
    monkeypatch.setenv("GLOSSARION_MOBILE", "1")
    cases = {None: 2, "": 2, "0": 2, "1": 1, " 1 ": 1, "2": 2, "8": 2, "-3": 2, "x": 2, "1.5": 2}
    for raw, expected in cases.items():
        if raw is None:
            monkeypatch.delenv("AI_HUNTER_MAX_WORKERS", raising=False)
        else:
            monkeypatch.setenv("AI_HUNTER_MAX_WORKERS", raw)
        assert qa_scan_runtime.mobile_qa_max_workers() == expected, raw
        assert qa_scan_runtime.mobile_qa_env_overrides() == {
            "QA_USE_THREAD_EXECUTOR": "1", "AI_HUNTER_MAX_WORKERS": str(expected)}
    monkeypatch.delenv("GLOSSARION_MOBILE")
    assert qa_scan_runtime.mobile_qa_env_overrides() == {} and not qa_scan_runtime.mobile_qa_forcing_active()


class _Widget:
    def __init__(self, value):
        self._value = value

    def text(self):
        return self._value

    def isChecked(self):
        return bool(self._value)

    def get(self):
        return self._value


def _rand_prepare_args(rng):
    kind = rng.choice(["owner", "owner", "config", "none"])
    owner = config = None
    if kind == "owner":
        attrs = {}
        if rng.random() < 0.7:
            mode = rng.choice(["vision", "text", None, ""])
            attrs["_get_output_mode"] = lambda self, m=mode: m
        owner = type("PrepOwner", (), attrs)()
        owner.config = {k: rng.choice([True, False, "", "k", 3, None, ["a"]]) for k in rng.sample(
            ["api_key", "model", "output_language", "use_qa_scan_keys", "qa_scan_keys", "skip_title_tag_translation",
             "use_ai_truncation_detection_keys", "force_key_rotation", "rotation_frequency", "batch_size",
             "output_mode"], rng.randint(0, 8))}
        if rng.random() < 0.5:
            owner.config["qa_scanner_settings"] = {"check_ai_truncation_detection": rng.choice([True, False, 0])} \
                if rng.random() < 0.7 else {}
        for attr, value in (("api_key_entry", _Widget(rng.choice(["", " sk-1 ", "k"]))),
                            ("model_var", rng.choice(["", "gpt-5", None])),
                            ("batch_translation_var", _Widget(rng.choice([True, False]))),
                            ("batch_size_entry", _Widget(rng.choice(["", "4", "x"]))),
                            ("use_qa_scan_keys_var", rng.choice([_Widget(True), True, "1", None])),
                            ("skip_title_tag_translation_var", rng.choice([_Widget(False), "on", None])),
                            ("use_ai_truncation_detection_keys_var", rng.choice([_Widget(True), 0]))):
            if rng.random() < 0.6:
                setattr(owner, attr, value)
    elif kind == "config":
        config = types.SimpleNamespace(**{k: v for k, v in (("API_KEY", "k1"), ("MODEL", "m1"), ("TEMP", 0.7),
                                                           ("MAX_OUTPUT_TOKENS", 100), ("BATCH_SIZE", 3),
                                                           ("OUTPUT_MODE", "vision")) if rng.random() < 0.5})
    output_mode = rng.choice([None, None, "text", "vision"])
    return owner, config, output_mode


@pytest.mark.parametrize("mobile", [False, True])
def test_prepare_settings_desktop_is_legacy_and_mobile_forces_threads(legacy_runtime, mobile, monkeypatch):
    if mobile:
        monkeypatch.setenv("GLOSSARION_MOBILE", "1")
    rng = random.Random(SEED + 6)
    for _ in range(STATES):
        settings = _rand_settings(rng, legacy_runtime)
        _preset_env(rng)
        owner, config, output_mode = _rand_prepare_args(rng)
        old = _outcome(legacy_runtime.prepare_qa_scan_settings, settings, owner, config, output_mode)
        new = _outcome(qa_scan_runtime.prepare_qa_scan_settings, settings, owner, config, output_mode)
        if not mobile or old[0] != "ok":
            assert new == old
        else:
            assert new == ("ok", {**old[1], "use_thread_executor": True})


# =============================================================================================
# 5. end to end: quick scan, desktop environment vs mobile environment
# =============================================================================================

BOOK = "e2e-glossary-off"
E2E_CONFIG = {"model": "gpt-4o-mini", "api_key": "sk-glossarion-u6-qa-dummy", "output_language": "English",
              "delay": 0, "auto_glossary_mode": "off", "enable_auto_glossary": False}
RESPONSE_SHELL = "<!DOCTYPE html>\n<html>\n<head>\n    <meta charset=\"utf-8\">\n</head>\n<body>\n{body}\n</body>\n</html>\n"

_DRIVER = r'''
# Spawned process-pool workers import this file as __mp_main__: everything runs under main().
import json, os, sys


def main(root):
    sys.path.insert(0, {src!r})
    if os.environ.get("GLOSSARION_MOBILE") == "1":
        import importlib.abc

        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in ("PySide6", "shiboken6", "translator_gui", "dpi_setup", "QA_Scanner_GUI"):
                    raise ImportError("blocked on the mobile path: " + name)
                return None

        sys.meta_path.insert(0, Block())
    import app_paths
    assert os.path.normcase(app_paths.CONFIG_FILE).startswith(os.path.normcase(root)), app_paths.CONFIG_FILE
    from headless_owner import HeadlessOwner
    import qa_scan_runtime
    import scan_html_folder

    seen = {{}}
    real_scan = scan_html_folder.scan_html_folder

    def spy(*args, **kwargs):
        seen["env"] = {{k: os.environ.get(k) for k in ("QA_USE_THREAD_EXECUTOR", "AI_HUNTER_MAX_WORKERS")}}
        seen["use_thread_executor"] = (kwargs.get("qa_settings") or {{}}).get("use_thread_executor")
        return real_scan(*args, **kwargs)

    scan_html_folder.scan_html_folder = spy
    log = []
    owner = HeadlessOwner.from_config_store(host=type("Host", (), {{"log": staticmethod(lambda *a: None)}})(),
                                            path=app_paths.CONFIG_FILE)
    settings = qa_scan_runtime.normalize_qa_scan_settings(
        owner.config.get("qa_scanner_settings", {{}}),
        target_language=owner.config.get("output_language") or os.getenv("OUTPUT_LANGUAGE", ""))
    qa_scan_runtime.run_qa_scan_path(
        os.path.join(root, "Output", {book!r}), log=log.append, stop_flag=lambda: False, mode="quick-scan",
        qa_settings=settings, epub_path=os.path.join(root, "Inbox", {book!r} + ".epub"), selected_files=None,
        text_file_mode=None, owner=owner)
    qt = sorted(m for m in sys.modules if m.split(".")[0] in ("PySide6", "QA_Scanner_GUI", "translator_gui"))
    with open(os.path.join(root, "result.json"), "w", encoding="utf-8") as fh:
        json.dump({{"log": [str(x) for x in log], "seen": seen, "qt": qt}}, fh, ensure_ascii=False)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main(sys.argv[1])
'''

_SEED_LANGDETECT = (
    "try:\n    from langdetect import DetectorFactory\n    DetectorFactory.seed = 0\nexcept Exception:\n    pass\n"
)


def _build_workspace(root: Path) -> None:
    """A U3-E2E-style workspace: the self-test EPUB and its chapters translated by the E2E fake model."""
    pytest.importorskip("ebooklib")
    pytest.importorskip("bs4")
    for path in (str(MOBILE_TOOLS), str(MOBILE_APP)):
        if path not in sys.path:
            sys.path.insert(0, path)
    import zipfile

    import prepare_assets
    from glossarion_mobile.diagnostics.fake_llm_server import fake_translate

    (root / "Inbox").mkdir(parents=True)
    epub = root / "Inbox" / f"{BOOK}.epub"
    prepare_assets.build_selftest_epub(epub)
    out = root / "Output" / BOOK
    (out / "images").mkdir(parents=True)
    chapters = {}
    with zipfile.ZipFile(epub) as zf:
        for name in sorted(zf.namelist()):
            match = re.search(r"chapter(\d{4})\.xhtml$", name)
            if match:
                html = zf.read(name).decode("utf-8")
                body = re.search(r"<body[^>]*>(.*)</body>", html, re.S).group(1)
                chapters[int(match.group(1))] = (name, body)
            elif name.endswith("emblem.png"):
                (out / "images" / "chapter0006_img_1.png").write_bytes(zf.read(name))
    progress = {"chapters": {}, "chapter_chunks": {}, "image_chunks": {}, "version": "2.1",
                "output_mode": "text", "completed_list": []}
    for num, (name, body) in sorted(chapters.items()):
        body = body.replace("../images/emblem.png", f"../images/chapter{num:04d}_img_1.png")
        response = f"response_chapter{num:04d}.html"
        (out / response).write_text(RESPONSE_SHELL.format(body=fake_translate(body)), encoding="utf-8")
        key = f"{num}@{num + 1}"
        progress["chapters"][key] = {"actual_num": num, "output_file": response, "status": "completed",
                                     "last_updated": 1791259073.0 + num, "model_name": "gpt-4o-mini",
                                     "original_basename": os.path.basename(name)}
        progress["completed_list"].append({"num": num, "idx": 0, "title": f"Chapter {num}", "file": response,
                                           "key": key})
    (out / "translation_progress.json").write_text(json.dumps(progress, ensure_ascii=False, indent=2),
                                                   encoding="utf-8")
    assert len(chapters) == 12


@pytest.fixture(scope="module")
def e2e_workspace(tmp_path_factory):
    if os.environ.get("PARITY_U6_QA_E2E", "1") == "0":
        pytest.skip("PARITY_U6_QA_E2E=0")
    root = tmp_path_factory.mktemp("qa_e2e_source")
    kept = os.environ.get("PARITY_U6_QA_E2E_SANDBOX", "").strip()
    if kept:
        shutil.copytree(Path(kept) / "Output" / BOOK, root / "Output" / BOOK)
        (root / "Inbox").mkdir()
        shutil.copy2(Path(kept) / "Inbox" / f"{BOOK}.epub", root / "Inbox" / f"{BOOK}.epub")
    else:
        _build_workspace(root)
    return root


def _normalized_tree(root: Path) -> dict:
    folder = root / "Output" / BOOK
    variants = sorted({str(root), str(root).replace("\\", "/"), str(root).replace("\\", "\\\\"),
                       json.dumps(str(root))[1:-1]}, key=len, reverse=True)
    out = {}
    for path in sorted(folder.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(folder).as_posix()
        data = path.read_bytes()
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            out[rel] = data
            continue
        for v in variants:
            text = text.replace(v, "<ROOT>")
        text = re.sub(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?", "<TS>", text)
        text = re.sub(r'("(?:last_updated|qa_timestamp|timestamp)"\s*:\s*)\d+(?:\.\d+)?', r"\1<EPOCH>", text)
        out[rel] = text
    return out


def test_quick_scan_report_desktop_path_equals_mobile_path(e2e_workspace, tmp_path):
    driver = tmp_path / "qa_scan_driver.py"
    driver.write_text(_DRIVER.format(src=str(SRC), book=BOOK), encoding="utf-8")
    seed_dir = tmp_path / "seed_site"
    seed_dir.mkdir()
    (seed_dir / "sitecustomize.py").write_text(_SEED_LANGDETECT, encoding="utf-8")
    results = {}
    for side in ("desktop", "mobile"):
        root = tmp_path / side
        shutil.copytree(e2e_workspace, root)
        for name in ("src", "cwd", "home", "tmp", "appdata", "localappdata", "Library"):
            (root / name).mkdir(exist_ok=True)
        (root / "src" / "config.json").write_text(json.dumps(E2E_CONFIG, indent=2), encoding="utf-8")
        # A fresh scan: no stop signal another test of this process may have left behind.
        env = {k: v for k, v in os.environ.items() if not k.startswith(("GLOSSARION_", "FLET_"))
               and k not in ("TRANSLATION_CANCELLED", "GRACEFUL_STOP", "GRACEFUL_STOP_COMPLETED",
                             "WAIT_FOR_CHUNKS", "GLOSSARY_STOP_FILE")}
        env.update({
            "GLOSSARION_APP_DIR": str(root / "src"), "CONFIG_FILE": str(root / "src" / "config.json"),
            "HOME": str(root / "home"), "USERPROFILE": str(root / "home"), "APPDATA": str(root / "appdata"),
            "LOCALAPPDATA": str(root / "localappdata"), "TEMP": str(root / "tmp"), "TMP": str(root / "tmp"),
            "OUTPUT_DIRECTORY": str(root / "Output"), "GLOSSARION_LIBRARY_DIR": str(root / "Library"),
            # unified_api_client logs every request (the scanner's tiktoken download too) into src/
            # http_requests unless this is off; the filter above dropped the caller's value
            "GLOSSARION_HTTP_LOG": "0",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONPATH": os.pathsep.join(p for p in (str(seed_dir), os.environ.get("PYTHONPATH", "")) if p),
        })
        if side == "mobile":
            env.update({"GLOSSARION_MOBILE": "1", "GLOSSARION_NO_PROCESSES": "1",
                        "GLOSSARION_DATA_DIR": str(root / "src")})
        proc = subprocess.run([sys.executable, str(driver), str(root)], cwd=str(root / "cwd"), env=env,
                              capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=900)
        result_file = root / "result.json"
        assert result_file.is_file(), f"{side} scan failed (exit {proc.returncode}):\n{proc.stderr[-4000:]}"
        results[side] = json.loads(result_file.read_text(encoding="utf-8"))
        result_file.unlink()

    desktop, mobile = results["desktop"], results["mobile"]
    assert desktop["seen"]["use_thread_executor"] in (False, None)
    assert desktop["seen"]["env"]["QA_USE_THREAD_EXECUTOR"] == "0"
    assert not any("Using ThreadPoolExecutor" in line for line in desktop["log"])
    assert mobile["seen"] == {"env": {"QA_USE_THREAD_EXECUTOR": "1", "AI_HUNTER_MAX_WORKERS": "2"},
                              "use_thread_executor": True}
    assert any("Using ThreadPoolExecutor" in line for line in mobile["log"])
    assert mobile["qt"] == []

    desk_tree, mob_tree = _normalized_tree(tmp_path / "desktop"), _normalized_tree(tmp_path / "mobile")
    reports = sorted(k for k in desk_tree if "_Scan Report/" in k)
    assert f"{BOOK}_Scan Report/validation_results.html" in reports
    assert f"{BOOK}_Scan Report/validation_results.json" in reports
    for side in ("desktop", "mobile"):  # both scans really processed every chapter
        report = tmp_path / side / "Output" / BOOK / f"{BOOK}_Scan Report" / "validation_results.json"
        rows = json.loads(report.read_text(encoding="utf-8"))
        assert sorted(r["filename"] for r in rows if r["filename"].startswith("response_")) ==             [f"response_chapter{n:04d}.html" for n in range(1, 13)], side
    assert sorted(desk_tree) == sorted(mob_tree)
    for rel in sorted(desk_tree):
        assert desk_tree[rel] == mob_tree[rel], f"{rel} differs between the desktop and the mobile scan"
    progress = json.loads((tmp_path / "desktop" / "Output" / BOOK / "translation_progress.json").read_text("utf-8"))
    assert progress["chapters"], "the scan dropped the progress entries"


def test_update_new_format_progress_hashes_flagged_translation_artifacts(tmp_path):
    """User-approved desktop fix (2026-10-06): a QA-flagged translation artifact
    (translated_headers.txt / TOC.txt) used to end the scan with UnboundLocalError on
    'hashlib', because a function-local `import hashlib` shadowed the module import."""
    import hashlib
    import scan_html_folder as shf

    for name in ("translated_headers.txt", "TOC.txt"):
        folder = tmp_path / name.replace(".", "_")
        folder.mkdir()
        artifact = folder / name
        artifact.write_text("Chapter 1: Title\n", encoding="utf-8")
        prog = {"version": "2.1", "chapters": {}, "chapter_chunks": {}, "content_hashes": {}}
        faulty = [{"filename": name, "file_index": 0, "issues": ["non_english_content"],
                   "file_path": str(artifact)}]
        shf.update_new_format_progress(prog, faulty, [], lambda *_: None, str(folder))
        entries = [e for e in prog["chapters"].values() if e.get("output_file") == name]
        assert entries, (name, prog["chapters"])
        assert entries[0]["content_hash"] == hashlib.sha256(artifact.read_bytes()).hexdigest()
