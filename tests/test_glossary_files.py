"""U6: glossary_files (the main window's glossary file actions) and the glossary Stop protocol.

``TranslatorGUI``'s glossary file code moved to ``glossary_files`` with explicit parameters:
``create_glossary_backup`` / ``_clean_old_backups``, the 🗑️ / ↩️ closures of the auto-glossary
row (``_delete_current_glossary``, ``_find_latest_backup``, ``_restore_glossary_backup``), the
data steps of the Map Glossaries to EPUBs dialog and the JSON repair helpers; the non-widget
part of ``stop_glossary_extraction`` moved to ``stop_control.request_glossary_stop`` (+
``kill_glossary_helper_subprocesses``). TranslatorGUI keeps thin wrappers. The oracle is
translator_gui.py at ``U6_BASE_SHA`` (``git show``); frozen methods and closures are executed
against the live translator_gui globals with recording stand-ins for Qt, the clock and the
backend modules.

Tiers:
* H (hygiene): glossary_files imports without PySide6 / translator_gui / dpi_setup; both
  modules parse as Python 3.10; line endings stay uniform and translator_gui keeps its BOM;
* V (verbatim): every moved body equals the frozen lines plus the documented edits
  (``MOVED_BODIES``), the Stop protocol keeps every frozen statement in order, and
  translator_gui.py differs from the frozen file only inside the rewired spans;
* D / F (differential fuzz on file-system fixtures, >= ``PARITY_U6_STATES`` = 500 states):
  editor backups (+ pruning and the failure question), delete / latest backup / restore of a
  book's glossary files across the shared ``Glossary/`` and per-book layouts, the JSON repair
  helpers, and the owner-free auto-mapping adapters; the frozen code and the working-tree
  wrappers run in the same fresh folder and must leave the same files, logs, owner state,
  environment and message boxes;
* T (trace): ``stop_glossary_extraction`` frozen vs working tree vs a direct
  ``request_glossary_stop`` call (the mobile path) - the ordered trace of environment flags,
  latch, module stop flags, cleanup thread, helper-process kills, stop file and log lines;
* S (smoke): the Map Glossaries to EPUBs dialog, frozen and working tree, offscreen: prefill,
  Auto-Fill, Use one Glossary, Clear All and Save (missing files, Manual Glossary Only copy).

Run (repository root)::

    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic \\
        tests/test_glossary_files.py
"""

from __future__ import annotations

import ast
import copy
import datetime as _datetime_module
import importlib.util
import json
import logging
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

import glossary_files as gf  # noqa: E402
import stop_control  # noqa: E402

HAS_QT = importlib.util.find_spec("PySide6") is not None
needs_qt = pytest.mark.skipif(not HAS_QT, reason="needs PySide6")

STATES = max(1, int(os.environ.get("PARITY_U6_STATES", "500")))
SEED = int(os.environ.get("PARITY_U6_SEED", "6063"))
#: Map Glossaries dialog runs (each opens two offscreen dialogs).
DIALOG_RUNS = max(1, int(os.environ.get("PARITY_U6_DIALOG_RUNS", "40")))

U6_BASE_SHA = "e28e3a0f8e2ef3bd2aea59c87049b336ea469ac0"
TG = "src/translator_gui.py"

_APPEND = "append"
_PREPEND = "prepend"

#: new function -> ([(frozen file, first line, last line), ...], edits); see
#: tests/test_parallel_epub_core.py for the conventions (dedent, rstrip, edit kinds).
MOVED_BODIES = {
    ("glossary_files", "create_glossary_backup"): ([(TG, 10346, 10394)], [
        ("if not self.current_glossary_data or not hasattr(self, 'editor_file_entry') or not "
         "self.editor_file_entry.text():", "if not current_glossary_data or not glossary_path:"),
        ("re", r"self\.config\.get\(", "config.get("),
        ("original_path = self.editor_file_entry.text()", "original_path = glossary_path"),
        ("re", r"self\.append_log\(", "append_log("),
        ("json.dump(self.current_glossary_data,", "json.dump(current_glossary_data,"),
        ("self._clean_old_backups(backup_dir, original_name, max_backups)",
         "clean_old_backups(backup_dir, original_name, max_backups, append_log)"),
        ("    reply = QMessageBox.question(self, \"Backup Failed\",\n"
         "                              f\"Failed to create backup: {str(e)}\\n\\nContinue anyway?\",\n"
         "                              QMessageBox.Yes | QMessageBox.No)\n"
         "    return reply == QMessageBox.Yes",
         "    if ask_continue is None:\n"
         "        return False\n"
         "    return bool(ask_continue(\"Backup Failed\",\n"
         "                             f\"Failed to create backup: {str(e)}\\n\\nContinue anyway?\"))"),
    ]),
    ("glossary_files", "clean_old_backups"): ([(TG, 10436, 10456)], [
        ("re", r"self\.append_log\(", "append_log("),
    ]),
    ("glossary_files", "selected_glossary_epubs"): ([(TG, 19465, 19470)], [
        ("files = list(getattr(self, 'selected_files', []) or [])", "files = list(selected_files or [])"),
        ("if not epubs and hasattr(self, 'get_current_epub_path'):",
         "if not epubs and get_current_epub_path is not None:"),
        ("ep = self.get_current_epub_path()", "ep = get_current_epub_path()"),
        (_APPEND, "return epubs"),
    ]),
    ("glossary_files", "collect_glossary_files_for_inputs"): ([(TG, 19475, 19564)], [
        (_PREPEND, "if guess_glossary is None:\n    def guess_glossary(path):\n"
                   "        return guess_glossary_for_input_file(path, config)\n"),
        ("re", r"self\.config\.get\(", "config.get("),
        ("is_balanced_full = mode in ('balanced', 'full')",
         "is_balanced_full = mode in ('balanced', 'full')  # noqa: F841 - unused, kept from the desktop"),
        ("self._guess_glossary_for_input_file(epub_path)", "guess_glossary(epub_path)"),
        ("_auto_gp = getattr(self, 'auto_loaded_glossary_path', None) or getattr(self, 'manual_glossary_path', None)",
         "_auto_gp = auto_loaded_glossary_path or manual_glossary_path"),
        ("not getattr(self, 'manual_glossary_manually_loaded', False)", "not manually_loaded"),
        (_APPEND, "return all_files"),
    ]),
    ("glossary_files", "glossary_delete_display"): ([(TG, 19580, 19589)], [
        (_APPEND, "return display"),
    ]),
    ("glossary_files", "delete_glossary_files"): ([(TG, 19604, 19616)], [
        ("re", r"self\.append_log\(", "append_log("),
        (_APPEND, "return deleted"),
    ]),
    ("glossary_files", "find_latest_glossary_backup"): ([(TG, 19637, 19637), (TG, 19647, 19674)], [
        ("re", r"self\.config\.get\(", "config.get("),
    ]),
    ("glossary_files", "restore_glossary_backup"): ([(TG, 19705, 19715)], [
        (_PREPEND, "import shutil"),
        ("re", r"self\.append_log\(", "append_log("),
        (_APPEND, "return restored"),
    ]),
    ("glossary_files", "normalize_glossary_drop_path"): ([(TG, 31652, 31658)], []),
    ("glossary_files", "is_allowed_glossary_file"): ([(TG, 31661, 31666)], [
        ("in allowed_gloss_exts", "in ALLOWED_GLOSSARY_EXTENSIONS"),
    ]),
    ("glossary_files", "mapped_glossary_for_input"): ([(TG, 31766, 31771)], [
        (_APPEND, "return gp"),
    ]),
    ("glossary_files", "build_glossary_mapping"): ([(TG, 31816, 31825)], [
        ("for _ep, _le in rows:", "for _ep, _text in rows:"),
        ("p = _le.text().strip()", "p = _text.strip()"),
        (_APPEND, "return mapping, missing"),
    ]),
    ("glossary_files", "copy_mapped_glossaries_to_outputs"): ([(TG, 31884, 31971)], [
        ("self.config.get('output_directory') if hasattr(self, 'config') else None",
         "config.get('output_directory') if config is not None else None"),
        ("re", r"self\.append_log\(", "append_log("),
        ("    )\nexcept Exception as e:", "    )\n    return copied, skipped, failed\nexcept Exception as e:"),
        (_APPEND, "    return None"),
    ]),
    ("glossary_files", "comprehensive_json_fix"): ([(TG, 32290, 32373)], [
        # this line continues a triple-quoted dict key: it keeps its absolute indentation
        ("\n    ''': \"'\",  # Right smart apostrophe", "\n        ''': \"'\",  # Right smart apostrophe"),
    ]),
    ("glossary_files", "analyze_json_errors"): ([(TG, 32377, 32414)], []),
    ("stop_control", "kill_glossary_helper_subprocesses"): ([(TG, 26652, 26718)], [
        (_PREPEND, "if not subprocesses_available():\n    return"),
    ]),
}

#: translator_gui.py rewire spans (frozen first line, frozen last line, new line count).
TG_REWIRE_SPANS = (
    # imports: request_glossary_stop, glossary_files
    (1332, 1331, 1), (1342, 1341, 4),
    # create_glossary_backup, _clean_old_backups
    (10349, 10350, 2), (10352, 10389, 2), (10391, 10392, 1), (10395, 10394, 9), (10436, 10456, 1),
    # _delete_current_glossary, _find_latest_backup, _restore_glossary_backup
    (19465, 19470, 4), (19475, 19564, 10), (19581, 19589, 1), (19604, 19616, 2), (19638, 19643, 4),
    (19647, 19672, 1), (19679, 19679, 0), (19705, 19715, 4),
    # parallel pair: glossary folder, mapping sidecar path / write / read (parallel_epub_core)
    (20048, 20062, 1), (20064, 20070, 2), (20078, 20083, 4), (20085, 20091, 0), (20096, 20104, 3),
    (20109, 20128, 3),
    # stop_glossary_extraction -> register_stop_click + stop_control.request_glossary_stop
    (26544, 26549, 4), (26589, 26611, 0), (26613, 26635, 6), (26637, 26743, 7),
    # parallel pair: chapter loader, working EPUB, activated-pair record, saved-pair rebuild
    (28460, 28460, 1), (28462, 28467, 1), (28472, 28475, 1), (28477, 28496, 1), (28519, 28518, 2),
    (28520, 28525, 0), (28537, 28550, 10), (28651, 28655, 6), (28657, 28696, 0),
    # Map Glossaries to EPUBs: drop filter, prefill, Save's mapping, Manual Glossary Only copy
    (31649, 31666, 3), (31766, 31771, 1), (31816, 31825, 3), (31884, 31971, 5),
    # _comprehensive_json_fix, _analyze_json_errors
    (32290, 32373, 1), (32377, 32414, 1),
)

#: The Stop protocol statements of stop_glossary_extraction, in order: each must appear, in
#: this order, in request_glossary_stop / kill_glossary_helper_subprocesses (ast.unparse text).
STOP_PROTOCOL = (
    "os.environ['GRACEFUL_STOP'] = '1' if graceful_stop else '0'",
    "os.environ['TRANSLATION_CANCELLED'] = '1'",
    "os.environ['GRACEFUL_STOP_COMPLETED'] = '0'",
    "glossary_stop_flag(True)",
    "extract_glossary_from_epub.set_stop_flag(True)",
    "unified_api_client.set_stop_flag(True)",
    "unified_api_client.UnifiedClient._global_cancelled = True",
    "if stop_run_id and current_run_id != stop_run_id:",
    "_uac.hard_cancel_all()",
    "f.write('stop')",
    "wait_for_chunks = os.environ.get('WAIT_FOR_CHUNKS') == '1'",
    "'⏳ Graceful stop — waiting for in-flight API calls to complete...'",
    "'🛑 Stop requested — cancelling glossary API calls (WAIT_FOR_CHUNKS=0)'",
    "'❌ Glossary extraction stop requested.'",
)


# =============================================================================================
# frozen sources
# =============================================================================================

_GIT_CACHE = {}


def git_text(relpath, sha=U6_BASE_SHA):
    key = (relpath, sha)
    if key not in _GIT_CACHE:
        try:
            data = subprocess.run(["git", "show", f"{sha}:{relpath}"], cwd=str(REPO_ROOT),
                                  capture_output=True, check=True).stdout
        except (OSError, subprocess.CalledProcessError) as exc:
            pytest.skip(f"frozen source {relpath}@{sha[:8]} unavailable: {exc}")
        _GIT_CACHE[key] = data.decode("utf-8-sig").replace("\r\n", "\n")
    return _GIT_CACHE[key]


def current_text(name):
    return (SRC / name).read_text(encoding="utf-8-sig").replace("\r\n", "\n")


def _norm_block(text):
    lines = [line.rstrip() for line in textwrap.dedent(text).split("\n")]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def _apply_edits(text, edits, name):
    for edit in edits:
        if edit[0] == "re":
            text = re.sub(edit[1], edit[2], text)
        elif edit[0] == _PREPEND:
            text = edit[1] + "\n" + text
        elif edit[0] == _APPEND:
            text = text + "\n" + edit[1]
        else:
            old, new = edit
            assert old in text, f"{name}: documented edit not found: {old!r}"
            text = text.replace(old, new)
    return text


def frozen_body(key):
    segments, edits = MOVED_BODIES[key]
    parts = []
    for rel, first, last in segments:
        lines = git_text(rel).split("\n")[first - 1:last]
        parts.append("\n".join("" if not line.strip() else line for line in lines))
    joined = textwrap.dedent("\n".join(parts))
    return _norm_block(_apply_edits(_norm_block(joined), edits, key))


_TREES = {}
_SOURCES = {}


def _tree(text):
    """ast.parse with a cache (translator_gui.py is ~1.5 MB; the fuzz loops need its methods often)."""
    key = hash(text)
    if key not in _TREES:
        _TREES[key] = ast.parse(text)
    return _TREES[key]


def function_body(text, name):
    tree = _tree(text)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    lines = text.split("\n")
    first = node.body[0]
    if isinstance(first, ast.Expr) and isinstance(getattr(first, "value", None), ast.Constant) \
            and isinstance(first.value.value, str):
        start = first.end_lineno
    else:
        start = first.lineno - 1
        while start - 1 > node.lineno - 1 and lines[start - 1].strip().startswith("#"):
            start -= 1
    return _norm_block("\n".join(lines[start:node.end_lineno]))


def method_source(text, name, cls="TranslatorGUI"):
    key = ("method", hash(text), name, cls)
    if key not in _SOURCES:
        tree = _tree(text)
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
        meth = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == name)
        lines = text.split("\n")
        start = (meth.decorator_list[0].lineno if meth.decorator_list else meth.lineno) - 1
        _SOURCES[key] = textwrap.dedent("\n".join(lines[start:meth.end_lineno]))
    return _SOURCES[key]


def nested_source(text, name):
    """Dedented source of the one (nested) function called ``name`` in a file."""
    key = ("nested", hash(text), name)
    if key not in _SOURCES:
        nodes = [n for n in ast.walk(_tree(text)) if isinstance(n, ast.FunctionDef) and n.name == name]
        assert len(nodes) == 1, (name, len(nodes))
        node = nodes[0]
        _SOURCES[key] = textwrap.dedent("\n".join(text.split("\n")[node.lineno - 1:node.end_lineno]))
    return _SOURCES[key]


@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def tg_module(qapp):
    import translator_gui
    return translator_gui


def _exec_in(ns, source, name, label):
    exec(compile(source, label, "exec"), ns)
    return ns[name]


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    """Never touch the real Library / output roots / app dir; restore the process env and cwd."""
    original = dict(os.environ)
    monkeypatch.setenv("GLOSSARION_LIBRARY_DIR", str(tmp_path / "_library"))
    for key in ("OUTPUT_DIRECTORY", "OUTPUT_DIR", "EPUB_PATH", "GLOSSARY_SHARED_DIR", "MANUAL_GLOSSARY",
                "APPEND_GLOSSARY", "FUZZY_AUTO_MAPPING", "FUZZY_AUTO_MAPPING_THRESHOLD", "GLOSSARY_STOP_FILE",
                "GLOSSARION_RUN_ID", "WAIT_FOR_CHUNKS", "GRACEFUL_STOP", "TRANSLATION_CANCELLED",
                "GRACEFUL_STOP_COMPLETED"):
        monkeypatch.delenv(key, raising=False)
    # glossary_files and the U3 auto-mapping helpers fall back to the app folder (src/ in a
    # checkout, with real glossaries the auto-mapper would migrate) without an output override
    guard = lambda: str(tmp_path / "_app_dir_guard")  # noqa: E731
    monkeypatch.setattr(gf, "_get_app_dir", guard)
    import translation_pipeline
    monkeypatch.setattr(translation_pipeline, "_get_app_dir", guard)
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
    except Exception as exc:
        return ("raise", type(exc).__name__, str(exc))


def _fresh_dir(root):
    os.chdir(str(TESTS))
    if Path(root).exists():
        shutil.rmtree(root)
    Path(root).mkdir(parents=True)


def _tree_bytes(root):
    out = {}
    root = Path(root)
    if root.exists():
        for path in sorted(root.rglob("*")):
            rel = path.relative_to(root).as_posix()
            out[rel + ("" if path.is_file() else "/")] = path.read_bytes() if path.is_file() else b""
    return out


# =============================================================================================
# H: hygiene
# =============================================================================================

def test_modules_are_gui_free_cheap_and_python_310():
    for name in ("glossary_files.py", "stop_control.py"):
        ast.parse((SRC / name).read_text(encoding="utf-8"), feature_version=(3, 10))
    code = (
        "import sys; sys.modules['PySide6'] = None; sys.path.insert(0, %r); "
        "import glossary_files as g, stop_control as s; "
        "heavy = [m for m in ('translator_gui', 'dpi_setup', 'PySide6', 'translation_pipeline', 'run_env', "
        "'extract_glossary_from_epub', 'unified_api_client') if sys.modules.get(m)]; "
        "print(heavy, g.build_glossary_mapping([])[0], callable(s.request_glossary_stop))" % str(SRC)
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == "[] {} True"
    assert all(hasattr(gf, name) for name in gf.__all__)
    assert {"request_glossary_stop", "kill_glossary_helper_subprocesses"} <= set(stop_control.__all__)


@pytest.mark.parametrize("name", ["glossary_files.py", "stop_control.py", "translator_gui.py"])
def test_line_endings_are_uniform_and_bom_unchanged(name):
    data = (SRC / name).read_bytes()
    assert data.count(b"\r\n") in (0, data.count(b"\n")), "mixed line endings"
    assert data.startswith(b"\xef\xbb\xbf") == (name == "translator_gui.py")


# =============================================================================================
# V: verbatim
# =============================================================================================

@pytest.mark.parametrize("key", sorted(MOVED_BODIES), ids=lambda k: f"{k[0]}.{k[1]}")
def test_moved_bodies_are_verbatim(key):
    module, name = key
    assert function_body(current_text(f"{module}.py"), name) == frozen_body(key)


def test_glossary_stop_protocol_keeps_every_frozen_statement_in_order():
    frozen = ast.unparse(ast.parse(method_source(git_text(TG), "stop_glossary_extraction")))
    tree = ast.parse(current_text("stop_control.py"))
    new = "\n".join(ast.unparse(n) for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name in ("request_glossary_stop",))
    position_frozen = position_new = -1
    for stmt in STOP_PROTOCOL:
        found_frozen = frozen.find(stmt, position_frozen + 1)
        found_new = new.find(stmt, position_new + 1)
        assert found_frozen > position_frozen, ("frozen", stmt)
        assert found_new > position_new, ("new", stmt)
        position_frozen, position_new = found_frozen, found_new
    # graceful-stop logger silencing: the same list stop_control already shares
    assert "['httpx', 'openai', 'google', 'google.api_core', 'google.generativeai', 'urllib3']" in frozen
    assert "silence_http_loggers()" in new
    # the wrapper keeps the widgets, the double-click rule and the idle poll
    wrapper = method_source(current_text("translator_gui.py"), "stop_glossary_extraction")
    for stmt in ("already_in_graceful_stop = (button_text == \"Finishing...\")",
                 "register_stop_click(self._glossary_stop_click_times, current_time)",
                 "request_glossary_stop(",
                 "QTimer.singleShot(500, self._reset_glossary_stop_flags_if_idle)"):
        assert stmt in wrapper, stmt


def _changed_spans(legacy, current):
    import difflib
    return tuple((i1 + 1, i2, j2 - j1) for tag, i1, i2, j1, j2 in
                 difflib.SequenceMatcher(None, legacy, current, autojunk=False).get_opcodes() if tag != "equal")


#: The U6 commit: the spans above are what U6 changed in translator_gui.py. Later milestones
#: (U7: the image / RPG Maker runners moved to image_job / rpgmaker_job, the Direct Text rules
#: and the GCP project rule to direct_text_store / authgem_auth) pin their own translator_gui
#: edits (tests/test_image_job.py::test_translator_gui_changed_only_in_the_u7_spans).
U6_COMMIT_SHA = "41814faa95e273e956870bd3aac5a2c6fb7d66b1"


def test_translator_gui_changed_only_in_rewired_spans():
    legacy = git_text(TG).split("\n")
    current = git_text(TG, sha=U6_COMMIT_SHA).split("\n")
    assert _changed_spans(legacy, current) == TG_REWIRE_SPANS


# =============================================================================================
# recording stand-ins
# =============================================================================================

class FakeEntry:
    def __init__(self, text=""):
        self._text = text

    def text(self):
        return self._text


def make_box_class(log, answers):
    """A QMessageBox stand-in: instances (setText / exec) and the static boxes, answers scripted."""

    class Box:
        Yes, No, Ok, Cancel = 0x4000, 0x10000, 0x400, 0x400000
        Information, Warning, Question, Critical = 1, 2, 4, 3

        def __init__(self, *_args):
            self.title = self.text = ""

        def setIcon(self, _icon):
            pass

        def setWindowTitle(self, title):
            self.title = title

        def setText(self, text):
            self.text = text

        def setStandardButtons(self, _buttons):
            pass

        def setDefaultButton(self, _button):
            pass

        def setStyleSheet(self, _style):
            pass

        def exec(self):
            answer = answers.pop(0) if answers else Box.Yes
            log.append(("box", self.title, self.text, answer))
            return answer

        @staticmethod
        def warning(_parent, title, text, *_rest):
            log.append(("warning", title, text))

        @staticmethod
        def information(_parent, title, text, *_rest):
            log.append(("information", title, text))

        @staticmethod
        def question(_parent, title, text, *_rest):
            answer = answers.pop(0) if answers else Box.Yes
            log.append(("question", title, text, answer))
            return answer

    return Box


class _Owner:
    """Attribute bag standing in for TranslatorGUI."""

    def __init__(self, log, **attrs):
        self._log = log
        self.__dict__.update(attrs)

    def append_log(self, message):
        self._log.append(("log", message))


class _FixedDatetime(_datetime_module.datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 10, 6, 12, 34, 56)


class _Stamp:
    """``time`` stand-in for the backup names: one deterministic stamp per backup."""

    def __init__(self):
        self.count = 0

    def strftime(self, _fmt):
        self.count += 1
        return f"20260101_{self.count:06d}"


# =============================================================================================
# D/F: editor backups (create_glossary_backup / _clean_old_backups)
# =============================================================================================

@needs_qt
def test_editor_backups_match_frozen(tmp_path, monkeypatch, tg_module):
    TranslatorGUI = tg_module.TranslatorGUI
    text = git_text(TG)
    exercised = {"created": 0, "pruned": 0, "failed": 0, "asked": 0, "skipped": 0}
    for state in range(STATES):
        root = tmp_path / "b"
        sides = []
        for side in ("frozen", "new", "shared"):
            _fresh_dir(root)
            r = random.Random(f"{SEED}-{state}")
            log = []
            answers = [r.choice([0x4000, 0x10000]) for _ in range(3)]
            Box = make_box_class(log, answers)
            stamp = _Stamp()
            glossary = root / r.choice(["book_glossary.csv", "a.json", "dir/b c.csv", "x"])
            glossary.parent.mkdir(parents=True, exist_ok=True)
            backups = glossary.parent / "Backups"
            op = r.choice(["manual", "before_save", "before_delete_3", "before_clean"])
            layout = r.random()
            if layout < 0.2:
                backups.write_text("not a folder", encoding="utf-8")  # makedirs fails
            elif layout < 0.75:
                backups.mkdir()
                stem = glossary.stem
                for i in range(r.randint(0, 6)):
                    name = r.choice([f"{stem}_auto_{i}.json", f"{stem}x{i}.json", f"other_{i}.json",
                                     f"{stem}_{i}.txt"])
                    path = backups / name
                    path.write_text(str(i), encoding="utf-8")
                    os.utime(path, (1_700_000_000 + i * 10, 1_700_000_000 + i * 10))
                if r.random() < 0.25:
                    (backups / f"{glossary.stem}_{op}_20260101_000001.json").mkdir()  # open() fails
            config = {}
            if r.random() < 0.8:
                config["glossary_auto_backup"] = r.choice([True, False, 0, 1])
            if r.random() < 0.8:
                config["glossary_max_backups"] = r.choice([0, 1, 2, 3, 50, -1])
            data = r.choice([None, [], {}, [{"raw_name": "김", "translated_name": "Kim"}], {"k": "v"},
                             [{"raw_name": "이", "translated_name": "Lee"}], {"a": "b"}])
            path_text = r.choice([str(glossary), "", str(glossary), str(glossary)])
            has_entry = r.random() < 0.95
            if side == "shared":
                monkeypatch.setattr(gf, "time", stamp)
                result = _run(gf.create_glossary_backup, path_text if has_entry else "", data, op, config=config,
                              append_log=lambda m: log.append(("log", m)),
                              ask_continue=lambda title, body: Box.question(None, title, body) == Box.Yes)
            else:
                owner = _Owner(log, config=config, current_glossary_data=data)
                if has_entry:
                    owner.editor_file_entry = FakeEntry(path_text)
                if side == "frozen":
                    ns = dict(vars(tg_module))
                    ns.update(QMessageBox=Box, time=stamp)
                    create = _exec_in(ns, method_source(text, "create_glossary_backup"), "create_glossary_backup",
                                      "<frozen create_glossary_backup>")
                    clean = _exec_in(ns, method_source(text, "_clean_old_backups"), "_clean_old_backups",
                                     "<frozen _clean_old_backups>")
                else:
                    monkeypatch.setattr(tg_module, "QMessageBox", Box)
                    monkeypatch.setattr(gf, "time", stamp)
                    create, clean = TranslatorGUI.create_glossary_backup, TranslatorGUI._clean_old_backups
                owner._clean_old_backups = types.MethodType(clean, owner)
                result = _run(types.MethodType(create, owner), op)
            sides.append((result, _tree_bytes(root), log))
        assert sides[1] == sides[0], state
        assert sides[2] == sides[0], state
        logs = " ".join(str(entry) for entry in sides[0][2])
        exercised["created"] += "Backup created" in logs
        exercised["pruned"] += "Removed old backup" in logs
        exercised["failed"] += "Failed to create backup directory" in logs or "Backup failed" in logs
        exercised["asked"] += "question" in logs
        exercised["skipped"] += not sides[0][2]
    assert all(count >= 5 for count in exercised.values()), exercised


# =============================================================================================
# D/F: delete / latest backup / restore (the auto-glossary row closures)
# =============================================================================================

_BOOK_NAMES = ("Book One", "[123] 소설", "a.b", "Novel_glossary", "x")


def _rand_glossary_layout(r, root, books):
    """Glossary artifacts of the books across the layouts the closures search."""
    base = root / r.choice(["out", "app"])
    gdir = base / "Glossary"
    for book in books:
        for _ in range(r.randint(0, 7)):
            where = r.random()
            ext = r.choice([".csv", ".json", ".txt", ".md", ".bak"])
            if where < 0.25:
                path = gdir / book / r.choice([f"{book}_glossary{ext}", f"{book}_glossary_progress.json",
                                               f"{book}_gender_tracker.json", f"{book}_glossary_history.json",
                                               f"{book}_other{ext}"])
            elif where < 0.45:
                path = gdir / r.choice([f"{book}_glossary{ext}", f"{book}_glossary_progress.json",
                                        f"{book}_gender_tracker.json", f"{book.lower()}_glossary{ext}"])
            elif where < 0.65:
                path = base / book / r.choice([f"glossary{ext}", "glossary_progress.json",
                                               f"{book}_glossary_progress.json", "translation_progress.json"])
            elif where < 0.8:
                path = base / book / "Glossary" / f"{book}_glossary{ext}"
            else:
                path = root / "elsewhere" / f"{book}_mapped{ext}"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"{book}:{path.name}", encoding="utf-8")
    return base


def _rand_backup_layout(r, root, books):
    for _ in range(r.randint(0, 6)):
        book = r.choice(books)
        parent = r.choice([root / "out" / "Glossary" / book, root / "out" / "Glossary", root / "out" / book,
                           root / "cwd" / "Glossary" / book, root / "cwd" / "Glossary", root / "app" / book,
                           root / "app" / "Glossary" / book])
        stamp = f"2026-0{r.randint(1, 9)}-1{r.randint(0, 9)}_1{r.randint(0, 9)}0000"
        folder = parent / "Backups" / stamp
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(r.randint(0, 3)):
            (folder / f"{book}_glossary{r.choice(['.csv', '.json'])}").write_text(f"v{i}", encoding="utf-8")
        if r.random() < 0.2:
            (folder / "sub").mkdir(exist_ok=True)
        if r.random() < 0.15:
            (parent / "Backups" / "loose.json").write_text("x", encoding="utf-8")


def _closure_ns(tg_module, text, names, Box, root, extra):
    ns = dict(vars(tg_module))
    ns.update(QMessageBox=Box, _get_app_dir=lambda: str(root / "app"))
    ns.update(extra)
    funcs = {}
    for name in names:
        funcs[name] = _exec_in(ns, nested_source(text, name), name, f"<{name}>")
    return funcs


class _FakeWinsound:
    SND_ALIAS = 1
    SND_ASYNC = 2
    MB_OK = 0

    def __init__(self, log):
        self.log = log

    def PlaySound(self, name, flags):
        self.log.append(("sound", name))

    def MessageBeep(self, kind):
        self.log.append(("beep", kind))


def _owner_for_files(tg_module, log, r, root, epubs, config):
    TranslatorGUI = tg_module.TranslatorGUI
    owner = _Owner(log, config=config, base_dir=str(root / "base"))
    owner.selected_files = list(epubs)
    if r.random() < 0.3:
        owner.get_current_epub_path = lambda: r_current
    r_current = r.choice([None, "", str(root / "in" / "Current Book.epub")])
    owner.auto_loaded_glossary_path = r.choice([None, str(root / "elsewhere" / "Book One_mapped.csv")])
    owner.manual_glossary_path = r.choice([None, str(root / "out" / "Glossary" / "x_glossary.csv")])
    owner.manual_glossary_manually_loaded = r.random() < 0.3
    owner.auto_loaded_glossary_for_file = "something"
    for name in ("_guess_glossary_for_input_file", "_get_glossary_dir_candidates", "_glossary_dir_signature"):
        setattr(owner, name, types.MethodType(getattr(TranslatorGUI, name), owner))
    owner.auto_load_glossary_for_file = lambda path: log.append(("auto_load", os.path.basename(path)))
    return owner


@needs_qt
def test_delete_and_restore_closures_match_frozen(tmp_path, monkeypatch, tg_module):
    frozen_text = git_text(TG)
    new_text = current_text("translator_gui.py")
    monkeypatch.setattr(_datetime_module, "datetime", _FixedDatetime)
    exercised = {"deleted": 0, "nothing": 0, "declined": 0, "restored": 0, "no_backup": 0, "no_file": 0}
    for state in range(STATES):
        root = tmp_path / "d"
        sides = []
        for side, text in (("frozen", frozen_text), ("new", new_text)):
            _fresh_dir(root)
            r = random.Random(f"{SEED}-d-{state}")
            log = []
            Box = make_box_class(log, [r.choice([0x4000, 0x10000]) for _ in range(3)])
            monkeypatch.setitem(sys.modules, "winsound", _FakeWinsound(log))
            monkeypatch.setattr(gf, "_get_app_dir", lambda _root=root: str(_root / "app"))
            monkeypatch.setattr(sys.modules["translation_pipeline"], "_get_app_dir",
                                lambda _root=root: str(_root / "app"))
            books = r.sample(_BOOK_NAMES, r.randint(1, 3))
            _rand_glossary_layout(r, root, books)
            _rand_backup_layout(r, root, books)
            (root / "cwd").mkdir(exist_ok=True)
            os.chdir(root / "cwd")
            override = r.choice([str(root / "out"), None])
            config = {"auto_glossary_mode": r.choice(["off", "balanced", "Full", "minimal"])}
            if override and r.random() < 0.5:
                config["output_directory"] = override
            elif override:
                monkeypatch.setenv("OUTPUT_DIRECTORY", override)
            else:
                monkeypatch.delenv("OUTPUT_DIRECTORY", raising=False)
            if r.random() < 0.3:
                monkeypatch.setenv("FUZZY_AUTO_MAPPING", "1")
            else:
                monkeypatch.delenv("FUZZY_AUTO_MAPPING", raising=False)
            epubs = [str(root / "in" / f"{b}.epub") for b in books]
            if r.random() < 0.2:
                epubs = [str(root / "in" / "notes.txt")]
            os.environ["MANUAL_GLOSSARY"] = "set"
            owner = _owner_for_files(tg_module, log, r, root, epubs, config)
            visibility = []
            funcs = _closure_ns(tg_module, text, ("_find_latest_backup", "_restore_glossary_backup",
                                                  "_delete_current_glossary"), Box, root,
                                {"self": owner, "_update_restore_visibility": lambda: visibility.append(1)})
            action = r.choice(["delete", "delete", "find", "restore", "restore"])
            if action == "delete":
                result = _run(funcs["_delete_current_glossary"])
            elif action == "find":
                result = _run(funcs["_find_latest_backup"])
            else:
                result = _run(funcs["_restore_glossary_backup"])
            state_after = {k: getattr(owner, k, "<unset>") for k in (
                "auto_loaded_glossary_path", "auto_loaded_glossary_for_file", "manual_glossary_path",
                "manual_glossary_manually_loaded")}
            sides.append((action, result, _tree_bytes(root), log, state_after, os.environ.get("MANUAL_GLOSSARY"),
                          visibility))
        assert sides[1] == sides[0], state
        logs = " ".join(str(entry) for entry in sides[0][3])
        exercised["deleted"] += "files backed up" in logs
        exercised["nothing"] += "Nothing to Delete" in logs
        exercised["declined"] += "Delete Glossary" in logs and "65536" in logs
        exercised["restored"] += "Restored from" in logs
        exercised["no_backup"] += sides[0][0] == "find" and sides[0][1] == ("ok", (None, []))
        exercised["no_file"] += "No input file selected" in logs
    assert all(count >= 5 for count in exercised.values()), exercised


@needs_qt
def test_glossary_file_helpers_match_frozen_on_layouts(tmp_path, monkeypatch, tg_module):
    """The mobile entry points (no owner) give what the frozen closures compute."""
    frozen_text = git_text(TG)
    monkeypatch.setattr(_datetime_module, "datetime", _FixedDatetime)
    for state in range(STATES // 2):
        root = tmp_path / "h"
        sides = []
        for side in ("frozen", "shared"):
            _fresh_dir(root)
            r = random.Random(f"{SEED}-h-{state}")
            log = []
            Box = make_box_class(log, [0x4000, 0x4000, 0x4000])
            monkeypatch.setitem(sys.modules, "winsound", _FakeWinsound([]))
            monkeypatch.setattr(gf, "_get_app_dir", lambda _root=root: str(_root / "app"))
            books = r.sample(_BOOK_NAMES, r.randint(1, 3))
            _rand_glossary_layout(r, root, books)
            _rand_backup_layout(r, root, books)
            (root / "cwd").mkdir(exist_ok=True)
            os.chdir(root / "cwd")
            config = {"output_directory": str(root / "out")} if r.random() < 0.7 else {}
            epubs = [str(root / "in" / f"{b}.epub") for b in books]
            owner = _owner_for_files(tg_module, log, r, root, epubs, config)
            if side == "frozen":
                funcs = _closure_ns(tg_module, frozen_text, ("_find_latest_backup", "_restore_glossary_backup",
                                                             "_delete_current_glossary"), Box, root,
                                    {"self": owner, "_update_restore_visibility": lambda: None})
                latest = funcs["_find_latest_backup"]()
                funcs["_restore_glossary_backup"]()
                funcs["_delete_current_glossary"]()
            else:
                epub_list = gf.selected_glossary_epubs(owner.selected_files,
                                                       getattr(owner, "get_current_epub_path", None))
                latest = gf.find_latest_glossary_backup(epub_list, config=config) if epub_list else (None, [])
                if latest[0] and latest[1]:
                    restored = gf.restore_glossary_backup(latest[0], latest[1], owner.append_log)
                    if restored:
                        log.append(("log", f"↩️ Restored from {os.path.basename(latest[0])}: {', '.join(restored)}"))
                        if len([p for p in owner.selected_files if p.lower().endswith('.epub')]) == 1:
                            owner.auto_load_glossary_for_file(owner.selected_files[0])
                files = gf.collect_glossary_files_for_inputs(
                    epub_list, config=config, guess_glossary=owner._guess_glossary_for_input_file,
                    auto_loaded_glossary_path=owner.auto_loaded_glossary_path,
                    manual_glossary_path=owner.manual_glossary_path,
                    manually_loaded=owner.manual_glossary_manually_loaded) if epub_list else []
                if files:
                    deleted = gf.delete_glossary_files(files, owner.append_log)
                    if deleted:
                        log.append(("log", f"🗑️ Deleted ({len(deleted)} files backed up): {', '.join(deleted)}"))
            sides.append((latest, _tree_bytes(root),
                          [e for e in log if e[0] in ("log", "auto_load")]))
        assert sides[1] == sides[0], state


def test_delete_display_and_mapping_helpers():
    files = [("A", "/x/A_glossary.csv"), ("B", "/y/glossary.json"), ("A", "/x/A_gender_tracker.json")]
    assert gf.glossary_delete_display(files) == ["[A]", "  A_glossary.csv", "  A_gender_tracker.json", "[B]",
                                                 "  glossary.json"]
    assert gf.mapped_glossary_for_input({"a.epub": "g.csv"}, "a.epub") == "g.csv"
    assert gf.mapped_glossary_for_input(None, "a.epub") is None
    assert gf.normalize_glossary_drop_path('  "x.csv" ') == os.path.normpath(os.path.abspath("x.csv"))
    assert gf.normalize_glossary_drop_path(None) == ""


# =============================================================================================
# D: JSON repair helpers
# =============================================================================================

_JSON_SNIPPETS = (
    '{"a": 1, "b": [1, 2, 3]}', "{'a': 'b'}", '{a: 1}', '{"a": 1,}', '[1, 2,]', '{"a": "x\\y"}',
    '// comment\n{"a": 1}', '/* c */ {"a": 2}', '{"a": 1} {"b": 2}', '[1] [2]', '﻿{"x": 1}',
    '{"q": “smart”}', "{'x': ‘y’}", '{"d": "a—b–c…"}', '{"z": "​ "}', '{"a": [1, {"b": 2}',
    '{"k": "v"', '"a" "b":', ',,,', '', '{"a": 1}\n,', 'not json at all', '{"a": "line1\nline2"}',
    '{"x": tru}', '{"x": undefined}', "{ok: 'fine', 'n': 3,}",
)


@needs_qt
def test_json_repair_helpers_match_frozen(tg_module):
    text = git_text(TG)
    ns = dict(vars(tg_module))
    fix = _exec_in(ns, method_source(text, "_comprehensive_json_fix"), "_comprehensive_json_fix", "<fix>")
    analyze = _exec_in(ns, method_source(text, "_analyze_json_errors"), "_analyze_json_errors", "<analyze>")
    TranslatorGUI = tg_module.TranslatorGUI
    rng = random.Random(SEED + 3)
    for state in range(STATES):
        content = "".join(rng.choice(_JSON_SNIPPETS) for _ in range(rng.randint(1, 3)))
        if rng.random() < 0.3:
            content = content.replace(rng.choice(['"', "{", ",", ":", "]"]), rng.choice(["'", "", ",,", " "]))
        fixed = fix(None, content)
        assert gf.comprehensive_json_fix(content) == fixed, state
        assert TranslatorGUI._comprehensive_json_fix(None, content) == fixed, state
        errors = []
        for candidate in (content, fixed):
            try:
                json.loads(candidate)
                errors.append(ValueError("ok"))
            except json.JSONDecodeError as exc:
                errors.append(exc)
        expected = analyze(None, content, fixed, errors[0], errors[1])
        assert gf.analyze_json_errors(content, fixed, errors[0], errors[1]) == expected, state
        assert TranslatorGUI._analyze_json_errors(None, content, fixed, errors[0], errors[1]) == expected


# =============================================================================================
# D: the owner-free auto-mapping adapters (U3 mixin methods)
# =============================================================================================

@needs_qt
def test_owner_free_adapters_match_the_mixin_methods(tmp_path, monkeypatch, tg_module):
    TranslatorGUI = tg_module.TranslatorGUI
    import translation_pipeline
    root = tmp_path / "g"
    for state in range(STATES // 2):
        sides = {}
        for side in ("mixin", "adapter"):
            _fresh_dir(root)
            os.chdir(root)
            r = random.Random(f"{SEED}-g-{state}")
            books = r.sample(_BOOK_NAMES, r.randint(1, 3))
            _rand_glossary_layout(r, root, books)
            monkeypatch.setattr(gf, "_get_app_dir", lambda _root=root: str(_root / "app"))
            monkeypatch.setattr(translation_pipeline, "_get_app_dir", lambda _root=root: str(_root / "app"))
            if r.random() < 0.4:
                monkeypatch.setenv("FUZZY_AUTO_MAPPING", "1")
                monkeypatch.setenv("FUZZY_AUTO_MAPPING_THRESHOLD", r.choice(["60", "80", "95", "x"]))
            else:
                monkeypatch.delenv("FUZZY_AUTO_MAPPING", raising=False)
            config = {"output_directory": str(root / "out")} if r.random() < 0.6 else {}
            base_dir = r.choice(["", str(root / "app"), str(root / "out")])
            log = []
            owner = _Owner(log, config=config, base_dir=base_dir)
            for name in ("_guess_glossary_for_input_file", "_get_glossary_dir_candidates", "_glossary_dir_signature",
                         "_copy_glossary_to_output_folders", "_resolve_translation_output_dir",
                         "_subtitle_zip_output_info", "_get_output_base_dir"):
                setattr(owner, name, types.MethodType(getattr(TranslatorGUI, name), owner))
            guesses = []
            for book in books + ["Unknown"]:
                path = str(root / "in" / f"{book}{r.choice(['.epub', '.txt', '.EPUB'])}")
                if side == "mixin":
                    guesses.append(owner._guess_glossary_for_input_file(path))
                else:
                    guesses.append(gf.guess_glossary_for_input_file(path, config, base_dir=base_dir))
            glossaries = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix in (".csv", ".json", ".md", ".txt"))
            copy_result = None
            if glossaries:
                source = str(r.choice(glossaries))
                inputs = [str(root / "in" / f"{b}.epub") for b in books] + r.choice([[], ["tool.exe"]])
                if side == "mixin":
                    copy_result = owner._copy_glossary_to_output_folders(source, input_files=list(inputs))
                else:
                    copy_result = gf.copy_glossary_to_output_folders(source, inputs, config=config,
                                                                      append_log=lambda m: log.append(("log", m)))
            sides[side] = (guesses, copy_result, _tree_bytes(root), log)
        assert sides["adapter"] == sides["mixin"], state


# =============================================================================================
# T: the glossary Stop protocol
# =============================================================================================

STOP_ENV = ("GRACEFUL_STOP", "TRANSLATION_CANCELLED", "GRACEFUL_STOP_COMPLETED", "WAIT_FOR_CHUNKS")
LOGGERS = ("httpx", "openai", "google", "google.api_core", "google.generativeai", "urllib3")


class _Trace:
    def __init__(self, owner_ref):
        self.events = []
        self.owner_ref = owner_ref

    def __call__(self, name, *args):
        owner = self.owner_ref()
        self.events.append((name, args, tuple(os.environ.get(k) for k in STOP_ENV),
                            getattr(owner, "stop_requested", None) if owner is not None else None))


class _TracedOwner(_Owner):
    def __setattr__(self, name, value):
        trace = self.__dict__.get("_trace")
        if trace is not None and name in ("stop_requested", "_glossary_stop_click_times"):
            trace("set:" + name, value if name == "stop_requested" else len(value))
        object.__setattr__(self, name, value)


class _FakeProc:
    def __init__(self, pid, cmd, trace):
        self.pid, self.cmd, self.trace = pid, cmd, trace

    def cmdline(self):
        if self.cmd is None:
            raise PermissionError("denied")
        return self.cmd

    def terminate(self):
        self.trace("terminate", self.pid)

    def kill(self):
        self.trace("kill", self.pid)


def _fake_psutil(trace, children, alive_pids):
    class NoSuchProcess(Exception):
        pass

    class AccessDenied(Exception):
        pass

    def wait_procs(procs, timeout=None):
        trace("wait_procs", tuple(p.pid for p in procs), timeout)
        alive = [p for p in procs if p.pid in alive_pids]
        return [p for p in procs if p not in alive], alive

    process = types.SimpleNamespace(children=lambda recursive=False: list(children))
    return types.SimpleNamespace(Process=lambda pid: process, wait_procs=wait_procs,
                                 NoSuchProcess=NoSuchProcess, AccessDenied=AccessDenied)


def _stop_scenario(r):
    return {
        "graceful": r.choice([True, False, None, "missing"]),
        "label": r.choice([None, "Finishing...", "Stopping...", "Extract Glossary"]),
        "button": r.random() < 0.7,
        "clicks": [r.choice([-5.0, -0.5, -0.2]) for _ in range(r.randint(0, 2))],
        "wait_for_chunks": r.choice([None, "1", "0"]),
        "stop_file": r.random() < 0.6,
        "run_id": r.choice([None, "glossary-aaa"]),
        "env_run_id": r.choice([None, "glossary-aaa", "glossary-bbb"]),
        "new_run_before_cleanup": r.random() < 0.25,
        "stop_flag": r.random() < 0.6,
        "extract_has_flag": r.random() < 0.8,
        "client": r.choice(["full", "partial", "missing"]),
        "children": [(100 + i, r.choice([["python", "chapter_extraction_worker.py"], ["x", "--run-pdf-extraction"],
                                         ["python", "-c", "multiprocessing.spawn"], ["notepad"], None,
                                         ["pdf_extractor", "--authnd-mint-token"], ["x", "--gemini-free-search"]]))
                     for i in range(r.randint(0, 4))],
        "alive": r.random() < 0.5,
        "protected": r.random() < 0.3,
    }


def _install_stop_backends(monkeypatch, trace, sc, tmp_path):
    for key in ("GRACEFUL_STOP", "TRANSLATION_CANCELLED", "GRACEFUL_STOP_COMPLETED"):
        monkeypatch.delenv(key, raising=False)
    for key, value in (("WAIT_FOR_CHUNKS", sc["wait_for_chunks"]), ("GLOSSARION_RUN_ID", sc["env_run_id"])):
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)
    stop_file = tmp_path / "glossary.stop"
    if stop_file.exists():
        stop_file.unlink()
    if sc["stop_file"]:
        monkeypatch.setenv("GLOSSARY_STOP_FILE", str(stop_file))
    else:
        monkeypatch.delenv("GLOSSARY_STOP_FILE", raising=False)
    extract = types.ModuleType("extract_glossary_from_epub")
    if sc["extract_has_flag"]:
        extract.set_stop_flag = lambda value: trace("extract.set_stop_flag", value)
    monkeypatch.setitem(sys.modules, "extract_glossary_from_epub", extract)
    client = types.ModuleType("unified_api_client")
    if sc["client"] != "missing":
        client.set_stop_flag = lambda value: trace("client.set_stop_flag", value)
        client.hard_cancel_all = lambda: trace("client.hard_cancel_all")
    if sc["client"] == "full":
        class UnifiedClient:
            pass
        client.UnifiedClient = UnifiedClient
    monkeypatch.setitem(sys.modules, "unified_api_client", client)
    children = [_FakeProc(pid, cmd, trace) for pid, cmd in sc["children"]]
    alive = {pid for pid, _cmd in sc["children"]} if sc["alive"] else set()
    monkeypatch.setitem(sys.modules, "psutil", _fake_psutil(trace, children, alive))
    manga = types.ModuleType("manga_translator")
    pool = {}
    if sc["protected"] and children:
        worker = types.SimpleNamespace(_mp_worker=types.SimpleNamespace(pid=children[0].pid))
        pool = {"k": {"spares": [worker, None]}}
    manga.MangaTranslator = types.SimpleNamespace(_inpaint_pool=pool)
    monkeypatch.setitem(sys.modules, "manga_translator", manga)
    for name in LOGGERS:
        monkeypatch.setattr(logging.getLogger(name), "level", logging.NOTSET)
    return stop_file


def _stop_owner(sc, trace, log):
    owner = _TracedOwner(log)
    if sc["graceful"] != "missing":
        owner.graceful_stop_var = sc["graceful"]
    if sc["label"] is not None:
        owner.glossary_text_label = types.SimpleNamespace(
            text=lambda _t=sc["label"]: _t, setText=lambda t: trace("label.setText", t))
    if sc["button"]:
        owner.glossary_button = types.SimpleNamespace(setEnabled=lambda v: trace("button.setEnabled", v),
                                                      setStyleSheet=lambda s: trace("button.style"))
    if sc["clicks"]:
        owner._glossary_stop_click_times = [1000.0 + d for d in sc["clicks"]]
    if sc["run_id"] is not None:
        owner._glossary_run_id = sc["run_id"]
    owner._reset_glossary_stop_flags_if_idle = lambda: trace("reset_if_idle")
    owner.append_log = lambda m: (log.append(m), trace("log", m))
    owner.__dict__["_trace"] = trace
    return owner


class _TraceThread:
    """threading.Thread stand-in: records the start and runs the target (a newer run may start first)."""

    def __init__(self, sc, trace, owner_ref):
        self.sc, self.trace, self.owner_ref = sc, trace, owner_ref

    def __call__(self, target=None, daemon=None, name=None, **_kw):
        sc, trace, owner_ref = self.sc, self.trace, self.owner_ref

        class _Started:
            def start(self_inner):
                trace("thread.start", name, daemon)
                if sc["new_run_before_cleanup"]:
                    os.environ["GLOSSARION_RUN_ID"] = "glossary-newer"
                    owner = owner_ref()
                    if owner is not None and "_glossary_run_id" in owner.__dict__:
                        owner.__dict__["_glossary_run_id"] = "glossary-newer"
                target()
                trace("thread.done")

            def is_alive(self_inner):
                return False

        return _Started()


@needs_qt
def test_glossary_stop_trace_matches_frozen(tmp_path, monkeypatch, tg_module):
    frozen_text = git_text(TG)
    TranslatorGUI = tg_module.TranslatorGUI
    monkeypatch.setattr(time, "time", lambda: 1000.0)
    monkeypatch.setattr(stop_control, "subprocesses_available", lambda: True)
    exercised = {"graceful": 0, "immediate": 0, "double": 0, "guarded": 0, "killed": 0}
    for state in range(STATES):
        r = random.Random(f"{SEED}-s-{state}")
        sc = _stop_scenario(r)
        sides = []
        for side in ("frozen", "new", "mixins"):
            log = []
            holder = {}
            trace = _Trace(lambda: holder.get("owner"))
            stop_file = _install_stop_backends(monkeypatch, trace, sc, tmp_path)
            owner = _stop_owner(sc, trace, log)
            holder["owner"] = owner
            flag = (lambda value: trace("glossary_stop_flag", value)) if sc["stop_flag"] else None
            thread_cls = _TraceThread(sc, trace, lambda: holder.get("owner"))
            timer = types.SimpleNamespace(singleShot=lambda ms, fn: (trace("QTimer.singleShot", ms), fn()))
            if side == "frozen":
                ns = dict(vars(tg_module))
                ns.update(glossary_stop_flag=flag, QTimer=timer)
                monkeypatch.setattr(sys.modules["threading"], "Thread", thread_cls)
                stop = _exec_in(ns, method_source(frozen_text, "stop_glossary_extraction"),
                                "stop_glossary_extraction", "<frozen stop_glossary_extraction>")
                result = _run(types.MethodType(stop, owner))
            elif side == "new":
                monkeypatch.setattr(tg_module, "glossary_stop_flag", flag)
                monkeypatch.setattr(tg_module, "QTimer", timer)
                monkeypatch.setattr(stop_control.threading, "Thread", thread_cls)
                result = _run(types.MethodType(TranslatorGUI.stop_glossary_extraction, owner))
            else:
                # the mobile composition: the effective mode (no Finishing... label on mobile, so the
                # desktop's double click never applies here) and the owner's latch / run id
                graceful = getattr(owner, "graceful_stop_var", False)
                if sc["label"] == "Finishing..." and len(
                        [t for t in getattr(owner, "_glossary_stop_click_times", []) + [1000.0]
                         if 1000.0 - t < 1.0]) >= 2:
                    log.append("⚡ Double-click detected — forcing immediate stop!")
                    trace("log", "⚡ Double-click detected — forcing immediate stop!")
                    graceful = False
                monkeypatch.setattr(stop_control.threading, "Thread", thread_cls)

                def latch(_owner=owner):
                    _owner.stop_requested = True

                result = _run(stop_control.request_glossary_stop, graceful=graceful, set_stop_requested=latch,
                              log=owner.append_log, glossary_stop_flag=flag,
                              get_run_id=lambda _owner=owner: getattr(_owner, "_glossary_run_id", None))
                result = ("ok", None) if result[0] == "ok" else result
            protocol = [e for e in trace.events if not e[0].startswith(("label.", "button.", "QTimer", "reset_if",
                                                                         "set:_glossary_stop_click"))]
            sides.append({
                "result": result,
                "trace": trace.events,
                "protocol": protocol,
                "log": log,
                "env": {k: os.environ.get(k) for k in STOP_ENV + ("GLOSSARION_RUN_ID",)},
                "stop_requested": getattr(owner, "stop_requested", None),
                "stop_file": stop_file.read_text(encoding="utf-8") if stop_file.exists() else None,
                "loggers": [logging.getLogger(n).level for n in LOGGERS],
                "client_cancelled": getattr(getattr(sys.modules["unified_api_client"], "UnifiedClient", None),
                                            "_global_cancelled", None),
            })
        assert sides[1] == sides[0], (state, sc)
        mobile = dict(sides[2])
        desktop = dict(sides[0])
        assert mobile.pop("protocol") == desktop.pop("protocol"), (state, sc)
        for key in ("trace", "result"):
            mobile.pop(key)
            desktop.pop(key)
        assert mobile == desktop, (state, sc)
        text = " ".join(desktop["log"])
        exercised["graceful"] += "Graceful stop" in text or "Stop requested" in text
        exercised["immediate"] += "Glossary extraction stop requested" in text
        exercised["double"] += "Double-click" in text
        exercised["guarded"] += sc["new_run_before_cleanup"] and "Glossary extraction stop requested" in text
        exercised["killed"] += any(e[0] == "terminate" for e in sides[0]["trace"])
    assert all(count >= 10 for count in exercised.values()), exercised


# =============================================================================================
# S: the Map Glossaries to EPUBs dialog, offscreen (frozen vs working tree)
# =============================================================================================

def _mapping_owner_class():
    from PySide6.QtWidgets import QWidget

    class MappingOwner(QWidget):
        def __init__(self, log, config, base_dir, mode_var):
            super().__init__()
            self._log = log
            self.config = config
            self.base_dir = base_dir
            self.manual_glossary_map = {}
            self.manual_glossary_path = "previous.csv"
            self.manual_glossary_manually_loaded = False
            self.append_glossary_var = False
            if mode_var is not None:
                self.auto_glossary_mode_var = mode_var

        def append_log(self, message):
            self._log.append(("log", message))

        def save_config(self, show_message=True):
            self._log.append(("save_config", show_message))

    return MappingOwner


def _drive_mapping_dialog(qapp, owner, open_fn, r, glossaries):
    from PySide6.QtWidgets import QLineEdit, QPushButton
    open_fn(owner)
    dialog = owner._glossary_mapping_dialog
    edits = [w for w in dialog.findChildren(QLineEdit)]
    buttons = {b.text(): b for b in dialog.findChildren(QPushButton)}
    prefill = [e.text() for e in edits]
    for _ in range(r.randint(0, 3)):
        action = r.choice(["autofill", "clear", "one", "set", "set"])
        if action == "autofill":
            buttons["Auto-Fill"].click()
        elif action == "clear":
            buttons["Clear All"].click()
        elif action == "one":
            buttons["Use one Glossary"].click()
        elif edits:
            value = r.choice(["missing.csv", "", "  "]) if r.random() < 0.35 else r.choice(glossaries or [""])
            r.choice(edits).setText(value)
    filled = [e.text() for e in edits]
    if r.random() < 0.85:
        buttons["Save"].click()
    else:
        buttons["Cancel"].click()
    qapp.processEvents()
    return prefill, filled


@needs_qt
def test_map_glossaries_dialog_offscreen_matches_frozen(qapp, tmp_path, monkeypatch, tg_module):
    from PySide6.QtWidgets import QFileDialog, QMessageBox
    frozen_text = git_text(TG)
    TranslatorGUI = tg_module.TranslatorGUI
    ns = dict(vars(tg_module))
    frozen_open = _exec_in(ns, method_source(frozen_text, "_open_glossary_mapping_dialog"),
                           "_open_glossary_mapping_dialog", "<frozen _open_glossary_mapping_dialog>")
    MappingOwner = _mapping_owner_class()
    exercised = {"saved": 0, "missing": 0, "copied": 0, "autofill_none": 0}
    for run in range(DIALOG_RUNS):
        root = tmp_path / "m"
        sides = []
        for side in ("frozen", "new"):
            _fresh_dir(root)
            os.chdir(root)
            r = random.Random(f"{SEED}-m-{run}")
            log = []
            monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda _p, t, x, *a: log.append(("warning", t, x))))
            monkeypatch.setattr(QMessageBox, "information",
                                staticmethod(lambda _p, t, x, *a: log.append(("information", t, x))))
            books = r.sample(_BOOK_NAMES, r.randint(2, 4))
            _rand_glossary_layout(r, root, books)
            glossaries = sorted(str(p) for p in root.rglob("*") if p.is_file())
            pick = r.choice(glossaries + [""]) if glossaries else ""
            monkeypatch.setattr(QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (pick, "")))
            out = r.choice([str(root / "outputs"), ""])
            if out and r.random() < 0.5:
                monkeypatch.setenv("OUTPUT_DIRECTORY", out)
            else:
                monkeypatch.delenv("OUTPUT_DIRECTORY", raising=False)
            config = {"output_directory": out} if out and r.random() < 0.7 else {}
            if r.random() < 0.5:
                config["auto_glossary_mode"] = r.choice(["off_no_automap", "balanced", ""])
            mode_var = r.choice([None, "off_no_automap", "OFF_NO_AUTOMAP", "minimal"])
            owner = MappingOwner(log, config, str(root / "out"), mode_var)
            owner._guess_glossary_for_input_file = types.MethodType(TranslatorGUI._guess_glossary_for_input_file, owner)
            owner._get_glossary_dir_candidates = types.MethodType(TranslatorGUI._get_glossary_dir_candidates, owner)
            owner._glossary_dir_signature = types.MethodType(TranslatorGUI._glossary_dir_signature, owner)
            epubs = [str(root / "in" / f"{b}.epub") for b in books] + r.choice([[], [str(root / "in" / "x.txt")]])
            existing = {}
            if r.random() < 0.4 and glossaries:
                existing[os.path.normpath(os.path.abspath(epubs[0]))] = r.choice(glossaries)
            owner.manual_glossary_map = existing
            open_fn = (lambda o, _e=epubs: frozen_open(o, _e)) if side == "frozen" else \
                (lambda o, _e=epubs: TranslatorGUI._open_glossary_mapping_dialog(o, _e))
            prefill, filled = _drive_mapping_dialog(qapp, owner, open_fn, r, glossaries)
            dialog = getattr(owner, "_glossary_mapping_dialog", None)
            if dialog is not None:
                dialog.close()
                qapp.processEvents()
            sides.append({
                "prefill": prefill, "filled": filled, "log": log,
                "state": {k: getattr(owner, k, None) for k in (
                    "manual_glossary_map", "manual_glossary_path", "manual_glossary_manually_loaded",
                    "append_glossary_var")},
                "config": copy.deepcopy(config), "env": os.environ.get("APPEND_GLOSSARY"),
                "tree": _tree_bytes(root),
            })
            os.environ.pop("APPEND_GLOSSARY", None)
            owner.deleteLater()
            qapp.processEvents()
        assert sides[1] == sides[0], run
        text = " ".join(str(e) for e in sides[0]["log"])
        exercised["saved"] += "Saved glossary mapping" in text
        exercised["missing"] += "Missing glossary file" in text
        exercised["copied"] += "Manual Glossary Only: mapping applied" in text
        exercised["autofill_none"] += "No matching glossaries" in text
    assert exercised["saved"] >= 3 and exercised["copied"] >= 1 and exercised["missing"] >= 1, exercised
