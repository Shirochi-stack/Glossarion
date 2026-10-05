"""text_jobs / input_preparation and the U3 TranslatorGUI rewiring (Glossarion mobile rewrite, milestone U3).

TranslatorGUI's per-file runners (``_process_text_file``, ``_extract_glossary_from_text_file``,
``_run_parallel_metadata_files``, the compile runners' try/except) moved into
``text_jobs.TextJobsMixin`` and its ZIP / HTML / subtitle input resolution into
``input_preparation.InputPreparationMixin`` (+ ``resolve_input_to_epub``); TranslatorGUI
inherits them first, ``HeadlessOwner`` runs the same code on mobile.

What is checked:
* import hygiene (PySide6 blocked; no translator_gui / dpi_setup / epub_library) and Python
  3.10 syntax of the new and touched modules;
* the moves are verbatim: each moved body equals the frozen legacy body (git show of the
  oracle SHA) after exactly the documented edits (AST compare);
* MRO: the U3 mixins come first, moved bodies left TranslatorGUI, hooks are overridden only
  by the desktop, no GUI mixin shadows a moved name; the desktop compile runners are thin
  wrappers keeping their ``finally``;
* tier G for HeadlessOwner: the job entries (per-file translation, glossary extraction,
  EPUB / PDF compile) reproduce the desktop goldens (env, argv and cwd at the stubbed backend,
  logs, client-pool calls, stdout) for all 12 golden scenarios;
* ``resolve_input_to_epub`` used directly (mobile FileBridge) behaves like the owner method.

The behavioural parity of every moved method against the frozen oracle is tier D
(``tests/parity/test_parity_tiers.py``, >= 500 states each); the restore order is tier T
(``tests/test_job_runner.py``).

Run (repository root)::

    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests/test_text_jobs.py
"""

from __future__ import annotations

import ast
import copy
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
TESTS = REPO_ROOT / "tests"
#: The commit the U3 moves were made from (the U2 commit); the verbatim checks read its source
#: with ``git show`` (pinned: later milestones re-freeze legacy/LATEST.txt at newer commits).
BASE_SHA = "1719fb59dcab56953ca0f32392d4f2159c703c2a"
for _p in (str(TESTS), str(SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

NEW_MODULES = ("stop_control", "job_runner", "text_jobs", "input_preparation")
TOUCHED = NEW_MODULES + ("translator_gui", "headless_owner", "library_core", "epub_library")
U3_HOOKS = ("_backend_entry", "_ui_request", "_notify_compile_result")
MOVED_TEXT = ("_process_text_file", "_extract_glossary_from_text_file", "_run_parallel_metadata_files")
MOVED_INPUT = ("_extract_subtitle_zip_input_if_needed", "_has_epub_conversion_inputs",
               "_convert_zip_input_to_epub_if_needed", "_resolve_zip_inputs_for_translation")


def _src_text(module: str) -> str:
    return (SRC / f"{module}.py").read_bytes().decode("utf-8-sig").replace("\r\n", "\n")


def _tree(module: str) -> ast.Module:
    return ast.parse(_src_text(module))


def _class(tree, name):
    return next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)


def _methods(cls):
    return {n.name: n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


@pytest.fixture(scope="module")
def legacy_source():
    from parity import freeze_legacy

    sha = BASE_SHA
    try:
        return sha, freeze_legacy.git_show_text(sha, "src/translator_gui.py")
    except Exception as exc:  # pragma: no cover - shallow clone
        pytest.skip(f"git show {sha[:12]}:src/translator_gui.py unavailable: {exc}")


@pytest.fixture(scope="module")
def legacy_tg(legacy_source):
    return _methods(_class(ast.parse(legacy_source[1]), "TranslatorGUI"))


# ---------------------------------------------------------------------------
# import hygiene / Python 3.10
# ---------------------------------------------------------------------------

_HYGIENE_PROBE = r"""
import sys
sys.modules['PySide6'] = None
sys.path.insert(0, {src!r})
import importlib
importlib.import_module({module!r})
leaked = [n for n in ('translator_gui', 'dpi_setup', 'epub_library', 'PySide6.QtCore', 'PySide6.QtWidgets')
          if sys.modules.get(n) is not None]
print('LEAKED=' + ','.join(leaked))
"""


@pytest.mark.parametrize("module", NEW_MODULES + ("headless_owner", "library_core"))
def test_module_imports_without_qt_or_the_gui(module):
    env = {k: v for k, v in os.environ.items() if not k.startswith("GLOSSARION_")}
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run([sys.executable, "-c", _HYGIENE_PROBE.format(src=str(SRC), module=module)],
                          capture_output=True, text=True, encoding="utf-8", env=env, timeout=300)
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert proc.stdout.strip().splitlines()[-1] == "LEAKED=", proc.stdout[-2000:]


@pytest.mark.parametrize("module", TOUCHED)
def test_touched_files_parse_as_python_310(module):
    ast.parse(_src_text(module), feature_version=(3, 10))


def test_new_modules_never_import_qt_or_the_gui():
    for module in NEW_MODULES:
        for node in ast.walk(_tree(module)):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else (
                [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
            for name in names:
                assert not name.startswith(("PySide6", "translator_gui", "dpi_setup", "epub_library")), (module, name)


# ---------------------------------------------------------------------------
# verbatim moves (AST compare after the documented edits)
# ---------------------------------------------------------------------------


def _stmt(src: str):
    return ast.parse(src).body


class _Edits(ast.NodeTransformer):
    """Apply statement-level edits inside every statement list of a function."""

    def __init__(self, replace=(), insert_before=(), drop_finally=False, drop_def=None, def_with=None):
        self.replace = [([ast.unparse(s) for s in _stmt(old)], new) for old, new in replace]
        self.insert_before = [(ast.unparse(_stmt(anchor)[0]), new) for anchor, new in insert_before]
        self.drop_def = drop_def
        self.def_with = def_with
        self.hits = 0

    def _edit_list(self, stmts):
        out = []
        i = 0
        while i < len(stmts):
            done = False
            for old, new in self.replace:
                window = [ast.unparse(s) for s in stmts[i:i + len(old)]]
                if window == old:
                    out.extend(_stmt(new) if new else [])
                    i += len(old)
                    self.hits += 1
                    done = True
                    break
            if done:
                continue
            stmt = stmts[i]
            if self.drop_def and isinstance(stmt, ast.FunctionDef) and stmt.name == self.drop_def:
                out.extend(_stmt(self.def_with))
                self.hits += 1
                i += 1
                continue
            text = ast.unparse(stmt)
            for anchor, new in self.insert_before:
                if text.split("\n")[0] == anchor.split("\n")[0] and text == anchor:
                    out.extend(_stmt(new))
                    self.hits += 1
            out.append(stmt)
            i += 1
        return out

    def generic_visit(self, node):
        super().generic_visit(node)
        for field in ("body", "orelse", "finalbody"):
            value = getattr(node, field, None)
            if isinstance(value, list) and value and isinstance(value[0], ast.stmt):
                setattr(node, field, self._edit_list(value))
        return node


def _normalized(node) -> str:
    return ast.dump(ast.parse(ast.unparse(node)))


def _assert_moved(legacy_node, new_node, edits: _Edits | None = None, expected_hits=None):
    legacy = copy.deepcopy(legacy_node)
    if edits is not None:
        legacy = edits.visit(legacy)
        if expected_hits is not None:
            assert edits.hits == expected_hits, (new_node.name, edits.hits)
    assert _normalized(legacy) == _normalized(new_node), new_node.name


_SNAPSHOT_RESTORE = dict(
    replace=[
        ("old_argv = sys.argv\nold_env = dict(os.environ)", "process_state = scoped_process_state().snapshot()"),
        ("sys.argv = old_argv\nos.environ.clear()\nos.environ.update(old_env)\n"
         "try:\n    import large_env\n    large_env.clear_store()\nexcept Exception:\n    pass",
         "process_state.restore()"),
    ],
)


def test_process_text_file_is_moved_verbatim(legacy_tg):
    new = _methods(_class(_tree("text_jobs"), "TextJobsMixin"))
    edits = _Edits(insert_before=[("if translation_main is None:\n    self.append_log('❌ Translation module is not available')\n    return False",
                                   "translation_main = self._backend_entry('translation_main')")],
                   **_SNAPSHOT_RESTORE)
    _assert_moved(legacy_tg["_process_text_file"], new["_process_text_file"], edits, expected_hits=3)
    _assert_moved(legacy_tg["_run_parallel_metadata_files"], new["_run_parallel_metadata_files"])


def test_extract_glossary_is_moved_verbatim(legacy_tg):
    new = _methods(_class(_tree("text_jobs"), "TextJobsMixin"))
    edits = _Edits(
        replace=[
            ("old_argv = sys.argv\nold_env = dict(os.environ)",
             "process_state = scoped_process_state(clear_large_env=False).snapshot()"),
            ("sys.argv = old_argv\nos.environ.clear()\nos.environ.update(old_env)", "process_state.restore()"),
            ("import traceback\nglossary_main(log_callback=self.append_log, stop_callback=enhanced_stop_callback)",
             "import traceback\nglossary_main = self._backend_entry('glossary_main')\n"
             "glossary_main(log_callback=self.append_log, stop_callback=enhanced_stop_callback)"),
        ],
        drop_def="enhanced_stop_callback",
        def_with="enhanced_stop_callback = make_glossary_stop_callback(lambda: self.stop_requested, "
                 "lambda: getattr(self, 'graceful_stop_active', False))",
    )
    _assert_moved(legacy_tg["_extract_glossary_from_text_file"], new["_extract_glossary_from_text_file"], edits,
                  expected_hits=4)


def _compile_core(legacy_runner):
    """The legacy runner's try/except without its desktop finally."""
    node = copy.deepcopy(legacy_runner)
    try_node = node.body[-1]
    assert isinstance(try_node, ast.Try) and try_node.finalbody
    try_node.finalbody = []
    return try_node


def test_compile_runners_are_moved_verbatim(legacy_tg):
    new = _methods(_class(_tree("text_jobs"), "TextJobsMixin"))
    # PDF
    pdf_try = _compile_core(legacy_tg["run_pdf_converter_direct"])
    edits = _Edits(replace=[
        ("QTimer.singleShot(0, lambda p=compiled_path: QMessageBox.information(self, 'PDF Compilation Success', f'Created: {p}'))",
         "result.path = compiled_path\nself._notify_compile_result('pdf', path=compiled_path)"),
        ("QTimer.singleShot(0, lambda message=str(exc): QMessageBox.critical(self, 'PDF Compilation Failed', f'Error: {message}'))",
         "result.error = str(exc)\nself._notify_compile_result('pdf', error=result.error)"),
    ])
    pdf_try = edits.visit(pdf_try)
    assert edits.hits == 2
    new_pdf = new["_run_pdf_compile"]
    assert [ast.unparse(s) for s in new_pdf.body[1:3]] == [
        "if folder is not None:\n    self.pdf_folder = folder", "result = CompileResult('pdf')"]
    assert _normalized(new_pdf.body[3]) == _normalized(pdf_try)
    assert ast.unparse(new_pdf.body[4]) == "return result"
    # EPUB
    epub_try = _compile_core(legacy_tg["run_epub_converter_direct"])
    edits = _Edits(
        replace=[
            ("QTimer.singleShot(0, lambda p=compiled_path: QMessageBox.information(self, 'EPUB Compilation Success', f'Created: {p}'))",
             "result.path = compiled_path\nself._notify_compile_result('epub', path=compiled_path)"),
            ("QTimer.singleShot(0, lambda: QMessageBox.information(self, 'EPUB Compilation Success', f'Created: {out_file}'))",
             "result.path = out_file\nself._notify_compile_result('epub', path=out_file)"),
            ("QTimer.singleShot(0, lambda: QMessageBox.critical(self, 'EPUB Converter Failed', f'Error: {error_str}'))",
             "self._notify_compile_result('epub', error=error_str)"),
            ("error_str = str(e)\nself.append_log(f'❌ EPUB Converter error: {error_str}')",
             "error_str = str(e)\nself.append_log(f'❌ EPUB Converter error: {error_str}')\nresult.error = error_str"),
        ],
        insert_before=[("compiled_path = fallback_compile_epub(folder, log_callback=self.append_log)",
                        "fallback_compile_epub = self._backend_entry('fallback_compile_epub')")],
    )
    epub_try = edits.visit(epub_try)
    assert edits.hits == 5
    stopped = next(s for s in epub_try.body if isinstance(s, ast.If) and ast.unparse(s.test) == "not self.stop_requested")
    assert not stopped.orelse
    stopped.orelse = _stmt("result.stopped = True")
    new_epub = new["_run_epub_compile"]
    assert [ast.unparse(s) for s in new_epub.body[1:3]] == [
        "if folder is not None:\n    self.epub_folder = folder", "result = CompileResult('epub')"]
    assert _normalized(new_epub.body[3]) == _normalized(epub_try)


def test_desktop_compile_runners_are_wrappers_with_the_original_finally(legacy_tg):
    tg = _methods(_class(_tree("translator_gui"), "TranslatorGUI"))
    for runner, core in (("run_pdf_converter_direct", "_run_pdf_compile"), ("run_epub_converter_direct", "_run_epub_compile")):
        node = tg[runner]
        try_node = node.body[-1]
        assert [ast.unparse(s) for s in try_node.body] == [f"self.{core}()"]
        assert not try_node.handlers
        legacy_finally = legacy_tg[runner].body[-1].finalbody
        assert [ast.unparse(s) for s in try_node.finalbody] == [ast.unparse(s) for s in legacy_finally]


def test_input_preparation_is_moved_verbatim(legacy_tg):
    tree = _tree("input_preparation")
    new = _methods(_class(tree, "InputPreparationMixin"))
    _assert_moved(legacy_tg["_extract_subtitle_zip_input_if_needed"], new["_extract_subtitle_zip_input_if_needed"])
    _assert_moved(legacy_tg["_has_epub_conversion_inputs"], new["_has_epub_conversion_inputs"])
    edits = _Edits(replace=[("try:\n    self.input_files_updated_signal.emit(resolved)\nexcept Exception:\n    pass",
                             "try:\n    self._ui_request('input_files_updated', resolved)\nexcept Exception:\n    pass")])
    _assert_moved(legacy_tg["_resolve_zip_inputs_for_translation"], new["_resolve_zip_inputs_for_translation"],
                  edits, expected_hits=1)
    # resolve_input_to_epub == _convert_zip_input_to_epub_if_needed's body with owner reads as parameters
    legacy = copy.deepcopy(legacy_tg["_convert_zip_input_to_epub_if_needed"])
    edits = _Edits(replace=[
        ("should_stop = lambda: bool(getattr(self, 'stop_requested', False))", ""),
        ("self._zip_conversion_active = True", "if set_active is not None:\n    set_active(True)"),
        ("self._zip_conversion_active = False", "if set_active is not None:\n    set_active(False)"),
    ])
    legacy = edits.visit(legacy)
    assert edits.hits == 3
    legacy_body = [ast.unparse(s) for s in legacy.body[1:]]  # docstring dropped
    legacy_body = [s.replace("getattr(self, '_direct_text_archive_conversion_dir', '')", "conversion_dir")
                    .replace("self.append_log(", "log(") for s in legacy_body]
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "resolve_input_to_epub")
    assert [ast.unparse(s) for s in fn.body[1:]] == legacy_body
    wrapper = new["_convert_zip_input_to_epub_if_needed"]
    assert "resolve_input_to_epub(" in ast.unparse(wrapper)


def test_is_traditional_translation_api_moved_and_reexported(legacy_source):
    legacy = next(n for n in ast.parse(legacy_source[1]).body
                  if isinstance(n, ast.FunctionDef) and n.name == "is_traditional_translation_api")
    new = next(n for n in _tree("text_jobs").body
               if isinstance(n, ast.FunctionDef) and n.name == "is_traditional_translation_api")
    assert ast.unparse(new) == ast.unparse(legacy)
    tg = _tree("translator_gui")
    assert not any(isinstance(n, ast.FunctionDef) and n.name == "is_traditional_translation_api" for n in tg.body)
    imports = [n for n in tg.body if isinstance(n, ast.ImportFrom) and n.module == "text_jobs"]
    assert imports and "is_traditional_translation_api" in [a.name for a in imports[0].names]


# ---------------------------------------------------------------------------
# MRO / hooks / no duplicate bodies
# ---------------------------------------------------------------------------


def test_translator_gui_bases_and_moved_bodies(legacy_tg):
    cls = _class(_tree("translator_gui"), "TranslatorGUI")
    bases = [ast.unparse(b) for b in cls.bases]
    assert bases[:5] == ["TextJobsMixin", "InputPreparationMixin", "SettingsPersistenceMixin", "RunEnvMixin",
                         "ConfigStateMixin"] or bases[1:6] == [
        "TextJobsMixin", "InputPreparationMixin", "SettingsPersistenceMixin", "RunEnvMixin", "ConfigStateMixin"]
    body = _methods(cls)
    for name in MOVED_TEXT + MOVED_INPUT:
        assert name in legacy_tg and name not in body, name
    for hook in U3_HOOKS:
        assert hook in body and hook not in legacy_tg, hook
    hooks = _methods(_class(_tree("job_runner"), "JobHooksMixin"))
    assert set(U3_HOOKS) <= set(hooks)
    text_jobs = _methods(_class(_tree("text_jobs"), "TextJobsMixin"))
    inputs = _methods(_class(_tree("input_preparation"), "InputPreparationMixin"))
    assert set(MOVED_TEXT) <= set(text_jobs) and set(MOVED_INPUT) <= set(inputs)
    assert not set(text_jobs) & set(inputs) and not (set(text_jobs) | set(inputs)) & set(U3_HOOKS)


@pytest.mark.parametrize("module,cls", [("QA_Scanner_GUI", "QAScannerMixin"), ("Retranslation_GUI", "RetranslationMixin"),
                                        ("GlossaryManager_GUI", "GlossaryManagerMixin")])
def test_no_gui_mixin_shadows_a_u3_name(module, cls):
    names = set(_methods(_class(_tree(module), cls)))
    u3 = set(MOVED_TEXT + MOVED_INPUT + U3_HOOKS) | {"_run_epub_compile", "_run_pdf_compile"}
    assert not names & u3, sorted(names & u3)


def test_headless_owner_composition():
    import headless_owner
    import input_preparation
    import text_jobs

    mro = headless_owner.HeadlessOwner.__mro__
    # TranslatorGUI's base order: the U3 pipeline mixin (step 2) first, then the job runners
    bases = headless_owner.HeadlessOwner.__bases__
    assert bases.index(text_jobs.TextJobsMixin) + 1 == bases.index(input_preparation.InputPreparationMixin)
    assert mro.index(text_jobs.TextJobsMixin) < mro.index(input_preparation.InputPreparationMixin)
    modules = [m for m, _c in headless_owner.OWNER_CONTRACT_MODULES]
    assert {"text_jobs", "input_preparation", "job_runner"} <= set(modules)
    contract = set(headless_owner.compute_owner_contract())
    # (_reset_api_watchdog_progress left the contract with step 2: every call sits in a
    # try/except Exception, which the scanner now counts as guarded like owner_contract.py)
    assert {"api_key_entry", "prompt_text", "token_limit_entry", "append_log"} <= contract


def test_headless_owner_satisfies_the_u3_contract(tmp_path, monkeypatch):
    from _headless_env import headless_owner as build

    import headless_owner

    deferred = {"auto_load_glossary_for_file"}  # glossary auto-mapping: the translation pipeline step
    with build(tmp_path, monkeypatch, {"model": "gpt-4o"}) as owner:
        missing = sorted(n for n in headless_owner.compute_owner_contract()
                         if not hasattr(owner, n) and n not in deferred)
    assert not missing, missing


# ---------------------------------------------------------------------------
# tier G: HeadlessOwner runs the job entries like the desktop goldens
# ---------------------------------------------------------------------------

JOB_ENTRIES = ("process_text_file", "extract_glossary", "epub_compile", "pdf_compile")


class _RecordingHost:
    def __init__(self, recorder):
        self.recorder = recorder

    def log(self, message, **_kw):
        self.recorder.record("append_log", [message], {})

    def emit(self, kind, **data):
        if kind == "thread_complete":  # the desktop's thread_complete_signal.emit()
            self.recorder.record(f"signal:{kind}_signal", [], {})


@pytest.fixture(scope="module")
def headless_parity():
    pytest.importorskip("PySide6")  # the goldens / frozen oracle need the desktop import closure
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from parity import capture_golden as cg
    from parity import fakes, freeze_legacy, scenarios

    try:
        bundle = freeze_legacy.load_legacy()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    legacy_factory = fakes.make_legacy_owner_factory(bundle)
    goldens = {}

    def golden(name):
        if name not in goldens:
            try:
                goldens[name] = cg.load_golden(bundle.sha, name)
            except (FileNotFoundError, OSError):
                goldens[name] = cg.roundtrip(cg.capture(legacy_factory, scenarios.get(name), entries=list(JOB_ENTRIES)))
        return goldens[name]

    def factory(scenario, ctx):
        from config_store import load_config
        from headless_owner import HeadlessOwner

        fakes.patch_shared_module_paths(ctx)
        try:
            config = load_config(str(ctx.sandbox.config_file))
        except Exception:
            config = {}

        class DesktopRunners(HeadlessOwner):
            """HeadlessOwner + the desktop compile runners' wrappers (the entries call those)."""

            def _backend_entry(self, name):
                return ctx.backend_stubs[name]

            def run_epub_converter_direct(self):
                try:
                    self._run_epub_compile()
                finally:
                    self.epub_thread = None
                    self.epub_future = None
                    self.stop_requested = False
                    self._ui_request("thread_complete")

            def run_pdf_converter_direct(self):
                try:
                    self._run_pdf_compile()
                finally:
                    self.pdf_thread = None
                    self.pdf_future = None
                    self.stop_requested = False
                    self._ui_request("thread_complete")

        return DesktopRunners(config, host=_RecordingHost(ctx.recorder))

    factory.kind = "headless-u3"
    return {"cg": cg, "golden": golden, "factory": factory, "scenarios": scenarios}


def _scenario_names():
    from parity import scenarios

    return list(scenarios.SCENARIO_NAMES)


@pytest.mark.parametrize("name", _scenario_names())
def test_headless_owner_reproduces_the_desktop_job_entries(headless_parity, name):
    cg = headless_parity["cg"]
    scenario = headless_parity["scenarios"].get(name)
    entries = [e for e in scenario["entries"] if e in JOB_ENTRIES]
    assert entries
    result = cg.capture(headless_parity["factory"], scenario, entries=entries)
    golden = headless_parity["golden"](name)
    problems = []
    for entry in entries:
        got, exp = cg.roundtrip(result["entries"][entry]), golden["entries"][entry]
        assert "boot_error" not in got, got.get("boot_error")
        # owner attributes differ by design where the desktop's widgets/threads differ from shims
        got = {k: v for k, v in got.items() if k != "attrs"}
        exp = {k: v for k, v in exp.items() if k != "attrs"}
        problems += [f"[{entry}] {p}" for p in cg.diff(exp, got)]
    assert not problems, "\n".join(problems[:30])


# ---------------------------------------------------------------------------
# resolve_input_to_epub (used directly by the mobile FileBridge)
# ---------------------------------------------------------------------------


def test_resolve_input_to_epub_direct(tmp_path):
    from input_preparation import resolve_input_to_epub

    html = tmp_path / "Story.html"
    html.write_text("<html><head><title>T</title></head><body><p>본문</p></body></html>", encoding="utf-8")
    logs, active = [], []
    out_dir = tmp_path / "converted"
    epub = resolve_input_to_epub(str(html), conversion_dir=str(out_dir), should_stop=lambda: False,
                                 log=logs.append, set_active=active.append)
    assert epub == str(out_dir / "Story.html.epub") and zipfile.is_zipfile(epub)
    assert active == [True, False]
    assert logs[0].startswith("📄 Preparing HTML document")
    # not an archive: unchanged, no flag flips
    active.clear()
    assert resolve_input_to_epub(str(tmp_path / "book.epub"), should_stop=lambda: False, log=logs.append,
                                 set_active=active.append) == str(tmp_path / "book.epub")
    assert active == []
    # cancelled before conversion: input returned, cancellation logged
    logs.clear()
    assert resolve_input_to_epub(str(html), conversion_dir=str(tmp_path / "c2"), should_stop=lambda: True,
                                 log=logs.append) == str(html)
    assert any("conversion cancelled" in line for line in logs)
