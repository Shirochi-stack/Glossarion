"""translation_pipeline and the U3 step-2 TranslatorGUI rewiring (Glossarion mobile rewrite, milestone U3).

TranslatorGUI's translation and glossary pipelines moved into ``translation_pipeline``:
``run_translation_thread``'s worker closure became ``TranslationPipelineMixin._translation_worker``
and its run set-up ``_prepare_translation_run() -> RunRequest``; ``run_translation_direct``,
the QA-failure / multipass refinement planning, ``run_glossary_extraction_direct``, the
image-folder glossary, glossary auto-loading / auto-mapping and the input-selection helpers
moved verbatim (``GlossaryPipelineMixin``); ``PipelineHooksMixin`` holds the GUI-free
defaults of the hooks and of the desktop GUI methods the moved code calls. TranslatorGUI
inherits ``TranslationPipelineMixin`` first; ``HeadlessOwner`` runs the same code on mobile.

What is checked here:
* import hygiene (PySide6 blocked; no translator_gui / dpi_setup / epub_library at import)
  and Python 3.10 syntax;
* the moves are verbatim: every moved method's text equals the frozen legacy text
  (``git show <oracle sha>``) after exactly the documented edits; ``_prepare_translation_run``
  / ``_translation_worker`` are the two slices of the legacy ``run_translation_thread``, and the
  desktop ``run_translation_thread`` is the legacy preflight + the shared calls + the legacy
  thread launch;
* MRO and hooks: the pipeline mixin is TranslatorGUI's first base, the moved bodies left
  TranslatorGUI, the desktop keeps its own GUI methods (they win over every hook default),
  no GUI mixin shadows a pipeline name (GlossaryManagerMixin only re-exports the moved
  ``_glossary_editor_input_sources``); the U7 runners (image_job / rpgmaker_job) replaced the
  placeholders and come from TranslationPipelineMixin's bases;
* the GUI-free hooks: the blocking glossary-approval question goes to ``host.ask`` and
  completes the desktop's Event handshake, other requests reach ``host.emit``;
* a real ``HeadlessOwner`` runs the mobile composition end to end with stubbed backends:
  Balanced pre-glossary then translation, the Direct Text approval (Yes / No), stop flags reset.

The pipeline-order and stop-race parity against the frozen desktop (legacy closure vs new
worker, desktop and mobile) is tier T: ``tests/parity/test_trace_parity.py``.

Run (repository root)::

    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests/test_translation_pipeline.py
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import threading
import types
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

MODULE = "translation_pipeline"

#: TranslatorGUI methods moved into TranslationPipelineMixin (run_translation_thread is split)
TRANSLATION_MOVED = (
    "_clear_automatic_glossary_for_non_epub_selection",
    "_flatten_translation_qa_issue_text",
    "_is_foreign_character_translation_qa_issue",
    "_entry_has_foreign_character_qa_failure",
    "_collect_translation_qa_failures",
    "_chapter_scope_filename_key",
    "_filter_translation_qa_failures_to_current_range",
    "_translation_qa_failure_key",
    "_qa_failure_matches_resolution_request",
    "_prepare_multipass_qa_refinement_run",
    "_clear_translation_run_overrides",
    "_format_chapter_list",
    "_log_translation_qa_failure_summary",
    "run_translation_direct",
    "_await_direct_text_glossary_approval",
)
#: TranslatorGUI methods moved into GlossaryPipelineMixin
GLOSSARY_MOVED = (
    "_is_special_file",
    "_should_skip_special_file",
    "_get_spine_filenames_for_preview",
    "_get_pdf_range_entries_for_preview",  # U9: the Plan card "Choose chapters" PDF preview
    "_get_opf_file_order",
    "run_glossary_extraction_direct",
    "_process_image_folder_for_glossary",
    "_init_image_glossary_progress_manager",
    "_save_intermediate_glossary_with_skip",
    "_call_api_with_interrupt",
    "auto_load_glossary_for_file",
    "_windows_supported_input_path",
    "_windows_glossary_rename_dirs",
    "_path_lookup_key",
    "_remap_windows_renamed_epub_glossary",
    "_normalize_windows_input_filenames",
    "_auto_load_glossary_after_extraction",
    "_autofill_glossary_for_current_selection",
    "_glossary_dir_signature",
    "_get_glossary_dir_candidates",
    "_guess_glossary_for_input_file",
    "_copy_glossary_to_output_folders",
    "_sync_automapped_glossaries_to_output",
)
MOVED = TRANSLATION_MOVED + GLOSSARY_MOVED
NEW_METHODS = ("_prepare_translation_run", "_translation_worker")

#: The documented edits per moved method: (old text, new text, occurrences).
EDITS = {
    "_filter_translation_qa_failures_to_current_range": [
        ("TranslatorGUI._live_chapter_range_settings(self)", "RunEnvMixin._live_chapter_range_settings(self)", 1),
        ("TranslatorGUI._chapter_scope_filename_key(", "TranslationPipelineMixin._chapter_scope_filename_key(", 2),
    ],
    "_prepare_multipass_qa_refinement_run": [
        ("TranslatorGUI._filter_translation_qa_failures_to_current_range(",
         "TranslationPipelineMixin._filter_translation_qa_failures_to_current_range(", 2),
    ],
    "run_translation_direct": [
        ("image_extensions = set(_InputOutputDialog._IMAGE_ATTACHMENT_EXTENSIONS)",
         "image_extensions = set(IMAGE_ATTACHMENT_EXTENSIONS)", 1),
        ("self.input_files_updated_signal.emit(list(self.selected_files))",
         "self._ui_request('input_files_updated', list(self.selected_files))", 1),
        # U7: a game folder the mobile RPG Maker entry registered is dispatched like a game .exe
        ("elif ext == '.exe':",
         "elif ext == '.exe' or file_path in (getattr(self, RPGMAKER_GAME_INPUTS_ATTR, None) or ()):", 1),
    ],
    "_await_direct_text_glossary_approval": [
        ("self.direct_text_glossary_approval_signal.emit(", "self._ui_request(\n            'direct_text_glossary_approval',", 1),
    ],
    "run_glossary_extraction_direct": [
        ("            if glossary_main is None:\n",
         "            glossary_main = self._backend_entry('glossary_main')\n            if glossary_main is None:\n", 1),
        ("            if glossary_stop_flag:\n                glossary_stop_flag(False)\n",
         "            glossary_stop_flag = self._backend_entry('glossary_stop_flag')\n"
         "            if glossary_stop_flag:\n                glossary_stop_flag(False)\n", 1),
        ("            self.thread_complete_signal.emit()", "            self._ui_request('thread_complete')", 1),
    ],
}
PREPARE_EDITS = [
    ('                    QMessageBox.critical(self, "Error", "Please select file(s) to translate.")\n',
     '                    self._ui_message(\'critical\', "Error", "Please select file(s) to translate.")\n', 1),
    ("        if translation_stop_flag:\n            translation_stop_flag(False)\n",
     "        translation_stop_flag = self._backend_entry('translation_stop_flag')\n"
     "        if translation_stop_flag:\n            translation_stop_flag(False)\n", 1),
]
WORKER_EDITS = [
    ("                    self.trigger_qa_scan_signal.emit()\n", "                    self._ui_request('trigger_qa_scan')\n", 1),
    ("                if translation_stop_flag:\n                    translation_stop_flag(False)\n",
     "                translation_stop_flag = self._backend_entry('translation_stop_flag')\n"
     "                if translation_stop_flag:\n                    translation_stop_flag(False)\n", 1),
    ("            self.thread_complete_signal.emit()", "            self._ui_request('thread_complete')", 1),
    # U3 fix pass: the worker returns its outcome (False on every early return and after the
    # caught exception, run_translation_direct's result at the end); the desktop thread ignores it
    ('                    self.append_log("❌ Failed to load modules")\n                    return\n',
     '                    self.append_log("❌ Failed to load modules")\n                    return False\n', 1),
    ("                self._resolve_zip_inputs_for_translation()\n                if self.stop_requested:\n"
     "                    return\n",
     "                self._resolve_zip_inputs_for_translation()\n                if self.stop_requested:\n"
     "                    return False\n", 1),
    ("                                os.environ.pop('MANUAL_GLOSSARY', None)\n                            return\n",
     "                                os.environ.pop('MANUAL_GLOSSARY', None)\n                            return False\n", 1),
    ('                                    "the glossary approval step"\n                                )\n'
     "                                return\n",
     '                                    "the glossary approval step"\n                                )\n'
     "                                return False\n", 1),
    ('                            f"glossary is below 100% ({reason})."\n                        )\n'
     "                        return\n",
     '                            f"glossary is below 100% ({reason})."\n                        )\n'
     "                        return False\n", 1),
    ("                self.append_log(traceback.format_exc())\n            \n        except Exception as e:\n"
     '            self.append_log(f"❌ Error in thread: {e}")\n            import traceback\n'
     "            self.append_log(traceback.format_exc())\n        finally:\n",
     "                self.append_log(traceback.format_exc())\n            \n            return translation_completed\n"
     "        except Exception as e:\n"
     '            self.append_log(f"❌ Error in thread: {e}")\n            import traceback\n'
     "            self.append_log(traceback.format_exc())\n            return False\n        finally:\n", 1),
]
#: U3 fix pass: the set-up's Library raw-input registry write is the _record_library_raw_inputs
#: hook (TranslatorGUI's override keeps the epub_library import; the GUI-free default skips it)
REGISTRY_BLOCK = (
    "            try:\n"
    "                from epub_library import record_library_raw_input\n"
    "                for _p in (self.selected_files or []):\n"
    "                    if _p and os.path.isfile(_p):\n"
    "                        record_library_raw_input(_p)\n"
    "            except Exception:\n"
    "                pass\n"
)
PREPARE_EDITS.append((REGISTRY_BLOCK, "            self._record_library_raw_inputs(self.selected_files)\n", 1))


# ---------------------------------------------------------------------------
# source helpers
# ---------------------------------------------------------------------------


def _src_text(module: str) -> str:
    return (SRC / f"{module}.py").read_bytes().decode("utf-8-sig").replace("\r\n", "\n")


def _tree(module: str) -> ast.Module:
    return ast.parse(_src_text(module))


def _class(tree, name):
    return next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)


def _methods(cls):
    return {n.name: n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


def _node_text(text: str, node) -> str:
    lines = text.split("\n")
    start = min([node.lineno] + [d.lineno for d in node.decorator_list])
    return "\n".join(lines[start - 1:node.end_lineno])


def _apply(text: str, edits, where: str) -> str:
    for old, new, count in edits:
        assert text.count(old) == count, f"{where}: expected {count} x {old!r}, found {text.count(old)}"
        text = text.replace(old, new)
    return text


def _pipeline_methods():
    text = _src_text(MODULE)
    tree = ast.parse(text)
    out = {}
    for cls_name in ("PipelineHooksMixin", "GlossaryPipelineMixin", "TranslationPipelineMixin"):
        for name, node in _methods(_class(tree, cls_name)).items():
            out[name] = (cls_name, node, _node_text(text, node))
    return out


@pytest.fixture(scope="module")
def legacy_source():
    from parity import freeze_legacy

    sha = BASE_SHA
    try:
        tg = freeze_legacy.git_show_text(sha, "src/translator_gui.py").replace("\r\n", "\n")
        gm = freeze_legacy.git_show_text(sha, "src/GlossaryManager_GUI.py").replace("\r\n", "\n")
    except Exception as exc:  # pragma: no cover - shallow clone
        if os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"):
            pytest.fail(f"git show {sha[:12]} unavailable: {exc}")
        pytest.skip(f"git show {sha[:12]} unavailable: {exc}")
    return sha, tg, gm


@pytest.fixture(scope="module")
def legacy_tg(legacy_source):
    _sha, text, _gm = legacy_source
    tree = ast.parse(text)
    return text, _methods(_class(tree, "TranslatorGUI"))


# ---------------------------------------------------------------------------
# import hygiene / Python 3.10
# ---------------------------------------------------------------------------

_HYGIENE_PROBE = r"""
import sys
sys.modules['PySide6'] = None
sys.path.insert(0, {src!r})
import importlib
importlib.import_module({module!r})
leaked = [n for n in ('translator_gui', 'dpi_setup', 'epub_library', 'GlossaryManager_GUI', 'PySide6.QtCore',
                      'PySide6.QtWidgets') if sys.modules.get(n) is not None]
print('LEAKED=' + ','.join(leaked))
"""


@pytest.mark.parametrize("module", (MODULE, "headless_owner"))
def test_module_imports_without_qt_or_the_gui(module):
    env = {k: v for k, v in os.environ.items() if not k.startswith("GLOSSARION_")}
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run([sys.executable, "-c", _HYGIENE_PROBE.format(src=str(SRC), module=module)],
                          capture_output=True, text=True, encoding="utf-8", env=env, timeout=300)
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert proc.stdout.strip().splitlines()[-1] == "LEAKED=", proc.stdout[-2000:]


@pytest.mark.parametrize("module", (MODULE, "headless_owner", "translator_gui", "GlossaryManager_GUI"))
def test_touched_files_parse_as_python_310(module):
    ast.parse(_src_text(module), feature_version=(3, 10))


def test_pipeline_module_never_imports_qt_or_the_gui():
    tree = _tree(MODULE)
    for node in ast.walk(tree):
        names = [a.name for a in node.names] if isinstance(node, ast.Import) else (
            [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
        for name in names:
            assert not name.startswith(("PySide6", "translator_gui", "dpi_setup", "GlossaryManager_GUI")), name
    # epub_library (Qt) is only reached lazily inside the moved run set-up, like on desktop
    top = {(a.name if isinstance(n, ast.Import) else n.module) for n in tree.body
           if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}
    assert "epub_library" not in top


def test_line_endings_are_crlf_without_bom():
    # Line endings follow the checkout (CRLF on Windows with core.autocrlf, LF on
    # Linux CI). What must hold everywhere: the module matches translator_gui.py's
    # convention, has no BOM, and translator_gui keeps its BOM.
    raw = (SRC / f"{MODULE}.py").read_bytes()
    gui = (SRC / "translator_gui.py").read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    assert gui.startswith(b"\xef\xbb\xbf")
    if gui.count(b"\r\n") == gui.count(b"\n"):
        assert raw.count(b"\r\n") == raw.count(b"\n")
    else:
        assert b"\r\n" not in raw


# ---------------------------------------------------------------------------
# verbatim moves
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", MOVED)
def test_moved_method_is_the_legacy_text_plus_documented_edits(legacy_tg, name):
    text, methods = legacy_tg
    legacy = _apply(_node_text(text, methods[name]), EDITS.get(name, ()), name)
    cls_name, _node, new = _pipeline_methods()[name]
    expected = "TranslationPipelineMixin" if name in TRANSLATION_MOVED else "GlossaryPipelineMixin"
    assert cls_name == expected
    assert new == legacy


def test_glossary_editor_input_sources_moved_from_glossary_manager(legacy_source):
    _sha, _tg, gm = legacy_source
    legacy = _node_text(gm, _methods(_class(ast.parse(gm), "GlossaryManagerMixin"))["_glossary_editor_input_sources"])
    cls_name, _node, new = _pipeline_methods()["_glossary_editor_input_sources"]
    assert cls_name == "GlossaryPipelineMixin" and new == legacy
    manager = _class(_tree("GlossaryManager_GUI"), "GlossaryManagerMixin")
    assert "_glossary_editor_input_sources" not in _methods(manager)
    aliases = [ast.unparse(s) for s in manager.body if isinstance(s, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "_glossary_editor_input_sources" for t in s.targets)]
    assert aliases == ["_glossary_editor_input_sources = GlossaryPipelineMixin._glossary_editor_input_sources"]


#: U3 step 1 (stop_control) already replaced these legacy run_translation_thread blocks by
#: calls (tests/test_job_runner.py proves the calls run the same statements):
#: (first line, last line, replacement lines)
STEP1_BLOCKS = (
    ("        os.environ.pop('TRANSLATION_CANCELLED', None)  # Clear hard-stop state from previous run",
     "        os.environ['GRACEFUL_STOP_API_ACTIVE'] = '0'  # Reset API active flag",
     ["        reset_stop_env('translation')"]),
    ("        try:\n            import uuid as _uuid",
     "            os.environ['GLOSSARION_RUN_ID'] = str(int(time.time()))",
     ["        os.environ['GLOSSARION_RUN_ID'] = make_run_id('translation')"]),
    ("        # CRITICAL: Before starting a new run, hard-cancel any lingering streams",
     "                unified_api_client.UnifiedClient.set_global_cancellation(False)\n        except Exception:\n"
     "            pass",
     ["        # Close the previous run's lingering streams, then reset the client's",
      "        # global cancellation (streaming stop) for the new run",
      "        clear_client_cancellation()"]),
    ("        try:\n            import tempfile, os as _os",
     "            if _os.path.exists(stop_file):\n                _os.remove(stop_file)\n        except Exception:\n"
     "            pass",
     ["        prepare_glossary_stop_file()"]),
    # U3 fix pass: the preflight's wait for the previous stop's cleanup thread is
    # stop_control.wait_for_stop_cleanup (the mobile JobService waits the same way)
    ("        # Immediate Stop performs connection teardown on a helper thread. Do",
     '                self.append_log("⏹️ Previous translation is still stopping; try Start again shortly.")\n'
     "                return",
     ["        # Immediate Stop performs connection teardown on a helper thread. Do",
      "        # not clear cancellation state for a new run while that older cleanup",
      "        # can still close transports underneath it (stop_control, shared with mobile).",
      "        if not wait_for_stop_cleanup(getattr(self, '_translation_stop_cleanup_thread', None), self.append_log):",
      "            return"]),
)


def _apply_step1(lines):
    text = "\n".join(lines)
    for first, last, new in STEP1_BLOCKS:
        start = text.index(first)
        end = text.index(last, start) + len(last)
        assert text.count(first) == 1, first
        text = text[:start] + "\n".join(new) + text[end:]
    return text.split("\n")


def _run_translation_thread_parts(text, node):
    lines = _apply_step1(_node_text(text, node).split("\n"))

    def find(needle, start=0):
        return next(i for i in range(start, len(lines)) if lines[i] == needle)

    p_start = find("        # Check if files are selected")
    p_end = find("        self._stop_notice_shown = False", p_start)
    c_def = find("        def simple_thread_target():", p_end)
    tail = find('        thread_name = f"TranslationThread_{int(time.time())}"', c_def)
    return lines, p_start, p_end, c_def, tail


def test_prepare_and_worker_are_the_run_translation_thread_slices(legacy_tg):
    text, methods = legacy_tg
    lines, p_start, p_end, c_def, tail = _run_translation_thread_parts(text, methods["run_translation_thread"])
    pipeline = _pipeline_methods()

    # _prepare_translation_run: docstring, the mobile *files* override, the verbatim slice, RunRequest
    _cls, node, new = pipeline["_prepare_translation_run"]
    assert [a.arg for a in node.args.args] == ["self", "files"] and ast.unparse(node.args.defaults[0]) == "None"
    body = new.split("\n")
    head = body.index("        if files is not None:")
    assert body[head + 1] == "            self.selected_files = list(files)"
    slice_end = body.index("        return RunRequest(")
    assert body[slice_end - 1] == ""
    expected = _apply("\n".join(lines[p_start:p_end + 1]), PREPARE_EDITS, "prepare")
    assert "\n".join(body[head + 2:slice_end - 1]) == expected
    assert isinstance(node.body[-1], ast.Return) and ast.unparse(node.body[-1].value).startswith("RunRequest(")

    # _translation_worker: docstring + the closure body, one indentation level less
    _cls, node, new = pipeline["_translation_worker"]
    assert [a.arg for a in node.args.args] == ["self", "request"]
    closure = lines[c_def + 1:tail - 1]
    dedented = "\n".join(ln[4:] if ln.startswith("    ") else ln for ln in closure)
    expected = _apply(dedented, WORKER_EDITS, "worker")
    body = new.split("\n")
    first = body.index("        try:")
    assert "\n".join(body[first:]) == expected
    assert len(node.body) == 2 and isinstance(node.body[1], ast.Try)  # docstring + the closure's try


def test_desktop_run_translation_thread_is_preflight_plus_the_shared_pipeline(legacy_tg):
    text, methods = legacy_tg
    lines, p_start, _p_end, _c_def, tail = _run_translation_thread_parts(text, methods["run_translation_thread"])
    tg_text = _src_text("translator_gui")
    new = _node_text(tg_text, _methods(_class(ast.parse(tg_text), "TranslatorGUI"))["run_translation_thread"])
    launch = _apply("\n".join(lines[tail:]), [
        ("            target=simple_thread_target,\n",
         "            target=self._translation_worker,\n            args=(request,),\n", 1)], "launch")
    expected = "\n".join(lines[:p_start]) + "\n" + (
        "        # The run set-up and the worker thread body are shared with the mobile JobService\n"
        "        # (translation_pipeline.TranslationPipelineMixin).\n"
        "        request = self._prepare_translation_run()\n"
        "        if request is None:\n"
        "            return\n"
        "        \n"
    ) + launch
    assert new == expected


def test_desktop_library_registry_hook_is_the_legacy_block():
    """TranslatorGUI._record_library_raw_inputs runs the set-up's moved registry block verbatim."""
    tg_text = _src_text("translator_gui")
    hook = _node_text(tg_text, _methods(_class(ast.parse(tg_text), "TranslatorGUI"))["_record_library_raw_inputs"])
    block = REGISTRY_BLOCK.replace("for _p in (self.selected_files or []):", "for _p in (files or []):")
    assert "\n".join(line[4:] for line in block.rstrip("\n").split("\n")) in hook
    assert "def _record_library_raw_inputs(self, files):" in hook


# ---------------------------------------------------------------------------
# MRO / hooks / no duplicate bodies
# ---------------------------------------------------------------------------


def test_translator_gui_composition(legacy_tg):
    _text, legacy = legacy_tg
    tg_tree = _tree("translator_gui")
    cls = _class(tg_tree, "TranslatorGUI")
    assert [ast.unparse(b) for b in cls.bases][:3] == ["TranslationPipelineMixin", "TextJobsMixin",
                                                      "InputPreparationMixin"]
    body = _methods(cls)
    for name in MOVED:
        assert name in legacy and name not in body, name
    for name in NEW_METHODS:
        assert name not in body and name not in legacy, name
    import translation_pipeline as tp

    # the desktop keeps its own GUI methods / hook overrides (TranslatorGUI's body wins)
    assert set(tp.PIPELINE_HOOKS) <= set(body), sorted(set(tp.PIPELINE_HOOKS) - set(body))
    assert "_ui_message" not in legacy and "_ui_request" not in legacy
    # one image-extension set for the Direct Text dialog and run_translation_direct
    dialog = _class(tg_tree, "_InputOutputDialog")
    assign = next(s for s in dialog.body if isinstance(s, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == "_IMAGE_ATTACHMENT_EXTENSIONS" for t in s.targets))
    assert ast.unparse(assign.value) == "IMAGE_ATTACHMENT_EXTENSIONS"
    legacy_dialog = _class(ast.parse(_text), "_InputOutputDialog")
    legacy_assign = next(s for s in legacy_dialog.body if isinstance(s, ast.Assign)
                         and any(isinstance(t, ast.Name) and t.id == "_IMAGE_ATTACHMENT_EXTENSIONS" for t in s.targets))
    assert ast.literal_eval(legacy_assign.value) == tp.IMAGE_ATTACHMENT_EXTENSIONS


def test_pipeline_classes_and_hook_registry():
    import translation_pipeline as tp

    assert tp.TranslationPipelineMixin.__mro__[1:4] == (tp.GlossaryPipelineMixin, tp.PipelineHooksMixin,
                                                        tp.JobHooksMixin)
    names = _pipeline_methods()
    hooks = {n for n, (c, _node, _t) in names.items() if c == "PipelineHooksMixin"}
    assert hooks == set(tp.PIPELINE_HOOKS)
    assert not hasattr(tp, "U7_PLACEHOLDERS")
    moved_or_new = {n for n, (c, _node, _t) in names.items() if c != "PipelineHooksMixin"}
    assert moved_or_new == set(MOVED) | set(NEW_METHODS) | {"_glossary_editor_input_sources"}


@pytest.mark.parametrize("module,cls", [("QA_Scanner_GUI", "QAScannerMixin"), ("Retranslation_GUI", "RetranslationMixin"),
                                        ("GlossaryManager_GUI", "GlossaryManagerMixin")])
def test_no_gui_mixin_shadows_a_pipeline_name(module, cls):
    import translation_pipeline as tp

    node = _class(_tree(module), cls)
    names = set(_methods(node))
    shared = set(_pipeline_methods()) | set(tp.PIPELINE_HOOKS)
    assert not names & shared, sorted(names & shared)


def test_glossary_manager_alias_is_the_shared_function():
    pytest.importorskip("PySide6")
    import GlossaryManager_GUI
    import translation_pipeline as tp

    assert (GlossaryManager_GUI.GlossaryManagerMixin._glossary_editor_input_sources
            is tp.GlossaryPipelineMixin._glossary_editor_input_sources)


def test_headless_owner_composition():
    import headless_owner
    import input_preparation
    import text_jobs
    import translation_pipeline as tp

    mro = headless_owner.HeadlessOwner.__mro__
    assert mro[1:6] == (tp.TranslationPipelineMixin, tp.GlossaryPipelineMixin, tp.PipelineHooksMixin,
                        text_jobs.TextJobsMixin, input_preparation.InputPreparationMixin)
    for name in MOVED + NEW_METHODS + ("auto_load_glossary_for_file", "_glossary_editor_input_sources"):
        assert hasattr(headless_owner.HeadlessOwner, name), name
    modules = set(headless_owner.OWNER_CONTRACT_MODULES)
    assert {(MODULE, "TranslationPipelineMixin"), (MODULE, "GlossaryPipelineMixin"),
            (MODULE, "PipelineHooksMixin")} <= modules
    contract = set(headless_owner.compute_owner_contract())
    assert "entry_epub" in contract and "auto_load_glossary_for_file" not in contract
    assert "log_text" not in contract  # read only inside try/except Exception (guarded)


#: The runners run_translation_direct dispatches to (U3 placeholders until U7 moved the real ones).
U7_RUNNERS = {"_process_image_file": "ImageJobMixin", "_run_generative_prompt_mode": "ImageJobMixin",
              "_process_rpgmaker_game": "RpgMakerJobMixin"}


def test_u7_runners_replaced_the_placeholders():
    """The image / RPG Maker / generative runners are defined once, by their U7 mixins, which
    TranslationPipelineMixin inherits: no placeholder (or other shared definition) is left."""
    import headless_owner
    import translation_pipeline as tp

    assert tp.TranslationPipelineMixin.__bases__[0] is tp.GlossaryPipelineMixin
    assert [c.__name__ for c in tp.TranslationPipelineMixin.__bases__[1:]] == ["ImageJobMixin", "RpgMakerJobMixin"]
    for name, mixin in U7_RUNNERS.items():
        owners = [k.__name__ for k in headless_owner.HeadlessOwner.__mro__ if name in vars(k)]
        assert owners == [mixin], (name, owners)
        assert name not in tp.PIPELINE_HOOKS and name not in vars(tp.PipelineHooksMixin)


def test_owner_contract_scanner_counts_try_guards_and_skips_nested_classes():
    import headless_owner

    tree = ast.parse(
        "class M:\n"
        "    def a(self):\n"
        "        try:\n            self.in_try.x()\n        except Exception:\n            self.in_handler()\n"
        "        try:\n            self.in_value_try()\n        except ValueError:\n            pass\n"
        "        class Nested:\n            def b(self):\n                self.nested_only()\n"
        "        return self.plain\n")
    reads, _stores = headless_owner._unguarded_reads(tree.body[0])
    assert reads == {"in_handler", "in_value_try", "plain"}


# ---------------------------------------------------------------------------
# GUI-free hook defaults
# ---------------------------------------------------------------------------


class _Host:
    def __init__(self, answers=()):
        self.answers = list(answers)
        self.logs, self.events, self.questions = [], [], []

    def log(self, text, **_kw):
        self.logs.append(str(text))

    def emit(self, kind, **data):
        self.events.append((kind, data))

    def ask(self, kind, **data):
        self.questions.append((kind, data))
        answer = self.answers.pop(0) if self.answers else None
        if isinstance(answer, BaseException):
            raise answer
        return answer

    def is_stop_requested(self):
        return False

    def is_graceful_stop(self):
        return False


def _owner(host=None, base="PipelineHooksMixin"):
    import translation_pipeline as tp

    cls = type("_Owner", (getattr(tp, base),), {
        "append_log": lambda self, message: self.host.log(message) if self.host else None,
    })
    owner = cls()
    owner.host = host
    owner.stop_requested = False
    return owner


@pytest.mark.parametrize("answer,expected", [(True, True), (False, False), (None, False), ("yes", True)])
def test_glossary_question_completes_the_desktop_handshake(answer, expected):
    host = _Host([answer])
    owner = _owner(host)
    request = {"event": threading.Event(), "accepted": False}
    result = owner._ui_request("direct_text_glossary_approval", "C:/x/Book_glossary.json", request)
    assert result is expected and request["accepted"] is expected and request["event"].is_set()
    assert host.questions == [("direct_text_glossary_approval", {"path": "C:/x/Book_glossary.json"})]
    assert host.events == []


def test_glossary_question_without_a_host_or_with_a_failing_host_is_declined():
    request = {"event": threading.Event(), "accepted": True}
    assert _owner(None)._ui_request("direct_text_glossary_approval", "", request) is False
    assert request == {"event": request["event"], "accepted": False} and request["event"].is_set()
    host = _Host([RuntimeError("ui gone")])
    request = {"event": threading.Event(), "accepted": True}
    assert _owner(host)._ui_request("direct_text_glossary_approval", "p", request) is False
    assert request["accepted"] is False and request["event"].is_set()
    assert any("Could not ask" in line and "ui gone" in line for line in host.logs)


def test_other_ui_requests_reach_host_emit():
    host = _Host()
    owner = _owner(host)
    owner._ui_request("thread_complete")
    owner._ui_request("trigger_qa_scan")
    owner._ui_request("input_files_updated", ["a.epub"])
    assert host.events == [("thread_complete", {}), ("trigger_qa_scan", {}),
                           ("input_files_updated", {"files": ["a.epub"]})]
    # the QA scanner is not shared yet: the log says so instead of promising a scan
    import translation_pipeline as tp

    assert host.logs == [tp.UNSHARED_UI_REQUESTS["trigger_qa_scan"]]
    owner._ui_message("critical", "Error", "Please select file(s) to translate.")
    assert host.events[-1] == ("message", {"level": "critical", "title": "Error",
                                           "text": "Please select file(s) to translate."})
    assert host.logs[-1] == "❌ Error: Please select file(s) to translate."


def test_await_direct_text_glossary_approval_asks_the_host(tmp_path):
    glossary = tmp_path / "Book_glossary.json"
    host = _Host([True, False])
    owner = _owner(host, "TranslationPipelineMixin")
    assert owner._await_direct_text_glossary_approval(str(glossary)) is True
    assert owner._await_direct_text_glossary_approval("") is False
    assert host.questions == [("direct_text_glossary_approval", {"path": os.path.abspath(str(glossary))}),
                              ("direct_text_glossary_approval", {"path": ""})]


def test_gui_method_defaults():
    host = _Host()
    owner = _owner(host)
    assert owner._attach_gui_logging_handlers() is None
    assert owner._create_watchdog_snapshot(context="translation", model="m") is None
    assert owner._start_autoscroll_delay(0) is None
    assert owner._update_manual_glossary_status() is None
    assert owner._record_library_raw_inputs(["C:/x/book.epub"]) is None  # no Qt registry here
    # U7: the hook mixin no longer carries runner placeholders (tests/test_image_job.py and
    # tests/test_rpgmaker_job.py cover the real runners)
    for name in U7_RUNNERS:
        assert not hasattr(owner, name), name
    assert host.logs == []


def test_lazy_load_modules_default_imports_the_backend_entries():
    host = _Host()
    owner = _owner(host)
    owner._modules_loaded = False
    seen = []
    owner._backend_entry = lambda name: seen.append(name) or (lambda **_kw: None)
    assert owner._lazy_load_modules() is True
    assert owner._modules_loaded is True and owner._modules_loading is False
    assert seen == ["translation_main", "glossary_main", "fallback_compile_epub"]
    assert owner._lazy_load_modules() is True and len(seen) == 3  # once
    owner._modules_loaded = False
    owner._backend_entry = lambda name: None
    assert owner._lazy_load_modules() is False
    assert host.logs[-1].startswith("❌ Critical module loading failed")


# ---------------------------------------------------------------------------
# HeadlessOwner end to end (mobile composition, stubbed backends)
# ---------------------------------------------------------------------------


class _Backends:
    """Recording stand-ins for TransateKRtoEN.main / extract_glossary_from_epub.main."""

    def __init__(self):
        self.calls = []

    def translation_main(self, log_callback=None, stop_callback=None):
        self.calls.append(("translation", {k: os.environ.get(k) for k in (
            "MODEL", "MANUAL_GLOSSARY", "OUTPUT_DIRECTORY", "GRACEFUL_STOP", "TRANSLATION_CANCELLED")},
            list(sys.argv)))
        if callable(log_callback):
            log_callback("[stub] translated")
        return None

    def glossary_main(self, log_callback=None, stop_callback=None):
        import json

        output_path = os.environ.get("OUTPUT_PATH") or ""
        self.calls.append(("glossary", {"OUTPUT_PATH": output_path,
                                        "GLOSSARY_REQUEST_MERGING_ENABLED": os.environ.get(
                                            "GLOSSARY_REQUEST_MERGING_ENABLED")}, list(sys.argv)))
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as fh:
                json.dump([{"type": "character", "raw_name": "김상현", "translated_name": "Kim Sang-hyun"}],
                          fh, ensure_ascii=False)
        return None


def _run_mobile_job(tmp_path, monkeypatch, config, *, direct_text=False, answers=()):
    import TransateKRtoEN
    import extract_glossary_from_epub
    import job_runner
    import stop_control
    from _headless_env import headless_owner
    from parity.trace_scenarios import build_epub

    backends = _Backends()
    monkeypatch.setattr(TransateKRtoEN, "main", backends.translation_main)
    monkeypatch.setattr(extract_glossary_from_epub, "main", backends.glossary_main)
    monkeypatch.setitem(sys.modules, "epub_library", None)  # Qt module: unavailable on mobile
    inputs = tmp_path / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    epub = inputs / "Pipeline Novel.epub"
    epub.write_bytes(build_epub("Pipeline Novel"))
    host = _Host(answers)
    cfg = {"model": "gpt-4o-mini", "api_key": "sk-test-0000", "output_directory": str(tmp_path / "out"),
           "auto_update_check": False, "batch_translation": False}
    cfg.update(config)
    saved_argv = list(sys.argv)
    with headless_owner(tmp_path / "app", monkeypatch, cfg, host=host) as owner:
        os.environ["GLOSSARION_LIBRARY_DIR"] = str(tmp_path / "Library")
        if direct_text:
            from headless_owner import DirectTextRunOptions

            DirectTextRunOptions(selected_files=[str(epub)], force_no_glossary=False,
                                 force_multipass_off=True).apply_to(owner)
        with job_runner.job_process_state(host.log):
            stop_control.reset_for_new_run("translation")
            request = owner._prepare_translation_run([str(epub)])
            assert request is not None
            outcome = owner._translation_worker(request)
        state = {"stop_requested": owner.stop_requested, "translation_thread": owner.translation_thread,
                 "modules_loaded": owner._modules_loaded, "manual_glossary_path": owner.manual_glossary_path}
    assert sys.argv == saved_argv
    return backends, host, request, outcome, state, epub


def test_headless_owner_runs_balanced_pre_glossary_then_translation(tmp_path, monkeypatch):
    import translation_pipeline as tp

    backends, host, request, outcome, state, epub = _run_mobile_job(
        tmp_path, monkeypatch, {"auto_glossary_mode": "balanced"})
    assert isinstance(request, tp.RunRequest)
    assert request.files == [str(epub)] and request.run_id and not request.direct_text
    assert outcome is True  # run_translation_direct's result (the desktop thread ignores it)
    assert [c[0] for c in backends.calls] == ["glossary", "translation"], host.logs[-40:]
    glossary_env = backends.calls[0][1]
    assert glossary_env["GLOSSARY_REQUEST_MERGING_ENABLED"] == "1"
    assert glossary_env["OUTPUT_PATH"].replace("\\", "/").endswith("Pipeline Novel_glossary.json")
    translation_env = backends.calls[1][1]
    assert translation_env["MODEL"] == "gpt-4o-mini"
    assert (translation_env["MANUAL_GLOSSARY"] or "").replace("\\", "/").endswith("Pipeline Novel_glossary.json")
    assert any("📑 Auto Glossary Mode: Balanced" in line for line in host.logs)
    assert any("📑 Auto-loaded generated glossary" in line for line in host.logs)
    assert any("✅ Translation completed successfully!" in line for line in host.logs), host.logs[-30:]
    assert ("thread_complete", {}) in host.events
    assert host.questions == []
    assert state["stop_requested"] is False and state["translation_thread"] is None
    assert state["modules_loaded"] is True


@pytest.mark.parametrize("answer", [True, False])
def test_headless_owner_direct_text_waits_for_glossary_approval(tmp_path, monkeypatch, answer):
    backends, host, request, outcome, _state, _epub = _run_mobile_job(
        tmp_path, monkeypatch, {"auto_glossary_mode": "balanced"}, direct_text=True, answers=[answer])
    assert request.direct_text is True
    assert outcome is answer  # declined approval: the worker reports the run did not translate
    assert len(host.questions) == 1 and host.questions[0][0] == "direct_text_glossary_approval"
    assert host.questions[0][1]["path"].replace("\\", "/").endswith("Pipeline Novel_glossary.json")
    assert any("waiting for approval before translation" in line for line in host.logs)
    if answer:
        assert [c[0] for c in backends.calls] == ["glossary", "translation"]
    else:
        assert [c[0] for c in backends.calls] == ["glossary"]
        assert any("cancelled at the glossary approval step" in line for line in host.logs)
    assert ("thread_complete", {}) in host.events


def test_headless_owner_prepare_with_no_input_reports_instead_of_raising(tmp_path, monkeypatch):
    from _headless_env import headless_owner

    host = _Host()
    with headless_owner(tmp_path / "app", monkeypatch, {"model": "gpt-4o-mini"}, host=host) as owner:
        assert owner.entry_epub.text() == "No file selected"
        assert owner._prepare_translation_run([]) is None
    assert ("message", {"level": "critical", "title": "Error",
                        "text": "Please select file(s) to translate."}) in host.events
