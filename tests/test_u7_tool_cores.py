"""U7 tool cores: frozen desktop handlers vs the shared GUI-free functions they now call.

Moved in U7 (base ``U7_BASE_SHA``, read with ``git show``):

* QA Scanner: the ``run_scan`` worker body of ``QA_Scanner_GUI.run_qa_scan`` ->
  ``qa_scan_runtime.run_bulk_qa_scan``; its ``_load_current_qa_settings`` closure ->
  ``load_current_qa_settings``; the cancel-flag reset -> ``reset_qa_cancel_flags``; the flag
  blocks of ``translator_gui.stop_qa_scan`` / ``_do_qa_force_stop`` / ``_check_qa_stop_done`` ->
  ``next_qa_stop_phase`` / ``apply_qa_graceful_stop_flags`` / ``apply_qa_force_stop_flags`` /
  ``clear_qa_stop_flags`` (translator_gui calls them since the U7 integration; the frozen and the
  rewired methods are both run here);
* other_settings: "Delete Header Files" / "Delete TOC.txt" -> ``output_tools_core``
  plan / delete / result; "Validate EPUB Structure" -> ``validate_epub_outputs``; the
  "Load Font…" copy loop -> ``import_custom_fonts``; the "Translate Headers Now" worker ->
  ``translate_headers_standalone.run_translate_headers_now``;
* ``translate_headers_standalone.run_translate_headers_gui`` -> ``translate_headers_now``
  (GUI-free; the wrapper passes the message box and the event pump);
* metadata_batch_translator ``configure_metadata_fields`` tables and save rules -> module level.

Each frozen function / closure body is executed against the live module globals (Qt calls
recorded, nothing shown) next to the rewired code on the same sandboxed workspaces, and every
observable is compared: log lines, message boxes (title, text, buttons), the scanner calls, the
files left on disk and translation_progress.json.

Run (repository root)::

    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic \\
        tests/test_u7_tool_cores.py
"""

from __future__ import annotations

import ast
import functools
import json
import os
import random
import re
import subprocess
import sys
import textwrap
import types
import zipfile
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import metadata_batch_translator as mbt  # noqa: E402
import output_tools_core as otc  # noqa: E402
import qa_scan_runtime as qsr  # noqa: E402
import translate_headers_standalone as ths  # noqa: E402

U7_BASE_SHA = "41814faa95e273e956870bd3aac5a2c6fb7d66b1"
SEED = int(os.environ.get("PARITY_U7_TOOLS_SEED", "7707"))
STATES = max(1, int(os.environ.get("PARITY_U7_TOOLS_STATES", "40")))


@functools.lru_cache(maxsize=None)
def _frozen(relpath):
    try:
        data = subprocess.run(["git", "show", f"{U7_BASE_SHA}:{relpath}"], cwd=str(REPO_ROOT),
                              capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        return None, str(exc)
    return data.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n"), None


def frozen_text(relpath):
    text, error = _frozen(relpath)
    if text is None:
        pytest.skip(f"{relpath}@{U7_BASE_SHA[:8]} unavailable: {error}")
    return text


def node_source(text, node):
    lines = text.split("\n")
    return textwrap.dedent("\n".join(lines[node.lineno - 1:node.end_lineno]))


def find_function(text, name, cls=None):
    tree = ast.parse(text)
    scope = tree.body
    if cls:
        scope = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls).body
    return next(n for n in scope if isinstance(n, ast.FunctionDef) and n.name == name)


def nested_function(text, outer, inner, cls=None):
    fn = find_function(text, outer, cls)
    return next(n for n in ast.walk(fn) if isinstance(n, ast.FunctionDef) and n.name == inner)


def frozen_function(relpath, name, globals_, cls=None):
    """The frozen top-level function / method ``name`` compiled against ``globals_``."""
    text = frozen_text(relpath)
    src = node_source(text, find_function(text, name, cls))
    ns = dict(globals_)
    exec(compile(src, f"<frozen {relpath}:{name}>", "exec"), ns)
    fn = ns[name]
    # run against the live module dict itself, so monkeypatched module globals apply to both sides
    frozen = types.FunctionType(fn.__code__, globals_, fn.__name__, fn.__defaults__, fn.__closure__)
    frozen.__kwdefaults__ = fn.__kwdefaults__
    return frozen


def norm_lines(src):
    return [line.strip() for line in src.split("\n") if line.strip() and not line.strip().startswith("#")]


# =============================================================================================
# QA Scanner bulk loop
# =============================================================================================

def _frozen_run_scan(qa_globals):
    """The frozen ``run_scan`` worker as ``run_scan(self, <closure vars>)`` (its try body)."""
    text = frozen_text("src/QA_Scanner_GUI.py")
    run_scan = nested_function(text, "run_qa_scan", "run_scan", cls="QAScannerMixin")
    outer = run_scan.body[0]
    lines = text.split("\n")
    body = textwrap.dedent("\n".join(lines[outer.body[0].lineno - 1:outer.body[-1].end_lineno]))
    params = ("self, folders_to_scan, mode, epub_path, qa_settings, _load_current_qa_settings, selected_mode_value, "
              "disable_word_count_for_run, _epub_basename_map, global_selected_files")
    src = f"def frozen_run_scan({params}):\n" + textwrap.indent(body, "    ")
    ns = dict(qa_globals)
    exec(compile(src, "<frozen run_scan>", "exec"), ns)
    return ns["frozen_run_scan"]


class _ScanRecorder:
    def __init__(self, fail_on=()):
        self.calls = []
        self.fail_on = set(fail_on)

    def __call__(self, folder, log=print, stop_flag=None, mode=None, qa_settings=None, epub_path=None,
                 selected_files=None, text_file_mode=None, owner=None, **kw):
        self.calls.append({"folder": folder, "mode": mode, "settings": dict(qa_settings or {}), "epub": epub_path,
                           "selected": selected_files, "text_mode": text_file_mode, "owner": owner is not None,
                           "stopped": bool(stop_flag())})
        log(f"scanned {os.path.basename(folder)}")
        if os.path.basename(folder) in self.fail_on:
            raise RuntimeError(f"scan failed for {os.path.basename(folder)}")
        report = os.path.join(folder, os.path.basename(folder.rstrip("/\\")) + "_Scan Report", "validation_results.html")
        os.makedirs(os.path.dirname(report), exist_ok=True)
        Path(report).write_text("r", encoding="utf-8")


def _qa_workspace(rng, root):
    root = Path(root)
    names = [f"Book{i}{rng.choice(['', '_output', '_translated', '_en'])}" for i in range(rng.randint(1, 4))]
    folders, epubs = [], {}
    for name in names:
        folder = root / "out" / name
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "c1.html").write_text("<p>x</p>", encoding="utf-8")
        folders.append(str(folder))
        base = re.sub(r"(_output|_translated|_en)$", "", name)
        where = rng.choice(["map", "map_stripped", "disk_parent", "disk_inside", "pdf", "none"])
        if where == "map":
            epubs[name] = str(root / "raw" / f"{name}.epub")
        elif where == "map_stripped":
            epubs[base] = str(root / "raw" / f"{base}.epub")
        elif where == "disk_parent":
            (root / "out" / f"{base}.epub").write_bytes(b"x")
        elif where == "disk_inside":
            (folder / f"{name}.epub").write_bytes(b"x")
        elif where == "pdf":
            (root / "out" / f"{name}.pdf").write_bytes(b"%PDF")
    return folders, epubs


def test_bulk_qa_scan_matches_the_frozen_worker(tmp_path, monkeypatch):
    import QA_Scanner_GUI

    frozen_scan = _frozen_run_scan(vars(QA_Scanner_GUI))
    rng = random.Random(SEED)
    for state in range(STATES):
        root = tmp_path / f"s{state}"
        folders, epubs = _qa_workspace(rng, root)
        settings = {"check_word_count_ratio": rng.random() < 0.6, "check_ai_truncation_detection": rng.random() < 0.3,
                    "check_silent_truncation": rng.random() < 0.3, "warn_name_mismatch": rng.random() < 0.8,
                    "cache_show_stats": False, "custom_output_suffixes": "", "custom_mode_settings": {"x": 1}}
        mode = rng.choice(["quick-scan", "aggressive", "custom"])
        epub_path = rng.choice([None, str(root / "raw" / "Global.epub")])
        disable_wc = rng.random() < 0.2
        fail_on = {os.path.basename(f) for f in folders if rng.random() < 0.2}
        stop_at = rng.choice([None, None, 1, 2])
        selected = rng.choice([None, ["a.html"]])
        results = []
        for side in ("frozen", "shared"):
            recorder = _ScanRecorder(fail_on)
            monkeypatch.setattr(qsr, "run_qa_scan_path", recorder)
            logs = []
            owner = types.SimpleNamespace(append_log=logs.append, stop_requested=False, last_qa_report_path=None)
            qa_settings = dict(settings)
            loads = []

            def load():
                loads.append(1)
                if stop_at is not None and len(loads) >= stop_at:
                    owner.stop_requested = True
                return {k: v for k, v in settings.items() if k != "custom_mode_settings"}

            error = None
            try:
                if side == "frozen":
                    frozen_scan(owner, folders, mode, epub_path, qa_settings, load, mode, disable_wc, dict(epubs), selected)
                else:
                    qsr.run_bulk_qa_scan(folders, mode=mode, epub_path=epub_path, qa_settings=qa_settings,
                                         load_settings=load, selected_mode_value=mode,
                                         disable_word_count_for_run=disable_wc, epub_basename_map=dict(epubs),
                                         global_selected_files=selected, log=owner.append_log,
                                         stop_flag=lambda: owner.stop_requested, owner=owner,
                                         on_report=lambda p: setattr(owner, "last_qa_report_path", p))
            except Exception as exc:  # single-folder failures propagate on both sides
                error = repr(exc)
            results.append({"logs": logs, "calls": recorder.calls, "report": owner.last_qa_report_path,
                            "qa_settings": qa_settings, "error": error})
        assert results[0] == results[1], (state, folders, epubs)


def test_qa_settings_loader_reset_and_stop_flags_are_the_frozen_blocks(monkeypatch):
    qg = frozen_text("src/QA_Scanner_GUI.py")
    loader = norm_lines(node_source(qg, nested_function(qg, "run_qa_scan", "_load_current_qa_settings",
                                                          cls="QAScannerMixin")))
    live = norm_lines(node_source(Path(qsr.__file__).read_text(encoding="utf-8"),
                                  find_function(Path(qsr.__file__).read_text(encoding="utf-8"), "load_current_qa_settings")))
    for line in ("main_lang = self.config.get('output_language') or os.getenv('OUTPUT_LANGUAGE', '')",
                 "self.config.get('qa_scanner_settings', {}),",
                 "return dict(self.config.get('qa_scanner_settings', {}) or {})"):
        assert line in loader and line.replace("self.config", "config") in live, line
    assert qsr.load_current_qa_settings({"qa_scanner_settings": {"check_word_count_ratio": False},
                                         "output_language": "English"})["check_word_count_ratio"] is False

    # the stop escalation against the frozen translator_gui methods (flag effects only)
    for key in ("GRACEFUL_STOP", "TRANSLATION_CANCELLED"):
        # record the original state for teardown: a delenv(raising=False) of an absent key records
        # nothing, and the code under test sets these directly (GRACEFUL_STOP=1 leaked to later files)
        monkeypatch.setenv(key, "x")
        monkeypatch.delenv(key)
    tg = frozen_text("src/translator_gui.py")
    calls = []
    fake_client = types.SimpleNamespace(set_stop_flag=lambda v: calls.append(("set_stop_flag", v)),
                                        UnifiedClient=types.SimpleNamespace(_global_cancelled=None),
                                        _cancel_event=types.SimpleNamespace(clear=lambda: calls.append("clear_event")))
    monkeypatch.setitem(sys.modules, "unified_api_client", fake_client)
    monkeypatch.setitem(sys.modules, "scan_html_folder", types.SimpleNamespace(stop_scan=lambda: calls.append("stop_scan")))
    timer = types.SimpleNamespace(singleShot=lambda *a: calls.append("timer"))
    ns = {"os": os, "QTimer": timer}
    for name in ("stop_qa_scan", "_do_qa_force_stop"):
        exec(compile(node_source(tg, find_function(tg, name, cls="TranslatorGUI")), name, "exec"), ns)
    # the rewired translator_gui methods (U7 integration: they call the qa_scan_runtime helpers)
    live_tg = (SRC / "translator_gui.py").read_text(encoding="utf-8-sig").replace("\r\n", "\n")
    live_ns = {"os": os, "QTimer": timer}
    for name in ("stop_qa_scan", "_do_qa_force_stop", "_check_qa_stop_done"):
        exec(compile(node_source(live_tg, find_function(live_tg, name, cls="TranslatorGUI")), name, "exec"), live_ns)
    for name in ("_check_qa_stop_done",):
        exec(compile(node_source(tg, find_function(tg, name, cls="TranslatorGUI")), name, "exec"), ns)
    for graceful in (True, False):
        for clicks in (1, 2, 3):
            snapshots = []
            for side in ("frozen", "shared", "rewired"):
                for key in ("GRACEFUL_STOP", "TRANSLATION_CANCELLED"):
                    monkeypatch.delenv(key, raising=False)
                calls.clear()
                fake_client.UnifiedClient._global_cancelled = None
                owner = types.SimpleNamespace(graceful_stop_var=graceful, append_log=lambda m: None, stop_requested=False,
                                              _do_qa_force_stop=None, _check_qa_stop_done=None)
                owner._do_qa_force_stop = types.MethodType((live_ns if side == "rewired" else ns)["_do_qa_force_stop"],
                                                           owner)
                phase = "idle"
                for _ in range(clicks):
                    if side == "frozen":
                        ns["stop_qa_scan"](owner)
                    elif side == "rewired":
                        live_ns["stop_qa_scan"](owner)
                    else:
                        new = qsr.next_qa_stop_phase(phase, graceful)
                        if new == "graceful":
                            owner.stop_requested = True
                            qsr.apply_qa_graceful_stop_flags()
                        elif new == "force":
                            owner.stop_requested = True
                            qsr.apply_qa_force_stop_flags()
                        if new:
                            owner._qa_stop_phase = phase = new
                snapshots.append((os.environ.get("GRACEFUL_STOP"), os.environ.get("TRANSLATION_CANCELLED"),
                                  [c for c in calls if c != "timer"], fake_client.UnifiedClient._global_cancelled,
                                  getattr(owner, "_qa_stop_phase", None), owner.stop_requested))
            assert snapshots[0] == snapshots[1] == snapshots[2], (graceful, clicks)
    # _check_qa_stop_done after a stopped scan: the frozen vs the rewired delayed flag cleanup
    done = []
    for side_ns in (ns, live_ns):
        side_ns["QTimer"] = types.SimpleNamespace(singleShot=lambda _ms, fn: fn())  # run the 3 s cleanup now
        for phase_after in ("idle", "graceful"):
            calls.clear()
            monkeypatch.setenv("TRANSLATION_CANCELLED", "1")
            monkeypatch.setenv("GRACEFUL_STOP", "1")
            fake_client.UnifiedClient._global_cancelled = True
            owner = types.SimpleNamespace(_qa_stop_poll_generation=3, _qa_stop_poll_current_gen=3, _qa_stop_phase="force",
                                          qa_thread=None, qa_future=None, update_run_button=lambda: None)
            if phase_after != "idle":  # a new scan started before the cleanup ran
                owner.update_run_button = lambda o=owner: setattr(o, "_qa_stop_phase", phase_after)
            side_ns["_check_qa_stop_done"](owner)
            done.append((phase_after, os.environ.get("TRANSLATION_CANCELLED"), os.environ.get("GRACEFUL_STOP"),
                         list(calls), fake_client.UnifiedClient._global_cancelled, owner._qa_stop_phase))
        side_ns["QTimer"] = timer
    assert done[:2] == done[2:] and done[0][1] is None and done[1][1] == "1", done
    # the delayed cleanup and the run-start reset
    cleanup = norm_lines(node_source(tg, nested_function(tg, "_check_qa_stop_done", "_delayed_flag_cleanup",
                                                           cls="TranslatorGUI")))
    shared_cleanup = norm_lines(node_source(Path(qsr.__file__).read_text(encoding="utf-8"),
                                            find_function(Path(qsr.__file__).read_text(encoding="utf-8"), "clear_qa_stop_flags")))
    for line in ("os.environ.pop('TRANSLATION_CANCELLED', None)", "unified_api_client.set_stop_flag(False)",
                 "unified_api_client.UnifiedClient._global_cancelled = False"):
        assert line in cleanup and line in shared_cleanup, line
    calls.clear()
    monkeypatch.setenv("TRANSLATION_CANCELLED", "1")
    monkeypatch.setenv("GRACEFUL_STOP", "1")
    qsr.reset_qa_cancel_flags()
    assert os.environ["TRANSLATION_CANCELLED"] == "0" and "GRACEFUL_STOP" not in os.environ
    assert calls == ["clear_event", ("set_stop_flag", False)] and fake_client.UnifiedClient._global_cancelled is False


# =============================================================================================
# other_settings: Delete Header Files / Delete TOC.txt, Validate EPUB Structure, Load Font
# =============================================================================================

@pytest.fixture
def qt_boxes(monkeypatch):
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    QMessageBox = QtWidgets.QMessageBox
    record = {"boxes": [], "answers": []}

    def fake_exec(box):
        buttons = [b.text() for b in box.buttons()]
        answer = record["answers"].pop(0) if record["answers"] else "yes"
        record["boxes"].append([box.windowTitle(), box.text(), sorted(buttons), answer])
        chosen = next((b for b in box.buttons() if b.text() == answer), None)
        box.setProperty("u7_choice", answer)
        record.setdefault("chosen", {})[id(box)] = chosen
        return QMessageBox.Yes if answer == "yes" else (QMessageBox.No if answer == "no" else QMessageBox.Cancel)

    monkeypatch.setattr(QMessageBox, "exec", fake_exec)
    monkeypatch.setattr(QMessageBox, "clickedButton", lambda box: record.get("chosen", {}).get(id(box)))
    yield record
    app.processEvents()


def _cache_workspace(rng, root, kind):
    """EPUB names + an OUTPUT_DIRECTORY with some workspaces holding the caches."""
    root = Path(root)
    out = root / "out"
    epubs = []
    for i in range(rng.randint(1, 4)):
        name = f"Novel {i}"
        epubs.append(str(root / "raw" / f"{name}.epub"))
        folder = out / name
        layout = rng.choice(["both", "headers", "toc", "toc_lower", "none", "no_folder", "empty_folder", "linked"])
        if layout == "no_folder":
            continue
        folder.mkdir(parents=True, exist_ok=True)
        if layout != "empty_folder":
            (folder / "c1.xhtml").write_text("<p>x</p>", encoding="utf-8")
        if layout in ("both", "headers", "linked"):
            (folder / "translated_headers.txt").write_text("h", encoding="utf-8")
        if layout in ("both", "toc", "linked"):
            (folder / "TOC.txt").write_text("t", encoding="utf-8")
        if layout == "toc_lower":
            (folder / "toc.txt").write_text("t", encoding="utf-8")
        if layout == "linked":
            (folder / "translation_progress.json").write_text(json.dumps({"chapters": {
                "hdr": {"special_type": "headers", "output_file": "translated_headers.txt", "status": "completed",
                        "model_name": "RECYCLED"},
                "toc": {"special_type": "toc", "output_file": "TOC.txt", "status": "completed", "model_name": "m"}}}),
                encoding="utf-8")
    return epubs, out


def _snapshot(folder):
    out = {}
    for path in sorted(Path(folder).rglob("*")) if Path(folder).exists() else []:
        if path.is_file():
            data = path.read_text(encoding="utf-8")
            if path.name == "translation_progress.json":
                data = re.sub(r"\d{4}-\d{2}-\d{2}T[\d:.]+", "<ts>", data)
                data = re.sub(r"\b\d{10}\.\d+\b", "<epoch>", data)
            out[path.relative_to(folder).as_posix()] = data
    return out


@pytest.mark.parametrize("name,kind", [("delete_translated_headers_file", "headers"), ("delete_toc_txt_file", "toc")])
def test_cache_deletion_matches_the_frozen_buttons(name, kind, qt_boxes, tmp_path, monkeypatch):
    import other_settings
    import translation_artifacts

    frozen = frozen_function("src/other_settings.py", name, vars(other_settings))
    live = getattr(other_settings, name)
    rng = random.Random(SEED + (1 if kind == "toc" else 0))
    seen = set()
    for state in range(STATES):
        snapshot_rng_state = rng.getstate()
        results = []
        answer = rng.choice(["yes", "no", "Delete Both Linked Files", "Delete Only Header Files",
                             "Delete Only TOC Files", "Cancel"])
        for side, fn in (("frozen", frozen), ("live", live)):
            rng.setstate(snapshot_rng_state)
            root = tmp_path / f"{side}{state}"
            epubs, out = _cache_workspace(rng, root, kind)
            monkeypatch.setenv("OUTPUT_DIRECTORY", str(out))
            monkeypatch.chdir(root.parent)
            logs = []
            selection = rng.choice(["selected", "current", "entry", "none"])
            owner = types.SimpleNamespace(config={}, append_log=logs.append,
                                          selected_files=epubs if selection == "selected" else [],
                                          get_current_epub_path=lambda: epubs[0] if selection == "current" else None,
                                          entry_epub=types.SimpleNamespace(text=lambda: "No file selected"))
            qt_boxes["boxes"] = []
            qt_boxes["answers"] = [answer, "yes"]
            fn(owner)
            results.append({"logs": [l.replace(str(root), "<root>") for l in logs],
                            "boxes": [[b[0], b[1].replace(str(root), "<root>"), b[2], b[3]] for b in qt_boxes["boxes"]],
                            "files": _snapshot(out)})
            seen.add(len(qt_boxes["boxes"]))
        if results[0] != results[1]:
            diff = {k: (results[0]["files"].get(k), results[1]["files"].get(k))
                    for k in set(results[0]["files"]) | set(results[1]["files"])
                    if results[0]["files"].get(k) != results[1]["files"].get(k)}
            if os.environ.get("U7_DEBUG_DIR"):
                Path(os.environ["U7_DEBUG_DIR"], "diff.json").write_text(json.dumps(diff, indent=1), encoding="utf-8")
            assert results[0] == results[1], (state, diff)
    assert {1, 2} <= seen  # confirmation-only and confirmation + result paths ran
    assert translation_artifacts.translation_artifacts_are_recycled_linked is otc.translation_artifacts_are_recycled_linked


def test_validate_epub_structure_matches_the_frozen_button(qt_boxes, tmp_path, monkeypatch):
    import other_settings

    frozen = frozen_function("src/other_settings.py", "validate_epub_structure_gui", vars(other_settings))
    fake = types.ModuleType("TransateKRtoEN")
    fake.validate_epub_structure = lambda folder: "bad" not in folder
    fake.check_epub_readiness = lambda folder: "warn" not in folder

    def boom(folder):
        if "boom" in folder:
            raise RuntimeError("cannot read")
        return "bad" not in folder

    fake.validate_epub_structure = boom
    monkeypatch.setitem(sys.modules, "TransateKRtoEN", fake)
    monkeypatch.setitem(sys.modules, "winsound", types.SimpleNamespace(MessageBeep=lambda *_: None, MB_OK=0))
    rng = random.Random(SEED + 5)
    for state in range(STATES):
        names = [rng.choice(["ok", "warn", "bad", "boom", "missing"]) + str(i) for i in range(rng.randint(1, 4))]
        results = []
        for side, fn in (("frozen", frozen), ("live", other_settings.validate_epub_structure_gui)):
            root = tmp_path / f"{side}{state}"
            out = root / "out"
            for name in names:
                if not name.startswith("missing"):
                    (out / name).mkdir(parents=True, exist_ok=True)
                    (out / name / "c.xhtml").write_text("x", encoding="utf-8")
            monkeypatch.setenv("OUTPUT_DIRECTORY", str(out))
            monkeypatch.chdir(root.parent)
            logs = []
            btn = types.SimpleNamespace(texts=[], setText=lambda t, b=None: None, setStyleSheet=lambda s: None)
            btn.setText = lambda t, _b=btn: _b.texts.append(t)
            owner = types.SimpleNamespace(config={}, append_log=logs.append,
                                          selected_files=[str(root / "raw" / f"{n}.epub") for n in names],
                                          _validate_btn=btn, _validate_status=None)
            qt_boxes["boxes"] = []
            fn(owner)
            results.append({"logs": logs, "boxes": [[b[0], b[1], b[2]] for b in qt_boxes["boxes"]], "btn": btn.texts})
        assert results[0] == results[1], (state, names)


def test_load_font_copy_rules_are_the_frozen_loop(tmp_path):
    text = frozen_text("src/other_settings.py")
    closure = next(n for n in ast.walk(ast.parse(text)) if isinstance(n, ast.FunctionDef) and n.name == "_on_load_font_clicked")
    block = next(n for n in ast.walk(closure) if isinstance(n, ast.If) and ast.unparse(n.test) == "files")
    frozen_loop = norm_lines(textwrap.dedent("\n".join(text.split("\n")[block.body[2].lineno - 1:block.body[-3].end_lineno])))
    live_src = node_source(Path(otc.__file__).read_text(encoding="utf-8"),
                           find_function(Path(otc.__file__).read_text(encoding="utf-8"), "import_custom_fonts"))
    live = norm_lines(live_src)
    for line in frozen_loop:
        assert line.replace("_os.", "os.").replace("_FONT_EXTS", "font_exts") in live, line
    zipped = tmp_path / "fonts.zip"
    with zipfile.ZipFile(zipped, "w") as zf:
        zf.writestr("a/One.ttf", b"1")
        zf.writestr("Two.WOFF2", b"2")
        zf.writestr("readme.txt", b"x")
        zf.writestr("dir/", b"")
    single = tmp_path / "Three.otf"
    single.write_bytes(b"3")
    target = tmp_path / "fonts"
    target.mkdir()
    assert otc.import_custom_fonts([str(zipped), str(single), str(tmp_path / "skip.txt"), str(tmp_path / "gone.ttf")],
                                   str(target)) == 3
    assert sorted(os.listdir(target)) == ["One.ttf", "Three.otf", "Two.WOFF2"]


# =============================================================================================
# Translate Headers Now
# =============================================================================================

class _HeaderGui:
    def __init__(self, files, current, stop_after=None):
        self.logs = []
        self.selected_files = files
        self._current = current
        self.config = {"batch_header_system_prompt": "S", "batch_header_prompt": "P"}
        self.api_client = object()
        self.headers_per_batch_var = 5
        self.failed_translation_retry_attempts_var = 2
        self.update_html_headers_var = True
        self.save_header_translations_var = False
        self._headers_stop_requested = False
        self._stop_after = stop_after

    def append_log(self, message):
        self.logs.append(str(message))
        if self._stop_after is not None and len(self.logs) >= self._stop_after:
            self._headers_stop_requested = True

    def get_current_epub_path(self):
        return self._current


def _header_fakes(monkeypatch, calls):
    def translate(epub_path, output_dir, api_client, config, update_html, save_to_file, log_callback, gui_instance):
        calls.append(("translate", os.path.basename(epub_path), os.path.basename(output_dir), sorted(config)))
        log_callback("translated")
        if "noop" in epub_path:
            return types.SimpleNamespace(successful_noop=True, __bool__=lambda self: False)
        return {"c1": "One"} if "empty" not in epub_path else {}

    monkeypatch.setattr(ths, "translate_headers_standalone", translate)
    monkeypatch.setattr(ths, "apply_existing_translations",
                        lambda **kw: calls.append(("apply", os.path.basename(kw["epub_path"]))) or {"c1": "x"})
    monkeypatch.setattr(ths, "repair_translation_file", lambda *a, **k: calls.append(("repair", os.path.basename(a[0]))))
    monkeypatch.setattr(ths, "retry_failed_header_translations", lambda *a, **k: ({}, {}, {}))
    monkeypatch.setattr(ths, "extract_source_chapters_with_opf_mapping", lambda *a, **k: ({}, []))
    monkeypatch.setitem(sys.modules, "metadata_batch_translator", types.SimpleNamespace(
        BatchHeaderTranslator=lambda client, config: types.SimpleNamespace(translate_headers_batch=lambda *a, **k: {})))
    monkeypatch.setitem(sys.modules, "pdf_workspace_compiler", types.SimpleNamespace(
        load_pdf_workspace_artifact_chapters=lambda d: [],
        translate_pdf_workspace_artifacts=lambda *a, **k: {"headers": 3}))


def test_translate_headers_now_matches_the_frozen_gui_function(qt_boxes, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    boxes = []
    monkeypatch.setattr(QMessageBox, "critical", staticmethod(lambda parent, title, text, *a: boxes.append([title, text])))
    frozen = frozen_function("src/translate_headers_standalone.py", "run_translate_headers_gui", vars(ths))
    rng = random.Random(SEED + 11)
    for state in range(STATES):
        kinds = [rng.choice(["epub", "epub", "pdf", "existing", "missing", "noop", "empty"]) for _ in range(rng.randint(0, 3))]
        results = []
        for side in ("frozen", "live"):
            root = tmp_path / f"{side}{state}"
            out = root / "out"
            files = []
            for i, kind in enumerate(kinds):
                ext = ".pdf" if kind == "pdf" else ".epub"
                name = f"{kind}{i}"
                files.append(str(root / "raw" / f"{name}{ext}"))
                folder = out / (f"{name}_PDF" if kind == "pdf" else name)
                if kind != "missing":
                    folder.mkdir(parents=True, exist_ok=True)
                    (folder / "c1.xhtml").write_text("x", encoding="utf-8")
                if kind == "existing":
                    (folder / "translated_headers.txt").write_text("Chapter 1:\n", encoding="utf-8")
            monkeypatch.setenv("OUTPUT_DIRECTORY", str(out))
            monkeypatch.chdir(root.parent)
            calls = []
            _header_fakes(monkeypatch, calls)
            gui = _HeaderGui(files, files[0] if files else None, stop_after=rng.choice([None, None, 6]) if side == "frozen" else None)
            stop_after = gui._stop_after
            if side == "live":
                gui._stop_after = results[0]["stop_after"]
            boxes.clear()
            if side == "frozen":
                frozen(gui)
            else:
                ths.run_translate_headers_gui(gui)
            results.append({"logs": [l.replace(str(root), "<root>") for l in gui.logs], "calls": calls,
                            "boxes": [[t, x.replace(str(root), "<root>")] for t, x in boxes],
                            "stop_after": stop_after if side == "frozen" else results[0]["stop_after"]})
        if results[0] != results[1] and os.environ.get("U7_DEBUG_DIR"):
            Path(os.environ["U7_DEBUG_DIR"], "headers.json").write_text(json.dumps(results, indent=1, default=str),
                                                                       encoding="utf-8")
        assert results[0] == results[1], (state, kinds)


def test_translate_headers_now_runs_without_qt_and_reports_counts(tmp_path, monkeypatch):
    out = tmp_path / "out"
    for name in ("a", "b"):
        (out / name).mkdir(parents=True)
        (out / name / "c1.xhtml").write_text("x", encoding="utf-8")
    monkeypatch.setenv("OUTPUT_DIRECTORY", str(tmp_path / "elsewhere"))
    calls = []
    _header_fakes(monkeypatch, calls)
    gui = _HeaderGui([str(tmp_path / "raw" / "a.epub"), str(tmp_path / "raw" / "b.epub"), str(tmp_path / "raw" / "c.epub")], None)
    errors = []
    folders = {"a.epub": str(out / "a"), "b.epub": str(out / "b")}
    result = ths.translate_headers_now(gui, show_error=lambda t, x: errors.append(x),
                                       output_dir_for=lambda source: folders.get(os.path.basename(source)))
    assert result == (2, 1) and errors == []
    assert [c[1:3] for c in calls if c[0] == "translate"] == [("a.epub", "a"), ("b.epub", "b")]
    assert any("Output directory not found for: c" in line for line in gui.logs)
    nothing = _HeaderGui([], None)
    assert ths.translate_headers_now(nothing, show_error=lambda t, x: errors.append(x)) is None
    assert errors == ["No EPUB or PDF file selected, or the file does not exist."]
    # the default error sink is the log (no Qt)
    ths.translate_headers_now(nothing)
    assert nothing.logs[-1] == "❌ No EPUB or PDF file selected, or the file does not exist."


def test_header_worker_is_the_frozen_thread_body():
    text = frozen_text("src/other_settings.py")
    thread = nested_function(text, "run_standalone_translate_headers", "translation_thread")
    body = node_source(text, thread.body[0])
    frozen = [line.replace("self", "gui_instance") for line in norm_lines(body)
              if "run_translate_headers_gui" not in line and not line.startswith(("try:", "except", "finally:"))
              and "_headers_translation_running" not in line and "_headers_stop_requested" not in line]
    live_text = Path(ths.__file__).read_text(encoding="utf-8")
    live = norm_lines(node_source(live_text, find_function(live_text, "run_translate_headers_now")))
    for line in frozen:
        assert line in live, line
    assert "headers_runner(gui_instance)" in live


# =============================================================================================
# metadata field rules
# =============================================================================================

def test_metadata_field_rules_are_the_frozen_dialog_code():
    text = frozen_text("src/metadata_batch_translator.py")
    dialog = find_function(text, "configure_metadata_fields", cls="MetadataBatchTranslatorUI")
    assigns = {t.id: s.value for s in ast.walk(dialog) if isinstance(s, ast.Assign) for t in s.targets
               if isinstance(t, ast.Name)}
    assert ast.literal_eval(assigns["standard_fields"]) == mbt.METADATA_STANDARD_FIELDS
    assert list(ast.literal_eval(assigns["standard_fields"])) == list(mbt.METADATA_STANDARD_FIELDS)
    assert ast.literal_eval(assigns["default_enabled_fields"]) == set(mbt.METADATA_DEFAULT_ENABLED_FIELDS)
    frozen = norm_lines(node_source(text, dialog))
    live_text = Path(mbt.__file__).read_text(encoding="utf-8-sig")
    for helper, lines in (
        ("saved_metadata_fields_for_epub", ("per_epub = translate_fields_config.get('_per_epub', {})",
                                            "return per_epub[basename]", "return translate_fields_config")),
        ("metadata_field_checked", ("checked = field_sync_state[field]", "field_sync_state[field] = checked")),
        ("store_metadata_field_selection", ("translate_fields_config['_per_epub'] = {}",
                                            "translate_fields_config.clear()", "translate_fields_config.update(saved)")),
        ("final_metadata_fields_config", ("for basename, fields in per_epub.items():", "combined.update(fields)",
                                          "combined['_per_epub'] = per_epub")),
    ):
        live = norm_lines(node_source(live_text, find_function(live_text, helper)))
        for line in lines:
            assert line in frozen and line in live, (helper, line)
    # behaviour: one EPUB flat, several merged + per EPUB
    cfg = {"_per_epub": {"Old.epub": {"rights": True}}}
    mbt.store_metadata_field_selection(cfg, ["a.epub", "b.epub"], "/x/a.epub", {"title": False})
    assert mbt.final_metadata_fields_config(cfg, ["a.epub", "b.epub"]) == {
        "rights": True, "title": False, "_per_epub": {"Old.epub": {"rights": True}, "a.epub": {"title": False}}}
    flat = {"_per_epub": {}, "x": 1}
    mbt.store_metadata_field_selection(flat, ["a.epub"], "a.epub", {"title": True})
    assert mbt.final_metadata_fields_config(flat, ["a.epub"]) == {"title": True}
    sync = {}
    assert mbt.metadata_field_checked("title", {}, sync, True) is True and sync == {"title": True}
    assert mbt.metadata_field_checked("title", {"title": False}, sync, True) is True  # synced wins


def test_output_tools_core_is_gui_free_and_python310():
    for name in ("output_tools_core.py", "qa_scan_runtime.py"):
        data = (SRC / name).read_bytes()
        tree = ast.parse(data.decode("utf-8-sig"), feature_version=(3, 10))
        top = {a.name.split(".")[0] for n in tree.body if isinstance(n, ast.Import) for a in n.names}
        top |= {n.module.split(".")[0] for n in tree.body if isinstance(n, ast.ImportFrom) and n.module}
        assert not top & {"PySide6", "translator_gui", "dpi_setup", "other_settings", "QA_Scanner_GUI"}, (name, top)
        assert data.count(b"\r\n") in (0, data.count(b"\n")), name
    text = (SRC / "translate_headers_standalone.py").read_text(encoding="utf-8")
    body = node_source(text.replace("\r\n", "\n"), find_function(text.replace("\r\n", "\n"), "translate_headers_now"))
    assert "PySide6" not in body and "QMessageBox" not in body
