"""Hotfix 1, issue 6 (owner device report): the post-translation QA scan never ran on the phone.

The owner's log::

    ✅ Translation completed successfully!
    🧪 Scanning phase enabled — launching QA Scanner in quick-scan mode...
    ⚠️ The post-translation QA scan is not available in this build yet; skipped
    ■ Done: [TS] I'm the Only Mechanical Genius Girl

Cause: the shared worker (``translation_pipeline._translation_worker``) asks the front end for the scan with
``self._ui_request('trigger_qa_scan')``; desktop ``TranslatorGUI._trigger_qa_scan_on_main_thread`` runs
``run_qa_scan(mode_override=self._get_scan_phase_mode(), non_interactive=True)``, while the GUI-free default
(``PipelineHooksMixin._ui_request``) only logs ``UNSHARED_UI_REQUESTS['trigger_qa_scan']`` and emits a host event
nothing on mobile handles. Fix: ``job_kinds.translate`` catches the request on the job's owner and runs the
scan in the same job (``job_kinds.qa.run_scan_phase`` -> ``qa.scan``, the ``qa_scan`` job's own scan).

* Unit level (no backend): ``translate.run_translation`` over a fake owner / context and a fake QA runtime:
  the scan runs once after the worker returned, in the owner's scan-phase mode, on the job's output folder,
  with the mobile Quick Scan sample size; the "not available" line is never logged; other ``_ui_request``
  kinds still reach the owner; the owner loses the wrapper afterwards; no trigger -> no scan; a Stop before the
  scan skips it, a Stop during it escalates like a QA job; Auto-search output off and Direct Text sources skip
  like the desktop; a scan error leaves the translation Done.
* In a child process (``diagnostics.e2e.E2ESession``: the real JobService, HeadlessOwner, shared pipeline,
  ``qa_scan_runtime`` and ``scan_html_folder`` against the fake OpenAI server, offline): a ``translate`` job of
  the 12-chapter self-test EPUB with the scanning phase on (Quick Scan) ends DONE after the scan, its report
  is written and listed after the compiled EPUB, the chapter the fake model left partly in Korean is
  ``qa_failed`` in ``translation_progress.json`` (the Library / Progress Manager QA flags; the Chapters tab
  row shows the issue) and the job's result carries the same QA fields as a ``qa_scan`` job of the same
  folder; Aggressive mode follows ``scan_phase_mode``; a Stop during the scan ends the job Stopped; with the
  scanning phase off nothing is scanned.

Real data is never touched: the child's HOME / USERPROFILE / APPDATA / LOCALAPPDATA / GLOSSARION_* /
OUTPUT_DIRECTORY / TMP point under pytest's tmp dir, E2ESession redirects CONFIG_FILE, GLOSSARION_HTTP_LOG=0.

Run from src/mobile with the mobile venv::

    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_hotfix1_post_qa.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = MOBILE_DIR / "app"
SRC_DIR = MOBILE_DIR.parent
for _entry in (str(APP_DIR), str(SRC_DIR)):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

UNSHARED_LINE = "⚠️ The post-translation QA scan is not available in this build yet; skipped"
TRIGGER_LINE = "🧪 Scanning phase enabled — launching QA Scanner in"
E2E_DEPENDENCIES = ("ebooklib", "openai", "httpx", "lxml", "bs4", "tiktoken")
SELFTEST_EPUB = APP_DIR / "assets" / "selftest" / "selftest_ko_12ch.epub"  # tools/prepare_assets.py (CI: prepare)
TIKTOKEN_ASSETS = APP_DIR / "assets" / "tiktoken"
CHILD = "--hotfix1-post-qa-child"


# ==========================================================================
# 1. Unit level: translate.run_translation + qa.run_scan_phase over fakes
# ==========================================================================


class FakeHost:
    def __init__(self) -> None:
        self.graceful = True

    def is_graceful_stop(self) -> bool:
        return self.graceful


class FakeCtx:
    """The ``JobContext`` calls the adapters make, recorded."""

    def __init__(self, owner, inputs, config=None) -> None:
        self.owner = owner
        self.inputs = tuple(inputs)
        self.params: dict = {}
        self.config = dict(config or {})
        self.host = FakeHost()
        self.output_dir = None
        self.lines: list = []
        self.phases: list = []
        self.outputs: list = []
        self.results: dict = {}
        self.output_dirs: dict = {}
        self.stop = False
        self.events: list = []

    def log(self, text) -> None:
        self.lines.append(str(text))

    def stop_requested(self) -> bool:
        return self.stop

    def phase(self, label) -> None:
        self.phases.append(label)

    def set_output_dirs(self, mapping) -> None:
        self.output_dirs.update(mapping)
        self.output_dir = self.output_dir or next(iter(mapping.values()), None)

    def set_output_dir(self, path) -> None:
        self.output_dir = path

    def add_outputs(self, paths) -> None:
        self.events.append(("outputs", list(paths)))
        for path in paths:
            if path not in self.outputs:
                self.outputs.append(path)

    def set_result(self, **data) -> None:
        self.results.update(data)

    def ask(self, kind, **data):
        return False

    def request_stop(self, reason=""):
        self.stop = True


class FakeOwner:
    """The shared pipeline seen from the adapter: set-up, worker (asks for the scan like the desktop worker)."""

    def __init__(self, root: Path, *, scan_phase=True, mode="quick-scan", files_per_book=3, ctx=None) -> None:
        self.root = root
        self.config = {"scan_phase_mode": mode}
        self.scan_phase = scan_phase
        self.files_per_book = files_per_book
        self.selected_files: list = []
        self.ui: list = []
        self.logs: list = []
        self.ctx = ctx

    def append_log(self, text) -> None:
        self.logs.append(str(text))
        if self.ctx is not None:
            self.ctx.log(text)

    def _ui_request(self, kind, *args, **data):  # the GUI-free default (PipelineHooksMixin + JobHooksMixin)
        self.ui.append(kind)
        if kind == "trigger_qa_scan":
            self.append_log(UNSHARED_LINE)
        return None

    def _get_scan_phase_mode(self):
        return self.config.get("scan_phase_mode", "quick-scan")

    def _resolve_translation_output_dir(self, path):
        return str(self.root / "Output" / Path(path).stem)

    def _prepare_translation_run(self, files):
        self.selected_files = list(files)
        return types.SimpleNamespace(files=list(files))

    def _translation_worker(self, request):
        for path in self.selected_files:
            folder = Path(self._resolve_translation_output_dir(path))
            folder.mkdir(parents=True, exist_ok=True)
            for index in range(1, self.files_per_book + 1):
                (folder / f"response_{index:03d}.html").write_text("<html><body>x</body></html>", encoding="utf-8")
            (folder / f"{Path(path).stem}.epub").write_bytes(b"PK")
        self.append_log("✅ Translation completed successfully!")
        if self.scan_phase:
            self.append_log(f"{TRIGGER_LINE} {self._get_scan_phase_mode()} mode...")
            self._ui_request("trigger_qa_scan")
        self._ui_request("thread_complete")
        return True


class FakeRuntime:
    """``qa_scan_runtime`` as ``job_kinds.qa`` uses it; ``run_bulk_qa_scan`` writes a report per folder."""

    def __init__(self, *, error=None, during=None) -> None:
        self.calls: list = []
        self.flags: list = []
        self.error = error
        self.during = during

    @staticmethod
    def is_direct_text_qa_path(path) -> bool:
        parts = str(path or "").replace("\\", "/").split("/")
        return any(p.strip().lower().replace(" ", "_") == "direct_text" for p in parts)

    @staticmethod
    def is_html_like_path(path) -> bool:
        return str(path).lower().endswith((".html", ".xhtml", ".htm"))

    def reset_qa_cancel_flags(self) -> None:
        self.flags.append("reset")

    @staticmethod
    def load_current_qa_settings(config) -> dict:
        return {"check_word_count_ratio": False}

    @staticmethod
    def next_qa_stop_phase(phase, graceful):
        if phase == "idle":
            return "graceful" if graceful else "force"
        return "force" if phase == "graceful" else None

    def apply_qa_graceful_stop_flags(self) -> None:
        self.flags.append("graceful")

    def apply_qa_force_stop_flags(self) -> None:
        self.flags.append("force")

    def clear_qa_stop_flags(self) -> None:
        self.flags.append("clear")

    def run_bulk_qa_scan(self, folders, **kw):
        self.calls.append({"folders": list(folders), **kw})
        if self.during is not None:
            self.during(kw)
        if self.error is not None:
            raise self.error
        log = kw["log"]
        done = 0
        for folder in folders:
            if kw["stop_flag"]():
                log(f"⚠️ Bulk scan stopped by user at folder {done + 1}/{len(folders)}")
                break
            report = Path(folder) / (Path(folder).name + "_Scan Report") / "validation_results.html"
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text("<html>report</html>", encoding="utf-8")
            kw["on_report"](str(report))
            done += 1
        log("✅ QA scan completed successfully.")
        return done, 0


@pytest.fixture
def unit(tmp_path, monkeypatch):
    from glossarion_mobile.job_kinds import qa

    runtime = FakeRuntime()
    monkeypatch.setattr(qa, "_qa_runtime", lambda: runtime)
    raw = tmp_path / "Library" / "Raw"
    raw.mkdir(parents=True)
    book = raw / "Mechanical Genius.epub"
    book.write_bytes(b"PK")

    def make(**kw):
        owner = FakeOwner(tmp_path, **kw)
        ctx = FakeCtx(owner, [str(book)])
        owner.ctx = ctx
        return owner, ctx

    return types.SimpleNamespace(runtime=runtime, book=book, make=make, root=tmp_path)


def _run(ctx):
    from glossarion_mobile.job_kinds.translate import run_translation

    return run_translation(ctx, [os.path.abspath(p) for p in ctx.inputs])


def test_the_scan_runs_in_the_job_after_the_worker(unit):
    owner, ctx = unit.make()
    result = _run(ctx)
    folder = os.path.abspath(owner._resolve_translation_output_dir(unit.book))
    report = os.path.join(folder, os.path.basename(folder) + "_Scan Report", "validation_results.html")
    assert result["ok"] is True and not result.get("error")
    assert UNSHARED_LINE not in ctx.lines and UNSHARED_LINE not in owner.logs
    # the worker's own request kinds still reach the owner; the scan request does not
    assert owner.ui == ["thread_complete"]
    assert "_ui_request" not in vars(owner), "the owner kept the job's wrapper"
    # one scan, after the worker returned, of the job's output folder with its source
    assert len(unit.runtime.calls) == 1
    call = unit.runtime.calls[0]
    assert call["folders"] == [folder] and call["mode"] == "quick-scan" and call["selected_mode_value"] == "quick-scan"
    assert os.path.normcase(call["epub_path"]) == os.path.normcase(str(unit.book))
    assert call["qa_settings"]["quick_scan_sample_size"] == 0  # the mobile Quick Scan default
    assert call["owner"] is owner and call["disable_word_count_for_run"] is False
    lines = ctx.lines
    order = [lines.index(t) for t in ("✅ Translation completed successfully!", "✅ QA scanner initialized successfully",
                                      f"📁 Auto-selected output folder for {unit.book.stem}: {folder}",
                                      "   Found 3 HTML/XHTML files to scan",
                                      f"📚 Using EPUB from scan list: {unit.book.name}",
                                      f"🔍 Starting QA scan in QUICK-SCAN mode for folder: {folder}",
                                      "✅ QA scan completed successfully.")]
    assert order == sorted(order), lines
    assert "Scanning" in ctx.phases and unit.runtime.flags == ["reset"]
    # surfaced like a qa_scan job: the report in the outputs (after the compiled book) and the QA result fields
    epub = os.path.join(folder, unit.book.stem + ".epub")
    assert result["outputs"] == [epub, report]
    assert ctx.events[0] == ("outputs", [epub]) and ctx.outputs == [epub, report]
    assert ctx.results == {"qa_reports": [report], "qa_mode": "quick-scan", "qa_folders": [folder]}
    assert os.path.isfile(report)


def test_an_instance_hook_set_before_the_run_is_put_back(unit):
    owner, ctx = unit.make()
    seen: list = []

    def earlier_hook(kind, *args, **data):  # e.g. another adapter's own wrapper on this owner
        seen.append(kind)
        return FakeOwner._ui_request(owner, kind, *args, **data)

    owner._ui_request = earlier_hook
    _run(ctx)
    assert vars(owner)["_ui_request"] is earlier_hook
    assert seen == ["thread_complete"] and len(unit.runtime.calls) == 1


def test_the_scan_follows_the_scan_phase_mode(unit):
    owner, ctx = unit.make(mode="aggressive")
    _run(ctx)
    assert [c["mode"] for c in unit.runtime.calls] == ["aggressive"]
    assert any(line.startswith("🔍 Starting QA scan in AGGRESSIVE mode") for line in ctx.lines)
    assert not any("sample size" in line for line in ctx.lines)  # the Quick Scan line is Quick Scan's


def test_no_trigger_no_scan(unit):
    owner, ctx = unit.make(scan_phase=False)
    result = _run(ctx)
    assert result["ok"] is True and unit.runtime.calls == []
    assert "✅ QA scanner initialized successfully" not in ctx.lines and owner.ui == ["thread_complete"]
    assert not ctx.results


def test_a_stop_before_the_scan_skips_it(unit):
    owner, ctx = unit.make()
    original = owner._translation_worker

    def worker_then_stop(request):
        outcome = original(request)
        ctx.stop = True  # Stop tapped as the translation finished
        return outcome

    owner._translation_worker = worker_then_stop
    _run(ctx)
    assert unit.runtime.calls == []
    assert "⏹️ Post-translation QA scan skipped: the job was stopped" in ctx.lines


def test_a_stop_during_the_scan_uses_the_qa_stop_escalation(unit):
    owner, ctx = unit.make()
    unit.runtime.during = lambda kw: setattr(ctx, "stop", True)
    _run(ctx)
    assert len(unit.runtime.calls) == 1
    assert unit.runtime.flags == ["reset", "graceful", "clear"]
    assert any(line.startswith("⚠️ Bulk scan stopped by user") for line in ctx.lines)
    assert "qa_reports" not in ctx.results


def test_auto_search_off_skips_like_the_desktop(unit):
    owner, ctx = unit.make()
    ctx.config["qa_auto_search_output"] = False
    result = _run(ctx)
    assert result["ok"] is True and unit.runtime.calls == []
    assert ctx.lines[-2:] == ["⚠️ Scanning phase: No matching output folders found for 1 EPUB file(s)",
                              "⚠️ Skipping scan"]
    assert not any("Auto-search output is off" in line for line in ctx.lines)  # no mobile-only line


def _desktop_source_extensions() -> tuple:
    """The extensions ``QA_Scanner_GUI._qa_auto_search_sources`` filters the selection with (its
    ``current_epub_files = [f for f in ... if f.lower().endswith((...))]``)."""
    import ast

    tree = ast.parse((SRC_DIR / "QA_Scanner_GUI.py").read_text(encoding="utf-8-sig"))
    function = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_qa_auto_search_sources")
    for node in ast.walk(function):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "current_epub_files"
                                                for t in node.targets) and isinstance(node.value, ast.ListComp):
            for sub in ast.walk(node.value):
                if (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute) and sub.func.attr == "endswith"
                        and sub.args and isinstance(sub.args[0], ast.Tuple)):
                    return tuple(ast.literal_eval(sub.args[0]))
    raise AssertionError("no source-extension tuple in QA_Scanner_GUI._qa_auto_search_sources")


#: ``run_scan_phase`` lines the desktop scanning phase has no counterpart for (a Stop before the scan: the desktop
#: worker asks for the scan only after a translation that was not stopped)
MOBILE_ONLY_SCAN_LINES = ("⏹️ Post-translation QA scan skipped: the job was stopped",)


def test_scan_phase_sources_and_log_lines_are_the_desktop_ones():
    """``job_kinds.qa.run_scan_phase`` follows the desktop scanning phase (``QA_Scanner_GUI.run_qa_scan`` with
    ``non_interactive=True``): the source extensions are its ``_qa_auto_search_sources`` tuple, and every line it
    logs is a desktop line (QA_Scanner_GUI / translator_gui), word for word, but the listed mobile-only one."""
    import ast

    from glossarion_mobile.job_kinds import qa

    assert qa.SCAN_PHASE_SOURCE_EXTENSIONS == _desktop_source_extensions()
    desktop = "\n".join((SRC_DIR / name).read_text(encoding="utf-8-sig") for name in ("QA_Scanner_GUI.py", "translator_gui.py"))
    tree = ast.parse(Path(qa.__file__).read_text(encoding="utf-8-sig"))
    function = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "run_scan_phase")
    texts = []
    for node in ast.walk(function):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "log" and node.args:
            arg = node.args[0]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                texts.append((arg.value,))
            elif isinstance(arg, ast.JoinedStr):
                texts.append(tuple(v.value for v in arg.values if isinstance(v, ast.Constant) and isinstance(v.value, str)))
    assert len(texts) >= 12, texts
    missing = [parts for parts in texts if "".join(parts) not in MOBILE_ONLY_SCAN_LINES
               and not all(part in desktop for part in parts)]
    assert not missing, f"scan-phase lines that are not desktop lines: {missing}"


def test_direct_text_sources_never_start_the_scan(unit):
    owner, ctx = unit.make()
    dt = unit.root / "Output" / "Direct Text" / "chat" / "Attachments" / "book.epub"
    dt.parent.mkdir(parents=True)
    dt.write_bytes(b"PK")
    ctx.inputs = (str(dt),)
    _run(ctx)
    assert unit.runtime.calls == []
    assert f"⏭️ Excluding Direct Text source from QA scan: {os.path.abspath(dt)}" in ctx.lines
    assert ctx.lines[-1] == "⏭️ QA scan skipped: Direct Text cannot trigger automatic QA scanning."


def test_a_folder_without_chapters_is_not_scanned(unit):
    owner, ctx = unit.make(files_per_book=0)
    _run(ctx)
    folder = os.path.abspath(owner._resolve_translation_output_dir(unit.book))
    assert unit.runtime.calls == []
    assert ctx.lines[-4:] == [f"  [{unit.book.stem}] Folder exists but contains no HTML/XHTML files: {folder}",
                              f"  ⚠️ No output folder found for {unit.book.stem}",
                              "⚠️ Scanning phase: No matching output folders found for 1 EPUB file(s)",
                              "⚠️ Skipping scan"]


def test_a_scan_error_leaves_the_translation_done(unit):
    owner, ctx = unit.make()
    unit.runtime.error = RuntimeError("scanner exploded")
    result = _run(ctx)
    assert result["ok"] is True and not result.get("error")
    assert "❌ QA scan error: scanner exploded" in ctx.lines
    assert any(line.startswith("Traceback: ") for line in ctx.lines)


def test_the_qa_job_kind_still_runs_its_own_scan(unit, tmp_path):
    """``qa_scan`` jobs go through the extracted ``qa.scan`` unchanged."""
    from glossarion_mobile.job_kinds import qa

    folder = tmp_path / "Output" / "Book"
    folder.mkdir(parents=True)
    (folder / "response_001.html").write_text("x", encoding="utf-8")
    ctx = FakeCtx(object(), [str(folder)])
    ctx.params = {"mode": "aggressive", "disable_word_count": True}
    result = qa.run(ctx)
    report = str(folder / "Book_Scan Report" / "validation_results.html")
    assert result == {"ok": True, "outputs": [report]}
    call = unit.runtime.calls[0]
    assert call["mode"] == "aggressive" and call["disable_word_count_for_run"] is True
    assert ctx.results == {"qa_reports": [report], "qa_mode": "aggressive", "qa_folders": [str(folder)]}


# ==========================================================================
# 2. The real job stack in a child process
# ==========================================================================


def _missing_modules(*modules: str) -> list:
    return [m for m in modules if importlib.util.find_spec(m) is None]


def _child_env(tmp_path: Path) -> dict:
    """Every data, output, temp and home folder under ``tmp_path``; the mobile runtime contract; no HTTP log."""
    import site

    env = {k: v for k, v in os.environ.items() if not k.startswith(("FLET_", "GLOSSARION_"))}
    env.pop("CONFIG_FILE", None)
    folders = {}
    for name in ("home", "appdata", "localappdata", "Library", "data", "Output", "work", "tiktoken"):
        folders[name] = tmp_path / name
        folders[name].mkdir(parents=True, exist_ok=True)
    for blob in TIKTOKEN_ASSETS.iterdir():
        if blob.is_file():
            shutil.copy2(blob, folders["tiktoken"] / blob.name)
    if "PYTHONUSERBASE" not in env:
        try:
            env["PYTHONUSERBASE"] = site.getuserbase()  # HOME / APPDATA move: the user site must stay found
        except Exception:
            pass
    env.update(
        HOME=str(folders["home"]), USERPROFILE=str(folders["home"]), APPDATA=str(folders["appdata"]),
        LOCALAPPDATA=str(folders["localappdata"]), GLOSSARION_LIBRARY_DIR=str(folders["Library"]),
        GLOSSARION_DATA_DIR=str(folders["data"]), OUTPUT_DIRECTORY=str(folders["Output"]),
        GLOSSARION_HTTP_LOG="0", GLOSSARION_MOBILE="1", GLOSSARION_NO_PROCESSES="1",
        TIKTOKEN_CACHE_DIR=str(folders["tiktoken"]), TMPDIR=str(folders["work"]), TEMP=str(folders["work"]),
        TMP=str(folders["work"]), PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1",
    )
    return env


@pytest.fixture(scope="module")
def child(tmp_path_factory):
    if not SELFTEST_EPUB.is_file() or not TIKTOKEN_ASSETS.is_dir():
        pytest.skip("run tools/prepare_assets.py first")
    missing = _missing_modules(*E2E_DEPENDENCIES)
    if missing:
        pytest.skip(f"backend dependencies missing: {missing}")
    tmp_path = tmp_path_factory.mktemp("hotfix1_post_qa")
    result_file = tmp_path / "post_qa.json"
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), CHILD, str(result_file), str(tmp_path)],
        cwd=str(tmp_path / "work") if (tmp_path / "work").is_dir() else str(tmp_path),
        env=_child_env(tmp_path), capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=1500,
    )
    tail = (proc.stdout or "")[-3000:] + "\n" + (proc.stderr or "")[-6000:]
    assert result_file.is_file(), f"no result (exit {proc.returncode}):\n{tail}"
    report = json.loads(result_file.read_text(encoding="utf-8"))
    assert report.get("error") is None, f"{report.get('error')}\n{tail}"
    report["_tail"] = tail
    return report


def _lines(job: dict) -> list:
    return list(job.get("log") or [])


def _has_line(job: dict, start: str) -> bool:
    return any(line.startswith(start) for line in _lines(job))


def test_e2e_quick_scan_runs_after_the_translation_and_writes_its_report(child):
    job = child["quick"]
    assert job["state"] == "DONE", (job["state"], job["error"], _lines(job)[-30:])
    lines = _lines(job)
    assert UNSHARED_LINE not in lines
    folder = job["output_dir"]
    report = os.path.join(folder, os.path.basename(folder) + "_Scan Report", "validation_results.html")
    assert job["report_exists"] and os.path.normcase(job["report"]) == os.path.normcase(report)
    starts = ("✅ Translation completed successfully!", TRIGGER_LINE + " quick-scan mode...",
              "✅ QA scanner initialized successfully", "📁 Auto-selected output folder for ",
              "📚 Using EPUB from scan list: ", "🔍 Starting QA scan in QUICK-SCAN mode for folder: ",
              "⚡ Quick Scan duplicate check sample size: 0 (duplicate check off)", "✅ QA scan completed successfully.",
              "■ Done: ")
    found = [next((i for i, line in enumerate(lines) if line.startswith(s)), -1) for s in starts]
    assert -1 not in found and found == sorted(found), list(zip(starts, found))
    # one scan of the job's own workspace with its source, Quick Scan with the mobile sample size 0
    scans = job["scans"]
    assert len(scans) == 1 and os.path.normcase(scans[0]["folder"]) == os.path.normcase(folder)
    assert scans[0]["mode"] == "quick-scan" and scans[0]["sample"] == 0
    assert os.path.basename(scans[0]["epub_path"] or "") == "hotfix1-post-qa.epub"
    assert scans[0]["thread"] == job["job_thread"], "the scan must run inside the translation's job"
    # outputs: the compiled book first, then the report
    outputs = job["outputs"]
    assert outputs and outputs[0].lower().endswith(".epub") and outputs[-1] == job["report"]
    assert job["result"]["qa_reports"] == [job["report"]] and job["result"]["qa_mode"] == "quick-scan"
    assert [os.path.normcase(p) for p in job["result"]["qa_folders"]] == [os.path.normcase(folder)]


def test_e2e_scan_marks_qa_flags_like_a_qa_job(child):
    """The Progress Manager / Library QA flags: the chapter the model left partly in Korean is qa_failed."""
    job = child["quick"]
    chapter = str(child["raw_chapter"])
    assert job["chapters"][chapter] == "qa_failed", job["chapters"]
    assert all(status == "completed" for n, status in job["chapters"].items() if n != chapter), job["chapters"]
    assert job["progress_failed"] >= 1  # the job's "N QA failed" chip
    assert any("korean_text_found" in line.lower() for line in job["row_qa_lines"]), job["row_qa_lines"]
    # a qa_scan job of the same workspace reports the same way
    qa_job = child["qa_job"]
    assert qa_job["state"] == "DONE", (qa_job["error"], _lines(qa_job)[-20:])
    assert set(qa_job["result_keys"]) <= set(job["result"]) and qa_job["report"] == job["report"]
    assert qa_job["chapters"] == job["chapters"]


def test_e2e_scan_follows_the_scan_phase_mode(child):
    job = child["aggressive"]
    assert job["state"] == "DONE", (job["error"], _lines(job)[-20:])
    assert _has_line(job, TRIGGER_LINE + " aggressive mode...")
    assert _has_line(job, "🔍 Starting QA scan in AGGRESSIVE mode for folder: ")
    assert [s["mode"] for s in job["scans"]] == ["aggressive"] and job["result"]["qa_mode"] == "aggressive"


def test_e2e_stop_during_the_scan_stops_the_job(child):
    job = child["stopped"]
    assert job["state"] == "CANCELLED" and job["state_label"] == "Stopped", (job["state"], _lines(job)[-20:])
    assert job["scans"] and job["scans"][0]["stop_seen"], job["scans"]
    assert _has_line(job, "⛔ ") or _has_line(job, "⚠️ Bulk scan stopped"), _lines(job)[-20:]
    assert job["secs_after_stop"] < 30


def test_e2e_scanning_phase_off_scans_nothing(child):
    job = child["off"]
    assert job["state"] == "DONE", (job["error"], _lines(job)[-20:])
    assert job["scans"] == [] and not _has_line(job, TRIGGER_LINE)
    assert "✅ QA scanner initialized successfully" not in _lines(job) and UNSHARED_LINE not in _lines(job)
    assert "qa_reports" not in job["result"]


def test_e2e_job_stays_in_its_sandbox(child):
    """Nothing written outside the sandbox, no process spawned (the scan runs on threads on mobile); every job
    restored the process state (``E2ESession.run_job`` fails the child otherwise)."""
    assert child["writes_outside"] == [], child["writes_outside"]
    assert child["spawns"] == [], child["spawns"]


# ---- the child process ------------------------------------------------------------------------------------


def _child_scenario(tmp_root: Path) -> dict:
    from glossarion_mobile.diagnostics import e2e, library_check
    from glossarion_mobile.diagnostics.e2e import E2ESession
    from glossarion_mobile.services.jobs import JobSpec
    from glossarion_mobile.ui.library import progress_model as pm

    work = tmp_root / "work"
    paths = types.SimpleNamespace(assets_dir=str(APP_DIR / "assets"), temp=str(work),
                                  writable_dirs=lambda: {"tmp": str(tmp_root)}, backend_dir=str(SRC_DIR))
    session = E2ESession(paths, root=str(work / "e2e"), keep=False, isolated=True, log=lambda line: None)
    out: dict = {"error": None, "raw_chapter": E2ESession.RAW_QA_CHAPTER}
    try:
        session.setup()
        service = session.service
        import qa_scan_runtime

        real_scan_path = qa_scan_runtime.run_qa_scan_path
        scans: list = []
        stop_plan = {"armed": False, "job": None, "stopped_at": None}
        threads = {}

        def spy(folder_path, *args, **kwargs):  # records what the shared loop hands the scanner, runs it
            settings = kwargs.get("qa_settings") or {}
            entry = {"folder": str(folder_path), "mode": kwargs.get("mode"), "epub_path": kwargs.get("epub_path"),
                     "sample": settings.get("quick_scan_sample_size"), "thread": threading.current_thread().name}
            scans.append(entry)
            stop = kwargs.get("stop_flag")
            if stop_plan["armed"]:  # the user taps Stop while the scan starts
                worker = threading.Thread(target=lambda: service.request_stop(stop_plan["job"]), name="tap-stop")
                worker.start()
                worker.join(30)
                stop_plan["stopped_at"] = time.monotonic()
                deadline = time.monotonic() + 30
                while stop is not None and not stop() and time.monotonic() < deadline:
                    time.sleep(0.05)
                entry["stop_seen"] = bool(stop is not None and stop())
            return real_scan_path(folder_path, *args, **kwargs)

        qa_scan_runtime.run_qa_scan_path = spy
        name = "hotfix1-post-qa.epub"
        path = session.import_epub(name)

        def record(label: str, spec, *, during=None) -> dict:
            mark = len(scans)
            job_ids: list = []

            def started(job_id):
                job_ids.append(job_id)
                stop_plan["job"] = job_id
                if during is not None:
                    during(job_id)

            seed = e2e._seed_langdetect(0)
            try:
                outcome = session.run_job(service, spec, label, during=started)
            finally:
                e2e._seed_langdetect(seed)
            snap = service.snapshot(outcome.job_id)
            log = service.read_log_tail(outcome.job_id, 20000)
            output_dir = outcome.output_dir or (spec.inputs[0] if spec.kind == "qa_scan" else "")
            report = os.path.join(output_dir, os.path.basename(output_dir) + "_Scan Report", "validation_results.html")
            job_thread = threads.get(outcome.job_id, "")
            return {"state": snap.state.value, "state_label": snap.state_label, "error": snap.error,
                    "outputs": list(snap.outputs), "output_dir": output_dir, "result": dict(snap.result or {}),
                    "log": log, "scans": scans[mark:], "report": report, "report_exists": os.path.isfile(report),
                    "chapters": {str(k): v for k, v in e2e._progress_chapters(output_dir).items()},
                    "progress_failed": int(getattr(snap.progress, "failed", 0) or 0), "job_thread": job_thread,
                    "stopped_at": stop_plan["stopped_at"], "finished": time.monotonic()}

        # the job thread's name, for "the scan runs inside the job"
        original_execute = service._execute

        def execute(job):
            threads[job.id] = threading.current_thread().name
            return original_execute(job)

        service._execute = execute

        # 1. scanning phase on, Quick Scan; the fake model leaves part of one chapter in Korean
        session.configure("off", scan_phase_enabled=True, scan_phase_mode="quick-scan")
        session.server.leave_raw_once[E2ESession.RAW_QA_CHAPTER] = E2ESession.RAW_LEFTOVER
        quick = record("translate + post-translation QA", session.translate_spec(path, name))
        with library_check.isolated_library(session.root, session.store.snapshot()) as library:
            book = session._library_book(library, quick["output_dir"])
            view = pm.load_progress_view(library, book, show_special=False, show_model_info=True, full=True)
            row = session._chapter_rows(view).get(E2ESession.RAW_QA_CHAPTER)
            quick["row_qa_lines"] = list(getattr(row, "qa_lines", None) or [])
        out["quick"] = quick

        # 2. a qa_scan job of the same workspace (what a QA job leaves)
        qa_spec = JobSpec(kind="qa_scan", title=os.path.basename(quick["output_dir"]), inputs=(quick["output_dir"],),
                          params={"mode": "quick-scan", "targets": [{"folder": quick["output_dir"], "source": path}]},
                          origin={"type": "e2e", "label": "E2E"})
        qa_job = record("QA quick scan", qa_spec)
        qa_job["result_keys"] = sorted(qa_job["result"])
        out["qa_job"] = qa_job

        # 3. Aggressive mode from scan_phase_mode
        session.configure("off", scan_phase_enabled=True, scan_phase_mode="aggressive")
        out["aggressive"] = record("translate + post-translation QA (aggressive)", session.translate_spec(path, name))

        # 4. Stop during the scan
        session.configure("off", scan_phase_enabled=True, scan_phase_mode="quick-scan")
        stop_plan["armed"] = True
        try:
            stopped = record("translate + post-translation QA (Stop)", session.translate_spec(path, name))
        finally:
            stop_plan["armed"] = False
        stopped["secs_after_stop"] = (stopped["finished"] - stopped["stopped_at"]) if stopped["stopped_at"] else 999.0
        out["stopped"] = stopped

        # 5. scanning phase off
        session.configure("off", scan_phase_enabled=False, scan_phase_mode="quick-scan")
        out["off"] = record("translate (scanning phase off)", session.translate_spec(path, name))
        out["writes_outside"] = [v.get("path") for v in (session.audit.violations if session.audit else [])][:10]
        out["spawns"] = [f"{e['api']} on {e['thread']} ({' < '.join(e['stack'][-3:])})"
                         for e in (session.guards.spawns if session.guards else []) if not e.get("stdlib")][:10]
    except BaseException as exc:  # the parent shows it with the child's output
        import traceback

        session.failed = True
        out["error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
    finally:
        session.close()
    return out


def _child_main(argv: list) -> int:
    result_file, tmp_root = Path(argv[0]), Path(argv[1])
    result = _child_scenario(tmp_root)
    result_file.write_text(json.dumps(result, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return 0 if result.get("error") is None else 1


if __name__ == "__main__" and CHILD in sys.argv:
    sys.exit(_child_main(sys.argv[sys.argv.index(CHILD) + 1:]))
