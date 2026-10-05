"""Mobile-compat patches in the QA scanner, PDF and EPUB I/O backend (plan §3, U1).

Covers build-ci patches P7-P11, P13-P16, P22, the epub_converter part of P23,
P30 wiring, and the critique additions: pdf_extractor's pdf2htmlEX
``subprocess.run``, epub_converter's ``_sp.Popen`` compression workers, and
the in-process PdfGenerationManager mode.

Every patch is gated on ``mobile_runtime``. With the GLOSSARION_* variables
unset the desktop branch (process pools, subprocesses, exe/script-relative
paths) must run; with ``GLOSSARION_NO_PROCESSES=1`` the thread / sequential /
in-process branch must run and no process API may be touched. Process pools
are replaced by recorders that run the work on threads, so the desktop branch
is observed without spawning interpreters.
"""
import ast
import concurrent.futures
import json
import os
import subprocess
import sys
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import mobile_runtime  # noqa: E402

PATCHED_FILES = (
    "scan_html_folder.py",
    "pdf_fast_extractor.py",
    "pdf_extractor.py",
    "pdf_workspace_compiler.py",
    "_pdf_worker.py",
    "pdf_generation_manager.py",
    "epub_converter.py",
    "pdf_mupdf_html.py",
)
_MOBILE_ENV = (
    "GLOSSARION_MOBILE",
    "GLOSSARION_NO_PROCESSES",
    "GLOSSARION_DATA_DIR",
    "GLOSSARION_PDF_ENGINE",
    "FLET_PLATFORM",
    "CONFIG_FILE",
)


@pytest.fixture
def desktop_env(monkeypatch):
    """The desktop never sets any mobile variable (nor CONFIG_FILE)."""
    for name in _MOBILE_ENV:
        monkeypatch.delenv(name, raising=False)
    assert mobile_runtime.processes_available()
    return monkeypatch


@pytest.fixture
def mobile_env(desktop_env):
    desktop_env.setenv("GLOSSARION_NO_PROCESSES", "1")
    assert not mobile_runtime.processes_available()
    assert not mobile_runtime.subprocesses_available()
    return desktop_env


@pytest.fixture(params=["desktop", "mobile"])
def gate(request, monkeypatch):
    for name in _MOBILE_ENV:
        monkeypatch.delenv(name, raising=False)
    if request.param == "mobile":
        monkeypatch.setenv("GLOSSARION_NO_PROCESSES", "1")
    return request.param


class PoolRecorder:
    """Stands in for ProcessPoolExecutor: records each construction, runs on threads."""

    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append(kwargs if not args else dict(kwargs, max_workers=args[0]))
        workers = kwargs.get("max_workers") or (args[0] if args else 1)
        return ThreadPoolExecutor(max_workers=workers)


@pytest.fixture
def process_pools(monkeypatch):
    """Route every ProcessPoolExecutor the patched modules can reach to one recorder."""
    recorder = PoolRecorder()
    monkeypatch.setattr(concurrent.futures, "ProcessPoolExecutor", recorder)
    monkeypatch.setattr(mobile_runtime, "ProcessPoolExecutor", recorder)
    import pdf_workspace_compiler

    monkeypatch.setattr(pdf_workspace_compiler, "ProcessPoolExecutor", recorder)
    return recorder


@pytest.fixture
def scan_config(monkeypatch, tmp_path):
    """Isolate the QA scanner from the real src/config.json (keeps the gate's env).

    Desktop ignores CONFIG_FILE in the scanner, so the scanner's config path
    helper is pointed at the fixture file directly.
    """
    import scan_html_folder as scanner

    config_path = tmp_path / "scan_config.json"
    config_path.write_text(json.dumps({
        "qa_scanner_settings": {"use_thread_executor": False},
        "qa_scanner_config": {"max_workers": 2},
    }), encoding="utf-8")
    monkeypatch.setattr(scanner, "_scan_config_json_path", lambda: str(config_path))
    monkeypatch.delenv("QA_USE_THREAD_EXECUTOR", raising=False)
    monkeypatch.delenv("AI_HUNTER_MAX_WORKERS", raising=False)
    return config_path


# --------------------------------------------------------------------------
# Hygiene
# --------------------------------------------------------------------------

def import_pdf_worker():
    """Import _pdf_worker without its worker-process stdout/stderr re-wrap.

    The re-wrap is skipped where subprocesses are unavailable; importing it
    under the desktop gate inside pytest would wrap (and later close) the
    capture streams.
    """
    if "_pdf_worker" not in sys.modules:
        previous = os.environ.get("GLOSSARION_NO_PROCESSES")
        os.environ["GLOSSARION_NO_PROCESSES"] = "1"
        try:
            import _pdf_worker  # noqa: F401
        finally:
            if previous is None:
                os.environ.pop("GLOSSARION_NO_PROCESSES", None)
            else:
                os.environ["GLOSSARION_NO_PROCESSES"] = previous
    return sys.modules["_pdf_worker"]


@pytest.mark.parametrize("name", PATCHED_FILES)
def test_patched_modules_stay_python_310_compatible(name):
    source = (SRC / name).read_text(encoding="utf-8")
    ast.parse(source, filename=name, feature_version=(3, 10))
    if name == "pdf_mupdf_html.py":
        assert "PySide6" not in source


@pytest.mark.parametrize(
    "worker", ["chapter_extraction_worker.py", "sdlxliff_extraction_worker.py", "_pdf_worker.py"]
)
def test_worker_scripts_put_their_dir_on_sys_path_before_project_imports(worker, tmp_path):
    """``python src/<worker>.py`` with PYTHONSAFEPATH=1 (``python -P``, 3.11+): the script dir is
    not on sys.path, so the worker's own bootstrap must run before ``import mobile_runtime``."""
    env = {k: v for k, v in os.environ.items() if k not in _MOBILE_ENV and k != "PYTHONPATH"}
    env.update(PYTHONSAFEPATH="1", PYTHONIOENCODING="utf-8")
    proc = subprocess.run(
        [sys.executable, str(SRC / worker)], cwd=str(tmp_path), env=env,
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300,
    )
    output = proc.stdout + proc.stderr
    assert "ModuleNotFoundError" not in output, output
    assert "Usage:" in output, output


def test_pdf_worker_import_keeps_host_streams_where_subprocesses_are_unavailable(mobile_env):
    import importlib

    saved = sys.modules.pop("_pdf_worker", None)
    stdout, stderr = sys.stdout, sys.stderr
    try:
        module = importlib.import_module("_pdf_worker")
        assert sys.stdout is stdout and sys.stderr is stderr
        assert callable(module.run_pdf_generation)
    finally:
        if saved is not None:
            sys.modules["_pdf_worker"] = saved


# --------------------------------------------------------------------------
# scan_html_folder: P22 config path, P7/P8 executors
# --------------------------------------------------------------------------

def test_scan_config_path_is_unchanged_on_desktop(desktop_env):
    import scan_html_folder as scanner

    expected = os.path.join(os.path.dirname(os.path.abspath(scanner.__file__)), "config.json")
    assert scanner._scan_config_json_path() == expected


def test_scan_config_ignores_config_file_and_data_dir_on_desktop(desktop_env, tmp_path):
    import scan_html_folder as scanner

    other_tool = tmp_path / "other_tool_config.json"
    other_tool.write_text(json.dumps({"refusal_patterns": ["zz other tool marker"]}), encoding="utf-8")
    desktop_env.setenv("CONFIG_FILE", str(other_tool))
    desktop_env.setenv("GLOSSARION_DATA_DIR", str(tmp_path / "data"))
    here = os.path.dirname(os.path.abspath(scanner.__file__))
    assert scanner._scan_config_json_path() == os.path.join(here, "config.json")
    assert scanner._config_json_in(here) == os.path.join(here, "config.json")
    assert "zz other tool marker" not in scanner._get_refusal_patterns_for_scan()


def test_scan_config_path_honours_config_file_and_data_dir_on_mobile(desktop_env, tmp_path):
    import scan_html_folder as scanner

    desktop_env.setenv("GLOSSARION_MOBILE", "1")
    config_path = tmp_path / "mobile_config.json"
    config_path.write_text(json.dumps({
        "refusal_patterns": ["zz custom refusal marker"],
        "refusal_pattern_length_limit": 5000,
        "qa_scanner_config": {"max_workers": 3},
    }), encoding="utf-8")
    desktop_env.delenv("AI_HUNTER_MAX_WORKERS", raising=False)

    data_dir = tmp_path / "data"
    desktop_env.setenv("GLOSSARION_DATA_DIR", str(data_dir))
    assert scanner._scan_config_json_path() == os.path.join(str(data_dir), "config.json")

    desktop_env.setenv("CONFIG_FILE", str(config_path))
    assert scanner._scan_config_json_path() == str(config_path)
    assert scanner._get_refusal_patterns_for_scan() == ["zz custom refusal marker"]
    assert scanner._resolve_scan_max_workers_config() == 3
    # 1,500 characters: over the 900 default limit, under the configured 5,000.
    text = ("Ordinary translated prose. " * 54) + " ZZ custom refusal marker."
    artifacts = scanner.detect_ai_artifacts(text)
    assert any(item["type"] == "ai_refusal_pattern" for item in artifacts)


def test_enhanced_duplicate_detection_executor_gate(gate, scan_config, monkeypatch):
    import scan_html_folder as scanner

    recorder = PoolRecorder()
    monkeypatch.setattr(scanner.concurrent.futures, "ProcessPoolExecutor", recorder)
    logs = []
    results = [
        {"filename": "chapter0001.html", "chapter_num": 1, "dup_text": "A" * 1000},
        {"filename": "chapter0002.html", "chapter_num": 2, "dup_text": "B" * 1000},
    ]
    config = types.SimpleNamespace(get_threshold=lambda _name: 0.85)

    scanner.enhance_duplicate_detection(results, {}, {}, config, logs.append)

    if gate == "desktop":
        assert recorder.calls and recorder.calls[0]["initializer"] is scanner._init_worker_process
        assert any("ProcessPoolExecutor enabled" in message for message in logs)
    else:
        assert recorder.calls == []
        assert any("ThreadPoolExecutor enabled" in message for message in logs)


def test_deep_similarity_executor_gate(gate, scan_config, process_pools):
    import scan_html_folder as scanner

    body = "The same long chapter body repeated for the deep check. " * 20
    results = [{"filename": f"chapter{i:04d}.html", "dup_text": body} for i in range(1, 7)]
    groups, confidence, logs = {}, {}, []

    scanner.perform_deep_similarity_check(results, groups, confidence, 0.85, logs.append, lambda: False)

    assert any("Deep similarity check complete" in message for message in logs), logs
    assert len(set(groups[r["filename"]] for r in results)) == 1
    if gate == "desktop":
        assert [call["max_workers"] for call in process_pools.calls] == [min(2, os.cpu_count() or 1)]
    else:
        assert process_pools.calls == []


def test_ai_hunter_executor_gate(gate, scan_config, process_pools, monkeypatch):
    import scan_html_folder as scanner

    batches = []

    def fake_batch(args):
        batches.append(len(args[0]))
        # parallel_ai_hunter_check divides by the elapsed time when it logs
        # its speed; an instant stub can make that zero.
        time.sleep(0.05)
        return []

    monkeypatch.setattr(scanner, "process_comparison_batch_fast", fake_batch)
    results = [
        {"filename": f"chapter{i}.html", "normalized_text": f"text {i}",
         "semantic_sig": {}, "structural_sig": {}}
        for i in range(1, 4)
    ]
    config = types.SimpleNamespace(get_threshold=lambda _name: 0.85)

    scanner.parallel_ai_hunter_check(results, {}, {}, config, lambda _msg: None, lambda: False)

    assert sum(batches) == 3
    if gate == "desktop":
        assert len(process_pools.calls) == 1
        assert process_pools.calls[0]["initializer"] is scanner._init_worker_process
    else:
        assert process_pools.calls == []


def test_quick_scan_file_pool_gate(gate, scan_config, process_pools, tmp_path):
    import scan_html_folder as scanner

    folder = tmp_path / "book"
    folder.mkdir()
    for number in range(1, 4):
        (folder / f"response_{number:03d}_chapter{number}.html").write_text(
            f"<html><body><h1>Chapter {number}</h1><p>"
            + ("An ordinary English sentence for the scanner. " * 40)
            + "</p></body></html>",
            encoding="utf-8",
        )
    logs = []

    scanner.scan_html_folder(
        str(folder), log=logs.append, stop_flag=lambda: False,
        mode="quick-scan", qa_settings={"use_thread_executor": False},
    )

    uses_threads = any("Using ThreadPoolExecutor" in message for message in logs)
    if gate == "desktop":
        assert not uses_threads
        assert process_pools.calls and process_pools.calls[0]["initializer"] is scanner._init_worker_process
    else:
        assert uses_threads
        assert process_pools.calls == []


# --------------------------------------------------------------------------
# PDF worker counts (P9) and process pools (P10, P11)
# --------------------------------------------------------------------------

def test_pdf_worker_counts_gate(gate, monkeypatch):
    import pdf_extractor
    import pdf_fast_extractor
    import pdf_workspace_compiler

    monkeypatch.setattr(os, "cpu_count", lambda: 8)
    monkeypatch.setenv("PDF_EXTRACTION_WORKERS", "4")
    monkeypatch.setenv("PDF_FAST_MAX_WORKERS", "8")
    expected = 4 if gate == "desktop" else 1
    assert pdf_fast_extractor.resolve_pdf_extraction_workers() == expected
    assert pdf_fast_extractor.resolve_pdf_extraction_workers("3", cpu_count=8) == (3 if gate == "desktop" else 1)
    assert pdf_fast_extractor._fast_pdf_worker_count(816, 68) == expected
    assert pdf_workspace_compiler._rapid_render_worker_count(6, 4) == expected
    assert pdf_extractor._pdf_extraction_worker_count() == expected


def _write_numbered_pdf(path, pages):
    fitz = pytest.importorskip("fitz")
    document = fitz.open()
    for number in range(1, pages + 1):
        page = document.new_page()
        page.insert_text((72, 72), f"Numbered page {number}")
    document.save(str(path))
    document.close()


def test_large_pdf_text_extraction_pool_gate(gate, process_pools, monkeypatch, tmp_path):
    import pdf_extractor

    monkeypatch.setenv("PDF_EXTRACTION_WORKERS", "2")
    monkeypatch.setattr(os, "cpu_count", lambda: 8)
    pdf_path = tmp_path / "sixty.pdf"
    _write_numbered_pdf(pdf_path, 60)

    text = pdf_extractor.extract_text_from_pdf(str(pdf_path))

    assert "Numbered page 1" in text and "Numbered page 60" in text
    assert text.index("Numbered page 59") < text.index("Numbered page 60")
    if gate == "desktop":
        assert [call["max_workers"] for call in process_pools.calls] == [2]
    else:
        assert process_pools.calls == []


def test_fast_pdf_extraction_runs_sequentially_without_processes(mobile_env, process_pools, tmp_path):
    # Desktop takes the page-range process pool for this input (covered by
    # test_pdf_fast_extractor's parallel pool test); mobile must stay in-process.
    from pdf_fast_extractor import extract_pdf_fast

    mobile_env.setenv("EXTRACTION_WORKERS", "2")
    mobile_env.setenv("PDF_EXTRACTION_WORKERS", "2")
    mobile_env.setenv("PDF_FAST_CHUNK_PAGES", "2")
    pdf_path = tmp_path / "nine.pdf"
    _write_numbered_pdf(pdf_path, 9)

    pages, _images = extract_pdf_fast(str(pdf_path), str(tmp_path / "out"),
                                      mode="fast_semantic", page_by_page=True)

    assert [number for number, _html in pages] == list(range(1, 10))
    assert "Numbered page 9" in pages[-1][1]
    assert process_pools.calls == []


def test_pdf2htmlex_spawn_gate(gate, monkeypatch, tmp_path):
    import pdf_extractor

    runs = []

    def fake_run(cmd, **kwargs):
        runs.append(cmd)
        raise subprocess.CalledProcessError(1, cmd, stderr=b"boom")

    monkeypatch.setattr(pdf_extractor.shutil, "which", lambda name: r"C:\fake\pdf2htmlEX.exe")
    monkeypatch.setattr(pdf_extractor.subprocess, "run", fake_run)

    if gate == "desktop":
        with pytest.raises(RuntimeError, match="pdf2htmlEX failed"):
            pdf_extractor._extract_with_pdf2htmlex("book.pdf", str(tmp_path))
        assert len(runs) == 1
    else:
        with pytest.raises(FileNotFoundError, match="cannot launch subprocesses"):
            pdf_extractor._extract_with_pdf2htmlex("book.pdf", str(tmp_path))
        assert runs == []


def test_rapid_workspace_renderer_pool_gate(gate, process_pools, monkeypatch, tmp_path):
    pytest.importorskip("fitz")
    import pdf_workspace_compiler

    monkeypatch.setenv("GLOSSARION_PDF_ENGINE", "mupdf")
    monkeypatch.setenv("PDF_EXTRACTION_WORKERS", "2")
    monkeypatch.setattr(os, "cpu_count", lambda: 8)
    jobs = [
        (0, '<html><body><div id="chapter-1"><h1>One</h1><p>a</p></div></body></html>',
         [("one.html", 1, "One")]),
        (1, '<html><body><div id="chapter-2"><h1>Two</h1><p>b</p></div></body></html>',
         [("two.html", 2, "Two")]),
    ]

    bundle = pdf_workspace_compiler.render_workspace_bookmarks_rapid(jobs, str(tmp_path))
    try:
        assert [result["pages"] for result in bundle["results"]] == [1, 1]
        assert all(os.path.isfile(result["path"]) for result in bundle["results"])
        assert bundle["results"][1]["anchor_pages"] == {"2": 0, "two.html": 0}
    finally:
        import shutil

        shutil.rmtree(bundle.get("temp_dir", ""), ignore_errors=True)
    if gate == "desktop":
        assert [call["max_workers"] for call in process_pools.calls] == [2]
    else:
        assert bundle["workers"] == 1
        assert process_pools.calls == []


def _two_chapter_pdf_config(tmp_path, env_vars):
    output_dir = tmp_path / "output"
    (output_dir / "images").mkdir(parents=True)
    (output_dir / "css").mkdir()
    for number in (1, 2):
        (output_dir / f"chapter-{number}.html").write_text(
            f"<html><body><h1>Heading {number}</h1><p>Body {number}</p></body></html>",
            encoding="utf-8",
        )
    config_path = tmp_path / "pdf-config.json"
    config_path.write_text(json.dumps({
        "output_dir": str(output_dir),
        "images_dir": str(output_dir / "images"),
        "css_dir": str(output_dir / "css"),
        "html_files": ["chapter-1.html", "chapter-2.html"],
        "chapter_titles_info": {
            "1": ["Chapter One", 1.0, "chapter-1.html"],
            "2": ["Chapter Two", 1.0, "chapter-2.html"],
        },
        "processed_images": {},
        "cover_file": None,
        "metadata": {"title": "Gate Fixture"},
        "env_vars": env_vars,
    }), encoding="utf-8")
    return config_path, output_dir


def test_pdf_worker_rapid_compiler_gate(gate, process_pools, monkeypatch, tmp_path):
    pytest.importorskip("fitz")
    _pdf_worker = import_pdf_worker()

    monkeypatch.setenv("GLOSSARION_PDF_ENGINE", "mupdf")
    monkeypatch.setattr(os, "cpu_count", lambda: 8)
    config_path, output_dir = _two_chapter_pdf_config(tmp_path, {
        "PDF_PAGE_NUMBERS": "0",
        "PDF_GENERATE_TOC": "0",
        "PDF_USE_RAPID_WORKSPACE_COMPILER": "1",
        "PDF_EXTRACTION_WORKERS": "2",
        "ENABLE_IMAGE_COMPRESSION": "0",
    })
    lines = []

    _pdf_worker.run_pdf_generation(str(config_path), emit=lines.append, should_stop=lambda: False)

    log = "\n".join(lines)
    assert '"success": true' in log.lower(), log
    assert len(list(output_dir.glob("*.pdf"))) == 1
    if gate == "desktop":
        assert "Rapid Workspace Compiler" in log
        assert [call["max_workers"] for call in process_pools.calls] == [2]
    else:
        assert "Rapid Workspace Compiler" not in log
        assert "standard sequential compiler" in log
        assert process_pools.calls == []


# --------------------------------------------------------------------------
# epub_converter: P23 custom fonts dir, P16 compression workers, _generate_pdf
# --------------------------------------------------------------------------

def test_custom_fonts_dir_follows_data_dir(desktop_env, tmp_path):
    import epub_converter

    compiler = object.__new__(epub_converter.EPUBCompiler)
    expected = os.path.join(os.path.dirname(os.path.abspath(epub_converter.__file__)), "custom_fonts")
    assert compiler._get_global_custom_fonts_dir() == expected
    desktop_env.setenv("GLOSSARION_DATA_DIR", str(tmp_path))
    assert compiler._get_global_custom_fonts_dir() == os.path.join(str(tmp_path), "custom_fonts")


def test_image_compression_worker_spawn_gate(gate, monkeypatch, tmp_path):
    from PIL import Image
    import epub_converter

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    for name, color in (("one.png", (200, 10, 10)), ("two.png", (10, 200, 10))):
        Image.new("RGB", (64, 48), color).save(images_dir / name)
    spawned = []

    def fake_popen(cmd, **kwargs):
        spawned.append(cmd)
        raise OSError("spawn blocked in test")

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setenv("EXTRACTION_WORKERS", "2")
    monkeypatch.setenv("EXCLUDE_COVER_COMPRESSION", "1")
    monkeypatch.setenv("EXCLUDE_GIF_COMPRESSION", "1")
    compiler = object.__new__(epub_converter.EPUBCompiler)
    compiler.images_dir = str(images_dir)
    logs = []
    compiler.log = logs.append
    compiler.is_stopped = lambda: False

    processed, cover = compiler._compress_images({"one.png": "one.png", "two.png": "two.png"}, None)

    assert cover is None
    assert set(processed) == {"one.png", "two.png"}
    assert all(name.endswith(".webp") for name in processed.values()), processed
    assert all((images_dir / name).is_file() for name in processed.values())
    assert any("falling back to sequential" in message for message in logs)
    assert len(spawned) == (2 if gate == "desktop" else 0)


def test_generate_pdf_uses_shim_when_selected(desktop_env, tmp_path):
    fitz = pytest.importorskip("fitz")
    import epub_converter

    desktop_env.setenv("GLOSSARION_PDF_ENGINE", "mupdf")
    desktop_env.setenv("PDF_GENERATE_TOC", "1")
    desktop_env.setenv("PDF_PAGE_NUMBERS", "1")
    desktop_env.setenv("ENABLE_IMAGE_COMPRESSION", "0")
    output_dir = tmp_path / "out"
    for sub in ("images", "css"):
        (output_dir / sub).mkdir(parents=True)
    html_files = []
    titles = {}
    for number in (1, 2, 3):
        name = f"chapter{number}.xhtml"
        html_files.append(name)
        titles[number] = (f"Chapter {number}", 1.0, name)
        (output_dir / name).write_text(
            f"<html><body><h1>Chapter {number}</h1><p>본문 {number}</p></body></html>",
            encoding="utf-8",
        )
    compiler = object.__new__(epub_converter.EPUBCompiler)
    compiler.output_dir = str(output_dir)
    compiler.images_dir = str(output_dir / "images")
    compiler.css_dir = str(output_dir / "css")
    logs = []
    compiler.log = logs.append
    compiler.is_stopped = lambda: False

    compiler._generate_pdf(html_files, titles, {}, None, {"title": "Shim Book"})

    assert any("mupdf-story" in message for message in logs), logs
    pdf_files = list(output_dir.glob("*.pdf"))
    assert len(pdf_files) == 1
    with fitz.open(pdf_files[0]) as document:
        titles_in_outline = [row[1] for row in document.get_toc()]
        assert titles_in_outline == ["Chapter 1", "Chapter 2", "Chapter 3"]
        assert document.page_count >= 4  # TOC page + one page per chapter


# --------------------------------------------------------------------------
# pdf_generation_manager: P14 in-process mode
# --------------------------------------------------------------------------

def test_pdf_generation_manager_mode_gate(gate, monkeypatch):
    import pdf_generation_manager

    chosen = []
    done = threading.Event()

    def fake_subprocess(self, config_path, completion_callback):
        chosen.append("subprocess")
        done.set()

    def fake_inprocess(self, config_path, completion_callback):
        chosen.append("inprocess")
        done.set()

    monkeypatch.setattr(pdf_generation_manager.PdfGenerationManager, "_run_pdf_subprocess", fake_subprocess)
    monkeypatch.setattr(pdf_generation_manager.PdfGenerationManager, "_run_pdf_inprocess", fake_inprocess)

    assert pdf_generation_manager.PdfGenerationManager(log_callback=lambda _m: None).generate_pdf_async("cfg.json")
    assert done.wait(10)
    assert chosen == (["subprocess"] if gate == "desktop" else ["inprocess"])


# Scripted worker output: every protocol record type, a multi-line traceback
# record, a plain line, an ignored "[...]" line, blank lines and the result.
_SCRIPTED_PROTOCOL = [
    "[PROGRESS] 📄 Generating PDF...",
    "[INFO] fixture info",
    "plain worker output line",
    "[DEBUG] bracketed lines without a known tag are ignored",
    "",
    "[PROGRESS]   [DEBUG] Traceback (most recent call last):\n  File \"x.py\", line 1, in <module>\nValueError: boom",
    "[ERROR] fixture error",
    '[RESULT] {"success": true, "pdf_path": "book.pdf", "file_size": 3, "elapsed": 0.5}',
]


def _run_manager(manager, config_path):
    finished = threading.Event()
    outcome = {}

    def completion(success, result):
        outcome["success"] = success
        outcome["result"] = result
        finished.set()

    assert manager.generate_pdf_async(str(config_path), completion_callback=completion)
    assert finished.wait(60)
    return outcome


def test_pdf_generation_manager_inprocess_parses_same_protocol_as_subprocess(desktop_env, tmp_path):
    _pdf_worker = import_pdf_worker()
    import pdf_generation_manager

    config_path = tmp_path / "scripted.json"
    config_path.write_text("{}", encoding="utf-8")
    script = tmp_path / "fake_pdf_worker.py"
    script.write_text(
        "import sys\n"
        "sys.stdout.reconfigure(encoding='utf-8')\n"
        f"for record in {_SCRIPTED_PROTOCOL!r}:\n"
        "    print(record, flush=True)\n",
        encoding="utf-8",
    )

    # Subprocess mode: the manager's real pipe reader, fed by a scripted worker.
    real_popen = subprocess.Popen
    commands = []

    def scripted_popen(cmd, **kwargs):
        commands.append(cmd)
        return real_popen([sys.executable, str(script), cmd[-1]], **kwargs)

    sub_logs = []
    desktop_env.setattr(pdf_generation_manager.subprocess, "Popen", scripted_popen)
    sub_outcome = _run_manager(pdf_generation_manager.PdfGenerationManager(log_callback=sub_logs.append), config_path)
    desktop_env.setattr(pdf_generation_manager.subprocess, "Popen", real_popen)
    assert commands and commands[0][-1] == str(config_path)

    # In-process mode: _pdf_worker.run_pdf_generation emits the same records.
    calls = []

    def scripted_generation(path, emit=None, should_stop=None):
        calls.append((path, should_stop()))
        for record in _SCRIPTED_PROTOCOL:
            emit(record)

    desktop_env.setenv("GLOSSARION_NO_PROCESSES", "1")
    desktop_env.setattr(_pdf_worker, "run_pdf_generation", scripted_generation)
    desktop_env.setattr(pdf_generation_manager.subprocess, "Popen", lambda *a, **k: pytest.fail("spawned"))
    in_logs = []
    in_outcome = _run_manager(pdf_generation_manager.PdfGenerationManager(log_callback=in_logs.append), config_path)

    assert calls == [(str(config_path), False)]
    starting = ("🚀 Starting PDF generation", "  ⏳ PDF subprocess starting")
    assert [m for m in in_logs if not m.startswith(starting)] == [m for m in sub_logs if not m.startswith(starting)]
    assert in_outcome == sub_outcome
    assert in_outcome["success"] is True and in_outcome["result"]["pdf_path"] == "book.pdf"
    assert "ℹ️ fixture info" in in_logs and "❌ fixture error" in in_logs
    assert "ValueError: boom" in in_logs


def test_pdf_generation_manager_inprocess_reports_worker_failures(mobile_env, monkeypatch, tmp_path):
    _pdf_worker = import_pdf_worker()
    import pdf_generation_manager

    def failing_generation(path, emit=None, should_stop=None):
        emit("[PROGRESS] about to fail")
        raise ValueError("render exploded")

    monkeypatch.setattr(_pdf_worker, "run_pdf_generation", failing_generation)
    logs = []
    outcome = _run_manager(pdf_generation_manager.PdfGenerationManager(log_callback=logs.append), tmp_path / "x.json")

    assert outcome["success"] is False
    assert outcome["result"]["error"] == "render exploded"
    assert "ValueError: render exploded" in outcome["result"]["traceback"]
    assert "❌ PDF generation failed: render exploded" in logs


def test_pdf_generation_manager_inprocess_stop_is_cooperative(mobile_env, monkeypatch, tmp_path):
    _pdf_worker = import_pdf_worker()
    import pdf_generation_manager

    started = threading.Event()
    release = threading.Event()

    def slow_generation(path, emit=None, should_stop=None):
        emit("[PROGRESS] batch 1")
        started.set()
        release.wait(10)
        if should_stop():
            raise _pdf_worker.PdfGenerationStopped("PDF generation stopped by user")
        emit('[RESULT] {"success": true}')

    monkeypatch.setattr(_pdf_worker, "run_pdf_generation", slow_generation)
    manager = pdf_generation_manager.PdfGenerationManager(log_callback=lambda _m: None)
    finished = threading.Event()
    outcome = {}

    def completion(success, result):
        outcome.update(success=success, result=result)
        finished.set()

    manager.generate_pdf_async(str(tmp_path / "x.json"), completion_callback=completion)
    assert started.wait(10)
    manager.stop()
    release.set()
    assert finished.wait(10)
    assert outcome == {"success": False, "result": {"success": False, "error": "PDF generation stopped by user"}}
    assert manager.is_running is False


def _slow_inprocess_generation(_pdf_worker, started, release):
    """Stand-in for run_pdf_generation: holds the in-process lock like the real one."""
    def generation(path, emit=None, should_stop=None):
        with _pdf_worker._INPROCESS_LOCK:
            emit("[PROGRESS] rendering batch 1")
            started.set()
            release.wait(30)
            if should_stop():
                raise _pdf_worker.PdfGenerationStopped("PDF generation stopped by user")
            emit('[RESULT] {"success": true}')
    return generation


def test_pdf_generation_manager_wait_blocks_until_the_inprocess_thread_is_done(mobile_env, monkeypatch, tmp_path):
    _pdf_worker = import_pdf_worker()
    import pdf_generation_manager

    started, release = threading.Event(), threading.Event()
    monkeypatch.setattr(_pdf_worker, "run_pdf_generation", _slow_inprocess_generation(_pdf_worker, started, release))
    manager = pdf_generation_manager.PdfGenerationManager(log_callback=lambda _m: None)
    assert manager.wait(timeout=0) is True  # nothing started yet
    manager.generate_pdf_async(str(tmp_path / "x.json"))
    assert started.wait(10)

    manager.stop()  # cooperative: returns at once, the render thread keeps the lock
    assert manager.is_running is True
    assert manager.wait(timeout=0.2) is False
    threading.Timer(0.3, release.set).start()
    assert manager.wait(timeout=10) is True
    assert manager.is_running is False and not manager._thread.is_alive()
    assert _pdf_worker._INPROCESS_LOCK.acquire(blocking=False)
    _pdf_worker._INPROCESS_LOCK.release()


def _epub_converter_pdf_wait_loop():
    """The ``while not _pdf_done.is_set():`` loop of EPUBCompiler.compile, compiled on its own."""
    import ast as _ast
    import epub_converter

    tree = _ast.parse(Path(epub_converter.__file__).read_text(encoding="utf-8"))
    loops = [
        node for node in _ast.walk(tree)
        if isinstance(node, _ast.While) and _ast.unparse(node.test) == "not _pdf_done.is_set()"
    ]
    assert len(loops) == 1, "PDF wait loop not found in EPUBCompiler.compile"
    return compile(_ast.Module(body=loops, type_ignores=[]), epub_converter.__file__, "exec")


def test_compile_does_not_return_while_the_inprocess_pdf_thread_is_alive(mobile_env, monkeypatch, tmp_path):
    _pdf_worker = import_pdf_worker()
    import pdf_generation_manager

    # Parse epub_converter (~10k lines) before the renderer starts: on a slow runner the
    # parse alone outlasted a fixed release timer, the render finished first and the
    # loop (correctly) never ran its stop branch.
    wait_loop = _epub_converter_pdf_wait_loop()
    started, release = threading.Event(), threading.Event()
    monkeypatch.setattr(_pdf_worker, "run_pdf_generation", _slow_inprocess_generation(_pdf_worker, started, release))
    manager = pdf_generation_manager.PdfGenerationManager(log_callback=lambda _m: None)
    real_stop, stops = manager.stop, []

    def stop():
        # The renderer reaches its next checkpoint only some time after the stop request,
        # so a loop that returns right after stop() leaves the render thread alive.
        stops.append(True)
        real_stop()
        threading.Timer(0.5, release.set).start()

    manager.stop = stop
    pdf_done = threading.Event()
    manager.generate_pdf_async(str(tmp_path / "x.json"), completion_callback=lambda _ok, _result: pdf_done.set())
    assert started.wait(10)
    logs = []
    compiler = types.SimpleNamespace(is_stopped=lambda: True, log=logs.append)

    exec(wait_loop, {
        "self": compiler, "_pdf_mgr": manager, "_pdf_done": pdf_done, "mobile_runtime": mobile_runtime,
    })

    assert stops == [True]
    assert release.is_set() and pdf_done.is_set()
    assert not manager._thread.is_alive() and manager.is_running is False
    assert _pdf_worker._INPROCESS_LOCK.acquire(blocking=False)
    _pdf_worker._INPROCESS_LOCK.release()
    assert "🛑 PDF generation stopped by user" in logs


def test_compile_stop_keeps_desktop_subprocess_path(desktop_env):
    calls = []

    class SubprocessManager:
        def stop(self):
            calls.append("stop")  # terminates the worker process

        def wait(self, timeout=None):
            calls.append("wait")
            return True

    compiler = types.SimpleNamespace(is_stopped=lambda: True, log=lambda _m: None)
    exec(_epub_converter_pdf_wait_loop(), {
        "self": compiler, "_pdf_mgr": SubprocessManager(), "_pdf_done": threading.Event(),
        "mobile_runtime": mobile_runtime,
    })
    assert calls == ["stop"]
