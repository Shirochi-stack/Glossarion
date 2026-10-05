"""Mobile-compat patches in the core translation/glossary backend (plan §3, U1).

Covers build-ci patches P1-P6, P12, the TransateKRtoEN parts of P23, the
``../logs`` glossary log, and the critique corrections (async extraction gated
at its consumers, auto-glossary in-thread, glossary_process_worker
``capture_stdio``/``isolate_env``, ``generate_glossary_async``).

Every patch is gated on ``mobile_runtime``. With the GLOSSARION_* variables
unset the desktop branch (process pools, exe/script-relative paths) must be
the one that runs; with ``GLOSSARION_NO_PROCESSES=1`` the thread/in-process
branch must run and no process API may be touched.
"""
import ast
import concurrent.futures
import functools
import importlib.util
import logging
import multiprocessing
import os
import sys
import threading
import types
import zipfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import mobile_runtime  # noqa: E402

PREPARE_ASSETS = SRC / "mobile" / "tools" / "prepare_assets.py"
PATCHED_FILES = (
    "TransateKRtoEN.py",
    "extract_glossary_from_epub.py",
    "glossary_process_worker.py",
    "Chapter_Extractor.py",
    "GlossaryManager.py",
    "unified_glossary.py",
)
_MOBILE_ENV = ("GLOSSARION_MOBILE", "GLOSSARION_NO_PROCESSES", "GLOSSARION_DATA_DIR", "FLET_PLATFORM")


@pytest.fixture
def desktop_env(monkeypatch):
    """The desktop never sets any mobile variable."""
    for name in _MOBILE_ENV:
        monkeypatch.delenv(name, raising=False)
    assert mobile_runtime.processes_available()
    return monkeypatch


@pytest.fixture
def no_processes_env(desktop_env):
    desktop_env.setenv("GLOSSARION_NO_PROCESSES", "1")
    assert not mobile_runtime.processes_available()
    return desktop_env


class _Boom(AssertionError):
    pass


def _forbidden(what):
    def _raise(*_args, **_kwargs):
        raise _Boom(f"{what} must not be used when processes are unavailable")
    return _raise


# --------------------------------------------------------------------------- AST helpers

@functools.lru_cache(maxsize=None)
def _tree(name):
    """Parsed (read-only) module with ``_parent`` links; feature_version keeps 3.10 compatibility."""
    src = (SRC / name).read_bytes().decode("utf-8-sig").replace("\r\n", "\n")
    tree = ast.parse(src, filename=name, feature_version=(3, 10))
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child._parent = node
    return tree


def _parents(node):
    while getattr(node, "_parent", None) is not None:
        node = node._parent
        yield node


def _is_gate_call(node):
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in ("processes_available", "subprocesses_available")
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "mobile_runtime"
    )


def _has_gate(node):
    return any(_is_gate_call(n) for n in ast.walk(node))


def _enclosing_function(node):
    for parent in _parents(node):
        if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return parent
    return None


def _function(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"function {name} not found")


def _gated_names(func):
    """Names assigned (at least once) from an expression that calls the gate."""
    names = set()
    for node in ast.walk(func):
        if isinstance(node, ast.Assign) and _has_gate(node.value):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
    return names


def _test_is_gated(test, gated_names):
    if _has_gate(test):
        return True
    return any(isinstance(n, ast.Name) and n.id in gated_names for n in ast.walk(test))


def _always_exits(stmts):
    if not stmts:
        return False
    last = stmts[-1]
    if isinstance(last, (ast.Return, ast.Raise)):
        return True
    if isinstance(last, (ast.With, ast.AsyncWith)):
        return _always_exits(last.body)
    if isinstance(last, ast.If):
        return _always_exits(last.body) and _always_exits(last.orelse)
    return False


def _branch_gated(node, func):
    """True when ``node`` only runs on a branch chosen by the mobile_runtime gate."""
    gated_names = _gated_names(func) if func is not None else set()
    child = node
    for parent in _parents(node):
        if isinstance(parent, ast.If) and child is not parent.test and _test_is_gated(parent.test, gated_names):
            return True
        # Early exit: ``if not processes_available(): ...; return`` before this statement.
        for field in ("body", "orelse", "finalbody"):
            block = getattr(parent, field, None)
            if isinstance(block, list) and child in block:
                for prev in block[:block.index(child)]:
                    if isinstance(prev, ast.If) and _has_gate(prev.test) and _always_exits(prev.body):
                        return True
        if parent is func:
            break
        child = parent
    return False


def _spawn_sites(tree):
    """Process-creating calls / references (ProcessPoolExecutor, get_context, Manager, subprocess)."""
    sites = []
    for node in ast.walk(tree):
        target = None
        if isinstance(node, ast.Call):
            target = node.func
        elif isinstance(node, ast.Assign):
            target = node.value
        if isinstance(target, ast.Name) and target.id == "ProcessPoolExecutor":
            sites.append(node)
        elif isinstance(target, ast.Attribute):
            owner = target.value.id if isinstance(target.value, ast.Name) else None
            if target.attr == "ProcessPoolExecutor":
                sites.append(node)
            elif owner == "multiprocessing" and target.attr in ("get_context", "Manager", "Pool", "Process"):
                sites.append(node)
            elif owner == "subprocess" and target.attr in ("Popen", "run", "call", "check_call", "check_output"):
                sites.append(node)
    return sites


def _call_sites_gated(tree, func_name):
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == func_name]
    return bool(calls) and all(_branch_gated(c, _enclosing_function(c)) for c in calls)


def _is_use_async_read(node):
    if not (isinstance(node, ast.Call) and node.args):
        return False
    first = node.args[0]
    if not (isinstance(first, ast.Constant) and first.value == "USE_ASYNC_CHAPTER_EXTRACTION"):
        return False
    func = node.func
    return isinstance(func, ast.Attribute) and func.attr in ("getenv", "get")


# --------------------------------------------------------------------------- static checks

@pytest.mark.parametrize("name", PATCHED_FILES)
def test_patched_modules_stay_python_310_compatible_and_import_the_gate(name):
    tree = _tree(name)
    imports = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    assert "mobile_runtime" in imports


@pytest.mark.parametrize("name", PATCHED_FILES)
def test_every_process_spawn_site_is_behind_the_gate(name):
    tree = _tree(name)
    sites = _spawn_sites(tree)
    ungated = []
    for site in sites:
        func = _enclosing_function(site)
        if _branch_gated(site, func):
            continue
        if func is not None and _call_sites_gated(tree, func.name):
            continue
        ungated.append((getattr(func, "name", "<module>"), site.lineno))
    assert not ungated, f"{name}: ungated process spawn sites {ungated}"


def test_spawn_site_inventory_matches_the_patch_table():
    counts = {name: len(_spawn_sites(_tree(name))) for name in PATCHED_FILES}
    # Desktop branches are kept verbatim, so every original spawn site is still there.
    assert counts["Chapter_Extractor.py"] == 1
    assert counts["GlossaryManager.py"] == 3          # spawn Pool, filter pool, scoring executor_cls
    assert counts["TransateKRtoEN.py"] == 3           # PDF Manager + pool, auto-glossary pool
    assert counts["glossary_process_worker.py"] == 1
    assert counts["unified_glossary.py"] == 1
    assert counts["extract_glossary_from_epub.py"] == 0


@pytest.mark.parametrize("name, expected", [("TransateKRtoEN.py", 3), ("extract_glossary_from_epub.py", 1)])
def test_async_chapter_extraction_is_gated_at_every_consumer(name, expected):
    """The GUI writes USE_ASYNC_CHAPTER_EXTRACTION=1, so env defaults cannot stop it (critique)."""
    tree = _tree(name)
    reads = [n for n in ast.walk(tree) if _is_use_async_read(n)]
    assert len(reads) == expected
    for read in reads:
        compare = read._parent
        assert isinstance(compare, ast.Compare)
        # The desktop comparison itself is unchanged.
        assert isinstance(compare.ops[0], ast.Eq) and compare.comparators[0].value == "1"
        assert read.args[1].value == "0"
        bool_op = compare._parent
        assert isinstance(bool_op, ast.BoolOp) and isinstance(bool_op.op, ast.And), read.lineno
        assert bool_op.values[0] is compare
        assert any(_is_gate_call(v) for v in bool_op.values), f"{name}:{read.lineno} not gated"


def test_auto_glossary_runs_in_thread_without_processes():
    func = _function(_tree("TransateKRtoEN.py"), "main")
    assigns = [n for n in ast.walk(func) if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "_glossary_in_thread" for t in n.targets)]
    assert len(assigns) == 1
    value = assigns[0].value
    assert isinstance(value, ast.UnaryOp) and isinstance(value.op, ast.Not) and _is_gate_call(value.operand)

    branches = [n for n in ast.walk(func) if isinstance(n, ast.If)
                and isinstance(n.test, ast.Name) and n.test.id == "_glossary_in_thread"]
    assert len(branches) == 3
    log_branch, executor_branch, submit_branch = sorted(branches, key=lambda n: n.lineno)

    # ../logs file only on the desktop branch; in-thread there is no log file to tail.
    assert ast.unparse(log_branch.body[0]) == "glossary_log_fp = None"
    desktop_log = "\n".join(ast.unparse(s) for s in log_branch.orelse)
    assert "os.path.join(_project_root, 'logs')" in desktop_log
    assert "glossary_subprocess_" in desktop_log

    mobile_exec = ast.unparse(executor_branch.body[-1])
    assert mobile_exec.startswith("_glossary_executor = mobile_runtime.make_pool_executor(1,")
    assert ast.unparse(executor_branch.orelse[-1]) == (
        "_glossary_executor = concurrent.futures.ProcessPoolExecutor(max_workers=1)")

    mobile_submit = submit_branch.body[0].value
    assert ast.unparse(mobile_submit.func) == "executor.submit"
    assert ast.unparse(mobile_submit.args[0]) == "generate_glossary_in_process"
    assert {k.arg: k.value.value for k in mobile_submit.keywords} == {"capture_stdio": False, "isolate_env": True}
    assert [ast.unparse(a) for a in mobile_submit.args[5:]] == ["None", "None"]
    desktop_submit = submit_branch.orelse[0].value
    assert [ast.unparse(a) for a in desktop_submit.args] == [
        "generate_glossary_in_process", "out", "worker_chapters", "instructions", "env_vars",
        "log_queue", "glossary_log_fp"]
    assert desktop_submit.keywords == []

    seen = sorted((n for n in ast.walk(func) if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "_seen_worker_output" for t in n.targets)),
                  key=lambda n: n.lineno)
    # Initial value (no "waiting for glossary subprocess" notices in-thread), then the tail loop's True.
    assert [ast.unparse(n.value) for n in seen] == ["_glossary_in_thread", "True"]


def test_glossary_manager_pools_follow_the_gate():
    func = _function(_tree("GlossaryManager.py"), "_filter_text_for_glossary")
    assigned = {}
    for node in sorted((n for n in ast.walk(func) if isinstance(n, ast.Assign)), key=lambda n: n.lineno):
        for target in node.targets:
            if isinstance(target, ast.Name):
                assigned.setdefault(target.id, []).append(node.value)
    assert [ast.unparse(v) for v in assigned["use_process_pool"][1:]] == ["False"]  # daemonic fallback
    first = assigned["use_process_pool"][0]
    assert ast.unparse(first) == "len(sentences) > 5000 and mobile_runtime.processes_available()"
    filtering = assigned["use_process_pool_filtering"]
    assert len(filtering) == 1
    assert ast.unparse(filtering[0]) == (
        "not in_subprocess and len(check_batches) > 3 and mobile_runtime.processes_available()")
    # The scoring pool picks its class from the same gated flag.
    picks = [n for n in ast.walk(func) if isinstance(n, ast.If)
             and any(isinstance(s, ast.Assign) and ast.unparse(s.value) == "ProcessPoolExecutor" for s in n.body)]
    assert len(picks) == 1 and ast.unparse(picks[0].test) == "use_process_pool"
    assert ast.unparse(picks[0].orelse[0].value) == "ThreadPoolExecutor"


def test_chapter_extractor_keeps_the_desktop_pool_expression():
    func = _function(_tree("Chapter_Extractor.py"), "_extract_chapters_universal")
    gate_ifs = [n for n in ast.walk(func) if isinstance(n, ast.If) and _is_gate_call(n.test)]
    assert len(gate_ifs) == 1
    branch = gate_ifs[0]
    assert ast.unparse(branch.body[0]) == "_extraction_executor = ProcessPoolExecutor(max_workers=max_workers)"
    assert "mobile_runtime.make_pool_executor(1," in ast.unparse(branch.orelse[0])


# --------------------------------------------------------------------------- Chapter_Extractor (P1)

@pytest.fixture(scope="module")
def selftest_epub(tmp_path_factory):
    """The 12-chapter Korean self-test EPUB, built by src/mobile/tools/prepare_assets.py.

    app/assets/selftest/ is generated and gitignored (absent on CI), so build a fresh
    copy with the same builder into a temp dir instead of reading the app's assets.
    """
    pytest.importorskip("ebooklib")
    spec = importlib.util.spec_from_file_location("_mobile_prepare_assets", PREPARE_ASSETS)
    prepare_assets = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prepare_assets)
    path = tmp_path_factory.mktemp("selftest_assets") / prepare_assets.SELFTEST_EPUB
    prepare_assets.build_selftest_epub(path)
    return path


def _extract_selftest(epub_path, out_dir, monkeypatch):
    import Chapter_Extractor as CE

    monkeypatch.setenv("EXTRACTION_WORKERS", "2")
    monkeypatch.setenv("EXTRACTION_MODE", "smart")
    monkeypatch.delenv("SINGLE_CHAPTER_FILTER", raising=False)
    with zipfile.ZipFile(epub_path) as zf:
        html_files = [n for n in zf.namelist() if n.endswith(".xhtml")]
        assert len(html_files) == 13  # > 10 files -> Chapter_Extractor's pool path
        chapters = CE.extract_chapters(zf, str(out_dir))
    return chapters


def _normalised(chapters, out_dir):
    def norm(value):
        if isinstance(value, str):
            return value.replace(str(out_dir), "<OUT>")
        if isinstance(value, dict):
            return {k: norm(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [norm(v) for v in value]
        return value
    return norm(chapters)


def _spy_make_pool_executor(monkeypatch):
    made = []
    real = mobile_runtime.make_pool_executor

    def spy(*args, **kwargs):
        executor = real(*args, **kwargs)
        made.append((args, kwargs, executor))
        return executor

    monkeypatch.setattr(mobile_runtime, "make_pool_executor", spy)
    return made


def test_chapter_extraction_pool_path_runs_on_threads_without_processes(
    tmp_path, selftest_epub, no_processes_env, capsys
):
    import Chapter_Extractor as CE

    no_processes_env.setattr(CE, "ProcessPoolExecutor", _forbidden("ProcessPoolExecutor"))
    made = _spy_make_pool_executor(no_processes_env)

    chapters = _extract_selftest(selftest_epub, tmp_path / "out", no_processes_env)

    out = capsys.readouterr().out
    assert "Using parallel processing with 2 workers" in out
    assert len(made) == 1
    args, kwargs, executor = made[0]
    assert args == (1,) and kwargs == {"thread_name_prefix": "chapter-extract"}
    assert isinstance(executor, ThreadPoolExecutor)
    assert len(chapters) == 13
    names = {ch.get("original_basename") for ch in chapters}
    assert names == {"nav"} | {f"chapter{i:04d}" for i in range(1, 13)}
    assert all(ch.get("body") for ch in chapters)


def test_chapter_extraction_uses_a_real_process_pool_on_desktop(tmp_path, selftest_epub, desktop_env):
    """Desktop: the original ProcessPoolExecutor(max_workers=EXTRACTION_WORKERS) runs, same result as threads."""
    import Chapter_Extractor as CE

    used = []
    real_ppe = CE.ProcessPoolExecutor

    class RecordingProcessPool(real_ppe):
        def __init__(self, *args, **kwargs):
            used.append((args, kwargs))
            super().__init__(*args, **kwargs)

    desktop_env.setattr(CE, "ProcessPoolExecutor", RecordingProcessPool)
    made = _spy_make_pool_executor(desktop_env)
    desktop = _extract_selftest(selftest_epub, tmp_path / "desktop", desktop_env)
    assert used == [((), {"max_workers": 2})]
    assert made == []

    desktop_env.setenv("GLOSSARION_NO_PROCESSES", "1")
    desktop_env.setattr(CE, "ProcessPoolExecutor", _forbidden("ProcessPoolExecutor"))
    threaded = _extract_selftest(selftest_epub, tmp_path / "threads", desktop_env)
    assert _normalised(threaded, tmp_path / "threads") == _normalised(desktop, tmp_path / "desktop")


# --------------------------------------------------------------------------- GlossaryManager (P5/P6)

def _synthetic_korean_text():
    first, mid, last = "김이박최정강조윤장임한오서신권황안송류홍", "서민지현수영하도준우예은시유채윤다", "연호우진아빈희율원린경석훈람솔"
    names = [a + b + c for a in first for b in mid for c in last][:1600]
    lines = []
    for i in range(7000):
        lines.append(f"{names[i % len(names)]}님은 {names[(i * 7 + 3) % len(names)]}에게 말했다. 그는 웃었다.")
    return "\n".join(lines)  # 14,001 sentences: above the 5000-sentence process-pool threshold


def test_glossary_filter_uses_threads_without_processes(tmp_path, no_processes_env, capsys):
    import GlossaryManager as GM

    no_processes_env.chdir(tmp_path)
    no_processes_env.setenv("EXTRACTION_WORKERS", "2")
    no_processes_env.setattr(multiprocessing, "get_context", _forbidden("multiprocessing.get_context"))
    no_processes_env.setattr(GM, "ProcessPoolExecutor", _forbidden("ProcessPoolExecutor"))

    result = GM._filter_text_for_glossary(_synthetic_korean_text(), min_frequency=2, max_sentences=50)

    out = capsys.readouterr().out
    assert "Found 14,001 sentences" in out
    assert "Using ThreadPoolExecutor for sentence processing" in out
    assert "Using ProcessPoolExecutor" not in out
    assert isinstance(result, tuple) and result[0]


def test_glossary_filter_still_picks_the_spawn_pool_on_desktop(tmp_path, desktop_env):
    import GlossaryManager as GM

    class _DesktopPathChosen(Exception):
        pass

    contexts = []

    def fake_get_context(method=None):
        contexts.append(method)
        raise _DesktopPathChosen()

    desktop_env.chdir(tmp_path)
    desktop_env.setenv("EXTRACTION_WORKERS", "2")
    desktop_env.setattr(multiprocessing, "get_context", fake_get_context)
    with pytest.raises(_DesktopPathChosen):
        GM._filter_text_for_glossary(_synthetic_korean_text(), min_frequency=2, max_sentences=50)
    assert contexts == ["spawn"]


# --------------------------------------------------------------------------- unified_glossary (P12)

def test_unified_dedupe_stays_in_process_without_processes(no_processes_env):
    import unified_glossary as ug

    no_processes_env.setenv("UNIFIED_GLOSSARY_SUBPROCESS_MIN_ENTRIES", "1")
    no_processes_env.setattr(ug, "_run_dedupe_in_subprocess", _forbidden("_run_dedupe_in_subprocess"))
    logs = []
    result = ug._run_dedupe("full", [{"type": "term", "raw_name": "마탑", "translated_name": "Mage Tower"}],
                            log=logs.append)
    assert [e["raw_name"] for e in result] == ["마탑"]
    assert not any("failed" in line for line in logs), logs


def test_unified_dedupe_uses_the_child_process_on_desktop(desktop_env):
    import unified_glossary as ug

    calls = []

    def fake_subprocess(kind, first, second, total, log):
        calls.append((kind, total))
        return ["from-child"]

    desktop_env.setenv("UNIFIED_GLOSSARY_SUBPROCESS_MIN_ENTRIES", "1")
    desktop_env.setattr(ug, "_run_dedupe_in_subprocess", fake_subprocess)
    result = ug._run_dedupe("full", [{"type": "term", "raw_name": "마탑", "translated_name": "Mage Tower"}])
    assert result == ["from-child"] and calls == [("full", 1)]


# --------------------------------------------------------------------------- TransateKRtoEN paths (P23)

def _spy_data_dir(monkeypatch):
    defaults = []
    real = mobile_runtime.data_dir

    def spy(default):
        defaults.append(default)
        return real(default)

    monkeypatch.setattr(mobile_runtime, "data_dir", spy)
    return defaults


def _font_css(tmp_path):
    css = tmp_path / "loaded.css"
    css.write_text("@font-face { font-family: X; src: url('../fonts/MobileOnly.ttf'); }", encoding="utf-8")
    return css


def test_custom_fonts_are_read_from_the_script_dir_on_desktop(tmp_path, desktop_env):
    import TransateKRtoEN as T

    defaults = _spy_data_dir(desktop_env)
    desktop_env.setenv("EPUB_CSS_OVERRIDE_PATH", str(_font_css(tmp_path)))
    out = tmp_path / "out"
    T.sync_loaded_css_and_fonts_to_output(str(out))
    assert defaults == [os.path.dirname(os.path.abspath(T.__file__))]
    assert not (out / "fonts" / "MobileOnly.ttf").exists()


def test_custom_fonts_are_read_from_the_data_dir_on_mobile(tmp_path, desktop_env):
    import TransateKRtoEN as T

    data = tmp_path / "data"
    (data / "custom_fonts").mkdir(parents=True)
    (data / "custom_fonts" / "MobileOnly.ttf").write_bytes(b"font-bytes")
    desktop_env.setenv("GLOSSARION_DATA_DIR", str(data))
    desktop_env.setenv("EPUB_CSS_OVERRIDE_PATH", str(_font_css(tmp_path)))
    out = tmp_path / "out"
    T.sync_loaded_css_and_fonts_to_output(str(out))
    assert (out / "fonts" / "MobileOnly.ttf").read_bytes() == b"font-bytes"


def test_postprocess_output_candidates_follow_the_data_dir(tmp_path, desktop_env):
    import TransateKRtoEN as T

    desktop_env.delenv("OUTPUT_DIRECTORY", raising=False)
    desktop_env.delenv("OUTPUT_DIR", raising=False)
    current = str(tmp_path / "current")
    script_dir = os.path.dirname(os.path.abspath(T.__file__))

    desktop = T._postprocess_output_candidates("book.epub", "book", current)
    assert desktop == [os.path.abspath(current), os.path.abspath(os.path.join(script_dir, "book"))]

    desktop_env.setenv("GLOSSARION_DATA_DIR", str(tmp_path / "data"))
    mobile = T._postprocess_output_candidates("book.epub", "book", current)
    assert mobile == [os.path.abspath(current), os.path.abspath(str(tmp_path / "data" / "book"))]


# --------------------------------------------------------------------------- glossary_process_worker (P4)

def _fake_glossary_manager(monkeypatch, behaviour):
    module = types.ModuleType("GlossaryManager")

    def save_glossary(output_dir, chapters, instructions, language="korean", log_callback=None):
        save_glossary.last_run_complete = True
        return behaviour(output_dir, chapters, instructions, log_callback)

    module.save_glossary = save_glossary
    monkeypatch.setitem(sys.modules, "GlossaryManager", module)
    return module


def test_in_thread_worker_leaves_stdio_logging_and_env_alone(tmp_path, monkeypatch, capsys):
    import glossary_process_worker as gpw

    monkeypatch.setenv("EPUB_PATH", "book.epub")
    monkeypatch.setenv("GLOSSARY_SAME", "same")
    monkeypatch.setenv("GLOSSARY_ADDED", "placeholder")
    monkeypatch.delenv("GLOSSARY_ADDED")
    seen = {}
    stdout_before, stderr_before = sys.stdout, sys.stderr
    root_handlers_before = list(logging.getLogger().handlers)
    api_logger = logging.getLogger("unified_api_client")
    api_state_before = (list(api_logger.handlers), api_logger.level, api_logger.propagate)

    def behaviour(output_dir, chapters, instructions, log_callback):
        seen["stdout"] = sys.stdout
        seen["stderr"] = sys.stderr
        seen["log_callback"] = log_callback
        seen["env"] = {k: os.environ.get(k) for k in ("EPUB_PATH", "GLOSSARY_SAME", "GLOSSARY_ADDED")}
        print("glossary worker says hi")
        return {"terms": 3}

    _fake_glossary_manager(monkeypatch, behaviour)
    log_file = tmp_path / "worker.log"
    result = gpw.generate_glossary_in_process(
        str(tmp_path), ["c1"], "", {"EPUB_PATH": "ocr_source.pdf", "GLOSSARY_SAME": "same", "GLOSSARY_ADDED": "1"},
        None, str(log_file), capture_stdio=False, isolate_env=True,
    )

    assert result["success"] is True and result["complete"] is True and result["result"] == {"terms": 3}
    assert seen["stdout"] is stdout_before and seen["stderr"] is stderr_before
    assert sys.stdout is stdout_before and sys.stderr is stderr_before
    assert seen["log_callback"] is None  # GlossaryManager.set_output_redirect would swap sys.stdout
    assert "glossary worker says hi" in capsys.readouterr().out
    assert list(logging.getLogger().handlers) == root_handlers_before
    assert (list(api_logger.handlers), api_logger.level, api_logger.propagate) == api_state_before
    # env_vars applied during the run, restored afterwards
    assert seen["env"] == {"EPUB_PATH": "ocr_source.pdf", "GLOSSARY_SAME": "same", "GLOSSARY_ADDED": "1"}
    assert os.environ["EPUB_PATH"] == "book.epub"
    assert os.environ["GLOSSARY_SAME"] == "same"
    assert "GLOSSARY_ADDED" not in os.environ
    assert "glossary worker says hi" not in (log_file.read_text(encoding="utf-8") if log_file.exists() else "")


def test_isolate_env_restores_after_a_failure(tmp_path, monkeypatch):
    import glossary_process_worker as gpw

    monkeypatch.setenv("EPUB_PATH", "book.epub")

    def behaviour(*_args):
        assert os.environ["EPUB_PATH"] == "ocr_source.pdf"
        raise RuntimeError("api down")

    _fake_glossary_manager(monkeypatch, behaviour)
    result = gpw.generate_glossary_in_process(str(tmp_path), [], "", {"EPUB_PATH": "ocr_source.pdf"},
                                              capture_stdio=False, isolate_env=True)
    assert result["success"] is False and "api down" in result["error"]
    assert os.environ["EPUB_PATH"] == "book.epub"


def test_worker_defaults_keep_the_subprocess_behaviour(tmp_path, monkeypatch):
    """Desktop call (positional args only): capture into the log file, leave env_vars applied."""
    import glossary_process_worker as gpw

    monkeypatch.setenv("EPUB_PATH", "book.epub")
    seen = {}
    stdout_before = sys.stdout
    api_logger = logging.getLogger("unified_api_client")
    api_state_before = (list(api_logger.handlers), api_logger.level, api_logger.propagate)
    root_handlers_before = list(logging.getLogger().handlers)

    def behaviour(output_dir, chapters, instructions, log_callback):
        seen["stdout"] = sys.stdout
        seen["log_callback"] = log_callback
        print("captured line")
        return {}

    _fake_glossary_manager(monkeypatch, behaviour)
    log_file = tmp_path / "glossary_subprocess.log"
    try:
        result = gpw.generate_glossary_in_process(str(tmp_path), [], "", {"EPUB_PATH": "ocr_source.pdf"},
                                                  None, str(log_file))
    finally:
        api_logger.handlers, api_logger.level, api_logger.propagate = (
            api_state_before[0], api_state_before[1], api_state_before[2])
        logging.getLogger().handlers = root_handlers_before

    assert result["success"] is True
    assert seen["stdout"] is not stdout_before and hasattr(seen["stdout"], "queue")
    assert callable(seen["log_callback"])
    assert sys.stdout is stdout_before
    text = log_file.read_text(encoding="utf-8")
    assert "[glossary-worker] started pid=" in text and "captured line" in text
    assert os.environ["EPUB_PATH"] == "ocr_source.pdf"  # a worker process keeps env_vars applied


def test_generate_glossary_async_runs_in_a_thread_without_processes(tmp_path, no_processes_env):
    import glossary_process_worker as gpw

    calls = []

    def fake_worker(*args, **kwargs):
        calls.append((args, kwargs, threading.current_thread().name))
        return {"success": True}

    no_processes_env.setattr(gpw, "generate_glossary_in_process", fake_worker)
    no_processes_env.setattr(concurrent.futures, "ProcessPoolExecutor", _forbidden("ProcessPoolExecutor"))
    future = gpw.generate_glossary_async(str(tmp_path), ["c"], "", extraction_workers=2)
    assert future.result(timeout=5) == {"success": True}
    (args, kwargs, thread_name), = calls
    assert args[:3] == (str(tmp_path), ["c"], "")
    assert kwargs == {"capture_stdio": False, "isolate_env": True}
    assert thread_name.startswith("glossary-worker")


def test_generate_glossary_async_uses_a_process_pool_on_desktop(tmp_path, desktop_env):
    import glossary_process_worker as gpw

    pools = []

    class FakeProcessPool:
        def __init__(self, **kwargs):
            pools.append(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def submit(self, fn, *args, **kwargs):
            future = concurrent.futures.Future()
            future.set_result(fn(*args, **kwargs))
            return future

    calls = []
    desktop_env.setattr(gpw, "generate_glossary_in_process", lambda *a, **k: calls.append((a, k)) or {"ok": 1})
    desktop_env.setattr(concurrent.futures, "ProcessPoolExecutor", FakeProcessPool)
    future = gpw.generate_glossary_async(str(tmp_path), ["c"], "", extraction_workers=2)
    assert future.result() == {"ok": 1}
    assert pools == [{"max_workers": 1}]
    (args, kwargs), = calls
    assert len(args) == 4 and kwargs == {}


def test_make_pool_executor_choice_follows_the_gate(desktop_env):
    with mobile_runtime.make_pool_executor(1) as executor:
        assert isinstance(executor, ProcessPoolExecutor)
    desktop_env.setenv("GLOSSARION_NO_PROCESSES", "1")
    with mobile_runtime.make_pool_executor(1, thread_name_prefix="glossary-worker") as executor:
        assert isinstance(executor, ThreadPoolExecutor)
