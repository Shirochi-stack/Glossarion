"""QA scans in child interpreters (owner 2026-11-01: "If it works. Replace any in thread process in the mobile app with
it."): ``services.interpreters`` and the scan on it (``job_kinds.qa._ScanChildren``).

MEASURED before this was written (Android emulator, 4 vCPU x86_64, the real scanner on real chapters, findings identical):
Quick 100 chapters 4 workers 13.7 s on threads, 7.1 s in interpreters, the loop's lateness p95 73 to 15 ms; Aggressive
80 chapters 4 workers 72.9 to 41.9 s. Local runs are Python 3.12 (``_xxsubinterpreters``), CI 3.13 (``_interpreters``).
The glossary's smart filter and Generate MD / TXT are at the end of this file (``job_kinds.glossary``, ``job_kinds.compile``).

The budgets compare numbers of the same run, so a slow runner does not fail them.
"""

from __future__ import annotations

import ast
import asyncio
import concurrent.futures
import ctypes
import importlib.util
import json
import os
import shutil
import statistics
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = MOBILE_DIR / "app"
SRC_DIR = MOBILE_DIR.parent
for _entry in (str(APP_DIR), str(SRC_DIR), str(Path(__file__).parent)):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from glossarion_mobile.services import interpreters  # noqa: E402

needs_interpreters = pytest.mark.skipif(not interpreters.available(), reason="this Python has no own-GIL interpreters")
WORKERS = "interp_workers"


def _load(path: Path, alias: str):
    spec = importlib.util.spec_from_file_location(alias, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(alias, module)
    spec.loader.exec_module(module)
    return module


bench = _load(MOBILE_DIR / "tools" / "qa_scan_bench.py", "_glossarion_qa_scan_bench_interp")


def _backend_error() -> str:
    try:
        import qa_scan_runtime  # noqa: F401
        import scan_html_folder  # noqa: F401
    except Exception as exc:  # pragma: no cover - a venv without the backend dependencies
        return f"the shared QA scanner is not importable here ({exc})"
    return ""


_BACKEND_ERROR = _backend_error()
needs_backend = pytest.mark.skipif(bool(_BACKEND_ERROR), reason=_BACKEND_ERROR or "backend importable")


def _alive() -> int:
    """Interpreters that exist in this process, the main one too."""
    return len(interpreters._api.list_all())


@pytest.fixture
def pool(tmp_path):
    made = interpreters.InterpreterPool(2, work_dir=str(tmp_path))
    yield made
    made.close()


@pytest.fixture
def phone(tmp_path, monkeypatch):
    """The phone's gates, every path in tmp."""
    for name, value in (("GLOSSARION_MOBILE", "1"), ("GLOSSARION_NO_PROCESSES", "1"), ("GLOSSARION_HTTP_LOG", "0")):
        monkeypatch.setenv(name, value)
    for name, sub in (("HOME", "home"), ("USERPROFILE", "home"), ("APPDATA", "appdata"), ("LOCALAPPDATA", "local"),
                      ("GLOSSARION_LIBRARY_DIR", "library"), ("OUTPUT_DIRECTORY", "out"), ("GLOSSARION_DATA_DIR", "data")):
        (tmp_path / sub).mkdir(exist_ok=True)
        monkeypatch.setenv(name, str(tmp_path / sub))
    monkeypatch.setenv("CONFIG_FILE", str(tmp_path / "data" / "config.json"))
    monkeypatch.setenv("AI_HUNTER_MAX_WORKERS", "4")
    for name in ("QA_MIN_DUPLICATE_WORD_COUNT", "SKIP_TITLE_TAG_TRANSLATION"):
        monkeypatch.delenv(name, raising=False)
    stop_flags = ("GRACEFUL_STOP", "TRANSLATION_CANCELLED")
    for name in stop_flags:
        os.environ.pop(name, None)
    yield tmp_path
    for name in stop_flags:  # a stopped scan leaves them for the next run's reset: not for the next test
        os.environ.pop(name, None)
    interpreters.drop_kept_pool()
    deadline = time.monotonic() + 60  # children a stopped scan left in a batch end at their next item
    while interpreters._api is not None and _alive() > 1 and time.monotonic() < deadline:
        time.sleep(0.05)


# ---- the pool --------------------------------------------------------------------------------------------------------


@needs_interpreters
def test_a_call_runs_in_another_interpreter_and_returns_its_value(pool):
    import interp_workers

    args = {"text": "안녕하세요", "numbers": [1, 2.5, None], "nested": ("a", {"b": b"\x00\xff"})}
    assert pool.call(WORKERS, "echo", args) == args
    assert pool.call(WORKERS, "where") != interp_workers.where()
    assert interp_workers.CALLS == 0 and pool.call(WORKERS, "count_calls") >= 1 and interp_workers.CALLS == 0


@needs_interpreters
def test_submit_gives_a_future(pool):
    futures = [pool.submit(WORKERS, "echo", number) for number in range(6)]
    assert isinstance(futures[0], concurrent.futures.Future)
    assert sorted(future.result(timeout=120) for future in futures) == list(range(6))


@needs_interpreters
def test_interpreters_are_reused_and_released(tmp_path):
    before = _alive()
    made = interpreters.InterpreterPool(2, work_dir=str(tmp_path))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as outside:  # the scanner's pools call from their own threads
        for _round in range(3):  # three pools of a scan, the same children
            assert list(outside.map(lambda n: made.call(WORKERS, "echo", n), range(8))) == list(range(8))
    assert made.started == 2 and made.calls == 24 and made.live() == 2 and _alive() == before + 2
    # one child done a lot of calls: the same interpreter, its module state kept
    counts = [made.call(WORKERS, "count_calls") for _ in range(5)]
    assert counts == sorted(counts) and counts[-1] - counts[0] == 4
    folder = made._dir
    assert folder and os.path.isdir(folder) and Path(folder).parent == tmp_path
    made.close()
    assert made.live() == 0 and made.destroyed == 2 and _alive() == before
    assert not os.path.exists(folder), "the pool's private folder stays behind"
    with pytest.raises(interpreters.InterpreterUnavailable):
        made.call(WORKERS, "echo", 1)
    with pytest.raises(interpreters.InterpreterUnavailable):
        made.submit(WORKERS, "echo", 1)


@needs_interpreters
def test_a_reused_child_runs_with_the_callers_environment(pool, monkeypatch):
    monkeypatch.setenv("INTERP_KEEP", "one")
    monkeypatch.setenv("INTERP_GONE", "here")
    names = ["INTERP_KEEP", "INTERP_GONE", "INTERP_NEW"]
    pool.size = 1
    assert pool.call(WORKERS, "get_env", names) == {"INTERP_KEEP": "one", "INTERP_GONE": "here", "INTERP_NEW": None}
    monkeypatch.setenv("INTERP_KEEP", "two")
    monkeypatch.delenv("INTERP_GONE")
    monkeypatch.setenv("INTERP_NEW", "안녕")
    assert pool.call(WORKERS, "get_env", names) == {"INTERP_KEEP": "two", "INTERP_GONE": None, "INTERP_NEW": "안녕"}
    assert pool.started == 1


@needs_interpreters
def test_what_a_child_writes_to_its_environment_stays_there(pool):
    pool.size = 1
    assert pool.call(WORKERS, "set_env", {"INTERP_FROM_CHILD": "x"}) is True
    assert "INTERP_FROM_CHILD" not in os.environ
    assert pool.call(WORKERS, "get_env", ["INTERP_FROM_CHILD"]) == {"INTERP_FROM_CHILD": "x"}
    # a plain dict: os.environ's own would call os.putenv, on the C environment all the interpreters share
    assert pool.call(WORKERS, "environ_type") == "dict"


@needs_interpreters
def test_a_key_in_the_environment_or_the_arguments_never_reaches_a_file(tmp_path, monkeypatch):
    secret = "sk-interp-test-key-0123456789"
    monkeypatch.setenv("API_KEY", secret)
    made = interpreters.InterpreterPool(1, work_dir=str(tmp_path))
    seen: list = []
    stop = threading.Event()

    def watch() -> None:
        while not stop.is_set():
            for path in Path(tmp_path).rglob("*"):
                try:
                    seen.append(path.read_bytes() if path.is_file() else b"")
                except OSError:
                    pass

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    try:
        assert made.call(WORKERS, "items", ({"api_key": secret, "count": 1}["count"], 0.3, None),
                         should_stop=lambda: False)["done"] == 1
        made.call(WORKERS, "echo", {"not": "the key"})
        monkeypatch.setenv("API_KEY", secret + "-changed")  # the next call's env difference
        made.call(WORKERS, "items", (2, 0.1, None), should_stop=lambda: False)
    finally:
        stop.set()
        watcher.join(10)
        made.close()
    assert seen, "the watcher saw no file: this test proves nothing"
    assert not any(secret.encode() in data for data in seen)


@needs_interpreters
def test_a_worker_exception_surfaces_with_its_type_and_where_it_happened(pool):
    with pytest.raises(ValueError, match="the batch died") as caught:
        pool.call(WORKERS, "boom", "the batch died")
    assert "interp_workers.py" in str(caught.value.__cause__) and "boom" in str(caught.value.__cause__)
    future = pool.submit(WORKERS, "boom", "from a future")
    assert isinstance(future.exception(timeout=60), ValueError)
    # the child stays good
    assert pool.call(WORKERS, "echo", 1) == 1 and pool.started <= 2 and pool.destroyed == 0


@needs_interpreters
def test_an_exception_that_cannot_cross_still_surfaces(pool):
    with pytest.raises(interpreters.WorkerError, match="_NoRebuild|one two"):
        pool.call(WORKERS, "boom_no_rebuild")


@needs_interpreters
def test_a_missing_function_is_an_error_of_the_call_not_of_the_child(pool):
    with pytest.raises(AttributeError):
        pool.call(WORKERS, "no_such_worker", 1)
    with pytest.raises(ImportError):
        pool.call("no_such_module_here", "worker", 1)
    assert pool.call(WORKERS, "echo", 2) == 2 and pool.destroyed == 0


@needs_interpreters
def test_what_a_worker_prints_reaches_the_callers_stdout(pool, capsys):
    assert pool.call(WORKERS, "speak", "한글 line") == 7
    out = capsys.readouterr().out
    assert "한글 line" in out and "to stderr" in out


@needs_interpreters
def test_a_child_gets_the_callers_recursion_limit(tmp_path):
    previous = sys.getrecursionlimit()
    sys.setrecursionlimit(previous + 123)
    try:
        with interpreters.InterpreterPool(1, work_dir=str(tmp_path)) as made:
            assert made.call(WORKERS, "recursion_limit") == previous + 123
    finally:
        sys.setrecursionlimit(previous)


@needs_interpreters
def test_after_call_runs_and_may_fail(tmp_path):
    with interpreters.InterpreterPool(1, after_call=(WORKERS, "clear_marker"), work_dir=str(tmp_path)) as made:
        made.call(WORKERS, "echo", 1)
        assert made.call(WORKERS, "get_env", ["INTERP_AFTER_CALL"]) == {"INTERP_AFTER_CALL": "1"}
    with interpreters.InterpreterPool(1, after_call=(WORKERS, "fail_after_call"), work_dir=str(tmp_path)) as made:
        assert made.call(WORKERS, "echo", 5) == 5 and made.call(WORKERS, "echo", 6) == 6 and made.destroyed == 0


@needs_interpreters
def test_a_stop_reaches_the_worker_between_its_items(pool):
    stopped = threading.Event()
    future = pool.submit(WORKERS, "items", (10_000, 0.01, None), should_stop=stopped.is_set)
    time.sleep(0.5)
    assert not future.done()
    asked = time.monotonic()
    stopped.set()
    result = future.result(timeout=30)
    assert result["stopped_at"] is not None and time.monotonic() - asked < 5
    # later calls with the same check see it at once; another check does not
    assert pool.call(WORKERS, "items", (10, 0.0, None), should_stop=stopped.is_set) == {"stopped_at": 0}
    assert pool.call(WORKERS, "items", (2, 0.0, None), should_stop=lambda: False)["done"] == 2


@needs_interpreters
def test_pool_stop_tells_every_child(pool):
    futures = [pool.submit(WORKERS, "items", (10_000, 0.01, None), should_stop=lambda: False) for _ in range(2)]
    time.sleep(0.5)
    pool.stop()
    assert all(future.result(timeout=30)["stopped_at"] is not None for future in futures)


@needs_interpreters
def test_close_does_not_wait_for_a_child_left_in_a_batch_and_it_is_released_when_that_ends(tmp_path):
    before = _alive()
    made = interpreters.InterpreterPool(1, work_dir=str(tmp_path))
    box: dict = {}
    thread = threading.Thread(target=lambda: box.update(result=made.call(WORKERS, "items", (10_000, 0.01, None),
                                                                         should_stop=lambda: False)))
    thread.start()
    deadline = time.monotonic() + 60
    while not made.live() and time.monotonic() < deadline:
        time.sleep(0.02)
    time.sleep(0.3)
    started = time.monotonic()
    made.close()
    assert time.monotonic() - started < 2
    thread.join(30)
    assert box["result"]["stopped_at"] is not None  # close tells it, the batch's result is still returned
    assert _alive() == before and made.live() == 0 and not list(Path(tmp_path).iterdir())


@needs_interpreters
def test_close_can_let_a_running_batch_finish(tmp_path):
    made = interpreters.InterpreterPool(1, work_dir=str(tmp_path))
    future = concurrent.futures.ThreadPoolExecutor(max_workers=1).submit(
        made.call, WORKERS, "items", (30, 0.02, None), should_stop=lambda: False)
    deadline = time.monotonic() + 60
    while not made._busy and time.monotonic() < deadline:
        time.sleep(0.01)
    time.sleep(0.2)
    made.close(stop_running=False)
    assert future.result(timeout=60) == {"stopped_at": None, "done": 30}
    assert made.live() == 0


@needs_interpreters
def test_a_child_that_cannot_be_prepared_is_unavailable_and_not_left_behind(tmp_path):
    before = _alive()
    made = interpreters.InterpreterPool(2, preload=("no_such_module_anywhere",), work_dir=str(tmp_path))
    with pytest.raises(interpreters.InterpreterUnavailable, match="no_such_module_anywhere"):
        made.call(WORKERS, "echo", 1)
    made.close()
    assert _alive() == before


def test_no_interpreter_module_is_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr(interpreters, "_api", None)
    assert interpreters.available() is False and "no interpreter module" in interpreters.unavailable_reason()
    with pytest.raises(interpreters.InterpreterUnavailable):
        interpreters.InterpreterPool(1, work_dir=str(tmp_path)).call(WORKERS, "echo", 1)


@needs_interpreters
def test_arguments_that_cannot_cross_are_unavailable_and_keep_the_child(pool):
    with pytest.raises(interpreters.InterpreterUnavailable, match="cannot cross"):
        pool.call(WORKERS, "echo", lambda: None)
    assert pool.call(WORKERS, "echo", 3) == 3 and pool.destroyed == 0


@needs_interpreters
def test_when_only_some_children_start_they_do_the_work(tmp_path, monkeypatch):
    real = interpreters._create
    made_count = [0]

    def create_one():
        made_count[0] += 1
        if made_count[0] > 1:
            raise RuntimeError("no room for another")
        return real()

    monkeypatch.setattr(interpreters, "_create", create_one)
    with interpreters.InterpreterPool(4, work_dir=str(tmp_path)) as made:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as outside:
            assert sorted(outside.map(lambda n: made.call(WORKERS, "echo", n), range(12))) == list(range(12))
        assert made.started == 1 and made.size == 1


def test_importable_is_a_top_level_function_of_a_module():
    import interp_workers

    assert interpreters.importable(interp_workers.echo)
    assert not interpreters.importable(interp_workers.not_at_the_top_level())
    assert not interpreters.importable(lambda: None)
    assert not interpreters.importable(_main_like)


def _main_like():
    pass


_main_like.__module__ = "__main__"

MB = 1024 * 1024


@pytest.mark.parametrize("wanted,free_mb,total_mb,expected", [
    (4, 4000, 8000, 4),  # plenty
    (4, 500, 8000, 4),  # 300 stay free, 4 x 50 fit in the rest
    (4, 450, 8000, 3),
    (4, 400, 8000, 2),
    (4, 350, 8000, 1),
    (4, 349, 8000, 0),  # threads
    (4, 100, 8000, 0),
    (8, 620, 8000, 6),
    (2, 4000, 8000, 2),  # never more than wanted
    (4, None, 4000, 4),  # iOS: a quarter of the RAM counts as free
    (4, None, 1500, 1),
    (4, None, 1000, 0),
    (4, None, 0, 4),  # nothing known
])
def test_memory_rule(wanted, free_mb, total_mb, expected):
    free = None if free_mb is None else free_mb * MB
    assert interpreters.size_for_memory(wanted, free, total_mb * MB) == expected


# ---- the book -------------------------------------------------------------------------------------------------------


def _book(folder: Path, chapters: int = 24, words: int = 500, seed: int = 9151) -> str:
    """``tools/qa_scan_bench.make_book``'s chapters with the faults a phone's book has: Korean left in, broken tags,
    duplicates, a truncated file."""
    book = bench.make_book(folder, chapters=chapters, words=words, seed=seed)
    names = sorted(name for name in os.listdir(book) if name.endswith(".html"))

    def edit(index, change):
        path = os.path.join(book, names[index])
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(change(text))

    edit(3, lambda t: t.replace("</p>", "</p><p>안녕하세요 그는 말했다 진짜로 말이야.</p>", 12))
    edit(6, lambda t: t.replace("<p>", '<p"', 3))
    shutil.copy(os.path.join(book, names[12]), os.path.join(book, names[13]))
    edit(15, lambda t: t[:300])
    edit(18, lambda t: t.replace("</p>", "</p><p>Here is the translation of the chapter you asked for:</p>", 1))
    return book


def _settings(mode: str, **extra) -> dict:
    settings = {"target_language": "english", "quick_scan_sample_size": 1000, "check_translation_artifacts": True,
                "check_ai_artifacts": True, "foreign_char_threshold": 0}
    settings.update(extra)
    return settings


def _paths_out(rows: list, folder: str) -> list:
    return json.loads(json.dumps(rows).replace(folder.replace("\\", "\\\\"), "<book>"))


def _report_rows(book: str) -> list:
    """Every field of every row of the scan report."""
    path = os.path.join(book, os.path.basename(book) + "_Scan Report", "validation_results.json")
    with open(path, encoding="utf-8") as handle:
        rows = json.load(handle)
    return _paths_out(sorted(rows, key=lambda row: row.get("filename", "")), book)


class _Lines(list):
    def __call__(self, line) -> None:
        self.append(str(line))


class _PoolSpy:
    """The pools a scan makes (``InterpreterPool`` is the real one)."""

    def __init__(self, monkeypatch) -> None:
        self.made: list = []
        self.real = interpreters.InterpreterPool
        spy = self

        def making(*args, **kwargs):
            pool = spy.real(*args, **kwargs)
            spy.made.append(pool)
            return pool

        monkeypatch.setattr(interpreters, "InterpreterPool", making)


def _scan(book: str, mode: str, settings: dict, *, interpreted: bool, monkeypatch, stop=None) -> _Lines:
    """One scan the way ``job_kinds.qa`` runs it, through the scanner's own entry."""
    import scan_html_folder as shf

    from glossarion_mobile.job_kinds import qa as qa_kind

    monkeypatch.setattr(qa_kind, "SCAN_INTERPRETERS", interpreted)
    lines = _Lines()
    with qa_kind._scan_turns(True, lines):
        shf.clear_qa_caches()
        shf.scan_html_folder(book, log=lines, mode=mode, qa_settings=dict(settings), stop_flag=stop)
    return lines


def _went_to_children(lines: list) -> bool:
    return any("separate Python interpreter" in line for line in lines)


def _fell_back(lines: list) -> list:
    return [line for line in lines if "Scanning on threads" in line]


# ---- the scan: findings -------------------------------------------------------------------------------------------


@needs_interpreters
@needs_backend
@pytest.mark.parametrize("mode", ["quick-scan", "aggressive", "ai-hunter"])
def test_findings_are_the_same_on_threads_and_in_interpreters(phone, monkeypatch, mode):
    runs = {}
    for name, interpreted in (("threads", False), ("children", True)):
        book = _book(phone / "out" / name / "Book")
        lines = _scan(book, mode, _settings(mode), interpreted=interpreted, monkeypatch=monkeypatch)
        assert _went_to_children(lines) is interpreted, lines[:8]
        assert not any("Error processing batch" in line for line in lines), lines
        assert not _fell_back(lines), lines
        runs[name] = _report_rows(book)
    assert runs["threads"] == runs["children"]
    issues = {row["filename"]: row.get("issues") or [] for row in runs["children"]}
    flagged = [name for name, found in issues.items() if found]
    # the faults were found, whichever way: the comparison is not of two empty reports
    assert len(runs["children"]) == 24 and len(flagged) >= 3, issues
    assert any("Korean_text" in found for found in issues["response_0004_ch0004.html"]), issues
    assert "malformed_html" in issues["response_0007_ch0007.html"], "broken tags were not found"
    assert "unclosed_html_tags" in issues["response_0016_ch0016.html"], "the truncated file was not found"
    if mode != "quick-scan":  # Quick Scan's duplicate check does other work
        assert any("DUPLICATE" in found for found in issues["response_0013_ch0013.html"]), issues


@needs_interpreters
@needs_backend
def test_changed_settings_reach_interpreters_that_are_reused(phone, monkeypatch):
    """Within one job's children (a bulk scan): the settings change between two scans, through the arguments and through
    the env, and the second scan finds what threads find. The change has to change something, or this proves nothing."""
    import scan_html_folder as shf

    from glossarion_mobile.job_kinds import qa as qa_kind

    spy = _PoolSpy(monkeypatch)

    def two_scans(name: str, interpreted: bool) -> list:
        monkeypatch.setattr(qa_kind, "SCAN_INTERPRETERS", interpreted)
        monkeypatch.delenv("QA_MIN_DUPLICATE_WORD_COUNT", raising=False)
        lines = _Lines()
        out = []
        with qa_kind._scan_turns(True, lines):
            for round_index, (settings, env) in enumerate((
                    (_settings("quick-scan"), {"SKIP_TITLE_TAG_TRANSLATION": "1"}),
                    # a word-count threshold in the env (small-file checks), the title tags counted, no artifact checks
                    (_settings("quick-scan", check_translation_artifacts=False, check_ai_artifacts=False,
                               foreign_char_threshold=10_000, check_missing_tag_bracket=False),
                     {"SKIP_TITLE_TAG_TRANSLATION": "0", "QA_MIN_DUPLICATE_WORD_COUNT": "100000"}))):
                for key, value in env.items():
                    monkeypatch.setenv(key, value)
                book = _book(phone / "out" / name / str(round_index) / "Book", seed=9151)
                # a title only the second scan's env lets the foreign-character check see
                first = os.path.join(book, "response_0002_ch0002.html")
                with open(first, encoding="utf-8") as handle:
                    text = handle.read()
                with open(first, "w", encoding="utf-8", newline="\n") as handle:
                    handle.write(text.replace("<title>", "<title>안녕하세요 안녕하세요 ", 1))
                shf.clear_qa_caches()
                shf.scan_html_folder(book, log=lines, mode="quick-scan", qa_settings=settings)
                out.append(_report_rows(book))
        assert _went_to_children(lines) is interpreted, lines[:8]
        return out

    on_threads = two_scans("threads", False)
    in_children = two_scans("children", True)
    assert on_threads[0] != on_threads[1], "the settings change changed nothing: this test proves nothing"
    assert in_children == on_threads
    assert len(spy.made) == 1, "one job, one set of children"


# ---- the scan: the interpreters ------------------------------------------------------------------------------------


@needs_interpreters
@needs_backend
def test_all_the_pools_of_a_scan_share_the_children_and_they_stay(phone, monkeypatch):
    spy = _PoolSpy(monkeypatch)
    before = _alive()
    book = _book(phone / "out" / "Book")
    workers = 3
    monkeypatch.setenv("AI_HUNTER_MAX_WORKERS", str(workers))
    # Aggressive: the files pool, the enhanced pools and the deep similarity pool (three on the emulator: 12 interpreters)
    lines = _scan(book, "aggressive", _settings("aggressive"), interpreted=True, monkeypatch=monkeypatch)
    assert _went_to_children(lines) and len(spy.made) == 1
    made = spy.made[0]
    pools = sum("Split into" in line or "Processing" in line for line in lines)
    assert pools >= 2, lines
    assert 1 <= made.started <= workers and made.calls > workers, (made.started, made.calls)
    assert made.live() == made.started and made.destroyed == 0 and _alive() == before + made.started


@needs_interpreters
@needs_backend
def test_a_scan_that_reaches_no_batch_starts_nothing(phone, monkeypatch):
    spy = _PoolSpy(monkeypatch)
    from glossarion_mobile.job_kinds import qa as qa_kind

    with qa_kind._scan_turns(True, _Lines()):
        pass
    assert spy.made == []


@needs_interpreters
@needs_backend
def test_the_scanner_is_put_back_and_the_desktop_never_goes_this_way(phone, monkeypatch):
    import scan_html_folder as shf

    real = shf._submit_worker
    from glossarion_mobile.job_kinds import qa as qa_kind

    with qa_kind._scan_turns(True, _Lines()):
        assert shf._submit_worker is not real
    assert shf._submit_worker is real
    with pytest.raises(RuntimeError):
        with qa_kind._scan_turns(True, _Lines()):
            raise RuntimeError("the scan died")
    assert shf._submit_worker is real
    # the shared module and the desktop code do not know the interpreters at all
    for shared in ("scan_html_folder.py", "qa_scan_runtime.py", "mobile_runtime.py", "QA_Scanner_GUI.py"):
        assert "interpreters" not in (SRC_DIR / shared).read_text(encoding="utf-8-sig"), shared


@needs_interpreters
@needs_backend
def test_a_worker_a_child_cannot_find_stays_on_a_thread(phone, monkeypatch):
    import scan_html_folder as shf

    from glossarion_mobile.job_kinds import qa as qa_kind

    spy = _PoolSpy(monkeypatch)
    here = threading.get_ident()

    def local_worker(args, should_stop):
        return args, threading.get_ident() != here, should_stop()

    with qa_kind._scan_turns(True, _Lines()):
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            assert shf._submit_worker(pool, local_worker, "x", lambda: False).result(timeout=30) == ("x", True, False)
    assert spy.made == []


# ---- the two paths that run a scan -------------------------------------------------------------------------------------


class _Ctx:
    """What ``job_kinds.qa.scan`` needs of a job context, over the real ``HeadlessOwner``."""

    def __init__(self, config: dict, stop_when=None) -> None:
        from headless_owner import HeadlessOwner

        self.config = config
        self.owner = HeadlessOwner(json.loads(json.dumps(config)))
        self.lines = _Lines()
        self.results: dict = {}
        self.stop_when = stop_when or (lambda _lines: False)

    def log(self, text) -> None:
        self.lines.append(str(text))

    def phase(self, _label) -> None:
        pass

    def stop_requested(self) -> bool:
        return bool(self.stop_when(self.lines))

    def add_outputs(self, _paths) -> None:
        pass

    def set_result(self, **data) -> None:
        self.results.update(data)


def _job_config() -> dict:
    return {"output_language": "English", "qa_scanner_settings": {"quick_scan_sample_size": 1000},
            "ai_hunter_config": {"ai_hunter_max_workers": 3}}


@needs_interpreters
@needs_backend
def test_a_qa_scan_job_runs_in_children_by_default_and_finds_what_threads_find(phone, monkeypatch):
    from glossarion_mobile.job_kinds import qa as qa_kind

    monkeypatch.setattr(os, "cpu_count", lambda: 8)
    before = _alive()
    assert qa_kind.SCAN_INTERPRETERS is True
    book = _book(phone / "out" / "children" / "Book")
    ctx = _Ctx(_job_config())
    assert qa_kind.scan(ctx, [(book, None, False)], "quick-scan", disable_word_count=True)["ok"] is True
    assert _went_to_children(ctx.lines) and not _fell_back(ctx.lines), ctx.lines
    assert _alive() == before + interpreters.kept_pool().started and interpreters.kept_pool().destroyed == 0
    found = _report_rows(book)
    monkeypatch.setattr(qa_kind, "SCAN_INTERPRETERS", False)
    book = _book(phone / "out" / "threads" / "Book")
    ctx = _Ctx(_job_config())
    assert qa_kind.scan(ctx, [(book, None, False)], "quick-scan", disable_word_count=True)["ok"] is True
    assert not _went_to_children(ctx.lines)
    assert found == _report_rows(book) and any(row["issues"] for row in found)


@needs_interpreters
@needs_backend
def test_the_multipass_scan_of_a_translation_runs_in_children_too(phone, monkeypatch, capsys):
    """``TranslateKRtoEN`` is too heavy to import for this: a stand-in with the real function's name, the real scanner."""
    import types

    import scan_html_folder as shf

    from glossarion_mobile.job_kinds import qa as qa_kind

    monkeypatch.setitem(sys.modules, "TransateKRtoEN", types.SimpleNamespace(_run_multipass_refinement_qa_scan=None))
    backend = sys.modules["TransateKRtoEN"]
    book = _book(phone / "out" / "Book")
    seen = _Lines()
    original = lambda: shf.scan_html_folder(book, log=seen, mode="quick-scan", qa_settings=_settings("quick-scan"))  # noqa: E731
    backend._run_multipass_refinement_qa_scan = original
    with qa_kind.multipass_scan_turns():
        assert backend._run_multipass_refinement_qa_scan is not original
        backend._run_multipass_refinement_qa_scan()
    assert backend._run_multipass_refinement_qa_scan is original
    assert "separate Python interpreter" in capsys.readouterr().out, "the job's log tees print"
    assert not any("Error processing batch" in line for line in seen) and _report_rows(book)


@needs_interpreters
@needs_backend
@pytest.mark.parametrize("interpreted", [False, True])
def test_a_stopped_job_ends_like_a_stopped_job(phone, monkeypatch, interpreted):
    """Stop through the job's own check: ``ok`` None, no report, the scan over soon."""
    from glossarion_mobile.job_kinds import qa as qa_kind

    monkeypatch.setattr(qa_kind, "SCAN_INTERPRETERS", interpreted)
    monkeypatch.setattr(os, "cpu_count", lambda: 8)
    book = bench.make_book(phone / "out" / "Book", chapters=120, words=2500)
    marks: dict = {}

    reached = "separate Python interpreter" if interpreted else "batches of"  # the line of a child is there first

    def stop_when(lines) -> bool:
        if "first" not in marks and any(reached in line for line in lines):
            marks["first"] = time.monotonic()
        return "first" in marks and time.monotonic() - marks["first"] > 0.4

    ctx = _Ctx(_job_config(), stop_when)
    result = qa_kind.scan(ctx, [(book, None, False)], "aggressive", disable_word_count=True)
    ended = time.monotonic()
    assert "first" in marks, "the scan ended before it reached a batch"
    assert result["ok"] is None and result["outputs"] == [], ctx.lines[-6:]
    assert not os.path.exists(os.path.join(book, "Book_Scan Report", "validation_results.json"))
    assert ended - marks["first"] < 30
    assert _went_to_children(ctx.lines) is interpreted


# ---- the scan: Stop ----------------------------------------------------------------------------------------------------


def _stopped_scan(phone, name: str, interpreted: bool, monkeypatch) -> tuple:
    """A real Aggressive scan stopped once it is well into its batches. Returns (seconds from Stop to the scan's end,
    the lines, the book)."""
    import scan_html_folder as shf

    from glossarion_mobile.job_kinds import qa as qa_kind

    monkeypatch.setattr(qa_kind, "SCAN_INTERPRETERS", interpreted)
    book = bench.make_book(phone / "out" / name / "Book", chapters=120, words=2500)
    lines = _Lines()
    stop = threading.Event()

    def stop_soon() -> None:
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            if any("batches of" in line for line in lines) and (not interpreted or _went_to_children(lines)):
                break  # its batches are in the pool
            time.sleep(0.01)
        time.sleep(0.4)
        stop.set()

    box: dict = {}
    threading.Thread(target=stop_soon, daemon=True).start()
    with qa_kind._scan_turns(True, lines):
        shf.clear_qa_caches()
        box["result"] = shf.scan_html_folder(book, log=lines, mode="aggressive", qa_settings=_settings("aggressive"),
                                             stop_flag=stop.is_set)
        asked = time.monotonic()
    assert stop.is_set(), "the scan ended before the Stop: this test proves nothing"
    # stop.is_set() is polled every 0.1 s: the scan was over about when it was set
    return time.monotonic() - asked, lines, book, box["result"]


@needs_interpreters
@needs_backend
def test_stop_ends_a_scan_at_once_and_writes_no_report_like_on_threads(phone, monkeypatch):
    spy = _PoolSpy(monkeypatch)
    before = _alive()
    outcomes = {}
    for name, interpreted in (("threads", False), ("children", True)):
        started = time.monotonic()
        _left, lines, book, result = _stopped_scan(phone, name, interpreted, monkeypatch)
        outcomes[name] = {
            "seconds": time.monotonic() - started,
            "result": result,
            "report": os.path.exists(os.path.join(book, "Book_Scan Report", "validation_results.json")),
            "stop_line": any("Stop requested" in line for line in lines),
            "interrupted": any("interrupted" in line for line in lines),
        }
        assert _went_to_children(lines) is interpreted
    for name, outcome in outcomes.items():
        assert outcome["result"] is None and outcome["report"] is False and outcome["stop_line"], (name, outcome)
    # the children left in a batch end at their next item and are free again, without the scan waiting for them
    made = spy.made[0]
    deadline = time.monotonic() + 60
    while made._busy and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not made._busy and made.destroyed == 0 and _alive() == before + made.live()
    assert made._watched == {} and os.listdir(made._dir) == []


@needs_interpreters
@needs_backend
def test_stop_while_a_child_runs_a_real_batch_returns_within_a_poll(phone, monkeypatch):
    """A long batch of the real worker: a Stop reaches it inside the child, between two files."""
    import scan_html_folder as shf

    book = bench.make_book(phone / "out" / "Book", chapters=120, words=2500)
    names = sorted(name for name in os.listdir(book) if name.endswith(".html"))
    args = (list(enumerate(names)), book, _settings("aggressive"), "aggressive", {}, {}, False, {}, {})
    stop = threading.Event()
    with interpreters.InterpreterPool(1, preload=("scan_html_folder",), work_dir=str(phone)) as made:
        made.call(WORKERS, "echo", 0)  # started: the time below is the batch's
        future = made.submit("scan_html_folder", "process_html_file_batch", args, should_stop=stop.is_set)
        time.sleep(0.5)
        assert not future.done()
        asked = time.monotonic()
        stop.set()
        rows = future.result(timeout=60)
        took = time.monotonic() - asked
    assert len(rows) < len(names), "the batch ran to its end"
    full = shf.process_html_file_batch((list(enumerate(names[:3])), book, _settings("aggressive"), "aggressive",
                                        {}, {}, False, {}, {}))
    assert len(full) == 3
    assert took < 10, took


# ---- the scan: fallback -----------------------------------------------------------------------------------------------


@needs_interpreters
@needs_backend
@pytest.mark.parametrize("how", ["no_module", "create_fails", "child_cannot_import", "no_memory", "dies_in_a_call"])
def test_a_device_where_children_do_not_work_scans_on_threads_with_the_same_findings(phone, monkeypatch, how):
    import scan_html_folder as shf

    book = _book(phone / "out" / "threads" / "Book")
    _scan(book, "quick-scan", _settings("quick-scan"), interpreted=False, monkeypatch=monkeypatch)
    expected = _report_rows(book)
    if how == "no_module":
        monkeypatch.setattr(interpreters, "_api", None)
    elif how == "create_fails":
        def no_child():
            raise RuntimeError("interpreter creation failed")
        monkeypatch.setattr(interpreters, "_create", no_child)
    elif how == "child_cannot_import":
        real_source = interpreters._child_source
        monkeypatch.setattr(interpreters, "_child_source", lambda: "import no_such_native_module\n" + real_source())
    elif how == "no_memory":
        monkeypatch.setattr(interpreters, "memory_now", lambda: (2000 * MB, 100 * MB))
    elif how == "dies_in_a_call":
        real_run = interpreters._run

        def run(child, code, shared=None):
            return "SystemError: the call failed" if code == "_gl_call()" else real_run(child, code, shared)

        monkeypatch.setattr(interpreters, "_run", run)
    turns_used: list = []
    from glossarion_mobile.job_kinds import qa as qa_kind

    real_take = qa_kind._WorkerTurns.take
    monkeypatch.setattr(qa_kind._WorkerTurns, "take", lambda self, stopped: (turns_used.append(1), real_take(self, stopped))[1])
    book = _book(phone / "out" / "fallback" / "Book")
    lines = _scan(book, "quick-scan", _settings("quick-scan"), interpreted=True, monkeypatch=monkeypatch)
    said = _fell_back(lines)
    assert len(said) == 1, lines  # once, however many batches
    assert not any("Error processing batch" in line for line in lines), lines
    assert turns_used, "the fallback did not take turns"
    assert _report_rows(book) == expected
    assert shf._submit_worker.__name__ == "_submit_worker"


@needs_interpreters
@needs_backend
def test_the_fallback_says_why_in_its_log_line(phone, monkeypatch):
    monkeypatch.setattr(interpreters, "memory_now", lambda: (2000 * MB, 100 * MB))
    lines = _scan(_book(phone / "out" / "Book"), "quick-scan", _settings("quick-scan"), interpreted=True,
                  monkeypatch=monkeypatch)
    assert any("100 MB of memory" in line for line in _fell_back(lines)), lines


def test_no_retry_loop_after_a_failure(monkeypatch):
    """Failing once decides for the rest of the scan: no second try to start a child (owner's rule on loop guards)."""
    from glossarion_mobile.job_kinds import qa as qa_kind

    attempts: list = []

    def no_child():
        attempts.append(1)
        raise RuntimeError("no")

    monkeypatch.setattr(interpreters, "_create", no_child)
    monkeypatch.setattr(interpreters, "_api", interpreters._api or object())
    monkeypatch.setattr(interpreters, "unavailable_reason", lambda: None)
    lines = _Lines()
    children = qa_kind._ScanChildren(lambda *a: None, qa_kind._WorkerTurns(), lines)
    import interp_workers

    def worker(args, _check):
        return args

    worker.__module__, worker.__name__ = interp_workers.echo.__module__, "echo"  # what a child would be asked for
    for number in range(5):
        assert children._batch(worker, number, lambda: False, 4) == number  # each batch still ran, on this thread
    assert len(attempts) == 1 and len(_fell_back(lines)) == 1


@needs_interpreters
@needs_backend
def test_falling_back_does_not_cut_short_the_batches_running_in_other_children(phone, monkeypatch):
    """One child fails a call while others are in their batches: those results count, in full, so they must not be told
    to stop (a stopped worker returns what it has, which would look like a finished batch)."""
    import interp_workers

    from glossarion_mobile.job_kinds import qa as qa_kind

    real_run = interpreters._run
    failed = threading.Event()

    def run(child, code, shared=None):
        if code == "_gl_call()" and b"echo" in shared["_gl_request"] and not failed.is_set():
            failed.set()
            return "SystemError: this child failed"
        return real_run(child, code, shared)

    monkeypatch.setattr(interpreters, "_run", run)
    lines = _Lines()
    children = qa_kind._ScanChildren(None, qa_kind._WorkerTurns(), lines)
    stop = lambda: False  # noqa: E731
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        long_batch = pool.submit(children._batch, interp_workers.items, (200, 0.02, None), stop, 2)
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline and not (children._job._pool and children._job._pool.pool.calls == 0 and children._job._pool.pool._busy):
            time.sleep(0.01)
        time.sleep(0.4)
        # the worker exists here too, for the thread path this batch ends up on
        assert pool.submit(children._batch, interp_workers.echo_checked, "x", stop, 2).result(timeout=60) == "x"
        assert failed.is_set() and len(_fell_back(lines)) == 1
        assert not long_batch.done(), "it ended before the other child failed: this test proves nothing"
        assert long_batch.result(timeout=60) == {"stopped_at": None, "done": 200}


@needs_interpreters
@needs_backend
def test_a_child_forgets_the_scanners_caches_after_each_batch(tmp_path, phone):
    """``clear_qa_caches`` alone leaves the inner cache of ``calculate_similarity_ratio``: in a child its keys are a
    batch's own strings, 20,000 pairs of them at most, and a long book's all-pairs stage would keep them all."""
    after = ("glossarion_mobile.services.scan_child", "forget_caches")
    with interpreters.InterpreterPool(1, preload=("scan_html_folder",), work_dir=str(tmp_path)) as made:
        assert made.call(WORKERS, "fill_similarity_cache", 10) == [10, 10]
        assert made.call(WORKERS, "similarity_cache_sizes") == [10, 10]  # no hook: they stay
    with interpreters.InterpreterPool(1, preload=("scan_html_folder",), after_call=after, work_dir=str(tmp_path)) as made:
        # the hook runs after the result left: the outer and the inner cache both held them
        assert made.call(WORKERS, "fill_similarity_cache", 10) == [10, 10]
        assert made.call(WORKERS, "similarity_cache_sizes") == [0, 0]


# ---- the scan: a worker's exception -----------------------------------------------------------------------------------


@needs_interpreters
@needs_backend
def test_a_worker_exception_in_a_scan_surfaces_from_its_future_and_stays_on_the_children(phone, monkeypatch):
    import scan_html_folder as shf

    from glossarion_mobile.job_kinds import qa as qa_kind

    spy = _PoolSpy(monkeypatch)
    lines = _Lines()
    with qa_kind._scan_turns(True, lines):
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            bad = shf._submit_worker(pool, shf.process_html_file_batch, ("not", "the", "arguments"), lambda: False)
            error = bad.exception(timeout=120)
            assert isinstance(error, ValueError), error
            assert "process_html_file_batch" in str(error.__cause__), "no traceback from the child"
            good = shf._submit_worker(pool, shf.process_comparison_batch_fast,
                                      ([], {"all_data": [], "thresholds": {"similarity": 0.9}}), lambda: False)
            assert good.result(timeout=120) == []
    assert not _fell_back(lines) and len(spy.made) == 1


@needs_interpreters
@needs_backend
def test_a_scan_logs_a_batch_that_fails_in_a_child_like_on_threads(phone, monkeypatch):
    """The scanner's own handling of a failed file batch: its "Error processing batch" line and the traceback."""
    import scan_html_folder as shf

    from glossarion_mobile.job_kinds import qa as qa_kind

    book = _book(phone / "out" / "Book")
    results = {}
    for interpreted in (False, True):
        monkeypatch.setattr(qa_kind, "SCAN_INTERPRETERS", interpreted)
        lines = _Lines()
        with qa_kind._scan_turns(True, lines):
            inner = shf._submit_worker  # what a worker gets is bad on both paths (not monkeypatch: _scan_turns puts it back)
            shf._submit_worker = lambda pool, worker, args, stop: inner(
                pool, worker, ("spoiled",) if worker is shf.process_html_file_batch else args, stop)
            try:
                shf.scan_html_folder(book, log=lines, mode="quick-scan", qa_settings=_settings("quick-scan"))
            except Exception:
                pass
            finally:
                shf._submit_worker = inner
        results[interpreted] = [line for line in lines if "Error processing batch" in line]
    assert results[False] and results[True]
    assert results[False] == results[True]
    assert shf._submit_worker.__name__ == "_submit_worker"


# ---- parity of what a worker reaches --------------------------------------------------------------------------------


@needs_backend
def test_nothing_a_scan_worker_reaches_changes_in_a_child():
    """What a child cannot load (lxml, regex, numpy, charset_normalizer) must not decide anything a worker does: the 'xml'
    parser and a default parser would raise or differ there, and a try/except import would hide it. A worker stays
    free of that by never using them: read from the source, every function it reaches."""
    source = (SRC_DIR / "scan_html_folder.py").read_text(encoding="utf-8-sig")
    tree = ast.parse(source)
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    reached, stack = set(), ["process_html_file_batch", "process_enhance_duplicate_batch",
                             "process_deep_similarity_batch", "process_comparison_batch_fast"]
    while stack:
        name = stack.pop()
        name = name if name in functions else f"{name}_impl"
        if name in reached or name not in functions:
            continue
        reached.add(name)
        stack.extend({node.id for node in ast.walk(functions[name]) if isinstance(node, ast.Name)})
    assert len(reached) > 40
    allowed_imports = {"re", "os", "hashlib", "difflib", "traceback", "bs4"}
    for name in sorted(reached):
        for node in ast.walk(functions[name]):
            if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "BeautifulSoup":
                parser = [ast.literal_eval(arg) for arg in node.args[1:] if isinstance(arg, ast.Constant)]
                assert parser == ["html.parser"], f"{name}:{node.lineno} parses with {parser}"
            if isinstance(node, ast.Import):
                assert {alias.name.split(".")[0] for alias in node.names} <= allowed_imports, f"{name}:{node.lineno}"
            if isinstance(node, ast.ImportFrom):
                assert (node.module or "").split(".")[0] in allowed_imports, f"{name}:{node.lineno}"
            if isinstance(node, ast.Global):
                raise AssertionError(f"{name}:{node.lineno} writes module state a parent would not see: {node.names}")
            if isinstance(node, ast.Name) and node.id in ("psutil", "MinHash", "MinHashLSH", "MINHASH_AVAILABLE"):
                raise AssertionError(f"{name}:{node.lineno} depends on the optional {node.id}")


@needs_interpreters
@needs_backend
def test_a_child_sees_the_scanner_as_the_phone_needs_it(tmp_path, phone):
    """The modules the workers import at the top of the scanner load in a child, with the phone's gates."""
    with interpreters.InterpreterPool(1, preload=("scan_html_folder",), work_dir=str(tmp_path)) as made:
        assert made.call(WORKERS, "get_env", ["GLOSSARION_MOBILE"]) == {"GLOSSARION_MOBILE": "1"}
        assert made.call("scan_html_folder", "_is_truncation_qa_context", {"_qa_context": "truncation"}) is True
        assert made.call("scan_html_folder", "normalize_text", "A  b") == made.call("scan_html_folder", "normalize_text", "A  b")


# ---- the E2E's tripwires -------------------------------------------------------------------------------------------


@needs_interpreters
def test_the_e2e_tripwires_hold_in_a_child(tmp_path, monkeypatch):
    """An interpreter is neither a process nor a fork: the guards let it start, and what runs in it is refused and
    recorded the same way. Without ``set_child_setup`` the tripwires of the session would not reach it."""
    from glossarion_mobile.diagnostics import e2e

    events = tmp_path / "child_guards.jsonl"
    monkeypatch.setenv(e2e.CHILD_EVENTS_ENV, str(events))
    monkeypatch.setenv(e2e.CHILD_ALLOWED_ENV, json.dumps([str(tmp_path)]))
    guards = e2e.ProcessGuards()
    guards.install()
    interpreters.set_child_setup(("glossarion_mobile.diagnostics.e2e", "install_child_guards"))
    try:
        with interpreters.InterpreterPool(1, work_dir=str(tmp_path)) as made:
            assert made.call(WORKERS, "echo", "fine") == "fine"
            with pytest.raises(OSError, match="process tripwire"):
                made.call(WORKERS, "spawn")
    finally:
        interpreters.set_child_setup(None)
        guards.uninstall()
    assert guards.spawns == [], "the session's guards saw the interpreter as a process"
    seen = e2e.ProcessGuards.child_events(events)
    assert [event["api"] for event in seen["spawns"]] == ["subprocess.Popen"], seen


# ---- what the QA screen says ----------------------------------------------------------------------------------------


def test_the_qa_screen_says_what_is_true_on_each_path():
    source = (APP_DIR / "glossarion_mobile" / "ui" / "tools" / "qa_screen.py").read_text(encoding="utf-8-sig")
    tree = ast.parse(source)
    reason = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                  and getattr(node.targets[0], "id", "") == "THREADS_REASON")
    assert "always use threads" not in reason
    assert "separate Python interpreters" in reason and "scan faster" in reason and "threads" in reason
    assert "Maximum parallel workers" in reason


@needs_backend
def test_the_workers_hint_says_what_more_workers_do(monkeypatch):
    pytest.importorskip("flet")
    from glossarion_mobile.ui.tools import qa_model as qm

    monkeypatch.setattr(os, "cpu_count", lambda: 8)
    hint = qm.workers_hint()
    assert "1 to 8" in hint and "default 4" in hint
    assert "More scan faster" in hint and "40 MB" in hint and "until the app closes" in hint
    assert "more do not scan faster" in hint


# ---- the self-test --------------------------------------------------------------------------------------------------


@needs_interpreters
@needs_backend
def test_the_smoke_check_passes_here_and_says_so(tmp_path, phone):
    from glossarion_mobile.diagnostics import interpreter_check

    detail = interpreter_check.check_scan_in_child(tmp_path / "work", "windows")
    assert detail["result"] == "pass" and detail["files"] == 3 and detail["issues"] >= 1 and detail["released"], detail


@needs_backend
@pytest.mark.parametrize("platform,fails", [("ios", False), ("android", True), ("windows", False)])
def test_the_smoke_check_records_a_fallback_and_fails_only_android(tmp_path, phone, monkeypatch, platform, fails):
    from glossarion_mobile.diagnostics import interpreter_check

    def no_child():
        raise RuntimeError("a child cannot start here")

    monkeypatch.setattr(interpreters, "_create", no_child)
    if interpreters._api is None:  # pragma: no cover - a Python below 3.12
        monkeypatch.setattr(interpreters, "_api", object())
    if fails:
        with pytest.raises(AssertionError, match="does not work on Android"):
            interpreter_check.check_scan_in_child(tmp_path / "work", platform)
    else:
        detail = interpreter_check.check_scan_in_child(tmp_path / "work", platform)
        assert detail["result"] == "fallback" and "a child cannot start here" in detail["reason"]


@needs_interpreters
@needs_backend
def test_the_smoke_check_fails_on_other_findings_everywhere(tmp_path, phone, monkeypatch):
    from glossarion_mobile.diagnostics import interpreter_check

    real_pool = interpreters.InterpreterPool

    class Wrong(real_pool):
        def call(self, *args, **kwargs):
            rows = super().call(*args, **kwargs)
            return [dict(row, issues=["something else"]) for row in rows]

    monkeypatch.setattr(interpreters, "InterpreterPool", Wrong)
    with pytest.raises(AssertionError, match="finds other things"):
        interpreter_check.check_scan_in_child(tmp_path / "work", "ios")


@needs_backend
def test_the_check_is_in_the_smoke_suite_and_ci_reads_its_line(tmp_path):
    from glossarion_mobile.diagnostics import selftest

    assert "scan_interpreters" in [name for name, _check in selftest.SUITES["smoke"]]
    smoke = (MOBILE_DIR / "ci" / "android_smoke.sh").read_text(encoding="utf-8")
    assert "GLOSSARION_INTERPRETERS" in smoke and "interpreters_result.txt" in smoke
    workflow_path = SRC_DIR.parent / ".github" / "workflows" / "build-mobile.yml"
    if workflow_path.is_file():
        workflow = workflow_path.read_text(encoding="utf-8")
        assert "GLOSSARION_INTERPRETERS" in workflow and "FALLBACK to threads" in workflow


# ---- speed and the loop (relative: a slow runner does not fail them) -------------------------------------------------


def _spin(units: int) -> None:
    for _ in range(units):
        total = 0
        for i in range(1000):
            total += i * i % 7


def _spin_units(target: float = 0.01) -> int:
    units = 1
    while True:
        started = time.perf_counter()
        _spin(units)
        took = time.perf_counter() - started
        if took >= target / 2:
            return max(1, int(units * target / took))
        units *= 2


def _ui_while(run, units: int) -> tuple:
    """Run ``run()`` on a thread while this thread acts like the UI loop: sleep a frame (how late it wakes), then a fixed
    burst of Python (how long it takes). Returns (lateness ms, burst ms, seconds)."""
    finished = threading.Event()
    box: dict = {}

    def target() -> None:
        try:
            box["value"] = run()
        except BaseException as exc:  # noqa: BLE001 - reported on the test's thread
            box["error"] = exc
        finally:
            finished.set()

    timer = getattr(getattr(ctypes, "windll", None), "winmm", None)  # Windows: 15.6 ms ticks would hide the wait
    if timer is not None:
        timer.timeBeginPeriod(1)
    late: list = []
    bursts: list = []
    started = time.perf_counter()
    threading.Thread(target=target, daemon=True).start()
    while not finished.is_set():
        slept = time.perf_counter()
        time.sleep(0.016)
        late.append((time.perf_counter() - slept - 0.016) * 1000)
        begun = time.perf_counter()
        _spin(units)
        bursts.append((time.perf_counter() - begun) * 1000)
    if timer is not None:
        timer.timeEndPeriod(1)
    if "error" in box:
        raise box["error"]
    return late, bursts, time.perf_counter() - started


@needs_interpreters
@needs_backend
@pytest.mark.skipif((os.cpu_count() or 1) < 4, reason="needs cores for 4 workers to run at once")
def test_budget_interpreters_scan_faster_and_leave_the_loop_alone(phone, monkeypatch):
    """The same real Aggressive scan with 4 workers next to a loop: on threads (taking turns) against interpreters.

    Host, 3.12, in this order of magnitude (see the report); the emulator's own numbers are in the module's docstring.
    Asked of a shared runner: it must not be slower, and a burst of work on the loop no slower than on threads with
    a margin. Taking turns the loop shares the GIL with one worker, interpreters with the scan's own thread."""
    monkeypatch.setenv("AI_HUNTER_MAX_WORKERS", "4")
    units = _spin_units(0.01)
    results: dict = {"threads": ([], [], 0.0), "children": ([], [], 0.0)}
    for round_index in range(2):
        for name, interpreted in (("threads", False), ("children", True)):
            seed = 100 + round_index * 2 + interpreted  # a book of its own: the scanner's text caches
            book = bench.make_book(phone / "out" / f"{name}{round_index}" / "Book", chapters=80, words=2500, seed=seed)
            late, bursts, seconds = _ui_while(
                lambda: _scan(book, "quick-scan", _settings("quick-scan"), interpreted=interpreted, monkeypatch=monkeypatch),
                units)
            results[name][0].extend(late)
            results[name][1].extend(bursts)
            results[name] = (results[name][0], results[name][1], results[name][2] + seconds)
    report = {name: {"scan_s": round(values[2], 1), "late_p50": round(statistics.median(values[0]), 2),
                     "late_p95": round(bench.percentile(values[0], 0.95), 2),
                     "burst_p50": round(statistics.median(values[1]), 2),
                     "burst_p95": round(bench.percentile(values[1], 0.95), 2)} for name, values in results.items()}
    print("interpreter budget:", report)
    assert report["children"]["scan_s"] <= report["threads"]["scan_s"], report
    assert report["children"]["burst_p50"] <= report["threads"]["burst_p50"] * 1.15, report
    assert report["children"]["late_p95"] <= report["threads"]["late_p95"] * 1.5 + 2.0, report


@needs_interpreters
@pytest.mark.skipif((os.cpu_count() or 1) < 4, reason="needs cores to run at once")
def test_budget_children_really_run_at_once(tmp_path):
    """CPU work and nothing else, 1 child against 4, each round the same total: 4 children take well under the time of 1
    (threads would take the same). The loop-free check of the claim everything here rests on."""
    with interpreters.InterpreterPool(4, work_dir=str(tmp_path)) as made:
        def rounds(at_once: int) -> float:
            started = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=at_once) as outside:
                list(outside.map(lambda _n: made.call(WORKERS, "cpu", 400_000), range(8)))
            return time.perf_counter() - started

        rounds(4)  # all four started: the times below are the work's
        one, four = rounds(1), rounds(4)
    assert four <= one * 0.7, f"1 child {one:.2f} s, 4 children {four:.2f} s"


def test_the_full_loop_runs_a_coroutine_next_to_a_pool(tmp_path):
    """An asyncio loop keeps its timers while the pool's calls run on threads (the way the app's job runs them)."""
    if not interpreters.available():
        pytest.skip("no own-GIL interpreters")

    async def scenario():
        with interpreters.InterpreterPool(2, work_dir=str(tmp_path)) as made:
            future = asyncio.wrap_future(made.submit(WORKERS, "items", (20, 0.01, None), should_stop=lambda: False))
            ticks = 0
            while not future.done():
                await asyncio.sleep(0.01)
                ticks += 1
            assert (await future)["done"] == 20
            return ticks

    assert asyncio.run(scenario()) >= 3


# ---- output while a long call runs ------------------------------------------------------------------------------------


@needs_interpreters
def test_live_output_reaches_stdout_while_the_call_runs(tmp_path, capsys):
    gate = tmp_path / "go_on"
    made = interpreters.InterpreterPool(1, work_dir=str(tmp_path))
    seen: list = []

    class Stream:
        def write(self, text):
            seen.append(text)
            if "한글 first" in text:  # the call only ends once this line got here
                gate.write_bytes(b"")

        def flush(self):
            pass

    previous, sys.stdout = sys.stdout, Stream()
    try:
        result = made.call(WORKERS, "speak_until", str(gate), live_output=True)
    finally:
        sys.stdout = previous
        made.close()
    assert result == "done"
    assert "".join(seen).replace("\r\n", "\n") == "한글 first\nthen a piece without its line end"


@needs_interpreters
def test_output_files_do_not_stay_behind(tmp_path, pool, monkeypatch):
    monkeypatch.setattr(interpreters, "OUTPUT_POLL", 0.01)
    made = interpreters.InterpreterPool(1, work_dir=str(tmp_path))
    try:
        assert made.call(WORKERS, "speak", "x", live_output=True) == 1
        assert not [name for name in os.listdir(made._folder()) if name.endswith((".out", ".reply"))]
    finally:
        made.close()
    assert not os.listdir(tmp_path)


def test_a_line_is_not_passed_on_in_pieces(tmp_path, capsys):
    path = tmp_path / "out"
    path.write_bytes("whole\npie".encode("utf-8"))
    offset = interpreters._pass_output(str(path), 0)
    assert capsys.readouterr().out == "whole\n" and offset == len(b"whole\n")
    assert interpreters._pass_output(str(path), offset) == offset and not capsys.readouterr().out
    assert interpreters._pass_output(str(path), offset, everything=True) == path.stat().st_size
    assert capsys.readouterr().out == "pie"
    assert interpreters._pass_output(str(tmp_path / "none"), 3) == 3


# ---- one job's children ----------------------------------------------------------------------------------------------


@needs_interpreters
def test_job_children_decide_once_and_leave_when_asked(tmp_path, monkeypatch):
    lines = _Lines()
    monkeypatch.setattr(interpreters, "memory_now", lambda: (8 * 1024 ** 3, 4 * 1024 ** 3))
    job = interpreters.JobChildren(lines, "Working")
    first = job.pool(3)
    assert job.pool(3) is first and first.size == 3
    assert lines == [], "no child is there yet"
    assert first.call(WORKERS, "echo", 1) == 1
    assert lines == ["   ⚡ Working in 3 separate Python interpreters"]
    job.release()
    second = job.pool(1)
    assert second is not first and second.size == 1 and not job.gone
    assert second.call(WORKERS, "echo", 2) == 2
    assert len(lines) == 1, "the job says so once, however many times it asks for children"
    assert second.pool is first.pool and second.pool.started == 1 and second.pool.destroyed == 0
    job.close()
    assert job.pool(3) is None and job.gone and len(lines) == 1, "a closed job starts nothing and has nothing to say"


@needs_interpreters
def test_job_children_fall_back_once_when_memory_is_short(monkeypatch):
    lines = _Lines()
    monkeypatch.setattr(interpreters, "memory_now", lambda: (2 * 1024 ** 3, 310 * 1024 ** 2))
    job = interpreters.JobChildren(lines, "Working")
    for _attempt in range(3):
        assert job.pool(4) is None
    assert job.gone and len(lines) == 1 and "Working on threads" in lines[0] and "310 MB" in lines[0]


# ---- the glossary's smart filter -------------------------------------------------------------------------------------

_NAMES = ("김서연", "박지훈", "이수아", "최민준", "정하은", "강도윤", "윤서준", "장예은", "임시우", "한지민", "오현우", "서윤아")
_TITLES = ("공작", "황제", "기사단장", "마법사", "성녀", "대공")


def _korean_book(sentences: int, seed: int = 15) -> str:
    """Sentences with names, titles and honorifics, the shape the filter looks for."""
    import random

    rng = random.Random(seed)
    verbs = ("걸어갔다", "말했다", "바라보았다", "웃었다", "돌아섰다", "고개를 끄덕였다", "한숨을 쉬었다")
    places = ("왕궁", "던전", "숲속", "학원", "성벽", "시장", "도서관", "연구실")
    lines = []
    for number in range(sentences):
        name, other = rng.choice(_NAMES), rng.choice(_NAMES)
        title = rng.choice(_TITLES)
        lines.append(f"{name}{rng.choice(('님은', '씨가', '은', '이'))} {rng.choice(places)}에서 {other}에게 {title} 이야기를 하며 "
                     f"{rng.choice(verbs)}. 그는 {number % 97}번째 날을 떠올렸다.")
    return " ".join(lines)


@pytest.fixture
def glossary_phone(phone, monkeypatch):
    """The phone's comparison modules in the parent too: this PC's jellyfish does not load in a child, a phone has none."""
    import duplicate_detection_config as ddc
    from rapidfuzz.distance import JaroWinkler

    monkeypatch.setattr(ddc, "_HAS_JELLYFISH", False)
    monkeypatch.setattr(ddc, "_jellyfish", None)
    monkeypatch.setattr(ddc, "_rf_jaro_winkler", JaroWinkler)
    monkeypatch.setenv("GLOSSARY_STOP_FILE", str(phone / "glossary.stop"))
    for name in ("GLOSSARY_INCLUDE_GENDER_CONTEXT", "GLOSSARY_INCLUDE_ALL_CHARACTERS", "GRACEFUL_STOP_COMPLETED",
                 "EXTRACTION_WORKERS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(phone)
    yield phone


def _gm():
    import GlossaryManager

    return GlossaryManager


def _filtered(text: str, *, child: bool, lines: list = None) -> tuple:
    from glossarion_mobile.job_kinds.glossary import filter_in_child

    backend = _gm()
    if not child:
        return backend._filter_text_for_glossary(text, 2, 200)
    with filter_in_child((lines if lines is not None else []).append):
        return backend._filter_text_for_glossary(text, 2, 200)


def _child_ran(lines: list) -> bool:
    return any("separate Python interpreter" in line for line in lines) and not any("on threads" in line for line in lines)


@needs_interpreters
@needs_backend
@pytest.mark.parametrize("sentences, every", [(600, False), (2600, False), (600, True), (1400, True)])
def test_the_filter_gives_the_same_text_and_terms_in_a_child(glossary_phone, monkeypatch, capsys, sentences, every):
    """Through the real wiring, below and above the size where the shared function starts its own thread pool."""
    if every:
        monkeypatch.setenv("GLOSSARY_INCLUDE_ALL_CHARACTERS", "1")
    text = _korean_book(sentences)
    on_thread = _filtered(text, child=False)
    capsys.readouterr()
    lines: list = []
    in_child = _filtered(text, child=True, lines=lines)
    assert _child_ran(lines), lines
    assert in_child[0] == on_thread[0] and in_child[0]
    assert in_child[1] == on_thread[1] and len(in_child[1]) >= len(_NAMES)
    assert "FILTERING COMPLETE" in capsys.readouterr().out, "its log reaches the job's stdout"
    assert not interpreters.kept_pool()._busy and interpreters.kept_pool().live() == 1, "one child, free again"


@needs_interpreters
@needs_backend
def test_changed_settings_reach_the_next_filter(glossary_phone, monkeypatch, capsys):
    text = _korean_book(800)
    plain = _filtered(text, child=True)
    monkeypatch.setenv("GLOSSARY_INCLUDE_GENDER_CONTEXT", "1")
    monkeypatch.setenv("GLOSSARY_CONTEXT_WINDOW", "2")
    with_context = _filtered(text, child=True)
    assert with_context[0] != plain[0], "the setting did not reach the child"
    assert with_context[1] == _filtered(text, child=False)[1]


@needs_interpreters
@needs_backend
def test_with_the_gender_context_the_terms_are_the_same_in_a_child(glossary_phone, monkeypatch):
    """The text of this path differs between two runs on threads too: its pool merges batches as they finish (seen on the
    device, 3 different texts in 3 thread runs of one book). The terms do not."""
    monkeypatch.setenv("GLOSSARY_INCLUDE_GENDER_CONTEXT", "1")
    text = _korean_book(2600)
    lines: list = []
    in_child = _filtered(text, child=True, lines=lines)
    on_thread = _filtered(text, child=False)
    assert _child_ran(lines) and in_child[1] == on_thread[1] and in_child[0] and on_thread[0]


@needs_interpreters
@needs_backend
def test_a_short_text_never_starts_a_child(glossary_phone):
    from glossarion_mobile.job_kinds import glossary as glossary_kind

    text = _korean_book(40)
    assert len(text) < glossary_kind.FILTER_CHILD_MIN_CHARS
    lines: list = []
    assert _filtered(text, child=True, lines=lines) == _filtered(text, child=False)
    assert not lines


@needs_interpreters
@needs_backend
def test_a_filter_uses_one_kept_child_however_many_books_a_job_has(glossary_phone, capsys):
    """A translation runs for hours after its glossary's filter, and a destroyed child leaves its memory behind."""
    from glossarion_mobile.job_kinds.glossary import filter_in_child

    backend = _gm()
    text = _korean_book(800)
    lines: list = []
    with filter_in_child(lines.append):
        assert backend._filter_text_for_glossary(text, 2, 200)[0]
        assert _alive() == 2 and not interpreters.kept_pool()._busy, "it stays, and nothing runs in it now"
        assert backend._filter_text_for_glossary(text, 2, 200)[0], "the next book's filter gets the same one"
        assert _alive() == 2
    assert _child_ran(lines) and len(lines) == 1, lines
    assert interpreters.kept_pool().started == 1 and interpreters.kept_pool().destroyed == 0


@needs_interpreters
@needs_backend
def test_a_filter_child_does_not_load_the_api_client(tmp_path, glossary_phone):
    """``glossary_child`` blocks what ``GlossaryManager`` would import for names the filter does not use. On the device the
    API client's native modules are refused; here most of them load, so look at what is in the child."""
    made = interpreters.InterpreterPool(1, preload=("glossarion_mobile.services.glossary_child",), work_dir=str(tmp_path))
    try:
        loaded = made.call(WORKERS, "loaded", ["GlossaryManager", "unified_api_client", "extract_glossary_from_epub", "requests",
                                               "tiktoken", "PySide6", "numpy", "lxml.etree"])
        blocked = made.call(WORKERS, "blocked", ["extract_glossary_from_epub", "unified_api_client", "GlossaryManager"])
    finally:
        made.close()
    assert loaded == {"GlossaryManager": True, "unified_api_client": False, "extract_glossary_from_epub": False,
                      "requests": False, "tiktoken": False, "PySide6": False, "numpy": False, "lxml.etree": False}
    assert blocked == ["extract_glossary_from_epub", "unified_api_client"], "this holds where a child could load tiktoken too"


@needs_interpreters
@needs_backend
def test_the_filter_is_put_back_and_the_switch_turns_it_off(glossary_phone, monkeypatch):
    from glossarion_mobile.job_kinds import glossary as glossary_kind

    backend = _gm()
    real = backend._filter_text_for_glossary
    with glossary_kind.filter_in_child():
        assert backend._filter_text_for_glossary is not real
    assert backend._filter_text_for_glossary is real
    monkeypatch.setattr(glossary_kind, "FILTER_IN_CHILD", False)
    with glossary_kind.filter_in_child():
        assert backend._filter_text_for_glossary is real


def _long_filter(backend: Any, seen: list, started: threading.Event) -> threading.Thread:
    """A filter that takes seconds, on a thread of its own, the ``started`` event set when its first line is out."""
    from glossarion_mobile.job_kinds.glossary import filter_in_child

    class Watch:
        def write(self, text):
            if "Step 5" in text:
                started.set()

        def flush(self):
            pass

    def run():
        previous, sys.stdout = sys.stdout, Watch()
        try:
            with filter_in_child():
                seen.append(backend._filter_text_for_glossary(_korean_book(30000), 2, 200))
        finally:
            sys.stdout = previous

    thread = threading.Thread(target=run)
    thread.start()
    return thread


@needs_interpreters
@needs_backend
@pytest.mark.parametrize("how", ["stop_flag", "graceful_env"])
def test_stop_ends_a_filter_in_a_child_like_on_a_thread(glossary_phone, how):
    backend = _gm()
    seen: list = []
    started = threading.Event()
    thread = _long_filter(backend, seen, started)
    try:
        assert started.wait(60), "the filter never logged"
        at = time.monotonic()
        if how == "stop_flag":
            backend.set_stop_flag(True)
        else:
            os.environ["GRACEFUL_STOP"] = "1"
        thread.join(30)
        assert not thread.is_alive() and time.monotonic() - at < 5
    finally:
        backend.set_stop_flag(False)
        os.environ.pop("GRACEFUL_STOP", None)
        thread.join(60)
    assert seen == [("", [])], "a stopped filter returns what the shared decorator returns"


@needs_interpreters
@needs_backend
def test_a_worker_exception_in_the_filter_surfaces(glossary_phone):
    """A wrong argument fails in the same place on both paths, with the child's traceback behind it."""
    from glossarion_mobile.job_kinds.glossary import filter_in_child

    backend = _gm()
    text = _korean_book(800)
    with pytest.raises(TypeError) as on_thread:
        backend._filter_text_for_glossary(text, "two", 200)
    with filter_in_child(), pytest.raises(TypeError) as in_child:
        backend._filter_text_for_glossary(text, "two", 200)
    assert str(in_child.value) == str(on_thread.value)
    assert "GlossaryManager" in str(in_child.value.__cause__)


@needs_interpreters
@needs_backend
@pytest.mark.parametrize("how", ["no_child", "cannot_prepare", "no_memory", "compares_differently"])
def test_where_a_child_does_not_work_the_filter_runs_here_with_the_same_result(glossary_phone, monkeypatch, how):
    import duplicate_detection_config as ddc

    text = _korean_book(800)
    expected = _filtered(text, child=False)
    if how == "no_child":
        monkeypatch.setattr(interpreters, "_create", lambda: (_ for _ in ()).throw(RuntimeError("no room")))
    elif how == "cannot_prepare":
        monkeypatch.setattr(interpreters, "_child_source", lambda: "raise ImportError('not on this phone')\n")
    elif how == "no_memory":
        monkeypatch.setattr(interpreters, "memory_now", lambda: (2 * 1024 ** 3, 100 * 1024 ** 2))
    else:
        monkeypatch.setattr(ddc, "_HAS_JELLYFISH", True)
    lines: list = []
    assert _filtered(text, child=True, lines=lines) == expected
    back = [line for line in lines if "on threads" in line]
    assert len(back) == 1, lines
    assert _alive() == (2 if how == "compares_differently" else 1)  # that one works: it stays for the next job


@needs_interpreters
@needs_backend
def test_the_filter_tries_a_failing_child_once_in_a_job(glossary_phone, monkeypatch):
    from glossarion_mobile.job_kinds.glossary import filter_in_child

    attempts: list = []

    def no_child():
        attempts.append(1)
        raise RuntimeError("no")

    monkeypatch.setattr(interpreters, "_create", no_child)
    lines: list = []
    backend = _gm()
    text = _korean_book(800)
    with filter_in_child(lines.append):
        for _book in range(3):
            assert backend._filter_text_for_glossary(text, 2, 200)[0]
    assert len(attempts) == 1 and len([line for line in lines if "on threads" in line]) == 1


def test_nothing_the_filter_reaches_needs_what_a_child_does_not_load():
    """Read from the source: the whole function, what it calls in its module, and that it asks for html.parser by name."""
    source = (SRC_DIR / "GlossaryManager.py").read_text(encoding="utf-8-sig")
    tree = ast.parse(source)
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    reached, stack = set(), ["_filter_text_for_glossary", "_check_filter_stop", "_process_sentence_batch_for_extraction",
                             "_score_sentence_batch"]
    while stack:
        name = stack.pop()
        if name in reached or name not in functions:
            continue
        reached.add(name)
        stack.extend({node.id for node in ast.walk(functions[name]) if isinstance(node, ast.Name)})
    reached.discard("is_stop_requested")  # replaced in a child (``glossary_child.filter_text``)
    reached.discard("set_stop_flag")
    assert {"_html_soup", "_has_honorific", "_filter_completed", "_filter_executor"} <= reached
    assert len(reached) > 8
    absent = {"extract_glossary_from_epub", "unified_api_client", "TransateKRtoEN", "tiktoken", "lxml", "numpy", "regex"}
    allowed = {"os", "re", "collections", "concurrent", "time", "multiprocessing", "functools", "duplicate_detection_config",
               "bs4", "warnings"}
    for name in sorted(reached):
        for node in ast.walk(functions[name]):
            if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "BeautifulSoup":
                assert [arg.value for arg in node.args[1:]] == ["html.parser"], f"{name}:{node.lineno}"
            if isinstance(node, ast.Import):
                assert {alias.name.split(".")[0] for alias in node.names} <= allowed, f"{name}:{node.lineno}"
            if isinstance(node, ast.ImportFrom):
                assert (node.module or "").split(".")[0] in allowed, f"{name}:{node.lineno}"
            if isinstance(node, ast.Name):
                assert node.id not in absent and node.id not in ("DEFAULT_AUTO_GLOSARY_PROMPT3", "_apply_subject_tracking_placeholder"), \
                    f"{name}:{node.lineno} uses {node.id}, which a child does not have"
            if isinstance(node, ast.keyword) and node.arg == "daemon":
                raise AssertionError(f"{name}:{node.lineno}: a child refuses daemon threads")


@needs_backend
def test_rapidfuzz_python_scores_like_its_cpp_on_names():
    """A child gets RapidFuzz's pure Python (its C++ is refused on the device), and the filter compares names with it."""
    import random

    import duplicate_detection_config as ddc
    from rapidfuzz import fuzz_cpp, fuzz_py
    from rapidfuzz.distance import JaroWinkler, JaroWinkler_py

    rng = random.Random(4)
    pool = [a + b for a in _NAMES for b in ("", "님", " 공작", "씨", "Sir ", "-sama")] + ["Lee So-jeong", "Yoo So-jeong", "Mikiyo", "Miki"]
    for _pair in range(4000):
        a, b = rng.choice(pool), rng.choice(pool)
        if rng.random() < 0.3:
            b = a[:rng.randrange(len(a) + 1)]
        for name in ("ratio", "token_sort_ratio", "token_set_ratio", "partial_ratio"):
            assert getattr(fuzz_py, name)(a, b) == pytest.approx(getattr(fuzz_cpp, name)(a, b), abs=1e-9), (name, a, b)
        assert JaroWinkler_py.normalized_similarity(a, b) == pytest.approx(JaroWinkler.normalized_similarity(a, b), abs=1e-9)
    assert ddc._HAS_RAPIDFUZZ


@pytest.mark.parametrize("path, entries", [
    ("job_kinds/translate.py", "filter_in_child()"),  # every translation: the auto glossary of a book, Vision, RPG Maker...
    ("job_kinds/glossary.py", "with filter_in_child():"),
])
def test_every_job_that_reaches_the_filter_goes_through_the_child(path, entries):
    source = (APP_DIR / "glossarion_mobile" / path).read_text(encoding="utf-8")
    assert entries in source
    if path.endswith("glossary.py"):
        assert source.count("with filter_in_child():") == 2, "Extract Glossary and the Parallel EPUB pair"


# ---- Generate MD / TXT -----------------------------------------------------------------------------------------------


def _english_outputs(folder: Path, count: int = 24) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    paragraph = ("The knight <b>walked</b> into the hall and greeted everyone - 1. warmly, while the *rain* fell over the old "
                 "castle walls &amp; the merchants argued. ")
    for number in range(count):
        (folder / f"response_{number:03d}.html").write_text(
            f"<html><head><title>Chapter {number}</title></head><body><h1>Chapter {number}</h1>"
            f"<p>{paragraph * (3 + number % 5)}</p><ul><li>one {number}</li></ul><p>&lt;Skill&gt; [Status]</p></body></html>",
            encoding="utf-8")


def _korean_outputs(folder: Path, count: int = 6) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for number in range(count):
        (folder / f"response_{number:03d}.html").write_text(
            f"<h1>제{number}화</h1><p>{'기사는 홀로 걸어 들어가 모두에게 인사했다. ' * 12}</p><p>&lt;스킬&gt;</p>", encoding="utf-8")


def _tree(folder: Path) -> dict:
    return {str(path.relative_to(folder)): path.read_bytes() for path in sorted(folder.rglob("*"))
            if path.is_file() and path.suffix in (".md", ".txt")}


def _generate(folder: Path, *, children: bool, should_stop=None) -> tuple:
    import md_txt_sidecar_writer as writer

    from glossarion_mobile.job_kinds.compile import SIDECARS_CHILD_MIN_FILES, sidecars_in_children

    lines: list = []
    with sidecars_in_children(lines.append, SIDECARS_CHILD_MIN_FILES if children else 0):
        result = writer.generate_md_txt_for_output_dir(str(folder), do_md=True, do_txt=True, max_workers=4, log=lines.append,
                                                       should_stop=should_stop)
    return result, lines


@needs_interpreters
@needs_backend
@pytest.mark.parametrize("make", [_english_outputs, _korean_outputs])
def test_sidecars_are_the_same_files_in_children(phone, capfd, make):
    """Korean chapters go through lxml on a thread (a child cannot load it): they come back to the thread, without a
    failure in the log or the extractor's own failure line in the job's output."""
    make(phone / "threads")
    make(phone / "children")
    on_threads, _lines = _generate(phone / "threads", children=False)
    capfd.readouterr()
    in_children, lines = _generate(phone / "children", children=True)
    assert in_children == on_threads and on_threads[1] == 0 and on_threads[0] == on_threads[2]
    assert _tree(phone / "children") == _tree(phone / "threads") and _tree(phone / "threads")
    assert any("separate Python interpreter" in line for line in lines), lines
    assert not any("⚠" in line or "on threads" in line for line in lines), lines
    assert "Enhanced extraction failed" not in capfd.readouterr().out
    assert not interpreters.kept_pool()._busy and interpreters.kept_pool().destroyed == 0


@needs_interpreters
@needs_backend
def test_a_book_in_korean_stops_asking_children(phone, monkeypatch):
    import interp_workers  # noqa: F401

    from glossarion_mobile.job_kinds import compile as compile_kind

    monkeypatch.setattr(compile_kind, "SIDECARS_CHILD_MIN_FILES", 1)
    calls: list = []
    real_call = interpreters.InterpreterPool.call

    def counting(self, module, function, *args, **kwargs):
        calls.append(function)
        return real_call(self, module, function, *args, **kwargs)

    monkeypatch.setattr(interpreters.InterpreterPool, "call", counting)
    _korean_outputs(phone / "book", 30)
    result, _lines = _generate(phone / "book", children=True)
    assert result == (30, 0, 30) and len(_tree(phone / "book")) == 60
    assert len(calls) <= compile_kind.SIDECAR_LXML_LIMIT + compile_kind.SIDECAR_WORKERS, len(calls)


@needs_interpreters
@needs_backend
def test_a_small_folder_stays_on_the_threads(phone):
    _english_outputs(phone / "book", 4)
    from glossarion_mobile.job_kinds.compile import sidecars_in_children

    import md_txt_sidecar_writer as writer

    real = writer._write_html_md_txt_sidecars
    with sidecars_in_children(print, 4):
        assert writer._write_html_md_txt_sidecars is real


@needs_interpreters
@needs_backend
def test_stop_between_sidecars_as_on_the_threads(phone):
    _english_outputs(phone / "book", 40)
    written: list = []
    import md_txt_sidecar_writer as writer

    def stop() -> bool:
        written.append(1)
        return len(written) > 10

    (done, failed, total), lines = _generate(phone / "book", children=True, should_stop=stop)
    assert total == 40 and done < 40 and not failed
    assert any("Stop requested" in line for line in lines), lines
    assert writer._write_html_md_txt_sidecars.__name__ == "_write_html_md_txt_sidecars", "the writer is put back"
    assert interpreters.kept_pool().destroyed == 0


@needs_interpreters
@needs_backend
def test_a_file_that_fails_in_a_child_fails_like_on_a_thread(phone):
    """A file named MD where the folder should go: every chapter fails to write, and says why, on both paths."""
    for name in ("threads", "children"):
        _english_outputs(phone / name)
        (phone / name / "MD").write_text("in the way", encoding="utf-8")
    on_threads, thread_lines = _generate(phone / "threads", children=False)
    in_children, child_lines = _generate(phone / "children", children=True)
    assert on_threads == in_children and on_threads[0] == 0 and on_threads[1] == on_threads[2]
    failed = [line for line in child_lines if "⚠" in line]
    assert len(failed) == on_threads[1] == len([line for line in thread_lines if "⚠" in line])


@needs_interpreters
@needs_backend
@pytest.mark.parametrize("how", ["no_child", "no_memory"])
def test_where_children_do_not_work_the_sidecars_are_the_same(phone, monkeypatch, how):
    _english_outputs(phone / "threads")
    _english_outputs(phone / "fallback")
    _generate(phone / "threads", children=False)
    attempts: list = []
    if how == "no_child":
        def no_child():
            attempts.append(1)
            raise RuntimeError("no room")

        monkeypatch.setattr(interpreters, "_create", no_child)
    else:
        monkeypatch.setattr(interpreters, "memory_now", lambda: (2 * 1024 ** 3, 100 * 1024 ** 2))
    result, lines = _generate(phone / "fallback", children=True)
    assert result[1] == 0 and _tree(phone / "fallback") == _tree(phone / "threads")
    assert len([line for line in lines if "on threads" in line]) == 1, lines
    assert len(attempts) <= 4, "each worker of the job's first files may have tried once; none tries again"


@needs_interpreters
@needs_backend
def test_the_md_txt_job_goes_through_the_children(phone, monkeypatch):
    from glossarion_mobile.job_kinds import compile as compile_kind

    _english_outputs(phone / "out" / "Book", compile_kind.SIDECARS_CHILD_MIN_FILES)
    monkeypatch.setattr(compile_kind, "resolve_folders", lambda _ctx: [str(phone / "out" / "Book")])
    ctx = _Ctx({})
    ctx.params = {"format": "md"}
    ctx.phase = lambda *_a, **_k: None
    assert compile_kind.run_md_txt(ctx)["ok"] is True
    assert _went_to_children(ctx.lines), ctx.lines
    assert ctx.results["md_txt"]["ok"] == compile_kind.SIDECARS_CHILD_MIN_FILES
    assert len(list((phone / "out" / "Book" / "MD").glob("*.md"))) == compile_kind.SIDECARS_CHILD_MIN_FILES


@needs_interpreters
@needs_backend
def test_the_smoke_check_covers_the_glossary_filter_and_the_sidecars(tmp_path, glossary_phone):
    from glossarion_mobile.diagnostics import interpreter_check

    detail = interpreter_check.check_scan_in_child(tmp_path / "work", "android")
    assert detail["result"] == "pass"
    assert detail["filter"] == "pass" and detail["sidecars"] == "pass", detail


# ---- kept children: a destroyed one does not give its memory back -------------------------------------------------


def _job(lines=None, what="Working", **options):
    return interpreters.JobChildren(lines if lines is not None else _Lines(), what, **options)


@pytest.fixture
def plenty_of_memory(monkeypatch):
    monkeypatch.setattr(interpreters, "memory_now", lambda: (8 * 1024 ** 3, 4 * 1024 ** 3))


@needs_interpreters
def test_a_job_keeps_its_children_for_the_next_job_and_destroys_none(plenty_of_memory):
    """Emulator, Python 3.13: each destroyed child left about 17 MB in the process for good, 9 scans at 4 workers took the
    app from 489 to 1178 MB. 4 children reused for 12 rounds stayed flat."""
    before = _alive()
    seen = set()
    for _job_number in range(5):
        job = _job()
        pool = job.pool(1)
        seen.add(pool.call(WORKERS, "where"))
        job.close()
    kept = interpreters.kept_pool()
    assert len(seen) == 1 and kept.started == 1 and kept.destroyed == 0
    assert _alive() == before + 1 and kept.live() == 1


@needs_interpreters
def test_the_kept_children_never_outgrow_what_a_job_asks(plenty_of_memory):
    for wanted in (3, 1, 2, 3):
        job = _job()
        pool = job.pool(wanted)
        with concurrent.futures.ThreadPoolExecutor(max_workers=wanted) as outside:
            assert sorted(outside.map(lambda n: pool.call(WORKERS, "echo", n), range(wanted * 4))) == list(range(wanted * 4))
        job.close()
    kept = interpreters.kept_pool()
    assert kept.started <= 3 and kept.destroyed == 0


@needs_interpreters
def test_the_memory_rule_counts_only_children_that_are_new(monkeypatch):
    free = [4 * 1024 ** 3]
    monkeypatch.setattr(interpreters, "memory_now", lambda: (8 * 1024 ** 3, free[0]))
    job = _job()
    pool = job.pool(2)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as outside:
        list(outside.map(lambda n: pool.call(WORKERS, "sleep_then", (0.2, n)), range(4)))
    job.close()
    kept = interpreters.kept_pool()
    assert kept.live() == 2
    free[0] = 100 * 1024 ** 2  # no room for one more: the two that exist cost nothing more
    job = _job()
    pool = job.pool(4)
    assert pool is not None and not job.gone and pool.size == 2
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as outside:
        list(outside.map(lambda n: pool.call(WORKERS, "echo", n), range(8)))
    job.close()
    assert kept.started == 2 and kept.size == 2
    free[0] = 2 * 1024 ** 3  # room again: a later job may have more
    job = _job()
    assert job.pool(4).size == 4
    job.close()


@needs_interpreters
def test_interpreters_are_created_one_at_a_time(tmp_path, monkeypatch):
    """Created at the same moment, one failed now and then on Python 3.13 (CI): and that decides for as long as the app runs."""
    real = interpreters._create
    state = {"now": 0, "most": 0}
    guard = threading.Lock()

    def counted():
        with guard:
            state["now"] += 1
            state["most"] = max(state["most"], state["now"])
        time.sleep(0.1)
        try:
            return real()
        finally:
            with guard:
                state["now"] -= 1

    monkeypatch.setattr(interpreters, "_create", counted)
    with interpreters.InterpreterPool(3, work_dir=str(tmp_path)) as made:
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as outside:
            assert sorted(outside.map(lambda n: made.call(WORKERS, "sleep_then", (0.3, n)), range(3))) == [0, 1, 2]
        assert made.started == 3
    assert state["most"] == 1


@needs_interpreters
def test_a_child_that_failed_means_threads_for_the_rest_of_this_run_of_the_app(plenty_of_memory, monkeypatch):
    """No second try at something that failed, and a failing child costs its memory for good: each job would leak another."""
    attempts: list = []
    real = interpreters._create

    def no_child():
        attempts.append(1)
        raise RuntimeError("no room")

    monkeypatch.setattr(interpreters, "_create", no_child)
    for _job_number in range(3):
        lines = _Lines()
        job = _job(lines)
        pool = job.pool(4)
        if pool is not None:
            with pytest.raises(interpreters.InterpreterUnavailable):
                pool.call(WORKERS, "echo", 1)
            job.fall_back("no room")
        assert job.pool(4) is None and sum("on threads" in line for line in lines) == 1, lines
        job.close()
    assert len(attempts) == 1
    monkeypatch.setattr(interpreters, "_create", real)
    assert _job().pool(1) is None, "a later job does not try again"


@needs_interpreters
def test_a_child_that_dies_in_a_call_is_not_replaced(plenty_of_memory, monkeypatch):
    real_run = interpreters._run
    monkeypatch.setattr(interpreters, "_run", lambda child, code, shared=None: (
        "SystemError: the call failed" if code == "_gl_call()" else real_run(child, code, shared)))
    job = _job()
    with pytest.raises(interpreters.InterpreterUnavailable):
        job.pool(1).call(WORKERS, "echo", 1)
    job.close()
    assert _job().pool(1) is None
    assert interpreters.kept_pool().started == 1


@needs_interpreters
def test_arguments_that_cannot_cross_do_not_end_the_children_for_later_jobs(plenty_of_memory):
    job = _job()
    with pytest.raises(interpreters.InterpreterUnavailable):
        job.pool(1).call(WORKERS, "echo", threading.Lock())
    job.close()
    job = _job()
    assert job.pool(1).call(WORKERS, "echo", 1) == 1
    job.close()
    assert interpreters.kept_pool().destroyed == 0


@needs_interpreters
def test_the_interpreters_line_comes_only_when_a_child_is_there(plenty_of_memory, monkeypatch):
    """The emulator's forced fallback logged "Scanning in 4 separate Python interpreters" and then that it could not."""
    monkeypatch.setattr(interpreters, "_create", lambda: (_ for _ in ()).throw(RuntimeError("no room")))
    lines = _Lines()
    job = _job(lines, "Scanning")
    pool = job.pool(4)
    if pool is not None:
        with pytest.raises(interpreters.InterpreterUnavailable):
            pool.call(WORKERS, "echo", 1)
    assert not any("separate Python interpreter" in line for line in lines), lines
    job.close()


@needs_interpreters
def test_a_job_that_ended_is_not_called_and_stops_only_its_own_batches(plenty_of_memory):
    first, second = _job(), _job()
    one, two = first.pool(2), second.pool(2)
    stop_one, stop_two = (lambda: False), (lambda: False)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as outside:
        running_one = outside.submit(one.call, WORKERS, "items", (400, 0.01, None), should_stop=stop_one)
        running_two = outside.submit(two.call, WORKERS, "items", (40, 0.01, None), should_stop=stop_two)
        time.sleep(0.4)
        first.close()
        assert running_two.result(timeout=60) == {"stopped_at": None, "done": 40}, "the other job's batch was cut short"
        assert running_one.result(timeout=60)["stopped_at"] is not None, "the ended job's batch was not told"
    with pytest.raises(interpreters.InterpreterUnavailable):
        one.call(WORKERS, "echo", 1)
    second.close()


@needs_interpreters
def test_nothing_of_a_finished_job_stays_in_the_kept_pool(plenty_of_memory):
    job = _job()
    pool = job.pool(1)
    assert pool.call(WORKERS, "echo_checked", "x", should_stop=lambda: False) == "x"
    job.close()
    kept = interpreters.kept_pool()
    assert kept._watched == {}, "each job's stop check would stay, and be asked"
    assert os.listdir(kept._dir) == [], "a job's files stay behind"


@needs_interpreters
def test_after_call_and_preload_belong_to_the_job_not_to_the_child(plenty_of_memory):
    first = _job(after_call=(WORKERS, "clear_marker"))
    first.pool(1).call(WORKERS, "echo", 1)
    first.close()
    assert interpreters.kept_pool().call(WORKERS, "get_env", ["INTERP_AFTER_CALL"]) == {"INTERP_AFTER_CALL": "1"}
    second = _job(preload=("colorsys",))
    assert second.pool(1).call(WORKERS, "loaded", ["colorsys"]) == {"colorsys": True}
    second.close()


@needs_interpreters
def test_a_module_a_child_cannot_load_sends_that_work_to_threads_and_keeps_the_child(plenty_of_memory):
    """A failed import leaves the interpreter as it was. One kind of job the device cannot do in a child does not take
    the children from the others, and it is not tried again."""
    missing = ("no_such_module_anywhere",)
    job = _job(preload=missing)
    pool = job.pool(1)
    with pytest.raises(interpreters.InterpreterUnavailable, match="no_such_module_anywhere"):
        pool.call(WORKERS, "echo", 1)
    job.fall_back("no")
    job.close()
    kept = interpreters.kept_pool()
    assert kept.started == 1 and kept.destroyed == 0 and kept._busy == 0
    assert _job(preload=missing).pool(1) is None, "not tried again"
    other = _job()
    assert other.pool(1).call(WORKERS, "echo", 2) == 2, "other work keeps its child"
    other.close()
    assert kept.started == 1


@needs_interpreters
def test_a_call_of_a_job_that_ended_while_it_waited_does_not_run(plenty_of_memory):
    """The next job should not wait behind a batch of a job that is over."""
    job = _job()
    use = job.pool(1)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as outside:
        first = outside.submit(use.call, WORKERS, "sleep_then", (0.6, "first"))
        time.sleep(0.2)
        waiting = outside.submit(use.call, WORKERS, "echo", "second")
        time.sleep(0.1)
        job.close()
        assert first.result(timeout=30) == "first"
        with pytest.raises(interpreters.InterpreterUnavailable):
            waiting.result(timeout=30)
    assert interpreters.kept_pool()._busy == 0 and interpreters.kept_pool().destroyed == 0


@needs_interpreters
def test_a_new_child_setup_does_not_reach_kept_children_so_they_go(plenty_of_memory, tmp_path, monkeypatch):
    """An E2E run's tripwires are in each child that starts: one kept from before the run would have none."""
    job = _job()
    job.pool(1).call(WORKERS, "echo", 1)
    job.close()
    old = interpreters.kept_pool()
    interpreters.set_child_setup((WORKERS, "clear_marker"))
    try:
        assert interpreters.kept_pool() is not old and old.live() == 0
        job = _job()
        job.pool(1).call(WORKERS, "echo", 1)
        job.close()
    finally:
        interpreters.set_child_setup(None)
    assert interpreters.kept_pool().live() == 0


@needs_interpreters
def test_the_folder_comes_back_when_the_system_cleared_it(plenty_of_memory):
    """A phone clears an app's cache and temp folders when it wants to, and children are kept for long."""
    job = _job()
    pool = job.pool(1)
    assert pool.call(WORKERS, "echo_checked", "x", should_stop=lambda: False) == "x"
    shutil.rmtree(interpreters.kept_pool()._dir)
    assert pool.call(WORKERS, "echo_checked", "y", should_stop=lambda: False) == "y"
    job.close()
    assert interpreters.kept_pool().destroyed == 0


def test_a_folder_a_killed_app_left_behind_goes_at_the_next_start(tmp_path, monkeypatch):
    old, new, other = tmp_path / "glossarion-interp-old", tmp_path / "glossarion-interp-new", tmp_path / "other-old"
    for folder in (old, new, other):
        folder.mkdir()
    (old / "7.reply").write_bytes(b"a chapter's result")
    long_ago = time.time() - 2 * interpreters.LEFTOVER_AGE
    for folder in (old, other):
        os.utime(folder, (long_ago, long_ago))
    interpreters._remove_leftovers(str(tmp_path))
    assert not old.exists() and new.exists() and other.exists()


# ---- what a review of the children found --------------------------------------------------------------------------


@needs_interpreters
@pytest.mark.parametrize("should_stop", [None, lambda: False])
def test_no_room_for_the_pools_files_is_unavailable_and_frees_the_child(tmp_path, monkeypatch, should_stop):
    """It raised OSError past the code that frees the child: the next batches then waited for it, until a Stop."""
    made = interpreters.InterpreterPool(1, work_dir=str(tmp_path))
    real = tempfile.mkdtemp
    monkeypatch.setattr(tempfile, "mkdtemp", lambda **_kw: (_ for _ in ()).throw(OSError(28, "No space left on device")))
    try:
        for _attempt in range(2):
            with pytest.raises(interpreters.InterpreterUnavailable, match="No space left"):
                made.call(WORKERS, "echo", 1, should_stop=should_stop)
            assert made._busy == 0
        monkeypatch.setattr(tempfile, "mkdtemp", real)
        assert made.call(WORKERS, "echo", 2) == 2
        assert made.started == 1 and made.destroyed == 0
    finally:
        made.close()


class _EnvironThatChanges(dict):
    """``os.environ`` while another thread takes a key out of it (``remote_routes`` from the UI, during a scan)."""

    def __iter__(self):
        return iter(list(self.keys()) + ["GONE_WHILE_READING"])

    def keys(self):
        return list(super().keys()) + ["GONE_WHILE_READING"]


@needs_interpreters
def test_a_key_that_goes_while_the_environment_is_read_is_not_an_error(pool, monkeypatch):
    """``dict(os.environ)`` raised KeyError 113 times in 296,362 calls with a thread deleting a key 10 times a second."""
    monkeypatch.setenv("INTERP_KEEP", "one")
    environ = _EnvironThatChanges(os.environ)
    monkeypatch.setattr(interpreters.os, "environ", environ)
    assert pool.call(WORKERS, "get_env", ["INTERP_KEEP", "GONE_WHILE_READING"]) == {"INTERP_KEEP": "one", "GONE_WHILE_READING": None}


@needs_interpreters
def test_a_batch_a_closed_pool_let_finish_still_hears_a_stop(tmp_path):
    """``_watch_loop`` ended when the pool closed: 4.84 s past the Stop in the review's probe, to the batch's end."""
    made = interpreters.InterpreterPool(1, work_dir=str(tmp_path))
    told = threading.Event()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as outside:
        running = outside.submit(made.call, WORKERS, "items", (500, 0.01, None), should_stop=told.is_set)
        time.sleep(0.5)
        made.close(stop_running=False)
        told.set()
        assert running.result(timeout=60)["stopped_at"] is not None


@needs_interpreters
def test_a_batch_that_starts_as_the_pool_closes_is_not_cut_short(tmp_path, monkeypatch):
    """``close(stop_running=False)``: this batch's result counts. ``_watch`` told each call of a closed pool to stop, and
    its partial result looks like a finished one."""
    made = interpreters.InterpreterPool(1, work_dir=str(tmp_path))
    real_watch = made._watch

    def closed_first(should_stop):
        made.close(stop_running=False)
        return real_watch(should_stop)

    monkeypatch.setattr(made, "_watch", closed_first)
    assert made.call(WORKERS, "items", (20, 0.01, None), should_stop=lambda: False) == {"stopped_at": None, "done": 20}


def test_the_smoke_check_imports_the_scanner_here_before_a_child_does():
    """A single-phase native module a child loads first is refused there and can corrupt the heap when this interpreter
    loads it later (host 3.12, ``_decimal``: 0xc0000374 in ``create_module``)."""
    source = (APP_DIR / "glossarion_mobile" / "diagnostics" / "interpreter_check.py").read_text(encoding="utf-8")
    assert source.index("import scan_html_folder") < source.index("interpreters.InterpreterPool(")


@needs_backend
def test_no_module_the_scanner_loads_needs_what_a_child_refuses():
    """``test_nothing_a_scan_worker_reaches_changes_in_a_child`` reads ``scan_html_folder.py``: this reads the modules of
    this repo it imports when it loads, and what those import when they load."""
    refused = {"lxml", "regex", "numpy", "charset_normalizer", "pydantic_core", "cryptography", "tiktoken", "PIL", "cv2",
               "onnxruntime", "fitz", "pymupdf", "grpc", "llama_cpp", "ebooklib"}

    def at_load(tree):
        """Imports that run when the module loads: not those in a function."""
        stack = list(tree.body)
        while stack:
            node = stack.pop()
            if isinstance(node, ast.Import):
                yield from (alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                yield node.module.split(".")[0]
            elif not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                stack.extend(ast.iter_child_nodes(node))

    seen, todo, found = set(), ["scan_html_folder"], {}
    while todo:
        name = todo.pop()
        path = SRC_DIR / f"{name}.py"
        if name in seen or not path.exists():
            continue
        seen.add(name)
        for module in at_load(ast.parse(path.read_text(encoding="utf-8-sig"))):
            todo.append(module)
            if module in refused:
                found.setdefault(name, set()).add(module)
    assert len(seen) >= 2
    assert not found, found


# ---- kept children: the real work -----------------------------------------------------------------------------------


@needs_interpreters
@needs_backend
def test_changed_settings_reach_the_kept_children_of_the_next_job(phone, monkeypatch):
    """Each scan its own job, as on the phone: the children stay, with what the scan before read and imported. The
    settings change through the arguments and the env, and the scan finds what threads find."""
    import scan_html_folder as shf

    from glossarion_mobile.job_kinds import qa as qa_kind

    def rounds(name: str, interpreted: bool) -> list:
        monkeypatch.setattr(qa_kind, "SCAN_INTERPRETERS", interpreted)
        out = []
        for index, (settings, env) in enumerate((
                (_settings("quick-scan"), {"SKIP_TITLE_TAG_TRANSLATION": "1"}),
                (_settings("quick-scan", check_translation_artifacts=False, check_ai_artifacts=False,
                           foreign_char_threshold=10_000, check_missing_tag_bracket=False),
                 {"SKIP_TITLE_TAG_TRANSLATION": "0", "QA_MIN_DUPLICATE_WORD_COUNT": "100000"}),
                (_settings("quick-scan"), {"SKIP_TITLE_TAG_TRANSLATION": "1", "QA_MIN_DUPLICATE_WORD_COUNT": None}))):
            for key, value in env.items():
                if value is None:
                    monkeypatch.delenv(key, raising=False)
                else:
                    monkeypatch.setenv(key, value)
            book = _book(phone / "out" / name / str(index) / "Book", seed=9151)
            first = os.path.join(book, "response_0002_ch0002.html")
            with open(first, encoding="utf-8") as handle:
                text = handle.read()
            with open(first, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(text.replace("<title>", "<title>안녕하세요 안녕하세요 ", 1))
            lines = _Lines()
            with qa_kind._scan_turns(True, lines):
                shf.clear_qa_caches()
                shf.scan_html_folder(book, log=lines, mode="quick-scan", qa_settings=settings)
            assert _went_to_children(lines) is interpreted, lines[:8]
            out.append(_report_rows(book))
        return out

    on_threads = rounds("threads", False)
    in_children = rounds("children", True)
    assert on_threads[0] != on_threads[1] and on_threads[0] == on_threads[2], "the change changed nothing: this proves nothing"
    assert in_children == on_threads
    kept = interpreters.kept_pool()
    assert kept.destroyed == 0 and 1 <= kept.started <= 4


@needs_interpreters
@needs_backend
def test_a_stopped_scan_leaves_the_next_one_whole_on_the_kept_children(phone, monkeypatch):
    """A batch a Stop left behind ends at its next item and that child does the next scan: nothing of the Stop stays."""
    _left, lines, _stopped_book, _result = _stopped_scan(phone, "first", True, monkeypatch)
    assert _went_to_children(lines), lines[:8]
    for name in ("GRACEFUL_STOP", "TRANSLATION_CANCELLED"):  # a new run's reset
        os.environ.pop(name, None)
    book = _book(phone / "out" / "threads" / "Book")
    _scan(book, "quick-scan", _settings("quick-scan"), interpreted=False, monkeypatch=monkeypatch)
    expected = _report_rows(book)
    book = _book(phone / "out" / "again" / "Book")
    lines = _scan(book, "quick-scan", _settings("quick-scan"), interpreted=True, monkeypatch=monkeypatch)
    assert _went_to_children(lines) and not _fell_back(lines), lines
    assert _report_rows(book) == expected
    assert interpreters.kept_pool().destroyed == 0


@needs_interpreters
@needs_backend
@pytest.mark.parametrize("filter_first", [False, True])
def test_one_child_does_a_scan_a_filter_and_sidecars_with_the_results_of_threads(glossary_phone, monkeypatch, capsys, filter_first):
    """The children every job shares: a filter's child has blocked the API client, a scan's has the scanner's caches
    emptied, and none of that changes what the next kind of job gets."""
    from glossarion_mobile.job_kinds import compile as compile_kind

    phone = glossary_phone
    monkeypatch.setenv("AI_HUNTER_MAX_WORKERS", "1")

    def scan(name: str, interpreted: bool) -> list:
        book = _book(phone / "out" / name / "Book")
        _scan(book, "quick-scan", _settings("quick-scan"), interpreted=interpreted, monkeypatch=monkeypatch)
        return _report_rows(book)

    def sidecars(name: str, children: bool) -> dict:
        folder = phone / "out" / name
        folder.mkdir(parents=True, exist_ok=True)
        _english_outputs(folder)
        return {"done": _generate(folder, children=children)[0], "tree": _tree(folder)}

    text = _korean_book(800)
    expected = (scan("threads", False), _filtered(text, child=False), sidecars("side_threads", False))
    if filter_first:  # what it blocked (the API client) is then blocked for all the work after it
        assert _filtered(text, child=True) == expected[1]
    assert scan("child_1", True) == expected[0]
    kept = interpreters.kept_pool()
    assert kept.started >= 1
    started = kept.started
    assert _filtered(text, child=True) == expected[1]
    assert sidecars("side_child", True)["tree"] == expected[2]["tree"]
    assert _filtered(text, child=True) == expected[1]
    assert scan("child_2", True) == expected[0]
    assert kept.destroyed == 0
    assert kept.started <= max(started, compile_kind.SIDECAR_WORKERS)


@needs_interpreters
@needs_backend
def test_ios_has_a_number_for_free_memory_where_the_system_gives_one(monkeypatch):
    from glossarion_mobile.services.local_llm import service

    monkeypatch.setattr(service, "memory_info", lambda: (8 * 1024 ** 3, None))
    monkeypatch.setattr(interpreters, "_room_before_jetsam", lambda: 500 * 1024 ** 2)
    assert interpreters.memory_now() == (8 * 1024 ** 3, 500 * 1024 ** 2)
    assert interpreters.size_for_memory(4, *reversed(interpreters.memory_now())) == 4
    monkeypatch.setattr(interpreters, "_room_before_jetsam", lambda: 320 * 1024 ** 2)
    assert interpreters.size_for_memory(4, *reversed(interpreters.memory_now())) == 0
    monkeypatch.setattr(interpreters, "_room_before_jetsam", lambda: None)  # a quarter of the RAM, with a limit in the workers
    assert interpreters.memory_now() == (8 * 1024 ** 3, None)
    assert interpreters.size_for_memory(4, *reversed(interpreters.memory_now())) == 4


def test_room_before_jetsam_is_none_where_there_is_no_such_call():
    assert sys.platform in ("ios", "darwin") or interpreters._room_before_jetsam() is None


@needs_interpreters
@needs_backend
def test_the_smoke_check_says_how_many_children_stayed(tmp_path, phone):
    from glossarion_mobile.diagnostics.interpreter_check import check_scan_in_child

    detail = check_scan_in_child(tmp_path, sys.platform)
    assert detail["result"] == "pass" and detail["kept"] >= 1 and detail["kept_destroyed"] == 0, detail
