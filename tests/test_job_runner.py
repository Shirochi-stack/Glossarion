"""job_runner + stop_control (Glossarion mobile rewrite, milestone U3).

What is checked:
* ``job_runner.scoped_process_state``: argv / env / ``large_env`` store / stdout /
  cwd / key pools restored even when the block raises; the desktop restore order
  (argv, ``os.environ.clear()`` + ``update``, ``large_env.clear_store()``) and the
  key-wise mode (never clears); ``JOB_LOCK`` held for the whole block;
* ``ProgressWatcher`` on a synthetic ``translation_progress.json`` and a fake watchdog;
* ``JobHooksMixin`` defaults (``host.emit`` payloads, lazy backend entry points);
* tier T (stop protocol): the working-tree ``TranslatorGUI.stop_translation`` (widgets +
  ``stop_control.request_stop``) produces exactly the event trace of the frozen legacy
  ``stop_translation`` (env writes/reads, module stop flags, owner latch, stop files,
  background cleanup, widget calls, logs, prints, timers) over 300 seeded scenarios incl.
  double clicks; a mobile ``request_stop(force=True)`` equals the desktop double-click
  flag protocol;
* tier T (process state): ``_process_text_file`` / ``_extract_glossary_from_text_file``
  (now ``scoped_process_state``) produce the frozen legacy's exact environment-operation
  trace, restore order included, over seeded fuzz states;
* the stop_control pieces are the desktop statements (AST) and behave like the legacy
  closures (glossary stop callback, click window, run-start resets).

Run (repository root)::

    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests/test_job_runner.py
"""

from __future__ import annotations

import ast
import contextlib
import inspect
import io
import json
import logging
import os
import random
import sys
import textwrap
import threading
import time
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

import job_runner  # noqa: E402
import stop_control  # noqa: E402

MISSING = object()


# ===========================================================================
# scoped_process_state
# ===========================================================================


@pytest.fixture
def clean_process(monkeypatch, tmp_path):
    """A small private environment / argv / cwd, put back by monkeypatch afterwards."""
    saved_env = dict(os.environ)
    saved_argv = list(sys.argv)
    saved_cwd = os.getcwd()
    import large_env

    saved_store = dict(large_env._store)
    os.environ.clear()
    os.environ.update({k: v for k, v in saved_env.items() if k in ("SYSTEMROOT", "PATH", "TEMP", "TMP")})
    os.environ["U3_KEEP"] = "keep"
    sys.argv = ["prog", "--flag"]
    large_env._store.clear()
    try:
        yield tmp_path
    finally:
        os.environ.clear()
        os.environ.update(saved_env)
        sys.argv = saved_argv
        os.chdir(saved_cwd)
        large_env._store.clear()
        large_env._store.update(saved_store)


def test_scoped_process_state_restores_everything_even_on_exceptions(clean_process):
    import large_env

    before_env = dict(os.environ)
    before_argv = sys.argv
    with pytest.raises(RuntimeError):
        with job_runner.scoped_process_state(["TransateKRtoEN.py", "book.epub"],
                                             env_updates={"MODEL": "gpt-4o", "BIG": "x" * 40000}):
            assert sys.argv == ["TransateKRtoEN.py", "book.epub"]
            assert os.environ["MODEL"] == "gpt-4o"
            assert large_env.get_env("BIG") == "x" * 40000  # over the Windows limit -> store
            os.environ["ADDED"] = "1"
            os.environ.pop("U3_KEEP")
            raise RuntimeError("job failed")
    assert dict(os.environ) == before_env
    assert sys.argv is before_argv and sys.argv == ["prog", "--flag"]
    assert large_env._store == {}


def test_scoped_process_state_desktop_restore_order(clean_process, monkeypatch):
    """argv first, then os.environ.clear() + update(saved), then large_env.clear_store()."""
    import large_env

    events = []
    real_environ = os.environ

    class Env(dict):
        def clear(self):
            events.append(("env.clear", list(sys.argv)))
            super().clear()

        def update(self, *a, **k):
            events.append(("env.update", list(sys.argv)))
            super().update(*a, **k)

    env = Env(real_environ)
    monkeypatch.setattr(os, "environ", env)
    monkeypatch.setattr(large_env, "clear_store", lambda: events.append(("large_env.clear_store", len(os.environ))))
    state = job_runner.scoped_process_state().snapshot()
    sys.argv = ["changed"]
    os.environ["X"] = "1"
    state.restore()
    assert events == [("env.clear", ["prog", "--flag"]), ("env.update", ["prog", "--flag"]),
                      ("large_env.clear_store", len(env))]
    events.clear()
    state = job_runner.scoped_process_state(clear_large_env=False).snapshot()
    state.restore()
    assert [e[0] for e in events] == ["env.clear", "env.update"]  # glossary extraction: no store reset


def test_scoped_process_state_snapshot_before_enter(clean_process):
    """The desktop glossary extractor snapshots, computes paths, then enters its try."""
    state = job_runner.scoped_process_state(clear_large_env=False).snapshot()
    os.environ["BETWEEN"] = "1"  # changed after the snapshot, before the block
    with state:
        os.environ["INSIDE"] = "1"
    assert "BETWEEN" not in os.environ and "INSIDE" not in os.environ


def test_scoped_process_state_keywise_never_clears(clean_process, monkeypatch, tmp_path):
    import large_env

    calls = []
    real_environ = os.environ

    class Env(dict):
        def clear(self):
            calls.append("clear")
            super().clear()

    env = Env(real_environ)
    monkeypatch.setattr(os, "environ", env)
    large_env._store["OLD"] = "keep"
    argv_obj = sys.argv
    cwd = os.getcwd()
    (tmp_path / "elsewhere").mkdir()
    with job_runner.scoped_process_state(["job"], keywise=True, restore_cwd=True):
        os.environ["NEW"] = "1"
        os.environ["U3_KEEP"] = "changed"
        large_env._store["NEW"] = "1"
        large_env._store.pop("OLD")
        sys.argv.append("extra")
        os.chdir(tmp_path / "elsewhere")
    assert calls == []
    assert os.environ["U3_KEEP"] == "keep" and "NEW" not in os.environ
    assert large_env._store == {"OLD": "keep"}
    assert sys.argv is argv_obj and sys.argv == ["prog", "--flag"]
    assert os.getcwd() == cwd


def test_scoped_process_state_isolates_unifiedclient_key_pools(clean_process, monkeypatch):
    class UnifiedClient:
        _in_memory_multi_keys = ["a"]
        _main_key_pool = object()
        _multi_pool_logged = True
        _pool_lock = object()

    module = types.ModuleType("unified_api_client")
    module.UnifiedClient = UnifiedClient
    monkeypatch.setitem(sys.modules, "unified_api_client", module)
    old_pool = UnifiedClient._main_key_pool
    with job_runner.scoped_process_state(isolate_key_pools=True, keywise=True):
        assert UnifiedClient._main_key_pool is None  # fresh pool for the job
        UnifiedClient._main_key_pool = "job pool"
        UnifiedClient._in_memory_multi_keys = ["job"]
        UnifiedClient._in_memory_glossary_keys = ["new"]
    assert UnifiedClient._main_key_pool is old_pool
    assert UnifiedClient._in_memory_multi_keys == ["a"]
    assert not hasattr(UnifiedClient, "_in_memory_glossary_keys")


def test_scoped_process_state_captures_stdout_lines(clean_process):
    lines = []
    real_out = sys.stdout
    with job_runner.scoped_process_state(capture_stdout=lines.append):
        print("first line")
        sys.stdout.write("partial")
        print(" done\nsecond", end="")
        print("err line", file=sys.stderr)
    assert sys.stdout is real_out
    assert lines == ["first line", "partial done", "err line", "second"]


def test_job_process_state_is_the_mobile_job_scope(clean_process):
    lines = []
    scope = job_runner.job_process_state(lines.append)
    assert scope.lock is job_runner.JOB_LOCK and scope.keywise and scope.isolate_key_pools and scope.restore_cwd
    before = dict(os.environ)
    with scope:
        assert job_runner.JOB_LOCK._is_owned()
        os.environ["JOB_SET"] = "1"
        print("job output")
    assert lines == ["job output"] and dict(os.environ) == before


def test_job_lock_is_held_for_the_block(clean_process):
    assert isinstance(job_runner.JOB_LOCK, type(threading.RLock()))
    entered = threading.Event()
    release = threading.Event()
    order = []

    def job():
        with job_runner.scoped_process_state(lock=job_runner.JOB_LOCK):
            order.append("job in")
            entered.set()
            release.wait(5)
            order.append("job out")

    t = threading.Thread(target=job)
    t.start()
    assert entered.wait(5)
    assert not job_runner.JOB_LOCK.acquire(timeout=0.2)  # a preview / second job must wait
    release.set()
    assert job_runner.run_exclusive(lambda: order.append("exclusive") or "ok") == "ok"
    t.join(5)
    assert order == ["job in", "job out", "exclusive"]
    # released when the block raises, too
    with pytest.raises(ValueError):
        with job_runner.scoped_process_state(lock=job_runner.JOB_LOCK):
            raise ValueError
    assert job_runner.JOB_LOCK.acquire(timeout=1)
    job_runner.JOB_LOCK.release()


# ===========================================================================
# ProgressWatcher / hooks / events
# ===========================================================================


class _Host:
    def __init__(self):
        self.events = []
        self.lines = []

    def emit(self, kind, **data):
        self.events.append((kind, data))

    def log(self, text, **kw):
        self.lines.append(text)

    def is_stop_requested(self):
        return False

    def is_graceful_stop(self):
        return False

    def ask(self, kind, **data):
        return None


def _write_progress(folder: Path, statuses: dict) -> Path:
    chapters = {}
    for i, status in enumerate(statuses.values(), start=1):
        out = f"response_{i:04d}_ch{i}.html"
        chapters[f"key{i}"] = {"status": status, "output_file": out, "original_basename": f"ch{i}.xhtml"}
        if status == "completed":
            (folder / out).write_text("<p>x</p>", encoding="utf-8")
    path = folder / "translation_progress.json"
    path.write_text(json.dumps({"chapters": chapters}), encoding="utf-8")
    return path


def test_progress_watcher_on_a_synthetic_progress_file(tmp_path):
    folder = tmp_path / "Book"
    folder.mkdir()
    host = _Host()
    state = {"in_flight": 0}
    watcher = job_runner.ProgressWatcher(lambda: str(folder), host, interval=0.05,
                                         watchdog_state=lambda: {"in_flight": state["in_flight"], "backlog": 1,
                                                                 "in_flight_entries": [{"status": "queued"}]})
    # no progress file yet: only the api state
    assert [k for k, _d in watcher.poll_once()] == ["api_state"]
    path = _write_progress(folder, {"a": "completed", "b": "completed", "c": "in_progress", "d": "failed",
                                    "e": "pending"})
    emitted = watcher.poll_once()
    assert emitted == [("progress", {"total": 5, "completed": 2, "in_progress": 2, "failed": 1, "path": str(path)})]
    assert watcher.poll_once() == []  # unchanged file and watchdog: nothing new
    # a completed chapter whose response file was deleted counts as in progress (library rule)
    (folder / "response_0001_ch1.html").unlink()
    os.utime(path, (time.time() + 5, time.time() + 5))
    assert watcher.poll_once()[0][1]["completed"] == 1
    state["in_flight"] = 3
    api = watcher.poll_once()
    assert api[0][0] == "api_state" and api[0][1]["in_flight"] == 3 and api[0][1]["queued_entries"] == 1
    assert host.events[-1][0] == "api_state"
    # thread mode emits through host.emit and stops cleanly
    host2 = _Host()
    with job_runner.ProgressWatcher(str(folder), host2, interval=0.02, watchdog_state=lambda: None):
        deadline = time.time() + 5
        while not host2.events and time.time() < deadline:
            time.sleep(0.01)
    assert host2.events and host2.events[0][0] == "progress"


def test_progress_watcher_uses_the_library_summary():
    import inspect

    import library_core

    assert job_runner._default_summary_reader.__code__.co_names.count("_read_progress_summary") == 1
    assert "library_core" in inspect.getsource(job_runner._default_summary_reader)
    assert library_core._read_progress_summary.__module__ == "library_core"


def test_job_hooks_defaults_report_to_the_host(monkeypatch):
    host = _Host()

    class Owner(job_runner.JobHooksMixin):
        pass

    owner = Owner()
    owner.host = host
    owner._ui_request("input_files_updated", ["a.epub"])
    owner._ui_request("thread_complete")
    owner._ui_request("custom_kind", 1, 2, extra="x")
    owner._notify_compile_result("epub", path="book.epub")
    assert host.events == [
        ("input_files_updated", {"files": ["a.epub"]}),
        ("thread_complete", {}),
        ("custom_kind", {"args": [1, 2], "extra": "x"}),
        ("compile_result", {"compile_kind": "epub", "path": "book.epub", "error": None}),
    ]
    Owner()._ui_request("thread_complete")  # no host: no-op
    fake = types.ModuleType("TransateKRtoEN")
    fake.main = lambda **kw: "ran"
    monkeypatch.setitem(sys.modules, "TransateKRtoEN", fake)
    assert owner._backend_entry("translation_main")() == "ran"
    monkeypatch.setitem(sys.modules, "epub_converter", None)  # import fails -> None (desktop: lazy load failed)
    assert owner._backend_entry("fallback_compile_epub") is None
    event = job_runner.JobEvent("progress", "job1", data={"total": 1})
    assert event.ts > 0 and isinstance(host, job_runner.JobHost)


# ===========================================================================
# stop_control pieces
# ===========================================================================


def test_stop_click_window():
    times = []
    times = stop_control.register_stop_click(times, 100.0)
    assert times == [100.0]
    times = stop_control.register_stop_click(times, 100.5)
    assert times == [100.0, 100.5]
    assert stop_control.register_stop_click([100.0], 101.0) == [101.0]  # exactly 1 s apart: not a double click
    tracker = stop_control.StopClickTracker()
    assert tracker.register(10.0) is False
    assert tracker.register(10.6) is True
    assert tracker.times == []  # forgotten after a forced stop
    assert tracker.register(20.0) is False


def test_kill_helper_subprocesses_is_a_noop_without_processes(monkeypatch):
    sentinel = types.ModuleType("psutil")

    def boom(*a, **k):
        raise AssertionError("psutil used on a platform without subprocesses")

    sentinel.Process = boom
    monkeypatch.setitem(sys.modules, "psutil", sentinel)
    monkeypatch.setenv("GLOSSARION_NO_PROCESSES", "1")
    stop_control.kill_helper_subprocesses()


def test_reset_for_new_run(monkeypatch, tmp_path):
    calls = []
    uac = types.ModuleType("unified_api_client")
    uac.hard_cancel_all = lambda: calls.append("hard_cancel_all")
    uac.set_stop_flag = lambda flag: calls.append(("client_stop", flag))
    tk = types.ModuleType("TransateKRtoEN")
    tk.set_stop_flag = lambda flag: calls.append(("translation_stop", flag))
    gl = types.ModuleType("extract_glossary_from_epub")
    gl.set_stop_flag = lambda flag: calls.append(("glossary_stop", flag))
    for name, mod in (("unified_api_client", uac), ("TransateKRtoEN", tk), ("extract_glossary_from_epub", gl)):
        monkeypatch.setitem(sys.modules, name, mod)
    stop_file = tmp_path / "stop.flag"
    stop_file.write_text("stop", encoding="utf-8")
    for key, value in (("TRANSLATION_CANCELLED", "1"), ("GRACEFUL_STOP", "1"), ("WAIT_FOR_CHUNKS", "1"),
                       ("GLOSSARY_STOP_FILE", str(stop_file)), ("TRANSLATION_ANTI_DUPLICATE_LOGGED", "1")):
        monkeypatch.setenv(key, value)
    run_id = stop_control.reset_for_new_run("translation")
    assert len(run_id) == 10 and os.environ["GLOSSARION_RUN_ID"] == run_id
    assert "TRANSLATION_CANCELLED" not in os.environ and "TRANSLATION_ANTI_DUPLICATE_LOGGED" not in os.environ
    assert os.environ["GRACEFUL_STOP"] == os.environ["WAIT_FOR_CHUNKS"] == os.environ["GRACEFUL_STOP_API_ACTIVE"] == "0"
    assert not stop_file.exists()
    assert calls == [("translation_stop", False), "hard_cancel_all", ("client_stop", False)]
    calls.clear()
    monkeypatch.setenv("TRANSLATION_CANCELLED", "1")
    run_id = stop_control.reset_for_new_run("glossary")
    assert run_id.startswith("glossary-") and os.environ["TRANSLATION_CANCELLED"] == "1"
    assert calls == [("glossary_stop", False)]


# ---- AST: the stop_control parts are the desktop statements --------------------------------


def _legacy_methods():
    """The U3 base SHA; only its git source is read (no PySide6 needed)."""
    return BASE_SHA


def _legacy_tg_source(sha) -> str:
    from parity import freeze_legacy

    try:
        return freeze_legacy.git_show_text(sha, "src/translator_gui.py")
    except Exception as exc:  # pragma: no cover - shallow clone
        if os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"):
            pytest.fail(f"git show {sha[:12]}:src/translator_gui.py unavailable: {exc}")
        pytest.skip(f"git show {sha[:12]}:src/translator_gui.py unavailable: {exc}")


def _method(tree_text, cls_name, name):
    tree = ast.parse(tree_text)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls_name)
    return next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)


def _function(module_name, name):
    text = (SRC / f"{module_name}.py").read_bytes().decode("utf-8-sig").replace("\r\n", "\n")
    return next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == name)


def _stmts(node):
    body = list(node.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]  # docstring
    return [ast.unparse(s) for s in body]


def _contains_sequence(haystack, needle):
    n = len(needle)
    return any(haystack[i:i + n] == needle for i in range(len(haystack) - n + 1))


def test_stop_control_parts_are_the_desktop_statements():
    bundle = _legacy_methods()
    text = _legacy_tg_source(bundle)
    run_tt = _stmts(_method(text, "TranslatorGUI", "run_translation_thread"))
    run_gt = _stmts(_method(text, "TranslatorGUI", "run_glossary_extraction_thread"))
    stop_tr = _method(text, "TranslatorGUI", "stop_translation")
    stop_body = _stmts(stop_tr)
    watchdog = _stmts(_method(text, "TranslatorGUI", "_reset_api_watchdog_progress"))

    reset_env = _function("stop_control", "reset_stop_env")
    tr_branch = [ast.unparse(s) for s in reset_env.body if not isinstance(s, (ast.If, ast.Expr))]
    gl_branch = [ast.unparse(s) for s in next(s for s in reset_env.body if isinstance(s, ast.If)).body[:-1]]
    assert _contains_sequence(run_tt, tr_branch)
    assert _contains_sequence(run_gt, gl_branch)
    for name in ("clear_client_cancellation",):
        assert _contains_sequence(run_tt, _stmts(_function("stop_control", name)))
    prepare = _stmts(_function("stop_control", "prepare_glossary_stop_file"))[:-1]  # + return
    assert _contains_sequence(run_tt, prepare) and _contains_sequence(run_gt, prepare)
    assert _contains_sequence(watchdog, _stmts(_function("stop_control", "reset_api_watchdog")))
    # force-stop flags: the double-click branch of stop_translation
    force = next(s for s in stop_tr.body if isinstance(s, ast.If) and ast.unparse(s.test) == "force_stop")
    force_body = [ast.unparse(s) for s in force.body]
    assert _contains_sequence(force_body, _stmts(_function("stop_control", "apply_force_stop_flags")))
    # request_stop: every flag statement of the legacy protocol, in order
    req = _function("stop_control", "request_stop")
    req_text = ast.unparse(req)
    legacy_proto = ast.unparse(stop_tr)
    for stmt in ("os.environ['TRANSLATION_CANCELLED'] = '1'",
                 "os.environ['GRACEFUL_STOP'] = '1' if graceful_stop else '0'",
                 "os.environ['GRACEFUL_STOP_COMPLETED'] = '0'",
                 "cancelled_queued = int(cancel_queued() or 0)",
                 "cleared_watchdog = int(clear_pending() or 0)",
                 "os.environ['WAIT_FOR_CHUNKS'] = '1' if wait_for_chunks else '0'",
                 "unified_api_client.reset_api_call_stagger()",
                 "f.write('stop')",
                 "TransateKRtoEN.set_stop_flag(True)",
                 "unified_api_client.global_stop_flag = True",
                 "unified_api_client.UnifiedClient._global_cancelled = True",
                 "_uac.hard_cancel_all()"):
        assert stmt in legacy_proto and stmt in req_text, stmt
    psutil_legacy = legacy_proto[legacy_proto.index("import psutil"):legacy_proto.index("print(f'Error terminating")]
    kill = ast.unparse(_function("stop_control", "kill_helper_subprocesses"))
    psutil_new = kill[kill.index("import psutil"):kill.index("print(f'Error terminating")]
    assert textwrap.dedent(psutil_new).split() == textwrap.dedent(psutil_legacy).split()


def test_stop_tail_and_cleanup_wait_are_the_desktop_statements():
    """U3 fix pass: stop_translation's non-widget tail (``stop_epub_converter``, ``announce_stop``)
    and run_translation_thread's wait for the previous cleanup (``wait_for_stop_cleanup``) are the
    legacy statements; the desktop and the mobile JobService both call them."""
    text = _legacy_tg_source(_legacy_methods())
    stop_tr = ast.unparse(_method(text, "TranslatorGUI", "stop_translation"))
    run_tt = ast.unparse(_method(text, "TranslatorGUI", "run_translation_thread"))
    converter = ast.unparse(_function("stop_control", "stop_epub_converter"))
    for stmt in ("import epub_converter", "if hasattr(epub_converter, 'set_stop_flag'):",
                 "epub_converter.set_stop_flag(True)"):
        assert stmt in stop_tr and stmt in converter, stmt
    announce = ast.unparse(_function("stop_control", "announce_stop")) + ast.unparse(
        _function("stop_control", "silence_http_loggers"))
    loggers = "['httpx', 'openai', 'google', 'google.api_core', 'google.generativeai', 'urllib3']"
    assert loggers in stop_tr and repr(stop_control.HTTP_LOGGER_NAMES) == loggers
    for stmt in ("logging.getLogger(logger_name).setLevel(logging.CRITICAL)",
                 "wait_for_chunks = os.environ.get('WAIT_FOR_CHUNKS') == '1'",
                 "⏳ Graceful stop — waiting for in-flight API calls to complete...",
                 "⏳ Graceful stop — waiting for the current in-flight API call; queued work will not continue "
                 "(WAIT_FOR_CHUNKS=0)",
                 "🛑 Force stop requested — aborting queued/in-flight API calls"):
        assert stmt in stop_tr and stmt in announce, stmt
    wait = ast.unparse(_function("stop_control", "wait_for_stop_cleanup"))
    for stmt in ("if previous_stop_cleanup is not None and previous_stop_cleanup.is_alive():",
                 "⏳ Waiting for the previous translation HTTP session to close...",
                 "previous_stop_cleanup.join(timeout=",
                 "⏹️ Previous translation is still stopping; try Start again shortly."):
        assert stmt in run_tt and stmt in wait, stmt
    assert "previous_stop_cleanup.join(timeout=3.0)" in run_tt
    assert inspect.signature(stop_control.wait_for_stop_cleanup).parameters["timeout"].default == 3.0


def test_wait_for_stop_cleanup_and_announce_stop_behaviour(monkeypatch):
    lines = []
    assert stop_control.wait_for_stop_cleanup(None, lines.append) is True and lines == []
    release = threading.Event()
    thread = threading.Thread(target=release.wait, args=(5,), daemon=True)
    thread.start()
    assert stop_control.wait_for_stop_cleanup(thread, lines.append, timeout=0.05) is False
    assert lines == ["⏳ Waiting for the previous translation HTTP session to close...",
                     "⏹️ Previous translation is still stopping; try Start again shortly."]
    release.set()
    thread.join(5)
    assert stop_control.wait_for_stop_cleanup(thread, lines.append) is True and len(lines) == 2
    logged = []
    monkeypatch.setenv("WAIT_FOR_CHUNKS", "0")
    levels = {name: logging.getLogger(name).level for name in stop_control.HTTP_LOGGER_NAMES}
    try:
        stop_control.announce_stop(True, logged.append)
        stop_control.announce_stop(False, logged.append)
        assert all(logging.getLogger(name).level == logging.CRITICAL for name in stop_control.HTTP_LOGGER_NAMES)
    finally:
        for name, level in levels.items():
            logging.getLogger(name).setLevel(level)
    assert logged == ["⏳ Graceful stop — waiting for the current in-flight API call; queued work will not continue "
                      "(WAIT_FOR_CHUNKS=0)", "🛑 Force stop requested — aborting queued/in-flight API calls"]


def test_glossary_stop_callback_matches_the_legacy_closure(monkeypatch):
    """make_glossary_stop_callback vs the nested enhanced_stop_callback of the frozen legacy
    _extract_glossary_from_text_file, over every input combination."""
    bundle = _legacy_methods()
    text = _legacy_tg_source(bundle)
    method = _method(text, "TranslatorGUI", "_extract_glossary_from_text_file")
    nested = next(n for n in ast.walk(method) if isinstance(n, ast.FunctionDef) and n.name == "enhanced_stop_callback")
    lines = text.split("\n")
    src = textwrap.dedent("\n".join(lines[nested.lineno - 1:nested.end_lineno]))
    factory = "def _make(self):\n" + textwrap.indent(src, "    ") + "\n    return enhanced_stop_callback\n"
    ns = {"os": os}
    exec(compile(factory, "<legacy enhanced_stop_callback>", "exec"), ns)
    make_legacy = ns["_make"]

    uac = types.ModuleType("unified_api_client")
    egl = types.ModuleType("extract_glossary_from_epub")
    monkeypatch.setitem(sys.modules, "unified_api_client", uac)
    monkeypatch.setitem(sys.modules, "extract_glossary_from_epub", egl)
    checked = 0
    for stop in (True, False):
        for graceful in (True, False, None, MISSING):
            for env in ("1", "0", MISSING):
                for in_flight in (0, 2, "raise", None):
                    for extractor in (True, False, MISSING):
                        if env is MISSING:
                            monkeypatch.delenv("GRACEFUL_STOP", raising=False)
                        else:
                            monkeypatch.setenv("GRACEFUL_STOP", env)

                        def state(value=in_flight):
                            if value == "raise":
                                raise RuntimeError("watchdog unavailable")
                            return None if value is None else {"in_flight": value}

                        uac.get_api_watchdog_state = state
                        if extractor is MISSING:
                            egl.__dict__.pop("is_stop_requested", None)
                        else:
                            egl.is_stop_requested = (lambda v=extractor: v)
                        owner = types.SimpleNamespace(stop_requested=stop)
                        if graceful is not MISSING:
                            owner.graceful_stop_active = graceful
                        legacy = make_legacy(owner)()
                        new = stop_control.make_glossary_stop_callback(
                            lambda: owner.stop_requested,
                            lambda: getattr(owner, "graceful_stop_active", False))()
                        assert legacy == new, (stop, graceful, env, in_flight, extractor)
                        checked += 1
    assert checked == 2 * 4 * 3 * 4 * 3


# ===========================================================================
# Tier T: the desktop stop protocol, legacy stop_translation vs the working tree
# ===========================================================================

#: env reads the new code adds on purpose (mobile_runtime's process gates)
GATE_KEYS = frozenset({"GLOSSARION_NO_PROCESSES", "GLOSSARION_MOBILE", "FLET_PLATFORM"})


class TraceEnv(dict):
    """os.environ stand-in recording every read and write in order."""

    def __init__(self, data, trace):
        super().__init__(data)
        self._trace = trace

    def _log(self, *event):
        if len(event) > 1 and event[1] in GATE_KEYS and event[0] in ("env.get", "env.contains"):
            return
        self._trace.append(event)

    def __setitem__(self, key, value):
        self._log("env.set", key, value)
        super().__setitem__(key, value)

    def __delitem__(self, key):
        self._log("env.del", key)
        super().__delitem__(key)

    def __getitem__(self, key):
        self._log("env.getitem", key)
        return super().__getitem__(key)

    def __contains__(self, key):
        self._log("env.contains", key)
        return super().__contains__(key)

    def get(self, key, default=None):
        self._log("env.get", key)
        return super().get(key, default)

    def pop(self, key, *default):
        self._log("env.pop", key)
        return super().pop(key, *default)

    def clear(self):
        self._log("env.clear", tuple(sys.argv))
        super().clear()

    def update(self, *args, **kwargs):
        data = dict(*args, **kwargs)
        self._log("env.update", tuple(sys.argv), len(data))
        super().update(data)

    def copy(self):
        return dict(self)


class TraceModule(types.ModuleType):
    """Module whose attribute writes are traced (global_stop_flag, ...)."""

    def __init__(self, name, trace):
        super().__init__(name)
        object.__setattr__(self, "_trace", trace)

    def __setattr__(self, name, value):
        if not name.startswith("__") and not callable(value):
            self._trace.append(("mod.setattr", self.__name__, name, value))
        object.__setattr__(self, name, value)


def _recording_fn(trace, label, result=None):
    def fn(*args, **kwargs):
        trace.append(("mod", label, args, tuple(sorted(kwargs.items()))))
        if isinstance(result, BaseException):
            raise result
        return result

    return fn


class TraceWidget:
    def __init__(self, name, trace, text=""):
        self._name, self._trace, self._text = name, trace, text

    def __getattr__(self, method):
        if method.startswith("_"):
            raise AttributeError(method)

        def call(*args, **kwargs):
            self._trace.append(("widget", self._name, method, args))
            if method == "setText":
                self._text = args[0]
            return None

        return call

    def text(self):
        self._trace.append(("widget", self._name, "text", ()))
        return self._text


class FakeThread:
    trace = None

    def __init__(self, target=None, daemon=None, name=None, **kwargs):
        self._target, self.name, self.daemon = target, name, daemon
        FakeThread.trace.append(("thread.new", name, daemon))

    def start(self):
        FakeThread.trace.append(("thread.start", self.name))
        self._target()  # inline: deterministic order

    def is_alive(self):
        return False

    def __repr__(self):
        return f"<FakeThread {self.name}>"


WATCHED_ATTRS = ("stop_requested", "graceful_stop_active", "_last_stop_was_graceful",
                 "_last_stop_translation_ts", "_translation_stop_cleanup_thread", "_stop_click_times")
RECORDED_METHODS = ("append_log", "save_config", "_reset_api_watchdog_progress", "preserve_file_path",
                    "_reset_stop_flags_if_idle")


def _owner_class(stop_fn, trace):
    def __setattr__(self, name, value):
        if name in WATCHED_ATTRS:
            shown = repr(value) if isinstance(value, FakeThread) else value
            if isinstance(value, list):
                shown = list(value)
            trace.append(("set", name, shown))
        object.__setattr__(self, name, value)

    ns = {"__setattr__": __setattr__, "stop_translation": stop_fn}
    for name in RECORDED_METHODS:
        ns[name] = (lambda n: lambda self, *a, **k: trace.append(("call", n, a, tuple(sorted(k.items())))))(name)
    return type("StopOwner", (), ns)


class _Logger:
    def __init__(self, name, trace):
        self.name, self.trace = name, trace

    def setLevel(self, level):
        self.trace.append(("logger.setLevel", self.name, level))


def _scenario(seed, tmp_path):
    rng = random.Random(seed)
    pick = rng.choice
    return {
        "graceful": pick([True, False, MISSING]),
        "wait": pick([True, False, MISSING]),
        "zip_active": pick([True, False, MISSING]),
        "prior_clicks": pick([MISSING, [], [-0.4], [-3.0], [-0.2, -0.5]]),
        "button_text": pick([MISSING, "Run", "Finishing..."]),
        "run_button": pick([True, False]),
        "entry_epub": pick([MISSING, "", "book.epub"]),
        "progress_bar": pick([MISSING, True]),
        "env": {
            "BATCH_TRANSLATION": pick(["1", "0", MISSING]),
            "WAIT_FOR_CHUNKS": pick(["0", "1", MISSING]),
            "PDF_EXTRACTION_STOP_FILE": pick([str(tmp_path / f"pdf{seed}" / "pdf.stop"), MISSING]),
            "GLOSSARY_STOP_FILE": pick([str(tmp_path / f"gloss{seed}.stop"), MISSING]),
            "TRANSLATION_CANCELLED": pick(["0", MISSING]),
        },
        "epub": pick([MISSING, "running", "done", "thread_alive"]),
        "clear_pending": pick([MISSING, 0, 2]),
        "stagger": pick(["module", "class", MISSING]),
        "global_stop_flag": pick([True, False]),
        "cancel_queued": pick([MISSING, 0, 3, "raise"]),
        "translation_stop_flag": pick([None, "recorder"]),
        "double": rng.random() < 0.35,
    }


def _install_fakes(monkeypatch, trace, sc):
    uac = TraceModule("unified_api_client", trace)
    uac.set_stop_flag = _recording_fn(trace, "uac.set_stop_flag")
    uac.hard_cancel_all = _recording_fn(trace, "uac.hard_cancel_all")
    uac._api_watchdog_reset = _recording_fn(trace, "uac._api_watchdog_reset")
    if sc["clear_pending"] is not MISSING:
        uac._api_watchdog_clear_pending_requests = _recording_fn(trace, "uac.clear_pending", sc["clear_pending"])
    if sc["stagger"] == "module":
        uac.reset_api_call_stagger = _recording_fn(trace, "uac.reset_api_call_stagger")
    if sc["global_stop_flag"]:
        object.__setattr__(uac, "global_stop_flag", False)

    class _Meta(type):
        def __setattr__(cls, name, value):
            trace.append(("cls.setattr", name, value))
            super().__setattr__(name, value)

    client_ns = {}
    if sc["stagger"] == "class":
        client_ns["reset_api_call_stagger"] = staticmethod(_recording_fn(trace, "UnifiedClient.reset_api_call_stagger"))
    uac.UnifiedClient = _Meta("UnifiedClient", (), client_ns)
    tk = TraceModule("TransateKRtoEN", trace)
    tk.set_stop_flag = _recording_fn(trace, "tk.set_stop_flag")
    if sc["cancel_queued"] is not MISSING:
        result = RuntimeError("cancel failed") if sc["cancel_queued"] == "raise" else sc["cancel_queued"]
        tk.cancel_queued_translation_sends = _recording_fn(trace, "tk.cancel_queued", result)
    epub = TraceModule("epub_converter", trace)
    epub.set_stop_flag = _recording_fn(trace, "epub.set_stop_flag")
    psutil = types.ModuleType("psutil")

    class _Proc:
        def __init__(self, pid):
            trace.append(("psutil.Process",))

        def children(self, recursive=False):
            trace.append(("psutil.children", recursive))
            return []

    psutil.Process = _Proc
    psutil.NoSuchProcess = type("NoSuchProcess", (Exception,), {})
    psutil.AccessDenied = type("AccessDenied", (Exception,), {})
    psutil.wait_procs = _recording_fn(trace, "psutil.wait_procs", ([], []))
    manga = types.ModuleType("manga_translator")
    manga.MangaTranslator = type("MangaTranslator", (), {"_inpaint_pool": None})
    for name, mod in (("unified_api_client", uac), ("TransateKRtoEN", tk), ("epub_converter", epub),
                      ("psutil", psutil), ("manga_translator", manga)):
        monkeypatch.setitem(sys.modules, name, mod)
    FakeThread.trace = trace
    monkeypatch.setattr(threading, "Thread", FakeThread)
    import logging

    monkeypatch.setattr(logging, "getLogger", lambda name=None: _Logger(name, trace))


def _build_owner(owner_cls, sc, trace, now):
    owner = owner_cls()
    trace.clear()  # construction is not part of the trace
    if sc["graceful"] is not MISSING:
        object.__setattr__(owner, "graceful_stop_var", sc["graceful"])
    if sc["wait"] is not MISSING:
        object.__setattr__(owner, "wait_for_chunks_var", sc["wait"])
    if sc["zip_active"] is not MISSING:
        object.__setattr__(owner, "_zip_conversion_active", sc["zip_active"])
    if sc["prior_clicks"] is not MISSING:
        object.__setattr__(owner, "_stop_click_times", [now + d for d in sc["prior_clicks"]])
    if sc["button_text"] is not MISSING:
        object.__setattr__(owner, "run_button_text", TraceWidget("run_button_text", trace, sc["button_text"]))
    if sc["run_button"]:
        object.__setattr__(owner, "run_button", TraceWidget("run_button", trace))
    if sc["entry_epub"] is not MISSING:
        object.__setattr__(owner, "entry_epub", TraceWidget("entry_epub", trace, sc["entry_epub"]))
    if sc["progress_bar"] is not MISSING:
        object.__setattr__(owner, "progress_bar", TraceWidget("progress_bar", trace))
    if sc["epub"] == "running":
        object.__setattr__(owner, "epub_future", types.SimpleNamespace(done=lambda: False))
    elif sc["epub"] == "done":
        object.__setattr__(owner, "epub_future", types.SimpleNamespace(done=lambda: True))
    elif sc["epub"] == "thread_alive":
        object.__setattr__(owner, "epub_thread", types.SimpleNamespace(is_alive=lambda: True))
    return owner


class _TraceStdout(io.TextIOBase):
    def __init__(self, trace):
        self.trace = trace

    def write(self, text):
        if text.strip():
            self.trace.append(("print", text.rstrip("\n")))
        return len(text)


@pytest.fixture(scope="module")
def stop_sides():
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from parity import fuzz_moved as fm

    sess = fm.session()
    try:
        bundle = sess.bundle
        tg = sess.translator_gui_class()
    except fm.Unavailable as exc:
        pytest.skip(str(exc))
    import translator_gui

    assert "stop_translation" in vars(bundle.methods), "legacy oracle lacks stop_translation; re-run freeze_legacy.py"
    return {
        "legacy": (vars(bundle.methods)["stop_translation"], bundle.namespace),
        "new": (vars(tg)["stop_translation"], vars(translator_gui)),
    }


def _run_stop(side, sc, monkeypatch, tmp_path, trace):
    fn, namespace = side
    now = 1_000_000.0
    clock = {"t": now}
    monkeypatch.setattr(time, "time", lambda: clock["t"])
    timers = []

    class QTimer:
        @staticmethod
        def singleShot(ms, fn):
            trace.append(("QTimer.singleShot", ms, getattr(fn, "__name__", "<fn>")))
            timers.append(fn)

    monkeypatch.setitem(namespace, "QTimer", QTimer)
    flag = None if sc["translation_stop_flag"] is None else _recording_fn(trace, "translation_stop_flag")
    monkeypatch.setitem(namespace, "translation_stop_flag", flag)
    env = TraceEnv({k: v for k, v in sc["env"].items() if v is not MISSING}, trace)
    env.update({"SYSTEMROOT": os.environ.get("SYSTEMROOT", "")})
    trace.clear()
    monkeypatch.setattr(os, "environ", env)
    owner = _build_owner(_owner_class(fn, trace), sc, trace, now)
    with contextlib.redirect_stdout(_TraceStdout(trace)):
        owner.stop_translation()
        if sc["double"]:
            clock["t"] = now + 0.3
            trace.append(("second click",))
            owner.stop_translation()
    files = {}
    for key in ("PDF_EXTRACTION_STOP_FILE", "GLOSSARY_STOP_FILE"):
        path = sc["env"].get(key)
        if path is not MISSING and os.path.exists(path):
            files[key] = Path(path).read_text(encoding="utf-8")
            os.remove(path)
    state = {k: (repr(v) if isinstance(v, FakeThread) else v) for k, v in vars(owner).items()
             if isinstance(v, (FakeThread, bool, int, float, str, list, tuple, type(None)))}
    return list(trace), files, dict(env), state


def test_stop_translation_trace_matches_legacy(stop_sides, monkeypatch, tmp_path):
    """Tier T: 300 seeded scenarios, full event traces must be identical."""
    checked = 0
    double_clicks = forced = graceful_runs = 0
    for seed in range(300):
        sc = _scenario(seed, tmp_path)
        results = {}
        for label in ("legacy", "new"):
            with monkeypatch.context() as mp:
                trace = []
                _install_fakes(mp, trace, sc)
                results[label] = _run_stop(stop_sides[label], sc, mp, tmp_path, trace)
        legacy, new = results["legacy"], results["new"]
        if legacy[0] != new[0]:
            first = next(i for i, (a, b) in enumerate(zip(legacy[0], new[0])) if a != b) \
                if any(a != b for a, b in zip(legacy[0], new[0])) else min(len(legacy[0]), len(new[0]))
            raise AssertionError(
                f"seed {seed}: traces differ at event {first}\n"
                f"  legacy: {legacy[0][max(0, first - 3):first + 4]}\n"
                f"  new:    {new[0][max(0, first - 3):first + 4]}\n  scenario: {sc}")
        assert legacy[1:] == new[1:], (seed, sc)
        trace = legacy[0]
        double_clicks += any(e == ("set", "_stop_click_times", []) for e in trace)
        forced += any(e[:2] == ("thread.new", "translation-stop-cleanup") for e in trace)
        graceful_runs += any(e == ("env.set", "GRACEFUL_STOP", "1") for e in trace)
        checked += 1
    assert checked == 300
    assert double_clicks > 20 and forced > 50 and graceful_runs > 50, (double_clicks, forced, graceful_runs)


def test_stop_trace_harness_catches_a_latch_moved_before_the_env_flags(stop_sides, monkeypatch, tmp_path):
    """Harness self-test: the race the shared protocol guards against (latch first) is detected."""
    import translator_gui

    real = stop_control.request_stop

    def latch_first(**kwargs):
        latch = kwargs.pop("set_stop_requested")
        latch()
        return real(set_stop_requested=lambda: None, **kwargs)

    sc = _scenario(3, tmp_path)
    sc.update(graceful=True, double=False, prior_clicks=MISSING)
    traces = {}
    for label in ("legacy", "new"):
        with monkeypatch.context() as mp:
            if label == "new":
                mp.setattr(translator_gui, "request_stop", latch_first)
            trace = []
            _install_fakes(mp, trace, sc)
            traces[label] = _run_stop(stop_sides[label], sc, mp, tmp_path, trace)[0]
    assert traces["legacy"] != traces["new"]
    first_set = traces["new"].index(("set", "stop_requested", True))
    assert first_set < traces["new"].index(("env.set", "GRACEFUL_STOP", "1"))


def _protocol(trace):
    """Flag protocol events only (no widgets, logs, prints, timers, owner bookkeeping)."""
    keep = []
    for event in trace:
        kind = event[0]
        if kind in ("env.set", "env.pop", "env.del", "mod", "mod.setattr", "cls.setattr", "thread.new",
                    "thread.start", "psutil.Process", "psutil.children"):
            keep.append(event)
        elif event[:2] == ("set", "stop_requested"):
            keep.append(("latch",))
        elif event[:2] == ("call", "_reset_api_watchdog_progress") and keep and keep[-1][0] == "mod":
            keep.append(("clear_watchdog",))
    return keep


@pytest.mark.parametrize("graceful", [True, False])
@pytest.mark.parametrize("wait", [True, False])
def test_mobile_force_stop_equals_the_desktop_double_click_protocol(stop_sides, monkeypatch, tmp_path, graceful, wait):
    """JobService maps a double tap to request_stop(force=True): same flags, same order as the
    desktop's double-click stop (widgets, logs and config save aside)."""
    sc = _scenario(0, tmp_path)
    sc.update(graceful=graceful, wait=wait, prior_clicks=[-0.3], double=False, epub=MISSING,
              translation_stop_flag=None, zip_active=False, button_text="Finishing...", run_button=True,
              entry_epub=MISSING, progress_bar=MISSING)
    sc["env"].update(PDF_EXTRACTION_STOP_FILE=str(tmp_path / "p.stop"), GLOSSARY_STOP_FILE=str(tmp_path / "g.stop"))
    with monkeypatch.context() as mp:
        trace = []
        _install_fakes(mp, trace, sc)
        desktop, _files, desktop_env, _state = _run_stop(stop_sides["legacy"], sc, mp, tmp_path, trace)
    with monkeypatch.context() as mp:
        trace = []
        _install_fakes(mp, trace, sc)
        env = TraceEnv({k: v for k, v in sc["env"].items() if v is not MISSING}, trace)
        env.update({"SYSTEMROOT": os.environ.get("SYSTEMROOT", "")})
        trace.clear()
        mp.setattr(os, "environ", env)
        with contextlib.redirect_stdout(_TraceStdout(trace)):
            thread = stop_control.request_stop(
                graceful=graceful, wait_for_chunks=wait, force=True,
                set_stop_requested=lambda: trace.append(("set", "stop_requested", True)),
                log=lambda m: trace.append(("log", m)),
                clear_watchdog=lambda: trace.append(("call", "_reset_api_watchdog_progress", (), ())))
        mobile_env = dict(env)
    assert isinstance(thread, FakeThread)
    assert _protocol(trace) == _protocol(desktop)
    assert mobile_env == desktop_env


# ===========================================================================
# Tier T: process-state restore order of the moved runners (fuzz states, full env traces)
# ===========================================================================


def _restore_trace_hooks(traces):
    import large_env

    def hooks(label):
        saved = {}

        def enter():
            trace = traces.setdefault(label, [])
            trace.clear()
            saved["environ"] = os.environ
            saved["clear_store"] = large_env.clear_store
            env = TraceEnv(dict(os.environ), trace)
            os.environ = env  # noqa: B003 - restored in exit

            def clear_store():
                trace.append(("large_env.clear_store", tuple(sys.argv), len(os.environ)))
                saved["clear_store"]()

            large_env.clear_store = clear_store

        def exit_():
            env = os.environ
            real = saved["environ"]
            os.environ = real
            real.clear()
            real.update(dict(env))
            large_env.clear_store = saved["clear_store"]

        return enter, exit_

    return hooks


@pytest.mark.parametrize("name,expect_store_clear", [
    ("_process_text_file", True),
    ("_extract_glossary_from_text_file", False),
])
def test_runner_process_state_trace_matches_legacy(name, expect_store_clear):
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from parity import fuzz_moved as fm
    from parity import moved_functions as mf

    sess = fm.session()
    spec = mf.get(name)
    try:
        fm.check_available(spec, sess)
    except fm.Unavailable as exc:
        pytest.skip(str(exc))
    traces = {}
    seen = {"states": 0, "restores": 0}
    problems = []

    def on_state(state, a, b):
        seen["states"] += 1
        legacy, new = traces.get("legacy", []), traces.get("new", [])
        if legacy != new:
            first = next((i for i, (x, y) in enumerate(zip(legacy, new)) if x != y), min(len(legacy), len(new)))
            problems.append(f"state #{state.index}: event {first}: {legacy[first:first + 2]} != {new[first:first + 2]}")
            return
        clears = [i for i, e in enumerate(new) if e[0] == "env.clear"]
        if not clears:
            return
        i = clears[-1]
        seen["restores"] += 1
        # restore order: argv already back when the env is cleared, then update, then the store
        assert new[i][1] == tuple(fm.normalize.BASELINE_ARGV), new[i]
        assert new[i + 1][0] == "env.update" and new[i + 1][1] == new[i][1]
        tail = [e[0] for e in new[i:]]
        if expect_store_clear:
            assert tail == ["env.clear", "env.update", "large_env.clear_store"], tail
        else:
            assert "large_env.clear_store" not in tail, tail

    report = fm.fuzz_moved(spec, sess, states=200, check=False, on_state=on_state,
                           hooks=_restore_trace_hooks(traces))
    assert report.ok, report.failure_text()
    assert not problems, "\n".join(problems[:5])
    assert seen["states"] == 200 and seen["restores"] > 60, seen
