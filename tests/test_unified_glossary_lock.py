"""unified_glossary.unified_lock: the Unified Glossary is the one file several Glossarion instances
share, so its read -> merge -> write must be safe across processes AND threads.

* two processes (multiprocessing, spawn; the worker lives in tests/_unified_glossary_lock_worker.py)
  and two threads merge different books into the same unified glossary many times: no entry is
  lost (without the lock the same run loses entries, see the worker's ``--no-lock`` mode);
* the lock is re-entrant for its thread, blocks other threads, breaks stale lock files (owner pid
  gone, or a heartbeat that stopped moving) but never a live owner's, honours Stop while waiting;
* merge_book / rebuild / rebuild_now hold it for their whole body; the dedup child never takes it.

Python 3.10 compatible. Run:
    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests/test_unified_glossary_lock.py
"""

import ast
import json
import multiprocessing
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
SRC_DIR = TESTS_DIR.parent / "src"
for _path in (SRC_DIR, TESTS_DIR):
    if str(_path) not in sys.path:  # CI has no tests/conftest.py
        sys.path.insert(0, str(_path))

import _unified_glossary_lock_worker as worker  # noqa: E402
import unified_glossary as ug  # noqa: E402

_REAL_STOP_REQUESTED = ug._stop_requested


@pytest.fixture
def shared(tmp_path, monkeypatch):
    """A Glossary folder; Stop is never requested unless a test says so."""
    folder = tmp_path / "Glossary"
    folder.mkdir()
    monkeypatch.setattr(ug, "_stop_requested", lambda: False)
    return folder


def _lock_path(shared_dir):
    return Path(ug.unified_root(shared_dir)) / ug.LOCK_FILENAME


def _dead_pid():
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


def _write_lock(path, **owner):
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"pid": os.getpid(), "host": ug._HOST, "instance": 2, "token": "someone-else",
              "created": time.time(), "beat": 0}
    record.update(owner)
    path.write_text(json.dumps(record), encoding="utf-8")


def _read_owner(path):
    """The lock file's record. _foreign_owner rewrites it with write_text (truncate, then write),
    so a read can land in between and see an empty or half-written file: read again."""
    for _attempt in range(100):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, PermissionError):
            time.sleep(0.01)
    return json.loads(path.read_text(encoding="utf-8"))


def _in_thread(target):
    box = {}

    def run():
        try:
            box["result"] = target()
        except BaseException as exc:  # pragma: no cover - surfaced by the assertion below
            box["error"] = exc

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, box


# --------------------------------------------------------------------------
# lock mechanics
# --------------------------------------------------------------------------

def test_lock_file_exists_while_held_and_is_reentrant(shared):
    path = _lock_path(shared)
    with ug.unified_lock(str(shared)) as held:
        assert Path(held) == path and path.is_file()
        owner = json.loads(path.read_text(encoding="utf-8"))
        assert owner["pid"] == os.getpid() and owner["host"] == ug._HOST and owner["token"]
        with ug.unified_lock(str(shared)):  # same thread: no deadlock
            assert json.loads(path.read_text(encoding="utf-8"))["token"] == owner["token"]
        assert path.is_file()  # the outer hold is still active
    assert not path.exists()
    assert sorted(p.name for p in path.parent.iterdir()) == []  # no tombs, no temp files


def test_other_threads_wait_for_the_holder(shared):
    order = []
    inside = threading.Event()
    release = threading.Event()

    def holder():
        with ug.unified_lock(str(shared)):
            order.append("holder in")
            inside.set()
            release.wait(5)
            order.append("holder out")

    def waiter():
        inside.wait(5)
        with ug.unified_lock(str(shared)):
            order.append("waiter in")

    threads = [threading.Thread(target=holder), threading.Thread(target=waiter)]
    for thread in threads:
        thread.start()
    inside.wait(5)
    time.sleep(0.5)
    assert order == ["holder in"]
    release.set()
    for thread in threads:
        thread.join(10)
    assert order == ["holder in", "holder out", "waiter in"]
    assert not _lock_path(shared).exists()


def test_stale_lock_of_a_dead_process_is_broken(shared):
    path = _lock_path(shared)
    _write_lock(path, pid=_dead_pid())
    logs = []
    started = time.monotonic()
    with ug.unified_lock(str(shared), log=logs.append):
        assert json.loads(path.read_text(encoding="utf-8"))["pid"] == os.getpid()
    assert time.monotonic() - started < 5
    assert any("stale lock" in line and "instance 2" in line for line in logs), logs
    assert not path.exists()


def _foreign_owner(path, release, beat_every=0.05):
    """Another owner of ``path`` (pid = this process, like a second module copy on mobile): it
    beats like ``_heartbeat`` until ``release`` is set, then deletes its lock file."""
    _write_lock(path)

    def run():
        beat = 0
        while not release.wait(beat_every):
            beat += 1
            _write_lock(path, beat=beat)
        for _attempt in range(100):
            try:
                path.unlink()
                return
            except PermissionError:  # Windows: a waiter is reading it right now
                time.sleep(0.01)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


def test_live_owner_is_waited_for_not_broken(shared, monkeypatch):
    """An owner whose pid runs keeps its lock while its heartbeat moves, however long the wait."""
    monkeypatch.setattr(ug, "_LOCK_STALE_POLLS", 8)
    path = _lock_path(shared)
    release = threading.Event()
    owner = _foreign_owner(path, release)
    acquired = threading.Event()

    def take():
        with ug.unified_lock(str(shared)):
            acquired.set()

    thread, box = _in_thread(take)
    time.sleep(2.0)  # 20 polls, far beyond _LOCK_STALE_POLLS: the heartbeat keeps it alive
    assert not acquired.is_set()
    assert _read_owner(path)["token"] == "someone-else"
    release.set()
    owner.join(5)
    thread.join(5)
    assert acquired.is_set() and "error" not in box
    assert not path.exists()


@pytest.mark.parametrize("owner", [
    {},                               # a reused pid that runs but never beats
    {"host": "some-other-computer"},  # a shared folder: the pid cannot be checked
    None,                             # crashed between creating and writing the file
])
def test_lock_whose_heartbeat_stopped_is_broken(shared, monkeypatch, owner):
    monkeypatch.setattr(ug, "_LOCK_STALE_POLLS", 5)
    path = _lock_path(shared)
    if owner is None:
        path.parent.mkdir(parents=True)
        path.write_bytes(b"")
    else:
        _write_lock(path, **owner)
    logs = []
    with ug.unified_lock(str(shared), log=logs.append):
        assert json.loads(path.read_text(encoding="utf-8"))["pid"] == os.getpid()
    assert any("stale lock" in line for line in logs), logs


def test_holder_heartbeat_keeps_a_cross_process_waiter_out(shared, monkeypatch):
    monkeypatch.setattr(ug, "_LOCK_HEARTBEAT_SECONDS", 0.05)
    monkeypatch.setattr(ug, "_LOCK_STALE_POLLS", 5)
    path = _lock_path(shared)
    got = []
    with ug.unified_lock(str(shared)):
        # a second process would run exactly this; it must not take the live lock
        thread, box = _in_thread(lambda: ug._acquire_lock_file(str(path), "other-process", print) or got.append(1))
        time.sleep(1.5)
        assert got == []
        assert json.loads(path.read_text(encoding="utf-8"))["beat"] > 0
    thread.join(5)
    assert got == [1] and "error" not in box
    assert json.loads(path.read_text(encoding="utf-8"))["token"] == "other-process"
    ug._release_lock_file(str(path), "other-process")
    assert not path.exists()


def test_stop_while_waiting_raises_and_leaves_the_owner_alone(shared, monkeypatch):
    path = _lock_path(shared)
    release = threading.Event()
    owner = _foreign_owner(path, release)
    stop = {"now": False}
    monkeypatch.setattr(ug, "_stop_requested", lambda: stop["now"])

    def take():
        with ug.unified_lock(str(shared)):
            pass

    thread, box = _in_thread(take)
    time.sleep(0.3)
    stop["now"] = True
    thread.join(5)
    assert isinstance(box.get("error"), ug._DedupeStopped)
    assert _read_owner(path)["token"] == "someone-else"
    release.set()
    owner.join(5)


def test_rebuild_now_ignores_stop_while_waiting(shared, monkeypatch):
    """Rebuild Now is not part of a run: a stop flag left by an earlier run must not abort its wait."""
    monkeypatch.setattr(ug, "_stop_requested", _REAL_STOP_REQUESTED)
    monkeypatch.setenv("TRANSLATION_CANCELLED", "1")
    path = _lock_path(shared)
    release = threading.Event()
    owner = _foreign_owner(path, release)
    threading.Timer(0.5, release.set).start()
    calls, logs = [], []
    monkeypatch.setattr(ug, "_rebuild_locked", lambda *args: calls.append(path.is_file()))
    assert ug.rebuild_now(shared_dir=str(shared), log=logs.append) is True
    owner.join(5)
    assert calls == [True], logs
    assert not any("failed" in line for line in logs), logs


def test_pid_alive():
    assert ug._pid_alive(os.getpid()) is True
    assert ug._pid_alive(_dead_pid()) is False
    assert ug._pid_alive(0) is False


# --------------------------------------------------------------------------
# who holds the lock
# --------------------------------------------------------------------------

def test_merge_book_and_rebuild_run_under_the_lock(shared, monkeypatch):
    seen = []
    monkeypatch.setattr(ug, "_merge_book_locked", lambda *args: seen.append(("merge", _lock_path(shared).is_file())))
    monkeypatch.setattr(ug, "_rebuild_locked", lambda *args: seen.append(("rebuild", _lock_path(shared).is_file())))
    ug.merge_book(str(shared), "korean-english", [{"raw_name": "x"}], log=lambda *_: None)
    ug.rebuild(str(shared), "korean-english", log=lambda *_: None)
    assert seen == [("merge", True), ("rebuild", True)]
    assert not _lock_path(shared).exists()


def test_the_dedup_child_never_takes_the_lock():
    tree = ast.parse((SRC_DIR / "unified_glossary.py").read_text(encoding="utf-8-sig"), feature_version=(3, 10))
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    for name in ("_dedupe_job", "_incremental_merge", "_full_dedupe", "_run_dedupe", "_run_dedupe_in_subprocess"):
        names = {n.id for n in ast.walk(functions[name]) if isinstance(n, ast.Name)}
        assert not names & {"unified_lock", "merge_book", "rebuild", "_acquire_lock_file"}, name


# --------------------------------------------------------------------------
# real merges racing each other
# --------------------------------------------------------------------------

ITERATIONS = 20
PER_ITERATION = 4


def _expected(books):
    return [worker.raw_name(book, i, j) for book in books for i in range(ITERATIONS) for j in range(PER_ITERATION)]


def _unified_text(shared_dir):
    csv_path = Path(ug.unified_paths(str(shared_dir), worker.KEY)[2])
    return csv_path.read_text(encoding="utf-8")


def test_two_processes_merging_into_one_unified_glossary_lose_nothing(tmp_path):
    shared_dir = tmp_path / "Glossary"
    shared_dir.mkdir()
    ctx = multiprocessing.get_context("spawn")
    start = ctx.Barrier(2)
    procs = [
        ctx.Process(target=worker.merge_many,
                    args=(str(SRC_DIR), str(shared_dir), book, ITERATIONS, PER_ITERATION, start))
        for book in ("A", "B")
    ]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(600)
    assert [proc.exitcode for proc in procs] == [0, 0]
    text = _unified_text(shared_dir)
    missing = [name for name in _expected(("A", "B")) if name not in text]
    assert not missing, f"{len(missing)} entries lost: {missing[:10]}"
    leftovers = [p.name for p in Path(ug.unified_root(str(shared_dir))).iterdir() if p.is_file()]
    assert leftovers == [], leftovers  # lock released, no stale tombs


def test_two_threads_merging_into_one_unified_glossary_lose_nothing(tmp_path, monkeypatch):
    shared_dir = tmp_path / "Glossary"
    shared_dir.mkdir()
    for name, value in worker.MERGE_ENV.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("TRANSLATION_CANCELLED", raising=False)
    monkeypatch.setattr(ug, "_stop_requested", lambda: False)
    ug._extractor()
    start = threading.Barrier(2)
    errors = []

    def merge(book):
        try:
            start.wait(30)
            for iteration in range(ITERATIONS):
                ug.merge_book(str(shared_dir), worker.KEY,
                              worker.book_entries(book, iteration, PER_ITERATION), log=lambda *_: None)
        except BaseException as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=merge, args=(book,)) for book in ("C", "D")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(600)
    assert not errors, errors
    text = _unified_text(shared_dir)
    missing = [name for name in _expected(("C", "D")) if name not in text]
    assert not missing, f"{len(missing)} entries lost: {missing[:10]}"
    assert not _lock_path(shared_dir).exists()
