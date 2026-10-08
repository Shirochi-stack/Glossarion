"""Host tests for U3 jobs: JobService, job_kinds adapters, BackgroundExecution, notifications, Jobs UI.

Run from src/mobile with the mobile venv (Flet installed):
    .venv/Scripts/python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_jobs.py

The shared backend (``job_runner``, ``stop_control``, ``headless_owner``,
``shutdown_utils``) is replaced by ``FakeBackend``, which follows the U3
contracts (JOB_LOCK held around the owner build, scoped process state restored
after every job, stop flags published before the ``set_stop_requested`` latch)
and records every call. Native calls go to ``FakeNative``. The Flet parts
build screens in the in-memory fake Flet session from ``test_bootstrap`` and
are skipped when Flet is missing.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import json
import os
import shutil
import sys
import threading
import time
import types
from pathlib import Path

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = MOBILE_DIR / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from glossarion_mobile import job_kinds  # noqa: E402
from glossarion_mobile.services.background import (  # noqa: E402
    CONTINUED_ID_PREFIX,
    PREF_BATTERY_PROMPT,
    PREF_KEEP_SCREEN_ON,
    BackgroundExecution,
)
from glossarion_mobile.services.jobs import (  # noqa: E402
    JobBackend,
    JobError,
    JobService,
    JobSnapshot,
    JobSpec,
    JobState,
    Progress,
    format_duration,
    notification_text,
    progress_line,
    strip_model_for,
)
from glossarion_mobile.services.notifications import JobNotifications, done_text  # noqa: E402

try:
    import flet  # noqa: F401

    HAVE_FLET = True
except ImportError:
    HAVE_FLET = False

needs_flet = pytest.mark.skipif(not HAVE_FLET, reason="Flet is not installed")
TIMEOUT = 5.0


# ==========================================================================
# Fakes
# ==========================================================================


class TrackingLock:
    """RLock that remembers which thread holds it (JOB_LOCK stand-in)."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.holder = None
        self.depth = 0
        self.acquired = 0

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        ok = self._lock.acquire(blocking, timeout)
        if ok:
            self.holder = threading.current_thread().name
            self.depth += 1
            self.acquired += 1
        return ok

    def release(self) -> None:
        self.depth -= 1
        if self.depth == 0:
            self.holder = None
        self._lock.release()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        self.release()

    def held_by_me(self) -> bool:
        return self.holder == threading.current_thread().name and self.depth > 0


class FakeOwner:
    def __init__(self, backend: "FakeBackend", config: dict, host) -> None:
        self.backend = backend
        self.config = config
        self.host = host
        self.stop_requested = False
        self.graceful_stop_var = config.get("graceful_stop", True)
        self.wait_for_chunks_var = config.get("wait_for_chunks", True)
        self.selected_files = []
        self.calls = []
        self.built_on = threading.current_thread().name
        self.built_with_lock = backend.lock.held_by_me()
        self.built_env_marker = os.environ.get("FAKE_SCOPE")

    def _resolve_translation_output_dir(self, path: str) -> str:
        return os.path.join(self.backend.out_root, os.path.splitext(os.path.basename(path))[0])

    def _prepare_translation_run(self, files):
        self.calls.append(("prepare", list(files)))
        if self.backend.on_prepare is not None:
            hook, self.backend.on_prepare = self.backend.on_prepare, None
            hook(self)
        return {"files": list(files)}

    def _translation_worker(self, request):
        self.calls.append(("worker", request))
        return self.backend.behavior(self, request)

    def run_glossary_extraction_direct(self, force_balanced_request_merging=False):
        self.calls.append(("glossary", list(self.selected_files), force_balanced_request_merging))
        return None

    def _run_epub_compile(self, folder):
        self.calls.append(("compile_epub", folder))
        path = os.path.join(folder, os.path.basename(folder) + ".epub")
        Path(path).write_bytes(b"epub")
        return types.SimpleNamespace(ok=True, path=path, error=None)

    def _run_pdf_compile(self, folder):
        self.calls.append(("compile_pdf", folder))
        return types.SimpleNamespace(ok=False, path=None, error="WeasyPrint missing")


class FakeWatcher:
    def __init__(self, resolver, host, interval) -> None:
        self.resolver = resolver
        self.host = host
        self.interval = interval
        self.started = self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True


class FakeBackend:
    """Follows the job_runner / stop_control contracts and records every call."""

    def __init__(self, out_root: str) -> None:
        self.out_root = out_root
        self.lock = TrackingLock()
        self.events = []
        self.stop_calls = []
        self.owners = []
        self.watchers = []
        self.restored = []
        self.run_ids = 0
        self.forced = threading.Event()
        self.behavior = lambda owner, request: quick_worker(owner, request)
        self.on_reset = None
        self.on_prepare = None
        self.scopes_active = 0

    @property
    def job_lock(self):
        return self.lock

    @contextlib.contextmanager
    def scoped_process_state(self, *, capture_stdout, lock):
        with lock:
            saved = dict(os.environ)
            self.scopes_active += 1
            os.environ["FAKE_SCOPE"] = "1"
            self.events.append(("scope_enter", threading.current_thread().name))
            try:
                yield
            finally:
                os.environ.clear()
                os.environ.update(saved)
                self.scopes_active -= 1
                self.events.append(("scope_exit", threading.current_thread().name))

    def make_owner(self, config, *, host):
        owner = FakeOwner(self, config, host)
        self.owners.append(owner)
        self.events.append(("owner", threading.current_thread().name))
        return owner

    def reset_for_new_run(self, kind):
        self.run_ids += 1
        self.events.append(("reset", kind))
        if self.on_reset is not None:
            hook, self.on_reset = self.on_reset, None
            hook()
        os.environ.pop("FAKE_STOP_MODE", None)  # reset clears the stop flags
        self.forced.clear()
        return self.run_ids

    def request_stop(self, *, graceful, wait_for_chunks, force, set_stop_requested, log, kind="translation",
                     job_kind="", owner=None):
        # stop_control order: env flags first, then the shared latch
        os.environ["FAKE_STOP_MODE"] = "force" if force else ("graceful" if graceful else "immediate")
        env_at_latch = []

        def latch():
            env_at_latch.append(os.environ.get("FAKE_STOP_MODE"))
            set_stop_requested()

        if force:
            self.forced.set()
        latch()
        self.stop_calls.append({"graceful": graceful, "wait_for_chunks": wait_for_chunks, "force": force,
                                "env_at_latch": env_at_latch[0], "thread": threading.current_thread().name,
                                "kind": kind})
        self.events.append(("stop", "force" if force else ("graceful" if graceful else "immediate")))

    def progress_watcher(self, resolver, host, interval):
        watcher = FakeWatcher(resolver, host, interval)
        self.watchers.append(watcher)
        return watcher

    def restore_in_progress(self, *, input_files, output_dirs, config, app_dir):
        self.restored.append({"input_files": list(input_files), "output_dirs": dict(output_dirs),
                              "app_dir": app_dir, "lock_held": self.lock.held_by_me()})


def quick_worker(owner, request):
    for path in request["files"]:
        out = owner._resolve_translation_output_dir(path)
        os.makedirs(out, exist_ok=True)
        Path(out, os.path.basename(out) + ".epub").write_bytes(b"compiled")
        owner.host.emit("progress", total=3, completed=3, in_progress=0, failed=0)
    print("stdout line from the backend")
    return None


class Gate:
    """A worker that waits until released (or stopped), emitting progress."""

    def __init__(self) -> None:
        self.release = threading.Event()
        self.entered = threading.Event()
        self.ticks = 0

    def __call__(self, owner, request):
        self.entered.set()
        owner.host.emit("progress", total=10, completed=1, in_progress=1, failed=0)
        owner.host.emit("api_state", in_flight=2, queued=1)
        while not self.release.wait(0.02):
            if owner.host.is_stop_requested():
                if owner.backend.forced.is_set() or not owner.host.is_graceful_stop():
                    return None  # force / immediate: abandon at once
                self.ticks += 1
                if self.ticks > 500:  # graceful: in-flight chunks would finish (10 s cap)
                    return None
        return None


class FakeStore:
    def __init__(self, data=None) -> None:
        self.data = dict(data or {})
        self.calls = []
        self.job_running = False

    def flush(self):
        self.calls.append("flush")
        return False

    def snapshot(self):
        self.calls.append("snapshot")
        return json.loads(json.dumps(self.data))

    def set_job_running(self, running):
        self.calls.append(f"running:{running}")
        self.job_running = running

    def get(self, key, default=None):
        return self.data.get(key, default)


def make_service(tmp_path, *, store=None, backend=None, **kwargs) -> tuple:
    backend = backend or FakeBackend(str(tmp_path / "out"))
    service = JobService(jobs_dir=tmp_path / "jobs", logs_dir=tmp_path / "logs", data_dir=tmp_path / "data",
                         config_store=store if store is not None else FakeStore(), backend=backend, **kwargs)
    return service, backend


def epub(tmp_path, name="Book.epub") -> str:
    path = tmp_path / "in" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"PK fake epub")
    return str(path)


def wait_for(predicate, timeout=TIMEOUT) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def state_of(service, job_id):
    snap = service.snapshot(job_id)
    return snap.state if snap is not None else None


# ==========================================================================
# JobService: running, queueing, config snapshot, process state
# ==========================================================================


def test_job_runs_on_gl_job_thread_with_owner_built_under_the_job_lock(tmp_path):
    store = FakeStore({"model": "authgpt/gpt-6-luna", "batch_size": 5})
    service, backend = make_service(tmp_path, store=store)
    transitions = []
    service.on_transition(lambda snap, prev: transitions.append((snap.state, prev)))
    source = epub(tmp_path)
    job_id = service.submit(JobSpec("translate", "Book.epub", (source,), params={"config_overrides": {"model": "x/y"}}))
    assert service.wait_idle(TIMEOUT)
    snap = service.snapshot(job_id)
    assert snap.state is JobState.DONE and snap.error is None
    owner = backend.owners[0]
    # HeadlessOwner built on the job thread, inside scoped_process_state, holding JOB_LOCK
    assert owner.built_on == "gl-job" and owner.built_with_lock and owner.built_env_marker == "1"
    assert "FAKE_SCOPE" not in os.environ and backend.scopes_active == 0  # process state restored
    # config: flushed + snapshotted at job start; overrides merged; banner flag on then off
    assert store.calls[:3] == ["flush", "snapshot", "running:True"] and store.calls[-1] == "running:False"
    assert owner.config["model"] == "x/y" and owner.config["batch_size"] == 5
    # reset before the owner; the adapter called the shared pair with the input
    order = [e[0] for e in backend.events]
    assert order.index("reset") < order.index("owner") < order.index("scope_exit")
    assert owner.calls[0] == ("prepare", [os.path.abspath(source)])
    out_dir = os.path.join(backend.out_root, "Book")
    assert snap.output_dirs == {os.path.normcase(os.path.abspath(source)): out_dir}
    assert snap.outputs == (os.path.join(out_dir, "Book.epub"),)
    assert snap.progress.total == 3 and snap.progress.completed == 3
    assert [t[0] for t in transitions] == [JobState.QUEUED, JobState.STARTING, JobState.RUNNING, JobState.DONE]
    # stdout is the scoped capture's business: the fake does not tee, the log still has the job lines
    lines = service.read_log_tail(job_id)
    assert lines[0].startswith("▶ Translating Book.epub") and lines[-1].startswith("■ Done")
    watcher = backend.watchers[0]
    assert watcher.started and watcher.stopped and watcher.interval == 2.0
    assert watcher.resolver() == out_dir and watcher.resolver(source) == out_dir
    assert not os.path.exists(service.active_state_path)  # nothing running, nothing queued
    service.close()


def test_submit_queues_after_current_and_queue_controls(tmp_path):
    service, backend = make_service(tmp_path)
    gate = Gate()
    backend.behavior = gate
    a = service.submit(JobSpec("translate", "A", (epub(tmp_path, "A.epub"),)))
    assert gate.entered.wait(TIMEOUT)
    backend.behavior = quick_worker
    b = service.submit(JobSpec("translate", "B", (epub(tmp_path, "B.epub"),)))
    c = service.submit(JobSpec("translate", "C", (epub(tmp_path, "C.epub"),)))
    view = service.view()
    assert view.active.id == a and [s.id for s in view.queue] == [b, c] and view.queued_count == 2
    assert all(s.state is JobState.QUEUED for s in view.queue)
    saved = json.loads(Path(service.active_state_path).read_text(encoding="utf-8"))
    assert saved["active"]["id"] == a and [q["id"] for q in saved["queue"]] == [b, c]
    assert service.move_queued(c, 0) and [s.id for s in service.view().queue] == [c, b]
    assert service.cancel_queued(b) and state_of(service, b) is JobState.CANCELLED
    assert service.view().history[0].id == b and service.view().history[0].started is None
    service.pause_queue()
    gate.release.set()
    assert wait_for(lambda: state_of(service, a) is JobState.DONE)
    time.sleep(0.1)
    assert state_of(service, c) is JobState.QUEUED and service.view().paused  # paused: C waits
    service.resume_queue()
    assert service.wait_idle(TIMEOUT) and state_of(service, c) is JobState.DONE
    assert [o.config is not None for o in backend.owners] == [True, True]  # B never got an owner
    service.close()


def test_unknown_kind_is_rejected(tmp_path):
    service, _backend = make_service(tmp_path)
    with pytest.raises(JobError):
        service.submit(JobSpec("nope", "x"))
    service.close()


SRC_DIR = MOBILE_DIR.parent


def test_real_job_runner_scope_watcher_and_stdout_capture(tmp_path):
    """JobService with the shared job_runner: env restored key by key, stdout teed, real ProgressWatcher."""
    if str(SRC_DIR) not in sys.path:
        sys.path.append(str(SRC_DIR))
    job_runner = pytest.importorskip("job_runner")
    pytest.importorskip("library_core")
    if not hasattr(job_runner, "ProgressWatcher"):
        pytest.skip("job_runner has no ProgressWatcher yet")
    fake = FakeBackend(str(tmp_path / "out"))

    class RealScopeBackend(JobBackend):
        def make_owner(self, config, *, host):
            owner = FakeOwner(fake, config, host)
            owner.lock_held = job_runner.JOB_LOCK._is_owned()
            fake.owners.append(owner)
            return owner

        def reset_for_new_run(self, kind):
            return 1

        def request_stop(self, *, graceful, wait_for_chunks, force, set_stop_requested, log, **kwargs):
            set_stop_requested()

    def worker(owner, request):
        out = owner._resolve_translation_output_dir(request["files"][0])
        os.makedirs(out, exist_ok=True)
        Path(out, "response_1.html").write_text("<p>x</p>", encoding="utf-8")
        progress = {"chapters": {"1": {"status": "completed", "output_file": "response_1.html"},
                                 "2": {"status": "in_progress"}, "3": {"status": "pending"}}}
        Path(out, "translation_progress.json").write_text(json.dumps(progress), encoding="utf-8")
        os.environ["U3_JOB_SCOPE_TEST"] = "1"
        print("hello from the backend")
        time.sleep(0.2)
        return None

    fake.behavior = worker
    service = JobService(jobs_dir=tmp_path / "jobs", config_store=FakeStore(), backend=RealScopeBackend(),
                         watcher_interval=0.05)
    os.environ.pop("U3_JOB_SCOPE_TEST", None)
    job_id = service.submit(JobSpec("translate", "Book", (epub(tmp_path),)))
    assert service.wait_idle(TIMEOUT)
    snap = service.snapshot(job_id)
    assert snap.state is JobState.DONE, snap.error
    assert fake.owners[0].lock_held  # HeadlessOwner built while JOB_LOCK is held
    assert "U3_JOB_SCOPE_TEST" not in os.environ  # restored after the job
    assert any("hello from the backend" in line for line in service.read_log_tail(job_id))
    # the counts are the shared library_core summary (the Library card numbers), not derived here
    assert (snap.progress.total, snap.progress.completed) == (3, 1) and snap.progress.in_progress >= 1
    service.close()


# ==========================================================================
# Stop semantics
# ==========================================================================


def test_graceful_stop_then_force_within_2s(tmp_path):
    service, backend = make_service(tmp_path, store=FakeStore({"graceful_stop": True, "wait_for_chunks": True}))
    gate = Gate()
    backend.behavior = gate
    job_id = service.submit(JobSpec("translate", "Book", (epub(tmp_path),)))
    assert gate.entered.wait(TIMEOUT)
    assert wait_for(lambda: state_of(service, job_id) is JobState.RUNNING)
    assert service.request_stop() == "graceful"
    snap = service.snapshot()
    assert snap.state is JobState.STOPPING and snap.stop_mode == "graceful"
    first = backend.stop_calls[0]
    assert first == {"graceful": True, "wait_for_chunks": True, "force": False, "env_at_latch": "graceful",
                     "thread": "MainThread", "kind": "translation"}
    owner = backend.owners[0]
    assert owner.stop_requested and owner.graceful_stop_active is True  # the desktop latch attributes
    time.sleep(0.1)
    assert state_of(service, job_id) is JobState.STOPPING  # graceful: in-flight work continues
    started = time.monotonic()
    assert service.request_stop() == "force"  # second tap forces
    assert backend.stop_calls[1]["force"] and not backend.stop_calls[1]["graceful"]
    assert backend.stop_calls[1]["env_at_latch"] == "force" and owner.graceful_stop_active is False
    assert wait_for(lambda: state_of(service, job_id) is JobState.CANCELLED, 2.0)
    assert time.monotonic() - started < 2.0
    snap = service.snapshot(job_id)
    assert snap.stopped and snap.state_label == "Stopped" and snap.stop_mode == "force"
    service.close()


def test_immediate_stop_when_graceful_stop_is_off_and_system_stops_never_escalate(tmp_path):
    service, backend = make_service(tmp_path, store=FakeStore({"graceful_stop": False}))
    gate = Gate()
    backend.behavior = gate
    job_id = service.submit(JobSpec("translate", "Book", (epub(tmp_path),)))
    assert gate.entered.wait(TIMEOUT)
    assert service.request_stop(escalate=False) == "immediate"
    assert service.snapshot().state is JobState.FORCE_STOPPING if service.snapshot() else True
    call = backend.stop_calls[0]
    assert not call["graceful"] and not call["force"]  # stop_control ignores wait_for_chunks without graceful
    assert wait_for(lambda: state_of(service, job_id) is JobState.CANCELLED)
    service.close()

    service, backend = make_service(tmp_path / "b", store=FakeStore({"graceful_stop": True}))
    gate = Gate()
    backend.behavior = gate
    job_id = service.submit(JobSpec("translate", "Book", (epub(tmp_path),)))
    assert gate.entered.wait(TIMEOUT)
    assert service.request_stop(escalate=False) == "graceful"
    assert service.request_stop(escalate=False) == "graceful"  # a second system stop stays graceful
    assert len(backend.stop_calls) == 1
    gate.release.set()
    assert wait_for(lambda: state_of(service, job_id) is JobState.CANCELLED)
    service.close()


def test_stop_during_start_is_reissued_after_the_run_reset(tmp_path):
    service, backend = make_service(tmp_path)
    holder = {}
    backend.on_reset = lambda: holder.setdefault("mode", service.request_stop())
    job_id = service.submit(JobSpec("translate", "Book", (epub(tmp_path),)))
    assert service.wait_idle(TIMEOUT)
    assert holder["mode"] == "graceful"
    kinds = [e for e in backend.events if e[0] in ("reset", "stop", "owner")]
    # the first stop happened during the reset (its flags were cleared); it is issued again after the owner
    assert kinds[0] == ("reset", "translation") and kinds[1] == ("stop", "graceful")
    assert kinds[2][0] == "owner" and kinds[3] == ("stop", "graceful")
    assert state_of(service, job_id) is JobState.CANCELLED
    service.close()


def test_job_backend_drives_the_shared_modules_in_desktop_order(tmp_path, monkeypatch):
    import stop_control as shared_stop_control

    calls = []
    cleanup_gate = threading.Event()
    stop_control = types.ModuleType("stop_control")
    stop_control.apply_force_stop_flags = lambda: calls.append("force_flags")
    stop_control.reset_api_watchdog = lambda clear_stale_external_files=True: calls.append(
        f"watchdog:{clear_stale_external_files}")
    # the stop tail and the cleanup wait are the shared desktop code itself
    stop_control.announce_stop = shared_stop_control.announce_stop
    stop_control.stop_epub_converter = shared_stop_control.stop_epub_converter
    stop_control.wait_for_stop_cleanup = shared_stop_control.wait_for_stop_cleanup

    def request_stop(*, graceful, wait_for_chunks, force=False, set_stop_requested, log=print, clear_watchdog=None,
                     stop_flag_hook=None, cleanup_thread_created=None, thread_name=None):
        calls.append(("request_stop", graceful, wait_for_chunks, force, clear_watchdog is not None,
                      stop_flag_hook is not None))
        set_stop_requested()
        if not graceful:  # immediate: the module flag hook, then the background cleanup
            if stop_flag_hook is not None:
                stop_flag_hook()
            if clear_watchdog is not None:
                clear_watchdog()
            cleanup = threading.Thread(target=cleanup_gate.wait, args=(10,), daemon=True)
            if cleanup_thread_created is not None:
                cleanup_thread_created(cleanup)
            cleanup.start()

    stop_control.request_stop = request_stop
    stop_control.reset_for_new_run = lambda kind="translation": calls.append(("reset", kind)) or "run-1"
    converter = types.ModuleType("epub_converter")
    converter.set_stop_flag = lambda value: calls.append(("converter_stop", value))
    shutdown_utils = types.ModuleType("shutdown_utils")

    def restore(**kwargs):
        calls.append(("restore", kwargs["input_files"], kwargs["output_dir_resolver"]("A.epub"),
                      kwargs["output_dir_resolver"]("missing"), kwargs["app_dir"]))

    shutdown_utils.restore_in_progress_rows_for_shutdown = restore
    headless_owner = types.ModuleType("headless_owner")
    headless_owner.HeadlessOwner = lambda config, host=None: ("owner", config, host)
    translation = types.ModuleType("TransateKRtoEN")
    translation.set_stop_flag = lambda value: calls.append(("translation_stop_flag", value))
    for name, module in (("stop_control", stop_control), ("epub_converter", converter),
                         ("shutdown_utils", shutdown_utils), ("headless_owner", headless_owner),
                         ("TransateKRtoEN", translation)):
        monkeypatch.setitem(sys.modules, name, module)
    backend = JobBackend()
    latched, logs = [], []
    monkeypatch.setenv("WAIT_FOR_CHUNKS", "1")
    backend.request_stop(graceful=True, wait_for_chunks=True, force=False, set_stop_requested=lambda: latched.append(1),
                         log=logs.append)
    assert calls == [("request_stop", True, True, False, True, True)] and latched == [1]
    assert logs == ["⏳ Graceful stop — waiting for in-flight API calls to complete..."]
    calls.clear()
    logs.clear()
    # the desktop double-click: graceful state dropped on the owner, force flags, the owner's watchdog reset,
    # the immediate protocol (+ translation_stop_flag hook and the cleanup's watchdog reset), converter flag
    owner = types.SimpleNamespace(graceful_stop_active=True, _last_stop_was_graceful=True,
                                  _reset_api_watchdog_progress=lambda clear_stale_external_files=True: calls.append(
                                      f"owner_watchdog:{clear_stale_external_files}"))
    backend.request_stop(graceful=True, wait_for_chunks=True, force=True, set_stop_requested=lambda: None,
                         log=logs.append, job_kind="compile_epub", owner=owner)
    assert calls == ["force_flags", "owner_watchdog:True", ("request_stop", False, True, False, True, True),
                     ("translation_stop_flag", True), "owner_watchdog:True", ("converter_stop", True)]
    assert owner.graceful_stop_active is False and owner._last_stop_was_graceful is False
    assert logs[0].startswith("⚡ Force stop") and logs[-1] == "🛑 Force stop requested — aborting queued/in-flight API calls"
    calls.clear()
    # without an owner the shared watchdog reset runs (stop_control.reset_api_watchdog)
    backend.request_stop(graceful=False, wait_for_chunks=True, force=False, set_stop_requested=lambda: None, log=print)
    assert calls == ["watchdog:True", ("request_stop", False, True, False, True, True), ("translation_stop_flag", True),
                     "watchdog:True"]
    calls.clear()
    # glossary jobs take the same protocol (it latches the owner's stop_requested for the extractor)
    backend.request_stop(graceful=True, wait_for_chunks=True, force=False, set_stop_requested=lambda: None,
                         log=print, kind="glossary")
    assert calls == [("request_stop", True, True, False, True, True)]
    calls.clear()
    # the immediate stop's cleanup thread is kept; the next job waits for it (desktop preflight, 3 s)
    cleanup = backend.stop_cleanup_thread
    assert cleanup is not None and cleanup.is_alive()
    real_wait = shared_stop_control.wait_for_stop_cleanup
    waited = []

    def quick_wait(thread, log=print, timeout=3.0):
        waited.append(timeout)  # the desktop's 3 s, shortened for the test
        return real_wait(thread, log, timeout=0.2)

    stop_control.wait_for_stop_cleanup = quick_wait
    started = time.monotonic()
    lines = []
    assert backend.wait_for_stop_cleanup(lines.append) is False  # still closing: the next run must not start
    assert waited == [3.0] and time.monotonic() - started < 3
    assert lines == ["⏳ Waiting for the previous translation HTTP session to close...",
                     "⏹️ Previous translation is still stopping; try Start again shortly."]
    cleanup_gate.set()
    cleanup.join(5)
    assert backend.wait_for_stop_cleanup(lines.append) is True and backend.stop_cleanup_thread is None
    calls.clear()
    assert backend.reset_for_new_run("glossary") == "run-1" and calls == [("reset", "glossary")]
    assert backend.make_owner({"model": "m"}, host="H") == ("owner", {"model": "m"}, "H")
    calls.clear()
    backend.restore_in_progress(input_files=["A.epub"], output_dirs={os.path.normcase(os.path.abspath("A.epub")): "/o/A"},
                                config={}, app_dir="/data")
    assert calls == [("restore", ["A.epub"], "/o/A", None, "/data")]


def test_request_stop_on_a_queued_job_cancels_it(tmp_path):
    service, backend = make_service(tmp_path)
    gate = Gate()
    backend.behavior = gate
    service.submit(JobSpec("translate", "A", (epub(tmp_path, "A.epub"),)))
    assert gate.entered.wait(TIMEOUT)
    b = service.submit(JobSpec("translate", "B", (epub(tmp_path, "B.epub"),)))
    assert service.request_stop(b) == "cancelled" and state_of(service, b) is JobState.CANCELLED
    assert service.request_stop("missing") is None
    gate.release.set()
    assert service.wait_idle(TIMEOUT)
    assert service.request_stop() is None  # nothing running
    service.close()


# ==========================================================================
# Failures, questions, logs, request stream
# ==========================================================================


def test_failures_job_errors_and_system_exit(tmp_path):
    service, backend = make_service(tmp_path)

    def boom(owner, request):
        raise ValueError("model refused")

    backend.behavior = boom
    a = service.submit(JobSpec("translate", "A", (epub(tmp_path),)))
    assert service.wait_idle(TIMEOUT)
    snap = service.snapshot(a)
    assert snap.state is JobState.FAILED and "model refused" in snap.error
    assert any("Traceback" in line for line in service.read_log_tail(a))

    b = service.submit(JobSpec("translate", "B", (str(tmp_path / "missing.epub"),)))
    assert service.wait_idle(TIMEOUT)
    assert service.snapshot(b).error == "File not found: missing.epub"

    backend.behavior = lambda owner, request: (_ for _ in ()).throw(SystemExit(0))
    c = service.submit(JobSpec("translate", "C", (epub(tmp_path),)))
    assert service.wait_idle(TIMEOUT) and state_of(service, c) is JobState.DONE

    backend.behavior = lambda owner, request: {"ok": False, "error": "API key invalid"}
    d = service.submit(JobSpec("translate", "D", (epub(tmp_path),)))
    assert service.wait_idle(TIMEOUT)
    assert state_of(service, d) is JobState.FAILED and service.snapshot(d).error == "API key invalid"
    assert service._thread.is_alive()  # the worker survived everything
    service.close()


def test_missing_shared_modules_fail_the_job_with_a_clear_message(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "job_runner", None)  # import raises ImportError
    service = JobService(jobs_dir=tmp_path / "jobs", config_store=FakeStore(), backend=JobBackend())
    job_id = service.submit(JobSpec("translate", "Book", (epub(tmp_path),)))
    assert service.wait_idle(TIMEOUT)
    snap = service.snapshot(job_id)
    assert snap.state is JobState.FAILED and "job_runner" in snap.error
    service.close()


def test_question_blocks_the_job_until_answered(tmp_path):
    service, backend = make_service(tmp_path)
    answers = {}

    def asking(owner, request):
        answers["no_listener"] = owner.host.ask("glossary_approval", path="g.csv", default="auto")
        return None

    backend.behavior = asking
    service.submit(JobSpec("translate", "A", (epub(tmp_path),)))
    assert service.wait_idle(TIMEOUT) and answers["no_listener"] == "auto"

    seen = []

    def listener(snap, question):
        seen.append((snap.question is not None, question["kind"], dict(question["data"])))
        strip = strip_model_for(service.snapshot())
        seen.append(("strip", strip.warning, strip.subtitle))
        threading.Timer(0.05, service.answer, (question["id"], "edited")).start()

    service.on_question(listener)

    def asking2(owner, request):
        answers["listener"] = owner.host.ask("glossary_approval", path="g.csv", default="auto")
        return None

    backend.behavior = asking2
    b = service.submit(JobSpec("translate", "B", (epub(tmp_path),)))
    assert service.wait_idle(TIMEOUT)
    assert answers["listener"] == "edited"
    assert seen[0] == (True, "glossary_approval", {"path": "g.csv"})
    assert seen[1] == ("strip", True, "Waiting for your glossary decision")
    assert service.snapshot(b).question is None and service.pending_question() is None
    service.close()


def test_logs_go_to_buffer_and_file_and_feed_the_shared_request_stream(tmp_path, monkeypatch):
    fake_stream = types.ModuleType("direct_text_stream")
    made = []

    class RequestStream:
        def __init__(self):
            self.segs = []
            self.fed = []
            self.drains = []

        def feed(self, line):
            self.fed.append(line)
            if line.startswith("[DIRECT_TEXT_RESPONSE_PAYLOAD]"):
                return "suppress-main-log"
            if "Sending API call" in line:
                self.segs.append({"label": f"Chapter {len(self.segs) + 1}", "phase": "processing", "text": "",
                                  "tokens": 12})
            return None

        def drain(self, final=False):
            self.drains.append(final)
            return False

        def segments(self, drain=True):
            assert drain is False  # screens never trigger the unbudgeted drain
            return list(self.segs)

    def make_stream(**kwargs):
        made.append(kwargs)
        return RequestStream()

    fake_stream.make_stream = make_stream
    monkeypatch.setitem(sys.modules, "direct_text_stream", fake_stream)
    service, backend = make_service(tmp_path)

    def chatty(owner, request):
        owner.host.log("📤 Sending API call now")
        owner.host.log("[DIRECT_TEXT_RESPONSE_PAYLOAD] {\"t\": 1}")
        owner.host.log("Hello world.\n\nSecond paragraph")  # streamed text keeps its blank line
        owner.host.log("")
        owner.host.log("❌ error: something", kind="error")
        owner.host.emit("log", text="📤 Sending API call now")
        owner.host.emit("custom_event", value=3)
        return None

    events = []
    service.on_event(lambda job_id, kind, data: events.append((kind, dict(data))))
    service.stream_drain_interval = 0.01
    backend.behavior = chatty
    job_id = service.submit(JobSpec("translate", "Book", (epub(tmp_path),)))
    assert service.wait_idle(TIMEOUT)
    buffer = service.log_buffer(job_id)
    texts = [line.text for line in buffer.snapshot()]
    assert "📤 Sending API call now" in texts and not any("PAYLOAD" in t for t in texts)
    assert "Second paragraph" in texts and "" not in texts  # the buffer and the file keep non-blank lines
    assert [line.kind for line in buffer.snapshot() if line.text.startswith("❌")] == ["error"]
    file_lines = service.read_log_tail(job_id)
    assert any("PAYLOAD" in t for t in file_lines)  # the file keeps everything
    stream = service.request_stream(job_id)
    assert isinstance(stream, RequestStream)
    # the desktop listener's view: every message whole, blank ones included (append_log -> _on_log_line)
    assert "Hello world.\n\nSecond paragraph" in stream.fed and "" in stream.fed
    assert [s["label"] for s in service.request_segments(job_id)] == ["Chapter 1", "Chapter 2"]
    # drained on the job side while it ran, and completely when it ended (still on the job thread)
    assert stream.drains and stream.drains[-1] is True
    assert made == [{"source_is_attachment": False, "request_number": None, "model": None}]
    assert events == [("custom_event", {"value": 3})]
    service.close()


# ==========================================================================
# Checkpoint, history and recovery
# ==========================================================================


def test_checkpoint_kill_recover_and_resume(tmp_path, monkeypatch):
    jobs_dir = tmp_path / "jobs"
    service, backend = make_service(tmp_path, checkpoint_interval=0.0)
    gate = Gate()
    backend.behavior = gate
    source = epub(tmp_path)
    spec = JobSpec("translate", "Book.epub", (source,), origin={"type": "chat", "cid": "3", "label": "Chat · Book"})
    job_id = service.submit(spec)
    assert gate.entered.wait(TIMEOUT)
    assert wait_for(lambda: json.loads(Path(service.active_state_path).read_text(encoding="utf-8"))["active"]
                    ["progress"]["completed"] == 1)
    service.checkpoint()
    saved = json.loads(Path(service.active_state_path).read_text(encoding="utf-8"))
    active = saved["active"]
    assert active["id"] == job_id and active["state"] == "RUNNING" and active["spec"]["inputs"] == [source]
    assert active["output_dirs"] == {os.path.normcase(os.path.abspath(source)): os.path.join(backend.out_root, "Book")}
    # "kill": copy the state as it is on disk, then let the old process finish elsewhere
    killed = tmp_path / "killed"
    shutil.copytree(jobs_dir, killed / "jobs")
    gate.release.set()
    assert service.wait_idle(TIMEOUT)
    service.close()

    backend2 = FakeBackend(str(tmp_path / "out"))
    store = FakeStore({"output_directory": "x"})
    service2 = JobService(jobs_dir=killed / "jobs", data_dir=tmp_path / "data", config_store=store, backend=backend2)
    # adopted at construction: a job submitted before recover() cannot overwrite the leftover state
    assert not (killed / "jobs" / "active.state").exists() and (killed / "jobs" / "interrupted.state").exists()
    assert [s.id for s in service2.interrupted] == [job_id] and service2.interrupted[0].restore_pending
    early = service2.submit(JobSpec("translate", "Other", (epub(tmp_path, "Other.epub"),)))
    assert service2.wait_idle(TIMEOUT) and state_of(service2, early) is JobState.DONE
    interrupted = service2.recover()
    assert [s.id for s in interrupted] == [job_id] and interrupted[0].state is JobState.INTERRUPTED
    assert interrupted[0].recovered and interrupted[0].spec == spec and not interrupted[0].restore_pending
    restored = backend2.restored[0]
    assert restored["input_files"] == [source] and restored["lock_held"]
    assert restored["output_dirs"] == active["output_dirs"] and restored["app_dir"] == str(tmp_path / "data")
    # survives another launch (rows are not restored twice), then Resume resubmits the same spec
    service3 = JobService(jobs_dir=killed / "jobs", config_store=store, backend=backend2)
    assert [s.id for s in service3.recover()] == [job_id] and len(backend2.restored) == 1
    new_id = service3.resume(job_id)
    assert new_id and new_id != job_id
    assert service3.wait_idle(TIMEOUT) and state_of(service3, new_id) is JobState.DONE
    assert service3.snapshot(new_id).spec == spec and backend2.owners[-1].calls[0][1] == [os.path.abspath(source)]
    assert service3.interrupted == () and not (killed / "jobs" / "interrupted.state").exists()
    history = {s.id: s for s in service3.view().history}
    assert history[job_id].resolution == "resumed"
    service2.close()
    service3.close()


def test_queued_jobs_are_recovered_too_and_discard(tmp_path):
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    queued = JobSnapshot(id="q1", spec=JobSpec("translate", "Q", ("x.epub",)), state=JobState.QUEUED, created=1.0)
    (jobs / "active.state").write_text(json.dumps({"version": 1, "active": None, "queue": [queued.to_dict()]}),
                                       encoding="utf-8")
    service, backend = make_service(tmp_path)
    recovered = service.recover()
    assert [s.id for s in recovered] == ["q1"] and backend.restored == []  # never started: nothing to restore
    assert not (jobs / "active.state").exists()
    assert service.discard("q1") and service.interrupted == ()
    assert service.view().history[0].resolution == "discarded"
    service.close()


def test_history_keeps_the_last_50_and_reloads(tmp_path):
    service, backend = make_service(tmp_path, history_limit=50)
    for index in range(53):
        service.submit(JobSpec("translate", f"Book {index}", (epub(tmp_path),)))
    assert service.wait_idle(TIMEOUT * 3)
    history = service.view().history
    assert len(history) == 50 and history[0].title == "Book 52"
    service.close()
    reloaded, _ = make_service(tmp_path)
    assert [s.id for s in reloaded.view().history] == [s.id for s in history]
    reloaded.clear_finished()
    assert reloaded.view().history == () and json.loads(
        Path(reloaded.history_state_path).read_text(encoding="utf-8"))["jobs"] == []
    reloaded.close()


def test_corrupt_state_files_are_moved_aside(tmp_path):
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    (jobs / "history.state").write_text("{not json", encoding="utf-8")
    service, _ = make_service(tmp_path)
    assert service.view().history == ()
    assert any(p.name.startswith("history.state.corrupt-") for p in jobs.iterdir())
    service.close()


# ==========================================================================
# Adapters (job_kinds) against the fake owner
# ==========================================================================


def test_glossary_and_compile_adapters_call_shared_owner_methods(tmp_path):
    service, backend = make_service(tmp_path)
    source = epub(tmp_path)
    g = service.submit(JobSpec("extract_glossary", "Book", (source,), params={"force_balanced_request_merging": True}))
    assert service.wait_idle(TIMEOUT) and state_of(service, g) is JobState.DONE
    owner = backend.owners[-1]
    assert owner.calls == [("glossary", [os.path.abspath(source)], True)]
    assert job_kinds.get_kind("extract_glossary").stop_kind == "glossary"
    assert ("reset", "glossary") in backend.events

    folder = tmp_path / "out" / "Book"
    folder.mkdir(parents=True, exist_ok=True)
    c = service.submit(JobSpec("compile_epub", "Book", (source,)))  # a source file maps to its output folder
    assert service.wait_idle(TIMEOUT)
    snap = service.snapshot(c)
    assert snap.state is JobState.DONE and snap.outputs == (str(folder / "Book.epub"),)
    assert snap.output_dir == str(folder)
    p = service.submit(JobSpec("compile_pdf", "Book", (), params={"folder": str(folder)}))
    assert service.wait_idle(TIMEOUT)
    assert state_of(service, p) is JobState.FAILED and service.snapshot(p).error == "WeasyPrint missing"
    missing = service.submit(JobSpec("compile_epub", "Nope", (), params={"folder": str(tmp_path / "nope")}))
    assert service.wait_idle(TIMEOUT) and service.snapshot(missing).error == "Folder not found: nope"
    service.close()


def _direct_text_store_error():
    try:
        import direct_text_store  # noqa: F401  (the shared run-environment helper)
    except ImportError as exc:
        return f"direct_text_store is not importable here ({exc}); use the project venv"
    return ""


@pytest.mark.skipif(bool(_direct_text_store_error()), reason=_direct_text_store_error() or "importable")
def test_direct_text_adapter_applies_options_and_run_environment(tmp_path, monkeypatch):
    calls = []
    fake_ho = types.ModuleType("headless_owner")

    from dataclasses import dataclass, field

    @dataclass
    class DirectTextRunOptions:
        selected_files: list = field(default_factory=list)
        force_stream_all: bool = True
        output_mode: str = "text"
        skip_thinking: bool = False

        def apply_to(self, owner):
            calls.append(("apply_to", list(self.selected_files), self.output_mode, self.skip_thinking))
            owner.selected_files = self.selected_files

    fake_ho.DirectTextRunOptions = DirectTextRunOptions
    monkeypatch.setitem(sys.modules, "headless_owner", fake_ho)
    service, backend = make_service(tmp_path)
    seen_env = {}
    keys = ("OUTPUT_DIRECTORY", "OUTPUT_DIR", "DIRECT_TEXT_ACTIVE", "DIRECT_TEXT_PRESERVE_MARKUP",
            "DIRECT_TEXT_ORDERED_BATCH", "ORDER_BATCH_REQUESTS_BY_SPINE")

    def worker(owner, request):
        seen_env.update({k: os.environ.get(k) for k in keys})
        calls.append(("env_steps", owner.calls[:2]))
        owner.manual_glossary_path = str(glossary)  # what the run's glossary step left on the owner
        return None

    backend.behavior = worker
    FakeOwner._apply_forced_streaming_environment = lambda self: self.calls.append(("forced_stream",))
    FakeOwner._apply_direct_text_runtime_environment = lambda self: self.calls.append(("dt_runtime",))
    try:
        text_input = tmp_path / "run" / "direct_text_1.txt"
        text_input.parent.mkdir(parents=True)
        text_input.write_text("안녕하세요", encoding="utf-8")
        glossary = tmp_path / "run" / "glossary.csv"
        glossary.write_text("type,raw_name,translated_name\n", encoding="utf-8")
        job_id = service.submit(JobSpec("direct_text", "Chat", (), params={
            "input_path": str(text_input),
            "options": {"output_mode": "vision", "skip_thinking": True, "bogus": 1},
            "output_root": str(tmp_path / "run"),
            "is_attachment": True,
        }))
        assert service.wait_idle(TIMEOUT)
        assert state_of(service, job_id) is JobState.DONE
        assert calls[0] == ("apply_to", [str(text_input)], "vision", True)
        # direct_text_store.apply_direct_text_run_environment: the dialog's env block, then the owner steps
        assert calls[1] == ("env_steps", [("forced_stream",), ("dt_runtime",)])
        run = str(tmp_path / "run")
        assert seen_env == {"OUTPUT_DIRECTORY": run, "OUTPUT_DIR": run, "DIRECT_TEXT_ACTIVE": "1",
                            "DIRECT_TEXT_PRESERVE_MARKUP": "1", "DIRECT_TEXT_ORDERED_BATCH": "1",
                            "ORDER_BATCH_REQUESTS_BY_SPINE": "1"}
        assert os.environ.get("DIRECT_TEXT_ACTIVE") is None  # restored with the scoped process state
        # what ChatStore.finish_run needs from the live run, recorded before the restore
        assert service.snapshot(job_id).result["glossary_path"] == str(glossary)
        # without an output root the job fails clearly
        bad = service.submit(JobSpec("direct_text", "Chat", (str(text_input),)))
        assert service.wait_idle(TIMEOUT)
        assert "has no output folder" in service.snapshot(bad).error
    finally:
        del FakeOwner._apply_forced_streaming_environment
        del FakeOwner._apply_direct_text_runtime_environment
        service.close()


def test_result_fields_and_compiled_outputs(tmp_path):
    assert job_kinds.result_fields(None) == (None, [], None)
    assert job_kinds.result_fields(False) == (False, [], None)
    obj = types.SimpleNamespace(ok=False, path=None, error=None, message="boom")
    assert job_kinds.result_fields(obj) == (False, [], "boom")
    assert job_kinds.result_fields({"success": True, "epub_path": "a.epub"}) == (True, ["a.epub"], None)
    (tmp_path / "Book.epub").write_bytes(b"x")
    (tmp_path / "Book_translated.txt").write_text("x", encoding="utf-8")
    (tmp_path / "chapter1.html").write_text("x", encoding="utf-8")
    assert [os.path.basename(p) for p in job_kinds.compiled_outputs([str(tmp_path), None])] == [
        "Book.epub", "Book_translated.txt"]
    assert job_kinds.get_kind("manga").stop_kind == "translation"  # registered since U8 (Tools › Manga)
    assert job_kinds.get_kind("manga_step").kind == "manga_step"
    with pytest.raises(KeyError):  # an unknown kind
        job_kinds.get_kind("no_such_kind")


# ==========================================================================
# Presentation helpers
# ==========================================================================


def _snap(**kwargs):
    base = dict(id="abc123abc123", spec=JobSpec("translate", "Book.epub", ("a",), origin={"type": "chat", "cid": "7"}),
                state=JobState.RUNNING, created=0.0, started=100.0)
    base.update(kwargs)
    return JobSnapshot(**base)


def test_strip_model_progress_line_and_notification_text():
    snap = _snap(progress=Progress(total=80, completed=12), in_flight=3)
    assert progress_line(snap, now=861.0) == "Ch 12/80 · 3 in flight · 12:41"
    model = strip_model_for(snap, queued=2, now=861.0)
    assert model.title == "Translating · Book.epub" and model.subtitle == "Ch 12/80 · 3 in flight · 12:41"
    assert model.progress == pytest.approx(0.15) and model.queued == 2 and model.state == "running"
    assert model.owner_chat == "7" and model.kind_icon == "TRANSLATE"
    assert strip_model_for(_snap(state=JobState.STOPPING)).state == "finishing"
    assert strip_model_for(_snap(state=JobState.FORCE_STOPPING)).state == "stopping"
    done = strip_model_for(_snap(state=JobState.DONE, finished=200.0, progress=Progress(total=80, completed=80)))
    assert (done.title, done.state, done.subtitle) == ("Done · Book.epub", "done", "80/80 chapters")
    failed = strip_model_for(_snap(state=JobState.FAILED, error="boom"))
    assert (failed.title, failed.state) == ("Failed · Book.epub", "failed")
    assert strip_model_for(_snap(state=JobState.CANCELLED)).title == "Stopped · Book.epub"
    assert strip_model_for(None) is None
    assert notification_text(snap) == "Translating Book.epub: 12/80 chapters · 3 in flight"
    assert format_duration(3723) == "1:02:03" and format_duration(None) == ""
    assert done_text(_snap(state=JobState.CANCELLED, progress=Progress(total=80, completed=12))) == (
        "Stopped: Book.epub (12/80)", "Tap to resume")
    assert done_text(_snap(state=JobState.DONE, progress=Progress(total=80, completed=80)))[0] == "Done: Book.epub"
    assert done_text(_snap(state=JobState.FAILED, error="x"))[0] == "Failed: Book.epub"


def test_snapshot_round_trips_through_json():
    snap = _snap(progress=Progress(total=5, completed=2, failed=1), outputs=("a.epub",), output_dirs={"a": "b"})
    again = JobSnapshot.from_dict(json.loads(json.dumps(snap.to_dict())))
    assert again.spec == snap.spec and again.progress == snap.progress and again.outputs == ("a.epub",)
    assert JobSnapshot.from_dict({"state": "bogus"}).state is JobState.INTERRUPTED


# ==========================================================================
# Dispatcher delivery (UI loop)
# ==========================================================================


def test_listeners_run_on_the_ui_loop_and_never_see_a_stale_view(tmp_path):
    from glossarion_mobile.services.dispatcher import UiDispatcher
    from glossarion_mobile.state.store import LoopGuard

    async def scenario():
        dispatcher = UiDispatcher(None, interval=0.01, guard=LoopGuard()).bind()
        dispatcher.start()
        service, backend = make_service(tmp_path, dispatcher=dispatcher)
        loop_thread = threading.get_ident()
        views, transitions = [], []
        service.subscribe(lambda v: views.append((threading.get_ident(), v)))
        service.on_transition(lambda s, p: transitions.append((threading.get_ident(), s.state)))
        job_id = service.submit(JobSpec("translate", "Book", (epub(tmp_path),)))
        for _ in range(500):
            if transitions and transitions[-1][1] is JobState.DONE:
                break
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.05)
        assert [t[1] for t in transitions] == [JobState.QUEUED, JobState.STARTING, JobState.RUNNING, JobState.DONE]
        assert all(t[0] == loop_thread for t in transitions + views)
        final = views[-1][1]
        assert final.active is None and final.history[0].id == job_id and final.history[0].state is JobState.DONE
        service.close()
        await dispatcher.stop()

    asyncio.run(scenario())


# ==========================================================================
# BackgroundExecution and notifications with a fake native bridge
# ==========================================================================


class FakeNative:
    def __init__(self) -> None:
        self.calls = []
        self.results = {"start_job_service": True, "begin_background_task": 7, "start_continued_processing": True,
                        "init_notifications": True, "show_notification": True, "save_to_downloads": "content://x"}

    async def call(self, method, *args, default=None, **kwargs):
        self.calls.append((method, args, kwargs))
        return self.results.get(method, default)

    async def init_notifications(self):
        self.calls.append(("init_notifications", (), {}))
        return True

    async def cancel_notification(self, notification_id):
        self.calls.append(("cancel_notification", (notification_id,), {}))

    async def clear_shared(self, delete_files=False):
        self.calls.append(("clear_shared", (delete_files,), {}))

    def names(self):
        return [c[0] for c in self.calls]


class FakePermissions:
    def __init__(self) -> None:
        self.asked = []

    async def request(self, name):
        self.asked.append(name)
        return "granted"


class FakeWakelock:
    def __init__(self) -> None:
        self.calls = []

    async def enable(self):
        self.calls.append("enable")

    async def disable(self):
        self.calls.append("disable")


class FakePrefs(dict):
    def set(self, key, value):
        self[key] = value


class FakeJobs:
    def __init__(self, snap=None) -> None:
        self.snap = snap
        self.stops = []

    def snapshot(self, job_id=None):
        return self.snap

    def request_stop(self, job_id=None, *, force=False, escalate=True, reason=""):
        self.stops.append({"escalate": escalate, "force": force, "reason": reason})
        return "graceful"


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self):
        return self.now


def _background(platform, **kwargs):
    native = FakeNative()
    clock = Clock()
    notifications = JobNotifications(native, platform=platform)
    jobs = kwargs.pop("jobs", FakeJobs())
    background = BackgroundExecution(native, platform=platform, jobs=jobs, notifications=notifications,
                                     prefs=kwargs.pop("prefs", FakePrefs()), permissions=FakePermissions(),
                                     wakelock=FakeWakelock(), clock=clock, **kwargs)
    return background, native, clock, jobs


def test_android_background_service_lifecycle():
    confirmations = []

    async def confirm(title, body):
        confirmations.append(title)
        return True

    navigated = []
    background, native, clock, jobs = _background("android", confirm=confirm, navigate_route=navigated.append)

    async def scenario():
        spec = JobSpec("translate", "Book.epub", ("a",))
        await background.prepare_for_run(spec)
        await background.prepare_for_run(spec)  # asked once only
        assert background.permissions.asked == ["NOTIFICATION", "IGNORE_BATTERY_OPTIMIZATIONS"]
        assert confirmations == ["Keep translations running"] and background.prefs[PREF_BATTERY_PROMPT]
        snap = _snap(state=JobState.STARTING)
        jobs.snap = snap
        await background.on_transition(snap, JobState.QUEUED)
        start = [c for c in native.calls if c[0] == "start_job_service"][0]
        assert start[1] == ("Glossarion", "Translating Book.epub")
        assert start[2]["buttons"] == [{"id": "stop", "text": "Stop"}, {"id": "open", "text": "Open"}]
        assert background.service_running and background.wakelock.calls == []  # Android: FGS wake lock only
        running = _snap(progress=Progress(total=80, completed=1))
        clock.now += 0.5
        assert not await background.job_progress(running)  # throttled to 1/s
        clock.now += 0.6
        assert await background.job_progress(running)
        updates = [c for c in native.calls if c[0] == "update_job_service"]
        assert updates[-1][2]["text"] == "Translating Book.epub: 1/80 chapters"
        # notification Stop -> request_stop (escalating, like a tap); Open -> the job route
        await background.on_foreground_event({"type": "button", "button_id": "stop"})
        await background.on_foreground_event({"type": "button", "button_id": "open"})
        assert jobs.stops[0]["escalate"] and navigated == [f"/job/{running.id}"]
        # Android 15 dataSync timeout -> graceful stop that never escalates + "Paused (system limit)"
        await background.on_foreground_event({"type": "timeout", "is_timeout": True})
        assert jobs.stops[1]["escalate"] is False
        paused = [c for c in native.calls if c[0] == "show_notification"][-1]
        assert paused[1][1] == "Paused (system limit): tap to resume" and paused[2]["channel_id"] == "jobs.action"
        assert paused[2]["payload"] == f"glossarion://app/job/{running.id}"
        background.service_running = True
        done = _snap(state=JobState.DONE, finished=200.0, outputs=("/o/Book.epub",),
                     progress=Progress(total=80, completed=80))
        await background.on_transition(done, JobState.RUNNING)
        assert "stop_job_service" in native.names() and not background.service_running
        note = [c for c in native.calls if c[0] == "show_notification"][-1]
        assert note[1][1:] == ("Done: Book.epub", "80/80 chapters") and note[2]["channel_id"] == "jobs.done"
        assert [a["id"] for a in note[2]["actions"]] == ["open", "share"]

    asyncio.run(scenario())


def test_queued_jobs_inherit_the_foreground_service():
    class QueueJobs(FakeJobs):
        def __init__(self):
            super().__init__()
            self.queue = ("next",)

        def view(self):
            return types.SimpleNamespace(queue=self.queue, paused=False)

    jobs = QueueJobs()
    background, native, _clock, _jobs = _background("android", jobs=jobs)

    async def scenario():
        await background.job_started(_snap(state=JobState.STARTING))
        assert native.names().count("start_job_service") == 1
        await background.job_finished(_snap(state=JobState.DONE, finished=1.0))
        assert "stop_job_service" not in native.names() and background.service_running  # next job inherits it
        await background.job_started(_snap(id="def456def456", state=JobState.STARTING))
        assert native.names().count("start_job_service") == 1  # updated, not restarted from the background
        jobs.queue = ()
        await background.job_finished(_snap(id="def456def456", state=JobState.DONE, finished=2.0))
        assert "stop_job_service" in native.names() and not background.service_running
        done = [c for c in native.calls if c[0] == "show_notification"]
        assert len(done) == 2  # one "Done" notification per job

    asyncio.run(scenario())


def test_ios_background_task_and_continued_processing():
    prefs = FakePrefs()
    background, native, clock, jobs = _background("ios", prefs=prefs)

    async def scenario():
        await background.prepare_for_run(JobSpec("translate", "Book.epub", ("a",)))
        cp = [c for c in native.calls if c[0] == "start_continued_processing"][0]
        assert cp[1][0].startswith(CONTINUED_ID_PREFIX) and cp[1][1] == "Book.epub"
        assert background.continued_started and background.permissions.asked == []
        snap = _snap(state=JobState.STARTING)
        jobs.snap = snap
        await background.on_transition(snap, JobState.QUEUED)
        begin = [c for c in native.calls if c[0] == "begin_background_task"][0]
        assert begin[1] == (f"job-{snap.id}",) and begin[2]["expiration_payload"] == f"glossarion://app/job/{snap.id}"
        assert background.bg_task_id == 7 and background.wakelock.calls == ["enable"]  # iOS default: keep screen on
        clock.now += 2
        await background.job_progress(_snap(progress=Progress(total=10, completed=4)))
        update = [c for c in native.calls if c[0] == "update_continued_processing"][-1]
        assert update[1][:2] == (4, 10)
        await background.on_background_task_event({"type": "expiring", "task_id": 7})
        assert background.bg_task_id == -1 and jobs.stops[-1]["escalate"] is False
        await background.on_background_task_event({"type": "continued_expired"})
        assert not background.continued_started
        assert [c for c in native.calls if c[0] == "show_notification"][-1][1][1].startswith("Paused")
        background.continued_started = True
        background.bg_task_id = 9
        await background.on_transition(_snap(state=JobState.CANCELLED, finished=1.0), JobState.STOPPING)
        assert ("finish_continued_processing", (False,), {}) in native.calls
        assert ("end_background_task", (9,), {}) in native.calls and background.wakelock.calls == ["enable", "disable"]

    asyncio.run(scenario())
    prefs[PREF_KEEP_SCREEN_ON] = False
    assert not background.keep_screen_on()


def test_desktop_background_is_a_no_op():
    background, native, _clock, _jobs = _background("desktop")

    async def scenario():
        await background.prepare_for_run(JobSpec("translate", "B", ()))
        await background.on_transition(_snap(state=JobState.STARTING), JobState.QUEUED)
        await background.job_progress(_snap(), force=True)
        assert [n for n in native.names() if n not in ("init_notifications",)] == []

    asyncio.run(scenario())


def test_notification_payloads_parse_back_to_routes():
    tap = JobNotifications.parse_event({"payload": "glossarion://app/job/abc123abc123", "action_id": "share"})
    assert (tap.action, tap.route, tap.job_id) == ("share", "/job/abc123abc123", "abc123abc123")
    assert JobNotifications.parse_event({"payload": "content://evil"}) is None
    assert JobNotifications.parse_event({"payload": "glossarion://app/job/x", "action_id": "rm"}).action == "open"


# ==========================================================================
# Flet: Jobs page, job detail, JobStrip, feature wiring
# ==========================================================================


# The fake Flet session and the app fixtures are shared with test_bootstrap.py (one copy).
_TB_SPEC = importlib.util.spec_from_file_location("_glossarion_tb_helpers_jobs", Path(__file__).with_name(
    "test_bootstrap.py"))
_TB = importlib.util.module_from_spec(_TB_SPEC)
_TB_SPEC.loader.exec_module(_TB)
storage = _TB.storage
app_env = _TB.app_env


def _tb():
    return _TB


def _match(route):
    from glossarion_mobile.ui.router import parse_route

    return parse_route(route)


@needs_flet
def test_jobs_screen_sections_actions_and_detail(tmp_path):
    pytest.importorskip("msgpack")
    from glossarion_mobile.ui.screens.job_detail import JobDetailScreen
    from glossarion_mobile.ui.screens.jobs import JobsScreen

    async def scenario():
        conn, session = _tb()._fake_session("android")
        page = session.page
        service, backend = make_service(tmp_path)
        gate = Gate()
        backend.behavior = gate
        running = service.submit(JobSpec("translate", "Running book", (epub(tmp_path),),
                                         origin={"type": "chat", "cid": "3", "label": "Chat · Novel"}))
        assert await asyncio.to_thread(gate.entered.wait, TIMEOUT)
        queued = service.submit(JobSpec("translate", "Queued book", (epub(tmp_path),)))
        navigated = []
        screen = JobsScreen(_match("/jobs"), service=service, page=page,
                            navigate=lambda name, params=None: navigated.append((name, params)))
        page.views[0].controls.append(screen.get_body())
        page.update()
        screen.did_show()
        keys = [getattr(c, "key", None) for c in screen.list_view.controls]
        assert f"job-row-{running}" in keys and "jobs-queue" in keys and screen.fab.visible
        assert [c.key for c in screen.queue_list.controls] == [f"job-row-{queued}"]
        dialog = screen._cancel(service.snapshot(queued))
        await dialog._on_confirm()
        assert state_of(service, queued) is JobState.CANCELLED
        screen._open(service.snapshot(running))
        assert navigated == [("jobs.detail", {"jid": running})]

        detail = JobDetailScreen(_match(f"/jobs/{running}"), service=service, page=page,
                                 navigate=lambda name, params=None: navigated.append((name, params)))
        page.views[0].controls.append(detail.get_body())
        page.update()
        detail.did_show()
        assert detail.state_chip.text == "Running" and detail.stop_button.visible
        assert detail.origin_button.visible and detail.origin_button.content == "Open Chat · Novel"
        assert detail.progress_text.value.startswith("Ch 1/10")
        detail._on_stop()
        assert service.snapshot(running).state is JobState.STOPPING
        detail._apply(service.snapshot(running))
        assert detail.stop_button.content == "Force stop"
        detail._on_stop()
        assert await asyncio.to_thread(wait_for, lambda: state_of(service, running) is JobState.CANCELLED)
        detail._apply(service.snapshot(running))
        assert detail.resume_button.visible and detail.resume_button.content == "Resume"
        detail._open_origin()
        assert navigated[-1] == ("chat", {"cid": "3"})
        screen._render(service.view())
        assert screen.view.history[0].id == running
        empty = JobsScreen(_match("/jobs"), service=make_service(tmp_path / "empty")[0])
        assert empty.get_body() is not None and empty.list_view.controls[0].key == "jobs-empty"
        missing = JobDetailScreen(_match("/jobs/nothere"), service=service)
        assert missing.get_body().key == "job-missing"
        detail.dispose()
        screen.dispose()
        service.close()

    asyncio.run(scenario())


@needs_flet
def test_job_strip_ended_states_and_dismiss():
    from glossarion_mobile.state.app_state import JobStripModel
    from glossarion_mobile.ui.shell.job_strip import JobStrip

    opened = []
    strip = JobStrip(on_open=lambda: opened.append(1))
    strip.set_model(JobStripModel("Translating · Book", "Ch 1/2", 0.5))
    assert strip.visible and strip.ring.visible and not strip.open_button.visible and strip.stop_button.visible
    strip.set_model(JobStripModel("Done · Book", "2/2 chapters", 1.0, state="done"))
    assert strip.visible and strip.end_icon.visible and not strip.ring.visible and strip.open_button.visible
    assert strip.open_button.content == "Open" and not strip.stop_button.visible
    strip.set_model(JobStripModel("Failed · Book", "boom", None, state="failed"))
    assert strip.open_button.content == "View"
    strip._on_drag_end(types.SimpleNamespace(primary_velocity=900.0))
    assert not strip.visible and strip.dismissed_state == "failed"
    strip.set_model(JobStripModel("Failed · Book", "boom again", None, state="failed"))
    assert not strip.visible  # same state: stays hidden
    strip.set_model(JobStripModel("Translating · Next", "", None, state="running"))
    assert strip.visible and strip.dismissed_state is None
    strip._open()
    assert opened == [1]


@needs_flet
def test_jobs_feature_wires_routes_strip_and_recovery(app_env, tmp_path):
    tb = _tb()
    from glossarion_mobile.ui.screens.files import FileBrowserScreen
    from glossarion_mobile.ui.screens.job_detail import JobDetailScreen
    from glossarion_mobile.ui.screens.jobs import JobsFeature, JobsScreen

    async def scenario():
        spec = importlib.util.spec_from_file_location("glossarion_mobile_app_main_u3", APP_DIR / "main.py")
        main_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(main_module)
        conn, session = tb._fake_session("android")
        page = session.page
        await main_module.main(page)
        await session.after_event(page)
        app = page.data
        try:
            # a job left behind by a killed process
            jobs_dir = Path(app.paths.data) / "jobs"
            jobs_dir.mkdir(parents=True, exist_ok=True)
            left = JobSnapshot(id="dead00dead00", spec=JobSpec("translate", "Old.epub", ("o.epub",)),
                               state=JobState.RUNNING, created=1.0, started=2.0)
            (jobs_dir / "active.state").write_text(json.dumps({"version": 1, "active": left.to_dict(),
                                                               "queue": []}), encoding="utf-8")
            backend = FakeBackend(str(tmp_path / "out"))
            service = JobService(jobs_dir=jobs_dir, data_dir=app.paths.data, config_store=FakeStore(),
                                 dispatcher=app.dispatcher, backend=backend)
            feature = JobsFeature(app, service=service)
            feature.attach(app)
            feature.background.permissions = FakePermissions()  # no real permission prompt in the fake session

            async def confirm(title, body):
                return True

            feature.background.confirm = confirm
            assert app.jobs is feature and app.job_service is service and app.files is feature.files
            recovered = await feature.recover()
            assert [s.id for s in recovered] == ["dead00dead00"] and feature.banner is not None
            assert backend.restored[0]["input_files"] == ["o.epub"]

            await tb._route(session, "/jobs")
            assert isinstance(app.shell.top_screen, JobsScreen)
            await tb._route(session, "/job/dead00dead00")  # notification alias -> job detail
            assert isinstance(app.shell.top_screen, JobDetailScreen)
            await tb._route(session, "/tools/files/output")
            assert isinstance(app.shell.top_screen, FileBrowserScreen)
            assert feature.screens_built == ["jobs", "jobs.detail", "tools.files"]

            gate = Gate()
            backend.behavior = gate
            book = Path(app.paths.data) / "Inbox" / "Book.epub"
            book.parent.mkdir(parents=True, exist_ok=True)
            book.write_bytes(b"x")
            job_id = await feature.submit(JobSpec("translate", "Book.epub", (str(book),)))
            assert feature.background.permissions.asked == ["NOTIFICATION", "IGNORE_BATTERY_OPTIMIZATIONS"]
            assert await asyncio.to_thread(gate.entered.wait, TIMEOUT)
            for _ in range(200):
                model = app.state.job_strip.value
                if model is not None and model.state == "running" and "Ch 1/10" in model.subtitle:
                    break
                await asyncio.sleep(0.02)
            model = app.state.job_strip.value
            assert model.title == "Translating · Book.epub" and app.state.jobs_badge.value.running == 1
            assert app.shell.global_strip.on_stop == feature.stop_from_strip
            assert feature.stop_from_strip() == "graceful"
            assert feature.stop_from_strip() == "force"
            for _ in range(200):
                model = app.state.job_strip.value
                if model is not None and model.state == "done":
                    break
                await asyncio.sleep(0.02)
            assert app.state.job_strip.value.title == "Stopped · Book.epub"
            assert state_of(service, job_id) is JobState.CANCELLED
            # going to the background checkpoints active.state
            await session.dispatch_event(page._i, "app_lifecycle_state_change", {"state": "hide"})
            assert not feature.background.app_visible
            feature.close()
        finally:
            if getattr(app, "settings", None) is not None:
                app.settings.close()
            await app.dispatcher.stop()
            await asyncio.sleep(0)

    asyncio.run(scenario())


@needs_flet
def test_text_editor_route_back_keeps_unsaved_edits(app_env, tmp_path, monkeypatch):
    """U7 review: the ``tools.text`` route (Chapters ✏️ Edit file, SDLXLIFF Edit Output, File browser
    Open with) must not drop unsaved edits on Android back / the iOS swipe / the app-bar arrow. Its
    View cannot pop while the text is dirty; Back asks first and Discard leaves through the app."""
    from glossarion_mobile.ui.tools import common as tools_common
    from glossarion_mobile.ui.tools.text_editor import TextEditorScreen

    spec = importlib.util.spec_from_file_location("_glossarion_jobs_tf_helpers", Path(__file__).with_name("test_ui_foundations.py"))
    tf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tf)
    shown = []
    show = tools_common.ChoiceDialog.show
    monkeypatch.setattr(tools_common.ChoiceDialog, "show", lambda self, page: (shown.append(self), show(self, page))[1])

    async def scenario():
        _m, conn, session, page, app = await tf._start("android")
        try:
            await tf._wait(lambda: app.state.engine_ready)
            assert await tf._wait(lambda: app.jobs is not None, timeout=10)
            root = app.jobs.file_roots()["output"]
            os.makedirs(root, exist_ok=True)
            note = Path(root) / "note.txt"
            note.write_text("original\n", encoding="utf-8")
            app.navigate_to("tools.text", {"fid": app.prefs.file_ref(str(note))})
            assert await tf._wait(lambda: tf._routes(page)[-1].startswith("/tools/text/"), timeout=5)
            entry = app.shell.stack[-1]
            screen = entry.screen
            assert isinstance(screen, TextEditorScreen)
            assert await tf._wait(lambda: screen.loaded is not None, timeout=5)
            screen.editor.value = "edited but unsaved\n"
            assert screen.dirty and entry.view.can_pop is False and callable(entry.view.on_confirm_pop)
            await entry.view.on_confirm_pop(None)  # Android back / iOS swipe / app-bar arrow
            assert await tf._wait(lambda: shown and shown[-1]._future is not None, timeout=5)
            assert app.shell.stack[-1].screen is screen  # still open while it asks
            shown[-1].choose("discard")
            assert await tf._wait(lambda: tf._routes(page) == ["/"], timeout=5)
            assert all(e.screen is not screen for e in app.shell.stack)
            assert note.read_text(encoding="utf-8") == "original\n"
        finally:
            await tf._stop(app)

    asyncio.run(scenario())


# ==========================================================================
# U3 review fixes: worker outcome, stops during set-up, stop protocol off the UI loop,
# cleanup wait, resume once, sign-in events, ordered background transitions
# ==========================================================================


def test_translate_adapter_maps_the_worker_outcome(tmp_path):
    """``_translation_worker`` returns run_translation_direct's result (False: the run did not
    translate); a False without a Stop is Failed, after a Stop it is Stopped (Resume)."""
    from glossarion_mobile.job_kinds.translate import NOT_COMPLETED

    service, backend = make_service(tmp_path)
    backend.behavior = lambda owner, request: False
    failed = service.submit(JobSpec("translate", "A", (epub(tmp_path),)))
    assert service.wait_idle(TIMEOUT)
    snap = service.snapshot(failed)
    assert snap.state is JobState.FAILED and snap.error == NOT_COMPLETED

    backend.behavior = lambda owner, request: True
    done = service.submit(JobSpec("translate", "B", (epub(tmp_path),)))
    assert service.wait_idle(TIMEOUT) and state_of(service, done) is JobState.DONE

    def stopped_then_false(owner, request):
        service.request_stop()  # the user stops; the worker then reports an unfinished run
        return False

    backend.behavior = stopped_then_false
    stopped = service.submit(JobSpec("translate", "C", (epub(tmp_path),)))
    assert service.wait_idle(TIMEOUT)
    snap = service.snapshot(stopped)
    assert snap.state is JobState.CANCELLED and snap.error is None and snap.state_label == "Stopped"
    service.close()


def test_a_stop_erased_by_the_run_set_up_still_stops_the_job(tmp_path):
    """A Stop that lands before the shared set-up's "Reset stop flags" block (the owner latch is
    cleared there, as on desktop) survives in the job's own latch: the worker never starts."""
    service, backend = make_service(tmp_path)

    def stop_then_reset(owner):
        service.request_stop()
        assert owner.stop_requested is True
        owner.stop_requested = False  # the desktop set-up: self.stop_requested = False, reset_stop_env ...

    backend.on_prepare = stop_then_reset
    job_id = service.submit(JobSpec("translate", "Book", (epub(tmp_path),)))
    assert service.wait_idle(TIMEOUT)
    owner = backend.owners[-1]
    assert [c[0] for c in owner.calls] == ["prepare"]  # no worker
    assert state_of(service, job_id) is JobState.CANCELLED
    assert "⏹️ Translation stopped before it started" in service.read_log_tail(job_id)
    service.close()


class LoopDispatcher:
    """A bound UiDispatcher stand-in: the test thread is "the UI loop"; posts run at once."""

    bound = True

    def __init__(self) -> None:
        self.loop_thread = threading.current_thread()

    def on_loop_thread(self) -> bool:
        return threading.current_thread() is self.loop_thread

    def post(self, fn, *args):
        fn(*args)
        return True


def test_stop_on_the_ui_loop_runs_the_protocol_off_the_loop(tmp_path):
    service, backend = make_service(tmp_path, dispatcher=LoopDispatcher())
    gate = Gate()
    backend.behavior = gate
    release_protocol = threading.Event()
    real_stop = backend.request_stop

    def slow_stop(**kwargs):
        release_protocol.wait(TIMEOUT)  # env writes / backend imports / watchdog reset take a while
        real_stop(**kwargs)

    backend.request_stop = slow_stop
    job_id = service.submit(JobSpec("translate", "Book", (epub(tmp_path),)))
    assert gate.entered.wait(TIMEOUT)
    started = time.monotonic()
    assert service.request_stop() == "graceful"
    assert time.monotonic() - started < 1.0  # the UI loop never waits for the protocol
    job = service._active
    assert job.stop_event.is_set() and state_of(service, job_id) is JobState.STOPPING
    release_protocol.set()
    assert wait_for(lambda: backend.stop_calls)
    assert backend.stop_calls[0]["thread"] == "gl-job-stop" and backend.stop_calls[0]["graceful"]
    assert service.wait_stops_idle(TIMEOUT)
    gate.release.set()
    assert service.wait_idle(TIMEOUT) and state_of(service, job_id) is JobState.CANCELLED
    # a protocol that arrives after its job ended never reaches the next job
    service._issue_stop_soon(job, "force")
    assert service.wait_stops_idle(TIMEOUT) and len(backend.stop_calls) == 1
    service.close()


def test_the_next_job_waits_for_the_previous_stop_cleanup(tmp_path):
    class CleanupBackend(FakeBackend):
        def __init__(self, out_root):
            super().__init__(out_root)
            self.cleanup_ready = False
            self.waits = []

        def wait_for_stop_cleanup(self, log):
            self.waits.append(self.lock.held_by_me())
            if not self.cleanup_ready:
                log("⏹️ Previous translation is still stopping; try Start again shortly.")
            return self.cleanup_ready

    backend = CleanupBackend(str(tmp_path / "out"))
    service, _ = make_service(tmp_path, backend=backend)
    blocked = service.submit(JobSpec("translate", "A", (epub(tmp_path),)))
    assert service.wait_idle(TIMEOUT)
    snap = service.snapshot(blocked)
    assert snap.state is JobState.FAILED and "still stopping" in snap.error
    assert not any(e[0] == "reset" for e in backend.events)  # the cancellation state was left alone
    backend.cleanup_ready = True
    ok = service.submit(JobSpec("translate", "B", (epub(tmp_path),)))
    assert service.wait_idle(TIMEOUT) and state_of(service, ok) is JobState.DONE
    assert backend.waits == [True, True]  # under the job lock, before the run reset
    service.close()


def test_an_interrupted_job_is_resumed_once(tmp_path):
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir(parents=True)
    spec = JobSpec("translate", "Book", (epub(tmp_path),))
    killed = JobSnapshot(id="killedjob001", spec=spec, state=JobState.RUNNING, created=1.0, started=2.0)
    (jobs_dir / "active.state").write_text(json.dumps({"version": 1, "saved_at": 3.0, "active": killed.to_dict(),
                                                       "queue": []}), encoding="utf-8")
    service, backend = make_service(tmp_path)
    assert [s.id for s in service.interrupted] == ["killedjob001"]
    first = service.resume("killedjob001")
    assert first and service.interrupted == ()
    assert service.resume("killedjob001") is None  # resolved "resumed": a second Resume is refused
    assert service.wait_idle(TIMEOUT)
    assert len(backend.owners) == 1
    service.close()


def test_a_lost_chatgpt_login_sends_one_sign_in_event(tmp_path):
    service, backend = make_service(tmp_path)

    def login_lost(owner, request):
        owner.host.log("🔄 AuthGPT: No valid token found – starting browser login…")
        owner.host.log("🔐 AuthGPT: Session expired – opening browser login…")
        return False

    events = []
    service.on_event(lambda job_id, kind, data: events.append((job_id, kind)))
    backend.behavior = login_lost
    job_id = service.submit(JobSpec("translate", "Book", (epub(tmp_path),)))
    assert service.wait_idle(TIMEOUT)
    assert events == [(job_id, "sign_in_required")]
    service.close()


def test_background_transitions_run_in_order():
    """A job that fails right after STARTING (Stop while starting, a missing Inbox file) must not
    leave the Android foreground service up: the transitions are handled one after another."""

    class SlowNative(FakeNative):
        def __init__(self):
            super().__init__()
            self.running = False

        async def call(self, method, *args, default=None, **kwargs):
            await asyncio.sleep(0.03)  # a platform-channel round trip
            if method == "start_job_service":
                self.running = True
            elif method == "stop_job_service":
                self.running = False
            return await super().call(method, *args, default=default, **kwargs)

    class Idle:
        def view(self):
            return types.SimpleNamespace(queue=(), paused=False)

        def snapshot(self, job_id=None):
            return None

    native = SlowNative()
    background = BackgroundExecution(native, platform="android", jobs=Idle(),
                                     notifications=JobNotifications(native, platform="android"),
                                     prefs=FakePrefs(), permissions=FakePermissions(), wakelock=FakeWakelock())

    async def scenario():
        starting = _snap(state=JobState.STARTING)
        failed = _snap(state=JobState.FAILED, finished=101.0, error="File not found: a")
        # JobsFeature spawns one task per transition
        await asyncio.gather(asyncio.ensure_future(background.on_transition(starting, JobState.QUEUED)),
                             asyncio.ensure_future(background.on_transition(failed, JobState.STARTING)))

    asyncio.run(scenario())
    names = native.names()
    assert names.index("start_job_service") < names.index("stop_job_service")
    assert not native.running and not background.service_running


def test_glossary_question_routes_to_the_owning_chat():
    from glossarion_mobile.services.notifications import question_route

    assert question_route(_snap()) == "/chat/7"  # the approval card lives in the chat
    plain = _snap(spec=JobSpec("translate", "Book.epub", ("a",)))
    assert question_route(plain) == f"/job/{plain.id}"
    tap = JobNotifications.parse_event({"payload": "glossarion://app/chat/7"})
    assert tap.route == "/chat/7" and tap.job_id is None and tap.action == "open"


# ==========================================================================
# Owner device report 6: glossary-review notification (Accept / Review, cancel on answer), progress while
# the app is hidden, the real permission state, a swiped job notification posted again
# ==========================================================================


class _HiddenPage:
    """The dispatcher's page (``test_ui_foundations._FakePage``): ``app_visible`` False parks the UI pump."""

    def __init__(self) -> None:
        self.app_visible = True
        self._visible = asyncio.Event()
        self._visible.set()
        self.calls = []

    def update(self, *controls):
        self.calls.append(controls)

    async def wait_until_visible(self):
        await self._visible.wait()

    def hide(self):
        self.app_visible = False
        self._visible.clear()

    def show(self):
        self.app_visible = True
        self._visible.set()


class Asker:
    """A worker that reports chapter 1, waits for ``go``, reports ``steps``, pauses, then blocks on a question."""

    def __init__(self, kind="direct_text_glossary_approval", steps=(), pause=0.0, **data) -> None:
        self.kind = kind
        self.steps = steps
        self.pause = pause
        self.data = data or {"path": "glossary.csv"}
        self.go = threading.Event()
        self.entered = threading.Event()
        self.answers = []

    def __call__(self, owner, request):
        owner.host.emit("progress", total=10, completed=1, in_progress=1, failed=0)
        self.entered.set()
        self.go.wait(TIMEOUT)
        for completed in self.steps:
            owner.host.emit("progress", total=10, completed=completed, in_progress=1, failed=0)
            time.sleep(0.02)
        time.sleep(self.pause)
        self.answers.append(owner.host.ask(self.kind, default=False, **self.data))
        return True


async def _until(predicate, timeout=TIMEOUT) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.01)
    return bool(predicate())


async def _settle(rounds=5):
    for _ in range(rounds):
        await asyncio.sleep(0)


def _jobs_feature(tmp_path, *, dispatcher=None, native=None, runs=None, platform="android"):
    """The real JobsFeature over a real JobService (FakeBackend) with a fake app: snackbars -> ``notes``,
    ``app.navigate`` -> ``routes``."""
    from glossarion_mobile.ui.screens.jobs import JobsFeature

    service, backend = make_service(tmp_path, dispatcher=dispatcher)
    notes, routes = [], []

    async def navigate(route):
        routes.append(route)

    app = types.SimpleNamespace(
        page=None, dispatcher=dispatcher, state=None, prefs=FakePrefs(), native=native or FakeNative(),
        paths=types.SimpleNamespace(data=str(tmp_path / "data"), logs=str(tmp_path / "logs")), shell=None,
        notify=lambda message, label=None, action=None: notes.append((message, label, action)), navigate=navigate,
        chat_feature=types.SimpleNamespace(runs=runs) if runs is not None else None)
    feature = JobsFeature(app, service=service)
    feature.platform = feature.background.platform = feature.notifications.platform = platform
    feature.background.permissions = FakePermissions()
    feature.attach(app)
    return feature, app, backend, notes, routes


def _service_texts(native):
    return [c[2].get("text") or "" for c in native.calls if c[0] == "update_job_service"]


def _posted(native, channel=None):
    return [c for c in native.calls if c[0] == "show_notification" and channel in (None, c[2].get("channel_id"))]


@needs_flet
def test_fgs_progress_updates_while_app_hidden(tmp_path):
    """Owner report: the ongoing notification froze once the app left the screen (the jobs view rides the UI
    pump, which parks while hidden). The ticker keeps it current: chapters and the glossary question."""
    from glossarion_mobile.services.dispatcher import UiDispatcher
    from glossarion_mobile.state.store import LoopGuard

    async def scenario():
        page = _HiddenPage()
        dispatcher = UiDispatcher(page, interval=0.01, guard=LoopGuard()).bind()
        dispatcher.start()
        native = FakeNative()
        feature, app, backend, notes, routes = _jobs_feature(tmp_path, dispatcher=dispatcher, native=native)
        background = feature.background
        background.update_interval = 0.05
        asker = Asker(steps=(2, 3, 4, 5), pause=0.4)
        backend.behavior = asker
        try:
            job_id = await feature.submit(JobSpec("translate", "Book.epub", (epub(tmp_path),),
                                                  origin={"type": "chat", "cid": "7"}))
            assert await asyncio.to_thread(asker.entered.wait, TIMEOUT)
            assert await _until(lambda: background.service_running)
            # visible: the jobs view feeds the notification (the ticker only sends a held-back update)
            assert await _until(lambda: not background.ticker_running)
            # the user leaves the app: Flet parks the pump, the lifecycle reaches BackgroundExecution
            page.hide()
            background.on_lifecycle("hide")
            assert background.ticker_running
            asker.go.set()
            assert await _until(lambda: any("waiting for your glossary decision" in t for t in _service_texts(native)))
            texts = _service_texts(native)
            assert any(t.endswith("5/10 chapters") or "5/10 chapters ·" in t for t in texts), texts
            assert not page.app_visible  # all of it while the app was hidden
            note = _posted(native, "jobs.action")[-1]  # hidden: the glossary question notified
            assert note[1][1] == "Glossary ready: review needed" and note[2]["payload"] == "glossarion://app/chat/7"
            assert [a["id"] for a in note[2]["actions"]] == ["accept", "open"]
            question = feature.service.pending_question(job_id)
            assert feature.service.answer(question["id"], True)
            assert await _until(lambda: "stop_job_service" in native.names())
            assert await _until(lambda: not background.ticker_running)  # stopped with the service
            assert asker.answers == [True]
            page.show()
            background.on_lifecycle("resume")
            assert not background.ticker_running
        finally:
            feature.close()
            await dispatcher.stop()

    asyncio.run(scenario())


def test_job_progress_sends_the_throttled_update_later():
    """A change inside the 1/s window (the first progress right after the start) was dropped until the next
    change; now the ticker sends it once the window has passed (trailing edge)."""
    background, native, clock, jobs = _background("android", update_interval=0.05)

    async def scenario():
        await background.job_started(_snap(state=JobState.STARTING))
        first = _snap(progress=Progress(total=80, completed=1))
        jobs.snap = first
        clock.now += 0.01
        assert not await background.job_progress(first)  # inside the window: held back, not dropped
        assert background.ticker_running
        await asyncio.sleep(0.2)  # ticks while the (fake) clock stands still: still held back
        assert _service_texts(native) == []
        clock.now += 0.1
        assert await _until(lambda: _service_texts(native) == ["Translating Book.epub: 1/80 chapters"])
        assert await _until(lambda: not background.ticker_running)  # visible app: nothing left to send
        clock.now += 0.01
        assert not await background.job_progress(first)  # an unchanged text is never queued
        assert not background.ticker_running

    asyncio.run(scenario())


@needs_flet
def test_glossary_question_notifies_unless_its_surface_is_on_screen(tmp_path):
    """Owner report: no glossary notification while the app was open. Now: hidden -> the notification
    (Accept / Review); visible elsewhere -> notification + snackbar; its chat / Book page on top -> nothing."""

    async def scenario():
        native = FakeNative()
        feature, app, _backend, notes, routes = _jobs_feature(tmp_path, native=native)
        question = {"id": "q1", "kind": "direct_text_glossary_approval", "data": {"path": "g.csv"}}
        snap = _snap(question=question)
        try:
            feature.background.on_lifecycle("hide")
            feature._on_question(snap, question)
            await _settle()
            note = _posted(native)[-1]
            assert (note[1][1], note[2]["channel_id"], note[2]["payload"]) == (
                "Glossary ready: review needed", "jobs.action", "glossarion://app/chat/7")
            assert note[2]["actions"] == [{"id": "accept", "title": "Accept"}, {"id": "open", "title": "Review"}]
            assert notes == []  # nobody sees a snackbar while hidden
            # visible on another screen: the notification and a snackbar whose Review opens the chat
            feature.background.on_lifecycle("resume")
            app.shell = types.SimpleNamespace(current_route="/library")
            feature._on_question(snap, question)
            await _settle()
            assert len(_posted(native)) == 2
            assert notes[-1][:2] == ("Glossary ready: review needed", "Review")
            notes[-1][2]()
            await _settle()
            assert routes == ["/chat/7"]
            # the owning chat is on screen (its approval card): nothing
            app.chat_view = types.SimpleNamespace(cid="7")
            for current in ("/chat/7", "/chat/7/m/0123456789ab", "/chat/3", "/"):
                # the chat view's own chat decides: "New chat" / chat switches leave the route behind
                app.shell.current_route = current
                feature._on_question(snap, question)
            await _settle()
            assert len(_posted(native)) == 2 and len(notes) == 1
            app.chat_view = types.SimpleNamespace(cid="3")  # the home shows another chat
            feature._on_question(snap, question)
            await _settle()
            assert len(_posted(native)) == 3
            app.shell.current_route = "/chat/7"  # a stale route: chat 3 is what the user sees
            feature._on_question(snap, question)
            await _settle()
            assert len(_posted(native)) == 4 and len(notes) == 3
            # a Library job's review gate is answered on its Book page
            library = _snap(spec=JobSpec("translate", "Book.epub", ("a",), origin={"type": "library", "bid": "b1"}),
                            question=dict(question, kind="glossary_approval"))
            app.shell.current_route = "/library/book/b1"
            feature._on_question(library, library.question)
            await _settle()
            assert len(_posted(native)) == 4
            app.shell.current_route = "/library"
            feature._on_question(library, library.question)
            await _settle()
            assert _posted(native)[-1][2]["payload"] == "glossarion://app/library/book/b1"
        finally:
            feature.close()

    asyncio.run(scenario())


@needs_flet
def test_async_batch_question_gets_its_own_notification(tmp_path):
    async def scenario():
        native = FakeNative()
        feature, app, _backend, notes, _routes = _jobs_feature(tmp_path, native=native)
        question = {"id": "q2", "kind": "async_batch_question", "data": {"title": "Split the batch?"}}
        snap = _snap(spec=JobSpec("async_batch", "Batch 3", (), origin={"type": "tools"}), question=question)
        try:
            feature.background.on_lifecycle("hide")
            feature._on_question(snap, question)
            await _settle()
            note = _posted(native)[-1]
            assert note[1][1] == "Answer needed: Split the batch?"
            assert note[2]["payload"] == "glossarion://app/tools/async" and not note[2].get("actions")
            assert not any("Glossary" in c[1][1] for c in _posted(native))
            # visible on Tools › Async batch: its own dialog asks, nothing else
            feature.background.on_lifecycle("resume")
            app.shell = types.SimpleNamespace(current_route="/tools/async")
            feature._on_question(snap, question)
            await _settle()
            assert len(_posted(native)) == 1 and notes == []
            app.shell.current_route = "/jobs"
            feature._on_question(snap, question)
            await _settle()
            assert len(_posted(native)) == 2 and notes[-1][:2] == ("Answer needed: Split the batch?", "Answer")
            # the FGS text names the question kind (C1: notification_text per kind)
            assert notification_text(snap).endswith("waiting for your answer")
        finally:
            feature.close()

    asyncio.run(scenario())


@needs_flet
def test_notification_accept_answers_through_the_chat_controller(tmp_path):
    from glossarion_mobile.services.dispatcher import UiDispatcher
    from glossarion_mobile.services.notifications import action_notification_id
    from glossarion_mobile.state.store import LoopGuard

    tap = JobNotifications.parse_event({"payload": "glossarion://app/chat/7", "action_id": "accept",
                                        "notification_id": 41611})
    assert (tap.action, tap.route, tap.notification_id) == ("accept", "/chat/7", 41611)
    assert JobNotifications.parse_event({"payload": "glossarion://app/chat/7", "notification_id": -1}).notification_id is None

    class Runs:
        """``ChatRuns.answer_glossary``: the approval card's ✓ Yes path."""

        def __init__(self):
            self.calls = []
            self.service = None

        def answer_glossary(self, cid, accepted):
            self.calls.append((cid, accepted))
            question = self.service.pending_question()
            return bool(question) and self.service.answer(question["id"], accepted)

    async def scenario():
        dispatcher = UiDispatcher(None, interval=0.01, guard=LoopGuard()).bind()
        dispatcher.start()
        runs = Runs()
        feature, app, backend, notes, routes = _jobs_feature(tmp_path, dispatcher=dispatcher, runs=runs)
        runs.service = feature.service
        service = feature.service
        accepted = ("Glossary accepted · translating", None, None)
        try:
            # a chat job: answered through the chat's run controller, then its chat opens
            asker = Asker()
            asker.go.set()
            backend.behavior = asker
            chat_job = service.submit(JobSpec("translate", "Book.epub", (epub(tmp_path),),
                                              origin={"type": "chat", "cid": "7"}))
            assert await _until(lambda: service.pending_question(chat_job) is not None)
            event = {"payload": "glossarion://app/chat/7", "action_id": "accept",
                     "notification_id": action_notification_id(chat_job), "launched_app": False}
            await feature._on_notification(event)
            assert runs.calls == [("7", True)] and notes.count(accepted) == 1 and routes == ["/chat/7"]
            assert await _until(lambda: asker.answers == [True])
            assert await asyncio.to_thread(service.wait_idle, TIMEOUT)
            # the same tap again: the job is gone, it only opens the chat
            await feature._on_notification(event)
            assert runs.calls == [("7", True)] and notes.count(accepted) == 1 and routes == ["/chat/7", "/chat/7"]

            # a Library job's review gate: answered through JobService
            library = Asker(kind="glossary_approval")
            library.go.set()
            backend.behavior = library
            lib_job = service.submit(JobSpec("translate", "Lib.epub", (epub(tmp_path, "Lib.epub"),),
                                             origin={"type": "library", "bid": "b1"}))
            assert await _until(lambda: service.pending_question(lib_job) is not None)
            stale = {"payload": "glossarion://app/library/book/b1", "action_id": "accept",
                     "notification_id": action_notification_id(lib_job) + 1}  # another job's notification
            await feature._on_notification(stale)
            assert service.pending_question(lib_job) is not None and routes[-1] == "/library/book/b1"
            await feature._on_notification(dict(stale, notification_id=action_notification_id(lib_job)))
            assert await _until(lambda: library.answers == [True])
            assert runs.calls == [("7", True)] and notes.count(accepted) == 2
            assert await asyncio.to_thread(service.wait_idle, TIMEOUT)

            # not a glossary question: Accept answers nothing, it only opens the route
            other = Asker(kind="async_batch_question", title="Split?")
            other.go.set()
            backend.behavior = other
            tools_job = service.submit(JobSpec("translate", "Batch", (epub(tmp_path, "B.epub"),),
                                               origin={"type": "tools"}))
            assert await _until(lambda: service.pending_question(tools_job) is not None)
            await feature._on_notification({"payload": "glossarion://app/tools/async", "action_id": "accept",
                                            "notification_id": action_notification_id(tools_job)})
            assert service.pending_question(tools_job) is not None and routes[-1] == "/tools/async"
            assert notes.count(accepted) == 2
            service.answer(service.pending_question(tools_job)["id"], False)
            assert await asyncio.to_thread(service.wait_idle, TIMEOUT)
        finally:
            feature.close()
            await dispatcher.stop()

    asyncio.run(scenario())


@needs_flet
def test_answered_question_cancels_its_action_notification(tmp_path):
    """Answered in the app (or given up), the "Glossary ready" notification goes away, also while the app is
    hidden and the UI pump is parked (``question_resolved`` is posted, not a channel value)."""
    from glossarion_mobile.services.dispatcher import UiDispatcher
    from glossarion_mobile.services.notifications import action_notification_id
    from glossarion_mobile.state.store import LoopGuard

    async def scenario():
        page = _HiddenPage()
        dispatcher = UiDispatcher(page, interval=0.01, guard=LoopGuard()).bind()
        dispatcher.start()
        native = FakeNative()
        feature, app, backend, notes, routes = _jobs_feature(tmp_path, dispatcher=dispatcher, native=native)
        service = feature.service
        cancel = ("cancel_notification", None, {})
        try:
            for hidden in (False, True):
                asker = Asker()
                asker.go.set()
                backend.behavior = asker
                job_id = service.submit(JobSpec("translate", "Book.epub", (epub(tmp_path),),
                                                origin={"type": "chat", "cid": "7"}))
                assert await _until(lambda: service.pending_question(job_id) is not None)
                assert await _until(lambda: _posted(native, "jobs.action"))
                cancel = ("cancel_notification", (action_notification_id(job_id),), {})
                if hidden:
                    page.hide()
                    feature.background.on_lifecycle("hide")
                assert cancel not in native.calls
                service.answer(service.pending_question(job_id)["id"], True)  # the approval card's ✓ Yes
                assert await _until(lambda: cancel in native.calls)
                assert page.app_visible is (not hidden)  # delivered while the pump was parked
                assert await asyncio.to_thread(service.wait_idle, TIMEOUT)
                page.show()
                feature.background.on_lifecycle("resume")
        finally:
            feature.close()
            await dispatcher.stop()

    asyncio.run(scenario())


@needs_flet
def test_launch_notification_is_routed_at_startup(tmp_path):
    """A tap that cold-started the app is only kept as the launch notification (never sent as an event):
    the app routes it once it is ready; a stale Accept (no job runs after a cold start) just opens it."""

    async def scenario():
        native = FakeNative()
        native.results["get_launch_notification"] = {"notification_id": 41205, "action_id": None,
                                                      "payload": "glossarion://app/job/abc123abc123",
                                                      "launched_app": True}
        feature, app, _backend, notes, routes = _jobs_feature(tmp_path / "a", native=native)
        try:
            assert await feature.route_launch_notification()
            assert routes == ["/job/abc123abc123"]
            assert not await feature.route_launch_notification()  # once per launch
            assert routes == ["/job/abc123abc123"]
        finally:
            feature.close()
        cold = FakeNative()
        cold.results["get_launch_notification"] = {"notification_id": 41611, "action_id": "accept",
                                                    "payload": "glossarion://app/chat/7", "launched_app": True}
        feature, app, _backend, notes, routes = _jobs_feature(tmp_path / "b", native=cold)
        try:
            assert await feature.route_launch_notification()
            assert routes == ["/chat/7"] and notes == []
        finally:
            feature.close()
        nothing = FakeNative()
        feature, app, _backend, notes, routes = _jobs_feature(tmp_path / "c", native=nothing)
        try:
            assert not await feature.route_launch_notification() and routes == []
        finally:
            feature.close()

    asyncio.run(scenario())


class _FlakyPermissions(FakePermissions):
    """``request("NOTIFICATION")`` answers from ``answers`` (an exception is raised, like a timed-out invoke)."""

    def __init__(self, answers) -> None:
        super().__init__()
        self.answers = list(answers)

    async def request(self, name):
        self.asked.append(name)
        if name != "NOTIFICATION":
            return "granted"
        answer = self.answers.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return answer


def test_prepare_for_run_marks_notification_asked_only_on_a_definite_status():
    """Owner report: no notifications at all. The 'asked' pref was set before the request, so a request that
    failed or was dismissed was never repeated. Now only a definite answer counts."""
    from glossarion_mobile.services.background import (
        NOTIFICATIONS_OFF_HINT,
        PREF_NOTIFICATION_ASKED,
        PREF_NOTIFICATIONS_OFF_HINT,
    )
    from glossarion_mobile.ui.screens.pages_feature import AccountsProfilesFeature

    notes, navigated = [], []
    background, native, clock, jobs = _background("android", notify=lambda *a: notes.append(a),
                                                   navigate_route=navigated.append)
    background.permissions = _FlakyPermissions([asyncio.TimeoutError(), "None", "denied"])
    spec = JobSpec("translate", "Book.epub", ("a",))

    async def scenario():
        await background.prepare_for_run(spec)  # the request timed out
        assert PREF_NOTIFICATION_ASKED not in background.prefs and notes == []
        await background.prepare_for_run(spec)  # the handler returned nothing
        assert PREF_NOTIFICATION_ASKED not in background.prefs and notes == []
        await background.prepare_for_run(spec)  # the user said no: remembered, and told once
        assert background.permissions.asked.count("NOTIFICATION") == 3
        assert background.prefs[PREF_NOTIFICATION_ASKED] is True and background.prefs[PREF_NOTIFICATIONS_OFF_HINT]
        assert [n[:2] for n in notes] == [(NOTIFICATIONS_OFF_HINT, "Turn on")]
        notes[0][2]()
        assert navigated == ["/settings/notifications"]
        await background.prepare_for_run(spec)
        assert background.permissions.asked.count("NOTIFICATION") == 3 and len(notes) == 1
        assert background.permissions.asked.count("IGNORE_BATTERY_OPTIMIZATIONS") == 1

        # asked on an earlier launch and still off (the real state): the hint once, only for long jobs
        later, later_native, _c, _j = _background("android", notify=lambda *a: notes.append(a),
                                                  prefs=FakePrefs({PREF_NOTIFICATION_ASKED: True}))
        later_native.results["get_platform_info"] = {"post_notifications_granted": False,
                                                     "notifications_enabled": False}
        await later.prepare_for_run(spec, long_job=False)
        assert len(notes) == 1 and later.permissions.asked == []
        await later.prepare_for_run(spec)
        assert len(notes) == 2 and later.permissions.asked == ["IGNORE_BATTERY_OPTIMIZATIONS"]
        # notifications on: no hint
        on, on_native, _c, _j = _background("android", notify=lambda *a: notes.append(a),
                                            prefs=FakePrefs({PREF_NOTIFICATION_ASKED: True}))
        on_native.results["get_platform_info"] = {"post_notifications_granted": True, "notifications_enabled": True}
        await on.prepare_for_run(spec)
        assert len(notes) == 2

        # Welcome step 4 / Settings › Notifications use the same request
        page_path, _n, _c, _j = _background("android")
        page_path.permissions = _FlakyPermissions([RuntimeError("no activity"), "granted"])
        assert (await AccountsProfilesFeature.request_notifications(page_path)).startswith("error RuntimeError")
        assert PREF_NOTIFICATION_ASKED not in page_path.prefs
        assert await AccountsProfilesFeature.request_notifications(page_path) == "granted"
        assert page_path.prefs[PREF_NOTIFICATION_ASKED] is True

    asyncio.run(scenario())


@needs_flet
def test_notifications_page_shows_real_permission_status():
    from glossarion_mobile.services.background import notification_state
    from glossarion_mobile.ui.chat.direct_text_rules import AUTO_ACCEPT_GLOSSARY_PREF
    from glossarion_mobile.ui.screens.notifications import NotificationsScreen
    from glossarion_mobile.ui.screens.pages_feature import AccountsProfilesFeature

    class PagePermissions(FakePermissions):
        def __init__(self):
            super().__init__()
            self.current = "permanentlyDenied"
            self.opened = 0

        async def status(self, name):
            return self.current

        async def open_app_settings(self):
            self.opened += 1
            return True

    background, native, _clock, _jobs = _background("android")
    background.permissions = PagePermissions()
    native.results["get_platform_info"] = {"post_notifications_granted": False, "notifications_enabled": False}
    said = []

    async def scenario():
        assert await background.notification_status() == {"granted": False, "enabled": False,
                                                          "status": "permanentlyDenied", "state": "blocked"}
        screen = NotificationsScreen(None, types.SimpleNamespace(say=said.append), background=background,
                                     request_notifications=lambda: AccountsProfilesFeature.request_notifications(
                                         background))
        screen.get_body()
        assert not screen.open_settings_button.visible  # until the real state is read (did_show)
        assert await screen.refresh_status() == "blocked"
        assert screen.notification_status.value.startswith("Notifications: Blocked in system settings")
        assert screen.open_settings_button.visible
        assert await screen.open_system_settings() and background.permissions.opened == 1 and said == []
        # the test notification posts on jobs.done
        assert await screen.send_test()
        note = _posted(native)[-1]
        assert note[2]["channel_id"] == "jobs.done" and screen.test_status.value == "Test notification sent."
        # turned on in the system settings: back in the app (lifecycle "resume") the page reads it again
        native.results["get_platform_info"] = {"post_notifications_granted": True, "notifications_enabled": True}
        background.permissions.current = "granted"
        screen.app_resumed()
        for _ in range(50):
            await asyncio.sleep(0)
            if screen.permission_state == "on":
                break
        assert screen.permission_state == "on" and not screen.open_settings_button.visible
        assert await screen.allow_notifications() == "granted"
        assert screen.permission_state == "on" and screen.notification_status.value == "Notifications: On"
        assert not screen.open_settings_button.visible
        # Glossary review: the global "Always accept generated glossaries" (Prefs, never config.json)
        assert screen.auto_accept.value is False
        screen.auto_accept.value = True
        screen._on_auto_accept(types.SimpleNamespace(control=screen.auto_accept))
        assert background.prefs[AUTO_ACCEPT_GLOSSARY_PREF] is True
        # the platform refuses a post: reported on the page and kept for diagnostics
        native.results["show_notification"] = False
        assert not await screen.send_test() and "did not show" in screen.test_status.value
        assert background.notifications.last_result["ok"] is False

    asyncio.run(scenario())
    assert notification_state({"granted": False, "status": "permanentlyDenied"}) == "blocked"
    assert notification_state({"granted": False, "status": "denied"}) == "off"
    assert notification_state({"granted": True, "enabled": False, "status": "granted"}) == "blocked"
    assert notification_state({"granted": True, "enabled": True, "status": ""}) == "on"
    desktop, _n, _c, _j = _background("desktop")
    assert asyncio.run(desktop.notification_status())["state"] == "unavailable"


def test_dismissed_fgs_notification_is_reposted_while_a_job_runs():
    """Owner: the job notification should stay like a VPN's. Android 14+ lets users swipe a foreground
    service's notification away; while the job runs it is posted again (at most once per second)."""
    background, native, clock, jobs = _background("android")

    async def scenario():
        starting = _snap(state=JobState.STARTING)
        jobs.snap = starting
        await background.on_transition(starting, JobState.QUEUED)
        running = _snap(progress=Progress(total=80, completed=3))
        jobs.snap = running
        clock.now += 2
        assert await background.job_progress(running)
        sent = len(_service_texts(native))
        await background.on_foreground_event({"type": "dismissed"})
        assert _service_texts(native)[sent:] == ["Translating Book.epub: 3/80 chapters"]  # the same text again
        await background.on_foreground_event({"type": "dismissed"})  # within a second: not at once ...
        assert len(_service_texts(native)) == sent + 1
        assert background._repost_wanted and background.ticker_running  # ... but the swipe is kept
        assert not await background.tick() and len(_service_texts(native)) == sent + 1  # still in the second
        clock.now += 1.0
        assert await background.tick()  # the ticker posts it once the second is over (the text is unchanged)
        assert _service_texts(native)[sent + 1:] == ["Translating Book.epub: 3/80 chapters"]
        assert not background._repost_wanted
        clock.now += 1.5
        await background.on_foreground_event({"type": "dismissed"})
        assert len(_service_texts(native)) == sent + 3 and jobs.stops == []  # a swipe never stops the job
        # the job ended and the service stopped: nothing comes back
        await background.on_transition(_snap(state=JobState.DONE, finished=200.0), JobState.RUNNING)
        assert "stop_job_service" in native.names() and not background.service_running
        clock.now += 5
        await background.on_foreground_event({"type": "dismissed"})
        assert len(_service_texts(native)) == sent + 3
        assert not await background.tick() and not background.ticker_running

    asyncio.run(scenario())
