"""Host tests for the mobile multi-instance core (design §6): ``services.instance_runtime`` and the JobService.

Run from src/mobile with the mobile venv:
    .venv/Scripts/python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_multi_instance_jobs.py

In-process, no network: the shared ``job_runner`` / ``stop_control`` (and the client they reach) run for real,
copied per instance; the job kind is a probe that records what its instance sees, and the owner is a fake. Every
test that installs the runtime uninstalls it again (``runtime`` fixture), so the tests after them see the process
untouched. The real translation of two books at once is ``test_multi_instance_e2e.py``.
"""

from __future__ import annotations

import builtins
import concurrent.futures
import io
import json
import logging
import os
import sys
import threading
import time
import types
from pathlib import Path

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = MOBILE_DIR / "app"
SRC_DIR = MOBILE_DIR.parent
for entry in (str(APP_DIR), str(SRC_DIR)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from glossarion_mobile.job_kinds import KindInfo  # noqa: E402
from glossarion_mobile.services import instance_runtime as ir  # noqa: E402
from glossarion_mobile.services.background import BackgroundExecution  # noqa: E402
from glossarion_mobile.services.jobs import (  # noqa: E402
    INSTANCE_PARAM,
    JobBackend,
    JobService,
    JobSnapshot,
    JobSpec,
    JobState,
    JobsView,
    Progress,
    notification_summary,
)
from glossarion_mobile.services.notifications import JobNotifications  # noqa: E402

TIMEOUT = 60.0


def wait_for(predicate, timeout=TIMEOUT) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return bool(predicate())


@pytest.fixture
def runtime():
    yield ir
    ir.uninstall()


@pytest.fixture(scope="module", autouse=True)
def _mobile_env():
    """The mobile gates the app sets before any backend import (no processes, no HTTP log)."""
    keys = {"GLOSSARION_MOBILE": "1", "GLOSSARION_NO_PROCESSES": "1", "GLOSSARION_HTTP_LOG": "0"}
    saved = {key: os.environ.get(key) for key in keys}
    os.environ.update(keys)
    yield
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _process_untouched(import_before) -> None:
    """Nothing of the runtime is left. ``import_before``: ``builtins.__import__`` before the test (another library's
    hook may stand there: PySide6's shiboken installs its own once something imported it)."""
    assert type(sys) is types.ModuleType
    assert type(os.environ) is os._Environ
    assert builtins.__import__ is import_before and builtins.__import__ is not ir._import_hook
    assert "_gl_instance" not in vars(threading.Thread)
    assert threading.Thread.__init__.__module__ == "threading"
    assert concurrent.futures.ThreadPoolExecutor.submit.__module__ == "concurrent.futures.thread"
    assert logging.Handler.handle.__module__ == "logging"


class FakeStore:
    def __init__(self, data=None) -> None:
        self.data = dict(data or {})
        self.calls = []

    def flush(self):
        return False

    def snapshot(self):
        return json.loads(json.dumps(self.data))

    def set_job_running(self, running):
        self.calls.append(bool(running))

    def get(self, key, default=None):
        return self.data.get(key, default)


class Owner:
    def __init__(self, config, host) -> None:
        self.config = config
        self.host = host
        self.stop_requested = False
        self.graceful_stop_var = False  # every Stop of these tests is immediate
        self.wait_for_chunks_var = False


class Backend(JobBackend):
    """The real job scope, run reset and Stop protocol (each instance's own copies); a fake owner."""

    def make_owner(self, config, *, host):
        return Owner(config, host)


class Probe:
    """A job kind that records what its instance sees, then waits for its release or a Stop."""

    def __init__(self) -> None:
        self.seen: dict = {}
        self.entered: dict = {}
        self.release: dict = {}

    def gate(self, job_id: str) -> threading.Event:
        return self.release.setdefault(job_id, threading.Event())

    def run(self, ctx):
        import job_runner
        import stop_control

        job_id = ctx.job_id
        record = self.seen.setdefault(job_id, {})
        os.environ["PROBE_JOB"] = job_id
        sys.argv = ["probe", job_id]
        print(f"probe line of {job_id}")
        record.update(instance=os.environ.get("GLOSSARION_INSTANCE"), lock=id(job_runner.JOB_LOCK),
                      stop_control=id(stop_control), stop_file=os.environ.get("GLOSSARY_STOP_FILE"),
                      config_keys=[row.get("api_key") for row in ctx.config.get("multi_api_keys") or []])
        inner: dict = {}
        thread = threading.Thread(target=lambda: inner.update(env=os.environ.get("PROBE_JOB")))
        thread.start()
        thread.join()
        with concurrent.futures.ThreadPoolExecutor(1) as pool:
            inner["pool"] = pool.submit(lambda: os.environ.get("PROBE_JOB")).result()
        record["inner"] = inner
        self.entered.setdefault(job_id, threading.Event()).set()
        cancelled = []
        while not self.gate(job_id).wait(0.02):
            cancelled.append(os.environ.get("TRANSLATION_CANCELLED"))
            if ctx.stop_requested():
                break
        record.update(env_at_end=os.environ.get("PROBE_JOB"), argv_at_end=list(sys.argv),
                      saw_cancel="1" in cancelled, stopped=ctx.stop_requested())
        return {"ok": None if ctx.stop_requested() else True, "outputs": []}


def _kinds(probe: Probe):
    info = KindInfo(kind="translate", verb="Probing", icon="SCIENCE", stop_kind="translation", run=probe.run)
    return lambda kind: info


def _service(tmp_path, probe, store=None, jobs_dir=None):
    return JobService(jobs_dir=jobs_dir or tmp_path / "jobs", logs_dir=tmp_path / "logs", data_dir=tmp_path,
                      config_store=store if store is not None else FakeStore(), backend=Backend(),
                      kinds=_kinds(probe), watcher_interval=0.05)


def _book(tmp_path, name) -> str:
    path = tmp_path / "in" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"PK fake epub")
    return str(path)


KEYS = {"use_multi_api_keys": True, "multi_api_keys": [
    {"api_key": "sk-one", "model": "gpt-4o-mini", "enabled": True},
    {"api_key": "sk-two", "model": "gpt-4o-mini", "enabled": True, "instance": 2},
]}


# ==========================================================================
# One instance: nothing changes
# ==========================================================================


def test_single_instance_jobs_never_install_the_runtime(tmp_path):
    ir.uninstall()  # whatever ran before in this process
    import_before = builtins.__import__
    probe = Probe()
    store = FakeStore(KEYS)
    service = _service(tmp_path, probe, store)
    plain = service.submit(JobSpec("translate", "Plain", (_book(tmp_path, "a.epub"),)))
    explicit = service.submit_job(JobSpec("translate", "Explicit", (_book(tmp_path, "b.epub"),)), instance=1).job_id
    for job_id in (plain, explicit):
        assert probe.entered.setdefault(job_id, threading.Event()).wait(TIMEOUT)
        probe.gate(job_id).set()
    assert service.wait_idle(TIMEOUT)
    assert service.snapshot(plain).state is JobState.DONE and service.snapshot(explicit).state is JobState.DONE
    assert not ir.installed() and not ir.runtimes()
    _process_untouched(import_before)
    # instance 1 on the gl-job worker; a plain job keeps the whole pool, an explicit instance-1 job its own keys
    assert probe.seen[plain]["instance"] is None and probe.seen[plain]["config_keys"] == ["sk-one", "sk-two"]
    assert probe.seen[explicit]["config_keys"] == ["sk-one"]
    assert service.snapshot(explicit).instance == 1 and service.snapshot(explicit).spec.params[INSTANCE_PARAM] == 1
    assert INSTANCE_PARAM not in service.snapshot(plain).spec.params
    assert store.calls == [True, False, True, False]  # set_job_running exactly as before, job by job
    service.close()


# ==========================================================================
# The runtime itself
# ==========================================================================


def test_runtime_routes_environment_streams_argv_threads_and_logging(runtime):
    runtime.uninstall()
    import_before = builtins.__import__
    rt = runtime.ensure(2)
    rt.prepare(env={**dict(runtime.base_environ()), "SHARED_AT_START": "yes"}, argv=["base-argv"])
    base_argv = list(sys.argv)
    out = io.StringIO()
    seen: dict = {}
    base_records: list = []
    instance_records: list = []

    class Collect(logging.Handler):
        def __init__(self, sink):
            super().__init__(logging.DEBUG)
            self.sink = sink

        def emit(self, record):
            self.sink.append(record.getMessage())

    probe_logger = logging.getLogger("instance_runtime_probe")
    probe_logger.setLevel(logging.DEBUG)
    probe_logger.propagate = False
    base_handler = Collect(base_records)
    probe_logger.addHandler(base_handler)

    def in_instance():
        os.environ["ONLY_TWO"] = "2"
        sys.argv = ["two"]
        sys.stdout = out
        print("printed by instance 2")
        handler = Collect(instance_records)
        probe_logger.addHandler(handler)
        probe_logger.info("record of instance 2")
        seen.update(instance=os.getenv("GLOSSARION_INSTANCE"), start=os.environ.get("SHARED_AT_START"),
                    stop_file=os.environ.get("GLOSSARY_STOP_FILE"))
        nested: dict = {}
        thread = threading.Thread(target=lambda: nested.update(env=os.environ.get("ONLY_TWO"), argv=list(sys.argv)))
        thread.start()
        thread.join()
        seen["nested"] = nested
        seen["pool"] = concurrent.futures.ThreadPoolExecutor(1)  # its worker thread is instance 2's
        seen["pool"].submit(lambda: None).result()
        seen["handler"] = handler

    thread = rt.thread(in_instance, name="probe-i2")
    thread.start()
    thread.join()
    probe_logger.info("record of instance 1")
    # a shared pool runs each item as the instance that submitted it
    assert seen["pool"].submit(lambda: os.environ.get("ONLY_TWO")).result() is None
    seen["pool"].shutdown()
    print("printed by instance 1")

    assert seen["instance"] == "2" and seen["start"] == "yes" and seen["stop_file"].endswith("_instance2.flag")
    assert seen["nested"] == {"env": "2", "argv": ["two"]}
    assert "ONLY_TWO" not in os.environ and os.environ.get("GLOSSARION_INSTANCE") is None
    assert sys.argv == base_argv and rt.argv == ["two"]
    assert out.getvalue() == "printed by instance 2\n" and sys.stdout is not out
    assert instance_records == ["record of instance 2"] and base_records == ["record of instance 1"]
    probe_logger.removeHandler(base_handler)
    probe_logger.removeHandler(seen["handler"])
    # with os._Environ's rules
    with pytest.raises(TypeError):
        with runtime.activated(2):
            os.environ["NOT_A_STR"] = 5
    runtime.uninstall()
    _process_untouched(import_before)


def test_module_copies_are_per_instance_and_share_code(runtime):
    import job_runner
    import stop_control

    rt2, rt3 = runtime.ensure(2), runtime.ensure(3)
    for rt in (rt2, rt3):
        rt.prepare(env=dict(runtime.base_environ()))
        rt.warm(("job_runner", "stop_control"))
    jr2, jr3, sc2 = rt2.modules["job_runner"], rt3.modules["job_runner"], rt2.modules["stop_control"]
    assert jr2 is not job_runner and jr2 is not jr3 and sys.modules["job_runner"] is job_runner
    assert len({id(job_runner.JOB_LOCK), id(jr2.JOB_LOCK), id(jr3.JOB_LOCK)}) == 3
    assert jr2.__dict__[runtime.MARKER] == 2 and runtime.MARKER not in vars(job_runner)
    # one code object for the base module and every instance: only the globals differ
    assert jr2.restore_mapping.__code__ is jr3.restore_mapping.__code__ is job_runner.restore_mapping.__code__
    # "from instance_plan import current_instance" bound the instance's own copy
    assert sc2.current_instance.__globals__[runtime.MARKER] == 2 and stop_control.current_instance.__globals__ is not \
        sc2.current_instance.__globals__
    # sys.modules as a copy sees it: the instance's own copies, None until the instance loads one
    assert jr2.sys.modules.get("job_runner") is jr2 and jr2.sys.modules.get("large_env") is None
    with runtime.activated(2):
        import large_env as large2
    assert jr2.sys.modules.get("large_env") is large2 is runtime.loaded_module("large_env", 2)
    assert large2 is not sys.modules.get("large_env") and jr2.sys.modules.get("json") is sys.modules["json"]
    # the standard modules with process-wide registrations have the instance's own stand-ins
    builtins2 = rt2.shadow("builtins")
    builtins2._GL_INSTANCE_FLAG = True
    assert not hasattr(builtins, "_GL_INSTANCE_FLAG") and rt2.builtins_dict["_GL_INSTANCE_FLAG"] is True
    json2 = rt2.shadow("json")
    json2.load = lambda *a, **k: "patched"
    assert json.load is not json2.load and json2.dumps is json.dumps
    import atexit

    before = atexit._ncallbacks()
    ran = []
    rt2.shadow("atexit").register(ran.append, "first")
    rt2.shadow("atexit").register(ran.append, "second")
    # one process exit hook runs the instance's own registrations, latest first; none runs twice
    assert atexit._ncallbacks() == before + 1 and len(rt2.atexit_callbacks) == 2
    rt2.run_atexit()
    assert ran == ["second", "first"] and not rt2.atexit_callbacks
    # a module that is not the backend's is shared, one that does not exist fails like an import
    with runtime.activated(2):
        import json as json_in_base_code
        with pytest.raises(ModuleNotFoundError):
            rt2.module("glossarion_mobile.job_kinds.no_such_adapter")
    assert json_in_base_code is json and not runtime.copyable("library_core") and runtime.copyable("TransateKRtoEN")


def test_another_import_hook_over_ours_keeps_both_working(runtime):
    """PySide6's shiboken installs its own ``builtins.__import__`` the first time it is imported (on the host a job
    imports it through dpi_setup); racing the install it could take the place of ours. Ours goes back in front at the
    next thread / job start and keeps the other one as its fallback; one that chains to ours never loops."""
    runtime.uninstall()
    pristine = builtins.__import__
    try:
        rt = runtime.ensure(2)
        rt.prepare(env=dict(runtime.base_environ()))
        stale: list = []

        def stale_hook(name, *args, **kwargs):  # took the import function before ours was installed
            stale.append(name)
            return pristine(name, *args, **kwargs)

        builtins.__import__ = stale_hook
        thread = threading.Thread(target=lambda: None)  # any new thread puts ours back in front
        thread.start()
        thread.join()
        assert builtins.__import__ is ir._import_hook
        with runtime.activated(2):
            import job_runner as copied
        assert copied is rt.modules["job_runner"]
        import colorsys  # noqa: F401  (not routed: through the other hook)
        assert "colorsys" in stale
        chained: list = []
        ours = builtins.__import__

        def chained_hook(name, *args, **kwargs):  # installed after ours: it took ours as its original
            chained.append(name)
            return ours(name, *args, **kwargs)

        builtins.__import__ = chained_hook
        ir._keep_import_hook()
        import wave  # noqa: F401
        assert builtins.__import__ is ir._import_hook and "wave" in chained
        with runtime.activated(2):
            import stop_control as copied_stop
        assert copied_stop is rt.module("stop_control")
    finally:
        runtime.uninstall()
        builtins.__import__ = pristine


# ==========================================================================
# JobService: two instances at once
# ==========================================================================


def test_two_instances_run_at_once_with_their_own_state(tmp_path, runtime):
    import unified_api_client  # noqa: F401  (the base client: the Stop protocol of each instance reaches its own)

    probe = Probe()
    store = FakeStore(KEYS)
    service = _service(tmp_path, probe, store)
    one = service.submit_job(JobSpec("translate", "Book one", (_book(tmp_path, "one.epub"),)), instance=1).job_id
    two = service.submit_job(JobSpec("translate", "Book two", (_book(tmp_path, "two.epub"),)), instance=2).job_id
    assert probe.entered.setdefault(one, threading.Event()).wait(TIMEOUT)
    assert probe.entered.setdefault(two, threading.Event()).wait(TIMEOUT)

    view = service.view()
    assert [s.id for s in view.running] == [one, two] and view.running_count == 2 and view.active.id == one
    assert view.instance_job(2).id == two and service.instance_busy(2) and service.busy and service.any_busy
    assert [s.id for s in service.running_snapshots()] == [one, two]
    assert service.snapshot().id == one  # the old no-argument form: instance 1's job
    state = json.loads(Path(service.active_state_path).read_text(encoding="utf-8"))
    assert state["active"]["id"] == one and state["instances"]["2"]["active"]["id"] == two

    # Stop instance 2 only: its own stop_control sets its own flags
    assert service.request_stop(two, force=True) == "force"
    assert wait_for(lambda: service.snapshot(two).is_terminal)
    assert service.snapshot(two).state is JobState.CANCELLED
    time.sleep(0.3)
    assert service.snapshot(one).state is JobState.RUNNING and not probe.seen[one].get("stopped")
    uac2 = runtime.loaded_module("unified_api_client", 2)
    assert uac2 is not None and uac2.UnifiedClient._global_cancelled is True
    assert getattr(sys.modules["unified_api_client"].UnifiedClient, "_global_cancelled", False) is False
    assert store.calls[-1] is True  # still running: the banner stays
    probe.gate(one).set()
    assert service.wait_idle(TIMEOUT)
    assert service.snapshot(one).state is JobState.DONE
    assert store.calls.count(False) == 1 and store.calls[-1] is False  # off once, when the last job ended

    first, second = probe.seen[one], probe.seen[two]
    assert first["instance"] is None and second["instance"] == "2"
    assert first["lock"] != second["lock"] and first["stop_control"] != second["stop_control"]
    assert first["config_keys"] == ["sk-one"] and second["config_keys"] == ["sk-two"]
    assert first["env_at_end"] == one and second["env_at_end"] == two  # neither overwrote the other's
    assert first["argv_at_end"] == ["probe", one] and second["argv_at_end"] == ["probe", two]
    assert first["inner"] == {"env": one, "pool": one} and second["inner"] == {"env": two, "pool": two}
    assert second["stop_file"].endswith("_instance2.flag") and first["stop_file"] != second["stop_file"]
    assert second["saw_cancel"] and not first["saw_cancel"]
    log_one, log_two = "\n".join(service.read_log_tail(one)), "\n".join(service.read_log_tail(two))
    assert f"probe line of {one}" in log_one and two not in log_one
    assert f"probe line of {two}" in log_two and one not in log_two
    assert "PROBE_JOB" not in os.environ  # instance 1's job scope restored the process environment
    assert not Path(service.active_state_path).exists()
    service.close()


def test_instances_queue_separately_and_recover_on_their_instance(tmp_path, runtime):
    probe = Probe()
    service = _service(tmp_path, probe, FakeStore(KEYS))
    first = service.submit_job(JobSpec("translate", "A", (_book(tmp_path, "a.epub"),)), instance=2).job_id
    queued = service.submit_job(JobSpec("translate", "B", (_book(tmp_path, "b.epub"),)), instance=2).job_id
    assert probe.entered.setdefault(first, threading.Event()).wait(TIMEOUT)
    view = service.view()
    assert [s.id for s in view.running] == [first] and [s.id for s in view.queue] == [queued]
    assert view.active.id == first and service.snapshot() is None  # instance 1 is idle
    saved = Path(service.active_state_path).read_text(encoding="utf-8")
    # the app is killed now: the next launch finds both, Interrupted, and Resume keeps instance 2
    relaunch_dir = tmp_path / "relaunch"
    relaunch_dir.mkdir()
    (relaunch_dir / "active.state").write_text(saved, encoding="utf-8")
    relaunched = _service(tmp_path, Probe(), FakeStore(KEYS), jobs_dir=relaunch_dir)
    assert {s.id for s in relaunched.interrupted} == {first, queued}
    assert all(s.instance == 2 for s in relaunched.interrupted)
    relaunched.close()
    probe.gate(first).set()
    assert probe.entered.setdefault(queued, threading.Event()).wait(TIMEOUT)
    probe.gate(queued).set()
    assert service.wait_idle(TIMEOUT)
    service.close()


def test_an_instance_that_cannot_be_isolated_fails_its_job(tmp_path, runtime, monkeypatch):
    probe = Probe()
    service = _service(tmp_path, probe, FakeStore(KEYS))

    def refuse(self, names):
        raise ir.InstanceIsolationError("stop_control: cannot be copied")

    monkeypatch.setattr(ir.InstanceRuntime, "warm", refuse)
    job_id = service.submit_job(JobSpec("translate", "A", (_book(tmp_path, "a.epub"),)), instance=2).job_id
    assert service.wait_idle(TIMEOUT)
    snap = service.snapshot(job_id)
    assert snap.state is JobState.FAILED and "could not be isolated" in snap.error and job_id not in probe.seen
    service.close()


def test_an_instance_without_a_key_of_its_own_fails_its_job(tmp_path, runtime):
    probe = Probe()
    service = _service(tmp_path, probe, FakeStore(KEYS))
    job_id = service.submit_job(JobSpec("translate", "A", (_book(tmp_path, "a.epub"),)), instance=3).job_id
    assert service.wait_idle(TIMEOUT)
    snap = service.snapshot(job_id)
    assert snap.state is JobState.FAILED and "Instance 3 has no enabled Translation API key" in snap.error
    service.close()


# ==========================================================================
# The notification of several instances
# ==========================================================================


def _snap(job_id, instance, title, completed=1, state=JobState.RUNNING):
    return JobSnapshot(id=job_id, spec=JobSpec("translate", title, ("x",), params={INSTANCE_PARAM: instance}),
                       state=state, created=1.0, started=1.0, progress=Progress(total=10, completed=completed))


class Native:
    def __init__(self) -> None:
        self.calls = []

    async def call(self, method, *args, default=None, **kwargs):
        self.calls.append((method, args, kwargs))
        return {"start_job_service": True, "show_notification": True}.get(method, default)

    async def init_notifications(self):
        return True

    async def cancel_notification(self, notification_id):
        self.calls.append(("cancel_notification", (notification_id,), {}))


class TwoJobs:
    def __init__(self, running) -> None:
        self.running = list(running)
        self.stopped = []

    def snapshot(self, job_id=None):
        return next((s for s in self.running if s.instance == 1), None)

    def running_snapshots(self):
        return list(self.running)

    def stream_tail(self, job_id=None):
        return {"a1": "the knight raised her sword", "b2": "<p>dawn broke</p>"}.get(job_id, "")

    def stream_thinking(self, job_id=None):
        return job_id == "b2"

    def device_row(self, job_id=None, threads=False):
        return ""

    def request_stop_all(self, **kwargs):
        self.stopped.append(kwargs)
        return {}

    def view(self):
        return JobsView(running=tuple(self.running))


def test_one_notification_summarises_every_running_instance():
    import asyncio

    one, two = _snap("a1", 1, "Book one"), _snap("b2", 2, "Book two", completed=3)
    jobs = TwoJobs([one, two])
    native = Native()
    navigated = []
    background = BackgroundExecution(native, platform="android", jobs=jobs, notifications=JobNotifications(
        native, platform="android"), navigate_route=navigated.append, clock=lambda: 1000.0)

    async def scenario():
        await background.job_started(one)
        await background.job_progress(two, force=True)
        update = [c for c in native.calls if c[0] == "update_job_service"][-1]
        assert update[2]["title"] == "2 jobs running"
        lines = update[2]["text"].split("\n")
        assert lines[0].startswith("[1] Translating • 1/10") and lines[0].endswith("Book one")
        assert "    ✍️ the knight raised her sword" in lines
        assert any(line.startswith("[2] Translating • 3/10") for line in lines)
        assert "    🧠 dawn broke" in lines
        await background.on_foreground_event({"type": "button", "button_id": "stop"})
        assert jobs.stopped == [{"reason": "Stop pressed in the notification"}]
        await background.on_foreground_event({"type": "tap"})
        assert navigated == ["/jobs"]
        # instance 1 ends while instance 2 runs: the service stays, the notification shows instance 2
        jobs.running = [two]
        await background.job_finished(_snap("a1", 1, "Book one", completed=10, state=JobState.DONE))
        assert "stop_job_service" not in [c[0] for c in native.calls] and background.service_running
        last = [c for c in native.calls if c[0] == "update_job_service"][-1]
        assert last[2]["title"] == "Translating • 3/10"
        jobs.running = []
        await background.job_finished(_snap("b2", 2, "Book two", completed=10, state=JobState.DONE))
        assert "stop_job_service" in [c[0] for c in native.calls] and not background.service_running

    asyncio.run(scenario())


def test_notification_summary_keeps_one_job_as_before():
    one = _snap("a1", 1, "Book one")
    headline, lines = notification_summary([one], lambda snap: ["✍️ text"])
    assert (headline, lines) == ("Translating • 1/10", ["Book one", "✍️ text"])
