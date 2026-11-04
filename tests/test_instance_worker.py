"""Multi-instance translation: the instance process (src/instance_worker.py), GUI-free.

* the command line / environment / stop-file protocol the main window and the process share;
* ``restrict_keys``: the process's API keys are narrowed to one instance (``GLOSSARION_KEY_INSTANCE``)
  with the client's pools detached and put back by the undo; the main window's key variables are
  never inherited;
* ``request_stop``: the desktop Stop sequence on the instance's owner;
* ``run_job``: the mobile translate-job path (``_prepare_translation_run`` + ``_translation_worker``)
  on a HeadlessOwner built with the process host, the watchdog folder of the main window, the
  Library registry left to the main window, the QA-scan marker, exit codes 0 / 1 / 2;
* ``main``: bad job files fail cleanly; the snapshot replaces config.json for the process.

No real backend is imported (stand-in ``unified_api_client`` / ``headless_owner`` modules). Run:
    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests/test_instance_worker.py
"""

import ast
import json
import os
import sys
import threading
import time
import types
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:  # CI has no tests/conftest.py
    sys.path.insert(0, str(SRC_DIR))

import instance_plan  # noqa: E402
import instance_worker as iw  # noqa: E402


@pytest.fixture
def clean_env():
    """os.environ as it was before the test (the stop protocol writes process-wide flags)."""
    saved = dict(os.environ)
    try:
        yield
    finally:
        for key in [k for k in os.environ if k not in saved]:
            os.environ.pop(key, None)
        for key, value in saved.items():
            if os.environ.get(key) != value:
                os.environ[key] = value


def _fake_client_module(monkeypatch):
    """A stand-in unified_api_client whose two key setters record the lists they keep: like the real
    client they narrow every list to the key scope (``instance_plan.scope_keys``) and hold the pool
    as a class attribute (which ``job_runner.isolated_key_pools`` detaches and puts back)."""
    calls = []

    class UnifiedClient:
        _api_key_pool = None
        _in_memory_multi_keys = None

        @classmethod
        def set_in_memory_multi_keys(cls, keys_list, force_rotation=True, rotation_frequency=1):
            keys_list = instance_plan.scope_keys(keys_list)
            calls.append(("memory", [k.get("api_key") for k in keys_list]))
            cls._in_memory_multi_keys = keys_list
            cls.setup_multi_key_pool(keys_list, force_rotation=force_rotation)
            return True

        @classmethod
        def setup_multi_key_pool(cls, keys_list, force_rotation=True, rotation_frequency=1):
            keys_list = instance_plan.scope_keys(keys_list)
            calls.append(("pool", [k.get("api_key") for k in keys_list]))
            cls._api_key_pool = [k.get("api_key") for k in keys_list]

    module = types.ModuleType("unified_api_client")
    module.UnifiedClient = UnifiedClient
    monkeypatch.setitem(sys.modules, "unified_api_client", module)
    return UnifiedClient, calls


KEYS = [
    {"api_key": "one-a", "enabled": True},
    {"api_key": "two-a", "enabled": True, "instance": 2},
    {"api_key": "one-b", "enabled": True, "instance": "1"},
    {"api_key": "two-b", "enabled": False, "instance": 2.0},
]


# --------------------------------------------------------------------------
# protocol shared with instance_runs
# --------------------------------------------------------------------------

def test_command_line_frozen_and_source(monkeypatch):
    monkeypatch.delattr(sys, "frozen", raising=False)
    cmd = iw.command_line("C:/jobs/job.json")
    assert cmd[0] == sys.executable and cmd[-2:] == [iw.FLAG, "C:/jobs/job.json"]
    assert Path(cmd[1]).name == "translator_gui.py" and Path(cmd[1]).parent == SRC_DIR
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    assert iw.command_line("job.json") == [sys.executable, "--run-translation-instance", "job.json"]


def test_child_environment_drops_the_parents_per_process_values():
    base = {name: "parent" for name in iw.PER_PROCESS_ENV}
    base.update({"PATH": "p", "MODEL": "m", "GLOSSARION_INSTANCE": "1", "GLOSSARION_KEY_INSTANCE": "1"})
    env = iw.child_environment(base, "3")
    assert env["GLOSSARION_INSTANCE"] == "3" and env["GLOSSARION_KEY_INSTANCE"] == "3"
    assert env["PATH"] == "p" and env["MODEL"] == "m"
    for name in iw.PER_PROCESS_ENV:
        if name not in (instance_plan.ENV_INSTANCE, instance_plan.ENV_KEY_INSTANCE):
            assert name not in env, name
    assert env["PYTHONIOENCODING"] == "utf-8" and env["PYTHONUNBUFFERED"] == "1"
    assert base["GLOSSARY_STOP_FILE"] == "parent"  # the caller's mapping is untouched


def test_child_environment_never_inherits_the_main_windows_keys():
    """The main GUI key (instance 1's) and every key list / pool switch of the main window stay
    behind; the instance exports its own from its snapshot. Windows reports names upper-cased."""
    base = {name: "parent-key" for name in instance_plan.KEY_ENV}
    base["OPENAI_OR_GEMINI_API_KEY"] = "parent-key"  # as os.environ lists it on Windows
    base.update({"PATH": "p", "SEND_INTERVAL_SECONDS": "2"})
    env = iw.child_environment(base, 2)
    assert "parent-key" not in env.values(), sorted(k for k, v in env.items() if v == "parent-key")
    assert env["PATH"] == "p" and env["SEND_INTERVAL_SECONDS"] == "2"
    assert iw.inheritable("MODEL") and not iw.inheritable("fallback_keys".upper())
    assert not iw.inheritable("api_key") and not iw.inheritable("GLOSSARION_KEY_INSTANCE")


def test_stop_requests_round_trip(tmp_path):
    stop_file = tmp_path / "sub" / "stop.flag"
    assert iw.read_stop_request(str(stop_file)) is None
    for mode in (iw.STOP_GRACEFUL, iw.STOP_IMMEDIATE, iw.STOP_FORCE):
        iw.write_stop_request(str(stop_file), mode)
        assert iw.read_stop_request(str(stop_file)) == mode
    stop_file.write_text("whatever", encoding="utf-8")
    assert iw.read_stop_request(str(stop_file)) is None
    assert sorted(p.name for p in stop_file.parent.iterdir()) == ["stop.flag"]  # no temp file left


# --------------------------------------------------------------------------
# restrict_keys
# --------------------------------------------------------------------------

def test_restrict_keys_narrows_every_pool_and_restores(monkeypatch, clean_env):
    client, calls = _fake_client_module(monkeypatch)
    monkeypatch.delenv(instance_plan.ENV_KEY_INSTANCE, raising=False)
    monkeypatch.delenv(instance_plan.ENV_INSTANCE, raising=False)
    client.set_in_memory_multi_keys(KEYS)  # the pools before the run: every key
    before_pool, before_memory = client._api_key_pool, client._in_memory_multi_keys
    calls.clear()
    restore = iw.restrict_keys(2, {"use_multi_api_keys": True, "multi_api_keys": KEYS})
    assert instance_plan.key_scope() == 2 and os.environ[instance_plan.ENV_KEY_INSTANCE] == "2"
    assert calls == [("memory", ["two-a", "two-b"]), ("pool", ["two-a", "two-b"])]
    assert client._api_key_pool == ["two-a", "two-b"]
    assert before_pool == ["one-a", "two-a", "one-b", "two-b"]  # never narrowed in place
    client.setup_multi_key_pool(KEYS)
    assert client._api_key_pool == ["two-a", "two-b"]
    restore()
    assert instance_plan.ENV_KEY_INSTANCE not in os.environ and instance_plan.key_scope() is None
    assert client._api_key_pool is before_pool and client._in_memory_multi_keys is before_memory
    calls.clear()
    client.setup_multi_key_pool(KEYS)
    assert calls == [("pool", ["one-a", "two-a", "one-b", "two-b"])]


def test_restrict_keys_instance_one_and_objects(monkeypatch, clean_env):
    client, calls = _fake_client_module(monkeypatch)
    monkeypatch.setenv(instance_plan.ENV_KEY_INSTANCE, "7")  # put back by the undo
    restore = iw.restrict_keys(1)
    try:
        assert calls == []  # without a config nothing is applied
        client.setup_multi_key_pool(KEYS)
        assert calls == [("pool", ["one-a", "one-b"])]
        entry = types.SimpleNamespace(instance=2, get=lambda name, default=None: "obj")
        calls.clear()
        client.setup_multi_key_pool([entry])
        assert calls == [("pool", [])]  # APIKeyEntry-like objects are read through .instance
    finally:
        restore()
    assert os.environ[instance_plan.ENV_KEY_INSTANCE] == "7"


# --------------------------------------------------------------------------
# request_stop
# --------------------------------------------------------------------------

class _Owner:
    def __init__(self):
        self.stop_requested = False
        self.graceful_stop_active = None
        self._last_stop_was_graceful = None
        self.wait_for_chunks_var = False
        self.resets = []

    def _reset_api_watchdog_progress(self, *, clear_stale_external_files=True):
        self.resets.append(clear_stale_external_files)


@pytest.mark.parametrize("mode, graceful, forced", [
    (iw.STOP_GRACEFUL, True, False), (iw.STOP_IMMEDIATE, False, False), (iw.STOP_FORCE, False, True)])
def test_request_stop_runs_the_desktop_stop_sequence(monkeypatch, clean_env, mode, graceful, forced):
    import stop_control

    seen = []
    monkeypatch.setattr(stop_control, "apply_force_stop_flags", lambda: seen.append("force_flags"))

    def fake_request_stop(**kwargs):
        seen.append(("request_stop", kwargs["graceful"], kwargs["wait_for_chunks"]))
        kwargs["set_stop_requested"]()

    monkeypatch.setattr(stop_control, "request_stop", fake_request_stop)
    monkeypatch.setattr(stop_control, "announce_stop", lambda g, log: seen.append(("announce", g)))
    owner = _Owner()
    lines = []
    iw.request_stop(owner, mode, lines.append)
    assert owner.stop_requested is True and owner.graceful_stop_active is graceful
    assert owner._last_stop_was_graceful is graceful
    assert seen[-2:] == [("request_stop", graceful, False), ("announce", graceful)]
    assert ("force_flags" in seen) is forced
    assert owner.resets == ([] if graceful else [True])  # a full stop resets the watchdog


def test_stop_watcher_escalates_once_per_level(tmp_path):
    stop_file = tmp_path / "stop.flag"
    modes = []
    watcher = iw._StopWatcher(str(stop_file), None, modes.append, lambda text: None, interval=0.02)
    watcher.start()
    try:
        for mode in (iw.STOP_IMMEDIATE, iw.STOP_GRACEFUL, iw.STOP_IMMEDIATE, iw.STOP_FORCE):
            iw.write_stop_request(str(stop_file), mode)
            time.sleep(0.15)
    finally:
        watcher.stop()
        watcher.join(2)
    assert modes == [iw.STOP_IMMEDIATE, iw.STOP_FORCE] and watcher.mode == iw.STOP_FORCE


# --------------------------------------------------------------------------
# run_job
# --------------------------------------------------------------------------

class _Host:
    def __init__(self):
        self.lines = []

    def log(self, text, **_kw):
        self.lines.append(str(text))


@pytest.fixture
def job_env(tmp_path, monkeypatch, clean_env):
    """run_job with stand-in backend modules; returns (job, owners, host, client calls)."""
    client, calls = _fake_client_module(monkeypatch)
    owners = []
    behaviour = {"worker": lambda owner, request: True}

    class FakeOwner:
        def __init__(self, config, host=None):
            self.config = config
            self.host = host
            self.stop_requested = False
            self.wait_for_chunks_var = True
            self.events = []
            # HeadlessOwner's _init_watchdog_dir: a private folder
            private = tmp_path / "private_wd"
            private.mkdir(exist_ok=True)
            os.environ["GLOSSARION_WATCHDOG_DIR"] = str(private)
            owners.append(self)

        def append_log(self, message):
            self.host.log(message)

        def _ui_request(self, kind, *args, **data):
            self.events.append(kind)

        def _record_library_raw_inputs(self, files):
            raise AssertionError("the main window records the Library registry")

        def _reset_api_watchdog_progress(self, *, clear_stale_external_files=True):
            pass

        def _prepare_translation_run(self, files):
            self.prepared = list(files)
            self.watchdog_dir = os.environ.get("GLOSSARION_WATCHDOG_DIR")
            client.setup_multi_key_pool(self.config["multi_api_keys"])
            self._record_library_raw_inputs(files)  # run_job made it a no-op
            return {"files": files}

        def _translation_worker(self, request):
            self._ui_request("trigger_qa_scan")
            self._ui_request("thread_complete")
            return behaviour["worker"](self, request)

    headless = types.ModuleType("headless_owner")
    headless.HeadlessOwner = FakeOwner
    monkeypatch.setitem(sys.modules, "headless_owner", headless)
    headers = types.ModuleType("translate_headers_standalone")
    headers._attach_logging_handlers = lambda owner: owner.events.append("log_handlers")
    monkeypatch.setitem(sys.modules, "translate_headers_standalone", headers)
    import config_store

    snapshot = {"use_multi_api_keys": True, "multi_api_keys": [dict(k) for k in KEYS]}
    monkeypatch.setattr(config_store, "load_config", lambda path: json.loads(json.dumps(snapshot)))
    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    book = tmp_path / "book.epub"
    book.write_text("x", encoding="utf-8")
    monkeypatch.setenv("GLOSSARION_INSTANCE", "2")
    job = {"version": 1, "instance": 2, "files": [str(book)], "config_path": str(config_path), "env": {},
           "watchdog_dir": str(tmp_path / "gui_wd" / "instance_2"), "stop_file": str(tmp_path / "stop.flag"),
           "log_path": None, "parent_pid": None}
    return types.SimpleNamespace(job=job, owners=owners, calls=calls, behaviour=behaviour, client=client,
                                 tmp=tmp_path)


def test_run_job_completed(job_env):
    host = _Host()
    original = vars(job_env.client)["setup_multi_key_pool"]
    assert iw.run_job(job_env.job, host) == iw.EXIT_OK
    owner, = job_env.owners
    assert owner.host is host and owner.prepared == job_env.job["files"]
    # only instance 2's keys reached the pool while the job ran; undone afterwards
    assert job_env.calls == [("pool", ["two-a", "two-b"])]
    assert vars(job_env.client)["setup_multi_key_pool"] is original
    # the main window's watchdog subfolder replaced the owner's private one (removed again)
    assert owner.watchdog_dir == job_env.job["watchdog_dir"] and os.path.isdir(owner.watchdog_dir)
    assert not (job_env.tmp / "private_wd").exists()
    # the post-translation QA request goes to the main window, the other kinds to the owner
    assert iw.QA_SCAN_MARKER in host.lines and owner.events == ["log_handlers", "thread_complete"]
    assert any(line.startswith("▶ Instance 2: 1 file(s)") for line in host.lines)


def test_run_job_failed_and_missing_inputs(job_env):
    job_env.behaviour["worker"] = lambda owner, request: False
    assert iw.run_job(job_env.job, _Host()) == iw.EXIT_FAILED
    host = _Host()
    assert iw.run_job(dict(job_env.job, files=[]), host) == iw.EXIT_FAILED and "no input file" in host.lines[-1]
    host = _Host()
    assert iw.run_job(dict(job_env.job, config_path=str(job_env.tmp / "missing.json")), host) == iw.EXIT_FAILED


def test_run_job_stop_file_stops_the_run(job_env, monkeypatch):
    import stop_control

    monkeypatch.setattr(iw._StopWatcher.__init__, "__defaults__", (0.02,))

    def worker(owner, request):
        iw.write_stop_request(job_env.job["stop_file"], iw.STOP_IMMEDIATE)
        deadline = time.time() + 5
        while not owner.stop_requested and time.time() < deadline:
            time.sleep(0.01)
        return False  # a stopped run returns False

    job_env.behaviour["worker"] = worker
    requests = []
    monkeypatch.setattr(stop_control, "request_stop",
                        lambda **kw: (requests.append(kw["graceful"]), kw["set_stop_requested"]()))
    monkeypatch.setattr(stop_control, "announce_stop", lambda graceful, log: None)
    assert iw.run_job(job_env.job, _Host()) == iw.EXIT_STOPPED
    assert requests == [False]


def test_main_rejects_bad_job_files(tmp_path, monkeypatch, clean_env):
    monkeypatch.setattr(iw, "_configure_output", lambda log_path, instance: None)
    assert iw.main(None, hard_exit=False) == iw.EXIT_FAILED
    assert iw.main(str(tmp_path / "missing.json"), hard_exit=False) == iw.EXIT_FAILED
    bad = tmp_path / "job.json"
    bad.write_text(json.dumps({"version": 99}), encoding="utf-8")
    assert iw.main(str(bad), hard_exit=False) == iw.EXIT_FAILED


def test_main_points_config_json_at_the_snapshot(tmp_path, monkeypatch, clean_env):
    import app_paths
    import large_env

    monkeypatch.setattr(iw, "_configure_output", lambda log_path, instance: None)
    monkeypatch.setattr(app_paths, "CONFIG_FILE", app_paths.CONFIG_FILE)  # restored after the test
    monkeypatch.setattr(large_env, "_store", {})
    seen = {}

    def fake_run_job(job, host=None, parent=None):
        seen.update(config=app_paths.CONFIG_FILE, env=os.environ.get("CONFIG_FILE"),
                    instance=os.environ.get("GLOSSARION_INSTANCE"), stop=os.environ.get("GLOSSARY_STOP_FILE"),
                    big=large_env.get_env("BIG_PROMPT"))
        return iw.EXIT_STOPPED

    monkeypatch.setattr(iw, "run_job", fake_run_job)
    os.environ["GLOSSARY_STOP_FILE"] = "parent.flag"
    job = {"version": 1, "instance": 3, "config_path": str(tmp_path / "config.json"), "env": {"BIG_PROMPT": "p" * 10}}
    job_path = tmp_path / "job.json"
    job_path.write_text(json.dumps(job), encoding="utf-8")
    assert iw.main(str(job_path), hard_exit=False) == iw.EXIT_STOPPED
    assert seen == {"config": str(tmp_path / "config.json"), "env": str(tmp_path / "config.json"),
                    "instance": "3", "stop": None, "big": "p" * 10}


# --------------------------------------------------------------------------
# hygiene
# --------------------------------------------------------------------------

def test_instance_worker_is_gui_free_and_python_310():
    source = (SRC_DIR / "instance_worker.py").read_text(encoding="utf-8")
    tree = ast.parse(source, feature_version=(3, 10))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not imported & {"PySide6", "translator_gui", "dpi_setup", "instance_runs"}, imported
    # the module level imports only the stdlib and instance_plan (it runs before config.json is redirected)
    top = {alias.name for node in tree.body if isinstance(node, ast.Import) for alias in node.names}
    assert top <= {"functools", "json", "os", "sys", "threading", "time", "traceback", "instance_plan"}, top


def test_translator_gui_hands_the_flag_to_main_before_the_gui_imports():
    source = (SRC_DIR / "translator_gui.py").read_text(encoding="utf-8-sig")
    tree = ast.parse(source)
    main_block = next(node for node in tree.body if isinstance(node, ast.If)
                      and ast.unparse(node.test) == "__name__ == '__main__'")
    text = ast.unparse(main_block)
    assert "'--run-translation-instance' in _early_sys.argv" in text
    assert "from instance_worker import main as _instance_worker_main" in text
    assert "raise SystemExit(_instance_worker_main(_job_path))" in text
    first_import = next(i for i, node in enumerate(tree.body) if isinstance(node, (ast.Import, ast.ImportFrom)))
    assert tree.body.index(main_block) < first_import  # nothing (Qt included) is imported before it
    assert iw.FLAG == "--run-translation-instance"
