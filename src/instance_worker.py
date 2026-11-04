"""instance_worker: instance 2..N of a desktop multi-instance translation, run in its own process.

Desktop only and GUI-free: never imports PySide6 or translator_gui. The main window's orchestrator
(``instance_runs``) writes a job file per instance and starts this program again for it,
``Glossarion.exe --run-translation-instance <job.json>`` when frozen and
``python translator_gui.py --run-translation-instance <job.json>`` from source
(:func:`command_line`). The early-argv hook at the top of translator_gui.py calls :func:`main`
before anything imports Qt.

The child runs what a mobile translate job runs (``job_kinds/translate.py``): inside
``job_runner.job_process_state`` it resets the stop state (``stop_control.reset_for_new_run``),
builds a ``HeadlessOwner`` from the config snapshot and calls ``_prepare_translation_run(files)``
and ``_translation_worker(request)``, the desktop Run Translation code.

Nothing is shared with the other instances except the Unified Glossary (``unified_glossary``
locks it across processes):

* ``GLOSSARION_INSTANCE`` is this instance's number, so its payloads go to
  ``Payloads/Instance N`` and its watchdog file says which instance wrote it;
* the config snapshot (``config_path``, encrypted like config.json) holds only this instance's
  keys, in every key pool, and instead of the main GUI key (instance 1's) one of the instance's
  own Translation keys (``instance_plan.config_for_instance``). It replaces config.json for every
  reader in this process (``app_paths.CONFIG_FILE``) and in its helper processes
  (``CONFIG_FILE``), so config.json itself is never read for keys and never written. The main
  window's key variables (:data:`instance_plan.KEY_ENV`) are never inherited, and
  :func:`restrict_keys` additionally keeps every other instance's key out of each key list the
  API client is given or reads, whatever wrote it;
* the watchdog folder is a subfolder of the main window's (``instance_N``), whose aggregator
  reads it; the stop file is this instance's own, and :data:`PER_PROCESS_ENV` (the parent's stop
  files, watchdog folder, run id ...) is never inherited;
* stdout is line-buffered UTF-8 (the main window shows it in this instance's log) and is also
  appended to ``logs/run_instance_N.log``; the main window's run.log is not touched.

Stop: the main window writes :data:`STOP_GRACEFUL` / :data:`STOP_IMMEDIATE` / :data:`STOP_FORCE`
into the stop file (:func:`write_stop_request`); :class:`_StopWatcher` polls it and runs the
desktop Stop protocol (:func:`request_stop`). A force stop from the main window kills the process
tree; when the main window process itself goes away the instance force-stops and exits.

Job file keys: ``version``, ``instance``, ``files``, ``config_path``, ``env`` (values too long for
the process environment, applied through ``large_env``), ``watchdog_dir``, ``stop_file``,
``log_path``, ``parent_pid``, ``key_owners`` (fingerprints of the keys of every other instance and
of the main GUI key: the API client refuses to send one, ``instance_plan.foreign_key_owners``).

Exit codes: 0 completed, 1 failed, 2 stopped.

Rules: Python 3.10 compatible; never import PySide6 or translator_gui.
"""

import json
import os
import sys
import threading
import time
import traceback

import instance_plan

__all__ = [
    "EXIT_FAILED",
    "EXIT_OK",
    "EXIT_STOPPED",
    "FLAG",
    "PER_PROCESS_ENV",
    "QA_SCAN_MARKER",
    "STOP_FORCE",
    "STOP_GRACEFUL",
    "STOP_IMMEDIATE",
    "child_environment",
    "command_line",
    "inheritable",
    "main",
    "read_stop_request",
    "request_stop",
    "restrict_keys",
    "restrict_translation_keys",
    "run_job",
    "write_stop_request",
]

#: The command-line flag translator_gui.py hands to :func:`main`.
FLAG = "--run-translation-instance"
JOB_VERSION = 1

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_STOPPED = 2

#: Stop requests the main window writes into an instance's stop file, mildest first.
STOP_GRACEFUL = "graceful"
STOP_IMMEDIATE = "immediate"
STOP_FORCE = "force"
_STOP_RANK = {STOP_GRACEFUL: 1, STOP_IMMEDIATE: 2, STOP_FORCE: 3}
STOP_POLL_SECONDS = 0.3
#: After the main window process is gone, how long a force-stopped instance may take to finish.
ORPHAN_EXIT_SECONDS = 15.0

#: Printed on its own line when the run asks for the post-translation QA scan; the main window
#: runs it once every instance has finished (desktop: ``trigger_qa_scan_signal``).
QA_SCAN_MARKER = "##GLOSSARION_INSTANCE_QA_SCAN##"

#: The parent's per-process / per-run variables an instance must not inherit: its stop files,
#: watchdog folder and run id, and instance 1's key narrowing (``MULTI_API_KEYS``, ``CONFIG_FILE``,
#: ``GLOSSARION_KEY_INSTANCE``). The key variables (``instance_plan.KEY_ENV``) are dropped too.
PER_PROCESS_ENV = (
    "GLOSSARION_WATCHDOG_DIR",
    "GLOSSARY_STOP_FILE",
    "PDF_EXTRACTION_STOP_FILE",
    "GLOSSARION_RUN_ID",
    "MULTI_API_KEYS",
    "CONFIG_FILE",
    instance_plan.ENV_INSTANCE,
    instance_plan.ENV_KEY_INSTANCE,
)


#: Upper case: Windows reports environment names upper-cased (``OPENAI_OR_GEMINI_API_KEY``).
_NOT_INHERITED = frozenset(name.upper() for name in PER_PROCESS_ENV + instance_plan.KEY_ENV)


def inheritable(name):
    """False for the main window's per-process variables and its API keys / key-pool switches."""
    return str(name).upper() not in _NOT_INHERITED


# ---------------------------------------------------------------------------
# Shared with the main window's orchestrator (instance_runs)
# ---------------------------------------------------------------------------


def command_line(job_path):
    """The command that runs the job at *job_path*: the frozen app itself, or translator_gui.py."""
    if getattr(sys, "frozen", False):
        return [sys.executable, FLAG, job_path]
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "translator_gui.py")
    return [sys.executable, script, FLAG, job_path]


def child_environment(base, instance):
    """The process environment of instance *instance*: *base* without :data:`PER_PROCESS_ENV` and
    without the main window's API keys and key-pool switches (``instance_plan.KEY_ENV``: the
    instance exports its own from its snapshot), with ``GLOSSARION_INSTANCE`` /
    ``GLOSSARION_KEY_INSTANCE`` and UTF-8, unbuffered output."""
    number = str(instance_plan.coerce_instance(instance))
    env = {key: value for key, value in dict(base).items() if inheritable(key)}
    env[instance_plan.ENV_INSTANCE] = number
    env[instance_plan.ENV_KEY_INSTANCE] = number
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONLEGACYWINDOWSSTDIO"] = "0"
    return env


def write_stop_request(stop_file, mode):
    """Ask the instance owning *stop_file* to stop (*mode*: one of the ``STOP_*`` values)."""
    folder = os.path.dirname(os.path.abspath(stop_file))
    os.makedirs(folder, exist_ok=True)
    temp_path = f"{stop_file}.{os.getpid()}.tmp"
    with open(temp_path, "w", encoding="utf-8") as handle:
        handle.write(mode)
    os.replace(temp_path, stop_file)


def read_stop_request(stop_file):
    """The stop request in *stop_file* (a ``STOP_*`` value), or None."""
    try:
        with open(stop_file, "r", encoding="utf-8") as handle:
            mode = handle.read().strip()
    except OSError:
        return None
    return mode if mode in _STOP_RANK else None


def restrict_keys(instance, config=None, *, key_owners=None):
    """From now on this process uses only *instance*'s API keys, in every key pool; returns the
    callable that undoes it.

    Sets ``GLOSSARION_KEY_INSTANCE`` (``instance_plan.key_scope``; the helper processes the run
    starts inherit it): the API client then keeps only that instance's rows of every key list it
    is given or reads (in-memory pools, ``FALLBACK_KEYS`` / ``GLOSSARY_API_KEYS`` / ... variables,
    config.json), and a dedicated pool without an enabled row of the instance is off for it. The
    client's key pools are detached first (``job_runner.isolated_key_pools``), so pool objects
    built before (the Multi API Key Manager shows the Translation one) are never narrowed in place,
    and with *config* rebuilt from it at once (``key_pools.apply_key_pools_to_runtime``).

    *key_owners* (``instance_plan.key_owners`` of the unscoped config: key fingerprints, never the
    keys) tells the client's request guard whose key is whose, so a key of another instance that
    reached a request by any path is refused instead of sent; default: computed from *config*
    (right when *config* is the main window's whole config). The undo puts the variable, the exact
    previous pools and the previous key owners back. Used for instance 1 in the main window while
    a multi-instance run is alive, for each instance's QA scan there, and in every instance process.
    """
    import job_runner
    import unified_api_client  # noqa: F401 - isolated_key_pools acts on an imported client only

    number = instance_plan.coerce_instance(instance)
    if key_owners is None and config is not None:
        key_owners = instance_plan.key_owners(config)
    pools = job_runner.isolated_key_pools()
    pools.__enter__()
    previous = os.environ.get(instance_plan.ENV_KEY_INSTANCE)
    os.environ[instance_plan.ENV_KEY_INSTANCE] = str(number)
    previous_owners = instance_plan.set_key_owners(key_owners) if key_owners is not None else None

    def restore():
        if previous is None:
            os.environ.pop(instance_plan.ENV_KEY_INSTANCE, None)
        else:
            os.environ[instance_plan.ENV_KEY_INSTANCE] = previous
        if previous_owners is not None:
            instance_plan.set_key_owners(previous_owners)
        pools.__exit__(None, None, None)

    if config is not None:
        try:
            from key_pools import apply_key_pools_to_runtime

            apply_key_pools_to_runtime(config, export_env=False)
        except BaseException:
            restore()
            raise
    return restore


#: The former name (it narrowed only the Translation pool); Glossarion Mobile's JobService calls it.
restrict_translation_keys = restrict_keys


# ---------------------------------------------------------------------------
# The instance process
# ---------------------------------------------------------------------------


def request_stop(owner, mode, log=print):
    """The main window's Stop for this instance: the desktop ``stop_translation`` sequence without
    its widgets (as the mobile JobService's ``JobBackend.request_stop`` replays it)."""
    import stop_control

    graceful = mode == STOP_GRACEFUL
    if mode == STOP_FORCE:
        owner.graceful_stop_active = False
        owner._last_stop_was_graceful = False
        stop_control.apply_force_stop_flags()
        log("⚡ Force stop — aborting the current requests now")

    def clear_watchdog():
        owner._reset_api_watchdog_progress(clear_stale_external_files=True)

    if not graceful:
        try:
            clear_watchdog()
        except Exception:
            pass

    def latch():
        owner.graceful_stop_active = graceful
        owner._last_stop_translation_ts = time.time()
        owner._last_stop_was_graceful = graceful
        owner.stop_requested = True

    def translation_stop_flag_hook():
        module = sys.modules.get("TransateKRtoEN")
        set_flag = getattr(module, "set_stop_flag", None) if module is not None else None
        if callable(set_flag):
            set_flag(True)

    stop_control.request_stop(
        graceful=graceful,
        wait_for_chunks=bool(getattr(owner, "wait_for_chunks_var", True)),
        set_stop_requested=latch,
        log=log,
        clear_watchdog=clear_watchdog,
        stop_flag_hook=translation_stop_flag_hook,
    )
    stop_control.announce_stop(graceful, log)


class _ParentProcess:
    """The main window process, taken as early as possible (``main`` does it before any backend
    import): a window that is already gone then reads as gone, never as "unknown"."""

    _SYNCHRONIZE = 0x00100000
    _WAIT_TIMEOUT = 0x102

    def __init__(self, pid):
        self.pid = int(pid) if pid else None
        self._proc = None
        self._handle = None
        self.gone = False
        if not self.pid:
            return  # no parent to watch (a job file written by hand)
        try:
            import psutil
        except ImportError:
            psutil = None
        if psutil is not None:
            try:
                self._proc = psutil.Process(self.pid)
                return
            except psutil.NoSuchProcess:
                self.gone = True
                return
            except Exception:
                pass
        if sys.platform == "win32":
            try:
                import ctypes

                handle = ctypes.windll.kernel32.OpenProcess(self._SYNCHRONIZE, False, self.pid)
            except Exception:
                handle = None
            if handle:
                self._handle = handle
            else:
                self.gone = True

    def running(self):
        if self.gone:
            return False
        if self._proc is not None:
            try:
                return self._proc.is_running()
            except Exception:
                return False
        if self._handle is not None:
            try:
                import ctypes

                return ctypes.windll.kernel32.WaitForSingleObject(self._handle, 0) == self._WAIT_TIMEOUT
            except Exception:
                return False
        if self.pid and sys.platform != "win32":
            return os.getppid() == self.pid  # re-parented once the window is gone
        return True


class _StopWatcher(threading.Thread):
    """Polls this instance's stop file and the main window process; escalates the stop it finds."""

    def __init__(self, stop_file, parent, on_stop, log, interval=STOP_POLL_SECONDS):
        super().__init__(name="instance-stop-watch", daemon=True)
        self.stop_file = stop_file
        self.on_stop = on_stop
        self.log = log
        self.interval = interval
        self.mode = None
        self.parent_gone_at = None
        self._done = threading.Event()
        # *parent*: a _ParentProcess, or the main window's pid (taken here, later than main() does)
        self._parent = parent if isinstance(parent, _ParentProcess) else _ParentProcess(parent)

    def _parent_running(self):
        return self._parent.running()

    def run(self):
        while not self._done.wait(self.interval):
            mode = read_stop_request(self.stop_file) if self.stop_file else None
            if self.parent_gone_at is None and not self._parent_running():
                self.parent_gone_at = time.monotonic()
                self.log("⚠️ The main Glossarion window is gone; stopping this instance")
            if self.parent_gone_at is not None:
                mode = STOP_FORCE
                if time.monotonic() - self.parent_gone_at > ORPHAN_EXIT_SECONDS:
                    _exit(EXIT_STOPPED)
            if mode and _STOP_RANK[mode] > _STOP_RANK.get(self.mode, 0):
                self.mode = mode
                try:
                    self.on_stop(mode)
                except Exception as exc:
                    self.log(f"⚠️ Stop failed: {exc}")

    def stop(self):
        self._done.set()


class _ProcessHost:
    """The ``job_runner.JobHost`` of an instance process.

    Log lines go straight to the stream stdout was at start (the pipe to the main window, plus
    ``run_instance_N.log``), never through ``print``: the backend's ``main()`` points ``sys.stdout``
    at its log callback (this host), so printing here would loop. Compile results are logged;
    questions get their default answer (no one can answer them in this process).
    """

    def __init__(self, stream):
        self._stream = stream
        self._lock = threading.Lock()

    def log(self, text, **_kw):
        with self._lock:
            try:
                self._stream.write(f"{text}\n")
                self._stream.flush()
            except Exception:
                pass

    def emit(self, kind, **data):
        if kind == "compile_result" and data.get("error"):
            self.log(f"❌ {str(data.get('compile_kind') or '').upper()} compilation failed: {data['error']}")

    def ask(self, kind, **data):
        return data.get("default")

    def is_stop_requested(self):
        return False

    def is_graceful_stop(self):
        return False


#: Loggers whose records the API client log handlers forward to the log (as the GUI log shows them);
#: they reach run_instance_N.log through stdout, so the file handler leaves them out.
_CLIENT_LOGGERS = ("unified_api_client", "httpx", "requests.packages.urllib3", "openai")


def _configure_output(log_path, instance):
    """Line-buffered UTF-8 stdout/stderr, every line also appended to *log_path*, which also gets
    the logging records (run.log's format). The root logger has a handler from the start, so a
    later ``logging.basicConfig`` cannot echo every record to stdout a second time."""
    import logging

    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if stream is None:
            try:
                stream = open(1 if name == "stdout" else 2, "w", encoding="utf-8", errors="replace",
                              buffering=1, closefd=False)
            except Exception:
                stream = None
        elif hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
            except Exception:
                pass
        setattr(sys, name, stream)
    root = logging.getLogger()
    log_file = None
    if log_path:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
            log_file = open(log_path, "a", encoding="utf-8", errors="replace", buffering=1)
        except Exception:
            log_file = None
    if log_file is None:
        root.addHandler(logging.NullHandler())
        return None
    records = logging.StreamHandler(log_file)
    records.setFormatter(logging.Formatter(
        fmt="%(asctime)s %(levelname)s [%(process)d:%(threadName)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    records.addFilter(lambda record: not record.name.startswith(_CLIENT_LOGGERS))
    root.addHandler(records)
    root.setLevel(logging.INFO)
    lock = threading.Lock()

    def to_log(line):
        with lock:
            log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [instance {instance}] {line}\n")

    from job_runner import _LineTee

    sys.stdout = _LineTee(sys.stdout, to_log)
    sys.stderr = _LineTee(sys.stderr, to_log)
    return log_file


def _use_watchdog_dir(watchdog_dir):
    """Publish this instance's watchdog file in the main window's folder (``HeadlessOwner`` made a
    private one, which is removed again)."""
    if not watchdog_dir:
        return
    private = os.environ.get("GLOSSARION_WATCHDOG_DIR")
    os.makedirs(watchdog_dir, exist_ok=True)
    os.environ["GLOSSARION_WATCHDOG_DIR"] = watchdog_dir
    if private and os.path.normcase(os.path.abspath(private)) != os.path.normcase(os.path.abspath(watchdog_dir)):
        try:
            os.rmdir(private)
        except OSError:
            pass


def _forward_qa_scan_request(owner, log):
    """The worker's post-translation scan request goes to the main window (:data:`QA_SCAN_MARKER`)
    instead of the GUI-free default; every other ``_ui_request`` kind is unchanged."""
    original = owner._ui_request

    def ui_request(kind, *args, **data):
        if kind == "trigger_qa_scan":
            log(QA_SCAN_MARKER)
            return None
        return original(kind, *args, **data)

    owner._ui_request = ui_request


def run_job(job, host=None, parent=None):
    """Run one instance's translation from a parsed job file; returns the exit code.

    *host* receives the run's log (default: a :class:`_ProcessHost` on the current stdout);
    *parent* is the main window process (:class:`_ParentProcess`; default: taken from the job)."""
    host = host if host is not None else _ProcessHost(sys.stdout)
    log = host.log
    instance = instance_plan.coerce_instance(job.get("instance"))
    files = [os.path.abspath(str(path)) for path in (job.get("files") or []) if path]
    config_path = str(job.get("config_path") or "")
    if not files:
        log(f"❌ Instance {instance}: no input file")
        return EXIT_FAILED
    if not config_path or not os.path.isfile(config_path):
        log(f"❌ Instance {instance}: the settings snapshot was not found: {config_path}")
        return EXIT_FAILED

    import job_runner
    import stop_control
    from config_store import load_config

    watcher = None
    restore_keys = None
    outcome = None
    stop_seen = False
    try:
        with job_runner.job_process_state():
            config = load_config(config_path)
            # whose key is whose: fingerprints of every other instance's keys and the main GUI key
            restore_keys = restrict_keys(instance, key_owners=job.get("key_owners") or {})
            stop_control.reset_for_new_run("translation")
            from headless_owner import HeadlessOwner

            owner = HeadlessOwner(config, host=host)
            _use_watchdog_dir(job.get("watchdog_dir"))
            try:
                import translate_headers_standalone

                # the API client's log records (retries, streamed text) reach stdout like the GUI log
                translate_headers_standalone._attach_logging_handlers(owner)
            except Exception:
                pass
            _forward_qa_scan_request(owner, log)
            # the main window recorded every input of the run in the Library registry before it
            # started the instances; recording again here would race its writes
            owner._record_library_raw_inputs = lambda _files: None
            watcher = _StopWatcher(job.get("stop_file"), parent if parent is not None else job.get("parent_pid"),
                                   lambda mode: request_stop(owner, mode, log), log)
            watcher.start()
            log(f"▶ Instance {instance}: {len(files)} file(s) — " + ", ".join(os.path.basename(p) for p in files))
            request = owner._prepare_translation_run(files)
            if request is None or request is False:
                stop_seen = watcher.mode is not None
                if not stop_seen:
                    log(f"❌ Instance {instance}: the translation did not start (see the log)")
                return EXIT_STOPPED if stop_seen else EXIT_FAILED
            if watcher.mode is not None:
                # a Stop that came before the set-up's own stop-flag reset
                log("⏹️ Translation stopped before it started")
                stop_seen = True
                return EXIT_STOPPED
            outcome = owner._translation_worker(request)
    except SystemExit as exc:  # backend mains may sys.exit()
        if exc.code not in (None, 0):
            log(f"❌ Instance {instance}: the backend exited with code {exc.code}")
            outcome = False
    except BaseException:
        log("".join(traceback.format_exc()))
        outcome = False
    finally:
        if watcher is not None:
            watcher.stop()
            stop_seen = stop_seen or watcher.mode is not None
        try:
            stop_control.clear_own_watchdog_file()
        except Exception:
            pass
        if restore_keys is not None:
            restore_keys()
    if stop_seen:
        return EXIT_STOPPED
    return EXIT_OK if outcome else EXIT_FAILED


def _load_job(job_path):
    with open(job_path, "r", encoding="utf-8") as handle:
        job = json.load(handle)
    if not isinstance(job, dict):
        raise ValueError("the job file does not hold a JSON object")
    if int(job.get("version", 0) or 0) != JOB_VERSION:
        raise ValueError(f"unsupported job version {job.get('version')!r}")
    return job


def main(job_path=None, *, hard_exit=True):
    """Run the instance job at *job_path*; returns its exit code (or, with *hard_exit*, ends the
    process with it once the run's helper processes are gone)."""
    code = EXIT_FAILED
    log_file = None
    try:
        if not job_path:
            print("[ERROR] Translation instance: no job file was given", flush=True)
            return EXIT_FAILED
        job = _load_job(job_path)
        # first: a main window that died while this process was starting must read as gone
        parent = _ParentProcess(job.get("parent_pid"))
        instance = instance_plan.coerce_instance(job.get("instance"))
        if parent.gone:
            print(f"⚠️ Instance {instance}: the main Glossarion window is gone; not starting", flush=True)
            code = EXIT_STOPPED
            return code
        # before any backend import: the payload folder, watchdog file and key filter follow it
        for key in [k for k in os.environ if not inheritable(k)]:
            os.environ.pop(key, None)
        os.environ[instance_plan.ENV_INSTANCE] = str(instance)
        os.environ[instance_plan.ENV_KEY_INSTANCE] = str(instance)
        log_file = _configure_output(job.get("log_path"), instance)
        env = job.get("env") or {}
        if not isinstance(env, dict):
            raise ValueError("the job environment must be an object")
        import large_env

        large_env.update_env({str(k): "" if v is None else str(v) for k, v in env.items() if inheritable(str(k))})
        # the snapshot replaces config.json for this process (modules read app_paths.CONFIG_FILE
        # when they are imported, which happens after this) and for its helper processes
        import app_paths

        config_path = os.path.abspath(str(job.get("config_path") or ""))
        app_paths.CONFIG_FILE = config_path
        os.environ["CONFIG_FILE"] = config_path
        code = run_job(job, parent=parent)
        return code
    except Exception as exc:
        print(f"[ERROR] Translation instance failed: {exc}", flush=True)
        traceback.print_exc()
        code = EXIT_FAILED
        return code
    finally:
        if hard_exit:
            _exit(code, log_file)


def _exit(code, log_file=None):
    """End this process: its helper processes first (glossary workers), then a hard exit so a
    backend thread still closing a stream cannot keep it alive."""
    try:
        from shutdown_utils import terminate_current_process_children

        terminate_current_process_children(timeout=1.5)
    except Exception:
        pass
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:
            pass
    if log_file is not None:
        try:
            log_file.close()
        except Exception:
            pass
    os._exit(int(code))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
