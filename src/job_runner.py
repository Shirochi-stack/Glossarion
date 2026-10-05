"""job_runner: process-state scoping, job hooks and progress events for one backend job at a time.

Shared GUI-free core (Glossarion mobile rewrite, milestone U3). The backend runs in
process and its configuration channel is process-global (``os.environ``, ``sys.argv``,
``large_env``'s overflow store, ``sys.stdout``, ``UnifiedClient`` class-level key
pools), so exactly one job may own it at a time:

* ``JOB_LOCK``: the lock a mobile job (and anything else that builds a
  ``HeadlessOwner``: env preview, key tests) holds while it owns the process state.
  The desktop keeps its own concurrency gating and never takes it.
* ``scoped_process_state(...)``: snapshot/restore of argv + environment (+ the
  ``large_env`` store, optionally stdout/stderr, the cwd and the key pools). The
  default restore order is the desktop's (``_process_text_file``): ``sys.argv``
  rebound to the saved list, ``os.environ.clear()`` + ``update(saved)``, then
  ``large_env.clear_store()``. ``keywise=True`` restores key by key instead (never
  clears, so other threads keep reading unchanged variables; the mobile JobService
  and the env preview use it); ``restore_mapping`` / ``isolated_key_pools`` moved
  here from the mobile env preview.
* ``JobHooksMixin``: GUI-free defaults of the job hooks the moved desktop code calls
  (``_backend_entry``, ``_ui_request``, ``_notify_compile_result``); TranslatorGUI
  overrides them with its globals, Qt signals and message boxes.
* ``JobHost`` (protocol), ``JobEvent`` and ``ProgressWatcher``, which turns a
  workspace's ``translation_progress.json`` (``library_core._read_progress_summary``)
  and ``unified_api_client.get_api_watchdog_state()`` into ``progress`` /
  ``api_state`` events.

Rules: Python 3.10 compatible; never import PySide6, translator_gui or dpi_setup.
"""

import contextlib
import io
import os
import sys
import threading
import time
from dataclasses import dataclass, field

try:  # typing.Protocol exists on 3.8+; keep the module importable everywhere
    from typing import Protocol, runtime_checkable
except ImportError:  # pragma: no cover
    Protocol = object

    def runtime_checkable(cls):
        return cls

__all__ = [
    "BACKEND_ENTRY_POINTS",
    "JOB_LOCK",
    "JobEvent",
    "JobHooksMixin",
    "JobHost",
    "ProgressWatcher",
    "UI_REQUEST_FIELDS",
    "isolated_key_pools",
    "job_process_state",
    "restore_mapping",
    "run_exclusive",
    "scoped_process_state",
]

#: One backend job owns the process state at a time (mobile JobService, env preview,
#: anything that builds a HeadlessOwner: its init writes ~40 environment variables).
JOB_LOCK = threading.RLock()


def run_exclusive(fn, *args, **kwargs):
    """``fn(*args, **kwargs)`` while holding JOB_LOCK."""
    with JOB_LOCK:
        return fn(*args, **kwargs)


# ---------------------------------------------------------------------------
# Host protocol and events
# ---------------------------------------------------------------------------


@runtime_checkable
class JobHost(Protocol):
    """What a job reports to and asks of its front end (mobile JobService, tests).

    ``emit(kind, **data)`` kinds: ``log``, ``phase``, ``progress``, ``api_state``,
    ``result``, ``question`` and the ``_ui_request`` kinds (``thread_complete``,
    ``input_files_updated``, ``compile_result``, ...). ``ask`` blocks the job thread
    until the user answers (glossary approval) and returns the answer.
    """

    def log(self, text, **kw): ...

    def is_stop_requested(self) -> bool: ...

    def is_graceful_stop(self) -> bool: ...

    def emit(self, kind, **data): ...

    def ask(self, kind, **data): ...


@dataclass
class JobEvent:
    """One event of a job, as the front end receives it."""

    kind: str
    job_id: str = ""
    ts: float = field(default_factory=time.time)
    data: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Process-state scoping (moved from the mobile env preview: restore_mapping,
# isolated_key_pools; desktop restore order from _process_text_file)
# ---------------------------------------------------------------------------


def restore_mapping(target, saved):
    """Put ``target`` back to ``saved`` key by key: drop added keys, reset changed ones.

    Never clears the mapping, so keys that did not change (``GLOSSARION_*``,
    ``CONFIG_FILE``, ...) stay readable by other threads the whole time; the
    ``mobile_runtime`` gates read ``os.environ`` live."""
    for key in [k for k in list(target.keys()) if k not in saved]:
        target.pop(key, None)
    for key, value in saved.items():
        if target.get(key) != value:
            target[key] = value


#: UnifiedClient class attributes that ``key_pools.apply_key_pools_to_runtime`` writes through the
#: ``set_/clear_in_memory_*`` and ``setup_*_key_pool`` class methods (beyond the ``_in_memory_*``
#: lists, ``*_key_pool`` objects and ``*_pool_logged`` / ``_last_*_pool_setup_status`` flags).
_POOL_EXTRA_ATTRS = ("_force_rotation", "_rotation_frequency", "_rate_limit_cache")


def _is_pool_state_attr(name):
    if name.endswith("_lock"):
        return False
    return (name.startswith("_in_memory_") or name.endswith(("_key_pool", "_pool_logged", "_pool_setup_status"))
            or name in _POOL_EXTRA_ATTRS)


#: Class-body members that are code, not pool state: ``setup_multi_key_pool``, ``get_key_pool``,
#: ``_get_active_key_pool`` ... match the ``*_key_pool`` names but must never be detached.
_CODE_MEMBER_TYPES = (classmethod, staticmethod, property, type(_is_pool_state_attr))


def _pool_state_items(cls):
    """``{name: value}`` of the pool state attributes in ``vars(cls)`` (methods excluded)."""
    return {n: v for n, v in vars(cls).items()
            if _is_pool_state_attr(n) and not isinstance(v, _CODE_MEMBER_TYPES)}


@contextlib.contextmanager
def isolated_key_pools():
    """Give the body fresh ``UnifiedClient`` key pools and put the previous pool state back afterwards.

    ``run_env.build_translation_env`` applies the snapshot's key pools
    (``key_pools.apply_key_pools_to_runtime``): class attributes are replaced and the shared
    ``APIKeyPool`` objects are reloaded in place. The pool objects are detached first
    (``setup_*_key_pool`` builds a new pool when the attribute is None), so the previous pools
    are never mutated, and every pool attribute is restored on exit. Only acts when
    ``unified_api_client`` is already imported."""
    module = sys.modules.get("unified_api_client")
    cls = getattr(module, "UnifiedClient", None) if module is not None else None
    saved = _pool_state_items(cls) if cls is not None else {}
    if cls is not None:
        for name, value in saved.items():
            if name.endswith("_key_pool") and value is not None:
                setattr(cls, name, None)
    try:
        yield
    finally:
        if cls is not None:
            missing = object()
            for name in [n for n in _pool_state_items(cls) if n not in saved]:
                try:
                    delattr(cls, name)
                except AttributeError:
                    pass
            for name, value in saved.items():
                if vars(cls).get(name, missing) is not value:
                    setattr(cls, name, value)


class _LineTee(io.TextIOBase):
    """Text stream that forwards writes to *stream* and complete lines to *sink*."""

    def __init__(self, stream, sink):
        super().__init__()
        self._stream = stream
        self._sink = sink
        self._buffer = ""
        self._lock = threading.Lock()

    def writable(self):
        return True

    def write(self, text):
        if not isinstance(text, str):
            text = str(text)
        if self._stream is not None:
            try:
                self._stream.write(text)
            except Exception:
                pass
        lines = []
        with self._lock:
            self._buffer += text
            while "\n" in self._buffer:
                line, self._buffer = self._buffer.split("\n", 1)
                lines.append(line.rstrip("\r"))
        for line in lines:
            try:
                self._sink(line)
            except Exception:
                pass
        return len(text)

    def flush(self):
        if self._stream is not None:
            try:
                self._stream.flush()
            except Exception:
                pass

    def drain(self):
        """Send a trailing partial line to the sink."""
        with self._lock:
            rest, self._buffer = self._buffer, ""
        if rest:
            try:
                self._sink(rest.rstrip("\r"))
            except Exception:
                pass

    @property
    def encoding(self):
        return getattr(self._stream, "encoding", "utf-8")

    def isatty(self):
        return False

    def fileno(self):
        if self._stream is None:
            raise OSError("no underlying stream")
        return self._stream.fileno()


class scoped_process_state:  # noqa: N801 - used like a function: ``with scoped_process_state(...):``
    """Snapshot argv/env (and optionally more) and restore them when the block exits.

    * ``argv``: set ``sys.argv`` to this list on entry (None: leave it).
    * ``env_updates``: exported on entry (through ``large_env.update_env``, so values
      over the Windows limit land in its store).
    * ``restore_env``: restore ``os.environ`` on exit (desktop compile jobs never did).
    * ``clear_large_env``: undo ``large_env``'s overflow store on exit.
    * ``capture_stdout``: callable receiving every complete line printed to stdout /
      stderr inside the block (the output still reaches the real streams).
    * ``lock``: held for the whole block (mobile: ``JOB_LOCK``); acquired before the
      snapshot.
    * ``keywise``: restore the environment and the ``large_env`` store key by key
      (``restore_mapping``) and ``sys.argv`` in place, instead of the desktop's
      clear + update / ``clear_store()`` / rebinding.
    * ``isolate_key_pools``: run the block inside ``isolated_key_pools()``.
    * ``restore_cwd``: chdir back to the working directory on exit.

    The snapshot is taken on entry; ``snapshot()`` takes it earlier (the desktop
    glossary extractor snapshots before computing its output paths). Exit order:
    key pools, ``sys.argv``, environment, ``large_env``, cwd, stdout/stderr, lock.
    """

    def __init__(self, argv=None, *, env_updates=None, restore_env=True, clear_large_env=True,
                 capture_stdout=None, lock=None, keywise=False, isolate_key_pools=False,
                 restore_cwd=False):
        self.argv = argv
        self.env_updates = env_updates
        self.restore_env = restore_env
        self.clear_large_env = clear_large_env
        self.capture_stdout = capture_stdout
        self.lock = lock
        self.keywise = keywise
        self.isolate_key_pools = isolate_key_pools
        self.restore_cwd = restore_cwd
        self._taken = False
        self._locked = False
        self._pools = None
        self._tees = None

    # -- snapshot -----------------------------------------------------------
    def snapshot(self):
        """Take the snapshot now (idempotent); returns self."""
        if self._taken:
            return self
        self.old_argv = sys.argv
        self.saved_argv = list(sys.argv)
        self.old_env = dict(os.environ)
        large = sys.modules.get("large_env")
        store = getattr(large, "_store", None)
        self.old_large_store = dict(store) if isinstance(store, dict) else None
        if self.restore_cwd:
            try:
                self.old_cwd = os.getcwd()
            except OSError:
                self.old_cwd = None
        else:
            self.old_cwd = None
        self._taken = True
        return self

    # -- context manager ----------------------------------------------------
    def __enter__(self):
        if self.lock is not None:
            self.lock.acquire()
            self._locked = True
        try:
            self.snapshot()
            if self.isolate_key_pools:
                self._pools = isolated_key_pools()
                self._pools.__enter__()
            if self.capture_stdout is not None:
                self._tees = (sys.stdout, sys.stderr)
                sys.stdout = _LineTee(sys.stdout, self.capture_stdout)
                sys.stderr = _LineTee(sys.stderr, self.capture_stdout)
            if self.env_updates:
                try:
                    import large_env
                    large_env.update_env(self.env_updates)
                except ImportError:
                    for key, value in self.env_updates.items():
                        os.environ[str(key)] = '' if value is None else str(value)
            if self.argv is not None:
                sys.argv = list(self.argv)
        except BaseException:
            self._release()
            raise
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if self._pools is not None:
                pools, self._pools = self._pools, None
                pools.__exit__(exc_type, exc, tb)
            self.restore()
        finally:
            self._release()
        return False

    def restore(self):
        """Restore what the snapshot holds (the order documented on the class)."""
        if self.keywise:
            if sys.argv is not self.old_argv:
                sys.argv = self.old_argv
            sys.argv[:] = self.saved_argv
            if self.restore_env:
                restore_mapping(os.environ, self.old_env)
            if self.clear_large_env:
                large = sys.modules.get("large_env")
                store = getattr(large, "_store", None)
                if isinstance(store, dict):
                    restore_mapping(store, self.old_large_store or {})
        else:
            sys.argv = self.old_argv
            if self.restore_env:
                os.environ.clear()
                os.environ.update(self.old_env)
            if self.clear_large_env:
                try:
                    import large_env
                    large_env.clear_store()
                except Exception:
                    pass
        if self.restore_cwd and self.old_cwd is not None:
            try:
                if os.getcwd() != self.old_cwd:
                    os.chdir(self.old_cwd)
            except OSError:
                pass

    def _release(self):
        if self._tees is not None:
            out, err = self._tees
            self._tees = None
            for tee in (sys.stdout, sys.stderr):
                if isinstance(tee, _LineTee):
                    tee.drain()
            sys.stdout, sys.stderr = out, err
        if self._locked:
            self._locked = False
            self.lock.release()


def job_process_state(capture_stdout=None, *, lock=JOB_LOCK):
    """The scope one mobile job runs in (the JobService wraps every job with it).

    Holds ``JOB_LOCK``; restores the environment, the ``large_env`` store, ``sys.argv``
    and the cwd key by key (other threads keep reading unchanged variables); gives the
    job fresh ``UnifiedClient`` key pools and puts the previous pool state back; sends
    every printed line to *capture_stdout*. The desktop runners inside the job keep
    their own clear + update scope (``_process_text_file``), exactly as on desktop.
    """
    return scoped_process_state(capture_stdout=capture_stdout, lock=lock, keywise=True,
                                isolate_key_pools=True, restore_cwd=True)


# ---------------------------------------------------------------------------
# Job hooks (GUI-free defaults; TranslatorGUI overrides them)
# ---------------------------------------------------------------------------

#: ``_backend_entry(name)`` -> (module, attribute). The desktop resolves the same
#: names from its lazily loaded translator_gui globals.
BACKEND_ENTRY_POINTS = {
    "translation_main": ("TransateKRtoEN", "main"),
    "translation_stop_flag": ("TransateKRtoEN", "set_stop_flag"),
    "glossary_main": ("extract_glossary_from_epub", "main"),
    "glossary_stop_flag": ("extract_glossary_from_epub", "set_stop_flag"),
    "fallback_compile_epub": ("epub_converter", "fallback_compile_epub"),
}

#: ``_ui_request(kind, *args)`` positional arguments -> event field names.
UI_REQUEST_FIELDS = {
    "thread_complete": (),
    "input_files_updated": ("files",),
    "trigger_qa_scan": (),
    "refresh_preview": (),
    "open_progress_manager": (),
}


class JobHooksMixin:
    """GUI-free defaults of the hooks the moved desktop job code calls (shared-core §3.3).

    ``TranslatorGUI`` overrides each one with the original code (module globals,
    Qt signals, message boxes); ``HeadlessOwner`` uses these, reporting to
    ``self.host`` (a ``JobHost``) when it has one.
    """

    def _backend_entry(self, name):
        """Backend entry point *name* (``BACKEND_ENTRY_POINTS``), imported lazily; None when unavailable.

        Static lazy imports (not importlib) so the mobile backend collector sees the edges.
        """
        module_name, attr = BACKEND_ENTRY_POINTS[name]
        try:
            if module_name == "TransateKRtoEN":
                import TransateKRtoEN as module
            elif module_name == "extract_glossary_from_epub":
                import extract_glossary_from_epub as module
            elif module_name == "epub_converter":
                import epub_converter as module
            else:  # pragma: no cover - BACKEND_ENTRY_POINTS lists only the three above
                return None
        except Exception:
            return None
        return getattr(module, attr, None)

    def _ui_request(self, kind, *args, **data):
        """Tell the front end something happened (desktop: ``<kind>_signal.emit(*args)``)."""
        host = getattr(self, 'host', None)
        emit = getattr(host, 'emit', None)
        if not callable(emit):
            return None
        names = UI_REQUEST_FIELDS.get(kind)
        payload = dict(zip(names, args)) if names is not None and len(args) <= len(names) else (
            {"args": list(args)} if args else {})
        payload.update(data)
        emit(kind, **payload)
        return None

    def _notify_compile_result(self, kind, path=None, error=None):
        """A compile job finished (desktop: success / failure message box)."""
        host = getattr(self, 'host', None)
        emit = getattr(host, 'emit', None)
        if callable(emit):
            emit('compile_result', compile_kind=kind, path=path, error=error)
        return None


# ---------------------------------------------------------------------------
# Progress events
# ---------------------------------------------------------------------------

_PROGRESS_FIELDS = ("total", "completed", "in_progress", "failed")


def _default_summary_reader(progress_file):
    from library_core import _read_progress_summary

    return _read_progress_summary(progress_file)


def _default_watchdog_state():
    module = sys.modules.get("unified_api_client")
    if module is None:
        return None
    getter = getattr(module, "get_api_watchdog_state", None)
    return getter() if callable(getter) else None


class ProgressWatcher:
    """Poll a workspace's ``translation_progress.json`` and the API watchdog; emit events.

    ``output_dir_resolver``: the output folder (str) or a callable returning it (None
    while unknown); mobile passes ``lambda: owner._resolve_translation_output_dir(path)``.
    Every *interval* seconds (and on ``poll_once()``) it emits through ``host.emit``:

    * ``progress`` ``{total, completed, in_progress, failed, path}`` when the progress
      file changed (mtime/size gated) and its summary differs from the last one; the
      counts are ``library_core._read_progress_summary`` (the Library card numbers);
    * ``api_state`` ``{in_flight, backlog, scheduler_queued, waiting, queued_entries,
      peak_in_flight, last_context, last_model}`` from
      ``unified_api_client.get_api_watchdog_state()`` when it changed (only once the
      client module is loaded).
    """

    def __init__(self, output_dir_resolver, host, interval=2.0, *,
                 progress_filename="translation_progress.json", summary_reader=None,
                 watchdog_state=None, name="gl-progress"):
        self._resolver = output_dir_resolver
        self.host = host
        self.interval = max(0.05, float(interval))
        self.progress_filename = progress_filename
        self._summary_reader = summary_reader or _default_summary_reader
        self._watchdog_state = watchdog_state or _default_watchdog_state
        self.name = name
        self._signature = None
        self._last_progress = None
        self._last_api = None
        self._stop = threading.Event()
        self._thread = None
        self._poll_lock = threading.Lock()

    # -- polling ------------------------------------------------------------
    def output_dir(self):
        try:
            value = self._resolver() if callable(self._resolver) else self._resolver
        except Exception:
            return None
        return str(value) if value else None

    def progress_path(self):
        folder = self.output_dir()
        return os.path.join(folder, self.progress_filename) if folder else None

    def _emit(self, kind, data):
        emit = getattr(self.host, "emit", None)
        if callable(emit):
            try:
                emit(kind, **data)
            except Exception:
                pass

    def _poll_progress(self):
        path = self.progress_path()
        if not path:
            return None
        try:
            st = os.stat(path)
            signature = (path, st.st_mtime_ns, st.st_size)
        except OSError:
            signature = None
        if signature == self._signature:
            return None
        self._signature = signature
        if signature is None:
            return None
        try:
            summary = self._summary_reader(path)
        except Exception:
            summary = None
        if not isinstance(summary, dict):
            return None
        data = {key: int(summary.get(key, 0) or 0) for key in _PROGRESS_FIELDS}
        data["path"] = path
        if data == self._last_progress:
            return None
        self._last_progress = data
        return data

    def _poll_api(self):
        try:
            state = self._watchdog_state()
        except Exception:
            state = None
        if not isinstance(state, dict):
            return None
        entries = [e for e in (state.get("in_flight_entries") or []) if isinstance(e, dict)]

        def _count(key):
            try:
                return max(0, int(state.get(key, 0) or 0))
            except (TypeError, ValueError):
                return 0

        data = {
            "in_flight": _count("in_flight"),
            "backlog": _count("backlog"),
            "scheduler_queued": _count("scheduler_queued"),
            "waiting": sum(1 for e in entries if e.get("status") == "waiting_cooldown"),
            "queued_entries": sum(1 for e in entries if e.get("status") == "queued"),
            "peak_in_flight": _count("peak_in_flight"),
            "last_context": state.get("last_context"),
            "last_model": state.get("last_model"),
        }
        if data == self._last_api:
            return None
        self._last_api = data
        return data

    def poll_once(self):
        """Poll both sources now; returns the ``[(kind, data), ...]`` it emitted."""
        emitted = []
        with self._poll_lock:
            progress = self._poll_progress()
            if progress is not None:
                emitted.append(("progress", progress))
            api = self._poll_api()
            if api is not None:
                emitted.append(("api_state", api))
        for kind, data in emitted:
            self._emit(kind, dict(data))
        return emitted

    # -- thread ---------------------------------------------------------------
    def _run(self):
        while True:
            try:
                self.poll_once()
            except Exception:
                pass
            if self._stop.wait(self.interval):
                return

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return self
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name=self.name, daemon=True)
        self._thread.start()
        return self

    def stop(self, final_poll=True, timeout=5.0):
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout)
        if final_poll:
            try:
                self.poll_once()
            except Exception:
                pass

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc, tb):
        self.stop()
        return False
