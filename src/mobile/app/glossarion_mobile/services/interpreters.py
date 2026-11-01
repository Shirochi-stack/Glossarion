"""CPU work outside the GIL of the Flet loop: a pool of own-GIL Python interpreters in this process.

Android and iOS have no process pools, and threads share one GIL with the UI loop. A child interpreter
(PEP 684, ``_interpreters`` on Python 3.13, ``_xxsubinterpreters`` on 3.12) has its own GIL, so a batch in one neither
waits for the loop nor makes the loop wait.

    pool = InterpreterPool(4, preload=("scan_html_folder",))
    future = pool.submit("scan_html_folder", "process_html_file_batch", args, should_stop=stop)
    value = pool.call(...)   # the same, waiting here; a thread pool of the caller's own decides the order
    pool.close()

Each child lives on one thread of its own, from its creation to its destruction. On Python 3.12 destroying a child
from another thread than the one it ran on hung once that thread had ended (host, ``_xxsubinterpreters.destroy``).

What a batch needs to know (a child shares no Python object with this interpreter):

* the function is found by module and name in the child (``importable``), its arguments and result are pickled.
  The request goes across in memory (``shared``), so a key in the arguments or the environment never reaches a
  file; the result comes back in a file in the pool's private folder;
* the environment: a child holds a plain dict in ``os.environ``. Each call brings it up to the caller's, and what
  a batch writes there stays in the child: ``os.putenv`` is not safe from several GILs at once;
* ``sys.path`` and the recursion limit are the caller's at the time the child starts. Module globals are the
  child's own: what a worker reads must be in its arguments or the environment;
* a Stop: ``should_stop`` is polled here (``STOP_POLL``) and a child sees it through a file, at most ``CHILD_POLL``
  old, in the check it gets as the worker's second argument. A child cannot be killed: leaving a batch behind
  after a Stop is fine, it ends at its next item and its interpreter is released then;
* what a batch prints is written to this interpreter's ``sys.stdout`` when it ends (a job's log tees that); with
  ``live_output`` while it runs, at most ``OUTPUT_POLL`` late, for a call that takes minutes;
* an exception in a worker is raised by ``call`` / the future like it was raised here, with the child's traceback
  as its ``__cause__``. Only a child that cannot start or do a call raises ``InterpreterUnavailable``.

Not everything loads in a child. Refused on the device (Android x86_64, Python 3.13): lxml.etree, regex, numpy,
charset_normalizer, pydantic_core, cryptography's Rust core, tiktoken, PIL.Image, cv2, onnxruntime, pymupdf, grpc,
llama_cpp, ebooklib.epub (through lxml). Loaded: re, json,
hashlib, zlib, unicodedata, _multibytecodec, _ctypes, sqlite3, ssl, pyexpat, html.parser, difflib, bs4, soupsieve,
langdetect, tqdm, chardet, html2text, requests, scan_html_folder, GlossaryManager (``services.glossary_child``). RapidFuzz
loads, but its C++ is refused and it silently takes its pure Python, 20 times slower (the same scores on real names,
``test_interpreters.py``): work that lives on it stays on a thread. Also ``import lxml`` succeeds and BeautifulSoup then
raises ``FeatureNotFound`` for 'lxml'. Daemon threads are refused in a child, so a child cannot run this pool itself. A
worker whose ``try: import`` falls back when a module is refused would silently do something else in a child: check that
before moving a worker here. Work that only waits on the network stays on threads: it holds no GIL while it waits.

Memory: about 40 MB per interpreter on the device at a scan's peak (emulator, Python 3.13: 4 of them +157 to +203 MB,
12 of them +409 MB), so one pool is reused for all the batches of a job and closed after it; 60 to 150 MB of the peak
stays in the allocator for a while after it closes. ``size_for_memory`` is the rule for how many.

Python 3.10 compatible to import (``available()`` says False below 3.12).
"""

from __future__ import annotations

import atexit
import concurrent.futures
import importlib
import importlib.util
import itertools
import logging
import os
import pickle
import queue
import shutil
import sys
import tempfile
import threading
import weakref
from typing import Any, Callable, Optional, Sequence

__all__ = ["CHILD_MB", "HEADROOM_MB", "InterpreterPool", "InterpreterUnavailable", "JobChildren", "WorkerError",
           "available", "importable", "last_line", "memory_now", "set_child_setup", "size_for_memory",
           "unavailable_reason"]

log = logging.getLogger("glossarion.interpreters")

#: how often (s) a ``should_stop`` is asked while a call that uses it runs
STOP_POLL = 0.05
#: how old (s) a child's knowledge of a Stop may be: its check looks at the file at most this often
CHILD_POLL = 0.02
#: how late (s) a line of a ``live_output`` call reaches this interpreter's ``sys.stdout``
OUTPUT_POLL = 0.2
#: what one interpreter is counted as: at most 50 MB each measured on the device at a scan's peak, the scanner imported
#: and a batch's arguments and results in it (4 of them +157 to +203 MB)
CHILD_MB = 50
#: free memory that stays free for the app and the rest of the phone when children start
HEADROOM_MB = 300
#: the share of RAM counted as free where the system does not say (iOS: an app is stopped well before all of it)
UNKNOWN_FREE_SHARE = 0.25
#: how long (s) ``close`` waits for an idle child's interpreter to be destroyed (milliseconds on the host)
RELEASE_WAIT = 10.0


class InterpreterUnavailable(RuntimeError):
    """A child could not start, or could not do the call: do this work another way (it did not run, or its result
    is lost)."""


class WorkerError(RuntimeError):
    """A worker raised something that cannot cross between interpreters; the message has its traceback."""


class _ChildTraceback(Exception):
    """The ``__cause__`` of an exception from a worker: where it happened, in the child."""

    def __str__(self) -> str:
        return "\n" + str(self.args[0] if self.args else "")


def _low_level() -> Any:
    for name in ("_interpreters", "_xxsubinterpreters"):  # 3.13, 3.12
        try:
            return importlib.import_module(name)
        except ImportError:
            continue
    return None


_api = _low_level()
#: (module, function) every new child runs, ``set_child_setup``
_child_setup: Optional[tuple] = None


def available() -> bool:
    """This Python can make own-GIL interpreters (whether a child can load anything is found out by starting one)."""
    return unavailable_reason() is None


def unavailable_reason() -> Optional[str]:
    if _api is None:
        return f"Python {sys.version_info[0]}.{sys.version_info[1]} has no interpreter module"
    return None


def set_child_setup(setup: Optional[tuple]) -> None:
    """A ``(module, function)`` each child started from now on runs first (the E2E's tripwires); None for none."""
    global _child_setup
    _child_setup = tuple(setup) if setup else None


def importable(function: Any) -> bool:
    """A child can find this function: at the top level of a module, and what that module's name gives here."""
    module = getattr(function, "__module__", None)
    name = getattr(function, "__qualname__", "") or ""
    if not module or module == "__main__" or "." in name:
        return False
    return getattr(sys.modules.get(module), name, None) is function


def size_for_memory(wanted: int, available_bytes: Optional[int] = None, total_bytes: int = 0) -> int:
    """How many of ``wanted`` interpreters to start: as many as leave ``HEADROOM_MB`` free at ``CHILD_MB`` each, 0
    when not even one does. Free memory comes from ``available_bytes``, else the share ``UNKNOWN_FREE_SHARE`` of
    ``total_bytes``; when neither is known, the wanted number (no number says otherwise)."""
    wanted = max(0, int(wanted))
    free = available_bytes if available_bytes else int(total_bytes * UNKNOWN_FREE_SHARE)
    if not free:
        return wanted
    return max(0, min(wanted, (int(free / (1024 * 1024)) - HEADROOM_MB) // CHILD_MB))


def _create() -> Any:
    if _api is None:
        raise InterpreterUnavailable(unavailable_reason())
    if hasattr(_api, "exec"):  # 3.13
        return _api.create("isolated")
    return _api.create(isolated=True)


def _run(child: Any, code: str, shared: Optional[dict] = None) -> Optional[str]:
    """Run ``code`` in the child on this thread (this interpreter's GIL is released meanwhile). What went wrong outside
    the worker's own try (a worker's exception comes back as a result), else None."""
    if hasattr(_api, "exec"):  # 3.13: returns what went wrong
        failure = _api.exec(child, code, shared) if shared else _api.exec(child, code)
        return None if failure is None else str(getattr(failure, "formatted", failure))
    try:  # 3.12: raises it
        if shared:
            _api.run_string(child, code, shared)
        else:
            _api.run_string(child, code)
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


# What a child runs once. Its __main__ keeps these between the calls. Nothing here may need a native module the child
# has not loaded yet: the module finder of a packaged app comes first (``_child_source``).
_CHILD_CODE = r"""
import io as _gl_io, importlib as _gl_importlib, os as _gl_os, pickle as _gl_pickle, sys as _gl_sys, time as _gl_time
import traceback as _gl_traceback

_gl_os.environ = dict(_gl_os.environ)  # this child's: no os.putenv from several GILs


def _gl_stop_check(path, poll):
    state = [0.0, False]

    def should_stop():
        if not state[1]:
            now = _gl_time.monotonic()
            if now - state[0] >= poll:
                state[0] = now
                state[1] = _gl_os.path.exists(path)
        return state[1]

    return should_stop


def _gl_function(module, name):
    return getattr(_gl_importlib.import_module(module), name)


def _gl_prepare():
    env, setup, preload, after = _gl_pickle.loads(globals().pop("_gl_request"))
    _gl_os.environ.clear()
    _gl_os.environ.update(env)
    if setup is not None:
        _gl_function(*setup)()
    for module in preload:
        _gl_importlib.import_module(module)
    globals()["_gl_after"] = _gl_function(*after) if after else None


def _gl_call():
    changed, gone, module, name, args, stop_path, poll, reply_path, output_path = _gl_pickle.loads(globals().pop("_gl_request"))
    _gl_os.environ.update(changed)
    for key in gone:
        _gl_os.environ.pop(key, None)
    printed = (open(output_path, "a", encoding="utf-8", errors="replace", newline="\n", buffering=1) if output_path
               else _gl_io.StringIO())
    streams = _gl_sys.stdout, _gl_sys.stderr
    _gl_sys.stdout = _gl_sys.stderr = printed
    try:
        function = _gl_function(module, name)
        value = function(args) if stop_path is None else function(args, _gl_stop_check(stop_path, poll))
        reply = ("value", value, None)
    except BaseException as exc:
        reply = ("raised", exc, _gl_traceback.format_exc())
    finally:
        _gl_sys.stdout, _gl_sys.stderr = streams
    del args
    if output_path:
        printed.close()
        printed = _gl_io.StringIO()
    try:
        data = _gl_pickle.dumps(reply + (printed.getvalue(),), protocol=_gl_pickle.HIGHEST_PROTOCOL)
        if reply[0] == "raised":
            _gl_pickle.loads(data)  # some exceptions pickle and then cannot be rebuilt
    except BaseException as exc:
        data = _gl_pickle.dumps(("error", f"{reply[1]!r} cannot cross ({exc!r})", reply[2] or _gl_traceback.format_exc(),
                                 printed.getvalue()), protocol=_gl_pickle.HIGHEST_PROTOCOL)
    del reply
    with open(reply_path, "wb") as handle:
        handle.write(data)
    if _gl_after is not None:
        try:
            _gl_after()
        except Exception:  # the result is out; a cache that did not clear is no reason to lose this child
            pass
"""


def _child_source() -> str:
    """The child's first lines: the caller's ``sys.path`` and recursion limit, and on a packaged Android app the
    finder without which a child cannot load any native module (``_sp_bootstrap``; iOS has none and is said not to
    need one: ``diagnostics.selftest`` records what happens there)."""
    lines = [f"import sys; sys.path[:] = {list(sys.path)!r}; sys.setrecursionlimit({sys.getrecursionlimit()})"]
    if importlib.util.find_spec("_sp_bootstrap") is not None:
        lines.append("import _sp_bootstrap; _sp_bootstrap.install()")
    return "\n".join(lines) + "\n" + _CHILD_CODE


class _Child:
    """An interpreter and the one thread that touches it. ``do`` runs something on that thread and waits; ``finish``
    ends the thread, which destroys the interpreter (and says so through ``on_destroyed``)."""

    __slots__ = ("handle", "env", "on_destroyed", "_jobs", "_thread")

    def __init__(self) -> None:
        self.handle: Any = None
        self.env: dict = {}
        self.on_destroyed: Optional[Callable[[], None]] = None
        self._jobs: queue.SimpleQueue = queue.SimpleQueue()
        self._thread = threading.Thread(target=self._loop, name="glossarion-interpreter", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while True:
            job = self._jobs.get()
            if job is None:
                break
            work, done = job
            try:
                done.set_result(work())
            except BaseException as exc:
                done.set_exception(exc)
            del job, work, done
        if self.handle is not None:
            try:
                _api.destroy(self.handle)
            except Exception:
                log.debug("a child interpreter could not be destroyed", exc_info=True)
            if self.on_destroyed is not None:
                self.on_destroyed()

    def start(self, work: Callable[[], Any]) -> concurrent.futures.Future:
        done: concurrent.futures.Future = concurrent.futures.Future()
        self._jobs.put((work, done))
        return done

    def do(self, work: Callable[[], Any]) -> Any:
        return self.start(work).result()

    def finish(self, wait: float = 0.0) -> None:
        self._jobs.put(None)
        if wait:
            self._thread.join(wait)


class _Watched:
    """A ``should_stop`` and the file that tells the children about it."""

    __slots__ = ("function", "path", "calls", "is_set")

    def __init__(self, function: Callable[[], Any], path: str) -> None:
        self.function = function
        self.path = path
        self.calls = 0
        self.is_set = False


_open_pools: "weakref.WeakSet" = weakref.WeakSet()


@atexit.register
def _close_open_pools() -> None:
    """Python's own shutdown does not expect a child that is still there."""
    for pool in list(_open_pools):
        pool.close()


class InterpreterPool:
    """Up to ``size`` child interpreters, started when work needs them, reused until ``close``.

    ``preload``: modules each child imports when it starts (their cost then falls on the first batch of each child,
    and a child that cannot load them is found out before any work is given to it).
    ``after_call``: a ``(module, function)`` a child calls after each result left it (a worker's caches).
    """

    def __init__(self, size: int, *, preload: Sequence[str] = (), after_call: Optional[tuple] = None,
                 work_dir: Optional[str] = None) -> None:
        self.size = max(1, int(size))
        self._preload = tuple(preload)
        self._after = tuple(after_call) if after_call else None
        self._parent_dir = work_dir
        self._dir: Optional[str] = None
        self._lock = threading.Condition()
        self._idle: list = []
        self._live = 0  # started and not destroyed, busy or idle
        self._starting = 0
        self._busy = 0
        self._closed = False
        self._counter = itertools.count(1)
        self._watched: dict = {}
        self._watcher: Optional[threading.Thread] = None
        self._executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
        self.started = 0
        self.destroyed = 0
        self.calls = 0
        _open_pools.add(self)

    # ---- the work -------------------------------------------------------------------------------------------

    def submit(self, module: str, function: str, args: Any = None, *,
               should_stop: Optional[Callable[[], Any]] = None) -> concurrent.futures.Future:
        """``call`` on a thread of this pool's own."""
        with self._lock:
            if self._closed:
                raise InterpreterUnavailable("the interpreter pool is closed")
            if self._executor is None:
                self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.size,
                                                                       thread_name_prefix="glossarion-interpreter")
        return self._executor.submit(self.call, module, function, args, should_stop=should_stop)

    def call(self, module: str, function: str, args: Any = None, *, should_stop: Optional[Callable[[], Any]] = None,
             live_output: bool = False) -> Any:
        """``module.function(args)`` in a free child on this thread; with ``should_stop``,
        ``module.function(args, check)``. Waits for a free one, starting it when there is room. ``live_output``: what it
        prints reaches this interpreter's ``sys.stdout`` from this thread while it runs (it passes through a file in the
        pool's private folder, so only for work that prints no secrets)."""
        child = self._acquire()
        healthy = False
        watched = self._watch(should_stop) if should_stop is not None else None
        call_number = next(self._counter)
        reply_path = os.path.join(self._folder(), f"{call_number}.reply")
        output_path = os.path.join(self._folder(), f"{call_number}.out") if live_output else None
        try:
            current = dict(os.environ)
            changed = {key: value for key, value in current.items() if child.env.get(key) != value}
            gone = [key for key in child.env if key not in current]
            try:
                request = pickle.dumps((changed, gone, module, function, args, watched.path if watched else None,
                                        CHILD_POLL, reply_path, output_path), protocol=pickle.HIGHEST_PROTOCOL)
            except Exception as exc:
                healthy = True  # nothing happened to the child
                raise InterpreterUnavailable(f"the arguments of {function} cannot cross ({exc})") from exc
            child.env = current
            running = child.start(lambda: _run(child.handle, "_gl_call()", {"_gl_request": request}))
            failure = self._wait_passing_output(running, output_path)
            del request
            if failure is not None:
                raise InterpreterUnavailable(f"a child could not run {function}: {failure}")
            try:
                with open(reply_path, "rb") as handle:
                    data = handle.read()
            except OSError as exc:
                raise InterpreterUnavailable(f"{function} left no result ({exc})") from exc
            healthy = True
            with self._lock:
                self.calls += 1
        finally:
            if watched is not None:
                self._unwatch(watched)
            self._release(child, healthy)
        for path in (reply_path, output_path):
            try:
                if path:
                    os.remove(path)
            except OSError:
                pass
        return self._unpack(pickle.loads(data))

    @staticmethod
    def _wait_passing_output(running: concurrent.futures.Future, output_path: Optional[str]) -> Any:
        """The result of ``running``, what the call printed so far written to ``sys.stdout`` while it waits."""
        if output_path is None:
            return running.result()
        offset = 0
        while True:
            try:
                failure = running.result(OUTPUT_POLL)
                break
            except concurrent.futures.TimeoutError:
                offset = _pass_output(output_path, offset)
        _pass_output(output_path, offset, everything=True)
        return failure

    @staticmethod
    def _unpack(reply: tuple) -> Any:
        kind, value, details, printed = reply
        if printed:
            try:
                sys.stdout.write(printed)
            except Exception:  # pragma: no cover - a closed stream must not lose the result
                log.debug("a child's output could not be written", exc_info=True)
        if kind == "value":
            return value
        if kind == "error":
            raise WorkerError(f"{value}\n{details}")
        raise value from _ChildTraceback(details)

    def stop(self) -> None:
        """Tell every child's check, whatever its ``should_stop`` says."""
        with self._lock:
            watched = list(self._watched.values())
        for entry in watched:
            self._set(entry)

    # ---- the children ---------------------------------------------------------------------------------------

    def _acquire(self) -> _Child:
        with self._lock:
            while True:
                if self._closed:
                    raise InterpreterUnavailable("the interpreter pool is closed")
                if self._idle:
                    self._busy += 1
                    return self._idle.pop()  # the one used last: its memory is warm
                if self._live + self._starting < self.size:
                    self._starting += 1
                    break
                self._lock.wait(0.5)
        try:
            child = self._start_child()
        except BaseException as exc:
            with self._lock:
                self._starting -= 1
                others = self._live + self._starting
                self._lock.notify_all()
                if not others or not isinstance(exc, InterpreterUnavailable):
                    raise
                # the ones that did start do the work (a phone that ran out of room for one more)
                self.size = others
                log.info("interpreter pool stays at %d: %s", others, exc)
            return self._acquire()
        with self._lock:
            self._starting -= 1
            self._live += 1
            self._busy += 1
            self.started += 1
            closed = self._closed
        if closed:  # the job ended while this one started
            self._release(child, False)
            raise InterpreterUnavailable("the interpreter pool is closed")
        return child

    def _start_child(self) -> _Child:
        reason = unavailable_reason()
        if reason is not None:
            raise InterpreterUnavailable(reason)
        child = _Child()
        try:
            child.do(lambda: self._prepare(child))
        except BaseException:
            child.finish(RELEASE_WAIT)
            raise
        child.on_destroyed = self._destroyed_one
        return child

    def _prepare(self, child: _Child) -> None:
        """On the child's thread."""
        try:
            child.handle = _create()
        except InterpreterUnavailable:
            raise
        except Exception as exc:
            raise InterpreterUnavailable(f"no child interpreter: {type(exc).__name__}: {exc}") from exc
        child.env = dict(os.environ)
        try:
            failure = _run(child.handle, _child_source())
            if failure is None:
                request = pickle.dumps((child.env, _child_setup, self._preload, self._after))
                failure = _run(child.handle, "_gl_prepare()", {"_gl_request": request})
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
        if failure is not None:
            raise InterpreterUnavailable(f"a child interpreter could not be prepared: {failure}")

    def _release(self, child: _Child, healthy: bool) -> None:
        with self._lock:
            self._busy -= 1
            keep = healthy and not self._closed
            if keep:
                self._idle.append(child)
            finished = self._closed and not self._busy and not self._starting
            self._lock.notify_all()
        if not keep:
            child.finish(RELEASE_WAIT)  # a call that failed outside its worker: this child is not trusted again
        if finished:
            self._remove_folder()

    def _destroyed_one(self) -> None:
        with self._lock:
            self._live -= 1
            self.destroyed += 1
            self._lock.notify_all()

    def live(self) -> int:
        """Children that exist now, busy, idle or starting (until their interpreter is destroyed)."""
        with self._lock:
            return self._live + self._starting

    def close(self, *, stop_running: bool = True) -> None:
        """Release what is idle now. A child still in a batch (left behind by a Stop, or by a job that raised) is told
        to stop, unless ``stop_running`` is False because its result still counts, and is released when it ends; this
        does not wait for that."""
        if stop_running:
            self.stop()
        with self._lock:
            self._closed = True
            idle, self._idle = self._idle, []
            executor, self._executor = self._executor, None
            finished = not self._busy and not self._starting
            self._lock.notify_all()
        for child in idle:
            child.finish()
        for child in idle:
            child.finish(RELEASE_WAIT)
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
        if finished:
            self._remove_folder()

    def __enter__(self) -> "InterpreterPool":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    # ---- Stop -----------------------------------------------------------------------------------------------

    def _watch(self, should_stop: Callable[[], Any]) -> _Watched:
        with self._lock:
            entry = self._watched.get(should_stop)
            if entry is None:
                entry = _Watched(should_stop, os.path.join(self._folder(), f"{next(self._counter)}.stop"))
                self._watched[should_stop] = entry
            entry.calls += 1
            if self._closed:
                self._set(entry)
            elif self._watcher is None:
                self._watcher = threading.Thread(target=self._watch_loop, name="glossarion-interpreter-stop", daemon=True)
                self._watcher.start()
            self._lock.notify_all()
        return entry

    def _unwatch(self, entry: _Watched) -> None:
        with self._lock:
            entry.calls -= 1

    def _watch_loop(self) -> None:
        """Asks only while a call that uses the answer runs: nothing wakes while the pool is idle."""
        def asked() -> list:
            return [entry for entry in self._watched.values() if entry.calls > 0 and not entry.is_set]

        while True:
            with self._lock:
                while not self._closed and not asked():
                    self._lock.wait()
                if not self._closed:
                    self._lock.wait(STOP_POLL)
                if self._closed:
                    self._watcher = None
                    return
                entries = asked()
            for entry in entries:
                try:
                    if entry.function():
                        self._set(entry)
                except Exception:
                    log.debug("a stop check raised", exc_info=True)

    def _set(self, entry: _Watched) -> None:
        entry.is_set = True
        try:
            with open(entry.path, "wb"):
                pass
        except OSError:  # the folder went with a closed pool: nothing is left to tell
            log.debug("the stop file could not be written", exc_info=True)

    # ---- the private folder ---------------------------------------------------------------------------------

    def _folder(self) -> str:
        with self._lock:
            if self._dir is None:
                self._dir = tempfile.mkdtemp(prefix="glossarion-interp-", dir=self._parent_dir)
            return self._dir

    def _remove_folder(self) -> None:
        with self._lock:
            folder, self._dir = self._dir, None
        if folder:
            shutil.rmtree(folder, ignore_errors=True)


def _pass_output(path: str, offset: int, *, everything: bool = False) -> int:
    """The whole lines written to ``path`` after ``offset`` go to ``sys.stdout``; the new offset."""
    try:
        with open(path, "rb") as handle:
            handle.seek(offset)
            data = handle.read()
        if not everything:
            data = data[:data.rfind(b"\n") + 1]
        if data:
            sys.stdout.write(data.decode("utf-8", "replace"))
    except (OSError, ValueError):  # no output yet, or a closed stream: not worth a result
        log.debug("a child's output could not be passed on", exc_info=True)
        return offset
    return offset + len(data)


def last_line(text: Any, limit: int = 160) -> str:
    """A traceback's last line says what happened."""
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    line = lines[-1] if lines else ""
    return line if len(line) <= limit else line[:limit - 1] + "…"


class JobChildren:
    """The children of one job's work, decided once: ``pool`` makes them when work first asks (work that never comes
    costs nothing), by the memory rule. When there is not enough free memory, or a child cannot start or do its work
    (``fall_back``), the job says so once and ``pool`` gives None from then on: the caller runs that work on a thread and
    all the next too. No second try at something that failed.

    ``what`` names the work in its two log lines ("Scanning"). ``preload`` and ``after_call`` are the pool's."""

    def __init__(self, log_line: Callable[[Any], Any], what: str, *, preload: Sequence[str] = (),
                 after_call: Optional[tuple] = None) -> None:
        self._log = log_line
        self._what = what
        self._preload = tuple(preload)
        self._after = after_call
        self._lock = threading.Lock()
        self._pool: Optional[InterpreterPool] = None
        self._said = False
        self.gone = False  # on threads for the rest of this job

    def pool(self, workers: int) -> Optional[InterpreterPool]:
        """The pool (the first call makes it); None on the thread path."""
        with self._lock:
            if self.gone or self._pool is not None:
                return self._pool
            reason = unavailable_reason()
            total, free = memory_now()
            size = size_for_memory(workers, free, total)
            if reason is None and size < 1:
                reason = f"only {int((free or total) / (1024 * 1024))} MB of memory is free"
            if reason is None:
                self._pool = InterpreterPool(size, preload=self._preload, after_call=self._after)
                if not self._said:
                    self._said = True
                    self._say(f"   ⚡ {self._what} in {size} separate Python interpreter{'s' if size != 1 else ''}")
                return self._pool
        self.fall_back(reason)
        return None

    def fall_back(self, reason: str) -> None:
        with self._lock:
            told, self.gone = self.gone, True
            pool, self._pool = self._pool, None
        if pool is not None:
            pool.close(stop_running=False)  # the batches in the other children are good: their results count
        if not told:
            self._say(f"   🧵 {self._what} on threads: separate interpreters are not available here ({last_line(reason)})")

    def _say(self, line: str) -> None:
        try:
            self._log(line)
        except Exception:  # pragma: no cover - a log line is never worth the work
            pass

    def release(self) -> None:
        """The children go now, and the next work starts new ones (work with long gaps between its calls)."""
        with self._lock:
            pool, self._pool = self._pool, None
        if pool is not None:
            pool.close()

    def close(self) -> None:
        """The job ended: the children are released (one a Stop left in a batch, when that batch ends)."""
        with self._lock:
            pool, self._pool = self._pool, None
            self.gone = True
        if pool is not None:
            pool.close()


def memory_now() -> tuple:
    """``(total, available)`` bytes, as the on-device model's admission reads them."""
    try:
        from glossarion_mobile.services.local_llm.service import memory_info

        return memory_info()
    except Exception:
        log.debug("memory_info is not available", exc_info=True)
        return 0, None
