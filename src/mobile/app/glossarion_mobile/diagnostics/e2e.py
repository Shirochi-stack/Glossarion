"""Offline end-to-end test of the mobile job stack (U3 MVP exit; plan §9 "Host" and "Device").

Drives the REAL code paths of the app, with only the model replaced by the fake
OpenAI server on 127.0.0.1 (``fake_llm_server``):

    ChatRuns / JobsAdapter (chat)  or  JobSpec (translate / compile)
      -> services.jobs.JobService (gl-job thread, job_runner.job_process_state, JOB_LOCK,
         MobileConfigStore flush + snapshot, ProgressWatcher, request stream, checkpoints)
      -> job_kinds adapters -> headless_owner.HeadlessOwner
      -> shared pipeline (translation_pipeline, text_jobs, TransateKRtoEN,
         extract_glossary_from_epub, epub_converter)
      -> unified_api_client -> openai -> httpx -> fake server
    finished chat runs -> direct_text_store.ChatStore.finish_run (desktop direct_text_chats.json v2)

on the 12-chapter Korean self-test EPUB (``assets/selftest``). The backend reaches the
fake server through its ordinary custom-endpoint route: the config snapshot holds
``use_custom_openai_endpoint`` / ``openai_base_url`` (exported as
``USE_CUSTOM_OPENAI_ENDPOINT=1`` / ``OPENAI_CUSTOM_BASE_URL``) and a dummy key; the
process runs with the mobile contract (``GLOSSARION_MOBILE=1``,
``GLOSSARION_NO_PROCESSES=1``) set by ``runtime_bootstrap``.

Scenarios (one self-test check each, in this order):

1. ``e2e_chat_balanced_glossary``: a chat attachment send (``direct_text`` job) with the
   glossary mode the desktop welcome writes for "Balanced"; the run stops at the glossary
   approval question, the approval card answers Yes, every chapter is translated with the
   glossary applied, request cards stream while it runs, the chat folder holds the
   compiled EPUB (the fake marker in every chapter, no Hangul left), a complete
   ``translation_progress.json`` and the glossary, and ``direct_text_chats.json`` is the
   desktop v2 format (the shared ChatStore reads it back).
2. ``e2e_translate_glossary_off``: a ``translate`` job with the welcome's "Off" choice:
   no glossary request at all, compiled EPUB complete; then a ``compile_epub`` job
   recompiles the output folder, and the workspace opens in the Library (Completed shelf,
   12/12, its raw source found through the raw-inputs registry the job's set-up wrote),
   the Book page, the Chapters tab (12 completed rows) and the Reader (every translated
   chapter carries the fake marker, the Original side the Korean source; built as the
   mobile page) through ``diagnostics.library_check`` (U5).
3. ``e2e_glossary_edit_qa_pdf`` (U6): an ``extract_glossary`` job (the Glossaries "Extract
   glossary" spec) writes the book glossary; the Glossary Manager's document
   (``services.glossary.GlossaryService`` over ``glossary_document``) changes one translated
   name and saves (the desktop backup first); a ``translate`` job then sends that edited
   entry in every chapter prompt (never the old name); a ``qa_scan`` job (quick scan through
   ``qa_scan_runtime.run_qa_scan_path``, threads only) reports every chapter; and the Book
   page's "Compile PDF" (``LibraryService.compile_spec``: the EPUB compile with "Create PDF
   after EPUB") writes a PDF through the PyMuPDF shim with one page per chapter at least and
   an outline entry per chapter.
4. ``e2e_graceful_stop_resume``: Stop after the first chapter response; the job ends
   gracefully within 30 s with its progress saved, and Resume translates exactly the
   chapters that were missing.
5. ``e2e_force_stop_kill_resume``: the model stops answering; Stop twice (graceful, then
   force) must end the job within 30 s anyway. The ``active.state`` checkpoint taken
   before the stop is then handed to a fresh JobService (an app killed and relaunched):
   it is adopted as an Interrupted job, ``recover()`` restores its progress rows, and
   Resume finishes only the missing chapters.
6. ``e2e_process_hygiene``: after every job ``os.environ``, ``sys.argv``,
   ``sys.stdout``/``sys.stderr``, the cwd, the ``large_env`` store and the
   ``UnifiedClient`` key pools are what they were before it; Glossarion code asked for
   no process (tripwire on the spawn APIs, as in ``tools/host_smoke.py``; a stdlib
   helper that handles the refusal itself, e.g. Windows ``platform.platform()`` on
   Python <= 3.11, is reported as a warning); nothing was written outside the app's
   writable dirs (``sys.addaudithook``); no non-loopback connection was attempted (a
   failure in an isolated process such as the host CLI, a warning inside the running
   app, whose other features may use the network); the backend read the sandbox
   config.json, and the app's own settings, chat history and jobs state are unchanged.

Everything runs in a sandbox (``<temp>/glossarion-e2e/<stamp>``): its own config.json
(``CONFIG_FILE`` points there while the test runs), chat history, Output, Library, Inbox
and jobs folders, so on a device the user's chats, settings and jobs are never touched
(only the backend's API payload dumps land in the app's usual ``Payloads`` folder).
The sandbox is removed after a passing run and kept after a failure (the two newest
failed ones survive the next run).

Entry points: the ``e2e`` self-test suite (Diagnostics "Run end-to-end test", deep link
``glossarion://app/__selftest__?suite=e2e``, CI ``ci/android_smoke.sh``) and, on the host,
``python -m glossarion_mobile.diagnostics.e2e`` (``tests_host/test_e2e_offline.py``, CI
prepare). Importing this module imports no backend module. Python 3.10 compatible.
"""

from __future__ import annotations

import asyncio
import errno
import json
import os
import re
import shutil
import socket
import sys
import threading
import time
import traceback
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

__all__ = [
    "CHAPTERS",
    "SCENARIOS",
    "E2EFailure",
    "E2ESession",
    "ProcessGuards",
    "WriteAudit",
    "diff_process_state",
    "epub_chapter_report",
    "main",
    "process_state",
]

CHAPTERS = 12  # assets/selftest/MANIFEST.toml
JOB_TIMEOUT = 600.0  # one job, seconds (an x86_64 emulator is far slower than a host)
STOP_DEADLINE = 30.0  # a stopped job must have ended this long after Stop
ISOLATED_ENV = "GLOSSARION_E2E_ISOLATED"  # set by main(): nothing else runs in this process
ENOTSUP = getattr(errno, "ENOTSUP", getattr(errno, "EOPNOTSUPP", 95))
_HANGUL = re.compile("[가-힣]")
_TAG = re.compile(r"<[^>]+>")


class E2EFailure(AssertionError):
    """A scenario check failed (message is the reason)."""


def _seed_langdetect(seed: Any) -> Any:
    """Set ``langdetect.DetectorFactory.seed`` (None = random); return the previous value."""
    try:
        from langdetect import DetectorFactory
    except Exception:
        return None
    previous = getattr(DetectorFactory, "seed", None)
    DetectorFactory.seed = seed
    return previous


def _check(condition: Any, message: str) -> None:
    if not condition:
        raise E2EFailure(message)


def _norm(path: Any) -> str:
    return os.path.normcase(os.path.abspath(os.fspath(path)))


def _under(path: str, roots: Any) -> bool:
    return any(path == root or path.startswith(root.rstrip("\\/") + os.sep) for root in roots)


# ---------------------------------------------------------------------------
# Process state (what job_runner.job_process_state must restore)
# ---------------------------------------------------------------------------


def _key_pool_state() -> Optional[dict]:
    module = sys.modules.get("unified_api_client")
    cls = getattr(module, "UnifiedClient", None) if module is not None else None
    if cls is None:
        return None
    try:
        from job_runner import _is_pool_state_attr  # the shared rule of isolated_key_pools()
    except Exception:
        return None
    return {name: id(value) for name, value in vars(cls).items() if _is_pool_state_attr(name)}


def process_state() -> dict:
    """Process-global state a job may touch (compare with ``diff_process_state``)."""
    large = sys.modules.get("large_env")
    store = getattr(large, "_store", None)
    try:
        cwd = os.getcwd()
    except OSError:
        cwd = None
    return {
        "env": dict(os.environ),
        "argv": list(sys.argv),
        "argv_id": id(sys.argv),
        "stdout": id(sys.stdout),
        "stderr": id(sys.stderr),
        "cwd": cwd,
        "large_env": dict(store) if isinstance(store, dict) else None,
        "key_pools": _key_pool_state(),
    }


def diff_process_state(before: dict, after: dict) -> list:
    """Human-readable differences (empty when the job left the process as it found it)."""
    problems: list = []
    env_before, env_after = before["env"], after["env"]
    added = sorted(set(env_after) - set(env_before))
    removed = sorted(set(env_before) - set(env_after))
    changed = sorted(k for k in set(env_before) & set(env_after) if env_before[k] != env_after[k])
    if added:
        problems.append(f"os.environ gained {added[:12]}")
    if removed:
        problems.append(f"os.environ lost {removed[:12]}")
    if changed:
        problems.append(f"os.environ changed {changed[:12]}")
    if before["argv"] != after["argv"] or before["argv_id"] != after["argv_id"]:
        problems.append(f"sys.argv is {after['argv'][:4]} (was {before['argv'][:4]})")
    if before["stdout"] != after["stdout"]:
        problems.append("sys.stdout was not restored")
    if before["stderr"] != after["stderr"]:
        problems.append("sys.stderr was not restored")
    if before["cwd"] != after["cwd"]:
        problems.append(f"cwd is {after['cwd']!r} (was {before['cwd']!r})")
    if before["large_env"] is not None and before["large_env"] != after["large_env"]:
        problems.append("the large_env store was not restored")
    if before["key_pools"] is not None and before["key_pools"] != after["key_pools"]:
        names = sorted(k for k in set(before["key_pools"]) | set(after["key_pools"] or {})
                       if before["key_pools"].get(k) != (after["key_pools"] or {}).get(k))
        problems.append(f"UnifiedClient key pools changed: {names[:8]}")
    return problems


# ---------------------------------------------------------------------------
# Guards: process tripwire + loopback-only network, and the write audit
# ---------------------------------------------------------------------------


#: Frames skipped when naming who asked for a process / socket / write (this module and the
#: stdlib layers that implement the API).
_IMPL_FILES = frozenset({"e2e.py", "subprocess.py", "os.py", "socket.py", "pathlib.py", "shutil.py", "tempfile.py",
                         "threading.py", "process.py", "pool.py", "context.py", "popen_spawn_win32.py",
                         "popen_fork.py", "_base.py", "thread.py", "codecs.py", "_bootstrap_external.py"})


def _caller_frames() -> list:
    return [f for f in traceback.extract_stack()[:-2] if os.path.basename(f.filename) not in _IMPL_FILES]


def _stack_summary(limit: int = 12) -> list:
    """The innermost ``limit`` frames that are not this module or a stdlib layer of the API, innermost last."""
    return [f"{os.path.basename(f.filename)}:{f.lineno} {f.name}" for f in _caller_frames()[-limit:]]


def _stdlib_dirs() -> tuple:
    import sysconfig

    dirs = set()
    for key in ("stdlib", "platstdlib"):
        try:
            dirs.add(_norm(sysconfig.get_paths()[key]))
        except (KeyError, OSError, ValueError):
            pass
    return tuple(dirs)


def _in_stdlib(filename: str) -> bool:
    """A frame of the Python standard library itself (not site-packages, not the app or backend)."""
    path = _norm(filename)
    return "site-packages" not in path.split(os.sep) and _under(path, _stdlib_dirs())


class ProcessGuards:
    """Refuse process spawning (``OSError(ENOTSUP)``, as on iOS) and non-loopback sockets.

    The same APIs as ``tools/host_smoke.py``'s tripwire: ``subprocess.Popen``,
    ``ProcessPoolExecutor``, multiprocessing ``Process.start`` / ``Pool``, ``os.fork``,
    ``os.system``, ``os.posix_spawn*``, ``os.spawn*``, ``os.exec*``, ``os.popen``,
    ``os.startfile``. Every attempt is recorded with its thread and stack.
    """

    OS_FUNCTIONS = (
        "fork", "forkpty", "posix_spawn", "posix_spawnp", "system", "popen", "startfile",
        "spawnl", "spawnle", "spawnlp", "spawnlpe", "spawnv", "spawnve", "spawnvp", "spawnvpe",
        "execl", "execle", "execlp", "execlpe", "execv", "execve", "execvp", "execvpe",
    )

    def __init__(self) -> None:
        self.spawns: list = []
        self.network: list = []
        self._lock = threading.Lock()
        self._originals: list = []

    def _record(self, bucket: list, **event: Any) -> None:
        frames = _caller_frames()
        caller = frames[-1] if frames else None
        event.update(
            thread=threading.current_thread().name,
            stack=[f"{os.path.basename(f.filename)}:{f.lineno} {f.name}" for f in frames[-12:]],
            # A stdlib helper asking (Windows platform.platform() runs "ver" on Python <= 3.11 and
            # falls back on OSError) is reported, not failed: what counts is Glossarion's own code.
            stdlib=bool(caller is not None and _in_stdlib(caller.filename)),
        )
        with self._lock:
            bucket.append(event)

    def _spawn_error(self, api: str) -> OSError:
        self._record(self.spawns, api=api)
        return OSError(ENOTSUP, f"Glossarion E2E: {api} is not available on mobile (process tripwire)")

    def _patch(self, owner: Any, name: str, replacement: Callable) -> None:
        original = getattr(owner, name, None)
        if original is None:
            return
        setattr(owner, name, replacement)
        self._originals.append((owner, name, original))

    def _refuse(self, api: str) -> Callable:
        guards = self

        def refused(*_args: Any, **_kwargs: Any) -> Any:
            raise guards._spawn_error(api)

        return refused

    def install(self) -> None:
        import concurrent.futures.process as cf_process
        import multiprocessing.pool as mp_pool
        import multiprocessing.process as mp_process
        import subprocess

        self._patch(cf_process.ProcessPoolExecutor, "__init__", self._refuse("ProcessPoolExecutor"))
        self._patch(mp_process.BaseProcess, "start", self._refuse("multiprocessing Process.start"))
        guards = self

        def pool_init(pool: Any, *_args: Any, **_kwargs: Any) -> None:
            pool._pool = []  # Pool.__del__ reads these
            pool._state = mp_pool.INIT
            raise guards._spawn_error("multiprocessing.Pool")

        self._patch(mp_pool.Pool, "__init__", pool_init)
        self._patch(subprocess.Popen, "__init__", self._refuse("subprocess.Popen"))
        for name in self.OS_FUNCTIONS:
            if callable(getattr(os, name, None)):
                self._patch(os, name, self._refuse(f"os.{name}"))
        self._install_network_guard()

    @staticmethod
    def _is_local(host: Any) -> bool:
        if host is None:
            return True
        if isinstance(host, bytes):
            host = host.decode("ascii", "replace")
        host = str(host).strip("[]").lower()
        return host in ("", "localhost", "::1", "0.0.0.0", "::") or host.startswith("127.") or host.endswith(".localhost")

    def _install_network_guard(self) -> None:
        guards = self
        original_connect = socket.socket.connect
        original_connect_ex = socket.socket.connect_ex
        original_getaddrinfo = socket.getaddrinfo

        def check(sock: Any, address: Any, api: str) -> None:
            if getattr(sock, "family", None) in (socket.AF_INET, getattr(socket, "AF_INET6", None)) \
                    and isinstance(address, tuple) and not guards._is_local(address[0]):
                guards._record(guards.network, api=api, target=f"{address[0]}:{address[1] if len(address) > 1 else '?'}")
                raise OSError(errno.ENETUNREACH, f"Glossarion E2E: network access to {address[0]} is blocked (offline test)")

        def connect(sock: Any, address: Any) -> Any:
            check(sock, address, "socket.connect")
            return original_connect(sock, address)

        def connect_ex(sock: Any, address: Any) -> Any:
            check(sock, address, "socket.connect_ex")
            return original_connect_ex(sock, address)

        def getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:
            if not guards._is_local(host) and not _is_ip(host):
                guards._record(guards.network, api="socket.getaddrinfo", target=str(host))
                raise socket.gaierror(getattr(socket, "EAI_NONAME", -2), f"Glossarion E2E: DNS lookup of {host} is blocked")
            return original_getaddrinfo(host, *args, **kwargs)

        self._patch(socket.socket, "connect", connect)
        self._patch(socket.socket, "connect_ex", connect_ex)
        self._patch(socket, "getaddrinfo", getaddrinfo)

    def uninstall(self) -> None:
        while self._originals:
            owner, name, original = self._originals.pop()
            try:
                setattr(owner, name, original)
            except (AttributeError, TypeError):
                pass


def _is_ip(host: Any) -> bool:
    import ipaddress

    if isinstance(host, bytes):
        host = host.decode("ascii", "replace")
    try:
        ipaddress.ip_address(str(host).strip("[]"))
        return True
    except ValueError:
        return False


_AUDIT_LOCK = threading.Lock()
_AUDIT_RECORDER: Optional["WriteAudit"] = None
_AUDIT_INSTALLED = False
_WRITE_FLAGS = (getattr(os, "O_WRONLY", 1) | getattr(os, "O_RDWR", 2) | getattr(os, "O_APPEND", 0)
                | getattr(os, "O_CREAT", 0) | getattr(os, "O_TRUNC", 0))
#: audit event -> indexes of the path arguments that are written / removed / created.
_PATH_EVENTS = {
    "os.mkdir": (0,), "os.rename": (0, 1), "os.remove": (0,), "os.rmdir": (0,), "os.truncate": (0,),
    "os.chmod": (0,), "os.utime": (0,), "os.symlink": (1,), "os.link": (1,), "shutil.copyfile": (1,),
    "shutil.copytree": (1,), "shutil.rmtree": (0,), "shutil.move": (0, 1), "shutil.make_archive": (0,),
}


def _audit_hook(event: str, args: tuple) -> None:
    recorder = _AUDIT_RECORDER
    if recorder is None:
        return
    if event == "open":
        if len(args) < 3:
            return
        path, mode, flags = args[0], args[1], args[2]
        if isinstance(mode, str):
            writing = any(c in mode for c in "wax+")
        else:
            writing = isinstance(flags, int) and bool(flags & _WRITE_FLAGS)
        if writing:
            recorder._seen(event, path)
        return
    indexes = _PATH_EVENTS.get(event)
    if indexes:
        for index in indexes:
            if index < len(args):
                recorder._seen(event, args[index])


class WriteAudit:
    """Records file writes outside ``allowed`` roots through ``sys.addaudithook``.

    Python's audit hooks cannot be removed, so one module-level hook is installed once and
    forwards to the active recorder (none outside a test). Writes into ``__pycache__``
    (bytecode), ``os.devnull`` and the allowed roots are fine; everything else is a violation.
    Native code that writes without Python's ``open`` is not seen.
    """

    def __init__(self, allowed: Any) -> None:
        roots = set()
        for root in allowed:
            if not root:
                continue
            roots.add(_norm(root))
            try:
                roots.add(os.path.normcase(os.path.realpath(os.fspath(root))))
            except OSError:
                pass
        self.allowed = tuple(sorted(roots))
        # Bytecode caching is the interpreter's business: PYTHONPYCACHEPREFIX (set by CI) moves
        # the __pycache__ trees elsewhere, so treat that prefix like a __pycache__ directory.
        prefix = getattr(sys, "pycache_prefix", None)
        self.pycache_prefix = (os.path.normcase(os.path.abspath(prefix)),) if prefix else ()
        self.devnull = os.path.normcase(os.path.abspath(os.devnull))
        self.violations: list = []
        self._lock = threading.Lock()
        self._busy = threading.local()

    def _seen(self, event: str, path: Any) -> None:
        if isinstance(path, int) or path is None or getattr(self._busy, "on", False):
            return
        self._busy.on = True
        try:
            try:
                text = os.fsdecode(path)
            except (TypeError, ValueError):
                return
            full = os.path.normcase(os.path.abspath(text))
            if full == self.devnull or text in (os.devnull, "nul", "NUL"):
                return
            if ("__pycache__" in full.split(os.sep) or _under(full, self.allowed)
                    or (self.pycache_prefix and _under(full, self.pycache_prefix))):
                return
            try:  # same folder via a symlinked prefix (Android /data/data -> /data/user/0)
                if _under(os.path.normcase(os.path.realpath(text)), self.allowed):
                    return
            except (OSError, ValueError):
                pass
            with self._lock:
                if len(self.violations) < 200:
                    self.violations.append({"event": event, "path": text, "thread": threading.current_thread().name,
                                            "stack": _stack_summary(8)})
        finally:
            self._busy.on = False

    def start(self) -> None:
        global _AUDIT_RECORDER, _AUDIT_INSTALLED
        with _AUDIT_LOCK:
            if not _AUDIT_INSTALLED:
                sys.addaudithook(_audit_hook)
                _AUDIT_INSTALLED = True
            _AUDIT_RECORDER = self

    def stop(self) -> None:
        global _AUDIT_RECORDER
        with _AUDIT_LOCK:
            if _AUDIT_RECORDER is self:
                _AUDIT_RECORDER = None


# ---------------------------------------------------------------------------
# Output inspection
# ---------------------------------------------------------------------------


def epub_chapter_report(path: str, marker: str) -> dict:
    """Per chapter document of a compiled EPUB: does it carry ``marker``, is any Hangul left."""
    chapters: dict = {}
    with zipfile.ZipFile(path) as archive:
        for name in sorted(archive.namelist()):
            base = os.path.basename(name).lower()
            if not base.endswith((".xhtml", ".html", ".htm")) or any(k in base for k in ("nav", "toc", "cover")):
                continue
            html = archive.read(name).decode("utf-8", "replace")
            body = html.split("<body", 1)[-1]
            text = _TAG.sub(" ", body)
            chapters[name] = {"marker": marker in text, "hangul": len(_HANGUL.findall(text))}
    return chapters


def _progress_chapters(output_dir: str) -> dict:
    """``{chapter number: status}`` of the chapter rows of ``translation_progress.json``."""
    path = os.path.join(output_dir, "translation_progress.json")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return {}
    out: dict = {}
    for entry in (data.get("chapters") or {}).values():
        if not isinstance(entry, dict):
            continue
        output_file = str(entry.get("output_file") or "")
        if not os.path.basename(output_file).startswith("response_chapter"):
            continue
        try:
            number = int(entry.get("actual_num"))
        except (TypeError, ValueError):
            continue
        status = str(entry.get("status") or "")
        if status == "completed" and not os.path.isfile(os.path.join(output_dir, output_file)):
            status = "completed (file missing)"
        out[number] = status
    return out


def _progress_summary(output_dir: str) -> dict:
    """The Library's own summary of ``translation_progress.json`` (``library_core``)."""
    import library_core

    summary = library_core._read_progress_summary(os.path.join(output_dir, "translation_progress.json"))
    return dict(summary or {})


def _completed(chapters: dict) -> set:
    return {n for n, status in chapters.items() if status == "completed"}


def _chapter_requests(records: list) -> list:
    """Chapter numbers of chapter-content requests (one chapter each), in request order."""
    return [r.chapters[0] for r in records if r.kind == "translation" and len(r.chapters) == 1]


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


@dataclass
class JobOutcome:
    label: str
    job_id: str
    state: str
    secs: float
    stop_mode: Optional[str] = None
    error: Optional[str] = None
    outputs: tuple = ()
    output_dir: Optional[str] = None
    process_diff: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"label": self.label, "job_id": self.job_id, "state": self.state, "secs": round(self.secs, 2),
                "stop_mode": self.stop_mode, "error": self.error, "outputs": list(self.outputs),
                "output_dir": self.output_dir, "process_diff": list(self.process_diff)}


class E2ESession:
    """The sandbox, fake server, guards and services shared by the scenarios of one run."""

    def __init__(self, paths: Any = None, *, root: Optional[str] = None, keep: bool = False,
                 isolated: Optional[bool] = None, job_timeout: float = JOB_TIMEOUT,
                 log: Optional[Callable[[str], Any]] = None) -> None:
        self.paths = paths
        stamp = time.strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}"
        base = Path(root) if root else Path(getattr(paths, "temp", None) or os.environ.get("TMPDIR") or ".") / "glossarion-e2e"
        self.root = base / stamp if not root else base
        self.keep = keep
        self.isolated = bool(os.environ.get(ISOLATED_ENV) == "1") if isolated is None else bool(isolated)
        self.job_timeout = float(job_timeout)
        self._log = log
        self.ready = False
        self.setup_error: Optional[BaseException] = None
        self.failed = False
        self.jobs: list = []
        self.server: Any = None
        self.guards: Optional[ProcessGuards] = None
        self.audit: Optional[WriteAudit] = None
        self.store: Any = None
        self.service: Any = None
        self.files: Any = None
        self._env_saved: dict = {}
        self._original_config_file: Optional[str] = None
        self._backend_root: Optional[str] = None
        self._user_files: dict = {}
        self._closers: list = []
        self.epub: Optional[Path] = None

    # ---- logging -------------------------------------------------------------------------

    def log(self, text: str) -> None:
        """Progress line: the app log, and stderr (logcat ``flet.python`` on Android, CI output)."""
        import logging

        line = f"[e2e] {text}"
        logging.getLogger("glossarion.e2e").info(line)
        sink = self._log
        try:
            if sink is not None:
                sink(line)
            else:
                print(line, file=sys.stderr, flush=True)
        except Exception:
            pass

    # ---- setup / teardown -----------------------------------------------------------------------

    def _dir(self, name: str) -> Path:
        path = self.root / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _set_env(self, key: str, value: str) -> None:
        if key not in self._env_saved:
            self._env_saved[key] = os.environ.get(key)
        os.environ[key] = value

    def _backend_modules(self) -> list:
        """Loaded modules that live in the backend dir (only they bind ``app_paths.CONFIG_FILE``)."""
        root = self._backend_root
        modules = []
        for module in list(sys.modules.values()):
            filename = getattr(module, "__file__", None) if module is not None else None
            if root and filename and _norm(filename).startswith(root + os.sep):
                modules.append(module)
        return modules

    def _swap_config_file(self, old: str, new: str) -> None:
        for module in self._backend_modules():
            try:
                if vars(module).get("CONFIG_FILE") == old:
                    setattr(module, "CONFIG_FILE", new)
            except Exception:
                pass

    def _redirect_config_file(self, sandbox_config: str) -> None:
        """Point ``CONFIG_FILE`` (env, ``app_paths`` and the module copies made by
        ``from app_paths import CONFIG_FILE``, e.g. the glossary extractor's ``--config``) at the
        sandbox for the length of the test, so the user's settings are never read or written."""
        import app_paths

        backend = getattr(self.paths, "backend_dir", None) or os.path.dirname(os.path.abspath(app_paths.__file__))
        self._backend_root = _norm(backend)
        self._original_config_file = app_paths.CONFIG_FILE
        self._user_files = self._user_file_state()
        self._set_env("CONFIG_FILE", sandbox_config)
        self._swap_config_file(self._original_config_file, sandbox_config)

    def _user_file_state(self) -> dict:
        """(size, mtime) of the app's own settings / chat history / jobs state (must stay untouched)."""
        config = self._original_config_file or os.environ.get("CONFIG_FILE") or ""
        folder = os.path.dirname(config) if config else ""
        names = ("config.json", "direct_text_chats.json", "direct_text_chats.mobile.json",
                 os.path.join("jobs", "active.state"), os.path.join("jobs", "history.state"))
        state = {}
        for name in names:
            path = os.path.join(folder, name) if folder else name
            try:
                st = os.stat(path)
                state[path] = (st.st_size, st.st_mtime_ns)
            except OSError:
                state[path] = None
        return state

    def _restore_config_file(self) -> None:
        original = self._original_config_file
        if original is None:
            return
        # Also the backend modules first imported during the test (they bound the sandbox path).
        self._swap_config_file(str(self.root / "config.json"), original)
        self._original_config_file = None

    def _prune_old_sandboxes(self, keep: int = 2) -> None:
        """Failed runs keep their sandbox for inspection; only the newest few survive the next run."""
        try:
            siblings = sorted((p for p in self.root.parent.iterdir() if p.is_dir() and p != self.root),
                              key=lambda p: p.stat().st_mtime)
        except OSError:
            return
        for old in siblings[:-keep] if keep else siblings:
            shutil.rmtree(old, ignore_errors=True)

    def setup(self) -> "E2ESession":
        if self.ready:
            return self
        if self.setup_error is not None:
            raise self.setup_error
        try:
            self._setup()
            self.ready = True
            return self
        except BaseException as exc:
            self.setup_error = exc
            raise

    def _setup(self) -> None:
        from glossarion_mobile.diagnostics import fixtures
        from glossarion_mobile.diagnostics.fake_llm_server import FakeLLMServer

        assets = getattr(self.paths, "assets_dir", None)
        epub = fixtures.find_selftest_epub(assets)
        _check(epub is not None, "the self-test EPUB is missing from assets/selftest (tools/prepare_assets.py)")
        # One backend job at a time per process (job_runner.JOB_LOCK): inside the app a running
        # translation would hold every E2E job back until the timeouts, so refuse to start instead.
        import job_runner

        if not job_runner.JOB_LOCK.acquire(timeout=2.0):
            raise E2EFailure("A job is running in the app; run the end-to-end test after it has finished.")
        job_runner.JOB_LOCK.release()
        self.epub = Path(epub)
        self.root.mkdir(parents=True, exist_ok=True)
        if self.root.parent.name == "glossarion-e2e":
            self._prune_old_sandboxes()
        self.log(f"sandbox {self.root}")
        for name in ("Inbox", "Output", "Library", "jobs", "logs", "runs"):
            self._dir(name)

        # Encryption keys: the app installs them from SecureStorage at start; a bare host
        # process (CLI) gets fresh ones, so config.json writes never fall back to a key file.
        from glossarion_mobile.services import secure_keys

        if secure_keys.current_status() is None:
            status = secure_keys.install_backend_keys(secure_keys.new_api_key(), secure_keys.new_token_key(),
                                                      source="e2e")
            _check(status.installed, f"could not install encryption keys: {status.errors}")

        # Sandbox paths for everything the backend resolves from the environment.
        self._set_env("OUTPUT_DIRECTORY", str(self.root / "Output"))
        self._set_env("GLOSSARION_LIBRARY_DIR", str(self.root / "Library"))
        self._set_env("GLOSSARION_DIRECT_TEXT_HISTORY", str(self.root / "direct_text_chats.json"))
        # The fake server must never go through a proxy (os.environ is case-insensitive on Windows).
        for key in ("NO_PROXY",) if os.name == "nt" else ("NO_PROXY", "no_proxy"):
            self._set_env(key, ",".join(p for p in (os.environ.get(key, ""), "127.0.0.1,localhost") if p))
        self._redirect_config_file(str(self.root / "config.json"))

        self.server = FakeLLMServer().start()
        self._closers.append(self.server.stop)
        self.log(f"fake LLM server on {self.server.url}")

        allowed = [self.root]  # + the app's writable dirs (TMPDIR is paths.temp after bootstrap)
        if self.paths is not None:
            allowed.extend(self.paths.writable_dirs().values())
        self.audit = WriteAudit(allowed)
        self.guards = ProcessGuards()
        self.guards.install()
        self.audit.start()

        from glossarion_mobile.services.files import FileBridge
        from glossarion_mobile.services.jobs import JobService
        from glossarion_mobile.state.config_store import MobileConfigStore

        self.store = MobileConfigStore(self.root / "config.json", debounce=0.05)
        self.store.load()
        self._closers.append(self.store.close)
        self.files = FileBridge(inbox_dir=str(self.root / "Inbox"), library_raw_dir=str(self.root / "Library" / "Raw"))
        self.service = self.new_service(self.root / "jobs")

    def new_service(self, jobs_dir: Path) -> Any:
        """A JobService on ``jobs_dir`` (a second one stands for the relaunched app)."""
        from glossarion_mobile.services.jobs import JobService

        service = JobService(jobs_dir=jobs_dir, logs_dir=self.root / "logs", data_dir=self.root,
                             config_store=self.store, watcher_interval=0.5)
        self._closers.append(service.close)
        return service

    def configure(self, glossary_mode: str, **extra: Any) -> dict:
        """The sandbox config: the fake endpoint, a dummy key and the welcome's glossary choice."""
        from glossarion_mobile.diagnostics.fake_llm_server import FAKE_MODEL
        from glossarion_mobile.ui.screens.welcome_flow import welcome_glossary_updates

        values = {
            "model": FAKE_MODEL,
            "api_key": "sk-glossarion-e2e-dummy",
            "use_custom_openai_endpoint": True,
            "openai_base_url": self.server.url,
            "output_language": "English",
            "delay": 0,  # API call spacing (desktop default 5 s); timing only
        }
        values.update(welcome_glossary_updates(glossary_mode))
        values.update(extra)
        for key in list(self.store.keys()):
            if key not in values:
                self.store.unset(key)
        self.store.set_many(values)
        self.store.flush()
        _check(self.store.save_error is None and not self.store.dirty,
               f"config.json could not be saved: {self.store.save_error}")
        return values

    def import_epub(self, name: str) -> str:
        """The self-test EPUB copied into the sandbox Inbox under ``name`` (FileBridge import)."""
        imported = self.files.import_paths([str(self.epub)], names=[name])
        _check(imported, f"FileBridge did not import {name}")
        return imported[0].path

    def close(self) -> None:
        if self.audit is not None:
            self.audit.stop()
        if self.guards is not None:
            self.guards.uninstall()
        if self.server is not None:
            try:
                self.server.release(abort=True)
            except Exception:
                pass
        for closer in reversed(self._closers):
            try:
                closer()
            except Exception:
                pass
        self._closers = []
        self._restore_config_file()
        for key, value in self._env_saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self._env_saved = {}
        if not self.failed and not self.keep and self.ready:
            shutil.rmtree(self.root, ignore_errors=True)
        elif self.ready or self.root.exists():
            self.log(f"sandbox kept for inspection: {self.root}")

    # ---- job helpers ------------------------------------------------------------------------------

    def wait(self, service: Any, job_id: str, *, timeout: Optional[float] = None) -> Any:
        """Wait until ``job_id`` is terminal (and its service idle); returns its snapshot."""
        deadline = time.monotonic() + (timeout or self.job_timeout)
        while time.monotonic() < deadline:
            snap = service.snapshot(job_id)
            if snap is not None and snap.is_terminal and not service.busy:
                return snap
            time.sleep(0.05)
        snap = service.snapshot(job_id)
        self.abort_job(service, job_id)
        raise E2EFailure(f"job {job_id} did not finish within {timeout or self.job_timeout:.0f} s "
                         f"(state {getattr(snap, 'state', None)}, last line {getattr(snap, 'last_line', '')!r})")

    def abort_job(self, service: Any, job_id: str) -> None:
        """A failed scenario must not leave its job running for the next one: force stop it."""
        try:
            if self.server is not None:
                self.server.release(abort=True)
            service.request_stop(job_id, force=True, reason="E2E scenario failed")
            deadline = time.monotonic() + STOP_DEADLINE
            while time.monotonic() < deadline:
                snap = service.snapshot(job_id)
                if snap is None or snap.is_terminal:
                    break
                time.sleep(0.1)
        except Exception:
            pass

    def run_job(self, service: Any, spec: Any, label: str, *, during: Optional[Callable[[str], Any]] = None,
                timeout: Optional[float] = None) -> JobOutcome:
        """Submit, wait, and record the process-state diff around the job."""
        before = process_state()
        t0 = time.monotonic()
        job_id = service.submit(spec)
        if during is not None:
            try:
                during(job_id)
            except BaseException:
                self.abort_job(service, job_id)
                raise
        snap = self.wait(service, job_id, timeout=timeout)
        return self._outcome(label, job_id, snap, t0, before)

    def _outcome(self, label: str, job_id: str, snap: Any, t0: float, before: dict) -> JobOutcome:
        diff = diff_process_state(before, process_state())
        outcome = JobOutcome(label=label, job_id=job_id, state=snap.state.value, secs=time.monotonic() - t0,
                             stop_mode=snap.stop_mode, error=snap.error, outputs=tuple(snap.outputs),
                             output_dir=snap.output_dir, process_diff=diff)
        self.jobs.append(outcome)
        self.log(f"{label}: {outcome.state} in {outcome.secs:.1f} s" + (f" ({outcome.error})" if outcome.error else ""))
        _check(not diff, f"{label}: the job did not restore the process state: {diff}")
        return outcome

    def expect_quiet_after_stop(self, label: str, since: int, settle: float = 1.5) -> None:
        """A stopped run must not reach the model after its job ended (no retries, no stragglers)."""
        time.sleep(settle)
        late = [r for r in self.server.records(since=since)]
        _check(not late, f"{label}: {len(late)} request(s) reached the model after the stopped job ended: "
                         + "; ".join(f"{r.kind} ch{r.chapters} {r.status}" for r in late[:4]))

    def translate_spec(self, path: str, title: str) -> Any:
        from glossarion_mobile.services.jobs import JobSpec

        return JobSpec(kind="translate", title=title, inputs=(path,), origin={"type": "e2e", "label": "E2E"})

    def _expect_complete_book(self, output_dir: str, label: str, epub_dir: Optional[str] = None) -> dict:
        """12/12 chapters in translation_progress.json and a compiled EPUB tagged in every chapter."""
        from glossarion_mobile.diagnostics.fake_llm_server import FAKE_MARKER

        summary = _progress_summary(output_dir)
        chapters = _progress_chapters(output_dir)
        _check(summary.get("total") == CHAPTERS and summary.get("completed") == CHAPTERS,
               f"{label}: translation_progress.json is not complete: {summary}")
        _check(_completed(chapters) == set(range(1, CHAPTERS + 1)), f"{label}: chapter rows {chapters}")
        folder = epub_dir or output_dir
        epubs = sorted(p for p in Path(folder).glob("*.epub") if p.is_file())
        _check(epubs, f"{label}: no compiled EPUB in {folder}")
        report = epub_chapter_report(str(epubs[0]), FAKE_MARKER)
        _check(len(report) == CHAPTERS, f"{label}: the compiled EPUB has {len(report)} chapter documents, not {CHAPTERS}")
        untagged = [n for n, r in report.items() if not r["marker"]]
        raw = {n: r["hangul"] for n, r in report.items() if r["hangul"]}
        _check(not untagged, f"{label}: chapters without the translation marker: {untagged}")
        _check(not raw, f"{label}: untranslated Hangul left in {raw}")
        counts = {key: summary.get(key) for key in ("total", "completed", "in_progress", "failed")}
        return {"epub": epubs[0].name, "chapters": len(report), "progress": counts}

    # ---- scenario 1: chat attachment, Balanced glossary, approval card ------------------------------------

    def chat_balanced_glossary(self) -> dict:
        from glossarion_mobile.diagnostics.fake_llm_server import FAKE_MARKER
        from glossarion_mobile.state.chat_store_adapter import ChatStoreAdapter, ChatStoreBinding
        from glossarion_mobile.ui.chat.direct_text_rules import DirectTextSettings
        from glossarion_mobile.ui.chat.job_binding import JobsAdapter
        from glossarion_mobile.ui.chat.run_controller import ChatRuns
        from glossarion_mobile.ui.chat.run_request import attachment_record

        self.setup()
        self.configure("balanced")
        server = self.server
        mark = server.mark()
        history = str(self.root / "direct_text_chats.json")
        chats = ChatStoreAdapter(ChatStoreBinding(history_path=history, output_root=str(self.root / "Output")),
                                 history_path=history, save_delay=0.05)
        _check(chats.load(), f"the chat history could not be loaded: {chats.load_error}")
        runs = ChatRuns(chats, JobsAdapter(self.service), temp_dir=str(self.root / "runs"),
                        model_name=lambda: self.store.get("model"))
        runs.attach()
        try:
            cid = chats.new_chat()
            record = attachment_record(self.import_epub(self.epub.name))
            _check(record is not None, "the imported EPUB is not a file")
            before = process_state()
            t0 = time.monotonic()
            run = asyncio.run(runs.send(cid, text="", attachment=record, settings=DirectTextSettings(),
                                        output_mode="text"))
            _check(run.params.get("options", {}).get("force_no_glossary") is False,
                   "an attachment send must not force No Glossary (Attachments Only policy)")
            question: dict = {}
            answered_at = None
            live_cards = 0
            deadline = time.monotonic() + self.job_timeout
            while run.live and time.monotonic() < deadline:
                if run.awaiting_glossary and answered_at is None:
                    question = dict(run.question or {})
                    question["chapter_requests_before"] = len(_chapter_requests(server.records(since=mark)))
                    answered_at = time.time()  # the answer releases the job at once
                    question["entries"] = self._answer_approval(runs, cid, question)
                if run.state == "running" and answered_at is not None:
                    run.stream.drain()  # what the chat view's render tick does
                    streamed = [s for s in (run.stream.segments() or []) if FAKE_MARKER in str(s.get("content") or "")]
                    live_cards = max(live_cards, len(streamed))
                time.sleep(0.05)
            _check(not run.live, f"the chat run did not finish within {self.job_timeout:.0f} s (state {run.state})")
            for thread in list(runs.finish_threads):
                thread.join(60)
            snap = self.wait(self.service, run.job_id)
            outcome = self._outcome("chat (balanced glossary)", run.job_id, snap, t0, before)
            _check(outcome.state == "DONE", f"chat job ended {outcome.state}: {outcome.error}")
            _check(run.state == "done" and run.error is None, f"chat run ended {run.state}: {run.error}")
            _check(runs.caption(cid) is None, f"the composer still shows {runs.caption(cid)!r} (expected Ready)")

            # glossary gate: asked once, before any chapter request, with the generated glossary
            _check(question, "the run never asked for glossary approval")
            _check(question.get("kind") == "direct_text_glossary_approval", f"unexpected question {question.get('kind')}")
            path = str((question.get("data") or {}).get("path") or "")
            _check(path.lower().endswith("_glossary.csv"), f"approval question names {path!r}")
            _check(question["chapter_requests_before"] == 0, "chapters were sent before the glossary was approved")
            _check(question["entries"] == len(server.glossary),
                   f"the approval card counted {question['entries']} entries, the glossary has {len(server.glossary)}")
            records = server.records(since=mark)
            glossary_requests = [r for r in records if r.kind == "glossary"]
            chapter_records = [r for r in records if r.kind == "translation" and len(r.chapters) == 1]
            _check(glossary_requests, "no glossary extraction request reached the model")
            _check(all(r.started >= answered_at - 0.001 for r in chapter_records),
                   "a chapter request started before the approval answer")
            _check(sorted(_chapter_requests(records)) == list(range(1, CHAPTERS + 1)),
                   f"chapter requests {sorted(_chapter_requests(records))}")
            unglossed = [r.chapters[0] for r in chapter_records if not r.glossary_applied]
            _check(not unglossed, f"chapters translated without the approved glossary: {unglossed}")
            _check(all(r.stream for r in chapter_records), "Direct Text requests must stream")

            # streaming request cards
            segments = self.service.request_segments(run.job_id)
            card_count = sum(1 for s in segments if FAKE_MARKER in str(s.get("content") or ""))
            _check(card_count >= CHAPTERS, f"the job's request stream holds {card_count} chapter cards")
            _check(live_cards >= 1, "no request card streamed while the job was running")

            # outputs persisted into the chat folder
            folder = run.output_folder
            _check(folder and os.path.isdir(folder), f"no chat attachment folder ({folder!r})")
            _check(_under(_norm(folder), [_norm(self.root / "Output" / "Direct Text")]),
                   f"attachment folder outside Output/Direct Text: {folder}")
            book = self._expect_complete_book(folder, "chat")
            glossary = os.path.join(folder, "glossary.csv")
            _check(os.path.isfile(glossary), "glossary.csv is missing from the attachment folder")
            text = Path(glossary).read_text(encoding="utf-8-sig")
            missing = [e.translated_name for e in server.glossary if e.translated_name not in text]
            _check(not missing, f"glossary.csv lacks {missing}")
            history_detail = self._expect_desktop_history(history, cid, folder)
            return {
                "secs": round(outcome.secs, 1),
                "requests": {"glossary": len(glossary_requests), "chapters": len(chapter_records), "total": len(records)},
                "approval_entries": question["entries"],
                "live_cards": live_cards,
                "stream_cards": card_count,
                "book": book,
                "history": history_detail,
            }
        finally:
            runs.detach()
            chats.close()

    def _answer_approval(self, runs: Any, cid: str, question: dict) -> int:
        """Answer Yes on the approval card (``cards.GlossaryApprovalCard``, as ChatView builds it);
        without Flet (desktop Python) the card's handler (``ChatRuns.answer_glossary``) directly."""
        path = str((question.get("data") or {}).get("path") or "")
        entries = 0
        try:
            from glossarion_mobile.ui.chat.cards import GlossaryApprovalCard, glossary_preview
        except ImportError:
            GlossaryApprovalCard = None  # noqa: N806
        if GlossaryApprovalCard is not None:
            info = glossary_preview(path)
            entries = int(info.get("entries") or 0)
            card = GlossaryApprovalCard(path=path, info=info, on_answer=lambda ok: runs.answer_glossary(cid, ok))
            card.yes_button.on_click(None)  # the ✓ Yes tap
            _check(card.answered is True, "the approval card did not take the Yes answer")
        else:
            from glossary_usage import parse_glossary_file  # what glossary_preview counts with

            entries = len(parse_glossary_file(path))
            _check(runs.answer_glossary(cid, True), "ChatRuns.answer_glossary(Yes) was refused")
        self.log(f"approval card answered Yes ({entries} entries)")
        return entries

    def _expect_desktop_history(self, history: str, cid: str, folder: str) -> dict:
        """direct_text_chats.json v2 as the desktop writes it; the shared ChatStore reads it back."""
        from glossarion_mobile.diagnostics.fake_llm_server import FAKE_MARKER
        import direct_text_store

        data = json.loads(Path(history).read_text(encoding="utf-8"))
        _check(data.get("version") == 2, f"history version {data.get('version')}")
        allowed = {"id", "title", "messages", "draft", "attachment", "output_folder", "output_folder_name",
                   "next_output_index", "expanded"}
        session = next((s for s in data.get("sessions") or [] if str(s.get("id")) == str(cid)), None)
        _check(session is not None, f"chat {cid} is not in direct_text_chats.json")
        _check(set(session) <= allowed, f"mobile-only keys in the v2 file: {sorted(set(session) - allowed)}")
        messages = session["messages"]
        first = messages[0]
        _check(first[0] == "user_file" and first[1] == self.epub.name and len(first) == 6,
               f"first message is {first[:2]}")
        assistant = [m for m in messages if m and m[0] == "assistant"]
        labels = [str(m[5]) for m in assistant if len(m) > 5]
        _check(len(assistant) >= CHAPTERS + 2, f"{len(assistant)} assistant cards in the history")
        _check("Extraction report" in labels and "Attachment actions" in labels,
               f"report cards missing (labels: {labels[-4:]})")
        inline = [str(m[5]) for m in assistant if len(m) > 1 and m[1] == ""
                  and not (len(m) > 6 and isinstance(m[6], dict) and m[6].get("content_path"))]
        _check(not inline, f"assistant cards without a body or Chat Messages/ file: {inline[:3]}")
        _check(sum(1 for m in assistant if len(m) > 6 and isinstance(m[6], dict) and m[6].get("content_path")) >= CHAPTERS,
               "chapter bodies were not externalised to Chat Messages/")
        # Compare resolved paths: on Android the same folder is reachable as
        # /data/data/<pkg> and /data/user/0/<pkg> (symlink).
        _out = os.path.realpath(str(session.get("output_folder") or ""))
        _expected = os.path.realpath(str(self.root / "Output" / "Direct Text"))
        _check(bool(session.get("output_folder")) and (_out == _expected or _out.startswith(_expected + os.sep)),
               f"chat output folder {session.get('output_folder')!r}")
        sidecar = Path(history).with_name("direct_text_chats.mobile.json")
        _check(sidecar.is_file(), "the mobile sidecar was not written")

        # The desktop's own persistence code (moved verbatim into direct_text_store) reads it back.
        store = direct_text_store.ChatStore(history, output_root=str(self.root / "Output"))
        sessions, _current = store.load_chat_history()
        reread = next((s for s in sessions if str(s.get("id")) == str(cid)), None)
        _check(reread is not None and len(reread["messages"]) == len(messages), "ChatStore read back a different chat")
        requests = [i for i, m in enumerate(reread["messages"]) if m[0] == "assistant" and "Request" in str(m[5])]
        bodies = {i: store.assistant_message_text(reread, i) for i in requests}
        translated = [i for i in requests if FAKE_MARKER in bodies[i]]
        _check(len(translated) >= CHAPTERS, f"only {len(translated)} chapter card bodies hold the translation")
        # The glossary request card was frozen into the chat at the approval gate, before the chapter
        # cards (the dialog's _commit_active_request_phase; ChatStore.commit_request_phase).
        _check(requests and requests[0] not in translated and requests[0] < min(translated),
               "the glossary request card was not committed at the approval gate (before the chapters)")
        return {"messages": len(messages), "assistant_cards": len(assistant), "labels": labels[-3:],
                "gate_card": str(reread["messages"][requests[0]][5])}

    # ---- scenario 2: translate job, glossary Off, then Compile EPUB ------------------------------------------

    def translate_glossary_off(self) -> dict:
        from glossarion_mobile.services.jobs import JobSpec

        self.setup()
        self.configure("off")
        server = self.server
        mark = server.mark()
        path = self.import_epub("e2e-glossary-off.epub")
        outcome = self.run_job(self.service, self.translate_spec(path, "e2e-glossary-off.epub"), "translate (glossary off)")
        _check(outcome.state == "DONE", f"translate job ended {outcome.state}: {outcome.error}")
        records = server.records(since=mark)
        _check(not [r for r in records if r.kind == "glossary"], "glossary mode Off sent a glossary request")
        _check(sorted(_chapter_requests(records)) == list(range(1, CHAPTERS + 1)),
               f"chapter requests {sorted(_chapter_requests(records))}")
        _check(not any(r.glossary_applied for r in records), "a glossary was applied although the mode is Off")
        output_dir = outcome.output_dir
        _check(output_dir and _under(_norm(output_dir), [_norm(self.root / "Output")]),
               f"output folder {output_dir!r} is not under the sandbox Output")
        snap = self.service.snapshot(outcome.job_id)
        _check(snap.progress.total == CHAPTERS and snap.progress.completed == CHAPTERS,
               f"ProgressWatcher reported {snap.progress}")
        _check(any(p.lower().endswith(".epub") for p in outcome.outputs), f"job outputs {outcome.outputs}")
        book = self._expect_complete_book(output_dir, "translate")
        compiled = self.run_job(self.service, JobSpec(kind="compile_epub", title="Compile", params={"folder": output_dir}),
                                "compile EPUB")
        _check(compiled.state == "DONE", f"compile job ended {compiled.state}: {compiled.error}")
        epubs = [p for p in compiled.outputs if p.lower().endswith(".epub") and os.path.isfile(p)]
        _check(epubs, f"compile outputs {compiled.outputs}")
        self._expect_complete_book(output_dir, "compile")
        library = self._expect_library_book(output_dir, path)
        return {"secs": round(outcome.secs, 1), "requests": len(records), "book": book,
                "compile_secs": round(compiled.secs, 1), "compiled": os.path.basename(epubs[0]),
                "library": library}

    def _expect_library_book(self, output_dir: str, raw_path: str) -> dict:
        """The translated + compiled workspace in the Library, Book page, Chapters tab and Reader (U5)."""
        from glossarion_mobile.diagnostics import library_check
        from glossarion_mobile.diagnostics.fake_llm_server import FAKE_MARKER

        try:
            with library_check.isolated_library(self.root, self.store.snapshot()) as service:
                import library_core

                # The translate job's set-up recorded its raw input like the desktop run does
                # (HeadlessOwner._record_library_raw_inputs -> library_core.record_library_raw_inputs).
                registered = [_norm(p) for p in library_core.load_library_raw_inputs()]
                _check(_norm(raw_path) in registered,
                       f"the translate job did not record {os.path.basename(raw_path)} in library_raw_inputs.txt")
                detail = library_check.verify_book(
                    service, output_dir=Path(output_dir), raw_path=Path(raw_path), shelf="completed",
                    completed=CHAPTERS, total=CHAPTERS, statuses={"completed": CHAPTERS}, marker=FAKE_MARKER,
                    translated_chapters=CHAPTERS)
                detail["raw_inputs_registered"] = len(registered)
                return detail
        except library_check.LibraryCheckFailure as exc:
            raise E2EFailure(f"Library / Reader: {exc}") from exc

    # ---- scenario 3: extract glossary -> edit -> save -> translate -> QA quick scan -> PDF (U6) -----------------

    #: The entry the scenario edits (a recurring name of the self-test EPUB) and its new translation.
    EDIT_RAW = "이서연"
    EDIT_NAME = "Seo-yeon Lumen"

    def glossary_edit_qa_pdf(self) -> dict:
        from glossarion_mobile.job_kinds.qa import report_path_for
        from glossarion_mobile.services.glossary import GlossaryService
        from glossarion_mobile.services.jobs import JobSpec
        from glossarion_mobile.services.library import LibraryService

        self.setup()
        self.configure("balanced")
        server = self.server
        name = "e2e-glossary-edit.epub"
        path = self.import_epub(name)
        glossaries = GlossaryService(config=self.store)

        # 1. Extract glossary (Glossaries › Extract glossary)
        mark = server.mark()
        extracted = self.run_job(self.service, glossaries.extract_spec([path]), "extract glossary")
        _check(extracted.state == "DONE", f"glossary job ended {extracted.state}: {extracted.error}")
        records = server.records(since=mark)
        _check([r for r in records if r.kind == "glossary"], "no glossary extraction request reached the model")
        _check(not [r for r in records if r.kind == "translation"], "the glossary job sent translation requests")
        csvs = [p for p in extracted.outputs if p.lower().endswith(".csv") and os.path.isfile(p)]
        _check(csvs, f"the glossary job listed no glossary file: {extracted.outputs}")
        glossary = csvs[0]
        _check(_under(_norm(glossary), [_norm(self.root / "Output")]), f"glossary outside the sandbox: {glossary}")

        # 2. Edit one entry in the Glossary Manager's document and save it
        doc = glossaries.open_document(glossary, source_path=path)
        specs = glossaries.row_specs(doc)
        target = next((s for s in specs if str(s.entry.get("raw_name") or "") == self.EDIT_RAW), None)
        _check(target is not None, f"{self.EDIT_RAW} is not in the extracted glossary ({len(specs)} rows)")
        old_name = str(target.entry.get("translated_name") or "")
        glossaries.update_entry(doc, target.ref, {"translated_name": self.EDIT_NAME})
        _check(glossaries.translated_changes(doc) == [(old_name, self.EDIT_NAME)],
               f"translated changes {glossaries.translated_changes(doc)}")
        saved = glossaries.save_edits(doc, update_outputs=False)
        _check(saved.get("saved"), f"the edited glossary was not saved: {saved} {glossaries.log_lines[-3:]}")
        backups = sorted(Path(glossary).parent.glob("Backups/*_before_save_*.json"))
        _check(backups, "no 'before_save' backup was written (glossary_files.create_glossary_backup)")
        reread = glossaries.open_document(glossary)
        names = {str(s.entry.get("raw_name")): str(s.entry.get("translated_name"))
                 for s in glossaries.row_specs(reread)}
        _check(names.get(self.EDIT_RAW) == self.EDIT_NAME, f"re-read glossary has {names.get(self.EDIT_RAW)!r}")

        # 3. Translate with the edited glossary
        mark = server.mark()
        outcome = self.run_job(self.service, self.translate_spec(path, name), "translate (edited glossary)")
        _check(outcome.state == "DONE", f"translate job ended {outcome.state}: {outcome.error}")
        records = server.records(since=mark)
        chapter_records = [r for r in records if r.kind == "translation" and len(r.chapters) == 1]
        _check(sorted(_chapter_requests(records)) == list(range(1, CHAPTERS + 1)),
               f"chapter requests {sorted(_chapter_requests(records))}")
        mentions = [r for r in chapter_records if self.EDIT_RAW in str(r.preview) or
                    any(self.EDIT_RAW in line for line in r.glossary_lines)]
        edited = [r.chapters[0] for r in chapter_records
                  if any(self.EDIT_RAW in line and self.EDIT_NAME in line for line in r.glossary_lines)]
        stale = [r.chapters[0] for r in chapter_records
                 if any(self.EDIT_RAW in line and old_name and old_name in line and self.EDIT_NAME not in line
                        for line in r.glossary_lines)]
        _check(edited, "no chapter prompt carried the edited glossary entry "
                       f"({self.EDIT_RAW} = {self.EDIT_NAME}); {len(mentions)} prompt(s) name {self.EDIT_RAW}")
        _check(not stale, f"chapter prompts still carried the old name {old_name!r}: {stale}")
        output_dir = outcome.output_dir
        _check(output_dir and _under(_norm(output_dir), [_norm(self.root / "Output")]),
               f"output folder {output_dir!r} is not under the sandbox Output")
        book = self._expect_complete_book(output_dir, "translate (edited glossary)")

        # 4. QA quick scan (Tools › QA Scanner). scan_html_folder never seeds langdetect, so
        # its verdicts on short text vary between runs (DISCREPANCIES U6 item 8); about 1 run
        # in 10 then flags TOC.txt / translated_headers.txt, which hits a recorded desktop bug
        # (update_new_format_progress: UnboundLocalError on 'hashlib'). The check pins the
        # seed for the scan, like the desktop QA parity tests, so it is deterministic.
        seed_before = _seed_langdetect(0)
        try:
            qa = self.run_job(self.service, JobSpec(
                kind="qa_scan", title=os.path.basename(output_dir), inputs=(output_dir,),
                params={"mode": "quick-scan", "targets": [{"folder": output_dir, "source": path}]},
                origin={"type": "e2e", "label": "E2E"}), "QA quick scan")
        finally:
            _seed_langdetect(seed_before)
        _check(qa.state == "DONE", f"QA job ended {qa.state}: {qa.error}")
        report = report_path_for(output_dir)
        _check(os.path.isfile(report), f"no QA report at {report}")
        rows = json.loads(Path(report).with_name("validation_results.json").read_text(encoding="utf-8"))
        scanned = sorted(str(r.get("filename") or "") for r in rows if str(r.get("filename") or "").startswith("response_"))
        _check(scanned == [f"response_chapter{n:04d}.html" for n in range(1, CHAPTERS + 1)],
               f"the QA report lists the chapter files {scanned}")
        _check(any(os.path.normcase(p) == os.path.normcase(report) for p in qa.outputs), f"QA outputs {qa.outputs}")

        # 5. Compile PDF (Book page › Compile PDF on an EPUB workspace: the EPUB compile + PDF, shim)
        spec = LibraryService(config=self.store).compile_spec(
            {"name": os.path.basename(output_dir), "output_folder": output_dir}, "compile_pdf")
        _check(spec.kind == "compile_epub" and spec.params.get("config_overrides") == {"enable_pdf_output": True},
               f"Compile PDF of an EPUB workspace planned {spec.kind} {spec.params}")
        before = {p: p.stat().st_mtime for p in Path(output_dir).glob("*.pdf")}
        compiled = self.run_job(self.service, spec, "compile PDF (shim)")
        _check(compiled.state == "DONE", f"compile job ended {compiled.state}: {compiled.error}")
        pdfs = [p for p in Path(output_dir).glob("*.pdf") if before.get(p) != p.stat().st_mtime]
        _check(pdfs, f"no PDF was written to {output_dir} (outputs {compiled.outputs})")
        listed = {os.path.normcase(os.path.abspath(p)) for p in compiled.outputs}
        _check(all(os.path.normcase(os.path.abspath(str(p))) in listed for p in pdfs),
               f"the compile job's outputs {compiled.outputs} miss the PDF(s) {[p.name for p in pdfs]}")
        import fitz

        pdf = max(pdfs, key=lambda p: p.stat().st_size)
        with fitz.open(str(pdf)) as document:
            pages = document.page_count
            toc = document.get_toc()
            text = "".join(document[i].get_text() for i in range(min(pages, 40)))
            producer = str((document.metadata or {}).get("producer") or "")
        from glossarion_mobile.diagnostics.fake_llm_server import FAKE_MARKER

        _check(pages >= CHAPTERS, f"the PDF has {pages} page(s) for {CHAPTERS} chapters")
        _check(len(toc) >= CHAPTERS, f"the PDF outline has {len(toc)} entr(ies) for {CHAPTERS} chapters")
        _check(FAKE_MARKER in text, "the PDF text lacks the translation marker")
        self._expect_complete_book(output_dir, "compile PDF")
        return {"glossary": os.path.basename(glossary), "edited": {self.EDIT_RAW: [old_name, self.EDIT_NAME]},
                "backups": len(backups), "edited_prompts": len(edited), "book": book,
                "qa": {"files": len(rows), "chapters": len(scanned), "issues": sum(len(r.get("issues") or []) for r in rows),
                       "secs": round(qa.secs, 1)},
                "pdf": {"name": pdf.name, "pages": pages, "outline": len(toc), "bytes": pdf.stat().st_size,
                        "producer": producer, "secs": round(compiled.secs, 1)}}

    # ---- scenario 4: graceful stop, then Resume ----------------------------------------------------------------

    def graceful_stop_resume(self) -> dict:
        self.setup()
        self.configure("off")
        server = self.server
        server.set_delay("translation", 0.4)  # keep a few requests in flight when Stop arrives
        stop: dict = {}

        def first_chapter_done(record: Any) -> None:
            if record.kind == "translation" and len(record.chapters) == 1 and "at" not in stop:
                stop["at"] = time.monotonic()
                stop["chapter"] = record.chapters[0]
                stop["mode"] = self.service.request_stop()

        server.on_response.append(first_chapter_done)
        try:
            mark = server.mark()
            path = self.import_epub("e2e-graceful-stop.epub")
            outcome = self.run_job(self.service, self.translate_spec(path, "e2e-graceful-stop.epub"), "graceful stop")
            ended_mark = server.mark()
        finally:
            server.on_response.remove(first_chapter_done)
            server.set_delay("translation", 0.0)
        _check("at" in stop, "no chapter response arrived, so Stop was never pressed")
        _check(stop.get("mode") == "graceful", f"the first Stop was {stop.get('mode')!r}, expected graceful")
        _check(outcome.state == "CANCELLED" and outcome.stop_mode == "graceful",
               f"stopped job ended {outcome.state} ({outcome.stop_mode}): {outcome.error}")
        ended = time.monotonic()
        stop_secs = ended - stop["at"]
        _check(stop_secs <= STOP_DEADLINE, f"the graceful stop took {stop_secs:.1f} s (limit {STOP_DEADLINE:.0f} s)")
        self.expect_quiet_after_stop("graceful stop", ended_mark)
        output_dir = outcome.output_dir
        chapters = _progress_chapters(output_dir)
        done = _completed(chapters)
        _check(done, f"no chapter was saved before the stop: {chapters}")
        _check(len(done) < CHAPTERS, "every chapter finished before the stop took effect; nothing left to resume")
        first = _chapter_requests(server.records(since=mark))
        _check(stop["chapter"] in done, f"the chapter answered before Stop ({stop['chapter']}) is not saved")

        remaining = set(range(1, CHAPTERS + 1)) - done
        mark = server.mark()
        before = process_state()
        t0 = time.monotonic()
        new_id = self.service.resume(outcome.job_id)
        _check(new_id, "Resume returned no job")
        snap = self.wait(self.service, new_id)
        resumed = self._outcome("resume after graceful stop", new_id, snap, t0, before)
        _check(resumed.state == "DONE", f"resumed job ended {resumed.state}: {resumed.error}")
        again = _chapter_requests(server.records(since=mark))
        _check(sorted(again) == sorted(remaining),
               f"Resume translated chapters {sorted(again)}; the missing ones were {sorted(remaining)}")
        book = self._expect_complete_book(output_dir, "resume")
        return {"stop_secs": round(stop_secs, 1), "saved_before_resume": sorted(done),
                "sent_before_stop": len(first), "resumed": sorted(again), "book": book}

    # ---- scenario 5: model stuck, force stop, kill & relaunch, Resume ----------------------------------------

    def force_stop_kill_resume(self) -> dict:
        self.setup()
        self.configure("off")
        server = self.server
        server.set_delay("translation", 0.3)
        answered: list = []

        def stall_after_two(record: Any) -> None:
            if record.kind == "translation" and len(record.chapters) == 1:
                answered.append(record.chapters[0])
                if len(answered) == 2:
                    server.hold()  # the model stops answering

        server.on_response.append(stall_after_two)
        killed_state = self.root / "killed-active.state"
        taps: dict = {}

        def double_tap(job_id: str) -> None:
            deadline = time.monotonic() + self.job_timeout
            while time.monotonic() < deadline and not (server.holding and server.parked >= 1):
                snap = self.service.snapshot(job_id)
                if snap is not None and snap.is_terminal:
                    break
                time.sleep(0.05)
            _check(server.holding and server.parked >= 1, "the run never reached the stalled model")
            active = Path(self.service.active_state_path)
            _check(active.is_file(), "no active.state checkpoint while the job runs")
            shutil.copy2(active, killed_state)  # what a killed app leaves behind
            taps["at"] = time.monotonic()
            taps["first"] = self.service.request_stop()
            time.sleep(0.2)
            taps["second"] = self.service.request_stop()  # tap again = force
            taps["parked"] = server.parked

        try:
            path = self.import_epub("e2e-force-stop.epub")
            outcome = self.run_job(self.service, self.translate_spec(path, "e2e-force-stop.epub"), "force stop",
                                   during=double_tap)
            still_parked = server.parked
            ended_mark = server.mark()
        finally:
            server.on_response.remove(stall_after_two)
            server.release(abort=True)  # the stalled requests never answer the stopped run
            server.set_delay("translation", 0.0)
        stop_secs = time.monotonic() - taps.get("at", time.monotonic())
        _check(taps.get("first") == "graceful" and taps.get("second") == "force",
               f"stop taps gave {taps.get('first')!r} then {taps.get('second')!r}")
        _check(outcome.state == "CANCELLED" and outcome.stop_mode == "force",
               f"force-stopped job ended {outcome.state} ({outcome.stop_mode}): {outcome.error}")
        _check(stop_secs <= STOP_DEADLINE + 2, f"the force stop took {stop_secs:.1f} s with the model stalled")
        output_dir = outcome.output_dir
        self.expect_quiet_after_stop("force stop", ended_mark)  # the dropped requests are not retried

        # Kill & relaunch: a fresh JobService finds the killed run's active.state.
        restart_dir = self._dir("jobs-relaunch")
        shutil.copy2(killed_state, restart_dir / "active.state")
        relaunched = self.new_service(restart_dir)
        _check(outcome.job_id in relaunched.adopted_ids, f"the relaunch did not adopt {outcome.job_id}")
        interrupted = relaunched.recover()
        match = [s for s in interrupted if s.id == outcome.job_id]
        _check(match and not match[0].restore_pending and match[0].state.value == "INTERRUPTED",
               f"recover() left {[(s.id, s.state.value, s.restore_pending) for s in interrupted]}")
        chapters = _progress_chapters(output_dir)
        done = _completed(chapters)
        _check(len(done) < CHAPTERS, "the stalled run finished every chapter; nothing to resume")
        remaining = set(range(1, CHAPTERS + 1)) - done
        mark = server.mark()
        before = process_state()
        t0 = time.monotonic()
        new_id = relaunched.resume(outcome.job_id)
        _check(new_id, "Resume of the interrupted job returned no job")
        snap = self.wait(relaunched, new_id)
        resumed = self._outcome("resume after kill", new_id, snap, t0, before)
        _check(resumed.state == "DONE", f"resumed job ended {resumed.state}: {resumed.error}")
        _check(not relaunched.interrupted, "the resumed job is still listed as Interrupted")
        again = _chapter_requests(server.records(since=mark))
        _check(sorted(again) == sorted(remaining),
               f"Resume translated chapters {sorted(again)}; the missing ones were {sorted(remaining)}")
        book = self._expect_complete_book(output_dir, "resume after kill")
        return {"stop_secs": round(stop_secs, 1), "parked_at_stop": taps.get("parked"), "parked_after": still_parked,
                "saved_before_resume": sorted(done), "resumed": sorted(again), "book": book}

    # ---- scenario 6: hygiene ----------------------------------------------------------------------------------------

    def process_hygiene(self) -> dict:
        self.setup()
        problems: list = []
        for job in self.jobs:
            if job.process_diff:
                problems.append(f"{job.label}: {job.process_diff}")
        all_spawns = list(self.guards.spawns) if self.guards is not None else []
        spawns = [e for e in all_spawns if not e.get("stdlib")]
        stdlib_spawns = [e for e in all_spawns if e.get("stdlib")]
        network = list(self.guards.network) if self.guards is not None else []
        writes = list(self.audit.violations) if self.audit is not None else []
        if spawns:
            problems.append(f"{len(spawns)} process spawn attempt(s): " +
                            "; ".join(f"{e['api']} on {e['thread']} ({' < '.join(e['stack'][-3:])})" for e in spawns[:3]))
        if writes:
            problems.append(f"{len(writes)} write(s) outside the data dirs: " +
                            "; ".join(f"{w['event']} {w['path']} on {w['thread']}" for w in writes[:4]))
        if network and self.isolated:
            problems.append(f"{len(network)} non-loopback connection attempt(s): " +
                            "; ".join(f"{n['api']} {n['target']} on {n['thread']}" for n in network[:4]))
        sandbox_config = str(self.root / "config.json")
        bound = {m.__name__: vars(m).get("CONFIG_FILE") for m in self._backend_modules() if "CONFIG_FILE" in vars(m)}
        leaked = sorted(name for name, value in bound.items() if value != sandbox_config)
        if leaked:
            problems.append(f"backend modules still read the app's own config.json: {leaked}")
        touched = sorted(path for path, before in self._user_files.items() if self._user_file_state().get(path) != before)
        if touched:
            problems.append(f"the app's own settings / chats / jobs changed: {touched}")
        _check(self.jobs, "no job ran; the scenarios above failed early")
        _check(not problems, " | ".join(problems))
        return {
            "jobs": [job.to_dict() for job in self.jobs],
            "spawn_attempts": len(spawns),
            "stdlib_spawn_warnings": sorted({f"{e['api']} from {e['stack'][-1] if e['stack'] else '?'}"
                                             for e in stdlib_spawns}),
            "network_attempts": len(network),
            "network_warnings": [f"{n['api']} {n['target']}" for n in network[:5]] if not self.isolated else [],
            "writes_outside": len(writes),
            "config_file_bound": sorted(bound),
            "user_files_checked": len(self._user_files),
            "requests": self.server.count(status=None) if self.server is not None else 0,
        }


#: (check name, E2ESession method) in run order; ``selftest.SUITES["e2e"]`` is built from it.
SCENARIOS = (
    ("e2e_chat_balanced_glossary", "chat_balanced_glossary"),
    ("e2e_translate_glossary_off", "translate_glossary_off"),
    ("e2e_glossary_edit_qa_pdf", "glossary_edit_qa_pdf"),
    ("e2e_graceful_stop_resume", "graceful_stop_resume"),
    ("e2e_force_stop_kill_resume", "force_stop_kill_resume"),
    ("e2e_process_hygiene", "process_hygiene"),
)


def main(argv: Optional[list] = None) -> int:
    """Host entry: bootstrap into the ``FLET_APP_STORAGE_*`` dirs, run the ``e2e`` suite."""
    import argparse

    parser = argparse.ArgumentParser(description="Glossarion mobile offline end-to-end test (fake OpenAI server)")
    parser.add_argument("--json", help="write the full result JSON to this file")
    parser.add_argument("--keep", action="store_true", help="keep the sandbox after a passing run")
    parser.add_argument("--strict", action="store_true", help="missing packages fail instead of skip")
    parser.add_argument("--only", action="append", help="run only this check (repeatable)")
    args = parser.parse_args(argv)
    if args.json:
        # bootstrap() chdirs into the data dir, so resolve a relative path against the caller's cwd now.
        args.json = os.path.abspath(args.json)
    os.environ[ISOLATED_ENV] = "1"
    if args.keep:
        os.environ["GLOSSARION_E2E_KEEP"] = "1"
    from glossarion_mobile import runtime_bootstrap as rb
    from glossarion_mobile.diagnostics import selftest

    rb.bootstrap()
    result = selftest.run_selftest("e2e", strict=True if args.strict else None,
                                   only=set(args.only) if args.only else None)
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    return 0 if result["ok"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
