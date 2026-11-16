# ocagy_cli.py - OpenCode Antigravity OAuth provider for Glossarion
"""Use NoeFabris/opencode-antigravity-auth through the OpenCode CLI.

This provider intentionally does not reimplement the plugin's private Google API
protocol.  Glossarion launches ``opencode run`` in an isolated, tool-disabled
workspace; OpenCode loads ``opencode-antigravity-auth`` and the plugin owns OAuth,
token refresh, quota handling, and multi-account rotation.

Glossarion model prefixes:
    ocagy0/<friendly-model>  # plugin-managed shared account pool
    ocagy/<friendly-model>   # account 1
    ocagy1/<friendly-model>  # account 2
    ocagy2/<friendly-model>  # account 3, and so on

Examples:
    ocagy/gemini-3.1-pro-high
    ocagy/gemini-3.1-pro-low
    ocagy/gemini-3-flash-high

Environment variables:
    OCAGY_CLI_PATH          Explicit path to opencode/opencode.exe.
    OPENCODE_CLI_PATH       Alternate explicit path.
    OCAGY_WORKSPACE         Base workspace used for isolated CLI requests.
    OCAGY_PLUGIN_PACKAGE    Fixed plugin package; turns Glossarion's plugin auto-update off
                            (default: the newest npm release, checked every 12 hours).
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import queue
import re
import shlex
import shutil
import signal
import socket
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from installer_utils import run_logged_subprocess

try:
    import httpx
except ImportError:  # pragma: no cover - httpx is a required app dependency
    httpx = None

_CANCEL_EVENT = threading.Event()
_ACTIVE_PROCESSES: set[subprocess.Popen] = set()
_ACTIVE_LOCK = threading.RLock()
_ANSI_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
_PLUGIN_NAME = "opencode-antigravity-auth"
# An explicit OCAGY_PLUGIN_PACKAGE turns Glossarion's plugin auto-update off.
_PLUGIN_PACKAGE_OVERRIDE = str(os.environ.get("OCAGY_PLUGIN_PACKAGE", "") or "").strip()
_PLUGIN_PACKAGE = _PLUGIN_PACKAGE_OVERRIDE or f"{_PLUGIN_NAME}@latest"
_PLUGIN_REGISTRY_URL = f"https://registry.npmjs.org/{_PLUGIN_NAME}/latest"
OCAGY_PLUGIN_UPDATE_INTERVAL_SECONDS = 12 * 3600
OCAGY_PLUGIN_UPDATE_RETRY_SECONDS = 3600
OCAGY_PLUGIN_REGISTRY_TIMEOUT_SECONDS = 5.0
_PLUGIN_STATE_FILENAME = "glossarion-plugin-state.json"
OCAGY_PLUGIN_UPDATE_TIMEOUT_SECONDS = 900
_PLUGIN_UPDATE_THREAD_LOCK = threading.Lock()
_PLUGIN_UPDATE_THREAD: Optional[threading.Thread] = None
# The background CLI update first lets npm download the new OpenCode into a
# throw-away folder (killing that run is harmless), then replaces the global CLI
# offline from npm's cache. Killing that second run could leave a half-replaced
# CLI, so its cap is generous although it normally takes a few seconds.
OCAGY_CLI_DOWNLOAD_TIMEOUT_SECONDS = 900
OCAGY_CLI_UPDATE_TIMEOUT_SECONDS = 600
# A stored timestamp further ahead of the clock than this is treated as invalid.
OCAGY_CLOCK_SKEW_SECONDS = 600
# Cross-process coordination (see "Update and launch locks" below). Every wait
# is done in OCAGY_LOCK_POLL_SECONDS slices that check Stop, and has a cap.
OCAGY_LOCK_POLL_SECONDS = 0.2
_UPDATE_LOCK_FILENAME = "glossarion-update.lock"
_LAUNCH_LOCK_FILENAME = "glossarion-launch.lock"
# The CLI update waits this long for OpenCode starts in progress, then skips.
OCAGY_CLI_UPDATE_LAUNCH_WAIT_SECONDS = 60
# A request waits this long for a running CLI install or update. That is the
# longest the launch lock can be held exclusively: the wait above (new starts
# already queue behind the waiting writer), npm's own cap, and the kill.
OCAGY_LAUNCH_WAIT_SECONDS = OCAGY_CLI_UPDATE_LAUNCH_WAIT_SECONDS + OCAGY_CLI_UPDATE_TIMEOUT_SECONDS + 30
# The plugin's README names the oldest OpenCode its legacy entrypoint supports
# ("OpenCode 1.18.29+ remains supported ..." in 2.x). This table is only used
# when that sentence cannot be read from the downloaded package.
_PLUGIN_MIN_OPENCODE_BY_MAJOR: Dict[int, Tuple[int, int, int]] = {2: (1, 18, 29)}
_PLUGIN_MIN_OPENCODE_RE = re.compile(
    r"OpenCode\s+\**v?(\d+\.\d+\.\d+)\+?\**\s+remains\s+supported", re.IGNORECASE
)
OPENCODE_NPM_INSTALL_COMMAND = "npm install -g --allow-scripts=opencode-ai opencode-ai"
NODEJS_WINGET_INSTALL_COMMAND = "winget install --id OpenJS.NodeJS.LTS --exact"
OPENCODE_INSTALL_TIMEOUT_SECONDS = 600
OCAGY_PLUGIN_INSTALL_TIMEOUT_SECONDS = 300
# A request that must install the plugin waits this long for another thread or
# Glossarion process doing the same (that work is bounded by its own timeouts).
OCAGY_PLUGIN_SETUP_WAIT_SECONDS = OCAGY_PLUGIN_INSTALL_TIMEOUT_SECONDS * 2 + 60
# A request that must install a missing OpenCode CLI waits this long for another
# thread or process that holds the update lock. The longest holder is another
# install of a missing CLI (Node.js, then OpenCode), then a plugin download.
OCAGY_CLI_INSTALL_WAIT_SECONDS = max(OCAGY_PLUGIN_UPDATE_TIMEOUT_SECONDS, 2 * OPENCODE_INSTALL_TIMEOUT_SECONDS) + 60
_PLUGIN_BOOTSTRAP_LOCK = threading.Lock()
_PLUGIN_BOOTSTRAPPED_EXECUTABLES: set[str] = set()
_ANTIGRAVITY_CLOUD_CODE_BASE = "https://cloudcode-pa.googleapis.com"
_ANTIGRAVITY_FALLBACK_PROJECT_ID = "rising-fact-p41fc"
_ANTIGRAVITY_QUOTA_USER_AGENT = "antigravity/windows/amd64"

# OpenCode lists the plugin's base model IDs, while Glossarion exposes the
# selectable thinking variants as separate ocagy/ entries. Keep the expansion
# here, next to the CLI integration that owns both representations.
_CATALOG_MODEL_VARIANTS: Tuple[Tuple[str, str, Tuple[str, ...]], ...] = (
    ("google/antigravity-gemini-3.1-pro", "gemini-3.1-pro", ("high", "low")),
    ("google/antigravity-gemini-3-pro", "gemini-3-pro", ("high", "low")),
    (
        "google/antigravity-gemini-3-flash",
        "gemini-3-flash",
        ("minimal", "low", "medium", "high"),
    ),
    # Plugin 2.x: the thinking tier is part of Google's model ID, not a variant.
    ("google/antigravity-gemini-3.8-flash-low", "gemini-3.8-flash-low", ()),
    ("google/antigravity-gemini-3.8-flash-medium", "gemini-3.8-flash-medium", ()),
    ("google/antigravity-gemini-3.8-flash-high", "gemini-3.8-flash-high", ()),
    ("google/antigravity-claude-sonnet-4-6", "claude-sonnet-4-6", ()),
    (
        "google/antigravity-claude-opus-4-6-thinking",
        "claude-opus-4-6-thinking",
        ("low", "max"),
    ),
)


class OcAgyError(RuntimeError):
    """Raised when OpenCode or the Antigravity OAuth plugin cannot complete a request."""


def get_install_instructions() -> str:
    """Return concise, copyable OpenCode setup instructions."""
    if os.name == "nt":
        return (
            "OpenCode CLI was not found.\n\n"
            "Open a new PowerShell window and run:\n"
            f"  {OPENCODE_NPM_INSTALL_COMMAND}\n\n"
            "If npm is not installed, install Node.js LTS first (it includes npm):\n"
            f"  {NODEJS_WINGET_INSTALL_COMMAND}\n\n"
            "After installing Node.js, close and reopen PowerShell, run the OpenCode command, "
            "then restart Glossarion. Advanced users may instead set OCAGY_CLI_PATH."
        )
    return (
        "OpenCode CLI was not found. Install Node.js/npm if needed, then run:\n"
        f"  {OPENCODE_NPM_INSTALL_COMMAND}\n\n"
        "Restart Glossarion afterward. Advanced users may instead set OCAGY_CLI_PATH."
    )


def cancel_stream() -> None:
    _CANCEL_EVENT.set()
    with _ACTIVE_LOCK:
        processes = list(_ACTIVE_PROCESSES)
    for proc in processes:
        _terminate_process_tree(proc)


def reset_cancel() -> None:
    if os.environ.get("TRANSLATION_CANCELLED") == "1":
        return
    _CANCEL_EVENT.clear()


def is_cancelled() -> bool:
    return _CANCEL_EVENT.is_set() or os.environ.get("TRANSLATION_CANCELLED") == "1"


_CANCELLED_MESSAGE = "OpenCode Antigravity request cancelled by user"


# ---------------------------------------------------------------------------
# Update and launch locks
#
# Several Glossarion processes (a second window, glossary worker processes) may
# use this route at once. Two lock files in the workspace base coordinate them:
#   glossarion-update.lock  exclusive: only one process updates the plugin/CLI.
#   glossarion-launch.lock  shared while a process starts OpenCode, exclusive
#                           while npm replaces the global OpenCode CLI.
# Each is an OS lock (LockFileEx on Windows, flock elsewhere) combined with an
# in-process lock for the threads of one process. When the OS lock cannot be
# used, coordination falls back to this process alone (logged once).
# ---------------------------------------------------------------------------


class _LockWaitTimeout(Exception):
    """A bounded lock wait reached its cap."""


class _LockWaitCancelled(Exception):
    """Stop was pressed while waiting for a lock."""


class _SharedExclusiveLock:
    """In-process shared/exclusive lock; a waiting exclusive holder goes first.

    The thread holding it exclusively may take it again (shared or exclusive):
    that nested hold is a no-op.
    """

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._readers = 0
        self._writer: Optional[int] = None
        self._writers_waiting = 0

    def acquire(
        self,
        exclusive: bool,
        timeout: float,
        should_stop: Callable[[], bool],
        on_wait: Optional[Callable[[], None]] = None,
    ) -> str:
        me = threading.get_ident()
        deadline = time.monotonic() + max(0.0, float(timeout))
        with self._cond:
            if self._writer == me:
                return "nested"
            if exclusive:
                self._writers_waiting += 1
            try:
                waited = False
                while True:
                    if self._writer is None and (self._readers == 0 if exclusive else self._writers_waiting == 0):
                        if exclusive:
                            self._writer = me
                        else:
                            self._readers += 1
                        return "held"
                    if should_stop():
                        raise _LockWaitCancelled()
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise _LockWaitTimeout()
                    if not waited and on_wait is not None:
                        on_wait()
                    waited = True
                    self._cond.wait(min(OCAGY_LOCK_POLL_SECONDS, remaining))
            finally:
                if exclusive:
                    self._writers_waiting -= 1

    def release(self, exclusive: bool, token: str) -> None:
        if token != "held":
            return
        with self._cond:
            if exclusive:
                self._writer = None
            else:
                self._readers -= 1
            self._cond.notify_all()

    def held_exclusive_by_me(self) -> bool:
        return self._writer == threading.get_ident()


_UPDATE_LOCAL_LOCK = _SharedExclusiveLock()
_LAUNCH_LOCAL_LOCK = _SharedExclusiveLock()
_LOCK_FALLBACK_LOGGED: set[str] = set()


_WIN_LOCK_API: Optional[Tuple[Any, Any, Any]] = None


def _win_lock_api() -> Tuple[Any, Any, Any]:
    """(LockFileEx, UnlockFileEx, OVERLAPPED) from kernel32, set up once."""
    global _WIN_LOCK_API
    if _WIN_LOCK_API is None:
        import ctypes

        class Overlapped(ctypes.Structure):
            _fields_ = [
                ("Internal", ctypes.c_size_t),
                ("InternalHigh", ctypes.c_size_t),
                ("Offset", ctypes.c_uint32),
                ("OffsetHigh", ctypes.c_uint32),
                ("hEvent", ctypes.c_void_p),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        lock_file_ex = kernel32.LockFileEx
        lock_file_ex.argtypes = [
            ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
            ctypes.POINTER(Overlapped),
        ]
        lock_file_ex.restype = ctypes.c_int
        unlock_file_ex = kernel32.UnlockFileEx
        unlock_file_ex.argtypes = [
            ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(Overlapped),
        ]
        unlock_file_ex.restype = ctypes.c_int
        _WIN_LOCK_API = (lock_file_ex, unlock_file_ex, Overlapped)
    return _WIN_LOCK_API


def _os_try_lock(fd: int, exclusive: bool) -> bool:
    """Try to lock byte 0 of ``fd`` without blocking. False: held by someone else.

    Raises OSError when this file system or platform cannot lock the file.
    """
    if os.name == "nt":
        import ctypes
        import msvcrt

        lock_file_ex, _unlock, overlapped_type = _win_lock_api()
        flags = 0x1 | (0x2 if exclusive else 0)  # LOCKFILE_FAIL_IMMEDIATELY | LOCKFILE_EXCLUSIVE_LOCK
        if lock_file_ex(msvcrt.get_osfhandle(fd), flags, 0, 1, 0, ctypes.byref(overlapped_type())):
            return True
        error = ctypes.get_last_error()
        if error in (33, 32):  # ERROR_LOCK_VIOLATION, ERROR_SHARING_VIOLATION
            return False
        raise OSError(error, ctypes.FormatError(error).strip())
    import fcntl

    try:
        fcntl.flock(fd, (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    return True


def _os_unlock(fd: int) -> None:
    if os.name == "nt":
        import ctypes
        import msvcrt

        _lock, unlock_file_ex, overlapped_type = _win_lock_api()
        unlock_file_ex(msvcrt.get_osfhandle(fd), 0, 1, 0, ctypes.byref(overlapped_type()))
        return
    import fcntl

    fcntl.flock(fd, fcntl.LOCK_UN)


@contextlib.contextmanager
def _coordination_lock(
    filename: str,
    local: _SharedExclusiveLock,
    *,
    exclusive: bool,
    timeout: float,
    should_stop: Optional[Callable[[], bool]] = None,
    logger: Optional[Callable[[str], None]] = None,
    on_wait: Optional[Callable[[], None]] = None,
):
    """Hold ``filename`` (in the workspace base) shared or exclusive across processes.

    Waits at most ``timeout`` seconds in short slices; raises _LockWaitTimeout at
    the cap and _LockWaitCancelled when ``should_stop()`` turns true. A lock file
    that cannot be used never fails the caller: only the in-process lock is held.
    """
    stop = should_stop or (lambda: False)
    deadline = time.monotonic() + max(0.0, float(timeout))
    token = local.acquire(exclusive, timeout, stop, on_wait)
    fd: Optional[int] = None
    locked = False
    try:
        if token == "held":
            try:
                path = _workspace_base() / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                fd = os.open(str(path), os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0), 0o600)
                waited = False
                while not _os_try_lock(fd, exclusive):
                    if stop():
                        raise _LockWaitCancelled()
                    if time.monotonic() >= deadline:
                        raise _LockWaitTimeout()
                    if not waited and on_wait is not None:
                        on_wait()
                    waited = True
                    time.sleep(OCAGY_LOCK_POLL_SECONDS)
                locked = True
            except OSError as exc:
                if fd is not None:
                    os.close(fd)
                    fd = None
                if filename not in _LOCK_FALLBACK_LOGGED:
                    _LOCK_FALLBACK_LOGGED.add(filename)
                    (logger or (lambda _m: None))(
                        f"⚠️ OcAgy: cannot lock {filename} ({exc.strerror or type(exc).__name__}); "
                        "OpenCode updates are coordinated within this Glossarion process only."
                    )
        yield
    finally:
        if fd is not None:
            if locked:
                try:
                    _os_unlock(fd)
                except OSError:
                    pass
            os.close(fd)
        local.release(exclusive, token)


_REQUEST_STOP = "request"  # default: a translation request, released by Stop
_LAST_LAUNCH_WAIT_NOTICE = float("-inf")


@contextlib.contextmanager
def _opencode_launch_slot(
    logger: Optional[Callable[[str], None]] = None,
    *,
    should_stop: Any = _REQUEST_STOP,
    wait_timeout: Optional[float] = None,
):
    """Hold the launch lock shared while starting OpenCode (never during npm's run).

    A request waits here while any Glossarion process replaces the OpenCode CLI;
    Stop releases it at once (``should_stop``; None: not stoppable), and the
    wait ends after ``wait_timeout`` (default OCAGY_LAUNCH_WAIT_SECONDS).
    """
    log = logger or (lambda _m: None)
    stop = (lambda: is_cancelled()) if should_stop == _REQUEST_STOP else should_stop
    cap = OCAGY_LAUNCH_WAIT_SECONDS if wait_timeout is None else max(0.0, float(wait_timeout))

    def waiting_notice() -> None:
        # One line for all requests that start waiting on the same update.
        global _LAST_LAUNCH_WAIT_NOTICE
        now = time.monotonic()
        if now - _LAST_LAUNCH_WAIT_NOTICE >= OCAGY_LAUNCH_WAIT_SECONDS:
            _LAST_LAUNCH_WAIT_NOTICE = now
            log("⏳ OcAgy: waiting for the OpenCode update to finish...")

    try:
        with _coordination_lock(
            _LAUNCH_LOCK_FILENAME,
            _LAUNCH_LOCAL_LOCK,
            exclusive=False,
            timeout=cap,
            should_stop=stop,
            logger=log,
            on_wait=waiting_notice,
        ):
            yield
    except _LockWaitCancelled:
        raise OcAgyError(_CANCELLED_MESSAGE) from None
    except _LockWaitTimeout:
        raise OcAgyError(
            f"An OpenCode install or update by Glossarion was still running after {int(cap)} s, "
            "so OpenCode was not started. Try again when it has finished."
        ) from None


def _spawn_opencode(
    command: List[str],
    logger: Optional[Callable[[str], None]] = None,
    *,
    should_stop: Any = _REQUEST_STOP,
    wait_timeout: Optional[float] = None,
    **kwargs: Any,
) -> subprocess.Popen:
    """Start an OpenCode process, never while npm is replacing the CLI."""
    with _opencode_launch_slot(logger, should_stop=should_stop, wait_timeout=wait_timeout):
        return subprocess.Popen(command, **kwargs)


def _launch_wait_stop(stop_check: Optional[Callable[[], bool]]) -> Optional[Callable[[], bool]]:
    """What ends a setup command's wait for the launch lock.

    Its own ``stop_check`` when given. Otherwise Stop on a request thread (short
    checks such as ``opencode --version`` run there on a first install), and
    nothing on the background update thread, which a translation's Stop never
    interrupts.
    """
    if stop_check is not None:
        return stop_check
    if threading.current_thread() is _PLUGIN_UPDATE_THREAD:
        return None
    return is_cancelled


def _wait_thread_lock(lock: Any, *, timeout: float, what: str) -> None:
    """Acquire a threading lock in short slices that honour Stop; capped."""
    deadline = time.monotonic() + timeout
    while not lock.acquire(timeout=OCAGY_LOCK_POLL_SECONDS):
        if is_cancelled():
            raise OcAgyError(_CANCELLED_MESSAGE)
        if time.monotonic() >= deadline:
            raise OcAgyError(f"{what} did not finish within {int(timeout)} s. Try again later.")


def _candidate_paths() -> Iterable[Path]:
    for key in ("OCAGY_CLI_PATH", "OPENCODE_CLI_PATH"):
        raw = str(os.environ.get(key, "") or "").strip().strip('"')
        if raw:
            yield Path(raw).expanduser()

    # npm's .cmd shim needs node.exe on PATH. Prefer a native binary when both
    # are installed, including when PATH lists the shim first.
    found_native = shutil.which("opencode.exe")
    if found_native:
        yield Path(found_native)

    home = Path.home()
    local_appdata = Path(os.environ.get("LOCALAPPDATA", home / "AppData" / "Local"))
    appdata = Path(os.environ.get("APPDATA", home / "AppData" / "Roaming"))

    # Native installer / common package-manager locations.
    yield home / ".opencode" / "bin" / "opencode.exe"
    yield home / ".opencode" / "bin" / "opencode"
    yield home / ".local" / "bin" / "opencode"
    yield home / ".bun" / "bin" / "opencode.exe"
    yield home / "scoop" / "apps" / "opencode" / "current" / "opencode.exe"
    yield local_appdata / "Programs" / "opencode" / "opencode.exe"
    yield local_appdata / "opencode" / "opencode.exe"
    yield appdata / "npm" / "node_modules" / "opencode-ai" / "node_modules" / "opencode-windows-x64" / "bin" / "opencode.exe"
    yield Path("C:/ProgramData/chocolatey/bin/opencode.exe")
    found = shutil.which("opencode") or shutil.which("opencode.cmd")
    if found:
        yield Path(found)
    yield appdata / "npm" / "opencode.cmd"


def find_executable(explicit_path: Optional[str] = None) -> str:
    candidates: List[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path).expanduser())
    candidates.extend(_candidate_paths())

    seen: set[str] = set()
    for candidate in candidates:
        try:
            normalized = str(candidate.resolve(strict=False))
        except Exception:
            normalized = str(candidate)
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        if candidate.is_file():
            return str(candidate.resolve())

    raise OcAgyError(get_install_instructions())


def _candidate_command(name: str) -> Optional[str]:
    found = shutil.which(name)
    if found:
        return found

    if os.name != "nt":
        return None

    home = Path.home()
    appdata = Path(os.environ.get("APPDATA", home / "AppData" / "Roaming"))
    local_appdata = Path(os.environ.get("LOCALAPPDATA", home / "AppData" / "Local"))
    program_files = Path(os.environ.get("ProgramFiles", "C:/Program Files"))
    candidates: List[Path] = []
    if name in {"npm", "npx"}:
        candidates.extend(
            [
                program_files / "nodejs" / f"{name}.cmd",
                local_appdata / "Programs" / "nodejs" / f"{name}.cmd",
                appdata / "npm" / f"{name}.cmd",
            ]
        )
    elif name == "winget":
        candidates.append(local_appdata / "Microsoft" / "WindowsApps" / "winget.exe")

    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())
    return None


def _split_setup_override(value: str) -> List[str]:
    try:
        return shlex.split(value, posix=(os.name != "nt"))
    except Exception:
        return value.split()


def _format_setup_command(command: List[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(command)
    return shlex.join(command)


def _run_setup_command(
    command: List[str],
    *,
    action: str,
    log_fn: Optional[Callable[[str], None]] = None,
    timeout: int = OPENCODE_INSTALL_TIMEOUT_SECONDS,
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
    stop_check: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    """Run one setup command, bounded: on timeout or Stop its whole process tree is killed.

    The result's ``reason`` is a one-line summary without command output or
    local paths, for log lines the user sees during a background update.
    """
    logger = log_fn or (lambda _message: None)
    command_text = _format_setup_command(command)
    logger(f"▶ Running: {command_text}")
    try:
        result = run_logged_subprocess(
            command,
            log_fn=logger,
            timeout=max(1, int(timeout)),
            cwd=str(cwd) if cwd is not None else None,
            env=env if env is not None else _subprocess_env(),
            # Own process group/session, so the tree kill reaches npm's node or
            # opencode.exe behind a .cmd shim and never Glossarion itself.
            popen_kwargs={"creationflags": _creation_flags(), "start_new_session": os.name != "nt"},
            terminate_fn=_terminate_process_tree,
            stop_check=stop_check,
            popen_fn=lambda args, **kwargs: _spawn_opencode(args, logger, should_stop=_launch_wait_stop(stop_check), **kwargs),
        )
    except OcAgyError:
        raise
    except Exception as exc:
        error = f"{action} could not start: {exc}"
        logger(f"❌ {error}")
        return {"ok": False, "command": command_text, "error": error, "reason": f"could not start ({type(exc).__name__})"}

    output = str(result.get("output") or "No installer output was returned.")
    if result.get("cancelled"):
        return {"ok": False, "cancelled": True, "command": command_text, "error": f"{action} was stopped.", "reason": "stopped", "output": output}
    if result.get("timed_out"):
        error = f"{action} timed out after {max(1, int(timeout))}s."
        logger(f"❌ {error}")
        return {"ok": False, "command": command_text, "error": error, "output": output, "reason": f"timed out after {max(1, int(timeout))} s"}
    returncode = int(result.get("returncode", -1))
    if returncode != 0:
        error = f"{action} exited with code {returncode}: {output}"
        logger(f"❌ {error}")
        npm_code = re.search(r"npm (?:ERR!|error) code (\w+)", output)
        return {
            "ok": False,
            "command": command_text,
            "error": error,
            "output": output,
            "reason": f"exited with code {returncode}" + (f" (npm {npm_code.group(1)})" if npm_code else ""),
        }
    return {
        "ok": True,
        "command": command_text,
        "output": output,
    }


def _nodejs_install_command() -> Optional[List[str]]:
    override = str(os.environ.get("OCAGY_NODE_INSTALL_CMD", "") or "").strip()
    if override:
        return _split_setup_override(override)
    if os.name != "nt":
        return None
    winget = _candidate_command("winget")
    if not winget:
        return None
    return [
        winget,
        "install",
        "--id",
        "OpenJS.NodeJS.LTS",
        "--exact",
        "--silent",
        "--accept-package-agreements",
        "--accept-source-agreements",
        "--disable-interactivity",
    ]


def _opencode_install_command() -> Optional[List[str]]:
    override = str(os.environ.get("OCAGY_OPENCODE_INSTALL_CMD", "") or "").strip()
    if override:
        return _split_setup_override(override)

    # OpenCode recommends its user-local install script on macOS/Linux. On
    # Windows, npm is the supported native route.
    if os.name != "nt":
        curl = _candidate_command("curl")
        bash = _candidate_command("bash")
        if curl and bash:
            return [
                bash,
                "-c",
                f"{shlex.quote(curl)} -fsSL https://opencode.ai/install | {shlex.quote(bash)}",
            ]

    npm = _candidate_command("npm")
    if npm:
        return [npm, "install", "-g", "--allow-scripts=opencode-ai", "opencode-ai"]
    return None


def ensure_opencode_installed(log_fn=None) -> str:
    """Find OpenCode or install it automatically, retaining manual fallback text."""
    logger = log_fn or (lambda _message: None)
    # Waits (Stop-aware, capped) while a Glossarion process replaces the CLI, so
    # a half-written install is never mistaken for a missing one.
    with _opencode_launch_slot(logger):
        try:
            return find_executable()
        except OcAgyError:
            pass

    # One installer across threads and Glossarion processes (the update lock).
    # The wait is Stop-aware and capped; so is the install.
    try:
        with _coordination_lock(
            _UPDATE_LOCK_FILENAME,
            _UPDATE_LOCAL_LOCK,
            exclusive=True,
            timeout=OCAGY_CLI_INSTALL_WAIT_SECONDS,
            should_stop=is_cancelled,
            logger=logger,
        ):
            return _install_missing_opencode(logger)
    except _LockWaitCancelled:
        raise OcAgyError(_CANCELLED_MESSAGE) from None
    except _LockWaitTimeout:
        raise OcAgyError(
            "OpenCode CLI was not found, and another Glossarion task kept the OpenCode setup busy, "
            "so it was not installed now. Try again later.\n\n" + get_install_instructions()
        ) from None


def _install_missing_opencode(logger: Callable[[str], None]) -> str:
    """Install OpenCode when it is still missing; the caller holds the update lock.

    Re-checks first: another thread or process may have installed it while this
    one waited. Stop ends a running installer at once (its whole tree). No
    OpenCode starts anywhere while the OpenCode installer writes the CLI (the
    launch lock, exclusive); the Node.js install before it does not block starts.
    """
    try:
        return find_executable()
    except OcAgyError:
        pass

    logger("🧰 OpenCode CLI was not found. Glossarion is installing it automatically...")
    details: List[str] = []
    command = _opencode_install_command()

    if command is None and os.name == "nt":
        node_command = _nodejs_install_command()
        if node_command is not None:
            logger("🧰 npm is unavailable. Installing Node.js LTS first...")
            node_result = _run_setup_command(
                node_command,
                action="Node.js LTS installation",
                log_fn=logger,
                stop_check=is_cancelled,
            )
            if node_result.get("cancelled"):
                raise OcAgyError(_CANCELLED_MESSAGE)
            if not node_result.get("ok"):
                details.append(str(node_result.get("error") or "Node.js installation failed."))
            command = _opencode_install_command()
        else:
            details.append("npm and a supported automatic Node.js installer were not found.")

    if command is not None:
        with _coordination_lock(
            _LAUNCH_LOCK_FILENAME,
            _LAUNCH_LOCAL_LOCK,
            exclusive=True,
            timeout=OCAGY_CLI_UPDATE_LAUNCH_WAIT_SECONDS,
            should_stop=is_cancelled,
            logger=logger,
        ):
            install_result = _run_setup_command(
                command,
                action="OpenCode installation",
                log_fn=logger,
                stop_check=is_cancelled,
            )
        if install_result.get("cancelled"):
            raise OcAgyError(_CANCELLED_MESSAGE)
        if not install_result.get("ok"):
            details.append(str(install_result.get("error") or "OpenCode installation failed."))
    elif not details:
        details.append("No supported automatic OpenCode installer was found for this system.")

    try:
        executable = find_executable()
        logger(f"✅ OpenCode installed successfully: {executable}")
        return executable
    except OcAgyError:
        pass

    detail = "\n".join(dict.fromkeys(details))
    raise OcAgyError(
        "Glossarion could not install OpenCode automatically.\n"
        f"Installer details: {detail or 'The installer completed, but opencode was not found.'}\n\n"
        + get_install_instructions()
    )


def _plugin_fallback_instructions() -> str:
    return (
        "Fallback: verify that the Glossarion workspace opencode.json contains "
        f'\"plugin\": [\"{_plugin_spec()}\"], then run `opencode models google` '
        "or `opencode auth login` once."
    )


def _verify_auth_plugin(
    executable: str,
    logger: Callable[[str], None],
    workspace: Path,
    *,
    stop_check: Optional[Callable[[], bool]] = None,
) -> Tuple[bool, str]:
    """Load the configured plugin through OpenCode (installing it when missing).

    Returns (ok, short reason). On a request (``stop_check`` given) Stop ends
    the run at once with the route's cancelled error.
    """
    result = _run_setup_command(
        [executable, "models", "google"],
        action="OpenCode Antigravity auth plugin installation",
        log_fn=logger,
        timeout=OCAGY_PLUGIN_INSTALL_TIMEOUT_SECONDS,
        cwd=workspace,
        stop_check=stop_check,
    )
    if result.get("cancelled"):
        raise OcAgyError(_CANCELLED_MESSAGE)
    output = str(result.get("output") or "")
    if not result.get("ok") or "google/antigravity-" not in output.lower():
        return False, str(
            result.get("error")
            or "OpenCode finished without exposing any Antigravity plugin models."
        )
    return True, ""


def ensure_auth_plugin_installed(executable: str, log_fn=None) -> None:
    """Resolve and verify opencode-antigravity-auth through OpenCode itself.

    Runs once per process. A missing plugin is pinned to the newest npm release
    before OpenCode loads it; an installed plugin is checked for updates in the
    background (see ``_update_auth_plugin``) so requests never wait on npm.
    """
    logger = log_fn or (lambda _message: None)
    key = str(Path(executable).resolve(strict=False)).casefold()
    if key in _PLUGIN_BOOTSTRAPPED_EXECUTABLES:
        return

    # Other request threads wait here while one installs the plugin: in short
    # slices, so Stop releases them at once, and never longer than the cap.
    _wait_thread_lock(
        _PLUGIN_BOOTSTRAP_LOCK,
        timeout=OCAGY_PLUGIN_SETUP_WAIT_SECONDS,
        what="The OpenCode Antigravity auth plugin setup",
    )
    try:
        if key in _PLUGIN_BOOTSTRAPPED_EXECUTABLES:
            return

        workspace = _workspace_dir()
        _secure_plugin_account_storage(required=False)
        verified = False
        if not _PLUGIN_PACKAGE_OVERRIDE and _installed_plugin_version(_plugin_spec()) is None:
            # First install (or a cleared cache): pin the newest release before
            # OpenCode loads the plugin for the first time.
            verified = _update_auth_plugin(executable, logger, force=True)
            workspace = _workspace_dir()
        if not verified:
            logger(
                "🧩 Ensuring the OpenCode Antigravity auth plugin is installed "
                f"({_plugin_spec()})..."
            )
            ok, detail = _verify_auth_plugin(executable, logger, workspace, stop_check=is_cancelled)
            if not ok:
                # A plugin that needs a newer OpenCode fails here; the background
                # check may still bring OpenCode up to date for the next request.
                _start_background_plugin_update(executable, logger)
                raise OcAgyError(
                    "Glossarion could not install or verify opencode-antigravity-auth automatically.\n"
                    f"Installer details: {detail}\n\n{_plugin_fallback_instructions()}"
                )

        _secure_plugin_account_storage(required=True)
        _PLUGIN_BOOTSTRAPPED_EXECUTABLES.add(key)
        version = _installed_plugin_version(_plugin_spec())
        logger(
            "✅ OpenCode Antigravity auth plugin is installed and ready"
            + (f" ({_PLUGIN_NAME} {version})." if version else ".")
        )
        _start_background_plugin_update(executable, logger)
    finally:
        _PLUGIN_BOOTSTRAP_LOCK.release()


# ---------------------------------------------------------------------------
# Plugin auto-update
#
# OpenCode installs a plugin given as ``name@spec`` once into
# <cache>/opencode/packages/<name>@<spec> and never re-resolves it, so
# ``@latest`` stays on whatever was newest on the first run. Glossarion therefore
# resolves the newest release itself and pins that exact version in the
# workspace config. Each version gets its own cache folder, so the previous
# working version stays installed and switching back needs no download.
# ---------------------------------------------------------------------------


def _workspace_base() -> Path:
    configured = str(os.environ.get("OCAGY_WORKSPACE", "") or "").strip()
    return Path(configured).expanduser() if configured else Path.home() / ".glossarion" / "opencode_antigravity"


def _plugin_state_path() -> Path:
    return _workspace_base() / _PLUGIN_STATE_FILENAME


def _read_plugin_state() -> Dict[str, Any]:
    try:
        payload = json.loads(_plugin_state_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_plugin_state(state: Dict[str, Any]) -> bool:
    """Write the state through a unique temporary file; never raises.

    Writers hold the update lock. On Windows the final replace can still fail
    for a moment while another process is reading the file; the replace is
    attempted a few more times within about a second, then the write is
    skipped (the next check simply runs again).
    """
    path = _plugin_state_path()
    temporary: Optional[str] = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
        for attempt in range(10):
            try:
                os.replace(temporary, path)
                temporary = None
                return True
            except PermissionError:
                if attempt == 9:
                    raise
                time.sleep(0.1)
    except OSError:
        return False
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except OSError:
                pass
    return False


def _valid_timestamp(value: Any, now: float) -> float:
    """A stored time, or 0 when it is missing, unreadable or ahead of the clock."""
    try:
        stamp = float(value or 0)
    except (TypeError, ValueError):
        return 0.0
    return stamp if 0 < stamp <= now + OCAGY_CLOCK_SKEW_SECONDS else 0.0


def _plugin_spec() -> str:
    """The plugin entry Glossarion writes into the workspace opencode.json."""
    if _PLUGIN_PACKAGE_OVERRIDE:
        return _PLUGIN_PACKAGE_OVERRIDE
    spec = str(_read_plugin_state().get("spec") or "")
    if re.fullmatch(re.escape(_PLUGIN_NAME) + r"@(?:latest|\d+\.\d+\.\d+)", spec):
        return spec
    return _PLUGIN_PACKAGE


def _opencode_cache_dir() -> Path:
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")).expanduser() / "opencode"


def _plugin_package_dir(spec: str) -> Path:
    return _opencode_cache_dir() / "packages" / spec


def _plugin_root(spec: str) -> Path:
    return _plugin_package_dir(spec) / "node_modules" / _PLUGIN_NAME


def _installed_plugin_version(spec: str) -> Optional[str]:
    try:
        payload = json.loads((_plugin_root(spec) / "package.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    version = str(payload.get("version", "") or "").strip() if isinstance(payload, dict) else ""
    return version or None


def _version_tuple(value: Any) -> Optional[Tuple[int, int, int]]:
    match = re.search(r"(?<![\d.])(\d+)\.(\d+)\.(\d+)(?![\d.]*-)", str(value or ""))
    return (int(match.group(1)), int(match.group(2)), int(match.group(3))) if match else None


def _format_version(value: Optional[Tuple[int, int, int]]) -> str:
    return ".".join(str(part) for part in value) if value else "unknown"


def _fetch_latest_plugin_version(timeout: float = OCAGY_PLUGIN_REGISTRY_TIMEOUT_SECONDS) -> str:
    """Return the npm ``latest`` release of the plugin (no account data is sent)."""
    request = urllib.request.Request(
        _PLUGIN_REGISTRY_URL,
        headers={"Accept": "application/json", "User-Agent": "Glossarion"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read(1 << 20).decode("utf-8", errors="replace"))
    version = str(payload.get("version", "") if isinstance(payload, dict) else "").strip()
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise OcAgyError(f"npm returned no release version for {_PLUGIN_NAME}")
    return version


def _opencode_version(executable: str) -> Optional[Tuple[int, int, int]]:
    """The CLI's version when it is a plain stable release (x.y.z), else None.

    None also covers an unreadable version (no output, an error, a timeout) and
    pre-releases such as 0.0.0-dev-... or 2.0.0-beta.1. Bounded: on timeout the
    whole process tree is killed. On a request (a first install) Stop ends it
    with the route's cancelled error; the background update is not interrupted.
    """
    result = _run_setup_command(
        [executable, "--version"],
        action="OpenCode version check",
        timeout=30,
        stop_check=_launch_wait_stop(None),
    )
    if result.get("cancelled"):
        raise OcAgyError(_CANCELLED_MESSAGE)
    if not result.get("ok"):
        return None
    return _version_tuple(_clean_text(str(result.get("output") or "")))


def _plugin_min_opencode(root: Path, version: str) -> Optional[Tuple[int, int, int]]:
    """Oldest OpenCode CLI the downloaded plugin states it supports."""
    try:
        match = _PLUGIN_MIN_OPENCODE_RE.search((root / "README.md").read_text(encoding="utf-8", errors="replace"))
    except OSError:
        match = None
    if match:
        return _version_tuple(match.group(1))
    major = (_version_tuple(version) or (0, 0, 0))[0]
    known = [key for key in _PLUGIN_MIN_OPENCODE_BY_MAJOR if key <= major]
    return _PLUGIN_MIN_OPENCODE_BY_MAJOR[max(known)] if known else None


def _npm_global_prefix(npm: str) -> Optional[Path]:
    """``npm prefix -g`` of this npm (bounded, output never logged), or None."""
    result = _run_setup_command([npm, "prefix", "-g"], action="npm prefix check", log_fn=_quiet, timeout=60)
    if not result.get("ok"):
        return None
    for line in reversed(_clean_text(str(result.get("output") or "")).splitlines()):
        line = line.strip().strip('"')
        if line and Path(line).is_absolute():
            return Path(line)
    return None


def _path_within(path: Path, folder: Path) -> bool:
    child = os.path.normcase(str(path.resolve(strict=False)))
    parent = os.path.normcase(str(folder.resolve(strict=False)))
    try:
        return os.path.commonpath([child, parent]) == parent
    except ValueError:  # different drives
        return False


def _npm_for_global_opencode(executable: str) -> Optional[Tuple[str, Path, Path]]:
    """(npm, global prefix, package folder) when this OpenCode CLI is an
    ``opencode-ai`` install under that npm's global prefix, else None.

    Only such a CLI is updated automatically, by that same npm. OpenCode from
    bun, yarn, another Node version (nvm), the install script or a package
    manager is left alone: npm would add a second global copy instead. The
    update passes the returned prefix to npm explicitly, so npm writes exactly
    where the CLI was found, also when npm printed its prefix masked.
    """
    npm = _candidate_command("npm")
    if not npm:
        return None
    prefix = _npm_global_prefix(npm)
    exe = Path(executable)
    if prefix is not None and "***" in str(prefix):
        # npm prints values that look like tokens (a UUID folder name, say) as
        # "***": take the folder above the CLI that matches the printed path.
        pattern = re.escape(os.path.normcase(str(prefix))).replace(re.escape("***"), r"[^\\/]+")
        resolved = exe.resolve(strict=False)
        prefix = next(
            (folder for folder in (resolved, *resolved.parents) if re.fullmatch(pattern, os.path.normcase(str(folder)))),
            None,
        )
    if prefix is None:
        return None
    for package in (prefix / "node_modules" / "opencode-ai", prefix / "lib" / "node_modules" / "opencode-ai"):
        if _path_within(exe, package):
            return npm, prefix, package
        # npm's own shims: <prefix>/opencode.cmd on Windows, <prefix>/bin/opencode elsewhere.
        if (
            package.is_dir()
            and exe.name.casefold().startswith("opencode")
            and any(_path_within(exe.parent, folder) and _path_within(folder, exe.parent) for folder in (prefix, prefix / "bin"))
        ):
            return npm, prefix, package
    return None


def _manual_cli_update_command(npm_global: bool, required: Tuple[int, int, int]) -> str:
    """The command a user runs to get an OpenCode the plugin supports."""
    return f"npm install -g opencode-ai@{required[0]}" if npm_global else "opencode upgrade"


def _download_opencode_cli(npm: str, major: int) -> Tuple[str, str]:
    """Let npm download the newest stable ``opencode-ai@<major>`` into its cache.

    The install goes into a throw-away folder, so killing it at its timeout
    leaves the user's global OpenCode untouched. Returns (version, "") or
    ("", short reason). The global install that follows then runs ``--offline``
    from the cache: short, and no network while the old CLI is being replaced.
    """
    base = _workspace_base()
    base.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="cli-download-", dir=str(base)))
    version = ""
    try:
        result = _run_setup_command(
            [npm, "install", "--prefix", str(stage), "--ignore-scripts", "--no-audit", "--no-fund", f"opencode-ai@{int(major)}"],
            action="OpenCode CLI download",
            log_fn=_quiet,
            timeout=OCAGY_CLI_DOWNLOAD_TIMEOUT_SECONDS,
            cwd=stage,
        )
        try:
            payload = json.loads((stage / "node_modules" / "opencode-ai" / "package.json").read_text(encoding="utf-8"))
            version = str(payload.get("version", "") or "").strip() if isinstance(payload, dict) else ""
        except (OSError, ValueError):
            version = ""
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    if not result.get("ok"):
        return "", str(result.get("reason") or "the download failed")
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        return "", "npm resolved no stable release"
    return version, ""


def _plugin_update_due(state: Dict[str, Any], now: float) -> bool:
    failed = _valid_timestamp(state.get("check_failed_at"), now)
    if failed and now - failed < OCAGY_PLUGIN_UPDATE_RETRY_SECONDS:
        return False
    checked = _valid_timestamp(state.get("checked_at"), now)
    return not checked or now - checked >= OCAGY_PLUGIN_UPDATE_INTERVAL_SECONDS


def _mark_plugin_retry(state: Dict[str, Any], now: float) -> None:
    """A transient failure: try again after OCAGY_PLUGIN_UPDATE_RETRY_SECONDS."""
    state["check_failed_at"] = now
    state.pop("checked_at", None)


def _storage_hook_fingerprint() -> str:
    """Identify the storage hook's anchors and replacements (changes when a release changes them)."""
    import proxy_token_storage

    texts: List[str] = []

    def walk(code: Any) -> None:
        for const in code.co_consts:
            if isinstance(const, str):
                texts.append(const)
            elif hasattr(const, "co_consts"):
                walk(const)

    for name in ("patch_ocagy", "_patch_ocagy_save_serialization", "_patch_ocagy_oauth"):
        function = getattr(proxy_token_storage, name, None)
        code = getattr(function, "__code__", None)
        if code is not None:
            walk(code)
    return hashlib.sha256("\0".join(texts).encode("utf-8")).hexdigest()[:16]


def _plugin_version_rejected(state: Dict[str, Any], version: str, now: float) -> bool:
    entry = (state.get("rejected") or {}).get(version)
    if not isinstance(entry, dict):
        return False
    # A hook-incompatible version stays rejected only while the hook is unchanged
    # (a Glossarion release with new anchors tries it again); a failed check is
    # retried after a week.
    if entry.get("permanent"):
        return entry.get("hook") == _storage_hook_fingerprint()
    at = _valid_timestamp(entry.get("at"), now)
    return bool(at) and now - at < 7 * 86400


def _reject_plugin_version(state: Dict[str, Any], version: str, reason: str, *, permanent: bool) -> None:
    rejected = state.setdefault("rejected", {})
    entry: Dict[str, Any] = {"reason": reason[:300], "at": time.time(), "permanent": permanent}
    if permanent:
        entry["hook"] = _storage_hook_fingerprint()
    rejected[version] = entry


def _install_plugin_candidate(
    executable: str,
    spec: str,
    logger: Callable[[str], None],
    *,
    timeout: float = OCAGY_PLUGIN_UPDATE_TIMEOUT_SECONDS,
    stop_check: Optional[Callable[[], bool]] = None,
) -> str:
    """Let OpenCode download one plugin version into its cache, without accounts.

    The new version is not yet patched with Glossarion's encrypted storage hook,
    so it runs once against an empty, throw-away config and data directory: it
    can neither read nor rewrite the real account store. Returns a short error
    reason, or "" on success. Bounded by ``timeout``; on a request, Stop
    (``stop_check``) ends it at once with the route's cancelled error.
    """
    base = _workspace_base()
    base.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="plugin-update-", dir=str(base)))
    try:
        (stage / "opencode.json").write_text(
            json.dumps({"$schema": "https://opencode.ai/config.json", "plugin": [spec], "share": "disabled"}),
            encoding="utf-8",
        )
        env = _subprocess_env(stage / "config")
        for name, folder in (("XDG_CONFIG_HOME", "xdg-config"), ("XDG_DATA_HOME", "data"), ("APPDATA", "appdata")):
            (stage / folder).mkdir(parents=True, exist_ok=True)
            env[name] = str(stage / folder)
        (stage / "config").mkdir(parents=True, exist_ok=True)
        # Load only the throw-away config: without this OpenCode also reads the
        # opencode.json of parent folders, i.e. the real workspace config with
        # the installed plugin, and that run hung until its timeout in a test.
        env["OPENCODE_CONFIG"] = str(stage / "opencode.json")
        env["OPENCODE_DISABLE_PROJECT_CONFIG"] = "1"
        env.pop("OPENCODE_CONFIG_CONTENT", None)
        result = _run_setup_command(
            [executable, "models", "google"],
            action=f"Download of {spec}",
            log_fn=logger,
            timeout=timeout,
            cwd=stage,
            env=env,
            stop_check=stop_check,
        )
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    if result.get("cancelled"):
        raise OcAgyError(_CANCELLED_MESSAGE)
    version = spec.rsplit("@", 1)[-1]
    if _installed_plugin_version(spec) == version:
        return ""
    return str(result.get("reason") or "OpenCode did not install it")


def _quiet(_message: str) -> None:
    """Logger for command output the GUI log never shows (npm, OpenCode catalogs)."""


def _short_reason(text: Any) -> str:
    """First line of an error, cut before any command output."""
    line = next((part.strip() for part in str(text or "").splitlines() if part.strip()), "")
    match = re.match(r"(.*?exited with code -?\d+)", line)
    line = match.group(1) if match else line
    # Never show the local home folder (it carries the user name).
    home = str(Path.home())
    for form in {home, home.replace("\\", "/"), home.replace("\\", "\\\\")}:
        if len(form) > 3:
            line = re.sub(re.escape(form), "~", line, flags=re.IGNORECASE)
    return line[:200]


def _error_reason(exc: BaseException) -> str:
    """A short, path-free reason for an unexpected error in the background update."""
    if isinstance(exc, OSError) and (exc.filename is not None or getattr(exc, "filename2", None) is not None):
        # str() of a file error names the file; strerror alone does not.
        return type(exc).__name__ + (f": {exc.strerror}" if exc.strerror else "")
    return _short_reason(_quota_error_message(exc))


def _update_opencode_cli(
    executable: str,
    cli_version: Optional[Tuple[int, int, int]],
    required: Tuple[int, int, int],
    target: str,
    current_text: str,
    logger: Callable[[str], None],
    *,
    allow_update: bool,
) -> Optional[str]:
    """Bring the global OpenCode CLI to ``required`` when that is safe.

    Runs npm only for a stable npm-global ``opencode-ai`` of the required major,
    and never on a request (``allow_update`` False). Every other case leaves the
    CLI alone and logs one line with the command the user can run. Returns the
    (re-detected) executable after a successful update, "" when the update is
    left to the background check (a request), else None.
    """
    need = f"{_PLUGIN_NAME} {target} needs OpenCode {_format_version(required)} or newer"
    if cli_version is None:
        why = "the installed OpenCode version could not be read or is not a stable release"
    elif cli_version[0] != required[0]:
        why = f"the installed OpenCode {_format_version(cli_version)} is a different major version"
    else:
        why = ""
    if not why and not allow_update:
        # A request runs no npm command at all, not even `npm prefix -g`: the
        # background check finds out whether it can update this CLI.
        logger(
            f"⚠️ OcAgy: {need} (installed: {_format_version(cli_version)}). "
            "Glossarion updates OpenCode in the background when it can."
        )
        return ""
    # `npm prefix -g` runs only here: in the background, when the plugin needs a newer CLI.
    found = _npm_for_global_opencode(executable) if allow_update else None
    npm, prefix, package = found if found else ("", None, None)
    manual = _manual_cli_update_command(bool(npm), required)
    if not why and not npm:
        why = (
            f"the installed OpenCode {_format_version(cli_version)} is not a global npm install "
            "that Glossarion can update with npm"
        )
    if not why and not os.access(str(package.parent), os.W_OK):
        # e.g. a root-owned /usr/local: npm would fail with EACCES on every check.
        why = (
            f"the installed OpenCode {_format_version(cli_version)} is in a folder "
            "Glossarion may not write to"
        )
    if why:
        logger(f"⚠️ OcAgy: {need}; {why}. Keeping {current_text}. To use it, run: {manual}")
        return None

    retry = f"Keeping {current_text}; Glossarion tries again in {OCAGY_PLUGIN_UPDATE_INTERVAL_SECONDS // 3600} h."
    logger(f"⬆️ OcAgy: {need} (installed: {_format_version(cli_version)}). Updating OpenCode with npm...")
    # 1. Download (slow, network): into a throw-away folder; requests run on.
    version, reason = _download_opencode_cli(npm, cli_version[0])
    if not version:
        logger(f"⚠️ OcAgy: could not download the new OpenCode ({_short_reason(reason)}). {retry} To update it yourself, run: {manual}")
        return None
    newest = _version_tuple(version)
    if newest is None or newest < required:
        logger(
            f"⚠️ OcAgy: the newest OpenCode {cli_version[0]}.x is {version}, older than {_format_version(required)}. "
            f"Keeping {current_text}."
        )
        return None

    # 2. Replace the global CLI offline from npm's cache (a few seconds). npm
    # replaces a running opencode.exe fine on Windows (it moves the busy folder
    # aside). No Glossarion process starts OpenCode meanwhile: the launch lock
    # is held exclusively, and every start takes it shared. The prefix is the
    # folder the CLI was found under, so npm writes there and nowhere else.
    command = [
        npm, "install", "-g", "--prefix", str(prefix), "--offline", "--no-audit", "--no-fund",
        "--allow-scripts=opencode-ai", f"opencode-ai@{version}",
    ]
    try:
        with _coordination_lock(
            _LAUNCH_LOCK_FILENAME,
            _LAUNCH_LOCAL_LOCK,
            exclusive=True,
            timeout=OCAGY_CLI_UPDATE_LAUNCH_WAIT_SECONDS,
            logger=logger,
        ):
            result = _run_setup_command(
                command,
                action="OpenCode CLI update",
                log_fn=_quiet,
                timeout=OCAGY_CLI_UPDATE_TIMEOUT_SECONDS,
            )
    except _LockWaitTimeout:
        logger(f"⚠️ OcAgy: OpenCode stayed busy starting requests, so it was not updated now. {retry}")
        return None
    # npm is done: requests start again while the new version is read.
    try:
        executable = find_executable()
    except OcAgyError:
        pass
    new_cli = _opencode_version(executable)
    if not result.get("ok") or new_cli is None or new_cli < required:
        # Next attempt with the next regular check (12 h), never on app start.
        reason = str(result.get("reason") or "")
        if new_cli is None:
            # It was readable before: npm stopped half-way (killed at its cap).
            logger(
                "❌ OcAgy: the OpenCode update did not finish" +(f" ({reason})" if reason else "")
                + f" and OpenCode may no longer start. To repair it, run: npm install -g opencode-ai@{version}"
            )
            return None
        logger(
            f"⚠️ OcAgy: the OpenCode update did not reach {_format_version(required)} "
            f"(now: {_format_version(new_cli)}" + (f"; {reason}" if reason else "")
            + f"). {retry} To update it yourself, run: {manual}"
        )
        return None
    logger(f"✅ OcAgy: OpenCode CLI updated {_format_version(cli_version)} → {_format_version(new_cli)}.")
    return executable


def _finish_deferred_cli_update(
    state: Dict[str, Any], executable: str, current_version: str, logger: Callable[[str], None]
) -> None:
    """Run the CLI update a first install left to the background check (``cli_needed``).

    The marker stays after a failure, so the next regular check tries again.
    """
    pending = state.get("cli_needed")
    if not isinstance(pending, dict):
        return
    required = _version_tuple(pending.get("required"))
    if pending.get("version") != current_version or required is None:
        state.pop("cli_needed", None)  # the plugin moved on; its own check covers the CLI
        return
    cli_version = _opencode_version(executable)
    if cli_version is not None and cli_version >= required:
        state.pop("cli_needed", None)
        return
    if _update_opencode_cli(
        executable, cli_version, required, current_version, current_version, logger, allow_update=True
    ):
        state.pop("cli_needed", None)


def _update_auth_plugin(
    executable: str,
    logger: Callable[[str], None],
    *,
    force: bool = False,
) -> bool:
    """Move the workspace to the newest plugin release when that is safe.

    Called from the background update thread, and on a first install (no plugin
    cached yet, ``force``) from a request. Returns True when a new version was
    switched to and passed its check. Every failure leaves the previously
    configured version in place and is logged once. The check after a switch
    (``opencode models google``) proves the plugin loads and lists its models;
    it sends no generation request.

    Only one thread of one Glossarion process updates at a time (the update
    lock file). The background check skips when another one is at it; a first
    install waits for it, Stop-aware and capped.
    """
    if _PLUGIN_PACKAGE_OVERRIDE:
        return False
    if not force and not _plugin_update_due(_read_plugin_state(), time.time()):
        return False
    try:
        with _coordination_lock(
            _UPDATE_LOCK_FILENAME,
            _UPDATE_LOCAL_LOCK,
            exclusive=True,
            timeout=OCAGY_PLUGIN_SETUP_WAIT_SECONDS if force else 0,
            should_stop=is_cancelled if force else None,
            logger=logger,
            on_wait=(lambda: logger(f"⏳ OcAgy: waiting for another Glossarion process to set up {_PLUGIN_NAME}...")) if force else None,
        ):
            return _update_auth_plugin_locked(executable, logger, force=force)
    except _LockWaitCancelled:
        raise OcAgyError(_CANCELLED_MESSAGE) from None
    except _LockWaitTimeout:
        if force:
            raise OcAgyError(
                f"Another Glossarion process was still setting up {_PLUGIN_NAME} after "
                f"{OCAGY_PLUGIN_SETUP_WAIT_SECONDS} s. Try again later."
            ) from None
        return False  # another thread or process is checking right now


def _update_auth_plugin_locked(executable: str, logger: Callable[[str], None], *, force: bool) -> bool:
    from proxy_token_storage import patch_ocagy

    # Re-read under the lock: another process may have just checked or switched.
    state = _read_plugin_state()
    now = time.time()
    current_spec = _plugin_spec()
    current_version = _installed_plugin_version(current_spec)
    current_text = current_version or current_spec
    if force:
        if current_version is not None:
            return False  # installed meanwhile; the caller verifies it
    elif not _plugin_update_due(state, now):
        return False
    stop_check = is_cancelled if force else None

    try:
        target = _fetch_latest_plugin_version()
    except Exception as exc:
        _mark_plugin_retry(state, now)
        _write_plugin_state(state)
        logger(
            f"⚠️ OcAgy: could not check npm for a newer {_PLUGIN_NAME} "
            f"({_error_reason(exc)}); using the installed plugin"
            + (f" {current_version}." if current_version else ".")
        )
        return False
    state["checked_at"] = now
    state.pop("check_failed_at", None)
    if current_version and (_version_tuple(target) or (0, 0, 0)) <= (_version_tuple(current_version) or (0, 0, 0)):
        state.setdefault("spec", current_spec)
        state.setdefault("version", current_version)
        _finish_deferred_cli_update(state, executable, current_version, logger)
        _write_plugin_state(state)
        return False
    if _plugin_version_rejected(state, target, now):
        _write_plugin_state(state)
        return False

    spec = f"{_PLUGIN_NAME}@{target}"
    root = _plugin_root(spec)
    if _installed_plugin_version(spec) != target:
        logger(
            f"⬆️ OcAgy: downloading {_PLUGIN_NAME} {target}"
            + (f" (installed: {current_version})..." if current_version else "...")
        )
        # The throw-away run prints OpenCode's whole Google catalog, and it
        # exits with "Provider not found: google" on OpenCode 1.x even when
        # the download worked (no accounts, no model config). None of that is
        # logged; a real failure is logged once below.
        error = _install_plugin_candidate(
            executable,
            spec,
            _quiet,
            timeout=OCAGY_PLUGIN_INSTALL_TIMEOUT_SECONDS if force else OCAGY_PLUGIN_UPDATE_TIMEOUT_SECONDS,
            stop_check=stop_check,
        )
        if error:
            _mark_plugin_retry(state, now)
            _write_plugin_state(state)
            logger(f"⚠️ OcAgy: could not download {_PLUGIN_NAME} {target} ({_short_reason(error)}). Keeping {current_text}.")
            return False

    required_files = [
        root / "dist" / "src" / "plugin" / "storage.js",
        root / "dist" / "src" / "plugin" / "auth.js",
        root / "dist" / "src" / "antigravity" / "oauth.js",
    ]
    if not all(path.is_file() for path in required_files):
        shutil.rmtree(_plugin_package_dir(spec), ignore_errors=True)
        _mark_plugin_retry(state, now)
        _write_plugin_state(state)
        logger(f"⚠️ OcAgy: the download of {spec} is incomplete; it was removed. Keeping {current_text}.")
        return False
    try:
        patch_ocagy(root)
    except Exception as exc:
        # Never run a plugin that would store OAuth credentials in plaintext.
        shutil.rmtree(_plugin_package_dir(spec), ignore_errors=True)
        _reject_plugin_version(state, target, str(exc), permanent=True)
        _write_plugin_state(state)
        logger(
            f"⚠️ OcAgy: {spec} does not match Glossarion's encrypted account storage hook ({_error_reason(exc)}); "
            f"not switching to it. Keeping {current_text}."
        )
        return False

    required = _plugin_min_opencode(root, target)
    if required:
        cli_version = _opencode_version(executable)
        if cli_version is None or cli_version < required:
            updated = _update_opencode_cli(
                executable,
                cli_version,
                required,
                target,
                current_text,
                logger,
                allow_update=not force,
            )
            if not updated:
                if force:
                    # A first install never runs npm on a request, and there is
                    # no older setup to keep: the request uses this patched
                    # download (OpenCode would otherwise fetch the same release
                    # as @latest, unpatched).
                    state.update(spec=spec, version=target)
                    if updated == "":
                        # The background check that follows the first start
                        # brings OpenCode up to date right away.
                        state["cli_needed"] = {"version": target, "required": _format_version(required)}
                        state.pop("checked_at", None)
                _write_plugin_state(state)
                return False
            executable = updated

    state.pop("cli_needed", None)
    state.update(
        spec=spec,
        version=target,
        previous_spec=current_spec,
        previous_version=current_version,
        updated_at=now,
    )
    _write_plugin_state(state)
    workspace = _workspace_dir()
    # The check's output is OpenCode's whole model catalog: never logged.
    ok, detail = _verify_auth_plugin(executable, _quiet, workspace, stop_check=stop_check)
    if not ok:
        state.update(spec=current_spec, version=current_version)
        _reject_plugin_version(state, target, detail, permanent=False)
        _write_plugin_state(state)
        _workspace_dir()
        logger(f"⚠️ OcAgy: {spec} failed its check ({_short_reason(detail)}); restored {current_text}.")
        return False
    new_models = sorted(
        set(_declared_google_models()) - set(_declared_google_models(current_spec))
    )
    logger(
        f"✅ OcAgy: updated {_PLUGIN_NAME} {current_version or 'not installed'} → {target}."
        + (f" New models: {', '.join(model.replace('antigravity-', '', 1) for model in new_models)}." if new_models and current_version else "")
    )
    return True


def _start_background_plugin_update(executable: str, logger: Callable[[str], None]) -> None:
    """Check npm for a newer plugin at most every OCAGY_PLUGIN_UPDATE_INTERVAL_SECONDS."""
    global _PLUGIN_UPDATE_THREAD
    if _PLUGIN_PACKAGE_OVERRIDE or not _plugin_update_due(_read_plugin_state(), time.time()):
        return
    with _PLUGIN_UPDATE_THREAD_LOCK:
        if _PLUGIN_UPDATE_THREAD is not None and _PLUGIN_UPDATE_THREAD.is_alive():
            return

        def run() -> None:
            try:
                _update_auth_plugin(executable, logger)
            except Exception as exc:  # never disturb translation
                logger(f"⚠️ OcAgy: plugin update check failed: {_error_reason(exc)}")

        _PLUGIN_UPDATE_THREAD = threading.Thread(target=run, name="ocagy-plugin-update", daemon=True)
        _PLUGIN_UPDATE_THREAD.start()


# Google answers a request from a plugin version it no longer accepts with this
# sentence as the model's reply ("This version of Antigravity is no longer
# supported. Please download and install the latest version ..."). It is not a
# translation: the request waits for the plugin update and is sent again.
_UNSUPPORTED_PLUGIN_NOTICE = "this version of antigravity is no longer supported"
# The longest a request waits for that update: plugin download, CLI download,
# CLI install and the check after the switch.
OCAGY_UNSUPPORTED_PLUGIN_WAIT_SECONDS = (
    OCAGY_PLUGIN_UPDATE_TIMEOUT_SECONDS + OCAGY_CLI_DOWNLOAD_TIMEOUT_SECONDS + OCAGY_CLI_UPDATE_TIMEOUT_SECONDS + 300
)
_UNSUPPORTED_PLUGIN_LOCK = threading.Lock()
_UNSUPPORTED_PLUGIN_NOTICE_LOGGED: Dict[str, float] = {}
_UNSUPPORTED_PLUGIN_CHECK_FORCED: Dict[str, float] = {}


def _is_unsupported_plugin_notice(text: Any) -> bool:
    """True when a reply is Google's "this version is no longer supported" sentence."""
    content = _clean_text(str(text or "")).strip()
    return len(content) < 1000 and content.casefold().startswith(_UNSUPPORTED_PLUGIN_NOTICE)


class _UnsupportedNoticeFilter:
    """Keeps that sentence out of the stream log (one copy per request of a batch).

    The first characters of a reply are held back until they can no longer be
    the start of the sentence; everything else passes through unchanged.
    """

    def __init__(self) -> None:
        self.held = ""
        self.is_notice: Optional[bool] = None  # None: not decided yet

    def feed(self, fragment: str) -> str:
        """Return the text to log now ("" while undecided or for the notice)."""
        if self.is_notice is not None:
            return "" if self.is_notice else fragment
        self.held += fragment
        head = self.held.lstrip().casefold()
        if head.startswith(_UNSUPPORTED_PLUGIN_NOTICE):
            self.is_notice = True
            return ""
        if _UNSUPPORTED_PLUGIN_NOTICE.startswith(head):
            return ""
        self.is_notice = False
        text, self.held = self.held, ""
        return text

    def rest(self) -> str:
        """Text still held when the stream ended before a decision (a very short reply)."""
        if self.is_notice is None:
            text, self.held = self.held, ""
            return text
        return ""


def _wait_for_newer_plugin(used_version: Optional[str], executable: str, logger: Callable[[str], None]) -> bool:
    """After Google refused the plugin version a request used: wait for a newer one.

    Checks npm now (whatever the 12 h interval says, once an hour per refused
    version) and waits for the background update, Stop-aware and capped. One
    log line however many requests wait. Returns True when the workspace now
    uses another plugin version, so the request can be sent again.
    """

    def moved_on() -> bool:
        return _installed_plugin_version(_plugin_spec()) != used_version

    if _PLUGIN_PACKAGE_OVERRIDE:
        return False
    if moved_on():
        return True  # updated while this request was running
    key = str(used_version or "")
    now = time.monotonic()
    with _UNSUPPORTED_PLUGIN_LOCK:
        announce = now - _UNSUPPORTED_PLUGIN_NOTICE_LOGGED.get(key, float("-inf")) >= OCAGY_PLUGIN_UPDATE_RETRY_SECONDS
        if announce:
            _UNSUPPORTED_PLUGIN_NOTICE_LOGGED[key] = now
        force = now - _UNSUPPORTED_PLUGIN_CHECK_FORCED.get(key, float("-inf")) >= OCAGY_PLUGIN_UPDATE_RETRY_SECONDS
        if force:
            _UNSUPPORTED_PLUGIN_CHECK_FORCED[key] = now
            thread = _PLUGIN_UPDATE_THREAD
            if thread is None or not thread.is_alive():
                state = _read_plugin_state()
                state.pop("checked_at", None)
                state.pop("check_failed_at", None)
                _write_plugin_state(state)
            # Started under the lock: the other waiting requests then see the thread.
            _start_background_plugin_update(executable, logger)
    if announce:
        logger(
            f"⏳ OcAgy: Google no longer accepts {_PLUGIN_NAME}"
            + (f" {used_version}" if used_version else "")
            + ". Waiting for the plugin update, then sending the request again..."
        )
    deadline = time.monotonic() + OCAGY_UNSUPPORTED_PLUGIN_WAIT_SECONDS
    while not moved_on():
        thread = _PLUGIN_UPDATE_THREAD
        if thread is None or not thread.is_alive() or time.monotonic() >= deadline:
            break
        if is_cancelled():
            raise OcAgyError(_CANCELLED_MESSAGE)
        thread.join(OCAGY_LOCK_POLL_SECONDS)
    return moved_on()


def _unsupported_plugin_error(used_version: Optional[str]) -> OcAgyError:
    return OcAgyError(
        f"Google no longer accepts this version of the Antigravity plugin ({_PLUGIN_NAME}"
        + (f" {used_version}" if used_version else "")
        + "), and Glossarion could not switch to a newer one. The log shows why the plugin update "
        "did not happen; without one, wait for a new plugin release."
    )


def _workspace_dir() -> Path:
    path = _workspace_base()
    path.mkdir(parents=True, exist_ok=True)
    _write_workspace_config(path)
    _ensure_plugin_config()
    return path


_DEFAULT_MODALITIES: Dict[str, List[str]] = {"input": ["text", "image", "pdf"], "output": ["text"]}


def _js_block(source: str, start: int) -> str:
    """Return the ``{...}`` block that opens at or after ``start`` in JS source."""
    opening = source.find("{", start)
    if opening < 0:
        return ""
    depth = 0
    for index in range(opening, len(source)):
        char = source[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return source[opening:index + 1]
    return ""


def _js_model_definition(body: str, name: str) -> Optional[Dict[str, Any]]:
    limit = re.search(r"limit:\s*\{\s*context:\s*(\d+),\s*output:\s*(\d+)", body)
    if not limit:
        return None
    definition: Dict[str, Any] = {
        "name": name.replace("(Antigravity)", "(Antigravity OAuth)"),
        "temperature": True,
        "limit": {"context": int(limit.group(1)), "output": int(limit.group(2))},
        "modalities": json.loads(json.dumps(_DEFAULT_MODALITIES)),
    }
    variants_at = body.find("variants:")
    if variants_at >= 0:
        block = _js_block(body, variants_at)
        variants: Dict[str, Any] = {}
        for variant, level in re.findall(r"(\w+):\s*\{\s*thinkingLevel:\s*\"(\w+)\"\s*\}", block):
            variants[variant] = {"thinkingLevel": level}
        for variant, budget in re.findall(
            r"(\w+):\s*\{\s*thinkingConfig:\s*\{\s*thinkingBudget:\s*(\d+)\s*\}\s*\}", block
        ):
            variants[variant] = {"thinkingConfig": {"thinkingBudget": int(budget)}}
        if variants:
            definition["variants"] = variants
    return definition


_DECLARED_MODELS_CACHE: Dict[str, Tuple[Tuple[float, ...], Dict[str, Dict[str, Any]]]] = {}


def _plugin_declared_models(root: Path) -> Dict[str, Dict[str, Any]]:
    """Model definitions an installed plugin declares for OpenCode 1.x configs.

    OpenCode 1.x only offers models listed in the config; the plugin keeps its
    own list in dist/src/plugin/config/models.js (plus the Gemini 3.8 Flash tier
    loop in models-v2.js). Reading them lets new plugin models appear without a
    Glossarion release. Anything that cannot be parsed is skipped.
    """
    config_dir = Path(root) / "dist" / "src" / "plugin" / "config"
    files = [config_dir / "models.js", config_dir / "models-v2.js"]
    try:
        stamp = tuple(path.stat().st_mtime if path.is_file() else 0.0 for path in files)
    except OSError:
        return {}
    cache_key = str(config_dir).casefold()
    cached = _DECLARED_MODELS_CACHE.get(cache_key)
    if cached is not None and cached[0] == stamp:
        return json.loads(json.dumps(cached[1]))

    models: Dict[str, Dict[str, Any]] = {}
    try:
        source = files[0].read_text(encoding="utf-8")
    except OSError:
        source = ""
    for match in re.finditer(r"\"(antigravity-[a-z0-9][a-z0-9.\-]*)\"\s*:\s*\{", source):
        body = _js_block(source, match.end() - 1)
        name = re.search(r"name:\s*\"([^\"]+)\"", body)
        definition = _js_model_definition(body, name.group(1) if name else match.group(1))
        if definition:
            models[match.group(1)] = definition
    try:
        source_v2 = files[1].read_text(encoding="utf-8")
    except OSError:
        source_v2 = ""
    for match in re.finditer(
        r"for\s*\(const\s+(\w+)\s+of\s+\[([^\]]*)\]\)\s*\{\s*\w+\[`(antigravity-[a-z0-9][a-z0-9.\-]*)-\$\{\1\}`\]\s*=\s*\{",
        source_v2,
    ):
        body = _js_block(source_v2, match.end() - 1)
        template = re.search(r"name:\s*`([^`]+)`", body)
        for tier in re.findall(r"[\"']([a-z]+)[\"']", match.group(2)):
            name = (template.group(1) if template else f"{match.group(3)}-{tier}").replace(
                "${" + match.group(1) + "}", tier
            )
            definition = _js_model_definition(body, name)
            if definition:
                models[f"{match.group(3)}-{tier}"] = definition
    _DECLARED_MODELS_CACHE[cache_key] = (stamp, models)
    return json.loads(json.dumps(models))


def _active_plugin_root(spec: Optional[str] = None) -> Optional[Path]:
    spec = spec or _plugin_spec()
    root = _plugin_root(spec)
    if root.is_dir():
        return root
    for candidate in _plugin_install_candidates():
        if candidate.is_dir():
            return candidate
    return None


def _declared_google_models(spec: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """Google models written into the workspace config: Glossarion's + the plugin's."""
    models = json.loads(json.dumps(_STATIC_GOOGLE_MODELS))
    root = _active_plugin_root(spec)
    if root is not None:
        for model_id, definition in _plugin_declared_models(root).items():
            # Glossarion's own entries keep their names and limits stable.
            models.setdefault(model_id, definition)
    return models


def _workspace_config(*, temperature: Optional[float] = None) -> Dict[str, Any]:
    # Project-local config keeps this integration independent from the user's
    # normal OpenCode projects while still reusing the plugin's account store.
    config: Dict[str, Any] = {
        "$schema": "https://opencode.ai/config.json",
        "plugin": [_plugin_spec()],
        "share": "disabled",
        "permission": {"*": "deny"},
        "agent": {
            "glossarion": {
                "description": "Non-interactive text generation backend for Glossarion",
                "mode": "primary",
                "prompt": (
                    "Act only as a text-generation backend for Glossarion. Never use tools, "
                    "never inspect files or the workspace, never browse, and never explain your process. "
                    "Follow the supplied instructions and return only the requested final text."
                ),
                "permission": {"*": "deny"},
            }
        },
        "provider": {
            "google": {
                # OpenCode only forwards the agent's numeric temperature when
                # the selected custom model advertises temperature support.
                "models": _declared_google_models(),
            }
        },
    }
    if temperature is not None:
        config["agent"]["glossarion"]["temperature"] = float(temperature)
    return config


# Glossarion's fixed entries. Models the installed plugin declares are added to
# these by _declared_google_models(), so new plugin models need no release.
_STATIC_GOOGLE_MODELS: Dict[str, Dict[str, Any]] = {
                    "antigravity-gemini-3.1-pro": {
                        "name": "Gemini 3.1 Pro (Antigravity OAuth)",
                        "temperature": True,
                        "limit": {"context": 1048576, "output": 65535},
                        "modalities": {"input": ["text", "image", "pdf"], "output": ["text"]},
                        "variants": {
                            "low": {"thinkingLevel": "low"},
                            "high": {"thinkingLevel": "high"},
                        },
                    },
                    "antigravity-gemini-3-pro": {
                        "name": "Gemini 3 Pro (Antigravity OAuth)",
                        "temperature": True,
                        "limit": {"context": 1048576, "output": 65535},
                        "modalities": {"input": ["text", "image", "pdf"], "output": ["text"]},
                        "variants": {
                            "low": {"thinkingLevel": "low"},
                            "high": {"thinkingLevel": "high"},
                        },
                    },
                    "antigravity-gemini-3-flash": {
                        "name": "Gemini 3 Flash (Antigravity OAuth)",
                        "temperature": True,
                        "limit": {"context": 1048576, "output": 65536},
                        "modalities": {"input": ["text", "image", "pdf"], "output": ["text"]},
                        "variants": {
                            "minimal": {"thinkingLevel": "minimal"},
                            "low": {"thinkingLevel": "low"},
                            "medium": {"thinkingLevel": "medium"},
                            "high": {"thinkingLevel": "high"},
                        },
                    },
                    "antigravity-claude-sonnet-4-6": {
                        "name": "Claude Sonnet 4.6 (Antigravity OAuth)",
                        "temperature": True,
                        "limit": {"context": 200000, "output": 64000},
                        "modalities": {"input": ["text", "image", "pdf"], "output": ["text"]},
                    },
                    "antigravity-claude-opus-4-6-thinking": {
                        "name": "Claude Opus 4.6 Thinking (Antigravity OAuth)",
                        "temperature": True,
                        "limit": {"context": 200000, "output": 64000},
                        "modalities": {"input": ["text", "image", "pdf"], "output": ["text"]},
                        "variants": {
                            "low": {"thinkingConfig": {"thinkingBudget": 8192}},
                            "max": {"thinkingConfig": {"thinkingBudget": 32768}},
                        },
                    },
}
# Plugin 2.x serves Gemini 3.8 Flash as three Google model IDs; the thinking
# tier is part of the ID, so these entries have no variants.
for _tier in ("low", "medium", "high"):
    _STATIC_GOOGLE_MODELS[f"antigravity-gemini-3.8-flash-{_tier}"] = {
        "name": f"Gemini 3.8 Flash ({_tier}) (Antigravity OAuth)",
        "temperature": True,
        "limit": {"context": 1048576, "output": 65536},
        "modalities": {"input": ["text", "image", "pdf"], "output": ["text"]},
    }
del _tier


def _write_workspace_config(path: Path, *, temperature: Optional[float] = None) -> Path:
    target = path / "opencode.json"
    desired = _workspace_config(temperature=temperature)
    current: Any = None
    if target.is_file():
        try:
            current = json.loads(target.read_text(encoding="utf-8"))
        except Exception:
            current = None
    if current != desired:
        target.write_text(json.dumps(desired, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target


def _model_output_limit(model_id: str) -> Optional[int]:
    """Return the configured hard output ceiling for one OpenCode model."""
    try:
        provider_id, provider_model = str(model_id).split("/", 1)
        value = _workspace_config()["provider"][provider_id]["models"][provider_model]["limit"]["output"]
        return int(value) if int(value) > 0 else None
    except (KeyError, TypeError, ValueError):
        return None


def _effective_output_limit(model_id: str, requested_max_tokens: int) -> int:
    requested = max(1, int(requested_max_tokens))
    model_limit = _model_output_limit(model_id)
    return min(requested, model_limit) if model_limit is not None else requested


def _log_generation_controls(
    logger: Callable[[str], None],
    *,
    model_id: str,
    temperature: float,
    max_tokens: int,
) -> None:
    requested = max(1, int(max_tokens))
    effective = _effective_output_limit(model_id, requested)
    if effective < requested:
        logger(
            "🎛️ Glossarion controls OpenCode generation: "
            f"temperature={temperature}, max_tokens={effective:,} "
            f"(requested {requested:,}; model cap {effective:,})."
        )
        return
    logger(
        "🎛️ Glossarion controls OpenCode generation: "
        f"temperature={temperature}, max_tokens={effective:,}."
    )


def _config_dir() -> Path:
    configured = str(os.environ.get("OPENCODE_CONFIG_DIR", "") or "").strip()
    if configured:
        return Path(configured).expanduser()
    modern = Path.home() / ".config" / "opencode"
    legacy = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming")) / "opencode"
    if (modern / "antigravity-accounts.json").exists() or not (legacy / "antigravity-accounts.json").exists():
        return modern
    return legacy


def _ensure_plugin_config() -> Dict[str, Any]:
    """Enable PID-based account offsets for Glossarion's parallel workers.

    Existing plugin settings are preserved. If a user's file contains JSONC or
    otherwise cannot be parsed safely, it is left untouched.
    """
    config_dir = _config_dir()
    path = config_dir / "antigravity.json"
    result: Dict[str, Any] = {"plugin_config": str(path), "pid_offset_enabled": False}
    try:
        config_dir.mkdir(parents=True, exist_ok=True)
        payload: Any = {}
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            payload = {}
        if "pid_offset_enabled" not in payload:
            payload["pid_offset_enabled"] = True
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        result["pid_offset_enabled"] = bool(payload.get("pid_offset_enabled"))
    except Exception as exc:
        result["plugin_config_error"] = str(exc)
    return result


def _account_summary() -> Dict[str, Any]:
    path = _config_dir() / "antigravity-accounts.json"
    result: Dict[str, Any] = {"accounts_file": str(path), "account_count": 0, "emails": []}
    if not path.is_file():
        return result
    try:
        from proxy_token_storage import load_accounts
        payload = load_accounts(path)
        accounts = payload.get("accounts", []) if isinstance(payload, dict) else []
        enabled = [item for item in accounts if isinstance(item, dict) and item.get("enabled", True)]
        result["account_count"] = len(enabled)
        result["emails"] = [str(item.get("email", "") or "") for item in enabled if item.get("email")]
    except Exception as exc:
        result["accounts_error"] = str(exc)
    return result


def get_account_summary() -> Dict[str, Any]:
    """Return the plugin's local OAuth account summary without exposing tokens."""
    return _account_summary()


def _load_account_store() -> Tuple[Path, Dict[str, Any]]:
    """Read the plugin account store used by the normal OpenCode profile."""
    path = _config_dir() / "antigravity-accounts.json"
    if not path.is_file():
        return path, {}
    try:
        from proxy_token_storage import load_accounts
        payload = load_accounts(path)
    except Exception as exc:
        raise OcAgyError(f"The OpenCode Antigravity OAuth account store could not be read: {exc}") from exc
    if not isinstance(payload, dict):
        raise OcAgyError("The OpenCode Antigravity OAuth account store has an invalid format.")
    return path, payload


def _select_account_slot(account_number: int) -> Tuple[Dict[str, Any], str]:
    """Return a specific saved account using Antigravity's one-based slot order."""
    _path, payload = _load_account_store()
    accounts = payload.get("accounts", [])
    if not isinstance(accounts, list):
        accounts = []
    index = account_number - 1
    if index < 0 or index >= len(accounts):
        raise OcAgyError(
            f"OcAgy account #{account_number} is not linked. Click OCAGY Login, add that "
            "Google account, then retry. Use ocagy0/ if you want the plugin-managed shared pool."
        )
    account = accounts[index]
    if not isinstance(account, dict) or not account.get("refreshToken"):
        raise OcAgyError(
            f"OcAgy account #{account_number} is not usable because its OAuth credential is missing. "
            "Sign in again with OCAGY Login."
        )
    if not account.get("enabled", True):
        raise OcAgyError(
            f"OcAgy account #{account_number} is disabled in the OpenCode Antigravity account store. "
            "Enable it or use another numbered OcAgy prefix."
        )
    email = str(account.get("email", "") or "").strip()
    return account, email


def _require_oauth_account(account_number: int = 0) -> Optional[Tuple[Dict[str, Any], str]]:
    """Fail before launching OpenCode when the requested OAuth account is unavailable."""
    if account_number > 0:
        return _select_account_slot(account_number)

    account = _account_summary()
    if account.get("account_count"):
        return None

    detail = ""
    if account.get("accounts_error"):
        detail = f" The OAuth account store could not be read: {account['accounts_error']}"
    raise OcAgyError(
        "OpenCode Antigravity OAuth is not configured. The ocagy/ route does not use or require "
        "a Google API key. Click the OCAGY Login button, select Google, then "
        "OAuth with Google (Antigravity), finish signing in, and retry. Do not select "
        f"Manually enter API Key.{detail}"
    )


def _write_private_json(path: Path, payload: Dict[str, Any]) -> None:
    if path.name == "antigravity-accounts.json":
        from proxy_token_storage import save_accounts
        save_accounts(path, payload)
        return
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        pass


def _prepare_isolated_account_config(
    request_dir: Path,
    account_number: int,
    selected_account: Dict[str, Any],
) -> Path:
    """Create a request-local config containing only the explicitly selected account."""
    config_dir = request_dir / f".ocagy-account-{account_number}"
    config_dir.mkdir(parents=True, exist_ok=False)
    try:
        config_dir.chmod(0o700)
    except OSError:
        pass

    # Deep-copy through JSON so the shared account object is never mutated
    # while forming the isolated plugin store.
    account_copy = json.loads(json.dumps(selected_account))
    _source_path, source_store = _load_account_store()
    account_store = {
        "version": source_store.get("version", 1),
        "accounts": [account_copy],
        "activeIndex": 0,
        "activeIndexByFamily": {"gemini": 0, "claude": 0},
    }
    _write_private_json(config_dir / "antigravity-accounts.json", account_store)

    plugin_settings: Dict[str, Any] = {}
    shared_settings = _config_dir() / "antigravity.json"
    try:
        if shared_settings.is_file():
            loaded = json.loads(shared_settings.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                plugin_settings = loaded
    except Exception:
        # The shared file may be JSONC. The isolated request only needs the
        # deterministic account-selection overrides below.
        plugin_settings = {}
    plugin_settings["pid_offset_enabled"] = False
    plugin_settings["account_selection_strategy"] = "sticky"
    _write_private_json(config_dir / "antigravity.json", plugin_settings)
    return config_dir


def _request_subprocess_env(
    request_dir: Path,
    account_number: int,
    selected: Optional[Tuple[Dict[str, Any], str]],
    logger: Callable[[str], None],
    max_tokens: int,
) -> Dict[str, str]:
    if selected is None:
        logger("🧭 OcAgy: using shared plugin account pool (ocagy0/)")
        env = _subprocess_env()
        # OpenCode otherwise applies its own global output ceiling before the
        # model-specific limit in opencode.json.
        env["OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX"] = str(max(1, int(max_tokens)))
        return env

    selected_account, email = selected
    isolated_config = _prepare_isolated_account_config(
        request_dir,
        account_number,
        selected_account,
    )
    logger(
        f"🧭 OcAgy: using account slot #{account_number}"
        + (f" ({email})" if email else "")
    )
    env = _subprocess_env(isolated_config)
    env["OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX"] = str(max(1, int(max_tokens)))
    return env


def _creation_flags(*, visible: bool = False) -> int:
    if os.name != "nt":
        return 0
    if visible:
        return getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
    return getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)


def _subprocess_env(config_dir: Optional[Path] = None) -> Dict[str, str]:
    env = dict(os.environ)
    if os.name == "nt":
        # winget may install Node.js while Glossarion is already running. Its
        # process PATH is then stale, so npm's opencode.cmd cannot find node.exe.
        home = Path.home()
        local_appdata = Path(env.get("LOCALAPPDATA", home / "AppData" / "Local"))
        program_files = Path(env.get("ProgramFiles", "C:/Program Files"))
        node_dirs = (program_files / "nodejs", local_appdata / "Programs" / "nodejs")
        path_entries = env.get("PATH", "").split(os.pathsep)
        existing = {entry.rstrip("\\/").casefold() for entry in path_entries}
        additions = [str(folder) for folder in node_dirs if (folder / "node.exe").is_file() and str(folder).casefold() not in existing]
        if additions:
            env["PATH"] = os.pathsep.join([*additions, env.get("PATH", "")])
    env.setdefault("NO_COLOR", "1")
    env.setdefault("TERM", "dumb")
    env.setdefault("PYTHONUTF8", "1")
    if config_dir is not None:
        env["OPENCODE_CONFIG_DIR"] = str(config_dir)
    return env


def _terminate_process_tree(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                capture_output=True,
                timeout=5,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                check=False,
            )
        else:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except Exception:
                proc.terminate()
    except Exception:
        try:
            proc.terminate()
        except Exception:
            pass
    try:
        proc.wait(timeout=2)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def _clean_text(value: str) -> str:
    return _ANSI_RE.sub("", str(value or "")).replace("\r\n", "\n").replace("\r", "\n").strip()


def _run_short(args: List[str], timeout: int = 60) -> subprocess.CompletedProcess:
    """Run a short OpenCode command; on timeout its whole process tree is killed."""
    exe = find_executable()
    workspace = _workspace_dir()
    command = [exe, *args]
    proc = _spawn_opencode(
        command,
        # A model poll waits for a running CLI update no longer than its own timeout.
        wait_timeout=timeout,
        should_stop=None,
        cwd=str(workspace),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=_creation_flags(),
        start_new_session=(os.name != "nt"),
        env=_subprocess_env(),
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _terminate_process_tree(proc)
        try:
            proc.communicate(timeout=5)
        except Exception:
            pass
        raise subprocess.TimeoutExpired(command, timeout) from None
    return subprocess.CompletedProcess(command, proc.returncode, stdout, stderr)


def get_status() -> Dict[str, Any]:
    """Return OpenCode installation, plugin model visibility, and account count."""
    try:
        exe = find_executable()
    except Exception as exc:
        return {"installed": False, "authenticated": False, "plugin_ready": False, "error": str(exc), "models": []}

    version_text = ""
    try:
        version = _run_short(["--version"], timeout=20)
        version_text = _clean_text(version.stdout or version.stderr)
    except Exception as exc:
        version_text = str(exc)

    models: List[str] = []
    models_error = ""
    plugin_ready = False
    try:
        listing = _run_short(["models", "google"], timeout=120)
        combined = _clean_text((listing.stdout or "") + "\n" + (listing.stderr or ""))
        if listing.returncode == 0:
            models = sorted(set(re.findall(r"google/[A-Za-z0-9._-]+", combined)))
            plugin_ready = any("antigravity-" in item for item in models)
        else:
            models_error = combined or f"opencode models exited with code {listing.returncode}"
    except Exception as exc:
        models_error = str(exc)

    account = _account_summary()
    plugin_config = _ensure_plugin_config()
    return {
        "installed": True,
        "authenticated": bool(account.get("account_count")),
        "plugin_ready": plugin_ready,
        "executable": exe,
        "version": version_text,
        "models": models,
        "error": models_error,
        **account,
        **plugin_config,
    }


def _quota_form_json(url: str, values: Dict[str, str], timeout: float) -> Dict[str, Any]:
    data = urllib.parse.urlencode(values).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read()
    payload = json.loads(raw.decode("utf-8", errors="replace")) if raw else {}
    return payload if isinstance(payload, dict) else {}


def _plugin_install_candidates() -> Iterable[Path]:
    """Yield likely roots for the OpenCode-owned Antigravity auth plugin."""
    override = str(os.environ.get("OCAGY_PLUGIN_ROOT", "") or "").strip()
    if override:
        yield Path(override).expanduser()

    package_name = _PLUGIN_NAME
    cache_base = _opencode_cache_dir()
    # The configured version first: it is the one OpenCode loads.
    yield _plugin_root(_plugin_spec())
    yield cache_base / "node_modules" / package_name
    yield cache_base / "packages" / _PLUGIN_PACKAGE / "node_modules" / package_name
    packages_dir = cache_base / "packages"
    if packages_dir.is_dir():
        yield from sorted(
            packages_dir.glob(f"{package_name}@*/node_modules/{package_name}"),
            reverse=True,
        )


def _secure_plugin_account_storage(*, required):
    from proxy_token_storage import patch_ocagy, load_accounts, migrate_opencode_oauth
    roots = [root for root in _plugin_install_candidates()
             if (root / "dist" / "src" / "plugin" / "storage.js").is_file()]
    if not roots:
        if required:
            raise OcAgyError("Cannot locate the OAuth plugin to enable encrypted credential storage")
        return
    active = _plugin_root(_plugin_spec())
    active_installed = (active / "dist" / "src" / "plugin" / "storage.js").is_file()
    for root in dict.fromkeys(roots):
        try:
            patch_ocagy(root)
        except Exception:
            # Other cached versions are not loaded; only the configured one
            # (or every one, when it is missing) must carry the hook.
            if not active_installed or root == active:
                raise
    load_accounts(_config_dir() / "antigravity-accounts.json", migrate=True)
    data_home = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    migrate_opencode_oauth(data_home / "opencode" / "auth.json")


def _plugin_oauth_client_credentials() -> Tuple[str, str]:
    """Read OAuth application metadata from the plugin that owns authentication.

    The desktop OAuth application's metadata ships with the installed plugin.
    Glossarion must not duplicate those values in its own source or Git history.
    """
    configured_id = str(os.environ.get("OCAGY_GOOGLE_CLIENT_ID", "") or "").strip()
    configured_secret = str(os.environ.get("OCAGY_GOOGLE_CLIENT_SECRET", "") or "").strip()
    if configured_id and configured_secret:
        return configured_id, configured_secret

    patterns = {
        "client_id": re.compile(
            r"\bANTIGRAVITY_CLIENT_ID\s*=\s*['\"]([^'\"]+)['\"]"
        ),
        "client_secret": re.compile(
            r"\bANTIGRAVITY_CLIENT_SECRET\s*=\s*['\"]([^'\"]+)['\"]"
        ),
    }
    for root in _plugin_install_candidates():
        for constants_path in (
            root / "dist" / "src" / "constants.js",
            root / "src" / "constants.ts",
        ):
            if not constants_path.is_file():
                continue
            try:
                source = constants_path.read_text(encoding="utf-8")
            except OSError:
                continue
            client_id_match = patterns["client_id"].search(source)
            client_secret_match = patterns["client_secret"].search(source)
            if client_id_match and client_secret_match:
                return client_id_match.group(1), client_secret_match.group(1)

    raise OcAgyError(
        "Could not read OAuth application metadata from the installed "
        "opencode-antigravity-auth plugin. Restart OpenCode once so the plugin "
        "is installed, or set OCAGY_PLUGIN_ROOT to its package directory."
    )


def _quota_post_json(
    url: str,
    access_token: str,
    payload: Dict[str, Any],
    timeout: float,
    *,
    user_agent: str = _ANTIGRAVITY_QUOTA_USER_AGENT,
) -> Dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "User-Agent": user_agent,
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read()
    result = json.loads(raw.decode("utf-8", errors="replace")) if raw else {}
    return result if isinstance(result, dict) else {}


def _quota_error_message(error: BaseException) -> str:
    """Return bounded quota-check detail without reflecting credentials."""
    if isinstance(error, urllib.error.HTTPError):
        detail = ""
        try:
            body = error.read(4096).decode("utf-8", errors="replace")
            payload = json.loads(body)
            candidate = payload.get("error", payload) if isinstance(payload, dict) else payload
            if isinstance(candidate, dict):
                detail = str(
                    candidate.get("message")
                    or candidate.get("error_description")
                    or candidate.get("status")
                    or ""
                )
            elif isinstance(candidate, str):
                detail = candidate
        except Exception:
            detail = ""
        clean_detail = re.sub(r"\s+", " ", detail).strip()[:200]
        suffix = f" — {clean_detail}" if clean_detail else ""
        return f"HTTP {int(getattr(error, 'code', 0) or 0)}{suffix}"
    return re.sub(r"\s+", " ", str(error or "")).strip()[:240] or type(error).__name__


def _quota_reset_timestamp(value: Any) -> Optional[float]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    except (TypeError, ValueError):
        return None


def _quota_reset_in(value: Any) -> str:
    timestamp = _quota_reset_timestamp(value)
    if timestamp is None:
        return ""
    remaining = max(0, int(round(timestamp - time.time())))
    if remaining <= 0:
        return "now"
    days, remainder = divmod(remaining, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes = remainder // 60
    if days:
        return f"{days}d {hours}h" if hours else f"{days}d"
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


def _normalize_quota_fraction(value: Any) -> Optional[float]:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    return max(0.0, min(1.0, float(value)))


def _aggregate_antigravity_quota(models: Any) -> Dict[str, Dict[str, Any]]:
    groups: Dict[str, Dict[str, Any]] = {}
    if not isinstance(models, dict):
        return groups
    for model_name, raw_info in models.items():
        info = raw_info if isinstance(raw_info, dict) else {}
        model_id = str(model_name or "").strip().lower()
        # Same selection and grouping as the plugin (2.x): chat models only,
        # Gemini split into Flash and Pro, so Gemini 3.8 Flash and later
        # Gemini generations are counted too.
        if info.get("disabled") or not re.match(r"(gemini-|claude-)", model_id):
            continue
        if re.search(r"image|imagen|-(disabled|tiered)$", model_id):
            continue
        group = "claude" if "claude" in model_id else ("gemini-flash" if "flash" in model_id else "gemini-pro")
        quota = info.get("quotaInfo")
        if not isinstance(quota, dict):
            continue
        remaining = _normalize_quota_fraction(quota.get("remainingFraction"))
        reset_time = str(quota.get("resetTime", "") or "")
        existing = groups.setdefault(group, {"model_count": 0, "models": []})
        existing["model_count"] += 1
        model_entry: Dict[str, Any] = {
            "model_id": str(model_name),
            "label": str(info.get("displayName") or model_name),
        }
        if remaining is not None:
            model_entry["remaining_fraction"] = remaining
        if _quota_reset_timestamp(reset_time) is not None:
            model_entry["reset_time"] = reset_time
            model_entry["reset_in"] = _quota_reset_in(reset_time)
        existing["models"].append(model_entry)
        if remaining is not None:
            current = existing.get("remaining_fraction")
            existing["remaining_fraction"] = remaining if current is None else min(current, remaining)
        reset_timestamp = _quota_reset_timestamp(reset_time)
        existing_timestamp = _quota_reset_timestamp(existing.get("reset_time"))
        if reset_timestamp is not None and (
            existing_timestamp is None or reset_timestamp < existing_timestamp
        ):
            existing["reset_time"] = reset_time
            existing["reset_in"] = _quota_reset_in(reset_time)
    for group in groups.values():
        group["models"].sort(key=lambda item: str(item["label"]).casefold())
    return groups


def _aggregate_gemini_cli_quota(buckets: Any) -> List[Dict[str, Any]]:
    models: List[Dict[str, Any]] = []
    if not isinstance(buckets, list):
        return models
    for bucket in buckets:
        if not isinstance(bucket, dict):
            continue
        model_id = str(bucket.get("modelId", "") or "").strip()
        if not (model_id.startswith("gemini-3-") or model_id == "gemini-2.5-pro"):
            continue
        remaining = _normalize_quota_fraction(bucket.get("remainingFraction"))
        entry: Dict[str, Any] = {"model_id": model_id}
        if remaining is not None:
            entry["remaining_fraction"] = remaining
        reset_time = str(bucket.get("resetTime", "") or "")
        if reset_time:
            entry["reset_time"] = reset_time
            entry["reset_in"] = _quota_reset_in(reset_time)
        models.append(entry)
    return sorted(models, key=lambda item: str(item.get("model_id", "")).casefold())


def _check_account_quota(account: Dict[str, Any], index: int, timeout: float) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "index": index,
        "email": str(account.get("email", "") or "") or f"Account {index + 1}",
        "disabled": account.get("enabled", True) is False,
        "status": "error",
    }
    refresh_token = str(account.get("refreshToken", "") or "").strip()
    if not refresh_token:
        result["error"] = "Stored OAuth account has no refresh token"
        return result
    try:
        client_id, client_secret = _plugin_oauth_client_credentials()
        token_payload = _quota_form_json(
            "https://oauth2.googleapis.com/token",
            {
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": client_id,
                "client_secret": client_secret,
            },
            timeout,
        )
        access_token = str(token_payload.get("access_token", "") or "").strip()
        if not access_token:
            raise OcAgyError("Google did not return an OAuth access token")

        project_id = str(
            account.get("managedProjectId") or account.get("projectId") or ""
        ).strip()
        if not project_id:
            try:
                project_payload = _quota_post_json(
                    f"{_ANTIGRAVITY_CLOUD_CODE_BASE}/v1internal:loadCodeAssist",
                    access_token,
                    {"metadata": {"ideType": "ANTIGRAVITY"}},
                    timeout,
                )
                project_value = project_payload.get("cloudaicompanionProject")
                if isinstance(project_value, str):
                    project_id = project_value
                elif isinstance(project_value, dict):
                    project_id = str(project_value.get("id", "") or "")
            except Exception:
                project_id = ""
        project_id = project_id or _ANTIGRAVITY_FALLBACK_PROJECT_ID
        body = {"project": project_id}

        try:
            antigravity_payload = _quota_post_json(
                f"{_ANTIGRAVITY_CLOUD_CODE_BASE}/v1internal:fetchAvailableModels",
                access_token,
                body,
                timeout,
            )
            result["antigravity"] = _aggregate_antigravity_quota(
                antigravity_payload.get("models")
            )
            if not result["antigravity"]:
                result["antigravity_error"] = "No Antigravity quota information returned"
        except Exception as exc:
            result["antigravity"] = {}
            result["antigravity_error"] = _quota_error_message(exc)

        try:
            cli_payload = _quota_post_json(
                f"{_ANTIGRAVITY_CLOUD_CODE_BASE}/v1internal:retrieveUserQuota",
                access_token,
                body,
                timeout,
                user_agent="GeminiCLI/1.0.0/gemini-2.5-pro",
            )
            result["gemini_cli"] = _aggregate_gemini_cli_quota(cli_payload.get("buckets"))
            if not result["gemini_cli"]:
                result["gemini_cli_error"] = "No Gemini CLI quota information returned"
        except Exception as exc:
            result["gemini_cli"] = []
            result["gemini_cli_error"] = _quota_error_message(exc)

        result["status"] = "ok"
        return result
    except Exception as exc:
        result["error"] = _quota_error_message(exc)
        return result


def get_quota_status(timeout: float = 15.0, log_fn=None) -> Dict[str, Any]:
    """Fetch sanitized live quota data using the plugin's stored OAuth accounts."""
    executable = ensure_opencode_installed(log_fn=log_fn)
    ensure_auth_plugin_installed(executable, log_fn=log_fn)
    _require_oauth_account()
    account_path = _config_dir() / "antigravity-accounts.json"
    try:
        from proxy_token_storage import load_accounts
        payload = load_accounts(account_path)
    except Exception as exc:
        raise OcAgyError(f"Could not read the OpenCode Antigravity account store: {exc}") from exc
    accounts = payload.get("accounts", []) if isinstance(payload, dict) else []
    if not isinstance(accounts, list):
        accounts = []
    timeout = max(1.0, float(timeout))
    results = [
        _check_account_quota(account, index, timeout)
        for index, account in enumerate(accounts)
        if isinstance(account, dict)
    ]
    enabled_count = sum(1 for item in results if not item.get("disabled"))
    return {
        "installed": True,
        "authenticated": enabled_count > 0,
        "plugin_ready": True,
        "executable": executable,
        "account_count": enabled_count,
        "emails": [item["email"] for item in results if not item.get("disabled")],
        "quota_accounts": results,
        "checked_at": time.time(),
    }


def normalize_polled_models(raw_models: Iterable[str]) -> List[str]:
    """Convert OpenCode plugin model IDs into selectable Glossarion IDs."""
    available = {
        str(model or "").strip().casefold(): str(model or "").strip()
        for model in raw_models
        if str(model or "").strip().casefold().startswith("google/antigravity-")
    }
    result: List[str] = []
    known = set()
    for raw_model, friendly, variants in _CATALOG_MODEL_VARIANTS:
        key = raw_model.casefold()
        known.add(key)
        if key not in available:
            continue
        if variants:
            result.extend(f"ocagy/{friendly}-{variant}" for variant in variants)
        else:
            result.append(f"ocagy/{friendly}")

    # Models the installed plugin declares get the same friendly form (variants
    # expanded); anything else keeps its antigravity-* suffix, which
    # resolve_model() accepts as is.
    declared = {key.casefold(): value for key, value in _declared_google_models().items()}
    for key in sorted(set(available) - known):
        provider_model = available[key].split("/", 1)[1]
        definition = declared.get(provider_model.casefold())
        if definition is None:
            result.append(f"ocagy/{provider_model}")
            continue
        friendly = provider_model[len("antigravity-"):]
        variants = list((definition.get("variants") or {}).keys())
        if variants:
            result.extend(f"ocagy/{friendly}-{variant}" for variant in variants)
        else:
            result.append(f"ocagy/{friendly}")
    return list(dict.fromkeys(result))


def poll_zen_models(timeout: float = 8.0) -> List[str]:
    """Poll free Zen models recognized by the CLI used to send ocz/ requests."""
    try:
        listing = _run_short(["models", "opencode"], timeout=max(1, int(round(float(timeout)))))
    except subprocess.TimeoutExpired as exc:
        raise OcAgyError("OpenCode Zen model polling timed out") from exc
    if listing.returncode != 0:
        detail = _zen_error_detail(listing.stderr or listing.stdout or '')
        raise OcAgyError(f"OpenCode Zen model polling failed: {detail}")
    # Only accept model IDs printed as catalog rows, never IDs in diagnostic logs.
    rows = _clean_text(listing.stdout or '').splitlines()
    models = list(dict.fromkeys(
        'ocz/' + row.strip().split('/', 1)[1]
        for row in rows
        if re.fullmatch(r'opencode/[A-Za-z0-9._-]+-free', row.strip())
    ))
    if not models:
        raise OcAgyError("OpenCode returned no usable free Zen model IDs")
    return models


def _signed_in_providers() -> List[str]:
    """Provider IDs OpenCode holds a login for (auth.json keys; secrets are not read)."""
    data_home = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    try:
        payload = json.loads((data_home / "opencode" / "auth.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [str(key) for key in payload] if isinstance(payload, dict) else []


def poll_models(timeout: float = 8.0) -> List[str]:
    """Poll every model the ocagy/ route can run.

    That is the Antigravity OAuth models (google/antigravity-*) plus all models
    of each other provider OpenCode is signed in to through OCAGY Login,
    offered as ocagy/<provider>/<model>.
    """
    others = [name for name in _signed_in_providers() if name != "google"]
    try:
        _require_oauth_account()
        has_google = True
    except OcAgyError:
        if not others:
            raise
        has_google = False
    seconds = max(1, int(round(float(timeout))))
    models: List[str] = []
    if has_google:
        try:
            listing = _run_short(["models", "google"], timeout=seconds)
        except subprocess.TimeoutExpired as exc:
            raise OcAgyError("OpenCode model polling timed out") from exc
        combined = _clean_text((listing.stdout or "") + "\n" + (listing.stderr or ""))
        if listing.returncode != 0:
            raise _classify_error(
                combined or f"opencode models exited with code {listing.returncode}",
                listing.returncode,
            )
        models = normalize_polled_models(re.findall(r"google/[A-Za-z0-9._-]+", combined))
    for provider in others:
        # Each extra provider is optional: its failure must not hide the others.
        try:
            listing = _run_short(["models", provider], timeout=seconds)
        except subprocess.TimeoutExpired:
            continue
        if listing.returncode != 0:
            continue
        # Only catalog rows, never model IDs that appear inside diagnostic log lines.
        # Zen's free models refuse this route ("free tier can only be used from
        # within OpenCode"); ocz/ runs them.
        models.extend(
            f"ocagy/{row.strip()}"
            for row in _clean_text(listing.stdout or "").splitlines()
            if re.fullmatch(re.escape(provider) + r"/[A-Za-z0-9._:-]+", row.strip())
            and not row.strip().endswith("-free")
        )
    models = list(dict.fromkeys(models))
    if not models:
        raise OcAgyError("OpenCode returned no usable model IDs for the ocagy/ route")
    return models


def launch_login(log_fn=None) -> Dict[str, Any]:
    """Open OpenCode's plugin-aware OAuth login in a visible terminal."""
    exe = ensure_opencode_installed(log_fn=log_fn)
    ensure_auth_plugin_installed(exe, log_fn=log_fn)
    workspace = _workspace_dir()
    if os.name == "nt":
        # The project-local opencode.json already declares the plugin and models.
        command = (
            "$Host.UI.RawUI.WindowTitle='Glossarion OCAGY Login'; "
            "Write-Host 'Glossarion OCAGY OAuth setup' -ForegroundColor Cyan; "
            "Write-Host 'Select Google, then OAuth with Google (Antigravity).' -ForegroundColor Yellow; "
            "Write-Host 'Quotas: use the chart (quota) button next to OCAGY Login in Glossarion, not the plugin menu.' -ForegroundColor Yellow; "
            f"Set-Location -LiteralPath {json.dumps(str(workspace))}; "
            f"& {json.dumps(exe)} auth login; "
            "Write-Host ''; Write-Host 'When login is complete, return to Glossarion. The account count updates automatically; click the quota button for live usage.' -ForegroundColor Green"
        )
        proc = subprocess.Popen(
            ["powershell.exe", "-NoExit", "-ExecutionPolicy", "Bypass", "-Command", command],
            cwd=str(workspace),
            creationflags=_creation_flags(visible=True),
            env=_subprocess_env(),
        )
    else:
        shell_command = f"cd {shlex_quote(str(workspace))}; {shlex_quote(exe)} auth login; exec $SHELL"
        terminal = shutil.which("x-terminal-emulator") or shutil.which("gnome-terminal") or shutil.which("konsole")
        if terminal:
            if Path(terminal).name == "gnome-terminal":
                proc = subprocess.Popen([terminal, "--", "sh", "-lc", shell_command])
            else:
                proc = subprocess.Popen([terminal, "-e", "sh", "-lc", shell_command])
        else:
            proc = subprocess.Popen([exe, "auth", "login"], cwd=str(workspace), start_new_session=True)
    return {"started": True, "pid": proc.pid, "executable": exe, "workspace": str(workspace)}


def shlex_quote(value: str) -> str:
    import shlex
    return shlex.quote(value)


def _message_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                kind = str(item.get("type", "") or "").lower()
                if kind in {"text", "input_text", "output_text"} or "text" in item:
                    parts.append(str(item.get("text", "") or ""))
                elif kind in {"image", "image_url", "input_image"}:
                    parts.append("[Image input omitted: OpenCode Antigravity text mode]")
        return "\n".join(part for part in parts if part)
    if isinstance(content, dict):
        if "text" in content:
            return str(content.get("text", "") or "")
        return json.dumps(content, ensure_ascii=False)
    return str(content)


def build_prompt(messages: List[Dict[str, Any]]) -> str:
    system_sections: List[str] = []
    conversation: List[str] = []
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "user") or "user").strip().lower()
        text = _message_text(message.get("content"))
        if role in {"system", "developer"}:
            system_sections.append(text)
        else:
            label = "ASSISTANT" if role == "assistant" else "USER"
            conversation.append(f"<{label}>\n{text}\n</{label}>")
    return (
        "You are being called by Glossarion as a non-interactive text-generation backend.\n"
        "Never use tools. Do not inspect files, the current directory, or external resources.\n"
        "Do not explain your process. Return only the final text requested by the instructions.\n\n"
        "<SYSTEM_INSTRUCTIONS>\n"
        + "\n\n".join(section for section in system_sections if section)
        + "\n</SYSTEM_INSTRUCTIONS>\n\n<CONVERSATION>\n"
        + "\n\n".join(conversation)
        + "\n</CONVERSATION>\n"
    )


def parse_account_route(model: str) -> Tuple[int, str]:
    """Return ``(account_number, model_suffix)`` for an OcAgy route.

    Account number 0 means the plugin-managed shared pool. Unprefixed model
    suffixes are also treated as pooled for backwards compatibility with
    callers that invoke this module directly.
    """
    value = str(model or "").strip()
    match = re.fullmatch(r"ocagy(?P<number>\d{1,4})?(?:/(?P<suffix>.*))?", value, re.IGNORECASE)
    if not match:
        return 0, value
    number = match.group("number")
    account_number = 1 if number is None else (0 if int(number) == 0 else int(number) + 1)
    return account_number, str(match.group("suffix") or "").strip()


def resolve_model(model: str) -> Tuple[str, Optional[str]]:
    """Map a friendly Glossarion route/suffix to OpenCode's model + variant."""
    _account_number, suffix = parse_account_route(model)
    value = suffix.lower()

    # provider/model runs any model of a provider OpenCode is signed in to.
    direct = re.match(r"^([a-z0-9][a-z0-9-]*)/(.+)$", suffix.strip())
    if direct:
        return direct.group(1).lower() + "/" + direct.group(2), None

    mappings: List[Tuple[str, str, Optional[str]]] = [
        ("gemini-3.1-pro-high", "google/antigravity-gemini-3.1-pro", "high"),
        ("gemini-3.1-pro-low", "google/antigravity-gemini-3.1-pro", "low"),
        ("gemini-3-pro-high", "google/antigravity-gemini-3-pro", "high"),
        ("gemini-3-pro-low", "google/antigravity-gemini-3-pro", "low"),
        ("gemini-3-flash-minimal", "google/antigravity-gemini-3-flash", "minimal"),
        ("gemini-3-flash-low", "google/antigravity-gemini-3-flash", "low"),
        ("gemini-3-flash-medium", "google/antigravity-gemini-3-flash", "medium"),
        ("gemini-3-flash-high", "google/antigravity-gemini-3-flash", "high"),
        # Gemini 3.8 Flash: the tier is part of Google's model ID (no variant).
        ("gemini-3.8-flash-low", "google/antigravity-gemini-3.8-flash-low", None),
        ("gemini-3.8-flash-medium", "google/antigravity-gemini-3.8-flash-medium", None),
        ("gemini-3.8-flash-high", "google/antigravity-gemini-3.8-flash-high", None),
        ("claude-sonnet-4-6", "google/antigravity-claude-sonnet-4-6", None),
        ("claude-opus-4-6-thinking-low", "google/antigravity-claude-opus-4-6-thinking", "low"),
        ("claude-opus-4-6-thinking-max", "google/antigravity-claude-opus-4-6-thinking", "max"),
    ]
    for friendly, model_id, variant in mappings:
        if value == friendly:
            return model_id, variant
    if value.startswith("antigravity-"):
        return "google/" + value, None
    if value:
        # Models the installed plugin declares: an exact model ID wins over a
        # <model>-<variant> split (gemini-3.8-flash-high is itself an ID).
        definitions = _declared_google_models()
        declared = {key.casefold(): key for key in definitions}
        variants = {
            key.casefold(): [str(name) for name in (definition.get("variants") or {})]
            for key, definition in definitions.items()
        }
        exact = declared.get(f"antigravity-{value}")
        if exact:
            return "google/" + exact, None
        for key, names in variants.items():
            base = key[len("antigravity-"):] if key.startswith("antigravity-") else key
            for name in names:
                if value == f"{base}-{name.casefold()}":
                    return "google/" + declared[key], name
    if not value:
        return "google/antigravity-gemini-3.1-pro", "high"
    raise OcAgyError(
        f"Unknown ocagy model '{model}'. Use ocagy/gemini-3.1-pro-high, "
        "ocagy1/gemini-3.1-pro-high, ocagy0/gemini-3.1-pro-high, or another listed model."
    )


def _parse_json_events(stdout: str) -> Tuple[str, List[Dict[str, Any]], List[str]]:
    events: List[Dict[str, Any]] = []
    text_parts: List[str] = []
    non_json: List[str] = []
    for raw in _clean_text(stdout).splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            non_json.append(line)
            continue
        if not isinstance(item, dict):
            continue
        events.append(item)
        if item.get("type") == "text":
            part = item.get("part")
            if isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str) and text:
                    text_parts.append(text)
    content = "".join(text_parts).strip()
    return content, events, non_json


def _usage_from_value(value: Any) -> Optional[Dict[str, int]]:
    if isinstance(value, dict):
        candidates = [value]
        for key in ("usage", "usageMetadata", "usage_metadata", "tokens"):
            item = value.get(key)
            if isinstance(item, dict):
                candidates.insert(0, item)
        for candidate in candidates:
            def number(*keys: str) -> int:
                for key in keys:
                    raw = candidate.get(key)
                    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                        return int(raw)
                return 0
            prompt = number("input_tokens", "prompt_tokens", "inputTokens", "promptTokenCount", "input")
            completion = number("output_tokens", "completion_tokens", "outputTokens", "candidatesTokenCount", "output")
            total = number("total_tokens", "totalTokens", "totalTokenCount", "total") or prompt + completion
            cache = number("cache_read_tokens", "cached_input_tokens", "cacheReadTokens")
            if any((prompt, completion, total, cache)):
                result = {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total}
                if cache:
                    result["cache_read_tokens"] = cache
                return result
        for nested in value.values():
            found = _usage_from_value(nested)
            if found:
                return found
    elif isinstance(value, list):
        for nested in value:
            found = _usage_from_value(nested)
            if found:
                return found
    return None


def _event_error(events: List[Dict[str, Any]]) -> str:
    messages: List[str] = []
    for event in events:
        if event.get("type") != "error":
            continue
        error = event.get("error")
        if isinstance(error, dict):
            data = error.get("data")
            if isinstance(data, dict) and data.get("message"):
                messages.append(str(data.get("message")))
            elif error.get("message"):
                messages.append(str(error.get("message")))
            else:
                messages.append(json.dumps(error, ensure_ascii=False))
        elif error:
            messages.append(str(error))
    return "\n".join(messages)


def _normalize_opencode_finish_reason(reason: Any) -> Optional[str]:
    """Normalize an authoritative OpenCode reason for Glossarion routing."""
    raw = str(reason or "").strip()
    if not raw:
        return None
    normalized = raw.lower().replace(" ", "_")
    if normalized in {
        "end_turn",
        "end-turn",
        "completed",
        "complete",
    }:
        return "stop"
    if normalized in {
        "length",
        "max_tokens",
        "max-tokens",
        "max_output_tokens",
        "max-output-tokens",
        "truncated",
    }:
        return "length"
    if normalized in {
        "content-filter",
        "content_filter",
        "prohibited-content",
        "prohibited_content",
        "censorship-blocked",
        "censorship_blocked",
        "blocked",
        "safety",
        "recitation",
        "blocklist",
        "spii",
    }:
        return "prohibited_content"
    return normalized


def _finish_reason_from_info(info: Any) -> Optional[str]:
    """Read a finish value from one persisted/streamed assistant message."""
    if not isinstance(info, dict):
        return None
    for key in (
        "finish",
        "finish_reason",
        "finishReason",
        "stop_reason",
        "stopReason",
    ):
        value = info.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _finish_reason_from_message_records(
    records: Any,
    assistant_message_ids: Optional[Iterable[str]] = None,
) -> Tuple[Optional[str], str, Dict[str, Any]]:
    """Extract the latest server-persisted assistant finish reason."""
    if isinstance(records, dict):
        message_records = [records]
    elif isinstance(records, list):
        message_records = records
    else:
        return None, "", {}

    allowed_ids = {
        str(message_id)
        for message_id in (assistant_message_ids or [])
        if str(message_id or "").strip()
    }
    for record in reversed(message_records):
        if not isinstance(record, dict):
            continue
        info = record.get("info")
        if not isinstance(info, dict):
            info = {}
        if str(info.get("role", "") or "").lower() not in ("", "assistant"):
            continue
        message_id = str(info.get("id", "") or "")
        if allowed_ids and message_id not in allowed_ids:
            continue

        parts = record.get("parts")
        if isinstance(parts, list):
            for part in reversed(parts):
                if not isinstance(part, dict):
                    continue
                part_type = str(part.get("type", "") or "").replace("_", "-").lower()
                reason = str(part.get("reason", "") or "").strip()
                if part_type == "step-finish" and reason:
                    return reason, "server_message_step_finish", {
                        "message_id": message_id,
                        "part_id": str(part.get("id", "") or ""),
                    }

        reason = _finish_reason_from_info(info)
        if reason:
            return reason, "server_message_info", {
                "message_id": message_id,
            }
    return None, "", {}


def _finish_reason_from_cli_events(
    events: Iterable[Dict[str, Any]],
) -> Tuple[Optional[str], str]:
    """Extract OpenCode's reason from buffered ``run --format json`` events."""
    for event in reversed(list(events or [])):
        if not isinstance(event, dict):
            continue
        event_type = str(event.get("type", "") or "").replace("_", "-").lower()
        part = event.get("part")
        if event_type == "step-finish" and isinstance(part, dict):
            reason = str(part.get("reason", "") or "").strip()
            if reason:
                return reason, "cli_step_finish"
        if event_type == "message.updated":
            properties = event.get("properties")
            info = properties.get("info") if isinstance(properties, dict) else None
            reason = _finish_reason_from_info(info)
            if reason:
                return reason, "cli_message_info"
    return None, ""


def _http_json(
    base_url: str,
    path: str,
    *,
    method: str = "GET",
    payload: Optional[Dict[str, Any]] = None,
    timeout: float = 10.0,
) -> Any:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=data,
        headers=headers,
        method=method,
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read()
    if not raw:
        return None
    return json.loads(raw.decode("utf-8", errors="replace"))


def _loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _drain_process_pipe(pipe: Any, output: List[str]) -> None:
    try:
        for line in iter(pipe.readline, ""):
            output.append(str(line))
            if len(output) > 500:
                del output[:100]
    except Exception:
        pass
    finally:
        try:
            pipe.close()
        except Exception:
            pass


def _read_sse_events(response: Any, output: "queue.Queue[Tuple[str, Any]]") -> None:
    """Forward each httpx SSE event immediately instead of buffering the body."""
    data_lines: List[str] = []
    try:
        for raw in response.iter_lines():
            line = (
                raw.decode("utf-8", errors="replace")
                if isinstance(raw, bytes)
                else str(raw)
            ).rstrip("\r\n")
            if not line:
                if data_lines:
                    value = "\n".join(data_lines)
                    data_lines.clear()
                    try:
                        event = json.loads(value)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(event, dict):
                        output.put(("event", event))
                continue
            if line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
        if data_lines:
            try:
                event = json.loads("\n".join(data_lines))
            except json.JSONDecodeError:
                event = None
            if isinstance(event, dict):
                output.put(("event", event))
    except Exception as exc:
        output.put(("error", exc))
    finally:
        output.put(("eof", None))


def _stream_log_text(text: str, buffer: List[str], logger: Callable[[str], None]) -> None:
    """Emit readable line-sized chunks, matching the Antigravity stream logger."""
    if not text:
        return
    combined = "".join(buffer) + text
    for tag in ("</h1>", "</h2>", "</h3>", "</h4>", "</h5>", "</h6>", "</p>"):
        combined = combined.replace(tag, tag + "\n")
    if "\n" in combined:
        parts = combined.split("\n")
        for part in parts[:-1]:
            logger(part)
        buffer[:] = [parts[-1]]
    else:
        buffer[:] = [combined]
        if len(combined) > 150:
            logger(combined)
            buffer.clear()


def _forced_stream_thinking_logging_enabled(log_stream: bool) -> bool:
    """Reasoning is part of OcAgy's forced live stream visibility."""
    return bool(log_stream)


class _OpenCodeStreamState:
    """Collect authoritative output while exposing incremental SSE deltas."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.assistant_message_ids: set[str] = set()
        self.assistant_message_order: List[str] = []
        self.text_order: List[str] = []
        self.text_by_part: Dict[str, str] = {}
        self.reasoning_by_part: Dict[str, str] = {}
        self.part_types: Dict[str, str] = {}
        self.step_events: List[Dict[str, Any]] = []
        self.finish_reason: Optional[str] = None
        self.raw_finish_reason: Optional[str] = None
        self.finish_reason_source = ""
        self.finish_reason_evidence: Dict[str, Any] = {}
        self.error = ""
        self.complete = False

    def record_finish_reason(
        self,
        reason: Any,
        source: str,
        evidence: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Record a real OpenCode reason without inventing a default."""
        raw_reason = str(reason or "").strip()
        normalized = _normalize_opencode_finish_reason(raw_reason)
        if not raw_reason or not normalized:
            return False
        self.raw_finish_reason = raw_reason
        self.finish_reason = normalized
        self.finish_reason_source = str(source or "")
        self.finish_reason_evidence = dict(evidence or {})
        return True

    @staticmethod
    def _merge_part(
        part: Dict[str, Any],
        delta: Any,
        store: Dict[str, str],
    ) -> str:
        part_id = str(part.get("id", "") or "")
        if not part_id:
            return ""
        previous = store.get(part_id, "")
        full = part.get("text")
        full_text = str(full) if isinstance(full, str) else ""
        delta_text = str(delta) if isinstance(delta, str) else ""
        if full_text:
            fragment = full_text[len(previous):] if full_text.startswith(previous) else delta_text
            if not fragment and full_text != previous:
                fragment = full_text
            store[part_id] = full_text
            return fragment
        if delta_text:
            store[part_id] = previous + delta_text
            return delta_text
        return ""

    @staticmethod
    def _error_text(error: Any) -> str:
        if isinstance(error, dict):
            data = error.get("data")
            if isinstance(data, dict) and data.get("message"):
                return str(data.get("message"))
            return str(error.get("message") or error.get("name") or json.dumps(error, ensure_ascii=False))
        return str(error or "OpenCode session failed")

    def feed(self, event: Dict[str, Any]) -> List[Tuple[str, str]]:
        emitted: List[Tuple[str, str]] = []
        event_type = str(event.get("type", "") or "")
        properties = event.get("properties")
        if not isinstance(properties, dict):
            properties = {}

        if event_type == "message.updated":
            info = properties.get("info")
            if isinstance(info, dict) and info.get("role") == "assistant":
                message_id = str(info.get("id", "") or "")
                if message_id:
                    self.assistant_message_ids.add(message_id)
                    if message_id not in self.assistant_message_order:
                        self.assistant_message_order.append(message_id)
                reason = _finish_reason_from_info(info)
                if reason:
                    self.record_finish_reason(
                        reason,
                        "sse_message_info",
                        {"message_id": message_id},
                    )
            return emitted

        if event_type == "message.part.updated":
            part = properties.get("part")
            if not isinstance(part, dict) or part.get("sessionID") != self.session_id:
                return emitted
            message_id = str(part.get("messageID", "") or "")
            part_type = str(part.get("type", "") or "")
            if part_type == "step-finish":
                self.step_events.append({"type": "step_finish", "part": part})
                reason = str(part.get("reason", "") or "")
                if reason:
                    self.record_finish_reason(
                        reason,
                        "sse_step_finish",
                        {
                            "message_id": message_id,
                            "part_id": str(part.get("id", "") or ""),
                        },
                    )
                return emitted
            if message_id not in self.assistant_message_ids:
                return emitted
            part_id = str(part.get("id", "") or "")
            if part_id and part_type:
                self.part_types[part_id] = part_type
            if part_type == "text":
                if part_id and part_id not in self.text_order:
                    self.text_order.append(part_id)
                fragment = self._merge_part(part, properties.get("delta"), self.text_by_part)
                if fragment:
                    emitted.append(("text", fragment))
            elif part_type == "reasoning":
                fragment = self._merge_part(part, properties.get("delta"), self.reasoning_by_part)
                if fragment:
                    emitted.append(("reasoning", fragment))
            return emitted

        # OpenCode 1.18+ publishes the actual live token fragments separately
        # from message.part.updated. The updated event identifies the part and
        # the delta event carries each new piece of its text. Ignoring this
        # event makes an otherwise-live SSE connection appear buffered until
        # OpenCode sends the final accumulated part snapshot.
        if event_type == "message.part.delta":
            if properties.get("sessionID") != self.session_id:
                return emitted
            message_id = str(properties.get("messageID", "") or "")
            if message_id not in self.assistant_message_ids:
                return emitted
            if str(properties.get("field", "") or "") != "text":
                return emitted
            part_id = str(properties.get("partID", "") or "")
            delta = properties.get("delta")
            delta_text = str(delta) if isinstance(delta, str) else ""
            if not part_id or not delta_text:
                return emitted
            part_type = self.part_types.get(part_id, "")
            if part_type == "text":
                if part_id not in self.text_order:
                    self.text_order.append(part_id)
                self.text_by_part[part_id] = self.text_by_part.get(part_id, "") + delta_text
                emitted.append(("text", delta_text))
            elif part_type == "reasoning":
                self.reasoning_by_part[part_id] = (
                    self.reasoning_by_part.get(part_id, "") + delta_text
                )
                emitted.append(("reasoning", delta_text))
            return emitted

        if event_type == "session.error":
            event_session = str(properties.get("sessionID", "") or "")
            if not event_session or event_session == self.session_id:
                self.error = self._error_text(properties.get("error"))
                self.complete = True
            return emitted

        if event_type == "session.idle" and properties.get("sessionID") == self.session_id:
            self.complete = True
        elif event_type == "session.status" and properties.get("sessionID") == self.session_id:
            status = properties.get("status")
            if isinstance(status, dict) and status.get("type") == "idle":
                self.complete = True
        return emitted

    def content(self) -> str:
        return "".join(self.text_by_part.get(part_id, "") for part_id in self.text_order)


def _retrieve_server_finish_reason(
    base_url: str,
    session_id: str,
    assistant_message_ids: Iterable[str],
    timeout: float,
) -> Tuple[Optional[str], str, Dict[str, Any], str]:
    """Query OpenCode's persisted message records for a missed finish event."""
    deadline = time.monotonic() + max(0.1, float(timeout or 0.1))
    ordered_ids = [
        str(message_id)
        for message_id in assistant_message_ids
        if str(message_id or "").strip()
    ]
    errors: List[str] = []

    def remaining() -> float:
        return max(0.1, deadline - time.monotonic())

    encoded_session = urllib.parse.quote(str(session_id), safe="")
    for message_id in reversed(ordered_ids):
        if time.monotonic() >= deadline:
            break
        encoded_message = urllib.parse.quote(message_id, safe="")
        try:
            record = _http_json(
                base_url,
                f"/session/{encoded_session}/message/{encoded_message}",
                timeout=remaining(),
            )
            reason, source, evidence = _finish_reason_from_message_records(
                record,
                [message_id],
            )
            if reason:
                return reason, source, evidence, ""
        except Exception as exc:
            errors.append(str(exc))

    if time.monotonic() < deadline:
        try:
            records = _http_json(
                base_url,
                f"/session/{encoded_session}/message?limit=20",
                timeout=remaining(),
            )
            reason, source, evidence = _finish_reason_from_message_records(
                records,
                ordered_ids,
            )
            if reason:
                return reason, source, evidence, ""
        except Exception as exc:
            errors.append(str(exc))

    return None, "", {}, "; ".join(error for error in errors if error)


def _send_via_server(
    *,
    exe: str,
    request_dir: Path,
    prompt: str,
    model_id: str,
    variant: Optional[str],
    timeout_seconds: float,
    logger: Callable[[str], None],
    log_stream: bool,
    subprocess_env: Dict[str, str],
) -> Dict[str, Any]:
    """Run an isolated OpenCode server and consume true token deltas over SSE."""
    port = _loopback_port()
    base_url = f"http://127.0.0.1:{port}"
    command = [exe, "serve", "--hostname", "127.0.0.1", "--port", str(port)]
    proc = _spawn_opencode(
        command,
        logger,
        cwd=str(request_dir),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=_creation_flags(),
        start_new_session=(os.name != "nt"),
        env=subprocess_env,
    )
    with _ACTIVE_LOCK:
        _ACTIVE_PROCESSES.add(proc)

    server_stdout: List[str] = []
    server_stderr: List[str] = []
    for pipe, target in ((proc.stdout, server_stdout), (proc.stderr, server_stderr)):
        if pipe is not None:
            threading.Thread(
                target=_drain_process_pipe,
                args=(pipe, target),
                daemon=True,
            ).start()

    event_response = None
    event_stream_context = None
    session_id = ""
    started = time.time()

    def remaining_request_timeout() -> float:
        remaining = float(timeout_seconds) - (time.time() - started)
        if remaining <= 0:
            raise OcAgyError(
                f"OpenCode Antigravity timed out after {timeout_seconds}s. "
                "Increase the API timeout or use a shorter chunk."
            )
        return max(0.1, remaining)

    try:
        startup_deadline = started + timeout_seconds
        while time.time() < startup_deadline:
            if is_cancelled():
                raise OcAgyError("OpenCode Antigravity request cancelled by user")
            if proc.poll() is not None:
                detail = _clean_text("".join(server_stderr + server_stdout))
                raise OcAgyError(
                    "OpenCode Antigravity server exited during startup."
                    + (f" Details: {detail}" if detail else "")
                )
            try:
                health = _http_json(
                    base_url,
                    "/global/health",
                    timeout=min(0.75, remaining_request_timeout()),
                )
                if isinstance(health, dict) and health.get("healthy"):
                    break
            except Exception:
                time.sleep(0.1)
        else:
            raise OcAgyError(
                f"OpenCode Antigravity server did not become ready within {timeout_seconds} seconds."
            )

        session = _http_json(
            base_url,
            "/session",
            method="POST",
            payload={"title": "Glossarion translation"},
            timeout=remaining_request_timeout(),
        )
        if not isinstance(session, dict) or not session.get("id"):
            raise OcAgyError("OpenCode Antigravity could not create a streaming session.")
        session_id = str(session["id"])

        if httpx is None:
            raise OcAgyError(
                "OcAgy real-time streaming requires httpx, but it is not installed."
            )
        event_queue: "queue.Queue[Tuple[str, Any]]" = queue.Queue()
        connect_timeout = remaining_request_timeout()
        stream_timeout = httpx.Timeout(
            remaining_request_timeout(),
            connect=connect_timeout,
        )
        event_stream_context = httpx.stream(
            "GET",
            base_url + "/event",
            headers={
                "Accept": "text/event-stream",
                "Accept-Encoding": "identity",
                "Cache-Control": "no-cache",
            },
            timeout=stream_timeout,
        )
        event_response = event_stream_context.__enter__()
        if event_response.status_code != 200:
            try:
                detail = event_response.read().decode("utf-8", errors="replace")
            except Exception:
                detail = f"HTTP {event_response.status_code}"
            raise _classify_error(detail, int(event_response.status_code or 1))
        threading.Thread(
            target=_read_sse_events,
            args=(event_response, event_queue),
            daemon=True,
        ).start()

        connected = False
        connect_deadline = started + timeout_seconds
        pending_events: List[Dict[str, Any]] = []
        while time.time() < connect_deadline:
            try:
                kind, value = event_queue.get(timeout=0.5)
            except queue.Empty:
                if is_cancelled():
                    raise OcAgyError("OpenCode Antigravity request cancelled by user")
                continue
            if kind == "event" and isinstance(value, dict):
                if value.get("type") == "server.connected":
                    connected = True
                    break
                pending_events.append(value)
            elif kind == "error":
                raise OcAgyError(f"OpenCode Antigravity event stream failed: {value}")
            elif kind == "eof":
                break
        if not connected:
            raise OcAgyError("OpenCode Antigravity event stream did not connect.")

        provider_id, model_name = model_id.split("/", 1)
        payload: Dict[str, Any] = {
            "agent": "glossarion",
            "model": {"providerID": provider_id, "modelID": model_name},
            "parts": [{"type": "text", "text": prompt}],
        }
        if variant:
            payload["variant"] = variant
        _http_json(
            base_url,
            f"/session/{session_id}/prompt_async",
            method="POST",
            payload=payload,
            timeout=remaining_request_timeout(),
        )

        state = _OpenCodeStreamState(session_id)
        text_buffer: List[str] = []
        thinking_buffer: List[str] = []
        thinking_started = False
        first_text = False
        stream_thinking = _forced_stream_thinking_logging_enabled(log_stream)
        # Google's "no longer supported" reply to an outdated plugin is not logged.
        notice_filter = _UnsupportedNoticeFilter()

        def consume(event: Dict[str, Any]) -> None:
            nonlocal thinking_started, first_text
            for fragment_type, fragment in state.feed(event):
                if fragment_type == "reasoning" and log_stream and stream_thinking:
                    if not thinking_started:
                        thinking_started = True
                        logger("🧠 [ocagy] Thinking...")
                    _stream_log_text(fragment, thinking_buffer, logger)
                elif fragment_type == "text" and log_stream:
                    fragment = notice_filter.feed(fragment)
                    if not fragment:
                        continue
                    if not first_text:
                        first_text = True
                        logger(f"OpenCode Antigravity: first token in {time.time() - started:.1f}s, streaming...")
                    _stream_log_text(fragment, text_buffer, logger)

        for event in pending_events:
            consume(event)

        stream_eof = False
        while not state.complete:
            if is_cancelled():
                try:
                    _http_json(
                        base_url,
                        f"/session/{session_id}/abort",
                        method="POST",
                        payload={},
                        timeout=2,
                    )
                except Exception:
                    pass
                raise OcAgyError("OpenCode Antigravity request cancelled by user")
            if time.time() - started >= timeout_seconds:
                raise OcAgyError(
                    f"OpenCode Antigravity timed out after {timeout_seconds}s. "
                    "Increase the API timeout or use a shorter chunk."
                )
            if proc.poll() is not None:
                detail = _clean_text("".join(server_stderr + server_stdout))
                raise OcAgyError(
                    "OpenCode Antigravity server exited before the response completed."
                    + (f" Details: {detail}" if detail else "")
                )
            try:
                kind, value = event_queue.get(timeout=0.25)
            except queue.Empty:
                continue
            if kind == "event" and isinstance(value, dict):
                consume(value)
            elif kind == "error":
                raise OcAgyError(f"OpenCode Antigravity event stream failed: {value}")
            elif kind == "eof":
                stream_eof = True
                break

        if stream_eof and not state.complete:
            raise OcAgyError("OpenCode Antigravity event stream closed before completion.")
        if state.error:
            raise _classify_error(state.error, 1)

        finish_reason_lookup_error = ""
        if not state.finish_reason:
            remaining_after_stream = timeout_seconds - (time.time() - started)
            if remaining_after_stream > 0.1:
                lookup_timeout = min(3.0, remaining_after_stream)
                (
                    persisted_reason,
                    persisted_source,
                    persisted_evidence,
                    finish_reason_lookup_error,
                ) = _retrieve_server_finish_reason(
                    base_url,
                    session_id,
                    state.assistant_message_order,
                    lookup_timeout,
                )
                if persisted_reason:
                    state.record_finish_reason(
                        persisted_reason,
                        persisted_source,
                        persisted_evidence,
                    )
            else:
                finish_reason_lookup_error = (
                    "request deadline exhausted before server lookup"
                )

        finish_reason_fallback = not bool(state.finish_reason)
        if finish_reason_fallback:
            state.finish_reason = "stop"
            state.raw_finish_reason = None
            state.finish_reason_source = "fallback_stop"
            state.finish_reason_evidence = {}
            warning = (
                "⚠️ OpenCode Antigravity: no finish reason was present in "
                "the live events or persisted server message; falling back to 'stop'."
            )
            if finish_reason_lookup_error:
                warning += f" Server lookup: {finish_reason_lookup_error}"
            logger(warning)
        else:
            raw_suffix = (
                f", raw: '{state.raw_finish_reason}'"
                if state.raw_finish_reason
                and state.raw_finish_reason != state.finish_reason
                else ""
            )
            logger(
                "🏁 OpenCode Antigravity: finish reason "
                f"'{state.finish_reason}' (source: {state.finish_reason_source}"
                f"{raw_suffix})"
            )

        if log_stream and thinking_buffer:
            tail = "".join(thinking_buffer).strip()
            if tail:
                logger(f"    {tail}")
        if log_stream:
            held = notice_filter.rest()
            if held:
                text_buffer.append(held)
        if log_stream and text_buffer:
            tail = "".join(text_buffer).strip()
            if tail:
                logger(tail)

        return {
            "content": state.content(),
            "finish_reason": state.finish_reason,
            "raw_finish_reason": state.raw_finish_reason,
            "finish_reason_source": state.finish_reason_source,
            "finish_reason_fallback": finish_reason_fallback,
            "finish_reason_evidence": state.finish_reason_evidence,
            "finish_reason_lookup_error": finish_reason_lookup_error,
            "usage": _usage_from_value(state.step_events),
            "raw_response": state.step_events,
        }
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8", errors="replace")
        except Exception:
            detail = str(exc)
        raise _classify_error(detail, int(getattr(exc, "code", 1) or 1))
    except urllib.error.URLError as exc:
        raise OcAgyError(f"OpenCode Antigravity local streaming connection failed: {exc}")
    except TimeoutError as exc:
        raise OcAgyError(
            f"OpenCode Antigravity timed out after {timeout_seconds}s. "
            "Increase the API timeout or use a shorter chunk."
        ) from exc
    except Exception as exc:
        if httpx is not None and isinstance(exc, httpx.TimeoutException):
            raise OcAgyError(
                f"OpenCode Antigravity real-time stream timed out after {timeout_seconds}s."
            ) from exc
        if httpx is not None and isinstance(exc, httpx.RequestError):
            raise OcAgyError(
                f"OpenCode Antigravity local real-time stream failed: {exc}"
            ) from exc
        raise
    finally:
        if event_response is not None:
            try:
                event_response.close()
            except Exception:
                pass
        if event_stream_context is not None:
            try:
                event_stream_context.__exit__(None, None, None)
            except Exception:
                pass
        _terminate_process_tree(proc)
        with _ACTIVE_LOCK:
            _ACTIVE_PROCESSES.discard(proc)


def _classify_error(detail: str, returncode: int) -> OcAgyError:
    text = _clean_text(detail) or f"opencode exited with code {returncode}"
    lower = text.lower()
    if any(x in lower for x in (
        "api key is missing",
        "api key missing",
        "google_generative_ai_api_key",
    )):
        return OcAgyError(
            "OpenCode did not find an Antigravity OAuth session and fell back to its built-in "
            "Google provider. The ocagy/ route does not use or require a Google API key. Click "
            "the OCAGY Login button, select Google, then OAuth with Google "
            "(Antigravity), finish signing in, and retry. Do not select Manually enter API Key."
        )
    if any(x in lower for x in ("invalid_grant", "not authenticated", "auth login", "oauth", "credential", "api key missing")):
        return OcAgyError(
            "OpenCode Antigravity authentication failed. Click the OCAGY Login button, "
            "select Google → OAuth with Google (Antigravity), then retry. Details: " + text
        )
    if any(x in lower for x in ("rate limit", "rate-limited", "quota", "resource_exhausted", "too many requests", "429")):
        return OcAgyError("OpenCode Antigravity quota/rate limit: " + text)
    if any(x in lower for x in ("model not found", "unknown model", "model does not exist", "cannot be resolved")):
        return OcAgyError("OpenCode Antigravity model error: " + text)
    if any(x in lower for x in ("plugin", "opencode-antigravity-auth", "provider google")):
        return OcAgyError("OpenCode Antigravity plugin/configuration error: " + text)
    return OcAgyError(f"OpenCode Antigravity failed (exit {returncode}): {text}")


def _send_chat_completion_buffered(
    *,
    messages: List[Dict[str, Any]],
    model: str,
    temperature: float = 0.3,
    max_tokens: int = 65536,
    timeout: float = 1800,
    log_fn: Optional[Callable[[str], None]] = None,
    _after_plugin_update: bool = False,
) -> Dict[str, Any]:
    """Run one isolated ``opencode run --format json`` request via the plugin."""
    if is_cancelled():
        raise OcAgyError("OpenCode Antigravity request cancelled by user")

    logger = log_fn or (lambda _message: None)
    exe = ensure_opencode_installed(log_fn=logger)
    account_number, _model_suffix = parse_account_route(model)
    model_id, variant = resolve_model(model)
    if model_id.startswith("google/"):
        selected = _require_oauth_account(account_number)
    else:
        # Other providers use OpenCode's own stored login, not an Antigravity slot.
        account_number, selected = 0, None
    ensure_auth_plugin_installed(exe, log_fn=logger)
    plugin_version = _installed_plugin_version(_plugin_spec())
    prompt = build_prompt(messages)
    timeout_seconds = max(1.0, float(timeout or 1800))

    base = _workspace_dir()
    request_dir = Path(tempfile.mkdtemp(prefix="request-", dir=str(base)))
    try:
        process_env = _request_subprocess_env(
            request_dir,
            account_number,
            selected,
            logger,
            max_tokens,
        )
        _write_workspace_config(request_dir, temperature=temperature)
    except Exception:
        shutil.rmtree(request_dir, ignore_errors=True)
        raise
    command = [
        exe,
        "run",
        "--format",
        "json",
        "--model",
        model_id,
        "--agent",
        "glossarion",
        "--dir",
        str(request_dir),
        "--title",
        "Glossarion translation",
    ]
    if variant:
        command.extend(["--variant", variant])

    logger(f"🪐 OpenCode Antigravity: {Path(exe).name} run --model {model_id}" + (f" --variant {variant}" if variant else ""))
    _log_generation_controls(
        logger,
        model_id=model_id,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    start = time.time()
    try:
        proc = _spawn_opencode(
            command,
            logger,
            cwd=str(request_dir),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=_creation_flags(),
            start_new_session=(os.name != "nt"),
            env=process_env,
        )
    except Exception:
        shutil.rmtree(request_dir, ignore_errors=True)
        raise
    with _ACTIVE_LOCK:
        _ACTIVE_PROCESSES.add(proc)

    stdout = ""
    stderr = ""
    try:
        try:
            stdout, stderr = proc.communicate(input=prompt, timeout=1.0)
        except subprocess.TimeoutExpired:
            while True:
                if is_cancelled():
                    _terminate_process_tree(proc)
                    raise OcAgyError("OpenCode Antigravity request cancelled by user")
                if time.time() - start >= timeout_seconds:
                    _terminate_process_tree(proc)
                    raise OcAgyError(
                        f"OpenCode Antigravity timed out after {timeout_seconds}s. "
                        "Increase the API timeout or use a shorter chunk."
                    )
                try:
                    stdout, stderr = proc.communicate(timeout=0.5)
                    break
                except subprocess.TimeoutExpired:
                    continue
    finally:
        with _ACTIVE_LOCK:
            _ACTIVE_PROCESSES.discard(proc)
        shutil.rmtree(request_dir, ignore_errors=True)

    elapsed = time.time() - start
    content, events, non_json = _parse_json_events(stdout)
    event_error = _event_error(events)
    if proc.returncode != 0 or event_error:
        detail = "\n".join(part for part in (event_error, _clean_text(stderr), "\n".join(non_json)) if part)
        raise _classify_error(detail, int(proc.returncode or 1))

    if not content:
        # Compatibility fallback for older OpenCode versions that ignore --format json.
        fallback = "\n".join(non_json).strip()
        if fallback:
            content = fallback
    content = _clean_text(content)
    if _is_unsupported_plugin_notice(content):
        if not _after_plugin_update and _wait_for_newer_plugin(plugin_version, exe, logger):
            return _send_chat_completion_buffered(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                log_fn=log_fn,
                _after_plugin_update=True,
            )
        raise _unsupported_plugin_error(plugin_version)
    if not content:
        detail = _clean_text(stderr)
        raise OcAgyError("OpenCode Antigravity returned an empty response." + (f" Details: {detail}" if detail else ""))

    usage = _usage_from_value(events)
    raw_finish_reason, finish_reason_source = _finish_reason_from_cli_events(
        events
    )
    finish_reason = _normalize_opencode_finish_reason(raw_finish_reason)
    finish_reason_fallback = not bool(finish_reason)
    if finish_reason_fallback:
        finish_reason = "stop"
        finish_reason_source = "fallback_stop"
        logger(
            "⚠️ OpenCode Antigravity: buffered events did not include a finish "
            "reason; falling back to 'stop'."
        )
    else:
        raw_suffix = (
            f", raw: '{raw_finish_reason}'"
            if raw_finish_reason != finish_reason
            else ""
        )
        logger(
            "🏁 OpenCode Antigravity: finish reason "
            f"'{finish_reason}' (source: {finish_reason_source}{raw_suffix})"
        )
    logger(f"✅ OpenCode Antigravity: completed in {elapsed:.1f}s")
    return {
        "content": content,
        "finish_reason": finish_reason,
        "raw_finish_reason": raw_finish_reason,
        "finish_reason_source": finish_reason_source,
        "finish_reason_fallback": finish_reason_fallback,
        "usage": usage,
        "model": model_id,
        "variant": variant,
        "provider": "ocagy",
        "elapsed_seconds": elapsed,
        "stderr": _clean_text(stderr),
        "raw_response": events,
    }


def send_chat_completion(
    *,
    messages: List[Dict[str, Any]],
    model: str,
    temperature: float = 0.3,
    max_tokens: int = 65536,
    timeout: float = 1800,
    log_fn: Optional[Callable[[str], None]] = None,
    log_stream: bool = True,
    _after_plugin_update: bool = False,
) -> Dict[str, Any]:
    """Run one isolated request through OpenCode's live local event API.

    When Google answers that the plugin version is no longer supported, the
    request waits for the plugin update and is sent once more with the new one.
    """
    if is_cancelled():
        raise OcAgyError("OpenCode Antigravity request cancelled by user")

    logger = log_fn or (lambda _message: None)
    exe = ensure_opencode_installed(log_fn=logger)
    account_number, _model_suffix = parse_account_route(model)
    model_id, variant = resolve_model(model)
    if model_id.startswith("google/"):
        selected = _require_oauth_account(account_number)
    else:
        # Other providers use OpenCode's own stored login, not an Antigravity slot.
        account_number, selected = 0, None
    ensure_auth_plugin_installed(exe, log_fn=logger)
    plugin_version = _installed_plugin_version(_plugin_spec())
    prompt = build_prompt(messages)
    timeout_seconds = max(1.0, float(timeout or 1800))

    base = _workspace_dir()
    request_dir = Path(tempfile.mkdtemp(prefix="request-", dir=str(base)))
    start = time.time()
    try:
        process_env = _request_subprocess_env(
            request_dir,
            account_number,
            selected,
            logger,
            max_tokens,
        )
        _write_workspace_config(request_dir, temperature=temperature)
        logger(
            f"🪐 OpenCode Antigravity: {Path(exe).name} live stream --model {model_id}"
            + (f" --variant {variant}" if variant else "")
        )
        _log_generation_controls(
            logger,
            model_id=model_id,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        result = _send_via_server(
            exe=exe,
            request_dir=request_dir,
            prompt=prompt,
            model_id=model_id,
            variant=variant,
            timeout_seconds=timeout_seconds,
            logger=logger,
            log_stream=bool(log_stream),
            subprocess_env=process_env,
        )
    finally:
        shutil.rmtree(request_dir, ignore_errors=True)

    elapsed = time.time() - start
    content = _clean_text(str(result.get("content", "") or ""))
    if _is_unsupported_plugin_notice(content):
        if not _after_plugin_update and _wait_for_newer_plugin(plugin_version, exe, logger):
            return send_chat_completion(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                log_fn=log_fn,
                log_stream=log_stream,
                _after_plugin_update=True,
            )
        raise _unsupported_plugin_error(plugin_version)
    if not content:
        raise OcAgyError("OpenCode Antigravity returned an empty response.")

    logger(f"✅ OpenCode Antigravity: stream finished in {elapsed:.1f}s")
    return {
        "content": content,
        "finish_reason": str(result.get("finish_reason", "") or "stop"),
        "raw_finish_reason": result.get("raw_finish_reason"),
        "finish_reason_source": str(
            result.get("finish_reason_source", "")
            or (
                "send_chat_completion_fallback_stop"
                if not result.get("finish_reason")
                else ""
            )
        ),
        "finish_reason_fallback": bool(
            result.get("finish_reason_fallback", False)
            or not result.get("finish_reason")
        ),
        "finish_reason_evidence": result.get("finish_reason_evidence") or {},
        "finish_reason_lookup_error": result.get(
            "finish_reason_lookup_error", ""
        ),
        "usage": result.get("usage"),
        "model": model_id,
        "variant": variant,
        "provider": "ocagy",
        "elapsed_seconds": elapsed,
        "stderr": "",
        "raw_response": result.get("raw_response"),
    }


# ---------------------------------------------------------------------------
# OpenCode Zen free-tier — ocz/ prefix
# ---------------------------------------------------------------------------


def _zen_error_detail(stderr: str) -> str:
    """Extract logged error messages without repeating their stack-trace causes."""
    errors = re.findall(r'\berror=("(?:[^"\\]|\\.)*")', stderr)
    if errors:
        try:
            return "\n".join(dict.fromkeys(str(json.loads(error)) for error in errors))[:4000]
        except ValueError:
            pass
    # Newer OpenCode builds log Effect causes rather than an error= field.
    # Unwrap the underlying exception, omitting timestamps and subscriber logs.
    causes = re.findall(r'\bcause=("(?:[^"\\]|\\.)*")', stderr)
    details = []
    for cause in causes:
        try:
            decoded = str(json.loads(cause))
        except ValueError:
            continue
        match = re.fullmatch(r'Cause\(\[(?:Fail|Die)\((.*)\)\]\)', decoded, re.DOTALL)
        if match:
            detail = re.sub(r'^\w*(?:Error|Exception):\s*', '', match.group(1))
            details.append(detail)
    if details:
        return "\n".join(dict.fromkeys(details))[:4000]
    return _clean_text(stderr)[:4000]


def _run_opencode_zen_buffered(
    *,
    exe: str,
    prompt: str,
    effective_model: str,
    timeout_seconds: float,
    logger: Callable[[str], None],
    subprocess_env: Dict[str, str],
) -> Dict[str, Any]:
    """Buffered (non-streaming) path: ``opencode run --format json``."""
    command = [
        exe,
        "run",
        "--print-logs", "--log-level", "ERROR",
        "--format", "json",
        "--model", effective_model,
        "--title", "Glossarion translation",
    ]

    try:
        proc = _spawn_opencode(
            command,
            logger,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=_creation_flags(),
            start_new_session=(os.name != "nt"),
            env=subprocess_env,
        )
    except Exception:
        raise
    with _ACTIVE_LOCK:
        _ACTIVE_PROCESSES.add(proc)

    stdout = ""
    stderr = ""
    start = time.time()
    try:
        try:
            stdout, stderr = proc.communicate(input=prompt, timeout=1.0)
        except subprocess.TimeoutExpired:
            while True:
                if is_cancelled():
                    _terminate_process_tree(proc)
                    raise OcAgyError("OpenCode Zen request cancelled by user")
                if time.time() - start >= timeout_seconds:
                    _terminate_process_tree(proc)
                    raise OcAgyError(
                        f"OpenCode Zen timed out after {timeout_seconds}s. "
                        "Increase the API timeout or use a shorter chunk."
                    )
                try:
                    stdout, stderr = proc.communicate(timeout=0.5)
                    break
                except subprocess.TimeoutExpired:
                    continue
    finally:
        with _ACTIVE_LOCK:
            _ACTIVE_PROCESSES.discard(proc)

    content, events, non_json = _parse_json_events(stdout)
    event_error = _event_error(events)
    if proc.returncode != 0 or event_error:
        detail = "\n".join(
            part for part in (event_error, _zen_error_detail(stderr), "\n".join(non_json)) if part
        )
        text = _clean_text(detail) or f"opencode exited with code {proc.returncode}"
        lower = text.lower()
        if any(
            x in lower
            for x in ("rate limit", "rate-limited", "quota", "resource_exhausted", "too many requests", "429")
        ):
            raise OcAgyError("OpenCode Zen quota/rate limit: " + text)
        if any(
            x in lower
            for x in ("model not found", "unknown model", "model does not exist", "cannot be resolved", "model is unavailable")
        ):
            raise OcAgyError("OpenCode Zen model error: " + text)
        raise OcAgyError(f"OpenCode Zen failed (exit {proc.returncode}): {text}")

    if not content:
        fallback = "\n".join(non_json).strip()
        if fallback:
            content = fallback
    content = _clean_text(content)
    if not content:
        detail = _clean_text(stderr)
        raise OcAgyError(
            "OpenCode Zen returned an empty response."
            + (f" Details: {detail}" if detail else "")
        )

    usage = _usage_from_value(events)
    raw_finish_reason, finish_reason_source = _finish_reason_from_cli_events(events)
    finish_reason = _normalize_opencode_finish_reason(raw_finish_reason)
    finish_reason_fallback = not bool(finish_reason)
    if finish_reason_fallback:
        finish_reason = "stop"
        finish_reason_source = "fallback_stop"

    return {
        "content": content,
        "finish_reason": finish_reason,
        "raw_finish_reason": raw_finish_reason,
        "finish_reason_source": finish_reason_source,
        "finish_reason_fallback": finish_reason_fallback,
        "usage": usage,
        "raw_response": events,
    }


def _send_via_server_zen(
    *,
    exe: str,
    prompt: str,
    model_id: str,
    timeout_seconds: float,
    logger: Callable[[str], None],
    log_stream: bool,
    subprocess_env: Dict[str, str],
) -> Dict[str, Any]:
    """Run an OpenCode serve instance and stream the response via SSE.

    Same event-based protocol as ``_send_via_server`` but without the
    antigravity plugin/OAuth.  The server runs from the current working
    directory so OpenCode derives a real projectID (required by the
    free-tier gate).
    """
    port = _loopback_port()
    base_url = f"http://127.0.0.1:{port}"
    command = [exe, "serve", "--hostname", "127.0.0.1", "--port", str(port), "--pure",
               "--print-logs", "--log-level", "ERROR"]
    proc = _spawn_opencode(
        command,
        logger,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=_creation_flags(),
        start_new_session=(os.name != "nt"),
        env=subprocess_env,
    )
    with _ACTIVE_LOCK:
        _ACTIVE_PROCESSES.add(proc)

    server_stdout: List[str] = []
    server_stderr: List[str] = []
    for pipe, target in ((proc.stdout, server_stdout), (proc.stderr, server_stderr)):
        if pipe is not None:
            threading.Thread(
                target=_drain_process_pipe,
                args=(pipe, target),
                daemon=True,
            ).start()

    event_response = None
    event_stream_context = None
    session_id = ""
    started = time.time()

    def remaining() -> float:
        r = float(timeout_seconds) - (time.time() - started)
        if r <= 0:
            raise OcAgyError(
                f"OpenCode Zen timed out after {timeout_seconds}s. "
                "Increase the API timeout or use a shorter chunk."
            )
        return max(0.1, r)

    try:
        # Wait for server health
        while time.time() - started < timeout_seconds:
            if is_cancelled():
                raise OcAgyError("OpenCode Zen request cancelled by user")
            if proc.poll() is not None:
                detail = _clean_text("".join(server_stderr + server_stdout))
                raise OcAgyError(
                    "OpenCode Zen server exited during startup."
                    + (f" Details: {detail}" if detail else "")
                )
            try:
                health = _http_json(
                    base_url,
                    "/global/health",
                    timeout=min(0.75, remaining()),
                )
                if isinstance(health, dict) and health.get("healthy"):
                    break
            except Exception:
                time.sleep(0.1)
        else:
            raise OcAgyError(
                f"OpenCode Zen server did not become ready within {timeout_seconds}s."
            )

        # Create session
        session = _http_json(
            base_url,
            "/session",
            method="POST",
            payload={"title": "Glossarion translation"},
            timeout=remaining(),
        )
        if not isinstance(session, dict) or not session.get("id"):
            raise OcAgyError("OpenCode Zen could not create a streaming session.")
        session_id = str(session["id"])

        # Connect SSE event stream
        if httpx is None:
            raise OcAgyError(
                "OpenCode Zen real-time streaming requires httpx, but it is not installed."
            )
        event_queue: "queue.Queue[Tuple[str, Any]]" = queue.Queue()
        stream_timeout = httpx.Timeout(remaining(), connect=remaining())
        event_stream_context = httpx.stream(
            "GET",
            base_url + "/event",
            headers={
                "Accept": "text/event-stream",
                "Accept-Encoding": "identity",
                "Cache-Control": "no-cache",
            },
            timeout=stream_timeout,
        )
        event_response = event_stream_context.__enter__()
        if event_response.status_code != 200:
            try:
                detail = event_response.read().decode("utf-8", errors="replace")
            except Exception:
                detail = f"HTTP {event_response.status_code}"
            raise OcAgyError(f"OpenCode Zen event stream HTTP {event_response.status_code}: {detail}")
        threading.Thread(
            target=_read_sse_events,
            args=(event_response, event_queue),
            daemon=True,
        ).start()

        # Wait for server.connected
        connected = False
        pending_events: List[Dict[str, Any]] = []
        while time.time() - started < timeout_seconds:
            try:
                kind, value = event_queue.get(timeout=0.5)
            except queue.Empty:
                if is_cancelled():
                    raise OcAgyError("OpenCode Zen request cancelled by user")
                continue
            if kind == "event" and isinstance(value, dict):
                if value.get("type") == "server.connected":
                    connected = True
                    break
                pending_events.append(value)
            elif kind == "error":
                raise OcAgyError(f"OpenCode Zen event stream failed: {value}")
            elif kind == "eof":
                break
        if not connected:
            raise OcAgyError("OpenCode Zen event stream did not connect.")

        # Send prompt
        provider_id, model_name = model_id.split("/", 1)
        payload: Dict[str, Any] = {
            "model": {"providerID": provider_id, "modelID": model_name},
            "parts": [{"type": "text", "text": prompt}],
        }
        _http_json(
            base_url,
            f"/session/{session_id}/prompt_async",
            method="POST",
            payload=payload,
            timeout=remaining(),
        )

        # Consume streaming events
        state = _OpenCodeStreamState(session_id)
        text_buffer: List[str] = []
        thinking_buffer: List[str] = []
        thinking_started = False
        first_text = False
        stream_thinking = log_stream and os.getenv(
            "STREAM_THINKING_LOGS", "0"
        ).strip().lower() not in ("0", "false", "no", "off")

        def consume(event: Dict[str, Any]) -> None:
            nonlocal thinking_started, first_text
            for fragment_type, fragment in state.feed(event):
                if fragment_type == "reasoning" and stream_thinking:
                    if not thinking_started:
                        thinking_started = True
                        logger("🧠 [ocz] Thinking...")
                    _stream_log_text(fragment, thinking_buffer, logger)
                elif fragment_type == "text" and log_stream:
                    if not first_text:
                        first_text = True
                        logger(f"OpenCode Zen: first token in {time.time() - started:.1f}s, streaming...")
                    _stream_log_text(fragment, text_buffer, logger)

        for event in pending_events:
            consume(event)

        stream_eof = False
        while not state.complete:
            if is_cancelled():
                try:
                    _http_json(
                        base_url,
                        f"/session/{session_id}/abort",
                        method="POST",
                        payload={},
                        timeout=2,
                    )
                except Exception:
                    pass
                raise OcAgyError("OpenCode Zen request cancelled by user")
            if time.time() - started >= timeout_seconds:
                raise OcAgyError(
                    f"OpenCode Zen timed out after {timeout_seconds}s. "
                    "Increase the API timeout or use a shorter chunk."
                )
            if proc.poll() is not None:
                detail = _clean_text("".join(server_stderr + server_stdout))
                raise OcAgyError(
                    "OpenCode Zen server exited before the response completed."
                    + (f" Details: {detail}" if detail else "")
                )
            try:
                kind, value = event_queue.get(timeout=0.25)
            except queue.Empty:
                continue
            if kind == "event" and isinstance(value, dict):
                consume(value)
            elif kind == "error":
                raise OcAgyError(f"OpenCode Zen event stream failed: {value}")
            elif kind == "eof":
                stream_eof = True
                break

        if stream_eof and not state.complete:
            raise OcAgyError("OpenCode Zen event stream closed before completion.")
        if state.error:
            lower = state.error.lower()
            if any(x in lower for x in ("rate limit", "rate-limited", "quota", "resource_exhausted", "too many requests", "429")):
                raise OcAgyError("OpenCode Zen quota/rate limit: " + state.error)
            raise OcAgyError(f"OpenCode Zen error: {state.error}")

        # Try server-side finish reason lookup
        finish_reason_lookup_error = ""
        if not state.finish_reason:
            remaining_after = timeout_seconds - (time.time() - started)
            if remaining_after > 0.1:
                (
                    persisted_reason,
                    persisted_source,
                    persisted_evidence,
                    finish_reason_lookup_error,
                ) = _retrieve_server_finish_reason(
                    base_url,
                    session_id,
                    state.assistant_message_order,
                    min(3.0, remaining_after),
                )
                if persisted_reason:
                    state.record_finish_reason(
                        persisted_reason, persisted_source, persisted_evidence,
                    )

        finish_reason_fallback = not bool(state.finish_reason)
        if finish_reason_fallback:
            state.finish_reason = "stop"
            state.raw_finish_reason = None
            state.finish_reason_source = "fallback_stop"
            state.finish_reason_evidence = {}

        if log_stream and thinking_buffer:
            tail = "".join(thinking_buffer).strip()
            if tail:
                logger(f"    {tail}")
        if log_stream and text_buffer:
            tail = "".join(text_buffer).strip()
            if tail:
                logger(tail)

        return {
            "content": state.content(),
            "finish_reason": state.finish_reason,
            "raw_finish_reason": state.raw_finish_reason,
            "finish_reason_source": state.finish_reason_source,
            "finish_reason_fallback": finish_reason_fallback,
            "finish_reason_evidence": state.finish_reason_evidence,
            "finish_reason_lookup_error": finish_reason_lookup_error,
            "usage": _usage_from_value(state.step_events),
            "raw_response": state.step_events,
        }
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8", errors="replace")
        except Exception:
            detail = str(exc)
        server_detail = _zen_error_detail("\n".join(server_stderr))
        raise OcAgyError(f"OpenCode Zen HTTP error: {detail}"
                         + (f"\n{server_detail}" if server_detail else ""))
    except urllib.error.URLError as exc:
        raise OcAgyError(f"OpenCode Zen local streaming connection failed: {exc}")
    except TimeoutError as exc:
        raise OcAgyError(
            f"OpenCode Zen timed out after {timeout_seconds}s."
        ) from exc
    except OcAgyError as exc:
        server_detail = _zen_error_detail("\n".join(server_stderr))
        if server_detail and server_detail not in str(exc):
            raise OcAgyError(f"{exc}\n{server_detail}") from exc
        raise
    except Exception as exc:
        if httpx is not None and isinstance(exc, httpx.TimeoutException):
            raise OcAgyError(
                f"OpenCode Zen real-time stream timed out after {timeout_seconds}s."
            ) from exc
        if httpx is not None and isinstance(exc, httpx.RequestError):
            raise OcAgyError(
                f"OpenCode Zen local real-time stream failed: {exc}"
            ) from exc
        raise
    finally:
        if event_response is not None:
            try:
                event_response.close()
            except Exception:
                pass
        if event_stream_context is not None:
            try:
                event_stream_context.__exit__(None, None, None)
            except Exception:
                pass
        _terminate_process_tree(proc)
        with _ACTIVE_LOCK:
            _ACTIVE_PROCESSES.discard(proc)


def send_opencode_zen_completion(
    *, messages: List[Dict[str, Any]], model: str, temperature: float = 0.3,
    max_tokens: int = 65536, timeout: float = 1800,
    log_fn: Optional[Callable[[str], None]] = None, log_stream: bool = True,
) -> Dict[str, Any]:
    kwargs = dict(messages=messages, model=model, temperature=temperature,
                  max_tokens=max_tokens, timeout=timeout, log_fn=log_fn, log_stream=log_stream)
    from tor_proxy import TorProxyError, enabled, request_proxy
    if str(kwargs.get("model", "")).strip().startswith("ocz/") and enabled():
        try:
            with request_proxy(kwargs.get("log_fn"), is_cancelled) as proxy:
                try:
                    return _send_opencode_zen_completion_impl(**kwargs, proxy_url=proxy)
                except OcAgyError as exc:
                    if re.search(r'\b(?:403|429)\b|rate.?limit|too many requests|captcha|cloudflare|access denied',
                                 str(exc), re.IGNORECASE):
                        from tor_proxy import notify_block
                        notify_block(proxy, kwargs.get('log_fn'))
                    raise
        except TorProxyError as exc:
            raise OcAgyError(f"OpenCode Zen Tor proxy: {exc}") from exc
    return _send_opencode_zen_completion_impl(**kwargs)


def _send_opencode_zen_completion_impl(
    *,
    messages: List[Dict[str, Any]],
    model: str,
    temperature: float = 0.3,
    max_tokens: int = 65536,
    timeout: float = 1800,
    log_fn: Optional[Callable[[str], None]] = None,
    log_stream: bool = True,
    proxy_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Run one request through the OpenCode CLI for free-tier Zen models (ocz/).

    Uses ``opencode serve`` + SSE for real-time thinking/text streaming,
    matching the Antigravity provider's streaming behaviour.
    """
    if is_cancelled():
        raise OcAgyError("OpenCode Zen request cancelled by user")

    logger = log_fn or (lambda _message: None)
    exe = ensure_opencode_installed(log_fn=logger)

    raw_model = str(model or "").strip()
    for prefix in ("ocz/", "oc/", "opencode/", "opencode-go/"):
        if raw_model.startswith(prefix):
            raw_model = raw_model[len(prefix):]
            break
    if not raw_model:
        raise OcAgyError("No model specified for OpenCode Zen request.")
    effective_model = f"opencode/{raw_model}"

    prompt = build_prompt(messages)
    timeout_seconds = max(1.0, float(timeout or 1800))

    env = _subprocess_env()
    env["OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX"] = str(max(1, int(max_tokens)))
    # Apply only to ocz/; paid OpenCode aliases retain their existing transport.
    if proxy_url:
        from tor_proxy import proxy_environment
        proxy = proxy_url
        env = proxy_environment(env, proxy)
        proxy_address = urllib.parse.urlsplit(proxy)
        circuit_label = hashlib.sha256(proxy.encode("utf-8")).hexdigest()[:8]
        logger(f"🧄 OpenCode Zen: routing request through Tor "
               f"({proxy_address.hostname}:{proxy_address.port}, fresh circuit identity {circuit_label})")

    start = time.time()

    if log_stream:
        # Real-time streaming via opencode serve + SSE
        logger(
            f"🌐 OpenCode Zen: {Path(exe).name} serve --model {effective_model}"
        )
        result = _send_via_server_zen(
            exe=exe,
            prompt=prompt,
            model_id=effective_model,
            timeout_seconds=timeout_seconds,
            logger=logger,
            log_stream=True,
            subprocess_env=env,
        )
        elapsed = time.time() - start
        content = _clean_text(str(result.get("content", "") or ""))
        if not content:
            raise OcAgyError("OpenCode Zen returned an empty response.")
        logger(f"✅ OpenCode Zen: stream finished in {elapsed:.1f}s")
    else:
        # Buffered mode via opencode run (streaming disabled)
        logger(f"🌐 OpenCode Zen: {Path(exe).name} run --model {effective_model}")
        result = _run_opencode_zen_buffered(
            exe=exe,
            prompt=prompt,
            effective_model=effective_model,
            timeout_seconds=timeout_seconds,
            logger=logger,
            subprocess_env=env,
        )
        elapsed = time.time() - start
        content = _clean_text(str(result.get("content", "") or ""))
        if not content:
            raise OcAgyError("OpenCode Zen returned an empty response.")
        logger(f"✅ OpenCode Zen: completed in {elapsed:.1f}s")
    return {
        "content": content,
        "finish_reason": str(result.get("finish_reason", "") or "stop"),
        "raw_finish_reason": result.get("raw_finish_reason"),
        "finish_reason_source": str(
            result.get("finish_reason_source", "")
            or (
                "send_chat_completion_fallback_stop"
                if not result.get("finish_reason")
                else ""
            )
        ),
        "finish_reason_fallback": bool(
            result.get("finish_reason_fallback", False)
            or not result.get("finish_reason")
        ),
        "finish_reason_evidence": result.get("finish_reason_evidence") or {},
        "finish_reason_lookup_error": result.get("finish_reason_lookup_error", ""),
        "usage": result.get("usage"),
        "model": effective_model,
        "provider": "opencode-zen",
        "elapsed_seconds": elapsed,
        "stderr": "",
        "raw_response": result.get("raw_response"),
    }
