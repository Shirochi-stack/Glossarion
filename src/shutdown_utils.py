"""
Shutdown utilities to ensure child processes are terminated and the interpreter exits.
Designed for Windows-first behavior, with best-effort cross-platform fallbacks.
"""

from __future__ import annotations

import os
import sys
import json
import time
import subprocess
import tempfile
import shutil
import concurrent.futures
import uuid
from contextlib import contextmanager
from typing import Callable, Iterable, Optional


_last_qt_shutdown_drain_at = 0.0
_WINDOWS_CREATE_NO_WINDOW = 0x08000000
_WINDOWS_SW_HIDE = 0
_BROWSER_STATE_PROFILE_PARENT_DIRS = (
    "authnd_browser",
    "gemini_free_browser",
    "qtwebengine_requests",
)
_BROWSER_STATE_CLEANUP_HANDOFF_DIR = ".startup_cleanup_deleting"
_BROWSER_STATE_HELPER_ENV_VARS = (
    "AUTHND_TOKEN_HELPER",
    "GEMINI_FREE_HELPER",
)


def subprocess_no_window_kwargs(**kwargs):
    """Return subprocess kwargs that suppress transient console windows on Windows."""
    if os.name != "nt":
        return kwargs

    merged = dict(kwargs)
    try:
        creationflags = int(merged.get("creationflags") or 0)
    except Exception:
        creationflags = 0
    merged["creationflags"] = creationflags | getattr(
        subprocess,
        "CREATE_NO_WINDOW",
        _WINDOWS_CREATE_NO_WINDOW,
    )

    try:
        startupinfo = merged.get("startupinfo")
        if startupinfo is None:
            startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = _WINDOWS_SW_HIDE
        merged["startupinfo"] = startupinfo
    except Exception:
        pass

    return merged


def run_no_window(args, **kwargs):
    """Run a subprocess without flashing a console window on Windows."""
    return subprocess.run(args, **subprocess_no_window_kwargs(**kwargs))


def popen_no_window(args, **kwargs):
    """Start a subprocess without flashing a console window on Windows."""
    return subprocess.Popen(args, **subprocess_no_window_kwargs(**kwargs))


def _taskkill_pid_tree(pid, *, force: bool = True, timeout: float = 3.0) -> bool:
    if os.name != "nt" or not pid:
        return False
    args = ["taskkill"]
    if force:
        args.append("/F")
    args.extend(["/T", "/PID", str(pid)])
    try:
        run_no_window(
            args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
        )
        return True
    except Exception:
        return False


def terminate_subprocess_tree(proc, *, kill: bool = False, timeout: float = 3.0) -> None:
    """Terminate a Popen-style process and its children without console flashes."""
    try:
        if proc is None or proc.poll() is not None:
            return
    except Exception:
        return

    if os.name == "nt":
        try:
            if _taskkill_pid_tree(proc.pid, force=kill, timeout=timeout):
                return
        except Exception:
            pass

    try:
        if kill:
            proc.kill()
        else:
            proc.terminate()
    except Exception:
        pass


def _normalize_exit_code(code) -> int:
    if code is None:
        return 0
    if isinstance(code, int):
        return code
    return 1


def _cpu_worker_cap() -> int:
    """Return the user's logical CPU count as a safe worker ceiling."""
    try:
        return max(1, int(os.cpu_count() or 1))
    except Exception:
        return 1


def _run_cleanup_fns(cleanup_fns: Optional[Iterable[Callable[[], None]]]) -> None:
    if not cleanup_fns:
        return
    for fn in cleanup_fns:
        try:
            if callable(fn):
                fn()
        except Exception:
            # Best-effort cleanup
            pass


def _truthy_env(name: str) -> bool:
    return str(os.environ.get(name, "")).strip().lower() in ("1", "true", "yes", "on")


def _is_browser_state_helper_process() -> bool:
    return any(_truthy_env(name) for name in _BROWSER_STATE_HELPER_ENV_VARS)


def _add_glossarion_state_candidate(candidates, base, *parts) -> None:
    if not base:
        return
    try:
        path = os.path.join(os.path.expandvars(os.path.expanduser(str(base))), *parts)
        candidates.append(os.path.abspath(path))
    except Exception:
        pass


def glossarion_state_roots_for_shutdown() -> list[str]:
    """Return existing user-state roots that may contain browser-generated files."""
    candidates = []
    home = os.path.expanduser("~")
    if home and home != "~":
        _add_glossarion_state_candidate(candidates, home, ".glossarion")
        _add_glossarion_state_candidate(candidates, home, "Library", "Application Support", "Glossarion")
        _add_glossarion_state_candidate(candidates, home, ".local", "share", "Glossarion")
        _add_glossarion_state_candidate(candidates, home, ".config", "Glossarion")

    for env_name in ("APPDATA", "LOCALAPPDATA", "XDG_DATA_HOME", "XDG_CONFIG_HOME"):
        _add_glossarion_state_candidate(candidates, os.environ.get(env_name), "Glossarion")

    roots = []
    seen = set()
    for path in candidates:
        try:
            if not os.path.isdir(path):
                continue
            key = os.path.normcase(os.path.realpath(os.path.abspath(path)))
        except Exception:
            continue
        if key in seen:
            continue
        seen.add(key)
        roots.append(path)
    return roots


def _path_is_under_root(path: str, root: str) -> bool:
    try:
        path_abs = os.path.abspath(path)
        root_abs = os.path.abspath(root)
        return os.path.commonpath([path_abs, root_abs]) == root_abs
    except Exception:
        return False


def _browser_profile_parent(root: str, dirname: str) -> Optional[str]:
    if dirname not in _BROWSER_STATE_PROFILE_PARENT_DIRS:
        return None
    try:
        root_abs = os.path.abspath(root)
        parent = os.path.abspath(os.path.join(root_abs, dirname))
        if os.path.basename(parent) != dirname:
            return None
        if not _path_is_under_root(parent, root_abs):
            return None
        return parent
    except Exception:
        return None


def _validated_browser_profile_target(profile_dir: str, expected_parent_name: str) -> tuple[Optional[str], Optional[str]]:
    if expected_parent_name not in _BROWSER_STATE_PROFILE_PARENT_DIRS:
        return None, None
    try:
        target = os.path.abspath(os.path.expanduser(str(profile_dir or "")))
        parent = os.path.abspath(os.path.dirname(target))
        root = os.path.abspath(os.path.dirname(parent))
        if os.path.basename(parent) != expected_parent_name:
            return None, None
        if os.path.basename(target) in ("", ".", "..", expected_parent_name):
            return None, None
        if os.path.dirname(target) != parent:
            return None, None
        if not _path_is_under_root(target, parent):
            return None, None
        if os.path.normcase(target) == os.path.normcase(parent):
            return None, None
        return root, target
    except Exception:
        return None, None


def _rmtree_onerror(func, path, exc_info) -> None:
    del exc_info
    try:
        os.chmod(path, 0o700)
    except Exception:
        pass
    func(path)


def _remove_path_once(path: str) -> bool:
    if not path:
        return False
    try:
        if os.path.islink(path) or os.path.isfile(path):
            os.remove(path)
            return True
        if os.path.isdir(path):
            shutil.rmtree(path, onerror=_rmtree_onerror)
            return True
    except FileNotFoundError:
        return False
    return False


def _unique_cleanup_handoff_path(root: str, dirname: str) -> str:
    handoff_root = os.path.join(root, _BROWSER_STATE_CLEANUP_HANDOFF_DIR)
    os.makedirs(handoff_root, exist_ok=True)
    safe_name = os.path.basename(dirname.rstrip("/\\") or "cleanup")
    return os.path.join(handoff_root, f"{safe_name}-{os.getpid()}-{uuid.uuid4().hex}")


def _sweep_browser_cleanup_handoff(root: str, stats: dict) -> None:
    handoff_root = os.path.join(root, _BROWSER_STATE_CLEANUP_HANDOFF_DIR)
    try:
        entries = os.listdir(handoff_root)
    except FileNotFoundError:
        return
    except Exception:
        stats["failed"] += 1
        return

    for entry in entries:
        path = os.path.abspath(os.path.join(handoff_root, entry))
        if not _path_is_under_root(path, handoff_root):
            stats["failed"] += 1
            continue
        try:
            if _remove_path_once(path):
                stats["removed"] += 1
            elif os.path.lexists(path):
                stats["failed"] += 1
        except Exception:
            stats["failed"] += 1

    try:
        os.rmdir(handoff_root)
    except OSError:
        pass
    except Exception:
        pass


def _remove_or_handoff_browser_state(root: str, target: str, stats: dict) -> None:
    if not os.path.lexists(target):
        return

    try:
        if _remove_path_once(target):
            stats["removed"] += 1
            return
    except Exception:
        pass

    try:
        handoff_path = _unique_cleanup_handoff_path(root, os.path.basename(target))
        os.rename(target, handoff_path)
        stats["moved"] += 1
    except FileNotFoundError:
        return
    except Exception:
        stats["failed"] += 1
        return

    try:
        if _remove_path_once(handoff_path):
            stats["removed"] += 1
        elif os.path.lexists(handoff_path):
            stats["failed"] += 1
    except Exception:
        stats["failed"] += 1
    try:
        os.rmdir(os.path.dirname(handoff_path))
    except OSError:
        pass
    except Exception:
        pass


def cleanup_generated_browser_profile_dir(profile_dir: str, expected_parent_name: str) -> dict:
    """Remove one generated browser profile child directory, never its parent."""
    stats = {
        "removed": 0,
        "moved": 0,
        "failed": 0,
        "skipped": 0,
    }
    root, target = _validated_browser_profile_target(profile_dir, expected_parent_name)
    if not root or not target:
        stats["skipped"] = 1
        return stats
    if not os.path.lexists(target):
        return stats
    if not os.path.isdir(target) or os.path.islink(target):
        stats["skipped"] = 1
        return stats
    _remove_or_handoff_browser_state(root, target, stats)
    return stats


def _cleanup_browser_profile_children(parent_dir: str, expected_parent_name: str, stats: dict) -> None:
    try:
        entries = os.listdir(parent_dir)
    except FileNotFoundError:
        return
    except Exception:
        stats["failed"] += 1
        return

    for entry in entries:
        child = os.path.abspath(os.path.join(parent_dir, entry))
        child_stats = cleanup_generated_browser_profile_dir(child, expected_parent_name)
        stats["removed"] += child_stats.get("removed", 0)
        stats["moved"] += child_stats.get("moved", 0)
        stats["failed"] += child_stats.get("failed", 0)


def cleanup_browser_generated_state_for_shutdown() -> dict:
    """Remove generated Qt/Chromium browser state without touching user config."""
    stats = {
        "roots": 0,
        "removed": 0,
        "moved": 0,
        "failed": 0,
        "skipped": 0,
    }
    if _is_browser_state_helper_process():
        stats["skipped"] = 1
        return stats

    roots = glossarion_state_roots_for_shutdown()
    stats["roots"] = len(roots)
    for root in roots:
        _sweep_browser_cleanup_handoff(root, stats)
        for dirname in _BROWSER_STATE_PROFILE_PARENT_DIRS:
            parent = _browser_profile_parent(root, dirname)
            if not parent or not os.path.isdir(parent):
                continue
            _cleanup_browser_profile_children(parent, dirname, stats)

    if stats["removed"] or stats["moved"] or stats["failed"]:
        try:
            print(
                "[CLOSE] Browser state cleanup: "
                f"removed={stats['removed']}, moved={stats['moved']}, "
                f"failed={stats['failed']}, roots={stats['roots']}"
            )
        except Exception:
            pass
    return stats


def drain_qt_events_for_shutdown(duration_ms: int = 350, slice_ms: int = 50) -> None:
    """Let Qt process close/deleteLater work before a hard process exit.

    QtWebEngine keeps native GPU/VSync helper threads alive behind the
    widgets. A tiny event drain after closing WebEngine views gives those
    threads a chance to unwind before ``os._exit`` cuts the process off.
    """
    if duration_ms <= 0:
        return
    try:
        from PySide6.QtCore import QCoreApplication, QEvent, QEventLoop, QThread
    except Exception:
        return

    global _last_qt_shutdown_drain_at
    try:
        app = QCoreApplication.instance()
        if app is None:
            return
        try:
            if QThread.currentThread() != app.thread():
                return
        except Exception:
            pass

        deadline = time.monotonic() + (duration_ms / 1000.0)
        slice_ms = max(1, int(slice_ms or 1))
        while time.monotonic() < deadline:
            try:
                QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
                QCoreApplication.sendPostedEvents(None, 0)
                app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, slice_ms)
                QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            except Exception:
                try:
                    app.processEvents()
                except Exception:
                    return
            time.sleep(0.01)
        _last_qt_shutdown_drain_at = time.monotonic()
    except Exception:
        pass


def _terminate_multiprocessing_children(timeout: float = 1.5) -> None:
    try:
        import multiprocessing as mp
        children = mp.active_children()
        if not children:
            return
        for p in children:
            try:
                p.terminate()
            except Exception:
                pass

        def _join_or_kill(p):
            try:
                p.join(timeout=timeout)
            except Exception:
                pass
            try:
                if p.is_alive():
                    if hasattr(p, "kill"):
                        p.kill()
                    else:
                        p.terminate()
                    p.join(timeout=timeout)
            except Exception:
                pass

        # Anything still alive after terminate/join: hard kill so it can
        # release file handles to DLLs under _MEIPASS before the bootloader
        # tries to clean the temp directory. Join/kill in parallel so many
        # helper processes do not multiply the timeout.
        workers = min(len(children), _cpu_worker_cap(), 8)
        if workers <= 1:
            for p in children:
                _join_or_kill(p)
        else:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=workers,
                thread_name_prefix="shutdown-mp-child",
            ) as pool:
                list(pool.map(_join_or_kill, children))
    except Exception:
        pass


def _ensure_safe_tempdir() -> None:
    """
    Ensure a writable temp directory is set to avoid hangs when the default temp
    location is inaccessible (common on locked-down Windows environments).
    """
    try:
        candidates = []
        for var in ("TMPDIR", "TEMP", "TMP"):
            val = os.environ.get(var)
            if val:
                candidates.append(val)
        try:
            candidates.append(os.path.join(os.getcwd(), "_tmp"))
        except Exception:
            pass
        try:
            candidates.append(os.path.join(os.path.expanduser("~"), ".glossarion_tmp"))
        except Exception:
            pass

        for path in candidates:
            try:
                if not path:
                    continue
                os.makedirs(path, exist_ok=True)
                test_path = os.path.join(path, ".__tmp_test__")
                with open(test_path, "wb") as f:
                    f.write(b"x")
                try:
                    os.remove(test_path)
                except Exception:
                    pass
                os.environ["TMPDIR"] = path
                os.environ["TEMP"] = path
                os.environ["TMP"] = path
                tempfile.tempdir = path
                return
            except Exception:
                continue
    except Exception:
        pass


def _terminate_psutil_children(timeout: float = 1.5) -> int:
    """Terminate every descendant of the current process.

    Any grandchild that still has DLLs from `_MEIPASS` mapped will cause the
    PyInstaller bootloader's final `rmtree` to fail and pop the
    "Failed to remove temporary directory" dialog, so we err on the side of
    killing aggressively and then waiting for handles to drop.
    """
    try:
        import psutil
        parent = psutil.Process(os.getpid())
        children = parent.children(recursive=True)
        count = len(children)
        if not children:
            return 0
        for proc in children:
            try:
                proc.terminate()
            except Exception:
                pass
        try:
            gone, alive = psutil.wait_procs(children, timeout=timeout)
        except Exception:
            gone, alive = [], children
        for proc in alive:
            try:
                proc.kill()
            except Exception:
                pass
            try:
                _taskkill_pid_tree(proc.pid, force=True, timeout=min(1.0, max(0.2, timeout)))
            except Exception:
                pass
        # Re-check after kill to make sure handles to _MEIPASS DLLs are gone.
        try:
            still = [p for p in alive if p.is_running()]
            if still:
                psutil.wait_procs(still, timeout=timeout)
        except Exception:
            pass

        # Some helpers can spawn during teardown; do one final quick pass so
        # late children do not hold PyInstaller one-file DLLs open.
        try:
            late_children = parent.children(recursive=True)
            count = max(count, len(late_children))
            for proc in late_children:
                try:
                    proc.kill()
                except Exception:
                    pass
                try:
                    _taskkill_pid_tree(proc.pid, force=True, timeout=0.5)
                except Exception:
                    pass
            if late_children:
                psutil.wait_procs(late_children, timeout=min(0.5, max(0.1, timeout)))
        except Exception:
            pass
        return count
    except Exception:
        return 0


def _current_meipass_dir() -> str:
    try:
        meipass = getattr(sys, "_MEIPASS", "") or ""
        if not meipass:
            return ""
        return os.path.abspath(str(meipass))
    except Exception:
        return ""


def _path_points_under_meipass(value, meipass: str) -> bool:
    if not value or not meipass:
        return False
    try:
        path = os.path.normcase(os.path.abspath(str(value).strip().strip('"')))
        root = os.path.normcase(os.path.abspath(str(meipass)))
        return os.path.commonpath([path, root]) == root
    except Exception:
        return False


def _text_mentions_meipass(value, meipass: str) -> bool:
    if not value or not meipass:
        return False
    try:
        if isinstance(value, (list, tuple)):
            value = " ".join(str(part) for part in value)
        text = os.path.normcase(str(value).replace("/", "\\"))
        needle = os.path.normcase(str(meipass).replace("/", "\\"))
        return bool(needle and needle in text)
    except Exception:
        return False


def _process_info_points_to_meipass(info: dict, meipass: str) -> bool:
    if not isinstance(info, dict) or not meipass:
        return False
    for key in ("exe", "cwd"):
        if _path_points_under_meipass(info.get(key), meipass):
            return True
    return _text_mentions_meipass(info.get("cmdline"), meipass)


def _process_handles_meipass(proc, meipass: str, *, include_expensive: bool = False) -> bool:
    if not proc or not meipass:
        return False
    try:
        if _process_info_points_to_meipass(getattr(proc, "info", {}) or {}, meipass):
            return True
    except Exception:
        pass
    if not include_expensive:
        return False

    # Keep these checks opt-in only. On Windows, psutil.open_files() and
    # memory_maps() can block inside native process/file enumeration during
    # PyInstaller one-file teardown, which is exactly when force_shutdown()
    # needs to be quick and boring.
    for attr_name in ("open_files", "memory_maps"):
        try:
            entries = getattr(proc, attr_name)()
        except Exception:
            continue
        for entry in entries or []:
            try:
                path = getattr(entry, "path", None)
                if path is None and isinstance(entry, (tuple, list)) and entry:
                    path = entry[0]
                if _path_points_under_meipass(path, meipass):
                    return True
            except Exception:
                continue
    return False


def _terminate_psutil_meipass_lock_holders(timeout: float = 1.5) -> int:
    """Terminate non-current processes that still point at PyInstaller _MEIPASS."""
    meipass = _current_meipass_dir()
    if not meipass:
        return 0

    try:
        import psutil
        own_pid = os.getpid()
        matches = []
        allow_expensive_scan = os.environ.get(
            "GLOSSARION_SHUTDOWN_EXPENSIVE_MEIPASS_SCAN",
            "",
        ).strip().lower() in ("1", "true", "yes", "on")
        deadline = time.monotonic() + max(0.3, float(timeout or 0.3))
        for proc in psutil.process_iter(["pid", "ppid", "name", "exe", "cmdline", "cwd"]):
            try:
                pid = int(getattr(proc, "pid", 0) or (getattr(proc, "info", {}) or {}).get("pid", 0) or 0)
            except Exception:
                pid = 0
            if pid <= 0 or pid == own_pid:
                continue
            include_expensive = allow_expensive_scan and time.monotonic() < deadline
            if not _process_handles_meipass(proc, meipass, include_expensive=include_expensive):
                continue
            matches.append(proc)
    except Exception:
        return 0

    if not matches:
        return 0

    for proc in matches:
        try:
            proc.terminate()
        except Exception:
            pass
    try:
        _gone, alive = psutil.wait_procs(matches, timeout=max(0.2, float(timeout or 0.2)))
    except Exception:
        alive = matches
    for proc in alive:
        try:
            proc.kill()
        except Exception:
            pass
        try:
            _taskkill_pid_tree(proc.pid, force=True, timeout=min(1.0, max(0.2, timeout)))
        except Exception:
            pass
    try:
        if alive:
            psutil.wait_procs(alive, timeout=min(0.8, max(0.2, timeout)))
    except Exception:
        pass
    try:
        print(f"[CLOSE] Terminated {len(matches)} process(es) using _MEIPASS")
    except Exception:
        pass
    return len(matches)


def _parse_pid_lines(output) -> list[int]:
    pids = []
    try:
        text = output if isinstance(output, str) else (output or b"").decode(errors="ignore")
    except Exception:
        text = ""
    for line in text.splitlines():
        line = line.strip()
        if not line or not line.isdigit():
            continue
        try:
            pid = int(line)
        except Exception:
            continue
        if pid > 0:
            pids.append(pid)
    return pids


def _run_pid_query_command(args, timeout: float):
    proc = None
    try:
        proc = popen_no_window(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        stdout, _stderr = proc.communicate(timeout=timeout)
        return stdout or "", getattr(proc, "pid", None)
    except subprocess.TimeoutExpired:
        try:
            terminate_subprocess_tree(proc, kill=True, timeout=0.5)
        except Exception:
            pass
    except Exception:
        pass
    return "", getattr(proc, "pid", None) if proc is not None else None


def _windows_direct_child_pids(parent_pid: int, timeout: float = 0.8) -> list[int]:
    if os.name != "nt" or not parent_pid:
        return []

    # WMIC is fast when available. Newer Windows images may not include it, so
    # keep PowerShell/CIM as a fallback.
    try:
        stdout, query_pid = _run_pid_query_command(
            [
                "wmic",
                "process",
                "where",
                f"(ParentProcessId={int(parent_pid)})",
                "get",
                "ProcessId",
                "/value",
            ],
            timeout=timeout,
        )
        pids = []
        for line in (stdout or "").splitlines():
            line = line.strip()
            if not line.lower().startswith("processid="):
                continue
            value = line.split("=", 1)[1].strip()
            if value.isdigit():
                pid = int(value)
                if pid != query_pid and pid != parent_pid:
                    pids.append(pid)
        if pids:
            return pids
    except Exception:
        pass

    try:
        stdout, query_pid = _run_pid_query_command(
            [
                "powershell",
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                (
                    "Get-CimInstance Win32_Process "
                    f"-Filter 'ParentProcessId={int(parent_pid)}' "
                    "| ForEach-Object { $_.ProcessId }"
                ),
            ],
            timeout=timeout,
        )
        return [
            pid for pid in _parse_pid_lines(stdout)
            if pid != query_pid and pid != parent_pid
        ]
    except Exception:
        return []


def _windows_meipass_lock_holder_pids(meipass: str, timeout: float = 1.0) -> list[int]:
    if os.name != "nt" or not meipass:
        return []

    safe_meipass = str(meipass).replace("'", "''")
    own_pid = os.getpid()
    command = (
        f"$mei = '{safe_meipass}'; "
        f"$own = {int(own_pid)}; "
        "Get-CimInstance Win32_Process | "
        "Where-Object { "
        "$_.ProcessId -ne $own -and $_.ProcessId -ne $PID -and "
        "(("
        "$_.ExecutablePath -and $_.ExecutablePath.StartsWith($mei, [StringComparison]::OrdinalIgnoreCase)"
        ") -or ("
        "$_.CommandLine -and $_.CommandLine.IndexOf($mei, [StringComparison]::OrdinalIgnoreCase) -ge 0"
        "))"
        "} | ForEach-Object { $_.ProcessId }"
    )
    try:
        stdout, query_pid = _run_pid_query_command(
            [
                "powershell",
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                command,
            ],
            timeout=timeout,
        )
        return [
            pid for pid in _parse_pid_lines(stdout)
            if pid not in (own_pid, query_pid)
        ]
    except Exception:
        return []


def _terminate_windows_meipass_lock_holders(timeout: float = 1.5) -> int:
    meipass = _current_meipass_dir()
    if os.name != "nt" or not meipass:
        return 0
    pids = _windows_meipass_lock_holder_pids(meipass, timeout=min(1.0, max(0.2, timeout)))
    if not pids:
        return 0
    for pid in pids:
        try:
            _taskkill_pid_tree(pid, force=True, timeout=min(1.0, max(0.2, timeout)))
        except Exception:
            pass
    try:
        print(f"[CLOSE] Terminated {len(pids)} Windows process(es) referencing _MEIPASS")
    except Exception:
        pass
    return len(pids)


def _terminate_windows_descendants(timeout: float = 1.5) -> int:
    """Native Windows descendant sweep used when psutil is unavailable/misses."""
    if os.name != "nt":
        return 0

    root_pid = os.getpid()
    seen = {root_pid}
    queue = [root_pid]
    descendants = []
    deadline = time.monotonic() + max(0.2, float(timeout or 0.2))

    while queue and time.monotonic() < deadline:
        parent_pid = queue.pop(0)
        child_pids = _windows_direct_child_pids(parent_pid, timeout=0.6)
        for child_pid in child_pids:
            if child_pid in seen:
                continue
            seen.add(child_pid)
            descendants.append(child_pid)
            queue.append(child_pid)

    # Kill deepest children first, then direct children. taskkill /T on each PID
    # handles races where a child spawned another child after enumeration.
    for pid in reversed(descendants):
        try:
            _taskkill_pid_tree(pid, force=True, timeout=min(1.0, max(0.2, timeout)))
        except Exception:
            pass

    return len(descendants)


def terminate_current_process_children(timeout: float = 1.5) -> int:
    """Terminate descendants of this process; returns the initial child count."""
    count = _terminate_psutil_children(timeout=timeout)
    try:
        count = max(count, _terminate_windows_descendants(timeout=timeout))
    except Exception:
        pass
    return count


def _terminate_all_children_for_shutdown(timeout: float = 1.5) -> int:
    """Terminate Python and native child processes before one-file cleanup."""
    count = 0
    try:
        _terminate_multiprocessing_children(timeout=timeout)
    except Exception:
        pass
    try:
        count = max(count, _terminate_psutil_children(timeout=timeout))
    except Exception:
        pass
    try:
        count = max(count, _terminate_windows_descendants(timeout=timeout))
    except Exception:
        pass
    try:
        count = max(count, _terminate_psutil_meipass_lock_holders(timeout=timeout))
    except Exception:
        pass
    try:
        count = max(count, _terminate_windows_meipass_lock_holders(timeout=timeout))
    except Exception:
        pass
    try:
        count = max(count, _terminate_windows_descendants(timeout=min(0.8, max(0.2, timeout))))
    except Exception:
        pass
    return count


def _taskkill_self_tree() -> None:
    if os.name != "nt":
        return
    if os.environ.get("GLOSSARION_TASKKILL_SELF_ON_EXIT", "").strip().lower() not in ("1", "true", "yes", "on"):
        return
    try:
        popen_no_window(
            ["taskkill", "/F", "/T", "/PID", str(os.getpid())],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass


def request_hard_stop_for_shutdown(owner=None, translation_stop_flag=None, glossary_stop_flag=None) -> None:
    """Set all known immediate-stop flags before shutdown cleanup starts."""
    try:
        if owner is not None:
            owner.stop_requested = True
            owner.graceful_stop_active = False
        os.environ["TRANSLATION_CANCELLED"] = "1"
        os.environ["GRACEFUL_STOP"] = "0"
        os.environ["GRACEFUL_STOP_COMPLETED"] = "0"
        os.environ["WAIT_FOR_CHUNKS"] = "0"
    except Exception:
        pass

    try:
        stop_file = os.environ.get("GLOSSARY_STOP_FILE")
        if stop_file:
            with open(stop_file, "w", encoding="utf-8") as f:
                f.write("stop")
    except Exception:
        pass

    for setter in (translation_stop_flag, glossary_stop_flag):
        try:
            if setter:
                setter(True)
        except Exception:
            pass

    for module_name in ("TransateKRtoEN", "extract_glossary_from_epub", "GlossaryManager"):
        try:
            module = sys.modules.get(module_name)
            if hasattr(module, "set_stop_flag"):
                module.set_stop_flag(True)
        except Exception:
            pass

    try:
        import unified_api_client
        if hasattr(unified_api_client, "set_stop_flag"):
            unified_api_client.set_stop_flag(True)
        if hasattr(unified_api_client, "global_stop_flag"):
            unified_api_client.global_stop_flag = True
        if hasattr(unified_api_client, "UnifiedClient"):
            unified_api_client.UnifiedClient._global_cancelled = True
            if hasattr(unified_api_client.UnifiedClient, "set_global_cancellation"):
                unified_api_client.UnifiedClient.set_global_cancellation(True)
            # Fire the browser-backed route cancels (AuthND / AuthGPT / AuthGem /
            # AuthCD / Gemini-Free) and close in-flight HTTP sessions. This sets
            # each helper module's _cancel_event and terminates their token-helper
            # subprocesses, so a request that was mid-flight when the user hit Stop
            # cannot keep running — and its Chromium child cannot linger holding
            # PyInstaller _MEIPASS DLLs into the bootloader's temp-dir cleanup
            # (the "Failed to remove temporary directory" warning).
            if hasattr(unified_api_client.UnifiedClient, "hard_cancel_all"):
                try:
                    unified_api_client.UnifiedClient.hard_cancel_all()
                except Exception:
                    pass
    except Exception:
        pass

    # Best-effort direct cancel of each browser-backed helper module, in case the
    # unified client's hard_cancel_all path was unavailable (e.g. import order).
    for _helper_module in (
        "authnd_auth",
        "authgpt_auth",
        "authgem_auth",
        "authcd_auth",
        "gemini_free",
    ):
        try:
            module = sys.modules.get(_helper_module)
            if module is not None and hasattr(module, "cancel_stream"):
                module.cancel_stream()
        except Exception:
            pass


def _dedupe_paths(paths):
    seen = set()
    unique = []
    for path in paths:
        if not path:
            continue
        try:
            norm = os.path.normcase(os.path.abspath(path))
        except Exception:
            norm = str(path)
        if norm in seen:
            continue
        seen.add(norm)
        unique.append(path)
    return unique


def _selected_input_files_for_shutdown(input_files=None, entry_file=None):
    files = list(input_files or [])
    if entry_file and os.path.isfile(str(entry_file)):
        files.append(str(entry_file))
    return [
        file_path for file_path in files
        if file_path and file_path != "__generative_mode__"
    ]


def translation_progress_paths_for_shutdown(input_files=None, entry_file=None, output_dir_resolver=None):
    paths = []
    for file_path in _selected_input_files_for_shutdown(input_files, entry_file):
        try:
            output_dir = output_dir_resolver(file_path) if output_dir_resolver else None
            if output_dir:
                paths.append(os.path.join(str(output_dir), "translation_progress.json"))
        except Exception:
            pass
    return _dedupe_paths(paths)


def glossary_progress_paths_for_shutdown(
    input_files=None,
    entry_file=None,
    output_dir_resolver=None,
    config=None,
    app_dir=None,
    output_path=None,
):
    paths = []
    files = _selected_input_files_for_shutdown(input_files, entry_file)
    roots = []

    def _add_root(root):
        if not root:
            return
        try:
            roots.append(os.path.abspath(str(root)))
        except Exception:
            pass

    try:
        override_dir = os.environ.get("OUTPUT_DIRECTORY") or (config or {}).get("output_directory")
    except Exception:
        override_dir = os.environ.get("OUTPUT_DIRECTORY")
    if override_dir:
        _add_root(os.path.join(override_dir, "Glossary"))
    _add_root(os.environ.get("GLOSSARY_SHARED_DIR"))
    _add_root(app_dir and os.path.join(app_dir, "Glossary"))
    if files:
        try:
            _add_root(os.path.join(os.path.dirname(os.path.abspath(files[0])), "Glossary"))
        except Exception:
            pass

    for file_path in files:
        try:
            base = os.path.splitext(os.path.basename(file_path))[0]
            for root in roots:
                paths.append(os.path.join(root, base, f"{base}_glossary_progress.json"))
                paths.append(os.path.join(root, f"{base}_glossary_progress.json"))
            try:
                output_dir = output_dir_resolver(file_path) if output_dir_resolver else None
                if output_dir:
                    paths.append(os.path.join(output_dir, "glossary_progress.json"))
                    paths.append(os.path.join(output_dir, "Glossary", f"{base}_glossary_progress.json"))
            except Exception:
                pass
        except Exception:
            pass

    try:
        output_path = (output_path or os.environ.get("OUTPUT_PATH", "")).strip()
        if output_path:
            out_dir = os.path.dirname(os.path.abspath(output_path))
            base = os.path.splitext(os.path.basename(output_path))[0]
            if base.lower().endswith("_glossary"):
                base = base[:-len("_glossary")]
            paths.append(os.path.join(out_dir, f"{base}_glossary_progress.json"))
    except Exception:
        pass

    try:
        extract_module = sys.modules.get("extract_glossary_from_epub")
        module_progress = getattr(extract_module, "PROGRESS_FILE", "")
        if module_progress and (
            os.path.isabs(str(module_progress))
            or os.path.basename(str(module_progress)).lower() != "glossary_progress.json"
        ):
            paths.append(module_progress)
    except Exception:
        pass

    return _dedupe_paths(paths)


def _unique_int_list(values):
    seen = set()
    result = []
    for value in values or []:
        try:
            idx = int(value)
        except (TypeError, ValueError):
            continue
        if idx not in seen:
            seen.add(idx)
            result.append(idx)
    return result


def _remove_idx(values, idx):
    return [value for value in _unique_int_list(values) if value != idx]


def _add_idx(values, idx):
    values = _unique_int_list(values)
    if idx not in values:
        values.append(idx)
    return values


def _progress_entry_index(info, key=None):
    if isinstance(info, dict):
        for idx_key in ("chapter_index", "idx", "index"):
            try:
                return int(info.get(idx_key))
            except (TypeError, ValueError):
                pass
    try:
        return int(key)
    except (TypeError, ValueError):
        return None


def _progress_chapter_key(idx):
    try:
        return str(int(idx))
    except (TypeError, ValueError):
        return str(idx)


def _progress_filename(value):
    name = os.path.basename(str(value or "").strip())
    if not name:
        return ""
    if os.path.splitext(name)[1].lower() in (".epub", ".pdf", ".zip", ".cbz"):
        return ""
    return name


def _progress_field_value(progress_data, field_name, idx):
    values = progress_data.get(field_name, {})
    if not isinstance(values, dict):
        return None
    return values.get(str(idx), values.get(idx))


def _restore_previous_progress_entry(info):
    if not isinstance(info, dict):
        return None
    previous_status = str(info.get("previous_status", "") or "").lower()
    previous_entry = info.get("previous_progress_entry")
    if isinstance(previous_entry, dict):
        restored = dict(previous_entry)
        restored_status = str(restored.get("status", previous_status) or previous_status).lower()
        if restored_status and restored_status not in ("in_progress", "not_completed", "not translated", "not_translated"):
            restored.pop("previous_status", None)
            restored.pop("previous_progress_entry", None)
            restored.pop("previous_status_unknown", None)
            return restored

    if previous_status in ("qa_failed", "failed", "error", "pending", "merged", "completed"):
        restored = dict(info)
        restored["status"] = "failed" if previous_status == "error" else previous_status
        restored.pop("previous_status", None)
        restored.pop("previous_progress_entry", None)
        restored.pop("previous_status_unknown", None)
        return restored

    if info.get("previous_status_unknown"):
        restored = dict(info)
        restored["status"] = "failed"
        restored.pop("previous_status", None)
        restored.pop("previous_progress_entry", None)
        restored.pop("previous_status_unknown", None)
        return restored

    if previous_status in ("not_completed", "not translated", "not_translated", ""):
        if previous_status:
            return None
        if info.get("output_file"):
            restored = dict(info)
            restored["status"] = "failed"
            restored.pop("previous_status", None)
            restored.pop("previous_progress_entry", None)
            restored.pop("previous_status_unknown", None)
            return restored
    return None


def _failed_from_in_progress_entry(info):
    if not isinstance(info, dict):
        info = {}
    failed = dict(info)
    failed["status"] = "failed"
    failed.pop("previous_status", None)
    failed.pop("previous_progress_entry", None)
    failed.pop("previous_status_unknown", None)
    return failed


def _atomic_replace_progress_file(temp_path, target_path, max_retries=3, delay=0.5):
    last_err = None
    for attempt in range(max_retries):
        try:
            os.replace(temp_path, target_path)
            return
        except (PermissionError, OSError) as e:
            last_err = e
            if attempt < max_retries - 1:
                time.sleep(delay)
    if os.path.exists(temp_path):
        try:
            os.remove(temp_path)
        except Exception:
            pass
    if last_err:
        raise last_err


@contextmanager
def _locked_progress_file(progress_file):
    if not progress_file:
        yield
        return

    progress_dir = os.path.dirname(progress_file) or "."
    os.makedirs(progress_dir, exist_ok=True)
    lock_path = f"{progress_file}.lock"
    lock_f = open(lock_path, "a+b")
    locked = False
    try:
        if lock_f.seek(0, os.SEEK_END) == 0:
            lock_f.write(b"\0")
            lock_f.flush()
            os.fsync(lock_f.fileno())
        lock_f.seek(0)

        if os.name == "nt":
            import msvcrt
            while True:
                try:
                    lock_f.seek(0)
                    msvcrt.locking(lock_f.fileno(), msvcrt.LK_NBLCK, 1)
                    locked = True
                    break
                except OSError:
                    time.sleep(0.05)
        else:
            import fcntl
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
            locked = True

        yield
    finally:
        try:
            if locked:
                lock_f.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(lock_f.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)
        finally:
            lock_f.close()


def restore_glossary_in_progress_for_hard_stop(progress_file):
    """Resolve persisted glossary in-progress rows before an immediate shutdown."""
    if not progress_file or not os.path.exists(progress_file):
        return 0

    with _locked_progress_file(progress_file):
        try:
            with open(progress_file, "r", encoding="utf-8") as f:
                progress_data = json.load(f)
        except Exception:
            return 0
        if not isinstance(progress_data, dict):
            return 0

        chapters = progress_data.get("chapters", {})
        if not isinstance(chapters, dict):
            chapters = {}

        completed_clean = _unique_int_list(progress_data.get("completed", []))
        failed_clean = _unique_int_list(progress_data.get("failed", []))
        merged_clean = _unique_int_list(progress_data.get("merged_indices", []))
        in_progress_clean = _unique_int_list(progress_data.get("in_progress", []))

        changed = 0
        processed_indices = set()

        def _apply_resolved_status(idx, resolved_status):
            nonlocal completed_clean, failed_clean, merged_clean
            if idx is None:
                return
            completed_clean = _remove_idx(completed_clean, idx)
            failed_clean = _remove_idx(failed_clean, idx)
            merged_clean = _remove_idx(merged_clean, idx)
            if resolved_status == "completed":
                completed_clean = _add_idx(completed_clean, idx)
            elif resolved_status == "merged":
                merged_clean = _add_idx(merged_clean, idx)
            elif resolved_status in ("failed", "qa_failed", "error"):
                failed_clean = _add_idx(failed_clean, idx)

        for chapter_key, chapter_info in list(chapters.items()):
            if not isinstance(chapter_info, dict):
                continue
            if str(chapter_info.get("status", "")).lower() != "in_progress":
                continue
            idx = _progress_entry_index(chapter_info, chapter_key)
            if idx is not None:
                processed_indices.add(idx)

            restored = _restore_previous_progress_entry(chapter_info)
            if restored:
                chapters[chapter_key] = restored
                resolved_status = str(restored.get("status", "") or "").lower()
            else:
                failed = _failed_from_in_progress_entry(chapter_info)
                chapters[chapter_key] = failed
                resolved_status = str(failed.get("status", "failed") or "failed").lower()
            _apply_resolved_status(idx, resolved_status)
            changed += 1

        for idx in in_progress_clean:
            if idx in processed_indices:
                continue
            chapter_key = _progress_chapter_key(idx)
            existing_info = chapters.get(chapter_key)
            if isinstance(existing_info, dict) and str(existing_info.get("status", "")).lower() == "in_progress":
                continue
            if isinstance(existing_info, dict) and existing_info:
                existing_status = str(existing_info.get("status", "") or "").lower()
                processed_indices.add(idx)
                _apply_resolved_status(idx, existing_status)
                changed += 1
                continue

            actual_num = _progress_field_value(progress_data, "chapter_numbers", idx)
            try:
                actual_num = int(actual_num)
            except (TypeError, ValueError):
                actual_num = idx + 1
            chapter_info = {
                "chapter_index": idx,
                "actual_num": actual_num,
                "chapter_num": actual_num,
                "status": "failed",
                "last_updated": time.time(),
            }
            chapter_file = _progress_filename(_progress_field_value(progress_data, "chapter_filenames", idx))
            if chapter_file:
                chapter_info["output_file"] = chapter_file
            for key in ("model_name", "key_identifier", "key_pool"):
                if progress_data.get(key):
                    chapter_info[key] = progress_data[key]
            chapters[chapter_key] = chapter_info
            processed_indices.add(idx)
            _apply_resolved_status(idx, "failed")
            changed += 1

        if not changed:
            return 0

        progress_data["chapters"] = chapters
        progress_data["completed"] = completed_clean
        progress_data["failed"] = failed_clean
        progress_data["merged_indices"] = merged_clean
        progress_data["in_progress"] = [idx for idx in in_progress_clean if idx not in processed_indices]

        temp_path = None
        try:
            progress_dir = os.path.dirname(progress_file) or "."
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=progress_dir, delete=False, suffix=".tmp") as temp_f:
                temp_path = temp_f.name
                json.dump(progress_data, temp_f, ensure_ascii=False, indent=2)
                temp_f.flush()
                os.fsync(temp_f.fileno())
            _atomic_replace_progress_file(temp_path, progress_file)
        except Exception:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
            raise

        return changed


def _restore_image_progress_manager_for_shutdown(manager) -> int:
    try:
        progress = getattr(manager, "prog", None)
        images = progress.get("images", {}) if isinstance(progress, dict) else {}
        if not isinstance(images, dict):
            return 0
        changed = 0
        now = time.time()
        for content_hash, image_info in list(images.items()):
            if not isinstance(image_info, dict):
                continue
            if str(image_info.get("status", "")).lower() != "in_progress":
                continue
            updated = dict(image_info)
            updated["status"] = "cancelled"
            updated["last_updated"] = now
            updated["cancelled_at"] = now
            images[content_hash] = updated
            changed += 1
        if changed and hasattr(manager, "save"):
            manager.save()
        return changed
    except Exception:
        return 0


def restore_in_progress_rows_for_shutdown(
    input_files=None,
    entry_file=None,
    output_dir_resolver=None,
    config=None,
    app_dir=None,
    image_progress_managers=None,
    max_workers: int = 8,
) -> None:
    """Resolve persisted in-progress rows in parallel before a forced exit."""
    cleanup_tasks = []

    try:
        from TransateKRtoEN import ProgressManager
    except Exception:
        ProgressManager = None

    def _restore_translation_progress(progress_file):
        if not progress_file or not os.path.exists(progress_file) or ProgressManager is None:
            return 0
        manager = ProgressManager(os.path.dirname(progress_file))
        changed = manager.restore_all_in_progress_for_hard_stop()
        if changed:
            manager.save()
        return changed

    for progress_file in translation_progress_paths_for_shutdown(input_files, entry_file, output_dir_resolver):
        if progress_file and os.path.exists(progress_file) and ProgressManager is not None:
            cleanup_tasks.append((
                f"translation progress {progress_file}",
                _restore_translation_progress,
                progress_file,
            ))

    def _restore_glossary_progress(progress_file):
        if not progress_file or not os.path.exists(progress_file):
            return 0
        return restore_glossary_in_progress_for_hard_stop(progress_file)

    for progress_file in glossary_progress_paths_for_shutdown(
        input_files=input_files,
        entry_file=entry_file,
        output_dir_resolver=output_dir_resolver,
        config=config,
        app_dir=app_dir,
    ):
        if progress_file and os.path.exists(progress_file):
            cleanup_tasks.append((
                f"glossary progress {progress_file}",
                _restore_glossary_progress,
                progress_file,
            ))

    for name, manager in image_progress_managers or []:
        if manager is not None:
            cleanup_tasks.append((
                f"image progress manager {name}",
                _restore_image_progress_manager_for_shutdown,
                manager,
            ))

    if not cleanup_tasks:
        return

    cpu_cap = _cpu_worker_cap()
    requested_workers = max(1, int(max_workers or 1))
    workers = min(requested_workers, len(cleanup_tasks), cpu_cap)
    print(
        f"[CLOSE] Shutdown progress cleanup workers: {workers} "
        f"(cpu_cap={cpu_cap}, tasks={len(cleanup_tasks)}, "
        f"requested={requested_workers})"
    )
    try:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="shutdown-progress-cleanup",
        ) as pool:
            future_map = {
                pool.submit(task_fn, task_arg): task_name
                for task_name, task_fn, task_arg in cleanup_tasks
            }
            for future in concurrent.futures.as_completed(future_map):
                task_name = future_map[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"[CLOSE] Warning: could not restore {task_name}: {e}")
    except Exception as e:
        print(f"[CLOSE] Warning: parallel shutdown progress cleanup failed: {e}")


def _cleanup_pyinstaller_temp_dir(retries: int = 4, delay: float = 0.2) -> None:
    """Intentional no-op.

    Previously this tried to `shutil.rmtree(sys._MEIPASS)` from inside the
    frozen child process. That is actively harmful: DLLs/PYDs loaded from
    `_MEIPASS` are still mapped into this process, so the tree ends up only
    partially deleted. The PyInstaller bootloader then runs its own cleanup
    after we exit, walks the half-deleted tree, fails on the mapped files,
    and shows the

        "Failed to remove temporary directory: ...\\_MEIxxxxxx"

    warning dialog. There is no way to suppress that dialog from Python
    because the bootloader shows it *after* our interpreter is already gone.

    The correct fix is to leave `_MEIPASS` alone and make sure every child
    process is dead (so its mapped DLLs get unmapped) before we exit. Then
    the bootloader's own cleanup succeeds and no dialog is ever shown.
    """
    return


def force_shutdown(exit_code: int = 0, cleanup_fns: Optional[Iterable[Callable[[], None]]] = None) -> None:
    """
    Best-effort cleanup then forcefully exit the current process.
    Terminates child processes (multiprocessing + psutil if available) and
    can optionally fall back to taskkill on Windows via
    GLOSSARION_TASKKILL_SELF_ON_EXIT=1.

    Ordering note: children must be terminated BEFORE we exit so that any
    DLLs they loaded from PyInstaller's `_MEIPASS` are unmapped. Otherwise
    the bootloader's final temp-dir cleanup pops a warning dialog.
    We deliberately do NOT touch `_MEIPASS` from Python; see
    `_cleanup_pyinstaller_temp_dir` for the rationale.
    """
    code = _normalize_exit_code(exit_code)
    _ensure_safe_tempdir()
    _run_cleanup_fns(cleanup_fns)
    if time.monotonic() - _last_qt_shutdown_drain_at > 0.75:
        drain_qt_events_for_shutdown(duration_ms=350)
    else:
        print("[CLOSE] Skipping duplicate Qt shutdown event drain")
    # Kill descendants first so their handles to _MEIPASS drop before the
    # bootloader tries to rmtree it after we return. The helper includes a
    # native Windows fallback for Lite builds where psutil is missing/broken.
    _terminate_all_children_for_shutdown(timeout=1.5)
    cleanup_browser_generated_state_for_shutdown()
    _cleanup_pyinstaller_temp_dir()  # no-op, kept for backwards compatibility
    # Disabled by default. Killing our own process tree with taskkill can
    # interrupt Qt/PySide/native DLL teardown at an arbitrary instruction and
    # show Windows' "memory could not be read" dialog. Child processes are
    # already handled above; this remains available as an opt-in last resort.
    _taskkill_self_tree()
    try:
        os._exit(code)
    except Exception:
        sys.exit(code)


def run_cli_main(main_fn: Callable[[], Optional[int]], cleanup_fns: Optional[Iterable[Callable[[], None]]] = None) -> None:
    """
    Run a CLI-style main function and always force shutdown with the appropriate exit code.
    """
    exit_code = 0
    try:
        result = main_fn()
        if isinstance(result, int):
            exit_code = result
    except SystemExit as e:
        if isinstance(e.code, str):
            try:
                print(e.code, file=sys.stderr)
            except Exception:
                pass
        exit_code = _normalize_exit_code(e.code)
    except Exception as e:
        try:
            print(f"❌ Unhandled error: {e}", file=sys.stderr)
        except Exception:
            pass
        exit_code = 1
    finally:
        force_shutdown(exit_code, cleanup_fns=cleanup_fns)
