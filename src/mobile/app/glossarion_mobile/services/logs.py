"""Logs & diagnostics data (UI_SPEC §4.16; GUI-free, no Flet): HTTP logging, payload dumps, log files,
the redacted "Share logs bundle", the debug-cache size cap, the freeze watchdog and memory stats.

* HTTP logging: ``GLOSSARION_HTTP_LOG`` (bootstrap default "0") and ``GLOSSARION_HTTP_LOG_DIR`` =
  ``<logs>/http_requests`` for the next request (``http_logger.enable_detailed_http_logging``
  patches ``requests`` once; jobs read the variable when they start). Persisted in Prefs
  (``PREF_HTTP_LOG``) and re-applied at launch.
* Save payloads: the API client reads the environment variable ``SAVE_PAYLOAD`` (default "1"; there
  is no config key), so the switch is a Prefs value (``PREF_SAVE_PAYLOAD``) exported to the
  environment at launch and on change.
* Folders: ``<data>/Payloads`` (``unified_api_client._payloads_dir`` on mobile) and
  ``<logs>/http_requests``: size, clear (inside the app's own folders only), and the desktop
  400 MB cap (``shutdown_utils.sweep_large_caches``, moved from translator_gui) at launch.
* Log files: ``run.log`` (+ rotations), ``crash.log``, ``freeze.log`` under ``<logs>``.
* ``build_logs_bundle``: a zip of the log files with every config secret replaced by
  ``<REDACTED>`` plus ``environment.txt`` (``env_preview.redact_env``: secret names / values
  redacted) for sharing.
* ``FreezeWatchdog``: a heartbeat through the UI dispatcher; when the loop does not answer for
  ``stall_seconds`` every thread's stack goes to ``<logs>/freeze.log`` (faulthandler), once per
  stall.
* Memory stats: ``memory_usage_reporter.start_global_memory_logger`` / ``stop_global_memory_logger``
  (off by default; writes ``memory.log`` under ``GLOSSARION_LOG_DIR``). The desktop has the
  logger start commented out ("TEMPORARILY DISABLED to fix GIL issue"), so it stays opt-in.
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
import time
import zipfile
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

__all__ = [
    "FreezeWatchdog",
    "LogFile",
    "PREF_DEVELOPER",
    "PREF_HTTP_LOG",
    "PREF_MEMORY_STATS",
    "PREF_SAVE_PAYLOAD",
    "PRIORITY_REASON",
    "apply_http_logging",
    "apply_memory_stats",
    "apply_save_payload",
    "build_logs_bundle",
    "clear_debug_folder",
    "debug_folders",
    "folder_usage",
    "http_log_dir",
    "log_files",
    "payloads_dir",
    "redact_text",
    "sweep_debug_caches",
]

log = logging.getLogger("glossarion.diagnostics")

PREF_HTTP_LOG = "diagnostics_http_log"
PREF_SAVE_PAYLOAD = "diagnostics_save_payload"
PREF_MEMORY_STATS = "diagnostics_memory_stats"
PREF_DEVELOPER = "developer_mode"  # Library card ⋯ › Copy Path (ui/library/home.py)
PREF_CRASH_SEEN = "diagnostics_crash_seen"  # crash.log mtime the user dismissed the banner for

#: The schema rule's reason (settings_schema UNAVAILABLE_RULES, process priority / CPU affinity).
PRIORITY_REASON = "Process priority / CPU affinity cannot be changed by mobile apps."
CACHE_CAP_BYTES = 400 * 1024 * 1024  # the desktop cap (translator_gui._sweep_large_caches)
LOG_NAMES = ("run.log", "crash.log", "freeze.log", "memory.log")


@dataclass(frozen=True)
class LogFile:
    name: str
    path: str
    size: int
    mtime: float


# ---- folders ---------------------------------------------------------------------------------------


def payloads_dir(data_dir: Any) -> str:
    """Where the API client saves request/response dumps on mobile (``_payloads_dir``: the data dir)."""
    return os.path.join(os.fspath(data_dir), "Payloads") if data_dir else ""


def http_log_dir(logs_dir: Any) -> str:
    return os.path.join(os.fspath(logs_dir), "http_requests") if logs_dir else ""


def debug_folders(data_dir: Any, logs_dir: Any) -> list:
    """``[(id, label, path)]`` of the debug dump folders."""
    return [("payloads", "Payloads", payloads_dir(data_dir)), ("http_requests", "HTTP requests", http_log_dir(logs_dir))]


def folder_usage(path: str) -> tuple:
    """Blocking: ``(bytes, files)`` under ``path``."""
    total = files = 0
    if not path or not os.path.isdir(path):
        return 0, 0
    for root, _dirs, names in os.walk(path):
        for name in names:
            try:
                total += os.path.getsize(os.path.join(root, name))
                files += 1
            except OSError:
                pass
    return total, files


def clear_debug_folder(path: str, allowed_roots: Sequence[str]) -> int:
    """Blocking: delete the contents of a debug folder that lies inside one of ``allowed_roots``
    (the app's data / logs folders; anything else is refused). Returns the entries removed."""
    if not path or not os.path.isdir(path):
        return 0
    real = os.path.normcase(os.path.realpath(path))
    if not any(real.startswith(os.path.normcase(os.path.realpath(root)) + os.sep) for root in allowed_roots if root):
        raise ValueError("Only the app's own debug folders can be cleared")
    removed = 0
    for name in os.listdir(path):
        target = os.path.join(path, name)
        try:
            if os.path.isdir(target) and not os.path.islink(target):
                shutil.rmtree(target)
            else:
                os.remove(target)
            removed += 1
        except OSError as exc:
            log.info("could not remove %s: %s", target, exc)
    return removed


def sweep_debug_caches(data_dir: Any, logs_dir: Any, max_bytes: int = CACHE_CAP_BYTES) -> None:
    """Blocking: the desktop debug-cache cap over the app's Payloads / http_requests folders
    (``shutdown_utils.sweep_large_caches``, moved from translator_gui)."""
    try:
        from shutdown_utils import sweep_large_caches
    except Exception as exc:  # pragma: no cover - bundle without it
        log.info("cache sweep unavailable: %s", exc)
        return
    sweep_large_caches(max_bytes, "startup", extra_roots=[str(r) for r in (data_dir, logs_dir) if r])


# ---- switches --------------------------------------------------------------------------------------


def apply_http_logging(enabled: bool, logs_dir: Any) -> Optional[str]:
    """Turn HTTP request logging on / off for the next requests; returns the log folder when on."""
    if not enabled:
        os.environ["GLOSSARION_HTTP_LOG"] = "0"
        return None
    folder = http_log_dir(logs_dir)
    os.environ["GLOSSARION_HTTP_LOG"] = "1"
    if folder:
        os.environ["GLOSSARION_HTTP_LOG_DIR"] = folder
        os.makedirs(folder, exist_ok=True)
    try:
        import http_logger

        http_logger.enable_detailed_http_logging()
    except Exception as exc:
        log.info("http_logger unavailable: %s", exc)
    return folder


def apply_save_payload(enabled: bool) -> None:
    """``SAVE_PAYLOAD`` (the API client's request/response dumps; desktop default "1")."""
    os.environ["SAVE_PAYLOAD"] = "1" if enabled else "0"


def apply_memory_stats(enabled: bool) -> bool:
    """Start / stop the shared memory usage logger (off by default)."""
    try:
        import memory_usage_reporter as mur
    except Exception as exc:
        log.info("memory_usage_reporter unavailable: %s", exc)
        return False
    try:
        if enabled:
            mur.start_global_memory_logger()
        else:
            mur.stop_global_memory_logger()
    except Exception:
        log.exception("memory stats switch failed")
        return False
    return True


# ---- log files / bundle ------------------------------------------------------------------------------


def log_files(logs_dir: Any) -> list:
    """Blocking: the log files under ``logs_dir`` (run.log and its rotations, crash, freeze, memory)."""
    out: list = []
    if not logs_dir or not os.path.isdir(os.fspath(logs_dir)):
        return out
    folder = os.fspath(logs_dir)
    for name in sorted(os.listdir(folder)):
        if not any(name == base or name.startswith(base + ".") for base in LOG_NAMES):
            continue
        path = os.path.join(folder, name)
        try:
            stat = os.stat(path)
        except OSError:
            continue
        if os.path.isfile(path):
            out.append(LogFile(name, path, int(stat.st_size), float(stat.st_mtime)))
    return out


def redact_text(text: str, secrets: Iterable[str]) -> str:
    """Every known secret string replaced by ``<REDACTED>`` (longest first)."""
    out = str(text or "")
    for secret in sorted({s for s in secrets if isinstance(s, str) and len(s) >= 6}, key=len, reverse=True):
        out = out.replace(secret, "<REDACTED>")
    return out


def build_logs_bundle(logs_dir: Any, out_dir: Any, *, config: Optional[Mapping[str, Any]] = None,
                      env: Optional[Mapping[str, Any]] = None, extra_lines: Sequence[str] = (),
                      max_file_bytes: int = 8 * 1024 * 1024) -> str:
    """Blocking: ``<out_dir>/glossarion-logs-<time>.zip`` with the log files (secrets redacted, the
    last ``max_file_bytes`` of each) and ``environment.txt`` (redacted env rows + ``extra_lines``)."""
    from glossarion_mobile.ui.screens.env_preview import config_secrets, redact_env, rows_as_text

    secrets = config_secrets(dict(config or {}))
    os.makedirs(os.fspath(out_dir), exist_ok=True)
    path = os.path.join(os.fspath(out_dir), time.strftime("glossarion-logs-%Y%m%d-%H%M%S.zip"))
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for item in log_files(logs_dir):
            try:
                with open(item.path, "rb") as handle:
                    if item.size > max_file_bytes:
                        handle.seek(item.size - max_file_bytes)
                    data = handle.read().decode("utf-8", errors="replace")
            except OSError:
                continue
            archive.writestr(item.name + (".txt" if not item.name.endswith(".log") else ""), redact_text(data, secrets))
        rows = redact_env(dict(env if env is not None else os.environ), secrets)
        summary = "\n".join(list(extra_lines) + ["", "[environment]", rows_as_text(rows)])
        archive.writestr("environment.txt", redact_text(summary, secrets))
    return path


# ---- freeze watchdog ---------------------------------------------------------------------------------


class FreezeWatchdog:
    """UI-loop stall detector: ``post(fn)`` must run ``fn`` on the loop (``UiDispatcher.post``); when a
    heartbeat is not answered within ``stall_seconds`` every thread's stack is appended to
    ``<logs>/freeze.log`` (once per stall)."""

    def __init__(self, post: Callable[[Callable[[], Any]], Any], logs_dir: Any, *, interval: float = 5.0,
                 stall_seconds: float = 20.0) -> None:
        self.post = post
        self.path = os.path.join(os.fspath(logs_dir), "freeze.log") if logs_dir else ""
        self.interval = max(0.05, float(interval))
        self.stall_seconds = max(self.interval, float(stall_seconds))
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._answered = time.monotonic()
        self._pending_since: Optional[float] = None
        self._reported = False
        self.dumps = 0

    def _beat(self) -> None:
        self._answered = time.monotonic()
        self._pending_since = None
        self._reported = False

    def check(self, now: Optional[float] = None) -> bool:
        """One watchdog step: post a heartbeat, or dump when the pending one is overdue. True on a dump."""
        current = time.monotonic() if now is None else now
        if self._pending_since is None:
            self._pending_since = current
            try:
                self.post(self._beat)
            except Exception:
                self._pending_since = None
            return False
        if not self._reported and current - self._pending_since >= self.stall_seconds:
            self._reported = True
            self.dump(current - self._pending_since)
            return True
        return False

    def dump(self, stalled: float) -> None:
        if not self.path:
            return
        import faulthandler

        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(f"\n=== UI loop stalled {stalled:.1f} s at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
                handle.flush()
                faulthandler.dump_traceback(file=handle, all_threads=True)
            self.dumps += 1
            log.warning("UI loop stalled %.1f s; stacks written to %s", stalled, self.path)
        except Exception:
            log.exception("writing freeze.log failed")

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self.check()

    def start(self) -> "FreezeWatchdog":
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="gl-freeze-watchdog", daemon=True)
            self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
