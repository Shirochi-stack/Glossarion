# memory_usage_reporter.py
"""
Background memory usage reporter.
- Logs process RSS, VMS, peak (if available), GC counts, and optional tracemalloc stats
- Writes to logs/memory.log and also propagates to root logger (run.log) via a child logger
- Designed to be lightweight and safe in GUI apps
"""
import os
import sys
import time
import threading
import logging
import gc
from logging.handlers import RotatingFileHandler

try:
    import psutil
except Exception:
    psutil = None

# Global singletons
_GLOBAL_THREAD = None
_GLOBAL_STOP = threading.Event()


def _ensure_logs_dir() -> str:
    # Prefer explicit override from main app
    try:
        env_dir = os.environ.get("GLOSSARION_LOG_DIR")
        if env_dir:
            dir_path = os.path.expanduser(env_dir)
            os.makedirs(dir_path, exist_ok=True)
            return dir_path
    except Exception:
        pass

    def _can_write(p: str) -> bool:
        try:
            os.makedirs(p, exist_ok=True)
            test_file = os.path.join(p, ".write_test")
            with open(test_file, "w", encoding="utf-8") as f:
                f.write("ok")
            os.remove(test_file)
            return True
        except Exception:
            return False

    # Frozen exe: try next to the executable first
    try:
        if getattr(sys, 'frozen', False) and hasattr(sys, 'executable'):
            exe_dir = os.path.dirname(sys.executable)
            candidate = os.path.join(exe_dir, "logs")
            if _can_write(candidate):
                return candidate
    except Exception:
        pass

    # User-local app data (persistent and writable)
    try:
        base = os.environ.get('LOCALAPPDATA') or os.environ.get('APPDATA') or os.path.expanduser('~')
        candidate = os.path.join(base, 'Glossarion', 'logs')
        if _can_write(candidate):
            return candidate
    except Exception:
        pass

    # Development fallback: next to this file
    try:
        base_dir = os.path.abspath(os.path.dirname(__file__))
        candidate = os.path.join(base_dir, "logs")
        if _can_write(candidate):
            return candidate
    except Exception:
        pass

    # Final fallback: CWD
    fallback = os.path.join(os.getcwd(), "logs")
    os.makedirs(fallback, exist_ok=True)
    return fallback


def _make_logger() -> logging.Logger:
    logger = logging.getLogger("memory")
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers if called more than once
    if not any(isinstance(h, RotatingFileHandler) for h in logger.handlers):
        logs_dir = _ensure_logs_dir()
        file_path = os.path.join(logs_dir, "memory.log")
        fh = RotatingFileHandler(file_path, maxBytes=2 * 1024 * 1024, backupCount=3, encoding="utf-8")
        fmt = logging.Formatter(
            fmt="%(asctime)s %(levelname)s [%(process)d:%(threadName)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    # Do NOT propagate to root; keep memory logs out of console and only in memory.log
    logger.propagate = False
    return logger


def _get_process() -> "psutil.Process | None":
    if psutil is None:
        return None
    try:
        return psutil.Process()
    except Exception:
        return None


def _format_bytes(num: int) -> str:
    try:
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if num < 1024.0:
                return f"{num:,.1f}{unit}"
            num /= 1024.0
        return f"{num:,.1f}PB"
    except Exception:
        return str(num)


def _collect_stats(proc) -> dict:
    stats = {}
    try:
        if proc is not None:
            mi = proc.memory_info()
            stats["rss"] = mi.rss
            stats["vms"] = getattr(mi, "vms", 0)
            # Peak RSS on Windows via psutil.Process.memory_info() may expose peak_wset in private API; skip for portability
        else:
            stats["rss"] = 0
            stats["vms"] = 0
    except Exception:
        stats["rss"] = stats.get("rss", 0)
        stats["vms"] = stats.get("vms", 0)

    # GC stats
    try:
        counts = gc.get_count()
        stats["gc"] = counts
    except Exception:
        stats["gc"] = (0, 0, 0)

    return stats


def _worker(interval_sec: float, include_tracemalloc: bool):
    log = _make_logger()
    proc = _get_process()

    # Optional tracemalloc
    if include_tracemalloc:
        try:
            import tracemalloc
            if not tracemalloc.is_tracing():
                tracemalloc.start()
            tm_enabled = True
        except Exception:
            tm_enabled = False
    else:
        tm_enabled = False

    while not _GLOBAL_STOP.is_set():
        try:
            st = _collect_stats(proc)
            rss = st.get("rss", 0)
            vms = st.get("vms", 0)
            gc0, gc1, gc2 = st.get("gc", (0, 0, 0))

            msg = (
                f"RSS={_format_bytes(rss)} VMS={_format_bytes(vms)} "
                f"GC={gc0}/{gc1}/{gc2}"
            )

            if tm_enabled:
                try:
                    import tracemalloc
                    cur, peak = tracemalloc.get_traced_memory()
                    msg += f" TM_CUR={_format_bytes(cur)} TM_PEAK={_format_bytes(peak)}"
                except Exception:
                    pass

            log.info(msg)
        except Exception as e:
            try:
                log.warning("memory reporter error: %s", e)
            except Exception:
                pass
        finally:
            # Sleep in small chunks to react faster to stop
            for _ in range(int(max(1, interval_sec * 10))):
                if _GLOBAL_STOP.is_set():
                    break
                time.sleep(0.1)


def start_global_memory_logger(interval_sec: float = 3.0, include_tracemalloc: bool = False) -> None:
    """Start the background memory logger once per process.

    interval_sec: how often to log
    include_tracemalloc: if True, also log tracemalloc current/peak
    """
    global _GLOBAL_THREAD
    if _GLOBAL_THREAD and _GLOBAL_THREAD.is_alive():
        return

    _GLOBAL_STOP.clear()
    t = threading.Thread(target=_worker, args=(interval_sec, include_tracemalloc), name="mem-logger", daemon=True)
    _GLOBAL_THREAD = t
    try:
        t.start()
    except Exception:
        # Do not raise to avoid breaking GUI startup
        pass


def stop_global_memory_logger() -> None:
    try:
        _GLOBAL_STOP.set()
        if _GLOBAL_THREAD and _GLOBAL_THREAD.is_alive():
            # Give it a moment to exit
            _GLOBAL_THREAD.join(timeout=2.0)
    except Exception:
        pass
