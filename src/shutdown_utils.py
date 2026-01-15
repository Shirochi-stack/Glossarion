"""
Shutdown utilities to ensure child processes are terminated and the interpreter exits.
Designed for Windows-first behavior, with best-effort cross-platform fallbacks.
"""

from __future__ import annotations

import os
import sys
import time
import subprocess
from typing import Callable, Iterable, Optional


def _normalize_exit_code(code) -> int:
    if code is None:
        return 0
    if isinstance(code, int):
        return code
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


def _terminate_multiprocessing_children(timeout: float = 1.5) -> None:
    try:
        import multiprocessing as mp
        children = mp.active_children()
        for p in children:
            try:
                p.terminate()
            except Exception:
                pass
        for p in children:
            try:
                p.join(timeout=timeout)
            except Exception:
                pass
    except Exception:
        pass


def _terminate_psutil_children(timeout: float = 1.5) -> None:
    try:
        import psutil
        parent = psutil.Process(os.getpid())
        children = parent.children(recursive=True)
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
    except Exception:
        pass


def _taskkill_self_tree() -> None:
    if os.name != "nt":
        return
    try:
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        subprocess.Popen(
            ["taskkill", "/F", "/T", "/PID", str(os.getpid())],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
        )
    except Exception:
        pass


def force_shutdown(exit_code: int = 0, cleanup_fns: Optional[Iterable[Callable[[], None]]] = None) -> None:
    """
    Best-effort cleanup then forcefully exit the current process.
    Terminates child processes (multiprocessing + psutil if available) and
    falls back to taskkill on Windows to ensure no background process remains.
    """
    code = _normalize_exit_code(exit_code)
    _run_cleanup_fns(cleanup_fns)
    _terminate_multiprocessing_children()
    _terminate_psutil_children()
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
