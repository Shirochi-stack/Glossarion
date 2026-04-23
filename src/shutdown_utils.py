"""
Shutdown utilities to ensure child processes are terminated and the interpreter exits.
Designed for Windows-first behavior, with best-effort cross-platform fallbacks.
"""

from __future__ import annotations

import os
import sys
import time
import subprocess
import tempfile
import shutil
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
        # Anything still alive after terminate/join: hard kill so it can
        # release file handles to DLLs under _MEIPASS before the bootloader
        # tries to clean the temp directory.
        for p in children:
            try:
                if p.is_alive():
                    if hasattr(p, "kill"):
                        p.kill()
                    else:
                        p.terminate()
                    p.join(timeout=timeout)
            except Exception:
                pass
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


def _terminate_psutil_children(timeout: float = 1.5) -> None:
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
        # Re-check after kill to make sure handles to _MEIPASS DLLs are gone.
        try:
            still = [p for p in alive if p.is_running()]
            if still:
                psutil.wait_procs(still, timeout=timeout)
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
    falls back to taskkill on Windows to ensure no background process remains.

    Ordering note: children must be terminated BEFORE we exit so that any
    DLLs they loaded from PyInstaller's `_MEIPASS` are unmapped. Otherwise
    the bootloader's final temp-dir cleanup pops a warning dialog.
    We deliberately do NOT touch `_MEIPASS` from Python; see
    `_cleanup_pyinstaller_temp_dir` for the rationale.
    """
    code = _normalize_exit_code(exit_code)
    _ensure_safe_tempdir()
    _run_cleanup_fns(cleanup_fns)
    # Kill descendants first so their handles to _MEIPASS drop before the
    # bootloader tries to rmtree it after we return.
    _terminate_multiprocessing_children()
    _terminate_psutil_children()
    _cleanup_pyinstaller_temp_dir()  # no-op, kept for backwards compatibility
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
