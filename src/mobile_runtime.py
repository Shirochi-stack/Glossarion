"""Runtime capability gates shared by the desktop app and Glossarion Mobile.

The desktop never sets any of the environment variables read here, so every
function returns the desktop behaviour by default (process pools allowed,
paths unchanged). The mobile app (src/mobile) sets GLOSSARION_MOBILE=1 and
GLOSSARION_NO_PROCESSES=1 before importing any backend module, because
Android has no sem_open for multiprocessing and iOS forbids spawning
processes.

Backend call sites gate on these helpers instead of on their own env
defaults, since the GUI-built run environment overwrites env defaults
(e.g. USE_ASYNC_CHAPTER_EXTRACTION='1').

Stdlib only; must stay importable on Python 3.10.
"""

import os
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

_MOBILE_PLATFORMS = ("android", "ios")
_NO_PROCESS_PLATFORMS = ("android", "ios", "emscripten", "wasi")


def _env_flag(name):
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def is_mobile():
    """True when running inside Glossarion Mobile (or a host simulation of it)."""
    if _env_flag("GLOSSARION_MOBILE"):
        return True
    if os.environ.get("FLET_PLATFORM", "").strip().lower() in _MOBILE_PLATFORMS:
        return True
    return sys.platform in _MOBILE_PLATFORMS


def processes_available():
    """True when the backend may create process pools / multiprocessing workers."""
    if _env_flag("GLOSSARION_NO_PROCESSES"):
        return False
    if is_mobile():
        return False
    return sys.platform not in _NO_PROCESS_PLATFORMS


def subprocesses_available():
    """True when the backend may launch helper subprocesses (sys.executable, CLIs)."""
    return processes_available()


def data_dir(default):
    """Writable app-data directory for files desktop keeps next to the code.

    Desktop never sets GLOSSARION_DATA_DIR, so ``default`` (the existing
    exe/script-relative expression at each call site) is returned unchanged.
    """
    override = os.environ.get("GLOSSARION_DATA_DIR", "").strip()
    return override if override else default


def config_file(default_dir):
    """Path of config.json: CONFIG_FILE env wins, else <data_dir(default_dir)>/config.json."""
    explicit = os.environ.get("CONFIG_FILE", "").strip()
    if explicit:
        return explicit
    return os.path.join(data_dir(default_dir), "config.json")


def make_pool_executor(max_workers, initializer=None, initargs=(),
                       thread_name_prefix="glossarion-pool"):
    """ProcessPoolExecutor on desktop, ThreadPoolExecutor when processes are unavailable.

    In thread mode a process-only ``initializer`` (nice/affinity/per-process
    globals) is deliberately NOT run, because it would affect the whole app.
    """
    workers = max(1, int(max_workers or 1))
    if processes_available():
        if initializer is not None:
            return ProcessPoolExecutor(max_workers=workers, initializer=initializer,
                                       initargs=tuple(initargs or ()))
        return ProcessPoolExecutor(max_workers=workers)
    return ThreadPoolExecutor(max_workers=workers, thread_name_prefix=thread_name_prefix)
