"""Application directory and config.json location shared by desktop and mobile.

Moved verbatim from translator_gui.py (module level) so GUI-free code can
resolve the same paths without importing the Qt application. translator_gui
re-imports every name (``_APP_DIR``, ``CONFIG_FILE``, ``_atomic_json_write``,
``_get_app_dir``), so ``translator_gui.CONFIG_FILE`` and all existing call
sites resolve exactly as before. ``_atomic_json_write`` was later rewritten to be
truly atomic (no direct-write fallback, fsync, unique temp file, one writer lock).

The only override is the pre-existing ``GLOSSARION_APP_DIR`` seam; desktop
never sets it. Stdlib only; must stay importable on Python 3.10 without Qt.
"""

import json
import os
import platform
import sys
import tempfile
import threading
import time

# Resolve the application directory (where config.json, logs, etc. live).
# In frozen (PyInstaller) builds, this is next to the executable.
# In dev mode, this is next to the source file.
if getattr(sys, 'frozen', False) and hasattr(sys, 'executable'):
    _APP_DIR = os.path.dirname(os.path.abspath(sys.executable))
else:
    _APP_DIR = os.path.dirname(os.path.abspath(__file__))

_ENV_APP_DIR = os.environ.get("GLOSSARION_APP_DIR")
if _ENV_APP_DIR:
    try:
        os.makedirs(_ENV_APP_DIR, exist_ok=True)
        _APP_DIR = os.path.abspath(_ENV_APP_DIR)
    except OSError:
        pass

# On macOS .app bundles, App Translocation makes the bundle directory
# read-only.  Detect this and redirect config/data to a writable location.
if sys.platform == 'darwin' and getattr(sys, 'frozen', False):
    try:
        _test_path = os.path.join(_APP_DIR, ".write_test")
        with open(_test_path, "w") as _f:
            _f.write("ok")
        os.remove(_test_path)
    except OSError:
        # Bundle dir is read-only — use ~/Library/Application Support/Glossarion
        _mac_app_support = os.path.join(
            os.path.expanduser("~"), "Library", "Application Support", "Glossarion"
        )
        os.makedirs(_mac_app_support, exist_ok=True)
        # Migrate config from the bundle if it exists and hasn't been migrated yet
        _bundle_config = os.path.join(_APP_DIR, "config.json")
        _new_config = os.path.join(_mac_app_support, "config.json")
        if os.path.isfile(_bundle_config) and not os.path.isfile(_new_config):
            try:
                import shutil as _shutil
                _shutil.copy2(_bundle_config, _new_config)
            except Exception:
                pass
        _APP_DIR = _mac_app_support

CONFIG_FILE = os.path.join(_APP_DIR, "config.json")


# One writer at a time per process (GUI thread saves, the key manager's background save,
# dialogs that write config.json directly).
_ATOMIC_WRITE_LOCK = threading.RLock()

# Windows refuses os.replace onto a file another handle holds open (WinError 5/32: a
# reader in a worker thread or subprocess, antivirus, cloud sync); such locks are short.
_REPLACE_RETRY_SECONDS = 3.0


def _atomic_json_write(filepath, data):
    """Write JSON atomically: serialize, write a unique temp file, fsync, os.replace.

    The old file stays untouched until the complete new one replaces it, so a crash,
    power loss, a value that cannot be serialized or a dict changed by another thread
    never leaves a truncated file. On failure the temp file is removed and the error
    raised; there is no direct-write fallback (that truncated config.json whenever
    os.replace was refused on Windows and the dump then failed or was interrupted).
    """
    text = json.dumps(data, ensure_ascii=False, indent=2)
    directory = os.path.dirname(os.path.abspath(filepath))
    with _ATOMIC_WRITE_LOCK:
        fd, tmp_path = tempfile.mkstemp(
            dir=directory, prefix='.' + os.path.basename(filepath) + '.', suffix='.tmp')
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                f.write(text)
                f.flush()
                os.fsync(f.fileno())
            deadline = time.monotonic() + _REPLACE_RETRY_SECONDS
            while True:
                try:
                    os.replace(tmp_path, filepath)
                    break
                except PermissionError:
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(0.05)
        except BaseException:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            raise


def _get_app_dir() -> str:
    """Return the application's base directory.

    On Windows the CWD can be Downloads/Desktop when launching a .exe,
    so we always use the exe/script directory. On macOS/Linux we normally
    keep the launcher's CWD, but packaged apps can start at "/" or another
    unwritable directory. In that case, use the already-resolved writable
    app data directory.
    """
    if platform.system() == 'Windows':
        if getattr(sys, 'frozen', False):
            return os.path.dirname(sys.executable)
        return os.path.dirname(os.path.abspath(__file__))
    try:
        cwd = os.path.abspath(os.getcwd())
        if cwd == os.path.abspath(os.sep) or not os.access(cwd, os.W_OK):
            return _APP_DIR
        return cwd
    except Exception:
        return _APP_DIR


def config_file_path() -> str:
    """Current config.json path, read at call time (tests may patch ``CONFIG_FILE``)."""
    return CONFIG_FILE
