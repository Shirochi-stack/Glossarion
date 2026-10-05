"""Application directory and config.json location shared by desktop and mobile.

Moved verbatim from translator_gui.py (module level) so GUI-free code can
resolve the same paths without importing the Qt application. translator_gui
re-imports every name (``_APP_DIR``, ``CONFIG_FILE``, ``_atomic_json_write``,
``_get_app_dir``), so ``translator_gui.CONFIG_FILE`` and all existing call
sites resolve exactly as before.

The only override is the pre-existing ``GLOSSARION_APP_DIR`` seam; desktop
never sets it. Stdlib only; must stay importable on Python 3.10 without Qt.
"""

import json
import os
import platform
import sys

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


def _atomic_json_write(filepath, data):
    """Write JSON atomically using write-to-temp-then-rename.

    Prevents config corruption if the app crashes or power is lost mid-write.
    os.replace is atomic on POSIX and near-atomic on Windows.
    """
    tmp_path = filepath + ".tmp"
    try:
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, filepath)
    except Exception:
        # Fallback: direct write (better than losing data entirely)
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


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
