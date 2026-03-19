# dpi_setup.py — Cross-platform DPI-scaling bootstrap for PySide6
#
# Call  dpi_setup.configure()  ONCE, **before** importing any PySide6 modules.
# Safe to call multiple times (idempotent). Works on Windows, macOS, and Linux.
#
# What this does:
#   1. Sets QT_ENABLE_HIGHDPI_SCALING=0  → disables Qt6 automatic DPI scaling
#   2. Sets QT_FONT_DPI=96              → forces logical 96 DPI (1:1 rendering)
#   3. Sets QT_SCALE_FACTOR=<value>     → applies the user's chosen scale factor
#
# The scale factor is read from config.json ("gui_scale_factor" key).
# If the file is missing or unreadable the default 1.75 is used.

import json
import os
import sys

# Force UTF-8 console output to prevent UnicodeEncodeError on Windows cp1252
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

_configured = False

# Default scale factor when config.json has no value
DEFAULT_SCALE_FACTOR = 1.2


def _get_screen_resolution():
    """Return (width, height) of the primary monitor using stdlib only.

    Windows  → ctypes + GetSystemMetrics
    macOS    → subprocess + system_profiler (JSON)
    Linux    → subprocess + xrandr
    Returns (0, 0) on failure.
    """
    width, height = 0, 0

    # ── Windows ────────────────────────────────────────────────────────────
    if sys.platform == "win32":
        try:
            import ctypes
            user32 = ctypes.windll.user32
            try:
                ctypes.windll.shcore.SetProcessDpiAwareness(1)
            except Exception:
                try:
                    user32.SetProcessDPIAware()
                except Exception:
                    pass
            width = user32.GetSystemMetrics(0)
            height = user32.GetSystemMetrics(1)
        except Exception:
            pass

    # ── macOS ──────────────────────────────────────────────────────────────
    elif sys.platform == "darwin":
        try:
            import subprocess, re as _re
            out = subprocess.check_output(
                ["system_profiler", "SPDisplaysDataType"],
                timeout=5, stderr=subprocess.DEVNULL,
            ).decode("utf-8", errors="replace")
            # Look for "Resolution: 2560 x 1440" (or Retina variant)
            m = _re.search(r"Resolution:\s+(\d+)\s*x\s*(\d+)", out)
            if m:
                width, height = int(m.group(1)), int(m.group(2))
        except Exception:
            pass

    # ── Linux / X11 ────────────────────────────────────────────────────────
    else:
        try:
            import subprocess, re as _re
            out = subprocess.check_output(
                ["xrandr", "--current"],
                timeout=5, stderr=subprocess.DEVNULL,
            ).decode("utf-8", errors="replace")
            # Match the *connected primary* line first, fall back to any connected
            m = _re.search(r"connected primary\s+(\d+)x(\d+)", out)
            if not m:
                m = _re.search(r"connected\s+(\d+)x(\d+)", out)
            if m:
                width, height = int(m.group(1)), int(m.group(2))
        except Exception:
            pass

    return width, height


def _get_default_scale_for_resolution():
    """Return a sensible default scale factor based on the primary monitor's resolution.

    Works on Windows, macOS, and Linux (no PySide6 needed).
    Falls back to DEFAULT_SCALE_FACTOR if detection fails.
    """
    try:
        width, height = _get_screen_resolution()
        if width <= 0 or height <= 0:
            return DEFAULT_SCALE_FACTOR

        # Choose scale factor based on horizontal resolution
        if width >= 3840:       # 4K (3840×2160)
            return 1.7
        elif width >= 2560:     # 1440p / QHD (2560×1440)
            return 1.2
        elif width >= 1920:     # 1080p (1920×1080)
            return 1.0
        elif width >= 1366:     # 768p / common laptops
            return 1.0
        else:                   # 720p or lower
            return 0.9
    except Exception:
        return DEFAULT_SCALE_FACTOR


def _read_scale_factor():
    """Read gui_scale_factor from config.json (stdlib only, no PySide6)."""
    try:
        # In frozen (PyInstaller) builds, __file__ points to _MEIPASS temp dir.
        # config.json lives next to the executable, not inside the bundle.
        if getattr(sys, 'frozen', False) and hasattr(sys, 'executable'):
            base_dir = os.path.dirname(os.path.abspath(sys.executable))
        else:
            base_dir = os.path.dirname(os.path.abspath(__file__))
        cfg_path = os.path.join(base_dir, "config.json")
        if os.path.isfile(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            val = cfg.get("gui_scale_factor", None)
            if val is None:
                return _get_default_scale_for_resolution()
            factor = float(val)
            # Clamp to sane range
            if factor < 0.5:
                factor = 0.5
            elif factor > 3.0:
                factor = 3.0
            return factor
    except Exception:
        pass
    return _get_default_scale_for_resolution()


def configure():
    """Disable Qt6 automatic DPI scaling and apply the user's scale factor.

    Must be called *before* importing any PySide6/Qt modules.
    Idempotent — subsequent calls are harmless no-ops.
    """
    global _configured
    if _configured:
        return
    _configured = True

    # ── Disable Qt's built-in high-DPI scaling ────────────────────────────
    os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "0"

    # ── Lock font DPI to the traditional 96 ──────────────────────────────
    os.environ["QT_FONT_DPI"] = "96"

    # ── Apply user-configured scale factor ────────────────────────────────
    factor = _read_scale_factor()
    os.environ["QT_SCALE_FACTOR"] = str(factor)

    print(f"✅ DPI scaling configured (QT_SCALE_FACTOR={factor})")
