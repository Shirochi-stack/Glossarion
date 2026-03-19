# dpi_setup.py — Cross-platform DPI-scaling bootstrap for PySide6
#
# Call  dpi_setup.configure()  ONCE, **before** importing any PySide6 modules.
# Safe to call multiple times (idempotent). Works on Windows, macOS, and Linux.
#
# What this does:
#   1. Sets QT_ENABLE_HIGHDPI_SCALING=1  → enables Qt6 native high-DPI rendering
#      (text is rasterised at the correct resolution → sharp glyphs)
#   2. Detects the OS-level DPI scale (e.g. 1.5× on a 150 % Windows display)
#   3. Sets QT_SCALE_FACTOR = desired / system_scale so the user's configured
#      factor represents the *total* scale, not an additional multiplier.
#
# The scale factor is read from config.json ("gui_scale_factor" key).
# If the file is missing or unreadable a resolution-based default is used.

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
DEFAULT_SCALE_FACTOR = 1.0


def _ensure_dpi_aware():
    """Make the process DPI-aware on Windows (idempotent, no-op on other platforms)."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass
    except Exception:
        pass


def _get_system_dpi_scale():
    """Return the OS-level DPI scale factor (e.g. 1.0, 1.25, 1.5, 2.0).

    Windows  → GetDpiForSystem / GetDeviceCaps
    macOS    → Retina backing-scale from system_profiler (2.0 for Retina, 1.0 otherwise)
    Linux    → Xrdb / GDK_SCALE / xrandr DPI
    Fallback → 1.0
    """
    # ── Windows ────────────────────────────────────────────────────────────
    if sys.platform == "win32":
        _ensure_dpi_aware()
        try:
            import ctypes
            # GetDpiForSystem (Windows 10 1607+)
            try:
                dpi = ctypes.windll.user32.GetDpiForSystem()
                if dpi > 0:
                    return dpi / 96.0
            except Exception:
                pass
            # Fallback: GetDeviceCaps(LOGPIXELSX)
            hdc = ctypes.windll.user32.GetDC(0)
            if hdc:
                dpi = ctypes.windll.gdi32.GetDeviceCaps(hdc, 88)  # LOGPIXELSX
                ctypes.windll.user32.ReleaseDC(0, hdc)
                if dpi > 0:
                    return dpi / 96.0
        except Exception:
            pass

    # ── macOS ──────────────────────────────────────────────────────────────
    elif sys.platform == "darwin":
        # Retina displays have a backing scale factor of 2.0.
        # system_profiler reports "Retina" in the resolution line.
        try:
            import subprocess, re as _re
            out = subprocess.check_output(
                ["system_profiler", "SPDisplaysDataType"],
                timeout=5, stderr=subprocess.DEVNULL,
            ).decode("utf-8", errors="replace")
            # "Resolution: 2560 x 1600 Retina" → scale 2.0
            if _re.search(r"Resolution:.*Retina", out, _re.IGNORECASE):
                return 2.0
        except Exception:
            pass

    # ── Linux / X11 / Wayland ──────────────────────────────────────────────
    else:
        # 1. GDK_SCALE (integer scale set by GNOME/GTK)
        try:
            gdk = os.environ.get("GDK_SCALE", "")
            if gdk:
                val = int(gdk)
                if val >= 1:
                    return float(val)
        except (ValueError, TypeError):
            pass

        # 2. Xft.dpi from xrdb (set by most desktop environments)
        try:
            import subprocess, re as _re
            out = subprocess.check_output(
                ["xrdb", "-query"],
                timeout=5, stderr=subprocess.DEVNULL,
            ).decode("utf-8", errors="replace")
            m = _re.search(r"Xft\.dpi:\s*(\d+)", out)
            if m:
                dpi = int(m.group(1))
                if dpi > 0:
                    return dpi / 96.0
        except Exception:
            pass

    return 1.0


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
        _ensure_dpi_aware()
        try:
            import ctypes
            user32 = ctypes.windll.user32
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
            return 0.62
        else:                   # 720p or lower
            return 0.6
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
            # If auto DPI scaling is enabled, use resolution-based default
            if cfg.get("auto_dpi_scale", True):
                return _get_default_scale_for_resolution()
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
    """Enable Qt6 native high-DPI rendering and apply the user's scale factor.

    Must be called *before* importing any PySide6/Qt modules.
    Idempotent — subsequent calls are harmless no-ops.
    """
    global _configured
    if _configured:
        return
    _configured = True

    # ── Enable Qt6 native high-DPI scaling (sharp text rendering) ─────────
    os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"

    # Do NOT set QT_FONT_DPI — let Qt derive font DPI from the device pixel
    # ratio so glyphs are rasterised at the correct resolution (not 96 DPI
    # stretched up, which causes blurriness).
    os.environ.pop("QT_FONT_DPI", None)

    # ── Apply user-configured scale factor ────────────────────────────────
    # The user's factor is the desired *total* scale.  Qt will auto-detect
    # the system DPI (e.g. 1.5× on a 150 % Windows display), so we divide
    # out the system scale to keep the total correct.
    factor = _read_scale_factor()
    system_scale = _get_system_dpi_scale()
    qt_scale = factor / system_scale if system_scale > 0.5 else factor
    qt_scale = max(0.25, min(4.0, qt_scale))
    os.environ["QT_SCALE_FACTOR"] = str(round(qt_scale, 4))

    print(f"✅ DPI scaling configured (target={factor}, system={system_scale:.2f}, QT_SCALE_FACTOR={qt_scale:.4f})")
