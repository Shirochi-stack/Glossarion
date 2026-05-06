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
_stylesheet_patch_installed = False
_stylesheet_scale_factor = 1.0

# Default scale factor when config.json has no value
DEFAULT_SCALE_FACTOR = 1.0
DEFAULT_FONT_SCALE = 1.0


def _ensure_dpi_aware():
    """Make the process DPI-aware on Windows (idempotent, no-op on other platforms)."""
    # Let Qt set the Windows DPI awareness context. Calling the native Windows
    # APIs here can make Qt's later SetProcessDpiAwarenessContext call fail
    # with "Access is denied" because awareness can only be set once.
    return


def _mac_get_backing_scale_factor():
    """Return the macOS backing scale factor using CoreGraphics (ctypes).

    Uses CGMainDisplayID + backingScaleFactor via Cocoa/ObjC runtime.
    Falls back to pixel-density heuristic, then system_profiler as last resort.
    This avoids shelling out to system_profiler which can hang or crash on
    Hackintosh / non-Apple hardware (AMD CPUs, non-standard GPU kexts).

    Returns 2.0 for Retina, 1.0 for standard displays.
    """
    # ── Method 1: Cocoa NSScreen.mainScreen.backingScaleFactor via ObjC runtime ──
    try:
        import ctypes
        import ctypes.util
        objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library('objc'))

        # objc_getClass / sel_registerName / objc_msgSend
        objc.objc_getClass.restype = ctypes.c_void_p
        objc.objc_getClass.argtypes = [ctypes.c_char_p]
        objc.sel_registerName.restype = ctypes.c_void_p
        objc.sel_registerName.argtypes = [ctypes.c_char_p]
        objc.objc_msgSend.restype = ctypes.c_void_p
        objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

        NSScreen = objc.objc_getClass(b'NSScreen')
        if NSScreen:
            sel_main = objc.sel_registerName(b'mainScreen')
            main_screen = objc.objc_msgSend(NSScreen, sel_main)
            if main_screen:
                sel_scale = objc.sel_registerName(b'backingScaleFactor')
                # backingScaleFactor returns CGFloat (double on 64-bit)
                objc.objc_msgSend.restype = ctypes.c_double
                scale = objc.objc_msgSend(main_screen, sel_scale)
                # Reset restype for safety
                objc.objc_msgSend.restype = ctypes.c_void_p
                if scale >= 1.0:
                    return float(scale)
    except Exception:
        pass

    # ── Method 2: CoreGraphics pixel density heuristic ──────────────────────
    try:
        import ctypes
        import ctypes.util
        cg_path = ctypes.util.find_library('CoreGraphics')
        if cg_path:
            cg = ctypes.cdll.LoadLibrary(cg_path)
            cg.CGMainDisplayID.restype = ctypes.c_uint32
            display_id = cg.CGMainDisplayID()
            cg.CGDisplayPixelsWide.restype = ctypes.c_size_t
            cg.CGDisplayPixelsWide.argtypes = [ctypes.c_uint32]
            pixel_w = cg.CGDisplayPixelsWide(display_id)
            # CGDisplayScreenSize returns CGSize (two doubles: width, height in mm)
            class CGSize(ctypes.Structure):
                _fields_ = [("width", ctypes.c_double), ("height", ctypes.c_double)]
            cg.CGDisplayScreenSize.restype = CGSize
            cg.CGDisplayScreenSize.argtypes = [ctypes.c_uint32]
            phys = cg.CGDisplayScreenSize(display_id)
            if phys.width > 0:
                dpi = (pixel_w / phys.width) * 25.4  # mm → inches
                if dpi > 170:  # Retina threshold (~220 DPI typical)
                    return 2.0
    except Exception:
        pass

    # ── Method 3: system_profiler as last resort (shorter timeout) ──────────
    try:
        import subprocess, re as _re
        out = subprocess.check_output(
            ["system_profiler", "SPDisplaysDataType"],
            timeout=3, stderr=subprocess.DEVNULL,
        ).decode("utf-8", errors="replace")
        if _re.search(r"Resolution:.*Retina", out, _re.IGNORECASE):
            return 2.0
    except Exception:
        pass

    return 1.0


def _mac_get_screen_resolution():
    """Return (width, height) of the main macOS display using CoreGraphics.

    Uses CGMainDisplayID + CGDisplayPixelsWide/High which are reliable on all
    macOS-compatible hardware including Hackintosh / AMD CPU systems.
    Falls back to system_profiler as last resort.
    """
    # ── Method 1: CoreGraphics (fast, no subprocess, Hackintosh-safe) ──────
    try:
        import ctypes
        import ctypes.util
        cg_path = ctypes.util.find_library('CoreGraphics')
        if cg_path:
            cg = ctypes.cdll.LoadLibrary(cg_path)
            cg.CGMainDisplayID.restype = ctypes.c_uint32
            display_id = cg.CGMainDisplayID()
            cg.CGDisplayPixelsWide.restype = ctypes.c_size_t
            cg.CGDisplayPixelsWide.argtypes = [ctypes.c_uint32]
            cg.CGDisplayPixelsHigh.restype = ctypes.c_size_t
            cg.CGDisplayPixelsHigh.argtypes = [ctypes.c_uint32]
            width = cg.CGDisplayPixelsWide(display_id)
            height = cg.CGDisplayPixelsHigh(display_id)
            if width > 0 and height > 0:
                return (int(width), int(height))
    except Exception:
        pass

    # ── Method 2: system_profiler as last resort (shorter timeout) ──────────
    try:
        import subprocess, re as _re
        out = subprocess.check_output(
            ["system_profiler", "SPDisplaysDataType"],
            timeout=3, stderr=subprocess.DEVNULL,
        ).decode("utf-8", errors="replace")
        m = _re.search(r"Resolution:\s+(\d+)\s*x\s*(\d+)", out)
        if m:
            return (int(m.group(1)), int(m.group(2)))
    except Exception:
        pass

    return (0, 0)


def _win_get_current_display_mode_resolution():
    """Return the primary Windows display mode in physical pixels."""
    try:
        import ctypes
        from ctypes import wintypes

        CCHDEVICENAME = 32
        CCHFORMNAME = 32
        ENUM_CURRENT_SETTINGS = -1

        class DEVMODEW(ctypes.Structure):
            _fields_ = [
                ("dmDeviceName", wintypes.WCHAR * CCHDEVICENAME),
                ("dmSpecVersion", wintypes.WORD),
                ("dmDriverVersion", wintypes.WORD),
                ("dmSize", wintypes.WORD),
                ("dmDriverExtra", wintypes.WORD),
                ("dmFields", wintypes.DWORD),
                ("dmOrientation", wintypes.SHORT),
                ("dmPaperSize", wintypes.SHORT),
                ("dmPaperLength", wintypes.SHORT),
                ("dmPaperWidth", wintypes.SHORT),
                ("dmScale", wintypes.SHORT),
                ("dmCopies", wintypes.SHORT),
                ("dmDefaultSource", wintypes.SHORT),
                ("dmPrintQuality", wintypes.SHORT),
                ("dmColor", wintypes.SHORT),
                ("dmDuplex", wintypes.SHORT),
                ("dmYResolution", wintypes.SHORT),
                ("dmTTOption", wintypes.SHORT),
                ("dmCollate", wintypes.SHORT),
                ("dmFormName", wintypes.WCHAR * CCHFORMNAME),
                ("dmLogPixels", wintypes.WORD),
                ("dmBitsPerPel", wintypes.DWORD),
                ("dmPelsWidth", wintypes.DWORD),
                ("dmPelsHeight", wintypes.DWORD),
                ("dmDisplayFlags", wintypes.DWORD),
                ("dmDisplayFrequency", wintypes.DWORD),
                ("dmICMMethod", wintypes.DWORD),
                ("dmICMIntent", wintypes.DWORD),
                ("dmMediaType", wintypes.DWORD),
                ("dmDitherType", wintypes.DWORD),
                ("dmReserved1", wintypes.DWORD),
                ("dmReserved2", wintypes.DWORD),
                ("dmPanningWidth", wintypes.DWORD),
                ("dmPanningHeight", wintypes.DWORD),
            ]

        user32 = ctypes.windll.user32
        devmode = DEVMODEW()
        devmode.dmSize = ctypes.sizeof(DEVMODEW)
        try:
            user32.EnumDisplaySettingsW.argtypes = [
                wintypes.LPCWSTR,
                wintypes.DWORD,
                ctypes.POINTER(DEVMODEW),
            ]
            user32.EnumDisplaySettingsW.restype = wintypes.BOOL
        except Exception:
            pass

        if user32.EnumDisplaySettingsW(None, ENUM_CURRENT_SETTINGS, ctypes.byref(devmode)):
            width = int(devmode.dmPelsWidth)
            height = int(devmode.dmPelsHeight)
            if width > 0 and height > 0:
                return (width, height)
    except Exception:
        pass
    return (0, 0)


def _get_system_dpi_scale():
    """Return the OS-level DPI scale factor (e.g. 1.0, 1.25, 1.5, 2.0).

    Windows  → GetDpiForSystem / GetDeviceCaps
    macOS    → CoreGraphics backing scale (Hackintosh-safe, no system_profiler)
    Linux    → Xrdb / GDK_SCALE / xrandr DPI
    Fallback → 1.0
    """
    # ── Windows ────────────────────────────────────────────────────────────
    if sys.platform == "win32":
        _ensure_dpi_aware()
        try:
            import ctypes

            # Before Qt creates the application, this process may still be DPI
            # unaware. In that state GetDpiForSystem can report 96 DPI even when
            # Windows is scaling the desktop. Compare the real display mode with
            # the DPI-virtualized screen size first so QT_SCALE_FACTOR is divided
            # by the scale Qt will later apply natively.
            try:
                user32 = ctypes.windll.user32
                physical_w, physical_h = _win_get_current_display_mode_resolution()
                logical_w = int(user32.GetSystemMetrics(0))
                logical_h = int(user32.GetSystemMetrics(1))
                if physical_w > 0 and physical_h > 0 and logical_w > 0 and logical_h > 0:
                    scale_w = physical_w / logical_w
                    scale_h = physical_h / logical_h
                    scale = max(scale_w, scale_h)
                    if 1.01 <= scale <= 4.0 and abs(scale_w - scale_h) < 0.05:
                        return scale
            except Exception:
                pass

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
        return _mac_get_backing_scale_factor()

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
    macOS    → CoreGraphics (Hackintosh-safe, no system_profiler)
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
            width, height = _win_get_current_display_mode_resolution()

            if width <= 0 or height <= 0:
                width = user32.GetSystemMetrics(0)
                height = user32.GetSystemMetrics(1)
        except Exception:
            pass

    # ── macOS ──────────────────────────────────────────────────────────────
    elif sys.platform == "darwin":
        width, height = _mac_get_screen_resolution()
        # CGDisplayPixelsWide/High returns logical points on Retina;
        # multiply by backing scale factor to get physical pixel count
        # so the auto-scaler correctly recognises high-DPI displays.
        backing = _mac_get_backing_scale_factor()
        width = int(width * backing)
        height = int(height * backing)

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

        if 1600 <= width < 1920:
            return 0.85
        if 1366 <= width < 1600:
            return 0.7

        # Choose scale factor based on horizontal resolution
        if width >= 3840:       # 4K (3840×2160)
            return 1.7
        elif width >= 2560:     # 1440p / QHD (2560×1440)
            return 1.15
        elif width >= 1920:     # 1080p (1920×1080)
            return 1.0
        elif width >= 900:     # 900p (1600×900)
            return 0.85            
        elif width >= 1366:     # 768p / common laptops
            return 0.7
        else:                   # 720p or lower
            return 0.62
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


def _read_font_scale():
    """Read gui_font_scale from config.json (stdlib only; no auto behavior)."""
    try:
        if getattr(sys, 'frozen', False) and hasattr(sys, 'executable'):
            base_dir = os.path.dirname(os.path.abspath(sys.executable))
        else:
            base_dir = os.path.dirname(os.path.abspath(__file__))
        cfg_path = os.path.join(base_dir, "config.json")
        if os.path.isfile(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            factor = float(cfg.get("gui_font_scale", DEFAULT_FONT_SCALE))
            if factor < 0.5:
                factor = 0.5
            elif factor > 1.5:
                factor = 1.5
            return factor
    except Exception:
        pass
    return DEFAULT_FONT_SCALE


def _format_scaled_font_size(value, unit, factor):
    scaled = max(1.0, float(value) * factor)
    if unit.lower() == "px":
        scaled = round(scaled)
        return str(int(scaled))
    text = f"{scaled:.2f}".rstrip("0").rstrip(".")
    return text or "1"


def _scale_stylesheet_font_sizes(style, factor):
    if not style or factor == 1.0:
        return style
    try:
        import re

        def repl(match):
            prefix, value, unit = match.groups()
            return f"{prefix}{_format_scaled_font_size(value, unit, factor)}{unit}"

        return re.sub(r"(font-size\s*:\s*)([0-9]+(?:\.[0-9]+)?)(pt|px)", repl, style, flags=re.IGNORECASE)
    except Exception:
        return style


def _install_stylesheet_font_scale_patch(factor):
    """Scale future QWidget stylesheets that hardcode font-size values."""
    global _stylesheet_patch_installed, _stylesheet_scale_factor
    _stylesheet_scale_factor = factor
    if _stylesheet_patch_installed:
        return
    try:
        from PySide6.QtWidgets import QWidget

        original_set_stylesheet = QWidget.setStyleSheet

        def scaled_set_stylesheet(widget, style):
            try:
                base_style = style or ""
                widget.setProperty("_glossarion_base_stylesheet", base_style)
                style = _scale_stylesheet_font_sizes(base_style, _stylesheet_scale_factor)
            except Exception:
                pass
            return original_set_stylesheet(widget, style)

        QWidget._glossarion_original_setStyleSheet = original_set_stylesheet
        QWidget.setStyleSheet = scaled_set_stylesheet
        _stylesheet_patch_installed = True
    except Exception:
        pass


def _rescale_existing_widget_styles(app, factor):
    """Reapply stylesheets already set before the font-scale patch was installed."""
    try:
        from PySide6.QtWidgets import QWidget

        original_set_stylesheet = getattr(QWidget, "_glossarion_original_setStyleSheet", QWidget.setStyleSheet)
        for widget in app.allWidgets():
            try:
                base_style = widget.property("_glossarion_base_stylesheet")
                if base_style is None:
                    base_style = widget.styleSheet()
                    if base_style:
                        widget.setProperty("_glossarion_base_stylesheet", base_style)
                if base_style:
                    original_set_stylesheet(widget, _scale_stylesheet_font_sizes(base_style, factor))
            except Exception:
                pass
    except Exception:
        pass


def apply_font_scale(app=None):
    """Apply the configured GUI font scale after QApplication exists."""
    try:
        from PySide6.QtWidgets import QApplication

        if app is None:
            app = QApplication.instance()
        if app is None:
            return DEFAULT_FONT_SCALE

        factor = _read_font_scale()
        font = app.font()
        base_size = getattr(app, "_glossarion_base_font_point_size", None)
        if not base_size:
            base_size = font.pointSizeF()
            if base_size <= 0:
                base_size = float(font.pointSize() or 9)
            setattr(app, "_glossarion_base_font_point_size", base_size)

        font.setPointSizeF(max(6.0, base_size * factor))
        app.setFont(font)
        _install_stylesheet_font_scale_patch(factor)
        _rescale_existing_widget_styles(app, factor)
        return factor
    except Exception:
        return DEFAULT_FONT_SCALE


def _qgui_application_exists():
    """Best-effort check without importing PySide unless it is already loaded."""
    try:
        if "PySide6.QtGui" not in sys.modules and "PySide6.QtWidgets" not in sys.modules:
            return False
        from PySide6.QtGui import QGuiApplication
        return QGuiApplication.instance() is not None
    except Exception:
        return False


def configure():
    """Enable Qt6 native high-DPI rendering and apply the user's scale factor.

    Must be called *before* importing any PySide6/Qt modules.
    Idempotent — subsequent calls are harmless no-ops.
    """
    global _configured
    if _configured:
        return
    _configured = True

    if _qgui_application_exists():
        # Qt's scale-factor policy and scale env vars are pre-QGuiApplication
        # only. If another entry point already created the app, leave it alone.
        return

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
    if sys.platform == "darwin":
        # Qt 6 handles Retina 2.0x scaling natively via
        # QT_ENABLE_HIGHDPI_SCALING; dividing by the backing factor
        # would double-downscale the UI, making it unreadably small.
        qt_scale = factor
    else:
        qt_scale = factor / system_scale if system_scale > 0.5 else factor
    qt_scale = max(0.25, min(4.0, qt_scale))
    os.environ["QT_SCALE_FACTOR"] = str(round(qt_scale, 4))

    print(f"✅ DPI scaling configured (target={factor}, system={system_scale:.2f}, QT_SCALE_FACTOR={qt_scale:.4f})")
