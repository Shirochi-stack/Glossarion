# dpi_setup.py — Windows DPI-scaling bootstrap for PySide6
#
# Call  dpi_setup.configure()  ONCE, **before** importing any PySide6 modules.
# Safe to call multiple times (idempotent) and on non-Windows platforms (no-op).
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

_configured = False

# Default scale factor when config.json has no value
DEFAULT_SCALE_FACTOR = 1.75


def _read_scale_factor():
    """Read gui_scale_factor from config.json (stdlib only, no PySide6)."""
    try:
        cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
        if os.path.isfile(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            val = cfg.get("gui_scale_factor", DEFAULT_SCALE_FACTOR)
            factor = float(val)
            # Clamp to sane range
            if factor < 0.5:
                factor = 0.5
            elif factor > 3.0:
                factor = 3.0
            return factor
    except Exception:
        pass
    return DEFAULT_SCALE_FACTOR


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
