# dpi_setup.py — Windows DPI-scaling bootstrap for PySide6
#
# Call  dpi_setup.configure()  ONCE, **before** importing any PySide6 modules.
# Safe to call multiple times (idempotent) and on non-Windows platforms (no-op).
#
# What this does:
#   1. Sets QT_ENABLE_HIGHDPI_SCALING=0  → disables Qt6 automatic DPI scaling
#   2. Sets QT_FONT_DPI=96              → forces logical 96 DPI (1:1 rendering)
#   3. Sets QT_SCALE_FACTOR=1           → explicit 1× scale factor
#
# The net effect is that the GUI renders at a consistent physical-pixel size
# regardless of the Windows display scaling percentage (100%, 125%, 150%, etc).

import os
import sys

_configured = False


def configure():
    """Disable Qt6 automatic DPI scaling so the GUI size is independent of
    the Windows display-scaling percentage.

    Must be called *before* importing any PySide6/Qt modules.
    Idempotent — subsequent calls are harmless no-ops.
    """
    global _configured
    if _configured:
        return
    _configured = True

    # ── Disable Qt's built-in high-DPI scaling ────────────────────────────
    # Qt6 enables high-DPI scaling by default. Setting this to "0" tells Qt
    # not to apply any scale factor to widgets or painter coordinates.
    # A 1-pixel line stays 1 physical pixel regardless of monitor DPI.
    os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "0"

    # ── Lock font DPI to the traditional 96 ──────────────────────────────
    # Without this, fonts may still scale up on high-DPI screens even when
    # widget scaling is disabled.  96 is the baseline Windows DPI.
    os.environ["QT_FONT_DPI"] = "96"

    # ── Explicit 1.25× scale factor ────────────────────────────────────────
    # With auto-scaling disabled, 1.0 renders too compact on most screens.
    # 1.25 provides a comfortable default size across resolutions.
    os.environ["QT_SCALE_FACTOR"] = "1.67"

    print("✅ DPI scaling disabled (QT_ENABLE_HIGHDPI_SCALING=0, QT_FONT_DPI=96, QT_SCALE_FACTOR=1)")
