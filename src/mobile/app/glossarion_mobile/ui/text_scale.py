"""Effective text scale (UI_SPEC §6.2, §7.5): Appearance text size × the system font scale.

Flutter scales every ``Text`` by the system font size (Android up to 200 %, iOS Dynamic Type)
on top of the app's own Appearance scale (85–130 %, baked into the theme's text styles).
Flet 1.0.3 does not report the system scale to Python, so the layout rules that switch at
>= 160 % (composer pills → "Options (n)", the mode chip, the header's model-only subtitle,
wrapped stat rows, scrollable tab bars, hidden bottom-bar labels) only saw the app scale and
never engaged at 200 % system text.

``TextScaleProbe`` measures it: an invisible ``Text`` of ``PROBE_SP`` sp with line height 1.0
reports its laid-out height through ``on_size_change``; height / ``PROBE_SP`` is the system
scale (Android 14+ scales large text non-linearly, so the probe size is close to body text).
``effective(state)`` is what layouts compare with ``tokens.COMPACT_TEXT_SCALE``. Without a
measurement (host tests, a client that never reports sizes) the system scale is 1.0 and the
layout is exactly what it was.

Pure Python at import time (Flet is imported by ``TextScaleProbe`` only), so feature modules that
stay Flet-free until they build a screen can use ``effective``.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

__all__ = [
    "PROBE_SP",
    "TextScaleProbe",
    "effective",
    "os_scale",
    "scale_from_height",
    "set_os_scale",
    "subscribe",
]

log = logging.getLogger("glossarion.ui")

PROBE_SP = 16.0
_RANGE = (0.5, 4.0)
_STEP = 0.05  # changes smaller than this are measurement noise
_state = {"os": 1.0}
_listeners: list[Callable[[float], Any]] = []


def os_scale() -> float:
    """The last measured system font scale (1.0 until measured)."""
    return float(_state["os"])


def scale_from_height(height: Any, probe_sp: float = PROBE_SP) -> Optional[float]:
    """System scale for a probe line ``height`` (None when unusable)."""
    try:
        value = float(height)
    except (TypeError, ValueError):
        return None
    if value <= 0 or probe_sp <= 0:
        return None
    return round(min(_RANGE[1], max(_RANGE[0], value / probe_sp)), 2)


def set_os_scale(value: Any) -> bool:
    """Record a measurement; listeners run when it moved by at least ``_STEP``. True on a change."""
    try:
        scale = round(min(_RANGE[1], max(_RANGE[0], float(value))), 2)
    except (TypeError, ValueError):
        return False
    if abs(scale - _state["os"]) < _STEP:
        return False
    _state["os"] = scale
    for callback in list(_listeners):
        try:
            callback(scale)
        except Exception:
            log.debug("text scale listener failed", exc_info=True)
    return True


def subscribe(callback: Callable[[float], Any]) -> Callable[[], None]:
    _listeners.append(callback)

    def unsubscribe() -> None:
        try:
            _listeners.remove(callback)
        except ValueError:
            pass

    return unsubscribe


def effective(state: Any = None, app_scale: Optional[float] = None) -> float:
    """App text scale (``AppState.text_scale`` or ``app_scale``) × the system scale."""
    if app_scale is None:
        signal = getattr(state, "text_scale", None)
        app_scale = getattr(signal, "value", None) if signal is not None else None
    try:
        app = float(app_scale) if app_scale is not None else 1.0
    except (TypeError, ValueError):
        app = 1.0
    return round(app * os_scale(), 3)


class TextScaleProbe:
    """An invisible one-line ``Text`` in ``page.overlay`` whose height reports the system scale."""

    def __init__(self, on_change: Optional[Callable[[float], Any]] = None) -> None:
        import flet as ft

        self.on_change = on_change
        self.measured: Optional[float] = None
        self.text = ft.Text(
            "Mg",
            size=PROBE_SP,
            style=ft.TextStyle(height=1.0),
            no_wrap=True,
            on_size_change=self._on_size,
            key="text-scale-probe-text",
        )
        self.control = ft.Container(
            content=self.text,
            opacity=0.0,
            ignore_interactions=True,
            left=0,
            top=0,
            key="text-scale-probe",
        )

    def _on_size(self, e: Any) -> None:
        scale = scale_from_height(getattr(e, "height", None))
        if scale is None:
            return
        self.measured = scale
        if set_os_scale(scale) and self.on_change is not None:
            self.on_change(scale)

    def install(self, page: Any) -> bool:
        """Add the probe to ``page.overlay`` once; True when added."""
        overlay = getattr(page, "overlay", None)
        if overlay is None or any(item is self.control for item in overlay):
            return False
        try:
            overlay.append(self.control)
        except Exception:
            log.debug("installing the text scale probe failed", exc_info=True)
            return False
        return True
