"""Reduce motion (UI_SPEC §6.3, §7.5): one switch, honoured by every animated control.

"Reduce motion" (Settings › Appearance) turns shimmer and rotations into cross-fades and
switches route transitions off (``screens.appearance.apply_appearance``). The flag lives
here so components read it when they animate instead of each screen holding a copy:

* ``switcher_transition(preferred)``: an ``AnimatedSwitcher`` transition (scale / rotation
  become a fade);
* ``shimmer(content, ...)``: an ``ft.Shimmer``, or a static tinted copy of ``content``
  (``Skeleton`` and the ModelSheet's polling header use it);
* ``rotation_animation(ms)``: ``animate_rotation`` for a turning icon (None: the icon swaps
  without turning);
* ``duration(ms)``: a state-change duration, unchanged (cross-fades stay; they are the
  reduced form).

``subscribe(callback)`` lets a mounted control re-apply its motion when the switch flips.
Pure state plus thin Flet helpers; the flag is process-wide (one page per app process).
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import flet as ft

from glossarion_mobile.ui import tokens

__all__ = [
    "duration",
    "reduced",
    "rotation_animation",
    "set_reduce_motion",
    "shimmer",
    "static_placeholder",
    "subscribe",
    "switcher_transition",
]

log = logging.getLogger("glossarion.ui")

_state = {"reduce": False}
_listeners: list[Callable[[bool], Any]] = []


def reduced() -> bool:
    """True while "Reduce motion" is on."""
    return bool(_state["reduce"])


def set_reduce_motion(value: bool) -> bool:
    """Set the switch; listeners run only on a change. Returns the new value."""
    value = bool(value)
    if value == _state["reduce"]:
        return value
    _state["reduce"] = value
    for callback in list(_listeners):
        try:
            callback(value)
        except Exception:
            log.debug("reduce-motion listener failed", exc_info=True)
    return value


def subscribe(callback: Callable[[bool], Any]) -> Callable[[], None]:
    """Call ``callback(reduced)`` whenever the switch flips; returns the unsubscribe function."""
    _listeners.append(callback)

    def unsubscribe() -> None:
        try:
            _listeners.remove(callback)
        except ValueError:
            pass

    return unsubscribe


def duration(ms: int = tokens.MOTION["state_ms"]) -> int:
    """A cross-fade / state-change duration (kept under reduce motion: fades are the reduced form)."""
    return int(ms)


def switcher_transition(preferred: Any = None) -> Any:
    """``AnimatedSwitcher.transition``: ``preferred`` (scale, rotation, …), a fade under reduce motion."""
    if preferred is None or reduced():
        return ft.AnimatedSwitcherTransition.FADE
    return preferred


def rotation_animation(ms: int = tokens.MOTION["sheet_ms"]) -> Optional[int]:
    """``animate_rotation`` for a turning control; None under reduce motion (it swaps instead)."""
    return None if reduced() else int(ms)


def static_placeholder(content: ft.Control, *, color: Any = None, key: Optional[str] = None) -> ft.Control:
    """``content`` dimmed instead of shimmering (the reduce-motion form of ``shimmer``)."""
    return ft.Container(content=content, opacity=0.55, bgcolor=color, key=key)


def shimmer(
    content: ft.Control,
    *,
    base_color: Any = ft.Colors.SURFACE_CONTAINER_HIGHEST,
    highlight_color: Any = ft.Colors.SURFACE_CONTAINER_LOW,
    period: int = 1500,
    key: Optional[str] = None,
) -> ft.Control:
    """A loading shimmer over ``content``; under reduce motion a static tinted copy."""
    if reduced():
        return static_placeholder(content, key=key)
    return ft.Shimmer(content=content, base_color=base_color, highlight_color=highlight_color, period=period, key=key)
