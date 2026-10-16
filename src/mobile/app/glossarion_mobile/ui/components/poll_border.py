"""PollBorder: the animated perimeter the desktop draws around the model field while an online model
catalog is polled (``translator_gui._ModelCatalogPollBorder``), for Flet.

The wrapped field sits on an opaque inner surface inside a ``BORDER`` dp padding; while ``active`` the
outer container paints the desktop's cyan / violet sweep gradient in that padding and turns it: every
``TICK_S`` the sweep's start / end angles advance (the container's implicit animation tweens between
ticks, so the client turns it smoothly). Under reduce motion (UI_SPEC §6.3) the border stays still.
Idle, the padding is transparent, so the layout never jumps when a poll starts or ends.

``start(spawn)`` / ``stop()`` are idempotent; ``stop()`` cancels the turning task, and the task ends by
itself once the border is stopped or no longer on a page, so the indicator can never keep spinning.
"""

from __future__ import annotations

import asyncio
import logging
import math
from typing import Any, Callable, Optional

import flet as ft

from glossarion_mobile.ui import motion

__all__ = ["PollBorder"]

log = logging.getLogger("glossarion.ui")

#: The desktop stops and colours (QConicalGradient in _ModelCatalogPollBorder.paintEvent): (stop, colour, alpha)
_DESKTOP_STOPS = (
    (0.00, "#67E8F9", 0.92),
    (0.16, "#A78BFA", 0.90),
    (0.38, "#3B82F6", 0.37),
    (0.70, "#1E293B", 0.22),
    (0.88, "#22D3EE", 0.82),
    (1.00, "#67E8F9", 0.92),
)


class PollBorder:
    """Wraps ``content`` (a search field); ``control`` is what goes into the layout."""

    BORDER = 2
    TICK_S = 0.1
    TURN_S = 2.0  # one full turn (desktop: 5 degrees per 40 ms, about 2.9 s)
    MAX_UNMOUNTED_TICKS = 50  # a border started before its sheet is shown waits this long to be mounted

    def __init__(self, content: ft.Control, *, radius: float = 8, surface: Any = ft.Colors.SURFACE_CONTAINER_HIGH,
                 key: Optional[str] = None) -> None:
        self.content = content
        self.inner = ft.Container(content=content, bgcolor=surface, border_radius=radius)
        self.control = ft.Container(content=self.inner, padding=self.BORDER, border_radius=radius + self.BORDER,
                                    key=key)
        self.angle = 0.0
        self.active = False
        self.ticks = 0
        self._task: Any = None

    # ---- drawing ---------------------------------------------------------------------------------------

    def gradient(self) -> ft.SweepGradient:
        start = self.angle
        return ft.SweepGradient(
            colors=[ft.Colors.with_opacity(alpha, color) for _stop, color, alpha in _DESKTOP_STOPS],
            stops=[stop for stop, _color, _alpha in _DESKTOP_STOPS],
            start_angle=start,
            end_angle=start + 2 * math.pi,
            tile_mode=ft.GradientTileMode.REPEATED,
        )

    def _push(self) -> bool:
        try:
            self.control.update()
            return True
        except Exception:  # not on a page (yet / any more)
            return False

    # ---- state -----------------------------------------------------------------------------------------

    @property
    def spinning(self) -> bool:
        task = self._task
        return task is not None and not task.done()

    def start(self, spawn: Optional[Callable[[Any], Any]] = None) -> None:
        if self.active:
            return
        self.active = True
        self.control.gradient = self.gradient()
        reduce = motion.reduced()
        self.control.animate = None if reduce else ft.Animation(int(self.TICK_S * 1000), ft.AnimationCurve.LINEAR)
        self._push()
        if reduce:
            return
        coro = self._spin()
        try:
            self._task = spawn(coro) if spawn is not None else asyncio.ensure_future(coro)
        except Exception:  # no running loop: a still border is still an indicator
            coro.close()
            self._task = None
            log.debug("poll border cannot turn without a loop", exc_info=True)

    def stop(self) -> None:
        task, self._task = self._task, None
        if task is not None and not task.done():
            task.cancel()
        if not self.active:
            return
        self.active = False
        self.control.gradient = None
        self.control.animate = None
        self._push()

    async def _spin(self) -> None:
        step = 2 * math.pi * self.TICK_S / self.TURN_S
        mounted, misses, gone = False, 0, False
        try:
            while self.active:
                await asyncio.sleep(self.TICK_S)
                if not self.active:
                    break
                self.angle += step
                self.control.gradient = self.gradient()
                self.ticks += 1
                if self._push():
                    mounted, misses = True, 0
                    continue
                misses += 1
                # gone after being shown (the sheet closed), or never shown at all: nothing to turn
                if mounted or misses >= self.MAX_UNMOUNTED_TICKS:
                    gone = True
                    break
        except asyncio.CancelledError:
            pass
        finally:
            if gone:
                # not "active" any more, so a later start() (the sheet shown again) turns it afresh
                self.active = False
                self.control.gradient = None
                self.control.animate = None
