"""Haptics with a global off switch (UI_SPEC §6.6).

Wraps Flet's ``HapticFeedback`` service (created by the app in page context
and kept strongly referenced). ``fire(kind)`` never blocks or raises:
``light_impact`` (send, ＋ items, chip toggles, copy), ``selection_click``
(segmented changes, slider ticks), ``medium_impact`` (selection mode, job
start), ``heavy_impact`` (force stop, destructive confirm), ``vibrate``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

__all__ = ["Haptics", "KINDS"]

log = logging.getLogger("glossarion.haptics")

KINDS = ("light_impact", "medium_impact", "heavy_impact", "selection_click", "vibrate")


class Haptics:
    def __init__(self, service: Any = None, *, enabled: bool = True) -> None:
        self.service = service
        self.enabled = enabled
        self.fired: list[str] = []  # last kinds, for diagnostics/tests
        self._tasks: set = set()

    def fire(self, kind: str) -> bool:
        if not self.enabled or self.service is None or kind not in KINDS:
            return False
        method = getattr(self.service, kind, None)
        if method is None:
            return False
        try:
            result = method()
            if asyncio.iscoroutine(result):
                task = asyncio.ensure_future(result)
                self._tasks.add(task)
                task.add_done_callback(self._tasks.discard)
        except Exception as exc:  # no client / unsupported platform
            log.debug("haptic %s failed: %s", kind, exc)
            return False
        self.fired.append(kind)
        del self.fired[:-20]
        return True
