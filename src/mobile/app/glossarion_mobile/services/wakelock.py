"""SharedWakelock: one reference-counted owner of the platform wakelock.

Flet's ``Wakelock`` service toggles a single global screen wakelock. Two features
want it independently: the job service ("Keep screen on during jobs") and the Reader
("Keep screen on"). Each used to drive its own ``Wakelock`` service, so whichever
released last-but-one turned the screen lock off under the other (closing the Reader
during a job, or a job ending while the Reader is open).

``SharedWakelock(wakelock)`` wraps the one platform service; each feature gets a
``holder(name)`` with the same async ``enable()`` / ``disable()`` interface the
wakelock itself has (``BackgroundExecution`` takes it unchanged). The platform lock is
on while at least one holder has enabled it. Runs on the UI loop; pure asyncio.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

__all__ = ["SharedWakelock", "WakelockHolder"]

log = logging.getLogger("glossarion.wakelock")


class SharedWakelock:
    def __init__(self, wakelock: Any) -> None:
        self.wakelock = wakelock
        self.holders: set = set()
        self.on = False
        self.calls: list = []  # what reached the platform service (diagnostics / tests)
        self._lock: Optional[asyncio.Lock] = None
        self._loop: Any = None

    def holder(self, name: str) -> "WakelockHolder":
        return WakelockHolder(self, str(name))

    def _guard(self) -> asyncio.Lock:
        loop = asyncio.get_running_loop()
        if self._lock is None or self._loop is not loop:
            self._lock = asyncio.Lock()
            self._loop = loop
        return self._lock

    async def acquire(self, name: str) -> None:
        async with self._guard():
            if not self.on and self.wakelock is not None:
                # A failed platform call raises before the holder is counted: a caller that
                # sees the failure (BackgroundExecution) never releases, so nothing may pin it.
                await self.wakelock.enable()
                self.on = True
                self.calls.append(("enable", name))
            self.holders.add(name)

    async def release(self, name: str) -> None:
        async with self._guard():
            # Uncounted first, so a failed platform call cannot pin the count either.
            self.holders.discard(name)
            if self.on and not self.holders and self.wakelock is not None:
                await self.wakelock.disable()
                self.on = False
                self.calls.append(("disable", name))


class WakelockHolder:
    """One feature's handle on the shared wakelock (``enable`` / ``disable`` like ``ft.Wakelock``)."""

    def __init__(self, owner: SharedWakelock, name: str) -> None:
        self.owner = owner
        self.name = name

    @property
    def held(self) -> bool:
        return self.name in self.owner.holders

    async def enable(self) -> None:
        await self.owner.acquire(self.name)

    async def disable(self) -> None:
        await self.owner.release(self.name)
