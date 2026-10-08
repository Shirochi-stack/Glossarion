"""No polling while the app is in the background (UI_SPEC §7.3, the U9 performance budget).

Flet 1.0.3 tracks the app lifecycle on the page: ``page.app_visible`` is False while the app
is hidden / paused and ``page.wait_until_visible()`` resolves when it comes back. Screen
loops that poll the disk or repaint (the Glossary editor's auto-reload, the Reader's overlay
refresh, the SDLXLIFF reviewer, the API keys screen's live key stats, the chat's streaming
repaint) sleep with ``poll_sleep``: after each interval they park until the app is visible
again, so a backgrounded app wakes no timers for them (the job itself keeps running on its own
thread; a repaint after resuming shows everything that arrived meanwhile). Pages without the
API (host-test fakes) never park.
"""

from __future__ import annotations

import asyncio
from typing import Any

__all__ = ["app_visible", "park_while_hidden", "poll_sleep"]


def app_visible(page: Any) -> bool:
    """False only when the page says the app is hidden."""
    return getattr(page, "app_visible", True) is not False


async def park_while_hidden(page: Any) -> bool:
    """Wait until the app is visible again; True when it had to wait."""
    if page is None or app_visible(page):
        return False
    waiter = getattr(page, "wait_until_visible", None)
    if waiter is None:
        return False
    result = waiter()
    if asyncio.iscoroutine(result) or isinstance(result, asyncio.Future):
        await result
    return True


async def poll_sleep(page: Any, seconds: float) -> bool:
    """One polling interval, then park while the app is hidden; True when it had to park (a loop
    that skips unchanged ticks refreshes once on resume)."""
    await asyncio.sleep(seconds)
    return await park_while_hidden(page)
