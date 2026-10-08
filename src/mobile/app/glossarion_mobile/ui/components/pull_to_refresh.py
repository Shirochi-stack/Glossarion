"""PullToRefresh (UI_SPEC §5.1, §3.12): pulling a list down at its top refreshes it.

Flet 1.0.3 has no RefreshIndicator; a ``ListView`` / ``GridView`` reports the pull as an
``on_scroll`` event of type ``overscroll`` with a negative ``overscroll`` while it sits at
its top (the Library's U5 pull rule, now shared). ``handle(e)`` is the list's ``on_scroll``
filter: it returns True for overscroll events (the caller ignores them) and starts one refresh
per pull - a pull fires several overscroll notifications, so a refresh already running absorbs
the rest. While it runs a thin indeterminate ``bar`` shows above the list; the pull threshold
gives a ``selection_click`` haptic (UI_SPEC §6.6). ⋯ Refresh stays the fallback everywhere.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Optional

import flet as ft

__all__ = ["PullToRefresh", "is_overscroll", "is_top_pull"]

log = logging.getLogger("glossarion.ui")


def _event_type(e: Any) -> str:
    kind = getattr(e, "event_type", "")
    return str(getattr(kind, "value", kind) or "")


def is_overscroll(e: Any) -> bool:
    return _event_type(e) == "overscroll"


def is_top_pull(e: Any) -> bool:
    """An overscroll past the top edge of the list (a pull down)."""
    if not is_overscroll(e):
        return False
    overscroll = getattr(e, "overscroll", 0) or 0
    pixels = getattr(e, "pixels", 0) or 0
    minimum = getattr(e, "min_scroll_extent", 0) or 0
    return overscroll < 0 and pixels <= minimum + 1


class PullToRefresh:
    def __init__(
        self,
        on_refresh: Callable[[], Any],
        *,
        spawn: Optional[Callable[[Any], Any]] = None,
        haptic: Optional[Callable[[str], Any]] = None,
        key: Optional[str] = None,
    ) -> None:
        self.on_refresh = on_refresh
        self._spawn = spawn
        self.haptic = haptic
        self.refreshing = False
        self.pulls = 0
        self.bar = ft.ProgressBar(value=None, bar_height=3, visible=False, key=key)

    def handle(self, e: Any) -> bool:
        """The list's scroll event: True when it was an overscroll (consumed; the caller returns)."""
        if not is_overscroll(e):
            return False
        if is_top_pull(e) and not self.refreshing:
            self.trigger()
        return True

    def trigger(self) -> Any:
        """Start one refresh (no-op while one runs)."""
        if self.refreshing:
            return None
        self.refreshing = True
        self.pulls += 1
        if self.haptic is not None:
            try:
                self.haptic("selection_click")
            except Exception:
                pass
        coro = self.run()
        if self._spawn is not None:
            return self._spawn(coro)
        try:
            return asyncio.ensure_future(coro)
        except RuntimeError:
            coro.close()
            self.refreshing = False
            return None

    async def run(self) -> None:
        self._show(True)
        try:
            result = self.on_refresh()
            if asyncio.iscoroutine(result):
                await result
        except Exception:
            log.exception("pull-to-refresh failed")
        finally:
            self.refreshing = False
            self._show(False)

    def _show(self, visible: bool) -> None:
        self.bar.visible = visible
        try:
            self.bar.update()
        except Exception:  # not mounted (tests)
            pass
