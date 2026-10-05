"""Calling user callbacks that may be plain functions or coroutine functions."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Optional

log = logging.getLogger("glossarion.ui")


def call_handler(handler: Optional[Callable[..., Any]], *args: Any) -> Optional[asyncio.Future]:
    """Run ``handler(*args)`` on the loop thread; a returned coroutine becomes a task."""
    if handler is None:
        return None
    try:
        result = handler(*args)
    except Exception:
        log.exception("UI handler %s failed", getattr(handler, "__qualname__", handler))
        return None
    if asyncio.iscoroutine(result):
        task = asyncio.ensure_future(result)
        _TASKS.add(task)
        task.add_done_callback(_done)
        return task
    return None


async def await_handler(handler: Optional[Callable[..., Any]], *args: Any) -> Any:
    """Run ``handler(*args)`` and await it when it returns a coroutine."""
    if handler is None:
        return None
    result = handler(*args)
    if asyncio.iscoroutine(result):
        return await result
    return result


_TASKS: set = set()


def _done(task: asyncio.Future) -> None:
    _TASKS.discard(task)
    if not task.cancelled() and task.exception() is not None:
        log.error("UI handler task failed", exc_info=task.exception())
