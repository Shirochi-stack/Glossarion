"""``UiDriver``: the Glossarion UI tests' steps over a Flet tester.

The same flows (``flows.py``) run
  * on a device/emulator under ``flet test android`` (``flet_app.tester`` is Flet's
    ``RemoteTester``: it drives the on-device Flutter ``WidgetTester``), and
  * on the host in ``tests_host/test_ui_flows.py`` against the real app on a fake Flet session
    (``host_tester.PyTester`` implements the same calls on the Python control tree),
so every key, tooltip and text a flow uses is checked on every host test run.

Finders follow Flutter: ``key`` (a control's ``key``), ``text`` (exact text of a Text or a
button label), ``contains`` (substring of a text), ``tooltip``. ``wait`` polls with short pumps
(never ``pump_and_settle``: progress rings never settle). Native screens (the Android file
picker) are outside Flutter; a ``picker`` object handles them (``android_device.DocumentsPicker``
on a device, the host's stub picker in host runs).

On a device, Flutter only builds the list rows in the viewport and a tap needs its target to be
hit-testable (``driver_patch`` makes a missed tap an error): ``wait(scroll=True)`` swipes the
list up between polls, and ``tap`` / ``enter`` scroll a step (or pump) and retry while the target
is found but would not receive the pointer yet.
"""

from __future__ import annotations

import asyncio
import os
import re
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

__all__ = ["DEFAULT_TIMEOUT", "RETRYABLE_ACTION_ERRORS", "UiDriver", "UiTimeout", "retryable_action_error"]

DEFAULT_TIMEOUT = float(os.environ.get("GLOSSARION_UI_TIMEOUT", "60"))

#: Errors a device tap / text entry can hit while its target is still settling; ``UiDriver`` scrolls
#: a step (or pumps) and retries them until the action's deadline:
#: * Flutter's hit-test check (``WidgetController.hitTestWarningShouldBeFatal``, set by
#:   ``driver_patch``): the target is built but would not receive the pointer (a list row below the
#:   fold whose card is already on screen, a route mid-transition);
#: * the finder, which Flutter evaluates again at tap time, matched nothing (a rebuild in between).
RETRYABLE_ACTION_ERRORS = (
    "would not receive pointer events",
    "would not hit test",
    "could not find any matching widgets",
)


class UiTimeout(AssertionError):
    """A finder did not match (or did not go away) in time."""


def _describe(**spec: Any) -> str:
    return ", ".join(f"{k}={v!r}" for k, v in spec.items() if v is not None)


def retryable_action_error(exc: BaseException) -> bool:
    text = str(exc)
    return any(marker in text for marker in RETRYABLE_ACTION_ERRORS)


def _first_line(exc: BaseException) -> str:
    text = str(exc).strip()
    return (text.splitlines() or [type(exc).__name__])[0][:200]


class UiDriver:
    def __init__(self, tester: Any, *, artifacts: Optional[os.PathLike] = None, picker: Any = None,
                 back: Optional[Callable[[], Awaitable[Any]]] = None,
                 scroll: Optional[Callable[[], Awaitable[Any]]] = None, poll_ms: int = 400,
                 log: Callable[[str], Any] = print) -> None:
        self.t = tester
        self.artifacts = Path(artifacts) if artifacts else None
        self.picker = picker
        self._back = back
        self._scroll = scroll  # one "swipe up" on the device (Flutter builds list rows lazily)
        self.poll_ms = poll_ms
        self.log = log
        self.steps: list = []
        self._shots = 0

    # ---- primitives -------------------------------------------------------------------------

    async def pump(self, ms: Optional[int] = None) -> None:
        """Pump frames for ``ms`` milliseconds (default ``poll_ms``). Flet's ``DurationValue``: an
        int is milliseconds. (A ``timedelta`` is not JSON-serialisable over the RemoteTester
        socket: it raised after the reply future was registered, so every poll leaked a future and
        pumped with no delay.)"""
        await self.t.pump(int(self.poll_ms if ms is None else ms))

    async def find(self, *, key: Any = None, text: Optional[str] = None, contains: Optional[str] = None,
                   tooltip: Optional[str] = None) -> Any:
        if key is not None:
            return await self.t.find_by_key(key)
        if text is not None:
            return await self.t.find_by_text(text)
        if contains is not None:
            # Flutter's find.textContaining: keep flows to plain words (no regex characters).
            return await self.t.find_by_text_containing(contains)
        if tooltip is not None:
            return await self.t.find_by_tooltip(tooltip)
        raise ValueError("a finder needs key, text, contains or tooltip")

    async def count(self, **spec: Any) -> int:
        return int(getattr(await self.find(**spec), "count", 0))

    async def exists(self, *, timeout: float = 0.0, **spec: Any) -> bool:
        try:
            await self.wait(timeout=timeout, quiet=True, **spec)
            return True
        except UiTimeout:
            return False

    async def wait(self, *, timeout: Optional[float] = None, gone: bool = False, quiet: bool = False,
                   scroll: bool = False, **spec: Any) -> Any:
        """The finder once it matches (``gone``: once it no longer matches). ``scroll``: the target
        may sit below the fold of a list; swipe up between polls (a device only: on the host every
        row is in the tree)."""
        budget = DEFAULT_TIMEOUT if timeout is None else timeout
        deadline = time.monotonic() + budget
        while True:
            finder = await self.find(**spec)
            matched = int(getattr(finder, "count", 0)) > 0
            if matched != gone:
                return finder
            if time.monotonic() >= deadline:
                if not quiet:
                    await self.screenshot("timeout")
                raise UiTimeout(f"{'still found' if gone else 'not found'} after {budget:.0f}s: "
                                f"{_describe(**spec)}")
            if scroll and not gone and self._scroll is not None:
                self.step(f"scroll for {_describe(**spec)}")
                await self._scroll()
            await self.pump()

    async def wait_any(self, *specs: dict, timeout: Optional[float] = None) -> int:
        """Index of the first spec that matches."""
        deadline = time.monotonic() + (DEFAULT_TIMEOUT if timeout is None else timeout)
        while True:
            for index, spec in enumerate(specs):
                if await self.count(**spec):
                    return index
            if time.monotonic() >= deadline:
                await self.screenshot("timeout")
                raise UiTimeout("none found: " + " | ".join(_describe(**s) for s in specs))
            await self.pump()

    async def _act(self, verb: str, action: Callable[[Any], Awaitable[Any]], *, timeout: Optional[float],
                   index: int, scroll: bool, spec: dict) -> None:
        """Find (scrolling if asked), then act. A retryable miss (``RETRYABLE_ACTION_ERRORS``)
        scrolls one step (or just pumps) and tries again until the deadline."""
        budget = DEFAULT_TIMEOUT if timeout is None else timeout
        deadline = time.monotonic() + budget
        while True:
            finder = await self.wait(timeout=max(0.0, deadline - time.monotonic()), scroll=scroll, **spec)
            target = finder.at(index) if index else finder.first
            self.step(f"{verb} {_describe(**spec)}")
            try:
                await action(target)
            except Exception as exc:
                if not retryable_action_error(exc):
                    raise
                if time.monotonic() >= deadline:
                    await self.screenshot("timeout")
                    raise UiTimeout(f"{verb} kept missing for {budget:.0f}s: {_describe(**spec)}: "
                                    f"{_first_line(exc)}") from exc
                scrolling = scroll and self._scroll is not None
                self.step(f"{verb} missed ({_first_line(exc)}); {'scroll' if scrolling else 'pump'} and retry")
                if scrolling:
                    await self._scroll()
                await self.pump()
                continue
            await self.pump(150)
            return

    async def tap(self, *, timeout: Optional[float] = None, index: int = 0, scroll: bool = False,
                  **spec: Any) -> None:
        await self._act("tap", self.t.tap, timeout=timeout, index=index, scroll=scroll, spec=spec)

    async def enter(self, value: str, *, timeout: Optional[float] = None, scroll: bool = False,
                    **spec: Any) -> None:
        async def enter_text(target: Any) -> None:
            await self.t.enter_text(target, value)

        await self._act(f"enter {value!r} into", enter_text, timeout=timeout, index=0, scroll=scroll, spec=spec)

    async def back(self) -> None:
        self.step("back")
        if self._back is not None:
            await self._back()
        elif await self.count(tooltip="Back"):
            await self.tap(tooltip="Back")
        else:
            raise UiTimeout("no way back from this screen")
        await self.pump(300)

    async def pick_file(self, name: str, open_picker: Callable[[], Awaitable[Any]]) -> None:
        """Open a native file picker with ``open_picker`` and choose ``name`` in it."""
        if self.picker is None:
            raise UiTimeout("no file picker driver in this run")
        self.step(f"pick {name!r}")
        await self.picker.arm(name)
        await open_picker()
        await self.picker.choose(name)
        await self.pump(500)

    # ---- diagnostics ------------------------------------------------------------------------

    def step(self, label: str) -> None:
        self.steps.append(label)
        self.log(f"[ui] {label}")

    async def screenshot(self, name: str) -> Optional[Path]:
        if self.artifacts is None:
            return None
        self._shots += 1
        path = self.artifacts / f"{self._shots:02d}_{re.sub(r'[^A-Za-z0-9_.-]+', '_', name)}.png"
        try:
            data = await self.t.take_screenshot(name)
        except Exception as exc:  # screenshots are best effort
            self.log(f"[ui] screenshot {name} failed: {exc}")
            return None
        if data:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            return path
        return None


async def poll(predicate: Callable[[], Any], *, timeout: float, interval: float = 0.5) -> Any:
    """Await a plain predicate (host-side state, adb output) with a deadline."""
    deadline = time.monotonic() + timeout
    while True:
        value = predicate()
        if asyncio.iscoroutine(value):
            value = await value
        if value:
            return value
        if time.monotonic() >= deadline:
            return value
        await asyncio.sleep(interval)
