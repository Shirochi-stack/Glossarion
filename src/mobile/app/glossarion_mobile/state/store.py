"""Small reactive store for the Flet UI: ``Signal``, ``Computed`` and the loop guard.

Threading rule (UI_SPEC §5.0, plan §1): Flet controls and UI state are owned by
the asyncio loop thread. Worker threads never call ``Signal.set`` or
``control.update()``; they hand values to ``UiDispatcher`` (``post`` or a
latest-wins ``Channel``), which applies them on the loop.

``LoopGuard`` remembers which thread runs the UI loop once ``bind()`` has been
called there; after that, ``Signal.set`` from any other thread raises
``WrongThreadError`` instead of silently racing the Flet diff engine. Before
binding (module import time, pure unit tests) nothing is enforced.

Pure Python, Python 3.10 compatible; never imports Flet or backend modules.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any, Callable, Generic, Iterable, Optional, TypeVar

__all__ = [
    "Computed",
    "LoopGuard",
    "Signal",
    "UI_LOOP",
    "WrongThreadError",
    "assert_loop_thread",
    "bind_ui_loop",
]

log = logging.getLogger("glossarion.state")

T = TypeVar("T")


class WrongThreadError(RuntimeError):
    """UI state was touched from a thread other than the UI loop thread."""


class LoopGuard:
    """Remembers the UI loop and its thread; ``check()`` asserts callers run there."""

    def __init__(self, *, strict: bool = True) -> None:
        self.strict = strict
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread_id: Optional[int] = None
        self._thread_name = ""

    def bind(self, loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
        """Bind to ``loop`` (default: the running loop). Call it on the loop thread."""
        if loop is None:
            loop = asyncio.get_running_loop()
        self._loop = loop
        self._thread_id = threading.get_ident()
        self._thread_name = threading.current_thread().name

    def unbind(self) -> None:
        self._loop = None
        self._thread_id = None
        self._thread_name = ""

    @property
    def bound(self) -> bool:
        return self._thread_id is not None

    @property
    def loop(self) -> Optional[asyncio.AbstractEventLoop]:
        return self._loop

    def on_loop_thread(self) -> bool:
        """True on the bound loop thread, or when nothing is bound yet."""
        return self._thread_id is None or threading.get_ident() == self._thread_id

    def check(self, what: str = "UI state") -> None:
        if self.strict and not self.on_loop_thread():
            raise WrongThreadError(
                f"{what} must run on the UI loop thread ({self._thread_name or '?'}), "
                f"not on {threading.current_thread().name!r}; post it through UiDispatcher"
            )


# Process-wide guard bound by the app in ``main(page)``.
UI_LOOP = LoopGuard()


def bind_ui_loop(loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
    UI_LOOP.bind(loop)


def assert_loop_thread(what: str = "UI state") -> None:
    UI_LOOP.check(what)


Subscriber = Callable[[Any], Any]


class Signal(Generic[T]):
    """An observable value owned by the UI loop.

    ``set()`` notifies subscribers synchronously (on the loop thread) when the
    value changes (``equals`` decides; default ``==``). Subscriber exceptions
    are logged and never stop the other subscribers.
    """

    __slots__ = ("_value", "_name", "_guard", "_equals", "_subscribers", "_version")

    def __init__(
        self,
        value: T,
        *,
        name: str = "",
        guard: Optional[LoopGuard] = None,
        equals: Optional[Callable[[Any, Any], bool]] = None,
    ) -> None:
        self._value = value
        self._name = name
        self._guard = guard if guard is not None else UI_LOOP
        self._equals = equals
        self._subscribers: list[Subscriber] = []
        self._version = 0

    def __repr__(self) -> str:
        return f"Signal({self._name or '?'}={self._value!r})"

    @property
    def name(self) -> str:
        return self._name

    @property
    def value(self) -> T:
        return self._value

    @value.setter
    def value(self, new: T) -> None:
        self.set(new)

    @property
    def version(self) -> int:
        """Incremented on every change (cheap "did it change since" checks)."""
        return self._version

    def get(self) -> T:
        return self._value

    def _same(self, old: Any, new: Any) -> bool:
        if self._equals is not None:
            return bool(self._equals(old, new))
        if old is new:
            return True
        try:
            return bool(old == new)
        except Exception:  # exotic __eq__
            return False

    def set(self, new: T, *, force: bool = False) -> bool:
        """Store ``new``; returns True when subscribers were notified."""
        self._guard.check(f"Signal({self._name or '?'}).set")
        if not force and self._same(self._value, new):
            return False
        self._value = new
        self._version += 1
        self._notify(new)
        return True

    def update(self, fn: Callable[[T], T]) -> bool:
        return self.set(fn(self._value))

    def _notify(self, value: Any) -> None:
        for callback in list(self._subscribers):
            try:
                callback(value)
            except Exception:
                log.exception("subscriber of %r failed", self)

    def subscribe(self, callback: Subscriber, *, immediate: bool = False) -> Callable[[], None]:
        """Register ``callback(value)``; returns an idempotent unsubscribe function."""
        self._subscribers.append(callback)
        if immediate:
            callback(self._value)

        def unsubscribe() -> None:
            try:
                self._subscribers.remove(callback)
            except ValueError:
                pass

        return unsubscribe

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)


class Computed(Signal[T]):
    """A read-only Signal derived from other Signals (recomputed on change)."""

    __slots__ = ("_fn", "_deps", "_unsubs")

    def __init__(
        self,
        fn: Callable[[], T],
        deps: Iterable[Signal[Any]],
        *,
        name: str = "",
        guard: Optional[LoopGuard] = None,
        equals: Optional[Callable[[Any, Any], bool]] = None,
    ) -> None:
        super().__init__(fn(), name=name, guard=guard, equals=equals)
        self._fn = fn
        self._deps = list(deps)
        self._unsubs = [dep.subscribe(self._recompute) for dep in self._deps]

    def _recompute(self, _value: Any = None) -> None:
        Signal.set(self, self._fn())

    def set(self, new: T, *, force: bool = False) -> bool:  # noqa: D401 - read-only
        raise TypeError(f"Computed({self._name or '?'}) is read-only")

    def dispose(self) -> None:
        for unsub in self._unsubs:
            unsub()
        self._unsubs = []
