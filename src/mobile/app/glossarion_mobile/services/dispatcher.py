"""UiDispatcher: the only bridge from worker threads to the Flet UI.

Flet 1.0 computes and sends control patches from whichever thread calls
``update()``, and its own ``schedule_update`` is not thread-safe, so worker
threads (the job thread, the I/O pool, native callbacks) must never touch
controls (UI_SPEC §5.0, §7.3; mobile-app design §4.2). They use:

* ``post(fn, *args)``: run ``fn`` on the loop (``call_soon_threadsafe``), for
  rare events such as job state changes, errors and results;
* ``Channel``: a latest-value-wins slot written from any thread (progress,
  watchdog, notification text); the pump delivers only the newest value;
* ``LogBuffer``: a thread-safe ring of log lines with sequence numbers; the
  pump hands each subscriber (a mounted LogConsole) only the lines it has not
  seen, at most ``log_batch`` per tick.

A pump task wakes every 120 ms on the loop, drains channels and logs, then
sends one ``page.update(*dirty_controls)`` for every control marked with
``mark_dirty()`` since the last tick. With nothing to do it parks until woken:
by ``Channel.set``, ``mark_dirty``/``mark_page_dirty``, a new log
subscription, or a line appended to a subscribed ``LogBuffer`` (from any
thread; off-loop wake-ups are coalesced into one ``call_soon_threadsafe``).
While the app is hidden (``page.app_visible`` is False) it parks on
``page.wait_until_visible()``.

``run_in_thread(fn)`` runs blocking work on a fresh daemon thread (it inherits
the 16 MiB stack size set by ``runtime_bootstrap``) and resolves an asyncio
future on the loop.

Pure asyncio + threading: never imports Flet (``page`` is duck-typed) or any
backend module. Python 3.10 compatible.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Generic, Iterable, Optional, TypeVar

from glossarion_mobile.state.store import UI_LOOP, LoopGuard

__all__ = [
    "Channel",
    "LogBuffer",
    "LogBufferHandler",
    "LogLine",
    "LOG_KINDS",
    "UiDispatcher",
    "classify_log_line",
]

log = logging.getLogger("glossarion.dispatcher")

T = TypeVar("T")

DEFAULT_INTERVAL = 0.12  # seconds between pump ticks (UI_SPEC §7.3)
DEFAULT_LOG_BATCH = 400  # log lines handed to one subscriber per tick
DEFAULT_LOG_MAXLEN = 20000

# LogConsole filter groups (UI_SPEC §1.8: All / Errors / Thinking / API).
LOG_KINDS = ("info", "error", "thinking", "api")

_ERROR_MARKERS = (
    "❌",
    "traceback (most recent call last)",
    " error ",
    "error:",
    "[error]",
    "exception:",
    " failed:",
)
_THINKING_MARKERS = ("🧠", "💭", "[thinking]", "thinking:")
_API_MARKERS = ("📤", "📥", "🌐", "api call", "api request", "http ", "status code", "request id", "[api]")


def classify_log_line(text: str) -> str:
    """Coarse LogConsole filter group for one line (``info``/``error``/``thinking``/``api``).

    Only the console's All / Errors / Thinking / API filter uses it: request cards come from the
    shared ``direct_text_stream`` request model and the running card's wait chips (rate limit,
    key cooldown, network) from the shared ``direct_text_stream.classify_request_issue``
    (``services.jobs.classify_issue``). It keys on the desktop log's emoji and words.
    """
    lowered = f" {text.lower()} "
    if any(marker in lowered for marker in _THINKING_MARKERS):
        return "thinking"
    if any(marker in lowered for marker in _ERROR_MARKERS):
        return "error"
    if any(marker in lowered for marker in _API_MARKERS):
        return "api"
    return "info"


@dataclass(frozen=True)
class LogLine:
    seq: int
    ts: float
    text: str
    kind: str


class LogBuffer:
    """Thread-safe ring buffer of log lines; every line gets a sequence number.

    Readers keep a cursor (the last ``seq`` they consumed) and call
    ``since(cursor)``; ``gap`` tells them how many lines were evicted before
    they could read them (the ring holds ``maxlen`` lines).
    """

    def __init__(
        self,
        maxlen: int = DEFAULT_LOG_MAXLEN,
        *,
        name: str = "main",
        on_append: Optional[Callable[["LogBuffer"], Any]] = None,
    ) -> None:
        if maxlen <= 0:
            raise ValueError("maxlen must be positive")
        self.name = name
        self.maxlen = maxlen
        self._lines: deque[LogLine] = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._seq = 0
        # Called (outside the lock, on the appending thread) after every append;
        # UiDispatcher uses it to wake a parked pump.
        self.on_append = on_append

    def append(self, text: Any, kind: Optional[str] = None) -> int:
        """Add one entry (multi-line text stays one entry); returns its seq."""
        value = str(text).rstrip("\r\n")
        kind = kind if kind in LOG_KINDS else classify_log_line(value)
        with self._lock:
            self._seq += 1
            self._lines.append(LogLine(self._seq, time.time(), value, kind))
            seq = self._seq
        callback = self.on_append
        if callback is not None:
            try:
                callback(self)
            except Exception:  # never let the wake-up hook break the logging caller
                pass
        return seq

    def extend(self, lines: Iterable[Any]) -> int:
        seq = self.last_seq
        for line in lines:
            seq = self.append(line)
        return seq

    @property
    def last_seq(self) -> int:
        with self._lock:
            return self._seq

    @property
    def first_seq(self) -> int:
        """Seq of the oldest line still held (``last_seq + 1`` when empty)."""
        with self._lock:
            return self._lines[0].seq if self._lines else self._seq + 1

    def __len__(self) -> int:
        with self._lock:
            return len(self._lines)

    def since(self, seq: int, limit: Optional[int] = None) -> tuple[list[LogLine], int]:
        """Lines with ``line.seq > seq`` (oldest first, at most ``limit``) and the gap.

        ``gap`` is the number of lines after ``seq`` that were already evicted.
        """
        with self._lock:
            if not self._lines or seq >= self._seq:
                return [], 0
            first = self._lines[0].seq
            gap = max(0, first - seq - 1)
            start = max(0, seq + 1 - first)
            count = len(self._lines) - start
            if limit is not None:
                count = min(count, max(0, limit))
            out = [self._lines[start + i] for i in range(count)]
        return out, gap

    def tail(self, n: int) -> list[LogLine]:
        with self._lock:
            if n <= 0:
                return []
            return list(self._lines)[-n:]

    def snapshot(self) -> list[LogLine]:
        with self._lock:
            return list(self._lines)

    def clear(self) -> None:
        """Drop held lines; sequence numbers keep increasing."""
        with self._lock:
            self._lines.clear()


class LogBufferHandler(logging.Handler):
    """``logging`` handler that appends formatted records to a ``LogBuffer``."""

    def __init__(self, buffer: LogBuffer, level: int = logging.INFO) -> None:
        super().__init__(level)
        self.buffer = buffer
        self.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            kind = "error" if record.levelno >= logging.ERROR else None
            self.buffer.append(self.format(record), kind)
        except Exception:  # never let logging break the caller
            self.handleError(record)


_UNSET = object()


class Channel(Generic[T]):
    """Latest-value-wins slot: any thread ``set()``s, the pump delivers on the loop.

    Intermediate values written between two pump ticks are dropped on purpose
    (progress text, counters). ``get()`` returns the newest value from any thread.
    """

    def __init__(self, name: str, dispatcher: Optional["UiDispatcher"] = None, initial: Any = _UNSET) -> None:
        self.name = name
        self._dispatcher = dispatcher
        self._lock = threading.Lock()
        self._value: Any = None if initial is _UNSET else initial
        self._has_value = initial is not _UNSET
        self._pending = False
        self._writes = 0
        self._subscribers: list[Callable[[Any], Any]] = []

    def set(self, value: T) -> None:
        with self._lock:
            self._value = value
            self._has_value = True
            self._pending = True
            self._writes += 1
        dispatcher = self._dispatcher
        if dispatcher is not None:
            dispatcher._wake()

    def get(self, default: Any = None) -> Any:
        with self._lock:
            return self._value if self._has_value else default

    @property
    def pending(self) -> bool:
        with self._lock:
            return self._pending

    @property
    def writes(self) -> int:
        with self._lock:
            return self._writes

    def _take(self) -> tuple[bool, Any]:
        with self._lock:
            if not self._pending:
                return False, None
            self._pending = False
            return True, self._value

    def subscribe(self, callback: Callable[[T], Any], *, immediate: bool = False) -> Callable[[], None]:
        """Register ``callback(value)`` (runs on the loop); returns unsubscribe."""
        self._subscribers.append(callback)
        if immediate and self._has_value:
            callback(self.get())

        def unsubscribe() -> None:
            try:
                self._subscribers.remove(callback)
            except ValueError:
                pass

        return unsubscribe

    def _deliver(self) -> bool:
        has_value, value = self._take()
        if not has_value:
            return False
        for callback in list(self._subscribers):
            try:
                callback(value)
            except Exception:
                log.exception("channel %s subscriber failed", self.name)
        return True


class _LogSubscription:
    __slots__ = ("buffer", "callback", "cursor", "batch")

    def __init__(self, buffer: LogBuffer, callback: Callable[[list[LogLine], int], Any], cursor: int, batch: int) -> None:
        self.buffer = buffer
        self.callback = callback
        self.cursor = cursor
        self.batch = batch


class UiDispatcher:
    """Thread-safe posting, latest-wins channels, log fan-out and batched updates."""

    def __init__(
        self,
        page: Any = None,
        *,
        interval: float = DEFAULT_INTERVAL,
        log_batch: int = DEFAULT_LOG_BATCH,
        guard: Optional[LoopGuard] = None,
    ) -> None:
        self.page = page
        self.interval = interval
        self.log_batch = log_batch
        self.guard = guard if guard is not None else UI_LOOP
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self._channels: dict[str, Channel[Any]] = {}
        self._logs: dict[str, LogBuffer] = {}
        self._log_subs: list[_LogSubscription] = []
        self._dirty: list[Any] = []
        self._dirty_ids: set[int] = set()
        self._page_dirty = False
        self._pump_task: Optional[asyncio.Task] = None
        self._wake_event: Optional[asyncio.Event] = None
        self._wake_scheduled = False  # an off-loop wake-up is queued on the loop
        self._tasks: set[asyncio.Future] = set()
        self._running = False
        self.ticks = 0
        self.updates_sent = 0

    # ---- binding --------------------------------------------------------------

    def bind(self, loop: Optional[asyncio.AbstractEventLoop] = None) -> "UiDispatcher":
        """Bind to the UI loop; call on the loop thread (also binds the LoopGuard)."""
        if loop is None:
            loop = asyncio.get_running_loop()
        self.loop = loop
        self.guard.bind(loop)
        self._wake_event = asyncio.Event()
        return self

    @property
    def bound(self) -> bool:
        return self.loop is not None

    def on_loop_thread(self) -> bool:
        return self.guard.on_loop_thread()

    # ---- posting ------------------------------------------------------------------

    def post(self, fn: Callable[..., Any], *args: Any) -> bool:
        """Run ``fn(*args)`` on the loop (callable from any thread).

        Returns False when there is no usable loop (not bound / closed).
        Coroutine functions are scheduled as tasks.
        """
        loop = self.loop
        if loop is None or loop.is_closed():
            return False

        def run() -> None:
            try:
                result = fn(*args)
                if asyncio.iscoroutine(result):
                    self.spawn(result)
            except Exception:
                log.exception("posted UI callback %s failed", getattr(fn, "__qualname__", fn))

        try:
            loop.call_soon_threadsafe(run)
        except RuntimeError:  # loop closed while shutting down
            return False
        return True

    def submit(self, coro_fn: Callable[..., Any], *args: Any) -> Optional[concurrent.futures.Future]:
        """Schedule ``coro_fn(*args)`` on the loop from any thread."""
        loop = self.loop
        if loop is None or loop.is_closed():
            return None
        try:
            future = asyncio.run_coroutine_threadsafe(coro_fn(*args), loop)
        except RuntimeError:
            return None

        def done(f: concurrent.futures.Future) -> None:
            if not f.cancelled() and f.exception() is not None:
                log.warning("%s failed: %s", getattr(coro_fn, "__qualname__", coro_fn), f.exception())

        future.add_done_callback(done)
        return future

    def spawn(self, coro: Any) -> asyncio.Future:
        """``ensure_future`` on the loop thread, keeping a strong reference."""
        task = asyncio.ensure_future(coro)
        self._tasks.add(task)
        task.add_done_callback(self._task_done)
        return task

    def _task_done(self, task: asyncio.Future) -> None:
        self._tasks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            log.error("UI task failed", exc_info=task.exception())

    async def run_in_thread(self, fn: Callable[..., Any], *args: Any, name: str = "gl-worker") -> Any:
        """Run blocking ``fn(*args)`` on a fresh daemon thread; await its result."""
        loop = asyncio.get_running_loop()
        future = loop.create_future()

        def resolve(value: Any, error: Optional[BaseException]) -> None:
            if future.done():
                return
            if error is not None:
                future.set_exception(error)
            else:
                future.set_result(value)

        def run() -> None:
            try:
                value = fn(*args)
            except BaseException as exc:  # noqa: BLE001 - forwarded to the awaiting coroutine
                try:
                    loop.call_soon_threadsafe(resolve, None, exc)
                except RuntimeError:
                    pass
            else:
                try:
                    loop.call_soon_threadsafe(resolve, value, None)
                except RuntimeError:
                    pass

        threading.Thread(target=run, name=name, daemon=True).start()
        return await future

    # ---- channels and logs -----------------------------------------------------------

    def channel(self, name: str, initial: Any = _UNSET) -> Channel[Any]:
        existing = self._channels.get(name)
        if existing is None:
            existing = Channel(name, self, initial)
            self._channels[name] = existing
        return existing

    def log_buffer(self, name: str = "main", maxlen: int = DEFAULT_LOG_MAXLEN) -> LogBuffer:
        buffer = self._logs.get(name)
        if buffer is None:
            buffer = LogBuffer(maxlen, name=name, on_append=self._on_log_append)
            self._logs[name] = buffer
        return buffer

    def _on_log_append(self, buffer: LogBuffer) -> None:
        """``LogBuffer.on_append`` (any thread): wake the pump when someone reads ``buffer``."""
        if any(sub.buffer is buffer for sub in tuple(self._log_subs)):
            self._wake()

    def subscribe_log(
        self,
        buffer: LogBuffer,
        callback: Callable[[list[LogLine], int], Any],
        *,
        backlog: int = 0,
        batch: Optional[int] = None,
    ) -> Callable[[], None]:
        """Deliver new lines of ``buffer`` as ``callback(lines, gap)`` on the loop.

        ``backlog`` lines already in the buffer are replayed on the next tick.
        Returns an idempotent unsubscribe function.
        """
        if buffer.on_append is None:  # a buffer not made by log_buffer()
            buffer.on_append = self._on_log_append
        cursor = max(0, buffer.last_seq - max(0, backlog))
        sub = _LogSubscription(buffer, callback, cursor, batch or self.log_batch)
        self._log_subs.append(sub)
        self._wake()  # deliver the backlog even if the pump is parked

        def unsubscribe() -> None:
            try:
                self._log_subs.remove(sub)
            except ValueError:
                pass

        return unsubscribe

    # ---- batched updates -------------------------------------------------------------

    def mark_dirty(self, *controls: Any) -> None:
        """Queue controls for the next batched ``page.update`` (loop thread only)."""
        self.guard.check("UiDispatcher.mark_dirty")
        for control in controls:
            if control is None:
                continue
            if id(control) not in self._dirty_ids:
                self._dirty_ids.add(id(control))
                self._dirty.append(control)
        self._wake()

    def mark_page_dirty(self) -> None:
        self.guard.check("UiDispatcher.mark_page_dirty")
        self._page_dirty = True
        self._wake()

    @property
    def dirty_count(self) -> int:
        return len(self._dirty) + (1 if self._page_dirty else 0)

    def _wake(self) -> None:
        """Wake a parked pump (any thread)."""
        event = self._wake_event
        loop = self.loop
        if event is None or loop is None or loop.is_closed():
            return
        if self.guard.on_loop_thread() and self.guard.bound:
            event.set()
            return
        # Off the loop: at most one queued wake-up. Writers publish their data
        # before calling _wake(), so the queued one also covers them.
        if self._wake_scheduled:
            return
        self._wake_scheduled = True
        try:
            loop.call_soon_threadsafe(self._wake_on_loop)
        except RuntimeError:  # loop closed while shutting down
            self._wake_scheduled = False

    def _wake_on_loop(self) -> None:
        self._wake_scheduled = False
        event = self._wake_event
        if event is not None:
            event.set()

    @staticmethod
    def _is_mounted(control: Any) -> bool:
        try:
            return control.page is not None
        except Exception:  # Flet raises RuntimeError for unmounted controls
            return False

    @staticmethod
    def _ancestors(control: Any) -> Iterable[int]:
        parent = getattr(control, "parent", None)
        depth = 0
        while parent is not None and depth < 256:
            yield id(parent)
            parent = getattr(parent, "parent", None)
            depth += 1

    def _take_dirty(self) -> tuple[list[Any], bool]:
        controls, page_dirty = self._dirty, self._page_dirty
        self._dirty, self._dirty_ids, self._page_dirty = [], set(), False
        if page_dirty or not controls:
            return controls, page_dirty
        ids = {id(c) for c in controls}
        # A control whose ancestor is also dirty is covered by the ancestor's diff.
        top = [c for c in controls if not any(a in ids for a in self._ancestors(c))]
        return top, page_dirty

    def flush(self) -> int:
        """One pump tick (loop thread): channels, logs, then a single update.

        Returns the number of controls sent to ``page.update`` (1 for a full
        page update, 0 when nothing was dirty).
        """
        self.guard.check("UiDispatcher.flush")
        self.ticks += 1
        for channel in list(self._channels.values()):
            channel._deliver()
        for sub in list(self._log_subs):
            lines, gap = sub.buffer.since(sub.cursor, sub.batch)
            if not lines and not gap:
                continue
            if lines:
                sub.cursor = lines[-1].seq
            try:
                sub.callback(lines, gap)
            except Exception:
                log.exception("log subscriber failed")
        controls, page_dirty = self._take_dirty()
        page = self.page
        if page is None:
            return 0
        if page_dirty:
            try:
                page.update()
                self.updates_sent += 1
                return 1
            except Exception:
                log.exception("page.update() failed")
                return 0
        mounted = [c for c in controls if self._is_mounted(c)]
        if not mounted:
            return 0
        try:
            page.update(*mounted)
            self.updates_sent += 1
        except Exception:
            log.exception("batched page.update(%d controls) failed", len(mounted))
            return 0
        return len(mounted)

    # ---- pump -------------------------------------------------------------------------

    def start(self) -> asyncio.Task:
        """Start the pump task (on the loop thread)."""
        if self.loop is None:
            self.bind()
        if self._pump_task is not None and not self._pump_task.done():
            return self._pump_task
        self._running = True
        self._pump_task = asyncio.ensure_future(self._pump())
        return self._pump_task

    async def stop(self) -> None:
        self._running = False
        task = self._pump_task
        self._pump_task = None
        if self._wake_event is not None:
            self._wake_event.set()
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        for pending in list(self._tasks):
            pending.cancel()
        self._tasks.clear()

    @property
    def running(self) -> bool:
        return self._pump_task is not None and not self._pump_task.done()

    def _has_work(self) -> bool:
        if self._dirty or self._page_dirty:
            return True
        if any(c.pending for c in self._channels.values()):
            return True
        return any(sub.buffer.last_seq > sub.cursor for sub in self._log_subs)

    async def _pump(self) -> None:
        event = self._wake_event
        while self._running:
            try:
                # Sleep one interval; extra wake-ups inside it are coalesced.
                await asyncio.sleep(self.interval)
                page = self.page
                if page is not None and getattr(page, "app_visible", True) is False:
                    waiter = getattr(page, "wait_until_visible", None)
                    if waiter is not None:
                        await waiter()
                if not self._has_work():
                    if event is not None:
                        event.clear()
                        if not self._has_work():
                            await event.wait()
                    continue
                self.flush()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("dispatcher pump tick failed")
