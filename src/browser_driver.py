"""Pluggable browser driver for the browser-backed model routes (``authnd/``, ``search/gemini``).

Desktop runs these routes in QtWebEngine: ``authnd_auth`` mints NVIDIA Build's hCaptcha token
and ``gemini_free`` drives Google Search AI Mode, in a helper subprocess by default
(``AUTHND_TOKEN_MODE`` / ``GEMINI_FREE_MODE`` = ``subprocess``) or inline (``inline``). Desktop
registers no driver, so none of that changes.

Glossarion Mobile has neither Qt nor helper processes. Its WebViewBridge feature
(``src/mobile/app/glossarion_mobile/services/webview_bridge.py``) registers a driver whose
pages are hidden in-app WebViews, and the routes run their own flows on those pages
(``authnd_auth._mint_captcha_token_flow``, ``gemini_free._submit_ai_mode_prompt``): the same
scripts, timing and retries as the desktop helper. Only the page operations differ. Two
mobile drivers exist: the native in-app browser (``services/native_webview.py``, pages the
app's Flet extension owns outside the Flutter widget tree, so screen changes never rebuild
them) and the flet-webview fallback (hidden ``flet_webview`` controls).

Which path a route takes (:func:`use_driver`):

* no helper subprocesses (``mobile_runtime.subprocesses_available()`` is False, i.e. mobile):
  the driver, or a clear :class:`BrowserUnavailable` error when none is registered. A helper
  process is never started there;
* the route's mode variable set to ``driver``: the driver; set to anything else: the desktop
  helper exactly as before;
* the mode variable unset: the driver only when one is registered.

Page contract (:class:`BrowserPage`). Every method is called from a worker thread and may
block that thread (never the UI loop):

* ``load(url)`` starts a navigation and returns, like ``QWebEnginePage.load``;
* ``load_state`` is a live dict with the Qt helper's keys (:func:`new_load_state`):
  ``generation`` (+1 per navigation start), ``finished_generation`` (the generation that last
  finished; -1 while loading), ``finished_at`` (``time.monotonic()`` of that finish), ``ok``,
  ``error`` and ``last_event`` (diagnostics);
* ``url()`` / ``title()``: the current document's;
* ``run_js(script, js_timeout_ms)`` evaluates one JavaScript *expression* (a trailing ``;`` is
  allowed) in the page and returns its JSON value, or None when the page gave no value in time
  (``QWebEnginePage.runJavaScript`` callback semantics);
* ``wait(ms)`` lets the page run for ``ms`` milliseconds (the Qt helper pumps its event loop);
* ``close()`` releases the page; idempotent;
* optional ``renderer_gone`` (bool attribute): True once the page's document is gone for good
  (its renderer stopped); scripts answer None from then on. A driver without it never loses
  pages that way;
* optional ``content_suspended()``: True while the system suspends the page's web content by
  design (iOS while Glossarion is off screen: WebKit pauses it even while a background task keeps
  Python running). :class:`ReplyWatch` does not count that time; :func:`content_suspended` asks.

Blocking page calls raise :class:`BrowserCancelled` (``"stream cancelled"``, the routes' own
cancel message) once :func:`cancel_pages` cancelled the page or its ``cancel_check`` returns
True. :func:`cancel_pages` never blocks: Stop handlers reach it through the routes'
``cancel_stream``.

A route that polls a reply on a driver page (``authchatgpt/``, ``authds/``) bounds the poll by the
page's liveness with :class:`ReplyWatch` (hotfix 4): a page that stops answering, stops running its
timers (Android freezes a page it treats as hidden) or whose reply stream stays silent fails the
request with the reason instead of waiting out the request timeout. Time the system suspends the
page by design (iOS in the background, ``content_suspended``) is not counted: the reply waits until
the app is opened again.

Stdlib only (plus ``mobile_runtime``); Python 3.10 compatible; never imports Qt or Flet.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Callable, Dict, Optional, Protocol, Tuple

import mobile_runtime

__all__ = [
    "BrowserCancelled",
    "BrowserDriver",
    "BrowserPage",
    "BrowserUnavailable",
    "CANCELLED",
    "DEFAULT_VIEWPORT",
    "MODE_DRIVER",
    "PAGE_STALL_SECONDS",
    "ReplyWatch",
    "cancel_pages",
    "content_suspended",
    "get_driver",
    "new_load_state",
    "register_driver",
    "require_driver",
    "unavailable_message",
    "unregister_driver",
    "use_driver",
    "wait_until_loaded",
]

#: Value of ``AUTHND_TOKEN_MODE`` / ``GEMINI_FREE_MODE`` that selects the registered driver.
MODE_DRIVER = "driver"
#: The routes' cancel message (``authnd_auth`` / ``gemini_free`` raise ``RuntimeError`` with it).
CANCELLED = "stream cancelled"
#: Page size (logical pixels) a driver uses when a route asks for none.
DEFAULT_VIEWPORT = (1280, 900)


class BrowserUnavailable(RuntimeError):
    """No browser can run this route here: no driver is registered and helper processes are
    unavailable (Glossarion Mobile where flet-webview does not run), or ``driver`` mode was
    asked for without a driver."""


class BrowserCancelled(RuntimeError):
    """A blocking page call was cancelled (Stop, the request's cancel check, page closed)."""

    def __init__(self, message: str = CANCELLED) -> None:
        super().__init__(message)


class BrowserPage(Protocol):
    """One page (tab) a driver opened; see the module docstring for the contract."""

    load_state: Dict[str, Any]

    def load(self, url: str) -> None: ...

    def url(self) -> str: ...

    def title(self) -> str: ...

    def run_js(self, script: str, js_timeout_ms: int = 15000) -> Any: ...

    def wait(self, ms: int = 100) -> None: ...

    def close(self) -> None: ...


class BrowserDriver(Protocol):
    """Opens pages for the browser-backed routes and cancels them on Stop."""

    name: str

    def open_page(
        self,
        *,
        owner: str,
        user_agent: Optional[str] = None,
        viewport: Optional[Tuple[int, int]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> BrowserPage:
        """A new blank page tagged with ``owner`` (``"authnd"`` / ``"gemini_free"``).

        ``user_agent`` is the desktop helper's; a driver that cannot set one (flet-webview)
        ignores it. ``cancel_check`` is polled by the page's blocking calls (worker thread)."""
        ...

    def cancel_pages(self, owner: Optional[str] = None, reason: str = CANCELLED) -> None:
        """Cancel the open pages of ``owner`` (all when None). Must never block."""
        ...


_lock = threading.Lock()
_driver: Optional[BrowserDriver] = None


def register_driver(driver: BrowserDriver) -> Optional[BrowserDriver]:
    """Make ``driver`` the browser for the browser-backed routes; returns the previous one."""
    global _driver
    with _lock:
        previous, _driver = _driver, driver
    return previous


def unregister_driver(driver: Optional[BrowserDriver] = None) -> bool:
    """Remove ``driver`` (whichever is registered when None); True when one was removed."""
    global _driver
    with _lock:
        if _driver is None or (driver is not None and _driver is not driver):
            return False
        _driver = None
        return True


def get_driver() -> Optional[BrowserDriver]:
    with _lock:
        return _driver


def use_driver(mode_env: Optional[str] = None) -> bool:
    """True when a browser-backed route must run on the registered driver (see the module
    docstring). ``mode_env`` is the route's mode variable (``AUTHND_TOKEN_MODE`` /
    ``GEMINI_FREE_MODE``)."""
    if not mobile_runtime.subprocesses_available():
        return True
    value = os.environ.get(mode_env, "").strip().lower() if mode_env else ""
    if value:
        return value == MODE_DRIVER
    return get_driver() is not None


def unavailable_message(route: str) -> str:
    return (
        f"{route} needs an embedded browser. Glossarion Mobile runs it in a hidden in-app "
        "browser (WebView), which is not available here: flet-webview runs on Android, iOS "
        "and macOS only. The desktop app uses its QtWebEngine helper instead."
    )


def require_driver(route: str) -> BrowserDriver:
    """The registered driver, or :class:`BrowserUnavailable` naming ``route``."""
    driver = get_driver()
    if driver is None:
        raise BrowserUnavailable(unavailable_message(route))
    return driver


def cancel_pages(owner: Optional[str] = None, reason: str = CANCELLED) -> None:
    """Cancel the open pages of ``owner`` (every page when None).

    Never blocks and never raises; a no-op without a registered driver (desktop)."""
    driver = get_driver()
    if driver is None:
        return
    try:
        driver.cancel_pages(owner, reason)
    except Exception:
        pass


def new_load_state() -> Dict[str, Any]:
    """A page's initial ``load_state`` (the Qt helper's keys and starting values)."""
    return {
        "generation": 0,
        "finished_generation": -1,
        "finished_at": 0.0,
        "ok": False,
        "error": "",
        "last_event": "",
    }


def wait_until_loaded(page: BrowserPage, timeout_seconds: Optional[float], *, poll_ms: int = 100) -> bool:
    """Block until the page finished its latest navigation; False when ``timeout_seconds``
    passed first.

    The driver counterpart of the Qt helpers' "QEventLoop until loadFinished" wait (same
    one-second minimum): True with the finished navigation's ``ok``. ``timeout_seconds`` None
    waits without a limit; a cancelled page still ends the wait (``page.wait`` raises)."""
    deadline = None
    if timeout_seconds is not None:
        deadline = time.monotonic() + max(1.0, float(timeout_seconds))
    while True:
        state = page.load_state
        generation = int(state.get("generation", 0))
        if generation > 0 and int(state.get("finished_generation", -1)) == generation:
            return bool(state.get("ok"))
        if deadline is not None and time.monotonic() >= deadline:
            return False
        page.wait(poll_ms)


#: :class:`ReplyWatch`: seconds a reply poll on a driver page may go without any answer from the
#: page, or with the page's timers stopped, before the request fails.
PAGE_STALL_SECONDS = 90.0
#: A poll's ``paused`` (ms) above this: the page is not running its timers. A hidden page still runs
#: a 0 ms timer within about a second; a busy one a little late.
PAUSED_MS = 2000
#: Seconds an open reply stream may bring nothing before the request fails (a stalled connection).
STREAM_QUIET_SECONDS = 600.0
#: A reply poll takes at most about 40 s (a 30 s script plus a 10 s probe). A longer gap means the
#: app itself was paused meanwhile (iOS suspended it, Android froze the process): the clocks start
#: again instead of counting that time against the page.
POLL_GAP_SECONDS = 60.0
#: Seconds of a stall before it is logged (once per stall).
STALL_NOTICE_SECONDS = 30.0


class ReplyWatch:
    """The liveness bound of a route's reply poll on a driver page (Glossarion Mobile's in-app
    browser: ``authchatgpt_auth``, ``authds_auth``). With ``enabled`` False (desktop pages) every
    method is a no-op and the route's poll is bounded by its request timeout alone, as before.

    Android stops running a page it treats as hidden: a minute or five after the app left the
    screen Blink freezes the page's timers and fetch streams, and a frozen renderer may not even
    answer scripts. Such a page answered every poll with "reply on its way, nothing new", or with
    nothing, so a request waited out its whole timeout (10 h with Retry Timeout off) without a
    log line and a job stood still on one chapter for hours. The watch fails the request instead,
    with the reason, and the job's own retry handling sends it again on a new page:

    * ``silent``: no poll answered for :data:`PAGE_STALL_SECONDS`;
    * ``paused``: the polls answer, but the 0 ms probe timer each poll schedules (the poll's
      ``paused``: how many ms it has been waiting) stayed pending past :data:`PAUSED_MS` for
      :data:`PAGE_STALL_SECONDS`: the page runs no timer, so no reply comes either;
    * ``quiet``: an open reply stream brought nothing for :data:`STREAM_QUIET_SECONDS`.

    The clocks run on the time this process saw (``time.monotonic``): a gap between polls, or a
    poll, longer than :data:`POLL_GAP_SECONDS` (the app itself was paused) starts them again. So
    does every poll while ``hidden()`` is True: the system suspends the page by design (iOS while
    Glossarion is off screen, the page's ``content_suspended``; WebKit pauses web content in the
    background even while a background task or continued processing keeps Python running), so the
    reply waits until the app is opened again instead of failing (logged once). A stall is logged
    once after :data:`STALL_NOTICE_SECONDS` through ``log_fn``. ``error(message, error_type)`` makes
    the route's exception (``browser_error``; ``timeout`` for ``quiet``)."""

    def __init__(self, enabled: bool, *, route: str, site: str, host: str,
                 error: Callable[[str, str], Exception], log_fn: Optional[Callable[[str], Any]] = None,
                 hidden: Optional[Callable[[], bool]] = None) -> None:
        self.on = bool(enabled)
        self.route, self.site, self.host = route, site, host
        self.make_error = error
        self.log_fn = log_fn
        self.hidden = hidden
        self.last = time.monotonic()  # when the previous poll ended
        self.resumed: Optional[float] = None  # when the clocks last started again (paused app, suspended page)
        self.silent_since: Optional[float] = None  # the first poll of a run of unanswered ones
        self.paused_since: Optional[float] = None  # the first answer of a run with the timers stopped
        self.quiet_since: Optional[float] = None  # the last answer with stream data while a stream is open
        self.told: set = set()

    def poll_started(self) -> float:
        """The poll's start time; restarts the clocks when the app was paused since the last poll or
        the system suspends the page now."""
        now = time.monotonic()
        if not self.on:
            return now
        if self._suspended():
            self._restart(now)
            self._notice("suspended", f"⏸ {self.route}: the system paused the in-app {self.site} page while "
                                      "Glossarion is in the background; the reply continues when you open the app.")
        else:
            self.told.discard("suspended")
            if now - self.last > POLL_GAP_SECONDS:
                self._restart(now)
        return now

    def running_since(self, started: float) -> float:
        """Seconds since ``started`` that count against the page: from the later of ``started`` and the
        last time the clocks started again (just ``now - started`` on a desktop page)."""
        now = time.monotonic()
        if self.resumed is not None and self.resumed > started:
            return now - self.resumed
        return now - started

    def unanswered(self, asked: float) -> None:
        """A poll (``asked``: its start) the page did not answer."""
        if not self.on:
            return
        now, asked = self._poll_ended(asked)
        if self.silent_since is None:
            self.silent_since = asked
        self._stalled("silent", now - self.silent_since)

    def answered(self, asked: float, *, paused: Any = 0, data: bool = False, streaming: bool = False) -> None:
        """A poll the page answered: its ``paused`` (ms), whether it brought stream ``data``, whether a
        reply stream is open (``streaming``)."""
        if not self.on:
            return
        now, _asked = self._poll_ended(asked)
        self.silent_since = None
        self.told.discard("silent")
        if isinstance(paused, (int, float)) and not isinstance(paused, bool) and paused > PAUSED_MS and not data:
            if self.paused_since is None:
                self.paused_since = now
            self._stalled("paused", now - self.paused_since)
        else:
            self.paused_since = None
            self.told.discard("paused")
        if not streaming:
            self.quiet_since = None
        elif self.quiet_since is None or data:
            self.quiet_since = now
        elif now - self.quiet_since >= STREAM_QUIET_SECONDS:
            raise self.error("quiet", STREAM_QUIET_SECONDS)

    def paused(self) -> bool:
        """The page answers but its last answer said it runs no timers."""
        return self.on and self.paused_since is not None

    def error(self, kind: str, seconds: float) -> Exception:
        """The route's exception for a ``silent`` / ``paused`` / ``quiet`` stall of ``seconds``."""
        s = int(seconds)
        if kind == "quiet":
            return self.make_error(f"{self.route}: {self.host} sent nothing for {s} s while answering. "
                                   "Partial translation discarded.", "timeout")
        if kind == "paused":
            what = f"stopped running for {s} s (the system paused it while Glossarion was in the background)"
        else:
            what = (f"gave no answer for {s} s (the system may have paused it while Glossarion was in the "
                    "background)")
        return self.make_error(f"{self.route}: the in-app {self.site} page {what}. Partial translation discarded.",
                               "browser_error")

    # ---- internals ---------------------------------------------------------------------------

    def _poll_ended(self, asked: float) -> Tuple[float, float]:
        """``(now, asked)``; a poll that took longer than a poll can (the app was paused while it
        waited), or one that ends while the system suspends the page, restarts the clocks and counts
        from its end."""
        now = time.monotonic()
        if now - asked > POLL_GAP_SECONDS or self._suspended():
            self._restart(now)
            asked = now
        self.last = now
        return now, asked

    def _suspended(self) -> bool:
        """``hidden()``: the system suspends the page by design right now (never raises)."""
        probe = self.hidden
        if probe is None:
            return False
        try:
            return probe() is True
        except Exception:
            return False

    def _restart(self, now: float) -> None:
        self.resumed = now
        self.silent_since = now if self.silent_since is not None else None
        self.paused_since = now if self.paused_since is not None else None
        self.quiet_since = now if self.quiet_since is not None else None

    def _notice(self, kind: str, line: str) -> None:
        """``line`` through ``log_fn``, once per run of ``kind``."""
        if kind in self.told or self.log_fn is None:
            return
        self.told.add(kind)
        try:
            self.log_fn(line)
        except Exception:
            pass

    def _stalled(self, kind: str, seconds: float) -> None:
        if seconds >= PAGE_STALL_SECONDS:
            raise self.error(kind, PAGE_STALL_SECONDS)
        if seconds >= STALL_NOTICE_SECONDS:
            state = "has not answered for" if kind == "silent" else "has not run (paused by the system?) for"
            self._notice(kind, f"⏳ {self.route}: the in-app {self.site} page {state} {int(seconds)} s; giving it up "
                               f"to {int(PAGE_STALL_SECONDS)} s.")


def content_suspended(page: Any) -> bool:
    """The page's optional ``content_suspended()`` (see the module docstring); False for a page
    without it (desktop) and when it fails."""
    probe = getattr(page, "content_suspended", None)
    if not callable(probe):
        return False
    try:
        return probe() is True
    except Exception:
        return False
