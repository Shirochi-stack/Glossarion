"""Pluggable browser driver for the browser-backed model routes (``authnd/``, ``search/gemini``).

Desktop runs these routes in QtWebEngine: ``authnd_auth`` mints NVIDIA Build's hCaptcha token
and ``gemini_free`` drives Google Search AI Mode, in a helper subprocess by default
(``AUTHND_TOKEN_MODE`` / ``GEMINI_FREE_MODE`` = ``subprocess``) or inline (``inline``). Desktop
registers no driver, so none of that changes.

Glossarion Mobile has neither Qt nor helper processes. Its WebViewBridge
(``src/mobile/app/glossarion_mobile/services/webview_bridge.py``) registers a driver whose
pages are hidden in-app WebViews, and the routes run their own flows on those pages
(``authnd_auth._mint_captcha_token_flow``, ``gemini_free._submit_ai_mode_prompt``): the same
scripts, timing and retries as the desktop helper. Only the page operations differ.

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
* ``close()`` releases the page; idempotent.

Blocking page calls raise :class:`BrowserCancelled` (``"stream cancelled"``, the routes' own
cancel message) once :func:`cancel_pages` cancelled the page or its ``cancel_check`` returns
True. :func:`cancel_pages` never blocks: Stop handlers reach it through the routes'
``cancel_stream``.

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
    "cancel_pages",
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
