"""WebViewBridge (mobile-app design §5.6): hidden in-app browser pages for ``authnd/``,
``search/gemini``, ``authds/`` and ``authchatgpt/``.

The desktop runs these routes in QtWebEngine helper processes. Mobile has neither Qt nor
helper processes, so this bridge registers a ``browser_driver`` driver (``src/browser_driver.py``)
whose pages are hidden ``flet_webview.WebView`` controls (``ui/components/hidden_webview.py``).
The routes keep their own flow, the desktop helper's scripts and timing
(``authnd_auth._mint_captcha_token_flow``, ``gemini_free._submit_ai_mode_prompt``); only the
page operations come from here.

Threads. Route code runs on worker threads (the job thread, the API worker pool) and calls a
:class:`WebViewPage` synchronously. Each page operation is posted to the Flet loop through
``UiDispatcher.submit`` and its answer comes back in a ``concurrent.futures.Future``:

* ``run_js(expression)`` sends :func:`page_script`, which evaluates the expression and logs
  ``GLWVB:{"id", "ok", "value"}`` to the console (split into ``part``/``parts`` chunks when
  long); ``on_console_message`` (loop thread) parses it with the Reader's console parser
  (``ui/reader/bridge.parse_console_message``, the same channel with another prefix) and
  resolves the future. No answer in time gives None, like ``QWebEnginePage.runJavaScript``;
* ``load`` / ``title`` wait for the loop coroutine; ``load_state`` follows
  ``on_page_started`` / ``on_page_ended`` (the Qt helper's loadStarted / loadFinished record);
* worker threads wait in short slices and the loop never waits for them. A Stop
  (``authnd_auth.cancel_stream`` / ``gemini_free.cancel_stream`` ->
  ``browser_driver.cancel_pages``) fails every pending future of the route's pages at once and
  unmounts their WebViews; a page also ends its calls when the request's ``cancel_check`` says
  so. Calling a page from the loop thread is refused (it would deadlock).

At most :data:`MAX_PAGES` pages are open at once (the routes' own token / sub-chunk
concurrency settings usually keep it at one); further ``open_page`` calls wait for a slot.

Background (hotfix 1, issue 3: authchatgpt/ stopped while the app was hidden). Flutter builds
widgets only in frames and produces no frames while the app is hidden or paused
(``SchedulerBinding.scheduleFrame`` returns at once unless the lifecycle state is resumed or
inactive). flet-webview creates its native WebView, starts its first load and subscribes to
``load_request`` / ``run_javascript`` in the widget's ``initState``, so a WebView mounted while
the app is hidden does nothing until the app shows again: its calls end with Flet's "Timeout
waiting for invoke method listener" and the route's page waits run out. A WebView that was
built while the app was on screen keeps working in the background (method calls and console /
page events go over the Flet connection, not through frames). So:

* while jobs run (:func:`jobs_running`, from ``BackgroundExecution``), a closed page's WebView
  is parked on :data:`BLANK_URL` instead of unmounted, and the next page takes it over
  (``_WebViewSlot`` routes the WebView's events to the page using it);
* a job whose model is a browser route (:func:`uses_browser_route`) gets one WebView built at
  its start, while the app is still on screen (:meth:`WebViewBridge.prewarm`);
* a page that needs a WebView while the app is hidden and finds none parked never builds one:
  it waits (cancellable) until one is parked or the app is visible again.

The native in-app browser (hotfix 3). Flet 1.0.3 draws ``page.overlay`` only in the top View, so
every screen push / pop moved these hidden WebViews into another widget tree and Flutter built a
new one, which loaded its URL again without the page's state (an authchatgpt/ reply was lost).
Where the extension offers it (``get_platform_info()["web_pages"]``), the pages are instead
``android.webkit.WebView`` / ``WKWebView`` objects the extension creates and owns outside the
Flutter widget tree (``services/native_webview.py``): screen changes never touch them, and they
open while the app is hidden too. :class:`HiddenPage` / :class:`PageDriver` are the worker-side
parts both drivers share. :class:`WebViewBridgeFeature` installs the native driver first and this
flet-webview bridge (with the parking above, which only it needs) as the fallback.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import threading
import time
import uuid
from typing import Any, Callable, Mapping, Optional, Tuple

from glossarion_mobile.ui.reader.bridge import parse_console_message

__all__ = [
    "BLANK_URL",
    "BROWSER_ROUTE_PREFIXES",
    "CHUNK_CHARS",
    "CONSOLE_PREFIX",
    "HiddenPage",
    "LIMITATIONS",
    "MAX_PAGES",
    "NATIVE_LIMITATIONS",
    "PageDriver",
    "UNSUPPORTED_REASON",
    "WebViewBridge",
    "WebViewBridgeFeature",
    "WebViewPage",
    "WebViewUnavailable",
    "availability",
    "configured_models",
    "current",
    "job_models",
    "jobs_running",
    "limitations",
    "page_script",
    "platform_supported",
    "suspends_hidden_pages",
    "uses_browser_route",
]

log = logging.getLogger("glossarion.webview_bridge")

#: Console prefix of the hidden pages' answers (the Reader's pages use ``GLRDR:``).
CONSOLE_PREFIX = "GLWVB:"
#: Longest console line a page writes; longer answers are split into numbered chunks.
CHUNK_CHARS = 16000
MAX_CHUNKS = 4096
MAX_PAGES = 4
#: Page size (logical pixels) when a route asks for none (authnd/: only the invisible hCaptcha).
DEFAULT_VIEWPORT = (480, 800)
LOAD_TIMEOUT = 20.0  # seconds for the loop to mount the WebView / start a navigation
TITLE_TIMEOUT = 5.0
POLL_SECONDS = 0.05
#: Seconds a request waits for a free page (every one in use) before the wait is logged.
SLOT_NOTICE_SECONDS = 30.0
#: The hidden pages never leave the web: app / store / local links are refused.
PREVENT_LINKS = ("intent:", "market:", "file:", "content:")
CANCELLED = "stream cancelled"  # browser_driver.CANCELLED (the routes' cancel message)
#: Flet's error when a method call (``load_request``) finds no listener on the Flutter side: the
#: WebView widget is being (re)built (a screen change, a busy UI loop) and has not subscribed yet.
NO_LISTENER = "waiting for invoke method listener"
#: What a parked WebView shows between pages (no network, no script).
BLANK_URL = "about:blank"
#: Seconds a parked WebView's blank page may take to finish before it counts as parked anyway.
PARK_SETTLE = 3.0
#: Seconds a page (app visible) waits for a WebView being parked before it builds its own.
PARK_WAIT = 5.0
#: Times ``load`` waits for a WebView again after the app was hidden at the moment it needed one.
LOAD_ATTEMPTS = 3
#: Models whose requests run on these hidden pages (``authnd`` also covers ``authnd/``).
BROWSER_ROUTE_PREFIXES = ("authchatgpt/", "authnd", "authds/", "search/gemini")
HIDDEN_WAIT_MESSAGE = (
    "in-app browser: Glossarion is in the background and no in-app page is ready; the request "
    "waits until one is free or the app is opened again (Android and iOS build no new page "
    "while the app is hidden)"
)

UNSUPPORTED_REASON = (
    "The hidden in-app browser (flet-webview) runs on Android, iOS and macOS only, so AuthND and "
    "Gemini Free cannot run here."
)
_LIMITATIONS_HEAD = (
    "Experimental, best effort.\n"
    "• hCaptcha may ask for an interactive challenge, which the hidden browser cannot answer: the "
    "AuthND request then fails after the token timeout (Settings › Response handling › NIM / AuthND "
    "token helpers). Retry later or use another route.\n"
    "• The in-app browser identifies itself as a mobile WebView (its user agent cannot be changed), "
    "so NVIDIA and Google may serve other pages or extra verification than on desktop.\n"
    "• Google may show a consent or verification page instead of AI Mode; the request then fails "
    "with that message.\n"
)
_LIMITATIONS_TAIL = (
    f"• Every request opens its own hidden page; Token concurrency and the Gemini Free sub-chunk "
    f"concurrency limit how many run at once (never more than {MAX_PAGES})."
)
LIMITATIONS = (
    _LIMITATIONS_HEAD
    + "• In the background: on Android these requests keep running while a job runs (its "
    "notification is showing): the job opens its in-app page while Glossarion is on screen and "
    "reuses it afterwards. A request that needs a new page while Glossarion is hidden waits until "
    "you open the app again. iOS pauses the in-app browser in the background: keep Glossarion "
    "open while these requests run.\n"
    + _LIMITATIONS_TAIL
)
#: ``LIMITATIONS`` with the native in-app browser (hotfix 3): only the background bullet differs.
NATIVE_LIMITATIONS = (
    _LIMITATIONS_HEAD
    + "• In the background: on Android these requests keep running while a job runs (its "
    "notification is showing), and new in-app pages open there too. iOS pauses the in-app browser "
    "in the background: keep Glossarion open while these requests run.\n"
    + _LIMITATIONS_TAIL
)

_status_lock = threading.Lock()
_current: Optional["PageDriver"] = None
_reason = UNSUPPORTED_REASON


def current() -> Optional["PageDriver"]:
    """The installed (registered) bridge, if any: the native driver or the flet-webview one."""
    with _status_lock:
        return _current


def availability() -> Tuple[bool, str]:
    """``(True, "")`` while a bridge is registered, else ``(False, why)``."""
    with _status_lock:
        if _current is not None and _current.registered:
            return True, ""
        return False, _reason


def _set_current(bridge: Optional["PageDriver"], reason: str = "") -> None:
    global _current, _reason
    with _status_lock:
        _current = bridge
        _reason = reason or UNSUPPORTED_REASON


def limitations() -> str:
    """What the Experimental rows say about the in-app browser: :data:`NATIVE_LIMITATIONS` while
    the native driver (hotfix 3) is installed, else :data:`LIMITATIONS`."""
    from glossarion_mobile.services.native_webview import NATIVE_NAME

    bridge = current()
    if bridge is not None and getattr(bridge, "name", None) == NATIVE_NAME:
        return NATIVE_LIMITATIONS
    return LIMITATIONS


def platform_supported(page: Any) -> bool:
    """flet-webview renders here (the Reader's check: Android, iOS or macOS, never web)."""
    from glossarion_mobile.ui.reader.reader_view import webview_supported

    return webview_supported(page)


def suspends_hidden_pages() -> bool:
    """``PageDriver.suspends_hidden_pages`` here (hotfix 4): True on iOS, where WebKit suspends web
    content by design while the app is off screen; False on Android and elsewhere."""
    try:
        from glossarion_mobile.runtime_bootstrap import detect_platform

        return detect_platform() == "ios"
    except Exception:
        return False


def uses_browser_route(model: Any) -> bool:
    """``model`` runs on these hidden pages (authchatgpt/, authnd/, authds/, search/gemini)."""
    return str(model or "").strip().lower().startswith(BROWSER_ROUTE_PREFIXES)


def job_models(snap: Any) -> list:
    """The models a job's spec names (a chat send's ``model``, ``config_overrides.model``)."""
    params = getattr(getattr(snap, "spec", None), "params", None)
    if not isinstance(params, Mapping):
        return []
    overrides = params.get("config_overrides")
    found = [params.get("model"), overrides.get("model") if isinstance(overrides, Mapping) else None]
    return [str(model) for model in found if model]


def configured_models(store: Any) -> list:
    """The models the settings send requests with: ``model`` and the multi-key / fallback keys'."""
    get = getattr(store, "get", None)
    if not callable(get):
        return []
    try:
        found = [get("model")]
        for key in ("multi_api_keys", "fallback_keys"):
            entries = get(key)
            if isinstance(entries, (list, tuple)):
                found.extend(entry.get("model") for entry in entries if isinstance(entry, Mapping))
    except Exception as exc:  # the store is closing / not loaded: no warm-up, nothing breaks
        log.debug("configured models unavailable: %s", exc)
        return []
    return [str(model) for model in found if model]


def jobs_running(running: bool, snap: Any = None) -> None:
    """``BackgroundExecution``: a job started (``running``; ``snap`` is its snapshot) or the queue
    drained. While jobs run the flet-webview bridge parks used WebViews for the next page, and a
    job whose model is a browser route gets one built now, while the app is still on screen. No-op
    without a registered bridge (desktop, web) and with the native driver (its pages open any
    time, also while the app is hidden: :class:`PageDriver`'s no-ops)."""
    bridge = current()
    if bridge is None:
        return
    if running:
        bridge.job_started(job_models(snap))
    else:
        bridge.keep_warm(False)


class WebViewUnavailable(RuntimeError):
    """The page needed a new WebView while the app is hidden (none can be built then)."""


def _clean_expression(expression: str) -> str:
    text = str(expression or "").strip()
    while text.endswith(";"):
        text = text[:-1].rstrip()
    return text or "undefined"


def page_script(request_id: str, expression: str, *, chunk_chars: int = CHUNK_CHARS) -> str:
    """The ``run_javascript`` source that evaluates ``expression`` and logs its JSON answer.

    The expression sits on its own lines inside parentheses (a trailing ``;`` is dropped; a
    trailing ``//`` comment cannot swallow the closing parenthesis). The answer is
    ``GLWVB:{"id", "ok": true, "value"}`` or ``{"id", "ok": false, "error"}``; one longer than
    ``chunk_chars`` goes out as ``{"id", "part", "parts", "chunk"}`` lines. The source ends with
    ``true`` so WKWebView gets a result type it supports.
    """
    head = (
        "(function () {\n"
        f"  var id = {json.dumps(str(request_id))}, prefix = {json.dumps(CONSOLE_PREFIX)}, "
        f"size = {max(256, int(chunk_chars))};\n"
        "  function send(text) {\n"
        "    if (text.length <= size) { console.log(prefix + text); return; }\n"
        "    var parts = Math.ceil(text.length / size);\n"
        "    for (var i = 0; i < parts; i++) {\n"
        "      console.log(prefix + JSON.stringify({id: id, part: i, parts: parts,\n"
        "                                         chunk: text.slice(i * size, (i + 1) * size)}));\n"
        "    }\n"
        "  }\n"
        "  var envelope;\n"
        "  try {\n"
        "    var value = (\n"
    )
    tail = (
        "\n    );\n"
        "    envelope = {id: id, ok: true, value: value === undefined ? null : value};\n"
        "  } catch (error) {\n"
        "    envelope = {id: id, ok: false, error: String(error && (error.message || error))};\n"
        "  }\n"
        "  var text;\n"
        "  try { text = JSON.stringify(envelope); }\n"
        "  catch (error) { text = JSON.stringify({id: id, ok: false, error: 'the value is not JSON'}); }\n"
        "  send(text);\n"
        "})();\n"
        "true;"
    )
    return head + _clean_expression(expression) + tail


def _flet_webview(slot: "_WebViewSlot", url: str) -> Any:
    """The real hidden WebView of ``slot`` (the bridge's default factory); its events go to the
    slot, which hands them to the page using the WebView."""
    import flet_webview as fwv

    width, height = slot.viewport
    return fwv.WebView(
        url=url,
        width=width,
        height=height,
        prevent_links=list(PREVENT_LINKS),
        on_page_started=slot.on_page_started,
        on_page_ended=slot.on_page_ended,
        on_url_change=slot.on_url_change,
        on_web_resource_error=slot.on_resource_error,
        on_console_message=slot.on_console,
    )


def _short_url(url: Any) -> str:
    text = str(url or "")
    return text.split("#", 1)[0].split("?", 1)[0][:300]


class HiddenPage:
    """What every hidden page shares (``browser_driver.BrowserPage``): the cancellation rules,
    the waits in short slices, the calls posted to the UI loop and the ``load_state`` record.

    :class:`WebViewPage` (a flet-webview WebView) and ``native_webview.NativeWebPage`` (a page the
    extension owns, hotfix 3) add how they reach their WebView. ``driver`` is the
    :class:`PageDriver` that opened the page. The public methods are the worker-thread API;
    ``_page_started`` / ``_page_finished`` / ``_url_changed`` / ``_resource_error`` are the
    load-state transitions (loop thread), the Qt helper's loadStarted / loadFinished record.
    """

    def __init__(
        self,
        driver: Any,
        page_id: str,
        owner: str,
        viewport: Tuple[int, int],
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> None:
        from browser_driver import new_load_state

        self.driver = driver
        self.id = page_id
        self.owner = owner
        self.viewport = viewport
        self.cancel_check = cancel_check
        self.load_state = new_load_state()
        self.closed = False
        self._url = ""
        self._cancel_reason: Optional[str] = None
        self._lock = threading.Lock()
        self._pending: dict[str, concurrent.futures.Future] = {}

    # ---- worker-thread API (browser_driver.BrowserPage) -----------------------------------

    def url(self) -> str:
        return self._url

    def wait(self, ms: int = 100) -> None:
        """Let the page run for ``ms`` milliseconds (cancellable)."""
        deadline = time.monotonic() + max(0, int(ms)) / 1000.0
        while True:
            self._raise_if_cancelled()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(min(POLL_SECONDS, remaining))

    def content_suspended(self) -> bool:
        """``browser_driver.BrowserPage``'s optional ``content_suspended``: the system suspends this
        page by design right now (iOS while Glossarion is off screen; :meth:`PageDriver.pages_suspended`)."""
        try:
            return self.driver.pages_suspended() is True
        except Exception:
            return False

    # ---- cancellation (any thread; never blocks) -------------------------------------------

    def _cancelled(self) -> Optional[str]:
        reason = self._cancel_reason
        if reason is None and self.cancel_check is not None:
            try:
                if self.cancel_check():
                    reason = CANCELLED
            except Exception:
                reason = None
        return reason

    def _raise_if_cancelled(self) -> None:
        from browser_driver import BrowserCancelled

        reason = self._cancelled()
        if reason is not None:
            raise BrowserCancelled(reason)

    def _check_usable(self) -> None:
        self._raise_if_cancelled()
        if self.closed:
            raise RuntimeError("the in-app browser page is closed")
        if self.driver.on_loop_thread():
            raise RuntimeError("the in-app browser cannot be driven from the UI thread")

    @staticmethod
    def _fail(futures: list, reason: str) -> None:
        from browser_driver import BrowserCancelled

        for future in futures:
            if not future.done():
                try:
                    future.set_exception(BrowserCancelled(reason))
                except concurrent.futures.InvalidStateError:
                    pass

    def _wait(self, future: concurrent.futures.Future, deadline: Optional[float]) -> Any:
        """``future``'s result, waiting in slices; None once ``deadline`` passed."""
        while True:
            remaining = None if deadline is None else deadline - time.monotonic()
            try:
                return future.result(timeout=POLL_SECONDS if remaining is None
                                     else max(0.0, min(POLL_SECONDS, remaining)))
            except concurrent.futures.TimeoutError:
                pass
            self._raise_if_cancelled()
            if remaining is not None and remaining <= 0:
                return None

    def _call(self, coro_fn: Callable[..., Any], *args: Any, timeout: float, what: str) -> Any:
        """Run ``coro_fn(*args)`` on the loop and wait for its result; a cancel or the timeout
        cancels the loop call too."""
        self._check_usable()
        future = self.driver.submit(coro_fn, *args)
        if future is None:
            raise RuntimeError(f"the in-app browser could not {what}: the app's UI loop is not running")
        deadline = time.monotonic() + timeout
        while True:
            try:
                return future.result(timeout=max(0.0, min(POLL_SECONDS, deadline - time.monotonic())))
            except concurrent.futures.TimeoutError:
                pass
            try:
                self._raise_if_cancelled()
            except Exception:
                future.cancel()
                raise
            if time.monotonic() >= deadline:
                future.cancel()
                raise RuntimeError(f"the in-app browser did not {what} within {timeout:g} s")

    def _resolve(self, request_id: str, envelope: Any) -> None:
        with self._lock:
            future = self._pending.get(request_id)
        if future is None or future.done():
            return
        try:
            future.set_result(envelope)
        except concurrent.futures.InvalidStateError:
            pass

    # ---- load_state (loop thread): the Qt helper's loadStarted / loadFinished record --------

    def _page_started(self, url: str) -> None:
        state = self.load_state
        state["generation"] = int(state.get("generation", 0)) + 1
        state["finished_generation"] = -1
        state["finished_at"] = 0.0
        state["ok"] = False
        state["error"] = ""
        if url:
            self._url = url
        state["last_event"] = f"page started url={_short_url(url)}"

    def _page_finished(self, url: str, *, error: str = "") -> None:
        """A navigation finished: ``ok`` is True as with flet-webview's page-ended event; a
        failure the platform reported (``error``) goes to ``load_state["error"]`` only."""
        state = self.load_state
        if url:
            self._url = url
        state["ok"] = True
        if error:
            state["error"] = str(error)[:300]
        state["finished_generation"] = int(state.get("generation", 0))
        state["finished_at"] = time.monotonic()
        state["last_event"] = f"page ended url={_short_url(url)}"

    def _url_changed(self, url: str) -> None:
        if url:
            self._url = url

    def _resource_error(self, text: Any, *, main_frame: bool = False) -> None:
        # Sub-resources fail too (ads, trackers), so this never marks the document failed; a main
        # frame failure the platform names (the native pages) is kept as the document's error.
        detail = str(text or "")[:300]
        self.load_state["last_event"] = f"resource error: {detail}"
        if main_frame and detail:
            self.load_state["error"] = detail


class PageDriver:
    """What both ``browser_driver`` drivers share: the page registry, at most ``max_pages`` open
    pages (further ``open_page`` calls wait, cancellably), Stop (``cancel_pages``; never blocks),
    the UI-loop plumbing and the registration as the backend's browser driver.

    :class:`WebViewBridge` (flet-webview) and ``native_webview.NativeWebDriver`` (pages the
    extension owns, hotfix 3) are the two drivers; ``page_class`` is the page each one opens.
    ``job_started`` / ``keep_warm`` are no-ops here: only the flet-webview bridge keeps WebViews
    for jobs.

    ``suspends_hidden_pages`` (hotfix 4, set by the installers on iOS): WebKit suspends web content
    by design while Glossarion is off screen, even while a background task or continued processing
    keeps Python running. :meth:`pages_suspended` (each page's ``content_suspended``) is then True
    and the routes' reply watch (``browser_driver.ReplyWatch``) waits for the app instead of failing
    the request. Android never: a page that stops there is a fault the watch bounds.
    """

    name = ""
    page_class: Any = HiddenPage

    def __init__(
        self,
        dispatcher: Any,
        *,
        max_pages: int = MAX_PAGES,
        viewport: Tuple[int, int] = DEFAULT_VIEWPORT,
    ) -> None:
        self.dispatcher = dispatcher
        self.max_pages = max(1, int(max_pages))
        self.default_viewport = (int(viewport[0]), int(viewport[1]))
        self.registered = False
        self.opened = 0
        self.suspends_hidden_pages = False
        self._slots = threading.BoundedSemaphore(self.max_pages)
        self._pages: dict[str, Any] = {}
        self._lock = threading.Lock()

    def app_shown(self) -> bool:
        """Glossarion is on screen (the drivers answer from Flet's ``page.app_visible``); True here."""
        return True

    def pages_suspended(self) -> bool:
        """The system suspends this driver's pages by design right now: iOS (``suspends_hidden_pages``)
        while Glossarion is off screen. Any thread; never raises."""
        if not self.suspends_hidden_pages:
            return False
        try:
            return not self.app_shown()
        except Exception:
            return False

    # ---- browser_driver.BrowserDriver ------------------------------------------------------

    def open_page(
        self,
        *,
        owner: str,
        user_agent: Optional[str] = None,
        viewport: Optional[Tuple[int, int]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> Any:
        """A new hidden page (worker thread). ``user_agent`` is ignored: neither flet-webview nor
        the native pages set one, so the page keeps the system WebView's."""
        from browser_driver import BrowserCancelled

        del user_agent
        if self.on_loop_thread():
            raise RuntimeError("the in-app browser cannot be driven from the UI thread")
        # Bounded by the pages in use: every route bounds its own page waits and replies (hotfix 4:
        # browser_driver.ReplyWatch fails a reply whose page stopped). A long wait is logged once.
        waiting_since = time.monotonic()
        told = False
        while not self._slots.acquire(timeout=0.1):
            try:
                cancelled = bool(cancel_check()) if cancel_check is not None else False
            except Exception:
                cancelled = False
            if cancelled:
                raise BrowserCancelled()
            if not told and time.monotonic() - waiting_since >= SLOT_NOTICE_SECONDS:
                told = True
                log.info("in-app browser: a %s request waits for a free page (%d of %d in use)",
                         str(owner or "browser")[:40], self.open_pages, self.max_pages)
        size = viewport or self.default_viewport
        try:
            page = self._new_page(uuid.uuid4().hex[:16], str(owner or ""),
                                  (max(1, int(size[0])), max(1, int(size[1]))), cancel_check)
        except BaseException:
            self._slots.release()
            raise
        with self._lock:
            self._pages[page.id] = page
            self.opened += 1
        return page

    def _new_page(self, page_id: str, owner: str, viewport: Tuple[int, int],
                  cancel_check: Optional[Callable[[], bool]]) -> Any:
        return self.page_class(self, page_id, owner, viewport, cancel_check)

    def cancel_pages(self, owner: Optional[str] = None, reason: str = CANCELLED) -> None:
        """Cancel the open pages of ``owner`` (all when None); never blocks."""
        with self._lock:
            pages = [page for page in self._pages.values() if owner is None or page.owner == owner]
        for page in pages:
            page.cancel(reason)

    # ---- plumbing ------------------------------------------------------------------------------

    def forget(self, page: Any) -> None:
        with self._lock:
            removed = self._pages.pop(page.id, None)
        if removed is not None:
            self._slots.release()

    @property
    def open_pages(self) -> int:
        with self._lock:
            return len(self._pages)

    def pages(self, owner: Optional[str] = None) -> list:
        with self._lock:
            return [page for page in self._pages.values() if owner is None or page.owner == owner]

    def submit(self, coro_fn: Callable[..., Any], *args: Any) -> Optional[concurrent.futures.Future]:
        return self.dispatcher.submit(coro_fn, *args)

    def post(self, fn: Callable[..., Any], *args: Any) -> bool:
        return bool(self.dispatcher.post(fn, *args))

    def on_loop_thread(self) -> bool:
        guard = getattr(self.dispatcher, "guard", None)
        return bool(guard is not None and getattr(guard, "bound", False) and guard.on_loop_thread())

    # ---- jobs (``jobs_running``): only the flet-webview bridge keeps WebViews for them --------

    def job_started(self, models: Any = ()) -> None:
        """A job started (any thread). No-op: a page opens whenever a request needs one."""

    def keep_warm(self, on: bool) -> None:
        """While jobs run (any thread). No-op: nothing is kept between pages."""

    # ---- registration ---------------------------------------------------------------------------

    def register(self) -> "PageDriver":
        """Become the backend's browser driver (authnd/ and search/gemini then use it)."""
        import browser_driver

        browser_driver.register_driver(self)
        self.registered = True
        _set_current(self)
        return self

    def unregister(self, reason: str = "The in-app browser was shut down.") -> None:
        import browser_driver

        browser_driver.unregister_driver(self)
        self.registered = False
        if current() is self:
            _set_current(None, reason)

    def shutdown(self) -> None:
        """Unregister and cancel every page; let go of what is kept for jobs (app teardown)."""
        self.unregister()
        self.cancel_pages(None, "the app is closing")
        self.keep_warm(False)

    def status(self) -> dict:
        with self._lock:
            owners = sorted({page.owner for page in self._pages.values()})
            count = len(self._pages)
        return {"registered": self.registered, "driver": self.name, "open_pages": count, "owners": owners,
                "opened": self.opened, "max_pages": self.max_pages}


class _WebViewSlot:
    """One mounted hidden WebView and the page using it now (None while it is parked).

    The WebView's event handlers are the slot's, so the page that takes a parked WebView over
    gets its events. ``alive`` turns True with the first event: Flutter built the WebView (its
    ``initState`` ran), so ``load_request`` / ``run_javascript`` reach it even while the app is
    hidden. Events of the blank page a parked WebView shows reach no page. Other attributes
    (``load_state``, ``url()``, ``owner``, ``id``) are the current page's, for diagnostics.
    Loop thread only, except ``alive`` (a bool) which workers may read.
    """

    def __init__(self, bridge: "WebViewBridge", viewport: Tuple[int, int],
                 page: Optional["WebViewPage"] = None) -> None:
        self.bridge = bridge
        self.viewport = viewport
        self.page = page
        self.webview: Any = None
        self.entry: Any = None
        self.alive = False
        self.parking = False  # navigating to BLANK_URL before it becomes a spare (or a warm-up WebView)
        self.warmup = False  # built by prewarm(): its first event proves it is built (no old document)
        self.parked: Optional[asyncio.Event] = None

    def __getattr__(self, name: str) -> Any:
        page = self.__dict__.get("page")
        if page is None:
            raise AttributeError(name)
        return getattr(page, name)

    def _target(self, e: Any, *, ended: bool = False) -> Optional["WebViewPage"]:
        self.alive = True
        if str(getattr(e, "data", "") or "").lower().startswith("about:"):
            if self.parking and (ended or self.warmup):
                self.bridge.parked_on_loop(self)
            return None  # the routes only open web pages: this is a parked WebView's blank page
        return self.page

    def on_page_started(self, e: Any) -> None:
        page = self._target(e)
        if page is not None:
            page.on_page_started(e)

    def on_page_ended(self, e: Any) -> None:
        page = self._target(e, ended=True)
        if page is not None:
            page.on_page_ended(e)

    def on_url_change(self, e: Any) -> None:
        page = self._target(e)
        if page is not None:
            page.on_url_change(e)

    def on_resource_error(self, e: Any) -> None:  # ``data`` is the error text, not a URL
        self.alive = True
        page = self.page
        if page is not None:
            page.on_resource_error(e)

    def on_console(self, e: Any) -> None:
        self.alive = True
        page = self.page
        if page is not None:
            page.on_console(e)


class WebViewPage(HiddenPage):
    """One hidden page: a ``browser_driver.BrowserPage`` over one off-stage WebView.

    The public methods are the worker-thread API; ``on_*`` are the WebView's event handlers
    (loop thread); ``_*_on_loop`` coroutines run on the loop.
    """

    def __init__(
        self,
        bridge: "WebViewBridge",
        page_id: str,
        owner: str,
        viewport: Tuple[int, int],
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> None:
        super().__init__(bridge, page_id, owner, viewport, cancel_check)
        self.bridge = bridge
        self.slot: Optional[_WebViewSlot] = None  # the WebView this page uses (loop thread only)
        self.webview: Any = None  # the slot's WebView (loop thread only)
        self.entry: Any = None  # host overlay entry (loop thread only)
        self._claimed: Optional[_WebViewSlot] = None  # a parked WebView claimed for the first load
        self.builds = 0  # WebViews built for this page (each overlay entry gets its own key)
        self.reused = 0  # parked WebViews this page took over
        self.console_lines = 0
        self._chunks: dict[str, list] = {}  # loop thread only

    # ---- worker-thread API (browser_driver.BrowserPage) -----------------------------------

    def load(self, url: str) -> None:
        url = str(url or "")
        if not url.lower().startswith(("https://", "http://")):
            raise ValueError(f"the in-app browser only opens web pages, not {url[:40]!r}")
        self._url = url
        for attempt in range(LOAD_ATTEMPTS):
            if self.webview is None and self._claimed is None:
                self.bridge.claim(self)  # a parked WebView, or the app is visible (waits otherwise)
            try:
                self._call(self._load_on_loop, url, timeout=LOAD_TIMEOUT, what="open the page")
                return
            except WebViewUnavailable as exc:  # the app was hidden when the page needed a new WebView
                if attempt + 1 >= LOAD_ATTEMPTS:
                    raise RuntimeError(f"the in-app browser could not open the page: {exc}") from exc

    def title(self) -> str:
        try:
            value = self._call(self._title_on_loop, timeout=TITLE_TIMEOUT, what="read the page title")
        except Exception as exc:
            if self._cancelled() is not None:
                raise
            log.debug("page title unavailable: %s", exc)
            return ""
        return str(value or "")

    def run_js(self, script: str, js_timeout_ms: int = 15000) -> Any:
        """The expression's JSON value, or None when the page gave none within ``js_timeout_ms``."""
        self._check_usable()
        request_id = uuid.uuid4().hex
        future: concurrent.futures.Future = concurrent.futures.Future()
        with self._lock:
            self._pending[request_id] = future
        try:
            source = page_script(request_id, script)
            if self.bridge.submit(self._run_js_on_loop, request_id, source) is None:
                return None  # the UI loop is gone (app closing)
            deadline = time.monotonic() + max(0, int(js_timeout_ms)) / 1000.0
            envelope = self._wait(future, deadline)
        finally:
            with self._lock:
                self._pending.pop(request_id, None)
        if not isinstance(envelope, dict):
            return None
        if not envelope.get("ok"):
            log.debug("page script failed: %s", str(envelope.get("error") or "")[:300])
            return None
        return envelope.get("value")

    def close(self) -> None:
        """Release the page (any thread; idempotent): fail its waits, then park its WebView for
        the next page while jobs run (:func:`jobs_running`), else unmount it."""
        with self._lock:
            if self.closed:
                return
            self.closed = True
            pending = list(self._pending.values())
            self._pending.clear()
        self._fail(pending, "the in-app browser page was closed")
        if self._cancel_reason is None:
            self.bridge.expect_parking(self.slot)  # before the slot is free: the next page waits for it
        self.bridge.forget(self)
        self.bridge.post(self._release_on_loop)

    # ---- cancellation (any thread; never blocks) -------------------------------------------

    def cancel(self, reason: str = CANCELLED) -> None:
        with self._lock:
            if self._cancel_reason is None:
                self._cancel_reason = str(reason or CANCELLED)
            pending = list(self._pending.values())
        self._fail(pending, self._cancel_reason)
        self.bridge.post(self._unmount_on_loop)

    # ---- loop side ---------------------------------------------------------------------------

    async def _load_on_loop(self, url: str) -> None:
        if self.closed or self._cancel_reason is not None:
            raise RuntimeError("the in-app browser page is closed")
        if self.webview is None:
            claimed, self._claimed = self._claimed, None
            if claimed is not None and claimed.webview is not None:
                self._adopt_on_loop(claimed)  # a parked WebView: navigated below like a later load
            elif not self.bridge.app_visible():
                # Flutter would build a new WebView only once the app shows again (no frames
                # while hidden): load() waits for a parked one or for the app instead.
                raise WebViewUnavailable("Glossarion is in the background")
            else:
                # flet-webview loads its ``url`` when the control is built; later loads use
                # load_request (a changed ``url`` property does not navigate a built WebView).
                self._build_on_loop(url)
                return
        # the Reader's helper (one place for the rule); a failed load raises out of load()
        from glossarion_mobile.ui.reader.reader_view import navigate_webview

        try:
            await navigate_webview(self.webview, url)
        except Exception as exc:
            if NO_LISTENER not in str(exc):
                raise
            if not self.bridge.app_visible():
                # The WebView's Flutter State is gone and none is built while the app is hidden.
                log.info("in-app browser: no load_request listener while the app is hidden (%s)",
                         str(exc)[:200])
                self._unmount_on_loop()
                raise WebViewUnavailable("Glossarion is in the background") from exc
            # No listener answered load_request within Flet's 10 s: the Flutter WebView was being
            # built again and never took the call (authchatgpt/ on a phone: "did not open
            # chatgpt.com/", twice, then the request failed). A new WebView built with ``url``
            # loads it the way every first load does; the cookie jar is shared and the routes set
            # up their page again after each load, so nothing of the old document is needed.
            log.info("in-app browser: no load_request listener (%s); building the page's WebView again",
                     str(exc)[:200])
            self._build_on_loop(url)

    def _build_on_loop(self, url: str) -> None:
        """Mount a new WebView that loads ``url`` (replacing the page's current one)."""
        old, self.slot = self.slot, None
        if old is not None:
            self.bridge.retire_on_loop(old, reuse=False)
        self.builds += 1
        self._chunks.clear()
        slot = _WebViewSlot(self.bridge, self.viewport, self)
        slot.webview = self.bridge.make_webview(slot, url)
        # a Flet 1.0.3 control re-rendered under an old key keeps its old state: a new key per build
        key = f"glwvb-{self.id}" if self.builds == 1 else f"glwvb-{self.id}-{self.builds}"
        slot.entry = self.bridge.host.add(slot.webview, size=self.viewport, key=key)
        self._use_slot(slot)

    def _adopt_on_loop(self, slot: _WebViewSlot) -> None:
        """Take a parked WebView over (its blank page is replaced by the next navigation)."""
        self._chunks.clear()
        self.reused += 1
        if tuple(slot.viewport) != tuple(self.viewport):
            slot.viewport = self.viewport
            resize = getattr(self.bridge.host, "resize", None)
            if callable(resize) and slot.entry is not None:
                resize(slot.entry, self.viewport)
        self._use_slot(slot)

    def _use_slot(self, slot: _WebViewSlot) -> None:
        slot.page = self
        slot.warmup = False
        self.slot = slot
        self.webview = slot.webview
        self.entry = slot.entry

    async def _title_on_loop(self) -> str:
        if self.webview is None:
            return ""
        return await self.webview.get_title() or ""

    async def _run_js_on_loop(self, request_id: str, source: str) -> None:
        try:
            if self.webview is None:
                raise RuntimeError("no page is loaded")
            await self.webview.run_javascript(source)
        except Exception as exc:  # no answer will come: resolve now instead of at the timeout
            log.debug("run_javascript failed: %s", exc)
            self._resolve(request_id, None)

    def _unmount_on_loop(self) -> None:
        """Cancel / lost State: the page's WebView goes (an unused claimed one is kept)."""
        self._drop_slots_on_loop(reuse=False)

    def _release_on_loop(self) -> None:
        """Close: a WebView that served a request without a cancel is parked while jobs run."""
        self._drop_slots_on_loop(reuse=self._cancel_reason is None)

    def _drop_slots_on_loop(self, *, reuse: bool) -> None:
        slot, self.slot = self.slot, None
        claimed, self._claimed = self._claimed, None
        self.entry = None
        self.webview = None
        self._chunks.clear()
        if slot is not None:
            self.bridge.retire_on_loop(slot, reuse=reuse)
        if claimed is not None:  # claimed for a load that never ran: still parked on the blank page
            self.bridge.retire_on_loop(claimed, reuse=True, blank=True)

    # WebView events (loop thread). load_state mirrors the Qt helper's loadStarted/loadFinished.

    def on_page_started(self, e: Any) -> None:
        self._page_started(str(getattr(e, "data", "") or ""))

    def on_page_ended(self, e: Any) -> None:
        self._page_finished(str(getattr(e, "data", "") or ""))

    def on_url_change(self, e: Any) -> None:
        self._url_changed(str(getattr(e, "data", "") or ""))

    def on_resource_error(self, e: Any) -> None:
        self._resource_error(getattr(e, "data", ""))

    def on_console(self, e: Any) -> None:
        payload = parse_console_message(getattr(e, "message", ""), prefix=CONSOLE_PREFIX)
        if payload is not None:
            self.console_lines += 1
            self.deliver(payload)

    def deliver(self, payload: dict) -> None:
        """One ``GLWVB:`` answer or chunk (loop thread)."""
        request_id = str(payload.get("id") or "")
        with self._lock:
            known = request_id in self._pending
        if not known:  # late answer of a timed-out call, or not ours
            self._chunks.pop(request_id, None)
            return
        if "parts" not in payload:
            self._resolve(request_id, payload)
            return
        try:
            parts = int(payload.get("parts"))
            part = int(payload.get("part"))
        except (TypeError, ValueError):
            return
        chunk = payload.get("chunk")
        if not (1 <= parts <= MAX_CHUNKS and 0 <= part < parts and isinstance(chunk, str)):
            return
        slots = self._chunks.get(request_id)
        if slots is None or len(slots) != parts:
            slots = self._chunks[request_id] = [None] * parts
        slots[part] = chunk
        if any(item is None for item in slots):
            return
        del self._chunks[request_id]
        try:
            envelope = json.loads("".join(slots))
        except ValueError:
            envelope = None
        self._resolve(request_id, envelope if isinstance(envelope, dict) else None)


class WebViewBridge(PageDriver):
    """The flet-webview ``browser_driver`` driver of Glossarion Mobile: hidden WebView pages in
    ``page.overlay`` (the fallback where the native in-app browser is not offered)."""

    name = "flet-webview"
    page_class = WebViewPage

    def __init__(
        self,
        dispatcher: Any,
        host: Any,
        *,
        max_pages: int = MAX_PAGES,
        viewport: Tuple[int, int] = DEFAULT_VIEWPORT,
        webview_factory: Optional[Callable[[WebViewPage, str], Any]] = None,
    ) -> None:
        super().__init__(dispatcher, max_pages=max_pages, viewport=viewport)
        self.host = host
        self.webview_factory = webview_factory or _flet_webview
        # Hotfix 1 (issue 3): built WebViews kept for the next page while jobs run.
        self.warm = False  # jobs run: park used WebViews instead of unmounting them
        self.max_spares = self.max_pages
        self.config_models: Optional[Callable[[], Any]] = None  # the settings' models (prewarm check)
        self.prewarmed = 0
        self.hidden_waits = 0  # pages that waited because the app was hidden and nothing was parked
        self._spares: list = []  # parked _WebViewSlots on BLANK_URL (lock)
        self._parking: set = set()  # slots on their way to becoming spares (lock)
        self._tasks: set = set()

    # ---- plumbing ------------------------------------------------------------------------------

    def make_webview(self, slot: _WebViewSlot, url: str) -> Any:
        return self.webview_factory(slot, url)

    def app_visible(self) -> bool:
        """The app is on screen (Flet's ``page.app_visible``: False while hidden or paused), so
        Flutter builds widgets; True when the host has no page (tests, desktop)."""
        try:
            return bool(getattr(getattr(self.host, "page", None), "app_visible", True))
        except Exception:
            return True

    def app_shown(self) -> bool:
        return self.app_visible()

    # ---- built WebViews kept for the next page (hotfix 1, issue 3) ---------------------------

    def job_started(self, models: Any = ()) -> None:
        """A job started (any thread): keep used WebViews while jobs run, and build one now when
        the job (or the settings) use a browser route and none is ready."""
        self.keep_warm(True)
        wanted = list(models or ())
        if self.config_models is not None:
            try:
                wanted.extend(self.config_models() or ())
            except Exception as exc:
                log.debug("configured models unavailable: %s", exc)
        if any(uses_browser_route(model) for model in wanted):
            self.prewarm()

    def keep_warm(self, on: bool) -> None:
        """While on, closed pages park their WebViews for the next page; off unmounts the parked
        ones (any thread)."""
        with self._lock:
            self.warm = bool(on)
        if not on:
            self.post(self._drop_spares_on_loop)

    def prewarm(self) -> None:
        """Build one parked WebView now (any thread), while the app is on screen: a request
        that needs a page after the app was hidden takes it over instead of waiting."""
        self.post(self._prewarm_on_loop)

    def claim(self, page: WebViewPage) -> None:
        """Worker thread, before ``page`` loads its first URL: hand it a parked WebView (any
        viewport; one of its own size first), or return when it may build one (the app is
        visible and no WebView is about to be parked). While the app is hidden nothing is built
        (Flutter would build it only once the app shows again), so the page waits, cancellably."""
        page._check_usable()
        since: Optional[float] = None
        told = False
        while True:
            visible = self.app_visible()
            with self._lock:
                spare = next((slot for slot in self._spares if tuple(slot.viewport) == tuple(page.viewport)),
                             self._spares[0] if self._spares else None)
                if spare is not None:
                    self._spares.remove(spare)
                    page._claimed = spare
                    return
                parking = len(self._parking)
            if visible:
                if not parking:
                    return
                since = time.monotonic() if since is None else since
                if time.monotonic() - since >= PARK_WAIT:
                    return
            elif not parking and not told:  # a WebView being parked is built: wait for it quietly
                told = True
                with self._lock:
                    self.hidden_waits += 1
                log.info(HIDDEN_WAIT_MESSAGE)
            page._raise_if_cancelled()
            if page.closed:
                raise RuntimeError("the in-app browser page is closed")
            time.sleep(POLL_SECONDS)

    def _has_room(self, slot: Optional[_WebViewSlot] = None) -> bool:
        """Lock held: another WebView may be parked."""
        parking = len(self._parking) - (1 if slot is not None and slot in self._parking else 0)
        return self.warm and self.registered and len(self._spares) + parking < self.max_spares

    def expect_parking(self, slot: Optional[_WebViewSlot]) -> None:
        """A page closes (worker thread): its WebView will be parked, so a page opening next
        waits for it instead of building another (``retire_on_loop`` settles it)."""
        if slot is None or not slot.alive:
            return
        with self._lock:
            if self._has_room(slot):
                self._parking.add(slot)

    def retire_on_loop(self, slot: _WebViewSlot, *, reuse: bool, blank: bool = False) -> None:
        """A page is done with ``slot``: park it (``reuse``, while jobs run and there is room;
        ``blank``: it still shows the blank page), else unmount it."""
        with self._lock:
            keep = reuse and slot.alive and slot.webview is not None and self._has_room(slot)
            if keep:
                self._parking.add(slot)
        if not keep:
            self._unmount_slot(slot)
            return
        slot.page = None
        slot.parking = True
        if blank:
            self.parked_on_loop(slot)
            return
        task = asyncio.ensure_future(self._park_on_loop(slot))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _park_on_loop(self, slot: _WebViewSlot) -> None:
        """Navigate a used WebView to the blank page (stops the old site), then make it a spare."""
        from glossarion_mobile.ui.reader.reader_view import navigate_webview

        slot.parked = asyncio.Event()
        if slot.webview is None:  # dropped before this task ran (the queue drained)
            return
        try:
            await navigate_webview(slot.webview, BLANK_URL)
            await asyncio.wait_for(slot.parked.wait(), PARK_SETTLE)
        except asyncio.TimeoutError:
            pass  # no page-ended event for the blank page: the WebView is built, park it anyway
        except Exception as exc:
            log.info("in-app browser: a used page could not be kept for the next request: %s", str(exc)[:200])
            slot.parking = False
            self._unmount_slot(slot)
            return
        self.parked_on_loop(slot)

    def parked_on_loop(self, slot: _WebViewSlot) -> None:
        """The blank page of a parked (or warm-up) WebView finished: it becomes a spare."""
        if not slot.parking:
            return
        slot.parking = False
        if slot.parked is not None:
            slot.parked.set()
        with self._lock:
            self._parking.discard(slot)
            keep = slot.webview is not None and slot.alive and self._has_room()
            if keep:
                self._spares.append(slot)
        if not keep:
            self._unmount_slot(slot)

    def _unmount_slot(self, slot: _WebViewSlot) -> None:
        """Remove ``slot``'s WebView from the host (loop thread; idempotent)."""
        with self._lock:
            self._parking.discard(slot)
            if slot in self._spares:
                self._spares.remove(slot)
        slot.parking = False
        entry, slot.entry = slot.entry, None
        slot.webview = None
        if entry is not None:
            self.host.remove(entry)

    def _drop_spares_on_loop(self) -> None:
        with self._lock:
            if self.warm:
                return  # a job started again before this ran
            slots = list(self._spares) + list(self._parking)
        for slot in slots:
            self._unmount_slot(slot)

    def _prewarm_on_loop(self) -> None:
        if not (self.registered and self.app_visible()):
            return
        with self._lock:
            if not self.warm or self._spares or self._parking:
                return
            if any(page.slot is not None for page in self._pages.values()):
                return  # a page has a WebView: it is parked for the next one when it closes
            self.prewarmed += 1
            count = self.prewarmed
        slot = _WebViewSlot(self, self.default_viewport)
        slot.parking = slot.warmup = True
        with self._lock:
            self._parking.add(slot)
        try:
            slot.webview = self.make_webview(slot, BLANK_URL)
            slot.entry = self.host.add(slot.webview, size=slot.viewport, key=f"glwvb-warm-{count}")
        except Exception as exc:  # pages still build their own WebViews while the app is visible
            log.warning("in-app browser: no WebView could be prepared for the job: %s", exc)
            self._unmount_slot(slot)

    def status(self) -> dict:
        with self._lock:
            owners = sorted({page.owner for page in self._pages.values()})
            count = len(self._pages)
            spares, parking, warm = len(self._spares), len(self._parking), self.warm
        return {"registered": self.registered, "open_pages": count, "owners": owners,
                "opened": self.opened, "max_pages": self.max_pages, "warm": warm, "spares": spares,
                "parking": parking, "prewarmed": self.prewarmed, "hidden_waits": self.hidden_waits}


class WebViewBridgeFeature:
    """``app._install_webview_bridge``: the native in-app browser where the extension offers it
    (hotfix 3), else the flet-webview bridge where flet-webview runs."""

    @classmethod
    async def install(cls, app: Any) -> Optional[PageDriver]:
        try:
            from glossarion_mobile.services import native_webview

            driver = await native_webview.install(app)  # None: not offered, switched off, not mobile
        except Exception:
            log.exception("native in-app browser unavailable; trying flet-webview")
            driver = None
        if driver is not None:
            app.webview_bridge = driver
            log.info("WebView bridge registered: native in-app browser (authnd/, authds/, authchatgpt/, "
                     "search/gemini)")
            return driver
        page = getattr(app, "page", None)
        try:
            supported = platform_supported(page)
        except Exception as exc:
            log.warning("flet-webview check failed: %s", exc)
            supported = False
        if not supported:
            _set_current(None, UNSUPPORTED_REASON)
            app.webview_bridge = None
            log.info("WebView bridge not installed: flet-webview is unavailable on this platform")
            return None
        from glossarion_mobile.ui.components.hidden_webview import HiddenWebViewHost

        bridge = WebViewBridge(app.dispatcher, HiddenWebViewHost(page))
        bridge.suspends_hidden_pages = suspends_hidden_pages()
        bridge.register()
        store = getattr(app, "config_store", None)
        if store is not None:
            bridge.config_models = lambda: configured_models(store)  # a job warms a WebView for these
        app.webview_bridge = bridge
        log.info("WebView bridge registered (authnd/, authds/, authchatgpt/, search/gemini)")
        return bridge
