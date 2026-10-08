"""WebViewBridge (mobile-app design §5.6): hidden in-app browser pages for ``authnd/`` and
``search/gemini``.

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

Known limitations (best effort; shown in Accounts › Experimental): see :data:`LIMITATIONS`.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import threading
import time
import uuid
from typing import Any, Callable, Optional, Tuple

from glossarion_mobile.ui.reader.bridge import parse_console_message

__all__ = [
    "CHUNK_CHARS",
    "CONSOLE_PREFIX",
    "LIMITATIONS",
    "MAX_PAGES",
    "UNSUPPORTED_REASON",
    "WebViewBridge",
    "WebViewBridgeFeature",
    "WebViewPage",
    "availability",
    "current",
    "page_script",
    "platform_supported",
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
#: The hidden pages never leave the web: app / store / local links are refused.
PREVENT_LINKS = ("intent:", "market:", "file:", "content:")
CANCELLED = "stream cancelled"  # browser_driver.CANCELLED (the routes' cancel message)

UNSUPPORTED_REASON = (
    "The hidden in-app browser (flet-webview) runs on Android, iOS and macOS only, so AuthND and "
    "Gemini Free cannot run here."
)
LIMITATIONS = (
    "Experimental, best effort.\n"
    "• hCaptcha may ask for an interactive challenge, which the hidden browser cannot answer: the "
    "AuthND request then fails after the token timeout (Settings › Response handling › NIM / AuthND "
    "token helpers). Retry later or use another route.\n"
    "• The in-app browser identifies itself as a mobile WebView (its user agent cannot be changed), "
    "so NVIDIA and Google may serve other pages or extra verification than on desktop.\n"
    "• Google may show a consent or verification page instead of AI Mode; the request then fails "
    "with that message.\n"
    "• Keep Glossarion open while these requests run: Android and iOS can pause a hidden browser "
    "in the background.\n"
    f"• Every request opens its own hidden page; Token concurrency and the Gemini Free sub-chunk "
    f"concurrency limit how many run at once (never more than {MAX_PAGES})."
)

_status_lock = threading.Lock()
_current: Optional["WebViewBridge"] = None
_reason = UNSUPPORTED_REASON


def current() -> Optional["WebViewBridge"]:
    """The installed (registered) bridge, if any."""
    with _status_lock:
        return _current


def availability() -> Tuple[bool, str]:
    """``(True, "")`` while a bridge is registered, else ``(False, why)``."""
    with _status_lock:
        if _current is not None and _current.registered:
            return True, ""
        return False, _reason


def _set_current(bridge: Optional["WebViewBridge"], reason: str = "") -> None:
    global _current, _reason
    with _status_lock:
        _current = bridge
        _reason = reason or UNSUPPORTED_REASON


def platform_supported(page: Any) -> bool:
    """flet-webview renders here (the Reader's check: Android, iOS or macOS, never web)."""
    from glossarion_mobile.ui.reader.reader_view import webview_supported

    return webview_supported(page)


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


def _flet_webview(page: "WebViewPage", url: str) -> Any:
    """The real hidden WebView of ``page`` (the bridge's default factory)."""
    import flet_webview as fwv

    width, height = page.viewport
    return fwv.WebView(
        url=url,
        width=width,
        height=height,
        prevent_links=list(PREVENT_LINKS),
        on_page_started=page.on_page_started,
        on_page_ended=page.on_page_ended,
        on_url_change=page.on_url_change,
        on_web_resource_error=page.on_resource_error,
        on_console_message=page.on_console,
    )


def _short_url(url: Any) -> str:
    text = str(url or "")
    return text.split("#", 1)[0].split("?", 1)[0][:300]


class WebViewPage:
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
        from browser_driver import new_load_state

        self.bridge = bridge
        self.id = page_id
        self.owner = owner
        self.viewport = viewport
        self.cancel_check = cancel_check
        self.load_state = new_load_state()
        self.webview: Any = None  # loop thread only
        self.entry: Any = None  # host overlay entry (loop thread only)
        self.console_lines = 0
        self.closed = False
        self._url = ""
        self._cancel_reason: Optional[str] = None
        self._lock = threading.Lock()
        self._pending: dict[str, concurrent.futures.Future] = {}
        self._chunks: dict[str, list] = {}  # loop thread only

    # ---- worker-thread API (browser_driver.BrowserPage) -----------------------------------

    def load(self, url: str) -> None:
        url = str(url or "")
        if not url.lower().startswith(("https://", "http://")):
            raise ValueError(f"the in-app browser only opens web pages, not {url[:40]!r}")
        self._url = url
        self._call(self._load_on_loop, url, timeout=LOAD_TIMEOUT, what="open the page")

    def url(self) -> str:
        return self._url

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

    def wait(self, ms: int = 100) -> None:
        """Let the page run for ``ms`` milliseconds (cancellable)."""
        deadline = time.monotonic() + max(0, int(ms)) / 1000.0
        while True:
            self._raise_if_cancelled()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(min(POLL_SECONDS, remaining))

    def close(self) -> None:
        """Release the page (any thread; idempotent): fail its waits, unmount its WebView."""
        with self._lock:
            if self.closed:
                return
            self.closed = True
            pending = list(self._pending.values())
            self._pending.clear()
        self._fail(pending, "the in-app browser page was closed")
        self.bridge.forget(self)
        self.bridge.post(self._unmount_on_loop)

    # ---- cancellation (any thread; never blocks) -------------------------------------------

    def cancel(self, reason: str = CANCELLED) -> None:
        with self._lock:
            if self._cancel_reason is None:
                self._cancel_reason = str(reason or CANCELLED)
            pending = list(self._pending.values())
        self._fail(pending, self._cancel_reason)
        self.bridge.post(self._unmount_on_loop)

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
        if self.bridge.on_loop_thread():
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
        self._check_usable()
        future = self.bridge.submit(coro_fn, *args)
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

    # ---- loop side ---------------------------------------------------------------------------

    async def _load_on_loop(self, url: str) -> None:
        if self.closed or self._cancel_reason is not None:
            raise RuntimeError("the in-app browser page is closed")
        if self.webview is None:
            # flet-webview loads its ``url`` when the control is built; later loads use
            # load_request (a changed ``url`` property does not navigate a built WebView).
            self.webview = self.bridge.make_webview(self, url)
            self.entry = self.bridge.host.add(self.webview, size=self.viewport, key=f"glwvb-{self.id}")
        else:
            # the Reader's helper (one place for the rule); a failed load raises out of load()
            from glossarion_mobile.ui.reader.reader_view import navigate_webview

            await navigate_webview(self.webview, url)

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
        entry, self.entry = self.entry, None
        self.webview = None
        self._chunks.clear()
        if entry is not None:
            self.bridge.host.remove(entry)

    # WebView events (loop thread). load_state mirrors the Qt helper's loadStarted/loadFinished.

    def on_page_started(self, e: Any) -> None:
        state = self.load_state
        state["generation"] = int(state.get("generation", 0)) + 1
        state["finished_generation"] = -1
        state["finished_at"] = 0.0
        state["ok"] = False
        state["error"] = ""
        url = str(getattr(e, "data", "") or "")
        if url:
            self._url = url
        state["last_event"] = f"page started url={_short_url(url)}"

    def on_page_ended(self, e: Any) -> None:
        state = self.load_state
        url = str(getattr(e, "data", "") or "")
        if url:
            self._url = url
        state["ok"] = True
        state["finished_generation"] = int(state.get("generation", 0))
        state["finished_at"] = time.monotonic()
        state["last_event"] = f"page ended url={_short_url(url)}"

    def on_url_change(self, e: Any) -> None:
        url = str(getattr(e, "data", "") or "")
        if url:
            self._url = url

    def on_resource_error(self, e: Any) -> None:
        # Sub-resources fail too (ads, trackers), so this never marks the document failed.
        self.load_state["last_event"] = f"resource error: {str(getattr(e, 'data', '') or '')[:300]}"

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

    def _resolve(self, request_id: str, envelope: Optional[dict]) -> None:
        with self._lock:
            future = self._pending.get(request_id)
        if future is None or future.done():
            return
        try:
            future.set_result(envelope)
        except concurrent.futures.InvalidStateError:
            pass


class WebViewBridge:
    """The ``browser_driver`` driver of Glossarion Mobile: hidden WebView pages."""

    name = "flet-webview"

    def __init__(
        self,
        dispatcher: Any,
        host: Any,
        *,
        max_pages: int = MAX_PAGES,
        viewport: Tuple[int, int] = DEFAULT_VIEWPORT,
        webview_factory: Optional[Callable[[WebViewPage, str], Any]] = None,
    ) -> None:
        self.dispatcher = dispatcher
        self.host = host
        self.max_pages = max(1, int(max_pages))
        self.default_viewport = (int(viewport[0]), int(viewport[1]))
        self.webview_factory = webview_factory or _flet_webview
        self.registered = False
        self.opened = 0
        self._slots = threading.BoundedSemaphore(self.max_pages)
        self._pages: dict[str, WebViewPage] = {}
        self._lock = threading.Lock()

    # ---- browser_driver.BrowserDriver ------------------------------------------------------

    def open_page(
        self,
        *,
        owner: str,
        user_agent: Optional[str] = None,
        viewport: Optional[Tuple[int, int]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> WebViewPage:
        """A new hidden page (worker thread). ``user_agent`` is ignored: flet-webview cannot set
        one, so the page keeps the system WebView's."""
        from browser_driver import BrowserCancelled

        del user_agent
        if self.on_loop_thread():
            raise RuntimeError("the in-app browser cannot be driven from the UI thread")
        while not self._slots.acquire(timeout=0.1):
            try:
                cancelled = bool(cancel_check()) if cancel_check is not None else False
            except Exception:
                cancelled = False
            if cancelled:
                raise BrowserCancelled()
        size = viewport or self.default_viewport
        page = WebViewPage(self, uuid.uuid4().hex[:16], str(owner or ""),
                           (max(1, int(size[0])), max(1, int(size[1]))), cancel_check)
        with self._lock:
            self._pages[page.id] = page
            self.opened += 1
        return page

    def cancel_pages(self, owner: Optional[str] = None, reason: str = CANCELLED) -> None:
        """Cancel the open pages of ``owner`` (all when None); never blocks."""
        with self._lock:
            pages = [page for page in self._pages.values() if owner is None or page.owner == owner]
        for page in pages:
            page.cancel(reason)

    # ---- plumbing ------------------------------------------------------------------------------

    def forget(self, page: WebViewPage) -> None:
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

    def make_webview(self, page: WebViewPage, url: str) -> Any:
        return self.webview_factory(page, url)

    def submit(self, coro_fn: Callable[..., Any], *args: Any) -> Optional[concurrent.futures.Future]:
        return self.dispatcher.submit(coro_fn, *args)

    def post(self, fn: Callable[..., Any], *args: Any) -> bool:
        return bool(self.dispatcher.post(fn, *args))

    def on_loop_thread(self) -> bool:
        guard = getattr(self.dispatcher, "guard", None)
        return bool(guard is not None and getattr(guard, "bound", False) and guard.on_loop_thread())

    def register(self) -> "WebViewBridge":
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
        """Unregister and cancel every page (app teardown)."""
        self.unregister()
        self.cancel_pages(None, "the app is closing")

    def status(self) -> dict:
        with self._lock:
            owners = sorted({page.owner for page in self._pages.values()})
            count = len(self._pages)
        return {"registered": self.registered, "open_pages": count, "owners": owners,
                "opened": self.opened, "max_pages": self.max_pages}


class WebViewBridgeFeature:
    """``app._install_webview_bridge``: register the bridge where flet-webview runs."""

    @classmethod
    async def install(cls, app: Any) -> Optional[WebViewBridge]:
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

        bridge = WebViewBridge(app.dispatcher, HiddenWebViewHost(page)).register()
        app.webview_bridge = bridge
        log.info("WebView bridge registered (authnd/, search/gemini)")
        return bridge
