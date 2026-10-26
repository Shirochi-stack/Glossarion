"""The native in-app browser (hotfix 3): hidden pages the extension owns, for ``authchatgpt/``,
``authnd/``, ``authds/`` and ``search/gemini``.

Why. The flet-webview bridge (``services/webview_bridge.py``) keeps its hidden WebViews in
``page.overlay``. Flet 1.0.3 draws the overlay only in the top View, so every screen push / pop
moved them into another widget tree; Flutter then built a new native WebView, which loaded its
URL again without the route's page hook (an authchatgpt/ reply in progress was lost: the route
sent the message again). Flutter also builds nothing while the app is hidden, which is why the
flet bridge parks WebViews for jobs (hotfix 1).

Here the pages are ``android.webkit.WebView`` / ``WKWebView`` objects that the extension's
Kotlin / Swift code (``HiddenWebPages``) creates and owns outside the Flutter widget tree: one
1 x 1 clipped container behind the Flutter view, laid out at each page's viewport, never hidden
(so Chromium / WebKit do not treat it as invisible; on Android every page also reports its window
as visible, hotfix 4: Chromium hid the pages while the app was in the background and Blink froze
them a minute or five later, so a reply waited until the app was opened again), sharing the default cookie store and
profile with the flet-webview pages (sign-ins carry over). Screen changes never touch them, and
a page opens and loads while the app is hidden too (no parking, prewarm or claim). Python drives
them through the extension's ``GlossarionNative`` service (``flet_glossarion_native.web_pages``
is the channel contract): ``web_open`` / ``web_load`` / ``web_eval`` / ``web_title`` /
``web_close`` / ``web_reset`` / ``web_status`` and the ``web`` event (``started``, ``finished``,
``url``, ``error``, ``gone``), through ``NativeBridge`` (``services/native.py``).

Threads (the flet bridge's rules, :class:`~glossarion_mobile.services.webview_bridge.HiddenPage`):
route code calls a :class:`NativeWebPage` from its worker thread and waits in short, cancellable
slices; each page operation goes to the Flet loop through ``UiDispatcher.submit``; the loop
only awaits the service call and resolves the page's future. Script results come back as the
text of :func:`eval_script`'s JSON envelope and are decoded on the worker, never on the loop.
A page is lost only when its renderer stops (Android ``onRenderProcessGone``, iOS
``webViewWebContentProcessDidTerminate``): ``renderer_gone`` turns True, pending scripts answer
None and ``authchatgpt_auth`` sends the message again on a new page. A page that stops answering
or running its timers (frozen by the system) fails its reply after 90 s with the reason
(``browser_driver.ReplyWatch``) instead of waiting out the request timeout.

Selection (:func:`install`, from ``WebViewBridgeFeature.install``): this driver where the
extension reports the ``web_pages`` capability and ``GLOSSARION_NATIVE_WEBVIEW`` is not
``0`` / ``false`` / ``off`` / ``no`` (a developer / CI switch); otherwise the flet-webview
bridge as before. Desktop registers neither. iOS: WebKit still suspends web content soon after
the app leaves the screen, so requests pause until it returns (screen changes no longer lose the
page). Never logged: scripts, results, titles, page console output; URLs only without query.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import functools
import json
import logging
import os
import threading
import time
import uuid
from typing import Any, Awaitable, Callable, Mapping, Optional, Tuple

from glossarion_mobile.services.webview_bridge import (
    CANCELLED,
    DEFAULT_VIEWPORT,
    LOAD_TIMEOUT,
    MAX_PAGES,
    POLL_SECONDS,
    PREVENT_LINKS,
    TITLE_TIMEOUT,
    HiddenPage,
    PageDriver,
    _clean_expression,
    suspends_hidden_pages,
)

__all__ = [
    "CAPABILITY_TIMEOUT",
    "ENV_SWITCH",
    "EVAL_HEAD",
    "EVAL_TAIL",
    "LOST_MESSAGE",
    "NATIVE_NAME",
    "NativeWebDriver",
    "NativeWebPage",
    "capability",
    "disabled_by_env",
    "eval_script",
    "install",
]

log = logging.getLogger("glossarion.native_webview")

#: ``browser_driver`` driver name (``webview_bridge.limitations`` and the self-test check it).
NATIVE_NAME = "native-webview"
#: ``0`` / ``false`` / ``off`` / ``no``: use the flet-webview bridge instead (developer / CI switch).
ENV_SWITCH = "GLOSSARION_NATIVE_WEBVIEW"
#: Seconds :func:`install` waits for ``get_platform_info`` (the capability handshake).
CAPABILITY_TIMEOUT = 10.0
#: Seconds a worker waits for ``web_open`` (the extension's own bound is 20 s too).
OPEN_TIMEOUT = 20.0
#: Viewport bounds (logical pixels) sent to ``web_open``.
MAX_VIEWPORT = 4096
LOST_MESSAGE = "the in-app browser page was lost (its renderer stopped)"

# Answer codes of the channel contract (``flet_glossarion_native.web_pages.WebPageError``).
_GONE = "gone"
_NO_PAGE = "no_page"
_EXISTS = "exists"
_TIMEOUT = "timeout"
_NATIVE_ERROR = "native_error"

#: :func:`eval_script` around an expression (the extension's fakes split scripts on these).
EVAL_HEAD = (
    "(function () {\n"
    "  var envelope;\n"
    "  try {\n"
    "    var value = (\n"
)
EVAL_TAIL = (
    "\n    );\n"
    "    envelope = {ok: true, value: value === undefined ? null : value};\n"
    "  } catch (error) {\n"
    "    envelope = {ok: false, error: String(error && (error.message || error))};\n"
    "  }\n"
    "  try { return JSON.stringify(envelope); }\n"
    "  catch (error) { return JSON.stringify({ok: false, error: 'the value is not JSON'}); }\n"
    "})()"
)


def disabled_by_env() -> bool:
    """``GLOSSARION_NATIVE_WEBVIEW`` switches the native in-app browser off."""
    return os.environ.get(ENV_SWITCH, "").strip().lower() in ("0", "false", "off", "no")


def eval_script(expression: str) -> str:
    """The ``web_eval`` program for one route expression: its completion value is the JSON text
    ``{"ok": true, "value"}`` or ``{"ok": false, "error"}`` (Android's ``evaluateJavascript`` and
    iOS's ``evaluateJavaScript`` both hand that one string back). The expression sits on its own
    lines (a trailing ``//`` comment cannot swallow the parenthesis; a trailing ``;`` is dropped),
    as in ``webview_bridge.page_script``."""
    return EVAL_HEAD + _clean_expression(expression) + EVAL_TAIL


def _envelope_value(text: Any) -> Any:
    """The value of an :func:`eval_script` answer (worker thread: answers may be megabytes)."""
    if not isinstance(text, str) or not text:
        return None
    try:
        envelope = json.loads(text)
    except ValueError:
        return None
    if not isinstance(envelope, dict):
        return None
    if not envelope.get("ok"):
        log.debug("page script failed: %s", str(envelope.get("error") or "")[:300])
        return None
    return envelope.get("value")


def capability(info: Any) -> Optional[dict]:
    """The ``web_pages`` capability of a ``get_platform_info`` answer, or None (the extension's
    ``web_pages.capability``; None as well when this build's extension has no such module)."""
    try:
        from flet_glossarion_native import web_pages
    except Exception as exc:  # an extension from before hotfix 3: it has no native pages either
        log.info("native in-app browser: the extension has no web pages (%s)", type(exc).__name__)
        return None
    try:
        found = web_pages.capability(info)
    except Exception as exc:
        log.info("native in-app browser: unreadable capability (%s)", type(exc).__name__)
        return None
    return dict(found) if isinstance(found, Mapping) else None


def _failure(answer: Any) -> Tuple[str, str]:
    """``(code, message)`` of a failed answer (messages never carry page content)."""
    if not isinstance(answer, Mapping):
        return _NATIVE_ERROR, "no answer"
    return str(answer.get("error") or _NATIVE_ERROR), str(answer.get("message") or "")[:200]


def _side(value: Any) -> int:
    try:
        return max(1, min(MAX_VIEWPORT, int(value)))
    except (TypeError, ValueError):
        return 1


class NativeWebPage(HiddenPage):
    """One in-app page the extension owns: a ``browser_driver.BrowserPage``.

    The native page is created lazily by the first ``load`` / ``run_js`` (``web_open`` with the
    page's id, viewport and :data:`~glossarion_mobile.services.webview_bridge.PREVENT_LINKS`).
    The public methods are the worker-thread API; ``on_native_event`` and the ``_*_on_loop``
    coroutines run on the loop.
    """

    def __init__(
        self,
        driver: "NativeWebDriver",
        page_id: str,
        owner: str,
        viewport: Tuple[int, int],
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> None:
        super().__init__(driver, page_id, owner, viewport, cancel_check)
        #: ``browser_driver``'s optional page attribute: True once the page's document is gone for
        #: good (its renderer stopped). Scripts answer None from then on and ``load`` raises.
        self.renderer_gone = False
        self.crashed: Optional[bool] = None  # the platform's word on the renderer loss
        self.opened = False  # web_open answered (loop thread writes; workers read)
        self.open_sent = False  # web_open went out: closing must send web_close
        self.events = 0  # native events this page got
        self._close_posted = False
        self._open_lock = threading.Lock()

    # ---- worker-thread API (browser_driver.BrowserPage) -----------------------------------

    def load(self, url: str) -> None:
        url = str(url or "")
        if not url.lower().startswith(("https://", "http://")):
            raise ValueError(f"the in-app browser only opens web pages, not {url[:40]!r}")
        self._check_usable()
        if self.renderer_gone:
            raise RuntimeError(LOST_MESSAGE)
        self._url = url
        self._ensure_open()
        self._call(self._load_on_loop, url, timeout=LOAD_TIMEOUT, what="open the page")

    def title(self) -> str:
        try:
            self._check_usable()
            if self.renderer_gone or not self.opened:
                return ""
            value = self._call(self._title_on_loop, timeout=TITLE_TIMEOUT, what="read the page title")
        except Exception as exc:
            if self._cancelled() is not None:
                raise
            log.debug("page title unavailable: %s", type(exc).__name__)
            return ""
        return str(value or "")

    def run_js(self, script: str, js_timeout_ms: int = 15000) -> Any:
        """The expression's JSON value, or None when the page gave none within ``js_timeout_ms``
        (its renderer is gone, the page could not be opened). A late answer is dropped; the loop
        call is never cancelled (the extension answers every evaluation)."""
        from browser_driver import BrowserCancelled

        self._check_usable()
        if self.renderer_gone:
            return None
        try:
            self._ensure_open()
        except BrowserCancelled:
            raise
        except Exception as exc:
            log.debug("in-app page not open: %s", str(exc)[:200])
            return None
        seconds = max(0, int(js_timeout_ms)) / 1000.0
        request_id = uuid.uuid4().hex
        future: concurrent.futures.Future = concurrent.futures.Future()
        with self._lock:
            self._pending[request_id] = future
        try:
            if self.renderer_gone:  # lost while it opened: _lost resolved nothing of this call yet
                return None
            if self.driver.submit(self._eval_on_loop, request_id, eval_script(script), seconds) is None:
                return None  # the UI loop is gone (app closing)
            text = self._wait(future, time.monotonic() + seconds)
        finally:
            with self._lock:
                self._pending.pop(request_id, None)
        return _envelope_value(text)

    def close(self) -> None:
        """Release the page (any thread; idempotent; never blocks): fail its waits, then close the
        native page (``web_close``)."""
        with self._lock:
            if self.closed:
                return
            self.closed = True
            pending = list(self._pending.values())
            self._pending.clear()
        self._fail(pending, "the in-app browser page was closed")
        self.driver.forget(self)
        self._post_close()

    def cancel(self, reason: str = CANCELLED) -> None:
        """Stop (any thread; never blocks): fail every wait and close the native page at once; the
        page stays registered until the route closes it."""
        with self._lock:
            if self._cancel_reason is None:
                self._cancel_reason = str(reason or CANCELLED)
            pending = list(self._pending.values())
        self._fail(pending, self._cancel_reason)
        self._post_close()

    # ---- worker helpers ----------------------------------------------------------------------

    def _ensure_open(self) -> None:
        """Create the native page once (whichever worker asks first; the others wait for it).

        The wait is one of the page's pending calls, so a Stop or ``close()`` ends it at once; the
        loop call itself is never cancelled: an open that lands after that closes the page again
        (``_open_on_loop``), so no native page is left behind."""
        if self.opened:
            return
        while not self._open_lock.acquire(timeout=POLL_SECONDS):
            self._raise_if_cancelled()
        try:
            if self.opened:
                return
            self._check_usable()
            request_id = uuid.uuid4().hex
            future: concurrent.futures.Future = concurrent.futures.Future()
            with self._lock:
                self._pending[request_id] = future
            try:
                if self.driver.submit(self._open_on_loop, request_id) is None:
                    raise RuntimeError("the in-app browser could not open a page: the app's UI loop is not running")
                outcome = self._wait(future, time.monotonic() + OPEN_TIMEOUT)
            finally:
                with self._lock:
                    self._pending.pop(request_id, None)
            if outcome is not True:
                if self.renderer_gone:
                    raise RuntimeError(LOST_MESSAGE)
                raise RuntimeError(str(outcome or f"the in-app browser did not open a page within {OPEN_TIMEOUT:g} s"))
        finally:
            self._open_lock.release()

    def _post_close(self) -> None:
        with self._lock:
            if self._close_posted:
                return
            self._close_posted = True
        self.driver.post(self._close_on_loop)

    # ---- loop side ---------------------------------------------------------------------------

    async def _open_on_loop(self, request_id: str) -> None:
        """``web_open``; resolves the worker's wait with True or the failure's text."""
        if self.opened:
            self._resolve(request_id, True)
            return
        if self.closed or self._cancel_reason is not None:
            self._resolve(request_id, "the in-app browser page is closed")
            return
        width, height = _side(self.viewport[0]), _side(self.viewport[1])
        self.open_sent = True
        try:
            answer = await self.driver.native.call("web_open", self.id, width=width, height=height,
                                                   prevent_links=list(PREVENT_LINKS), default=None)
        except asyncio.CancelledError:  # the loop is going away; the open may still land natively
            self.driver.post(self._send_close)
            self._resolve(request_id, None)
            raise
        except Exception as exc:
            answer = {"ok": False, "error": _NATIVE_ERROR, "message": type(exc).__name__}
        if not (isinstance(answer, Mapping) and (answer.get("ok") or answer.get("error") == _EXISTS)):
            code, message = _failure(answer)
            if code not in (_TIMEOUT, _NATIVE_ERROR):
                self.open_sent = False  # refused: nothing exists natively
            elif self.closed or self._cancel_reason is not None:
                await self._send_close()
            self._resolve(request_id, f"the in-app browser could not open a page ({code}: {message})")
            return
        self.opened = True
        if self.closed or self._cancel_reason is not None:
            await self._send_close()  # closed / stopped while it opened: leave no native page behind
        self._resolve(request_id, True)

    async def _load_on_loop(self, url: str) -> None:
        if self.closed or self._cancel_reason is not None:
            raise RuntimeError("the in-app browser page is closed")
        try:
            answer = await self.driver.native.call("web_load", self.id, url, default=None)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            answer = {"ok": False, "error": _NATIVE_ERROR, "message": type(exc).__name__}
        if isinstance(answer, Mapping) and answer.get("ok"):
            return
        code, message = _failure(answer)
        if self._lost_by(code):
            raise RuntimeError(LOST_MESSAGE)
        raise RuntimeError(f"the in-app browser could not open the page ({code}: {message})")

    async def _eval_on_loop(self, request_id: str, source: str, seconds: float) -> None:
        answer: Any = None
        try:
            answer = await self.driver.native.call("web_eval", self.id, source, timeout=seconds, default=None)
        except asyncio.CancelledError:
            self._resolve(request_id, None)
            raise
        except Exception as exc:  # never the text: Flet's errors carry the call's arguments
            log.debug("in-app page script did not run: %s", type(exc).__name__)
        if isinstance(answer, Mapping):
            if answer.get("ok"):
                self._note_url(answer.get("url"))
                result = answer.get("result")
                self._resolve(request_id, result if isinstance(result, str) else None)
                return
            self._lost_by(_failure(answer)[0])
        self._resolve(request_id, None)

    async def _title_on_loop(self) -> str:
        try:
            answer = await self.driver.native.call("web_title", self.id, default=None)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.debug("in-app page title unavailable: %s", type(exc).__name__)
            return ""
        if isinstance(answer, Mapping) and answer.get("ok"):
            self._note_url(answer.get("url"))
            return str(answer.get("title") or "")
        self._lost_by(_failure(answer)[0])
        return ""

    async def _close_on_loop(self) -> None:
        if self.open_sent:
            await self._send_close()

    async def _send_close(self) -> None:
        try:
            await self.driver.native.call("web_close", self.id, default=None)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.debug("closing an in-app page failed: %s", type(exc).__name__)

    def _note_url(self, url: Any) -> None:
        """The page's URL from an answer (iOS reports same-document URL changes no other way), once
        the latest navigation finished (an answer from the old document must not undo ``load``)."""
        state = self.load_state
        settled = int(state.get("generation", 0)) > 0 and state.get("finished_generation") == state.get("generation")
        if settled and isinstance(url, str) and url.lower().startswith(("https://", "http://")):
            self._url = url

    def _lost_by(self, code: str) -> bool:
        """``gone`` (the renderer stopped) or ``no_page`` for a page that was open (the extension
        no longer has it: a reset, a new engine) means the document is lost for good."""
        if code == _GONE or (code == _NO_PAGE and self.opened):
            self._lost(None)
            return True
        return False

    # ---- native events (loop thread) ---------------------------------------------------------

    def on_native_event(self, payload: Mapping[str, Any]) -> None:
        """A ``web`` event of this page: the flet bridge's ``load_state`` rules (``finished`` always
        sets ``ok``; a failure the platform reported only fills ``load_state["error"]``)."""
        self.events += 1
        kind = str(payload.get("type") or "")
        url = payload.get("url")
        url = url if isinstance(url, str) else ""
        if kind == "started":
            self._page_started(url)
        elif kind == "finished":
            error = ""
            if payload.get("ok") is False:
                error = str(payload.get("error") or "the page did not finish loading")
            self._page_finished(url, error=error)
        elif kind == "url":
            self._url_changed(url)
        elif kind == "error":
            code = payload.get("code")
            self._resource_error(payload.get("error") or f"error {code}", main_frame=payload.get("main_frame") is True)
        elif kind == "gone":
            crashed = payload.get("crashed")
            self._lost(None if crashed is None else bool(crashed))

    def _lost(self, crashed: Optional[bool]) -> None:
        """The renderer stopped: the document is gone for good. Pending scripts answer None (not
        cancelled: a crash is not a Stop)."""
        if self.renderer_gone:
            if crashed is not None and self.crashed is None:  # the gone event after an answer that said so
                self.crashed = crashed
                self.load_state["last_event"] = "renderer gone (crashed)" if crashed else "renderer gone"
            return
        self.renderer_gone = True
        self.crashed = crashed
        self.load_state["last_event"] = "renderer gone (crashed)" if crashed else "renderer gone"
        with self._lock:
            request_ids = list(self._pending)
        for request_id in request_ids:
            self._resolve(request_id, None)
        page_lost = getattr(self.driver, "page_lost", None)
        if callable(page_lost):
            page_lost(self)


class NativeWebDriver(PageDriver):
    """The ``browser_driver`` driver over the extension's in-app pages (hotfix 3).

    ``native`` is the app's ``NativeBridge``; its ``web`` events reach :meth:`on_web_event` while
    the driver is registered. At most ``max_pages`` pages are open at once (route concurrency;
    the extension's own cap is a leak guard). ``job_started`` / ``keep_warm`` are
    :class:`PageDriver`'s no-ops: nothing is parked or prepared for jobs.
    """

    name = NATIVE_NAME
    page_class = NativeWebPage

    def __init__(
        self,
        dispatcher: Any,
        native: Any,
        *,
        capability: Optional[Mapping[str, Any]] = None,
        max_pages: int = MAX_PAGES,
        viewport: Tuple[int, int] = DEFAULT_VIEWPORT,
    ) -> None:
        super().__init__(dispatcher, max_pages=max_pages, viewport=viewport)
        self.native = native
        self.capability = dict(capability or {})
        self.events = 0
        self.renderer_losses = 0
        self.resets = 0
        #: ``async () -> dict``: push and pop a screen and open a sheet (the self-test, set by
        #: :func:`install`; ``diagnostics.webview_check.screen_churn``).
        self.screen_churn: Optional[Callable[[], Awaitable[dict]]] = None
        #: ``() -> bool``: Glossarion is on screen (Flet's ``page.app_visible``; set by :func:`install`).
        #: The pages open, load and run whatever the app shows; the background self-test
        #: (``diagnostics.webview_check.run_background``) and, on iOS only, :meth:`pages_suspended` read it.
        self.app_on_screen: Optional[Callable[[], bool]] = None
        self._listening = False

    def app_shown(self) -> bool:
        probe = self.app_on_screen
        return True if probe is None else bool(probe())

    # ---- events (loop thread) ----------------------------------------------------------------

    def on_web_event(self, payload: Any) -> None:
        """``NativeBridge`` listener ``web``: hand the event to its page (constant time)."""
        if not isinstance(payload, Mapping):
            return
        page_id = str(payload.get("page_id") or "")
        with self._lock:
            self.events += 1
            page = self._pages.get(page_id)
        if page is None:
            return  # a closed page's late event, or another session's
        try:
            page.on_native_event(payload)
        except Exception:
            log.exception("in-app page event failed")

    def page_lost(self, page: NativeWebPage) -> None:
        with self._lock:
            self.renderer_losses += 1
        log.info("in-app browser: the renderer of a %s page stopped (%s); its scripts answer None from now on",
                 page.owner or "browser", "crashed" if page.crashed else "killed or unknown")

    # ---- calls on the loop ------------------------------------------------------------------------

    async def reset(self) -> int:
        """Close every native page of this engine (``web_reset``); the number closed."""
        try:
            answer = await self.native.call("web_reset", default=None)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.debug("resetting the in-app pages failed: %s", type(exc).__name__)
            return 0
        self.resets += 1
        if isinstance(answer, Mapping) and answer.get("ok"):
            try:
                return int(answer.get("closed") or 0)
            except (TypeError, ValueError):
                return 0
        return 0

    async def native_status(self) -> dict:
        """The extension's view of its pages (``web_status``; the self-test, diagnostics)."""
        try:
            answer = await self.native.call("web_status", default=None)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.debug("in-app page status unavailable: %s", type(exc).__name__)
            return {}
        return dict(answer) if isinstance(answer, Mapping) else {}

    # ---- registration ---------------------------------------------------------------------------

    def register(self) -> "NativeWebDriver":
        add_listener = getattr(self.native, "add_listener", None)
        if not self._listening and callable(add_listener):
            add_listener("web", self.on_web_event)
            self._listening = True
        super().register()
        return self

    def unregister(self, reason: str = "The in-app browser was shut down.") -> None:
        super().unregister(reason)
        if self._listening:
            remove_listener = getattr(self.native, "remove_listener", None)
            if callable(remove_listener):
                remove_listener("web", self.on_web_event)
            self._listening = False

    def shutdown(self) -> None:
        """Unregister, cancel every page (each closes its native page), then ``web_reset``."""
        super().shutdown()
        self.post(self.reset)

    def status(self) -> dict:
        status = super().status()
        with self._lock:
            gone = sum(1 for page in self._pages.values() if getattr(page, "renderer_gone", False))
            events, losses, resets = self.events, self.renderer_losses, self.resets
        status.update({"capability": dict(self.capability), "renderer_gone": gone, "renderer_losses": losses,
                       "events": events, "resets": resets})
        return status


def _app_on_screen(app: Any) -> bool:
    """``NativeWebDriver.app_on_screen``: False only while Flet says the app is hidden or paused."""
    from glossarion_mobile.ui.foreground import app_visible

    return app_visible(getattr(app, "page", None))


async def install(app: Any) -> Optional[NativeWebDriver]:
    """Register the native in-app browser for ``app`` (loop thread), or None: switched off by
    :data:`ENV_SWITCH`, no native extension (desktop, host tests, ``flet run``), no answer to the
    capability handshake within :data:`CAPABILITY_TIMEOUT`, or a build without the capability.
    ``WebViewBridgeFeature.install`` then installs the flet-webview bridge as before."""
    if disabled_by_env():
        log.info("native in-app browser switched off by %s; using flet-webview", ENV_SWITCH)
        return None
    native = getattr(app, "native", None)
    dispatcher = getattr(app, "dispatcher", None)
    if native is None or dispatcher is None or getattr(native, "is_stub", True):
        return None
    platform_info = getattr(native, "platform_info", None)
    if not callable(platform_info):
        return None
    try:
        info = await asyncio.wait_for(platform_info(), CAPABILITY_TIMEOUT)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # asyncio.TimeoutError included
        log.info("native in-app browser: no platform answer (%s); using flet-webview", type(exc).__name__)
        return None
    found = capability(info)
    if found is None:
        log.info("native in-app browser not offered by this build; using flet-webview")
        return None
    from glossarion_mobile.diagnostics import webview_check

    try:
        native_cap = int(found.get("max_pages") or MAX_PAGES)
    except (TypeError, ValueError):
        native_cap = MAX_PAGES
    driver = NativeWebDriver(dispatcher, native, capability=found, max_pages=max(1, min(MAX_PAGES, native_cap)))
    driver.screen_churn = functools.partial(webview_check.screen_churn, app)
    driver.app_on_screen = functools.partial(_app_on_screen, app)
    # iOS: WebKit suspends the pages while Glossarion is off screen; replies wait for the app (hotfix 4).
    driver.suspends_hidden_pages = suspends_hidden_pages()
    # Pages an earlier Python session left in this engine go first, before a route can open one
    # (posted, not awaited: start-up does not wait for it; the loop runs it ahead of any page call).
    driver.post(driver.reset)
    try:
        driver.register()
    except Exception:
        driver.unregister("The native in-app browser could not start.")
        raise
    log.info("native in-app browser ready (%s, contract version %s)",
             str(found.get("provider") or "platform WebView")[:120], found.get("version"))
    return driver
