"""In-app browser pages (hotfix 3): the contract shared by the Python API, the native code and test fakes.

Routes that drive a web page on the phone (authchatgpt/ chatgpt.com web chat, authnd/, authds/,
search/gemini) need a browser page that survives screen changes and keeps running while Glossarion
is in the background. Hidden ``flet-webview`` controls cannot: Flet 1.0.3 draws ``page.overlay`` only in
the top View, so every screen push / pop moves the WebView into another widget tree and Flutter builds a
new native WebView, which reloads without the page's hook. The extension therefore creates and owns
the pages itself, outside the Flutter widget tree:

* Android (``HiddenWebPages.kt``): ``android.webkit.WebView``s in one 1 x 1 px clipped ``FrameLayout``
  added at index 0 of the Activity's ``android.R.id.content`` (behind the opaque FlutterView), each laid
  out at the page viewport (dp). Never ``GONE`` / ``INVISIBLE``, alpha 1, never ``onPause()`` /
  ``pauseTimers()``, and every page reports its window as visible (hotfix 4: Chromium hides a page whose
  window is not, as in the background, and Blink freezes a hidden page a minute or five later), so
  Chromium keeps the pages running in the background too. Process-wide ``CookieManager`` and the
  default profile: logins made in flet-webview pages carry over.
* iOS (``HiddenWebPages.swift``): ``WKWebView``s in a 1 x 1 pt clipped ``UIView`` at index 0 of the key
  window, ``WKWebsiteDataStore.default()`` (cookies and localStorage shared with flet-webview).
  WebKit still suspends web content soon after the app leaves the screen: requests pause until
  Glossarion is back (screen changes no longer lose the page).

This module has no Flet import: the app, the fakes and the tests use it on any host.

Handshake
---------
``get_platform_info()["web_pages"] == {"version": 1, "max_pages": 8, "provider": str | None}`` when the
native side has the pages (Android leaves the key out when API >= 26 reports no WebView provider).
:func:`capability` reads it.

Methods (Python -> Dart -> Kotlin / Swift; the same names on every layer)
---------------------------------------------------------------------------
Every answer is a map ``{"ok": bool, "error": code | None, "message": str | None, ...}``; the platform
code never raises a channel error for these methods. Off-device the answer is ``unavailable``.

``web_open``   ``page_id`` (Python-made, 16 hex), ``width`` / ``height`` (logical px = dp / pt, 1..4096),
               ``prevent_links`` (URL prefixes the page may not navigate to; Android only, WebKit never
               hands ``intent:`` / ``market:`` links to other apps) -> ``{page_id, attached}``.
               Errors: ``bad_args``, ``exists``, ``too_many`` (more than :data:`NATIVE_MAX_PAGES`),
               ``unavailable`` (no WebView could be made), ``native_error``.
``web_load``   ``page_id``, ``url`` (http / https only) -> ``{}`` once the navigation started.
``web_eval``   ``page_id``, ``script`` -> ``{result: str | None, url}``: the program's completion value
               when it is a string, else None. Python wraps every expression in an IIFE that returns
               ``JSON.stringify({ok, value})`` (or ``{ok: false, error}``), so both platforms hand back
               exactly one string and the worker thread ``json.loads`` it; there is no chunking.
               Errors: ``no_page``, ``gone``, ``closed`` (the page was closed while it ran).
``web_title``  ``page_id`` -> ``{title, url}``.
``web_close``  ``page_id`` -> ``{closed: bool}`` (False: it was not open; idempotent).
``web_reset``  -> ``{closed: int}``: every page of this plugin instance (the app sends it at start-up and
               at shutdown).
``web_status`` -> ``{pages: [{page_id, url, gone, width, height, pending}], attached, ...}``.

Error codes: :class:`WebPageError` (``timeout`` is Python's own: the Flet call did not answer in time).
Messages never carry page content; exceptions are reported by their type name only (Flet's
``TimeoutError`` text embeds the call arguments, i.e. the script and the prompt).

Events (platform -> Dart ``triggerEvent('web')`` -> Python ``on_web`` :class:`WebPageEvent`)
--------------------------------------------------------------------------------------------
Keys :data:`WEB_EVENT_KEYS`; ``type`` is one of :data:`WEB_EVENT_TYPES`:

``started``   a main-frame navigation began (Android ``onPageStarted``, iOS ``didStartProvisionalNavigation``)
``finished``  it ended (``onPageFinished`` / ``didFinish``; iOS also after a real failure) with ``ok``
              (False after a main-frame network error) and that ``error``
``url``       the document URL changed (``doUpdateVisitedHistory`` incl. pushState / ``didCommit``)
``error``     a load error with ``code``, ``error`` and ``main_frame`` (Android forwards at most
              :data:`MAX_SUBRESOURCE_ERRORS` sub-resource errors per document)
``gone``      the page's renderer stopped (``onRenderProcessGone`` /
              ``webViewWebContentProcessDidTerminate``), ``crashed`` (Android ``didCrash()``; null on
              iOS, which gives no reason); the page id stays registered and answers ``gone`` until
              ``web_close``

Console output, page content, scripts and results are never sent as events or logged.

Threading
---------
Route code runs on worker threads; the app hops to the Flet loop (``UiDispatcher.submit``) for every
call. Kotlin and Swift run every page operation, WebView callback and ``channel.invokeMethod`` on the
main thread, answer each channel result exactly once (a pending ``web_eval`` gets ``closed`` / ``gone``
when its page is destroyed) and keep the page registry per plugin instance: flutter_foreground_task's
headless engine gets its own instance, which never opens a page.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Mapping, Optional
from urllib.parse import urlsplit

__all__ = [
    "DEFAULT_EVAL_TIMEOUT",
    "EVAL_SLACK",
    "LOAD_TIMEOUT",
    "MAX_SIDE",
    "MAX_SUBRESOURCE_ERRORS",
    "NATIVE_MAX_PAGES",
    "OPEN_TIMEOUT",
    "QUERY_TIMEOUT",
    "WEB_EVENT_KEYS",
    "WEB_EVENT_TYPES",
    "WEB_METHODS",
    "WEB_PAGES_VERSION",
    "WebPageError",
    "capability",
    "clamp_side",
    "error_result",
    "is_http_url",
    "normalize_result",
]

#: Version of this contract (``get_platform_info()["web_pages"]["version"]``).
WEB_PAGES_VERSION = 1
#: Pages one plugin instance keeps at most (a leak guard; the app opens at most 4 at a time).
NATIVE_MAX_PAGES = 8
#: Android forwards at most this many sub-resource ``error`` events per document (main-frame errors always).
MAX_SUBRESOURCE_ERRORS = 8
#: Largest page side in logical pixels.
MAX_SIDE = 4096

WEB_METHODS: tuple[str, ...] = (
    "web_open", "web_load", "web_eval", "web_title", "web_close", "web_reset", "web_status",
)
WEB_EVENT_TYPES: tuple[str, ...] = ("started", "finished", "url", "error", "gone")
WEB_EVENT_KEYS: tuple[str, ...] = ("page_id", "type", "url", "ok", "error", "code", "main_frame", "crashed")

#: Seconds Python waits for ``web_open`` / ``web_load`` (creating a WebView can take a moment the first time).
OPEN_TIMEOUT = 20.0
LOAD_TIMEOUT = 20.0
#: ``web_title``, ``web_close``, ``web_reset``, ``web_status``.
QUERY_TIMEOUT = 10.0
#: ``web_eval`` waits the script's own timeout plus this much for the answer to travel back.
EVAL_SLACK = 5.0
DEFAULT_EVAL_TIMEOUT = 30.0


class WebPageError(str, Enum):
    """``error`` codes of in-app page answers."""

    UNAVAILABLE = "unavailable"
    """No native pages here (desktop, web, companion app, no WebView provider, disabled)."""
    BAD_ARGS = "bad_args"
    EXISTS = "exists"
    """``web_open`` with a ``page_id`` that is already open."""
    TOO_MANY = "too_many"
    """More than :data:`NATIVE_MAX_PAGES` pages are open."""
    NO_PAGE = "no_page"
    """No page with that ``page_id`` (never opened, closed, or reset)."""
    GONE = "gone"
    """The page's renderer stopped (crash or killed by the system); open a new page."""
    CLOSED = "closed"
    """The page was closed while this call was waiting."""
    NATIVE_ERROR = "native_error"
    """Any other failure; ``message`` is an exception type name."""
    TIMEOUT = "timeout"
    """Python stopped waiting for the answer (Python only)."""


_ERROR_VALUES = frozenset(e.value for e in WebPageError)


def error_result(code: Any, message: Optional[str] = None, **extra: Any) -> dict:
    """A failure answer in the shared shape (unknown codes become ``native_error``)."""
    value = getattr(code, "value", code)
    if value not in _ERROR_VALUES:
        value = WebPageError.NATIVE_ERROR.value
    out: dict[str, Any] = {"ok": False, "error": value, "message": message}
    out.update(extra)
    return out


def normalize_result(raw: Any) -> dict:
    """Coerce a native answer into the shared shape (never raises)."""
    if not isinstance(raw, Mapping):
        return error_result(WebPageError.NATIVE_ERROR, "Unexpected answer from the native side")
    out = {str(k): v for k, v in raw.items()}
    if out.get("ok") is True:
        out["error"] = None
        out.setdefault("message", None)
        return out
    code = out.get("error")
    if code not in _ERROR_VALUES:
        code = WebPageError.NATIVE_ERROR.value
    out["ok"] = False
    out["error"] = code
    message = out.get("message")
    out["message"] = message if isinstance(message, str) else None
    return out


def capability(info: Any) -> Optional[dict]:
    """``info["web_pages"]`` of a ``get_platform_info()`` answer when this contract's pages exist there.

    None unless the answer came from the native side (``native`` is True) and announces version >= 1.
    """
    if not isinstance(info, Mapping) or info.get("native") is not True:
        return None
    pages = info.get("web_pages")
    if not isinstance(pages, Mapping):
        return None
    version = pages.get("version")
    if isinstance(version, bool):
        return None
    try:
        number = int(version)
    except (TypeError, ValueError):
        return None
    if number < 1:
        return None
    out = {str(k): v for k, v in pages.items()}
    out["version"] = number
    try:
        out["max_pages"] = max(1, int(out.get("max_pages") or NATIVE_MAX_PAGES))
    except (TypeError, ValueError):
        out["max_pages"] = NATIVE_MAX_PAGES
    return out


def clamp_side(value: Any, default: int) -> int:
    """A page side in logical pixels, 1..:data:`MAX_SIDE` (``default`` for anything that is not a number)."""
    if isinstance(value, bool):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number:  # NaN
        return default
    return int(min(max(round(number), 1), MAX_SIDE))


def is_http_url(url: Any) -> bool:
    """True for an ``http://`` / ``https://`` URL with a host (the only pages ``web_load`` opens)."""
    if not isinstance(url, str) or not url:
        return False
    try:
        parts = urlsplit(url)
    except ValueError:
        return False
    return parts.scheme.lower() in ("http", "https") and bool(parts.netloc)
