"""ChatGPT web chat (chatgpt.com) in a browser page: no API key, login optional.

Works without an account: chatgpt.com answers logged-out visitors. The ChatGPT login is
optional and is the same one ``authgpt/`` uses (``authgpt_auth`` default token store): when
it is present, the page's backend calls carry that OAuth access token, so the chat runs on
the user's account and plan. The page itself produces the anti-abuse proofs (sentinel,
Turnstile, proof of work) for every request; Glossarion only replaces the prompt, the model
and the thinking effort of the request the page sends, and reads the streamed answer.

Models are named like chatgpt.com's own picker: ``authchatgpt/<version>-<effort>`` such as
``gpt-6-instant``, ``gpt-6-medium``, ``gpt-5.6-sol-extra-high`` or ``gpt-6-pro``. The page
resolves the name against the account's live picker (``/models`` ``versions`` and their
``intelligence_presets``: model slug + thinking effort). ``auto`` and plain chatgpt.com model
slugs work too.

Desktop runs the page in a dedicated headless Chrome/Edge profile driven over DevTools
(``cdp_browser``). Glossarion Mobile runs it in a hidden in-app WebView (``browser_driver``),
like ``authnd/`` and ``authds/``. Every completion opens a new page and a new chat;
logged-in chats are temporary chats, so translations do not fill the account's history.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import threading
import time
import uuid
from contextlib import contextmanager, nullcontext

import browser_driver
import cdp_browser
import mobile_runtime
from authds_auth import build_prompt as _build_prompt

BASE_URL = "https://chatgpt.com"
PREFIX = "authchatgpt/"
DEFAULT_MODEL = "auto"
#: browser_driver page owner (Glossarion Mobile's in-app WebView): cancel_stream() closes only these pages.
BROWSER_OWNER = "authchatgpt"
# chatgpt.com serves logged-out visitors a different web app whose requests cannot carry
# the OAuth token; any session cookie makes it serve the regular app (still logged out
# until the token is added to its backend calls).
_APP_COOKIE = {"name": "__Secure-next-auth.session-token", "value": "glossarion",
               "url": BASE_URL + "/", "httpOnly": True, "secure": True}
# Scripts for the mobile WebView bridge must be single expressions: it evaluates them as
# ``var value = (<script>);`` (webview_bridge.page_script), so no statement lists.
_APP_COOKIE_JS = ("(document.cookie = '__Secure-next-auth.session-token=glossarion; Secure; Path=/; "
                  "SameSite=Lax', true)")
_PLANS = {"free": "Free", "go": "Go", "plus": "Plus", "pro": "Pro", "prolite": "Pro", "team": "Team",
          "business": "Business", "enterprise": "Enterprise", "edu": "Edu", "guest": "Guest"}
_cancel = threading.Event()
_announced = {}  # run id -> login lines already logged in that run
_rejected = set()  # access tokens chatgpt.com refused: later requests go logged out at once
_announce_lock = threading.Lock()


class AuthChatGPTError(RuntimeError):
    def __init__(self, message, error_type="api_error"):
        super().__init__(message)
        self.error_type = error_type


def cancel_stream():
    _cancel.set()
    browser_driver.cancel_pages(BROWSER_OWNER)


def reset_cancel():
    _cancel.clear()


def _check(cancel_check=None):
    if _cancel.is_set() or (cancel_check and cancel_check()):
        raise AuthChatGPTError("AuthChatGPT: translation stopped by user", "cancelled")


def profile_dir():
    return Path(os.environ.get("AUTHCHATGPT_PROFILE_DIR") or
                (Path.home() / ".glossarion" / "authchatgpt_browser"))


def resolve_model(model):
    """The picker name of ``authchatgpt/<name>`` (``auto`` when empty); the page resolves it."""
    name = re.sub(r"^authchatgpt/", "", str(model).strip(), flags=re.IGNORECASE).strip().lower()
    if not name:
        return DEFAULT_MODEL
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", name):
        raise AuthChatGPTError(f"Unknown AuthChatGPT model '{name}'. Use a name from chatgpt.com's picker "
                               "such as gpt-6-instant, gpt-6-medium or gpt-6-pro, or auto.", "config_error")
    return name


def build_prompt(messages, images=None):
    return _build_prompt(messages, AuthChatGPTError, "AuthChatGPT", images=images)


#: logged-out chatgpt.com cannot attach files ("Log in to use")
IMAGES_NEED_LOGIN = ("AuthChatGPT needs a ChatGPT login to send images (logged-out ChatGPT cannot attach files). "
                     "Click ChatGPT Login.")


# ---------------------------------------------------------------------------
# Optional login: the authgpt/ ChatGPT token
# ---------------------------------------------------------------------------

def _token_store():
    import authgpt_auth
    return authgpt_auth.get_default_store()


def has_login():
    """Whether the authgpt/ ChatGPT login is saved (it may still need a refresh)."""
    try:
        return _token_store().has_tokens
    except Exception:
        return False


def login_identity(token):
    """(email, plan name) carried by a ChatGPT access token; blanks when unknown."""
    try:
        import authgpt_auth
        claims = authgpt_auth._decode_jwt_payload(token) or {}
    except Exception:
        claims = {}
    email = (claims.get("https://api.openai.com/profile") or {}).get("email") or claims.get("email") or ""
    plan = str((claims.get("https://api.openai.com/auth") or {}).get("chatgpt_plan_type") or "")
    return email, _PLANS.get(plan.lower(), plan.replace("_", " ").title())


def _announce(key, message, log_fn):
    """Log ``message`` once per Glossarion run (``GLOSSARION_RUN_ID``) and login state."""
    run = os.environ.get("GLOSSARION_RUN_ID", "")
    with _announce_lock:
        seen = _announced.setdefault(run, set())
        if key in seen:
            return
        seen.add(key)
    log_fn(message)


def _announce_logged_in(token, log_fn):
    email, plan = login_identity(token)
    _announce(f"in:{email}", f"🔐 AuthChatGPT: Logged in to ChatGPT as {email or 'your ChatGPT account'}"
              + (f" · {plan} plan" if plan else "") + ".", log_fn)


def _announce_logged_out(log_fn):
    _announce("out", "🔓 AuthChatGPT: No ChatGPT login found. Using ChatGPT logged out (guest models and "
              "limits); click ChatGPT Login to use your account.", log_fn)


def _reject(token, reason, log_fn):
    _rejected.add(token)
    _announce("rejected", f"🔓 AuthChatGPT: ChatGPT did not accept the saved login ({reason.rstrip('.')}). Using "
              "ChatGPT logged out; click ChatGPT Login to sign in again.", log_fn)


def _access_token(log_fn):
    """The saved ChatGPT access token, refreshed when due; None means logged out. Never
    opens a login: without a usable token the chat runs logged out."""
    try:
        store = _token_store()
        if not store.has_tokens:
            _announce_logged_out(log_fn)
            return None
        token = store.get_valid_access_token(auto_login=False)
    except Exception:
        _announce("unusable", "🔓 AuthChatGPT: The saved ChatGPT login could not be refreshed. Using ChatGPT "
                  "logged out; click ChatGPT Login to sign in again.", log_fn)
        return None
    if token in _rejected:
        _announce("rejected", "🔓 AuthChatGPT: ChatGPT did not accept the saved login. Using ChatGPT logged out; "
                  "click ChatGPT Login to sign in again.", log_fn)
        return None
    return token


def _refreshed_token(rejected):
    """A new access token after chatgpt.com rejected ``rejected``, or None.

    Runs under the token store's own lock: concurrent requests (batch translation) that all
    got the rejection refresh once, and the others take that new token. Two refreshes with
    the same refresh token would rotate it twice and lose the login."""
    try:
        store = _token_store()
        with getattr(store, "_lock", None) or nullcontext():
            tokens = store.load_tokens() or {}
            if tokens.get("access_token") and tokens["access_token"] != rejected:
                return tokens["access_token"]  # another request already refreshed it
            refreshed = store._try_refresh(tokens) if tokens else None
            return (refreshed or {}).get("access_token")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Browser: desktop Chrome/Edge over DevTools
# ---------------------------------------------------------------------------

class _Page(cdp_browser.CdpPage):
    command_failed = "AuthChatGPT browser command failed."
    command_timed_out = "AuthChatGPT browser command timed out."
    script_failed = "AuthChatGPT browser script failed. Reload ChatGPT and try again."

    def check(self, cancel_check):
        _check(cancel_check)

    def error(self, message, error_type):
        return AuthChatGPTError(message, error_type)


def _browser_binary():
    found = cdp_browser.find_browser_binary(
        "AUTHCHATGPT_BROWSER_BINARY", lambda message: AuthChatGPTError(message, "config_error"))
    if found:
        return found
    raise AuthChatGPTError("AuthChatGPT needs Google Chrome or Microsoft Edge installed. "
                           "You can also set AUTHCHATGPT_BROWSER_BINARY.", "config_error")


@contextmanager
def _launch_browser(*, visible=False, cancel_check=None):
    if not mobile_runtime.subprocesses_available():
        raise AuthChatGPTError("AuthChatGPT needs the in-app browser here, or the desktop app with Chrome/Edge.",
                               "config_error")
    with cdp_browser.launch(
            binary=_browser_binary(), profile=profile_dir(), start_url="about:blank",
            ready=lambda page, check: None, page_factory=lambda url, check: _Page(url, check),
            check=_check, error=AuthChatGPTError,
            closed_message="AuthChatGPT browser closed. Try again.",
            connect_message="AuthChatGPT could not connect to its browser.", visible=visible,
            extra_args=("--disable-blink-features=AutomationControlled", "--window-size=1400,1000"),
            cancel_check=cancel_check) as page:
        yield page


_READY = ("location.origin === 'https://chatgpt.com' && !!window.__glossarionGPT && "
          "!!document.querySelector('#prompt-textarea')")
_CHALLENGE = "document.title === 'Just a moment...'"
_NOT_LOADED = "ChatGPT page did not finish loading. Check your connection and try again."


class _CookieReset:
    """Desktop: every tab of the route's browser shares chatgpt.com's cookies.

    A tab stuck on Cloudflare's check starts them over (clear + reload). Concurrent tabs
    (batch translation) must not each clear them again, wiping the clearance another tab just
    earned, nor clear them while a tab is sending its message (its sentinel / Turnstile
    requests need them). So resets run one at a time, wait for sending tabs, and a tab whose
    wait began before another tab's reset only reloads."""

    def __init__(self):
        self.cond = threading.Condition()
        self.generation = 0  # resets done
        self.resetting = False
        self.sending = 0  # tabs between composer and conversation request

    def _wait(self, cancel_check):
        self.cond.wait(.2)
        _check(cancel_check)

    @contextmanager
    def sending_message(self, cancel_check=None):
        with self.cond:
            while self.resetting:
                self._wait(cancel_check)
            self.sending += 1
        try:
            yield
        finally:
            with self.cond:
                self.sending -= 1
                self.cond.notify_all()

    def reset(self, page, seen, cancel_check=None, drain=45):
        """Start the cookies over unless another tab did since generation ``seen``; then reload."""
        with self.cond:
            while self.resetting:
                self._wait(cancel_check)
            fresh = self.generation == seen
            if fresh:
                deadline = time.monotonic() + drain
                while self.sending and time.monotonic() < deadline:
                    self._wait(cancel_check)
                self.resetting = True
        try:
            if fresh:
                page.call("Network.clearBrowserCookies")
                page.call("Network.setCookie", _APP_COOKIE)
            page.call("Page.reload")
        finally:
            if fresh:
                with self.cond:
                    self.resetting = False
                    self.generation += 1
                    self.cond.notify_all()


_COOKIES = _CookieReset()


def _tab_ready(page, cancel_check):
    seen = _COOKIES.generation
    try:
        cdp_browser.wait_until(page, _READY, timeout=40, check=_check, error=AuthChatGPTError,
                               cancel_check=cancel_check, message=_NOT_LOADED)
        return
    except AuthChatGPTError as exc:
        if exc.error_type == "cancelled" or not page.evaluate(_CHALLENGE):
            raise
    # A Cloudflare check that never clears is tied to the profile's cookies: start them over.
    _COOKIES.reset(page, seen, cancel_check)
    cdp_browser.wait_until(page, _READY, timeout=45, check=_check, error=AuthChatGPTError,
                           cancel_check=cancel_check, message=_NOT_LOADED)


_LEASES = cdp_browser.BrowserLeases(
    launch=lambda **kwargs: _launch_browser(**kwargs), page_factory=lambda url, check: _Page(url, check),
    check=_check, url=BASE_URL + "/", ready=_tab_ready)


def _prepare(config):
    def prepare(page):
        page.call("Page.enable")
        # Cloudflare stops the headless user agent; the browser is otherwise unchanged.
        agent = page.call("Browser.getVersion").get("userAgent", "").replace("HeadlessChrome", "Chrome")
        if agent:
            page.call("Emulation.setUserAgentOverride", {"userAgent": agent})
        page.call("Network.setCookie", _APP_COOKIE)
        page.call("Page.addScriptToEvaluateOnNewDocument", {"source": _hook(config)})
    return prepare


# ---------------------------------------------------------------------------
# Browser: Glossarion Mobile's hidden in-app WebView
# ---------------------------------------------------------------------------

_FRAME_ID = "glossarion-chatgpt"
_FRAME = f"document.getElementById('{_FRAME_ID}')"
# The top page only hosts the chat iframe: a small same-origin document, not a second app.
_HOST_URL = BASE_URL + "/robots.txt"


class _DriverPage:
    """A ``browser_driver`` page with the CDP page's ``evaluate`` (mobile).

    Once the chat iframe exists, scripts run against its window and document."""

    driver = True

    def __init__(self, page):
        self.page = page
        self.framed = False

    def evaluate(self, expression):
        if self.framed:
            expression = (f"((window, document) => ({expression}))"
                          f"({_FRAME}.contentWindow, {_FRAME}.contentWindow.document)")
        try:
            return self.page.run_js(expression, 30000)
        except browser_driver.BrowserCancelled as exc:
            raise AuthChatGPTError("AuthChatGPT: translation stopped by user", "cancelled") from exc
        except Exception as exc:
            raise AuthChatGPTError("AuthChatGPT browser script failed. Try again.", "browser_error") from exc

    def hook_lost(self):
        """True when the hooked chat window is gone (a new document in the page or the frame).

        A None from ``evaluate`` does not say so here: the in-app browser also gives None when
        the page's answer did not arrive in time or the call could not be posted. This asks the
        top page itself (never through the frame, which may be the thing that is gone);
        None means no answer again, which is not proof either. A page whose renderer stopped
        (``renderer_gone``, the native in-app pages) lost its document for sure."""
        if _renderer_gone(self):
            return True
        script = (f"(() => {{const f = {_FRAME};"
                  f" if (!f) return {'true' if self.framed else '!window.__glossarionGPT'};"
                  " try { return !f.contentWindow.__glossarionGPT; } catch (e) { return true; }})()")
        try:
            return self.page.run_js(script, 10000)
        except browser_driver.BrowserCancelled as exc:
            raise AuthChatGPTError("AuthChatGPT: translation stopped by user", "cancelled") from exc
        except Exception:
            return None

    def content_suspended(self):
        """True while the system suspends the page by design (iOS, Glossarion off screen): the reply
        poll's ``browser_driver.ReplyWatch`` does not count that time."""
        return browser_driver.content_suspended(self.page)


#: Times one request is sent again on a new page after its in-app page lost its document
#: (mobile only; see ``_PageLost``).
MAX_PAGE_RESENDS = 3


class _PageLost(AuthChatGPTError):
    """The hooked chatgpt.com document went away while a reply was on its way (or, for a page
    whose renderer stopped, while the page was still being set up).

    On Glossarion Mobile's flet-webview fallback this is the app changing screens: Flet 1.0.3
    draws ``page.overlay`` (where those hidden WebViews live) only in the top View, so pushing
    or popping a View moves every hidden WebView into another widget tree. Flutter then builds
    a new WebView, which loads its ``url`` again without the hook. The reply on the old page is
    lost. The native in-app pages (hotfix 3) are owned outside Flutter and survive screen
    changes; they lose their document only when its renderer stops (``renderer_gone``) or
    chatgpt.com reloads itself. Either way the message goes again on a new page, up to
    ``MAX_PAGE_RESENDS`` times.

    The reply poll (``_complete``) raises it for a new document. A page whose renderer stopped
    raises it from every wait before the reply too (``_check_page``: loading chatgpt.com, the
    chat frame, attaching images, sending), which could otherwise only run out their deadlines
    on a dead page. ``page`` is the page that lost its document."""

    def __init__(self, page=None):
        super().__init__("AuthChatGPT page was reloaded during translation.", "browser_error")
        self.page = page


#: ``services/native_webview.NATIVE_NAME`` (Glossarion Mobile): the driver whose pages the app's
#: extension owns outside the Flutter widget tree, so screen changes never rebuild them.
_NATIVE_DRIVER = "native-webview"


def _renderer_gone(wrapped):
    """True once the page's renderer stopped (``browser_driver``'s optional ``renderer_gone``, the
    native in-app pages): its document is gone for good and every script answers None. Desktop
    pages and the flet-webview pages have no such attribute."""
    return getattr(getattr(wrapped, "page", None), "renderer_gone", False) is True


def _check_page(wrapped, cancel_check=None):
    """``_check``, then the page itself: a stopped renderer raises ``_PageLost`` (the request goes
    again on a new page) instead of letting a wait poll a dead page until its deadline."""
    _check(cancel_check)
    if _renderer_gone(wrapped):
        raise _PageLost(wrapped)


def _page_lost_message(wrapped, resends):
    """The log line of a resend: what took the page's document (see ``_PageLost``)."""
    again = f"sending the message again ({resends}/{MAX_PAGE_RESENDS})."
    if _renderer_gone(wrapped):
        return f"🔄 AuthChatGPT: the in-app ChatGPT page lost its document (its renderer stopped); {again}"
    driver = getattr(getattr(wrapped, "page", None), "driver", None)
    if getattr(driver, "name", None) == _NATIVE_DRIVER:
        # No screen change rebuilds a native page: chatgpt.com replaced its own document.
        return f"🔄 AuthChatGPT: chatgpt.com reloaded its page while it was answering; {again}"
    return ("🔄 AuthChatGPT: the in-app ChatGPT page was rebuilt while it was answering "
            f"(a screen change); {again}")


def _frame_script(config):
    """Adds the chat iframe to the host page and hooks its window in the same task.

    A WebView cannot run a script before a page's own, but a same-origin iframe keeps its
    first (about:blank) window when chatgpt.com loads in it: hooked at insertion, the app
    starts with the hook in place, exactly like the desktop browser's document-start script.
    The ChatGPT login needs that (a hook added after the app started gets "Unusual activity").
    """
    return (f"(() => {{const old = {_FRAME}; if (old) old.remove();"
            " const frame = document.createElement('iframe');"
            f" frame.id = '{_FRAME_ID}';"
            " frame.style.cssText = 'position:fixed;inset:0;width:100%;height:100%;border:0;background:#fff';"
            " frame.src = 'https://chatgpt.com/'; document.documentElement.appendChild(frame);"
            f" {_hook(config, 'frame.contentWindow')}; return true;}})()")


#: Seconds a phone gets for each page wait (a cold WebView, a busy UI loop, a Cloudflare check).
_DRIVER_TIMEOUT = 90
#: Cloudflare's check page in any language: its title is translated (the phone's language),
#: its options object is not.
_DRIVER_CHALLENGE = "(" + _CHALLENGE + " || !!window._cf_chl_opt)"
#: The chat iframe's app is gone for good: a finished chatgpt.com document in a window without
#: the hook. Cloudflare's check, once passed, reloads the frame into a new window (the hook was
#: on the first one); chatgpt.com may also reload itself (a new build). Waiting cannot fix it.
_FRAME_LOST = ("!window.__glossarionGPT && location.href !== 'about:blank' && "
               "document.readyState === 'complete' && !" + _DRIVER_CHALLENGE)
#: Mobile tabs load chatgpt.com one at a time (see ``_driver_loading``).
_DRIVER_LOADING = threading.Lock()


def _not_loaded(why):
    """``_NOT_LOADED`` naming the wait that failed, so the log says which one."""
    return _NOT_LOADED.replace(".", f" ({why}).", 1)


def _driver_wait(wrapped, predicate, cancel_check, why):
    cdp_browser.wait_until(wrapped, predicate, timeout=_DRIVER_TIMEOUT,
                           check=lambda cancel: _check_page(wrapped, cancel), error=AuthChatGPTError,
                           cancel_check=cancel_check, message=_not_loaded(f"{why} within {_DRIVER_TIMEOUT:g} s"),
                           tolerate=("browser_error", "timeout"), interval=.3)


def _driver_load(wrapped, url, cancel_check):
    """Navigate to ``url`` and wait for the document that navigation brings, loading it once
    more before failing.

    The current document is marked first: ``load`` only starts the navigation, and the page's
    old chatgpt.com document (the same URL reloaded, or the host page after the app) is already
    loaded, so without the mark it passed for the new one and scripts ran on a page about to go.
    """
    what = url.replace(BASE_URL, "chatgpt.com")
    mark = uuid.uuid4().hex
    try:
        wrapped.evaluate(f"(window.__glossarionNav = '{mark}', true)")
    except AuthChatGPTError as exc:
        if exc.error_type == "cancelled":
            raise
    loaded = ("document.readyState !== 'loading' && location.origin === 'https://chatgpt.com' && "
              f"window.__glossarionNav !== '{mark}'")
    for attempt in range(2):
        _check_page(wrapped, cancel_check)
        try:
            wrapped.page.load(url)
        except browser_driver.BrowserCancelled as exc:
            raise AuthChatGPTError("AuthChatGPT: translation stopped by user", "cancelled") from exc
        except Exception as exc:  # the UI loop did not start the navigation in time
            if _renderer_gone(wrapped):  # a native page whose renderer stopped refuses every load
                raise _PageLost(wrapped) from exc
            if attempt:
                raise AuthChatGPTError(_not_loaded(f"the in-app browser did not open {what}: {str(exc)[:200]}"),
                                       "browser_error") from exc
            continue
        try:
            _driver_wait(wrapped, loaded, cancel_check, f"{what} did not load")
            return
        except AuthChatGPTError as exc:
            if exc.error_type == "cancelled" or attempt or isinstance(exc, _PageLost):
                raise


def _driver_insert_frame(wrapped, config):
    wrapped.framed = False
    wrapped.evaluate(_frame_script(config))
    wrapped.framed = True


def _driver_frame_gone(wrapped):
    """True when the host page has no chat iframe (asked of the top page itself)."""
    framed, wrapped.framed = wrapped.framed, False
    try:
        return wrapped.evaluate(f"!{_FRAME}") is True
    except AuthChatGPTError as exc:
        if exc.error_type == "cancelled":
            raise
        return False
    finally:
        wrapped.framed = framed


def _driver_frame_ready(wrapped, config, cancel_check):
    """Waits until the chat iframe runs the hooked app. A frame whose app lost the hook, or
    that is still not ready at the deadline, is inserted again once before failing.

    A frame missing from the host page is not the frame's failure: the host document itself
    was replaced (Flet builds a hidden WebView again when the app changes screens, and the new
    one loads its ``url``, the bare host page). Probes through the frame then only answer None,
    so the wait ran out its whole deadline and spent the frame's one recovery on it. The host
    page is loaded again and the frame inserted at once instead, within the same deadline."""
    recovered = False
    deadline = time.monotonic() + _DRIVER_TIMEOUT
    while True:
        _check_page(wrapped, cancel_check)  # a stopped renderer would answer None past both deadlines
        ready = None
        try:
            ready = wrapped.evaluate(_READY)
            if ready:
                return
            lost = wrapped.evaluate(_FRAME_LOST) is True
        except AuthChatGPTError as exc:
            if exc.error_type == "cancelled":
                raise
            lost = False
        if lost or time.monotonic() >= deadline:
            if recovered:
                raise AuthChatGPTError(_not_loaded(
                    "the chat frame lost its hook again" if lost
                    else f"the chat frame was not ready within {_DRIVER_TIMEOUT:g} s"), "browser_error")
            recovered = True
            _driver_insert_frame(wrapped, config)
            deadline = time.monotonic() + _DRIVER_TIMEOUT
        elif ready is None and _driver_frame_gone(wrapped):
            wrapped.framed = False
            _driver_load(wrapped, _HOST_URL, cancel_check)
            _driver_insert_frame(wrapped, config)
        time.sleep(.3)


def _driver_frame(wrapped, config, cancel_check):
    """Logged in: chatgpt.com in the chat iframe of a small host page, hooked at insertion."""
    _driver_load(wrapped, _HOST_URL, cancel_check)
    if not wrapped.evaluate("document.cookie.indexOf('oai-did=') >= 0"):
        # A first visit: chatgpt.com sets up its device cookies on a top-level page load.
        _driver_load(wrapped, BASE_URL + "/", cancel_check)
        _driver_wait(wrapped, "document.cookie.indexOf('oai-did=') >= 0", cancel_check,
                     "chatgpt.com set no device cookie")
        _driver_load(wrapped, _HOST_URL, cancel_check)
    wrapped.evaluate(_APP_COOKIE_JS)
    _driver_insert_frame(wrapped, config)
    _driver_frame_ready(wrapped, config, cancel_check)
    wrapped.evaluate("window.__glossarionGPT.watch()")


@contextmanager
def _driver_loading(cancel_check):
    """One mobile tab loads chatgpt.com at a time; loaded tabs still chat in parallel.

    The in-app WebViews share one cookie jar and one renderer. Parallel requests (glossary
    extraction, batch translation) opened their tabs together, so every cold tab met
    Cloudflare's check on its own (a passed check reloads the chat frame without its hook,
    spending the frame's one recovery in ``_driver_frame_ready``), each ran the first-visit
    cookie loads, and several chatgpt.com apps started at once on one phone renderer. One more
    hiccup then failed the request ("did not finish loading") until a later retry found the
    cookies in place. Loaded one after another, the first tab earns the cookies and the next
    ones load with them."""
    while not _DRIVER_LOADING.acquire(timeout=.2):
        _check(cancel_check)
    try:
        yield
    finally:
        _DRIVER_LOADING.release()


@contextmanager
def _driver_tab(config, cancel_check=None):
    """A hooked chatgpt.com page of the registered browser driver (mobile; no subprocess)."""
    try:
        driver = browser_driver.require_driver("AuthChatGPT")
    except browser_driver.BrowserUnavailable as exc:
        raise AuthChatGPTError("AuthChatGPT needs the in-app browser (Glossarion Mobile) or the desktop app "
                               "with Chrome/Edge.", "config_error") from exc

    def cancelled():
        return _cancel.is_set() or bool(cancel_check and cancel_check())

    try:  # waits while every page slot is in use (other requests); a Stop ends the wait
        page = driver.open_page(owner=BROWSER_OWNER, cancel_check=cancelled)
    except browser_driver.BrowserCancelled as exc:
        raise AuthChatGPTError("AuthChatGPT: translation stopped by user", "cancelled") from exc
    try:
        wrapped = _DriverPage(page)
        with _driver_loading(cancel_check):
            if config.get("token"):
                _driver_frame(wrapped, config, cancel_check)
            else:
                # Logged out, the app runs as the page itself (in the iframe chatgpt.com refuses a
                # logged-out turn): the hook is added once it started, which logged-out chat allows.
                _driver_load(wrapped, BASE_URL + "/", cancel_check)
                if not wrapped.evaluate("document.cookie.indexOf('__Secure-next-auth.session-token=') >= 0"):
                    wrapped.evaluate(_APP_COOKIE_JS)
                    _driver_load(wrapped, BASE_URL + "/", cancel_check)
                composer = "!!document.querySelector('#prompt-textarea')"
                try:
                    _driver_wait(wrapped, composer, cancel_check, "the ChatGPT composer did not appear")
                except AuthChatGPTError as exc:
                    if exc.error_type == "cancelled" or isinstance(exc, _PageLost):
                        raise
                    _driver_load(wrapped, BASE_URL + "/", cancel_check)  # one reload before failing
                    _driver_wait(wrapped, composer, cancel_check,
                                 "the ChatGPT composer did not appear after a reload")
                wrapped.evaluate("(" + _hook(config) + ", true)")
        yield wrapped
    finally:
        page.close()


@contextmanager
def _open_tab(config, cancel_check=None):
    """A new chatgpt.com page, hooked with ``config``: a tab of the shared desktop browser,
    or a hidden in-app WebView page on Glossarion Mobile."""
    if browser_driver.use_driver("AUTHCHATGPT_MODE"):
        with _driver_tab(config, cancel_check) as page:
            yield page
        return
    if not mobile_runtime.subprocesses_available():
        raise AuthChatGPTError("AuthChatGPT needs the in-app browser here, or the desktop app with Chrome/Edge.",
                               "config_error")
    with _LEASES.open(cancel_check=cancel_check, prepare=_prepare(config)) as page:
        yield page


def _hook(config, target="window"):
    """The page hook for ``config``, installed into the window ``target`` (a JS expression)."""
    return _HOOK_SCRIPT.replace("__CONFIG__", json.dumps(config)).replace("__TARGET__", target)


# Logged in, every backend call goes to the account API with the token; the conversation
# request gets Glossarion's prompt, model and thinking effort, and its streams (the reply
# may continue on /resume) are copied into a queue Python drains. The picker catalog
# (window.__glossarionGPT.loadCatalog) maps picker names to a model slug and an effort.
_HOOK_SCRIPT = r"""((window) => {
 if (window.__glossarionGPT) return;
 const cfg = __CONFIG__;
 const state = window.__glossarionGPT = {chunks:[], requested:false, streams:0, open:0, closedAt:0, status:0,
  error:null, kind:null, sentinel:0, resolved:null, catalog:null, catalogError:null};
 const nativeFetch = window.fetch;
 const backend = /^\/backend-(anon|api)\//, conversation = /^\/backend-(anon|api)\/f\/conversation(\/resume)?$/;
 const norm = s => String(s || '').toLowerCase().trim().replace(/[^a-z0-9.]+/g, '-').replace(/^-+|-+$/g, '');
 let catalogPromise = null;
 // A handed-off answer (Pro runs as a background task) is read from the conversation itself,
 // as chatgpt.com's own poller does, when no stream delivers it. ``limitMs`` (the in-app pages):
 // a read that takes longer is aborted. A read that fails (or finds no conversation to read) adds
 // one to ``pollFailures`` (``pollFailure`` says how); a good read sets it back to 0.
 state.poll = limitMs => {
  if (state.polling) return false;
  const failed = how => { state.pollFailures = (state.pollFailures || 0) + 1; state.pollFailure = how; };
  if (!state.conversationId) { failed('no conversation'); return false; }
  state.polling = true;
  const controller = limitMs ? new AbortController() : null;
  const timer = controller ? setTimeout(() => controller.abort(), limitMs) : 0;
  (async () => {
   try {
    const base = cfg.token ? '/backend-api' : '/backend-anon';
    const init = {credentials:'include', headers: cfg.token ? {authorization:'Bearer ' + cfg.token} : {}};
    if (controller) init.signal = controller.signal;
    const response = await nativeFetch.call(window, base + '/conversation/' + state.conversationId, init);
    state.pollStatus = response.status;
    if (!response.ok) { failed('HTTP ' + response.status); return; }
    const data = await response.json();
    state.pollFailures = 0;
    state.pollFailure = null;
    let best = null;
    for (const node of Object.values(data.mapping || {})) {
     const m = node && node.message, c = (m && m.content) || {};
     if (!m || !m.author || m.author.role !== 'assistant' || c.content_type !== 'text') continue;
     if ((m.recipient || 'all') !== 'all' || !(m.channel == null || m.channel === 'final')) continue;
     if (m.status !== 'finished_successfully' || m.end_turn === false) continue;
     if (!best || (m.create_time || 0) >= (best.create_time || 0)) best = m;
    }
    if (best) state.final = {text:(best.content.parts || []).filter(p => typeof p === 'string').join(''),
     model:(best.metadata || {}).resolved_model_slug || (best.metadata || {}).model_slug || null};
    // generated images: the image tool's multimodal messages (one per variant)
    const images = [];
    for (const node of Object.values(data.mapping || {})) {
     const m = node && node.message, c = (m && m.content) || {};
     if (!m || !m.author || m.author.role !== 'tool' || m.status !== 'finished_successfully') continue;
     for (const part of c.parts || []) if (part && part.asset_pointer)
      images.push({pointer:part.asset_pointer, index:(m.metadata || {}).generation_index || 0});
    }
    if (images.length) state.finalImages = images.sort((a, b) => a.index - b.index);
   } catch (e) {
    state.pollError = String(e && e.message || e);
    failed(controller && controller.signal.aborted ? 'no answer in time'
     : e && e.name === 'SyntaxError' ? 'unreadable answer' : 'network error');
   }
   finally { if (timer) clearTimeout(timer); state.polling = false; }
  })();
  return true;
 };
 // A generated image: its signed download URL, then the bytes (base64) for Python.
 state.downloads = {};
 state.download = pointer => {
  if (state.downloads[pointer]) return true;
  state.downloads[pointer] = {pending:true};
  (async () => {
   try {
    const fileId = String(pointer).replace(/^[a-z-]+:\/\//, '');
    let info = null, response = null, failure = '';
    // a just-finished image can take a moment to become downloadable: retry for up to a minute
    for (let attempt = 0; attempt < 20; attempt++) {
     if (attempt) await new Promise(r => setTimeout(r, 3000));
     const listed = await nativeFetch.call(window, '/backend-api/files/download/' + encodeURIComponent(fileId) +
      '?conversation_id=' + encodeURIComponent(state.conversationId || '') + '&inline=false',
      {credentials:'include', headers:{authorization:'Bearer ' + cfg.token}});
     info = listed.ok ? await listed.json() : null;
     if (!info || !info.download_url) { failure = 'HTTP ' + listed.status + (info && info.detail ? ': ' + info.detail : ''); continue; }
     // the signed link is a /backend-api/ URL too: without a ChatGPT cookie session it needs the token
     const sameSite = new URL(info.download_url, location.href).origin === location.origin;
     response = await nativeFetch.call(window, info.download_url, {credentials:'include',
      headers: sameSite ? {authorization:'Bearer ' + cfg.token} : {}});
     if (response.ok) break;
     failure = 'HTTP ' + response.status;
     response = null;
    }
    if (!response) throw Error(failure || 'no download URL');
    const bytes = new Uint8Array(await response.arrayBuffer());
    let binary = '';
    for (let i = 0; i < bytes.length; i += 32768) binary += String.fromCharCode.apply(null, bytes.subarray(i, i + 32768));
    state.downloads[pointer] = {b64:btoa(binary), mime:response.headers.get('content-type') || info.mime_type || 'image/png'};
   } catch (e) { state.downloads[pointer] = {error:String(e && e.message || e)}; }
  })();
  return true;
 };
 state.loadCatalog = () => catalogPromise || (catalogPromise = (async () => {
  const path = (cfg.token ? '/backend-api' : '/backend-anon') + '/models?iim=false&is_gizmo=false&supports_model_picker_upgrade_presets=true';
  const response = await nativeFetch.call(window, path, {credentials:'include',
   headers: cfg.token ? {authorization:'Bearer ' + cfg.token} : {}});
  if (!response.ok) throw Error('HTTP ' + response.status);
  const data = await response.json(), names = {}, locked = {};
  for (const version of data.versions || []) {
   if (version.enabled === false) continue;
   let label = norm(version.display_text_for_intelligence || version.id);
   if (!label || label === 'auto') continue;
   if (!label.startsWith('gpt')) label = 'gpt-' + label;
   for (const preset of version.intelligence_presets || []) {
    const name = label + '-' + norm(preset.title || preset.selected_display_title);
    const into = preset.preset_type === 'available' ? names : locked;
    if (!(name in into)) into[name] = {model:preset.model_slug, effort:preset.thinking_effort || null,
     title:((version.display_text_for_intelligence || version.id) + ' ' + (preset.title || '')).trim()};
   }
  }
  return state.catalog = {names, locked, slugs:(data.models || []).map(m => m.slug)};
 })().catch(e => { state.catalogError = String(e && e.message || e); catalogPromise = null; return null; }));
 async function resolve(name) {
  const catalog = await state.loadCatalog();
  if (!catalog) return {model:name, effort:null, title:name};  // picker unavailable: a plain slug
  if (catalog.names[name]) return catalog.names[name];
  if (name === 'auto' || catalog.slugs.includes(name)) return {model:name, effort:null, title:name};
  if (catalog.locked[name]) return {error:name + ' is not included in ' + (cfg.token ? 'this ChatGPT plan' : 'logged-out ChatGPT') + '.'};
  return {error:'Unknown model ' + name + (cfg.token ? '' : ' for logged-out ChatGPT (log in with ChatGPT Login for more models)')
   + '. Available: ' + Object.keys(catalog.names).concat('auto').join(', ')};
 }
 function install(win) {
  if (win.__glossarionHooked) return;
  win.__glossarionHooked = true;
  const nativeFetch = win.fetch, Request = win.Request, Headers = win.Headers, Response = win.Response,
   URL = win.URL, location = win.location;
  win.fetch = async function(input, init) {
  // Building a Request from a Request consumes its body: inspect first, rebuild only to rewrite.
  const url = new URL(input instanceof Request ? input.url : String(input), location.href);
  if (url.origin !== location.origin || !backend.test(url.pathname)) return nativeFetch.call(this, input, init);
  const method = ((init && init.method) || (input instanceof Request ? input.method : 'GET')).toUpperCase();
  const isConversation = method === 'POST' && conversation.test(url.pathname);
  if (!cfg.token && !isConversation) return nativeFetch.call(this, input, init);
  const request = new Request(input, init), headers = new Headers(request.headers);
  if (cfg.token) {
   url.pathname = url.pathname.replace(/^\/backend-anon\//, '/backend-api/');
   headers.set('authorization', 'Bearer ' + cfg.token);
  }
  let body = ['GET', 'HEAD'].includes(request.method) ? undefined : await request.clone().arrayBuffer();
  if (isConversation && !url.pathname.endsWith('/resume')) {
   // One chat turn per page: a repeated Send never becomes a second request.
   if (state.requested) return new Response(null, {status:409});
   state.requested = true;
   const pick = await resolve(cfg.model);
   if (pick.error) { state.error = pick.error; state.kind = 'config_error'; return new Response(null, {status:400}); }
   state.resolved = pick;
   const payload = JSON.parse(new TextDecoder().decode(body));
   const last = payload.messages[payload.messages.length - 1];
   // the page uploaded the images itself: keep its image parts, replace only the text
   const media = ((last.content && Array.isArray(last.content.parts)) ? last.content.parts : []).filter(p => p && typeof p === 'object');
   if ((cfg.images || 0) > media.length) {
    state.error = 'ChatGPT did not attach the image (' + media.length + ' of ' + cfg.images + ').';
    state.kind = 'browser_error';
    return new Response(null, {status:400});
   }
   last.content = media.length ? {content_type:'multimodal_text', parts:[...media, cfg.prompt]}
    : {content_type:'text', parts:[cfg.prompt]};
   payload.model = pick.model;
   if (pick.effort) payload.thinking_effort = pick.effort; else delete payload.thinking_effort;
   if (cfg.temporary) payload.history_and_training_disabled = true;
   // image output: what the composer's "Create image" sends
   if (cfg.imageOutput) payload.system_hints = ['picture_v2'];
   body = JSON.stringify(payload);
  }
  const response = await nativeFetch.call(this, url.toString(), {method:request.method, headers, body,
   signal:request.signal, credentials:'include', cache:request.cache, redirect:request.redirect,
   referrer:request.referrer, referrerPolicy:request.referrerPolicy});
  if (/\/sentinel\/chat-requirements/.test(url.pathname) && !response.ok) state.sentinel = response.status;
  if (!isConversation) return response;
  state.status = response.status;
  if (!response.ok || !response.body) {
   let detail = '';
   try { const j = await response.clone().json(); detail = typeof j.detail === 'string' ? j.detail : (j.detail && j.detail.message) || ''; } catch (_) {}
   state.error = ('HTTP ' + response.status + (detail ? ': ' + detail : '')).slice(0, 300);
   return response;
  }
  const [page, copy] = response.body.tee();
  state.streams += 1; state.open += 1;
  (async () => {
   const reader = copy.getReader(), decoder = new TextDecoder();
   try { while (true) { const {done, value} = await reader.read(); if (done) break;
          const text = decoder.decode(value, {stream:true}), id = /"conversation_id":"([0-9a-f-]{36})"/.exec(text);
          if (id) state.conversationId = id[1];
          state.chunks.push(text); }
         state.chunks.push(decoder.decode() + '\n\n'); }
   catch (_) { state.chunks.push('\n\n'); }
   finally { state.open -= 1; state.closedAt = Date.now(); }
  })();
  return new Response(page, {status:response.status, statusText:response.statusText, headers:response.headers});
  };
 }
 install(window);
 // The page's sentinel iframe (anti-abuse proofs) sends its own requests, which must carry the
 // same login. A same-origin iframe keeps its first (about:blank) window when its page loads,
 // so hooking it on insertion runs the hook before the iframe's own scripts.
 state.watch = () => {
  if (!window.MutationObserver || !window.document) return false;
  if (state.watched === window.document) return true;
  state.watched = window.document;
  new window.MutationObserver(records => {
   for (const record of records) for (const node of record.addedNodes) {
    const frames = node.tagName === 'IFRAME' ? [node] : (node.querySelectorAll ? [...node.querySelectorAll('iframe')] : []);
    for (const frame of frames) { try { if (frame.contentWindow) install(frame.contentWindow); } catch (_) {} }
   }
  }).observe(window.document, {childList:true, subtree:true});
  return true;
 };
 state.watch();
})(__TARGET__)"""

# ``paused``: milliseconds a 0 ms probe timer has been waiting to run (0 once it ran). A page whose
# timers stopped (Android froze it as a hidden page) keeps answering scripts but never runs it.
# ``pollFailures`` / ``pollFailure`` / ``pollStatus``: the reads of a handed-off answer (``state.poll``).
_POLL = """(() => {const s = window.__glossarionGPT; if (!s) return null; const now = Date.now();
 if (!s.probeAt) { s.probeAt = now; setTimeout(() => { s.probeAt = 0; }, 0); }
 return {chunks:s.chunks.splice(0), streams:s.streams, open:s.open, idle:s.closedAt ? now - s.closedAt : 0,
 paused:s.probeAt ? now - s.probeAt : 0, status:s.status, error:s.error, kind:s.kind, sentinel:s.sentinel,
 resolved:s.resolved, final:s.final || null, finalImages:s.finalImages || null,
 pollFailures:s.pollFailures || 0, pollFailure:s.pollFailure || null, pollStatus:s.pollStatus || 0};})()"""
_POLL_ANSWER = "(() => window.__glossarionGPT.poll())()"
#: the download state of one generated image (``__POINTER__``): {b64, mime} / {error} / {pending}
_DOWNLOAD = "(() => {const s = window.__glossarionGPT; s.download(__POINTER__); return s.downloads[__POINTER__] || null;})()"
#: logged-out chatgpt.com cannot create images ("Log in to use")
IMAGE_OUTPUT_NEEDS_LOGIN = ("AuthChatGPT needs a ChatGPT login to create images (logged-out ChatGPT cannot). "
                            "Click ChatGPT Login.")
#: seconds between reads of a handed-off answer (chatgpt.com's own poll_interval_ms is 10 s)
_HANDOFF_POLL_SECONDS = 10
#: In-app pages: a read of a handed-off answer (Pro) or of a generated image's conversation that
#: takes longer than this (ms) is aborted, so a hung read cannot stop the next ones ...
_HANDOFF_READ_MS = 60000
#: ... and this many failed reads in a row (an HTTP error, a network error, no answer in time, no
#: conversation to read) fail the request with the reason instead of waiting out its timeout (10 h
#: with Retry Timeout off). Desktop pages keep reading until the request's deadline, as before.
_HANDOFF_READ_FAILURES = 6
_POLL_ANSWER_TIMED = "(() => window.__glossarionGPT.poll(%d))()" % _HANDOFF_READ_MS
#: How a read failed (``pollFailure``), as the hook writes it (anything else is not shown).
_READ_FAILURES = ("no answer in time", "unreadable answer", "network error", "no conversation")
#: In-app pages: seconds a generated image may take to download (the page itself retries for a
#: minute; desktop keeps the request's deadline).
_DRIVER_IMAGE_SECONDS = 300
_COMPOSER = ("(() => {const c = document.querySelector('#prompt-textarea'), b = document.querySelector('[data-testid=\"send-button\"]');"
             " return {text: c ? c.innerText.trim() : null, send: !!b && !b.disabled,"
             " requested: !!(window.__glossarionGPT && (window.__glossarionGPT.requested || window.__glossarionGPT.sentinel))};})()")
# Plain DOM typing (no DevTools input), so the WebView driver can do it too.
_TYPE = ("(() => {const c = document.querySelector('#prompt-textarea'); if (!c) return false; c.focus();"
         " return document.execCommand('insertText', false, '.');})()")
_SEND = "(() => {const b = document.querySelector('[data-testid=\"send-button\"]'); if (b) b.click(); return true;})()"
# Images go in through the composer's own file input, so the page uploads them itself.
_ATTACH = r"""(() => {
 const list = window.__glossarionImages || [];
 const input = document.querySelector('#upload-files') ||
  [...document.querySelectorAll('input[type=file]')].find(i => /image/.test(i.accept || ''));
 if (!input) return 'no-input';
 const files = new window.DataTransfer();
 for (const image of list) {
  const raw = atob(image.b64), bytes = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
  files.items.add(new window.File([bytes], image.name, {type: image.mime}));
 }
 input.files = files.files;
 input.dispatchEvent(new window.Event('change', {bubbles: true}));
 return 'ok';
})()"""
_ATTACHED = ("(() => {const busy = document.querySelector('form [role=\"progressbar\"], form circle[stroke-dasharray]');"
             " return !busy && document.querySelectorAll('form img').length;})()")
_LOAD_CATALOG = "(() => {window.__glossarionGPT.loadCatalog(); return true;})()"
_CATALOG = ("(() => {const s = window.__glossarionGPT; return s ? {catalog: s.catalog && {names: s.catalog.names},"
            " error: s.catalogError} : null;})()")


# ---------------------------------------------------------------------------
# Stream
# ---------------------------------------------------------------------------

_CODE_OPEN = re.compile(r'<CodeBlock(?:\s+language="([^"]*)")?[^>]*>\n?')


def clean_markup(text):
    """chatgpt.com's reply markup back to the model's markdown: ``<escape>`` wraps raw
    HTML and ``<CodeBlock language="x">`` is a fenced code block."""
    text = _CODE_OPEN.sub(lambda m: f"```{m.group(1) or ''}\n", text)
    text = re.sub(r"\n?</CodeBlock>", "\n```", text)
    return text.replace("<escape>", "").replace("</escape>", "")


class StreamParser:
    """Incremental parser of chatgpt.com's conversation stream (SSE, delta encoding v1).

    Only the assistant's visible text message is the answer; reasoning summaries, tool
    calls and system messages are kept out of it.
    """

    def __init__(self):
        self.buffer = ""
        self.message = None  # the message the patches apply to
        self.answer = None  # the latest visible assistant text message
        self.path = None
        self.operation = None
        self.reasoning = ""
        self.finished = False
        self.error = None
        self.model = None
        self.on_thinking = None
        # Pro (and other long) answers run as a background task: the first stream hands off
        # ("stream_handoff") and ends; the answer arrives on the stream that resumes the turn.
        self.handoff = False
        self.generated = []  # (generation index, asset pointer) of generated images

    @property
    def content(self):
        if not self.answer:
            return ""
        return clean_markup("".join(p for p in self.answer["content"].get("parts", []) if isinstance(p, str)))

    @staticmethod
    def _visible(message):
        content = message.get("content")
        content = content if isinstance(content, dict) else {}
        return ((message.get("author") or {}).get("role") == "assistant" and
                content.get("content_type") == "text" and message.get("recipient", "all") == "all" and
                message.get("channel") in (None, "final"))

    def feed(self, text):
        self.buffer = (self.buffer + text).replace("\r\n", "\n")
        outputs = []
        while "\n\n" in self.buffer:
            frame, self.buffer = self.buffer.split("\n\n", 1)
            if frame.startswith("event: delta_encoding"):
                self.handoff = False  # a new stream of the turn
            values = [line[5:].lstrip() for line in frame.splitlines() if line.startswith("data:")]
            data = "\n".join(values)
            if not data:
                continue
            if data == "[DONE]":
                self.finished = not self.handoff
                continue
            try:
                payload = json.loads(data)
            except ValueError as exc:
                raise AuthChatGPTError("AuthChatGPT returned an invalid stream frame.") from exc
            if isinstance(payload, dict):
                outputs.extend(self._event(payload))
        return outputs

    def _event(self, payload):
        if payload.get("type") == "stream_handoff":
            self.handoff = True
            return []
        if payload.get("type") == "message_stream_complete":
            self.finished = not self.handoff
            return []
        if payload.get("error"):
            error = payload["error"]
            self.error = error if isinstance(error, str) else (error.get("message") if isinstance(error, dict) else None) or "error"
            return []
        if "type" in payload and "v" not in payload:
            return []  # markers, metadata, resume tokens
        value = payload.get("v")
        if isinstance(value, dict) and isinstance(value.get("message"), dict):
            return self._start(value["message"])
        if "p" in payload:
            self.path, self.operation = payload.get("p"), payload.get("o", self.operation)
        elif "o" in payload:
            self.operation = payload["o"]
        is_patch = isinstance(value, list) and value and all(isinstance(v, dict) and "p" in v for v in value)
        if isinstance(value, list) and (payload.get("o") == "patch" or ("p" not in payload and is_patch)):
            outputs = []
            for patch in value:
                if isinstance(patch, dict) and "p" in patch:
                    self.path, self.operation = patch["p"], patch.get("o", "replace")
                    outputs.extend(self._apply(self.path, self.operation, patch.get("v")))
            return outputs
        if self.path is not None:
            return self._apply(self.path, self.operation or "append", value)
        return []

    def _collect_images(self, message):
        # the pointer shows up while the image is still being made: only a finished one is ready
        if (message.get("author") or {}).get("role") != "tool" or message.get("status") != "finished_successfully":
            return
        content = message.get("content")
        index = (message.get("metadata") or {}).get("generation_index") or 0
        for part in (content.get("parts") or []) if isinstance(content, dict) else []:
            pointer = part.get("asset_pointer") if isinstance(part, dict) else None
            if pointer and pointer not in [p for _i, p in self.generated]:
                self.generated.append((index, pointer))

    def _start(self, message):
        self.message = message
        self.path = self.operation = None
        self._collect_images(message)
        metadata = message.get("metadata") or {}
        self.model = metadata.get("resolved_model_slug") or metadata.get("model_slug") or self.model
        if not self._visible(message):
            content = message.get("content")
            if isinstance(content, dict) and content.get("content_type") == "thoughts":
                self._thoughts(content.get("thoughts"))
            elif isinstance(content, dict) and content.get("content_type") == "reasoning_recap":
                self._thoughts([f"({content.get('content') or 'Finished reasoning'})\n"])  # e.g. "Worked for 48s"
            return []
        message.setdefault("content", {}).setdefault("parts", [""])
        self.answer = message
        return [p for p in message["content"]["parts"] if isinstance(p, str) and p]

    def _thoughts(self, thoughts):
        for thought in thoughts or ():
            text = _thought_text(thought)
            if text:
                self.reasoning += text
                if self.on_thinking:
                    self.on_thinking(text)

    def _apply(self, path, operation, value):
        if self.message is None or not isinstance(path, str) or not path.startswith("/message"):
            return []
        keys = [k for k in path.split("/")[2:] if k != ""]
        if not keys:
            return []
        node = self.message
        for key in keys[:-1]:
            if isinstance(node, list):
                if not key.isdigit() or int(key) >= len(node):
                    return []
                node = node[int(key)]
            elif isinstance(node, dict):
                node = node.setdefault(key, {})
            else:
                return []
        last = keys[-1]
        if isinstance(node, list):
            if not last.isdigit():
                return []
            index = int(last)
            while len(node) <= index:
                node.append("")
            current = node[index]
        elif isinstance(node, dict):
            current = node.get(last)
        else:
            return []
        if operation == "append" and isinstance(current, str) and isinstance(value, str):
            new = current + value
        elif operation == "append" and isinstance(current, list):
            new = current + (value if isinstance(value, list) else [value])
        elif operation == "append" and isinstance(current, dict) and isinstance(value, dict):
            new = {**current, **value}
        elif operation in ("append", "replace", "add"):
            new = value
        elif operation == "remove" and isinstance(node, dict):
            node.pop(last, None)
            return []
        else:
            return []
        if isinstance(node, list):
            node[int(last)] = new
        else:
            node[last] = new
        if keys == ["content", "thoughts"] and operation == "append":
            self._thoughts(value if isinstance(value, list) else [value])
            return []
        if keys[:2] == ["content", "parts"] or keys == ["status"]:
            self._collect_images(self.message)
        text = value if isinstance(value, str) and operation == "append" else ""
        if self._visible(self.message):
            self.answer = self.message
            return [text] if text and keys[:2] == ["content", "parts"] else []
        if text and "thoughts" in keys:
            self.reasoning += text
            if self.on_thinking:
                self.on_thinking(text)
        return []


def _thought_text(thought):
    if not isinstance(thought, dict):
        return thought if isinstance(thought, str) else ""
    summary, content = str(thought.get("summary") or "").strip(), str(thought.get("content") or "").strip()
    return "\n".join(part for part in (f"**{summary}**" if summary else "", content) if part) + "\n\n"


def _kind(status):
    return {401: "auth_error", 403: "access_denied", 429: "rate_limit"}.get(status, "api_error")


def _attach(page, images, cancel_check):
    """Adds ``images`` to the composer and waits until the page has uploaded them."""
    page.evaluate("(window.__glossarionImages = " + json.dumps(images) + ", true)")
    if page.evaluate(_ATTACH) != "ok":
        if _renderer_gone(page):  # its None is no answer at all
            raise _PageLost(page)
        raise AuthChatGPTError("ChatGPT's page has no image upload here; images need a ChatGPT login.",
                               "browser_error")
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        _check_page(page, cancel_check)
        if (page.evaluate(_ATTACHED) or 0) >= len(images):
            time.sleep(1.5)  # the thumbnail shows before the upload's last step
            if (page.evaluate(_ATTACHED) or 0) >= len(images):
                return
        time.sleep(.5)
    raise AuthChatGPTError("ChatGPT did not finish uploading the image.", "timeout")


def _submit(page, cancel_check, before_send_callback):
    """Send one message from the composer; the hook swaps in the real prompt and model.

    The composer needs some text before the page enables Send. While the app is still
    starting, a re-render can wipe that text or ignore a click, so this types and clicks
    again until the page has made its conversation request.
    """
    deadline = time.monotonic() + 45
    clicked = None
    while time.monotonic() < deadline:
        _check_page(page, cancel_check)
        state = page.evaluate(_COMPOSER) or {}
        if state.get("requested"):
            return
        waiting = clicked is not None and time.monotonic() - clicked < 8
        if state.get("text") == "" and not waiting:
            page.evaluate(_TYPE)
        elif state.get("text") and state.get("send") and not waiting:
            if clicked is None and before_send_callback:
                before_send_callback()
            page.evaluate(_SEND)
            clicked = time.monotonic()
        time.sleep(.3)
    raise AuthChatGPTError("ChatGPT did not accept the message. Try again.", "browser_error")


def _download_image(page, pointer, deadline, cancel_check):
    """The bytes of a generated image (downloaded by the page, signed URL and all)."""
    script = _DOWNLOAD.replace("__POINTER__", json.dumps(pointer))
    # An in-app page that stopped never ends the wait itself: it gets _DRIVER_IMAGE_SECONDS, not
    # counting time the system suspends it by design (iOS in the background).
    driver = getattr(page, "driver", False) is True
    cap = time.monotonic() + _DRIVER_IMAGE_SECONDS
    while time.monotonic() < deadline and (not driver or time.monotonic() < cap):
        _check_page(page, cancel_check)
        if driver and browser_driver.content_suspended(page):
            cap = time.monotonic() + _DRIVER_IMAGE_SECONDS
        state = page.evaluate(script) or {}
        if state.get("b64"):
            import base64
            return {"mime": state.get("mime") or "image/png", "data": base64.b64decode(state["b64"])}
        if state.get("error"):
            raise AuthChatGPTError(f"AuthChatGPT could not download the generated image ({state['error']}).")
        time.sleep(.5)
    raise AuthChatGPTError("AuthChatGPT timed out downloading the generated image.", "timeout")


def _answered(watch, asked, state):
    """``state`` (a ``_POLL`` answer) for the reply's ``browser_driver.ReplyWatch``."""
    watch.answered(asked, paused=state.get("paused"), data=bool(state.get("chunks")),
                   streaming=bool(state.get("open")))


def _read_answer(page, watch):
    """Start one read of a handed-off answer from the conversation (an in-app page's read is aborted
    after ``_HANDOFF_READ_MS``; a desktop page's read runs as before)."""
    page.evaluate(_POLL_ANSWER_TIMED if watch.on else _POLL_ANSWER)


def _check_reads(watch, state):
    """In-app pages: ``_HANDOFF_READ_FAILURES`` failed reads of the conversation in a row fail the
    request with the reason (an HTTP status keeps its error kind). Desktop pages: nothing."""
    failures = state.get("pollFailures")
    if (not watch.on or isinstance(failures, bool) or not isinstance(failures, int)
            or failures < _HANDOFF_READ_FAILURES):
        return
    how = str(state.get("pollFailure") or "")
    status = state.get("pollStatus")
    if re.fullmatch(r"HTTP \d{3}", how) and isinstance(status, int) and not isinstance(status, bool):
        kind = _kind(status)
    elif how in _READ_FAILURES:
        kind = "timeout" if how == "no answer in time" else "browser_error"
    else:
        how, kind = "read failed", "browser_error"
    raise AuthChatGPTError(f"AuthChatGPT: ChatGPT's answer could not be read ({how}, {failures} times in a row).",
                           kind)


def _complete(page, timeout, cancel_check, on_delta, before_send_callback, on_thinking=None, images=None,
              image_output=False, log_fn=None):
    if images:
        _attach(page, images, cancel_check)
    with _COOKIES.sending_message(cancel_check):
        _submit(page, cancel_check, before_send_callback)
    parser = StreamParser()
    parser.on_thinking = on_thinking
    started = time.monotonic()
    deadline = started + timeout
    resolved = None
    polled = 0.0
    # Glossarion Mobile's in-app pages: a page that stopped answering or running fails the request
    # (``browser_driver.ReplyWatch``), except while the system suspends it by design (iOS in the
    # background: the reply continues when the app is opened); desktop CDP pages have no ``driver``
    # flag and poll as before.
    watch = browser_driver.ReplyWatch(getattr(page, "driver", False) is True, route="AuthChatGPT", site="ChatGPT",
                                      host="chatgpt.com", error=AuthChatGPTError, log_fn=log_fn,
                                      hidden=getattr(page, "content_suspended", None))
    while time.monotonic() < deadline:
        _check(cancel_check)
        asked = watch.poll_started()
        state = page.evaluate(_POLL)
        if not state:
            # Desktop: no hook state means a new document. The in-app browser also answers
            # None when an answer was late, so there the page is asked whether the hook is gone.
            probe = getattr(page, "hook_lost", None)
            if probe is None or probe() is True:
                raise _PageLost(page)
            watch.unanswered(asked)  # an in-app page that never answers fails the request
            time.sleep(.3)
            continue
        resolved = state.get("resolved") or resolved
        if state.get("error"):
            # Server detail text only: tokens and browser state never reach the log.
            raise AuthChatGPTError(f"AuthChatGPT: {state['error']}",
                                   state.get("kind") or _kind(state.get("status", 0)))
        for chunk in state.get("chunks", []):
            for delta in parser.feed(chunk):
                if on_delta:
                    on_delta(delta)
        if parser.error:
            raise AuthChatGPTError(f"AuthChatGPT: {parser.error}")
        if image_output and not state.get("open") and (parser.finished or parser.handoff):
            # the image: from the stream, else from the conversation (generation may finish later)
            generated = sorted(parser.generated) or [(i.get("index") or 0, i["pointer"])
                                                      for i in (state.get("finalImages") or [])]
            if generated:
                image = _download_image(page, generated[0][1], deadline, cancel_check)
                final = state.get("final") or {}
                return {"content": parser.content or clean_markup(final.get("text") or ""),
                        "reasoning_content": parser.reasoning, "finish_reason": "stop", "usage": None,
                        "model": parser.model or final.get("model"), "images": [image],
                        "thinking_effort": (resolved or {}).get("effort")}
            _answered(watch, asked, state)
            _check_reads(watch, state)  # an in-app page whose reads keep failing fails the request
            if time.monotonic() - polled >= _HANDOFF_POLL_SECONDS:
                if not polled and on_thinking:
                    on_thinking("ChatGPT is creating the image; waiting for it…\n")
                polled = time.monotonic()
                _read_answer(page, watch)
            time.sleep(.3)
            continue
        if parser.finished and not state.get("open"):
            if not parser.content.strip():
                raise AuthChatGPTError("AuthChatGPT returned no answer text.")
            return {"content": parser.content, "reasoning_content": parser.reasoning,
                    "finish_reason": "stop", "usage": None, "model": parser.model,
                    "thinking_effort": (resolved or {}).get("effort")}
        final = state.get("final")
        if parser.handoff and final and str(final.get("text") or "").strip():
            content = clean_markup(final["text"])
            if on_delta:
                on_delta(content)
            return {"content": content, "reasoning_content": parser.reasoning, "finish_reason": "stop",
                    "usage": None, "model": final.get("model") or parser.model,
                    "thinking_effort": (resolved or {}).get("effort")}
        _answered(watch, asked, state)  # an in-app page that stopped running fails the request
        if parser.handoff and not state.get("open"):
            _check_reads(watch, state)  # an in-app page whose reads keep failing fails the request
        if parser.handoff and not state.get("open") and time.monotonic() - polled >= _HANDOFF_POLL_SECONDS:
            if not polled and on_thinking:
                on_thinking("ChatGPT is working on this in the background (Pro); waiting for its answer…\n")
            polled = time.monotonic()
            _read_answer(page, watch)
        sentinel = state.get("sentinel") or 0
        if sentinel and not state.get("streams"):
            raise AuthChatGPTError(f"AuthChatGPT: ChatGPT refused the request check (HTTP {sentinel}).",
                                   _kind(sentinel))
        # (An in-app page that stopped running starts no reply either: that is no usage limit, and
        # the watch fails it as paused. Time the app or the page was paused does not count.)
        if not state.get("streams") and watch.running_since(started) > 60 and not watch.paused():
            raise AuthChatGPTError("AuthChatGPT: ChatGPT did not start a reply. It may be showing a "
                                   "usage limit; try again later or click ChatGPT Login.", "rate_limit")
        # A reply that moved to another stream (/resume) is picked up there.
        if (state.get("streams") and not state.get("open") and state.get("idle", 0) > 30000
                and not parser.finished and not parser.handoff):
            raise AuthChatGPTError("AuthChatGPT stream ended before its completion marker. "
                                   "Partial translation discarded.")
        time.sleep(.1)
    raise AuthChatGPTError("AuthChatGPT translation timed out; partial translation discarded.", "timeout")


def _login_rejected(exc):
    """A request that failed because of the login (not the prompt or a limit)."""
    return exc.error_type == "auth_error" or (
        exc.error_type == "access_denied" and "unusual activity" in str(exc).lower())


def send_chat_completion(*, messages, model=DEFAULT_MODEL, timeout=300, log_fn=print,
                         cancel_check=None, on_delta=None, before_send_callback=None, on_thinking=None,
                         image_output=False):
    """One completion. ``image_output``: ask ChatGPT's image tool for an image (the result's
    ``images``); that needs the login and a regular chat (temporary chats cannot edit images)."""
    name = resolve_model(model)
    images = []
    prompt = build_prompt(messages, images=images)
    token = _access_token(log_fn)
    refreshed = False
    resends = 0
    sent = []

    def before_send():
        # once per request: a resend on a new page is the same request
        if not sent:
            sent.append(True)
            if before_send_callback:
                before_send_callback()

    while True:
        if images and not token:
            raise AuthChatGPTError(IMAGES_NEED_LOGIN, "config_error")
        if image_output and not token:
            raise AuthChatGPTError(IMAGE_OUTPUT_NEEDS_LOGIN, "config_error")
        config = {"token": token, "prompt": prompt, "model": name, "images": len(images),
                  "temporary": bool(token) and not image_output, "imageOutput": bool(image_output)}
        try:
            with _open_tab(config, cancel_check) as page:
                try:
                    result = _complete(page, float(timeout), cancel_check, on_delta, before_send, on_thinking,
                                       images=images, image_output=image_output, log_fn=log_fn)
                    if token:
                        _announce_logged_in(token, log_fn)
                    return result
                except _PageLost:
                    raise  # handled below, once the page is closed
                except AuthChatGPTError as exc:
                    if not token or not _login_rejected(exc):
                        raise
                    kind, reason = exc.error_type, str(exc).replace("AuthChatGPT: ", "")
        except _PageLost as exc:
            # The in-app page lost its hooked document while it loaded or answered (see _PageLost);
            # it is closed now, and the one-tab-loads-at-a-time lock is free again. The message goes
            # again on a new page instead of failing the request, up to MAX_PAGE_RESENDS times (on
            # the flet-webview fallback every screen change while it answers rebuilds the page).
            if resends >= MAX_PAGE_RESENDS or not getattr(exc.page, "driver", False):
                raise
            resends += 1
            log_fn(_page_lost_message(exc.page, resends))
            continue
        # The login failed this request: one refresh, then the chat runs logged out.
        replacement = None if refreshed or kind != "auth_error" else _refreshed_token(token)
        refreshed = True
        if replacement:
            _rejected.add(token)
            token = replacement
            continue
        _reject(token, reason, log_fn)
        token = None


def list_models(timeout=90, log_fn=print, cancel_check=None):
    """``authchatgpt/`` picker names of the account's (or logged-out) chatgpt.com picker."""
    token = _access_token(log_fn)
    while True:
        config = {"token": token, "prompt": "", "model": DEFAULT_MODEL, "temporary": bool(token)}
        with _open_tab(config, cancel_check) as page:
            page.evaluate(_LOAD_CATALOG)
            deadline = time.monotonic() + timeout
            error = None
            while time.monotonic() < deadline and not error:
                _check(cancel_check)
                state = page.evaluate(_CATALOG) or {}
                if state.get("catalog"):
                    names = state["catalog"].get("names") or {}
                    return [PREFIX + DEFAULT_MODEL] + [PREFIX + name for name in names]
                error = state.get("error")
                time.sleep(.3)
        if not error:
            raise AuthChatGPTError("AuthChatGPT timed out reading ChatGPT's models.", "timeout")
        if token and error == "HTTP 401":
            _reject(token, error, log_fn)
            token = None
            continue
        raise AuthChatGPTError(f"AuthChatGPT could not read ChatGPT's models ({error}).")
