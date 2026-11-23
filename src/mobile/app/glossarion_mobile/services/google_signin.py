"""Google sign-in for authgem/ (the Gemini app) on the phone, with no manual step.

authgem/ sends the Gemini app's own requests signed in with the google.com session cookies
(``authgem_auth``, "authgem/ — the Gemini app"). Google refuses its sign-in page to agents that name an
embedded web view, so the native extension opens a full-screen sign-in page with the phone's browser agent
(``web_sign_in``, contract version 2 of ``flet_glossarion_native.web_pages``). The session then lives in the
WebView cookie jar every in-app page shares, and authgem/ reads it from there (``web_cookies``) whenever it
builds a session: first a hidden page visits gemini.google.com, so the jar holds the session cookies Google
just rotated (desktop does the same in its Chrome profile), then the cookies go to ``authgem_auth`` through
``set_web_cookie_provider``. Sign-out expires the jar's google.com cookies (``web_clear_cookies``).

Because the jar is shared, search/gemini's hidden AI Mode pages are signed in by the same sign-in.
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
import time
from typing import Any, Callable, Optional

log = logging.getLogger(__name__)

GEMINI_APP_URL = "https://gemini.google.com/app"
SIGN_IN_URL = ("https://accounts.google.com/ServiceLogin?hl=en"
               "&continue=https%3A%2F%2Fgemini.google.com%2Fapp")
DONE_URL = "https://gemini.google.com/"
DONE_COOKIE = "__Secure-1PSID"
COOKIE_URL = "https://www.google.com/"
COOKIE_DOMAIN = ".google.com"
TITLE = "Sign in to Google"
#: The google.com cookies authgem/ needs; the jar is read for these URLs.
COOKIE_URLS = (GEMINI_APP_URL, COOKIE_URL)
MARKER_NAME = "google_web_signed_in"
#: Seconds a worker waits for one cookie read, and for the hidden page that renews them.
READ_TIMEOUT = 20.0
RENEW_TIMEOUT = 20.0
UNAVAILABLE = "Google sign-in needs the in-app browser of the Android / iOS app."

_current: Optional["GoogleSignIn"] = None


def current() -> Optional["GoogleSignIn"]:
    return _current


def _marker_dir() -> str:
    try:
        import oauth_session

        return oauth_session.default_token_dir()
    except Exception:
        return os.path.join(os.path.expanduser("~"), ".glossarion")


def parse_cookie_header(text: Any) -> list:
    """``"a=1; b=2"`` -> ``[{name, value, domain, path}]`` (google.com cookies for ``authgem_auth``)."""
    cookies = []
    seen = set()
    for part in str(text or "").split(";"):
        name, sep, value = part.strip().partition("=")
        name = name.strip()
        if not sep or not name or name in seen:
            continue
        seen.add(name)
        cookies.append({"name": name, "value": value.strip(), "domain": COOKIE_DOMAIN, "path": "/"})
    return cookies


class GoogleSignIn:
    """The phone's Google sign-in: ``native`` is the app's ``NativeBridge``, ``dispatcher`` its UiDispatcher."""

    def __init__(self, native: Any, dispatcher: Any, *, marker_dir: Optional[str] = None,
                 page_driver: Optional[Callable[[], Any]] = None) -> None:
        self.native = native
        self.dispatcher = dispatcher
        self.marker = os.path.join(marker_dir or _marker_dir(), MARKER_NAME)
        self._page_driver = page_driver

    # ---- state -----------------------------------------------------------------------------------

    def has_session(self) -> bool:
        """A sign-in was saved (a hint: every authgem/ session checks the cookies itself)."""
        return os.path.exists(self.marker)

    def _mark(self, present: bool) -> None:
        try:
            if present:
                os.makedirs(os.path.dirname(self.marker), exist_ok=True)
                with open(self.marker, "w", encoding="utf-8") as handle:
                    handle.write(str(int(time.time())))
            elif os.path.exists(self.marker):
                os.remove(self.marker)
        except OSError as exc:
            log.warning("Google sign-in marker: %s", type(exc).__name__)

    @staticmethod
    def _reset_session() -> None:
        try:
            import authgem_auth

            authgem_auth.reset_web_session()
        except Exception:
            pass

    # ---- loop thread -----------------------------------------------------------------------------

    async def sign_in(self) -> bool:
        """The full-screen Google sign-in; True once signed in, False when the user cancelled it.
        Raises RuntimeError with the reason when the page could not open."""
        answer = await self.native.call("web_sign_in", SIGN_IN_URL, done_url=DONE_URL, done_cookie=DONE_COOKIE,
                                        cookie_url=COOKIE_URL, title=TITLE, default=None)
        if not isinstance(answer, dict) or not answer.get("ok"):
            reason = (answer or {}).get("message") if isinstance(answer, dict) else None
            raise RuntimeError(reason or UNAVAILABLE)
        signed = answer.get("signed_in") is True
        if signed:
            self._mark(True)
            self._reset_session()
        return signed

    async def sign_out(self) -> None:
        """Expire the jar's google.com cookies and forget the sign-in."""
        self._mark(False)
        self._reset_session()
        await self.native.call("web_clear_cookies", COOKIE_URL, COOKIE_DOMAIN, default=None)

    # ---- worker threads (authgem_auth's cookie provider) ------------------------------------------

    def _renew(self, cancel_check: Optional[Callable[[], bool]]) -> None:
        """Best effort: a hidden page loads the Gemini app so Google's own scripts renew the rotating
        session cookies in the jar before they are read."""
        driver = None
        try:
            if self._page_driver is not None:
                driver = self._page_driver()
            else:
                from glossarion_mobile.services import webview_bridge

                driver = webview_bridge.current()
        except Exception:
            driver = None
        if driver is None:
            return
        page = None
        try:
            import browser_driver

            page = driver.open_page(owner="authgem", cancel_check=cancel_check)
            page.load(GEMINI_APP_URL)
            browser_driver.wait_until_loaded(page, RENEW_TIMEOUT)
            page.wait(1500)
        except Exception as exc:
            log.debug("Google cookie renewal skipped: %s", type(exc).__name__)
        finally:
            if page is not None:
                try:
                    page.close()
                except Exception:
                    pass

    def _read(self, url: str) -> str:
        future = self.dispatcher.submit(self.native.call, "web_cookies", url)
        if future is None:
            raise RuntimeError("the app's UI loop is not running")
        try:
            answer = future.result(timeout=READ_TIMEOUT)
        except concurrent.futures.TimeoutError as exc:
            raise RuntimeError("reading the Google sign-in timed out") from exc
        if not isinstance(answer, dict) or not answer.get("ok"):
            reason = answer.get("message") if isinstance(answer, dict) else None
            raise RuntimeError(reason or UNAVAILABLE)
        return str(answer.get("cookies") or "")

    def cookies(self, cancel_check: Optional[Callable[[], bool]] = None) -> list:
        """``authgem_auth`` cookie provider: the jar's google.com cookies (worker thread)."""
        self._renew(cancel_check)
        merged: dict = {}
        for url in COOKIE_URLS:
            for cookie in parse_cookie_header(self._read(url)):
                merged.setdefault(cookie["name"], cookie)
        cookies = list(merged.values())
        signed = any(c["name"] in (DONE_COOKIE, "SID") and c["value"] for c in cookies)
        if signed != self.has_session():
            self._mark(signed)
        return cookies


async def install(app: Any) -> Optional[GoogleSignIn]:
    """Register the phone's Google sign-in as authgem/'s cookie source (loop thread), or None where the
    native extension has no sign-in (desktop, ``flet run``, an older build without ``web_pages["sign_in"]``)."""
    global _current
    native = getattr(app, "native", None)
    dispatcher = getattr(app, "dispatcher", None)
    if native is None or dispatcher is None or getattr(native, "is_stub", True):
        return None
    try:
        from flet_glossarion_native.web_pages import capability

        info = await native.platform_info()
        found = capability(info)
        if found is None or found.get("sign_in") is not True:
            log.info("Google sign-in: this build's in-app browser has no sign-in page")
            return None
    except Exception as exc:
        log.info("Google sign-in unavailable: %s", type(exc).__name__)
        return None
    service = GoogleSignIn(native, dispatcher)
    import authgem_auth

    authgem_auth.set_web_cookie_provider(service.cookies, has_session=service.has_session)
    _current = service
    app.google_signin = service
    return service
