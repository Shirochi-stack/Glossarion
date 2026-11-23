"""Google sign-in for authgem/ (the Gemini app) on the phone, with no manual step.

authgem/ sends the Gemini app's own requests from inside a signed-in gemini.google.com page (``authgem_auth``,
"authgem/ — the Gemini app"): on the phone a hidden page of the browser driver, which shares the WebView
cookie jar every in-app page uses. The cookies never leave that jar (Google ends a session whose cookies are
used elsewhere). Google refuses its sign-in page to agents that name an embedded web view, so the native
extension opens a full-screen sign-in page with the phone's browser agent (``web_sign_in``, the extension's
sign-in family), which fills that jar. This module runs the sign-in, remembers that it happened (authgem/'s
``has_web_session``) and signs out by expiring the jar's google.com cookies (``web_clear_cookies``).
``check_session`` reads the jar (``web_cookies``) to correct that memory.

Because the jar is shared, search/gemini's hidden AI Mode pages are signed in by the same sign-in.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Optional

log = logging.getLogger(__name__)

GEMINI_APP_URL = "https://gemini.google.com/app"
SIGN_IN_URL = ("https://accounts.google.com/ServiceLogin?hl=en"
               "&continue=https%3A%2F%2Fgemini.google.com%2Fapp")
DONE_URL = "https://gemini.google.com/"
DONE_COOKIE = "__Secure-1PSID"
COOKIE_URL = "https://www.google.com/"
COOKIE_DOMAIN = ".google.com"
TITLE = "Sign in to Google"
MARKER_NAME = "google_web_signed_in"
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

    def __init__(self, native: Any, dispatcher: Any, *, marker_dir: Optional[str] = None) -> None:
        self.native = native
        self.dispatcher = dispatcher
        self.marker = os.path.join(marker_dir or _marker_dir(), MARKER_NAME)

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

    async def check_session(self) -> bool:
        """Read the jar (loop thread): whether it holds a Google session; the saved hint follows it."""
        answer = await self.native.call("web_cookies", COOKIE_URL, default=None)
        if not isinstance(answer, dict) or not answer.get("ok"):
            return self.has_session()
        cookies = parse_cookie_header(answer.get("cookies"))
        signed = any(c["name"] in (DONE_COOKIE, "SID") and c["value"] for c in cookies)
        if signed != self.has_session():
            self._mark(signed)
        return signed


async def install(app: Any) -> Optional[GoogleSignIn]:
    """Register the phone's Google sign-in for authgem/ (loop thread), or None where the
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

    authgem_auth.set_web_session_probe(service.has_session)
    _current = service
    app.google_signin = service
    return service
