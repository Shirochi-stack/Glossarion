"""OAuthBridge: ChatGPT (AuthGPT) sign-in on the phone (plan §4, UI_SPEC §4.13, §4.17).

The default model is ``authgpt/gpt-6-luna`` (desktop default), so sign-in ships in the
MVP. Nothing here re-implements OAuth: it drives the shared ``authgpt_auth`` split
added in U1,

* ``begin_oauth(store=..., account_id=...)`` - PKCE + state, loopback server on
  ``localhost:1455`` (IPv4 + IPv6), pending state persisted encrypted next to the
  token file for the paste fallback;
* ``complete_from_redirect(session_or_store, url_or_code)`` - state check, code
  exchange, tokens saved to the token store (``HOME/.glossarion``, encrypted with
  the SecureStorage key installed at boot).

Flow (``LoginSheet`` steps Opening browser -> Waiting for sign-in -> Exchanging token
-> Done):

1. ``begin_oauth`` on a worker thread; ``webbrowser.open(auth_url)``, which the
   bootstrap routes to ``UrlLauncher.launch_url(mode=IN_APP_BROWSER_VIEW)`` (Custom
   Tabs / SFSafariViewController);
2. on Android a short "Signing in…" foreground service keeps the process (and the
   loopback server) alive while the browser is in front - only when no job service
   is already running;
3. the loopback success page redirects to ``glossarion://app/oauth/return?p=authgpt``
   (``GLOSSARION_OAUTH_RETURN_URL``); the app routes it to ``on_return_link`` which
   closes the iOS in-app browser;
4. the waiter sees the callback and exchanges the code; or the user taps **Paste
   redirect URL / code** and ``complete_with_paste`` finishes from the saved pending
   state (works after the app or the loopback server was killed).

``state`` is a ``SignInState`` (observable; listeners are marshalled with ``post``).
Pure Python (3.10); no Flet import. ``auth_module`` and ``opener`` are injectable.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import webbrowser
from dataclasses import dataclass, replace
from typing import Any, Callable, Optional

__all__ = ["OAuthBridge", "PROVIDERS", "SignInState", "STEP_LABELS", "provider_for_model"]

log = logging.getLogger("glossarion.oauth")

DEFAULT_TIMEOUT = 300  # seconds (authgpt run_oauth_flow default)

#: Providers on the Accounts screen. Only ChatGPT signs in from U3; the rest arrive in U4.
PROVIDERS = (
    ("authgpt", "ChatGPT", None),
    ("authgem", "Gemini", "Coming in U4"),
    ("authcd", "Claude", "Coming in U4"),
    ("authgrok", "Grok", "Coming in U4"),
)

STEP_LABELS = {
    "idle": "",
    "opening": "Opening browser…",
    "waiting": "Waiting for sign-in…",
    "exchanging": "Exchanging token…",
    "done": "Signed in",
    "error": "Sign-in failed",
    "cancelled": "Sign-in cancelled",
}


def provider_for_model(model: Optional[str]) -> Optional[tuple]:
    """``("authgpt", account_id)`` for ``authgpt/…`` / ``authgpt2/…`` models, else None."""
    prefix = str(model or "").split("/", 1)[0].lower()
    if prefix.startswith("authgpt"):
        rest = prefix[7:]
        if rest == "":
            return ("authgpt", 0)
        if rest.isdigit():
            return ("authgpt", int(rest))
    return None


@dataclass(frozen=True)
class SignInState:
    provider: str = "authgpt"
    step: str = "idle"  # idle | opening | waiting | exchanging | done | error | cancelled
    message: str = ""
    account_id: int = 0
    auth_url: str = ""
    email: str = ""
    plan: str = ""

    @property
    def busy(self) -> bool:
        return self.step in ("opening", "waiting", "exchanging")

    @property
    def label(self) -> str:
        return STEP_LABELS.get(self.step, self.step)


class OAuthBridge:
    def __init__(
        self,
        *,
        auth_module: Any = None,
        opener: Optional[Callable[[str], Any]] = None,
        close_browser: Optional[Callable[[], Any]] = None,
        native: Any = None,
        post: Optional[Callable[..., Any]] = None,
        run_io: Optional[Callable[..., Any]] = None,
        timeout: int = DEFAULT_TIMEOUT,
        is_android: bool = False,
    ) -> None:
        self._auth = auth_module
        self.opener = opener or webbrowser.open
        self.close_browser = close_browser
        self.native = native
        self.post = post
        self.run_io = run_io
        self.timeout = timeout
        self.is_android = is_android
        self.state = SignInState()
        self._listeners: list = []
        self._session: Any = None
        self._cancel = threading.Event()
        self._lock = threading.RLock()
        self._fgs_started = False
        self._pasting = False
        self._paste_claimed = False
        self.signed_in: set = set()

    # ---- plumbing ------------------------------------------------------------------------

    @property
    def auth(self) -> Any:
        if self._auth is None:
            import authgpt_auth  # backend module (on sys.path after bootstrap)

            self._auth = authgpt_auth
        return self._auth

    def subscribe(self, callback: Callable[[SignInState], Any]) -> Callable[[], None]:
        self._listeners.append(callback)

        def unsubscribe() -> None:
            try:
                self._listeners.remove(callback)
            except ValueError:
                pass

        return unsubscribe

    def _set(self, **changes: Any) -> SignInState:
        with self._lock:
            self.state = replace(self.state, **changes)
            state = self.state
        for callback in list(self._listeners):
            try:
                if self.post is not None:
                    self.post(callback, state)
                else:
                    callback(state)
            except Exception:
                log.exception("sign-in listener failed")
        return state

    async def _io(self, fn: Callable[..., Any], *args: Any) -> Any:
        if self.run_io is not None:
            return await self.run_io(fn, *args)
        return await asyncio.to_thread(fn, *args)

    def store(self, account_id: int = 0) -> Any:
        return self.auth.get_store(account_id or None)

    # ---- status -------------------------------------------------------------------------------

    def status(self, account_id: int = 0) -> dict:
        """Blocking (decrypts the token file): ``{signed_in, email, plan, expires_at}``."""
        try:
            store = self.store(account_id)
            tokens = store.load_tokens() or {}
            info = store.account_info if tokens.get("id_token") else {}
        except Exception as exc:
            log.warning("reading the ChatGPT token store failed: %s", exc)
            return {"signed_in": False, "email": "", "plan": "", "expires_at": None, "error": str(exc)}
        signed = bool(tokens.get("access_token") or tokens.get("refresh_token"))
        with self._lock:
            key = "authgpt" if not account_id else f"authgpt{account_id}"
            if signed:
                self.signed_in.add(key)
            else:
                self.signed_in.discard(key)
        return {
            "signed_in": signed,
            "email": str(info.get("email") or ""),
            "plan": str(info.get("plan_type") or ""),
            "expires_at": tokens.get("expires_at"),
        }

    async def refresh_status(self, account_id: int = 0) -> dict:
        return await self._io(self.status, account_id)

    def has_pending(self, account_id: int = 0) -> bool:
        if self._session is not None:
            return True
        try:
            return bool(self.store(account_id).load_pending_oauth())
        except Exception:
            return False

    # ---- sign in ------------------------------------------------------------------------------

    async def sign_in(self, provider: str = "authgpt", account_id: int = 0) -> dict:
        """Run the browser sign-in; returns the account status ({} when cancelled or
        finished by a paste). Raises on failure (the state then reads ``error``)."""
        if provider != "authgpt":
            raise RuntimeError(f"{provider} sign-in arrives in U4")
        if self.state.busy:
            raise RuntimeError("A sign-in is already in progress")
        self._cancel.clear()
        self._pasting = False
        self._paste_claimed = False
        self._set(provider=provider, step="opening", message="", account_id=int(account_id or 0), auth_url="")
        try:
            # A previous sign-in finished by a failed paste keeps its loopback server on
            # localhost:1455 until its watchdog fires; free the port before listening again.
            # (The pending state stays in the token store, so pasting still works after this.)
            await self._io(self._close_session)
            store = self.store(account_id)
            session = await self._io(lambda: self.auth.begin_oauth(store=store, account_id=account_id or None, timeout=self.timeout))
        except Exception as exc:
            self._set(step="error", message=str(exc))
            raise
        with self._lock:
            self._session = session
        await self._start_fgs()
        self._set(step="waiting", auth_url=session.auth_url)
        self._open(session.auth_url)
        try:
            received = await self._io(self._wait_for_callback, session)
            if self._cancel.is_set():
                if self._paste_claimed:
                    return {}  # complete_with_paste finishes (and reports) this sign-in
                self._close_session()
                self._set(step="cancelled", message="")
                await self._stop_fgs()
                return {}
            if not received:
                raise RuntimeError("OAuth login timed out – no callback received.")
            self._set(step="exchanging")
            await self._io(self.auth.complete_from_redirect, session)
        except Exception as exc:
            self._close_session()
            self._set(step="error", message=str(exc))
            await self._stop_fgs()
            raise
        return await self._finished(account_id)

    def _wait_for_callback(self, session: Any) -> bool:
        deadline = time.monotonic() + self.timeout
        while not self._cancel.is_set():
            if session.wait_for_callback(0.25):
                return True
            if time.monotonic() >= deadline:
                return False
            if not getattr(session, "server_running", True) and not getattr(session, "callback_received", False):
                # Listener stopped (timeout watchdog / killed): only the paste path can finish now.
                return bool(getattr(session, "callback_received", False))
        return False

    async def _finished(self, account_id: int) -> dict:
        self._close_session()
        status = await self._io(self.status, account_id)
        self._set(step="done", message="", email=status.get("email", ""), plan=status.get("plan", ""))
        await self._stop_fgs()
        return status

    def _open(self, url: str) -> None:
        try:
            self.opener(url)
        except Exception as exc:
            log.warning("opening the sign-in page failed: %s", exc)

    def reopen_browser(self) -> bool:
        url = self.state.auth_url or getattr(self._session, "auth_url", "")
        if not url:
            return False
        self._open(url)
        return True

    async def complete_with_paste(self, text: str, account_id: Optional[int] = None) -> dict:
        """Finish from a pasted redirect URL, ``code#state`` or bare code."""
        value = str(text or "").strip()
        if not value:
            raise RuntimeError("Paste the redirect URL (or the code) to finish the ChatGPT sign-in.")
        if account_id is None:
            account_id = self.state.account_id
        with self._lock:
            session = self._session
            self._pasting = True
            self._paste_claimed = True
        self._cancel.set()  # stop the loopback waiter; the paste wins
        self._set(step="exchanging", message="")
        target = session if session is not None else self.store(account_id)
        try:
            await self._io(self.auth.complete_from_redirect, target, value)
        except Exception as exc:
            self._pasting = False
            self._set(step="error", message=str(exc))
            await self._stop_fgs()
            raise
        self._pasting = False
        return await self._finished(account_id)

    def cancel(self) -> None:
        self._cancel.set()
        self._close_session()
        if self.state.busy:
            self._set(step="cancelled", message="")

    def _close_session(self) -> None:
        with self._lock:
            session, self._session = self._session, None
        if session is not None:
            try:
                session.close()
            except Exception:
                pass

    def on_return_link(self, provider: Optional[str] = None) -> None:
        """``glossarion://app/oauth/return?p=authgpt``: close the iOS in-app browser.

        The waiter finishes on its own once the loopback server recorded the callback.
        """
        if self.close_browser is not None:
            try:
                result = self.close_browser()
                if asyncio.iscoroutine(result):
                    asyncio.ensure_future(result)
            except Exception:
                log.debug("closing the in-app browser failed", exc_info=True)

    async def sign_out(self, account_id: int = 0) -> None:
        def clear() -> None:
            store = self.store(account_id)
            store.clear_tokens()
            if hasattr(store, "clear_pending_oauth"):
                store.clear_pending_oauth()

        await self._io(clear)
        with self._lock:
            self.signed_in.discard("authgpt" if not account_id else f"authgpt{account_id}")
        self._set(step="idle", message="", email="", plan="")

    # ---- Android short sign-in FGS -------------------------------------------------------------

    async def _start_fgs(self) -> None:
        if not self.is_android or self.native is None:
            return
        try:
            if await self.native.is_job_service_running():
                return  # a translation already holds the foreground service
            self._fgs_started = bool(await self.native.start_job_service("Signing in to ChatGPT", "Waiting for sign-in…"))
        except Exception as exc:
            log.info("sign-in foreground service unavailable: %s", exc)

    async def _stop_fgs(self) -> None:
        if not self._fgs_started or self.native is None:
            return
        self._fgs_started = False
        try:
            await self.native.stop_job_service()
        except Exception:
            pass
