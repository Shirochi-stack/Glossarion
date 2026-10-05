# authgpt_auth.py - ChatGPT Plus/Pro subscription OAuth authentication
# Merged from authgpt/{oauth,token_store,chatgpt_api}.py
# Uses the same OAuth PKCE flow as OpenAI's Codex CLI / OpenCode plugin.
# Prefix models with 'authgpt/' to route through the ChatGPT backend API
# using your ChatGPT subscription instead of OpenAI Platform API credits.
"""
OAuth 2.0 PKCE flow for ChatGPT subscription authentication, persistent
token storage with automatic refresh, and ChatGPT backend API adapter.

Flow:
  1. Generate PKCE code_verifier + code_challenge
  2. Open browser to auth.openai.com/oauth/authorize
  3. Spin up a local HTTP callback server (port 1455)
  4. User logs in via browser → callback receives auth code
  5. Exchange auth code for access + refresh tokens
  6. Store tokens locally (~/.glossarion/authgpt_tokens.json)

run_oauth_flow() runs all of it. begin_oauth() / complete_from_redirect()
split it for callers that open the browser themselves (Glossarion Mobile)
and add a paste-the-redirect-URL fallback.
"""
import os
import json
import re
import time
import hashlib
import base64
import secrets
import logging
import socket
import sys
import threading
import webbrowser
from html import escape as _html_escape
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlencode, urlparse, parse_qs
from typing import Optional, Dict, List, Tuple, Any

import requests
import oauth_session
from app_version import APP_VERSION
from reasoning_compatibility import normalize_none_effort, call_with_reasoning_retry

logger = logging.getLogger(__name__)

# Module-level cancellation flag — set by unified_api_client.hard_cancel_all()
_cancel_event = threading.Event()

def cancel_stream():
    """Signal any active AuthGPT stream to abort immediately."""
    _cancel_event.set()

def reset_cancel():
    """Clear the cancellation flag (call before starting a new request).

    Refuse to clear while a hard stop is active: this is called per-attempt and
    per-chapter (reset_cleanup_state), so a just-starting worker must not wipe
    the cancel signal the Stop button set. TRANSLATION_CANCELLED is only set on
    an immediate stop (never graceful), so honoring it here is safe.
    """
    if os.environ.get("TRANSLATION_CANCELLED") == "1":
        return
    _cancel_event.clear()

def is_cancelled() -> bool:
    # Also honor the hard-abort env var so a racing reset_cancel() can't let an
    # in-flight browser-backed request bypass Stop.
    return _cancel_event.is_set() or os.environ.get("TRANSLATION_CANCELLED") == "1"

# ===========================================================================
# Constants – mirror the values used by OpenAI's Codex CLI / OpenCode plugin
# ===========================================================================
OPENAI_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
OPENAI_AUTH_URL = "https://auth.openai.com/oauth/authorize"
OPENAI_TOKEN_URL = "https://auth.openai.com/oauth/token"
CALLBACK_HOST = "localhost"
CALLBACK_PORT = 1455  # Fixed – OpenAI's Codex client ID only accepts this port
CALLBACK_PATH = "/auth/callback"
SCOPES = "openid profile email offline_access"
TOKEN_REFRESH_MARGIN_SECONDS = 300  # refresh when <5 min remaining
PENDING_OAUTH_MAX_AGE_SECONDS = 3600  # saved PKCE state for the paste fallback

CHATGPT_BASE_URL = "https://chatgpt.com/backend-api"
RESPONSES_ENDPOINT = "/codex/responses"
ACCOUNT_MODELS_URL = "https://api.openai.com/v1/models"

# ~/.glossarion, or GLOSSARION_TOKEN_DIR (set only by Glossarion Mobile before any backend import)
_DEFAULT_TOKEN_DIR = oauth_session.default_token_dir()
_DEFAULT_TOKEN_FILE = os.path.join(_DEFAULT_TOKEN_DIR, "authgpt_tokens.json")


# ===========================================================================
# PKCE helpers
# ===========================================================================

def generate_pkce() -> Tuple[str, str]:
    """Generate PKCE code_verifier and code_challenge (S256).

    Returns (code_verifier, code_challenge).
    """
    raw = secrets.token_bytes(32)
    code_verifier = base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")
    digest = hashlib.sha256(code_verifier.encode("ascii")).digest()
    code_challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return code_verifier, code_challenge


def build_auth_url(code_challenge: str, state: str, redirect_uri: str) -> str:
    """Build the full authorization URL."""
    params = {
        "client_id": OPENAI_CLIENT_ID,
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "scope": SCOPES,
        "state": state,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "audience": "https://api.openai.com/v1",
    }
    return f"{OPENAI_AUTH_URL}?{urlencode(params)}"


# ===========================================================================
# Token exchange / refresh
# ===========================================================================

def exchange_code_for_tokens(
    auth_code: str,
    code_verifier: str,
    redirect_uri: str,
) -> Dict:
    """Exchange authorization code for access + refresh tokens.

    Returns dict with keys: access_token, refresh_token, expires_in, id_token, …
    """
    payload = {
        "grant_type": "authorization_code",
        "client_id": OPENAI_CLIENT_ID,
        "code": auth_code,
        "redirect_uri": redirect_uri,
        "code_verifier": code_verifier,
    }
    resp = requests.post(OPENAI_TOKEN_URL, data=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    # Attach an absolute expiry timestamp for convenience
    data["expires_at"] = time.time() + data.get("expires_in", 3600)
    return data


def refresh_access_token(refresh_token: str) -> Dict:
    """Use a refresh token to obtain a new access token.

    Returns the same dict shape as ``exchange_code_for_tokens``.
    """
    payload = {
        "grant_type": "refresh_token",
        "client_id": OPENAI_CLIENT_ID,
        "refresh_token": refresh_token,
    }
    resp = requests.post(OPENAI_TOKEN_URL, data=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    data["expires_at"] = time.time() + data.get("expires_in", 3600)
    return data


# ===========================================================================
# JWT helpers (lightweight, no external dependency)
# ===========================================================================

def _decode_jwt_payload(token: str) -> Optional[Dict]:
    """Decode the payload of a JWT **without** verifying the signature.

    Only used for extracting ChatGPT account info (e.g. plan type) from the
    id_token.  Security-critical validation is left to the server.
    """
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        payload_b64 = parts[1]
        # Pad to 4-byte boundary
        payload_b64 += "=" * ((4 - len(payload_b64) % 4) % 4)
        decoded = base64.urlsafe_b64decode(payload_b64)
        return json.loads(decoded)
    except Exception:
        return None


def extract_account_info(id_token: str) -> Dict:
    """Extract ChatGPT account info from the id_token JWT."""
    claims = _decode_jwt_payload(id_token) or {}
    auth_claims = claims.get("https://api.openai.com/auth", {})
    return {
        "chatgpt_account_id": auth_claims.get("chatgpt_account_id", ""),
        "plan_type": auth_claims.get("chatgpt_plan_type", ""),
        "email": claims.get("email", ""),
    }


# ===========================================================================
# Local callback server
# ===========================================================================

def _oauth_return_target(return_url: str) -> str:
    """'<return_url>?p=authgpt' ('&p=authgpt' when it already has a query)."""
    separator = "&" if "?" in return_url else "?"
    return f"{return_url}{separator}p=authgpt"


def _oauth_success_html() -> str:
    """HTML of the loopback /success page.

    When GLOSSARION_OAUTH_RETURN_URL is set (Glossarion Mobile) the page also
    sends the user back to the app: a button plus an automatic redirect to
    '<return_url>?p=authgpt'. Desktop leaves it unset and gets the original page.
    """
    return_url = os.environ.get("GLOSSARION_OAUTH_RETURN_URL", "").strip()
    if not return_url:
        return (
            "<html><body style='font-family:sans-serif;text-align:center;padding-top:60px'>"
            "<h1>&#10004; Authenticated!</h1>"
            "<p>You can close this tab and return to Glossarion.</p>"
            "</body></html>"
        )
    target = _html_escape(_oauth_return_target(return_url), quote=True)
    return (
        "<html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'></head>"
        "<body style='font-family:sans-serif;text-align:center;padding-top:60px'>"
        "<h1>&#10004; Authenticated!</h1>"
        "<p>Returning to Glossarion&hellip;</p>"
        f"<p><a id='glossarion-return' href='{target}' "
        "style='display:inline-block;margin-top:16px;padding:12px 20px;border-radius:20px;"
        "background:#E18F98;color:#121826;text-decoration:none;font-weight:bold'>"
        "Return to Glossarion</a></p>"
        "<script>setTimeout(function(){location.href="
        "document.getElementById('glossarion-return').href;},300);</script>"
        "</body></html>"
    )


class _OAuthCallbackHandler(BaseHTTPRequestHandler):
    """HTTP request handler that captures the OAuth callback."""

    # Shared across instances via the server reference
    auth_code: Optional[str] = None
    returned_state: Optional[str] = None
    error: Optional[str] = None

    def log_message(self, format, *args):  # noqa: A002
        # Suppress default stderr logging
        pass

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == CALLBACK_PATH:
            qs = parse_qs(parsed.query)
            self.server._auth_code = qs.get("code", [None])[0]
            self.server._returned_state = qs.get("state", [None])[0]
            self.server._error = qs.get("error", [None])[0]
            session = getattr(self.server, "_oauth_session", None)
            if session is not None:
                session._record_callback(
                    self.server._auth_code, self.server._returned_state, self.server._error
                )

            # Redirect to a friendly success page
            self.send_response(302)
            self.send_header("Location", f"http://{CALLBACK_HOST}:{self.server.server_port}/success")
            self.end_headers()

        elif parsed.path == "/success":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            html = _oauth_success_html()
            self.wfile.write(html.encode("utf-8"))
            # Signal the server to stop (handled in a thread below)
            session = getattr(self.server, "_oauth_session", None)
            stop = session.close if session is not None else self.server.shutdown
            threading.Thread(target=stop, daemon=True).start()

        else:
            self.send_response(404)
            self.end_headers()


def _is_mobile() -> bool:
    """mobile_runtime.is_mobile(); False when the shared module is unavailable."""
    if sys.platform in ("ios", "android"):
        return True
    try:
        import mobile_runtime
        return mobile_runtime.is_mobile()
    except Exception:
        return False


def _find_available_port() -> int:
    """Return the callback port for OAuth.

    OpenAI's Codex client ID (app_EMoamEEZ73f0CkXaXp7hrann) has a fixed
    redirect URI registered as http://localhost:1455/auth/callback.
    We *must* use port 1455 or OpenAI will reject the request.
    """
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if os.name != "nt" and _is_mobile():
                # Android/iOS: a sign-in that just finished leaves TIME_WAIT
                # sockets on the port, which fail a plain bind for ~60 s. The
                # callback server binds with SO_REUSEADDR anyway; a socket
                # still listening on the port keeps failing this probe.
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((CALLBACK_HOST, CALLBACK_PORT))
            return CALLBACK_PORT
    except OSError:
        raise RuntimeError(
            f"Port {CALLBACK_PORT} is already in use. "
            "Please close any application using that port and try again. "
            "(OpenAI's OAuth requires this exact port.)"
        )


class _IPv6OAuthCallbackServer(HTTPServer):
    address_family = socket.AF_INET6


def _make_ipv6_loopback_server(port: int, timeout: int) -> Optional[HTTPServer]:
    """A second callback listener on [::1]:*port*, or None when IPv6 is unavailable.

    Mobile browsers may resolve 'localhost' to ::1 before 127.0.0.1.
    """
    if not getattr(socket, "has_ipv6", False):
        return None
    try:
        server = _IPv6OAuthCallbackServer(("::1", port), _OAuthCallbackHandler)
    except Exception:
        return None
    server._auth_code = None
    server._returned_state = None
    server._error = None
    server.timeout = timeout
    return server


# ===========================================================================
# OAuth flow orchestrator
# ===========================================================================

def _shutdown_quietly(server: HTTPServer) -> None:
    try:
        server.shutdown()
    except Exception:
        pass


class OAuthSession:
    """A ChatGPT sign-in started by begin_oauth(), waiting for its redirect.

    The caller opens ``auth_url`` in a browser. When the browser is sent to
    ``redirect_uri``, the loopback server fills ``auth_code``,
    ``returned_state`` and ``error``; complete_from_redirect() then checks
    the state and exchanges the code. ``pending_state()`` is what
    begin_oauth() persists for the paste-the-redirect fallback.
    """

    def __init__(
        self,
        auth_url: str,
        code_verifier: str,
        state: str,
        redirect_uri: str,
        store: Optional["AuthGPTTokenStore"] = None,
        account_id: Optional[int] = None,
        timeout: int = 300,
    ):
        self.auth_url = auth_url
        self.code_verifier = code_verifier
        self.state = state
        self.redirect_uri = redirect_uri
        self.store = store
        self.account_id = account_id
        self.timeout = timeout
        self.created_at = time.time()
        self.auth_code: Optional[str] = None
        self.returned_state: Optional[str] = None
        self.error: Optional[str] = None
        self._servers: List[HTTPServer] = []
        self._serving = 0
        self._closed = False
        self._watchdog: Optional[threading.Timer] = None
        self._lock = threading.Lock()
        self._callback_event = threading.Event()
        self._done_event = threading.Event()
        self._close_finished = threading.Event()

    @property
    def server_running(self) -> bool:
        """True while a loopback listener is serving."""
        with self._lock:
            return self._serving > 0 and not self._closed

    @property
    def callback_received(self) -> bool:
        """True once the loopback server received the redirect (code or error)."""
        return self._callback_event.is_set()

    def pending_state(self) -> Dict:
        """What complete_from_redirect() needs to finish this sign-in later."""
        return {
            "provider": "authgpt",
            "auth_url": self.auth_url,
            "code_verifier": self.code_verifier,
            "state": self.state,
            "redirect_uri": self.redirect_uri,
            "account_id": self.account_id,
            "created_at": self.created_at,
        }

    def wait_for_callback(self, timeout: Optional[float] = None) -> bool:
        """Block until the loopback server received the redirect; False on timeout."""
        return self._callback_event.wait(timeout)

    def wait(self, timeout: Optional[float] = None) -> bool:
        """Block until the success page was served or the listeners stopped."""
        return self._done_event.wait(timeout)

    def close(self) -> None:
        """Stop the loopback listener(s) and free the port. Safe to call repeatedly.

        A second caller returns once the first one has released the port.
        """
        with self._lock:
            already_closing = self._closed
            self._closed = True
            servers = list(self._servers)
            watchdog, self._watchdog = self._watchdog, None
        if already_closing:
            self._close_finished.wait()
            return
        if watchdog is not None:
            watchdog.cancel()
        # shutdown() waits up to one poll interval per listener; stop them in parallel.
        stoppers = [
            threading.Thread(target=_shutdown_quietly, args=(server,), daemon=True)
            for server in servers
            if getattr(server, "_glossarion_serving", False)
        ]
        for stopper in stoppers:
            stopper.start()
        for stopper in stoppers:
            stopper.join()
        for server in servers:
            try:
                server.server_close()
            except Exception:
                pass
        self._done_event.set()
        self._close_finished.set()

    def _add_server(self, server: HTTPServer) -> None:
        server._oauth_session = self
        self._servers.append(server)

    def _start(self, auto_close: bool) -> None:
        for server in self._servers:
            thread = threading.Thread(
                target=self._serve, args=(server,), daemon=True, name="authgpt-oauth-callback"
            )
            with self._lock:
                self._serving += 1
            server._glossarion_serving = True
            try:
                thread.start()
            except BaseException:
                server._glossarion_serving = False
                with self._lock:
                    self._serving -= 1
                raise
        if auto_close and self.timeout:
            watchdog = threading.Timer(self.timeout, self.close)
            watchdog.daemon = True
            with self._lock:
                self._watchdog = watchdog
            watchdog.start()

    def _serve(self, server: HTTPServer) -> None:
        try:
            server.serve_forever()
        finally:
            with self._lock:
                self._serving -= 1
                finished = self._serving <= 0
            if finished:
                self._done_event.set()

    def _record_callback(self, code: Optional[str], state: Optional[str], error: Optional[str]) -> None:
        with self._lock:
            self.auth_code = code
            self.returned_state = state
            self.error = error
        self._callback_event.set()


def begin_oauth(
    store: Optional["AuthGPTTokenStore"] = None,
    account_id: Optional[int] = None,
    timeout: int = 300,
    persist: bool = True,
    dual_stack: bool = True,
    auto_close: bool = True,
) -> OAuthSession:
    """Start a ChatGPT sign-in without opening a browser.

    Generates the PKCE verifier/challenge and state, starts the loopback
    callback server on localhost:1455 (127.0.0.1, plus ::1 when *dual_stack*
    and IPv6 are available) and returns an :class:`OAuthSession`. The caller
    opens ``session.auth_url`` itself (Glossarion Mobile: Custom Tabs /
    SFSafariViewController) and then calls :func:`complete_from_redirect`.

    *store* (or ``get_store(account_id)`` when only *account_id* is given)
    receives the tokens on completion. With *persist*, the pending verifier
    and state are saved encrypted next to the store's token file, so a pasted
    redirect URL still completes the sign-in after the loopback server or the
    app was killed. *auto_close* stops the listener after *timeout* seconds.

    Raises RuntimeError when port 1455 is in use.
    """
    if store is None and account_id is not None:
        store = get_store(account_id)
    if account_id is None and store is not None:
        account_id = getattr(store, "_account_id", None)

    port = _find_available_port()
    redirect_uri = f"http://{CALLBACK_HOST}:{port}{CALLBACK_PATH}"
    code_verifier, code_challenge = generate_pkce()
    state = secrets.token_urlsafe(32)

    auth_url = build_auth_url(code_challenge, state, redirect_uri)
    session = OAuthSession(
        auth_url, code_verifier, state, redirect_uri,
        store=store, account_id=account_id, timeout=timeout,
    )

    try:
        # Start local callback server
        server = HTTPServer((CALLBACK_HOST, port), _OAuthCallbackHandler)
        server._auth_code = None
        server._returned_state = None
        server._error = None
        server.timeout = timeout
        session._add_server(server)
        if dual_stack:
            server_v6 = _make_ipv6_loopback_server(port, timeout)
            if server_v6 is not None:
                session._add_server(server_v6)
        session._start(auto_close)
    except BaseException:
        session.close()
        raise

    if persist and store is not None and hasattr(store, "save_pending_oauth"):
        store.save_pending_oauth(session.pending_state())
    return session


def _parse_oauth_redirect(value: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """(code, state, error) from a pasted redirect URL, its query string,
    'code#state', or a bare authorization code."""
    text = str(value or "").strip()
    if not text:
        return None, None, None

    def from_query(query: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        qs = parse_qs(query.lstrip("?#"))
        return qs.get("code", [None])[0], qs.get("state", [None])[0], qs.get("error", [None])[0]

    if "?" in text:
        query, _, fragment = text.split("?", 1)[1].partition("#")
        code, state, error = from_query(query)
        if not code and not error and "=" in fragment:
            code, state, error = from_query(fragment)
        return code, state, error
    if "=" in text:
        if "://" in text and "#" in text:
            return from_query(text.split("#", 1)[1])
        return from_query(text)
    if "#" in text:
        code, _, state = text.partition("#")
        return code.strip() or None, state.strip() or None, None
    return text, None, None


def complete_from_redirect(
    session_or_saved_state: Any = None,
    redirect_url_or_code: Optional[str] = None,
    store: Optional["AuthGPTTokenStore"] = None,
) -> Dict:
    """Finish a sign-in started by :func:`begin_oauth`: check, exchange, save.

    *session_or_saved_state* is the :class:`OAuthSession`, a saved pending
    dict (``OAuthSession.pending_state()`` / ``store.load_pending_oauth()``),
    or a token store whose persisted pending sign-in is used (paste fallback
    after the loopback server or the app was killed).

    *redirect_url_or_code* is None to use what the loopback server captured
    (session only), else the pasted redirect URL
    (``http://localhost:1455/auth/callback?code=...&state=...``), its query
    string, ``code#state`` or the bare code. A state that is present must
    match; a bare code is still bound to this sign-in by the PKCE verifier.

    Tokens are saved to *store* (default: the session's store) and the
    pending state is cleared; with no store they are only returned
    (run_oauth_flow's callers save them). Raises RuntimeError on an OAuth
    error, a missing code, a state mismatch or when nothing is pending.
    """
    session: Optional[OAuthSession] = None
    if isinstance(session_or_saved_state, OAuthSession):
        session = session_or_saved_state
        if store is None:
            store = session.store
        pending = session.pending_state()
    elif isinstance(session_or_saved_state, dict):
        pending = session_or_saved_state
    else:
        if session_or_saved_state is not None:
            store = session_or_saved_state
        if store is None or not hasattr(store, "load_pending_oauth"):
            raise RuntimeError("No pending ChatGPT sign-in to complete. Start the sign-in again.")
        pending = store.load_pending_oauth()
        if not pending:
            raise RuntimeError(
                "No pending ChatGPT sign-in to complete (it may have expired). Start the sign-in again."
            )

    code_verifier = pending.get("code_verifier")
    expected_state = pending.get("state")
    redirect_uri = pending.get("redirect_uri")
    if not (code_verifier and expected_state and redirect_uri):
        raise RuntimeError("The saved ChatGPT sign-in is incomplete. Start the sign-in again.")

    use_loopback = session is not None and redirect_url_or_code is None
    if use_loopback:
        code, returned_state, error = session.auth_code, session.returned_state, session.error
    elif redirect_url_or_code is None:
        raise RuntimeError("Paste the redirect URL (or the code) to finish the ChatGPT sign-in.")
    else:
        code, returned_state, error = _parse_oauth_redirect(redirect_url_or_code)

    if error:
        raise RuntimeError(f"OAuth error: {error}")
    if not code:
        if use_loopback:
            raise RuntimeError("OAuth login timed out – no callback received.")
        raise RuntimeError("OAuth redirect did not contain an authorization code.")
    if (use_loopback or returned_state is not None) and returned_state != expected_state:
        raise RuntimeError("OAuth state mismatch – possible CSRF attack.")

    # Exchange code for tokens
    print("🔑 Exchanging authorization code for tokens…")
    tokens = exchange_code_for_tokens(code, code_verifier, redirect_uri)
    print("✅ ChatGPT OAuth authentication successful!")

    # Extract and log account info (non-sensitive)
    id_token = tokens.get("id_token", "")
    if id_token:
        info = extract_account_info(id_token)
        plan = info.get("plan_type", "unknown")
        email = info.get("email", "")
        if email:
            print(f"   Account: {email} (plan: {plan})")

    if store is not None:
        store.save_tokens(tokens)
        if hasattr(store, "clear_pending_oauth"):
            store.clear_pending_oauth()
    if session is not None and not use_loopback:
        session.close()
    return tokens


def run_oauth_flow(timeout: int = 300) -> Dict:
    """Run the full OAuth PKCE login flow.

    1. Opens a browser for the user to authenticate with ChatGPT.
    2. Captures the callback on a local server.
    3. Exchanges the code for tokens.

    Returns the token dict (access_token, refresh_token, expires_at, …).
    Raises RuntimeError on failure or timeout.

    Parameters
    ----------
    timeout : int
        Maximum seconds to wait for the user to complete the browser login.

    Composes :func:`begin_oauth` (IPv4 listener only, nothing persisted) and
    :func:`complete_from_redirect` (tokens are returned, not saved).
    """
    session = begin_oauth(timeout=timeout, persist=False, dual_stack=False, auto_close=False)
    auth_url = session.auth_url

    try:
        print(f"🔐 Opening browser for ChatGPT login…")
        print(f"   If the browser doesn't open, visit:\n   {auth_url}")
        webbrowser.open(auth_url)

        # Serve until callback is received or timeout
        session.wait(timeout)
    finally:
        # Cleanup
        session.close()

    return complete_from_redirect(session)


# ===========================================================================
# Token store (persistent, thread-safe)
# ===========================================================================

class AuthGPTTokenStore:
    """Thread-safe token store backed by a JSON file."""

    def __init__(self, token_file: Optional[str] = None, account_id: int = 0):
        self._token_file = (
            token_file
            or os.environ.get("AUTHGPT_TOKEN_FILE")
            or _DEFAULT_TOKEN_FILE
        )
        self._account_id = account_id
        self._lock = threading.RLock()
        self._tokens: Optional[Dict] = None
        self._on_change_callbacks: List = []  # called after save/clear
        # Eagerly load cached tokens from disk (if any)
        self._load_from_disk()

    def on_token_change(self, callback):
        """Register *callback* to be called (no args) after tokens change."""
        self._on_change_callbacks.append(callback)

    def _fire_change_callbacks(self):
        for cb in self._on_change_callbacks:
            try:
                cb()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _ensure_dir(self):
        d = os.path.dirname(self._token_file)
        if d:
            os.makedirs(d, exist_ok=True)

    def _load_from_disk(self):
        """Load tokens from the encrypted file into memory.

        If decryption fails (corrupt file, wrong user, etc.), the file is
        removed so the next login produces a fresh, correctly-encrypted file.
        """
        try:
            if os.path.isfile(self._token_file):
                try:
                    from token_encryption import load_encrypted_tokens
                    self._tokens = load_encrypted_tokens(self._token_file)
                except ImportError:
                    # token_encryption module not available — read plain JSON
                    with open(self._token_file, "r", encoding="utf-8") as f:
                        self._tokens = json.load(f)
                        logger.warning("🔓 AuthGPT: loaded UNENCRYPTED credentials; encryption module is unavailable")
                except Exception as dec_exc:
                    # Decryption failed — file is corrupt or from a different
                    # user/machine.  Delete it so re-login creates a fresh one.
                    logger.warning("❌ AuthGPT token decryption failed (%s) — removing corrupt file", type(dec_exc).__name__)
                    try:
                        os.remove(self._token_file)
                    except OSError:
                        pass
                    self._tokens = None
                    return
                logger.debug("AuthGPT tokens loaded from %s", self._token_file)
        except Exception as exc:
            logger.warning("Failed to load authgpt tokens: %s", exc)
            self._tokens = None

    def save_tokens(self, tokens: Dict):
        """Encrypt and save tokens to disk, and cache in memory.

        If encryption fails for any reason, falls back to plain JSON so
        the tokens are not lost and the app continues working.
        """
        with self._lock:
            self._tokens = tokens
            try:
                self._ensure_dir()
                saved = False
                try:
                    from token_encryption import save_encrypted_tokens
                    save_encrypted_tokens(tokens, self._token_file)
                    saved = True
                except ImportError:
                    logger.warning("⚠️ AuthGPT: encryption module unavailable; falling back to UNENCRYPTED JSON storage")
                except Exception as enc_exc:
                    logger.warning("⚠️ AuthGPT token encryption failed (%s) — saving as plain JSON", type(enc_exc).__name__)
                if not saved:
                    # Fallback: plain JSON (still better than losing tokens)
                    with open(self._token_file, "w", encoding="utf-8") as f:
                        json.dump(tokens, f, indent=2)
                    logger.warning("🔓 AuthGPT: credentials saved WITHOUT ENCRYPTION (plain JSON fallback)")
                logger.debug("AuthGPT tokens saved to %s", self._token_file)
            except Exception as exc:
                logger.warning("Failed to save authgpt tokens: %s", exc)
        self._fire_change_callbacks()

    def load_tokens(self) -> Optional[Dict]:
        """Return cached tokens (loading from disk if needed)."""
        with self._lock:
            if self._tokens is None:
                self._load_from_disk()
            return self._tokens

    def clear_tokens(self):
        """Delete stored tokens (logout)."""
        with self._lock:
            self._tokens = None
            try:
                if os.path.isfile(self._token_file):
                    os.remove(self._token_file)
                    logger.info("AuthGPT tokens removed")
            except Exception as exc:
                logger.warning("Failed to remove token file: %s", exc)
        self._fire_change_callbacks()

    # ------------------------------------------------------------------
    # Pending sign-in (begin_oauth -> complete_from_redirect paste fallback)
    # ------------------------------------------------------------------

    @property
    def pending_oauth_file(self) -> str:
        """Encrypted PKCE verifier/state of an unfinished sign-in, next to the token file."""
        return os.path.splitext(self._token_file)[0] + ".oauth_pending"

    def save_pending_oauth(self, pending: Dict) -> bool:
        """Save a pending sign-in encrypted like the tokens (never as plain text)."""
        path = self.pending_oauth_file
        with self._lock:
            try:
                from token_encryption import encrypt_tokens
                data = encrypt_tokens(dict(pending))
                self._ensure_dir()
                with open(path, "wb") as f:
                    f.write(data)
                if os.name != "nt":
                    try:
                        os.chmod(path, 0o600)
                    except OSError:
                        pass
                return True
            except Exception as exc:
                logger.warning("⚠️ AuthGPT: could not save the pending sign-in (%s)", type(exc).__name__)
                return False

    def load_pending_oauth(self) -> Optional[Dict]:
        """Return the saved pending sign-in, or None when missing, unreadable or expired."""
        path = self.pending_oauth_file
        with self._lock:
            if not os.path.isfile(path):
                return None
            try:
                with open(path, "rb") as f:
                    data = f.read()
                from token_encryption import decrypt_tokens
                pending = decrypt_tokens(data)
            except Exception as exc:
                logger.warning("⚠️ AuthGPT: discarding unreadable pending sign-in (%s)", type(exc).__name__)
                self.clear_pending_oauth()
                return None
            created_at = pending.get("created_at") if isinstance(pending, dict) else None
            try:
                expired = time.time() - float(created_at) > PENDING_OAUTH_MAX_AGE_SECONDS
            except (TypeError, ValueError):
                expired = True
            if expired:
                self.clear_pending_oauth()
                return None
            return pending

    def clear_pending_oauth(self) -> None:
        """Forget a pending sign-in."""
        path = self.pending_oauth_file
        with self._lock:
            try:
                if os.path.isfile(path):
                    os.remove(path)
            except OSError as exc:
                logger.warning("Failed to remove pending sign-in file: %s", exc)

    # ------------------------------------------------------------------
    # Token access
    # ------------------------------------------------------------------

    def _is_token_expired(self, tokens: Dict) -> bool:
        """Check if the access token has expired or is about to."""
        expires_at = tokens.get("expires_at", 0)
        return time.time() >= (expires_at - TOKEN_REFRESH_MARGIN_SECONDS)

    def _try_refresh(self, tokens: Dict) -> Optional[Dict]:
        """Attempt to refresh the access token using the stored refresh token."""
        rt = tokens.get("refresh_token")
        if not rt:
            return None
        try:
            new_tokens = refresh_access_token(rt)
            # Preserve fields from old tokens that aren't returned by refresh
            merged = {**tokens, **new_tokens}
            self.save_tokens(merged)
            logger.info("AuthGPT access token refreshed successfully")
            return merged
        except Exception as exc:
            logger.warning("AuthGPT token refresh failed: %s", exc)
            return None

    def get_valid_access_token(self, auto_login: bool = True) -> str:
        """Return a valid access token, refreshing or re-authenticating as needed.

        Parameters
        ----------
        auto_login : bool
            If True and no valid token can be obtained, launch the browser
            OAuth flow interactively.

        Returns
        -------
        str
            A valid Bearer access token.

        Raises
        ------
        RuntimeError
            If a valid token cannot be obtained.
        """
        with self._lock:
            tokens = self.load_tokens()

            # Happy path – have a valid token
            if tokens and tokens.get("access_token") and not self._is_token_expired(tokens):
                return tokens["access_token"]

            # Try refresh
            if tokens and tokens.get("refresh_token"):
                refreshed = self._try_refresh(tokens)
                if refreshed and refreshed.get("access_token"):
                    return refreshed["access_token"]

            # No usable tokens – need interactive login
            if not auto_login:
                raise RuntimeError(
                    "AuthGPT: No valid tokens and auto_login is disabled. "
                    "Run the OAuth login flow first."
                )

            # Detect headless environments (HF Spaces, Docker, etc.) where browser login is impossible
            is_headless = (
                os.environ.get("SPACE_ID") is not None
                or os.environ.get("HF_SPACES") == "true"
                or os.environ.get("DOCKER_CONTAINER") == "true"
                or os.environ.get("KUBERNETES_SERVICE_HOST") is not None
            )
            if is_headless:
                # Check for manually-provided tokens via environment variables
                env_access = os.environ.get("AUTHGPT_ACCESS_TOKEN", "").strip()
                env_refresh = os.environ.get("AUTHGPT_REFRESH_TOKEN", "").strip()
                if env_access:
                    # User provided an access token directly — save and use it
                    manual_tokens = {
                        "access_token": env_access,
                        "expires_at": time.time() + 3600,  # assume 1h validity
                    }
                    if env_refresh:
                        manual_tokens["refresh_token"] = env_refresh
                    self.save_tokens(manual_tokens)
                    logger.info("AuthGPT: Using access token from AUTHGPT_ACCESS_TOKEN env var")
                    return env_access
                if env_refresh:
                    # Try refreshing with the provided refresh token
                    try:
                        refreshed = refresh_access_token(env_refresh)
                        self.save_tokens(refreshed)
                        logger.info("AuthGPT: Obtained access token via AUTHGPT_REFRESH_TOKEN env var")
                        return refreshed["access_token"]
                    except Exception as ref_exc:
                        raise RuntimeError(
                            f"AuthGPT: AUTHGPT_REFRESH_TOKEN was set but refresh failed: {ref_exc}\n"
                            "The refresh token may be expired. Please obtain a new one."
                        )
                raise RuntimeError(
                    "AuthGPT: Browser-based OAuth login is not available in headless environments "
                    "(e.g. Hugging Face Spaces, Docker containers).\n"
                    "To use AuthGPT models, set one of these as environment secrets:\n"
                    "  • AUTHGPT_ACCESS_TOKEN — a valid ChatGPT OAuth access token\n"
                    "  • AUTHGPT_REFRESH_TOKEN — a ChatGPT OAuth refresh token (will auto-refresh)\n"
                    "You can obtain these by running the OAuth flow locally first, then copying\n"
                    "the tokens from ~/.glossarion/authgpt_tokens.json"
                )

            print("🔄 AuthGPT: No valid token found – starting browser login…")
            new_tokens = run_oauth_flow()
            self.save_tokens(new_tokens)
            return new_tokens["access_token"]

    def recover_from_unauthorized(
        self,
        rejected_access_token: Optional[str] = None,
        auto_login: bool = True,
    ) -> str:
        """Replace a server-rejected token through the interactive login flow.

        A backend 401 is authoritative even when the cached ``expires_at`` is
        still in the future.  If another request thread already replaced the
        rejected token, reuse that replacement instead of opening a second
        browser window.
        """
        with self._lock:
            tokens = self.load_tokens() or {}
            current_access_token = tokens.get("access_token")
            if (
                rejected_access_token
                and current_access_token
                and current_access_token != rejected_access_token
                and not self._is_token_expired(tokens)
            ):
                logger.info(
                    "AuthGPT: Reusing access token replaced by another request thread"
                )
                return current_access_token

            logger.warning(
                "AuthGPT: Backend rejected the cached access token; forcing re-login"
            )
            self.clear_tokens()
            if not auto_login:
                raise RuntimeError(
                    "AuthGPT: Access token was rejected and interactive login is disabled."
                )
            print("🔐 AuthGPT: Session expired – opening browser login…")
            return self.get_valid_access_token(auto_login=True)

    @property
    def has_tokens(self) -> bool:
        """Return True if any tokens are cached (may be expired)."""
        tokens = self.load_tokens()
        return bool(tokens and tokens.get("access_token"))

    @property
    def account_info(self) -> Dict:
        """Return account info extracted from cached id_token, or empty dict."""
        tokens = self.load_tokens()
        if not tokens or not tokens.get("id_token"):
            return {}
        return extract_account_info(tokens["id_token"])


# Module-level singleton for convenience (lazy-initialized)
_default_store: Optional[AuthGPTTokenStore] = None
_default_store_lock = threading.Lock()


def get_default_store() -> AuthGPTTokenStore:
    """Return the module-level default token store (singleton)."""
    global _default_store
    if _default_store is None:
        with _default_store_lock:
            if _default_store is None:
                _default_store = AuthGPTTokenStore()
    return _default_store


# Per-account store registry (account_id → AuthGPTTokenStore)
_account_stores: Dict[int, AuthGPTTokenStore] = {}
_account_stores_lock = threading.Lock()


def get_store(account_id: Optional[int] = None) -> AuthGPTTokenStore:
    """Return the token store for a specific account slot.

    Parameters
    ----------
    account_id : int or None
        ``None`` or ``0`` returns the default store (``authgpt_tokens.json``).
        Any positive integer *N* returns a dedicated store backed by
        ``authgpt_tokens_N.json`` in the same directory, enabling
        multi-account usage via ``authgptN/`` model prefixes.

    Each numbered account triggers its own independent OAuth browser login
    the first time it is used, so users can authenticate with different
    ChatGPT accounts for each slot.
    """
    if account_id is None or account_id == 0:
        return get_default_store()

    with _account_stores_lock:
        if account_id in _account_stores:
            return _account_stores[account_id]

        # Build a token file path like  ~/.glossarion/authgpt_tokens_2.json
        token_file = os.path.join(_DEFAULT_TOKEN_DIR, f"authgpt_tokens_{account_id}.json")
        store = AuthGPTTokenStore(token_file=token_file, account_id=account_id)
        _account_stores[account_id] = store
        return store


def fetch_available_models(
    access_token: str,
    timeout: int = 10,
    base_url: Optional[str] = None,
) -> List[str]:
    """Return displayable models for the signed-in ChatGPT account.

    The account catalog is the primary source. The older Codex backend
    manifest remains a fallback for tokens that cannot access that catalog.
    Polling is read-only and never opens an interactive login window.
    """
    effective_base = str(
        base_url or os.getenv("AUTHGPT_BASE_URL", CHATGPT_BASE_URL) or ""
    ).rstrip("/")
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/json",
    }

    def parse_models(payload: object, *, require_list_visibility: bool) -> List[str]:
        if not isinstance(payload, dict):
            return []
        entries = payload.get("models")
        if not require_list_visibility and not isinstance(entries, list):
            entries = payload.get("data")
        if not isinstance(entries, list):
            return []
        models: List[str] = []
        seen = set()
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            visibility = str(entry.get("visibility", "") or "").casefold()
            if (require_list_visibility and visibility != "list") or visibility == "hide":
                continue
            model = str(entry.get("slug") or entry.get("id") or "").strip()
            key = model.casefold()
            if model and key not in seen:
                seen.add(key)
                models.append(model)
        return models

    if effective_base == CHATGPT_BASE_URL:
        try:
            response = requests.get(
                ACCOUNT_MODELS_URL,
                headers=headers,
                timeout=max(1, int(round(timeout))),
            )
            response.raise_for_status()
            models = parse_models(response.json(), require_list_visibility=True)
            if models:
                return models
        except (requests.RequestException, ValueError):
            logger.debug("AuthGPT: Account model catalog unavailable; trying Codex manifest")

    codex_base = (
        effective_base
        if effective_base.casefold().endswith("/codex")
        else f"{effective_base}/codex"
    )
    response = requests.get(
        f"{codex_base}/models",
        # The backend uses this semantic version for compatibility filtering.
        # Sending 0.0.0 hides models that require a newer client.
        params={"client_version": APP_VERSION},
        headers=headers,
        timeout=max(1, int(round(timeout))),
    )
    response.raise_for_status()
    return parse_models(response.json(), require_list_visibility=False)


# ===========================================================================
# ChatGPT backend API adapter (Codex Responses API)
# ===========================================================================
# The Codex CLI uses /backend-api/codex/responses with the standard OpenAI
# Responses API format.  The /conversation endpoint is reserved for the
# ChatGPT web UI and rejects third-party OAuth tokens with 403.
# ===========================================================================


def _build_responses_body(
    messages: List[Dict],
    model: str,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    reasoning: Optional[Dict[str, Any]] = None,
) -> Dict:
    """Build a Codex Responses API request body from standard OpenAI messages.

    The /backend-api/codex/responses endpoint expects:
      - "instructions": system-level instructions (required)
      - "input": list of {type, role, content:[{type, text}]} message objects
      - "model", "store", "stream", and optional "reasoning"

    Temperature is accepted for caller compatibility and omitted. A supplied
    max_tokens is sent as max_output_tokens except for gpt-6-astra, which
    rejects it. For other models, the send layer retries without it, with
    an explicit log message, only if the backend rejects the field.
    """
    instructions = ""
    input_items: List[Dict[str, Any]] = []

    def _convert_content_parts(content, role="user") -> List[Dict[str, Any]]:
        """Convert Chat Completions content to Responses API content parts.

        Handles both plain strings and multi-modal lists (text + image_url).
        The Codex Responses API requires 'output_text' for assistant messages
        and 'input_text' for user/developer messages.
        """
        # Assistant role must use output_text (Responses API schema requirement)
        text_type = "output_text" if role == "assistant" else "input_text"

        if isinstance(content, str):
            return [{"type": text_type, "text": content}]

        if not isinstance(content, list):
            return [{"type": text_type, "text": str(content)}]

        parts: List[Dict[str, Any]] = []
        for part in content:
            if isinstance(part, str):
                parts.append({"type": text_type, "text": part})
                continue
            if not isinstance(part, dict):
                continue
            ptype = part.get("type", "")
            if ptype == "text":
                parts.append({"type": text_type, "text": part.get("text", "")})
            elif ptype == "image_url":
                # Chat Completions: {"type": "image_url", "image_url": {"url": "..."}}
                # Responses API:    {"type": "input_image", "image_url": "..."}
                img = part.get("image_url", {})
                url = img.get("url", "") if isinstance(img, dict) else str(img)
                if url:
                    parts.append({"type": "input_image", "image_url": url})
            elif ptype == "input_text":
                # Already in Responses format — but fix the type if this is an assistant msg
                if role == "assistant":
                    parts.append({"type": "output_text", "text": part.get("text", "")})
                else:
                    parts.append(part)
            elif ptype == "output_text":
                parts.append(part)
            elif ptype == "input_image":
                parts.append(part)
            else:
                # Unknown part type — pass text if present
                text = part.get("text")
                if text:
                    parts.append({"type": text_type, "text": str(text)})
        return parts or [{"type": text_type, "text": ""}]

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        # System messages become the instructions blob
        if role == "system":
            # instructions must be a string; extract text from multi-part content
            if isinstance(content, str):
                instructions = content
            elif isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, str):
                        text_parts.append(part)
                    elif isinstance(part, dict):
                        t = part.get("text", "")
                        if t:
                            text_parts.append(str(t))
                instructions = "\n".join(text_parts)
            else:
                instructions = str(content)
            continue

        # Map to Responses API structured message format
        input_items.append({
            "type": "message",
            "role": "developer" if role == "system" else role,
            "content": _convert_content_parts(content, role),
        })

    body: Dict[str, Any] = {
        "model": model,
        "instructions": instructions,
        "input": input_items,
        "store": False,
        "stream": True,
    }

    if max_tokens is not None and model.strip().lower() != "gpt-6-astra":
        body["max_output_tokens"] = max_tokens

    if reasoning:
        body["reasoning"] = dict(reasoning)

    return body


def _parse_responses_result(data: Dict) -> Dict:
    """Extract content from a Responses API result."""
    content = ""
    finish_reason = "stop"
    usage = None
    response_id = data.get("id")

    # The output field contains a list of items; find the message
    for item in data.get("output", []):
        if item.get("type") == "message":
            for part in item.get("content", []):
                if part.get("type") == "output_text":
                    content += part.get("text", "")

    # Check status
    status = data.get("status", "")
    if status == "completed":
        finish_reason = "stop"
    elif status == "incomplete":
        incomplete = data.get("incomplete_details", {}) or {}
        reason = incomplete.get("reason", "")
        finish_reason = "length" if "tokens" in reason else reason or "incomplete"
    elif status == "failed":
        finish_reason = "error"

    # Usage
    raw_usage = data.get("usage")
    if raw_usage:
        usage = {
            "prompt_tokens": raw_usage.get("input_tokens", 0),
            "completion_tokens": raw_usage.get("output_tokens", 0),
            "total_tokens": raw_usage.get("total_tokens", 0),
        }

    result = {
        "content": content,
        "finish_reason": finish_reason,
        "conversation_id": response_id,
        "message_id": response_id,
        "usage": usage,
    }
    if data.get("error"):
        result["error_details"] = data.get("error")
    return result


def _parse_sse_responses(raw_text: str) -> Dict:
    """Parse SSE stream from the Codex Responses endpoint.

    The endpoint may return a single JSON object or an SSE stream.
    """
    # Try direct JSON first (non-streaming response)
    stripped = raw_text.strip()
    if stripped.startswith("{"):
        try:
            return _parse_responses_result(json.loads(stripped))
        except json.JSONDecodeError:
            pass

    # SSE stream – look for response.completed event
    last_data = None
    content_parts: List[str] = []
    for line in raw_text.splitlines():
        line = line.strip()
        if not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload == "[DONE]":
            break
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            continue

        event_type = data.get("type", "")

        # Accumulate text deltas
        if event_type == "response.output_text.delta":
            content_parts.append(data.get("delta", ""))
        # Final completed event has the full response
        elif event_type in ("response.completed", "response.incomplete"):
            resp_obj = data.get("response", data)
            result = _parse_responses_result(resp_obj)
            # The Codex API's completed event often omits the full text
            # in its output items — only metadata (usage, status) is present.
            # Fall back to accumulated deltas when content is empty.
            if not result.get("content") and content_parts:
                result["content"] = "".join(content_parts)
            if event_type == "response.incomplete" and result.get("finish_reason") == "stop":
                result["finish_reason"] = "length"
            return result
        elif event_type == "response.failed":
            resp_obj = data.get("response", data)
            result = _parse_responses_result(resp_obj)
            result["content"] = ""
            result["finish_reason"] = "error"
            result["error_details"] = (
                resp_obj.get("error")
                or data.get("error")
                or {"type": "response.failed"}
            )
            if content_parts:
                result["partial_content"] = "".join(content_parts)
            return result

        last_data = data

    # Fallback: if we accumulated deltas but no completed event
    if content_parts:
        return {
            "content": "".join(content_parts),
            "finish_reason": "stop",
            "conversation_id": None,
            "message_id": None,
            "usage": None,
        }

    # Last resort: try parsing the last data line
    if last_data:
        return _parse_responses_result(last_data)

    return {
        "content": "",
        "finish_reason": "error",
        "conversation_id": None,
        "message_id": None,
        "usage": None,
    }


# ---------------------------------------------------------------------------
# SSE stream processing helpers
# ---------------------------------------------------------------------------

def _collect_reasoning_text(value: Any, force_text: bool = False) -> str:
    """Extract reasoning summary text from Responses stream payload fragments."""
    text_keys = {"delta", "text", "summary_text", "reasoning_text"}
    container_keys = {
        "summary",
        "summaries",
        "content",
        "reasoning",
        "reasoning_summary",
        "reasoning_content",
        "item",
        "output_item",
        "part",
    }

    def walk(node: Any, allow_text: bool = False) -> List[str]:
        if node is None:
            return []
        if isinstance(node, str):
            return [node] if allow_text else []
        if isinstance(node, list):
            parts: List[str] = []
            for child in node:
                parts.extend(walk(child, allow_text))
            return parts
        if not isinstance(node, dict):
            return []

        type_hint = str(node.get("type", "") or "").lower()
        in_reasoning = allow_text or "reasoning" in type_hint or "summary" in type_hint
        parts: List[str] = []
        for key, child in node.items():
            key_l = str(key).lower()
            if key_l in text_keys:
                parts.extend(walk(child, in_reasoning or force_text))
            elif (
                key_l in container_keys
                or "reasoning" in key_l
                or "summary" in key_l
            ):
                parts.extend(walk(child, True))
        return parts

    return "".join(walk(value, force_text))


def _extract_reasoning_item(value: Any) -> Optional[Any]:
    """Return an explicit reasoning item from a stream payload, if present."""
    if not isinstance(value, dict):
        return None
    candidates = [
        value.get("item"),
        value.get("output_item"),
        value.get("part"),
    ]
    for candidate in candidates:
        if isinstance(candidate, dict) and "reasoning" in str(candidate.get("type", "") or "").lower():
            return candidate
    return None


def _flush_stream_log_buf(state: Dict, _log) -> None:
    if state.get("log_buf"):
        remainder = "".join(state["log_buf"]).strip()
        if remainder:
            _log(remainder.replace('\x1f', '\\x1F'))
        state["log_buf"] = []


def _authgpt_stream_thinking_text_enabled() -> bool:
    raw = os.getenv("STREAM_AUTHGPT_THINKING_TEXT")
    if raw is None:
        raw = os.getenv("STREAM_THINKING_LOGS", "0")
    return str(raw).lower() not in ("0", "false")


def _flush_thinking_display_buf(state: Dict, _log, force: bool = False) -> None:
    buf = state.get("thinking_display_buf", "")
    if not buf:
        return

    def emit(text: str) -> None:
        for line in text.splitlines():
            if line.strip():
                safe_line = line.replace('\x1f', '\\x1F')
                if safe_line.strip().startswith("**"):
                    _log("")
                _log(f"    {safe_line}")

    # AuthGPT often sends markdown headings with no preceding newline after
    # a sentence delta. Split those so sections do not get stitched together.
    buf = re.sub(r"(?<!^)(?<!\n)(\*\*[^*\n]{3,120}\*\*)", r"\n\1", buf)

    while "\n" in buf:
        line, buf = buf.split("\n", 1)
        emit(line)

    if force:
        if buf.strip():
            emit(buf)
        state["thinking_display_buf"] = ""
        return

    # Flush readable chunks, not token-sized fragments. Prefer sentence-ish
    # boundaries, then spaces, and only hard-cut as a last resort.
    max_buf = 220
    while len(buf) >= max_buf:
        boundary = max(buf.rfind(mark, 0, max_buf) for mark in (". ", "? ", "! ", "; ", ": "))
        if boundary >= 80:
            cut = boundary + 1
        else:
            cut = buf.rfind(" ", 0, max_buf)
            if cut < 120:
                cut = max_buf
        emit(buf[:cut].strip())
        buf = buf[cut:].lstrip()
    state["thinking_display_buf"] = buf


def _append_thinking_display_text(state: Dict, text: str, _log) -> None:
    if state.get("text_stream_started") and state.get("display_phase") != "thinking":
        _flush_stream_log_buf(state, _log)
        _log("─" * 50)
        _log("🧠 [authgpt] Thinking...")
        state["display_phase"] = "thinking"
    _flush_stream_log_buf(state, _log)
    state["thinking_display_buf"] = state.get("thinking_display_buf", "") + text
    _flush_thinking_display_buf(state, _log, force=bool(state.get("text_stream_started")))


def _start_text_stream(state: Dict, _log) -> None:
    if state.get("text_stream_started"):
        if state.get("display_phase") != "text":
            if _authgpt_stream_thinking_text_enabled():
                _flush_thinking_display_buf(state, _log, force=True)
            _log("─" * 50)
            _log("📡 AuthGPT: Text streaming...")
            state["display_phase"] = "text"
        return
    if _authgpt_stream_thinking_text_enabled():
        _flush_thinking_display_buf(state, _log, force=True)
    state["text_stream_started"] = True
    if state.get("thinking_started"):
        _log("─" * 50)
    _log("📡 AuthGPT: Text streaming...")
    state["display_phase"] = "text"


def _append_reasoning_delta(state: Dict, text: str) -> str:
    text = str(text or "").replace("\\n", "\n")
    if not text:
        return ""
    return text


def _mark_reasoning_event(state: Dict, _log) -> None:
    now = time.time()
    if state["thinking_start_ts"] is None:
        state["thinking_start_ts"] = now

    if not state["thinking_started"]:
        state["thinking_started"] = True
        if not state.get("text_stream_started"):
            _flush_stream_log_buf(state, _log)
            _log("🧠 [authgpt] Thinking...")
            state["display_phase"] = "thinking"
        state["last_thinking_progress_ts"] = now
    elif not state.get("text_stream_started"):
        _flush_thinking_display_buf(state, _log)


def _append_reasoning_text(state: Dict, text: str, _log, is_delta: bool = False) -> None:
    _mark_reasoning_event(state, _log)
    delta = _append_reasoning_delta(state, text)
    if not delta:
        return
    state["thinking_chunks"] += 1
    state["thinking_text_parts"].append(delta)

    # AuthGPT can emit reasoning-summary chunks interleaved with output-text
    # chunks. Keep capturing them, but do not print the text into the live
    # translation stream unless explicitly requested.
    stream_thinking_text = _authgpt_stream_thinking_text_enabled()
    if not stream_thinking_text:
        return

    _append_thinking_display_text(state, delta, _log)


def _count_thinking_tokens(text: str, model: Optional[str] = None) -> Optional[int]:
    if not text:
        return 0
    try:
        import tiktoken
        try:
            enc = tiktoken.encoding_for_model(model or "")
        except Exception:
            enc = tiktoken.get_encoding("o200k_base")
        return len(enc.encode(text))
    except Exception:
        return None


def _process_sse_line(
    line: str,
    state: Dict,
    _log,
    log_stream: bool,
    t_start: float,
) -> bool:
    """Process a single SSE line, updating *state* in-place.

    Returns True when the stream should stop (saw [DONE] or response.completed).
    """
    state["raw_lines"].append(line)

    if line.startswith("event: "):
        state["pending_event_type"] = line[7:].strip()
        return False

    if not state["got_first_data"] and line.startswith("data: "):
        state["got_first_data"] = True
        ttft = time.time() - t_start
        _log(f"📡 AuthGPT: First stream event in {ttft:.1f}s, streaming…")

    data = None
    event_type = ""
    if line.startswith("data: ") and line[6:] != "[DONE]":
        try:
            data = json.loads(line[6:])
            event_type = str(data.get("type", "") or state.get("pending_event_type", "") or "")
        except json.JSONDecodeError:
            data = None

    if data is not None and event_type:
        state["event_types_seen"].add(event_type)
        if os.getenv("AUTHGPT_DEBUG_STREAM_EVENTS", "0").lower() not in ("0", "false"):
            if event_type not in state["event_types_logged"]:
                state["event_types_logged"].add(event_type)
                _log(f"🔎 AuthGPT stream event: {event_type}")

    # Extract text deltas and display in real-time
    if data is not None and event_type == "response.output_text.delta":
        try:
            delta_text = data.get("delta", "")
            state["streamed_chars"] += len(delta_text)
            if log_stream and delta_text:
                _start_text_stream(state, _log)
                log_buf = state["log_buf"]
                combined = "".join(log_buf) + delta_text
                for tag in ('</h1>', '</h2>', '</h3>', '</h4>', '</h5>', '</h6>', '</p>'):
                    combined = combined.replace(tag, tag + '\n')
                if "\n" in combined:
                    parts = combined.split("\n")
                    for part in parts[:-1]:
                        _log(part.replace('\x1f', '\\x1F'))
                    state["log_buf"] = [parts[-1]]
                else:
                    log_buf.append(delta_text)
                    if len("".join(log_buf)) > 150:
                        _log("".join(log_buf).replace('\x1f', '\\x1F'))
                        state["log_buf"] = []
        except (json.JSONDecodeError, KeyError):
            pass

    # Responses streams may emit reasoning summary/text events before output text.
    # Capture them separately so thinking does not get mixed into the final answer.
    if data is not None and "reasoning" in event_type:
        try:
            _mark_reasoning_event(state, _log)
            force_text = event_type.endswith(".delta")
            reasoning_text = _collect_reasoning_text(data, force_text=force_text)
            if reasoning_text and (
                force_text
                or state.get("thinking_chunks", 0) == 0
                or _authgpt_stream_thinking_text_enabled()
            ):
                _append_reasoning_text(state, reasoning_text, _log, is_delta=force_text)
        except Exception:
            pass
    elif data is not None:
        try:
            # Some Responses streams carry completed reasoning items under
            # response.output_item.done instead of a reasoning-specific event.
            reasoning_item = _extract_reasoning_item(data)
            reasoning_text = _collect_reasoning_text(reasoning_item, force_text=False) if reasoning_item else ""
            if reasoning_text and (
                state.get("thinking_chunks", 0) == 0
                or _authgpt_stream_thinking_text_enabled()
            ):
                _append_reasoning_text(state, reasoning_text, _log, is_delta=False)
        except Exception:
            pass

    # Stop signals
    if line.strip() == "data: [DONE]":
        return True
    if (
        '"type":"response.completed"' in line
        or '"type": "response.completed"' in line
        or '"type":"response.incomplete"' in line
        or '"type": "response.incomplete"' in line
    ):
        return True
    return False


def _finalize_stream(state: Dict, _log, log_stream: bool, t_start: float) -> Dict:
    """Flush log buffer, parse collected SSE lines, return result dict."""
    if log_stream and state["log_buf"]:
        _flush_stream_log_buf(state, _log)
    if state.get("text_stream_started") and not state.get("text_stream_complete_logged"):
        _log("📡 AuthGPT: Text streaming complete")
        state["text_stream_complete_logged"] = True

    stream_thinking_text = _authgpt_stream_thinking_text_enabled()
    if stream_thinking_text:
        _flush_thinking_display_buf(state, _log, force=True)
    if state.get("thinking_started"):
        thinking_dur = time.time() - (state.get("thinking_start_ts") or time.time())
        thinking_text = "".join(state.get("thinking_text_parts", []))
        thinking_tokens = _count_thinking_tokens(thinking_text, state.get("model"))
        if thinking_tokens is None:
            _log(f"🧠 [authgpt] Thinking tokens used: unavailable ({thinking_dur:.1f}s)")
        else:
            _log(f"🧠 [authgpt] Thinking tokens used: {thinking_tokens:,} ({thinking_dur:.1f}s)")

    raw_text = "\n".join(state["raw_lines"])
    t_total = time.time() - t_start
    _log(f"📡 AuthGPT: Stream finished in {t_total:.1f}s")
    result = _parse_sse_responses(raw_text)
    if state.get("thinking_chunks", 0) > 0:
        result["thinking_chunks"] = state.get("thinking_chunks", 0)
        result["thinking_text"] = "".join(state.get("thinking_text_parts", []))

    content = result.get("content", "")
    if not content:
        event_types = sorted(state.get("event_types_seen", []))
        _log(f"⚠️ AuthGPT: Empty content after parsing. Event types seen: {event_types}")
    return result


def _new_stream_state() -> Dict:
    return {
        "raw_lines": [],
        "got_first_data": False,
        "streamed_chars": 0,
        "log_buf": [],
        "thinking_started": False,
        "thinking_start_ts": None,
        "last_thinking_progress_ts": 0.0,
        "thinking_chunks": 0,
        "thinking_log_buf": [],
        "thinking_text_parts": [],
        "thinking_display_buf": "",
        "text_stream_started": False,
        "text_stream_complete_logged": False,
        "display_phase": None,
        "pending_event_type": "",
        "event_types_seen": set(),
        "event_types_logged": set(),
        "model": None,
    }


# ---------------------------------------------------------------------------
# httpx-based SSE reader (preferred — real-time, no buffering)
# ---------------------------------------------------------------------------

class _AuthGPTHTTPError(RuntimeError):
    def __init__(self, message, status_code, body):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class _UnsupportedOutputLimitError(RuntimeError):
    """The backend rejected max_output_tokens before generation began."""


def _is_unsupported_output_limit(status_code: int, error_body: str) -> bool:
    if status_code != 400:
        return False
    try:
        data = json.loads(error_body)
        message = data.get("detail") or data.get("error") or data
        if isinstance(message, dict):
            message = message.get("message", "")
    except (ValueError, AttributeError):
        message = error_body
    return bool(re.fullmatch(
        r"\s*Unsupported parameter:\s*['\"]?max_output_tokens['\"]?\.?\s*",
        str(message), re.IGNORECASE,
    ))


def _stream_with_httpx(
    _httpx,
    url: str,
    body: Dict,
    headers: Dict,
    timeout: int,
    t_start: float,
    _log,
    log_stream: bool,
    connect_timeout: Optional[float] = None,
) -> Dict:
    """Stream SSE using httpx (same stack as the openai Python SDK)."""
    state = _new_stream_state()
    state["model"] = body.get("model")
    # httpx timeout: connect + read.  When connect_timeout is None the connect
    # timeout falls back to the main ``timeout`` value (no separate limit).
    _timeout = _httpx.Timeout(timeout, connect=connect_timeout)
    with _httpx.stream(
        "POST", url,
        json=body,
        headers=headers,
        timeout=_timeout,
    ) as resp:
        if resp.status_code >= 400:
            error_body = resp.read().decode("utf-8", errors="replace")
            if "max_output_tokens" in body and _is_unsupported_output_limit(resp.status_code, error_body):
                raise _UnsupportedOutputLimitError("AuthGPT: backend rejects max_output_tokens")
            reason = getattr(resp, "reason_phrase", "") or ""
            detail = error_body
            try:
                detail = json.loads(error_body).get("detail", error_body)
            except Exception:
                pass
            if not detail:
                detail = "empty-body"
            summary = detail or reason or "Bad Request"
            try:
                suppress = (resp.status_code == 429) and ("usage_limit_reached" in str(error_body).lower())
            except Exception:
                suppress = False
            if not suppress:
                _log(f"❌ AuthGPT HTTP {resp.status_code}. {summary}")
            raise _AuthGPTHTTPError(
                f"AuthGPT: {resp.status_code} – {summary} [reason={reason}]", resp.status_code, error_body
            )

        # iter_lines() in httpx yields str lines as they arrive
        for line in resp.iter_lines():
            if is_cancelled():
                resp.close()
                raise RuntimeError("AuthGPT: stream cancelled by user")
            if _process_sse_line(line, state, _log, log_stream, t_start):
                break

    return _finalize_stream(state, _log, log_stream, t_start)


# ---------------------------------------------------------------------------
# requests-based SSE reader (fallback — may buffer due to urllib3/http.client)
# ---------------------------------------------------------------------------

def _stream_with_requests(
    url: str,
    body: Dict,
    headers: Dict,
    timeout: int,
    t_start: float,
    _log,
    log_stream: bool,
) -> Dict:
    """Stream SSE using requests (fallback when httpx is not available)."""
    state = _new_stream_state()
    state["model"] = body.get("model")
    resp = requests.post(url, json=body, headers=headers, timeout=timeout, stream=True)

    if resp.status_code >= 400:
        try:
            error_body = resp.text
        except Exception:
            error_body = ""
        if "max_output_tokens" in body and _is_unsupported_output_limit(resp.status_code, error_body):
            resp.close()
            raise _UnsupportedOutputLimitError("AuthGPT: backend rejects max_output_tokens")
        try:
            reason = resp.reason or ""
        except Exception:
            reason = ""
        detail = error_body
        try:
            detail = resp.json().get("detail", error_body)
        except Exception:
            pass
        if not detail:
            detail = "empty-body"
        summary = detail or reason or "Bad Request"
        try:
            suppress = (resp.status_code == 429) and ("usage_limit_reached" in str(error_body).lower())
        except Exception:
            suppress = False
        if not suppress:
            _log(f"❌ AuthGPT HTTP {resp.status_code}. {summary}")
        resp.close()
        raise _AuthGPTHTTPError(
            f"AuthGPT: {resp.status_code} – {summary} [reason={reason}]", resp.status_code, error_body
        )

    for raw_line in resp.iter_lines(chunk_size=1):
        if is_cancelled():
            resp.close()
            raise RuntimeError("AuthGPT: stream cancelled by user")
        if raw_line is None:
            continue
        line = raw_line.decode("utf-8", errors="replace") if isinstance(raw_line, bytes) else raw_line
        if _process_sse_line(line, state, _log, log_stream, t_start):
            break

    return _finalize_stream(state, _log, log_stream, t_start)


# ---------------------------------------------------------------------------
# Public API – send chat completion
# ---------------------------------------------------------------------------

def send_chat_completion(
    access_token: str,
    messages: List[Dict],
    model: str = "gpt-5.2",
    temperature: Optional[float] = 0.7,
    max_tokens: Optional[int] = None,
    timeout: int = 600,
    base_url: Optional[str] = None,
    log_fn: Optional[Any] = None,
    connect_timeout: Optional[float] = None,
    reasoning: Optional[Dict[str, Any]] = None,
) -> Dict:
    """Send a chat completion request via the ChatGPT Codex Responses API.

    Parameters
    ----------
    access_token : str
        OAuth access token (Bearer token).
    messages : list of dict
        Standard OpenAI-format messages (role + content).
    model : str
        Model name without the ``authgpt/`` prefix.
    temperature : float or None
        Accepted for caller compatibility; not sent to the Codex backend.
    max_tokens : int or None
        Maximum response tokens. Omitted for gpt-6-astra with a note. For
        other models, send as max_output_tokens; if the backend explicitly
        rejects it, retry once without it and log that the limit cannot be
        enforced.
    timeout : int
        Request timeout in seconds.
    base_url : str or None
        Override the ChatGPT backend base URL.

    Returns
    -------
    dict
        ``{"content": str, "finish_reason": str, "usage": dict|None,
           "conversation_id": str|None, "message_id": str|None}``

    Raises
    ------
    requests.HTTPError
        On non-200 responses from the backend.
    RuntimeError
        On unexpected response format.
    """
    effective_base = base_url or os.getenv("AUTHGPT_BASE_URL", CHATGPT_BASE_URL)
    url = f"{effective_base.rstrip('/')}{RESPONSES_ENDPOINT}"

    body = _build_responses_body(
        messages=messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        reasoning=reasoning,
    )

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "Accept-Encoding": "identity",  # Disable gzip so SSE streams in real-time
    }

    _log = log_fn or print
    normalize_none_effort(body, _log)
    logger.info("AuthGPT: POST %s  model=%s", url, model)
    if max_tokens is not None:
        if "max_output_tokens" in body:
            _log(f"📏 AuthGPT max_output_tokens={max_tokens}")
        else:
            _log(
                f"📝 AuthGPT ({model}): max_output_tokens is omitted because the backend does not support it. "
                f"The configured output limit ({max_tokens}) does not apply."
            )

    # AuthGPT always streams (the API requires it), so streaming log is on
    # by default.  During batch translation, silence it unless the user
    # explicitly enabled the authgpt-specific batch log toggle.
    log_stream = os.getenv("LOG_STREAM_CHUNKS", "1").lower() not in ("0", "false")
    if os.getenv("BATCH_TRANSLATION", "0") == "1":
        log_stream = os.getenv("ALLOW_AUTHGPT_BATCH_STREAM_LOGS", "0").lower() not in ("0", "false")

    t_start = time.time()

    # Use httpx for SSE streaming — its h11-based HTTP parser yields data as
    # it arrives from the socket, unlike requests/urllib3 which buffers
    # entire SSL records through http.client's internal BufferedIOBase.
    # This is the same HTTP stack the official openai Python SDK uses.
    try:
        import httpx as _httpx
    except ImportError:
        _httpx = None
        _log("⚠️ AuthGPT: httpx not installed, falling back to requests (streaming may be buffered)")

    def _send_once(request_body):
        if _httpx is not None:
            return _stream_with_httpx(
                _httpx, url, request_body, headers, timeout, t_start,
                _log, log_stream, connect_timeout=connect_timeout,
            )
        return _stream_with_requests(
            url, request_body, headers, timeout, t_start,
            _log, log_stream,
        )

    def _check_retry_cancel():
        if is_cancelled() or os.getenv('GRACEFUL_STOP') == '1':
            raise RuntimeError("AuthGPT: stream cancelled by user")

    def _send(request_body):
        return call_with_reasoning_retry(_send_once, request_body, _log, _check_retry_cancel)

    try:
        return _send(body)
    except _UnsupportedOutputLimitError:
        if is_cancelled():
            raise RuntimeError("AuthGPT: stream cancelled by user")
        retry_body = dict(body)
        retry_body.pop("max_output_tokens")
        _log(
            f"⚠️ AuthGPT ({model}): backend rejects max_output_tokens; retrying without it. "
            f"The configured output limit ({max_tokens}) cannot be enforced for this request."
        )
        return _send(retry_body)
