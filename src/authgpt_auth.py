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
"""
import os
import json
import time
import hashlib
import base64
import secrets
import logging
import threading
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlencode, urlparse, parse_qs
from typing import Optional, Dict, List, Tuple, Any

import requests

logger = logging.getLogger(__name__)

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

CHATGPT_BASE_URL = "https://chatgpt.com/backend-api"
RESPONSES_ENDPOINT = "/codex/responses"

_DEFAULT_TOKEN_DIR = os.path.join(os.path.expanduser("~"), ".glossarion")
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

            # Redirect to a friendly success page
            self.send_response(302)
            self.send_header("Location", f"http://{CALLBACK_HOST}:{self.server.server_port}/success")
            self.end_headers()

        elif parsed.path == "/success":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            html = (
                "<html><body style='font-family:sans-serif;text-align:center;padding-top:60px'>"
                "<h1>&#10004; Authenticated!</h1>"
                "<p>You can close this tab and return to Glossarion.</p>"
                "</body></html>"
            )
            self.wfile.write(html.encode("utf-8"))
            # Signal the server to stop (handled in a thread below)
            threading.Thread(target=self.server.shutdown, daemon=True).start()

        else:
            self.send_response(404)
            self.end_headers()


def _find_available_port() -> int:
    """Return the callback port for OAuth.

    OpenAI's Codex client ID (app_EMoamEEZ73f0CkXaXp7hrann) has a fixed
    redirect URI registered as http://localhost:1455/auth/callback.
    We *must* use port 1455 or OpenAI will reject the request.
    """
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((CALLBACK_HOST, CALLBACK_PORT))
            return CALLBACK_PORT
    except OSError:
        raise RuntimeError(
            f"Port {CALLBACK_PORT} is already in use. "
            "Please close any application using that port and try again. "
            "(OpenAI's OAuth requires this exact port.)"
        )


# ===========================================================================
# OAuth flow orchestrator
# ===========================================================================

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
    """
    port = _find_available_port()
    redirect_uri = f"http://{CALLBACK_HOST}:{port}{CALLBACK_PATH}"
    code_verifier, code_challenge = generate_pkce()
    state = secrets.token_urlsafe(32)

    auth_url = build_auth_url(code_challenge, state, redirect_uri)

    # Start local callback server
    server = HTTPServer((CALLBACK_HOST, port), _OAuthCallbackHandler)
    server._auth_code = None
    server._returned_state = None
    server._error = None
    server.timeout = timeout

    print(f"🔐 Opening browser for ChatGPT login…")
    print(f"   If the browser doesn't open, visit:\n   {auth_url}")
    webbrowser.open(auth_url)

    # Serve until callback is received or timeout
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    server_thread.join(timeout=timeout)

    # Cleanup
    server.shutdown()

    if server._error:
        raise RuntimeError(f"OAuth error: {server._error}")
    if not server._auth_code:
        raise RuntimeError("OAuth login timed out – no callback received.")
    if server._returned_state != state:
        raise RuntimeError("OAuth state mismatch – possible CSRF attack.")

    # Exchange code for tokens
    print("🔑 Exchanging authorization code for tokens…")
    tokens = exchange_code_for_tokens(server._auth_code, code_verifier, redirect_uri)
    print("✅ ChatGPT OAuth authentication successful!")

    # Extract and log account info (non-sensitive)
    id_token = tokens.get("id_token", "")
    if id_token:
        info = extract_account_info(id_token)
        plan = info.get("plan_type", "unknown")
        email = info.get("email", "")
        if email:
            print(f"   Account: {email} (plan: {plan})")

    return tokens


# ===========================================================================
# Token store (persistent, thread-safe)
# ===========================================================================

class AuthGPTTokenStore:
    """Thread-safe token store backed by a JSON file."""

    def __init__(self, token_file: Optional[str] = None):
        self._token_file = (
            token_file
            or os.environ.get("AUTHGPT_TOKEN_FILE")
            or _DEFAULT_TOKEN_FILE
        )
        self._lock = threading.RLock()
        self._tokens: Optional[Dict] = None
        # Eagerly load cached tokens from disk (if any)
        self._load_from_disk()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _ensure_dir(self):
        d = os.path.dirname(self._token_file)
        if d:
            os.makedirs(d, exist_ok=True)

    def _load_from_disk(self):
        """Load tokens from the JSON file into memory."""
        try:
            if os.path.isfile(self._token_file):
                with open(self._token_file, "r", encoding="utf-8") as f:
                    self._tokens = json.load(f)
                logger.debug("AuthGPT tokens loaded from %s", self._token_file)
        except Exception as exc:
            logger.warning("Failed to load authgpt tokens: %s", exc)
            self._tokens = None

    def save_tokens(self, tokens: Dict):
        """Save tokens to disk and cache in memory."""
        with self._lock:
            self._tokens = tokens
            try:
                self._ensure_dir()
                with open(self._token_file, "w", encoding="utf-8") as f:
                    json.dump(tokens, f, indent=2)
                logger.debug("AuthGPT tokens saved to %s", self._token_file)
            except Exception as exc:
                logger.warning("Failed to save authgpt tokens: %s", exc)

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

            print("🔄 AuthGPT: No valid token found – starting browser login…")
            new_tokens = run_oauth_flow()
            self.save_tokens(new_tokens)
            return new_tokens["access_token"]

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
) -> Dict:
    """Build a Codex Responses API request body from standard OpenAI messages.

    The /backend-api/codex/responses endpoint expects:
      - "instructions": system-level instructions (required)
      - "input": list of {type, role, content:[{type, text}]} message objects
      - "model", "store", and optional temperature / max_output_tokens
    """
    instructions = ""
    input_items: List[Dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        # System messages become the instructions blob
        if role == "system":
            instructions = content
            continue

        # Map to Responses API structured message format
        input_items.append({
            "type": "message",
            "role": "developer" if role == "system" else role,
            "content": [{"type": "input_text", "text": content}],
        })

    body: Dict[str, Any] = {
        "model": model,
        "instructions": instructions,
        "input": input_items,
        "store": False,
        "stream": True,
    }

    if max_tokens is not None:
        body["max_output_tokens"] = max_tokens

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

    # Usage
    raw_usage = data.get("usage")
    if raw_usage:
        usage = {
            "prompt_tokens": raw_usage.get("input_tokens", 0),
            "completion_tokens": raw_usage.get("output_tokens", 0),
            "total_tokens": raw_usage.get("total_tokens", 0),
        }

    return {
        "content": content,
        "finish_reason": finish_reason,
        "conversation_id": response_id,
        "message_id": response_id,
        "usage": usage,
    }


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
        elif event_type == "response.completed":
            resp_obj = data.get("response", data)
            return _parse_responses_result(resp_obj)

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
        Sampling temperature.
    max_tokens : int or None
        Maximum response tokens.
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
    )

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }

    logger.info("AuthGPT: POST %s  model=%s", url, model)

    resp = requests.post(url, json=body, headers=headers, timeout=timeout, stream=True)

    if resp.status_code >= 400:
        # Read the error body so it can be included in the exception
        try:
            error_body = resp.text[:500]
        except Exception:
            error_body = ""
        # Try to extract a detail message from JSON
        detail = error_body
        try:
            detail = resp.json().get("detail", error_body)
        except Exception:
            pass
        raise RuntimeError(
            f"AuthGPT: {resp.status_code} – {detail}"
        )

    # Read the SSE stream line by line
    raw_lines: List[str] = []
    for line in resp.iter_lines(decode_unicode=True):
        if line is not None:
            raw_lines.append(line)
    raw_text = "\n".join(raw_lines)
    result = _parse_sse_responses(raw_text)

    if not result.get("content"):
        logger.warning("AuthGPT: empty response from backend (model=%s)", model)

    return result
