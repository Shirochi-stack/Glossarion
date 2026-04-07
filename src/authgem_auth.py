# authgem_auth.py - Gemini subscription OAuth authentication
# Uses the same OAuth2 credentials as Gemini CLI (google-gemini/gemini-cli).
# Prefix models with 'authgem/' to route through the Gemini API using your
# Google account instead of a GEMINI_API_KEY.
"""
OAuth 2.0 flow for Gemini API authentication via Google Account, persistent
token storage with automatic refresh, and Gemini API adapter.

Flow:
  1. Open browser to Google OAuth consent screen
  2. Spin up a local HTTP callback server on a random available port
  3. User logs in via browser → callback receives auth code
  4. Exchange auth code for access + refresh tokens
  5. Store tokens locally (~/.glossarion/authgem_tokens.json)
  6. Use access token as Bearer auth for generativelanguage.googleapis.com
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

# Module-level cancellation flag — set by unified_api_client.hard_cancel_all()
_cancel_event = threading.Event()


def cancel_stream():
    """Signal any active AuthGem stream to abort immediately."""
    _cancel_event.set()


def reset_cancel():
    """Clear the cancellation flag (call before starting a new request)."""
    _cancel_event.clear()


def is_cancelled() -> bool:
    return _cancel_event.is_set()


# ===========================================================================
# Constants – Google ADC OAuth client (same as gcloud auth)
# ===========================================================================
# These are the *public* OAuth credentials used by Google's own Cloud SDK
# (gcloud auth application-default login).  They are embedded in every gcloud
# installation and have ALL Google API scopes pre-registered, which means the
# "generative-language" scope works out of the box with zero user setup.
#
# Source: google-auth-library-python → google/auth/_cloud_sdk.py
# They are NOT secrets — Google treats installed-app client secrets as public.
#
# Users can override with AUTHGEM_CLIENT_ID / AUTHGEM_CLIENT_SECRET env vars
# if they prefer to use their own GCP project credentials.
_ADC_CLIENT_ID = (
    "764086051850-6qr4p6gpi6hn506pt8ejuq83di341hur"
    ".apps.googleusercontent.com"
)
_ADC_CLIENT_SECRET = "d-FL95Q19q7MQmFpd7hHD0Ty"

GOOGLE_CLIENT_ID = os.environ.get("AUTHGEM_CLIENT_ID", _ADC_CLIENT_ID)
GOOGLE_CLIENT_SECRET = os.environ.get("AUTHGEM_CLIENT_SECRET", _ADC_CLIENT_SECRET)

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v2/userinfo"

# Scopes needed for Gemini API access + user info
OAUTH_SCOPES = [
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
]

CALLBACK_HOST = "127.0.0.1"
CALLBACK_PATH = "/oauth2callback"

TOKEN_REFRESH_MARGIN_SECONDS = 300  # refresh when <5 min remaining

# Vertex AI endpoint — works with cloud-platform scope (unlike AI Studio)
# Users MUST set GOOGLE_CLOUD_PROJECT.  GOOGLE_CLOUD_LOCATION defaults to
# us-central1 which has the widest model availability.
VERTEX_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
VERTEX_LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

_DEFAULT_TOKEN_DIR = os.path.join(os.path.expanduser("~"), ".glossarion")
_DEFAULT_TOKEN_FILE = os.path.join(_DEFAULT_TOKEN_DIR, "authgem_tokens.json")

# Success/failure redirect URLs after OAuth (same as gemini-cli)
SIGN_IN_SUCCESS_URL = "https://developers.google.com/gemini-code-assist/auth_success_gemini"
SIGN_IN_FAILURE_URL = "https://developers.google.com/gemini-code-assist/auth_failure_gemini"


# ===========================================================================
# Port helpers
# ===========================================================================

def _find_available_port() -> int:
    """Find an available port for the OAuth callback server."""
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((CALLBACK_HOST, 0))
    port = s.getsockname()[1]
    s.close()
    return port


# ===========================================================================
# Token exchange / refresh
# ===========================================================================

def exchange_code_for_tokens(
    auth_code: str,
    redirect_uri: str,
) -> Dict:
    """Exchange authorization code for access + refresh tokens.

    Returns dict with keys: access_token, refresh_token, expires_in, …
    """
    payload = {
        "grant_type": "authorization_code",
        "client_id": GOOGLE_CLIENT_ID,
        "client_secret": GOOGLE_CLIENT_SECRET,
        "code": auth_code,
        "redirect_uri": redirect_uri,
    }
    resp = requests.post(GOOGLE_TOKEN_URL, data=payload, timeout=30)
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
        "client_id": GOOGLE_CLIENT_ID,
        "client_secret": GOOGLE_CLIENT_SECRET,
        "refresh_token": refresh_token,
    }
    resp = requests.post(GOOGLE_TOKEN_URL, data=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    data["expires_at"] = time.time() + data.get("expires_in", 3600)
    return data


# ===========================================================================
# User info helpers
# ===========================================================================

def fetch_user_info(access_token: str) -> Dict:
    """Fetch user info (email, name) from Google's userinfo endpoint."""
    try:
        resp = requests.get(
            GOOGLE_USERINFO_URL,
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=15,
        )
        if resp.ok:
            return resp.json()
    except Exception as exc:
        logger.debug("Failed to fetch Google user info: %s", exc)
    return {}


def extract_account_info(tokens: Dict) -> Dict:
    """Extract account info from stored tokens or fetch if needed."""
    # Check if we already have cached user info
    cached = tokens.get("_user_info")
    if cached:
        return cached
    return {}


# ===========================================================================
# Local callback server
# ===========================================================================

class _OAuthCallbackHandler(BaseHTTPRequestHandler):
    """HTTP request handler that captures the Google OAuth callback."""

    def log_message(self, format, *args):  # noqa: A002
        # Suppress default stderr logging
        pass

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == CALLBACK_PATH:
            qs = parse_qs(parsed.query)
            error = qs.get("error", [None])[0]

            if error:
                self.server._error = error
                self.send_response(302)
                self.send_header("Location", SIGN_IN_FAILURE_URL)
                self.end_headers()
            else:
                code = qs.get("code", [None])[0]
                state = qs.get("state", [None])[0]
                self.server._auth_code = code
                self.server._returned_state = state

                # Show a success page, then redirect
                self.send_response(302)
                self.send_header("Location", f"http://{CALLBACK_HOST}:{self.server.server_port}/success")
                self.end_headers()

        elif parsed.path == "/success":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            html = (
                "<html><body style='font-family:sans-serif;text-align:center;padding-top:60px;"
                "background:#1a1a2e;color:#e0e0e0'>"
                "<h1 style='color:#4db8ff'>&#10004; Gemini Authenticated!</h1>"
                "<p>You can close this tab and return to Glossarion.</p>"
                "</body></html>"
            )
            self.wfile.write(html.encode("utf-8"))
            # Signal the server to stop
            threading.Thread(target=self.server.shutdown, daemon=True).start()

        else:
            self.send_response(404)
            self.end_headers()


# ===========================================================================
# OAuth flow orchestrator
# ===========================================================================

def run_oauth_flow(timeout: int = 300) -> Dict:
    """Run the full Google OAuth login flow for Gemini.

    1. Opens a browser for the user to authenticate with Google.
    2. Captures the callback on a local server.
    3. Exchanges the code for tokens.
    4. Fetches user info (email, etc.).

    Returns the token dict (access_token, refresh_token, expires_at, …).
    Raises RuntimeError on failure or timeout.

    Parameters
    ----------
    timeout : int
        Maximum seconds to wait for the user to complete the browser login.
    """
    port = _find_available_port()
    redirect_uri = f"http://{CALLBACK_HOST}:{port}{CALLBACK_PATH}"
    state = secrets.token_urlsafe(32)

    params = {
        "client_id": GOOGLE_CLIENT_ID,
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "scope": " ".join(OAUTH_SCOPES),
        "state": state,
        "access_type": "offline",
        "prompt": "consent",  # Always show consent to get refresh_token
    }
    auth_url = f"{GOOGLE_AUTH_URL}?{urlencode(params)}"

    # Start local callback server
    server = HTTPServer((CALLBACK_HOST, port), _OAuthCallbackHandler)
    server._auth_code = None
    server._returned_state = None
    server._error = None
    server.timeout = timeout

    print(f"🔐 Opening browser for Gemini (Google) login…")
    print(f"   If the browser doesn't open, visit:\n   {auth_url}")
    webbrowser.open(auth_url)

    # Serve until callback is received or timeout
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    server_thread.join(timeout=timeout)

    # Cleanup
    server.shutdown()

    if server._error:
        raise RuntimeError(f"Google OAuth error: {server._error}")
    if not server._auth_code:
        raise RuntimeError("Google OAuth login timed out – no callback received.")
    if server._returned_state != state:
        raise RuntimeError("OAuth state mismatch – possible CSRF attack.")

    # Exchange code for tokens
    print("🔑 Exchanging authorization code for tokens…")
    tokens = exchange_code_for_tokens(server._auth_code, redirect_uri)
    print("✅ Gemini OAuth authentication successful!")

    # Fetch and cache user info
    access_token = tokens.get("access_token", "")
    if access_token:
        user_info = fetch_user_info(access_token)
        if user_info:
            tokens["_user_info"] = {
                "email": user_info.get("email", ""),
                "name": user_info.get("name", ""),
                "picture": user_info.get("picture", ""),
            }
            email = user_info.get("email", "")
            if email:
                print(f"   Account: {email}")

    return tokens


# ===========================================================================
# Token store (persistent, thread-safe) – mirrors AuthGPTTokenStore API
# ===========================================================================

class AuthGemTokenStore:
    """Thread-safe token store backed by a JSON file."""

    def __init__(self, token_file: Optional[str] = None):
        self._token_file = (
            token_file
            or os.environ.get("AUTHGEM_TOKEN_FILE")
            or _DEFAULT_TOKEN_FILE
        )
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
        """Load tokens from the JSON file into memory."""
        try:
            if os.path.isfile(self._token_file):
                with open(self._token_file, "r", encoding="utf-8") as f:
                    self._tokens = json.load(f)
                logger.debug("AuthGem tokens loaded from %s", self._token_file)
        except Exception as exc:
            logger.warning("Failed to load authgem tokens: %s", exc)
            self._tokens = None

    def save_tokens(self, tokens: Dict):
        """Save tokens to disk and cache in memory."""
        with self._lock:
            self._tokens = tokens
            try:
                self._ensure_dir()
                with open(self._token_file, "w", encoding="utf-8") as f:
                    json.dump(tokens, f, indent=2)
                logger.debug("AuthGem tokens saved to %s", self._token_file)
            except Exception as exc:
                logger.warning("Failed to save authgem tokens: %s", exc)
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
                    logger.info("AuthGem tokens removed")
            except Exception as exc:
                logger.warning("Failed to remove token file: %s", exc)
        self._fire_change_callbacks()

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
            logger.info("AuthGem access token refreshed successfully")
            return merged
        except Exception as exc:
            logger.warning("AuthGem token refresh failed: %s", exc)
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
                    "AuthGem: No valid tokens and auto_login is disabled. "
                    "Run the OAuth login flow first."
                )

            # Detect headless environments
            is_headless = (
                os.environ.get("SPACE_ID") is not None
                or os.environ.get("HF_SPACES") == "true"
                or os.environ.get("DOCKER_CONTAINER") == "true"
                or os.environ.get("KUBERNETES_SERVICE_HOST") is not None
            )
            if is_headless:
                env_access = os.environ.get("AUTHGEM_ACCESS_TOKEN", "").strip()
                env_refresh = os.environ.get("AUTHGEM_REFRESH_TOKEN", "").strip()
                if env_access:
                    manual_tokens = {
                        "access_token": env_access,
                        "expires_at": time.time() + 3600,
                    }
                    if env_refresh:
                        manual_tokens["refresh_token"] = env_refresh
                    self.save_tokens(manual_tokens)
                    logger.info("AuthGem: Using access token from AUTHGEM_ACCESS_TOKEN env var")
                    return env_access
                if env_refresh:
                    try:
                        refreshed = refresh_access_token(env_refresh)
                        self.save_tokens(refreshed)
                        logger.info("AuthGem: Obtained access token via AUTHGEM_REFRESH_TOKEN env var")
                        return refreshed["access_token"]
                    except Exception as ref_exc:
                        raise RuntimeError(
                            f"AuthGem: AUTHGEM_REFRESH_TOKEN was set but refresh failed: {ref_exc}\n"
                            "The refresh token may be expired. Please obtain a new one."
                        )
                raise RuntimeError(
                    "AuthGem: Browser-based OAuth login is not available in headless environments.\n"
                    "To use AuthGem models, set one of these as environment secrets:\n"
                    "  • AUTHGEM_ACCESS_TOKEN — a valid Google OAuth access token\n"
                    "  • AUTHGEM_REFRESH_TOKEN — a Google OAuth refresh token (will auto-refresh)\n"
                    "You can obtain these by running the OAuth flow locally first, then copying\n"
                    "the tokens from ~/.glossarion/authgem_tokens.json"
                )

            print("🔄 AuthGem: No valid token found – starting browser login…")
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
        """Return account info from cached tokens, or empty dict."""
        tokens = self.load_tokens()
        if not tokens:
            return {}
        user_info = tokens.get("_user_info", {})
        return {
            "email": user_info.get("email", ""),
            "name": user_info.get("name", ""),
        }


# Module-level singleton for convenience (lazy-initialized)
_default_store: Optional[AuthGemTokenStore] = None
_default_store_lock = threading.Lock()


def get_default_store() -> AuthGemTokenStore:
    """Return the module-level default token store (singleton)."""
    global _default_store
    if _default_store is None:
        with _default_store_lock:
            if _default_store is None:
                _default_store = AuthGemTokenStore()
    return _default_store


# ===========================================================================
# Gemini API adapter
# ===========================================================================

def _build_gemini_request_body(
    messages: List[Dict],
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> Dict:
    """Convert OpenAI-style messages to Gemini API generateContent format.

    Maps:
      - system → systemInstruction
      - user → user parts
      - assistant → model parts
    """
    system_parts = []
    contents = []

    for msg in messages:
        role = msg.get("role", "user")
        text = msg.get("content", "")
        if not text:
            continue

        if role == "system":
            system_parts.append({"text": text})
        elif role == "user":
            contents.append({
                "role": "user",
                "parts": [{"text": text}],
            })
        elif role == "assistant":
            contents.append({
                "role": "model",
                "parts": [{"text": text}],
            })
        else:
            # Default to user
            contents.append({
                "role": "user",
                "parts": [{"text": text}],
            })

    body = {"contents": contents}

    if system_parts:
        body["systemInstruction"] = {"parts": system_parts}

    # Generation config
    gen_config = {}
    if temperature is not None:
        gen_config["temperature"] = temperature
    if max_tokens is not None:
        gen_config["maxOutputTokens"] = max_tokens
    if gen_config:
        body["generationConfig"] = gen_config

    return body


def send_chat_completion(
    access_token: str,
    messages: List[Dict],
    model: str = "gemini-2.5-flash",
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    timeout: int = 300,
    log_fn=None,
    connect_timeout: Optional[float] = None,
) -> Dict:
    """Send a chat completion request to the Gemini API using OAuth token.

    Parameters
    ----------
    access_token : str
        Valid Google OAuth access token (Bearer).
    messages : list
        OpenAI-style message list (role/content dicts).
    model : str
        Gemini model name (e.g. 'gemini-2.5-flash', 'gemini-2.5-pro').
    temperature : float, optional
        Sampling temperature.
    max_tokens : int, optional
        Maximum output tokens.
    timeout : int
        Request timeout in seconds.
    log_fn : callable, optional
        Logging function (e.g. print).
    connect_timeout : float, optional
        Separate connect timeout.

    Returns
    -------
    dict
        {'content': str, 'finish_reason': str, 'usage': dict or None}
    """
    if is_cancelled():
        raise RuntimeError("AuthGem: stream cancelled by user")

    _log = log_fn or (lambda *a, **kw: None)

    body = _build_gemini_request_body(messages, temperature, max_tokens)

    # Resolve project — env var, or stored in tokens, or fail with clear message
    project = VERTEX_PROJECT or os.environ.get("GCLOUD_PROJECT", "")
    location = VERTEX_LOCATION
    if not project:
        raise RuntimeError(
            "AuthGem: GOOGLE_CLOUD_PROJECT environment variable is required.\n"
            "Set it to your GCP project ID, e.g.:\n"
            "  set GOOGLE_CLOUD_PROJECT=my-project-id\n"
            "You can find your project ID at https://console.cloud.google.com"
        )

    url = (
        f"https://{location}-aiplatform.googleapis.com/v1beta1/"
        f"projects/{project}/locations/{location}/"
        f"publishers/google/models/{model}:generateContent"
    )
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }

    # Build timeout tuple
    if connect_timeout is not None:
        req_timeout = (connect_timeout, timeout)
    else:
        req_timeout = timeout

    try:
        resp = requests.post(url, json=body, headers=headers, timeout=req_timeout)
    except requests.exceptions.Timeout:
        raise RuntimeError(f"AuthGem: Request timed out after {timeout}s")
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(f"AuthGem: Connection error: {exc}")

    if is_cancelled():
        raise RuntimeError("AuthGem: stream cancelled by user")

    if not resp.ok:
        error_text = resp.text[:500] if resp.text else "No response body"
        raise RuntimeError(f"AuthGem: {resp.status_code} - {error_text}")

    data = resp.json()

    # Parse the Gemini response
    candidates = data.get("candidates", [])
    if not candidates:
        # Check for promptFeedback (blocked)
        feedback = data.get("promptFeedback", {})
        block_reason = feedback.get("blockReason", "")
        if block_reason:
            raise RuntimeError(f"AuthGem: Prompt blocked - {block_reason}")
        raise RuntimeError("AuthGem: No candidates in response")

    candidate = candidates[0]
    content_parts = candidate.get("content", {}).get("parts", [])

    # Extract text, filtering out thought parts
    text_parts = []
    for part in content_parts:
        # Skip thought parts (Gemini's internal reasoning)
        if part.get("thought", False):
            continue
        if "text" in part:
            text_parts.append(part["text"])

    content = "".join(text_parts)
    finish_reason = candidate.get("finishReason", "STOP")

    # Map Gemini finish reasons to OpenAI-style
    finish_reason_map = {
        "STOP": "stop",
        "MAX_TOKENS": "length",
        "SAFETY": "content_filter",
        "RECITATION": "content_filter",
    }
    mapped_reason = finish_reason_map.get(finish_reason, finish_reason.lower())

    # Extract usage metadata
    usage_metadata = data.get("usageMetadata", {})
    usage = None
    if usage_metadata:
        usage = {
            "prompt_tokens": usage_metadata.get("promptTokenCount", 0),
            "completion_tokens": usage_metadata.get("candidatesTokenCount", 0),
            "total_tokens": usage_metadata.get("totalTokenCount", 0),
        }

    return {
        "content": content,
        "finish_reason": mapped_reason,
        "usage": usage,
    }
