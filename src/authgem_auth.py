# authgem_auth.py - Gemini subscription OAuth authentication
# Uses the same OAuth2 credentials as Gemini CLI (google-gemini/gemini-cli).
#
# Routing prefixes:
#   authgem/         → Google AI Studio via OAuth (no GCP project needed)
#   authgem-key/     → Google AI Studio via GEMINI_API_KEY (no OAuth needed)
#   authgem-vertex/  → Vertex AI via OAuth (requires GCP project w/ billing)
"""
OAuth 2.0 flow for Gemini API authentication via Google Account, persistent
token storage with automatic refresh, and Gemini API adapters.

Three endpoint modes:
  - OAuth (authgem/): cloudcode-pa.googleapis.com Code Assist proxy (same as Gemini CLI)
  - AI Studio + API key: generativelanguage.googleapis.com with ?key= param
  - Vertex AI + OAuth: {region}-aiplatform.googleapis.com with Bearer token

OAuth Flow:
  1. Open browser to Google OAuth consent screen
  2. Spin up a local HTTP callback server on a random available port
  3. User logs in via browser → callback receives auth code
  4. Exchange auth code for access + refresh tokens
  5. Store tokens locally (~/.glossarion/authgem_tokens.json)
  6. Use access token as Bearer auth
"""
import os
import json
import time
import hashlib
import base64
import secrets
import uuid
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
# Constants – Gemini CLI OAuth client (same as google-gemini/gemini-cli)
# ===========================================================================
# These are the *public* OAuth credentials used by Google's own Gemini CLI.
# Source: google-gemini/gemini-cli → packages/core/src/code_assist/oauth2.ts
#
# They are NOT secrets — Google treats installed-app client secrets as public:
# https://developers.google.com/identity/protocols/oauth2#installed
# "The process results in a client ID and, in some cases, a client secret,
# which you embed in the source code of your application. (In this context,
# the client secret is obviously not treated as a secret.)"
#
# This client is registered with Google's Generative Language API, so it works
# for both AI Studio (generativelanguage.googleapis.com) AND Vertex AI.
# The gcloud ADC client (764086051850-...) does NOT work for AI Studio.
#
# Users can override with AUTHGEM_CLIENT_ID / AUTHGEM_CLIENT_SECRET env vars
# if they prefer to use their own GCP project credentials.
_GEMINI_CLI_CLIENT_ID = (
    "681255809395-oo8ft2oprdrnp9e3aqf6av3hmdib135j"
    ".apps.googleusercontent.com"
)
_GEMINI_CLI_CLIENT_SECRET = "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"

GOOGLE_CLIENT_ID = os.environ.get("AUTHGEM_CLIENT_ID", _GEMINI_CLI_CLIENT_ID)
GOOGLE_CLIENT_SECRET = os.environ.get("AUTHGEM_CLIENT_SECRET", _GEMINI_CLI_CLIENT_SECRET)

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v2/userinfo"

# Scopes — cloud-platform covers Code Assist proxy (cloudcode-pa.googleapis.com)
# which is what the Gemini CLI uses for "Sign in with Google".
# Direct OAuth to generativelanguage.googleapis.com is NOT possible with
# public OAuth clients (the generative-language.retriever scope is unregistered).
# For direct AI Studio access, use authgem-key/ (API key) instead.
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

# Thinking level → budget mapping (same as grpc_gemini_client.py line 650)
# Used by authgem-vertex/ and authgem/ to convert thinkingLevel to thinkingBudget,
# since the Vertex AI REST API and Code Assist proxy suppress thought annotations
# (thought: true on response parts) when thinkingLevel is present.
# AI Studio (authgem-key/) handles thinkingLevel correctly, so no conversion needed there.
_THINKING_LEVEL_TO_BUDGET_MAP = {
    'MINIMAL': 0,
    'LOW': 4096,
    'MEDIUM': 12288,
    'HIGH': 32768,
}


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
# GCP project auto-detection
# ===========================================================================

_cached_project_id: Optional[str] = None


def detect_gcp_project(access_token: str) -> Optional[str]:
    """Auto-detect the user's GCP project ID using the Resource Manager API.

    Uses the OAuth token to list projects the user has access to.
    Only selects projects with billing enabled.  Result is cached for the
    session.  The GUI dropdown can override this by setting
    ``_cached_project_id`` directly.
    """
    global _cached_project_id
    if _cached_project_id:
        return _cached_project_id

    # Check env vars first
    for env_key in ("GOOGLE_CLOUD_PROJECT", "GCLOUD_PROJECT", "GCP_PROJECT"):
        val = os.environ.get(env_key, "").strip()
        if val:
            _cached_project_id = val
            return val

    headers = {"Authorization": f"Bearer {access_token}"}
    try:
        resp = requests.get(
            "https://cloudresourcemanager.googleapis.com/v1/projects",
            headers=headers,
            params={"filter": "lifecycleState:ACTIVE", "pageSize": 50},
            timeout=15,
        )
        if resp.ok:
            projects = resp.json().get("projects", [])
            # Check billing for each project — only pick one with billing
            for p in projects:
                pid = p.get("projectId", "")
                if not pid:
                    continue
                try:
                    # Try Cloud Billing API first
                    br = requests.get(
                        f"https://cloudbilling.googleapis.com/v1/projects/{pid}/billingInfo",
                        headers=headers, timeout=10,
                    )
                    if br.ok and br.json().get("billingEnabled", False):
                        _cached_project_id = pid
                        logger.info("AuthGem: Auto-detected GCP project (billing OK): %s", pid)
                        print(f"🔍 AuthGem: Using GCP project: {pid}")
                        return pid
                    elif br.ok:
                        # Billing API says explicitly not billed
                        continue
                    # Billing API returned 403 — probe Vertex AI directly
                    # If getting location info succeeds, billing is active
                    vr = requests.get(
                        f"https://us-central1-aiplatform.googleapis.com/v1/projects/{pid}/locations/us-central1",
                        headers=headers, timeout=8,
                    )
                    if vr.ok:
                        _cached_project_id = pid
                        logger.info("AuthGem: Auto-detected GCP project (Vertex AI probe OK): %s", pid)
                        print(f"🔍 AuthGem: Using GCP project: {pid}")
                        return pid
                except Exception:
                    continue

            # If no billed project found, warn
            if projects:
                names = ", ".join(p.get("projectId", "?") for p in projects[:5])
                logger.warning("AuthGem: No projects with billing enabled found. Projects: %s", names)
                print(f"⚠️ AuthGem: No GCP projects with billing enabled (checked: {names})")
    except Exception as exc:
        logger.debug("Failed to auto-detect GCP project: %s", exc)

    return None


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
                except Exception as dec_exc:
                    # Decryption failed — file is corrupt or from a different
                    # user/machine.  Delete it so re-login creates a fresh one.
                    logger.warning("AuthGem token decryption failed (%s) — removing corrupt file", dec_exc)
                    try:
                        os.remove(self._token_file)
                    except OSError:
                        pass
                    self._tokens = None
                    return
                logger.debug("AuthGem tokens loaded from %s", self._token_file)
        except Exception as exc:
            logger.warning("Failed to load authgem tokens: %s", exc)
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
                    pass
                except Exception as enc_exc:
                    logger.warning("AuthGem token encryption failed (%s) — saving as plain JSON", enc_exc)
                if not saved:
                    # Fallback: plain JSON (still better than losing tokens)
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

def _convert_content_to_gemini_parts(content) -> List[Dict]:
    """Convert OpenAI-style content (string or multimodal list) to Gemini parts.

    Handles:
      - Plain string → [{"text": "..."}]
      - List of parts (OpenAI multimodal format):
          {"type": "text", "text": "..."} → {"text": "..."}
          {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAA"}}
              → {"inlineData": {"mimeType": "image/jpeg", "data": "AAA"}}
    """
    if isinstance(content, str):
        return [{"text": content}]

    if not isinstance(content, list):
        return [{"text": str(content)}]

    parts = []
    for item in content:
        if isinstance(item, str):
            parts.append({"text": item})
            continue
        if not isinstance(item, dict):
            parts.append({"text": str(item)})
            continue

        item_type = item.get("type", "")

        if item_type == "text":
            text_val = item.get("text", "")
            if text_val:
                parts.append({"text": text_val})

        elif item_type in ("image_url", "image"):
            # OpenAI format: {"type": "image_url", "image_url": {"url": "data:mime;base64,..."}}
            image_url_obj = item.get("image_url", {})
            url = image_url_obj.get("url", "") if isinstance(image_url_obj, dict) else str(image_url_obj)

            if url.startswith("data:") and "base64," in url:
                # Parse data URI → inlineData
                # Format: data:image/jpeg;base64,/9j/4AAQ...
                header, b64_data = url.split("base64,", 1)
                # Extract MIME type from "data:image/jpeg;..."
                mime_type = header.replace("data:", "").rstrip(";").strip()
                if not mime_type:
                    mime_type = "image/jpeg"
                parts.append({
                    "inlineData": {
                        "mimeType": mime_type,
                        "data": b64_data,
                    }
                })
            elif url.startswith(("http://", "https://")):
                # URL-referenced image — use fileData for Vertex AI
                parts.append({
                    "fileData": {
                        "mimeType": "image/jpeg",
                        "fileUri": url,
                    }
                })
            else:
                logger.warning("AuthGem: Unsupported image URL format (not data: or http): %.80s…", url)
        else:
            # Unknown part type — try to extract text
            text_val = item.get("text", "") or item.get("content", "")
            if text_val:
                parts.append({"text": str(text_val)})

    return parts if parts else [{"text": ""}]


def _build_gemini_request_body(
    messages: List[Dict],
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    model: str = "",
) -> Dict:
    """Convert OpenAI-style messages to Gemini API generateContent format.

    Maps:
      - system → systemInstruction
      - user → user parts (text and/or images)
      - assistant → model parts

    Reads thinking settings from environment variables:
      - ENABLE_GEMINI_THINKING: "1" (default) or "0"
      - THINKING_BUDGET: 0=disabled, 512-24576=limited, -1=dynamic (default)
      - GEMINI_THINKING_LEVEL: minimal/low/medium/high (Gemini 3)
    """
    system_parts = []
    contents = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if not content:
            continue

        gemini_parts = _convert_content_to_gemini_parts(content)

        if role == "system":
            # System instruction only supports text parts
            for p in gemini_parts:
                if "text" in p:
                    system_parts.append(p)
        elif role == "user":
            contents.append({
                "role": "user",
                "parts": gemini_parts,
            })
        elif role == "assistant":
            contents.append({
                "role": "model",
                "parts": gemini_parts,
            })
        else:
            contents.append({
                "role": "user",
                "parts": gemini_parts,
            })

    body = {"contents": contents}

    if system_parts:
        body["systemInstruction"] = {"parts": system_parts}

    # Generation config
    gen_config: Dict = {}
    if temperature is not None:
        gen_config["temperature"] = temperature
    if max_tokens is not None:
        gen_config["maxOutputTokens"] = max_tokens

    # ── Thinking / reasoning configuration ──
    enable_thinking = os.getenv("ENABLE_GEMINI_THINKING", "1") == "1"
    thinking_budget = int(os.getenv("THINKING_BUDGET", "-1"))
    thinking_level = os.getenv("GEMINI_THINKING_LEVEL", "high").lower()
    stream_thinking = os.getenv("STREAM_THINKING_LOGS", "1") not in ("0", "false")

    model_lower = model.lower() if model else ""
    is_gemini_3 = "gemini-3" in model_lower

    if thinking_level not in ("minimal", "low", "medium", "high"):
        thinking_level = "high"

    if enable_thinking:
        thinking_config: Dict = {}

        if is_gemini_3:
            # Gemini 3: level-based thinking
            if thinking_budget == 0:
                if "flash" in model_lower:
                    thinking_level = "minimal"
                else:
                    thinking_level = "low"
            thinking_config["thinkingLevel"] = thinking_level.upper()
            # Gemini 3 Pro doesn't support minimal
            if "pro" in model_lower and "flash" not in model_lower:
                if thinking_level == "minimal":
                    thinking_config["thinkingLevel"] = "LOW"
        else:
            # Gemini 2.5 and earlier: budget-based thinking
            if thinking_budget == 0:
                thinking_config["thinkingBudget"] = 0
            elif thinking_budget > 0:
                thinking_config["thinkingBudget"] = thinking_budget
            # -1 = dynamic (don't set budget, let model decide)

        # Include thoughts in stream so we can log them in real-time
        if stream_thinking:
            thinking_config["includeThoughts"] = True

        if thinking_config:
            gen_config["thinkingConfig"] = thinking_config
    else:
        # Thinking explicitly disabled
        if is_gemini_3:
            # Gemini 3 can't fully disable thinking; use lowest level
            lowest = "MINIMAL" if "flash" in model_lower else "LOW"
            gen_config["thinkingConfig"] = {"thinkingLevel": lowest}
        else:
            gen_config["thinkingConfig"] = {"thinkingBudget": 0}

    if gen_config:
        body["generationConfig"] = gen_config

    # ── Safety settings ──
    # AuthGem endpoints always disable safety filters (ignores DISABLE_GEMINI_SAFETY toggle).
    # Threshold is configurable via dropdown: OFF, BLOCK_NONE, BLOCK_ONLY_HIGH, etc.
    _threshold = os.getenv("GEMINI_SAFETY_THRESHOLD", "OFF").strip().upper()
    if _threshold not in ("OFF", "BLOCK_NONE", "BLOCK_ONLY_HIGH", "BLOCK_MEDIUM_AND_ABOVE", "BLOCK_LOW_AND_ABOVE"):
        _threshold = "OFF"
    body["safetySettings"] = [
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": _threshold},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": _threshold},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": _threshold},
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": _threshold},
        {"category": "HARM_CATEGORY_CIVIC_INTEGRITY", "threshold": _threshold},
    ]

    return body


# ===========================================================================
# Code Assist proxy (cloudcode-pa.googleapis.com) — OAuth Bearer token
# No GCP project required.  Used by the  authgem/  prefix.
# This is the same endpoint the Gemini CLI uses with "Sign in with Google".
# Accepts the cloud-platform scope; no generative-language.retriever needed.
# ===========================================================================

_CODE_ASSIST_ENDPOINT = "https://cloudcode-pa.googleapis.com"
_CODE_ASSIST_API_VERSION = "v1internal"

# Cached Code Assist project ID (set once per session by _code_assist_setup)
_code_assist_project_id: Optional[str] = None
_code_assist_setup_done = False
# Session ID for Code Assist — generated once per process, like Gemini CLI
_code_assist_session_id: str = str(uuid.uuid4())

def _code_assist_base_url() -> str:
    return f"{_CODE_ASSIST_ENDPOINT}/{_CODE_ASSIST_API_VERSION}"

def _code_assist_setup(access_token: str, _log=None) -> Optional[str]:
    """Onboard user with Code Assist (mirrors Gemini CLI's setupUser).

    Calls ``loadCodeAssist`` to register the user and obtain a managed
    GCP project ID.  For free-tier users the server creates a project
    automatically — no user configuration needed.

    Returns the project ID (may be None for some tiers).
    """
    global _code_assist_project_id, _code_assist_setup_done
    if _code_assist_setup_done:
        return _code_assist_project_id

    _log = _log or (lambda *a, **kw: None)
    url = f"{_code_assist_base_url()}:loadCodeAssist"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }
    payload = {
        "metadata": {
            "ideType": "IDE_UNSPECIFIED",
            "platform": "PLATFORM_UNSPECIFIED",
            "pluginType": "GEMINI",
        },
        "mode": "HEALTH_CHECK",
    }

    try:
        import httpx as _httpx
        resp = _httpx.post(url, json=payload, headers=headers, timeout=30)
        resp_data = resp.json()
    except ImportError:
        resp = requests.post(url, json=payload, headers=headers, timeout=30)
        resp_data = resp.json()
    except Exception as exc:
        logger.warning("Code Assist setup failed: %s", exc)
        _code_assist_setup_done = True
        return None

    project = resp_data.get("cloudaicompanionProject")
    tier = resp_data.get("currentTier", {})
    tier_name = tier.get("name", "unknown")
    tier_id = tier.get("id", "unknown")
    allowed_tiers = resp_data.get("allowedTiers", [])
    allowed_names = [t.get("name", t.get("id", "?")) for t in allowed_tiers] if allowed_tiers else []

    # Determine subscription level for user-friendly display
    tier_lower = tier_name.lower() if tier_name else ""
    if "ultra" in tier_lower:
        sub_label = "Google AI Ultra ✨"
    elif "pro" in tier_lower:
        sub_label = "Google AI Pro ⭐"
    elif "enterprise" in tier_lower:
        sub_label = "Enterprise 🏢"
    elif "standard" in tier_lower:
        sub_label = "Standard"
    elif tier_name and tier_name != "unknown":
        sub_label = tier_name
    else:
        sub_label = "Free tier"

    logger.info("Code Assist setup: tier=%s (id=%s)  project=%s  allowed=%s",
                tier_name, tier_id, project, allowed_names)
    _log(f"🔧 Code Assist: tier={tier_name}")
    _log(f"🔑 Subscription: {sub_label} | Credits: GOOGLE_ONE_AI enabled")
    if allowed_names:
        logger.info("Code Assist allowed tiers: %s", allowed_names)

    # If user needs onboarding (no currentTier), trigger it
    if not tier and resp_data.get("allowedTiers"):
        allowed = resp_data["allowedTiers"]
        default_tier = next((t for t in allowed if t.get("isDefault")), allowed[0] if allowed else {})
        tier_id = default_tier.get("id", "STANDARD")
        onboard_url = f"{_code_assist_base_url()}:onboardUser"
        onboard_payload = {
            "tierId": tier_id,
            "metadata": {
                "ideType": "IDE_UNSPECIFIED",
                "platform": "PLATFORM_UNSPECIFIED",
                "pluginType": "GEMINI",
            },
        }
        try:
            import httpx as _httpx
            ob_resp = _httpx.post(onboard_url, json=onboard_payload, headers=headers, timeout=60)
            ob_data = ob_resp.json()
        except ImportError:
            ob_resp = requests.post(onboard_url, json=onboard_payload, headers=headers, timeout=60)
            ob_data = ob_resp.json()

        # Poll LRO if needed
        if not ob_data.get("done") and ob_data.get("name"):
            op_name = ob_data["name"]
            for _ in range(12):  # up to 60s
                time.sleep(5)
                try:
                    import httpx as _httpx
                    poll = _httpx.get(
                        f"{_code_assist_base_url()}/{op_name}",
                        headers=headers, timeout=30,
                    )
                    ob_data = poll.json()
                except ImportError:
                    poll = requests.get(
                        f"{_code_assist_base_url()}/{op_name}",
                        headers=headers, timeout=30,
                    )
                    ob_data = poll.json()
                if ob_data.get("done"):
                    break

        project = (ob_data.get("response", {})
                   .get("cloudaicompanionProject", {})
                   .get("id", project))
        _log(f"🔧 Code Assist: onboarded (project={project})")

    _code_assist_project_id = project
    _code_assist_setup_done = True
    return project


def send_chat_completion_aistudio(
    access_token: str,
    messages: List[Dict],
    model: str = "gemini-2.5-flash",
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    timeout: int = 300,
    log_fn=None,
    connect_timeout: Optional[float] = None,
) -> Dict:
    """Send a chat completion via Google Code Assist proxy using OAuth.

    Routes through ``cloudcode-pa.googleapis.com`` (same as Gemini CLI).
    No GCP project required — works with any Google account.
    """
    if is_cancelled():
        raise RuntimeError("AuthGem: stream cancelled by user")

    _log = log_fn or (lambda *a, **kw: None)

    # Ensure user is set up with Code Assist (runs once per session)
    project_id = _code_assist_setup(access_token, _log)

    inner_body = _build_gemini_request_body(messages, temperature, max_tokens, model=model)

    # Warn user if Gemini 3 thought streaming is requested — Code Assist proxy
    # doesn't return thought=true annotations for Gemini 3 models.
    model_lower = model.lower() if model else ""
    tc = inner_body.get("generationConfig", {}).get("thinkingConfig")
    if "gemini-3" in model_lower:
        stream_thinking = os.getenv("STREAM_THINKING_LOGS", "1") not in ("0", "false")
        if stream_thinking and tc and tc.get("includeThoughts"):
            _log("⚠️ AuthGem: Gemini 3 thought streaming is not supported on authgem/ — use authgem-key/ instead")
            _log("🧠 Model is thinking internally (thoughts will not be streamed)")

    # Wrap in Code Assist envelope matching Gemini CLI's converter.ts format:
    # {model, project, user_prompt_id, enabled_credit_types, request: {contents, ..., session_id}}
    inner_body["session_id"] = _code_assist_session_id
    body: Dict = {
        "model": model,
        "project": project_id or "",
        "user_prompt_id": str(uuid.uuid4()),
        "enabled_credit_types": ["GOOGLE_ONE_AI"],
        "request": inner_body,
    }

    url = f"{_code_assist_base_url()}:streamGenerateContent?alt=sse"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "Accept-Encoding": "identity",
    }
    if project_id:
        headers["x-goog-user-project"] = project_id

    logger.info("AuthGem-CodeAssist: POST %s  model=%s  credits=GOOGLE_ONE_AI", url.split("?")[0], model)

    return _stream_gemini_common(url, body, headers, timeout, _log, connect_timeout)


# ===========================================================================
# AI Studio endpoint — API key (no OAuth).  Used by the  authgem-key/  prefix.
# ===========================================================================

def send_chat_completion_aistudio_key(
    api_key: str,
    messages: List[Dict],
    model: str = "gemini-2.5-flash",
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    timeout: int = 300,
    log_fn=None,
    connect_timeout: Optional[float] = None,
) -> Dict:
    """Send a chat completion via Google AI Studio using an API key.

    Uses ``generativelanguage.googleapis.com`` with ``?key=`` query parameter.
    No OAuth login needed — the user supplies a GEMINI_API_KEY.
    Free tier: 250 req/day, Flash model only.
    """
    if is_cancelled():
        raise RuntimeError("AuthGem: stream cancelled by user")

    _log = log_fn or (lambda *a, **kw: None)
    body = _build_gemini_request_body(messages, temperature, max_tokens, model=model)

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/"
        f"models/{model}:streamGenerateContent?key={api_key}&alt=sse"
    )
    headers = {
        "Content-Type": "application/json",
        "Accept-Encoding": "identity",
    }

    logger.info("AuthGem-Key: POST %s  model=%s", url.split("?")[0], model)

    return _stream_gemini_common(url, body, headers, timeout, _log, connect_timeout)


# ===========================================================================
# Vertex AI endpoint — OAuth Bearer token + GCP project.
# Used by the  authgem-vertex/  prefix (and legacy  send_chat_completion ).
# ===========================================================================

def send_chat_completion_vertex(
    access_token: str,
    messages: List[Dict],
    model: str = "gemini-2.5-flash",
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    timeout: int = 300,
    log_fn=None,
    connect_timeout: Optional[float] = None,
) -> Dict:
    """Send a chat completion via Vertex AI using an OAuth Bearer token.

    Uses ``{region}-aiplatform.googleapis.com`` — requires a GCP project
    with billing enabled and the Vertex AI API turned on.
    """
    if is_cancelled():
        raise RuntimeError("AuthGem: stream cancelled by user")

    _log = log_fn or (lambda *a, **kw: None)
    body = _build_gemini_request_body(messages, temperature, max_tokens, model=model)

    # Warn user if Gemini 3 thought streaming is requested — Vertex AI v1beta1
    # doesn't return thought=true annotations for Gemini 3 models.
    model_lower = model.lower() if model else ""
    if "gemini-3" in model_lower:
        stream_thinking = os.getenv("STREAM_THINKING_LOGS", "1") not in ("0", "false")
        tc = body.get("generationConfig", {}).get("thinkingConfig", {})
        if stream_thinking and tc.get("includeThoughts"):
            _log("⚠️ AuthGem: Gemini 3 thought streaming is not supported on authgem-vertex/ — use authgem-key/ instead")
            _log("🧠 Model is thinking internally (thoughts will not be streamed)")

    # Resolve project — auto-detect from the OAuth token
    project = detect_gcp_project(access_token)
    location = VERTEX_LOCATION
    if not project:
        raise RuntimeError(
            "AuthGem-Vertex: Could not detect your GCP project ID.\n"
            "Set the GOOGLE_CLOUD_PROJECT environment variable, e.g.:\n"
            "  set GOOGLE_CLOUD_PROJECT=my-project-id\n"
            "You can find your project ID at https://console.cloud.google.com"
        )

    # Preview models often require 'global' location instead of a regional one
    effective_location = location
    if "preview" in model.lower():
        effective_location = "global"

    # Global location uses a different URL format (no region prefix on hostname)
    if effective_location == "global":
        url = (
            f"https://aiplatform.googleapis.com/v1beta1/"
            f"projects/{project}/locations/global/"
            f"publishers/google/models/{model}:streamGenerateContent?alt=sse"
        )
    else:
        url = (
            f"https://{effective_location}-aiplatform.googleapis.com/v1beta1/"
            f"projects/{project}/locations/{effective_location}/"
            f"publishers/google/models/{model}:streamGenerateContent?alt=sse"
        )
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "Accept-Encoding": "identity",
    }

    logger.info("AuthGem-Vertex: POST %s  model=%s", url.split("?")[0], model)

    return _stream_gemini_common(url, body, headers, timeout, _log, connect_timeout)


# Backward-compatible alias — old code imports send_chat_completion
send_chat_completion = send_chat_completion_vertex


# ===========================================================================
# Shared streaming dispatcher used by all three endpoint functions
# ===========================================================================

def _stream_gemini_common(
    url: str,
    body: Dict,
    headers: Dict,
    timeout: int,
    _log,
    connect_timeout: Optional[float] = None,
) -> Dict:
    """Shared streaming logic for all Gemini endpoints (AI Studio & Vertex)."""
    # Respect the "Enable streaming responses" toggle from other_settings.py
    _enable_streaming = os.getenv("ENABLE_STREAMING", "1").lower() not in ("0", "false")
    log_stream = _enable_streaming and os.getenv("LOG_STREAM_CHUNKS", "1").lower() not in ("0", "false")
    if os.getenv("BATCH_TRANSLATION", "0") == "1":
        log_stream = os.getenv("ALLOW_BATCH_STREAM_LOGS", "0").lower() not in ("0", "false")

    # Thinking stream uses its own env var (matches other providers)
    stream_thinking = os.getenv("STREAM_THINKING_LOGS", "1") not in ("0", "false")

    # Log generation config summary
    _inner = body.get("request", body)  # Code Assist wraps in {"request": ...}
    _gc = _inner.get("generateContentConfig") or _inner.get("generationConfig") or {}
    _tc = _gc.get("thinkingConfig", {})
    _temp = _gc.get("temperature", "default")
    _think_desc = ""
    if "thinkingLevel" in _tc:
        _think_desc = f"level={_tc['thinkingLevel']}"
    elif "thinkingBudget" in _tc:
        b = _tc["thinkingBudget"]
        _think_desc = f"budget={'dynamic' if b < 0 else b}"
    else:
        _think_desc = "dynamic"
    if _tc.get("includeThoughts"):
        _think_desc += "+stream"
    # Check if safety settings are in the body
    _safety = _inner.get("safetySettings")
    if _safety and isinstance(_safety, list) and len(_safety) > 0:
        _safety_desc = _safety[0].get("threshold", "OFF")
    else:
        _safety_desc = "default"
    _log(f"⚙️ AuthGem: temperature={_temp}, thinking={_think_desc}, safety={_safety_desc}")

    # Emit "in progress" here (after config summary) so it appears last
    import threading as _threading
    _log(f"📤 [{_threading.current_thread().name}] API call in progress")

    t_start = time.time()

    # ── Non-streaming path: use generateContent instead of streamGenerateContent ──
    if not _enable_streaming:
        # Rewrite URL: streamGenerateContent?alt=sse → generateContent
        non_stream_url = url.replace(":streamGenerateContent?alt=sse", ":generateContent")
        non_stream_url = non_stream_url.replace(":streamGenerateContent", ":generateContent")
        # Remove alt=sse from query string if still present
        non_stream_url = non_stream_url.replace("&alt=sse", "").replace("?alt=sse", "")
        try:
            import httpx as _httpx
            _ct = connect_timeout or 30.0
            _client_timeout = _httpx.Timeout(timeout, connect=_ct)
            resp = _httpx.post(non_stream_url, json=body, headers=headers, timeout=_client_timeout)
        except ImportError:
            resp = requests.post(non_stream_url, json=body, headers=headers, timeout=timeout)

        elapsed = time.time() - t_start
        status = resp.status_code
        if status != 200:
            err_text = resp.text[:500] if hasattr(resp, 'text') else str(resp.content[:500])
            raise RuntimeError(f"AuthGem HTTP {status}. {err_text}")

        data = resp.json()
        _log(f"📡 AuthGem: Response received in {elapsed:.1f}s (non-streaming)")

        # Unwrap Code Assist envelope
        if "response" in data and isinstance(data["response"], dict):
            data = data["response"]

        # Extract text from candidates
        text_parts = []
        thought_parts = []
        finish_reason = "STOP"
        for candidate in data.get("candidates", []):
            fr = candidate.get("finishReason", "")
            if fr:
                finish_reason = fr
            for part in candidate.get("content", {}).get("parts", []):
                text = part.get("text", "")
                if not text:
                    continue
                is_thought = part.get("thought", False)
                if not is_thought and "thoughtSignature" in part and "thought" not in part:
                    is_thought = True
                if is_thought:
                    thought_parts.append(text)
                else:
                    text_parts.append(text)

        usage = data.get("usageMetadata", {})
        final_content = "".join(text_parts)
        thought_text = "".join(thought_parts) if thought_parts else None

        # Fallback: if text content is empty but thoughts contain text,
        # the API returned everything as thought-annotated parts.
        # Use the thought content as the main content — but ONLY when
        # finish_reason is STOP (normal completion).  If the model was
        # blocked (SAFETY, PROHIBITED_CONTENT, etc.) we must NOT leak
        # thinking output into the content field.
        if not final_content and thought_text and finish_reason == "STOP":
            final_content = thought_text
            thought_text = None

        return {
            "content": final_content,
            "finish_reason": finish_reason,
            "thought_content": thought_text,
            "usage_metadata": usage,
        }

    # ── Streaming path: use streamGenerateContent (SSE) ──
    # Prefer httpx for true real-time SSE (same stack as openai SDK)
    try:
        import httpx as _httpx
        return _stream_with_httpx_gemini(
            _httpx, url, body, headers, timeout, t_start,
            _log, log_stream, connect_timeout=connect_timeout,
            stream_thinking=stream_thinking,
        )
    except ImportError:
        _log("⚠️ AuthGem: httpx not installed, falling back to requests (streaming may be buffered)")
        return _stream_with_requests_gemini(
            url, body, headers, timeout, t_start,
            _log, log_stream, stream_thinking=stream_thinking,
        )


# ---------------------------------------------------------------------------
# Gemini SSE stream processing helpers
# ---------------------------------------------------------------------------

def _new_gemini_stream_state() -> Dict:
    """Create fresh state dict for a Gemini streaming session."""
    return {
        "text_parts": [],        # Non-thought text chunks
        "thought_parts": [],     # Thought/reasoning chunks (filtered from output)
        "got_first_data": False,
        "streamed_chars": 0,
        "log_buf": [],
        "finish_reason": "STOP",
        "usage_metadata": {},
    }


def _process_gemini_sse_line(
    line: str,
    state: Dict,
    _log,
    log_stream: bool,
    t_start: float,
    stream_thinking: bool = True,
) -> bool:
    """Process a single SSE line from Vertex AI streamGenerateContent.

    Updates *state* in-place.  Returns True when the stream should stop.
    """
    line = line.strip()
    if not line:
        return False

    # SSE data lines start with "data: "
    if not line.startswith("data: "):
        return False

    payload = line[6:]
    if payload == "[DONE]":
        return True

    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return False

    if not state["got_first_data"]:
        state["got_first_data"] = True
        ttft = time.time() - t_start
        state["_ttft"] = ttft
        if log_stream:
            _log(f"📡 AuthGem: First token in {ttft:.1f}s, streaming…")

    # Unwrap Code Assist response envelope if present
    # Code Assist wraps: {"response": {"candidates": [...]}} 
    # Direct AI Studio / Vertex use flat {"candidates": [...]}
    if "response" in data and isinstance(data["response"], dict):
        data = data["response"]

    # Extract candidates
    candidates = data.get("candidates", [])
    for candidate in candidates:
        # Update finish reason from the last candidate
        fr = candidate.get("finishReason", "")
        if fr:
            state["finish_reason"] = fr

        content_parts = candidate.get("content", {}).get("parts", [])
        for part in content_parts:
            text = part.get("text", "")
            if not text:
                continue

            # Separate thinking/thought parts from actual output
            # Primary: check explicit thought boolean
            # Fallback: if part has thoughtSignature but no thought key,
            # treat it as a thought part (Vertex AI Gemini 3 omits thought=true).
            is_thought = part.get("thought", False)
            if not is_thought and "thoughtSignature" in part and "thought" not in part:
                is_thought = True
            if is_thought:
                state["thought_parts"].append(text)
                # Log thinking in real-time — same format as gemini-grpc
                if stream_thinking and text:
                    if not state.get("_thinking_started"):
                        _log("🧠 [authgem] Thinking...")
                        state["_thinking_started"] = True
                        state["_thinking_chunks"] = 0
                        state["_thinking_start_ts"] = time.time()
                    state["_thinking_chunks"] = state.get("_thinking_chunks", 0) + 1
                    thought_buf = state.get("_thought_log_buf", [])
                    thought_buf.append(text)
                    combined = "".join(thought_buf)
                    if "\n" in combined:
                        parts_t = combined.split("\n")
                        for p in parts_t[:-1]:
                            _log(f"    {p}")
                        state["_thought_log_buf"] = [parts_t[-1]]
                    else:
                        state["_thought_log_buf"] = thought_buf
                continue

            state["text_parts"].append(text)
            state["streamed_chars"] += len(text)

            # Emit separator when transitioning from thinking to text output
            if state.get("_thinking_started") and not state.get("_thinking_ended"):
                state["_thinking_ended"] = True
                if stream_thinking:
                    # Flush remaining thinking buffer first
                    thought_rem = state.get("_thought_log_buf", [])
                    if thought_rem:
                        remainder = "".join(thought_rem).rstrip("\n")
                        if remainder:
                            for p in remainder.split("\n"):
                                _log(f"    {p}")
                        state["_thought_log_buf"] = []
                    chunks = state.get("_thinking_chunks", len(state["thought_parts"]))
                    dur = time.time() - state.get("_thinking_start_ts", t_start)
                    _log(f"🧠 [authgem] Thinking complete ({chunks} chunks, {dur:.1f}s)")
                    _log("─" * 50)
                    _log("📡 Text streaming...")

            # Real-time log output (mirrors authgpt's delta logging)
            if log_stream and text:
                log_buf = state["log_buf"]
                combined = "".join(log_buf) + text
                # Insert newlines after HTML closing tags for readability
                for tag in ('</h1>', '</h2>', '</h3>', '</h4>', '</h5>', '</h6>', '</p>'):
                    combined = combined.replace(tag, tag + '\n')
                if "\n" in combined:
                    parts = combined.split("\n")
                    for p in parts[:-1]:
                        _log(p)
                    state["log_buf"] = [parts[-1]]
                else:
                    log_buf.append(text)
                    if len("".join(log_buf)) > 150:
                        _log("".join(log_buf))
                        state["log_buf"] = []

    # Update usage metadata if present
    usage = data.get("usageMetadata", {})
    if usage:
        state["usage_metadata"] = usage

    return False


def _finalize_gemini_stream(state: Dict, _log, log_stream: bool, t_start: float, stream_thinking: bool = True) -> Dict:
    """Flush log buffer, build result dict from accumulated state."""
    # Flush remaining thinking log buffer (only if not already flushed during transition)
    if not state.get("_thinking_ended") and state.get("_thought_log_buf"):
        remainder = "".join(state["_thought_log_buf"]).rstrip("\n")
        if remainder and stream_thinking:
            for p in remainder.split("\n"):
                _log(f"    {p}")
    # Log thinking completion summary (only if not already logged during transition)
    if state.get("_thinking_started") and state["thought_parts"] and not state.get("_thinking_ended"):
        chunks = state.get("_thinking_chunks", len(state["thought_parts"]))
        dur = time.time() - state.get("_thinking_start_ts", t_start)
        if stream_thinking:
            _log(f"🧠 [authgem] Thinking complete ({chunks} chunks, {dur:.1f}s)")

    # Flush remaining output log buffer
    if log_stream and state["log_buf"]:
        remainder = "".join(state["log_buf"]).strip()
        if remainder:
            _log(remainder)

    t_total = time.time() - t_start

    # Infer thinking when the endpoint doesn't stream thought parts
    # (Code Assist strips thought content from the SSE stream)
    if not state.get("_thinking_started") and not state["thought_parts"]:
        ttft = state.get("_ttft", 0)
        um = state["usage_metadata"]
        thinking_tokens = um.get("thoughtsTokenCount", 0) if um else 0
        if thinking_tokens > 0:
            _log(f"🧠 [authgem] Model used {thinking_tokens} thinking tokens (TTFT {ttft:.1f}s)")
        elif ttft > 5.0:
            _log(f"🧠 [authgem] Model thinking inferred from TTFT ({ttft:.1f}s) — endpoint doesn't stream thoughts")

    _log(f"📡 AuthGem: Stream finished in {t_total:.1f}s ({state['streamed_chars']} chars)")

    content = "".join(state["text_parts"])
    thought_content = "".join(state["thought_parts"]) if state["thought_parts"] else None

    # Fallback: if text is empty but thoughts exist and finish_reason is STOP,
    # the API returned everything as thought-annotated parts.
    # Use thought content as main content (safe — blocked responses have non-STOP finish).
    raw_fr = state["finish_reason"]
    if not content and thought_content and raw_fr == "STOP":
        _log("⚠️ AuthGem: Text empty but thoughts present — using thought content as output")
        content = thought_content
        thought_content = None
    elif not content and thought_content:
        _log("⚠️ AuthGem: Response contained only thinking/reasoning — no output text.")
    elif not content and not thought_content:
        _log("⚠️ AuthGem: Empty response — no text received.")

    # Map Gemini finish reasons to OpenAI-style
    finish_reason_map = {
        "STOP": "stop",
        "MAX_TOKENS": "length",
        "SAFETY": "content_filter",
        "RECITATION": "content_filter",
        "FINISH_REASON_UNSPECIFIED": "stop",
    }
    raw_fr = state["finish_reason"]
    mapped_reason = finish_reason_map.get(raw_fr, raw_fr.lower() if raw_fr else "stop")

    # Extract usage
    usage = None
    um = state["usage_metadata"]
    if um:
        usage = {
            "prompt_tokens": um.get("promptTokenCount", 0),
            "completion_tokens": um.get("candidatesTokenCount", 0),
            "total_tokens": um.get("totalTokenCount", 0),
        }
        thinking_tokens = um.get("thoughtsTokenCount", 0)
        if thinking_tokens:
            usage["thinking_tokens"] = thinking_tokens

    return {
        "content": content,
        "finish_reason": mapped_reason,
        "usage": usage,
    }


# ---------------------------------------------------------------------------
# httpx-based SSE reader (preferred — real-time, no buffering)
# ---------------------------------------------------------------------------

def _stream_with_httpx_gemini(
    _httpx,
    url: str,
    body: Dict,
    headers: Dict,
    timeout: int,
    t_start: float,
    _log,
    log_stream: bool,
    connect_timeout: Optional[float] = None,
    stream_thinking: bool = True,
) -> Dict:
    """Stream SSE from Vertex AI using httpx (same stack as the openai SDK)."""
    state = _new_gemini_stream_state()
    _timeout = _httpx.Timeout(timeout, connect=connect_timeout)

    with _httpx.stream(
        "POST", url,
        json=body,
        headers=headers,
        timeout=_timeout,
    ) as resp:
        if resp.status_code >= 400:
            error_body = resp.read().decode("utf-8", errors="replace")
            reason = getattr(resp, "reason_phrase", "") or ""
            detail = error_body
            try:
                detail = json.loads(error_body).get("error", {}).get("message", error_body)
            except Exception:
                pass
            if not detail:
                detail = "empty-body"
            summary = detail or reason or "Bad Request"
            _log(f"❌ AuthGem HTTP {resp.status_code}. {summary}")
            raise RuntimeError(
                f"AuthGem: {resp.status_code} – {summary}"
            )

        for line in resp.iter_lines():
            if _cancel_event.is_set():
                resp.close()
                raise RuntimeError("AuthGem: stream cancelled by user")
            if _process_gemini_sse_line(line, state, _log, log_stream, t_start, stream_thinking=stream_thinking):
                break

    return _finalize_gemini_stream(state, _log, log_stream, t_start, stream_thinking=stream_thinking)


# ---------------------------------------------------------------------------
# requests-based SSE reader (fallback — may buffer due to urllib3/http.client)
# ---------------------------------------------------------------------------

def _stream_with_requests_gemini(
    url: str,
    body: Dict,
    headers: Dict,
    timeout: int,
    t_start: float,
    _log,
    log_stream: bool,
    stream_thinking: bool = True,
) -> Dict:
    """Stream SSE from Vertex AI using requests (fallback when httpx unavailable)."""
    state = _new_gemini_stream_state()
    resp = requests.post(url, json=body, headers=headers, timeout=timeout, stream=True)

    if resp.status_code >= 400:
        try:
            error_body = resp.text
        except Exception:
            error_body = ""
        try:
            reason = resp.reason or ""
        except Exception:
            reason = ""
        detail = error_body
        try:
            detail = resp.json().get("error", {}).get("message", error_body)
        except Exception:
            pass
        if not detail:
            detail = "empty-body"
        summary = detail or reason or "Bad Request"
        _log(f"❌ AuthGem HTTP {resp.status_code}. {summary}")
        raise RuntimeError(
            f"AuthGem: {resp.status_code} – {summary}"
        )

    for raw_line in resp.iter_lines(chunk_size=1):
        if _cancel_event.is_set():
            resp.close()
            raise RuntimeError("AuthGem: stream cancelled by user")
        if raw_line is None:
            continue
        line = raw_line.decode("utf-8", errors="replace") if isinstance(raw_line, bytes) else raw_line
        if _process_gemini_sse_line(line, state, _log, log_stream, t_start, stream_thinking=stream_thinking):
            break

    return _finalize_gemini_stream(state, _log, log_stream, t_start, stream_thinking=stream_thinking)

