# authcd_auth.py - Claude subscription OAuth authentication
# Uses the same OAuth PKCE flow as Claude Code CLI.
# Prefix models with 'authcd/' to route through the Anthropic Messages API
# using your Claude Pro/Max subscription instead of API key credits.
"""
OAuth 2.0 PKCE flow for Claude subscription authentication, persistent
token storage with automatic refresh, and Anthropic Messages API adapter.

Flow:
  1. Generate PKCE code_verifier + code_challenge
  2. Open browser to claude.ai/oauth/authorize
  3. Spin up a local HTTP callback server (port 54545)
  4. User logs in via browser -> callback receives auth code
  5. Exchange auth code for access + refresh tokens
  6. Store tokens locally (~/.glossarion/authcd_tokens.json)
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

# Module-level cancellation flag
_cancel_event = threading.Event()

def cancel_stream():
    """Signal any active AuthCD stream to abort immediately."""
    _cancel_event.set()

def reset_cancel():
    """Clear the cancellation flag (call before starting a new request)."""
    _cancel_event.clear()

def is_cancelled() -> bool:
    return _cancel_event.is_set()

# ===========================================================================
# Constants - mirror Claude Code CLI OAuth values
# ===========================================================================
CLAUDE_CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
CLAUDE_AUTH_URL = "https://claude.ai/oauth/authorize"
CLAUDE_TOKEN_URL = "https://console.anthropic.com/v1/oauth/token"
CALLBACK_HOST = "localhost"
CALLBACK_PORT = 54545
CALLBACK_PATH = "/callback"
SCOPES = "user:profile user:inference org:create_api_key"
TOKEN_REFRESH_MARGIN_SECONDS = 300  # refresh when <5 min remaining

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_API_VERSION = "2023-06-01"

_DEFAULT_TOKEN_DIR = os.path.join(os.path.expanduser("~"), ".glossarion")
_DEFAULT_TOKEN_FILE = os.path.join(_DEFAULT_TOKEN_DIR, "authcd_tokens.json")

# Claude Code credential paths (for parasitic fallback)
_CLAUDE_CODE_CREDS = os.path.join(os.path.expanduser("~"), ".claude", ".credentials.json")


# ===========================================================================
# PKCE helpers
# ===========================================================================

def generate_pkce() -> Tuple[str, str]:
    """Generate PKCE code_verifier and code_challenge (S256)."""
    raw = secrets.token_bytes(32)
    code_verifier = base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")
    digest = hashlib.sha256(code_verifier.encode("ascii")).digest()
    code_challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return code_verifier, code_challenge


def build_auth_url(code_challenge: str, state: str, redirect_uri: str) -> str:
    """Build the full authorization URL."""
    params = {
        "client_id": CLAUDE_CLIENT_ID,
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "scope": SCOPES,
        "state": state,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
    }
    return f"{CLAUDE_AUTH_URL}?{urlencode(params)}"


# ===========================================================================
# Token exchange / refresh
# ===========================================================================

def exchange_code_for_tokens(auth_code: str, code_verifier: str, redirect_uri: str) -> Dict:
    """Exchange authorization code for access + refresh tokens."""
    payload = {
        "grant_type": "authorization_code",
        "client_id": CLAUDE_CLIENT_ID,
        "code": auth_code,
        "redirect_uri": redirect_uri,
        "code_verifier": code_verifier,
    }
    resp = requests.post(CLAUDE_TOKEN_URL, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    data["expires_at"] = time.time() + data.get("expires_in", 3600)
    return data


def refresh_access_token(refresh_token: str) -> Dict:
    """Use a refresh token to obtain a new access token."""
    payload = {
        "grant_type": "refresh_token",
        "client_id": CLAUDE_CLIENT_ID,
        "refresh_token": refresh_token,
    }
    resp = requests.post(CLAUDE_TOKEN_URL, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    data["expires_at"] = time.time() + data.get("expires_in", 3600)
    return data


# ===========================================================================
# Local callback server
# ===========================================================================

class _OAuthCallbackHandler(BaseHTTPRequestHandler):
    """HTTP request handler that captures the OAuth callback."""

    def log_message(self, format, *args):
        pass  # Suppress default stderr logging

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == CALLBACK_PATH or parsed.path == "/oauth/callback":
            qs = parse_qs(parsed.query)
            self.server._auth_code = qs.get("code", [None])[0]
            self.server._returned_state = qs.get("state", [None])[0]
            self.server._error = qs.get("error", [None])[0]
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
                "<h1 style='color:#d97706'>&#10004; Claude Authenticated!</h1>"
                "<p>You can close this tab and return to Glossarion.</p>"
                "</body></html>"
            )
            self.wfile.write(html.encode("utf-8"))
            threading.Thread(target=self.server.shutdown, daemon=True).start()
        else:
            self.send_response(404)
            self.end_headers()


def _find_available_port() -> int:
    """Return the callback port for OAuth."""
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((CALLBACK_HOST, CALLBACK_PORT))
            return CALLBACK_PORT
    except OSError:
        # If default port is busy, use a random one
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((CALLBACK_HOST, 0))
            return s.getsockname()[1]


# ===========================================================================
# OAuth flow orchestrator
# ===========================================================================

def run_oauth_flow(timeout: int = 300) -> Dict:
    """Run the full OAuth PKCE login flow."""
    port = _find_available_port()
    redirect_uri = f"http://{CALLBACK_HOST}:{port}{CALLBACK_PATH}"
    code_verifier, code_challenge = generate_pkce()
    state = secrets.token_urlsafe(32)

    auth_url = build_auth_url(code_challenge, state, redirect_uri)

    server = HTTPServer((CALLBACK_HOST, port), _OAuthCallbackHandler)
    server._auth_code = None
    server._returned_state = None
    server._error = None
    server.timeout = timeout

    print(f"🔐 Opening browser for Claude login…")
    print(f"   If the browser doesn't open, visit:\n   {auth_url}")
    webbrowser.open(auth_url)

    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    server_thread.join(timeout=timeout)
    server.shutdown()

    if server._error:
        raise RuntimeError(f"OAuth error: {server._error}")
    if not server._auth_code:
        raise RuntimeError("OAuth login timed out – no callback received.")
    if server._returned_state != state:
        raise RuntimeError("OAuth state mismatch – possible CSRF attack.")

    print("🔑 Exchanging authorization code for tokens…")
    tokens = exchange_code_for_tokens(server._auth_code, code_verifier, redirect_uri)
    print("✅ Claude OAuth authentication successful!")
    return tokens


# ===========================================================================
# Claude Code credentials parasitic loader
# ===========================================================================

def _load_claude_code_credentials() -> Optional[Dict]:
    """Try to load existing credentials from Claude Code's local store."""
    try:
        if not os.path.isfile(_CLAUDE_CODE_CREDS):
            return None
        with open(_CLAUDE_CODE_CREDS, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Claude Code stores as { claudeAiOauth: { accessToken, refreshToken, expiresAt } }
        oauth = data.get("claudeAiOauth") or data
        access = oauth.get("accessToken") or oauth.get("access_token")
        if not access:
            return None
        refresh = oauth.get("refreshToken") or oauth.get("refresh_token")
        expires_at_str = oauth.get("expiresAt") or oauth.get("expires_at")
        expires_at = 0
        if isinstance(expires_at_str, (int, float)):
            expires_at = float(expires_at_str)
        elif isinstance(expires_at_str, str):
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(expires_at_str.replace("Z", "+00:00"))
                expires_at = dt.timestamp()
            except Exception:
                expires_at = time.time() + 3600
        return {
            "access_token": access,
            "refresh_token": refresh,
            "expires_at": expires_at,
            "_source": "claude_code",
        }
    except Exception as exc:
        logger.debug("Failed to load Claude Code credentials: %s", exc)
        return None


# ===========================================================================
# Token store (persistent, thread-safe)
# ===========================================================================

class AuthCDTokenStore:
    """Thread-safe token store backed by a JSON file."""

    def __init__(self, token_file: Optional[str] = None, account_id: int = 0):
        self._token_file = (
            token_file
            or os.environ.get("AUTHCD_TOKEN_FILE")
            or _DEFAULT_TOKEN_FILE
        )
        self._account_id = account_id
        self._lock = threading.RLock()
        self._tokens: Optional[Dict] = None
        self._on_change_callbacks: List = []
        self._load_from_disk()

    def on_token_change(self, callback):
        self._on_change_callbacks.append(callback)

    def _fire_change_callbacks(self):
        for cb in self._on_change_callbacks:
            try:
                cb()
            except Exception:
                pass

    def _ensure_dir(self):
        d = os.path.dirname(self._token_file)
        if d:
            os.makedirs(d, exist_ok=True)

    def _load_from_disk(self):
        """Load tokens from encrypted file, plain JSON, or Claude Code credentials."""
        try:
            if os.path.isfile(self._token_file):
                try:
                    from token_encryption import load_encrypted_tokens
                    self._tokens = load_encrypted_tokens(self._token_file)
                except ImportError:
                    with open(self._token_file, "r", encoding="utf-8") as f:
                        self._tokens = json.load(f)
                except Exception as dec_exc:
                    logger.warning("AuthCD token decryption failed (%s) — removing corrupt file", dec_exc)
                    try:
                        os.remove(self._token_file)
                    except OSError:
                        pass
                    self._tokens = None
                    return
                logger.debug("AuthCD tokens loaded from %s", self._token_file)
                return
        except Exception as exc:
            logger.warning("Failed to load authcd tokens: %s", exc)
            self._tokens = None

        # Fallback: try Claude Code's own credentials
        if self._account_id == 0 and self._tokens is None:
            cc_tokens = _load_claude_code_credentials()
            if cc_tokens:
                logger.info("AuthCD: Using existing Claude Code credentials")
                self._tokens = cc_tokens

    def save_tokens(self, tokens: Dict):
        """Encrypt and save tokens to disk."""
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
                    logger.warning("AuthCD token encryption failed (%s) — saving as plain JSON", enc_exc)
                if not saved:
                    with open(self._token_file, "w", encoding="utf-8") as f:
                        json.dump(tokens, f, indent=2)
                logger.debug("AuthCD tokens saved to %s", self._token_file)
            except Exception as exc:
                logger.warning("Failed to save authcd tokens: %s", exc)
        self._fire_change_callbacks()

    def load_tokens(self) -> Optional[Dict]:
        with self._lock:
            if self._tokens is None:
                self._load_from_disk()
            return self._tokens

    def clear_tokens(self):
        with self._lock:
            self._tokens = None
            try:
                if os.path.isfile(self._token_file):
                    os.remove(self._token_file)
                    logger.info("AuthCD tokens removed")
            except Exception as exc:
                logger.warning("Failed to remove token file: %s", exc)
        self._fire_change_callbacks()

    def _is_token_expired(self, tokens: Dict) -> bool:
        expires_at = tokens.get("expires_at", 0)
        return time.time() >= (expires_at - TOKEN_REFRESH_MARGIN_SECONDS)

    def _try_refresh(self, tokens: Dict) -> Optional[Dict]:
        rt = tokens.get("refresh_token")
        if not rt:
            return None
        try:
            new_tokens = refresh_access_token(rt)
            merged = {**tokens, **new_tokens}
            merged.pop("_source", None)  # No longer from Claude Code
            self.save_tokens(merged)
            logger.info("AuthCD access token refreshed successfully")
            return merged
        except Exception as exc:
            logger.warning("AuthCD token refresh failed: %s", exc)
            return None

    def get_valid_access_token(self, auto_login: bool = True) -> str:
        """Return a valid access token, refreshing or re-authenticating as needed."""
        with self._lock:
            tokens = self.load_tokens()

            if tokens and tokens.get("access_token") and not self._is_token_expired(tokens):
                return tokens["access_token"]

            if tokens and tokens.get("refresh_token"):
                refreshed = self._try_refresh(tokens)
                if refreshed and refreshed.get("access_token"):
                    return refreshed["access_token"]

            if not auto_login:
                raise RuntimeError(
                    "AuthCD: No valid tokens and auto_login is disabled. "
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
                env_access = os.environ.get("AUTHCD_ACCESS_TOKEN", "").strip()
                env_refresh = os.environ.get("AUTHCD_REFRESH_TOKEN", "").strip()
                # Also check CLAUDE_CODE_OAUTH_TOKEN (official Claude Code env var)
                if not env_access:
                    env_access = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN", "").strip()
                if env_access:
                    manual_tokens = {
                        "access_token": env_access,
                        "expires_at": time.time() + 3600,
                    }
                    if env_refresh:
                        manual_tokens["refresh_token"] = env_refresh
                    self.save_tokens(manual_tokens)
                    return env_access
                if env_refresh:
                    try:
                        refreshed = refresh_access_token(env_refresh)
                        self.save_tokens(refreshed)
                        return refreshed["access_token"]
                    except Exception as ref_exc:
                        raise RuntimeError(
                            f"AuthCD: AUTHCD_REFRESH_TOKEN was set but refresh failed: {ref_exc}"
                        )
                raise RuntimeError(
                    "AuthCD: Browser-based OAuth login is not available in headless environments.\n"
                    "Set one of these environment secrets:\n"
                    "  • AUTHCD_ACCESS_TOKEN or CLAUDE_CODE_OAUTH_TOKEN\n"
                    "  • AUTHCD_REFRESH_TOKEN (will auto-refresh)\n"
                    "You can obtain these by running the OAuth flow locally first."
                )

            print("🔄 AuthCD: No valid token found – starting browser login…")
            new_tokens = run_oauth_flow()
            self.save_tokens(new_tokens)
            return new_tokens["access_token"]

    @property
    def has_tokens(self) -> bool:
        tokens = self.load_tokens()
        return bool(tokens and tokens.get("access_token"))

    @property
    def account_info(self) -> Dict:
        tokens = self.load_tokens()
        if not tokens:
            return {}
        source = tokens.get("_source", "glossarion")
        return {"source": source}


# Module-level singleton
_default_store: Optional[AuthCDTokenStore] = None
_default_store_lock = threading.Lock()

_account_stores: Dict[int, AuthCDTokenStore] = {}
_account_stores_lock = threading.Lock()


def get_default_store() -> AuthCDTokenStore:
    global _default_store
    if _default_store is None:
        with _default_store_lock:
            if _default_store is None:
                _default_store = AuthCDTokenStore()
    return _default_store


def get_store(account_id: Optional[int] = None) -> AuthCDTokenStore:
    if account_id is None or account_id == 0:
        return get_default_store()
    with _account_stores_lock:
        if account_id in _account_stores:
            return _account_stores[account_id]
        token_file = os.path.join(_DEFAULT_TOKEN_DIR, f"authcd_tokens_{account_id}.json")
        store = AuthCDTokenStore(token_file=token_file, account_id=account_id)
        _account_stores[account_id] = store
        return store


# ===========================================================================
# Anthropic Messages API adapter
# ===========================================================================

def _convert_messages_to_anthropic(messages: List[Dict]) -> Tuple[str, List[Dict]]:
    """Convert OpenAI-style messages to Anthropic format.
    Returns (system_prompt, anthropic_messages).
    """
    system_prompt = ""
    anthropic_messages = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "system":
            if system_prompt:
                system_prompt += "\n\n" + (content if isinstance(content, str) else str(content))
            else:
                system_prompt = content if isinstance(content, str) else str(content)
        elif role == "assistant":
            anthropic_messages.append({"role": "assistant", "content": content})
        else:
            anthropic_messages.append({"role": "user", "content": content})

    # Merge consecutive same-role messages
    merged = []
    for msg in anthropic_messages:
        if merged and merged[-1]["role"] == msg["role"]:
            prev = merged[-1]["content"]
            cur = msg["content"]
            if isinstance(prev, str) and isinstance(cur, str):
                merged[-1]["content"] = prev + "\n\n" + cur
            else:
                merged[-1]["content"] = str(prev) + "\n\n" + str(cur)
        else:
            merged.append(msg)

    if not merged or merged[0]["role"] != "user":
        merged.insert(0, {"role": "user", "content": "Please continue."})

    return system_prompt, merged


# ===========================================================================
# SSE stream helpers
# ===========================================================================

def _process_sse_line(line: str, state: Dict, _log, log_stream: bool, t_start: float) -> bool:
    """Process a single SSE line. Returns True when stream should stop."""
    state["raw_lines"].append(line)

    if not state["got_first_data"] and line.startswith("data: "):
        state["got_first_data"] = True
        ttft = time.time() - t_start
        _log(f"📡 AuthCD: First token in {ttft:.1f}s, streaming…")

    if line.startswith("data: ") and '"content_block_delta"' in line:
        try:
            data = json.loads(line[6:])
            delta = data.get("delta", {})
            text = delta.get("text", "")
            state["streamed_text"].append(text)
            if log_stream and text:
                log_buf = state["log_buf"]
                combined = "".join(log_buf) + text
                for tag in ('</h1>', '</h2>', '</h3>', '</h4>', '</h5>', '</h6>', '</p>'):
                    combined = combined.replace(tag, tag + '\n')
                if "\n" in combined:
                    parts = combined.split("\n")
                    for part in parts[:-1]:
                        _log(part)
                    state["log_buf"] = [parts[-1]]
                else:
                    log_buf.append(text)
                    if len("".join(log_buf)) > 150:
                        import builtins
                        builtins.print("".join(log_buf), end="", flush=True)
                        state["log_buf"] = []
        except (json.JSONDecodeError, KeyError):
            pass

    if line.startswith("data: ") and '"message_stop"' in line:
        return True
    if line.strip() == "data: [DONE]":
        return True
    return False


def _finalize_stream(state: Dict, _log, log_stream: bool, t_start: float) -> Dict:
    """Parse SSE results into a result dict."""
    if log_stream and state["log_buf"]:
        remainder = "".join(state["log_buf"]).strip()
        if remainder:
            _log(remainder)

    t_total = time.time() - t_start
    _log(f"📡 AuthCD: Stream finished in {t_total:.1f}s")

    content = "".join(state["streamed_text"])

    # Parse usage from message_start or message_delta events
    usage = None
    finish_reason = "stop"
    for rl in state["raw_lines"]:
        if not rl.startswith("data: "):
            continue
        payload = rl[6:]
        if payload == "[DONE]":
            continue
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            continue
        evt_type = data.get("type", "")
        if evt_type == "message_start":
            msg = data.get("message", {})
            u = msg.get("usage", {})
            if u:
                usage = {
                    "prompt_tokens": u.get("input_tokens", 0),
                    "completion_tokens": u.get("output_tokens", 0),
                    "total_tokens": u.get("input_tokens", 0) + u.get("output_tokens", 0),
                }
        elif evt_type == "message_delta":
            delta = data.get("delta", {})
            sr = delta.get("stop_reason")
            if sr:
                finish_reason = "stop" if sr == "end_turn" else ("length" if sr == "max_tokens" else sr)
            u = data.get("usage", {})
            if u and usage:
                usage["completion_tokens"] = u.get("output_tokens", usage.get("completion_tokens", 0))
                usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]

    return {
        "content": content,
        "finish_reason": finish_reason,
        "usage": usage,
    }


def _new_stream_state() -> Dict:
    return {
        "raw_lines": [],
        "got_first_data": False,
        "streamed_text": [],
        "log_buf": [],
    }


# ===========================================================================
# Public API – send chat completion
# ===========================================================================

def send_chat_completion(
    access_token: str,
    messages: List[Dict],
    model: str = "claude-sonnet-4-6",
    temperature: Optional[float] = 0.7,
    max_tokens: Optional[int] = None,
    timeout: int = 600,
    base_url: Optional[str] = None,
    log_fn: Optional[Any] = None,
    connect_timeout: Optional[float] = None,
) -> Dict:
    """Send a chat completion request via the Anthropic Messages API.

    Uses Authorization: Bearer with the OAuth token (not x-api-key).
    """
    effective_base = base_url or os.getenv("AUTHCD_BASE_URL", ANTHROPIC_API_URL)
    url = effective_base.rstrip("/")
    if not url.endswith("/messages"):
        url = url.rstrip("/") + "/v1/messages"

    system_prompt, anthropic_messages = _convert_messages_to_anthropic(messages)

    body: Dict[str, Any] = {
        "model": model,
        "messages": anthropic_messages,
        "max_tokens": max_tokens or 8192,
        "stream": True,
    }
    if system_prompt:
        body["system"] = system_prompt
    if temperature is not None:
        body["temperature"] = temperature

    headers = {
        "Authorization": f"Bearer {access_token}",
        "anthropic-version": ANTHROPIC_API_VERSION,
        "Content-Type": "application/json",
        "Accept-Encoding": "identity",
    }

    _log = log_fn or print
    logger.info("AuthCD: POST %s  model=%s", url, model)

    log_stream = os.getenv("LOG_STREAM_CHUNKS", "1").lower() not in ("0", "false")
    if os.getenv("BATCH_TRANSLATION", "0") == "1":
        log_stream = os.getenv("ALLOW_AUTHCD_BATCH_STREAM_LOGS", "0").lower() not in ("0", "false")

    t_start = time.time()
    state = _new_stream_state()

    # Prefer httpx for real-time SSE streaming
    try:
        import httpx as _httpx
        _timeout = _httpx.Timeout(timeout, connect=connect_timeout)
        with _httpx.stream("POST", url, json=body, headers=headers, timeout=_timeout) as resp:
            if resp.status_code >= 400:
                error_body = resp.read().decode("utf-8", errors="replace")
                reason = getattr(resp, "reason_phrase", "") or ""
                detail = error_body
                try:
                    detail = json.loads(error_body).get("error", {}).get("message", error_body)
                except Exception:
                    pass
                _log(f"❌ AuthCD HTTP {resp.status_code}. {detail}")
                raise RuntimeError(f"AuthCD: {resp.status_code} – {detail} [reason={reason}]")
            for line in resp.iter_lines():
                if _cancel_event.is_set():
                    resp.close()
                    raise RuntimeError("AuthCD: stream cancelled by user")
                if _process_sse_line(line, state, _log, log_stream, t_start):
                    break
        return _finalize_stream(state, _log, log_stream, t_start)
    except ImportError:
        pass

    # Fallback: requests
    _log("⚠️ AuthCD: httpx not installed, falling back to requests")
    resp = requests.post(url, json=body, headers=headers, timeout=timeout, stream=True)
    if resp.status_code >= 400:
        error_body = resp.text
        detail = error_body
        try:
            detail = resp.json().get("error", {}).get("message", error_body)
        except Exception:
            pass
        _log(f"❌ AuthCD HTTP {resp.status_code}. {detail}")
        raise RuntimeError(f"AuthCD: {resp.status_code} – {detail}")

    for raw_line in resp.iter_lines(chunk_size=1):
        if _cancel_event.is_set():
            resp.close()
            raise RuntimeError("AuthCD: stream cancelled by user")
        if raw_line is None:
            continue
        line = raw_line.decode("utf-8", errors="replace") if isinstance(raw_line, bytes) else raw_line
        if _process_sse_line(line, state, _log, log_stream, t_start):
            break

    return _finalize_stream(state, _log, log_stream, t_start)
