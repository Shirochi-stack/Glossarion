# authza_auth.py - Z.AI subscription authentication via Google OAuth
# Z.AI's chat interface (https://chat.z.ai) is powered by Open WebUI,
# which uses Google OAuth → JWT stored in localStorage.
# Prefix models with 'authza/' to route through the Z.AI chat backend
# using your Z.AI subscription instead of API key credits.
"""
Google OAuth flow for Z.AI subscription authentication:

  1. Open browser to https://chat.z.ai → user clicks "Continue with Google"
  2. Google OAuth completes → Z.AI issues a JWT
  3. Capture the JWT from the browser (via local callback interception)
  4. Store JWT locally (~/.glossarion/authza_tokens.json)
  5. Hit https://chat.z.ai/api/v1/chat/completions with Bearer <JWT>

The chat backend is OpenAI-compatible (Open WebUI), so standard
messages/model/temperature/max_tokens parameters work directly.
"""
import os
import json
import time
import logging
import threading
import webbrowser
import socket
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlencode, urlparse, parse_qs
from typing import Optional, Dict, List, Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level cancellation flag
# ---------------------------------------------------------------------------
_cancel_event = threading.Event()


def cancel_stream():
    """Signal any active AuthZA stream to abort immediately."""
    _cancel_event.set()


def reset_cancel():
    """Clear the cancellation flag (call before starting a new request)."""
    _cancel_event.clear()


def is_cancelled() -> bool:
    return _cancel_event.is_set()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ZA_CHAT_BASE_URL = "https://chat.z.ai"
ZA_LOGIN_URL = "https://chat.z.ai"
ZA_CHAT_COMPLETIONS = "/openai/v1/chat/completions"
ZA_MODELS_ENDPOINT = "/api/models"
# The OAuth callback seen in the site's login flow
ZA_OAUTH_CALLBACK_PATH = "/login/callback"
_DEFAULT_TOKEN_DIR = os.path.join(os.path.expanduser("~"), ".glossarion")
_DEFAULT_TOKEN_FILE = os.path.join(_DEFAULT_TOKEN_DIR, "authza_tokens.json")

# Token refresh margin — re-login when JWT is about to expire
# Open WebUI JWTs typically last 24-48 hours
TOKEN_REFRESH_MARGIN_SECONDS = 1800  # 30 minutes


# ---------------------------------------------------------------------------
# Token Store
# ---------------------------------------------------------------------------
class AuthZATokenStore:
    """File-backed JWT store for Z.AI, mirroring AuthGPTTokenStore's API.

    After Google OAuth login on chat.z.ai, the JWT is captured and stored
    locally.  ``get_valid_access_token()`` triggers a browser login flow
    when no valid JWT is present.
    """

    def __init__(self, token_file: Optional[str] = None, account_id: int = 0):
        self._token_file = token_file or _DEFAULT_TOKEN_FILE
        self._account_id = account_id
        self._lock = threading.Lock()
        self._tokens: Dict[str, Any] = {}
        os.makedirs(os.path.dirname(self._token_file), exist_ok=True)
        self._load()

    # -- persistence ---------------------------------------------------------

    def _load(self):
        if os.path.exists(self._token_file):
            try:
                with open(self._token_file, "r", encoding="utf-8") as fh:
                    self._tokens = json.load(fh)
            except Exception as exc:
                logger.warning("AuthZA: Failed to load %s: %s", self._token_file, exc)
                self._tokens = {}

    def _save(self):
        try:
            tmp = self._token_file + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(self._tokens, fh, indent=2)
            if os.path.exists(self._token_file):
                os.replace(tmp, self._token_file)
            else:
                os.rename(tmp, self._token_file)
        except Exception as exc:
            logger.error("AuthZA: Failed to save tokens: %s", exc)

    # -- public API ----------------------------------------------------------

    def save_tokens(self, jwt_token: str):
        with self._lock:
            self._tokens["jwt"] = jwt_token
            self._tokens["saved_at"] = time.time()
            self._save()

    def clear_tokens(self):
        with self._lock:
            self._tokens = {}
            self._save()

    @property
    def jwt(self) -> Optional[str]:
        return self._tokens.get("jwt")

    @property
    def account_id(self) -> int:
        return self._account_id

    def account_info(self) -> str:
        """Return a masked summary of the stored JWT for display."""
        token = self.jwt
        if not token:
            return "(not logged in)"
        return f"{token[:20]}…" if len(token) > 20 else token

    def _is_jwt_expired(self) -> bool:
        """Check if the stored JWT is likely expired.

        Parse the JWT payload (base64-encoded middle segment) to read
        the ``exp`` claim.  If unparseable, assume it's valid.
        """
        token = self.jwt
        if not token:
            return True
        try:
            import base64
            parts = token.split(".")
            if len(parts) != 3:
                return False  # Not a standard JWT — can't check
            # JWT base64url decode (add padding)
            payload_b64 = parts[1]
            payload_b64 += "=" * (4 - len(payload_b64) % 4)
            payload = json.loads(base64.urlsafe_b64decode(payload_b64))
            exp = payload.get("exp")
            if exp is None:
                return False  # No expiry claim — assume valid
            return time.time() >= (exp - TOKEN_REFRESH_MARGIN_SECONDS)
        except Exception:
            return False  # Can't parse — assume valid

    def get_valid_access_token(self, auto_login: bool = True) -> str:
        """Return the stored JWT, triggering a browser login if needed."""
        token = self.jwt
        if token and not self._is_jwt_expired():
            return token

        if not auto_login:
            raise RuntimeError(
                "AuthZA: No valid JWT and auto_login is False.  "
                "Run the login flow first."
            )

        acct_label = f" (Account #{self._account_id})" if self._account_id else ""
        if token and self._is_jwt_expired():
            print(f"🔄 AuthZA{acct_label}: JWT expired — launching browser to re-login…")
        else:
            print(f"🔐 AuthZA{acct_label}: No JWT found — launching browser to login…")

        captured_jwt = _run_browser_login(self._account_id)
        if not captured_jwt:
            raise RuntimeError(
                "AuthZA: Browser login was cancelled or timed out.  "
                "Please try again."
            )
        self.save_tokens(captured_jwt)
        print(f"✅ AuthZA{acct_label}: JWT saved ({self.account_info()})")
        return captured_jwt


# ---------------------------------------------------------------------------
# Browser-based login + JWT capture
# ---------------------------------------------------------------------------

def _find_free_port() -> int:
    """Find a free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


def _run_browser_login(account_id: int = 0, timeout: int = 300) -> Optional[str]:
    """Open Z.AI login page and serve a local page to capture the JWT.

    Flow:
      1. Open https://chat.z.ai → user logs in via Google
      2. After login, user is redirected to the chat interface
      3. Open a local helper page that reads localStorage['token'] from
         chat.z.ai and sends it back to our local server
      4. Alternatively, user can paste the JWT manually

    Returns the captured JWT string, or None on timeout/cancel.
    """
    port = _find_free_port()
    captured: Dict[str, Optional[str]] = {"jwt": None}
    server_ready = threading.Event()
    server_ref: Dict[str, Any] = {"server": None}

    acct_label = f" (Account #{account_id})" if account_id else ""

    # The local capture page — instructs the user to paste the JWT
    # from localStorage after logging in.
    _HTML_PAGE = f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>AuthZA — Z.AI Login Token Capture</title>
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    color: #e0e0e0;
    display: flex; justify-content: center; align-items: center;
    min-height: 100vh; margin: 0;
  }}
  .card {{
    background: rgba(255,255,255,0.06);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 40px 36px;
    max-width: 580px; width: 100%;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
  }}
  h1 {{ margin: 0 0 8px; font-size: 1.6rem; color: #7dd3fc; }}
  p {{ color: #9ca3af; line-height: 1.6; margin: 0 0 20px; font-size: 0.95rem; }}
  .step {{ color: #a78bfa; font-weight: 600; }}
  code {{
    background: rgba(0,0,0,0.4); padding: 2px 7px; border-radius: 4px;
    font-size: 0.9rem; color: #f9a8d4;
  }}
  .code-block {{
    background: rgba(0,0,0,0.5); padding: 10px 14px; border-radius: 8px;
    font-family: 'Consolas', 'Monaco', monospace; font-size: 0.85rem;
    color: #86efac; margin: 12px 0; word-break: break-all;
    user-select: all; cursor: pointer;
    border: 1px solid rgba(255,255,255,0.1);
  }}
  textarea {{
    width: 100%; padding: 12px 14px; border-radius: 8px;
    border: 1px solid rgba(255,255,255,0.2); background: rgba(0,0,0,0.3);
    color: #f3f4f6; font-size: 0.95rem; margin-bottom: 16px;
    box-sizing: border-box; outline: none; min-height: 80px;
    font-family: 'Consolas', 'Monaco', monospace;
    transition: border-color 0.2s;
    resize: vertical;
  }}
  textarea:focus {{ border-color: #7dd3fc; }}
  button {{
    width: 100%; padding: 12px; border-radius: 8px;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: #fff; font-size: 1rem; font-weight: 600;
    border: none; cursor: pointer;
    transition: opacity 0.2s, transform 0.1s;
  }}
  button:hover {{ opacity: 0.9; transform: translateY(-1px); }}
  button:active {{ transform: translateY(0); }}
  .info {{ font-size: 0.82rem; color: #6b7280; margin-top: 16px; }}
  a {{ color: #7dd3fc; text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  .success {{ display: none; text-align: center; }}
  .success h2 {{ color: #4ade80; }}
  .divider {{ border-top: 1px solid rgba(255,255,255,0.1); margin: 20px 0; }}
</style>
</head>
<body>
<div class="card" id="form-card">
  <h1>🔐 AuthZA — Z.AI Login{acct_label}</h1>
  <p>
    <span class="step">Step 1:</span> Log in to Z.AI at
    <a href="{ZA_LOGIN_URL}" target="_blank">chat.z.ai</a>
    using <b>Continue with Google</b>
    (should have opened in another tab).
  </p>
  <p>
    <span class="step">Step 2:</span> After logging in, open your browser's
    <b>Developer Console</b> (press <code>F12</code> → Console tab) and run:
  </p>
  <div class="code-block" onclick="navigator.clipboard.writeText(this.textContent.trim())" title="Click to copy">
    copy(localStorage.getItem('token'))
  </div>
  <p style="font-size:0.85rem; color:#9ca3af; margin-top:-8px;">
    (Click the box above to copy the command)
  </p>
  <p>
    <span class="step">Step 3:</span> Paste the token below and click <b>Save</b>.
  </p>
  <textarea id="jwt-input" placeholder="Paste your JWT token here…" rows="3"></textarea>
  <button onclick="submitToken()">Save Token</button>
  <div class="info">
    Your token is stored locally in <code>~/.glossarion/</code> and sent only
    to <code>chat.z.ai</code>. It typically expires after 24-48 hours.
  </div>
</div>
<div class="card success" id="success-card">
  <h2>✅ Token Saved!</h2>
  <p>You can close this tab. Glossarion will use the token automatically.<br>
  When it expires, you'll be prompted to log in again.</p>
</div>
<script>
function submitToken() {{
  const jwt = document.getElementById('jwt-input').value.trim();
  if (!jwt) {{ alert('Please paste a valid JWT token.'); return; }}
  if (jwt.length < 20) {{ alert('That doesn\\'t look like a valid JWT. It should be a long string.'); return; }}
  fetch('/capture', {{
    method: 'POST',
    headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify({{jwt: jwt}})
  }}).then(r => {{
    if (r.ok) {{
      document.getElementById('form-card').style.display = 'none';
      document.getElementById('success-card').style.display = 'block';
    }} else {{
      alert('Error saving token. Please try again.');
    }}
  }}).catch(e => alert('Connection error: ' + e));
}}
// Auto-submit on paste (UX optimization)
document.getElementById('jwt-input').addEventListener('paste', function() {{
  setTimeout(() => submitToken(), 150);
}});
</script>
</body>
</html>"""

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(_HTML_PAGE.encode("utf-8"))

        def do_POST(self):
            if self.path == "/capture":
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                try:
                    data = json.loads(body)
                    captured["jwt"] = data.get("jwt", "").strip()
                except Exception:
                    pass
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(b'{{"ok":true}}')
                # Schedule server shutdown after response is sent
                threading.Thread(target=server_ref["server"].shutdown, daemon=True).start()
            else:
                self.send_response(404)
                self.end_headers()

        def do_OPTIONS(self):
            """Handle CORS preflight."""
            self.send_response(200)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

        def log_message(self, format, *args):
            pass  # suppress request logs

    def _serve():
        server = HTTPServer(("localhost", port), _Handler)
        server.timeout = timeout
        server_ref["server"] = server
        server_ready.set()
        server.serve_forever()

    thread = threading.Thread(target=_serve, daemon=True)
    thread.start()
    server_ready.wait(5)

    # Open Z.AI login page and our local capture page
    try:
        webbrowser.open(ZA_LOGIN_URL)
        time.sleep(0.5)
        webbrowser.open(f"http://localhost:{port}/")
    except Exception as exc:
        print(f"⚠️ AuthZA: Could not open browser: {exc}")
        print(f"   Open these URLs manually:")
        print(f"   1. {ZA_LOGIN_URL}  (log in with Google)")
        print(f"   2. http://localhost:{port}/  (paste your JWT)")

    thread.join(timeout=timeout)

    return captured.get("jwt")


# ---------------------------------------------------------------------------
# Module-level store singletons
# ---------------------------------------------------------------------------
_default_store: Optional[AuthZATokenStore] = None
_default_store_lock = threading.Lock()


def get_default_store() -> AuthZATokenStore:
    """Return the module-level default token store (singleton)."""
    global _default_store
    if _default_store is None:
        with _default_store_lock:
            if _default_store is None:
                _default_store = AuthZATokenStore()
    return _default_store


_account_stores: Dict[int, AuthZATokenStore] = {}
_account_stores_lock = threading.Lock()


def get_store(account_id: Optional[int] = None) -> AuthZATokenStore:
    """Return the token store for a specific account slot.

    Parameters
    ----------
    account_id : int or None
        ``None`` or ``0`` returns the default store (``authza_tokens.json``).
        Any positive integer *N* returns a dedicated store backed by
        ``authza_tokens_N.json``, enabling multi-account usage via
        ``authzaN/`` model prefixes.
    """
    if account_id is None or account_id == 0:
        return get_default_store()

    with _account_stores_lock:
        if account_id in _account_stores:
            return _account_stores[account_id]

        token_file = os.path.join(_DEFAULT_TOKEN_DIR, f"authza_tokens_{account_id}.json")
        store = AuthZATokenStore(token_file=token_file, account_id=account_id)
        _account_stores[account_id] = store
        return store


# ---------------------------------------------------------------------------
# SSE streaming helpers (OpenAI-compatible format from Open WebUI)
# ---------------------------------------------------------------------------

def _process_sse_line(line: str, state: dict, log_fn, log_stream: bool):
    """Process a single SSE data line and accumulate content.

    Open WebUI uses standard OpenAI SSE format:
    ```
    data: {"id":"...","choices":[{"delta":{"content":"Hello"}}]}
    data: [DONE]
    ```
    """
    if not line.startswith("data: "):
        return

    payload = line[6:].strip()
    if payload == "[DONE]":
        state["done"] = True
        return

    try:
        chunk = json.loads(payload)
    except json.JSONDecodeError:
        return

    choices = chunk.get("choices", [])
    if not choices:
        # Check for usage in the final chunk
        usage = chunk.get("usage")
        if usage:
            state["usage"] = usage
        return

    choice = choices[0]
    delta = choice.get("delta", {})
    content = delta.get("content", "")
    finish = choice.get("finish_reason")

    if content:
        state["content_parts"].append(content)
        if log_stream:
            log_fn(content, end="", flush=True)

    if finish:
        state["finish_reason"] = finish

    # Capture usage if present in chunk
    usage = chunk.get("usage")
    if usage:
        state["usage"] = usage


def _finalize_stream(state: dict, t_start: float, log_fn, log_stream: bool) -> Dict:
    """Build the final result dict from accumulated stream state."""
    content = "".join(state["content_parts"])
    elapsed = time.time() - t_start

    if log_stream and content:
        log_fn("")  # newline after streamed content

    log_fn(f"✅ AuthZA: Stream complete ({elapsed:.1f}s, {len(content)} chars)")

    return {
        "content": content,
        "finish_reason": state.get("finish_reason", "stop"),
        "usage": state.get("usage"),
    }


# ---------------------------------------------------------------------------
# Chat completion sender
# ---------------------------------------------------------------------------

def _stream_with_httpx(
    _httpx,
    url: str,
    body: dict,
    headers: dict,
    timeout: int,
    t_start: float,
    log_fn,
    log_stream: bool,
    connect_timeout: Optional[float] = None,
) -> Dict:
    """Stream SSE via httpx (preferred — real-time line delivery)."""
    if connect_timeout:
        _timeout = _httpx.Timeout(timeout, connect=connect_timeout)
    else:
        _timeout = _httpx.Timeout(timeout)

    state = {"content_parts": [], "finish_reason": None, "usage": None, "done": False}

    with _httpx.Client(timeout=_timeout) as client:
        with client.stream("POST", url, json=body, headers=headers) as resp:
            if resp.status_code == 401 or resp.status_code == 403:
                resp.read()
                raise RuntimeError(
                    f"AuthZA: HTTP {resp.status_code} — JWT expired or invalid. "
                    f"Please re-login via 'authza/' prefix."
                )
            if resp.status_code != 200:
                resp.read()
                raise RuntimeError(
                    f"AuthZA: HTTP {resp.status_code} from {url}\n{resp.text}"
                )
            for raw_line in resp.iter_lines():
                if is_cancelled():
                    raise RuntimeError("AuthZA: stream cancelled by user")
                _process_sse_line(raw_line, state, log_fn, log_stream)
                if state["done"]:
                    break

    return _finalize_stream(state, t_start, log_fn, log_stream)


def _stream_with_requests(
    url: str,
    body: dict,
    headers: dict,
    timeout: int,
    t_start: float,
    log_fn,
    log_stream: bool,
) -> Dict:
    """Fallback SSE streaming via requests (buffered)."""
    import requests as _requests

    state = {"content_parts": [], "finish_reason": None, "usage": None, "done": False}

    resp = _requests.post(url, json=body, headers=headers, stream=True, timeout=timeout)
    if resp.status_code in (401, 403):
        raise RuntimeError(
            f"AuthZA: HTTP {resp.status_code} — JWT expired or invalid. "
            f"Please re-login via 'authza/' prefix."
        )
    if resp.status_code != 200:
        raise RuntimeError(
            f"AuthZA: HTTP {resp.status_code} from {url}\n{resp.text}"
        )

    try:
        for raw_line in resp.iter_lines(decode_unicode=True):
            if is_cancelled():
                raise RuntimeError("AuthZA: stream cancelled by user")
            if raw_line:
                _process_sse_line(raw_line, state, log_fn, log_stream)
                if state["done"]:
                    break
    finally:
        resp.close()

    return _finalize_stream(state, t_start, log_fn, log_stream)


def send_chat_completion(
    access_token: str,
    messages: List[Dict],
    model: str = "GLM-4.7-Flash",
    temperature: Optional[float] = 0.7,
    max_tokens: Optional[int] = None,
    timeout: int = 600,
    base_url: Optional[str] = None,
    log_fn: Optional[Any] = None,
    connect_timeout: Optional[float] = None,
    account_id: int = 0,
) -> Dict:
    """Send a chat completion request via Z.AI's Open WebUI backend.

    Parameters
    ----------
    access_token : str
        JWT obtained from Z.AI Google OAuth login.
    messages : list of dict
        Standard OpenAI-format messages (role + content).
    model : str
        Model name without the ``authza/`` prefix.
    temperature : float or None
        Sampling temperature.
    max_tokens : int or None
        Maximum response tokens.
    timeout : int
        Request timeout in seconds.
    base_url : str or None
        Override the Z.AI chat base URL.
    log_fn : callable or None
        Logging function (defaults to ``print``).
    connect_timeout : float or None
        TCP connect timeout.
    account_id : int
        Account slot for logging.

    Returns
    -------
    dict
        ``{"content": str, "finish_reason": str, "usage": dict|None}``

    Raises
    ------
    RuntimeError
        On non-200 responses, expired JWT, or stream cancellation.
    """
    effective_base = base_url or os.getenv("ZA_CHAT_BASE_URL", ZA_CHAT_BASE_URL)
    url = f"{effective_base.rstrip('/')}{ZA_CHAT_COMPLETIONS}"

    body: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": True,
    }
    if temperature is not None:
        body["temperature"] = temperature
    if max_tokens is not None:
        body["max_tokens"] = max_tokens

    # Include stream_options to get usage in the final chunk
    body["stream_options"] = {"include_usage": True}

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "Accept-Encoding": "identity",
    }

    _log = log_fn or print
    acct_label = f" (Account #{account_id})" if account_id else ""
    logger.info("AuthZA%s: POST %s  model=%s", acct_label, url, model)

    log_stream = os.getenv("LOG_STREAM_CHUNKS", "1").lower() not in ("0", "false")
    if os.getenv("BATCH_TRANSLATION", "0") == "1":
        log_stream = os.getenv("ALLOW_AUTHZA_BATCH_STREAM_LOGS", "0").lower() not in ("0", "false")

    t_start = time.time()

    try:
        import httpx as _httpx
        return _stream_with_httpx(
            _httpx, url, body, headers, timeout, t_start,
            _log, log_stream, connect_timeout=connect_timeout,
        )
    except ImportError:
        _log("⚠️ AuthZA: httpx not installed, falling back to requests (streaming may be buffered)")
        return _stream_with_requests(
            url, body, headers, timeout, t_start,
            _log, log_stream,
        )
