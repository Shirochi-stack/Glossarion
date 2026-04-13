# authza_auth.py - Z.AI (Zhipu AI) pseudo-OAuth API key management
# Mirrors authgpt_auth.py's architecture but uses a browser-based key-paste
# flow instead of real OAuth, since Z.AI relies on static API keys.
# Prefix models with 'authza/' to route through this module.
"""
Pseudo-OAuth key-capture flow for Z.AI (Zhipu AI):

  1. Start a local HTTP server on a random high port
  2. Open the Z.AI API key management page in the user's browser
  3. Serve a local HTML page where the user pastes their API key
  4. Capture the key via the local callback
  5. Store the key encrypted in ~/.glossarion/authza_tokens.json

The send_chat_completion() function then uses that key to hit Z.AI's
OpenAI-compatible chat completions endpoint with SSE streaming.
"""
import os
import json
import time
import logging
import threading
import webbrowser
import socket
from http.server import HTTPServer, BaseHTTPRequestHandler
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
ZA_BASE_URL = "https://api.z.ai/api/paas/v4"
ZA_KEY_PAGE = "https://open.bigmodel.cn/usercenter/apikeys"
_DEFAULT_TOKEN_DIR = os.path.join(os.path.expanduser("~"), ".glossarion")
_DEFAULT_TOKEN_FILE = os.path.join(_DEFAULT_TOKEN_DIR, "authza_tokens.json")


# ---------------------------------------------------------------------------
# Token / Key Store
# ---------------------------------------------------------------------------
class AuthZATokenStore:
    """File-backed API key store for Z.AI, mirroring AuthGPTTokenStore's API.

    Z.AI keys do not expire, so there is no refresh logic.  The store
    simply persists the key and provides ``get_valid_access_token()``
    which triggers a browser flow when no key is present.
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
            # Atomic rename (best-effort on Windows)
            if os.path.exists(self._token_file):
                os.replace(tmp, self._token_file)
            else:
                os.rename(tmp, self._token_file)
        except Exception as exc:
            logger.error("AuthZA: Failed to save tokens: %s", exc)

    # -- public API ----------------------------------------------------------

    def save_tokens(self, api_key: str):
        with self._lock:
            self._tokens["api_key"] = api_key
            self._tokens["saved_at"] = time.time()
            self._save()

    def clear_tokens(self):
        with self._lock:
            self._tokens = {}
            self._save()

    @property
    def api_key(self) -> Optional[str]:
        return self._tokens.get("api_key")

    @property
    def account_id(self) -> int:
        return self._account_id

    def account_info(self) -> str:
        """Return a masked summary of the stored key for display."""
        key = self.api_key
        if not key:
            return "(no key)"
        return f"{key[:8]}…" if len(key) > 8 else key

    def get_valid_access_token(self, auto_login: bool = True) -> str:
        """Return the stored API key, triggering a browser flow if needed."""
        key = self.api_key
        if key:
            return key

        if not auto_login:
            raise RuntimeError(
                "AuthZA: No API key stored and auto_login is False.  "
                "Run the login flow first."
            )

        acct_label = f" (Account #{self._account_id})" if self._account_id else ""
        print(f"🔐 AuthZA{acct_label}: No API key found — launching browser to capture key…")
        captured_key = _run_browser_key_capture(self._account_id)
        if not captured_key:
            raise RuntimeError(
                "AuthZA: Browser key capture was cancelled or timed out.  "
                "Please try again."
            )
        self.save_tokens(captured_key)
        print(f"✅ AuthZA{acct_label}: API key saved ({self.account_info()})")
        return captured_key


# ---------------------------------------------------------------------------
# Browser-based key capture
# ---------------------------------------------------------------------------

def _find_free_port() -> int:
    """Find a free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


def _run_browser_key_capture(account_id: int = 0, timeout: int = 300) -> Optional[str]:
    """Open the Z.AI key management page and serve a local form for key paste.

    Returns the captured API key string, or None on timeout/cancel.
    """
    port = _find_free_port()
    captured: Dict[str, Optional[str]] = {"key": None}
    server_ready = threading.Event()

    acct_label = f" (Account #{account_id})" if account_id else ""

    _HTML_PAGE = f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>AuthZA — Paste Your Z.AI API Key</title>
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
    max-width: 520px; width: 100%;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
  }}
  h1 {{ margin: 0 0 8px; font-size: 1.6rem; color: #7dd3fc; }}
  p {{ color: #9ca3af; line-height: 1.6; margin: 0 0 20px; font-size: 0.95rem; }}
  .step {{ color: #a78bfa; font-weight: 600; }}
  input[type=text] {{
    width: 100%; padding: 12px 14px; border-radius: 8px;
    border: 1px solid rgba(255,255,255,0.2); background: rgba(0,0,0,0.3);
    color: #f3f4f6; font-size: 1rem; margin-bottom: 16px;
    box-sizing: border-box; outline: none;
    transition: border-color 0.2s;
  }}
  input[type=text]:focus {{ border-color: #7dd3fc; }}
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
</style>
</head>
<body>
<div class="card" id="form-card">
  <h1>🔐 AuthZA Key Capture{acct_label}</h1>
  <p>
    <span class="step">Step 1:</span> Copy your API key from the
    <a href="{ZA_KEY_PAGE}" target="_blank">Z.AI API Keys page</a>
    (should have opened in another tab).<br><br>
    <span class="step">Step 2:</span> Paste the key below and click <b>Save</b>.
  </p>
  <input type="text" id="key-input" placeholder="Paste your Z.AI API key here…" autofocus>
  <button onclick="submitKey()">Save Key</button>
  <div class="info">
    Your key is stored locally in <code>~/.glossarion/</code> and never sent anywhere
    except to the Z.AI API.
  </div>
</div>
<div class="card success" id="success-card">
  <h2>✅ Key Saved!</h2>
  <p>You can close this tab. Glossarion will use the key automatically.</p>
</div>
<script>
function submitKey() {{
  const key = document.getElementById('key-input').value.trim();
  if (!key) {{ alert('Please paste a valid API key.'); return; }}
  fetch('/capture', {{
    method: 'POST',
    headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify({{key: key}})
  }}).then(r => {{
    if (r.ok) {{
      document.getElementById('form-card').style.display = 'none';
      document.getElementById('success-card').style.display = 'block';
    }} else {{
      alert('Error saving key. Please try again.');
    }}
  }}).catch(e => alert('Connection error: ' + e));
}}
// Auto-submit on paste (UX optimization)
document.getElementById('key-input').addEventListener('paste', function() {{
  setTimeout(() => submitKey(), 100);
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
                    captured["key"] = data.get("key", "").strip()
                except Exception:
                    pass
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"ok":true}')
                # Schedule server shutdown after response is sent
                threading.Thread(target=self.server.shutdown, daemon=True).start()
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, format, *args):
            pass  # suppress request logs

    def _serve():
        server = HTTPServer(("localhost", port), _Handler)
        server.timeout = timeout
        server_ready.set()
        server.serve_forever()

    thread = threading.Thread(target=_serve, daemon=True)
    thread.start()
    server_ready.wait(5)

    # Open both the Z.AI key page and our local capture page
    try:
        webbrowser.open(ZA_KEY_PAGE)
        time.sleep(0.3)
        webbrowser.open(f"http://localhost:{port}/")
    except Exception as exc:
        print(f"⚠️ AuthZA: Could not open browser: {exc}")
        print(f"   Open these URLs manually:")
        print(f"   1. {ZA_KEY_PAGE}")
        print(f"   2. http://localhost:{port}/")

    thread.join(timeout=timeout)

    return captured.get("key")


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
# SSE streaming helpers (OpenAI-compatible format)
# ---------------------------------------------------------------------------

def _process_sse_line(line: str, state: dict, log_fn, log_stream: bool):
    """Process a single SSE data line and accumulate content.

    SSE format from Z.AI (OpenAI-compatible):
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
    model: str = "glm-4-plus",
    temperature: Optional[float] = 0.7,
    max_tokens: Optional[int] = None,
    timeout: int = 600,
    base_url: Optional[str] = None,
    log_fn: Optional[Any] = None,
    connect_timeout: Optional[float] = None,
    account_id: int = 0,
) -> Dict:
    """Send a chat completion request via Z.AI's OpenAI-compatible endpoint.

    Parameters
    ----------
    access_token : str
        Z.AI API key (used as Bearer token).
    messages : list of dict
        Standard OpenAI-format messages (role + content).
    model : str
        Model name without the ``authza/`` prefix (e.g. ``glm-4-plus``).
    temperature : float or None
        Sampling temperature.
    max_tokens : int or None
        Maximum response tokens.
    timeout : int
        Request timeout in seconds.
    base_url : str or None
        Override the Z.AI base URL.
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
        On non-200 responses or stream cancellation.
    """
    effective_base = base_url or os.getenv("ZA_BASE_URL", ZA_BASE_URL)
    url = f"{effective_base.rstrip('/')}/chat/completions"

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
