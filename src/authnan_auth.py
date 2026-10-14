"""NanoGPT browser-approved keys and saved accounts for authnan/ routes.

Chat uses NanoGPT's subscription API. The handoff issues a normal API key,
which has no refresh token and remains usable until revoked or expired.
"""
import base64
import hashlib
import logging
import os
import re
import secrets
import shutil
import subprocess
import tempfile
import threading
import time
import webbrowser
from contextlib import contextmanager, nullcontext
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse

import requests
import mobile_runtime
import oauth_session

logger = logging.getLogger(__name__)
AUTH_URL = "https://nano-gpt.com/auth"
KEY_URL = "https://nano-gpt.com/api/v1/auth/keys"
SUBSCRIPTION_BASE_URL = "https://nano-gpt.com/api/subscription/v1"
API_BASE_URL = "https://nano-gpt.com/api/v1"
SCOPES = "api.use models.read"
_DEFAULT_TOKEN_DIR = oauth_session.default_token_dir()
_DEFAULT_TOKEN_FILE = os.path.join(_DEFAULT_TOKEN_DIR, "authnan_tokens.json")
_cancel_event = threading.Event()
_sessions = set()
_sessions_lock = threading.Lock()
_stores = {}
_stores_lock = threading.RLock()
_completion_lock = threading.RLock()
_pool_rotation_cursor = 0
_pool_rotation_lock = threading.Lock()
_fresh_browsers = {}
_fresh_browsers_lock = threading.RLock()


class OAuthSession(oauth_session.LoopbackOAuthSession):
    def close(self):
        try:
            super().close()
        finally:
            with _sessions_lock:
                _sessions.discard(self)


def is_cancelled():
    return _cancel_event.is_set() or os.environ.get("TRANSLATION_CANCELLED") == "1"


def cancel_stream():
    _cancel_event.set()
    with _sessions_lock:
        sessions = list(_sessions)
    for session in sessions:
        session.close()
    with _fresh_browsers_lock:
        for process, (directory, root) in list(_fresh_browsers.items()):
            _close_fresh_account_browser(process, directory, root)


def reset_cancel():
    if os.environ.get("TRANSLATION_CANCELLED") != "1":
        _cancel_event.clear()


def generate_pkce():
    verifier = secrets.token_urlsafe(32)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest()).rstrip(b"=").decode("ascii")
    return verifier, challenge


def build_auth_url(code_challenge, state, redirect_uri):
    return AUTH_URL + "?" + urlencode({
        "callback_url": redirect_uri, "code_challenge": code_challenge,
        "code_challenge_method": "S256", "scope": SCOPES, "state": state,
    })


def _normalize_tokens(tokens):
    if not isinstance(tokens, dict):
        raise RuntimeError("NanoGPT returned an invalid key response")
    key = tokens.get("key") or tokens.get("access_token")
    if not isinstance(key, str) or not key.strip() or any(c.isspace() for c in key):
        raise RuntimeError("NanoGPT did not return a usable API key")
    result = dict(tokens)
    result.pop("key", None)
    result["access_token"] = key
    result.setdefault("token_type", "Bearer")
    result.setdefault("created_at", time.time())
    if result.get("expires_in") is not None and result.get("expires_at") is None:
        result["expires_at"] = time.time() + float(result["expires_in"])
    return result


def exchange_code_for_tokens(code, code_verifier):
    response = requests.post(KEY_URL, json={
        "grant_type": "authorization_code", "code": code, "code_verifier": code_verifier,
    }, timeout=30)
    try:
        if response.status_code != 200:
            # Do not reflect provider bodies containing codes or credentials.
            raise RuntimeError(f"NanoGPT key exchange failed (HTTP {response.status_code})")
        return _normalize_tokens(response.json())
    finally:
        response.close()


class _CallbackHandler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def do_GET(self):
        session = oauth_session.session_of(self.server)
        parsed = urlparse(self.path)
        if session is None or parsed.path != "/callback":
            self.send_response(404)
            self.end_headers()
            return
        query = parse_qs(parsed.query)
        state = query.get("state", [None])[0]
        code = query.get("code", [None])[0]
        error = query.get("error", [None])[0]
        if state != session.state:
            error, code = "OAuth state mismatch", None
        elif error:
            error, code = "NanoGPT approval was denied", None
        elif not code:
            error = "Missing authorization code"
        session._record_callback(code, state, error)
        body = "Approval received. Return to Glossarion to finish signing in." if code else "NanoGPT sign-in failed. Return to Glossarion."
        if code and oauth_session.oauth_return_url():
            # Glossarion Mobile: the shared "Return to Glossarion" page, like the other sign-ins
            page = oauth_session.oauth_success_html("authnan", body, "&#10004; NanoGPT approved")
            oauth_session.send_page_quietly(self, 200, page, "text/html; charset=utf-8")
        else:
            oauth_session.send_page_quietly(self, 200 if code else 400, body, "text/plain; charset=utf-8")
        threading.Thread(target=session.close, daemon=True).start()


def begin_oauth(store=None, account_id=None, timeout=300, persist=True, auto_close=True, new_account=False):
    if store is None and account_id is not None:
        store = get_store(account_id)
    if account_id is None and store is not None:
        account_id = store._account_id
    verifier, challenge = generate_pkce()
    state = secrets.token_urlsafe(32)
    # The threaded listener times out idle browser connections too.
    server = oauth_session.make_mobile_loopback_server(("127.0.0.1", 0), _CallbackHandler)
    redirect_uri = f"http://127.0.0.1:{server.server_address[1]}/callback"
    session = OAuthSession(build_auth_url(challenge, state, redirect_uri), verifier, state,
                           redirect_uri, store=store, account_id=account_id,
                           timeout=timeout, provider="authnan",
                           extra={"expires_at": time.time() + timeout, "new_account": bool(new_account)})
    session._add_server(server)
    try:
        if persist and store is not None:
            store.save_pending_oauth(session.pending_state())
        with _sessions_lock:
            _sessions.add(session)
        session._start(auto_close)
    except BaseException:
        session.close()
        raise
    return session


def complete_from_redirect(session_or_saved_state=None, redirect_url_or_code=None, store=None):
    session = session_or_saved_state if isinstance(session_or_saved_state, OAuthSession) else None
    done = None
    try:
        done = oauth_session.resolve_completion(session_or_saved_state, redirect_url_or_code, store,
                                                provider="authnan", label="NanoGPT")
        if done.error:
            raise RuntimeError("NanoGPT sign-in failed or approval was denied")
        if not done.code:
            raise RuntimeError("NanoGPT login timed out or no authorization code was received")
        if done.returned_state != done.expected_state:
            raise RuntimeError("NanoGPT OAuth state mismatch; start sign-in again")
        expires_at = done.pending.get("expires_at")
        if expires_at is None:
            expires_at = float(done.pending.get("created_at", 0)) + 300
        if time.time() >= float(expires_at):
            raise RuntimeError("NanoGPT sign-in expired; start sign-in again")
        if is_cancelled():
            raise RuntimeError("NanoGPT sign-in cancelled")
        tokens = exchange_code_for_tokens(done.code, done.pending["code_verifier"])
        if is_cancelled():
            raise RuntimeError("NanoGPT sign-in cancelled")
        with _completion_lock:
            if done.pending.get("new_account") and tokens.get("user_id"):
                target_slot = done.pending.get("account_id")
                for slot in [0] + _numbered_account_ids():
                    other = get_store(slot)
                    if slot != target_slot and other.account_info.get("user_id") == tokens["user_id"]:
                        raise RuntimeError(f"This NanoGPT account is already saved in slot #{slot}. Sign in with a different account.")
            return oauth_session.finish_completion(done, tokens)
    finally:
        if done is not None and done.store is not None:
            done.store.clear_pending_oauth()
        if session is not None:
            session.close()


def _fresh_browser_binary():
    configured = os.environ.get("AUTHNAN_BROWSER_BINARY", "").strip()
    if configured:
        if Path(configured).is_file():
            return configured
        raise RuntimeError("AUTHNAN_BROWSER_BINARY must point to Chrome, Edge, Chromium, or Firefox.")
    candidates = []
    for root in (os.environ.get("PROGRAMFILES"), os.environ.get("PROGRAMFILES(X86)"), os.environ.get("LOCALAPPDATA")):
        if root:
            candidates.extend(str(Path(root) / path) for path in (
                "Google/Chrome/Application/chrome.exe", "Microsoft/Edge/Application/msedge.exe", "Mozilla Firefox/firefox.exe"))
    candidates.extend(("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                       "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
                       "/Applications/Firefox.app/Contents/MacOS/firefox"))
    candidates.extend(filter(None, (shutil.which(name) for name in (
        "google-chrome", "chromium", "chromium-browser", "microsoft-edge", "firefox"))))
    for candidate in candidates:
        if Path(candidate).is_file():
            return candidate
    raise RuntimeError("Adding a NanoGPT account needs Chrome, Edge, Chromium, or Firefox for a fresh signed-out window.")


@contextmanager
def _fresh_account_browser(auth_url):
    """Give + N its own empty profile, including a fresh Google login session.

    NanoGPT's sign-out URL requires a same-origin CSRF POST and confirmation;
    opening that URL alone cannot sign out the normal browser automatically.
    Never silently fall back to a browser that can reuse the previous account.
    """
    if not mobile_runtime.subprocesses_available():
        # Glossarion Mobile: a new slot signs in through the in-app sign-in sheet instead
        raise RuntimeError("A separate fresh browser for a new NanoGPT account needs the desktop app")
    binary = _fresh_browser_binary()
    temporary_root = Path(tempfile.gettempdir()).resolve()
    directory = temporary_root / ("glossarion-authnan-" + secrets.token_hex(16))
    # On Windows, inherit the user's temp-folder ACL. Python's mode=0700
    # can produce inaccessible directories under managed Windows policies.
    directory.mkdir(mode=0o777 if os.name == "nt" else 0o700)
    process = None
    try:
        if "firefox" in Path(binary).name.lower():
            args = [binary, "-no-remote", "-profile", str(directory), "-new-window", auth_url]
        else:
            args = [binary, f"--user-data-dir={directory}", "--no-first-run",
                    "--no-default-browser-check", "--disable-extensions", "--new-window", auth_url]
        process = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        with _fresh_browsers_lock:
            _fresh_browsers[process] = (directory, temporary_root)
        if is_cancelled():
            raise RuntimeError("NanoGPT sign-in cancelled")
        yield process
    finally:
        _close_fresh_account_browser(process, directory, temporary_root)


def _close_fresh_account_browser(process, directory, temporary_root):
    if not mobile_runtime.subprocesses_available():
        return  # a phone never started one
    with _fresh_browsers_lock:
        if process is not None and _fresh_browsers.pop(process, None) is None:
            return  # Cancellation already closed and cleaned this window.
        if process is not None and process.poll() is None:
            if os.name == "nt":
                # This PID belongs to the new profile process we launched.
                # Close its children too, so Chromium releases profile files.
                try:
                    subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   timeout=5, creationflags=subprocess.CREATE_NO_WINDOW)
                except (OSError, subprocess.TimeoutExpired):
                    pass
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        # Verify the absolute target before recursive cleanup; never remove
        # a browser profile other than this generated temporary directory.
        if not directory.is_symlink() and directory.resolve().parent == temporary_root:
            try:
                shutil.rmtree(directory)
            except OSError:
                logger.warning("Could not remove the temporary NanoGPT browser profile")


def run_oauth_flow(timeout=300, store=None, account_id=None, open_browser=None, new_account=False):
    session = begin_oauth(store=store, account_id=account_id, timeout=timeout, new_account=new_account)
    try:
        if is_cancelled():
            raise RuntimeError("NanoGPT sign-in cancelled")
        print("🔐 Opening browser for NanoGPT approval…")
        fresh_browser = _fresh_account_browser(session.auth_url) if new_account and open_browser is None else nullcontext()
        with fresh_browser as process:
            if not new_account or open_browser is not None:
                if (open_browser or webbrowser.open)(session.auth_url) is False:
                    raise RuntimeError("Could not open the browser for NanoGPT login")
            deadline = time.monotonic() + timeout
            while not session.callback_received:
                if is_cancelled():
                    raise RuntimeError("NanoGPT sign-in cancelled")
                if process is not None and process.poll() is not None:
                    raise RuntimeError("NanoGPT sign-in window was closed")
                if session.closed or time.monotonic() >= deadline:
                    raise RuntimeError("NanoGPT login timed out")
                session.wait_for_callback(0.2)
            return complete_from_redirect(session)
    finally:
        session.close()
        if session.store is not None:
            session.store.clear_pending_oauth()
        with _sessions_lock:
            _sessions.discard(session)


class AuthNanTokenStore(oauth_session.PendingOAuthMixin):
    _pending_label = "AuthNan"

    def __init__(self, token_file=None, account_id=0):
        self._token_file = token_file or os.environ.get("AUTHNAN_TOKEN_FILE") or _DEFAULT_TOKEN_FILE
        self._account_id = int(account_id or 0)
        self._lock = threading.RLock()
        self._login_lock = threading.RLock()
        self._tokens = None
        self._on_change_callbacks = []
        self._load_from_disk()

    def _ensure_dir(self):
        directory = os.path.dirname(self._token_file)
        if directory:
            os.makedirs(directory, exist_ok=True)

    def _load_from_disk(self):
        if os.path.isfile(self._token_file):
            try:
                from token_encryption import load_encrypted_tokens
                self._tokens = _normalize_tokens(load_encrypted_tokens(self._token_file))
            except Exception as exc:
                logger.warning("Could not load AuthNan credentials (%s)", type(exc).__name__)
                self._tokens = None

    def on_token_change(self, callback):
        self._on_change_callbacks.append(callback)

    def _fire_change_callbacks(self):
        for callback in self._on_change_callbacks:
            try:
                callback()
            except Exception:
                pass

    def save_tokens(self, tokens):
        from token_encryption import save_encrypted_tokens
        tokens = _normalize_tokens(tokens)
        with self._lock:
            self._ensure_dir()
            save_encrypted_tokens(tokens, self._token_file)
            self._tokens = tokens
        self._fire_change_callbacks()

    def load_tokens(self):
        with self._lock:
            if self._tokens is None:
                self._load_from_disk()
            return dict(self._tokens) if self._tokens is not None else None

    def clear_tokens(self):
        with self._lock:
            if os.path.isfile(self._token_file):
                os.remove(self._token_file)
            self._tokens = None
            self.clear_pending_oauth()
        self._fire_change_callbacks()

    @property
    def has_tokens(self):
        tokens = self.load_tokens()
        if not tokens or not tokens.get("access_token"):
            return False
        try:
            return tokens.get("expires_at") is None or float(tokens["expires_at"]) > time.time()
        except (TypeError, ValueError):
            return False

    @property
    def account_info(self):
        tokens = self.load_tokens() or {}
        return {k: tokens[k] for k in ("user_id", "email", "scope") if tokens.get(k)}

    def get_valid_access_token(self, auto_login=True):
        with self._lock:
            if self.has_tokens:
                return self._tokens["access_token"]
        if not auto_login:
            raise RuntimeError(f"Sign in with NanoGPT for account slot #{self._account_id}")
        # Serialize login without holding the credential lock during browser
        # interaction. Other slots can inspect identity metadata concurrently.
        with self._login_lock:
            with self._lock:
                if self.has_tokens:
                    return self._tokens["access_token"]
            tokens = run_oauth_flow(store=self)
            return tokens["access_token"]

    def recover_from_unauthorized(self, rejected_access_token, auto_login=True):
        with self._login_lock:
            with self._lock:
                tokens = self.load_tokens()
                if tokens and tokens['access_token'] == rejected_access_token:
                    self.clear_tokens()
            return self.get_valid_access_token(auto_login=auto_login)


def get_store(account_id=None):
    account_id = int(account_id or 0)
    if not 0 <= account_id <= 9999:
        raise ValueError("NanoGPT account slot must be between 0 and 9999")
    path = (os.environ.get("AUTHNAN_TOKEN_FILE") or _DEFAULT_TOKEN_FILE) if not account_id else os.path.join(_DEFAULT_TOKEN_DIR, f"authnan_tokens_{account_id}.json")
    with _stores_lock:
        key = (account_id, os.path.abspath(path))
        if key not in _stores:
            _stores[key] = AuthNanTokenStore(path, account_id)
        return _stores[key]


def get_default_store():
    return get_store(0)


def _numbered_account_ids():
    try:
        return sorted({int(match.group(1)) for name in os.listdir(_DEFAULT_TOKEN_DIR)
                       if (match := re.fullmatch(r"authnan_tokens_([1-9]\d{0,3})\.json", name))})
    except OSError:
        return []


def get_account_pool():
    candidates, identities = [], set()
    for account_id in [0] + _numbered_account_ids():
        store = get_store(account_id)
        if not store.has_tokens:
            continue
        identity = store.account_info.get("user_id")
        if identity and identity in identities:
            continue
        if identity:
            identities.add(identity)
        candidates.append((account_id, store))
    return candidates


def get_rotating_account_pool():
    global _pool_rotation_cursor
    candidates = get_account_pool()
    with _pool_rotation_lock:
        if not candidates:
            return []
        start = _pool_rotation_cursor % len(candidates)
        _pool_rotation_cursor += 1
    return candidates[start:] + candidates[:start]


def fetch_available_models(access_token, timeout=15):
    from model_options import ProviderCatalogSpec, _extract_catalog_entries, _http_get_json
    spec = ProviderCatalogSpec("authnan", "", SUBSCRIPTION_BASE_URL + "/models")
    headers = {"Authorization": f"Bearer {access_token}"}
    models = []
    for url in (SUBSCRIPTION_BASE_URL + "/models?detailed=true",
                API_BASE_URL + "/image-models", API_BASE_URL + "/video-models"):
        payload = _http_get_json(url, headers, timeout)
        entries = _extract_catalog_entries(payload, spec)
        for entry in entries:
            if isinstance(entry, dict) and (entry.get("active") is False or entry.get("archived") is True):
                continue
            model = entry if isinstance(entry, str) else entry.get("id", entry.get("model", "")) if isinstance(entry, dict) else ""
            if isinstance(model, str) and model and not any(c.isspace() for c in model) and model not in models:
                models.append(model)
    return models
