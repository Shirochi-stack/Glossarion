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
import sys
import json
import time
import hashlib
import base64
import secrets
import logging
import threading
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlencode, urlparse, parse_qs, quote
from typing import Optional, Dict, List, Tuple, Any

import requests

logger = logging.getLogger(__name__)

# Module-level cancellation flag
_cancel_event = threading.Event()

def cancel_stream():
    """Signal any active AuthCD stream to abort immediately."""
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
# Constants - mirror Claude Code CLI OAuth values
# ===========================================================================
CLAUDE_CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
CLAUDE_AUTH_URL = "https://claude.ai/oauth/authorize"
CLAUDE_TOKEN_URL = "https://platform.claude.com/v1/oauth/token"
# Anthropic's registered redirect_uri – localhost is NOT supported by this client
CLAUDE_REDIRECT_URI = "https://platform.claude.com/oauth/code/callback"
SCOPES = "user:profile user:inference org:create_api_key"
TOKEN_REFRESH_MARGIN_SECONDS = 300  # refresh when <5 min remaining

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_API_VERSION = "2023-06-01"
# Claude Code reads the signed-in account (email, organization) from here.
ANTHROPIC_OAUTH_PROFILE_URL = "https://api.anthropic.com/api/oauth/profile"

# Claude Code identity headers – required for OAuth token acceptance.
# Anthropic reads the Claude Code version from the User-Agent and refuses
# newer models for old versions ("Claude Code 2.1.119 does not support this
# model; version 2.1.280 or newer is required"). The version sent is the
# newest of this default, the installed Claude Code CLI, and any version a
# refusal asked for (remembered in _CLIENT_VERSION_FILE).
_CLAUDE_CODE_DEFAULT_VERSION = "2.1.280"
_CLAUDE_CODE_BETA_FLAGS = "claude-code-20250219,oauth-2025-04-20"

_DEFAULT_TOKEN_DIR = os.path.join(os.path.expanduser("~"), ".glossarion")
_DEFAULT_TOKEN_FILE = os.path.join(_DEFAULT_TOKEN_DIR, "authcd_tokens.json")
_CLIENT_VERSION_FILE = os.path.join(_DEFAULT_TOKEN_DIR, "authcd_client_version.json")
_client_version_lock = threading.Lock()
_client_version: Optional[str] = None


def _version_tuple(version: str) -> Tuple[int, ...]:
    try:
        return tuple(int(part) for part in str(version).strip().split("."))
    except (TypeError, ValueError):
        return ()


def _newest_version(*versions: Optional[str]) -> str:
    valid = [v for v in versions if v and _version_tuple(v)]
    return max(valid, key=_version_tuple) if valid else _CLAUDE_CODE_DEFAULT_VERSION


def _installed_claude_code_version() -> Optional[str]:
    """Version of the npm-installed Claude Code CLI, read from its package.json."""
    import shutil
    shim = shutil.which("claude")
    if not shim:
        return None
    base = os.path.dirname(os.path.realpath(shim))
    for candidate in (
        os.path.join(base, "node_modules", "@anthropic-ai", "claude-code", "package.json"),
        os.path.join(base, "..", "lib", "node_modules", "@anthropic-ai", "claude-code", "package.json"),
        os.path.join(base, "..", "package.json"),
    ):
        try:
            with open(candidate, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("name") == "@anthropic-ai/claude-code" and data.get("version"):
                return str(data["version"])
        except Exception:
            continue
    return None


def _saved_client_version() -> Optional[str]:
    try:
        with open(_CLIENT_VERSION_FILE, "r", encoding="utf-8") as f:
            return str(json.load(f).get("version") or "") or None
    except Exception:
        return None


def claude_code_client_version() -> str:
    """The Claude Code version AuthCD identifies as."""
    global _client_version
    with _client_version_lock:
        if _client_version is None:
            _client_version = _newest_version(
                _CLAUDE_CODE_DEFAULT_VERSION, _installed_claude_code_version(), _saved_client_version(),
            )
        return _client_version


def _claude_code_user_agent() -> str:
    return f"claude-code/{claude_code_client_version()}"


def _adopt_client_version(required: str) -> bool:
    """Identify as ``required`` from now on (and after restarts). False if not newer."""
    global _client_version
    current = claude_code_client_version()
    if not _version_tuple(required) or _version_tuple(required) <= _version_tuple(current):
        return False
    with _client_version_lock:
        _client_version = required
    try:
        os.makedirs(_DEFAULT_TOKEN_DIR, exist_ok=True)
        with open(_CLIENT_VERSION_FILE, "w", encoding="utf-8") as f:
            json.dump({"version": required}, f)
    except Exception as exc:
        logger.debug("AuthCD: could not save client version: %s", exc)
    return True

# Claude Code credential paths (for parasitic fallback)
_CLAUDE_CODE_CREDS = os.path.join(os.path.expanduser("~"), ".claude", ".credentials.json")

# Glossarion runs `claude auth login` against its own Claude Code config
# directory. The user's ~/.claude/settings.json can carry an "env" block
# (e.g. ANTHROPIC_AUTH_TOKEN / ANTHROPIC_BASE_URL for a proxy) that Claude
# Code applies to every run; its post-login validation then rejects the new
# OAuth login and exits without keeping it. A fresh config dir has no such
# settings, and there the CLI writes a plain .credentials.json we can read.
GLOSSARION_CLAUDE_CONFIG_DIR = os.path.join(_DEFAULT_TOKEN_DIR, "claude-code")
_GLOSSARION_CLAUDE_CODE_CREDS = os.path.join(GLOSSARION_CLAUDE_CONFIG_DIR, ".credentials.json")


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


def build_auth_url(code_challenge: str, state: str) -> str:
    """Build the full authorization URL using the official redirect_uri."""
    params = {
        "client_id": CLAUDE_CLIENT_ID,
        "response_type": "code",
        "redirect_uri": CLAUDE_REDIRECT_URI,
        "scope": SCOPES,
        "state": state,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
    }
    return f"{CLAUDE_AUTH_URL}?{urlencode(params)}"


# ===========================================================================
# Token exchange / refresh
# ===========================================================================

# Claude Code posts to the OAuth token endpoint through axios with only a JSON
# content type, so the request carries axios's default User-Agent and Accept.
# The Messages-API identity headers (claude-code UA, anthropic-beta, x-app)
# are for inference calls only; on the token endpoint they drew HTTP 429
# "Rate limited" while a plain request was answered normally.
_TOKEN_ENDPOINT_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/plain, */*",
    "User-Agent": "axios/1.15.2",
}
_TOKEN_RATE_LIMIT_WAITS = (5, 15, 30, 60)


def _post_token_endpoint(payload: Dict) -> "requests.Response":
    """POST to the token endpoint, waiting out HTTP 429 (a 429 does not use up the code)."""
    resp = None
    for attempt, default_wait in enumerate((0,) + _TOKEN_RATE_LIMIT_WAITS):
        if attempt:
            wait = default_wait
            try:
                retry_after = float(resp.headers.get("Retry-After", ""))
                if 0 < retry_after <= 120:
                    wait = retry_after
            except (TypeError, ValueError):
                pass
            logger.warning("AuthCD token endpoint rate limited; retrying in %.0fs", wait)
            print(f"⏳ AuthCD: Anthropic rate-limited the sign-in; retrying in {wait:.0f}s…")
            deadline = time.time() + wait
            while time.time() < deadline:
                if is_cancelled():
                    raise RuntimeError("AuthCD: Login cancelled.")
                time.sleep(0.2)
        resp = requests.post(CLAUDE_TOKEN_URL, json=payload, headers=_TOKEN_ENDPOINT_HEADERS, timeout=30)
        if resp.status_code != 429:
            break
    return resp


def exchange_code_for_tokens(auth_code: str, code_verifier: str,
                             redirect_uri: Optional[str] = None,
                             state: Optional[str] = None) -> Dict:
    """Exchange authorization code for access + refresh tokens."""
    payload = {
        "grant_type": "authorization_code",
        "code": auth_code,
        "redirect_uri": redirect_uri or CLAUDE_REDIRECT_URI,
        "client_id": CLAUDE_CLIENT_ID,
        "code_verifier": code_verifier,
    }
    if state:
        payload["state"] = state
    logger.info("AuthCD token exchange")
    resp = _post_token_endpoint(payload)
    if resp.status_code >= 400:
        try:
            err_body = resp.json()
        except Exception:
            err_body = resp.text
        logger.error("AuthCD token exchange failed: %s %s", resp.status_code, err_body)
        raise RuntimeError(
            f"Token exchange failed ({resp.status_code}): {err_body}"
        )
    data = resp.json()
    data["expires_at"] = time.time() + data.get("expires_in", 3600)
    return data


def refresh_access_token(refresh_token: str) -> Dict:
    """Use a refresh token to obtain a new access token."""
    payload = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": CLAUDE_CLIENT_ID,
        # Claude Code sends the scope list it signed in with.
        "scope": " ".join(s for s in CLAUDE_CODE_LOGIN_SCOPES.split() if s != "org:create_api_key"),
    }
    resp = _post_token_endpoint(payload)
    if resp.status_code == 400 and "invalid_scope" in (resp.text or ""):
        # A token from an older sign-in may carry fewer scopes; Claude Code
        # retries such a refresh without the scope list too.
        payload.pop("scope", None)
        resp = _post_token_endpoint(payload)
    resp.raise_for_status()
    data = resp.json()
    data["expires_at"] = time.time() + data.get("expires_in", 3600)
    return data


# ===========================================================================
# Automatic browser login (what `claude auth login` does, done in-process)
# ===========================================================================
# Claude Code's automatic login opens the browser with a localhost redirect,
# catches the code on 127.0.0.1 and exchanges it. Running that here means the
# token request goes through Python/certifi. The native Claude Code CLI on
# Windows cannot verify platform.claude.com's current Let's Encrypt chain
# (YE1 -> Root YE -> ISRG Root X2) and fails with UNABLE_TO_GET_ISSUER_CERT,
# whatever NODE_EXTRA_CA_CERTS says.
CLAUDE_AI_AUTHORIZE_URL = "https://claude.com/cai/oauth/authorize"
CLAUDE_AI_SUCCESS_URL = "https://platform.claude.com/oauth/code/success?app=claude-code"
# The sign-in opens this page first so the browser drops its current claude.ai
# session and the user can pick the account. Its returnTo (a claude.ai path)
# then continues to the authorize page in the same tab; claude.com's
# authorize URL is a redirect to claude.ai/oauth/authorize.
CLAUDE_AI_LOGOUT_URL = "https://claude.ai/logout"
CLAUDE_AI_AUTHORIZE_PATH = "/oauth/authorize"
CLAUDE_AI_SIGN_OUT_WAIT_SECONDS = 4.0
CLAUDE_CODE_LOGIN_SCOPES = (
    "org:create_api_key user:profile user:inference user:sessions:claude_code "
    "user:mcp_servers user:file_upload user:plugins"
)


def _bind_oauth_callback_server(handler_cls):
    """Bind 127.0.0.1 on a free port in Claude Code's callback port range."""
    low, high = (39152, 49151) if os.name == "nt" else (49152, 65535)
    for _ in range(50):
        port = secrets.randbelow(high - low + 1) + low
        try:
            return HTTPServer(("127.0.0.1", port), handler_cls)
        except OSError:
            continue
    return HTTPServer(("127.0.0.1", 0), handler_cls)


def run_automatic_oauth_login(timeout: int = 180, open_browser=None,
                              sign_out_first: bool = True,
                              sign_out_wait: float = CLAUDE_AI_SIGN_OUT_WAIT_SECONDS) -> Dict:
    """Sign in through the browser and return tokens, with no pasting.

    ``open_browser(url)`` defaults to ``webbrowser.open``. With
    ``sign_out_first`` the browser is first signed out of claude.ai, so the
    authorize page asks for an account instead of reusing the one already
    signed in. The browser then redirects to http://localhost:<port>/callback,
    which this function answers, and is sent on to Anthropic's success page.
    """
    code_verifier, code_challenge = generate_pkce()
    state = secrets.token_urlsafe(32)
    result: Dict[str, Optional[str]] = {"code": None, "error": None}
    received = threading.Event()

    class _CallbackHandler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            return

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path != "/callback":
                self.send_response(404)
                self.end_headers()
                return
            query = parse_qs(parsed.query)
            if query.get("state", [None])[0] != state:
                result["error"] = "state mismatch in the sign-in callback"
            elif query.get("error"):
                result["error"] = query.get("error_description", query["error"])[0]
            elif query.get("code"):
                result["code"] = query["code"][0]
            else:
                result["error"] = "no authorization code in the sign-in callback"
            if result["code"]:
                self.send_response(302)
                self.send_header("Location", CLAUDE_AI_SUCCESS_URL)
                self.end_headers()
            else:
                body = f"Claude sign-in failed: {result['error']}".encode("utf-8")
                self.send_response(400)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            received.set()

    server = _bind_oauth_callback_server(_CallbackHandler)
    server.timeout = 0.5
    port = server.server_address[1]
    redirect_uri = f"http://localhost:{port}/callback"
    params = {
        "code": "true",
        "client_id": CLAUDE_CLIENT_ID,
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "scope": CLAUDE_CODE_LOGIN_SCOPES,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "state": state,
    }
    query = urlencode(params)
    if sign_out_first:
        # One tab: sign out, then claude.ai goes on to the authorize page,
        # which asks for the account (login?selectAccount=true).
        return_to = quote(f"{CLAUDE_AI_AUTHORIZE_PATH}?{query}", safe="")
        auth_url = f"{CLAUDE_AI_LOGOUT_URL}?returnTo={return_to}"
    else:
        auth_url = f"{CLAUDE_AI_AUTHORIZE_URL}?{query}"
    del sign_out_wait  # kept for callers; the redirect replaces the wait
    open_url = open_browser or webbrowser.open
    try:
        open_url(auth_url)
        deadline = time.time() + timeout
        while not received.is_set():
            if is_cancelled():
                raise RuntimeError("AuthCD: Login cancelled.")
            if time.time() > deadline:
                raise TimeoutError("AuthCD: Login timed out waiting for the browser sign-in.")
            server.handle_request()
    finally:
        server.server_close()
    if not result["code"]:
        raise RuntimeError(f"AuthCD: {result['error']}")
    tokens = exchange_code_for_tokens(
        result["code"], code_verifier, redirect_uri=redirect_uri, state=state,
    )
    tokens["_source"] = "glossarion_oauth"
    return tokens


def login_automatically(store, claude_bin: Optional[str] = None, timeout: int = 180,
                        sign_out_first: bool = True) -> Dict:
    """Browser sign-in for the authcd store; the Claude Code CLI is only a fallback.

    The browser is signed out of claude.ai first so the account can be chosen.
    """
    # No Claude Code CLI fallback: it starts a second browser sign-in, and
    # on Windows its own TLS check fails against platform.claude.com.
    del claude_bin
    tokens = run_automatic_oauth_login(timeout=timeout, sign_out_first=sign_out_first)
    store.save_tokens(tokens)
    return tokens


# ===========================================================================
# OAuth flow – split API for GUI integration
# ===========================================================================

def start_oauth_flow() -> Dict:
    """Start the OAuth flow: generate PKCE, build URL, open browser.

    Returns a dict with {auth_url, code_verifier, state} needed
    to complete the flow after the user copies the auth code from
    Anthropic's callback page.
    """
    code_verifier, code_challenge = generate_pkce()
    state = secrets.token_urlsafe(32)
    auth_url = build_auth_url(code_challenge, state)

    print(f"🔐 Opening browser for Claude login…")
    print(f"   If the browser doesn't open, visit:\n   {auth_url}")
    webbrowser.open(auth_url)

    return {
        "auth_url": auth_url,
        "code_verifier": code_verifier,
        "state": state,
    }


def complete_oauth_exchange(auth_code: str, code_verifier: str) -> Dict:
    """Complete the OAuth flow by exchanging the code for tokens.

    Args:
        auth_code: The authorization code copied from Anthropic's callback page.
        code_verifier: The PKCE code_verifier from start_oauth_flow().

    Returns:
        Token dict with access_token, refresh_token, expires_at, etc.
    """
    print("🔑 Exchanging authorization code for tokens…")
    tokens = exchange_code_for_tokens(auth_code, code_verifier)
    print("✅ Claude OAuth authentication successful!")
    return tokens


def run_oauth_flow(timeout: int = 300) -> Dict:
    """Run the full OAuth PKCE login flow (CLI mode).

    For GUI usage, prefer start_oauth_flow() + complete_oauth_exchange().
    This function opens a browser and prompts for the code on stdin.
    """
    flow = start_oauth_flow()

    print("\n   After logging in, Anthropic will show an Authentication Code.")
    print("   Copy it and paste it here:")
    auth_code = input("   Auth code: ").strip()
    if not auth_code:
        raise RuntimeError("No auth code provided.")

    return complete_oauth_exchange(auth_code, flow["code_verifier"])


# ===========================================================================
# Claude Code credentials parasitic loader
# ===========================================================================

# Claude Code on Windows (Bun-based builds, 2.1.x) keeps the login in Windows
# Credential Manager instead of ~/.claude/.credentials.json. Service
# "Claude Code-credentials" (plus "-<hash>" when CLAUDE_CONFIG_DIR is set),
# account "claude-code-user"; values over 2400 bytes are split into base64
# chunks "claude-code-user#0..n" with a {"n","l"} record at "#m".
_CLAUDE_CODE_CREDMAN_SERVICE = "Claude Code-credentials"
_CLAUDE_CODE_CREDMAN_ACCOUNT = "claude-code-user"


def _credman_decode(blob: bytes) -> Optional[str]:
    if not blob:
        return None
    if len(blob) >= 2 and blob[1:2] == b"\x00":
        try:
            return blob.decode("utf-16-le")
        except UnicodeDecodeError:
            pass
    try:
        return blob.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _read_windows_credman_entries(service_substring: str) -> Dict[str, str]:
    """Return {entry name: value} for generic credentials of one service.

    The entry name is the account part ("claude-code-user", "claude-code-user#m",
    ...). Works for either target layout ("service/account" or a bare service
    target with the account in UserName). ``service_substring`` must match the
    whole service name for callers that need one exact store; see
    :func:`_claude_code_credman_service`.
    """
    if os.name != "nt":
        return {}
    import ctypes
    from ctypes import wintypes

    class CREDENTIALW(ctypes.Structure):
        _fields_ = [
            ("Flags", wintypes.DWORD), ("Type", wintypes.DWORD),
            ("TargetName", wintypes.LPWSTR), ("Comment", wintypes.LPWSTR),
            ("LastWritten", wintypes.FILETIME),
            ("CredentialBlobSize", wintypes.DWORD),
            ("CredentialBlob", ctypes.POINTER(ctypes.c_ubyte)),
            ("Persist", wintypes.DWORD), ("AttributeCount", wintypes.DWORD),
            ("Attributes", ctypes.c_void_p), ("TargetAlias", wintypes.LPWSTR),
            ("UserName", wintypes.LPWSTR),
        ]

    entries: Dict[str, str] = {}
    try:
        advapi = ctypes.WinDLL("advapi32", use_last_error=True)
        count = wintypes.DWORD()
        creds = ctypes.POINTER(ctypes.POINTER(CREDENTIALW))()
        # CredEnumerateW filters only by "prefix*", and the service name can
        # carry a config-dir hash suffix, so enumerate and match here.
        if not advapi.CredEnumerateW(None, 0, ctypes.byref(count), ctypes.byref(creds)):
            return {}
        try:
            for i in range(count.value):
                cred = creds[i].contents
                target = cred.TargetName or ""
                if service_substring not in target:
                    continue
                user = cred.UserName or ""
                if user.startswith(_CLAUDE_CODE_CREDMAN_ACCOUNT):
                    name = user
                else:
                    name = target.rsplit("/", 1)[-1] if "/" in target else user
                if not name.startswith(_CLAUDE_CODE_CREDMAN_ACCOUNT):
                    continue
                size = int(cred.CredentialBlobSize or 0)
                blob = bytes(cred.CredentialBlob[:size]) if size else b""
                value = _credman_decode(blob)
                if value is not None:
                    entries[name] = value
        finally:
            advapi.CredFree(creds)
    except Exception as exc:
        logger.debug("Credential Manager read failed: %s", exc)
    return entries


def _claude_code_credman_service(config_dir: Optional[str] = None) -> str:
    """Service name Claude Code uses for one config dir (hash only for a custom dir)."""
    if not config_dir:
        return _CLAUDE_CODE_CREDMAN_SERVICE
    import unicodedata
    digest = hashlib.sha256(
        unicodedata.normalize("NFC", config_dir).encode("utf-8")
    ).hexdigest()[:8]
    return f"{_CLAUDE_CODE_CREDMAN_SERVICE}-{digest}"


def _decode_keychain_value(value: str) -> Optional[str]:
    """`security -w` prints binary items as hex; JSON items as-is."""
    value = (value or "").strip()
    if value and len(value) % 2 == 0 and all(c in "0123456789abcdefABCDEF" for c in value):
        try:
            decoded = bytes.fromhex(value).decode("utf-8")
            if decoded.lstrip().startswith("{"):
                return decoded
        except (ValueError, UnicodeDecodeError):
            pass
    return value or None


def _load_claude_code_keychain_json(config_dir: Optional[str] = None) -> Optional[Dict]:
    """Claude Code's stored credential JSON from the macOS Keychain."""
    if sys.platform != "darwin":
        return None
    import getpass
    import subprocess
    service = _claude_code_credman_service(config_dir)
    accounts = [_CLAUDE_CODE_CREDMAN_ACCOUNT]
    try:
        accounts.append(getpass.getuser())  # older Claude Code builds
    except Exception:
        pass
    for account in accounts:
        try:
            result = subprocess.run(
                ["security", "find-generic-password", "-a", account, "-w", "-s", service],
                capture_output=True, text=True, timeout=10,
            )
        except Exception:
            continue
        if result.returncode != 0:
            continue
        text = _decode_keychain_value(result.stdout)
        if not text:
            continue
        try:
            data = json.loads(text)
        except ValueError:
            continue
        if isinstance(data, dict):
            return data
    return None


def _load_claude_code_credman_json(config_dir: Optional[str] = None) -> Optional[Dict]:
    """Claude Code's stored credential JSON from Windows Credential Manager."""
    service = _claude_code_credman_service(config_dir)
    entries = _read_windows_credman_entries(service + "/") or (
        _read_windows_credman_entries(service) if config_dir else {}
    )
    if not entries and not config_dir:
        entries = _read_windows_credman_entries(service)
    if not entries:
        return None
    account = _CLAUDE_CODE_CREDMAN_ACCOUNT
    text = None
    meta_raw = entries.get(f"{account}#m")
    if meta_raw:
        try:
            meta = json.loads(meta_raw)
            parts = [entries.get(f"{account}#{i}") for i in range(int(meta["n"]))]
            joined = "".join(parts) if all(p is not None for p in parts) else ""
            if joined and len(joined) == int(meta["l"]):
                text = base64.b64decode(joined).decode("utf-8")
        except Exception as exc:
            logger.debug("Chunked Credential Manager entry unreadable: %s", exc)
    if text is None:
        text = entries.get(account)
    if not text:
        return None
    try:
        data = json.loads(text)
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def claude_cli_environment() -> Dict[str, str]:
    """Environment for running the Claude Code CLI from Glossarion.

    Glossarion sets ANTHROPIC_BASE_URL (often to an empty string or a
    third-party gateway) and ANTHROPIC_API_KEY for its own API routes. The
    CLI would inherit them and use them during `claude auth login`, which
    breaks the OAuth login even though the browser step succeeds.
    """
    env = dict(os.environ)
    for name in list(env):
        if name.upper().startswith("ANTHROPIC_"):
            env.pop(name, None)
    env.pop("CLAUDE_SECURESTORAGE_CONFIG_DIR", None)
    env.pop("CLAUDE_CODE_FORCE_WINDOWS_CREDMAN", None)
    # platform.claude.com (the OAuth token endpoint) is served from Let's
    # Encrypt's newer hierarchy. A Windows root store that has not fetched
    # that root yet fails the CLI's token exchange with
    # UNABLE_TO_GET_ISSUER_CERT after the browser step. certifi's bundle is
    # added (NODE_EXTRA_CA_CERTS extends, never replaces, the trusted roots).
    if not env.get("NODE_EXTRA_CA_CERTS"):
        try:
            import certifi
            bundle = certifi.where()
            if bundle and os.path.isfile(bundle):
                env["NODE_EXTRA_CA_CERTS"] = bundle
        except Exception:
            pass
    try:
        os.makedirs(GLOSSARION_CLAUDE_CONFIG_DIR, exist_ok=True)
    except OSError:
        pass
    env["CLAUDE_CONFIG_DIR"] = GLOSSARION_CLAUDE_CONFIG_DIR
    return env


def run_claude_cli_login(claude_bin: str, timeout: int = 180) -> Tuple[Optional[int], str]:
    """Run `claude auth login` automatically; return (exit code, output tail).

    No console window: the CLI opens the browser and receives the OAuth
    callback on its own. Its output goes to a log file so a failure can be
    reported with the CLI's own message instead of a vanished window.
    """
    import subprocess
    log_path = os.path.join(GLOSSARION_CLAUDE_CONFIG_DIR, "last_login.log")
    env = claude_cli_environment()
    returncode = None
    with open(log_path, "w", encoding="utf-8", errors="replace") as log_file:
        proc = subprocess.Popen(
            [claude_bin, "auth", "login"],
            stdin=subprocess.PIPE, stdout=log_file, stderr=subprocess.STDOUT,
            env=env, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        try:
            returncode = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
            raise
        finally:
            try:
                if proc.stdin:
                    proc.stdin.close()
            except Exception:
                pass
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as log_file:
            lines = [line.rstrip() for line in log_file if line.strip()]
    except OSError:
        lines = []
    # The authorize URL is long and carries the PKCE challenge; keep it out.
    lines = [line for line in lines if "oauth/authorize" not in line]
    return returncode, "\n".join(lines[-8:])


def claude_cli_auth_status(claude_bin: str) -> Optional[Dict]:
    """`claude auth status` as a dict, or None when it cannot be read."""
    import subprocess
    try:
        result = subprocess.run(
            [claude_bin, "auth", "status"], capture_output=True, text=True,
            timeout=30, env=claude_cli_environment(),
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        data = json.loads((result.stdout or "").strip() or "null")
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def claude_login_failure_message(claude_bin: Optional[str], cli_output: str = "") -> str:
    """Explain a login that finished without usable credentials."""
    status = claude_cli_auth_status(claude_bin) if claude_bin else None
    detail = f"\n\nClaude Code said:\n{cli_output}" if cli_output else ""
    if status is not None and not status.get("loggedIn"):
        return (
            "Claude Code did not keep the login after the browser step."
            + detail
        )
    if status is not None and status.get("loggedIn"):
        return (
            "Claude Code reports it is logged in "
            f"(method: {status.get('authMethod')}), but its stored credentials "
            "could not be read." + detail
        )
    return "Claude login completed but no credentials found." + detail


def _load_claude_code_credentials() -> Optional[Dict]:
    """Try to load existing credentials from Claude Code's local store.

    Checks ~/.claude/.credentials.json first, then Windows Credential
    Manager, where current Claude Code builds keep them on Windows.
    """
    try:
        def _has_token(value):
            return isinstance(value, dict) and bool(
                value.get("claudeAiOauth") or value.get("accessToken") or value.get("access_token")
            )

        data = None
        # Glossarion's own login (isolated config dir) first, then the
        # user's regular Claude Code login.
        for path, config_dir in (
            (_GLOSSARION_CLAUDE_CODE_CREDS, GLOSSARION_CLAUDE_CONFIG_DIR),
            (_CLAUDE_CODE_CREDS, None),
        ):
            source_path = None
            if os.path.isfile(path):
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                source_path = path
            if not _has_token(data):
                source_path = None
                data = (
                    _load_claude_code_credman_json(config_dir)
                    or _load_claude_code_keychain_json(config_dir)
                )
            if _has_token(data):
                break
        if not _has_token(data):
            return None
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
            if expires_at > 1e12:
                # Claude Code records expiresAt in epoch milliseconds.
                expires_at /= 1000.0
        elif isinstance(expires_at_str, str):
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(expires_at_str.replace("Z", "+00:00"))
                expires_at = dt.timestamp()
            except Exception:
                expires_at = time.time() + 3600
        creds = {
            "access_token": access,
            "refresh_token": refresh,
            "expires_at": expires_at,
            "_source": "claude_code",
        }
        if source_path == _GLOSSARION_CLAUDE_CODE_CREDS:
            # Plaintext handoff file from Glossarion's own login; see
            # import_claude_code_login(), which removes it once stored.
            creds["_plaintext_handoff"] = source_path
        return creds
    except Exception as exc:
        logger.debug("Failed to load Claude Code credentials: %s", exc)
        return None


def import_claude_code_login(store) -> Optional[Dict]:
    """Copy Claude Code's login into the encrypted authcd store.

    The token then lives in Glossarion's token_encryption store (DPAPI on
    Windows, Keychain-held key on macOS, 0600 key file on Linux). The
    plaintext .credentials.json that the CLI wrote into Glossarion's own
    config dir is removed once the encrypted copy is saved; the user's
    personal ~/.claude credentials are never touched.
    """
    creds = _load_claude_code_credentials()
    if not creds or not creds.get("access_token"):
        return None
    handoff = creds.pop("_plaintext_handoff", None)
    store.save_tokens(creds)
    if handoff and os.path.isfile(store._token_file) and os.path.abspath(handoff) == os.path.abspath(_GLOSSARION_CLAUDE_CODE_CREDS):
        try:
            os.remove(handoff)
        except OSError as exc:
            logger.warning("AuthCD: could not remove plaintext login handoff file: %s", exc)
    return creds


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
        self._cleared = False  # prevents fallback reload after explicit logout
        self._on_change_callbacks: List = []
        self._email_lookup_failed_token: Optional[str] = None
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
                        logger.warning("🔓 AuthCD: loaded UNENCRYPTED credentials; encryption module is unavailable")
                except Exception as dec_exc:
                    logger.warning("❌ AuthCD token decryption failed (%s) — removing corrupt file", type(dec_exc).__name__)
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

        # Fallback: try Claude Code's own credentials (skip if user explicitly logged out)
        if self._account_id == 0 and self._tokens is None and not self._cleared:
            cc_tokens = _load_claude_code_credentials()
            if cc_tokens:
                logger.info("AuthCD: Using existing Claude Code credentials")
                self._tokens = cc_tokens

    def save_tokens(self, tokens: Dict):
        """Encrypt and save tokens to disk."""
        with self._lock:
            self._tokens = tokens
            self._cleared = False
            try:
                self._ensure_dir()
                saved = False
                try:
                    from token_encryption import save_encrypted_tokens
                    save_encrypted_tokens(tokens, self._token_file)
                    saved = True
                except ImportError:
                    logger.warning("⚠️ AuthCD: encryption module unavailable; falling back to UNENCRYPTED JSON storage")
                except Exception as enc_exc:
                    logger.warning("⚠️ AuthCD token encryption failed (%s) — saving as plain JSON", type(enc_exc).__name__)
                if not saved:
                    with open(self._token_file, "w", encoding="utf-8") as f:
                        json.dump(tokens, f, indent=2)
                    logger.warning("🔓 AuthCD: credentials saved WITHOUT ENCRYPTION (plain JSON fallback)")
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
            self._cleared = True  # prevent fallback from re-importing
            try:
                if os.path.isfile(self._token_file):
                    os.remove(self._token_file)
                    logger.info("AuthCD tokens removed")
            except Exception as exc:
                logger.warning("Failed to remove token file: %s", exc)
        self._fire_change_callbacks()

    def clear_logout_flag(self):
        """Allow the Claude Code fallback again after an explicit logout (user asked to log in)."""
        with self._lock:
            self._cleared = False

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

            import shutil, subprocess
            # In-app browser sign-in; the Claude Code CLI, when installed, is
            # only a fallback.
            claude_bin = shutil.which("claude")

            print("\U0001f504 AuthCD: No valid token found \u2013 opening browser for login\u2026")
            try:
                tokens = login_automatically(self, claude_bin=claude_bin)
            except (TimeoutError, subprocess.TimeoutExpired):
                raise RuntimeError("AuthCD: Login timed out (3 min).")
            return tokens["access_token"]

    @property
    def has_tokens(self) -> bool:
        tokens = self.load_tokens()
        return bool(tokens and tokens.get("access_token"))

    def account_email(self, fetch: bool = True) -> str:
        """Email of the signed-in Claude account, or "" when unknown.

        The token response of a browser sign-in names the account. Tokens
        imported from Claude Code do not, so with ``fetch`` the OAuth profile
        is asked once and the answer is saved with the tokens.
        """
        with self._lock:
            tokens = self.load_tokens()
            if not tokens:
                return ""
            email = account_email_from_tokens(tokens)
            access_token = tokens.get("access_token")
            if email or not fetch or not access_token:
                return email
            if self._email_lookup_failed_token == access_token:
                return ""
            try:
                profile = fetch_account_profile(access_token)
            except Exception as exc:
                logger.debug("AuthCD: account profile lookup failed: %s", exc)
                profile = {}
            account = profile.get("account") if isinstance(profile.get("account"), dict) else {}
            email = str(account.get("email") or account.get("email_address") or "").strip()
            if not email:
                self._email_lookup_failed_token = access_token
                return ""
            updated = dict(tokens)
            updated["_account_email"] = email
            organization = profile.get("organization")
            if isinstance(organization, dict) and organization.get("name"):
                updated["_organization_name"] = str(organization["name"])
            self.save_tokens(updated)
            return email

    @property
    def account_info(self) -> Dict:
        tokens = self.load_tokens()
        if not tokens:
            return {}
        source = tokens.get("_source", "glossarion")
        info = {"source": source}
        email = account_email_from_tokens(tokens)
        if email:
            info["email"] = email
        return info


def account_email_from_tokens(tokens: Optional[Dict]) -> str:
    """The account email saved with the tokens (token response or profile)."""
    if not isinstance(tokens, dict):
        return ""
    account = tokens.get("account") if isinstance(tokens.get("account"), dict) else {}
    return str(
        account.get("email_address") or account.get("email") or tokens.get("_account_email") or ""
    ).strip()


def fetch_account_profile(access_token: str, timeout: int = 10) -> Dict:
    """The OAuth profile of the signed-in account ({"account": {"email", ...}, "organization": {...}})."""
    resp = requests.get(
        ANTHROPIC_OAUTH_PROFILE_URL,
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "Cache-Control": "no-cache",
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    return data if isinstance(data, dict) else {}


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


def fetch_available_models(access_token: str, timeout: int = 10) -> List[str]:
    """List Anthropic models available to the existing Claude OAuth session."""
    response = requests.get(
        "https://api.anthropic.com/v1/models",
        params={"limit": 1000},
        headers={
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json",
            "anthropic-version": ANTHROPIC_API_VERSION,
            "User-Agent": _claude_code_user_agent(),
            "anthropic-beta": _CLAUDE_CODE_BETA_FLAGS,
            "x-app": "cli",
        },
        timeout=max(1, int(round(timeout))),
    )
    response.raise_for_status()
    payload = response.json()
    entries = payload.get("data", []) if isinstance(payload, dict) else []
    models: List[str] = []
    seen = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        model = str(entry.get("id") or entry.get("name") or "").strip()
        key = model.casefold()
        if model and key not in seen:
            seen.add(key)
            models.append(model)
    return models


# ===========================================================================
# Anthropic Messages API adapter
# ===========================================================================

def _convert_content_parts(content):
    """Convert OpenAI-style content (string or list of parts) to Anthropic format.

    Handles:
      - Plain string → returned as-is.
      - List with image_url parts → converted to Anthropic image blocks.
      - List with only text parts → joined into a single string.
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)

    anthropic_parts = []
    has_images = False
    for part in content:
        if not isinstance(part, dict):
            anthropic_parts.append({"type": "text", "text": str(part)})
            continue

        ptype = part.get("type", "")

        if ptype == "text":
            anthropic_parts.append({"type": "text", "text": part.get("text", "")})

        elif ptype == "image_url":
            # OpenAI format: {"type":"image_url","image_url":{"url":"data:image/png;base64,AAA..."}}
            image_url = part.get("image_url", {})
            url = image_url.get("url", "") if isinstance(image_url, dict) else str(image_url)

            if url.startswith("data:") and "base64," in url:
                # Parse data-URI: data:<media_type>;base64,<data>
                header, _, b64_data = url.partition("base64,")
                # header is e.g. "data:image/jpeg;"
                media_type = header.replace("data:", "").rstrip(";").strip()
                if not media_type:
                    media_type = "image/png"
                anthropic_parts.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": b64_data,
                    },
                })
                has_images = True
            elif url.startswith(("http://", "https://")):
                # Remote URL — use Anthropic's url source type
                anthropic_parts.append({
                    "type": "image",
                    "source": {
                        "type": "url",
                        "url": url,
                    },
                })
                has_images = True
            else:
                # Unknown URL format — pass as text fallback
                anthropic_parts.append({"type": "text", "text": f"[image: {url[:120]}]"})

        elif ptype == "image":
            # Already Anthropic-native — pass through
            anthropic_parts.append(part)
            has_images = True

        else:
            # Unknown part type — stringify
            text = part.get("text", "") or str(part)
            anthropic_parts.append({"type": "text", "text": text})

    # If no images were found, collapse to a plain string for simpler payloads
    if not has_images:
        return "\n\n".join(p.get("text", "") for p in anthropic_parts if p.get("type") == "text")

    return anthropic_parts


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
            anthropic_messages.append({"role": "assistant", "content": _convert_content_parts(content)})
        else:
            anthropic_messages.append({"role": "user", "content": _convert_content_parts(content)})

    # Merge consecutive same-role messages
    merged = []
    for msg in anthropic_messages:
        if merged and merged[-1]["role"] == msg["role"]:
            prev = merged[-1]["content"]
            cur = msg["content"]
            if isinstance(prev, str) and isinstance(cur, str):
                merged[-1]["content"] = prev + "\n\n" + cur
            elif isinstance(prev, list) or isinstance(cur, list):
                # Normalize both sides to lists for proper merging
                def _to_parts(c):
                    if isinstance(c, list):
                        return c
                    return [{"type": "text", "text": str(c)}]
                merged[-1]["content"] = _to_parts(prev) + _to_parts(cur)
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

# Models that answered HTTP 400 "`temperature` is deprecated for this model"
# during this session. They are sent without temperature from then on, as if
# "Disable temperature" were ticked for that model only.
_TEMPERATURE_REJECTED_MODELS: set = set()
_TEMPERATURE_REJECTED_LOCK = threading.Lock()


class _TemperatureRejected(RuntimeError):
    """The model refused the temperature parameter (HTTP 400)."""


class _ClientVersionRejected(RuntimeError):
    """Anthropic wants a newer Claude Code version for this model (HTTP 400)."""

    def __init__(self, message: str, required: str, detail: str):
        super().__init__(message)
        self.required = required
        self.detail = detail


def _required_client_version(status: int, detail: str) -> Optional[str]:
    """The version named by "... version 2.1.280 or newer is required", else None."""
    import re
    if status != 400:
        return None
    match = re.search(r"version\s+v?(\d+(?:\.\d+)+)\s+or\s+newer\s+is\s+required", str(detail or ""), re.I)
    return match.group(1) if match else None


def _is_temperature_rejection(status: int, detail: str) -> bool:
    text = str(detail or "").lower()
    return status == 400 and "temperature" in text and any(
        marker in text for marker in ("deprecated", "not supported", "unsupported", "not allowed")
    )


def model_rejects_temperature(model: str) -> bool:
    """Whether this session already learned that ``model`` refuses temperature."""
    with _TEMPERATURE_REJECTED_LOCK:
        return str(model or "") in _TEMPERATURE_REJECTED_MODELS


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

    Uses Authorization: Bearer with the OAuth token (not x-api-key). When the
    model refuses ``temperature``, the request is sent again without it and
    the model is remembered for the rest of the session.
    """
    kwargs = dict(
        access_token=access_token, messages=messages, model=model,
        temperature=temperature, max_tokens=max_tokens, timeout=timeout,
        base_url=base_url, log_fn=log_fn, connect_timeout=connect_timeout,
    )
    _log = log_fn or print
    include_temperature = not model_rejects_temperature(model)
    version_retried = False
    while True:
        try:
            return _send_chat_completion_once(include_temperature=include_temperature, **kwargs)
        except _TemperatureRejected:
            if not include_temperature:
                raise
            with _TEMPERATURE_REJECTED_LOCK:
                _TEMPERATURE_REJECTED_MODELS.add(str(model or ""))
            _log(
                f"🌡️ AuthCD: {model} does not accept temperature; retrying without it "
                "(kept off for this model for the rest of the session)"
            )
            include_temperature = False
        except _ClientVersionRejected as exc:
            previous = claude_code_client_version()
            if version_retried or not _adopt_client_version(exc.required):
                _log(f"❌ AuthCD HTTP 400. {exc.detail}")
                raise RuntimeError(str(exc)) from None
            version_retried = True
            _log(
                f"⬆️ AuthCD: {model} needs Claude Code {exc.required} or newer; "
                f"now identifying as {exc.required} (was {previous}) and retrying"
            )


def _send_chat_completion_once(
    access_token: str,
    messages: List[Dict],
    model: str = "claude-sonnet-4-6",
    temperature: Optional[float] = 0.7,
    max_tokens: Optional[int] = None,
    timeout: int = 600,
    base_url: Optional[str] = None,
    log_fn: Optional[Any] = None,
    connect_timeout: Optional[float] = None,
    include_temperature: bool = True,
) -> Dict:
    """One Messages API request (see :func:`send_chat_completion`)."""
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
    # Some models have deprecated the temperature parameter.
    _no_temp_models = ("claude-opus-4-7", "claude-opus-4-8", "claude-fable-5")
    if (
        include_temperature
        and temperature is not None
        and not any(m in model for m in _no_temp_models)
    ):
        body["temperature"] = temperature

    headers = {
        "Authorization": f"Bearer {access_token}",
        "anthropic-version": ANTHROPIC_API_VERSION,
        "Content-Type": "application/json",
        "Accept-Encoding": "identity",
        # Claude Code identity headers – required for OAuth token acceptance
        "User-Agent": _claude_code_user_agent(),
        "anthropic-beta": _CLAUDE_CODE_BETA_FLAGS,
        "x-app": "cli",
    }

    _log = log_fn or print
    logger.info("AuthCD: POST %s  model=%s", url, model)

    log_stream = os.getenv("LOG_STREAM_CHUNKS", "1").lower() not in ("0", "false")
    if os.getenv("BATCH_TRANSLATION", "0") == "1":
        log_stream = os.getenv("ALLOW_AUTHGPT_BATCH_STREAM_LOGS", "0").lower() not in ("0", "false")

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
                if "temperature" in body and _is_temperature_rejection(resp.status_code, detail):
                    raise _TemperatureRejected(f"AuthCD: {resp.status_code} – {detail}")
                _required = _required_client_version(resp.status_code, detail)
                if _required:
                    raise _ClientVersionRejected(
                        f"AuthCD: {resp.status_code} – {detail} [reason={reason}]", _required, detail)
                _log(f"❌ AuthCD HTTP {resp.status_code}. {detail}")
                raise RuntimeError(f"AuthCD: {resp.status_code} – {detail} [reason={reason}]")
            for line in resp.iter_lines():
                if is_cancelled():
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
        if "temperature" in body and _is_temperature_rejection(resp.status_code, detail):
            raise _TemperatureRejected(f"AuthCD: {resp.status_code} – {detail}")
        _required = _required_client_version(resp.status_code, detail)
        if _required:
            raise _ClientVersionRejected(f"AuthCD: {resp.status_code} – {detail}", _required, detail)
        _log(f"❌ AuthCD HTTP {resp.status_code}. {detail}")
        raise RuntimeError(f"AuthCD: {resp.status_code} – {detail}")

    for raw_line in resp.iter_lines(chunk_size=1):
        if is_cancelled():
            resp.close()
            raise RuntimeError("AuthCD: stream cancelled by user")
        if raw_line is None:
            continue
        line = raw_line.decode("utf-8", errors="replace") if isinstance(raw_line, bytes) else raw_line
        if _process_sse_line(line, state, _log, log_stream, t_start):
            break

    return _finalize_stream(state, _log, log_stream, t_start)
