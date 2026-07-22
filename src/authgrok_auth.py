# authgrok_auth.py - Grok OAuth authentication and Responses API adapter
"""OAuth support for Grok models through an xAI account session.

Prefix models with ``authgrok/`` (or ``authgrokN/`` for numbered account
slots) to use xAI's OAuth-backed Grok CLI proxy without an API key.

The browser is opened on xAI's authorization page.  Users may choose any
sign-in method offered there, including Google account sign-in.  The flow is
OAuth 2.0 Authorization Code + PKCE and validates the returned OIDC ID token.

The wire contract follows the public, MIT-licensed pi-xai-oauth reference:
https://github.com/BlockedPath/pi-xai-oauth
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import secrets
import threading
import time
import uuid
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlencode, urlparse

import requests

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------

_cancel_event = threading.Event()


def cancel_stream() -> None:
    """Signal active AuthGrok response streams to stop."""
    _cancel_event.set()


def reset_cancel() -> None:
    """Clear cancellation unless Glossarion currently has a hard stop active."""
    if os.environ.get("TRANSLATION_CANCELLED") == "1":
        return
    _cancel_event.clear()


def is_cancelled() -> bool:
    return _cancel_event.is_set() or os.environ.get("TRANSLATION_CANCELLED") == "1"


# ---------------------------------------------------------------------------
# xAI OAuth and proxy constants
# ---------------------------------------------------------------------------

XAI_OAUTH_ISSUER = "https://auth.x.ai"
XAI_OAUTH_DISCOVERY_URL = f"{XAI_OAUTH_ISSUER}/.well-known/openid-configuration"
XAI_OAUTH_AUTHORIZATION_URL = f"{XAI_OAUTH_ISSUER}/oauth2/authorize"
XAI_OAUTH_TOKEN_URL = f"{XAI_OAUTH_ISSUER}/oauth2/token"
XAI_OAUTH_JWKS_URL = f"{XAI_OAUTH_ISSUER}/.well-known/jwks.json"
XAI_OAUTH_CLIENT_ID = "b1a00492-073a-47ea-816f-4c329264a828"
XAI_OAUTH_SCOPE = (
    "openid profile email offline_access grok-cli:access api:access "
    "conversations:read conversations:write"
)
XAI_OAUTH_REDIRECT_HOST = "127.0.0.1"
XAI_OAUTH_REDIRECT_PORT = 56121
XAI_OAUTH_REDIRECT_PATH = "/callback"
TOKEN_REFRESH_MARGIN_SECONDS = 120

XAI_CLI_BASE_URL = "https://cli-chat-proxy.grok.com/v1"
XAI_CLI_RESPONSES_URL = f"{XAI_CLI_BASE_URL}/responses"
XAI_CLI_MODELS_URL = f"{XAI_CLI_BASE_URL}/models-v2"
DEFAULT_MODEL = "grok-4.5"

try:
    from app_version import APP_VERSION as _APP_VERSION
except Exception:  # pragma: no cover - standalone use outside Glossarion
    _APP_VERSION = "unknown"

XAI_CLIENT_IDENTIFIER = "glossarion"
XAI_CLIENT_VERSION = str(_APP_VERSION)
XAI_USER_AGENT = f"Glossarion/{XAI_CLIENT_VERSION}"

_DEFAULT_TOKEN_DIR = os.path.join(os.path.expanduser("~"), ".glossarion")
_DEFAULT_TOKEN_FILE = os.path.join(_DEFAULT_TOKEN_DIR, "authgrok_tokens.json")
_OFFICIAL_GROK_AUTH_FILE = os.path.join(os.path.expanduser("~"), ".grok", "auth.json")
_OFFICIAL_GROK_SCOPE_KEY = f"{XAI_OAUTH_ISSUER}::{XAI_OAUTH_CLIENT_ID}"
_OFFICIAL_GROK_LEGACY_SCOPE_KEY = "https://accounts.x.ai/sign-in"


# ---------------------------------------------------------------------------
# PKCE, OIDC discovery, and ID-token validation
# ---------------------------------------------------------------------------


def _b64url_decode(value: str) -> bytes:
    value = str(value or "")
    value += "=" * ((4 - len(value) % 4) % 4)
    return base64.urlsafe_b64decode(value.encode("ascii"))


def _decode_jwt_part(token: str, index: int) -> Dict[str, Any]:
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return {}
        value = json.loads(_b64url_decode(parts[index]).decode("utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def generate_pkce() -> Tuple[str, str]:
    """Return a high-entropy PKCE verifier and its S256 challenge."""
    verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).rstrip(b"=").decode("ascii")
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


def _oauth_headers() -> Dict[str, str]:
    return {
        "Accept": "application/json",
        "Content-Type": "application/x-www-form-urlencoded",
        "User-Agent": XAI_USER_AGENT,
        "X-Grok-Client-Version": XAI_CLIENT_VERSION,
        "X-Grok-Client-Surface": "ui",
    }


def _load_oidc_discovery(timeout: int = 15) -> Dict[str, Any]:
    response = requests.get(
        XAI_OAUTH_DISCOVERY_URL,
        headers={"Accept": "application/json", "User-Agent": XAI_USER_AGENT},
        timeout=timeout,
        allow_redirects=False,
    )
    if response.is_redirect or response.is_permanent_redirect:
        raise RuntimeError("xAI OIDC discovery unexpectedly redirected")
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise RuntimeError("xAI OIDC discovery returned an invalid document")

    pinned = {
        "issuer": XAI_OAUTH_ISSUER,
        "authorization_endpoint": XAI_OAUTH_AUTHORIZATION_URL,
        "token_endpoint": XAI_OAUTH_TOKEN_URL,
        "jwks_uri": XAI_OAUTH_JWKS_URL,
    }
    for key, expected in pinned.items():
        if data.get(key) != expected:
            raise RuntimeError(f"xAI OIDC discovery advertised an untrusted {key}")
    if "ES256" not in (data.get("id_token_signing_alg_values_supported") or []):
        raise RuntimeError("xAI OIDC discovery did not advertise ES256 ID tokens")
    return data


def build_auth_url(
    code_challenge: str,
    state: str,
    redirect_uri: str,
    nonce: str,
    authorization_endpoint: str = XAI_OAUTH_AUTHORIZATION_URL,
) -> str:
    """Build the xAI authorization URL for the browser PKCE flow."""
    if authorization_endpoint != XAI_OAUTH_AUTHORIZATION_URL:
        raise RuntimeError("Refusing to use an untrusted xAI authorization endpoint")
    params = {
        "response_type": "code",
        "client_id": XAI_OAUTH_CLIENT_ID,
        "redirect_uri": redirect_uri,
        "scope": XAI_OAUTH_SCOPE,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "state": state,
        "nonce": nonce,
    }
    return f"{authorization_endpoint}?{urlencode(params)}"


def _validate_id_token(
    id_token: str,
    nonce: str,
    discovery: Dict[str, Any],
    timeout: int = 15,
) -> Dict[str, Any]:
    """Verify the fresh-login xAI ID token and return its claims."""
    header = _decode_jwt_part(id_token, 0)
    claims = _decode_jwt_part(id_token, 1)
    parts = id_token.split(".")
    if len(parts) != 3 or not header or not claims:
        raise RuntimeError("xAI token response did not include a valid signed ID token")
    if header.get("alg") != "ES256" or not isinstance(header.get("kid"), str):
        raise RuntimeError("xAI ID token used an unsupported signing policy")

    response = requests.get(
        discovery["jwks_uri"],
        headers={"Accept": "application/json", "User-Agent": XAI_USER_AGENT},
        timeout=timeout,
        allow_redirects=False,
    )
    if response.is_redirect or response.is_permanent_redirect:
        raise RuntimeError("xAI JWKS request unexpectedly redirected")
    response.raise_for_status()
    jwks = response.json()
    keys = jwks.get("keys", []) if isinstance(jwks, dict) else []
    matches = [
        key for key in keys
        if isinstance(key, dict) and key.get("kid") == header["kid"]
    ]
    if len(matches) != 1:
        raise RuntimeError("xAI ID token signing key was missing or ambiguous")
    key = matches[0]
    if (
        key.get("kty") != "EC"
        or key.get("crv") != "P-256"
        or key.get("alg", "ES256") != "ES256"
        or not key.get("x")
        or not key.get("y")
    ):
        raise RuntimeError("xAI ID token signing key did not match the ES256 policy")

    try:
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature

        public_key = ec.EllipticCurvePublicNumbers(
            int.from_bytes(_b64url_decode(key["x"]), "big"),
            int.from_bytes(_b64url_decode(key["y"]), "big"),
            ec.SECP256R1(),
        ).public_key()
        raw_signature = _b64url_decode(parts[2])
        if len(raw_signature) != 64:
            raise ValueError("invalid ES256 signature length")
        der_signature = encode_dss_signature(
            int.from_bytes(raw_signature[:32], "big"),
            int.from_bytes(raw_signature[32:], "big"),
        )
        public_key.verify(
            der_signature,
            f"{parts[0]}.{parts[1]}".encode("ascii"),
            ec.ECDSA(hashes.SHA256()),
        )
    except ImportError as exc:
        raise RuntimeError("cryptography is required to verify the xAI login") from exc
    except Exception as exc:
        raise RuntimeError("xAI ID token signature validation failed") from exc

    now = time.time()
    audience = claims.get("aud")
    audience_ok = (
        audience == XAI_OAUTH_CLIENT_ID
        or (isinstance(audience, list) and XAI_OAUTH_CLIENT_ID in audience)
    )
    if claims.get("iss") != XAI_OAUTH_ISSUER:
        raise RuntimeError("xAI ID token issuer mismatch")
    if not audience_ok or (claims.get("azp") not in (None, XAI_OAUTH_CLIENT_ID)):
        raise RuntimeError("xAI ID token audience mismatch")
    if not isinstance(claims.get("sub"), str) or not claims["sub"]:
        raise RuntimeError("xAI ID token subject was invalid")
    if not isinstance(claims.get("exp"), (int, float)) or claims["exp"] <= now:
        raise RuntimeError("xAI ID token has expired")
    if not isinstance(claims.get("iat"), (int, float)) or claims["iat"] > now + 60:
        raise RuntimeError("xAI ID token issued-at time was invalid")
    if not isinstance(claims.get("nonce"), str) or not secrets.compare_digest(claims["nonce"], nonce):
        raise RuntimeError("xAI ID token nonce mismatch")
    return claims


# ---------------------------------------------------------------------------
# Token exchange and refresh
# ---------------------------------------------------------------------------


def _post_oauth_token(payload: Dict[str, str], timeout: int = 30) -> Dict[str, Any]:
    response = requests.post(
        XAI_OAUTH_TOKEN_URL,
        data=payload,
        headers=_oauth_headers(),
        timeout=timeout,
        allow_redirects=False,
    )
    if response.is_redirect or response.is_permanent_redirect:
        raise RuntimeError("xAI token endpoint unexpectedly redirected")
    if response.status_code >= 400:
        raise RuntimeError(f"xAI token request failed with HTTP {response.status_code}")
    try:
        data = response.json()
    except Exception as exc:
        raise RuntimeError("xAI token endpoint returned invalid JSON") from exc
    if not isinstance(data, dict) or not isinstance(data.get("access_token"), str):
        raise RuntimeError("xAI token response did not include an access token")
    expires_in = data.get("expires_in", 3600)
    if not isinstance(expires_in, (int, float)) or expires_in <= 0:
        expires_in = 3600
    data["expires_at"] = time.time() + float(expires_in)
    data["token_endpoint"] = XAI_OAUTH_TOKEN_URL
    return data


def exchange_code_for_tokens(auth_code: str, code_verifier: str, redirect_uri: str) -> Dict[str, Any]:
    data = _post_oauth_token({
        "grant_type": "authorization_code",
        "code": auth_code,
        "redirect_uri": redirect_uri,
        "client_id": XAI_OAUTH_CLIENT_ID,
        "code_verifier": code_verifier,
    })
    if not data.get("refresh_token"):
        raise RuntimeError("xAI token response did not include a refresh token")
    if not data.get("id_token"):
        raise RuntimeError("xAI token response did not include an ID token")
    return data


def refresh_access_token(refresh_token: str) -> Dict[str, Any]:
    """Exchange an xAI refresh token for a fresh access token."""
    if not refresh_token:
        raise RuntimeError("AuthGrok refresh token is missing")
    return _post_oauth_token({
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": XAI_OAUTH_CLIENT_ID,
    })


# ---------------------------------------------------------------------------
# Browser callback server
# ---------------------------------------------------------------------------


class _OAuthCallbackHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):  # noqa: A002
        pass

    def _write_cors(self) -> None:
        origin = self.headers.get("Origin", "")
        if origin in ("https://accounts.x.ai", "https://auth.x.ai"):
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Allow-Private-Network", "true")
            self.send_header("Vary", "Origin")

    def do_OPTIONS(self):
        self.send_response(204)
        self._write_cors()
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path != XAI_OAUTH_REDIRECT_PATH:
            self.send_response(404)
            self.end_headers()
            return

        query = parse_qs(parsed.query)
        returned_state = query.get("state", [None])[0]
        expected_state = str(getattr(self.server, "_expected_state", ""))
        if (
            not isinstance(returned_state, str)
            or not secrets.compare_digest(returned_state, expected_state)
        ):
            self.send_response(400)
            self._write_cors()
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(
                b"<html><body><h1>xAI authorization state mismatch.</h1>"
                b"Return to Glossarion and try again.</body></html>"
            )
            return

        self.server._auth_code = query.get("code", [None])[0]
        self.server._returned_state = returned_state
        self.server._error = query.get("error", [None])[0]
        self.send_response(200)
        self._write_cors()
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.server._error:
            html = "<html><body><h1>xAI authorization failed.</h1>You can close this tab.</body></html>"
        else:
            html = "<html><body><h1>Grok authorization received.</h1>You can close this tab and return to Glossarion.</body></html>"
        self.wfile.write(html.encode("utf-8"))
        threading.Thread(target=self.server.shutdown, daemon=True).start()


def _create_callback_server(state: str) -> HTTPServer:
    try:
        server = HTTPServer((XAI_OAUTH_REDIRECT_HOST, XAI_OAUTH_REDIRECT_PORT), _OAuthCallbackHandler)
    except OSError:
        server = HTTPServer((XAI_OAUTH_REDIRECT_HOST, 0), _OAuthCallbackHandler)
    server._expected_state = state
    server._auth_code = None
    server._returned_state = None
    server._error = None
    return server


def run_oauth_flow(timeout: int = 300) -> Dict[str, Any]:
    """Open xAI login in the browser and return validated OAuth tokens."""
    discovery = _load_oidc_discovery()
    verifier, challenge = generate_pkce()
    state = secrets.token_urlsafe(32)
    nonce = secrets.token_urlsafe(32)
    server = _create_callback_server(state)
    redirect_uri = (
        f"http://{XAI_OAUTH_REDIRECT_HOST}:{server.server_port}"
        f"{XAI_OAUTH_REDIRECT_PATH}"
    )
    auth_url = build_auth_url(
        challenge,
        state,
        redirect_uri,
        nonce,
        discovery["authorization_endpoint"],
    )

    print("Opening browser for Grok login (Google sign-in is supported by xAI)...")
    print(f"If the browser does not open, visit:\n{auth_url}")
    webbrowser.open(auth_url)

    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    server_thread.join(timeout=timeout)
    server.shutdown()
    server.server_close()

    if server._error:
        raise RuntimeError("xAI OAuth authorization was denied or failed")
    if not server._auth_code:
        raise RuntimeError("xAI OAuth login timed out; no callback was received")
    if not secrets.compare_digest(str(server._returned_state), state):
        raise RuntimeError("xAI OAuth state mismatch")

    tokens = exchange_code_for_tokens(server._auth_code, verifier, redirect_uri)
    claims = _validate_id_token(tokens["id_token"], nonce, discovery)
    tokens["account"] = {
        "email": str(claims.get("email", "") or ""),
        "name": str(claims.get("name", "") or ""),
        "subject": str(claims.get("sub", "") or ""),
    }
    print("Grok OAuth authentication successful.")
    if tokens["account"]["email"]:
        print(f"Account: {tokens['account']['email']}")
    return tokens


# ---------------------------------------------------------------------------
# Official Grok CLI credential import
# ---------------------------------------------------------------------------


def _parse_expiry(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)) and value > 0:
        return float(value / 1000 if value > 10_000_000_000 else value)
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        numeric = float(value)
        return numeric / 1000 if numeric > 10_000_000_000 else numeric
    except ValueError:
        pass
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc).timestamp()
    except ValueError:
        return None


def load_grok_cli_credentials(auth_file: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Read, but never modify, credentials created by the official Grok CLI."""
    path = auth_file or _OFFICIAL_GROK_AUTH_FILE
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError, TypeError):
        return None

    oidc = data.get(_OFFICIAL_GROK_SCOPE_KEY) if isinstance(data, dict) else None
    if isinstance(oidc, dict):
        access = str(oidc.get("key") or oidc.get("access_token") or oidc.get("token") or "")
        if access:
            return {
                "access_token": access,
                "refresh_token": str(oidc.get("refresh_token") or oidc.get("refresh") or ""),
                "expires_at": _parse_expiry(oidc.get("expires_at")) or time.time() + 6 * 3600,
                "token_type": "Bearer",
                "token_endpoint": XAI_OAUTH_TOKEN_URL,
                "_source": "grok-cli",
            }

    legacy = data.get(_OFFICIAL_GROK_LEGACY_SCOPE_KEY) if isinstance(data, dict) else None
    if isinstance(legacy, dict):
        access = str(legacy.get("key") or legacy.get("access_token") or legacy.get("token") or "")
        if access:
            return {
                "access_token": access,
                "refresh_token": "",
                "expires_at": time.time() + 30 * 24 * 3600,
                "token_type": "Bearer",
                "_source": "grok-cli-legacy",
            }

    if isinstance(data, dict):
        access = str(data.get("access_token") or data.get("token") or "")
        if access:
            return {
                "access_token": access,
                "refresh_token": str(data.get("refresh_token") or data.get("refresh") or ""),
                "expires_at": _parse_expiry(data.get("expires_at") or data.get("expires")) or time.time() + 30 * 24 * 3600,
                "token_type": str(data.get("token_type") or "Bearer"),
                "token_endpoint": XAI_OAUTH_TOKEN_URL,
                "_source": "grok-cli",
            }
    return None


# ---------------------------------------------------------------------------
# Encrypted, account-slot-aware token store
# ---------------------------------------------------------------------------


class AuthGrokTokenStore:
    """Thread-safe AuthGrok token store backed by an encrypted JSON file."""

    def __init__(self, token_file: Optional[str] = None, account_id: int = 0):
        self._token_file = token_file or os.environ.get("AUTHGROK_TOKEN_FILE") or _DEFAULT_TOKEN_FILE
        self._account_id = int(account_id or 0)
        self._lock = threading.RLock()
        self._tokens: Optional[Dict[str, Any]] = None
        self._on_change_callbacks: List[Any] = []
        self._load_from_disk()

    def on_token_change(self, callback) -> None:
        self._on_change_callbacks.append(callback)

    def _fire_change_callbacks(self) -> None:
        for callback in self._on_change_callbacks:
            try:
                callback()
            except Exception:
                pass

    def _ensure_dir(self) -> None:
        directory = os.path.dirname(self._token_file)
        if directory:
            os.makedirs(directory, exist_ok=True)

    def _load_from_disk(self) -> None:
        try:
            if not os.path.isfile(self._token_file):
                return
            try:
                from token_encryption import load_encrypted_tokens
                self._tokens = load_encrypted_tokens(self._token_file)
            except ImportError:
                with open(self._token_file, "r", encoding="utf-8") as handle:
                    self._tokens = json.load(handle)
            except Exception as exc:
                logger.warning("AuthGrok token decryption failed (%s); removing corrupt file", exc)
                try:
                    os.remove(self._token_file)
                except OSError:
                    pass
                self._tokens = None
        except Exception as exc:
            logger.warning("Failed to load AuthGrok tokens: %s", exc)
            self._tokens = None

    def save_tokens(self, tokens: Dict[str, Any]) -> None:
        with self._lock:
            self._tokens = dict(tokens)
            try:
                self._ensure_dir()
                saved = False
                try:
                    from token_encryption import save_encrypted_tokens
                    save_encrypted_tokens(self._tokens, self._token_file)
                    saved = True
                except ImportError:
                    pass
                except Exception as exc:
                    logger.warning("AuthGrok token encryption failed (%s); saving plain JSON", exc)
                if not saved:
                    with open(self._token_file, "w", encoding="utf-8") as handle:
                        json.dump(self._tokens, handle, indent=2)
            except Exception as exc:
                logger.warning("Failed to save AuthGrok tokens: %s", exc)
        self._fire_change_callbacks()

    def load_tokens(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            if self._tokens is None:
                self._load_from_disk()
            return self._tokens

    def clear_tokens(self) -> None:
        with self._lock:
            self._tokens = None
            try:
                if os.path.isfile(self._token_file):
                    os.remove(self._token_file)
            except OSError as exc:
                logger.warning("Failed to remove AuthGrok token file: %s", exc)
        self._fire_change_callbacks()

    @staticmethod
    def _is_token_expired(tokens: Dict[str, Any]) -> bool:
        expires_at = tokens.get("expires_at", 0)
        return not isinstance(expires_at, (int, float)) or time.time() >= expires_at - TOKEN_REFRESH_MARGIN_SECONDS

    def _try_refresh(self, tokens: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        refresh_token = str(tokens.get("refresh_token") or "")
        if not refresh_token:
            return None
        try:
            refreshed = refresh_access_token(refresh_token)
            merged = {**tokens, **refreshed}
            if not refreshed.get("refresh_token"):
                merged["refresh_token"] = refresh_token
            self.save_tokens(merged)
            return merged
        except Exception as exc:
            logger.warning("AuthGrok token refresh failed: %s", exc)
            return None

    def get_valid_access_token(self, auto_login: bool = True, force_refresh: bool = False) -> str:
        """Return a usable access token, refreshing or opening login as needed."""
        with self._lock:
            tokens = self.load_tokens()

            if tokens and tokens.get("access_token") and not force_refresh and not self._is_token_expired(tokens):
                return str(tokens["access_token"])
            if tokens and tokens.get("refresh_token"):
                refreshed = self._try_refresh(tokens)
                if refreshed and refreshed.get("access_token"):
                    return str(refreshed["access_token"])

            # Default account may reuse an official Grok CLI login read-only.
            if self._account_id == 0:
                cli_tokens = load_grok_cli_credentials()
                if cli_tokens:
                    if self._is_token_expired(cli_tokens) and cli_tokens.get("refresh_token"):
                        cli_tokens = self._try_refresh(cli_tokens) or cli_tokens
                    if cli_tokens.get("access_token") and not self._is_token_expired(cli_tokens):
                        self.save_tokens(cli_tokens)
                        return str(cli_tokens["access_token"])

            env_access = os.environ.get("AUTHGROK_ACCESS_TOKEN", "").strip()
            env_refresh = os.environ.get("AUTHGROK_REFRESH_TOKEN", "").strip()
            if env_access:
                env_tokens: Dict[str, Any] = {
                    "access_token": env_access,
                    "expires_at": time.time() + 3600,
                    "_source": "environment",
                }
                if env_refresh:
                    env_tokens["refresh_token"] = env_refresh
                self.save_tokens(env_tokens)
                return env_access
            if env_refresh:
                refreshed = refresh_access_token(env_refresh)
                if not refreshed.get("refresh_token"):
                    refreshed["refresh_token"] = env_refresh
                self.save_tokens(refreshed)
                return str(refreshed["access_token"])

            if not auto_login:
                raise RuntimeError("AuthGrok has no valid token; run the Grok OAuth login first")

            is_headless = (
                os.environ.get("SPACE_ID") is not None
                or os.environ.get("HF_SPACES") == "true"
                or os.environ.get("DOCKER_CONTAINER") == "true"
                or os.environ.get("KUBERNETES_SERVICE_HOST") is not None
            )
            if is_headless:
                raise RuntimeError(
                    "AuthGrok browser login is unavailable in this headless environment. "
                    "Set AUTHGROK_ACCESS_TOKEN and optionally AUTHGROK_REFRESH_TOKEN."
                )

            tokens = run_oauth_flow()
            self.save_tokens(tokens)
            return str(tokens["access_token"])

    @property
    def has_tokens(self) -> bool:
        tokens = self.load_tokens()
        return bool(tokens and tokens.get("access_token"))

    @property
    def account_info(self) -> Dict[str, str]:
        tokens = self.load_tokens() or {}
        account = tokens.get("account") if isinstance(tokens.get("account"), dict) else {}
        claims = _decode_jwt_part(str(tokens.get("id_token") or ""), 1)
        return {
            "email": str(account.get("email") or claims.get("email") or ""),
            "name": str(account.get("name") or claims.get("name") or ""),
            "source": str(tokens.get("_source") or "glossarion"),
        }


_default_store: Optional[AuthGrokTokenStore] = None
_default_store_lock = threading.Lock()
_account_stores: Dict[int, AuthGrokTokenStore] = {}
_account_stores_lock = threading.Lock()


def get_default_store() -> AuthGrokTokenStore:
    global _default_store
    if _default_store is None:
        with _default_store_lock:
            if _default_store is None:
                _default_store = AuthGrokTokenStore()
    return _default_store


def get_store(account_id: Optional[int] = None) -> AuthGrokTokenStore:
    account_id = int(account_id or 0)
    if account_id == 0:
        return get_default_store()
    with _account_stores_lock:
        if account_id not in _account_stores:
            token_file = os.path.join(_DEFAULT_TOKEN_DIR, f"authgrok_tokens_{account_id}.json")
            _account_stores[account_id] = AuthGrokTokenStore(token_file, account_id)
        return _account_stores[account_id]


# ---------------------------------------------------------------------------
# Model discovery and Responses request construction
# ---------------------------------------------------------------------------


def _client_mode() -> str:
    if (
        os.environ.get("SPACE_ID") is not None
        or os.environ.get("DOCKER_CONTAINER") == "true"
        or os.environ.get("KUBERNETES_SERVICE_HOST") is not None
    ):
        return "headless"
    return "interactive"


def _proxy_headers(
    access_token: str,
    model: str,
    session_id: Optional[str] = None,
    request_id: Optional[str] = None,
    streaming: bool = True,
) -> Dict[str, str]:
    session_id = session_id or str(uuid.uuid4())
    request_id = request_id or str(uuid.uuid4())
    normalized_model = str(model or DEFAULT_MODEL).lower().split("/")[-1]
    return {
        "Accept": "text/event-stream" if streaming else "application/json",
        "Accept-Encoding": "identity",
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "User-Agent": XAI_USER_AGENT,
        "X-XAI-Token-Auth": "xai-grok-cli",
        "x-authenticateresponse": "authenticate-response",
        "x-grok-client-identifier": XAI_CLIENT_IDENTIFIER,
        "x-grok-client-version": XAI_CLIENT_VERSION,
        "x-grok-client-mode": _client_mode(),
        "x-grok-conv-id": session_id,
        "x-grok-req-id": request_id,
        "x-grok-model-override": normalized_model,
        "x-grok-session-id": session_id,
    }


def fetch_available_models(access_token: str, timeout: int = 15) -> List[str]:
    """Return model IDs visible to the authenticated xAI account."""
    headers = _proxy_headers(access_token, DEFAULT_MODEL, streaming=False)
    for name in ("Content-Type", "x-grok-conv-id", "x-grok-req-id", "x-grok-model-override", "x-grok-session-id"):
        headers.pop(name, None)
    response = requests.get(
        XAI_CLI_MODELS_URL,
        headers=headers,
        timeout=timeout,
        allow_redirects=False,
    )
    if response.is_redirect or response.is_permanent_redirect:
        raise RuntimeError("xAI model catalog unexpectedly redirected")
    if response.status_code >= 400:
        raise RuntimeError(f"AuthGrok model catalog failed with HTTP {response.status_code}")
    data = response.json()
    entries = data.get("models", data.get("data", [])) if isinstance(data, dict) else data
    if not isinstance(entries, list):
        return []
    result: List[str] = []
    for entry in entries:
        if isinstance(entry, str):
            model_id = entry
        elif isinstance(entry, dict):
            model_id = entry.get("id") or entry.get("model") or entry.get("name")
        else:
            model_id = None
        if isinstance(model_id, str) and model_id and model_id not in result:
            result.append(model_id)
    return result


def _text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)
    chunks: List[str] = []
    for part in content:
        if isinstance(part, str):
            chunks.append(part)
        elif isinstance(part, dict) and part.get("text") is not None:
            chunks.append(str(part.get("text") or ""))
    return "\n".join(chunks)


def _convert_content_parts(content: Any, role: str) -> List[Dict[str, Any]]:
    text_type = "output_text" if role == "assistant" else "input_text"
    if isinstance(content, str):
        return [{"type": text_type, "text": content}]
    if not isinstance(content, list):
        return [{"type": text_type, "text": str(content)}]
    result: List[Dict[str, Any]] = []
    for part in content:
        if isinstance(part, str):
            result.append({"type": text_type, "text": part})
            continue
        if not isinstance(part, dict):
            continue
        part_type = str(part.get("type") or "")
        if part_type in ("text", "input_text", "output_text"):
            result.append({"type": text_type, "text": str(part.get("text") or "")})
        elif part_type in ("image_url", "input_image") and role != "assistant":
            image = part.get("image_url")
            url = image.get("url", "") if isinstance(image, dict) else image
            if url:
                result.append({"type": "input_image", "image_url": str(url), "detail": part.get("detail", "auto")})
    return result or [{"type": text_type, "text": ""}]


def _build_responses_body(
    messages: List[Dict[str, Any]],
    model: str,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    reasoning: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    instructions: List[str] = []
    input_items: List[Dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role") or "user").lower()
        content = message.get("content", "")
        if role in ("system", "developer"):
            text = _text_from_content(content).strip()
            if text:
                instructions.append(text)
            continue
        if role not in ("user", "assistant"):
            role = "user"
        input_items.append({
            "type": "message",
            "role": role,
            "content": _convert_content_parts(content, role),
        })

    body: Dict[str, Any] = {
        "model": model or DEFAULT_MODEL,
        "input": input_items,
        "store": False,
        "stream": True,
        "include": ["reasoning.encrypted_content"],
    }
    if instructions:
        body["instructions"] = "\n\n".join(instructions)
    if max_tokens is not None:
        body["max_output_tokens"] = int(max_tokens)
    if temperature is not None:
        body["temperature"] = float(temperature)

    effort = str((reasoning or {}).get("effort") or "").strip().lower()
    if effort in ("minimal", "none"):
        effort = "low"
    if effort == "xhigh" and "multi-agent" not in str(model).lower():
        effort = "high"
    allowed_efforts = {"low", "medium", "high"}
    if "multi-agent" in str(model).lower():
        allowed_efforts.add("xhigh")
    if effort in allowed_efforts:
        body["reasoning"] = {"effort": effort}
    return body


# ---------------------------------------------------------------------------
# Responses SSE parsing
# ---------------------------------------------------------------------------


def _parse_responses_result(data: Dict[str, Any]) -> Dict[str, Any]:
    content_parts: List[str] = []
    for item in data.get("output", []) if isinstance(data.get("output"), list) else []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for part in item.get("content", []) if isinstance(item.get("content"), list) else []:
            if isinstance(part, dict) and part.get("type") == "output_text":
                content_parts.append(str(part.get("text") or ""))
    status = str(data.get("status") or "")
    finish_reason = "stop"
    if status == "incomplete":
        detail = data.get("incomplete_details") or {}
        reason = str(detail.get("reason") or "") if isinstance(detail, dict) else ""
        finish_reason = "length" if "token" in reason else reason or "incomplete"
    elif status == "failed":
        finish_reason = "error"
    raw_usage = data.get("usage") if isinstance(data.get("usage"), dict) else None
    usage = None
    if raw_usage:
        usage = {
            "prompt_tokens": raw_usage.get("input_tokens", 0),
            "completion_tokens": raw_usage.get("output_tokens", 0),
            "total_tokens": raw_usage.get("total_tokens", 0),
        }
    result: Dict[str, Any] = {
        "content": "".join(content_parts),
        "finish_reason": finish_reason,
        "conversation_id": data.get("id"),
        "message_id": data.get("id"),
        "usage": usage,
    }
    if data.get("error"):
        result["error_details"] = data["error"]
    return result


def _parse_sse_responses(raw_text: str) -> Dict[str, Any]:
    stripped = raw_text.strip()
    if stripped.startswith("{"):
        try:
            return _parse_responses_result(json.loads(stripped))
        except (ValueError, TypeError):
            pass
    deltas: List[str] = []
    last_data: Optional[Dict[str, Any]] = None
    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        try:
            data = json.loads(payload)
        except ValueError:
            continue
        if not isinstance(data, dict):
            continue
        last_data = data
        event_type = str(data.get("type") or "")
        if event_type == "response.output_text.delta":
            deltas.append(str(data.get("delta") or ""))
        elif event_type in ("response.completed", "response.incomplete"):
            response = data.get("response") if isinstance(data.get("response"), dict) else data
            result = _parse_responses_result(response)
            if not result["content"]:
                result["content"] = "".join(deltas)
            if event_type == "response.incomplete" and result["finish_reason"] == "stop":
                result["finish_reason"] = "length"
            return result
        elif event_type in ("response.failed", "error"):
            response = data.get("response") if isinstance(data.get("response"), dict) else data
            result = _parse_responses_result(response)
            result["content"] = ""
            result["finish_reason"] = "error"
            result["error_details"] = response.get("error") or data.get("error") or {"type": event_type}
            if deltas:
                result["partial_content"] = "".join(deltas)
            return result
    if deltas:
        return {
            "content": "".join(deltas),
            "finish_reason": "stop",
            "conversation_id": None,
            "message_id": None,
            "usage": None,
        }
    if last_data:
        return _parse_responses_result(last_data)
    return {
        "content": "",
        "finish_reason": "error",
        "conversation_id": None,
        "message_id": None,
        "usage": None,
    }


def _safe_error_detail(response_body: str) -> str:
    value = str(response_body or "")[:16_384]
    try:
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            error = parsed.get("error", parsed.get("detail", parsed))
            if isinstance(error, dict):
                return str(error.get("message") or error.get("type") or "request failed")[:1000]
            return str(error)[:1000]
    except ValueError:
        pass
    return value.strip()[:1000] or "request failed"


def _is_terminal_sse_line(line: str) -> bool:
    payload = line[5:].strip() if line.startswith("data:") else ""
    if payload == "[DONE]":
        return True
    if not payload:
        return False
    try:
        event = json.loads(payload)
    except ValueError:
        return False
    return isinstance(event, dict) and event.get("type") in {
        "response.completed",
        "response.incomplete",
        "response.failed",
        "error",
    }


def _append_stream_log_delta(
    text_buffer: List[str],
    delta: str,
    log_fn,
    log_stream: bool = True,
) -> None:
    """Log streamed text in readable lines without changing response content."""
    if not log_stream or not delta:
        return
    text_buffer.append(str(delta).replace("\r", ""))
    combined = "".join(text_buffer)
    for tag in ("</h1>", "</h2>", "</h3>", "</h4>", "</h5>", "</h6>", "</p>"):
        combined = combined.replace(tag, tag + "\n")
    if "\n" in combined:
        parts = combined.split("\n")
        for part in parts[:-1]:
            log_fn(part.replace("\x1f", "\\x1F"))
        text_buffer[:] = [parts[-1]]
    elif len(combined) >= 160:
        log_fn(combined.replace("\x1f", "\\x1F"))
        text_buffer.clear()


def _flush_stream_log_buffer(text_buffer: List[str], log_fn, log_stream: bool = True) -> None:
    if not log_stream or not text_buffer:
        return
    remainder = "".join(text_buffer).rstrip("\n")
    if remainder:
        log_fn(remainder.replace("\x1f", "\\x1F"))
    text_buffer.clear()


def _stream_with_httpx(
    httpx_module,
    url: str,
    body: Dict[str, Any],
    headers: Dict[str, str],
    timeout: int,
    connect_timeout: Optional[float],
    log_fn,
    log_stream: bool = True,
) -> Dict[str, Any]:
    lines: List[str] = []
    text_buffer: List[str] = []
    http_timeout = httpx_module.Timeout(timeout, connect=connect_timeout)
    with httpx_module.stream(
        "POST",
        url,
        json=body,
        headers=headers,
        timeout=http_timeout,
        follow_redirects=False,
    ) as response:
        if response.status_code >= 400:
            detail = _safe_error_detail(response.read().decode("utf-8", errors="replace"))
            raise RuntimeError(f"AuthGrok HTTP {response.status_code}: {detail}")
        for line in response.iter_lines():
            if is_cancelled():
                response.close()
                raise RuntimeError("AuthGrok stream cancelled by user")
            lines.append(line)
            if line.startswith("data:") and line[5:].strip() not in ("", "[DONE]"):
                try:
                    event = json.loads(line[5:].strip())
                except ValueError:
                    event = None
                if isinstance(event, dict) and event.get("type") == "response.output_text.delta":
                    delta = str(event.get("delta") or "")
                    _append_stream_log_delta(text_buffer, delta, log_fn, log_stream)
            if _is_terminal_sse_line(line):
                break
    _flush_stream_log_buffer(text_buffer, log_fn, log_stream)
    return _parse_sse_responses("\n".join(lines))


def _stream_with_requests(
    url: str,
    body: Dict[str, Any],
    headers: Dict[str, str],
    timeout: int,
    log_fn,
    log_stream: bool = True,
) -> Dict[str, Any]:
    response = requests.post(
        url,
        json=body,
        headers=headers,
        timeout=timeout,
        stream=True,
        allow_redirects=False,
    )
    if response.is_redirect or response.is_permanent_redirect:
        raise RuntimeError("AuthGrok Responses endpoint unexpectedly redirected")
    if response.status_code >= 400:
        raise RuntimeError(f"AuthGrok HTTP {response.status_code}: {_safe_error_detail(response.text)}")
    lines: List[str] = []
    text_buffer: List[str] = []
    for raw_line in response.iter_lines(chunk_size=1):
        if is_cancelled():
            response.close()
            raise RuntimeError("AuthGrok stream cancelled by user")
        if raw_line is None:
            continue
        line = raw_line.decode("utf-8", errors="replace") if isinstance(raw_line, bytes) else str(raw_line)
        lines.append(line)
        if line.startswith("data:") and line[5:].strip() not in ("", "[DONE]"):
            try:
                event = json.loads(line[5:].strip())
            except ValueError:
                event = None
            if isinstance(event, dict) and event.get("type") == "response.output_text.delta":
                _append_stream_log_delta(
                    text_buffer,
                    str(event.get("delta") or ""),
                    log_fn,
                    log_stream,
                )
        if _is_terminal_sse_line(line):
            break
    _flush_stream_log_buffer(text_buffer, log_fn, log_stream)
    return _parse_sse_responses("\n".join(lines))


def _stream_logging_enabled() -> bool:
    """Return whether forced AuthGrok stream chunks should be shown in the log."""
    enabled = os.getenv("LOG_STREAM_CHUNKS", "1").strip().lower() not in (
        "0", "false", "no", "off"
    )
    if os.getenv("BATCH_TRANSLATION", "0") == "1":
        enabled = os.getenv("ALLOW_AUTHGPT_BATCH_STREAM_LOGS", "0").strip().lower() not in (
            "0", "false", "no", "off"
        )
    return enabled


def send_chat_completion(
    access_token: str,
    messages: List[Dict[str, Any]],
    model: str = DEFAULT_MODEL,
    temperature: Optional[float] = 0.7,
    max_tokens: Optional[int] = None,
    timeout: int = 600,
    base_url: Optional[str] = None,
    log_fn: Optional[Any] = None,
    connect_timeout: Optional[float] = None,
    reasoning: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Send OpenAI-style messages through xAI's OAuth Responses proxy."""
    effective_base = (base_url or os.environ.get("AUTHGROK_BASE_URL") or XAI_CLI_BASE_URL).rstrip("/")
    url = effective_base if effective_base.endswith("/responses") else f"{effective_base}/responses"
    body = _build_responses_body(messages, model, temperature, max_tokens, reasoning)
    session_id = str(uuid.uuid4())
    headers = _proxy_headers(access_token, model, session_id=session_id, streaming=True)
    output = log_fn or print
    logger.info("AuthGrok: POST %s model=%s", url, model)
    reset_cancel()
    log_stream = _stream_logging_enabled()

    try:
        import httpx
        return _stream_with_httpx(
            httpx, url, body, headers, timeout, connect_timeout, output, log_stream
        )
    except ImportError:
        return _stream_with_requests(url, body, headers, timeout, output, log_stream)
