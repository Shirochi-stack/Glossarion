# authgem_auth.py - Gemini routes: the Gemini app (session cookies), AI Studio key, Vertex OAuth
#
# Routing prefixes:
#   authgem/         → the Gemini app (gemini.google.com) with the Google Login's session cookies
#   authgem-key/     → Google AI Studio via GEMINI_API_KEY (no OAuth needed)
#   authgem-vertex/  → Vertex AI via OAuth (requires GCP project w/ billing)
"""
Gemini adapters, plus the Google OAuth flow and token store authgem-vertex/ uses (the same
OAuth2 client as Gemini CLI).

Three endpoint modes:
  - Gemini app (authgem/): gemini.google.com's own requests, signed in with google.com cookies
  - AI Studio + API key: generativelanguage.googleapis.com with ?key= param
  - Vertex AI + OAuth: {region}-aiplatform.googleapis.com with Bearer token

OAuth Flow:
  1. Open browser to Google OAuth consent screen
  2. Spin up a local HTTP callback server on a random available port
  3. User logs in via browser → callback receives auth code
  4. Exchange auth code for access + refresh tokens
  5. Store tokens locally (~/.glossarion/authgem_tokens.json)
  6. Use access token as Bearer auth

run_oauth_flow() runs all of it. begin_oauth() / complete_from_redirect()
split it for callers that open the browser themselves (Glossarion Mobile)
and add a paste-the-redirect-URL fallback.
"""
import os
import re
import json
import time
import hashlib
import base64
import secrets
import uuid
import logging
import threading
from contextlib import contextmanager
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlencode, urlparse, parse_qs
from typing import Optional, Dict, List, Tuple, Any

import requests

import oauth_session

logger = logging.getLogger(__name__)

# Module-level cancellation flag — set by unified_api_client.hard_cancel_all()
_cancel_event = threading.Event()


def cancel_stream():
    """Signal any active AuthGem stream to abort immediately."""
    _cancel_event.set()
    _close_web_streams()


def reset_cancel():
    """Clear the cancellation flag (call before starting a new request).

    Refuse to clear while a hard stop is active: this is called per-attempt and
    per-chapter (reset_cleanup_state), so a just-starting worker must not wipe
    the cancel signal the Stop button set. TRANSLATION_CANCELLED is only set on
    an immediate stop (never graceful), so honoring it here is safe and matches
    _is_externally_stopped() below.
    """
    if os.environ.get("TRANSLATION_CANCELLED") == "1":
        return
    _cancel_event.clear()


def reset_verification():
    """Clear the verification-pending flag.

    Call this when the user starts a NEW translation session (not per-request).
    This allows the browser to re-open for verification on the next 403.
    """
    global _verification_pending
    _verification_pending = False


def is_cancelled() -> bool:
    # Also honor the hard-abort env var so a racing reset_cancel() can't let an
    # in-flight browser-backed request bypass Stop.
    return _cancel_event.is_set() or os.environ.get("TRANSLATION_CANCELLED") == "1"


def _is_externally_stopped() -> bool:
    """Check if any **hard** stop flag has been raised.

    Only checks signals that indicate the user wants to abort the current
    in-flight request immediately:

    - ``_cancel_event``: set by ``cancel_stream()`` / ``hard_cancel_all()``
    - ``TRANSLATION_CANCELLED``: hard-abort env var from the manga/translation engine

    NOTE: ``global_stop_flag`` and ``GRACEFUL_STOP`` are intentionally NOT
    checked here.  Those are *soft* stop signals meaning "don't start new
    work, but let the current request finish."  The outer retry loop in
    ``unified_api_client.py`` handles them via ``_should_abort_retry()``.
    """
    if _cancel_event.is_set():
        return True
    # Env-var-based hard abort set by the manga / translation engine
    if os.environ.get("TRANSLATION_CANCELLED") == "1":
        return True
    return False


_GEMINI_FINISH_REASON_MAP = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "SAFETY": "content_filter",
    "RECITATION": "content_filter",
    "FINISH_REASON_UNSPECIFIED": "stop",
    "BLOCKLIST": "safety",
    "PROHIBITED_CONTENT": "prohibited_content",
    "SPII": "safety",
    "LANGUAGE": "other",
    "OTHER": "other",
    "MALFORMED_FUNCTION_CALL": "tool_calls",
}

_GEMINI_FINISH_REASON_INT_MAP = {
    0: "stop",
    1: "stop",
    2: "length",
    3: "content_filter",
    4: "content_filter",
    5: "other",
    6: "other",
    7: "safety",
    8: "prohibited_content",
    9: "safety",
    10: "tool_calls",
}


def _candidate_finish_reason(candidate: Dict[str, Any]) -> Any:
    """Read Gemini finish reasons across Code Assist/Vertex response shapes."""
    if not isinstance(candidate, dict):
        return None
    for key in ("finishReason", "finish_reason", "finishReasonEnum", "finish_reason_enum"):
        value = candidate.get(key)
        if value not in (None, ""):
            return value
    return None


def _gemini_part_image(part: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return {"mime", "data"} for a non-thought inline image part, else None."""
    if not isinstance(part, dict) or part.get("thought"):
        return None
    inline = part.get("inlineData") or part.get("inline_data")
    if not isinstance(inline, dict) or not inline.get("data"):
        return None
    try:
        data = base64.b64decode(inline["data"])
    except Exception:
        return None
    return {"mime": inline.get("mimeType") or inline.get("mime_type") or "image/png", "data": data}


def _normalize_gemini_finish_reason(finish_reason: Any) -> str:
    """Map Gemini finish reasons to OpenAI-style values without hiding unknowns."""
    if finish_reason in (None, ""):
        return "stop"

    if isinstance(finish_reason, int) and not isinstance(finish_reason, bool):
        return _GEMINI_FINISH_REASON_INT_MAP.get(finish_reason, str(finish_reason))

    raw = str(finish_reason).strip()
    if not raw:
        return "stop"

    try:
        numeric_reason = int(raw)
    except (TypeError, ValueError):
        numeric_reason = None
    if numeric_reason is not None:
        return _GEMINI_FINISH_REASON_INT_MAP.get(numeric_reason, raw.lower())

    reason = raw.upper().split(".")[-1]
    if reason in _GEMINI_FINISH_REASON_MAP:
        return _GEMINI_FINISH_REASON_MAP[reason]

    if reason.startswith("FINISH_REASON_"):
        reason_suffix = reason[len("FINISH_REASON_"):]
        if reason_suffix in _GEMINI_FINISH_REASON_MAP:
            return _GEMINI_FINISH_REASON_MAP[reason_suffix]

    if "MAX" in reason and "TOKEN" in reason:
        return "length"
    if reason == "STOP" or reason.endswith("_STOP"):
        return "stop"
    if "PROHIBITED" in reason:
        return "prohibited_content"
    if (
        "SAFETY" in reason
        or "RECITATION" in reason
        or "BLOCK" in reason
        or "SPII" in reason
    ):
        return "content_filter"
    if "MALFORMED_FUNCTION_CALL" in reason:
        return "tool_calls"

    return raw.lower()


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
# Exact defaults verified against Google's published Gemini CLI on 2026-10-14.
# See docs/authgem_oauth_migration.md for pinned provenance and history review.
# This classification does not cover overrides or user access/refresh tokens.
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
# GOOGLE_CLOUD_LOCATION defaults to global; newer Gemini models are not always
# exposed on regional publisher-model endpoints even when older models are.
VERTEX_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
VERTEX_LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")

# ~/.glossarion; GLOSSARION_TOKEN_DIR when set (Glossarion Mobile: <data>/home/.glossarion)
_DEFAULT_TOKEN_DIR = oauth_session.default_token_dir()
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

# One-shot flag: print the maxOutputTokens cap warning only once per session
_token_cap_warned = False


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

_cached_project_id: Dict[int, Optional[str]] = {}
_project_set_by_gui: Dict[int, bool] = {}


def detect_gcp_project(access_token: str, account_id: int = 0) -> Optional[str]:
    """Auto-detect the user's GCP project ID using the Resource Manager API.

    Uses the OAuth token to list projects the user has access to.
    Only selects projects with billing enabled.  Result is cached per-account
    for the session.  The GUI dropdown can override this by setting
    ``_cached_project_id[account_id]`` directly.
    """
    global _cached_project_id
    if _cached_project_id.get(account_id):
        return _cached_project_id[account_id]

    # Check env vars first (only for default account)
    if account_id == 0:
        for env_key in ("GOOGLE_CLOUD_PROJECT", "GCLOUD_PROJECT", "GCP_PROJECT"):
            val = os.environ.get(env_key, "").strip()
            if val:
                _cached_project_id[account_id] = val
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
                        _cached_project_id[account_id] = pid
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
                        _cached_project_id[account_id] = pid
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


def list_gcp_projects(access_token: str, *, on_listed=None, http=None):
    """List the user's active GCP projects and check their billing (GCP project picker).

    Moved from the worker of the desktop ``TranslatorGUI._fetch_authgem_projects``; the
    desktop picker and the mobile Accounts screen both call it.

    1. Lists the active projects with the Resource Manager API (fast). Returns None when
       that request fails or finds no project.
    2. Calls ``on_listed(billed, unbilled, unknown)`` with every project still unknown
       (the desktop shows them at once as ❔).
    3. Checks billing for each project in parallel (Cloud Billing API, else a Vertex AI
       probe) and returns ``(billed, unbilled, unknown)`` project-id lists (completion order).

    *http* is the ``requests`` module to use (default: this module's). Errors other than a
    failed billing check propagate (the desktop logs and drops them).
    """
    _req = requests if http is None else http
    headers = {"Authorization": f"Bearer {access_token}"}

    # 1. List active projects (fast)
    resp = _req.get(
        "https://cloudresourcemanager.googleapis.com/v1/projects",
        headers=headers,
        params={"filter": "lifecycleState:ACTIVE", "pageSize": 50},
        timeout=15,
    )
    if not resp.ok:
        return None
    projects = resp.json().get("projects", [])
    if not projects:
        return None

    all_pids = [p.get("projectId", "") for p in projects if p.get("projectId")]

    # Phase 1: Show all projects immediately as unknown (❔)
    # This ensures the dropdown appears fast
    if on_listed is not None:
        on_listed([], [], list(all_pids))

    # Phase 2: Check billing in parallel, then update
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _check_billing(pid):
        """Return (pid, 'billed'|'unbilled'|'unknown')."""
        try:
            # Try Cloud Billing API first (fast, works with ADC client)
            br = _req.get(
                f"https://cloudbilling.googleapis.com/v1/projects/{pid}/billingInfo",
                headers=headers, timeout=8,
            )
            if br.ok:
                if br.json().get("billingEnabled", False):
                    return (pid, "billed")
                return (pid, "unbilled")
            # Billing API returned 403 — probe Vertex AI directly
            vr = _req.get(
                f"https://us-central1-aiplatform.googleapis.com/v1/projects/{pid}/locations/us-central1",
                headers=headers, timeout=8,
            )
            if vr.ok:
                return (pid, "billed")
            return (pid, "unbilled")
        except Exception:
            return (pid, "unknown")

    billed = []
    unbilled = []
    unknown = []
    with ThreadPoolExecutor(max_workers=min(6, len(all_pids))) as pool:
        futures = {pool.submit(_check_billing, pid): pid for pid in all_pids}
        for future in as_completed(futures):
            pid, status = future.result()
            if status == "billed":
                billed.append(pid)
            elif status == "unbilled":
                unbilled.append(pid)
            else:
                unknown.append(pid)

    # Phase 2 complete: billing info for every project
    return billed, unbilled, unknown


def authgem_project_items(billed, unbilled, unknown):
    """``(label, project id)`` items of the GCP project picker, in the desktop dropdown order.

    Moved from the desktop ``TranslatorGUI._authgem_projects_loaded`` (U7; the mobile
    Accounts project picker shows the same list): billed projects first (✅, confirmed
    usable), then unknown ones (❔, billing not readable, still selectable), then the
    unbilled ones (⚠️ ... (no billing)).
    """
    items = []
    # Add billed projects first (confirmed usable)
    for pid in billed:
        items.append((f"✅ {pid}", pid))
    # Add unknown projects (billing API inaccessible — still selectable)
    for pid in unknown:
        items.append((f"❔ {pid}", pid))
    # Add unbilled projects last (confirmed no billing)
    for pid in unbilled:
        items.append((f"⚠️ {pid} (no billing)", pid))
    return items


def choose_authgem_project_index(saved, billed, unbilled, unknown):
    """Index into ``authgem_project_items(billed, unbilled, unknown)`` the picker selects; -1 keeps none.

    The desktop ``_authgem_projects_loaded`` rule (U7): keep the saved project unless it is
    known unbilled; otherwise the first billed project, else the first unknown one.
    """
    project_ids = [pid for _label, pid in authgem_project_items(billed, unbilled, unknown)]
    selected_idx = -1
    if saved:
        # Only restore saved project if it's NOT known-unbilled;
        # prefer a billed project over a previously-saved unbilled one.
        if saved not in unbilled:
            selected_idx = project_ids.index(saved) if saved in project_ids else -1
    # If no saved or saved not found/unbilled, pick first billed; if none, first unknown
    if selected_idx < 0 and billed:
        selected_idx = 0
    elif selected_idx < 0 and unknown:
        selected_idx = len(billed)  # first unknown item
    return selected_idx


def reset_cached_project(account_id: int = 0):
    """Clear the cached GCP project ID for a specific account.

    Call this on logout or when the user switches accounts so that the
    next ``detect_gcp_project()`` call re-queries the Resource Manager API
    instead of returning the stale cached project.
    """
    global _cached_project_id, _project_set_by_gui
    _cached_project_id.pop(account_id, None)
    _project_set_by_gui.pop(account_id, None)


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

def _oauth_success_html() -> str:
    """HTML of the loopback /success page.

    When GLOSSARION_OAUTH_RETURN_URL is set (Glossarion Mobile) the page sends
    the user back to the app ('<return_url>?p=authgem'). Desktop leaves it
    unset and gets the original page.
    """
    return oauth_session.oauth_success_html(
        "authgem",
        (
            "<html><body style='font-family:sans-serif;text-align:center;padding-top:60px;"
            "background:#1a1a2e;color:#e0e0e0'>"
            "<h1 style='color:#4db8ff'>&#10004; Gemini Authenticated!</h1>"
            "<p>You can close this tab and return to Glossarion.</p>"
            "</body></html>"
        ),
        "&#10004; Gemini Authenticated!",
    )


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
                session = oauth_session.session_of(self.server)
                if session is not None:
                    session._record_callback(self.server._auth_code, self.server._returned_state, error)
                self.send_response(302)
                self.send_header("Location", SIGN_IN_FAILURE_URL)
                self.end_headers()
            else:
                code = qs.get("code", [None])[0]
                state = qs.get("state", [None])[0]
                self.server._auth_code = code
                self.server._returned_state = state
                session = oauth_session.session_of(self.server)
                if session is not None:
                    session._record_callback(code, state, self.server._error)

                # Show a success page, then redirect
                self.send_response(302)
                self.send_header("Location", f"http://{CALLBACK_HOST}:{self.server.server_port}/success")
                self.end_headers()

        elif parsed.path == "/success":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            html = _oauth_success_html()
            self.wfile.write(html.encode("utf-8"))
            # Signal the server to stop
            session = oauth_session.session_of(self.server)
            stop = session.close if session is not None else self.server.shutdown
            threading.Thread(target=stop, daemon=True).start()

        else:
            self.send_response(404)
            self.end_headers()


# ===========================================================================
# OAuth flow orchestrator
# ===========================================================================

#: A Gemini sign-in started by begin_oauth(), waiting for its redirect (shared
#: loopback session: auth_url, state, redirect_uri, wait_for_callback(), close(), ...).
OAuthSession = oauth_session.LoopbackOAuthSession


def _build_auth_url(redirect_uri: str, state: str) -> str:
    """The Google consent URL of the Gemini CLI client (no PKCE: installed-app secret)."""
    params = {
        "client_id": GOOGLE_CLIENT_ID,
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "scope": " ".join(OAUTH_SCOPES),
        "state": state,
        "access_type": "offline",
        "prompt": "consent",  # Always show consent to get refresh_token
    }
    return f"{GOOGLE_AUTH_URL}?{urlencode(params)}"


def begin_oauth(
    store: Optional["AuthGemTokenStore"] = None,
    account_id: Optional[int] = None,
    timeout: int = 300,
    persist: bool = True,
    auto_close: bool = True,
) -> OAuthSession:
    """Start a Gemini (Google) sign-in without opening a browser.

    Generates the state, starts the loopback callback server on a random
    127.0.0.1 port (``/oauth2callback``) and returns an :class:`OAuthSession`.
    The caller opens ``session.auth_url`` itself (Glossarion Mobile: Custom
    Tabs / SFSafariViewController) and then calls :func:`complete_from_redirect`.

    *store* (or ``get_store(account_id)`` when only *account_id* is given)
    receives the tokens on completion. With *persist*, the pending state and
    redirect URI are saved encrypted next to the store's token file, so a pasted
    redirect URL still completes the sign-in after the loopback server or the
    app was killed. *auto_close* stops the listener after *timeout* seconds.

    The GCP project selection (``_cached_project_id`` / ``detect_gcp_project``)
    is untouched: it stays per account, as with the desktop login.
    """
    if store is None and account_id is not None:
        store = get_store(account_id)
    if account_id is None and store is not None:
        account_id = getattr(store, "_account_id", None)

    port = _find_available_port()
    redirect_uri = f"http://{CALLBACK_HOST}:{port}{CALLBACK_PATH}"
    state = secrets.token_urlsafe(32)

    auth_url = _build_auth_url(redirect_uri, state)
    session = OAuthSession(
        auth_url, None, state, redirect_uri,
        store=store, account_id=account_id, timeout=timeout, provider="authgem",
    )

    try:
        # Start local callback server
        server = HTTPServer((CALLBACK_HOST, port), _OAuthCallbackHandler)
        server._auth_code = None
        server._returned_state = None
        server._error = None
        server.timeout = timeout
        session._add_server(server)
        session._start(auto_close)
    except BaseException:
        session.close()
        raise

    if persist and store is not None and hasattr(store, "save_pending_oauth"):
        store.save_pending_oauth(session.pending_state())
    return session


def complete_from_redirect(
    session_or_saved_state: Any = None,
    redirect_url_or_code: Optional[str] = None,
    store: Optional["AuthGemTokenStore"] = None,
) -> Dict:
    """Finish a sign-in started by :func:`begin_oauth`: check, exchange, save.

    *session_or_saved_state* is the :class:`OAuthSession`, a saved pending
    dict (``OAuthSession.pending_state()`` / ``store.load_pending_oauth()``),
    or a token store whose persisted pending sign-in is used (paste fallback
    after the loopback server or the app was killed).

    *redirect_url_or_code* is None to use what the loopback server captured
    (session only), else the pasted redirect URL
    (``http://127.0.0.1:<port>/oauth2callback?state=...&code=...``) or its
    query string. Google sign-ins have no PKCE verifier, so the state must be
    present and match.

    The code is exchanged, the account's email/name are fetched, and the tokens
    are saved to *store* (default: the session's store) with the pending state
    cleared; with no store they are only returned (run_oauth_flow's callers
    save them). Raises RuntimeError on an OAuth error, a missing code or
    state, a state mismatch or when nothing is pending.
    """
    done = oauth_session.resolve_completion(
        session_or_saved_state, redirect_url_or_code, store,
        provider="authgem", label="Gemini", require_verifier=False,
    )
    if done.error:
        raise RuntimeError(f"Google OAuth error: {done.error}")
    if not done.code:
        if done.use_loopback:
            raise RuntimeError("Google OAuth login timed out – no callback received.")
        raise RuntimeError("OAuth redirect did not contain an authorization code.")
    if not done.use_loopback and done.returned_state is None:
        raise RuntimeError(
            "Paste the whole redirect URL (it carries the sign-in state) to finish the Gemini sign-in."
        )
    if done.state_mismatch:
        raise RuntimeError("OAuth state mismatch – possible CSRF attack.")

    # Exchange code for tokens
    print("🔑 Exchanging authorization code for tokens…")
    tokens = exchange_code_for_tokens(done.code, done.pending["redirect_uri"])
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

    return oauth_session.finish_completion(done, tokens)


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

    Composes :func:`begin_oauth` (nothing persisted, no watchdog) and
    :func:`complete_from_redirect` (tokens are returned, not saved).
    """
    session = begin_oauth(timeout=timeout, persist=False, auto_close=False)
    auth_url = session.auth_url

    try:
        print(f"🔐 Opening browser for Gemini (Google) login…")
        print(f"   If the browser doesn't open, visit:\n   {auth_url}")
        webbrowser.open(auth_url)

        # Serve until callback is received or timeout
        session.wait(timeout)
    finally:
        # Cleanup
        session.close()

    return complete_from_redirect(session)


# ===========================================================================
# Token store (persistent, thread-safe) – mirrors AuthGPTTokenStore API
# ===========================================================================

class AuthGemTokenStore(oauth_session.PendingOAuthMixin):
    """Thread-safe token store backed by a JSON file.

    Also keeps the encrypted pending sign-in of begin_oauth()
    (``pending_oauth_file``, ``save/load/clear_pending_oauth``).
    """

    _pending_label = "AuthGem"

    def __init__(self, token_file: Optional[str] = None, account_id: int = 0):
        self._token_file = (
            token_file
            or os.environ.get("AUTHGEM_TOKEN_FILE")
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
                        logger.warning("🔓 AuthGem: loaded UNENCRYPTED credentials; encryption module is unavailable")
                except Exception as dec_exc:
                    # Decryption failed — file is corrupt or from a different
                    # user/machine.  Delete it so re-login creates a fresh one.
                    logger.warning("❌ AuthGem token decryption failed (%s) — removing corrupt file", type(dec_exc).__name__)
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
                    logger.warning("⚠️ AuthGem: encryption module unavailable; falling back to UNENCRYPTED JSON storage")
                except Exception as enc_exc:
                    logger.warning("⚠️ AuthGem token encryption failed (%s) — saving as plain JSON", type(enc_exc).__name__)
                if not saved:
                    # Fallback: plain JSON (still better than losing tokens)
                    with open(self._token_file, "w", encoding="utf-8") as f:
                        json.dump(tokens, f, indent=2)
                    logger.warning("🔓 AuthGem: credentials saved WITHOUT ENCRYPTION (plain JSON fallback)")
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
        # Clear cached GCP project so re-login detects the new account's project
        reset_cached_project(self._account_id)
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

# Per-account store registry (account_id → AuthGemTokenStore)
_account_stores: Dict[int, AuthGemTokenStore] = {}
_account_stores_lock = threading.Lock()


def get_default_store() -> AuthGemTokenStore:
    """Return the module-level default token store (singleton)."""
    global _default_store
    if _default_store is None:
        with _default_store_lock:
            if _default_store is None:
                _default_store = AuthGemTokenStore()
    return _default_store


def get_store(account_id: Optional[int] = None) -> AuthGemTokenStore:
    """Return the token store for a specific account slot.

    Parameters
    ----------
    account_id : int or None
        ``None`` or ``0`` returns the default store (``authgem_tokens.json``).
        Any positive integer *N* returns a dedicated store backed by
        ``authgem_tokens_N.json`` in the same directory, enabling
        multi-account usage via ``authgemN/`` model prefixes.

    Each numbered account triggers its own independent OAuth browser login
    the first time it is used, so users can authenticate with different
    Google accounts for each slot.
    """
    if account_id is None or account_id == 0:
        return get_default_store()

    with _account_stores_lock:
        if account_id in _account_stores:
            return _account_stores[account_id]

        # Build a token file path like  ~/.glossarion/authgem_tokens_2.json
        token_file = os.path.join(_DEFAULT_TOKEN_DIR, f"authgem_tokens_{account_id}.json")
        store = AuthGemTokenStore(token_file=token_file, account_id=account_id)
        _account_stores[account_id] = store
        return store


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
    image_output: Optional[Dict] = None,
) -> Dict:
    """Convert OpenAI-style messages to Gemini API generateContent format.

    Maps:
      - system → systemInstruction
      - user → user parts (text and/or images)
      - assistant → model parts

    ``image_output`` ({"size": "1K", "aspect_ratio": "auto"}) asks for an image
    reply (responseModalities TEXT+IMAGE plus imageConfig), as the native Gemini
    route does for image models; thinking is not configured for those requests.

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
        # AuthGem endpoints (Code Assist proxy & Vertex AI) hard-limit at 65536
        _AUTHGEM_MAX_OUTPUT = 65536
        if max_tokens > _AUTHGEM_MAX_OUTPUT:
            global _token_cap_warned
            if not _token_cap_warned:
                _token_cap_warned = True
                print(f"⚠️ AuthGem: Capped maxOutputTokens from {max_tokens} → {_AUTHGEM_MAX_OUTPUT}")
            max_tokens = _AUTHGEM_MAX_OUTPUT
        gen_config["maxOutputTokens"] = max_tokens

    # ── Thinking / reasoning configuration ──
    # Image models do not take thinking settings (the native route logs them as
    # "thinking not supported"), so an image request skips the thinking block.
    enable_thinking = os.getenv("ENABLE_GEMINI_THINKING", "1") == "1" and not image_output
    thinking_budget = int(os.getenv("THINKING_BUDGET", "-1"))
    thinking_level = os.getenv("GEMINI_THINKING_LEVEL", "high").lower()
    stream_thinking = os.getenv("STREAM_THINKING_LOGS", "1") not in ("0", "false")

    model_lower = model.lower() if model else ""
    from gemini_policy import uses_gemini_thinking_level, omit_gemini_legacy_parameters
    is_gemini_3 = uses_gemini_thinking_level(model_lower)

    if is_gemini_3:
        omit_gemini_legacy_parameters(gen_config, model_lower)

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

        # Include thoughts in stream so we can log them in real-time.
        # SKIP for minimal thinking (or budget=0) — at that level the API
        # tags the actual output text as thought=True parts, which causes
        # translated content to display under "🧠 Thinking..." instead of
        # as normal text output.
        _is_minimal = (thinking_level == "minimal") or (thinking_budget == 0)
        if stream_thinking and not _is_minimal:
            thinking_config["includeThoughts"] = True

        if thinking_config:
            gen_config["thinkingConfig"] = thinking_config
    elif image_output:
        gen_config["responseModalities"] = ["TEXT", "IMAGE"]
        image_config = {"imageSize": str(image_output.get("size") or "1K")}
        aspect_ratio = str(image_output.get("aspect_ratio") or "auto")
        if aspect_ratio != "auto":
            image_config["aspectRatio"] = aspect_ratio
        gen_config["imageConfig"] = image_config
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
    # Each category may have its own threshold (Other Settings › API Safety Settings).
    from gemini_policy import gemini_safety_thresholds
    body["safetySettings"] = [
        {"category": category, "threshold": threshold}
        for category, threshold in gemini_safety_thresholds("BLOCK_NONE")
    ]

    return body


# ===========================================================================
# authgem/  —  the Gemini app (gemini.google.com) on the Google sign-in's own browser session
#
# Google retired the Gemini CLI's Code Assist sign-in for personal accounts (Nov 2026:
# loadCodeAssist answers UNSUPPORTED_CLIENT for every CLI version and points to Antigravity),
# so authgem/ sends the requests gemini.google.com itself sends. It runs on the Gemini app's
# own usage limits (separate from Antigravity and the API) and offers the app's model picker
# (Auto, 4 Hydrogen, 4 Neon, and 4 Carbon on plans that include it).
#
# The requests run INSIDE a signed-in gemini.google.com page (fetch() in the page), never with
# cookies copied out of it: Google binds the session to the browser and revoked one whose
# cookies were used from Python. Desktop: a headless tab of the Google Login's Chrome profile
# (gemini_free.signed_in_tab, the profile search/gemini signs in with). Glossarion Mobile: a
# hidden page of the registered browser driver, which shares the WebView cookie jar its
# in-app Google sign-in filled. Each request's chat is deleted from the account's history after
# the reply, so translations do not fill the user's Gemini history.
# ===========================================================================

GEMINI_WEB_ORIGIN = "https://gemini.google.com"
GEMINI_WEB_APP_URL = f"{GEMINI_WEB_ORIGIN}/app?hl=en"
_GEMINI_WEB_STREAM_PATH = "/_/BardChatUi/data/assistant.lamda.BardFrontendService/StreamGenerate"
_GEMINI_WEB_BATCH_PATH = "/_/BardChatUi/data/batchexecute"
_GEMINI_WEB_UPLOAD_URL = "https://content-push.googleapis.com/upload"
_GEMINI_WEB_MODEL_HEADER = "x-goog-ext-525001261-jspb"
_GEMINI_WEB_FORM_HEADERS = {
    "Content-Type": "application/x-www-form-urlencoded;charset=utf-8",
    "X-Same-Domain": "1",
}
#: batchexecute RPC ids the Gemini app uses: its model picker, and "delete this chat".
_RPC_LIST_MODELS = "otAQ7b"
_RPC_DELETE_CHAT = "GzXR5e"
#: The model picker list is read again after this many seconds.
_GEMINI_WEB_MODELS_TTL = 600.0
#: Seconds between two reads of a request running in the page.
_GEMINI_WEB_POLL = 0.25
#: Seconds the page may take to load the Gemini app and its tokens.
_GEMINI_WEB_LOAD_TIMEOUT = 45.0
#: The browser driver owner name of authgem/'s pages (Glossarion Mobile).
WEB_BROWSER_OWNER = "authgem"
DEFAULT_WEB_MODEL = "auto"
#: The picker's models as seen on 2026-11-23: name -> (model id, label). The live picker list
#: (``_RPC_LIST_MODELS``) replaces it whenever the Gemini app answers.
GEMINI_WEB_KNOWN_MODELS: Dict[str, Tuple[str, str]] = {
    "auto": ("8c46e95b1a07cecc", "Auto"),
    "gemini-4-hydrogen": ("56fdd199312815e2", "4 Hydrogen"),
    "gemini-4-neon": ("e6fa609c3fa255c0", "4 Neon"),
    "gemini-4-carbon": ("95a4c0abcba24364", "4 Carbon"),
}
#: Gemini app error codes inside an answer frame that mean "this model's usage limit is used up".
_GEMINI_WEB_LIMIT_CODES = {1037}
_HEX16 = re.compile(r"^[0-9a-f]{16}$")
#: Source markers the Gemini app writes into answers about attached files (``[cite: 1]``).
_CITE_MARKER_RE = re.compile(r"\[cite(?:_start|_end)?(?::\s*[\d,\s]*)?\]")
_SERVED_MODEL_RE = re.compile(r'"([0-9a-f]{16})",null,null,"([^"]{1,40})",(?:true|false)')


class GeminiWebError(RuntimeError):
    """An authgem/ request failed. ``kind``: auth_error (no / ended Google session),
    config_error (bad request), model_unavailable (the plan lacks the model), api_error."""

    def __init__(self, message: str, kind: str = "api_error") -> None:
        super().__init__(message)
        self.kind = kind


SIGN_IN_NEEDED = (
    "authgem/ needs a Google sign-in. Click Google Login next to the model and sign in to the "
    "Google account whose Gemini plan you want to use."
)
SESSION_ENDED = (
    "The saved Google session ended. Click Google Login next to the model to sign in again."
)

_web_session_probe = None
_web_models_cache: Dict[str, Any] = {"models": None, "at": 0.0}
_web_models_lock = threading.Lock()


def set_web_session_probe(has_session) -> None:
    """Glossarion Mobile: ``has_session()`` answers :func:`has_web_session` (its in-app Google sign-in)."""
    global _web_session_probe
    _web_session_probe = has_session
    reset_web_session()


def _on_browser_driver() -> bool:
    """authgem/ pages come from the registered browser driver (Glossarion Mobile) instead of Chrome."""
    try:
        import browser_driver
        return browser_driver.use_driver(None)
    except Exception:
        return False


def has_web_session() -> bool:
    """Whether a Google sign-in for authgem/ was saved (a hint: every request checks the page itself)."""
    if _web_session_probe is not None:
        try:
            return bool(_web_session_probe())
        except Exception:
            return False
    try:
        import gemini_free
        return gemini_free.has_session()
    except Exception:
        return False


def reset_web_session() -> None:
    """Forget what was learned from the old session (call after a Google login or logout)."""
    with _web_models_lock:
        _web_models_cache["models"] = None
        _web_models_cache["at"] = 0.0


def web_model_name(label: Any) -> str:
    """The authgem/ name of a picker label: ``4 Neon`` -> ``gemini-4-neon``, ``Auto`` -> ``auto``."""
    text = re.sub(r"[^a-z0-9.]+", "-", str(label or "").strip().lower()).strip("-")
    if not text or text == "auto":
        return text
    return text if text.startswith("gemini-") else f"gemini-{text}"


def _web_frames(text: str):
    """The JSON arrays of a Gemini app response (``)]}'`` prefix, length lines, frames)."""
    for line in str(text or "").splitlines():
        line = line.strip()
        if not line.startswith("["):
            continue
        try:
            frame = json.loads(line)
        except ValueError:
            continue
        if isinstance(frame, list):
            yield frame


def _web_error_codes(value: Any, out: Optional[List[int]] = None) -> List[int]:
    """Integer codes nested in a frame's error part (``[..., [[..., [1037]]]]``)."""
    out = [] if out is None else out
    if isinstance(value, list):
        for item in value:
            if isinstance(item, int) and not isinstance(item, bool) and 1000 <= item < 10000:
                out.append(item)
            else:
                _web_error_codes(item, out)
    return out


# ---- the page ------------------------------------------------------------------------------

# Requests run as jobs in the page: ``fetch`` with the page's own cookies, the body read as it
# arrives into ``window.__agw.jobs[id].chunks``; Python takes the new chunks every
# ``_GEMINI_WEB_POLL`` seconds (so the live log follows the answer) and can abort a job (Stop).
_JS_START = r"""(() => {
  const A = window.__agw || (window.__agw = {seq: 0, jobs: {}});
  const id = String(++A.seq);
  const job = A.jobs[id] = {chunks: [], done: false, status: 0, error: ""};
  const ctl = new AbortController();
  job.ctl = ctl;
  const spec = __SPEC__;
  let body = spec.body;
  if (spec.file) {
    const bin = atob(spec.file.b64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    body = new FormData();
    body.append("file", new Blob([bytes], {type: spec.file.mime}), spec.file.name);
  }
  fetch(spec.url, {method: "POST", credentials: "include", headers: spec.headers, body, signal: ctl.signal})
    .then(async (r) => {
      job.status = r.status;
      const reader = r.body.getReader();
      const decoder = new TextDecoder();
      for (;;) {
        const step = await reader.read();
        if (step.done) break;
        job.chunks.push(decoder.decode(step.value, {stream: true}));
      }
      job.done = true;
    })
    .catch((e) => { job.error = String((e && e.message) || e); job.done = true; });
  return id;
})()"""

_JS_TAKE = r"""(() => {
  const A = window.__agw, job = A && A.jobs[__ID__];
  if (!job) return JSON.stringify({gone: true});
  const out = {c: job.chunks.splice(0).join(""), done: job.done, status: job.status, error: job.error};
  if (job.done) delete A.jobs[__ID__];
  return JSON.stringify(out);
})()"""

_JS_ABORT = r"""(() => { const A = window.__agw, job = A && A.jobs[__ID__];
  if (job && job.ctl) job.ctl.abort(); return true; })()"""

_JS_TOKENS = r"""(() => { const w = window.WIZ_global_data || {};
  return JSON.stringify({at: w.SNlM0e || "", bl: w.cfb2h || "", sid: w.FdrFJe || "", push: w.qKIAYe || "",
                         host: location.hostname, ready: document.readyState}); })()"""


class _WebPage:
    """A signed-in gemini.google.com page: ``run_js(expression)`` returns the expression's value,
    ``wait(ms)`` lets it run (cancellable)."""

    def __init__(self, run_js, wait) -> None:
        self._run_js = run_js
        self.wait = wait
        self.at = self.bl = self.sid = self.push_id = ""
        self._reqid = secrets.randbelow(90000) + 10000

    def js(self, expression: str) -> Any:
        value = self._run_js(expression)
        if isinstance(value, str):
            try:
                return json.loads(value)
            except ValueError:
                return value
        return value

    def load_tokens(self) -> None:
        """The page tokens the app's requests carry; auth_error when the page is signed out."""
        deadline = time.monotonic() + _GEMINI_WEB_LOAD_TIMEOUT
        info: Dict[str, Any] = {}
        while time.monotonic() < deadline:
            if is_cancelled():
                raise RuntimeError("AuthGem: stream cancelled by user")
            try:
                found = self.js(_JS_TOKENS)
            except Exception:
                found = None  # a navigation replaced the page's script context
            info = found if isinstance(found, dict) else {}
            if info.get("at"):
                break
            if info.get("ready") == "complete" and info.get("host") and info.get("host") != "gemini.google.com":
                break  # sent to accounts.google.com: signed out
            if info.get("ready") == "complete" and info.get("bl"):
                break  # the app loaded without a session token: signed out
            self.wait(250)
        if not info.get("at"):
            raise GeminiWebError(SESSION_ENDED, "auth_error")
        self.at, self.bl, self.sid = info["at"], info.get("bl", ""), info.get("sid", "")
        self.push_id = info.get("push", "")

    def query(self, **extra) -> str:
        self._reqid += 100000
        params = {"bl": self.bl, "f.sid": self.sid, "hl": "en", "_reqid": str(self._reqid), "rt": "c"}
        params.update(extra)
        return urlencode(params)

    def post(self, url: str, *, body: Optional[str] = None, headers: Optional[Dict[str, str]] = None,
             file: Optional[Dict[str, Any]] = None, on_text=None, timeout: float = 300.0) -> str:
        """POST from the page; the whole response text (``on_text(chunk)`` sees it as it arrives)."""
        spec = {"url": url, "body": body, "headers": headers or {}, "file": file}
        job = self.js(_JS_START.replace("__SPEC__", json.dumps(spec)))
        job_id = json.dumps(str(job))
        take = _JS_TAKE.replace("__ID__", job_id)
        parts: List[str] = []
        deadline = time.monotonic() + max(30.0, float(timeout or 300))
        status = 0
        try:
            while True:
                if is_cancelled():
                    raise RuntimeError("AuthGem: stream cancelled by user")
                state = self.js(take)
                if not isinstance(state, dict) or state.get("gone"):
                    raise RuntimeError("The Gemini app page lost the request")
                chunk = state.get("c") or ""
                if chunk:
                    parts.append(chunk)
                    if on_text is not None:
                        on_text(chunk)
                status = int(state.get("status") or 0)
                if state.get("done"):
                    if state.get("error") and not status:
                        raise RuntimeError(f"The Gemini app request failed: {state['error']}")
                    break
                if time.monotonic() > deadline:
                    raise RuntimeError(f"Gemini app timeout: no complete answer after {timeout:.0f}s")
                self.wait(int(_GEMINI_WEB_POLL * 1000))
        except BaseException:
            try:
                self.js(_JS_ABORT.replace("__ID__", job_id))
            except Exception:
                pass
            raise
        text = "".join(parts)
        _raise_for_web_status(status, text[:300])
        return text


@contextmanager
def _open_web_page(cancel_check=None):
    """A signed-in gemini.google.com page with its tokens loaded (see the section comment)."""
    if _on_browser_driver():
        import browser_driver
        driver = browser_driver.require_driver("authgem/")
        page = driver.open_page(owner=WEB_BROWSER_OWNER, cancel_check=cancel_check)
        try:
            page.load(GEMINI_WEB_APP_URL)
            browser_driver.wait_until_loaded(page, _GEMINI_WEB_LOAD_TIMEOUT)
            web = _WebPage(page.run_js, page.wait)
            web.load_tokens()
            yield web
        finally:
            page.close()
        return
    import gemini_free

    def wait(ms: int) -> None:
        end = time.monotonic() + max(1, int(ms)) / 1000.0
        while time.monotonic() < end:
            if is_cancelled():
                raise RuntimeError("AuthGem: stream cancelled by user")
            time.sleep(min(0.05, max(0.0, end - time.monotonic())))

    try:
        with gemini_free.signed_in_tab(GEMINI_WEB_APP_URL, cancel_check=cancel_check) as tab:
            web = _WebPage(tab.evaluate, wait)
            try:
                web.load_tokens()
            except GeminiWebError as exc:
                if exc.kind == "auth_error":
                    gemini_free.mark_session_ended()
                raise
            yield web
    except gemini_free.AIModeError as exc:
        kind = "auth_error" if exc.kind == gemini_free.KIND_SIGN_IN else exc.kind
        raise GeminiWebError(str(exc), kind) from exc


def _raise_for_web_status(status: int, body: str = "") -> None:
    if status in (401, 403):
        raise GeminiWebError(SESSION_ENDED, "auth_error")
    if status == 429:
        raise RuntimeError("429 Gemini app rate limit: too many requests")
    if status >= 400:
        raise RuntimeError(f"{status} Gemini app error: {body.strip()[:200]}")


def _web_rpc(page: _WebPage, rpcid: str, payload: Any, timeout: float = 30.0) -> Any:
    """One batchexecute call from the page; its decoded payload (None when the app sent none)."""
    body = urlencode({"at": page.at, "f.req": json.dumps([[[rpcid, json.dumps(payload), None, "generic"]]])})
    url = f"{GEMINI_WEB_ORIGIN}{_GEMINI_WEB_BATCH_PATH}?{page.query(rpcids=rpcid, **{'source-path': '/app'})}"
    text = page.post(url, body=body, headers=_GEMINI_WEB_FORM_HEADERS, timeout=timeout)
    for frame in _web_frames(text):
        for item in frame:
            if isinstance(item, list) and len(item) > 2 and item[0] == "wrb.fr" and item[1] == rpcid:
                return json.loads(item[2]) if item[2] else None
    return None


def _parse_web_models(payload: Any) -> Dict[str, Tuple[str, str]]:
    """name -> (model id, label) from the model picker RPC (entries ``[id, ..., label]``)."""
    found: Dict[str, Tuple[str, str]] = {}

    def walk(value: Any) -> None:
        if not isinstance(value, list):
            return
        entries = [item for item in value if isinstance(item, list) and len(item) > 3
                   and isinstance(item[0], str) and _HEX16.match(item[0])]
        if len(entries) >= 2:
            for entry in entries:
                label = entry[-1] if isinstance(entry[-1], str) and entry[-1].strip() else entry[2]
                name = web_model_name(label)
                if name and name not in found:
                    found[name] = (entry[0], str(label).strip())
            return
        for item in value:
            walk(item)

    walk(payload)
    return found


def _web_models(page: _WebPage) -> Dict[str, Tuple[str, str]]:
    """The picker list, cached for ``_GEMINI_WEB_MODELS_TTL``; the known list when it cannot be read."""
    with _web_models_lock:
        cached = _web_models_cache["models"]
        if cached and time.monotonic() - _web_models_cache["at"] < _GEMINI_WEB_MODELS_TTL:
            return cached
    try:
        models = _parse_web_models(_web_rpc(page, _RPC_LIST_MODELS, []))
    except GeminiWebError:
        raise
    except Exception as exc:
        logger.debug("AuthGem: model list unavailable: %s", exc)
        models = {}
    if not models:
        return dict(GEMINI_WEB_KNOWN_MODELS)
    with _web_models_lock:
        _web_models_cache["models"] = models
        _web_models_cache["at"] = time.monotonic()
    return models


def fetch_web_models(timeout: float = 30.0, cancel_check=None) -> List[str]:
    """authgem/ model names the signed-in account's Gemini picker offers."""
    with _open_web_page(cancel_check) as page:
        models = _parse_web_models(_web_rpc(page, _RPC_LIST_MODELS, [], timeout=timeout))
    if not models:
        raise GeminiWebError("The Gemini app sent no model list.", "api_error")
    with _web_models_lock:
        _web_models_cache["models"] = models
        _web_models_cache["at"] = time.monotonic()
    return list(models)


def _resolve_web_model(models: Dict[str, Tuple[str, str]], model: str) -> Tuple[str, str, str]:
    """(name, model id, label) for an authgem/ model name (empty = ``auto``)."""
    name = web_model_name(model) or DEFAULT_WEB_MODEL
    if name in models:
        return (name,) + tuple(models[name])
    if name in GEMINI_WEB_KNOWN_MODELS:
        return (name,) + GEMINI_WEB_KNOWN_MODELS[name]
    available = ", ".join(f"authgem/{key}" for key in models)
    raise GeminiWebError(f"The Gemini app has no model '{model}'. Available: {available}.", "config_error")


def _upload_web_image(page: _WebPage, image: Dict[str, Any], timeout: float) -> list:
    """Uploads one prompt image from the page; the file part of the Gemini app's request."""
    file_id = page.post(_GEMINI_WEB_UPLOAD_URL, timeout=timeout,
                        headers={"Push-ID": page.push_id or "feeds/mcudyrk2a4khkz", "X-Tenant-Id": "bard-storage"},
                        file={"b64": image["b64"], "mime": image["mime"], "name": image["name"]}).strip()
    if not file_id:
        raise RuntimeError("Gemini app image upload returned no file id")
    return [[file_id, 1], image["name"]]


class _WebStreamLog:
    """Live log of an authgem/ answer in the old authgem/ format: the thinking phase, then the answer line by
    line as it arrives.

    The Gemini app sends no thinking text: while a model thinks it sends a keep-alive frame about once a
    minute (``{"37": [n]}``) and, just before the answer, a short headline of what it worked on
    (``{"7": [..., ["Translating Korean to English"]]}``). Those become ``🧠 [authgem] Thinking...`` lines.
    ``log_stream`` / ``stream_thinking``: the streaming log toggles (LOG_STREAM_CHUNKS, the batch
    ALLOW_AUTHGPT_BATCH_STREAM_LOGS, STREAM_THINKING_LOGS)."""

    _BREAK_AFTER = ('</h1>', '</h2>', '</h3>', '</h4>', '</h5>', '</h6>', '</p>')

    def __init__(self, log, *, log_stream: bool, stream_thinking: bool) -> None:
        self.log = log
        self.log_stream = log_stream
        self.stream_thinking = stream_thinking
        self.started = time.time()
        self.thinking = False
        self.thinking_done = False
        self.beats = 0
        self.seen = ""
        self.pending = ""
        self.headlines: List[str] = []

    def _show_thinking(self) -> bool:
        return self.log_stream and self.stream_thinking

    def _begin_thinking(self) -> None:
        if not self.thinking:
            self.thinking = True
            if self._show_thinking():
                self.log("🧠 [authgem] Thinking...")

    def heartbeat(self) -> None:
        if self.thinking_done:
            return
        self.beats += 1
        self._begin_thinking()
        if self._show_thinking():
            self.log(f"    … still thinking ({time.time() - self.started:.0f}s)")

    def headline(self, text: str) -> None:
        text = str(text or "").strip()
        if not text or text in self.headlines or self.thinking_done:
            return
        self.headlines.append(text)
        self._begin_thinking()
        if self._show_thinking():
            self.log(f"    **{text}**")

    def text(self, full: str) -> None:
        if not full.startswith(self.seen):
            self.seen = full  # the app rewrote earlier text: log only what follows from here
            return
        delta = full[len(self.seen):]
        self.seen = full
        if not delta:
            return
        if self.thinking and not self.thinking_done and self._show_thinking():
            self.log(f"🧠 [authgem] Thinking complete ({time.time() - self.started:.1f}s)")
            self.log("─" * 50)
            self.log("📡 Text streaming...")
        self.thinking_done = True
        if not self.log_stream:
            return
        combined = self.pending + delta
        for tag in self._BREAK_AFTER:
            combined = combined.replace(tag, tag + "\n")
        *lines, self.pending = combined.split("\n")
        for line in lines:
            if line.strip():
                self.log(line.replace('\x1f', '\\x1F'))

    def finish(self) -> None:
        if self.log_stream and self.pending.strip():
            self.log(self.pending.strip().replace('\x1f', '\\x1F'))
        self.pending = ""


def _web_stream_log(log) -> _WebStreamLog:
    """The streaming log toggles, as the API routes of this module read them (authgem/ always streams)."""
    log_stream = os.getenv("LOG_STREAM_CHUNKS", "1").lower() not in ("0", "false")
    if os.getenv("BATCH_TRANSLATION", "0") == "1":
        log_stream = os.getenv("ALLOW_AUTHGPT_BATCH_STREAM_LOGS", "0").lower() not in ("0", "false")
    stream_thinking = os.getenv("STREAM_THINKING_LOGS", "1") not in ("0", "false")
    return _WebStreamLog(log, log_stream=log_stream, stream_thinking=stream_thinking)


class _WebAnswer:
    """Reads the answer frames of a StreamGenerate response as they arrive (``stream_log``: the live log).
    ``feed`` takes any piece of the response; whole lines are parsed."""

    def __init__(self, stream_log: Optional[_WebStreamLog] = None) -> None:
        self.text = ""
        self.chat_id = ""
        self.served = ""
        self.codes: List[int] = []
        self.stream_log = stream_log
        self._pending = ""

    def feed(self, chunk: str) -> None:
        *lines, self._pending = (self._pending + chunk).split("\n")
        for line in lines:
            self._line(line)

    def close(self) -> None:
        if self._pending:
            self._line(self._pending)
            self._pending = ""
        if self.stream_log is not None:
            self.stream_log.finish()

    def _status(self, part: Any) -> None:
        """Frames without an answer: the thinking keep-alive (key 37) and the thinking headline (key 7)."""
        log = self.stream_log
        if log is None or not isinstance(part, dict):
            return
        if "37" in part:
            log.heartbeat()
        headline = part.get("7")
        if isinstance(headline, list) and headline and isinstance(headline[-1], list) and headline[-1]:
            log.headline(headline[-1][0])

    def _line(self, line: str) -> None:
        for frame in _web_frames(line):
            for item in frame:
                if not (isinstance(item, list) and len(item) > 2 and item[0] == "wrb.fr"):
                    continue
                if not item[2]:
                    self.codes.extend(_web_error_codes(item[5:] if len(item) > 5 else []))
                    continue
                for match in _SERVED_MODEL_RE.finditer(item[2]):
                    self.served = match.group(2)
                try:
                    inner = json.loads(item[2])
                except ValueError:
                    continue
                if isinstance(inner, list) and inner and isinstance(inner[0], dict):
                    self._status(inner[0])
                    continue
                try:
                    if inner[1] and inner[1][0]:
                        self.chat_id = str(inner[1][0])
                except (IndexError, TypeError):
                    pass
                if isinstance(inner, list) and len(inner) > 2:
                    self._status(inner[2])
                try:
                    text = inner[4][0][1][0]
                except (IndexError, TypeError):
                    continue
                if isinstance(text, str) and len(text) >= len(self.text):
                    self.text = text
                    if self.stream_log is not None:
                        self.stream_log.text(text)


def _delete_web_chat(page: _WebPage, chat_id: str) -> None:
    if not chat_id:
        return
    try:
        _web_rpc(page, _RPC_DELETE_CHAT, [chat_id], timeout=20)
    except Exception as exc:
        logger.debug("AuthGem: could not delete chat %s: %s", chat_id, exc)


def _web_generate(page: _WebPage, prompt, images, model_id, timeout, _log) -> _WebAnswer:
    files = [_upload_web_image(page, image, 120) for image in images]
    inner = [[prompt, 0, None, files or None, None, None, 0], ["en"],
             ["", "", "", None, None, None, None, None, None, ""]]
    headers = dict(_GEMINI_WEB_FORM_HEADERS)
    headers[_GEMINI_WEB_MODEL_HEADER] = json.dumps([1, None, None, None, model_id, None, None, 0, [4]])
    answer = _WebAnswer(_web_stream_log(_log))
    body = urlencode({"at": page.at, "f.req": json.dumps([None, json.dumps(inner)])})
    page.post(f"{GEMINI_WEB_ORIGIN}{_GEMINI_WEB_STREAM_PATH}?{page.query()}", body=body, headers=headers,
              on_text=answer.feed, timeout=timeout)
    answer.close()
    return answer


def send_chat_completion_web(
    messages: List[Dict],
    model: str = DEFAULT_WEB_MODEL,
    timeout: int = 300,
    log_fn=None,
    connect_timeout: Optional[float] = None,
) -> Dict:
    """One authgem/ request on gemini.google.com; ``{content, finish_reason, usage, model}``.

    The chat messages become one prompt (``authds_auth.build_prompt``: a ``[role]`` header per
    message when there are several); prompt images are uploaded the way the Gemini app uploads
    them. The answer streams into the log, is checked against the model asked for (the app
    silently answers with 4 Hydrogen when the plan lacks a model) and the chat is deleted afterwards.
    """
    import authds_auth
    del connect_timeout  # the page's own connection
    if is_cancelled():
        raise RuntimeError("AuthGem: stream cancelled by user")
    _log = log_fn or (lambda *a, **kw: None)
    images: List[Dict[str, Any]] = []
    prompt = authds_auth.build_prompt(messages, error=GeminiWebError, label="AuthGem", images=images)
    started = time.time()
    with _open_web_page(cancel_check=is_cancelled) as page:
        name, model_id, label = _resolve_web_model(_web_models(page), model)
        answer = _web_generate(page, prompt, images, model_id, timeout, _log)
        _delete_web_chat(page, answer.chat_id)
    elapsed = time.time() - started
    if not answer.text:
        if set(answer.codes) & _GEMINI_WEB_LIMIT_CODES:
            raise RuntimeError(f"429 The Gemini app's usage limit for {label} is used up for now.")
        detail = f" (error code {answer.codes[0]})" if answer.codes else ""
        raise RuntimeError(f"The Gemini app sent no answer{detail}.")
    served = answer.served
    if name != DEFAULT_WEB_MODEL and served and served.casefold() != label.casefold():
        raise GeminiWebError(
            f"Gemini answered with {served} instead of {label}: this Google account's Gemini plan "
            f"does not include {label}. Choose another authgem/ model.", "model_unavailable")
    content = _CITE_MARKER_RE.sub("", answer.text)
    _log(f"📡 AuthGem: {served or label} answered in {elapsed:.1f}s ({len(content)} chars)")
    return {"content": content, "finish_reason": "stop", "usage": None, "model": served or label}


def _close_web_streams() -> None:
    """Stop: the requests running in pages see the cancel flag on their next read and abort."""
    return None


# The 403 verification helper below serves the OAuth routes (authgem-vertex/, authgem-key/).
_verification_pending = False


def _handle_403_verification(error_body: str, _log, account_id: int = 0) -> None:
    """Check a 403 error response for account verification URLs and open them.

    Only opens the browser ONCE per session — subsequent 403s log a reminder
    without spamming new tabs.
    """
    global _verification_pending

    # An auto-detected Vertex project is looked up again on the next attempt; a project the
    # user picked in the GUI stays.
    if not _verification_pending and not _project_set_by_gui.get(account_id, False):
        reset_cached_project(account_id)

    # If we already opened verification, raise immediately — don't let
    # the caller fall through to generic error handling that would retry.
    if _verification_pending:
        _log(f"⚠️ AuthGem: 403 — verification still pending. Complete it in your browser, then retry.")
        raise RuntimeError(
            "AuthGem: Account verification required. Complete it in your browser, then retry."
        )

    # Try to parse a JSON error body for a verification URL
    verification_url = None
    try:
        err_data = json.loads(error_body)
        # Pattern 1: {"error": {"details": [{"validationUrl": "..."}]}}
        for detail in err_data.get("error", {}).get("details", []):
            v_url = detail.get("validationUrl", "")
            if v_url:
                verification_url = v_url
                break
        # Pattern 2: top-level or nested validationUrl
        if not verification_url:
            verification_url = err_data.get("validationUrl", "")
        # Pattern 3: ineligibleTiers (same as loadCodeAssist response)
        if not verification_url:
            for inelig in err_data.get("ineligibleTiers", []):
                v_url = inelig.get("validationUrl", "")
                if v_url and inelig.get("reasonCode") == "VALIDATION_REQUIRED":
                    verification_url = v_url
                    break
    except (json.JSONDecodeError, AttributeError):
        pass

    # Fallback: scan raw text for common verification URL patterns
    if not verification_url:
        import re
        url_match = re.search(
            r'https://(?:accounts\.google\.com|myaccount\.google\.com|g\.co)/[^\s"\'}\]]+',
            error_body,
        )
        if url_match:
            verification_url = url_match.group(0)

    if verification_url:
        _verification_pending = True
        _log(f"⚠️ AuthGem: Account verification required")
        _log(f"🔗 Opening verification in browser…")
        logger.warning("AuthGem 403 verification URL: %s", verification_url)
        try:
            import webbrowser
            webbrowser.open(verification_url)
        except Exception:
            _log(f"🔗 Could not open browser. Visit: {verification_url}")
        # Raise immediately — don't let the retry loop keep going
        raise RuntimeError(
            "AuthGem: Account verification required. Complete it in your browser, then retry."
        )
    else:
        _log(f"⚠️ AuthGem: 403 Forbidden — account may need verification or permissions are insufficient")
        logger.warning("AuthGem 403 with no verification URL detected. Body: %s", error_body[:500])


# ===========================================================================
# Gemini CLI identification helpers — used by authgem-key/ to match the
# headers that the real Gemini CLI sends, unlocking higher rate limits.
# ===========================================================================

# Version of the Gemini CLI we impersonate (from package.json)
_GEMINI_CLI_VERSION = "0.39.0"
# Version of @google/genai SDK the CLI bundles
_GENAI_JS_VERSION = "1.30.0"

_installation_id_cache: Optional[str] = None


def _cli_platform() -> str:
    """Return a platform string matching the Gemini CLI's User-Agent format."""
    import platform
    plat = {"win32": "win32", "darwin": "darwin", "linux": "linux"}.get(
        __import__("sys").platform, __import__("sys").platform
    )
    arch = {"AMD64": "x64", "x86_64": "x64", "aarch64": "arm64",
            "arm64": "arm64"}.get(platform.machine(), platform.machine())
    return f"{plat}; {arch}"


def _get_installation_id() -> str:
    """Return a persistent installation UUID (stored in ~/.glossarion/)."""
    global _installation_id_cache
    if _installation_id_cache:
        return _installation_id_cache

    id_file = os.path.join(_DEFAULT_TOKEN_DIR, "installation_id")
    try:
        if os.path.isfile(id_file):
            with open(id_file, "r", encoding="utf-8") as f:
                _installation_id_cache = f.read().strip()
                if _installation_id_cache:
                    return _installation_id_cache
    except Exception:
        pass

    # Generate a new installation ID
    _installation_id_cache = str(uuid.uuid4())
    try:
        os.makedirs(_DEFAULT_TOKEN_DIR, exist_ok=True)
        with open(id_file, "w", encoding="utf-8") as f:
            f.write(_installation_id_cache)
    except Exception:
        pass
    return _installation_id_cache


def _build_cli_headers(api_key: str, model: str) -> Dict[str, str]:
    """Build HTTP headers matching the Gemini CLI's request format.

    The Gemini CLI (google-gemini/gemini-cli) sends specific identifying
    headers that Google's backend uses to apply higher rate limits
    (e.g. 250 RPD vs 20 RPD for bare API key calls).
    """
    return {
        "Content-Type": "application/json",
        "Accept-Encoding": "identity",
        "x-goog-api-key": api_key,
        "User-Agent": (
            f"GeminiCLI/{_GEMINI_CLI_VERSION}/{model} "
            f"({_cli_platform()}; terminal)"
        ),
        "x-goog-api-client": f"genai-js/{_GENAI_JS_VERSION} gl-js/20.0.0",
        "x-gemini-api-privileged-user-id": _get_installation_id(),
    }


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
    image_output: Optional[Dict] = None,
) -> Dict:
    """Send a chat completion via Google AI Studio using an API key.

    Uses ``generativelanguage.googleapis.com`` with the ``x-goog-api-key``
    header (matching the Gemini CLI / @google/genai SDK behaviour).
    No OAuth login needed — the user supplies a GEMINI_API_KEY.
    """
    if is_cancelled():
        raise RuntimeError("AuthGem: stream cancelled by user")

    _log = log_fn or (lambda *a, **kw: None)
    body = _build_gemini_request_body(messages, temperature, max_tokens, model=model,
                                      image_output=image_output)

    # API key is sent via x-goog-api-key header (not ?key= query param)
    # to match the @google/genai SDK's default behaviour.
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/"
        f"models/{model}:streamGenerateContent?alt=sse"
    )
    headers = _build_cli_headers(api_key, model)

    logger.info("AuthGem-Key: POST %s  model=%s", url.split("?")[0], model)

    return _stream_gemini_common(url, body, headers, timeout, _log, connect_timeout, account_id=0)


# ===========================================================================
# Vertex AI endpoint — OAuth Bearer token + GCP project.
# Used by the  authgem-vertex/  prefix (and legacy  send_chat_completion ).
# ===========================================================================

def _vertex_hostname(location: str) -> str:
    """Global and multi-region endpoints use dedicated hostnames; normal regions
    keep the usual {region}-aiplatform.googleapis.com shape."""
    location = (location or "global").strip().lower()
    if location == "global":
        return "aiplatform.googleapis.com"
    if location in {"us", "eu"}:
        return f"aiplatform.{location}.rep.googleapis.com"
    return f"{location}-aiplatform.googleapis.com"


def fetch_vertex_models(
    access_token: str,
    timeout: int = 15,
    account_id: int = 0,
) -> List[str]:
    """Return the Gemini publisher models Vertex lists for ``VERTEX_LOCATION``.

    ``publishers.models.list`` is not project-scoped, so this works before a
    billing project is picked.  Embedding and Live (bidi-only) models are left
    out because the route only calls ``streamGenerateContent``.
    """
    del account_id  # the list is the same for every Google account
    url = f"https://{_vertex_hostname(VERTEX_LOCATION)}/v1beta1/publishers/google/models"
    headers = {"Authorization": f"Bearer {access_token}"}
    models: List[str] = []
    seen = set()
    page_token = ""
    for _page in range(20):
        params = {"pageSize": 200}
        if page_token:
            params["pageToken"] = page_token
        resp = requests.get(url, headers=headers, params=params, timeout=timeout)
        if resp.status_code != 200:
            raise RuntimeError(f"Vertex model list HTTP {resp.status_code}: {resp.text[:240]}")
        data = resp.json()
        for entry in data.get("publisherModels", []) or []:
            model = str(entry.get("name", "") if isinstance(entry, dict) else "").rsplit("/", 1)[-1].strip()
            key = model.casefold()
            if (
                not key.startswith("gemini-")
                or "embedding" in key
                or "-live" in key or key.startswith("gemini-live")
                or key in seen
            ):
                continue
            seen.add(key)
            models.append(model)
        page_token = str(data.get("nextPageToken") or "")
        if not page_token:
            break
    return models


def send_chat_completion_vertex(
    access_token: str,
    messages: List[Dict],
    model: str = "gemini-2.5-flash",
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    timeout: int = 300,
    log_fn=None,
    connect_timeout: Optional[float] = None,
    account_id: int = 0,
    image_output: Optional[Dict] = None,
) -> Dict:
    """Send a chat completion via Vertex AI using an OAuth Bearer token.

    Uses ``{region}-aiplatform.googleapis.com`` — requires a GCP project
    with billing enabled and the Vertex AI API turned on.
    """
    if is_cancelled():
        raise RuntimeError("AuthGem: stream cancelled by user")

    _log = log_fn or (lambda *a, **kw: None)
    body = _build_gemini_request_body(messages, temperature, max_tokens, model=model,
                                      image_output=image_output)


    # Resolve project — auto-detect from the OAuth token (per-account cache)
    project = detect_gcp_project(access_token, account_id=account_id)
    location = VERTEX_LOCATION
    if not project:
        raise RuntimeError(
            "AuthGem-Vertex: Could not detect your GCP project ID.\n"
            "Set the GOOGLE_CLOUD_PROJECT environment variable, e.g.:\n"
            "  set GOOGLE_CLOUD_PROJECT=my-project-id\n"
            "You can find your project ID at https://console.cloud.google.com"
        )

    # Preview models often require 'global' location instead of a regional one.
    # Respect an explicit GOOGLE_CLOUD_LOCATION override, otherwise use the
    # broadest publisher-model endpoint.
    effective_location = (location or "global").strip().lower()
    if "preview" in model.lower() and "GOOGLE_CLOUD_LOCATION" not in os.environ:
        effective_location = "global"

    hostname = _vertex_hostname(effective_location)

    if effective_location == "global":
        url = (
            f"https://{hostname}/v1beta1/"
            f"projects/{project}/locations/global/"
            f"publishers/google/models/{model}:streamGenerateContent?alt=sse"
        )
    else:
        url = (
            f"https://{hostname}/v1beta1/"
            f"projects/{project}/locations/{effective_location}/"
            f"publishers/google/models/{model}:streamGenerateContent?alt=sse"
        )
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "Accept-Encoding": "identity",
    }

    logger.info("AuthGem-Vertex: POST %s  model=%s", url.split("?")[0], model)

    return _stream_gemini_common(url, body, headers, timeout, _log, connect_timeout, account_id=account_id)


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
    account_id: int = 0,
) -> Dict:
    """Shared streaming logic for the API Gemini endpoints (AI Studio key & Vertex)."""
    # Respect the "Enable streaming responses" toggle from other_settings.py.
    _enable_streaming = os.getenv("ENABLE_STREAMING", "1").lower() not in ("0", "false")
    log_stream = _enable_streaming and os.getenv("LOG_STREAM_CHUNKS", "1").lower() not in ("0", "false")
    if os.getenv("BATCH_TRANSLATION", "0") == "1":
        # AuthGem always streams (like AuthGPT/Antigravity), so use the
        # forced-stream batch log toggle for parity.
        log_stream = os.getenv("ALLOW_AUTHGPT_BATCH_STREAM_LOGS", "0").lower() not in ("0", "false")

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
        _safety_desc = _safety[0].get("threshold", "BLOCK_NONE")
    else:
        _safety_desc = "default"
    if log_stream:
        _log(f"⚙️ AuthGem: temperature={_temp}, thinking={_think_desc}, safety={_safety_desc}")

    # Always emit "in progress" with thread name (visible even when streaming is off)
    import threading as _threading
    _model_name = body.get("model") or body.get("request", {}).get("model") or "?"
    # Build a Gemini-style thinking suffix (level for Gemini 3, budget for 2.x)
    _think_suffix = ""
    try:
        if "thinkingLevel" in _tc:
            _think_suffix = f" (thinking level: {_tc['thinkingLevel']})"
        elif "thinkingBudget" in _tc:
            _b = _tc["thinkingBudget"]
            if _b == 0:
                _think_suffix = " (thinking disabled)"
            elif _b is None or _b < 0:
                _think_suffix = " (dynamic thinking)"
            else:
                _think_suffix = f" (thinking budget: {_b:,})"
        else:
            _think_suffix = " (dynamic thinking)"
    except Exception:
        _think_suffix = ""
    _log(f"📤 [{_threading.current_thread().name}] API call in progress{_think_suffix}")

    # Pre-flight: bail out immediately if a stop was already requested
    if _is_externally_stopped():
        raise RuntimeError("AuthGem: stream cancelled by user")

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
            # Grab the full response body so we can extract the structured error
            try:
                err_text_full = resp.text if hasattr(resp, 'text') else resp.content.decode('utf-8', errors='replace')
            except Exception:
                err_text_full = ""

            # Vertex returns a JSON envelope: {"error": {"code": 400, "message": "...",
            # "status": "INVALID_ARGUMENT", "details": [...]}}.  The message field is
            # what actually tells you which request field was rejected.
            err_message = err_text_full
            err_status = ""
            err_details = None
            try:
                _parsed = json.loads(err_text_full) if err_text_full else {}
                _err_obj = _parsed.get("error", {}) if isinstance(_parsed, dict) else {}
                err_message = _err_obj.get("message", err_text_full) or err_text_full
                err_status = _err_obj.get("status", "") or ""
                err_details = _err_obj.get("details")
            except Exception:
                pass

            # On any 4xx, dump a redacted request summary so the caller can see
            # which field Vertex is rejecting.  We avoid logging the full message
            # bodies (possibly sensitive user content) unless AUTHGEM_DEBUG_PAYLOAD=1.
            if 400 <= status < 500:
                try:
                    _inner = body.get("request", body) if isinstance(body, dict) else {}
                    _gc = _inner.get("generateContentConfig") or _inner.get("generationConfig") or {}
                    _contents = _inner.get("contents") or []
                    _tools = _inner.get("tools")
                    _tool_cfg = _inner.get("toolConfig")
                    _sys_inst = _inner.get("systemInstruction") or _inner.get("system_instruction")
                    _roles = []
                    for _c in _contents if isinstance(_contents, list) else []:
                        if isinstance(_c, dict):
                            _parts_count = len(_c.get("parts", []) or [])
                            _roles.append(f"{_c.get('role', '?')}(parts={_parts_count})")
                    _log(f"❌ AuthGem HTTP {status} ({err_status}) url={url.split('?')[0]}")
                    _log(f"   error.message: {err_message}")
                    if err_details:
                        try:
                            _log(f"   error.details: {json.dumps(err_details)[:2000]}")
                        except Exception:
                            _log(f"   error.details: {err_details!r}")
                    _log(f"   request keys: {list(_inner.keys())}")
                    _log(f"   contents roles: {_roles}")
                    try:
                        _log(f"   generationConfig: {json.dumps(_gc)[:1500]}")
                    except Exception:
                        _log(f"   generationConfig: {_gc!r}")
                    if _tools is not None:
                        _log(f"   tools: present ({len(_tools) if isinstance(_tools, list) else 'non-list'})")
                    if _tool_cfg is not None:
                        _log(f"   toolConfig: {_tool_cfg!r}")
                    if _sys_inst is not None:
                        _log(f"   systemInstruction: present")
                    if status != 429:
                        try:
                            _log(f"   FULL REQUEST BODY: {json.dumps(body)}")
                        except Exception:
                            _log(f"   FULL REQUEST BODY (repr): {body!r}")
                except Exception as _diag_exc:
                    _log(f"   (diagnostic dump failed: {_diag_exc})")

            # 403 — check for verification URL and open browser
            if status == 403:
                _handle_403_verification(err_text_full, _log, account_id=account_id)
            # Clarify misleading "No capacity" 429 — it's temporary server overload, not quota exhaustion
            if status == 429 and "No capacity" in err_text_full:
                raise RuntimeError("AuthGem HTTP 429. Server busy — no capacity for this model right now, retrying")
            # Include the full (un-truncated) Vertex error message in the exception
            # so it propagates up to the retry/UI layer instead of being cut at 500 chars.
            raise RuntimeError(f"AuthGem HTTP {status} ({err_status}). {err_message}")

        data = resp.json()

        # Unwrap Code Assist envelope
        if "response" in data and isinstance(data["response"], dict):
            data = data["response"]

        # Extract text from candidates
        text_parts = []
        thought_parts = []
        images = []
        finish_reason = "STOP"
        for candidate in data.get("candidates", []):
            fr = _candidate_finish_reason(candidate)
            if fr:
                finish_reason = fr
            for part in candidate.get("content", {}).get("parts", []):
                image = _gemini_part_image(part)
                if image:
                    images.append(image)
                    continue
                text = part.get("text", "")
                if not text:
                    continue
                is_thought = part.get("thought", False)
                if is_thought:
                    thought_parts.append(text)
                else:
                    text_parts.append(text)

        usage = data.get("usageMetadata", {})
        final_content = "".join(text_parts)
        thought_text = "".join(thought_parts) if thought_parts else None

        # Always report response summary + thinking tokens (non-streaming)
        thinking_tokens = usage.get("thoughtsTokenCount", 0) if usage else 0
        total_chars = len(final_content)
        summary = f"📡 AuthGem: Response received in {elapsed:.1f}s ({total_chars} chars, non-streaming)"
        if thinking_tokens > 0:
            summary += f" | 🧠 {thinking_tokens} thinking tokens used"
        _log(summary)

        # Log when text is empty but thoughts exist — do NOT promote
        # thought content to output, as it's typically internal reasoning
        # (not structured data).  Let downstream handle the empty response.
        if images:
            _log(f"🖼️ AuthGem: Received {len(images)} image(s) ({images[0]['mime']})")
        elif not final_content and thought_text:
            _log("⚠️ AuthGem: Text empty — response contained only thinking/reasoning, no output text.")
        elif not final_content:
            _log("⚠️ AuthGem: Empty response — no text received.")

        # Map Gemini finish reasons to OpenAI-style (same mapping as streaming path)
        mapped_finish_reason = _normalize_gemini_finish_reason(finish_reason)

        result = {
            "content": final_content,
            "finish_reason": mapped_finish_reason,
            "thought_content": thought_text,
            "usage": usage,
            "_silent_block": (not final_content and not images and mapped_finish_reason == "stop"
                              and usage.get("candidatesTokenCount", 0) == 0),
        }
        if images:
            result["images"] = images
        return result

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
        "images": [],            # Inline image parts ({"mime", "data"}) of an image reply
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
            state["_log_stream"] = True

    # Unwrap Code Assist response envelope if present
    # Code Assist wraps: {"response": {"candidates": [...]}} 
    # Direct AI Studio / Vertex use flat {"candidates": [...]}
    if "response" in data and isinstance(data["response"], dict):
        data = data["response"]

    # Extract candidates
    candidates = data.get("candidates", [])
    for candidate in candidates:
        # Update finish reason from the last candidate
        fr = _candidate_finish_reason(candidate)
        if fr:
            state["finish_reason"] = fr

        content_parts = candidate.get("content", {}).get("parts", [])
        for part in content_parts:
            image = _gemini_part_image(part)
            if image:
                state["images"].append(image)
                continue
            text = part.get("text", "")
            if not text:
                continue

            # Separate thinking/thought parts from actual output
            # Only the explicit `thought: true` boolean is authoritative.
            # `thoughtSignature` is an opaque token that can appear on ANY
            # part (including the final text part) — NOT a thought indicator.
            is_thought = part.get("thought", False)
            if is_thought:
                state["thought_parts"].append(text)
                # Log thinking in real-time — same format as gemini-grpc
                if stream_thinking and text:
                    if not state.get("_thinking_started"):
                        if log_stream:
                            _log("🧠 [authgem] Thinking...")
                        state["_thinking_started"] = True
                        state["_thinking_chunks"] = 0
                        state["_thinking_start_ts"] = time.time()
                    state["_thinking_chunks"] = state.get("_thinking_chunks", 0) + 1
                    if log_stream:
                        thought_buf = state.get("_thought_log_buf", [])
                        # Replace literal \n sequences with real newlines
                        # (API sometimes double-escapes newlines in thought parts)
                        _clean = text.replace("\\n", "\n")
                        thought_buf.append(_clean)
                        combined = "".join(thought_buf)
                        if "\n" in combined:
                            parts_t = combined.split("\n")
                            for p in parts_t[:-1]:
                                # Add visual separator before **bold** section headings
                                # (matches gemini-grpc formatting)
                                if p.strip().startswith("**") and state.get("_thinking_chunks", 0) > 1:
                                    _log("\u200b")
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
                if stream_thinking and log_stream:
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
                        _log(p.replace('\x1f', '\\x1F'))
                    state["log_buf"] = [parts[-1]]
                else:
                    log_buf.append(text)
                    if len("".join(log_buf)) > 150:
                        print("".join(log_buf).replace('\x1f', '\\x1F'), end="", flush=True)
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
        if remainder and stream_thinking and log_stream:
            for p in remainder.split("\n"):
                _log(f"    {p}")
    # Log thinking completion summary (only if not already logged during transition)
    if state.get("_thinking_started") and state["thought_parts"] and not state.get("_thinking_ended"):
        chunks = state.get("_thinking_chunks", len(state["thought_parts"]))
        dur = time.time() - state.get("_thinking_start_ts", t_start)
        if stream_thinking and log_stream:
            _log(f"🧠 [authgem] Thinking complete ({chunks} chunks, {dur:.1f}s)")

    # Flush remaining output log buffer
    if log_stream and state["log_buf"]:
        remainder = "".join(state["log_buf"]).strip()
        if remainder:
            _log(remainder.replace('\x1f', '\\x1F'))

    t_total = time.time() - t_start

    # Infer thinking when the endpoint doesn't stream thought parts
    # (Code Assist strips thought content from the SSE stream).
    # Always log the count — including 0 — so the user can see when a
    # request skipped thinking (e.g. thinking disabled / minimal level).
    if not state.get("_thinking_started") and not state["thought_parts"]:
        ttft = state.get("_ttft", 0)
        um = state["usage_metadata"]
        thinking_tokens = um.get("thoughtsTokenCount", 0) if um else 0
        if log_stream:
            _log(f"🧠 [authgem] Model used {thinking_tokens} thinking tokens (TTFT {ttft:.1f}s)")

    if log_stream:
        _log(f"📡 AuthGem: Stream finished in {t_total:.1f}s ({state['streamed_chars']} chars)")
        # Report thinking tokens used (from usageMetadata or estimation).
        # Always log — including 0 — so the user can see whether thinking
        # was actually used on this request.
        try:
            um = state.get("usage_metadata") or {}
            _thinking_tok = um.get("thoughtsTokenCount", 0) or 0
            if not _thinking_tok:
                # Fallback: estimate from total - prompt - completion
                _total = um.get("totalTokenCount", 0) or 0
                _prompt = um.get("promptTokenCount", 0) or 0
                _completion = um.get("candidatesTokenCount", 0) or 0
                _diff = _total - _prompt - _completion
                if _diff > 0:
                    _thinking_tok = _diff
            _log(f"   💭 Thinking tokens used: {_thinking_tok:,}")
        except Exception:
            pass

    content = "".join(state["text_parts"])
    thought_content = "".join(state["thought_parts"]) if state["thought_parts"] else None
    images = state.get("images") or []

    # Log when text is empty — do NOT promote thought content to output,
    # as it's typically internal reasoning (not structured data).
    # Let downstream handle the empty response (retry / error).
    if images:
        _log(f"🖼️ AuthGem: Received {len(images)} image(s) ({images[0]['mime']})")
    elif not content and thought_content:
        _log("⚠️ AuthGem: Text empty — response contained only thinking/reasoning, no output text.")
    elif not content:
        _log("⚠️ AuthGem: Empty response — no text received.")

    # Map Gemini finish reasons to OpenAI-style
    raw_fr = state["finish_reason"]
    mapped_reason = _normalize_gemini_finish_reason(raw_fr)

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

    # Detect silent content blocking: API returns finish_reason=STOP but
    # content is empty with 0 completion tokens — this is a disguised
    # prohibition, not a legitimate empty response.
    _silent_block = False
    if not content and not images and mapped_reason == "stop":
        _completion_tokens = (usage or {}).get("completion_tokens", 0)
        if _completion_tokens == 0:
            _silent_block = True
            mapped_reason = "content_filter"
            _log("   🚫 Silent content block detected (empty content + 0 completion tokens + stop) — overriding finish_reason to content_filter")

    result = {
        "content": content,
        "finish_reason": mapped_reason,
        "usage": usage,
        "_silent_block": _silent_block,
    }
    if images:
        result["images"] = images
    return result


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
            # 403 — check for verification URL and open browser
            if resp.status_code == 403:
                _handle_403_verification(error_body, _log, account_id=account_id)
            # Clarify misleading "No capacity" 429
            if resp.status_code == 429 and "No capacity" in str(summary):
                summary = f"Server busy — no capacity for this model right now, retrying"
            _log(f"❌ AuthGem HTTP {resp.status_code}. {summary}")
            raise RuntimeError(
                f"AuthGem: {resp.status_code} – {summary}"
            )

        for line in resp.iter_lines():
            if _is_externally_stopped():
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
        # 403 — check for verification URL and open browser
        if resp.status_code == 403:
            _handle_403_verification(error_body, _log, account_id=account_id)
        # Clarify misleading "No capacity" 429
        if resp.status_code == 429 and "No capacity" in str(summary):
            summary = f"Server busy — no capacity for this model right now, retrying"
        _log(f"❌ AuthGem HTTP {resp.status_code}. {summary}")
        raise RuntimeError(
            f"AuthGem: {resp.status_code} – {summary}"
        )

    for raw_line in resp.iter_lines(chunk_size=1):
        if _is_externally_stopped():
            resp.close()
            raise RuntimeError("AuthGem: stream cancelled by user")
        if raw_line is None:
            continue
        line = raw_line.decode("utf-8", errors="replace") if isinstance(raw_line, bytes) else raw_line
        if _process_gemini_sse_line(line, state, _log, log_stream, t_start, stream_thinking=stream_thinking):
            break

    return _finalize_gemini_stream(state, _log, log_stream, t_start, stream_thinking=stream_thinking)

