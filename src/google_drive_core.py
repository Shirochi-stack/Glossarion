"""Google Drive for Glossarion, shared by the desktop Library and Glossarion Mobile.

* OAuth: PKCE (:func:`make_pkce`), the authorization URL, the code exchange, access-token refresh
  and revoke against configurable endpoints (:class:`OAuthClient`, :class:`GoogleOAuth`). Mobile
  signs in natively (Android AuthorizationClient; iOS with :data:`IOS_CLIENT_ID` and a custom-scheme
  redirect, :func:`authorization_request` / :func:`callback_code`). The desktop signs in in the
  system browser with a loopback redirect on 127.0.0.1 (:class:`DesktopSignIn`, built on
  ``oauth_session.LoopbackOAuthSession``), which Google allows only for a "Desktop app" OAuth
  client: :data:`BUILTIN_DESKTOP_CLIENT_ID` / :data:`BUILTIN_DESKTOP_CLIENT_SECRET` (empty until the
  project registers one) or the user's own client (:func:`desktop_client`).
* :class:`DriveClient`: the Drive v3 REST calls the app needs - find / create folders, list a
  folder with md5 checksums, resumable upload of a new file or of new content for an existing
  file id (update in place), anyone-with-the-link permission, trash.
* :class:`GoogleDriveProvider`: "Share file via link" through Drive (the share-link provider API of
  ``share_hosting``).

The scope is ``drive.file``: the app sees only files and folders it created (or the user opened
with it), so it cannot browse the user's existing folders; a sync folder is a NAME the app creates.

Network: ``share_hosting.HttpClient`` (stdlib ``http.client``; ``requests`` / ``httpx`` are patched
by the optional HTTP log). Access tokens stay in memory; refresh tokens are the caller's to store
(encrypted). Nothing here logs a token. Pure Python 3.10, no GUI import; blocking (worker threads).
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
import secrets
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Optional
from urllib.parse import parse_qs, urlencode, urlsplit

from share_hosting import CancelToken, HttpClient, HttpResponse, ShareError, UploadResult, provider_info

__all__ = [
    "AUTH_URL",
    "BUILTIN_DESKTOP_CLIENT_ID",
    "BUILTIN_DESKTOP_CLIENT_SECRET",
    "DesktopSignIn",
    "DriveClient",
    "DriveFile",
    "DriveOAuth",
    "FOLDER_MIME",
    "GoogleDriveProvider",
    "GoogleOAuth",
    "IOS_CLIENT",
    "NO_DESKTOP_CLIENT",
    "OAuthClient",
    "SCOPE",
    "SignInResult",
    "authorization_request",
    "authorization_url",
    "callback_code",
    "desktop_client",
    "file_md5",
    "make_pkce",
]

log = logging.getLogger("glossarion.drive")

SCOPE = "https://www.googleapis.com/auth/drive.file"
AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
REVOKE_URL = "https://oauth2.googleapis.com/revoke"
API_BASE = "https://www.googleapis.com/drive/v3"
UPLOAD_BASE = "https://www.googleapis.com/upload/drive/v3"
DRIVE_FILES = API_BASE + "/files"
UPLOAD_URL = UPLOAD_BASE + "/files?uploadType=resumable&fields=id"
FOLDER_MIME = "application/vnd.google-apps.folder"
#: Bytes per resumable PUT (a multiple of 256 KiB, as Drive requires). Read at call time.
CHUNK_SIZE = 8 * 1024 * 1024
FILE_FIELDS = "id,name,mimeType,md5Checksum,size,trashed,parents"
_ID = re.compile(r"[a-zA-Z0-9_-]+")

# ---- the OAuth clients ----------------------------------------------------------------------------

#: Glossarion Mobile on iOS: the installed-app client with its custom-scheme redirect (no secret).
IOS_CLIENT_ID = "957824331511-1445nlka8d0ea1070r36109c4bqmshb2.apps.googleusercontent.com"
CALLBACK_SCHEME = "com.googleusercontent.apps." + IOS_CLIENT_ID[: -len(".apps.googleusercontent.com")]
REDIRECT_URI = CALLBACK_SCHEME + ":/oauth2redirect"

#: The desktop app's built-in Google OAuth client ("Desktop app" type, loopback redirect). Empty
#: until the project registers one; while it is empty, desktop sign-in is unavailable (the settings
#: window has no client fields; ``LibraryCloudStore.set_own_client`` remains for tests).
#: Google treats a Desktop client's secret as not confidential, so it may ship in the app.
BUILTIN_DESKTOP_CLIENT_ID = ""
BUILTIN_DESKTOP_CLIENT_SECRET = ""
#: One line for the desktop settings while no client is available.
NO_DESKTOP_CLIENT = "Google sign-in is not set up in this build of Glossarion."


@dataclass(frozen=True)
class OAuthClient:
    """A Google OAuth client and the endpoints it talks to (tests point them at a local server)."""

    client_id: str
    client_secret: str = ""
    auth_url: str = AUTH_URL
    token_url: str = TOKEN_URL
    revoke_url: str = REVOKE_URL
    scope: str = SCOPE

    @property
    def configured(self) -> bool:
        return bool(str(self.client_id or "").strip())

    def __repr__(self) -> str:  # never print the secret
        return f"OAuthClient(client_id={self.client_id!r}, secret={'set' if self.client_secret else 'none'})"


IOS_CLIENT = OAuthClient(IOS_CLIENT_ID)


def desktop_client(own_client_id: str = "", own_client_secret: str = "", *, use_own: Optional[bool] = None,
                   **endpoints: str) -> Optional[OAuthClient]:
    """The OAuth client desktop sign-in uses: the user's own when ``use_own`` (or, with ``use_own``
    None, whenever an own client ID is given), else the built-in one; None when neither is set
    (sign-in is then unavailable, see :data:`NO_DESKTOP_CLIENT`). ``endpoints``: ``auth_url`` /
    ``token_url`` / ``revoke_url`` overrides (tests)."""
    own_id = str(own_client_id or "").strip()
    own = bool(own_id) if use_own is None else bool(use_own)
    if own:
        if not own_id:
            return None
        return OAuthClient(own_id, str(own_client_secret or "").strip(), **endpoints)
    if not BUILTIN_DESKTOP_CLIENT_ID:
        return None
    return OAuthClient(BUILTIN_DESKTOP_CLIENT_ID, BUILTIN_DESKTOP_CLIENT_SECRET, **endpoints)


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def make_pkce() -> tuple:
    """``(verifier, S256 challenge)``."""
    verifier = secrets.token_urlsafe(48)
    return verifier, _b64url(hashlib.sha256(verifier.encode("ascii")).digest())


def authorization_url(client: OAuthClient, *, redirect_uri: str, challenge: str, state: str,
                      prompt: str = "consent select_account") -> str:
    params = {
        "client_id": client.client_id, "redirect_uri": redirect_uri, "response_type": "code",
        "scope": client.scope, "code_challenge": challenge, "code_challenge_method": "S256",
        "state": state, "access_type": "offline", "prompt": prompt,
    }
    return client.auth_url + "?" + urlencode(params)


def authorization_request() -> tuple:
    """Mobile iOS: ``(native call arguments, state, verifier)`` for the custom-scheme sign-in."""
    state = secrets.token_urlsafe(32)
    verifier, challenge = make_pkce()
    url = authorization_url(IOS_CLIENT, redirect_uri=REDIRECT_URI, challenge=challenge, state=state)
    return {"interactive": True, "scheme": CALLBACK_SCHEME, "url": url}, state, verifier


def callback_code(callback: str, state: str) -> str:
    """Mobile iOS: the authorization code of the custom-scheme callback (state checked)."""
    parts = urlsplit(callback)
    query = parse_qs(parts.query)
    if parts.scheme != CALLBACK_SCHEME or parts.netloc or parts.path != "/oauth2redirect":
        raise ShareError("auth", "Google returned an unexpected sign-in callback.")
    if query.get("state") != [state]:
        raise ShareError("auth", "Google sign-in could not be verified. Try connecting again.")
    if query.get("error") == ["access_denied"]:
        raise ShareError("cancelled", "Google Drive sign-in was cancelled.")
    codes = query.get("code", [])
    if len(codes) != 1:
        raise ShareError("auth", "Google did not return a Drive authorization code.")
    return codes[0]


class GoogleOAuth:
    """Code exchange, access-token refresh (cached in memory) and revoke for one OAuth client.

    Errors are ``ShareError``: ``sign_in`` when Google says the grant is no longer valid (revoked,
    expired: the user must sign in again), ``auth`` for other refusals, ``server`` for 5xx,
    ``network`` when Google could not be reached.
    """

    def __init__(self, http: Optional[HttpClient] = None, client: OAuthClient = IOS_CLIENT) -> None:
        self.http = http or HttpClient()
        self.client = client
        self._lock = threading.Lock()
        self._cached: Optional[tuple] = None

    def _post(self, url: str, params: Mapping[str, str]) -> HttpResponse:
        try:
            return self.http.request("POST", url, headers={"Content-Type": "application/x-www-form-urlencoded"},
                                     body=urlencode(dict(params)).encode("ascii"))
        except ShareError as exc:
            if exc.code == "maybe_uploaded":  # a small form post that timed out: not an upload
                raise ShareError("network", "Google's sign-in service did not answer.") from exc
            raise

    def _token(self, **params: str) -> dict:
        body = {"client_id": self.client.client_id}
        if self.client.client_secret:
            body["client_secret"] = self.client.client_secret
        body.update(params)
        result = self._post(self.client.token_url, body)
        payload = result.json()
        if result.status >= 500:
            raise ShareError("server", "Google’s sign-in service is temporarily unavailable.")
        if result.status != 200 or not isinstance(payload, dict) or not payload.get("access_token"):
            code = "sign_in" if isinstance(payload, dict) and payload.get("error") == "invalid_grant" else "auth"
            raise ShareError(code, "Google Drive needs you to connect your account again.")
        return payload

    def exchange_code(self, code: str, verifier: str, redirect_uri: str) -> dict:
        """Tokens for an authorization code: ``{"refresh_token", "access_token", "expires_in", ...}``."""
        data = self._token(grant_type="authorization_code", code=code, code_verifier=verifier,
                           redirect_uri=redirect_uri)
        if not data.get("refresh_token"):
            raise ShareError("auth", "Google did not return a Drive refresh token.")
        with self._lock:
            self._cached = (data["refresh_token"], data["access_token"],
                            time.monotonic() + max(0, int(data.get("expires_in", 3600) or 0) - 90))
        return data

    def exchange(self, callback: str, state: str, verifier: str) -> str:
        """Mobile iOS: the refresh token for a custom-scheme callback."""
        code = callback_code(callback, state)
        return self.exchange_code(code, verifier, REDIRECT_URI)["refresh_token"]

    def token(self, refresh: str, rejected: str = "") -> str:
        """A valid access token for ``refresh`` (a fresh one when ``rejected`` was refused)."""
        with self._lock:
            if self._cached:
                grant, token, expiry = self._cached
                if grant == refresh and token != rejected and expiry > time.monotonic():
                    return token
            data = self._token(grant_type="refresh_token", refresh_token=refresh)
            token = data["access_token"]
            self._cached = (refresh, token, time.monotonic() + max(0, int(data.get("expires_in", 3600) or 0) - 90))
            return token

    def forget(self) -> None:
        with self._lock:
            self._cached = None

    def revoke(self, token: str) -> bool:
        """Revoke a refresh (or access) token; True when Google confirmed. Errors propagate."""
        with self._lock:
            self._cached = None
        response = self._post(self.client.revoke_url, {"token": token})
        return response.status == 200


#: Mobile's name for the iOS client's token helper.
DriveOAuth = GoogleOAuth


# ---- desktop sign-in (loopback + PKCE in the system browser) -----------------------------------------


@dataclass(frozen=True)
class SignInResult:
    refresh_token: str
    access_token: str = ""
    expires_in: int = 0

    def __repr__(self) -> str:
        return "SignInResult(<tokens hidden>)"


_SIGNED_IN_PAGE = ("<html><head><meta charset='utf-8'></head><body style='font-family:sans-serif;"
                   "text-align:center;padding-top:60px'><h1>&#10004; Signed in to Google Drive</h1>"
                   "<p>Return to Glossarion. You can close this tab.</p></body></html>")
_FAILED_PAGE = "Google sign-in did not complete. Return to Glossarion and try again."


def _callback_handler():
    from http.server import BaseHTTPRequestHandler

    import oauth_session

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args: Any) -> None:  # never log the query (it holds the code)
            pass

        def do_GET(self) -> None:  # noqa: N802 - http.server API
            session = oauth_session.session_of(self.server)
            parts = urlsplit(self.path)
            query = parse_qs(parts.query)
            code = query.get("code", [None])[0]
            error = query.get("error", [None])[0]
            if session is None or parts.path not in ("", "/") or not (code or error):
                # a favicon, a browser preconnect or a stray request: keep waiting
                self.send_response(404)
                self.end_headers()
                return
            state = query.get("state", [None])[0]
            if state != session.state:
                code, error = None, "state"
            elif error:
                code = None
            session._record_callback(code, state, error)
            if code:
                page = oauth_session.oauth_success_html("gdrive", _SIGNED_IN_PAGE, "&#10004; Signed in")
                oauth_session.send_page_quietly(self, 200, page)
            else:
                oauth_session.send_page_quietly(self, 400, _FAILED_PAGE, "text/plain; charset=utf-8")
            threading.Thread(target=session.close, daemon=True).start()

    return Handler


class DesktopSignIn:
    """A desktop Google sign-in: open :attr:`auth_url` in the system browser, then call :meth:`wait`
    on a worker thread; it returns the tokens (store ``refresh_token`` encrypted).

    The redirect is ``http://127.0.0.1:<free port>`` (Google's loopback flow for Desktop clients),
    served by ``oauth_session``'s threaded listener; the PKCE verifier never leaves this object.
    """

    def __init__(self, client: Optional[OAuthClient], *, http: Optional[HttpClient] = None,
                 timeout: float = 300.0, host: str = "127.0.0.1") -> None:
        if client is None or not client.configured:
            raise ShareError("unsupported", NO_DESKTOP_CLIENT)
        import oauth_session

        self.client = client
        self.oauth = GoogleOAuth(http, client)
        self.timeout = float(timeout)
        self._verifier, challenge = make_pkce()
        state = secrets.token_urlsafe(32)
        server = oauth_session.make_mobile_loopback_server((host, 0), _callback_handler())
        self.redirect_uri = f"http://{host}:{server.server_address[1]}"
        self.auth_url = authorization_url(client, redirect_uri=self.redirect_uri, challenge=challenge, state=state,
                                          prompt="consent")
        self.session = oauth_session.LoopbackOAuthSession(self.auth_url, None, state, self.redirect_uri,
                                                          timeout=self.timeout, provider="gdrive")
        self.session._add_server(server)
        try:
            self.session._start(auto_close=True)
        except BaseException:
            self.session.close()
            raise

    def cancel(self) -> None:
        self.session.close()

    def wait(self, timeout: Optional[float] = None, cancel: Optional[CancelToken] = None) -> SignInResult:
        """Block until the browser comes back (or ``timeout`` / the session's own timeout / cancel)."""
        deadline = time.monotonic() + (self.timeout if timeout is None else float(timeout))
        try:
            while not self.session.callback_received:
                if cancel is not None and cancel.cancelled:
                    raise ShareError("cancelled", "Google Drive sign-in was cancelled.")
                if self.session.closed:
                    raise ShareError("cancelled", "Google Drive sign-in was cancelled or timed out.")
                if time.monotonic() >= deadline:
                    raise ShareError("cancelled", "Google Drive sign-in timed out. Try again.")
                self.session.wait_for_callback(0.2)
            if self.session.error == "access_denied":
                raise ShareError("cancelled", "Google Drive sign-in was cancelled.")
            if self.session.error == "state" or self.session.returned_state != self.session.state:
                raise ShareError("auth", "Google sign-in could not be verified. Try again.")
            if self.session.error or not self.session.auth_code:
                raise ShareError("auth", "Google sign-in did not complete. Try again.")
            data = self.oauth.exchange_code(self.session.auth_code, self._verifier, self.redirect_uri)
            return SignInResult(str(data["refresh_token"]), str(data.get("access_token") or ""),
                                int(data.get("expires_in") or 0))
        finally:
            self.session.close()


# ---- the Drive REST client ------------------------------------------------------------------------------


@dataclass(frozen=True)
class DriveFile:
    id: str
    name: str = ""
    mime: str = ""
    md5: str = ""
    size: Optional[int] = None
    trashed: bool = False
    parents: tuple = ()

    @property
    def is_folder(self) -> bool:
        return self.mime == FOLDER_MIME

    @classmethod
    def from_json(cls, data: Any) -> "DriveFile":
        if not isinstance(data, Mapping):
            raise ShareError("protocol", "Google Drive sent an unexpected answer.")
        file_id = data.get("id")
        if not isinstance(file_id, str) or not _ID.fullmatch(file_id):
            raise ShareError("protocol", "Google Drive did not return a file ID.")
        try:
            size = int(data["size"]) if data.get("size") is not None else None
        except (TypeError, ValueError):
            size = None
        parents = data.get("parents")
        return cls(file_id, str(data.get("name") or ""), str(data.get("mimeType") or ""),
                   str(data.get("md5Checksum") or "").lower(), size, bool(data.get("trashed")),
                   tuple(p for p in parents if isinstance(p, str)) if isinstance(parents, list) else ())


def file_md5(path: str, cancel: Optional[CancelToken] = None, chunk: int = 1024 * 1024) -> str:
    digest = hashlib.md5()  # noqa: S324 - compared with Drive's md5Checksum, not a security use
    with open(path, "rb") as handle:
        while True:
            if cancel is not None:
                cancel.check()
            data = handle.read(chunk)
            if not data:
                break
            digest.update(data)
    return digest.hexdigest()


def _q_literal(text: str) -> str:
    return "'" + str(text).replace("\\", "\\\\").replace("'", "\\'") + "'"


def _check_id(file_id: str) -> str:
    text = str(file_id or "")
    if not _ID.fullmatch(text):
        raise ShareError("not_found", "This Google Drive item has no valid ID.")
    return text


#: Codes after which a sync stops (the next file would fail the same way).
FATAL_CODES = frozenset({"sign_in", "auth", "cancelled", "limits", "network", "rate_limited", "server",
                         "protocol", "maybe_uploaded"})


class DriveClient:
    """Drive v3 calls for one account. ``access(rejected)`` returns a valid access token (a new one
    when ``rejected`` was refused with HTTP 401); every request is retried once after a 401.

    ``api_base`` / ``upload_base`` point the client at a local fake server in tests.
    """

    def __init__(self, access: Callable[[str], str], http: Optional[HttpClient] = None, *,
                 api_base: str = API_BASE, upload_base: str = UPLOAD_BASE, chunk_size: Optional[int] = None) -> None:
        if not callable(access):
            raise ShareError("sign_in", "Connect Google Drive first.")
        self.access = access
        self.http = http or HttpClient()
        self.api_base = api_base.rstrip("/")
        self.upload_base = upload_base.rstrip("/")
        self.chunk_size = chunk_size

    # ---- plumbing -----------------------------------------------------------------------------------

    def _request(self, method: str, url: str, cancel: Optional[CancelToken] = None, *,
                 headers: Optional[Mapping[str, str]] = None, body: Any = None, upload: bool = False) -> HttpResponse:
        cancel = cancel or CancelToken()
        cancel.check()
        token = self.access("")
        response = None
        for attempt in range(2):
            try:
                response = self.http.request(method, url, headers={**(headers or {}),
                                                                   "Authorization": f"Bearer {token}"},
                                             body=body, cancel=cancel)
            except ShareError as exc:
                if exc.code == "maybe_uploaded" and not upload:
                    raise ShareError("network", "Google Drive did not answer in time.") from exc
                raise
            if response.status != 401 or attempt:
                return response
            cancel.check()
            token = self.access(token)
        return response

    @staticmethod
    def require(response: HttpResponse, *, expected: Iterable[int] = (200, 201, 204)) -> None:
        status = response.status
        if status in tuple(expected):
            return
        data = response.json() or {}
        error = data.get("error") if isinstance(data, dict) else None
        errors = error.get("errors") or () if isinstance(error, dict) else ()
        reasons = {item.get("reason") for item in errors if isinstance(item, dict)}
        if status == 401:
            raise ShareError("auth", "Google Drive needs you to reconnect your account.", status=status)
        if "storageQuotaExceeded" in reasons:
            raise ShareError("limits", "Your Google Drive is out of storage.", status=status)
        if status == 429 or reasons.intersection({"rateLimitExceeded", "userRateLimitExceeded"}):
            raise ShareError("rate_limited", "Google Drive is limiting requests. Try again later.", status=status)
        if status == 404:
            raise ShareError("not_found", "Google Drive could not find the item.", status=status)
        if status >= 500:
            raise ShareError("server", "Google Drive is temporarily unavailable.", status=status)
        raise ShareError("protocol", f"Google Drive declined the request (HTTP {status}).", status=status)

    @staticmethod
    def _json(data: Any) -> bytes:
        return json.dumps(data).encode("utf-8")

    def _files_url(self, file_id: str = "", **params: str) -> str:
        url = self.api_base + "/files" + (f"/{_check_id(file_id)}" if file_id else "")
        return url + ("?" + urlencode(params) if params else "")

    # ---- metadata -----------------------------------------------------------------------------------

    def get(self, file_id: str, cancel: Optional[CancelToken] = None) -> Optional[DriveFile]:
        """The file's metadata; None when Drive no longer has it for this app (404)."""
        response = self._request("GET", self._files_url(file_id, fields=FILE_FIELDS), cancel)
        if response.status == 404:
            return None
        self.require(response, expected=(200,))
        return DriveFile.from_json(response.json())

    def list_children(self, folder_id: str, cancel: Optional[CancelToken] = None, *, name: Optional[str] = None,
                      folders_only: bool = False) -> list:
        """The folder's non-trashed children this app can see (all pages)."""
        terms = [f"{_q_literal(_check_id(folder_id))} in parents", "trashed = false"]
        if name is not None:
            terms.append(f"name = {_q_literal(name)}")
        if folders_only:
            terms.append(f"mimeType = {_q_literal(FOLDER_MIME)}")
        files: list = []
        page = ""
        for _ in range(1000):  # a hard stop against a looping page token
            params = {"q": " and ".join(terms), "fields": f"nextPageToken,files({FILE_FIELDS})",
                      "pageSize": "1000", "spaces": "drive"}
            if page:
                params["pageToken"] = page
            response = self._request("GET", self._files_url(**params), cancel)
            self.require(response, expected=(200,))
            payload = response.json()
            if not isinstance(payload, dict):
                raise ShareError("protocol", "Google Drive sent an unexpected answer.")
            for item in payload.get("files") or ():
                try:
                    found = DriveFile.from_json(item)
                except ShareError:
                    continue
                if not found.trashed:
                    files.append(found)
            page = str(payload.get("nextPageToken") or "")
            if not page:
                break
        return files

    def find_child(self, folder_id: str, name: str, cancel: Optional[CancelToken] = None, *,
                   folder: bool = False) -> Optional[DriveFile]:
        for item in self.list_children(folder_id, cancel, name=name, folders_only=folder):
            if item.name == name and item.is_folder == folder:
                return item
        return None

    def create_folder(self, name: str, parent: str = "root", cancel: Optional[CancelToken] = None) -> DriveFile:
        body = self._json({"name": name, "mimeType": FOLDER_MIME, "parents": [_check_id(parent)]})
        response = self._request("POST", self._files_url(fields=FILE_FIELDS), cancel,
                                 headers={"Content-Type": "application/json"}, body=body)
        self.require(response, expected=(200, 201))
        return DriveFile.from_json(response.json())

    def ensure_folder(self, name: str, parent: str = "root", cancel: Optional[CancelToken] = None) -> tuple:
        """``(folder, created)``: the app's folder ``name`` in ``parent``, created when missing."""
        found = self.find_child(parent, name, cancel, folder=True)
        if found is not None:
            return found, False
        return self.create_folder(name, parent, cancel), True

    def ensure_path(self, parts: Iterable[str], cancel: Optional[CancelToken] = None, *,
                    parent: str = "root") -> str:
        """The id of the folder path ``parts`` (e.g. ``("Books", "Glossarion")``) under ``parent``."""
        current = parent
        for part in parts:
            current = self.ensure_folder(part, current, cancel)[0].id
        return current

    def share_anyone(self, file_id: str, cancel: Optional[CancelToken] = None) -> None:
        response = self._request("POST", f"{self.api_base}/files/{_check_id(file_id)}/permissions", cancel,
                                 headers={"Content-Type": "application/json"},
                                 body=self._json({"type": "anyone", "role": "reader", "allowFileDiscovery": False}))
        self.require(response)

    def trash(self, file_id: str, cancel: Optional[CancelToken] = None) -> bool:
        """Move to Trash; False when Drive no longer had it."""
        response = self._request("PATCH", self._files_url(file_id, fields="id,trashed"), cancel,
                                 headers={"Content-Type": "application/json"}, body=self._json({"trashed": True}))
        if response.status == 404:
            return False
        self.require(response)
        return True

    # ---- content ------------------------------------------------------------------------------------

    def _check_session(self, location: str) -> str:
        session = urlsplit(location)
        base = urlsplit(self.upload_base)
        if (session.scheme != base.scheme or session.netloc != base.netloc
                or not session.path.startswith(base.path + "/files")):
            raise ShareError("protocol", "Google Drive returned an invalid upload session.")
        return location

    def upload(self, path: str, *, name: str, mime: str = "application/octet-stream", parent: Optional[str] = None,
               file_id: Optional[str] = None, size: Optional[int] = None,
               progress: Optional[Callable[[int, int], Any]] = None, cancel: Optional[CancelToken] = None,
               fields: str = FILE_FIELDS) -> DriveFile:
        """Resumable upload: a new file in ``parent`` (``None``: My Drive's root), or, with ``file_id``,
        new content for that file (files.update: same id, same link, same place)."""
        cancel = cancel or CancelToken()
        size = os.path.getsize(path) if size is None else int(size)
        metadata: dict = {"name": name}
        if file_id:
            url = f"{self.upload_base}/files/{_check_id(file_id)}?" + urlencode({"uploadType": "resumable",
                                                                                    "fields": fields})
            method = "PATCH"
        else:
            url = f"{self.upload_base}/files?" + urlencode({"uploadType": "resumable", "fields": fields})
            method = "POST"
            if parent:
                metadata["parents"] = [_check_id(parent)]
        response = self._request(method, url, cancel, body=self._json(metadata),
                                 headers={"Content-Type": "application/json", "X-Upload-Content-Type": mime,
                                          "X-Upload-Content-Length": str(size)})
        if file_id and response.status == 404:
            raise ShareError("not_found", "Google Drive no longer has this file.", status=404)
        self.require(response)
        session = self._check_session(response.headers.get("location", ""))
        if size == 0:
            response = self._request("PUT", session, cancel, body=b"", upload=True,
                                     headers={"Content-Type": mime, "Content-Range": "bytes */0"})
            self.require(response, expected=(200, 201))
            if progress is not None:
                progress(0, 0)
            return DriveFile.from_json(response.json())
        chunk_size = int(self.chunk_size or CHUNK_SIZE)
        offset = 0
        with open(path, "rb") as handle:
            while offset < size:
                cancel.check()
                chunk = handle.read(min(chunk_size, size - offset))
                if not chunk:
                    raise ShareError("missing_file", "The upload copy ended early.")
                end = offset + len(chunk)
                response = self._request("PUT", session, cancel, body=chunk, upload=True,
                                         headers={"Content-Type": mime,
                                                  "Content-Range": f"bytes {offset}-{end - 1}/{size}"})
                self.require(response, expected=(200, 201, 308))
                if response.status == 308:
                    stored = re.fullmatch(r"bytes=0-(\d+)", response.headers.get("range", ""))
                    acknowledged = int(stored[1]) + 1 if stored else 0
                    if not offset < acknowledged <= end or acknowledged >= size:
                        raise ShareError("protocol", "Google Drive did not acknowledge the upload chunk.")
                    offset = acknowledged
                    handle.seek(offset)
                else:
                    if end != size:
                        raise ShareError("protocol", "Google Drive ended the upload before the last chunk.")
                    offset = end
                if progress is not None:
                    progress(offset, size)
        return DriveFile.from_json(response.json())


# ---- "Share file via link" through Drive ----------------------------------------------------------------


class GoogleDriveProvider:
    """The share-link provider: upload to My Drive, then set Anyone with the link (Viewer).

    ``credentials["access_token"]`` is the ``access(rejected)`` callable of :class:`DriveClient`.
    """

    info = provider_info("gdrive")

    def __init__(self, http: Optional[HttpClient] = None, *, api_base: str = API_BASE,
                 upload_base: str = UPLOAD_BASE) -> None:
        self.http = http or HttpClient()
        self.api_base = api_base
        self.upload_base = upload_base

    def _client(self, credentials: Optional[Mapping[str, Any]]) -> DriveClient:
        access = (credentials or {}).get("access_token")
        if access is None:
            raise ShareError("sign_in", "Connect Google Drive in Settings › Cloud sync & sharing.")
        return DriveClient(access, self.http, api_base=self.api_base, upload_base=self.upload_base)

    def upload(self, path, *, name, mime, size, credentials, options=None, progress=None, cancel=None) -> UploadResult:
        cancel = cancel or CancelToken()
        client = self._client(credentials)
        file_id = client.upload(path, name=name, mime=mime, size=size, progress=progress, cancel=cancel,
                                fields="id").id
        try:
            client.share_anyone(file_id, cancel)
        except ShareError as exc:
            try:
                client.trash(file_id)
            except ShareError:
                raise ShareError("maybe_uploaded", "The file reached Drive but its share link could not be set up. "
                                 "Check your Google Drive before trying again.") from exc
            if exc.status == 403:
                raise ShareError("auth", "Your Google Drive account does not allow anyone-with-the-link sharing.") from exc
            raise
        return UploadResult(f"https://drive.google.com/file/d/{file_id}/view", file_id, {"id": file_id})

    def delete(self, handle, credentials=None, cancel=None) -> bool:
        file_id = str((handle or {}).get("id") or "")
        if not _ID.fullmatch(file_id):
            raise ShareError("not_found", "This Google Drive link has no file ID.")
        if not self._client(credentials).trash(file_id, cancel or CancelToken()):
            raise ShareError("not_found", "Google Drive could not find this file in the connected account.")
        return True
