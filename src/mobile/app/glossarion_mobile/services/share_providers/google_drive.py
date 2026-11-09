"""Google Drive sharing using the same drive.file grant and resumable protocol as NovelDB."""

from __future__ import annotations

import base64
import hashlib
import json
import re
import secrets
import threading
import time
from urllib.parse import parse_qs, urlencode, urlsplit

from glossarion_mobile.services.share_providers import (
    CancelToken, HttpClient, ShareError, UploadResult, provider_info,
)

SCOPE = "https://www.googleapis.com/auth/drive.file"
IOS_CLIENT_ID = "957824331511-1445nlka8d0ea1070r36109c4bqmshb2.apps.googleusercontent.com"
CALLBACK_SCHEME = "com.googleusercontent.apps." + IOS_CLIENT_ID.removesuffix(".apps.googleusercontent.com")
REDIRECT_URI = CALLBACK_SCHEME + ":/oauth2redirect"
TOKEN_URL = "https://oauth2.googleapis.com/token"
REVOKE_URL = "https://oauth2.googleapis.com/revoke"
DRIVE_FILES = "https://www.googleapis.com/drive/v3/files"
UPLOAD_URL = "https://www.googleapis.com/upload/drive/v3/files?uploadType=resumable&fields=id"
CHUNK_SIZE = 8 * 1024 * 1024
_ID = re.compile(r"[a-zA-Z0-9_-]+")


def authorization_request() -> tuple[dict, str, str]:
    state, verifier = secrets.token_urlsafe(32), secrets.token_urlsafe(48)
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
    params = {
        "client_id": IOS_CLIENT_ID, "redirect_uri": REDIRECT_URI, "response_type": "code",
        "scope": SCOPE, "code_challenge": challenge, "code_challenge_method": "S256",
        "state": state, "access_type": "offline", "prompt": "consent select_account",
    }
    return ({"interactive": True, "scheme": CALLBACK_SCHEME,
             "url": "https://accounts.google.com/o/oauth2/v2/auth?" + urlencode(params)}, state, verifier)


def callback_code(callback: str, state: str) -> str:
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


class DriveOAuth:
    """The iOS refresh token is persisted by ShareLinkService. Access tokens are memory only."""

    def __init__(self, http: HttpClient | None = None) -> None:
        self.http = http or HttpClient()
        self._lock = threading.Lock()
        self._cached = None

    def _token(self, **params) -> dict:
        result = self.http.request("POST", TOKEN_URL,
                                   headers={"Content-Type": "application/x-www-form-urlencoded"},
                                   body=urlencode({"client_id": IOS_CLIENT_ID, **params}).encode("ascii"))
        payload = result.json()
        if result.status >= 500:
            raise ShareError("server", "Google’s sign-in service is temporarily unavailable.")
        if result.status != 200 or not isinstance(payload, dict) or not payload.get("access_token"):
            code = "sign_in" if isinstance(payload, dict) and payload.get("error") == "invalid_grant" else "auth"
            raise ShareError(code, "Google Drive needs you to connect your account again.")
        return payload

    def exchange(self, callback: str, state: str, verifier: str) -> str:
        code = callback_code(callback, state)
        data = self._token(grant_type="authorization_code", code=code, code_verifier=verifier,
                           redirect_uri=REDIRECT_URI)
        refresh = data.get("refresh_token")
        if not refresh:
            raise ShareError("auth", "Google did not return a Drive refresh token.")
        return refresh

    def token(self, refresh: str, rejected: str = "") -> str:
        with self._lock:
            if self._cached:
                grant, token, expiry = self._cached
                if grant == refresh and token != rejected and expiry > time.monotonic():
                    return token
            data = self._token(grant_type="refresh_token", refresh_token=refresh)
            token = data["access_token"]
            self._cached = (refresh, token, time.monotonic() + max(0, int(data.get("expires_in", 3600)) - 90))
            return token

    def revoke(self, refresh: str) -> None:
        with self._lock:
            self._cached = None
            self.http.request("POST", REVOKE_URL, headers={"Content-Type": "application/x-www-form-urlencoded"},
                              body=urlencode({"token": refresh}).encode("ascii"))


class GoogleDriveProvider:
    info = provider_info("gdrive")

    def __init__(self, http: HttpClient | None = None) -> None:
        self.http = http or HttpClient()

    def _request(self, method, url, credentials, cancel, *, headers=None, body=None):
        access = (credentials or {}).get("access_token")
        if access is None:
            raise ShareError("sign_in", "Connect Google Drive in Settings › Cloud sync & sharing.")
        cancel.check()
        token = access("")
        for attempt in range(2):
            response = self.http.request(method, url, headers={**(headers or {}), "Authorization": f"Bearer {token}"},
                                         body=body, cancel=cancel)
            if response.status != 401 or attempt:
                return response
            cancel.check()
            token = access(token)

    @staticmethod
    def _json(data) -> bytes:
        return json.dumps(data).encode("utf-8")

    @staticmethod
    def _require(response, *, expected=(200, 201, 204)) -> None:
        status = response.status
        if status in expected:
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
        if status >= 500:
            raise ShareError("server", "Google Drive is temporarily unavailable.", status=status)
        raise ShareError("protocol", f"Google Drive declined the request (HTTP {status}).", status=status)

    def upload(self, path, *, name, mime, size, credentials, options, progress, cancel) -> UploadResult:
        response = self._request("POST", UPLOAD_URL, credentials, cancel, body=self._json({"name": name}),
                                 headers={"Content-Type": "application/json", "X-Upload-Content-Type": mime,
                                          "X-Upload-Content-Length": str(size)})
        self._require(response)
        session = response.headers.get("location", "")
        parts = urlsplit(session)
        if parts.scheme != "https" or parts.netloc != "www.googleapis.com" or parts.path != "/upload/drive/v3/files":
            raise ShareError("protocol", "Google Drive returned an invalid upload session.")

        offset = 0
        with open(path, "rb") as file:
            while offset < size:
                cancel.check()
                chunk = file.read(min(CHUNK_SIZE, size - offset))
                if not chunk:
                    raise ShareError("missing_file", "The upload copy ended early.")
                end = offset + len(chunk)
                response = self._request("PUT", session, credentials, cancel, body=chunk,
                                         headers={"Content-Type": mime,
                                                  "Content-Range": f"bytes {offset}-{end - 1}/{size}"})
                self._require(response, expected=(200, 201, 308))
                if response.status == 308:
                    stored = re.fullmatch(r"bytes=0-(\d+)", response.headers.get("range", ""))
                    acknowledged = int(stored[1]) + 1 if stored else 0
                    if not offset < acknowledged <= end or acknowledged >= size:
                        raise ShareError("protocol", "Google Drive did not acknowledge the upload chunk.")
                    offset = acknowledged
                    file.seek(offset)
                else:
                    if end != size:
                        raise ShareError("protocol", "Google Drive ended the upload before the last chunk.")
                    offset = end
                progress(offset, size)

        payload = response.json()
        file_id = payload.get("id", "") if isinstance(payload, dict) else ""
        if not isinstance(file_id, str) or not _ID.fullmatch(file_id):
            raise ShareError("protocol", "Google Drive did not return a file ID.")
        try:
            permission = self._request(
                "POST", f"{DRIVE_FILES}/{file_id}/permissions", credentials, cancel,
                headers={"Content-Type": "application/json"},
                body=self._json({"type": "anyone", "role": "reader", "allowFileDiscovery": False}),
            )
            self._require(permission)
        except ShareError as exc:
            try:
                self.delete({"id": file_id}, credentials)
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
        response = self._request("PATCH", f"{DRIVE_FILES}/{file_id}?fields=id,trashed", credentials,
                                 cancel or CancelToken(), headers={"Content-Type": "application/json"},
                                 body=self._json({"trashed": True}))
        if response.status == 404:
            raise ShareError("not_found", "Google Drive could not find this file in the connected account.")
        self._require(response)
        return True
