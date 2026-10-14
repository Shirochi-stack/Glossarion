"""pixeldrain (pixeldrain.com, Fornaxian Technologies) with the user's own API key (pixeldrain.com/api).

pixeldrain has no anonymous uploads, so the user pastes the API key of their own free account
(Settings › Cloud sync & sharing); it is stored encrypted on the phone and sent only to pixeldrain,
as HTTP Basic auth with an empty user name.

* Upload: ``PUT https://pixeldrain.com/api/file/<name>`` with the raw file as the body; ``201``
  ``{"id": ...}``; the link is ``https://pixeldrain.com/u/<id>``.
* Delete: ``DELETE https://pixeldrain.com/api/file/<id>`` with the same key.
* Check key: ``GET https://pixeldrain.com/api/user`` (only when the user taps "Check key").

No automatic retries (U10 critic).
"""

from __future__ import annotations

import logging
import re
from typing import Any, Mapping, Optional

from glossarion_mobile.services.share_providers import (
    CancelToken,
    HttpClient,
    Progress,
    ShareError,
    UploadResult,
    basic_auth,
    file_chunks,
    parse_retry_after,
    provider_info,
    quote_path_name,
)

__all__ = ["API_BASE", "PixeldrainProvider", "clean_api_key"]

log = logging.getLogger("glossarion.share.pixeldrain")

API_BASE = "https://pixeldrain.com/api"
LINK_PREFIX = "https://pixeldrain.com/u/"
_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_KEY = re.compile(r"^[\x21-\x7e]{8,128}$")


def clean_api_key(value: Any) -> str:
    """The pasted key without surrounding whitespace; ``ShareError('no_key')`` when it cannot be one."""
    key = str(value or "").strip()
    if not _KEY.match(key):
        raise ShareError("no_key", "That does not look like a pixeldrain API key (copy it from pixeldrain › "
                         "Account › API keys).")
    return key


class PixeldrainProvider:
    info = provider_info("pixeldrain")

    def __init__(self, http: Optional[HttpClient] = None, *, api_base: str = API_BASE,
                 link_prefix: str = LINK_PREFIX) -> None:
        self.http = http or HttpClient()
        self.api_base = api_base.rstrip("/")
        self.link_prefix = link_prefix

    @staticmethod
    def _auth(credentials: Optional[Mapping[str, str]]) -> str:
        key = str((credentials or {}).get("api_key") or "")
        if not key:
            raise ShareError("no_key", "Add your pixeldrain API key in Settings › Cloud sync & sharing first.")
        return basic_auth("", key)

    def _error(self, response: Any, action: str) -> ShareError:
        payload = response.json()
        value = str(payload.get("value") or "") if isinstance(payload, Mapping) else ""
        if response.status in (401, 403) or value in ("authentication_required", "authentication_failed",
                                                       "unauthorized"):
            return ShareError("auth", "pixeldrain did not accept the API key. Check it in Settings.",
                              status=response.status)
        if response.status == 413 or value == "file_too_large":
            return ShareError("too_large", "The file is larger than pixeldrain allows for this account.",
                              status=response.status)
        if response.status == 404 or value == "not_found":
            return ShareError("not_found", "pixeldrain no longer has this file.", status=response.status)
        if response.status == 429:
            return ShareError("rate_limited", "pixeldrain is busy (too many requests). Try again later.",
                              retry_after=parse_retry_after(response.headers.get("retry-after")),
                              status=response.status)
        if response.status >= 500:
            return ShareError("server", f"pixeldrain could not {action} right now (HTTP {response.status}).",
                              status=response.status)
        return ShareError("server", f"pixeldrain could not {action} ({value or 'HTTP ' + str(response.status)}).",
                          status=response.status)

    def upload(self, path: str, *, name: str, mime: str, size: int, credentials: Mapping[str, str],
               options: Optional[Mapping[str, Any]] = None, progress: Optional[Progress] = None,
               cancel: Optional[CancelToken] = None) -> UploadResult:
        cancel = cancel or CancelToken()
        headers = {"Authorization": self._auth(credentials), "Content-Type": mime or "application/octet-stream"}
        try:
            response = self.http.request("PUT", f"{self.api_base}/file/{quote_path_name(name)}", headers=headers,
                                         body=file_chunks(path, int(size), progress=progress, cancel=cancel),
                                         length=int(size), cancel=cancel)
        except ShareError as exc:
            # A server that refuses the key answers and closes while the body is still being sent; the
            # connection reset can swallow that answer. Ask once whether the key is the problem.
            if exc.code == "network" and not cancel.cancelled:
                try:
                    valid = self.check_key(credentials, cancel=cancel)
                except ShareError:
                    valid = True
                if not valid:
                    raise ShareError("auth", "pixeldrain did not accept the API key. Check it in Settings.") from exc
            raise
        if response.status not in (200, 201):
            raise self._error(response, "take the upload")
        payload = response.json()
        file_id = str(payload.get("id") or "") if isinstance(payload, Mapping) else ""
        if not _ID.match(file_id):
            raise ShareError("protocol", "pixeldrain's answer did not contain a file id")
        return UploadResult(url=self.link_prefix + file_id, remote_id=file_id, delete_handle={"id": file_id})

    def delete(self, handle: Mapping[str, Any], credentials: Optional[Mapping[str, str]] = None,
               cancel: Optional[CancelToken] = None) -> bool:
        file_id = str((handle or {}).get("id") or "")
        if not _ID.match(file_id):
            raise ShareError("not_found", "This pixeldrain link has no file id to delete.")
        response = self.http.request("DELETE", f"{self.api_base}/file/{file_id}", cancel=cancel,
                                     headers={"Authorization": self._auth(credentials)})
        if response.status in (200, 204):
            return True
        error = self._error(response, "delete the file")
        if error.code == "not_found":
            return False
        if error.code == "auth":
            raise ShareError("auth", "pixeldrain refused: this link was uploaded with a different API key.",
                             status=error.status)
        raise error

    def check_key(self, credentials: Mapping[str, str], cancel: Optional[CancelToken] = None) -> bool:
        """True when pixeldrain accepts the key (``GET /user``); ``ShareError`` for network trouble."""
        response = self.http.request("GET", f"{self.api_base}/user", cancel=cancel,
                                     headers={"Authorization": self._auth(credentials)})
        if response.status == 200:
            return True
        error = self._error(response, "check the key")
        if error.code == "auth":
            return False
        raise error
