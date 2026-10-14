"""Gofile (gofile.io, WOJTEK SAS) through its documented API (gofile.io/api, marked BETA).

* Upload: ``POST https://upload.gofile.io/uploadfile`` (multipart field ``file``), with the saved
  guest token as ``Authorization: Bearer`` when there is one. Without ``folderId`` Gofile creates a
  new public folder per upload; the link is ``data.downloadPage``.
* One guest account per install: the first upload goes without a token and Gofile answers with
  ``data.guestToken``, which the caller stores (encrypted) and reuses, as Gofile's docs ask ("create
  one account and reuse its token"). No account is minted per upload.
* Delete: ``DELETE https://api.gofile.io/contents`` ``{"contentsId": <folder id>}`` with the token
  that uploaded it. The handle is the upload's ``parentFolder`` (critic: deleting only the file
  would leave an empty public folder), together with that token, so a later new guest account
  does not orphan older links.
* Retries (critic): HTTP 429 is retried once, and only when Retry-After asks for at most
  ``MAX_RETRY_AFTER`` seconds; anything else (no header, a longer wait, a second 429) is "busy, try
  later". A timeout after the body was sent is never retried (the folder may exist already).
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Optional

from glossarion_mobile.services.share_providers import (
    MAX_RETRY_AFTER,
    CancelToken,
    HttpClient,
    Progress,
    ShareError,
    UploadResult,
    multipart_file_body,
    parse_retry_after,
    provider_info,
)

__all__ = ["API_BASE", "GofileProvider", "UPLOAD_URL"]

log = logging.getLogger("glossarion.share.gofile")

API_BASE = "https://api.gofile.io"
UPLOAD_URL = "https://upload.gofile.io/uploadfile"
LINK_PREFIX = "https://gofile.io/d/"


def _status(payload: Any) -> str:
    return str(payload.get("status") or "") if isinstance(payload, Mapping) else ""


def _data(payload: Any) -> Mapping:
    data = payload.get("data") if isinstance(payload, Mapping) else None
    return data if isinstance(data, Mapping) else {}


class GofileProvider:
    info = provider_info("gofile")

    def __init__(self, http: Optional[HttpClient] = None, *, api_base: str = API_BASE, upload_url: str = UPLOAD_URL,
                 link_prefix: str = LINK_PREFIX, max_retry_after: float = MAX_RETRY_AFTER) -> None:
        self.http = http or HttpClient()
        self.api_base = api_base.rstrip("/")
        self.upload_url = upload_url
        self.link_prefix = link_prefix
        self.max_retry_after = float(max_retry_after)

    # ---- errors ------------------------------------------------------------------------------

    def _error(self, response: Any, action: str) -> ShareError:
        payload = response.json()
        status = _status(payload)
        if response.status == 429 or status == "error-rateLimit":
            retry = parse_retry_after(response.headers.get("retry-after"))
            wait = f" Try again in about {max(1, int(round(retry / 60)))} min." if retry and retry > 60 else \
                " Try again later."
            return ShareError("rate_limited", "Gofile is busy (too many requests)." + wait, retry_after=retry,
                              status=response.status)
        if status == "error-limits":
            return ShareError("limits", "Gofile's free limits for this guest account are reached.",
                              status=response.status)
        if response.status in (401, 403) or status in ("error-auth", "error-token", "error-notPremium"):
            return ShareError("auth", "Gofile did not accept the saved guest account. Tap Retry to use a new one.",
                              status=response.status)
        if response.status == 404 or status == "error-notFound":
            return ShareError("not_found", "Gofile no longer has this upload.", status=response.status)
        if response.status >= 500:
            return ShareError("server", f"Gofile could not {action} right now (HTTP {response.status}).",
                              status=response.status)
        return ShareError("server", f"Gofile could not {action} ({status or 'HTTP ' + str(response.status)}).",
                          status=response.status)

    # ---- upload -------------------------------------------------------------------------------

    def upload(self, path: str, *, name: str, mime: str, size: int, credentials: Mapping[str, str],
               options: Optional[Mapping[str, Any]] = None, progress: Optional[Progress] = None,
               cancel: Optional[CancelToken] = None) -> UploadResult:
        cancel = cancel or CancelToken()
        token = str((credentials or {}).get("token") or "")
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        content_type, length, make_body = multipart_file_body("file", name, mime, path, size, progress=progress,
                                                              cancel=cancel)
        headers["Content-Type"] = content_type
        retried = False
        while True:
            response = self.http.request("POST", self.upload_url, headers=headers, body=make_body(), length=length,
                                         cancel=cancel)
            if response.status == 429 and not retried:
                retry = parse_retry_after(response.headers.get("retry-after"))
                if retry is not None and retry <= self.max_retry_after:
                    retried = True
                    log.info("Gofile asked to wait %.0f s (HTTP 429); retrying once", retry)
                    if cancel.wait(retry):
                        cancel.check()
                    continue
            break
        payload = response.json()
        if response.status != 200 or _status(payload) != "ok":
            raise self._error(response, "take the upload")
        data = _data(payload)
        url = str(data.get("downloadPage") or "")
        folder = str(data.get("parentFolder") or "")
        if not url.startswith(self.link_prefix) or not folder:
            raise ShareError("protocol", "Gofile's answer did not contain a link")
        new_token = "" if token else str(data.get("guestToken") or "")
        return UploadResult(
            url=url, remote_id=str(data.get("id") or ""),
            delete_handle={"folder": folder, "token": token or new_token} if (token or new_token) else None,
            account_token=new_token or None,
        )

    # ---- delete -------------------------------------------------------------------------------

    def delete(self, handle: Mapping[str, Any], credentials: Optional[Mapping[str, str]] = None,
               cancel: Optional[CancelToken] = None) -> bool:
        """True when deleted, False when Gofile no longer had it; ``ShareError`` otherwise."""
        folder = str((handle or {}).get("folder") or "")
        token = str((handle or {}).get("token") or (credentials or {}).get("token") or "")
        if not folder or not token:
            raise ShareError("auth", "This Gofile link cannot be deleted from the app (no guest account saved).")
        import json

        body = json.dumps({"contentsId": folder}).encode("utf-8")
        response = self.http.request("DELETE", f"{self.api_base}/contents", body=body, cancel=cancel,
                                     headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
        payload = response.json()
        if response.status == 200 and _status(payload) == "ok":
            return True
        error = self._error(response, "delete the upload")
        if error.code == "not_found":
            return False
        raise error
