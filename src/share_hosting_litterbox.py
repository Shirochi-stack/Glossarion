"""Litterbox (litterbox.catbox.moe, Catbox's temporary host) through its documented API (its Tools page).

* Upload: ``POST https://litterbox.catbox.moe/resources/internals/api.php``, multipart form with
  ``reqtype=fileupload``, ``time`` (``1h`` / ``12h`` / ``24h`` / ``72h``) and the file in
  ``fileToUpload``. Uploads are anonymous (no account, no key). The answer body is the link itself in
  plain text (``https://litter.catbox.moe/<name>.<ext>``).
* The file expires by itself after the chosen time (up to 3 days); the uploader cannot delete it, so
  there is no delete handle. Temporary uploads are limited to 1 GB (the site's front page).
* Retries: none (the U10 rule). A timeout after the whole body was sent is ``maybe_uploaded``
  (``HttpClient``), never retried.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Mapping, Optional

from share_hosting import (
    CancelToken,
    HttpClient,
    Progress,
    ShareError,
    UploadResult,
    multipart_file_body,
    provider_info,
)

__all__ = ["API_URL", "DEFAULT_TIME", "LINK_PREFIX", "LitterboxProvider", "TIME_CHOICES"]

log = logging.getLogger("glossarion.share.litterbox")

API_URL = "https://litterbox.catbox.moe/resources/internals/api.php"
LINK_PREFIX = "https://litter.catbox.moe/"
#: ``time`` values the API accepts, with the label the settings show and the lifetime in seconds.
TIME_CHOICES = (("1h", "1 hour", 3600), ("12h", "12 hours", 12 * 3600), ("24h", "1 day", 86400),
                ("72h", "3 days", 3 * 86400))
DEFAULT_TIME = "72h"
_SECONDS = {value: seconds for value, _label, seconds in TIME_CHOICES}


class LitterboxProvider:
    info = provider_info("litterbox")

    def __init__(self, http: Optional[HttpClient] = None, *, api_url: str = API_URL,
                 link_prefix: str = LINK_PREFIX) -> None:
        self.http = http or HttpClient()
        self.api_url = api_url
        self.link_prefix = link_prefix

    @staticmethod
    def options(raw: Optional[Mapping[str, Any]]) -> str:
        """The stored expiry (``{"time": "72h"}``) as one of the API's values; anything else is the default."""
        value = str((raw or {}).get("time") or "").strip().lower()
        return value if value in _SECONDS else DEFAULT_TIME

    def upload(self, path: str, *, name: str, mime: str, size: int, credentials: Optional[Mapping[str, str]] = None,
               options: Optional[Mapping[str, Any]] = None, progress: Optional[Progress] = None,
               cancel: Optional[CancelToken] = None) -> UploadResult:
        cancel = cancel or CancelToken()
        expiry = self.options(options)
        content_type, length, make_body = multipart_file_body(
            "fileToUpload", name, mime, path, size, fields=(("reqtype", "fileupload"), ("time", expiry)),
            progress=progress, cancel=cancel)
        response = self.http.request("POST", self.api_url, headers={"Content-Type": content_type},
                                     body=make_body(), length=length, cancel=cancel)
        if response.status == 413:
            raise ShareError("too_large", "The file is larger than Litterbox allows.", status=response.status)
        if response.status == 429:
            raise ShareError("rate_limited", "Litterbox is busy (too many uploads). Try again later.",
                             status=response.status)
        if response.status >= 500:
            raise ShareError("server", f"Litterbox could not take the upload right now (HTTP {response.status}).",
                             status=response.status)
        url = response.body.decode("utf-8", "replace").strip()
        if response.status != 200 or not url.startswith(self.link_prefix) or any(c.isspace() for c in url):
            # the answer is not a link (e.g. a refused file type): never shown verbatim (it could be HTML)
            raise ShareError("server", f"Litterbox did not accept the upload (HTTP {response.status}).",
                             status=response.status)
        return UploadResult(url=url, remote_id=url[len(self.link_prefix):], delete_handle=None,
                            expires=time.time() + _SECONDS[expiry])

    def delete(self, handle: Any = None, credentials: Any = None, cancel: Any = None) -> bool:
        raise ShareError("unsupported", "Litterbox uploads cannot be deleted; they expire by themselves.")
