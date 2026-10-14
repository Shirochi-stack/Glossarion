"""Send (send.vis.ee): end-to-end encrypted share links with the timvisee/send protocol (v3).

The file is encrypted on the phone; the server only stores ciphertext. The 16-byte secret travels
only in the link's fragment (``…/download/<id>/#<secret>``), which browsers never send to a server.
Following timvisee/send ``docs/encryption.md`` and its client (``app/keychain.js``, ``app/ece.js``,
``app/api.js``):

* keys (HKDF-SHA256, empty salt, from the secret): metadata key = 16 bytes, info ``metadata``;
  auth key = 64 bytes (the HMAC-SHA256 block size WebCrypto derives), info ``authentication``;
* metadata: AES-128-GCM, 12 zero bytes IV, over the JSON ``{"name","size","type","manifest":
  {"files":[{"name","size","type"}]}}``, sent base64url (no padding) as ``fileMetadata``;
* content: RFC 8188 ``aes128gcm`` with the secret as IKM, a random 16-byte salt and 64 KiB records:
  header = salt + record size (uint32 BE) + key-id length 0; key = HKDF(salt, ikm, "Content-Encoding:
  aes128gcm\\0", 16); nonce base = HKDF(…, "Content-Encoding: nonce\\0", 12); record ``i`` uses nonce
  base XOR ``i``; every record but the last carries ``rs - 17`` plaintext bytes + delimiter 0x01,
  the last one its bytes + 0x02;
* upload over ``wss://<host>/api/ws``: a JSON text frame ``{fileMetadata, authorization: "send-v1
  <auth key>", timeLimit, dlimit}`` -> the server answers ``{url, ownerToken, id}``; then the
  header and the records as binary frames, a single ``0x00`` byte as end-of-file, and the server
  answers ``{"ok": true}`` (or ``{"error": <HTTP status>}``);
* delete: ``POST /api/delete/<id>`` ``{"owner_token": …}``.

Limits of send.vis.ee: 2.5 GiB, at most 3 days and 20 downloads. No automatic retries. Uses the
pinned ``cryptography`` (AES-GCM, HKDF) and ``websockets`` (sync client), imported lazily.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import time
from typing import Any, Callable, Iterator, Mapping, Optional, Tuple
from urllib.parse import urlsplit

from glossarion_mobile.services.share_providers import (
    CancelToken,
    HttpClient,
    Progress,
    ShareError,
    UploadResult,
    _ssl_context,
    abort_socket,
    is_loopback_host,
    provider_info,
    user_agent,
)

__all__ = [
    "BASE_URL",
    "DEFAULT_DOWNLOADS",
    "DEFAULT_EXPIRE",
    "DOWNLOAD_CHOICES",
    "EXPIRY_CHOICES",
    "Keychain",
    "MAX_DOWNLOADS",
    "MAX_EXPIRE",
    "RECORD_SIZE",
    "SendProvider",
    "b64url",
    "b64url_decode",
    "ece_encrypt",
    "encrypted_size",
    "metadata_json",
]

log = logging.getLogger("glossarion.share.send")

BASE_URL = "https://send.vis.ee"
RECORD_SIZE = 64 * 1024  # ECE_RECORD_SIZE
TAG_LENGTH = 16
KEY_LENGTH = 16
NONCE_LENGTH = 12
HEADER_LENGTH = KEY_LENGTH + 5
MAX_DOWNLOADS = 20
MAX_EXPIRE = 3 * 24 * 3600
#: send.vis.ee's choices (5 min, 1 h, 24 h, 3 days; 1-20 downloads).
EXPIRY_CHOICES = (300, 3600, 86400, MAX_EXPIRE)
DOWNLOAD_CHOICES = (1, 2, 3, 5, 10, 20)
#: Owner decision (U10): the instance's maximum, 3 days / 20 downloads.
DEFAULT_EXPIRE = MAX_EXPIRE
DEFAULT_DOWNLOADS = MAX_DOWNLOADS
_ID = re.compile(r"^[0-9a-fA-F]{8,64}$")
_OWNER = re.compile(r"^[0-9a-fA-F]{8,128}$")


def b64url(data: bytes) -> str:
    """``arrayToB64``: URL-safe base64 without padding."""
    return base64.urlsafe_b64encode(bytes(data)).decode("ascii").rstrip("=")


def b64url_decode(text: str) -> bytes:
    text = str(text or "").strip()
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def _hkdf(ikm: bytes, length: int, info: bytes, salt: bytes = b"") -> bytes:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF

    return HKDF(algorithm=hashes.SHA256(), length=length, salt=salt, info=info).derive(bytes(ikm))


def metadata_json(name: str, size: int, mime: str) -> bytes:
    """``Keychain.encryptMetadata``'s plaintext: ``JSON.stringify`` of the archive metadata (one file)."""
    mime = mime or "application/octet-stream"
    entry = {"name": name, "size": int(size), "type": mime}
    payload = {"name": name, "size": int(size), "type": mime, "manifest": {"files": [entry]}}
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


class Keychain:
    """The upload's secret and the keys derived from it (``app/keychain.js``)."""

    def __init__(self, secret: Optional[bytes] = None) -> None:
        self.secret = bytes(secret) if secret is not None else os.urandom(KEY_LENGTH)
        if len(self.secret) != KEY_LENGTH:
            raise ValueError("the Send secret is 16 bytes")
        self.meta_key = _hkdf(self.secret, 16, b"metadata")
        self.auth_key = _hkdf(self.secret, 64, b"authentication")

    @property
    def secret_b64(self) -> str:
        return b64url(self.secret)

    def authorization(self) -> str:
        return f"send-v1 {b64url(self.auth_key)}"

    def encrypt_metadata(self, name: str, size: int, mime: str) -> bytes:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        return AESGCM(self.meta_key).encrypt(bytes(NONCE_LENGTH), metadata_json(name, size, mime), None)


def encrypted_size(size: int, rs: int = RECORD_SIZE, tag_size: int = TAG_LENGTH) -> int:
    """``encryptedSize``: header + data + (tag + delimiter) per record."""
    meta = tag_size + 1
    records = -(-int(size) // (rs - meta)) if size else 0
    return HEADER_LENGTH + int(size) + meta * records


def ece_encrypt(read: Callable[[int], bytes], size: int, ikm: bytes, *, salt: Optional[bytes] = None,
                rs: int = RECORD_SIZE, cancel: Optional[CancelToken] = None) -> Iterator[Tuple[bytes, int]]:
    """``(chunk, plaintext bytes consumed)``: the header first, then one encrypted record per chunk.

    ``read(n)`` returns the next plaintext bytes; exactly ``size`` bytes are consumed (a shorter
    source is an error: the snapshot changed).
    """
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    if rs < TAG_LENGTH + 2 or rs > 0xFFFFFFFF:
        raise ValueError("record size out of range")
    salt = bytes(salt) if salt is not None else os.urandom(KEY_LENGTH)
    if len(salt) != KEY_LENGTH:
        raise ValueError("the salt is 16 bytes")
    key = _hkdf(ikm, 16, b"Content-Encoding: aes128gcm\x00", salt)
    nonce_base = _hkdf(ikm, NONCE_LENGTH, b"Content-Encoding: nonce\x00", salt)
    aead = AESGCM(key)
    head, tail = nonce_base[:-4], int.from_bytes(nonce_base[-4:], "big")
    yield salt + int(rs).to_bytes(4, "big") + b"\x00", 0
    chunk = rs - TAG_LENGTH - 1
    records = -(-int(size) // chunk) if size else 0
    if records > 0xFFFFFFFF + 1:
        raise ValueError("too many records")
    consumed = 0
    for seq in range(records):
        if cancel is not None:
            cancel.check()
        want = min(chunk, int(size) - consumed)
        data = read(want)
        if len(data) != want:
            raise ShareError("missing_file", "The file got shorter while it was being encrypted")
        consumed += want
        last = seq == records - 1
        padded = data + (b"\x02" if last else b"\x01" + bytes(rs - want - TAG_LENGTH - 1))
        nonce = head + ((tail ^ seq) & 0xFFFFFFFF).to_bytes(4, "big")
        yield aead.encrypt(nonce, padded, None), consumed


def _ws_connect(url: str, *, open_timeout: float, close_timeout: float) -> Any:
    import inspect

    from websockets.sync.client import connect

    kwargs: dict = {"open_timeout": open_timeout, "close_timeout": close_timeout, "max_size": 1024 * 1024,
                    "compression": None, "user_agent_header": user_agent()}
    if url.lower().startswith("wss://"):
        kwargs["ssl"] = _ssl_context()
    elif is_loopback_host(urlsplit(url).hostname):
        try:  # websockets >= 15 would route even a loopback test server through a system proxy
            if "proxy" in inspect.signature(connect).parameters:
                kwargs["proxy"] = None
        except (TypeError, ValueError):
            pass
    return connect(url, **kwargs)


def _ws_errors() -> tuple:
    try:
        from websockets.exceptions import WebSocketException

        return (OSError, TimeoutError, WebSocketException)
    except Exception:  # pragma: no cover - websockets missing
        return (OSError, TimeoutError)


class SendProvider:
    info = provider_info("send")

    def __init__(self, http: Optional[HttpClient] = None, *, base_url: str = BASE_URL, ws_url: Optional[str] = None,
                 connect: Optional[Callable[..., Any]] = None, open_timeout: float = 15.0,
                 reply_timeout: float = 30.0, final_timeout: float = 120.0, rs: int = RECORD_SIZE) -> None:
        self.http = http or HttpClient()
        self.base_url = base_url.rstrip("/")
        parts = urlsplit(self.base_url)
        default_ws = ("wss://" if parts.scheme == "https" else "ws://") + parts.netloc + "/api/ws"
        self.ws_url = ws_url or default_ws
        self._connect = connect or _ws_connect
        self.open_timeout = float(open_timeout)
        self.reply_timeout = float(reply_timeout)
        self.final_timeout = float(final_timeout)
        self.rs = int(rs)

    # ---- helpers ---------------------------------------------------------------------------------

    def _check_url(self) -> None:
        parts = urlsplit(self.ws_url)
        if parts.scheme == "wss" or (parts.scheme == "ws" and is_loopback_host(parts.hostname)):
            return
        raise ShareError("protocol", "Only secure (wss) connections are allowed")

    @staticmethod
    def options(options: Optional[Mapping[str, Any]]) -> tuple:
        """``(expire seconds, download limit)`` clamped to the instance's limits."""
        options = options or {}
        try:
            expire = int(options.get("expire") or DEFAULT_EXPIRE)
        except (TypeError, ValueError):
            expire = DEFAULT_EXPIRE
        try:
            downloads = int(options.get("downloads") or DEFAULT_DOWNLOADS)
        except (TypeError, ValueError):
            downloads = DEFAULT_DOWNLOADS
        return max(60, min(MAX_EXPIRE, expire)), max(1, min(MAX_DOWNLOADS, downloads))

    @staticmethod
    def _server_error(code: Any) -> ShareError:
        try:
            status = int(code)
        except (TypeError, ValueError):
            status = 0
        if status == 413:
            return ShareError("too_large", "The file is larger than this Send server allows.", status=status)
        if status == 400:
            return ShareError("limits", "This Send server refused the expiry or download limit.", status=status)
        if status in (401, 403):
            return ShareError("auth", "This Send server only accepts uploads from signed-in accounts.", status=status)
        return ShareError("server", f"The Send server could not store the file (error {status or code}).",
                          status=status or None)

    def _reply(self, ws: Any, timeout: float, *, stage: str) -> dict:
        try:
            message = ws.recv(timeout=timeout)
        except TimeoutError as exc:
            raise ShareError("network", f"The Send server did not answer ({stage}). Try again.") from exc
        except _ws_errors() as exc:
            raise ShareError("network", f"The connection to the Send server closed ({stage}). Try again.") from exc
        if isinstance(message, (bytes, bytearray)):
            message = bytes(message).decode("utf-8", "replace")
        try:
            payload = json.loads(message)
        except (TypeError, ValueError):
            raise ShareError("protocol", "The Send server sent an unexpected answer") from None
        if not isinstance(payload, dict):
            raise ShareError("protocol", "The Send server sent an unexpected answer")
        if payload.get("error") not in (None, False, ""):
            raise self._server_error(payload.get("error"))
        return payload

    # ---- upload ------------------------------------------------------------------------------------

    def upload(self, path: str, *, name: str, mime: str, size: int, credentials: Optional[Mapping[str, str]] = None,
               options: Optional[Mapping[str, Any]] = None, progress: Optional[Progress] = None,
               cancel: Optional[CancelToken] = None, keychain: Optional[Keychain] = None,
               salt: Optional[bytes] = None) -> UploadResult:
        cancel = cancel or CancelToken()
        self._check_url()
        expire, downloads = self.options(options)
        keychain = keychain or Keychain()
        metadata = keychain.encrypt_metadata(name, size, mime)
        cancel.check()
        try:
            ws = self._connect(self.ws_url, open_timeout=self.open_timeout, close_timeout=5.0)
        except ShareError:
            raise
        except _ws_errors() as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status in (401, 403):
                raise ShareError("auth", "The Send server refused the connection.", status=status) from exc
            raise ShareError("network", f"Could not reach the Send server ({type(exc).__name__})") from exc

        remove = cancel.on_cancel(lambda: abort_socket(getattr(ws, "socket", None)))
        started = time.time()
        try:
            try:
                ws.send(json.dumps({"fileMetadata": b64url(metadata), "authorization": keychain.authorization(),
                                    "timeLimit": expire, "dlimit": downloads}))
            except _ws_errors() as exc:
                cancel.check()
                raise ShareError("network", "The connection to the Send server closed. Try again.") from exc
            info = self._reply(ws, self.reply_timeout, stage="before the upload")
            file_id = str(info.get("id") or "")
            owner = str(info.get("ownerToken") or "")
            if not _ID.match(file_id) or not _OWNER.match(owner):
                raise ShareError("protocol", "The Send server's answer had no file id")
            if progress is not None:
                progress(0, int(size))
            with open(path, "rb") as handle:
                for chunk, consumed in ece_encrypt(handle.read, int(size), keychain.secret, salt=salt, rs=self.rs,
                                                   cancel=cancel):
                    cancel.check()
                    try:
                        ws.send(chunk)
                    except _ws_errors() as exc:
                        cancel.check()
                        raise ShareError("network", "The connection to the Send server broke during the upload. "
                                         "Try again.") from exc
                    if progress is not None and consumed:
                        progress(consumed, int(size))
            cancel.check()
            try:
                ws.send(b"\x00")
            except _ws_errors() as exc:
                cancel.check()
                raise ShareError("network", "The connection to the Send server broke at the end. Try again.") from exc
            done = self._reply(ws, self.final_timeout, stage="after the upload")
            if done.get("ok") is not True:
                raise ShareError("protocol", "The Send server did not confirm the upload")
        except ShareError:
            cancel.check()
            raise
        finally:
            remove()
            try:
                ws.close()
            except Exception:
                pass
        link_base = self._link_base(str(info.get("url") or ""), file_id)
        return UploadResult(
            url=f"{link_base}#{keychain.secret_b64}", remote_id=file_id,
            delete_handle={"id": file_id, "owner": owner}, expires=started + expire, downloads_limit=downloads,
        )

    def _link_base(self, url: str, file_id: str) -> str:
        """The server's ``…/download/<id>/`` when it is on our host, else built from the id."""
        expected = f"{self.base_url}/download/{file_id}/"
        if url == expected or url.rstrip("/") == expected.rstrip("/"):
            return expected
        if url:
            log.info("Send returned a download URL on another host; using %s instead", self.base_url)
        return expected

    # ---- delete ------------------------------------------------------------------------------------

    def delete(self, handle: Mapping[str, Any], credentials: Optional[Mapping[str, str]] = None,
               cancel: Optional[CancelToken] = None) -> bool:
        file_id = str((handle or {}).get("id") or "")
        owner = str((handle or {}).get("owner") or "")
        if not _ID.match(file_id) or not owner:
            raise ShareError("not_found", "This Send link has no owner token to delete it with.")
        response = self.http.request("POST", f"{self.base_url}/api/delete/{file_id}", cancel=cancel,
                                     headers={"Content-Type": "application/json"},
                                     body=json.dumps({"owner_token": owner}).encode("utf-8"))
        if response.status in (200, 204):
            return True
        if response.status == 404:
            return False
        if response.status in (401, 403):
            raise ShareError("auth", "The Send server refused to delete this link.", status=response.status)
        raise ShareError("server", f"The Send server could not delete the link (HTTP {response.status}).",
                         status=response.status)
