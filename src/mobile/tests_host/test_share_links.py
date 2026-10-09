"""Host tests for U10 "Share file via link" (``services/share_links.py`` + ``services/share_providers``).

Every network request goes to a local fake server on 127.0.0.1 (Gofile, pixeldrain, and Send with
its WebSocket upload); a socket guard fails the test if anything tries to reach another host. The
Send link is decrypted with an independent reference implementation of timvisee/send's v3 scheme
(written here, not imported from the app), and the RFC 8188 ``aes128gcm`` example vector pins the
content encryption.

Covered: every provider off by default and gated by its consent (version), the Gofile guest token
reused and stored encrypted, the Gofile 429 rule (one retry only with a short Retry-After), no retry
after a timeout once the body was sent, pixeldrain with the user's key (encrypted), delete for
Gofile / pixeldrain / Send, Cancel, size checks, eligibility (no key or config exports), the
transfer.it handoff with zero requests, records relocating with a book, the platform holds, no
secret in the sidecar, Prefs or the logs.

Run from src/mobile:
    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_share_links.py
"""

from __future__ import annotations

import ast
import asyncio
import base64
import dataclasses
import errno
import hmac
import importlib.util
import json
import logging
import os
import re
import secrets
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional
from urllib.parse import unquote

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = MOBILE_DIR / "app"
SRC_DIR = MOBILE_DIR.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

cryptography = pytest.importorskip("cryptography")
pytest.importorskip("websockets")

from cryptography.fernet import Fernet  # noqa: E402
from cryptography.hazmat.primitives import hashes  # noqa: E402
from cryptography.hazmat.primitives.ciphers.aead import AESGCM  # noqa: E402
from cryptography.hazmat.primitives.kdf.hkdf import HKDF  # noqa: E402

from glossarion_mobile.services import share_links as sl  # noqa: E402
from glossarion_mobile.services import share_providers as sp  # noqa: E402
from glossarion_mobile.services.share_providers import gofile as gf  # noqa: E402
from glossarion_mobile.services.share_providers import pixeldrain as pd  # noqa: E402
from glossarion_mobile.services.share_providers import send_e2ee as se  # noqa: E402
from glossarion_mobile.services.share_providers import transferit_handoff as th  # noqa: E402

SERVICES_DIR = APP_DIR / "glossarion_mobile" / "services"
PROVIDER_FILES = sorted((SERVICES_DIR / "share_providers").glob("*.py"))


# ---------------------------------------------------------------------------
# isolation: env, encryption key, network guard
# ---------------------------------------------------------------------------


def _is_local(host: Any) -> bool:
    if isinstance(host, bytes):
        host = host.decode("ascii", "replace")
    text = str(host or "").strip("[]").lower()
    return text in ("", "localhost", "::1", "0.0.0.0") or text.startswith("127.")


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Scratch Library / Output / data / HOME (no real data), HTTP logging off, a Fernet key installed,
    and every non-loopback connection refused (recorded in ``env['blocked']``)."""
    import api_key_encryption

    folders = {}
    for name in ("Library", "Output", "data", "home", "cache"):
        folders[name] = tmp_path / name
        folders[name].mkdir()
    for var, name in (("GLOSSARION_LIBRARY_DIR", "Library"), ("OUTPUT_DIRECTORY", "Output"),
                      ("GLOSSARION_DATA_DIR", "data"), ("HOME", "home"), ("USERPROFILE", "home"),
                      ("APPDATA", "home")):
        monkeypatch.setenv(var, str(folders[name]))
    monkeypatch.setenv("GLOSSARION_HTTP_LOG", "0")
    for var in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(var, raising=False)
    previous_key = getattr(api_key_encryption, "_key_material", None)
    api_key_encryption.set_key_material(Fernet.generate_key())

    blocked: list = []
    original_connect = socket.socket.connect
    original_getaddrinfo = socket.getaddrinfo

    def connect(sock, address):
        if isinstance(address, tuple) and not _is_local(address[0]):
            blocked.append(address)
            raise OSError(errno.ENETUNREACH, f"test: network access to {address[0]} is blocked")
        return original_connect(sock, address)

    def getaddrinfo(host, *args, **kwargs):
        if not _is_local(host):
            blocked.append(host)
            raise socket.gaierror(-2, f"test: DNS lookup of {host} is blocked")
        return original_getaddrinfo(host, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    folders["blocked"] = blocked
    try:
        yield folders
    finally:
        api_key_encryption.set_key_material(previous_key)  # what the session had before (usually None)
    assert not blocked, f"a request left the machine: {blocked}"


def _book(env, name="Novel.epub", data: Optional[bytes] = None, *, where="Library") -> tuple:
    """(workspace, compiled file) inside the scratch Library."""
    workspace = env[where] / "Novel"
    workspace.mkdir(exist_ok=True)
    path = workspace / name
    path.write_bytes(data if data is not None else os.urandom(300_000))
    return str(workspace), str(path)


def _service(env, providers=None, **kwargs) -> sl.ShareLinkService:
    return sl.ShareLinkService(env["data"] / sl.STORE_FILE, allowed_roots=[env["Library"], env["Output"]],
                               cache_dir=env["cache"], providers=providers, **kwargs)


async def _enable(svc, *ids):
    for pid in ids:
        await svc.set_enabled(pid, True, consent=True)


# ---------------------------------------------------------------------------
# fake servers
# ---------------------------------------------------------------------------


class _Quiet(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: Any

    def log_message(self, *args):  # noqa: D401 - silence
        pass

    def _body(self) -> bytes:
        length = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(length) if length else b""

    def _reply(self, status: int, payload: Any = None, headers: Optional[dict] = None) -> None:
        data = b"" if payload is None else (payload if isinstance(payload, bytes) else json.dumps(payload).encode())
        self.send_response(status)
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class _Server:
    def __init__(self, handler: type) -> None:
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.httpd.daemon_threads = True
        self.httpd.fake = self
        self.port = self.httpd.server_address[1]
        self.base = f"http://127.0.0.1:{self.port}"
        self.requests: list = []
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(5)


def _parse_multipart(body: bytes, content_type: str) -> dict:
    boundary = content_type.split("boundary=", 1)[1].encode()
    out = {}
    for part in body.split(b"--" + boundary)[1:]:
        if part.startswith(b"--"):
            break
        head, _, content = part[2:].partition(b"\r\n\r\n")
        if content.endswith(b"\r\n"):
            content = content[:-2]
        text = head.decode("utf-8")
        name = re.search(r'; name="([^"]*)"', text).group(1)
        filename = re.search(r'filename="([^"]*)"', text)
        out[name] = (filename.group(1) if filename else None, content)
    return out


class GofileFake(_Server):
    """``POST /uploadfile`` (multipart ``file``) and ``DELETE /contents``, like gofile.io/api."""

    def __init__(self) -> None:
        self.folders: dict = {}
        self.script: list = []  # queued (status, headers, payload) answers for /uploadfile
        self.hang_after_body = 0.0
        super().__init__(self._handler())

    def _handler(self):
        fake = self

        class Handler(_Quiet):
            def do_POST(self):
                body = self._body()
                auth = self.headers.get("Authorization") or ""
                with fake.lock:
                    fake.requests.append(("POST", self.path, auth, len(body)))
                    scripted = fake.script.pop(0) if fake.script else None
                if fake.hang_after_body:
                    time.sleep(fake.hang_after_body)
                    return
                if scripted is not None:
                    self._reply(*scripted)
                    return
                parts = _parse_multipart(body, self.headers["Content-Type"])
                filename, content = parts["file"]
                token = auth[len("Bearer "):] if auth.startswith("Bearer ") else ""
                created = ""
                if not token:
                    token = created = "guest" + secrets.token_hex(12)
                elif not any(f["token"] == token for f in fake.folders.values()) and token.startswith("bad"):
                    self._reply(401, {"status": "error-auth", "data": {}})
                    return
                folder, code = secrets.token_hex(8), secrets.token_hex(3)
                with fake.lock:
                    fake.folders[folder] = {"token": token, "name": filename, "content": content}
                data = {"downloadPage": f"https://gofile.io/d/{code}", "id": secrets.token_hex(8),
                        "parentFolder": folder, "parentFolderCode": code, "name": filename, "size": len(content)}
                if created:
                    data["guestToken"] = created
                self._reply(200, {"status": "ok", "data": data})

            def do_DELETE(self):
                body = json.loads(self._body() or b"{}")
                auth = self.headers.get("Authorization") or ""
                with fake.lock:
                    fake.requests.append(("DELETE", self.path, auth, body))
                folder = fake.folders.get(body.get("contentsId"))
                if folder is None:
                    self._reply(404, {"status": "error-notFound", "data": {}})
                elif auth != f"Bearer {folder['token']}":
                    self._reply(401, {"status": "error-auth", "data": {}})
                else:
                    del fake.folders[body["contentsId"]]
                    self._reply(200, {"status": "ok", "data": {}})

        return Handler

    def provider(self, **kwargs) -> gf.GofileProvider:
        return gf.GofileProvider(sp.HttpClient(**kwargs), api_base=self.base, upload_url=self.base + "/uploadfile")


class PixeldrainFake(_Server):
    """``PUT /api/file/<name>``, ``DELETE /api/file/<id>``, ``GET /api/user`` with Basic ``:<key>``."""

    def __init__(self, keys=("pd-key-0123456789",)) -> None:
        self.keys = set(keys)
        self.files: dict = {}
        self.gate: Optional[threading.Event] = None  # read the body only after it is set
        self.started = threading.Event()
        self.early_close = False  # refuse a bad key at once and close without reading the body
        super().__init__(self._handler())

    def _key(self, handler) -> Optional[str]:
        auth = handler.headers.get("Authorization") or ""
        if not auth.startswith("Basic "):
            return None
        user, _, key = base64.b64decode(auth[6:]).decode().partition(":")
        return key if not user and key in self.keys else None

    def _handler(self):
        fake = self

        class Handler(_Quiet):
            def do_PUT(self):
                key = fake._key(self)
                with fake.lock:
                    fake.requests.append(("PUT", self.path, self.headers.get("Authorization")))
                if key is None:
                    if fake.early_close:
                        self.close_connection = True
                        self._reply(401, {"success": False, "value": "authentication_required"},
                                    {"Connection": "close"})
                        return
                    self._body()
                    self._reply(401, {"success": False, "value": "authentication_required"})
                    return
                length = int(self.headers.get("Content-Length") or 0)
                received = bytearray(self.rfile.read(min(length, 65536)))
                fake.started.set()
                if fake.gate is not None and not fake.gate.wait(30):
                    return
                try:
                    while len(received) < length:
                        chunk = self.rfile.read(min(65536, length - len(received)))
                        if not chunk:
                            return
                        received += chunk
                except OSError:
                    return
                file_id = secrets.token_hex(4)
                with fake.lock:
                    fake.files[file_id] = {"key": key, "name": unquote(self.path.rsplit("/", 1)[1]),
                                           "content": bytes(received), "type": self.headers.get("Content-Type")}
                self._reply(201, {"success": True, "id": file_id})

            def do_DELETE(self):
                key = fake._key(self)
                file_id = self.path.rsplit("/", 1)[1]
                with fake.lock:
                    fake.requests.append(("DELETE", self.path, self.headers.get("Authorization")))
                entry = fake.files.get(file_id)
                if key is None or (entry is not None and entry["key"] != key):
                    self._reply(401, {"success": False, "value": "unauthorized"})
                elif entry is None:
                    self._reply(404, {"success": False, "value": "not_found"})
                else:
                    del fake.files[file_id]
                    self._reply(200, {"success": True, "value": "file_deleted"})

            def do_GET(self):
                if self.path == "/api/user":
                    self._reply(200 if fake._key(self) else 401, {"username": "u"} if fake._key(self) else
                                {"success": False, "value": "authentication_required"})
                else:
                    self._reply(404, {"success": False, "value": "not_found"})

        return Handler

    def provider(self, **kwargs) -> pd.PixeldrainProvider:
        return pd.PixeldrainProvider(sp.HttpClient(**kwargs), api_base=self.base + "/api")


class SendFake:
    """timvisee/send: the ``/api/ws`` upload (server/routes/ws.js) on a websockets server, and the
    HTTP side - ``POST /api/delete/<id>`` (owner token), ``GET /api/metadata/<id>`` and
    ``GET /api/download/<id>`` behind the ``send-v1`` HMAC nonce check (server/middleware/auth.js)."""

    MAX_EXPIRE = 259200
    MAX_DOWNLOADS = 20

    def __init__(self) -> None:
        from websockets.sync.server import serve

        self.files: dict = {}
        self.raw: list = []  # every frame the server received
        self.uploads: list = []  # the JSON headers
        self.error: Optional[int] = None
        self.http = _Server(self._http_handler())
        self.public = self.http.base
        self.ws = serve(self._ws_handler, "127.0.0.1", 0, compression=None)
        self.ws_port = self.ws.socket.getsockname()[1]
        self.ws_thread = threading.Thread(target=self.ws.serve_forever, daemon=True)
        self.ws_thread.start()

    def close(self) -> None:
        self.ws.shutdown()
        self.ws_thread.join(5)
        self.http.close()

    def _ws_handler(self, ws) -> None:
        message = ws.recv()
        self.raw.append(message)
        info = json.loads(message)
        self.uploads.append(info)
        auth = str(info.get("authorization") or "")
        time_limit, dlimit = info.get("timeLimit") or 86400, info.get("dlimit") or 1
        if self.error or not info.get("fileMetadata") or not auth.startswith("send-v1 ") or time_limit <= 0 \
                or time_limit > self.MAX_EXPIRE or dlimit > self.MAX_DOWNLOADS:
            ws.send(json.dumps({"error": self.error or 400}))
            return
        file_id, owner = secrets.token_hex(8), secrets.token_hex(10)
        ws.send(json.dumps({"url": f"{self.public}/download/{file_id}/", "ownerToken": owner, "id": file_id}))
        data = bytearray()
        while True:
            chunk = ws.recv()
            self.raw.append(chunk)
            if isinstance(chunk, str):
                ws.send(json.dumps({"error": 400}))
                return
            if chunk == b"\x00":
                break
            data += chunk
        self.files[file_id] = {"owner": owner, "metadata": info["fileMetadata"], "auth": auth.split(" ", 1)[1],
                               "nonce": base64.b64encode(os.urandom(16)).decode(), "data": bytes(data),
                               "dlimit": dlimit, "time_limit": time_limit}
        ws.send(json.dumps({"ok": True}))

    def _http_handler(self):
        fake = self

        class Handler(_Quiet):
            def _hmac_ok(self, entry) -> bool:
                header = self.headers.get("Authorization") or ""
                expected = hmac.new(_b64d(entry["auth"]), base64.b64decode(entry["nonce"]), "sha256").digest()
                ok = header.startswith("send-v1 ") and hmac.compare_digest(expected, _b64d(header.split(" ", 1)[1]))
                if ok:
                    entry["nonce"] = base64.b64encode(os.urandom(16)).decode()
                return ok

            def do_GET(self):
                match = re.match(r"^/api/(metadata|download)/([0-9a-f]+)$", self.path)
                entry = fake.files.get(match.group(2)) if match else None
                if entry is None:
                    self._reply(404)
                    return
                if not self._hmac_ok(entry):
                    self._reply(401, headers={"WWW-Authenticate": f"send-v1 {entry['nonce']}"})
                    return
                nonce = {"WWW-Authenticate": f"send-v1 {entry['nonce']}"}
                if match.group(1) == "metadata":
                    self._reply(200, {"metadata": entry["metadata"], "finalDownload": False, "ttl": 1000}, nonce)
                else:
                    self._reply(200, entry["data"], nonce)

            def do_POST(self):
                body = json.loads(self._body() or b"{}")
                match = re.match(r"^/api/delete/([0-9a-f]+)$", self.path)
                entry = fake.files.get(match.group(1)) if match else None
                if entry is None:
                    self._reply(404)
                elif body.get("owner_token") != entry["owner"]:
                    self._reply(401)
                else:
                    del fake.files[match.group(1)]
                    self._reply(200)

        return Handler

    def provider(self, **kwargs) -> se.SendProvider:
        return se.SendProvider(sp.HttpClient(), base_url=self.public, ws_url=f"ws://127.0.0.1:{self.ws_port}/api/ws",
                               **kwargs)


@pytest.fixture
def gofile():
    fake = GofileFake()
    yield fake
    fake.close()


@pytest.fixture
def pixeldrain():
    fake = PixeldrainFake()
    yield fake
    if fake.gate is not None:
        fake.gate.set()
    fake.close()


@pytest.fixture
def send():
    fake = SendFake()
    yield fake
    fake.close()


# ---------------------------------------------------------------------------
# independent reference implementation of the Send v3 recipient (timvisee/send)
# ---------------------------------------------------------------------------


def _b64d(text: str) -> bytes:
    text = text.replace("-", "+").replace("_", "/")
    return base64.b64decode(text + "=" * (-len(text) % 4))


def _b64u(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


def _ref_hkdf(ikm: bytes, length: int, info: bytes, salt: Optional[bytes] = None) -> bytes:
    return HKDF(algorithm=hashes.SHA256(), length=length, salt=salt, info=info).derive(ikm)


def ref_ece_decrypt(body: bytes, ikm: bytes) -> bytes:
    """RFC 8188 aes128gcm decoding (96-bit nonce XOR, padding = zeros after the 1/2 delimiter)."""
    salt, rs, idlen = body[:16], int.from_bytes(body[16:20], "big"), body[20]
    key = _ref_hkdf(ikm, 16, b"Content-Encoding: aes128gcm\x00", salt)
    base = int.from_bytes(_ref_hkdf(ikm, 12, b"Content-Encoding: nonce\x00", salt), "big")
    records = [body[i:i + rs] for i in range(21 + idlen, len(body), rs)]
    out = bytearray()
    for seq, record in enumerate(records):
        plain = AESGCM(key).decrypt((base ^ seq).to_bytes(12, "big"), record, None)
        stripped = plain.rstrip(b"\x00")
        assert stripped[-1] == (2 if seq == len(records) - 1 else 1), "wrong record delimiter"
        out += stripped[:-1]
    return bytes(out)


def _ref_get(url: str, auth_key: bytes, nonce: str) -> tuple:
    import http.client
    from urllib.parse import urlsplit

    parts = urlsplit(url)
    for _ in range(2):  # fetchWithAuthAndRetry: a 401 with a new nonce is retried once
        sig = hmac.new(auth_key, base64.b64decode(nonce), "sha256").digest()
        conn = http.client.HTTPConnection(parts.hostname, parts.port, timeout=10)
        conn.request("GET", parts.path, headers={"Authorization": f"send-v1 {_b64u(sig)}"})
        response = conn.getresponse()
        data = response.read()
        conn.close()
        new_nonce = (response.getheader("WWW-Authenticate") or " ").split(" ", 1)[1]
        if response.status == 401 and new_nonce != nonce:
            nonce = new_nonce
            continue
        return response.status, data, new_nonce
    return response.status, data, nonce


def ref_receive(link: str) -> tuple:
    """What a recipient's browser does with the link: (metadata dict, file bytes)."""
    url, _, key = link.partition("#")
    secret = _b64d(key)
    assert len(secret) == 16
    file_id = url.rstrip("/").rsplit("/", 1)[1]
    base = url.split("/download/", 1)[0]
    auth_key = _ref_hkdf(secret, 64, b"authentication", b"")  # WebCrypto HMAC-SHA256 default: 512 bits
    meta_key = _ref_hkdf(secret, 16, b"metadata", b"")
    status, data, nonce = _ref_get(f"{base}/api/metadata/{file_id}", auth_key, "yRCdyQ1EMSA3mo4rqSkuNQ==")
    assert status == 200, status
    metadata = json.loads(AESGCM(meta_key).decrypt(bytes(12), _b64d(json.loads(data)["metadata"]), None))
    status, data, _ = _ref_get(f"{base}/api/download/{file_id}", auth_key, nonce)
    assert status == 200, status
    return metadata, ref_ece_decrypt(data, secret)


# RFC 8188 section 3.1: "I am the walrus", rs 4096, one record.
RFC_IKM = "yqdlZ-tYemfogSmv7Ws5PQ"
RFC_BODY = "I1BsxtFttlv3u_Oo94xnmwAAEAAA-NAVub2qFgBEuQKRapoZu-IxkIva3MEB1PD-ly8Thjg"


def test_ece_matches_the_rfc_8188_example_vector():
    body, ikm = _b64d(RFC_BODY), _b64d(RFC_IKM)
    assert ref_ece_decrypt(body, ikm) == b"I am the walrus"
    plain = b"I am the walrus"
    chunks = list(se.ece_encrypt(_reader(plain), len(plain), ikm, salt=body[:16], rs=4096))
    assert b"".join(c for c, _ in chunks) == body
    assert se.encrypted_size(len(plain), 4096) == len(body)


def _reader(data: bytes):
    view = memoryview(data)
    pos = [0]

    def read(n: int) -> bytes:
        out = bytes(view[pos[0]:pos[0] + n])
        pos[0] += len(out)
        return out

    return read


@pytest.mark.parametrize("size", [1, 65519, 65520, 3 * 65519, 200_001])
def test_ece_records_round_trip_at_record_boundaries(size):
    data = os.urandom(size)
    ikm = os.urandom(16)
    chunks = list(se.ece_encrypt(_reader(data), size, ikm))
    body = b"".join(c for c, _ in chunks)
    assert len(body) == se.encrypted_size(size)
    assert chunks[0][0][16:21] == (65536).to_bytes(4, "big") + b"\x00"
    assert all(len(c) == 65536 for c, _ in chunks[1:-1])
    assert chunks[-1][1] == size
    assert ref_ece_decrypt(body, ikm) == data


def test_keychain_and_metadata_follow_the_v3_client():
    secret = bytes(range(16))
    keys = se.Keychain(secret)
    assert keys.auth_key == _ref_hkdf(secret, 64, b"authentication", b"") and len(keys.auth_key) == 64
    assert keys.meta_key == _ref_hkdf(secret, 16, b"metadata", b"")
    assert keys.authorization() == "send-v1 " + _b64u(keys.auth_key)
    assert keys.secret_b64 == _b64u(secret) and "=" not in keys.secret_b64
    meta = AESGCM(keys.meta_key).decrypt(bytes(12), keys.encrypt_metadata("책 1.epub", 5, "application/epub+zip"), None)
    assert meta == ('{"name":"책 1.epub","size":5,"type":"application/epub+zip","manifest":{"files":'
                    '[{"name":"책 1.epub","size":5,"type":"application/epub+zip"}]}}').encode("utf-8")


# ---------------------------------------------------------------------------
# defaults, consent, eligibility
# ---------------------------------------------------------------------------


def test_every_provider_is_off_by_default_and_nothing_is_sent(env, gofile):
    svc = _service(env, providers={"gofile": gofile.provider()})
    workspace, path = _book(env)

    async def run():
        await svc.load()
        states = svc.provider_states()
        assert [s.id for s in states] == ["transferit", "gofile", "send", "pixeldrain"]
        assert all(not s.enabled and not s.consented and not s.ready for s in states)
        with pytest.raises(sp.ShareError) as info:
            await svc.upload("gofile", path, book=workspace)
        assert info.value.code == "disabled"
        with pytest.raises(sp.ShareError) as info:
            await svc.start_handoff(path, book=workspace)
        assert info.value.code == "disabled"
        # enabled without the consent sheet: still refused
        await svc.set_enabled("gofile", True)
        assert svc.provider_state("gofile").reason and not svc.provider_state("gofile").consented
        with pytest.raises(sp.ShareError) as info:
            await svc.upload("gofile", path, book=workspace)
        assert info.value.code == "consent"
        assert [r for (_, r) in svc.menu(path)] == [
            "Turn it on in Settings › Cloud sync & sharing", "Read and accept what this service can see first",
            "Turn it on in Settings › Cloud sync & sharing", "Turn it on in Settings › Cloud sync & sharing"]

    asyncio.run(run())
    assert gofile.requests == []
    assert svc.state.phase in ("idle", "failed")


def test_a_new_consent_version_asks_again(env, monkeypatch):
    svc = _service(env)

    async def run():
        await _enable(svc, "send")
        assert svc.provider_state("send").ready
        bumped = dataclasses.replace(sp._INFO["send"], consent_version=2)
        monkeypatch.setitem(sp._INFO, "send", bumped)
        state = svc.provider_state("send")
        assert state.enabled and not state.consented and not state.ready
        assert svc.consent_text("send").version == 2
        await svc.give_consent("send")
        assert svc.provider_state("send").ready
        await svc.revoke_consent("send")
        assert not svc.provider_state("send").enabled and not svc.provider_state("send").consented

    asyncio.run(run())


def test_consent_texts_are_neutral_and_say_who_can_read_the_file():
    for pid in sp.PROVIDER_IDS:
        text = sp.consent_text(pid, file_name="Book.epub", size=4_800_000)
        joined = " ".join(text.lines)
        assert text.title.startswith("Share file via link")
        assert "Book.epub" in joined and "Anyone who has the link" in joined
        assert "IP address" in joined and "developers receive nothing" in joined
        assert "right to share" in joined + text.checkbox
        info = sp.provider_info(pid)
        if info.e2ee:
            assert "cannot read the file" in joined and "End-to-end encrypted" in joined
        else:
            assert f"Not end-to-end encrypted: {info.label} can read the file." in joined
        for word in ("novel", "translation", "pirat"):
            assert word not in joined.lower()
    assert "sends nothing to transfer.it" in " ".join(sp.consent_text("transferit").lines)
    assert "your own pixeldrain account" in " ".join(sp.consent_text("pixeldrain").lines)


def test_only_book_outputs_inside_the_library_or_output_qualify(env, tmp_path):
    svc = _service(env)
    _, epub = _book(env)
    keys = env["Library"] / "Novel" / "api_keys_export.json"
    keys.write_text("{}")
    outside = tmp_path / "elsewhere.epub"
    outside.write_bytes(b"x")
    assert svc.eligible(epub)
    assert not svc.eligible(str(keys)) and not svc.eligible(str(env["data"] / "config.json"))
    assert not svc.eligible(str(outside))
    assert all(reason == "Only book files from the Library can be shared by link" for _, reason in svc.menu(str(keys)))


def test_the_generic_export_sheet_never_offers_a_link(env):
    """Critic: ``FileBridge.export_options`` also serves the plaintext API-key export (Keys screen), so
    "Share file via link" must never become one of its options; it lives on the book surfaces only."""
    from glossarion_mobile.services.files import FileBridge

    keys_export = env["data"] / "glossarion-api-keys.json"
    keys_export.write_text("{}")
    svc = _service(env)
    for platform in ("android", "ios", "desktop"):
        bridge = FileBridge(inbox_dir=str(env["data"] / "Inbox"), platform=platform,
                            files_visible_root=str(env["data"]))
        ids = [option.id for option in bridge.export_options(str(keys_export))]
        assert not [i for i in ids if "link" in i or i in sp.PROVIDER_IDS], ids
    assert not svc.eligible(str(keys_export))


def test_size_checks_refuse_before_any_request(env, send, monkeypatch):
    svc = _service(env, providers={"send": send.provider()})
    workspace, path = _book(env, data=os.urandom(5000))
    monkeypatch.setitem(sp._INFO, "send", dataclasses.replace(sp._INFO["send"], max_bytes=4096))

    async def run():
        await _enable(svc, "send")
        pre = await svc.preflight("send", path)
        assert pre.blocked == "too_large" and "up to 4.0 KB" in pre.message
        with pytest.raises(sp.ShareError) as info:
            await svc.upload("send", path, book=workspace)
        assert info.value.code == "too_large"
        empty = Path(workspace) / "Empty.epub"
        empty.write_bytes(b"")
        with pytest.raises(sp.ShareError) as info:
            await svc.upload("send", str(empty), book=workspace)
        assert info.value.code == "empty"

    asyncio.run(run())
    assert send.uploads == []


# ---------------------------------------------------------------------------
# Gofile
# ---------------------------------------------------------------------------


def test_gofile_upload_reuses_one_guest_account_and_deletes_the_folder(env, gofile, caplog):
    caplog.set_level(logging.DEBUG)
    svc = _service(env, providers={"gofile": gofile.provider()})
    workspace, path = _book(env, name="책 one (1).epub")
    seen: list = []
    unsubscribe = svc.subscribe(lambda kind: seen.append((kind, svc.state.phase, svc.state.sent, svc.state.total)))

    async def run():
        await _enable(svc, "gofile")
        first = await svc.upload("gofile", path, book=workspace)
        unsubscribe()
        second = await svc.upload("gofile", path, book=workspace)
        return first, second

    first, second = asyncio.run(run())
    assert first.url.startswith("https://gofile.io/d/") and first.can_delete and not first.e2ee
    assert first.name == "책 one (1).epub" and first.size == os.path.getsize(path)
    # one guest account: the first upload had no token, the second used the one Gofile returned
    (_, _, auth1, _), (_, _, auth2, _) = gofile.requests
    assert auth1 == "" and auth2.startswith("Bearer guest")
    tokens = {f["token"] for f in gofile.folders.values()}
    assert len(tokens) == 1 and auth2 == f"Bearer {tokens.pop()}"
    assert {f["name"] for f in gofile.folders.values()} == {"책 one (1).epub"}
    assert all(f["content"] == Path(path).read_bytes() for f in gofile.folders.values())
    # progress went up to the size, then the link
    uploads = [s for s in seen if s[0] == "upload"]
    assert ("upload", "preparing", 0, 0) in uploads and uploads[-1][1] == "done"
    sizes = [s[2] for s in uploads if s[1] in ("uploading", "finishing")]
    assert sizes == sorted(sizes) and sizes[-1] == os.path.getsize(path)

    raw = (env["data"] / sl.STORE_FILE).read_text("utf-8")
    store = json.loads(raw)
    assert store["secrets"]["gofile_token"].startswith("ENC:")
    assert all(r["url"].startswith("ENC:") and r["delete"].startswith("ENC:") for r in store["links"])
    for secret in [auth2[7:], first.url, second.url, first.url.rsplit("/", 1)[1]]:
        assert secret not in raw and secret not in caplog.text

    async def delete():
        links = await svc.links_for(book=workspace)
        assert [link.id for link in links] == [second.id, first.id]
        result = await svc.delete_link(first.id)
        assert result == sl.DeleteResult(True, "deleted")
        gone = await svc.delete_link(first.id)
        assert gone.removed is False
        return await svc.links_for(book=workspace)

    left = asyncio.run(delete())
    assert [link.id for link in left] == [second.id]
    assert len(gofile.folders) == 1
    method, route, auth, body = gofile.requests[-1]
    assert (method, route) == ("DELETE", "/contents") and auth == auth2 and set(body) == {"contentsId"}


def test_multipart_body_escapes_the_file_name_like_browsers(tmp_path):
    path = tmp_path / "f.bin"
    path.write_bytes(b"abc")
    content_type, length, make = sp.multipart_file_body("file", 'a"b\r\n책.epub', "application/epub+zip",
                                                        str(path), 3)
    body = b"".join(make())
    assert len(body) == length and body == b"".join(make())  # a retry rebuilds the same body
    assert _parse_multipart(body, content_type)["file"] == ("a%22b%0D%0A책.epub", b"abc")
    assert 'filename="a%22b%0D%0A책.epub"'.encode("utf-8") in body


def test_gofile_429_is_retried_once_only_with_a_short_retry_after(env, gofile):
    svc = _service(env, providers={"gofile": gofile.provider()})
    workspace, path = _book(env, data=os.urandom(1000))
    busy = (429, {"status": "error-rateLimit", "data": {}})

    async def attempt():
        try:
            await svc.upload("gofile", path, book=workspace)
            return "ok"
        except sp.ShareError as exc:
            return exc

    async def run():
        await _enable(svc, "gofile")
        out = []
        gofile.script[:] = [(*busy, {"Retry-After": "0"})]  # one 429, then success
        out.append((await attempt(), len(gofile.requests)))
        gofile.script[:] = [(*busy, {"Retry-After": "0"}), (*busy, {"Retry-After": "0"})]  # never a 2nd retry
        out.append((await attempt(), len(gofile.requests)))
        gofile.script[:] = [(*busy, {})]  # no Retry-After: no retry
        out.append((await attempt(), len(gofile.requests)))
        gofile.script[:] = [(*busy, {"Retry-After": "3600"})]  # too long to wait: no retry
        out.append((await attempt(), len(gofile.requests)))
        return out

    (ok, n1), (twice, n2), (none, n3), (long_wait, n4) = asyncio.run(run())
    assert ok == "ok" and n1 == 2
    assert twice.code == "rate_limited" and n2 - n1 == 2
    assert none.code == "rate_limited" and n3 - n2 == 1 and "later" in none.message
    assert long_wait.code == "rate_limited" and n4 - n3 == 1 and "60 min" in long_wait.message
    assert long_wait.retry_after == 3600


def test_no_retry_after_a_timeout_once_the_body_was_sent(env, gofile):
    svc = _service(env, providers={"gofile": gofile.provider(io_timeout=0.5)})
    workspace, path = _book(env, data=os.urandom(2000))
    gofile.hang_after_body = 2.0

    async def run():
        await _enable(svc, "gofile")
        with pytest.raises(sp.ShareError) as info:
            await svc.upload("gofile", path, book=workspace)
        return info.value

    error = asyncio.run(run())
    assert error.code == "maybe_uploaded" and error.maybe_uploaded and "second copy" in error.message
    assert len(gofile.requests) == 1
    assert svc.state.phase == "failed" and svc.state.error_code == "maybe_uploaded"


def test_gofile_rejected_token_is_dropped_so_retry_starts_a_new_guest_account(env, gofile):
    svc = _service(env, providers={"gofile": gofile.provider()})
    workspace, path = _book(env, data=os.urandom(1000))

    async def run():
        await _enable(svc, "gofile")
        await svc.load()
        token = svc.secrets.seal("bad-token")
        await svc._io(svc.store.mutate, lambda d: d.setdefault("secrets", {}).update(gofile_token=token))
        with pytest.raises(sp.ShareError) as info:
            await svc.upload("gofile", path, book=workspace)
        assert info.value.code == "auth"
        assert "gofile_token" not in svc.store.read()["secrets"]
        link = await svc.upload("gofile", path, book=workspace)
        assert link.url.startswith("https://gofile.io/d/")

    asyncio.run(run())
    assert gofile.requests[0][2] == "Bearer bad-token" and gofile.requests[1][2] == ""


# ---------------------------------------------------------------------------
# pixeldrain
# ---------------------------------------------------------------------------


def test_pixeldrain_uses_the_users_key_stored_encrypted(env, pixeldrain, caplog):
    caplog.set_level(logging.DEBUG)
    svc = _service(env, providers={"pixeldrain": pixeldrain.provider()})
    workspace, path = _book(env, name="Novel (1).epub")

    async def run():
        await _enable(svc, "pixeldrain")
        assert svc.provider_state("pixeldrain").reason == \
            "Add your pixeldrain API key in Settings › Cloud sync & sharing"
        with pytest.raises(sp.ShareError) as info:
            await svc.upload("pixeldrain", path, book=workspace)
        assert info.value.code == "no_key"
        with pytest.raises(sp.ShareError):
            await svc.set_pixeldrain_key("  no  ")
        await svc.set_pixeldrain_key("  wrong-key-000000  ")
        assert await svc.check_pixeldrain_key() is False
        with pytest.raises(sp.ShareError) as info:
            await svc.upload("pixeldrain", path, book=workspace)
        assert info.value.code == "auth"
        await svc.set_pixeldrain_key("pd-key-0123456789")
        assert await svc.check_pixeldrain_key() is True
        link = await svc.upload("pixeldrain", path, book=workspace)
        return link

    link = asyncio.run(run())
    file_id = link.url.rsplit("/", 1)[1]
    assert link.url == f"https://pixeldrain.com/u/{file_id}" and link.can_delete
    entry = pixeldrain.files[file_id]
    assert entry["name"] == "Novel (1).epub" and entry["content"] == Path(path).read_bytes()
    assert entry["type"] == "application/epub+zip"
    raw = (env["data"] / sl.STORE_FILE).read_text("utf-8")
    assert "pd-key-0123456789" not in raw and "wrong-key" not in raw and file_id not in raw
    assert json.loads(raw)["secrets"]["pixeldrain_key"].startswith("ENC:")
    assert "pd-key-0123456789" not in caplog.text and link.url not in caplog.text

    async def delete():
        await svc.delete_link(link.id)
        return await svc.links_for(book=workspace)

    assert asyncio.run(delete()) == [] and pixeldrain.files == {}


def test_a_refused_key_is_reported_even_when_the_server_cuts_the_upload(env, pixeldrain):
    """A server that answers 401 and closes mid-body can lose its answer to the connection reset: the
    provider then asks pixeldrain once whether the key works (no upload retry)."""
    svc = _service(env, providers={"pixeldrain": pixeldrain.provider()})
    workspace, path = _book(env, data=os.urandom(4 * 1024 * 1024))
    pixeldrain.early_close = True

    async def run():
        await _enable(svc, "pixeldrain")
        await svc.set_pixeldrain_key("wrong-key-000000")
        with pytest.raises(sp.ShareError) as info:
            await svc.upload("pixeldrain", path, book=workspace)
        return info.value

    assert asyncio.run(run()).code == "auth"
    puts = [r for r in pixeldrain.requests if r[0] == "PUT"]
    assert len(puts) == 1 and pixeldrain.files == {}


def test_cancel_stops_a_running_upload_and_cleans_up(env, pixeldrain):
    svc = _service(env, providers={"pixeldrain": pixeldrain.provider()})
    workspace, path = _book(env, data=os.urandom(8 * 1024 * 1024))
    pixeldrain.gate = threading.Event()

    async def run():
        await _enable(svc, "pixeldrain")
        await svc.set_pixeldrain_key("pd-key-0123456789")
        task = asyncio.ensure_future(svc.upload("pixeldrain", path, book=workspace))
        deadline = time.monotonic() + 30
        while not (pixeldrain.started.is_set() and svc.state.phase in ("uploading", "finishing")):
            assert time.monotonic() < deadline and not task.done()
            await asyncio.sleep(0.01)
        with pytest.raises(sp.ShareError) as info:
            await svc.upload("pixeldrain", path, book=workspace)
        assert info.value.code == "busy"
        assert svc.cancel() is True
        with pytest.raises(sp.ShareError) as info:
            await asyncio.wait_for(task, 30)
        return info.value

    error = asyncio.run(run())
    pixeldrain.gate.set()
    assert error.code == "cancelled" and svc.state.phase == "cancelled" and not svc.uploading
    assert pixeldrain.files == {} and json.loads((env["data"] / sl.STORE_FILE).read_text("utf-8"))["links"] == []
    assert not any((env["cache"] / sl.SNAPSHOT_DIR).iterdir())


# ---------------------------------------------------------------------------
# Send (end-to-end encrypted)
# ---------------------------------------------------------------------------


def test_send_link_decrypts_with_the_reference_algorithm_and_the_key_never_leaves(env, send, caplog):
    caplog.set_level(logging.DEBUG)
    svc = _service(env, providers={"send": send.provider()})
    data = os.urandom(200_001)  # 4 records
    workspace, path = _book(env, name="번역 Novel.epub", data=data)

    async def run():
        await _enable(svc, "send")
        assert svc.send_options() == {"expire": 259200, "downloads": 20}
        return await svc.upload("send", path, book=workspace)

    before = time.time()
    link = asyncio.run(run())
    url, _, key = link.url.partition("#")
    assert re.match(rf"^{re.escape(send.public)}/download/[0-9a-f]{{16}}/$", url) and len(key) == 22
    assert link.e2ee and link.can_delete and link.downloads_limit == 20
    assert before + 259200 <= link.expires <= time.time() + 259200
    upload = send.uploads[0]
    assert upload["timeLimit"] == 259200 and upload["dlimit"] == 20
    assert set(upload) == {"fileMetadata", "authorization", "timeLimit", "dlimit"}
    # the server got ciphertext only: no plaintext, no secret, no file name
    secret = _b64d(key)
    received = b"".join(m if isinstance(m, bytes) else m.encode() for m in send.raw)
    for needle in (secret, key.encode(), data[1000:1032], "번역".encode(), b"Novel"):
        assert needle not in received
    stored = next(iter(send.files.values()))
    assert len(stored["data"]) == se.encrypted_size(len(data))
    metadata, plain = ref_receive(link.url)
    assert plain == data
    assert metadata == {"name": "번역 Novel.epub", "size": len(data), "type": "application/epub+zip",
                        "manifest": {"files": [{"name": "번역 Novel.epub", "size": len(data),
                                                "type": "application/epub+zip"}]}}
    raw = (env["data"] / sl.STORE_FILE).read_text("utf-8")
    assert key not in raw and url not in raw and key not in caplog.text

    async def options_and_delete():
        await svc.set_send_options(expire=3600, downloads=5)
        await svc.set_send_options(expire=10 ** 9, downloads=999)  # clamped to the instance limits
        clamped = svc.send_options()
        await svc.set_send_options(expire=3600, downloads=5)
        second = await svc.upload("send", path, book=workspace)
        result = await svc.delete_link(link.id)
        return clamped, second, result

    clamped, second, result = asyncio.run(options_and_delete())
    assert clamped == {"expire": 259200, "downloads": 20}
    assert send.uploads[1]["timeLimit"] == 3600 and send.uploads[1]["dlimit"] == 5
    assert second.downloads_limit == 5 and result == sl.DeleteResult(True, "deleted")
    assert list(send.files) == [second.url.split("/download/")[1].split("/")[0]]


def test_send_server_errors_are_mapped_and_nothing_is_saved(env, send):
    svc = _service(env, providers={"send": send.provider()})
    workspace, path = _book(env, data=os.urandom(1000))
    send.error = 413

    async def run():
        await _enable(svc, "send")
        with pytest.raises(sp.ShareError) as info:
            await svc.upload("send", path, book=workspace)
        return info.value

    assert asyncio.run(run()).code == "too_large"
    assert json.loads((env["data"] / sl.STORE_FILE).read_text("utf-8"))["links"] == []


def test_plain_http_or_ws_is_refused_except_on_loopback():
    with pytest.raises(sp.ShareError) as info:
        sp.HttpClient().request("GET", "http://example.com/api")
    assert info.value.code == "protocol"
    provider = se.SendProvider(ws_url="ws://example.com/api/ws")
    with pytest.raises(sp.ShareError) as info:
        provider._check_url()
    assert info.value.code == "protocol"
    assert se.SendProvider().ws_url == "wss://send.vis.ee/api/ws"
    assert gf.GofileProvider().upload_url == "https://upload.gofile.io/uploadfile"
    assert pd.PixeldrainProvider().api_base == "https://pixeldrain.com/api"


# ---------------------------------------------------------------------------
# transfer.it handoff
# ---------------------------------------------------------------------------


def test_transferit_is_a_browser_handoff_with_no_request(env):
    opened, saved = [], []
    entries: dict = {}

    async def save_to_downloads(path, *, replace_uri=None, entry=False):
        """FileBridge.save_to_downloads on a phone where Downloads/Glossarion already holds a "Novel.epub" of
        another app: MediaStore names a new entry "Novel (1).epub"; an entry of ours is overwritten in place."""
        saved.append((path, replace_uri, entry))
        if replace_uri in entries:
            uri = replace_uri
        else:
            uri = f"content://media/external/downloads/{42 + len(entries)}"
            entries[uri] = "Novel (1).epub"
        return {"uri": uri, "name": entries[uri]} if entry else uri

    svc = _service(env, platform="android", open_url=opened.append, save_to_downloads=save_to_downloads)
    workspace, path = _book(env)

    async def run():
        await _enable(svc, "transferit")
        plan = await svc.start_handoff(path, book=workspace)
        with pytest.raises(sp.ShareError) as info:
            await svc.add_pasted_link("https://evil.example/t/abcdef")
        assert info.value.code == "bad_link"
        link = await svc.add_pasted_link("Here it is: https://transfer.it/t/AbC_d-123456 thanks")
        listed = await svc.links_for(book=workspace)
        with pytest.raises(sp.ShareError) as info:
            await svc.delete_link(link.id)
        assert info.value.code == "unsupported"
        forgot = await svc.forget_link(link.id)
        again = await svc.start_handoff(path, book=workspace)  # a recompiled book handed off again
        return plan, link, listed, forgot, await svc.links_for(book=workspace), again

    plan, link, listed, forgot, after, again = asyncio.run(run())
    first_uri = "content://media/external/downloads/42"
    # the second handoff overwrites the entry the first one made (never "Novel (2).epub", never the old copy)
    assert opened == [th.START_URL, th.START_URL]
    assert saved == [(path, None, True), (path, first_uri, True)]
    assert plan.location == "Downloads › Glossarion › Novel (1).epub" and plan.saved_to == first_uri
    assert again.saved_to == first_uri and again.location == plan.location and len(entries) == 1
    raw = json.loads((env["data"] / sl.STORE_FILE).read_text(encoding="utf-8"))
    assert first_uri not in json.dumps(raw)  # the entry is kept encrypted like the links
    assert "Keep the page open" in plan.hint
    assert link.url == "https://transfer.it/t/AbC_d-123456" and not link.can_delete and link.provider == "transferit"
    assert [x.id for x in listed] == [link.id] and forgot and after == []


def test_transferit_on_ios_shows_the_files_path_without_copying(env):
    opened = []
    svc = _service(env, platform="ios", open_url=opened.append, files_visible_root=str(env["Library"].parent))
    workspace, path = _book(env)

    async def run():
        await _enable(svc, "transferit")
        return await svc.start_handoff(path, book=workspace)

    plan = asyncio.run(run())
    assert plan.location == "On My iPhone › Glossarion › Library › Novel › Novel.epub" and plan.show_in_files
    assert opened == [th.START_URL] and plan.saved_to is None
    assert sorted(p.name for p in (env["Library"] / "Novel").iterdir()) == ["Novel.epub"]


def test_parse_link_accepts_only_transfer_it_transfer_links():
    assert th.parse_link(" https://transfer.it/t/abcDEF123/ ") == "https://transfer.it/t/abcDEF123"
    for bad in ("", "transfer.it/t/abcdef", "http://transfer.it/t/abcdef", "https://transfer.it/start",
                "https://transfer.it.evil.com/t/abcdef", "https://mega.nz/file/abc#key", "https://transfer.it/t/ab"):
        with pytest.raises(sp.ShareError):
            th.parse_link(bad)


# ---------------------------------------------------------------------------
# records, lifecycle, secrets
# ---------------------------------------------------------------------------


def test_records_follow_a_book_into_the_library_and_go_with_it(env, gofile):
    svc = _service(env, providers={"gofile": gofile.provider()})
    attachments, path = _book(env, where="Output")
    target = env["Library"] / "Novel"

    async def run():
        await _enable(svc, "gofile")
        link = await svc.upload("gofile", path, book=attachments)
        os.replace(attachments, target)  # the chat book auto-migrates
        assert await svc.relocate(attachments, str(target)) == 1
        moved = await svc.links_for(book=str(target))
        by_path = await svc.links_for(paths=[str(target / "Novel.epub")])
        old = await svc.links_for(book=attachments)
        pre = await svc.preflight("gofile", str(target / "Novel.epub"))
        failed = await svc.forget_book(str(target), delete_remote=True)
        return link, moved, by_path, old, pre, failed, await svc.links_for()

    link, moved, by_path, old, pre, failed, left = asyncio.run(run())
    assert [x.id for x in moved] == [link.id] == [x.id for x in by_path] and old == []
    assert moved[0].source == str(target / "Novel.epub")
    assert pre.ok and pre.existing is not None and pre.existing.id == link.id
    assert failed == [] and left == [] and gofile.folders == {}


def test_deleting_a_book_drops_its_records_even_unreadable_ones(env):
    svc = _service(env)
    workspace, path = _book(env)
    other, other_path = str(env["Library"] / "Other"), str(env["Library"] / "Other" / "Other.epub")

    async def run():
        await _enable(svc, "transferit")
        kept = await svc.add_pasted_link("https://transfer.it/t/abcdef123", path=other_path, book=other)
        await svc.add_pasted_link("https://transfer.it/t/xyz987654", path=path, book=workspace)
        # a record whose secret no longer opens (the app's keys changed) still belongs to the book
        await svc._io(svc.store.mutate, lambda d: d["links"].append(
            {"id": "deadbeef0001", "provider": "gofile", "book": workspace, "source": path, "name": "Novel.epub",
             "size": 1, "created": 1.0, "url": "ENC:not-a-token", "delete": None}))
        assert len(svc.store.read()["links"]) == 3 and len(await svc.links_for(book=workspace)) == 1
        failed = await svc._io(svc.forget_books_blocking, [workspace])
        return kept, failed, svc.store.read()["links"]

    kept, failed, left = asyncio.run(run())
    assert failed == [] and [r["id"] for r in left] == [kept.id]


def test_secrets_are_refused_without_real_encryption(env):
    import api_key_encryption

    handler = api_key_encryption.APIKeyEncryption.__new__(api_key_encryption.APIKeyEncryption)
    handler.cipher = api_key_encryption._NullCipher()
    box = sl.SecretBox(handler=handler)
    assert not box.available()
    with pytest.raises(sp.ShareError) as info:
        box.seal("secret")
    assert info.value.code == "secrets_unavailable"
    with pytest.raises(sp.ShareError):
        sl.SecretBox().open("plain-text-not-encrypted")
    svc = _service(env, secrets=box)
    _, path = _book(env)

    async def run():
        await _enable(svc, "gofile")
        state = svc.provider_state("gofile")
        assert state.reason == "Secure storage is not available on this phone"
        with pytest.raises(sp.ShareError) as info:
            await svc.set_pixeldrain_key("pd-key-0123456789")
        return info.value

    assert asyncio.run(run()).code == "secrets_unavailable"
    assert "pd-key" not in (env["data"] / sl.STORE_FILE).read_text("utf-8")
    not_ready = sl.SecretBox(ready=lambda: False)
    assert not not_ready.available()


def test_wipe_forgets_links_keys_and_snapshots(env, gofile):
    svc = _service(env, providers={"gofile": gofile.provider()})
    workspace, path = _book(env, data=os.urandom(1000))
    stale = env["cache"] / sl.SNAPSHOT_DIR / "deadbeef"
    stale.mkdir(parents=True)
    (stale / "old.epub").write_bytes(b"x")

    async def run():
        await svc.load()
        assert not stale.exists()  # a killed process's snapshot is swept at start
        await _enable(svc, "gofile")
        await svc.upload("gofile", path, book=workspace)
        await svc.wipe()
        return svc.provider_states(), await svc.links_for()

    states, links = asyncio.run(run())
    assert links == [] and not (env["data"] / sl.STORE_FILE).exists()
    assert all(not s.enabled for s in states)


def test_corrupt_sidecar_is_moved_aside(env):
    path = env["data"] / sl.STORE_FILE
    path.write_text("{not json", "utf-8")
    svc = _service(env)
    asyncio.run(svc.load())
    assert svc.store.load_error and list(env["data"].glob("mobile_share_links.corrupt-*.json"))


# ---------------------------------------------------------------------------
# platform holds and the notification's Stop
# ---------------------------------------------------------------------------


class _FakeNative:
    def __init__(self) -> None:
        from glossarion_mobile.services.native import ServiceHolds

        self.service_holds = ServiceHolds()
        self.calls: list = []
        self.listeners: dict = {}

    async def call(self, method, *args, default=None, **kwargs):
        self.calls.append((method, args, kwargs))
        return 7 if method == "begin_background_task" else default

    def add_listener(self, event, callback):
        self.listeners.setdefault(event, []).append(callback)


class _FakeLease:
    instances: list = []

    def __init__(self, native, name) -> None:
        self.native, self.name, self.events = native, name, []
        _FakeLease.instances.append(self)

    async def acquire(self, title, text):
        self.events.append(("acquire", title, text))
        self.native.service_holds.hold(self.name, title, text)
        return True

    async def update(self, text, title=None):
        self.events.append(("update", text))

    async def release(self):
        self.events.append(("release",))
        self.native.service_holds.release(self.name)

    def drop(self):
        self.events.append(("drop",))


def test_android_holds_the_foreground_service_and_stop_cancels(env, pixeldrain, monkeypatch):
    from glossarion_mobile.services import native as native_mod

    monkeypatch.setattr(native_mod, "ServiceLease", _FakeLease, raising=False)
    _FakeLease.instances.clear()
    native = _FakeNative()
    svc = _service(env, platform="android", native=native, providers={"pixeldrain": pixeldrain.provider()})
    workspace, path = _book(env, data=os.urandom(4 * 1024 * 1024))
    pixeldrain.gate = threading.Event()

    async def run():
        await _enable(svc, "pixeldrain")
        await svc.set_pixeldrain_key("pd-key-0123456789")
        task = asyncio.ensure_future(svc.upload("pixeldrain", path, book=workspace))
        deadline = time.monotonic() + 30
        while not pixeldrain.started.is_set():
            assert time.monotonic() < deadline and not task.done()
            await asyncio.sleep(0.01)
        lease = _FakeLease.instances[-1]
        assert lease.name == sl.SHARE_HOLD and lease.events[0] == ("acquire", "Share file via link",
                                                                    "pixeldrain · Novel.epub")
        # a job also holds the service: its Stop belongs to the job
        native.service_holds.hold("jobs", "Glossarion", "Chapter 3/40")
        await svc.on_foreground_event({"type": "button", "button_id": "stop"})
        assert svc.uploading
        native.service_holds.release("jobs")
        await svc.on_foreground_event({"type": "destroyed"})
        assert lease.events[-1] == ("drop",)
        await svc.on_foreground_event({"type": "button", "button_id": "stop"})
        with pytest.raises(sp.ShareError) as info:
            await asyncio.wait_for(task, 30)
        return info.value, lease

    error, lease = asyncio.run(run())
    pixeldrain.gate.set()
    assert error.code == "cancelled" and lease.events[-1] == ("release",)
    assert sl.SHARE_HOLD not in native.service_holds


def test_ios_holds_a_background_task_during_the_upload(env, gofile):
    native = _FakeNative()
    svc = _service(env, platform="ios", native=native, providers={"gofile": gofile.provider()})
    workspace, path = _book(env, data=os.urandom(1000))

    async def run():
        await _enable(svc, "gofile")
        return await svc.upload("gofile", path, book=workspace)

    asyncio.run(run())
    methods = [c[0] for c in native.calls]
    assert methods == ["begin_background_task", "end_background_task"]
    assert native.calls[0][1] == (sl.SHARE_HOLD,) and native.calls[1][1] == (7,)


def test_service_lease_when_native_provides_it():
    """``native.ServiceLease`` (extracted from OAuthBridge's sign-in service, U10 Integrate)."""
    from glossarion_mobile.services import native as native_mod

    lease_cls = getattr(native_mod, "ServiceLease", None)
    if lease_cls is None:
        pytest.skip("native.ServiceLease not installed yet (patch request for Integrate)")

    class Native:
        def __init__(self, running=False):
            self.service_holds = native_mod.ServiceHolds()
            self.running, self.calls = running, []

        async def is_job_service_running(self):
            return self.running

        async def start_job_service(self, title, text):
            self.calls.append(("start", title, text))
            self.running = True
            return True

        async def update_job_service(self, text=None, title=None):
            self.calls.append(("update", title, text))

        async def stop_job_service(self):
            self.calls.append(("stop",))
            self.running = False

    async def run():
        alone = Native()
        lease = lease_cls(alone, "share-link")
        assert await lease.acquire("Share", "Gofile") and lease.started and lease.held
        await lease.update("50%")
        await lease.release()
        assert alone.calls == [("start", "Share", "Gofile"), ("update", "Share", "50%"), ("stop",)]
        shared = Native(running=True)
        shared.service_holds.hold("jobs", "Glossarion", "Chapter 1")
        lease = lease_cls(shared, "share-link")
        assert await lease.acquire("Share", "Gofile") and lease.held and not lease.started
        await lease.update("50%")  # the job's progress keeps the notification
        await lease.release()
        assert shared.calls == [("update", "Glossarion", "Chapter 1")] and "share-link" not in shared.service_holds

    asyncio.run(run())


# ---------------------------------------------------------------------------
# static guarantees
# ---------------------------------------------------------------------------


def _imports(path: Path) -> set:
    tree = ast.parse(path.read_text("utf-8"), feature_version=(3, 10))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            names.add(node.module.split(".")[0])
    return names


def test_modules_parse_for_3_10_and_import_no_flet_requests_or_httpx():
    files = [SERVICES_DIR / "share_links.py"] + PROVIDER_FILES
    assert {p.name for p in PROVIDER_FILES} >= {"__init__.py", "gofile.py", "pixeldrain.py", "send_e2ee.py",
                                                "transferit_handoff.py"}
    for path in files:
        imports = _imports(path)
        assert not imports & {"flet", "requests", "httpx"}, path.name  # http_logger patches requests/httpx
        data = path.read_bytes()
        assert data.count(b"\r\n") in (0, data.count(b"\n")), f"mixed line endings in {path.name}"


def test_the_transferit_module_cannot_make_requests():
    imports = _imports(SERVICES_DIR / "share_providers" / "transferit_handoff.py")
    assert not imports & {"http", "socket", "urllib3", "websockets", "ssl", "requests", "httpx"}
    source = (SERVICES_DIR / "share_providers" / "transferit_handoff.py").read_text("utf-8")
    assert "HttpClient" not in source and "api.mega" not in source and "bt7" not in source


def test_nothing_is_written_to_config_or_prefs(env, gofile):
    svc = _service(env, providers={"gofile": gofile.provider()})
    workspace, path = _book(env, data=os.urandom(1000))

    async def run():
        await _enable(svc, "gofile", "send", "transferit")
        await svc.upload("gofile", path, book=workspace)

    asyncio.run(run())
    files = sorted(p.name for p in env["data"].iterdir())
    assert files == [sl.STORE_FILE]


@pytest.mark.skipif(importlib.util.find_spec("flet") is None, reason="flet not installed")
def test_from_app_wires_the_app_and_refuses_the_data_dir(env):
    """``ShareLinkService.from_app``: Library + Output roots only (Android's data dir holds the config and
    Inbox), the foreground listener, the app's ``post`` / ``run_in_thread``, and the busy check of the
    Attachments guard (a queued job on the book's folder)."""
    from types import SimpleNamespace

    posted, threads, opened = [], [], []

    class Dispatcher:
        def post(self, fn, *args):
            posted.append(fn)
            fn(*args)
            return True

        async def run_in_thread(self, fn, *args, name="gl-worker"):
            threads.append(name)
            return await asyncio.to_thread(fn, *args)

    native = _FakeNative()
    workspace, path = _book(env, where="Output")
    job = SimpleNamespace(spec=SimpleNamespace(inputs=[], params={"folder": workspace}), output_dir=None,
                          output_dirs={})
    jobs = SimpleNamespace(view=lambda: SimpleNamespace(active=None, queue=(job,)))
    paths = SimpleNamespace(data=env["data"], library=env["Library"], output=env["Output"], docs=env["data"],
                            cache=env["cache"])
    app = SimpleNamespace(paths=paths, files=SimpleNamespace(platform="android", save_to_downloads=None),
                          dispatcher=Dispatcher(), native=native, opener=SimpleNamespace(launch=opened.append),
                          key_status=SimpleNamespace(installed=True), job_service=jobs)
    svc = sl.ShareLinkService.from_app(app)
    assert native.listeners["foreground"] == [svc.on_foreground_event]
    assert svc.eligible(path) and not svc.eligible(str(env["data"] / "Inbox" / "Book.epub"))
    assert svc.store.path == env["data"] / sl.STORE_FILE and svc.platform == "android"

    async def run():
        await _enable(svc, "gofile")
        with pytest.raises(sp.ShareError) as info:
            await svc.upload("gofile", path, book=workspace)
        return info.value

    assert asyncio.run(run()).code == "busy"
    assert threads and set(threads) == {"gl-share"} and posted
    jobs.view = lambda: SimpleNamespace(active=None, queue=())
    assert svc.is_busy(path) is False
