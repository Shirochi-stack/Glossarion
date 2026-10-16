"""Shared fakes for the on-device engine tests (``services.local_llm``): Prefs, a local fake Hugging Face
(``/api/models/<repo>/revision|tree/<rev>``, ``/<repo>/resolve/<rev>/<file>`` honouring ``Range``) and the
real ``unsloth/Qwen3.5-0.8B-GGUF`` tree listing (2026-10-16) as a fixture. No network."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

MOBILE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = MOBILE_DIR / "app"
SRC_DIR = MOBILE_DIR.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))
os.environ.setdefault("GLOSSARION_HTTP_LOG", "0")

#: (path, size, sha256) of the real unsloth/Qwen3.5-0.8B-GGUF main tree (plus .gitattributes / README).
QWEN_08B_TREE = [
    (".gitattributes", 3104, None),
    ("Qwen3.5-0.8B-BF16.gguf", 1516744736, "cedf89af31c9041b601fa58303285bc46d99c51baee1b13f5e919626ca526ee5"),
    ("Qwen3.5-0.8B-IQ4_NL.gguf", 506859776, "de9f7f28a4155c1678cbdae54c61ae63aa6326f7fffd2cdacfb765cfeab21670"),
    ("Qwen3.5-0.8B-IQ4_XS.gguf", 492605696, "619917ae92f61eb6515a7070e944a0a7c2b198a2cf6536386f475485188a36ff"),
    ("Qwen3.5-0.8B-Q3_K_M.gguf", 470167808, "c49ad509cd0a3f8584c4eff50d84202336bf9475b8a724c89f4061687c1fb510"),
    ("Qwen3.5-0.8B-Q4_0.gguf", 507154688, "444406ddd926550c724ec18d5120a9d40ded44908a063b0e66e9a7e5464c652c"),
    ("Qwen3.5-0.8B-Q4_K_M.gguf", 532517120, "bd258782e35f7f458f8aced1adc053e6e92e89bc735ba3be89d38a06121dc517"),
    ("Qwen3.5-0.8B-Q4_K_S.gguf", 508104960, "5f7ccfa6e9df0d9ebbaff9ee095b18202bec1e0ac313ca688d2c57c9c80a6bc9"),
    ("Qwen3.5-0.8B-Q5_K_M.gguf", 590057728, "c3ef5827b322c4be08a3a26ce424460d8b37daf593356c2cc1f4e40a7ab0581b"),
    ("Qwen3.5-0.8B-Q8_0.gguf", 811843840, "0ad885ffd4bb022fc4f0d33a3308fa108ef8613159d3b3a67e23abca056b7a6c"),
    ("Qwen3.5-0.8B-UD-Q4_K_XL.gguf", 558772480, "3177ebd67afe4438374da19e690bc1b98756f7e0fea9240e1be404336156a7b5"),
    ("README.md", 62950, None),
    ("imatrix_unsloth.gguf_file", 1134720, "9922aa6e3425aaa2aa2f0c9fe62ca0f1289318df7da3a4981132f1890391dd28"),
    ("mmproj-BF16.gguf", 207346528, "d312c4d02fd46eea7a16e4f3bbb58840e6222209322ca1e33ca03247ad8935d6"),
    ("mmproj-F16.gguf", 204987232, "56e4c6cfe73b0c82e3e82bc518d7591997e61d81f723fc41a586f4fa69ea2453"),
]


def tree_json(rows) -> list:
    out = []
    for path, size, sha in rows:
        entry = {"type": "file", "oid": hashlib.sha1(path.encode()).hexdigest(), "size": size, "path": path}
        if sha:
            entry["lfs"] = {"oid": sha, "size": size, "pointerSize": 135}
        out.append(entry)
    return out


class FakePrefs:
    def __init__(self, data=None):
        self.data = dict(data or {})

    def get(self, key, default=None):
        return self.data.get(key, default)

    def set(self, key, value):
        self.data[key] = value


class FakeHF:
    """A tiny Hugging Face: ``repos[repo] = {file: bytes}`` (plus ``trees[repo]`` listing overrides).

    Knobs: ``ignore_range`` (answer 200 to a Range request), ``cut_after`` (close the connection after
    that many body bytes, once), ``status`` (force an HTTP status for /api), ``lie_sha`` (list a wrong
    sha256), ``chunk_delay`` (seconds between 64 KiB writes)."""

    COMMIT = "c0ffee" * 6 + "abcd"

    def __init__(self):
        self.repos: dict = {}
        self.trees: dict = {}
        self.ignore_range = False
        self.cut_after = None
        self.status = None
        self.lie_sha = False
        self.chunk_delay = 0.0
        self.requests: list = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def do_GET(self):
                owner.requests.append((self.path, self.headers.get("Range")))
                path = unquote(urlsplit(self.path).path)
                parts = [p for p in path.split("/") if p]
                if parts[:2] == ["api", "models"]:
                    return owner._api(self, parts[2:])
                if len(parts) >= 5 and parts[2] == "resolve":
                    return owner._resolve(self, f"{parts[0]}/{parts[1]}", "/".join(parts[4:]))
                self._reply(404, b"{}")

            def _reply(self, status, body, ctype="application/json", headers=None):
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def close(self):
        self.server.shutdown()
        self.server.server_close()

    def add(self, repo: str, files: dict):
        self.repos[repo] = dict(files)

    def _api(self, handler, rest):
        if self.status:
            return handler._reply(self.status, b'{"error":"nope"}')
        if len(rest) < 4:
            return handler._reply(404, b"{}")
        repo = f"{rest[0]}/{rest[1]}"
        if repo not in self.repos and repo not in self.trees:
            return handler._reply(401, b'{"error":"Repository not found"}')
        if rest[2] == "revision":
            return handler._reply(200, json.dumps({"id": repo, "sha": self.COMMIT}).encode())
        if rest[2] == "tree":
            if repo in self.trees:
                rows = self.trees[repo]
            else:
                rows = [(name, len(data), ("0" * 64) if self.lie_sha else hashlib.sha256(data).hexdigest())
                        for name, data in self.repos[repo].items()]
            return handler._reply(200, json.dumps(tree_json(rows)).encode())
        return handler._reply(404, b"{}")

    def _resolve(self, handler, repo, file):
        data = self.repos.get(repo, {}).get(file)
        if data is None:
            return handler._reply(404, b"not found", "text/plain")
        start = 0
        rng = handler.headers.get("Range")
        status = 200
        if rng and not self.ignore_range:
            start = int(rng.split("=", 1)[1].split("-", 1)[0])
            if start >= len(data):
                return handler._reply(416, b"", "text/plain", {"Content-Range": f"bytes */{len(data)}"})
            status = 206
        body = data[start:]
        handler.send_response(status)
        handler.send_header("Content-Type", "application/octet-stream")
        handler.send_header("Content-Length", str(len(body)))
        handler.send_header("Accept-Ranges", "bytes")
        handler.send_header("X-Linked-Size", str(len(data)))
        if status == 206:
            handler.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
        handler.end_headers()
        cut = self.cut_after
        sent = 0
        import time

        for offset in range(0, len(body), 65536):
            block = body[offset:offset + 65536]
            if cut is not None and sent + len(block) > cut:
                block = block[: max(0, cut - sent)]
                try:
                    handler.wfile.write(block)
                    handler.wfile.flush()
                except OSError:
                    pass
                self.cut_after = None
                handler.close_connection = True
                try:
                    handler.connection.shutdown(2)
                except OSError:
                    pass
                return
            handler.wfile.write(block)
            sent += len(block)
            if self.chunk_delay:
                time.sleep(self.chunk_delay)
