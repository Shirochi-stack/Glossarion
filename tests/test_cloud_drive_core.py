"""Shared Google Drive sync / share-link core (google_drive_core, drive_sync_core, library_cloud,
share_hosting*) against local fake HTTP servers. Nothing here talks to Google or a hosting site."""

import ast
import hashlib
import json
import os
import re
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import drive_sync_core as dsc  # noqa: E402
import google_drive_core as gdc  # noqa: E402
import library_cloud as lc  # noqa: E402
import share_hosting as sh  # noqa: E402

SHARED = ["share_hosting.py", "share_hosting_gofile.py", "share_hosting_pixeldrain.py", "share_hosting_send.py",
          "share_hosting_transferit.py", "google_drive_core.py", "drive_sync_core.py", "library_cloud.py"]


# ---------------------------------------------------------------------------------------------------
# a fake Google (OAuth + Drive v3) on 127.0.0.1
# ---------------------------------------------------------------------------------------------------


class FakeGoogle:
    def __init__(self):
        self.files = {}  # id -> dict(name, mime, parents, content, trashed)
        self.sessions = {}
        self.refresh_tokens = {"refresh-1"}
        self.codes = {}  # code -> (challenge, redirect_uri)
        self.access_tokens = set()
        self.revoked = []
        self.requests = []  # (method, path, query)
        self.token_calls = []
        self.hold = None  # threading.Event: PUT waits for it
        self.held = threading.Event()
        self._n = 0
        self.lock = threading.Lock()
        handler = self._handler()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.server.daemon_threads = True
        self.server.handle_error = lambda *args: None  # a cancelled client drops its connection
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        if self.hold is not None:
            self.hold.set()
        self.server.shutdown()
        self.server.server_close()

    # ---- helpers ----------------------------------------------------------------------------------

    def new_id(self, prefix="f"):
        with self.lock:
            self._n += 1
            return f"{prefix}{self._n}"

    def meta(self, file_id):
        item = self.files[file_id]
        data = {"id": file_id, "name": item["name"], "mimeType": item["mime"], "trashed": item["trashed"],
                "parents": list(item["parents"])}
        if item["mime"] != gdc.FOLDER_MIME:
            data["md5Checksum"] = hashlib.md5(item["content"]).hexdigest()
            data["size"] = str(len(item["content"]))
        return data

    def children(self, parent, *, include_trashed=False):
        return [fid for fid, item in self.files.items()
                if parent in item["parents"] and (include_trashed or not item["trashed"])]

    def path_of(self, file_id):
        parts = []
        current = file_id
        while current != "root":
            item = self.files[current]
            parts.append(item["name"])
            current = item["parents"][0]
        return "/".join(reversed(parts))

    def tree(self):
        """``{path: content}`` of the non-trashed files (folders end with '/')."""
        out = {}
        for fid, item in self.files.items():
            if item["trashed"]:
                continue
            path = self.path_of(fid)
            out[path + "/" if item["mime"] == gdc.FOLDER_MIME else path] = item["content"]
        return out

    def uploads(self):
        return [r for r in self.requests if r[0] in ("POST", "PATCH") and r[1].startswith("/upload/")]

    def drive_requests(self):
        return [r for r in self.requests if r[1].startswith(("/drive/", "/upload/"))]

    def _handler(self):
        fake = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def _body(self):
                length = int(self.headers.get("Content-Length") or 0)
                return self.rfile.read(length) if length else b""

            def _send(self, status, payload=None, headers=None):
                data = b"" if payload is None else (payload if isinstance(payload, bytes)
                                                     else json.dumps(payload).encode("utf-8"))
                self.send_response(status)
                for name, value in (headers or {}).items():
                    self.send_header(name, value)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _authorized(self):
                auth = self.headers.get("Authorization", "")
                if auth.startswith("Bearer ") and auth[7:] in fake.access_tokens:
                    return True
                self._send(401, {"error": {"code": 401, "message": "Invalid Credentials"}})
                return False

            def do_POST(self):
                self._route("POST")

            def do_GET(self):
                self._route("GET")

            def do_PATCH(self):
                self._route("PATCH")

            def do_PUT(self):
                self._route("PUT")

            def _route(self, method):
                parts = urlsplit(self.path)
                query = {k: v[0] for k, v in parse_qs(parts.query).items()}
                body = self._body()
                fake.requests.append((method, parts.path, query))
                path = parts.path
                if path == "/token":
                    return self._token(parse_qs(body.decode("ascii")))
                if path == "/revoke":
                    token = parse_qs(body.decode("ascii")).get("token", [""])[0]
                    fake.revoked.append(token)
                    fake.refresh_tokens.discard(token)
                    return self._send(200, {})
                if not self._authorized():
                    return None
                if path == "/drive/v3/files" and method == "GET":
                    return self._list(query)
                if path == "/drive/v3/files" and method == "POST":
                    meta = json.loads(body)
                    fid = fake.new_id("d" if meta.get("mimeType") == gdc.FOLDER_MIME else "f")
                    fake.files[fid] = {"name": meta["name"], "mime": meta.get("mimeType", ""),
                                       "parents": meta.get("parents") or ["root"], "content": b"", "trashed": False}
                    return self._send(200, fake.meta(fid))
                match = re.fullmatch(r"/drive/v3/files/([\w-]+)(/permissions)?", path)
                if match:
                    fid = match.group(1)
                    if fid not in fake.files:
                        return self._send(404, {"error": {"code": 404}})
                    if match.group(2):
                        fake.files[fid]["shared"] = json.loads(body)
                        return self._send(200, {"id": "perm"})
                    if method == "PATCH":
                        fake.files[fid].update({k: v for k, v in json.loads(body).items() if k == "trashed"})
                    return self._send(200, fake.meta(fid))
                if path == "/upload/drive/v3/files" and method == "POST":
                    return self._start_session(None, json.loads(body))
                match = re.fullmatch(r"/upload/drive/v3/files/([\w-]+)", path)
                if match and method == "PATCH":
                    if match.group(1) not in fake.files:
                        return self._send(404, {"error": {"code": 404}})
                    return self._start_session(match.group(1), json.loads(body))
                if path.startswith("/upload/drive/v3/files") and method == "PUT":
                    return self._put(query.get("upload_id"), body)
                return self._send(404, {"error": {"code": 404}})

            def _token(self, form):
                fake.token_calls.append({k: v[0] for k, v in form.items()})
                grant = form.get("grant_type", [""])[0]
                if grant == "refresh_token":
                    if form.get("refresh_token", [""])[0] not in fake.refresh_tokens:
                        return self._send(400, {"error": "invalid_grant"})
                elif grant == "authorization_code":
                    code = form.get("code", [""])[0]
                    if code not in fake.codes:
                        return self._send(400, {"error": "invalid_grant"})
                    challenge, redirect = fake.codes.pop(code)
                    verifier = form.get("code_verifier", [""])[0]
                    import base64
                    digest = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
                    if digest != challenge or form.get("redirect_uri", [""])[0] != redirect:
                        return self._send(400, {"error": "invalid_grant"})
                    fake.refresh_tokens.add("refresh-from-code")
                else:
                    return self._send(400, {"error": "unsupported_grant_type"})
                token = fake.new_id("access-")
                fake.access_tokens.add(token)
                payload = {"access_token": token, "expires_in": 3600}
                if grant == "authorization_code":
                    payload["refresh_token"] = "refresh-from-code"
                return self._send(200, payload)

            def _list(self, query):
                q = query.get("q", "")
                parent = re.search(r"'((?:[^'\\]|\\.)*)' in parents", q).group(1)
                name = re.search(r"name = '((?:[^'\\]|\\.)*)'", q)
                name = name.group(1).replace("\\'", "'").replace("\\\\", "\\") if name else None
                folders = f"mimeType = '{gdc.FOLDER_MIME}'" in q
                found = []
                for fid in fake.children(parent):
                    item = fake.files[fid]
                    if name is not None and item["name"] != name:
                        continue
                    if folders and item["mime"] != gdc.FOLDER_MIME:
                        continue
                    found.append(fake.meta(fid))
                page = int(query.get("pageToken") or 0)
                chunk, rest = found[page:page + 2], found[page + 2:]  # 2 per page: exercise paging
                payload = {"files": chunk}
                if rest:
                    payload["nextPageToken"] = str(page + 2)
                return self._send(200, payload)

            def _start_session(self, file_id, meta):
                upload_id = fake.new_id("u")
                fake.sessions[upload_id] = {"file_id": file_id, "meta": meta, "data": b"",
                                            "size": int(self.headers.get("X-Upload-Content-Length") or 0),
                                            "mime": self.headers.get("X-Upload-Content-Type") or ""}
                return self._send(200, b"", {"Location": f"{fake.base}/upload/drive/v3/files?upload_id={upload_id}"})

            def _put(self, upload_id, body):
                if fake.hold is not None:
                    fake.held.set()
                    fake.hold.wait(10)
                session = fake.sessions.get(upload_id)
                if session is None:
                    return self._send(404, {"error": {"code": 404}})
                session["data"] += body
                if len(session["data"]) < session["size"]:
                    return self._send(308, b"", {"Range": f"bytes=0-{len(session['data']) - 1}"})
                if session["file_id"]:
                    fid = session["file_id"]
                    fake.files[fid]["content"] = session["data"]
                    fake.files[fid]["name"] = session["meta"].get("name", fake.files[fid]["name"])
                else:
                    fid = fake.new_id("f")
                    fake.files[fid] = {"name": session["meta"]["name"], "mime": session["mime"],
                                       "parents": session["meta"].get("parents") or ["root"],
                                       "content": session["data"], "trashed": False}
                del fake.sessions[upload_id]
                return self._send(200, fake.meta(fid))

        return Handler


@pytest.fixture
def google():
    fake = FakeGoogle()
    yield fake
    fake.close()


def _oauth(google, **kw):
    return gdc.OAuthClient("client-id", "client-secret", auth_url=google.base + "/auth",
                           token_url=google.base + "/token", revoke_url=google.base + "/revoke", **kw)


def _client(google, refresh="refresh-1", chunk_size=None):
    oauth = gdc.GoogleOAuth(None, _oauth(google))
    return gdc.DriveClient(lambda rejected: oauth.token(refresh, rejected), api_base=google.base + "/drive/v3",
                           upload_base=google.base + "/upload/drive/v3", chunk_size=chunk_size), oauth


def _write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return str(path)


def _run(google, plan, state, *, book="book-1", folder="Glossarion", cancel=None, chunk_size=None):
    client, _ = _client(google, chunk_size=chunk_size)
    return dsc.DriveSyncRunner(client, state).run(plan, book_key=book, folder=folder, label="Book", cancel=cancel)


# ---------------------------------------------------------------------------------------------------
# compiled output mode
# ---------------------------------------------------------------------------------------------------


def test_compiled_mode_creates_the_folder_and_updates_the_same_file_on_recompile(google, tmp_path):
    epub = _write(tmp_path / "out" / "Book.epub", b"first compile")
    state = {}
    plan = dsc.plan_sync("compiled", output_folder=str(tmp_path / "out"), compiled_files=[epub])
    first = _run(google, plan, state, folder="Books/Glossarion")
    assert first.ok and (first.uploaded, first.updated, first.folders_created) == (1, 0, 2), first
    assert google.tree() == {"Books/": b"", "Books/Glossarion/": b"", "Books/Glossarion/Book.epub": b"first compile"}
    file_id = state["books"]["book-1"]["files"]["Book.epub"]

    _write(epub, b"second compile, longer")
    second = _run(google, plan, state, folder="Books/Glossarion")
    assert second.ok and (second.uploaded, second.updated, second.folders_created) == (0, 1, 0), second
    assert google.tree()["Books/Glossarion/Book.epub"] == b"second compile, longer"
    assert [f for f in google.files.values() if f["name"] == "Book.epub"].__len__() == 1
    assert state["books"]["book-1"]["files"]["Book.epub"] == file_id
    # files.update on the remembered id, never a new file
    assert [r[1] for r in google.uploads()][-1] == f"/upload/drive/v3/files/{file_id}"

    third = _run(google, plan, state, folder="Books/Glossarion")
    assert (third.uploaded, third.updated, third.unchanged) == (0, 0, 1)


def test_compiled_mode_reuses_the_same_named_app_file_when_the_id_is_gone(google, tmp_path):
    epub = _write(tmp_path / "Book.epub", b"v1")
    plan = dsc.plan_sync("compiled", compiled_files=[epub])
    assert _run(google, plan, {}).uploaded == 1  # remembered in a state that is then lost
    _write(epub, b"v2")
    result = _run(google, plan, {"books": {"book-1": {"files": {"Book.epub": "gone123"}}}})
    assert (result.uploaded, result.updated) == (0, 1)
    assert sorted(google.tree()) == ["Glossarion/", "Glossarion/Book.epub"]
    assert google.tree()["Glossarion/Book.epub"] == b"v2"


def test_compiled_mode_with_a_pdf_and_an_empty_plan(google, tmp_path):
    pdf = _write(tmp_path / "Book_translated.pdf", b"%PDF-1.7 fake")
    result = _run(google, dsc.plan_sync("compiled", compiled_files=[pdf]), {})
    assert result.ok and google.tree()["Glossarion/Book_translated.pdf"] == b"%PDF-1.7 fake"
    empty = _run(google, dsc.plan_sync("compiled", compiled_files=[]), {})
    assert empty.error_code == "empty" and not empty.ok


# ---------------------------------------------------------------------------------------------------
# full output folder modes
# ---------------------------------------------------------------------------------------------------


def _output_folder(tmp_path):
    root = tmp_path / "My Novel"
    _write(root / "Book.epub", b"compiled")
    _write(root / "response_001.html", b"<p>one</p>")
    _write(root / "images" / "cover.jpg", b"\xff\xd8jpeg")
    _write(root / "images" / "deep" / "x.png", b"png")
    _write(root / "translation_progress.json", b"{}")
    _write(root / "empty.txt", b"")
    (root / "empty_dir").mkdir()
    # never uploaded
    _write(root / "~$lock.docx", b"lock")
    _write(root / ".translation_progress.json.123.tmp", b"tmp")
    _write(root / "notes.part", b"partial")
    _write(root / "__pycache__" / "a.pyc", b"pyc")
    return root


def test_full_mode_mirrors_the_folder_tree_file_by_file_and_is_incremental(google, tmp_path):
    root = _output_folder(tmp_path)
    outside = tmp_path / "outside"
    _write(outside / "secret.txt", b"not part of the book")
    try:
        os.symlink(outside, root / "link_out", target_is_directory=True)
        os.symlink(outside / "secret.txt", root / "linked.txt")
    except (OSError, NotImplementedError):
        pass  # Windows without symlink rights: the rest still runs
    state = {}
    plan = dsc.plan_sync("full", output_folder=str(root), compiled_files=[str(root / "Book.epub")])
    first = _run(google, plan, state)
    assert first.ok, first
    tree = google.tree()
    expected_files = {"Glossarion/My Novel/Book.epub": b"compiled",
                      "Glossarion/My Novel/response_001.html": b"<p>one</p>",
                      "Glossarion/My Novel/images/cover.jpg": b"\xff\xd8jpeg",
                      "Glossarion/My Novel/images/deep/x.png": b"png",
                      "Glossarion/My Novel/translation_progress.json": b"{}",
                      "Glossarion/My Novel/empty.txt": b""}
    assert {k: v for k, v in tree.items() if not k.endswith("/")} == expected_files
    assert {k for k in tree if k.endswith("/")} == {"Glossarion/", "Glossarion/My Novel/", "Glossarion/My Novel/images/",
                                                   "Glossarion/My Novel/images/deep/", "Glossarion/My Novel/empty_dir/"}
    assert first.uploaded == 6 and not any(name.endswith(".zip") for name in tree)

    _write(root / "response_001.html", b"<p>one, fixed</p>")
    _write(root / "response_002.html", b"<p>two</p>")
    before = len(google.uploads())
    second = _run(google, dsc.plan_sync("full", output_folder=str(root)), state)
    assert (second.uploaded, second.updated, second.unchanged, second.folders_created) == (1, 1, 5, 0), second
    assert len(google.uploads()) - before == 2
    assert google.tree()["Glossarion/My Novel/response_001.html"] == b"<p>one, fixed</p>"
    assert len([f for f in google.files.values() if f["name"] == "response_001.html"]) == 1


def test_full_raw_mode_sends_the_raw_source_to_a_RAW_subfolder(google, tmp_path):
    root = _output_folder(tmp_path)
    source = _write(tmp_path / "Raw" / "My Novel.epub", b"raw source")
    result = _run(google, dsc.plan_sync("full_raw", output_folder=str(root), source_file=source), {})
    assert result.ok
    assert google.tree()["Glossarion/My Novel/RAW/My Novel.epub"] == b"raw source"
    full_only = dsc.plan_sync("full", output_folder=str(root), source_file=source)
    assert not [i for i in full_only.items if i.role == "raw"]


def test_folder_parts_and_skip_rules():
    assert dsc.folder_parts("") == ("Glossarion",)
    assert dsc.folder_parts(" Books\\ Glossarion /../.") == ("Books", "Glossarion")
    assert dsc.normalize_mode("nope") == "compiled" and dsc.MODES == ("compiled", "full", "full_raw")
    assert dsc.MODE_LABELS["full_raw"] == "Full output folder + raw source"
    for name in ("~$a.docx", ".~lock.a#", "a.tmp", "x.part", "Thumbs.db", ".a.json.1.tmp", "desktop.ini"):
        assert dsc.is_skipped_name(name), name
    for name in ("Book.epub", "response_1.html", "a.tmpl.txt"):
        assert not dsc.is_skipped_name(name), name


def test_resumable_upload_in_several_chunks_and_an_empty_file(google, tmp_path):
    data = os.urandom(1000)
    path = _write(tmp_path / "big.bin", data)
    client, _ = _client(google, chunk_size=300)
    progress = []
    uploaded = client.upload(path, name="big.bin", progress=lambda s, t: progress.append((s, t)))
    assert google.files[uploaded.id]["content"] == data and uploaded.md5 == hashlib.md5(data).hexdigest()
    assert progress == [(300, 1000), (600, 1000), (900, 1000), (1000, 1000)]
    puts = [r for r in google.requests if r[0] == "PUT"]
    assert len(puts) == 4
    empty = client.upload(_write(tmp_path / "e.txt", b""), name="e.txt")
    assert google.files[empty.id]["content"] == b""


# ---------------------------------------------------------------------------------------------------
# tokens, failures, cancel
# ---------------------------------------------------------------------------------------------------


def test_an_expired_access_token_is_refreshed_once(google, tmp_path):
    client, oauth = _client(google)
    client.ensure_folder("A")
    assert len(google.token_calls) == 1
    google.access_tokens.clear()  # Google expired it early: 401, refresh, retry
    client.ensure_folder("B")
    assert len(google.token_calls) == 2
    assert google.token_calls[-1]["client_secret"] == "client-secret"
    assert google.token_calls[-1]["grant_type"] == "refresh_token"


def test_a_revoked_grant_stops_the_sync_and_asks_to_sign_in_again(google, tmp_path):
    epub = _write(tmp_path / "Book.epub", b"x")
    google.refresh_tokens.clear()
    result = _run(google, dsc.plan_sync("compiled", compiled_files=[epub]), {})
    assert result.error_code == "sign_in" and not result.ok
    assert not google.drive_requests()


def test_a_network_failure_ends_in_the_result_not_an_exception(tmp_path):
    epub = _write(tmp_path / "Book.epub", b"x")
    import socket

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()  # nothing listens there now
    client = gdc.DriveClient(lambda rejected: "token", api_base=f"http://127.0.0.1:{port}/drive/v3",
                             upload_base=f"http://127.0.0.1:{port}/upload/drive/v3")
    result = dsc.DriveSyncRunner(client, {}).run(dsc.plan_sync("compiled", compiled_files=[epub]), book_key="b")
    assert result.error_code == "network" and not result.ok, result


def test_cancel_stops_a_running_upload(google, tmp_path):
    epub = _write(tmp_path / "Book.epub", os.urandom(4000))
    google.hold = threading.Event()
    cancel = sh.CancelToken()
    out = {}

    def worker():
        out["result"] = _run(google, dsc.plan_sync("compiled", compiled_files=[epub]), {}, cancel=cancel)

    thread = threading.Thread(target=worker)
    thread.start()
    assert google.held.wait(10)
    cancel.cancel()
    google.hold.set()
    thread.join(10)
    assert out["result"].cancelled and not out["result"].ok


def test_https_only_except_loopback():
    client = gdc.DriveClient(lambda rejected: "t", api_base="http://example.com/drive/v3")
    with pytest.raises(sh.ShareError) as info:
        client.get("abc")
    assert info.value.code == "protocol"


# ---------------------------------------------------------------------------------------------------
# desktop sign-in (loopback + PKCE)
# ---------------------------------------------------------------------------------------------------


def _browser(google, auth_url, *, tamper_state=False, error=None):
    """What the browser does: Google approves and redirects to the loopback listener."""
    params = {k: v[0] for k, v in parse_qs(urlsplit(auth_url).query).items()}
    assert params["code_challenge_method"] == "S256" and params["access_type"] == "offline"
    assert params["scope"] == gdc.SCOPE and params["redirect_uri"].startswith("http://127.0.0.1:")
    google.codes["code-1"] = (params["code_challenge"], params["redirect_uri"])
    state = "wrong" if tamper_state else params["state"]
    query = f"error={error}&state={state}" if error else f"code=code-1&state={state}"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # never via a system proxy
    try:
        opener.open(f"{params['redirect_uri']}/?{query}", timeout=10).read()
    except Exception:
        pass  # a 400 page for the failure cases


def test_desktop_sign_in_with_pkce_over_loopback(google):
    sign_in = gdc.DesktopSignIn(_oauth(google), timeout=20)
    assert "code_verifier" not in sign_in.auth_url
    threading.Thread(target=_browser, args=(google, sign_in.auth_url), daemon=True).start()
    result = sign_in.wait(15)
    assert result.refresh_token == "refresh-from-code" and "refresh" not in repr(result)
    exchange = [c for c in google.token_calls if c["grant_type"] == "authorization_code"][0]
    assert exchange["client_id"] == "client-id" and exchange["client_secret"] == "client-secret"
    assert sign_in.session.closed


def test_desktop_sign_in_refuses_a_wrong_state_and_a_denial(google):
    sign_in = gdc.DesktopSignIn(_oauth(google), timeout=20)
    threading.Thread(target=_browser, args=(google, sign_in.auth_url), kwargs={"tamper_state": True},
                     daemon=True).start()
    with pytest.raises(sh.ShareError) as info:
        sign_in.wait(15)
    assert info.value.code == "auth"
    denied = gdc.DesktopSignIn(_oauth(google), timeout=20)
    threading.Thread(target=_browser, args=(google, denied.auth_url), kwargs={"error": "access_denied"},
                     daemon=True).start()
    with pytest.raises(sh.ShareError) as info:
        denied.wait(15)
    assert info.value.code == "cancelled"
    assert not [c for c in google.token_calls if c["grant_type"] == "authorization_code"]


def test_desktop_sign_in_needs_a_client():
    assert gdc.BUILTIN_DESKTOP_CLIENT_ID == "" and gdc.BUILTIN_DESKTOP_CLIENT_SECRET == ""
    assert gdc.desktop_client() is None
    assert gdc.desktop_client("", "s", use_own=True) is None
    own = gdc.desktop_client("my-id.apps.googleusercontent.com", " Qx7value ")
    assert own.client_id.startswith("my-id") and own.client_secret == "Qx7value" and "Qx7value" not in repr(own)
    with pytest.raises(sh.ShareError) as info:
        gdc.DesktopSignIn(None)
    assert info.value.code == "unsupported" and "own Google OAuth client" in info.value.message


def test_cancelling_a_desktop_sign_in(google):
    sign_in = gdc.DesktopSignIn(_oauth(google), timeout=20)
    cancel = sh.CancelToken()
    threading.Timer(0.3, cancel.cancel).start()
    with pytest.raises(sh.ShareError) as info:
        sign_in.wait(10, cancel)
    assert info.value.code == "cancelled" and sign_in.session.closed


# ---------------------------------------------------------------------------------------------------
# the desktop store and controller
# ---------------------------------------------------------------------------------------------------


@pytest.fixture
def box(monkeypatch):
    from cryptography.fernet import Fernet

    import api_key_encryption

    monkeypatch.setenv("GLOSSARION_API_KEY_FERNET", Fernet.generate_key().decode("ascii"))
    handler = api_key_encryption.APIKeyEncryption()  # process-local key: no key file is written
    assert handler.key_file is None
    return sh.SecretBox(handler=handler, device="computer")


def _cloud(google, tmp_path, box, own_client=True, **kw):
    store = lc.LibraryCloudStore(str(tmp_path / "cfg" / "library_cloud.json"), secrets=box)
    endpoints = {"auth_url": google.base + "/auth", "token_url": google.base + "/token",
                 "revoke_url": google.base + "/revoke", "api_base": google.base + "/drive/v3",
                 "upload_base": google.base + "/upload/drive/v3"}
    lines = []
    if own_client:
        store.set_own_client(use_own=True, client_id="client-id", client_secret="client-secret")
    cloud = lc.LibraryCloud(store, endpoints=endpoints, log_line=lines.append, **kw)
    return cloud, store, lines


def test_the_store_defaults_to_off_and_keeps_secrets_encrypted(tmp_path, box, monkeypatch):
    import app_paths

    monkeypatch.setattr(app_paths, "CONFIG_FILE", str(tmp_path / "config.json"))
    assert lc.default_store_path() == str(tmp_path / "library_cloud.json")
    store = lc.LibraryCloudStore(str(tmp_path / "library_cloud.json"), secrets=box)
    settings = store.drive_settings()
    assert (settings.enabled, settings.mode, settings.folder, settings.signed_in) == (False, "compiled",
                                                                                     "Glossarion", False)
    store.set_own_client(use_own=True, client_id="id-1", client_secret="very-secret-value")
    store.save_refresh_token("refresh-token-value")
    store.set_pixeldrain_key("pixeldrain-key-value")
    store.save_gofile_token("gofile-token-value")
    store.add_link("book", str(tmp_path / "Book.epub"), "gofile",
                   sh.UploadResult("https://gofile.io/d/abcdef", "x", {"folder": "f", "token": "t-value"}))
    raw = (tmp_path / "library_cloud.json").read_text(encoding="utf-8")
    for secret in ("very-secret-value", "refresh-token-value", "pixeldrain-key-value", "gofile-token-value",
                   "gofile.io/d/abcdef", "t-value"):
        assert secret not in raw, secret
    assert raw.count("ENC:") == 6
    again = lc.LibraryCloudStore(str(tmp_path / "library_cloud.json"), secrets=box)
    assert again.refresh_token() == "refresh-token-value"
    assert again.oauth_client().client_secret == "very-secret-value"
    assert again.links("book")[0]["url"] == "https://gofile.io/d/abcdef"
    # switching the OAuth client signs out (a refresh token belongs to its client)
    again.set_own_client(client_id="id-2")
    assert not again.drive_settings().signed_in
    assert not (tmp_path / "config.json").exists()


def test_sync_is_off_by_default_and_a_compile_sends_nothing(google, tmp_path, box):
    cloud, store, lines = _cloud(google, tmp_path, box)
    epub = _write(tmp_path / "out" / "Book.epub", b"x")
    assert cloud.on_compile_finished("epub", epub) is False
    store.save_refresh_token("refresh-1")  # signed in, but sync still off
    assert cloud.on_compile_finished("epub", epub) is False
    assert cloud.on_compile_finished("epub", None) is False
    assert cloud.on_compile_finished("epub", str(tmp_path / "missing.epub")) is False
    assert cloud.on_compile_finished("epub", epub, error="boom") is False
    cloud.sync.worker.wait_idle(5)
    assert google.requests == [] and lines == []
    cloud.shutdown()


def test_a_compile_uploads_in_the_background_when_sync_is_on(google, tmp_path, box):
    changes = []
    cloud, store, lines = _cloud(google, tmp_path, box, on_change=changes.append)
    store.save_refresh_token("refresh-1")
    store.set_drive_options(enabled=True, folder="Library Sync")
    epub = _write(tmp_path / "out" / "Book.epub", b"v1")
    assert cloud.on_compile_finished("epub", epub) is True
    assert cloud.sync.worker.wait_idle(15)
    _write(epub, b"v2")
    assert cloud.on_compile_finished("epub", epub)
    assert cloud.sync.worker.wait_idle(15)
    assert google.tree() == {"Library Sync/": b"", "Library Sync/Book.epub": b"v2"}
    assert store.last_sync()["ok"] and store.last_sync()["updated"] == 1
    assert len(lines) == 4 and lines[0].startswith("☁") and "Drive sync done" in lines[-1]
    assert "sync" in changes
    # full mode with the raw source, through the controller's manual "Sync to Drive now"
    store.set_drive_options(mode="full_raw")
    raw = _write(tmp_path / "Raw" / "out.txt", b"raw")
    assert cloud.sync_folder(str(tmp_path / "out"), source_file=raw)
    assert cloud.sync.worker.wait_idle(15)
    assert google.tree()["Library Sync/out/RAW/out.txt"] == b"raw"
    assert google.tree()["Library Sync/out/Book.epub"] == b"v2"
    cloud.shutdown()


def test_a_revoked_grant_signs_the_desktop_out(google, tmp_path, box):
    cloud, store, lines = _cloud(google, tmp_path, box)
    store.save_refresh_token("revoked-elsewhere")
    store.set_drive_options(enabled=True)
    assert cloud.on_compile_finished("epub", _write(tmp_path / "o" / "B.epub", b"x"))
    assert cloud.sync.worker.wait_idle(15)
    assert not store.drive_settings().signed_in
    assert store.last_sync()["error_code"] == "sign_in"
    assert any("sign in again" in line for line in lines)
    cloud.shutdown()


def test_desktop_sign_in_and_sign_out_through_the_controller(google, tmp_path, box):
    cloud, store, lines = _cloud(google, tmp_path, box, own_client=False)
    available, reason = cloud.sign_in_status()
    assert not available and reason == gdc.NO_DESKTOP_CLIENT
    with pytest.raises(sh.ShareError):
        cloud.begin_sign_in()
    # a token without a client (e.g. the own client was switched off) is kept, and nothing is sent
    store.save_refresh_token("refresh-1")
    store.set_drive_options(enabled=True)
    assert cloud.on_compile_finished("epub", _write(tmp_path / "o" / "B.epub", b"x"))
    assert cloud.sync.worker.wait_idle(15)
    assert store.refresh_token() == "refresh-1" and store.last_sync()["error_code"] == "unsupported"
    assert google.requests == []
    store.set_drive_options(enabled=False)
    store.clear_refresh_token()
    store.set_own_client(use_own=True, client_id="client-id", client_secret="client-secret")
    assert cloud.sign_in_status() == (True, "")
    sign_in = cloud.begin_sign_in(timeout=20)
    threading.Thread(target=_browser, args=(google, sign_in.auth_url), daemon=True).start()
    cloud.finish_sign_in(sign_in, timeout=15)
    assert store.refresh_token() == "refresh-from-code" and cloud.signed_in()
    assert cloud.sign_out() is True
    assert google.revoked == ["refresh-from-code"] and not cloud.signed_in()


def test_shutdown_cancels_a_running_sync(google, tmp_path, box):
    cloud, store, lines = _cloud(google, tmp_path, box)
    store.save_refresh_token("refresh-1")
    store.set_drive_options(enabled=True)
    google.hold = threading.Event()
    assert cloud.on_compile_finished("epub", _write(tmp_path / "o" / "B.epub", os.urandom(2000)))
    assert google.held.wait(10)
    started = time.monotonic()
    threading.Timer(0.2, google.hold.set).start()
    cloud.shutdown(timeout=10)
    assert time.monotonic() - started < 10
    assert not cloud.sync.worker.submit(dsc.SyncJob("x"))


def test_worker_coalesces_queued_jobs_per_book():
    ran = []
    gate = threading.Event()

    def run(job, cancel):
        if job.book_key == "first":
            gate.wait(5)
        ran.append((job.book_key, job.label))

    worker = dsc.DriveSyncWorker(run)
    worker.submit(dsc.SyncJob("first"))
    time.sleep(0.1)
    worker.submit(dsc.SyncJob("b", label="old"))
    worker.submit(dsc.SyncJob("c"))
    worker.submit(dsc.SyncJob("b", label="new"))
    gate.set()
    assert worker.wait_idle(5)
    assert ran == [("first", ""), ("c", ""), ("b", "new")]
    worker.shutdown()


def test_worker_merges_the_files_of_a_book_queued_twice():
    ran = []
    gate = threading.Event()

    def run(job, cancel):
        if job.book_key == "other":
            gate.wait(5)
        ran.append((job.book_key, job.compiled_files, job.reason))

    worker = dsc.DriveSyncWorker(run)
    worker.submit(dsc.SyncJob("other"))
    time.sleep(0.1)
    worker.submit(dsc.SyncJob("book", reason="compile", compiled_files=("a.epub",)))
    worker.submit(dsc.SyncJob("book", reason="compile", compiled_files=("a.pdf",)))
    worker.submit(dsc.SyncJob("manual", reason="manual", compiled_files=("m.epub", "m.pdf")))
    worker.submit(dsc.SyncJob("manual", reason="compile", compiled_files=("m.pdf",)))
    worker.submit(dsc.SyncJob("all", reason="compile", compiled_files=("x.epub",)))
    worker.submit(dsc.SyncJob("all", reason="manual"))  # no files named: every compiled file
    assert worker.queued == 3
    gate.set()
    assert worker.wait_idle(5)
    assert ran[1:] == [("book", ("a.epub", "a.pdf"), "compile"), ("manual", ("m.epub", "m.pdf"), "manual"),
                       ("all", (), "manual")]
    worker.shutdown()


def test_a_folder_change_puts_the_recompile_in_the_new_folder(google, tmp_path):
    epub = _write(tmp_path / "out" / "Book One.epub", b"v1")
    plan = dsc.plan_sync("compiled", compiled_files=[epub])
    state = {}
    assert _run(google, plan, state, folder="Glossarion").uploaded == 1
    _write(epub, b"v2")
    moved = _run(google, plan, state, folder="Books/Other")
    assert moved.ok and (moved.uploaded, moved.updated) == (1, 0), moved
    assert google.tree() == {"Books/": b"", "Books/Other/": b"", "Books/Other/Book One.epub": b"v2",
                             "Glossarion/": b"", "Glossarion/Book One.epub": b"v1"}
    _write(epub, b"v3")  # and the next compile updates the copy in the new folder
    again = _run(google, plan, state, folder="Books/Other")
    assert (again.uploaded, again.updated) == (0, 1)
    assert google.tree()["Books/Other/Book One.epub"] == b"v3"


def test_two_books_with_the_same_file_name_never_overwrite_each_other(google, tmp_path):
    a = _write(tmp_path / "SeriesA" / "Volume 1.epub", b"book A")
    b = _write(tmp_path / "SeriesB" / "Volume 1.epub", b"book B")
    state = {}
    assert _run(google, dsc.plan_sync("compiled", compiled_files=[a]), state, book="A").uploaded == 1
    second = _run(google, dsc.plan_sync("compiled", compiled_files=[b]), state, book="B")
    assert (second.uploaded, second.updated) == (1, 0), second
    assert google.tree() == {"Glossarion/": b"", "Glossarion/Volume 1.epub": b"book A",
                             "Glossarion/Volume 1 (2).epub": b"book B"}
    _write(a, b"book A v2")
    _write(b, b"book B v2")
    assert _run(google, dsc.plan_sync("compiled", compiled_files=[a]), state, book="A").updated == 1
    assert _run(google, dsc.plan_sync("compiled", compiled_files=[b]), state, book="B").updated == 1
    assert google.tree() == {"Glossarion/": b"", "Glossarion/Volume 1.epub": b"book A v2",
                             "Glossarion/Volume 1 (2).epub": b"book B v2"}


def test_sync_now_of_a_book_without_an_output_folder_sends_only_its_own_file(google, tmp_path, box, monkeypatch):
    import library_core

    library = tmp_path / "Library"
    shelf = library / "Translated"
    alpha = _write(shelf / "Alpha.epub", b"alpha")
    _write(shelf / "Beta.epub", b"beta")
    _write(shelf / "Gamma.epub", b"gamma")
    monkeypatch.setattr(library_core, "get_library_dir", lambda: str(library))
    monkeypatch.setattr(library_core, "_resolve_book_output_folder", lambda book: "")
    monkeypatch.setattr(library_core, "_resolve_book_translated_file", lambda book: book["path"])
    monkeypatch.setattr(library_core, "_resolve_book_source_file", lambda book: "")
    cloud, store, lines = _cloud(google, tmp_path, box)
    store.save_refresh_token("refresh-1")
    book = {"in_library": True, "path": alpha, "name": "Alpha"}
    assert cloud.sync_book(book)
    assert cloud.sync.worker.wait_idle(15)
    assert google.tree() == {"Glossarion/": b"", "Glossarion/Alpha.epub": b"alpha"}
    assert lc.shelf_book_key(book, "", alpha) == lc.book_key_for(alpha)
    # the full modes refuse (no output folder) instead of uploading the shelf
    store.set_drive_options(mode="full")
    assert cloud.sync_book(book)
    assert cloud.sync.worker.wait_idle(15)
    assert store.last_sync()["error_code"] == "no_output_folder"
    assert sorted(google.tree()) == ["Glossarion/", "Glossarion/Alpha.epub"]
    # a compile that lands on the shelf is keyed by its file, never the shelf folder
    store.set_drive_options(enabled=True)
    assert cloud.on_compile_finished("epub", alpha)
    assert cloud.sync.worker.wait_idle(15)
    assert store.last_sync()["error_code"] == "no_output_folder"
    assert lc.book_key_for(str(shelf)) not in store.sync_state().get("books", {})
    cloud.shutdown()


def test_changing_the_oauth_client_while_signed_in_revokes_the_old_grant(google, tmp_path, box):
    cloud, store, lines = _cloud(google, tmp_path, box)
    store.save_refresh_token("refresh-old")
    cloud.set_own_client(client_id="client-id")  # unchanged: stays signed in, nothing sent
    assert cloud.signed_in() and google.revoked == []
    cloud.set_own_client(client_id="client-id-2")
    assert not cloud.signed_in()
    cloud._revoke_thread.join(10)
    assert google.revoked == ["refresh-old"]
    assert any("OAuth client changed" in line for line in lines)
    cloud.shutdown()


def test_an_instance_process_reports_its_compiles_through_a_marker_line(tmp_path):
    import instance_worker

    printed = []
    listener = instance_worker._forward_compiles(printed.append)
    try:
        dsc.compile_finished("epub", str(tmp_path / "Book.epub"))
        dsc.compile_finished("pdf", str(tmp_path / "x.pdf"), error="boom")  # a failed compile is not reported
    finally:
        dsc.remove_compile_listener(listener)
    assert printed == [f"{instance_worker.COMPILED_MARKER}epub\t{tmp_path / 'Book.epub'}"]
    assert instance_worker.parse_compiled_marker(printed[0]) == ("epub", str(tmp_path / "Book.epub"))
    assert instance_worker.parse_compiled_marker("plain log line") is None
    pytest.importorskip("PySide6")
    import instance_runs

    seen = []

    def parent_listener(kind, path, error):
        seen.append((kind, path))

    dsc.add_compile_listener(parent_listener)
    try:
        instance_runs._report_instance_compile("epub", str(tmp_path / "Book.epub"))
    finally:
        dsc.remove_compile_listener(parent_listener)
    assert seen == [("epub", str(tmp_path / "Book.epub"))]


def test_the_service_asks_the_platform_for_compiled_files_a_job_does_not_name(google, tmp_path):
    """Mobile's ``resolve_compiled``: a Library-filed EPUB outside the workspace; a failing resolver falls back to
    the output folder's compiled files and never raises."""
    workspace = tmp_path / "Work"
    _write(workspace / "Work_translated.epub", b"workspace epub")
    filed = _write(tmp_path / "Library" / "Translated" / "Filed.epub", b"filed epub")
    client, _ = _client(google)
    asked = []

    def service(resolver):
        return dsc.DriveSyncService(
            settings=lambda: dsc.SyncSettings(enabled=True, mode="compiled", signed_in=True),
            access=lambda: client.access, load_state=lambda: {}, save_state=lambda state: None,
            api_base=google.base + "/drive/v3", upload_base=google.base + "/upload/drive/v3",
            resolve_compiled=resolver)

    def resolver(job):
        asked.append(job.book_key)
        return [filed]

    result = service(resolver).run_blocking(dsc.SyncJob("w", output_folder=str(workspace), reason="compile"))
    assert result.ok and asked == ["w"] and sorted(google.tree()) == ["Glossarion/", "Glossarion/Filed.epub"]

    def broken(job):
        raise RuntimeError("resolver bug")

    result = service(broken).run_blocking(dsc.SyncJob("w", output_folder=str(workspace), reason="compile"))
    assert result.ok and google.tree()["Glossarion/Work_translated.epub"] == b"workspace epub"
    # a job that names its files never asks
    asked.clear()
    named = _write(workspace / "Named.epub", b"named")
    assert service(resolver).run_blocking(dsc.SyncJob("w", output_folder=str(workspace), compiled_files=(named,),
                                                      reason="compile")).ok
    assert asked == [] and google.tree()["Glossarion/Named.epub"] == b"named"


def test_compile_listeners_get_each_reported_compile_and_never_raise_into_it(google, tmp_path, box):
    heard = []

    def broken(*args):
        raise RuntimeError("listener bug")

    dsc.add_compile_listener(broken)
    def listener(*args):
        heard.append(args)

    dsc.add_compile_listener(listener)
    try:
        dsc.add_compile_listener(listener)  # once only
        dsc.compile_finished("epub", tmp_path / "B.epub")
        dsc.compile_finished("pdf", None, RuntimeError("boom"))
        assert heard == [("epub", str(tmp_path / "B.epub"), None), ("pdf", None, "boom")]
    finally:
        dsc.remove_compile_listener(broken)
        dsc.remove_compile_listener(listener)
    dsc.compile_finished("epub", "x.epub")  # no listener: nothing happens
    assert len(heard) == 2
    # the desktop hook routes a reported compile to the shared controller
    cloud, store, lines = _cloud(google, tmp_path, box)
    store.save_refresh_token("refresh-1")
    store.set_drive_options(enabled=True)
    previous, lc._INSTANCE = lc._INSTANCE, cloud
    try:
        lc.install_compile_hook()
        dsc.compile_finished("epub", _write(tmp_path / "o" / "Hooked.epub", b"hooked"))
        assert cloud.sync.worker.wait_idle(15)
        assert google.tree()["Glossarion/Hooked.epub"] == b"hooked"
    finally:
        lc.uninstall_compile_hook()
        lc._INSTANCE = previous
        cloud.shutdown()


# ---------------------------------------------------------------------------------------------------
# share via link on the desktop
# ---------------------------------------------------------------------------------------------------


class FakeGofile:
    def __init__(self):
        self.bodies = []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                fake.bodies.append((self.path, dict(self.headers), body))
                data = json.dumps({"status": "ok", "data": {
                    "downloadPage": "https://gofile.io/d/AbC123", "parentFolder": "folder-1", "id": "file-1",
                    "guestToken": "guest-token-1"}}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def test_share_upload_to_a_hosting_site_records_the_link(google, tmp_path, box):
    import share_hosting_gofile as gf

    gofile = FakeGofile()
    try:
        provider = gf.GofileProvider(api_base=gofile.base, upload_url=gofile.base + "/uploadfile")
        cloud, store, lines = _cloud(google, tmp_path, box, providers={"gofile": provider})
        epub = _write(tmp_path / "out" / "Book.epub", b"epub bytes")
        with pytest.raises(sh.ShareError) as info:
            cloud.share_upload("gofile", epub)
        assert info.value.code == "disabled" and not gofile.bodies
        with pytest.raises(sh.ShareError) as info:
            store.set_provider_enabled("gofile", True)
        assert info.value.code == "consent"
        store.set_provider_enabled("gofile", True, consent=True)
        progress = []
        record = cloud.share_upload("gofile", epub, progress=lambda s, t: progress.append((s, t)))
        assert record["url"] == "https://gofile.io/d/AbC123" and record["provider"] == "gofile"
        assert b"epub bytes" in gofile.bodies[0][2] and progress[-1] == (10, 10)
        assert store.gofile_token() == "guest-token-1"
        assert cloud.links_for(lc.book_key_for(str(tmp_path / "out")))[0]["url"] == record["url"]
        assert "gofile.io/d/AbC123" not in (tmp_path / "cfg" / "library_cloud.json").read_text("utf-8")
        record2 = cloud.share_upload("gofile", epub)
        assert gofile.bodies[1][1].get("Authorization") == "Bearer guest-token-1"
        assert len(store.links()) == 2 and record2["id"] != record["id"]
        import tempfile

        assert not list(Path(tempfile.gettempdir()).glob("glossarion_share_*/Book.epub"))
    finally:
        gofile.close()


def test_share_by_drive_link_and_the_consent_wording(google, tmp_path, box):
    cloud, store, lines = _cloud(google, tmp_path, box)
    epub = _write(tmp_path / "out" / "Book.epub", b"shared via drive")
    store.set_provider_enabled("gdrive", True, consent=True)
    assert store.provider_state("gdrive").reason.startswith("Sign in to Google Drive")
    store.save_refresh_token("refresh-1")
    record = cloud.share_upload("gdrive", epub)
    file_id = record["url"].split("/d/")[1].split("/")[0]
    assert google.files[file_id]["shared"] == {"type": "anyone", "role": "reader", "allowFileDiscovery": False}
    assert cloud.delete_link(record["id"]) is True and google.files[file_id]["trashed"]
    text = " ".join(cloud.consent_text("gofile", epub).lines)
    assert "from this computer to Gofile" in text and "mobile data" not in text and "phone" not in text
    phone = " ".join(sh.consent_text("gofile").lines)
    assert "from this phone to Gofile" in phone and "mobile data" in phone
    assert [s.id for s in store.provider_states()] == list(lc.DESKTOP_PROVIDERS)


def test_transferit_is_a_browser_handoff_on_the_desktop(google, tmp_path, box):
    cloud, store, lines = _cloud(google, tmp_path, box)
    epub = _write(tmp_path / "out" / "Book.epub", b"x")
    store.set_provider_enabled("transferit", True, consent=True)
    plan = cloud.transferit_plan(epub)
    assert plan.url == "https://transfer.it/start" and plan.location == str(tmp_path / "out")
    record = cloud.add_pasted_link(epub, "here: https://transfer.it/t/AbCdEf12 thanks")
    assert record["url"] == "https://transfer.it/t/AbCdEf12"
    with pytest.raises(sh.ShareError):
        cloud.share_upload("transferit", epub)


# ---------------------------------------------------------------------------------------------------
# static guarantees
# ---------------------------------------------------------------------------------------------------


def _imports(path):
    tree = ast.parse(path.read_text("utf-8"), feature_version=(3, 10))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            names.add(node.module.split(".")[0])
    return names


def test_shared_modules_parse_for_3_10_and_import_no_gui_requests_or_httpx():
    for name in SHARED:
        path = SRC / name
        imports = _imports(path)
        assert not imports & {"flet", "requests", "httpx", "PySide6", "PyQt5", "PyQt6", "tkinter",
                              "glossarion_mobile"}, name
        data = path.read_bytes()
        assert data.count(b"\r\n") in (0, data.count(b"\n")), f"mixed line endings in {name}"
    transferit = _imports(SRC / "share_hosting_transferit.py")
    assert not transferit & {"http", "socket", "urllib3", "websockets", "ssl"}


def _spec_list(source, name):
    lines = source.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(f"{name} = ["))
    end = next(j for j in range(start + 1, len(lines)) if lines[j].rstrip() == "]")
    return ast.literal_eval("\n".join(lines[start:end + 1]).split("=", 1)[1].strip())


def test_new_modules_are_in_every_pyinstaller_spec():
    """GUI-free shared modules: both blocks of all 14 specs (the compile hook in the core desktop
    modules imports them, also in the Lite builds that leave epub_library out)."""
    specs = sorted(SRC.glob("translator*.spec"))
    assert len(specs) == 14
    for spec_path in specs:
        source = spec_path.read_text(encoding="utf-8")
        files = [Path(entry[0]).stem for entry in _spec_list(source, "app_files")]
        modules = _spec_list(source, "app_modules")
        for name in SHARED:
            stem = name[:-3]
            assert files.count(stem) == 1 and modules.count(stem) == 1, (spec_path.name, stem)
