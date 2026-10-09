"""U10 acceptance item 2: the per-file fallback ("Save each file separately") and the phone folder.

The REAL app on a fake Android 14 session (``test_bootstrap._fake_session``) with the REAL
``flet_glossarion_native.GlossarionNative`` service. Only the phone below the plugin's method channel is fake:

* the document API (``DocumentDestinations.kt``) is answered by the extension's own Kotlin stand-in
  ``documents_fake.FakeDocumentsNative`` over two document providers: Google Drive (``supports_tree=False``:
  ``ACTION_OPEN_DOCUMENT_TREE`` does not offer it, ``ACTION_CREATE_DOCUMENT`` does) and the phone's own storage,
  whose ``Download/Glossarion`` is Glossarion's folder (the plugin's ``isOwnFolder`` sets ``own_folder``);
* MediaStore Downloads for ``save_to_downloads(replace_uri=)`` (``GlossarionNativePlugin.saveToDownloads``: a row
  Glossarion owns is overwritten in place and its DISPLAY_NAME never changes; anything else gets a new row, and
  MediaStore makes a taken name unique with " (1)");
* the foreground job service, notifications and the permission plugin.

Recompiles are real ``compile_epub`` jobs (``LibraryService.compile_spec`` + ``submit``, the Book page's Compile)
through the real JobService; only the shared desktop EPUB compiler is a stand-in that writes the next version of
the book's files (``job_kinds.compile.owner_method``). Nothing touches the network (a socket guard records any
non-loopback connect); every path is under ``tmp_path`` (``app_env`` + USERPROFILE / APPDATA).

Android scenario:
 1. Settings › Cloud sync & sharing lists the three destinations; the phone folder tile and the
    "My cloud app isn't listed" help explain the TeraBox / folder-backup route (Download/Glossarion, the backup
    app's own folder access, RSAF, Save each file separately), and Settings › Storage opens the same help.
 2. "Choose a folder…": Drive is not offered as a folder, the user backs out: nothing is linked.
 3. "Choose a folder…" -> Download/Glossarion (Glossarion's own folder): refused, the grant given back.
 4. "Save each file separately" + "Copy finished books automatically": no picker opens by itself, the book's
    EPUB "needs a save location", one notification whose payload is a route (never a path or a URI).
 5. The Book page › Output tab: a tap on the EPUB's cloud line opens the system Save dialog once and the file is
    written into the chosen Drive file at once (from a private snapshot, never the workspace file).
 6. Recompiles update the same Drive document by themselves (no picker, same document, exact bytes; a shorter
    book leaves no stale tail), also after a title change (the cloud file keeps its first name).
 7. A NEW output (Compile PDF) needs its own save location: one notification, the EPUB of the same run is
    updated without asking; a later drain neither asks nor notifies again; a cancelled Save dialog changes
    nothing; a save location inside Glossarion's own folder is refused (and leaves no file behind); then the PDF
    is saved once and updated automatically.
 8. Change to the phone folder (asks first; the per-file grants are given back): every output goes to
    Download/Glossarion/<book>/; recompiles overwrite the SAME MediaStore row (same URI, same name, one row), also
    after a title change; with the copy-once switch on, books still land only once; a row the user deleted is
    created again once.
 9. Settings › Storage says the phone folder's books are replaced in place.

Android upgrade: a user who had the pre-U10 "Mirror outputs to Downloads/Glossarion" switch on gets the phone folder
as the destination (copies on, shown in use); recompiles overwrite one row in Download/Glossarion/<book>/.

iOS: Files › On My iPhone › Glossarion (the app's own storage) is refused both when the plugin flags it and when
only its path says so (the Python safety net); an iCloud Drive folder is accepted.

Defects that do not stop the Android scenario are collected (``Harness.problems``) and fail it at the end, all listed.

Run from src/mobile with the mobile venv:
    .venv/Scripts/python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_u10_accept2.py
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import importlib.util
import itertools
import json
import os
import socket
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Optional

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = MOBILE_DIR / "app"
TESTS_DIR = MOBILE_DIR / "tests"
SRC_DIR = MOBILE_DIR.parent
EXTENSION_SRC = MOBILE_DIR / "extensions" / "flet_glossarion_native" / "src"
for _path in (APP_DIR, TESTS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


def _has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _native_extension() -> bool:
    return _has("flet_glossarion_native") or (EXTENSION_SRC / "flet_glossarion_native" / "__init__.py").is_file()


pytestmark = [
    pytest.mark.skipif(not (_has("flet") and _has("msgpack")), reason="flet / msgpack not installed"),
    pytest.mark.skipif(not _native_extension(), reason="flet_glossarion_native extension sources not present"),
    pytest.mark.skipif(not (_has("requests") and _has("bs4")), reason="the real features drive the backend"),
]

_TB_SPEC = importlib.util.spec_from_file_location("_glossarion_tb_helpers_u10a2",
                                                  Path(__file__).with_name("test_bootstrap.py"))
_TB = importlib.util.module_from_spec(_TB_SPEC)
_TB_SPEC.loader.exec_module(_TB)
storage = _TB.storage
app_env = _TB.app_env

PACKAGE = "com.glossarion.app"
DRIVE = "com.google.android.apps.docs.storage"
STORAGE = "com.android.externalstorage.documents"
EPUB_MIME = "application/epub+zip"
BOOK = "Moonlit Sword"
NEW_TITLE = "Sword of the Moon"
DOC_METHODS = frozenset({
    "pick_folder", "pick_save_location", "pick_document", "list_children", "create_file", "create_folder",
    "write_file", "rename_document", "stat", "delete", "query_root", "release", "list_grants",
    "cancel_document_op", "take_document_results",
})


def _foundations():
    spec = importlib.util.spec_from_file_location("_glossarion_tf_helpers_u10a2",
                                                  Path(__file__).with_name("test_ui_foundations.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest() if path.is_file() else ""


def _norm(path: Any) -> str:
    return os.path.normcase(os.path.abspath(str(path)))


async def _until(predicate, timeout: float = 30.0, interval: float = 0.05):
    deadline = time.monotonic() + timeout
    while True:
        value = predicate()
        if value:
            return value
        if time.monotonic() >= deadline:
            return value
        await asyncio.sleep(interval)


# ==========================================================================
# Books
# ==========================================================================


def make_epub(path: Path, title: str, text: str) -> bytes:
    """A small valid EPUB (stored, so its size follows ``text``); returns its bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("mimetype", EPUB_MIME)
        zf.writestr("META-INF/container.xml",
                    '<?xml version="1.0"?><container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:'
                    'container"><rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-'
                    'package+xml"/></rootfiles></container>')
        zf.writestr("OEBPS/content.opf",
                    '<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf" version="3.0"><metadata '
                    f'xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>{title}</dc:title><dc:creator>Author'
                    '</dc:creator><dc:language>en</dc:language></metadata><manifest><item id="c0" href="ch001.xhtml" '
                    'media-type="application/xhtml+xml"/></manifest><spine><itemref idref="c0"/></spine></package>')
        zf.writestr("OEBPS/ch001.xhtml", f"<html><head><title>{title}</title></head><body><p>{text}</p></body></html>")
    return path.read_bytes()


def make_workspace(output: Path, library: Path) -> Path:
    """Output/<BOOK>: a finished EPUB translation (raw in Library/Raw, response files, progress) with its
    compiled EPUB; returns the workspace."""
    raw = library / "Raw" / f"{BOOK}.epub"
    make_epub(raw, f"{BOOK} raw", "raw chapter")
    ws = output / BOOK
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "source_epub.txt").write_text(str(raw), encoding="utf-8")
    (ws / "response_ch001.html").write_text("<html><head><title>One</title></head><body>one</body></html>",
                                            encoding="utf-8")
    (ws / "translation_progress.json").write_text(json.dumps({"version": "2.1", "chapters": {
        "1": {"actual_num": 1, "status": "completed", "output_file": "response_ch001.html",
              "original_basename": "ch001.xhtml", "content_hash": "a"}}}), encoding="utf-8")
    (ws / "metadata.json").write_text(json.dumps({"title": BOOK, "creator": "Author"}), encoding="utf-8")
    make_epub(ws / f"{BOOK}.epub", BOOK, "version 1 " * 40)
    _bump(ws / f"{BOOK}.epub")
    return ws


_MTIME = itertools.count(1)


def _bump(path: Path) -> None:
    """Each compile leaves a newer mtime (the compiler writes the file again)."""
    t = time.time() + next(_MTIME) * 5
    os.utime(path, (t, t))


class Compiler:
    """Stand-in for the shared EPUB compiler (``TextJobsMixin._run_epub_compile``) of the real ``compile_epub``
    job: writes the planned files into the workspace and reports the EPUB, like ``CompileResult``."""

    def __init__(self) -> None:
        self.plans: dict = {}
        self.runs: list = []

    def plan(self, ws: Path, files: dict) -> None:
        self.plans[_norm(ws)] = dict(files)

    def __call__(self, folder: str) -> dict:
        files = self.plans.pop(_norm(folder))
        written = []
        for name, (title, text) in files.items():
            path = Path(folder) / name
            if name.lower().endswith(".epub"):
                make_epub(path, title, text)
            else:
                path.write_bytes(f"%PDF-1.7 {title} {text}".encode("utf-8"))
            _bump(path)
            written.append(str(path))
        self.runs.append([os.path.basename(p) for p in written])
        return {"ok": True, "outputs": [p for p in written if p.lower().endswith(".epub")]}


# ==========================================================================
# The phone below GlossarionNative
# ==========================================================================


class MediaStore:
    """MediaStore Downloads as ``GlossarionNativePlugin.saveToDownloads`` (API 29+) uses it."""

    def __init__(self) -> None:
        self.rows: dict = {}
        self._ids = itertools.count(1000)
        self.calls: list = []

    @staticmethod
    def uri(rid: int) -> str:
        return f"content://media/external/downloads/{rid}"

    def save(self, args: dict) -> str:
        self.calls.append(dict(args))
        source = Path(str(args["path"]))
        if not source.is_file():
            raise FileNotFoundError(str(source))
        data = source.read_bytes()
        name = str(args.get("display_name") or source.name)
        folder = "Download/" + str(args.get("subdir") or "").strip("/")
        replace = args.get("replace_uri")
        if replace:
            row = self.rows.get(int(str(replace).rsplit("/", 1)[-1])) if str(replace).rsplit("/", 1)[-1].isdigit() \
                else None
            if row is not None and row["owner"] == PACKAGE:  # ownsDownloadsEntry -> overwrite 'wt' in place
                row["data"] = data
                row["versions"] += 1
                return str(replace)
        taken = {r["name"] for r in self.rows.values() if r["folder"] == folder}
        final, n = name, 1
        stem, ext = os.path.splitext(name)
        while final in taken:  # MediaStore keeps names unique in a folder
            final = f"{stem} ({n}){ext}"
            n += 1
        rid = next(self._ids)
        self.rows[rid] = {"folder": folder, "name": final, "data": data, "owner": PACKAGE, "versions": 1,
                          "mime": args.get("mime_type")}
        return self.uri(rid)

    def files(self, folder: str) -> dict:
        return {r["name"]: r["data"] for r in self.rows.values() if r["folder"] == folder}

    def user_deletes(self, uri: str) -> None:
        self.rows.pop(int(uri.rsplit("/", 1)[-1]))


class Phone:
    """An Android 14 phone as GlossarionNative's method channel sees it."""

    def __init__(self, platform: str = "android") -> None:
        from flet_glossarion_native.documents_fake import FakeCloudProvider, FakeDocumentsNative

        self.platform = platform
        self.drive = FakeCloudProvider(authority=DRIVE, label="Drive", supports_tree=False)
        self.storage = FakeCloudProvider(authority=STORAGE, label="Phone storage")
        self.download = self.storage.add_folder("Download")
        self.own = self.storage.add_folder("Glossarion", parent=self.download)  # Glossarion's own folder
        self.books = self.drive.add_folder("Books")
        self.docs = {DRIVE: FakeDocumentsNative(self.drive), STORAGE: FakeDocumentsNative(self.storage)}
        self.media = MediaStore()
        self.user: list = []  # what the user does in the next system pickers: (authority, node) or None (back out)
        self.pickers: list = []  # (kind, args) of every system picker the app opened
        self.calls: list = []  # (control, method)
        self.posted: list = []  # accepted show_notification args
        self.fgs = None
        self.fgs_log: list = []
        self.ios_answers: list = []  # iOS: the plugin's pick_folder answers, in order

    # ---- synchronous plugin methods ------------------------------------------------------------------

    def answer(self, control: str, method: str, args: dict) -> Any:
        self.calls.append((control, method))
        if control == "PermissionHandler":
            return "granted" if method in ("request", "get_status") else True
        if control != "GlossarionNative":
            return None
        if method == "get_platform_info":
            if self.platform == "ios":
                return {"platform": "ios", "system_version": "18.0", "notifications_enabled": True,
                        "continued_processing": False, "documents": True}
            return {"platform": "android", "sdk_int": 34, "notifications_enabled": True,
                    "post_notifications_granted": True, "fgs_types": ["dataSync"], "save_to_downloads": True,
                    "documents": True, "persisted_grant_limit": 512}
        if method in ("get_initial_shared", "take_document_results"):
            return []
        if method == "init_notifications":
            return True
        if method == "show_notification":
            self.posted.append(dict(args))
            return True
        if method == "start_job_service":
            self.fgs = {"title": args.get("title"), "text": args.get("text")}
            self.fgs_log.append(("start", args.get("text")))
            return True
        if method == "update_job_service":
            if self.fgs is None:
                return False
            self.fgs["text"] = args.get("text") or self.fgs["text"]
            self.fgs_log.append(("update", args.get("text")))
            return True
        if method == "stop_job_service":
            self.fgs = None
            self.fgs_log.append(("stop", None))
            return True
        if method == "is_job_service_running":
            return self.fgs is not None
        if method == "save_to_downloads":
            return self.media.save(args)
        if method == "begin_background_task":
            return 7
        return None

    # ---- the document API (async: answered like DocumentDestinations.kt) -------------------------------

    @staticmethod
    def _authority(ref: Any) -> Optional[str]:
        if isinstance(ref, dict):
            for key in ("uri", "document"):
                value = ref.get(key)
                if isinstance(value, str) and value.startswith("content://"):
                    return value.split("/")[2]
            return ref.get("provider")
        if isinstance(ref, str) and ref.startswith("content://"):
            return ref.split("/")[2]
        return None

    def _node(self, authority: str, ref: dict) -> Any:
        provider = self.drive if authority == DRIVE else self.storage
        uri = str(ref.get("document") or "")
        return provider.nodes.get(uri.rstrip("/").rsplit("/", 1)[-1])

    def _is_own(self, authority: str, node: Any) -> bool:
        """``DocumentDestinations.isOwnFolder``: Download/Glossarion (and below) of the phone's storage."""
        while authority == STORAGE and node is not None:
            if node is self.own:
                return True
            node = node.parent
        return False

    async def document(self, method: str, args: dict) -> Any:
        self.calls.append(("GlossarionNative", method))
        if method in ("pick_folder", "pick_save_location", "pick_document"):
            if self.platform == "ios":
                return self.ios_answers.pop(0) if self.ios_answers else {"ok": False, "error": "cancelled"}
            return await self._pick(method, args)
        if method == "take_document_results":
            return []
        if method == "list_grants":
            return [g for docs in self.docs.values() for g in await docs.list_grants()]
        if method == "cancel_document_op":
            return any([await docs.cancel_document_op(str(args.get("op_id") or "")) for docs in self.docs.values()])
        ref = next((args.get(k) for k in ("folder", "ref", "document", "target") if args.get(k) is not None), None)
        docs = self.docs.get(self._authority(ref) or "")
        if docs is None:
            return {"ok": False, "error": "bad_args", "message": "unknown provider", "retryable": False}
        if method == "list_children":
            return await docs.list_children(args["folder"], names=args.get("names"))
        if method == "create_file":
            return await docs.create_file(args["folder"], args["name"], args.get("mime_type"),
                                          on_exists=args.get("on_exists") or "rename")
        if method == "create_folder":
            return await docs.create_folder(args["folder"], args["name"], on_exists=args.get("on_exists") or "adopt")
        if method == "write_file":
            return await docs.write_file(args["ref"], args.get("source_path"), name=args.get("name"),
                                         mime_type=args.get("mime_type"), on_exists=args.get("on_exists") or "rename",
                                         mode_chain=args.get("mode_chain") or ("wt", "rwt", "w"),
                                         verify=bool(args.get("verify", True)), op_id=args.get("op_id"))
        if method == "rename_document":
            return await docs.rename_document(args["document"], args["name"])
        if method == "stat":
            return await docs.stat(args["document"])
        if method == "delete":
            return await docs.delete(args["document"])
        if method == "query_root":
            return await docs.query_root(args["target"])
        if method == "release":
            return await docs.release(args["target"])
        return None

    async def _pick(self, method: str, args: dict) -> dict:
        kind = {"pick_folder": "folder", "pick_save_location": "save_location", "pick_document": "document"}[method]
        self.pickers.append((kind, dict(args)))
        choice = self.user.pop(0) if self.user else None
        if choice is None:  # the user backed out of the system picker
            return {"ok": False, "error": "cancelled", "message": "No location was chosen", "retryable": False}
        authority, node = choice
        docs = self.docs[authority]
        docs.provider.next_pick(kind, node)
        if kind == "folder":
            out = await docs.pick_folder(initial=args.get("initial"), op_id=args.get("op_id"))
        elif kind == "save_location":
            out = await docs.pick_save_location(args.get("name"), args.get("mime_type"), args.get("source_path"),
                                                initial=args.get("initial"),
                                                mode_chain=args.get("mode_chain") or ("wt", "rwt", "w"),
                                                op_id=args.get("op_id"))
        else:
            out = await docs.pick_document(args.get("mime_types"), op_id=args.get("op_id"))
        ref = out.get("target") or out.get("document") if out.get("ok") else None
        if isinstance(ref, dict) and self._is_own(authority, self._node(authority, ref)):
            ref["own_folder"] = True
        return out


def _install_phone(conn, session, phone: Phone) -> None:
    """Answer the app's invoke_method calls from ``phone`` (the plain fake client answers None)."""
    from flet.messaging.protocol import MessageAction

    pending: dict = {}
    send = conn.send_message

    def send_message(message):
        if message.action == MessageAction.INVOKE_METHOD:
            pending[message.body.call_id] = message.body
        send(message)

    real_handle = type(session).handle_invoke_method_results

    def handle(control_id, call_id, result, error):
        body = pending.pop(call_id, None)
        if body is None:
            real_handle(session, control_id, call_id, result, error)
            return
        control = type(session.index.get(control_id)).__name__
        args = dict(body.args) if isinstance(body.args, dict) else {}
        if control == "GlossarionNative" and body.name in DOC_METHODS:
            async def answer():
                try:
                    value, err = await phone.document(body.name, args), None
                except Exception as exc:  # pragma: no cover - a broken fake must show up
                    value, err = None, f"phone: {exc!r}"
                real_handle(session, control_id, call_id, value, err)

            asyncio.ensure_future(answer())
            return
        try:
            value, err = phone.answer(control, body.name, args), None
        except Exception as exc:  # pragma: no cover
            value, err = None, f"phone: {exc!r}"
        real_handle(session, control_id, call_id, value, err)

    conn.send_message = send_message
    session.handle_invoke_method_results = handle


# ==========================================================================
# Harness
# ==========================================================================


def _isolate(tmp_path: Path, monkeypatch) -> list:
    """USERPROFILE / APPDATA into tmp (``app_env`` moved HOME, data, Output and the Library); no network: every
    non-loopback connect is recorded (and refused)."""
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    for name in ("USERPROFILE", "APPDATA", "LOCALAPPDATA"):
        monkeypatch.setenv(name, str(home))
    monkeypatch.setenv("GLOSSARION_HTTP_LOG", "0")
    attempts: list = []
    original, original_ex = socket.socket.connect, socket.socket.connect_ex

    def guard(fn):
        def connect(self, address, *args, **kwargs):
            host = address[0] if isinstance(address, tuple) and address else address
            if host not in ("127.0.0.1", "::1", "localhost"):
                attempts.append(repr(address))
                raise OSError(f"network access attempted: {address!r}")
            return fn(self, address, *args, **kwargs)

        return connect

    monkeypatch.setattr(socket.socket, "connect", guard(original))
    monkeypatch.setattr(socket.socket, "connect_ex", guard(original_ex))
    from glossarion_mobile import runtime_bootstrap as rb

    sandbox_config = str(rb.get_paths().config_file)
    app_paths = sys.modules.get("app_paths")
    if app_paths is not None and getattr(app_paths, "CONFIG_FILE", None) != sandbox_config:
        old = app_paths.CONFIG_FILE  # imported by an earlier test file: never src/config.json
        for module in list(sys.modules.values()):
            try:
                if module is not None and vars(module).get("CONFIG_FILE") == old:
                    monkeypatch.setattr(module, "CONFIG_FILE", sandbox_config)
            except Exception:
                pass
    return attempts


class Harness:
    def __init__(self, app, page, session, tester, phone: Phone, snacks: list) -> None:
        self.app = app
        self.page = page
        self.session = session
        self.t = tester
        self.phone = phone
        self.snacks = snacks
        self.problems: list = []  # defects that do not stop the rest of the scenario (all listed at the end)

    @property
    def cloud(self):
        return self.app.cloud_sync

    async def go(self, route: str):
        match = await self.app.navigate(route)
        assert match is not None, route
        await asyncio.sleep(0.1)
        return self.app.shell.top_screen

    def _find(self, *, key: Optional[str] = None, key_prefix: Optional[str] = None,
              text: Optional[str] = None) -> list:
        from host_tester import _key_value, _texts

        out = []
        for control in self.t._walk():
            own = str(_key_value(getattr(control, "key", None)) or "")
            if key is not None and own == key:
                out.append(control)
            elif key_prefix is not None and own.startswith(key_prefix):
                out.append(control)
            elif text is not None and text in _texts(control):
                out.append(control)
        return out

    async def tap(self, *, key: Optional[str] = None, key_prefix: Optional[str] = None, text: Optional[str] = None,
                  timeout: float = 10.0):
        found = await _until(lambda: self._find(key=key, key_prefix=key_prefix, text=text), timeout=timeout)
        assert found, f"nothing to tap: key {key or key_prefix!r} text {text!r}\n" + "\n".join(
            map(str, self.t.dump(120)))
        await self.t.tap(self.t._register(found[:1]))
        await asyncio.sleep(0.05)

    def texts(self) -> list:
        from host_tester import _texts

        return [t for control in self.t._walk() for t in _texts(control)]

    def state(self) -> dict:
        return self.cloud.ui_state()

    def book_files(self, ws: Path) -> dict:
        return self.cloud.book_state(str(ws))["files"]

    def records(self, ws: Path) -> dict:
        dest = self.cloud.destination()
        key = self.cloud.store.resolve(_norm(ws)) if hasattr(self.cloud.store, "resolve") else _norm(ws)
        return {kind: self.cloud.store.record(dest.id, key, kind) or {} for kind in ("epub", "pdf", "txt", "html")}

    async def idle(self) -> None:
        for _ in range(4):
            await asyncio.sleep(0.05)
            await self.cloud.wait_idle()

    def book_row(self, ws: Path) -> dict:
        from glossarion_mobile.services.library import book_identity

        rows = [b for b in self.app.library.snapshot.all_books() if _norm(book_identity(b)) == _norm(ws)]
        return dict(rows[0]) if rows else {}

    async def compile(self, ws: Path, kind: str = "compile_epub") -> Any:
        """The Book page's Compile: a real job through the real JobService, awaited until it ends."""
        service = self.app.library
        before = set(j.id for j in self._jobs())
        job_id = await service.submit(service.compile_spec(self.book_row(ws), kind, folder=str(ws)))
        assert job_id, "the compile job did not start"
        _step(f"   {kind} job {job_id} submitted")
        snap = await _until(lambda: (s := self.app.job_service.snapshot(job_id)) is not None and s.is_terminal and s,
                            timeout=60)
        assert snap and str(getattr(snap.state, "value", snap.state)) == "DONE", (job_id, snap, before)
        await self.idle()
        assert await _until(lambda: self.app.job_service.view().active is None, timeout=10), \
            "timed out: self.app.job_service.view().active is None"
        await self.idle()
        return snap

    def _jobs(self) -> list:
        view = self.app.job_service.view()
        return ([view.active] if view.active is not None else []) + list(view.queue or ())

    def needs_location_posts(self) -> list:
        return [p for p in self.phone.posted if "save location" in str(p.get("title") or "")]


async def _start(phone: Phone, platform: str = "android", *, prefs: Optional[dict] = None):
    from glossarion_mobile import runtime_bootstrap as rb
    from glossarion_mobile.services.background import PREF_BATTERY_PROMPT, PREF_NOTIFICATION_ASKED

    # A returning user: the first long job's notification / battery questions were answered already.
    state_file = Path(rb.get_paths().data) / "mobile_state.json"
    state = json.loads(state_file.read_text(encoding="utf-8")) if state_file.is_file() else {}
    state.update({PREF_BATTERY_PROMPT: True, PREF_NOTIFICATION_ASKED: True})
    state.update(prefs or {})
    state_file.write_text(json.dumps(state), encoding="utf-8")
    tf = _foundations()
    main_module = tf._load_main_module()
    conn, session = _TB._fake_session(platform)
    _install_phone(conn, session, phone)
    session.apply_page_patch({"width": 412, "height": 860})
    page = session.page
    await main_module.main(page)
    await session.after_event(page)
    return tf, conn, session, page, page.data


async def _shutdown(tf, app) -> None:
    try:
        app.job_service.request_stop(force=True, reason="test teardown")
        await asyncio.to_thread(app.job_service.wait_idle, 30)
    except Exception:
        pass
    cloud = getattr(app, "cloud_sync", None)
    if cloud is not None:
        try:
            await cloud.wait_idle()
        except Exception:
            pass
    try:
        app.jobs.close()
    except Exception:
        pass
    await tf._stop(app)
    if cloud is not None:
        cloud.close()


# ==========================================================================
# Android
# ==========================================================================


def _android_setup(tmp_path: Path, monkeypatch) -> tuple:
    """Isolation + network guard, the book workspace, the compiler stand-in and the snackbar spy."""
    if not _has("flet_glossarion_native"):
        monkeypatch.syspath_prepend(str(EXTENSION_SRC))
    attempts = _isolate(tmp_path, monkeypatch)
    from glossarion_mobile import runtime_bootstrap as rb
    from glossarion_mobile.job_kinds import compile as compile_kind

    paths = rb.get_paths()
    output, library = Path(paths.output), Path(paths.library)
    for real in (Path.home() / "Documents" / "Glossarion", SRC_DIR):
        assert not _norm(output).startswith(_norm(real)) and not _norm(library).startswith(_norm(real))
    ws = make_workspace(output, library)
    compiler = Compiler()
    real_owner_method = compile_kind.owner_method

    def owner_method(owner, name):
        if name == "_run_epub_compile":
            return compiler
        return real_owner_method(owner, name)

    monkeypatch.setattr(compile_kind, "owner_method", owner_method)
    from glossarion_mobile import app as app_module
    from glossarion_mobile.ui.components import dialogs as dialogs_module

    snacks: list = []
    real_snackbar = dialogs_module.show_snackbar

    def show_snackbar(page, message, **kwargs):
        snacks.append(str(message))
        return real_snackbar(page, message, **kwargs)

    monkeypatch.setattr(app_module, "show_snackbar", show_snackbar)
    return attempts, ws, compiler, snacks


def test_per_file_fallback_and_phone_folder_on_android(app_env, tmp_path, monkeypatch):
    src_config = SRC_DIR / "config.json"
    src_config_md5 = _md5(src_config)
    attempts, ws, compiler, snacks = _android_setup(tmp_path, monkeypatch)

    async def scenario():
        from host_tester import PyTester

        phone = Phone()
        tf, _conn, session, page, app = await _start(phone)
        h = Harness(app, page, session, PyTester(session, page), phone, snacks)
        try:
            assert await _until(lambda: app.state.engine_ready, timeout=60), "timed out: app.state.engine_ready"
            native = app.native.native
            assert type(native).__name__ == "GlossarionNative" and not app.native.is_stub
            assert h.cloud is not None and h.cloud.supported and h.cloud.platform == "android"
            await app.library.refresh(quiet=True, reason="test")
            assert h.book_row(ws), "the workspace is not a Library book"
            await _android_scenario(h, ws, compiler)
            assert attempts == [], f"network access attempted: {attempts}"
            assert not h.problems, "\n\n".join(h.problems)
        except BaseException:
            print("\n--- phone calls (last 80) ---")
            for row in phone.calls[-80:]:
                print(row)
            print("--- pickers ---")
            for row in phone.pickers:
                print(row[0], {k: v for k, v in row[1].items() if k != "source_path"})
            print("--- notifications ---", phone.posted)
            print("--- foreground service ---", phone.fgs_log)
            print("--- snackbars ---", snacks[-20:])
            print("--- cloud events ---", list(getattr(h.cloud, "events", [])))
            print("--- screen ---")
            for row in h.t.dump(150):
                print(row)
            raise
        finally:
            await _shutdown(tf, app)

    asyncio.run(scenario())
    assert _md5(src_config) == src_config_md5, "src/config.json changed"


def _step(text: str) -> None:
    print(f"[u10-accept2] {time.strftime('%H:%M:%S')} {text}", flush=True)


async def _android_scenario(h: Harness, ws: Path, compiler: Compiler) -> None:
    from glossarion_mobile.services.cloud_sync import OWN_FOLDER_REASON
    from glossarion_mobile.ui.screens import cloud_sync as u10
    from glossarion_mobile.ui.screens.storage import NOT_LISTED_TITLE

    app, phone = h.app, h.phone
    drive, storage = phone.drive, phone.storage

    # ---- 1. Settings › Cloud sync & sharing: destinations and the TeraBox / folder-backup help -----------------
    _step("1. Settings › Cloud sync & sharing: destinations and the TeraBox / folder")
    screen = await h.go("/settings/cloud")
    assert type(screen).__name__ == "CloudSyncScreen"
    assert await _until(lambda: screen.loaded, timeout=10), "timed out: screen.loaded"
    texts = h.texts()
    for title in ("Choose a folder…", "Save each file separately", "Phone folder (Downloads/Glossarion)"):
        assert title in texts, (title, texts)
    phone_tile = next(t for t in texts if "TeraBox" in t and "Downloads/Glossarion" in t)
    assert "backup app" in phone_tile
    help_tile = h._find(key_prefix="cloud-not-listed")
    assert help_tile and type(help_tile[0]).__name__ == "ExpansionTile"
    await h.tap(key_prefix="cloud-not-listed")
    help_text = "\n".join(t for c in h._find(key_prefix="cloud-not-listed") for t in _md_values(c))
    assert help_text == u10.NOT_LISTED_HELP_ANDROID, help_text
    for needle in ("TeraBox", "**phone folder** (Downloads/Glossarion)", "Download/Glossarion",
                   "All files access", "RSAF", "**Save each file separately**", "Share"):
        assert needle in help_text, needle
    assert h.state()["destination"] is None and not h.state()["enabled"]
    # the auto switch waits for a destination (ReasonChip, never silently off)
    assert screen.auto_switch.disabled and u10.NO_DESTINATION_REASON in _chip_reasons(screen.auto_reason)

    # ---- 2. "Choose a folder…": Drive is not offered as a folder; the user backs out -------------------------
    _step("2. 'Choose a folder…': Drive is not offered as a folder; the user backs o")
    phone.user.append(None)
    await h.tap(text="Choose a folder…")
    assert await _until(lambda: phone.pickers and not h.cloud.settings().pending_pick, timeout=10), \
        "timed out: phone.pickers and not h.cloud.settings().pending_pick"
    await h.idle()
    assert [p[0] for p in phone.pickers] == ["folder"]
    assert h.cloud.destination() is None and drive.grants == {} and storage.grants == {}
    assert not any("could not" in s.lower() for s in h.snacks), h.snacks  # a cancelled picker says nothing

    # ---- 3. Glossarion's own folder is refused (and its grant given back) ---------------------------------------
    _step("3. Glossarion's own folder is refused (and its grant given back)")
    phone.user.append((STORAGE, phone.own))
    await h.tap(text="Choose a folder…")
    assert await _until(lambda: OWN_FOLDER_REASON in h.snacks, timeout=10), "timed out: OWN_FOLDER_REASON in h.snacks"
    assert len(phone.pickers) == 2 and phone.pickers[-1][0] == "folder"
    assert h.cloud.destination() is None, "Glossarion's own folder became the destination"
    assert storage.grants == {}, "the refused folder's permission was kept"
    assert ("GlossarionNative", "release") in phone.calls

    # ---- 4. Save each file separately + copy automatically: nothing is written, one notification ----------------
    _step("4. Save each file separately + copy automatically: nothing is written, on")
    await h.tap(text="Save each file separately")
    assert await _until(lambda: (h.cloud.destination() or None) is not None, timeout=10), \
        "timed out: (h.cloud.destination() or None) is not None"
    dest = h.cloud.destination()
    assert dest.mode == "files" and h.state()["destination"]["mode"] == "files"
    assert await _until(lambda: "Save locations · one file at a time" in h.texts(), timeout=10), \
        "timed out: 'Save locations · one file at a time' in h.texts()"
    await h.tap(key="cloud-auto")
    assert await _until(lambda: h.cloud.settings().enabled, timeout=10), "timed out: h.cloud.settings().enabled"
    assert await _until(lambda: h.needs_location_posts(), timeout=15), "timed out: h.needs_location_posts()"
    await h.idle()
    pickers_before = len(phone.pickers)
    assert pickers_before == 2, "a system picker opened without a tap"
    assert drive.all_files() == [] and not [c for c in phone.calls if c[1] == "write_file"]
    files = h.book_files(ws)
    assert files["epub"]["status"] == "needs_pick", files
    posts = h.needs_location_posts()
    assert len(posts) == 1, posts
    payload = str(posts[0].get("payload") or "")
    bid = app.library.bid_for(h.book_row(ws))
    assert posts[0]["title"] == "1 book needs a save location"
    assert payload.endswith(f"/library/book/{bid}?tab=output"), payload
    assert str(ws) not in json.dumps(posts) and "content://" not in json.dumps(posts) and BOOK not in payload

    # ---- 5. Book page › Output: one tap opens the Save dialog once, the file is written at once ------------------
    _step("5. Book page › Output: one tap opens the Save dialog once, the file is wr")
    book = await h.go(f"/library/book/{bid}?tab=output")
    assert type(book).__name__ == "BookPageScreen"
    output_tab = book.output
    assert await _until(lambda: output_tab.outputs and output_tab.cloud_line(0) is not None, timeout=15), \
        "timed out: output_tab.outputs and output_tab.cloud_line(0) is not None"
    epub_index = next(i for i, (p, k) in enumerate(output_tab.outputs) if k == "epub")
    icon, text, tone = output_tab.cloud_line(epub_index)
    assert text == "Choose where to save · tap to choose" and tone == "action", text
    snap_dir = Path(h.cloud._snapshot_dir())
    target = drive.add_file(f"{BOOK}.epub", parent=phone.books)  # the document the Save dialog creates
    phone.user.append((DRIVE, target))
    await h.tap(key_prefix=f"out-cloud-line-{epub_index}-")
    assert await _until(lambda: bytes(target.data) == (ws / f"{BOOK}.epub").read_bytes(), timeout=15), \
        "timed out: bytes(target.data) == (ws / f'{BOOK}.epub').read_bytes()"
    await h.idle()
    saves = [p for p in phone.pickers if p[0] == "save_location"]
    assert len(saves) == 1, saves
    save_args = saves[0][1]
    assert save_args["name"] == f"{BOOK}.epub" and save_args["mime_type"] == EPUB_MIME
    snapshot = Path(save_args["source_path"])
    assert _norm(snapshot.parent) == _norm(snap_dir) and not snapshot.exists(), "not a private snapshot / kept"
    assert "wt" in drive.opened_modes and "w" not in drive.opened_modes  # truncating first ("r": the read-back)
    assert h.book_files(ws)["epub"]["status"] == "ok"
    assert await _until(lambda: "Saved to" in (output_tab.cloud_line(epub_index) or ("", "", ""))[1], timeout=10), \
        "timed out: 'Saved to' in (output_tab.cloud_line(epub_index) or ('', '', ''))[1]"
    first_doc = h.records(ws)["epub"]["doc"]["document"]  # the document URI (the ref's size / mtime change)
    assert first_doc == drive.doc_uri(target)
    assert [n.name for n in drive.all_files()] == [f"{BOOK}.epub"]

    # ---- 6. recompiles update the same Drive file automatically ----------------------------------------------
    _step("6. recompiles update the same Drive file automatically")
    pickers = len(phone.pickers)
    compiler.plan(ws, {f"{BOOK}.epub": (BOOK, "version 2 is longer " * 80)})
    await h.compile(ws)
    v2 = (ws / f"{BOOK}.epub").read_bytes()
    assert await _until(lambda: bytes(target.data) == v2, timeout=15), "timed out: bytes(target.data) == v2"
    assert len(phone.pickers) == pickers, "a recompile asked for a save location again"
    assert h.records(ws)["epub"]["doc"]["document"] == first_doc and len(drive.all_files()) == 1
    compiler.plan(ws, {f"{BOOK}.epub": (BOOK, "v3")})  # a much shorter book: no stale tail
    await h.compile(ws)
    v3 = (ws / f"{BOOK}.epub").read_bytes()
    assert len(v3) < len(v2)
    assert await _until(lambda: bytes(target.data) == v3, timeout=15), "timed out: bytes(target.data) == v3"
    assert len(phone.pickers) == pickers and len(drive.all_files()) == 1
    # a title change: the compiler writes "<new title>.epub" and leaves the old one; the cloud file keeps its name
    compiler.plan(ws, {f"{NEW_TITLE}.epub": (NEW_TITLE, "renamed book " * 30)})
    await h.compile(ws)
    renamed = (ws / f"{NEW_TITLE}.epub").read_bytes()
    assert (ws / f"{BOOK}.epub").is_file()
    assert await _until(lambda: bytes(target.data) == renamed, timeout=15), "timed out: bytes(target.data) == renamed"
    assert target.name == f"{BOOK}.epub" and len(drive.all_files()) == 1
    assert len(phone.pickers) == pickers and h.records(ws)["epub"]["doc"]["document"] == first_doc
    assert len(h.needs_location_posts()) == 1, "an update posted 'needs a save location' again"

    # ---- 7. a NEW output asks once; the EPUB of the same run updates without asking -----------------------------
    _step("7. a NEW output asks once; the EPUB of the same run updates without askin")
    compiler.plan(ws, {f"{NEW_TITLE}.epub": (NEW_TITLE, "with a pdf " * 30), f"{NEW_TITLE}.pdf": (NEW_TITLE, "pdf 1")})
    await h.compile(ws, "compile_pdf")
    epub_now = (ws / f"{NEW_TITLE}.epub").read_bytes()
    assert await _until(lambda: bytes(target.data) == epub_now, timeout=15), "timed out: bytes(target.data) == epub_now"
    assert await _until(lambda: len(h.needs_location_posts()) == 2, timeout=15), \
        "timed out: len(h.needs_location_posts()) == 2"
    files = h.book_files(ws)
    assert files["pdf"]["status"] == "needs_pick" and files["epub"]["status"] == "ok", files
    assert len(phone.pickers) == pickers, "a picker opened by itself for the new output"
    assert all(n.name != f"{NEW_TITLE}.pdf" for n in drive.all_files())
    # another drain (another recompile) neither asks nor notifies again for the waiting PDF
    compiler.plan(ws, {f"{NEW_TITLE}.epub": (NEW_TITLE, "again " * 31)})
    await h.compile(ws)
    epub_now = (ws / f"{NEW_TITLE}.epub").read_bytes()
    assert await _until(lambda: bytes(target.data) == epub_now, timeout=15), "timed out: bytes(target.data) == epub_now"
    h.cloud.retry_now()
    await h.idle()
    assert len(h.needs_location_posts()) == 2 and len(phone.pickers) == pickers
    assert h.book_files(ws)["pdf"]["status"] == "needs_pick"
    # the Output tab: the PDF row asks, the EPUB row is saved
    await output_tab.reload()
    listed = [os.path.basename(p) for p, _k in output_tab.outputs]
    pdf_rows = [i for i, (_p, k) in enumerate(output_tab.outputs) if k == "pdf"]
    if not pdf_rows:
        h.problems.append(
            f"Book page › Output has no row for the EPUB book's PDF ({NEW_TITLE}.pdf, made by Compile PDF; rows: "
            f"{listed}): OutputTab.reload lists LibraryService.compiled_outputs_blocking = "
            "library_core.list_compiled_outputs, which only knows '*_translated.pdf' (critic #0: 'The Output tab has "
            "the same blind spot'). In save-locations mode cloud sync marks that PDF 'needs a save location' and "
            "posts a notification that opens this tab, but there is no cloud line / 'Choose where to save…' row "
            "action for it: the user cannot give the PDF a save location (the chat card points here too).")
    else:
        line = output_tab.cloud_line(pdf_rows[0])
        if not line or line[1] != "Choose where to save · tap to choose":
            h.problems.append(f"the PDF row's cloud line does not ask for a save location: {line}")

    async def choose_pdf_location() -> None:
        """A tap on the PDF row's cloud line; without the row, the same action (``choose_location``)."""
        if pdf_rows and output_tab.cloud_line(pdf_rows[0]) is not None:
            await h.tap(key_prefix=f"out-cloud-line-{pdf_rows[0]}-")
        else:
            await output_tab.choose_location("pdf")

    # the user backs out of the Save dialog: nothing changes
    phone.user.append(None)
    await choose_pdf_location()
    assert await _until(lambda: len(phone.pickers) == pickers + 1, timeout=10), \
        "timed out: len(phone.pickers) == pickers + 1"
    await h.idle()
    assert h.book_files(ws)["pdf"]["status"] == "needs_pick" and drive.grants.keys() == {drive.doc_uri(target)}
    # a save location inside Glossarion's own folder is refused, and nothing is left there
    own_pdf = storage.add_file(f"{NEW_TITLE}.pdf", parent=phone.own)
    phone.user.append((STORAGE, own_pdf))
    snacks_before = len(h.snacks)
    await choose_pdf_location()
    assert await _until(lambda: OWN_FOLDER_REASON in h.snacks[snacks_before:], timeout=10), \
        "timed out: OWN_FOLDER_REASON in h.snacks[snacks_before:]"
    await h.idle()
    assert h.book_files(ws)["pdf"]["status"] == "needs_pick", "a save location in Glossarion's folder was kept"
    assert storage.grants == {}, "the refused file's permission was kept"
    pickers += 2
    leftovers = [n.name for n in storage.all_files() if n.data]
    if leftovers:
        h.problems.append(
            f"A save location inside Glossarion's own folder (Download/Glossarion/{NEW_TITLE}.pdf) is refused only "
            "after the native side already wrote the book there (DocumentDestinations.completePick writes "
            "source_path before Python's is_own_location check in CloudSyncService._finish_save_pick; the refusal "
            f"then only releases the grant): an untracked copy stays behind ({leftovers}) while the app says the "
            "folder was refused. iOS has the same order (DocumentDestinations.swift exports the staged copy with "
            "forExporting asCopy:false, then describe() sets own_folder), so a save location picked in Files › On "
            "My iPhone › Glossarion leaves the book inside the app's own folder. Check own_folder before writing "
            "(or delete the written document on refusal).")
    # then the PDF is saved once and updated automatically
    pdf_target = drive.add_file(f"{NEW_TITLE}.pdf", parent=phone.books)
    phone.user.append((DRIVE, pdf_target))
    await choose_pdf_location()
    assert await _until(lambda: bytes(pdf_target.data) == (ws / f"{NEW_TITLE}.pdf").read_bytes(), timeout=15), \
        "timed out: bytes(pdf_target.data) == (ws / f'{NEW_TITLE}.pdf').read_bytes()"
    pickers += 1
    compiler.plan(ws, {f"{NEW_TITLE}.epub": (NEW_TITLE, "pdf again " * 33), f"{NEW_TITLE}.pdf": (NEW_TITLE, "pdf 2!")})
    await h.compile(ws, "compile_pdf")
    pdf_now = (ws / f"{NEW_TITLE}.pdf").read_bytes()
    epub_now = (ws / f"{NEW_TITLE}.epub").read_bytes()
    assert await _until(lambda: bytes(pdf_target.data) == pdf_now and bytes(target.data) == epub_now, timeout=15), \
        "timed out: bytes(pdf_target.data) == pdf_now and bytes(target.data) == epub_now"
    assert len(phone.pickers) == pickers and len(h.needs_location_posts()) == 2
    assert sorted(n.name for n in drive.all_files()) == sorted([f"{BOOK}.epub", f"{NEW_TITLE}.pdf"])
    per_file_grants = set(drive.grants)
    assert per_file_grants == {drive.doc_uri(target), drive.doc_uri(pdf_target)}

    # ---- 8. the phone folder: Download/Glossarion/<book>, the same MediaStore row every time --------------------
    _step("8. the phone folder: Download/Glossarion/<book>, the same MediaStore row ")
    settings_screen = await h.go("/settings/cloud")
    assert await _until(lambda: settings_screen.loaded, timeout=10), "timed out: settings_screen.loaded"
    await h.tap(text="Phone folder (Downloads/Glossarion)")
    await h.tap(text="Change")  # "Change destination?" (copies already made stay where they are)
    assert await _until(lambda: (h.cloud.destination() and h.cloud.destination().mode) == "phone", timeout=10), \
        "timed out: (h.cloud.destination() and h.cloud.destination().mode) == 'phone'"
    await h.idle()
    assert drive.grants == {}, f"the per-file grants were kept after the change: {sorted(drive.grants)}"
    folder = f"Download/Glossarion/{BOOK}"
    both = {f"{NEW_TITLE}.epub", f"{NEW_TITLE}.pdf"}
    assert await _until(lambda: set(phone.media.files(folder)) == both, timeout=15), \
        f"the phone folder holds {sorted(phone.media.files(folder))}"
    rows = phone.media.files(folder)
    assert rows[f"{NEW_TITLE}.epub"] == epub_now and rows[f"{NEW_TITLE}.pdf"] == pdf_now
    epub_uri = h.records(ws)["epub"]["doc"]
    assert str(epub_uri).startswith("content://media/") and all(not c.get("replace_uri") for c in phone.media.calls)
    # the Storage switch "copy other outputs" on: books still land once, in their folder only
    app.prefs.set("mirror_outputs", True)
    compiler.plan(ws, {f"{NEW_TITLE}.epub": (NEW_TITLE, "phone folder update " * 50)})
    await h.compile(ws)
    epub_now = (ws / f"{NEW_TITLE}.epub").read_bytes()
    assert await _until(lambda: phone.media.files(folder).get(f"{NEW_TITLE}.epub") == epub_now, timeout=15), \
        "timed out: phone.media.files(folder).get(f'{NEW_TITLE}.epub') == epub_now"
    await h.idle()
    assert h.records(ws)["epub"]["doc"] == epub_uri, "the phone folder copy got a new MediaStore row"
    assert phone.media.calls[-1]["replace_uri"] == epub_uri
    assert len([r for r in phone.media.rows.values() if r["name"].endswith(".epub")]) == 1, phone.media.rows
    flat = phone.media.files("Download/Glossarion")
    assert not [n for n in flat if n.lower().endswith((".epub", ".pdf"))], f"books also copied flat: {sorted(flat)}"
    # a much shorter book: the row is replaced, not appended to
    compiler.plan(ws, {f"{NEW_TITLE}.epub": (NEW_TITLE, "short")})
    await h.compile(ws)
    epub_now = (ws / f"{NEW_TITLE}.epub").read_bytes()
    assert await _until(lambda: phone.media.files(folder).get(f"{NEW_TITLE}.epub") == epub_now, timeout=15), \
        "timed out: phone.media.files(folder).get(f'{NEW_TITLE}.epub') == epub_now"
    # a title change: the same row keeps its first name
    compiler.plan(ws, {f"Third Title.epub": ("Third Title", "third " * 20)})
    await h.compile(ws)
    epub_now = (ws / "Third Title.epub").read_bytes()
    assert await _until(lambda: phone.media.files(folder).get(f"{NEW_TITLE}.epub") == epub_now, timeout=15), \
        "timed out: phone.media.files(folder).get(f'{NEW_TITLE}.epub') == epub_now"
    assert set(phone.media.files(folder)) == {f"{NEW_TITLE}.epub", f"{NEW_TITLE}.pdf"}
    assert h.records(ws)["epub"]["doc"] == epub_uri
    # the user deleted the copy in Downloads: the next compile makes it again, once
    phone.media.user_deletes(epub_uri)
    compiler.plan(ws, {"Third Title.epub": ("Third Title", "after delete " * 20)})
    await h.compile(ws)
    epub_now = (ws / "Third Title.epub").read_bytes()
    assert await _until(lambda: phone.media.files(folder).get(f"{NEW_TITLE}.epub") == epub_now, timeout=15), \
        "timed out: phone.media.files(folder).get(f'{NEW_TITLE}.epub') == epub_now"
    new_uri = h.records(ws)["epub"]["doc"]
    assert new_uri != epub_uri and set(phone.media.files(folder)) == {f"{NEW_TITLE}.epub", f"{NEW_TITLE}.pdf"}
    compiler.plan(ws, {"Third Title.epub": ("Third Title", "and again " * 20)})
    await h.compile(ws)
    epub_now = (ws / "Third Title.epub").read_bytes()
    assert await _until(lambda: phone.media.files(folder).get(f"{NEW_TITLE}.epub") == epub_now, timeout=15), \
        "timed out: phone.media.files(folder).get(f'{NEW_TITLE}.epub') == epub_now"
    assert h.records(ws)["epub"]["doc"] == new_uri and len(phone.media.rows) == 2
    assert all(r["owner"] == PACKAGE for r in phone.media.rows.values())
    assert len(h.needs_location_posts()) == 2  # the phone folder never asks

    # ---- 9. Settings › Storage: where books go; the same "isn't listed" help -------------------------------
    _step("9. Settings › Storage: where books go; the same 'isn't listed' help")
    storage_screen = await h.go("/settings/storage")
    books_text = lambda: str(getattr(storage_screen.books_text, "value", ""))  # noqa: E731
    assert await _until(lambda: "replaced in place" in books_text(), timeout=10), books_text()
    sheet = storage_screen.show_not_listed_help()
    assert sheet is not None and sheet.title == NOT_LISTED_TITLE and sheet.body == u10.NOT_LISTED_HELP_ANDROID
    assert not [n for n in phone.media.files("Download/Glossarion") if n.lower().endswith((".epub", ".pdf"))]


def _md_values(control) -> list:
    """Markdown values under ``control`` (the ExpansionTile's help)."""
    out = []
    stack = [control]
    while stack:
        node = stack.pop()
        if type(node).__name__ == "Markdown":
            out.append(str(getattr(node, "value", "") or ""))
        for name in ("controls", "content"):
            value = getattr(node, name, None)
            if isinstance(value, list):
                stack.extend(value)
            elif value is not None and not isinstance(value, (str, int, float, bool)):
                stack.append(value)
    return out


def _chip_reasons(row) -> list:
    return [getattr(c, "reason", None) for c in (getattr(row, "controls", None) or [])]


# ==========================================================================
# Android: the pre-U10 "Mirror outputs to Downloads/Glossarion" switch becomes the phone folder
# ==========================================================================


def test_mirror_switch_becomes_the_phone_folder_destination(app_env, tmp_path, monkeypatch):
    """A user who had "Mirror outputs to Downloads/Glossarion" on before U10: after the update the phone folder is
    the destination with automatic copies on, Settings shows it in use, the Library's books are in
    Download/Glossarion/<book>/ and a recompile overwrites the same row (never a second, flat copy)."""
    from glossarion_mobile.services.files import MIRROR_PREF

    attempts, ws, compiler, snacks = _android_setup(tmp_path, monkeypatch)

    async def scenario():
        from host_tester import PyTester

        phone = Phone()
        tf, _conn, session, page, app = await _start(phone, prefs={MIRROR_PREF: True})
        h = Harness(app, page, session, PyTester(session, page), phone, snacks)
        try:
            assert await _until(lambda: app.state.engine_ready, timeout=60), "timed out: app.state.engine_ready"
            await app.library.refresh(quiet=True, reason="test")
            assert h.book_row(ws), "the workspace is not a Library book"
            dest = h.cloud.destination()
            assert dest is not None and dest.mode == "phone" and h.cloud.settings().enabled, h.cloud.settings()
            assert app.prefs.get(MIRROR_PREF, False) is True  # still copies the other outputs once
            screen = await h.go("/settings/cloud")
            assert await _until(lambda: screen.loaded, timeout=10), "timed out: screen.loaded"
            texts = h.texts()
            assert "Phone folder · Downloads/Glossarion" in texts, texts
            assert "Phone folder (Downloads/Glossarion) · in use" in texts, texts
            assert screen.auto_switch.value is True and not screen.auto_switch.disabled
            folder = f"Download/Glossarion/{BOOK}"
            await h.idle()
            # The migration queues the Library's books once the Library has published them, as turning the switch
            # on by hand does; the flat copies the old mirror made in Download/Glossarion stay where they are
            # (Glossarion never deletes a file it handed to the user).
            assert all(n.endswith(".epub") for n in phone.media.files(folder))
            compiler.plan(ws, {f"{BOOK}.epub": (BOOK, "after the update " * 30)})
            await h.compile(ws)
            v2 = (ws / f"{BOOK}.epub").read_bytes()
            assert await _until(lambda: phone.media.files(folder).get(f"{BOOK}.epub") == v2, timeout=15), \
                "timed out: phone.media.files(folder).get(f'{BOOK}.epub') == v2"
            uri = h.records(ws)["epub"]["doc"]
            compiler.plan(ws, {f"{BOOK}.epub": (BOOK, "again")})
            await h.compile(ws)
            v3 = (ws / f"{BOOK}.epub").read_bytes()
            assert await _until(lambda: phone.media.files(folder).get(f"{BOOK}.epub") == v3, timeout=15), \
                "timed out: phone.media.files(folder).get(f'{BOOK}.epub') == v3"
            assert h.records(ws)["epub"]["doc"] == uri and len(phone.media.rows) == 1, phone.media.rows
            flat = phone.media.files("Download/Glossarion")
            assert not [n for n in flat if n.lower().endswith((".epub", ".pdf"))], f"flat book copies: {sorted(flat)}"
            assert phone.pickers == [] and h.needs_location_posts() == []
            assert attempts == [], f"network access attempted: {attempts}"
            assert not h.problems, "\n\n".join(h.problems)
        except BaseException:
            print("--- phone calls ---", phone.calls[-60:])
            print("--- media ---", {k: (v["folder"], v["name"]) for k, v in phone.media.rows.items()})
            print("--- cloud events ---", list(getattr(h.cloud, "events", [])))
            raise
        finally:
            await _shutdown(tf, app)

    asyncio.run(scenario())


# ==========================================================================
# iOS: Files › On My iPhone › Glossarion is refused
# ==========================================================================


def _ios_folder(path: str, name: str, *, own: bool, provider: str = "local", label: str = "On My iPhone") -> dict:
    return {"ok": True, "error": None, "message": None, "retryable": False, "persisted": True,
            "target": {"platform": "ios", "kind": "folder", "id": "ios-" + hashlib.sha1(path.encode()).hexdigest()[:12],
                       "uri": None, "document": None, "bookmark": "Ym9va21hcms=", "root": None, "path": path,
                       "name": name, "provider": provider, "provider_label": label, "can_write": True,
                       "can_create": True, "can_delete": True, "own_folder": own, "persisted": True}}


def test_ios_refuses_the_apps_own_folder(app_env, tmp_path, monkeypatch):
    if not _has("flet_glossarion_native"):
        monkeypatch.syspath_prepend(str(EXTENSION_SRC))
    attempts = _isolate(tmp_path, monkeypatch)
    from glossarion_mobile import app as app_module
    from glossarion_mobile import runtime_bootstrap as rb
    from glossarion_mobile.ui.components import dialogs as dialogs_module

    snacks: list = []
    real_snackbar = dialogs_module.show_snackbar

    def show_snackbar(page, message, **kwargs):
        snacks.append(str(message))
        return real_snackbar(page, message, **kwargs)

    monkeypatch.setattr(app_module, "show_snackbar", show_snackbar)
    docs_root = str(rb.get_paths().docs)

    async def scenario():
        from glossarion_mobile.services.cloud_sync import OWN_FOLDER_REASON
        from host_tester import PyTester

        phone = Phone(platform="ios")
        tf, _conn, session, page, app = await _start(phone, "ios")
        h = Harness(app, page, session, PyTester(session, page), phone, snacks)
        try:
            assert await _until(lambda: app.state.engine_ready, timeout=60), "timed out: app.state.engine_ready"
            assert h.cloud is not None and h.cloud.supported and h.cloud.platform == "ios"
            screen = await h.go("/settings/cloud")
            assert await _until(lambda: screen.loaded, timeout=10), "timed out: screen.loaded"
            texts = h.texts()
            assert "Choose a folder…" in texts and "Save each file separately" in texts
            assert u10_android_only() in h.texts() or any("Android only" in t for t in texts)
            # (a) the plugin flags the app container (Files › On My iPhone › Glossarion)
            phone.ios_answers.append(_ios_folder("/private/var/mobile/Containers/Data/Application/X/Documents",
                                                 "Glossarion", own=True, label="Glossarion"))
            await h.tap(text="Choose a folder…")
            assert await _until(lambda: snacks.count(OWN_FOLDER_REASON) == 1, timeout=10), \
                "timed out: snacks.count(OWN_FOLDER_REASON) == 1"
            assert h.cloud.destination() is None
            # (b) only the path says so: inside the app's Files-visible documents (the Python safety net)
            phone.ios_answers.append(_ios_folder(os.path.join(docs_root, "Exports"), "Exports", own=False))
            await h.tap(text="Choose a folder…")
            assert await _until(lambda: snacks.count(OWN_FOLDER_REASON) == 2, timeout=10), \
                "timed out: snacks.count(OWN_FOLDER_REASON) == 2"
            assert h.cloud.destination() is None
            # an iCloud Drive folder is accepted
            phone.ios_answers.append(_ios_folder(
                "/private/var/mobile/Library/Mobile Documents/com~apple~CloudDocs/Books", "Books", own=False,
                provider="icloud", label="iCloud Drive"))
            await h.tap(text="Choose a folder…")
            assert await _until(lambda: h.cloud.destination() is not None, timeout=10), \
                "timed out: h.cloud.destination() is not None"
            dest = h.cloud.destination()
            assert dest.mode == "folder" and dest.label == "Books" and dest.provider_label == "iCloud Drive"
            assert [c for c in phone.calls if c[1] == "pick_folder"] and len(phone.ios_answers) == 0
            assert attempts == [], f"network access attempted: {attempts}"
        except BaseException:
            print("--- phone calls ---", phone.calls[-60:])
            print("--- snackbars ---", snacks[-20:])
            raise
        finally:
            await _shutdown(tf, app)

    asyncio.run(scenario())


def u10_android_only() -> str:
    from glossarion_mobile.ui.screens.cloud_sync import ANDROID_ONLY

    return ANDROID_ONLY
