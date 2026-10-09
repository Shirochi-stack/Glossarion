"""U10 acceptance, item 5: "Privacy: nothing leaves the phone except what the user asked for".

Proved end to end on the REAL app objects: the app started on the fake Flet session as Android
(``test_ui_foundations._start``, the tests_host UI harness), its real ``GlossarionNative`` Python service
(the extension) whose Dart side is an in-process fake (``FakeDart``: the extension's own
``documents_fake.FakeDocumentsNative`` / ``FakeCloudProvider`` for the document API, a scratch MediaStore
for ``save_to_downloads``, the job service and notifications recorded), the real JobService (the
compile run replaced by writing the outputs, ``test_jobs.FakeBackend`` as the engine), LibraryService,
CloudSyncService, ShareLinkService, Settings › Cloud sync & sharing (its consent sheets tapped) and the
Book page actions (``U10Actions``); the share-link services are the ``test_share_links`` fakes on
127.0.0.1 (Gofile, pixeldrain, Send over HTTP + WebSocket). The user is a returning one (the first
Run's notification / battery questions answered, a config.json with the app's defaults), and the
harness sends the client's ``dismiss`` for every sheet the user leaves (the fake session never does).

``NetSpy`` sees every network attempt: a ``sys.addaudithook`` on socket.connect / sendto / getaddrinfo /
gethostbyname (C level, so a library holding its own reference is caught too), the Windows proactor's
ConnectEx and every ``requests`` / ``httpx`` send. Each attempt is recorded with the scenario phase, the
thread and whether U10 code is on the stack; anything off the machine is refused.

1. ``test_cloud_sync_book_lifecycle_sends_nothing_over_the_network``: sync off by default (a finished
   book is not copied); linking a folder is not opting in; sync on with HTML off copies only the
   book's EPUB / PDF / TXT (never chapters, glossary, metadata, progress or the raw source); a
   recompile overwrites in place; a book set to Never stays home until "Send now"; deleting a book
   forgets its records without touching the cloud; the phone folder overwrites its Downloads entries;
   "Wipe app data" gives every grant back. Zero network attempts in every cloud phase, no URL opened.
2. ``test_share_links_reach_only_the_enabled_consented_provider``: every service off by default;
   switched on without consent, or consent declined / revoked: refused with zero requests; after the
   consent sheet only the chosen service is contacted (Gofile, pixeldrain with the user's key, Send
   end to end encrypted: the service never sees the text, the file name or the key); transfer.it is a
   browser hand-off with zero requests; key exports, config and data-folder files are never eligible;
   deleting the book and wiping send nothing. No secret (Gofile guest token, pixeldrain key, Send owner
   token and key, link URLs, delete handles) in plain text in any file (config.json,
   mobile_state.json, the sidecars, logs), in the log records, on stdout / stderr, in a message to the
   Flutter client (clipboard, secure storage, browser) or to the native side (notifications).
   Both scenarios: no new config.json key; desktop ``src/*.py`` unchanged while the app ran.
3. ``test_cloud_and_share_sources_have_no_network_path``: the cloud-sync sources (Python, Kotlin,
   Swift) import no network API; the share code reaches the network only through the providers'
   stdlib client (never ``requests`` / ``httpx``, which the HTTP log patches) and only their hosts.
4. ``test_desktop_sources_are_byte_identical_to_head``: no top-level ``src/*.py`` differs from HEAD.

Real data stays untouched: FLET_APP_STORAGE_* (HOME, OUTPUT_DIRECTORY, GLOSSARION_LIBRARY_DIR,
GLOSSARION_DATA_DIR, CONFIG_FILE through the bootstrap), USERPROFILE / APPDATA / LOCALAPPDATA point
into tmp_path; GLOSSARION_HTTP_LOG=0.

Run from src/mobile with the mobile venv::

    .venv/Scripts/python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic \
        tests_host/test_u10_accept5.py
"""

from __future__ import annotations

import ast
import asyncio
import dataclasses
import errno
import hashlib
import importlib.util
import json
import logging
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = MOBILE_DIR / "app"
SRC_DIR = MOBILE_DIR.parent
REPO_DIR = SRC_DIR.parent
PACKAGE = APP_DIR / "glossarion_mobile"
EXTENSION_SRC = MOBILE_DIR / "extensions" / "flet_glossarion_native" / "src"
EXTENSION_PKG = EXTENSION_SRC / "flet_glossarion_native"
FLUTTER_PKG = EXTENSION_SRC / "flutter" / "flet_glossarion_native"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from glossarion_mobile import job_kinds  # noqa: E402
from glossarion_mobile import runtime_bootstrap as rb  # noqa: E402
from glossarion_mobile.services import jobs as jobs_service  # noqa: E402
from glossarion_mobile.services.jobs import JobState  # noqa: E402


def _has(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _load(name: str, alias: str):
    module = sys.modules.get(alias)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(alias, Path(__file__).with_name(name))
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(alias, None)
        raise
    return module


_TB = _load("test_bootstrap.py", "_glossarion_tb_helpers_u10accept5")
storage = _TB.storage
app_env = _TB.app_env

_APP_NEEDS = ("flet", "msgpack", "requests", "bs4")
needs_app = pytest.mark.skipif(not all(_has(m) for m in _APP_NEEDS),
                               reason=f"needs {', '.join(_APP_NEEDS)} (the mobile venv)")
needs_crypto = pytest.mark.skipif(not (_has("cryptography") and _has("websockets")),
                                  reason="cryptography / websockets missing")


def _tf():
    return _load("test_ui_foundations.py", "_glossarion_tf_helpers_u10accept5")


def _norm(path: Any) -> str:
    return os.path.normcase(os.path.normpath(os.path.abspath(os.fspath(path))))


# ===================================================================================================
# the network spy
# ===================================================================================================

_SPIES: list = []
_HOOKED: list = []

#: U10 code: a network attempt with one of these on the stack is the cloud sync's / share links' own.
_U10_NAMES = {"cloud_sync.py", "cloud_records.py", "share_links.py", "documents.py", "documents_fake.py"}


def _is_u10(filename: str) -> bool:
    text = filename.replace("\\", "/")
    return os.path.basename(text) in _U10_NAMES or "/share_providers/" in text


def _audit(event: str, args: tuple) -> None:
    if not _SPIES or not event.startswith("socket."):
        return
    spy = _SPIES[-1]
    if event == "socket.connect":
        spy.on_connect(args[1], "socket.connect")
    elif event == "socket.sendto":
        spy.on_connect(args[1], "socket.sendto")
    elif event == "socket.getaddrinfo":
        spy.on_lookup(args[0], args[1], "socket.getaddrinfo")
    elif event in ("socket.gethostbyname", "socket.gethostbyname_ex", "socket.gethostbyaddr"):
        spy.on_lookup(args[0], None, event)


class NetSpy:
    """Every network attempt while active (``with spy:``): ``events`` (dicts with ``phase``, ``api``,
    ``host``, ``port``, ``label`` (a registered fake, ``internal`` for asyncio's socketpair, ``loopback``,
    ``remote``), ``u10``, ``thread``, ``where``). Off-machine connects and lookups are refused."""

    def __init__(self) -> None:
        self.events: list = []
        self.phase = "setup"
        self.ports: dict = {}
        self.lock = threading.Lock()

    def __enter__(self) -> "NetSpy":
        if not _HOOKED:
            sys.addaudithook(_audit)  # permanent for the process; inert while no spy is active
            _HOOKED.append(True)
        _SPIES.append(self)
        return self

    def __exit__(self, *exc: Any) -> None:
        if self in _SPIES:
            _SPIES.remove(self)

    @staticmethod
    def _local(host: Any) -> bool:
        if isinstance(host, bytes):
            host = host.decode("ascii", "replace")
        text = str(host or "").strip("[]").lower()
        return text in ("", "localhost", "::1", "0.0.0.0", "::") or text.startswith("127.")

    def _record(self, api: str, host: Any, port: Any) -> dict:
        frames = []
        frame = sys._getframe(2)
        while frame is not None and len(frames) < 120:
            frames.append((frame.f_code.co_filename, frame.f_code.co_name))
            frame = frame.f_back
        local = self._local(host)
        try:
            port = int(port) if port not in (None, "", b"") else None
        except (TypeError, ValueError):
            port = None
        label = self.ports.get(port) if local else None
        if label is None and local and any("socketpair" in name for _fn, name in frames):
            label = "internal"
        event = {"api": api, "host": str(host), "port": port, "phase": self.phase, "local": local,
                 "label": label or ("loopback" if local else "remote"),
                 "u10": any(_is_u10(fn) for fn, _name in frames),
                 "thread": threading.current_thread().name,
                 "where": [f"{os.path.basename(fn)}:{name}" for fn, name in frames[:14]]}
        with self.lock:
            self.events.append(event)
        return event

    def on_connect(self, address: Any, api: str) -> None:
        if not isinstance(address, tuple) or len(address) < 2:
            return  # AF_UNIX
        event = self._record(api, address[0], address[1])
        if not event["local"]:
            raise OSError(errno.ENETUNREACH, f"u10 privacy test: a connection to {address[0]} is blocked")

    def on_lookup(self, host: Any, port: Any, api: str) -> None:
        if host is None:
            return  # getaddrinfo(None, port): a local bind
        event = self._record(api, host, port)
        if not event["local"]:
            raise socket.gaierror(socket.EAI_NONAME, f"u10 privacy test: looking up {host} is blocked")

    def on_library(self, api: str, url: Any) -> None:
        self._record(api, f"<{api}> {url}", None)["label"] = "library"

    def in_phases(self, prefix: str) -> list:
        return [e for e in self.events if str(e["phase"]).startswith(prefix)]

    def phase_labels(self, phase: str) -> set:
        return {e["label"] for e in self.events if e["phase"] == phase and e["label"] != "internal"}

    @staticmethod
    def show(events: list) -> str:
        return "\n".join(f"  [{e['phase']}] {e['api']} {e['host']}:{e['port']} label={e['label']} u10={e['u10']} "
                         f"thread={e['thread']} via {' < '.join(e['where'][:8])}" for e in events) or "  (none)"

    def install_patches(self, monkeypatch) -> None:
        """The paths the audit hook cannot see (Windows ConnectEx) and the HTTP libraries."""
        spy = self
        try:
            from asyncio import windows_events
        except ImportError:  # not Windows
            windows_events = None
        if windows_events is not None:
            original = windows_events.IocpProactor.connect

            def proactor_connect(proactor, conn, address):
                spy.on_connect(address, "proactor.connect")
                return original(proactor, conn, address)

            monkeypatch.setattr(windows_events.IocpProactor, "connect", proactor_connect)
        if _has("requests"):
            import requests.sessions

            send = requests.sessions.Session.send

            def requests_send(session, request, **kwargs):
                spy.on_library("requests", getattr(request, "url", ""))
                return send(session, request, **kwargs)

            monkeypatch.setattr(requests.sessions.Session, "send", requests_send)
        if _has("httpx"):
            import httpx

            client_send, async_send = httpx.Client.send, httpx.AsyncClient.send

            def httpx_send(client, request, *args, **kwargs):
                spy.on_library("httpx", getattr(request, "url", ""))
                return client_send(client, request, *args, **kwargs)

            async def httpx_async_send(client, request, *args, **kwargs):
                spy.on_library("httpx", getattr(request, "url", ""))
                return await async_send(client, request, *args, **kwargs)

            monkeypatch.setattr(httpx.Client, "send", httpx_send)
            monkeypatch.setattr(httpx.AsyncClient, "send", httpx_async_send)


# ===================================================================================================
# the Dart / Kotlin side of flet_glossarion_native, faked in process
# ===================================================================================================


class FakeDart:
    """Answers ``GlossarionNative._invoke_method``: the document API through the extension's
    ``FakeDocumentsNative`` over a ``FakeCloudProvider`` ("Drive"), ``save_to_downloads`` into a scratch
    MediaStore (``Download/<subdir>``, ``replace_uri`` overwrites in place), the job service and the
    notifications recorded. ``calls``: ``(phase, method, arguments)``."""

    PLATFORM = {"platform": "android", "sdk_int": 34, "save_to_downloads": True, "documents": True,
                "notifications_enabled": True, "ignoring_battery_optimizations": True,
                "fgs_types": ["dataSync"], "persisted_grant_limit": 512, "continued_processing": False}

    def __init__(self, root: Path, docs_module: Any, phase: Any) -> None:
        self.provider = docs_module.FakeCloudProvider(authority="com.google.android.apps.docs.storage",
                                                      label="Drive")
        self.docs = docs_module.FakeDocumentsNative(self.provider)
        self.default_chain = list(docs_module.DEFAULT_MODE_CHAIN)
        self.phase = phase
        self.calls: list = []
        self.downloads = root / "Download"
        self.media: dict = {}
        self.service_running = False

    def methods(self, prefix: str = "") -> list:
        return [m for p, m, _a in self.calls if str(p).startswith(prefix)]

    def _save(self, args: dict) -> Optional[str]:
        source = str(args.get("path") or "")
        if not os.path.isfile(source):
            return None
        replace_uri = args.get("replace_uri")
        if replace_uri and replace_uri in self.media and os.path.isfile(self.media[replace_uri]):
            shutil.copyfile(source, self.media[replace_uri])
            return replace_uri
        folder = self.downloads.joinpath(*[p for p in str(args.get("subdir") or "Glossarion").split("/") if p])
        folder.mkdir(parents=True, exist_ok=True)
        name = str(args.get("display_name") or os.path.basename(source))
        stem, ext = os.path.splitext(name)
        target, n = folder / name, 0
        while target.exists():
            n += 1
            target = folder / f"{stem} ({n}){ext}"
        shutil.copyfile(source, target)
        uri = f"content://media/external/downloads/{len(self.media) + 1}"
        self.media[uri] = str(target)
        return uri

    async def invoke(self, method: str, arguments: Any) -> Any:
        args = dict(arguments or {})
        self.calls.append((self.phase(), method, args))
        d = self.docs
        chain = args.get("mode_chain") or self.default_chain
        if method == "get_platform_info":
            return dict(self.PLATFORM)
        if method == "get_initial_shared":
            return []
        if method in ("init_notifications", "show_notification"):
            return True
        if method == "start_job_service":
            self.service_running = True
            return True
        if method == "stop_job_service":
            self.service_running = False
            return None
        if method == "is_job_service_running":
            return self.service_running
        if method == "begin_background_task":
            return -1
        if method == "save_to_downloads":
            return self._save(args)
        if method == "pick_folder":
            return await d.pick_folder(initial=args.get("initial"), op_id=args.get("op_id"))
        if method == "pick_save_location":
            return await d.pick_save_location(args.get("name") or "file", args.get("mime_type") or "",
                                              args.get("source_path"), initial=args.get("initial"),
                                              mode_chain=chain, op_id=args.get("op_id"))
        if method == "pick_document":
            return await d.pick_document(args.get("mime_types"), initial=args.get("initial"), op_id=args.get("op_id"))
        if method == "list_children":
            return await d.list_children(args.get("folder"), names=args.get("names"))
        if method == "create_file":
            return await d.create_file(args.get("folder"), args.get("name"), args.get("mime_type") or "",
                                       on_exists=args.get("on_exists") or "rename")
        if method == "create_folder":
            return await d.create_folder(args.get("folder"), args.get("name"), on_exists=args.get("on_exists") or "adopt")
        if method == "write_file":
            return await d.write_file(args.get("ref"), args.get("source_path"), name=args.get("name"),
                                      mime_type=args.get("mime_type"), on_exists=args.get("on_exists") or "rename",
                                      mode_chain=chain, verify=bool(args.get("verify", True)), op_id=args.get("op_id"))
        if method == "rename_document":
            return await d.rename_document(args.get("document"), args.get("name"))
        if method == "stat":
            return await d.stat(args.get("document"))
        if method == "delete":
            return await d.delete(args.get("document"))
        if method == "query_root":
            return await d.query_root(args.get("target"))
        if method == "release":
            return await d.release(args.get("target"))
        if method == "list_grants":
            return await d.list_grants()
        if method == "cancel_document_op":
            return await d.cancel_document_op(args.get("op_id"))
        if method == "take_document_results":
            return await d.take_document_results()
        return None


# ===================================================================================================
# the app harness
# ===================================================================================================

PRIVATE = b"PRIVATE-CHAPTER-TEXT-u10accept5"  # in every workspace file that must never leave the phone


def _book_outputs(title: str, version: int) -> dict:
    """What a compile writes: the EPUB, its PDF (no suffix: ``pdf_after_epub``), TXT and HTML outputs."""
    def body(kind: str, size: int) -> bytes:
        head = f"{kind}-v{version}|{title}|".encode()
        return head + bytes((i * 7 + version) % 251 for i in range(size + version * 512))

    return {f"{title}.epub": b"PK" + body("EPUB", 40_000), f"{title}.pdf": b"%PDF" + body("PDF", 30_000),
            f"{title}_translated.txt": body("TXT", 9_000), f"{title}_translated.html": body("HTML", 12_000)}


class Harness:
    """The real app on a fake Android session with the fake Dart side, the fake job engine and the spy."""

    def __init__(self, monkeypatch, tmp_path: Path) -> None:
        self.mp = monkeypatch
        self.tmp = tmp_path
        for var, sub in (("USERPROFILE", "userprofile"), ("APPDATA", "appdata"), ("LOCALAPPDATA", "localappdata")):
            (tmp_path / sub).mkdir(exist_ok=True)
            monkeypatch.setenv(var, str(tmp_path / sub))
        monkeypatch.setenv("GLOSSARION_HTTP_LOG", "0")
        for var in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
            monkeypatch.delenv(var, raising=False)
        self.spy = NetSpy()
        self.spy.install_patches(monkeypatch)
        # the extension's Python service, with the Dart side answered in process
        self._ext_before = {m for m in sys.modules if m.split(".")[0] == "flet_glossarion_native"}
        monkeypatch.syspath_prepend(str(EXTENSION_SRC))
        from flet_glossarion_native import documents_fake
        from flet_glossarion_native.native import GlossarionNative

        self.dart = FakeDart(tmp_path / "phone", documents_fake, lambda: self.spy.phase)
        dart = self.dart

        async def invoke_method(native, method_name, arguments=None, timeout=None):
            return await dart.invoke(method_name, arguments)

        monkeypatch.setattr(GlossarionNative, "_invoke_method", invoke_method)
        # jobs: the real JobService; the engine is test_jobs.FakeBackend and a compile writes the outputs
        self.compiles: dict = {}
        info = job_kinds.get_kind("compile_epub")
        monkeypatch.setitem(job_kinds._CACHE, "compile_epub", dataclasses.replace(info, run=self._compile))
        tj = _load("test_jobs.py", "_glossarion_tj_helpers_u10accept5")
        monkeypatch.setattr(jobs_service, "JobBackend", lambda: tj.FakeBackend(str(tmp_path / "job-out")))
        self.records = _Records()
        self.app: Any = None
        self.conn: Any = None
        self.session: Any = None
        self.src_hashes = _src_hashes()

    # ---- jobs ----

    def _compile(self, ctx) -> dict:
        folder = str(ctx.params.get("folder") or "")
        written = []
        for name, data in (self.compiles.get(_norm(folder)) or {}).items():
            path = os.path.join(folder, name)
            with open(path, "wb") as handle:
                handle.write(data)
            written.append(path)
        ctx.log(f"compiled {len(written)} file(s)")
        return {"ok": True, "outputs": [p for p in written if p.lower().endswith((".epub", ".pdf"))]}

    async def compile(self, folder: str, files: dict) -> Any:
        """A Library "Compile EPUB" through the real LibraryService + JobService."""
        app = self.app
        self.compiles[_norm(folder)] = files
        row = {"name": os.path.basename(folder), "output_folder": folder, "path": folder, "type": "in_progress"}
        job_id = await app.library.submit(app.library.compile_spec(row, "compile_epub"))
        assert job_id, "the compile job was not queued"

        def finished() -> bool:
            snap = app.job_service.snapshot(job_id)
            return snap is not None and snap.is_terminal

        assert await _tf()._wait(finished, timeout=30), "the compile job did not finish"
        snap = app.job_service.snapshot(job_id)
        assert snap.state is JobState.DONE, (snap.state, snap.error)
        return snap

    # ---- app ----

    async def start(self) -> Any:
        tf = _tf()
        self.spy.phase = "startup"
        # a returning user who already answered the first Run's notification / battery questions
        from glossarion_mobile.services import background

        state_file = Path(rb.get_paths().data) / "mobile_state.json"
        try:
            state = json.loads(state_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            state = {}
        state.update({background.PREF_NOTIFICATION_ASKED: True, background.PREF_BATTERY_PROMPT: True,
                      background.PREF_NOTIFICATIONS_OFF_HINT: True})
        state_file.write_text(json.dumps(state), encoding="utf-8")
        config_file = Path(rb.get_paths().config_file)
        if not config_file.exists():  # a returning user's config (the defaults the app shows)
            config_file.write_text(json.dumps({"model": "authgpt/gpt-6-luna", "auto_update_check": True}),
                                   encoding="utf-8")
        _m, self.conn, self.session, _page, app = await tf._start("android")
        self.app = app
        assert await tf._wait(lambda: app.state.engine_ready, timeout=60), "the engine did not get ready"
        assert await tf._wait(lambda: app.cloud_sync is not None and app.share_links is not None, timeout=30)
        assert type(app.native.native).__name__ == "GlossarionNative" and not app.native.is_stub
        await app.cloud_sync.wait_idle()
        await asyncio.sleep(0.5)  # startup work (catalog, chats, Library scan) settles
        self.records.attach()
        return app

    async def stop(self) -> None:
        self.records.detach()
        app = self.app
        if app is None:
            return
        try:
            if getattr(app, "prefs", None) is not None:
                app.prefs.flush()
            cloud = getattr(app, "cloud_sync", None)
            if cloud is not None:
                await cloud.wait_idle()
                cloud.store.flush()
        finally:
            try:
                await _tf()._stop(app)
            finally:
                cloud = getattr(app, "cloud_sync", None)
                if cloud is not None:
                    cloud.close()

    def cleanup_modules(self) -> None:
        for name in [m for m in sys.modules if m.split(".")[0] == "flet_glossarion_native"]:
            if name not in self._ext_before:
                sys.modules.pop(name, None)

    async def settle_cloud(self, timeout: float = 20.0) -> None:
        cloud = self.app.cloud_sync
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            await asyncio.sleep(0.05)
            await cloud.wait_idle()
            await asyncio.sleep(0.05)
            task = cloud._drain_task
            if (task is None or task.done()) and cloud.store.due(cloud.clock()) is None:
                return
        raise AssertionError(f"the cloud queue did not drain: {cloud.store.queue()} events={cloud.events[-10:]}")

    async def dismiss_sheets(self) -> None:
        """What the Flutter client does once a sheet / dialog has gone: the user leaves every open one
        (Done / swipe down) and the client reports ``dismiss`` after the close animation, so the page drops
        it (the fake session never sends it on its own)."""
        from glossarion_mobile.ui.components.dialogs import close_dialog

        page = self.app.page
        for dialog in list(page._dialogs.controls):
            if getattr(dialog, "open", False):
                close_dialog(page, dialog)
            await self.session.dispatch_event(dialog._i, "dismiss", None)
        await asyncio.sleep(0.05)

    def launched_urls(self) -> list:
        from flet.messaging.protocol import MessageAction

        out = []
        for message in list(self.conn.messages):
            if message.action == MessageAction.INVOKE_METHOD and message.body.name == "launch_url":
                out.append(str((message.body.args or {}).get("url")))
        return out

    def client_invocations(self) -> list:
        from flet.messaging.protocol import MessageAction

        return [(m.body.name, repr(m.body.args)) for m in list(self.conn.messages)
                if m.action == MessageAction.INVOKE_METHOD]

    def config_keys(self) -> Optional[set]:
        paths = rb.get_paths()
        try:
            with open(paths.config_file, encoding="utf-8") as handle:
                return set(json.load(handle))
        except (OSError, ValueError):
            return None


class _Records(logging.Handler):
    """Every log record while attached: a handler on the root logger (at the app's own levels) and the U10
    loggers at DEBUG, so their most talkative records are checked too. (Raising the root logger would add
    Flet's debug dump of every UI patch, which carries what the user typed into a field.)"""

    DEBUG_NAMES = ("glossarion.cloud", "glossarion.share", "glossarion.share.transferit", "flet_glossarion_native")

    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.lines: list = []
        self._levels: dict = {}
        self._attached: list = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            text = record.getMessage()
        except Exception:
            text = f"{record.msg} {record.args}"
        if record.exc_info and record.exc_info[1] is not None:
            text += f" | {record.exc_info[1]!r}"
        self.lines.append(f"{record.name}: {text}")

    def attach(self) -> None:
        loggers = [logging.getLogger()]
        for name in self.DEBUG_NAMES:
            logger = logging.getLogger(name)
            self._levels[name] = logger.level
            logger.setLevel(logging.DEBUG)
            if not logger.propagate:
                loggers.append(logger)
        for logger in loggers:
            logger.addHandler(self)
            self._attached.append(logger)

    def detach(self) -> None:
        for name, level in self._levels.items():
            logging.getLogger(name).setLevel(level)
        for logger in self._attached:
            if self in logger.handlers:
                logger.removeHandler(self)
        self._levels, self._attached = {}, []


def _run(h: "Harness", scenario: Any, timeout: float = 300.0) -> None:
    """``asyncio.run(scenario())`` under the spy, with a watchdog: a step that hangs fails the test with the
    phase and the stack of every pending task instead of blocking the run."""
    import io
    import traceback

    async def guarded() -> None:
        task = asyncio.ensure_future(scenario())
        done, _pending = await asyncio.wait({task}, timeout=timeout)
        if task in done:
            task.result()
            return
        dump = io.StringIO()
        for other in asyncio.all_tasks():
            if other is asyncio.current_task():
                continue
            dump.write(f"--- {other.get_name()}\n")
            other.print_stack(limit=12, file=dump)
        task.cancel()
        try:
            await asyncio.wait_for(task, 30)
        except BaseException:
            pass
        raise AssertionError(f"the scenario hung in phase {h.spy.phase!r}:\n{dump.getvalue()[-12000:]}"
                             f"\nthreads:\n" + "\n".join(
                                 f"{t.name}: {''.join(traceback.format_stack(sys._current_frames()[t.ident], 6))}"
                                 for t in threading.enumerate() if t.ident in sys._current_frames()))

    try:
        with h.spy:
            asyncio.run(guarded())
    finally:
        h.cleanup_modules()


async def _fresh(screen: Any) -> None:
    """What the user sees after the screen caught up: no refresh running, then one more read."""
    await _tf()._wait(lambda: not screen._refreshing, timeout=10)
    await screen.refresh()


def _src_hashes() -> dict:
    out = {}
    for path in sorted(SRC_DIR.glob("*.py")):
        out[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def _workspace(root: Path, title: str) -> str:
    """A translated book's workspace: chapters, glossary, metadata and progress (none of them may leave the
    phone; only the compiled outputs a later compile writes may)."""
    folder = root / title
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "response_001_Chapter 1.html").write_bytes(b"<html><body><p>" + PRIVATE + b" one</p></body></html>")
    (folder / "response_002_Chapter 2.html").write_bytes(b"<html><body><p>" + PRIVATE + b" two</p></body></html>")
    (folder / "glossary.json").write_bytes(b'{"entries": ["' + PRIVATE + b'"]}')
    (folder / "metadata.json").write_text(json.dumps({"title": title, "note": PRIVATE.decode()}), encoding="utf-8")
    (folder / "translation_progress.json").write_text(json.dumps({"chapters": {
        "1": {"status": "completed", "output_file": "response_001_Chapter 1.html"},
        "2": {"status": "completed", "output_file": "response_002_Chapter 2.html"}}}), encoding="utf-8")
    return str(folder)


def _private_names(folder: str) -> set:
    return {name for name in os.listdir(folder)} if os.path.isdir(folder) else set()


# ===================================================================================================
# 1. cloud sync: a whole book lifecycle without a single network attempt
# ===================================================================================================


@needs_app
def test_cloud_sync_book_lifecycle_sends_nothing_over_the_network(app_env, monkeypatch, tmp_path):
    from glossarion_mobile.services.library import book_identity
    from glossarion_mobile.state.cloud_records import book_key
    from glossarion_mobile.ui.screens.cloud_sync import CloudFacade, U10Actions

    h = Harness(monkeypatch, tmp_path)
    facts: dict = {}

    def cloud_files() -> dict:
        provider = h.dart.provider
        out = {}
        for node in provider.all_files():
            parts, parent = [node.name], node.parent
            while parent is not None and parent is not provider.root:
                parts.insert(0, parent.name)
                parent = parent.parent
            out["/".join(parts)] = bytes(node.data)
        return out

    def phone_files() -> dict:
        root = h.dart.downloads
        return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()} \
            if root.is_dir() else {}

    async def scenario():
        app = await h.start()
        try:
            cloud, library, provider = app.cloud_sync, app.library, h.dart.provider
            paths = rb.get_paths()
            output = Path(paths.output)
            facts["config_before"] = h.config_keys()
            launched_before = len(h.launched_urls())

            # 0. defaults: off, no destination, every share service off
            h.spy.phase = "cloud:defaults"
            assert cloud.docs.available and cloud.platform == "android"
            state = await asyncio.to_thread(cloud.ui_state)
            assert state["enabled"] is False and state["destination"] is None, state
            assert not any(s.enabled for s in app.share_links.provider_states())

            # 1. a finished book with sync off and no destination: stays on the phone
            h.spy.phase = "cloud:off"
            raw_dir = Path(paths.library) / "Raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            (raw_dir / "Moonlit Archive.epub").write_bytes(b"PK" + PRIVATE + b" untranslated source")
            # a book imported into the Library and never translated: it is not a finished book
            (raw_dir / "Unread Novel.epub").write_bytes(b"PK" + PRIVATE + b" untranslated source, never opened")
            book_a = _workspace(output, "Moonlit Archive")
            await h.compile(book_a, _book_outputs("Moonlit Archive", 1))
            await h.settle_cloud()
            assert cloud_files() == {} and not h.dart.methods("cloud:off").count("write_file")
            assert "save_to_downloads" not in h.dart.methods("cloud:off")  # the old mirror switch is off

            # 2. Settings › Choose a folder…: linking is not opting in
            h.spy.phase = "cloud:link"
            folder_node = provider.add_folder("Glossarion Books")
            provider.next_pick("folder", folder_node)
            await app.navigate("/settings/cloud")
            screen = app.shell.top_screen
            assert type(screen).__name__ == "CloudSyncScreen"
            await _fresh(screen)
            linked = await screen.choose("folder", confirm=False)
            assert linked.get("ok"), linked
            await h.settle_cloud()
            assert cloud_files() == {}, "linking a folder copied books although sync is off"
            tree_uri = next(uri for uri in provider.grants if "/tree/" in uri)

            # 3. sync on, HTML off: only the opted-in formats of the finished book, nothing private
            h.spy.phase = "cloud:enable"
            await library.refresh(quiet=True)  # the Library as the user sees it: every shelf scanned
            assert (await screen.set_kind("html", False)).get("ok")
            await _fresh(screen)
            assert (await screen.set_auto(True)).get("ok")
            await h.settle_cloud()
            outputs_a1 = _book_outputs("Moonlit Archive", 1)
            files = cloud_files()
            wanted = {outputs_a1[n] for n in ("Moonlit Archive.epub", "Moonlit Archive.pdf",
                                              "Moonlit Archive_translated.txt")}
            assert set(files.values()) == wanted, sorted(files)
            assert len(files) == 3, sorted(files)
            assert all(PRIVATE not in data for data in files.values())
            private = _private_names(book_a) - set(outputs_a1)
            assert not {name.rsplit("/", 1)[-1] for name in files} & private, sorted(files)
            assert not any(data.endswith(b"untranslated source") for data in files.values())
            facts["cloud_after_enable"] = sorted(files)
            facts["library_rows"] = sorted((str(r.get("type") or ""), os.path.basename(str(r.get("path") or "")))
                                           for r in library.snapshot.all_books())

            # 4. a recompile overwrites the same cloud documents
            h.spy.phase = "cloud:recompile"
            ids_before = sorted(n.doc_id for n in provider.all_files())
            await h.compile(book_a, _book_outputs("Moonlit Archive", 2))
            await h.settle_cloud()
            outputs_a2 = _book_outputs("Moonlit Archive", 2)
            assert sorted(n.doc_id for n in provider.all_files()) == ids_before
            assert set(cloud_files().values()) == {outputs_a2[n] for n in (
                "Moonlit Archive.epub", "Moonlit Archive.pdf", "Moonlit Archive_translated.txt")}

            # 5. a book set to Never stays home after its compile...
            h.spy.phase = "cloud:never"
            book_b = _workspace(output, "Second Light")
            assert (await CloudFacade(cloud).call("set_override", book_b, "never")).get("ok")
            await h.compile(book_b, _book_outputs("Second Light", 1))
            await h.settle_cloud()
            outputs_b1 = _book_outputs("Second Light", 1)
            assert not set(outputs_b1.values()) & set(cloud_files().values()), "a Never book was copied"

            # 6. ...until the user taps Send now on its Book page
            h.spy.phase = "cloud:send-now"
            actions = U10Actions(cloud=cloud, shares=app.share_links, page=app.page, say=lambda *a, **k: None,
                                 platform="android")
            sent = await actions.send_now(book_b, await asyncio.to_thread(CloudFacade(cloud).snapshot))
            assert sent.get("ok"), sent
            await h.settle_cloud()
            assert {outputs_b1[n] for n in ("Second Light.epub", "Second Light.pdf",
                                            "Second Light_translated.txt")} <= set(cloud_files().values())
            assert outputs_b1["Second Light_translated.html"] not in cloud_files().values()

            # 7. deleting book A in the Library forgets its records; the cloud is not touched
            h.spy.phase = "cloud:delete-book"
            before_delete = cloud_files()
            await library.refresh(quiet=True)
            row = next(r for r in library.snapshot.all_books() if _norm(book_identity(r)) == _norm(book_a))
            plan = await asyncio.to_thread(library.plan_delete_blocking, [row])
            await asyncio.to_thread(library.execute_delete_blocking, plan)
            await h.settle_cloud()
            assert not os.path.exists(book_a)
            dest_id = cloud.destination().id
            assert cloud.store.book(dest_id, book_key(book_a)) is None and cloud.store.queued(book_a) is None
            assert cloud_files() == before_delete and "delete" not in h.dart.methods("cloud:delete-book")

            # 8. the phone folder: Downloads/Glossarion entries overwritten in place; the folder grant goes back
            h.spy.phase = "cloud:phone-folder"
            switched = await screen.choose("phone", confirm=False)
            assert switched.get("ok"), switched
            assert tree_uri not in provider.grants, "the old folder's grant was kept"
            assert (await actions.send_now(book_b, await asyncio.to_thread(CloudFacade(cloud).snapshot))).get("ok")
            await h.settle_cloud()
            first = phone_files()
            assert set(first.values()) == {outputs_b1[n] for n in ("Second Light.epub", "Second Light.pdf",
                                                                    "Second Light_translated.txt")}, sorted(first)
            await h.compile(book_b, _book_outputs("Second Light", 2))
            assert (await actions.send_now(book_b, await asyncio.to_thread(CloudFacade(cloud).snapshot))).get("ok")
            await h.settle_cloud()
            second = phone_files()
            outputs_b2 = _book_outputs("Second Light", 2)
            assert sorted(second) == sorted(first), "the phone folder added copies instead of replacing"
            assert set(second.values()) == {outputs_b2[n] for n in ("Second Light.epub", "Second Light.pdf",
                                                                     "Second Light_translated.txt")}
            facts["phone_files"] = sorted(second)

            # 9. Wipe app data: every persisted grant goes back to the system first
            h.spy.phase = "cloud:wipe"
            provider.next_pick("folder", folder_node)
            assert (await screen.choose("folder", confirm=False)).get("ok")
            await h.settle_cloud()
            assert provider.grants, "the folder was not linked again"
            await cloud.wipe()
            assert provider.grants == {}, provider.grants
            assert cloud.destination() is None and not cloud.settings().enabled
            h.spy.phase = "cloud:done"
            facts["launched"] = h.launched_urls()[launched_before:]
            facts["config_after"] = h.config_keys()
            facts["dart"] = [(p, m) for p, m, _a in h.dart.calls if str(p).startswith("cloud:")]
        finally:
            await h.stop()

    _run(h, scenario)

    cloud_events = h.spy.in_phases("cloud:")
    assert not [e for e in cloud_events if e["label"] != "internal"], \
        "network attempts while the cloud sync ran:\n" + NetSpy.show(cloud_events)
    assert not [e for e in h.spy.events if e["u10"] and e["label"] != "internal"], \
        "network attempts from U10 code:\n" + NetSpy.show([e for e in h.spy.events if e["u10"]])
    assert facts["launched"] == [], f"the cloud sync opened {facts['launched']}"
    allowed = {"get_platform_info", "init_notifications", "is_job_service_running", "start_job_service", "update_job_service",
               "stop_job_service", "show_notification", "cancel_notification", "begin_background_task",
               "end_background_task", "save_to_downloads", "pick_folder", "list_children", "create_file",
               "create_folder", "write_file", "stat", "query_root", "release", "list_grants", "take_document_results",
               "rename_document", "delete", "cancel_document_op"}
    assert {m for _p, m in facts["dart"]} <= allowed, {m for _p, m in facts["dart"]} - allowed
    _assert_no_new_config_keys(facts["config_before"], facts["config_after"])
    assert _src_hashes() == h.src_hashes, "a desktop src/*.py changed while the app ran"
    print(f"U10 accept5 cloud: {len(h.spy.events)} network attempt(s) in all, "
          f"{len([e for e in h.spy.events if e['phase'] == 'startup'])} at startup "
          f"({sorted({e['label'] for e in h.spy.events})}); cloud files {facts['cloud_after_enable']}; "
          f"phone folder {facts['phone_files']}; native calls {sorted({m for _p, m in facts['dart']})}; "
          f"Library rows {facts['library_rows']}; events {NetSpy.show(h.spy.events)}")


def _assert_no_new_config_keys(before: Optional[set], after: Optional[set]) -> None:
    assert after is not None, "config.json is missing after the scenario"
    if before is not None:
        assert after <= before, f"new config.json keys: {sorted(after - before)}"
    words = ("cloud", "share_link", "gofile", "pixeldrain", "transfer", "send_e2ee", "mobile_")
    flagged = sorted(k for k in after if any(w in str(k).lower() for w in words) and (before is None or k not in before))
    assert not flagged, f"U10 keys in config.json: {flagged}"


# ===================================================================================================
# 2. share links: only the service the user turned on and consented to, only on a tap
# ===================================================================================================


@needs_app
@needs_crypto
def test_share_links_reach_only_the_enabled_consented_provider(app_env, monkeypatch, tmp_path, capfd):
    from glossarion_mobile.services.share_providers import ShareError

    ts = _load("test_share_links.py", "_glossarion_ts_helpers_u10accept5")
    h = Harness(monkeypatch, tmp_path)
    key = "pd-key-u10accept5-" + hashlib.sha256(str(tmp_path).encode()).hexdigest()[:16]
    gofile, pixeldrain, send = ts.GofileFake(), ts.PixeldrainFake(keys=(key,)), ts.SendFake()
    h.spy.ports.update({gofile.port: "gofile", pixeldrain.port: "pixeldrain", send.http.port: "send-http",
                        send.ws_port: "send-ws"})
    facts: dict = {"secrets": {"pixeldrain key": key}}

    def fake_requests() -> dict:
        return {"gofile": len(gofile.requests), "pixeldrain": len(pixeldrain.requests),
                "send": len(send.raw) + len(send.http.requests)}

    async def refused(coro, code: str) -> None:
        with pytest.raises(ShareError) as info:
            await coro
        assert info.value.code == code, (info.value.code, info.value.message)

    async def consent(screen, pid: str, accept: bool) -> bool:
        before = screen.actions_.last_sheet
        task = asyncio.ensure_future(screen.set_provider(pid, True))
        shown = await _tf()._wait(lambda: isinstance(screen.actions_.last_sheet, dict)
                                  and screen.actions_.last_sheet is not before
                                  and "finish" in screen.actions_.last_sheet, timeout=10)
        assert shown, f"no consent sheet for {pid}"
        parts = screen.actions_.last_sheet
        facts.setdefault("consent", {})[pid] = list(parts["text"]["lines"])
        if accept:
            parts["set_rights"](True)
            parts["confirm"].on_click(None)
        else:
            parts["cancel"].on_click(None)
        answer = await asyncio.wait_for(task, 10)
        await h.dismiss_sheets()
        return answer

    async def scenario():
        app = await h.start()
        try:
            shares = app.share_links
            paths = rb.get_paths()
            output, data = Path(paths.output), Path(paths.data)
            shares._providers = {"gofile": gofile.provider(), "send": send.provider(),
                                 "pixeldrain": pixeldrain.provider()}
            assert shares.secrets.available(), "the app's API-key encryption is not available on the fake phone"
            book = _workspace(output, "Lantern Road")
            epub = os.path.join(book, "Lantern Road.epub")
            content = b"PK" + PRIVATE + b" compiled book " + os.urandom(200_000)
            with open(epub, "wb") as handle:
                handle.write(content)
            facts["config_before"] = h.config_keys()
            launched_before = len(h.launched_urls())

            # 1. every service is off: nothing can upload, nothing is contacted
            h.spy.phase = "share:off"
            assert not any(s.enabled or s.consented for s in shares.provider_states())
            for pid in ("gofile", "send", "pixeldrain"):
                await refused(shares.upload(pid, epub, book=book), "disabled")
            await refused(shares.start_handoff(epub, book=book), "disabled")
            assert fake_requests() == {"gofile": 0, "pixeldrain": 0, "send": 0}

            # 2. switched on without the consent sheet: still refused
            h.spy.phase = "share:no-consent"
            await shares.set_enabled("gofile", True)
            await refused(shares.upload("gofile", epub, book=book), "consent")
            await shares.set_enabled("gofile", False)
            assert fake_requests() == {"gofile": 0, "pixeldrain": 0, "send": 0}

            # 3. Settings: the consent sheet declined, then accepted (it says the file leaves the phone)
            h.spy.phase = "share:consent"
            await app.navigate("/settings/cloud")
            screen = app.shell.top_screen
            await _fresh(screen)
            assert await consent(screen, "gofile", accept=False) is False
            assert not shares.provider_state("gofile").enabled
            assert await consent(screen, "gofile", accept=True) is True
            state = shares.provider_state("gofile")
            assert state.enabled and state.consented and state.ready
            lines = " ".join(facts["consent"]["gofile"])
            assert "from this phone to Gofile" in lines and "Gofile can read the file" in lines, lines
            assert fake_requests() == {"gofile": 0, "pixeldrain": 0, "send": 0}  # turning it on sends nothing

            # 4. the user's tap: Gofile only
            h.spy.phase = "share:upload-gofile"
            await _fresh(screen)
            link = await screen.actions_.run_upload(screen.provider("gofile"), epub, book)
            await h.dismiss_sheets()
            assert link.get("ok") and link.get("url"), link
            assert fake_requests() == {"gofile": 1, "pixeldrain": 0, "send": 0}
            folder_id, entry = next(iter(gofile.folders.items()))
            assert entry["content"] == content  # exactly the file the user picked
            facts["secrets"].update({"gofile guest token": entry["token"], "gofile folder id": folder_id,
                                     "gofile link": link["url"]})
            assert h.spy.phase_labels("share:upload-gofile") <= {"gofile"}, h.spy.phase_labels("share:upload-gofile")

            # 5. the other services are still off
            h.spy.phase = "share:others-off"
            await refused(shares.upload("pixeldrain", epub, book=book), "disabled")
            await refused(shares.upload("send", epub, book=book), "disabled")
            assert fake_requests() == {"gofile": 1, "pixeldrain": 0, "send": 0}

            # 6. pixeldrain with the user's own key (typed into Settings, stored encrypted)
            h.spy.phase = "share:pixeldrain"
            assert await consent(screen, "pixeldrain", accept=True) is True
            await _fresh(screen)
            screen.key_fields["pixeldrain"].value = key
            assert await screen.save_key("pixeldrain")
            assert screen.key_fields["pixeldrain"].value == ""  # never left on screen
            assert fake_requests() == {"gofile": 1, "pixeldrain": 0, "send": 0}  # saving a key sends nothing
            await _fresh(screen)
            link = await screen.actions_.run_upload(screen.provider("pixeldrain"), epub, book)
            await h.dismiss_sheets()
            assert link.get("ok") and link.get("url"), link
            assert fake_requests() == {"gofile": 1, "pixeldrain": 1, "send": 0}
            assert next(iter(pixeldrain.files.values()))["content"] == content
            assert not any(key in str(r) for r in gofile.requests), "the pixeldrain key reached Gofile"
            facts["secrets"]["pixeldrain link"] = link["url"]
            assert h.spy.phase_labels("share:pixeldrain") <= {"pixeldrain"}, h.spy.phase_labels("share:pixeldrain")

            # 7. Send: end-to-end encrypted, the service sees neither the text, the name nor the key
            h.spy.phase = "share:send"
            assert await consent(screen, "send", accept=True) is True
            assert "Send (send.vis.ee) cannot read the file" in " ".join(facts["consent"]["send"])
            await _fresh(screen)
            link = await screen.actions_.run_upload(screen.provider("send"), epub, book)
            await h.dismiss_sheets()
            assert link.get("ok") and "#" in str(link.get("url")), link
            secret_key = str(link["url"]).split("#", 1)[1]
            frames = b"".join(f if isinstance(f, bytes) else f.encode() for f in send.raw)
            assert PRIVATE not in frames and b"compiled book" not in frames, "Send received plain text"
            assert b"Lantern Road" not in frames, "Send received the file name"
            assert secret_key.encode() not in frames, "Send received the decryption key"
            (file_id, entry), = send.files.items()
            facts["secrets"].update({"send link": link["url"], "send key": secret_key, "send owner token": entry["owner"]})
            facts["send_link_id"] = link.get("id")
            assert fake_requests()["gofile"] == 1 and fake_requests()["pixeldrain"] == 1
            assert h.spy.phase_labels("share:send") <= {"send-http", "send-ws"}, h.spy.phase_labels("share:send")

            # 8. transfer.it: a browser hand-off; Glossarion itself sends nothing anywhere
            h.spy.phase = "share:transferit"
            before = fake_requests()
            assert await consent(screen, "transferit", accept=True) is True
            assert "Glossarion sends nothing to transfer.it" in " ".join(facts["consent"]["transferit"])
            plan = await screen.share_facade.start_handoff(epub, book)
            assert plan.get("ok"), plan
            assert h.launched_urls()[launched_before:] == ["https://transfer.it/start"]
            saved = [a for p, m, a in h.dart.calls if p == "share:transferit" and m == "save_to_downloads"]
            assert len(saved) == 1 and _norm(saved[0]["path"]) == _norm(epub)
            pasted = "https://transfer.it/t/" + hashlib.sha256(key.encode()).hexdigest()[:20]
            stored = await screen.share_facade.save_pasted_link(f"Here: {pasted}", path=epub, workspace=book)
            assert stored.get("ok"), stored
            facts["secrets"]["transfer.it link"] = pasted
            assert fake_requests() == before
            assert not h.spy.phase_labels("share:transferit"), h.spy.phase_labels("share:transferit")

            # 9. never a key export, the config or anything from the data folder
            h.spy.phase = "share:not-a-book"
            exports = data / "Exports"
            exports.mkdir(exist_ok=True)
            key_export = exports / "glossarion-key-pools-20261009-120000.json"
            key_export.write_text(json.dumps({"openai": ["sk-not-a-real-key"]}), encoding="utf-8")
            stray_epub = exports / "Lantern Road copy.epub"
            stray_epub.write_bytes(content)
            for path in (key_export, Path(paths.config_file), stray_epub):
                assert not shares.eligible(str(path)), path
                assert all(reason for _state, reason in shares.menu(str(path))), path
                await refused(shares.upload("gofile", str(path)), "not_allowed")
            options = app.files.export_options(str(key_export))
            assert not [o for o in options if "link" in f"{o.id} {o.label}".lower()], options
            assert fake_requests() == before

            # 10. consent withdrawn: refused again
            h.spy.phase = "share:revoked"
            await shares.revoke_consent("gofile")
            await refused(shares.upload("gofile", epub, book=book), "disabled")
            assert fake_requests() == before

            # 11. at rest: every secret sealed
            h.spy.phase = "share:at-rest"
            app.prefs.flush()
            store = json.loads((data / "mobile_share_links.json").read_text(encoding="utf-8"))
            assert {link["provider"] for link in store["links"]} == {"gofile", "pixeldrain", "send", "transferit"}
            for record in store["links"]:
                assert str(record["url"]).startswith("ENC:"), record["provider"]
                assert record["delete"] is None or str(record["delete"]).startswith("ENC:"), record["provider"]
            assert store["secrets"] and all(str(v).startswith("ENC:") for v in store["secrets"].values())
            facts["leaks_with_links"] = _leaks(tmp_path, facts["secrets"])

            # 12. Delete link (the user's tap): only Send is contacted
            h.spy.phase = "share:delete-link"
            removed = await screen.share_facade.delete_link(facts["send_link_id"])
            assert removed.get("ok"), removed
            assert not send.files and fake_requests()["gofile"] == 1 and fake_requests()["pixeldrain"] == 1
            assert h.spy.phase_labels("share:delete-link") <= {"send-http"}, h.spy.phase_labels("share:delete-link")

            # 13. deleting the book drops its links and deletes nothing remotely
            h.spy.phase = "share:delete-book"
            before = fake_requests()
            await app.library.refresh(quiet=True)
            from glossarion_mobile.services.library import book_identity

            row = next(r for r in app.library.snapshot.all_books() if _norm(book_identity(r)) == _norm(book))
            plan = await asyncio.to_thread(app.library.plan_delete_blocking, [row])
            await asyncio.to_thread(app.library.execute_delete_blocking, plan)
            assert not await shares.links_for(book=book)
            assert fake_requests() == before and gofile.folders and pixeldrain.files

            # 14. Wipe app data: links, keys and the guest token go; nothing is sent
            h.spy.phase = "share:wipe"
            await shares.wipe()
            assert not (data / "mobile_share_links.json").exists()
            assert fake_requests() == before
            h.spy.phase = "share:done"
            facts["config_after"] = h.config_keys()
            facts["client"] = h.client_invocations()
            facts["dart_args"] = [repr(a) for _p, _m, a in h.dart.calls]
        finally:
            await h.stop()

    try:
        _run(h, scenario)
    finally:
        for fake in (gofile, pixeldrain, send):
            fake.close()

    share_events = h.spy.in_phases("share:")
    assert not [e for e in share_events if not e["local"]], "off-machine attempts:\n" + NetSpy.show(share_events)
    assert not [e for e in share_events if e["label"] in ("loopback", "library")], \
        "requests to something other than the enabled service:\n" + NetSpy.show(share_events)
    quiet = ("share:off", "share:no-consent", "share:consent", "share:others-off", "share:transferit",
             "share:not-a-book", "share:revoked", "share:at-rest", "share:delete-book", "share:wipe", "share:done")
    noisy = [e for e in share_events if e["phase"] in quiet and e["label"] != "internal"]
    assert not noisy, "network attempts without an upload tap:\n" + NetSpy.show(noisy)
    # secrets: nowhere in plain text
    secrets = facts["secrets"]
    assert not facts["leaks_with_links"], f"plain-text secrets at rest: {facts['leaks_with_links']}"
    assert not _leaks(tmp_path, secrets), f"plain-text secrets at rest after the wipe: {_leaks(tmp_path, secrets)}"
    logs = "\n".join(h.records.lines)
    assert h.records.lines, "no log record was captured"
    assert not [n for n, s in secrets.items() if s in logs], [n for n, s in secrets.items() if s in logs]
    out, err = capfd.readouterr()
    assert not [n for n, s in secrets.items() if s in out or s in err], "a secret was printed"
    client = "\n".join(f"{name} {args}" for name, args in facts["client"])
    assert not [n for n, s in secrets.items() if s in client], "a secret was sent to the Flutter client"
    native = "\n".join(facts["dart_args"])
    assert not [n for n, s in secrets.items() if s in native], "a secret was sent to the native side"
    _assert_no_new_config_keys(facts["config_before"], facts["config_after"])
    assert _src_hashes() == h.src_hashes, "a desktop src/*.py changed while the app ran"
    by_phase: dict = {}
    for e in h.spy.events:
        by_phase.setdefault(e["phase"], set()).add(e["label"])
    print(f"U10 accept5 share: attempts by phase {sorted((k, sorted(v)) for k, v in by_phase.items())}; "
          f"secrets checked {sorted(secrets)}; {len(h.records.lines)} log records; "
          f"config keys {sorted(facts['config_after'] or ())}")


def _leaks(root: Path, secrets: dict) -> list:
    """``(file, secret name)`` for every secret found in plain text under ``root``."""
    found = []
    needles = {name: value.encode("utf-8") for name, value in secrets.items() if value}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        try:
            blob = path.read_bytes()
        except OSError:
            continue
        for name, needle in needles.items():
            if needle in blob:
                found.append((str(path.relative_to(root)), name))
    return found


# ===================================================================================================
# 3. static: no network path in the cloud sync, only the providers' own in the share links
# ===================================================================================================

_NETWORK_ROOTS = {"socket", "ssl", "http", "urllib3", "requests", "httpx", "aiohttp", "websockets", "websocket",
                  "ftplib", "smtplib", "telnetlib", "xmlrpc", "grpc"}


def _imports(path: Path) -> set:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path), feature_version=(3, 10))
    names: set = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
            if node.module == "urllib":
                names.update(f"urllib.{alias.name}" for alias in node.names)
    return names


def _network_imports(path: Path) -> set:
    return {n for n in _imports(path) if n.split(".")[0] in _NETWORK_ROOTS or n.startswith("urllib.request")}


def _url_hosts(path: Path) -> set:
    tree = ast.parse(path.read_text(encoding="utf-8"), feature_version=(3, 10))
    hosts = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            text = node.value.replace("\\.", ".")  # a regex literal such as ^https://transfer\.it/t/
            hosts.update(h.lower() for h in re.findall(r"\b(?:https?|wss?)://([A-Za-z0-9.-]+)", text) if h)
    return hosts


def test_cloud_and_share_sources_have_no_network_path():
    cloud_python = [PACKAGE / "services" / "cloud_sync.py", PACKAGE / "state" / "cloud_records.py",
                    PACKAGE / "ui" / "screens" / "cloud_sync.py", PACKAGE / "services" / "native.py",
                    EXTENSION_PKG / "documents.py", EXTENSION_PKG / "documents_fake.py", EXTENSION_PKG / "native.py"]
    for path in cloud_python:
        assert not _network_imports(path), (path.name, _network_imports(path))
        assert "open_connection" not in path.read_text(encoding="utf-8"), path.name
    for path in cloud_python[:2] + cloud_python[4:6]:
        assert not _url_hosts(path), (path.name, _url_hosts(path))
    kotlin = FLUTTER_PKG / "android" / "src" / "main" / "kotlin" / "com" / "glossarion" / "flet_glossarion_native" \
        / "DocumentDestinations.kt"
    swift = FLUTTER_PKG / "ios" / "flet_glossarion_native" / "Sources" / "flet_glossarion_native" \
        / "DocumentDestinations.swift"
    for path, banned in ((kotlin, ("java.net.", "HttpURLConnection", "okhttp", "OkHttp", "Socket(")),
                         (swift, ("URLSession", "NSURLConnection", "NWConnection", "import Network", "CFStream"))):
        text = path.read_text(encoding="utf-8")
        assert not [b for b in banned if b in text], (path.name, [b for b in banned if b in text])
    # share links: the service itself never talks to the network; the providers use the stdlib client
    # (``requests`` / ``httpx`` are patched by the optional HTTP log) and only their documented hosts
    services = PACKAGE / "services"
    assert not _network_imports(services / "share_links.py"), _network_imports(services / "share_links.py")
    allowed_imports = {"__init__.py": {"http.client", "socket", "ssl"},
                       "send_e2ee.py": {"websockets", "websockets.sync.client", "websockets.exceptions"}}
    allowed_hosts = {"transfer.it", "gofile.io", "api.gofile.io", "upload.gofile.io", "send.vis.ee",
                     "pixeldrain.com", "github.com"}
    providers = sorted((services / "share_providers").glob("*.py"))
    assert {p.name for p in providers} >= {"__init__.py", "gofile.py", "pixeldrain.py", "send_e2ee.py",
                                           "transferit_handoff.py"}
    for path in providers:
        extra = _network_imports(path) - allowed_imports.get(path.name, set())
        assert not extra, (path.name, extra)
        assert _url_hosts(path) <= allowed_hosts, (path.name, _url_hosts(path) - allowed_hosts)
    assert _url_hosts(services / "share_links.py") <= allowed_hosts


# ===================================================================================================
# 4. desktop untouched
# ===================================================================================================


def test_desktop_sources_are_byte_identical_to_head():
    git = shutil.which("git")
    if git is None or not (REPO_DIR / ".git").exists():
        pytest.skip("not a git checkout")

    def run(*args: str) -> list:
        done = subprocess.run([git, "-C", str(REPO_DIR), *args], capture_output=True, text=True, timeout=120)
        if done.returncode != 0:
            pytest.skip(f"git {' '.join(args)} failed: {done.stderr.strip()[:200]}")
        return [line for line in done.stdout.splitlines() if line.strip()]

    spec = ":(glob)src/*.py"
    changed = run("diff", "--name-only", "HEAD", "--", spec)
    untracked = run("ls-files", "--others", "--exclude-standard", "--", spec)
    assert not changed and not untracked, f"desktop modules differ from HEAD: {changed + untracked}"
