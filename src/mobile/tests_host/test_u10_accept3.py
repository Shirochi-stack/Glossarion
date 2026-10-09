"""U10 acceptance, item 3: "Failures never lose or duplicate files".

End to end on the real app: ``GlossarionApp`` started on a fake Flet session as an Android phone (the
``test_ui_foundations`` harness), its real ``LibraryService`` scanning real workspaces under the app's output
root, the real ``CloudSyncService`` with its ``CloudRecordStore`` (``mobile_cloud.json``) and Prefs on disk,
the real ``JobService`` transition listeners, ``NativeBridge`` / ``ServiceLease`` / notifications, Settings ›
Danger zone and the Library's delete. Only the phone's native side is a fake:
``flet_glossarion_native.documents_fake`` (``FakeDocumentsNative`` + ``FakeCloudProvider``, the semantics of
``DocumentDestinations.kt``: the ``wt -> rwt -> w`` chain with the non-truncating guard and read-back size,
"missing only when the folder root answers", create retries that adopt a late file, persisted grants), plus
the job-service / notification calls. The provider object outlives an app launch like a real cloud app does,
so a "killed" app is started again over the same cloud, grants and data folder. Nothing reaches the network
(non-loopback connects and DNS lookups fail).

Every scenario ends by checking the cloud folder file by file: exactly one current copy of each book file
(no ' (2)', no second same-name file, no torn copy), and the local book untouched.

* write modes: ``wt`` rejected ("Unsupported mode" and a bare FileNotFoundException) -> ``rwt`` in place; a
  non-truncating ``w`` (seekable and pipe): longer -> in place, shorter -> create-new + delete-old with a
  warning; a replace whose new copy fails keeps the old file;
* FileNotFoundException while the folder root answers (offline provider, a locked document) is never taken
  as "deleted"; a real deletion (file, whole book folder) is created again, once per attempt;
* a SecurityException on one document (moved out of the folder) affects only that book;
* provider offline then online: the queue drains (backoff timer, Retry now), one failure notification only;
  creates that fail (or succeed late) never leave a second file;
* the app killed mid-write: the torn cloud copy is rewritten in place from a fresh private snapshot at the
  next launch; a first write's torn file is adopted by name;
* the app killed mid-replace, and a replace whose delete-old fails: no orphan / duplicate copy stays;
* two books with the same compiled title in a flat folder never share a cloud file, also after the records
  are lost; a provider that cannot create folders goes flat;
* a deleted Library book drops its records and queue entry (also when deleted mid-write); cloud files stay;
* Settings › Danger zone "Wipe app data" releases every persisted grant first; cloud files stay.

Run from src/mobile with the mobile venv:
    .venv/Scripts/python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic \
        tests_host/test_u10_accept3.py
"""

from __future__ import annotations

import asyncio
import dataclasses
import importlib.util
import itertools
import json
import os
import socket
import sys
import time
from pathlib import Path

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = MOBILE_DIR / "app"
EXTENSION_SRC = MOBILE_DIR / "extensions" / "flet_glossarion_native" / "src"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


def _has(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _load(name: str, alias: str):
    spec = importlib.util.spec_from_file_location(alias, Path(__file__).with_name(name))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_TB = _load("test_bootstrap.py", "_glossarion_tb_helpers_u10accept3")
storage = _TB.storage
app_env = _TB.app_env

pytestmark = [
    pytest.mark.skipif(not (_has("flet") and _has("msgpack")), reason="flet/msgpack not installed"),
    pytest.mark.skipif(not (_has("requests") and _has("bs4")), reason="the real features drive the backend"),
    pytest.mark.skipif(not (EXTENSION_SRC / "flet_glossarion_native" / "documents_fake.py").is_file(),
                       reason="flet_glossarion_native extension sources not present"),
]

EPUB_A = b"PK\x03\x04 epub version A " * 300  # 7.2 KB
EPUB_B = b"PK\x03\x04 epub version B, a longer recompile " * 300  # 13 KB
EPUB_C = b"PK\x03\x04 C " * 200  # 1.8 KB: shorter than A and B
EPUB_D = b"PK\x03\x04 D " * 60  # shorter than C
DEFAULT_MODES = {"wt", "rwt", "w", "rw", "r"}
BACKOFF_WAIT = 12.0  # the first retry comes 2 s after a failure (cloud_sync.BACKOFF[0]); generous for CI


# ---------------------------------------------------------------------------------------------------
# the phone's native side
# ---------------------------------------------------------------------------------------------------


def _import_fake(monkeypatch):
    """``documents_fake`` from the extension sources, without leaving the extension importable for later tests
    (``native.load_extension`` would then build the real GlossarionNative on their fake sessions)."""
    before = {name for name in sys.modules if name.split(".")[0] == "flet_glossarion_native"}
    monkeypatch.syspath_prepend(str(EXTENSION_SRC))
    from flet_glossarion_native import documents_fake

    added = [name for name in sys.modules if name.split(".")[0] == "flet_glossarion_native" and name not in before]
    return documents_fake, added


class Phone:
    """One phone across app launches: the cloud app's provider (it outlives Glossarion's process, grants and
    all) and the native service of the current launch."""

    def __init__(self, fake) -> None:
        self.fake = fake
        self.provider = fake.FakeCloudProvider(authority="com.google.android.apps.docs.storage", label="Drive")
        self.folder = self.provider.add_folder("Glossarion books")  # the folder the user picks
        self.elsewhere = self.provider.add_folder("Elsewhere")  # outside the picked folder
        self.native = None
        self.natives: list = []

    def create_native(self, page=None, handlers=None):
        self.native = PhoneNative.build(self.fake, self.provider, handlers)
        self.natives.append(self.native)
        return self.native

    # ---- what is in the cloud ----
    def tree(self, node=None, prefix=""):
        """``{"Book/Book.epub": bytes}`` of the live files under the picked folder; a second live item with the
        same path (Drive allows duplicate names) is listed as ``path#2``."""
        node = node or self.folder
        out: dict = {}
        for child in node.children:
            if child.deleted:
                continue
            path = f"{prefix}{child.name}"
            items = self.tree(child, path + "/").items() if child.is_dir else [(path, bytes(child.data))]
            for key, value in items:
                final, n = key, 1
                while final in out:
                    n += 1
                    final = f"{key}#{n}"
                out[final] = value
        return out

    def node(self, path):
        current = self.folder
        for part in path.split("/"):
            current = next(c for c in current.children if not c.deleted and c.name == part)
        return current

    def delete_tree(self, node):
        """The user deletes a folder in the cloud app: it and everything in it."""
        node.deleted = True
        for child in node.children:
            self.delete_tree(child)

    def move_out(self, path):
        """The user moves a file out of the picked folder (it lives on elsewhere in their cloud)."""
        node = self.node(path)
        node.parent.children.remove(node)
        node.parent = self.elsewhere
        self.elsewhere.children.append(node)
        self.provider.move_out(node)
        return node

    def grant_count(self) -> int:
        return len(self.provider.grants)


class PhoneNative:
    """Built per launch: ``FakeDocumentsNative`` + the job-service / notification / background calls the app
    makes through ``NativeBridge``. ``kill`` makes the next write tear the cloud copy, take a picture of the
    app's files at that instant and never return (the process died); ``gate`` holds writes until set."""

    @classmethod
    def build(cls, fake, provider, handlers):
        base = fake.FakeDocumentsNative

        class _Native(cls, base):
            pass

        native = _Native(provider, chunk_bytes=4096)
        native.on_progress = native._progress
        native._setup(handlers)
        return native

    is_stub = False
    native_available = True

    def _setup(self, handlers):
        self.handlers = dict(handlers or {})
        self.service_running = False
        self.service_log: list = []
        self.notifications: list = []
        self.write_log: list = []  # (document uri, source path)
        self.kill = None
        self.gate = None
        self.fail_deletes = 0
        self.delete_log: list = []

    def _progress(self, event):
        handler = self.handlers.get("on_document")
        if handler is not None:
            try:
                asyncio.get_running_loop().create_task(handler(dict(event)))
            except RuntimeError:
                pass

    # ---- documents (the extension fake), plus the failure switches ----
    async def write_file(self, target_or_doc, source_path, **kwargs):
        parsed = self._parse(target_or_doc)
        self.write_log.append((parsed[2] if parsed else None, source_path))
        if self.gate is not None:
            await self.gate.wait()
        kill = self.kill
        if kill is not None and parsed is not None:
            self.kill = None
            document = parsed[2]
            modes = [m for m in kwargs.get("mode_chain") or ("wt", "rwt", "w") if m in self.provider.accepted_modes]
            fd = self.provider.open(document, modes[0])  # 'wt' truncates before the first byte (critic #5)
            data = Path(source_path).read_bytes()
            half = max(1, len(data) // 2)
            fd.write(data[:half])  # half the book reached the provider ...
            kill.update(source=source_path, document=document, snapshot=data, written=half)
            kill["state"] = kill["capture"]()  # what is on disk at the instant of death
            kill["event"].set()
            await asyncio.Event().wait()  # ... and the process died: nothing after this line ever runs
        return await super().write_file(target_or_doc, source_path, **kwargs)

    async def delete(self, document):
        parsed = self._parse(document)
        self.delete_log.append(parsed[2] if parsed else None)
        if self.fail_deletes > 0:
            self.fail_deletes -= 1
            return {"ok": False, "error": "provider_error", "message": "Drive: try again", "scope": "document",
                    "retryable": True}
        return await super().delete(document)

    # ---- the rest of GlossarionNative the app calls ----
    async def get_platform_info(self):
        return {"platform": "android", "native": True, "documents": True, "save_to_downloads": True,
                "sdk_int": 34, "persisted_grant_limit": 512}

    async def init_notifications(self, channels=None, request_permission=True):
        return True

    async def show_notification(self, notification_id, title, body, *, channel_id="jobs.action", payload=None,
                                actions=None):
        self.notifications.append({"id": notification_id, "title": title, "body": body, "channel": channel_id,
                                   "payload": payload})
        return True

    async def cancel_notification(self, notification_id):
        return None

    async def start_job_service(self, title, text, buttons=None):
        self.service_running = True
        self.service_log.append(("start", text))
        return True

    async def update_job_service(self, title=None, text=None):
        self.service_log.append(("update", text))

    async def stop_job_service(self):
        self.service_running = False
        self.service_log.append(("stop", None))

    async def is_job_service_running(self):
        return self.service_running

    async def get_initial_shared(self):
        return []

    async def clear_shared(self, delete_files=False):
        return None

    async def begin_background_task(self, name):
        return 1

    async def end_background_task(self, task_id):
        return None

    async def background_time_remaining(self):
        return None


# ---------------------------------------------------------------------------------------------------
# fixtures and helpers
# ---------------------------------------------------------------------------------------------------


@pytest.fixture
def phone(app_env, storage, tmp_path, monkeypatch):
    """The real app's isolated storage plus the phone: real data never (Library, output, HOME, USERPROFILE,
    APPDATA and the app data all live under tmp_path), no network, the fake native side."""
    for name, sub in (("USERPROFILE", "home"), ("APPDATA", "appdata"), ("LOCALAPPDATA", "localappdata")):
        folder = tmp_path / "isolated" / sub
        folder.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv(name, str(folder))
    monkeypatch.setenv("GLOSSARION_HTTP_LOG", "0")
    _no_network(monkeypatch)
    fake, added = _import_fake(monkeypatch)
    from glossarion_mobile.services import native as native_mod

    the_phone = Phone(fake)
    monkeypatch.setattr(native_mod, "create_native", the_phone.create_native)
    try:
        yield the_phone
    finally:
        for name in added:
            sys.modules.pop(name, None)
        try:
            import library_core

            library_core.uninstall_library_env()
        except Exception:
            pass


def _no_network(monkeypatch):
    """Cloud sync never touches the network; anything else the app tries offline-fails (no real service)."""
    original, original_ex = socket.socket.connect, socket.socket.connect_ex
    original_getaddrinfo = socket.getaddrinfo
    loopback = ("127.0.0.1", "::1", "localhost")

    def guard(fn):
        def connect(self, address, *args, **kwargs):
            host = address[0] if isinstance(address, tuple) and address else address
            if host not in loopback:
                raise ConnectionRefusedError(f"offline test: {address!r}")
            return fn(self, address, *args, **kwargs)

        return connect

    def getaddrinfo(host, *args, **kwargs):
        if host is not None and host not in loopback:
            raise socket.gaierror(f"offline test: {host!r}")
        return original_getaddrinfo(host, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", guard(original))
    monkeypatch.setattr(socket.socket, "connect_ex", guard(original_ex))
    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)


def _tf():
    return _load("test_ui_foundations.py", "_glossarion_tf_helpers_u10accept3")


def run(coro):
    return asyncio.run(coro)


async def launch(tf):
    main_module, conn, session, page, app = await tf._start("android")
    app._accept3_session = (main_module, conn, session, page)  # the fake Flet session lives as long as the app
    assert await tf._wait(lambda: app.cloud_sync is not None and app.library is not None, timeout=30)
    assert app.cloud_sync.supported and app.cloud_sync.platform == "android"
    return app


async def shutdown(tf, app):
    """A normal close (the app goes to the background, then away)."""
    try:
        for name in ("config_store", "prefs"):
            target = getattr(app, name, None)
            if target is not None:
                target.flush()
        if getattr(app, "jobs", None) is not None:
            app.jobs.close()
    finally:
        await tf._stop(app)
        if app.cloud_sync is not None:
            app.cloud_sync.close()


async def die(tf, app):
    """The process is killed: no store saves anything more (the disk keeps what the kill picture holds)."""
    for target in (app.cloud_sync.store, app.prefs, getattr(app, "config_store", None)):
        saver = getattr(target, "_saver", None)
        if saver is not None:
            saver.close(timeout=2.0)
    try:
        if getattr(app, "jobs", None) is not None:
            app.jobs.close()
    finally:
        await tf._stop(app)


def disk_state(app):
    """The cloud sync's files on disk: ``mobile_cloud.json``, Prefs (``mobile_state.json``), snapshots."""
    data = Path(app.paths.data)
    files = {}
    for name in ("mobile_cloud.json", "mobile_state.json"):
        path = data / name
        files[str(path)] = path.read_bytes() if path.exists() else None
    snap_dir = Path(app.cloud_sync._snapshot_dir())
    snaps = {str(p): p.read_bytes() for p in snap_dir.iterdir()} if snap_dir.is_dir() else {}
    return {"files": files, "snapshots": snaps, "snapshot_dir": str(snap_dir)}


def restore(state):
    for path, data in state["files"].items():
        if data is None:
            Path(path).unlink(missing_ok=True)
        else:
            Path(path).write_bytes(data)
    snap_dir = Path(state["snapshot_dir"])
    if snap_dir.is_dir():
        for leftover in snap_dir.iterdir():
            leftover.unlink()
    for path, data in state["snapshots"].items():
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(data)


async def settle(app, rounds: int = 8) -> None:
    cloud = app.cloud_sync
    for _ in range(rounds):
        await asyncio.sleep(0.02)
        await cloud.wait_idle()
    await asyncio.sleep(0.05)


async def until(predicate, timeout=10.0, what=""):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.05)
    assert predicate(), f"timed out waiting for {what() if callable(what) else what}"


async def refresh_library(app):
    await app.library.refresh(quiet=True, reason="test")


def make_book(app, name, files) -> str:
    """A finished Library workspace under the app's output root (what a translation + compile leaves)."""
    ws = Path(app.paths.output) / name
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "translation_progress.json").write_text(json.dumps({"version": "2.1", "chapters": {}, "chapter_chunks": {}}),
                                                  encoding="utf-8")
    (ws / "response_001_Chapter 1.html").write_text("<p>translated</p>", encoding="utf-8")
    for fname, data in files.items():
        (ws / fname).write_bytes(data)
    return str(ws)


def recompile(folder, name, data, *, later=5.0) -> str:
    path = Path(folder) / name
    path.write_bytes(data)
    stamp = time.time() + later
    os.utime(path, (stamp, stamp))
    return str(path)


_JOB_IDS = itertools.count(1)  # unique per delivery (the cloud sync ignores a job id it has seen)


def job_done(app, folder, *, outputs=(), kind="compile_epub"):
    """A Library Compile reaching DONE, delivered by the real JobService to every transition listener
    (JobsFeature / BackgroundExecution, the Library, the cloud sync)."""
    from glossarion_mobile.services.jobs import JobSnapshot, JobSpec, JobState

    spec = JobSpec(kind=kind, title=os.path.basename(folder), inputs=(folder,), params={"folder": folder},
                   origin={"type": "library", "bid": app.library.bid_for({"output_folder": folder})})
    snap = JobSnapshot(id=f"accept3-job-{next(_JOB_IDS)}", spec=spec, state=JobState.DONE, created=time.time(),
                       outputs=tuple(outputs), output_dir=folder)
    app.job_service._deliver_transition(snap, JobState.RUNNING)
    return snap


def facade(app):
    from glossarion_mobile.ui.screens.cloud_sync import CloudFacade

    return CloudFacade(app.cloud_sync)


async def link_folder(app, phone, *, enable=True):
    phone.provider.next_pick("folder", phone.folder)
    answer = await facade(app).call("pick_folder")
    assert answer["ok"], answer
    if enable:
        assert (await facade(app).call("set_enabled", True))["ok"]
    await settle(app)


def record(app, folder, kind="epub"):
    from glossarion_mobile.state.cloud_records import book_key

    dest = app.cloud_sync.destination()
    return app.cloud_sync.store.record(dest.id, book_key(folder), kind) or {}


def queued(app, folder):
    return app.cloud_sync.store.queued(folder)


def doc_id(rec) -> str:
    doc = rec.get("doc") or {}
    return str(doc.get("document") or "").rsplit("/", 1)[-1]


def creates(native, name=None) -> list:
    return [c for c in native.calls
            if c[0] in ("create_file", "create_folder") and (name is None or c[1]["name"] == name)]


def failure_notices(phone) -> list:
    return [n for native in phone.natives for n in native.notifications
            if "couldn't save" in n["title"].lower() or "lost access" in n["title"].lower()]


def assert_tree(phone, expected, what=""):
    actual = phone.tree()
    if actual != expected:
        def show(tree):
            return {k: (v[:18], len(v)) for k, v in sorted(tree.items())}

        raise AssertionError(f"cloud folder {what}: expected {show(expected)}, got {show(actual)}")


# ---------------------------------------------------------------------------------------------------
# 1. write modes: unsupported 'wt', non-truncating 'w', create-new + delete-old
# ---------------------------------------------------------------------------------------------------


def test_write_mode_failures_overwrite_in_place_or_replace_without_duplicates(phone):
    from glossarion_mobile.services.cloud_sync import REPLACED_NOTE

    tf = _tf()
    p = phone.provider

    async def scenario():
        app = await launch(tf)
        try:
            # --- wt rejected ("Unsupported mode", Drive issue 180526528): rwt overwrites the same document
            p.accepted_modes = {"rwt", "w", "rw", "r"}
            modes = make_book(app, "Modes", {"Modes.epub": EPUB_A})
            await refresh_library(app)
            await link_folder(app, phone)
            assert_tree(phone, {"Modes/Modes.epub": EPUB_A}, "after the first save")
            first = doc_id(record(app, modes))
            assert record(app, modes)["mode"] == "rwt"
            recompile(modes, "Modes.epub", EPUB_C)  # shorter: rwt truncates
            job_done(app, modes, outputs=[os.path.join(modes, "Modes.epub")])
            await settle(app)
            assert_tree(phone, {"Modes/Modes.epub": EPUB_C}, "after a shorter recompile over rwt")
            assert doc_id(record(app, modes)) == first and record(app, modes)["status"] == "ok"

            # --- wt refused with a bare FileNotFoundException (no "mode" in it): still not "deleted", rwt next
            p.accepted_modes = set(DEFAULT_MODES)
            plain_open = p.open

            def open_without_wt(uri, mode):
                if mode == "wt":
                    raise phone.fake.ProviderFault("not_found", "open failed: ENOENT (No such file or directory)")
                return plain_open(uri, mode)

            p.open = open_without_wt
            recompile(modes, "Modes.epub", EPUB_B, later=10)
            before = len(creates(phone.native))
            job_done(app, modes)
            await settle(app)
            assert_tree(phone, {"Modes/Modes.epub": EPUB_B}, "after wt threw FileNotFoundException")
            assert doc_id(record(app, modes)) == first and len(creates(phone.native)) == before
            p.open = plain_open

            # --- only a non-truncating 'w' (OneDrive-like), seekable: longer in place (cut to length) ...
            p.accepted_modes = {"w", "r"}
            p.truncates_w = False
            tail = make_book(app, "Tail", {"Tail.epub": EPUB_A})
            job_done(app, tail)
            await settle(app)
            tail_doc = doc_id(record(app, tail))
            recompile(tail, "Tail.epub", EPUB_B)
            job_done(app, tail)
            await settle(app)
            assert_tree(phone, {"Modes/Modes.epub": EPUB_B, "Tail/Tail.epub": EPUB_B}, "after a longer 'w'")
            assert doc_id(record(app, tail)) == tail_doc
            # ... shorter: never a stale tail; a new copy replaces the old one, with a warning (critic #5)
            recompile(tail, "Tail.epub", EPUB_C, later=10)
            job_done(app, tail)
            await settle(app)
            assert_tree(phone, {"Modes/Modes.epub": EPUB_B, "Tail/Tail.epub": EPUB_C}, "after a shorter 'w'")
            rec = record(app, tail)
            assert rec["status"] == "ok" and rec["note"] == REPLACED_NOTE and doc_id(rec) != tail_doc
            entry = (await app.cloud_sync._io(app.cloud_sync.book_state, tail))["files"]["epub"]
            assert entry["status"] == "ok" and entry["warning"]  # "its link changed"

            # --- a pipe (no ftruncate, size from the provider's row): longer in place, shorter replaced
            p.seekable = False
            pipe_doc = doc_id(record(app, tail))
            recompile(tail, "Tail.epub", EPUB_A, later=15)
            job_done(app, tail)
            await settle(app)
            assert_tree(phone, {"Modes/Modes.epub": EPUB_B, "Tail/Tail.epub": EPUB_A}, "after a longer pipe write")
            assert doc_id(record(app, tail)) == pipe_doc
            recompile(tail, "Tail.epub", EPUB_C, later=20)
            job_done(app, tail)
            await settle(app)
            assert_tree(phone, {"Modes/Modes.epub": EPUB_B, "Tail/Tail.epub": EPUB_C}, "after a shorter pipe write")

            # --- the replacement's write fails half way: the old copy stays, the partial new one is removed
            p.fail_after_bytes = 100
            recompile(tail, "Tail.epub", EPUB_D, later=25)
            job_done(app, tail)
            await settle(app)
            assert_tree(phone, {"Modes/Modes.epub": EPUB_B, "Tail/Tail.epub": EPUB_C}, "after a failed replace")
            assert queued(app, tail)["attempts"] == 1
            p.fail_after_bytes = None
            await until(lambda: queued(app, tail) is None, BACKOFF_WAIT, "the replace retry")
            await settle(app)
            assert_tree(phone, {"Modes/Modes.epub": EPUB_B, "Tail/Tail.epub": EPUB_D}, "after the retried replace")
            # --- 'wt' that does not truncate (it behaves like OneDrive's 'w'): the read-back length shows the
            # stale tail, and the file is replaced (never left as a corrupt ZIP)
            p.accepted_modes = set(DEFAULT_MODES)
            p.seekable = True

            def lying_open(uri, mode):
                return plain_open(uri, "w" if mode == "wt" else mode)  # truncates_w is still False

            p.open = lying_open
            liar = make_book(app, "Liar", {"Liar.epub": EPUB_B})
            job_done(app, liar)
            await settle(app)
            recompile(liar, "Liar.epub", EPUB_C)
            job_done(app, liar)
            await settle(app)
            p.open = plain_open
            assert_tree(phone, {"Modes/Modes.epub": EPUB_B, "Tail/Tail.epub": EPUB_D, "Liar/Liar.epub": EPUB_C},
                        "after a 'wt' that kept the old tail")
            assert record(app, liar)["note"] == REPLACED_NOTE
            # the local books are never touched
            assert Path(modes, "Modes.epub").read_bytes() == EPUB_B and Path(tail, "Tail.epub").read_bytes() == EPUB_D
            assert not failure_notices(phone)
        finally:
            await shutdown(tf, app)

    run(scenario())


# ---------------------------------------------------------------------------------------------------
# 2. FileNotFoundException is not "deleted"; a proven deletion is created again, once
# ---------------------------------------------------------------------------------------------------


def test_not_found_is_deleted_only_when_the_folder_proves_it(phone):
    tf = _tf()
    p = phone.provider

    async def scenario():
        app = await launch(tf)
        try:
            books = {name: make_book(app, name, {f"{name}.epub": EPUB_A}) for name in ("Offline", "Locked", "Gone",
                                                                                       "FolderGone", "Vanish")}
            await refresh_library(app)
            await link_folder(app, phone)
            expected = {f"{name}/{name}.epub": EPUB_A for name in books}
            assert_tree(phone, expected, "after the first save")
            docs = {name: doc_id(record(app, folder)) for name, folder in books.items()}

            # --- offline (Nextcloud "Error downloading file"; the root still answers): retried, nothing created
            p.offline = True
            recompile(books["Offline"], "Offline.epub", EPUB_B)
            before = len(creates(phone.native))
            job_done(app, books["Offline"])
            await settle(app)
            assert_tree(phone, expected, "while the provider is offline")  # the old copy intact, not torn
            entry = queued(app, books["Offline"])
            assert entry["attempts"] == 1 and entry["last_error"] == "provider_error"
            p.offline = False
            await until(lambda: queued(app, books["Offline"]) is None, BACKOFF_WAIT, "the offline retry")
            await settle(app)
            expected["Offline/Offline.epub"] = EPUB_B
            assert_tree(phone, expected, "after the provider came back")
            assert doc_id(record(app, books["Offline"])) == docs["Offline"] and len(creates(phone.native)) == before

            # --- a document that throws SecurityException while it is still listed in its folder (locked): the
            # native side calls it 'missing', the listing shows it is there -> retried, never created again
            locked = phone.node("Locked/Locked.epub")
            p.move_out(locked)  # access denied, but the node stays in the folder
            recompile(books["Locked"], "Locked.epub", EPUB_B)
            job_done(app, books["Locked"])
            await settle(app)
            assert_tree(phone, expected, "while one document is locked")
            assert queued(app, books["Locked"])["attempts"] == 1 and len(creates(phone.native)) == before
            assert not app.cloud_sync.destination().needs_relink
            p.moved_out.discard(locked.doc_id)
            app.cloud_sync.retry_now()
            await settle(app)
            expected["Locked/Locked.epub"] = EPUB_B
            assert_tree(phone, expected, "after the lock went away")
            assert doc_id(record(app, books["Locked"])) == docs["Locked"] and len(creates(phone.native)) == before

            # --- the user deleted the cloud file: proven by the folder listing -> created again, once
            n_gone = len(creates(phone.native, "Gone.epub"))
            p.delete_document(phone.node("Gone/Gone.epub"))
            recompile(books["Gone"], "Gone.epub", EPUB_B)
            job_done(app, books["Gone"])
            await settle(app)
            expected["Gone/Gone.epub"] = EPUB_B
            assert_tree(phone, expected, "after the cloud file was deleted")
            assert len(creates(phone.native, "Gone.epub")) == n_gone + 1 and queued(app, books["Gone"]) is None

            # --- the whole book folder deleted: folder and file made again, once each
            n_dir, n_file = len(creates(phone.native, "FolderGone")), len(creates(phone.native, "FolderGone.epub"))
            phone.delete_tree(phone.node("FolderGone"))
            recompile(books["FolderGone"], "FolderGone.epub", EPUB_B)
            job_done(app, books["FolderGone"])
            await settle(app)
            expected["FolderGone/FolderGone.epub"] = EPUB_B
            assert_tree(phone, expected, "after the book folder was deleted")
            assert len(creates(phone.native, "FolderGone")) == n_dir + 1
            assert len(creates(phone.native, "FolderGone.epub")) == n_file + 1

            # --- a provider that loses every new file: at most one re-create per attempt, then backoff
            n_vanish = len(creates(phone.native, "Vanish.epub"))
            p.delete_document(phone.node("Vanish/Vanish.epub"))
            plain_create = p.create

            def create_and_lose(parent_uri, name, is_dir):
                uri = plain_create(parent_uri, name, is_dir)
                p.nodes[p.doc_id_of(uri)].deleted = True
                return uri

            p.create = create_and_lose
            recompile(books["Vanish"], "Vanish.epub", EPUB_B)
            job_done(app, books["Vanish"])
            await settle(app)
            assert len(creates(phone.native, "Vanish.epub")) == n_vanish + 1  # one re-create, no loop
            assert queued(app, books["Vanish"])["attempts"] == 1
            p.create = plain_create
            await until(lambda: queued(app, books["Vanish"]) is None, BACKOFF_WAIT, "the re-create retry")
            await settle(app)
            expected["Vanish/Vanish.epub"] = EPUB_B
            assert_tree(phone, expected, "after the provider kept its files again")
            assert not failure_notices(phone)
        finally:
            await shutdown(tf, app)

    run(scenario())


# ---------------------------------------------------------------------------------------------------
# 3. SecurityException on one document only
# ---------------------------------------------------------------------------------------------------


def test_permission_error_on_one_document_affects_only_that_book(phone):
    tf = _tf()

    async def scenario():
        app = await launch(tf)
        try:
            moved = make_book(app, "Moved", {"Moved.epub": EPUB_A})
            other = make_book(app, "Other", {"Other.epub": EPUB_A})
            await refresh_library(app)
            await link_folder(app, phone)
            other_doc = doc_id(record(app, other))
            moved_node = phone.move_out("Moved/Moved.epub")  # SecurityException for that document from now on
            for folder, name in ((moved, "Moved.epub"), (other, "Other.epub")):
                recompile(folder, name, EPUB_B)
                job_done(app, folder)
            await settle(app)
            dest = app.cloud_sync.destination()
            assert not dest.needs_relink  # the folder still answers: the destination stays linked
            assert not failure_notices(phone)
            # the book whose file left the folder gets one new file there; the moved file is left alone
            assert_tree(phone, {"Moved/Moved.epub": EPUB_B, "Other/Other.epub": EPUB_B}, "after a moved-out file")
            assert bytes(moved_node.data) == EPUB_A and not moved_node.deleted
            assert doc_id(record(app, other)) == other_doc
            assert app.cloud_sync.store.queue() == []
        finally:
            await shutdown(tf, app)

    run(scenario())


# ---------------------------------------------------------------------------------------------------
# 4. provider offline, then online: the queue drains
# ---------------------------------------------------------------------------------------------------


def test_offline_provider_queue_drains_when_it_is_back(phone):
    from glossarion_mobile.services.cloud_sync import FAIL_AFTER

    tf = _tf()
    p = phone.provider

    async def scenario():
        app = await launch(tf)
        try:
            q1 = make_book(app, "Q1", {"Q1.epub": EPUB_A})
            q2 = make_book(app, "Q2", {"Q2.epub": EPUB_A})
            await refresh_library(app)
            await link_folder(app, phone)
            docs = {q1: doc_id(record(app, q1)), q2: doc_id(record(app, q2))}
            # the provider goes offline; two recompiles and a new book arrive (its folder create fails too)
            p.offline = True
            p.fail_next_creates = 2  # Drive: "create failed, try again" for both tries of the new book's folder
            recompile(q1, "Q1.epub", EPUB_B)
            recompile(q2, "Q2.epub", EPUB_B)
            q3 = make_book(app, "Q3", {"Q3.epub": EPUB_A})
            for folder in (q1, q2, q3):
                job_done(app, folder)
            await settle(app)
            assert_tree(phone, {"Q1/Q1.epub": EPUB_A, "Q2/Q2.epub": EPUB_A}, "while offline")
            assert all(queued(app, f)["attempts"] == 1 for f in (q1, q2, q3))
            assert not failure_notices(phone)  # a short outage posts nothing
            # back online: the backoff timer drains the queue on its own
            p.offline = False
            await until(lambda: not app.cloud_sync.store.queue(), BACKOFF_WAIT, "the queue to drain")
            await settle(app)
            assert_tree(phone, {"Q1/Q1.epub": EPUB_B, "Q2/Q2.epub": EPUB_B, "Q3/Q3.epub": EPUB_A}, "after the outage")
            assert doc_id(record(app, q1)) == docs[q1] and doc_id(record(app, q2)) == docs[q2]
            assert len(creates(phone.native, "Q3")) == 2 and len(creates(phone.native, "Q3.epub")) == 1

            # a longer outage: the 2nd retry is an hour away ... "Retry now" after the provider is back
            p.offline = True
            recompile(q1, "Q1.epub", EPUB_C, later=10)
            job_done(app, q1)
            await settle(app)
            await until(lambda: (queued(app, q1) or {}).get("attempts") == 2, BACKOFF_WAIT, "the second attempt")
            for _ in range(FAIL_AFTER):  # the user keeps tapping Retry now while it is still offline
                app.cloud_sync.retry_now()
                await settle(app)
            assert len(failure_notices(phone)) == 1  # one "Couldn't save" notification, never one per attempt
            notice = failure_notices(phone)[0]
            assert notice["payload"].startswith("glossarion://app/") and "content://" not in json.dumps(notice)
            assert_tree(phone, {"Q1/Q1.epub": EPUB_B, "Q2/Q2.epub": EPUB_B, "Q3/Q3.epub": EPUB_A}, "long outage")
            p.offline = False
            app.cloud_sync.retry_now()
            await settle(app)
            assert queued(app, q1) is None
            assert_tree(phone, {"Q1/Q1.epub": EPUB_C, "Q2/Q2.epub": EPUB_B, "Q3/Q3.epub": EPUB_A}, "after Retry now")
            assert doc_id(record(app, q1)) == docs[q1]

            # Drive's eventually consistent create: it throws but the file appears -> adopted, not duplicated
            q4 = make_book(app, "Q4", {"Q4.epub": EPUB_A})
            p.fail_next_creates = 1
            p.create_appears_late = True
            job_done(app, q4)
            await settle(app)
            p.create_appears_late = False
            assert [k for k in phone.tree() if k.startswith("Q4")] == ["Q4/Q4.epub"]  # one folder, one file
            assert phone.tree()["Q4/Q4.epub"] == EPUB_A
            assert queued(app, q4) is None
        finally:
            await shutdown(tf, app)

    run(scenario())


# ---------------------------------------------------------------------------------------------------
# 5. the app killed mid-write
# ---------------------------------------------------------------------------------------------------


def test_app_killed_mid_write_is_rewritten_from_a_fresh_snapshot(phone):
    tf = _tf()
    p = phone.provider
    seen: dict = {}

    async def first_launch():
        app = await launch(tf)
        try:
            killed = make_book(app, "Killed", {"Killed.epub": EPUB_A})
            await refresh_library(app)
            await link_folder(app, phone)
            seen["doc"] = doc_id(record(app, killed))
            seen["folder"] = killed
            app.prefs.flush()
            recompile(killed, "Killed.epub", EPUB_B)
            kill = {"event": asyncio.Event(), "capture": lambda: disk_state(app)}
            phone.native.kill = kill
            job_done(app, killed)
            await asyncio.wait_for(kill["event"].wait(), 10)
            seen["kill"] = kill
            seen["cache"] = str(app.paths.cache)
        except BaseException:
            await shutdown(tf, app)
            raise
        await die(tf, app)

    run(first_launch())
    kill = seen["kill"]
    # the copy streamed from a private snapshot in the app cache, never from the workspace (critic #6)
    assert Path(kill["source"]).parent == Path(kill["state"]["snapshot_dir"])
    assert str(Path(seen["cache"])) in kill["source"] and kill["snapshot"] == EPUB_B
    assert kill["state"]["snapshots"], "the snapshot is on disk while the copy runs"
    on_disk = json.loads(next(v for k, v in kill["state"]["files"].items() if k.endswith("mobile_cloud.json")))
    statuses = [rec.get("status") for books in on_disk["records"].values() for book in books.values()
                for rec in book["files"].values()]
    assert statuses == ["writing"]  # flushed before the first byte left the phone
    assert_tree(phone, {"Killed/Killed.epub": EPUB_B[: len(EPUB_B) // 2]}, "at the instant of death (torn)")
    restore(kill["state"])  # the disk exactly as the process left it

    async def second_launch():
        app = await launch(tf)
        try:
            await settle(app)
            await until(lambda: queued(app, seen["folder"]) is None, 10, "the resumed save")
            await settle(app)
            assert_tree(phone, {"Killed/Killed.epub": EPUB_B}, "after the next launch")
            rec = record(app, seen["folder"])
            assert doc_id(rec) == seen["doc"] and rec["status"] == "ok" and not rec.get("dirty")
            assert creates(phone.native) == []  # rewritten in place, nothing new
            sources = [src for _doc, src in phone.native.write_log]
            assert sources and all(src != kill["source"] and Path(src).parent == Path(kill["state"]["snapshot_dir"])
                                   for src in sources)  # a fresh snapshot
            snap_dir = Path(app.cloud_sync._snapshot_dir())
            assert not snap_dir.is_dir() or not list(snap_dir.iterdir())  # the dead launch's leftover is gone

            # the very first write of a new book is killed: the torn file is adopted by name next time
            fresh = make_book(app, "Fresh", {"Fresh.epub": EPUB_A})
            assert (await facade(app).call("set_enabled", False))["ok"]
            await refresh_library(app)
            app.prefs.flush()
            kill2 = {"event": asyncio.Event(), "capture": lambda: disk_state(app)}
            phone.native.kill = kill2
            await facade(app).call("send_now", fresh)
            await asyncio.wait_for(kill2["event"].wait(), 10)
            seen["kill2"] = kill2
            seen["fresh"] = fresh
            await die(tf, app)
        except BaseException:
            await shutdown(tf, app)
            raise

    run(second_launch())
    restore(seen["kill2"]["state"])

    async def third_launch():
        app = await launch(tf)
        try:
            await until(lambda: queued(app, seen["fresh"]) is None, 10, "the resumed first save")
            await settle(app)
            assert_tree(phone, {"Killed/Killed.epub": EPUB_B, "Fresh/Fresh.epub": EPUB_A}, "after the third launch")
            assert creates(phone.native) == []  # the torn file and its folder adopted, nothing duplicated
            assert p.grants  # the folder grant survived the process deaths
        finally:
            await shutdown(tf, app)

    run(third_launch())


def test_app_killed_during_a_replace_leaves_no_orphan_copy(phone):
    """A non-truncating provider: the shorter recompile is written to a new document first (create-new +
    delete-old). Killed while that new copy is half written, the next launch must not leave the torn
    orphan next to the finished file."""
    tf = _tf()
    p = phone.provider
    seen: dict = {}

    async def first_launch():
        p.accepted_modes = {"w", "r"}
        p.truncates_w = False
        app = await launch(tf)
        try:
            swap = make_book(app, "Swap", {"Swap.epub": EPUB_B})
            await refresh_library(app)
            await link_folder(app, phone)
            assert_tree(phone, {"Swap/Swap.epub": EPUB_B}, "after the first save")
            app.prefs.flush()
            recompile(swap, "Swap.epub", EPUB_C)  # shorter: needs a replace
            kill = {"event": asyncio.Event(), "capture": lambda: disk_state(app)}
            native = phone.native
            plain_write = native.write_file

            async def write_then_kill_the_replacement(target, source, **kwargs):
                # the first call goes to the recorded document (the native side skips 'w' for a shorter file
                # and answers unsupported_mode + needs_replace); the next one, to another document, is the
                # replacement's: the process dies there
                target_id = str(native._parse(target)[2]).rsplit("/", 1)[-1]
                if not seen.get("armed") and doc_id(record(app, swap)) != target_id:
                    seen["armed"] = True
                    native.kill = kill
                return await plain_write(target, source, **kwargs)

            native.write_file = write_then_kill_the_replacement
            seen["swap"] = swap
            job_done(app, swap)
            await asyncio.wait_for(kill["event"].wait(), 10)
            seen["kill"] = kill
        except BaseException:
            await shutdown(tf, app)
            raise
        await die(tf, app)

    run(first_launch())
    # at the instant of death: the old copy and the half-written replacement (Drive allows the same name)
    assert_tree(phone, {"Swap/Swap.epub": EPUB_B, "Swap/Swap.epub#2": EPUB_C[: len(EPUB_C) // 2]}, "at death")
    restore(seen["kill"]["state"])

    async def second_launch():
        app = await launch(tf)
        try:
            await until(lambda: queued(app, seen["swap"]) is None, 10, "the resumed save")
            await settle(app)
            assert_tree(phone, {"Swap/Swap.epub": EPUB_C}, "after the next launch (no torn orphan left)")
        finally:
            await shutdown(tf, app)

    run(second_launch())


def test_replace_whose_old_copy_cannot_be_deleted_leaves_no_duplicate(phone):
    """create-new + delete-old where the delete fails once (Drive: try again): the old copy must not stay
    next to the new one for good."""
    tf = _tf()
    p = phone.provider

    async def scenario():
        p.accepted_modes = {"w", "r"}
        p.truncates_w = False
        app = await launch(tf)
        try:
            book = make_book(app, "Twice", {"Twice.epub": EPUB_B})
            await refresh_library(app)
            await link_folder(app, phone)
            phone.native.fail_deletes = 1
            recompile(book, "Twice.epub", EPUB_C)
            job_done(app, book)
            await settle(app)
            # whatever the app does next (retry the delete, ...), the cloud must end with one copy
            app.cloud_sync.retry_now()
            await settle(app)
            recompile(book, "Twice.epub", EPUB_D, later=10)
            job_done(app, book)
            await settle(app)
            assert_tree(phone, {"Twice/Twice.epub": EPUB_D}, "after a replace whose delete failed once")
        finally:
            await shutdown(tf, app)

    run(scenario())


# ---------------------------------------------------------------------------------------------------
# 6. same titles in a flat folder
# ---------------------------------------------------------------------------------------------------


def test_same_title_books_in_a_flat_folder_never_share_a_file(phone):
    tf = _tf()

    async def scenario():
        app = await launch(tf)
        try:
            a = make_book(app, "Book A", {"Novel.epub": EPUB_A})
            b = make_book(app, "Book B", {"Novel.epub": EPUB_B})
            await refresh_library(app)
            await link_folder(app, phone, enable=False)
            cloud = app.cloud_sync
            # a destination that went flat (set directly: how a provider gets there is the next test's subject)
            cloud._set_destination(dataclasses.replace(cloud.destination(), layout="flat"))
            assert (await facade(app).call("set_enabled", True))["ok"]
            await settle(app)
            tree = phone.tree()
            assert sorted(tree) == ["Novel (2).epub", "Novel.epub"], sorted(tree)
            names = {a: record(app, a)["name"], b: record(app, b)["name"]}
            assert tree[names[a]] == EPUB_A and tree[names[b]] == EPUB_B  # each book its own file
            recompile(b, "Novel.epub", EPUB_C)
            job_done(app, b)
            await settle(app)
            assert phone.tree() == {names[a]: EPUB_A, names[b]: EPUB_C}
            recompile(a, "Novel.epub", EPUB_D)
            job_done(app, a)
            await settle(app)
            assert phone.tree() == {names[a]: EPUB_D, names[b]: EPUB_C}
            # the records are lost (mobile_cloud.json gone): both files adopted by name, no third file
            await cloud._io(cloud.store.delete_file)
            assert await cloud.sync_all("records lost") == 2
            await settle(app)
            tree = phone.tree()
            assert sorted(tree) == ["Novel (2).epub", "Novel.epub"], sorted(tree)
            assert sorted(tree.values()) == sorted([EPUB_D, EPUB_C])
            assert {record(app, a)["name"], record(app, b)["name"]} == {"Novel.epub", "Novel (2).epub"}
            assert tree[record(app, a)["name"]] == EPUB_D and tree[record(app, b)["name"]] == EPUB_C
        finally:
            await shutdown(tf, app)

    run(scenario())


def test_provider_that_cannot_create_folders_goes_flat(phone):
    """AOSP's default ``DocumentsProvider.createDocument`` throws UnsupportedOperationException("Create not
    supported"); a provider that takes files but refuses sub-folders answers like that for MIME_TYPE_DIR. The
    two same-title books must then land side by side in the picked folder (flat layout), never waiting forever."""
    tf = _tf()
    p = phone.provider
    plain_create = p.create

    def create_files_only(parent_uri, name, is_dir):
        if is_dir:
            raise phone.fake.ProviderFault("unsupported", "Create not supported")
        return plain_create(parent_uri, name, is_dir)

    p.create = create_files_only

    async def scenario():
        app = await launch(tf)
        try:
            make_book(app, "Book A", {"Novel.epub": EPUB_A})
            make_book(app, "Book B", {"Novel.epub": EPUB_B})
            await refresh_library(app)
            await link_folder(app, phone)
            waiting = [(os.path.basename(e["identity"]), e["last_error"], e["attempts"])
                       for e in app.cloud_sync.store.queue()]
            assert app.cloud_sync.destination().layout == "flat", \
                f"the destination never went flat; the books wait in the queue: {waiting}"
            tree = phone.tree()
            assert sorted(tree) == ["Novel (2).epub", "Novel.epub"]
            assert sorted(tree.values()) == sorted([EPUB_A, EPUB_B])
        finally:
            await shutdown(tf, app)

    run(scenario())


# ---------------------------------------------------------------------------------------------------
# 7. a deleted book
# ---------------------------------------------------------------------------------------------------


async def delete_book(app, folder):
    """The Library's delete (Book page / card ⋯ › Delete): plan + execute on the io pool, like the UI."""
    from glossarion_mobile.services.library import book_identity

    await refresh_library(app)
    row = next(b for b in app.library.snapshot.all_books()
               if os.path.normcase(book_identity(b)) == os.path.normcase(folder))
    plan = await app.library.io(app.library.plan_delete_blocking, [row])
    report = await app.library.io(app.library.execute_delete_blocking, plan)
    assert not os.path.exists(folder), report


def test_deleted_book_drops_its_records_and_queue_and_keeps_cloud_files(phone):
    from glossarion_mobile.state.cloud_records import book_key

    tf = _tf()
    p = phone.provider

    async def scenario():
        app = await launch(tf)
        try:
            doomed = make_book(app, "Doomed", {"Doomed.epub": EPUB_A})
            keeper = make_book(app, "Keeper", {"Keeper.epub": EPUB_A})
            await refresh_library(app)
            await link_folder(app, phone)
            cloud = app.cloud_sync
            dest_id = cloud.destination().id
            cloud.set_book_override(doomed, "always")
            await settle(app)
            # the book is waiting for a retry (provider offline) when the user deletes it in the Library
            p.offline = True
            recompile(doomed, "Doomed.epub", EPUB_B)
            job_done(app, doomed)
            await settle(app)
            assert queued(app, doomed)["attempts"] == 1
            await delete_book(app, doomed)
            await settle(app)
            assert queued(app, doomed) is None and cloud.store.book(dest_id, book_key(doomed)) is None
            assert cloud.store.override(doomed) == "default"
            p.offline = False
            writes = len(phone.native.write_log)
            await asyncio.sleep(2.5)  # past the first backoff: nothing wakes up for the deleted book
            app.cloud_sync.retry_now()
            await settle(app)
            assert len(phone.native.write_log) == writes
            # its cloud file stays (never deleted from the cloud); the other book is untouched
            assert_tree(phone, {"Doomed/Doomed.epub": EPUB_A, "Keeper/Keeper.epub": EPUB_A}, "after the delete")
            assert record(app, keeper)["status"] == "ok"
            assert not failure_notices(phone)
        finally:
            await shutdown(tf, app)

    run(scenario())


def test_book_deleted_while_its_file_is_copied_leaves_no_record(phone):
    from glossarion_mobile.state.cloud_records import book_key

    tf = _tf()

    async def scenario():
        app = await launch(tf)
        try:
            midway = make_book(app, "Midway", {"Midway.epub": EPUB_A})
            await refresh_library(app)
            await link_folder(app, phone)
            cloud = app.cloud_sync
            dest_id = cloud.destination().id
            writes = len(phone.native.write_log)
            phone.native.gate = asyncio.Event()
            recompile(midway, "Midway.epub", EPUB_B)
            job_done(app, midway)
            await until(lambda: len(phone.native.write_log) > writes, 10, "the copy to start")
            await delete_book(app, midway)
            phone.native.gate.set()
            phone.native.gate = None
            await settle(app)
            assert queued(app, midway) is None
            assert [k for k in phone.tree()] == ["Midway/Midway.epub"]  # one cloud file, kept
            assert cloud.store.book(dest_id, book_key(midway)) is None, \
                "the copy that finished after the delete stored the deleted book's record again"
        finally:
            await shutdown(tf, app)

    run(scenario())


# ---------------------------------------------------------------------------------------------------
# 8. Wipe app data
# ---------------------------------------------------------------------------------------------------


def test_wipe_app_data_releases_every_grant_and_keeps_cloud_files(phone):
    tf = _tf()
    p = phone.provider

    async def scenario():
        app = await launch(tf)
        try:
            make_book(app, "Wiped", {"Wiped.epub": EPUB_A})
            await refresh_library(app)
            await link_folder(app, phone)
            # a per-file grant an older build left behind (they live in the system, not in app files)
            orphan = p.add_file("Old single file.epub", EPUB_C, parent=phone.elsewhere)
            p.grants[p.doc_uri(orphan)] = {"read": True, "write": True, "persisted_time": 99}
            assert phone.grant_count() == 2
            before = phone.tree()
            await app.navigate("/settings/danger")
            screen = app.shell.top_screen
            assert type(screen).__name__ == "DangerZoneScreen"
            exits: list = []
            screen.exit_app = lambda: exits.append(True)  # the real one ends the process
            try:
                failed = await screen.wipe()
                error = None
            except Exception as exc:  # reported below, after the U10 part is checked
                failed, error = None, exc
            assert phone.grant_count() == 0  # every persisted grant given back first
            assert phone.tree() == before and bytes(orphan.data) == EPUB_C  # cloud files are never deleted
            assert not (Path(app.paths.data) / "mobile_cloud.json").exists()
            assert app.cloud_sync.destination() is None and app.cloud_sync.store.queue() == []
            # ... and the wipe itself then deletes the app's files and closes the app
            assert error is None, f"Wipe app data failed after the grants were released: {error!r}"
            assert exits and not failed
            assert not (Path(app.paths.data) / "mobile_state.json").exists()
        finally:
            await shutdown(tf, app)

    run(scenario())
