"""Host tests for U10 cloud sync (``services/cloud_sync.py`` + ``state/cloud_records.py``).

Run from src/mobile with the mobile venv:
    .venv/Scripts/python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic \
        tests_host/test_cloud_sync.py

The native document API is an in-memory fake with the ``flet_glossarion_native`` shapes (refs are
JSON dicts, ``write_file`` guards non-truncating modes and reads the length back, ``missing`` is only
reported when proven), reached through the real ``NativeDocs`` adapter. The Library is a real
``LibraryService`` over the real ``library_core.list_compiled_outputs`` (a stand-in with the same rule
when the shared module is missing). Everything lives under ``tmp_path``; nothing touches the network.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = MOBILE_DIR / "app"
SRC_DIR = MOBILE_DIR.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from glossarion_mobile.services import cloud_sync as cs  # noqa: E402
from glossarion_mobile.services.cloud_sync import (  # noqa: E402
    BACKOFF,
    BACKOFF_MAX,
    CLOUD_HOLD,
    FAIL_AFTER,
    CloudNotifier,
    CloudSyncService,
    Keepalive,
    NativeDocs,
    choose_outputs,
    collect_outputs,
    output_kind,
)
from glossarion_mobile.services.jobs import JobSnapshot, JobSpec, JobState  # noqa: E402
from glossarion_mobile.services.library import LibraryService, ScanSnapshot, SharedCore  # noqa: E402
from glossarion_mobile.services.native import ServiceHolds  # noqa: E402
from glossarion_mobile.state.cloud_records import CloudRecordStore, book_key  # noqa: E402

TREE = "content://com.fake.docs/tree/root"


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    """Real data never: Library / output / home / app data all point into tmp_path."""
    for name, sub in (("GLOSSARION_LIBRARY_DIR", "Library"), ("OUTPUT_DIRECTORY", "Output"), ("HOME", "home"),
                      ("USERPROFILE", "home"), ("APPDATA", "appdata"), ("GLOSSARION_DATA_DIR", "data")):
        path = tmp_path / sub
        path.mkdir(exist_ok=True)
        monkeypatch.setenv(name, str(path))
    monkeypatch.setenv("GLOSSARION_HTTP_LOG", "0")


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Cloud sync makes no network request: a connect to anything but loopback fails the test (the
    Windows event loop's self-pipe is a loopback socket pair)."""
    original, original_ex = socket.socket.connect, socket.socket.connect_ex

    def guard(fn):
        def connect(self, address, *args, **kwargs):
            host = address[0] if isinstance(address, tuple) and address else address
            if host not in ("127.0.0.1", "::1", "localhost"):
                raise AssertionError(f"network access attempted: {address!r}")
            return fn(self, address, *args, **kwargs)

        return connect

    monkeypatch.setattr(socket.socket, "connect", guard(original))
    monkeypatch.setattr(socket.socket, "connect_ex", guard(original_ex))


# ---------------------------------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------------------------------


class FakeCloud:
    """The ``GlossarionNative`` document API over an in-memory provider (one picked tree + picked files)."""

    is_stub = False

    def __init__(self, *, modes=("wt", "rwt", "w"), truncating=True, can_create_dirs=True, label="FakeDrive",
                 provider="com.fake.docs", can_rename=True) -> None:
        if not can_rename:
            self.rename_document = None  # an extension build without rename
        self.nodes = {"root": {"kind": "folder", "name": "Glossarion", "parent": None, "data": b""}}
        self.next_id = 1
        self.modes = set(modes)
        self.truncating = truncating  # False: every mode keeps the old tail (OneDrive 'w' bug, worst case)
        self.can_create_dirs = can_create_dirs
        self.label = label
        self.provider = provider
        self.offline = False
        self.offline_says_missing = False  # FileNotFoundException while offline (Nextcloud)
        self.revoked = False
        self.denied: set = set()
        self.fail_writes = 0
        self.calls: list = []
        self.writes: list = []
        self.creates: list = []
        self.deletes: list = []
        self.releases: list = []
        self.grants: set = set()
        self.pick_queue: list = []
        self.late: list = []
        self.downloads: dict = {}
        self.gate: asyncio.Event = None  # blocks writes until set
        self.cancelled: set = set()
        self.own_folder = False

    # ---- helpers ----
    def ref(self, nid):
        node = self.nodes[nid]
        in_tree = self._in_tree(nid)
        return {"platform": "android", "kind": node["kind"], "id": f"d{nid}", "uri": TREE if in_tree else None,
                "document": f"content://{self.provider}/document/{nid}", "name": node["name"],
                "size": len(node["data"]), "provider": self.provider, "provider_label": self.label,
                "can_write": True, "can_create": node["kind"] == "folder", "persisted": True}

    def _in_tree(self, nid):
        while nid is not None:
            if nid == "root":
                return True
            nid = self.nodes.get(nid, {}).get("parent")
        return False

    def nid(self, ref):
        if isinstance(ref, dict):
            doc = str(ref.get("document") or "")
        else:
            doc = str(ref or "")
        if doc.endswith("/tree/root") or doc == TREE:
            return "root"
        return doc.rsplit("/", 1)[-1] if "/document/" in doc else None

    def add(self, parent, name, kind="file", data=b""):
        nid = f"n{self.next_id}"
        self.next_id += 1
        self.nodes[nid] = {"kind": kind, "name": name, "parent": parent, "data": data}
        return nid

    def children(self, parent):
        return [nid for nid, node in self.nodes.items() if node["parent"] == parent]

    def files(self, parent=None):
        """{name: bytes} of the files under ``parent`` (every tree file when None: 'Folder/name')."""
        out = {}
        for nid, node in self.nodes.items():
            if node["kind"] != "file":
                continue
            if parent is not None:
                if node["parent"] == parent:
                    out[node["name"]] = node["data"]
                continue
            if not self._in_tree(nid):
                continue
            path = node["name"]
            p = node["parent"]
            while p not in (None, "root"):
                path = self.nodes[p]["name"] + "/" + path
                p = self.nodes[p]["parent"]
            if parent is None or node["parent"] == parent:
                out[path] = node["data"]
        return out

    def tree_files(self):
        return self.files()

    def folder_named(self, name):
        return next((nid for nid, n in self.nodes.items() if n["kind"] == "folder" and n["name"] == name
                     and n["parent"] == "root"), None)

    def err(self, code, scope="document", **extra):
        out = {"ok": False, "error": code, "message": code, "scope": scope}
        out.update(extra)
        return out

    def _gate_check(self):
        if self.revoked:
            return self.err("permission_lost", "target")
        if self.offline:
            return self.err("provider_error", "target")
        return None

    # ---- GlossarionNative document API ----
    async def get_platform_info(self):
        return {"platform": "android", "native": True, "documents": True, "save_to_downloads": True, "sdk_int": 34,
                "persisted_grant_limit": 512}

    async def pick_folder(self, *, initial=None, op_id=None):
        self.calls.append(("pick_folder", op_id))
        if self.pick_queue:
            return self.pick_queue.pop(0)
        self.grants.add(TREE)
        target = self.ref("root")
        target.update(uri=TREE, document=TREE, own_folder=self.own_folder)
        return {"ok": True, "error": None, "target": target, "persisted": True}

    async def pick_save_location(self, name, mime_type, source_path=None, *, initial=None, mode_chain=None,
                                 op_id=None):
        self.calls.append(("pick_save_location", name, op_id))
        if self.pick_queue:
            return self.pick_queue.pop(0)
        data = open(source_path, "rb").read() if source_path else b""
        nid = self.add("picked", name, data=data)
        self.grants.add(f"content://{self.provider}/document/{nid}")
        write = {"ok": True, "mode": "wt", "written": len(data), "verified_size": len(data)} if source_path else None
        return {"ok": True, "error": None, "document": self.ref(nid), "write": write}

    async def pick_document(self, mime_types=None, *, initial=None, op_id=None):
        return self.pick_queue.pop(0) if self.pick_queue else {"ok": False, "error": "cancelled"}

    async def list_children(self, folder, *, names=None):
        self.calls.append(("list_children", self.nid(folder), tuple(names or ())))
        problem = self._gate_check()
        if problem:
            return problem
        nid = self.nid(folder)
        if nid not in self.nodes:
            return self.err("missing", "document", proven=True)
        kids = [self.ref(c) for c in self.children(nid)]
        if names is not None:
            kids = [k for k in kids if k["name"] in names]
        return {"ok": True, "error": None, "children": kids, "complete": True}

    def _create(self, folder, name, kind, on_exists):
        problem = self._gate_check()
        if problem:
            return problem
        parent = self.nid(folder)
        if parent not in self.nodes:
            return self.err("missing", "document", proven=True)
        same = [c for c in self.children(parent) if self.nodes[c]["name"] == name]
        if same and on_exists == "fail":
            return self.err("exists", document=self.ref(same[0]))
        if same and on_exists == "adopt":
            return {"ok": True, "document": self.ref(same[0]), "created": False, "adopted": True}
        final, n = name, 1
        while any(self.nodes[c]["name"] == final for c in self.children(parent)):
            stem, ext = os.path.splitext(name)
            final = f"{stem} ({n}){ext}"
            n += 1
        nid = self.add(parent, final, kind)
        self.creates.append((kind, final))
        return {"ok": True, "error": None, "document": self.ref(nid), "created": True, "adopted": False}

    async def create_file(self, folder, name, mime_type, *, on_exists="rename"):
        self.calls.append(("create_file", name))
        return self._create(folder, name, "file", on_exists)

    async def create_folder(self, parent, name, *, on_exists="adopt"):
        self.calls.append(("create_folder", name))
        if not self.can_create_dirs:
            return self.err("read_only")
        made = self._create(parent, name, "folder", on_exists)
        if made.get("ok"):
            made["folder"] = made.pop("document")
        return made

    async def write_file(self, target_or_doc, source_path, *, name=None, mime_type=None, on_exists="rename",
                         mode_chain=("wt", "rwt", "w"), verify=True, op_id=None, timeout=None):
        self.calls.append(("write_file", self.nid(target_or_doc), op_id))
        if self.revoked:
            return self.err("permission_lost", "target")
        nid = self.nid(target_or_doc)
        if self.offline:
            # Drive / Nextcloud throw FileNotFoundException while offline; the native side only reports
            # 'missing' when the root query proves it, so this arrives as provider_error (or, from an older
            # build, as unproven 'missing').
            return self.err("missing", proven=False) if self.offline_says_missing else self.err("provider_error")
        if nid not in self.nodes:
            in_tree = isinstance(target_or_doc, dict) and target_or_doc.get("uri")
            return self.err("missing", proven=bool(in_tree))
        if nid in self.denied:
            return self.err("permission_lost", "document")
        if self.fail_writes:
            self.fail_writes -= 1
            return self.err("provider_error")
        if self.gate is not None:
            await self.gate.wait()
        if op_id in self.cancelled:
            return self.err("cancelled")
        data = open(source_path, "rb").read()
        node = self.nodes[nid]
        old = node["data"]
        for mode in mode_chain:
            if mode not in self.modes:
                continue
            if mode in ("w", "rw") and len(data) < len(old):
                continue  # the native guard: never a non-truncating mode over a longer cloud copy
            node["data"] = data if self.truncating else data + old[len(data):]
            self.writes.append((nid, mode, source_path))
            verified = len(node["data"])
            if verify and verified != len(data):
                return self.err("size_mismatch", stale_tail=True, needs_replace=True, verified_size=verified)
            return {"ok": True, "error": None, "document": self.ref(nid), "created": False, "mode": mode,
                    "written": len(data), "verified_size": verified, "op_id": op_id}
        return self.err("unsupported_mode", needs_replace=True)

    async def rename_document(self, document, name):
        nid = self.nid(document)
        if nid not in self.nodes:
            return self.err("missing")
        if any(self.nodes[c]["name"] == name for c in self.children(self.nodes[nid]["parent"]) if c != nid):
            return self.err("exists")
        self.nodes[nid]["name"] = name
        return {"ok": True, "document": self.ref(nid)}

    async def stat(self, document):
        nid = self.nid(document)
        if nid not in self.nodes:
            return self.err("missing", proven=bool(isinstance(document, dict) and document.get("uri")))
        return {"ok": True, "document": self.ref(nid)}

    async def delete(self, document):
        nid = self.nid(document)
        self.deletes.append(nid)
        if nid not in self.nodes:
            return self.err("missing")
        del self.nodes[nid]
        return {"ok": True}

    async def query_root(self, target):
        problem = self._gate_check()
        if problem:
            return problem
        if "root" not in self.nodes:
            return self.err("missing", "target")
        ref = self.ref("root")
        ref.update(uri=TREE, document=TREE)
        return {"ok": True, "target": ref}

    async def release(self, target):
        uri = target if isinstance(target, str) else (target.get("uri") or target.get("document"))
        self.releases.append(uri)
        self.grants.discard(uri)
        return True

    async def list_grants(self):
        return [{"uri": g, "read": True, "write": True} for g in sorted(self.grants)]

    async def cancel_document_op(self, op_id):
        self.cancelled.add(op_id)
        if self.gate is not None:
            self.gate.set()
        return True

    async def take_document_results(self):
        out, self.late = self.late, []
        return out

    async def save_to_downloads(self, path, display_name, mime_type, subdir="Glossarion", replace_uri=None):
        data = open(path, "rb").read()
        self.calls.append(("save_to_downloads", display_name, subdir, replace_uri))
        if replace_uri and replace_uri in self.downloads:
            self.downloads[replace_uri]["data"] = data  # in place: the name never changes
            return replace_uri
        uri = f"content://media/external/downloads/{len(self.downloads) + 1}"
        self.downloads[uri] = {"name": display_name, "subdir": subdir, "data": data}
        return uri


class FakeBridge:
    """``NativeBridge`` stand-in: ServiceHolds + ``call`` for the job service / background task methods."""

    def __init__(self, cloud: FakeCloud) -> None:
        self.native = cloud
        self.service_holds = ServiceHolds()
        self.running = False
        self.remaining = None
        self.log: list = []
        self.notifications: list = []
        self.listeners: dict = {}

    def add_listener(self, event, callback):
        self.listeners.setdefault(event, []).append(callback)

    async def call(self, method, *args, default=None, **kwargs):
        self.log.append((method, args, kwargs))
        if method == "is_job_service_running":
            return self.running
        if method == "start_job_service":
            self.running = True
            return True
        if method == "stop_job_service":
            self.running = False
            return None
        if method == "begin_background_task":
            return 7
        if method == "background_time_remaining":
            return self.remaining
        return default

    async def background_time_remaining(self):
        return self.remaining

    # NativeBridge's job-service methods (``native.ServiceLease`` calls them), logged through ``call``
    async def is_job_service_running(self):
        return bool(await self.call("is_job_service_running", default=False))

    async def start_job_service(self, title, text):
        return bool(await self.call("start_job_service", title, text, default=False))

    async def update_job_service(self, text=None, title=None):
        await self.call("update_job_service", title=title, text=text)

    async def stop_job_service(self):
        await self.call("stop_job_service")

    async def show_notification(self, nid, title, body, *, channel_id="jobs.action", payload=None):
        self.notifications.append({"id": nid, "title": title, "body": body, "payload": payload})
        return True

    async def cancel_notification(self, nid):
        self.log.append(("cancel_notification", (nid,), {}))

    def methods(self):
        return [m for m, _a, _k in self.log]


class FakeJobs:
    def __init__(self) -> None:
        self.active = None
        self.queue: list = []
        self.listeners: list = []

    def view(self):
        return SimpleNamespace(active=self.active, queue=tuple(self.queue), paused=False)

    def on_transition(self, callback):
        self.listeners.append(callback)
        return lambda: self.listeners.remove(callback)

    def snapshot(self, job_id=None):
        return self.active

    def deliver(self, snap, previous):
        for callback in list(self.listeners):
            callback(snap, previous)


class FakePrefs:
    def __init__(self, data=None) -> None:
        self.data = dict(data or {})

    def get(self, key, default=None):
        return json.loads(json.dumps(self.data[key])) if key in self.data else default

    def set(self, key, value):
        self.data[key] = json.loads(json.dumps(value))


class Clock:
    def __init__(self, now=1_000_000.0) -> None:
        self.now = now

    def __call__(self):
        return self.now


def _library_core():
    try:
        import library_core

        if hasattr(library_core, "list_compiled_outputs"):
            return library_core
    except Exception:
        pass
    module = types.ModuleType("library_core")

    def list_compiled_outputs(folder):  # same rule as library_core._list_compiled_outputs
        out = []
        names = sorted(os.listdir(folder))
        out += [(os.path.join(folder, n), "epub") for n in names if n.lower().endswith(".epub")]
        stems = {n.lower()[:-len("_translated.pdf")] for n in names if n.lower().endswith("_translated.pdf")}
        for suffix, kind in (("_translated.pdf", "pdf"), ("_translated.txt", "txt"), ("_translated.html", "html")):
            for n in names:
                if n.lower().endswith(suffix) and not (kind == "html" and n.lower()[:-len(suffix)] in stems):
                    out.append((os.path.join(folder, n), kind))
        return out

    module.list_compiled_outputs = list_compiled_outputs
    return module


class Env:
    def __init__(self, tmp: Path, *, platform="android", cloud: FakeCloud = None, prefs=None, clock=None) -> None:
        self.tmp = tmp
        self.output = tmp / "Output"
        self.library_dir = tmp / "Library"
        self.cache = tmp / "cache"
        self.data = tmp / "data"
        for d in (self.output, self.library_dir, self.cache, self.data):
            d.mkdir(parents=True, exist_ok=True)
        self.cloud = cloud or FakeCloud()
        self.bridge = FakeBridge(self.cloud)
        self.jobs = FakeJobs()
        self.prefs = prefs if prefs is not None else FakePrefs()
        self.clock = clock or Clock()
        paths = SimpleNamespace(output=str(self.output), library=str(self.library_dir), cache=str(self.cache))
        self.library = LibraryService(paths=paths, core=SharedCore({"library_core": _library_core()}), config={})
        self.books: list = []
        self.platform = platform
        self.background = SimpleNamespace(released=0)

        def release_kept_background():
            self.background.released += 1

        self.background.release_kept_background = release_kept_background
        self.store = CloudRecordStore(self.data / "mobile_cloud.json", clock=self.clock, save_delay=0.0)
        self.service = self.make_service(self.store)

    def make_service(self, store):
        service = CloudSyncService(
            docs=NativeDocs(self.bridge, platform=self.platform), store=store, prefs=self.prefs,
            library=self.library, jobs=self.jobs,
            keepalive=Keepalive(self.bridge, platform=self.platform, jobs=self.jobs, background=self.background),
            notifier=CloudNotifier(None, self.bridge), platform=self.platform, cache_dir=str(self.cache),
            app_roots=[str(self.data), str(self.output), str(self.library_dir), str(self.cache)],
            clock=self.clock)
        service.attach(jobs=self.jobs, native=self.bridge)
        return service

    def book(self, name, files, *, root=None, mtime_step=0):
        folder = (root or self.output) / name
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "translation_progress.json").write_text("{}", encoding="utf-8")
        for i, (fname, data) in enumerate(files.items()):
            path = folder / fname
            path.write_bytes(data)
            if mtime_step:
                t = time.time() - 1000 + i * mtime_step
                os.utime(path, (t, t))
        row = {"name": name, "folder_name": name, "path": str(folder), "output_folder": str(folder),
               "type": "in_progress", "is_in_progress": True}
        if all(r["output_folder"] != row["output_folder"] for r in self.books):
            self.books.append(row)
        self.library.set_snapshot(ScanSnapshot(in_progress=tuple(self.books)))
        return str(folder)

    def recompile(self, folder, name, data, *, later=10):
        path = Path(folder) / name
        path.write_bytes(data)
        t = time.time() + later
        os.utime(path, (t, t))
        return str(path)

    def done(self, folder, *, kind="compile_epub", jid="j1", outputs=(), origin=None, state=JobState.DONE,
             previous=JobState.RUNNING):
        spec = JobSpec(kind=kind, title=os.path.basename(folder), inputs=(folder,), params={"folder": folder},
                       origin=origin or {"type": "library", "bid": "x"})
        snap = JobSnapshot(id=jid, spec=spec, state=state, created=0.0, outputs=tuple(outputs), output_dir=folder)
        self.jobs.deliver(snap, previous)
        return snap

    async def link(self, enable=True):
        result = await self.service.pick_folder()
        assert result["ok"], result
        if enable:
            self.service.set_enabled(True)
        await self.settle()
        return result

    async def settle(self):
        for _ in range(5):
            await asyncio.sleep(0)
            await self.service.wait_idle()
        await asyncio.sleep(0)


def run(coro):
    return asyncio.run(coro)


def cloud_names(cloud, folder_name):
    nid = cloud.folder_named(folder_name)
    return sorted(cloud.nodes[c]["name"] for c in cloud.children(nid)) if nid else []


EPUB1 = b"PK-epub-version-1" * 50
EPUB2 = b"PK-epub-version-2-longer" * 50
PDF1 = b"%PDF-1" * 40


# ---------------------------------------------------------------------------------------------------
# source selection (critic #0 / #1)
# ---------------------------------------------------------------------------------------------------


def test_epub_derived_pdf_is_a_candidate_and_companion_html_is_not(tmp_path):
    env = Env(tmp_path)
    folder = env.book("BookA", {"New Title.epub": EPUB1, "New Title.pdf": PDF1, "BookA_translated.txt": b"t",
                                "Other_translated.pdf": PDF1, "Other_translated.html": b"<p>debug</p>"})
    lister = env.library.compiled_outputs_blocking
    listed = {os.path.basename(p) for p, _k in lister({"output_folder": folder, "path": folder})}
    found = collect_outputs({"output_folder": folder, "path": folder}, lister=lister)
    assert "New Title.pdf" in {os.path.basename(p) for p in found["pdf"]}
    if "New Title.pdf" not in listed:  # the shared list misses the EPUB-derived PDF: the union adds it
        assert "Other_translated.pdf" in listed
    assert "html" not in found  # the PDF's debug companion is never an output
    assert output_kind(os.path.join(folder, "Other_translated.html")) == ""
    assert output_kind(os.path.join(folder, "BookA_translated.txt")) == "txt"


def test_the_untranslated_source_is_never_an_output(tmp_path):
    env = Env(tmp_path)
    raw = tmp_path / "Inbox" / "Novel.epub"
    raw.parent.mkdir()
    raw.write_bytes(b"raw untranslated")
    folder = env.book("Novel", {"Novel_translated.txt": b"translated"})
    row = {"name": "Novel", "path": str(raw), "output_folder": folder, "is_in_progress": True,
           "raw_source_path": str(raw)}  # a desktop-style in-progress row: path = the source EPUB
    found = collect_outputs(row, lister=env.library.compiled_outputs_blocking)
    assert "epub" not in found and list(found) == ["txt"]
    filed = env.library_dir / "Translated" / "Novel.epub"  # an organized translation stays an output
    filed.parent.mkdir(parents=True)
    filed.write_bytes(b"translated epub")
    row = {"name": "Novel", "path": str(filed), "output_folder": folder, "in_library": True}
    assert collect_outputs(row, lister=env.library.compiled_outputs_blocking)["epub"] == [str(filed)]


def test_choice_is_deterministic_newest_or_reported(tmp_path):
    env = Env(tmp_path)
    folder = env.book("BookB", {"Old Title.epub": EPUB1, "New Title.epub": EPUB2}, mtime_step=100)
    old, new = os.path.join(folder, "Old Title.epub"), os.path.join(folder, "New Title.epub")
    found = collect_outputs({"output_folder": folder}, lister=env.library.compiled_outputs_blocking)
    for _ in range(5):
        assert choose_outputs(found)["epub"].path == new  # newest wins whatever the listing order
    assert choose_outputs(found)["epub"].conflicts == 1
    assert choose_outputs(found, reported=[old])["epub"].path == old  # the job's report wins


# ---------------------------------------------------------------------------------------------------
# folder mode: enable, recompile, title change, unchanged outputs
# ---------------------------------------------------------------------------------------------------


def test_enable_uploads_each_format_once_and_recompile_overwrites_in_place(tmp_path):
    async def scenario():
        env = Env(tmp_path)
        folder = env.book("BookA", {"BookA.epub": EPUB1, "BookA.pdf": PDF1})
        await env.link(enable=True)
        assert cloud_names(env.cloud, "BookA") == ["BookA.epub", "BookA.pdf"]  # the Library's per-book layout
        assert len(env.cloud.writes) == 2
        epub_doc = env.store.record(env.service.destination().id, book_key(folder), "epub")["doc"]
        # nothing changed: a resume writes nothing
        await env.service.drain_now("resume")
        env.service.sync_all  # noqa: B018
        assert len(env.cloud.writes) == 2
        # recompile -> the same document is overwritten (no second file)
        env.recompile(folder, "BookA.epub", EPUB2)
        env.done(folder, outputs=[os.path.join(folder, "BookA.epub")])
        await env.settle()
        assert cloud_names(env.cloud, "BookA") == ["BookA.epub", "BookA.pdf"]
        nid = env.cloud.nid(epub_doc)
        assert env.cloud.nodes[nid]["data"] == EPUB2
        assert len([c for c in env.cloud.creates if c[0] == "file"]) == 2
        # every write streamed from a private snapshot in the cache, never the workspace file
        assert all(str(env.cache) in src for _n, _m, src in env.cloud.writes)
        assert not os.listdir(env.cache / "cloud_sync")  # snapshots removed

    run(scenario())


def test_title_change_keeps_cloud_name_and_never_uploads_the_stale_epub(tmp_path):
    async def scenario():
        env = Env(tmp_path)
        folder = env.book("BookC", {"Old Title.epub": EPUB1})
        await env.link()
        assert cloud_names(env.cloud, "BookC") == ["Old Title.epub"]
        new = env.recompile(folder, "New Title.epub", EPUB2)  # the compiler leaves the old-title EPUB behind
        env.done(folder, outputs=[new, os.path.join(folder, "Old Title.epub")])
        await env.settle()
        names = cloud_names(env.cloud, "BookC")
        assert names == ["Old Title.epub"]  # same cloud file, its first name kept
        nid = env.cloud.folder_named("BookC")
        assert env.cloud.files(nid)["Old Title.epub"] == EPUB2  # ... with the new book in it

    run(scenario())


def test_unchanged_outputs_listed_again_are_not_rewritten(tmp_path):
    async def scenario():
        env = Env(tmp_path)
        folder = env.book("BookD", {"BookD.epub": EPUB1})
        await env.link()
        writes = len(env.cloud.writes)
        env.done(folder, kind="translate", jid="t1", outputs=[os.path.join(folder, "BookD.epub")])
        await env.settle()
        assert len(env.cloud.writes) == writes
        # mtime changes, content does not: the stored stat is refreshed, nothing is written
        path = os.path.join(folder, "BookD.epub")
        os.utime(path, (time.time() + 50, time.time() + 50))
        env.done(folder, kind="metadata", jid="m1")
        await env.settle()
        assert len(env.cloud.writes) == writes
        record = env.store.record(env.service.destination().id, book_key(folder), "epub")
        assert record["mtime_ns"] == os.stat(path).st_mtime_ns

    run(scenario())


# ---------------------------------------------------------------------------------------------------
# opt-in model
# ---------------------------------------------------------------------------------------------------


def test_override_never_always_formats_and_send_now(tmp_path):
    async def scenario():
        env = Env(tmp_path)
        a = env.book("Always", {"Always.epub": EPUB1, "Always.pdf": PDF1})
        n = env.book("Never", {"Never.epub": EPUB1})
        await env.link(enable=False)
        assert env.cloud.writes == []  # off by default: linking writes nothing
        env.service.set_book_override(a, "always")
        env.service.set_book_override(n, "never")
        env.service.set_kind_enabled("pdf", False)
        await env.settle()
        assert cloud_names(env.cloud, "Always") == ["Always.epub"]  # global off, Always on, PDF off
        env.service.set_enabled(True)
        await env.settle()
        assert env.cloud.folder_named("Never") is None  # Never wins over the global switch
        answer = await env.service.send_now(n)  # ...but Send now is explicit
        await env.settle()
        assert answer["ok"] and cloud_names(env.cloud, "Never") == ["Never.epub"]
        state = env.service.book_state(n)
        assert state["override"] == "never" and state["files"]["epub"]["status"] == "ok"

    run(scenario())


def test_job_transition_rules(tmp_path):
    async def scenario():
        env = Env(tmp_path)
        folder = env.book("BookE", {"BookE.epub": EPUB1})
        await env.link(enable=False)
        env.service.set_book_override(folder, "never")
        env.service._update_settings(enabled=True)  # switch on without the "sync everything" sweep
        env.service.set_book_override(folder, "default")
        env.store.clear_queue()
        await env.settle()
        writes = len(env.cloud.writes)
        env.done(folder, jid="f", state=JobState.FAILED)
        env.done(folder, jid="c", state=JobState.CANCELLED)
        env.done(folder, jid="q", previous=JobState.QUEUED)
        env.done(folder, jid="chat", kind="direct_text", origin={"type": "chat", "cid": "1"})
        env.done(folder, jid="tool", kind="qa_scan")
        await env.settle()
        assert len(env.cloud.writes) == writes and env.store.queue() == []
        env.done(folder, jid="ok")
        env.done(folder, jid="ok")  # the same job delivered twice
        await env.settle()
        assert len(env.cloud.writes) == writes + 1
        # a chat card's Compile of a book that is already in the Library is a trigger too
        env.recompile(folder, "BookE.epub", EPUB2)
        env.done(folder, jid="chat-compile", origin={"type": "chat", "cid": "1"})
        await env.settle()
        assert len(env.cloud.writes) == writes + 2

    run(scenario())


# ---------------------------------------------------------------------------------------------------
# busy guard, backoff, persistence, interrupted writes
# ---------------------------------------------------------------------------------------------------


def test_busy_guard_defers_until_the_job_ends(tmp_path):
    async def scenario():
        env = Env(tmp_path)
        folder = env.book("BookF", {"BookF.epub": EPUB1})
        env.jobs.active = JobSnapshot(id="run", spec=JobSpec(kind="translate", title="x", params={"folder": folder}),
                                      state=JobState.RUNNING, created=0.0)
        await env.link()
        assert env.cloud.writes == []
        entry = env.store.queued(folder)
        assert entry["last_error"] == cs.BUSY and entry["next_at"] > env.clock()  # no polling meanwhile
        env.jobs.active = None
        env.done(folder, jid="run", kind="qa_scan")  # any job end wakes the book
        await env.settle()
        assert cloud_names(env.cloud, "BookF") == ["BookF.epub"]

    run(scenario())


def test_provider_errors_back_off_and_survive_a_restart(tmp_path):
    async def scenario():
        clock = Clock()
        env = Env(tmp_path, clock=clock)
        folder = env.book("BookG", {"BookG.epub": EPUB1})
        env.cloud.fail_writes = 100
        await env.link()
        delays = []
        for attempt in range(1, 8):
            entry = env.store.queued(folder)
            assert entry["attempts"] == attempt
            delays.append(round(entry["next_at"] - clock.now))
            clock.now = entry["next_at"]
            await env.service.drain_now("timer")
        assert delays == [int(d) for d in BACKOFF] + [int(BACKOFF_MAX)] * 2
        failures = [n for n in env.bridge.notifications if "Couldn't save" in n["title"]]
        assert len(failures) == 1 and failures[0]["payload"].startswith("glossarion://app/")
        assert env.store.queued(folder)["attempts"] > FAIL_AFTER
        assert env.service.book_state(folder)["files"]["epub"]["status"] == "failed"
        # a restart keeps the queue and its attempts
        env.store.flush()
        reloaded = CloudRecordStore(env.data / "mobile_cloud.json", clock=clock, save_delay=0.0)
        reloaded.load()
        assert reloaded.queued(folder)["attempts"] == 8
        env.cloud.fail_writes = 0
        service = env.make_service(reloaded)
        clock.now += BACKOFF_MAX
        await service.drain_now("resume")
        assert cloud_names(env.cloud, "BookG") == ["BookG.epub"] and reloaded.queue() == []

    run(scenario())


def test_write_cut_off_by_process_death_is_rewritten(tmp_path):
    async def scenario():
        env = Env(tmp_path)
        folder = env.book("BookH", {"BookH.epub": EPUB1})
        await env.link()
        dest_id = env.service.destination().id
        # simulate a kill mid-write: the record says 'writing' on disk (hash equals the local file)
        env.store.update_record(dest_id, book_key(folder), "epub", folder, status="writing")
        env.store.enqueue(folder, "job:x")
        env.store.flush()
        reloaded = CloudRecordStore(env.data / "mobile_cloud.json", clock=env.clock, save_delay=0.0)
        reloaded.load()
        record = reloaded.record(dest_id, book_key(folder), "epub")
        assert record["status"] == "pending" and record["dirty"] is True
        writes = len(env.cloud.writes)
        service = env.make_service(reloaded)
        await service.drain_now("start")
        assert len(env.cloud.writes) == writes + 1  # rewritten although the hash matched
        assert reloaded.record(dest_id, book_key(folder), "epub")["status"] == "ok"

    run(scenario())


# ---------------------------------------------------------------------------------------------------
# provider failure table (critic #2 / #5 / #7 / #9)
# ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("says_missing", [False, True])
def test_offline_provider_never_creates_a_duplicate(tmp_path, says_missing):
    async def scenario():
        env = Env(tmp_path)
        folder = env.book("BookI", {"BookI.epub": EPUB1})
        await env.link()
        creates = len(env.cloud.creates)
        env.recompile(folder, "BookI.epub", EPUB2)
        env.cloud.offline = True
        env.cloud.offline_says_missing = says_missing
        env.done(folder, jid="j2")
        await env.settle()
        assert len(env.cloud.creates) == creates  # FileNotFoundException while offline is not "deleted"
        assert env.store.queued(folder)["attempts"] == 1
        env.cloud.offline = False
        env.clock.now += BACKOFF[0]
        await env.service.drain_now("timer")
        assert cloud_names(env.cloud, "BookI") == ["BookI.epub"]
        assert env.cloud.files(env.cloud.folder_named("BookI"))["BookI.epub"] == EPUB2
        assert len(env.cloud.creates) == creates

    run(scenario())


def test_deleted_cloud_file_is_created_again_once(tmp_path):
    async def scenario():
        env = Env(tmp_path)
        folder = env.book("BookJ", {"BookJ.epub": EPUB1})
        await env.link()
        nid = env.cloud.nid(env.store.record(env.service.destination().id, book_key(folder), "epub")["doc"])
        del env.cloud.nodes[nid]  # the user deleted it in the cloud app
        env.recompile(folder, "BookJ.epub", EPUB2)
        env.done(folder, jid="j2")
        await env.settle()
        assert cloud_names(env.cloud, "BookJ") == ["BookJ.epub"]
        assert env.cloud.files(env.cloud.folder_named("BookJ"))["BookJ.epub"] == EPUB2

    run(scenario())


def test_whole_book_folder_deleted_is_recreated(tmp_path):
    async def scenario():
        env = Env(tmp_path)
        folder = env.book("BookK", {"BookK.epub": EPUB1})
        await env.link()
        book_nid = env.cloud.folder_named("BookK")
        for c in env.cloud.children(book_nid):
            del env.cloud.nodes[c]
        del env.cloud.nodes[book_nid]
        env.recompile(folder, "BookK.epub", EPUB2)
        env.done(folder, jid="j2")
        await env.settle()
        assert cloud_names(env.cloud, "BookK") == ["BookK.epub"]

    run(scenario())


def test_existing_same_name_files_are_adopted_after_records_are_lost(tmp_path):
    async def scenario():
        env = Env(tmp_path)
        book_nid = env.cloud.add("root", "BookL", "folder")
        env.cloud.add(book_nid, "BookL.epub", data=b"old cloud copy")
        folder = env.book("BookL", {"BookL.epub": EPUB1})
        await env.link()  # e.g. after a reinstall wiped mobile_cloud.json
        assert env.cloud.creates == []  # folder and file adopted by name, nothing duplicated
        assert env.cloud.files(book_nid) == {"BookL.epub": EPUB1}
        assert folder

    run(scenario())


def test_non_truncating_provider_replaces_a_shorter_file_with_a_warning(tmp_path):
    async def scenario():
        env = Env(tmp_path, cloud=FakeCloud(modes=("w",)))  # only 'w' (OneDrive-like)
        folder = env.book("BookM", {"BookM.epub": EPUB2})
        await env.link()
        dest_id = env.service.destination().id
        first = env.store.record(dest_id, book_key(folder), "epub")["doc"]
        env.recompile(folder, "BookM.epub", EPUB1)  # shorter: 'w' would leave a stale tail
        env.done(folder, jid="j2")
        await env.settle()
        record = env.store.record(dest_id, book_key(folder), "epub")
        assert record["status"] == "ok" and record["note"] == cs.REPLACED_NOTE
        assert env.cloud.nid(record["doc"]) != env.cloud.nid(first)  # a new document; the old one deleted
        assert env.cloud.files(env.cloud.folder_named("BookM")) == {"BookM.epub": EPUB1}
        entry = env.service.book_state(folder)["files"]["epub"]
        assert entry["status"] == "ok" and entry["warning"]

    run(scenario())


def test_stale_tail_after_a_lying_truncating_mode_is_replaced(tmp_path):
    async def scenario():
        env = Env(tmp_path, cloud=FakeCloud(truncating=False, can_rename=False))
        folder = env.book("BookN", {"BookN.epub": EPUB2})
        await env.link()
        env.recompile(folder, "BookN.epub", EPUB1)
        env.done(folder, jid="j2")
        await env.settle()
        files = env.cloud.files(env.cloud.folder_named("BookN"))
        assert list(files.values()) == [EPUB1]  # one file, no corrupt ZIP with the old tail left
        # without rename the provider's ' (1)' name alternates with the original, it never piles up
        for i, data in enumerate((EPUB1[:400], EPUB1[:300], EPUB1[:200])):
            env.recompile(folder, "BookN.epub", data, later=20 + i)
            env.done(folder, jid=f"j{3 + i}")
            await env.settle()
            files = env.cloud.files(env.cloud.folder_named("BookN"))
            assert list(files.values()) == [data] and set(files) <= {"BookN.epub", "BookN (1).epub"}

    run(scenario())


def test_permission_lost_on_one_document_keeps_the_destination(tmp_path):
    async def scenario():
        env = Env(tmp_path)
        folder = env.book("BookO", {"BookO.epub": EPUB1})
        other = env.book("BookP", {"BookP.epub": EPUB1})
        await env.link()
        dest_id = env.service.destination().id
        nid = env.cloud.nid(env.store.record(dest_id, book_key(folder), "epub")["doc"])
        env.cloud.nodes[nid]["parent"] = "elsewhere"  # moved out of the folder by the user
        env.cloud.denied.add(nid)
        env.recompile(folder, "BookO.epub", EPUB2)
        env.recompile(other, "BookP.epub", EPUB2)
        env.done(folder, jid="j2")
        env.done(other, jid="j3")
        await env.settle()
        assert not env.service.destination().needs_relink
        assert env.cloud.files(env.cloud.folder_named("BookO")) == {"BookO.epub": EPUB2}  # created again
        assert env.cloud.files(env.cloud.folder_named("BookP")) == {"BookP.epub": EPUB2}

    run(scenario())


def test_revoked_destination_pauses_notifies_once_and_resumes_after_relink(tmp_path):
    async def scenario():
        env = Env(tmp_path)
        a = env.book("BookQ", {"BookQ.epub": EPUB1})
        b = env.book("BookR", {"BookR.epub": EPUB1})
        await env.link()
        env.cloud.revoked = True
        env.recompile(a, "BookQ.epub", EPUB2)
        env.recompile(b, "BookR.epub", EPUB2)
        env.done(a, jid="j2")
        env.done(b, jid="j3")
        await env.settle()
        dest = env.service.destination()
        assert dest.needs_relink == "revoked"
        lost = [n for n in env.bridge.notifications if "lost access" in n["title"].lower()]
        assert len(lost) == 1 and lost[0]["payload"] == "glossarion://app/settings/cloud"
        assert all("content://" not in json.dumps(n) for n in env.bridge.notifications)
        assert len(env.store.queue()) == 2  # paused, not dropped
        state = env.service.ui_state()
        assert state["destination"]["needs_relink"] == "revoked"
        env.cloud.revoked = False
        again = await env.service.pick_folder()  # "Choose again": the same folder keeps its records
        await env.settle()
        assert again["ok"] and not env.service.destination().needs_relink
        assert env.store.queue() == []
        assert cloud_names(env.cloud, "BookQ") == ["BookQ.epub"] and cloud_names(env.cloud, "BookR") == ["BookR.epub"]
        assert env.cloud.files(env.cloud.folder_named("BookQ"))["BookQ.epub"] == EPUB2

    run(scenario())


# ---------------------------------------------------------------------------------------------------
# destinations: change, own folder, per-file, phone folder, same titles
# ---------------------------------------------------------------------------------------------------


def test_change_destination_releases_the_old_grant_and_retires_its_records(tmp_path):
    async def scenario():
        env = Env(tmp_path)
        folder = env.book("BookS", {"BookS.epub": EPUB1})
        await env.link()
        old_id = env.service.destination().id
        other = FakeCloud(provider="com.other.docs", label="Other")
        other.nodes["root"]["name"] = "Elsewhere"
        target = other.ref("root")
        target.update(uri="content://com.other.docs/tree/root", document="content://com.other.docs/tree/root",
                      id="dother")
        env.cloud.pick_queue.append({"ok": True, "target": target, "persisted": True})
        result = await env.service.pick_folder()
        assert result["ok"]
        assert TREE in env.cloud.releases  # the old folder's grant is given back
        assert env.store.books(old_id) == {}  # nothing writes to the old place any more
        assert env.service.destination().id != old_id
        assert folder

    run(scenario())


def test_own_folder_is_refused(tmp_path):
    async def scenario():
        env = Env(tmp_path)
        env.cloud.own_folder = True
        result = await env.service.pick_folder()
        assert not result["ok"] and result["message"] == cs.OWN_FOLDER_REASON
        assert env.service.destination() is None and TREE in env.cloud.releases
        # an iOS-style answer: a path inside the app's own folders
        ios = {"ok": True, "target": {"id": "dx", "kind": "folder", "name": "Glossarion", "path": str(env.output)}}
        assert cs.is_own_location(ios, [str(env.output)])

    run(scenario())


def test_cancelled_picker_and_late_pick_after_process_death(tmp_path):
    async def scenario():
        env = Env(tmp_path)
        env.cloud.pick_queue.append({"ok": False, "error": "cancelled"})
        result = await env.service.pick_folder()
        assert not result["ok"] and result.get("cancelled")
        # the process died with the picker open: the answer arrives at the next start
        env.service._update_settings(pending_pick={"op": "folder", "op_id": "op1", "at": 0})
        target = env.cloud.ref("root")
        target.update(uri=TREE, document=TREE)
        env.cloud.late.append({"op_id": "op1", "kind": "folder", "status": "ok",
                               "result": {"ok": True, "target": target, "persisted": True}})
        await env.service.start()
        await env.settle()
        assert env.service.destination() is not None and env.service.settings().pending_pick is None

    run(scenario())


def test_files_mode_asks_once_then_updates_the_same_file(tmp_path):
    async def scenario():
        env = Env(tmp_path)
        folder = env.book("BookT", {"BookT.epub": EPUB1})
        result = await env.service.use_save_locations()
        assert result["ok"]
        env.service.set_enabled(True)
        await env.settle()
        state = env.service.book_state(folder)
        assert state["files"]["epub"]["status"] == "needs_pick"
        assert any("save location" in n["title"] for n in env.bridge.notifications)
        picked = await env.service.choose_save_location(folder, "epub")
        await env.settle()
        assert picked["ok"]
        record = env.store.record("files", book_key(folder), "epub")
        assert record["status"] == "ok"
        nid = env.cloud.nid(record["doc"])
        assert env.cloud.nodes[nid]["data"] == EPUB1
        env.recompile(folder, "BookT.epub", EPUB2)
        env.done(folder, jid="j2")
        await env.settle()
        assert env.cloud.nodes[nid]["data"] == EPUB2  # updated silently
        assert len([c for c in env.cloud.calls if c[0] == "pick_save_location"]) == 1
        # wipe gives every per-file grant back
        released = await env.service.wipe()
        assert released >= 1 and not env.cloud.grants

    run(scenario())


def test_files_mode_notifies_a_missing_save_location_once(tmp_path):
    async def scenario():
        env = Env(tmp_path)
        folder = env.book("BookTT", {"BookTT.epub": EPUB1})
        await env.service.use_save_locations()
        env.service.set_enabled(True)
        await env.settle()
        for i in range(3):  # recompiles while the book still has no save location
            env.recompile(folder, "BookTT.epub", EPUB2 + bytes([i]), later=10 + i)
            env.done(folder, jid=f"r{i}")
            await env.settle()
        asks = [n for n in env.bridge.notifications if "save location" in n["title"]]
        assert len(asks) == 1 and "content://" not in asks[0]["payload"]
        # an existing cloud file can be chosen instead ("Choose cloud file…")
        nid = env.cloud.add("picked", "Old copy.epub", data=b"old")
        env.cloud.pick_queue.append({"ok": True, "document": env.cloud.ref(nid)})
        assert (await env.service.choose_existing_file(folder, "epub"))["ok"]
        await env.settle()
        assert env.cloud.nodes[nid]["data"] == EPUB2 + bytes([2])
        assert env.store.record("files", book_key(folder), "epub")["status"] == "ok"

    run(scenario())


def test_folder_test_probe_and_write_progress(tmp_path):
    async def scenario():
        env = Env(tmp_path)
        folder = env.book("BookPP", {"BookPP.epub": EPUB1})
        await env.service.pick_folder()
        probe = await env.service.test_destination()
        assert probe["ok"] and env.cloud.files("root") == {}  # created, written and deleted again
        env.cloud.gate = asyncio.Event()
        await env.service.send_now(folder)
        for _ in range(50):
            await asyncio.sleep(0.01)
            if env.service._inflight:
                break
        op_id = env.service._inflight["op_id"]
        await env.service.on_document_event({"type": "progress", "op_id": op_id, "written": 400, "total": 850})
        state = env.service.ui_state()
        assert state["progress"] == {"name": "BookPP.epub", "written": 400, "total": 850}
        entry = env.service.book_state(folder)["files"]["epub"]
        assert entry["status"] == "writing" and entry["written"] == 400
        env.cloud.gate.set()
        await env.settle()
        assert env.service.ui_state()["progress"] is None

    run(scenario())


def test_files_mode_lost_file_asks_again_instead_of_guessing(tmp_path):
    async def scenario():
        env = Env(tmp_path)
        folder = env.book("BookU", {"BookU.epub": EPUB1})
        await env.service.use_save_locations()
        env.service.set_enabled(True)
        await env.settle()
        await env.service.choose_save_location(folder, "epub")
        await env.settle()
        nid = env.cloud.nid(env.store.record("files", book_key(folder), "epub")["doc"])
        del env.cloud.nodes[nid]
        env.recompile(folder, "BookU.epub", EPUB2)
        env.done(folder, jid="j2")
        await env.settle()
        # a per-file grant cannot prove deletion ('missing', proven False): no new file, retried later
        assert env.store.record("files", book_key(folder), "epub")["status"] != "ok"
        assert len([c for c in env.cloud.calls if c[0] == "pick_save_location"]) == 1

    run(scenario())


def test_phone_folder_overwrites_the_same_downloads_entry(tmp_path):
    async def scenario():
        env = Env(tmp_path)
        folder = env.book("BookV", {"BookV.epub": EPUB1})
        result = await env.service.use_phone_folder()
        assert result["ok"]
        env.service.set_enabled(True)
        await env.settle()
        assert len(env.cloud.downloads) == 1
        uri, entry = next(iter(env.cloud.downloads.items()))
        assert entry["subdir"] == "Glossarion/BookV" and entry["name"] == "BookV.epub"
        env.recompile(folder, "New Name.epub", EPUB2)
        os.remove(os.path.join(folder, "BookV.epub"))
        env.done(folder, jid="j2")
        await env.settle()
        assert list(env.cloud.downloads) == [uri]  # same entry, overwritten; never 'BookV (1).epub'
        assert env.cloud.downloads[uri] == {"name": "BookV.epub", "subdir": "Glossarion/BookV", "data": EPUB2}
        saves = [c for c in env.cloud.calls if c[0] == "save_to_downloads"]
        assert saves[-1][3] == uri

    run(scenario())


def test_mirror_switch_becomes_the_phone_folder_destination(tmp_path):
    async def scenario():
        env = Env(tmp_path, prefs=FakePrefs({"mirror_outputs": True}))
        await env.service.start()
        dest = env.service.destination()
        assert dest is not None and dest.mode == "phone" and env.service.settings().enabled

    run(scenario())


def test_same_title_books_never_share_a_cloud_file(tmp_path):
    async def scenario():
        env = Env(tmp_path)
        second_root = tmp_path / "Output2"
        a = env.book("Novel", {"Novel.epub": EPUB1})
        b = env.book("Novel", {"Novel.epub": EPUB2}, root=second_root)
        await env.link()
        assert sorted(n["name"] for n in env.cloud.nodes.values() if n["kind"] == "folder" and n["parent"] == "root") \
            == ["Novel", "Novel (2)"]
        assert a != b

    run(scenario())


def test_flat_layout_when_the_provider_cannot_make_folders(tmp_path):
    async def scenario():
        env = Env(tmp_path, cloud=FakeCloud(can_create_dirs=False))
        env.book("Novel", {"Novel.epub": EPUB1})
        env.book("Novel", {"Novel.epub": EPUB2}, root=tmp_path / "Output2")
        await env.link()
        assert env.service.destination().layout == "flat"
        assert sorted(env.cloud.files("root")) == ["Novel (2).epub", "Novel.epub"]

    run(scenario())


# ---------------------------------------------------------------------------------------------------
# chat books, relocation, deletion, wipe
# ---------------------------------------------------------------------------------------------------


def test_chat_book_waits_for_the_library_then_follows_its_move(tmp_path):
    async def scenario():
        env = Env(tmp_path)
        chat = tmp_path / "data" / "Direct Text" / "chat1" / "Attachments" / "Story"
        chat.mkdir(parents=True)
        (chat / "Story.epub").write_bytes(EPUB1)
        await env.link()
        answer = await env.service.send_now(str(chat))
        assert not answer["ok"] and answer["message"] == cs.NOT_IN_LIBRARY_REASON
        env.service.on_migrate_outcome({"status": "deferred", "folder": str(chat)})
        state = env.service.book_state(str(chat))
        assert state["in_library"] is False and state["waiting_library"]
        # auto-migrate moves it (worker thread) and reports the move
        target = env.output / "Story"
        os.replace(chat, target)
        env.books.append({"name": "Story", "path": str(target), "output_folder": str(target)})
        env.library.set_snapshot(ScanSnapshot(in_progress=tuple(env.books)))
        await asyncio.to_thread(env.service.on_workspace_moved, str(chat), str(target))
        await env.settle()
        assert cloud_names(env.cloud, "Story") == ["Story.epub"]

    run(scenario())


def test_merge_keeps_the_library_books_documents(tmp_path):
    store = CloudRecordStore(tmp_path / "c.json", save_delay=0.0)
    store.update_record("t", book_key(tmp_path / "chat"), "epub", tmp_path / "chat", doc={"id": "chat-doc"}, name="A")
    store.update_record("t", book_key(tmp_path / "chat"), "pdf", tmp_path / "chat", doc={"id": "chat-pdf"}, name="B")
    store.update_record("t", book_key(tmp_path / "lib"), "epub", tmp_path / "lib", doc={"id": "lib-doc"}, name="A")
    dropped = store.relocate(tmp_path / "chat", tmp_path / "lib", merge=True)
    assert dropped == [("t", {"id": "chat-doc"})]
    assert store.record("t", book_key(tmp_path / "lib"), "epub")["doc"] == {"id": "lib-doc"}
    assert store.record("t", book_key(tmp_path / "lib"), "pdf")["doc"] == {"id": "chat-pdf"}
    # a write that finishes after the move stores its result under the new key
    store.update_record("t", book_key(tmp_path / "chat"), "epub", None, status="ok")
    assert store.record("t", book_key(tmp_path / "lib"), "epub")["status"] == "ok"
    assert store.book("t", book_key(tmp_path / "chat")) is not None  # resolve follows the alias


def test_relocation_during_a_write_lands_on_the_new_key(tmp_path):
    async def scenario():
        env = Env(tmp_path)
        folder = env.book("BookW", {"BookW.epub": EPUB1})
        env.cloud.gate = asyncio.Event()
        await env.service.pick_folder()
        env.service.set_enabled(True)
        for _ in range(50):
            await asyncio.sleep(0.01)
            if any(c[0] == "write_file" for c in env.cloud.calls):
                break
        new = str(env.output / "BookW moved")
        await asyncio.to_thread(env.service.on_workspace_moved, folder, new)
        env.cloud.gate.set()
        await env.settle()
        dest_id = env.service.destination().id
        assert env.store.record(dest_id, book_key(new), "epub")["status"] == "ok"
        assert env.store.books(dest_id).get(book_key(folder)) is None

    run(scenario())


def test_book_deletion_drops_records_and_queue_but_keeps_cloud_files(tmp_path):
    async def scenario():
        env = Env(tmp_path)
        folder = env.book("BookX", {"BookX.epub": EPUB1})
        await env.link()
        env.store.enqueue(folder, "manual")
        await asyncio.to_thread(env.service.forget_books_threadsafe, [folder])
        await env.settle()
        dest_id = env.service.destination().id
        assert env.store.books(dest_id) == {} and env.store.queue() == []
        assert cloud_names(env.cloud, "BookX") == ["BookX.epub"]  # the cloud file stays

    run(scenario())


def test_wipe_releases_every_grant_and_forgets_everything(tmp_path):
    async def scenario():
        env = Env(tmp_path)
        env.book("BookY", {"BookY.epub": EPUB1})
        await env.link()
        env.cloud.grants.add("content://com.fake.docs/document/orphan")  # left by an old build
        released = await env.service.wipe()
        assert released >= 2 and not env.cloud.grants
        assert env.service.destination() is None and not (env.data / "mobile_cloud.json").exists()

    run(scenario())


# ---------------------------------------------------------------------------------------------------
# keeping the app alive
# ---------------------------------------------------------------------------------------------------


def test_android_hold_joins_the_job_service_synchronously_and_stops_it_once(tmp_path):
    async def scenario():
        from glossarion_mobile.services.background import BackgroundExecution

        env = Env(tmp_path)
        folder = env.book("BookZ", {"BookZ.epub": EPUB1})
        await env.link(enable=True)
        env.bridge.log.clear()
        background = BackgroundExecution(env.bridge, platform="android", jobs=env.jobs)
        background.service_running = True
        background.service_job = "j2"
        env.bridge.running = True
        env.bridge.service_holds.hold("jobs", "Glossarion", "Compiling…")
        spawned = []
        # JobsFeature's listener comes first and only spawns BackgroundExecution.on_transition
        env.jobs.listeners.insert(0, lambda snap, prev: spawned.append(
            asyncio.ensure_future(background.on_transition(snap, prev))))
        env.recompile(folder, "BookZ.epub", EPUB2)
        env.done(folder, jid="j2")
        assert CLOUD_HOLD in env.bridge.service_holds  # taken inside the callback, before job_finished ran
        await asyncio.gather(*spawned)
        await env.settle()
        methods = env.bridge.methods()
        assert methods.count("stop_job_service") == 1
        assert methods.index("stop_job_service") > methods.index("update_job_service")
        assert len(env.bridge.service_holds) == 0
        assert env.cloud.files(env.cloud.folder_named("BookZ"))["BookZ.epub"] == EPUB2

    run(scenario())


def test_android_hidden_without_service_waits_for_the_app(tmp_path):
    async def scenario():
        env = Env(tmp_path)
        folder = env.book("BookAA", {"BookAA.epub": EPUB1})
        await env.link(enable=False)
        env.service.on_lifecycle("hide")
        await env.service.send_now(folder)
        await env.settle()
        assert env.cloud.writes == [] and "start_job_service" not in env.bridge.methods()
        assert env.service._timer is None  # no retry timer spins while the app is in the background
        await asyncio.sleep(0.6)
        assert env.bridge.methods().count("is_job_service_running") == 1
        env.service.on_lifecycle("resume")
        await env.settle()
        assert cloud_names(env.cloud, "BookAA") == ["BookAA.epub"]

    run(scenario())


def test_foreground_timeout_cancels_the_write_and_drops_the_hold(tmp_path):
    async def scenario():
        env = Env(tmp_path)
        folder = env.book("BookAB", {"BookAB.epub": EPUB1})
        env.cloud.gate = asyncio.Event()
        await env.service.pick_folder()
        await env.service.send_now(folder)
        for _ in range(50):
            await asyncio.sleep(0.01)
            if any(c[0] == "write_file" for c in env.cloud.calls):
                break
        assert CLOUD_HOLD in env.bridge.service_holds
        await env.service.on_foreground_event({"type": "timeout"})
        await env.settle()
        assert CLOUD_HOLD not in env.bridge.service_holds and env.cloud.cancelled
        assert env.store.queued(folder) is not None  # kept for the next start / resume

    run(scenario())


def test_ios_drain_uses_a_background_task_and_respects_its_time(tmp_path):
    async def scenario():
        env = Env(tmp_path, platform="ios")
        folder = env.book("BookAC", {"BookAC.epub": EPUB1})
        await env.service.pick_folder()
        env.bridge.remaining = 5.0  # backgrounded, almost out of time
        await env.service.send_now(folder)
        await env.settle()
        assert env.cloud.writes == [] and env.store.queued(folder) is not None
        methods = env.bridge.methods()
        assert "begin_background_task" in methods and "end_background_task" in methods
        assert env.background.released >= 1  # the job's kept grant is let go when the drain ends
        env.bridge.remaining = None  # foreground again
        await env.service.drain_now("resume")
        assert cloud_names(env.cloud, "BookAC") == ["BookAC.epub"]

    run(scenario())


# ---------------------------------------------------------------------------------------------------
# privacy, UI contract, adapter
# ---------------------------------------------------------------------------------------------------


def test_ui_state_has_no_uris_and_nothing_goes_to_config_json(tmp_path):
    async def scenario():
        env = Env(tmp_path)
        folder = env.book("BookAD", {"BookAD.epub": EPUB1})
        await env.link()
        state = env.service.ui_state()
        text = json.dumps(state)
        assert "content://" not in text and state["destination"]["display"] == "FakeDrive › Glossarion"
        assert state["recent"] and state["recent"][0]["kind"] == "epub"
        assert "content://" not in json.dumps(env.service.book_state(folder))
        assert not list(tmp_path.rglob("config.json"))
        assert set(env.prefs.data) == {"cloud_sync"}  # one mobile-only Prefs key

    run(scenario())


def test_ui_facade_contract(tmp_path):
    screen = pytest.importorskip("glossarion_mobile.ui.screens.cloud_sync")
    facade_cls = getattr(screen, "CloudFacade", None)
    if facade_cls is None:
        pytest.skip("no CloudFacade in this build")

    async def scenario():
        env = Env(tmp_path)
        folder = env.book("BookAE", {"BookAE.epub": EPUB1})
        facade = facade_cls(env.service)
        assert (await facade.call("pick_folder"))["ok"]
        assert (await facade.call("set_enabled", True))["ok"]
        await env.settle()
        snap = facade.snapshot()
        assert snap["supported"] and snap["enabled"] and snap["destination"]["mode"] == "folder"
        book = facade.book(folder)
        assert book["files"]["epub"]["status"] == "ok" and book["in_library"]
        line = screen.file_status_line(book["files"]["epub"], dest_label="FakeDrive")
        assert line and line[0] == "CLOUD_DONE"
        assert (await facade.call("set_book_override", folder, "never"))["ok"]
        assert facade.book(folder)["override"] == "never"
        assert (await facade.call("send_now", folder))["ok"]
        assert (await facade.call("retry_now"))["ok"]
        assert (await facade.call("forget_destination"))["ok"]
        await env.settle()

    run(scenario())


def test_native_docs_adapter_on_a_stub_and_on_the_native_shapes(tmp_path):
    async def scenario():
        stub = NativeDocs(SimpleNamespace(native=SimpleNamespace(is_stub=True)), platform="android")
        assert not stub.available
        assert (await stub.query_root({"id": "x"}))["ok"] is False
        assert await stub.pick_folder() == {"ok": False, "error": "unavailable", "message": "", "scope": None}
        desktop = NativeDocs(SimpleNamespace(native=FakeCloud()), platform="desktop")
        assert not desktop.available
        cloud = FakeCloud()
        docs = NativeDocs(SimpleNamespace(native=cloud), platform="android")
        assert docs.available
        picked = await docs.pick_folder()
        assert picked["ok"] and picked["target"]["id"] == "droot"
        cloud.pick_queue.append({"ok": False, "error": "cancelled"})
        assert await docs.pick_folder() is None
        made = await docs.create_dir(picked["target"], "Book")
        assert made["ok"] and made["doc"]["kind"] == "folder"
        listing = await docs.list_children(picked["target"], ["Book"])
        assert listing["children"][0]["is_dir"] and listing["complete"]
        assert (await docs.list_children(picked["target"], ["nothing"]))["children"] == []
        grants = await docs.list_grants()
        assert grants == [TREE] and await docs.release(TREE) and await docs.list_grants() == []
        weird = await docs.write_file({"document": "content://x/document/none", "uri": TREE}, __file__)
        assert weird["ok"] is False and weird["error"] == "missing"
        assert cs.error_code({"error": "SecurityException"}) == "permission_lost"
        assert cs.error_code("totally unknown") == "provider_error"

    run(scenario())


def test_install_wires_the_app(tmp_path):
    async def scenario():
        env = Env(tmp_path)
        env.library.cloud_sync = None  # the attribute the Library's job-end mirror reads
        jobs_feature = SimpleNamespace(background=env.background, notifications=None)
        app = SimpleNamespace(
            paths=SimpleNamespace(data=str(env.data), cache=str(env.cache), docs=str(env.data),
                                  output=str(env.output), library=str(env.library_dir), temp=str(tmp_path / "t"),
                                  home=str(tmp_path / "home"), original_home=None),
            page=SimpleNamespace(platform=SimpleNamespace(value="android"), web=False),
            native=env.bridge, jobs=jobs_feature, job_service=env.jobs, prefs=env.prefs, library=env.library,
            files=None, dispatcher=None)
        service = await CloudSyncService.install(app)
        assert app.cloud_sync is service and env.library.cloud_sync is service
        assert service.supported and service.platform == "android"
        assert service.on_job_transition in env.jobs.listeners
        assert env.bridge.listeners["foreground"] and env.bridge.listeners["document"]
        assert service.store.loaded
        service.close()

    run(scenario())


def test_record_store_roundtrip_and_overrides(tmp_path):
    clock = Clock()
    store = CloudRecordStore(tmp_path / "c.json", clock=clock, save_delay=0.0)
    store.load()
    a = tmp_path / "a"
    store.enqueue(a, "job:1", reported=["x.epub"])
    store.enqueue(a, "manual", manual=True, reported=["y.epub"])
    entry = store.queued(a)
    assert entry["manual"] and entry["reported"] == ["x.epub", "y.epub"] and len(store.queue()) == 1
    store.set_override(a, "always")
    store.set_override(tmp_path / "b", "never")
    store.set_override(tmp_path / "b", "default")
    assert store.overrides() == {book_key(a): "always"}
    with pytest.raises(ValueError):
        store.set_override(a, "sometimes")
    store.flush()
    other = CloudRecordStore(tmp_path / "c.json", clock=clock, save_delay=0.0)
    assert other.load() and other.queued(a)["reason"] == "manual" and other.override(a) == "always"
    (tmp_path / "c.json").write_text("{broken", encoding="utf-8")
    broken = CloudRecordStore(tmp_path / "c.json", clock=clock)
    assert broken.load() is False and broken.queue() == []
    assert list(tmp_path.glob("c.json.corrupt-*"))


def test_sources_are_ascii_safe_and_python310_parsable():
    import ast

    for path in (APP_DIR / "glossarion_mobile" / "services" / "cloud_sync.py",
                 APP_DIR / "glossarion_mobile" / "state" / "cloud_records.py"):
        data = path.read_bytes()
        assert not data.startswith(b"\xef\xbb\xbf")
        assert data.count(b"\r\n") in (0, data.count(b"\n"))  # uniform line endings, never mixed
        ast.parse(data.decode("utf-8"), feature_version=(3, 10))
        assert b"import flet" not in data  # pure module: imports without the UI toolkit
