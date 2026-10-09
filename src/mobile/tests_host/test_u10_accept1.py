"""U10 acceptance, item 1: "A finished Library book is copied into the picked cloud folder automatically
and updated in place on recompile".

Proved end to end on the REAL app objects: the app started on a fake Flet session
(``test_ui_foundations._start`` through ``test_ui_flows._host_driver``, as the other tests_host UI tests),
the real JobService / ChatFeature / ChatRuns / LibraryService / CloudSyncService, the shared desktop
pipeline (chat translation, EPUB / PDF compile, metadata.json), the Settings › Cloud sync & sharing
screen and the Book page's Output tab driven through ``UiDriver`` taps. Only three things are fakes:

* the model: the offline fake OpenAI server (``diagnostics.fake_llm_server``) on 127.0.0.1;
* the native side: ``services.native.create_native`` returns a ``NativeStub`` whose document API is the
  extension's own Android fake (``flet_glossarion_native/documents_fake.py``: ``FakeDocumentsNative`` over a
  ``FakeCloudProvider`` labelled "Drive"), so the app's ``NativeBridge`` / ``NativeDocs`` /
  ``CloudSyncService.install`` wiring is the production one;
* the system folder picker's answer (the provider's ``next_pick``).

The scenario (one app session; the steps build on each other):

1. A chat book (＋ › Files, Send, Start) finishes and auto-migrates into the Library; its Book page
   compiles a PDF. With no destination nothing is written anywhere.
2. Settings › Cloud sync & sharing: "Choose a folder…" links the picked "Drive › Glossarion Books"
   folder; nothing is copied while "Copy finished books automatically" is off.
3. With PDF deselected, turning the switch on copies the EPUB once into a per-book folder named like the
   Library's; selecting PDF copies the PDF once; resuming the app and switching off / on again copy
   nothing more (each enabled format exactly once).
4. A second chat book finishing while sync is on is copied without any tap once it lands in the Library.
5. Recompile (Book page › Compile EPUB) after a chapter edit: the SAME cloud document (same document id,
   same name, no new file) gets the new bytes and its length is verified; a shorter recompile leaves no
   stale tail.
6. Title change (Book page › Edit metadata) + recompile: the EPUB on the phone gets the new title's name,
   the cloud file keeps its original name and document and gets the new bytes.
7. Per-book "Never" (Copy this book: Never): a recompile is not copied; back to "Default" copies it.
8. Global switch off: a recompile is not copied; "Send now" still copies it once.

Nothing touches the network: a socket guard refuses and records every connect / name lookup that is not
loopback (the fake model); only the model catalog's 24 h provider poll (desktop parity, not cloud sync)
is exempted, as in test_devfix_issue12. Real data stays untouched: temp FLET_APP_STORAGE_* dirs (the bootstrap points HOME,
OUTPUT_DIRECTORY, GLOSSARION_LIBRARY_DIR, GLOSSARION_DATA_DIR and CONFIG_FILE there), USERPROFILE /
APPDATA / LOCALAPPDATA in tmp_path, GLOSSARION_HTTP_LOG=0.

Run from src/mobile with the mobile venv (our Python 3.12; keep the shell's PYTHONPATH)::

    .venv/Scripts/python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic \
        tests_host/test_u10_accept1.py
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import os
import socket
import sys
import time
import traceback
import types
from pathlib import Path

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = MOBILE_DIR / "app"
SRC_DIR = MOBILE_DIR.parent
TESTS_DIR = MOBILE_DIR / "tests"
EXTENSION_PKG = MOBILE_DIR / "extensions" / "flet_glossarion_native" / "src" / "flet_glossarion_native"
for entry in (str(APP_DIR), str(SRC_DIR), str(TESTS_DIR)):
    if entry not in sys.path:
        sys.path.insert(0, entry)


def _has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


_NEEDED = ("flet", "msgpack", "ebooklib", "openai", "httpx", "tiktoken", "bs4", "lxml", "fitz")
pytestmark = pytest.mark.skipif(not all(_has(m) for m in _NEEDED),
                                reason=f"needs {', '.join(_NEEDED)} (the mobile venv)")


def _load(alias: str, file_name: str):
    spec = importlib.util.spec_from_file_location(alias, Path(__file__).with_name(file_name))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_UF = _load("_glossarion_uiflows_u10accept1", "test_ui_flows.py")  # _host_driver / _foundations
storage = _UF.storage
app_env = _UF.app_env

RUN_TIMEOUT = 180.0
JOB_TIMEOUT = 180.0
CLOUD_TIMEOUT = 60.0
PICKED = "Glossarion Books"
PROVIDER_LABEL = "Drive"
BOOK_A = "Cloud Alpha"
BOOK_B = "Cloud Beta"
NEW_TITLE = "Renamed Alpha"
MARKER = "U10 acceptance edit"


# ==========================================================================
# The extension's document fake, loaded without the extension package
# ==========================================================================


def _module_from(path: Path, alias: str):
    spec = importlib.util.spec_from_file_location(alias, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module  # dataclasses resolve their module while the class is built
    spec.loader.exec_module(module)
    return module


def _documents_fake():
    """``flet_glossarion_native.documents_fake`` without importing the package (its ``__init__`` builds the
    Flet service and a cached import would turn every later Android host test's stub into the real
    service): ``documents.py`` is pure, both load under private names."""
    cached = sys.modules.get("_u10accept1_documents_fake")
    if cached is not None:
        return cached
    documents = _module_from(EXTENSION_PKG / "documents.py", "_u10accept1_documents")
    name = "flet_glossarion_native.documents"
    saved = sys.modules.get(name)
    sys.modules[name] = documents
    try:
        return _module_from(EXTENSION_PKG / "documents_fake.py", "_u10accept1_documents_fake")
    finally:
        if saved is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = saved


#: The document API the cloud sync calls on the extension (``cloud_sync.NativeDocs``).
_DOC_METHODS = ("pick_folder", "pick_save_location", "pick_document", "list_children", "create_file",
                "create_folder", "write_file", "rename_document", "stat", "delete", "query_root", "release",
                "list_grants", "cancel_document_op", "take_document_results")


def _fake_native_class():
    from glossarion_mobile.services.native import NativeStub

    class FakeNative(NativeStub):
        """The app's native service on a phone: every non-document call is the stub's no-op (as on the other
        Android host tests), the document API is the extension's Android fake; write progress arrives as the
        bridge's ``document`` events like ``GlossarionNative.on_document``."""

        is_stub = False  # NativeDocs.available: the extension is there

        def __init__(self, provider, **handlers):
            super().__init__("u10 acceptance fake", **handlers)
            self.provider = provider
            self.docs = _documents_fake().FakeDocumentsNative(provider, chunk_bytes=16 * 1024,
                                                               on_progress=self._progress)
            self.doc_calls: list = []
            self.progress: list = []

        def _progress(self, payload):
            self.progress.append(dict(payload))
            handler = self.on_document
            if handler is None:
                return
            try:
                asyncio.get_running_loop().create_task(handler(dict(payload)))
            except RuntimeError:
                pass

        async def get_platform_info(self):
            return {"platform": "android", "native": True, "documents": True, "save_to_downloads": True,
                    "sdk_int": 34, "persisted_grant_limit": 512}

    def delegate(name):
        async def call(self, *args, **kwargs):
            self.doc_calls.append(name)
            return await getattr(self.docs, name)(*args, **kwargs)

        call.__name__ = name
        return call

    for method in _DOC_METHODS:
        setattr(FakeNative, method, delegate(method))
    return FakeNative


# ==========================================================================
# Fixtures
# ==========================================================================


@pytest.fixture
def iso(tmp_path, monkeypatch, request):
    """The app on temp storage, with the Windows profile variables redirected as well."""
    user = tmp_path / "user"
    for key, sub in (("USERPROFILE", "."), ("APPDATA", "AppData/Roaming"), ("LOCALAPPDATA", "AppData/Local")):
        folder = (user / sub).resolve()
        folder.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv(key, str(folder))
    monkeypatch.setenv("GLOSSARION_HTTP_LOG", "0")
    request.getfixturevalue("app_env")  # FLET_APP_STORAGE_* + bootstrap (HOME, OUTPUT_DIRECTORY, Library, data)
    picks = tmp_path / "picks"
    picks.mkdir()
    return types.SimpleNamespace(tmp=tmp_path, picks=picks, user=user)


_LOCAL_HOSTS = ("127.0.0.1", "::1", "localhost", "0.0.0.0", "")


def _host_of(address) -> str:
    if isinstance(address, tuple) and address:
        return str(address[0])
    return str(address or "")


@pytest.fixture
def network_guard(monkeypatch):
    """Every connect / name lookup that is not loopback is recorded and refused. (The fake model listens on
    127.0.0.1; the Windows event loop's self-pipe is a loopback pair.)"""
    attempts: list = []
    connect, connect_ex, getaddrinfo = socket.socket.connect, socket.socket.connect_ex, socket.getaddrinfo

    def guarded(fn):
        def wrapper(self, address, *args, **kwargs):
            if self.family in (socket.AF_INET, socket.AF_INET6) and _host_of(address) not in _LOCAL_HOSTS:
                attempts.append(("connect", repr(address), "".join(traceback.format_stack(limit=60)[:-1])))
                raise ConnectionRefusedError(f"network access attempted: {address!r}")
            return fn(self, address, *args, **kwargs)

        return wrapper

    def lookup(host, *args, **kwargs):
        name = host.decode() if isinstance(host, bytes) else str(host or "")
        if name not in _LOCAL_HOSTS:
            attempts.append(("getaddrinfo", name, "".join(traceback.format_stack(limit=60)[:-1])))
            raise socket.gaierror(socket.EAI_NONAME, f"network access attempted: {name}")
        return getaddrinfo(host, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", guarded(connect))
    monkeypatch.setattr(socket.socket, "connect_ex", guarded(connect_ex))
    monkeypatch.setattr(socket, "getaddrinfo", lookup)
    return attempts


def _catalog_poll(attempt: tuple) -> bool:
    """The selected model's 24 h provider-catalog auto-poll (``model_options._fetch_provider_catalog`` on a
    ``model-catalog`` thread; desktop parity, the configured dummy OpenAI key makes it ask api.openai.com):
    not part of cloud sync. The guard refuses it like every other attempt (the catalog keeps its static
    list); ``test_devfix_issue12`` exempts it the same way."""
    return "_fetch_provider_catalog" in str(attempt[2] if len(attempt) > 2 else "")


@pytest.fixture
def cloud_provider(monkeypatch):
    """The cloud app behind the system picker ("Drive" with a "Glossarion Books" folder) and the patched
    ``create_native`` that hands the app its fake native service."""
    from glossarion_mobile.services import native as native_mod

    fake_mod = _documents_fake()
    provider = fake_mod.FakeCloudProvider(label=PROVIDER_LABEL, authority="com.example.drive.documents")
    picked = provider.add_folder(PICKED)
    provider.add_file("someone else's notes.txt", b"not Glossarion's", parent=provider.root)
    FakeNative = _fake_native_class()
    made: list = []

    def create_native(page, handlers=None):
        native = FakeNative(provider, **dict(handlers or {}))
        made.append(native)
        return native

    monkeypatch.setattr(native_mod, "create_native", create_native)
    return types.SimpleNamespace(provider=provider, picked=picked, natives=made)


# ==========================================================================
# Helpers
# ==========================================================================


def _same(a, b) -> bool:
    return bool(a) and bool(b) and os.path.normcase(os.path.abspath(str(a))) == os.path.normcase(os.path.abspath(str(b)))


async def _until(predicate, timeout: float = 30.0, step: float = 0.05) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(step)
    return bool(predicate())


def _configure(app, server) -> None:
    """The settings the device flows import from desktop (``flows.ui_config``): the fake OpenAI endpoint,
    a dummy key, glossary off, no request spacing."""
    import flows
    from glossarion_mobile.diagnostics.fake_llm_server import FAKE_MODEL

    store = app.config_store
    store.set_many(flows.ui_config(server.url, FAKE_MODEL))
    store.flush()
    assert store.save_error is None, store.save_error


def _record_notes(app) -> list:
    notes: list = []
    original = app.notify

    def notify(message, action_label=None, on_action=None):
        notes.append(str(message))
        return original(message, action_label, on_action)

    app.notify = notify
    return notes


async def _join_finish_threads(app) -> None:
    for thread in list(app.chat_feature.runs.finish_threads):
        await asyncio.to_thread(thread.join, 60)


async def _chat_book(app, driver, epub_name: str) -> None:
    """The owner's chat flow (``flows.chat_translate_and_migrate``): ＋ › Files, Send, Start, Done, "Added to
    the Library" with no further tap."""
    import flows

    await flows.chat_translate_and_migrate(driver, epub=epub_name, timeout=RUN_TIMEOUT)
    await _join_finish_threads(app)


def _cloud_files(provider, folder) -> dict:
    """``{"Book/name": node}`` of the live files under ``folder`` (one level of book folders)."""
    out: dict = {}

    def walk(node, prefix):
        for child in node.children:
            if child.deleted:
                continue
            path = f"{prefix}{child.name}"
            if child.is_dir:
                walk(child, path + "/")
            else:
                out[path] = child

    walk(folder, "")
    return out


def _all_cloud_files(provider) -> dict:
    return _cloud_files(provider, provider.root)


def _writes(native) -> list:
    """``write_file`` calls the document fake answered (``(doc id, source path)``)."""
    out = []
    for name, args in native.docs.calls:
        if name != "write_file":
            continue
        ref = args.get("ref") or {}
        document = ref.get("document") if isinstance(ref, dict) else str(ref)
        out.append((native.provider.doc_id_of(str(document or "")), args.get("source_path")))
    return out


def _cloud_busy(cloud) -> bool:
    task = cloud._drain_task
    return (task is not None and not task.done()) or bool(cloud.store.queue())


async def _cloud_settled(cloud, timeout: float = CLOUD_TIMEOUT) -> None:
    """No drain running and nothing queued, for a few polls in a row (a kick spawns its drain a tick later)."""
    deadline = time.monotonic() + timeout
    quiet = 0
    while time.monotonic() < deadline:
        quiet = 0 if _cloud_busy(cloud) else quiet + 1
        if quiet >= 8:
            await cloud.wait_idle()
            return
        await asyncio.sleep(0.05)
    raise AssertionError(f"cloud sync did not settle: queue={cloud.store.queue()} events={cloud.events[-12:]}")


async def _cloud_quiet(cloud, seconds: float = 1.5) -> None:
    """Let a trigger that should do nothing have its chance (a drain spawned by it would show up)."""
    await asyncio.sleep(seconds)
    await _cloud_settled(cloud)


def _record(cloud, identity: str, kind: str) -> dict:
    dest = cloud.destination()
    assert dest is not None
    key = cloud.store.resolve(os.path.normcase(os.path.abspath(identity)))
    return cloud.store.record(dest.id, key, kind) or {}


def _record_doc_id(provider, record: dict) -> str:
    doc = record.get("doc") or {}
    return provider.doc_id_of(str(doc.get("document") if isinstance(doc, dict) else doc))


def _local(folder: Path, suffix: str) -> list:
    return sorted((p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == suffix),
                  key=lambda p: p.stat().st_mtime_ns)


def _chapter_file(folder: Path) -> Path:
    files = sorted(p for p in folder.iterdir()
                   if p.is_file() and p.name.startswith("response_") and p.suffix.lower() in (".html", ".xhtml", ".htm"))
    assert files, f"no translated chapter file in {sorted(p.name for p in folder.iterdir())}"
    return files[0]


def _edit_chapter(path: Path, text: str) -> None:
    """A chapter edit (what the Reader / Edit file saves): a paragraph ``text`` at the end of the chapter's
    (first) body; the model's answers nest a second ``<body>`` and the compiler keeps the inner one."""
    data = path.read_text(encoding="utf-8")
    at = data.lower().find("</body>")
    paragraph = f"<p>{text}</p>\n"
    data = data[:at] + paragraph + data[at:] if at >= 0 else data + paragraph
    path.write_text(data, encoding="utf-8")


def _words(seed: str, count: int) -> str:
    """Text that does not compress away (the EPUB is a zip: a repeated phrase would add a few bytes only)."""
    return " ".join(hashlib.sha1(f"{seed}{i}".encode()).hexdigest()[:10] for i in range(count))


def _assert_pdf_follows(cloud_pdf, local_pdf: Path, stamp: int) -> None:
    """The cloud PDF is the phone's PDF; an EPUB-only compile left the PDF alone, so it was not rewritten."""
    assert bytes(cloud_pdf.data) == local_pdf.read_bytes()
    if local_pdf.stat().st_mtime_ns == stamp:
        assert cloud_pdf.versions == 1, cloud_pdf.versions


class Jobs:
    """Every job transition (``JobService.on_transition``), to wait for the compile a tap started."""

    def __init__(self, service) -> None:
        self.service = service
        self.snaps: list = []
        self.unsub = service.on_transition(lambda snap, previous: self.snaps.append(snap))

    def ids(self, kind: str) -> set:
        return {s.id for s in self.snaps if s.spec.kind == kind}

    async def finished(self, kind: str, before: set, timeout: float = JOB_TIMEOUT):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            done = [s for s in self.snaps if s.spec.kind == kind and s.id not in before and s.is_terminal]
            if done:
                await asyncio.sleep(0.2)  # the other transition listeners (Library refresh, cloud queue)
                return done[-1]
            await asyncio.sleep(0.05)
        raise AssertionError(f"no {kind} job finished: {[(s.spec.kind, s.state) for s in self.snaps[-6:]]}")


async def _book_page(app, bid: str):
    await app.navigate(f"/library/book/{bid}?tab=output", reset=True)
    assert await _until(lambda: getattr(app.shell.top_screen, "bid", None) == bid, 15), app.shell.current_route
    return app.shell.top_screen


async def _compile(app, driver, jobs: Jobs, bid: str, button: str, job_kind: str = "compile_epub"):
    """Book page › Output › Compile EPUB (``out-epub``) / Compile PDF (``out-pdf``) through a tap; returns the
    finished job's snapshot (DONE asserted)."""
    await _book_page(app, bid)
    before = jobs.ids(job_kind)
    await driver.tap(key=button, timeout=30)
    snap = await jobs.finished(job_kind, before)
    assert str(getattr(snap.state, "value", snap.state)) == "DONE", (snap.state, snap.error)
    return snap


async def _cloud_screen(app):
    await app.navigate("/settings/cloud", reset=True)
    assert await _until(lambda: type(app.shell.top_screen).__name__ == "CloudSyncScreen", 15)
    screen = app.shell.top_screen
    assert await _until(lambda: screen.loaded, 15)
    return screen


async def _toggle_chip(tester, driver, key: str, selected: bool) -> None:
    """A format chip tap as the client sends it (Flutter flips ``selected``, then fires ``select``)."""
    finder = await driver.wait(key=key, timeout=15)
    chip = tester.control(finder.first)
    assert not tester._disabled(chip), key
    chip.selected = selected
    await tester._dispatch(chip, "select")
    await asyncio.sleep(0.2)


async def _set_switch(tester, driver, screen, value: bool) -> None:
    """"Copy finished books automatically" through a tap (the host tester flips it, as the client does)."""
    await _until(lambda: not screen.auto_switch.disabled, 10)
    if bool(screen.auto_switch.value) != value:
        await driver.tap(key="cloud-auto", timeout=15)
    assert await _until(lambda: screen.facade.service.settings().enabled is value, 10)


async def _set_override(app, driver, bid: str, value: str) -> None:
    """Book page › Output › "Copy this book: …" › Default / Always / Never."""
    await _book_page(app, bid)
    await driver.tap(contains="Copy this book:", timeout=30)
    await driver.tap(key=f"cloud-override-{value}", timeout=15)
    identity = app.library.identity_for_bid(bid)
    assert await _until(lambda: app.cloud_sync.override(identity) == value, 10)


# ==========================================================================
# The scenario
# ==========================================================================


def test_library_book_is_copied_to_the_picked_folder_and_updated_in_place(iso, cloud_provider, network_guard):
    from glossarion_mobile.diagnostics import fixtures
    from glossarion_mobile.diagnostics.fake_llm_server import FakeLLMServer

    provider, picked = cloud_provider.provider, cloud_provider.picked
    epub_a = fixtures.build_tiny_epub(iso.picks / f"{BOOK_A}.epub", chapters=3)
    epub_b = fixtures.build_tiny_epub(iso.picks / f"{BOOK_B}.epub", chapters=2)

    async def scenario(server):
        tf = _UF._foundations()
        app, tester, driver = await _UF._host_driver(tf, {epub_a.name: epub_a, epub_b.name: epub_b})
        cloud = None
        try:
            import flows

            await flows.wait_home(driver)
            _configure(app, server)
            notes = _record_notes(app)
            jobs = Jobs(app.job_service)
            out_root = Path(app.paths.output)
            for env, root in (("OUTPUT_DIRECTORY", out_root), ("GLOSSARION_LIBRARY_DIR", None),
                              ("GLOSSARION_DATA_DIR", None), ("HOME", None), ("USERPROFILE", None)):
                value = os.environ.get(env)
                assert value and Path(value).resolve().is_relative_to(iso.tmp.resolve()), (env, value)
                if root is not None:
                    assert _same(value, root)

            # ---- the app's own wiring: the bridge's native service is the fake, the cloud sync uses it ------
            assert cloud_provider.natives, "the app did not build its native service through create_native"
            native = app.native.native
            assert native is cloud_provider.natives[-1] and not app.native.is_stub
            cloud = app.cloud_sync
            assert cloud is not None and cloud.platform == "android" and cloud.supported
            assert cloud.docs.bridge is app.native and app.library.cloud_sync is cloud
            assert cloud.destination() is None and not cloud.settings().enabled  # off by default

            # =========== 1. a chat book lands in the Library; its PDF is compiled; nothing is copied ========
            await _chat_book(app, driver, epub_a.name)
            book_a = out_root / BOOK_A
            assert (book_a / "translation_progress.json").is_file(), sorted(p.name for p in out_root.iterdir())
            assert "Added to the Library" in notes, notes
            library = app.library
            await library.refresh(quiet=True, reason="test")  # what opening the Library does
            row_a = next((b for b in library.snapshot.all_books() if _same(b.get("output_folder"), book_a)), None)
            assert row_a is not None, [(b.get("name"), b.get("output_folder")) for b in library.snapshot.all_books()]
            bid_a = library.bid_for(row_a)
            identity_a = str(book_a)
            snap = await _compile(app, driver, jobs, bid_a, "out-pdf")
            epubs, pdfs = _local(book_a, ".epub"), _local(book_a, ".pdf")
            assert epubs and len(pdfs) == 1, sorted(p.name for p in book_a.iterdir())
            # The compile's EPUB is the newest one; an older one (the chat run's, named while the workspace sat
            # in the chat's longer Attachments path, or before a title change) may stay beside it: never copied.
            current_epub = epubs[-1]
            stale_epubs = {p.name for p in epubs[:-1]}
            reported = [Path(p) for p in (snap.outputs or ())]
            assert any(_same(p, current_epub) for p in reported) or not reported, (snap.outputs, epubs)
            epub_name, pdf_name = current_epub.name, pdfs[0].name
            await _cloud_quiet(cloud, 0.5)
            assert _writes(native) == [] and not native.docs.calls, native.docs.calls
            assert set(_all_cloud_files(provider)) == {"someone else's notes.txt"}
            assert cloud.store.queue() == []

            # =========== 2. Settings › Cloud sync: choose the folder; switch still off ======================
            screen = await _cloud_screen(app)
            assert screen.auto_switch.disabled and screen.state.get("destination") is None
            provider.next_pick("folder", picked)
            await driver.tap(text="Choose a folder…", timeout=15)
            assert await _until(lambda: cloud.destination() is not None, 15), cloud.events
            dest = cloud.destination()
            assert dest.mode == "folder" and dest.display == f"{PROVIDER_LABEL} › {PICKED}", dest
            assert await _until(lambda: screen.destination_text.value == dest.display, 10), \
                screen.destination_text.value
            assert native.doc_calls.count("pick_folder") == 1
            await _cloud_quiet(cloud)
            assert _writes(native) == [], "copied while 'Copy finished books automatically' is off"
            assert set(_cloud_files(provider, picked)) == set()

            # =========== 3. switch on with PDF deselected: the EPUB once; PDF on: the PDF once ==============
            await _toggle_chip(tester, driver, "cloud-kind-pdf", False)
            assert await _until(lambda: cloud.settings().kinds["pdf"] is False, 10)
            await _set_switch(tester, driver, screen, True)
            await _cloud_settled(cloud)
            files = _cloud_files(provider, picked)
            assert set(files) == {f"{BOOK_A}/{epub_name}"}, set(files)
            cloud_epub = files[f"{BOOK_A}/{epub_name}"]
            assert bytes(cloud_epub.data) == current_epub.read_bytes() and cloud_epub.versions == 1
            assert not stale_epubs & {Path(n).name for n in files}, stale_epubs  # the stale EPUB stays home
            book_dir = [c for c in picked.children if not c.deleted]
            assert [(c.name, c.is_dir) for c in book_dir] == [(BOOK_A, True)]  # the Library's per-book layout
            await _toggle_chip(tester, driver, "cloud-kind-pdf", True)
            await _cloud_settled(cloud)
            files = _cloud_files(provider, picked)
            assert set(files) == {f"{BOOK_A}/{epub_name}", f"{BOOK_A}/{pdf_name}"}, set(files)
            cloud_pdf = files[f"{BOOK_A}/{pdf_name}"]
            assert bytes(cloud_pdf.data) == pdfs[0].read_bytes() and cloud_pdf.versions == 1
            assert cloud_epub.versions == 1, "the unchanged EPUB was written again"
            pdf_stamp = pdfs[0].stat().st_mtime_ns
            writes_before = len(_writes(native))
            await tester.session.dispatch_event(tester.page._i, "app_lifecycle_state_change", {"state": "resume"})
            await _cloud_quiet(cloud)
            await _set_switch(tester, driver, screen, False)
            await _set_switch(tester, driver, screen, True)
            await _cloud_quiet(cloud)
            assert len(_writes(native)) == writes_before and cloud_epub.versions == 1 and cloud_pdf.versions == 1
            epub_doc = _record_doc_id(provider, _record(cloud, identity_a, "epub"))
            assert epub_doc == cloud_epub.doc_id and _record(cloud, identity_a, "epub")["name"] == epub_name
            assert _record(cloud, identity_a, "pdf")["status"] == "ok"
            # the Settings activity and the Book page say so
            state = await asyncio.to_thread(cloud.ui_state)
            assert state["enabled"] and state["queue"] == [] and state["failed"] == 0
            assert {item["name"] for item in state["recent"]} >= {epub_name, pdf_name}
            entries = await asyncio.to_thread(cloud.file_entries, identity_a)
            assert entries["epub"]["status"] == "ok" and entries["pdf"]["status"] == "ok", entries

            # =========== 4. a chat book finishing while sync is on is copied without a tap ==================
            await _chat_book(app, driver, epub_b.name)
            book_b = out_root / BOOK_B
            assert (book_b / "translation_progress.json").is_file(), sorted(p.name for p in out_root.iterdir())
            epub_b_local = _local(book_b, ".epub")
            assert len(epub_b_local) == 1, sorted(p.name for p in book_b.iterdir())
            assert await _until(lambda: f"{BOOK_B}/{epub_b_local[0].name}" in _cloud_files(provider, picked),
                                CLOUD_TIMEOUT), (cloud.events[-12:], cloud.store.queue())
            await _cloud_settled(cloud)
            node_b = _cloud_files(provider, picked)[f"{BOOK_B}/{epub_b_local[0].name}"]
            assert bytes(node_b.data) == epub_b_local[0].read_bytes() and node_b.versions == 1
            assert native.doc_calls.count("pick_folder") == 1  # no second picker: the folder was picked once
            assert cloud_epub.versions == 1 and cloud_pdf.versions == 1  # book A untouched

            # =========== 5. recompile: the same document gets the new bytes, length verified ===============
            chapter = _chapter_file(book_a)
            original_chapter = chapter.read_bytes()
            _edit_chapter(chapter, MARKER + " " + _words("longer", 300))
            files_before = set(_cloud_files(provider, picked))
            await _compile(app, driver, jobs, bid_a, "out-epub")
            await _cloud_settled(cloud)
            local_epub = book_a / epub_name
            assert local_epub.is_file() and _local(book_a, ".epub")[-1] == local_epub, _local(book_a, ".epub")
            data = local_epub.read_bytes()
            assert cloud_epub.versions == 2 and bytes(cloud_epub.data) == data and cloud_epub.size == len(data)
            assert set(_cloud_files(provider, picked)) == files_before, "a recompile added a cloud file"
            record = _record(cloud, identity_a, "epub")
            assert _record_doc_id(provider, record) == epub_doc and record["name"] == epub_name
            assert record["remote_size"] == len(data) == record["size"], record
            assert record["mode"] == "wt" and provider.opened_modes.count("wt") == len(_writes(native))  # truncating
            longer = len(data)
            chapter.write_bytes(original_chapter)  # the edit undone: a shorter EPUB
            await _compile(app, driver, jobs, bid_a, "out-epub")
            await _cloud_settled(cloud)
            data = local_epub.read_bytes()
            assert len(data) < longer
            assert cloud_epub.versions == 3 and bytes(cloud_epub.data) == data and cloud_epub.size == len(data), \
                (cloud_epub.size, len(data))
            assert _record(cloud, identity_a, "epub")["remote_size"] == len(data)
            assert set(_cloud_files(provider, picked)) == files_before
            _assert_pdf_follows(cloud_pdf, book_a / pdf_name, pdf_stamp)

            # =========== 6. a title change keeps the cloud file's name ======================================
            await app.navigate(f"/library/book/{bid_a}/metadata", reset=True)
            assert await _until(lambda: type(app.shell.top_screen).__name__ == "MetadataEditorScreen", 15)
            editor = app.shell.top_screen
            assert await _until(lambda: bool(editor.initial), 15)
            await driver.enter(NEW_TITLE, key="meta-title", timeout=15)
            await driver.tap(key="meta-save", timeout=15)
            assert await _until(lambda: editor.saved is not None, 15)
            meta = json.loads((book_a / "metadata.json").read_text(encoding="utf-8"))
            assert meta.get("title") == NEW_TITLE, meta
            snap = await _compile(app, driver, jobs, bid_a, "out-epub")
            await _cloud_settled(cloud)
            renamed = book_a / f"{NEW_TITLE}.epub"
            assert renamed.is_file(), sorted(p.name for p in book_a.iterdir())  # the phone's copy has the new name
            assert any(_same(p, renamed) for p in (snap.outputs or ())), snap.outputs
            data = renamed.read_bytes()
            files = _cloud_files(provider, picked)
            assert f"{BOOK_A}/{NEW_TITLE}.epub" not in files and set(files) == files_before, set(files)
            assert cloud_epub.name == epub_name and cloud_epub.versions == 4 and bytes(cloud_epub.data) == data
            record = _record(cloud, identity_a, "epub")
            assert record["name"] == epub_name and _record_doc_id(provider, record) == epub_doc
            assert _same(record["source"], renamed), record["source"]

            # =========== 7. per-book Never: not copied; Default again: copied ==============================
            await _set_override(app, driver, bid_a, "never")
            _edit_chapter(chapter, MARKER + " never")
            versions = cloud_epub.versions
            writes_before = len(_writes(native))
            await _compile(app, driver, jobs, bid_a, "out-epub")
            await _cloud_quiet(cloud)
            assert cloud_epub.versions == versions and len(_writes(native)) == writes_before
            assert bytes(cloud_epub.data) != renamed.read_bytes()
            await _set_override(app, driver, bid_a, "default")
            await _cloud_settled(cloud)
            assert cloud_epub.versions == versions + 1 and bytes(cloud_epub.data) == renamed.read_bytes()

            # =========== 8. global switch off: not copied; Send now copies it once =========================
            screen = await _cloud_screen(app)
            await _set_switch(tester, driver, screen, False)
            _edit_chapter(chapter, MARKER + " switch off")
            versions = cloud_epub.versions
            writes_before = len(_writes(native))
            await _compile(app, driver, jobs, bid_a, "out-epub")
            await _cloud_quiet(cloud)
            assert cloud_epub.versions == versions and len(_writes(native)) == writes_before
            await _book_page(app, bid_a)
            await driver.tap(text="Send now", timeout=30)
            await _cloud_settled(cloud)
            assert cloud_epub.versions == versions + 1 and bytes(cloud_epub.data) == renamed.read_bytes()
            _assert_pdf_follows(cloud_pdf, book_a / pdf_name, pdf_stamp)

            # =========== the rest of the contract ==========================================================
            # one cloud document per book and format, nothing outside the picked folder, no replace fallback
            assert set(_all_cloud_files(provider)) == {"someone else's notes.txt"} | {
                f"{PICKED}/{name}" for name in _cloud_files(provider, picked)}
            assert native.doc_calls.count("pick_save_location") == 0 and native.doc_calls.count("delete") == 0
            assert all(doc == epub_doc for doc, _src in _writes(native)
                       if doc not in (cloud_pdf.doc_id, node_b.doc_id)), _writes(native)
            # every write went through a private snapshot in the app cache, never the Library file; none is left
            snapshots = Path(app.paths.cache) / "cloud_sync"
            sources = [src for _doc, src in _writes(native)]
            assert sources and all(_same(Path(s).parent, snapshots) for s in sources), sources
            assert not snapshots.is_dir() or not any(snapshots.iterdir()), list(snapshots.iterdir())
            # progress reached the service through the bridge's document events
            assert native.progress and all(p.get("type") == "progress" for p in native.progress)
            # mobile-only state: Prefs + mobile_cloud.json; config.json gets no cloud key
            await asyncio.to_thread(cloud.store.flush)
            data_dir = Path(app.paths.data)
            records = json.loads((data_dir / "mobile_cloud.json").read_text(encoding="utf-8"))
            assert set(records.get("records") or {}) == {dest.id}, records.get("records")
            config_text = (data_dir / "config.json").read_text(encoding="utf-8") \
                if (data_dir / "config.json").is_file() else "{}"
            config = json.loads(config_text)
            assert not [k for k in config if "cloud" in k.lower()], [k for k in config if "cloud" in k.lower()]
            assert "content://" not in config_text
        finally:
            try:
                app.jobs.close()
            finally:
                await tf._stop(app)
                if cloud is not None:
                    cloud.close()

    with FakeLLMServer() as server:
        asyncio.run(scenario(server))
    others = [a for a in network_guard if not _catalog_poll(a)]
    assert not others, "network access attempted:\n" + "\n".join(f"{a[0]} {a[1]}\n{a[2]}" for a in others[:3])
