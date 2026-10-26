"""Hotfix 3, owner issue 2: "Please add a Library button in my jobs menu, so I can jump to the attachments
library page quickly from it".

The owner's screenshot is the Job page (``/jobs/<jid>``) of a chat EPUB translation: Job card, Running chip,
"Open Chat · <book>", progress, Stop + Progress, Requests. The Job page now has a **Library** button next to
Stop / Progress (also for finished jobs) that opens the Library Book page of the job's book, pushed over the
Job page so Android Back returns there:

* a job started for a Library book (``origin['bid']``: Library / Book page / Reader / Tools jobs of one book,
  a chat's "Save to: Library" run) opens that book at once;
* a chat job over a book attachment is translated in the chat's ``Attachments/<stem>`` folder and the chat
  moves the finished book into the Library by itself (``ChatFeature.auto_migrate``). While it runs the
  button is shown disabled with the chat card's reason ("Added to the Library when the translation
  finishes"), unless its attached file already is a Library book's raw (``chat_view.library_book_for_file``);
  once the run ended it is enabled and a tap looks again (the move happens after the run), finding the book
  through the chat's workspace (``ChatFeature.job_workspace`` + ``ChatFeature.library_book``);
* any other job shows it only when its workspace is a Library book's (a Tools job over a Library workspace):
  a typed-text chat or a Tools job without a Library book has no button.

The Jobs list has no per-job menu (rows only carry Stop / Cancel / Resume / Open buttons), so the button
lives on the Job page only.

Test rule of the hotfix batch: every RUNNING / state assertion is driven with hold gates (a held job run, a
held Library lookup), never with sleeps.

1. Unit tests: ``JobDetailScreen`` on the fake Flet session with a fake job service and recorded async
   lookups (the button's states, the lookups each state runs, a stale lookup answering after the job moved on).
2. ``test_owner_flow_...``: the real app (``main.main``) on the fake Android session with the real JobsFeature,
   ChatFeature, LibraryService (scan) and shell navigation; only the job backend and the job run are stand-ins.

Real data stays untouched: HOME, USERPROFILE, APPDATA, LOCALAPPDATA, GLOSSARION_LIBRARY_DIR, OUTPUT_DIRECTORY,
GLOSSARION_DATA_DIR and CONFIG_FILE point into tmp, GLOSSARION_HTTP_LOG=0, and every test checks that
src/config.json is byte-identical afterwards.

Run from src/mobile with the mobile venv (Python 3.12, keep the shell PYTHONPATH)::

    .venv/Scripts/python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic \
        tests_host/test_hotfix3_job_library.py
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import importlib.util
import json
import os
import queue
import shutil
import sys
import threading
import time
import types
from pathlib import Path

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = MOBILE_DIR / "app"
SRC_DIR = MOBILE_DIR.parent
TESTS_DIR = MOBILE_DIR / "tests"
for _path in (APP_DIR, TESTS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from glossarion_mobile import job_kinds  # noqa: E402
from glossarion_mobile.services import jobs as jobs_service  # noqa: E402
from glossarion_mobile.services.jobs import JOB_TARGET_PARAM, JobSnapshot, JobSpec, JobState  # noqa: E402


def _has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


pytestmark = pytest.mark.skipif(not (_has("flet") and _has("msgpack")), reason="flet / msgpack not installed")


def _load(name: str, filename: str):
    """A sibling test module as a helper module (one copy of the fake session, backend and fixtures)."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


_TB = _load("_glossarion_hf3_joblib_tb_helpers", "test_bootstrap.py")
storage = _TB.storage
app_env = _TB.app_env


def _uf():
    return _load("_glossarion_hf3_joblib_uf_helpers", "test_ui_foundations.py")


def _tj():
    return _load("_glossarion_hf3_joblib_jobs_helpers", "test_jobs.py")


# ==========================================================================
# Isolation
# ==========================================================================


def _md5(path: Path):
    try:
        return hashlib.md5(path.read_bytes()).hexdigest()
    except OSError:
        return None


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """Nothing reads or writes the developer's Library, output folders, home, AppData or src/config.json
    (autouse: runs before ``storage`` / the bootstrap env contract)."""
    iso = tmp_path / "_iso"
    for name, sub in (("HOME", "home"), ("USERPROFILE", "home"), ("APPDATA", "appdata"),
                      ("LOCALAPPDATA", "localappdata"), ("GLOSSARION_LIBRARY_DIR", "Library"),
                      ("OUTPUT_DIRECTORY", "Output"), ("GLOSSARION_DATA_DIR", "data")):
        folder = iso / sub
        folder.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv(name, str(folder))
    monkeypatch.setenv("CONFIG_FILE", str(iso / "config.json"))  # never src/config.json
    monkeypatch.setenv("GLOSSARION_HTTP_LOG", "0")
    config = SRC_DIR / "config.json"
    before = _md5(config)
    yield iso
    assert _md5(config) == before, "src/config.json changed during the test"


def _key(path) -> str:
    """A path as ``services.jobs.job_target`` keeps it (absolute, real, case-folded where the FS is)."""
    return os.path.normcase(os.path.realpath(os.path.abspath(str(path))))


def _same(a, b) -> bool:
    return os.path.normcase(os.path.abspath(str(a))) == os.path.normcase(os.path.abspath(str(b)))


async def _until(predicate, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if predicate():
                return True
        except Exception:
            pass
        await asyncio.sleep(0.05)
    try:
        return bool(predicate())
    except Exception:
        return False


def _wait_reason() -> str:
    from glossarion_mobile.ui.chat.cards import LIBRARY_WAIT_REASON

    return LIBRARY_WAIT_REASON


# ==========================================================================
# 1. JobDetailScreen with a fake job service and recorded lookups
# ==========================================================================


class FakeJobService:
    """What ``JobDetailScreen`` reads from JobService: snapshots, the log, request cards and the view listener
    (``set`` publishes a new snapshot like a job transition)."""

    def __init__(self, *snaps: JobSnapshot) -> None:
        self.snaps = {snap.id: snap for snap in snaps}
        self.listeners: list = []

    def snapshot(self, jid=None):
        return self.snaps.get(jid)

    def log_buffer(self, jid):
        return None

    def read_log_tail(self, jid):
        return []

    def request_segments(self, jid):
        return []

    def has_request_stream(self, jid):
        return False

    def job_log_path(self, jid):
        return ""

    def subscribe(self, listener):
        self.listeners.append(listener)
        return lambda: self.listeners.remove(listener) if listener in self.listeners else None

    def set(self, snap: JobSnapshot) -> None:
        self.snaps[snap.id] = snap
        view = types.SimpleNamespace(find=lambda jid: self.snaps.get(jid))
        for listener in list(self.listeners):
            listener(view)


class Lookups:
    """The screen's injected lookups, recorded: ``library_book`` (workspace folder -> Library id, as
    ``ChatFeature.library_book``), ``library_source`` (raw file -> Library id, as ``chat_view.
    library_book_for_file``) and ``chat_workspace`` (a chat job's persisted workspace). ``hold("source")`` /
    ``hold("folder")`` parks the next lookup of that kind until the test releases it (a hold gate, never a
    sleep); ``entered[kind]`` is set once it is parked."""

    def __init__(self) -> None:
        self.books: dict = {}  # _key(folder) -> bid
        self.raws: dict = {}  # _key(file) -> bid
        self.workspace = ""
        self.folder_calls: list = []
        self.source_calls: list = []
        self._held: dict = {"source": [], "folder": []}
        self.entered: dict = {}

    def hold(self, kind: str) -> asyncio.Event:
        gate = asyncio.Event()
        self._held[kind].append(gate)
        self.entered[kind] = asyncio.Event()
        return gate

    async def _gate(self, kind: str) -> None:
        if self._held[kind]:
            gate = self._held[kind].pop(0)
            self.entered[kind].set()
            await gate.wait()

    async def library_book(self, folder: str):
        self.folder_calls.append(folder)
        await self._gate("folder")
        return self.books.get(_key(folder))

    async def library_source(self, path: str):
        self.source_calls.append(path)
        await self._gate("source")
        return self.raws.get(_key(path))

    def chat_workspace(self, snap) -> str:
        return self.workspace


def _snap(kind: str, *, state=JobState.RUNNING, inputs=(), params=None, origin=None, output_dir=None,
          jid="job000000001") -> JobSnapshot:
    started = None if state is JobState.QUEUED else 2.0
    return JobSnapshot(id=jid, spec=JobSpec(kind, "Novel.epub", tuple(str(p) for p in inputs), params or {},
                                            origin or {}), state=state, created=1.0, started=started,
                       output_dir=output_dir)


def _book(tmp_path: Path, name: str = "Novel.epub") -> Path:
    path = tmp_path / "in" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"PK fake epub")
    return path


class Harness:
    """A Job page on the fake Android session: taps record their navigation and snackbars."""

    def __init__(self, service: FakeJobService, jid: str, lookups: Lookups) -> None:
        from glossarion_mobile.ui.router import parse_route
        from glossarion_mobile.ui.screens.job_detail import JobDetailScreen

        self.navigated: list = []
        self.notes: list = []
        _conn, session = _TB._fake_session("android")
        self.page = session.page
        self.detail = JobDetailScreen(parse_route(f"/jobs/{jid}"), service=service, page=self.page,
                                      navigate=lambda name, params=None: self.navigated.append((name, params)),
                                      notify=self.notes.append, chat_workspace=lookups.chat_workspace,
                                      library_book=lookups.library_book, library_source=lookups.library_source)
        self.page.views[0].controls.append(self.detail.get_body())
        self.page.update()
        self.detail.did_show()

    async def settle(self) -> None:
        """Wait for the Library lookup the last job state started (it may be None: an origin book)."""
        task = self.detail._library_task
        if task is not None:
            await task

    @property
    def button(self):
        return self.detail.library_button

    @property
    def chip(self):
        return self.detail.library_reason

    def state(self) -> str:
        if not self.button.visible:
            return "hidden"
        if self.button.disabled:
            return "waiting" if self.chip.visible else "disabled"
        return "enabled"


def test_a_job_for_a_library_book_opens_its_book_page_at_once(tmp_path):
    """Library / Book page / Reader / Tools jobs of one book and a chat's "Save to: Library" run carry the book
    in their origin: the button is enabled from the first frame (running or finished) and needs no lookup."""
    raw = _book(tmp_path)

    async def scenario():
        for origin in ({"type": "library", "bid": "0123456789ab", "label": "Library · Novel"},
                       {"type": "chat", "cid": "4", "bid": "0123456789ab", "label": "Chat · Novel"}):
            for state in (JobState.QUEUED, JobState.RUNNING, JobState.DONE, JobState.CANCELLED):
                lookups = Lookups()
                snap = _snap("translate", state=state, inputs=(raw,), origin=origin)
                h = Harness(FakeJobService(snap), snap.id, lookups)
                assert h.state() == "enabled", (origin, state)
                assert h.button.content == "Library" and not h.chip.visible
                await h.settle()
                assert lookups.folder_calls == [] and lookups.source_calls == [], (origin, state)
                assert await h.detail._on_library() == "0123456789ab"
                assert h.navigated == [("library.book", {"bid": "0123456789ab"})], (origin, state)
                h.detail.dispose()

    asyncio.run(scenario())


def test_a_chat_book_waits_while_it_runs_and_opens_the_moved_book_after_the_run(tmp_path):
    """The owner's job: a chat EPUB translation (``direct_text``, the attached file named by ``job_target``).
    Running: the button waits (disabled + the chat card's reason) and only the attached file is looked up (its
    workspace is in the chat's Attachments folder). Done before the chat moved the book: enabled, nothing found
    yet. The chat moves it (the workspace is now a Library folder): the tap looks again and opens the book."""
    raw = _book(tmp_path)
    run_input = tmp_path / "run" / "Novel.epub"
    attachments = tmp_path / "Output" / "Direct Text" / "Chat" / "Attachments" / "novel"
    moved = tmp_path / "Output" / "novel"
    for folder in (run_input.parent, attachments, moved):
        folder.mkdir(parents=True, exist_ok=True)
    shutil.copy(raw, run_input)

    async def scenario():
        lookups = Lookups()
        lookups.workspace = str(attachments)
        snap = _snap("direct_text", inputs=(run_input,), output_dir=str(tmp_path / "run"),
                     params={JOB_TARGET_PARAM: [str(raw)], "user_index": 0, "is_attachment": True},
                     origin={"type": "chat", "cid": "4", "label": "Chat · Novel"})
        service = FakeJobService(snap)
        h = Harness(service, snap.id, lookups)
        assert h.state() == "waiting"
        assert h.chip.reason == _wait_reason() == "Added to the Library when the translation finishes"
        await h.settle()
        assert h.state() == "waiting"
        assert [_key(p) for p in lookups.source_calls] == [_key(raw)]  # the attached file, never the run input
        assert lookups.folder_calls == []  # still in Attachments: no Library rescan for it
        # the job ends; the chat has not moved the book yet
        service.set(dataclasses.replace(snap, state=JobState.DONE))
        await h.settle()
        assert h.state() == "enabled"
        assert [_key(p) for p in lookups.folder_calls] == [_key(attachments)]  # never the run root
        # the chat moved the finished book into the Library: the tap looks again and opens it
        lookups.workspace = str(moved)
        lookups.books[_key(moved)] = "feedfacecafe"
        assert await h.detail._on_library() == "feedfacecafe"
        assert h.navigated == [("library.book", {"bid": "feedfacecafe"})]
        assert h.state() == "enabled" and h.detail._library_found == "feedfacecafe"
        h.detail.dispose()

        # a stopped / failed chat book stays in Attachments (Resume continues it there): the tap says why
        for state, note in ((JobState.CANCELLED, _wait_reason()), (JobState.FAILED, _wait_reason()),
                            (JobState.DONE, "This book is not in the Library yet")):
            lookups = Lookups()
            lookups.workspace = str(attachments)
            ended = dataclasses.replace(snap, state=state)
            h = Harness(FakeJobService(ended), ended.id, lookups)
            await h.settle()
            assert h.state() == "enabled", state
            assert await h.detail._on_library() is None
            assert h.navigated == [] and h.notes == [note], state
            h.detail.dispose()

    asyncio.run(scenario())


def test_a_chat_attachment_that_is_a_library_book_opens_it_while_running(tmp_path):
    """A chat run of a file that is a Library book's raw (sent with "Save to: Chat"): the book is on the shelf
    already, so the running job's button is enabled once the file lookup found it."""
    raw = _book(tmp_path)

    async def scenario():
        lookups = Lookups()
        lookups.raws[_key(raw)] = "abcdefabcdef"
        snap = _snap("direct_text", inputs=(tmp_path / "run.epub",), params={JOB_TARGET_PARAM: [str(raw)]},
                     origin={"type": "chat", "cid": "4", "label": "Chat · Novel"})
        h = Harness(FakeJobService(snap), snap.id, lookups)
        assert h.state() == "waiting"
        await h.settle()
        assert h.state() == "enabled" and not h.chip.visible
        assert await h.detail._on_library() == "abcdefabcdef"
        assert h.navigated == [("library.book", {"bid": "abcdefabcdef"})]
        h.detail.dispose()

    asyncio.run(scenario())


def test_a_stale_lookup_never_overrides_the_next_job_state(tmp_path):
    """Hold gate: the running chat book's file lookup is parked; meanwhile the job ends and the done state's
    lookup finds the moved book. The parked lookup answers last (nothing) and must not hide the book."""
    raw = _book(tmp_path)
    moved = tmp_path / "Output" / "novel"
    moved.mkdir(parents=True)

    async def scenario():
        lookups = Lookups()
        gate = lookups.hold("source")
        snap = _snap("direct_text", inputs=(tmp_path / "run.epub",), params={JOB_TARGET_PARAM: [str(raw)]},
                     origin={"type": "chat", "cid": "4", "label": "Chat · Novel"})
        service = FakeJobService(snap)
        h = Harness(service, snap.id, lookups)
        running_lookup = h.detail._library_task
        await asyncio.wait_for(lookups.entered["source"].wait(), 5)  # parked inside the running state's lookup
        lookups.workspace = str(moved)
        lookups.books[_key(moved)] = "0badc0ffee00"
        service.set(dataclasses.replace(snap, state=JobState.DONE))
        done_lookup = h.detail._library_task
        assert done_lookup is not running_lookup
        await done_lookup
        assert h.state() == "enabled" and h.detail._library_found == "0badc0ffee00"
        gate.set()
        assert await running_lookup is None  # the parked lookup answered: nothing for the running state
        assert h.state() == "enabled" and h.detail._library_found == "0badc0ffee00"
        assert await h.detail._on_library() == "0badc0ffee00"
        h.detail.dispose()

    asyncio.run(scenario())


def test_a_tap_during_the_states_own_lookup_waits_for_it_then_looks_again(tmp_path):
    """Hold gate: the finished chat book's own lookup is parked inside its workspace lookup (a Library rescan
    in the app) when the owner taps Library, and the chat has moved the book meanwhile. The tap starts no
    second lookup while that one runs (``ChatFeature.library_book`` would not wait for the running scan and
    answer from the stale shelf); once it answered nothing, the tap looks again and opens the moved book."""
    raw = _book(tmp_path)
    attachments = tmp_path / "Output" / "Direct Text" / "Chat" / "Attachments" / "novel"
    moved = tmp_path / "Output" / "novel"
    for folder in (attachments, moved):
        folder.mkdir(parents=True)

    async def scenario():
        lookups = Lookups()
        lookups.workspace = str(attachments)
        gate = lookups.hold("folder")
        snap = _snap("direct_text", state=JobState.DONE, inputs=(tmp_path / "run.epub",),
                     params={JOB_TARGET_PARAM: [str(raw)]}, origin={"type": "chat", "cid": "4", "label": "Chat"})
        h = Harness(FakeJobService(snap), snap.id, lookups)
        own_lookup = h.detail._library_task
        await asyncio.wait_for(lookups.entered["folder"].wait(), 5)
        assert h.state() == "enabled"
        lookups.workspace = str(moved)  # the chat moved the book while the lookup was parked
        lookups.books[_key(moved)] = "1234abcd5678"
        tap = asyncio.ensure_future(h.detail._on_library())
        done, _pending = await asyncio.wait({tap}, timeout=1.0)  # a bound on a negative check only
        assert not done and lookups.folder_calls == [str(attachments)], lookups.folder_calls
        gate.set()
        assert await tap == "1234abcd5678"
        assert own_lookup.done() and [_key(p) for p in lookups.folder_calls] == [_key(attachments), _key(moved)]
        assert h.navigated == [("library.book", {"bid": "1234abcd5678"})] and h.notes == []
        h.detail.dispose()

    asyncio.run(scenario())


def test_a_double_tap_shares_one_lookup(tmp_path):
    """Hold gate: two quick taps on a finished chat book whose own lookup found nothing (the chat moved it
    afterwards). The second tap waits for the first tap's lookup (parked) instead of starting its own; both
    open the book that one lookup found, and neither says it is missing."""
    raw = _book(tmp_path)
    attachments = tmp_path / "Output" / "Direct Text" / "Chat" / "Attachments" / "novel"
    moved = tmp_path / "Output" / "novel"
    for folder in (attachments, moved):
        folder.mkdir(parents=True)

    async def scenario():
        lookups = Lookups()
        lookups.workspace = str(attachments)
        snap = _snap("direct_text", state=JobState.DONE, inputs=(tmp_path / "run.epub",),
                     params={JOB_TARGET_PARAM: [str(raw)]}, origin={"type": "chat", "cid": "4", "label": "Chat"})
        h = Harness(FakeJobService(snap), snap.id, lookups)
        await h.settle()
        assert h.detail._library_found is None and lookups.folder_calls == [str(attachments)]
        lookups.workspace = str(moved)
        lookups.books[_key(moved)] = "5555aaaa5555"
        gate = lookups.hold("folder")
        first = asyncio.ensure_future(h.detail._on_library())
        await asyncio.wait_for(lookups.entered["folder"].wait(), 5)  # the first tap's lookup is parked
        second = asyncio.ensure_future(h.detail._on_library())
        done, _pending = await asyncio.wait({second}, timeout=1.0)  # a bound on a negative check only
        assert not done and len(lookups.folder_calls) == 2, lookups.folder_calls
        gate.set()
        assert await first == "5555aaaa5555" and await second == "5555aaaa5555"
        assert len(lookups.folder_calls) == 2 and len(lookups.source_calls) == 1
        assert h.navigated == [("library.book", {"bid": "5555aaaa5555"})] * 2 and h.notes == []
        h.detail.dispose()

    asyncio.run(scenario())


def test_jobs_without_a_library_book_have_no_library_button(tmp_path):
    """A typed-text chat (its input is a temporary .txt of the run root, no ``job_target``) and a Tools job
    whose workspace is no Library book have no button; a Tools job over a Library book's workspace shows it
    once the lookup found the book."""
    library_folder = tmp_path / "Output" / "Novel"
    other_folder = tmp_path / "elsewhere" / "Scan"
    for folder in (library_folder, other_folder):
        folder.mkdir(parents=True)
    temp_input = tmp_path / "run" / "direct_text_20261026_ab12cd34.txt"
    temp_input.parent.mkdir(parents=True)
    temp_input.write_text("hello", encoding="utf-8")

    async def scenario():
        lookups = Lookups()
        text = _snap("direct_text", inputs=(temp_input,), output_dir=str(temp_input.parent),
                     origin={"type": "chat", "cid": "4", "label": "Chat · Hello"})
        for state in (JobState.RUNNING, JobState.DONE):
            h = Harness(FakeJobService(dataclasses.replace(text, state=state)), text.id, lookups)
            await h.settle()
            assert h.state() == "hidden", state
            h.detail.dispose()
        assert lookups.source_calls == [] and lookups.folder_calls == []  # nothing to look up

        lookups.books[_key(library_folder)] = "c0c0c0c0c0c0"
        for folder, expected in ((other_folder, "hidden"), (library_folder, "enabled")):
            scan = _snap("qa_scan", state=JobState.DONE, inputs=(folder,), output_dir=str(folder),
                         origin={"type": "tool", "route": "tools.qa", "label": "Tools · QA scanner"})
            h = Harness(FakeJobService(scan), scan.id, lookups)
            assert h.state() == "hidden"  # until its lookup answered
            await h.settle()
            assert h.state() == expected, folder
            if expected == "enabled":
                assert await h.detail._on_library() == "c0c0c0c0c0c0"
                assert h.navigated == [("library.book", {"bid": "c0c0c0c0c0c0"})]
            h.detail.dispose()

    asyncio.run(scenario())


# ==========================================================================
# 2. The owner's flow in the real app
# ==========================================================================


class HeldRuns:
    """The ``run`` of the job kinds under test, replaced: each run announces itself on ``started`` with its
    own release gate and waits until the test releases it (or the job is stopped)."""

    def __init__(self, monkeypatch, kinds) -> None:
        self.started: queue.Queue = queue.Queue()
        self.gates: list = []
        for kind in kinds:
            info = job_kinds.get_kind(kind)  # the real KindInfo: verb, icon, stop kind stay real
            monkeypatch.setitem(job_kinds._CACHE, kind, dataclasses.replace(info, run=self._run))

    def _run(self, ctx):
        release = threading.Event()
        self.gates.append(release)
        ctx.log("🚀 Starting translation")
        self.started.put((ctx.spec.title, release))
        while not release.wait(0.02):
            if ctx.stop_requested():
                break
        return None

    async def next(self, timeout: float = 15.0) -> threading.Event:
        _title, release = await asyncio.to_thread(self.started.get, True, timeout)
        return release

    def release_all(self) -> None:
        for gate in self.gates:
            gate.set()


def _fake_backend(monkeypatch, tmp_path) -> None:
    """The app's JobService is built with ``test_jobs.FakeBackend`` (no HeadlessOwner, no shared pipeline);
    JobsFeature, ChatFeature, the Library and the shell stay real."""
    tj = _tj()
    monkeypatch.setattr(jobs_service, "JobBackend", lambda: tj.FakeBackend(str(tmp_path / "job-out")))


async def _start_app():
    from host_tester import PyTester
    from ui_driver import UiDriver

    uf = _uf()
    tj = _tj()
    _main, conn, session, page, app = await uf._start("android")
    assert app.jobs is not None and type(app.job_service.backend).__name__ == "FakeBackend"
    app.jobs.background.permissions = tj.FakePermissions()  # no permission prompt on the fake session

    async def confirm(*_args, **_kwargs):
        return True

    app.jobs.background.confirm = confirm
    ui = UiDriver(PyTester(session, page), log=lambda _line: None, poll_ms=50)
    return types.SimpleNamespace(uf=uf, conn=conn, session=session, page=page, app=app, ui=ui)


async def _stop_app(shell, runs=None) -> None:
    app = shell.app
    if runs is not None:
        runs.release_all()
    service = getattr(app, "job_service", None)
    if service is not None:
        await asyncio.to_thread(service.wait_idle, 15)
    reader = getattr(app, "reader", None)
    if reader is not None and hasattr(reader, "detach"):
        try:
            reader.detach()
        except Exception:
            pass
    if getattr(app, "jobs", None) is not None:
        app.jobs.close()
    if getattr(app, "settings", None) is not None:
        app.settings.close()
    await shell.uf._stop(app)


def _library_workspace(output_root: Path, raw: Path) -> Path:
    """The chat's book after the move into the Library: an output-root workspace with its progress file
    (chapter 1 translated) and its raw source."""
    workspace = output_root / "Novel"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "source_epub.txt").write_text(str(raw), encoding="utf-8")
    (workspace / "response_chapter0001.html").write_text(
        "<html><head><title>Chapter One</title></head><body><h1>Chapter One</h1><p>Rain.</p></body></html>",
        encoding="utf-8")
    progress = {"chapters": {"1": {"status": "completed", "original_basename": "chapter0001.xhtml",
                                   "output_file": "response_chapter0001.html"},
                             "2": {"status": "pending", "original_basename": "chapter0002.xhtml"}},
                "chapter_chunks": {}, "version": "2.1"}
    (workspace / "translation_progress.json").write_text(json.dumps(progress), encoding="utf-8")
    return workspace


def _chat_job(raw: Path, run_input: Path, cid: str, user_index: int) -> JobSpec:
    """What ``ChatRuns.send`` submits for an attachment: the run root's input, the attached file as its target."""
    return JobSpec("direct_text", raw.name, (str(run_input),),
                   params={JOB_TARGET_PARAM: [str(raw)], "user_index": user_index, "is_attachment": True},
                   origin={"type": "chat", "cid": cid, "label": f"Chat · {raw.name}"})


@pytest.mark.skipif(not _has("ebooklib"), reason="ebooklib not installed")
def test_owner_flow_chat_epub_job_library_button_opens_the_book_and_back_returns(app_env, tmp_path, monkeypatch):
    from glossarion_mobile.diagnostics.fixtures import build_tiny_epub
    from glossarion_mobile.ui.library.book_page import BookPageScreen
    from glossarion_mobile.ui.screens.job_detail import JobDetailScreen

    runs = HeldRuns(monkeypatch, ("direct_text",))
    _fake_backend(monkeypatch, tmp_path)
    raw = tmp_path / "Inbox" / "[211151] Novel.epub"
    raw.parent.mkdir(parents=True)
    build_tiny_epub(raw, chapters=3)
    run_input = tmp_path / "runs" / "r1" / raw.name
    run_input.parent.mkdir(parents=True)
    shutil.copy(raw, run_input)

    async def scenario():
        shell = await _start_app()
        app, ui, session, page, service = shell.app, shell.ui, shell.session, shell.page, shell.app.job_service
        try:
            chats = app.chat_feature.chats
            cid = chats.new_chat()
            chats.append_messages(cid, [("user_file", raw.name, str(raw), raw.stat().st_size, "", "user")])

            # --- the owner's screenshot: the chat EPUB translation is running ---
            jid = service.submit(_chat_job(raw, run_input, cid, 0))
            release = await runs.next()
            await shell.uf._route(session, f"/jobs/{jid}")
            assert await _until(lambda: isinstance(app.shell.top_screen, JobDetailScreen))
            detail = app.shell.top_screen
            assert detail.job_id == jid and detail.library_book is not None and detail.library_source is not None
            assert await _until(lambda: detail.snap.state is JobState.RUNNING and detail.stop_button.visible)
            assert await _until(lambda: detail.library_button.visible and detail.library_button.disabled
                                and detail.library_reason.visible)
            assert detail.library_reason.reason == "Added to the Library when the translation finishes"

            # --- the run ends; the chat has not moved the book yet: enabled, the tap looks again ---
            release.set()
            assert await _until(lambda: service.snapshot(jid).is_terminal, 15)
            assert service.snapshot(jid).state is JobState.DONE, service.snapshot(jid).error
            assert await _until(lambda: detail.snap.state is JobState.DONE and detail.library_button.visible
                                and not detail.library_button.disabled and not detail.library_reason.visible)
            assert await _until(lambda: detail._library_task is not None and detail._library_task.done())
            assert detail._library_found is None  # the done state's own lookup: not on the shelf yet

            # --- the chat moved the finished book into the Library (its response now points there) ---
            workspace = _library_workspace(Path(app.paths.output), raw)
            chats.append_messages(cid, [("assistant", "Translated", "", "", str(workspace), raw.name)])
            app.library.mark_dirty()  # what ChatFeature._announce_moved does after a move

            await ui.tap(key="job-library", timeout=10)
            assert await _until(lambda: isinstance(app.shell.top_screen, BookPageScreen), 20), \
                type(app.shell.top_screen).__name__
            book_page = app.shell.top_screen
            assert _same(book_page.workspace, workspace), book_page.book
            bid = book_page.bid
            assert detail._library_found == bid
            assert [entry.match.name for entry in app.shell.stack][-2:] == ["jobs.detail", "library.book"]

            # --- Android Back returns to the Job page ---
            await session.dispatch_event(page._i, "view_pop", {"route": page.views[-1].route})
            assert await _until(lambda: isinstance(app.shell.top_screen, JobDetailScreen)
                                and app.shell.top_screen.job_id == jid)
            assert app.shell.current_route == f"/jobs/{jid}"

            # --- the same attachment again: its file is that Library book's raw, so the running job opens it ---
            assert await app.jobs._library_book_of_file(str(raw)) == bid
            chats.append_messages(cid, [("user_file", raw.name, str(raw), raw.stat().st_size, "", "user")])
            second = service.submit(_chat_job(raw, run_input, cid, 2))
            release = await runs.next()
            await app.navigate(f"/jobs/{second}")
            assert await _until(lambda: isinstance(app.shell.top_screen, JobDetailScreen)
                                and app.shell.top_screen.job_id == second)
            running = app.shell.top_screen
            assert await _until(lambda: running.snap.state is JobState.RUNNING and running.library_button.visible
                                and not running.library_button.disabled and not running.library_reason.visible)
            await ui.tap(key="job-library", timeout=10)
            assert await _until(lambda: isinstance(app.shell.top_screen, BookPageScreen)
                                and app.shell.top_screen.bid == bid, 20)
            await session.dispatch_event(page._i, "view_pop", {"route": page.views[-1].route})
            assert await _until(lambda: isinstance(app.shell.top_screen, JobDetailScreen)
                                and app.shell.top_screen.job_id == second)
            release.set()
            assert await _until(lambda: service.snapshot(second).is_terminal, 15)
        finally:
            await _stop_app(shell, runs)

    asyncio.run(scenario())
