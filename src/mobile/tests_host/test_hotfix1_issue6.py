"""Acceptance test, hotfix 1 owner issue 6: "Post QA scan is not running".

The owner's job log on the phone (Library book, scanning phase on)::

    ✅ Translation completed successfully!
    🧪 Scanning phase enabled — launching QA Scanner in quick-scan mode...
    ⚠️ The post-translation QA scan is not available in this build yet; skipped
    ■ Done: [TS] I'm the Only Mechanical Genius Girl

This file replays the owner's steps on the REAL app objects: ``main.main`` on the fake Flet session
(``test_ui_foundations._start`` through ``test_ui_flows._host_driver``, as the other tests_host UI tests), the
real Library (Import EPUB, the Book page with its Chapters tab - the Progress Manager - and its Output tab),
the real JobService with HeadlessOwner and the shared translation pipeline, ``job_kinds.translate`` /
``job_kinds.qa``, the shared ``qa_scan_runtime`` and the real ``scan_html_folder`` (thread executor). Only the
model is the offline fake OpenAI server on 127.0.0.1; the E2E's network guard refuses every other connection.
The book is the 12-chapter self-test EPUB under the owner's file name ("[TS] I'm the Only Mechanical Genius
Girl.epub": brackets and an apostrophe in every path); the fake model leaves part of chapter 5 in Korean, so
the scan has something to flag. The scanning phase is on, Quick Scan (the desktop default).

1. ``test_the_scan_runs_in_the_translation_job_and_is_surfaced_like_a_qa_job``: Library > Import EPUB > the
   book > Chapters > Run translation > Start. The scan starts inside the translation's own job (the gl-job
   thread, the job still RUNNING in its "Scanning" phase, no second job queued) on the book's workspace with
   its raw source, Quick Scan, sample size 0; both run buttons wait meanwhile. The job ends Done after the
   scan; the log reads completion -> scanning phase -> scanner initialized -> folder -> scan -> completed ->
   "■ Done: <title>" and never "not available". The report (HTML + JSON) is in the workspace; the job lists it
   after the compiled EPUB and carries ``qa_reports`` / ``qa_mode`` / ``qa_folders``; chapter 5 is ``qa_failed``
   in translation_progress.json and in the Chapters tab rows. Surfaced like a QA job: the Jobs page shows the
   report among the job's outputs and "1 QA failed"; Book > Output lists the QA report and opens it in the QA
   report viewer; then a Run QA scan job (Chapters tab) of the same workspace writes the same report, the same
   result fields and the same chapter marks.
2. ``test_stop_during_the_scan_stops_the_job`` (two stop points: as the real scanner starts, and inside it
   after it read the source EPUB and logged its file count): the same run; while the scan runs, the Jobs
   page's Stop. The job turns Stopping at once and, when the scanner continues, the real scanner sees the Stop
   ("⛔ Stop requested via GUI stop button", "⛔ QA scan interrupted before processing."); the job ends Stopped
   (no error, no traceback, no report, no "Scan complete") within seconds, every chapter stays translated, and
   a Run QA scan afterwards runs normally (the stop flags were cleared).

Hold gate (the batch's test rule: no sleeps decide a Stop or a RUNNING assertion): ``ScanGate`` wraps the real
scanner (``scan_html_folder.scan_html_folder``, which ``qa_scan_runtime.run_qa_scan_path`` imports at call
time) and parks the job thread at its entry, or inside it on one of its own log lines, until the test releases
it. The scan cannot end while held, so "still running" and "Stop lands during the scan" do not depend on the
runner's speed; released, the real scanner runs (on) unchanged. The model needs no gate (the scan sends no
request). The Start tap runs as its own task (a device tap is never waited on), so the first long job's
"Keep translations running" question can be answered whenever it comes.

Debugging aid: ``GLOSSARION_TEST_TRACE=1`` (with ``-s``) prints the UI steps and the scenario's stages.

Real data stays untouched: the app runs in temp FLET_APP_STORAGE_* dirs (the bootstrap points HOME,
OUTPUT_DIRECTORY, GLOSSARION_LIBRARY_DIR, GLOSSARION_DATA_DIR and CONFIG_FILE there; asserted), USERPROFILE /
APPDATA / LOCALAPPDATA point into tmp_path and HTTP logging is off. A failing scenario releases the gate and
force-stops its job before the app closes.

Run from src/mobile with the mobile venv (Python 3.12, keep the shell PYTHONPATH)::

    .venv/Scripts/python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic \
        tests_host/test_hotfix1_issue6.py
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
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
for _entry in (str(APP_DIR), str(SRC_DIR), str(TESTS_DIR)):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

SELFTEST_EPUB = APP_DIR / "assets" / "selftest" / "selftest_ko_12ch.epub"  # tools/prepare_assets.py
TIKTOKEN_ASSETS = APP_DIR / "assets" / "tiktoken"


def _has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _tiktoken_assets() -> bool:
    return TIKTOKEN_ASSETS.is_dir() and any(p.is_file() and not p.suffix for p in TIKTOKEN_ASSETS.iterdir())


_NEEDED = ("flet", "msgpack", "ebooklib", "openai", "httpx", "tiktoken", "bs4", "lxml")
pytestmark = [
    pytest.mark.skipif(not all(_has(m) for m in _NEEDED), reason=f"needs {', '.join(_NEEDED)} (the project venv)"),
    pytest.mark.skipif(not SELFTEST_EPUB.is_file() or not _tiktoken_assets(),
                       reason="app/assets/selftest + app/assets/tiktoken are generated by tools/prepare_assets.py"),
]


def _load(alias: str, file_name: str):
    spec = importlib.util.spec_from_file_location(alias, Path(__file__).with_name(file_name))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_UF = _load("_glossarion_uiflows_hotfix1_issue6", "test_ui_flows.py")  # _host_driver / _foundations
storage = _UF.storage
app_env = _UF.app_env

#: The owner's book (the job title in the owner's log is the Library book's name: the file stem).
BOOK_FILE = "[TS] I'm the Only Mechanical Genius Girl.epub"
BOOK_STEM = BOOK_FILE[:-len(".epub")]
CHAPTERS = 12
UNSHARED_LINE = "⚠️ The post-translation QA scan is not available in this build yet; skipped"
TRIGGER_LINE = "🧪 Scanning phase enabled — launching QA Scanner in quick-scan mode..."
RUN_TIMEOUT = 300.0
#: The gate parks the job thread at most this long (a failing test releases it in its teardown).
GATE_TIMEOUT = 240.0


# ==========================================================================
# Fixtures and helpers
# ==========================================================================


@pytest.fixture
def iso(tmp_path, monkeypatch, request):
    """The app on temp storage, the Windows profile variables redirected as well, the owner's book in picks."""
    user = tmp_path / "user"
    for key, sub in (("USERPROFILE", "."), ("APPDATA", "AppData/Roaming"), ("LOCALAPPDATA", "AppData/Local")):
        folder = (user / sub).resolve()
        folder.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv(key, str(folder))
    monkeypatch.setenv("GLOSSARION_HTTP_LOG", "0")
    request.getfixturevalue("app_env")  # FLET_APP_STORAGE_* + bootstrap (HOME, OUTPUT_DIRECTORY, Library, config)
    picks = tmp_path / "picks"
    picks.mkdir()
    book = picks / BOOK_FILE
    shutil.copy2(SELFTEST_EPUB, book)
    return types.SimpleNamespace(tmp=tmp_path, picks=picks, book=book)


@pytest.fixture
def offline():
    """The E2E's network guard (``diagnostics.e2e.ProcessGuards``): every non-loopback connect / DNS lookup
    fails at once and is recorded."""
    import traceback

    from glossarion_mobile.diagnostics.e2e import ProcessGuards

    guards = ProcessGuards()
    record = guards._record

    def record_with_origin(bucket, **event):
        event["origin"] = [f"{os.path.relpath(f.filename, SRC_DIR)}:{f.lineno} {f.name}"
                           for f in traceback.extract_stack()[:-2] if _inside(f.filename, SRC_DIR)][-12:]
        record(bucket, **event)

    guards._record = record_with_origin
    guards._install_network_guard()
    try:
        yield guards
    finally:
        guards.uninstall()


@pytest.fixture
def seeded_langdetect():
    """QA's language detection is randomised; the E2E seeds it the same way (``e2e._seed_langdetect``)."""
    from glossarion_mobile.diagnostics import e2e

    previous = e2e._seed_langdetect(0)
    try:
        yield
    finally:
        e2e._seed_langdetect(previous)


def _catalog_poll(event: dict) -> bool:
    """The selected model's 24 h provider-catalog auto-poll (desktop parity, a ``model-catalog`` thread): blocked
    like every other non-loopback attempt, and not part of the run."""
    return any("_fetch_provider_catalog" in frame for frame in event.get("origin") or [])


def _assert_offline(guards) -> None:
    others = [e for e in guards.network if not _catalog_poll(e)]
    lines = []
    for event in others[:3]:
        lines.append(f"{event.get('api')} {event.get('target')} [{event.get('thread')}]")
        lines.extend(f"    {frame}" for frame in (event.get("origin") or []) + (event.get("stack") or []))
    assert not others, "\n".join(lines)


def _inside(path, root) -> bool:
    try:
        a = os.path.normcase(os.path.abspath(str(path)))
        b = os.path.normcase(os.path.abspath(str(root)))
        return os.path.commonpath([a, b]) == b
    except ValueError:
        return False


def _same(a, b) -> bool:
    return os.path.normcase(os.path.abspath(str(a))) == os.path.normcase(os.path.abspath(str(b)))


def _assert_isolated(tmp) -> None:
    for key in ("HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA", "OUTPUT_DIRECTORY", "GLOSSARION_LIBRARY_DIR",
                "GLOSSARION_DATA_DIR", "CONFIG_FILE"):
        value = os.environ.get(key)
        assert value and _inside(value, tmp), f"{key}={value!r} is not inside the test's tmp dir"
    assert os.environ.get("GLOSSARION_HTTP_LOG") == "0"


async def _until(predicate, timeout: float = 30.0, step: float = 0.05):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        await asyncio.sleep(step)
    return predicate()


class ScanGate:
    """The real scanner, wrapped: every scan is recorded and, while ``armed``, the scanning thread parks until
    :meth:`release` (or ``GATE_TIMEOUT``). ``hold_on=None``: at the scanner's entry, before it runs; a text: inside
    the scanner, right after it logged a line starting with that text (its own progress). Released, the real
    ``scan_html_folder`` runs (on) unchanged."""

    def __init__(self, module, hold_on=None) -> None:
        self.module = module
        self.real = module.scan_html_folder
        self.hold_on = hold_on
        self.calls: list = []
        self.armed = True
        self.entered = threading.Event()
        self._release = threading.Event()

    def _park(self, call: dict) -> None:
        call["held"] = True
        call["held_thread"] = threading.current_thread().name
        self.entered.set()
        call["released"] = self._release.wait(GATE_TIMEOUT)
        call["t_released"] = time.monotonic()

    def __call__(self, folder_path, *args, **kwargs):
        settings = kwargs.get("qa_settings") or {}
        call = {"folder": str(folder_path), "mode": kwargs.get("mode"), "epub": kwargs.get("epub_path"),
                "sample": settings.get("quick_scan_sample_size"), "thread": threading.current_thread().name,
                "thread_executor": os.environ.get("QA_USE_THREAD_EXECUTOR"), "held": False, "held_thread": None,
                "released": None, "returned": False, "result": None, "error": None, "t_released": None}
        self.calls.append(call)
        if self.armed and self.hold_on is None:
            self._park(call)
        elif self.armed:
            real_log = kwargs.get("log", print)

            def log(message, *a, **k):
                real_log(message, *a, **k)
                if self.armed and not call["held"] and str(message).startswith(self.hold_on):
                    self._park(call)

            kwargs["log"] = log
        try:
            result = self.real(folder_path, *args, **kwargs)
        except BaseException as exc:
            call["error"] = f"{type(exc).__name__}: {exc}"
            raise
        call["returned"] = True
        call["result"] = repr(result)[:200]
        return result

    def release(self) -> None:
        self._release.set()


def _install_gate(monkeypatch, hold_on=None) -> ScanGate:
    """The gate on the module the app's shared runtime imports (the repo ``src/``, never the bundle copy)."""
    import qa_scan_runtime
    import scan_html_folder

    assert _same(Path(qa_scan_runtime.__file__).parent, SRC_DIR), qa_scan_runtime.__file__
    assert _same(Path(scan_html_folder.__file__).parent, SRC_DIR), scan_html_folder.__file__
    gate = ScanGate(scan_html_folder, hold_on=hold_on)
    monkeypatch.setattr(scan_html_folder, "scan_html_folder", gate)
    return gate


def _configure(app, server) -> None:
    """The owner's phone: the fake OpenAI endpoint + a dummy key (``flows.ui_config``: no glossary, English, no
    request spacing) and the scanning phase on in Quick Scan mode (the desktop default)."""
    import flows
    from glossarion_mobile.diagnostics.fake_llm_server import FAKE_MODEL

    values = flows.ui_config(server.url, FAKE_MODEL)
    values.update(scan_phase_enabled=True, scan_phase_mode="quick-scan")
    store = app.config_store
    store.set_many(values)
    store.flush()
    assert store.save_error is None, store.save_error
    assert store.get("scan_phase_enabled") is True and store.get("scan_phase_mode") == "quick-scan"


def _log(app, job_id: str) -> list:
    return list(app.job_service.read_log_tail(job_id, 20000))


def _index(lines: list, start: str, after: int = -1) -> int:
    return next((i for i, line in enumerate(lines) if i > after and str(line).startswith(start)), -1)


def _in_order(lines: list, starts) -> list:
    """Each start's first line after the previous one (-1: missing)."""
    found, last = [], -1
    for start in starts:
        index = _index(lines, start, last)
        found.append(index)
        if index >= 0:
            last = index
    return found


def _progress_chapters(folder) -> dict:
    from glossarion_mobile.diagnostics import e2e

    return {int(k): v for k, v in e2e._progress_chapters(str(folder)).items()}


def _chapter_rows(view) -> dict:
    import re

    rows = {}
    for row in getattr(view, "rows", ()) or ():
        match = re.match(r"response_chapter(\d+)\.html$", os.path.basename(str(row.output_file or "")))
        if row.kind == "chapter" and match:
            rows[int(match.group(1))] = row
    return rows


async def _tap(tester, control, *, wait: bool = True):
    """The tap Flutter would send to ``control``: it must be on screen and enabled. ``wait=False``: the tap's
    handler runs as its own task (returned), like a device tap the client never waits on; a handler that asks a
    question (Start's first long job: "Keep translations running") then does not block the test that answers it."""
    assert any(c is control for c in tester._walk()), f"{type(control).__name__} is not on screen"
    node, disabled = control, False
    while node is not None:
        disabled = disabled or bool(getattr(node, "disabled", False))
        node = getattr(node, "parent", None)
    assert not disabled, f"{type(control).__name__} {getattr(control, 'key', None)!r} is disabled"
    if wait:
        await tester._dispatch(control, "click")
        return None
    task = asyncio.ensure_future(tester._dispatch(control, "click"))
    await asyncio.sleep(0)
    return task


async def _answer_dialogs(driver) -> None:
    """The first long job on Android asks "Keep translations running" ("Not now", as the device flow answers);
    an "Output Folder Mismatch" question would be answered Yes (the Library's own folder)."""
    if await driver.count(text="Keep translations running"):
        await driver.tap(text="Not now")
    if await driver.count(key="lib-output-root"):
        await driver.tap(text="Yes")


async def _open_book(app, driver, iso) -> types.SimpleNamespace:
    """Library > Import EPUB (the owner's file) > the book's card > Chapters. Returns the Book page and its row."""
    import flows

    service = app.library
    await flows.open_drawer(driver)
    await driver.tap(key="dest-library")
    await driver.wait(key="lib-search", timeout=60)
    await driver.pick_file(BOOK_FILE, lambda: driver.tap(key="lib-fab"))  # the In progress shelf's Import EPUB
    assert await _until(lambda: len(service.snapshot.all_books()) >= 1, 60), "the import added no book"
    books = list(service.snapshot.all_books())
    assert len(books) == 1, [b.get("name") for b in books]
    row = dict(books[0])
    raw = str(row.get("raw_source_path") or "")
    workspace = str(row.get("output_folder") or "")
    assert raw and os.path.isfile(raw) and Path(raw).name == BOOK_FILE, row
    assert workspace and os.path.isdir(workspace), row  # Import scaffolded the book's workspace
    assert _inside(workspace, iso.tmp) and _inside(raw, iso.tmp)
    bid = service.bid_for(row)
    await driver.tap(key=f"book-{bid}", timeout=60)
    assert await _until(lambda: type(app.shell.top_screen).__name__ == "BookPageScreen", 30), app.shell.current_route
    page = app.shell.top_screen
    await driver.tap(text="Chapters", timeout=30)
    await driver.wait(key="ch-run", timeout=60)
    _trace("chapters tab shown")
    assert await _until(lambda: page.chapters.view is not None, 60), "the Chapters tab never loaded"
    _trace("chapters view loaded")
    return types.SimpleNamespace(page=page, row=row, bid=bid, raw=raw, workspace=workspace, title=str(row.get("name")))


async def _run_translation(app, driver, tester, page) -> str:
    """Chapters > Run translation > the Translate sheet's Start: the book's ``translate`` job (its id)."""
    assert await _until(lambda: not page.chapters.run_button.disabled, 30), page.chapters.run_button.tooltip
    _trace("run button enabled")
    await _tap(tester, page.chapters.run_button)
    found = await driver.wait(key="start", timeout=60)
    _trace("translate sheet shown")
    buttons = [tester.control(found.at(i)) for i in range(found.count)]
    buttons = [b for b in buttons if type(b).__name__ == "FilledButton"]
    assert len(buttons) == 1, f"{len(buttons)} Start buttons"
    button = buttons[0]
    assert await _until(lambda: not button.disabled, 30), "the Translate sheet's Start stays disabled"
    tap = await _tap(tester, button, wait=False)
    _trace("start tapped")
    service = app.job_service
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        await _answer_dialogs(driver)
        view = service.view()
        jobs = [s for s in ((view.active,) if view.active is not None else ()) + tuple(view.queue)
                if s is not None and s.kind == "translate"]
        if jobs:
            await asyncio.wait_for(tap, 30)  # the Start handler returned (its submit is done)
            return jobs[0].id
        await asyncio.sleep(0.05)
    raise AssertionError(f"Start submitted no translate job (tap handler done: {tap.done()})")


async def _wait_terminal(app, job_id: str, timeout: float = RUN_TIMEOUT):
    service = app.job_service

    def ended():
        snap = service.snapshot(job_id)
        return snap if snap is not None and snap.is_terminal else None

    snap = await _until(ended, timeout, 0.1)
    assert snap is not None, f"job {job_id} did not end: {service.snapshot(job_id)}\n" + "\n".join(_log(app, job_id)[-30:])
    return snap


async def _wait_gate(app, gate: ScanGate, job_id: str) -> None:
    def entered() -> bool:
        snap = app.job_service.snapshot(job_id)
        return gate.entered.is_set() or (snap is not None and snap.is_terminal)

    assert await _until(entered, RUN_TIMEOUT, 0.1), "\n".join(_log(app, job_id)[-30:])
    snap = app.job_service.snapshot(job_id)
    assert gate.entered.is_set(), (f"the job ended {snap.state_label} without starting the QA scan; log tail:\n"
                                   + "\n".join(_log(app, job_id)[-30:]))
    # the scan's phase event reaches the job's snapshot (the gate holds the scan: nothing else changes it)
    await _until(lambda: app.job_service.snapshot(job_id).phase == "Scanning", 10)


async def _end_job(app, gate, job_ids) -> None:
    """Teardown: release the gate and force-stop a job a failing scenario left running."""
    if gate is not None:
        gate.armed = False
        gate.release()
    service = getattr(app, "job_service", None)
    if service is None:
        return
    for job_id in [j for j in job_ids if j]:
        snap = service.snapshot(job_id)
        if snap is not None and not snap.is_terminal:
            for _attempt in range(3):
                service.request_stop(job_id, force=True, reason="test teardown")
                if await _until(lambda: service.snapshot(job_id).is_terminal, 30):
                    break
    await asyncio.to_thread(service.wait_idle, 60)


async def _shutdown(tf, app) -> None:
    try:
        for name in ("config_store", "prefs"):
            target = getattr(app, name, None)
            if target is not None:
                try:
                    target.flush()
                except Exception:
                    pass
    finally:
        app.jobs.close()
        await tf._stop(app)


async def _run_qa_scan(app, tester, page) -> object:
    """Chapters > Run QA scan (the Progress Manager's button, hotfix 1 issue 7): the finished ``qa_scan`` job."""
    service = app.job_service
    button = page.chapters.qa_button
    assert await _until(lambda: not button.disabled, 30), button.tooltip
    before = {s.id for s in service.view().history}
    await _tap(tester, button)

    def submitted():
        view = service.view()
        for snap in ((view.active,) if view.active is not None else ()) + tuple(view.queue) + tuple(view.history):
            if snap is not None and snap.kind == "qa_scan" and snap.id not in before:
                return snap.id
        return None

    job_id = await _until(submitted, 30)
    assert job_id, "Run QA scan queued no qa_scan job"
    return await _wait_terminal(app, job_id, 120)


async def _scenario(iso, server, body) -> None:
    tf = _UF._foundations()
    app, tester, driver = await _UF._host_driver(tf, {BOOK_FILE: iso.book})
    if os.environ.get("GLOSSARION_TEST_TRACE"):
        driver.log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
    ctx = types.SimpleNamespace(app=app, tester=tester, driver=driver, server=server, gate=None, jobs=[])
    try:
        import flows

        await flows.wait_home(driver)
        _assert_isolated(iso.tmp)
        _configure(app, server)
        await body(ctx)
    except BaseException:
        if os.environ.get("GLOSSARION_TEST_TRACE"):
            import traceback

            traceback.print_exc()
        raise
    finally:
        _trace("teardown: end jobs")
        try:
            await _end_job(app, ctx.gate, ctx.jobs)
        finally:
            _trace("teardown: shutdown")
            await _shutdown(tf, app)
            _trace("teardown: done")


def _trace(*parts) -> None:
    if os.environ.get("GLOSSARION_TEST_TRACE"):
        print(time.strftime("%H:%M:%S"), "[test]", *parts, flush=True)


def _check_held_scan(ctx, book, job_id: str) -> list:
    """While the gate holds the scan: it runs inside the translation's own job, after the translation."""
    app, gate = ctx.app, ctx.gate
    service = app.job_service
    snap = service.snapshot(job_id)
    assert snap.state.value == "RUNNING" and snap.kind == "translate", (snap.state, snap.kind)
    assert snap.phase == "Scanning", snap.phase
    view = service.view()
    assert view.active is not None and view.active.id == job_id and not view.queue, \
        f"another job next to the translation: active={view.active and view.active.kind} queue={view.queue}"
    assert len(gate.calls) == 1, gate.calls
    call = gate.calls[0]
    assert call["thread"] == "gl-job", f"the scan runs on {call['thread']}, not the translation's job thread"
    assert _same(call["folder"], book.workspace), (call["folder"], book.workspace)
    assert call["epub"] and _same(call["epub"], book.raw), (call["epub"], book.raw)
    assert call["mode"] == "quick-scan" and call["sample"] == 0, call
    assert call["thread_executor"] == "1", call  # mobile: the scanner's thread executor, no processes
    lines = _log(app, job_id)
    assert UNSHARED_LINE not in lines, lines[-20:]
    starts = ("✅ Translation completed successfully!", TRIGGER_LINE, "✅ QA scanner initialized successfully",
              f"📁 Auto-selected output folder for {BOOK_STEM}: ", f"📚 Using EPUB from scan list: {BOOK_FILE}",
              "🔍 Starting QA scan in QUICK-SCAN mode for folder: ",
              "⚡ Quick Scan duplicate check sample size: 0 (duplicate check off)")
    found = _in_order(lines, starts)
    assert -1 not in found, (list(zip(starts, found)), lines[-40:])
    assert _index(lines, "■ ") == -1, "the job already logged its end while the scan was held"
    # the run buttons wait while the translation (and its scan) runs (hotfix 1 issue 7)
    chapters = book.page.chapters
    assert chapters.run_button.disabled and chapters.qa_button.disabled, (chapters.run_button.tooltip,
                                                                           chapters.qa_button.tooltip)
    return lines


# ==========================================================================
# 1. The owner's run: the scan runs in the job and its result shows like a QA job's
# ==========================================================================


def test_the_scan_runs_in_the_translation_job_and_is_surfaced_like_a_qa_job(iso, offline, seeded_langdetect,
                                                                            monkeypatch):
    from glossarion_mobile.diagnostics.e2e import E2ESession
    from glossarion_mobile.diagnostics.fake_llm_server import FakeLLMServer
    from glossarion_mobile.job_kinds.qa import report_path_for

    raw_chapter = E2ESession.RAW_QA_CHAPTER

    async def body(ctx):
        app, tester, driver, server = ctx.app, ctx.tester, ctx.driver, ctx.server
        service = app.job_service
        ctx.gate = _install_gate(monkeypatch)
        book = await _open_book(app, driver, iso)
        server.leave_raw_once[raw_chapter] = E2ESession.RAW_LEFTOVER
        job_id = await _run_translation(app, driver, tester, book.page)
        ctx.jobs.append(job_id)

        # ---- the scan, held at the real scanner's entry --------------------------------------------
        await _wait_gate(app, ctx.gate, job_id)
        _check_held_scan(ctx, book, job_id)
        assert not server.leave_raw_once, "the model never answered chapter 5"
        ctx.gate.armed = False
        ctx.gate.release()

        # ---- Done after the scan ---------------------------------------------------------------------
        snap = await _wait_terminal(app, job_id)
        lines = _log(app, job_id)
        assert snap.state.value == "DONE" and snap.error is None, (snap.state_label, snap.error, lines[-30:])
        assert UNSHARED_LINE not in lines
        starts = ("✅ Translation completed successfully!", TRIGGER_LINE, "✅ QA scanner initialized successfully",
                  "🔍 Starting QA scan in QUICK-SCAN mode for folder: ", "✅ QA scan completed successfully.",
                  f"■ Done: {snap.title}")
        found = _in_order(lines, starts)
        assert -1 not in found, (list(zip(starts, found)), lines[-40:])
        assert snap.title == book.title, (snap.title, book.title)
        assert not [line for line in lines if "QA scan error" in line or line.startswith("Traceback")], lines[-40:]
        call = ctx.gate.calls[0]
        assert call["returned"] and call["error"] is None and len(ctx.gate.calls) == 1, ctx.gate.calls
        # the report, in the book's workspace
        workspace = snap.output_dir or book.workspace
        assert _same(workspace, book.workspace), (workspace, book.workspace)
        report = report_path_for(workspace)
        assert os.path.isfile(report), f"no report at {report}"
        report_json = os.path.join(os.path.dirname(report), "validation_results.json")
        assert os.path.isfile(report_json), report_json
        rows = json.loads(Path(report_json).read_text(encoding="utf-8"))
        assert rows, "the report lists no file"
        # the job's outputs: the compiled book first, the report last; the QA fields of a qa_scan job
        outputs = [str(p) for p in snap.outputs]
        assert outputs and outputs[0].lower().endswith(".epub") and _same(outputs[-1], report), outputs
        result = dict(snap.result or {})
        assert [os.path.normcase(p) for p in result.get("qa_reports") or ()] == [os.path.normcase(report)], result
        assert result.get("qa_mode") == "quick-scan", result
        assert [os.path.normcase(os.path.abspath(p)) for p in result.get("qa_folders") or ()] == \
            [os.path.normcase(os.path.abspath(workspace))], result
        # the QA marks: chapter 5 (left partly Korean) is qa_failed, the others completed
        chapters = _progress_chapters(workspace)
        assert sorted(chapters) == list(range(1, CHAPTERS + 1)), chapters
        assert chapters[raw_chapter] == "qa_failed", chapters
        assert all(v == "completed" for n, v in chapters.items() if n != raw_chapter), chapters
        assert int(getattr(snap.progress, "failed", 0) or 0) >= 1, snap.progress

        # ---- the Book page's Chapters tab shows the mark (the Progress Manager) ---------------------------
        page = book.page

        def marked():
            row = _chapter_rows(page.chapters.view).get(raw_chapter)
            return row is not None and row.status == "qa_failed"

        assert await _until(marked, 30), {n: r.status for n, r in _chapter_rows(page.chapters.view).items()}
        row = _chapter_rows(page.chapters.view)[raw_chapter]
        assert any("korean" in str(line).lower() for line in row.qa_lines), row.qa_lines

        # ---- Book > Output: the QA report, opened in the QA report viewer -----------------------------
        await driver.tap(text="Output", timeout=30)
        await driver.wait(contains="QA reports (1)", timeout=60)
        await driver.tap(contains="QA reports (1)")
        await driver.tap(key="out-qa-0", timeout=30)
        assert await _until(lambda: type(app.shell.top_screen).__name__ == "QaReportScreen", 30), \
            app.shell.current_route
        screen = app.shell.top_screen
        assert _same(screen.path, report), (screen.path, report)
        assert await _until(lambda: screen.summary is not None, 30), "the report viewer never loaded the report"
        assert screen.summary.total >= CHAPTERS and screen.summary.with_issues >= 1, screen.summary
        await driver.back()
        assert await _until(lambda: app.shell.top_screen is page, 30), app.shell.current_route

        # ---- the Jobs page: the report among the job's outputs, "1 QA failed" ------------------------------
        app.navigate_to("jobs.detail", {"jid": job_id})
        assert await _until(lambda: type(app.shell.top_screen).__name__ == "JobDetailScreen", 30), \
            app.shell.current_route
        detail = app.shell.top_screen
        assert await _until(lambda: detail.snap is not None and detail.snap.is_terminal, 10)
        chips = [c.label.value for c in detail.outputs_row.controls]
        assert "validation_results.html" in chips and chips[0].lower().endswith(".epub"), chips
        assert detail.outputs_card.visible
        assert detail.qa_chip.visible and detail.qa_chip.label.value == "1 QA failed", \
            (detail.qa_chip.visible, detail.qa_chip.label.value)
        assert detail.state_chip.text == "Done" and not detail.stop_button.visible
        await driver.back()
        assert await _until(lambda: app.shell.top_screen is page, 30), app.shell.current_route

        # ---- the same workspace through a QA job (Chapters > Run QA scan): the same result -------------------
        before_mtime = os.path.getmtime(report)
        await driver.tap(text="Chapters", timeout=30)
        qa_snap = await _run_qa_scan(app, tester, page)
        qa_lines = _log(app, qa_snap.id)
        assert qa_snap.state.value == "DONE" and qa_snap.error is None, (qa_snap.state_label, qa_lines[-20:])
        assert [str(p) for p in qa_snap.outputs] == [outputs[-1]], (qa_snap.outputs, outputs)
        assert os.path.getmtime(report) >= before_mtime
        qa_result = dict(qa_snap.result or {})
        assert set(qa_result) == {"qa_reports", "qa_mode", "qa_folders"}, qa_result
        for key, value in qa_result.items():
            assert result.get(key) == value, (key, result.get(key), value)
        assert _progress_chapters(workspace) == chapters
        _assert_offline(offline)

    with FakeLLMServer() as server:
        asyncio.run(_scenario(iso, server, body))


# ==========================================================================
# 2. Stop during the scan
# ==========================================================================


#: Where the Stop lands: as the real scanner starts, and inside it (it logged its file count and is about to
#: hand the files to its thread executor, after reading the source EPUB).
STOP_POINTS = {"scanner-entry": None, "inside-scanner": "🔍 Found "}


@pytest.mark.parametrize("stop_point", sorted(STOP_POINTS))
def test_stop_during_the_scan_stops_the_job(iso, offline, seeded_langdetect, monkeypatch, stop_point):
    from glossarion_mobile.diagnostics.fake_llm_server import FakeLLMServer
    from glossarion_mobile.job_kinds.qa import report_path_for

    async def body(ctx):
        app, tester, driver = ctx.app, ctx.tester, ctx.driver
        service = app.job_service
        ctx.gate = _install_gate(monkeypatch, hold_on=STOP_POINTS[stop_point])
        book = await _open_book(app, driver, iso)
        job_id = await _run_translation(app, driver, tester, book.page)
        ctx.jobs.append(job_id)
        await _wait_gate(app, ctx.gate, job_id)
        held = _check_held_scan(ctx, book, job_id)
        assert ctx.gate.calls[0]["held_thread"] == "gl-job", ctx.gate.calls[0]
        if STOP_POINTS[stop_point] is not None:
            assert _index(held, f"🔍 Found {CHAPTERS} HTML files. Starting parallel scan...") >= 0, held[-15:]

        # ---- the Jobs page's Stop while the scan runs -------------------------------------------------------
        app.navigate_to("jobs.detail", {"jid": job_id})
        assert await _until(lambda: type(app.shell.top_screen).__name__ == "JobDetailScreen", 30), \
            app.shell.current_route
        detail = app.shell.top_screen
        assert await _until(lambda: detail.stop_button.visible and detail.snap is not None
                            and detail.snap.state.value == "RUNNING", 10)
        await _tap(tester, detail.stop_button)
        assert await _until(lambda: service.snapshot(job_id).state.value in ("STOPPING", "FORCE_STOPPING"), 10), \
            service.snapshot(job_id).state
        snap = service.snapshot(job_id)
        assert not snap.is_terminal and snap.stop_mode is not None, (snap.state, snap.stop_mode)
        # the scanner continues; the real scan sees the Stop
        released = time.monotonic()
        ctx.gate.armed = False
        ctx.gate.release()
        snap = await _wait_terminal(app, job_id, 120)
        took = time.monotonic() - released
        lines = _log(app, job_id)
        assert snap.state.value == "CANCELLED" and snap.state_label == "Stopped", (snap.state_label, lines[-30:])
        assert snap.error is None, snap.error
        assert took < 30, f"the stopped scan took {took:.1f}s to end"
        call = ctx.gate.calls[0]
        assert call["returned"] and call["error"] is None, call
        scan_at = _index(lines, "🔍 Starting QA scan in QUICK-SCAN mode for folder: ")
        assert scan_at >= 0 and _index(lines, "⛔ Stop requested", scan_at) > scan_at, lines[scan_at:][-30:]
        assert _index(lines, f"■ Stopped: {snap.title}") > scan_at, lines[-10:]
        assert UNSHARED_LINE not in lines
        assert not [line for line in lines if "QA scan error" in line or line.startswith("Traceback")], lines[-40:]
        # the scanner stopped where it was: no report, no "Scan complete"
        assert _index(lines, "✅ Scan complete!", scan_at) == -1, lines[scan_at:][-30:]
        if STOP_POINTS[stop_point] is not None:
            assert _index(lines, "⛔ QA scan interrupted before processing.", scan_at) > scan_at, lines[scan_at:][-30:]
        report = report_path_for(book.workspace)
        assert not os.path.exists(report), "the stopped scan still wrote its report"
        assert "qa_reports" not in (snap.result or {}), snap.result
        # the Stop only cut the scan short: every chapter stays translated
        chapters = _progress_chapters(book.workspace)
        assert sorted(chapters) == list(range(1, CHAPTERS + 1)), chapters
        assert all(v == "completed" for v in chapters.values()), chapters
        assert await _until(lambda: detail.snap.is_terminal and not detail.stop_button.visible, 10)
        assert detail.state_chip.text == "Stopped", detail.state_chip.text
        await driver.back()
        assert await _until(lambda: app.shell.top_screen is book.page, 30), app.shell.current_route

        # ---- afterwards a QA scan runs normally (the stop flags were cleared) ---------------------------
        assert os.environ.get("TRANSLATION_CANCELLED") != "1", "the stopped scan left TRANSLATION_CANCELLED set"
        qa_snap = await _run_qa_scan(app, tester, book.page)
        qa_lines = _log(app, qa_snap.id)
        assert qa_snap.state.value == "DONE" and qa_snap.error is None, (qa_snap.state_label, qa_lines[-20:])
        assert not [line for line in qa_lines if line.startswith("⛔ ")], qa_lines[-20:]
        assert [str(p) for p in qa_snap.outputs] == [report] and os.path.isfile(report), qa_snap.outputs
        assert len(ctx.gate.calls) == 2 and ctx.gate.calls[1]["returned"], ctx.gate.calls
        _assert_offline(offline)

    with FakeLLMServer() as server:
        asyncio.run(_scenario(iso, server, body))
