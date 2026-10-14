"""Acceptance for the owner's device report #3 on the U8 APK (2026-10-08): "Log fonts are too
large, reduce it to 8".

The real app (``main.main``) runs on the fake Flet session as Android
(``test_bootstrap._fake_session``: every message is msgpack-encoded like the socket transport).
Two things are replaced:

* the job service's backend, by ``test_jobs.FakeBackend``;
* the ``run`` of the job kinds used here, by a stand-in that logs like a translation (thinking,
  API and error lines) and then waits until it is released.

Taps go through ``host_tester.PyTester`` / ``ui_driver.UiDriver``, the driver the device flows
use. For every log surface the owner can open, the test checks two things:

* the built control: ``Text(size=8, style.height=1.375, font_family="monospace", selectable)``;
* the bytes the phone is sent for that control (``WireRecorder``), which is what Flutter draws
  from.

Log surfaces covered:

* the job detail Log card of a running job, opened from its notification link (``/job/<id>``)
  and from its Jobs list row, and again after the Errors filter chip rebuilds the blocks;
* QA Scanner › View log, which opens the scan job's detail Log;
* Tools › Manga › Files, the log of the running batch;
* Logs & diagnostics:
  * Live log (the app's ``glossarion.*`` log records);
  * Log files › tap a file, which opens the viewer sheet;
  * Check environment lines (the real ``run_env_check`` pipeline over a stand-in owner);
* Reader › 🌐 Translate, the live panel's 🧠 Thinking pane (thinking and pipeline log lines).

The owner asked about logs only. Code editors and ErrorCard keep the general mono token (13):
the Text editor opened on a ``.log`` file, and the Glossaries list's ErrorCard.

Real data is never touched:

* HOME, USERPROFILE, APPDATA, GLOSSARION_LIBRARY_DIR, OUTPUT_DIRECTORY, CONFIG_FILE and the model
  caches point into tmp;
* GLOSSARION_HTTP_LOG=0;
* every test checks that src/config.json is byte-identical afterwards.

Run from src/mobile with the mobile venv:
    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_devfix_issue3.py
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import importlib.util
import logging
import os
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
from glossarion_mobile.services.jobs import JobSpec, JobState  # noqa: E402
from glossarion_mobile.ui import tokens  # noqa: E402


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


_TB = _load("_glossarion_devfix3_tb_helpers", "test_bootstrap.py")
storage = _TB.storage
app_env = _TB.app_env
_TR = _load("_glossarion_devfix3_reader_helpers", "test_reader.py")
cores = _TR.cores  # the real reader cores (reader_doc, library_core, ...) kept in tmp
novel = _TR.novel  # raw EPUB + an in-progress workspace (chapter 2 pending)


def _uf():
    return _load("_glossarion_devfix3_uf_helpers", "test_ui_foundations.py")


def _tj():
    return _load("_glossarion_devfix3_jobs_helpers", "test_jobs.py")


#: what the owner asked for: 8 sp, with the token's line height (11/8) instead of bodyMedium's 20/14
LOG_SIZE = 8
LOG_HEIGHT = round(11 / 8, 4)
MONO_SIZE = 13
ANDROID_MONO = "monospace"

#: (text, explicit LogLine kind) a translation job logs; the error line is what the Errors chip keeps.
#: The indented thinking body reaches the Reader's live pane (whole-message listener) but not the
#: job LogBuffer (the shared request stream keeps streamed text out of the main log, as on desktop).
JOB_LINES = (
    ("🚀 Starting translation", None),
    ("🧠 [gpt] Thinking...", None),
    ("    planning the chapter", None),
    ("🧠 [gpt] Thinking complete", None),
    ("📤 Sending API call (chunk 1/1)", None),
    ("❌ API error 429: rate limited, retrying in 5 s", "error"),
)


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
    """Nothing reads or writes the developer's Library, output folders, home, AppData, model caches
    or src/config.json (autouse: runs before ``storage`` / the bootstrap env contract)."""
    iso = tmp_path / "_iso"
    for name, sub in (("HOME", "home"), ("USERPROFILE", "home"), ("APPDATA", "appdata"),
                      ("GLOSSARION_LIBRARY_DIR", "Library"), ("OUTPUT_DIRECTORY", "Output"),
                      ("BUBBLE_CACHE_DIR", "models/detector"), ("MODEL_CACHE_DIR", "models/inpainting"),
                      ("ONNX_CACHE_DIR", "models/onnx")):
        folder = iso / sub
        folder.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv(name, str(folder))
    monkeypatch.setenv("CONFIG_FILE", str(iso / "config.json"))  # never src/config.json
    monkeypatch.setenv("GLOSSARION_HTTP_LOG", "0")
    config = SRC_DIR / "config.json"
    before = _md5(config)
    yield iso
    assert _md5(config) == before, "src/config.json changed during the test"


# ==========================================================================
# What the phone is sent
# ==========================================================================


class WireRecorder:
    """Keeps every msgpack frame the fake Flet client is sent (``_fake_session`` encodes each
    message with ``msgpack.packb`` exactly like the socket transport) and finds the full
    encodings of one control in them: the fields the phone's Flutter ``Text`` is built from."""

    def __init__(self, monkeypatch) -> None:
        import msgpack

        self._msgpack = msgpack
        self._lock = threading.Lock()
        self.frames: list = []
        self._decoded: list = []
        original = msgpack.packb

        def packb(obj, *args, **kwargs):
            data = original(obj, *args, **kwargs)
            with self._lock:
                self.frames.append(data)
            return data

        monkeypatch.setattr(msgpack, "packb", packb)

    def _decode_new(self) -> None:
        with self._lock:
            frames = self.frames[len(self._decoded):]
        for frame in frames:
            try:
                value = self._msgpack.unpackb(frame, raw=False, strict_map_key=False)
            except Exception:
                value = None
            self._decoded.append(value)

    def encodings(self, control) -> list:
        """Every full encoding (``{"_c": ..., "_i": <id>, ...}``) of ``control`` sent so far."""
        self._decode_new()
        target = getattr(control, "_i", None)
        found = []
        for value in self._decoded:
            stack = [value]
            while stack:
                node = stack.pop()
                if isinstance(node, dict):
                    if node.get("_i") == target and "_c" in node:
                        found.append(node)
                    stack.extend(node.values())
                elif isinstance(node, (list, tuple)):
                    stack.extend(node)
        return found


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


def _assert_log_text(text, *, family: str = ANDROID_MONO, where: str = "") -> None:
    """The built control is the owner's log size (and not the 13 sp mono token any more)."""
    import flet as ft

    label = f"{where}: {str(getattr(text, 'value', ''))[:60]!r}"
    assert isinstance(text, ft.Text), label
    assert text.size == LOG_SIZE == tokens.LOG_STYLE.size, f"{label} has size {text.size}"
    assert text.style is not None and text.style.height == LOG_HEIGHT, f"{label} line height {text.style}"
    assert text.selectable is True, label
    assert text.font_family == family, f"{label} family {text.font_family}"
    assert text.theme_style is None, f"{label} theme_style {text.theme_style}"


async def _assert_log_on_wire(wire: WireRecorder, text, *, family: str = ANDROID_MONO, where: str = "") -> None:
    """The phone was told to draw ``text`` at 8 sp with the 11/8 line height in the mono family."""
    assert await _until(lambda: wire.encodings(text)), f"{where}: the log text never reached the phone"
    for sent in wire.encodings(text):
        assert sent.get("_c") == "Text", (where, sent)
        assert sent.get("size") == LOG_SIZE, (where, sent)
        assert (sent.get("style") or {}).get("height") == LOG_HEIGHT, (where, sent)
        assert sent.get("font_family") == family, (where, sent)
        assert sent.get("selectable") is True, (where, sent)
        assert "theme_style" not in sent, (where, sent)


def _console_blocks(console) -> list:
    return [c for c in console.list_view.controls if c is not console.empty_text]


def _console_text(console) -> str:
    return "\n".join(str(b.value or "") for b in _console_blocks(console))


async def _check_console(console, wire: WireRecorder, *, expect: str, where: str) -> list:
    """A LogConsole showing ``expect``: every block it mounted is a log-size Text, on the wire too."""
    assert await _until(lambda: expect in _console_text(console), 15), \
        f"{where}: {expect!r} never appeared (shown: {_console_text(console)[:300]!r})"
    blocks = _console_blocks(console)
    assert blocks, where
    for block in blocks:
        _assert_log_text(block, where=where)
        await _assert_log_on_wire(wire, block, where=where)
    return blocks


def _texts(control) -> list:
    import flet as ft

    found, stack, seen = [], [control], set()
    while stack:
        node = stack.pop()
        if node is None or id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, ft.Text):
            found.append(node)
        for attr in ("content", "controls", "title", "subtitle", "leading", "trailing", "actions"):
            child = getattr(node, attr, None)
            if isinstance(child, list):
                stack.extend(c for c in child if isinstance(c, ft.Control))
            elif isinstance(child, ft.Control):
                stack.append(child)
    return found


# ==========================================================================
# The app with a fake job backend and stand-in job runs
# ==========================================================================


class FakeRuns:
    """The ``run`` of each job kind under test, replaced: it logs ``JOB_LINES`` through the job's
    own log host (so the lines take the JobService's real path into the job LogBuffer, the log
    file and the whole-message listeners), then waits until released (or stopped)."""

    def __init__(self, monkeypatch, kinds, *, heartbeat: float = 0.0) -> None:
        self.entered = {kind: threading.Event() for kind in kinds}
        self.release = {kind: threading.Event() for kind in kinds}
        self.heartbeat = heartbeat  # > 0: one more line every ``heartbeat`` s while waiting (a live stream)
        for kind in kinds:
            info = job_kinds.get_kind(kind)  # the real KindInfo: verb, icon, stop kind stay real
            monkeypatch.setitem(job_kinds._CACHE, kind, dataclasses.replace(info, run=self._run))

    def _run(self, ctx):
        kind = ctx.spec.kind
        for text, level in JOB_LINES:
            if level:
                ctx.host.log(text, kind=level)
            else:
                ctx.log(text)
        self.entered[kind].set()
        beat = time.monotonic()
        while not self.release[kind].wait(0.02):
            if ctx.stop_requested():
                break
            if self.heartbeat and time.monotonic() - beat >= self.heartbeat:
                beat = time.monotonic()
                ctx.log("⏳ waiting for the stream…")
        return None

    def release_all(self) -> None:
        for event in self.release.values():
            event.set()


def _fake_backend(monkeypatch, tmp_path) -> list:
    """The app's JobService is built with ``test_jobs.FakeBackend`` (no HeadlessOwner, no shared
    pipeline): ``JobsFeature`` keeps everything else real (routes, strip, notifications)."""
    tj = _tj()
    made: list = []

    def make_backend():
        backend = tj.FakeBackend(str(tmp_path / "job-out"))
        made.append(backend)
        return backend

    monkeypatch.setattr(jobs_service, "JobBackend", make_backend)
    return made


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


async def _finish(service, job_id, runs, kind) -> None:
    runs.release[kind].set()
    assert await _until(lambda: service.snapshot(job_id).is_terminal, 15)
    assert service.snapshot(job_id).state is JobState.DONE, service.snapshot(job_id).error


# ==========================================================================
# Job logs: job detail (notification link, Jobs list, Errors filter), QA › View log, Manga › Files
# ==========================================================================


def test_job_logs_render_at_8_in_job_detail_qa_and_manga(app_env, tmp_path, monkeypatch):
    from glossarion_mobile.ui.screens.job_detail import JobDetailScreen
    from glossarion_mobile.ui.screens.jobs import JobsScreen
    from glossarion_mobile.ui.tools.qa_screen import QaScannerScreen

    wire = WireRecorder(monkeypatch)
    runs = FakeRuns(monkeypatch, ("translate", "qa_scan", "manga"))
    _fake_backend(monkeypatch, tmp_path)
    book = tmp_path / "in" / "Book.epub"
    book.parent.mkdir(parents=True)
    book.write_bytes(b"PK fake epub")
    page_image = tmp_path / "in" / "1.png"
    page_image.write_bytes(b"\x89PNG\r\n\x1a\n")

    async def scenario():
        shell = await _start_app()
        app, ui, service = shell.app, shell.ui, shell.app.job_service
        try:
            # --- a running translation, opened from its notification link (/job/<id>) ---
            jid = service.submit(JobSpec("translate", "Book.epub", (str(book),)))
            assert await asyncio.to_thread(runs.entered["translate"].wait, 10)
            await shell.uf._route(shell.session, f"/job/{jid}")
            detail = app.shell.top_screen
            assert isinstance(detail, JobDetailScreen) and detail.job_id == jid
            assert detail.console.list_height == 360  # the owner's 360 dp Log card
            await _check_console(detail.console, wire, expect="Sending API call", where="job detail (link)")
            # the Errors chip rebuilds the blocks from the kept lines: still the log size
            await ui.tap(key="log-filter-errors", timeout=5)
            assert await _until(lambda: detail.console.filter_id == "errors"
                                and _console_text(detail.console).startswith("❌ API error 429"))
            await _check_console(detail.console, wire, expect="API error 429", where="job detail (Errors)")
            await ui.tap(key="log-filter-all", timeout=5)
            assert await _until(lambda: detail.console.filter_id == "all")
            await _check_console(detail.console, wire, expect="Sending API call", where="job detail (All)")

            # --- the same job from the Jobs list row ---
            await app.navigate("/jobs")
            assert await _until(lambda: isinstance(app.shell.top_screen, JobsScreen))
            await ui.tap(key=f"job-row-{jid}", timeout=10)
            assert await _until(lambda: isinstance(app.shell.top_screen, JobDetailScreen)
                                and app.shell.top_screen is not detail)
            await _check_console(app.shell.top_screen.console, wire, expect="Sending API call",
                                 where="job detail (Jobs list)")
            await _finish(service, jid, runs, "translate")

            # --- QA Scanner › View log: the scan job's detail Log ---
            qid = service.submit(JobSpec("qa_scan", "QA scan · Book", (str(tmp_path / "in"),)))
            assert await asyncio.to_thread(runs.entered["qa_scan"].wait, 10)
            await app.navigate("/tools/qa")
            qa = app.shell.top_screen
            assert isinstance(qa, QaScannerScreen)
            assert await _until(lambda: qa.log_button.visible and qa.active_job is not None
                                and qa.active_job.id == qid)
            await ui.tap(key="qa-log", timeout=5)
            assert await _until(lambda: isinstance(app.shell.top_screen, JobDetailScreen)
                                and app.shell.top_screen.job_id == qid)
            await _check_console(app.shell.top_screen.console, wire, expect="Sending API call",
                                 where="QA › View log")
            await _finish(service, qid, runs, "qa_scan")

            # --- Tools › Manga › Files: the running batch's log ---
            mid = service.submit(JobSpec("manga", "Series · 1 image", (str(page_image),),
                                         params={"files": [str(page_image)]}, resumable=False))
            assert await asyncio.to_thread(runs.entered["manga"].wait, 10)
            await app.navigate("/tools/manga?tab=files")
            manga = app.shell.top_screen
            assert type(manga).__name__ == "MangaScreen", type(manga)
            files_tab = manga.files_tab
            assert await _until(lambda: files_tab.console_job == mid and files_tab.console is not None, 15)
            assert files_tab.console.list_height == 260
            await _check_console(files_tab.console, wire, expect="Sending API call", where="Manga › Files")
            await _finish(service, mid, runs, "manga")
        finally:
            await _stop_app(shell, runs)

    asyncio.run(scenario())


# ==========================================================================
# Logs & diagnostics: Live log, Log files viewer, Check environment
# ==========================================================================


class _EnvOwner:
    """Stand-in HeadlessOwner for the real ``run_env_check`` (job lock, scoped process env,
    redaction, the ``[ENV_DEBUG]`` line filter): the desktop debug check's line shapes."""

    def __init__(self, config, *, host, api_key="") -> None:
        self.host = host

    def initialize_environment_variables(self) -> bool:
        self.host.log("🚀 [INIT] Initializing all environment variables from config...")
        return True

    def debug_environment_variables(self, show_all: bool = False) -> bool:
        self.host.log("✅ [ENV_DEBUG] OK: MODEL")
        self.host.log("[ENV_DEBUG] TRANSLATION_TEMPERATURE: 0.3")
        self.host.log("❌ [ENV_DEBUG] CRITICAL MISSING: OUTPUT_LANGUAGE - Target language")
        return False


#: a rotated log file name the running app never writes itself (``services.logs.LOG_NAMES`` + ".N")
VIEWED_LOG = "memory.log.1"
VIEWED_TEXT = "12:00:01 [memory] rss 412 MB\n12:00:31 [memory] rss 398 MB · gc 3"


def test_diagnostics_live_log_file_viewer_and_env_check_render_at_8(app_env, tmp_path, monkeypatch):
    import flet as ft

    from glossarion_mobile.ui.components.dialogs import close_dialog
    from glossarion_mobile.ui.screens import env_preview
    from glossarion_mobile.ui.screens.diagnostics import DiagnosticsScreen

    wire = WireRecorder(monkeypatch)
    _fake_backend(monkeypatch, tmp_path)
    real_check = env_preview.run_env_check
    monkeypatch.setattr(env_preview, "run_env_check",
                        lambda config, **kwargs: real_check(config, owner_factory=_EnvOwner, **kwargs))

    async def scenario():
        shell = await _start_app()
        app, ui, page = shell.app, shell.ui, shell.page
        try:
            logs_dir = Path(app.paths.logs)
            logs_dir.mkdir(parents=True, exist_ok=True)
            (logs_dir / VIEWED_LOG).write_bytes(VIEWED_TEXT.encode("utf-8"))
            await shell.uf._route(shell.session, "/settings/logs")
            screen = app.shell.top_screen
            assert isinstance(screen, DiagnosticsScreen) and screen.console is not None
            assert screen.console.list_height == 320

            # --- Live log: the app's own log records ---
            logging.getLogger("glossarion.devfix3").warning("⚠️ devfix3: a live log line")
            await _check_console(screen.console, wire, expect="devfix3: a live log line", where="Diagnostics › Live log")

            # --- Log files › tap a file: the viewer sheet ---
            await ui.tap(key=f"diag-logfile-{VIEWED_LOG}", timeout=10)

            def viewer():
                return next((d for d in page._dialogs.controls
                             if getattr(d, "key", None) == "diag-log-viewer" and getattr(d, "open", False)), None)

            assert await _until(lambda: viewer() is not None)
            sheet = viewer()
            body = [t for t in _texts(sheet) if t.value == VIEWED_TEXT]
            assert len(body) == 1, [t.value for t in _texts(sheet)]
            _assert_log_text(body[0], where="Diagnostics › log file viewer")
            await _assert_log_on_wire(wire, body[0], where="Diagnostics › log file viewer")
            close_dialog(page, sheet)
            assert await _until(lambda: viewer() is None)

            # --- Debug › Check environment: the [ENV_DEBUG] lines ---
            await ui.tap(key="diag-env-check", timeout=5)
            assert await _until(lambda: screen.env_check_lines.visible and screen.env_check_lines.controls, 15)
            lines = [c for c in screen.env_check_lines.controls if isinstance(c, ft.Text)]
            assert [t.value for t in lines] == ["[ENV_DEBUG] TRANSLATION_TEMPERATURE: 0.3",
                                                "❌ [ENV_DEBUG] CRITICAL MISSING: OUTPUT_LANGUAGE - Target language"]
            assert screen.env_check_status.value.startswith("❌")  # the verdict line is not a log line
            for text in lines:
                _assert_log_text(text, where="Diagnostics › Check environment")
                await _assert_log_on_wire(wire, text, where="Diagnostics › Check environment")
        finally:
            await _stop_app(shell)

    asyncio.run(scenario())


# ==========================================================================
# Reader › Translate: the live panel's Thinking / log pane
# ==========================================================================


def test_reader_live_thinking_pane_renders_at_8(app_env, novel, tmp_path, monkeypatch):
    if not _has("live_stream"):
        pytest.skip("the shared live_stream module is not importable here")
    import flet as ft

    from glossarion_mobile.ui.reader.reader_view import ReaderScreen

    wire = WireRecorder(monkeypatch)
    # the live panel re-renders at most every 250 ms when lines arrive: keep a trickle of lines coming
    runs = FakeRuns(monkeypatch, ("single_chapter",), heartbeat=0.3)
    _fake_backend(monkeypatch, tmp_path)

    async def scenario():
        shell = await _start_app()
        app, ui, service = shell.app, shell.ui, shell.app.job_service
        try:
            # the Library › book › Read path: ReaderFeature.open_book with the book row, chapter 2
            assert app.reader is not None
            assert app.reader.open_book(dict(novel.book), chapter=1)
            assert await _until(lambda: isinstance(app.reader.active, ReaderScreen)
                                and app.reader.active.state == "ready", 30)
            screen = app.reader.active
            assert screen.index == 1 and screen.deps.jobs is service
            # 🌐 Translate this chapter (chapter 2 is pending: no retranslate question)
            if not await ui.exists(key="reader-translate", timeout=2):
                screen.handle_payload({"type": "tap", "seq": 900, "zone": "center", "chapter": screen.index,
                                       "doc": screen.current_doc})  # the centre tap shows the chrome
            await ui.tap(key="reader-translate", timeout=5)
            assert await _until(lambda: screen.live is not None, 10), "🌐 Translate did not start a live run"
            live = screen.live
            assert service.snapshot(live.job_id).spec.kind == "single_chapter"
            assert await asyncio.to_thread(runs.entered["single_chapter"].wait, 10)
            panel = live.panel
            assert panel.is_open
            assert await _until(lambda: "planning the chapter" in (panel.side_text.value or ""), 15), \
                panel.side_text.value
            assert "❌ API error 429" in panel.side_text.value  # the pipeline log goes to the same pane
            _assert_log_text(panel.side_text, where="Reader › live Thinking pane")
            assert panel.side_text.color == ft.Colors.ON_SURFACE_VARIANT
            await _assert_log_on_wire(wire, panel.side_text, where="Reader › live Thinking pane")
            await _finish(service, live.job_id, runs, "single_chapter")
        finally:
            await _stop_app(shell, runs)

    asyncio.run(scenario())


# ==========================================================================
# Not logs: code editors and ErrorCard keep the general mono token (13)
# ==========================================================================


def test_text_editor_and_error_card_keep_the_mono_13_token(app_env, tmp_path, monkeypatch):
    from glossarion_mobile.services.glossary import GlossaryService
    from glossarion_mobile.ui.components.error_card import ErrorCard
    from glossarion_mobile.ui.tools.text_editor import TextEditorScreen

    assert tokens.MONO_STYLE.size == MONO_SIZE and tokens.LOG_STYLE.size == LOG_SIZE
    wire = WireRecorder(monkeypatch)
    _fake_backend(monkeypatch, tmp_path)

    def failing_listing(self):
        raise PermissionError("Permission denied: 'Glossary'")

    monkeypatch.setattr(GlossaryService, "list_glossaries", failing_listing)

    async def scenario():
        shell = await _start_app()
        app, page = shell.app, shell.page
        try:
            # Files › Logs › a .log file › Text editor: an editor, so the mono 13 token
            root = app.jobs.file_roots()["logs"]
            os.makedirs(root, exist_ok=True)
            log_copy = Path(root) / "exported_run.log"
            log_copy.write_bytes(b"12:00 [INFO] started\n12:01 [ERROR] boom\n")
            app.navigate_to("tools.text", {"fid": app.prefs.file_ref(str(log_copy))})
            assert await _until(lambda: page.views[-1].route.startswith("/tools/text/"))
            editor_screen = app.shell.stack[-1].screen
            assert isinstance(editor_screen, TextEditorScreen)
            assert await _until(lambda: editor_screen.loaded is not None)
            style = editor_screen.editor.text_style
            assert style.size == MONO_SIZE and style.font_family == ANDROID_MONO
            assert await _until(lambda: wire.encodings(editor_screen.editor))
            for sent in wire.encodings(editor_screen.editor):
                assert (sent.get("text_style") or {}).get("size") == MONO_SIZE, sent

            # Glossaries list that cannot be read: its ErrorCard's exception text stays mono 13
            await app.navigate("/glossary")
            screen = app.shell.top_screen

            def card():
                holder = getattr(screen, "list_holder", None)
                content = getattr(holder, "content", None)
                return content if isinstance(content, ErrorCard) else None

            assert await _until(lambda: card() is not None, 15)
            message = card().message_text
            assert message.value == "PermissionError: Permission denied: 'Glossary'"
            assert message.size == MONO_SIZE == tokens.MONO_STYLE.size and message.font_family == ANDROID_MONO
            assert await _until(lambda: wire.encodings(message))
            for sent in wire.encodings(message):
                assert sent.get("size") == MONO_SIZE and sent.get("font_family") == ANDROID_MONO, sent
        finally:
            await _stop_app(shell)

    asyncio.run(scenario())
