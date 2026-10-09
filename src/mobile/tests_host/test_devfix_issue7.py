"""Acceptance test, owner device report #7 (2026-10-08): QA scan from the chat, Quick Scan, sample size 0.

What the owner saw on the U8 APK: a chat's finished book could not be QA-scanned from the chat (the Result
card's QA button was disabled; ＋ › QA scan and ``/qa`` only opened Tools › QA Scanner, which refuses Direct
Text workspaces), and Tools › QA Scanner offered the desktop duplicate-check sample size 1000.

This test drives the REAL app: ``main.main`` on the in-memory Flet session of test_bootstrap /
test_ui_foundations, i.e. the real GlossarionApp with its ChatFeature over the desktop chat store, the real
JobService running ``job_kinds.qa``, the shared ``qa_scan_runtime`` and the real ``scan_html_folder`` (thread
executor). Only ``scan_html_folder`` / ``run_bulk_qa_scan`` are wrapped, to record what they receive; the
wrappers still run the real code.

* Two finished chat books: one the startup auto-migrate moves into the Library (``Output/<stem>``), one that
  stays in ``Output/Direct Text/<chat>/Attachments/<stem>`` (a different Library book already has its name).
* For each: the Result card's QA button, ＋ › QA scan and ``/qa`` (slash popover tap, and Send) start a
  ``qa_scan`` job in Quick Scan mode; ``scan_html_folder`` receives ``quick_scan_sample_size`` 0 and logs that
  the duplicate check is off; the Direct Text workspace passes through ``run_bulk_qa_scan``'s
  ``allow_direct_text=True`` opt-in (the Library one does not need it); the chat's "QA scan" card opens the
  report in the QA report viewer.
* config.json saved with the desktop 1000 (what the U6-U9 QA screen stored on the phone) becomes 0 once at
  app start; Tools › QA Scanner shows 0 and stays editable (a typed 1000 survives a relaunch, and the next
  chat scan uses it). A config without the key shows 0 in Tools › QA Scanner and Settings › QA Scanner
  Settings, scans with 0, and is never written.
* Desktop: qa_scan_runtime's default stays 1000 and desktop callers still refuse Direct Text.

Real data is never touched: ``app_env`` points the data / output / Library / HOME / CONFIG_FILE paths at
pytest tmp dirs; USERPROFILE and APPDATA are redirected too, GLOSSARION_HTTP_LOG=0.

Run from src/mobile with the mobile venv (``unset PYTHONPATH`` first):
    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_devfix_issue7.py
"""

from __future__ import annotations

import ast
import asyncio
import functools
import importlib.util
import inspect
import json
import os
import sys
import threading
import time
import zipfile
from pathlib import Path

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = MOBILE_DIR / "app"
SRC_DIR = MOBILE_DIR.parent
for _entry in (str(APP_DIR), str(SRC_DIR)):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)


def _has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _load(filename: str, alias: str):
    spec = importlib.util.spec_from_file_location(alias, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _backend_error() -> str:
    try:
        import direct_text_stream  # noqa: F401  (the chat's shared Direct Text code)
        import qa_scan_runtime  # noqa: F401
        import scan_html_folder  # noqa: F401
    except Exception as exc:  # pragma: no cover - a venv without the backend dependencies
        return f"the shared backend / QA scanner is not importable here ({exc})"
    return ""


_BACKEND_ERROR = _backend_error()
needs_backend = pytest.mark.skipif(bool(_BACKEND_ERROR), reason=_BACKEND_ERROR or "backend importable")
needs_flet = pytest.mark.skipif(not (_has("flet") and _has("msgpack")), reason="flet / msgpack not installed")

_TB = _load("test_bootstrap.py", "_glossarion_tb_helpers_devfix_issue7")
storage = _TB.storage  # noqa: F811  (pytest fixtures)
app_env = _TB.app_env

QUICK_KEY = ("qa_scanner_settings", "quick_scan_sample_size")
DUPLICATE_OFF_LINE = "⚡ Quick Scan: duplicate detection disabled (sample size set to 0)"
DESKTOP_SKIP_LINE = ("⏭️ QA scan skipped: Direct Text folders and temporary Direct Text "
                     "outputs are excluded from automatic QA scanning.")
JOB_TIMEOUT = 240.0

TEXT = "The knight walked into the hall and greeted everyone warmly. " * 30
RAW = "기사는 홀로 걸어 들어가 모두에게 따뜻하게 인사했다. " * 30


# ==========================================================================
# Fixtures: a phone's data folder with two finished chat books
# ==========================================================================


def make_epub(path: Path, title: str, chapters: int = 3) -> Path:
    """A small valid EPUB (the chat attachment / Library raw file)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    names = [f"ch{i:03d}.xhtml" for i in range(1, chapters + 1)]
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("mimetype", "application/epub+zip")
        zf.writestr("META-INF/container.xml",
                    '<?xml version="1.0"?><container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:'
                    'container"><rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-'
                    'package+xml"/></rootfiles></container>')
        manifest = "".join(f'<item id="c{i}" href="{n}" media-type="application/xhtml+xml"/>'
                           for i, n in enumerate(names))
        spine = "".join(f'<itemref idref="c{i}"/>' for i in range(len(names)))
        zf.writestr("OEBPS/content.opf",
                    '<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf" version="3.0"><metadata '
                    f'xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>{title}</dc:title><dc:creator>Author'
                    '</dc:creator><dc:language>ko</dc:language></metadata>'
                    f'<manifest>{manifest}</manifest><spine>{spine}</spine></package>')
        for index, name in enumerate(names, 1):
            zf.writestr(f"OEBPS/{name}", f"<html><head><title>{index}화</title></head><body><h1>{index}화</h1>"
                                         f"<p>{RAW}</p></body></html>")
    return path


def finished_workspace(folder: Path, raw: Path, chapters: int = 3) -> Path:
    """A finished translation workspace (response files, progress, ``source_epub.txt``)."""
    folder.mkdir(parents=True, exist_ok=True)
    progress = {}
    for index in range(1, chapters + 1):
        name = f"response_{index:03d}_ch{index:03d}.html"
        (folder / name).write_text(f"<html><head><title>Chapter {index}</title></head><body><h1>Chapter {index}"
                                   f"</h1><p>{TEXT}</p></body></html>", encoding="utf-8")
        progress[str(index)] = {"status": "completed", "output_file": name, "actual_num": index}
    (folder / "translation_progress.json").write_text(json.dumps({"chapters": progress}), encoding="utf-8")
    (folder / "source_epub.txt").write_text(str(raw), encoding="utf-8")
    return folder


def _chat_session(cid: int, title: str, conversation: Path, raw: Path, workspace: Path) -> dict:
    """A desktop ``direct_text_chats.json`` v2 session whose attachment turn finished in ``workspace``."""
    return {
        "id": cid,
        "title": title,
        "messages": [
            ["user_file", raw.name, str(raw), raw.stat().st_size, "", "user"],
            ["assistant", "Chapter one text", "", "Processing", str(workspace),
             "Chapter 1 (chunk 1/1) · ch001.xhtml · Request 1", {"created_at": "2026-10-01T10:11:00+09:00"}],
            ["assistant", "## Extraction report\n- Chapter payloads: 3/3 ready", "", "Processing", str(workspace),
             "Extraction report", {"created_at": "2026-10-01T10:12:00+09:00"}],
        ],
        "draft": "",
        "attachment": None,
        "output_folder": str(conversation),
        "output_folder_name": conversation.name,
        "next_output_index": 2,
        "expanded": [],
    }


class Phone:
    """The seeded data folder (``FLET_APP_STORAGE_DATA``) and the paths the test checks."""

    def __init__(self, data: Path, *, saved_sample=None) -> None:
        self.data = data
        self.out = data / "Output"
        inbox = data / "Inbox"
        self.alpha_raw = make_epub(inbox / "alpha.epub", "Alpha")
        self.beta_raw = make_epub(inbox / "beta.epub", "Beta")
        chat2 = self.out / "Direct Text" / "Migrated novel - 20261001_101010_abcdef12"
        chat5 = self.out / "Direct Text" / "Direct novel - 20261002_101010_abcdef34"
        self.alpha_chat_ws = finished_workspace(chat2 / "Attachments" / "alpha", self.alpha_raw)
        self.beta_ws = finished_workspace(chat5 / "Attachments" / "beta", self.beta_raw)
        self.alpha_ws = self.out / "alpha"  # where auto-migrate moves the first book
        # A different Library book already called "beta": the second chat book stays in Direct Text
        self.other_raw = make_epub(data / "Elsewhere" / "beta.epub", "Another Beta", chapters=2)
        self.library_beta = finished_workspace(self.out / "beta", self.other_raw, chapters=2)
        history = {"version": 2, "current_chat_id": 2, "sessions": [
            _chat_session(2, "Migrated novel", chat2, self.alpha_raw, self.alpha_chat_ws),
            _chat_session(5, "Direct novel", chat5, self.beta_raw, self.beta_ws),
        ]}
        (data / "direct_text_chats.json").write_text(json.dumps(history, ensure_ascii=False), encoding="utf-8")
        qa_settings: dict = {"check_ai_truncation_detection": False}
        if saved_sample is not None:
            qa_settings["quick_scan_sample_size"] = saved_sample
        # A returning user (the owner ran jobs on the U8 APK): the first-job Android prompts (notification
        # permission, "Keep translations running") were answered then, so a tap starts the job at once.
        state_path = data / "mobile_state.json"
        prefs = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
        prefs.update({"welcome_completed": True, "jobs_battery_prompt_done": True,
                      "jobs_notification_permission_asked": True, "jobs_notifications_off_hint_shown": True})
        state_path.write_text(json.dumps(prefs), encoding="utf-8")
        self.config_path = data / "config.json"
        self.config_path.write_text(json.dumps({"output_language": "English", "qa_scanner_settings": qa_settings}),
                                    encoding="utf-8")

    def saved_config(self) -> dict:
        return json.loads(self.config_path.read_text(encoding="utf-8"))


class ScanRecorder:
    """Wraps the shared scanner entry points (the real code still runs) and records what they receive."""

    def __init__(self, monkeypatch) -> None:
        import qa_scan_runtime
        import scan_html_folder as shf

        self.scans: list = []
        self.loops: list = []
        self._lock = threading.Lock()
        real_scan, real_loop = shf.scan_html_folder, qa_scan_runtime.run_bulk_qa_scan

        @functools.wraps(real_scan)
        def scan_html_folder(folder_path, log=print, stop_flag=None, mode="quick-scan", qa_settings=None,
                             epub_path=None, **kwargs):
            lines: list = []

            def tee(message="", *args, **kw):
                lines.append(str(message))
                return log(message, *args, **kw)

            record = {"folder": os.path.abspath(str(folder_path)), "mode": mode, "settings": dict(qa_settings or {}),
                      "epub": os.path.abspath(epub_path) if epub_path else None, "lines": lines,
                      "env": {k: os.environ.get(k) for k in ("QA_USE_THREAD_EXECUTOR", "GLOSSARION_NO_PROCESSES",
                                                              "GLOSSARION_MOBILE")}}
            with self._lock:
                self.scans.append(record)
            return real_scan(folder_path, log=tee, stop_flag=stop_flag, mode=mode, qa_settings=qa_settings,
                             epub_path=epub_path, **kwargs)

        @functools.wraps(real_loop)  # keeps the signature (qa_model checks it for the opt-in)
        def run_bulk_qa_scan(folders_to_scan, **kwargs):
            with self._lock:
                self.loops.append({"folders": [os.path.abspath(str(f)) for f in folders_to_scan],
                                   "mode": kwargs.get("mode"), "passed": "allow_direct_text" in kwargs,
                                   "allow_direct_text": kwargs.get("allow_direct_text", False)})
            return real_loop(folders_to_scan, **kwargs)

        monkeypatch.setattr(shf, "scan_html_folder", scan_html_folder)
        monkeypatch.setattr(qa_scan_runtime, "run_bulk_qa_scan", run_bulk_qa_scan)


@pytest.fixture
def phone_env(app_env, storage, tmp_path, monkeypatch):
    """The real app's isolated storage (``app_env``) + USERPROFILE / APPDATA in tmp; the recorder."""
    for name, sub in (("USERPROFILE", "home"), ("APPDATA", "appdata"), ("LOCALAPPDATA", "localappdata")):
        folder = tmp_path / "isolated" / sub
        folder.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv(name, str(folder))
    monkeypatch.setenv("GLOSSARION_HTTP_LOG", "0")
    return storage["data"], ScanRecorder(monkeypatch)


# ==========================================================================
# Driving the real app
# ==========================================================================


def _tf():
    return _load("test_ui_foundations.py", "_glossarion_tf_helpers_devfix_issue7")


async def _until(predicate, timeout: float = 20.0, what="") -> None:
    """Poll ``predicate`` on the app's loop; ``what`` (text, or a callable read at the timeout) explains."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.05)
    assert predicate(), f"timed out waiting for {what() if callable(what) else what}"


def _same(a, b) -> bool:
    return os.path.normcase(os.path.abspath(str(a))) == os.path.normcase(os.path.abspath(str(b)))


async def _open_chat(app, cid: str) -> None:
    app._open_chat(cid)
    await _until(lambda: app.chat_view.cid == cid and app.shell.current_route == f"/chat/{cid}", 10,
                 f"chat {cid}")


def _result_card(app, name: str):
    """The transcript's latest JobCard titled ``name`` (None when there is none)."""
    from glossarion_mobile.ui.chat.cards import JobCard

    cards = [c for c in app.chat_view.transcript.cards if isinstance(c, JobCard) and c.title_text.value == name]
    return cards[-1] if cards else None


def _enabled_qa_button(card):
    """The Result card's "QA scan" action as a tappable button (U8: a disabled button + ReasonChip row)."""
    import flet as ft

    button = card.action_buttons.get("qa")
    assert isinstance(button, (ft.FilledTonalButton, ft.FilledButton, ft.TextButton, ft.OutlinedButton)), (
        f"the Result card's QA action is not a button: {type(button).__name__}")
    assert not button.disabled and button.on_click is not None, "the Result card's QA button is disabled"
    return button


def _qa_messages(app, cid: str) -> list:
    from glossarion_mobile.ui.chat.transcript_model import QA_LABEL

    return [m for m in app.chat_feature.chats.messages(cid) if len(m) > 5 and m[5] == QA_LABEL]


def _record_notes(app) -> list:
    """The chat view's snackbars (the real ones still show)."""
    view = app.chat_view
    notes = getattr(view, "_devfix_issue7_notes", None)
    if notes is None:
        notes = []
        real = view.notify

        def notify(message, action_label=None, on_action=None):
            notes.append(str(message))
            return real(message, action_label=action_label, on_action=on_action)

        view.notify = notify
        view._devfix_issue7_notes = notes
    return notes


async def _chat_qa(app, recorder: ScanRecorder, cid: str, trigger) -> tuple:
    """Run ``trigger`` (a tap), then wait for the chat's QA job to finish: (snapshot, storage, scan record)."""
    view = app.chat_view
    notes = _record_notes(app)
    before_cards, before_scans, before_notes = len(_qa_messages(app, cid)), len(recorder.scans), len(notes)
    trigger()
    await _until(lambda: len(_qa_messages(app, cid)) > before_cards, 20,
                 lambda: f"the chat's 'QA scan' message (snackbars: {notes[before_notes:]}, "
                         f"route {app.shell.current_route})")
    stored = view._tool_storage(_qa_messages(app, cid)[-1])
    job_id = str(stored.get("qa_job") or "")
    assert job_id, stored
    service = app.job_service

    def ended() -> bool:
        snap = service.snapshot(job_id)
        return snap is not None and snap.is_terminal

    await _until(ended, JOB_TIMEOUT, f"QA job {job_id}")
    snap = service.snapshot(job_id)
    assert snap.state.value == "DONE", (snap.state, snap.last_line if hasattr(snap, "last_line") else "")
    scans = recorder.scans[before_scans:]
    assert len(scans) == 1, [s["folder"] for s in scans]
    return snap, stored, scans[0]


def _assert_chat_scan(snap, stored, scan, recorder, *, folder: Path, source: Path, direct_text: bool,
                      sample: int, cid: str, chat_title: str) -> None:
    """What the owner asked for: Quick Scan, the effective sample size, the turn's workspace and source."""
    spec = snap.spec
    assert spec.kind == "qa_scan" and spec.params["mode"] == "quick-scan"
    assert spec.inputs == (str(folder),)
    target = spec.params["targets"][0]
    assert _same(target["folder"], folder) and _same(target["source"], source)
    assert bool(target.get("direct_text")) is direct_text
    assert spec.origin["type"] == "chat" and spec.origin["cid"] == cid and spec.origin["label"] == f"Chat · {chat_title}"
    assert "chat_id" not in spec.params  # the chat's JobStrip shows it (no chat job card of its own)
    # what the real scanner received
    assert _same(scan["folder"], folder) and scan["mode"] == "quick-scan" and _same(scan["epub"], source)
    assert scan["settings"]["quick_scan_sample_size"] == sample
    assert scan["env"]["QA_USE_THREAD_EXECUTOR"] == "1" and scan["env"]["GLOSSARION_MOBILE"] == "1"
    if sample == 0:
        assert DUPLICATE_OFF_LINE in scan["lines"], scan["lines"][-15:]
    else:
        assert not any("duplicate detection disabled" in line for line in scan["lines"])
    loop = recorder.loops[-1]
    assert loop["folders"] == [os.path.abspath(str(folder))] and loop["mode"] == "quick-scan"
    # only a chat-submitted Direct Text workspace opts out of the shared Direct Text guard
    assert loop["allow_direct_text"] is direct_text and loop["passed"] is direct_text
    report = Path(folder) / f"{Path(folder).name}_Scan Report" / "validation_results.html"
    assert report.is_file() and tuple(snap.outputs) == (str(report),)
    assert stored["folder"] == str(folder) and stored["source"] == str(source)
    summary = "duplicate check off (sample size 0)" if sample == 0 else f"duplicate check sample size {sample}"
    assert stored["summary"] == f"Quick Scan · {summary}"


async def _open_report_from_card(app, name: str, report: Path) -> None:
    """The chat's "QA scan" card › Report opens the scan's report in the QA report viewer."""
    from glossarion_mobile.ui.chat.cards import JobCard
    from glossarion_mobile.ui.tools.qa_report import QaReportScreen

    def card_done():
        card = _result_card(app, f"QA scan · {name}")
        return card if card is not None and any(t.value == "Done" for t in _texts(card)) else None

    await _until(lambda: card_done() is not None, 15, "the 'QA scan' card to say Done")
    card = card_done()
    assert isinstance(card, JobCard)
    buttons = {b.key: b for b in card.buttons.controls}
    assert {"qajob-job", "qajob-report", "qajob-chapters"} <= set(buttons)
    buttons["qajob-report"].on_click(None)
    await _until(lambda: app.shell.current_route.startswith("/tools/qa/report/"), 10, "the QA report route")
    screen = app.shell.top_screen
    assert isinstance(screen, QaReportScreen) and _same(screen.path, report)
    await _until(lambda: screen.summary is not None and screen.renderer, 20, "the report to load")
    assert screen.summary.total == 3 and screen.renderer in ("native", "webview")


def _texts(control) -> list:
    """Every ft.Text below ``control`` (the card's status line)."""
    import flet as ft

    found, stack, seen = [], [control], set()
    while stack:
        node = stack.pop()
        if node is None or id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, ft.Text):
            found.append(node)
        for attr in ("content", "controls", "title", "subtitle", "leading", "trailing", "label"):
            child = getattr(node, attr, None)
            if isinstance(child, (list, tuple)):
                stack.extend(child)
            elif child is not None and hasattr(child, "__dict__"):
                stack.append(child)
    return found


def _slash_popover_tap(app, command: str) -> None:
    """Type ``/qa`` in the composer and tap the command in the slash popover."""
    composer = app.chat_view.composer
    composer.handle_text(f"/{command}")
    names = [c.name for c in composer.slash.commands]
    assert command in names, names
    composer.slash.list.controls[names.index(command)].on_click(None)


def _slash_send(app, command: str) -> None:
    """Type ``/qa`` and tap Send (the composer runs a complete command instead of sending it)."""
    from glossarion_mobile.ui.chat.send_state import SendAction, SendState

    composer = app.chat_view.composer
    composer.handle_text(f"/{command}")
    assert composer.send_button.state is SendState.IDLE_READY, composer.send_button.state
    assert composer.send_button.tap() is SendAction.SEND
    assert composer.text == ""  # the command left the field


def _plus_sheet_qa(app) -> None:
    """＋ › Tools › QA scan."""
    sheet = app.chat_view.open_plus_sheet()
    sheet.tool_tiles["qa"].on_click(None)


async def _tools_qa_screen(app):
    from glossarion_mobile.ui.tools.qa_screen import QaScannerScreen

    app.navigate_to("tools.qa")
    await _until(lambda: isinstance(app.shell.top_screen, QaScannerScreen), 10, "Tools › QA Scanner")
    screen = app.shell.top_screen
    await _until(lambda: getattr(screen, "sample_field", None) is not None, 10, "the sample size field")
    return screen


async def _settings_qa_tile(app):
    """Settings › QA Scanner Settings: the sample size tile (the section page builds its tiles in windows)."""
    from glossarion_mobile.ui.settings.section_page import SectionPage

    app.navigate_to("settings.section", {"section": "qa.settings"})
    await _until(lambda: isinstance(app.shell.top_screen, SectionPage), 10, "Settings › QA Scanner Settings")
    page = app.shell.top_screen
    await _until(lambda: page.keys, 10, "the section's settings")
    key = ".".join(QUICK_KEY)
    assert key in page.keys
    return page.tile(key)


async def _shutdown(tf, app) -> None:
    """What the app does when it goes to the background (settings integration on_lifecycle): flush
    config.json AND Prefs. Without the Prefs flush a fast runner lost the one-time migration flag."""
    try:
        for name in ("config_store", "prefs"):
            target = getattr(app, name, None)
            if target is not None:
                target.flush()
    finally:
        app.jobs.close()
        await tf._stop(app)


# ==========================================================================
# The owner's phone: config.json holds the desktop 1000
# ==========================================================================


@needs_flet
@needs_backend
def test_owner_phone_chat_qa_quick_scan_sample_size_zero(phone_env):
    """Both chat books scan from the Result card, ＋ › QA scan and /qa in Quick Scan with sample size 0 (the
    saved desktop 1000 is migrated once at app start); the reports open; Tools › QA Scanner shows 0 and stays
    editable; after a relaunch the typed value is kept and the chat's next scan uses it."""
    from glossarion_mobile.job_kinds import qa as qa_kind
    from glossarion_mobile.ui.tools import qa_model

    data, recorder = phone_env
    phone = Phone(data, saved_sample=1000)
    tf = _tf()

    async def first_launch():
        _m, _conn, _session, _page, app = await tf._start("android")
        try:
            await tf._wait(lambda: app.state.engine_ready, timeout=15)
            # --- the one-time mobile migration ran at app start (before any QA screen was opened)
            assert app.config_store.get(QUICK_KEY) == 0
            assert app.prefs.get(qa_model.QUICK_SAMPLE_MIGRATION_PREF) is True
            # --- the startup auto-migrate: alpha joins the Library, beta stays (name taken by another book)
            sweep = getattr(app.chat_feature, "startup_task", None)
            if sweep is not None:
                await asyncio.wait_for(asyncio.shield(sweep), 30)
            await _until(lambda: phone.alpha_ws.is_dir() and not phone.alpha_chat_ws.exists(), 30,
                         "alpha's auto-migrate into the Library")
            assert phone.beta_ws.is_dir() and _same(Path(phone.library_beta / "source_epub.txt").read_text(
                encoding="utf-8"), phone.other_raw)
            import qa_scan_runtime

            assert not qa_scan_runtime.is_direct_text_qa_path(str(phone.alpha_ws))
            assert qa_scan_runtime.is_direct_text_qa_path(str(phone.beta_ws))

            # ===== chat 2: the auto-migrated book (Output/alpha) =====
            await _open_chat(app, "2")
            await _until(lambda: _result_card(app, "alpha.epub") is not None, 10, "alpha's Result card")
            qa_button = _enabled_qa_button(_result_card(app, "alpha.epub"))
            from glossarion_mobile.ui.chat.cards import ATTACHMENT_ACTION_REASONS

            assert "qa" not in ATTACHMENT_ACTION_REASONS
            common = dict(folder=phone.alpha_ws, source=phone.alpha_raw, direct_text=False, sample=0, cid="2",
                          chat_title="Migrated novel")
            # Result card › QA scan
            result = await _chat_qa(app, recorder, "2", lambda: qa_button.on_click(None))
            _assert_chat_scan(*result, recorder, **common)
            await _open_report_from_card(app, "alpha", Path(qa_kind.report_path_for(str(phone.alpha_ws))))
            # ＋ › QA scan
            await _open_chat(app, "2")
            result = await _chat_qa(app, recorder, "2", lambda: _plus_sheet_qa(app))
            _assert_chat_scan(*result, recorder, **common)
            # /qa from the slash popover
            result = await _chat_qa(app, recorder, "2", lambda: _slash_popover_tap(app, "qa"))
            _assert_chat_scan(*result, recorder, **common)

            # ===== chat 5: the book still in Output/Direct Text/<chat>/Attachments/beta =====
            await _open_chat(app, "5")
            await _until(lambda: _result_card(app, "beta.epub") is not None, 10, "beta's Result card")
            qa_button = _enabled_qa_button(_result_card(app, "beta.epub"))
            common = dict(folder=phone.beta_ws, source=phone.beta_raw, direct_text=True, sample=0, cid="5",
                          chat_title="Direct novel")
            result = await _chat_qa(app, recorder, "5", lambda: qa_button.on_click(None))
            _assert_chat_scan(*result, recorder, **common)
            # the scan reached the real scanner: no desktop "skipped" line, and the report opens
            assert not any("QA scan skipped" in line or "Skipping Direct Text" in line
                           for line in recorder.scans[-1]["lines"])
            await _open_report_from_card(app, "beta", Path(qa_kind.report_path_for(str(phone.beta_ws))))
            await _open_chat(app, "5")
            result = await _chat_qa(app, recorder, "5", lambda: _plus_sheet_qa(app))
            _assert_chat_scan(*result, recorder, **common)
            result = await _chat_qa(app, recorder, "5", lambda: _slash_popover_tap(app, "qa"))
            _assert_chat_scan(*result, recorder, **common)
            # the Library's other "beta" was never scanned by these chat scans
            assert not any(_same(s["folder"], phone.library_beta) for s in recorder.scans)

            # ===== Settings › QA Scanner Settings and Tools › QA Scanner: 0 shown, still editable =====
            tile = await _settings_qa_tile(app)
            assert tile.field.value == "0" and tile.editable
            screen = await _tools_qa_screen(app)
            assert screen.sample_field.value == "0" and screen.mode == "quick-scan"
            screen.sample_field.value = "1000"  # the owner types the desktop value back on purpose
            screen._on_sample()
            assert app.config_store.get(QUICK_KEY) == 1000
        finally:
            await _shutdown(tf, app)

    asyncio.run(first_launch())
    saved = phone.saved_config()
    assert saved["qa_scanner_settings"]["quick_scan_sample_size"] == 1000  # the owner's own choice, on disk
    assert saved["qa_scanner_settings"]["check_ai_truncation_detection"] is False  # nothing else rewritten

    async def relaunch():
        _m, _conn, _session, _page, app = await tf._start("android")
        try:
            await tf._wait(lambda: app.state.engine_ready, timeout=15)
            # the migration is one-time: the 1000 the owner typed stays
            assert app.config_store.get(QUICK_KEY) == 1000
            screen = await _tools_qa_screen(app)
            assert screen.sample_field.value == "1000"
            assert (await _settings_qa_tile(app)).field.value == "1000"
            # the chat uses the same effective value the owner set in Tools › QA Scanner
            await _open_chat(app, "5")
            result = await _chat_qa(app, recorder, "5", lambda: _slash_popover_tap(app, "qa"))
            _assert_chat_scan(*result, recorder, folder=phone.beta_ws, source=phone.beta_raw, direct_text=True,
                              sample=1000, cid="5", chat_title="Direct novel")
            # and back to 0 from the same field (0 = duplicate check off)
            screen = await _tools_qa_screen(app)
            screen.sample_field.value = "0"
            screen._on_sample()
            assert app.config_store.get(QUICK_KEY) == 0
            await _open_chat(app, "2")
            # the owner is signed in with ChatGPT (the default model), so Send is enabled: /qa + Send
            app.state.signed_in.set(frozenset({"authgpt"}))
            result = await _chat_qa(app, recorder, "2", lambda: _slash_send(app, "qa"))
            _assert_chat_scan(*result, recorder, folder=phone.alpha_ws, source=phone.alpha_raw, direct_text=False,
                              sample=0, cid="2", chat_title="Migrated novel")
        finally:
            await _shutdown(tf, app)

    asyncio.run(relaunch())
    assert phone.saved_config()["qa_scanner_settings"]["quick_scan_sample_size"] == 0


# ==========================================================================
# A config without the key: 0 shown and used, never written
# ==========================================================================


@needs_flet
@needs_backend
def test_unset_sample_size_is_zero_on_mobile_and_never_written(phone_env):
    """No saved sample size: Tools › QA Scanner and Settings › QA Scanner Settings show 0, a chat scan runs
    with 0 ("Glossarion Mobile default" in the job log) and config.json gets no value the owner never chose."""
    from glossarion_mobile.ui.tools import qa_model

    data, recorder = phone_env
    phone = Phone(data, saved_sample=None)
    tf = _tf()

    async def scenario():
        _m, _conn, _session, _page, app = await tf._start("android")
        try:
            await tf._wait(lambda: app.state.engine_ready, timeout=15)
            assert app.config_store.get(QUICK_KEY) is None
            assert app.prefs.get(qa_model.QUICK_SAMPLE_MIGRATION_PREF) is True  # ran, nothing to move
            # Settings shows the value the scans use (display default; the desktop schema keeps 1000)
            assert app.config_store.effective(QUICK_KEY) == 0
            tile = await _settings_qa_tile(app)
            assert tile.field.value == "0" and tile.editable and not tile.stored
            screen = await _tools_qa_screen(app)
            assert screen.sample_field.value == "0"
            screen._on_sample()  # blur on the untouched default: still nothing saved
            assert app.config_store.get(QUICK_KEY) is None
            await _open_chat(app, "5")
            await _until(lambda: _result_card(app, "beta.epub") is not None, 10, "beta's Result card")
            qa_button = _enabled_qa_button(_result_card(app, "beta.epub"))
            snap, stored, scan = await _chat_qa(app, recorder, "5", lambda: qa_button.on_click(None))
            _assert_chat_scan(snap, stored, scan, recorder, folder=phone.beta_ws, source=phone.beta_raw,
                              direct_text=True, sample=0, cid="5", chat_title="Direct novel")
            log_path = app.job_service.job_log_path(snap.id)
            log_text = Path(log_path).read_text(encoding="utf-8", errors="replace")
            assert ("⚡ Quick Scan duplicate check sample size: 0 (duplicate check off) · Glossarion Mobile default"
                    in log_text)
            assert app.config_store.get(QUICK_KEY) is None
        finally:
            await _shutdown(tf, app)

    asyncio.run(scenario())
    assert "quick_scan_sample_size" not in phone.saved_config()["qa_scanner_settings"]


# ==========================================================================
# Desktop: the shared default stays 1000 and Direct Text stays refused
# ==========================================================================


@needs_backend
def test_desktop_keeps_1000_and_refuses_direct_text(tmp_path, monkeypatch):
    import qa_scan_runtime
    import scan_html_folder as shf

    for name, sub in (("HOME", "home"), ("USERPROFILE", "home"), ("APPDATA", "appdata")):
        (tmp_path / sub).mkdir(exist_ok=True)
        monkeypatch.setenv(name, str(tmp_path / sub))
    monkeypatch.setenv("OUTPUT_DIRECTORY", str(tmp_path / "Output"))
    monkeypatch.setenv("CONFIG_FILE", str(tmp_path / "config.json"))
    monkeypatch.setenv("GLOSSARION_HTTP_LOG", "0")
    # the desktop default and normaliser are unchanged
    assert qa_scan_runtime.default_qa_scan_settings()["quick_scan_sample_size"] == 1000
    assert qa_scan_runtime.normalize_qa_scan_settings({})["quick_scan_sample_size"] == 1000
    assert qa_scan_runtime.load_current_qa_settings({})["quick_scan_sample_size"] == 1000
    # the opt-in is keyword-only in practice and off by default on both shared entry points
    for fn in (qa_scan_runtime.run_qa_scan_path, qa_scan_runtime.run_bulk_qa_scan):
        assert inspect.signature(fn).parameters["allow_direct_text"].default is False
    calls: list = []
    monkeypatch.setattr(shf, "scan_html_folder", lambda folder, **kw: calls.append(folder) or None)
    workspace = tmp_path / "Output" / "Direct Text" / "Novel - 20261001_101010_abcdef12" / "Attachments" / "Book"
    workspace.mkdir(parents=True)
    (workspace / "response_001_ch001.html").write_text("<p>one</p>", encoding="utf-8")
    logs: list = []
    # TransateKRtoEN's call shape (automatic QA after a translation)
    assert qa_scan_runtime.run_qa_scan_path(str(workspace), log=logs.append, mode="quick-scan") is None
    assert logs == [DESKTOP_SKIP_LINE] and calls == []
    # QA_Scanner_GUI's call shape (the bulk loop, no opt-in)
    logs.clear()
    qa_scan_runtime.run_bulk_qa_scan(
        [str(workspace)], mode="quick-scan", epub_path=None, qa_settings={}, load_settings=lambda: {},
        selected_mode_value="quick-scan", disable_word_count_for_run=False, epub_basename_map={},
        global_selected_files=None, log=logs.append, stop_flag=lambda: False)
    assert calls == [] and DESKTOP_SKIP_LINE in logs
    # only the explicit opt-in lets a Direct Text folder through
    qa_scan_runtime.run_qa_scan_path(str(workspace), log=logs.append, allow_direct_text=True)
    assert calls == [str(workspace)]
    # no desktop caller passes the opt-in (Glossarion Mobile's job_kinds.qa is the only one)
    passing: list = []
    for path in sorted(SRC_DIR.glob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                name = getattr(func, "attr", None) or getattr(func, "id", None)
                if name in ("run_qa_scan_path", "run_bulk_qa_scan") and path.name != "qa_scan_runtime.py":
                    if any(k.arg == "allow_direct_text" for k in node.keywords):
                        passing.append(f"{path.name}:{node.lineno}")
    assert passing == []
    for caller in ("QA_Scanner_GUI.py", "TransateKRtoEN.py"):
        text = (SRC_DIR / caller).read_text(encoding="utf-8-sig")
        assert "allow_direct_text" not in text, caller
    # the Glossarion Mobile default lives in the mobile job kind, not in the shared module
    from glossarion_mobile.job_kinds import qa as qa_kind

    assert qa_kind.MOBILE_QUICK_SAMPLE_SIZE == 0
    assert "MOBILE_QUICK_SAMPLE_SIZE" not in (SRC_DIR / "qa_scan_runtime.py").read_text(encoding="utf-8-sig")
