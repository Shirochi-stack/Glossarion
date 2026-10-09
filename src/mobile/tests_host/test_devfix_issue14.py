"""Acceptance test, owner device report #14 (devfix4, owner decisions 2026-10-08): ONE Streaming toggle on
Glossarion Mobile, ON by default, driving the four desktop streaming keys and the desktop thoughts coupling;
OFF stops streaming everywhere on the phone.

What the owner saw on the U8/U9 APK: Settings showed the desktop "Real-time Translation (Streaming)" group as
four separate switches (Enable streaming responses, Stream thinking/reasoning logs, Allow streaming logs during
batch mode, Allow forced-stream batch log) plus a separate "Enable thoughts" in Thinking & reasoning, all OFF on a
fresh install, and the chat and the Reader's live translation streamed whatever they said.

This file proves the fix end to end on the REAL app objects: ``main.main`` on the in-memory Flet session of
test_bootstrap / test_ui_foundations (as Android, 412 dp), driven through ``tests/ui_driver.UiDriver`` +
``tests/host_tester.PyTester`` (the device flows' finders and taps: a Switch tap toggles it and sends
``change`` like Flutter), with the real MobileConfigStore, Settings pages and search, Import from desktop,
env preview, ChatFeature / ChatRuns, JobService, HeadlessOwner, the shared desktop pipeline, LibraryService
and ReaderScreen. Only the model is the offline fake OpenAI server (``diagnostics.fake_llm_server``) on
127.0.0.1, which records whether each request asked for a stream.

1. ``test_fresh_phone_one_streaming_switch_on_by_default``: a fresh install shows ONE "Streaming" tile, ON
   ("On · default"), first in Settings › Response handling & retries, no tile for the four desktop toggles or
   Enable thoughts on any Settings page, one search hit for every desktop name / env var; the effective values
   of the four keys and Enable thoughts are ON while config.json holds none of them; the env preview of the next
   run exports the four streaming env vars as 1. Tapping it OFF makes ONE store write of exactly the keys that
   change (the four toggles and enable_thoughts, unchecked as the desktop checkbox does), config.json gets no
   'streaming' key and nothing else changes; the env preview then exports 0. ON again writes them back.
2. ``test_streaming_off_stops_streaming_in_chats_book_jobs_and_the_reader``: with the default (ON) a chat
   message, a chat attachment (book in the chat), a Library book job, a Library glossary extraction and the
   Reader's 🌐 Translate all send streamed requests (the chat and Reader forcing every streaming switch, desktop
   parity); after the Settings tap OFF the same five runs send no streamed request (the chat's reply still
   reaches the transcript, whole), the chat run's environment
   has no forced streaming, the book jobs export the four streaming env vars as 0 and the Reader's live panel
   says streaming is off.
3. ``test_imported_mixed_desktop_config_shows_custom_and_one_tap_normalises``: a desktop config.json whose four
   toggles differ, imported through Settings › Import from desktop, shows "Custom · N of 4 on" with the switch
   following Enable streaming responses; one tap sets all four (only the changed keys are written: no
   no-op stream-thinking write that would uncheck Enable thoughts); a config with stream thinking on and
   Enable thoughts off gets the desktop thoughts lock back on ON.
4. ``test_desktop_other_settings_streaming_texts_identical``: the two desktop notes moved into settings_rules
   render exactly as before in Other Settings (offscreen Qt, the section built by the real builder, compared
   with the pre-move other_settings.py from git when the history is available); the desktop keeps its four
   checkboxes, unchecked for a config without the keys (the desktop default stays off); the mobile-only
   divergences are recorded in tests/parity/DISCREPANCIES.md.

Real data is never touched: ``app_env`` points the data / output / Library / HOME / CONFIG_FILE paths at
pytest tmp dirs; USERPROFILE / APPDATA / LOCALAPPDATA are redirected too, GLOSSARION_HTTP_LOG=0, and the
repo's src/config.json must keep its md5.

Run from src/mobile with the mobile venv::

    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_devfix_issue14.py
"""

from __future__ import annotations

import ast
import asyncio
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import textwrap
import threading
import time
import types
from pathlib import Path

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = MOBILE_DIR / "app"
SRC_DIR = MOBILE_DIR.parent
REPO_DIR = SRC_DIR.parent
TESTS_DIR = MOBILE_DIR / "tests"
for _entry in (str(APP_DIR), str(SRC_DIR), str(TESTS_DIR)):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)


def _has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


_UI_NEEDED = ("flet", "msgpack")
_RUN_NEEDED = _UI_NEEDED + ("ebooklib", "openai", "httpx", "tiktoken", "bs4", "lxml")
needs_flet = pytest.mark.skipif(not all(_has(m) for m in _UI_NEEDED), reason="flet / msgpack not installed")
needs_runs = pytest.mark.skipif(not all(_has(m) for m in _RUN_NEEDED),
                                reason=f"needs {', '.join(_RUN_NEEDED)} (the mobile project venv)")


def _load(alias: str, file_name: str):
    spec = importlib.util.spec_from_file_location(alias, Path(__file__).with_name(file_name))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_UF = _load("_glossarion_uiflows_devfix_issue14", "test_ui_flows.py")  # _host_driver / _foundations
storage = _UF.storage  # noqa: F811  (pytest fixtures)
app_env = _UF.app_env

#: The desktop "Real-time Translation (Streaming)" group (other_settings._create_response_handling_section).
FOUR = ("enable_streaming", "stream_thinking_logs", "allow_batch_stream_logs", "allow_authgpt_batch_stream_logs")
#: What the one switch writes: the four toggles and Enable thoughts (the stream-thinking lock).
WRITES = FOUR + ("enable_thoughts",)
#: The env names a run exports for the four toggles (run_env / translation_pipeline).
STREAM_ENV = ("ENABLE_STREAMING", "STREAM_THINKING_LOGS", "ALLOW_BATCH_STREAM_LOGS", "ALLOW_AUTHGPT_BATCH_STREAM_LOGS")
#: The literal desktop texts other_settings.py had before the owner-approved move (HEAD 55c46555).
DESKTOP_WARNING = "⚠️ Enabling this may result in silent truncation"
DESKTOP_NOTE = ("\U0001f510 AuthGPT, AuthGrok, AuthGem, AuthCD, Arena, Antigravity, and OcAgy always stream "
                "— this controls batch log visibility")
PRE_MOVE_SHA = "55c46555"  # the commit before devfix4 (desktop other_settings.py with the literals)
RUN_TIMEOUT = 120.0
_REPO_CONFIG = SRC_DIR / "config.json"


def _md5(path: Path):
    return hashlib.md5(path.read_bytes()).hexdigest() if path.is_file() else None


# ==========================================================================
# Fixtures
# ==========================================================================


@pytest.fixture
def phone(tmp_path, monkeypatch, request):
    """The real app's isolated storage (``app_env``: FLET_APP_STORAGE_*, bootstrap's HOME / OUTPUT_DIRECTORY /
    GLOSSARION_LIBRARY_DIR / GLOSSARION_DATA_DIR / CONFIG_FILE under tmp), the Windows profile folders in tmp,
    a returning user's prefs (the first-job Android prompts answered, as on the owner's phone) and a picks
    folder for the file picker. src/config.json must keep its md5."""
    repo_md5 = _md5(_REPO_CONFIG)
    user = tmp_path / "user"
    for key, sub in (("USERPROFILE", "."), ("APPDATA", "AppData/Roaming"), ("LOCALAPPDATA", "AppData/Local")):
        folder = (user / sub).resolve()
        folder.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv(key, str(folder))
    monkeypatch.setenv("GLOSSARION_HTTP_LOG", "0")
    dirs = request.getfixturevalue("storage")
    request.getfixturevalue("app_env")
    data = Path(dirs["data"])
    state_path = data / "mobile_state.json"
    prefs = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
    prefs.update({"welcome_completed": True, "jobs_battery_prompt_done": True,
                  "jobs_notification_permission_asked": True, "jobs_notifications_off_hint_shown": True})
    state_path.write_text(json.dumps(prefs), encoding="utf-8")
    picks = tmp_path / "picks"
    picks.mkdir()
    yield types.SimpleNamespace(tmp=tmp_path, data=data, picks=picks, config=data / "config.json")
    for name in ("GLOSSARION_LIBRARY_DIR", "OUTPUT_DIRECTORY", "HOME", "CONFIG_FILE", "GLOSSARION_DATA_DIR"):
        value = os.environ.get(name)
        if value:  # still the sandbox while the test's env is in place
            assert _inside(value, tmp_path), (name, value)
    assert _md5(_REPO_CONFIG) == repo_md5, "src/config.json changed"


# ==========================================================================
# Helpers
# ==========================================================================


def _inside(path, root) -> bool:
    try:
        a = os.path.normcase(os.path.abspath(str(path)))
        b = os.path.normcase(os.path.abspath(str(root)))
        return os.path.commonpath([a, b]) == b
    except ValueError:
        return False


async def _until(predicate, timeout: float = 30.0, what="") -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.05)
    assert predicate(), f"timed out waiting for {what() if callable(what) else what}"


async def _start_app(files: dict):
    import flows

    tf = _UF._foundations()
    app, tester, driver = await _UF._host_driver(tf, files)
    await flows.wait_home(driver)
    await tf._wait(lambda: app.state.engine_ready, timeout=15)
    return tf, app, tester, driver


async def _stop_app(tf, app) -> None:
    try:
        store = getattr(app, "config_store", None)
        if store is not None:
            store.flush()
    finally:
        try:
            app.jobs.close()
        finally:
            await tf._stop(app)


def _disk(phone) -> dict:
    return json.loads(phone.config.read_text(encoding="utf-8")) if phone.config.is_file() else {}


class StoreWrites:
    """Every write the app's MobileConfigStore receives (the real methods still run)."""

    def __init__(self, store) -> None:
        self.store = store
        self.calls: list = []
        for name in ("set_many", "set", "unset"):
            real = getattr(store, name)
            setattr(store, name, self._wrap(name, real))

    def _wrap(self, name, real):
        def wrapper(*args, **kwargs):
            if name == "set_many":
                self.calls.append(("set_many", dict(args[0])))
            elif name == "set":
                self.calls.append(("set", {args[0]: args[1]}))
            else:
                self.calls.append(("unset", args[0]))
            return real(*args, **kwargs)
        return wrapper

    def take(self) -> list:
        calls, self.calls = self.calls, []
        return calls


async def _settings_section(app, driver, section_id: str):
    """Settings home (drawer › Settings) › the section's tile, as the owner opens it."""
    import flows
    from glossarion_mobile.ui.settings.section_page import SectionPage

    await flows.open_settings(driver)
    await driver.tap(key=f"settings-section-{section_id}", timeout=30, scroll=True)

    def opened():
        screen = app.shell.top_screen
        return isinstance(screen, SectionPage) and screen.section_id == section_id and bool(screen.keys)

    await _until(opened, 15, f"Settings › {section_id}")
    return app.shell.top_screen


async def _streaming_tile(app, driver):
    """Settings › Response handling & retries and its Streaming tile (asserting the page shows ONE switch)."""
    from glossarion_mobile.ui.settings.tiles import StreamingTile

    page = await _settings_section(app, driver, "other.response")
    assert page.keys[0] == "streaming", page.keys[:6]
    assert page.headings.get("streaming") == "Streaming", page.headings
    assert not set(WRITES) & set(page.keys), "a desktop streaming key still has its own tile"
    tile = page.tile("streaming")
    assert isinstance(tile, StreamingTile) and tile.title_text.value == "Streaming"
    await driver.wait(text="Streaming", timeout=10)
    return page, tile


async def _tap(tester, control) -> None:
    """A finger tap on ``control`` (PyTester: a Switch toggles and sends ``change``, as Flutter does)."""
    await tester.tap(tester._register([control]))
    await asyncio.sleep(0.1)


def _chips(tile) -> list:
    from glossarion_mobile.ui.components.reason_chip import ReasonChip

    return [chip.reason for chip in tile.badges.controls if isinstance(chip, ReasonChip)]


def _desktop_labels(schema) -> dict:
    """key -> the desktop label the U8/U9 tiles showed (the schema's label)."""
    from glossarion_mobile.ui.settings.model import label_for

    return {key: label_for(schema.spec(key)) for key in WRITES}


async def _env_preview(app) -> dict:
    """Settings › Logs & diagnostics › Env preview › Build: the run env the next book job gets."""
    from glossarion_mobile.ui.screens.env_preview import EnvPreviewScreen

    app.navigate_to("settings.env_preview")
    await _until(lambda: isinstance(app.shell.top_screen, EnvPreviewScreen), 15, "the env preview screen")
    screen = app.shell.top_screen
    result = await screen.run_preview()
    assert result is not None and result.ok, getattr(result, "error", result)
    return {row.key: row.value for row in result.rows}


# ==========================================================================
# 1. Fresh install: one Streaming switch, ON; OFF writes only the changed keys
# ==========================================================================


@needs_runs  # the env preview builds the backend's run env
def test_fresh_phone_one_streaming_switch_on_by_default(phone):
    import settings_rules
    from glossarion_mobile.ui.settings.section_page import SectionPage

    async def scenario():
        tf, app, tester, driver = await _start_app({})
        try:
            store = app.config_store
            disk_before = _disk(phone)
            assert not set(WRITES) & set(disk_before), disk_before  # a fresh install stores none of them

            # ---- Settings › Response handling & retries: ONE switch, ON by default -----------------------
            page, tile = await _streaming_tile(app, driver)
            labels = _desktop_labels(page.ctx.schema)
            assert tile.switch.value is True and tile.value_text.value == "On · default", tile.value_text.value
            assert tile.editable and not tile.switch.disabled
            assert not any(str(c).startswith("Custom") for c in _chips(tile)), _chips(tile)
            assert tile.warning_text.value == DESKTOP_WARNING and tile.note_text.value == DESKTOP_NOTE
            titles = [page.tile(key).title_text.value for key in page.keys]
            assert titles.count("Streaming") == 1
            assert not set(labels.values()) & set(titles), (labels, titles)
            # the ⓘ lists the desktop settings it stands for, each ON by default
            body = tile.help_body()
            for key in WRITES:
                assert f"{labels[key]}" in body and "On (default)" in body.split(labels[key], 1)[1].split("\n")[0], body
            # every Settings page: no tile for a folded key, the switch once
            ctx = page.ctx
            shown = [str(spec.key) for section in ctx.schema.sections() for spec in ctx.schema.specs_for(section)]
            assert not set(WRITES) & set(shown) and shown.count("streaming") == 1
            # Thinking & reasoning (where U8/U9 had "Enable thoughts") has no Enable thoughts tile any more
            await driver.back()
            thinking = await _settings_section(app, driver, "thinking")
            assert "enable_thoughts" not in thinking.keys
            assert labels["enable_thoughts"] not in [thinking.tile(k).title_text.value for k in thinking.keys]
            await driver.back()

            # ---- search: one "Streaming" hit for every desktop name / env var -------------------------
            import flows

            await flows.open_settings(driver)
            for query in ("stream", "ENABLE_STREAMING", "Enable thoughts", "thinking logs", "forced-stream batch log"):
                await driver.enter(query, key="settings-search")
                await driver.wait(key="settings-hit-streaming", timeout=10)
                for key in WRITES:
                    assert await driver.count(key=f"settings-hit-{key}") == 0, (query, key)
                assert await driver.count(key="settings-hit-streaming") == 1, query
            # the hit opens the section on the switch (a desktop key's deep link lands there too)
            await driver.tap(key="settings-hit-streaming")
            await _until(lambda: isinstance(app.shell.top_screen, SectionPage)
                         and app.shell.top_screen.section_id == "other.response", 15, "the hit's section")
            assert app.shell.top_screen.focus_target == "streaming"
            # a deep link to a desktop key (e.g. from an older help text) lands on the switch
            for key in ("enable_streaming", "enable_thoughts"):
                previous = app.shell.top_screen
                await app.navigate(f"/settings/s/other.response#{key}")
                await _until(lambda: app.shell.top_screen is not previous
                             and isinstance(app.shell.top_screen, SectionPage), 15, f"the #{key} deep link")
                assert app.shell.top_screen.focus_target == "streaming", (key, app.shell.top_screen.focus_target)

            # ---- effective values: ON, while config.json holds none of the keys --------------------------
            for key in FOUR:
                assert store.effective(key) is True and not store.has(key), key
            assert store.effective("enable_thoughts") is True and not store.has("enable_thoughts")
            env = await _env_preview(app)
            assert {name: env.get(name) for name in STREAM_ENV} == {name: "1" for name in STREAM_ENV}, env
            store.flush()
            assert _disk(phone) == disk_before  # neither the default nor the preview wrote anything

            # ---- tap OFF: one write of exactly the keys that change ---------------------------------
            await driver.back()
            page, tile = await _streaming_tile(app, driver)
            writes = StoreWrites(store)
            await _tap(tester, tile.switch)
            calls = writes.take()
            assert calls == [("set_many", {key: False for key in WRITES})], calls
            # the desktop checkbox makes the same change (other_settings: stream thinking OFF unchecks thoughts)
            desktop = {key: True for key in WRITES}
            changed, _env = settings_rules.apply_change(desktop, "stream_thinking_logs", False)
            assert changed["enable_thoughts"] is False and store.get("enable_thoughts") is False
            assert tile.switch.value is False and tile.value_text.value == "Off" and tile.modified_dot.visible
            store.flush()
            disk = _disk(phone)
            assert "streaming" not in disk
            assert disk == {**disk_before, **{key: False for key in WRITES}}, disk
            env = await _env_preview(app)
            assert {name: env.get(name) for name in STREAM_ENV} == {name: "0" for name in STREAM_ENV}, env

            # ---- tap ON again: the same keys back, one write ---------------------------------------
            await driver.back()
            page, tile = await _streaming_tile(app, driver)
            assert tile.switch.value is False
            await _tap(tester, tile.switch)
            assert writes.take() == [("set_many", {key: True for key in WRITES})]
            assert tile.switch.value is True and tile.value_text.value == "On"
            store.flush()
            assert _disk(phone) == {**disk_before, **{key: True for key in WRITES}}
        finally:
            await _stop_app(tf, app)

    asyncio.run(scenario())


# ==========================================================================
# 2. OFF stops streaming everywhere: chat message, chat attachment, book jobs, Reader 🌐
# ==========================================================================


_KO = "비가 오래된 항구 도시에 계속 내렸고 야경꾼들은 밤새 순찰을 돌았다. "


def _write_epub(path: Path, title: str, chapters: int = 2) -> Path:
    """A small raw (Korean) EPUB with ``제N화`` chapter headings (the fake server tags its answers)."""
    from ebooklib import epub

    book = epub.EpubBook()
    book.set_identifier(f"glossarion-devfix-issue14-{path.stem}")
    book.set_title(title)
    book.set_language("ko")
    book.add_author("Glossarion tests")
    items = []
    for number in range(1, chapters + 1):
        chapter = epub.EpubHtml(title=f"제{number}화", file_name=f"chapter{number:04d}.xhtml", lang="ko")
        chapter.content = (f"<html><head><title>제{number}화</title></head><body><h1>제{number}화</h1>"
                           + "".join(f"<p>{_KO * 4}</p>" for _ in range(4)) + "</body></html>")
        book.add_item(chapter)
        items.append(chapter)
    book.toc = tuple(items)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav", *items]
    path.parent.mkdir(parents=True, exist_ok=True)
    epub.write_epub(str(path), book)
    return path


class RequestEnv:
    """The fake server's requests with the process environment each one met (the job sets os.environ for its
    run: ``job_runner.scoped_process_state``), keyed by request id."""

    def __init__(self, server) -> None:
        import run_env

        self.server = server
        self.keys = tuple(dict.fromkeys(tuple(run_env.FORCED_STREAM_ENV_KEYS) + STREAM_ENV + ("BATCH_TRANSLATION",)))
        self.env: dict = {}
        self._lock = threading.Lock()
        server.on_request.append(self._seen)

    def _seen(self, record) -> None:
        snapshot = {key: os.environ.get(key) for key in self.keys}
        with self._lock:
            self.env[record.id] = snapshot

    def since(self, mark: int) -> list:
        """[(record, env)] of the requests after ``mark`` that reached the model."""
        records = [r for r in self.server.records(since=mark) if r.kind != "unsupported"]
        with self._lock:
            return [(r, dict(self.env.get(r.id) or {})) for r in records]


def _configure(app, server) -> None:
    """flows.ui_config: what the device flows import from desktop (the fake endpoint, a dummy key, glossary
    off, no request spacing). It carries no streaming key: the mobile default applies."""
    import flows
    from glossarion_mobile.diagnostics.fake_llm_server import FAKE_MODEL

    store = app.config_store
    values = flows.ui_config(server.url, FAKE_MODEL)
    assert not set(WRITES) & set(values)
    store.set_many(values)
    store.flush()
    assert store.save_error is None, store.save_error


async def _chat_message(app, driver, text: str):
    """New chat, type a message, Send; wait until the chat run is Done."""
    from glossarion_mobile.ui.chat.send_state import SendState

    await driver.tap(tooltip="New chat")
    view = app.chat_view
    await _until(lambda: view.composer.send_state is SendState.IDLE_READY or view.composer.text == "", 10, "composer")
    view.composer.set_text(text)
    await _until(lambda: view.composer.send_state is SendState.IDLE_READY, 15,
                 lambda: f"the Send button (state {view.composer.send_state}, block {view.send_inputs().block})")
    await driver.tap(key="send-idle_ready", timeout=15)
    runs = app.chat_feature.runs
    await _until(lambda: view.cid and runs.run_for(view.cid) is not None, 30, "the chat run")
    run = runs.run_for(view.cid)
    await _until(lambda: not run.live, RUN_TIMEOUT, lambda: f"the chat run to end (state {run.state})")
    for thread in list(runs.finish_threads):
        await asyncio.to_thread(thread.join, 60)
    assert run.state == "done" and run.error is None, (run.state, run.error)
    # the reply reached the transcript (streamed or not): the last message is the model's answer
    chats = app.chat_feature.chats
    messages = chats.messages(view.cid)
    assert messages and messages[-1][0] == "assistant", messages[-3:]
    run.reply_text = chats.message_text(view.cid, len(messages) - 1)
    return run


async def _chat_attachment(app, driver, name: str):
    """New chat, ＋ › Files (the EPUB), Send, Start on the plan card; wait until the chat run is Done."""
    import flows

    await flows.go_home(driver)
    await driver.tap(tooltip="New chat")
    await driver.tap(tooltip=flows.ATTACH_TOOLTIP)
    await driver.pick_file(name, lambda: driver.tap(key="attach-files"))
    await driver.wait(contains=Path(name).stem, timeout=60)
    await driver.tap(key="send-idle_ready", timeout=60)
    await driver.wait(text="Ready to translate", timeout=60)
    await driver.tap(text="Start")
    if await driver.exists(text="Keep translations running", timeout=2):
        await driver.tap(text="Not now")
    runs = app.chat_feature.runs
    view = app.chat_view
    await _until(lambda: view.cid and runs.run_for(view.cid) is not None, 30, "the chat attachment run")
    run = runs.run_for(view.cid)
    await _until(lambda: not run.live, RUN_TIMEOUT, lambda: f"the chat attachment run to end (state {run.state})")
    for thread in list(runs.finish_threads):
        await asyncio.to_thread(thread.join, 60)
    assert run.state == "done" and run.error is None, (run.state, run.error)
    return run


async def _book_job(app, path: Path, kind: str = "translate"):
    """A Library job over a raw EPUB on the app's JobService: ``translate`` (``LibraryService.translate_spec``,
    the Translate sheet's job) or ``extract_glossary`` (Book page › Glossary › Extract, ``glossary_tab``)."""
    from glossarion_mobile.services.jobs import JobSpec

    service = app.job_service
    if kind == "translate":
        spec = app.library.translate_spec([{"name": path.stem}], sources=[str(path)])
    else:
        spec = JobSpec(kind=kind, title=path.stem, inputs=(str(path),), origin={"type": "library", "label": "Library"})
    job_id = service.submit(spec)

    def ended():
        snap = service.snapshot(job_id)
        return snap is not None and snap.is_terminal and not service.busy

    await _until(ended, RUN_TIMEOUT, lambda: f"book job {job_id} ({service.snapshot(job_id)})")
    snap = service.snapshot(job_id)
    assert snap.state.value == "DONE", (snap.state, snap.error)
    return snap


async def _reader_translate(app, tester, path: Path, index: int = 1):
    """Reader on the raw EPUB › 🌐 Translate (the chrome's ``reader-translate`` button: the live "Translate this
    chapter"); wait for the job."""
    from glossarion_mobile.ui.reader.reader_view import ReaderScreen

    previous = app.reader.active
    assert app.reader.open_book(path=str(path))
    await _until(lambda: isinstance(app.reader.active, ReaderScreen) and app.reader.active is not previous
                 and app.reader.active.state == "ready", 30, "the Reader")
    screen = app.reader.active
    if screen.index != index:
        await screen.render(index)
    await _until(lambda: screen.index == index, 10, "the chapter")
    button = screen.chrome.translate_button
    assert button.visible and button.on_click is not None and str(button.content) == "🌐 Translate", button.content
    await tester._dispatch(button, "click")  # the finger tap (Flutter sends ``click``)
    await _until(lambda: screen.live is not None, 15, "the live translation to start")
    live = screen.live
    status_at_start = live.panel.status.value
    service = app.job_service

    def ended():
        snap = service.snapshot(live.job_id)
        return snap is not None and snap.is_terminal and not service.busy

    await _until(ended, RUN_TIMEOUT, lambda: f"the Reader's job ({service.snapshot(live.job_id)})")
    snap = service.snapshot(live.job_id)
    assert snap.state.value == "DONE", (snap.state, snap.error)
    lines = list(service.log_buffer(live.job_id).snapshot()) if hasattr(service, "log_buffer") else []
    app.back()
    await _until(lambda: screen.disposed, 10, "the Reader to close")
    return types.SimpleNamespace(snap=snap, status=status_at_start, spec=snap.spec, lines=lines)


def _texts_of(lines) -> list:
    out = []
    for line in lines:
        out.append(str(getattr(line, "text", line)))
    return out


@needs_runs
def test_streaming_off_stops_streaming_in_chats_book_jobs_and_the_reader(phone):
    import run_env
    from glossarion_mobile.diagnostics.fake_llm_server import FakeLLMServer

    epubs = {}
    for label in ("on", "off"):
        for kind in ("chat", "book", "glossary", "reader"):
            name = f"Streaming {label} {kind}.epub"
            epubs[(label, kind)] = _write_epub(phone.picks / name, f"Streaming {label} {kind}")
    files = {path.name: path for (label, kind), path in epubs.items() if kind == "chat"}
    forced_only = tuple(k for k in run_env.FORCED_STREAM_ENV_KEYS if k not in STREAM_ENV + ("ENABLE_THOUGHTS",))

    async def scenario(server):
        tf, app, tester, driver = await _start_app(files)
        seen = RequestEnv(server)
        results: dict = {}
        try:
            _configure(app, server)
            for label in ("on", "off"):
                if label == "off":
                    # ---- the owner's tap: Settings › Response handling & retries › Streaming OFF ----------
                    page, tile = await _streaming_tile(app, driver)
                    assert tile.switch.value is True
                    await _tap(tester, tile.switch)
                    assert tile.switch.value is False and all(app.config_store.get(k) is False for k in WRITES)
                    app.config_store.flush()
                    import flows

                    await flows.go_home(driver)
                # chat message
                mark = server.mark()
                run = await _chat_message(app, driver, "안녕하세요. 오늘은 비가 옵니다.")
                results[(label, "chat")] = (seen.since(mark), run)
                # chat attachment (a book in the chat)
                mark = server.mark()
                run = await _chat_attachment(app, driver, epubs[(label, "chat")].name)
                results[(label, "attachment")] = (seen.since(mark), run)
                # Library book job
                mark = server.mark()
                snap = await _book_job(app, epubs[(label, "book")])
                results[(label, "book")] = (seen.since(mark), snap)
                # Library glossary extraction
                mark = server.mark()
                snap = await _book_job(app, epubs[(label, "glossary")], "extract_glossary")
                results[(label, "glossary")] = (seen.since(mark), snap)
                # Reader 🌐 Translate
                mark = server.mark()
                live = await _reader_translate(app, tester, epubs[(label, "reader")])
                results[(label, "reader")] = (seen.since(mark), live)
        finally:
            await _stop_app(tf, app)
        return results

    with FakeLLMServer() as server:
        results = asyncio.run(scenario(server))

    def flags(key):
        requests, _extra = results[key]
        assert requests, f"{key}: no request reached the model"
        return [r.stream for r, _env in requests]

    # every run reached the model with its chapters (2 per EPUB), so the flags below cover real requests
    minimum = {"chat": 1, "attachment": 2, "book": 2, "glossary": 1, "reader": 1}
    for (label, kind), (requests, _extra) in sorted(results.items()):
        wanted = [r for r, _env in requests if r.kind == ("glossary" if kind == "glossary" else "translation")]
        assert len(wanted) >= minimum[kind], (label, kind, [(r.kind, r.chapters) for r, _e in requests])
        print(f"[devfix-issue14] {label:3} {kind:10} requests={len(requests)} "
              f"stream={[r.stream for r, _e in requests]} kinds={sorted({r.kind for r, _e in requests})} "
              f"env={ {k: requests[0][1].get(k) for k in STREAM_ENV + ('ENABLE_THOUGHTS', 'LOG_STREAM_CHUNKS', 'BATCH_TRANSLATION')} }")

    # the chat still answers with Streaming off (the reply arrives whole instead of token by token)
    from glossarion_mobile.diagnostics.fake_llm_server import FAKE_MARKER

    for label in ("on", "off"):
        reply = results[(label, "chat")][1].reply_text
        assert FAKE_MARKER in reply, (label, reply[:200])

    # ---- ON (the mobile default, no streaming key stored): everything streams ------------------------
    for kind in ("chat", "attachment", "book", "glossary", "reader"):
        assert all(flags(("on", kind))), (kind, flags(("on", kind)))
    for kind in ("chat", "attachment"):  # the chat forces every streaming switch (desktop Direct Text parity)
        for record, env in results[("on", kind)][0]:
            assert all(env.get(k) == "1" for k in run_env.FORCED_STREAM_ENV_KEYS), (kind, env)
    for kind in ("book", "glossary"):  # book jobs follow the four toggles (absent = ON on mobile)
        for record, env in results[("on", kind)][0]:
            assert {k: env.get(k) for k in STREAM_ENV} == {k: "1" for k in STREAM_ENV}, (kind, env)
    on_reader = results[("on", "reader")][1]
    assert on_reader.spec.params.get("force_stream_all") is True
    assert "Streaming is off" not in str(on_reader.status), on_reader.status
    for record, env in results[("on", "reader")][0]:
        assert all(env.get(k) == "1" for k in run_env.FORCED_STREAM_ENV_KEYS), env

    # ---- OFF: no request streams, nothing is forced ----------------------------------------------
    for kind in ("chat", "attachment", "book", "glossary", "reader"):
        assert not any(flags(("off", kind))), (kind, flags(("off", kind)))
    for kind in ("chat", "attachment", "book", "glossary", "reader"):
        for record, env in results[("off", kind)][0]:
            assert {k: env.get(k) for k in STREAM_ENV} == {k: "0" for k in STREAM_ENV}, (kind, env)
            assert env.get("ENABLE_THOUGHTS") == "0", (kind, env)  # the desktop coupling: thoughts unchecked
            forced = {k: env.get(k) for k in forced_only if env.get(k) == "1"}
            assert not forced, (kind, forced)  # no forced streaming switch in the run's environment
    off_reader = results[("off", "reader")][1]
    assert "Streaming is off" in str(off_reader.status), off_reader.status
    assert any("Streaming is off" in line for line in _texts_of(off_reader.lines)), _texts_of(off_reader.lines)[-20:]


# ==========================================================================
# 3. Imported desktop config with mixed toggles: Custom, one tap normalises
# ==========================================================================


@needs_flet
def test_imported_mixed_desktop_config_shows_custom_and_one_tap_normalises(phone):
    import flows

    configs = {
        # the owner's desktop: streaming + stream thinking on, the batch log toggles off
        "desktop-mixed.json": {"output_language": "English", "enable_streaming": True, "stream_thinking_logs": True,
                               "allow_batch_stream_logs": False, "allow_authgpt_batch_stream_logs": False,
                               "enable_thoughts": True},
        # stream thinking already off, thoughts on: OFF must not run the no-op stream-thinking rule
        "desktop-streaming-only.json": {"enable_streaming": True, "stream_thinking_logs": False,
                                        "allow_batch_stream_logs": False, "allow_authgpt_batch_stream_logs": False,
                                        "enable_thoughts": True},
        # the desktop welcome wizard's write (stream thinking on, thoughts stored off), streaming responses off
        "desktop-wizard.json": {"enable_streaming": False, "stream_thinking_logs": True,
                                "allow_batch_stream_logs": False, "allow_authgpt_batch_stream_logs": True,
                                "enable_thoughts": False},
    }
    files = {}
    for name, values in configs.items():
        path = phone.picks / name
        path.write_text(json.dumps(values), encoding="utf-8")
        files[name] = path

    async def scenario():
        tf, app, tester, driver = await _start_app(files)
        try:
            store = app.config_store
            # ---- 1: the owner's mixed desktop config -------------------------------------------------
            await flows.import_desktop_config(driver, "desktop-mixed.json")
            assert [store.get(k) for k in FOUR] == [True, True, False, False]
            page, tile = await _streaming_tile(app, driver)
            assert tile.switch.value is True  # requests stream: the switch follows Enable streaming responses
            assert tile.value_text.value == "Custom · 2 of 4 on", tile.value_text.value
            assert "Custom · 2 of 4 on" in _chips(tile), _chips(tile)
            await driver.wait(contains="Custom · 2 of 4 on", timeout=10)
            writes = StoreWrites(store)
            await _tap(tester, tile.switch)  # one tap: all four off (+ thoughts unchecked, desktop coupling)
            assert writes.take() == [("set_many", {"enable_streaming": False, "stream_thinking_logs": False,
                                                   "enable_thoughts": False})]
            assert [store.get(k) for k in FOUR] == [False] * 4 and store.get("enable_thoughts") is False
            assert tile.switch.value is False and tile.value_text.value == "Off"
            assert not any(str(c).startswith("Custom") for c in _chips(tile)), _chips(tile)

            # ---- 2: Custom with stream thinking already off: OFF writes Enable streaming only ----------
            await flows.import_desktop_config(driver, "desktop-streaming-only.json")
            page, tile = await _streaming_tile(app, driver)
            assert tile.switch.value is True and tile.value_text.value == "Custom · 1 of 4 on"
            writes.take()  # the import's own write
            await _tap(tester, tile.switch)
            assert writes.take() == [("set_many", {"enable_streaming": False})]
            assert store.get("enable_thoughts") is True  # the no-op stream-thinking write never ran
            assert tile.value_text.value == "Off"

            # ---- 3: the desktop wizard's config: shown OFF (requests do not stream), ON re-locks thoughts ----
            await flows.import_desktop_config(driver, "desktop-wizard.json")
            page, tile = await _streaming_tile(app, driver)
            assert tile.switch.value is False and tile.value_text.value == "Custom · 2 of 4 on"
            writes.take()  # the import's own write
            await _tap(tester, tile.switch)
            assert writes.take() == [("set_many", {"enable_streaming": True, "allow_batch_stream_logs": True,
                                                   "enable_thoughts": True})]
            assert all(store.get(k) is True for k in WRITES) and tile.value_text.value == "On"
            store.flush()
            assert "streaming" not in _disk(phone)
        finally:
            await _stop_app(tf, app)

    asyncio.run(scenario())


# ==========================================================================
# 4. Desktop: Other Settings shows the same streaming texts
# ==========================================================================


_RENDER = textwrap.dedent(r"""
    import json, sys
    sys.path[:0] = [p for p in sys.argv[1:] if p]
    from PySide6.QtWidgets import QApplication, QCheckBox, QGridLayout, QLabel, QWidget
    app = QApplication.instance() or QApplication([])
    import other_settings

    class Owner(QWidget):
        def __init__(self):
            super().__init__()
            self.config = {}

    owner = Owner()
    other_settings.setup_other_settings_methods(owner)
    parent = QWidget()
    QGridLayout(parent)
    other_settings._create_response_handling_section(owner, parent)
    rows = [["QLabel", w.text(), w.styleSheet(), w.wordWrap()] for w in parent.findChildren(QLabel)]
    rows += [["QCheckBox", w.text(), w.styleSheet(), w.isChecked()] for w in parent.findChildren(QCheckBox)]
    print("ROWS=" + json.dumps(rows, ensure_ascii=True))
""")


def _render_response_section(tmp_path: Path, label: str, first_on_path: str = "") -> list:
    env = dict(os.environ)
    for name in ("HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA", "GLOSSARION_LIBRARY_DIR", "OUTPUT_DIRECTORY",
                 "GLOSSARION_DATA_DIR", "XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME"):
        folder = tmp_path / label / name.lower()
        folder.mkdir(parents=True, exist_ok=True)
        env[name] = str(folder)
    env.update(QT_QPA_PLATFORM="offscreen", GLOSSARION_HTTP_LOG="0", PYTHONIOENCODING="utf-8")
    # the desktop dialog renders in a desktop process: mobile runtime switches other host tests leave in this
    # process (test_models_keys.py sets GLOSSARION_HEADLESS_KEY_MANAGER=1 at import; with it the desktop's
    # multi_api_key_manager import ends the render without a word) must not reach it
    for name in ("GLOSSARION_HEADLESS_KEY_MANAGER", "GLOSSARION_MOBILE", "GLOSSARION_NO_PROCESSES"):
        env.pop(name, None)
    script = tmp_path / label / "render_response_section.py"
    script.write_text(_RENDER, encoding="utf-8")
    result = subprocess.run([sys.executable, str(script), first_on_path, str(SRC_DIR)], cwd=str(tmp_path / label),
                            env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)
    line = next((ln for ln in result.stdout.splitlines() if ln.startswith("ROWS=")), None)
    assert line is not None, (label, result.returncode, result.stdout[-2000:], result.stderr[-3000:])
    return json.loads(line[len("ROWS="):])


def _pre_move_other_settings() -> str:
    """other_settings.py of the commit before the move (None without git / history: a shallow CI clone)."""
    try:
        result = subprocess.run(["git", "-C", str(REPO_DIR), "show", f"{PRE_MOVE_SHA}:src/other_settings.py"],
                                capture_output=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0 or not result.stdout:
        return None
    return result.stdout.decode("utf-8-sig")


def test_desktop_other_settings_streaming_texts_identical(tmp_path):
    import settings_rules

    # the shared constants are the desktop texts, and other_settings shows them through the constants only
    assert settings_rules.STREAMING_TRUNCATION_WARNING == DESKTOP_WARNING
    assert settings_rules.FORCED_STREAM_NOTE == DESKTOP_NOTE
    source = (SRC_DIR / "other_settings.py").read_text(encoding="utf-8-sig")
    tree = ast.parse(source)
    assert any(isinstance(node, ast.Import) and any(a.name == "settings_rules" for a in node.names)
               for node in tree.body), "other_settings.py does not import settings_rules at module level"
    builder = next(node for node in tree.body
                   if isinstance(node, ast.FunctionDef) and node.name == "_create_response_handling_section")
    label_args = [ast.unparse(call.args[0]) for call in ast.walk(builder)
                  if isinstance(call, ast.Call) and getattr(call.func, "id", "") == "QLabel" and call.args]
    assert "settings_rules.STREAMING_TRUNCATION_WARNING" in label_args
    assert "settings_rules.FORCED_STREAM_NOTE" in label_args
    # the desktop's four checkboxes and their keys are still there (desktop keeps the group)
    for attr in ("enable_streaming_checkbox", "stream_thinking_logs_checkbox", "allow_batch_stream_logs_checkbox",
                 "allow_authgpt_batch_stream_logs_checkbox"):
        assert f"self.{attr} = " in ast.unparse(builder), attr

    # the mobile-only divergences are recorded (owner decision)
    discrepancies = REPO_DIR / "tests" / "parity" / "DISCREPANCIES.md"
    if discrepancies.is_file():
        text = discrepancies.read_text(encoding="utf-8")
        assert "Streaming: one switch on mobile" in text
        for needle in ("MOBILE_STREAMING_DEFAULT", "skip_forced_streaming", "force_stream_all", "Custom"):
            assert needle in text.split("Streaming: one switch on mobile", 1)[1][:6000], needle

    if not _has("PySide6"):
        pytest.skip("PySide6 is not installed: the offscreen render is skipped")
    rows = _render_response_section(tmp_path, "head")
    texts = [row[1] for row in rows]
    assert DESKTOP_WARNING in texts and DESKTOP_NOTE in texts
    # the desktop keeps its four streaming checkboxes, OFF for a config without the keys (desktop default)
    boxes = {row[1]: row[3] for row in rows if row[0] == "QCheckBox" and "stream" in row[1].lower()}
    assert len(boxes) == 4 and not any(boxes.values()), boxes
    old = _pre_move_other_settings()
    if old is None:
        pytest.skip(f"git history without {PRE_MOVE_SHA}: the before/after render comparison is skipped")
    assert "may result in silent truncation" in old  # really the pre-move file
    old_dir = tmp_path / "pre_move_module"
    old_dir.mkdir()
    (old_dir / "other_settings.py").write_text(old, encoding="utf-8")
    before = _render_response_section(tmp_path, "pre_move", str(old_dir))
    assert before == rows, "Other Settings › Response Handling renders differently than before the move"
