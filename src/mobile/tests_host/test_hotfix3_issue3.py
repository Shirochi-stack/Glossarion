"""Acceptance test, hotfix 3 owner issue 3: "We don't need force multipass off in the mobile version, just have it
toggled off by default".

What the owner saw on the phone: Chat settings › Run behaviour had the desktop Direct Text dialog's "Force Multipass
off" switch, ON on a fresh install, and every chat run (a message, a book in the chat) exported MULTIPASS_MODE=0
whatever Settings › Translation defaults › Multipass said.

Wanted (mobile only; the desktop dialog keeps its checkbox and its defaults): no "Force Multipass off" on mobile;
Multipass itself OFF by default (an explicitly saved or imported value is respected); chat runs simply follow the
Multipass setting, never forced off behind the user's back, also in a chat an older build stored
``force_multipass_off: True`` for.

1. ``test_owner_phone_chat_runs_follow_the_multipass_setting``: the REAL app (``main.main`` on the in-memory Flet
   session as Android, 412 dp, driven through ``tests/ui_driver.UiDriver`` + ``tests/host_tester.PyTester``) on a
   fresh isolated config with a chat an older build left behind (its sidecar says ``force_multipass_off: True``):
   Chat settings (This chat and All chats) has no Force Multipass off switch; Settings shows no Force Multipass off
   tile or search hit; Settings › Translation defaults › Multipass shows "Off · default"; a chat message sends its
   requests with MULTIPASS_MODE=0 and no refinement pass; the owner's tap ON makes the next chat message multipass
   (MULTIPASS_MODE=1, the refinement pass runs); the older build's chat follows the setting too (a message and a
   book in that chat: the refinement pass re-sends both chapters); tap OFF and the next run is single-pass again;
   an imported desktop config (Multipass ON next to the dialog's Force Multipass off ON) is respected.
   Only the model is the offline fake OpenAI server on 127.0.0.1 (``diagnostics.fake_llm_server``).
2. ``test_desktop_defaults_unchanged``: the shared desktop contract keeps its defaults (``multipass_mode`` off,
   ``DirectTextRunOptions.force_multipass_off`` on, the dialog's checkbox from ``direct_text_force_multipass_off`` /
   ``direct_text_force_simple_mode`` defaulting to on) and its behaviour (a desktop Direct Text run with the default
   options skips Multipass even when it is on; unchecked it follows it; a main-window run follows it).
3. ``test_desktop_files_identical_to_the_batch_base``: the desktop modules behind that contract are byte-identical
   (line endings aside) to the batch base while the batch is the tip (skipped without the git history or later).

Note: with Multipass ON a plain-text chat message runs the refinement pass of the shared pipeline
(``TransateKRtoEN``, not mobile code), which reports "translated output not found" for the .txt run's one chapter
and skips it, so the test asserts the pass ran (the job log), not a refinement request; a book in the chat is
refined chapter by chapter.

Real data is never touched: ``app_env`` points the data / output / Library / HOME / CONFIG_FILE paths at pytest tmp
dirs; USERPROFILE / APPDATA / LOCALAPPDATA are redirected too, GLOSSARION_HTTP_LOG=0, and src/config.json keeps its
md5.

Run from src/mobile with the mobile venv::

    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_hotfix3_issue3.py
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import threading
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
needs_runs = pytest.mark.skipif(not all(_has(m) for m in _RUN_NEEDED),
                                reason=f"needs {', '.join(_RUN_NEEDED)} (the mobile project venv)")
needs_backend = pytest.mark.skipif(not (SRC_DIR / "headless_owner.py").is_file(), reason="src/ backend not present")

#: The batch base (hotfix 2 HEAD): the desktop files must equal it.
BASE_SHA = "f4753489"
#: Both config keys the desktop dialog's "Force Multipass off" checkbox reads (translator_gui._InputOutputDialog).
DESKTOP_FORCE_KEYS = {"direct_text_force_multipass_off": True, "direct_text_force_simple_mode": True}
#: The older build's chat (desktop history id) and the message it holds.
OLD_CHAT_ID = 7
OLD_CHAT_TITLE = "Older build chat"
OLD_SERIES_ID = "saga_old"
KO_TEXT = "안녕하세요. 오늘은 비가 옵니다."
RUN_TIMEOUT = 120.0
_REPO_CONFIG = SRC_DIR / "config.json"


def _md5(path: Path):
    return hashlib.md5(path.read_bytes()).hexdigest() if path.is_file() else None


def _load(alias: str, file_name: str):
    spec = importlib.util.spec_from_file_location(alias, Path(__file__).with_name(file_name))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if all(_has(m) for m in _UI_NEEDED):
    # the devfix #14 acceptance harness: the real app on the in-memory Flet session, the isolated phone storage,
    # the Settings / chat helpers (reused, not copied)
    _DF = _load("_glossarion_df14_hotfix3_issue3", "test_devfix_issue14.py")
    storage = _DF.storage  # noqa: F811  (pytest fixtures)
    app_env = _DF.app_env
    phone = _DF.phone
else:  # pragma: no cover - a run without Flet runs the desktop test only
    _DF = None


@pytest.fixture(autouse=True)
def _repo_config_untouched():
    before = _md5(_REPO_CONFIG)
    yield
    assert _md5(_REPO_CONFIG) == before, "src/config.json changed"


# ==========================================================================
# Helpers
# ==========================================================================


class MultipassSeen:
    """MULTIPASS_MODE as each fake-model request met it (a job sets os.environ for its run)."""

    def __init__(self, server) -> None:
        self.server = server
        self.env: dict = {}
        self._lock = threading.Lock()
        server.on_request.append(self._seen)

    def _seen(self, record) -> None:
        value = os.environ.get("MULTIPASS_MODE")
        with self._lock:
            self.env[record.id] = value

    def since(self, mark: int) -> list:
        records = [r for r in self.server.records(since=mark) if r.kind != "unsupported"]
        with self._lock:
            return [(r, self.env.get(r.id)) for r in records]


def _refinements(seen: list) -> list:
    """The multipass refinement pass's requests: the model's own (fake-marked) translation sent back to it."""
    from glossarion_mobile.diagnostics.fake_llm_server import FAKE_MARKER

    return [r for r, _env in seen if FAKE_MARKER in (r.preview or "")]


def _describe(seen: list) -> list:
    return [(r.kind, list(r.chapters), env, (r.preview or "")[:60]) for r, env in seen]


def _seed_older_build_chat(phone) -> None:
    """direct_text_chats.json (desktop v2) with one finished chat, and what an older build wrote for it: "Force
    Multipass off" turned ON in that chat's Chat settings (sidecar ``overrides.force_multipass_off``) and in the
    Series defaults of the series the chat is in (``mobile_series.json``)."""
    history = phone.data / "direct_text_chats.json"
    history.write_text(json.dumps({"version": 2, "current_chat_id": OLD_CHAT_ID, "sessions": [{
        "id": OLD_CHAT_ID, "title": OLD_CHAT_TITLE,
        "messages": [["user", "안녕"], ["assistant", "Hello", "", "Completed", "", ""]],
        "draft": "", "attachment": None, "output_folder": "", "output_folder_name": "",
        "next_output_index": 1, "expanded": []}]}, ensure_ascii=False), encoding="utf-8")
    sidecar = phone.data / "direct_text_chats.mobile.json"
    sidecar.write_text(json.dumps({"version": 1, "chats": {str(OLD_CHAT_ID): {
        "overrides": {"force_multipass_off": True}, "series_id": OLD_SERIES_ID}}, "scratch": []}), encoding="utf-8")
    (phone.data / "mobile_series.json").write_text(json.dumps({"version": 1, "series": [{
        "id": OLD_SERIES_ID, "name": "Older build saga", "defaults": {"force_multipass_off": True}}]}),
        encoding="utf-8")


def _walk_keys_and_texts(control, keys: list, texts: list, seen=None) -> None:
    seen = set() if seen is None else seen
    if control is None or isinstance(control, (str, int, float, bool)) or id(control) in seen:
        return
    seen.add(id(control))
    key = getattr(control, "key", None)
    key = getattr(key, "value", key)
    if isinstance(key, str) and key:
        keys.append(key)
    for name in ("value", "label", "title", "text", "tooltip", "hint_text", "helper_text"):
        value = getattr(control, name, None)
        if isinstance(value, str) and value:
            texts.append(value)
    for name in ("controls", "content", "title", "subtitle", "leading", "trailing", "label", "actions", "options"):
        child = getattr(control, name, None)
        for item in (child if isinstance(child, (list, tuple)) else [child]):
            _walk_keys_and_texts(item, keys, texts, seen)


def _sheet_keys_and_texts(sheet) -> tuple:
    keys: list = []
    texts: list = []
    for root in (getattr(sheet, "body", None), getattr(sheet, "dialog", None)):
        _walk_keys_and_texts(root, keys, texts)
    return keys, texts


async def _open_chat_settings(app, driver):
    """The composer's ⚙ options pill (Chat settings), as the owner opens it."""
    view = app.chat_view
    previous = getattr(view, "settings_sheet", None)
    await driver.tap(key="pill-options", timeout=15)
    await _DF._until(lambda: getattr(view, "settings_sheet", None) is not None
                     and view.settings_sheet is not previous, 10, "Chat settings")
    return view.settings_sheet


def _assert_no_force_switch(sheet, where: str) -> None:
    for scope in ("chat", "global"):
        sheet.scope = scope
        sheet.rebuild()
        keys, texts = _sheet_keys_and_texts(sheet)
        assert "setting-disable_thinking" in keys, (where, scope, keys)  # the Run behaviour group is there
        assert "setting-force_multipass_off" not in keys, (where, scope, keys)
        assert not [t for t in texts if "multipass" in t.lower()], (where, scope, [t for t in texts if "ultipass" in t])


async def _search_hits(app, driver, query: str) -> dict:
    """Settings home › search: ``{key: "title — subtitle"}`` of the setting hits for ``query`` the owner sees."""
    import flows

    await flows.open_settings(driver)
    await driver.enter(query, key="settings-search")
    await driver.wait_any({"key": "settings-hit-multipass_mode"}, {"contains": "No settings match"},
                          {"contains": "Force"}, timeout=15)
    await driver.pump(200)
    hits = {}
    rows = getattr(getattr(app.shell.top_screen, "search", None), "result_rows", {}) or {}
    for key in ("direct_text_force_multipass_off", "direct_text_force_simple_mode", "multipass_mode"):
        if await driver.count(key=f"settings-hit-{key}"):
            row = rows.get(key)
            hits[key] = (f"{getattr(row.title, 'value', '')} — {getattr(row.subtitle, 'value', '')}"
                         if row is not None else "")
    await driver.enter("", key="settings-search")
    return hits


async def _wait_run(app, view, previous_run=None):
    runs = app.chat_feature.runs
    await _DF._until(lambda: view.cid and runs.run_for(view.cid) is not None
                     and runs.run_for(view.cid) is not previous_run, 30, "the chat run")
    run = runs.run_for(view.cid)
    await _DF._until(lambda: not run.live, RUN_TIMEOUT, lambda: f"the chat run to end (state {run.state})")
    for thread in list(runs.finish_threads):
        await asyncio.to_thread(thread.join, 60)
    assert run.state == "done" and run.error is None, (run.state, run.error)
    return run


async def _message_in_open_chat(app, driver, text: str):
    """Type a message in the chat on screen and Send; wait until its run is Done."""
    from glossarion_mobile.ui.chat.send_state import SendState

    view = app.chat_view
    runs = app.chat_feature.runs
    before = runs.run_for(view.cid) if view.cid else None
    view.composer.set_text(text)
    await _DF._until(lambda: view.composer.send_state is SendState.IDLE_READY, 15,
                     lambda: f"the Send button (state {view.composer.send_state}, block {view.send_inputs().block})")
    await driver.tap(key="send-idle_ready", timeout=15)
    return await _wait_run(app, view, before)


async def _book_in_open_chat(app, driver, name: str):
    """＋ › Files (the EPUB) in the chat on screen, Send, Start on the plan card; wait until the run is Done."""
    import flows

    view = app.chat_view
    runs = app.chat_feature.runs
    before = runs.run_for(view.cid) if view.cid else None
    await driver.tap(tooltip=flows.ATTACH_TOOLTIP)
    await driver.pick_file(name, lambda: driver.tap(key="attach-files"))
    await driver.wait(contains=Path(name).stem, timeout=60)
    await driver.tap(key="send-idle_ready", timeout=60)
    await driver.wait(text="Ready to translate", timeout=60)
    await driver.tap(text="Start")
    if await driver.exists(text="Keep translations running", timeout=2):
        await driver.tap(text="Not now")
    return await _wait_run(app, view, before)


def _job(app, run):
    snap = app.job_service.snapshot(run.job_id)
    assert snap is not None, run.job_id
    buffer = app.job_service.log_buffer(run.job_id)
    lines = [str(getattr(line, "text", line)) for line in (buffer.snapshot() if buffer is not None else [])]
    options = dict((snap.spec.params or {}).get("options") or {})
    start = next((i for i, line in enumerate(lines) if "MULTIPASS REFINEMENT" in line), None)
    return types.SimpleNamespace(snap=snap, lines=lines, options=options, refined=start is not None,
                                 refinement_log=lines[start:start + 40] if start is not None else [])


async def _open_old_chat(app):
    app._open_chat(str(OLD_CHAT_ID))
    view = app.chat_view
    await _DF._until(lambda: str(view.cid) == str(OLD_CHAT_ID), 15, "the older build's chat")
    return view


async def _multipass_tile(app, driver):
    page = await _DF._settings_section(app, driver, "translation_defaults")
    return page.tile("multipass_mode")


# ==========================================================================
# 1. The owner's phone: no Force Multipass off, Multipass off by default, chat runs follow it
# ==========================================================================


@needs_runs
@needs_backend
def test_owner_phone_chat_runs_follow_the_multipass_setting(phone):
    import flows
    from glossarion_mobile.diagnostics.fake_llm_server import FAKE_MODEL, FakeLLMServer

    _seed_older_build_chat(phone)
    book = _DF._write_epub(phone.picks / "Older chat book.epub", "Older chat book")
    files = {book.name: book}

    async def scenario(server):
        desktop = phone.picks / "desktop-config.json"
        # what Import from desktop brings from a desktop that had Multipass ON (the dialog's checkbox on, its default)
        desktop.write_text(json.dumps({**flows.ui_config(server.url, FAKE_MODEL), "multipass_mode": True,
                                       **DESKTOP_FORCE_KEYS}), encoding="utf-8")
        files[desktop.name] = desktop
        tf, app, tester, driver = await _DF._start_app(files)
        seen = MultipassSeen(server)
        problems: list = []
        try:
            _DF._configure(app, server)
            store = app.config_store
            disk = _DF._disk(phone)
            assert "multipass_mode" not in disk and not set(DESKTOP_FORCE_KEYS) & set(disk), disk
            assert app.chat_feature.chats.messages(str(OLD_CHAT_ID)), "the older build's chat did not load"

            # ---- fresh config: Chat settings (This chat / All chats) has no Force Multipass off ------------
            await flows.go_home(driver)
            await driver.tap(tooltip="New chat")
            sheet = await _open_chat_settings(app, driver)
            _assert_no_force_switch(sheet, "new chat")
            sheet.close()
            assert app.chat_view.option_pills() == []

            # ---- Settings: no Force Multipass off tile on a listed page, no search hit for it ------------------
            await flows.open_settings(driver)
            home = app.shell.top_screen
            listed = {key for group, sections in home.ctx.schema.groups() for section in sections
                      for key in home._listed_keys(section)}
            assert "multipass_mode" in listed and not set(DESKTOP_FORCE_KEYS) & listed, \
                sorted(set(DESKTOP_FORCE_KEYS) & listed)
            for query in ("Force Multipass", "force_multipass_off", "Force simple mode"):
                hits = await _search_hits(app, driver, query)
                stale = {key: shown for key, shown in hits.items() if key in DESKTOP_FORCE_KEYS}
                if stale:
                    problems.append(f"Settings search {query!r} still shows the retired switch: {stale}")
            assert "multipass_mode" in await _search_hits(app, driver, "Multipass")  # the real setting is found

            # ---- Settings › Translation defaults › Multipass: OFF, the default (nothing stored) -------------
            tile = await _multipass_tile(app, driver)
            assert tile.switch.value is False, tile.switch.value
            assert tile.value_text.value == "Off · default", tile.value_text.value
            assert store.effective("multipass_mode") is False and not store.has("multipass_mode")

            # ---- a chat message on the fresh config: MULTIPASS_MODE 0, one translation pass --------------------
            await flows.go_home(driver)
            mark = server.mark()
            run = await _DF._chat_message(app, driver, KO_TEXT)
            off = seen.since(mark)
            job = _job(app, run)
            assert off and {env for _r, env in off} == {"0"}, _describe(off)
            assert _refinements(off) == [] and not job.refined, (_describe(off), job.lines[-20:])
            assert job.options.get("force_multipass_off") is False, job.options

            # ---- the owner turns Multipass ON: the next chat message is multipass -----------------------------
            tile = await _multipass_tile(app, driver)
            await _DF._tap(tester, tile.switch)
            assert tile.switch.value is True and store.get("multipass_mode") is True
            store.flush()
            assert _DF._disk(phone).get("multipass_mode") is True
            assert not set(DESKTOP_FORCE_KEYS) & set(_DF._disk(phone))
            await flows.go_home(driver)
            mark = server.mark()
            run = await _DF._chat_message(app, driver, KO_TEXT)
            on = seen.since(mark)
            job = _job(app, run)
            assert on and {env for _r, env in on} == {"1"}, _describe(on)
            # the run's Multipass refinement pass ran after the translation (the shared pipeline's pass; for a
            # plain-text message it reports the chapter's output not found and skips it: the book below is refined)
            assert job.refined, (_describe(on), job.lines[-30:])
            assert job.options.get("force_multipass_off") is False, job.options

            # ---- the older build's series: its Defaults list no Force Multipass off, nor does the sheet --------
            await flows.go_home(driver)
            series = app.series
            assert series.store.has(OLD_SERIES_ID) and series.store.defaults(OLD_SERIES_ID) == {}, \
                series.store.defaults(OLD_SERIES_ID)
            series.open_defaults_sheet(OLD_SERIES_ID)
            await _DF._until(lambda: getattr(series, "sheet", None) is not None, 10, "the Series defaults sheet")
            keys, texts = _sheet_keys_and_texts(series.sheet)
            assert "setting-disable_thinking" in keys and "setting-force_multipass_off" not in keys, keys
            assert not [t for t in texts if "multipass" in t.lower()], [t for t in texts if "ultipass" in t]
            series.sheet.close()

            # ---- the older build's chat (sidecar force_multipass_off: True) no longer forces it -------------
            view = await _open_old_chat(app)
            assert app.chat_feature.chats.own_overrides(str(OLD_CHAT_ID)) == {}
            assert app.chat_feature.chats.overrides(str(OLD_CHAT_ID)) == {}  # nor through the series layer
            assert view.option_pills() == [], view.option_pills()
            sheet = await _open_chat_settings(app, driver)
            assert str(sheet.cid) == str(OLD_CHAT_ID)
            sheet.scope = "chat"
            sheet.rebuild()
            assert not any(sheet.is_overridden(f.name) for f in dataclasses.fields(type(sheet.effective()))), \
                "the older build's override still counts as a chat override"
            _assert_no_force_switch(sheet, "older build's chat")
            sheet.close()
            mark = server.mark()
            run = await _message_in_open_chat(app, driver, KO_TEXT)
            old_text = seen.since(mark)
            job = _job(app, run)
            assert str(view.cid) == str(OLD_CHAT_ID)
            assert old_text and {env for _r, env in old_text} == {"1"}, _describe(old_text)
            assert job.refined and job.options.get("force_multipass_off") is False, (job.options, job.refinement_log)
            # a book in that chat (the owner's EPUB in a chat): translated, then refined chapter by chapter
            mark = server.mark()
            run = await _book_in_open_chat(app, driver, book.name)
            old_book = seen.since(mark)
            job = _job(app, run)
            assert str(view.cid) == str(OLD_CHAT_ID)
            assert old_book and {env for _r, env in old_book} == {"1"}, _describe(old_book)
            # the chapter requests (one chapter each; the chapter-titles request names both)
            translated = sorted(tuple(r.chapters) for r, _e in old_book
                                if len(r.chapters) == 1 and r not in _refinements(old_book))
            assert job.refined and len(_refinements(old_book)) >= 2, (_describe(old_book), job.refinement_log)
            assert translated == [(1,), (2,)], _describe(old_book)

            # ---- Multipass OFF again: the next chat run is single-pass (follows the setting both ways) --------
            tile = await _multipass_tile(app, driver)
            await _DF._tap(tester, tile.switch)
            assert tile.switch.value is False and store.get("multipass_mode") is False
            await flows.go_home(driver)
            mark = server.mark()
            run = await _DF._chat_message(app, driver, KO_TEXT)
            off_again = seen.since(mark)
            assert {env for _r, env in off_again} == {"0"} and not _job(app, run).refined, _describe(off_again)

            # ---- an imported desktop config: Multipass ON is respected, its Force Multipass off is not ---------
            await flows.import_desktop_config(driver, desktop.name)
            assert store.get("multipass_mode") is True
            tile = await _multipass_tile(app, driver)
            assert tile.switch.value is True, tile.value_text.value
            await flows.go_home(driver)
            sheet = await _open_chat_settings(app, driver)
            _assert_no_force_switch(sheet, "after the desktop import")
            sheet.close()
            mark = server.mark()
            run = await _DF._chat_message(app, driver, KO_TEXT)
            imported = seen.since(mark)
            assert imported and {env for _r, env in imported} == {"1"}, _describe(imported)
            assert _job(app, run).refined
        finally:
            await _DF._stop_app(tf, app)
        assert not problems, "\n".join(problems)

    with FakeLLMServer() as server:
        asyncio.run(scenario(server))


# ==========================================================================
# 2. The desktop keeps its Force Multipass off checkbox, its defaults and its behaviour
# ==========================================================================


def _git_show(rel: str):
    try:
        out = subprocess.run(["git", "-C", str(REPO_DIR), "show", f"{BASE_SHA}:{rel}"], capture_output=True,
                             timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


def _normalized(data: bytes) -> bytes:
    return data.replace(b"\r\n", b"\n")


@needs_backend
def test_desktop_defaults_unchanged(tmp_path, monkeypatch):
    for name, sub in (("HOME", "home"), ("USERPROFILE", "home"), ("APPDATA", "appdata"),
                      ("LOCALAPPDATA", "localappdata"), ("GLOSSARION_LIBRARY_DIR", "lib"),
                      ("OUTPUT_DIRECTORY", "out"), ("GLOSSARION_DATA_DIR", "data")):
        folder = tmp_path / "_iso" / sub
        folder.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv(name, str(folder))
    monkeypatch.setenv("GLOSSARION_HTTP_LOG", "0")
    pytest.importorskip("run_env")
    glossary_paths = pytest.importorskip("glossary_paths")
    monkeypatch.setattr(glossary_paths, "migrate_all_legacy_glossary_files", lambda *a, **k: None)
    import headless_owner
    import settings_schema

    # ---- the desktop defaults ------------------------------------------------------------------------------
    assert settings_schema.effective_default("multipass_mode") is False
    defaults = {f.name: f.default for f in dataclasses.fields(headless_owner.DirectTextRunOptions)}
    assert defaults["force_multipass_off"] is True
    # the dialog's checkbox: config 'direct_text_force_multipass_off', default from the legacy simple-mode key (on)
    gui = (SRC_DIR / "translator_gui.py").read_text(encoding="utf-8")
    assert "translator.config.get('direct_text_force_simple_mode', True)" in gui
    assert "'direct_text_force_multipass_off', legacy_simple_mode" in gui
    assert "self.force_multipass_off_checkbox.isChecked()" in gui

    # ---- the desktop behaviour on the shared run env (the owner the dialog drives) -------------------------
    source = tmp_path / "direct_text.txt"
    source.write_text("hello", encoding="utf-8")
    before = dict(os.environ)

    def multipass_env(config: dict, direct_text: bool = True, **options) -> tuple:
        os.environ.pop("MULTIPASS_MODE", None)
        owner = headless_owner.HeadlessOwner(dict(config))
        if direct_text:
            headless_owner.DirectTextRunOptions(selected_files=[str(source)], **options).apply_to(owner)
        enabled, _mode = owner._export_multipass_runtime_env()
        return bool(enabled), os.environ.get("MULTIPASS_MODE")

    try:
        # a desktop Direct Text run with the dialog's defaults: Multipass skipped although it is on
        assert multipass_env({"multipass_mode": True}) == (False, "0")
        # the checkbox unchecked: the run follows Multipass
        assert multipass_env({"multipass_mode": True}, force_multipass_off=False) == (True, "1")
        assert multipass_env({}, force_multipass_off=False) == (False, "0")  # the default (off)
        # a main-window run: follows Multipass
        assert multipass_env({"multipass_mode": True}, direct_text=False) == (True, "1")
    finally:
        os.environ.clear()
        os.environ.update(before)


#: The desktop modules behind the dialog's checkbox and the Multipass run env.
DESKTOP_FILES = ("translator_gui.py", "run_env.py", "headless_owner.py", "owner_state.py", "settings_schema.py",
                 "settings_schema_data.py", "translation_pipeline.py", "text_jobs.py", "direct_text_store.py",
                 "TransateKRtoEN.py")


def _commits_since_base():
    try:
        out = subprocess.run(["git", "-C", str(REPO_DIR), "rev-list", "--count", f"{BASE_SHA}..HEAD"],
                             capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    return int(out.stdout.strip()) if out.returncode == 0 and out.stdout.strip().isdigit() else None


def test_desktop_files_identical_to_the_batch_base():
    """The hotfix 3 batch changed no desktop module issue 3 touches: byte-identical (line endings aside) to the
    batch base, while the batch is still the tip (later commits may change the desktop legitimately)."""
    since = _commits_since_base()
    if since is None:
        pytest.skip(f"git history with {BASE_SHA} not available")
    if since > 2:  # the batch commit (+ its merge); after that the desktop moves on
        pytest.skip(f"{since} commits since {BASE_SHA}: the batch is no longer the tip")
    for name in DESKTOP_FILES:
        base = _git_show(f"src/{name}")
        assert base is not None, f"src/{name} missing at {BASE_SHA}"
        assert _normalized((SRC_DIR / name).read_bytes()) == _normalized(base), f"src/{name} changed since {BASE_SHA}"
