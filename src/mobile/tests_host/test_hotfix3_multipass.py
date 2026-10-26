"""Hotfix 3, owner issue 3: "We don't need force multipass off in the mobile version, just have it toggled off by
default".

Before: Chat settings › Run behaviour (and the Series defaults card) showed the desktop Direct Text dialog's
"Force Multipass off" switch, ON on a fresh install (``direct_text_force_multipass_off``, default from the legacy
``direct_text_force_simple_mode``). Every chat run passed it to ``DirectTextRunOptions``, so the shared run env
(``run_env._export_multipass_runtime_env`` / ``_apply_direct_text_runtime_environment``) exported MULTIPASS_MODE=0
whatever Settings › Translation defaults › Multipass (or the Plan card's Multipass tile) said.

After (mobile only, the desktop dialog keeps its checkbox and its default):

* no "Force Multipass off" anywhere on mobile: Chat settings (This chat, All chats, Series defaults), the Series
  page's Defaults card and the composer's option pills;
* every chat run passes ``force_multipass_off=False`` (``direct_text_rules.FORCE_MULTIPASS_OFF``), so it follows the
  Multipass setting, which is OFF by default (``multipass_mode`` absent = off, the desktop default; Settings shows
  "Off · default" and config.json gets no key); a saved or imported Multipass ON is respected;
* a "Force Multipass off" stored by an older build (imported desktop key, chat sidecar override, Series default,
  the options of a run it sent) is never read, so it cannot keep forcing Multipass off behind the user's back.

``test_owner_flow_chat_runs_follow_the_multipass_setting`` proves it end to end on the REAL app objects (the
in-memory Flet session of test_ui_flows, the real ChatFeature / ChatRuns / JobService / HeadlessOwner and the shared
desktop pipeline; only the model is the offline fake OpenAI server on 127.0.0.1): what the requests of a chat run
met in the process environment, before and after the owner's tap on Settings › Translation defaults › Multipass.

Real data is never touched: HOME / USERPROFILE / APPDATA / LOCALAPPDATA / GLOSSARION_LIBRARY_DIR /
OUTPUT_DIRECTORY / GLOSSARION_DATA_DIR point at pytest's tmp dir, GLOSSARION_HTTP_LOG=0, and src/config.json keeps
its md5.

Run from src/mobile with the mobile venv::

    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_hotfix3_multipass.py
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import importlib.util
import json
import os
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


needs_flet = pytest.mark.skipif(not (_has("flet") and _has("msgpack")), reason="flet / msgpack not installed")
_RUN_NEEDED = ("flet", "msgpack", "ebooklib", "openai", "httpx", "tiktoken", "bs4", "lxml")
needs_runs = pytest.mark.skipif(not all(_has(m) for m in _RUN_NEEDED),
                                reason=f"needs {', '.join(_RUN_NEEDED)} (the mobile project venv)")
needs_backend = pytest.mark.skipif(not (SRC_DIR / "headless_owner.py").is_file(), reason="src/ backend not present")

#: What an older build / the desktop stored for the retired switch (both keys the dialog reads).
DESKTOP_FORCE_KEYS = {"direct_text_force_multipass_off": True, "direct_text_force_simple_mode": True}
_REPO_CONFIG = SRC_DIR / "config.json"
RUN_TIMEOUT = 120.0


def _md5(path: Path):
    return hashlib.md5(path.read_bytes()).hexdigest() if path.is_file() else None


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    """No test reads or writes the user's Library, output folders, home, app data or data folder."""
    repo_md5 = _md5(_REPO_CONFIG)
    for name, sub in (("HOME", "home"), ("USERPROFILE", "home"), ("APPDATA", "appdata"),
                      ("LOCALAPPDATA", "localappdata"), ("GLOSSARION_LIBRARY_DIR", "lib"),
                      ("OUTPUT_DIRECTORY", "out"), ("GLOSSARION_DATA_DIR", "data")):
        folder = tmp_path / "_iso" / sub
        folder.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv(name, str(folder))
    monkeypatch.setenv("GLOSSARION_HTTP_LOG", "0")
    monkeypatch.setenv("GLOSSARION_DIRECT_TEXT_HISTORY", str(tmp_path / "_iso" / "direct_text_chats.json"))
    yield tmp_path / "_iso"
    assert _md5(_REPO_CONFIG) == repo_md5, "src/config.json changed"


def _getter(values: dict):
    return lambda key, default=None: values.get(key, default)


def _restored_env():
    before = dict(os.environ)

    def restore():
        os.environ.clear()
        os.environ.update(before)

    return restore


# ==========================================================================
# 1. No "Force Multipass off" in the chat settings model, the sheet, the Series page or the pills
# ==========================================================================


def test_chat_settings_have_no_force_multipass_off_and_never_write_its_key():
    from glossarion_mobile.ui.chat import direct_text_rules as rules
    from glossarion_mobile.ui.chat.direct_text_rules import FORCE_MULTIPASS_OFF, DirectTextSettings

    assert FORCE_MULTIPASS_OFF is False
    assert "force_multipass_off" not in {f.name for f in dataclasses.fields(DirectTextSettings)}
    # an imported desktop config with the dialog's checkbox ON (its default) changes nothing on mobile
    fresh = DirectTextSettings.from_config(_getter({}))
    imported = DirectTextSettings.from_config(_getter(dict(DESKTOP_FORCE_KEYS)))
    assert imported == fresh
    # a chat override an older build stored is ignored; the other overrides still apply
    stale = fresh.with_overrides({"force_multipass_off": True, "disable_thinking": True})
    assert stale == dataclasses.replace(fresh, disable_thinking=True)
    # All chats never writes the desktop-only key
    assert not any("multipass" in key or "simple_mode" in key for key in fresh.config_updates())
    assert "force_multipass_off" not in rules.__dict__.get("SIDECAR_META_FIELDS", ())


@needs_flet
def test_chat_settings_sheet_and_series_page_show_no_force_multipass_off():
    from glossarion_mobile.ui.chat import series_page
    from glossarion_mobile.ui.sheets import chat_settings
    from glossarion_mobile.ui.sheets.chat_settings import ChatSettingsSheet

    assert "force_multipass_off" not in chat_settings.CHAT_SETTING_KEYS
    assert "force_multipass_off" not in chat_settings.DESCRIPTIONS
    assert chat_settings.global_updates_for("disable_thinking", True) == {"direct_text_disable_thinking": True}

    class Chats:
        def __init__(self):
            # what an older build left in the chat's sidecar
            self.values = {"force_multipass_off": True}

        def overrides(self, cid=None):
            return dict(self.values)

        own_overrides = overrides

        def meta(self, cid=None):
            return {}

        def set_override(self, cid, name, value):
            self.values[name] = value

        def set_meta(self, cid, name, value):
            return None

        def reset_overrides(self, cid=None):
            self.values.clear()

    class Config(dict):
        def set_many(self, updates):
            self.update(updates)

    def keys_and_texts(control, keys, texts):
        if control is None:
            return
        key = getattr(control, "key", None)
        if key:
            keys.append(str(key))
        for name in ("value", "label", "title", "content", "tooltip"):
            value = getattr(control, name, None)
            if isinstance(value, str):
                texts.append(value)
        for name in ("controls", "content", "title", "subtitle", "leading", "trailing"):
            child = getattr(control, name, None)
            for item in (child if isinstance(child, list) else [child]):
                if item is not None and not isinstance(item, str):
                    keys_and_texts(item, keys, texts)

    for subject, scope in (("chat", "chat"), ("chat", "global"), ("series", "chat")):
        config = Config(DESKTOP_FORCE_KEYS)
        sheet = ChatSettingsSheet(cid="2", config=config, chats=Chats(), scope=scope, subject=subject)
        run = next(c for c in sheet.body.controls if type(c).__name__ == "ExpansionTile" and c.title == "Run behaviour")
        keys, texts = [], []
        for control in sheet.body.controls:
            keys_and_texts(control, keys, texts)
        assert "setting-force_multipass_off" not in keys, (subject, scope)
        assert not any("Multipass" in text for text in texts), (subject, scope, [t for t in texts if "ultipass" in t])
        assert run.controls[0].key == "setting-disable_thinking", (subject, scope)  # Run behaviour starts there now
        assert config == DESKTOP_FORCE_KEYS  # building the sheet writes nothing
    # the Series page's Defaults card lists no such row, also for a default an older build stored
    assert "force_multipass_off" not in dict(series_page.DEFAULT_LABELS)
    assert series_page.defaults_summary({"force_multipass_off": True, "disable_thinking": True}) == [
        ("Disable all thinking", "On")]


@needs_flet
@needs_backend
def test_the_composer_pills_have_no_multipass_pill(tmp_path):
    from glossarion_mobile.ui.chat.chat_view import ChatView

    class Chats:
        def overrides(self, cid):
            return {"force_multipass_off": False}  # an older build's "Multipass on" chat

        def meta(self, cid):
            return {}

    view = types.SimpleNamespace(bound=True, cid="2", env=types.SimpleNamespace(chats=Chats()))
    view.settings = lambda cid=None: __import__(
        "glossarion_mobile.ui.chat.direct_text_rules", fromlist=["DirectTextSettings"]).DirectTextSettings()
    assert ChatView.option_pills(view) == []
    resets = []
    view.env.chats.set_override = lambda cid, name, value: resets.append(name)
    view.apply_settings_changed = lambda: None
    ChatView._on_pill_reset(view, "multipass")  # an old pill id resets nothing
    assert resets == []


# ==========================================================================
# 2. Chat runs never force Multipass off; stale stored values are not read
# ==========================================================================


def test_every_chat_run_follows_the_multipass_setting(tmp_path):
    from glossarion_mobile.ui.chat.direct_text_rules import DirectTextSettings
    from glossarion_mobile.ui.chat.run_request import (
        DirectTextRun,
        job_params,
        prepare_direct_text_run,
        run_options_dict,
        with_current_glossary,
    )

    if not (SRC_DIR / "direct_text_store.py").is_file():
        pytest.skip("src/direct_text_store.py not present")
    run = prepare_direct_text_run(text="안녕하세요", attachment=None, output_mode="text",
                                  attachment_prompt_role="user", temp_dir=str(tmp_path))
    for settings in (DirectTextSettings(),
                     DirectTextSettings.from_config(_getter(dict(DESKTOP_FORCE_KEYS))),
                     DirectTextSettings().with_overrides({"force_multipass_off": True})):
        assert run_options_dict(run, settings)["force_multipass_off"] is False
        assert job_params(chat_id=2, user_index=0, run=run, settings=settings)["options"]["force_multipass_off"] is False
    # Resume / Retry of a run an older build sent with "Force Multipass off" follows the setting now too
    old_options = {**run_options_dict(run, DirectTextSettings()), "force_multipass_off": True}
    resumed = with_current_glossary(old_options, DirectTextRun.from_dict(run.as_dict()), DirectTextSettings())
    assert resumed["force_multipass_off"] is False
    assert {k: v for k, v in resumed.items() if k != "force_multipass_off"} == {
        k: v for k, v in old_options.items() if k != "force_multipass_off"}


def test_stale_sidecar_override_and_series_default_are_not_read(tmp_path):
    if not (SRC_DIR / "direct_text_store.py").is_file():
        pytest.skip("src/direct_text_store.py not present")
    import direct_text_store
    from glossarion_mobile.state import series as S
    from glossarion_mobile.state.chat_store_adapter import (
        OVERRIDE_KEYS,
        SIDECAR_NAME,
        ChatStoreAdapter,
        ChatStoreBinding,
    )

    assert "force_multipass_off" not in OVERRIDE_KEYS and set(S.DEFAULT_KEYS) == set(OVERRIDE_KEYS)
    root = tmp_path / "chats"
    root.mkdir()
    history = root / "direct_text_chats.json"
    history.write_text(json.dumps({"version": 2, "current_chat_id": 2, "sessions": [
        {"id": 2, "title": "Old chat", "messages": [["user", "hello"]], "draft": "", "attachment": None,
         "output_folder": "", "output_folder_name": "", "next_output_index": 1, "expanded": []}]}), encoding="utf-8")
    # the sidecar an older build wrote: "Force Multipass off" turned ON for this chat, next to a real override
    (root / SIDECAR_NAME).write_text(json.dumps({"version": 1, "chats": {"2": {"overrides": {
        "force_multipass_off": True, "disable_thinking": True}}}, "scratch": []}), encoding="utf-8")
    binding = ChatStoreBinding(direct_text_store.ChatStore(str(history), output_root=str(tmp_path / "Output")),
                               history_path=str(history))
    chats = ChatStoreAdapter(binding, history_path=str(history), scratch_dir=str(tmp_path / "scratch"))
    try:
        assert chats.load(), chats.load_error
        assert chats.own_overrides("2") == {"disable_thinking": True}
        assert chats.overrides("2") == {"disable_thinking": True}
        with pytest.raises(KeyError):
            chats.set_override("2", "force_multipass_off", True)
        # a Series default an older build stored is dropped when the store loads
        store = S.SeriesStore(str(tmp_path / S.SERIES_FILE))
        (tmp_path / S.SERIES_FILE).write_text(json.dumps({"version": 1, "series": [
            {"id": "saga_1", "name": "Saga", "defaults": {"force_multipass_off": True, "model": "m"}}]}),
            encoding="utf-8")
        assert store.load() is True and store.defaults("saga_1") == {"model": "m"}
        chats.override_layers.append(lambda cid: {"force_multipass_off": True, "model": "m"})
        assert chats.overrides("2") == {"model": "m", "disable_thinking": True}
    finally:
        chats.close()


# ==========================================================================
# 3. Fresh config: Settings shows Multipass OFF and the run env matches; a saved ON is respected
# ==========================================================================


def test_a_job_an_older_build_stored_runs_with_multipass_following_the_setting(tmp_path):
    """The Jobs page / launch banner resume a job with its stored params: an older build's ``force_multipass_off:
    True`` (or no key: the desktop dataclass default is True) must not force Multipass off."""
    pytest.importorskip("headless_owner")
    from glossarion_mobile.job_kinds import direct_text

    source = tmp_path / "direct_text.txt"
    source.write_text("hello", encoding="utf-8")
    assert direct_text.build_options({"options": {"force_multipass_off": True}}, str(source)).force_multipass_off is False
    assert direct_text.build_options({}, str(source)).force_multipass_off is False


def _store(root: Path, data=None):
    from glossarion_mobile.state.config_store import MobileConfigStore
    from glossarion_mobile.ui.settings.schema_access import SchemaAccess

    root.mkdir(parents=True, exist_ok=True)
    path = root / "config.json"
    path.write_text(json.dumps(data or {}), encoding="utf-8")
    schema = SchemaAccess()
    store = MobileConfigStore(path, debounce=10, defaults=schema.effective_default,
                              reader=lambda p, decrypt=True: json.loads(Path(p).read_text(encoding="utf-8")),
                              writer=lambda disk, p, backup=False: Path(p).write_text(json.dumps(disk), encoding="utf-8"))
    store.load()
    return store, schema


def _job_config(store) -> dict:
    """The config snapshot a job's HeadlessOwner is built from (``JobService._config_snapshot``)."""
    from glossarion_mobile.services.jobs import JobService

    holder = types.SimpleNamespace(config_store=store, _config_loader=None)
    job = types.SimpleNamespace(spec=types.SimpleNamespace(params={"config_overrides": {}}))
    try:
        return JobService._config_snapshot(holder, job)
    finally:
        store.set_job_running(False)


def _chat_run_multipass_env(tmp_path: Path, config: dict, chat_overrides=None) -> dict:
    """A chat send's run on a real HeadlessOwner: the chat's options (``run_request``) applied by the
    ``direct_text`` job adapter, its run environment, then the export ``_prepare_translation_run`` makes."""
    import headless_owner
    from glossarion_mobile.job_kinds import direct_text
    from glossarion_mobile.ui.chat.direct_text_rules import DirectTextSettings
    from glossarion_mobile.ui.chat.run_request import prepare_direct_text_run, run_options_dict

    settings = DirectTextSettings.from_config(_getter(config)).with_overrides(chat_overrides or {})
    run = prepare_direct_text_run(text="안녕하세요. 오늘은 비가 옵니다.", attachment=None, output_mode="text",
                                  attachment_prompt_role="user", temp_dir=str(tmp_path))
    restore = _restored_env()
    try:
        os.environ.pop("MULTIPASS_MODE", None)
        owner = headless_owner.HeadlessOwner(config)
        options = direct_text.build_options({"options": run_options_dict(run, settings)}, run.source_path)
        options.apply_to(owner)
        direct_text.apply_run_environment(owner, {"output_root": run.temp_root, "is_attachment": False})
        after_env = os.environ.get("MULTIPASS_MODE")
        enabled, _mode = owner._export_multipass_runtime_env()
        return {"forced": bool(owner._direct_text_force_multipass_off), "after_run_env": after_env,
                "enabled": bool(enabled), "MULTIPASS_MODE": os.environ.get("MULTIPASS_MODE")}
    finally:
        restore()


@needs_backend
def test_fresh_config_shows_multipass_off_and_the_chat_run_env_matches(tmp_path, monkeypatch):
    pytest.importorskip("run_env")
    glossary_paths = pytest.importorskip("glossary_paths")
    import headless_owner
    import settings_schema

    monkeypatch.setattr(glossary_paths, "migrate_all_legacy_glossary_files", lambda *a, **k: None)
    # the desktop keeps its own defaults: Multipass off, the dialog's "Force Multipass off" on
    assert settings_schema.effective_default("multipass_mode") is False
    assert {f.name: f.default for f in dataclasses.fields(headless_owner.DirectTextRunOptions)}[
        "force_multipass_off"] is True

    # ---- a fresh install: Multipass shows OFF (the default, nothing stored) and the run exports 0 ----
    store, _schema = _store(tmp_path / "fresh")
    try:
        assert store.effective("multipass_mode") is False and not store.has("multipass_mode")
        fresh = _chat_run_multipass_env(tmp_path / "run_fresh", _job_config(store))
        store.flush()
        assert json.loads((tmp_path / "fresh" / "config.json").read_text(encoding="utf-8")) == {}
    finally:
        store._saver.close()
    # the dialog's env export (forced streaming re-exports the run env) already carries the setting, then the
    # pipeline's own export: 0 both times, never forced
    assert fresh == {"forced": False, "after_run_env": "0", "enabled": False, "MULTIPASS_MODE": "0"}, fresh

    # ---- Multipass ON (saved in Settings, or imported) is respected by the chat run ------------------------
    for label, data, overrides in (
        ("saved", {"multipass_mode": True}, None),
        # an imported desktop config / an older build's chat: "Force Multipass off" ON next to Multipass ON
        ("imported", {"multipass_mode": True, **DESKTOP_FORCE_KEYS}, {"force_multipass_off": True}),
    ):
        store, _schema = _store(tmp_path / label, data)
        try:
            assert store.effective("multipass_mode") is True
            seen = _chat_run_multipass_env(tmp_path / f"run_{label}", _job_config(store), overrides)
        finally:
            store._saver.close()
        # before hotfix 3 both exports said 0 here ("Force Multipass off" was on for every chat run)
        assert seen == {"forced": False, "after_run_env": "1", "enabled": True, "MULTIPASS_MODE": "1"}, (label, seen)


# ==========================================================================
# 4. The owner's flow on the real app: chat runs follow Settings › Translation defaults › Multipass
# ==========================================================================


def _load(alias: str, file_name: str):
    spec = importlib.util.spec_from_file_location(alias, Path(__file__).with_name(file_name))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if all(_has(m) for m in ("flet", "msgpack")):
    # the devfix #14 acceptance harness (real app on the in-memory Flet session, phone storage, chat send helper)
    _DF = _load("_glossarion_df14_hotfix3_multipass", "test_devfix_issue14.py")
    storage = _DF.storage  # noqa: F811  (pytest fixtures)
    app_env = _DF.app_env
    phone = _DF.phone
else:  # pragma: no cover - CI without Flet runs the pure tests only
    _DF = None


class MultipassEnv:
    """The MULTIPASS_MODE each fake-model request met (the job sets os.environ for its run)."""

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


def _sheet_keys(sheet) -> list:
    keys: list = []

    def walk(control):
        if control is None or isinstance(control, str):
            return
        key = getattr(control, "key", None)
        if key:
            keys.append(str(key))
        for name in ("controls", "content", "title", "subtitle", "leading", "trailing"):
            child = getattr(control, name, None)
            for item in (child if isinstance(child, list) else [child]):
                walk(item)

    for control in sheet.body.controls:
        walk(control)
    return keys


@needs_runs
@needs_backend
def test_owner_flow_chat_runs_follow_the_multipass_setting(phone):
    from glossarion_mobile.diagnostics.fake_llm_server import FAKE_MARKER, FakeLLMServer

    df = _DF

    def refinements(seen_requests) -> list:
        """The refinement pass's requests: the chapters again, as the fake model translated them (its marker)."""
        return [r for r, _e in seen_requests if FAKE_MARKER in (r.preview or "") and not r.chapters]
    books = {label: df._write_epub(phone.picks / f"Multipass {label}.epub", f"Multipass {label}")
             for label in ("off", "on")}

    async def scenario(server):
        tf, app, tester, driver = await df._start_app({path.name: path for path in books.values()})
        seen = MultipassEnv(server)
        try:
            df._configure(app, server)
            store = app.config_store
            disk_before = df._disk(phone)
            assert "multipass_mode" not in disk_before and not set(DESKTOP_FORCE_KEYS) & set(disk_before)

            # ---- Chat settings (the composer's ⚙): no "Force Multipass off" in either scope ------------------
            await driver.tap(tooltip="New chat")
            view = app.chat_view
            await driver.tap(key="pill-options", timeout=15)
            await df._until(lambda: getattr(view, "settings_sheet", None) is not None, 10, "Chat settings")
            sheet = view.settings_sheet
            for scope in ("chat", "global"):
                sheet.scope = scope
                sheet.rebuild()
                keys = _sheet_keys(sheet)
                assert "setting-disable_thinking" in keys and "setting-force_multipass_off" not in keys, (scope, keys)
            sheet.close()
            assert view.option_pills() == []  # nothing differs from the defaults

            # ---- Settings › Translation defaults › Multipass: OFF, the default (nothing stored) -------------
            page = await df._settings_section(app, driver, "translation_defaults")
            tile = page.tile("multipass_mode")
            assert tile.switch.value is False and tile.value_text.value.endswith("· default"), tile.value_text.value
            assert store.effective("multipass_mode") is False and not store.has("multipass_mode")
            import flows

            await flows.go_home(driver)

            # ---- a chat message on the fresh phone: one translation pass, MULTIPASS_MODE 0 -------------------
            mark = server.mark()
            await df._chat_message(app, driver, "안녕하세요. 오늘은 비가 옵니다.")
            off = seen.since(mark)
            assert off and {env for _r, env in off} == {"0"}, [(r.kind, env) for r, env in off]
            # ---- a book in the chat (＋ › Files, Plan card Start): one translation pass, MULTIPASS_MODE 0 ---------
            mark = server.mark()
            await df._chat_attachment(app, driver, books["off"].name)
            book_off = seen.since(mark)
            assert book_off and {env for _r, env in book_off} == {"0"}, [(r.kind, env) for r, env in book_off]
            assert refinements(book_off) == [], [r.preview for r, _e in book_off]

            # ---- the owner turns Multipass ON in Settings: the chat run follows it ---------------------------
            page = await df._settings_section(app, driver, "translation_defaults")
            tile = page.tile("multipass_mode")
            await df._tap(tester, tile.switch)
            assert tile.switch.value is True and store.get("multipass_mode") is True
            store.flush()
            assert df._disk(phone).get("multipass_mode") is True
            assert not set(DESKTOP_FORCE_KEYS) & set(df._disk(phone))  # no Force Multipass off key appears
            await flows.go_home(driver)
            mark = server.mark()
            await df._chat_message(app, driver, "안녕하세요. 오늘은 비가 옵니다.")
            on = seen.since(mark)
            assert on and {env for _r, env in on} == {"1"}, [(r.kind, env) for r, env in on]
            mark = server.mark()
            await df._chat_attachment(app, driver, books["on"].name)
            book_on = seen.since(mark)
            assert book_on and {env for _r, env in book_on} == {"1"}, [(r.kind, env) for r, env in book_on]
            # the multipass refinement pass ran over the translated chapters after the translation (the desktop
            # Multipass behaviour): both chapters are sent again, as translated, and nothing else changed
            assert len(refinements(book_on)) == 2, [(r.chapters, r.preview[:80]) for r, _e in book_on]
            assert sorted(map(tuple, (r.chapters for r, _e in book_on if r.chapters))) == sorted(
                map(tuple, (r.chapters for r, _e in book_off if r.chapters)))
        finally:
            await df._stop_app(tf, app)

    with FakeLLMServer() as server:
        asyncio.run(scenario(server))
