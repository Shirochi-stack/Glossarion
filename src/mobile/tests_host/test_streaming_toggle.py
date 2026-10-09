"""Host tests for devfix4 item 14: one Streaming switch on Glossarion Mobile (owner 2026-10-08).

The owner's report: Settings › Response handling & retries showed four separate streaming switches
(Enable streaming responses, Stream thinking/reasoning logs, Allow streaming logs during batch mode,
Allow forced-stream batch log) and Thinking & reasoning a separate "Enable thoughts". Decisions:

* ONE "Streaming" switch (``VIRTUAL_SPECS['streaming']`` / ``StreamingTile``) drives the four keys
  together with the exact desktop thoughts coupling (stream thinking ON locks Enable thoughts on, OFF
  unchecks it); the Enable thoughts tile is gone; only keys whose value changes are written (the
  stream-thinking rule turns thoughts off even on a no-op write);
* ON by default on mobile when the keys are absent (a saved / imported value wins, config.json gets
  nothing, the desktop default stays off); toggles that differ show "Custom", one tap normalises;
* OFF stops streaming everywhere on mobile: book jobs (every job's config snapshot), chats (the desktop
  Direct Text run always forces streaming; mobile skips that while the switch is off) and the Reader's
  live translation (``single_chapter``'s ``force_stream_all``).

Real data is never touched: HOME / USERPROFILE / APPDATA / LOCALAPPDATA / GLOSSARION_LIBRARY_DIR /
OUTPUT_DIRECTORY / GLOSSARION_DATA_DIR point at pytest's tmp dir (autouse fixture), GLOSSARION_HTTP_LOG=0.

Run from src/mobile:
    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_streaming_toggle.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = MOBILE_DIR / "app"
SRC_DIR = MOBILE_DIR.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))


def _has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


needs_flet = pytest.mark.skipif(not (_has("flet") and _has("msgpack")), reason="flet / msgpack not installed")

#: The desktop "Real-time Translation (Streaming)" group (other_settings._create_response_handling_section).
FOUR = ("enable_streaming", "stream_thinking_logs", "allow_batch_stream_logs", "allow_authgpt_batch_stream_logs")
WRITES = FOUR + ("enable_thoughts",)
ALL_OFF = {key: False for key in FOUR}


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    """No test reads or writes the user's Library, output folders, home, app data or data folder."""
    for name, sub in (("HOME", "home"), ("USERPROFILE", "home"), ("APPDATA", "appdata"),
                      ("LOCALAPPDATA", "localappdata"), ("GLOSSARION_LIBRARY_DIR", "lib"),
                      ("OUTPUT_DIRECTORY", "out"), ("GLOSSARION_DATA_DIR", "data")):
        folder = tmp_path / "_iso" / sub
        folder.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv(name, str(folder))
    monkeypatch.setenv("GLOSSARION_HTTP_LOG", "0")
    yield tmp_path / "_iso"


def _store(tmp_path, data=None):
    from glossarion_mobile.state.config_store import MobileConfigStore
    from glossarion_mobile.ui.settings.schema_access import SchemaAccess

    path = tmp_path / "config.json"
    path.write_text(json.dumps(data or {}), encoding="utf-8")
    schema = SchemaAccess()
    store = MobileConfigStore(path, debounce=10, defaults=schema.effective_default,
                              reader=lambda p, decrypt=True: json.loads(Path(p).read_text(encoding="utf-8")),
                              writer=lambda disk, p, backup=False: Path(p).write_text(json.dumps(disk), encoding="utf-8"))
    store.load()
    return store, schema


def _disk(tmp_path) -> dict:
    return json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))


def _ctx(tmp_path, data=None, **kw):
    from glossarion_mobile.ui.settings.context import SettingsContext

    store, schema = _store(tmp_path, data)
    return SettingsContext(page=None, store=store, schema=schema, **kw)


class _CountingStore:
    """Counts the store writes a Streaming tap makes (one ``set_many`` = one save, one observer pass)."""

    def __init__(self, store):
        self.store = store
        self.writes: list = []

    def __getattr__(self, name):
        return getattr(self.store, name)

    def set_many(self, values):
        self.writes.append(dict(values))
        return self.store.set_many(values)


def _restored_env():
    before = dict(os.environ)

    def restore():
        os.environ.clear()
        os.environ.update(before)

    return restore


# ==========================================================================
# the owner's complaint: one switch, no Enable thoughts tile
# ==========================================================================


def test_owner_complaint_one_streaming_switch_replaces_four_rows_and_enable_thoughts():
    """Owner: "one mobile Streaming toggle" - not the four desktop rows plus a separate Enable thoughts."""
    from glossarion_mobile.ui.settings.schema_access import SchemaAccess

    schema = SchemaAccess()
    shown = [str(spec.key) for section in schema.sections() for spec in schema.specs_for(section)]
    assert not set(WRITES) & set(shown), "a folded streaming key still has its own tile"
    assert shown.count("streaming") == 1
    response = schema.section("other.response")
    first = schema.specs_for(response)[0]
    assert (first.key, first.label, first.virtual) == ("streaming", "Streaming", "streaming")
    assert dict(response.headings)["streaming"] == "Streaming"
    # one search hit for the switch, whatever desktop name or env var is typed
    for query in ("stream", "ENABLE_STREAMING", "thinking logs", "Enable thoughts", "forced-stream batch log",
                  "ALLOW_AUTHGPT_BATCH_STREAM_LOGS"):
        keys = [hit.key for hit in schema.search(query)]
        assert [k for k in keys if k in WRITES or k == "streaming"] == ["streaming"], (query, keys)


def test_streaming_keys_match_the_desktop_group_and_the_shared_rules():
    import settings_rules
    import settings_schema
    from glossarion_mobile.state import setting_writes as sw
    from glossarion_mobile.ui.settings.model import FOLDED_KEYS, VIRTUAL_SPECS, env_names, virtual_writes
    from glossarion_mobile.ui.settings.schema_access import SchemaAccess

    assert sw.STREAMING_KEYS == settings_rules.STREAMING_KEYS == FOUR  # mobile literal == the shared rules
    assert tuple(settings_rules.STREAMING_ENV) == FOUR
    assert sw.STREAMING_WRITES == WRITES and settings_rules.THOUGHTS_LOCK_KEYS == ("enable_thoughts",)
    streaming_specs = {s.key for s in settings_schema.all_specs()
                       if any("STREAM" in name for name in env_names(s))}
    assert streaming_specs == set(FOUR)
    assert {settings_schema.spec(k).section for k in FOUR} == {"other.response"}
    assert FOLDED_KEYS == {key: "streaming" for key in WRITES}
    assert virtual_writes("streaming") == WRITES and virtual_writes("max_retries") == ("max_retries",)
    assert VIRTUAL_SPECS["streaming"].default is sw.MOBILE_STREAMING_DEFAULT is True
    schema = SchemaAccess()
    # every schema key is still in exactly one section (the folded ones in Response handling)
    before = sorted(k for s in settings_schema.sections() for k in s.keys)
    after = sorted(k for s in schema.sections() for k in s.keys if k not in VIRTUAL_SPECS)
    assert before == after
    for key in WRITES:
        assert schema.section_for_key(key).id == "other.response"
        assert schema.display_key(key) == "streaming"
    assert schema.represented_keys("streaming") == WRITES
    assert "enable_thoughts" not in schema.section("thinking").keys


# ==========================================================================
# default ON on mobile, the desktop default unchanged
# ==========================================================================


def test_absent_toggles_are_on_on_mobile_and_off_on_desktop(tmp_path):
    import settings_rules
    import settings_schema
    from glossarion_mobile.state.setting_writes import streaming_enabled, streaming_mode, write_setting

    store, _schema = _store(tmp_path)
    assert streaming_mode(store) == "on" and streaming_enabled(store)
    assert all(store.effective(key) is True for key in FOUR)  # Settings shows what runs use
    assert not any(store.is_modified(key) for key in WRITES)
    assert write_setting(store, "streaming", True) == []  # already on: nothing written
    store.flush()
    assert _disk(tmp_path) == {}  # config.json gets no key for the default
    # the desktop keeps its defaults (start-up / schema): absent means off there
    assert settings_rules.streaming_mode({}) == "off"
    assert all(settings_schema.effective_default(key) is False for key in FOUR)
    # a saved (or imported) value wins over the mobile default
    (tmp_path / "saved").mkdir()
    saved, _schema = _store(tmp_path / "saved", ALL_OFF)
    assert streaming_mode(saved) == "off" and not streaming_enabled(saved)
    for one in (store, saved):
        one._saver.close()


# ==========================================================================
# the writer: only changed keys, the desktop thoughts coupling, reset
# ==========================================================================


def test_switch_writes_only_changed_keys_with_the_desktop_thoughts_coupling(tmp_path):
    import settings_rules
    from glossarion_mobile.state.setting_writes import STREAMING_KEY, streaming_mode, write_setting

    store, _schema = _store(tmp_path)
    counting = _CountingStore(store)
    assert sorted(write_setting(counting, STREAMING_KEY, False)) == sorted(WRITES)
    assert counting.writes == [{key: False for key in WRITES}]  # one write, thoughts unchecked like the desktop
    store.flush()
    assert _disk(tmp_path) == {key: False for key in WRITES} and "streaming" not in _disk(tmp_path)
    counting.writes.clear()
    write_setting(counting, STREAMING_KEY, True)
    assert counting.writes == [{key: True for key in WRITES}] and streaming_mode(store) == "on"
    # the desktop checkbox does the same for thoughts (settings_rules.apply_change of stream thinking)
    desktop = {key: True for key in WRITES}
    changed, _env = settings_rules.apply_change(desktop, "stream_thinking_logs", False)
    assert changed == {"stream_thinking_logs": False, "enable_thoughts": False}
    store._saver.close()


def test_off_never_runs_the_no_op_thoughts_trap_and_on_reapplies_the_lock(tmp_path):
    from glossarion_mobile.state.setting_writes import STREAMING_KEY, streaming_mode, write_setting

    # stream thinking already off, thoughts on: OFF changes Enable streaming responses only
    custom = {**ALL_OFF, "enable_streaming": True, "enable_thoughts": True}
    store, _schema = _store(tmp_path, custom)
    assert streaming_mode(store) == "custom"
    assert write_setting(store, STREAMING_KEY, False) == ["enable_streaming"]
    assert store.get("enable_thoughts") is True and streaming_mode(store) == "off"
    store._saver.close()
    # a desktop-wizard config (streaming + stream thinking, thoughts stored off): ON re-applies the lock
    (tmp_path / "wizard").mkdir()
    wizard = {**ALL_OFF, "enable_streaming": True, "stream_thinking_logs": True, "enable_thoughts": False}
    store, _schema = _store(tmp_path / "wizard", wizard)
    write_setting(store, STREAMING_KEY, True)
    assert all(store.get(key) is True for key in WRITES)
    store._saver.close()


def test_reset_goes_back_to_the_mobile_default_with_undo(tmp_path):
    from glossarion_mobile.state.setting_writes import reset_streaming, streaming_mode

    stored = {**ALL_OFF, "enable_thoughts": False, "max_retries": 3}
    store, _schema = _store(tmp_path, stored)
    old = reset_streaming(store)
    assert old == {key: False for key in WRITES}
    assert streaming_mode(store) == "on" and store.snapshot() == {"max_retries": 3}
    store.set_many(old)  # Undo
    assert store.snapshot() == stored
    store._saver.close()


# ==========================================================================
# Streaming OFF everywhere: book jobs, chats, the Reader's live translation
# ==========================================================================


def _job_config(store, overrides=None):
    from glossarion_mobile.services.jobs import JobService

    holder = types.SimpleNamespace(config_store=store, _config_loader=None)
    job = types.SimpleNamespace(spec=types.SimpleNamespace(params={"config_overrides": dict(overrides or {})}))
    try:
        return JobService._config_snapshot(holder, job)
    finally:
        store.set_job_running(False)


def test_every_job_config_carries_the_mobile_default_and_the_thoughts_lock(tmp_path):
    store, _schema = _store(tmp_path)
    config = _job_config(store, {"model": "x/y"})
    assert {key: config[key] for key in FOUR} == {key: True for key in FOUR} and config["model"] == "x/y"
    assert "enable_thoughts" not in config  # its own default (on) applies
    assert store.snapshot() == {} and _disk(tmp_path) == {}  # the store and config.json are untouched
    store._saver.close()
    (tmp_path / "off").mkdir()
    store, _schema = _store(tmp_path / "off", {**ALL_OFF, "enable_thoughts": False})
    assert {key: _job_config(store)[key] for key in WRITES} == {key: False for key in WRITES}
    store._saver.close()
    (tmp_path / "stale").mkdir()
    store, _schema = _store(tmp_path / "stale", {"stream_thinking_logs": True, "enable_thoughts": False})
    assert _job_config(store)["enable_thoughts"] is True  # stream thinking on keeps thoughts on
    store._saver.close()


def test_book_job_env_follows_the_switch(tmp_path, monkeypatch):
    headless_owner = pytest.importorskip("headless_owner")
    run_env = pytest.importorskip("run_env")
    glossary_paths = pytest.importorskip("glossary_paths")
    from glossarion_mobile.state.setting_writes import with_mobile_streaming_defaults

    monkeypatch.setattr(glossary_paths, "migrate_all_legacy_glossary_files", lambda *a, **k: None)
    restore = _restored_env()
    try:
        exported = {}
        for label, config in (("default", {}), ("off", dict(ALL_OFF))):
            owner = headless_owner.HeadlessOwner(with_mobile_streaming_defaults(config))
            env = run_env.build_translation_env(owner, str(tmp_path / "in" / "Book.epub"), "")
            exported[label] = {settings_name: env.get(settings_name) for settings_name in
                               ("ENABLE_STREAMING", "ALLOW_BATCH_STREAM_LOGS", "ALLOW_AUTHGPT_BATCH_STREAM_LOGS",
                                "STREAM_THINKING_LOGS")}
    finally:
        restore()
    assert set(exported["default"].values()) == {"1"}
    assert set(exported["off"].values()) == {"0"}


def test_streaming_off_stops_the_chat_forcing_streaming(tmp_path):
    """Desktop Direct Text always forces every streaming switch on; on mobile only while Streaming is on."""
    headless_owner = pytest.importorskip("headless_owner")
    run_env = pytest.importorskip("run_env")
    from glossarion_mobile.job_kinds import direct_text
    from glossarion_mobile.state.setting_writes import with_mobile_streaming_defaults

    forced = run_env.FORCED_STREAM_ENV_KEYS
    restore = _restored_env()
    try:
        seen = {}
        for label, config in (("default", {}), ("off", {key: False for key in WRITES})):  # after an OFF tap
            for key in forced:
                os.environ.pop(key, None)
            owner = headless_owner.HeadlessOwner(with_mobile_streaming_defaults(config))
            source = tmp_path / "direct_text.txt"
            source.write_text("hello", encoding="utf-8")
            options = direct_text.build_options({"options": {"force_stream_all": True}}, str(source))
            options.apply_to(owner)
            direct_text.apply_run_environment(owner, {"output_root": str(tmp_path / "run"), "is_attachment": False})
            seen[label] = (owner._force_stream_all, {key: os.environ.get(key) for key in forced})
    finally:
        restore()
    stream_all, env = seen["default"]
    assert stream_all is True and set(env.values()) == {"1"}  # desktop parity while Streaming is on
    stream_all, env = seen["off"]
    assert stream_all is False and "1" not in env.values()  # nothing forced while it is off


def test_streaming_off_stops_the_reader_live_view_forcing(tmp_path, monkeypatch):
    from glossarion_mobile.job_kinds import single_chapter

    epub = tmp_path / "Book.epub"
    epub.write_bytes(b"PK")
    seen = []
    monkeypatch.setattr(single_chapter, "run_translation",
                        lambda ctx, files: seen.append(ctx.owner._force_stream_all) or {"ok": True, "outputs": []})
    for config in ({}, dict(ALL_OFF)):
        logs: list = []
        ctx = types.SimpleNamespace(inputs=(str(epub),), owner=types.SimpleNamespace(config=config), log=logs.append,
                                    params={"chapter_file": "OEBPS/chapter0001.xhtml", "force_stream_all": True})
        single_chapter.run(ctx)
        assert ctx.owner._force_stream_all is False  # reset after the run either way
    assert seen == [True, False]
    assert any("Streaming is off" in line for line in logs)


# ==========================================================================
# the tile, the section pages, Settings home
# ==========================================================================


@needs_flet
def test_streaming_tile_custom_state_and_one_tap_normalises(tmp_path):
    import settings_rules
    from glossarion_mobile.ui.components.reason_chip import ReasonChip
    from glossarion_mobile.ui.settings.tiles import StreamingTile, make_tile

    notes: list = []
    imported = {"enable_streaming": True, "stream_thinking_logs": True, "allow_batch_stream_logs": False,
                "allow_authgpt_batch_stream_logs": False}
    ctx = _ctx(tmp_path, imported, notify=lambda *args: notes.append(args))
    tile = make_tile(ctx.schema.spec("streaming"), ctx)
    assert isinstance(tile, StreamingTile) and tile.editable
    chips = [c.reason for c in tile.badges.controls if isinstance(c, ReasonChip)]
    assert tile.switch.value is True and tile.summary() == "Custom · 2 of 4 on"
    assert "Custom · 2 of 4 on" in chips and any("excluded on mobile" in reason for reason in chips)
    assert tile.warning_text.value == settings_rules.STREAMING_TRUNCATION_WARNING
    assert tile.note_text.value == settings_rules.FORCED_STREAM_NOTE
    assert tile.activate() and all(ctx.store.get(key) is False for key in WRITES)  # one tap: all four (+ thoughts)
    chips = [c.reason for c in tile.badges.controls if isinstance(c, ReasonChip)]
    assert tile.switch.value is False and tile.summary() == "Off" and not any(r.startswith("Custom") for r in chips)
    assert tile.modified_dot.visible
    assert tile.activate() and all(ctx.store.get(key) is True for key in WRITES)
    assert tile.switch.value is True and tile.summary() == "On"
    body = tile.help_body()
    assert "ENABLE_STREAMING" in body and "ALLOW_AUTHGPT_BATCH_STREAM_LOGS" in body and "ENABLE_THOUGHTS" in body
    assert "Default on Glossarion Mobile: On" in body
    # long-press reset: back to the default (keys removed), with Undo
    assert tile.reset() and not any(ctx.store.has(key) for key in FOUR)
    assert ctx.store.get("enable_thoughts") is True  # thoughts on is what the lock wants: kept
    assert tile.value_text.value == "On · default"
    message, action, undo = notes[-1]
    assert action == "Undo"
    undo()
    assert all(ctx.store.get(key) is True for key in FOUR)
    ctx.store._saver.close()


@needs_flet
def test_a_stale_enable_thoughts_off_from_the_old_tile_is_shown_as_run_and_reset_clears_it(tmp_path):
    """U8/U9 builds had their own Enable thoughts tile: a user who turned it off has enable_thoughts False stored
    and no streaming key. Every run uses thoughts ON (the desktop lock under the default-ON stream thinking), so
    the switch's ⓘ says so, and long-press Reset can clear the stale value (review finding mobile-behaviour-9)."""
    from glossarion_mobile.state.setting_writes import with_mobile_streaming_defaults
    from glossarion_mobile.ui.settings.tiles import make_tile

    ctx = _ctx(tmp_path, {"enable_thoughts": False}, notify=lambda *args: None)
    tile = make_tile(ctx.schema.spec("streaming"), ctx)
    assert with_mobile_streaming_defaults(dict(ctx.store.snapshot()))["enable_thoughts"] is True  # what runs
    assert tile.switch.value is True and tile.stored
    thoughts = [line for line in tile.help_body().splitlines() if "ENABLE_THOUGHTS" in line]
    assert thoughts and thoughts[0].endswith("On (locked by stream thinking)"), thoughts
    assert tile.reset() and not ctx.store.has("enable_thoughts") and not tile.stored
    assert [line for line in tile.help_body().splitlines() if "ENABLE_THOUGHTS" in line][0].endswith("On (default)")
    ctx.store._saver.close()


@needs_flet
def test_section_pages_show_one_switch_and_settings_home_counts_it(tmp_path):
    from glossarion_mobile.ui.router import parse_route
    from glossarion_mobile.ui.settings.section_page import SectionPage
    from glossarion_mobile.ui.settings.settings_home import SettingsHome
    from glossarion_mobile.ui.settings.tiles import StreamingTile

    ctx = _ctx(tmp_path)
    page = SectionPage(parse_route("/settings/s/other.response"), ctx)
    page.get_body()
    assert page.keys[0] == "streaming" and page.headings["streaming"] == "Streaming"
    assert not set(WRITES) & set(page.keys)
    tile = page.tile("streaming")
    assert isinstance(tile, StreamingTile) and tile.switch.value is True and tile.value_text.value == "On · default"
    thinking = SectionPage(parse_route("/settings/s/thinking"), ctx)
    thinking.get_body()
    assert "enable_thoughts" not in thinking.keys and thinking.keys  # the Enable thoughts tile is gone
    home = SettingsHome(parse_route("/settings"), ctx)
    section = ctx.schema.section("other.response")
    assert "changed" not in home._section_tile(section).subtitle.value
    tile.activate()  # off
    assert home._section_tile(section).subtitle.value.endswith("· 5 changed")
    ctx.store._saver.close()


# ==========================================================================
# after the Integrate patches (section_page / search / env_preview)
# ==========================================================================


@needs_flet
def test_deep_link_search_and_section_reset_reach_the_switch(tmp_path):
    from glossarion_mobile.ui.router import parse_route
    from glossarion_mobile.ui.settings.search import find_settings
    from glossarion_mobile.ui.settings.section_page import SectionPage

    ctx = _ctx(tmp_path, {**ALL_OFF, "enable_thoughts": False, "max_retries": 9})
    page = SectionPage(parse_route("/settings/s/other.response#enable_streaming"), ctx)
    page.get_body()
    assert page.focus_target == "streaming"  # a desktop key's deep link lands on the switch
    hits = find_settings(ctx, "", ["modified"])
    assert "streaming" in [hit.key for hit in hits]  # the Modified filter sees the switch
    hit = next(h for h in ctx.schema.search("ENABLE_STREAMING") if h.key == "streaming")
    assert ctx.schema.virtual_summary(ctx.store, hit.key) == "Off"
    removed = page.reset_section()
    assert "streaming" in removed and "max_retries" in removed
    assert not any(ctx.store.has(key) for key in WRITES) and ctx.store.snapshot() == {}
    ctx.store._saver.close()


def test_env_preview_builds_with_the_mobile_streaming_default(tmp_path):
    from glossarion_mobile.ui.screens import env_preview

    seen = {}

    def owner_factory(config, *, host, api_key):
        seen["config"] = dict(config)
        return object()

    result = env_preview.build_env_preview({}, input_path=str(tmp_path / "x.epub"), owner_factory=owner_factory,
                                           env_builder=lambda owner, path, key: {"ENABLE_STREAMING": "1"})
    assert result.ok and all(seen["config"][key] is True for key in FOUR)
