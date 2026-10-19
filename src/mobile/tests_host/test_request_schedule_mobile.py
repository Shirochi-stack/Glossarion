"""Host tests: 🗓️ Disable schedule and ⏳ Request Queue on Glossarion Mobile (shared ``request_schedule``).

Run from src/mobile:
    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_request_schedule_mobile.py

* ScheduleSheet (``ui/sheets/schedule_sheet.py``): DeepSeek preset prefilled, window / day edits,
  "Times are in" conversion, validation, Save / Clear, a phone-sized scrolling body.
* API keys: the selection ⋯ "⏰ Schedule…" entry (one schedule per key, "During the windows:" 🗓️ / ⏳;
  no queue for Fallback keys), the card status ("🗓️ Disabled until …" / "⏳ Queued until …") and the key fields
  written through the key service (round trip with the desktop ``APIKeyEntry``).
* The main key: ModelSheet ⋯ "⏳ Request Queue…", the Settings tile, config ``request_queue_schedule``
  exported as ``REQUEST_QUEUE_SCHEDULE`` by the job's HeadlessOwner, pool key schedules reaching the
  runtime pool.
* Jobs: the wait's log lines reach the job log through the mobile client log handlers and become the
  "⏳ Request Queue · until …" issue chip / notification text; every mobile Stop (graceful, immediate,
  force, glossary) ends a waiting request at once.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import sys
import threading
import time
import types
from datetime import datetime, timezone
from pathlib import Path

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]

_MK_SPEC = importlib.util.spec_from_file_location("_glossarion_mk_helpers_schedule",
                                                  Path(__file__).with_name("test_models_keys.py"))
MK = importlib.util.module_from_spec(_MK_SPEC)
_MK_SPEC.loader.exec_module(MK)
_TJ_SPEC = importlib.util.spec_from_file_location("_glossarion_tj_helpers_schedule",
                                                  Path(__file__).with_name("test_jobs.py"))
TJ = importlib.util.module_from_spec(_TJ_SPEC)
_TJ_SPEC.loader.exec_module(TJ)

import request_schedule as rs  # noqa: E402  (src is on sys.path through the helpers)

needs_flet = MK.needs_flet
needs_client = MK.needs_client
needs_run_env = MK.needs_run_env
UTC = timezone.utc
DEEPSEEK = rs.to_config(rs.DEEPSEEK_PEAK_SCHEDULE)
MON = 19  # 2026-10-19 is a Monday


def at(day, hour, minute=0):
    return datetime(2026, 10, day, hour, minute, tzinfo=UTC)


def _find_text(root, predicate):
    """The first Text value under ``root`` (content / controls / title trees) matching ``predicate``."""
    stack = [root]
    while stack:
        control = stack.pop()
        value = getattr(control, "value", None)
        if type(control).__name__ == "Text" and isinstance(value, str) and predicate(value):
            return value
        for name in ("content", "title", "subtitle", "leading", "trailing"):
            child = getattr(control, name, None)
            if child is not None and not isinstance(child, (str, int, float)):
                stack.append(child)
        for name in ("controls",):
            items = getattr(control, name, None)
            if isinstance(items, list):
                stack.extend(items)
    return None


# ==========================================================================
# ScheduleSheet
# ==========================================================================


def test_parse_time_text_accepts_phone_keyboard_forms():
    from glossarion_mobile.ui.sheets.schedule_sheet import parse_time_text

    assert parse_time_text("7:05") == 425 and parse_time_text("0705") == 425 and parse_time_text("705") == 425
    assert parse_time_text("07.05") == 425 and parse_time_text("24:00") == 0
    for bad in ("25:00", "7", "ab:cd", ""):
        with pytest.raises(rs.ScheduleError):
            parse_time_text(bad)


@needs_flet
def test_schedule_sheet_suggests_deepseek_and_saves_edits():
    from glossarion_mobile.ui.sheets.schedule_sheet import ScheduleSheet

    async def scenario():
        _conn, session = MK._fake_session("android")
        page = session.page
        saved = []
        sheet = ScheduleSheet(None, mode="queue", subject="deepseek-chat", on_save=saved.append,
                              now=lambda: at(MON, 3)).show(page)
        assert sheet.sheet.open and sheet.title == "⏳ Request Queue · deepseek-chat"
        # nothing saved: the DeepSeek peak hours are suggested, kept in UTC
        assert sheet.schedule() == (rs.DEEPSEEK_PEAK_SCHEDULE, None) and sheet.zone_dropdown.value == "UTC"
        preview = sheet.preview_text.value
        # off by default: the windows are dimmed until the switch is turned on
        assert not sheet.enabled and sheet.body.disabled and preview.startswith(rs.OFF_PREVIEW_LINE)
        assert "Suggested: DeepSeek peak hours" in preview and sheet.kind is None and sheet.kind_group is None
        assert sheet.set_enabled(True) and not sheet.body.disabled
        # the status line swaps, the line count stays (no layout jump)
        on = sheet.preview_text.value
        assert on.startswith(rs.ON_PREVIEW_LINE) and on.count("\n") == preview.count("\n")
        assert on.split("\n")[1:] == preview.split("\n")[1:]
        assert "• UTC: 01:00–04:00, 06:00–10:00 (UTC), Mon–Fri" in preview and "inside a window until" in preview
        # edit a time (a phone keyboard may drop the colon), add / remove a window, tick Saturday
        sheet.rows[0][1].value = "0030"
        sheet.add_window(20 * 60, 2 * 60)
        assert sheet.schedule()[0].windows[-1] == rs.Window(1200, 120)
        sheet.remove_window(sheet.rows[-1])
        assert sheet.toggle_day(5) is True
        # an invalid time keeps the sheet open with the reason
        sheet.rows[0][2].value = "25:00"
        assert "HH:MM" in sheet.save() and sheet.error.visible and sheet.rows[0][2].error and saved == []
        sheet.rows[0][2].value = "04:00"
        assert sheet.save() is None and not sheet.sheet.open
        assert saved == [{"windows": [{"start": "00:30", "end": "04:00"}, {"start": "06:00", "end": "10:00"}],
                          "weekdays": [0, 1, 2, 3, 4, 5], "timezone": "UTC"}]
        assert sheet.result == saved[0]

    asyncio.run(scenario())


@needs_flet
def test_schedule_sheet_zones_days_clear_and_errors():
    from glossarion_mobile.ui.sheets.schedule_sheet import ScheduleSheet

    async def scenario():
        _conn, session = MK._fake_session("android")
        page = session.page
        saved = []
        sheet = ScheduleSheet(DEEPSEEK, mode="disable", on_save=saved.append, now=lambda: at(MON, 12)).show(page)
        assert sheet.title == "🗓️ Disable schedule" and not sheet.suggested
        assert "Suggested" not in sheet.preview_text.value and "next one starts" in sheet.preview_text.value
        # "Times are in": the same instants rewritten in the chosen zone (the dropdown's handler)
        target = "Asia/Tokyo" if rs.zone_is_known("Asia/Tokyo") else "local"
        sheet._on_zone_select(types.SimpleNamespace(control=types.SimpleNamespace(value=target)))
        expected = rs.convert(rs.DEEPSEEK_PEAK_SCHEDULE, target, at(MON, 12))
        assert sheet.schedule()[0] == expected._replace(timezone=target) and sheet.zone_dropdown.value == target
        if target == "Asia/Tokyo":
            assert sheet.schedule()[0].windows == (rs.Window(600, 780), rs.Window(900, 1140))
        assert sheet.set_zone("UTC") is None and sheet.schedule()[0] == rs.DEEPSEEK_PEAK_SCHEDULE
        # an unknown zone keeps the old one
        assert "not a timezone" in sheet.set_zone("Mars/Olympus_Mons") and sheet.zone == "UTC"
        assert sheet.zone_dropdown.value == "UTC"
        # windows but no days: refused
        for day in sorted(sheet.days()):
            sheet.toggle_day(day)
        assert sheet.save().startswith("Tick at least one day") and saved == []
        # the "Also show in" zone adds a preview line
        if rs.zone_is_known("Asia/Tokyo"):
            sheet.toggle_day(0)
            sheet.preview_zone_dropdown.value = "Asia/Tokyo"
            assert "• Asia/Tokyo, UTC+9:" in sheet.update_preview()
        # 🐋 preset back, then 🧹 Clear: removes it and closes
        sheet.use_deepseek()
        assert sheet.schedule()[0] == rs.DEEPSEEK_PEAK_SCHEDULE
        assert sheet.clear() is None and saved == [{}] and not sheet.sheet.open
        # no windows at all: Save removes the schedule too
        empty = ScheduleSheet(DEEPSEEK, on_save=saved.append).show(page)
        while empty.rows:
            empty.remove_window(0)
        assert empty.preview_text.value.split("\n")[1].startswith("No windows or no days")
        assert empty.save() is None and saved[-1] == {}
        # an on_save error keeps the sheet open
        failing = ScheduleSheet(None, on_save=lambda config: "Disk full").show(page)
        assert failing.save() == "Disk full" and failing.sheet.open and failing.error.value == "Disk full"

    asyncio.run(scenario())


# ==========================================================================
# API keys
# ==========================================================================


def _controller(tmp_path, data, clock, module=None):
    from glossarion_mobile.ui.screens.keys import KeyBackend, KeysController

    store = MK._store(tmp_path, data)
    return store, KeysController(store, KeyBackend(module or MK._fake_key_service()), clock=clock)


def test_schedule_menu_kinds_follow_the_desktop_pools():
    from glossarion_mobile.ui.screens.keys import SCHEDULE_LABEL, key_schedule_info, schedule_kinds

    assert SCHEDULE_LABEL == "⏰ Schedule…"
    assert schedule_kinds("fallback") == ("disable",) == rs.pool_schedule_kinds("fallback")
    for pool in ("main", "glossary", "glossary_refinement", "qa_scan", "metadata", "ai_truncation_detection",
                 "rolling_summary", "truncation_retry", "inpainter", "tts"):
        assert schedule_kinds(pool) == ("disable", "queue") == rs.pool_schedule_kinds(pool)
    key = {"api_key": "k", "model": "deepseek-chat", "queue_schedule": DEEPSEEK}
    status, lines = key_schedule_info(key, at(MON, 2).timestamp())
    assert status == rs.key_schedule_summary(key, now=at(MON, 2))[0] and status.startswith("⏳ Queued until ")
    assert lines and lines[0].startswith("⏳ Request Queue ")
    assert key_schedule_info({"api_key": "k"}) == ("", [])


@needs_flet
def test_schedule_sheet_key_mode_offers_the_pool_behaviours():
    """Mode "key": "During the windows:" 🗓️ / ⏳ (the desktop ⏰ Schedules tab); Fallback has no queue."""
    from glossarion_mobile.ui.sheets.schedule_sheet import ScheduleSheet

    sheet = ScheduleSheet(None, mode="key", subject="2 keys", kind="queue")
    assert sheet.title == "⏰ Schedule · 2 keys" and sheet.kind == "queue"
    assert [radio.label for radio in sheet.kind_group.content.controls] == [rs.KIND_CHOICES["disable"],
                                                                            rs.KIND_CHOICES["queue"]]
    assert not any(radio.disabled for radio in sheet.kind_group.content.controls) and not sheet.kind_note.visible
    assert sheet.kind_group in sheet.body.controls  # dimmed with the rest while the schedule is off
    assert sheet.set_kind("disable") == "disable"
    fallback = ScheduleSheet(None, mode="key", kind="queue", kinds=("disable",))
    assert fallback.kind == "disable" and fallback.set_kind("queue") == "disable"
    radios = {radio.value: radio for radio in fallback.kind_group.content.controls}
    assert radios["queue"].disabled and not radios["disable"].disabled
    assert fallback.kind_note.visible and fallback.kind_note.value == rs.QUEUE_UNAVAILABLE_HELP
    # off by default: Save keeps the windows switched off
    saved = []
    fallback.on_save = saved.append
    assert fallback.save() is None and saved == [rs.to_config(rs.DEEPSEEK_PEAK_SCHEDULE, enabled=False)]


@needs_flet
def test_keys_screen_schedule_entry_status_and_clear(tmp_path):
    from glossarion_mobile.ui.screens.keys import SCHEDULE_LABEL, KeysScreen
    from glossarion_mobile.ui.sheets.schedule_sheet import ScheduleSheet

    async def scenario():
        _conn, session = MK._fake_session("android")
        page = session.page
        now = at(MON, 2)  # inside the DeepSeek peak window (until 04:00 UTC)
        store, keys = _controller(tmp_path, {"use_multi_api_keys": True, "multi_api_keys": [
            {"api_key": "sk-aaaaaaaaaaaa1111", "model": "deepseek-chat", "last_test_result": "passed"},
            {"api_key": "sk-bbbbbbbbbbbb2222", "model": "deepseek-chat"}],
            "fallback_keys": [{"api_key": "sk-cccccccccccc3333", "model": "gpt-6"}]}, clock=now.timestamp)
        notes = []
        screen = KeysScreen(MK.parse_route("/settings/keys/translation"), controller=keys,
                            ctx=MK._settings_ctx(store, page), page=page,
                            notify=lambda m, a=None, cb=None: notes.append(m),
                            spawn=lambda c: asyncio.ensure_future(c))
        page.views[0].controls.append(screen.get_body())
        page.update()
        screen.toggle_select(0)
        screen.toggle_select(1)
        labels = [item.label for item in screen.open_bulk_more().items]
        # one merged entry replaces "🗓️ Disable schedule…" / "⏳ Request Queue…"
        assert labels.count(SCHEDULE_LABEL) == 1 and labels.index(SCHEDULE_LABEL) == labels.index(
            "Set request contexts…") + 1
        assert not any(label in labels for label in ("🗓️ Disable schedule…", "⏳ Request Queue…"))
        # saving without enabling stores "enabled": false and nothing is active
        sheet = screen.open_schedule_sheet()
        assert isinstance(sheet, ScheduleSheet) and sheet.title == "⏰ Schedule · 2 keys" and sheet.suggested
        assert sheet.mode == "key" and sheet.kinds == ("disable", "queue") and not sheet.enabled
        assert sheet.set_kind("queue") == "queue" and sheet.save() is None
        off = rs.to_config(rs.DEEPSEEK_PEAK_SCHEDULE, enabled=False)
        assert [k.get("queue_schedule") for k in keys.keys("main")] == [off, off]
        assert not any(rs.key_queued_now(k, now) for k in keys.keys("main"))
        assert notes[-1] == "Request Queue saved (off until enabled) for 2 key(s)"
        # switched on: ⏳ queue for both keys
        sheet = screen.open_schedule_sheet()
        assert sheet.kind == "queue" and not sheet.enabled and not sheet.suggested
        assert sheet.set_enabled(True) and sheet.save() is None
        assert [k.get("queue_schedule") for k in keys.keys("main")] == [DEEPSEEK, DEEPSEEK]
        assert notes[-1] == "Request Queue saved for 2 key(s)"
        queued = rs.key_schedule_summary({"queue_schedule": DEEPSEEK}, now=now)[0]
        for card in screen.cards:  # "Passed" / "Active" give way to the schedule state, windows listed
            assert _find_text(card, lambda v: v == queued)
            assert _find_text(card, lambda v: v.startswith("⏳ Request Queue "))
        # the first key only, switched to 🗓️ Disable: the queue field is removed (one schedule per key)
        screen.clear_selection()
        screen.toggle_select(0)
        sheet = screen.open_schedule_sheet()
        assert sheet.title == "⏰ Schedule · deepseek-chat" and sheet.kind == "queue" and sheet.enabled
        assert sheet.set_kind("disable") == "disable" and sheet.save() is None
        first, second = keys.keys("main")
        assert first["disable_schedule"] == DEEPSEEK and "queue_schedule" not in first
        assert second["queue_schedule"] == DEEPSEEK and "disable_schedule" not in second
        disabled = rs.key_schedule_summary({"disable_schedule": DEEPSEEK}, now=now)[0]
        assert disabled.startswith("🗓️ Disabled until ") and _find_text(screen.cards[0], lambda v: v == disabled)
        # 🧹 Clear removes it again
        sheet = screen.open_schedule_sheet()
        assert sheet.kind == "disable" and sheet.schedule()[0] == rs.DEEPSEEK_PEAK_SCHEDULE
        assert sheet.clear() is None and not keys.keys("main")[0].get("disable_schedule")
        assert "queue_schedule" not in keys.keys("main")[0]
        assert notes[-1] == "Schedule removed for 1 key(s)"
        # Fallback keys: the same entry, without the queue behaviour (tried one by one)
        screen.clear_selection()
        screen.select_pool("fallback")
        screen.toggle_select(0)
        assert SCHEDULE_LABEL in [item.label for item in screen.open_bulk_more().items]
        sheet = screen.open_schedule_sheet()
        assert sheet.kinds == ("disable",) and sheet.kind == "disable" and sheet.kind_note.visible
        assert screen.apply_schedule([0], "queue", DEEPSEEK) is None  # a forced queue falls back to disable
        assert keys.keys("fallback")[0].get("disable_schedule") == DEEPSEEK
        assert "queue_schedule" not in keys.keys("fallback")[0]
        # stored on disk exactly as the desktop writes it
        on_disk = MK._on_disk(store)
        assert on_disk["multi_api_keys"][1]["queue_schedule"] == DEEPSEEK
        assert "disable_schedule" not in on_disk["multi_api_keys"][1]
        screen.dispose()
        store._saver.close()

    asyncio.run(scenario())


def test_one_schedule_per_key_round_trips_with_the_desktop_tab(tmp_path):
    """The mobile write ({chosen: config, other: {}}) through the real key service equals the desktop
    tab's ``request_schedule.set_key_schedule`` on the same key, switched-off configs included."""
    import key_pool_service

    from multi_api_key_manager import APIKeyEntry

    off = rs.to_config(rs.DEEPSEEK_PEAK_SCHEDULE, enabled=False)
    start = {"api_key": "sk-aaaa", "model": "deepseek-chat", "queue_schedule": DEEPSEEK}
    store, keys = _controller(tmp_path, {"multi_api_keys": [dict(start)]}, clock=time.time, module=key_pool_service)
    changed, errors = keys.set_key_fields("main", [0], {"disable_schedule": off, "queue_schedule": {}})
    assert (changed, errors) == (1, [])
    mobile = keys.keys("main")[0]
    desktop = rs.set_key_schedule(APIKeyEntry.from_dict(start), "disable", off).to_dict()
    assert mobile["disable_schedule"] == desktop["disable_schedule"] == off
    assert "queue_schedule" not in mobile and "queue_schedule" not in desktop
    assert rs.key_schedule_config(mobile) == rs.key_schedule_config(APIKeyEntry.from_dict(mobile)) == ("disable", off)
    assert not rs.key_disabled_by_schedule(mobile, at(MON, 2))  # off: never active
    store._saver.close()


def test_real_key_service_normalizes_schedules_and_round_trips_with_the_desktop(tmp_path):
    from glossarion_mobile.ui.screens.keys import KeyBackend

    import key_pool_service

    from multi_api_key_manager import APIKeyEntry

    store, keys = _controller(tmp_path, {"multi_api_keys": [{"api_key": "sk-aaaa", "model": "deepseek-chat"}]},
                              clock=time.time, module=key_pool_service)
    assert keys.backend.available and isinstance(keys.backend, KeyBackend)
    loose = {"windows": [{"start": "1:00", "end": "4:00"}], "weekdays": ["Mon", "Tue"], "timezone": "UTC"}
    changed, errors = keys.set_key_fields("main", [0], {"queue_schedule": loose})
    assert (changed, errors) == (1, [])
    stored = keys.keys("main")[0]["queue_schedule"]
    assert stored == rs.normalize_config(loose) == {"windows": [{"start": "01:00", "end": "04:00"}],
                                                    "weekdays": [0, 1], "timezone": "UTC"}
    # the desktop key object reads and writes the same dict
    assert APIKeyEntry.from_dict(keys.keys("main")[0]).to_dict()["queue_schedule"] == stored
    # an invalid schedule is refused with the core's reason; {} removes the field
    changed, errors = keys.set_key_fields("main", [0], {"disable_schedule": {"windows": [{"start": "99:00",
                                                                                          "end": "01:00"}]}})
    assert changed == 0 and errors and "Disable schedule" in errors[0]
    assert keys.set_key_fields("main", [0], {"queue_schedule": {}}) == (1, [])
    assert "queue_schedule" not in keys.keys("main")[0]
    store._saver.close()


# ==========================================================================
# the main key: ModelSheet, Settings tile, run environment
# ==========================================================================


@needs_flet
def test_model_sheet_request_queue_entry_saves_the_main_key_schedule(tmp_path):
    from glossarion_mobile.ui.sheets.model_sheet import ModelSheet, SheetEnv
    from glossarion_mobile.ui.sheets.schedule_sheet import ScheduleSheet, queue_summary

    async def scenario():
        _conn, session = MK._fake_session("android")
        page = session.page
        store = MK._store(tmp_path, {"model": "deepseek-chat"})
        notes = []
        env = SheetEnv(store=store, notify=notes.append)
        sheet = ModelSheet(current_model="deepseek-chat", models=["deepseek-chat"], env=env)
        sheet.show(page)
        assert sheet.queue_item is not None and sheet.queue_item in sheet.menu.items
        assert sheet.queue_item.content == "⏳ Request Queue…" and not sheet.queue_item.checked
        assert not sheet.queue_row.visible
        queue = sheet.open_request_queue()
        assert isinstance(queue, ScheduleSheet) and queue.mode == "main_queue" and queue.suggested
        assert queue.title == "⏳ Request Queue · deepseek-chat"
        assert not queue.enabled and queue.set_enabled(True)
        assert queue.save() is None
        assert store.get("request_queue_schedule") == DEEPSEEK
        assert MK._on_disk(store)["request_queue_schedule"] == DEEPSEEK
        assert sheet.queue_row.visible and sheet.queue_item.checked
        assert sheet.queue_text.value == queue_summary(DEEPSEEK)  # windows in local time, like the desktop
        assert sheet.queue_text.value.startswith("⏳ Request Queue · ") and "Mon–Fri" in sheet.queue_text.value
        assert notes[-1] == "⏳ Request Queue saved for the main key"
        # opened again on the saved schedule; 🧹 Clear turns it off ({} like the desktop)
        queue = sheet.open_request_queue()
        assert not queue.suggested and queue.clear() is None
        assert store.get("request_queue_schedule") == {} and not sheet.queue_row.visible
        assert not sheet.queue_item.checked
        # the field / one-shot sheets pick a model only
        for picker in (ModelSheet(current_model="x", env=env, field_mode=True),
                       ModelSheet(current_model="x", env=env, one_shot=True)):
            assert picker.queue_item is None and picker.open_request_queue() is None
        store._saver.close()

    asyncio.run(scenario())


@needs_flet
def test_settings_tile_opens_the_schedule_sheet(tmp_path):
    from glossarion_mobile.ui.settings.tiles import RequestQueueTile, make_tile
    from glossarion_mobile.ui.sheets.schedule_sheet import ScheduleSheet, queue_summary

    async def scenario():
        _conn, session = MK._fake_session("android")
        page = session.page
        store = MK._store(tmp_path, {"model": "deepseek-chat"})
        ctx = MK._settings_ctx(store, page)
        tile = make_tile(ctx.schema.spec("request_queue_schedule"), ctx)
        assert isinstance(tile, RequestQueueTile) and tile.editable
        assert tile.summary() == "Off: requests are sent at once"
        editor = tile.activate()
        assert isinstance(editor, ScheduleSheet) and editor.mode == "main_queue" and editor.sheet.open
        assert not editor.enabled and editor.set_enabled(True)
        assert editor.save() is None and store.get("request_queue_schedule") == DEEPSEEK
        assert tile.summary() == queue_summary(DEEPSEEK) and tile.summary().startswith("⏳ Request Queue · ")
        store._saver.close()

    asyncio.run(scenario())


@needs_run_env
def test_job_owner_exports_the_main_queue_and_pool_schedules_reach_the_runtime_pool(tmp_path, monkeypatch):
    headless_owner = pytest.importorskip("headless_owner")
    run_env = pytest.importorskip("run_env")
    glossary_paths = pytest.importorskip("glossary_paths")
    import job_runner
    import key_pools
    from glossarion_mobile.state.setting_writes import with_mobile_streaming_defaults

    monkeypatch.setattr(glossary_paths, "migrate_all_legacy_glossary_files", lambda *a, **k: None)
    for name, sub in (("HOME", "home"), ("USERPROFILE", "home"), ("APPDATA", "appdata"),
                      ("LOCALAPPDATA", "localappdata"), ("GLOSSARION_LIBRARY_DIR", "lib"),
                      ("OUTPUT_DIRECTORY", "out"), ("GLOSSARION_DATA_DIR", "data")):
        folder = tmp_path / "_iso" / sub
        folder.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv(name, str(folder))
    config = {"model": "deepseek-chat", "request_queue_schedule": DEEPSEEK, "use_multi_api_keys": True,
              "multi_api_keys": [{"api_key": "sk-a", "model": "deepseek-chat", "queue_schedule": DEEPSEEK},
                                 {"api_key": "sk-b", "model": "deepseek-chat", "disable_schedule": DEEPSEEK}]}
    with job_runner.scoped_process_state(keywise=True, isolate_key_pools=True):
        owner = headless_owner.HeadlessOwner(with_mobile_streaming_defaults(json.loads(json.dumps(config))))
        env = run_env.build_translation_env(owner, str(tmp_path / "in" / "Book.epub"), "")
        assert json.loads(env["REQUEST_QUEUE_SCHEDULE"]) == DEEPSEEK
        assert rs.main_queue_schedule_from_env(env) == rs.DEEPSEEK_PEAK_SCHEDULE
        key_pools.apply_key_pools_to_runtime(config)
        from unified_api_client import UnifiedClient

        pool = UnifiedClient._api_key_pool
        assert pool is not None
        assert [k.queue_schedule for k in pool.keys] == [DEEPSEEK, {}]
        assert [k.disable_schedule for k in pool.keys] == [{}, DEEPSEEK]
        # 02:00 UTC Monday: key b is disabled by its schedule, key a is queued -> the pool waits until 04:00
        assert rs.pool_queue_wait_until(pool.keys, at(MON, 2)) == at(MON, 4)
    assert os.environ.get("REQUEST_QUEUE_SCHEDULE") != json.dumps(DEEPSEEK)  # nothing leaks out of the job scope


# ==========================================================================
# jobs: log lines, issue chip, notification, Stop
# ==========================================================================


def test_request_queue_lines_become_the_issue_not_a_network_or_rate_limit_wait():
    from glossarion_mobile.services import jobs

    start = "⏳ Request Queue: waiting 4h 59m — requests start at 13:00 (local, UTC+4) / 09:00 UTC"
    progress = "⏳ Request Queue (glossary): waited 0h 03m, 4h 56m left (starts 13:00 Tue local)"
    assert jobs.request_queue_issue(start) == (4 * 3600 + 59 * 60, "13:00")
    assert jobs.request_queue_issue(progress) == (4 * 3600 + 56 * 60, "13:00 Tue")
    assert jobs.request_queue_issue("Rate limited, waiting 30s") is None
    assert jobs.classify_issue(start) is None and jobs.classify_issue(progress) is None  # never "network wait"
    assert jobs.issue_label({"kind": "request_queue", "starts": "13:00"}) == "⏳ Request Queue · until 13:00"


def test_job_waiting_for_the_queue_shows_it_and_a_stop_clears_it(tmp_path):
    lines = [
        "⏳ Request Queue: waiting 1h 00m — requests start at 13:00 (local, UTC+4) / 09:00 UTC",
        "⏳ Request Queue: waited 0h 03m, 0h 57m left (starts 13:00 local)",
    ]
    gate = TJ.Gate()

    def worker(owner, request):
        for line in lines:
            owner.host.log(line)
        return gate(owner, request)

    service, backend = TJ.make_service(tmp_path)
    backend.behavior = worker
    job_id = service.submit(TJ.JobSpec("translate", "Book.epub", (TJ.epub(tmp_path),)))
    assert gate.entered.wait(TJ.TIMEOUT)
    assert TJ.wait_for(lambda: service.snapshot(job_id).active_issue() is not None)
    snap = service.snapshot(job_id)
    issue = snap.active_issue()
    assert issue["kind"] == "request_queue" and issue["starts"] == "13:00" and issue["seconds"] == 57 * 60
    assert issue["until"] - issue["since"] == pytest.approx(57 * 60)  # labelled for the whole wait, not 45 s
    assert "⏳ waiting until 13:00" in TJ.notification_text(snap)
    assert "⏳ waiting until 13:00" in TJ.progress_line(snap)
    # the job log keeps both lines
    assert TJ.wait_for(lambda: all(line in Path(service.snapshot(job_id).log_path).read_text(encoding="utf-8")
                                   for line in lines))
    service.request_stop(job_id)
    assert service.snapshot(job_id).active_issue() is None  # Stop ended the wait: no stale chip
    gate.release.set()
    assert TJ.wait_for(lambda: service.snapshot(job_id).is_terminal)
    service.close()


@needs_client
@pytest.mark.parametrize("stop", ["graceful", "immediate", "force", "glossary"])
def test_every_mobile_stop_ends_a_request_queue_wait_at_once(tmp_path, monkeypatch, stop):
    """The real mobile Stop (``JobBackend.request_stop`` -> stop_control) against a request waiting for a
    whole-day window; the wait's lines reach the job through the mobile client log handlers."""
    import job_runner
    import stop_control
    import unified_api_client as api
    from glossarion_mobile.services.jobs import JobBackend, client_log_handlers
    from unified_api_client import UnifiedClientError

    monkeypatch.setenv("GLOSSARION_HTTP_LOG", "0")
    monkeypatch.setattr(rs, "WAITER", rs.QueueWaiter(poll=0.02, log_interval=0.05))
    whole_day = {"windows": [{"start": "00:00", "end": "00:00"}], "weekdays": list(range(7)), "timezone": "UTC"}

    class Host:
        def __init__(self):
            self.lines = []

        def append_log(self, message):
            self.lines.append(str(message))

    host = Host()
    outcome = {}
    with job_runner.scoped_process_state(keywise=True, isolate_key_pools=True):
        stop_control.reset_for_new_run("glossary" if stop == "glossary" else "translation")
        os.environ[rs.MAIN_QUEUE_ENV] = json.dumps(whole_day)
        if stop == "glossary":
            stop_control.prepare_glossary_stop_file()
        client = api.UnifiedClient(api_key="sk-test", model="deepseek-chat", output_dir=str(tmp_path / "out"))

        def run():
            try:
                client._wait_for_request_queue()
                outcome["result"] = "sent"
            except UnifiedClientError as exc:
                outcome["result"] = getattr(exc, "error_type", None)
            outcome["at"] = time.monotonic()

        with client_log_handlers(host):
            thread = threading.Thread(target=run, name="rq-worker", daemon=True)
            thread.start()
            assert TJ.wait_for(lambda: rs.is_thread_waiting(thread))
            assert TJ.wait_for(lambda: any("waited" in line for line in host.lines))  # progress lines keep coming
            assert rs.is_thread_waiting(thread)  # still waiting: nothing flags or ends it but a Stop
            pressed = time.monotonic()
            JobBackend().request_stop(graceful=stop in ("graceful", "glossary"), wait_for_chunks=False,
                                      force=stop == "force", set_stop_requested=lambda: None,
                                      log=host.append_log, kind="glossary" if stop == "glossary" else "translation")
            thread.join(5.0)
        assert not thread.is_alive() and outcome["result"] == "cancelled"
        assert outcome["at"] - pressed < 2.0
        assert host.lines[0].startswith("⏳ Request Queue: waiting ") and " UTC" in host.lines[0]
        stop_control.reset_for_new_run("translation")
    api.global_stop_flag = False
