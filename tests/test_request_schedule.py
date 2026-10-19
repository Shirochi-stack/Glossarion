"""Time-based API key scheduling (src/request_schedule.py): 🗓️ Disable schedule and ⏳ Request Queue.

Covers the schedule core (windows past midnight, weekdays, DST, back-to-back windows, empty
schedules, display), the key pool rules (a Disable schedule skips the key like the manual flag;
queued keys yield to usable keys outside their windows; with every usable key queued the request
waits for the earliest window end), the shared waiter (start line, a progress line every
3 minutes, one line per interval for many threads), the UnifiedClient send path (waits before the
request is sent, graceful and forced stops end the wait) and the wrapper chunk timeouts, which
never count the wait.

Run (repository root)::

    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests/test_request_schedule.py
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
os.environ.setdefault("GLOSSARION_HTTP_LOG", "0")

import request_schedule as rs  # noqa: E402
from request_schedule import DEEPSEEK_PEAK_SCHEDULE, Schedule, Window  # noqa: E402

UTC = timezone.utc


def at(year, month, day, hour, minute=0, tz=UTC):
    return datetime(year, month, day, hour, minute, tzinfo=tz)


# 2026-10-19 is a Monday
MON, FRI, SAT, SUN = 19, 23, 24, 25


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ("GRACEFUL_STOP", "GRACEFUL_STOP_COMPLETED", "TRANSLATION_CANCELLED", rs.MAIN_QUEUE_ENV,
                 "GLOSSARY_STOP_FILE", "BATCH_TRANSLATION", "RETRY_TIMEOUT", "CHUNK_TIMEOUT"):
        monkeypatch.delenv(name, raising=False)


class FakeClock:
    """Epoch clock whose sleep() advances time; *on_sleep(clock)* runs after every sleep."""

    def __init__(self, start, on_sleep=None):
        self.t = start.timestamp() if isinstance(start, datetime) else float(start)
        self.on_sleep = on_sleep
        self.sleeps = 0

    def time(self):
        return self.t

    def sleep(self, seconds):
        self.t += seconds
        self.sleeps += 1
        if self.on_sleep:
            self.on_sleep(self)


def fake_waiter(monkeypatch, start, on_sleep=None, poll=30.0):
    clock = FakeClock(start, on_sleep)
    waiter = rs.QueueWaiter(clock=clock.time, sleep=clock.sleep, poll=poll)
    monkeypatch.setattr(rs, "WAITER", waiter)
    return clock, waiter


# ------------------------------------------------------------------------------- config / parsing
def test_hhmm_parse_and_format():
    assert rs.parse_hhmm("7:05") == 425
    assert rs.parse_hhmm("00:00") == 0 and rs.parse_hhmm("24:00") == 0 and rs.parse_hhmm("23:59") == 1439
    assert rs.format_hhmm(425) == "07:05"
    for bad in ("25:00", "12:60", "noon", "", "1:2:3", True):
        with pytest.raises(rs.ScheduleError):
            rs.parse_hhmm(bad)


def test_config_round_trip_and_validation():
    config = rs.to_config(DEEPSEEK_PEAK_SCHEDULE)
    assert config == {"windows": [{"start": "01:00", "end": "04:00"}, {"start": "06:00", "end": "10:00"}],
                      "weekdays": [0, 1, 2, 3, 4], "timezone": "UTC"}
    assert rs.from_config(config) == DEEPSEEK_PEAK_SCHEDULE
    assert rs.from_config(json.dumps(config)) == DEEPSEEK_PEAK_SCHEDULE      # env JSON
    assert rs.from_config({"windows": [["22:00", "02:00"]], "weekdays": ["Fri", "sat"]}) == Schedule(
        (Window(1320, 120),), frozenset({4, 5}), "local")
    # weekdays omitted = every day
    assert rs.from_config({"windows": [{"start": "1:00", "end": "2:00"}]}).weekdays == rs.ALL_WEEKDAYS
    # empty / disabled / no windows / no days = no schedule
    for off in (None, {}, "", [], {"windows": []}, {"windows": [["1:00", "2:00"]], "weekdays": []},
                dict(config, enabled=False)):
        assert rs.from_config(off) is None and rs.to_config(off) == {}
    # invalid values: lenient None, strict error, validate() message
    bad = {"windows": [{"start": "25:00", "end": "02:00"}]}
    assert rs.from_config(bad) is None and rs.validate(bad)
    with pytest.raises(rs.ScheduleError):
        rs.from_config(bad, strict=True)
    assert rs.validate({"windows": [["1:00", "2:00"]], "timezone": "Mars/Olympus"}) == [
        "unknown timezone 'Mars/Olympus'"]
    assert rs.validate(config) == [] and rs.validate({}) == []
    assert rs.normalize_config("garbage") == {}


def test_unknown_zone_falls_back_to_the_system_local_zone():
    assert rs.resolve_tz("Not/AZone") is rs.SYSTEM_LOCAL
    assert rs.resolve_tz("local") is rs.SYSTEM_LOCAL and rs.resolve_tz("UTC") is UTC
    assert not rs.zone_is_known("Not/AZone") and rs.zone_is_known("Asia/Tokyo")


# ------------------------------------------------------------------------------------ time checks
def test_deepseek_peak_boundaries_back_to_back_minute():
    s = DEEPSEEK_PEAK_SCHEDULE
    assert not rs.is_in_window(s, at(2026, 10, MON, 0, 59))
    assert rs.is_in_window(s, at(2026, 10, MON, 1, 0))
    assert rs.is_in_window(s, at(2026, 10, MON, 3, 59))           # 03:59 in the window
    assert not rs.is_in_window(s, at(2026, 10, MON, 4, 0))        # 04:00: the window has ended
    assert rs.next_window_end(s, at(2026, 10, MON, 3, 59)) == at(2026, 10, MON, 4, 0)
    assert rs.next_window_end(s, at(2026, 10, MON, 4, 0)) is None
    assert rs.next_window_start(s, at(2026, 10, MON, 4, 0)) == at(2026, 10, MON, 6, 0)
    assert rs.current_window(s, at(2026, 10, MON, 7, 30)) == (at(2026, 10, MON, 6), at(2026, 10, MON, 10))
    assert not rs.is_in_window(s, at(2026, 10, MON, 10, 0))


def test_friday_to_weekend_and_next_monday():
    s = DEEPSEEK_PEAK_SCHEDULE
    assert rs.is_in_window(s, at(2026, 10, FRI, 9, 59))
    assert not rs.is_in_window(s, at(2026, 10, SAT, 2, 0))        # weekends are off-peak
    assert not rs.is_in_window(s, at(2026, 10, SUN, 7, 0))
    assert rs.next_window_start(s, at(2026, 10, FRI, 10, 0)) == at(2026, 10, 26, 1, 0)


def test_window_past_midnight_belongs_to_its_start_day():
    tokyo = rs.resolve_tz("Asia/Tokyo")
    s = rs.from_config({"windows": [["22:00", "02:00"]], "weekdays": ["Fri"], "timezone": "Asia/Tokyo"})
    assert not rs.is_in_window(s, at(2026, 10, FRI, 21, 59, tokyo))
    assert rs.is_in_window(s, at(2026, 10, FRI, 23, 0, tokyo))
    assert rs.is_in_window(s, at(2026, 10, SAT, 1, 30, tokyo))    # Friday's window, after midnight
    assert not rs.is_in_window(s, at(2026, 10, SAT, 2, 0, tokyo))
    assert not rs.is_in_window(s, at(2026, 10, SAT, 23, 0, tokyo))  # Saturday has no window
    assert rs.next_window_end(s, at(2026, 10, FRI, 23, 0, tokyo)) == at(2026, 10, SAT, 2, 0, tokyo)
    # a Sunday night window does not leak into Monday when Sunday is off
    weekdays_only = rs.from_config({"windows": [["22:00", "02:00"]], "weekdays": [0, 1, 2, 3, 4],
                                    "timezone": "UTC"})
    assert not rs.is_in_window(weekdays_only, at(2026, 10, 26, 1, 0))      # Monday 01:00 (Sunday's)
    assert rs.is_in_window(weekdays_only, at(2026, 10, 24, 1, 0))          # Saturday 01:00 (Friday's)


def test_back_to_back_windows_chain_to_the_last_end():
    s = rs.from_config({"windows": [["01:00", "04:00"], ["04:00", "06:00"]], "timezone": "UTC"})
    assert rs.is_in_window(s, at(2026, 10, MON, 4, 0))
    assert rs.next_window_end(s, at(2026, 10, MON, 3, 59)) == at(2026, 10, MON, 6, 0)
    # a window into midnight followed by one from midnight on the next day
    night = rs.from_config({"windows": [["20:00", "00:00"], ["00:00", "03:00"]], "timezone": "UTC"})
    assert rs.next_window_end(night, at(2026, 10, MON, 21, 0)) == at(2026, 10, 20, 3, 0)


def test_start_equal_end_is_a_whole_day():
    s = rs.from_config({"windows": [["09:00", "09:00"]], "weekdays": ["Mon"], "timezone": "UTC"})
    assert rs.is_in_window(s, at(2026, 10, MON, 9, 0)) and rs.is_in_window(s, at(2026, 10, 20, 8, 59))
    assert rs.next_window_end(s, at(2026, 10, MON, 12, 0)) == at(2026, 10, 20, 9, 0)
    always = rs.from_config({"windows": [["00:00", "00:00"]], "timezone": "UTC"})
    end = rs.next_window_end(always, at(2026, 10, MON, 12, 0))
    assert end is not None and end > at(2026, 10, MON, 12, 0) + timedelta(days=7)   # bounded, never None


def test_dst_transitions_in_a_non_utc_zone():
    ny = "America/New_York"
    # spring forward 2026-03-08 02:00 -> 03:00: a 01:00-03:00 window lasts one real hour
    spring = rs.from_config({"windows": [["01:00", "03:00"]], "weekdays": ["Sun"], "timezone": ny})
    assert rs.is_in_window(spring, at(2026, 3, 8, 6, 30))                    # 01:30 EST
    assert rs.next_window_end(spring, at(2026, 3, 8, 6, 30)) == at(2026, 3, 8, 7, 0)   # 03:00 EDT
    assert not rs.is_in_window(spring, at(2026, 3, 8, 7, 0))
    # fall back 2026-11-01: a 09:00-17:00 weekday window follows the wall clock
    office = rs.from_config({"windows": [["09:00", "17:00"]], "weekdays": [0, 1, 2, 3, 4], "timezone": ny})
    assert rs.is_in_window(office, at(2026, 10, 30, 13, 30))                 # Fri 09:30 EDT
    assert not rs.is_in_window(office, at(2026, 11, 2, 13, 30))              # Mon 08:30 EST
    assert rs.is_in_window(office, at(2026, 11, 2, 14, 0))                   # Mon 09:00 EST
    assert rs.next_window_end(office, at(2026, 11, 2, 14, 0)) == at(2026, 11, 2, 22, 0)
    # the UTC-defined DeepSeek schedule does not move with the user's DST
    assert rs.describe(DEEPSEEK_PEAK_SCHEDULE, ny, at(2026, 10, MON, 12)) != rs.describe(
        DEEPSEEK_PEAK_SCHEDULE, ny, at(2026, 11, 2, 12))


def test_empty_or_disabled_schedule_is_never_active():
    for off in (None, {}, {"windows": []}, rs.Schedule((), frozenset(), "UTC")):
        assert not rs.is_in_window(off, at(2026, 10, MON, 2))
        assert rs.next_window_end(off, at(2026, 10, MON, 2)) is None
        assert rs.next_window_start(off) is None
        assert rs.describe(off) == "No schedule"


def test_describe_and_convert_between_zones():
    now = at(2026, 10, MON, 12)
    assert rs.describe(DEEPSEEK_PEAK_SCHEDULE, "UTC", now) == "01:00–04:00, 06:00–10:00 (UTC), Mon–Fri"
    assert rs.describe(DEEPSEEK_PEAK_SCHEDULE, "Asia/Tokyo", now) == \
        "10:00–13:00, 15:00–19:00 (Asia/Tokyo), Mon–Fri"
    # in UTC-3 the first window starts the evening before: two groups with their own days
    assert rs.describe(DEEPSEEK_PEAK_SCHEDULE, "America/Sao_Paulo", now) == \
        "22:00–01:00 (America/Sao_Paulo), Sun–Thu; 03:00–07:00 (America/Sao_Paulo), Mon–Fri"
    tokyo = rs.convert(DEEPSEEK_PEAK_SCHEDULE, "Asia/Tokyo", now)
    assert tokyo == Schedule((Window(600, 780), Window(900, 1140)), rs.WEEKDAYS, "Asia/Tokyo")
    with pytest.raises(rs.ScheduleError):
        rs.convert(DEEPSEEK_PEAK_SCHEDULE, "America/Sao_Paulo", now)
    assert rs.zone_label("UTC") == "UTC" and rs.zone_label("Asia/Kolkata", now) == "Asia/Kolkata, UTC+5:30"
    assert rs.format_wait(4 * 3600 + 59 * 60 + 30) == "4h 59m"
    assert rs.format_wait(61, round_up=True) == "0h 02m"
    assert rs.format_clock(at(2026, 10, MON, 15), "UTC", at(2026, 10, MON, 10)) == "15:00"
    assert rs.format_clock(at(2026, 10, 20, 1), "UTC", at(2026, 10, MON, 10)) == "01:00 Tue"


# ------------------------------------------------------------------------------------------- keys
def _window_around_now(offset_hours=0.0, length_hours=1.0):
    """A UTC every-day schedule whose window starts *offset_hours* from the real current time."""
    now = datetime.now(UTC)
    start = now + timedelta(hours=offset_hours)
    start = start.replace(second=0, microsecond=0)
    end = start + timedelta(hours=length_hours)
    return {"windows": [{"start": start.strftime("%H:%M"), "end": end.strftime("%H:%M")}],
            "weekdays": list(range(7)), "timezone": "UTC"}


def test_editor_helpers_shared_by_desktop_and_mobile():
    """EDITOR_MODES / zone_choices / preview_lines: the desktop dialog and the mobile ScheduleSheet show these."""
    assert set(rs.EDITOR_MODES) == {"disable", "queue", "main_queue", "key"}
    assert set(rs.ENABLE_LABELS) == set(rs.EDITOR_MODES)
    assert rs.EDITOR_MODES["disable"][0] == "🗓️ Disable schedule" and rs.EDITOR_MODES["queue"][0] == "⏳ Request Queue"
    choices = rs.zone_choices()
    assert choices[0][1] == "local" and choices[0][0].startswith("Local (local") and choices[1] == ("UTC", "UTC")
    assert all(rs.zone_is_known(zone) for _label, zone in choices)
    assert rs.zone_choices(include_local=False)[0] == ("UTC", "UTC")
    lines = rs.preview_lines(DEEPSEEK_PEAK_SCHEDULE, "UTC", suggested=True, now=at(2026, 10, MON, 2))
    assert lines[0].startswith("Suggested") and "• UTC: 01:00–04:00, 06:00–10:00 (UTC), Mon–Fri" in lines
    assert lines[-1].startswith("Now: inside a window until ") and len(lines) == 4   # local, UTC (no third zone)
    lines = rs.preview_lines(rs.to_config(DEEPSEEK_PEAK_SCHEDULE), now=at(2026, 10, SAT, 12))
    assert lines[-1].startswith("Now: outside the windows; the next one starts ")
    if rs.zone_is_known("Asia/Tokyo"):
        assert any(line.startswith("• Asia/Tokyo, UTC+9: 10:00–13:00, 15:00–19:00")
                   for line in rs.preview_lines(DEEPSEEK_PEAK_SCHEDULE, "Asia/Tokyo"))
    empty = ["No windows or no days: saving removes the schedule."]
    assert rs.preview_lines(None) == empty and rs.preview_lines(Schedule((), rs.WEEKDAYS, "UTC")) == empty


ACTIVE = _window_around_now(-0.5, 2.0)      # contains the real now (re-checked by the tests)
INACTIVE = _window_around_now(5.0, 1.0)     # starts in five hours


def test_disable_schedule_counts_as_disabled_without_touching_enabled():
    from key_contexts import key_enabled_for_context
    assert rs.is_in_window(ACTIVE) and not rs.is_in_window(INACTIVE)
    key = {"api_key": "a", "model": "m", "enabled": True, "disable_schedule": ACTIVE}
    assert not key_enabled_for_context(key, "translation")
    assert key_enabled_for_context(key, "translation", include_schedule=False)   # stored settings
    assert key["enabled"] is True
    assert key_enabled_for_context(dict(key, disable_schedule=INACTIVE), "translation")
    status, lines = rs.key_schedule_summary(key)
    assert status.startswith("🗓️ Disabled until") and lines[0].startswith("🗓️ Disabled ")


def test_pool_wait_target_prefers_keys_outside_their_windows():
    now = at(2026, 10, MON, 2, 0)                         # inside DeepSeek peak (until 04:00)
    peak = rs.to_config(DEEPSEEK_PEAK_SCHEDULE)
    late = {"windows": [["01:00", "05:00"]], "timezone": "UTC"}
    queued_a = {"api_key": "a", "queue_schedule": peak}
    queued_b = {"api_key": "b", "queue_schedule": late}
    free = {"api_key": "c"}
    assert rs.pool_queue_wait_until([queued_a, free], now) is None            # use the free key
    assert rs.pool_queue_wait_until([queued_a, queued_b], now) == at(2026, 10, MON, 4)   # earliest end
    assert rs.pool_queue_wait_until([queued_b, dict(free, enabled=False)], now) == at(2026, 10, MON, 5)
    assert rs.pool_queue_wait_until([free], now) is None
    assert rs.pool_queue_wait_until([], now) is None
    assert rs.pool_queue_wait_until([queued_a], at(2026, 10, MON, 5)) is None  # window over
    # deferral: a queued key yields only to a usable peer outside its window
    assert rs.queue_deferred(queued_a, [queued_a, free], now)
    assert not rs.queue_deferred(queued_a, [queued_a, queued_b], now)
    assert not rs.queue_deferred(queued_a, [queued_a, dict(free, enabled=False)], now)


def _pool(entries):
    from multi_api_key_manager import APIKeyPool
    pool = APIKeyPool("test pool")
    pool.load_from_list(entries)
    return pool


def test_key_pool_selection_skips_scheduled_keys():
    base = {"model": "deepseek-chat", "cooldown": 60, "enabled": True}
    pool = _pool([dict(base, api_key="queued", queue_schedule=ACTIVE), dict(base, api_key="free")])
    picks = {pool.get_key_for_request(force_rotation=True, rotation_frequency=1)[0].api_key for _ in range(6)}
    assert picks == {"free"}
    assert pool.get_key_for_thread(force_rotation=True)[0].api_key == "free"
    # every usable key queued: still selectable (the client waits before sending)
    pool = _pool([dict(base, api_key="q1", queue_schedule=ACTIVE), dict(base, api_key="q2", queue_schedule=ACTIVE),
                  dict(base, api_key="off", enabled=False)])
    assert pool.get_key_for_request()[0].api_key in {"q1", "q2"}
    assert rs.pool_queue_wait_until(pool.keys) is not None
    # 🗓️ Disable schedule: skipped like a manually disabled key, enabled flag untouched
    pool = _pool([dict(base, api_key="sched", disable_schedule=ACTIVE), dict(base, api_key="ok")])
    assert {pool.get_key_for_request()[0].api_key for _ in range(4)} == {"ok"}
    assert pool.keys[0].enabled is True and not pool.keys[0].is_available()
    # outside the window the key is used again
    pool = _pool([dict(base, api_key="sched", disable_schedule=INACTIVE)])
    assert pool.get_key_for_request()[0].api_key == "sched"


def test_schedules_persist_only_when_set():
    from multi_api_key_manager import APIKeyEntry
    plain = APIKeyEntry("k", "m").to_dict()
    assert "disable_schedule" not in plain and "queue_schedule" not in plain
    entry = APIKeyEntry.from_dict({"api_key": "k", "model": "m", "queue_schedule": rs.to_config(DEEPSEEK_PEAK_SCHEDULE)})
    assert entry.to_dict()["queue_schedule"] == rs.to_config(DEEPSEEK_PEAK_SCHEDULE)
    import key_pool_service
    data, error = key_pool_service.validate_entry(
        {"api_key": "k", "model": "m", "disable_schedule": {"windows": [["1:00", "2:00"]], "timezone": "UTC"},
         "queue_schedule": {}}, pool="glossary")
    assert error is None and data["disable_schedule"]["windows"] == [{"start": "01:00", "end": "02:00"}]
    assert "queue_schedule" not in data
    _, error = key_pool_service.validate_entry(
        {"api_key": "k", "model": "m", "queue_schedule": {"windows": [["31:00", "2:00"]]}}, pool="main")
    assert error and error.startswith("Queue schedule:")


# ----------------------------------------------------------------------------------------- waiter
def test_waiter_logs_start_and_every_three_minutes(monkeypatch):
    start = at(2026, 10, MON, 3, 50)
    clock, waiter = fake_waiter(monkeypatch, start, poll=10.0)
    logs = []
    target = rs.next_window_end(DEEPSEEK_PEAK_SCHEDULE, start)          # 04:00 UTC
    assert waiter.wait_until(target, log=logs.append) is True
    assert clock.t == target.timestamp()
    assert logs[0].startswith("⏳ Request Queue: waiting 0h 10m — requests start at ")
    assert logs[0].endswith(" / 04:00 UTC") and "(local, UTC" in logs[0]
    progress = logs[1:]
    assert [line.split(",")[0] for line in progress] == [
        "⏳ Request Queue: waited 0h 03m", "⏳ Request Queue: waited 0h 06m", "⏳ Request Queue: waited 0h 09m"]
    assert "0h 07m left" in progress[0]
    assert not waiter.any_waiting()
    assert waiter.waited_seconds() == pytest.approx(600)
    # a target in the past returns at once without logging
    assert waiter.wait_until(start, log=logs.append) is True and len(logs) == 4


def test_waiter_shares_one_log_line_between_threads():
    waiter = rs.QueueWaiter(log_interval=0.1, poll=0.01)
    logs = []
    lock = threading.Lock()

    def log(line):
        with lock:
            logs.append(line)

    target = time.time() + 0.6
    threads = [threading.Thread(target=waiter.wait_until, args=(target,), kwargs={"log": log}) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5)
    starts = [line for line in logs if "waiting" in line]
    progress = [line for line in logs if "waited" in line]
    assert len(starts) == 1
    assert 1 <= len(progress) <= 8          # one line per 0.1 s interval, not one per thread (~30)
    assert not waiter.any_waiting()


def test_waiter_stops_on_request():
    waiter = rs.QueueWaiter(poll=0.01)
    flag = threading.Event()
    threading.Timer(0.1, flag.set).start()
    started = time.time()
    assert waiter.wait_until(time.time() + 30, stop_check=flag.is_set, log=None) is False
    assert time.time() - started < 5


# ---------------------------------------------------------------------------- UnifiedClient path
@pytest.fixture(scope="module")
def client(tmp_path_factory):
    import unified_api_client as api
    return api.UnifiedClient(api_key="sk-test", model="gpt-4o-mini",
                             output_dir=str(tmp_path_factory.mktemp("out")))


def _record_sends(monkeypatch, client, clock):
    sent = []

    def fake_send_internal(*args, **kwargs):
        sent.append(clock.t)
        return "translated", "stop"

    monkeypatch.setattr(client, "_send_internal", fake_send_internal)
    return sent


def _quiet(monkeypatch):
    import unified_api_client as api
    logs = []
    monkeypatch.setattr(api, "print", lambda *a, **k: logs.append(" ".join(map(str, a))), raising=False)
    return logs


def test_main_key_request_waits_until_the_window_ends(monkeypatch, client):
    logs = _quiet(monkeypatch)
    start = at(2026, 10, MON, 3, 0)                                      # off-peak starts in 1 hour
    clock, _waiter = fake_waiter(monkeypatch, start)
    monkeypatch.setenv(rs.MAIN_QUEUE_ENV, json.dumps(rs.to_config(DEEPSEEK_PEAK_SCHEDULE)))
    sent = _record_sends(monkeypatch, client, clock)
    assert client.send([{"role": "user", "content": "hi"}]) == ("translated", "stop")
    assert sent == [at(2026, 10, MON, 4).timestamp()]                    # nothing sent before 04:00 UTC
    queue_logs = [line for line in logs if "Request Queue" in line]
    assert queue_logs[0].startswith("⏳ Request Queue: waiting 1h 00m — requests start at ")
    assert sum("waited" in line for line in queue_logs) == 19            # every 3 minutes for an hour
    # outside the window: no wait, no log
    sent.clear()
    logs.clear()
    assert client.send([{"role": "user", "content": "hi"}]) == ("translated", "stop")
    assert sent == [at(2026, 10, MON, 4).timestamp()] and not any("Request Queue" in l for l in logs)


@pytest.mark.parametrize("stop", ["graceful", "forced", "glossary_stop_file"])
def test_stops_end_the_wait_without_sending(monkeypatch, client, stop, tmp_path):
    import unified_api_client as api
    from unified_api_client import UnifiedClientError
    _quiet(monkeypatch)

    def press_stop(clock):
        if clock.sleeps == 3:
            if stop == "graceful":
                os.environ["GRACEFUL_STOP"] = "1"
            elif stop == "forced":
                api.global_stop_flag = True
            else:
                stop_file = tmp_path / "stop"
                stop_file.write_text("graceful", encoding="utf-8")
                os.environ["GLOSSARY_STOP_FILE"] = str(stop_file)

    start = at(2026, 10, MON, 6, 0)                                      # four hours of peak left
    clock, _waiter = fake_waiter(monkeypatch, start, on_sleep=press_stop)
    monkeypatch.setenv(rs.MAIN_QUEUE_ENV, json.dumps(rs.to_config(DEEPSEEK_PEAK_SCHEDULE)))
    monkeypatch.setattr(api, "global_stop_flag", False)
    sent = _record_sends(monkeypatch, client, clock)
    try:
        with pytest.raises(UnifiedClientError) as info:
            client.send([{"role": "user", "content": "hi"}])
    finally:
        os.environ.pop("GRACEFUL_STOP", None)
        os.environ.pop("GLOSSARY_STOP_FILE", None)
        api.global_stop_flag = False
        client._cancelled = False
    assert getattr(info.value, "error_type", None) == "cancelled"
    assert sent == []                                                    # the request never went out
    assert clock.t - start.timestamp() <= 4 * 30                         # stopped right away


def test_multi_key_pool_waits_only_when_every_usable_key_is_queued(monkeypatch):
    import unified_api_client as api
    from multi_api_key_manager import APIKeyPool
    start = at(2026, 10, MON, 2, 0)
    clock, _waiter = fake_waiter(monkeypatch, start)
    logs = _quiet(monkeypatch)
    client = api.UnifiedClient.__new__(api.UnifiedClient)
    client._multi_key_mode = True
    client._is_stop_requested = lambda: False
    client._is_instance_cancel_requested = lambda: False
    client._ignore_graceful_stop = False
    client.model = "deepseek-chat"
    peak = rs.to_config(DEEPSEEK_PEAK_SCHEDULE)
    late = {"windows": [["00:00", "05:00"]], "timezone": "UTC"}
    pool = APIKeyPool("Translation keys")
    pool.load_from_list([{"api_key": "a", "model": "m", "queue_schedule": peak},
                         {"api_key": "b", "model": "m"}])
    client._api_key_pool = pool
    client._wait_for_request_queue()                                     # 'b' is free: no wait
    assert clock.t == start.timestamp() and not logs
    pool.load_from_list([{"api_key": "a", "model": "m", "queue_schedule": peak},
                         {"api_key": "b", "model": "m", "queue_schedule": late},
                         {"api_key": "c", "model": "m", "enabled": False}])
    client._wait_for_request_queue()                                     # earliest end: 04:00
    assert clock.t == at(2026, 10, MON, 4).timestamp()
    assert logs[0].startswith("⏳ Request Queue (Translation keys): waiting 2h 00m")
    # single-key mode ignores pool schedules and follows the main key's (none here)
    client._multi_key_mode = False
    assert client._request_queue_target(None, at(2026, 10, MON, 2)) == (None, None)


# ------------------------------------------------------------------------- wrapper chunk timeouts
class _QueuedClient:
    """send() spends *wait* seconds in a Request Queue wait (or a plain sleep) before answering."""

    def __init__(self, wait, queued=True):
        self.wait = wait
        self.queued = queued

    def send(self, messages, temperature=None, max_tokens=None, context=None):
        if self.queued:
            assert rs.WAITER.wait_until(time.time() + self.wait, log=None)
        else:
            time.sleep(self.wait)
        return "translated", "stop"


def test_translation_chunk_timeout_never_counts_the_wait(monkeypatch):
    import TransateKRtoEN
    from unified_api_client import UnifiedClientError
    monkeypatch.setenv("RETRY_TIMEOUT", "1")
    monkeypatch.setenv("THREAD_SUBMISSION_DELAY_SECONDS", "0")
    messages = [{"role": "user", "content": "hi"}]
    result = TransateKRtoEN.send_with_interrupt(messages, _QueuedClient(2.5), 0.3, 100, lambda: False,
                                                chunk_timeout=1)
    assert result[0] == "translated"                                      # 2.5 s wait > 1 s timeout
    with pytest.raises(UnifiedClientError, match="timed out after 1 seconds"):
        TransateKRtoEN.send_with_interrupt(messages, _QueuedClient(2.5, queued=False), 0.3, 100,
                                           lambda: False, chunk_timeout=1)


def test_glossary_chunk_timeout_never_counts_the_wait(monkeypatch):
    import extract_glossary_from_epub as glossary
    from unified_api_client import UnifiedClientError
    monkeypatch.setenv("RETRY_TIMEOUT", "1")
    monkeypatch.setenv("THREAD_SUBMISSION_DELAY_SECONDS", "0")
    monkeypatch.setenv("SEND_INTERVAL_SECONDS", "0")
    monkeypatch.setattr(glossary, "_glossary_last_thread_submit", 0, raising=False)
    messages = [{"role": "user", "content": "hi"}]
    content, finish, _raw = glossary.send_with_interrupt(messages, _QueuedClient(2.5), 0.3, 100, lambda: False,
                                                         chunk_timeout=1)
    assert (content, finish) == ("translated", "stop")
    with pytest.raises(UnifiedClientError):
        glossary.send_with_interrupt(messages, _QueuedClient(2.5, queued=False), 0.3, 100, lambda: False,
                                     chunk_timeout=1)


def test_image_wrappers_never_count_the_wait(monkeypatch):
    import image_translator
    from unified_api_client import UnifiedClientError
    messages = [{"role": "user", "content": "hi"}]
    assert image_translator.send_text_with_interrupt(_QueuedClient(2.0), messages, 0.3, 100, lambda: False,
                                                     chunk_timeout=1) == ("translated", "stop")
    with pytest.raises(UnifiedClientError, match="timed out"):
        image_translator.send_text_with_interrupt(_QueuedClient(2.0, queued=False), messages, 0.3, 100,
                                                  lambda: False, chunk_timeout=1)


# ------------------------------------------------------------------------------- settings plumbing
def test_main_key_setting_is_saved_and_exported():
    import settings_schema_data
    entry = settings_schema_data.SETTINGS[rs.MAIN_QUEUE_CONFIG_KEY]
    assert entry["default"] == {} and entry["type"] == "dict"
    assert {env for env, *_rest in entry["env"]} == {rs.MAIN_QUEUE_ENV}
    run_env = (SRC / "run_env.py").read_text(encoding="utf-8")
    assert run_env.count("'REQUEST_QUEUE_SCHEDULE'") == 3
    os.environ[rs.MAIN_QUEUE_ENV] = json.dumps(rs.to_config(DEEPSEEK_PEAK_SCHEDULE))
    try:
        assert rs.main_queue_schedule_from_env() == DEEPSEEK_PEAK_SCHEDULE
    finally:
        del os.environ[rs.MAIN_QUEUE_ENV]
    assert rs.main_queue_schedule_from_env() is None


def test_new_module_is_in_every_spec():
    import ast
    for spec in sorted(SRC.glob("*.spec")):
        tree = ast.parse(spec.read_text(encoding="utf-8"))
        names = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                if node.targets[0].id in ("app_files", "app_modules"):
                    names[node.targets[0].id] = ast.literal_eval(node.value)
        assert ("request_schedule.py", ".") in names["app_files"], spec.name
        assert "request_schedule" in names["app_modules"], spec.name


def test_sources_have_uniform_line_endings():
    for name in ("request_schedule.py", "key_contexts.py", "multi_api_key_manager.py", "unified_api_client.py",
                 "translator_gui.py", "run_env.py", "TransateKRtoEN.py", "extract_glossary_from_epub.py",
                 "image_translator.py", "key_pool_service.py"):
        data = (SRC / name).read_bytes()
        assert data.count(b"\r\n") in (0, data.count(b"\n")), name


def test_one_schedule_per_key_helpers():
    """key_schedule_config / set_key_schedule / key_schedule_overview / pool_schedule_kinds: the
    ⏰ Schedules tab and the mobile "⏰ Schedule…" sheet store one schedule per key."""
    peak, off = rs.to_config(DEEPSEEK_PEAK_SCHEDULE), rs.to_config(DEEPSEEK_PEAK_SCHEDULE, enabled=False)
    assert rs.pool_schedule_kinds("fallback") == ("disable",)
    for pool in ("main", "glossary", "metadata", "tts"):
        assert rs.pool_schedule_kinds(pool) == ("disable", "queue")
    assert set(rs.KIND_CHOICES) == set(rs.SCHEDULE_KINDS) == set(rs.KIND_FIELDS)
    assert rs.KIND_CHOICES["disable"].startswith("🗓️") and rs.KIND_CHOICES["queue"].startswith("⏳")
    # loading: whichever field is set, switched-off configs included
    assert rs.key_schedule_config({"api_key": "k"}) == ("disable", {})
    assert rs.key_schedule_config({"queue_schedule": off}) == ("queue", off)
    assert rs.key_schedule_config({"disable_schedule": peak}) == ("disable", peak)
    assert rs.key_schedule_config({"disable_schedule": off, "queue_schedule": peak}) == ("queue", peak)
    # switching behaviour moves the schedule to the other field
    key = {"api_key": "k", "queue_schedule": peak}
    rs.set_key_schedule(key, "disable", peak)
    assert key == {"api_key": "k", "disable_schedule": peak}
    rs.set_key_schedule(key, "queue", off)
    assert key == {"api_key": "k", "queue_schedule": off}
    rs.set_key_schedule(key, "disable", {})
    assert key == {"api_key": "k"}
    from multi_api_key_manager import APIKeyEntry
    entry = APIKeyEntry("k", "m")
    rs.set_key_schedule(entry, "queue", peak)
    assert entry.to_dict()["queue_schedule"] == peak and "disable_schedule" not in entry.to_dict()
    rs.set_key_schedule(entry, "disable", peak)
    assert entry.to_dict()["disable_schedule"] == peak and "queue_schedule" not in entry.to_dict()
    with pytest.raises(ValueError):
        rs.set_key_schedule({}, "sometimes", peak)
    # list overview: Off / saved but off / windows / current state
    monday_2 = at(2026, 10, MON, 2)
    assert rs.key_schedule_overview({"api_key": "k"}) == "Off"
    assert rs.key_schedule_overview({"queue_schedule": off}) == "Off (⏳ windows saved)"
    assert rs.key_schedule_overview({"disable_schedule": peak}, "UTC", monday_2).startswith("🗓️ Disabled until 04:00")
    assert rs.key_schedule_overview({"queue_schedule": peak}, "UTC", at(2026, 10, SAT, 12)) == \
        "⏳ 01:00–04:00, 06:00–10:00 (UTC), Mon–Fri"
    # off by default: a saved-but-off schedule is never active
    assert not rs.key_queued_now({"queue_schedule": off}, monday_2)
    assert not rs.key_disabled_by_schedule({"disable_schedule": off}, monday_2)


def test_preview_status_line_keeps_the_line_count():
    """The on / off status line is always the first preview line (switching never moves the layout)."""
    for schedule in (DEEPSEEK_PEAK_SCHEDULE, None):
        on = rs.editor_preview_lines(schedule, "UTC", enabled=True, now=at(2026, 10, MON, 2))
        off = rs.editor_preview_lines(schedule, "UTC", enabled=False, now=at(2026, 10, MON, 2))
        assert on[0] == rs.ON_PREVIEW_LINE and off[0] == rs.OFF_PREVIEW_LINE
        assert len(on) == len(off) and on[1:] == off[1:]


# ------------------------------------------------------------------------------------- desktop UI
def _qt(monkeypatch):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", os.environ.get("QT_QPA_PLATFORM", "offscreen"))
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    import multi_api_key_manager as mkm
    if not mkm.HAS_GUI:
        pytest.skip("key manager imported headless")
    return app, mkm


def test_schedule_dialog_edits_presets_and_zones(monkeypatch):
    app, mkm = _qt(monkeypatch)
    # no schedule yet: the DeepSeek peak hours are suggested (kept in UTC)
    dialog = mkm.RequestScheduleDialog(None, None, mode="main_queue", subject="deepseek-chat")
    editor = dialog.editor
    assert isinstance(editor, mkm.ScheduleEditor) and editor.kind_combo is None and editor.kind() is None
    assert editor._schedule() == DEEPSEEK_PEAK_SCHEDULE
    assert "Suggested" in editor.preview_label.text() and "UTC" in editor.preview_label.text()
    # off by default: the body is greyed out and Save keeps the windows switched off
    assert not editor.enable_check.isChecked() and not editor.body.isEnabled()
    assert editor.preview_label.text().startswith(rs.OFF_PREVIEW_LINE)
    assert "QLabel:disabled" in editor.body.styleSheet() and "QAbstractSpinBox:disabled" in editor.body.styleSheet()
    dialog._save()
    assert dialog.result_config == rs.to_config(DEEPSEEK_PEAK_SCHEDULE, enabled=False)
    assert rs.from_config(dialog.result_config) is None
    reopened = mkm.RequestScheduleDialog(None, dialog.result_config, mode="main_queue")
    assert not reopened.editor.enable_check.isChecked() and reopened.editor._schedule() == DEEPSEEK_PEAK_SCHEDULE
    assert not reopened.editor._suggested
    # switching on keeps the preview's line count and reserved height (no layout jump)
    lines_off = editor.preview_label.text().count("\n")
    height_off = editor.preview_label.heightForWidth(420)
    editor.enable_check.setChecked(True)
    assert editor.body.isEnabled() and editor.preview_label.text().startswith(rs.ON_PREVIEW_LINE)
    assert editor.preview_label.text().count("\n") == lines_off
    assert editor.preview_label.heightForWidth(420) == height_off
    dialog._save()
    assert dialog.result_config == rs.to_config(DEEPSEEK_PEAK_SCHEDULE)
    # switching "Times are in" rewrites the same instants in that zone
    dialog = mkm.RequestScheduleDialog(None, rs.to_config(DEEPSEEK_PEAK_SCHEDULE), mode="queue")
    editor = dialog.editor
    editor._set_combo_zone(editor.zone_combo, "Asia/Tokyo")
    editor._on_zone_changed()
    assert editor._schedule() == Schedule((Window(600, 780), Window(900, 1140)), rs.WEEKDAYS, "Asia/Tokyo")
    # add / remove windows, days, clear
    editor._add_row(20 * 60, 2 * 60)
    assert editor._schedule().windows[-1] == Window(1200, 120)
    editor._remove_row(editor._rows[-1])
    editor.day_checks[5].setChecked(True)
    assert 5 in editor._schedule().weekdays
    editor.clear_button.click()
    assert dialog.result_config == {}
    # the explanation takes only its text height
    assert editor.explanation_label.sizePolicy().verticalPolicy() == editor.explanation_label.sizePolicy().Policy.Maximum
    app.processEvents()


def _manager(mkm, config):
    from copy import deepcopy
    from types import SimpleNamespace
    saved = []
    gui = SimpleNamespace(config=config, save_config=lambda **_kwargs: saved.append(deepcopy(config)))
    return mkm.MultiAPIKeyDialog(None, gui), saved


def _select(tree, *rows):
    tree.clearSelection()
    for row in rows:
        tree.topLevelItem(row).setSelected(True)


def test_schedules_tab_edits_one_schedule_per_key(monkeypatch):
    app, mkm = _qt(monkeypatch)
    peak = rs.to_config(DEEPSEEK_PEAK_SCHEDULE)
    config = {
        "use_multi_api_keys": True,
        "multi_api_keys": [
            {"api_key": "sk-first-local-test-key", "model": "deepseek-chat", "enabled": True, "queue_schedule": peak},
            {"api_key": "sk-second-local-test-key", "model": "deepseek-chat", "enabled": True},
        ],
        "fallback_keys": [{"api_key": "fb-local-test-key", "model": "gemini-3-pro", "enabled": True}],
    }
    dialog, saved = _manager(mkm, config)
    try:
        tabs = dialog.manager_tabs
        assert [tabs.tabText(i) for i in range(tabs.count())] == ["🔑 Key Pools", "⏰ Schedules"]
        tabs.setCurrentIndex(1)
        editor, tree = dialog.schedule_editor, dialog.schedule_key_tree
        assert editor.mode == "key" and editor.kind_combo is not None
        # every pool of the manager can be picked; Halgakos arrows on every dropdown of the tab
        pools = [dialog.schedule_pool_combo.itemData(i) for i in range(dialog.schedule_pool_combo.count())]
        assert pools[:3] == ["main", "fallback", "glossary"] and set(pools) == set(mkm.POOL_CONTEXTS)
        for combo in (dialog.schedule_pool_combo, editor.kind_combo, editor.zone_combo, editor.preview_zone_combo):
            assert "down-arrow" in combo.styleSheet() and "Halgakos" in combo.styleSheet()
        assert dialog.schedule_pool_combo.itemText(0) == "✅ Translation Keys (2)"
        # the list: masked key, model, schedule state
        assert tree.topLevelItemCount() == 2
        assert tree.topLevelItem(0).text(1) == "…-key" and tree.topLevelItem(0).toolTip(1) == "sk-first...-key" and tree.topLevelItem(0).text(2) == "deepseek-chat"
        assert tree.topLevelItem(0).text(dialog.SCHEDULE_SCHEDULE_COLUMN).startswith("⏳") and tree.topLevelItem(1).text(dialog.SCHEDULE_SCHEDULE_COLUMN) == "Off"
        assert not editor.isEnabled()          # nothing selected
        # loading a key shows its field and enabled state
        _select(tree, 0)
        assert editor.isEnabled() and editor.kind() == "queue" and editor.enable_check.isChecked()
        _select(tree, 1)
        assert editor.kind() == "disable" and not editor.enable_check.isChecked() and editor._suggested
        editor.set_kind("queue")
        # off by default: saving without enabling stores "enabled": false and nothing is active
        dialog._apply_schedule_to_selected()
        second = config["multi_api_keys"][1]
        assert second["queue_schedule"] == rs.to_config(DEEPSEEK_PEAK_SCHEDULE, enabled=False)
        assert rs.key_schedule(dialog.key_pool.keys[1], rs.QUEUE_FIELD) is None
        assert not rs.key_queued_now(second, at(2026, 10, MON, 2))
        assert tree.topLevelItem(1).text(dialog.SCHEDULE_SCHEDULE_COLUMN) == "Off (⏳ windows saved)" and saved
        # switching the behaviour writes the chosen field and removes the other one
        _select(tree, 0, 1)
        assert "differ" in dialog.schedule_selection_label.text()
        editor.set_kind("disable")
        editor.enable_check.setChecked(True)
        dialog._apply_schedule_to_selected()
        for key in config["multi_api_keys"]:
            assert key["disable_schedule"] == peak and "queue_schedule" not in key
        assert all(entry.queue_schedule == {} and entry.disable_schedule == peak for entry in dialog.key_pool.keys)
        assert saved[-1]["multi_api_keys"][0] == config["multi_api_keys"][0]
        # the key list's Status column keeps showing the schedule state
        status, _lines = rs.key_schedule_summary(dialog.key_pool.keys[0])
        column = dialog.tree.topLevelItem(0).text(6)
        assert (status and column == status) or "🗓️" in column
        # 🧹 Clear removes both fields
        _select(tree, 0)
        editor.clear_button.click()
        first = config["multi_api_keys"][0]
        assert "disable_schedule" not in first and "queue_schedule" not in first
        assert tree.topLevelItem(0).text(dialog.SCHEDULE_SCHEDULE_COLUMN) == "Off"
        # Fallback keys are tried one by one: no queue option (disabled, with the reason)
        assert dialog.select_schedule_pool("fallback")
        queue_index = editor.kind_combo.findData("queue")
        assert not editor.kind_combo.model().item(queue_index).isEnabled()
        assert editor.kind_combo.itemData(queue_index, mkm.Qt.ToolTipRole) == rs.QUEUE_UNAVAILABLE_HELP
        _select(tree, 0)
        editor.set_kind("queue")
        assert editor.kind() == "disable"
        editor.enable_check.setChecked(True)
        dialog._apply_schedule_to_selected()
        assert config["fallback_keys"][0]["disable_schedule"] == peak
        assert "queue_schedule" not in config["fallback_keys"][0]
        # back on a rotating pool the queue option is offered again
        assert dialog.select_schedule_pool("glossary")
        assert editor.kind_combo.model().item(queue_index).isEnabled()
        assert tree.topLevelItemCount() == 0 and "no keys" in dialog.schedule_selection_label.text()
    finally:
        dialog.close()
        dialog.deleteLater()
        app.processEvents()


def test_desktop_and_mobile_store_the_same_schedule_config():
    """A schedule written by the desktop tab (set_key_schedule on a config dict) validates unchanged
    through the shared key_pool_service (the mobile KeyEditor / set_key_fields path) and back."""
    import key_pool_service
    from multi_api_key_manager import APIKeyEntry
    off = rs.to_config(DEEPSEEK_PEAK_SCHEDULE, enabled=False)
    key = rs.set_key_schedule({"api_key": "k", "model": "deepseek-chat", "disable_schedule": off}, "queue", off)
    data, error = key_pool_service.validate_entry(dict(key), pool="main")
    assert error is None and data["queue_schedule"] == off and "disable_schedule" not in data
    entry = APIKeyEntry.from_dict(data)
    assert entry.to_dict()["queue_schedule"] == off
    assert rs.key_schedule_config(entry) == ("queue", off)
    # the mobile write ({chosen: config, other: {}}) gives the same dict
    mobile = dict(key, disable_schedule={}, queue_schedule=off)
    assert key_pool_service.validate_entry(mobile, pool="main")[0] == data


def test_menu_entries_are_wired():
    manager = (SRC / "multi_api_key_manager.py").read_text(encoding="utf-8")
    # the per-kind key menu entries moved into the ⏰ Schedules tab
    assert "Disable schedule…" not in manager and "Request Queue…\")" not in manager
    assert "schedule_callback" not in manager and "_edit_selected_key_schedule" not in manager
    assert '"⏰ Schedules"' in manager and "def _ensure_schedules_tab" in manager
    gui = (SRC / "translator_gui.py").read_text(encoding="utf-8-sig")
    assert '"⏳ Request Queue… · On" if queue_on else "⏳ Request Queue…"' in gui
    assert "queue_action.setCheckable" not in gui and "def _edit_main_request_queue" in gui


# ------------------------------------------- ⏰ Schedules tab ↔ 🔑 Key Pools tab: one set of keys
def _sync_config():
    return {
        "use_multi_api_keys": True,
        "multi_api_keys": [
            {"api_key": "sk-main-one-local-test", "model": "deepseek-chat", "enabled": True},
            {"api_key": "sk-main-two-local-test", "model": "gemini-3-pro", "enabled": True},
            {"api_key": "sk-main-three-local-test", "model": "gpt-5.6-sol", "enabled": True},
        ],
        "use_fallback_keys": True,
        "fallback_keys": [
            {"api_key": "fb-one-local-test-key", "model": "gemini-3-pro", "enabled": True},
            {"api_key": "fb-two-local-test-key", "model": "deepseek-chat", "enabled": True},
        ],
        "glossary_keys": [
            {"api_key": "gl-one-local-test-key", "model": "gemini-3-pro", "enabled": True},
            {"api_key": "gl-two-local-test-key", "model": "deepseek-chat", "enabled": True},
        ],
        "metadata_keys": [
            {"api_key": "md-one-local-test-key", "model": "gemini-3-pro", "enabled": True},
            {"api_key": "md-two-local-test-key", "model": "deepseek-chat", "enabled": True},
        ],
    }


def _schedules_dialog(monkeypatch, config=None):
    app, mkm = _qt(monkeypatch)
    config = config or _sync_config()
    dialog, saved = _manager(mkm, config)
    dialog.manager_tabs.setCurrentIndex(1)
    app.processEvents()
    return app, mkm, dialog, config, saved


def _close(app, dialog):
    dialog.close()
    dialog.deleteLater()
    app.processEvents()


def _models(tree, column):
    return [tree.topLevelItem(row).text(column) for row in range(tree.topLevelItemCount())]


def _menu_texts(menu):
    texts = []
    for action in menu.actions():
        if action.isSeparator():
            texts.append("-")
        else:
            texts.append(action.text())
            if action.menu() is not None:
                texts.extend(f"  {sub.text()}" for sub in action.menu().actions())
    return texts


def test_schedules_tab_adds_keys_with_the_key_pools_form(monkeypatch):
    app, mkm, dialog, config, saved = _schedules_dialog(monkeypatch)
    errors = []
    monkeypatch.setattr(mkm.QMessageBox, "critical", lambda *args, **_kw: errors.append(args[2]))
    try:
        host, tree = dialog.schedule_add_host, dialog.schedule_key_tree
        # the Translation pool's own add form (same widgets) sits in the tab
        assert host.isAncestorOf(dialog.add_key_frame) and host.isAncestorOf(dialog.model_combo)
        assert "Halgakos" in dialog.model_combo.styleSheet()
        # same validation: no model → the Key Pools error, nothing added
        dialog.api_key_entry.setText("sk-added-from-schedules")
        dialog._set_combo_text_silently(dialog.model_combo, "")
        dialog.add_key_btn.click()
        assert errors and len(dialog.key_pool.keys) == 3
        dialog.model_combo.setCurrentText("deepseek-chat")
        dialog.add_key_btn.click()
        app.processEvents()
        assert [k["api_key"] for k in config["multi_api_keys"]][-1] == "sk-added-from-schedules"
        assert dialog.tree.topLevelItemCount() == 4 and tree.topLevelItemCount() == 4
        assert tree.topLevelItem(3).text(1) == "…ules" and tree.topLevelItem(3).toolTip(1) == "sk-added...ules" and tree.topLevelItem(3).text(2) == "deepseek-chat"
        assert dialog.schedule_pool_combo.itemText(0) == "✅ Translation Keys (4)"
        # another pool: its form comes in, the Translation form goes home
        assert dialog.select_schedule_pool("fallback")
        assert host.isAncestorOf(dialog.add_fallback_frame) and not host.isAncestorOf(dialog.add_key_frame)
        assert dialog.translation_key_pool_frame.isAncestorOf(dialog.add_key_frame)
        dialog.fallback_key_entry.setText("fb-added-from-schedules")
        dialog.fallback_model_combo.setCurrentText("gemini-3-pro")
        dialog._add_fallback_key()
        app.processEvents()
        assert config["fallback_keys"][-1]["api_key"] == "fb-added-from-schedules"
        assert dialog.fallback_tree.topLevelItemCount() == 3 and tree.topLevelItemCount() == 3
        # a dedicated pool too
        assert dialog.select_schedule_pool("metadata")
        assert host.isAncestorOf(dialog.metadata_add_frame)
        # back on 🔑 Key Pools every form is in its own section
        dialog.manager_tabs.setCurrentIndex(0)
        assert not host.isAncestorOf(dialog.metadata_add_frame) and not dialog.schedule_add_hint.isVisibleTo(dialog)
        assert dialog.add_fallback_frame.parentWidget() is not host
        # a key added on 🔑 Key Pools shows in the Schedules list
        dialog.select_schedule_pool("fallback")
        dialog.fallback_key_entry.setText("fb-added-on-key-pools")
        dialog.fallback_model_combo.setCurrentText("gemini-3-pro")
        dialog._add_fallback_key()
        app.processEvents()
        assert tree.topLevelItemCount() == 4 and tree.topLevelItem(3).toolTip(1) == "fb-added...ools"
    finally:
        _close(app, dialog)


def test_schedules_tab_context_menu_is_the_key_pools_menu(monkeypatch):
    app, mkm, dialog, config, saved = _schedules_dialog(monkeypatch)
    from PySide6.QtCore import QPoint
    menus, choose = [], {"text": None}

    def fake_exec(menu, _position):
        menus.append(_menu_texts(menu))
        for action in menu.actions():
            if choose["text"] and action.text().startswith(choose["text"]):
                action.trigger()

    dialog._exec_key_context_menu = fake_exec
    monkeypatch.setattr(mkm.QMessageBox, "question", lambda *_a, **_k: mkm.QMessageBox.Yes)
    tree = dialog.schedule_key_tree
    tree.itemAt = lambda _pos: tree.topLevelItem(0)
    try:
        for pool in ("main", "fallback", "glossary", "metadata"):
            assert dialog.select_schedule_pool(pool)
            pool_tree = dialog._key_context_tree(pool)
            pool_tree.itemAt = lambda _pos, t=pool_tree: t.topLevelItem(0)
            _select(pool_tree, 0, 1)
            dialog._pool_key_tree_actions(pool)["menu"](QPoint(1, 1))
            _select(tree, 0, 1)
            dialog._show_schedule_key_context_menu(QPoint(1, 1))
            key_pools_menu, schedules_menu = menus[-2:]
            assert schedules_menu == key_pools_menu, pool
            assert "📝 Change Model (2 selected)" in schedules_menu and "🚫 Disable Selected" in schedules_menu
            # the menu acts on the keys selected in the Schedules list, in both tabs
            choose["text"] = "🚫 Disable Selected"
            _select(tree, 0, 1)
            dialog._show_schedule_key_context_menu(QPoint(1, 1))
            app.processEvents()
            keys = dialog._pool_key_list(pool)
            assert [dialog._schedule_key_value(k, "enabled", True) for k in keys][:2] == [False, False], pool
            status_column = 6 if pool == "main" else 5
            assert pool_tree.topLevelItem(0).text(status_column) == "Disabled"
            assert tree.topLevelItem(0).text(dialog.SCHEDULE_STATUS_COLUMN) == "Disabled" and tree.topLevelItem(1).text(dialog.SCHEDULE_STATUS_COLUMN) == "Disabled"
            assert tree.topLevelItem(0).isSelected() and tree.topLevelItem(1).isSelected()
            choose["text"] = "✅ Enable Selected"
            dialog._show_schedule_key_context_menu(QPoint(1, 1))
            app.processEvents()
            assert all(dialog._schedule_key_value(k, "enabled", True) for k in dialog._pool_key_list(pool))
            assert tree.topLevelItem(0).text(dialog.SCHEDULE_STATUS_COLUMN) != "Disabled"
            # remove the second key from the Schedules menu
            choose["text"] = "🗑️ Remove Selected"
            count = len(dialog._pool_key_list(pool))
            removed = dialog._pool_key_list(pool)[1]
            _select(tree, 1)
            tree.itemAt = lambda _pos: tree.topLevelItem(1)
            dialog._show_schedule_key_context_menu(QPoint(1, 1))
            tree.itemAt = lambda _pos: tree.topLevelItem(0)
            choose["text"] = None
            app.processEvents()
            keys = dialog._pool_key_list(pool)
            assert len(keys) == count - 1 and all(k is not removed for k in keys), pool
            assert pool_tree.topLevelItemCount() == count - 1 and tree.topLevelItemCount() == count - 1
        assert config["fallback_keys"][0]["enabled"] and len(saved[-1]["metadata_keys"]) == 1
    finally:
        _close(app, dialog)


def test_schedules_tab_double_click_model_uses_the_key_pools_editor(monkeypatch):
    app, mkm, dialog, config, saved = _schedules_dialog(monkeypatch)
    editors = []

    def fake_editor(current_value="", title="Edit Model", label_text="Model Name:", accept_text="OK"):
        editors.append((current_value, title))
        return f"renamed-{len(editors)}", True

    dialog._show_model_selection_dialog = fake_editor
    tree = dialog.schedule_key_tree
    try:
        # Translation: the Key Pools double-click handler and its "Edit Model" editor
        tree.itemDoubleClicked.emit(tree.topLevelItem(1), dialog.SCHEDULE_MODEL_COLUMN)
        app.processEvents()
        assert editors[-1] == ("gemini-3-pro", "Edit Model")
        assert dialog.key_pool.keys[1].model == "renamed-1"
        assert dialog.tree.topLevelItem(1).text(1) == "renamed-1" and tree.topLevelItem(1).text(2) == "renamed-1"
        # other columns are not editors
        tree.itemDoubleClicked.emit(tree.topLevelItem(1), 1)
        assert len(editors) == 1
        # renamed on 🔑 Key Pools → shows here
        dialog.tree.itemDoubleClicked.emit(dialog.tree.topLevelItem(0), 1)
        app.processEvents()
        assert tree.topLevelItem(0).text(2) == "renamed-2"
        # Fallback: saved to config through the Key Pools path
        dialog.select_schedule_pool("fallback")
        tree.itemDoubleClicked.emit(tree.topLevelItem(0), dialog.SCHEDULE_MODEL_COLUMN)
        app.processEvents()
        assert editors[-1] == ("gemini-3-pro", "Edit Model")
        assert config["fallback_keys"][0]["model"] == "renamed-3" and saved[-1]["fallback_keys"][0]["model"] == "renamed-3"
        assert dialog.fallback_tree.topLevelItem(0).text(1) == "renamed-3" and tree.topLevelItem(0).text(2) == "renamed-3"
        dialog.fallback_tree.itemDoubleClicked.emit(dialog.fallback_tree.topLevelItem(1), 1)
        app.processEvents()
        assert tree.topLevelItem(1).text(2) == "renamed-4"
        # a dedicated pool: its double-click → its change-model editor
        dialog.select_schedule_pool("metadata")
        tree.itemDoubleClicked.emit(tree.topLevelItem(1), dialog.SCHEDULE_MODEL_COLUMN)
        app.processEvents()
        assert editors[-1][0] == "deepseek-chat" and editors[-1][1].startswith("Change Model for 1")
        assert config["metadata_keys"][1]["model"] == "renamed-5" and tree.topLevelItem(1).text(2) == "renamed-5"
        assert dialog.metadata_tree.topLevelItem(1).text(1) == "renamed-5"
    finally:
        _close(app, dialog)


def test_schedules_tab_editor_width_and_centered_save(monkeypatch):
    app, mkm, dialog, config, saved = _schedules_dialog(monkeypatch)
    try:
        dialog.resize(2000, 1200)
        dialog.show()
        app.processEvents()
        button = dialog.schedule_save_button
        layout = button.parentWidget().layout()
        assert layout.itemAt(layout.indexOf(button)).alignment() == mkm.Qt.AlignHCenter
        # the editor keeps the width of its widest row, the key list takes the rest
        splitter = dialog.schedule_splitter
        editor_scroll = splitter.widget(1)
        panel = editor_scroll.widget()
        width = editor_scroll.maximumWidth()
        assert dialog.SCHEDULE_EDITOR_MIN_WIDTH <= width <= 700, width
        assert width >= panel.minimumSizeHint().width()
        list_width, editor_width = splitter.sizes()
        assert editor_width <= width and list_width > editor_width
        # nothing in the editor needs more than that (the windows note wraps, the kind combo elides)
        assert dialog.schedule_editor.minimumSizeHint().width() <= width
        # a smaller window: the list gives way first, the splitter is kept
        dialog.resize(1100, 800)
        app.processEvents()
        assert splitter.sizes()[1] <= width and not splitter.childrenCollapsible()
    finally:
        _close(app, dialog)


def _parameters_config():
    config = _sync_config()
    config["multi_api_keys"][0].update(individual_output_token_limit=16384, individual_key_temperature=0.7,
                                       api_call_delay=2, cooldown=120)
    config["fallback_keys"][1].update(individual_key_temperature=1, api_call_delay="0.5")
    config["glossary_keys"][0].update(individual_output_token_limit="8192")
    config["metadata_keys"][1].update(individual_output_token_limit=4000, individual_key_temperature=0.2,
                                      api_call_delay=3.5)
    config["use_glossary_keys"] = True
    config["use_metadata_keys"] = True
    return config


def _key_pools_parameters(dialog, pool, row):
    """The 🔑 Key Pools list's Output Limit / Temperature / Delay (s) (and ⌛) texts of *row*."""
    columns = dialog._pool_key_tree_columns(pool)
    item = dialog._pool_tree_items_by_index(pool)[row]
    limit, temperature, delay = (item.text(columns[field]) for field in ("output_limit", "temperature", "api_call_delay"))
    values = [limit, temperature, delay if delay == "global" else f"{delay}s"]
    parts = [f"{icon}{value}" for icon, value in zip(("📤", "🌡️", "⏱️"), values)]
    if "cooldown" in columns:
        parts.append(f"⌛{item.text(columns['cooldown'])}")
    return " · ".join(parts)


def test_schedules_parameters_column_shows_the_key_pools_values(monkeypatch):
    app, mkm, dialog, config, saved = _schedules_dialog(monkeypatch, _parameters_config())
    tree, column = dialog.schedule_key_tree, dialog.SCHEDULE_PARAMETERS_COLUMN
    try:
        assert dialog.SCHEDULE_TREE_COLUMNS[column] == "Parameters" and column == dialog.SCHEDULE_MODEL_COLUMN + 1
        assert "temperature" in tree.headerItem().toolTip(column)
        for pool in ("main", "fallback", "glossary", "metadata"):
            assert dialog.select_schedule_pool(pool)
            for row in range(tree.topLevelItemCount()):
                assert tree.topLevelItem(row).text(column) == _key_pools_parameters(dialog, pool, row), (pool, row)
        dialog.select_schedule_pool("main")
        assert tree.topLevelItem(0).text(column) == "📤16384 · 🌡️0.7 · ⏱️2.0s · ⌛120s"
        assert tree.topLevelItem(1).text(column) == "📤global · 🌡️global · ⏱️global · ⌛60s"
        tip = tree.topLevelItem(0).toolTip(column)
        assert "Individual Output Token Limit: 16384" in tip and "cooldown: 120s" in tip
        dialog.select_schedule_pool("fallback")
        assert tree.topLevelItem(1).text(column) == "📤global · 🌡️1.0 · ⏱️0.5s"   # no ⌛ column there
        dialog.select_schedule_pool("glossary")
        assert tree.topLevelItem(0).text(column) == "📤8192 · 🌡️global · ⏱️global"
        assert "Using global temperature" in tree.topLevelItem(0).toolTip(column)
        # compact: no wider than all-global values with a cooldown
        metrics = tree.fontMetrics()
        assert tree.columnWidth(column) <= metrics.horizontalAdvance(dialog.SCHEDULE_PARAMETERS_MAX_WIDTH) + 24
    finally:
        _close(app, dialog)


PARAMETER_EDITS = [("📤 Output limit…", "output_limit", 4096, "4096"),
                   ("🌡️ Temperature…", "temperature", 0.3, "0.3"),
                   ("⏱️ API call delay…", "api_call_delay", 1.5, "1.5s")]
PARAMETER_ICONS = {"output_limit": "📤", "temperature": "🌡️", "api_call_delay": "⏱️"}


@pytest.mark.parametrize("label,field,value,text", PARAMETER_EDITS)
def test_schedules_parameters_double_click_uses_the_key_pools_editors(monkeypatch, label, field, value, text):
    app, mkm, dialog, config, saved = _schedules_dialog(monkeypatch, _parameters_config())
    menus, calls, editors = [], [], []

    def fake_exec(menu, _position):
        menus.append([action.text() for action in menu.actions()])
        next(action for action in menu.actions() if action.text() == label).trigger()

    dialog._exec_key_context_menu = fake_exec
    dialog._show_output_token_limit_dialog = lambda *_a, **_k: (editors.append("limit"), (value, True))[1]
    dialog._show_key_temperature_dialog = lambda *_a, **_k: (editors.append("temperature"), (value, True))[1]
    dialog._show_api_call_delay_dialog = lambda *_a, **_k: (editors.append("delay"), (value, True))[1]
    actions = dialog._pool_key_tree_actions

    def spy(pool_name):
        found = dict(actions(pool_name))
        handler = found["double_click"]
        found["double_click"] = lambda item, column: (calls.append((pool_name, column)), handler(item, column))
        return found

    dialog._pool_key_tree_actions = spy
    tree, column = dialog.schedule_key_tree, dialog.SCHEDULE_PARAMETERS_COLUMN
    position = {"output_limit": 0, "temperature": 1, "api_call_delay": 2}[field]
    try:
        for pool in ("main", "fallback", "glossary", "metadata"):
            assert dialog.select_schedule_pool(pool)
            columns = dialog._pool_key_tree_columns(pool)
            # one key: the Key Pools handler of the pool, with that pool's column for the field
            tree.itemDoubleClicked.emit(tree.topLevelItem(1), column)
            app.processEvents()
            assert calls[-1] == (pool, columns[field]), pool
            expected = ["📤 Output limit…", "🌡️ Temperature…", "⏱️ API call delay…"] + (
                ["⌛ Cooldown…"] if pool == "main" else [])
            assert menus[-1] == expected
            keys = dialog._pool_key_list(pool)
            assert dialog._key_parameter_texts(keys[1])[position] == text.rstrip("s"), pool
            assert tree.topLevelItem(1).text(column).split(" · ")[position] == PARAMETER_ICONS[field] + text, pool
            assert tree.topLevelItem(1).text(column) == _key_pools_parameters(dialog, pool, 1)
            if pool != "main":
                assert saved and dialog._key_parameter_texts(saved[-1][f"{pool}_keys"][1])[position] == text.rstrip("s")
            # several selected: as on 🔑 Key Pools (the double-clicked key; every selected one in
            # a dedicated pool, whose editors act on the selection)
            _select(tree, 0, 2) if tree.topLevelItemCount() > 2 else _select(tree, 0)
            before = dialog._key_parameter_texts(keys[0])[position]
            calls.clear()
            tree.itemDoubleClicked.emit(tree.topLevelItem(1), column)
            app.processEvents()
            changed = dialog._key_parameter_texts(dialog._pool_key_list(pool)[0])[position]
            if pool == "metadata":
                assert changed == text.rstrip("s")
            else:
                assert changed == before, pool
            assert calls == [(pool, columns[field])]
            # cancelled: nothing runs
            menus.clear()
            dialog._exec_key_context_menu = lambda menu, _pos: menus.append(menu)
            tree.itemDoubleClicked.emit(tree.topLevelItem(0), column)
            assert menus and calls == [(pool, columns[field])]
            dialog._exec_key_context_menu = fake_exec
        assert editors == [{"output_limit": "limit", "temperature": "temperature",
                            "api_call_delay": "delay"}[field]] * 8
    finally:
        _close(app, dialog)


def test_schedules_parameters_sync_both_ways(monkeypatch):
    app, mkm, dialog, config, saved = _schedules_dialog(monkeypatch, _parameters_config())
    dialog._show_key_temperature_dialog = lambda *_a, **_k: (0.9, True)
    dialog._show_output_token_limit_dialog = lambda *_a, **_k: (2048, True)
    dialog._show_cooldown_edit_dialog = lambda *_a, **_k: (300, True)
    dialog._exec_key_context_menu = lambda menu, _pos: next(
        action for action in menu.actions() if action.text() == "📤 Output limit…").trigger()
    tree, column = dialog.schedule_key_tree, dialog.SCHEDULE_PARAMETERS_COLUMN
    try:
        for pool in ("main", "fallback", "glossary", "metadata"):
            assert dialog.select_schedule_pool(pool)
            pool_tree = dialog._key_context_tree(pool)
            columns = dialog._pool_key_tree_columns(pool)
            # edited on 🔑 Key Pools → shows here
            _select(pool_tree, 0)
            pool_tree.itemDoubleClicked.emit(pool_tree.topLevelItem(0), columns["temperature"])
            app.processEvents()
            assert tree.topLevelItem(0).text(column).split(" · ")[1] == "🌡️0.9", pool
            if pool == "main":       # the ⌛ cooldown too
                pool_tree.itemDoubleClicked.emit(pool_tree.topLevelItem(0), columns["cooldown"])
                app.processEvents()
                assert tree.topLevelItem(0).text(column).endswith("⌛300s")
            # edited here → shows on 🔑 Key Pools
            tree.itemDoubleClicked.emit(tree.topLevelItem(1), column)
            app.processEvents()
            assert pool_tree.topLevelItem(1).text(columns["output_limit"]) == "2048", pool
            assert tree.topLevelItem(0).text(column) == _key_pools_parameters(dialog, pool, 0)
    finally:
        _close(app, dialog)


def _fake_internal_move(tree, to_row):
    """What Qt's InternalMove drop does: take the selected rows out, insert them at *to_row*."""
    def drop(_event):
        taken = [tree.takeTopLevelItem(tree.indexOfTopLevelItem(item)) for item in tree.selectedItems()]
        for offset, item in enumerate(taken):
            tree.insertTopLevelItem(to_row + offset, item)
    return drop


def test_schedules_tab_reorders_keys_like_the_key_pools_tab(monkeypatch):
    app, mkm, dialog, config, saved = _schedules_dialog(monkeypatch)
    tree = dialog.schedule_key_tree
    keys = lambda: [k.api_key for k in dialog.key_pool.keys]  # noqa: E731
    try:
        one, two, three = keys()
        # the reorder buttons: the Key Pools move handler; the moved key stays selected
        _select(tree, 2)
        dialog._schedule_move_selected("up")
        assert keys() == [one, three, two]
        assert _models(dialog.tree, 0)[1] == tree.topLevelItem(1).toolTip(1) == "sk-main-...test"
        assert tree.topLevelItem(1).text(1) == "…test"   # the compact column: the last 4 characters
        assert [i for i in range(3) if tree.topLevelItem(i).isSelected()] == [1]
        dialog._schedule_move_selected("top")
        assert keys() == [three, one, two] and tree.topLevelItem(0).isSelected()
        assert saved[-1]["multi_api_keys"][0]["api_key"] == three
        # a drop on the Schedules list (Qt's InternalMove) → the stored order and the Key Pools list
        tree.dropEvent = _fake_internal_move(tree, 2)
        dialog._install_key_tree_drop_sync(tree, None)
        _select(tree, 0)
        tree.dropEvent(None)
        assert keys() == [one, two, three]
        assert [k["api_key"] for k in config["multi_api_keys"]] == [one, two, three]
        assert [dialog.tree.topLevelItem(i).text(0) for i in range(3)] == [tree.topLevelItem(i).toolTip(1) for i in range(3)]
        assert [i for i in range(3) if tree.topLevelItem(i).isSelected()] == [2]
        # a drop on the Key Pools list shows here (and is stored: QTreeWidget never emits rowsMoved)
        dialog.tree.dropEvent = _fake_internal_move(dialog.tree, 0)
        dialog._install_key_tree_drop_sync(dialog.tree, "main")
        _select(dialog.tree, 1)
        dialog.tree.dropEvent(None)
        app.processEvents()
        assert keys() == [two, one, three] and dialog.tree.topLevelItem(0).isSelected()
        assert tree.topLevelItem(0).toolTip(1) == "sk-main-...test" and dialog.key_pool.keys[0].api_key == two
        assert [i for i in range(3) if tree.topLevelItem(i).isSelected()] == [2]   # selection follows its key
        # Fallback: buttons and drops write the config list
        dialog.select_schedule_pool("fallback")
        _select(tree, 1)
        dialog._schedule_move_selected("up")
        assert [k["api_key"] for k in config["fallback_keys"]] == ["fb-two-local-test-key", "fb-one-local-test-key"]
        assert _models(dialog.fallback_tree, 1) == _models(tree, 2) == ["deepseek-chat", "gemini-3-pro"]
        tree.dropEvent = _fake_internal_move(tree, 1)
        dialog._install_key_tree_drop_sync(tree, None)
        _select(tree, 0)
        tree.dropEvent(None)
        assert [k["api_key"] for k in saved[-1]["fallback_keys"]] == ["fb-one-local-test-key", "fb-two-local-test-key"]
        assert _models(dialog.fallback_tree, 1) == _models(tree, 2) == ["gemini-3-pro", "deepseek-chat"]
        # Key Pools reorder buttons → here
        _select(dialog.fallback_tree, 1)
        dialog._move_fallback_key("top")
        app.processEvents()
        assert _models(tree, 2) == ["deepseek-chat", "gemini-3-pro"]
    finally:
        _close(app, dialog)


def test_schedules_tab_never_writes_stale_keys(monkeypatch):
    app, mkm, dialog, config, saved = _schedules_dialog(monkeypatch)
    tree, editor = dialog.schedule_key_tree, dialog.schedule_editor
    dialog._show_model_selection_dialog = lambda *_a, **_k: ("renamed-on-key-pools", True)
    try:
        dialog.select_schedule_pool("fallback")
        _select(tree, 0)
        editor.enable_check.setChecked(True)            # unsaved edit in the editor
        # meanwhile on 🔑 Key Pools: rename key 2, add a key
        dialog.fallback_tree.itemDoubleClicked.emit(dialog.fallback_tree.topLevelItem(1), 1)
        dialog.fallback_key_entry.setText("fb-added-meanwhile")
        dialog.fallback_model_combo.setCurrentText("gemini-3-pro")
        dialog._add_fallback_key()
        app.processEvents()
        # the pool, the selection and the unsaved editor survive the refresh
        assert dialog._schedule_pool() == "fallback" and tree.topLevelItemCount() == 3
        assert [i for i in range(3) if tree.topLevelItem(i).isSelected()] == [0]
        assert editor.enable_check.isChecked()
        dialog._apply_schedule_to_selected()
        stored = saved[-1]["fallback_keys"]
        assert [k["api_key"] for k in stored] == ["fb-one-local-test-key", "fb-two-local-test-key", "fb-added-meanwhile"]
        assert stored[1]["model"] == "renamed-on-key-pools" and rs.is_enabled_config(stored[0]["disable_schedule"])
        # Translation keys: a Key Pools edit and a Schedules save write the same objects
        dialog.select_schedule_pool("main")
        dialog.tree.itemDoubleClicked.emit(dialog.tree.topLevelItem(2), 1)
        _select(tree, 0)
        dialog._apply_schedule_to_selected()
        assert saved[-1]["multi_api_keys"][2]["model"] == "renamed-on-key-pools"
        assert len(saved[-1]["multi_api_keys"]) == 3 and "disable_schedule" in saved[-1]["multi_api_keys"][0]
    finally:
        _close(app, dialog)


def test_one_reorder_column_and_drop_sync_for_every_key_list():
    manager = (SRC / "multi_api_key_manager.py").read_text(encoding="utf-8")
    assert manager.count("QPushButton(\"↑ ↑\")") == 0 and manager.count("_create_key_reorder_column(") == 6
    assert "rowsMoved.connect" not in manager and manager.count("_install_key_tree_drop_sync(") == 6


# ------------------------------------------------- ⏰ Schedules tab: lazy building, speed, pool switch
def _many_keys(prefix, count):
    return [{"api_key": f"{prefix}-key-{i:02d}-local-test", "model": "deepseek-chat" if i % 2 else "gemini-3-pro",
             "enabled": True} for i in range(count)]


def _pump(app, seconds, until=lambda: False):
    end = time.perf_counter() + seconds
    while time.perf_counter() < end and not until():
        app.processEvents()
        time.sleep(0.002)


def _pending_pools(dialog):
    return [entry[0] for entry in dialog._pending_key_pool_section_builders]


def test_schedules_tab_and_extra_sections_are_built_lazily(monkeypatch):
    app, mkm = _qt(monkeypatch)
    config = _sync_config()
    dialog, saved = _manager(mkm, config)
    try:
        deferred = [pool for pool, _builder in dialog._deferred_key_pool_section_builders()]
        # construction: no Schedules tab yet, only the Translation section; user actions are live
        assert getattr(dialog, "schedule_editor", None) is None and dialog.schedules_tab.layout() is None
        assert _pending_pools(dialog) == deferred and not hasattr(dialog, "fallback_tree")
        assert dialog._can_emit_live_key_pool_updates()
        # opening ⏰ Schedules builds the tab, not the pending sections; the background render pauses
        dialog.manager_tabs.setCurrentIndex(1)
        app.processEvents()
        assert dialog.schedule_editor is not None and _pending_pools(dialog) == deferred
        assert dialog.schedule_add_host.isAncestorOf(dialog.add_key_frame)
        dialog._deferred_key_pool_render_started = True
        dialog._render_next_deferred_key_pool_section()
        assert _pending_pools(dialog) == deferred and not dialog._deferred_key_pool_render_started
        # unbuilt pools still list their keys (from the stored keys)
        index = dialog.schedule_pool_combo.findData("metadata")
        dialog.schedule_pool_combo.setCurrentIndex(index)
        assert dialog.schedule_key_tree.topLevelItemCount() == 2 and "metadata" in _pending_pools(dialog)
        # ... and only the chosen pool's section is built, right after the switch, in its own place
        _pump(app, 2.0, until=lambda: getattr(dialog, "metadata_add_frame", None) is not None
              and dialog.schedule_add_host.isAncestorOf(dialog.metadata_add_frame))
        assert dialog.schedule_add_host.isAncestorOf(dialog.metadata_add_frame)
        assert _pending_pools(dialog) == [pool for pool in deferred if pool != "metadata"]
        slot = dialog.metadata_container.parentWidget()
        assert dialog._deferred_key_pool_host_layout.indexOf(slot) == deferred.index("metadata")
        assert dialog.metadata_tree.topLevelItemCount() == 2
        assert dialog.translation_key_pool_frame.isAncestorOf(dialog.add_key_frame)
        # the key actions build a pool's section on demand too
        assert dialog.select_schedule_pool("glossary") and "glossary" not in _pending_pools(dialog)
        assert dialog.schedule_add_host.isAncestorOf(dialog.add_glossary_frame)
        # back on 🔑 Key Pools the forms are home and the render resumes until every section exists
        dialog.manager_tabs.setCurrentIndex(0)
        assert dialog.glossary_container.isAncestorOf(dialog.add_glossary_frame)
        _pump(app, 10.0, until=lambda: not dialog._pending_key_pool_section_builders)
        assert not _pending_pools(dialog) and dialog._key_pool_render_finished
        assert dialog._can_emit_live_key_pool_updates()
        host = dialog._deferred_key_pool_host_layout
        built = [host.itemAt(i).widget() for i in range(host.count())]
        assert built.index(dialog.fallback_container.parentWidget()) == deferred.index("fallback")
        assert built.index(slot) == deferred.index("metadata") and all(not w.isHidden() for w in built)
    finally:
        _close(app, dialog)


def test_schedules_pool_switch_is_fast(monkeypatch):
    app, mkm = _qt(monkeypatch)
    config = {
        "use_multi_api_keys": True, "multi_api_keys": _many_keys("sk", 10),
        "use_fallback_keys": True, "fallback_keys": _many_keys("fb", 10),
        "use_glossary_keys": True, "glossary_keys": _many_keys("gl", 10),
        "use_metadata_keys": True, "metadata_keys": _many_keys("md", 10),
    }
    dialog, saved = _manager(mkm, config)
    try:
        dialog.show()
        app.processEvents()
        start = time.perf_counter()
        dialog.manager_tabs.setCurrentIndex(1)
        app.processEvents()
        first_open = time.perf_counter() - start
        pools = ["main", "fallback", "glossary", "metadata"]
        for pool in pools:                      # first visits build the sections
            dialog.select_schedule_pool(pool)
        timings = {}
        combo, tree = dialog.schedule_pool_combo, dialog.schedule_key_tree
        for pool in pools[1:] + pools[:1] + pools[1:]:
            start = time.perf_counter()
            combo.setCurrentIndex(combo.findData(pool))
            app.processEvents()
            shown = time.perf_counter() - start
            assert tree.topLevelItemCount() == 10 and dialog._schedule_pool() == pool
            _pump(app, 2.0, until=lambda: not dialog._schedule_pool_sync_queued)
            timings[pool] = max(timings.get(pool, 0), shown, time.perf_counter() - start)
            assert dialog.schedule_add_host.isAncestorOf(dialog._pool_add_form(pool))
            assert tree.topLevelItem(9).text(dialog.SCHEDULE_STATUS_COLUMN)                  # Status column from the Key Pools list
        assert max(timings.values()) < 0.3, timings
        assert first_open < 1.5, first_open
        # the moved forms keep their layout width (the model pickers no longer re-measure the catalog)
        combo_widths = {pool: dialog._pool_add_form(pool).findChildren(mkm.QComboBox)[0].minimumWidth() for pool in pools}
        assert all(width > 0 for width in combo_widths.values()), combo_widths
    finally:
        _close(app, dialog)


def _timed_slices(monkeypatch, mkm):
    """Record the duration of every slice of a section build (``_step_key_pool_section``)."""
    durations = []
    original = mkm.MultiAPIKeyDialog._step_key_pool_section

    def timed(self, pending):
        start = time.perf_counter()
        try:
            return original(self, pending)
        finally:
            durations.append((pending[0], time.perf_counter() - start))

    monkeypatch.setattr(mkm.MultiAPIKeyDialog, "_step_key_pool_section", timed)
    return durations


def test_schedules_pool_switch_streams_the_section_in_slices(monkeypatch):
    app, mkm = _qt(monkeypatch)
    assert Path(mkm.__file__).resolve().parent == SRC.resolve()
    durations = _timed_slices(monkeypatch, mkm)
    config = _sync_config()
    config["qa_scan_keys"] = _many_keys("qa", 3)
    app, mkm, dialog, config, saved = _schedules_dialog(monkeypatch, config)
    combo, tree, loading = dialog.schedule_pool_combo, dialog.schedule_key_tree, dialog.schedule_add_loading

    def borrowed(pool):
        form = dialog._pool_add_form(pool)
        return form is not None and dialog.schedule_add_host.isAncestorOf(form)

    try:
        dialog.show()
        app.processEvents()
        assert borrowed("main") and "metadata" in _pending_pools(dialog)
        # a click on an unbuilt pool switches at once: its keys show, nothing is built yet
        combo.setCurrentIndex(combo.findData("metadata"))
        assert dialog._schedule_pool() == "metadata" and tree.topLevelItemCount() == 2
        assert "metadata" in _pending_pools(dialog) and getattr(dialog, "metadata_add_frame", None) is None
        assert not durations and dialog._schedule_pool_sync_queued
        # the first slice: "Loading …" where the add form goes (the previous form went home)
        _pump(app, 2.0, until=lambda: bool(durations))
        assert durations == [("metadata", durations[0][1])] and "metadata" in _pending_pools(dialog)
        assert loading.isVisibleTo(dialog) and "Loading Metadata Keys" in loading.text()
        assert dialog.translation_key_pool_frame.isAncestorOf(dialog.add_key_frame)
        prebuilt = dialog._prebuilt_model_combos["metadata"]
        # another pool chosen before it finished: that one loads, the first waits (its work kept)
        combo.setCurrentIndex(combo.findData("qa_scan"))
        assert dialog._schedule_pool() == "qa_scan" and tree.topLevelItemCount() == 3
        _pump(app, 3.0, until=lambda: borrowed("qa_scan"))
        assert borrowed("qa_scan") and not loading.isVisibleTo(dialog)
        assert "metadata" in _pending_pools(dialog) and "qa_scan" not in _pending_pools(dialog)
        assert dialog._prebuilt_model_combos.get("metadata") is prebuilt
        assert {pool for pool, _ in durations} == {"metadata", "qa_scan"}
        # back to it: the rest streams in, with the model picker made in its first slice
        combo.setCurrentIndex(combo.findData("metadata"))
        _pump(app, 3.0, until=lambda: borrowed("metadata"))
        assert borrowed("metadata") and dialog.metadata_model_combo is prebuilt
        assert "metadata" not in dialog._prebuilt_model_combos and dialog.metadata_tree.topLevelItemCount() == 2
        assert tree.topLevelItem(1).text(dialog.SCHEDULE_STATUS_COLUMN)   # Status from the new section
        # each section took two slices, none of them long (a whole section used to be one step)
        assert [pool for pool, _ in durations].count("metadata") == 2
        assert max(seconds for _, seconds in durations) < 0.25, durations
        # nothing is built while a popup (the picker's own list) is open
        monkeypatch.setattr(mkm.MultiAPIKeyDialog, "_key_pool_popup_open", staticmethod(lambda: True))
        count = len(durations)
        combo.setCurrentIndex(combo.findData("tts"))
        _pump(app, 0.4)
        assert len(durations) == count and "tts" in _pending_pools(dialog) and dialog._schedule_pool_sync_queued
        monkeypatch.setattr(mkm.MultiAPIKeyDialog, "_key_pool_popup_open", staticmethod(lambda: False))
        _pump(app, 3.0, until=lambda: borrowed("tts"))
        assert borrowed("tts") and "tts" not in _pending_pools(dialog)
        # select_schedule_pool (other code, the key actions) still builds everything at once
        assert dialog.select_schedule_pool("inpainter") and borrowed("inpainter")
    finally:
        _close(app, dialog)


def test_background_section_render_runs_in_slices_and_waits_for_popups(monkeypatch):
    app, mkm = _qt(monkeypatch)
    durations = _timed_slices(monkeypatch, mkm)
    dialog, saved = _manager(mkm, _sync_config())
    try:
        deferred = [pool for pool, _builder in dialog._deferred_key_pool_section_builders()]
        render = dialog._render_next_deferred_key_pool_section
        render()                                   # the first slice: the model picker only
        assert _pending_pools(dialog) == deferred and set(dialog._prebuilt_model_combos) == {deferred[0]}
        render()                                   # the second: the section, with that picker
        assert _pending_pools(dialog) == deferred[1:] and not dialog._prebuilt_model_combos
        assert dialog.fallback_model_combo.minimumWidth() > 0
        monkeypatch.setattr(mkm.MultiAPIKeyDialog, "_key_pool_popup_open", staticmethod(lambda: True))
        render()
        assert _pending_pools(dialog) == deferred[1:] and not dialog._prebuilt_model_combos
        assert dialog._deferred_key_pool_render_queued      # it tries again later
        monkeypatch.setattr(mkm.MultiAPIKeyDialog, "_key_pool_popup_open", staticmethod(lambda: False))
        _pump(app, 15.0, until=lambda: not dialog._pending_key_pool_section_builders)
        assert not _pending_pools(dialog) and dialog._key_pool_render_finished
        assert len(durations) == 2 * len(deferred)
        assert max(seconds for _, seconds in durations) < 0.25, durations
    finally:
        _close(app, dialog)


def test_schedules_pool_enable_switch_follows_the_key_pools_toggle(monkeypatch):
    import key_pool_service
    env = dict(os.environ)
    app, mkm, dialog, config, saved = _schedules_dialog(monkeypatch)
    title = lambda pool: key_pool_service.POOL_SPECS[pool]["title"]  # noqa: E731
    combo, check, hint = dialog.schedule_pool_combo, dialog.schedule_pool_enable_check, dialog.schedule_add_hint

    def item(pool):
        return combo.itemText(combo.findData(pool))

    try:
        # the picker marks every pool on / off with its key count, built or not
        assert item("main") == f"✅ {title('main')} (3)" and item("fallback") == f"✅ {title('fallback')} (2)"
        assert item("metadata") == f"⛔ {title('metadata')} (2)" and "metadata" in _pending_pools(dialog)
        assert check.isChecked() and check.text() == f"Enable {title('main')}" and not hint.isVisibleTo(dialog)
        # Schedules switch → the Key Pools checkbox and its handler (config, section, saved)
        assert dialog.select_schedule_pool("fallback")
        check.setChecked(False)
        assert not dialog.use_fallback_checkbox.isChecked() and config["use_fallback_keys"] is False
        assert item("fallback").startswith("⛔") and hint.isVisibleTo(dialog)
        assert dialog.add_fallback_frame.isHidden()
        # schedules stay editable while the pool is off
        _select(dialog.schedule_key_tree, 0)
        assert dialog.schedule_editor.isEnabled()
        dialog.schedule_editor.enable_check.setChecked(True)
        dialog._apply_schedule_to_selected()
        assert rs.is_enabled_config(config["fallback_keys"][0]["disable_schedule"])
        # Key Pools checkbox → the Schedules switch, marker and note
        dialog.use_fallback_checkbox.setChecked(True)
        assert check.isChecked() and item("fallback").startswith("✅") and not hint.isVisibleTo(dialog)
        assert not dialog.add_fallback_frame.isHidden() and config["use_fallback_keys"] is True
        # an unbuilt pool: switching it on builds its section and uses its own checkbox
        assert dialog.select_schedule_pool("metadata")
        assert not check.isChecked() and hint.isVisibleTo(dialog)
        check.setChecked(True)
        assert dialog.use_metadata_keys_checkbox.isChecked() and config["use_metadata_keys"] is True
        assert item("metadata").startswith("✅") and saved
        dialog.use_metadata_keys_checkbox.setChecked(False)
        assert not check.isChecked() and item("metadata").startswith("⛔")
        tick = next(label for label in check.findChildren(mkm.QLabel) if label.text() == "✓")
        assert tick.isHidden()                              # the switch's ✓ overlay follows too
        # another pool toggled on 🔑 Key Pools: only its marker changes
        dialog.use_fallback_checkbox.setChecked(False)
        assert item("fallback").startswith("⛔") and not check.isChecked()
        dialog.use_metadata_keys_checkbox.setChecked(True)
        assert check.isChecked() and item("fallback").startswith("⛔")
        assert not tick.isHidden()
        # a config change synced from elsewhere (unbuilt pool) shows too
        assert "qa_scan" in _pending_pools(dialog) and item("qa_scan").startswith("⛔")
        config[key_pool_service.POOL_SPECS["qa_scan"]["toggle_key"]] = True
        dialog._refresh_pool_from_config("qa_scan")
        assert item("qa_scan").startswith("✅")
        # the picker keeps its Halgakos arrow
        assert "down-arrow" in combo.styleSheet() and "Halgakos" in combo.styleSheet()
    finally:
        _close(app, dialog)
        os.environ.clear()
        os.environ.update(env)
