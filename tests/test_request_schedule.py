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


# ------------------------------------------------------------------------------------- desktop UI
def test_schedule_dialog_edits_presets_and_zones(monkeypatch):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", os.environ.get("QT_QPA_PLATFORM", "offscreen"))
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    import multi_api_key_manager as mkm
    if not mkm.HAS_GUI:
        pytest.skip("key manager imported headless")
    # no schedule yet: the DeepSeek peak hours are suggested (kept in UTC)
    dialog = mkm.RequestScheduleDialog(None, None, mode="queue", subject="deepseek-chat")
    assert dialog._schedule() == DEEPSEEK_PEAK_SCHEDULE
    assert "Suggested" in dialog.preview_label.text() and "UTC" in dialog.preview_label.text()
    dialog._save()
    assert dialog.result_config == rs.to_config(DEEPSEEK_PEAK_SCHEDULE)
    # switching "Times are in" rewrites the same instants in that zone
    dialog = mkm.RequestScheduleDialog(None, rs.to_config(DEEPSEEK_PEAK_SCHEDULE), mode="disable")
    dialog._set_combo_zone(dialog.zone_combo, "Asia/Tokyo")
    dialog._on_zone_changed()
    assert dialog._schedule() == Schedule((Window(600, 780), Window(900, 1140)), rs.WEEKDAYS, "Asia/Tokyo")
    # add / remove windows, days, clear
    dialog._add_row(20 * 60, 2 * 60)
    assert dialog._schedule().windows[-1] == Window(1200, 120)
    dialog._remove_row(dialog._rows[-1])
    dialog.day_checks[5].setChecked(True)
    assert 5 in dialog._schedule().weekdays
    dialog._clear_and_accept()
    assert dialog.result_config == {}
    app.processEvents()


def test_menu_entries_are_wired():
    manager = (SRC / "multi_api_key_manager.py").read_text(encoding="utf-8")
    assert 'selected_label("🗓️ Disable schedule…")' in manager
    assert 'selected_label("⏳ Request Queue…")' in manager
    assert manager.count("disable_schedule_callback=lambda") == 4      # main, fallback, glossary, dedicated
    assert manager.count("queue_schedule_callback=lambda") == 3        # every pool with a key rotation
    gui = (SRC / "translator_gui.py").read_text(encoding="utf-8-sig")
    assert 'menu.addAction("⏳ Request Queue…")' in gui and "def _edit_main_request_queue" in gui
