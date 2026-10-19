"""Time-based API key scheduling: "🗓️ Disable schedule" and "⏳ Request Queue" (GUI-free core).

A *schedule* is a list of daily time windows, the weekdays they apply to and the timezone they
are written in. Two features use it:

* **Disable schedule** (per pool key, ``disable_schedule``): during a window the key counts as
  disabled (``key_contexts.key_enabled_for_context`` skips it exactly like a manually disabled
  key); outside the windows it is enabled again. The stored ``enabled`` flag is never touched.
* **Request Queue** (per pool key, ``queue_schedule``; the main GUI key via the config key
  ``request_queue_schedule`` / env ``REQUEST_QUEUE_SCHEDULE``): requests that would use the key
  WAIT before sending until its window ends. A pool prefers keys outside any window and only
  waits when every usable key is queued (until the earliest window end).

The default suggestion is :data:`DEEPSEEK_PEAK_SCHEDULE` (DeepSeek peak pricing: 01:00-04:00
and 06:00-10:00 UTC, Monday-Friday; written in UTC so it stays right across the user's DST).

Public API (shared by the desktop dialogs and Glossarion Mobile)
------------------------------------------------------------------
Model
    ``Window(start, end)``          minutes after midnight; ``end <= start`` wraps past midnight
                                    (``end == start`` is a whole 24 h window). The weekday is
                                    the day the window STARTS on.
    ``Schedule(windows, weekdays, timezone)``  weekdays: 0 = Monday ... 6 = Sunday;
                                    timezone: ``"UTC"``, ``"local"`` (the system zone) or an
                                    IANA name (``"Asia/Tokyo"``).
    ``DEEPSEEK_PEAK_SCHEDULE``, ``WEEKDAY_NAMES``, ``ALL_WEEKDAYS``, ``WEEKDAYS``
Config (JSON shape ``{"windows": [{"start": "01:00", "end": "04:00"}, ...],
"weekdays": [0, 1, 2, 3, 4], "timezone": "UTC"}``; ``{}`` / None = no schedule)
    ``from_config(value, strict=False)`` -> Schedule or None (strict raises ``ScheduleError``)
    ``to_config(schedule)`` -> dict (``{}`` for None), ``validate(value)`` -> [error, ...]
    ``normalize_config(value)`` -> the canonical dict (``{}`` when empty/invalid)
    ``parse_hhmm("7:05")`` -> 425, ``format_hhmm(425)`` -> ``"07:05"``
Time checks (``now``: aware datetime, epoch seconds or None for the current time)
    ``is_in_window(schedule, now)``, ``current_window(schedule, now)`` -> (start, end) UTC,
    ``next_window_end(schedule, now)`` -> aware UTC datetime when the current window (and any
    window back-to-back with it) ends, None outside a window;
    ``next_window_start(schedule, now)`` -> the next start after *now* (UTC) or None.
Display
    ``describe(schedule, tz="local", now=None)``  e.g.
    ``"10:00–13:00, 15:00–19:00 (local), Mon–Fri"``; ``convert(schedule, tz, now)`` (same
    instants written in another zone; ``ScheduleError`` when the windows would need different
    weekday shifts); ``zone_label(tz, now)``; ``format_clock(dt, tz)``; ``format_wait(seconds)``;
    ``resolve_tz(name)``; ``common_timezones()``; ``zone_is_known(name)``.
Editors (desktop RequestScheduleDialog, mobile ScheduleSheet)
    ``EDITOR_MODES`` (mode -> title, explanation), ``ENABLE_LABELS`` (mode -> the off-by-default
    on/off switch), ``OFF_PREVIEW_LINE``, ``DEEPSEEK_PRESET_HELP``,
    ``editable_config(value)`` -> (schedule or None, enabled): the saved windows even while
    switched off; ``to_config(schedule, enabled=False)`` saves them switched off,
    ``zone_choices(include_local=True)`` -> [(label, zone)], ``preview_lines(schedule,
    extra_zone, suggested=False)`` -> the preview text lines; ``editor_preview_lines(schedule,
    extra_zone, enabled=..., suggested=...)`` puts the always-present ``ON_PREVIEW_LINE`` /
    ``OFF_PREVIEW_LINE`` status line first (switching on / off never changes the line count).
Pool keys: one schedule per key (desktop ⏰ Schedules tab, mobile Keys ⋯ "⏰ Schedule…"; mode "key")
    ``SCHEDULE_KINDS`` ("disable", "queue"), ``KIND_FIELDS``, ``KIND_LABEL`` ("During the
    windows:"), ``KIND_CHOICES``, ``pool_schedule_kinds(pool)`` (Fallback: "disable" only,
    ``QUEUE_UNAVAILABLE_HELP``), ``key_schedule_config(key)`` -> (kind, config) of the field it
    holds, ``set_key_schedule(key, kind, config)`` writes that field and removes the other,
    ``key_schedule_overview(key)`` -> "Off" / the state / the windows for a key list row.
Keys (dicts or ``APIKeyEntry`` objects)
    ``DISABLE_FIELD`` / ``QUEUE_FIELD``; ``key_schedule(key, field)``;
    ``key_disabled_by_schedule(key, now)``; ``key_queued_now(key, now)``;
    ``key_schedule_summary(key, tz="local", now=None)`` (the key list status/tooltip);
    ``pool_queue_wait_until(keys, now, enabled=None)`` -> None when some usable key is outside
    its queue window (send now), else the earliest window end to wait for;
    ``queue_deferred(key, peers, now, enabled=None)`` -> a queued key that key selection must
    skip because a usable peer is outside every window.
Main key
    ``MAIN_QUEUE_CONFIG_KEY`` (``request_queue_schedule``), ``MAIN_QUEUE_ENV``
    (``REQUEST_QUEUE_SCHEDULE``), ``main_queue_schedule_from_env()``.
Waiting
    ``QueueWaiter`` / module singleton ``WAITER``: ``WAITER.now()``,
    ``WAITER.wait_until(target, stop_check=..., log=..., label=...)`` -> True when the window
    ended, False when *stop_check* asked to stop. One shared start line and one progress line
    every ``LOG_INTERVAL_SECONDS`` (3 minutes) whatever the number of waiting threads.
    ``is_thread_waiting(thread_or_ident)`` / ``thread_wait_seconds(ident)`` let callers' own
    timeout timers (chunk timeout, ...) leave the wait out.
"""
from __future__ import annotations

import json
import math
import os
import threading
import time
from datetime import date, datetime, timedelta, timezone, tzinfo
from typing import Callable, Iterable, NamedTuple, Optional

__all__ = [
    "ALL_WEEKDAYS", "DEEPSEEK_PEAK_SCHEDULE", "DEEPSEEK_PRESET_HELP", "DISABLE_FIELD", "EDITOR_MODES",
    "ENABLE_LABELS", "OFF_PREVIEW_LINE", "ON_PREVIEW_LINE", "editable_config", "editor_preview_lines",
    "is_enabled_config", "KIND_CHOICES", "KIND_FIELDS", "KIND_LABEL", "QUEUE_UNAVAILABLE_HELP",
    "SCHEDULE_KINDS", "key_schedule_config", "key_schedule_overview", "pool_schedule_kinds",
    "set_key_schedule",
    "LOG_INTERVAL_SECONDS",
    "MAIN_QUEUE_CONFIG_KEY", "MAIN_QUEUE_ENV", "QUEUE_FIELD", "QueueWaiter", "Schedule",
    "ScheduleError", "WAITER", "WEEKDAYS", "WEEKDAY_NAMES", "Window", "common_timezones", "convert",
    "current_window", "describe", "format_clock", "format_hhmm", "format_wait", "from_config",
    "is_in_window", "is_thread_waiting", "key_disabled_by_schedule", "key_queued_now",
    "key_schedule", "key_schedule_summary", "main_queue_schedule_from_env", "next_window_end",
    "next_window_start", "normalize_config", "parse_hhmm", "pool_queue_wait_until", "preview_lines",
    "queue_deferred", "resolve_tz", "thread_wait_seconds", "to_config", "validate", "zone_label",
]

WEEKDAY_NAMES = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
ALL_WEEKDAYS = frozenset(range(7))
WEEKDAYS = frozenset(range(5))
DISABLE_FIELD = "disable_schedule"
QUEUE_FIELD = "queue_schedule"
MAIN_QUEUE_CONFIG_KEY = "request_queue_schedule"
MAIN_QUEUE_ENV = "REQUEST_QUEUE_SCHEDULE"
LOG_INTERVAL_SECONDS = 180.0
_MINUTES_PER_DAY = 24 * 60
_MAX_CHAIN = 16  # back-to-back windows followed by next_window_end (> two weeks of whole days)


class ScheduleError(ValueError):
    """An invalid schedule (strict parsing / conversion)."""


class Window(NamedTuple):
    start: int  # minutes after midnight, 0..1439
    end: int    # minutes after midnight; <= start wraps past midnight

    @property
    def wraps(self) -> bool:
        return self.end <= self.start

    def label(self) -> str:
        return f"{format_hhmm(self.start)}–{format_hhmm(self.end)}"


class Schedule(NamedTuple):
    windows: tuple
    weekdays: frozenset
    timezone: str = "local"


DEEPSEEK_PEAK_SCHEDULE = Schedule((Window(60, 240), Window(360, 600)), WEEKDAYS, "UTC")


# --------------------------------------------------------------------------------- time zones
class _SystemLocal(tzinfo):
    """The operating system's local zone, DST-aware through ``time.localtime`` (the stdlib docs'
    LocalTimezone). Used for ``"local"`` and when an IANA zone cannot be loaded."""

    def _offset(self, dt):
        if dt is None:
            return timedelta(seconds=-time.timezone)
        stamp = (dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, dt.weekday(), 0, -1)
        try:
            local = time.localtime(time.mktime(stamp))
            return timedelta(seconds=local.tm_gmtoff)
        except (OverflowError, ValueError, OSError):
            return timedelta(seconds=-time.timezone)

    def utcoffset(self, dt):
        return self._offset(dt)

    def dst(self, dt):
        return timedelta(0)

    def tzname(self, dt):
        return "local"

    def fromutc(self, dt):
        stamp = (dt.replace(tzinfo=timezone.utc) - datetime(1970, 1, 1, tzinfo=timezone.utc)).total_seconds()
        try:
            local = time.localtime(stamp)
            offset = timedelta(seconds=local.tm_gmtoff)
        except (OverflowError, ValueError, OSError):
            offset = timedelta(seconds=-time.timezone)
        return (dt + offset).replace(tzinfo=self)

    def __repr__(self):
        return "_SystemLocal()"


SYSTEM_LOCAL = _SystemLocal()
_TZ_CACHE: dict = {}


def resolve_tz(name) -> tzinfo:
    """A tzinfo for ``"UTC"``, ``"local"`` / empty (the system zone) or an IANA name. A zone that
    cannot be loaded (no tzdata, typo) falls back to the system local zone."""
    if isinstance(name, tzinfo):
        return name
    key = str(name or "local").strip()
    if key.lower() in ("", "local", "system"):
        return SYSTEM_LOCAL
    if key.upper() in ("UTC", "Z", "GMT", "ETC/UTC"):
        return timezone.utc
    cached = _TZ_CACHE.get(key)
    if cached is not None:
        return cached
    zone = None
    try:
        from zoneinfo import ZoneInfo
        zone = ZoneInfo(key)
    except Exception:
        zone = None
    if zone is None:
        zone = SYSTEM_LOCAL
    _TZ_CACHE[key] = zone
    return zone


def zone_is_known(name) -> bool:
    """Whether *name* names a zone this machine can load (``"local"`` and ``"UTC"`` always can)."""
    key = str(name or "local").strip()
    if key.lower() in ("", "local", "system") or key.upper() in ("UTC", "Z", "GMT", "ETC/UTC"):
        return True
    try:
        from zoneinfo import ZoneInfo
        ZoneInfo(key)
        return True
    except Exception:
        return False


def common_timezones():
    """Zones offered by the schedule editors besides Local and UTC."""
    return (
        "Asia/Shanghai", "Asia/Tokyo", "Asia/Seoul", "Asia/Singapore", "Asia/Kolkata", "Asia/Dubai",
        "Europe/London", "Europe/Berlin", "Europe/Moscow", "America/New_York", "America/Chicago",
        "America/Los_Angeles", "America/Sao_Paulo", "Australia/Sydney",
    )


def _utc(now=None) -> datetime:
    if now is None:
        return datetime.now(timezone.utc)
    if isinstance(now, (int, float)):
        return datetime.fromtimestamp(float(now), timezone.utc)
    if isinstance(now, datetime):
        if now.tzinfo is None or now.tzinfo.utcoffset(now) is None:
            now = now.astimezone()  # naive = the system's local time
        return now.astimezone(timezone.utc)
    raise TypeError(f"unsupported time value: {now!r}")


def _at(day: date, minutes: int, tz: tzinfo) -> datetime:
    """Wall-clock *minutes* on *day* in *tz*, as an aware UTC datetime."""
    hours, mins = divmod(int(minutes) % _MINUTES_PER_DAY, 60)
    return datetime(day.year, day.month, day.day, hours, mins, tzinfo=tz).astimezone(timezone.utc)


def _offset_text(offset: timedelta) -> str:
    total = int(offset.total_seconds() // 60)
    sign = "+" if total >= 0 else "-"
    hours, mins = divmod(abs(total), 60)
    return f"UTC{sign}{hours}" + (f":{mins:02d}" if mins else "")


def zone_label(tz="local", now=None) -> str:
    """``"local, UTC+9"``, ``"UTC"`` or ``"Asia/Tokyo, UTC+9"`` for logs and previews."""
    zone = resolve_tz(tz)
    if zone is timezone.utc:
        return "UTC"
    name = "local" if zone is SYSTEM_LOCAL else str(tz)
    offset = _utc(now).astimezone(zone).utcoffset() or timedelta(0)
    return f"{name}, {_offset_text(offset)}"


def format_clock(moment, tz="local", now=None) -> str:
    """``"15:00"`` (or ``"15:00 Mon"`` when *moment* is on another day than *now*) in *tz*."""
    zone = resolve_tz(tz)
    local = _utc(moment).astimezone(zone)
    text = local.strftime("%H:%M")
    ref = _utc(now).astimezone(zone)
    if local.date() != ref.date():
        text += f" {WEEKDAY_NAMES[local.weekday()]}"
    return text


def format_wait(seconds, *, round_up=False) -> str:
    """``"4h 59m"``: hours and two-digit minutes (floored, or rounded up for a time left)."""
    seconds = max(0.0, float(seconds or 0))
    minutes = int(math.ceil(seconds / 60.0)) if round_up else int(seconds // 60)
    hours, mins = divmod(minutes, 60)
    return f"{hours}h {mins:02d}m"


# ------------------------------------------------------------------------------- (de)serialise
def parse_hhmm(text) -> int:
    """``"7:05"`` / ``"07:05"`` / ``"24:00"`` -> minutes after midnight (24:00 = 0)."""
    if isinstance(text, bool):
        raise ScheduleError(f"invalid time {text!r}")
    if isinstance(text, int):
        if 0 <= text <= _MINUTES_PER_DAY:
            return text % _MINUTES_PER_DAY
        raise ScheduleError(f"invalid time {text!r}")
    raw = str(text or "").strip()
    parts = raw.split(":")
    if len(parts) != 2 or not all(p.strip().isdigit() for p in parts):
        raise ScheduleError(f"invalid time {raw!r} (use HH:MM)")
    hours, mins = int(parts[0]), int(parts[1])
    if (hours == 24 and mins == 0):
        return 0
    if not (0 <= hours < 24 and 0 <= mins < 60):
        raise ScheduleError(f"invalid time {raw!r} (use HH:MM, 00:00-23:59)")
    return hours * 60 + mins


def format_hhmm(minutes) -> str:
    hours, mins = divmod(int(minutes) % _MINUTES_PER_DAY, 60)
    return f"{hours:02d}:{mins:02d}"


def _parse_weekdays(value) -> frozenset:
    if value is None:
        return ALL_WEEKDAYS
    if isinstance(value, str):
        value = [v for v in value.replace(";", ",").split(",") if v.strip()]
    out = set()
    for item in value:
        if isinstance(item, bool):
            raise ScheduleError(f"invalid weekday {item!r}")
        if isinstance(item, int):
            day = item
        else:
            text = str(item).strip()
            if text.isdigit():
                day = int(text)
            else:
                names = [n.lower() for n in WEEKDAY_NAMES]
                if text[:3].lower() not in names:
                    raise ScheduleError(f"invalid weekday {item!r}")
                day = names.index(text[:3].lower())
        if not 0 <= day <= 6:
            raise ScheduleError(f"invalid weekday {item!r} (0 = Monday ... 6 = Sunday)")
        out.add(day)
    return frozenset(out)


def _parse(value, honor_enabled=True) -> Optional[Schedule]:
    if value is None or value == "" or value == {} or value == []:
        return None
    if isinstance(value, Schedule):
        return value if value.windows and value.weekdays else None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception as exc:
            raise ScheduleError(f"invalid schedule JSON: {exc}")
        return _parse(value, honor_enabled)
    if not isinstance(value, dict):
        raise ScheduleError("a schedule is a JSON object")
    if honor_enabled and not is_enabled_config(value):
        return None
    raw_windows = value.get("windows") or []
    if not isinstance(raw_windows, (list, tuple)):
        raise ScheduleError("'windows' must be a list")
    windows = []
    for raw in raw_windows:
        if isinstance(raw, dict):
            start, end = raw.get("start"), raw.get("end")
        elif isinstance(raw, (list, tuple)) and len(raw) == 2:
            start, end = raw
        else:
            raise ScheduleError(f"invalid window {raw!r}")
        window = Window(parse_hhmm(start), parse_hhmm(end))
        if window not in windows:
            windows.append(window)
    tz_name = str(value.get("timezone") or "local").strip() or "local"
    if not zone_is_known(tz_name):
        raise ScheduleError(f"unknown timezone {tz_name!r}")
    weekdays = _parse_weekdays(value.get("weekdays"))
    if not windows or not weekdays:
        return None
    return Schedule(tuple(windows), weekdays, tz_name)


def is_enabled_config(value) -> bool:
    """False when the config value is switched off ("enabled": false). Schedules are off unless
    the user turned them on in the editor; a value without the flag (older configs) counts as on."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception:
            return True
    return not (isinstance(value, dict) and value.get("enabled", True) is False)


def editable_config(value):
    """(schedule, enabled) for an editor: the saved windows even while the schedule is switched
    off (None when nothing valid is saved) and whether it is switched on."""
    try:
        schedule = _parse(value, honor_enabled=False)
    except ScheduleError:
        schedule = None
    return schedule, bool(schedule) and is_enabled_config(value)


def from_config(value, strict=False) -> Optional[Schedule]:
    """The Schedule a config value holds, or None (empty / disabled / no windows / no weekdays).
    Invalid values raise :class:`ScheduleError` when *strict*, else give None."""
    try:
        return _parse(value)
    except ScheduleError:
        if strict:
            raise
        return None


def validate(value) -> list:
    """Error messages for a config value ([] when valid or empty)."""
    try:
        _parse(value)
    except ScheduleError as exc:
        return [str(exc)]
    return []


def to_config(schedule, enabled=True) -> dict:
    """The JSON-safe config dict of *schedule* ({} for None). ``enabled=False`` keeps the windows
    but switches the schedule off ("enabled": false)."""
    schedule = from_config(schedule) if not isinstance(schedule, Schedule) else schedule
    if not schedule or not schedule.windows or not schedule.weekdays:
        return {}
    config = {
        "windows": [{"start": format_hhmm(w.start), "end": format_hhmm(w.end)} for w in schedule.windows],
        "weekdays": sorted(schedule.weekdays),
        "timezone": schedule.timezone or "local",
    }
    if not enabled:
        config["enabled"] = False
    return config


def normalize_config(value) -> dict:
    """The canonical config dict of *value* ({} when empty or invalid); a switched-off schedule
    keeps its windows and "enabled": false."""
    schedule, enabled = editable_config(value)
    return to_config(schedule, enabled=enabled)


_COERCE_CACHE: dict = {}


def _coerce(value) -> Optional[Schedule]:
    """Lenient, cached parse for hot paths (key selection runs this per key per request)."""
    if value is None or value == {} or value == "":
        return None
    if isinstance(value, Schedule):
        return value if value.windows and value.weekdays else None
    try:
        cache_key = value if isinstance(value, str) else json.dumps(value, sort_keys=True, default=str)
    except Exception:
        return from_config(value)
    try:
        return _COERCE_CACHE[cache_key]
    except KeyError:
        pass
    parsed = from_config(value)
    if len(_COERCE_CACHE) > 512:
        _COERCE_CACHE.clear()
    _COERCE_CACHE[cache_key] = parsed
    return parsed


# ---------------------------------------------------------------------------------- time checks
def _occurrences(schedule: Schedule, moment: datetime):
    """``(start, end)`` UTC occurrences of *schedule* that contain *moment* (aware UTC)."""
    tz = resolve_tz(schedule.timezone)
    local_day = moment.astimezone(tz).date()
    found = []
    for back in (0, 1):
        day = local_day - timedelta(days=back)
        if day.weekday() not in schedule.weekdays:
            continue
        for window in schedule.windows:
            if back and not window.wraps:
                continue
            start = _at(day, window.start, tz)
            end = _at(day + timedelta(days=1) if window.wraps else day, window.end, tz)
            if start <= moment < end:
                found.append((start, end))
    return found


def current_window(schedule, now=None):
    """``(start, end)`` (aware UTC) of the window containing *now*, or None. Overlapping windows
    give the earliest start and the latest end."""
    schedule = _coerce(schedule)
    if not schedule:
        return None
    moment = _utc(now)
    found = _occurrences(schedule, moment)
    if not found:
        return None
    return min(s for s, _ in found), max(e for _, e in found)


def is_in_window(schedule, now=None) -> bool:
    """Whether *now* falls inside one of *schedule*'s windows (start inclusive, end exclusive)."""
    schedule = _coerce(schedule)
    return bool(schedule) and bool(_occurrences(schedule, _utc(now)))


def next_window_end(schedule, now=None) -> Optional[datetime]:
    """When the window containing *now* ends (aware UTC), following windows that start exactly
    when it ends (back-to-back). None when *now* is outside every window."""
    schedule = _coerce(schedule)
    if not schedule:
        return None
    moment = _utc(now)
    found = _occurrences(schedule, moment)
    if not found:
        return None
    end = max(e for _, e in found)
    for _ in range(_MAX_CHAIN):
        found = _occurrences(schedule, end)
        if not found:
            break
        end = max(e for _, e in found)
    return end


def next_window_start(schedule, now=None) -> Optional[datetime]:
    """The next window start strictly after *now* (aware UTC), or None without windows."""
    schedule = _coerce(schedule)
    if not schedule:
        return None
    moment = _utc(now)
    tz = resolve_tz(schedule.timezone)
    local_day = moment.astimezone(tz).date()
    best = None
    for delta in range(-1, 9):
        day = local_day + timedelta(days=delta)
        if day.weekday() not in schedule.weekdays:
            continue
        for window in schedule.windows:
            start = _at(day, window.start, tz)
            if start > moment and (best is None or start < best):
                best = start
    return best


# ---------------------------------------------------------------------------------------- display
def _day_runs(days) -> str:
    """``"Mon–Fri"``, ``"Sun–Thu"`` (runs may wrap past Sunday), ``"Sat, Sun"``, ``"every day"``."""
    days = set(days)
    if len(days) == 7:
        return "every day"
    if not days:
        return "no days"
    first = min(d for d in days if (d - 1) % 7 not in days)
    ordered = [(first + i) % 7 for i in range(7) if (first + i) % 7 in days]
    runs, run = [], [ordered[0]]
    for day in ordered[1:]:
        if day == (run[-1] + 1) % 7:
            run.append(day)
        else:
            runs.append(run)
            run = [day]
    runs.append(run)
    parts = []
    for run in runs:
        if len(run) >= 3:
            parts.append(f"{WEEKDAY_NAMES[run[0]]}–{WEEKDAY_NAMES[run[-1]]}")
        else:
            parts.extend(WEEKDAY_NAMES[d] for d in run)
    return ", ".join(parts)


def _same_zone(a, b) -> bool:
    return resolve_tz(a) is resolve_tz(b)


def _reference_day(schedule: Schedule, now) -> date:
    """A day of *schedule* near *now* (its own zone) used to convert wall times between zones."""
    tz = resolve_tz(schedule.timezone)
    today = _utc(now).astimezone(tz).date()
    for delta in range(7):
        day = today + timedelta(days=delta)
        if day.weekday() in schedule.weekdays:
            return day
    return today


def _converted_windows(schedule: Schedule, tz, now):
    """``[(Window in tz, day shift), ...]`` for *schedule* written in *tz* (around *now*)."""
    if _same_zone(schedule.timezone, tz):
        return [(w, 0) for w in schedule.windows]
    target = resolve_tz(tz)
    day = _reference_day(schedule, now)
    source = resolve_tz(schedule.timezone)
    out = []
    for window in schedule.windows:
        start_utc = _at(day, window.start, source)
        end_utc = _at(day + timedelta(days=1) if window.wraps else day, window.end, source)
        start_local = start_utc.astimezone(target)
        end_local = end_utc.astimezone(target)
        shift = (start_local.date() - day).days
        out.append((Window(start_local.hour * 60 + start_local.minute,
                           end_local.hour * 60 + end_local.minute), shift))
    return out


def convert(schedule, tz, now=None) -> Optional[Schedule]:
    """*schedule* rewritten in zone *tz* (the same instants around *now*). Raises
    :class:`ScheduleError` when its windows would land on different weekday shifts."""
    schedule = _coerce(schedule)
    if not schedule:
        return None
    converted = _converted_windows(schedule, tz, now)
    shifts = {shift for _, shift in converted}
    if len(shifts) > 1:
        raise ScheduleError("these windows fall on different days in that timezone")
    shift = shifts.pop() if shifts else 0
    weekdays = frozenset((d + shift) % 7 for d in schedule.weekdays)
    name = "local" if resolve_tz(tz) is SYSTEM_LOCAL else ("UTC" if resolve_tz(tz) is timezone.utc else str(tz))
    return Schedule(tuple(w for w, _ in converted), weekdays, name)


def describe(schedule, tz="local", now=None) -> str:
    """``"10:00–13:00, 15:00–19:00 (local), Mon–Fri"``: *schedule* shown in zone *tz*."""
    schedule = _coerce(schedule)
    if not schedule:
        return "No schedule"
    zone = resolve_tz(tz)
    label = "local" if zone is SYSTEM_LOCAL else ("UTC" if zone is timezone.utc else str(tz))
    groups = []  # [(weekday set, [window labels])] in window order
    for window, shift in _converted_windows(schedule, tz, now):
        days = frozenset((d + shift) % 7 for d in schedule.weekdays)
        for group_days, labels in groups:
            if group_days == days:
                labels.append(window.label())
                break
        else:
            groups.append((days, [window.label()]))
    return "; ".join(f"{', '.join(labels)} ({label}), {_day_runs(days)}" for days, labels in groups)


# ------------------------------------------------------------------------------------------ editors
#: The schedule editors' modes (desktop RequestScheduleDialog, mobile ScheduleSheet):
#: mode -> (title, explanation).
EDITOR_MODES = {
    "disable": ("🗓️ Disable schedule",
                "During these windows the key counts as disabled: requests skip it exactly like a "
                "manually disabled key. Outside them it is enabled again (the Enabled setting is not "
                "changed)."),
    "queue": ("⏳ Request Queue",
              "During these windows the key is not skipped, but requests that would use it wait until "
              "the window ends. Other usable keys outside their windows are preferred; requests only "
              "wait when every usable key is queued or disabled (until the earliest window end)."),
    "main_queue": ("⏳ Request Queue",
                   "During these windows requests that use the main API key wait before sending until "
                   "the window ends (logged every 3 minutes; Stop ends the wait). Translation, glossary "
                   "extraction and the other runs follow it. With Translation Keys (multi-key mode) on, "
                   "each pool key's own Request Queue applies instead."),
    # pool keys (Multi API Key Manager ⏰ Schedules tab, mobile Keys ⋯ "⏰ Schedule…"): one schedule
    # per key, its behaviour chosen under "During the windows:" (SCHEDULE_KINDS)
    "key": ("⏰ Schedule",
            "Time windows for the selected keys. Choose what happens during them: the key is skipped "
            "(other keys take over) or requests wait until the window ends. A key holds one schedule."),
}
#: The editors' on/off switch (off by default: nothing is scheduled until the user turns it on).
ENABLE_LABELS = {
    "disable": "🗓️ Enable Disable schedule",
    "queue": "⏳ Enable Request Queue",
    "main_queue": "⏳ Enable Request Queue",
    "key": "⏰ Enable schedule",
}
#: The preview's first line says whether the schedule is on; it is always there, so switching the
#: schedule on / off never changes the preview's height.
OFF_PREVIEW_LINE = "Off: saving keeps these windows but nothing is scheduled until this is enabled."
ON_PREVIEW_LINE = "On: requests follow these windows once saved (switch off to keep them unused)."
#: The per-key behaviours ("During the windows:"): kind -> key field / choice label.
SCHEDULE_KINDS = ("disable", "queue")
KIND_FIELDS = {"disable": DISABLE_FIELD, "queue": QUEUE_FIELD}
KIND_LABEL = "During the windows:"
KIND_CHOICES = {
    "disable": "🗓️ Disable the key (skip it; other keys take over)",
    "queue": "⏳ Queue requests (wait until the window ends)",
}
QUEUE_UNAVAILABLE_HELP = ("Fallback keys are tried one by one (no rotation), so there is nothing to queue "
                          "in: use 🗓️ Disable the key.")
DEEPSEEK_PRESET_HELP = ("DeepSeek peak pricing: 01:00–04:00 and 06:00–10:00 UTC, Monday–Friday "
                        "(off-peak costs half). Kept in UTC so it stays right when your clocks change.")


def zone_choices(include_local=True, now=None):
    """``[(label, zone), ...]`` the editors' zone pickers offer: Local (with its offset), UTC and
    the :func:`common_timezones` this machine can load."""
    out = [(f"Local ({zone_label('local', now)})", "local")] if include_local else []
    out.append(("UTC", "UTC"))
    out.extend((name, name) for name in common_timezones() if zone_is_known(name))
    return out


def preview_lines(schedule, extra_zone=None, *, suggested=False, now=None):
    """The editors' preview: the windows in local time, UTC and *extra_zone*, then whether *now*
    is inside a window (until when) or when the next one starts. *suggested* adds the "no
    schedule saved yet" note of a prefilled DeepSeek preset."""
    schedule = _coerce(schedule) if not isinstance(schedule, Schedule) else schedule
    if not schedule or not schedule.windows or not schedule.weekdays:
        return ["No windows or no days: saving removes the schedule."]
    moment = _utc(now)
    zones = ["local", "UTC"]
    if extra_zone and extra_zone not in zones and zone_is_known(extra_zone):
        zones.append(extra_zone)
    lines = ["Suggested: DeepSeek peak hours (no schedule saved yet)."] if suggested else []
    for zone in zones:
        lines.append(f"• {zone_label(zone, moment)}: {describe(schedule, zone, moment)}")
    end = next_window_end(schedule, moment)
    if end is not None:
        lines.append(f"Now: inside a window until {format_clock(end, 'local', moment)} local.")
    else:
        start = next_window_start(schedule, moment)
        if start is not None:
            lines.append(f"Now: outside the windows; the next one starts {format_clock(start, 'local', moment)} local.")
    return lines


def editor_preview_lines(schedule, extra_zone=None, *, enabled=False, suggested=False, now=None):
    """The editors' preview text: the on / off status line (:data:`ON_PREVIEW_LINE` /
    :data:`OFF_PREVIEW_LINE`, always present) followed by :func:`preview_lines`."""
    status = ON_PREVIEW_LINE if enabled else OFF_PREVIEW_LINE
    return [status] + preview_lines(schedule, extra_zone, suggested=suggested, now=now)


def pool_schedule_kinds(pool) -> tuple:
    """The "During the windows:" behaviours a pool's keys offer: both, except Fallback keys (tried
    one by one, no rotation to queue in), which only have "disable"."""
    return ("disable",) if pool == "fallback" else SCHEDULE_KINDS


# -------------------------------------------------------------------------------------------- keys
def key_schedule(key, field) -> Optional[Schedule]:
    """The Schedule a key (dict or ``APIKeyEntry``) holds in *field*, or None."""
    if key is None:
        return None
    value = key.get(field) if isinstance(key, dict) else getattr(key, field, None)
    return _coerce(value)


def key_disabled_by_schedule(key, now=None) -> bool:
    """Whether the key's 🗓️ Disable schedule disables it at *now*."""
    schedule = key_schedule(key, DISABLE_FIELD)
    return bool(schedule) and is_in_window(schedule, now)


def key_queued_now(key, now=None) -> bool:
    """Whether the key's ⏳ Request Queue window is active at *now*."""
    schedule = key_schedule(key, QUEUE_FIELD)
    return bool(schedule) and is_in_window(schedule, now)


def _usable(key, enabled, now=None) -> bool:
    if enabled is not None:
        try:
            return bool(enabled(key))
        except Exception:
            return False
    if isinstance(key, dict):
        flag = key.get("enabled", True)
    else:
        flag = getattr(key, "enabled", True)
    # the Disable schedule is checked at the caller's moment (not the wall clock)
    return bool(flag) and not key_disabled_by_schedule(key, now)


def pool_queue_wait_until(keys, now=None, enabled=None) -> Optional[datetime]:
    """When a pool must wait for its Request Queue windows: None when some usable key is
    outside its queue window (or no usable key is queued at all), else the earliest window end
    (aware UTC) among the usable queued keys. *enabled(key)* decides "usable" (default: the
    key's enabled flag and Disable schedule)."""
    moment = _utc(now)
    earliest = None
    any_queued = False
    for key in keys or ():
        if not _usable(key, enabled, moment):
            continue
        schedule = key_schedule(key, QUEUE_FIELD)
        if not schedule or not is_in_window(schedule, moment):
            return None
        any_queued = True
        end = next_window_end(schedule, moment)
        if end is not None and (earliest is None or end < earliest):
            earliest = end
    if not any_queued:
        return None
    return earliest


def queue_deferred(key, peers, now=None, enabled=None) -> bool:
    """A key in its Request Queue window that key selection skips because some usable *peer*
    is outside every queue window (those are preferred). When every usable key is queued the
    keys stay selectable and the request waits before sending instead."""
    schedule = key_schedule(key, QUEUE_FIELD)
    if not schedule:
        return False
    moment = _utc(now)
    if not is_in_window(schedule, moment):
        return False
    for peer in peers or ():
        if peer is key or not _usable(peer, enabled, moment):
            continue
        if not key_queued_now(peer, moment):
            return True
    return False


def key_schedule_summary(key, tz="local", now=None):
    """``(status, tooltip_lines)`` for a key list row: status is ``""``, ``"🗓️ Disabled until
    13:00"`` or ``"⏳ Queued until 13:00"``; the tooltip lines describe each schedule."""
    moment = _utc(now)
    status = ""
    lines = []
    disable = key_schedule(key, DISABLE_FIELD)
    queue = key_schedule(key, QUEUE_FIELD)
    if disable:
        lines.append(f"🗓️ Disabled {describe(disable, tz, moment)}")
        if is_in_window(disable, moment):
            status = f"🗓️ Disabled until {format_clock(next_window_end(disable, moment), tz, moment)}"
    if queue:
        lines.append(f"⏳ Request Queue {describe(queue, tz, moment)}")
        if not status and is_in_window(queue, moment):
            status = f"⏳ Queued until {format_clock(next_window_end(queue, moment), tz, moment)}"
    return status, lines


def _raw_field(key, field):
    if key is None:
        return None
    return key.get(field) if isinstance(key, dict) else getattr(key, field, None)


def key_schedule_config(key):
    """``(kind, config)`` of the one schedule a key holds (the editors load it): the field that is
    set ("disable" / "queue"), switched-off configs included; a key carrying both (older configs)
    gives the switched-on one, else the Disable schedule. ``("disable", {})`` without a schedule."""
    found = []
    for kind in SCHEDULE_KINDS:
        config = normalize_config(_raw_field(key, KIND_FIELDS[kind]))
        if config:
            found.append((kind, config))
    for kind, config in found:
        if is_enabled_config(config):
            return kind, config
    return found[0] if found else ("disable", {})


def set_key_schedule(key, kind, config):
    """Store *config* as the key's one schedule: the *kind* field gets it and the other field is
    removed (``{}`` / None removes both). Works on config dicts and ``APIKeyEntry`` objects (an
    object field is set to ``{}``, which ``to_dict`` leaves out). Returns *key*."""
    if kind not in KIND_FIELDS:
        raise ValueError(f"unknown schedule kind {kind!r}")
    config = normalize_config(config)
    for name, field in KIND_FIELDS.items():
        value = dict(config) if (name == kind and config) else None
        if isinstance(key, dict):
            if value:
                key[field] = value
            else:
                key.pop(field, None)
        else:
            setattr(key, field, value or {})
    return key


def key_schedule_overview(key, tz="local", now=None) -> str:
    """One line for a key in the schedule lists: "Off", "Off (🗓️ windows saved)", the current
    state ("⏳ Queued until 13:00") or the windows ("🗓️ 10:00–13:00 (local), Mon–Fri")."""
    kind, config = key_schedule_config(key)
    icon = "🗓️" if kind == "disable" else "⏳"
    schedule, enabled = editable_config(config)
    if not schedule:
        return "Off"
    if not enabled:
        return f"Off ({icon} windows saved)"
    status, _lines = key_schedule_summary(key, tz, now)
    return status or f"{icon} {describe(schedule, tz, now)}"


def main_queue_schedule_from_env(environ=None) -> Optional[Schedule]:
    """The main GUI key's Request Queue schedule exported by run_env (``REQUEST_QUEUE_SCHEDULE``)."""
    environ = os.environ if environ is None else environ
    return _coerce(environ.get(MAIN_QUEUE_ENV, "") or None)


# ------------------------------------------------------------------------------------------ waiting
class QueueWaiter:
    """Shared Request Queue waiter: threads block in :meth:`wait_until`; the log shows one start
    line per target and one progress line per *log_interval* for all threads together."""

    def __init__(self, clock: Callable[[], float] = time.time, sleep: Callable[[float], None] = time.sleep,
                 log_interval: float = LOG_INTERVAL_SECONDS, poll: float = 0.5):
        self.clock = clock
        self.sleep = sleep
        self.log_interval = float(log_interval)
        self.poll = float(poll)
        self._lock = threading.Lock()
        self._waiting = {}        # thread ident -> target timestamp
        self._totals = {}         # thread ident -> seconds waited (cumulative)
        self._episode_start = None
        self._announced = set()   # targets announced in this episode
        self._last_log = None
        self._current_start = {}  # thread ident -> start of its current wait

    def now(self) -> datetime:
        return datetime.fromtimestamp(self.clock(), timezone.utc)

    def is_waiting(self, ident=None) -> bool:
        ident = threading.get_ident() if ident is None else ident
        with self._lock:
            return ident in self._waiting

    def any_waiting(self) -> bool:
        with self._lock:
            return bool(self._waiting)

    def waited_seconds(self, ident=None) -> float:
        """Seconds the thread spent waiting so far (finished waits plus the current one)."""
        ident = threading.get_ident() if ident is None else ident
        with self._lock:
            total = self._totals.get(ident, 0.0)
            current = self._current_start.get(ident)
        if current is not None:
            total += max(0.0, self.clock() - current)
        return total

    def _start_line(self, target_ts, label, now_ts):
        target = datetime.fromtimestamp(target_ts, timezone.utc)
        now = datetime.fromtimestamp(now_ts, timezone.utc)
        who = f" ({label})" if label else ""
        return (f"⏳ Request Queue{who}: waiting {format_wait(target_ts - now_ts, round_up=True)} — requests start at "
                f"{format_clock(target, 'local', now)} ({zone_label('local', now)}) / "
                f"{format_clock(target, 'UTC', now)} UTC")

    def _progress_line(self, target_ts, label, now_ts):
        target = datetime.fromtimestamp(target_ts, timezone.utc)
        now = datetime.fromtimestamp(now_ts, timezone.utc)
        started = self._episode_start if self._episode_start is not None else now_ts
        who = f" ({label})" if label else ""
        return (f"⏳ Request Queue{who}: waited {format_wait(now_ts - started)}, "
                f"{format_wait(target_ts - now_ts, round_up=True)} left "
                f"(starts {format_clock(target, 'local', now)} local)")

    def wait_until(self, target, *, stop_check: Optional[Callable[[], bool]] = None,
                   log: Optional[Callable[[str], None]] = print, label: str = "") -> bool:
        """Block the calling thread until *target* (aware datetime or epoch seconds). True when it
        was reached, False as soon as *stop_check()* is true (graceful or forced stop)."""
        target_ts = target.timestamp() if isinstance(target, datetime) else float(target)
        ident = threading.get_ident()
        announce = None
        with self._lock:
            now_ts = self.clock()
            if target_ts <= now_ts:
                return True
            if not self._waiting:
                self._episode_start = now_ts
                self._announced = set()
                self._last_log = now_ts
            self._waiting[ident] = target_ts
            self._current_start[ident] = now_ts
            key = round(target_ts)
            if key not in self._announced:
                self._announced.add(key)
                announce = self._start_line(target_ts, label, now_ts)
        if announce and log:
            log(announce)
        try:
            while True:
                if stop_check is not None:
                    try:
                        if stop_check():
                            return False
                    except Exception:
                        pass
                now_ts = self.clock()
                if now_ts >= target_ts:
                    return True
                line = None
                with self._lock:
                    if self._last_log is None or now_ts - self._last_log >= self.log_interval:
                        self._last_log = now_ts
                        line = self._progress_line(target_ts, label, now_ts)
                if line and log:
                    log(line)
                self.sleep(max(0.0, min(self.poll, target_ts - now_ts)))
        finally:
            with self._lock:
                started = self._current_start.pop(ident, None)
                if started is not None:
                    self._totals[ident] = self._totals.get(ident, 0.0) + max(0.0, self.clock() - started)
                self._waiting.pop(ident, None)
                if not self._waiting:
                    self._episode_start = None
                    self._announced = set()
                    self._last_log = None


WAITER = QueueWaiter()


def _ident(thread_or_ident):
    if thread_or_ident is None:
        return threading.get_ident()
    if isinstance(thread_or_ident, int):
        return thread_or_ident
    return getattr(thread_or_ident, "ident", None)


def is_thread_waiting(thread_or_ident=None) -> bool:
    """Whether that thread (a ``Thread`` or ident; default the caller) is in a Request Queue wait.
    Timeout loops watching an API thread skip counting while this is true."""
    ident = _ident(thread_or_ident)
    return ident is not None and WAITER.is_waiting(ident)


def thread_wait_seconds(thread_or_ident=None) -> float:
    """Cumulative Request Queue wait of that thread (default the caller), for subtracting from a
    measured API call time."""
    ident = _ident(thread_or_ident)
    return 0.0 if ident is None else WAITER.waited_seconds(ident)
