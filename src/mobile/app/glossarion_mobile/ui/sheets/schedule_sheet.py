"""ScheduleSheet: the "🗓️ Disable schedule" / "⏳ Request Queue" editor (desktop ``RequestScheduleDialog``).

One sheet for every time-based key setting of the shared ``request_schedule`` core:

* API keys › selection ⋯ "🗓️ Disable schedule…" (every pool, key field ``disable_schedule``) and
  "⏳ Request Queue…" (pools with rotation, key field ``queue_schedule``);
* the ModelSheet ⋯ "⏳ Request Queue…" and Settings › Models & keys (the main key, config
  ``request_queue_schedule``; run_env exports it as ``REQUEST_QUEUE_SCHEDULE``).

Field for field the desktop dialog: "Times are in" zone (Local by default; switching converts the
windows, ``request_schedule.convert``), the windows (HH:MM start → end, an end before the start runs
past midnight; ➕ Add window, 🗑️ per row), the weekday chips (the day a window starts), "🐋 Use
DeepSeek peak hours" (prefilled when nothing is saved, kept in UTC), "🧹 Clear schedule" (removes it
and closes) and the preview in local time, UTC and one more zone ("Also show in"). Titles, texts,
zone choices and preview lines come from the core (``EDITOR_MODES``, ``zone_choices``,
``preview_lines``), so both editors say the same. ``on_save(config)`` gets the canonical config
dict (``{}`` = removed) and may return an error to keep the sheet open.

The body scrolls with Cancel / Save pinned below it (``components.sheet``), so it fits a 412 × 915
phone at any text scale.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import flet as ft

from glossarion_mobile.ui.components.dialogs import close_dialog
from glossarion_mobile.ui.components.sheet import bottom_sheet, scroll_column, sheet_frame
from glossarion_mobile.ui.theme import HIT_TARGET

__all__ = ["ScheduleSheet", "parse_time_text", "queue_summary"]

log = logging.getLogger("glossarion.schedule")

#: A new window's default (desktop "➕ Add window": 09:00 → 17:00).
NEW_WINDOW = (9 * 60, 17 * 60)


def _rs() -> Any:
    import request_schedule  # shared GUI-free core

    return request_schedule


def _push(*controls: Any) -> None:
    for control in controls:
        if control is None:
            continue
        try:
            control.update()
        except Exception:
            pass


def queue_summary(config: Any, now: Any = None) -> str:
    """The main key's Request Queue on the model sheet / Settings tile: ``""`` when off, else
    "⏳ Request Queue · 10:00–13:00 (local), Mon–Fri" plus " · waiting until 13:00" inside a window."""
    if not config:
        return ""
    try:
        rs = _rs()
        schedule = rs.from_config(config)
        if schedule is None:
            return ""
        text = f"⏳ Request Queue · {rs.describe(schedule, 'local', now)}"
        end = rs.next_window_end(schedule, now)
        if end is not None:
            text += f" · waiting until {rs.format_clock(end, 'local', now)}"
        return text
    except Exception:
        return ""


def parse_time_text(text: Any) -> int:
    """``"7:05"`` / ``"07:05"`` / ``"0705"`` / ``"705"`` -> minutes (``request_schedule.parse_hhmm``;
    a phone keyboard may leave the colon out). Raises ``request_schedule.ScheduleError``."""
    raw = str(text or "").strip().replace(".", ":").replace("h", ":")
    if raw.isdigit() and len(raw) in (3, 4):
        raw = f"{raw[:-2]}:{raw[-2:]}"
    return _rs().parse_hhmm(raw)


class ScheduleSheet:
    """The schedule editor sheet; ``show(page)`` opens it, ``result`` is the saved config (``{}`` when
    cleared, None while nothing was saved)."""

    def __init__(
        self,
        schedule: Any = None,
        *,
        mode: str = "queue",
        subject: str = "",
        on_save: Optional[Callable[[dict], Optional[str]]] = None,
        page: Any = None,
        now: Optional[Callable[[], Any]] = None,
    ) -> None:
        rs = _rs()
        self.rs = rs
        self.mode = mode if mode in rs.EDITOR_MODES else "queue"
        self.subject = subject
        self.on_save = on_save
        self.page = page
        self.now = now  # tests: a fixed clock for the preview
        self.result: Optional[dict] = None
        self.rows: list = []  # [(row control, start field, end field)]
        self.zone = "local"
        current, enabled = rs.editable_config(schedule)
        self.suggested = current is None  # nothing saved: the DeepSeek preset is a suggestion
        # Off by default: everything below is disabled (greyed out) until the schedule is enabled
        self.enable_switch = ft.Switch(value=bool(enabled), key="schedule-enabled",
                                       on_change=lambda e: self.set_enabled(bool(e.control.value)))
        self.enable_row = ft.Row([self.enable_switch, ft.Container(
            content=ft.Text(rs.ENABLE_LABELS.get(self.mode, rs.ENABLE_LABELS["queue"]),
                            theme_style=ft.TextThemeStyle.BODY_LARGE),
            expand=True, on_click=lambda e: self.set_enabled(not self.enable_switch.value))],
            vertical_alignment=ft.CrossAxisAlignment.CENTER)
        title, explanation = rs.EDITOR_MODES[self.mode]
        self.title = f"{title} · {subject}" if subject else title
        self.zone_dropdown = ft.Dropdown(label="Times are in", dense=True, expand=True, key="schedule-zone",
                                         options=self._zone_options(True), on_select=self._on_zone_select)
        self.rows_column = ft.Column([], spacing=6, tight=True, key="schedule-windows")
        self.add_button = ft.TextButton(content="➕ Add window", on_click=lambda e: self.add_window(*NEW_WINDOW),
                                        key="schedule-add")
        self.day_chips: list = []
        for index, name in enumerate(rs.WEEKDAY_NAMES):
            self.day_chips.append(ft.Chip(label=ft.Text(name), selected=False, show_checkmark=False,
                                          on_click=lambda e, i=index: self.toggle_day(i), key=f"schedule-day-{index}"))
        self.deepseek_button = ft.OutlinedButton(content="🐋 Use DeepSeek peak hours", tooltip=rs.DEEPSEEK_PRESET_HELP,
                                                 on_click=lambda e: self.use_deepseek(), key="schedule-deepseek")
        self.clear_button = ft.TextButton(content="🧹 Clear schedule", tooltip="Remove this schedule and close",
                                          on_click=lambda e: self.clear(), key="schedule-clear")
        self.preview_zone_dropdown = ft.Dropdown(label="Also show in", dense=True, expand=True, value="UTC",
                                                 options=self._zone_options(False), key="schedule-preview-zone",
                                                 on_select=lambda e: self.update_preview())
        self.preview_text = ft.Text("", theme_style=ft.TextThemeStyle.BODY_SMALL, selectable=True,
                                    key="schedule-preview")
        self.error = ft.Text("", color=ft.Colors.ERROR, visible=False, theme_style=ft.TextThemeStyle.BODY_SMALL,
                             key="schedule-error")
        self.save_button = ft.FilledButton(content="Save", on_click=lambda e: self.save(), key="schedule-save")

        def heading(text: str) -> ft.Text:
            return ft.Text(text, theme_style=ft.TextThemeStyle.LABEL_LARGE, weight=ft.FontWeight.W_600)

        self.body = ft.Column([
            ft.Row([self.zone_dropdown]),
            heading("Windows (start → end; an end before the start runs past midnight)"),
            self.rows_column,
            self.add_button,
            heading("Days (the day a window starts)"),
            ft.Row(self.day_chips, wrap=True, spacing=6, run_spacing=6),
            ft.Row([self.deepseek_button, self.clear_button], wrap=True, spacing=8, run_spacing=6),
            heading("Preview"),
            ft.Row([self.preview_zone_dropdown]),
            self.preview_text,
            self.error,
        ], tight=True, spacing=10, disabled=not enabled, opacity=1.0 if enabled else 0.5, key="schedule-body")

        self.sheet = bottom_sheet(sheet_frame(scroll_column([
            ft.Text(self.title, theme_style=ft.TextThemeStyle.TITLE_MEDIUM, weight=ft.FontWeight.W_600),
            ft.Text(explanation, theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT),
            self.enable_row,
            self.body,
        ], footer=[ft.Row([ft.TextButton(content="Cancel", on_click=lambda e: self.close(), key="schedule-cancel"),
                           self.save_button], alignment=ft.MainAxisAlignment.END)])))
        self.load(current or rs.DEEPSEEK_PEAK_SCHEDULE, push=False)

    # ---- widgets ------------------------------------------------------------------------------------

    def _zone_options(self, include_local: bool) -> list:
        return [ft.DropdownOption(key=zone, text=label) for label, zone in self.rs.zone_choices(include_local)]

    def _set_zone_value(self, zone: str) -> None:
        """Select ``zone`` (a stored zone the picker does not list is added, like the desktop combo)."""
        if not any(option.key == zone for option in self.zone_dropdown.options):
            self.zone_dropdown.options.append(ft.DropdownOption(key=zone, text=zone))
        self.zone_dropdown.value = zone

    def _time_field(self, minutes: int, label: str) -> ft.TextField:
        return ft.TextField(value=self.rs.format_hhmm(minutes), label=label, width=96, dense=True,
                            keyboard_type=ft.KeyboardType.DATETIME, hint_text="HH:MM",
                            on_change=lambda e: self.update_preview(), on_blur=lambda e: self.update_preview())

    def add_window(self, start: int, end: int, *, push: bool = True) -> tuple:
        start_field, end_field = self._time_field(start, "Start"), self._time_field(end, "End")
        row = ft.Row(vertical_alignment=ft.CrossAxisAlignment.CENTER, spacing=6)
        entry = (row, start_field, end_field)
        row.controls = [start_field, ft.Text("→"), end_field,
                        ft.IconButton(icon=ft.Icons.DELETE_OUTLINE, tooltip="Remove this window",
                                      size_constraints=HIT_TARGET, on_click=lambda e: self.remove_window(entry))]
        self.rows.append(entry)
        self.rows_column.controls.append(row)
        if push:
            _push(self.rows_column)
            self.update_preview()
        return entry

    def remove_window(self, entry: Any) -> None:
        if isinstance(entry, int):
            entry = self.rows[entry] if 0 <= entry < len(self.rows) else None
        if entry in self.rows:
            self.rows.remove(entry)
            self.rows_column.controls.remove(entry[0])
            _push(self.rows_column)
        self.update_preview()

    # ---- state --------------------------------------------------------------------------------------

    def load(self, schedule: Any, *, push: bool = True) -> None:
        """Show ``schedule`` (a Schedule): its zone, windows and days."""
        self.rows.clear()
        self.rows_column.controls.clear()
        self.zone = schedule.timezone or "local"
        self._set_zone_value(self.zone)
        for window in schedule.windows:
            self.add_window(window.start, window.end, push=False)
        for day, chip in enumerate(self.day_chips):
            chip.selected = day in schedule.weekdays
        if push:
            _push(self.rows_column, self.zone_dropdown, *self.day_chips)
        self.update_preview(push=push)

    @property
    def enabled(self) -> bool:
        return bool(self.enable_switch.value)

    def set_enabled(self, on: bool) -> bool:
        """The off-by-default switch: the rest of the sheet is disabled and dimmed while off."""
        self.enable_switch.value = bool(on)
        self.body.disabled = not on
        self.body.opacity = 1.0 if on else 0.5
        _push(self.enable_switch, self.body)
        self.update_preview()
        return bool(on)

    def use_deepseek(self) -> None:
        """🐋 Use DeepSeek peak hours (kept in UTC)."""
        self.load(self.rs.DEEPSEEK_PEAK_SCHEDULE)

    def toggle_day(self, day: int) -> bool:
        chip = self.day_chips[day]
        chip.selected = not chip.selected
        _push(chip)
        self.update_preview()
        return bool(chip.selected)

    def days(self) -> frozenset:
        return frozenset(i for i, chip in enumerate(self.day_chips) if chip.selected)

    def schedule(self) -> tuple:
        """``(Schedule, error)``: the windows as typed (duplicates dropped), the days and the zone;
        ``error`` names the first time that is not HH:MM (its field shows it too)."""
        rs = self.rs
        windows: list = []
        error: Optional[str] = None
        for index, (_row, start_field, end_field) in enumerate(self.rows):
            values = []
            for field in (start_field, end_field):
                try:
                    values.append(parse_time_text(field.value))
                    field.error = None
                except rs.ScheduleError:
                    field.error = "HH:MM"
                    error = error or f"Window {index + 1}: enter the times as HH:MM (00:00–23:59)"
            if len(values) == 2:
                window = rs.Window(values[0], values[1])
                if window not in windows:
                    windows.append(window)
        return rs.Schedule(tuple(windows), self.days(), self.zone), error

    def _clock(self) -> Any:
        if self.now is None:
            return None
        return self.now() if callable(self.now) else self.now

    def preview_lines(self) -> list:
        schedule, error = self.schedule()
        if error:
            return [error]
        lines = self.rs.preview_lines(schedule, self.preview_zone_dropdown.value, suggested=self.suggested,
                                      now=self._clock())
        return lines if self.enabled else [self.rs.OFF_PREVIEW_LINE] + lines

    def update_preview(self, *, push: bool = True) -> str:
        text = "\n".join(self.preview_lines())
        self.preview_text.value = text
        if push:
            _push(self.preview_text, *[f for _row, s, e in self.rows for f in (s, e)])
        return text

    def _on_zone_select(self, e: Any = None) -> None:
        self.set_zone(str(getattr(getattr(e, "control", None), "value", None) or self.zone_dropdown.value or "local"))

    def set_zone(self, zone: str) -> Optional[str]:
        """"Times are in": rewrite the windows in ``zone`` (the same instants; desktop
        ``_on_zone_changed``). Returns the error that kept the old zone, else None."""
        rs = self.rs
        if zone == self.zone:
            return None
        problem: Optional[str] = None
        schedule, error = self.schedule()
        if not rs.zone_is_known(zone):
            problem = f"'{zone}' is not a timezone this device knows."
        elif error:
            problem = error
        elif schedule.windows and schedule.weekdays:
            try:
                converted = rs.convert(schedule, zone, self._clock())
            except rs.ScheduleError as exc:
                problem = f"Cannot rewrite these windows in {zone}: {exc}."
            else:
                self.load(converted._replace(timezone=zone))
                self._show_error(None)
                return None
        if problem:
            self._set_zone_value(self.zone)
            _push(self.zone_dropdown)
            self._show_error(problem)
            return problem
        self.zone = zone
        self._set_zone_value(zone)
        self._show_error(None)
        self.update_preview()
        return None

    def _show_error(self, message: Optional[str]) -> None:
        self.error.value = message or ""
        self.error.visible = bool(message)
        _push(self.error)

    # ---- actions ------------------------------------------------------------------------------------

    def _finish(self, config: dict) -> Optional[str]:
        if self.result is not None:  # a second tap while the sheet closes
            return None
        problem = None
        if self.on_save is not None:
            try:
                problem = self.on_save(dict(config))
            except Exception as exc:  # pragma: no cover - a store failure is shown, not raised
                log.exception("saving the schedule failed")
                problem = str(exc) or type(exc).__name__
        if problem:
            self._show_error(str(problem))
            return str(problem)
        self.result = dict(config)
        self.close()
        return None

    def save(self) -> Optional[str]:
        schedule, error = self.schedule()
        if error:
            self._show_error(error)
            return error
        if self.enabled and schedule.windows and not schedule.weekdays:
            message = "Tick at least one day, or use 🧹 Clear schedule."
            self._show_error(message)
            return message
        return self._finish(self.rs.to_config(schedule, enabled=self.enabled))

    def clear(self) -> Optional[str]:
        """🧹 Clear schedule: removes it and closes."""
        return self._finish({})

    def show(self, page: Any = None) -> "ScheduleSheet":
        if page is not None:
            self.page = page
        if self.page is not None:
            self.page.show_dialog(self.sheet)
        return self

    def close(self) -> None:
        close_dialog(self.page, self.sheet)
