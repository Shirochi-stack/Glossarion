"""Editor tool sheets (UI_SPEC §4.1 "⋯ › Advanced"; desktop editor "Advanced editing" grid).

* ``ColumnFilterSheet`` - the column header filters (one column at a time: values with
  counts, a search box, Select all / Clear; type, gender, description and custom
  columns alike).
* ``TrimSheet`` - "Smart Trim Glossary": keep the first N entries + the desktop preview.
* ``FilterEntriesSheet`` - "Filter Glossary Entries": keep types (with optional "First N"
  per type), a text filter and the gender filter for gender-enabled types; Preview Filter
  / Apply Filter.
* ``BackupSettingsSheet`` - "Automatic Backup Settings": enable, maximum backups (0 =
  unlimited), the naming pattern and location, Backup Now.
* ``ConvertSheet`` - "Convert Format": token-efficient or legacy CSV (the
  ``glossary_use_legacy_csv`` setting) to the default path or a copy.
* ``NameSheet`` - Save As / Export Selection: a file name + JSON / CSV.
* ``about_format`` - the desktop "About Format" box (``glossary_document``'s duplicate
  detection text).

The sheets only collect input; the editor runs the shared operation (``GlossaryService``).
"""

from __future__ import annotations

import os
from typing import Any, Callable, Mapping, Optional, Sequence

import flet as ft

from glossarion_mobile.ui.components._handlers import call_handler
from glossarion_mobile.ui.glossary.common import SheetHost, sheet
from glossarion_mobile.ui.glossary.entry_sheet import field_label

__all__ = [
    "ABOUT_FORMAT",
    "ABOUT_FORMAT_TITLE",
    "BackupSettingsSheet",
    "ColumnFilterSheet",
    "ConvertSheet",
    "FilterEntriesSheet",
    "NameSheet",
    "TextSizeSheet",
    "TrimSheet",
    "about_format",
]

#: Fallback when glossary_document is missing (the desktop "About Format" box text).
ABOUT_FORMAT_TITLE = "Duplicate Detection"
ABOUT_FORMAT = (
    "Duplicate detection is based on the raw_name field.\n\n"
    "• Entries with identical raw_name values are considered duplicates\n"
    "• The first occurrence is kept, later ones are removed\n"
    "• Honorifics filtering can be toggled in the Manual Glossary tab\n\n"
    "When honorifics filtering is enabled, names are compared after removing honorifics."
)
MAX_FILTER_VALUES = 300


def about_format(service: Any) -> tuple:
    """(title, text) of the desktop "About Format" box (``DUPLICATE_DETECTION_INFO_*``)."""
    core = getattr(service, "core", None)
    title = core.value("glossary_document", "DUPLICATE_DETECTION_INFO_TITLE", default=None) if core else None
    text = core.value("glossary_document", "DUPLICATE_DETECTION_INFO_TEXT", default=None) if core else None
    return str(title or ABOUT_FORMAT_TITLE), str(text or ABOUT_FORMAT)


class ColumnFilterSheet:
    """One column's value filter (desktop header filter popup)."""

    def __init__(self, ctx: Any, *, fields: Sequence[str], values_for: Callable[[str], list],
                 active: Mapping[str, Any], on_apply: Callable[[str, Optional[frozenset]], Any],
                 initial_field: Optional[str] = None) -> None:
        self.ctx = ctx
        self.host = SheetHost(ctx)
        self.fields = [f for f in fields if f != "_section"]
        self.values_for = values_for
        self.active = {k: frozenset(v) for k, v in active.items()}
        self.on_apply = on_apply
        self.field = initial_field if initial_field in self.fields else (self.fields[0] if self.fields else "")
        self.checks: dict = {}
        self.field_picker = ft.Dropdown(label="Column", value=self.field, options=[
            ft.DropdownOption(key=f, text=field_label(f) + (" 🔽" if f in self.active else "")) for f in self.fields],
            on_select=self._on_field, expand=True, key="cf-field")
        self.search = ft.TextField(label="Search values", dense=True, on_change=lambda e: self._render(),
                                   key="cf-search")
        self.list = ft.Column(spacing=0, tight=True, key="cf-values")
        self.note = ft.Text("", theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT)
        self.dialog = sheet("Filter", [self.field_picker, self.search, self.note, self.list], actions=[
            ft.TextButton(content="Select all", on_click=lambda e: self._set_all(True), key="cf-all"),
            ft.TextButton(content="Clear", on_click=lambda e: self.clear(), key="cf-clear"),
            ft.FilledButton(content="Apply", on_click=lambda e: self.apply(), key="cf-apply"),
        ], key="cf-sheet")
        self._render()

    def _on_field(self, e: Any = None) -> None:
        self.field = self.field_picker.value or self.field
        self.search.value = ""
        self._render()
        self.ctx.push(self.dialog)

    def _render(self) -> None:
        values = self.values_for(self.field) if self.field else []
        query = (self.search.value or "").casefold()
        allowed = self.active.get(self.field)
        shown = [(v, n) for v, n in values if not query or query in v.casefold()]
        self.checks = {}
        controls = []
        for value, count in shown[:MAX_FILTER_VALUES]:
            check = ft.Checkbox(label=f"{value or '(empty)'}  ({count:,})", value=allowed is None or value in allowed,
                                data=value)
            self.checks[value] = check
            controls.append(check)
        self.list.controls = controls
        hidden = len(shown) - len(controls)
        self.note.value = (f"{len(values):,} distinct values" + (f" · {hidden:,} more not listed (search to narrow)"
                                                                  if hidden > 0 else ""))
        try:
            self.list.update()
            self.note.update()
        except Exception:
            pass

    def _set_all(self, on: bool) -> None:
        for check in self.checks.values():
            check.value = on
        self.ctx.push(self.list)

    def selected(self) -> Optional[frozenset]:
        """The allowed values (None = no filter on this column). Values hidden by the search keep their
        previous state (desktop ``_collect_glossary_filter_values`` without restrict_to_visible)."""
        allowed = set(self.active.get(self.field, frozenset(v for v, _n in self.values_for(self.field))))
        for value, check in self.checks.items():
            if check.value:
                allowed.add(value)
            else:
                allowed.discard(value)
        everything = {v for v, _n in self.values_for(self.field)}
        if allowed >= everything:
            return None
        return frozenset(allowed)

    def apply(self) -> Any:
        selected = self.selected()
        self.host.close()
        return call_handler(self.on_apply, self.field, selected)

    def clear(self) -> Any:
        self.host.close()
        return call_handler(self.on_apply, self.field, None)

    def show(self, page: Any = None) -> "ColumnFilterSheet":
        self.host.open(self.dialog)
        return self


class TrimSheet:
    """"Smart Trim Glossary" (``smart_trim_dialog``)."""

    def __init__(self, ctx: Any, *, total: int, type_summary: str, on_apply: Callable[[int], Any],
                 preview: Optional[Callable[[int], str]] = None) -> None:
        self.ctx = ctx
        self.host = SheetHost(ctx)
        self.total = int(total)
        self.on_apply = on_apply
        self.preview_text = preview
        self.field = ft.TextField(label="Keep first", value=str(min(100, self.total)),
                                  suffix=f"entries (out of {self.total})", keyboard_type=ft.KeyboardType.NUMBER,
                                  dense=True, key="trim-n")
        self.preview = ft.Text("Click 'Preview Changes' to see the effect", theme_style=ft.TextThemeStyle.BODY_SMALL,
                               color=ft.Colors.ON_SURFACE_VARIANT, key="trim-preview")
        stats = [ft.Text(f"Total entries: {self.total}", theme_style=ft.TextThemeStyle.BODY_SMALL)]
        if type_summary:
            stats.append(ft.Text(type_summary, theme_style=ft.TextThemeStyle.BODY_SMALL))
        self.dialog = sheet("Smart Trim Glossary", [
            ft.Text("Limit the number of entries in your glossary", color=ft.Colors.ON_SURFACE_VARIANT),
            ft.Text("Current Glossary Statistics", theme_style=ft.TextThemeStyle.LABEL_LARGE, color=ft.Colors.PRIMARY),
            *stats,
            ft.Text("Entry Limit", theme_style=ft.TextThemeStyle.LABEL_LARGE, color=ft.Colors.PRIMARY),
            ft.Text("Keep only the first N entries to reduce glossary size", theme_style=ft.TextThemeStyle.BODY_SMALL),
            self.field,
            self.preview,
            ft.Text("💡 Tip: Entries are kept in their original order", italic=True,
                    theme_style=ft.TextThemeStyle.BODY_SMALL),
        ], actions=[
            ft.TextButton(content="Preview Changes", on_click=lambda e: self.preview_changes(), key="trim-preview-btn"),
            ft.TextButton(content="Cancel", on_click=lambda e: self.host.close()),
            ft.FilledButton(content="Apply Trim", on_click=lambda e: self.apply(), key="trim-apply"),
        ], key="trim-sheet")

    def preview_changes(self) -> str:
        try:
            top_n = int(self.field.value)
            if self.preview_text is not None:
                self.preview.value = str(self.preview_text(top_n))
            else:
                removed = max(0, self.total - top_n)
                self.preview.value = f"Preview of changes:\n• Entries: {self.total} → {top_n} ({removed} removed)\n"
            self.preview.color = ft.Colors.PRIMARY
        except (TypeError, ValueError):
            self.preview.value = "Please enter a valid number"
            self.preview.color = ft.Colors.ERROR
        self.ctx.push(self.preview)
        return self.preview.value

    def apply(self) -> Any:
        try:
            top_n = int(self.field.value)
        except (TypeError, ValueError):
            self.preview.value = "Please enter valid numbers"
            self.preview.color = ft.Colors.ERROR
            self.ctx.push(self.preview)
            return None
        self.host.close()
        return call_handler(self.on_apply, top_n)

    def show(self, page: Any = None) -> "TrimSheet":
        self.host.open(self.dialog)
        return self


class FilterEntriesSheet:
    """"Filter Glossary Entries" (``filter_entries_dialog``)."""

    GENDERS = (("all", "All genders"), ("Male", "Male only"), ("Female", "Female only"), ("Unknown", "Unknown only"))

    def __init__(self, ctx: Any, *, total: int, types: Sequence[str], typed: bool,
                 on_preview: Callable[[dict], Any], on_apply: Callable[[dict], Any]) -> None:
        self.ctx = ctx
        self.host = SheetHost(ctx)
        self.total = int(total)
        self.on_preview = on_preview
        self.on_apply = on_apply
        self.type_checks: dict = {}
        self.type_limits: dict = {}
        controls: list = [
            ft.Text("Filter entries by type or content", color=ft.Colors.ON_SURFACE_VARIANT),
            ft.Text(f"Total entries: {self.total}", theme_style=ft.TextThemeStyle.BODY_SMALL),
        ]
        if typed:
            controls.append(ft.Text("Entry Type", theme_style=ft.TextThemeStyle.LABEL_LARGE, color=ft.Colors.PRIMARY))
            controls.append(ft.Text("Optional: keep only the first N entries of each type (blank = keep all)",
                                    theme_style=ft.TextThemeStyle.BODY_SMALL))
            for name in types:
                check = ft.Checkbox(label=f"Keep {name}", value=True, expand=True, key=f"fe-type-{name}")
                limit = ft.TextField(hint_text="All", width=90, dense=True, keyboard_type=ft.KeyboardType.NUMBER,
                                     key=f"fe-limit-{name}")
                self.type_checks[name] = check
                self.type_limits[name] = limit
                controls.append(ft.Row([check, ft.Text("First N:"), limit], spacing=6))
        controls.append(ft.Text("Text Content Filter", theme_style=ft.TextThemeStyle.LABEL_LARGE,
                                color=ft.Colors.PRIMARY))
        self.text = ft.TextField(label="Keep entries containing text (case-insensitive):", dense=True, key="fe-text")
        controls.append(self.text)
        self.gender = ft.RadioGroup(value="all", content=ft.Column(
            [ft.Radio(value=v, label=label) for v, label in self.GENDERS], spacing=0, tight=True), key="fe-gender")
        if typed:
            controls.append(ft.Text("Gender Filter (Gender-Enabled Types)", theme_style=ft.TextThemeStyle.LABEL_LARGE,
                                    color=ft.Colors.PRIMARY))
            controls.append(self.gender)
        self.preview = ft.Text("Click 'Preview Filter' to see how many entries match",
                               theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT,
                               key="fe-preview")
        controls.append(self.preview)
        self.dialog = sheet("Filter Glossary Entries", controls, actions=[
            ft.TextButton(content="Preview Filter", on_click=lambda e: self.ctx.spawn(self.preview_filter()),
                          key="fe-preview-btn"),
            ft.TextButton(content="Cancel", on_click=lambda e: self.host.close()),
            ft.FilledButton(content="Apply Filter", on_click=lambda e: self.apply(), key="fe-apply"),
        ], key="fe-sheet")

    def conditions(self) -> dict:
        """The ``GlossaryDocument.filter_matcher`` choices: kept_types {type: keep}, type_limits
        {type: "First N" text}, search_text, gender_value ('all' / 'Male' / 'Female' / 'Unknown')."""
        return {
            "kept_types": {name: bool(check.value) for name, check in self.type_checks.items()},
            "type_limits": {name: str(field.value or "") for name, field in self.type_limits.items()},
            "search_text": str(self.text.value or ""),
            "gender_value": self.gender.value or "all",
        }

    async def preview_filter(self) -> Any:
        result = call_handler(self.on_preview, self.conditions())
        if result is not None:
            result = await result
        message = getattr(result, "message", None) if result is not None else None
        if message:
            self.preview.value = message
            matching = getattr(result, "count", 0)
            self.preview.color = ft.Colors.PRIMARY if matching else ft.Colors.ERROR
            self.ctx.push(self.preview)
        return result

    def apply(self) -> Any:
        self.host.close()
        return call_handler(self.on_apply, self.conditions())

    def show(self, page: Any = None) -> "FilterEntriesSheet":
        self.host.open(self.dialog)
        return self


class BackupSettingsSheet:
    """"Automatic Backup Settings" (``backup_settings_dialog``)."""

    def __init__(self, ctx: Any, *, enabled: bool, max_backups: int, glossary_path: str,
                 on_save: Callable[[bool, int], Any], on_backup_now: Callable[[], Any]) -> None:
        self.ctx = ctx
        self.host = SheetHost(ctx)
        self.on_save = on_save
        self.on_backup_now = on_backup_now
        self.enabled = ft.Switch(label="Enable automatic backups before modifications", value=bool(enabled),
                                 on_change=lambda e: self._sync(), key="bk-enabled")
        self.max_field = ft.TextField(label="Maximum backups to keep", value=str(int(max_backups)), suffix="(0 = unlimited)",
                                      keyboard_type=ft.KeyboardType.NUMBER, dense=True, key="bk-max")
        location = [ft.Text("📁 Backup Location:", weight=ft.FontWeight.W_600)]
        if glossary_path:
            backups = os.path.join(os.path.dirname(glossary_path), "Backups")
            location.append(ft.Text("Backups/", color=ft.Colors.PRIMARY))
            if os.path.isdir(backups):
                try:
                    count = len([f for f in os.listdir(backups) if f.endswith(".json")])
                except OSError:
                    count = 0
                location.append(ft.Text(f"Currently contains {count} backup(s)",
                                        theme_style=ft.TextThemeStyle.BODY_SMALL, key="bk-count"))
        else:
            location.append(ft.Text("Backups", color=ft.Colors.ON_SURFACE_VARIANT))
        self.dialog = sheet("Automatic Backup Settings", [
            self.enabled, self.max_field,
            ft.Text("Backup naming pattern:", weight=ft.FontWeight.W_600),
            ft.Text("[original_name]_[operation]_[YYYYMMDD_HHMMSS].json", italic=True,
                    theme_style=ft.TextThemeStyle.BODY_SMALL),
            ft.Text("Example: my_glossary_before_delete_5_20240115_143052.json",
                    theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT),
            *location,
        ], actions=[
            ft.TextButton(content="Backup Now", on_click=lambda e: self._backup_now(), key="bk-now"),
            ft.TextButton(content="Cancel", on_click=lambda e: self.host.close()),
            ft.FilledButton(content="Save Settings", on_click=lambda e: self.save(), key="bk-save"),
        ], key="bk-sheet")
        self._sync()

    def _sync(self) -> None:
        self.max_field.disabled = not bool(self.enabled.value)
        self.ctx.push(self.max_field)

    def save(self) -> Any:
        try:
            limit = max(0, min(999, int(self.max_field.value)))
        except (TypeError, ValueError):
            limit = 50
        self.host.close()
        return call_handler(self.on_save, bool(self.enabled.value), limit)

    def _backup_now(self) -> Any:
        return call_handler(self.on_backup_now)

    def show(self, page: Any = None) -> "BackupSettingsSheet":
        self.host.open(self.dialog)
        return self


class ConvertSheet:
    """"Convert Format" (``convert_glossary_format``): where the CSV goes."""

    def __init__(self, ctx: Any, *, default_path: str, legacy: bool, on_convert: Callable[[str], Any]) -> None:
        self.ctx = ctx
        self.host = SheetHost(ctx)
        self.default_path = default_path
        self.on_convert = on_convert
        label = "legacy CSV" if legacy else "token-efficient"
        self.name = ft.TextField(label="File name", value=os.path.basename(default_path), dense=True, key="cv-name")
        self.dialog = sheet("Export Glossary to CSV", [
            ft.Text(f"Format: {label} (Settings › Glossary › Balanced/Full › Use legacy CSV format)",
                    theme_style=ft.TextThemeStyle.BODY_SMALL),
            ft.Text(f"Folder: {os.path.dirname(default_path)}", theme_style=ft.TextThemeStyle.BODY_SMALL,
                    color=ft.Colors.ON_SURFACE_VARIANT, selectable=True),
            self.name,
            ft.Text("A backup is created first (before_export).", theme_style=ft.TextThemeStyle.BODY_SMALL),
        ], actions=[
            ft.TextButton(content="Cancel", on_click=lambda e: self.host.close()),
            ft.FilledButton(content="Convert", on_click=lambda e: self.convert(), key="cv-convert"),
        ], key="cv-sheet")

    def dest(self) -> str:
        name = os.path.basename(str(self.name.value or "").strip()) or os.path.basename(self.default_path)
        if not name.lower().endswith(".csv"):
            name += ".csv"
        return os.path.join(os.path.dirname(self.default_path), name)

    def convert(self) -> Any:
        self.host.close()
        return call_handler(self.on_convert, self.dest())

    def show(self, page: Any = None) -> "ConvertSheet":
        self.host.open(self.dialog)
        return self


class NameSheet:
    """Save As / Export Selection: a file name and JSON / CSV."""

    def __init__(self, ctx: Any, *, title: str, folder: str, default_name: str, confirm: str,
                 on_confirm: Callable[[str], Any], note: str = "") -> None:
        self.ctx = ctx
        self.host = SheetHost(ctx)
        self.folder = folder
        self.on_confirm = on_confirm
        stem, ext = os.path.splitext(default_name)
        self.name = ft.TextField(label="File name", value=stem, dense=True, key="name-field")
        self.format = ft.SegmentedButton(selected=[(ext.lstrip(".") or "json").lower()], segments=[
            ft.Segment(value="json", label=ft.Text("JSON")), ft.Segment(value="csv", label=ft.Text("CSV"))],
            show_selected_icon=False, key="name-format")
        controls: list = [self.name, self.format]
        if note:
            controls.append(ft.Text(note, theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT))
        self.dialog = sheet(title, controls, actions=[
            ft.TextButton(content="Cancel", on_click=lambda e: self.host.close()),
            ft.FilledButton(content=confirm, on_click=lambda e: self.confirm(), key="name-confirm"),
        ], key="name-sheet")

    def path(self) -> str:
        stem = os.path.basename(str(self.name.value or "").strip()) or "glossary"
        stem = os.path.splitext(stem)[0]
        selected = list(self.format.selected or ["json"])
        return os.path.join(self.folder, f"{stem}.{selected[0] if selected else 'json'}")

    def confirm(self) -> Any:
        self.host.close()
        return call_handler(self.on_confirm, self.path())

    def show(self, page: Any = None) -> "NameSheet":
        self.host.open(self.dialog)
        return self


class TextSizeSheet:
    """Editor ⋯ › Text size (``glossary_editor_tree_font_size``, 8–32 like the desktop zoom)."""

    def __init__(self, ctx: Any, *, size: float, on_change: Callable[[int], Any]) -> None:
        self.ctx = ctx
        self.host = SheetHost(ctx)
        self.on_change = on_change
        self.label = ft.Text(f"{int(size)} pt", key="ts-label")
        self.slider = ft.Slider(min=8, max=32, divisions=24, value=float(size), on_change=lambda e: self._changed(),
                                on_change_end=lambda e: self._commit(), key="ts-slider")
        self.dialog = sheet("Text size", [ft.Row([self.slider, self.label])], actions=[
            ft.TextButton(content="Done", on_click=lambda e: self.host.close())], key="ts-sheet")

    def _changed(self) -> None:
        self.label.value = f"{int(self.slider.value)} pt"
        self.ctx.push(self.label)

    def _commit(self) -> Any:
        return call_handler(self.on_change, int(self.slider.value))

    def show(self, page: Any = None) -> "TextSizeSheet":
        self.host.open(self.dialog)
        return self
