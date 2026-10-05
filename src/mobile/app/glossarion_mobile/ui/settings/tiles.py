"""Schema-bound setting tiles (UI_SPEC §5.7).

One tile per ``SettingSpec``; ``make_tile`` picks the class from
``model.tile_kind``: SwitchTile · NumberTile · SliderTile · SegmentedTile ·
DropdownTile · TextTile · PromptTile (opens ``PromptEditor``) · SecretTile ·
PathTile · ListSettingTile · JsonTile.

Common anatomy: label + "modified" dot, one help line, the effective value
(the stored value, else the schema default marked "· default"; defaults are
display-only and never written), badges (locked purple 🔒 reason, a
``ReasonChip`` when ``is_available`` says no, "not used" when a
``visible_if`` rule is off), ⓘ opening an ``InfoSheet`` with the full help,
key, env names, default and recorded desktop discrepancies; long-press resets
to default (removes the key, with Undo). The outer ``Container`` carries
``key=ft.ScrollKey(<config key>)`` so search can ``scroll_to`` it.

Edits go through ``SchemaAccess.coerce`` and ``MobileConfigStore.set``; a
validation error is shown under the tile and nothing is stored.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, Iterator, Optional

import flet as ft

from glossarion_mobile.state.config_store import MISSING, MobileConfigStore
from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.components.info_sheet import InfoSheet
from glossarion_mobile.ui.components.reason_chip import ReasonChip
from glossarion_mobile.ui.settings.editors import JsonEditor, ListEditor, PathEditor, PromptEditor, SecretEditor
from glossarion_mobile.ui.settings.model import (
    choice_options,
    config_path,
    env_names,
    help_line,
    label_for,
    plain_text,
    slider_params,
    spec_attr,
    spec_type,
    summarize,
    tile_kind,
)
from glossarion_mobile.ui.theme import HIT_TARGET, semantic

__all__ = [
    "DropdownTile",
    "EffectiveConfig",
    "JsonTile",
    "ListSettingTile",
    "NumberTile",
    "PathTile",
    "PromptTile",
    "SecretTile",
    "SegmentedTile",
    "SettingTile",
    "SliderTile",
    "SwitchTile",
    "TILE_CLASSES",
    "TextTile",
    "make_tile",
]

log = logging.getLogger("glossarion.settings")

HIGHLIGHT = ft.Colors.with_opacity(0.14, ft.Colors.PRIMARY)


class EffectiveConfig(Mapping):
    """Read-only mapping of effective values (stored, else display default) for rule evaluation."""

    def __init__(self, store: MobileConfigStore) -> None:
        self.store = store

    def __getitem__(self, key: str) -> Any:
        if self.store.has(key):
            return self.store.get(key)
        value = self.store.default_for(key)
        if value is None:
            raise KeyError(key)
        return value

    def __iter__(self) -> Iterator[str]:
        return iter(self.store.keys())

    def __len__(self) -> int:
        return len(self.store)


def _parse_number(raw: Any, kind: str) -> Any:
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return int(raw) if kind == "int" and float(raw).is_integer() else raw
    text = str(raw or "").strip().replace(",", "")
    if not text:
        raise ValueError("Enter a number")
    try:
        if kind == "int":
            number = float(text)
            if not number.is_integer():
                raise ValueError
            return int(number)
        return float(text)
    except ValueError:
        raise ValueError("Enter a whole number" if kind == "int" else "Enter a number") from None


class SettingTile:
    kind = "text"

    def __init__(self, spec: Any, ctx: Any, *, config: Optional[Mapping] = None) -> None:
        self.spec = spec
        self.ctx = ctx
        self.key = str(spec_attr(spec, "key", ""))
        self.path = config_path(spec)  # nested settings live inside their parent object
        self.label = label_for(spec)
        self.help = help_line(spec)
        self.config = config if config is not None else EffectiveConfig(ctx.store)
        self.available, self.unavailable_reason = ctx.schema.availability(self.key)
        self.readonly_reason: Optional[str] = self.readonly()
        self.lock_reason: Optional[str] = None
        self.hidden_reason: Optional[str] = None
        self.error: Optional[str] = None
        self.highlighted = False
        self.editor: Any = None
        self._build()
        self.refresh(push=False)

    def readonly(self) -> Optional[str]:
        """Reason this tile only displays its value (edited on a dedicated screen)."""
        return None

    # ---- value -------------------------------------------------------------------------

    @property
    def store(self) -> MobileConfigStore:
        return self.ctx.store

    def value(self) -> Any:
        return self.store.effective(self.path)

    def default(self) -> Any:
        return self.store.default_for(self.path)

    @property
    def stored(self) -> bool:
        return self.store.has(self.path)

    @property
    def editable(self) -> bool:
        return self.available and self.lock_reason is None and self.readonly_reason is None

    def default_note(self) -> Optional[str]:
        note = getattr(self.ctx.schema, "default_note", None)
        return note(self.key) if note is not None else None

    def summary(self) -> str:
        value = self.value()
        if value is None and not self.stored:
            note = self.default_note()
            if note:
                return note[:1].upper() + note[1:]
        return summarize(value, self.kind, self.spec)

    # ---- building ------------------------------------------------------------------------

    def build_control(self) -> Optional[ft.Control]:
        """Trailing inline control (e.g. a Switch)."""
        return None

    def build_editor_row(self) -> Optional[ft.Control]:
        """Inline editor below the title (numbers, sliders, choices, text)."""
        return None

    def _build(self) -> None:
        self.title_text = ft.Text(self.label, theme_style=ft.TextThemeStyle.BODY_MEDIUM, weight=ft.FontWeight.W_500)
        self.modified_dot = ft.Container(width=8, height=8, border_radius=4, bgcolor=ft.Colors.PRIMARY, visible=False,
                                         tooltip="Modified")
        self.help_text = ft.Text(self.help, theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT,
                                 max_lines=1, overflow=ft.TextOverflow.ELLIPSIS, visible=bool(self.help))
        self.value_text = ft.Text("", theme_style=ft.TextThemeStyle.BODY_SMALL, max_lines=2,
                                  overflow=ft.TextOverflow.ELLIPSIS)
        self.badges = ft.Row([], spacing=6, wrap=True, run_spacing=4)
        self.error_text = ft.Text("", color=ft.Colors.ERROR, theme_style=ft.TextThemeStyle.BODY_SMALL, visible=False)
        self.info_button = ft.IconButton(icon=ft.Icons.INFO_OUTLINE, tooltip="About this setting",
                                         on_click=self.open_help, size_constraints=HIT_TARGET, icon_size=18)
        self.inline = self.build_control()
        self.editor_row = self.build_editor_row()
        trailing = [c for c in (self.inline, self.info_button) if c is not None]
        self.list_tile = ft.ListTile(
            title=ft.Row([ft.Container(content=self.title_text, expand=True), self.modified_dot], spacing=6,
                         vertical_alignment=ft.CrossAxisAlignment.CENTER),
            subtitle=ft.Column([self.help_text, self.value_text, self.badges], spacing=2, tight=True),
            trailing=ft.Row(trailing, spacing=0, tight=True, vertical_alignment=ft.CrossAxisAlignment.CENTER),
            on_click=self._on_tap,
            on_long_press=self._on_long_press,
            min_height=tokens.SIZES["hit_target"],
            content_padding=ft.Padding.only(left=12, right=4),
        )
        body: list[ft.Control] = [self.list_tile]
        if self.editor_row is not None:
            body.append(ft.Container(content=self.editor_row, padding=ft.Padding.only(left=12, right=12, bottom=8)))
        body.append(ft.Container(content=self.error_text, padding=ft.Padding.only(left=12, right=12)))
        self.control = ft.Container(
            key=ft.ScrollKey(self.key),
            content=ft.Column(body, spacing=0, tight=True),
            border_radius=tokens.RADII["tile"],
            bgcolor=ft.Colors.SURFACE_CONTAINER_LOW,
        )

    # ---- state ---------------------------------------------------------------------------

    def _evaluate_rules(self) -> None:
        try:
            self.lock_reason = self.ctx.schema.lock_reason(self.spec, self.config)
            self.hidden_reason = self.ctx.schema.hidden_reason(self.spec, self.config)
        except Exception:
            log.debug("rule evaluation failed for %s", self.key, exc_info=True)
            self.lock_reason = self.hidden_reason = None

    def _badge_controls(self) -> list[ft.Control]:
        out: list[ft.Control] = []
        if not self.available:
            out.append(ReasonChip(reason=self.unavailable_reason or "Not available on mobile",
                                  detail=f"{self.label}: {self.unavailable_reason or 'not available on mobile'}. "
                                         "Its config.json value is kept untouched."))
        if self.lock_reason:
            color = semantic("locked", False)
            out.append(
                ft.Container(
                    content=ft.Row([ft.Icon(ft.Icons.LOCK, size=12, color=color),
                                    ft.Text(self.lock_reason, theme_style=ft.TextThemeStyle.LABEL_SMALL, color=color)],
                                   spacing=4, tight=True),
                    padding=ft.Padding.symmetric(horizontal=6, vertical=2),
                    border=ft.Border.all(1, color),
                    border_radius=tokens.RADII["chip"],
                    tooltip=self.lock_reason,
                )
            )
        if self.readonly_reason and self.available:
            out.append(ReasonChip(reason=self.readonly_reason))
        if self.hidden_reason and self.available:
            out.append(ReasonChip(reason=self.hidden_reason))
        return out

    def refresh_value(self) -> None:
        """Kind-specific: copy the effective value into the inline control."""

    def refresh(self, push: bool = True) -> None:
        self._evaluate_rules()
        editable = self.editable
        text = self.summary()
        stored = self.stored
        noted = not stored and self.value() is None and bool(self.default_note())
        self.value_text.value = text if (stored or noted) else f"{text} · default"
        self.value_text.color = ft.Colors.PRIMARY if stored else ft.Colors.ON_SURFACE_VARIANT
        self.modified_dot.visible = self.store.is_modified(self.path)
        self.title_text.color = ft.Colors.ON_SURFACE if editable else ft.Colors.ON_SURFACE_VARIANT
        self.badges.controls = self._badge_controls()
        self.badges.visible = bool(self.badges.controls)
        for control in (self.inline, self.editor_row):
            if control is not None:
                control.disabled = not editable
        self.refresh_value()
        if push:
            self.ctx.push(self.control)

    def set_error(self, message: Optional[str]) -> None:
        self.error = message or None
        self.error_text.value = message or ""
        self.error_text.visible = bool(message)
        self.ctx.push(self.error_text)

    def set_highlight(self, on: bool) -> None:
        self.highlighted = on
        self.control.bgcolor = HIGHLIGHT if on else ft.Colors.SURFACE_CONTAINER_LOW
        self.ctx.push(self.control)

    # ---- editing ---------------------------------------------------------------------------

    def parse(self, raw: Any) -> Any:
        return raw

    def apply(self, raw: Any) -> bool:
        """Coerce and store ``raw``; returns False (and shows the error) when invalid or not editable."""
        if not self.editable:
            return False
        try:
            value = self.ctx.schema.coerce(self.key, self.parse(raw))
        except ValueError as exc:
            self.set_error(str(exc) or "Invalid value")
            self.refresh_value()
            self.ctx.push(self.control)
            return False
        if self.error:
            self.set_error(None)
        try:
            self.store.set(self.path, value)
        except ValueError as exc:  # e.g. the parent of a nested setting is not an object
            self.set_error(str(exc))
            return False
        self.refresh()
        return True

    def save_from_editor(self, value: Any) -> Optional[str]:
        return None if self.apply(value) else (self.error or "Invalid value")

    def reset(self) -> bool:
        if not self.editable or not self.stored:
            return False
        old = self.store.get(self.path, MISSING)
        self.store.unset(self.path)
        self.refresh()
        if old is not MISSING:
            self.ctx.say(f"{self.label} reset to default", "Undo", lambda: (self.store.set(self.path, old), self.refresh()))
        return True

    def activate(self) -> Any:
        """Tap on an editable tile (kind-specific)."""
        return None

    def _on_tap(self, e: Any = None) -> Any:
        if not self.editable:
            return self.open_help()
        return self.activate()

    def _on_long_press(self, e: Any = None) -> None:
        if self.stored and self.editable:
            self.reset()
        else:
            self.open_help()

    def help_body(self) -> str:
        lines: list[str] = []
        tooltip = plain_text(spec_attr(self.spec, "tooltip", ""))
        if tooltip:
            lines.append(tooltip)
        lines.append("Config key: " + (" › ".join(self.path) if len(self.path) > 1 else self.key))
        names = env_names(self.spec)
        if names:
            lines.append("Environment: " + ", ".join(names))
        default = self.default()
        if self.kind != "secret" and default is not None:
            lines.append("Default: " + summarize(default, self.kind, self.spec))
        elif default is None and self.default_note():
            lines.append(f"Default: {self.default_note()}.")
        if not self.stored:
            lines.append("Not stored in config.json yet: the default applies, exactly like a fresh desktop install.")
        if not self.available:
            lines.append(f"Unavailable: {self.unavailable_reason or 'not available on mobile'}.")
        if self.readonly_reason:
            lines.append(f"{self.readonly_reason}.")
        if self.lock_reason:
            lines.append(f"Locked: {self.lock_reason}.")
        for note in spec_attr(self.spec, "discrepancies", ()) or ():
            lines.append(f"Desktop note: {note}")
        return "\n\n".join(lines)

    def open_help(self, e: Any = None) -> InfoSheet:
        actions: list[ft.Control] = []
        if self.stored and self.editable:
            actions.append(ft.TextButton(content="Reset to default", icon=ft.Icons.RESTART_ALT,
                                         on_click=lambda ev: (sheet.close(), self.reset())))
        sheet = InfoSheet(title=self.label, body=self.help_body(), actions=actions)
        if self.ctx.page is not None:
            sheet.show(self.ctx.page)
        return sheet


class SwitchTile(SettingTile):
    kind = "switch"

    def build_control(self) -> Optional[ft.Control]:
        self.switch = ft.Switch(value=False, on_change=self._on_switch)
        return self.switch

    def refresh_value(self) -> None:
        self.switch.value = bool(self.value())

    def _on_switch(self, e: Any = None) -> None:
        value = bool(getattr(getattr(e, "control", None), "value", self.switch.value))
        self.apply(value)

    def activate(self) -> Any:
        return self.apply(not bool(self.value()))


class NumberTile(SettingTile):
    kind = "number"

    @property
    def number_kind(self) -> str:
        return "int" if spec_type(self.spec) == "int" else "float"

    def bounds(self) -> tuple[Optional[float], Optional[float]]:
        def num(v: Any) -> Optional[float]:
            try:
                return None if v is None or isinstance(v, bool) else float(v)
            except (TypeError, ValueError):
                return None

        return num(spec_attr(self.spec, "minimum", None)), num(spec_attr(self.spec, "maximum", None))

    def build_editor_row(self) -> Optional[ft.Control]:
        low, high = self.bounds()
        hint = " · ".join(p for p in (f"min {summarize(low, 'number')}" if low is not None else "",
                                      f"max {summarize(high, 'number')}" if high is not None else "") if p)
        self.field = ft.TextField(
            color=ft.Colors.ON_SURFACE,
            value="", dense=True, width=150, keyboard_type=ft.KeyboardType.NUMBER,
            border_radius=tokens.RADII["field"], on_submit=self._on_submit, on_blur=self._on_submit,
        )
        self.minus = ft.IconButton(icon=ft.Icons.REMOVE, tooltip="Decrease", on_click=lambda e: self.step(-1),
                                   size_constraints=HIT_TARGET)
        self.plus = ft.IconButton(icon=ft.Icons.ADD, tooltip="Increase", on_click=lambda e: self.step(1),
                                  size_constraints=HIT_TARGET)
        controls: list[ft.Control] = [self.minus, self.field, self.plus]
        if hint:
            controls.append(ft.Text(hint, theme_style=ft.TextThemeStyle.LABEL_SMALL, color=ft.Colors.ON_SURFACE_VARIANT))
        return ft.Row(controls, spacing=4, vertical_alignment=ft.CrossAxisAlignment.CENTER, wrap=True)

    def refresh_value(self) -> None:
        value = self.value()
        self.field.value = "" if value is None else summarize(value, "number")

    def parse(self, raw: Any) -> Any:
        value = _parse_number(raw, self.number_kind)
        low, high = self.bounds()
        if low is not None and value < low:
            raise ValueError(f"Must be at least {summarize(low, 'number')}")
        if high is not None and value > high:
            raise ValueError(f"Must be at most {summarize(high, 'number')}")
        return value

    def step(self, direction: int) -> bool:
        current = self.value()
        try:
            base = _parse_number(current, self.number_kind) if current not in (None, "") else 0
        except ValueError:
            base = 0
        delta = 1 if self.number_kind == "int" else 0.1
        value = base + direction * delta
        if self.number_kind == "float":
            value = round(value, 6)
        low, high = self.bounds()
        if low is not None:
            value = max(low, value)
        if high is not None:
            value = min(high, value)
        if self.number_kind == "int":
            value = int(value)
        return self.apply(value)

    def _on_submit(self, e: Any = None) -> None:
        text = self.field.value or ""
        if text.strip() == summarize(self.value(), "number") and not self.error:
            return
        self.apply(text)


class SliderTile(NumberTile):
    kind = "slider"

    def build_editor_row(self) -> Optional[ft.Control]:
        low, high, divisions = slider_params(self.spec) or (0.0, 1.0, 10)
        self.slider = ft.Slider(min=low, max=high, divisions=divisions, value=low, label="{value}",
                                on_change=self._on_slide, on_change_end=self._on_slide_end, expand=True)
        self.slider_value = ft.Text("", theme_style=ft.TextThemeStyle.LABEL_MEDIUM, width=56,
                                    text_align=ft.TextAlign.END)
        return ft.Row([self.slider, self.slider_value], spacing=4, vertical_alignment=ft.CrossAxisAlignment.CENTER)

    def refresh_value(self) -> None:
        value = self.value()
        low, high, _divisions = slider_params(self.spec) or (0.0, 1.0, 10)
        try:
            number = float(value) if value is not None and not isinstance(value, bool) else low
        except (TypeError, ValueError):
            number = low
        self.slider.value = min(high, max(low, number))
        self.slider_value.value = summarize(value, "number") if value is not None else ""

    def _slider_number(self, raw: Any) -> Any:
        number = float(raw)
        return int(round(number)) if self.number_kind == "int" else round(number, 4)

    def _on_slide(self, e: Any = None) -> None:
        raw = getattr(getattr(e, "control", None), "value", self.slider.value)
        self.slider_value.value = summarize(self._slider_number(raw), "number")
        self.ctx.push(self.slider_value)

    def _on_slide_end(self, e: Any = None) -> None:
        raw = getattr(getattr(e, "control", None), "value", self.slider.value)
        self.apply(self._slider_number(raw))

    def step(self, direction: int) -> bool:  # no steppers on sliders
        return False


class _ChoiceTile(SettingTile):
    def options(self) -> list[tuple[Any, str]]:
        options = choice_options(self.spec)
        value = self.value()
        if value is not None and value != "" and not any(o == value or str(o) == str(value) for o, _l in options):
            options = options + [(value, f"{value} (custom)")]
        return options

    def index_of(self, value: Any) -> Optional[int]:
        for index, (option, _label) in enumerate(self.options()):
            if option == value and type(option) is type(value):
                return index
        for index, (option, _label) in enumerate(self.options()):
            if str(option) == str(value):
                return index
        return None

    def choose(self, index: int) -> bool:
        options = self.options()
        if not 0 <= index < len(options):
            return False
        return self.apply(options[index][0])


class SegmentedTile(_ChoiceTile):
    kind = "segmented"

    def build_editor_row(self) -> Optional[ft.Control]:
        self.segmented = ft.SegmentedButton(
            segments=[ft.Segment(value=str(i), label=ft.Text(label)) for i, (_v, label) in enumerate(self.options())],
            selected=[],
            allow_empty_selection=True,
            allow_multiple_selection=False,
            show_selected_icon=False,
            on_change=self._on_segment,
        )
        return ft.Row([self.segmented], scroll=ft.ScrollMode.AUTO)

    def refresh_value(self) -> None:
        options = self.options()
        if len(self.segmented.segments) != len(options):
            self.segmented.segments = [ft.Segment(value=str(i), label=ft.Text(label)) for i, (_v, label) in enumerate(options)]
        index = self.index_of(self.value())
        self.segmented.selected = [str(index)] if index is not None else []

    def _on_segment(self, e: Any = None) -> None:
        selected = list(getattr(getattr(e, "control", None), "selected", None) or self.segmented.selected or [])
        if selected:
            self.choose(int(selected[0]))


class DropdownTile(_ChoiceTile):
    kind = "dropdown"

    def build_editor_row(self) -> Optional[ft.Control]:
        self.dropdown = ft.Dropdown(
            color=ft.Colors.ON_SURFACE,
            options=[], value=None, dense=True, expand=True, enable_filter=True, editable=False,
            border_radius=tokens.RADII["field"], on_select=self._on_dropdown,
        )
        return ft.Row([self.dropdown])

    def refresh_value(self) -> None:
        self.dropdown.options = [ft.DropdownOption(key=str(i), text=label) for i, (_v, label) in enumerate(self.options())]
        index = self.index_of(self.value())
        self.dropdown.value = str(index) if index is not None else None

    def _on_dropdown(self, e: Any = None) -> None:
        value = getattr(getattr(e, "control", None), "value", None) or self.dropdown.value
        if value is not None:
            self.choose(int(value))


class TextTile(SettingTile):
    kind = "text"

    def build_editor_row(self) -> Optional[ft.Control]:
        self.field = ft.TextField(color=ft.Colors.ON_SURFACE, value="", dense=True, expand=True,
                                  border_radius=tokens.RADII["field"], on_submit=self._on_submit,
                                  on_blur=self._on_submit)
        return ft.Row([self.field])

    def refresh_value(self) -> None:
        value = self.value()
        self.field.value = "" if value is None else str(value)

    def _on_submit(self, e: Any = None) -> None:
        text = self.field.value or ""
        current = self.value()
        if text == ("" if current is None else str(current)) and not self.error:
            return
        self.apply(text)


class PromptTile(SettingTile):
    kind = "prompt"

    def activate(self) -> PromptEditor:
        default = self.default()
        self.editor = PromptEditor(
            self.ctx, title=self.label, subtitle=self.key, value=self.value() or "",
            default=None if default is None else str(default), on_save=self.save_from_editor,
        ).show()
        return self.editor


class SecretTile(SettingTile):
    kind = "secret"

    def activate(self) -> SecretEditor:
        current = self.store.get(self.path, "")
        self.editor = SecretEditor(self.ctx, title=self.label, subtitle=self.key,
                                   value="" if str(current or "").startswith("ENC:") else (current or ""),
                                   on_save=self.save_from_editor).show()
        return self.editor


class PathTile(SettingTile):
    kind = "path"

    def activate(self) -> PathEditor:
        self.editor = PathEditor(
            self.ctx, title=self.label, subtitle=self.key, value=self.value() or "", on_save=self.save_from_editor,
            import_dir=self.ctx.extras.get("import_dir"),
        ).show()
        return self.editor


class ListSettingTile(SettingTile):
    kind = "list"

    def activate(self) -> ListEditor:
        value = self.value()
        self.editor = ListEditor(self.ctx, title=self.label, subtitle=self.key,
                                 value=list(value) if isinstance(value, (list, tuple)) else [],
                                 on_save=self.save_from_editor).show()
        return self.editor


class JsonTile(SettingTile):
    kind = "json"

    def readonly(self) -> Optional[str]:
        # Key-pool lists (typed "secret") hold API keys: never shown or edited as raw JSON here.
        if spec_type(self.spec) in ("secret", "password"):
            return "Edited in API keys (U4)"
        return None

    def summary(self) -> str:
        if spec_type(self.spec) in ("secret", "password"):
            value = self.value()
            count = len(value) if isinstance(value, (list, dict)) else 0
            return f"{count} key{'s' if count != 1 else ''}" if count else "No keys"
        return super().summary()

    def activate(self) -> Optional[JsonEditor]:
        if self.readonly_reason:
            return None
        value = self.value()
        sample = value if value is not None else self.default()
        expect = dict if isinstance(sample, dict) or spec_type(self.spec) in ("dict", "object", "map", "mapping") else (
            list if isinstance(sample, (list, tuple)) or spec_type(self.spec) in ("list", "tuple", "array") else None
        )
        self.editor = JsonEditor(self.ctx, title=self.label, subtitle=self.key, value=value, expect=expect,
                                 on_save=self.save_from_editor).show()
        return self.editor


TILE_CLASSES: dict[str, type] = {
    "switch": SwitchTile,
    "number": NumberTile,
    "slider": SliderTile,
    "segmented": SegmentedTile,
    "dropdown": DropdownTile,
    "text": TextTile,
    "prompt": PromptTile,
    "secret": SecretTile,
    "path": PathTile,
    "list": ListSettingTile,
    "json": JsonTile,
}


def make_tile(spec: Any, ctx: Any, *, config: Optional[Mapping] = None) -> SettingTile:
    cls = TILE_CLASSES.get(tile_kind(spec, ctx.store.get(config_path(spec))), TextTile)
    return cls(spec, ctx, config=config)
