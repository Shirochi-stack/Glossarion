"""EntrySheet and the Resolve Gender sheet (UI_SPEC §4.1, §5.6; desktop editor cell edit + "Resolve Gender…").

EntrySheet: every column of the row (``glossary_column_fields`` except the internal
``_section``) - type (configured + present types, editable), raw / translated names,
gender (Male / Female / Unknown / empty), description (multi-line), custom fields -
plus "Resolve gender…" when the row's tracked gender conflicts, and Save / Delete /
Cancel. Save applies the changed columns through ``GlossaryService.update_entry`` (the
desktop cell-edit data step: one undo step for the sheet; standard columns stay as ''
when cleared). "＋ Entry" opens the same sheet for a new row.

Resolve Gender: the desktop "Resolve Tracked Gender" dialog - the Auto result, the
tracking history per gender (count / ratio / first / last), the flips, and the decision
Auto / Male / Female (``GlossaryService.resolve_gender``).
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Optional, Sequence

import flet as ft

from glossarion_mobile.ui.components._handlers import call_handler
from glossarion_mobile.ui.glossary.common import SheetHost, sheet

__all__ = ["EntrySheet", "GENDER_OPTIONS", "GenderSheet", "field_label"]

GENDER_OPTIONS = ("", "Male", "Female", "Unknown")


def field_label(name: str) -> str:
    """The desktop column header text (``field.replace('_', ' ').title()``)."""
    return str(name or "").replace("_", " ").title()


class EntrySheet:
    def __init__(
        self,
        ctx: Any,
        *,
        fields: Sequence[str],
        values: Mapping[str, Any],
        types: Sequence[str] = (),
        new: bool = False,
        conflict: Optional[str] = None,
        on_save: Optional[Callable[[dict], Any]] = None,
        on_delete: Optional[Callable[[], Any]] = None,
        on_resolve_gender: Optional[Callable[[], Any]] = None,
        text_size: float = 14,
        title: Optional[str] = None,
    ) -> None:
        self.ctx = ctx
        self.host = SheetHost(ctx)
        self.fields = [f for f in fields if not str(f).startswith("_")]
        self.initial = {f: _text(values.get(f)) for f in self.fields}
        self.new = new
        self.on_save = on_save
        self.on_delete = on_delete
        self.on_resolve_gender = on_resolve_gender
        self.inputs: dict = {}
        controls: list = []
        if conflict:
            controls.append(ft.Container(
                content=ft.Row([ft.Icon(ft.Icons.WARNING_AMBER, color=ft.Colors.ERROR, size=18),
                                ft.Text(conflict, expand=True, theme_style=ft.TextThemeStyle.BODY_SMALL)],
                               spacing=6),
                bgcolor=ft.Colors.ERROR_CONTAINER, border_radius=8, padding=8, key="entry-conflict"))
        for name in self.fields:
            control = self._input(name, self.initial[name], types, text_size)
            self.inputs[name] = control
            controls.append(control)
        actions: list = []
        if on_resolve_gender is not None and conflict:
            actions.append(ft.TextButton(content="Resolve gender…", icon=ft.Icons.WC,
                                         on_click=lambda e: self._resolve(), key="entry-resolve"))
        if on_delete is not None and not new:
            actions.append(ft.TextButton(content="Delete", icon=ft.Icons.DELETE_OUTLINE,
                                         style=ft.ButtonStyle(color=ft.Colors.ERROR),
                                         on_click=lambda e: self._delete(), key="entry-delete"))
        actions.append(ft.TextButton(content="Cancel", on_click=lambda e: self.close(), key="entry-cancel"))
        actions.append(ft.FilledButton(content="Add" if new else "Save", on_click=lambda e: self.save(),
                                       key="entry-save"))
        self.error = ft.Text("", color=ft.Colors.ERROR, visible=False, key="entry-error")
        controls.append(self.error)
        self.dialog = sheet(title or ("New entry" if new else "Edit entry"), controls, actions=actions,
                            key="entry-sheet")

    @staticmethod
    def _input(name: str, value: str, types: Sequence[str], text_size: float) -> ft.Control:
        label = field_label(name)
        style = ft.TextStyle(size=text_size)
        if name == "type":
            options = list(dict.fromkeys([t for t in types if t] + ([value] if value else [])))
            return ft.Dropdown(label=label, value=value or None, editable=True, enable_filter=True,
                               options=[ft.DropdownOption(key=t, text=t) for t in options], expand=True,
                               key=f"entry-{name}")
        if name == "gender":
            options = list(GENDER_OPTIONS)
            if value and value not in options:
                options.append(value)
            return ft.Dropdown(label=label, value=value, options=[
                ft.DropdownOption(key=g, text=g or "—") for g in options], expand=True, key=f"entry-{name}")
        multiline = name in ("description",) or len(value) > 80
        return ft.TextField(label=label, value=value, multiline=multiline, min_lines=1,
                            max_lines=6 if multiline else 1, text_style=style, key=f"entry-{name}")

    # ---- values ----------------------------------------------------------------------------------------

    def values(self) -> dict:
        out = {}
        for name, control in self.inputs.items():
            value = getattr(control, "value", "")
            if isinstance(control, ft.Dropdown) and name == "type":
                value = getattr(control, "text", None) or value
            out[name] = _text(value)
        return out

    def changed(self) -> dict:
        values = self.values()
        if self.new:
            return {k: v for k, v in values.items() if v}
        return {k: v for k, v in values.items() if v != self.initial.get(k, "")}

    # ---- actions ---------------------------------------------------------------------------------------

    def show(self, page: Any = None) -> "EntrySheet":
        self.host.open(self.dialog)
        return self

    def close(self) -> None:
        self.host.close()

    def save(self) -> Any:
        values = self.values() if self.new else self.changed()
        if self.new and not (values.get("raw_name") or values.get("translated_name") or values.get("original")):
            self.error.value = "Enter the raw or translated name"
            self.error.visible = True
            try:
                self.error.update()
            except Exception:
                pass
            return None
        self.close()
        if not values and not self.new:
            return None
        return call_handler(self.on_save, values)

    def _delete(self) -> Any:
        self.close()
        return call_handler(self.on_delete)

    def _resolve(self) -> Any:
        self.close()
        return call_handler(self.on_resolve_gender)


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    if isinstance(value, dict):
        return ", ".join(f"{k}: {v}" for k, v in value.items())
    return str(value)


class GenderSheet:
    """"Resolve Tracked Gender" (``_open_gender_resolution``) from ``GlossaryService.gender_model``:
    heading, overview, the "Tracking history" lines (``gender_history_line``), the flips
    (``gender_flip_line``), the note and the Auto / Male / Female decision."""

    def __init__(self, ctx: Any, model: Mapping[str, Any], *, on_apply: Callable[[str], Any]) -> None:
        self.ctx = ctx
        self.host = SheetHost(ctx)
        self.model = dict(model)
        self.on_apply = on_apply
        lines: list = [
            ft.Text(str(model.get("heading") or ""), theme_style=ft.TextThemeStyle.TITLE_SMALL,
                    weight=ft.FontWeight.W_600, key="gender-head"),
            ft.Text(str(model.get("overview") or ""), theme_style=ft.TextThemeStyle.BODY_SMALL, key="gender-overview"),
            ft.Text("Tracking history", theme_style=ft.TextThemeStyle.LABEL_LARGE, color=ft.Colors.PRIMARY),
        ]
        for index, line in enumerate(model.get("history") or []):
            lines.append(ft.Text(str(line), theme_style=ft.TextThemeStyle.BODY_SMALL, key=f"gender-history-{index}"))
        lines.append(ft.Text(str(model.get("flips") or ""), theme_style=ft.TextThemeStyle.BODY_SMALL,
                             key="gender-flips"))
        latest = [str(line) for line in model.get("latest") or []]
        if latest:
            lines.append(ft.Text("Latest flips:\n" + "\n".join(latest), theme_style=ft.TextThemeStyle.BODY_SMALL,
                                 key="gender-latest"))
        lines.append(ft.Text("Counts are unique chapter/file tracker observations, not textual mention counts.",
                             italic=True, theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT))
        self.group = ft.RadioGroup(value=str(model.get("decision") or "auto"), content=ft.Row([
            ft.Radio(value="auto", label="Auto"), ft.Radio(value="male", label="Male"),
            ft.Radio(value="female", label="Female")], wrap=True), key="gender-decision")
        lines.append(ft.Text("Decision", theme_style=ft.TextThemeStyle.LABEL_LARGE, color=ft.Colors.PRIMARY))
        lines.append(self.group)
        self.dialog = sheet("Resolve Tracked Gender", lines, actions=[
            ft.TextButton(content="Cancel", on_click=lambda e: self.host.close()),
            ft.FilledButton(content="Apply", on_click=lambda e: self.apply(), key="gender-apply"),
        ], key="gender-sheet")

    def show(self, page: Any = None) -> "GenderSheet":
        self.host.open(self.dialog)
        return self

    def apply(self, decision: Optional[str] = None) -> Any:
        value = decision or self.group.value or "auto"
        self.host.close()
        return call_handler(self.on_apply, value)
