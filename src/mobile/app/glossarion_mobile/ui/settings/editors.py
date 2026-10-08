"""Full-screen editors opened by settings tiles (UI_SPEC §5.7).

``PromptEditor`` (mono multi-line text, char / approximate token count, Reset
to default), ``SecretEditor`` (masked field with reveal, Clear), ``PathEditor``
(path field, Import… into app data, Clear), ``ListEditor`` (rows with
add / remove / move up / move down) and ``JsonEditor`` (validated JSON).

Each is a ``BottomSheet(fullscreen=True)`` opened with ``page.show_dialog``.
``on_save(value)`` (sync or async) returns ``None`` on success or an error string,
which is shown under the field while the editor stays open. While a save runs
Save and Close are disabled and further taps are ignored; on success the editor
closes itself (by identity, never the topmost dialog) and then calls
``on_saved(value)``, the place for a confirmation snackbar.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import shutil
from typing import Any, Callable, Optional, Sequence

import flet as ft

from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.components._handlers import call_handler
from glossarion_mobile.ui.components.dialogs import close_dialog
from glossarion_mobile.ui.theme import HIT_TARGET, mono_family

log = logging.getLogger("glossarion.ui")

__all__ = ["EntryTypePickerEditor", "EntryTypesEditor", "FullScreenEditor", "JsonEditor", "ListEditor",
           "MultiplierEditor", "PathEditor", "PromptEditor", "SecretEditor",
           "count_label"]

SaveHandler = Callable[[Any], Optional[str]]


def count_label(text: str) -> str:
    """``"1,234 chars · ≈ 309 tokens"`` (chars / 4, the usual rough estimate; no tokenizer on the UI loop)."""
    chars = len(text or "")
    return f"{chars:,} chars · ≈ {max(0, (chars + 3) // 4):,} tokens"


class FullScreenEditor:
    def __init__(
        self,
        ctx: Any,
        *,
        title: str,
        subtitle: Optional[str] = None,
        on_save: Optional[SaveHandler] = None,
        save_label: str = "Save",
        on_saved: Optional[Callable[[Any], Any]] = None,
    ) -> None:
        self.ctx = ctx
        self.title = title
        self.on_save = on_save
        self.on_saved = on_saved  # runs once the editor has closed (feedback)
        self.saved_value: Any = None
        self.closed = False
        self.saving = False
        self.error_text = ft.Text("", color=ft.Colors.ERROR, theme_style=ft.TextThemeStyle.BODY_SMALL, visible=False)
        self.save_button = ft.FilledButton(content=save_label, on_click=self._on_save)
        self.close_button = ft.IconButton(icon=ft.Icons.CLOSE, tooltip="Close", on_click=self.close, size_constraints=HIT_TARGET)
        heading: list[ft.Control] = [
            ft.Text(title, theme_style=ft.TextThemeStyle.TITLE_MEDIUM, weight=ft.FontWeight.W_600, color=ft.Colors.ON_SURFACE,
                    max_lines=1, overflow=ft.TextOverflow.ELLIPSIS)
        ]
        if subtitle:
            heading.append(
                ft.Text(subtitle, theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT,
                        max_lines=1, overflow=ft.TextOverflow.ELLIPSIS)
            )
        self.header = ft.Row(
            [self.close_button, ft.Column(heading, spacing=0, tight=True, expand=True), self.save_button],
            spacing=8,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )
        self.body = self.build_body()
        self.footer = self.build_footer()
        column: list[ft.Control] = [self.header, self.body, self.error_text]
        if self.footer is not None:
            column.append(self.footer)
        self.sheet = ft.BottomSheet(
            content=ft.SafeArea(
                content=ft.Container(
                    padding=ft.Padding.only(left=8, right=12, top=8, bottom=12),
                    content=ft.Column(column, spacing=tokens.SPACING["sm"], expand=True),
                ),
                expand=True,
            ),
            fullscreen=True,
            scrollable=False,
            show_drag_handle=False,
            bgcolor=ft.Colors.SURFACE,
        )

    # ---- subclass hooks ------------------------------------------------------------------

    def build_body(self) -> ft.Control:
        raise NotImplementedError

    def build_footer(self) -> Optional[ft.Control]:
        return None

    def collect(self) -> Any:
        raise NotImplementedError

    # ---- lifecycle -------------------------------------------------------------------------

    def show(self) -> "FullScreenEditor":
        if self.closed:  # shown again after a save / close
            self.closed = False
            self._set_busy(False)
        self.ctx.show_dialog(self.sheet)
        return self

    def close(self, e: Any = None) -> None:
        self.closed = True
        # This sheet itself: page.pop_dialog() would close a snackbar shown since it opened.
        close_dialog(getattr(self.ctx, "page", None), self.sheet)

    def set_error(self, message: Optional[str]) -> None:
        self.error_text.value = message or ""
        self.error_text.visible = bool(message)
        self.ctx.push(self.error_text)

    def _set_busy(self, busy: bool) -> None:
        self.saving = busy
        self.save_button.disabled = busy
        self.close_button.disabled = busy
        self.ctx.push(self.save_button, self.close_button)

    def save(self) -> Any:
        """Validate, store (``on_save``) and close; True when saved, False otherwise (an async
        ``on_save`` gives an awaitable of that). Ignored while a save runs and once closed, so a
        repeated tap never stores the value twice."""
        if self.closed or self.saving:
            return False
        try:
            value = self.collect()
        except ValueError as exc:
            self.set_error(str(exc))
            return False
        self._set_busy(True)
        try:
            result = self.on_save(value) if self.on_save is not None else None
        except Exception as exc:
            log.exception("saving %s failed", self.title)
            return self._finish(value, str(exc) or type(exc).__name__)
        if inspect.isawaitable(result):
            return asyncio.ensure_future(self._finish_async(value, result))
        return self._finish(value, result)

    async def _finish_async(self, value: Any, pending: Any) -> bool:
        try:
            error = await pending
        except Exception as exc:
            log.exception("saving %s failed", self.title)
            error = str(exc) or type(exc).__name__
        return self._finish(value, error)

    def _finish(self, value: Any, error: Any) -> bool:
        if error:
            self._set_busy(False)
            self.set_error(str(error))
            return False
        self.saved_value = value
        self.set_error(None)
        self.saving = False  # the buttons stay disabled while the sheet closes
        self.close()
        call_handler(self.on_saved, value)
        return True

    def _on_save(self, e: Any = None) -> None:
        self.save()


class PromptEditor(FullScreenEditor):
    def __init__(self, ctx: Any, *, title: str, value: str, default: Optional[str] = None,
                 on_save: Optional[SaveHandler] = None, subtitle: Optional[str] = None) -> None:
        self.initial = "" if value is None else str(value)
        self.default = default
        super().__init__(ctx, title=title, subtitle=subtitle, on_save=on_save)

    def build_body(self) -> ft.Control:
        self.field = ft.TextField(
            color=ft.Colors.ON_SURFACE,
            value=self.initial,
            multiline=True,
            min_lines=12,
            expand=True,
            text_style=ft.TextStyle(font_family=mono_family(self.ctx.page), size=13),
            border_radius=tokens.RADII["field"],
            on_change=self._on_change,
        )
        self.counter = ft.Text(count_label(self.initial), theme_style=ft.TextThemeStyle.LABEL_SMALL,
                               color=ft.Colors.ON_SURFACE_VARIANT)
        return ft.Container(content=self.field, expand=True)

    def build_footer(self) -> Optional[ft.Control]:
        self.reset_button = ft.TextButton(
            content="Reset to default", icon=ft.Icons.RESTART_ALT, on_click=self._on_reset,
            disabled=self.default is None,
        )
        return ft.Row([self.counter, ft.Container(expand=True), self.reset_button],
                      vertical_alignment=ft.CrossAxisAlignment.CENTER)

    @property
    def dirty(self) -> bool:
        return (self.field.value or "") != self.initial

    def _on_change(self, e: Any = None) -> None:
        self.counter.value = count_label(self.field.value or "")
        self.ctx.push(self.counter)

    def _on_reset(self, e: Any = None) -> None:
        if self.default is None:
            return
        self.field.value = str(self.default)
        self._on_change()
        self.ctx.push(self.field)

    def collect(self) -> Any:
        return self.field.value or ""


class SecretEditor(FullScreenEditor):
    def __init__(self, ctx: Any, *, title: str, value: str, on_save: Optional[SaveHandler] = None,
                 subtitle: Optional[str] = None) -> None:
        self.initial = "" if value is None else str(value)
        super().__init__(ctx, title=title, subtitle=subtitle, on_save=on_save)

    def build_body(self) -> ft.Control:
        self.field = ft.TextField(
            color=ft.Colors.ON_SURFACE,
            value=self.initial,
            password=True,
            can_reveal_password=True,
            autocorrect=False,
            enable_suggestions=False,
            label="Value",
            border_radius=tokens.RADII["field"],
        )
        self.clear_button = ft.TextButton(content="Clear", icon=ft.Icons.BACKSPACE_OUTLINED, on_click=self._on_clear)
        note = ft.Text(
            "Stored encrypted in config.json with the key kept in the device's secure storage.",
            theme_style=ft.TextThemeStyle.BODY_SMALL,
            color=ft.Colors.ON_SURFACE_VARIANT,
        )
        return ft.Column([self.field, ft.Row([self.clear_button]), note], spacing=8, expand=True)

    def _on_clear(self, e: Any = None) -> None:
        self.field.value = ""
        self.ctx.push(self.field)

    def collect(self) -> Any:
        return (self.field.value or "").strip()


class PathEditor(FullScreenEditor):
    """Path field + Import… (copies the picked file into ``import_dir``) + Clear."""

    def __init__(self, ctx: Any, *, title: str, value: str, on_save: Optional[SaveHandler] = None,
                 import_dir: Optional[str] = None, allowed_extensions: Optional[Sequence[str]] = None,
                 subtitle: Optional[str] = None) -> None:
        self.initial = "" if value is None else str(value)
        self.import_dir = import_dir
        self.allowed_extensions = list(allowed_extensions) if allowed_extensions else None
        super().__init__(ctx, title=title, subtitle=subtitle, on_save=on_save)

    def build_body(self) -> ft.Control:
        self.field = ft.TextField(color=ft.Colors.ON_SURFACE, value=self.initial, label="Path",
                                  border_radius=tokens.RADII["field"])
        can_import = self.import_dir is not None and self.ctx.file_picker_factory is not None
        self.import_button = ft.FilledTonalButton(
            content="Import…", icon=ft.Icons.FILE_OPEN_OUTLINED, on_click=self._on_import, disabled=not can_import,
            tooltip=None if can_import else "File import is not available here",
        )
        self.clear_button = ft.TextButton(content="Clear", icon=ft.Icons.BACKSPACE_OUTLINED, on_click=self._on_clear)
        note = ft.Text(
            "Imported files are copied into the app's data folder; arbitrary device folders cannot be referenced.",
            theme_style=ft.TextThemeStyle.BODY_SMALL,
            color=ft.Colors.ON_SURFACE_VARIANT,
        )
        return ft.Column([self.field, ft.Row([self.import_button, self.clear_button], spacing=8), note],
                         spacing=8, expand=True)

    def _on_clear(self, e: Any = None) -> None:
        self.field.value = ""
        self.ctx.push(self.field)

    async def _on_import(self, e: Any = None) -> Optional[str]:
        if self.import_dir is None or self.ctx.file_picker_factory is None:
            return None
        try:
            picker = self.ctx.file_picker_factory()
            files = await picker.pick_files(dialog_title=self.title, allowed_extensions=self.allowed_extensions)
        except Exception as exc:
            self.set_error(f"Import failed: {exc}")
            return None
        if not files:
            return None
        source = getattr(files[0], "path", None)
        if not source:
            self.set_error("The picked file has no local path on this platform.")
            return None
        try:
            target = await self.ctx.run_io(self._copy_into_app_data, source)
        except Exception as exc:
            self.set_error(f"Import failed: {exc}")
            return None
        self.field.value = target
        self.set_error(None)
        self.ctx.push(self.field)
        return target

    def _copy_into_app_data(self, source: str) -> str:
        os.makedirs(self.import_dir, exist_ok=True)
        target = os.path.join(self.import_dir, os.path.basename(source))
        if os.path.abspath(source) != os.path.abspath(target):
            shutil.copy2(source, target)
        return target

    def collect(self) -> Any:
        return (self.field.value or "").strip()


class ListEditor(FullScreenEditor):
    """Rows of strings with add / remove / reorder (up / down buttons)."""

    def __init__(self, ctx: Any, *, title: str, value: Sequence[Any], on_save: Optional[SaveHandler] = None,
                 subtitle: Optional[str] = None) -> None:
        self.items: list[str] = [str(v) for v in (value or [])]
        super().__init__(ctx, title=title, subtitle=subtitle, on_save=on_save)

    def build_body(self) -> ft.Control:
        self.rows = ft.ListView(expand=True, spacing=4)
        self.add_button = ft.FilledTonalButton(content="Add", icon=ft.Icons.ADD, on_click=self._on_add)
        self._render()
        return ft.Column([self.rows, ft.Row([self.add_button])], spacing=8, expand=True)

    def _sync_from_fields(self) -> None:
        fields = getattr(self, "_fields", [])
        if fields:
            self.items = [f.value or "" for f in fields]

    def _render(self) -> None:
        self._fields: list[ft.TextField] = []
        controls: list[ft.Control] = []
        for index, item in enumerate(self.items):
            field = ft.TextField(color=ft.Colors.ON_SURFACE, value=item, dense=True, expand=True,
                                 border_radius=tokens.RADII["field"])
            self._fields.append(field)
            controls.append(
                ft.Row(
                    [
                        field,
                        ft.IconButton(icon=ft.Icons.ARROW_UPWARD, tooltip="Move up", disabled=index == 0,
                                      on_click=lambda e, i=index: self.move(i, -1), size_constraints=HIT_TARGET),
                        ft.IconButton(icon=ft.Icons.ARROW_DOWNWARD, tooltip="Move down",
                                      disabled=index == len(self.items) - 1,
                                      on_click=lambda e, i=index: self.move(i, 1), size_constraints=HIT_TARGET),
                        ft.IconButton(icon=ft.Icons.DELETE_OUTLINE, tooltip="Remove",
                                      on_click=lambda e, i=index: self.remove(i), size_constraints=HIT_TARGET),
                    ],
                    spacing=0,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    key=f"row-{index}",
                )
            )
        if not controls:
            controls.append(ft.Text("No items", theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT))
        self.rows.controls = controls

    def _rerender(self) -> None:
        self._render()
        self.ctx.push(self.rows)

    def add(self, value: str = "") -> None:
        self._sync_from_fields()
        self.items.append(value)
        self._rerender()

    def remove(self, index: int) -> None:
        self._sync_from_fields()
        if 0 <= index < len(self.items):
            del self.items[index]
            self._rerender()

    def move(self, index: int, delta: int) -> None:
        self._sync_from_fields()
        target = index + delta
        if 0 <= index < len(self.items) and 0 <= target < len(self.items):
            self.items[index], self.items[target] = self.items[target], self.items[index]
            self._rerender()

    def _on_add(self, e: Any = None) -> None:
        self.add("")

    def collect(self) -> Any:
        self._sync_from_fields()
        return [item for item in (s.strip() for s in self.items) if item]


class JsonEditor(FullScreenEditor):
    """JSON text with validation; ``expect`` is ``dict``, ``list`` or ``None`` (any)."""

    def __init__(self, ctx: Any, *, title: str, value: Any, on_save: Optional[SaveHandler] = None,
                 expect: Optional[type] = None, subtitle: Optional[str] = None) -> None:
        self.expect = expect
        try:
            self.initial = json.dumps(value, ensure_ascii=False, indent=2) if value is not None else ""
        except (TypeError, ValueError):
            self.initial = str(value)
        super().__init__(ctx, title=title, subtitle=subtitle, on_save=on_save)

    def build_body(self) -> ft.Control:
        self.field = ft.TextField(
            color=ft.Colors.ON_SURFACE,
            value=self.initial,
            multiline=True,
            min_lines=12,
            expand=True,
            text_style=ft.TextStyle(font_family=mono_family(self.ctx.page), size=13),
            border_radius=tokens.RADII["field"],
        )
        return ft.Container(content=self.field, expand=True)

    def collect(self) -> Any:
        text = (self.field.value or "").strip()
        if not text:
            return {} if self.expect is dict else [] if self.expect is list else None
        try:
            value = json.loads(text)
        except ValueError as exc:
            raise ValueError(f"Invalid JSON: {exc}") from exc
        if self.expect is dict and not isinstance(value, dict):
            raise ValueError("Expected a JSON object ({…})")
        if self.expect is list and not isinstance(value, list):
            raise ValueError("Expected a JSON list ([…])")
        return value


class MultiplierEditor(FullScreenEditor):
    """QA Scanner › Word count: the per-language multiplier grid (desktop ``word_multiplier_sliders``).

    ``defaults`` are the shared factory multipliers (``qa_scan_runtime.CANONICAL_WORD_COUNT_MULTIPLIERS``,
    the desktop ``base_multiplier_defaults``); ``value`` (the stored dict) is merged over them for
    display, like the desktop dialog. Save returns every language's multiplier (the desktop saves
    the whole grid); "Reset to defaults" refills the factory values."""

    def __init__(self, ctx: Any, *, title: str, value: Any, defaults: dict, on_save: Optional[SaveHandler] = None,
                 subtitle: Optional[str] = None) -> None:
        self.defaults = {str(k): float(v) for k, v in dict(defaults or {}).items()}
        merged = dict(self.defaults)
        if isinstance(value, dict):
            for key, number in value.items():
                try:
                    merged[str(key)] = float(number)
                except (TypeError, ValueError):
                    continue
        self.values = merged
        self.fields: dict = {}
        super().__init__(ctx, title=title, subtitle=subtitle, on_save=on_save)

    def build_body(self) -> ft.Control:
        rows: list = []
        for language, number in self.values.items():
            field = ft.TextField(value=f"{number:g}", dense=True, width=110, keyboard_type=ft.KeyboardType.NUMBER,
                                 border_radius=tokens.RADII["field"], color=ft.Colors.ON_SURFACE,
                                 key=f"multiplier-{language}")
            self.fields[language] = field
            rows.append(ft.Row([ft.Text(language.title(), expand=True, theme_style=ft.TextThemeStyle.BODY_MEDIUM),
                                field], vertical_alignment=ft.CrossAxisAlignment.CENTER))
        return ft.ListView(controls=rows, expand=True, spacing=4)

    def build_footer(self) -> Optional[ft.Control]:
        return ft.TextButton(content="Reset to defaults", icon=ft.Icons.RESTART_ALT, on_click=self._reset)

    def _reset(self, e: Any = None) -> None:
        for language, field in self.fields.items():
            field.value = f"{self.defaults.get(language, 1.0):g}"
        try:
            self.body.update()
        except Exception:
            pass

    def collect(self) -> Any:
        out: dict = {}
        for language, field in self.fields.items():
            text = str(field.value or "").strip().replace(",", ".")
            try:
                number = float(text)
            except ValueError:
                raise ValueError(f"{language.title()}: enter a number") from None
            if not 0.05 <= number <= 10.0:
                raise ValueError(f"{language.title()}: use a multiplier between 0.05 and 10")
            out[language] = round(number, 4)
        return out


def _glossary_document() -> Any:
    try:
        import glossary_document  # shared, GUI-free (U6)

        return glossary_document
    except Exception:
        log.debug("glossary_document unavailable", exc_info=True)
        return None


class EntryTypesEditor(FullScreenEditor):
    """Glossary › Balanced/Full › Entry Type Configuration (desktop "Active Entry Types" + "Add Custom
    Type"): a switch per type (built-in first), "(has gender field)", × for custom types (with the
    desktop confirmation), and Add type with "Include gender field". The rules are the shared
    ``glossary_document`` ones the desktop dialog calls (``add_entry_type`` /
    ``entry_type_remove_warning`` / ``sorted_entry_types`` / ``normalize_legacy_entry_types``)."""

    def __init__(self, ctx: Any, *, title: str, value: Any, on_save: Optional[SaveHandler] = None,
                 subtitle: Optional[str] = None) -> None:
        self.gd = _glossary_document()
        types = {str(k): dict(v) if isinstance(v, dict) else {"enabled": bool(v), "has_gender": False}
                 for k, v in (value or {}).items()} if isinstance(value, dict) else {}
        if self.gd is not None:
            self.gd.normalize_legacy_entry_types(types)
        self.types: dict = types
        self.switches: dict = {}
        self.pending_remove: Optional[str] = None
        super().__init__(ctx, title=title, subtitle=subtitle, on_save=on_save)

    def _ordered(self) -> list:
        if self.gd is not None:
            return self.gd.sorted_entry_types(self.types)
        return sorted(self.types.items(), key=lambda x: (x[0] not in ["character", "terms"], x[0]))

    def build_body(self) -> ft.Control:
        self.rows = ft.ListView(expand=True, spacing=2, key="entry-types-rows")
        self.new_name = ft.TextField(label="Type field", dense=True, expand=True, border_radius=tokens.RADII["field"],
                                     color=ft.Colors.ON_SURFACE, key="entry-types-new", on_submit=lambda e: self.add())
        self.new_gender = ft.Switch(label="Include gender field", value=False, key="entry-types-gender")
        self._render()
        return ft.Column([
            ft.Text("Active Entry Types", theme_style=ft.TextThemeStyle.TITLE_SMALL),
            self.rows,
            ft.Text("Add Custom Type", theme_style=ft.TextThemeStyle.TITLE_SMALL),
            ft.Row([self.new_name, ft.FilledTonalButton(content="Add type", icon=ft.Icons.ADD,
                                                        on_click=lambda e: self.add(), key="entry-types-add")],
                   vertical_alignment=ft.CrossAxisAlignment.CENTER),
            self.new_gender,
        ], spacing=8, expand=True)

    def _sync(self) -> None:
        for name, switch in self.switches.items():
            if name in self.types:
                self.types[name]["enabled"] = bool(switch.value)

    def _render(self) -> None:
        self.switches = {}
        controls: list = []
        builtin = getattr(self.gd, "BUILTIN_ENTRY_TYPES", ("character", "terms"))
        for name, config in self._ordered():
            switch = ft.Switch(value=bool(config.get("enabled", True)), key=f"entry-type-{name}")
            self.switches[name] = switch
            row: list = [switch, ft.Text(name, expand=True, theme_style=ft.TextThemeStyle.BODY_MEDIUM)]
            if config.get("has_gender", False):
                row.insert(2, ft.Text("(has gender field)", theme_style=ft.TextThemeStyle.LABEL_SMALL,
                                      color=ft.Colors.ON_SURFACE_VARIANT))
            if name not in builtin:
                row.append(ft.IconButton(icon=ft.Icons.CLOSE, tooltip=f"Remove {name}", icon_color=ft.Colors.ERROR,
                                         on_click=lambda e, n=name: self.ask_remove(n), size_constraints=HIT_TARGET,
                                         key=f"entry-type-remove-{name}"))
            controls.append(ft.Row(row, vertical_alignment=ft.CrossAxisAlignment.CENTER, spacing=6))
        self.rows.controls = controls

    def _rerender(self) -> None:
        self._render()
        self.ctx.push(self.rows)

    def add(self, name: Optional[str] = None, has_gender: Optional[bool] = None) -> Optional[str]:
        """Add Type: the shared rule (lower-cased; blank and duplicate names refused)."""
        self._sync()
        text = self.new_name.value if name is None else name
        gender = bool(self.new_gender.value) if has_gender is None else bool(has_gender)
        if self.gd is not None:
            type_name, warning = self.gd.add_entry_type(self.types, str(text or ""), gender)
        else:
            type_name = str(text or "").strip().lower()
            warning = None if type_name and type_name not in self.types else ("Invalid Input", "Please enter a type name")
            if warning is None:
                self.types[type_name] = {"enabled": True, "has_gender": gender}
        if warning:
            self.set_error(f"{warning[0]}: {warning[1]}")
            return None
        self.set_error(None)
        self.new_name.value = ""
        self.new_gender.value = False
        self._rerender()
        self.ctx.push(self.new_name, self.new_gender)
        return type_name

    def ask_remove(self, name: str) -> Any:
        warning = self.gd.entry_type_remove_warning(name) if self.gd is not None else None
        if warning:
            self.set_error(f"{warning[0]}: {warning[1]}")
            return None
        from glossarion_mobile.ui.components.dialogs import ConfirmDialog

        self.pending_remove = name
        dialog = ConfirmDialog(title="Confirm Removal", body=f"Remove type '{name}'?", confirm_label="Remove",
                               destructive=True, on_confirm=lambda: self.remove(name))
        if getattr(self.ctx, "page", None) is not None:
            dialog.show(self.ctx.page)
        return dialog

    def remove(self, name: str) -> bool:
        self._sync()
        warning = self.gd.entry_type_remove_warning(name) if self.gd is not None else None
        if warning or name not in self.types:
            return False
        del self.types[name]
        self._rerender()
        return True

    def collect(self) -> Any:
        self._sync()
        return {name: dict(config) for name, config in self.types.items()}


class EntryTypePickerEditor(FullScreenEditor):
    """A checkbox per entry type (desktop "Configure…" / Refinement "Selected Entry Types"): the
    configured ``custom_entry_types``; a saved value outside them stays listed (ticked) so it can be
    cleared. A saved "terms" also ticks "term" and the other way round (desktop picker rule)."""

    def __init__(self, ctx: Any, *, title: str, value: Sequence[Any], options: Sequence[str],
                 on_save: Optional[SaveHandler] = None, subtitle: Optional[str] = None, prompt: str = "") -> None:
        saved = [str(v).strip() for v in (value or []) if str(v).strip()]
        lowered = set()
        for item in saved:
            low = item.lower()
            lowered.update((low, low[:-1] if low.endswith("s") else low + "s"))
        names = [str(o) for o in options]
        for item in saved:
            if item.lower() not in {n.lower() for n in names} and not any(
                    item.lower() in (n.lower(), n.lower() + "s", n.lower()[:-1]) for n in names):
                names.append(item)
        self.options = names
        self.checked = {name: name.lower() in lowered for name in names}
        self.prompt = prompt
        self.boxes: dict = {}
        super().__init__(ctx, title=title, subtitle=subtitle, on_save=on_save)

    def build_body(self) -> ft.Control:
        controls: list = []
        if self.prompt:
            controls.append(ft.Text(self.prompt, theme_style=ft.TextThemeStyle.BODY_SMALL,
                                    color=ft.Colors.ON_SURFACE_VARIANT))
        for name in self.options:
            box = ft.Checkbox(label=name.capitalize(), value=self.checked.get(name, False), key=f"entry-pick-{name}")
            self.boxes[name] = box
            controls.append(box)
        if not self.options:
            controls.append(ft.Text("No entry types configured (Glossary › Balanced/Full › Entry Type Configuration).",
                                    theme_style=ft.TextThemeStyle.BODY_SMALL))
        return ft.ListView(controls=controls, expand=True, spacing=2)

    def collect(self) -> Any:
        return [name for name in self.options if self.boxes.get(name) is not None and self.boxes[name].value]
