"""Full-screen editors opened by settings tiles (UI_SPEC §5.7).

``PromptEditor`` (mono multi-line text, char / approximate token count, Reset
to default), ``SecretEditor`` (masked field with reveal, Clear), ``PathEditor``
(path field, Import… into app data, Clear), ``ListEditor`` (rows with
add / remove / move up / move down) and ``JsonEditor`` (validated JSON).

Each is a ``BottomSheet(fullscreen=True)`` opened with ``page.show_dialog``.
``on_save(value)`` returns ``None`` on success or an error string, which is
shown under the field while the editor stays open.
"""

from __future__ import annotations

import json
import os
import shutil
from typing import Any, Callable, Optional, Sequence

import flet as ft

from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.theme import HIT_TARGET, mono_family

__all__ = ["FullScreenEditor", "JsonEditor", "ListEditor", "PathEditor", "PromptEditor", "SecretEditor", "count_label"]

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
    ) -> None:
        self.ctx = ctx
        self.title = title
        self.on_save = on_save
        self.saved_value: Any = None
        self.closed = False
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
        self.ctx.show_dialog(self.sheet)
        return self

    def close(self, e: Any = None) -> None:
        self.closed = True
        if getattr(self.sheet, "open", False):
            self.ctx.pop_dialog()

    def set_error(self, message: Optional[str]) -> None:
        self.error_text.value = message or ""
        self.error_text.visible = bool(message)
        self.ctx.push(self.error_text)

    def save(self) -> bool:
        try:
            value = self.collect()
        except ValueError as exc:
            self.set_error(str(exc))
            return False
        error = self.on_save(value) if self.on_save is not None else None
        if error:
            self.set_error(error)
            return False
        self.saved_value = value
        self.set_error(None)
        self.close()
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
