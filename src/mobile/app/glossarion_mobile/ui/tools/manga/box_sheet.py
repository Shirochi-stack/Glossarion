"""BoxSheet (UI_SPEC §4.6 Editor, §5.10): long-press a box.

The desktop box context menu as a sheet: tabs "📝 OCR Recognition Result" and "🌍 Translation
Result" (the desktop popups' titles) to edit the texts, **Save**, **Save & Update Overlay**
(re-render the page with the edited text), **OCR this text**, **Translate this text**, **Clean
this box**, the box type (free text / bubble text: the renderer's background and the cleaner's
dilation follow it), **Exclude from Clean**, the inpainting iterations (Auto or 0-50) and
**Delete**. Every action is a ``MangaEditorSession`` call the editor tab makes (box edits on the
io pool, OCR / translate / clean / re-render as editor jobs).
"""

from __future__ import annotations

import itertools
from typing import Any, Callable, Mapping, Optional

import flet as ft

from glossarion_mobile.ui.components.dialogs import close_dialog
from glossarion_mobile.ui.components.sheet import bottom_sheet, scroll_column, sheet_frame

__all__ = ["BoxSheet", "ITERATION_CHOICES", "NOT_RECOGNIZED"]

#: Inpainting iterations: -1 = Auto (the dialog's range is -1..50).
ITERATION_CHOICES = (-1, 0, 1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 50)
_SHEETS = itertools.count(1)  # a new key per sheet (never a reused id())
NOT_RECOGNIZED = "Run “OCR this text” (or Recognize) first"


class BoxSheet:
    """``on_action(name, index, values)``: ``save`` / ``save_render`` / ``ocr`` / ``translate`` /
    ``clean`` / ``delete``; ``values`` = the edited texts and flags (always applied first)."""

    def __init__(self, box: Mapping[str, Any], index: int, *, on_action: Callable[[str, int, dict], Any],
                 busy_reason: Optional[str] = None, steps_reason: Optional[str] = None, tab: str = "ocr") -> None:
        self.box = dict(box)
        self.index = index
        self.on_action = on_action
        self._page: Any = None
        # As on the desktop, the OCR text is edited (and translated) once the box has been
        # recognized: the editor keeps a typed text only for a recognized box.
        recognized = bool(str(box.get("ocr_text") or "").strip())
        self.ocr_field = ft.TextField(value=str(box.get("ocr_text") or ""), multiline=True, min_lines=4, max_lines=10,
                                      label="Recognized text", key="box-ocr-text", read_only=not recognized,
                                      helper=None if recognized else NOT_RECOGNIZED)
        self.translation_field = ft.TextField(value=str(box.get("translation") or ""), multiline=True, min_lines=4,
                                              max_lines=10, label="Translation", key="box-translation")
        self.original_text = ft.Text(str(box.get("ocr_text") or ""), selectable=True,
                                     theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT,
                                     key="box-original")
        self.free_text = ft.Switch(label="Free text (off: bubble text)", value=bool(box.get("free_text")),
                                   key="box-free-text")
        self.exclude = ft.Switch(label="Exclude from Clean", value=bool(box.get("exclude_from_clean")),
                                 key="box-exclude")
        iterations = box.get("inpaint_iterations")
        current = -1 if iterations in (None, "") else int(iterations)
        self.iterations = ft.Dropdown(
            label="Inpainting iterations", dense=True, width=200, key="box-iterations",
            value=str(current if current in ITERATION_CHOICES else -1),
            options=[ft.DropdownOption(key=str(v), text="Auto" if v == -1 else str(v)) for v in ITERATION_CHOICES])
        reason = busy_reason or steps_reason
        self.save_button = ft.FilledTonalButton(content="Save", icon=ft.Icons.SAVE, key="box-save",
                                                on_click=lambda e: self.act("save"), disabled=busy_reason is not None)
        self.save_render_button = ft.FilledButton(content="Save & Update Overlay", icon=ft.Icons.AUTO_FIX_NORMAL,
                                                  key="box-save-render", tooltip=reason, disabled=reason is not None,
                                                  on_click=lambda e: self.act("save_render"))
        self.ocr_button = ft.TextButton(content="OCR this text", icon=ft.Icons.DOCUMENT_SCANNER, key="box-ocr",
                                        tooltip=reason, disabled=reason is not None, on_click=lambda e: self.act("ocr"))
        translate_reason = reason or (None if recognized else NOT_RECOGNIZED)
        self.translate_button = ft.TextButton(content="Translate this text", icon=ft.Icons.TRANSLATE,
                                              key="box-translate", tooltip=translate_reason,
                                              disabled=translate_reason is not None,
                                              on_click=lambda e: self.act("translate"))
        self.clean_button = ft.TextButton(content="Clean this box", icon=ft.Icons.CLEANING_SERVICES, key="box-clean",
                                          tooltip=reason, disabled=reason is not None,
                                          on_click=lambda e: self.act("clean"))
        self.delete_button = ft.TextButton(content="Delete box", icon=ft.Icons.DELETE_OUTLINE, key="box-delete",
                                           style=ft.ButtonStyle(color=ft.Colors.ERROR), disabled=busy_reason is not None,
                                           on_click=lambda e: self.act("delete"))
        self.tabs = ft.Tabs(
            length=2, selected_index=1 if tab == "translation" else 0, key="box-tabs",
            content=ft.Column([
                ft.TabBar(tabs=[ft.Tab(label="📝 OCR Recognition Result"), ft.Tab(label="🌍 Translation Result")]),
                ft.TabBarView(controls=[
                    ft.Column([self.ocr_field, ft.Row([self.ocr_button], wrap=True)], spacing=8),
                    ft.Column([ft.Text("Original text", theme_style=ft.TextThemeStyle.LABEL_MEDIUM), self.original_text,
                               self.translation_field, ft.Row([self.translate_button], wrap=True)], spacing=8),
                ], height=300),
            ], spacing=8, tight=True),
        )
        # components.sheet: a scrolling body inset above the navigation bar (owner device fix 57f1835c)
        self.dialog = bottom_sheet(sheet_frame(scroll_column([
            ft.Text(f"Box {index + 1}", theme_style=ft.TextThemeStyle.TITLE_MEDIUM, weight=ft.FontWeight.W_600),
            self.tabs,
            self.free_text,
            self.exclude,
            self.iterations,
            ft.Row([self.save_button, self.save_render_button], wrap=True, spacing=8),
            ft.Row([self.clean_button, self.delete_button], wrap=True, spacing=8),
        ], spacing=8)), key=f"manga-box-sheet-{next(_SHEETS)}")

    def values(self) -> dict:
        try:
            iterations = int(self.iterations.value)
        except (TypeError, ValueError):
            iterations = -1
        return {
            "ocr_text": str(self.ocr_field.value or ""),
            "translation": str(self.translation_field.value or ""),
            "free_text": bool(self.free_text.value),
            "exclude_from_clean": bool(self.exclude.value),
            "inpaint_iterations": iterations,
        }

    def changes(self) -> dict:
        """Only what the user changed (the editor applies these before any action)."""
        values = self.values()
        before = {
            "ocr_text": str(self.box.get("ocr_text") or ""),
            "translation": str(self.box.get("translation") or ""),
            "free_text": bool(self.box.get("free_text")),
            "exclude_from_clean": bool(self.box.get("exclude_from_clean")),
            "inpaint_iterations": -1 if self.box.get("inpaint_iterations") in (None, "") else int(
                self.box.get("inpaint_iterations")),
        }
        return {key: value for key, value in values.items() if value != before[key]}

    def act(self, name: str) -> Any:
        changes = self.changes()
        self.close()
        return self.on_action(name, self.index, changes)

    def show(self, page: Any) -> "BoxSheet":
        self._page = page
        if page is not None:
            page.show_dialog(self.dialog)
        return self

    def close(self) -> None:
        # This sheet itself: page.pop_dialog() would close a snackbar shown since it opened.
        if self._page is not None:
            close_dialog(self._page, self.dialog)
