"""The selected box's editors (UI_SPEC §4.6 Editor, §5.10).

* ``BoxPanel`` — inline, under the page: the desktop's "🌍 Translation Result" edited in place
  (a TextField per selected box), **Save & Update Overlay** (re-render the page with the edited
  text), **Translate this text**, and a ⋯ menu with Save (no re-render), **Edit OCR text…**, **OCR
  this text**, **Clean this box**, the box type (free text / bubble text: the renderer's background
  and the cleaner's dilation follow it), **Exclude from Clean**, the inpainting iterations (Auto or
  0-50) and **Delete**.
* ``OcrSheet`` — the desktop's "📝 OCR Recognition Result" popup as a compact bottom sheet (not
  full screen): the recognized text, **OCR this text** and **Save**.

Both take their texts from the editor's page snapshot (``manga_editor_core`` page state), and every
action is a ``MangaEditorSession`` call the editor tab makes (box edits on the io pool, OCR /
translate / clean / re-render as editor jobs).
"""

from __future__ import annotations

import itertools
from typing import Any, Callable, Mapping, Optional

import flet as ft

from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.components.dialogs import close_dialog
from glossarion_mobile.ui.components.sheet import bottom_sheet, fits_compact, scroll_column, sheet_frame, text_height
from glossarion_mobile.ui.theme import HIT_TARGET

__all__ = ["BoxPanel", "ITERATION_CHOICES", "NOT_RECOGNIZED", "OcrSheet"]

#: Inpainting iterations: -1 = Auto (the dialog's range is -1..50).
ITERATION_CHOICES = (-1, 0, 1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 50)
_KEYS = itertools.count(1)  # a new key per panel / sheet (never a reused id())
NOT_RECOGNIZED = "Run “OCR this text” (or Recognize) first"


def _iterations(box: Mapping[str, Any]) -> int:
    value = box.get("inpaint_iterations")
    try:
        return -1 if value in (None, "") else int(value)
    except (TypeError, ValueError):
        return -1


def box_signature(box: Mapping[str, Any], index: int) -> tuple:
    """What a panel shows of a box: a panel is rebuilt only when this changes (typing is kept)."""
    return (index, str(box.get("ocr_text") or ""), str(box.get("translation") or ""), bool(box.get("free_text")),
            bool(box.get("exclude_from_clean")), _iterations(box))


class BoxPanel:
    """``on_action(name, index, values)``: ``save`` / ``save_render`` / ``ocr`` / ``translate`` /
    ``clean`` / ``delete``; ``values`` = what changed (the edited translation, a flag), applied
    first. ``on_edit_ocr(index)`` opens the ``OcrSheet``; ``on_iterations(index)`` the iterations
    choice."""

    def __init__(self, box: Mapping[str, Any], index: int, *, on_action: Callable[[str, int, dict], Any],
                 on_edit_ocr: Optional[Callable[[int], Any]] = None,
                 on_iterations: Optional[Callable[[int], Any]] = None,
                 busy_reason: Optional[str] = None, steps_reason: Optional[str] = None) -> None:
        self.box = dict(box)
        self.index = index
        self.on_action = on_action
        self.signature = box_signature(box, index)
        n = next(_KEYS)
        ocr_text = str(box.get("ocr_text") or "")
        # As on the desktop, the OCR text is translated once the box has been recognized.
        recognized = bool(ocr_text.strip())
        reason = busy_reason or steps_reason
        self.title = ft.Text(f"Box {index + 1}", theme_style=ft.TextThemeStyle.TITLE_SMALL, weight=ft.FontWeight.W_600)
        self.ocr_preview = ft.Text(ocr_text if recognized else NOT_RECOGNIZED, max_lines=2,
                                   overflow=ft.TextOverflow.ELLIPSIS, theme_style=ft.TextThemeStyle.BODY_SMALL,
                                   color=ft.Colors.ON_SURFACE_VARIANT, key="box-original", expand=True,
                                   tooltip="Original text")
        self.edit_ocr_button = ft.IconButton(icon=ft.Icons.EDIT_NOTE, tooltip="Edit OCR text", key="box-edit-ocr",
                                             size_constraints=HIT_TARGET, disabled=busy_reason is not None,
                                             on_click=lambda e: on_edit_ocr(self.index) if on_edit_ocr else None)
        self.translation_field = ft.TextField(value=str(box.get("translation") or ""), multiline=True, min_lines=2,
                                              max_lines=5, dense=True, label="Translation", key="box-translation",
                                              hint_text="Type the translation" if recognized else NOT_RECOGNIZED,
                                              border_radius=tokens.RADII["field"])
        self.save_render_button = ft.FilledButton(content="Save & Update Overlay", icon=ft.Icons.AUTO_FIX_NORMAL,
                                                  key="box-save-render", tooltip=reason, disabled=reason is not None,
                                                  on_click=lambda e: self.act("save_render"))
        translate_reason = reason or (None if recognized else NOT_RECOGNIZED)
        self.translate_button = ft.TextButton(content="Translate this text", icon=ft.Icons.TRANSLATE,
                                              key="box-translate", tooltip=translate_reason,
                                              disabled=translate_reason is not None,
                                              on_click=lambda e: self.act("translate"))
        free_text = bool(box.get("free_text"))
        excluded = bool(box.get("exclude_from_clean"))
        iterations = _iterations(box)
        self.save_item = ft.PopupMenuItem(content="Save (no re-render)", icon=ft.Icons.SAVE,
                                          disabled=busy_reason is not None, on_click=lambda e: self.act("save"))
        self.edit_ocr_item = ft.PopupMenuItem(content="Edit OCR text…", icon=ft.Icons.EDIT_NOTE,
                                              disabled=busy_reason is not None,
                                              on_click=lambda e: on_edit_ocr(self.index) if on_edit_ocr else None)
        self.ocr_item = ft.PopupMenuItem(content="OCR this text", icon=ft.Icons.DOCUMENT_SCANNER,
                                         disabled=reason is not None, on_click=lambda e: self.act("ocr"))
        self.clean_item = ft.PopupMenuItem(content="Clean this box", icon=ft.Icons.CLEANING_SERVICES,
                                           disabled=reason is not None, on_click=lambda e: self.act("clean"))
        self.free_text_item = ft.PopupMenuItem(content="Free text (off: bubble text)", checked=free_text,
                                               disabled=busy_reason is not None,
                                               on_click=lambda e: self.set_flag("free_text", not free_text))
        self.exclude_item = ft.PopupMenuItem(content="Exclude from Clean", checked=excluded,
                                             disabled=busy_reason is not None,
                                             on_click=lambda e: self.set_flag("exclude_from_clean", not excluded))
        self.iterations_item = ft.PopupMenuItem(
            content=f"Inpainting iterations: {'Auto' if iterations == -1 else iterations}…", icon=ft.Icons.TUNE,
            disabled=busy_reason is not None,
            on_click=lambda e: on_iterations(self.index) if on_iterations else None)
        self.delete_item = ft.PopupMenuItem(content="Delete box", icon=ft.Icons.DELETE_OUTLINE,
                                            disabled=busy_reason is not None, on_click=lambda e: self.act("delete"))
        self.menu = ft.PopupMenuButton(
            icon=ft.Icons.MORE_VERT, tooltip="More box actions", key="box-more", size_constraints=HIT_TARGET,
            items=[self.save_item, self.edit_ocr_item, self.ocr_item, self.clean_item, ft.PopupMenuItem(),
                   self.free_text_item, self.exclude_item, self.iterations_item, ft.PopupMenuItem(), self.delete_item])
        self.control = ft.Container(
            content=ft.Column([
                ft.Row([self.title, self.ocr_preview, self.edit_ocr_button], spacing=8,
                       vertical_alignment=ft.CrossAxisAlignment.CENTER),
                self.translation_field,
                ft.Row([self.save_render_button, self.translate_button, self.menu], wrap=True, spacing=4,
                       vertical_alignment=ft.CrossAxisAlignment.CENTER),
            ], spacing=6, tight=True),
            padding=ft.Padding.symmetric(horizontal=10, vertical=6), bgcolor=ft.Colors.SURFACE_CONTAINER,
            border_radius=tokens.RADII["field"], key=f"me-box-panel-{n}",
        )

    def values(self) -> dict:
        return {"translation": str(self.translation_field.value or "")}

    def changes(self) -> dict:
        """Only what the user changed (the editor applies these before any action)."""
        if self.values()["translation"] != str(self.box.get("translation") or ""):
            return self.values()
        return {}

    def act(self, name: str) -> Any:
        return self.on_action(name, self.index, self.changes())

    def set_flag(self, name: str, value: Any) -> Any:
        """A box flag from the ⋯ menu (free text, exclude, iterations), saved with any typed text."""
        return self.on_action("save", self.index, {**self.changes(), name: value})


class OcrSheet:
    """The box's OCR text in a compact sheet. ``on_action(name, index, values)``: ``save`` / ``ocr``
    (the edited text is applied first)."""

    def __init__(self, box: Mapping[str, Any], index: int, *, on_action: Callable[[str, int, dict], Any],
                 busy_reason: Optional[str] = None, steps_reason: Optional[str] = None) -> None:
        self.box = dict(box)
        self.index = index
        self.on_action = on_action
        self._page: Any = None
        text = str(box.get("ocr_text") or "")
        # As on the desktop, the OCR text is edited once the box has been recognized: the editor
        # keeps a typed text only for a recognized box.
        recognized = bool(text.strip())
        self.ocr_field = ft.TextField(value=text, multiline=True, min_lines=3, max_lines=6, label="Recognized text",
                                      key="box-ocr-text", read_only=not recognized,
                                      helper=None if recognized else NOT_RECOGNIZED,
                                      border_radius=tokens.RADII["field"])
        reason = busy_reason or steps_reason
        self.ocr_button = ft.TextButton(content="OCR this text", icon=ft.Icons.DOCUMENT_SCANNER, key="box-ocr",
                                        tooltip=reason, disabled=reason is not None, on_click=lambda e: self.act("ocr"))
        self.save_button = ft.FilledTonalButton(content="Save", icon=ft.Icons.SAVE, key="box-save",
                                                disabled=busy_reason is not None or not recognized,
                                                on_click=lambda e: self.act("save"))
        self.body = scroll_column([
            ft.Text(f"Box {index + 1} · 📝 OCR Recognition Result", theme_style=ft.TextThemeStyle.TITLE_MEDIUM,
                    weight=ft.FontWeight.W_600),
            self.ocr_field,
            ft.Row([self.ocr_button, self.save_button], alignment=ft.MainAxisAlignment.END, wrap=True, spacing=8),
        ], spacing=8)
        self.dialog = bottom_sheet(sheet_frame(self.body), key=f"manga-ocr-sheet-{next(_KEYS)}")

    def estimated_height(self, width: float = 360.0) -> float:
        """Upper estimate (dp): title, the field at its 6 lines (or the text), the buttons."""
        field = max(6 * 14.0 * 2.0 * 1.5, text_height(self.ocr_field.value, width - 32)) + 40
        return 56 + field + 56

    def values(self) -> dict:
        return {"ocr_text": str(self.ocr_field.value or "")}

    def changes(self) -> dict:
        if self.values()["ocr_text"] != str(self.box.get("ocr_text") or ""):
            return self.values()
        return {}

    def act(self, name: str) -> Any:
        changes = self.changes()
        self.close()
        return self.on_action(name, self.index, changes)

    def show(self, page: Any) -> "OcrSheet":
        self._page = page
        # a short sheet stays compact: a scrolling body would open it at full height (components.sheet)
        width = 360.0
        try:
            width = float(getattr(page, "width", 0) or 360.0)
        except (TypeError, ValueError):
            pass
        self.body.scroll = None if fits_compact(page, self.estimated_height(width)) else ft.ScrollMode.AUTO
        if page is not None:
            page.show_dialog(self.dialog)
        return self

    def close(self) -> None:
        # This sheet itself: page.pop_dialog() would close a snackbar shown since it opened.
        if self._page is not None:
            close_dialog(self._page, self.dialog)
