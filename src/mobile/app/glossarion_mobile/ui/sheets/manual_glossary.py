"""ManualGlossarySheet (UI_SPEC §2.14 "Force Manual Glossary", §5.4): "Provide Manual Glossary".

Desktop ``_request_direct_text_manual_glossary``: a monospace box to paste glossary
contents, **Browse…** for a CSV / JSON / TXT / Markdown file, **Use glossary**. It is
asked before every send while the chat's glossary policy is Force Manual Glossary
and is prefilled with the last glossary of this session. A loaded file left
unedited is used by path; edited or pasted contents are written into the run's temp
folder with a sniffed extension (``direct_text_rules.manual_glossary_source``).
"""

from __future__ import annotations

import os
from typing import Any, Callable, Optional

import flet as ft

from glossarion_mobile.ui.chat.direct_text_rules import (
    MANUAL_GLOSSARY_EXTENSIONS,
    ManualGlossarySource,
    manual_glossary_source,
)
from glossarion_mobile.ui.components._handlers import call_handler
from glossarion_mobile.ui.components.dialogs import close_dialog
from glossarion_mobile.ui.components.sheet import scroll_column, sheet_frame

__all__ = ["ManualGlossarySheet", "read_glossary_file"]

TITLE = "Manual glossary required"
INSTRUCTIONS = (
    "Browse for a CSV, JSON, TXT, or Markdown glossary, or paste the glossary contents. "
    "This glossary applies only to the translation you are about to send."
)
NO_FILE = "No glossary file selected — pasted contents will be used"
EDITED = "Using edited/pasted glossary contents for this translation"
REQUIRED = "Drop a glossary file or paste glossary contents before continuing."


def read_glossary_file(path: str) -> Optional[str]:
    """Desktop ``load_path``: utf-8-sig, falling back to replacement decoding; None if unsupported."""
    path = os.path.abspath(os.path.expanduser(str(path or "")))
    if not (path and os.path.isfile(path) and os.path.splitext(path)[1].lower() in MANUAL_GLOSSARY_EXTENSIONS):
        return None
    try:
        with open(path, "r", encoding="utf-8-sig") as handle:
            return handle.read()
    except UnicodeDecodeError:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            return handle.read()
    except OSError:
        return None


class ManualGlossarySheet:
    def __init__(
        self,
        *,
        on_use: Optional[Callable[[ManualGlossarySource], Any]] = None,
        on_cancel: Optional[Callable[[], Any]] = None,
        pick_file: Optional[Callable[[], Any]] = None,  # async -> path | None
        initial: Optional[ManualGlossarySource] = None,
        mono: str = "monospace",
    ) -> None:
        self.on_use = on_use
        self.on_cancel = on_cancel
        self.pick_file = pick_file
        self.result: Optional[ManualGlossarySource] = None
        self.source_path = ""
        self.source_text: Optional[str] = None
        self.source_extension = ""
        self._page: Any = None
        self._loading = False
        self.editor = ft.TextField(
            multiline=True,
            min_lines=8,
            max_lines=14,
            hint_text="Paste glossary contents…",
            text_style=ft.TextStyle(font_family=mono, size=13),
            on_change=self._changed,
        )
        self.source_label = ft.Text(NO_FILE, theme_style=ft.TextThemeStyle.BODY_SMALL,
                                    color=ft.Colors.ON_SURFACE_VARIANT)
        self.error_label = ft.Text("", color=ft.Colors.ERROR, visible=False, theme_style=ft.TextThemeStyle.BODY_SMALL)
        self.use_button = ft.FilledButton(content="Use glossary", on_click=lambda e: self.use())
        if initial is not None:
            if initial.kind == "path":
                text = read_glossary_file(initial.path)
                if text is not None:
                    self.load_text(initial.path, text)
            else:
                self.editor.value = initial.content
        # The text scrolls and the buttons stay pinned below it: with the keyboard up the sheet is
        # shorter than the editor, and only a scroll view brings the focused editor into view.
        self.dialog = ft.BottomSheet(
            content=sheet_frame(
                padding=ft.Padding.only(left=16, right=16, bottom=24),
                content=scroll_column(
                    [
                        ft.Text(TITLE, theme_style=ft.TextThemeStyle.TITLE_LARGE),
                        ft.Text(INSTRUCTIONS, theme_style=ft.TextThemeStyle.BODY_MEDIUM),
                        self.editor,
                        self.source_label,
                        self.error_label,
                    ],
                    footer=[
                        ft.Row(
                            [
                                ft.TextButton(content="Browse…", on_click=self._browse),
                                ft.Container(expand=True),
                                ft.TextButton(content="Cancel", on_click=lambda e: self.cancel()),
                                self.use_button,
                            ],
                            spacing=8,
                        ),
                    ],
                    spacing=10,
                ),
            ),
            show_drag_handle=True,
            scrollable=True,
            bgcolor=ft.Colors.SURFACE_CONTAINER_HIGH,
        )

    # ---- state ----------------------------------------------------------------------------

    def load_text(self, path: str, text: str) -> None:
        self._loading = True
        try:
            self.editor.value = text
            self.source_path = os.path.abspath(path)
            self.source_extension = os.path.splitext(path)[1].lower()
            self.source_text = text
            self.source_label.value = f"Loaded file: {self.source_path}"
        finally:
            self._loading = False

    def load_path(self, path: str) -> bool:
        text = read_glossary_file(path)
        if text is None:
            self._error("Select a readable CSV, JSON, TXT, or Markdown glossary file.")
            return False
        self.load_text(path, text)
        self._push()
        return True

    def _changed(self, e: Any = None) -> None:
        if not self._loading and self.source_path and (self.editor.value or "") != self.source_text:
            self.source_path = ""
            self.source_label.value = EDITED
            self._push()

    def source(self) -> Optional[ManualGlossarySource]:
        return manual_glossary_source(
            self.editor.value or "",
            source_path=self.source_path,
            source_text=self.source_text,
            source_extension=self.source_extension,
        )

    def use(self) -> Optional[ManualGlossarySource]:
        if self.result is not None:  # a second tap while the sheet closes never sends twice
            return None
        source = self.source()
        if source is None:
            self._error(REQUIRED)
            return None
        self.result = source
        self.close()
        call_handler(self.on_use, source)
        return source

    def cancel(self) -> None:
        self.close()
        call_handler(self.on_cancel)

    def _error(self, message: str) -> None:
        self.error_label.value = message
        self.error_label.visible = True
        self._push()

    async def _browse(self, e: Any = None) -> None:
        if self.pick_file is None:
            return
        path = await self.pick_file()
        if path:
            self.load_path(path)

    # ---- presentation -------------------------------------------------------------------

    def _push(self) -> None:
        try:
            self.dialog.update()
        except Exception:
            pass

    def show(self, page: Any) -> None:
        self._page = page
        page.show_dialog(self.dialog)

    def close(self) -> None:
        close_dialog(self._page, self.dialog)
