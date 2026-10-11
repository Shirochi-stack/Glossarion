"""The selected box's editor (UI_SPEC §4.6 Editor, §5.10).

``BoxPanel`` — inline, under the page: the desktop's two box texts as two tabs, each edited in
place:

* **OCR** — the desktop's "📝 OCR Recognition Result": the recognized text, **OCR this text** and
  **Save**. As on the desktop, the text is edited once the box has been recognized.
* **Translated** — the desktop's "🌍 Translation Result": the translation, **Save & Update Overlay**
  (re-render the page with the edited text) and **Translate this text**.

The header row holds the box number, the two tabs and the ⋯ menu: Save (no re-render), **Clean
this box**, the box type (free text / bubble text: the renderer's background and the cleaner's
dilation follow it), **Exclude from Clean**, the inpainting iterations (Auto or 0-50) and
**Delete**. Compact type (body small / label small) so the page keeps the room.

The texts come from the editor's page snapshot (``manga_editor_core`` page state), and every
action is a ``MangaEditorSession`` call the editor tab makes (box edits on the io pool, OCR /
translate / clean / re-render as editor jobs).
"""

from __future__ import annotations

import itertools
from typing import Any, Callable, Mapping, Optional

import flet as ft

from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.theme import HIT_TARGET
from glossarion_mobile.ui.tools.manga.common import push

__all__ = ["BOX_TABS", "BoxPanel", "COMPACT_BUTTON", "ITERATION_CHOICES", "NOT_RECOGNIZED", "box_signature"]

#: Inpainting iterations: -1 = Auto (the dialog's range is -1..50).
ITERATION_CHOICES = (-1, 0, 1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 50)
_KEYS = itertools.count(1)  # a new key per panel (never a reused id())
NOT_RECOGNIZED = "Run “OCR this text” (or Recognize) first"
#: The panel's tabs: (id, label).
BOX_TABS = (("ocr", "OCR"), ("translated", "Translated"))

_BODY = tokens.TYPE_SCALE["body_small"]
_LABEL = tokens.TYPE_SCALE["label_small"]
BODY_TEXT = ft.TextStyle(size=_BODY.size)
LABEL_TEXT = ft.TextStyle(size=_LABEL.size, weight=ft.FontWeight.W_500)
#: Compact buttons of the editor (label small, tight padding).
COMPACT_BUTTON = ft.ButtonStyle(text_style=LABEL_TEXT, icon_size=16, visual_density=ft.VisualDensity.COMPACT,
                                padding=ft.Padding.symmetric(horizontal=8))


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


def _field(value: str, label: str, key: str, **kwargs: Any) -> ft.TextField:
    return ft.TextField(value=value, label=label, multiline=True, min_lines=2, max_lines=4, dense=True, key=key,
                        text_style=BODY_TEXT, label_style=LABEL_TEXT, hint_style=BODY_TEXT,
                        content_padding=ft.Padding.symmetric(horizontal=10, vertical=8),
                        border_radius=tokens.RADII["field"], **kwargs)


class BoxPanel:
    """``on_action(name, index, values)``: ``save`` / ``save_render`` / ``ocr`` / ``translate`` /
    ``clean`` / ``delete``; ``values`` = what changed (the edited OCR text and translation, a
    flag), applied first. ``on_iterations(index)`` opens the iterations choice. ``tab`` is the tab
    shown first (``ocr`` / ``translated``)."""

    def __init__(self, box: Mapping[str, Any], index: int, *, on_action: Callable[[str, int, dict], Any],
                 on_iterations: Optional[Callable[[int], Any]] = None,
                 busy_reason: Optional[str] = None, steps_reason: Optional[str] = None,
                 tab: str = "translated") -> None:
        self.box = dict(box)
        self.index = index
        self.on_action = on_action
        self.signature = box_signature(box, index)
        self.tab = tab if tab in dict(BOX_TABS) else "translated"
        n = next(_KEYS)
        ocr_text = str(box.get("ocr_text") or "")
        # As on the desktop, the OCR text is edited and translated once the box has been recognized.
        self.recognized = recognized = bool(ocr_text.strip())
        reason = busy_reason or steps_reason
        self.title = ft.Text(f"Box {index + 1}", theme_style=ft.TextThemeStyle.LABEL_MEDIUM, key="box-title")
        # ---- OCR tab
        self.ocr_field = _field(ocr_text, "Recognized text", "box-ocr-text", read_only=not recognized,
                                hint_text=None if recognized else NOT_RECOGNIZED)
        self.ocr_button = ft.TextButton(content="OCR this text", icon=ft.Icons.DOCUMENT_SCANNER, key="box-ocr",
                                        tooltip=reason, disabled=reason is not None, style=COMPACT_BUTTON,
                                        on_click=lambda e: self.act("ocr"))
        self.save_button = ft.TextButton(content="Save", icon=ft.Icons.SAVE, key="box-save",
                                         disabled=busy_reason is not None, style=COMPACT_BUTTON,
                                         on_click=lambda e: self.act("save"))
        # ---- Translated tab
        self.translation_field = _field(str(box.get("translation") or ""), "Translation", "box-translation",
                                        hint_text="Type the translation" if recognized else NOT_RECOGNIZED)
        self.save_render_button = ft.FilledButton(content="Save & Update Overlay", icon=ft.Icons.AUTO_FIX_NORMAL,
                                                  key="box-save-render", tooltip=reason, disabled=reason is not None,
                                                  style=COMPACT_BUTTON, on_click=lambda e: self.act("save_render"))
        translate_reason = reason or (None if recognized else NOT_RECOGNIZED)
        self.translate_button = ft.TextButton(content="Translate this text", icon=ft.Icons.TRANSLATE,
                                              key="box-translate", tooltip=translate_reason,
                                              disabled=translate_reason is not None, style=COMPACT_BUTTON,
                                              on_click=lambda e: self.act("translate"))
        # ---- ⋯ menu
        free_text = bool(box.get("free_text"))
        excluded = bool(box.get("exclude_from_clean"))
        iterations = _iterations(box)
        self.save_item = ft.PopupMenuItem(content="Save (no re-render)", icon=ft.Icons.SAVE,
                                          disabled=busy_reason is not None, on_click=lambda e: self.act("save"))
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
            items=[self.save_item, self.clean_item, ft.PopupMenuItem(), self.free_text_item, self.exclude_item,
                   self.iterations_item, ft.PopupMenuItem(), self.delete_item])
        # ---- the two tabs (a TabBar only: the bodies below are shown / hidden, so the panel keeps
        # its own height; a TabBarView needs a fixed one)
        names = [name for name, _label in BOX_TABS]
        self.tabs = ft.Tabs(
            length=len(BOX_TABS), selected_index=names.index(self.tab), on_change=self._on_tab, expand=True,
            content=ft.TabBar(tabs=[ft.Tab(label=label, height=34) for _name, label in BOX_TABS], scrollable=False,
                              label_text_style=LABEL_TEXT, unselected_label_text_style=LABEL_TEXT,
                              divider_height=0, key=f"box-tabbar-{n}"),
            key=f"box-tabs-{n}")
        self.ocr_body = ft.Column([
            self.ocr_field,
            ft.Row([self.ocr_button, self.save_button], spacing=4, scroll=ft.ScrollMode.AUTO),
        ], spacing=4, tight=True, visible=self.tab == "ocr", key=f"box-ocr-body-{n}")
        self.translation_body = ft.Column([
            self.translation_field,
            ft.Row([self.save_render_button, self.translate_button], spacing=4, scroll=ft.ScrollMode.AUTO),
        ], spacing=4, tight=True, visible=self.tab == "translated", key=f"box-translation-body-{n}")
        self.control = ft.Container(
            content=ft.Column([
                # one row: the box, its tabs and the ⋯ menu (never wrapped onto a row of its own)
                ft.Row([self.title, self.tabs, self.menu], spacing=4,
                       vertical_alignment=ft.CrossAxisAlignment.CENTER, key=f"box-header-{n}"),
                self.ocr_body,
                self.translation_body,
            ], spacing=2, tight=True),
            padding=ft.Padding.only(left=8, right=0, top=0, bottom=4), bgcolor=ft.Colors.SURFACE_CONTAINER,
            border_radius=tokens.RADII["field"], key=f"me-box-panel-{n}",
        )

    # ---- tabs ---------------------------------------------------------------------------------------

    def _on_tab(self, e: Any = None) -> None:
        index = int(getattr(self.tabs, "selected_index", 0) or 0)
        self.select_tab(BOX_TABS[max(0, min(index, len(BOX_TABS) - 1))][0])

    def select_tab(self, name: str) -> None:
        """Show the ``ocr`` or ``translated`` tab."""
        names = [n for n, _label in BOX_TABS]
        if name not in names:
            return
        self.tab = name
        self.tabs.selected_index = names.index(name)
        self.ocr_body.visible = name == "ocr"
        self.translation_body.visible = name == "translated"
        push(self.control)

    # ---- values -------------------------------------------------------------------------------------

    def values(self) -> dict:
        return {"ocr_text": str(self.ocr_field.value or ""), "translation": str(self.translation_field.value or "")}

    def changes(self) -> dict:
        """Only what the user changed (the editor applies these before any action). The OCR text
        counts once the box is recognized (its field is read-only before)."""
        values = self.values()
        changed: dict = {}
        if self.recognized and values["ocr_text"] != str(self.box.get("ocr_text") or ""):
            changed["ocr_text"] = values["ocr_text"]
        if values["translation"] != str(self.box.get("translation") or ""):
            changed["translation"] = values["translation"]
        return changed

    def keep_typing(self, other: "BoxPanel") -> None:
        """Carry what is typed in ``other`` (the same box, rebuilt for a new step state) over."""
        self.ocr_field.value = other.ocr_field.value
        self.translation_field.value = other.translation_field.value

    def act(self, name: str) -> Any:
        return self.on_action(name, self.index, self.changes())

    def set_flag(self, name: str, value: Any) -> Any:
        """A box flag from the ⋯ menu (free text, exclude, iterations), saved with any typed text."""
        return self.on_action("save", self.index, {**self.changes(), name: value})
