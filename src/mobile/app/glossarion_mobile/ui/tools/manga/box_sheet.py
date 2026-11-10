"""The selected box's editor (UI_SPEC §4.6 Editor, §5.10).

``BoxPanel`` — inline, under the page: the desktop's two box texts as two tabs, each edited in
place:

* **OCR** — the desktop's "📝 OCR Recognition Result": the recognized text, **OCR this text** and
  **Save**. As on the desktop, the text is edited once the box has been recognized.
* **Translated** — the desktop's "🌍 Translation Result": the translation, **Save & Update Overlay**
  (re-render the page with the edited text) and **Translate this text**.

* **Move** — the box's position and size in image pixels: X / Y / W / H fields with **Apply**, and
  nudge buttons (move by the step, W / H smaller / larger) for what a finger cannot place exactly.
* **Style** — the desktop Format toolbar for this box: font, size (Auto or px in 0.5 px steps, with
  − / + and Auto; a box on Auto shows the size it was last rendered at, ``rendered_font_size``),
  **B** / *I* / U, align left / center / right, text colour and Reset; each change re-renders the
  page (``MangaEditorSession.set_box_style``, then Save & Update Overlay when the page has
  translations).

The header row holds the box number, the tabs, the ⋯ menu (Save (no re-render), **Clean this
box**, the box type (free text / bubble text: the renderer's background and the cleaner's
dilation follow it), **Exclude from Clean**, the inpainting iterations (Auto or 0-50) and
**Delete**) and ✕ (done: the box is deselected). Compact type (body small / label small) so the
page keeps the room.

The texts come from the editor's page snapshot (``manga_editor_core`` page state), and every
action is a ``MangaEditorSession`` call the editor tab makes (box edits on the io pool, OCR /
translate / clean / re-render as editor jobs).
"""

from __future__ import annotations

import itertools
import os
from typing import Any, Callable, Mapping, Optional, Sequence

import flet as ft

from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.components.field_commit import commit_on_blur
from glossarion_mobile.ui.theme import HIT_TARGET
from glossarion_mobile.ui.tools.manga.common import push

__all__ = ["BOX_TABS", "BoxPanel", "COMPACT_BUTTON", "GEOMETRY_KEYS", "ITERATION_CHOICES", "NOT_RECOGNIZED",
           "NUDGE_STEPS", "SIZE_MAX", "SIZE_STEP", "STYLE_ALIGNMENTS", "STYLE_COLORS", "box_geometry",
           "box_signature", "box_size", "box_style", "size_text"]

#: Inpainting iterations: -1 = Auto (the dialog's range is -1..50).
ITERATION_CHOICES = (-1, 0, 1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 50)
_KEYS = itertools.count(1)  # a new key per panel (never a reused id())
NOT_RECOGNIZED = "Run “OCR this text” (or Recognize) first"
#: The panel's tabs: (id, label).
BOX_TABS = (("ocr", "OCR"), ("translated", "Translated"), ("move", "Move"), ("style", "Style"))
#: The Move tab's fields: (box key, label).
GEOMETRY_KEYS = (("x", "X"), ("y", "Y"), ("width", "W"), ("height", "H"))
#: The Move tab's nudge steps (image px); the step button cycles them.
NUDGE_STEPS = (1, 5, 20)
#: The Style tab's alignments: (align, icon, tooltip); center is the renderer's default.
STYLE_ALIGNMENTS = (("left", "FORMAT_ALIGN_LEFT", "Align left"), ("center", "FORMAT_ALIGN_CENTER", "Center"),
                    ("right", "FORMAT_ALIGN_RIGHT", "Align right"))
#: The Style tab's size: − / + step 0.5 px (the desktop Format toolbar's size box); at most 200 px.
SIZE_STEP = 0.5
SIZE_MAX = 200
#: The Style tab's text colours: (label, [r, g, b]; None = the Rendering Settings colour).
STYLE_COLORS = (("Default colour", None), ("Black", [0, 0, 0]), ("White", [255, 255, 255]), ("Red", [200, 0, 0]),
                ("Blue", [0, 70, 200]), ("Green", [0, 130, 60]), ("Yellow", [240, 200, 0]), ("Purple", [120, 40, 160]))

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


def box_style(box: Mapping[str, Any]) -> dict:
    """The box's text style (``manga_ocr_io.normalize_region_style`` keys; {} = Rendering Settings)."""
    style = box.get("style")
    return dict(style) if isinstance(style, Mapping) else {}


def _half_px(value: Any) -> Optional[float]:
    """A px size in 0.5 steps (28.4 -> 28.5), at most SIZE_MAX; None when it is not a positive number."""
    try:
        size = round(float(value) * 2) / 2
    except (TypeError, ValueError, OverflowError):
        return None
    return min(size, float(SIZE_MAX)) if size > 0 else None


def size_text(size: Any) -> str:
    """28 -> "28", 28.5 -> "28.5" (no trailing ".0"); "" without a size."""
    if size in (None, "") or isinstance(size, bool):
        return ""
    try:
        text = f"{float(size):.1f}"
    except (TypeError, ValueError):
        return ""
    return text[:-2] if text.endswith(".0") else text


def box_size(box: Mapping[str, Any]) -> tuple:
    """(size, auto) of a box: its own size (auto False), else the size it was last rendered at on Auto
    (``rendered_font_size``, in 0.5 px; auto True), else (None, True)."""
    own = _half_px(box_style(box).get("font_size"))
    if own is not None:
        return own, False
    return _half_px(box.get("rendered_font_size")), True


def _size_value(size: float) -> Any:
    """What a style stores: 28 (whole px) or 28.5."""
    return int(size) if float(size).is_integer() else size


def box_signature(box: Mapping[str, Any], index: int) -> tuple:
    """What a panel shows of a box: a panel is rebuilt only when this changes (typing is kept)."""
    style = tuple(sorted((key, tuple(value) if isinstance(value, list) else value)
                         for key, value in box_style(box).items()))
    return (index, str(box.get("ocr_text") or ""), str(box.get("translation") or ""), bool(box.get("free_text")),
            bool(box.get("exclude_from_clean")), _iterations(box), style, box.get("rendered_font_size"))


def box_geometry(box: Mapping[str, Any]) -> dict:
    """The box's {x, y, width, height} (image px, floats)."""
    return {key: float(box.get(key) or 0) for key, _label in GEOMETRY_KEYS}


def _number(value: float) -> str:
    return str(int(round(value)))


def _field(value: str, label: str, key: str, **kwargs: Any) -> ft.TextField:
    return ft.TextField(value=value, label=label, multiline=True, min_lines=2, max_lines=4, dense=True, key=key,
                        text_style=BODY_TEXT, label_style=LABEL_TEXT, hint_style=BODY_TEXT,
                        content_padding=ft.Padding.symmetric(horizontal=10, vertical=8),
                        border_radius=tokens.RADII["field"], **kwargs)


class BoxPanel:
    """``on_action(name, index, values)``: ``save`` / ``save_render`` / ``ocr`` / ``translate`` /
    ``clean`` / ``delete`` / ``style``; ``values`` = what changed (the edited OCR text and
    translation, a flag, ``style``: the Style tab's change, None = Reset), applied first.
    ``on_iterations(index)`` opens the iterations choice.
    ``on_geometry(index, {x, y, width, height})`` places the box (the Move tab). ``on_close()``
    deselects it (✕). ``tab`` is the tab shown first (``ocr`` / ``translated`` / ``move`` /
    ``style``). ``fonts``: the font files the Style tab offers."""

    def __init__(self, box: Mapping[str, Any], index: int, *, on_action: Callable[[str, int, dict], Any],
                 on_iterations: Optional[Callable[[int], Any]] = None,
                 on_geometry: Optional[Callable[[int, dict], Any]] = None,
                 on_close: Optional[Callable[[], Any]] = None,
                 busy_reason: Optional[str] = None, steps_reason: Optional[str] = None,
                 tab: str = "translated", step: int = NUDGE_STEPS[1], fonts: Sequence[str] = ()) -> None:
        self.box = dict(box)
        self.index = index
        self.on_action = on_action
        self.on_geometry = on_geometry
        self.step = step if step in NUDGE_STEPS else NUDGE_STEPS[1]
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
                                                  key="box-save-render", tooltip=busy_reason,
                                                  disabled=busy_reason is not None,
                                                  style=COMPACT_BUTTON, on_click=lambda e: self.act("save_render"))
        translate_reason = reason or (None if recognized else NOT_RECOGNIZED)
        self.translate_button = ft.TextButton(content="Translate this text", icon=ft.Icons.TRANSLATE,
                                              key="box-translate", tooltip=translate_reason,
                                              disabled=translate_reason is not None, style=COMPACT_BUTTON,
                                              on_click=lambda e: self.act("translate"))
        # ---- Move tab
        geometry = box_geometry(box)
        # Done or tapping away places the box (as Apply does); a field still showing the box's place
        # moves nothing.
        self.geometry_fields = {
            key: commit_on_blur(
                ft.TextField(value=_number(geometry[key]), label=label, dense=True, width=64, key=f"box-geo-{key}",
                             keyboard_type=ft.KeyboardType.NUMBER, text_style=BODY_TEXT, label_style=LABEL_TEXT,
                             content_padding=ft.Padding.symmetric(horizontal=8, vertical=8),
                             border_radius=tokens.RADII["field"]),
                lambda _text: self.apply_geometry(), current=lambda k=key: _number(box_geometry(self.box)[k]))
            for key, label in GEOMETRY_KEYS}
        self.apply_button = ft.TextButton(content="Apply", icon=ft.Icons.CHECK, key="box-geo-apply",
                                          disabled=busy_reason is not None, style=COMPACT_BUTTON,
                                          on_click=lambda e: self.apply_geometry())
        self.step_button = ft.TextButton(content=f"{self.step} px", key="box-geo-step", style=COMPACT_BUTTON,
                                         tooltip="Nudge step (tap to change)", on_click=lambda e: self.cycle_step())
        moves = (("ARROW_BACK", "Left", -1, 0), ("ARROW_UPWARD", "Up", 0, -1), ("ARROW_DOWNWARD", "Down", 0, 1),
                 ("ARROW_FORWARD", "Right", 1, 0))
        self.nudge_buttons = [
            ft.IconButton(icon=getattr(ft.Icons, icon), tooltip=f"Move {label.lower()}", key=f"box-nudge-{label.lower()}",
                          disabled=busy_reason is not None, size_constraints=HIT_TARGET,
                          on_click=lambda e, dx=dx, dy=dy: self.nudge(dx=dx, dy=dy))
            for icon, label, dx, dy in moves]
        sizes = (("W−", "Narrower", -1, 0), ("W+", "Wider", 1, 0), ("H−", "Shorter", 0, -1), ("H+", "Taller", 0, 1))
        self.size_buttons = [
            ft.TextButton(content=text, tooltip=label, key=f"box-size-{label.lower()}", style=COMPACT_BUTTON,
                          disabled=busy_reason is not None,
                          on_click=lambda e, dw=dw, dh=dh: self.nudge(dw=dw, dh=dh))
            for text, label, dw, dh in sizes]
        # ---- Style tab (the desktop Format toolbar)
        self.style = style = box_style(box)
        styling = busy_reason is not None
        font = str(style.get("font") or "")
        font_paths = [str(path) for path in fonts or ()]
        if font and font not in font_paths:
            font_paths.append(font)
        self.font_dropdown = ft.Dropdown(
            label="Font", value=font, dense=True, width=168, key="box-style-font", text_style=BODY_TEXT,
            label_style=LABEL_TEXT, disabled=styling,
            options=[ft.DropdownOption(key="", text="Default font")] + [
                ft.DropdownOption(key=path, text=os.path.splitext(os.path.basename(path))[0]) for path in font_paths],
            on_select=lambda e: self.set_style(font=e.control.value or None))
        # Show the fitted size as a value so it remains visible when the field is unfocused.
        size, auto = box_size(box)
        shown_size = size_text(size)
        self.size_field = commit_on_blur(
            ft.TextField(value=shown_size, label="Auto" if auto else "Size", dense=True,
                         width=76, key="box-style-size", keyboard_type=ft.KeyboardType.NUMBER, text_style=BODY_TEXT,
                         label_style=LABEL_TEXT, disabled=styling,
                         tooltip=(f"Auto: rendered at {shown_size} px" if size else
                                  "Update overlay to measure the fitted size") if auto else None,
                         content_padding=ft.Padding.symmetric(horizontal=8, vertical=8),
                         border_radius=tokens.RADII["field"]),
            self.set_size, current=lambda: shown_size)
        self.size_down_button = ft.IconButton(
            icon=ft.Icons.REMOVE, tooltip="Smaller (0.5 px)", key="box-style-size-down", size_constraints=HIT_TARGET,
            disabled=styling or size is None, on_click=lambda e: self.step_size(-SIZE_STEP))
        self.size_up_button = ft.IconButton(
            icon=ft.Icons.ADD, tooltip="Larger (0.5 px)", key="box-style-size-up", size_constraints=HIT_TARGET,
            disabled=styling or size is None, on_click=lambda e: self.step_size(SIZE_STEP))
        self.size_auto_button = ft.TextButton(
            content="Auto", key="box-style-size-auto", tooltip="Fit the box (the Rendering Settings size)",
            style=COMPACT_BUTTON, disabled=styling or not style.get("font_size"),
            on_click=lambda e: self.set_style(font_size=None))
        self.style_toggles = {}
        for key, icon, label in (("bold", "FORMAT_BOLD", "Bold"), ("italic", "FORMAT_ITALIC", "Italic"),
                                 ("underline", "FORMAT_UNDERLINED", "Underline")):
            on = bool(style.get(key))
            self.style_toggles[key] = ft.IconButton(
                icon=getattr(ft.Icons, icon), tooltip=label, key=f"box-style-{key}", selected=on,
                icon_color=ft.Colors.PRIMARY if on else None, size_constraints=HIT_TARGET, disabled=styling,
                on_click=lambda e, k=key: self.set_style(**{k: not self.style.get(k)}))
        align = style.get("align") or "center"
        self.align_buttons = {
            name: ft.IconButton(icon=getattr(ft.Icons, icon), tooltip=label, key=f"box-style-align-{name}",
                                selected=name == align, icon_color=ft.Colors.PRIMARY if name == align else None,
                                size_constraints=HIT_TARGET, disabled=styling,
                                on_click=lambda e, a=name: self.set_style(align=None if a == "center" else a))
            for name, icon, label in STYLE_ALIGNMENTS}
        color = style.get("color")
        self.color_menu = ft.PopupMenuButton(
            icon=ft.Icons.FORMAT_COLOR_TEXT, tooltip="Text colour", key="box-style-color", disabled=styling,
            icon_color=("#%02x%02x%02x" % tuple(color)) if color else None, size_constraints=HIT_TARGET,
            items=[ft.PopupMenuItem(content=label, checked=value == color,
                                    on_click=lambda e, v=value: self.set_style(color=v))
                   for label, value in STYLE_COLORS])
        self.reset_style_button = ft.TextButton(content="Reset", icon=ft.Icons.FORMAT_CLEAR, key="box-style-reset",
                                                tooltip="Back to the Rendering Settings", style=COMPACT_BUTTON,
                                                disabled=styling or not style, on_click=lambda e: self.reset_style())
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
                              label_padding=ft.Padding.symmetric(horizontal=6),  # the tabs on one line
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
        self.move_body = ft.Column([
            ft.Row([*self.geometry_fields.values(), self.apply_button], spacing=4, scroll=ft.ScrollMode.AUTO),
            ft.Row([*self.nudge_buttons, self.step_button, *self.size_buttons], spacing=0,
                   scroll=ft.ScrollMode.AUTO, vertical_alignment=ft.CrossAxisAlignment.CENTER),
        ], spacing=4, tight=True, visible=self.tab == "move", key=f"box-move-body-{n}")
        self.style_body = ft.Column([
            ft.Row([self.font_dropdown, self.size_down_button, self.size_field, self.size_up_button,
                    self.size_auto_button, self.reset_style_button], spacing=0,
                   scroll=ft.ScrollMode.AUTO, vertical_alignment=ft.CrossAxisAlignment.CENTER),
            ft.Row([*self.style_toggles.values(), *self.align_buttons.values(), self.color_menu], spacing=0,
                   scroll=ft.ScrollMode.AUTO, vertical_alignment=ft.CrossAxisAlignment.CENTER),
        ], spacing=4, tight=True, visible=self.tab == "style", key=f"box-style-body-{n}")
        self.close_button = ft.IconButton(icon=ft.Icons.CLOSE, tooltip="Done", key="box-close",
                                          size_constraints=HIT_TARGET,
                                          on_click=lambda e: on_close() if on_close else None)
        self.control = ft.Container(
            content=ft.Column([
                # one row: the box, its tabs, the ⋯ menu and ✕ (never wrapped onto a row of its own)
                ft.Row([self.title, self.tabs, self.menu, self.close_button], spacing=4,
                       vertical_alignment=ft.CrossAxisAlignment.CENTER, key=f"box-header-{n}"),
                self.ocr_body,
                self.translation_body,
                self.move_body,
                self.style_body,
            ], spacing=2, tight=True),
            padding=ft.Padding.only(left=8, right=0, top=0, bottom=4), bgcolor=ft.Colors.SURFACE_CONTAINER,
            border_radius=tokens.RADII["field"], border=ft.Border.all(1, ft.Colors.OUTLINE_VARIANT),
            shadow=ft.BoxShadow(blur_radius=8, color=ft.Colors.with_opacity(0.25, ft.Colors.SHADOW)),
            key=f"me-box-panel-{n}",
        )

    # ---- tabs ---------------------------------------------------------------------------------------

    def _on_tab(self, e: Any = None) -> None:
        index = int(getattr(self.tabs, "selected_index", 0) or 0)
        self.select_tab(BOX_TABS[max(0, min(index, len(BOX_TABS) - 1))][0])

    def select_tab(self, name: str) -> None:
        """Show the ``ocr`` / ``translated`` / ``move`` / ``style`` tab."""
        names = [n for n, _label in BOX_TABS]
        if name not in names:
            return
        self.tab = name
        self.tabs.selected_index = names.index(name)
        self.ocr_body.visible = name == "ocr"
        self.translation_body.visible = name == "translated"
        self.move_body.visible = name == "move"
        self.style_body.visible = name == "style"
        push(self.control)

    # ---- Move ---------------------------------------------------------------------------------------

    def set_geometry(self, box: Mapping[str, Any]) -> bool:
        """The box was moved or resized (a drag, a nudge): its fields show the new place. Fields
        are left alone while the geometry is unchanged (what is typed there stays)."""
        geometry = box_geometry(box)
        if geometry == box_geometry(self.box):
            return False
        self.box.update(geometry)
        for key, field in self.geometry_fields.items():
            field.value = _number(geometry[key])
        push(self.move_body)
        return True

    def typed_geometry(self) -> Optional[dict]:
        """The Move fields as {x, y, width, height}; None when one is not a number."""
        values: dict = {}
        for key, field in self.geometry_fields.items():
            try:
                values[key] = float(str(field.value or "").strip())
            except ValueError:
                return None
        return values

    def apply_geometry(self) -> Any:
        values = self.typed_geometry()
        if values is None or self.on_geometry is None:
            return None
        return self.on_geometry(self.index, values)

    def nudge(self, *, dx: int = 0, dy: int = 0, dw: int = 0, dh: int = 0) -> Any:
        """Move by / grow by one step (image px) from the box's current place."""
        if self.on_geometry is None:
            return None
        g = box_geometry(self.box)
        s = self.step
        return self.on_geometry(self.index, {"x": g["x"] + dx * s, "y": g["y"] + dy * s,
                                             "width": g["width"] + dw * s, "height": g["height"] + dh * s})

    def cycle_step(self) -> int:
        self.step = NUDGE_STEPS[(NUDGE_STEPS.index(self.step) + 1) % len(NUDGE_STEPS)]
        self.step_button.content = f"{self.step} px"
        push(self.step_button)
        return self.step

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
        for key, field in self.geometry_fields.items():
            field.value = other.geometry_fields[key].value
        self.step = other.step
        self.step_button.content = f"{self.step} px"

    def act(self, name: str) -> Any:
        return self.on_action(name, self.index, self.changes())

    # ---- Style --------------------------------------------------------------------------------------

    def set_style(self, **changes: Any) -> Any:
        """A Style tab change (None / False: back to the Rendering Settings for that entry), saved with
        any typed text; the editor stores it and re-renders the page."""
        return self.on_action("style", self.index, {**self.changes(), "style": changes})

    def set_size(self, text: str) -> Any:
        """The size field: a px value (0.5 px steps: 28.4 -> 28.5), or empty / 0 for Auto."""
        text = str(text or "").strip().replace(",", ".")
        try:
            float(text or 0)
        except ValueError:
            self.size_field.error_text = "A number of px, or empty for Auto"
            push(self.size_field)
            return None
        self.size_field.error_text = None
        size = _half_px(text) if text else None
        return self.set_style(font_size=_size_value(size) if size else None)

    def step_size(self, delta: float) -> Any:
        """− / +: the shown size (the box's own, or the one it was rendered at on Auto) ± 0.5 px."""
        size, _auto = box_size({**self.box, "style": self.style})
        if size is None:
            return None
        new = min(float(SIZE_MAX), max(SIZE_STEP, size + delta))
        return self.set_style(font_size=_size_value(new))

    def reset_style(self) -> Any:
        return self.on_action("style", self.index, {**self.changes(), "style": None})

    def set_flag(self, name: str, value: Any) -> Any:
        """A box flag from the ⋯ menu (free text, exclude, iterations), saved with any typed text."""
        return self.on_action("save", self.index, {**self.changes(), name: value})
