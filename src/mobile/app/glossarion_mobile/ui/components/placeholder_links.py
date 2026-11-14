"""Placeholder links: AI prompt placeholders marked like links, with what each says right now (UI_SPEC §5.5).

The mobile form of the desktop's blue placeholders with live hover tooltips (owner 2026-11-14: "can you make
it so mobile marks placeholders?"). A phone has no hover and Flet's ``TextField`` cannot colour part of its
own text, so one prompt editor gets three things from a ``PlaceholderLinker``:

* **Chips** (``chip`` / ``chips_row``): the placeholder chips are link-styled (link blue, underlined). A tap
  inserts at the caret as before; a long press opens the value sheet. ``hint()`` is the one caption line
  under the first chip row of an editor (``HINT_TEXT``).
* **Marked view** (``MarkedView``): a toggle (``SHOW_MARKED`` / ``BACK_TO_EDITING``) that swaps the editable
  field for a read-only, selectable, scrolling rich text of the same prompt in the same font. Every
  placeholder the prompt's request really replaces is blue and underlined and opens the value sheet when
  tapped; other ``{words}`` stay plain. The field itself is never touched: its value, the dirty state and
  the token count stay as they were, and unsaved edits are what the view shows.
* **Value sheet** (``PlaceholderValueSheet``): a bottom sheet titled with the placeholder that opens at
  once and works out the value off the UI loop: the exact expansion (verbatim, selectable, scrolling,
  Copy when the context can copy), or the inactive / per-request line. The lines are the shared
  ``prompt_defaults.placeholder_value_parts`` rules the desktop tooltip shows.

Which placeholders a prompt marks and where the values come from: ``services.prompt_placeholders``.
``marked_segments`` is the pure span rule (host-tested).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Iterable, Optional

import flet as ft

from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.components._handlers import call_handler
from glossarion_mobile.ui.components.dialogs import close_dialog
from glossarion_mobile.ui.components.sheet import fitted_column, sheet_frame
from glossarion_mobile.ui.theme import HIT_TARGET, is_dark, semantic

__all__ = [
    "BACK_TO_EDITING",
    "HINT_TEXT",
    "LOADING_TEXT",
    "MarkedView",
    "PlaceholderLinker",
    "PlaceholderValueSheet",
    "SHOW_MARKED",
    "link_color",
    "marked_segments",
    "value_lines",
]

log = logging.getLogger("glossarion.prompts")

#: The caption under the first placeholder chip row of an editor.
HINT_TEXT = "Tap to insert · hold to see what it says now"
#: The marked-view toggle's tooltip / semantics label in each state.
SHOW_MARKED = "Show placeholders marked (tap one to see what it says now)"
BACK_TO_EDITING = "Back to editing"
#: The value sheet's line while the value is worked out.
LOADING_TEXT = "Working out what it says now…"

_SAME_AS_LINKER = object()  # MarkedView font: the linker's mono font


def link_color(dark: bool) -> str:
    """The link blue (``tokens.SEMANTIC_COLORS['link']``: the desktop's light / dark link colours)."""
    return semantic("link", bool(dark))


def link_style(dark: bool, **kwargs: Any) -> ft.TextStyle:
    return ft.TextStyle(color=link_color(dark), decoration=ft.TextDecoration.UNDERLINE,
                        decoration_color=link_color(dark), **kwargs)


def marked_segments(text: str, names: Iterable[str]) -> list[tuple[str, Optional[str]]]:
    """``text`` cut into ``(segment, name)`` parts: ``name`` is set for each ``{name}`` the prompt's request
    replaces (``names``), None for the plain text between them. The segments join back to ``text`` exactly."""
    from prompt_defaults import placeholder_spans  # the desktop's rule (shared, GUI-free)

    text = str(text or "")
    out: list[tuple[str, Optional[str]]] = []
    at = 0
    for start, end, name in placeholder_spans(text, frozenset(names or ())):
        if start > at:
            out.append((text[at:start], None))
        out.append((text[start:end], name))
        at = end
    if at < len(text) or not out:
        out.append((text[at:], None))
    return out


def value_lines(value: Any) -> tuple[str, list[str]]:
    """``(text, notes)`` the value sheet shows: the expansion verbatim and the short muted lines
    (``prompt_defaults.placeholder_value_parts``, the desktop tooltip's rules)."""
    from prompt_defaults import placeholder_value_parts

    return placeholder_value_parts(value)


def _push(*controls: Any) -> None:
    for control in controls:
        try:
            control.update()
        except Exception:  # not mounted yet / the sheet closed
            pass


class PlaceholderValueSheet:
    """What one placeholder says right now, worked out when the sheet opens (``links.value`` off the loop)."""

    def __init__(self, name: str, links: Any, prompt_text: str, *, run_io: Optional[Callable[..., Any]] = None,
                 copy: Optional[Callable[[str], Any]] = None, mono: str = "monospace", key: str = "ph-sheet") -> None:
        self.name = str(name).strip("{}")
        self.links = links
        self.prompt_text = str(prompt_text or "")
        self.run_io = run_io
        self.copy = copy
        self.key = key
        self.value: Any = None
        self.text = ""
        self.notes: list[str] = []
        self.loaded = False
        self._page: Any = None
        self._task: Any = None
        self.title = ft.Text("{" + self.name + "}", theme_style=ft.TextThemeStyle.TITLE_LARGE,
                             weight=ft.FontWeight.W_600, font_family=mono, selectable=True, key=f"{key}-title")
        self.status = ft.Text(LOADING_TEXT, theme_style=ft.TextThemeStyle.BODY_SMALL,
                              color=ft.Colors.ON_SURFACE_VARIANT, key=f"{key}-status")
        self.value_text = ft.Text("", selectable=True, font_family=mono, size=tokens.MONO_STYLE.size,
                                  color=ft.Colors.ON_SURFACE, key=f"{key}-value")
        self.value_box = ft.Container(content=self.value_text, visible=False, padding=10,
                                      border_radius=tokens.RADII["field"], bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST)
        self.copy_button = ft.TextButton(content="Copy", icon=ft.Icons.CONTENT_COPY, visible=False,
                                         on_click=lambda e: self.copy_value(), key=f"{key}-copy")
        self.body = fitted_column([self.title, self.status, self.value_box, ft.Row([self.copy_button])],
                                  spacing=10, estimate=180)
        self.dialog = ft.BottomSheet(
            content=sheet_frame(self.body, padding=ft.Padding.only(left=16, right=16, bottom=24)),
            show_drag_handle=True, scrollable=True, draggable=True, bgcolor=ft.Colors.SURFACE_CONTAINER_HIGH,
            key=key,
        )

    def resolve(self) -> Any:
        """Blocking: what the placeholder says now (a failure becomes an ``error`` value, never a guess)."""
        try:
            return self.links.value(self.name, self.prompt_text)
        except Exception as exc:
            log.warning("placeholder value of {%s} failed", self.name, exc_info=True)
            return {"error": str(exc) or type(exc).__name__}

    def render(self, value: Any) -> None:
        self.value = value
        self.text, self.notes = value_lines(value)
        self.loaded = True
        self.value_text.value = self.text
        self.value_box.visible = bool(self.text)
        self.status.value = "\n".join(self.notes)
        self.status.visible = bool(self.notes)
        self.copy_button.visible = bool(self.text) and self.copy is not None
        _push(self.status, self.value_box, self.copy_button, self.body)

    async def load(self) -> Any:
        if self.run_io is not None:
            value = await self.run_io(self.resolve)
        else:
            value = self.resolve()
        self.render(value)
        return value

    def show(self, page: Any) -> "PlaceholderValueSheet":
        self._page = page
        if page is not None:
            page.show_dialog(self.dialog)
        try:
            self._task = asyncio.ensure_future(self.load())
        except RuntimeError:  # no running loop (a plain call in a test): work it out now
            self.render(self.resolve())
        return self

    def copy_value(self) -> None:
        if self.text and self.copy is not None:
            call_handler(self.copy, self.text)

    def close(self) -> None:
        close_dialog(self._page, self.dialog)


class PlaceholderLinker:
    """The placeholder links of one prompt editor: link-styled chips, the marked view and the value sheet.

    ``links``: a ``services.prompt_placeholders.PlaceholderLinks``. ``text()``: the prompt as the editor
    shows it now (unsaved edits included). ``page``: the page or a callable giving it (the sheet opens on
    it; its theme picks the link blue unless ``dark`` is given). ``run_io``: the editor's off-loop runner."""

    def __init__(self, links: Any, *, text: Callable[[], str], page: Any = None,
                 run_io: Optional[Callable[..., Any]] = None, copy: Optional[Callable[[str], Any]] = None,
                 mono: str = "monospace", dark: Optional[bool] = None, key: str = "ph") -> None:
        self.links = links
        self.text = text
        self._page = page
        self.run_io = run_io
        self.copy = copy
        self.mono = mono
        self._dark = dark
        self.key = key
        self._names: Optional[frozenset] = None
        self.chip_labels: list[ft.Text] = []
        self.sheet: Optional[PlaceholderValueSheet] = None
        self.views: list["MarkedView"] = []

    # ---- context -------------------------------------------------------------------------------------

    @property
    def page(self) -> Any:
        page = self._page() if callable(self._page) else self._page
        return page

    @property
    def dark(self) -> bool:
        if self._dark is not None:
            return bool(self._dark)
        page = self.page
        return bool(page is not None and is_dark(page))

    async def names(self) -> frozenset:
        """The placeholders this prompt's request replaces (worked out once, off the loop)."""
        if self._names is None:
            try:
                if self.run_io is not None:
                    self._names = await self.run_io(self.links.names_now)
                else:
                    self._names = self.links.names_now()
            except Exception:
                log.warning("placeholder names unavailable", exc_info=True)
                self._names = frozenset()
        return self._names

    # ---- chips ---------------------------------------------------------------------------------------

    def chip(self, placeholder: str, on_insert: Callable[[str], Any], *, key: Optional[str] = None,
             size: float = 12, font_family: Optional[str] = None) -> tuple[ft.Chip, ft.Control]:
        """``(chip, control)``: the link-styled chip (tap: ``on_insert(placeholder)``) and the control to place
        (the chip inside a long-press detector that opens the value sheet)."""
        name = str(placeholder).strip("{}")
        label = ft.Text(placeholder, size=size, font_family=font_family, style=link_style(self.dark))
        self.chip_labels.append(label)
        # No chip tooltip: on touch a Tooltip opens on long press and would take the hold from the detector
        # (the hint line under the row says what tap and hold do).
        chip = ft.Chip(label=label, on_click=lambda e, p=placeholder: on_insert(p), key=key)
        holder = ft.GestureDetector(content=chip, on_long_press=lambda e, n=name: self.open(n),
                                    key=f"{key}-hold" if key else None)
        return chip, holder

    def chips_row(self, placeholders: Iterable[str], on_insert: Callable[[str], Any], *,
                  key_of: Optional[Callable[[str], Optional[str]]] = None, size: float = 12,
                  font_family: Optional[str] = None, hint: bool = True,
                  spacing: float = 6) -> tuple[list[ft.Chip], list[ft.Control]]:
        """``(chips, controls)``: the chips and the rows to place (the wrapping chip row, then the hint line)."""
        chips: list[ft.Chip] = []
        holders: list[ft.Control] = []
        for placeholder in placeholders:
            chip, holder = self.chip(placeholder, on_insert, key=key_of(placeholder) if key_of else None, size=size,
                                     font_family=font_family)
            chips.append(chip)
            holders.append(holder)
        if not chips:
            return chips, []
        controls: list[ft.Control] = [ft.Row(holders, wrap=True, spacing=spacing, run_spacing=4)]
        if hint:
            controls.append(self.hint())
        return chips, controls

    def hint(self) -> ft.Text:
        return ft.Text(HINT_TEXT, theme_style=ft.TextThemeStyle.LABEL_SMALL, color=ft.Colors.ON_SURFACE_VARIANT,
                       key=f"{self.key}-hint")

    def refresh_colors(self) -> None:
        """The link blue of the current theme (the page is known once the editor is mounted)."""
        for label in self.chip_labels:
            label.style = link_style(self.dark)
        _push(*self.chip_labels)
        for view in self.views:
            if view.marked:
                view.render_spans()

    # ---- the value sheet -----------------------------------------------------------------------------

    def open(self, name: str) -> PlaceholderValueSheet:
        """The value sheet of ``{name}``, worked out from the prompt as it is now."""
        try:
            text = self.text() or ""
        except Exception:
            text = ""
        self.sheet = PlaceholderValueSheet(name, self.links, text, run_io=self.run_io, copy=self.copy,
                                           mono=self.mono, key=f"{self.key}-sheet")
        self.sheet.show(self.page)
        return self.sheet

    # ---- the marked view -----------------------------------------------------------------------------

    def marked_view(self, field: ft.TextField, holder: ft.Control, *, size: float = 13,
                    key: Optional[str] = None, height: Optional[float] = None, mono: bool = True,
                    label: Optional[str] = None, subject: Optional[str] = None) -> "MarkedView":
        """``height``: the view's height where the field does not expand (a field of fixed lines). ``mono``:
        False where the field uses the default font. ``label``: the field's label, kept above the marked text.
        ``subject``: names the prompt in the toggle's label where a screen has more than one marked view."""
        view = MarkedView(self, field, holder, size=size, key=key or f"{self.key}-marked", height=height,
                          font_family=self.mono if mono else None, label=label, subject=subject)
        self.views.append(view)
        return view


class MarkedView:
    """The prompt with its placeholders marked, in place of the editable field (``holder`` hosts the field).

    ``toggle`` switches between the two; ``body`` (the field holder and the view, one visible) replaces the
    holder in the editor. Switching never touches the field."""

    def __init__(self, linker: PlaceholderLinker, field: ft.TextField, holder: ft.Control, *, size: float = 13,
                 key: str = "ph-marked", height: Optional[float] = None, font_family: Any = _SAME_AS_LINKER,
                 label: Optional[str] = None, subject: Optional[str] = None) -> None:
        self.linker = linker
        self.field = field
        self.holder = holder
        self.key = key
        self.marked = False
        self.shown_text = ""
        self.names: frozenset = frozenset()
        self._busy = False
        self.show_label = SHOW_MARKED if not subject else f"Show {subject} placeholders marked (tap one to see what it says now)"
        self.back_label = BACK_TO_EDITING if not subject else f"Back to editing the {subject}"
        font = linker.mono if font_family is _SAME_AS_LINKER else font_family
        self.text = ft.Text(spans=[], selectable=True, font_family=font, size=size, color=ft.Colors.ON_SURFACE,
                            key=f"{key}-text")
        content: list[ft.Control] = []
        if label:  # the field's own label, which the field shows and the marked text would otherwise lose
            content.append(ft.Text(label, theme_style=ft.TextThemeStyle.LABEL_SMALL, color=ft.Colors.ON_SURFACE_VARIANT,
                                   key=f"{key}-label"))
        content.append(self.text)
        grows = bool(getattr(holder, "expand", None))
        self.view = ft.Container(
            content=ft.Column(content, scroll=ft.ScrollMode.AUTO, expand=True, spacing=4),
            expand=grows or None, height=None if grows else (height or 240), visible=False,
            padding=ft.Padding.symmetric(horizontal=12, vertical=10),
            border=ft.Border.all(1, ft.Colors.OUTLINE_VARIANT), border_radius=tokens.RADII["field"], key=key,
        )
        self.body = ft.Column([holder, self.view], spacing=0, expand=grows or None, tight=not grows)
        self.toggle = ft.IconButton(icon=ft.Icons.DATA_OBJECT, tooltip=self.show_label, size_constraints=HIT_TARGET,
                                    on_click=lambda e: self._on_toggle(), key=f"{key}-toggle")

    def _on_toggle(self) -> Any:
        coro = self.set_marked(not self.marked)
        try:
            return asyncio.ensure_future(coro)
        except RuntimeError:  # no running loop
            coro.close()
            return None

    def segments(self) -> list[tuple[str, Optional[str]]]:
        return marked_segments(self.shown_text, self.names)

    def render_spans(self) -> None:
        dark = self.linker.dark
        spans: list[ft.TextSpan] = []
        for segment, name in self.segments():
            if name is None:
                spans.append(ft.TextSpan(text=segment))
            else:
                spans.append(ft.TextSpan(text=segment, style=link_style(dark),
                                         on_click=lambda e, n=name: self.linker.open(n),
                                         semantics_label=f"{segment}, tap to see what it says now"))
        self.text.spans = spans
        _push(self.text)

    async def set_marked(self, on: bool) -> None:
        """Show the marked view (``True``) of the field's current text, or the field again (``False``)."""
        if self._busy:
            return
        if on:
            self._busy = True
            try:
                self.names = await self.linker.names()
            finally:
                self._busy = False
            self.shown_text = self.field.value or ""
            self.render_spans()
        self.set_marked_sync(on)

    def set_marked_sync(self, on: bool) -> None:
        """Swap the view and the field (back to editing needs nothing worked out, so it is immediate)."""
        self.marked = bool(on)
        self.view.visible = self.marked
        self.holder.visible = not self.marked
        self.toggle.icon = ft.Icons.EDIT_NOTE if self.marked else ft.Icons.DATA_OBJECT
        self.toggle.tooltip = self.back_label if self.marked else self.show_label
        _push(self.body, self.toggle)
