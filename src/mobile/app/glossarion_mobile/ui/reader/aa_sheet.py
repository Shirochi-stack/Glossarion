"""AaSheet (UI_SPEC §3.11 "Aa sheet", §5.8): reader typography, theme and layout.

``BottomSheet`` (<= 75% height) with a transparent barrier so the page stays
undimmed and previews every change live. Scope switch **This book · All
books** at the top, then tabs **Text · Theme · Layout**:

* Text: font family (Embedded CSS / Serif / Sans / Mono), size slider 8-32 pt
  with A− / A+ and a value pill, line spacing 1.0-3.0, page margins;
* Theme: the six ``READER_THEMES`` swatches (exact colours) + "Follow app theme";
* Layout: Single page / Scroll / Scroll all (Double page only on a tablet in
  landscape), tap-zone paging, keep screen on, show progress %.

Every change calls ``on_change({name: value})``; the Reader applies it with
``GLRDR.applyStyle`` (no reload) or re-renders for layout and family changes,
and saves it for the chosen scope (``model.config_updates`` for All books,
Prefs ``reader_book_settings`` for This book).
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Optional, Sequence

import flet as ft

from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.components._handlers import call_handler
from glossarion_mobile.ui.components.dialogs import close_dialog
from glossarion_mobile.ui.reader import model as rm
from glossarion_mobile.ui.theme import HIT_TARGET

__all__ = ["AaSheet", "SCOPE_ALL", "SCOPE_BOOK"]

SCOPE_BOOK = "book"
SCOPE_ALL = "all"
SCOPE_LABELS = {SCOPE_BOOK: "This book", SCOPE_ALL: "All books"}


class AaSheet:
    def __init__(
        self,
        settings: rm.ReaderSettings,
        themes: Sequence[Mapping[str, Any]],
        *,
        scope: str = SCOPE_ALL,
        double_allowed: bool = False,
        on_change: Optional[Callable[[dict, str], Any]] = None,
        on_scope: Optional[Callable[[str], Any]] = None,
        height: Optional[float] = None,
    ) -> None:
        self.settings = settings
        self.themes = [dict(t) for t in themes] or [{"name": "Dark", "bg": "#1e1e1e", "fg": "#d4d4d4"}]
        self.scope = scope if scope in SCOPE_LABELS else SCOPE_ALL
        self.double_allowed = double_allowed
        self.on_change = on_change
        self.on_scope = on_scope
        self._page: Any = None
        self.changes: list[dict] = []  # every change emitted (tests, diagnostics)

        self.scope_buttons = ft.SegmentedButton(
            segments=[ft.Segment(value=k, label=ft.Text(v)) for k, v in SCOPE_LABELS.items()],
            selected=[self.scope],
            show_selected_icon=False,
            on_change=self._on_scope,
            key="aa-scope",
        )
        self.tabs = ft.Tabs(
            length=3,
            selected_index=0,
            content=ft.Column(
                [
                    ft.TabBar(tabs=[ft.Tab(label="Text"), ft.Tab(label="Theme"), ft.Tab(label="Layout")]),
                    ft.TabBarView(controls=[self._text_tab(), self._theme_tab(), self._layout_tab()], expand=True),
                ],
                expand=True,
                spacing=0,
            ),
            expand=True,
            key="aa-tabs",
        )
        body = ft.Container(
            padding=ft.Padding.only(left=16, right=16, bottom=8),
            content=ft.Column([ft.Row([self.scope_buttons], alignment=ft.MainAxisAlignment.CENTER), self.tabs],
                              spacing=8, expand=True),
        )
        self.sheet = ft.BottomSheet(
            content=ft.Container(content=body, height=min(520.0, (height or 800) * 0.75)),
            show_drag_handle=True,
            barrier_color=ft.Colors.TRANSPARENT,
            bgcolor=ft.Colors.SURFACE_CONTAINER_HIGH,
            dismissible=True,
        )

    # ---- tabs ----------------------------------------------------------------------------

    def _text_tab(self) -> ft.Control:
        s = self.settings
        # A desktop config may name a system font (the desktop combo lists them): keep it selectable.
        families = list(rm.FONT_FAMILIES) + ([s.font_family] if s.font_family not in rm.FONT_FAMILIES else [])
        self.family = ft.Dropdown(
            label="Font",
            value=s.font_family,
            options=[ft.DropdownOption(key=f, text=f) for f in families],
            on_select=lambda e: self._emit({"font_family": e.control.value}),
            key="aa-family",
        )
        self.size_pill = ft.Container(
            content=ft.Text(f"{s.font_size}pt", theme_style=ft.TextThemeStyle.LABEL_LARGE),
            padding=ft.Padding.symmetric(horizontal=10, vertical=4),
            border_radius=tokens.RADII["chip"],
            bgcolor=ft.Colors.SECONDARY_CONTAINER,
            key="aa-size-pill",
        )
        self.size_slider = ft.Slider(min=rm.FONT_SIZE_RANGE[0], max=rm.FONT_SIZE_RANGE[1],
                                     divisions=rm.FONT_SIZE_RANGE[1] - rm.FONT_SIZE_RANGE[0], value=s.font_size,
                                     expand=True, on_change=self._on_size_drag,
                                     on_change_end=lambda e: self._set_size(e.control.value), key="aa-size")
        smaller = ft.IconButton(icon=ft.Icons.TEXT_DECREASE, tooltip="A−", size_constraints=HIT_TARGET,
                                on_click=lambda e: self._set_size(self.settings.font_size - 1), key="aa-smaller")
        larger = ft.IconButton(icon=ft.Icons.TEXT_INCREASE, tooltip="A+", size_constraints=HIT_TARGET,
                               on_click=lambda e: self._set_size(self.settings.font_size + 1), key="aa-larger")
        self.spacing_label = ft.Text(f"Line spacing {s.line_spacing:.1f}", theme_style=ft.TextThemeStyle.LABEL_MEDIUM)
        self.spacing_slider = ft.Slider(min=rm.LINE_SPACING_RANGE[0], max=rm.LINE_SPACING_RANGE[1], divisions=20,
                                        value=s.line_spacing, on_change=self._on_spacing_drag,
                                        on_change_end=lambda e: self._emit({"line_spacing": e.control.value}),
                                        key="aa-spacing")
        self.margin_label = ft.Text(f"Margins {s.margins} dp", theme_style=ft.TextThemeStyle.LABEL_MEDIUM)
        self.margin_slider = ft.Slider(min=rm.MARGIN_RANGE[0], max=rm.MARGIN_RANGE[1],
                                       divisions=(rm.MARGIN_RANGE[1] - rm.MARGIN_RANGE[0]) // 2, value=s.margins,
                                       on_change_end=lambda e: self._emit({"margins": e.control.value}),
                                       key="aa-margins")
        return ft.ListView(
            controls=[
                ft.Container(self.family, padding=ft.Padding.only(top=8)),
                ft.Row([smaller, self.size_slider, larger, self.size_pill],
                       vertical_alignment=ft.CrossAxisAlignment.CENTER, spacing=4),
                self.spacing_label,
                self.spacing_slider,
                self.margin_label,
                self.margin_slider,
            ],
            spacing=6,
            padding=ft.Padding.symmetric(vertical=4),
        )

    def _swatch(self, index: int, theme: Mapping[str, Any]) -> ft.Control:
        selected = index == self.settings.theme and not self.settings.follow_app_theme
        return ft.Container(
            width=96,
            height=72,
            bgcolor=str(theme.get("bg") or "#1e1e1e"),
            border_radius=tokens.RADII["card"],
            border=ft.Border.all(3 if selected else 1,
                                 ft.Colors.PRIMARY if selected else str(theme.get("border") or "#555555")),
            padding=8,
            ink=True,
            on_click=lambda e, i=index: self._pick_theme(i),
            content=ft.Column(
                [
                    ft.Text("Aa", color=str(theme.get("fg") or "#ffffff"), size=18, weight=ft.FontWeight.W_600),
                    ft.Text(str(theme.get("name") or f"Theme {index + 1}"), color=str(theme.get("heading") or theme.get("fg")),
                            size=12),
                ],
                spacing=2,
                tight=True,
            ),
            key=f"aa-theme-{index}",
        )

    def _theme_tab(self) -> ft.Control:
        self.swatch_row = ft.Row([self._swatch(i, t) for i, t in enumerate(self.themes)], wrap=True, spacing=8,
                                 run_spacing=8, key="aa-swatches")
        self.follow_switch = ft.Switch(label="Follow app theme", value=self.settings.follow_app_theme,
                                       on_change=lambda e: self._emit({"follow_app_theme": bool(e.control.value)}),
                                       key="aa-follow")
        return ft.ListView(controls=[ft.Container(self.swatch_row, padding=ft.Padding.only(top=8)), self.follow_switch],
                           spacing=8)

    def _layout_tab(self) -> ft.Control:
        layouts = [rm.LAYOUT_SINGLE, rm.LAYOUT_SCROLL, rm.LAYOUT_ALL]
        if self.double_allowed:
            layouts.insert(1, rm.LAYOUT_DOUBLE)
        current = self.settings.layout if self.settings.layout in layouts else rm.LAYOUT_SINGLE
        self.layout_buttons = ft.SegmentedButton(
            segments=[ft.Segment(value=k, label=ft.Text(rm.LAYOUT_LABELS[k])) for k in layouts],
            selected=[current],
            show_selected_icon=False,
            on_change=lambda e: self._emit({"layout": next(iter(e.control.selected or [rm.LAYOUT_SINGLE]))}),
            key="aa-layout",
        )
        self.zones_switch = ft.Switch(label="Tap edges to turn pages", value=self.settings.tap_zones,
                                      on_change=lambda e: self._emit({"tap_zones": bool(e.control.value)}),
                                      key="aa-zones")
        self.awake_switch = ft.Switch(label="Keep screen on", value=self.settings.keep_screen_on,
                                      on_change=lambda e: self._emit({"keep_screen_on": bool(e.control.value)}),
                                      key="aa-awake")
        self.progress_switch = ft.Switch(label="Show progress %", value=self.settings.show_progress,
                                         on_change=lambda e: self._emit({"show_progress": bool(e.control.value)}),
                                         key="aa-progress")
        controls: list[ft.Control] = [ft.Container(self.layout_buttons, padding=ft.Padding.only(top=8))]
        if not self.double_allowed:
            controls.append(ft.Text("Double page is available on tablets in landscape.",
                                    theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT))
        controls += [self.zones_switch, self.awake_switch, self.progress_switch]
        return ft.ListView(controls=controls, spacing=6)

    # ---- events --------------------------------------------------------------------------------

    def _on_scope(self, e: Any) -> None:
        selected = next(iter(getattr(e.control, "selected", None) or [SCOPE_ALL]))
        self.scope = selected if selected in SCOPE_LABELS else SCOPE_ALL
        call_handler(self.on_scope, self.scope)

    def _on_size_drag(self, e: Any) -> None:
        size = rm.clamp_font_size(e.control.value)
        self._pill(size)

    def _on_spacing_drag(self, e: Any) -> None:
        self.spacing_label.value = f"Line spacing {rm.clamp_line_spacing(e.control.value):.1f}"
        self._push(self.spacing_label)

    def _pill(self, size: int) -> None:
        text = self.size_pill.content
        if isinstance(text, ft.Text):
            text.value = f"{size}pt"
        self._push(self.size_pill)

    def _set_size(self, value: Any) -> None:
        size = rm.clamp_font_size(value)
        self.size_slider.value = size
        self._pill(size)
        self._push(self.size_slider)
        self._emit({"font_size": size})

    def _pick_theme(self, index: int) -> None:
        self._emit({"theme": index, "follow_app_theme": False})

    def _emit(self, changes: dict) -> None:
        if not changes:
            return
        self.changes.append(dict(changes))
        call_handler(self.on_change, dict(changes), self.scope)

    # ---- external updates (pinch, other sources) -----------------------------------------------

    def apply_settings(self, settings: rm.ReaderSettings) -> None:
        self.settings = settings
        self.size_slider.value = settings.font_size
        self._pill(settings.font_size)
        self.spacing_slider.value = settings.line_spacing
        self.spacing_label.value = f"Line spacing {settings.line_spacing:.1f}"
        self.margin_slider.value = settings.margins
        self.margin_label.value = f"Margins {settings.margins} dp"
        self.follow_switch.value = settings.follow_app_theme
        self.swatch_row.controls = [self._swatch(i, t) for i, t in enumerate(self.themes)]
        self._push(self.size_slider, self.spacing_slider, self.spacing_label, self.margin_slider, self.margin_label,
                   self.follow_switch, self.swatch_row)

    def show(self, page: Any) -> None:
        self._page = page
        page.show_dialog(self.sheet)

    def close(self) -> None:
        close_dialog(self._page, self.sheet)

    @staticmethod
    def _push(*controls: Any) -> None:
        for control in controls:
            try:
                control.update()
            except Exception:
                pass
