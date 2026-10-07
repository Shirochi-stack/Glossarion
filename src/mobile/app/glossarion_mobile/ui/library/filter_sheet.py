"""Library filter sheet (UI_SPEC §3.1): ``TabBar`` Filter · Sort · Display.

* **Filter:** format chips All / EPUB / TXT / PDF / HTML / IMG (``epub_library_format_filter``);
  tri-state state chips (In progress · Ready to compile · Not started · Outdated ·
  Has QA failures · Missing raw: off → only → hide); Series (U9, disabled).
* **Sort:** Date / A-Z / Size (``epub_library_sort``) + Reverse.
* **Display:** Grid / List · Density (all 11 desktop presets 2XS-6XL,
  ``epub_library_card_size``) · Raw titles (``epub_library_show_raw_titles``) ·
  Show language badge · Show progress bar on cover · Page size 20 / 50 / 100 /
  250 / 500 / All (``epub_library_page_size``).

Every change is applied at once through ``on_change(field, value)``; the home
screen persists the desktop keys (sparse writes) and the mobile-only display
switches in Prefs.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Optional

import flet as ft

from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.components.dialogs import close_dialog
from glossarion_mobile.ui.components.reason_chip import ReasonChip
from glossarion_mobile.ui.library.models import (
    DENSITY_LABELS,
    DENSITY_ORDER,
    FORMATS,
    PAGE_SIZES,
    SORTS,
    STATE_FILTERS,
    FilterState,
    next_tristate,
)

__all__ = ["FilterSheet", "tristate_label"]

_TAB_HEIGHT = 380


def tristate_label(label: str, value: Optional[bool]) -> str:
    if value is True:
        return f"✓ {label}"
    if value is False:
        return f"✕ {label}"
    return label


class FilterSheet:
    def __init__(
        self,
        state: FilterState,
        *,
        view_mode: str = "grid",
        density: str = "compact",
        raw_titles: bool = False,
        show_language: bool = False,
        show_progress: bool = True,
        page_size: str = "20",
        on_change: Optional[Callable[[str, Any], Any]] = None,
        initial_tab: int = 0,
    ) -> None:
        self.state = state
        self.view_mode = view_mode
        self.density = density
        self.raw_titles = raw_titles
        self.show_language = show_language
        self.show_progress = show_progress
        self.page_size = str(page_size)
        self.on_change = on_change
        self._page: Any = None
        self.format_chips: dict[str, ft.Chip] = {}
        self.state_chips: dict[str, ft.Chip] = {}
        self.density_chips: dict[str, ft.Chip] = {}
        self.page_chips: dict[str, ft.Chip] = {}
        self.sort_buttons = ft.SegmentedButton(
            segments=[ft.Segment(value=k, label=ft.Text(v)) for k, v in SORTS],
            selected=[state.sort if state.sort in dict(SORTS) else "date"],
            show_selected_icon=False,
            on_change=self._on_sort,
            key="sort",
        )
        self.reverse_switch = ft.Switch(label="Reverse", value=state.reverse, on_change=self._on_reverse,
                                        key="reverse")
        self.view_buttons = ft.SegmentedButton(
            segments=[ft.Segment(value="grid", label=ft.Text("Grid"), icon=ft.Icons.GRID_VIEW),
                      ft.Segment(value="list", label=ft.Text("List"), icon=ft.Icons.VIEW_LIST)],
            selected=[view_mode if view_mode in ("grid", "list") else "grid"],
            show_selected_icon=False,
            on_change=self._on_view,
            key="view",
        )
        self.raw_switch = ft.Switch(label="Raw titles", value=raw_titles,
                                    on_change=lambda e: self._emit("raw_titles", bool(e.control.value)),
                                    key="raw-titles")
        self.language_switch = ft.Switch(label="Show language badge", value=show_language,
                                         on_change=lambda e: self._emit("show_language", bool(e.control.value)),
                                         key="show-language")
        self.progress_switch = ft.Switch(label="Show progress bar on cover", value=show_progress,
                                         on_change=lambda e: self._emit("show_progress", bool(e.control.value)),
                                         key="show-progress")
        self.tabs = ft.Tabs(
            length=3,
            selected_index=max(0, min(2, initial_tab)),
            content=ft.Column([
                ft.TabBar(tabs=[ft.Tab(label="Filter"), ft.Tab(label="Sort"), ft.Tab(label="Display")]),
                ft.TabBarView(controls=[self._filter_tab(), self._sort_tab(), self._display_tab()],
                              height=_TAB_HEIGHT),
            ], tight=True, spacing=0),
        )
        self.sheet = ft.BottomSheet(
            content=ft.Container(content=self.tabs, padding=ft.Padding.only(left=12, right=12, bottom=12)),
            show_drag_handle=True,
            scrollable=True,
            bgcolor=ft.Colors.SURFACE_CONTAINER_HIGH,
        )

    # ---- tabs --------------------------------------------------------------------------------

    @staticmethod
    def _label(text: str) -> ft.Text:
        return ft.Text(text, theme_style=ft.TextThemeStyle.TITLE_SMALL, color=ft.Colors.PRIMARY)

    def _filter_tab(self) -> ft.Control:
        for value, label in FORMATS:
            self.format_chips[value] = ft.Chip(
                label=ft.Text(label), selected=self.state.fmt == value, show_checkmark=False,
                on_select=lambda e, v=value: self._on_format(v), key=f"fmt-{value}")
        for key, (label, _pred) in STATE_FILTERS.items():
            value = self.state.states.get(key)
            self.state_chips[key] = ft.Chip(
                label=ft.Text(tristate_label(label, value)), selected=value is not None, show_checkmark=False,
                on_select=lambda e, k=key: self._on_state(k), key=f"state-{key}")
        return ft.ListView([
            self._label("Format"),
            ft.Row(list(self.format_chips.values()), wrap=True, spacing=6, run_spacing=6),
            self._label("State"),
            ft.Text("Tap once to show only, twice to hide, three times to clear.",
                    theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT),
            ft.Row(list(self.state_chips.values()), wrap=True, spacing=6, run_spacing=6),
            ft.Row([ft.Text("Series", expand=True), ReasonChip(reason="Arrives in U9")]),
        ], spacing=tokens.SPACING["sm"], padding=ft.Padding.only(top=12))

    def _sort_tab(self) -> ft.Control:
        return ft.ListView([self._label("Sort by"), self.sort_buttons, self.reverse_switch],
                           spacing=tokens.SPACING["md"], padding=ft.Padding.only(top=12))

    def _display_tab(self) -> ft.Control:
        for key in DENSITY_ORDER:
            self.density_chips[key] = ft.Chip(
                label=ft.Text(DENSITY_LABELS[key]), selected=self.density == key, show_checkmark=False,
                on_select=lambda e, k=key: self._on_density(k), key=f"density-{key}")
        for value, label in PAGE_SIZES:
            self.page_chips[value] = ft.Chip(
                label=ft.Text(label), selected=self.page_size.lower() == value, show_checkmark=False,
                on_select=lambda e, v=value: self._on_page_size(v), key=f"page-{value}")
        return ft.ListView([
            self._label("View"),
            self.view_buttons,
            self._label("Density"),
            ft.Row(list(self.density_chips.values()), wrap=True, spacing=6, run_spacing=6),
            self.raw_switch,
            self.language_switch,
            self.progress_switch,
            self._label("Page size"),
            ft.Row(list(self.page_chips.values()), wrap=True, spacing=6, run_spacing=6),
        ], spacing=tokens.SPACING["sm"], padding=ft.Padding.only(top=12))

    # ---- events ---------------------------------------------------------------------------------

    def _emit(self, name: str, value: Any) -> None:
        if self.on_change is not None:
            self.on_change(name, value)

    @staticmethod
    def _selected_value(control: Any) -> Optional[str]:
        selected = getattr(control, "selected", None) or []
        return str(next(iter(selected))) if selected else None

    def _on_format(self, value: str) -> None:
        self.state.fmt = value
        for key, chip in self.format_chips.items():
            chip.selected = key == value
        self._refresh(*self.format_chips.values())
        self._emit("fmt", value)

    def _on_state(self, key: str) -> None:
        value = next_tristate(self.state.states.get(key))
        if value is None:
            self.state.states.pop(key, None)
        else:
            self.state.states[key] = value
        chip = self.state_chips[key]
        chip.selected = value is not None
        chip.label = ft.Text(tristate_label(STATE_FILTERS[key][0], value))
        self._refresh(chip)
        self._emit("states", dict(self.state.states))

    def _on_sort(self, e: Any = None) -> None:
        value = self._selected_value(self.sort_buttons) or "date"
        self.state.sort = value
        self._emit("sort", value)

    def _on_reverse(self, e: Any = None) -> None:
        self.state.reverse = bool(self.reverse_switch.value)
        self._emit("reverse", self.state.reverse)

    def _on_view(self, e: Any = None) -> None:
        value = self._selected_value(self.view_buttons) or "grid"
        self.view_mode = value
        self._emit("view", value)

    def _on_density(self, key: str) -> None:
        self.density = key
        for k, chip in self.density_chips.items():
            chip.selected = k == key
        self._refresh(*self.density_chips.values())
        self._emit("density", key)

    def _on_page_size(self, value: str) -> None:
        self.page_size = value
        for k, chip in self.page_chips.items():
            chip.selected = k == value
        self._refresh(*self.page_chips.values())
        self._emit("page_size", value)

    @staticmethod
    def _refresh(*controls: Any) -> None:
        for control in controls:
            try:
                control.update()
            except Exception:
                pass

    # ---- show ----------------------------------------------------------------------------------

    def show(self, page: Any) -> None:
        self._page = page
        page.show_dialog(self.sheet)

    def close(self) -> None:
        close_dialog(self._page, self.sheet)

    def snapshot(self) -> Mapping[str, Any]:
        return {"fmt": self.state.fmt, "sort": self.state.sort, "reverse": self.state.reverse,
                "states": dict(self.state.states), "view": self.view_mode, "density": self.density,
                "raw_titles": bool(self.raw_switch.value), "page_size": self.page_size}
