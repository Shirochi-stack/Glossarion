"""Settings search (UI_SPEC §4.15 "Search", §5.7 ``SettingsSearch``).

``find_settings`` runs the schema search (label, help, key, env names,
section path) and applies the filter chips Modified · Locked · Unavailable on
mobile; with only a filter selected it lists every matching setting.
``SettingsSearch`` is the field + chips + grouped results (≤ 50, breadcrumbs)
used by Settings home and by the SectionPage search sheet; tapping a result
calls ``on_open(hit)``, which opens ``/settings/s/<section>#<key>`` (the page
then re-centres its window and ``scroll_to``s the tile).
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Optional

import flet as ft

from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.settings.model import config_path, label_for, summarize, tile_kind
from glossarion_mobile.ui.settings.schema_access import SEARCH_LIMIT, SearchHit
from glossarion_mobile.ui.settings.tiles import EffectiveConfig

__all__ = ["FILTERS", "SettingsSearch", "find_settings", "matches_filters", "search_sheet"]

FILTERS = (
    ("modified", "Modified"),
    ("locked", "Locked"),
    ("advanced", "Advanced"),
)  # U12 item 1: no "Unavailable on mobile" filter; such settings are not listed at all

#: The Settings home group the "Advanced" chip keeps (UI_SPEC §4.15: Other Stored Settings, Internal
#: State and what the curated map left of the desktop Context Management & Memory section).
ADVANCED_GROUP = "Advanced"


def _modified(ctx: Any, hit: SearchHit) -> bool:
    """The hit's tile shows the "modified" dot (a virtual control: any key it stands for is modified)."""
    check = getattr(ctx.schema, "is_modified", None)
    return bool(check(ctx.store, hit.key)) if callable(check) else ctx.store.is_modified(config_path(hit.spec))


def matches_filters(ctx: Any, hit: SearchHit, filters: Iterable[str], config: Optional[Any] = None) -> bool:
    for name in filters:
        if name == "modified" and not _modified(ctx, hit):
            return False
        if name == "unavailable" and ctx.schema.availability(hit.key)[0]:
            return False
        if name == "advanced" and str(getattr(hit, "group", "") or "") != ADVANCED_GROUP:
            return False
        if name == "locked":
            view = config if config is not None else EffectiveConfig(ctx.store)
            if not ctx.schema.lock_reason(hit.spec, view):
                return False
    return True


def find_settings(ctx: Any, query: str, filters: Iterable[str] = (), limit: int = SEARCH_LIMIT) -> list[SearchHit]:
    filters = [f for f in filters if f in dict(FILTERS)]
    text = (query or "").strip()
    config = EffectiveConfig(ctx.store)
    if text:
        candidates = ctx.schema.search(text, limit=limit if not filters else 10_000)
    elif filters:
        candidates = []
        for section in ctx.schema.sections():
            for spec in ctx.schema.specs_for(section):
                key = str(getattr(spec, "key", ""))
                candidates.append(SearchHit(key, spec, section.id, section.title, section.group))
    else:
        return []
    out: list[SearchHit] = []
    seen: set[str] = set()
    for hit in candidates:
        if hit.key in seen or not ctx.schema.availability(hit.key)[0] or not matches_filters(ctx, hit, filters, config):
            continue
        seen.add(hit.key)
        out.append(hit)
        if len(out) >= limit:
            break
    return out


class SettingsSearch:
    def __init__(
        self,
        ctx: Any,
        *,
        on_open: Callable[[SearchHit], Any],
        on_change: Optional[Callable[[], Any]] = None,
        autofocus: bool = False,
        hint: str = "Search settings",
    ) -> None:
        self.ctx = ctx
        self.on_open = on_open
        self.on_change = on_change
        self.query = ""
        self.filters: set[str] = set()
        self.hits: list[SearchHit] = []
        self.field = ft.TextField(
            color=ft.Colors.ON_SURFACE,
            hint_text=hint,
            prefix_icon=ft.Icons.SEARCH,
            dense=True,
            filled=True,
            border=ft.NoInputBorder(),
            border_radius=28,
            content_padding=ft.Padding.symmetric(horizontal=12, vertical=10),
            autofocus=autofocus,
            on_change=lambda e: self.set_query(e.control.value or ""),
            key="settings-search",
        )
        self.filter_chips: dict[str, ft.Chip] = {
            fid: ft.Chip(
                label=ft.Text(label),
                selected=False,
                show_checkmark=False,
                on_select=lambda e, f=fid: self.toggle_filter(f),
                key=f"settings-filter-{fid}",
            )
            for fid, label in FILTERS
        }
        self.filter_row = ft.Row(list(self.filter_chips.values()), spacing=6, scroll=ft.ScrollMode.AUTO)
        self.results = ft.ListView(expand=True, spacing=2, padding=ft.Padding.symmetric(vertical=4))
        self.result_rows: dict[str, ft.ListTile] = {}

    @property
    def active(self) -> bool:
        return bool(self.query or self.filters)

    def set_query(self, query: str) -> None:
        self.query = (query or "").strip()
        self.refresh()

    def toggle_filter(self, name: str, on: Optional[bool] = None) -> None:
        enable = (name not in self.filters) if on is None else bool(on)
        if enable:
            self.filters.add(name)
        else:
            self.filters.discard(name)
        chip = self.filter_chips.get(name)
        if chip is not None:
            chip.selected = enable
        self.refresh()

    def refresh(self, push: bool = True) -> None:
        self.hits = find_settings(self.ctx, self.query, self.filters) if self.active else []
        self.results.controls = self.result_controls()
        if push:
            self.ctx.push(self.results, self.filter_row)
        if self.on_change is not None:
            self.on_change()

    def _note(self, text: str) -> ft.Control:
        return ft.Container(
            padding=ft.Padding.symmetric(horizontal=16, vertical=12),
            content=ft.Text(text, theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT),
        )

    def result_controls(self) -> list[ft.Control]:
        self.result_rows = {}
        if not self.active:
            return []
        if not self.ctx.schema.available:
            return [self._note("Search needs the settings schema, which is not available in this build.")]
        if not self.hits:
            what = f"“{self.query}”" if self.query else "these filters"
            return [self._note(f"No settings match {what}.")]
        controls: list[ft.Control] = []
        current_section = None
        for hit in self.hits:
            if hit.section_id != current_section:
                current_section = hit.section_id
                controls.append(
                    ft.Container(
                        padding=ft.Padding.only(left=16, top=10, bottom=2),
                        content=ft.Text(hit.breadcrumb, theme_style=ft.TextThemeStyle.LABEL_SMALL, color=ft.Colors.PRIMARY),
                        key=f"settings-hit-section-{hit.section_id}",
                    )
                )
            controls.append(self._row(hit))
        if len(self.hits) >= SEARCH_LIMIT:
            controls.append(self._note(f"Showing the first {SEARCH_LIMIT} matches; refine the search to see more."))
        return controls

    def _row(self, hit: SearchHit) -> ft.ListTile:
        path = config_path(hit.spec)
        value = self.ctx.store.effective(path)
        kind = tile_kind(hit.spec, self.ctx.store.get(path))
        shown = (f"{len(value)} keys" if isinstance(value, (list, dict)) else "Not set") if (
            kind == "json" and str(getattr(hit.spec, "type", "")) == "secret") else summarize(value, kind, hit.spec)
        virtual = getattr(self.ctx.schema, "virtual_summary", None)
        shown = (virtual(self.ctx.store, hit.key) if callable(virtual) else None) or shown
        subtitle = f"{hit.key} · {shown}"
        available, reason = self.ctx.schema.availability(hit.key)
        trailing: Optional[ft.Control] = None
        if not available:
            trailing = ft.Text(reason or "Not on mobile", theme_style=ft.TextThemeStyle.LABEL_SMALL,
                               color=ft.Colors.ON_SURFACE_VARIANT)
        elif _modified(self.ctx, hit):
            trailing = ft.Container(width=8, height=8, border_radius=4, bgcolor=ft.Colors.PRIMARY, tooltip="Modified")
        row = ft.ListTile(
            title=ft.Text(label_for(hit.spec), theme_style=ft.TextThemeStyle.BODY_MEDIUM, color=ft.Colors.ON_SURFACE),
            subtitle=ft.Text(subtitle, theme_style=ft.TextThemeStyle.BODY_SMALL, max_lines=1,
                             overflow=ft.TextOverflow.ELLIPSIS, color=ft.Colors.ON_SURFACE_VARIANT),
            trailing=trailing,
            dense=True,
            min_height=tokens.SIZES["hit_target"],
            on_click=lambda e, h=hit: self.on_open(h),
            key=f"settings-hit-{hit.key}",
        )
        self.result_rows[hit.key] = row
        return row

def search_sheet(search: "SettingsSearch") -> ft.BottomSheet:
    """The search sheet (field, filter chips, results) the section pages' search button and the
    chat's ``/settings <query>`` show."""
    return ft.BottomSheet(
        content=ft.Container(
            padding=ft.Padding.only(left=12, right=12, bottom=12),
            height=520,
            content=ft.Column([search.field, search.filter_row, search.results],
                              spacing=tokens.SPACING["sm"], expand=True,
                              horizontal_alignment=ft.CrossAxisAlignment.STRETCH),
        ),
        show_drag_handle=True,
        scrollable=True,
        bgcolor=ft.Colors.SURFACE_CONTAINER_HIGH,
    )
