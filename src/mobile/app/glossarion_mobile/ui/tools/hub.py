"""Tools hub (``/tools``, UI_SPEC §4.2).

A grid of tool tiles (icon, name, last source) in the UI_SPEC groups. A tile opens its
route; a tool whose milestone has not shipped stays visible, disabled, with a ReasonChip
("Arrives in U7"). The last source a tool ran on comes from ``mobile_state.json``
(``ToolsContext.last_source``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import flet as ft

from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.components.reason_chip import ReasonChip
from glossarion_mobile.ui.router import ROUTES_BY_NAME, RouteMatch
from glossarion_mobile.ui.screens.base import SHIPPED_MILESTONES, Screen
from glossarion_mobile.ui.theme import icon_data

__all__ = ["HUB_GROUPS", "ToolTile", "ToolsHubScreen", "tile_available"]


@dataclass(frozen=True)
class ToolTile:
    key: str  # last-source key (route name)
    name: str
    icon: str
    route: str
    params: Optional[dict] = None
    query: Optional[dict] = None
    hint: str = ""


#: UI_SPEC §4.2 groups and tiles.
HUB_GROUPS = (
    ("Translate", (
        ToolTile("tools.async", "Async batch", "SCHEDULE_SEND", "tools.async"),
        ToolTile("tools.review", "Review generator", "RATE_REVIEW", "tools.review"),
        ToolTile("tools.headers", "Headers & metadata", "TITLE", "tools.headers"),
        ToolTile("tools.rpgmaker", "RPG Maker", "VIDEOGAME_ASSET", "tools.rpgmaker"),
    )),
    ("Check", (
        ToolTile("tools.qa", "QA Scanner", "FACT_CHECK", "tools.qa"),
        ToolTile("tools.progress", "Progress manager", "LIST_ALT", "tools.progress"),
        ToolTile("tools.progress.glossary", "Glossary progress", "SPELLCHECK", "tools.progress.glossary"),
        ToolTile("tools.sdlxliff", "SDLXLIFF reviewer", "TRANSLATE", "tools.sdlxliff"),
    )),
    ("Build", (
        ToolTile("tools.convert", "Converter / Compile", "MENU_BOOK", "tools.convert"),
        ToolTile("tools.convert.validate", "Validate EPUB", "RULE", "tools.convert", query={"tab": "validate"}),
    )),
    ("Images", (
        ToolTile("tools.manga", "Manga translator", "AUTO_STORIES", "tools.manga"),
    )),
    ("Files", (
        ToolTile("tools.files", "File browser", "FOLDER_OPEN", "tools.files", params={"root": "output"}),
        ToolTile("tools.text", "Text editor", "EDIT_NOTE", "tools.files", params={"root": "output"},
                 hint="Open a file from the file browser"),
    )),
)


def tile_available(tile: ToolTile, implemented: frozenset = frozenset()) -> Optional[str]:
    """None when the tool opens; otherwise the reason (its milestone has not shipped)."""
    if tile.route in implemented:
        return None
    spec = ROUTES_BY_NAME.get(tile.route)
    if spec is None:
        return "Not in this build"
    if spec.milestone in SHIPPED_MILESTONES:
        return None
    return f"Arrives in {spec.milestone}"


class ToolsHubScreen(Screen):
    title = "Tools"

    def __init__(self, match: Optional[RouteMatch], ctx: Any, *, implemented: frozenset = frozenset()) -> None:
        super().__init__(match)
        self.ctx = ctx
        self.implemented = frozenset(implemented)
        self.tiles: dict = {}

    def build_body(self) -> ft.Control:
        controls: list[ft.Control] = []
        for group, tiles in HUB_GROUPS:
            controls.append(ft.Container(
                content=ft.Text(group, theme_style=ft.TextThemeStyle.TITLE_SMALL, color=ft.Colors.PRIMARY,
                                weight=ft.FontWeight.W_600),
                padding=ft.Padding.only(left=4, top=8), key=f"hub-group-{group}"))
            controls.append(ft.ResponsiveRow([self._tile(t) for t in tiles], spacing=8, run_spacing=8))
        return ft.ListView(controls=controls, expand=True, spacing=tokens.SPACING["sm"],
                           padding=ft.Padding.symmetric(horizontal=12, vertical=8), key="tools-hub")

    def _tile(self, tile: ToolTile) -> ft.Control:
        reason = tile_available(tile, self.implemented)
        last = self.ctx.last_source(tile.key) if hasattr(self.ctx, "last_source") else ""
        subtitle = last or tile.hint
        texts: list[ft.Control] = [
            ft.Text(tile.name, theme_style=ft.TextThemeStyle.TITLE_SMALL, weight=ft.FontWeight.W_600,
                    max_lines=2, overflow=ft.TextOverflow.ELLIPSIS),
        ]
        if subtitle:
            texts.append(ft.Text(subtitle, theme_style=ft.TextThemeStyle.BODY_SMALL,
                                 color=ft.Colors.ON_SURFACE_VARIANT, max_lines=1,
                                 overflow=ft.TextOverflow.ELLIPSIS, key=f"tile-sub-{tile.key}"))
        if reason:
            texts.append(ReasonChip(reason=reason))
        container = ft.Container(
            content=ft.Row([ft.Icon(icon_data(tile.icon), color=ft.Colors.PRIMARY, size=28),
                            ft.Column(texts, spacing=2, tight=True, expand=True)],
                           spacing=10, vertical_alignment=ft.CrossAxisAlignment.CENTER),
            height=88,
            padding=ft.Padding.symmetric(horizontal=12, vertical=8),
            bgcolor=ft.Colors.SURFACE_CONTAINER_LOW,
            border_radius=tokens.RADII["tile"],
            opacity=0.55 if reason else 1.0,
            on_click=None if reason else (lambda e, t=tile: self.open(t)),
            ink=reason is None,
            col={"xs": 12, "sm": 6, "md": 4, "lg": 3},
            key=f"tile-{tile.key}",
        )
        self.tiles[tile.key] = container
        return container

    def open(self, tile: ToolTile) -> None:
        if tile_available(tile, self.implemented) is not None:
            return
        self.ctx.go(tile.route, tile.params, tile.query)
