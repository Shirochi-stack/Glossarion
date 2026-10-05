"""Screen base, placeholder screens and simple hub screens.

A ``Screen`` provides a title, optional app-bar actions and a body control.
``AppShell`` wraps it in a pushed ``View`` with an app bar (phone) or places
the body in the tablet main area under a title bar.

Every whitelisted route already has a surface in U1: real screens where the
milestone has shipped (Logs & diagnostics), hubs that list their child routes
(Settings, Tools) and a ``PlaceholderScreen`` naming the milestone that ships
the rest (nothing silently missing, UI_SPEC §0 item 6).
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

import flet as ft

from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.components.empty_state import EmptyState
from glossarion_mobile.ui.components.reason_chip import ReasonChip
from glossarion_mobile.ui.router import ROUTES, ROUTES_BY_NAME, RouteMatch, RouteSpec

__all__ = ["HubScreen", "PlaceholderScreen", "Screen", "SHIPPED_MILESTONES", "ROUTE_ICONS", "build_screen_view"]

# Milestones whose surfaces exist in this build.
SHIPPED_MILESTONES = frozenset({"U0", "U1"})

ROUTE_ICONS = {
    "library": "LOCAL_LIBRARY",
    "jobs": "WORK_HISTORY",
    "glossary": "SPELLCHECK",
    "tools": "HANDYMAN",
    "settings": "SETTINGS",
    "reader": "AUTO_STORIES",
    "welcome": "WAVING_HAND",
    "chat": "CHAT_BUBBLE_OUTLINE",
    "series": "COLLECTIONS_BOOKMARK",
}


def _route_icon(spec: RouteSpec) -> str:
    head = spec.name.split(".", 1)[0]
    return ROUTE_ICONS.get(head, "CONSTRUCTION")


class Screen:
    """Base class: override ``build_body`` (and optionally ``actions``/``dispose``)."""

    title = ""

    def __init__(self, match: Optional[RouteMatch] = None) -> None:
        self.match = match
        self.body: Optional[ft.Control] = None

    @property
    def route(self) -> str:
        return self.match.route if self.match is not None else "/"

    def build_body(self) -> ft.Control:
        raise NotImplementedError

    def get_body(self) -> ft.Control:
        if self.body is None:
            self.body = self.build_body()
        return self.body

    def actions(self) -> list[ft.Control]:
        return []

    def did_show(self) -> None:
        """Called after the screen's View/body is on the page."""

    def dispose(self) -> None:
        """Called when the screen leaves the stack."""


class PlaceholderScreen(Screen):
    def __init__(self, match: RouteMatch) -> None:
        super().__init__(match)
        self.spec = match.spec
        self.title = self.spec.title

    def build_body(self) -> ft.Control:
        return EmptyState(
            icon=_route_icon(self.spec),
            title=self.spec.title,
            body=f"This screen arrives in {self.spec.milestone}. Its settings and data are kept untouched until then.",
            key=f"placeholder-{self.spec.name}",
        )


class HubScreen(Screen):
    """Lists the static child routes of a route (Settings, Tools)."""

    def __init__(self, match: RouteMatch, *, navigate: Callable[[str], Any], intro: Optional[str] = None) -> None:
        super().__init__(match)
        self.spec = match.spec
        self.title = self.spec.title
        self.navigate = navigate
        self.intro = intro
        self.tiles: dict[str, ft.ListTile] = {}

    def children(self) -> Sequence[RouteSpec]:
        return [s for s in ROUTES if s.parent == self.spec.name and s.is_static and s.alias_of is None]

    def build_body(self) -> ft.Control:
        controls: list[ft.Control] = []
        if self.intro:
            controls.append(
                ft.Container(
                    padding=ft.Padding.symmetric(horizontal=16, vertical=8),
                    content=ft.Text(self.intro, theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT),
                )
            )
        for child in self.children():
            shipped = child.milestone in SHIPPED_MILESTONES
            tile = ft.ListTile(
                title=ft.Text(child.title),
                trailing=None if shipped else ReasonChip(reason=f"Arrives in {child.milestone}"),
                leading=ft.Icon(ft.Icons.CHEVRON_RIGHT if shipped else ft.Icons.SCHEDULE),
                on_click=lambda e, name=child.name: self.navigate(name),
                min_height=tokens.SIZES["hit_target"],
                key=f"hub-{child.name}",
            )
            self.tiles[child.name] = tile
            controls.append(tile)
        return ft.ListView(controls=controls, expand=True, padding=ft.Padding.symmetric(vertical=8))


def build_screen_view(screen: Screen, route: str) -> ft.View:
    """A pushed phone View: app bar with the implied back arrow + the screen body."""
    return ft.View(
        route=route,
        appbar=ft.AppBar(
            title=ft.Text(screen.title, theme_style=ft.TextThemeStyle.TITLE_MEDIUM, weight=ft.FontWeight.W_600),
            actions=screen.actions() or None,
            bgcolor=ft.Colors.SURFACE,
            elevation_on_scroll=0,
            toolbar_height=tokens.SIZES["app_bar"],
            center_title=False,
        ),
        padding=0,
        spacing=0,
        controls=[ft.SafeArea(content=screen.get_body(), expand=True)],
    )


def spec_for(name: str) -> RouteSpec:
    return ROUTES_BY_NAME[name]
