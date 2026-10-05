"""AppShell (UI_SPEC §1.1-§1.2, §5.1): layout and navigation stack.

Phone and large phone: ``page.views`` is a stack. ``View("/")`` is the chat
home (``appbar=ChatHeader``, ``drawer=NavigationDrawer`` holding the
ChatDrawer); every destination is pushed as its own ``View(route)``; back pops.

Tablet (>= 900 dp): one root ``View`` with ``Row[Sidebar, MainArea, SidePanel?]``.
The router swaps the main area's content and keeps a back stack; the sidebar
(the same ChatDrawer content) never unmounts. Full-screen surfaces (Reader,
editors, Welcome) are pushed as top-level Views on every size class.

The layout is rebuilt only when the size class changes (``apply_width``);
crossing the phone/tablet boundary moves the drawer content between the modal
drawer and the sidebar and re-wraps the current screens.

Routes with presentation ``sheet`` open a placeholder bottom sheet without
touching the stack; ``handled`` routes never reach the shell (the app handles
them). Views pushed with ``push_overlay`` (the U0 device-checks screen) are
not routable and are dropped on the next navigation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import flet as ft

from glossarion_mobile.state.app_state import AppState, JobStripModel
from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.chat.chat_view import ChatView
from glossarion_mobile.ui.components.info_sheet import InfoSheet
from glossarion_mobile.ui.responsive import Layout, SizeClass, layout_for
from glossarion_mobile.ui.router import FULLSCREEN, HANDLED, ROOT, ROUTES_BY_NAME, SHEET, RouteMatch, parse_route
from glossarion_mobile.ui.screens.base import Screen, build_screen_view
from glossarion_mobile.ui.shell.drawer import ChatDrawer
from glossarion_mobile.ui.shell.job_strip import JobStrip
from glossarion_mobile.ui.shell.side_panel import SidePanel
from glossarion_mobile.ui.theme import HIT_TARGET

__all__ = ["AppShell", "StackEntry"]

log = logging.getLogger("glossarion.shell")

ScreenFactory = Callable[[RouteMatch], Screen]


@dataclass
class StackEntry:
    match: RouteMatch
    screen: Screen
    view: Optional[ft.View] = None  # phone (and full-screen) wrapper, built lazily
    footer: Optional[ft.Container] = None  # slot that holds the global JobStrip on the top view
    panel: Optional[ft.Control] = None  # tablet main-area wrapper
    extra: dict = field(default_factory=dict)

    @property
    def route(self) -> str:
        return self.match.route

    @property
    def fullscreen(self) -> bool:
        return self.match.presentation == FULLSCREEN


class AppShell:
    def __init__(
        self,
        page: Any,
        *,
        state: AppState,
        chat_view: ChatView,
        drawer: ChatDrawer,
        screen_factory: ScreenFactory,
        on_back: Optional[Callable[[], Any]] = None,
    ) -> None:
        self.page = page
        self.state = state
        self.chat_view = chat_view
        self.drawer = drawer
        self.screen_factory = screen_factory
        self.on_back = on_back  # tablet main-area back button -> app pops and syncs the route
        self.layout: Layout = layout_for(getattr(page, "width", None) or 0, state.text_scale.value)
        self.current: RouteMatch = parse_route("/")  # the chat-root route ("/" or "/chat/<cid>")
        self.stack: list[StackEntry] = []
        self.overlays: list[ft.View] = []
        self.side_panel = SidePanel()
        self.global_strip = JobStrip(on_open=lambda: None)
        self.root_view: Optional[ft.View] = None
        self.nav_drawer: Optional[ft.NavigationDrawer] = None
        self.drawer_box: Optional[ft.Container] = None
        self.sidebar: Optional[ft.Container] = None
        self.sidebar_strip_slot: Optional[ft.Container] = None
        self.main_area: Optional[ft.Container] = None
        self.tablet_back_button = ft.IconButton(
            icon=ft.Icons.ARROW_BACK, tooltip="Back", on_click=self._on_tablet_back, size_constraints=HIT_TARGET
        )
        self.tablet_title = ft.Text("", theme_style=ft.TextThemeStyle.TITLE_MEDIUM, weight=ft.FontWeight.W_600, expand=True)
        self.sheets_shown: list[InfoSheet] = []
        self.builds = 0
        self._unsubs: list[Callable[[], None]] = []

    # ---- properties --------------------------------------------------------------

    @property
    def tablet(self) -> bool:
        return self.layout.persistent_sidebar

    @property
    def size_class(self) -> SizeClass:
        return self.layout.size_class

    @property
    def current_route(self) -> str:
        """Route of what is on screen (top of the stack, else the chat root)."""
        if self.stack:
            return self.stack[-1].route
        return self.current.route

    @property
    def top_screen(self) -> Optional[Screen]:
        return self.stack[-1].screen if self.stack else None

    # ---- building ------------------------------------------------------------------

    def _drawer_height(self) -> float:
        height = getattr(self.page, "height", None) or 640
        media = getattr(self.page, "media", None)
        top = 0.0
        try:
            top = float(media.padding.top or 0) if media is not None and media.padding is not None else 0.0
        except Exception:
            top = 0.0
        return max(360.0, float(height) - top)

    def mount(self) -> None:
        """Build the layout for the current width and install ``page.views``."""
        self._build()
        self._install_views()

    def attach(self) -> None:
        if not self._unsubs:
            self._unsubs = [self.state.job_strip.subscribe(self._on_job_strip)]
        self.drawer.attach()
        self.chat_view.attach()

    def detach(self) -> None:
        for unsub in self._unsubs:
            unsub()
        self._unsubs = []
        self.drawer.detach()
        self.chat_view.detach()
        for entry in self.stack:
            entry.screen.dispose()

    def _build(self) -> None:
        self.builds += 1
        layout = self.layout
        self.chat_view.apply_layout(layout)
        self.global_strip.set_model(self.state.job_strip.value)
        # Every wrapper is re-created; inner controls (drawer content, chat column,
        # screen bodies) move into the new wrappers.
        for entry in self.stack:
            entry.view = None
            entry.footer = None
            entry.panel = None
        if not self.tablet:
            self.sidebar = None
            self.main_area = None
            self.sidebar_strip_slot = None
            self.drawer_box = ft.Container(content=self.drawer.content, height=self._drawer_height())
            self.nav_drawer = ft.NavigationDrawer(
                controls=[self.drawer_box],
                width=layout.drawer_width,
                bgcolor=ft.Colors.SURFACE_CONTAINER_LOW,
            )
            self.root_view = ft.View(
                route="/",
                appbar=self.chat_view.header.build(tablet=False),
                drawer=self.nav_drawer,
                padding=0,
                spacing=0,
                controls=[self.chat_view.build_body(layout)],
            )
            return
        self.nav_drawer = None
        self.drawer_box = None
        self.sidebar_strip_slot = ft.Container(content=None)
        self.sidebar = ft.Container(
            width=layout.sidebar_width,
            bgcolor=ft.Colors.SURFACE_CONTAINER_LOW,
            content=ft.SafeArea(
                content=ft.Column(
                    [ft.Container(content=self.drawer.content, expand=True), self.sidebar_strip_slot],
                    spacing=0,
                    expand=True,
                ),
                expand=True,
            ),
        )
        self.side_panel.set_pin_available(layout.size_class is SizeClass.WIDE)
        self.main_area = ft.Container(expand=True, content=self._main_content())
        self.root_view = ft.View(
            route="/",
            padding=0,
            spacing=0,
            controls=[
                ft.Row(
                    [self.sidebar, self.main_area, self.side_panel.control],
                    spacing=0,
                    expand=True,
                    vertical_alignment=ft.CrossAxisAlignment.STRETCH,
                )
            ],
        )

    def _main_available_width(self) -> float:
        width = self.layout.width - self.layout.sidebar_width
        if self.side_panel.is_open:
            width -= self.layout.side_panel_width
        return max(0.0, width)

    def _main_content(self) -> ft.Control:
        """Tablet main area: the chat, or the top non-full-screen screen with a title bar."""
        entry = self._top_panel_entry()
        if entry is None:
            return ft.Column(
                [
                    self.chat_view.header.build(tablet=True),
                    ft.Container(content=self.chat_view.build_body(self.layout, self._main_available_width()), expand=True),
                ],
                spacing=0,
                expand=True,
            )
        self.tablet_title = ft.Text(
            entry.screen.title, theme_style=ft.TextThemeStyle.TITLE_MEDIUM, weight=ft.FontWeight.W_600, expand=True
        )
        self.tablet_back_button = ft.IconButton(
            icon=ft.Icons.ARROW_BACK, tooltip="Back", on_click=self._on_tablet_back, size_constraints=HIT_TARGET
        )
        entry.panel = ft.Column(
            [
                ft.Container(
                    height=tokens.SIZES["app_bar"],
                    padding=ft.Padding.only(left=4, right=4),
                    content=ft.Row(
                        [self.tablet_back_button, self.tablet_title, *entry.screen.actions()],
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        spacing=4,
                    ),
                ),
                ft.Container(content=entry.screen.get_body(), expand=True),
            ],
            spacing=0,
            expand=True,
        )
        return entry.panel

    def _top_panel_entry(self) -> Optional[StackEntry]:
        for entry in reversed(self.stack):
            if not entry.fullscreen:
                return entry
        return None

    def _entry_view(self, entry: StackEntry) -> ft.View:
        if entry.view is None:
            entry.footer = ft.Container(content=None)
            view = build_screen_view(entry.screen, entry.route)
            safe = view.controls[0]
            safe.content = ft.Column([ft.Container(content=safe.content, expand=True), entry.footer], spacing=0, expand=True)
            entry.view = view
        return entry.view

    def _install_views(self) -> None:
        views: list[ft.View] = [self.root_view]
        if self.tablet:
            self.main_area.content = self._main_content()
            pushed = [e for e in self.stack if e.fullscreen]
        else:
            pushed = list(self.stack)
        for entry in pushed:
            views.append(self._entry_view(entry))
        views.extend(self.overlays)
        self.page.views.clear()
        self.page.views.extend(views)
        self._place_strip()

    def _place_strip(self) -> None:
        """The global JobStrip sits on the top pushed View (phone) or the sidebar (tablet)."""
        for entry in self.stack:
            if entry.footer is not None:
                entry.footer.content = None
        if self.sidebar_strip_slot is not None:
            self.sidebar_strip_slot.content = None
        if self.tablet:
            self.sidebar_strip_slot.content = self.global_strip
            return
        if self.stack and self.stack[-1].footer is not None and self.stack[-1].match.name != "reader":
            self.stack[-1].footer.content = self.global_strip

    def _on_job_strip(self, model: Optional[JobStripModel]) -> None:
        self.global_strip.set_model(model)

    # ---- navigation -------------------------------------------------------------------

    def _chain(self, match: RouteMatch) -> list[RouteMatch]:
        """The route plus its static parents (phone back stack), outermost first."""
        chain = [match]
        parent = match.spec.parent
        seen = {match.name}
        while parent and parent not in seen:
            seen.add(parent)
            spec = ROUTES_BY_NAME[parent]
            if spec.presentation == ROOT or not spec.is_static:
                break
            parent_match = parse_route(spec.pattern)
            if parent_match is None:
                break
            chain.insert(0, parent_match)
            parent = spec.parent
        return chain

    def show(self, match: RouteMatch) -> bool:
        """Show a whitelisted route. Returns False when nothing changed."""
        presentation = match.presentation
        if presentation == HANDLED:
            return False
        if presentation == SHEET:
            self.show_sheet(match)
            return True
        self.overlays = []
        if presentation == ROOT:
            if not self.stack and match.route == self.current.route:
                self._install_views()
                return False
            for entry in self.stack:
                entry.screen.dispose()
            self.stack = []
            self.current = match
            self._install_views()
            return True
        wanted = self._chain(match)
        existing = {entry.route: entry for entry in self.stack}
        new_stack: list[StackEntry] = []
        for item in wanted:
            entry = existing.pop(item.route, None)
            if entry is None:
                entry = StackEntry(item, self.screen_factory(item))
            new_stack.append(entry)
        for leftover in existing.values():
            leftover.screen.dispose()
        changed = [e.route for e in new_stack] != [e.route for e in self.stack]
        self.stack = new_stack
        self._install_views()
        for entry in new_stack:
            entry.screen.did_show()
        return changed

    def show_sheet(self, match: RouteMatch) -> InfoSheet:
        spec = match.spec
        sheet = InfoSheet(title=spec.title, body=f"This sheet arrives in {spec.milestone}.")
        self.sheets_shown.append(sheet)
        sheet.show(self.page)
        return sheet

    def pop(self) -> str:
        """Pop the top screen (or overlay); returns the route now on screen."""
        if self.overlays:
            self.overlays.pop()
        elif self.stack:
            self.stack.pop().screen.dispose()
        self._install_views()
        return self.current_route

    def pop_view(self, view: Any) -> str:
        """``page.on_view_pop``: drop ``view`` (and anything above it)."""
        if view is not None and view in self.overlays:
            index = self.overlays.index(view)
            del self.overlays[index:]
        else:
            for index, entry in enumerate(self.stack):
                if entry.view is not None and entry.view is view:
                    for removed in self.stack[index:]:
                        removed.screen.dispose()
                    del self.stack[index:]
                    break
            else:
                if self.overlays:
                    self.overlays.pop()
                elif self.stack:
                    self.stack.pop().screen.dispose()
        self._install_views()
        return self.current_route

    def push_overlay(self, view: ft.View) -> None:
        self.overlays.append(view)
        self.page.views.append(view)

    def _on_tablet_back(self, e: Any = None) -> None:
        if self.on_back is not None:
            self.on_back()
        else:
            self.pop()
            self.update()

    # ---- drawer --------------------------------------------------------------------------

    async def open_drawer(self) -> None:
        if self.tablet or self.root_view is None or self.root_view.drawer is None:
            return
        await self.root_view.show_drawer()

    async def close_drawer(self) -> None:
        if self.tablet or self.root_view is None or self.root_view.drawer is None:
            return
        try:
            await self.root_view.close_drawer()
        except Exception as exc:  # drawer already closed / no client
            log.debug("close_drawer: %s", exc)

    # ---- resizing ------------------------------------------------------------------------

    def apply_width(self, width: Optional[float], height: Optional[float] = None) -> bool:
        """``page.on_resize``: rebuild only when the size class changes."""
        new = layout_for(width or 0, self.state.text_scale.value)
        old = self.layout
        if self.drawer_box is not None:
            self.drawer_box.height = self._drawer_height()
        if new.size_class is old.size_class:
            self.layout = new
            if self.nav_drawer is not None:
                self.nav_drawer.width = new.drawer_width
            return False
        self.layout = new
        self.state.size_class.set(new.size_class)
        if new.persistent_sidebar != old.persistent_sidebar:
            self._build()
            self._install_views()
        else:
            self.chat_view.apply_layout(new)
            if self.tablet:
                self.sidebar.width = new.sidebar_width
                self.side_panel.set_pin_available(new.size_class is SizeClass.WIDE)
                self.main_area.content = self._main_content()
            else:
                self.nav_drawer.width = new.drawer_width
                self.root_view.controls = [self.chat_view.build_body(new)]
        return True

    def update(self) -> None:
        try:
            self.page.update()
        except Exception as exc:
            log.debug("page.update failed: %s", exc)
