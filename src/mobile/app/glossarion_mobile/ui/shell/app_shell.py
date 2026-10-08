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

U9 (UI_SPEC §1.1, §7.5):

* The shell is the page's ``components.surface`` host: on tablets chat settings, job
  detail, compare and the term sheet open in the right SidePanel (``present`` /
  ``open_screen_in_panel``); an unpinned panel closes on navigation. While it is open the
  chat column, and so the composer's output-mode style, is computed without its 380 dp.
* The layout follows the *effective* text scale (Appearance × the system font scale the
  ``ui.text_scale`` probe measures) and is re-applied when either changes, so the >= 160 %
  rules (pills → "Options (n)", the mode chip, the model-only header) engage at 200 %
  system text.
* A size-class change calls ``Screen.apply_size_class`` on every open screen (master-detail
  at >= 1200 dp: Settings, Book page, Manga).
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import flet as ft

from glossarion_mobile.state.app_state import AppState, JobStripModel
from glossarion_mobile.ui import text_scale as ts
from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.chat.chat_view import ChatView
from glossarion_mobile.ui.components import surface
from glossarion_mobile.ui.components.info_sheet import InfoSheet
from glossarion_mobile.ui.responsive import Layout, SizeClass, layout_for
from glossarion_mobile.ui.router import FULLSCREEN, HANDLED, ROOT, ROUTES_BY_NAME, SHEET, RouteMatch, parse_route
from glossarion_mobile.ui.screens.base import SHIPPED_MILESTONES, Screen, build_screen_view
from glossarion_mobile.ui.shell.drawer import ChatDrawer
from glossarion_mobile.ui.shell.job_strip import JobStrip
from glossarion_mobile.ui.shell.side_panel import SidePanel
from glossarion_mobile.ui.theme import HIT_TARGET

__all__ = ["AppShell", "StackEntry"]

log = logging.getLogger("glossarion.shell")

#: The phone drawer box never gets shorter than this (header, search, chips, footer and a few chat rows).
_DRAWER_MIN_HEIGHT = 300.0

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
        self.layout: Layout = layout_for(getattr(page, "width", None) or 0, ts.effective(state))
        self.current: RouteMatch = parse_route("/")  # the chat-root route ("/" or "/chat/<cid>")
        self.stack: list[StackEntry] = []
        self.overlays: list[ft.View] = []
        self.side_panel = SidePanel(on_change=self._on_side_panel_change)
        self.panel_screen: Optional[Screen] = None  # a screen shown in the SidePanel (job detail)
        self.text_probe = ts.TextScaleProbe()
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
        surface.register(page, self)

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

    def _system_insets(self) -> tuple[float, float]:
        """(top, bottom) system insets from ``page.media``: the larger of ``padding`` and
        ``view_padding`` on each side (``view_padding`` keeps the navigation bar while the keyboard is
        up, so the result does not change when the keyboard opens)."""
        media = getattr(self.page, "media", None)

        def side(name: str) -> float:
            values = [0.0]
            for attr in ("padding", "view_padding"):
                try:
                    values.append(float(getattr(getattr(media, attr, None), name, 0) or 0))
                except (TypeError, ValueError):
                    pass
            return max(values)

        return side("top"), side("bottom")

    def _drawer_height(self) -> float:
        """Height of the phone drawer's content box (``drawer_box``).

        Flutter's ``NavigationDrawer`` puts its children in its own ``ListView`` under
        ``SafeArea(bottom: false)``: that list is the screen height minus the top inset (status bar)
        and pads its end by the bottom inset (the navigation bar / gesture bar). A box of
        ``height - top`` (the U1 size) made that list one bottom inset taller than its viewport, so a
        drag on the drawer scrolled the whole content (footer included) and the footer sat under the
        navigation bar (owner request 17: the footer must stay pinned; only the chat list scrolls).
        The box is now exactly the list's viewport minus that end padding: the outer list cannot
        scroll, ``ChatDrawer.body`` is the only scrolling part and the footer sits just above the
        navigation bar. Below ``_DRAWER_MIN_HEIGHT`` (split screen) the content cannot fit anyway and
        the drawer's own list scrolls it as a whole, so nothing becomes unreachable.
        """
        height = getattr(self.page, "height", None) or 640
        top, bottom = self._system_insets()
        return max(_DRAWER_MIN_HEIGHT, float(height) - top - bottom)

    def apply_insets(self, e: Any = None) -> bool:
        """``page.on_media_change`` (system bars, navigation mode, rotation): re-fit the phone drawer
        box. True when its height changed (the keyboard alone never changes it)."""
        box = self.drawer_box
        if box is None:
            return False
        height = self._drawer_height()
        if box.height == height:
            return False
        box.height = height
        try:
            box.update()
        except Exception as exc:  # not mounted yet / no client
            log.debug("drawer box update failed: %s", exc)
        return True

    def _install_media_hook(self) -> None:
        """Follow ``page.on_media_change`` (chaining a handler someone else installed)."""
        previous = getattr(self.page, "on_media_change", None)
        if getattr(previous, "_glossarion_shell", None) is self:
            return

        def on_media_change(e: Any = None) -> None:
            self.apply_insets(e)
            if callable(previous):
                result = previous(e)
                if inspect.isawaitable(result):
                    asyncio.ensure_future(result)

        on_media_change._glossarion_shell = self  # type: ignore[attr-defined]
        try:
            self.page.on_media_change = on_media_change
        except Exception as exc:  # a page without media events (host fakes)
            log.debug("on_media_change not available: %s", exc)

    def mount(self) -> None:
        """Build the layout for the current width and install ``page.views`` (plus the invisible
        system text-scale probe in ``page.overlay``)."""
        self.text_probe.install(self.page)
        self._install_media_hook()
        self._build()
        self._install_views()

    def attach(self) -> None:
        if not self._unsubs:
            self._unsubs = [
                self.state.job_strip.subscribe(self._on_job_strip),
                self.state.text_scale.subscribe(lambda _value: self.apply_text_scale()),
                ts.subscribe(lambda _value: self.apply_text_scale()),
            ]
        self.drawer.attach()
        self.chat_view.attach()

    def detach(self) -> None:
        for unsub in self._unsubs:
            unsub()
        self._unsubs = []
        self.side_panel.close()
        self.drawer.detach()
        self.chat_view.detach()
        for entry in self.stack:
            entry.screen.dispose()
        surface.unregister(self.page, self)

    def _effective_scale(self) -> float:
        """Appearance text scale × the measured system font scale (``ui.text_scale``)."""
        return ts.effective(self.state)

    def chat_layout(self) -> Layout:
        """The layout the chat column gets: the window layout, narrowed by the SidePanel while it
        is open on a tablet (UI_SPEC §2.3 width rules follow the chat column)."""
        if not (self.tablet and self.side_panel.is_open):
            return self.layout
        return layout_for(self.layout.width, self.layout.text_scale, side_panel=True)

    def _build(self) -> None:
        self.builds += 1
        layout = self.layout
        self.chat_view.apply_layout(self.chat_layout())
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
                    ft.Container(content=self.chat_view.build_body(self.chat_layout(), self._main_available_width()),
                                 expand=True),
                ],
                spacing=0,
                expand=True,
            )
        self.tablet_title = ft.Text(
            entry.screen.title, theme_style=ft.TextThemeStyle.TITLE_MEDIUM, weight=ft.FontWeight.W_600, expand=True
        )
        entry.screen.app_bar_title = self.tablet_title  # a screen that renames itself updates it
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
            self._sync_root_back()
        else:
            pushed = list(self.stack)
        for entry in pushed:
            views.append(self._entry_view(entry))
        views.extend(self.overlays)
        self.page.views.clear()
        self.page.views.extend(views)
        self._place_strip()

    def _sync_root_back(self) -> None:
        """Tablet: the system / gesture back on the one root View pops the main-area stack.

        While the main area shows a screen the root View cannot pop (``can_pop=False``), so the
        back reaches ``on_confirm_pop``: the screen's ``handle_back`` first (leave selection mode,
        UI_SPEC §1.6 rule 2), else the main-area stack pops (rule 5). With the chat in the main
        area the root can pop and the system default applies (rule 6: leave the app). An open
        SidePanel is the tablet form of a sheet, so back closes it first (rule 1).
        """
        root = self.root_view
        if root is None:
            return
        root.can_pop = self._top_panel_entry() is None and not self.side_panel.is_open
        root.on_confirm_pop = self._on_root_confirm_pop

    async def _on_root_confirm_pop(self, e: Any = None) -> None:
        view = getattr(e, "control", None) or self.root_view
        leave = self._top_panel_entry() is None and not self.side_panel.is_open
        if self.side_panel.is_open:
            self.side_panel.close()  # rule 1: the panel (a sheet on phones) closes first
        elif not leave:
            try:
                self._on_tablet_back()
            except Exception:
                log.exception("tablet back failed")
        try:
            # Never let Flutter pop the only View while a screen was showing (that leaves the app).
            await view.confirm_pop(leave)
        except Exception as exc:  # no client (host tests) / the View was replaced
            log.debug("confirm_pop failed: %s", exc)

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

    def _push_base(self, match: RouteMatch, wanted: list[RouteMatch], in_app: bool = False) -> Optional[list[StackEntry]]:
        """The stack entries a pushed route keeps below it, or None to use its static chain.

        A full-screen surface (Reader, metadata editor) is pushed on top of the current
        stack, and so is a route with static parents opened in the app (``in_app``: Scan for
        raw, Files, a job's detail, a settings page), so Back returns to the screen it was
        opened from instead of the route's static parents (UI_SPEC §1.2 and §1.6 rule 5: a
        pushed View pops; §7.2: a quick look returns to context). A link from outside the app
        (deep link, notification) goes on top only when its static parents are already on
        the stack. Re-opening a screen already on the stack -- the same route, or another
        instance of it such as a second book -- returns to that depth first. The static chain
        stays for an empty stack (cold deep links, the chat root), for top-level destinations
        without parents (Library, Jobs, Glossaries, Tools, Settings) and for drawer
        navigation (``show(reset=True)``).
        """
        if not self.stack:
            return None
        if match.presentation != FULLSCREEN:
            if len(wanted) < 2:
                return None
            if not in_app:
                current = {entry.route for entry in self.stack}
                if not all(item.route in current for item in wanted[:-1]):
                    return None
        below: list[StackEntry] = []
        for entry in self.stack:
            if entry.route == match.route or entry.match.name == match.name:
                break
            below.append(entry)
        return below

    def _plan(self, match: RouteMatch, reset: bool, in_app: bool) -> tuple[list[RouteMatch], dict[str, StackEntry]]:
        """The routes of the stack after showing ``match`` (a non-root route) and the current entries
        by route, which ``show`` reuses for those routes and disposes otherwise."""
        wanted = self._chain(match)
        below = None if reset else self._push_base(match, wanted, in_app)
        if below is not None:
            # Pushed on top of the screens the user came from (UI_SPEC §1.6 rule 5).
            existing = {entry.route: entry for entry in self.stack[len(below):]}
            wanted = [entry.match for entry in below] + [match]
            existing.update({entry.route: entry for entry in below})
        else:
            existing = {entry.route: entry for entry in self.stack}
        return wanted, existing

    def leaving_entries(self, match: RouteMatch, *, reset: bool = False, in_app: bool = False) -> list[StackEntry]:
        """The stack entries ``show(match, reset=, in_app=)`` would dispose (nothing changes): the app asks
        their screens' ``confirm_leave`` first (unsaved edits)."""
        presentation = match.presentation
        if presentation in (HANDLED, SHEET):
            return []
        if presentation == ROOT:
            if not self.stack and match.route == self.current.route:
                return []
            return list(self.stack)
        wanted, existing = self._plan(match, reset, in_app)
        for item in wanted:
            existing.pop(item.route, None)
        return list(existing.values())

    def entries_above(self, view: Any) -> list[StackEntry]:
        """The entries above the one whose View is ``view`` (tablet: main-area screens hidden behind a
        full-screen View); ``pop_view(view)`` disposes them with it."""
        for index, entry in enumerate(self.stack):
            if entry.view is not None and entry.view is view:
                return list(self.stack[index + 1:])
        return []

    def show(self, match: RouteMatch, *, reset: bool = False, in_app: bool = False) -> bool:
        """Show a whitelisted route. Returns False when nothing changed.

        ``in_app``: opened from a screen of the app, so it goes on top of the current screens
        (``_push_base``). ``reset`` (drawer navigation) rebuilds the stack from the route's
        static parents instead.
        """
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
            self._close_unpinned_panel()
            for entry in self.stack:
                entry.screen.dispose()
            self.stack = []
            self.current = match
            self._install_views()
            return True
        wanted, existing = self._plan(match, reset, in_app)
        if presentation != FULLSCREEN:
            self._close_unpinned_panel()
        new_stack: list[StackEntry] = []
        created: list[StackEntry] = []
        for item in wanted:
            entry = existing.pop(item.route, None)
            if entry is None:
                entry = StackEntry(item, self.screen_factory(item))
                created.append(entry)
            new_stack.append(entry)
        for leftover in existing.values():
            leftover.screen.dispose()
        changed = [e.route for e in new_stack] != [e.route for e in self.stack]
        self.stack = new_stack
        self._install_views()
        for entry in new_stack:
            # A screen kept below the new top was shown before and keeps its own refresh;
            # showing it again would reload a hidden page (the Book page under the Reader).
            if entry is new_stack[-1] or any(entry is new for new in created):
                entry.screen.did_show()
        return changed

    def show_sheet(self, match: RouteMatch) -> InfoSheet:
        """A ``sheet`` route nothing handled (a deep link to it): say where it lives."""
        spec = match.spec
        if spec.milestone in SHIPPED_MILESTONES:
            body = f"{spec.title} opens from its screen; this link cannot show it on its own."
        else:
            body = f"This sheet arrives in {spec.milestone}."
        sheet = InfoSheet(title=spec.title, body=body)
        self.sheets_shown.append(sheet)
        sheet.show(self.page)
        return sheet

    def pop(self) -> str:
        """Pop the top screen (or overlay); returns the route now on screen."""
        if self.overlays:
            self.overlays.pop()
        elif self.stack:
            if not self.stack[-1].fullscreen:
                self._close_unpinned_panel()
            self.stack.pop().screen.dispose()
        self._install_views()
        return self.current_route

    def pop_view(self, view: Any, *, keep_above: bool = False) -> str:
        """``page.on_view_pop``: drop ``view`` (and anything above it). ``keep_above``: drop only that
        View's entry; the entries above it stay (a screen there kept its unsaved edits)."""
        if view is not None and view in self.overlays:
            index = self.overlays.index(view)
            del self.overlays[index:]
        else:
            for index, entry in enumerate(self.stack):
                if entry.view is not None and entry.view is view:
                    end = index + 1 if keep_above else len(self.stack)
                    for removed in self.stack[index:end]:
                        removed.screen.dispose()
                    del self.stack[index:end]
                    break
            else:
                if self.overlays:
                    self.overlays.pop()
                elif self.stack:
                    self.stack.pop().screen.dispose()
        self._install_views()
        return self.current_route

    def push_overlay(self, view: ft.View) -> None:
        """Push a non-routable full-screen View on top (device checks, the keyword delete view,
        a Model manager sub-screen).

        Its ``route`` must differ from every other View's: the client keys Views by route and
        Flet resolves ``view_pop`` to the first View with the popped route, so an overlay that
        reused a screen's route would make Android back pop that screen and leave the overlay.
        A colliding route gets an ``/overlay-<n>`` suffix.
        """
        taken = {getattr(v, "route", None) for v in self.page.views if v is not view}
        taken.update(entry.route for entry in self.stack)
        taken.update(getattr(v, "route", None) for v in self.overlays if v is not view)
        route = getattr(view, "route", None) or "/"
        if route in taken:
            base = route.rstrip("/")
            index = 1
            while f"{base}/overlay-{index}" in taken:
                index += 1
            view.route = f"{base}/overlay-{index}"
            log.debug("overlay route %r is taken; using %r", route, view.route)
        self.overlays.append(view)
        self.page.views.append(view)

    def _on_tablet_back(self, e: Any = None) -> None:
        entry = self._top_panel_entry()
        handler = getattr(entry.screen, "handle_back", None) if entry is not None else None
        if callable(handler):
            try:
                if handler():  # e.g. leave selection mode first (UI_SPEC §1.6 rule 2)
                    return
            except Exception:
                log.exception("back handler failed")
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

    # ---- side panel (components.surface host) ----------------------------------------------

    def present(self, content: ft.Control, *, title: str, owner: Any = None,
                on_close: Optional[Callable[[], Any]] = None) -> bool:
        """Show ``content`` in the SidePanel (tablets only); False on phones."""
        if not self.tablet:
            return False
        self.side_panel.open(title, content, owner=owner, on_close=on_close)
        return True

    def hosts(self, owner: Any) -> bool:
        return self.side_panel.hosts(owner)

    def dismiss(self, owner: Any) -> bool:
        return self.side_panel.dismiss(owner)

    def open_screen_in_panel(self, match: RouteMatch) -> bool:
        """A route's screen in the SidePanel (tablets: job detail from the JobStrip, UI_SPEC §1.7);
        the screen is disposed when the panel closes or shows something else."""
        if not self.tablet:
            return False
        screen = self.screen_factory(match)
        body = screen.get_body()
        actions = list(screen.actions() or [])
        content: ft.Control = body
        if actions:
            content = ft.Column([ft.Row(actions, alignment=ft.MainAxisAlignment.END, spacing=0, wrap=True),
                                 ft.Container(content=body, expand=True)], spacing=0, expand=True)

        def release(s: Screen = screen) -> None:
            if self.panel_screen is s:
                self.panel_screen = None
            s.dispose()

        self.side_panel.open(screen.title or match.spec.title, content, owner=screen, on_close=release)
        self.panel_screen = screen
        screen.did_show()
        return True

    def _close_unpinned_panel(self) -> None:
        if self.side_panel.is_open and not self.side_panel.pinned:
            self.side_panel.close()

    def _on_side_panel_change(self, is_open: bool) -> None:
        """The panel opened or closed: the main area is 380 dp narrower / wider."""
        if not self.tablet or self.main_area is None:
            return
        self.chat_view.apply_layout(self.chat_layout())
        self._sync_root_back()  # back closes an open panel first (UI_SPEC §1.6 rule 1)
        if self._top_panel_entry() is None:  # the chat is in the main area: re-centre its column
            self.main_area.content = self._main_content()
            try:
                self.main_area.update()
            except Exception:
                pass
        if self.root_view is not None:
            try:
                self.root_view.update()
            except Exception:
                pass

    # ---- resizing ------------------------------------------------------------------------

    def apply_width(self, width: Optional[float], height: Optional[float] = None) -> bool:
        """``page.on_resize``: rebuild only when the size class changes."""
        return self._apply_layout(layout_for(width or 0, self._effective_scale()))

    def apply_text_scale(self) -> bool:
        """The Appearance text size or the system font scale changed: re-apply the >= 160 % rules
        (composer, header) without a rebuild; True when the compact state changed."""
        old = self.layout
        new = layout_for(old.width, self._effective_scale())
        if new == old:
            return False
        self._apply_layout(new)
        self.update()  # the header height follows the scale even when the compact state does not change
        return (new.compact_text, new.output_row) != (old.compact_text, old.output_row)

    def _apply_layout(self, new: Layout) -> bool:
        old = self.layout
        if self.drawer_box is not None:
            self.drawer_box.height = self._drawer_height()
        if new.size_class is old.size_class:
            self.layout = new
            if self.nav_drawer is not None:
                self.nav_drawer.width = new.drawer_width
            chat = self.chat_layout()
            current = self.chat_view.layout
            if (chat.output_row, chat.compact_text, chat.text_scale) != (current.output_row, current.compact_text,
                                                                         current.text_scale):
                # the composer's output-mode control follows the chat column, which also
                # changes inside a class on tablets (UI_SPEC §2.3); no shell rebuild
                self.chat_view.apply_layout(chat)
            return False
        self.layout = new
        if not new.persistent_sidebar:
            self.side_panel.close()  # phones have no SidePanel: its owner's on_close runs
        self.state.size_class.set(new.size_class)
        if new.persistent_sidebar != old.persistent_sidebar:
            self._build()
            self._install_views()
        else:
            self.chat_view.apply_layout(self.chat_layout())
            if self.tablet:
                self.sidebar.width = new.sidebar_width
                self.side_panel.set_pin_available(new.size_class is SizeClass.WIDE)
                if new.size_class is not SizeClass.WIDE and self.side_panel.pinned:
                    self.side_panel.pinned = False
                self.main_area.content = self._main_content()
            else:
                self.nav_drawer.width = new.drawer_width
                self.root_view.controls = [self.chat_view.build_body(new)]
        self._notify_size_class(new.size_class)
        return True

    def _notify_size_class(self, size_class: SizeClass) -> None:
        """Open screens re-arrange for the new class (``Screen.apply_size_class``: master-detail)."""
        screens = [entry.screen for entry in self.stack]
        if self.panel_screen is not None:
            screens.append(self.panel_screen)
        for screen in screens:
            handler = getattr(screen, "apply_size_class", None)
            if callable(handler):
                try:
                    handler(size_class)
                except Exception:
                    log.exception("%s.apply_size_class failed", type(screen).__name__)

    def update(self) -> None:
        try:
            self.page.update()
        except Exception as exc:
            log.debug("page.update failed: %s", exc)
