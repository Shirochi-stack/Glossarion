"""SettingsContext: what every settings screen and tile needs, in one object.

Built by ``integration.SettingsFeature``; screens receive it instead of the
whole app so they stay testable on a fake page. All UI mutation happens on the
loop thread: store observers fire on the thread that changed the value, so
``on_ui`` re-posts through ``UiDispatcher`` when called from a worker.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from glossarion_mobile.state.config_store import MobileConfigStore
from glossarion_mobile.ui.router import RouteError, build_route
from glossarion_mobile.ui.settings.schema_access import SchemaAccess

__all__ = ["SettingsContext"]

log = logging.getLogger("glossarion.settings")


@dataclass
class SettingsContext:
    page: Any
    store: MobileConfigStore
    schema: SchemaAccess
    dispatcher: Any = None  # UiDispatcher
    navigate_route: Optional[Callable[[str], Any]] = None  # route string -> awaitable/None (app.navigate)
    navigate_name: Optional[Callable[..., Any]] = None  # (name, params=None, query=None) (app.navigate_to)
    notify: Optional[Callable[..., Any]] = None  # snackbar
    prefs: Any = None
    tablet: bool = False
    file_picker_factory: Optional[Callable[[], Any]] = None
    extras: dict = field(default_factory=dict)

    # ---- threading ------------------------------------------------------------------------

    def on_ui(self, fn: Callable[..., Any], *args: Any) -> None:
        """Run ``fn(*args)`` on the UI loop (directly when already there)."""
        dispatcher = self.dispatcher
        if dispatcher is None or not getattr(dispatcher, "bound", False) or dispatcher.on_loop_thread():
            fn(*args)
            return
        dispatcher.post(fn, *args)

    async def run_io(self, fn: Callable[..., Any], *args: Any) -> Any:
        """Blocking work off the loop (``UiDispatcher.run_in_thread``; ``asyncio.to_thread`` without one)."""
        dispatcher = self.dispatcher
        if dispatcher is not None and getattr(dispatcher, "bound", False):
            return await dispatcher.run_in_thread(fn, *args, name="gl-settings-io")
        return await asyncio.to_thread(fn, *args)

    def spawn(self, coro: Any) -> Any:
        dispatcher = self.dispatcher
        if dispatcher is not None and getattr(dispatcher, "bound", False):
            return dispatcher.spawn(coro)
        return asyncio.ensure_future(coro)

    @staticmethod
    def push(*controls: Any) -> None:
        """``control.update()`` for mounted controls (no-op before mounting)."""
        for control in controls:
            if control is None:
                continue
            try:
                control.update()
            except Exception:
                pass

    # ---- feedback / navigation ---------------------------------------------------------

    def say(self, message: str, action_label: Optional[str] = None, on_action: Any = None) -> None:
        if self.notify is None:
            log.info("settings: %s", message)
            return
        try:
            self.notify(message, action_label, on_action)
        except TypeError:
            self.notify(message)

    def show_dialog(self, dialog: Any) -> None:
        page = self.page
        if page is not None:
            page.show_dialog(dialog)

    def pop_dialog(self, dialog: Any = None) -> None:
        """Close ``dialog`` itself (``close_dialog``); without one, the topmost open dialog."""
        page = self.page
        if page is None:
            return
        if dialog is not None:
            from glossarion_mobile.ui.components.dialogs import close_dialog

            close_dialog(page, dialog)
            return
        try:
            page.pop_dialog()
        except Exception:
            pass

    def go(self, route_name: str, params: Optional[dict] = None, *, fragment: Optional[str] = None) -> Optional[str]:
        """Navigate to a whitelisted route (with an optional ``#fragment``); returns the route."""
        try:
            route = build_route(route_name, params, None, fragment)
        except RouteError:
            if fragment is None:
                log.error("bad settings route %s %s", route_name, params)
                return None
            try:
                route = build_route(route_name, params)
            except RouteError:
                log.error("bad settings route %s %s", route_name, params)
                return None
        if self.navigate_route is not None:
            result = self.navigate_route(route)
            if asyncio.iscoroutine(result):
                self.spawn(result)
        elif self.navigate_name is not None:
            self.navigate_name(route_name, params)
        return route

    def open_setting(self, section_id: str, key: Optional[str] = None) -> Optional[str]:
        schema = getattr(self, "schema", None)
        resolve = getattr(schema, "resolve_section_id", None)
        if callable(resolve):  # the curated map may show the key in another section (U9)
            try:
                section_id = resolve(section_id, key) or section_id
            except Exception:
                pass
        return self.go("settings.section", {"section": section_id}, fragment=key)
