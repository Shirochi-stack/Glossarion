"""The Glossarion mobile app: ``main(page)`` builds the chat-first shell.

Order inside ``GlossarionApp.start`` (``runtime_bootstrap.bootstrap()`` already
ran in ``app/main.py`` before Flet was imported):

1. bind the UiDispatcher (and the loop guard) to the Flet loop; create the
   page services and keep strong references (Flet unregisters unreferenced
   services): GlossarionNative bridge, UrlLauncher, Clipboard, HapticFeedback,
   SecureStorage;
2. theme (Halgakos Rose, compact density), chat view, drawer, AppShell;
   ``page.update()``;
3. register the in-app browser opener for ``webbrowser.open`` (OAuth);
4. print ``GLOSSARION_READY`` (CI contract, ``ci/android_smoke.sh``);
5. start the dispatcher pump; read (or create) the API-key and token
   encryption keys in SecureStorage and pass them to
   ``api_key_encryption.set_key_material`` / ``token_encryption.set_symmetric_key``
   (``services.secure_keys``), *then* start the backend warm import (prints
   ``GLOSSARION_BACKEND_READY``; Send stays "Preparing engine…" until then);
6. dispatch the initial route (a cold-start ``/__selftest__`` deep link runs
   the self-test, which prints ``GLOSSARION_SELFTEST PASS|FAIL``).

Handled routes (``/__selftest__``, ``/oauth/return``) never push a View; after
handling them the client route is put back to what is on screen, so the same
deep link can fire again (Flet drops a route event equal to the last one).
Every route the app pushes to the client goes through ``_push_client_route``,
which remembers it so the client's ``route_change`` echo is not mistaken for a
navigation (that would close overlays such as Device checks).
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Optional

import flet as ft

from glossarion_mobile import runtime_bootstrap as rb
from glossarion_mobile.services import secure_keys
from glossarion_mobile.services.browser import InAppUrlOpener
from glossarion_mobile.services.diagnostics import SelfTestRunner
from glossarion_mobile.services.dispatcher import LogBufferHandler, UiDispatcher
from glossarion_mobile.services.haptics import Haptics
from glossarion_mobile.services.native import NativeBridge
from glossarion_mobile.state.app_state import AppState
from glossarion_mobile.state.chat_index import ChatSummary
from glossarion_mobile.ui.chat.chat_view import ChatView
from glossarion_mobile.ui.components.action_sheet import ActionItem, ActionSheet
from glossarion_mobile.ui.components.dialogs import show_snackbar
from glossarion_mobile.ui.router import HANDLED, RouteError, RouteMatch, Router, build_route, launch_links, parse_route
from glossarion_mobile.ui.screens.base import HubScreen, PlaceholderScreen, Screen
from glossarion_mobile.ui.screens.diagnostics import DiagnosticsScreen
from glossarion_mobile.ui.shell.app_shell import AppShell
from glossarion_mobile.ui.shell.drawer import ChatDrawer
from glossarion_mobile.ui.theme import Appearance, apply_theme, is_dark

__all__ = ["GlossarionApp", "main"]

log = logging.getLogger("glossarion.app")

_HUB_INTROS = {
    "settings": "Settings mirror the desktop Other Settings. Pages arrive milestone by milestone; "
    "your config.json values are kept untouched until then.",
    "tools": "Every desktop tool has a place here; each one opens once its milestone ships.",
}

_LOG_HANDLER: Optional[LogBufferHandler] = None

# How long a route pushed to the client waits for its route_change echo.
_ROUTE_ECHO_WINDOW = 10.0


def _install_log_handler(buffer: Any) -> None:
    """Route ``glossarion.*`` log records into the app LogBuffer (one handler per process)."""
    global _LOG_HANDLER
    logger = logging.getLogger("glossarion")
    if _LOG_HANDLER is None:
        _LOG_HANDLER = LogBufferHandler(buffer)
        logger.addHandler(_LOG_HANDLER)
    else:
        _LOG_HANDLER.buffer = buffer
    if logger.level == logging.NOTSET or logger.level > logging.INFO:
        logger.setLevel(logging.INFO)


class GlossarionApp:
    def __init__(self, page: ft.Page) -> None:
        self.page = page
        self.boot = rb.get_state()
        self.paths = rb.get_paths()
        self.dispatcher = UiDispatcher(page).bind()
        self.state = AppState()
        self.router = Router()
        platform = getattr(getattr(page, "platform", None), "value", None)
        self.is_mobile = platform in ("android", "ios") and not getattr(page, "web", False)

        # Page services: keep strong references (Flet unregisters unreferenced services).
        self.native = NativeBridge(page)  # GlossarionNative on Android/iOS, stub elsewhere
        self.url_launcher = ft.UrlLauncher()
        self.clipboard = ft.Clipboard()
        self.haptics = Haptics(ft.HapticFeedback())
        self.secure_storage = self._make_secure_storage()
        self.opener = InAppUrlOpener(self.dispatcher, self.url_launcher, is_mobile=self.is_mobile)
        self.native.add_listener("share", self._on_share)

        self.key_status: Optional[secure_keys.KeyStatus] = None
        self.keys_ready = asyncio.Event()  # set once the backend has its encryption keys
        self._route_echoes: list[tuple[str, float]] = []  # (route pushed to the client, deadline)

        self.log_buffer = self.dispatcher.log_buffer("main")
        self.selftest = SelfTestRunner(self.dispatcher, self.state, before_run=self.keys_ready.wait)
        self.lifecycle: list[str] = []
        self.initial_route: Optional[str] = None
        self.initial_shared: list = []
        self._routed_link_ids: set[str] = set()
        self.spike: Any = None
        self.spike_view: Optional[ft.View] = None
        self.ready_at: Optional[float] = None

        self.chat_view: Optional[ChatView] = None
        self.drawer: Optional[ChatDrawer] = None
        self.shell: Optional[AppShell] = None
        # Schema-driven Settings (U2); SettingsFeature.attach also sets config_store and prefs.
        self.settings: Any = None
        self.config_store: Any = None
        self.prefs: Any = None

    @staticmethod
    def _make_secure_storage() -> Any:
        try:
            from flet_secure_storage import SecureStorage
        except ImportError as exc:
            log.warning("flet_secure_storage unavailable (%s); encryption keys use the fallback file", exc)
            return None
        return SecureStorage()

    # ---- start ------------------------------------------------------------------------

    async def start(self) -> None:
        page = self.page
        page.title = "Glossarion"
        apply_theme(page, Appearance.SYSTEM)
        page.padding = 0
        if not self.is_mobile and not getattr(page, "web", False) and page.platform is not None:
            try:  # phone-sized window for desktop dev (flet run)
                page.window.width = 412
                page.window.height = 860
            except Exception:
                pass
        _install_log_handler(self.log_buffer)

        dark = is_dark(page)
        self.chat_view = ChatView(
            page,
            state=self.state,
            navigate=self.navigate_to,
            notify=self.notify,
            open_drawer=self.open_drawer,
            haptics=self.haptics,
        )
        self.drawer = ChatDrawer(
            state=self.state,
            on_navigate=self._drawer_navigate,
            on_open_chat=self._open_chat,
            on_chat_long_press=self._chat_actions,
            on_new_chat=lambda e: self.chat_view._on_new_chat(e),
            on_new_scratch=lambda e: self.notify("Scratch chats arrive in U3"),
            on_status=self._on_status_chip,
            on_settings=lambda e: self._drawer_navigate("settings"),
            on_help=self._on_help,
            dark=dark,
        )
        self.shell = AppShell(
            page,
            state=self.state,
            chat_view=self.chat_view,
            drawer=self.drawer,
            screen_factory=self.make_screen,
            on_back=self.back,
        )
        self.shell.global_strip.on_open = lambda: self.navigate_to("jobs")
        self.shell.global_strip.on_stop = lambda: self.notify("No job is running")
        self.state.width.set(float(getattr(page, "width", 0) or 0))
        self.state.size_class.set(self.shell.size_class)
        self.shell.mount()
        self.shell.attach()
        page.on_route_change = self.on_route_change
        page.on_view_pop = self.on_view_pop
        page.on_resize = self.on_resize
        page.on_app_lifecycle_state_change = self.on_lifecycle
        page.update()

        rb.set_url_opener(self.opener.open_threadsafe)
        self.initial_route = page.route
        self.ready_at = time.time()
        rb.emit_marker(rb.MARKER_READY)
        log.info("UI ready (%s, %s)", self.shell.size_class.value, getattr(page.platform, "value", page.platform))

        self.dispatcher.start()
        await self._install_backend_keys()  # before anything imports or uses the backend
        await self._install_settings()  # decrypts config.json with those keys; before the first route
        rb.start_warm_import(on_done=lambda result: self.dispatcher.post(self._on_backend_ready, result))
        self.dispatcher.spawn(self._after_ready())
        await self.dispatch_route(page.route, source="initial")

    async def _install_backend_keys(self) -> None:
        """SecureStorage keys -> ``set_key_material`` / ``set_symmetric_key`` (plan §4 bootstrap)."""
        try:
            fallback_dir = self.paths.data if self.paths is not None else Path.cwd()
            self.key_status = await secure_keys.setup(self.secure_storage, fallback_dir)
        finally:
            self.keys_ready.set()

    async def _install_settings(self) -> None:
        """MobileConfigStore + Prefs + the schema-driven settings screens (U2).

        Installed before the initial route is dispatched, so a cold-start ``/settings``
        deep link already gets the settings screens. If it fails the app keeps the hub.
        """
        try:
            from glossarion_mobile.ui.settings.integration import SettingsFeature

            await SettingsFeature.install(self)  # sets self.settings / config_store / prefs
        except Exception:
            log.exception("settings feature unavailable; /settings shows the hub")

    def _on_backend_ready(self, result: dict) -> None:
        self.state.backend.set(dict(result or {}))
        if result and not result.get("ok"):
            log.error("backend warm import failed: %s", list((result.get("failed") or {}).keys()))

    async def _after_ready(self) -> None:
        """Native probes that must not delay GLOSSARION_READY."""
        try:
            items = await self.native.initial_shared()
        except Exception as exc:
            log.warning("get_initial_shared failed: %s", exc)
            items = []
        self.initial_shared = items
        await self._route_launch_links(items)
        if any(not (isinstance(i, dict) and i.get("kind") == "url" and i.get("source") == "launch") for i in items):
            self.notify("Importing shared files arrives in U3")  # Open-with at cold start

    # ---- screens --------------------------------------------------------------------------

    def make_screen(self, match: RouteMatch) -> Screen:
        if match.name == "settings.logs":
            return DiagnosticsScreen(
                match,
                page=self.page,
                state=self.state,
                dispatcher=self.dispatcher,
                runner=self.selftest,
                log_buffer=self.log_buffer,
                open_device_checks=self.open_device_checks,
                copy_handler=self._copy_text,
                dark=is_dark(self.page),
            )
        if match.name in _HUB_INTROS:
            return HubScreen(match, navigate=self.navigate_to, intro=_HUB_INTROS[match.name])
        return PlaceholderScreen(match)

    # ---- routing ------------------------------------------------------------------------------

    async def on_route_change(self, e: Any) -> None:
        await self.dispatch_route(getattr(e, "route", None), source="event")

    async def dispatch_route(self, raw: Optional[str], *, source: str) -> Optional[RouteMatch]:
        match = self.router.handle(raw)
        self._spike_routes_changed()
        if match is None:
            log.info("ignored route %r (%s)", raw, source)
            return None
        if match.name == "selftest":
            self.dispatcher.spawn(self.selftest.run(match.get("suite") or "smoke", source=f"route/{source}"))
            self.dispatcher.spawn(self._restore_route())
            return match
        if match.name == "oauth.return":
            self._on_oauth_return(match)
            self.dispatcher.spawn(self._restore_route())
            return match
        if match.presentation == "sheet":
            self.shell.show_sheet(match)
            self.dispatcher.spawn(self._restore_route())
            return match
        if match.route == self.shell.current_route:
            if source == "event" and self._consume_route_echo(match.route):
                return match  # the client echoing a route the app pushed (overlays stay)
            if not self.shell.overlays:
                return match  # echo of an in-app navigation
        self.shell.show(match)
        self.page.update()
        return match

    def _consume_route_echo(self, route: str) -> bool:
        now = time.monotonic()
        self._route_echoes = [(r, deadline) for r, deadline in self._route_echoes if deadline > now]
        for index, (pushed, _deadline) in enumerate(self._route_echoes):
            if pushed == route:
                del self._route_echoes[index]
                return True
        return False

    async def _push_client_route(self, route: str) -> None:
        """``page.push_route(route)``; its ``route_change`` echo is then not treated as navigation."""
        if self.page.route == route:
            return
        entry = (route, time.monotonic() + _ROUTE_ECHO_WINDOW)
        self._route_echoes.append(entry)
        del self._route_echoes[:-8]
        try:
            await asyncio.wait_for(self.page.push_route(route), 10)
        except Exception as exc:
            if entry in self._route_echoes:
                self._route_echoes.remove(entry)
            log.info("push_route(%r) failed: %s", route, exc)

    async def _restore_route(self) -> None:
        await self._push_client_route(self.shell.current_route if self.shell is not None else "/")

    async def navigate(self, route: str) -> Optional[RouteMatch]:
        """In-app navigation to a route string: apply it now, then sync the client route."""
        match = await self.dispatch_route(route, source="app")
        if match is not None and match.presentation not in (HANDLED, "sheet"):
            await self.close_drawer()
            await self._push_client_route(self.shell.current_route)
        return match

    def navigate_to(self, route_name: str, params: Optional[dict] = None, query: Optional[dict] = None) -> None:
        """Navigate by route name (the only way UI code builds routes)."""
        try:
            route = build_route(route_name, params, query)
        except RouteError as exc:
            log.error("bad in-app route %s: %s", route_name, exc)
            return
        self.dispatcher.spawn(self.navigate(route))

    def _drawer_navigate(self, route_name: str) -> None:
        self.navigate_to(route_name)

    async def on_view_pop(self, e: Any) -> None:
        route = self.shell.pop_view(getattr(e, "view", None))
        self.page.update()
        await self._sync_route(route)

    def back(self) -> None:
        """Tablet main-area back button."""
        route = self.shell.pop()
        self.page.update()
        self.dispatcher.spawn(self._sync_route(route))

    async def _sync_route(self, route: str) -> None:
        await self._push_client_route(route)

    async def _route_launch_links(self, items: Any) -> bool:
        routed = False
        for item_id, link in launch_links(items):
            if item_id in self._routed_link_ids:
                continue
            self._routed_link_ids.add(item_id)
            # De-duplicate against page.route: Flutter may have delivered it too.
            initial = parse_route(self.initial_route) if self.initial_route else None
            link_match = parse_route(link)
            if initial is not None and link_match is not None and initial.path == link_match.path != "/":
                continue
            await self.dispatch_route(link, source="launch-link")
            routed = True
        return routed

    # ---- resize / lifecycle -----------------------------------------------------------

    def on_resize(self, e: Any) -> None:
        width = getattr(e, "width", None) or getattr(self.page, "width", None)
        self.state.width.set(float(width or 0))
        if self.shell.apply_width(width, getattr(e, "height", None)):
            log.info("size class -> %s", self.shell.size_class.value)
        self.page.update()

    async def on_lifecycle(self, e: Any) -> None:
        state = getattr(e, "state", None)
        name = getattr(state, "value", str(state))
        if name in ("inactive", "hide", "pause", "detach"):
            rb.flush_logs()
        self.lifecycle.append(f"{time.strftime('%H:%M:%S')} {name}")
        del self.lifecycle[:-12]
        log.info("lifecycle: %s", name)
        if self.spike is not None:
            await self.spike.on_lifecycle(e)

    # ---- drawer / chats ---------------------------------------------------------------------

    async def open_drawer(self) -> None:
        await self.shell.open_drawer()

    async def close_drawer(self) -> None:
        await self.shell.close_drawer()

    def _open_chat(self, cid: str) -> None:
        self.state.current_chat.set(cid)
        self.navigate_to("home" if cid == "1" else "chat", None if cid == "1" else {"cid": cid})

    def _chat_actions(self, chat: ChatSummary) -> ActionSheet:
        def pin() -> None:
            self.state.chats.set_pinned(chat.cid, not chat.pinned)

        later = lambda what: (lambda: self.notify(f"{what} arrives in U3"))  # noqa: E731
        sheet = ActionSheet(
            [
                ActionItem("Rename", later("Renaming chats"), icon="DRIVE_FILE_RENAME_OUTLINE"),
                ActionItem("Unpin" if chat.pinned else "Pin", pin, icon="PUSH_PIN"),
                ActionItem(f"Attachments ({chat.attachments})", later("The attachments manager"), icon="ATTACH_FILE"),
                ActionItem("Export chat", later("Exporting chats"), icon="IOS_SHARE"),
                ActionItem("Duplicate as scratch", later("Scratch chats"), icon="CONTENT_COPY"),
                ActionItem("Delete", later("Deleting chats"), icon="DELETE_OUTLINE", destructive=True),
            ],
            title=chat.title,
            tablet=self.shell.tablet,
        )
        sheet.show(self.page)
        return sheet

    def _on_status_chip(self, e: Any = None) -> None:
        block = self.state.send_block()
        if block is not None and block.fix_action == "sign_in_chatgpt":
            self.navigate_to("settings.accounts")
        else:
            self.chat_view.open_model_sheet("model")

    def _on_help(self, e: Any = None) -> ActionSheet:
        sheet = ActionSheet(
            [
                ActionItem("User guide", disabled_reason="Arrives with the bundled user guide", icon="MENU_BOOK"),
                ActionItem("Logs & diagnostics", lambda: self.navigate_to("settings.logs"), icon="TERMINAL"),
                ActionItem("About", lambda: self.navigate_to("settings.about"), icon="INFO_OUTLINE"),
            ],
            title="Help",
            tablet=self.shell.tablet,
        )
        sheet.show(self.page)
        return sheet

    # ---- feedback ------------------------------------------------------------------------

    def notify(self, message: str, action_label: Optional[str] = None, on_action: Any = None) -> Any:
        try:
            return show_snackbar(self.page, message, action_label=action_label, on_action=on_action)
        except Exception as exc:
            log.info("snackbar failed (%s): %s", exc, message)
            return None

    async def _copy_text(self, text: str) -> None:
        try:
            await self.clipboard.set(text)
            self.haptics.fire("light_impact")
            self.notify("Copied")
        except Exception as exc:
            self.notify(f"Copy failed: {exc}")

    # ---- share / OAuth -------------------------------------------------------------------

    def _on_share(self, event: dict) -> None:
        """Open-with / Share: never routed. Launch links are; files wait for the IntentRouter (U3)."""
        if launch_links(event.get("items")):
            self.dispatcher.spawn(self._route_launch_links(event.get("items")))
            return
        count = len(event.get("items") or [])
        log.info("share event with %d item(s)", count)
        if self.spike is None or self.spike_view not in self.shell.overlays:
            self.notify("Importing shared files arrives in U3")

    def _on_oauth_return(self, match: RouteMatch) -> None:
        provider = match.get("p")
        if provider == "spike" and self.spike is not None:
            self.spike._on_oauth_return(match)
            return
        log.info("OAuth return for %r (sign-in arrives in U3)", provider)
        self.dispatcher.spawn(self.opener.close_in_app_view())

    # ---- device checks (U0 spike) ------------------------------------------------------------

    def open_device_checks(self) -> ft.View:
        from glossarion_mobile.ui.spike import SpikeApp

        if self.spike is None:
            self.spike = SpikeApp(
                self.page,
                native=self.native,
                dispatcher=self.dispatcher,
                router=self.router,
                opener=self.opener,
                embedded=True,
            )
            self.spike.initial_route = self.initial_route
        self.spike_view = self.spike.build_view()
        self.shell.push_overlay(self.spike_view)
        self.page.update()
        self.dispatcher.spawn(self.spike.after_open())
        return self.spike_view

    def _spike_routes_changed(self) -> None:
        if self.spike is not None:
            self.spike._refresh_routes()


async def main(page: ft.Page) -> None:
    app = GlossarionApp(page)
    page.data = app  # keep the app (and its services) strongly referenced
    await app.start()
