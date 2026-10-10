"""AccountsProfilesFeature: wires the U4 Accounts / Profiles / Data / About pages into GlossarionApp.

``await AccountsProfilesFeature.install(app)`` - call it in ``GlossarionApp.start``
right after ``_install_chat()`` (it uses the chat feature's OAuthBridge and ChatEnv):

1. wraps the shell's screen factory for ``/settings/accounts`` (all four providers,
   slots, project picker), ``/settings/profiles`` (+ ``/<pid>``), ``/settings/prefill``,
   ``/settings/appearance``, ``/settings/storage``, ``/settings/backup``,
   ``/settings/import``, ``/settings/about``, ``/settings/danger``, ``/settings/notifications`` (U9:
   the real permission state, Allow / Open system settings, a test notification, "Always accept generated
   glossaries", battery optimisation, iOS background note) and ``/welcome`` (the
   chat feature's Welcome flow plus the step-2 sign-ins, the Endpoints link and the
   permission requests); every other route falls through;
2. marks those pages implemented on the Settings home (no "Arrives in" chip) and
   points its "Import / Export profiles" chip at Profiles;
3. gives the OAuthBridge the config reader (account slots named by the model and the
   key pools) and applies the saved Appearance prefs to the page;
4. once the backend is ready (and after every sign-in / sign-out on the Accounts page)
   reads the status of every provider slot and mirrors the signed-in slots into
   ``AppState.signed_in`` (``"authgpt"``, ``"authgem2"``, ...), which the Send button,
   the drawer chip and the ModelSheet read.

Nothing here holds desktop logic: the screens call the shared cores
(``config_store``, ``prompt_profiles``, ``api_key_encryption``, the auth modules).
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Callable, Optional

from glossarion_mobile.ui.router import RouteMatch

__all__ = ["AccountsProfilesFeature", "IMPLEMENTED_ROUTES", "SCREEN_ROUTES"]

log = logging.getLogger("glossarion.pages")

SCREEN_ROUTES = (
    "settings.accounts",
    "settings.profiles",
    "settings.profiles.detail",
    "settings.prefill",
    "settings.appearance",
    "settings.storage",
    "settings.cloud",
    "settings.backup",
    "settings.import",
    "settings.about",
    "settings.danger",
    "settings.notifications",
    "welcome",
)
#: Settings home pages this feature ships (merged into ``SettingsHome.implemented``).
IMPLEMENTED_ROUTES = frozenset(name for name in SCREEN_ROUTES if name.startswith("settings.")
                               and name != "settings.profiles.detail")


class AccountsProfilesFeature:
    def __init__(self, app: Any, *, profiles: Any = None, prefill: Any = None, exit_app: Optional[Callable] = None) -> None:
        self.app = app
        self.page = getattr(app, "page", None)
        self._profiles = profiles
        self._prefill = prefill
        self.exit_app = exit_app or (lambda: os._exit(0))
        self._fallback_factory: Optional[Callable[[RouteMatch], Any]] = None
        self.screens_built: list[str] = []
        self._unsubs: list = []
        self._refresh_task: Any = None

    # ---- install ----------------------------------------------------------------------------

    @classmethod
    async def install(cls, app: Any, **kwargs: Any) -> "AccountsProfilesFeature":
        feature = cls(app, **kwargs)
        feature.attach()
        return feature

    def attach(self) -> None:
        app = self.app
        app.pages_feature = self
        shell = getattr(app, "shell", None)
        if shell is not None and self._fallback_factory is None:
            self._fallback_factory = shell.screen_factory
            shell.screen_factory = self.screen_factory
        oauth = self.oauth
        store = self.store
        if oauth is not None and store is not None and getattr(oauth, "config_get", None) is None:
            oauth.config_get = store.get
        self.apply_saved_appearance()
        state = getattr(app, "state", None)
        backend = getattr(state, "backend", None)
        if backend is not None and not self._unsubs and oauth is not None and hasattr(oauth, "statuses"):
            self._unsubs.append(backend.subscribe(self._on_backend))
            self._on_backend(backend.value)

    def _on_backend(self, result: Any) -> None:
        if result and result.get("ok") and self._refresh_task is None:
            self._refresh_task = self._spawn(self.refresh_sign_ins())

    def _spawn(self, coro: Any) -> Any:
        dispatcher = getattr(self.app, "dispatcher", None)
        if dispatcher is not None and getattr(dispatcher, "bound", False):
            return dispatcher.spawn(coro)
        try:
            return asyncio.ensure_future(coro)
        except RuntimeError:
            coro.close()
            return None

    async def refresh_sign_ins(self) -> frozenset:
        """Status of every provider slot (worker thread), mirrored into ``AppState.signed_in``."""
        oauth = self.oauth
        if oauth is None:
            return frozenset()
        ctx = self.ctx
        run_io = getattr(ctx, "run_io", None)

        def read() -> None:
            from glossarion_mobile.services.oauth import PROVIDER_INFO

            for provider in PROVIDER_INFO:
                try:
                    oauth.statuses(provider)
                except Exception:
                    log.info("reading the %s sign-ins failed", provider, exc_info=True)

        if run_io is not None:
            await run_io(read)
        else:
            await asyncio.to_thread(read)
        return self.sync_signed_in()

    def sync_signed_in(self) -> frozenset:
        """``AppState.signed_in`` = the bridge's signed-in slot keys (on the UI loop)."""
        oauth = self.oauth
        state = getattr(self.app, "state", None)
        signed = frozenset(getattr(oauth, "signed_in", ()) or ())
        if state is not None and hasattr(state, "signed_in") and state.signed_in.value != signed:
            state.signed_in.set(signed)
        return signed

    def _signed_in_changed(self, _value: Any = None) -> None:
        self.sync_signed_in()
        chat = self.chat
        if chat is not None and hasattr(chat, "refresh_sign_in"):
            chat._spawn(chat.refresh_sign_in())

    def detach(self) -> None:
        for unsub in self._unsubs:
            try:
                unsub()
            except Exception:
                pass
        self._unsubs = []

    def apply_saved_appearance(self) -> Optional[dict]:
        prefs = getattr(self.app, "prefs", None)
        if prefs is None or self.page is None:
            return None
        try:
            saved = prefs.get("appearance")
        except Exception:
            return None
        if not saved:
            return None
        from glossarion_mobile.ui.screens.appearance import apply_appearance

        try:
            return apply_appearance(self.page, saved, state=getattr(self.app, "state", None),
                                    haptics=getattr(self.app, "haptics", None))
        except Exception:
            log.exception("applying the saved appearance failed")
            return None

    # ---- shared services --------------------------------------------------------------------

    @property
    def chat(self) -> Any:
        return getattr(self.app, "chat_feature", None)

    @property
    def oauth(self) -> Any:
        return getattr(self.chat, "oauth", None)

    @property
    def store(self) -> Any:
        return getattr(self.app, "config_store", None)

    @property
    def ctx(self) -> Any:
        """The chat feature's ChatEnv (a SettingsContext with copy / pick / share), else the settings context."""
        env = getattr(self.chat, "env", None)
        if env is not None:
            env.tablet = bool(getattr(getattr(self.app, "shell", None), "tablet", False))
            return env
        settings = getattr(self.app, "settings", None)
        return getattr(settings, "ctx", None)

    def model(self) -> str:
        store = self.store
        try:
            return str(store.get("model", "") or "") if store is not None else ""
        except Exception:
            return ""

    @property
    def profiles(self) -> Any:
        if self._profiles is None:
            from glossarion_mobile.ui.screens.profiles import ProfileService

            self._profiles = ProfileService(self.store)
        return self._profiles

    @property
    def prefill(self) -> Any:
        if self._prefill is None:
            from glossarion_mobile.ui.screens.prefill_profiles import PrefillService

            self._prefill = PrefillService(self.store)
        return self._prefill

    def _paths(self) -> Any:
        paths = getattr(self.app, "paths", None)
        if paths is None:
            try:
                from glossarion_mobile import runtime_bootstrap as rb

                paths = rb.get_paths()
            except Exception:
                paths = None
        return paths

    def _platform(self) -> str:
        platform = getattr(getattr(self.page, "platform", None), "value", None)
        return str(platform or "desktop")

    def _share_one(self, path: str) -> Any:
        ctx = self.ctx
        share = getattr(ctx, "share_files", None)
        if share is None:
            return None
        return share([path])

    def _temp_dir(self) -> Optional[str]:
        paths = self._paths()
        return str(paths.temp) if paths is not None else None

    async def _before_wipe(self) -> None:
        """Before "Wipe app data" (U10): the cloud sync gives back every persisted folder / file grant and a
        share-link upload stops (its records go with the data folder)."""
        cloud = getattr(self.app, "cloud_sync", None)
        if cloud is not None:
            await cloud.wipe()
        shares = getattr(self.app, "share_links", None)
        if shares is not None:
            await shares.wipe()

    def _stop_stores(self) -> None:
        """Before a wipe: stop every debounced saver without flushing (the old state must not come back)."""
        cloud_store = getattr(getattr(self.app, "cloud_sync", None), "store", None)
        for target in (self.store, getattr(self.app, "prefs", None), getattr(self.chat, "chats", None), cloud_store):
            saver = getattr(target, "_saver", None)
            if saver is not None:
                try:
                    saver.close(timeout=2.0)
                except Exception:
                    log.debug("closing a saver failed", exc_info=True)

    # ---- screens ------------------------------------------------------------------------------

    def make_screen(self, match: RouteMatch) -> Any:
        name = match.name
        ctx = self.ctx
        app = self.app
        if name == "welcome":
            return self.welcome_screen(match)
        if ctx is None:
            return None
        if name == "settings.accounts":
            if self.oauth is None:
                return None
            from glossarion_mobile.ui.screens.accounts import AccountsScreen

            store = self.store
            return AccountsScreen(
                match, oauth=self.oauth, notify=getattr(app, "notify", None),
                on_signed_in_changed=self._signed_in_changed, on_refreshed=self.sync_signed_in,
                page=self.page, config_get=store.get if store is not None else None,
                config_set=store.set_many if store is not None else None,
                copy_text=getattr(app, "_copy_text", None), run_io=getattr(ctx, "run_io", None),
                tablet=bool(getattr(getattr(app, "shell", None), "tablet", False)),
                prefs=getattr(app, "prefs", None),
            )
        if name == "settings.profiles":
            from glossarion_mobile.ui.screens.profiles import ProfilesScreen

            return ProfilesScreen(match, ctx, service=self.profiles, model=self.model, share_file=self._share_one,
                                  pick_files=getattr(ctx, "pick_files", None), temp_dir=self._temp_dir())
        if name == "settings.profiles.detail":
            from glossarion_mobile.ui.screens.profiles import ProfileDetailScreen

            return ProfileDetailScreen(match, ctx, service=self.profiles, model=self.model,
                                       mono=getattr(ctx, "mono", "monospace"),
                                       on_close=lambda: getattr(app, "back", lambda: None)())
        if name == "settings.prefill":
            from glossarion_mobile.ui.screens.prefill_profiles import PrefillScreen

            return PrefillScreen(match, ctx, service=self.prefill, model=self.model, mono=getattr(ctx, "mono", "monospace"))
        if name == "settings.appearance":
            from glossarion_mobile.ui.screens.appearance import AppearanceScreen

            return AppearanceScreen(match, ctx, state=getattr(app, "state", None), haptics=getattr(app, "haptics", None))
        if name == "settings.storage":
            from glossarion_mobile.ui.screens.storage import StorageScreen

            return StorageScreen(match, ctx, paths=self._paths(), platform=self._platform(),
                                 files=getattr(app, "files", None), cloud=lambda: getattr(app, "cloud_sync", None))
        if name == "settings.cloud":
            # U10 Settings › Cloud sync & sharing (the services install after this feature: read late)
            from glossarion_mobile.ui.screens.cloud_sync import CloudSyncScreen

            files = getattr(app, "files", None)
            opener = getattr(app, "opener", None)
            return CloudSyncScreen(match, ctx, cloud=lambda: getattr(app, "cloud_sync", None),
                                   shares=lambda: getattr(app, "share_links", None), platform=self._platform(),
                                   open_url=getattr(opener, "launch", None) if opener is not None else None,
                                   share_text=getattr(files, "share_text", None) if files is not None else None,
                                   show_in_files=getattr(files, "show_in_files", None) if files is not None else None)
        if name == "settings.backup":
            from glossarion_mobile.ui.screens.backup import BackupScreen

            return BackupScreen(match, ctx, share_file=self._share_one, temp_dir=self._temp_dir(),
                                pick_files=getattr(ctx, "pick_files", None))
        if name == "settings.import":
            from glossarion_mobile.ui.screens.desktop_import import DesktopImportScreen

            files = getattr(app, "files", None)
            paths = self._paths()
            scrub = [getattr(files, "inbox_dir", None)]
            if paths is not None:
                scrub += [str(paths.cache), str(paths.temp), os.path.join(str(paths.data), "Inbox")]
            return DesktopImportScreen(match, ctx, pick_files=getattr(ctx, "pick_files", None), profiles=self.profiles,
                                       scrub_dirs=tuple(s for s in scrub if s))
        if name == "settings.about":
            from glossarion_mobile.ui.screens.about import AboutScreen

            oauth = self.oauth
            return AboutScreen(match, ctx, boot=getattr(app, "boot", None), paths=self._paths(),
                               platform_name=self._platform(),
                               open_url=getattr(oauth, "open_url", None) if oauth is not None else None)
        if name == "settings.notifications":
            from glossarion_mobile.ui.screens.notifications import NotificationsScreen

            background = getattr(getattr(app, "jobs", None), "background", None)
            return NotificationsScreen(
                match, ctx, background=background,
                request_notifications=(lambda: self.request_notifications(background)) if background is not None else None,
                request_battery=(lambda: self.request_battery(background))
                if background is not None and getattr(background, "is_android", False) else None,
            )
        if name == "settings.danger":
            from glossarion_mobile.ui.screens.danger_zone import DangerZoneScreen

            chat_view = getattr(app, "chat_view", None)
            return DangerZoneScreen(
                match, ctx, oauth=self.oauth, paths=self._paths(), stop_stores=self._stop_stores,
                exit_app=self.exit_app, before_wipe=self._before_wipe,
                on_reset=(lambda: chat_view.apply_settings_changed()) if chat_view is not None else None,
            )
        return None

    def welcome_screen(self, match: RouteMatch) -> Any:
        chat = self.chat
        if chat is None or not hasattr(chat, "welcome_screen"):
            return None
        screen = chat.welcome_screen(match)
        app = self.app
        screen.oauth = getattr(chat, "oauth", None)
        screen.navigate = getattr(app, "navigate_to", None)
        screen.copy_text = getattr(app, "_copy_text", None)
        screen.page = self.page
        background = getattr(getattr(app, "jobs", None), "background", None)
        if background is not None:
            screen.on_request_notifications = lambda: self.request_notifications(background)
            if getattr(background, "is_android", False):
                screen.on_request_battery = lambda: self.request_battery(background)
        return screen

    @staticmethod
    async def request_notifications(background: Any) -> str:
        """Welcome step 4 and Settings › Notifications & background: the same request as the first Run
        (``BackgroundExecution.request_notification_permission``: Android notification permission, iOS
        notification set-up; only a definite answer is remembered, so a failed request is asked again)."""
        return await background.request_notification_permission()

    @staticmethod
    async def request_battery(background: Any) -> str:
        """Welcome step 4 (Android): the battery-optimisation exemption (asked once, like the Run tap)."""
        from glossarion_mobile.services.background import PREF_BATTERY_PROMPT

        background._set_pref(PREF_BATTERY_PROMPT, True)
        return await background._request_permission("IGNORE_BATTERY_OPTIMIZATIONS")

    def screen_factory(self, match: RouteMatch) -> Any:
        screen = None
        if match.name in SCREEN_ROUTES:
            try:
                screen = self.make_screen(match)
            except Exception:
                log.exception("building the %s screen failed", match.name)
                screen = None
            if screen is not None:
                self.screens_built.append(match.name)
        if screen is None:
            if self._fallback_factory is None:
                raise LookupError(f"no screen for {match.name}")
            screen = self._fallback_factory(match)
            self._extend_settings_home(screen)
        return screen

    def _extend_settings_home(self, screen: Any) -> None:
        implemented = getattr(screen, "implemented", None)
        if not isinstance(implemented, frozenset) or not hasattr(screen, "_on_profiles"):
            return
        screen.implemented = implemented | IMPLEMENTED_ROUTES
        ctx = getattr(screen, "ctx", None)
        if ctx is not None:
            screen._on_profiles = lambda e=None: ctx.go("settings.profiles")
