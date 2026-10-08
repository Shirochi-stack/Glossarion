"""SettingsFeature: wires MobileConfigStore, Prefs and the settings screens into GlossarionApp.

``await SettingsFeature.install(app)`` (call it in ``GlossarionApp.start``
right after ``_install_backend_keys()``, because loading decrypts API keys
with the SecureStorage key):

1. creates ``MobileConfigStore(<data>/config.json)`` and
   ``Prefs(<data>/mobile_state.json)`` and loads both on a worker thread;
2. wraps the shell's screen factory so ``/settings``, ``/settings/s/<id>``
   and ``/settings/logs/env`` get the schema-driven screens (every other
   route falls through to the app's own factory);
3. wraps the page lifecycle handler so INACTIVE / HIDE / PAUSE / DETACH flush
   both stores synchronously before the app's own handler runs;
4. follows ``AppState.job_strip`` for the "Changes apply to the next run"
   banner until JobService (U3) calls ``store.set_job_running`` itself, and
   mirrors the global ``model`` key into the chat header context.

The app keeps working when the backend or the schema is missing: the store
then starts empty (nothing is written unless the user edits a value) and
the screens show a notice.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import os
from typing import Any, Callable, Optional

from glossarion_mobile.state.config_store import MISSING, MobileConfigStore
from glossarion_mobile.state.prefs import Prefs, prefs_path_for
from glossarion_mobile.ui.router import RouteMatch
from glossarion_mobile.ui.settings.context import SettingsContext
from glossarion_mobile.ui.settings.schema_access import SchemaAccess

__all__ = ["FLUSH_LIFECYCLE_STATES", "IMPLEMENTED_ROUTES", "SettingsFeature"]

log = logging.getLogger("glossarion.settings")

FLUSH_LIFECYCLE_STATES = ("inactive", "hide", "pause", "detach")
# Routes this feature builds screens for (plus the app's own Logs & diagnostics).
SCREEN_ROUTES = ("settings", "settings.section", "settings.env_preview")
IMPLEMENTED_ROUTES = frozenset(SCREEN_ROUTES + ("settings.logs", "settings.accounts"))  # Accounts: ChatFeature (U3)
_RUNNING_JOB_STATES = ("running", "finishing", "stopping")


class SettingsFeature:
    def __init__(
        self,
        *,
        page: Any,
        store: MobileConfigStore,
        prefs: Optional[Prefs] = None,
        schema: Optional[SchemaAccess] = None,
        dispatcher: Any = None,
        navigate_route: Optional[Callable[[str], Any]] = None,
        navigate_name: Optional[Callable[..., Any]] = None,
        notify: Optional[Callable[..., Any]] = None,
        copy_handler: Optional[Callable[[str], Any]] = None,
        data_dir: Optional[str] = None,
        tablet: Callable[[], bool] = lambda: False,
    ) -> None:
        self.page = page
        self.store = store
        self.prefs = prefs
        self.schema = schema or SchemaAccess()
        self.dispatcher = dispatcher
        self.copy_handler = copy_handler
        self.data_dir = data_dir
        self._tablet = tablet
        self._file_picker: Any = None
        self.ctx = SettingsContext(
            page=page,
            store=store,
            schema=self.schema,
            dispatcher=dispatcher,
            navigate_route=navigate_route,
            navigate_name=navigate_name,
            notify=notify,
            prefs=prefs,
            file_picker_factory=self._picker,
            extras={"import_dir": os.path.join(data_dir, "imports") if data_dir else None},
        )
        self.app: Any = None
        self._fallback_factory: Optional[Callable[[RouteMatch], Any]] = None
        self._unsubs: list[Callable[[], None]] = []
        self._warm_task: Any = None
        self.defaults_ready = False
        self.screens_built: list[str] = []

    # ---- construction -------------------------------------------------------------------------

    @classmethod
    def for_app(cls, app: Any) -> "SettingsFeature":
        paths = getattr(app, "paths", None)
        data_dir = str(paths.data) if paths is not None else os.getcwd()
        config_path = str(paths.config_file) if paths is not None else None
        shell = getattr(app, "shell", None)
        schema = SchemaAccess()
        return cls(
            page=app.page,
            store=MobileConfigStore(config_path, defaults=schema.effective_default),
            prefs=Prefs(prefs_path_for(data_dir)),
            schema=schema,
            dispatcher=getattr(app, "dispatcher", None),
            navigate_route=getattr(app, "navigate", None),
            navigate_name=getattr(app, "navigate_to", None),
            notify=getattr(app, "notify", None),
            copy_handler=getattr(app, "_copy_text", None),
            data_dir=data_dir,
            tablet=(lambda: bool(getattr(shell, "tablet", False))),
        )

    @classmethod
    async def install(cls, app: Any) -> "SettingsFeature":
        """Create, load and hook the feature into a running ``GlossarionApp``."""
        feature = cls.for_app(app)
        await feature.load()
        feature.attach(app)
        return feature

    async def load(self) -> None:
        """Load config.json and mobile_state.json off the UI loop."""

        def work() -> None:
            try:
                self.schema.sections()  # import the schema off the loop
            except Exception:
                log.exception("loading the settings schema failed")
            try:
                self.store.load()
            except Exception:
                log.exception("loading config.json failed")
            if self.prefs is not None:
                try:
                    self.prefs.load()
                except Exception:
                    log.exception("loading mobile_state.json failed")
            # owner 2026-10-08: a Quick Scan sample size saved as the desktop 1000 becomes the mobile 0, once
            # (the chat's QA scans read config.json directly, so this runs at app start, not only in Tools)
            try:
                from glossarion_mobile.ui.tools.qa_model import migrate_quick_sample_size

                if self.store.loaded and not self.store.read_only:
                    migrate_quick_sample_size(self.store.get, self.store.set, self.prefs)
            except Exception:
                log.exception("the QA sample size migration failed")

        dispatcher = self.dispatcher
        if dispatcher is not None and getattr(dispatcher, "bound", False):
            await dispatcher.run_in_thread(work, name="gl-settings-load")
        else:
            await asyncio.to_thread(work)

    def attach(self, app: Any) -> None:
        """Hook into the app: screen factory, lifecycle flush, app attributes, state mirrors."""
        self.app = app
        app.settings = self
        app.config_store = self.store
        app.prefs = self.prefs
        shell = getattr(app, "shell", None)
        if shell is not None and self._fallback_factory is None:
            self._fallback_factory = shell.screen_factory
            shell.screen_factory = self.screen_factory
        page = self.page
        original = getattr(page, "on_app_lifecycle_state_change", None) if page is not None else None
        if page is not None and getattr(original, "_glossarion_settings", False) is False:

            async def on_lifecycle(e: Any) -> None:
                state = getattr(e, "state", None)
                self.on_lifecycle(str(getattr(state, "value", state)))
                if original is not None:
                    result = original(e)
                    if hasattr(result, "__await__"):
                        await result

            on_lifecycle._glossarion_settings = True  # type: ignore[attr-defined]
            page.on_app_lifecycle_state_change = on_lifecycle
        state = getattr(app, "state", None)
        if state is not None and not self._unsubs:
            if hasattr(state, "job_strip"):
                self._unsubs.append(state.job_strip.subscribe(self._on_job_strip))
                self._on_job_strip(state.job_strip.value)
            if hasattr(state, "chat_context"):
                self._unsubs.append(self.store.observe("model", lambda key, value: self.ctx.on_ui(self._sync_model)))
                self._sync_model()
            if hasattr(state, "backend"):
                # $ref defaults import backend modules: resolve them once the warm import is done.
                self._unsubs.append(state.backend.subscribe(self._schedule_warm_defaults))
                self._schedule_warm_defaults(state.backend.value)

    def _schedule_warm_defaults(self, backend_result: Any) -> None:
        if backend_result is None or self._warm_task is not None:
            return
        self._warm_task = self.ctx.spawn(self.warm_defaults())

    async def warm_defaults(self) -> int:
        """Resolve the schema's lazy defaults on a worker thread, then refresh open settings screens."""
        try:
            count = await self.ctx.run_io(self.schema.warm_defaults)
        except Exception:
            log.exception("resolving settings defaults failed")
            return 0
        self.defaults_ready = True
        self.refresh_open_screens()
        return count

    def refresh_open_screens(self) -> None:
        shell = getattr(self.app, "shell", None)
        for entry in list(getattr(shell, "stack", None) or ()):
            refresh = getattr(entry.screen, "refresh_all", None)
            if refresh is not None:
                try:
                    refresh()
                except Exception:
                    log.exception("refreshing %s failed", type(entry.screen).__name__)

    def detach(self) -> None:
        for unsub in self._unsubs:
            unsub()
        self._unsubs = []

    # ---- screens ---------------------------------------------------------------------------------

    def _picker(self) -> Any:
        if self._file_picker is None:
            import flet as ft

            self._file_picker = ft.FilePicker()  # a page service: keep the reference
        return self._file_picker

    def make_screen(self, match: RouteMatch) -> Any:
        """Screen for a settings route, or None for routes this feature does not own."""
        self.ctx.tablet = bool(self._tablet())
        if match.name == "settings":
            from glossarion_mobile.ui.settings.settings_home import SettingsHome

            screen: Any = SettingsHome(match, self.ctx, implemented_routes=IMPLEMENTED_ROUTES)
        elif match.name == "settings.section":
            from glossarion_mobile.ui.settings.section_page import SectionPage
            from glossarion_mobile.ui.settings.settings_home import section_screen_for

            # a section with its own page (Glossary tabs, Endpoints) opens it, also from search hits
            screen = section_screen_for(self.ctx, match) or SectionPage(match, self.ctx)
        elif match.name == "settings.env_preview":
            from glossarion_mobile.ui.screens.env_preview import EnvPreviewScreen

            screen = EnvPreviewScreen(match, store=self.store, dispatcher=self.dispatcher, page=self.page,
                                      copy_handler=self.copy_handler, data_dir=self.data_dir)
        else:
            return None
        self.screens_built.append(match.name)
        return screen

    def screen_factory(self, match: RouteMatch) -> Any:
        screen = self.make_screen(match)
        if screen is None:
            if self._fallback_factory is None:
                raise LookupError(f"no screen for {match.name}")
            screen = self._fallback_factory(match)
        if self.prefs is not None:  # Prefs.last_routes (parents are built first, so the leaf wins)
            try:
                self.prefs.set_last_route(match.route)
            except Exception:
                log.debug("recording the last route failed", exc_info=True)
        return screen

    # ---- lifecycle / state ----------------------------------------------------------------------------

    def on_lifecycle(self, name: str) -> None:
        if name in FLUSH_LIFECYCLE_STATES:
            self.flush()

    def flush(self) -> None:
        """Synchronous save of pending edits (app going to the background)."""
        for target in (self.store, self.prefs):
            if target is None:
                continue
            try:
                target.flush()
            except Exception:
                log.exception("flush of %s failed", type(target).__name__)

    def close(self) -> None:
        self.detach()
        for target in (self.store, self.prefs):
            if target is not None:
                try:
                    target.close()
                except Exception:
                    log.exception("close of %s failed", type(target).__name__)

    def _on_job_strip(self, model: Any) -> None:
        running = model is not None and getattr(model, "state", "running") in _RUNNING_JOB_STATES
        self.store.set_job_running(running)

    def _sync_model(self) -> None:
        state = getattr(self.app, "state", None)
        if state is None:
            return
        value = self.store.get("model", MISSING)
        if value is MISSING or not isinstance(value, str) or not value.strip():
            return
        context = state.chat_context.value
        if getattr(context, "model", None) != value and dataclasses.is_dataclass(context):
            state.chat_context.set(dataclasses.replace(context, model=value))
