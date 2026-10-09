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
   (``services.secure_keys``); install Settings (MobileConfigStore, U2), the jobs
   layer (JobService, FileBridge, IntentRouter, background execution, Jobs /
   Files screens, U3) and the chat (ChatStoreAdapter over the shared
   ``direct_text_store``, ChatRuns, ChatGPT sign-in, Accounts / Welcome screens,
   U3), models / keys and the settings pages (U4), then the Library (Book page,
   Progress manager) and the Reader (U5); the backend warm import starts right after Settings, once the keys are in
   (prints ``GLOSSARION_BACKEND_READY``; Send stays "Preparing engine…" until then);
6. dispatch the initial route (a cold-start ``/__selftest__`` deep link runs
   the self-test, which prints ``GLOSSARION_SELFTEST PASS|FAIL``); on a first run
   that opened on the chat home the Welcome flow follows.

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
import os
import time
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote

import flet as ft

from glossarion_mobile import SELFTEST_ROUTE
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
from glossarion_mobile.ui.shell.drawer import KEYS_ROUTE, ChatDrawer
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
        # U3: JobsFeature.install sets jobs (the feature; JobService API), job_service, files and
        # intents; ChatFeature.install sets chat_feature.
        self.jobs: Any = None
        self.job_service: Any = None
        self.files: Any = None
        self.intents: Any = None
        self.chat_feature: Any = None
        # U4: ModelsKeysFeature.install sets models_keys (catalog, ModelSheet env, Models / Keys /
        # Endpoints); AccountsProfilesFeature.install sets pages_feature (Accounts, Profiles,
        # Prefill, Appearance, Storage, Backup, Import, About, Danger zone).
        self.models_keys: Any = None
        self.pages_feature: Any = None
        # U5: LibraryFeature.install sets library (LibraryService: scans, opaque book ids, the
        # shared Library actions) and library_feature (Library / Book page / Progress manager
        # screens, Open-in-Reader + Add-to-Library intents, job-finished refresh); ReaderFeature
        # .install sets reader (open_book / open_file, /reader/<bid>, the localhost page server).
        self.library: Any = None
        self.library_feature: Any = None
        self.reader: Any = None
        # U6: GlossaryFeature.install sets glossary (GlossaryService over glossary_document /
        # glossary_files / parallel_epub_core) and glossary_feature (Glossaries, the Glossary
        # Manager page, Unified glossary, Parallel EPUB pair, the Library / Book page glossary
        # hooks, the chat "Extract glossary" tool); ToolsFeature.install sets tools (Tools hub,
        # QA Scanner + report viewer, Converter / Compile, Headers & metadata).
        self.glossary: Any = None
        self.glossary_feature: Any = None
        self.tools: Any = None
        # U8: MangaFeature.install sets manga. U9: WebViewBridgeFeature sets webview_bridge (None
        # where flet-webview does not run), UpdatesFeature sets updates_feature, SeriesFeature sets
        # series, _install_keyboard sets keyboard (the Ctrl+= / - / 0 text-size shortcuts).
        self.manga: Any = None
        self.webview_bridge: Any = None
        self.updates_feature: Any = None
        self.series: Any = None
        # U10: CloudSyncService.install sets cloud_sync (Library books -> a folder of the user's own cloud
        # app, chosen once in the system picker); _install_share_links sets share_links (opt-in "Share file
        # via link": every provider off until enabled + consented in Settings › Cloud sync & sharing).
        self.cloud_sync: Any = None
        self.share_links: Any = None
        self.keyboard: Any = None
        self.freeze_watchdog: Any = None

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
            on_new_scratch=lambda e: self.chat_view._on_new_scratch(e),
            on_status=self._on_status_chip,
            on_settings=lambda e: self._drawer_navigate("settings"),
            on_keys=lambda e: self._drawer_shortcut(KEYS_ROUTE),
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
        await self._install_jobs()  # JobService + files + intents; the chat submits through it
        await self._install_chat()  # chat history (decrypted keys), runs, sign-in, welcome
        await self._install_models_keys()  # model catalog + ModelSheet services, keys, endpoints
        await self._install_pages()  # accounts (all providers), profiles, prefill, data pages
        await self._install_library()  # Library shelves, Book page, Progress manager, intents
        await self._install_reader()  # /reader/<bid> (after the Library: it resolves book ids)
        await self._install_glossary()  # Glossary Manager + the Library's glossary hooks (after the Library)
        await self._install_tools()  # Tools hub, QA Scanner, Converter, Headers & metadata (after Library + Reader)
        await self._install_manga()  # Tools › Manga (after Tools: it reuses its ToolsContext)
        await self._install_webview_bridge()  # authnd/ + search/gemini through a hidden WebView (U9)
        await self._install_updates()  # About › Updates + the startup check (U9)
        await self._install_series()  # optional chat Series (U9; after the chat and the Library)
        await self._install_cloud()  # U10 cloud sync (after the jobs, the chat and the Library)
        await self._install_share_links()  # U10 share links (after files, the Library and the chat)
        self._install_keyboard()  # Ctrl+= / Ctrl+- / Ctrl+0 text size on hardware keyboards (U9)
        self._install_diagnostics()  # HTTP log / payload / memory switches, cache cap, freeze watchdog (U9)
        self.dispatcher.spawn(self._after_ready())
        match = await self.dispatch_route(page.route, source="initial")
        await self._maybe_welcome(match)

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

    async def _install_jobs(self) -> None:
        """JobService, FileBridge, IntentRouter, background execution and the Jobs / Files
        screens (U3). Recovery of interrupted jobs runs in the background. If it fails the
        chat still opens (Send then reports that the job service is not running)."""
        try:
            from glossarion_mobile.ui.screens.jobs import JobsFeature

            await JobsFeature.install(self)  # sets self.jobs / job_service / files / intents
        except Exception:
            log.exception("jobs feature unavailable; jobs cannot run in this session")

    async def _install_chat(self) -> None:
        """The chat over the shared Direct Text store + ChatGPT sign-in + Accounts / Welcome (U3)."""
        try:
            from glossarion_mobile.ui.chat.integration import ChatFeature

            await ChatFeature.install(self)  # sets self.chat_feature; binds the chat view
        except Exception:
            log.exception("chat feature unavailable; the chat home stays a preview")
            return
        intents = self.intents
        if intents is not None:
            try:
                from glossarion_mobile.services.intents import ACTION_TRANSLATE_NEW_CHAT

                intents.handlers[ACTION_TRANSLATE_NEW_CHAT] = self._translate_in_new_chat
            except Exception:
                log.exception("registering the Translate-in-new-chat action failed")

    async def _install_models_keys(self) -> None:
        """Model catalog service, the full ModelSheet's services, Model Manager, Multi-Key Manager
        and Endpoints (U4). Without it the chat header keeps the plain model sheet."""
        try:
            from glossarion_mobile.ui.screens.model_manager import ModelsKeysFeature

            await ModelsKeysFeature.install(self)  # sets self.models_keys; wraps shell.screen_factory
        except Exception:
            log.exception("models & keys feature unavailable; those settings pages show placeholders")

    async def _install_pages(self) -> None:
        """Accounts for every sign-in provider, Profiles & prompts, Assistant prefill and the
        Appearance / Storage / Backup / Import / About / Danger zone pages (U4)."""
        try:
            from glossarion_mobile.ui.screens.pages_feature import AccountsProfilesFeature

            await AccountsProfilesFeature.install(self)  # sets self.pages_feature; wraps shell.screen_factory
        except Exception:
            log.exception("accounts & profiles feature unavailable; those settings pages show placeholders")

    async def _install_library(self) -> None:
        """The Library (shelves, Scan for raw, Book page with Overview / Chapters / Glossary /
        Output, metadata editor), the standalone Progress manager / Glossary progress, the
        "Add to Library" / "Open in Reader" intents and the job-finished Library refresh (U5).
        The Library folders are pinned (``library_core.install_library_env``) on the io pool
        right away, before any scan, import, cover or Reader call."""
        try:
            from glossarion_mobile.ui.library.feature import LibraryFeature

            feature = await LibraryFeature.install(self)  # sets self.library / library_feature
        except Exception:
            log.exception("library feature unavailable; /library shows a placeholder")
            return
        self.dispatcher.spawn(self._pin_library_env(feature))

    async def _pin_library_env(self, feature: Any) -> None:
        try:
            await feature.run_io(feature.service.ensure_env)
        except Exception:
            log.exception("pinning the Library folders failed")

    async def _install_reader(self) -> None:
        """The Reader at ``/reader/<bid>`` (flet-webview page from the in-app localhost server,
        native fallback) and ``open_book`` / ``open_file`` for the Library, job cards and the
        IntentRouter (U5)."""
        try:
            from glossarion_mobile.ui.reader.feature import ReaderFeature

            await ReaderFeature.install(self)  # sets self.reader; wraps shell.screen_factory
        except Exception:
            log.exception("reader feature unavailable; /reader shows a placeholder")

    async def _install_glossary(self) -> None:
        """The Glossary Manager (Glossaries list, the glossary page with the Editor / General /
        Balanced-Full / Minimal / Refinement tabs, Unified glossary, Parallel EPUB pair), the
        Library / Book page glossary hooks (open in editor, delete / restore glossary files,
        load as manual glossary, refine) and the chat "Extract glossary" tool (U6)."""
        try:
            from glossarion_mobile.ui.glossary.feature import GlossaryFeature

            await GlossaryFeature.install(self)  # sets self.glossary / glossary_feature; wraps the shell
        except Exception:
            log.exception("glossary feature unavailable; /glossary shows a placeholder")

    async def _install_tools(self) -> None:
        """The Tools hub, QA Scanner (+ report viewer), Converter / Compile EPUB-PDF and Headers &
        metadata screens (U6). Their job kinds (qa_scan, validate_epub, rename_outputs,
        translate_headers, metadata) are registered in ``job_kinds``."""
        try:
            from glossarion_mobile.ui.tools.feature import ToolsFeature

            await ToolsFeature.install(self)  # sets self.tools; wraps shell.screen_factory
        except Exception:
            log.exception("tools feature unavailable; /tools shows the hub")

    async def _install_manga(self) -> None:
        """Tools › Manga (Files / Settings / Editor tabs, the on-demand ONNX model manager), the
        IntentRouter "Manga translator" action and the chat's ＋ › Manga translator / "Translate as
        manga" hand-off (U8). Its job kinds (manga, manga_step) are registered in ``job_kinds``."""
        try:
            from glossarion_mobile.ui.tools.manga.feature import MangaFeature

            await MangaFeature.install(self)  # sets self.manga; wraps shell.screen_factory
        except Exception:
            log.exception("manga feature unavailable; /tools/manga shows a placeholder")

    async def _install_webview_bridge(self) -> None:
        """The in-app browser driver for the keyless browser-backed routes (authnd/ hCaptcha
        minting, search/gemini AI Mode): a hidden flet-webview registered as the backend's
        ``browser_driver`` where flet-webview runs (Android, iOS, macOS). Elsewhere those
        routes report that they need the in-app browser (U9)."""
        try:
            from glossarion_mobile.services.webview_bridge import WebViewBridgeFeature

            await WebViewBridgeFeature.install(self)  # sets self.webview_bridge (None when unsupported)
        except Exception:
            log.exception("webview bridge unavailable; authnd/ and search/gemini report it")

    async def _install_updates(self) -> None:
        """About › Updates (Check now, Check on startup, skip version, release notes, APK /
        IPA links) over the GUI-free ``update_core`` (U9). The mobile app is never
        published, so "no mobile asset" is a normal answer."""
        try:
            from glossarion_mobile.ui.screens.updates import UpdatesFeature

            await UpdatesFeature.install(self)  # sets self.updates_feature; wraps shell.screen_factory
        except Exception:
            log.exception("updates feature unavailable; /settings/updates shows a placeholder")

    async def _install_series(self) -> None:
        """Series: the optional, mobile-only chat grouping (``direct_text_chats.mobile.json``
        sidecar; drawer sections, Move to Series, series defaults; U9)."""
        try:
            from glossarion_mobile.ui.chat.series_feature import SeriesFeature

            await SeriesFeature.install(self)  # sets self.series; wraps shell.screen_factory
        except Exception:
            log.exception("series feature unavailable; Series actions stay disabled")

    async def _install_cloud(self) -> None:
        """U10: finished Library books copied into a folder of the user's own cloud app through the system
        picker (``services.cloud_sync``; no network, no developer credentials). Its queue drains at start."""
        try:
            from glossarion_mobile.services.cloud_sync import CloudSyncService

            await CloudSyncService.install(self)  # sets self.cloud_sync (and library.cloud_sync)
        except Exception:
            log.exception("cloud sync unavailable; Settings › Cloud sync & sharing shows it as unavailable")

    async def _install_share_links(self) -> None:
        """U10 "Share file via link" (``services/share_links.py``): the transfer.it browser handoff, Gofile,
        Send (end-to-end encrypted) and pixeldrain; nothing uploads without the user's tap."""
        try:
            from glossarion_mobile.services.share_links import ShareLinkService

            self.share_links = ShareLinkService.from_app(self)
            await self.share_links.load()
            if self.library is not None:
                self.library.share_links = self.share_links  # deleted books drop their links
            self.share_links.subscribe(self._on_share_link_change)
        except Exception:
            log.exception("share links unavailable; Share file via link stays disabled")

    def _on_share_link_change(self, kind: str) -> None:
        """U10: an upload the user started failed while the app is hidden -> one notification (failures only,
        like the cloud sync; the route opens the Book page, never a path or a link)."""
        shares = self.share_links
        background = getattr(self.jobs, "background", None)
        if kind != "upload" or shares is None or getattr(background, "app_visible", True):
            return
        try:
            state = shares.state  # ShareLinkService.state is a property (an UploadState), never a call
            if state.phase != "failed":
                return
            from glossarion_mobile.services.cloud_sync import CloudNotifier
            from glossarion_mobile.services.share_providers import provider_info
            from glossarion_mobile.ui.screens.cloud_sync import share_failed_notice

            label = provider_info(state.provider).label
            bid = None
            if state.book and self.library is not None:
                bid = self.library.bid_for({"output_folder": state.book, "name": os.path.basename(state.book)})
            title, body, route = share_failed_notice(state.name, label, state.message, bid)
            notifier = CloudNotifier(getattr(self.jobs, "notifications", None), self.native)
            self.dispatcher.spawn(notifier.show(f"share:{state.provider}", title, body, route))
        except Exception:
            log.exception("the share-link failure notification failed")

    def _install_diagnostics(self) -> None:
        """Logs & diagnostics at launch (U9, ``services.logs``): re-apply the HTTP logging / Save payloads /
        Memory stats switches from Prefs, cap the Payloads and http_requests folders at the desktop
        400 MB (io pool), start the UI-loop freeze watchdog (``<logs>/freeze.log``) and say so when the
        app crashed last time."""
        try:
            from glossarion_mobile.services import logs as dl

            prefs = self.prefs
            get = (lambda k, d: prefs.get(k, d)) if prefs is not None else (lambda k, d: d)
            logs_dir = getattr(self.paths, "logs", None)
            data_dir = getattr(self.paths, "data", None)
            if get(dl.PREF_HTTP_LOG, False):
                dl.apply_http_logging(True, logs_dir)
            if not get(dl.PREF_SAVE_PAYLOAD, True):
                dl.apply_save_payload(False)
            if get(dl.PREF_MEMORY_STATS, False):
                dl.apply_memory_stats(True)
            if data_dir or logs_dir:
                self.dispatcher.spawn(self.dispatcher.run_in_thread(
                    lambda: dl.sweep_debug_caches(data_dir, logs_dir), name="gl-cache-sweep"))
            if logs_dir and not os.environ.get("PYTEST_CURRENT_TEST"):
                self.freeze_watchdog = dl.FreezeWatchdog(lambda fn: self.dispatcher.post(fn), logs_dir).start()
            if getattr(self.boot, "previous_crash", False):
                self.notify("Glossarion closed unexpectedly last time", "Logs",
                            lambda e=None: self.navigate_to("settings.logs"))
        except Exception:
            log.exception("diagnostics start-up failed")

    def _install_keyboard(self) -> None:
        """Hardware keyboards (tablets, Chromebooks, desktop dev): Ctrl+= / Ctrl+- / Ctrl+0 change
        the text size of the current chat (ChatView.apply_chat_text_scale) or of the open Reader
        (its Aa font size), like the desktop zoom shortcuts (UI_SPEC §2.4)."""
        try:
            from glossarion_mobile.ui.keyboard import KeyboardShortcuts

            self.keyboard = KeyboardShortcuts(self)
            self.page.on_keyboard_event = self.keyboard.on_keyboard_event
        except Exception:
            log.exception("keyboard shortcuts unavailable")

    async def _maybe_welcome(self, match: Optional[RouteMatch]) -> None:
        """First run (the desktop first-run glossary-mode choice is not made yet): the Welcome
        flow, only when the app opened on the chat home (never over a deep link or the
        self-test). Awaited, so it is on screen before ``start`` returns."""
        feature = self.chat_feature
        if feature is None or (match is not None and match.name != "home"):
            return
        try:
            if feature.welcome_shown or not feature.needs_welcome():
                return
            feature.welcome_shown = True
            await self.navigate(build_route("welcome"))
        except Exception:
            log.exception("showing the welcome flow failed")

    def _translate_in_new_chat(self, imp: Any) -> Optional[str]:
        """IntentRouter "Translate in new chat": a fresh (or the empty current) chat with the
        shared file attached, ready to send."""
        chat_view = self.chat_view
        imported = getattr(imp, "imported", None)
        if chat_view is None or imported is None or not chat_view.bound:
            self.notify("The chat is not available in this session")
            return None
        cid = chat_view.env.chats.new_chat()
        self._open_chat(cid)
        if chat_view.cid != cid:
            chat_view.load_chat(cid)
        chat_view.attach_file(imported.path)
        return cid

    def prefill_composer(self, text: str) -> None:
        """Shared text (Open-with / Share): into the current chat's composer, as a draft."""
        chat_view = self.chat_view
        if chat_view is None:
            return
        chat_view.composer.set_text(str(text or ""))
        chat_view._on_draft_changed(str(text or ""))
        chat_view.refresh_send()
        self.navigate_to("home" if chat_view.cid in (None, "1") else "chat",
                         None if chat_view.cid in (None, "1") else {"cid": chat_view.cid})

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
        if self.jobs is not None:
            try:
                await self.jobs.route_launch_notification()  # a notification tap / Accept that cold-started the app
            except Exception:
                log.exception("routing the launch notification failed")
        shared = [i for i in items if not (isinstance(i, dict) and i.get("kind") == "url" and i.get("source") == "launch")]
        if shared and self.jobs is not None:
            try:
                await self.jobs.handle_initial_shared(shared)  # Open-with at cold start (IntentRouter)
            except Exception:
                log.exception("importing the shared files failed")
        await self._run_launch_env_selftest()

    async def _run_launch_env_selftest(self) -> None:
        """``GLOSSARION_CI_SELFTEST=<suite>`` in the launch environment runs that self-test once the
        backend is warm, exactly as the deep link does (the iOS simulator smoke: ``simctl openurl``
        stops at an "Open in ...?" alert nothing taps). The suite name goes through the router."""
        suite = os.environ.pop(rb.CI_SELFTEST_ENV, "").strip()
        if not suite:
            return
        deadline = time.monotonic() + 900
        while self.state.backend.value is None and time.monotonic() < deadline:
            await asyncio.sleep(0.5)
        await self.dispatch_route(f"{SELFTEST_ROUTE}?suite={quote(suite, safe='')}", source="launch-env")

    # ---- screens --------------------------------------------------------------------------

    def make_screen(self, match: RouteMatch) -> Screen:
        if match.name == "settings.logs":
            store = self.config_store
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
                prefs=self.prefs,
                paths=self.paths,
                files=self.files,
                config_snapshot=(store.snapshot if store is not None else None),
                navigate=self.navigate_to,
                notify=self.notify,
            )
        if match.name in _HUB_INTROS:
            return HubScreen(match, navigate=self.navigate_to, intro=_HUB_INTROS[match.name])
        return PlaceholderScreen(match)

    # ---- routing ------------------------------------------------------------------------------

    async def on_route_change(self, e: Any) -> None:
        await self.dispatch_route(getattr(e, "route", None), source="event")

    async def dispatch_route(self, raw: Optional[str], *, source: str, reset: bool = False,
                             alone: bool = False) -> Optional[RouteMatch]:
        """Apply a route. In-app navigation (``source="app"``) pushes a route opened over other
        screens on top of them; ``reset`` (drawer navigation) rebuilds the stack from the route's
        static parents (``alone``: without them, a drawer shortcut); a link from outside the app keeps
        the static parents unless they are open."""
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
        guarded = self._guarded(self.shell.leaving_entries(match, reset=reset, in_app=source == "app", alone=alone))
        if guarded and not await self._confirm_leave(guarded):
            log.info("navigation to %s cancelled: a screen kept its unsaved changes", match.route)
            if source != "app":
                self.dispatcher.spawn(self._restore_route())  # the client already shows the link's route
            return None
        self.shell.show(match, reset=reset, in_app=source == "app", alone=alone)
        self.page.update()
        return match

    @staticmethod
    def _guarded(entries: Any) -> list:
        """The entries whose screens ask before they are disposed (``Screen.confirm_leave``)."""
        return [entry for entry in entries or () if callable(getattr(entry.screen, "confirm_leave", None))]

    async def _confirm_leave(self, entries: Any) -> bool:
        """Ask the screens a navigation would dispose, top first (the glossary editor's "Unsaved
        changes"); False as soon as one keeps the user there."""
        for entry in reversed(list(entries)):
            confirm = getattr(entry.screen, "confirm_leave", None)
            if not callable(confirm):
                continue
            try:
                answer = confirm()
                if asyncio.iscoroutine(answer):
                    answer = await answer
            except Exception:
                log.exception("the leave check of %s failed", type(entry.screen).__name__)
                answer = True
            if not answer:
                return False
        return True

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

    async def navigate(self, route: str, *, reset: bool = False, alone: bool = False) -> Optional[RouteMatch]:
        """In-app navigation to a route string: apply it now, then sync the client route."""
        match = await self.dispatch_route(route, source="app", reset=reset, alone=alone)
        if match is not None and match.presentation not in (HANDLED, "sheet"):
            await self.close_drawer()
            await self._push_client_route(self.shell.current_route)
        return match

    def navigate_to(self, route_name: str, params: Optional[dict] = None, query: Optional[dict] = None, *,
                    reset: bool = False, alone: bool = False) -> None:
        """Navigate by route name (the only way UI code builds routes). A route opened over other
        screens is pushed on top of them (back returns there); ``reset`` is drawer navigation,
        ``alone`` a drawer shortcut (the route without its static parents)."""
        try:
            route = build_route(route_name, params, query)
        except RouteError as exc:
            log.error("bad in-app route %s: %s", route_name, exc)
            return
        self.dispatcher.spawn(self.navigate(route, reset=reset, alone=alone))

    def _drawer_navigate(self, route_name: str) -> None:
        """A drawer / sidebar destination: the stack restarts from the route's static parents."""
        self.navigate_to(route_name, reset=True)

    def _drawer_shortcut(self, route_name: str) -> None:
        """A drawer / sidebar footer shortcut into Settings (the footer's API keys, owner #17): the page alone
        on the stack, so one Back returns to the chat (Settings > API keys opened from Settings home still goes
        back to Settings)."""
        self.navigate_to(route_name, reset=True, alone=True)

    async def on_view_pop(self, e: Any) -> None:
        view = getattr(e, "view", None)
        # Tablet: main-area screens above a popped full-screen View (the Reader) go with it. The client has
        # already popped that View, so a screen keeping its unsaved edits ("Keep editing") stays and is
        # shown in the main area instead of being disposed.
        guarded = self._guarded(self.shell.entries_above(view))
        keep_above = bool(guarded) and not await self._confirm_leave(guarded)
        route = self.shell.pop_view(view, keep_above=keep_above)
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
        if name == "detach":
            bridge = getattr(self, "webview_bridge", None)
            if bridge is not None:
                try:
                    bridge.shutdown()  # unregister the browser driver, cancel its pages
                except Exception:
                    log.exception("shutting the webview bridge down failed")
        self.lifecycle.append(f"{time.strftime('%H:%M:%S')} {name}")
        del self.lifecycle[:-12]
        log.info("lifecycle: %s", name)
        if name == "resume":
            screen = getattr(getattr(self, "shell", None), "top_screen", None)
            resumed = getattr(screen, "app_resumed", None)
            if callable(resumed):
                try:
                    resumed()
                except Exception:
                    log.exception("%s.app_resumed failed", type(screen).__name__)
        cloud = getattr(self, "cloud_sync", None)
        if cloud is not None:
            try:
                cloud.on_lifecycle(name)  # hidden: Android drains only under a foreground service; resume: drain
            except Exception:
                log.exception("cloud sync lifecycle failed")
        if self.spike is not None:
            await self.spike.on_lifecycle(e)

    # ---- drawer / chats ---------------------------------------------------------------------

    async def open_drawer(self) -> None:
        await self.shell.open_drawer()

    async def close_drawer(self) -> None:
        await self.shell.close_drawer()

    def _open_chat(self, cid: str) -> None:
        name, params = ("home", None) if cid == "1" else ("chat", {"cid": cid})
        try:
            target = parse_route(build_route(name, params))
        except RouteError:
            target = None
        guarded = (self._guarded(self.shell.leaving_entries(target, in_app=True))
                   if target is not None and self.shell is not None else [])
        if guarded:
            # A screen with unsaved edits asks first; the chat switches only when it may be left.
            async def after_leave() -> None:
                if await self._confirm_leave(guarded):
                    self.state.current_chat.set(cid)
                    await self.navigate(build_route(name, params))

            self.dispatcher.spawn(after_leave())
            return
        self.state.current_chat.set(cid)
        self.navigate_to(name, params)

    def _chat_actions(self, chat: ChatSummary) -> ActionSheet:
        def pin() -> None:
            self.state.chats.set_pinned(chat.cid, not chat.pinned)

        # Fallback only: ChatFeature (U3) replaces this sheet once the chat store is installed.
        later = lambda what: (lambda: self.notify(f"{what} needs the chat store, which is not available"))  # noqa: E731
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

    def _open_user_guide(self) -> Any:
        """Help › User guide: the bundled ``assets/user_guide.md`` (the same sheet as About › Guides)."""
        from glossarion_mobile.ui.screens.about import show_user_guide

        run_io = None
        dispatcher = getattr(self, "dispatcher", None)
        if dispatcher is not None and getattr(dispatcher, "bound", False):
            run_io = dispatcher.run_in_thread
        coro = show_user_guide(self.page, getattr(self, "paths", None), run_io=run_io)
        spawn = getattr(dispatcher, "spawn", None) if dispatcher is not None else None
        if callable(spawn) and getattr(dispatcher, "bound", False):
            return spawn(coro)
        return asyncio.ensure_future(coro)

    def _on_status_chip(self, e: Any = None) -> None:
        block = self.state.send_block()
        if block is not None and block.fix_action == "sign_in_chatgpt":
            self._drawer_navigate("settings.accounts")
        else:
            self.chat_view.open_model_sheet("model")

    def _on_help(self, e: Any = None) -> ActionSheet:
        sheet = ActionSheet(
            [
                ActionItem("User guide", self._open_user_guide, icon="MENU_BOOK"),
                ActionItem("Logs & diagnostics", lambda: self._drawer_navigate("settings.logs"), icon="TERMINAL"),
                ActionItem("About", lambda: self._drawer_navigate("settings.about"), icon="INFO_OUTLINE"),
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
        """Open-with / Share: launch links are routed; files and text go to the IntentRouter,
        which JobsFeature registers as its own ``share`` listener (never routed)."""
        if launch_links(event.get("items")):
            self.dispatcher.spawn(self._route_launch_links(event.get("items")))
            return
        count = len(event.get("items") or [])
        log.info("share event with %d item(s)", count)
        if self.jobs is None and (self.spike is None or self.spike_view not in self.shell.overlays):
            self.notify("Shared files cannot be imported in this session")

    def _on_oauth_return(self, match: RouteMatch) -> None:
        provider = match.get("p")
        if provider == "spike" and self.spike is not None:
            self.spike._on_oauth_return(match)
            return
        log.info("OAuth return for %r (no sign-in is waiting for it)", provider)
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
