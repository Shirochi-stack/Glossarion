"""ReaderScreen (UI_SPEC §3.11, §5.8 ``ReaderView``): the full-screen Reader for ``/reader/<bid>``.

Layers (a ``Stack``): the page (``flet_webview.WebView`` loading the document
from ``ReaderServer``, or the native ``FallbackPage``), the chrome bars, the
floating selection chips and a loading / error overlay. The chapters drawer
is the View's ``end_drawer``.

Flow: resolve the route id (``LibraryService.book_for_bid`` or a file opened
from Open-with) → ``session.plan_open`` → ``ReaderSession.load`` on the io
pool → render the chapter (``document.DocumentBuilder`` → ``server.publish`` →
``WebView.url``). The page reports ``ready`` / ``page`` / ``edge`` / ``tap`` /
``sel`` / ``pinch`` / ``scroll`` / ``link`` events (``bridge``); Python answers
with ``run_javascript`` commands. Settings changes restyle live
(``GLRDR.applyStyle``); layout, flavour and family switches re-render with the
proportional page hint. In-progress books poll ``reader_overlay`` every 3 s
while the Reader is on screen. Reading positions are saved to Prefs
``reader_positions`` (1 s debounce, and when the Reader closes); bookmarks to
``reader_bookmarks``.

"🌐 Translate" runs a ``single_chapter`` job (``force_stream_all``) and opens
the native ``LivePanel`` fed from the job's log buffer.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional

import flet as ft

from glossarion_mobile.services.library import CoreMissing
from glossarion_mobile.ui.components.action_sheet import ActionItem, ActionSheet
from glossarion_mobile.ui.components.dialogs import ConfirmDialog
from glossarion_mobile.ui.reader import bridge
from glossarion_mobile.ui.reader import model as rm
from glossarion_mobile.ui.reader.aa_sheet import SCOPE_ALL, SCOPE_BOOK, AaSheet
from glossarion_mobile.ui.reader.chrome import ReaderChrome, SelectionChipRow
from glossarion_mobile.ui.reader.document import DocumentBuilder, build_native_blocks
from glossarion_mobile.ui.reader.fallback_view import FallbackPage
from glossarion_mobile.ui.reader.live import (
    RETRANSLATE_TITLE,
    LiveFeed,
    finish_outcome,
    retranslate_question,
    return_delay,
)
from glossarion_mobile.ui.reader.live_panel import LivePanel
from glossarion_mobile.ui.reader.search_sheet import ReaderSearchSheet
from glossarion_mobile.ui.reader.session import (
    MODE_OVERLAY,
    MODE_PLAIN,
    DocEngine,
    OpenPlan,
    ReaderSession,
    plan_for_file,
    plan_open,
)
from glossarion_mobile.ui.reader.sheets import BookmarksSheet, GlossaryEntryStub, ImageViewer
from glossarion_mobile.ui.reader.toc_drawer import ChaptersDrawer, TocRow, toc_rows
from glossarion_mobile.ui.router import RouteMatch
from glossarion_mobile.ui.screens.base import Screen

__all__ = ["ReaderDeps", "ReaderScreen", "webview_supported"]

log = logging.getLogger("glossarion.reader")

OVERLAY_POLL_SECONDS = 3.0
POSITION_SAVE_DELAY = 1.0
STYLE_SAVE_DELAY = 0.4
WEB_ERROR_GRACE = 4.0  # seconds the page has to report in after a WebView resource error
LIVE_FINAL_DRAIN = 0.3  # seconds for the dispatcher to deliver a finished job's last log lines
CHROME_AUTO_HIDE = 2.5  # seconds the chrome stays up after the first chapter appears
BOOK_SETTINGS_PREF = "reader_book_settings"
MOBILE_PREFS_KEY = "reader_prefs"
SPECIAL_FILES_KEY = "epub_details_show_special_files"
OPEN_LINK_TITLE = "Open link?"
ALREADY_RUNNING = ("A translation is already running.\n"
                   "Please wait for it to finish (or stop it) first.")


def webview_supported(page: Any) -> bool:
    """flet-webview renders on Android, iOS and macOS only (never on web)."""
    if page is None or getattr(page, "web", False):
        return False
    platform = getattr(getattr(page, "platform", None), "value", getattr(page, "platform", None))
    if str(platform or "").lower() not in ("android", "ios", "macos"):
        return False
    try:
        import flet_webview  # noqa: F401
    except Exception:
        return False
    return True


@dataclass
class ReaderDeps:
    """What the Reader needs from the app (``ReaderFeature`` fills it)."""

    page: Any
    dispatcher: Any = None
    prefs: Any = None
    config: Any = None  # MobileConfigStore (get / set_many)
    jobs: Any = None  # JobService
    resolve: Optional[Callable[[str], Optional[dict]]] = None  # bid -> {"book": row} | {"path": file}
    server: Any = None  # ReaderServer (started)
    core: Any = None  # services.library.SharedCore
    notify: Optional[Callable[..., Any]] = None
    navigate_to: Optional[Callable[..., Any]] = None
    go_back: Optional[Callable[[], Any]] = None
    copy_text: Optional[Callable[[str], Any]] = None
    open_url: Optional[Callable[[str], Any]] = None
    haptics: Any = None
    cache_dir: Optional[str] = None
    width_signal: Any = None
    ask_in_chat: Optional[Callable[[str], Any]] = None
    # add_to_glossary(term, book=row): the Glossary Manager's new-entry sheet on the book's glossary (U6)
    add_to_glossary: Optional[Callable[..., Any]] = None
    has_book_page: Optional[Callable[[str], bool]] = None
    webview_ok: Optional[Callable[[], bool]] = None
    wakelock: Any = None  # the Reader's holder of the shared wakelock (services.wakelock)
    extras: dict = field(default_factory=dict)


@dataclass
class LiveRun:
    job_id: str
    chapter_file: str
    row: int
    epub_path: str
    panel: LivePanel
    feed: LiveFeed
    active: bool = True
    unsub_log: Optional[Callable[[], None]] = None
    unsub_transition: Optional[Callable[[], None]] = None


class ReaderScreen(Screen):
    title = "Reader"

    def __init__(self, match: RouteMatch, deps: ReaderDeps, *, args: Optional[Mapping[str, Any]] = None) -> None:
        super().__init__(match)
        self.deps = deps
        self.page = deps.page
        self.args = dict(args or {})
        self.bid = match.params.get("bid", "") if match is not None else ""
        self.route_chapter = _int_or_none(match.get("ch")) if match is not None else None
        self.route_mode = match.get("mode") if match is not None else None
        self.engine = DocEngine(deps.core, config=self._config_snapshot())
        self.session: Optional[ReaderSession] = None
        self.builder: Optional[DocumentBuilder] = None
        self.themes = self.engine.themes()
        self.settings = self._resolve_settings()
        self.index = 0
        self.page_no = 0
        self.page_count = 0
        self.fraction = 0.0
        self.last_page = False
        self.layout = rm.LAYOUT_SINGLE
        self.renderer = "native"
        self.state = "loading"
        self.doc_serial = 0
        self.current_doc = ""
        self.render_generation = 0
        self.deduper = bridge.EventDeduper()
        self.console_ok = False
        self.http_off_doc = ""  # the page whose fetch fallback was switched off
        self.view: Optional[ft.View] = None
        self.webview: Any = None
        self.live: Optional[LiveRun] = None
        self.disposed = False
        self.aa_sheet: Optional[AaSheet] = None
        self.search_sheet: Optional[ReaderSearchSheet] = None
        self._poll_task: Optional[asyncio.Task] = None
        self._save_handle: Optional[asyncio.TimerHandle] = None
        self._style_handle: Optional[asyncio.TimerHandle] = None
        self._pending_style: dict = {}
        self._pending_scope = SCOPE_ALL
        self._unsubs: list = []
        self._wakelock: Any = None
        self._web_error_check: Any = None
        self._selection_rect: Optional[tuple] = None
        self._pinch_steps = 0
        self.events: list = []  # handled events (diagnostics / tests)
        self.resume_offer: Optional[rm.Position] = None
        self._hold_position = False  # a resume offer is pending and the reader has not moved yet
        self._jobs_unsub: Optional[Callable[[], None]] = None
        self._target_book: dict = {}
        self._build_controls()

    # ---- settings ----------------------------------------------------------------------------

    def _config_snapshot(self) -> dict:
        config = self.deps.config
        if config is None:
            return {}
        snap = getattr(config, "snapshot", None)
        if callable(snap):
            try:
                return dict(snap() or {})
            except Exception:
                return {}
        return dict(config) if isinstance(config, Mapping) else {}

    def _config_get(self, key: str, default: Any = None) -> Any:
        config = self.deps.config
        if config is None:
            return default
        try:
            value = config.get(key, default)
        except Exception:
            return default
        return default if value is None else value

    def _prefs_get(self, key: str, default: Any = None) -> Any:
        prefs = self.deps.prefs
        if prefs is None:
            return default
        try:
            value = prefs.get(key, default)
        except Exception:
            return default
        return default if value is None else value

    def _book_overrides(self) -> dict:
        prefs = self.deps.prefs
        getter = getattr(prefs, "reader_book_settings", None) if prefs is not None and self.bid else None
        if callable(getter):
            try:
                return dict(getter(self.bid) or {})
            except Exception:
                return {}
        data = self._prefs_get(BOOK_SETTINGS_PREF, {}) or {}
        entry = data.get(self.bid) if isinstance(data, Mapping) else None
        return dict(entry) if isinstance(entry, Mapping) else {}

    def _resolve_settings(self) -> rm.ReaderSettings:
        return rm.resolve_settings(self._config_get, self._book_overrides(), self._prefs_get(MOBILE_PREFS_KEY, {}),
                                   theme_count=max(1, len(self.themes)))

    def _app_dark(self) -> Optional[bool]:
        try:
            from glossarion_mobile.ui.theme import is_dark

            return bool(is_dark(self.page))
        except Exception:
            return None

    @property
    def theme(self) -> dict:
        return rm.theme_for(self.themes, self.settings, app_dark=self._app_dark())

    def _size(self) -> tuple:
        page = self.page
        return float(getattr(page, "width", 0) or 412), float(getattr(page, "height", 0) or 860)

    def _effective_layout(self) -> str:
        width, height = self._size()
        return rm.effective_layout(self.settings.layout, width, height)

    # ---- controls ---------------------------------------------------------------------------

    def _build_controls(self) -> None:
        width, height = self._size()
        self.chrome = ReaderChrome(
            on_back=lambda e=None: self._leave(),
            on_mode=self._on_mode,
            on_search=lambda e=None: self.open_search(),
            on_more=lambda e=None: self.open_more(),
            on_prev=lambda e=None: self._spawn(self.go_chapter(self.index - 1)),
            on_next=lambda e=None: self._spawn(self.go_chapter(self.index + 1)),
            on_chapters=lambda e=None: self._spawn(self.open_chapters()),
            on_aa=lambda e=None: self.open_aa(),
            on_translate=lambda e=None: self._spawn(self.translate_chapter()),
            on_slider=lambda index: self._spawn(self.go_chapter(index)),
            narrow=width < 400,
        )
        self.selection = SelectionChipRow(
            on_copy=self._copy, on_translate=self._google_translate, on_define=self._define,
            on_glossary=self._add_to_glossary, on_chat=self._ask_in_chat,
        )
        self.toc = ChaptersDrawer(on_open=self._on_toc_row, on_native_toc=self._on_native_toc,
                                  on_special_files=self._on_special_files, dark=bool(self._app_dark()))
        self.toc.set_height(height)
        self.fallback = FallbackPage(
            on_tap_zone=self._on_fallback_tap, on_pinch=self._on_pinch_step, on_pinch_end=self._on_pinch_end,
            on_paragraph=self._on_paragraph, on_scroll=self._on_fallback_scroll,
        )
        self.fallback.set_size(width, height)
        self.page_slot = ft.Container(expand=True, content=None, bgcolor=self.theme.get("bg"))
        self.loading_text = ft.Text("Loading…", theme_style=ft.TextThemeStyle.BODY_MEDIUM)
        self.loading = ft.Container(
            expand=True,
            alignment=ft.Alignment.CENTER,
            bgcolor=ft.Colors.with_opacity(0.6, ft.Colors.SURFACE),
            content=ft.Column([ft.ProgressRing(), self.loading_text], tight=True,
                              horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=12),
            key="reader-loading",
        )
        self.top_slot = ft.Container(content=self.chrome.top, left=0, right=0, top=0)
        self.bottom_slot = ft.Container(content=self.chrome.bottom, left=0, right=0, bottom=0)
        self.stack = ft.Stack(
            controls=[self.page_slot, self.top_slot, self.bottom_slot, self.selection.container, self.loading],
            expand=True,
            key="reader-stack",
        )

    def build_body(self) -> ft.Control:
        return ft.Container(content=self.stack, expand=True, bgcolor=self.theme.get("bg"))

    def build_view(self, route: str) -> ft.View:
        """The full-screen View (no app bar; ``base.build_screen_view`` uses this when present)."""
        view = ft.View(route=route, padding=0, spacing=0, controls=[self.get_body()], bgcolor=self.theme.get("bg"),
                       end_drawer=self.toc.drawer, can_pop=False, on_confirm_pop=self._on_confirm_pop)
        self.view = view
        return view

    def _adopt_view(self) -> None:
        """Without the ``build_view`` hook the shell gave the View an app bar: make it full screen."""
        if self.view is not None:
            return
        for view in list(getattr(self.page, "views", []) or []):
            if getattr(view, "route", None) == self.route:
                view.appbar = None
                view.padding = 0
                view.bgcolor = self.theme.get("bg")
                view.end_drawer = self.toc.drawer
                view.can_pop = False
                view.on_confirm_pop = self._on_confirm_pop
                self.view = view
                try:
                    view.update()
                except Exception:
                    pass
                return

    # ---- lifecycle ---------------------------------------------------------------------------

    def did_show(self) -> None:
        self._adopt_view()
        if self.session is None and self.state == "loading":
            self._spawn(self.open())
        signal = self.deps.width_signal
        if signal is not None and not self._unsubs:
            try:
                self._unsubs.append(signal.subscribe(self._on_width))
            except Exception:
                pass

    def dispose(self) -> None:
        if self.disposed:
            return
        self.disposed = True
        self._cancel_poll()
        unsub, self._jobs_unsub = self._jobs_unsub, None
        if unsub is not None:
            try:
                unsub()
            except Exception:
                pass
        self.save_position(flush=True)
        self._flush_style()
        for unsub in self._unsubs:
            try:
                unsub()
            except Exception:
                pass
        self._unsubs = []
        if self.live is not None:
            self.live.panel.close()
            self._detach_live_listeners(self.live)
        self._set_wakelock(False)
        builder, self.builder = self.builder, None
        if builder is not None:
            builder.close()
        session = self.session
        if session is not None:
            try:
                session.close()
            except Exception:
                pass

    async def _on_confirm_pop(self, e: Any = None) -> None:
        """Android back: hide visible chrome first; a second back leaves the Reader (UI_SPEC §1.6)."""
        view = self.view
        if self.selection.visible:
            self.selection.hide()
            await self._js(bridge.js_call("clearSelection"))
            should_pop = False
        elif self.chrome.visible and self.state == "ready":
            self.chrome.set_visible(False)
            should_pop = False
        else:
            should_pop = True
        if view is not None:
            try:
                await view.confirm_pop(should_pop)
            except Exception:
                if should_pop:
                    self._leave()

    def _leave(self) -> None:
        self.save_position(flush=True)
        if self.deps.go_back is not None:
            try:
                self.deps.go_back()
                return
            except Exception:
                log.exception("leaving the reader failed")

    # ---- helpers ------------------------------------------------------------------------------

    def _spawn(self, coro: Any) -> Any:
        dispatcher = self.deps.dispatcher
        if dispatcher is not None and getattr(dispatcher, "bound", False):
            return dispatcher.spawn(coro)
        try:
            return asyncio.ensure_future(coro)
        except RuntimeError:
            coro.close()
            return None

    async def _io(self, fn: Callable[..., Any], *args: Any) -> Any:
        dispatcher = self.deps.dispatcher
        if dispatcher is not None and getattr(dispatcher, "bound", False):
            return await dispatcher.run_in_thread(fn, *args, name="gl-reader-io")
        return await asyncio.to_thread(fn, *args)

    def _post(self, fn: Callable[..., Any], *args: Any) -> None:
        dispatcher = self.deps.dispatcher
        if dispatcher is not None and getattr(dispatcher, "bound", False):
            dispatcher.post(fn, *args)
        else:
            try:
                asyncio.get_running_loop().call_soon_threadsafe(fn, *args)
            except RuntimeError:
                fn(*args)

    def notify(self, message: str, action_label: Optional[str] = None, on_action: Any = None) -> None:
        notify = self.deps.notify
        if notify is None:
            log.info("reader: %s", message)
            return
        try:
            notify(message, action_label, on_action)
        except TypeError:
            notify(message)

    def _update(self, *controls: Any) -> None:
        for control in controls:
            try:
                control.update()
            except Exception:
                pass

    def _set_loading(self, text: Optional[str]) -> None:
        self.loading.visible = text is not None
        if text is not None:
            self.loading_text.value = text
        self._update(self.loading)

    def _show_error(self, message: str) -> None:
        self.state = "error"
        self.loading.visible = True
        self.loading.content = ft.Column(
            [ft.Icon(ft.Icons.ERROR_OUTLINE, size=40, color=ft.Colors.ERROR),
             ft.Text(message, text_align=ft.TextAlign.CENTER),
             ft.TextButton(content="Back", on_click=lambda e: self._leave())],
            tight=True, horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=12)
        self.chrome.set_visible(True)
        self._update(self.loading)

    # ---- opening --------------------------------------------------------------------------------

    def _resolve_target(self) -> Optional[dict]:
        if self.args.get("path") or self.args.get("book"):
            return {"path": self.args.get("path"), "book": self.args.get("book")}
        resolver = self.deps.resolve
        if resolver is None or not self.bid:
            return None
        return resolver(self.bid)

    def _plan(self) -> OpenPlan:
        target = self._resolve_target()
        if not target:
            raise LookupError("This item is no longer available")
        book = target.get("book")
        raw_only = bool(self.args.get("raw_only"))
        self._target_book = dict(book) if book else {}
        if book:
            return plan_open(book, raw_only=raw_only, engine=self.engine)
        path = str(target.get("path") or "")
        if path and os.path.isfile(path) and path.lower().endswith(".epub"):
            return plan_for_file(path)
        if path and os.path.isdir(path) and os.path.isfile(os.path.join(path, "translation_progress.json")):
            return plan_open({"path": path, "output_folder": path, "is_in_progress": True,
                              "name": os.path.basename(os.path.normpath(path))}, raw_only=raw_only, engine=self.engine)
        raise LookupError("This item is no longer available")

    async def open(self) -> None:
        """Resolve, load and show the first chapter."""
        self._set_loading("Loading EPUB…")
        try:
            plan = await self._io(self._plan)
            show_special = bool(self._config_get(SPECIAL_FILES_KEY, False))
            session = ReaderSession(plan, engine=self.engine, cache_dir=self.deps.cache_dir,
                                    show_special=show_special)
            saved = self._saved_position()
            flavor = self._initial_flavor(saved)
            if plan.mode == "dual" and flavor == rm.ORIGINAL:
                session.flavor = rm.ORIGINAL
            await self._io(session.load)
        except CoreMissing as exc:
            self._show_error(f"The reader engine is not available in this build ({exc.name}).")
            return
        except Exception as exc:
            log.exception("opening the reader failed")
            self._show_error(str(exc) or "This book could not be opened.")
            return
        if self.disposed:
            session.close()
            return
        self.session = session
        if not session.count:
            self._show_error("No readable content found for this book.")
            return
        if flavor != session.flavor:
            if flavor == rm.BILINGUAL and not session.available_modes(0).get(rm.BILINGUAL):
                flavor = rm.TRANSLATED
            if flavor == rm.ORIGINAL and not session.has_alternate:
                flavor = rm.TRANSLATED
            if flavor != session.flavor:
                await self._io(session.set_flavor, flavor)
                if self.disposed:  # left while the flavour loaded (dispose closed the session)
                    return
        self.renderer = self._choose_renderer()
        self.layout = self._effective_layout()
        start = self._start_chapter(session)
        positions = rm.Position.from_pref(saved, session.filenames) if saved else None
        hint: Optional[dict] = None
        offer: Optional[rm.Position] = None
        if positions is not None and not positions.at_start:
            if self.args.get("resume"):
                # ▶ Continue / "Continue · Ch N · P%": open at the saved position (UI_SPEC §3.2)
                start, hint = positions.chapter, positions.hint()
            elif positions.chapter != start or positions.fraction:
                offer = positions
        # While a resume offer is pending, the untouched start must not overwrite the saved position
        # (a chapter the caller asked for -- a Chapters row, ?ch= -- is the reader's choice and is kept).
        explicit = bool(self.args.get("chapter_filename")) or self.route_chapter is not None
        self._hold_position = offer is not None and not explicit
        self.state = "ready"
        self._refresh_toc()
        await self.render(start, hint=hint)
        if self.disposed:
            # Back while the first chapter was still loading: no wakelock, resume offer or job
            # subscription for a screen that is gone (dispose already released and unsubscribed).
            return
        self._set_loading(None)
        self._set_wakelock(self.settings.keep_screen_on)
        self._spawn(self._auto_hide_chrome())
        if offer is not None:
            self.offer_resume(offer)
        if session.plan.mode == MODE_OVERLAY:
            self._watch_jobs()

    async def _auto_hide_chrome(self) -> None:
        """The chrome is up while the book opens; it fades out once the reader has had a look."""
        await asyncio.sleep(CHROME_AUTO_HIDE)
        if not self.disposed and self.state == "ready" and self.chrome.visible and not self.selection.visible:
            self.chrome.set_visible(False)

    def _saved_position(self) -> Optional[dict]:
        prefs = self.deps.prefs
        if prefs is None or not self.bid:
            return None
        try:
            return prefs.reader_position(self.bid)
        except Exception:
            return None

    def _initial_flavor(self, saved: Optional[Mapping[str, Any]]) -> str:
        if self.route_mode in rm.READER_MODES:
            return self.route_mode
        if self.args.get("raw_only"):
            return rm.ORIGINAL
        if saved and saved.get("mode") in rm.READER_MODES:
            return saved["mode"]
        return rm.ORIGINAL if self.settings.show_raw else rm.TRANSLATED

    def _start_chapter(self, session: ReaderSession) -> int:
        """The chapter to open: the caller's chapter file name first (the desktop resolves the Book
        page's spine index to a file name because the two chapter lists can be ordered differently),
        then the route's ``ch`` index, then the shared open decision's chapter file, then 0."""
        def index_of(name: Any) -> Optional[int]:
            wanted = os.path.basename(str(name or "")).lower()
            return session.filenames.index(wanted) if wanted and wanted in session.filenames else None

        chapter = index_of(self.args.get("chapter_filename"))
        if chapter is None:
            chapter = self.route_chapter
        if chapter is None:
            chapter = index_of(session.plan.initial_chapter_filename)
        return max(0, min(int(chapter or 0), session.count - 1))

    def offer_resume(self, position: rm.Position) -> None:
        session = self.session
        if session is None:
            return
        self.resume_offer = position
        number = session.chapter_info(position.chapter).number
        percent = rm.book_percent(position.chapter, position.fraction, session.count)
        self.notify(rm.resume_label(number, percent), "Resume",
                    lambda: self._spawn(self.go_chapter(position.chapter, hint=position.hint())))

    def _choose_renderer(self) -> str:
        if self.settings.lightweight:
            return "native"
        check = self.deps.webview_ok
        ok = check() if check is not None else webview_supported(self.page)
        if not ok:
            return "native"
        if self.deps.server is None:
            return "native"
        return "webview"

    # ---- rendering ------------------------------------------------------------------------------

    def _make_webview(self) -> Any:
        import flet_webview as fwv

        webview = fwv.WebView(
            url="about:blank",
            expand=True,
            bgcolor=self.theme.get("bg"),
            on_console_message=self._on_console,
            on_web_resource_error=self._on_web_error,
            # The page only ever navigates to the in-app server (http://127.0.0.1); book links are
            # intercepted by the page bridge and opened in the in-app browser view instead.
            prevent_links=["https:", "intent:", "javascript:", "file:", "content:", "mailto:", "tel:", "data:"],
        )
        return webview

    async def render(self, index: int, *, hint: Optional[Mapping[str, Any]] = None,
                     find: Optional[Mapping[str, Any]] = None, anchor: Optional[str] = None) -> None:
        session = self.session
        if session is None or not session.count:
            return
        index = max(0, min(int(index), session.count - 1))
        self.render_generation += 1
        generation = self.render_generation
        self.index = index
        self.page_no, self.page_count = 0, 0
        self.fraction = float((hint or {}).get("fraction") or 0.0)
        self.last_page = bool((hint or {}).get("last"))
        self.selection.hide()
        self._refresh_chrome()
        if self.renderer == "webview":
            server = self.deps.server
            self.doc_serial += 1
            doc_id = f"d{self.doc_serial}"
            if self.builder is None or self.builder.session is not session:
                self.builder = DocumentBuilder(session, server.register_image)
            builder = self.builder
            settings, layout, theme = self.settings, self.layout, self.theme
            try:
                built = await self._io(lambda: builder.build(
                    index, settings=settings, layout=layout, theme=theme, doc_id=doc_id,
                    event_url=server.event_path, hint=hint, find=find, anchor=anchor))
            except (LookupError, CoreMissing) as exc:
                log.info("WebView document unavailable (%s); using the native reader", exc)
                self.renderer = "native"
                await self.render(index, hint=hint)
                return
            except Exception:
                log.exception("building the reader document failed")
                self.renderer = "native"
                await self.render(index, hint=hint)
                return
            if generation != self.render_generation or self.disposed:
                return
            url = server.publish(built.html, script_nonce=built.nonce or None) + (
                f"#{built.fragment}" if built.fragment else "")
            self.current_doc = doc_id
            self.deduper.set_document(doc_id, None if layout == rm.LAYOUT_ALL else index)
            if self.webview is None:
                self.webview = self._make_webview()
                self.page_slot.content = self.webview
                self.page_slot.bgcolor = self.theme.get("bg")
                self.webview.url = url
                self._update(self.page_slot)
            else:
                self.webview.url = url
                self.webview.bgcolor = self.theme.get("bg")
                self._update(self.webview)
        else:
            try:
                blocks = await self._io(build_native_blocks, session, index)
                images = await self._io(self._block_images, session, blocks)
            except Exception:
                log.exception("rendering the chapter natively failed")
                blocks, images = [], {}
            if generation != self.render_generation or self.disposed:
                return
            self.deduper.set_document("", None if self.layout == rm.LAYOUT_ALL else index)
            if self.page_slot.content is not self.fallback.control:
                self.page_slot.content = self.fallback.control
            self.fallback.set_size(*self._size())
            self.fallback.render(blocks, theme=self.theme, settings=self.settings, images=images)
            self.page_slot.bgcolor = self.theme.get("bg")
            self._update(self.page_slot)
            if self.last_page:
                await self.fallback.scroll_to_fraction(1.0)
        self.toc.set_current(index)
        self.schedule_position_save()

    @staticmethod
    def _block_images(session: ReaderSession, blocks: list) -> dict:
        images: dict = {}
        for block in blocks:
            if block.kind == "image" and block.src and block.src not in images:
                data = session.image_bytes(block.src)
                if data:
                    images[block.src] = data
        return images

    def _refresh_chrome(self) -> None:
        session = self.session
        if session is None:
            return
        index = self.index
        info = session.chapter_info(index)
        self.chrome.set_titles(session.chapter_title(index), session.plan.title)
        modes = session.available_modes(index)
        reason = "" if modes.get(rm.BILINGUAL) else (
            "Needs both versions of this chapter" if session.has_alternate else "")
        self.chrome.set_modes(modes, session.flavor, show=session.has_alternate, bilingual_reason=reason)
        percent = rm.book_percent(index, self.fraction, session.count)
        self.chrome.set_progress(index=index, total=session.count, display_number=info.number, percent=percent,
                                 page=self.page_no, count=self.page_count, paged=rm.is_paged(self.layout),
                                 show_percent=self.settings.show_progress)
        live = self.live is not None and self.live.active
        _epub, _name, reason = session.translate_target(index)
        self.chrome.set_translate(visible=session.translate_visible(index) and not reason, live=live, reason=reason)

    def _refresh_toc(self) -> None:
        session = self.session
        if session is None:
            return
        native = session.native_toc if self.settings.native_toc else None
        rows = toc_rows(session.titles(), session.display_numbers, session.statuses(), native)
        self.toc.set_rows(rows, self.index, native_available=bool(session.native_toc),
                          native_on=self.settings.native_toc, show_special=session.show_special)

    async def _js(self, script: str) -> bool:
        webview = self.webview
        if webview is None or self.renderer != "webview":
            return False
        try:
            await webview.run_javascript(script)
            return True
        except Exception as exc:
            log.debug("run_javascript failed: %s", exc)
            return False

    # ---- navigation ---------------------------------------------------------------------------------

    async def go_chapter(self, index: int, *, hint: Optional[Mapping[str, Any]] = None, last: bool = False,
                         find: Optional[Mapping[str, Any]] = None, anchor: Optional[str] = None) -> None:
        session = self.session
        if session is None or not 0 <= index < session.count:
            return
        self._release_position_hold()
        if last:
            hint = {"last": True}
        await self.render(index, hint=hint, find=find, anchor=anchor)

    async def open_chapters(self) -> None:
        self._refresh_toc()
        view = self.view
        if view is not None:
            try:
                await view.show_end_drawer()
            except Exception:
                log.debug("show_end_drawer failed", exc_info=True)

    async def _close_drawer(self) -> None:
        view = self.view
        if view is not None:
            try:
                await view.close_end_drawer()
            except Exception:
                pass

    async def _on_toc_row(self, row: TocRow) -> None:
        await self._close_drawer()
        await self.go_chapter(row.chapter, anchor=row.fragment or None)

    def _on_native_toc(self, value: bool) -> None:
        self._apply_settings({"native_toc": bool(value)}, SCOPE_ALL)
        self._refresh_toc()

    def _on_special_files(self, value: bool) -> None:
        config = self.deps.config
        if config is not None:
            try:
                config.set_many({SPECIAL_FILES_KEY: bool(value)})
            except Exception:
                log.exception("saving %s failed", SPECIAL_FILES_KEY)
        session = self.session
        if session is None:
            return
        session.show_special = bool(value)
        session.dual_cache.clear()
        self._spawn(self._reload_keeping_position())

    async def _reload_keeping_position(self) -> None:
        session = self.session
        if session is None:
            return
        href = session.filenames[self.index] if self.index < len(session.filenames) else ""
        hint = rm.capture_hint(self.page_no, self.page_count)
        self._set_loading("Loading EPUB…")
        builder, self.builder = self.builder, None
        if builder is not None:
            builder.close()
        try:
            await self._io(session.load)
        except Exception as exc:
            self._show_error(str(exc))
            return
        self._set_loading(None)
        index = session.filenames.index(href) if href in session.filenames else min(self.index, session.count - 1)
        self._refresh_toc()
        await self.render(index, hint=hint)

    def _on_mode(self, mode: str) -> None:
        self._spawn(self.set_mode(mode))

    async def set_mode(self, mode: str) -> None:
        """Original / Translated / Bilingual; the position survives with the page hint."""
        session = self.session
        if session is None or mode == session.flavor:
            return
        if not session.available_modes(self.index).get(mode):
            self._refresh_chrome()
            return
        hint = rm.capture_hint(self.page_no, self.page_count) if rm.is_paged(self.layout) else \
            {"fraction": self.fraction, "last": False}
        href = session.filenames[self.index] if self.index < len(session.filenames) else ""
        try:
            await self._io(session.set_flavor, mode)
        except Exception as exc:
            log.exception("switching the reader flavour failed")
            self.notify(f"Could not switch: {exc}")
            return
        index = session.filenames.index(href) if href in session.filenames else min(self.index, session.count - 1)
        self._refresh_toc()
        await self.render(index, hint=hint)

    # ---- page events ------------------------------------------------------------------------------

    def _on_console(self, e: Any) -> None:
        payload = bridge.parse_console_message(getattr(e, "message", ""))
        if payload is None:
            return
        payload["_channel"] = "console"
        self.console_ok = True
        # The console channel works: every new page (the shell starts each one on both
        # channels) is switched to console-only on its first event.
        if self.http_off_doc != self.current_doc:
            self.http_off_doc = self.current_doc
            self._spawn(self._js(bridge.js_call("httpOff")))
        self.handle_payload(payload)

    def on_http_event(self, payload: dict) -> None:
        """``ReaderServer`` event callback (server thread)."""
        self._post(self.handle_payload, payload)

    def _on_web_error(self, e: Any) -> None:
        """A resource failed. When the page itself never reported in (cleartext to 127.0.0.1
        blocked, a broken WebView), switch to the native reader; a missing image does not."""
        log.warning("reader WebView resource error: %s", getattr(e, "data", e))
        if self.renderer != "webview" or self.console_ok or self._web_error_check is not None:
            return
        doc = self.current_doc

        async def check() -> None:
            try:
                await asyncio.sleep(WEB_ERROR_GRACE)
                server = self.deps.server
                heard = self.console_ok or (server is not None and server.events_received > 0)
                if (not heard and not self.disposed and self.renderer == "webview" and self.state == "ready"
                        and self.current_doc == doc):
                    self.renderer = "native"
                    self.notify("The page view is unavailable here; using the lightweight reader")
                    await self.render(self.index, hint={"fraction": self.fraction})
            finally:
                self._web_error_check = None

        self._web_error_check = self._spawn(check())

    def handle_payload(self, payload: Mapping[str, Any]) -> Optional[bridge.ReaderEvent]:
        event = bridge.normalize_event(payload)
        if event is None or self.disposed:
            return None
        if not self.deduper.accept(event):
            return None
        self.events.append(event)
        del self.events[:-200]
        kind = event.type
        if kind == "page" and event.page or kind == "scroll" and (event.fraction or 0) > 0:
            self._release_position_hold()  # the reader paged / scrolled away from the start
        if kind in ("ready", "page"):
            if event.count is not None:
                self.page_count = event.count
            if event.page is not None:
                self.page_no = event.page
            if rm.is_paged(self.layout) and self.page_count:
                hint = rm.capture_hint(self.page_no, self.page_count)
                self.fraction, self.last_page = hint["fraction"], hint["last"]
            elif event.fraction is not None:
                self.fraction = event.fraction
            self._note_scroll_all(event)
            self._refresh_chrome()
            self.schedule_position_save()
        elif kind == "scroll":
            if event.fraction is not None:
                self.fraction = event.fraction
            self._note_scroll_all(event)
            self._refresh_chrome()
            self.schedule_position_save()
        elif kind == "edge":
            if event.direction > 0 and self.session is not None and self.index + 1 < self.session.count:
                self._spawn(self.go_chapter(self.index + 1))
            elif event.direction < 0 and self.index > 0:
                self._spawn(self.go_chapter(self.index - 1, last=True))
        elif kind == "tap":
            if self.selection.visible:
                self.selection.hide()
            else:
                self.chrome.toggle()
        elif kind == "sel":
            self._on_selection(event)
        elif kind == "selrect":
            self._selection_rect = event.rect
            if self.selection.visible and event.rect is not None:
                self.selection.place(event.rect, self._size()[1])
        elif kind == "pinch":
            for _ in range(abs(event.step)):
                self._on_pinch_step(1 if event.step > 0 else -1)
            if event.ended:
                self._on_pinch_end()
        elif kind == "link":
            self._on_link(event.href, event.external)
        elif kind == "img":
            self._spawn(self._show_image(event.href))
        elif kind == "found" and event.ok is False:
            self.notify("The match could not be highlighted on this page")
        elif kind == "err":
            log.info("reader page script error: %s", event.text)
        return event

    def _note_scroll_all(self, event: bridge.ReaderEvent) -> None:
        if self.layout != rm.LAYOUT_ALL or event.chapter is None or self.session is None:
            return
        if 0 <= event.chapter < self.session.count and event.chapter != self.index:
            self.index = event.chapter
            self.toc.set_current(self.index)
        if event.chapter_fraction is not None:
            self.fraction = event.chapter_fraction

    def _on_selection(self, event: bridge.ReaderEvent) -> None:
        text = event.text.strip()
        if not text:
            self.selection.hide()
            return
        session = self.session
        original = session is not None and session.flavor == rm.ORIGINAL
        self.selection.show(text, original_mode=original, target_language=self._target_language(),
                            rect=self._selection_rect, height=self._size()[1])

    def _target_language(self) -> str:
        return str(self._config_get("output_language", "English") or "English").strip() or "English"

    def _on_link(self, href: str, external: bool) -> None:
        if not href:
            return
        if external:
            self._spawn(self._confirm_external_link(href))
            return
        if href.startswith("#"):
            self._spawn(self._js(bridge.js_call("anchor", href[1:])))
            return
        session = self.session
        if session is None:
            return
        path, _sep, fragment = href.partition("#")
        name = os.path.basename(path.replace("\\", "/")).lower()
        if name.startswith("response_"):
            name = name[len("response_"):]
        stem = os.path.splitext(name)[0]
        for index, filename in enumerate(session.filenames):
            if filename == name or os.path.splitext(filename)[0] == stem:
                self._spawn(self.go_chapter(index, anchor=fragment or None))
                return

    async def _confirm_external_link(self, href: str) -> bool:
        """A book's external link opens only for http(s) / mailto and only after the reader says so:
        the page cannot open a URL by itself (U5 review)."""
        url = str(href or "").strip()
        scheme = url.split(":", 1)[0].lower() if ":" in url else ""
        if scheme not in ("http", "https", "mailto"):
            self.notify("This link cannot be opened")
            return False
        if not await self._confirm(OPEN_LINK_TITLE, url):
            return False
        self._open_url(url)
        return True

    async def _show_image(self, src: str) -> None:
        if not src:
            return
        server = self.deps.server
        data: Optional[bytes] = None
        if server is not None and "/img/" in src:
            found = await self._io(server.image, src.rsplit("/img/", 1)[1].split("?", 1)[0])
            data = found[0] if found else None
        elif self.session is not None:
            data = await self._io(self.session.image_bytes, src)
        if data:
            ImageViewer(data, height=self._size()[1]).show(self.page)

    # ---- fallback events ---------------------------------------------------------------------------

    def _on_fallback_tap(self, zone: str) -> None:
        if zone == "centre" or not self.settings.tap_zones:
            self.chrome.toggle()
            return
        self._spawn(self.fallback.page_by(-1 if zone == "prev" else 1))

    def _on_fallback_scroll(self, fraction: float) -> None:
        self.fraction = fraction
        if fraction and fraction > 0:
            self._release_position_hold()
        self.schedule_position_save()

    def _on_paragraph(self, text: str) -> None:
        session = self.session
        original = session is not None and session.flavor == rm.ORIGINAL
        language = self._target_language()
        items = [
            ActionItem("Copy", lambda: self._copy(text), icon="CONTENT_COPY"),
            ActionItem(f"Google Translate → {language}" if original else "Define on web",
                       (lambda: self._google_translate(text)) if original else (lambda: self._define(text)),
                       icon="TRANSLATE" if original else "MENU_BOOK"),
            ActionItem("Add to glossary", lambda: self._add_to_glossary(text), icon="PLAYLIST_ADD"),
            ActionItem("Ask in chat", lambda: self._ask_in_chat(text), icon="CHAT_BUBBLE_OUTLINE"),
        ]
        ActionSheet(items, title=text[:80] + ("…" if len(text) > 80 else "")).show(self.page)

    # ---- pinch / Aa ------------------------------------------------------------------------------------

    def _on_pinch_step(self, step: int) -> None:
        size = rm.clamp_font_size(self.settings.font_size + (1 if step > 0 else -1))
        if size == self.settings.font_size:
            return
        haptics = self.deps.haptics
        if haptics is not None:
            try:
                haptics.fire("selection_click")
            except Exception:
                pass
        self._apply_settings({"font_size": size}, self._pending_scope or SCOPE_ALL, save=False)
        if self.aa_sheet is not None:
            self.aa_sheet.apply_settings(self.settings)

    def _on_pinch_end(self) -> None:
        self._schedule_style_save({"font_size": self.settings.font_size}, self._default_scope())

    def _default_scope(self) -> str:
        return SCOPE_BOOK if self.settings.overridden else SCOPE_ALL

    def open_aa(self) -> None:
        width, height = self._size()
        sheet = AaSheet(self.settings, self.themes, scope=self._default_scope(),
                        double_allowed=rm.double_page_allowed(width, height),
                        on_change=lambda changes, scope: self._apply_settings(changes, scope),
                        on_scope=self._on_scope, height=height)
        self.aa_sheet = sheet
        sheet.show(self.page)

    def _on_scope(self, scope: str) -> None:
        self._pending_scope = scope
        if scope == SCOPE_ALL:
            self._clear_book_overrides()

    def _apply_settings(self, changes: Mapping[str, Any], scope: str, *, save: bool = True) -> None:
        """Apply Aa / pinch / switch changes now; persist them for ``scope`` (debounced)."""
        old = self.settings
        count = max(1, len(self.themes))
        cleaned = {k: rm.coerce_setting(k, v, count) for k, v in changes.items()
                   if k in rm.CONFIG_KEYS or k in rm.MOBILE_DEFAULTS}
        if not cleaned:
            return
        if scope == SCOPE_BOOK:
            overridden = old.overridden | frozenset(cleaned)
        else:
            overridden = old.overridden - frozenset(cleaned)
        new = old.with_changes(**cleaned, overridden=overridden)
        self.settings = new
        if save:
            self._schedule_style_save(cleaned, scope)
        relayout = (old.layout != new.layout or old.embedded_css != new.embedded_css
                    or old.lightweight != new.lightweight)
        if "keep_screen_on" in cleaned:
            self._set_wakelock(new.keep_screen_on)
        if old.lightweight != new.lightweight:
            self.renderer = self._choose_renderer()
            self.webview = None if self.renderer == "native" else self.webview
        if old.layout != new.layout:
            self.layout = self._effective_layout()
        if old.tap_zones != new.tap_zones and self.renderer == "webview":
            relayout = True
        if old.native_toc != new.native_toc:
            self._refresh_toc()
        if relayout:
            self._spawn(self.render(self.index, hint=self._current_hint()))
            return
        self.page_slot.bgcolor = self.theme.get("bg")
        self._update(self.page_slot)
        if self.renderer == "webview":
            if self.webview is not None:
                self.webview.bgcolor = self.theme.get("bg")
            self._spawn(self._js(bridge.js_call("applyStyle", rm.override_css(self.theme, self.settings,
                                                                               layout=self.layout))))
        else:
            self._spawn(self.render(self.index, hint={"fraction": self.fraction}))
        self._refresh_chrome()

    def _current_hint(self) -> dict:
        if rm.is_paged(self.layout) and self.page_count:
            return rm.capture_hint(self.page_no, self.page_count)
        return {"fraction": self.fraction, "last": False}

    def _schedule_style_save(self, changes: Mapping[str, Any], scope: str) -> None:
        self._pending_style.update(changes)
        self._pending_scope = scope
        if self._style_handle is not None:
            self._style_handle.cancel()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._flush_style()
            return
        self._style_handle = loop.call_later(STYLE_SAVE_DELAY, self._flush_style)

    def _flush_style(self) -> None:
        self._style_handle = None
        changes, self._pending_style = dict(self._pending_style), {}
        if not changes:
            return
        scope = self._pending_scope
        config_names = {k: v for k, v in changes.items() if k in rm.CONFIG_KEYS}
        mobile_names = {k: v for k, v in changes.items() if k in rm.MOBILE_DEFAULTS}
        prefs = self.deps.prefs
        if scope == SCOPE_BOOK and self.bid and prefs is not None:
            entry = self._book_overrides()
            entry.update(changes)
            self._store_book_overrides(entry)
            return
        updates = rm.config_updates(config_names, theme_count=max(1, len(self.themes)))
        config = self.deps.config
        if updates and config is not None:
            try:
                config.set_many(updates)
            except Exception:
                log.exception("saving the reader settings failed")
        if mobile_names and prefs is not None:
            data = dict(self._prefs_get(MOBILE_PREFS_KEY, {}) or {})
            data.update(mobile_names)
            try:
                prefs.set(MOBILE_PREFS_KEY, data)
            except Exception:
                log.exception("saving the reader prefs failed")
        self._drop_book_overrides(changes)

    def _drop_book_overrides(self, names: Mapping[str, Any]) -> None:
        prefs = self.deps.prefs
        if prefs is None or not self.bid:
            return
        entry = self._book_overrides()
        if not any(name in entry for name in names):
            return
        for name in names:
            entry.pop(name, None)
        self._store_book_overrides(entry)
        self.settings = self.settings.with_changes(overridden=frozenset(entry))

    def _store_book_overrides(self, entry: Mapping[str, Any]) -> None:
        """Prefs ``reader_book_settings[bid]`` (the Aa sheet's "This book" scope)."""
        prefs = self.deps.prefs
        if prefs is None or not self.bid:
            return
        try:
            setter = getattr(prefs, "set_reader_book_settings", None)
            if callable(setter):
                setter(self.bid, dict(entry))
                return
            data = dict(self._prefs_get(BOOK_SETTINGS_PREF, {}) or {})
            if entry:
                data[self.bid] = dict(entry)
            else:
                data.pop(self.bid, None)
            prefs.set(BOOK_SETTINGS_PREF, data)
        except Exception:
            log.exception("saving the book's reader settings failed")

    def _clear_book_overrides(self) -> None:
        self._drop_book_overrides({name: None for name in list(rm.CONFIG_KEYS) + list(rm.MOBILE_DEFAULTS)})

    def _set_wakelock(self, enabled: bool) -> None:
        """Keep the screen on while reading. The Reader holds the app's shared, reference-counted
        wakelock (``deps.wakelock``), so closing it never releases a running job's lock, and a job
        ending never releases the Reader's; a private ``ft.Wakelock`` only without one."""
        if not enabled and self._wakelock is None:
            return
        if enabled and self.disposed:  # nothing would release it again
            return
        try:
            if self._wakelock is None:
                self._wakelock = self.deps.wakelock if self.deps.wakelock is not None else ft.Wakelock()
            method = self._wakelock.enable if enabled else self._wakelock.disable
            self._spawn(method())
        except Exception:
            log.debug("wakelock unavailable", exc_info=True)

    def _on_width(self, _value: Any = None) -> None:
        if self.session is None or self.state != "ready":
            return
        layout = self._effective_layout()
        width, height = self._size()
        self.fallback.set_size(width, height)
        self.toc.set_height(height)
        if layout != self.layout:
            self.layout = layout
            self._spawn(self.render(self.index, hint=self._current_hint()))

    # ---- positions & bookmarks ----------------------------------------------------------------------

    def current_position(self) -> Optional[rm.Position]:
        session = self.session
        if session is None or not session.filenames:
            return None
        index = max(0, min(self.index, len(session.filenames) - 1))
        if rm.is_paged(self.layout) and self.page_count and self.renderer == "webview":
            hint = rm.capture_hint(self.page_no, self.page_count)
            fraction, last = hint["fraction"], hint["last"]
        else:
            fraction, last = self.fraction, False
        return rm.Position(chapter=index, href=session.filenames[index], page=self.page_no, pages=self.page_count,
                           fraction=fraction, last=last, mode=session.flavor)

    def schedule_position_save(self) -> None:
        if self._save_handle is not None:
            self._save_handle.cancel()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self.save_position(flush=False)
            return
        self._save_handle = loop.call_later(POSITION_SAVE_DELAY, self.save_position)

    def _release_position_hold(self) -> None:
        """The reader moved (chapter change, paging, scrolling): its position is the one to keep."""
        self._hold_position = False

    def save_position(self, flush: bool = False) -> Optional[dict]:
        if self._save_handle is not None:
            self._save_handle.cancel()
            self._save_handle = None
        if self._hold_position:
            # The Reader opened at the start with a Resume offer up and has not moved: keep the
            # saved position (the offer's) instead of overwriting it with the untouched start.
            return None
        prefs = self.deps.prefs
        position = self.current_position()
        if prefs is None or position is None or not self.bid:
            return None
        try:
            saved = prefs.set_reader_position(self.bid, position.href, position.fraction, page=position.page,
                                              mode=position.mode, chapter=position.chapter, pages=position.pages,
                                              last=position.last)
            if flush:
                flush_fn = getattr(prefs, "flush", None)
                if callable(flush_fn):
                    flush_fn()
            return saved
        except TypeError:  # an older Prefs without the extra fields
            return prefs.set_reader_position(self.bid, position.href, position.fraction, page=position.page,
                                             mode=position.mode)
        except Exception:
            log.exception("saving the reading position failed")
            return None

    def _bookmark_label(self, mark: Mapping[str, Any]) -> str:
        label = str(mark.get("label") or "")
        if label:
            return label
        session = self.session
        href = str(mark.get("href") or "")
        if session is not None and href in session.filenames:
            index = session.filenames.index(href)
            percent = int(round(float(mark.get("fraction") or 0) * 100))
            return f"{session.chapter_title(index)} · {percent}%"
        return href

    def add_bookmark(self) -> Optional[dict]:
        prefs = self.deps.prefs
        position = self.current_position()
        if prefs is None or position is None or not self.bid or self.session is None:
            self.notify("Bookmarks need a Library book")
            return None
        number = self.session.chapter_info(position.chapter).number
        label = f"Ch {number} · {self.session.chapter_title(position.chapter)} · {int(round(position.fraction * 100))}%"
        mark = prefs.add_bookmark(self.bid, position.href, position.fraction, label)
        self.notify("Bookmark added")
        return mark

    def open_bookmarks(self) -> Optional[BookmarksSheet]:
        prefs = self.deps.prefs
        if prefs is None or not self.bid:
            self.notify("Bookmarks need a Library book")
            return None
        holder: dict = {}

        def remove(index: int) -> None:
            prefs.remove_bookmark(self.bid, index)
            holder["sheet"].set_items(prefs.bookmarks(self.bid))

        def add() -> None:
            self.add_bookmark()
            holder["sheet"].set_items(prefs.bookmarks(self.bid))

        def open_mark(mark: Mapping[str, Any]) -> None:
            position = rm.Position.from_pref(mark, self.session.filenames if self.session else [])
            if position is not None:
                self._spawn(self.go_chapter(position.chapter, hint={"fraction": position.fraction}))

        sheet = BookmarksSheet(prefs.bookmarks(self.bid), label_for=self._bookmark_label, on_open=open_mark,
                               on_remove=remove, on_add=add)
        holder["sheet"] = sheet
        sheet.show(self.page)
        return sheet

    # ---- ⋯ menu -----------------------------------------------------------------------------------------

    def open_more(self) -> ActionSheet:
        session = self.session
        has_book = bool(self.bid) and (self.deps.has_book_page(self.bid) if self.deps.has_book_page else True) \
            and not self.args.get("path")
        lightweight_label = "Use the page view" if self.settings.lightweight else "Lightweight reader"
        webview_ok = self.deps.webview_ok() if self.deps.webview_ok else webview_supported(self.page)
        share_path = ""
        if session is not None:
            entry = session.overlay_entry(self.index)
            share_path = str(entry.get("path") or "")
        items = [
            ActionItem("Bookmarks", lambda: self.open_bookmarks(), icon="BOOKMARKS"),
            ActionItem("Add bookmark here", lambda: self.add_bookmark(), icon="BOOKMARK_ADD"),
            ActionItem("Open Book page", lambda: self._open_book_page(), icon="MENU_BOOK",
                       disabled_reason=None if has_book else "Not a Library book"),
            ActionItem(lightweight_label, lambda: self._apply_settings(
                {"lightweight": not self.settings.lightweight}, SCOPE_ALL), icon="TEXT_SNIPPET",
                disabled_reason=None if webview_ok else "The page view needs Android, iOS or macOS"),
            ActionItem("Share chapter", lambda: self._share(share_path), icon="IOS_SHARE",
                       disabled_reason=None if share_path and os.path.isfile(share_path)
                       else "No translated file for this chapter"),
        ]
        sheet = ActionSheet(items, title=session.plan.title if session else "Reader")
        sheet.show(self.page)
        return sheet

    def _open_book_page(self) -> None:
        navigate = self.deps.navigate_to
        if navigate is not None and self.bid:
            navigate("library.book", {"bid": self.bid}, {"tab": "chapters"})

    def _share(self, path: str) -> None:
        share = self.deps.extras.get("share_files")
        if share is None or not path:
            self.notify("Sharing is not available in this session")
            return
        result = share([path])
        if asyncio.iscoroutine(result):
            self._spawn(result)

    # ---- search -----------------------------------------------------------------------------------------

    def open_search(self) -> Optional[ReaderSearchSheet]:
        session = self.session
        if session is None:
            return None
        reason = None if session.engine.has("reader_doc", "search_chapters") else \
            "Search needs reader_doc.search_chapters (not in this build)"
        sheet = ReaderSearchSheet(on_search=self._run_search, on_pick=self._on_search_pick, unavailable_reason=reason,
                                  height=self._size()[1])
        self.search_sheet = sheet
        sheet.show(self.page)
        return sheet

    def _run_search(self, search_id: int, query: str) -> None:
        sheet = self.search_sheet
        session = self.session
        if sheet is None or session is None:
            return

        def worker() -> None:
            done = []

            def on_batch(rows: list, finished: bool) -> None:
                if finished:
                    done.append(True)
                self._post(sheet.add_batch, search_id, list(rows), bool(finished))

            try:
                session.search(query, on_batch=on_batch, cancelled=lambda: not sheet.is_current(search_id))
            except Exception as exc:
                self._post(sheet.fail, search_id, f"Search failed: {exc}")
                return
            if not done:  # stopped, or a core without batch callbacks
                self._post(sheet.add_batch, search_id, [], True)

        self._spawn(self._io(worker))

    def _on_search_pick(self, row: Mapping[str, Any]) -> None:
        try:
            index = int(row.get("chapter_idx", 0))
        except (TypeError, ValueError):
            return
        find = {"text": str(row.get("text") or ""), "occurrence": int(row.get("local_occurrence") or 0)}
        if index == self.index and self.renderer == "webview":
            self._spawn(self._js(bridge.js_call("find", find["text"], find["occurrence"])))
            return
        self._spawn(self.go_chapter(index, find=find))

    # ---- selection actions ------------------------------------------------------------------------------

    def _copy(self, text: str) -> None:
        copy = self.deps.copy_text
        if copy is not None:
            result = copy(text)
            if asyncio.iscoroutine(result):
                self._spawn(result)
        self.selection.hide()

    def _open_url(self, url: str) -> None:
        opener = self.deps.open_url
        if opener is None:
            self.notify("No browser is available in this session")
            return
        result = opener(url)
        if asyncio.iscoroutine(result):
            self._spawn(result)

    def _google_translate(self, text: str) -> None:
        session = self.session
        try:
            url = (session.engine if session else self.engine).google_translate_url(text, self._target_language())
        except CoreMissing:
            self.notify("Google Translate needs reader_doc.google_translate_url (not in this build)")
            return
        self._open_url(url)
        self.selection.hide()

    def _define(self, text: str) -> None:
        session = self.session
        try:
            url = (session.engine if session else self.engine).define_url(text)
        except CoreMissing:
            self.notify("Define needs reader_doc.define_url (not in this build)")
            return
        self._open_url(url)
        self.selection.hide()

    def _add_to_glossary(self, text: str) -> None:
        adder = self.deps.add_to_glossary
        self.selection.hide()
        if adder is not None:
            self.save_position(flush=True)
            result = adder(text.strip(), book=dict(self._target_book) or None)
            if asyncio.iscoroutine(result):
                self._spawn(result)
            return
        session = self.session
        GlossaryEntryStub(text, on_copy=self._copy, book_title=session.plan.title if session else "").show(self.page)

    def _ask_in_chat(self, text: str) -> None:
        ask = self.deps.ask_in_chat
        if ask is None:
            self.notify("The chat is not available in this session")
            return
        self.save_position(flush=True)
        quoted = "\n".join(f"> {line}" for line in text.strip().splitlines()) + "\n\n"
        result = ask(quoted)
        if asyncio.iscoroutine(result):
            self._spawn(result)
        self.selection.hide()

    # ---- overlay polling -------------------------------------------------------------------------------

    def _watch_jobs(self) -> None:
        """Overlay mode: poll the workspace (every 3 s) only while a job for this book runs
        (UI_SPEC §3.11 Modes); the job's end triggers one last refresh."""
        if self._jobs_unsub is not None or self.disposed:
            return
        jobs = self.deps.jobs
        subscribe = getattr(jobs, "subscribe", None) if jobs is not None else None
        if not callable(subscribe):
            return
        try:
            self._jobs_unsub = subscribe(self._on_jobs)
            view = jobs.view() if hasattr(jobs, "view") else None
        except Exception:
            log.debug("watching the jobs failed", exc_info=True)
            return
        if view is not None:
            self._on_jobs(view)

    def _job_book(self) -> dict:
        """What identifies this book's jobs: the Library row, else the plan's workspace / raw EPUB."""
        book = dict(self._target_book)
        session = self.session
        if session is not None:
            plan = session.plan
            if not book.get("output_folder"):
                book["output_folder"] = plan.output_folder or plan.workspace_dir or None
            if not book.get("raw_source_path"):
                book["raw_source_path"] = plan.raw_path or plan.source_path or None
        return book

    def _on_jobs(self, view: Any) -> None:
        session = self.session
        if self.disposed or session is None or session.plan.mode != MODE_OVERLAY:
            return
        from glossarion_mobile.ui.library.book_page import job_for_book

        # Only a running job writes chapters: one that waits in the queue (behind another book's
        # job) is not polled for (UI_SPEC §3.11).
        running = job_for_book(view, self.bid, self._job_book(), include_queued=False) is not None
        if running:
            self._start_poll()
        elif self._poll_task is not None:
            self._cancel_poll()
            self._spawn(self.refresh_overlay())  # the chapters the job wrote last

    def _start_poll(self) -> None:
        if self._poll_task is not None or self.disposed:
            return
        self._poll_task = self._spawn(self._poll_loop())

    def _cancel_poll(self) -> None:
        task, self._poll_task = self._poll_task, None
        if task is not None:
            task.cancel()

    def _visible(self) -> bool:
        if self.disposed:
            return False
        page = self.page
        visible = getattr(page, "app_visible", True)
        if visible is False:
            return False
        views = list(getattr(page, "views", []) or [])
        return not views or views[-1] is self.view or self.view is None

    async def _poll_loop(self) -> None:
        try:
            while not self.disposed:
                await asyncio.sleep(OVERLAY_POLL_SECONDS)
                if not self._visible() or self.session is None:
                    continue
                await self.refresh_overlay()
        except asyncio.CancelledError:
            pass

    async def refresh_overlay(self) -> set:
        session = self.session
        if session is None or session.plan.mode != MODE_OVERLAY:
            return set()
        hint = self._current_hint()
        changed = await self._io(session.refresh_overlay)
        if not changed and not self.disposed:
            self._refresh_chrome()
            return set()
        self._refresh_toc()
        if self.index in changed or (self.layout == rm.LAYOUT_ALL and changed):
            await self.render(self.index, hint=hint)
        else:
            self._refresh_chrome()
        return changed

    # ---- live "Translate this chapter" ---------------------------------------------------------------------

    async def translate_chapter(self) -> None:
        live = self.live
        if live is not None and live.active:
            live.panel.show(self.page, self._size()[1])
            return
        session = self.session
        jobs = self.deps.jobs
        if session is None:
            return
        epub, chapter_file, reason = session.translate_target(self.index)
        if reason:
            self.notify(reason)
            return
        if jobs is None or not getattr(jobs, "has_kind", lambda k: False)("single_chapter"):
            self.notify("Single-chapter translation is not available in this build")
            return
        busy = getattr(jobs, "busy", False)
        if callable(busy):
            busy = busy()
        if busy:
            ConfirmDialog(title="Translate chapter", body=ALREADY_RUNNING, confirm_label="OK",
                          cancel_label="Close").show(self.page)
            return
        entry = session.overlay_entry(self.index)
        if entry.get("path"):
            status = str(entry.get("status") or "").strip().lower()
            if status in ("", "completed"):
                confirmed = await self._confirm(RETRANSLATE_TITLE, retranslate_question(session.chapter_title(self.index)))
                if not confirmed:
                    return
            out_dir = os.path.dirname(str(entry["path"]))
            if out_dir and os.path.isdir(out_dir):
                try:
                    await self._io(session.engine.mark_pending, out_dir, chapter_file)
                except CoreMissing:
                    self.notify("Resetting the chapter needs library_core (not in this build)")
                    return
        self.start_live(epub, chapter_file, self.index)

    async def _confirm(self, title: str, body: str) -> bool:
        future: asyncio.Future = asyncio.get_running_loop().create_future()

        def done(value: bool) -> None:
            if not future.done():
                future.set_result(value)

        dialog = ConfirmDialog(title=title, body=body, confirm_label="Yes", cancel_label="No",
                               on_confirm=lambda: done(True), on_cancel=lambda: done(False))
        dialog.show(self.page)
        return await future

    def start_live(self, epub: str, chapter_file: str, row: int) -> Optional[LiveRun]:
        from glossarion_mobile.services.jobs import JobSpec

        jobs = self.deps.jobs
        session = self.session
        if jobs is None or session is None:
            return None
        try:
            feed = LiveFeed(chapter_file)
        except Exception as exc:
            log.info("live stream classifier unavailable: %s", exc)
            self.notify("The live view needs live_stream (not in this build)")
            return None
        panel = LivePanel(chapter_file=chapter_file, feed=feed, on_stop=self._stop_live, on_hide=self._on_live_hidden,
                          mono_family=self.deps.extras.get("mono_family", "monospace"))
        spec = JobSpec(
            kind="single_chapter",
            title=f"{session.plan.title or os.path.splitext(os.path.basename(epub))[0]} · {chapter_file}",
            inputs=(epub,),
            params={"chapter_file": chapter_file, "force_stream_all": True},
            origin={"type": "library", "bid": self.bid, "label": f"Reader · {session.plan.title}"} if self.bid
            else {"type": "reader", "label": "Reader"},
        )
        try:
            job_id = jobs.submit(spec)
        except Exception as exc:
            self.notify(f"Could not start translation: {exc}")
            return None
        live = LiveRun(job_id=job_id, chapter_file=chapter_file, row=row, epub_path=epub, panel=panel, feed=feed)
        self.live = live
        try:
            live.unsub_transition = jobs.on_transition(lambda snap, previous: self._on_live_transition(live, snap))
        except Exception:
            log.exception("watching the live job failed")
        self._attach_live_log(live)
        panel.show(self.page, self._size()[1])
        self._refresh_chrome()
        return live

    def _attach_live_log(self, live: LiveRun) -> None:
        if live.unsub_log is not None:
            return
        jobs = self.deps.jobs
        dispatcher = self.deps.dispatcher
        buffer = jobs.log_buffer(live.job_id) if jobs is not None and hasattr(jobs, "log_buffer") else None
        if buffer is None:
            return
        if dispatcher is not None and hasattr(dispatcher, "subscribe_log"):
            live.unsub_log = dispatcher.subscribe_log(buffer, lambda lines, gap: live.panel.add_lines(lines),
                                                      backlog=getattr(buffer, "maxlen", 5000))
        else:  # host tests without a dispatcher: read what is there
            live.panel.add_lines(buffer.snapshot())
            live.unsub_log = lambda: None

    def _on_live_transition(self, live: LiveRun, snap: Any) -> None:
        if getattr(snap, "id", None) != live.job_id or not live.active:
            return
        self._attach_live_log(live)
        if getattr(snap, "is_terminal", False):
            self._spawn(self._finish_live(live, snap))

    def _detach_live_listeners(self, live: LiveRun) -> None:
        for name in ("unsub_log", "unsub_transition"):
            unsub = getattr(live, name)
            setattr(live, name, None)
            if unsub is not None:
                try:
                    unsub()
                except Exception:
                    pass

    def _stop_live(self) -> None:
        live = self.live
        jobs = self.deps.jobs
        if live is None or jobs is None:
            return
        try:
            jobs.request_stop(live.job_id)
        except Exception:
            log.exception("stopping the live translation failed")

    def _on_live_hidden(self) -> None:
        self._refresh_chrome()

    def _live_output_folder(self, live: LiveRun, snap: Any) -> str:
        folder = str(getattr(snap, "output_dir", "") or "")
        if not folder:
            dirs = dict(getattr(snap, "output_dirs", {}) or {})
            folder = next((str(v) for v in dirs.values() if v), "")
        if folder and os.path.isdir(folder):
            return folder
        return self.session.live_output_folder(live.row) if self.session is not None else ""

    async def _finish_live(self, live: LiveRun, snap: Any) -> None:
        if not live.active:
            return
        live.active = False
        buffer = self.deps.jobs.log_buffer(live.job_id) if self.deps.jobs is not None else None
        if buffer is not None and live.unsub_log is None:
            live.panel.add_lines(buffer.snapshot())
        elif live.unsub_log is not None:
            # The final drain (desktop _finish_live_translation): the dispatcher delivers the
            # last lines on its next pump tick (120 ms), after the terminal transition.
            await asyncio.sleep(LIVE_FINAL_DRAIN)
        self._detach_live_listeners(live)
        session = self.session
        folder = self._live_output_folder(live, snap)
        completed_fn, cleanup_fn = (session.engine.completion_fns() if session is not None else (None, None))
        outcome = await self._io(lambda: finish_outcome(folder, live.chapter_file,
                                                        stopped=bool(getattr(snap, "stopped", False)),
                                                        chapter_completed=completed_fn,
                                                        cleanup_incomplete=cleanup_fn))
        live.panel.finish(outcome.text, stopped=outcome.stopped, completed=outcome.completed)
        if session is not None:
            try:
                if session.plan.mode == MODE_PLAIN and outcome.completed and folder:
                    await self._io(session.adopt_output_folder, folder)
                    self._watch_jobs()  # now an overlay: poll while a job for it runs
                elif session.plan.mode == MODE_OVERLAY:
                    await self._io(session.refresh_overlay)
            except Exception:
                log.exception("re-merging the overlay after the live run failed")
        await asyncio.sleep(return_delay(outcome.completed))
        if self.disposed or (self.live is not live):
            return
        live.panel.close()
        self._refresh_toc()
        if session is not None and 0 <= live.row < session.count:
            await self.render(live.row)
        self._refresh_chrome()


def _int_or_none(value: Any) -> Optional[int]:
    try:
        return int(value) if value is not None and str(value).strip() != "" else None
    except (TypeError, ValueError):
        return None
