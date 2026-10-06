"""Book page (``/library/book/<bid>``; UI_SPEC §3.5, §3.12).

Layout: the shell's app bar (title = book name, ⋯ menu), a sticky summary strip
(title · ``ProgressBar`` completed/total excluding skipped · "Mode: Text" · while a
job for this book runs "⏳ Translating · 12/48" with Stop) and ``Tabs``
**Overview · Chapters · Glossary · Output** (``?tab=`` deep links; ``?filter=`` a
Chapters status group).

Data (all on the io pool): book details from ``library_core.load_book_details``
(phase ``preview`` first, then ``full``), Chapters from ``progress_core``, Glossary
from ``glossary_progress_core`` (``ui/library/progress_model.py``).

Refresh: every 2 s while the page is on top and the app is in the foreground,
``progress_core.snapshot_signature`` (+ the glossary progress file's stat) is
compared on a worker thread; only a change reloads the rows (read-only), which
the tabs apply in place by row key. Pull-to-refresh / ⟳ Refresh is a full
reconcile (``read_only=False``).
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Mapping, Optional

import flet as ft

from glossarion_mobile.services.library import CoreMissing, Poller, book_identity
from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.components.action_sheet import ActionItem, ActionSheet
from glossarion_mobile.ui.components.empty_state import EmptyState
from glossarion_mobile.ui.library import progress_model as pm
from glossarion_mobile.ui.library.common import LibraryContext, icon_button, mode_badge
from glossarion_mobile.ui.screens.base import Screen

__all__ = ["BOOK_TABS", "BookPageScreen", "job_for_book"]

log = logging.getLogger("glossarion.library.ui")

BOOK_TABS = ("overview", "chapters", "glossary", "output")
_TAB_LABELS = {"overview": "Overview", "chapters": "Chapters", "glossary": "Glossary", "output": "Output"}


def _norm(path: Any) -> str:
    try:
        return os.path.normcase(os.path.normpath(os.path.abspath(str(path))))
    except Exception:
        return str(path or "")


def job_for_book(view: Any, bid: str, book: Mapping[str, Any], *, include_queued: bool = True) -> Optional[Any]:
    """The running (or, with ``include_queued``, queued) job of this book: origin ``bid``, or the
    same output folder / raw input."""
    if view is None:
        return None
    folder = _norm(book.get("output_folder")) if book.get("output_folder") else None
    raw = _norm(book.get("raw_source_path")) if book.get("raw_source_path") else None
    candidates = ([view.active] if getattr(view, "active", None) is not None else []) + (
        list(getattr(view, "queue", ()) or ()) if include_queued else [])
    for snap in candidates:
        spec = getattr(snap, "spec", None)
        origin = getattr(spec, "origin", {}) or {}
        if origin.get("bid") == bid:
            return snap
        dirs = [getattr(snap, "output_dir", None)] + list((getattr(snap, "output_dirs", {}) or {}).values())
        if folder and any(d and _norm(d) == folder for d in dirs):
            return snap
        if raw and any(_norm(p) == raw for p in getattr(spec, "inputs", ()) or ()):
            return snap
    return None


class BookPageScreen(Screen):
    def __init__(self, match: Any, ctx: LibraryContext) -> None:
        super().__init__(match)
        self.ctx = ctx
        self.service = ctx.service
        self.bid = str(match.params.get("bid") or "") if match is not None else ""
        self.book: dict = self.service.book_for_bid(self.bid) or {}
        self.title = str(self.book.get("name") or "Book")
        tab = match.get("tab") if match is not None else None
        self.initial_tab = tab if tab in BOOK_TABS else "overview"
        self.initial_filter = match.get("filter") if match is not None else None
        self.details: Optional[dict] = None
        self.details_phase = ""
        self.progress: Optional[pm.ProgressView] = None
        self.glossary: Optional[pm.GlossaryView] = None
        self.job: Any = None
        self.job_view: Any = None
        self._signature: Any = None
        self._gp_signature: Any = None
        self._unsubs: list = []
        self._loading_progress = False
        self._loading_glossary = False
        self._full_refreshing = False
        self._progress_lock = None  # asyncio.Lock, created on the loop
        self.tabs_built: dict = {}
        self.poller = Poller(self._tick, interval=2.0, visible=lambda: self.ctx.is_top(self),
                             foreground=self.ctx.foreground, spawn=self.ctx.spawn, name="book")
        from glossarion_mobile.ui.library.chapters_tab import ChaptersTab
        from glossarion_mobile.ui.library.glossary_tab import GlossaryTab
        from glossarion_mobile.ui.library.output_tab import OutputTab
        from glossarion_mobile.ui.library.overview_tab import OverviewTab

        self.overview = OverviewTab(self)
        self.chapters = ChaptersTab(self, initial_filter=self.initial_filter)
        self.glossary_tab = GlossaryTab(self)
        self.output = OutputTab(self)

    # ---- app bar ---------------------------------------------------------------------------------

    def actions(self) -> list:
        self.menu = ft.PopupMenuButton(icon=ft.Icons.MORE_VERT, tooltip="Book actions", key="book-menu",
                                       items=self._menu_items())
        return [self.menu]

    def _menu_items(self) -> list:
        def item(label: str, handler: Any, icon: Any = None, disabled: bool = False) -> ft.PopupMenuItem:
            return ft.PopupMenuItem(content=label, icon=icon, on_click=lambda e: handler(), disabled=disabled)

        has_workspace = bool(self.book.get("output_folder"))
        return [
            item("Translate…", lambda: self.ctx.spawn(self.open_translate()), ft.Icons.TRANSLATE),
            item("Compile EPUB", lambda: self.ctx.spawn(self.compile("compile_epub")), ft.Icons.MENU_BOOK,
                 not has_workspace),
            item("Compile PDF", lambda: self.ctx.spawn(self.compile("compile_pdf")), ft.Icons.PICTURE_AS_PDF,
                 not has_workspace),
            item("Translate Metadata", lambda: self.ctx.spawn(self.translate_metadata()), ft.Icons.LABEL),
            item("QA scan", lambda: self.ctx.go("tools.qa", None, {"out": self.bid}), ft.Icons.FACT_CHECK),
            item("Edit metadata.json", lambda: self.ctx.go("library.book.metadata", {"bid": self.bid}),
                 ft.Icons.DATA_OBJECT, not has_workspace),
            item("Files", self.open_files, ft.Icons.FOLDER_OPEN, not has_workspace),
            item("Clear saved raw link", lambda: self.ctx.spawn(self.clear_raw_link()), ft.Icons.LINK_OFF),
            item("Add to Series", lambda: self.ctx.say("Series arrive in U9"), ft.Icons.COLLECTIONS_BOOKMARK, True),
            item("Delete", lambda: self.ctx.spawn(self.delete()), ft.Icons.DELETE_OUTLINE),
        ]

    # ---- body ---------------------------------------------------------------------------------------

    def build_body(self) -> ft.Control:
        if not self.book:
            return EmptyState(icon="LOCAL_LIBRARY", title="Book not found",
                              body="This book is no longer in the Library.", key="book-missing")
        self.strip_title = ft.Text(self.title, max_lines=1, overflow=ft.TextOverflow.ELLIPSIS, expand=True,
                                   theme_style=ft.TextThemeStyle.TITLE_SMALL, key="strip-title")
        self.strip_bar = ft.ProgressBar(value=None, visible=False, bar_height=4, key="strip-bar")
        self.strip_mode = mode_badge(pm.mode_label("text"), key="strip-mode")
        self.strip_job_text = ft.Text("", theme_style=ft.TextThemeStyle.LABEL_MEDIUM, key="strip-job")
        self.strip_stop = icon_button("STOP_CIRCLE", "Stop", self._stop_job, key="strip-stop")
        self.strip_job = ft.Row([ft.Icon(ft.Icons.HOURGLASS_TOP, size=16), self.strip_job_text, self.strip_stop],
                                visible=False, spacing=4, tight=True, key="strip-job-row")
        self.strip = ft.Container(
            content=ft.Column([
                ft.Row([self.strip_title, self.strip_mode], spacing=8,
                       vertical_alignment=ft.CrossAxisAlignment.CENTER),
                self.strip_bar,
                self.strip_job,
            ], spacing=2, tight=True),
            bgcolor=ft.Colors.SURFACE_CONTAINER_LOW,
            padding=ft.Padding.symmetric(horizontal=12, vertical=6),
            key="book-strip",
        )
        self.tab_index = {name: i for i, name in enumerate(BOOK_TABS)}
        self.tabs = ft.Tabs(
            length=len(BOOK_TABS),
            selected_index=self.tab_index[self.initial_tab],
            on_change=self._on_tab,
            expand=True,
            content=ft.Column([
                ft.TabBar(tabs=[ft.Tab(label=_TAB_LABELS[name]) for name in BOOK_TABS],
                          scrollable=self.ctx.text_scale >= tokens.COMPACT_TEXT_SCALE, key="book-tabbar"),
                ft.TabBarView(controls=[self.overview.build(), self.chapters.build(), self.glossary_tab.build(),
                                        self.output.build()], expand=True, key="book-tabview"),
            ], expand=True, spacing=0),
            key="book-tabs",
        )
        return ft.Column([self.strip, self.tabs], spacing=0, expand=True)

    @property
    def current_tab(self) -> str:
        tabs = getattr(self, "tabs", None)
        index = getattr(tabs, "selected_index", 0) if tabs is not None else 0
        return BOOK_TABS[index] if 0 <= index < len(BOOK_TABS) else "overview"

    def set_tab(self, name: str, *, status_filter: Optional[str] = None) -> None:
        if name not in BOOK_TABS or getattr(self, "tabs", None) is None:
            return
        self.tabs.selected_index = self.tab_index[name]
        if name == "chapters" and status_filter is not None:
            self.chapters.set_filter(status_filter)
        elif name == "glossary" and status_filter is not None:
            self.glossary_tab.set_filter(status_filter)
        self.ctx.push(self.tabs)
        self._on_tab()

    def _on_tab(self, e: Any = None) -> None:
        tab = self.current_tab
        if tab == "glossary" and self.glossary is None:
            self.ctx.spawn(self.reload_glossary())
        if tab == "output":
            self.ctx.spawn(self.output.reload())

    # ---- lifecycle ---------------------------------------------------------------------------------------

    def did_show(self) -> None:
        if not self.book:
            return
        if not self._unsubs:
            self._unsubs.append(self.service.subscribe(self._on_library))
            jobs = self.ctx.jobs
            subscribe = getattr(jobs, "subscribe", None) if jobs is not None else None
            if callable(subscribe):
                try:
                    self._unsubs.append(subscribe(self._on_jobs))
                    view = jobs.view() if hasattr(jobs, "view") else None
                    if view is not None:
                        self._on_jobs(view)
                except Exception:
                    log.debug("subscribing to jobs failed", exc_info=True)
        self.poller.start()
        self.ctx.spawn(self.load())

    def handle_back(self) -> bool:
        """Android back leaves the Chapters / Glossary selection mode first (UI_SPEC §1.6 rule 2)."""
        for tab in (self.chapters, self.glossary_tab):
            if getattr(tab, "selecting", False):
                tab.exit_selection()
                return True
        return False

    def dispose(self) -> None:
        self.poller.stop()
        for unsub in self._unsubs:
            try:
                unsub()
            except Exception:
                pass
        self._unsubs = []
        self.chapters.dispose()

    async def load(self) -> None:
        await self.reload_details("preview")
        await self.reload_progress()
        await self.reload_details("full")
        if self.initial_tab == "glossary":
            await self.reload_glossary()
        if self.initial_tab == "output":
            await self.output.reload()

    async def _tick(self) -> None:
        """2 s poll: a changed workspace signature refreshes the rows (read-only); the details (EPUB
        parse) reload only when the refreshed view really changed."""
        signature = await self.ctx.io(pm.progress_signature, self.service, self.book, self.progress)
        if signature != self._signature:
            first = self._signature is None
            self._signature = signature
            before = self.progress.signature if self.progress is not None else None
            view = await self.reload_progress()
            if not first and view is not None and view.signature != before:
                await self.reload_details("full")
        if self.glossary is not None:
            gp_signature = await self.ctx.io(pm.glossary_signature, self.glossary, self.service, self.book,
                                             self.progress)
            if gp_signature != self._gp_signature:
                await self.reload_glossary()

    async def full_refresh(self) -> None:
        """⟳ Refresh / pull-to-refresh: full reconcile (``read_only=False``) + a Library rescan.

        One at a time: a pull fires several overscroll notifications, and each refresh queues a
        read-write reconcile, an EPUB parse and a glossary reload; the extra ones are dropped."""
        if self._full_refreshing:
            return
        self._full_refreshing = True
        try:
            await self.reload_progress(full=True)
            await self.reload_details("full")
            if self.glossary is not None:
                await self.reload_glossary()
            await self.service.refresh(reason="book page")
        finally:
            self._full_refreshing = False

    # ---- loaders ------------------------------------------------------------------------------------

    async def reload_details(self, phase: str = "full") -> Optional[dict]:
        try:
            details = await self.ctx.io(self.service.load_details_blocking, self.book, phase)
        except CoreMissing as exc:
            details = {"error": str(exc)}
        except Exception as exc:
            log.info("book details failed: %s", exc)
            details = {"error": f"{exc.__class__.__name__}: {exc}"}
        if self.details is None or not details.get("error") or phase == "preview":
            self.details = details
            self.details_phase = phase
        if details.get("chapters_info") is not None:
            self.chapters.set_titles(details.get("chapters_info"))
        self.overview.render()
        return self.details

    async def reload_progress(self, *, full: bool = False, force: bool = False) -> Optional[pm.ProgressView]:
        """Open the book's Progress Manager view, or refresh it (``full`` = read-write reconcile).

        Loads are serialised (one ``BookProgress`` is refreshed in place by the shared code): a
        poll tick that finds a load running is skipped; a forced reload waits for it.
        """
        if self._loading_progress and not (full or force):
            return self.progress
        if self._progress_lock is None:
            self._progress_lock = asyncio.Lock()
        async with self._progress_lock:
            self._loading_progress = True
            try:
                show_special = bool(self.chapters.show_special)
                show_model = bool(self.chapters.show_model)
                previous = self.progress if self.progress is not None and self.progress.state is not None else None
                view = await self.ctx.io(lambda: pm.load_progress_view(
                    self.service, self.book, show_special=show_special, show_model_info=show_model,
                    full=full or force, previous=previous))
            finally:
                self._loading_progress = False
        self.progress = view
        if view.created:
            self.ctx.say(f"📁 Created: {os.path.basename(str(view.created))}")
        self._render_strip()
        self.chapters.apply(view)
        self.overview.render()
        return view

    async def reload_glossary(self) -> Optional[pm.GlossaryView]:
        if self._loading_glossary:
            return self.glossary
        self._loading_glossary = True
        try:
            previous = self.glossary if self.glossary is not None and self.glossary.state is not None else None
            progress = self.progress
            view = await self.ctx.io(lambda: pm.load_glossary_view(self.service, self.book, previous=previous,
                                                                   progress=progress))
        finally:
            self._loading_glossary = False
        self.glossary = view
        self._gp_signature = view.signature
        self.glossary_tab.apply(view)
        self.overview.render()
        return view

    def details_model(self) -> Any:
        """``library_core.BookDetailsModel`` over the loaded details (cached per payload)."""
        key = (id(self.details), bool(self.chapters.show_special), id(self.book))
        if getattr(self, "_details_model_key", None) != key:
            self._details_model_key = key
            self._details_model = (self.service.details_model(self.book, self.details or {},
                                                              show_special_files=bool(self.chapters.show_special))
                                   if self.details is not None and not self.details.get("error") else None)
        return self._details_model

    # ---- summary strip ------------------------------------------------------------------------------

    def _render_strip(self) -> None:
        if getattr(self, "strip", None) is None:
            return
        view = self.progress
        if view is not None and view.total:
            self.strip_bar.value = max(0.0, min(1.0, view.completed / float(view.total)))
            self.strip_bar.visible = True
        else:
            self.strip_bar.visible = False
        self.strip_mode.content.value = pm.mode_label(view.mode if view is not None else "text")
        job = self.job
        if job is not None:
            from glossarion_mobile.services.jobs import kind_verb

            progress = getattr(job, "progress", None)
            line = kind_verb(job.kind)
            if progress is not None and getattr(progress, "total", None):
                line += f" · {progress.completed}/{progress.total}"
            elif getattr(job, "state", None) is not None and str(getattr(job.state, "value", "")) == "QUEUED":
                line += " · queued"
            self.strip_job_text.value = f"⏳ {line}"
            self.strip_job.visible = True
        else:
            self.strip_job.visible = False
        self.ctx.push(self.strip)

    def _on_jobs(self, view: Any) -> None:
        self.job_view = view
        job = job_for_book(view, self.bid, self.book)
        finished = self.job is not None and job is None
        self.job = job
        self._render_strip()
        if finished:
            self.ctx.spawn(self.reload_progress())
            if self.glossary is not None:  # a glossary extraction may have written its progress
                self.ctx.spawn(self.reload_glossary())

    def _stop_job(self, e: Any = None) -> None:
        jobs = self.ctx.jobs
        job = self.job
        if jobs is None or job is None:
            return
        try:
            mode = jobs.request_stop(job.id)
        except TypeError:
            mode = jobs.request_stop()
        if mode == "graceful":
            self.ctx.say("Stopping after the current request · tap again to force stop")

    def _on_library(self, snap: Any) -> None:
        """Keep the book row current (state, counts, compiled output) from the Library scans."""
        identity = _norm(book_identity(self.book))
        for row in snap.all_books():
            if _norm(book_identity(row)) == identity:
                if row != self.book:
                    self.book = dict(row)
                    self.overview.render()
                    self.output.mark_stale()
                return

    # ---- actions shared by the tabs ---------------------------------------------------------------------

    async def open_translate(self) -> Any:
        from glossarion_mobile.ui.library.translate_sheet import open_translate_sheet

        return await open_translate_sheet(self.ctx, [self.book])

    async def compile(self, kind: str) -> Optional[str]:
        service = self.service
        if service.jobs is None or not service.has_job_kind(kind):
            self.ctx.say("The job service is not running")
            return None
        try:
            job_id = await service.submit(service.compile_spec(self.book, kind))
        except Exception as exc:
            self.ctx.say(f"Could not start: {exc}")
            return None
        self.ctx.haptic("medium_impact")
        self.ctx.say(f"Compiling {'EPUB' if kind == 'compile_epub' else 'PDF'}…", "Jobs",
                     lambda: self.ctx.go("jobs"))
        return job_id

    def compile_menu(self) -> ActionSheet:
        sheet = ActionSheet([
            ActionItem("\U0001f4d8 Compile EPUB", lambda: self.ctx.spawn(self.compile("compile_epub")),
                       icon="MENU_BOOK"),
            ActionItem("\U0001f4c4 Compile PDF", lambda: self.ctx.spawn(self.compile("compile_pdf")),
                       icon="PICTURE_AS_PDF"),
        ], title="Compile", tablet=self.ctx.tablet)
        self.ctx.show(sheet)
        return sheet

    async def translate_metadata(self) -> Optional[str]:
        service = self.service
        if not service.has_job_kind("metadata"):
            self.ctx.say("Metadata translation is not available in this session")
            return None
        try:
            spec = await self.ctx.io(service.metadata_spec, [self.book])
            return await service.submit(spec)
        except Exception as exc:
            self.ctx.say(f"Could not start: {exc}")
            return None

    def open_files(self) -> None:
        folder = str(self.book.get("output_folder") or "")
        prefs = self.ctx.prefs
        if not folder or prefs is None:
            self.ctx.go("tools.files", {"root": "output"})
            return
        self.ctx.go("tools.files.folder", {"root": "output", "fid": prefs.file_ref(folder)})

    def open_reader(self, *, chapter: Optional[int] = None, chapter_filename: Optional[str] = None,
                    mode: Optional[str] = None, raw_only: bool = False, resume: bool = False) -> Optional[str]:
        """The Reader on this book (at a chapter: spine position + file name; ``resume``: at the
        saved reading position)."""
        return self.ctx.open_reader(self.book, bid=self.bid, chapter=chapter, chapter_filename=chapter_filename,
                                    mode=mode, raw_only=raw_only, resume=resume)

    async def clear_raw_link(self) -> Any:
        from glossarion_mobile.ui.library.organize import clear_raw_link_flow

        return await clear_raw_link_flow(self.ctx, [self.book])

    async def delete(self) -> Any:
        """Delete this book with the Library's two-level confirmation, then return to the Library."""
        from glossarion_mobile.ui.library.delete_confirm import DeleteFlow

        self.delete_flow = DeleteFlow(self.ctx, on_done=lambda report: self.ctx.go("library"))
        return await self.delete_flow.start([self.book])
