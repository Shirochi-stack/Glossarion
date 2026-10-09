"""LibraryFeature: wires the U5 Library, Book page and standalone Progress manager into GlossarionApp.

``await LibraryFeature.install(app)`` - call it in ``GlossarionApp.start`` right after
``_install_chat()`` (it needs the jobs feature's FileBridge / IntentRouter / JobService
and the chat's Translate-in-new-chat handler):

1. builds the ``LibraryService`` (app paths, MobileConfigStore, Prefs, FileBridge,
   JobsFeature, io pool = ``UiDispatcher.run_in_thread``) and exposes it as
   ``app.library`` (the Reader resolves ``/reader/<bid>`` with
   ``app.library.book_for_bid(bid)`` / ``identity_for_bid(bid)``) and the feature as
   ``app.library_feature``;
2. routes "Add to Library" copies through ``library_core.import_paths``
   (``FileBridge.library_import``) and registers the IntentRouter handlers
   ``open_in_reader`` (``/reader/<bid>`` for a shared EPUB) and ``add_to_library``
   (copy + register, then "Open" on the new book);
3. wraps the shell's screen factory for ``library``, ``library.scan_raw``,
   ``library.book``, ``library.book.metadata``, ``tools.progress`` and
   ``tools.progress.glossary`` (every other route falls through);
4. calls ``LibraryService.on_job_finished`` from ``JobService.on_transition`` for every
   job that ends (Library rescan + the Android "Mirror outputs" copy), once per job;
5. pokes the visible Library / Book page poller when the app returns to the
   foreground (UI_SPEC §3.12 "On resume, one immediate refresh").
"""

from __future__ import annotations

import asyncio
import logging
import os
from types import SimpleNamespace
from typing import Any, Callable, Optional

from glossarion_mobile.ui.router import RouteMatch
from glossarion_mobile.ui.text_scale import effective as effective_text_scale

__all__ = ["IMPLEMENTED_ROUTES", "LibraryFeature", "SCREEN_ROUTES"]

log = logging.getLogger("glossarion.library")

SCREEN_ROUTES = (
    "library",
    "library.scan_raw",
    "library.book",
    "library.book.metadata",
    "tools.progress",
    "tools.progress.glossary",
)
#: Routes this feature ships (for the Tools hub / drawer "implemented" sets; Integrate merges them).
IMPLEMENTED_ROUTES = frozenset(SCREEN_ROUTES)

_RESUME_STATES = ("resume", "show", "restart")
_BACKGROUND_STATES = ("pause", "hide", "inactive", "detach")


class LibraryFeature:
    def __init__(self, app: Any, *, service: Any = None) -> None:
        self.app = app
        self.page = getattr(app, "page", None)
        self.dispatcher = getattr(app, "dispatcher", None)
        self._fallback_factory: Optional[Callable[[RouteMatch], Any]] = None
        self._unsubs: list = []
        self._finished: set = set()
        self.foregrounded = True
        self.screens_built: list = []
        self.service = service or self._make_service()

    # ---- install ---------------------------------------------------------------------------------

    @classmethod
    async def install(cls, app: Any, **kwargs: Any) -> "LibraryFeature":
        feature = cls(app, **kwargs)
        feature.attach()
        return feature

    def _make_service(self) -> Any:
        from glossarion_mobile.services.library import LibraryService

        app = self.app
        return LibraryService(
            paths=getattr(app, "paths", None),
            config=getattr(app, "config_store", None),
            prefs=getattr(app, "prefs", None),
            files=getattr(app, "files", None),
            jobs=getattr(app, "jobs", None),
            run_io=self.run_io,
            post=self.post,
        )

    def attach(self) -> None:
        app = self.app
        app.library = self.service
        app.library_feature = self
        files = getattr(app, "files", None)
        if files is not None and getattr(files, "library_import", None) is None:
            files.library_import = self._library_import
        intents = getattr(app, "intents", None)
        if intents is not None:
            from glossarion_mobile.services.intents import ACTION_ADD_TO_LIBRARY, ACTION_OPEN_IN_READER

            intents.handlers.setdefault(ACTION_OPEN_IN_READER, self.open_shared_in_reader)
            intents.handlers.setdefault(ACTION_ADD_TO_LIBRARY, self.add_shared_to_library)
        shell = getattr(app, "shell", None)
        if shell is not None and self._fallback_factory is None:
            self._fallback_factory = shell.screen_factory
            shell.screen_factory = self.screen_factory
        jobs = getattr(app, "jobs", None)
        on_transition = getattr(jobs, "on_transition", None) if jobs is not None else None
        if callable(on_transition) and not self._unsubs:
            try:
                self._unsubs.append(on_transition(self._on_transition))
            except Exception:
                log.exception("subscribing to job transitions failed")
        drawer = getattr(app, "drawer", None)
        if drawer is not None and hasattr(drawer, "register_search"):
            drawer.register_search("books", self.search_books)  # the drawer's unified search (UI_SPEC §1.3)
        self._wrap_lifecycle()

    # ---- drawer search ---------------------------------------------------------------------------

    async def search_books(self, query: str) -> list:
        """Drawer › Books: both shelves through the shared ``book_matches_query`` (off the UI loop)."""
        return await self.run_io(self._search_books_blocking, query)

    def _search_books_blocking(self, query: str) -> list:
        from glossarion_mobile.ui.shell.drawer import SEARCH_LIMIT, SearchHit

        service = self.service
        hits: list = []
        for shelf, label in (("progress", "In progress"), ("completed", "Completed")):
            for book in service.snapshot.shelf(shelf):
                if len(hits) >= SEARCH_LIMIT:
                    return hits
                if not service.matches_query(book, query):
                    continue
                bid = service.bid_for(book)
                kind = str(book.get("type") or book.get("workspace_kind") or "").upper()
                hits.append(SearchHit(
                    title=str(book.get("name") or "Book"),
                    subtitle=" · ".join(part for part in (label, kind) if part),
                    icon="MENU_BOOK",
                    open=lambda b=bid: self._navigate("library.book", {"bid": b}),
                    key=bid,
                ))
        return hits

    def _navigate(self, name: str, params: Optional[dict] = None) -> None:
        navigate = getattr(self.app, "navigate_to", None)
        if navigate is not None:
            navigate(name, params)

    def detach(self) -> None:
        for unsub in self._unsubs:
            try:
                unsub()
            except Exception:
                pass
        self._unsubs = []

    # ---- plumbing ----------------------------------------------------------------------------------

    async def run_io(self, fn: Callable[..., Any], *args: Any) -> Any:
        dispatcher = self.dispatcher
        if dispatcher is not None and getattr(dispatcher, "bound", False):
            return await dispatcher.run_in_thread(fn, *args, name="gl-library")
        return await asyncio.to_thread(fn, *args)

    def post(self, fn: Callable[..., Any], *args: Any) -> None:
        dispatcher = self.dispatcher
        if dispatcher is None or not getattr(dispatcher, "bound", False) or dispatcher.on_loop_thread():
            fn(*args)
            return
        dispatcher.post(fn, *args)

    def spawn(self, coro: Any) -> Any:
        dispatcher = self.dispatcher
        if dispatcher is not None and getattr(dispatcher, "bound", False):
            return dispatcher.spawn(coro)
        try:
            return asyncio.ensure_future(coro)
        except RuntimeError:
            coro.close()
            return None

    def _library_import(self, paths: Any, target: str = "raw", record_origins: bool = False) -> Any:
        """``FileBridge.library_import``: ``library_core.import_paths(copy_into_library=True)``."""
        return self.service.import_paths_blocking(list(paths), target, bool(record_origins))

    def foreground(self) -> bool:
        background = getattr(getattr(self.app, "jobs", None), "background", None)
        visible = getattr(background, "app_visible", None)
        if isinstance(visible, bool):
            return visible and self.foregrounded
        return self.foregrounded

    def _platform(self) -> str:
        platform = getattr(getattr(self.page, "platform", None), "value", None)
        return str(platform or "desktop")

    def _dark(self) -> bool:
        try:
            from glossarion_mobile.ui.theme import is_dark

            return bool(is_dark(self.page))
        except Exception:
            return False

    def _push_overlay(self, view: Any) -> None:
        shell = getattr(self.app, "shell", None)
        if shell is None:
            return
        shell.push_overlay(view)
        try:
            self.page.update()
        except Exception:
            pass

    def _pop_overlay(self) -> None:
        back = getattr(self.app, "back", None)
        if callable(back):
            back()

    def context(self) -> Any:
        from glossarion_mobile.ui import tokens
        from glossarion_mobile.ui.library.common import LibraryContext

        app = self.app
        shell = getattr(app, "shell", None)
        state = getattr(app, "state", None)
        scale = 1.0
        try:
            scale = effective_text_scale(state) if state is not None else 1.0  # app x system (U9)
        except Exception:
            scale = 1.0
        ctx = LibraryContext(
            service=self.service,
            page=self.page,
            dispatcher=self.dispatcher,
            navigate=getattr(app, "navigate_to", None),
            notify=getattr(app, "notify", None),
            files=getattr(app, "files", None),
            jobs=getattr(app, "jobs", None),
            prefs=getattr(app, "prefs", None),
            haptics=getattr(app, "haptics", None),
            shell=shell,
            intents=getattr(app, "intents", None),
            copy_text=getattr(app, "_copy_text", None),
            reader=lambda: getattr(app, "reader", None),
            glossary=lambda: getattr(app, "glossary_feature", None),
            push_overlay=self._push_overlay,
            pop_overlay=self._pop_overlay,
            foreground=self.foreground,
            platform=self._platform(),
            tablet=bool(getattr(shell, "tablet", False)),
            dark=self._dark(),
            text_scale=scale,
        )
        try:
            from glossarion_mobile.ui.theme import mono_family

            ctx.mono = mono_family(self.page)  # type: ignore[attr-defined]
        except Exception:
            ctx.mono = tokens.MONO_FAMILIES.get(self._platform(), "monospace")  # type: ignore[attr-defined]
        # U10 (Book page › Output: cloud status, Send now, Share file via link): the services install after the
        # Library, so the Output tab reads them late (ui/screens/cloud_sync.U10Actions)
        opener = getattr(app, "opener", None)
        clipboard = getattr(app, "clipboard", None)
        ctx.extras.update({
            "cloud_sync": lambda: getattr(app, "cloud_sync", None),
            "share_links": lambda: getattr(app, "share_links", None),
            "open_url": getattr(opener, "launch", None) if opener is not None else None,
            "read_clipboard": getattr(clipboard, "get", None) if clipboard is not None else None,
        })
        return ctx

    # ---- screens --------------------------------------------------------------------------------

    def make_screen(self, match: RouteMatch) -> Any:
        name = match.name
        ctx = self.context()
        if name == "library":
            from glossarion_mobile.ui.library.home import LibraryScreen

            return LibraryScreen(match, ctx)
        if name == "library.scan_raw":
            from glossarion_mobile.ui.library.scan_raw import ScanRawScreen

            files = getattr(self.app, "files", None)
            return ScanRawScreen(match, ctx, inbox_dir=getattr(files, "inbox_dir", None))
        if name == "library.book":
            from glossarion_mobile.ui.library.book_page import BookPageScreen

            return BookPageScreen(match, ctx)
        if name == "library.book.metadata":
            from glossarion_mobile.ui.library.metadata_editor import MetadataEditorScreen

            return MetadataEditorScreen(match, ctx)
        if name in ("tools.progress", "tools.progress.glossary"):
            return self.progress_tool_screen(match, ctx)
        return None

    def progress_tool_screen(self, match: RouteMatch, ctx: Any) -> Any:
        """Standalone Progress manager / Glossary progress (``/tools/progress?out=<fid>``): the Book page
        on that output folder, opened on its Chapters (Glossary) tab."""
        from glossarion_mobile.ui.library.book_page import BookPageScreen
        from glossarion_mobile.ui.router import parse_route

        out = match.get("out")
        tab = "glossary" if match.name == "tools.progress.glossary" else "chapters"
        if out:
            book_match = parse_route(f"/library/book/{out}?tab={tab}")
            if book_match is not None:
                screen = BookPageScreen(book_match, ctx)
                screen.match = match
                screen.title = "Glossary progress" if tab == "glossary" else "Progress manager"
                return screen
        from glossarion_mobile.ui.components.empty_state import EmptyState
        from glossarion_mobile.ui.screens.base import Screen

        feature = self
        route_name = match.name

        class _Pick(Screen):
            """No ``?out=``: the SourcePicker (Recent outputs · Library · Chat workspaces · Browse; UI_SPEC
            §4.2 / §4.3), then the chosen output folder's Book page (Chapters / Glossary)."""

            title = "Glossary progress" if tab == "glossary" else "Progress manager"
            picker: Any = None
            _opened = False

            def build_body(self_inner) -> Any:
                return EmptyState(icon="LIST_ALT", title=self_inner.title,
                                  body="Choose an output folder: a Library book, a recent output, a chat workspace "
                                       "or a file you browse to.",
                                  primary=("Choose a workspace…", lambda e: self_inner.open_picker()),
                                  key="progress-pick")

            def did_show(self_inner) -> None:
                if not self_inner._opened:
                    self_inner._opened = True
                    self_inner.open_picker()

            def open_picker(self_inner) -> Any:
                self_inner.picker = feature.open_progress_picker(route_name)
                return self_inner.picker

        return _Pick(match)

    def open_progress_picker(self, route_name: str = "tools.progress") -> Any:
        """The Tools SourcePicker for the Progress manager / Glossary progress (eligible: an output folder);
        the chosen folder becomes a Library-style row (like ChatFeature.open_progress) and its route id."""
        tools = getattr(self.app, "tools", None)
        context = getattr(tools, "context", None)
        if not callable(context):
            navigate = getattr(self.app, "navigate_to", None)
            if navigate is not None:
                navigate("library")
            return None
        from glossarion_mobile.ui.tools.source_picker import SourcePicker

        ctx = context()

        def chosen(targets: list) -> None:
            if targets:
                self.spawn(self.open_progress_target(targets[0], route_name))

        picker = SourcePicker(ctx, title="Open the Progress manager on…", multi=False,
                              eligible=lambda t: None if t.folder else "Needs an output folder",
                              on_done=chosen, segment="library", find_folder=True)
        if self.page is not None:
            picker.show(self.page)
        return picker

    async def open_progress_target(self, target: Any, route_name: str = "tools.progress") -> Optional[str]:
        """``tools.progress?out=<bid>`` for a picked ToolTarget (its output folder + raw source)."""
        service = self.service
        folder = str(getattr(target, "folder", "") or "")
        if not folder or service is None:
            return None
        source = str(getattr(target, "source", "") or "")
        bid = str(getattr(target, "bid", "") or "")
        if not bid:
            def row() -> dict:
                name = os.path.basename(os.path.normpath(folder))
                progress = os.path.join(folder, "translation_progress.json")
                book = {"name": name, "folder_name": name, "path": folder, "output_folder": folder,
                        "type": "in_progress", "is_in_progress": True, "in_library": False,
                        "progress_file": progress if os.path.isfile(progress) else ""}
                if source and os.path.isfile(source):
                    book["raw_source_path"] = source
                return book

            book = await self.run_io(row)
            bid = service.bid_for(book)
        navigate = getattr(self.app, "navigate_to", None)
        if navigate is not None:
            navigate(route_name, None, {"out": bid})
        return bid

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
        return screen

    # ---- intents ---------------------------------------------------------------------------------

    def open_shared_in_reader(self, imp: Any) -> Optional[str]:
        """IntentRouter "Open in Reader": the shared EPUB (an Inbox copy) at ``/reader/<bid>``."""
        imported = getattr(imp, "imported", None)
        path = getattr(imported, "path", None)
        if not path or not os.path.isfile(path):
            return None
        return self.context().open_reader(path=path)

    async def add_shared_to_library(self, imp: Any) -> Any:
        """IntentRouter "Add to Library": copy into Library/Raw + ``import_paths``, then offer "Open"."""
        imported = getattr(imp, "imported", None)
        files = getattr(self.app, "files", None)
        if imported is None or files is None:
            return None
        added = await self.run_io(files.add_to_library, imported.path)
        await self.service.refresh(reason="add to library")
        book = None
        for row in self.service.snapshot.all_books():
            raw = row.get("raw_source_path")
            if raw and os.path.normcase(os.path.abspath(str(raw))) == os.path.normcase(os.path.abspath(added.path)):
                book = row
                break
        notify = getattr(self.app, "notify", None)
        navigate = getattr(self.app, "navigate_to", None)
        if notify is not None:
            if book is not None and navigate is not None:
                bid = self.service.bid_for(book)
                notify(f"Added to Library: {added.name}", "Open", lambda: navigate("library.book", {"bid": bid}))
            else:
                notify(f"Added to Library: {added.name}", "Library",
                       (lambda: navigate("library")) if navigate is not None else None)
        return added

    # ---- jobs / lifecycle ------------------------------------------------------------------------

    def _on_transition(self, snap: Any, previous: Any) -> None:
        if not getattr(snap, "is_terminal", False) or previous is None:
            return
        job_id = getattr(snap, "id", None)
        if job_id in self._finished:
            return
        self._finished.add(job_id)
        if len(self._finished) > 500:
            self._finished = set(list(self._finished)[-100:])
        self.spawn(self.on_job_finished(snap))

    async def on_job_finished(self, snap: Any, outputs: Any = None) -> list:
        try:
            return await self.service.on_job_finished(snap, outputs)
        except Exception:
            log.exception("library on_job_finished failed")
            return []

    def _top_poller(self) -> Any:
        shell = getattr(self.app, "shell", None)
        screen = getattr(shell, "top_screen", None) if shell is not None else None
        return getattr(screen, "poller", None)

    def _wrap_lifecycle(self) -> None:
        page = self.page
        if page is None:
            return
        original = getattr(page, "on_app_lifecycle_state_change", None)
        if getattr(original, "_glossarion_library", False):
            return

        async def on_lifecycle(e: Any) -> None:
            state = getattr(e, "state", None)
            name = str(getattr(state, "value", state))
            if name in _BACKGROUND_STATES:
                self.foregrounded = False
            elif name in _RESUME_STATES:
                self.foregrounded = True
                poller = self._top_poller()
                if poller is not None and hasattr(poller, "poke"):
                    poller.poke()
            if original is not None:
                result = original(e)
                if hasattr(result, "__await__"):
                    await result

        on_lifecycle._glossarion_library = True  # type: ignore[attr-defined]
        page.on_app_lifecycle_state_change = on_lifecycle

    # ---- test helpers --------------------------------------------------------------------------------

    def fake_import(self, path: str) -> Any:
        """``IntentImport``-shaped object for a local file (tests, the self-test)."""
        return SimpleNamespace(imported=SimpleNamespace(path=path, name=os.path.basename(path),
                                                        extension=os.path.splitext(path)[1].lower()))
