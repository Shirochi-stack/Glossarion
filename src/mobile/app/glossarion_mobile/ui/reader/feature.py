"""ReaderFeature: wires the U5 Reader into GlossarionApp.

``await ReaderFeature.install(app)``, called in ``GlossarionApp.start`` after
``_install_chat()`` (it reads the chat feature for "Ask in chat") and after the
Library feature (whose ``LibraryService.book_for_bid`` resolves route ids):

1. wraps the shell's screen factory for ``/reader/<bid>`` (``ReaderScreen``);
   every other route falls through;
2. owns one ``ReaderServer`` (127.0.0.1, random port, token paths), started
   lazily on the first open and stopped by ``detach``; its event fallback is
   routed to the Reader on screen;
3. offers ``open_book`` / ``open_file`` for the Library, the Book page, job
   cards and the IntentRouter's "Open in Reader": they register the target
   (Prefs file ref → opaque id; the book row or file is passed in-process,
   never in the route) and navigate to ``/reader/<bid>[?ch=&mode=]``;
4. marks the ``reader`` route implemented (``IMPLEMENTED_ROUTES``).

No desktop logic lives here: the Reader binds to ``reader_doc``,
``reader_overlay``, ``workspace_reader``, ``live_stream`` and ``library_core``.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Mapping, Optional

from glossarion_mobile.ui.router import RouteMatch

__all__ = ["IMPLEMENTED_ROUTES", "ReaderFeature", "SCREEN_ROUTES"]

log = logging.getLogger("glossarion.reader")

SCREEN_ROUTES = ("reader",)
MAX_PENDING = 64  # in-process open requests kept (book rows / files by route id)
IMPLEMENTED_ROUTES = frozenset(SCREEN_ROUTES)


class ReaderFeature:
    def __init__(self, app: Any, *, server_factory: Optional[Callable[..., Any]] = None,
                 core: Any = None) -> None:
        self.app = app
        self.page = getattr(app, "page", None)
        self._server_factory = server_factory
        self._core = core
        self.server: Any = None
        self._fallback_factory: Optional[Callable[[RouteMatch], Any]] = None
        self.pending: dict = {}  # bid -> what the id resolves to in-process (book row / file path)
        # bid -> one open request's arguments (chapter file name / raw_only / resume): consumed by the
        # screen it opens, so a later route-only open (deep link, notification) never reuses them.
        self.once: dict = {}
        self.screens: list = []
        self.active: Any = None

    # ---- install --------------------------------------------------------------------------

    @classmethod
    async def install(cls, app: Any, **kwargs: Any) -> "ReaderFeature":
        feature = cls(app, **kwargs)
        feature.attach()
        return feature

    def attach(self) -> None:
        app = self.app
        app.reader = self
        shell = getattr(app, "shell", None)
        if shell is not None and self._fallback_factory is None:
            self._fallback_factory = shell.screen_factory
            shell.screen_factory = self.screen_factory

    def detach(self) -> None:
        server, self.server = self.server, None
        if server is not None:
            try:
                server.stop()
            except Exception:
                pass

    # ---- shared services ----------------------------------------------------------------------

    @property
    def core(self) -> Any:
        if self._core is None:
            library = self.library
            core = getattr(library, "core", None)
            if core is None:
                from glossarion_mobile.services.library import SharedCore

                core = SharedCore()
            self._core = core
        return self._core

    @property
    def library(self) -> Any:
        app = self.app
        for name in ("library", "library_service"):
            candidate = getattr(app, name, None)
            if candidate is not None and hasattr(candidate, "book_for_bid"):
                return candidate
        for name in ("library_feature", "library"):
            feature = getattr(app, name, None)
            service = getattr(feature, "service", None)
            if service is not None and hasattr(service, "book_for_bid"):
                return service
        return None

    def ensure_server(self) -> Any:
        if self.server is not None and getattr(self.server, "running", True):
            return self.server
        factory = self._server_factory
        if factory is None:
            from glossarion_mobile.services.reader_server import ReaderServer

            factory = ReaderServer
        try:
            server = factory(on_event=self._on_server_event)
            server.start()
        except Exception:
            log.exception("the reader server could not start; the native reader is used")
            return None
        self.server = server
        return server

    def _on_server_event(self, payload: dict) -> None:
        screen = self.active
        if screen is not None and not getattr(screen, "disposed", True):
            screen.on_http_event(payload)

    # ---- resolving ids ----------------------------------------------------------------------------

    def resolve(self, bid: str) -> Optional[dict]:
        """Route id → ``{"book": row}`` or ``{"path": file}`` (blocking; the Reader calls it on the io pool)."""
        pending = self.pending.get(bid) or {}
        if pending.get("book"):
            return {"book": dict(pending["book"])}
        if pending.get("path"):
            return {"path": pending["path"]}
        library = self.library
        if library is not None:
            try:
                book = library.book_for_bid(bid)
            except Exception:
                log.exception("resolving book %s failed", bid)
                book = None
            if book:
                return {"book": dict(book)}
        prefs = getattr(self.app, "prefs", None)
        if prefs is not None and hasattr(prefs, "resolve_file_ref"):
            try:
                path = prefs.resolve_file_ref(bid)
            except Exception:
                path = None
            if path:
                return {"path": path}
        return None

    def has_book_page(self, bid: str) -> bool:
        library = self.library
        return library is not None and bool(bid) and bid not in {k for k, v in self.pending.items() if v.get("path")}

    def _register(self, *, book: Optional[Mapping[str, Any]] = None, path: Optional[str] = None) -> Optional[str]:
        library = self.library
        if book is not None and library is not None and hasattr(library, "bid_for"):
            try:
                return library.bid_for(book)
            except Exception:
                log.exception("registering the book failed")
        target = path or (str(book.get("output_folder") or book.get("path") or "") if book else "")
        if not target:
            return None
        prefs = getattr(self.app, "prefs", None)
        if prefs is not None and hasattr(prefs, "file_ref"):
            try:
                return prefs.file_ref(os.path.abspath(target), kind="book")
            except Exception:
                log.exception("registering %s failed", target)
        from glossarion_mobile.state.prefs import file_ref_id

        return file_ref_id(target)

    # ---- opening -----------------------------------------------------------------------------------

    def open_book(self, book: Optional[Mapping[str, Any]] = None, *, path: Optional[str] = None,
                  chapter: Optional[int] = None, chapter_filename: Optional[str] = None, mode: Optional[str] = None,
                  raw_only: bool = False, resume: bool = False) -> Optional[str]:
        """Open the Reader on a Library row or a file (no Library row needed); returns the route id.

        ``resume`` (the card's ▶ Continue, the Overview's "Continue · Ch N · P%") opens at the
        saved reading position instead of offering it."""
        bid = self._register(book=book, path=path)
        if not bid:
            notify = getattr(self.app, "notify", None)
            if notify is not None:
                notify("This book could not be opened in the Reader")
            return None
        args: dict = {}
        if book is not None:
            args["book"] = dict(book)
        if path:
            args["path"] = os.path.abspath(path)
        once: dict = {"raw_only": bool(raw_only)}
        if chapter_filename:
            once["chapter_filename"] = str(chapter_filename)
        if resume:
            once["resume"] = True
        self.pending.pop(bid, None)
        self.pending[bid] = args
        self.once[bid] = once
        while len(self.pending) > MAX_PENDING:
            self.pending.pop(next(iter(self.pending)))
        while len(self.once) > MAX_PENDING:
            self.once.pop(next(iter(self.once)))
        query: dict = {}
        if chapter is not None:
            query["ch"] = int(chapter)
        if mode in ("translated", "original", "bilingual"):
            query["mode"] = mode
        navigate = getattr(self.app, "navigate_to", None)
        if navigate is not None:
            navigate("reader", {"bid": bid}, query or None)
        return bid

    def open_file(self, path: str) -> Optional[str]:
        """IntentRouter "Open in Reader" for a shared EPUB (imported into the Inbox)."""
        return self.open_book(path=path)

    # ---- screens ------------------------------------------------------------------------------------

    def _deps(self) -> Any:
        from glossarion_mobile.ui.reader.reader_view import ReaderDeps, webview_supported

        app = self.app
        page = self.page
        paths = getattr(app, "paths", None)
        cache_dir = None
        if paths is not None and getattr(paths, "cache", None):
            cache_dir = os.path.join(os.fspath(paths.cache), "reader")
        chat_view = getattr(app, "chat_view", None)
        files = getattr(app, "files", None)
        extras: dict = {}
        try:
            from glossarion_mobile.ui.theme import mono_family

            extras["mono_family"] = mono_family(page)
        except Exception:
            pass
        share = getattr(files, "share", None) if files is not None else None  # FileBridge.share(paths)
        if share is not None:
            extras["share_files"] = share
        opener = getattr(app, "opener", None)
        state = getattr(app, "state", None)
        server = self.ensure_server() if webview_supported(page) else None
        # The jobs feature owns the one platform wakelock (reference-counted, services.wakelock).
        wakelock_owner = getattr(getattr(app, "jobs", None), "wakelock_owner", None)
        wakelock = wakelock_owner.holder("reader") if wakelock_owner is not None else None
        return ReaderDeps(
            page=page,
            dispatcher=getattr(app, "dispatcher", None),
            prefs=getattr(app, "prefs", None),
            config=getattr(app, "config_store", None),
            jobs=getattr(app, "job_service", None),
            resolve=self.resolve,
            server=server,
            core=self.core,
            notify=getattr(app, "notify", None),
            navigate_to=getattr(app, "navigate_to", None),
            go_back=self._go_back,
            copy_text=getattr(app, "_copy_text", None),
            open_url=getattr(opener, "launch", None),
            haptics=getattr(app, "haptics", None),
            cache_dir=cache_dir,
            width_signal=getattr(state, "width", None),
            ask_in_chat=self._ask_in_chat if chat_view is not None else None,
            has_book_page=self.has_book_page,
            webview_ok=lambda: webview_supported(page) and self.server is not None,
            wakelock=wakelock,
            extras=extras,
        )

    def _go_back(self) -> None:
        back = getattr(self.app, "back", None)
        if callable(back):
            back()

    def _ask_in_chat(self, text: str) -> None:
        """"Ask in chat": a fresh chat with the quoted selection as its draft."""
        app = self.app
        chat_view = getattr(app, "chat_view", None)
        try:
            env = getattr(chat_view, "env", None)
            chats = getattr(env, "chats", None)
            if chats is not None and getattr(chat_view, "bound", False):
                cid = chats.new_chat()
                open_chat = getattr(app, "_open_chat", None)
                if open_chat is not None:
                    open_chat(cid)
                if chat_view.cid != cid:
                    chat_view.load_chat(cid)
        except Exception:
            log.exception("creating a chat for the selection failed")
        prefill = getattr(app, "prefill_composer", None)
        if prefill is not None:
            prefill(text)

    def make_screen(self, match: RouteMatch) -> Any:
        from glossarion_mobile.ui.reader.reader_view import ReaderScreen

        bid = match.params.get("bid", "")
        args = dict(self.pending.get(bid) or {})
        args.update(self.once.pop(bid, None) or {})
        screen = ReaderScreen(match, self._deps(), args=args)
        self.active = screen
        self.screens.append(bid)
        del self.screens[:-20]
        return screen

    def screen_factory(self, match: RouteMatch) -> Any:
        if match.name in SCREEN_ROUTES:
            try:
                return self.make_screen(match)
            except Exception:
                log.exception("building the reader failed")
        if self._fallback_factory is None:
            raise LookupError(f"no screen for {match.name}")
        return self._fallback_factory(match)
