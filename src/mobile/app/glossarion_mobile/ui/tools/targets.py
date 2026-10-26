"""What a tool runs on: ``ToolTarget`` rows for the SourcePicker (UI_SPEC §4.2, §5.5).

A target pairs a translation **output folder** with its **raw source** file (either may
be missing). The rows come from the Library (``LibraryService`` snapshot: the shared
``library_core`` scan already resolved every book's output folder and raw source), from
the Direct Text workspace folder (chat books: ``<chat>/Attachments/<book>``, their raw file
found by the same shared resolvers, ``LibraryService.raw_source``: the translator's
``source_epub.txt`` names the attached file) or from a file picked with Browse
(its output folder found with the QA Scanner's own auto-search,
``qa_scan_runtime.automatic_qa_output_candidates``). Nothing here guesses a book's files
itself: the paths are the shared resolvers' answers.

The Tools pickers list one row per book (``merge_book_rows``: a Library book and a chat book
with the same output folder or raw file are one row, ``found_in`` says where it was found) and
the recently translated workspaces of both (``recent_output_targets``).

Pure Python (no Flet); every function that touches the disk is called on the io pool.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

__all__ = [
    "ATTACHMENTS_DIR",
    "CHAT_MESSAGES_DIR",
    "HTML_EXTENSIONS",
    "LIBRARY_SORT",
    "SOURCE_EXTENSIONS",
    "ToolTarget",
    "chat_workspace_targets",
    "folder_has_html",
    "folder_has_scan_files",
    "found_in",
    "last_activity",
    "library_targets",
    "merge_book_rows",
    "order_library_rows",
    "recent_output_targets",
    "target_for_book",
    "target_for_source",
    "target_key",
    "target_matches",
    "workspace_kind",
]

SOURCE_EXTENSIONS = (".epub", ".txt", ".pdf", ".md", ".html", ".htm", ".xhtml")
HTML_EXTENSIONS = (".html", ".xhtml", ".htm")
#: A chat's folders under ``<Output>/Direct Text/<chat>/`` (``direct_text_store``): the message bodies
#: (``Chat Messages``: the chat's text replies, never a book) and the attached books' workspaces.
CHAT_MESSAGES_DIR = "Chat Messages"
ATTACHMENTS_DIR = "Attachments"
PROGRESS_FILE = "translation_progress.json"
#: The in-chat Library picker's order: the Library's Date sort, newest first
#: (``library_core.sort_books`` SORT_DATE; owner decision 2026-10-08).
LIBRARY_SORT = "date"


def _norm(path: Any) -> str:
    try:
        return os.path.normcase(os.path.abspath(os.fspath(path)))
    except Exception:
        return str(path)


@dataclass(frozen=True)
class ToolTarget:
    """One pickable row: an output folder and/or a raw source file."""

    title: str
    folder: str = ""  # translation output folder ("" when not translated yet)
    source: str = ""  # raw source file ("" when unknown)
    origin: str = "library"  # library | recent | chat | browse
    bid: str = ""  # Library route id (opaque) when the row is a Library book
    mtime: float = 0.0
    kind: str = ""  # workspace / source kind: epub | pdf | txt | other
    direct_text: bool = False  # a Direct Text (chat) workspace: never QA-scanned
    extra: Mapping[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return target_key(self)

    @property
    def folder_name(self) -> str:
        return os.path.basename(self.folder.rstrip("/\\")) if self.folder else ""

    @property
    def source_name(self) -> str:
        return os.path.basename(self.source) if self.source else ""

    @property
    def is_epub_source(self) -> bool:
        return self.source.lower().endswith(".epub")

    def with_source(self, source: str) -> "ToolTarget":
        return replace(self, source=os.path.abspath(source) if source else "")

    def to_param(self) -> dict:
        """JSON form for a ``JobSpec`` (``params["targets"]``)."""
        return {"folder": self.folder or None, "source": self.source or None}


def target_key(target: ToolTarget) -> str:
    if target.folder:
        return _norm(target.folder)
    if not target.source:
        # An unresolved Library row (``library_targets(include_unresolved=True)``: neither an output
        # folder nor a raw file): its Library id, else its own file - never the shared cwd key.
        if target.bid:
            return "bid:" + target.bid
        book = target.extra.get("book") if isinstance(target.extra, Mapping) else None
        path = str((book or {}).get("path") or "")
        if path:
            return "path:" + _norm(path)
    return "src:" + _norm(target.source)


def folder_has_html(folder: str) -> bool:
    try:
        return any(name.lower().endswith(HTML_EXTENSIONS) for name in os.listdir(folder))
    except OSError:
        return False


def folder_has_scan_files(folder: str, text_mode: bool = False) -> bool:
    """The QA Scanner's folder check: HTML/XHTML files (text mode: also ``.txt``)."""
    try:
        names = os.listdir(folder)
    except OSError:
        return False
    for name in names:
        lower = name.lower()
        if lower.endswith(HTML_EXTENSIONS) or (text_mode and lower.endswith(".txt")):
            return True
    return False


def workspace_kind(folder: str, source: str = "", book: Optional[Mapping[str, Any]] = None) -> str:
    """``epub`` / ``pdf`` / ``txt`` / ``other``: the shared ``library_core`` classifiers.

    ``workspace_compile_kind`` (what the desktop Library compiles: PDF workspaces compile
    to PDF) wins for ``pdf``; ``_detect_workspace_kind`` fills in the rest.
    """
    try:
        import library_core
    except Exception:
        library_core = None  # type: ignore[assignment]
    if library_core is not None and folder:
        compile_kind = getattr(library_core, "workspace_compile_kind", None) or getattr(
            library_core, "_workspace_compile_kind", None)
        if callable(compile_kind):
            try:
                if compile_kind(dict(book or {}), folder) == "pdf":
                    return "pdf"
            except Exception:
                pass
        detect = getattr(library_core, "_detect_workspace_kind", None)
        if callable(detect):
            try:
                return str(detect(folder, source or "") or "other")
            except Exception:
                pass
    low = (source or "").lower()
    for ext, kind in ((".epub", "epub"), (".pdf", "pdf"), (".txt", "txt")):
        if low.endswith(ext):
            return kind
    return "other"


def _is_direct_text(path: str) -> bool:
    try:
        import qa_scan_runtime

        return bool(qa_scan_runtime.is_direct_text_qa_path(path))
    except Exception:
        return False


def _book_target(service: Any, book: Mapping[str, Any], origin: str, *,
                 include_unresolved: bool = False) -> Optional[ToolTarget]:
    folder = str(book.get("output_folder") or "")
    workspace_for = getattr(service, "workspace_for", None)
    if callable(workspace_for):  # an organized Library/Translated book keeps its workspace (C3)
        try:
            folder = str(workspace_for(book) or "") or folder
        except Exception:
            pass
    if folder and not os.path.isdir(folder):
        folder = ""
    source = ""
    raw_source = getattr(service, "raw_source", None)
    if callable(raw_source):
        try:
            source = str(raw_source(book) or "")
        except Exception:
            source = ""
    if not source:
        path = str(book.get("raw_source_path") or "")
        source = path if path and os.path.isfile(path) else ""
    kind_source = source
    if not folder and not source:
        if not include_unresolved:
            return None
        kind_source = str(book.get("path") or "")  # e.g. a Library/Translated EPUB without a raw
    bid = ""
    bid_for = getattr(service, "bid_for", None)
    if callable(bid_for):
        try:
            bid = str(bid_for(book) or "")
        except Exception:
            bid = ""
    try:
        mtime = float(book.get("mtime") or 0.0)
    except (TypeError, ValueError):
        mtime = 0.0
    if not mtime and folder:
        try:
            mtime = os.path.getmtime(folder)
        except OSError:
            mtime = 0.0
    title = str(book.get("name") or book.get("folder_name") or os.path.basename(folder or source or kind_source))
    # ``extra["book"]``: the scanned Library row (the in-chat picker searches it with the shared
    # ``book_matches_query`` and renders it as the Library's own card).
    return ToolTarget(title=title, folder=folder, source=source, origin=origin, bid=bid, mtime=mtime,
                      kind=workspace_kind(folder, source, book) if folder else workspace_kind("", kind_source),
                      direct_text=bool(folder) and _is_direct_text(folder),
                      extra={"type": str(book.get("type") or ""), "book": dict(book)})


def target_for_book(service: Any, book: Mapping[str, Any], origin: str = "library") -> Optional[ToolTarget]:
    """The ToolTarget of one Library row (None when it has neither output folder nor raw source)."""
    return _book_target(service, book, origin)


def library_targets(service: Any, *, include_unresolved: bool = False) -> list:
    """Every Library book (In progress, then Completed) that has an output folder or a raw source.

    ``include_unresolved`` also lists the rows that have neither (a Library/Translated EPUB whose raw
    is unknown), keyed by their Library id: the in-chat picker shows them disabled with a reason
    instead of hiding them. The Tools pickers keep the default."""
    snapshot = getattr(service, "snapshot", None)
    books: Iterable = ()
    if snapshot is not None:
        try:
            books = snapshot.all_books()
        except Exception:
            books = ()
    out: list = []
    seen: set = set()
    for book in books or ():
        target = _book_target(service, book, "library", include_unresolved=include_unresolved)
        if target is None or target.key in seen:
            continue
        seen.add(target.key)
        out.append(target)
    return out


def target_matches(service: Any, target: ToolTarget, query: str) -> bool:
    """A search query against a row: a Library row's scanned book through the shared
    ``book_matches_query`` (``LibraryService.matches_query``: titles, raw titles, tags), any other
    row by its title / folder / file name."""
    if not str(query or "").strip():
        return True
    book = target.extra.get("book") if isinstance(target.extra, Mapping) else None
    matches = getattr(service, "matches_query", None) if service is not None else None
    if isinstance(book, Mapping) and callable(matches):
        return bool(matches(book, query.strip()))
    return query.strip().casefold() in " ".join((target.title, target.folder_name, target.source_name)).casefold()


def order_library_rows(service: Any, rows: Sequence[ToolTarget], query: str = "", *,
                       sort: str = LIBRARY_SORT) -> list:
    """Library rows filtered by ``query`` and in the Library's order (default Date: newest first).

    The Library home's own ``models.visible_books`` (shared ``book_matches_query`` +
    ``LibraryService.sort_books``) over the rows' scanned books (``extra["book"]``); rows without
    one follow, in their order, when their title matches. The in-chat Library picker and
    ``/library <title>`` use it. Pure; safe on the io pool."""
    from glossarion_mobile.services.library import book_key
    from glossarion_mobile.ui.library.models import FilterState, visible_books

    query = str(query or "").strip()
    buckets: dict = {}
    books: list = []
    loose: list = []
    for target in rows:
        book = target.extra.get("book") if isinstance(target.extra, Mapping) else None
        if not isinstance(book, Mapping):
            loose.append(target)
            continue
        buckets.setdefault(book_key(book), []).append(target)
        books.append(book)
    # visible_books passes ``reverse`` positionally; LibraryService.sort_books takes it by keyword
    # (the Library home's adapter: the method itself raises TypeError there).
    ordered = visible_books(books, FilterState(query=query, sort=sort), matches=service.matches_query,
                            format_of=service.format_of,
                            sort=lambda bs, mode, rev: service.sort_books(bs, mode, reverse=rev))
    out: list = []
    for book in ordered:
        bucket = buckets.get(book_key(book))
        if bucket:
            out.append(bucket.pop(0))
    out.extend(t for t in loose if target_matches(service, t, query))
    return out


def found_in(target: ToolTarget) -> tuple:
    """Where a row's book was found: ``("library",)``, ``("chat",)`` or both (``merge_book_rows``)."""
    extra = target.extra if isinstance(target.extra, Mapping) else {}
    found = tuple(extra.get("found_in") or ())
    if found:
        return found
    return (target.origin,) if target.origin in ("library", "chat") else ()


def merge_book_rows(library_rows: Sequence[ToolTarget], chat_rows: Sequence[ToolTarget] = ()) -> list:
    """One row per book: the Library rows (in their order), then the chat books that are not one of them.

    A chat book is the same book as a row before it when both have the same output folder (a Library
    row whose workspace is the chat's) or the same raw file (the Library book the chat translated
    again; the shared resolvers give both the same path). The earlier row stays and records the chat
    in ``extra["found_in"]`` (``("library", "chat")``) and every workspace of the book in
    ``extra["workspaces"]`` (``last_activity``); a Library row without a workspace takes the chat's.
    Pure; the keys stay ``target_key`` (folder first), so Recent outputs and the Books segment name a
    book alike."""
    out: list = list(library_rows or ())
    by_folder: dict = {}
    by_source: dict = {}

    def index(position: int) -> None:
        row = out[position]
        if row.folder:
            by_folder.setdefault(_norm(row.folder), position)
        if row.source:
            by_source.setdefault(_norm(row.source), position)

    for position in range(len(out)):
        index(position)
    for chat in chat_rows or ():
        position = by_folder.get(_norm(chat.folder)) if chat.folder else None
        if position is None and chat.source:
            position = by_source.get(_norm(chat.source))
        if position is None:
            out.append(chat)
            index(len(out) - 1)
            continue
        row = out[position]
        found = found_in(row)
        found = found + tuple(f for f in found_in(chat) if f not in found)
        workspaces = _workspaces(row)
        workspaces += tuple(w for w in _workspaces(chat) if _norm(w) not in {_norm(x) for x in workspaces})
        changes: dict = {"extra": {**dict(row.extra or {}), "found_in": found, "workspaces": workspaces}}
        if not row.folder and chat.folder:
            changes.update(folder=chat.folder, kind=chat.kind or row.kind, direct_text=chat.direct_text)
        if not row.source and chat.source:
            changes["source"] = chat.source
        out[position] = replace(row, **changes)
        index(position)
    return out


def _workspaces(target: ToolTarget) -> tuple:
    """Every output folder of the row's book (``merge_book_rows``), else its own folder."""
    extra = target.extra if isinstance(target.extra, Mapping) else {}
    found = tuple(str(w) for w in extra.get("workspaces") or () if w)
    return found or ((target.folder,) if target.folder else ())


def last_activity(target: ToolTarget) -> float:
    """When the row's book was last translated into: the newest ``translation_progress.json`` (written on
    every chapter) of its workspaces (``merge_book_rows``: a Library book the chat translated again),
    else the folder, else the row's own time."""
    times: list = []
    for folder in _workspaces(target):
        for path in (os.path.join(folder, PROGRESS_FILE), folder):
            try:
                times.append(os.path.getmtime(path))
                break
            except OSError:
                continue
    return max(times) if times else float(target.mtime or 0.0)


def _not_started(service: Any, target: ToolTarget) -> bool:
    """A Library row whose card says "Not started" (the shared ``card_progress_view`` state the scan kept
    in ``ScanSnapshot.views``): an import with an empty workspace, nothing translated yet."""
    book = target.extra.get("book") if isinstance(target.extra, Mapping) else None
    if not isinstance(book, Mapping) or service is None:
        return False
    view = None
    views = getattr(getattr(service, "snapshot", None), "views", None)
    if isinstance(views, Mapping) and views:
        from glossarion_mobile.services.library import book_key

        view = views.get(book_key(book))
    if view is None and callable(getattr(service, "card_view", None)):
        try:
            view = service.card_view(book)
        except Exception:
            view = None
    return isinstance(view, Mapping) and view.get("state") == "not_started"


def recent_output_targets(service: Any, limit: int = 50, library_rows: Optional[Sequence[ToolTarget]] = None,
                          chat_rows: Sequence[ToolTarget] = ()) -> list:
    """The recently translated workspaces, newest first: every Library book with a workspace (In progress
    and Completed - a book whose translation just finished is on the Completed shelf) and every chat book
    (``chat_workspace_targets``), one row per book (``merge_book_rows``), ordered by ``last_activity``;
    Library imports nobody translated yet ("Not started") stay out.

    ``library_rows`` (``library_targets(service)``) avoids resolving every book twice.
    """
    if library_rows is None:
        library_rows = library_targets(service)
    out: list = []
    seen: set = set()
    for target in merge_book_rows(library_rows, chat_rows):
        if not target.folder or target.key in seen or _not_started(service, target):
            continue
        seen.add(target.key)
        out.append(replace(target, origin="recent", mtime=last_activity(target),
                           extra={**dict(target.extra or {}), "found_in": found_in(target)}))
    out.sort(key=lambda t: t.mtime, reverse=True)
    return out[:limit]


def _workspace_source(service: Any, folder: str) -> str:
    """The raw file of a chat book's workspace: ``LibraryService.raw_source`` over a workspace row (the
    shared ``library_core`` resolvers: origins, the raw-inputs registry, Library/Raw, the translator's
    ``source_epub.txt`` naming the attached file - what ``ChatFeature.open_reader`` asks too); '' when
    nothing resolves or there is no service."""
    raw_source = getattr(service, "raw_source", None) if service is not None else None
    if not callable(raw_source):
        return ""
    name = os.path.basename(os.path.normpath(folder))
    progress = os.path.join(folder, PROGRESS_FILE)
    row = {"name": name, "folder_name": name, "path": folder, "output_folder": folder, "type": "in_progress",
           "is_in_progress": True, "in_library": False, "progress_file": progress if os.path.isfile(progress) else ""}
    try:
        source = str(raw_source(row) or "")
    except Exception:
        return ""
    return source if source and os.path.isfile(source) else ""


def chat_workspace_targets(chats_root: str, *, depth: int = 3, limit: int = 200, service: Any = None) -> list:
    """The chat books under ``<Output>/Direct Text``: the folders below a chat's own folder that hold a
    translation workspace (``<chat>/Attachments/<book>``, the desktop's run-style attachment trees),
    newest first. A chat's own folder (its ``Direct Text N`` replies) and its ``Chat Messages`` are the
    chat's text messages, never a book. With ``service`` each row gets its raw file
    (``_workspace_source``)."""
    out: list = []
    if not chats_root or not os.path.isdir(chats_root):
        return out
    root_depth = os.path.abspath(chats_root).rstrip("/\\").count(os.sep)
    for current, dirs, files in os.walk(chats_root):
        level = os.path.abspath(current).rstrip("/\\").count(os.sep) - root_depth
        if level >= depth:
            dirs[:] = []
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != CHAT_MESSAGES_DIR]
        if level < 2 or os.path.basename(current) == ATTACHMENTS_DIR:
            continue  # Direct Text itself, a chat's own folder, its Attachments folder
        if PROGRESS_FILE in files or any(f.lower().endswith(HTML_EXTENSIONS) for f in files):
            try:
                mtime = os.path.getmtime(current)
            except OSError:
                mtime = 0.0
            folder = os.path.abspath(current)
            source = _workspace_source(service, folder)
            chat = os.path.relpath(current, chats_root).split(os.sep)[0]
            out.append(ToolTarget(title=os.path.basename(folder), folder=folder, source=source, origin="chat",
                                  mtime=mtime, kind=workspace_kind(current, source), direct_text=True,
                                  extra={"found_in": ("chat",), "chat": chat}))
            dirs[:] = []
            if len(out) >= limit:
                break
    out.sort(key=lambda t: t.mtime, reverse=True)
    return out


def target_for_source(source: str, *, output_root: Optional[str] = None, current_dir: Optional[str] = None,
                      script_dir: Optional[str] = None, text_mode: bool = False,
                      candidates: Optional[Callable[..., Sequence[str]]] = None) -> ToolTarget:
    """A picked source file and the output folder the QA Scanner's auto-search finds for it.

    ``qa_scan_runtime.automatic_qa_output_candidates`` in translator write priority; the first
    candidate holding scannable files wins (desktop ``run_qa_scan`` auto-search; text mode for
    ``.txt`` / ``.pdf`` sources).
    """
    source = os.path.abspath(source)
    if candidates is None:
        try:
            import qa_scan_runtime

            candidates = qa_scan_runtime.automatic_qa_output_candidates
        except Exception:
            candidates = None
    folder = ""
    if candidates is not None:
        text = text_mode or source.lower().endswith((".txt", ".pdf"))
        try:
            found = candidates(source, current_dir=current_dir or os.getcwd(),
                               script_dir=script_dir or os.getcwd(), output_root=output_root)
        except Exception:
            found = []
        for candidate in found or ():
            if os.path.isdir(candidate) and folder_has_scan_files(candidate, text):
                folder = os.path.abspath(candidate)
                break
    try:
        mtime = os.path.getmtime(source)
    except OSError:
        mtime = time.time()
    title = os.path.splitext(os.path.basename(source))[0]
    return ToolTarget(title=title, folder=folder, source=source, origin="browse", mtime=mtime,
                      kind=workspace_kind(folder, source) if folder else workspace_kind("", source))
