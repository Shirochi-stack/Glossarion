"""What a tool runs on: ``ToolTarget`` rows for the SourcePicker (UI_SPEC §4.2, §5.5).

A target pairs a translation **output folder** with its **raw source** file (either may
be missing). The rows come from the Library (``LibraryService`` snapshot: the shared
``library_core`` scan already resolved every book's output folder and raw source), from
the Direct Text workspace folder (chat attachments) or from a file picked with Browse
(its output folder found with the QA Scanner's own auto-search,
``qa_scan_runtime.automatic_qa_output_candidates``). Nothing here guesses a book's files
itself: the paths are the shared resolvers' answers.

Pure Python (no Flet); every function that touches the disk is called on the io pool.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

__all__ = [
    "HTML_EXTENSIONS",
    "LIBRARY_SORT",
    "SOURCE_EXTENSIONS",
    "ToolTarget",
    "chat_workspace_targets",
    "folder_has_html",
    "folder_has_scan_files",
    "library_targets",
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


def recent_output_targets(service: Any, limit: int = 50, library_rows: Optional[Sequence[ToolTarget]] = None) -> list:
    """Translation workspaces (the Library's In progress shelf), newest first.

    ``library_rows`` (``library_targets(service)``) avoids resolving every book twice.
    """
    snapshot = getattr(service, "snapshot", None)
    rows: Sequence = ()
    if snapshot is not None:
        rows = tuple(getattr(snapshot, "in_progress", ()) or ())
    out: list = []
    seen: set = set()
    if library_rows is not None:
        wanted = {_norm(b.get("output_folder")) for b in rows if b.get("output_folder")}
        for target in library_rows:
            if target.folder and _norm(target.folder) in wanted and target.key not in seen:
                seen.add(target.key)
                out.append(replace(target, origin="recent"))
    else:
        for book in rows:
            target = _book_target(service, book, "recent")
            if target is None or not target.folder or target.key in seen:
                continue
            seen.add(target.key)
            out.append(target)
    out.sort(key=lambda t: t.mtime, reverse=True)
    return out[:limit]


def chat_workspace_targets(chats_root: str, *, depth: int = 3, limit: int = 200) -> list:
    """Folders under ``<Output>/Direct Text`` that hold a translation workspace."""
    out: list = []
    if not chats_root or not os.path.isdir(chats_root):
        return out
    root_depth = os.path.abspath(chats_root).rstrip("/\\").count(os.sep)
    for current, dirs, files in os.walk(chats_root):
        level = os.path.abspath(current).rstrip("/\\").count(os.sep) - root_depth
        if level >= depth:
            dirs[:] = []
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        if "translation_progress.json" in files or any(f.lower().endswith(HTML_EXTENSIONS) for f in files):
            if os.path.abspath(current) == os.path.abspath(chats_root):
                continue
            try:
                mtime = os.path.getmtime(current)
            except OSError:
                mtime = 0.0
            rel = os.path.relpath(current, chats_root)
            out.append(ToolTarget(title=rel.replace(os.sep, " › "), folder=os.path.abspath(current),
                                  origin="chat", mtime=mtime, kind=workspace_kind(current), direct_text=True))
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
