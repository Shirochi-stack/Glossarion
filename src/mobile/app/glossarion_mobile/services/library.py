"""LibraryService: the mobile Library's data layer over the shared GUI-free cores (plan §5 Library, UI_SPEC §3).

Nothing here re-implements desktop Library or Progress Manager logic. Every scan,
count, status, path decision and file mutation is a call into the shared modules
the desktop ``epub_library`` dialogs run:

* ``library_core`` - ``install_library_env`` (this app's Library root, Output root
  and cache folder), ``scan_library`` (both scans + the ``_DualScannerThread``
  merge), ``card_signature`` / ``card_progress_view`` / ``card_raw_title`` /
  ``book_matches_query`` / ``format_of_book`` / ``sort_books``, ``LibraryShelf``
  (Organize / Undo / Delete / Clear raw link plans and executions, the counters),
  ``import_paths``, ``RawScanSession``, ``load_book_details`` / ``BookDetailsModel``
  / ``save_metadata_json_atomic`` and ``list_compiled_outputs``;
* ``library_covers`` - ``resolve_book_cover`` (cover cache in the app cache);
* ``progress_core`` / ``progress_actions`` / ``glossary_progress_core`` - the Book
  page's Chapters and Glossary tabs (``ui/library/progress_model.py``).

The modules are reached through :class:`SharedCore`, a lazy importer (tests pass
fakes). A function a build does not have raises :class:`CoreMissing`; the UI then
shows the action disabled with a reason instead of guessing (UI_SPEC §0 item 6).

Book ids: routes carry opaque 12-hex ids only (UI_SPEC §1.4). A book's id is the
``Prefs`` file ref of its *identity path* - the output workspace when it has one
(stable while the book moves from In progress to Completed), else its file - so
``/library/book/<bid>`` and the Reader's ``/reader/<bid>`` resolve the same book
(:meth:`LibraryService.book_for_bid`, :meth:`LibraryService.identity_for_bid`).

Threading: ``*_blocking`` methods do file I/O and run on the io pool (``io``);
``refresh`` and ``on_job_finished`` are coroutines for the UI loop. Listeners are
called on the loop (through ``post`` when one is given).

Pure Python, no Flet import (``ui/screens/storage.mirror_output`` is imported
lazily). Python 3.10 compatible.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import inspect
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field, replace
from typing import Any, Awaitable, Callable, Iterable, Mapping, Optional, Sequence

__all__ = [
    "BOOK_STATES",
    "CORE_MODULES",
    "CoreMissing",
    "DeletePlan",
    "DeleteReport",
    "DeleteTarget",
    "ImportReport",
    "LIBRARY_CONFIG_KEYS",
    "LibraryService",
    "Poller",
    "ScanSnapshot",
    "SharedCore",
    "bind_call",
    "book_identity",
    "book_key",
    "first_value",
]

log = logging.getLogger("glossarion.library")

#: The shared modules the Library reads (all GUI-free, flat in ``src/``).
CORE_MODULES = (
    "library_core",
    "library_covers",
    "progress_core",
    "progress_actions",
    "glossary_progress_core",
    "reader_doc",
)

#: ``translation_state`` values produced by ``library_core.scan_output_folders``.
BOOK_STATES = ("not_started", "in_progress", "ready_to_compile", "outdated_progress", "completed")

#: Existing desktop config keys the Library reads/writes (UI_SPEC §3.1, Appendix B; no new keys).
LIBRARY_CONFIG_KEYS = {
    "tab": "epub_library_tab",
    "sort": "epub_library_sort",
    "card_size": "epub_library_card_size",
    "format_filter": "epub_library_format_filter",
    "show_raw_titles": "epub_library_show_raw_titles",
    "page_size": "epub_library_page_size",
    "scan_raw_mode": "epub_library_scan_raw_mode",
    "scan_raw_threshold": "epub_library_scan_raw_threshold",
    "scan_raw_folder": "epub_library_scan_raw_folder",
    "scan_raw_auto": "epub_library_scan_raw_auto",
    "scan_raw_exts": "epub_library_scan_raw_exts",
    "details_show_special_files": "epub_details_show_special_files",
    "details_show_raw_titles": "epub_details_show_raw_titles",
    "details_chapter_page_size": "epub_details_chapter_page_size",
    "show_model_info": "retranslation_show_model_info",
    "manual_editing": "retranslation_manual_editing",
    "skip_unmatched": "glossary_progress_skip_unmatched_entries",
}
SCAN_RAW_KEYS = ("epub_library_scan_raw_mode", "epub_library_scan_raw_threshold", "epub_library_scan_raw_folder",
                 "epub_library_scan_raw_auto", "epub_library_scan_raw_exts")

#: Desktop delete keywords (``EpubLibraryDialog._DELETE_KEYWORDS``); the shared tuple wins when present.
DELETE_KEYWORDS = ("halgakos", "delete")

#: Job kinds whose outputs are reported inside a temporary run root that is moved when the job ends
#: (the chat's ``ChatRuns.finish``): the job-end mirror never copies from there.
_RUN_ROOT_KINDS = ("direct_text",)


class CoreMissing(RuntimeError):
    """A shared-core function this action needs does not exist in this build."""

    def __init__(self, name: str) -> None:
        super().__init__(f"{name} is not available in this build")
        self.name = name


# ---------------------------------------------------------------------------
# Calling the shared cores
# ---------------------------------------------------------------------------


def bind_call(fn: Callable[..., Any], *args: Any, **available: Any) -> Any:
    """Call ``fn(*args, **kw)`` where ``kw`` holds only the parameters ``fn`` declares.

    Used for optional hooks whose exact signature varies (cover resolvers, display
    helpers); ``None`` values are only passed to parameters without a default.
    """
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return fn(*args)
    kwargs: dict = {}
    positional = len(args)
    for param in signature.parameters.values():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        if positional and param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD):
            positional -= 1
            continue
        if param.kind is param.POSITIONAL_ONLY:
            if param.default is param.empty:
                raise TypeError(f"{getattr(fn, '__name__', fn)}() needs {param.name!r}")
            continue
        if param.name in available and (available[param.name] is not None or param.default is param.empty):
            kwargs[param.name] = available[param.name]
        elif param.default is param.empty:
            raise TypeError(f"{getattr(fn, '__name__', fn)}() needs {param.name!r}")
    return fn(*args, **kwargs)


def first_value(obj: Any, *names: str, default: Any = None) -> Any:
    """``obj[name]`` / ``obj.name`` for the first name that is set (dicts, dataclasses, objects)."""
    for name in names:
        if isinstance(obj, Mapping):
            if name in obj and obj[name] is not None:
                return obj[name]
        else:
            value = getattr(obj, name, None)
            if value is not None:
                return value
    return default


class SharedCore:
    """Lazy, injectable access to the shared modules (tests pass fakes as ``modules``)."""

    def __init__(self, modules: Optional[Mapping[str, Any]] = None, *, importer: Optional[Callable[[str], Any]] = None
                 ) -> None:
        self._modules: dict = dict(modules or {})
        self._importer = importer or importlib.import_module
        self._missing: set = set()
        self._lock = threading.Lock()

    def module(self, name: str) -> Any:
        """The module, or None when it cannot be imported (logged once)."""
        with self._lock:
            if name in self._modules:
                return self._modules[name]
            if name in self._missing:
                return None
        try:
            module = self._importer(name)
        except Exception as exc:  # ImportError, or a broken optional dependency
            log.info("shared module %s unavailable: %s", name, exc)
            with self._lock:
                self._missing.add(name)
            return None
        with self._lock:
            self._modules[name] = module
        return module

    def has_module(self, name: str) -> bool:
        return self.module(name) is not None

    def fn(self, module_name: str, *names: str) -> Optional[Callable[..., Any]]:
        """The first attribute of ``module_name`` named in ``names`` that is callable."""
        module = self.module(module_name)
        if module is None:
            return None
        for name in names:
            target = getattr(module, name, None)
            if callable(target):
                return target
        return None

    def value(self, module_name: str, *names: str, default: Any = None) -> Any:
        module = self.module(module_name)
        if module is None:
            return default
        for name in names:
            if hasattr(module, name):
                return getattr(module, name)
        return default

    def require(self, module_name: str, *names: str) -> Callable[..., Any]:
        found = self.fn(module_name, *names)
        if found is None:
            raise CoreMissing(f"{module_name}.{names[0]}")
        return found

    def available(self, module_name: str, *names: str) -> bool:
        return self.fn(module_name, *names) is not None


# ---------------------------------------------------------------------------
# Book identity
# ---------------------------------------------------------------------------


def _norm(path: Any) -> str:
    try:
        return os.path.normcase(os.path.normpath(os.path.abspath(os.fspath(path))))
    except Exception:
        return str(path or "")


def book_identity(book: Mapping[str, Any]) -> str:
    """The path that names a book for its whole life: its output workspace, else its file."""
    folder = str(book.get("output_folder") or "")
    if folder:
        return os.path.abspath(folder)
    path = str(book.get("path") or "")
    return os.path.abspath(path) if path else ""


def book_key(book: Mapping[str, Any]) -> str:
    """Stable key for list reconciliation (normalised identity path)."""
    identity = book_identity(book)
    return _norm(identity) if identity else str(book.get("name") or id(book))


def _fallback_signature(book: Mapping[str, Any]) -> tuple:
    try:
        return ("json", json.dumps(book, sort_keys=True, default=str))
    except Exception:
        return ("repr", repr(sorted(book.items(), key=lambda kv: str(kv[0]))))


# ---------------------------------------------------------------------------
# Snapshots and reports
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScanSnapshot:
    """One Library scan: both shelves plus what the home screen derives from them."""

    in_progress: tuple = ()
    completed: tuple = ()
    signatures: Mapping[str, Any] = field(default_factory=dict)  # book_key -> card_signature
    views: Mapping[str, Any] = field(default_factory=dict)  # book_key -> card_progress_view dict
    missing_raw: int = 0  # cards with ``missing_raw_file`` (the "Scan for raw (N)" chip)
    organize_count: int = 0  # desktop Organize (n) counter (the mobile Library offers no manual Organize)
    undo_count: int = 0  # desktop Undo (n) counter (likewise no control on mobile)
    #: book_key -> the translation workspace of a Completed row without one of its own (a
    #: Library/Translated EPUB): the desktop ``_resolve_book_output_folder`` origins lookup
    workspaces: Mapping[str, str] = field(default_factory=dict)
    scanned_at: float = 0.0
    seconds: float = 0.0
    error: Optional[str] = None
    generation: int = 0

    def shelf(self, name: str) -> tuple:
        return self.completed if name == "completed" else self.in_progress

    def all_books(self) -> tuple:
        return tuple(self.in_progress) + tuple(self.completed)

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass(frozen=True)
class ImportReport:
    imported: tuple = ()
    skipped: tuple = ()
    errors: tuple = ()
    copied: tuple = ()  # ({"source", "path", "reused"}, ...)


@dataclass(frozen=True)
class DeleteTarget:
    label: str
    path: str
    is_folder: bool
    book: Mapping[str, Any] = field(default_factory=dict)
    contents: tuple = ()  # summarize_folder_contents lines ("    · 12 translated chapter HTML files", ...)
    size_text: str = ""

    def as_tuple(self) -> tuple:
        return (self.label, self.path, self.is_folder, dict(self.book))


@dataclass(frozen=True)
class DeletePlan:
    targets: tuple = ()  # DeleteTarget
    unregister: tuple = ()  # (book, path) registry-only removals (outside the safe roots)
    needs_keyword: bool = True  # False only when every target is Not started
    keywords: tuple = DELETE_KEYWORDS
    simple_prompt: str = ""  # the desktop Yes/Cancel text for an all-Not-started batch
    raw: Any = None  # the shared plan, handed back to execute_delete


@dataclass(frozen=True)
class DeleteReport:
    deleted: int = 0
    total: int = 0
    errors: tuple = ()
    unregistered: int = 0
    summary_text: str = ""
    links_note: str = ""  # U10: what happened to the deleted books' share-link uploads

    @property
    def summary(self) -> str:
        """Desktop ``_on_delete_finished`` text (plus the U10 share-link note)."""
        if self.summary_text:
            return self.summary_text + (f"\n\n{self.links_note}" if self.links_note else "")
        text = f"Deleted {self.deleted} of {self.total} item{'s' if self.total != 1 else ''}."
        if self.errors:
            text += (f"\n\n{len(self.errors)} error{'s' if len(self.errors) != 1 else ''}:\n"
                     + "\n".join("  - " + e for e in list(self.errors)[:5]))
        if self.links_note:
            text += f"\n\n{self.links_note}"
        return text


def _size_text(size: Any) -> str:
    """Desktop size label: "x.x MB" from 1 MB, else "N KB"."""
    try:
        value = float(size)
    except (TypeError, ValueError):
        return "?"
    if value >= 1024 * 1024:
        return f"{value / (1024 * 1024):.1f} MB"
    return f"{value / 1024:.0f} KB"


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


# ---------------------------------------------------------------------------
# Poller (UI_SPEC §3.12)
# ---------------------------------------------------------------------------


class Poller:
    """``tick()`` every ``interval`` seconds while ``visible()`` and ``foreground()``.

    One tick at a time (a slow tick is never overlapped); ``poke()`` runs one at
    once (pull-to-refresh, resume, a finished job). ``stop()`` ends the loop.
    """

    def __init__(
        self,
        tick: Callable[[], Awaitable[Any]],
        *,
        interval: float = 2.0,
        visible: Callable[[], bool] = lambda: True,
        foreground: Callable[[], bool] = lambda: True,
        spawn: Optional[Callable[[Any], Any]] = None,
        sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
        name: str = "library",
    ) -> None:
        self.tick = tick
        self.interval = float(interval)
        self.visible = visible
        self.foreground = foreground
        self._spawn = spawn
        self._sleep = sleep
        self.name = name
        self.task: Any = None
        self.running = False
        self.ticks = 0
        self._busy = False
        self._stopped = False

    def _start_task(self, coro: Any) -> Any:
        if self._spawn is not None:
            return self._spawn(coro)
        try:
            return asyncio.ensure_future(coro)
        except RuntimeError:
            coro.close()
            return None

    def start(self) -> None:
        if self.running:
            return
        self._stopped = False
        self.running = True
        self.task = self._start_task(self._loop())

    def stop(self) -> None:
        self._stopped = True
        self.running = False
        task, self.task = self.task, None
        if task is not None and hasattr(task, "cancel"):
            try:
                task.cancel()
            except Exception:
                pass

    def active(self) -> bool:
        try:
            return bool(self.visible()) and bool(self.foreground())
        except Exception:
            return False

    async def run_once(self) -> bool:
        if self._busy:
            return False
        self._busy = True
        try:
            await self.tick()
            self.ticks += 1
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("%s poll tick failed", self.name)
        finally:
            self._busy = False
        return True

    def poke(self) -> Any:
        """One tick now (when visible); returns the task."""
        if self._stopped or not self.active():
            return None
        return self._start_task(self.run_once())

    async def _loop(self) -> None:
        try:
            while not self._stopped:
                await self._sleep(self.interval)
                if self._stopped:
                    break
                if self.active():
                    await self.run_once()
        except asyncio.CancelledError:
            pass
        finally:
            self.running = False


# ---------------------------------------------------------------------------
# The service
# ---------------------------------------------------------------------------


Listener = Callable[[ScanSnapshot], Any]


class LibraryService:
    def __init__(
        self,
        *,
        paths: Any = None,
        config: Any = None,  # MobileConfigStore (get / set / snapshot) or a plain dict
        prefs: Any = None,
        files: Any = None,  # FileBridge
        jobs: Any = None,  # JobsFeature / JobService (has_kind, submit)
        core: Optional[SharedCore] = None,
        run_io: Optional[Callable[..., Awaitable[Any]]] = None,
        post: Optional[Callable[..., Any]] = None,  # run a callable on the UI loop
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.paths = paths
        self.config = config
        self.prefs = prefs
        self.files = files
        self.jobs = jobs
        self.core = core or SharedCore()
        self._run_io = run_io
        self._post = post
        self.clock = clock
        self._snapshot = ScanSnapshot()
        self._shelf: Any = None  # library_core.LibraryShelf of the last scan
        self._listeners: list = []
        self._by_bid: dict = {}
        self._bid_by_key: dict = {}
        self._local_refs: dict = {}  # bid -> identity (when Prefs is unavailable)
        self._workspaces: dict = {}  # book_key -> resolved workspace (ScanSnapshot.workspaces of the last scan)
        self._generation = 0
        self._env_key: Any = None
        self.scanning = False
        self.deleting = False
        self.dirty = True  # first open scans at once
        self.compiling: set = set()  # normalised output folders with a compile job queued/running
        self.covers: dict = {}  # book_key -> cover thumbnail path (None: placeholder)
        self.mirrored: list = []  # last on_job_finished mirror URIs (diagnostics)
        #: The U10 cloud sync service, set when the app installs it: book outputs then go to its
        #: destination (the phone folder: one entry per output, updated in place), so the job-end
        #: mirror copies only the other outputs.
        self.cloud_sync: Any = None
        #: The U10 share-link service (``services/share_links.py``), set when the app installs it: the share
        #: links of deleted books are forgotten (``execute_delete_blocking``).
        self.share_links: Any = None

    # ---- plumbing ----------------------------------------------------------------------------

    async def io(self, fn: Callable[..., Any], *args: Any) -> Any:
        if self._run_io is not None:
            return await self._run_io(fn, *args)
        return await asyncio.to_thread(fn, *args)

    def _emit(self, snapshot: ScanSnapshot) -> None:
        def deliver() -> None:
            for listener in list(self._listeners):
                try:
                    listener(snapshot)
                except Exception:
                    log.exception("library listener failed")

        if self._post is not None:
            try:
                self._post(deliver)
                return
            except Exception:
                pass
        deliver()

    def subscribe(self, listener: Listener, *, immediate: bool = False) -> Callable[[], None]:
        self._listeners.append(listener)
        if immediate:
            listener(self._snapshot)

        def unsubscribe() -> None:
            if listener in self._listeners:
                self._listeners.remove(listener)

        return unsubscribe

    @property
    def snapshot(self) -> ScanSnapshot:
        return self._snapshot

    # ---- config --------------------------------------------------------------------------------

    def config_snapshot(self) -> dict:
        config = self.config
        if config is None:
            return {}
        if isinstance(config, Mapping):
            return dict(config)
        snap = getattr(config, "snapshot", None)
        if callable(snap):
            try:
                return dict(snap() or {})
            except Exception:
                log.exception("config snapshot failed")
        return {}

    def cfg(self, key: str, default: Any = None) -> Any:
        config = self.config
        if config is None:
            return default
        try:
            value = config.get(key, default)
        except Exception:
            return default
        return default if value is None else value

    def save_owner_config(self, config: Any) -> list:
        """A shared owner's ``save_config`` callback: write back only the keys it changed (sparse)."""
        if not isinstance(config, Mapping) or self.config is None:
            return []
        current = self.config_snapshot()
        changed = [key for key, value in config.items() if current.get(key) != value]
        for key in changed:
            self.set_cfg(key, config[key])
        return changed

    def set_cfg(self, key: str, value: Any) -> None:
        config = self.config
        if config is None:
            return
        try:
            if isinstance(config, dict):
                config[key] = value
            else:
                config.set(key, value)
        except Exception:
            log.exception("saving %s failed", key)

    # ---- locations / env -------------------------------------------------------------------------

    def library_root(self) -> str:
        library = getattr(self.paths, "library", None) if self.paths is not None else None
        if library:
            return os.fspath(library)
        env_dir = os.environ.get("GLOSSARION_LIBRARY_DIR")
        if env_dir:
            return env_dir
        getter = self.core.fn("library_core", "library_root_path", "get_library_dir")
        return os.fspath(getter()) if getter is not None else ""

    def output_roots(self) -> list:
        roots: list = []
        output = getattr(self.paths, "output", None) if self.paths is not None else None
        if output:
            roots.append(os.fspath(output))
        env_out = os.environ.get("OUTPUT_DIRECTORY")
        if env_out and _norm(env_out) not in {_norm(r) for r in roots}:
            roots.append(env_out)
        return roots

    def cache_dir(self) -> str:
        cache = getattr(self.paths, "cache", None) if self.paths is not None else None
        base = os.fspath(cache) if cache else os.path.join(self.library_root() or os.getcwd(), ".cache")
        return os.path.join(base, "library")

    def ensure_env(self) -> Any:
        """``library_core.install_library_env`` for this app (once; again when the folders change).

        Pins ``GLOSSARION_LIBRARY_DIR``, the default output root (the app's Output folder
        instead of the process cwd) and the cover / EPUB caches under the app cache.
        """
        lc = self.core.module("library_core")
        if lc is None or not hasattr(lc, "install_library_env") or not hasattr(lc, "LibraryEnv"):
            return None
        key = (self.library_root(), tuple(self.output_roots()), self.cache_dir())
        if key == self._env_key and getattr(lc, "current_library_env", lambda: None)() is not None:
            return lc.current_library_env()
        env = lc.LibraryEnv(key[0] or None, key[1], key[2], None)
        try:
            lc.install_library_env(env)
        except Exception:
            log.exception("installing the Library environment failed")
            return None
        self._env_key = key
        return env

    def env(self) -> Any:
        """The installed ``library_core.LibraryEnv`` (the Reader's ``env_factory``); None without one."""
        return self.ensure_env()

    def delete_keywords(self) -> tuple:
        value = self.core.value("library_core", "DELETE_KEYWORDS")
        return tuple(value) if value else DELETE_KEYWORDS

    def _lc(self) -> Any:
        lc = self.core.module("library_core")
        if lc is None:
            raise CoreMissing("library_core")
        self.ensure_env()
        return lc

    def _shelf_for(self, config: Optional[dict] = None) -> Any:
        """``library_core.LibraryShelf`` over the last scan (the desktop dialog's state)."""
        lc = self._lc()
        if not hasattr(lc, "LibraryShelf"):
            raise CoreMissing("library_core.LibraryShelf")
        snap = self._snapshot
        return lc.LibraryShelf([dict(b) for b in snap.in_progress], [dict(b) for b in snap.completed],
                               self.config_snapshot() if config is None else config)

    # ---- ids ------------------------------------------------------------------------------------

    def bid_for(self, book: Mapping[str, Any]) -> str:
        """The opaque route id of ``book`` (``Prefs.file_ref`` of its identity path)."""
        key = book_key(book)
        cached = self._bid_by_key.get(key)
        if cached:
            return cached
        identity = book_identity(book) or str(book.get("name") or "book")
        bid = None
        prefs = self.prefs
        if prefs is not None and hasattr(prefs, "file_ref"):
            try:
                bid = prefs.file_ref(identity, kind="book")
            except Exception:
                bid = None
        if not bid:
            bid = hashlib.sha1(_norm(identity).encode("utf-8", "surrogatepass")).hexdigest()[:12]
            self._local_refs[bid] = identity
        self._bid_by_key[key] = bid
        self._by_bid.setdefault(bid, dict(book))
        return bid

    def identity_for_bid(self, bid: str) -> Optional[str]:
        book = self._by_bid.get(str(bid))
        if book is not None:
            return book_identity(book)
        prefs = self.prefs
        if prefs is not None and hasattr(prefs, "resolve_file_ref"):
            try:
                path = prefs.resolve_file_ref(str(bid))
            except Exception:
                path = None
            if path:
                return path
        return self._local_refs.get(str(bid))

    def book_for_bid(self, bid: str) -> Optional[dict]:
        """The book row for a route id: from the last scan, else synthesised from its identity path.

        A synthesised row uses the scanner's field names (``path``, ``output_folder``,
        ``type``, ``name``) so the shared resolvers accept it; the next scan that
        sees the book replaces it.
        """
        book = self._by_bid.get(str(bid))
        if book is not None:
            return dict(book)
        identity = self.identity_for_bid(bid)
        if not identity:
            return None
        for candidate in self._snapshot.all_books():
            if _norm(book_identity(candidate)) == _norm(identity):
                self._by_bid[str(bid)] = dict(candidate)
                return dict(candidate)
        book = self._synthesise(identity)
        if book is not None:
            self._by_bid[str(bid)] = dict(book)
        return book

    @staticmethod
    def _synthesise(identity: str) -> Optional[dict]:
        if os.path.isdir(identity):
            name = os.path.basename(os.path.normpath(identity))
            progress = os.path.join(identity, "translation_progress.json")
            return {"name": name, "folder_name": name, "path": identity, "output_folder": identity,
                    "type": "in_progress", "is_in_progress": True, "in_library": False,
                    "progress_file": progress if os.path.isfile(progress) else "", "size": 0, "mtime": 0.0,
                    "synthesised": True}
        if os.path.isfile(identity):
            stem, ext = os.path.splitext(os.path.basename(identity))
            try:
                size = os.path.getsize(identity)
                mtime = os.path.getmtime(identity)
            except OSError:
                size, mtime = 0, 0.0
            return {"name": stem, "path": identity, "type": ext.lstrip(".").lower() or "epub", "size": size,
                    "mtime": mtime, "in_library": False, "is_in_progress": False, "synthesised": True}
        return None

    def remember(self, book: Mapping[str, Any]) -> str:
        """Refresh the cached row of a book and return its id."""
        bid = self.bid_for(book)
        self._by_bid[bid] = dict(book)
        return bid

    # ---- scanning (blocking) ------------------------------------------------------------------

    def card_signature(self, book: Mapping[str, Any]) -> Any:
        fn = self.core.fn("library_core", "card_signature")
        if fn is None:
            return _fallback_signature(book)
        try:
            return fn(book)
        except Exception:
            log.debug("card_signature failed", exc_info=True)
            return _fallback_signature(book)

    def card_view(self, book: Mapping[str, Any]) -> Optional[dict]:
        """``library_core.card_progress_view(book)`` (None: no pill / ribbon, or no core function)."""
        fn = self.core.fn("library_core", "card_progress_view")
        if fn is None:
            return None
        try:
            view = fn(dict(book))
        except Exception:
            log.debug("card_progress_view failed", exc_info=True)
            return None
        return dict(view) if isinstance(view, Mapping) else None

    def scan_blocking(self) -> ScanSnapshot:
        """``library_core.scan_library`` (both scans + merge), card views, Organize / Undo / raw counters and
        the workspaces of Completed Library rows (:meth:`resolve_workspaces_blocking`)."""
        started = time.monotonic()
        config = self.config_snapshot()
        lc = self._lc()
        if not hasattr(lc, "scan_library"):
            raise CoreMissing("library_core.scan_library")
        in_progress, completed = lc.scan_library(config)
        workspaces = self.resolve_workspaces_blocking(completed)
        signatures: dict = {}
        views: dict = {}
        for book in list(in_progress) + list(completed):
            key = book_key(book)
            signatures[key] = self.card_signature(book)
            view = self.card_view(book)
            if view is not None:
                views[key] = view
        organize = undo = 0
        missing = sum(1 for b in list(in_progress) + list(completed) if b.get("missing_raw_file"))
        if hasattr(lc, "LibraryShelf"):
            try:
                counts = lc.LibraryShelf(in_progress, completed, config).counts()
                organize = _as_int(counts.get("raw_count")) + _as_int(counts.get("trans_count"))
                undo = _as_int(counts.get("raw_undo")) + _as_int(counts.get("trans_undo"))
                missing = _as_int(counts.get("missing_raw", missing))
            except Exception:
                log.debug("Library counters failed", exc_info=True)
        self._generation += 1
        return ScanSnapshot(
            in_progress=tuple(dict(b) for b in in_progress),
            completed=tuple(dict(b) for b in completed),
            signatures=signatures,
            views=views,
            missing_raw=missing,
            organize_count=organize,
            undo_count=undo,
            workspaces=workspaces,
            scanned_at=self.clock(),
            seconds=time.monotonic() - started,
            generation=self._generation,
        )

    def resolve_workspaces_blocking(self, rows: Iterable[Mapping[str, Any]]) -> dict:
        """``book_key -> workspace`` for the Completed Library rows without a workspace of their own.

        A Library/Translated EPUB (Organize, a chat book filed by the desktop, "Add translation") carries
        no ``output_folder`` (``library_core`` merge), so the Book page, the card ⋯ Compile / Files and
        the metadata editor would have nothing to work on. Desktop resolves the folder at each use with
        ``_resolve_book_output_folder`` (the ``library_origins`` "translated" entry's folder); the
        result is kept only when it holds ``translation_progress.json`` (desktop
        ``_card_has_reader_workspace``). Read only: nothing is created, the rows and their ids stay as
        they are."""
        resolve = self.core.fn("library_core", "resolve_book_output_folder", "_resolve_book_output_folder")
        if resolve is None:
            return {}
        found: dict = {}
        for book in rows or ():
            folder = str(book.get("output_folder") or "")
            if (folder and os.path.isdir(folder)) or not book.get("in_library"):
                continue
            try:
                workspace = str(resolve(dict(book)) or "")
            except Exception:
                log.debug("resolving the workspace of %s failed", book.get("name"), exc_info=True)
                continue
            if workspace and os.path.isfile(os.path.join(workspace, "translation_progress.json")):
                found[book_key(book)] = workspace
        return found

    def workspace_for(self, book: Mapping[str, Any]) -> str:
        """The book's translation workspace: its ``output_folder`` when that is a folder, else the one
        the last scan resolved for a Library row (:meth:`resolve_workspaces_blocking`), else "".
        Sync and cheap (no disk work beyond one ``isdir``); never creates a folder."""
        folder = str(book.get("output_folder") or "")
        if folder and os.path.isdir(folder):
            return folder
        return str(self._workspaces.get(book_key(book)) or "")

    # ---- scanning (UI loop) --------------------------------------------------------------------

    def mark_dirty(self) -> None:
        self.dirty = True

    async def refresh(self, *, quiet: bool = False, reason: str = "") -> Optional[ScanSnapshot]:
        """Rescan on the io pool and publish; skips while a scan or a delete runs (desktop rule)."""
        if self.scanning or self.deleting:
            return None
        self.scanning = True
        try:
            try:
                snap = await self.io(self.scan_blocking)
            except CoreMissing as exc:
                snap = replace(self._snapshot, error=str(exc), scanned_at=self.clock())
            except Exception as exc:
                log.exception("library scan failed (%s)", reason or "refresh")
                snap = replace(self._snapshot, error=f"Couldn't read the Library ({exc.__class__.__name__}: {exc})",
                               scanned_at=self.clock())
        finally:
            self.scanning = False
        self.dirty = False
        self._apply_snapshot(snap)
        return snap

    def _apply_snapshot(self, snap: ScanSnapshot) -> None:
        self._snapshot = snap
        if snap.ok:
            self._workspaces = dict(snap.workspaces or {})
            for book in snap.all_books():
                bid = self.bid_for(book)
                self._by_bid[bid] = dict(book)
        self._emit(snap)

    def set_snapshot(self, snap: ScanSnapshot) -> None:
        """Publish a snapshot built elsewhere (tests, recovery)."""
        self._apply_snapshot(snap)

    # ---- query (pure; the shared desktop rules) ------------------------------------------------------

    def matches_query(self, book: Mapping[str, Any], query: str) -> bool:
        if not query:
            return True
        fn = self.core.fn("library_core", "book_matches_query")
        if fn is None:
            return query.casefold() in str(book.get("name") or "").casefold()
        try:
            return bool(fn(dict(book), query))
        except Exception:
            return False

    def format_of(self, book: Mapping[str, Any]) -> str:
        fn = self.core.fn("library_core", "format_of_book")
        if fn is not None:
            try:
                return str(fn(dict(book)) or "all")
            except Exception:
                pass
        return "all"

    def sort_books(self, books: Sequence[Mapping[str, Any]], mode: str, *, reverse: bool = False) -> list:
        fn = self.core.fn("library_core", "sort_books")
        ordered = list(books)
        if fn is not None:
            try:
                ordered = list(fn([dict(b) for b in books], mode))
            except Exception:
                log.debug("sort_books failed", exc_info=True)
        return list(reversed(ordered)) if reverse else ordered

    def raw_title(self, book: Mapping[str, Any]) -> str:
        fn = self.core.fn("library_core", "card_raw_title")
        if fn is not None:
            try:
                return str(fn(dict(book)) or "")
            except Exception:
                pass
        return str(book.get("name") or "")

    def size_presets(self) -> Optional[Mapping[str, Any]]:
        return self.core.value("library_core", "_SIZE_PRESETS")

    def card_badge(self, book: Mapping[str, Any]) -> tuple:
        """``(type badge text, size label)`` from ``card_type_badge`` / ``card_size_text`` (None, None
        without them)."""
        badge = size = None
        fn = self.core.fn("library_core", "card_type_badge")
        if fn is not None:
            try:
                badge = str(fn(dict(book))[0])
            except Exception:
                badge = None
        fn = self.core.fn("library_core", "card_size_text")
        if fn is not None:
            try:
                size = str(fn(book.get("size") or 0))
            except Exception:
                size = None
        return badge, size

    # ---- import ------------------------------------------------------------------------------------

    def import_paths_blocking(self, paths: Sequence[str], target: str = "raw",
                              record_origins: bool = False) -> ImportReport:
        """``library_core.import_paths(copy_into_library=True)``: copies a file into Library/Raw
        (Translated) unless it already sits there, registers it, scaffolds its workspace
        (``source_epub.txt`` + an empty v2.1 progress file) and, with ``record_origins``,
        remembers where the copy came from (an Inbox file) so Undo can move it back."""
        paths = [os.fspath(p) for p in paths if p]
        if not paths:
            return ImportReport()
        lc = self._lc()
        config = self.config_snapshot()
        if hasattr(lc, "import_paths"):
            result = lc.import_paths(paths, target, config, copy_into_library=True,
                                     record_origins=bool(record_origins))
        else:
            record = getattr(lc, "record_library_translated_input" if target == "translated"
                             else "record_library_raw_input", None)
            if record is None:
                raise CoreMissing("library_core.import_paths")
            for path in paths:
                record(path)
            result = {"imported": paths}
        self.mark_dirty()
        result = result if isinstance(result, Mapping) else {}
        return ImportReport(imported=tuple(result.get("imported") or ()), skipped=tuple(result.get("skipped") or ()),
                            errors=tuple(result.get("errors") or ()), copied=tuple(result.get("copied") or ()))

    # ---- delete -------------------------------------------------------------------------------------

    def summarize_folder(self, folder: str) -> tuple:
        fn = self.core.fn("library_core", "summarize_folder_contents")
        if fn is None:
            return ()
        try:
            return tuple(fn(folder) or ())
        except Exception:
            return ()

    def plan_delete_blocking(self, books: Sequence[Mapping[str, Any]]) -> DeletePlan:
        """``LibraryShelf.plan_delete``: safe-root gate, workspace vs file targets, the Library/Raw
        pair, the silent unregister list, the keyword rule; plus contents summaries."""
        raw = self._shelf_for().plan_delete([dict(b) for b in books])
        targets = []
        for label, path, is_folder, book in raw.get("targets") or ():
            if is_folder:
                contents, size_text = self.summarize_folder(path), ""
            else:
                contents = ()
                try:
                    size_text = _size_text(os.path.getsize(path))
                except OSError:
                    size_text = "?"
            targets.append(DeleteTarget(str(label), str(path), bool(is_folder), dict(book or {}), contents, size_text))
        return DeletePlan(targets=tuple(targets), unregister=tuple(raw.get("unregister") or ()),
                          needs_keyword=bool(raw.get("needs_keyword")), keywords=self.delete_keywords(),
                          simple_prompt=str(raw.get("simple_prompt") or ""), raw=raw)

    def execute_delete_blocking(self, plan: DeletePlan, selected: Optional[Iterable[str]] = None,
                                progress: Optional[Callable[[int, int, str], Any]] = None, *,
                                delete_remote_links: bool = False) -> DeleteReport:
        """``LibraryShelf.execute_delete`` on the confirmed targets (all, or the ticked paths).
        ``delete_remote_links``: the user chose to delete the books' share-link uploads on their services too
        (``deletable_links_blocking``); otherwise they stay online until they expire and the report says so."""
        keep = None if selected is None else {_norm(p) for p in selected}
        chosen = [t.as_tuple() for t in plan.targets if keep is None or _norm(t.path) in keep]
        self.deleting = True
        try:
            result = self._shelf_for().execute_delete(plan.raw, targets=chosen, on_progress=progress)
        finally:
            self.deleting = False
            self.mark_dirty()
        note = self._forget_deleted_blocking([t for t in plan.targets if keep is None or _norm(t.path) in keep],
                                             delete_remote=delete_remote_links)
        result = result if isinstance(result, Mapping) else {}
        return DeleteReport(deleted=_as_int(result.get("deleted")), total=len(chosen),
                            errors=tuple(str(e) for e in (result.get("errors") or ())),
                            unregistered=_as_int(result.get("unregistered")),
                            summary_text=str(result.get("summary") or ""), links_note=note)

    def deletable_links_blocking(self, books: Sequence[Mapping[str, Any]]) -> int:
        """U10, io pool: how many live share-link uploads of ``books`` Glossarion can still delete on their
        services (the delete asks whether to delete them too: after the book is gone it cannot)."""
        finder = getattr(self.share_links, "deletable_links_blocking", None)
        if not callable(finder):
            return 0
        identities = [i for i in (book_identity(b) for b in books or ()) if i]
        try:
            return len(finder(identities)) if identities else 0
        except Exception:
            log.debug("listing the share links of the books to delete failed", exc_info=True)
            return 0

    def _forget_deleted_blocking(self, targets: Sequence[DeleteTarget], *, delete_remote: bool = False) -> str:
        """U10, worker thread: the cloud sync records / queue and the share links of the books that are really
        gone. Files already in the cloud are never deleted; share-link uploads only with ``delete_remote``
        (the user's choice). Returns the report's note about those uploads ('' for none)."""
        cloud, shares = self.cloud_sync, self.share_links
        if cloud is None and shares is None:
            return ""
        books = {book_identity(t.book) for t in targets}
        gone = sorted(p for p in books | {t.path for t in targets} if p and not os.path.exists(p))
        if not gone:
            return ""
        if cloud is not None:
            try:
                cloud.forget_books_threadsafe([p for p in gone if p in books])
            except Exception:
                log.exception("forgetting the cloud records of deleted books failed")
        note = ""
        if shares is not None:
            try:
                finder = getattr(shares, "deletable_links_blocking", None)
                live = len(finder(gone)) if callable(finder) else 0
                failed = shares.forget_books_blocking(gone, delete_remote=delete_remote) if delete_remote \
                    else shares.forget_books_blocking(gone)
                kept = len(failed or ()) if delete_remote else live
                if kept:
                    note = (f"{kept} file{'s' if kept != 1 else ''} shared via a link "
                            f"{'are' if kept != 1 else 'is'} still online until the link expires"
                            + (" (the service could not be reached to delete it)." if delete_remote else "."))
                elif delete_remote and live:
                    note = f"Deleted {live} shared upload{'s' if live != 1 else ''} from {'their services' if live != 1 else 'its service'} too."
            except Exception:
                log.exception("forgetting the share links of deleted books failed")
        return note

    # ---- clear raw link / organize / undo -------------------------------------------------------------

    def plan_clear_raw_link_blocking(self, books: Sequence[Mapping[str, Any]]) -> dict:
        return dict(self._shelf_for().plan_clear_raw_link([dict(b) for b in books]))

    def execute_clear_raw_link_blocking(self, plan: Mapping[str, Any]) -> int:
        try:
            return _as_int(self._shelf_for().execute_clear_raw_link(dict(plan)))
        finally:
            self.mark_dirty()

    def plan_organize_blocking(self, books: Optional[Sequence[Mapping[str, Any]]] = None) -> dict:
        """The shelf's Organize plan; ``books``: only their moves, with the preview and the collisions
        recomputed by the shelf's own helpers. Service API only: the mobile Library has no manual
        Organize / Undo control (finished chat books reach the Library by themselves)."""
        shelf = self._shelf_for()
        plan = dict(shelf.plan_organize())
        if books is None:
            return plan
        wanted = {self.bid_for(book) for book in books}
        paths = {_norm(p) for book in books for p in (book.get("raw_source_path"), book.get("path")) if p}

        def keep(move: Any) -> bool:
            book, path = move
            return _norm(path) in paths or self.bid_for(book) in wanted

        plan["raw_moves"] = [m for m in plan.get("raw_moves") or () if keep(m)]
        plan["translated_moves"] = [m for m in plan.get("translated_moves") or () if keep(m)]
        plan["preview"] = type(shelf)._organize_preview_lines(plan)
        raw_collisions, trans_collisions = type(shelf)._organize_collisions(plan)
        plan["collisions"] = raw_collisions + trans_collisions
        plan["raw_collisions"] = raw_collisions
        plan["trans_collisions"] = trans_collisions
        return plan

    def execute_organize_blocking(self, plan: Mapping[str, Any], policy: str = "keep_both") -> dict:
        try:
            return dict(self._shelf_for().execute_organize(dict(plan), policy))
        finally:
            self.mark_dirty()

    def plan_undo_blocking(self) -> dict:
        return dict(self._shelf_for().plan_undo())

    def undo_collisions_blocking(self, plan: Mapping[str, Any], restore_raw: bool, restore_trans: bool) -> list:
        return list(self._shelf_for().undo_collisions(dict(plan), restore_raw, restore_trans))

    def execute_undo_blocking(self, plan: Mapping[str, Any], restore_raw: bool, restore_trans: bool,
                              policy: str = "keep_both", collisions: Any = None) -> dict:
        try:
            return dict(self._shelf_for().execute_undo(dict(plan), restore_raw, restore_trans, policy, collisions))
        finally:
            self.mark_dirty()

    # ---- output root (desktop "Load for translation": _ensure_output_override_matches) ------------------

    def _job_lock(self) -> Any:
        """``job_runner.JOB_LOCK``: a running job owns ``os.environ`` (its ``OUTPUT_DIRECTORY`` is its own
        run root); None when the module is missing."""
        return self.core.value("job_runner", "JOB_LOCK", default=None)

    def output_root_mismatch_blocking(self, books: Sequence[Mapping[str, Any]]) -> Optional[dict]:
        """``LibraryShelf.output_root_mismatch``: None when ``books`` can load as they are, else the switch
        the desktop proposes, with its "Output Folder Mismatch" text in ``prompt``.

        Checked only while no job owns the process environment (``JOB_LOCK`` taken without waiting): a
        running job's ``OUTPUT_DIRECTORY`` is its own run root, which would make every book look
        mismatched. None then; the queued translation starts in the app's output root anyway."""
        lock = self._job_lock()
        if lock is not None and not lock.acquire(blocking=False):
            return None
        try:
            info = self._shelf_for().output_root_mismatch([dict(b) for b in books or ()])
        finally:
            if lock is not None:
                lock.release()
        return dict(info) if info else None

    def apply_output_override_blocking(self, root: str) -> str:
        """``LibraryShelf.apply_output_override``: the process ``OUTPUT_DIRECTORY`` becomes ``root`` so the
        next translation writes into the folder that holds the book's progress. Returns the root used.

        Mobile differences: "" (the desktop's "back to the default root") names the default root
        (``library_core._default_output_root``, the app's Output folder), because the mobile env contract
        never leaves ``OUTPUT_DIRECTORY`` unset; and config.json's ``output_directory`` is not written
        (Settings › Storage: the output root is app storage). Refused (RuntimeError) while a job owns the
        process environment."""
        target = str(root or "")
        if not target:
            default = self.core.fn("library_core", "_default_output_root")
            target = str(default() or "") if default is not None else ""
            target = target or (self.output_roots() or [""])[0]
        if not target:
            raise ValueError("No output folder to switch to")
        lock = self._job_lock()
        if lock is not None and not lock.acquire(blocking=False):
            raise RuntimeError("Wait for the running job to finish, then try again.")
        try:
            self._shelf_for().apply_output_override(target)
        finally:
            if lock is not None:
                lock.release()
            self.mark_dirty()
        return target

    # ---- Scan for raw ---------------------------------------------------------------------------------

    def raw_scan_session_blocking(self) -> Any:
        """``library_core.RawScanSession`` over the last scan's rows (it keeps the workspaces missing a raw)."""
        lc = self._lc()
        if not hasattr(lc, "RawScanSession"):
            raise CoreMissing("library_core.RawScanSession")
        snap = self._snapshot
        return lc.RawScanSession([dict(b) for b in snap.all_books()], self.config_snapshot())

    def save_raw_scan_settings(self, session: Any) -> None:
        config = getattr(session, "_config", None)
        if isinstance(config, Mapping):
            for key in SCAN_RAW_KEYS:
                if key in config and self.cfg(key, None) != config[key]:
                    self.set_cfg(key, config[key])

    # ---- book details / metadata (blocking) --------------------------------------------------------

    def load_details_blocking(self, book: Mapping[str, Any], phase: str = "full") -> dict:
        """``library_core.load_book_details`` (``preview``: OPF + cover + metadata.json; ``full``: + chapters)."""
        lc = self._lc()
        if not hasattr(lc, "load_book_details"):
            raise CoreMissing("library_core.load_book_details")
        payload = lc.load_book_details(dict(book), self.config_snapshot(), phase)
        return dict(payload or {})

    def details_model(self, book: Mapping[str, Any], payload: Optional[Mapping[str, Any]] = None,
                      show_special_files: Optional[bool] = None) -> Any:
        """``library_core.BookDetailsModel`` (hero values, strip text, editor values); None without one."""
        lc = self.core.module("library_core")
        if lc is None or not hasattr(lc, "BookDetailsModel"):
            return None
        try:
            return lc.BookDetailsModel(dict(book), dict(payload or {}), self.config_snapshot(),
                                       show_special_files=show_special_files)
        except Exception:
            log.debug("BookDetailsModel failed", exc_info=True)
            return None

    def save_metadata_blocking(self, book: Mapping[str, Any], edits: Mapping[str, Any],
                               payload: Optional[Mapping[str, Any]] = None) -> Any:
        """``save_metadata_json_atomic``: merge the edited fields (``original_*`` kept,
        ``<field>_translated`` set) into the workspace metadata.json atomically."""
        lc = self._lc()
        if not hasattr(lc, "save_metadata_json_atomic"):
            raise CoreMissing("library_core.save_metadata_json_atomic")
        result = lc.save_metadata_json_atomic(dict(book), dict(edits), dict(payload or {}), self.config_snapshot())
        self.mark_dirty()
        return result

    def save_metadata_json_text_blocking(self, book: Mapping[str, Any], text: str) -> str:
        """Raw metadata.json editor (⋯ Edit metadata.json): validated JSON, atomic replace."""
        folder = str(book.get("output_folder") or "")
        if not folder or not os.path.isdir(folder):
            raise ValueError("This book has no output workspace")
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("metadata.json must hold a JSON object")
        path = os.path.join(folder, "metadata.json")
        tmp = f"{path}.{os.getpid()}.tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as handle:
                json.dump(data, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
        self.mark_dirty()
        return path

    # ---- covers (blocking) -----------------------------------------------------------------------

    def cover_blocking(self, book: Mapping[str, Any]) -> Optional[str]:
        """``library_covers.resolve_book_cover`` (the ``_CoverLoader`` chain; cache under the app cache)."""
        key = book_key(book)
        if key in self.covers:
            return self.covers[key]
        self.ensure_env()
        fn = self.core.fn("library_covers", "resolve_card_cover")
        path = None
        if fn is not None:
            try:
                found = fn(dict(book), self.config_snapshot())
            except Exception:
                log.debug("cover lookup failed", exc_info=True)
                found = None
            if isinstance(found, str) and found and os.path.isfile(found):
                path = found
        self.covers[key] = path
        return path

    # ---- outputs (blocking) ----------------------------------------------------------------------

    def compiled_outputs_blocking(self, book: Mapping[str, Any]) -> list:
        """``[(path, kind)]``: the workspace's compiled outputs (``list_compiled_outputs`` order) and a
        Library-filed EPUB (or a Library PDF, which has no workspace to compile) itself."""
        folder = str(book.get("output_folder") or "")
        out: list = []
        if folder and os.path.isdir(folder):
            fn = self.core.fn("library_core", "list_compiled_outputs")
            if fn is not None:
                for name, kind in fn(folder) or ():
                    path = name if os.path.isabs(str(name)) else os.path.join(folder, str(name))
                    out.append((path, str(kind)))
            # The PDF an EPUB book's Compile PDF writes is "<Title>.pdf" (``_pdf_worker``), which the shared
            # list (it knows "*_translated.pdf") leaves out: listed after the EPUBs, like the cloud sync takes
            # it (``job_kinds.compiled_outputs``), so the Output tab has its row (cloud line, save location).
            listed = {_norm(p) for p, _kind in out}
            raw = _norm(book.get("raw_source_path")) if book.get("raw_source_path") else ""
            try:
                names = sorted(os.listdir(folder), key=str.casefold)
            except OSError:
                names = []
            extra = [(os.path.join(folder, n), "pdf") for n in names if n.lower().endswith(".pdf")
                     and os.path.isfile(os.path.join(folder, n))
                     and _norm(os.path.join(folder, n)) not in listed and _norm(os.path.join(folder, n)) != raw]
            if extra:
                at = sum(1 for _p, kind in out if kind == "epub")
                out[at:at] = extra
        path = str(book.get("path") or "")
        lower = path.lower()
        kind = "epub" if lower.endswith(".epub") else (
            "pdf" if lower.endswith(".pdf") and book.get("in_library") and str(book.get("type") or "") == "pdf"
            else "")
        if kind and os.path.isfile(path) and all(_norm(p) != _norm(path) for p, _ in out):
            out.insert(0, (path, kind))
        return out

    # ---- jobs ------------------------------------------------------------------------------------

    def has_job_kind(self, kind: str) -> bool:
        jobs = self.jobs
        checker = getattr(jobs, "has_kind", None) if jobs is not None else None
        if callable(checker):
            try:
                return bool(checker(kind))
            except Exception:
                return False
        return False

    def origin_for(self, book: Mapping[str, Any]) -> dict:
        return {"type": "library", "bid": self.bid_for(book), "label": f"Library · {book.get('name') or ''}"}

    def compile_spec(self, book: Mapping[str, Any], kind: str = "compile_epub", folder: Optional[str] = None) -> Any:
        """A compile job over ``folder``, else the book's ``output_folder``, else the workspace the scan
        resolved for a Library row (:meth:`workspace_for`)."""
        from glossarion_mobile.services.jobs import JobSpec

        folder = str(folder or book.get("output_folder") or "") or self.workspace_for(book)
        if not folder:
            raise ValueError("This book has no output workspace to compile")
        params: dict = {"folder": folder}
        # The desktop compiles a workspace with the compiler of its source format
        # (library_core._workspace_compile_kind): a PDF workspace through the PDF workspace
        # compiler, an EPUB workspace through the EPUB converter, whose "Create PDF after EPUB"
        # (enable_pdf_output) gives its PDF - the Converter's compile_spec does the same.
        workspace_kind = "epub"
        decide = self.core.fn("library_core", "_workspace_compile_kind")
        if decide is not None:
            try:
                workspace_kind = str(decide(dict(book), folder) or "epub")
            except Exception:
                workspace_kind = "epub"
        if workspace_kind == "pdf":
            kind = "compile_pdf"
        elif kind == "compile_pdf":
            kind = "compile_epub"
            params["config_overrides"] = {"enable_pdf_output": True}
            params["pdf_after_epub"] = True
        return JobSpec(kind=kind, title=str(book.get("name") or os.path.basename(folder)), inputs=(folder,),
                       params=params, origin=self.origin_for(book))

    def translate_spec(self, books: Sequence[Mapping[str, Any]], *, review_glossary: bool = False,
                       sources: Optional[Sequence[str]] = None,
                       config_overrides: Optional[Mapping[str, Any]] = None) -> Any:
        """A ``translate`` job over the raw sources (desktop "Load for translation" + Run).
        ``config_overrides``: the TranslateSheet's "Only for this run" options (merged over the config
        snapshot at job start)."""
        from glossarion_mobile.services.jobs import JobSpec

        resolved = list(sources) if sources is not None else [self.raw_source(b) for b in books]
        inputs = tuple(s for s in resolved if s)
        if not inputs:
            raise ValueError("No raw source file resolves for the selection")
        first = books[0]
        title = str(first.get("name") or os.path.basename(inputs[0]))
        if len(inputs) > 1:
            title = f"{title} +{len(inputs) - 1}"
        params: dict = {"review_glossary": True} if review_glossary else {}
        if config_overrides:
            params["config_overrides"] = dict(config_overrides)
        origin = self.origin_for(first) if len(books) == 1 else {"type": "library", "label": "Library"}
        return JobSpec(kind="translate", title=title, inputs=inputs, params=params, origin=origin)

    def metadata_spec(self, books: Sequence[Mapping[str, Any]]) -> Any:
        """A ``metadata`` job (desktop ``start_metadata_translation(paths, output_roots=)``)."""
        from glossarion_mobile.services.jobs import JobSpec

        sources = tuple(s for s in (self.raw_source(b) for b in books) if s)
        if not sources:
            raise ValueError("No raw EPUB resolves for the selection")
        roots = [str(b.get("output_folder") or "") or self.workspace_for(b) for b in books]
        title = str(books[0].get("name") or "") if len(books) == 1 else f"{len(books)} books"
        origin = self.origin_for(books[0]) if len(books) == 1 else {"type": "library", "label": "Library"}
        return JobSpec(kind="metadata", title=title, inputs=sources, params={"output_roots": roots}, origin=origin)

    def raw_source(self, book: Mapping[str, Any]) -> str:
        """The book's raw file: the scanned ``raw_source_path``, else the shared resolvers
        (``resolve_book_source_file`` for Library EPUBs, ``find_raw_source_for_folder`` for a
        workspace - e.g. a row synthesised from a deep link before the first scan)."""
        path = str(book.get("raw_source_path") or "")
        if path and os.path.isfile(path):
            return path
        self.ensure_env()
        resolvers = [("resolve_book_source_file", dict(book))]
        if book.get("output_folder"):
            resolvers.append(("find_raw_source_for_folder", str(book.get("output_folder"))))
        for name, argument in resolvers:
            fn = self.core.fn("library_core", name)
            if fn is None:
                continue
            try:
                resolved = fn(argument)
            except Exception:
                resolved = ""
            if resolved and os.path.isfile(str(resolved)):
                return str(resolved)
        return ""

    async def submit(self, spec: Any) -> Optional[str]:
        jobs = self.jobs
        if jobs is None:
            return None
        result = jobs.submit(spec)
        if asyncio.iscoroutine(result) or isinstance(result, asyncio.Future):
            result = await result
        if spec.kind in ("compile_epub", "compile_pdf"):
            for path in spec.inputs:
                self.compiling.add(_norm(path))
        return result

    def is_compiling(self, book: Mapping[str, Any]) -> bool:
        folder = str(book.get("output_folder") or "") or self.workspace_for(book)
        return bool(folder) and _norm(folder) in self.compiling

    # ---- finished jobs ---------------------------------------------------------------------------

    async def on_job_finished(self, job: Any, outputs: Optional[Sequence[str]] = None) -> list:
        """JobService hook: refresh the Library and mirror the outputs (Android).

        (a) the Library is marked dirty and, when a Library screen listens, rescanned at
        once; (b) on Android with the Storage switch ``mirror_outputs`` on, the outputs of
        a DONE job are copied once to the public ``Downloads/Glossarion`` folder through
        ``ui/screens/storage.mirror_output`` (MediaStore; UI_SPEC Appendix C). Once the
        U10 cloud sync is installed (``cloud_sync``) book outputs are left to its
        destination (the phone folder keeps them updated in place under
        ``Downloads/Glossarion/<book>/``) and only the other outputs are copied here. A
        chat run (``direct_text``) is never mirrored from here: its outputs sit in a
        temporary run root the chat moves away while this runs; the book reaches the U10
        destination once it lands in the Library. Returns the mirrored URIs.
        """
        spec = getattr(job, "spec", None)
        for path in getattr(spec, "inputs", ()) or ():
            self.compiling.discard(_norm(path))
        params = getattr(spec, "params", {}) or {}
        if params.get("folder"):
            self.compiling.discard(_norm(params["folder"]))
        self.mark_dirty()
        self.covers = {k: v for k, v in self.covers.items() if v}  # placeholders: retry after a job
        if self._listeners:
            try:
                await self.refresh(quiet=True, reason="job finished")
            except Exception:
                log.exception("library refresh after a job failed")
        state = getattr(getattr(job, "state", None), "value", getattr(job, "state", None))
        if str(state) != "DONE":
            return []
        targets = list(outputs if outputs is not None else (getattr(job, "outputs", ()) or ()))
        if not targets or self.files is None:
            return []
        if str(getattr(spec, "kind", "") or "") in _RUN_ROOT_KINDS:
            return []
        try:
            from glossarion_mobile.ui.screens.storage import mirror_output
        except Exception:  # Flet missing (host tools): nothing to mirror
            return []
        skip_books = self.cloud_sync is not None
        saved: list = []
        for path in targets:
            try:
                saved.extend(await mirror_output(self.files, path, self.prefs, skip_books=skip_books))
            except Exception:
                log.exception("mirroring %s failed", path)
        self.mirrored = saved
        return saved
