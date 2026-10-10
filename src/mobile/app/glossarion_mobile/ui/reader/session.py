"""ReaderSession: one open book in the Reader (UI_SPEC §3.11 "Modes"), no Flet.

Everything that decides content comes from the shared GUI-free cores, reached
through ``services.library.SharedCore`` / ``bind_call`` (the Library's lazy,
contract-name importer: a function that does not exist in this build raises
``CoreMissing`` and the Reader shows the feature disabled with a reason):

* ``reader_doc``: ``load_epub_chapters`` (+ its pickle cache, pointed at the app
  cache), ``merge_overlay`` / ``overlay_signature``, ``READER_THEMES``,
  ``search_chapters``, ``chapter_display_numbers``, ``load_native_toc`` /
  ``map_native_toc``, ``google_translate_url`` / ``define_url``,
  ``build_bilingual_chapter``, ``load_workspace_chapters``, the image resolver
  (pages themselves are built by ``document.DocumentBuilder`` over
  ``reader_doc.ReaderDocument``);
* ``live_stream.resolve_live_output_folder`` (where a live run writes);
* ``reader_overlay.make_epub_overlay_provider`` (the in-progress overlay and its
  3 s refresh), ``workspace_reader`` (PDF/TXT workspaces: manifest + lazy raw
  PDF sections);
* ``library_core``: the raw-source / output-folder resolvers,
  ``mark_chapter_pending_for_retranslation``, ``chapter_completed_in_progress``,
  ``cleanup_incomplete_chapter_output`` and the special-file rule.

Open modes (``plan_open``: the shared ``library_core.plan_open_reader`` decision, the
desktop ``BookDetailsDialog._open_reader``): **workspace** (a translation
workspace whose raw source is a PDF, or any workspace without an EPUB),
**overlay** (in-progress book: raw EPUB + translated responses, refreshed
while visible), **dual** (completed book whose raw EPUB resolves: the compiled
EPUB and the raw one swap on Original/Translated) and **plain**.

TXT books (``OpenPlan.source_kind`` "txt"; ``text_book``): a ``.txt`` the desktop hands to the
system editor opens in plain mode as **text** (paragraph-bounded sections, or the sections of a
compiled ``_translated.txt``), and a TXT translation workspace whose split exists opens in
workspace mode on its sections (raw ``word_count`` text + translated responses: Original /
Translated / Bilingual).

Blocking methods (``load``, ``refresh_overlay``, ``set_flavor`` for dual
reloads, ``ensure_workspace_raw``, ``search``, ``image_bytes``) run on the io
pool. Python 3.10 compatible.
"""

from __future__ import annotations

import hashlib
import html as html_lib
import logging
import os
import threading
import zipfile
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence

from glossarion_mobile.services.library import CoreMissing, SharedCore, bind_call
from glossarion_mobile.ui.reader import model as rm
from glossarion_mobile.ui.reader import text_book

__all__ = [
    "DocEngine",
    "MODE_DUAL",
    "MODE_OVERLAY",
    "MODE_PLAIN",
    "MODE_WORKSPACE",
    "OpenPlan",
    "ReaderSession",
    "SOURCE_EPUB",
    "SOURCE_TXT",
    "TXT_TRANSLATE_REASON",
    "plan_for_file",
    "plan_open",
]

log = logging.getLogger("glossarion.reader")

MODE_PLAIN = "plain"
MODE_OVERLAY = "overlay"
MODE_DUAL = "dual"
MODE_WORKSPACE = "workspace"

#: ``OpenPlan.source_kind``: "txt" for a TXT book (text mode) or a TXT translation workspace.
SOURCE_EPUB = "epub"
SOURCE_TXT = "txt"
TXT_TRANSLATE_REASON = "Translate TXT files from the Library (whole file)"

_HTML_EXT = (".html", ".xhtml", ".htm")


def _norm(path: Any) -> str:
    try:
        return os.path.normcase(os.path.abspath(os.fspath(path)))
    except Exception:
        return str(path or "")


def _is_file(path: Any, ext: Optional[str] = None) -> bool:
    text = str(path or "")
    return bool(text) and os.path.isfile(text) and (ext is None or text.lower().endswith(ext))


def _base(name: Any) -> str:
    return os.path.basename(str(name or "").replace("\\", "/")).lower()


def _is_txt(path: Any) -> bool:
    return str(path or "").lower().endswith(".txt")


# ---------------------------------------------------------------------------
# Shared-core access
# ---------------------------------------------------------------------------


class DocEngine:
    """The shared reader functions the session calls (each resolved by contract name)."""

    def __init__(self, core: Optional[SharedCore] = None, *, config: Optional[Mapping[str, Any]] = None) -> None:
        self.core = core or SharedCore()
        self.config = dict(config or {})

    # ---- lookups -----------------------------------------------------------------------

    def fn(self, module: str, *names: str) -> Optional[Callable[..., Any]]:
        return self.core.fn(module, *names)

    def has(self, module: str, *names: str) -> bool:
        return self.fn(module, *names) is not None

    def call(self, module: str, names: Sequence[str], *args: Any, **available: Any) -> Any:
        """``module.<first existing name>(*args, <declared keyword args>)``; ``CoreMissing`` when absent."""
        fn = self.core.fn(module, *names)
        if fn is None:
            raise CoreMissing(f"{module}.{names[0]}")
        return bind_call(fn, *args, **available)

    def themes(self) -> list:
        value = self.core.value("reader_doc", "READER_THEMES", "_READER_THEMES")
        if isinstance(value, (list, tuple)) and value:
            return [dict(t) for t in value]
        return []

    # ---- library_core -----------------------------------------------------------------------

    def source_pointer(self, folder: str) -> str:
        fn = self.fn("library_core", "read_source_epub_pointer", "_read_source_epub_pointer")
        if fn is None or not folder:
            return ""
        try:
            return str(fn(folder) or "")
        except Exception:
            return ""

    def output_folder_for(self, book: Mapping[str, Any]) -> str:
        folder = str(book.get("output_folder") or "")
        if folder:
            return folder
        fn = self.fn("library_core", "resolve_book_output_folder", "_resolve_book_output_folder")
        if fn is None:
            return ""
        try:
            return str(bind_call(fn, dict(book), book=dict(book), config=self.config) or "")
        except Exception:
            return ""

    def output_roots(self) -> list:
        fn = self.fn("library_core", "resolve_output_roots", "_resolve_output_roots")
        if fn is None:
            roots = [os.environ.get("OUTPUT_DIRECTORY", "")]
            return [r for r in roots if r]
        try:
            return [str(r) for r in (bind_call(fn, self.config, config=self.config) or [])]
        except Exception:
            return []

    def is_special(self, filename: str) -> bool:
        fn = self.fn("library_core", "is_configured_special_file", "_is_configured_special_file")
        if fn is None:
            return False
        try:
            return bool(fn(filename, self.config))
        except Exception:
            return False

    def mark_pending(self, folder: str, chapter_file: str) -> bool:
        return bool(self.call("library_core", ("mark_chapter_pending_for_retranslation",
                                                 "_mark_chapter_pending_for_retranslation", "mark_chapter_pending"),
                              folder, chapter_file))

    def completion_fns(self) -> tuple:
        completed = self.fn("library_core", "chapter_completed_in_progress", "_chapter_completed_in_progress",
                            "chapter_completed")
        cleanup = self.fn("library_core", "cleanup_incomplete_chapter_output", "_cleanup_incomplete_chapter_output")
        return completed, cleanup

    # ---- reader_doc ---------------------------------------------------------------------------

    def use_cache_dir(self, cache_dir: Optional[str]) -> None:
        """Point the shared EPUB pickle cache at the app cache (once; a Library env may already have)."""
        if not cache_dir:
            return
        module = self.core.module("reader_doc")
        if module is None or getattr(module, "_EPUB_CACHE_DIR_OVERRIDE", None):
            return
        setter = getattr(module, "set_epub_cache_dir", None)
        if callable(setter):
            try:
                os.makedirs(os.path.join(cache_dir, "epub"), exist_ok=True)
                setter(os.path.join(cache_dir, "epub"))
            except Exception:
                log.debug("set_epub_cache_dir failed", exc_info=True)

    def load_epub(self, path: str, *, show_special: bool, cache_dir: Optional[str],
                  should_stop: Optional[Callable[[], bool]] = None) -> tuple:
        self.use_cache_dir(cache_dir)
        result = self.call("reader_doc", ("load_epub_chapters",), path, show_special_files=show_special,
                           config=self.config, should_stop=should_stop, use_cache=True)
        if result is None:
            raise RuntimeError("Loading the EPUB was cancelled")
        return _unpack_loaded(result)

    def load_text(self, path: str, *, should_stop: Optional[Callable[[], bool]] = None) -> tuple:
        """A ``.txt`` as (chapters, images, filenames): ``text_book.load_text_chapters``."""
        result = text_book.load_text_chapters(path, should_stop=should_stop)
        if result is None:
            raise RuntimeError("Loading the text file was cancelled")
        return _unpack_loaded(result)

    def merge(self, raw: list, images: Mapping, filenames: list, overlay: Mapping, extra_dirs: list,
              previous: Optional[list]) -> tuple:
        """(overlaid chapters, images, applied, retry_required, read signature)."""
        signature = self.signature(overlay)
        result = self.call("reader_doc", ("merge_overlay",), raw, dict(images), list(filenames), dict(overlay),
                           extra_image_dirs=list(extra_dirs), config=self.config, previous_chapters=previous)
        overlaid, merged_images, applied, retry = raw, images, False, False
        if result is not None and hasattr(result, "chapters") and hasattr(result, "overlay_applied"):
            # reader_doc.OverlayMergeResult (what _OverlayMergeThread hands the desktop dialog)
            overlaid = list(result.chapters or raw)
            merged_images = result.images or images
            applied = bool(result.overlay_applied)
            retry = bool(getattr(result, "retry_required", False))
            signature = getattr(result, "read_signature", signature)
            return list(overlaid), dict(merged_images or {}), applied, retry, signature
        if isinstance(result, Mapping):
            overlaid = list(result.get("chapters") or result.get("overlaid") or raw)
            merged_images = result.get("images") or images
            applied = bool(result.get("applied", result.get("overlay_applied", False)))
            retry = bool(result.get("retry", result.get("retry_required", False)))
            signature = result.get("signature", signature)
        elif isinstance(result, (tuple, list)) and result and isinstance(result[0], (list, tuple)) \
                and (not result[0] or isinstance(result[0][0], (list, tuple))):
            overlaid = list(result[0])
            if len(result) > 1 and isinstance(result[1], Mapping):
                merged_images = result[1]
            if len(result) > 2:
                applied = bool(result[2])
            if len(result) > 3:
                retry = bool(result[3])
        elif isinstance(result, list):
            overlaid = result
            applied = any(a != b for a, b in zip(result, raw, strict=False))
        if signature != self.signature(overlay):
            retry = True
        return list(overlaid), dict(merged_images or {}), applied, retry, signature

    def signature(self, overlay: Mapping) -> Any:
        fn = self.fn("reader_doc", "overlay_signature", "reader_overlay_signature", "_reader_overlay_signature")
        if fn is None:
            return tuple(sorted((k, str((v or {}).get("path") or "")) for k, v in overlay.items()))
        try:
            return fn(dict(overlay))
        except Exception:
            return None

    def display_numbers(self, filenames: list) -> list:
        fn = self.fn("reader_doc", "chapter_display_numbers")
        if fn is not None:
            try:
                numbers = bind_call(fn, list(filenames), filenames=list(filenames), config=self.config)
                if numbers is not None and len(list(numbers)) == len(filenames):
                    return list(numbers)
            except Exception:
                log.debug("chapter_display_numbers failed", exc_info=True)
        try:
            from chapter_display_numbering import filename_chapter_number, nonreset_chapter_display_numbers

            return list(nonreset_chapter_display_numbers(
                filename_chapter_number(name, is_special=self.is_special(name)) for name in filenames))
        except Exception:
            return list(range(1, len(filenames) + 1))

    def native_toc(self, toc_dir: str, epub_path: str, filenames: list) -> list:
        load = self.fn("reader_doc", "load_native_toc", "load_reader_native_toc", "_load_reader_native_toc")
        mapper = self.fn("reader_doc", "map_native_toc", "map_native_toc_to_chapters", "_map_native_toc_to_chapters")
        if load is None or mapper is None:
            return []
        try:
            entries = load(toc_dir or "", epub_path or "")
            return list(mapper(entries, list(filenames)) or [])
        except Exception:
            log.debug("native TOC failed", exc_info=True)
            return []

    def search(self, chapters: list, query: str, *, on_batch: Optional[Callable[[list, bool], Any]] = None,
               should_stop: Optional[Callable[[], bool]] = None) -> list:
        """``reader_doc.search_chapters``: every row; ``on_batch(rows, done)`` gets the 120-row batches."""
        rows = self.call("reader_doc", ("search_chapters",), chapters, query, config=self.config,
                         should_stop=should_stop, on_batch=on_batch)
        return [dict(r) for r in (rows or []) if isinstance(r, Mapping)]

    def bilingual(self, raw_html: str, translated_html: str) -> str:
        return str(self.call("reader_doc", ("build_bilingual_chapter",), raw_html, translated_html,
                             raw_html=raw_html, translated_html=translated_html))

    def google_translate_url(self, text: str, language: str) -> str:
        return str(self.call("reader_doc", ("google_translate_url",), text, language, text=text,
                             target_language=language, language=language, output_language=language))

    def define_url(self, text: str) -> str:
        return str(self.call("reader_doc", ("define_url", "web_define_url"), text, text=text))

    def workspace_manifest(self, workspace: str, source: Optional[str], *, source_kind: str = "") -> dict:
        """``workspace_reader.build_workspace_reader_manifest``; a TXT workspace (``source_kind`` "txt"
        or a ``.txt`` source) gets its sections from ``text_book.build_text_workspace_manifest``."""
        from workspace_reader import build_workspace_reader_manifest

        manifest = dict(build_workspace_reader_manifest(workspace, source_path=source or None))
        if source_kind == SOURCE_TXT or str(manifest.get("source_format") or "").lower() == SOURCE_TXT:
            return text_book.build_text_workspace_manifest(workspace, source_path=source or None, base=manifest)
        return manifest

    def workspace_chapters(self, manifest: Mapping) -> tuple:
        if str(manifest.get("source_format") or "").lower() == SOURCE_TXT:
            loaded = text_book.load_text_workspace_chapters(manifest)
            if loaded is None:
                raise RuntimeError("Loading the workspace was cancelled")
            raw, translated, filenames = loaded
            return list(raw), list(translated), list(filenames)
        result = self.call("reader_doc", ("load_workspace_chapters",), dict(manifest), manifest=dict(manifest))
        if isinstance(result, Mapping):
            return (list(result.get("raw") or result.get("raw_chapters") or []),
                    list(result.get("translated") or result.get("translated_chapters") or []),
                    list(result.get("filenames") or []))
        raw, translated, filenames = result
        return list(raw), list(translated), list(filenames)

    def ensure_pdf_raw(self, manifest: Mapping, entry: Mapping) -> str:
        from workspace_reader import ensure_pdf_raw_section

        path = ensure_pdf_raw_section(dict(manifest), dict(entry))
        with open(path, "r", encoding="utf-8", errors="replace") as stream:
            return stream.read()


def _unpack_loaded(result: Any) -> tuple:
    """(chapters [(title, html)], images {name: descriptor}, filenames) from a loader result."""
    if isinstance(result, Mapping):
        chapters = result.get("chapters") or []
        images = result.get("images") or {}
        filenames = result.get("filenames") or []
    elif isinstance(result, (tuple, list)) and len(result) == 3:
        chapters, images, filenames = result
    else:
        raise ValueError("reader_doc.load_epub_chapters returned an unexpected value")
    chapters = [(str(t or ""), str(h or "")) for t, h in chapters]
    return chapters, dict(images or {}), [str(f or "") for f in filenames]


# ---------------------------------------------------------------------------
# Open plan
# ---------------------------------------------------------------------------


@dataclass
class OpenPlan:
    mode: str
    epub_path: str = ""  # the EPUB the reader shows first (raw for overlay)
    translated_path: str = ""  # dual: compiled EPUB
    raw_path: str = ""  # dual: raw EPUB; overlay: the raw EPUB under the overlay
    output_folder: str = ""
    workspace_dir: str = ""
    source_path: str = ""  # workspace: the raw PDF/TXT source
    title: str = ""
    initial_raw: bool = False
    in_progress: bool = False
    toc_dir: str = ""
    css_dirs: list = field(default_factory=list)
    initial_chapter_filename: str = ""
    error: str = ""
    source_kind: str = SOURCE_EPUB  # SOURCE_TXT: epub_path is a .txt (text mode) / a TXT workspace


def plan_open(book: Mapping[str, Any], *, raw_only: bool = False, engine: Optional[DocEngine] = None,
              payload: Optional[Mapping[str, Any]] = None, show_special: Optional[bool] = None) -> OpenPlan:
    """How to open ``book`` (blocking): the shared ``library_core.plan_open_reader`` decision.

    The desktop ``BookDetailsDialog._open_reader`` decision lives in ``library_core``
    (``{"mode": "workspace" | "epub" | "system", "source", "kwargs"}`` with the
    ``EpubReaderDialog`` keyword arguments); this maps it onto the Reader's modes:

    * ``workspace`` -> **workspace** (``workspace_dir``; ``initial_show_raw``; without a
      details ``payload`` the decision cannot see the translated chapters, so only
      ``raw_only`` opens the raw side);
    * ``epub`` with an ``overlay_provider`` (an in-progress book) -> **overlay**: the Reader
      builds its own provider from the chapters it loaded (``reader_overlay``), because a
      payload-less decision has no chapter list;
    * ``epub`` with ``alt_epub_path`` -> **dual** (compiled EPUB + raw EPUB);
    * ``epub`` otherwise -> **plain**;
    * ``system`` (TXT / HTML / image workspaces, which the desktop hands to the OS viewer):
      a TXT book (its workspace's source, its raw source or the decision's target is a ``.txt``)
      -> **workspace** on the translation's sections once a run split it (``text_book``),
      else **plain** text mode on the decision's ``.txt`` (compiled output, or the raw file of a
      Not-started / Library book); any other book with a translation workspace -> **workspace**
      (the same manifest ``workspace_reader`` builds for PDF workspaces); else an error naming
      the file.
    """
    engine = engine or DocEngine()
    decide = engine.fn("library_core", "plan_open_reader")
    if decide is None:
        raise CoreMissing("library_core.plan_open_reader")
    plan = bind_call(decide, dict(book), book=dict(book), payload=dict(payload) if payload else None,
                     raw_only=bool(raw_only), config=engine.config, show_special_files=show_special)
    plan = dict(plan or {})
    kwargs = dict(plan.get("kwargs") or {})
    source = str(plan.get("source") or "")
    mode = str(plan.get("mode") or "")
    title = str(book.get("name") or book.get("folder_name") or "")
    in_progress = bool(book.get("is_in_progress"))
    output_folder = str(kwargs.get("toc_output_dir") or kwargs.get("workspace_dir") or "") \
        or engine.output_folder_for(book)
    if mode == "workspace":
        workspace = str(kwargs.get("workspace_dir") or output_folder)
        initial_raw = bool(kwargs.get("initial_show_raw")) if payload else bool(raw_only)
        return OpenPlan(mode=MODE_WORKSPACE, workspace_dir=workspace, source_path=source, output_folder=workspace,
                        title=title, initial_raw=initial_raw, in_progress=in_progress, toc_dir=workspace,
                        initial_chapter_filename=str(kwargs.get("initial_chapter_filename") or ""))
    if mode == "epub" and source:
        toc_dir = str(kwargs.get("toc_output_dir") or "") or os.path.dirname(source)
        css_dirs = [str(d) for d in (kwargs.get("translated_css_dirs") or []) if d]
        if kwargs.get("overlay_provider") is not None or kwargs.get("translated_overlay"):
            return OpenPlan(mode=MODE_OVERLAY, epub_path=source, raw_path=source, output_folder=output_folder,
                            title=title, in_progress=True, toc_dir=toc_dir, css_dirs=css_dirs,
                            initial_chapter_filename=str(kwargs.get("initial_chapter_filename") or ""))
        alt = str(kwargs.get("alt_epub_path") or "")
        if alt:
            return OpenPlan(mode=MODE_DUAL, epub_path=source, translated_path=source, raw_path=alt,
                            output_folder=output_folder, title=title, toc_dir=toc_dir,
                            initial_chapter_filename=str(kwargs.get("initial_chapter_filename") or ""))
        return OpenPlan(mode=MODE_PLAIN, epub_path=source, output_folder=output_folder, title=title,
                        initial_raw=bool(raw_only), in_progress=in_progress, toc_dir=toc_dir,
                        initial_chapter_filename=str(kwargs.get("initial_chapter_filename") or ""))
    target = str(plan.get("target") or book.get("path") or "")
    raw_source = str(book.get("raw_source_path") or "")
    if output_folder and os.path.isfile(os.path.join(output_folder, "translation_progress.json")):
        pointed = engine.source_pointer(output_folder)
        if _is_txt(pointed) or _is_txt(raw_source) or _is_txt(target):
            if text_book.has_text_workspace(output_folder):
                source = next((p for p in (pointed, raw_source) if _is_file(p, ".txt")), "")
                return OpenPlan(mode=MODE_WORKSPACE, workspace_dir=output_folder, source_path=source,
                                output_folder=output_folder, title=title, initial_raw=bool(raw_only),
                                in_progress=in_progress, toc_dir=output_folder, source_kind=SOURCE_TXT)
            text = next((p for p in (target, pointed, raw_source) if _is_file(p, ".txt")), "")
            if text:  # not split for translation yet (a Not-started import): the file itself
                return _text_plan(text, title=title, output_folder=output_folder, raw_only=raw_only,
                                  in_progress=in_progress)
        return OpenPlan(mode=MODE_WORKSPACE, workspace_dir=output_folder,
                        source_path=pointed if _is_file(pointed) else "", output_folder=output_folder, title=title,
                        initial_raw=False, in_progress=in_progress, toc_dir=output_folder)
    text = next((p for p in (target, raw_source) if _is_file(p, ".txt")), "")
    if text:
        return _text_plan(text, title=title, output_folder=output_folder, raw_only=raw_only, in_progress=in_progress)
    return OpenPlan(mode=MODE_PLAIN, title=title, error=(
        f"The Reader opens EPUB and TXT books and translation workspaces; "
        f"{os.path.basename(target) or 'this book'} is not one of them."))


def _text_plan(path: str, *, title: str = "", output_folder: str = "", raw_only: bool = False,
               in_progress: bool = False) -> OpenPlan:
    """Text mode: a ``.txt`` read in sections (``text_book.load_text_chapters``)."""
    path = os.path.abspath(path)
    return OpenPlan(mode=MODE_PLAIN, epub_path=path, output_folder=output_folder,
                    title=title or os.path.splitext(os.path.basename(path))[0], initial_raw=bool(raw_only),
                    in_progress=in_progress, toc_dir=os.path.dirname(path), source_kind=SOURCE_TXT)


def plan_for_file(path: str) -> OpenPlan:
    """A shared / picked EPUB or TXT opened directly (no Library row)."""
    if _is_txt(path):
        return _text_plan(path)
    return OpenPlan(mode=MODE_PLAIN, epub_path=os.path.abspath(path), title=os.path.splitext(os.path.basename(path))[0],
                    toc_dir=os.path.dirname(os.path.abspath(path)))


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


@dataclass
class ChapterInfo:
    index: int
    filename: str
    title_raw: str
    title_translated: str
    number: Any
    status: str  # overlay / workspace status ("" = none)
    has_raw: bool
    has_translation: bool


class ReaderSession:
    def __init__(self, plan: OpenPlan, *, engine: Optional[DocEngine] = None, cache_dir: Optional[str] = None,
                 show_special: bool = False) -> None:
        self.plan = plan
        self.engine = engine or DocEngine()
        self.cache_dir = cache_dir
        self.show_special = bool(show_special)
        self.lock = threading.RLock()
        self.loaded = False
        self.raw_chapters: list = []
        self.overlaid: list = []
        self.filenames: list = []
        self.images: dict = {}
        self.overlay: dict = {}
        self.extra_image_dirs: list = []
        self.overlay_applied = False
        self.retry_required = False
        self.overlay_signature: Any = None
        self.provider: Optional[Callable[[], Any]] = None
        self.display_numbers: list = []
        self.native_toc: list = []
        self.manifest: dict = {}
        self.workspace_raw_ready: set = set()
        self.dual_cache: dict = {}  # path -> (chapters, images, filenames)
        self.active_path = plan.epub_path
        self.flavor = rm.ORIGINAL if plan.initial_raw else rm.TRANSLATED
        self.generation = 0  # bumps whenever chapter content changes (documents re-render)
        self._zip: Optional[zipfile.ZipFile] = None
        self._zip_path = ""
        self._zip_names: dict = {}
        self._txt_layout_totals: dict = {}  # position_from_pref: sections of the other TXT layout

    @property
    def is_text(self) -> bool:
        """A TXT book: text mode or a TXT translation workspace."""
        return self.plan.source_kind == SOURCE_TXT or \
            str(self.manifest.get("source_format") or "").lower() == SOURCE_TXT

    # ---- loading (blocking) ----------------------------------------------------------------

    def load(self) -> None:
        plan = self.plan
        if plan.error:
            raise ValueError(plan.error)
        if plan.mode == MODE_WORKSPACE:
            self._load_workspace()
        else:
            path = plan.epub_path
            if plan.mode == MODE_DUAL and self.flavor == rm.ORIGINAL and plan.raw_path:
                path = plan.raw_path
            self._load_epub(path)
            if plan.mode == MODE_OVERLAY:
                self._attach_overlay(plan.output_folder)
            elif plan.mode == MODE_DUAL and self.flavor == rm.BILINGUAL:
                self._ensure_dual_both()  # a reload in Bilingual (Show special files) needs the raw EPUB too
        with self.lock:
            if self.is_text:  # sections in reading order; no EPUB TOC
                self.display_numbers = list(range(1, len(self.filenames) + 1))
                self.native_toc = []
            else:
                self.display_numbers = self.engine.display_numbers(self.filenames)
                self.native_toc = self.engine.native_toc(plan.toc_dir, self.active_path, self.filenames)
            self.loaded = True
            self.generation += 1

    def _load_epub(self, path: str) -> None:
        cached = self.dual_cache.get(_norm(path))
        if cached is None:
            cached = self._read_source(path)
            if self.plan.mode == MODE_DUAL:  # TXT too: a 3 MB book was re-read and re-split on every switch
                self.dual_cache[_norm(path)] = cached
        chapters, images, filenames = cached
        with self.lock:
            self.active_path = path
            self.raw_chapters = list(chapters)
            self.overlaid = list(chapters)
            self.images = dict(images)
            self.filenames = [_base(f) for f in filenames]
            self._close_zip()

    def _attach_overlay(self, output_folder: str, *, initial: Optional[Mapping] = None) -> None:
        from reader_overlay import make_epub_overlay_provider

        self.provider = make_epub_overlay_provider(output_folder, list(self.filenames), initial_overlay=initial)
        self.refresh_overlay(force=True)

    def _load_workspace(self) -> None:
        manifest = self.engine.workspace_manifest(self.plan.workspace_dir, self.plan.source_path,
                                                  source_kind=self.plan.source_kind)
        raw, translated, filenames = self.engine.workspace_chapters(manifest)
        with self.lock:
            self.manifest = manifest
            self.active_path = str(manifest.get("source_path") or manifest.get("workspace") or "")
            self.raw_chapters = list(raw)
            self.overlaid = list(translated)
            self.filenames = [_base(f) for f in filenames]
            self.extra_image_dirs = list(manifest.get("image_dirs") or [])
            self.images = {}
            self.overlay_applied = any(str(e.get("translated_path") or "") for e in manifest.get("entries") or [])

    # ---- overlay refresh (blocking) ----------------------------------------------------------

    def refresh_overlay(self, *, force: bool = False) -> set:
        """Poll the overlay provider; re-merge when the overlay changed. Returns the changed rows."""
        provider = self.provider
        if provider is None or not self.raw_chapters:
            return set()
        try:
            result = provider()
        except Exception:
            log.debug("overlay provider failed", exc_info=True)
            return set()
        if isinstance(result, tuple) and len(result) == 2:
            raw_overlay, extra_dirs = result
        elif isinstance(result, Mapping):
            raw_overlay, extra_dirs = result, self.extra_image_dirs
        else:
            return set()
        normalised: dict = {}
        for key, value in (raw_overlay or {}).items():
            name = _base(key)
            if not name:
                continue
            if isinstance(value, str):
                normalised[name] = {"path": value}
            elif isinstance(value, Mapping) and value.get("path"):
                normalised[name] = dict(value)
        signature = self.engine.signature(normalised)
        if (not force and signature == self.overlay_signature and list(extra_dirs or []) == self.extra_image_dirs
                and not self.retry_required):
            return set()
        with self.lock:
            raw = list(self.raw_chapters)
            filenames = list(self.filenames)
            previous = list(self.overlaid)
            images = dict(self.images)
        overlaid, merged_images, applied, retry, read_signature = self.engine.merge(
            raw, images, filenames, normalised, list(extra_dirs or []), previous)
        with self.lock:
            changed = {i for i in range(max(len(previous), len(overlaid)))
                       if i >= len(previous) or i >= len(overlaid) or previous[i] != overlaid[i]}
            self.overlay = normalised
            self.extra_image_dirs = list(extra_dirs or [])
            self.overlaid = overlaid
            self.images = merged_images or images
            self.overlay_applied = bool(applied)
            self.retry_required = bool(retry)
            self.overlay_signature = read_signature
            if changed:
                self.generation += 1
        return changed

    def adopt_output_folder(self, output_folder: str) -> set:
        """Plain EPUB translated live: overlay its new workspace (blocking)."""
        if self.plan.mode != MODE_PLAIN or self.plan.source_kind == SOURCE_TXT or not output_folder \
                or not os.path.isdir(output_folder):
            return set()
        css_dirs: list = []
        try:  # the shared decision for an in-progress book gives its translated CSS folders
            decided = plan_open({"name": self.plan.title, "path": output_folder, "output_folder": output_folder,
                                 "raw_source_path": self.plan.epub_path, "is_in_progress": True}, engine=self.engine)
            css_dirs = list(decided.css_dirs)
        except Exception:
            log.debug("planning the adopted workspace failed", exc_info=True)
        self.plan.mode = MODE_OVERLAY
        self.plan.output_folder = output_folder
        self.plan.raw_path = self.plan.epub_path
        self.plan.css_dirs = css_dirs
        self._attach_overlay(output_folder)
        return set(range(len(self.raw_chapters)))

    # ---- flavours -----------------------------------------------------------------------------

    @property
    def has_alternate(self) -> bool:
        mode = self.plan.mode
        if mode == MODE_OVERLAY:
            return self.overlay_applied
        if mode == MODE_DUAL:
            return bool(self.plan.raw_path)
        if mode == MODE_WORKSPACE:  # PDF: raw pages on demand; TXT: the raw word_count sections
            return str(self.manifest.get("source_format") or "").lower() in ("pdf", SOURCE_TXT)
        return False

    def set_flavor(self, flavor: str) -> bool:
        """Switch Original / Translated / Bilingual (blocking for dual reloads). True when content changed."""
        if flavor not in rm.READER_MODES or flavor == self.flavor:
            return False
        previous = self.flavor
        self.flavor = flavor
        if self.plan.mode == MODE_DUAL:
            wanted_raw = flavor == rm.ORIGINAL
            was_raw = previous == rm.ORIGINAL
            if flavor == rm.BILINGUAL:
                self._ensure_dual_both()
            elif wanted_raw != was_raw:
                self._load_epub(self.plan.raw_path if wanted_raw else self.plan.translated_path)
                with self.lock:
                    self.display_numbers = self.engine.display_numbers(self.filenames)
                    self.native_toc = self.engine.native_toc(self.plan.toc_dir, self.active_path, self.filenames)
        with self.lock:
            self.generation += 1
        return True

    def _read_source(self, path: str) -> tuple:
        """(chapters, images, filenames) of one version of the book (TXT sections or EPUB chapters)."""
        if self.plan.source_kind == SOURCE_TXT:
            return tuple(self.engine.load_text(path))
        return tuple(self.engine.load_epub(path, show_special=self.show_special, cache_dir=self.cache_dir))

    def prefetch(self) -> None:
        """Load every version of a dual book into ``dual_cache`` (blocking; the Reader runs it on the io pool
        after the first page), so Original / Translated / Bilingual switch without reading the files again."""
        if self.plan.mode != MODE_DUAL:
            return
        for path in (self.plan.raw_path, self.plan.translated_path):
            if path and _norm(path) not in self.dual_cache:
                try:
                    self.dual_cache[_norm(path)] = self._read_source(path)
                except Exception:
                    log.debug("prefetching %s failed", path, exc_info=True)

    def _ensure_dual_both(self) -> None:
        for path in (self.plan.raw_path, self.plan.translated_path):
            if path and _norm(path) not in self.dual_cache:
                self.dual_cache[_norm(path)] = self._read_source(path)
        translated = self.dual_cache.get(_norm(self.plan.translated_path))
        if translated is not None and _norm(self.active_path) != _norm(self.plan.translated_path):
            self._load_epub(self.plan.translated_path)

    def _dual_raw_html(self, index: int) -> Optional[tuple]:
        cached = self.dual_cache.get(_norm(self.plan.raw_path))
        if cached is None:
            return None
        chapters, _images, filenames = cached
        wanted = self.filenames[index] if index < len(self.filenames) else ""
        names = [_base(f) for f in filenames]
        if wanted in names:
            return chapters[names.index(wanted)]
        return chapters[index] if index < len(chapters) else None

    # ---- chapters -------------------------------------------------------------------------------

    @property
    def count(self) -> int:
        return len(self.raw_chapters) if self.flavor == rm.ORIGINAL else len(self.overlaid or self.raw_chapters)

    def _translated_list(self) -> list:
        return self.overlaid or self.raw_chapters

    def chapter_info(self, index: int) -> ChapterInfo:
        raw = self.raw_chapters[index] if index < len(self.raw_chapters) else ("", "")
        translated = self._translated_list()[index] if index < len(self._translated_list()) else raw
        filename = self.filenames[index] if index < len(self.filenames) else ""
        status = ""
        has_translation = True
        has_raw = True
        mode = self.plan.mode
        if mode == MODE_OVERLAY:
            entry = self.overlay.get(filename) or {}
            status = str(entry.get("status") or ("" if not entry else "completed")).strip().lower()
            has_translation = bool(entry.get("path"))
        elif mode == MODE_WORKSPACE:
            entries = self.manifest.get("entries") or []
            entry = entries[index] if index < len(entries) else {}
            status = str(entry.get("status") or "").strip().lower()
            has_translation = bool(entry.get("translated_path"))
            has_raw = self.has_alternate
        elif mode == MODE_DUAL:
            has_raw = bool(self.plan.raw_path)
        number = self.display_numbers[index] if index < len(self.display_numbers) else index + 1
        return ChapterInfo(index=index, filename=filename, title_raw=str(raw[0] or ""),
                           title_translated=str(translated[0] or ""), number=number, status=status,
                           has_raw=has_raw, has_translation=has_translation)

    def titles(self) -> list:
        source = self.raw_chapters if self.flavor == rm.ORIGINAL else self._translated_list()
        return [str(t or "") for t, _h in source]

    def statuses(self) -> list:
        return [self.chapter_info(i).status for i in range(len(self.filenames))]

    def chapter_title(self, index: int) -> str:
        info = self.chapter_info(index)
        title = info.title_raw if self.flavor == rm.ORIGINAL else info.title_translated
        return title or rm.chapter_label(info.number, "")

    def available_modes(self, index: int) -> dict:
        info = self.chapter_info(index) if 0 <= index < max(1, len(self.filenames)) else None
        has_translation = bool(info.has_translation) if info is not None else False
        if self.plan.mode == MODE_DUAL:
            has_translation = True
        available = rm.mode_availability(has_alternate=self.has_alternate,
                                         chapter_has_raw=bool(info.has_raw) if info is not None else False,
                                         chapter_has_translation=has_translation)
        if available.get(rm.BILINGUAL) and not self.engine.has("reader_doc", "build_bilingual_chapter"):
            available[rm.BILINGUAL] = False
        return available

    def effective_flavor(self, index: int) -> str:
        """The flavour chapter ``index`` shows: Bilingual needs both versions of the chapter, else the
        translation (the segment ``ReaderChrome.set_modes`` highlights); the chosen Bilingual comes back on
        the next chapter that has both."""
        if self.flavor == rm.BILINGUAL and not self.available_modes(index).get(rm.BILINGUAL):
            return rm.TRANSLATED
        return self.flavor

    def chapter_html(self, index: int, flavor: Optional[str] = None) -> str:
        flavor = flavor or self.effective_flavor(index)
        with self.lock:
            if not 0 <= index < max(len(self.raw_chapters), len(self.overlaid)):
                return ""
            raw = self.raw_chapters[index][1] if index < len(self.raw_chapters) else ""
            translated = self._translated_list()[index][1] if index < len(self._translated_list()) else raw
        if self.plan.mode == MODE_DUAL:
            if flavor == rm.BILINGUAL:
                raw_pair = self._dual_raw_html(index)
                return self.engine.bilingual(raw_pair[1] if raw_pair else "", translated)
            return translated if _norm(self.active_path) == _norm(self.plan.translated_path) or flavor != rm.ORIGINAL \
                else raw
        if flavor == rm.ORIGINAL:
            return raw
        if flavor == rm.BILINGUAL:
            return self.engine.bilingual(raw, translated)
        return translated

    def ensure_workspace_raw(self, index: int) -> bool:
        """PDF workspaces extract a section's raw pages on first view (blocking). True when replaced."""
        if self.plan.mode != MODE_WORKSPACE or index in self.workspace_raw_ready or not self.has_alternate:
            return False
        if str(self.manifest.get("source_format") or "").lower() != "pdf":
            return False  # a TXT workspace loaded its raw sections already
        entries = self.manifest.get("entries") or []
        if not 0 <= index < len(entries):
            return False
        try:
            content = self.engine.ensure_pdf_raw(self.manifest, entries[index])
        except Exception as exc:
            content = ('<!DOCTYPE html><html><head><meta charset="utf-8"></head><body>'
                       '<div style="max-width:48em;margin:4em auto;text-align:center;opacity:.72">'
                       f'<h2>{html_lib.escape(self.chapter_title(index))}</h2>'
                       f'<p>{html_lib.escape(str(exc))}</p></div></body></html>')
        with self.lock:
            title = self.raw_chapters[index][0] if index < len(self.raw_chapters) else ""
            self.raw_chapters[index] = (title, content)
            self.workspace_raw_ready.add(index)
            self.generation += 1
        return True

    # ---- saved positions --------------------------------------------------------------------------

    def position_from_pref(self, data: Optional[Mapping[str, Any]]) -> Optional[rm.Position]:
        """A saved ``reader_positions`` entry or bookmark for this book (``rm.Position.from_pref``).

        A TXT book has two section layouts: the Reader's own sections (``section_0001.txt``...) until
        a translation run splits it, then the translation's (``word_count`` names). A saved href of
        the other layout restores at the same place in the book (its book percent) instead of the
        same section number. Blocking the first time (it may count the other layout's sections)."""
        position = rm.Position.from_pref(data, self.filenames)
        if not isinstance(data, Mapping) or not self.is_text or not self.filenames:
            return position
        href = _base(str(data.get("href") or "").split("#", 1)[0])
        if not href or href in self.filenames:
            return position
        total = self._txt_layout_total(href)
        try:
            chapter = int(data.get("chapter"))
            fraction = float(data.get("fraction") or 0.0)
        except (TypeError, ValueError):
            return position
        if total <= 0 or fraction != fraction:
            return position
        share = (min(max(chapter, 0), total - 1) + max(0.0, min(1.0, fraction))) / float(total)
        exact = max(0.0, min(1.0, share)) * len(self.filenames)
        index = min(len(self.filenames) - 1, int(exact))
        mode = data.get("mode") if data.get("mode") in rm.READER_MODES else rm.TRANSLATED
        return rm.Position(chapter=index, href=self.filenames[index], fraction=max(0.0, min(1.0, exact - index)),
                           mode=mode)

    def _txt_layout_total(self, href: str) -> int:
        """Sections in the TXT layout ``href`` belongs to when it is not the open one (0: unknown)."""
        reader_layout = bool(text_book.READER_SECTION_NAME.match(href))
        workspace_open = self.plan.mode == MODE_WORKSPACE
        if reader_layout != workspace_open:
            return 0  # a name of the open layout that is gone (the file changed): keep the index rule
        key = "reader" if reader_layout else "workspace"
        with self.lock:
            if key in self._txt_layout_totals:
                return self._txt_layout_totals[key]
        total = 0
        try:
            if reader_layout:  # the TXT workspace's source, read in the Reader's own sections
                source = next((p for p in (self.plan.source_path, str(self.manifest.get("source_path") or ""))
                               if _is_file(p, ".txt")), "")
                total = text_book.text_section_count(source) if source else 0
            else:  # text mode: the book's translation split, if it has one
                folder = self.plan.output_folder or self.plan.workspace_dir
                total = text_book.workspace_section_count(folder) if folder and os.path.isdir(folder) else 0
        except Exception:
            log.debug("counting the other TXT layout failed", exc_info=True)
        with self.lock:
            self._txt_layout_totals[key] = total
        return total

    # ---- live translation helpers -----------------------------------------------------------------

    def translate_target(self, index: int) -> tuple:
        """(source EPUB, chapter file, reason) for "Translate this chapter" (desktop rules)."""
        if self.plan.mode == MODE_WORKSPACE:
            return "", "", "Workspace sections are translated from the Progress manager"
        if self.plan.source_kind == SOURCE_TXT:
            return "", "", TXT_TRANSLATE_REASON
        filename = self.filenames[index] if 0 <= index < len(self.filenames) else ""
        if not filename:
            return "", "", "Could not resolve this chapter's source filename."
        epub = self.plan.epub_path
        if self.plan.mode == MODE_DUAL:
            epub = self.plan.raw_path
        elif self.plan.mode == MODE_OVERLAY:
            epub = self.plan.raw_path or self.plan.epub_path
        if not _is_file(epub, ".epub"):
            return "", filename, "Could not resolve the source EPUB for this chapter."
        return epub, filename, ""

    def translate_visible(self, index: int) -> bool:
        """``_update_translate_btn_visibility``: hidden for completed overlay chapters and the compiled view."""
        if self.plan.mode == MODE_WORKSPACE or self.plan.source_kind == SOURCE_TXT:
            return False
        filename = self.filenames[index] if 0 <= index < len(self.filenames) else ""
        if self.plan.mode == MODE_OVERLAY and self.overlay:
            entry = self.overlay.get(filename)
            if entry and entry.get("path") and os.path.isfile(str(entry["path"])):
                return str(entry.get("status") or "").strip().lower() not in ("", "completed")
            return True
        if self.plan.mode == MODE_DUAL:
            return self.flavor == rm.ORIGINAL
        return True

    def overlay_entry(self, index: int) -> dict:
        filename = self.filenames[index] if 0 <= index < len(self.filenames) else ""
        return dict(self.overlay.get(filename) or {})

    def live_output_folder(self, index: int) -> str:
        """Where a live run of chapter ``index`` writes (``live_stream.resolve_live_output_folder``)."""
        entry = self.overlay_entry(index)
        filename = self.filenames[index] if 0 <= index < len(self.filenames) else ""
        epub, _name, _reason = self.translate_target(index)
        resolver = self.engine.fn("live_stream", "resolve_live_output_folder")
        if resolver is not None:
            try:
                folder = resolver(filename, {filename: entry} if entry else self.overlay, epub, self.engine.config)
            except Exception:
                folder = ""
            if folder:
                return str(folder)
        if self.plan.output_folder and os.path.isdir(self.plan.output_folder):
            return self.plan.output_folder
        return ""

    # ---- search (blocking) --------------------------------------------------------------------------

    def search(self, query: str, *, on_batch: Optional[Callable[[list, bool], Any]] = None,
               cancelled: Callable[[], bool] = lambda: False) -> list:
        """Search the active flavour's chapters (blocking); batches stream through ``on_batch``."""
        chapters = list(self.raw_chapters if self.flavor == rm.ORIGINAL else self._translated_list())
        return self.engine.search(chapters, query, on_batch=on_batch, should_stop=cancelled)

    # ---- images ----------------------------------------------------------------------------------------

    def image_bytes(self, src: str, chapter_filename: str = "") -> Optional[bytes]:
        """Bytes of an image referenced by chapter HTML (EPUB member, image table, or the workspace image dirs)."""
        resolver = self.engine.fn("reader_doc", "reader_image_resource", "_reader_image_resource")
        if resolver is not None:
            try:
                resource = resolver(src, self.images, self.extra_image_dirs, self.active_path)
            except Exception:
                resource = None
            if isinstance(resource, Mapping):
                return self._resource_bytes(resource)
        name = _base(src.split("#", 1)[0].split("?", 1)[0])
        for folder in self.extra_image_dirs:
            candidate = os.path.join(folder, name)
            if os.path.isfile(candidate):
                try:
                    with open(candidate, "rb") as handle:
                        return handle.read()
                except OSError:
                    return None
        value = self.images.get(src) or self.images.get(name)
        if isinstance(value, (bytes, bytearray)):
            return bytes(value)
        return self._zip_member(name)

    def _resource_bytes(self, resource: Mapping[str, Any]) -> Optional[bytes]:
        kind = resource.get("kind")
        if kind == "bytes":
            return bytes(resource.get("data") or b"") or None
        if kind == "file":
            try:
                with open(str(resource.get("path") or ""), "rb") as handle:
                    return handle.read()
            except OSError:
                return None
        if kind == "epub":
            return self._zip_member(str(resource.get("member") or ""), exact=True)
        return None

    def _zip_member(self, name: str, *, exact: bool = False) -> Optional[bytes]:
        path = self.active_path
        if not _is_file(path, ".epub") or not name:
            return None
        with self.lock:
            try:
                if self._zip is None or self._zip_path != _norm(path):
                    self._close_zip()
                    self._zip = zipfile.ZipFile(path, "r")
                    self._zip_path = _norm(path)
                    self._zip_names = {n.casefold(): n for n in self._zip.namelist()}
                if exact:
                    reader = self.engine.fn("reader_doc", "_read_epub_member_from_zip")
                    if reader is not None:
                        return reader(self._zip, name, self._zip_names) or None
                    member = self._zip_names.get(name.casefold())
                    return self._zip.read(member) if member else None
                wanted = _base(name)
                member = next((n for k, n in self._zip_names.items() if k.rsplit("/", 1)[-1] == wanted), None)
                return self._zip.read(member) if member else None
            except (OSError, zipfile.BadZipFile, KeyError):
                self._close_zip()
                return None

    def _close_zip(self) -> None:
        archive, self._zip = self._zip, None
        self._zip_path = ""
        self._zip_names = {}
        if archive is not None:
            try:
                archive.close()
            except Exception:
                pass

    def close(self) -> None:
        with self.lock:
            self._close_zip()


def image_key(*parts: Any) -> str:
    return hashlib.sha1("|".join(str(p) for p in parts).encode("utf-8", "surrogatepass")).hexdigest()[:16]
