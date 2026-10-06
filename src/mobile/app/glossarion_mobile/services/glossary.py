"""GlossaryService: the mobile Glossary Manager's data layer over the shared GUI-free cores (UI_SPEC §4.1).

Nothing here re-implements the desktop Glossary Manager. Every parse, save, tool,
find/replace, output-file update, backup, delete/restore and job decision is a call
into the shared modules the desktop dialogs run (plan §2 U6):

* ``glossary_document`` - the Glossary Editor lifted out of ``GlossaryManager_GUI``:
  ``GlossaryDocument`` is the open file (parse token CSV / legacy CSV / JSON list|dict;
  save with gender resolution and format preservation; cell edits; delete; clean empty
  fields; remove duplicates through the extractor's dedup engine; trim; filter; export
  selection; save as; convert format; Find / Replace incl. the output-file replace with
  its undo step; hide-unused usage scan; backups; undo / redo) - each method runs the
  desktop action's steps in the desktop order, dialogs left to the caller - plus the
  view helpers (``editor_row_specs``, ``editor_display_value``, the column-filter rules,
  ``EditorRow``) and the Manual Glossary Only copy;
* ``glossary_files`` - the ``translator_gui`` glossary closures: ``create_glossary_backup``
  (+ the old-backup cleanup), delete / restore glossary files for the selected inputs,
  the auto-load / auto-mapping helpers;
* ``parallel_epub_core`` - the Parallel EPUB pair mapper (auto map, restore/compact a
  selection, wrapper prompt rendering, writing the paired EPUB);
* existing shared modules: ``unified_glossary`` (listing + ``rebuild_now``),
  ``glossary_paths``, ``gender_tracking`` (row gender status, the resolution
  decision), ``prompt_profiles`` (the Default-plus-named profile machinery, through
  ``glossary_document``'s ``GlossaryPromptProfiles`` / ``RefinementPromptProfiles``),
  ``glossary_refinement`` / ``glossary_progress_core`` (manual refinement),
  ``job_runner.JOB_LOCK`` (editor tools that touch process-global state wait for
  the running job, like the desktop's "a run is in progress" checks).

The ``glossary_files`` / ``parallel_epub_core`` / refinement-runner functions are looked up
through :data:`CONTRACT` (operation -> the shared function's name; the injectable
``SharedCore`` lets host tests pass fakes) and called with
:func:`services.library.bind_call`, which passes only the keyword parameters a function
declares. A missing operation (a build without the module, or the manual-refinement runner
that is not shared yet) raises :class:`CoreMissing`; the UI then shows that action disabled
with the reason instead of guessing (UI_SPEC §0 item 6). Only pure *view*
concerns are local: which rows are visible (search, column filters, sort, hide-unused
rows), the window, selection, ＋ Entry (mobile-only) and the gid route ids.

Threading: methods that read or write files run on the io pool (the UI calls them
through ``io``); none touches Flet. Python 3.10 compatible.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import hashlib
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Iterable, Mapping, Optional, Sequence

from glossarion_mobile.services.library import CoreMissing, SharedCore, bind_call

__all__ = [
    "CONTRACT",
    "CORE_MODULES",
    "ContractMismatch",
    "GLOSSARY_EXTENSIONS",
    "GLOSSARY_MODES",
    "GlossaryFile",
    "GlossaryService",
    "JobBusy",
    "LIST_FORMATS",
    "OpReport",
    "PROFILE_BUCKETS",
    "ProfileOwner",
    "RowSpec",
    "ViewState",
    "display_path",
    "doc_count",
    "editor_display_value",
    "filter_values",
    "is_list_doc",
    "is_updated",
    "raw_field",
    "translated_field",
    "visible_rows",
]

log = logging.getLogger("glossarion.glossary")

#: Shared modules the Glossary Manager reads (all GUI-free, flat in ``src/``).
CORE_MODULES = (
    "glossary_document",
    "glossary_files",
    "parallel_epub_core",
    "unified_glossary",
    "glossary_paths",
    "gender_tracking",
    "prompt_profiles",
    "glossary_refinement",
    "glossary_progress_core",
    "glossary_usage",
    "settings_rules",
)

#: Files the editor / Import / Load accept (desktop editor: csv + json; Load / additional: + txt / md).
GLOSSARY_EXTENSIONS = (".csv", ".json", ".txt", ".md")

#: The 8 glossary modes (``auto_glossary_mode`` values) with the desktop combo labels
#: (GlossaryManager_GUI ``_setup_glossary_general_tab`` / translator_gui shortcut combo).
GLOSSARY_MODES = (
    ("off", "Off"),
    ("off_fuzzy_automap", "Off (Fuzzy Mapping)"),
    ("off_no_automap", "Manual Glossary Only"),
    ("no_glossary", "No Glossary"),
    ("minimal", "Minimal"),
    ("balanced", "Balanced"),
    ("full", "Full"),
    ("single_pass", "Single Pass"),
)

# The desktop editor's standard columns (cell edits keep them as '' instead of removing them).
_STANDARD_FIELDS = ("type", "raw_name", "translated_name", "gender", "description")

#: operation -> (module, (function name,)): the shared function each action calls.
CONTRACT: dict = {
    # --- glossary_files (translator_gui glossary closures, U6) -----------------------------------
    "create_backup": ("glossary_files", ("create_glossary_backup",)),
    "delete_plan": ("glossary_files", ("collect_glossary_files_for_inputs",)),
    "delete_display": ("glossary_files", ("glossary_delete_display",)),
    "delete_files": ("glossary_files", ("delete_glossary_files",)),
    "latest_backup": ("glossary_files", ("find_latest_glossary_backup",)),
    "restore_latest": ("glossary_files", ("restore_glossary_backup",)),
    "guess_glossary": ("glossary_files", ("guess_glossary_for_input_file",)),
    "copy_to_outputs": ("glossary_files", ("copy_glossary_to_output_folders",)),
    # --- parallel_epub_core (parallel_epub_glossary + the TranslatorGUI pair helpers, U6) ---------
    "pair_auto_map": ("parallel_epub_core", ("auto_map_epub_chapters",)),
    "pair_restore": ("parallel_epub_core", ("restore_parallel_epub_pairs",)),
    "pair_compact": ("parallel_epub_core", ("compact_parallel_epub_selection",)),
    "pair_write": ("parallel_epub_core", ("write_parallel_epub",)),
    "pair_filename": ("parallel_epub_core", ("parallel_epub_working_filename",)),
    "pair_render": ("parallel_epub_core", ("apply_parallel_epub_wrapper",)),
    "pair_system_prompt": ("parallel_epub_core", ("default_parallel_epub_system_prompt",)),
    "pair_offset": ("parallel_epub_core", ("offset_parallel_epub_mapping",)),
    "pair_validate": ("parallel_epub_core", ("validate_parallel_epub_pair",)),
    "pair_load": ("parallel_epub_core", ("load_parallel_epub_documents",)),
    "pair_chapters": ("parallel_epub_core", ("load_parallel_epub_chapters",)),
    "pair_profiles": ("parallel_epub_core", ("parallel_epub_profiles",)),
    "pair_active_profile": ("parallel_epub_core", ("active_parallel_epub_profile",)),
    "pair_prompt_settings": ("parallel_epub_core", ("parallel_epub_prompt_settings",)),
    "pair_prepare_saved": ("parallel_epub_core", ("prepare_persisted_parallel_epub_selection",)),
    "pair_selection_matches": ("parallel_epub_core", ("parallel_epub_selection_matches",)),
    "pair_sidecar_read": ("parallel_epub_core", ("read_parallel_epub_mapping_sidecar",)),
    "pair_sidecar_write": ("parallel_epub_core", ("write_parallel_epub_mapping_sidecar",)),
    # --- glossary_progress_core (manual refinement, Retranslation_GUI; not shared yet) ------------
    "refine_run": ("glossary_progress_core", ("run_manual_glossary_refinement",)),
    "refine_plan": ("glossary_progress_core", ("plan_manual_glossary_refinement",)),
}


class ContractMismatch(CoreMissing):
    """A shared function exists but its signature does not match the call (logged with the details)."""

    def __init__(self, name: str, detail: str = "") -> None:
        super().__init__(name)
        self.detail = detail


class JobBusy(RuntimeError):
    """An editor tool that touches process-global state must wait for the running job."""


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _norm(path: Any) -> str:
    try:
        return os.path.normcase(os.path.normpath(os.path.abspath(os.fspath(path))))
    except Exception:
        return str(path or "")


def display_path(path: str, roots: int = 3) -> str:
    """The desktop editor selector text: the last ``roots`` path parts (``_display_glossary_path``)."""
    if not path:
        return ""
    parts: list = []
    current = os.path.normpath(path)
    while current:
        parent, name = os.path.split(current)
        if name:
            parts.append(name)
        if not parent or parent == current or len(parts) >= roots:
            break
        current = parent
    return "/".join(reversed(parts)) if parts else os.path.basename(path)


def editor_display_value(entry: Any, field_name: str) -> str:
    """How a value shows in the editor (``_editor_display_value``: lists joined, dicts as k: v)."""
    value = entry.get(field_name, "") if isinstance(entry, Mapping) else ""
    if isinstance(value, list):
        value = ", ".join(str(v) for v in value)
    elif isinstance(value, dict):
        value = ", ".join(f"{k}: {v}" for k, v in value.items())
    elif value is None:
        value = ""
    return str(value)


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GlossaryFile:
    """One row of the Glossaries home (UI_SPEC §4.1)."""

    path: str
    kind: str  # "book" (Glossary/<book>/), "output" (<output>/<book>/glossary.*), "manual", "unified"
    name: str  # display text (last 3 path parts, like the desktop selector)
    book: str = ""  # book base name (folder) for book/output glossaries
    folder: str = ""
    mtime: float = 0.0
    size: int = 0
    entries: Optional[int] = None  # filled lazily (``count_entries``)
    language_key: str = ""  # unified glossaries: "korean-english"
    gid: str = ""

    @property
    def key(self) -> str:
        return _norm(self.path)

    @property
    def extension(self) -> str:
        return os.path.splitext(self.path)[1].lower()


@dataclass(frozen=True)
class RowSpec:
    """One editor row: ``source_idx`` (stable order), ``ref`` (list index or dict key), ``entry`` (a dict)."""

    source_idx: int
    ref: Any
    entry: Mapping[str, Any]

    @property
    def key(self) -> str:
        return f"g{self.source_idx}"


@dataclass
class OpReport:
    """What an editor tool did (the desktop message box text in ``message``)."""

    ok: bool = True
    message: str = ""
    title: str = ""
    changed: bool = False
    count: int = 0
    details: Any = None
    backup: Optional[str] = None


@dataclass
class ViewState:
    """Pure view state of the editor list (UI only; the document is untouched)."""

    query: str = ""
    filters: dict = field(default_factory=dict)  # field -> frozenset(allowed display values)
    sort_field: Optional[str] = None
    sort_desc: bool = False
    used_rows: Optional[frozenset] = None  # hide unused: the source indices that are used

    @property
    def active(self) -> bool:
        return bool(self.query or self.filters or self.used_rows is not None)


LIST_FORMATS = ("list", "token_csv")


def is_list_doc(doc: Any) -> bool:
    """A list / token CSV glossary (``current_glossary_format``)."""
    return getattr(doc, "current_glossary_format", None) in LIST_FORMATS


def dict_entries(doc: Any) -> dict:
    data = getattr(doc, "current_glossary_data", None) or {}
    entries = data.get("entries", data) if isinstance(data, dict) else {}
    return entries if isinstance(entries, dict) else {}


def doc_count(doc: Any) -> int:
    if is_list_doc(doc):
        return len(getattr(doc, "current_glossary_data", None) or [])
    return len(dict_entries(doc))


def translated_field(doc: Any) -> str:
    return "translated_name" if is_list_doc(doc) else "translated"


def raw_field(doc: Any) -> str:
    return "raw_name" if is_list_doc(doc) else "original"


def doc_fields(doc: Any) -> list:
    return list(getattr(doc, "glossary_column_fields", None) or [])


def is_updated(doc: Any, spec: "RowSpec") -> bool:
    """The row's translated name differs from the last saved one (desktop orange rows)."""
    name = translated_field(doc)
    checker = getattr(doc, "is_changed", None)
    if callable(checker):
        try:
            return bool(checker(spec.ref, name, spec.entry.get(name, "")))
        except Exception:
            return False
    return False


def visible_rows(doc: Any, state: ViewState, specs: Sequence[RowSpec], *,
                 display: Callable[[Any, str], str] = editor_display_value,
                 matches: Optional[Callable[..., bool]] = None) -> list:
    """Rows the editor list shows: hide unused · column filters · search · sort (view only).

    Column filters are the desktop header filters (``row_matches_column_filters``: the displayed
    value of every filtered column is one of its allowed values); search is the desktop Find test
    (case-insensitive, any column); sort is mobile-only (stable, by displayed value).
    """
    rows = list(specs)
    fields = doc_fields(doc)
    if state.used_rows is not None:
        rows = [r for r in rows if r.source_idx in state.used_rows]
    if state.filters:
        filters = {f: set(v) for f, v in state.filters.items()}

        def text_of(row: RowSpec) -> Callable[[int], str]:
            return lambda column: display(row.entry, fields[column - 1]) if 0 < column <= len(fields) else ""

        if matches is not None:
            rows = [r for r in rows if matches(text_of(r), fields, filters)]
        else:
            rows = [r for r in rows if all(display(r.entry, f) in allowed for f, allowed in filters.items())]
    query = (state.query or "").strip().casefold()
    if query:
        search_fields = fields or sorted({k for r in rows for k in r.entry})
        rows = [r for r in rows if any(query in display(r.entry, f).casefold() for f in search_fields)]
    if state.sort_field:
        key_field = state.sort_field
        rows.sort(key=lambda r: display(r.entry, key_field).casefold(), reverse=state.sort_desc)
    return rows


def filter_values(doc: Any, field_name: str, specs: Sequence[RowSpec], *,
                  display: Callable[[Any, str], str] = editor_display_value,
                  order: Optional[Callable[[Any], list]] = None) -> list:
    """(value, count) of one column in the desktop filter order (blanks first, then case-insensitive)."""
    counts: dict = {}
    for spec in specs:
        value = display(spec.entry, field_name)
        counts[value] = counts.get(value, 0) + 1
    values = order(list(counts)) if order is not None else sorted(counts, key=lambda v: (v != "", v.casefold()))
    return [(value, counts[value]) for value in values]


# ---------------------------------------------------------------------------
# Glossary Manager prompt profiles (glossary_document's shared profile rows)
# ---------------------------------------------------------------------------

#: The profile rows of the Glossary settings tabs: bucket -> the desktop row label.
PROFILE_BUCKETS = {
    "balanced_full": "Balanced/Full Profile",
    "minimal": "Minimal Profile",
    "refinement": "Refinement Profile",
}

#: config.json dicts the Balanced/Full and Minimal rows share (each row writes only its own entry).
_GLOSSARY_PROFILE_DICT_KEYS = ("glossary_prompt_profiles", "active_glossary_prompt_profiles",
                               "glossary_prompt_profile_defaults")


class ProfileOwner:
    """The TranslatorGUI stand-in the shared profile rows write to: a copy of config.json
    (``config``) plus the prompt attributes they set (``manual_glossary_prompt`` ...)."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = copy.deepcopy(dict(config))


# ---------------------------------------------------------------------------
# The service
# ---------------------------------------------------------------------------


class GlossaryService:
    def __init__(
        self,
        *,
        paths: Any = None,  # runtime_bootstrap paths (data, output, library)
        config: Any = None,  # MobileConfigStore (get / set / snapshot) or a dict (tests)
        prefs: Any = None,  # Prefs (file_ref / resolve_file_ref / get / set)
        files: Any = None,  # FileBridge
        jobs: Any = None,  # JobsFeature / JobService (has_kind, submit)
        library: Any = None,  # LibraryService
        core: Optional[SharedCore] = None,
        run_io: Optional[Callable[..., Awaitable[Any]]] = None,
        log_fn: Optional[Callable[[str], Any]] = None,
    ) -> None:
        self.paths = paths
        self.config = config
        self.prefs = prefs
        self.files = files
        self.jobs = jobs
        self.library = library
        self.core = core or SharedCore()
        self._run_io = run_io
        self.log_lines: list = []
        self._log_fn = log_fn
        self._local_refs: dict = {}
        self._counts: dict = {}  # path key -> (mtime, count)
        #: ``(title, text) -> bool``: a blocking Yes/No for "Continue anyway?" (set by the UI feature).
        self.ask_continue: Optional[Callable[[str, str], bool]] = None

    # ---- plumbing ------------------------------------------------------------------------------

    async def io(self, fn: Callable[..., Any], *args: Any) -> Any:
        if self._run_io is not None:
            return await self._run_io(fn, *args)
        return await asyncio.to_thread(fn, *args)

    def log(self, text: Any) -> None:
        """The desktop ``append_log`` target of the shared editor functions (kept for the log sheet)."""
        line = str(text)
        self.log_lines.append(line)
        del self.log_lines[:-500]
        log.info("glossary: %s", line)
        if self._log_fn is not None:
            try:
                self._log_fn(line)
            except Exception:
                pass

    def fn(self, op: str) -> Optional[Callable[..., Any]]:
        module, names = CONTRACT[op]
        return self.core.fn(module, *names)

    def available(self, op: str) -> bool:
        return self.fn(op) is not None

    def call(self, op: str, **available: Any) -> Any:
        """Call the shared function of ``op`` with the keyword parameters it declares."""
        module, names = CONTRACT[op]
        target = self.core.fn(module, *names)
        if target is None:
            raise CoreMissing(f"{module}.{names[0]}")
        try:
            return bind_call(target, **available)
        except TypeError as exc:
            if "needs" in str(exc):
                log.warning("glossary contract %s: %s", op, exc)
                raise ContractMismatch(f"{module}.{getattr(target, '__name__', names[0])}", str(exc)) from exc
            raise

    def reason(self, op: str) -> Optional[str]:
        """The disabled reason of an action that needs ``op`` (None when available)."""
        if self.available(op):
            return None
        module, names = CONTRACT[op]
        return f"Needs {module}.{names[0]} (not in this build)"

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

    def set_many(self, updates: Mapping[str, Any]) -> None:
        config = self.config
        if config is None:
            return
        setter = getattr(config, "set_many", None)
        if callable(setter) and not isinstance(config, dict):
            try:
                setter(dict(updates))
                return
            except Exception:
                log.exception("saving %s failed", list(updates))
        for key, value in updates.items():
            self.set_cfg(key, value)

    def custom_entry_types(self) -> dict:
        types = self.cfg("custom_entry_types", None)
        return dict(types) if isinstance(types, Mapping) and types else {
            "character": {"enabled": True, "has_gender": True},
            "terms": {"enabled": True, "has_gender": False},
            "surnames": {"enabled": True, "has_gender": False},
            "titles": {"enabled": True, "has_gender": True},
            "locations": {"enabled": True, "has_gender": False},
            "nicknames": {"enabled": True, "has_gender": True},
        }

    # ---- route ids -----------------------------------------------------------------------------

    def gid_for(self, path: str) -> str:
        """Opaque 12-hex route id of a glossary file (``Prefs.file_ref``; UI_SPEC §1.4)."""
        prefs = self.prefs
        gid = None
        if prefs is not None and hasattr(prefs, "file_ref"):
            try:
                gid = prefs.file_ref(path, kind="glossary")
            except Exception:
                gid = None
        if not gid:
            gid = hashlib.sha1(_norm(path).encode("utf-8", "surrogatepass")).hexdigest()[:12]
        self._local_refs[gid] = os.path.abspath(path)
        return gid

    def path_for_gid(self, gid: str) -> Optional[str]:
        path = self._local_refs.get(str(gid))
        if path:
            return path
        prefs = self.prefs
        if prefs is not None and hasattr(prefs, "resolve_file_ref"):
            try:
                path = prefs.resolve_file_ref(str(gid))
            except Exception:
                path = None
        return path or None

    # ---- locations -----------------------------------------------------------------------------

    def output_roots(self) -> list:
        roots: list = []
        output = getattr(self.paths, "output", None) if self.paths is not None else None
        for candidate in (output, os.environ.get("OUTPUT_DIRECTORY"), self.cfg("output_directory", None)):
            if candidate and _norm(candidate) not in {_norm(r) for r in roots}:
                roots.append(os.fspath(candidate))
        return roots

    def app_dir(self) -> str:
        fn = self.core.fn("app_paths", "_get_app_dir")
        if fn is not None:
            try:
                return os.fspath(fn())
            except Exception:
                pass
        data = getattr(self.paths, "data", None) if self.paths is not None else None
        return os.fspath(data) if data else os.getcwd()

    def shared_glossary_dirs(self) -> list:
        """The shared ``Glossary/`` folders a run writes to (``<output>/Glossary`` first, then the app dir)."""
        roots: list = []
        for base in self.output_roots() + [self.app_dir()]:
            folder = os.path.join(os.path.abspath(base), "Glossary")
            if _norm(folder) not in {_norm(r) for r in roots}:
                roots.append(folder)
        env_shared = str(os.environ.get("GLOSSARY_SHARED_DIR", "") or "").strip()
        if env_shared and _norm(env_shared) not in {_norm(r) for r in roots}:
            roots.append(os.path.abspath(env_shared))
        return roots

    def primary_shared_dir(self) -> str:
        """``_unified_glossary_shared_dir``: OUTPUT_DIRECTORY/Glossary, else <app dir>/Glossary."""
        roots = self.output_roots()
        if roots:
            return os.path.join(os.path.abspath(roots[0]), "Glossary")
        return os.path.join(self.app_dir(), "Glossary")

    # ---- listing (blocking) ----------------------------------------------------------------------

    def _file_row(self, path: str, kind: str, *, book: str = "", language_key: str = "") -> GlossaryFile:
        try:
            stat = os.stat(path)
            mtime, size = stat.st_mtime, stat.st_size
        except OSError:
            mtime, size = 0.0, 0
        cached = self._counts.get(_norm(path))
        entries = cached[1] if cached is not None and cached[0] == mtime else None
        return GlossaryFile(path=os.path.abspath(path), kind=kind, name=display_path(path), book=book,
                            folder=os.path.dirname(os.path.abspath(path)), mtime=mtime, size=size, entries=entries,
                            language_key=language_key, gid=self.gid_for(path))

    def list_glossaries(self) -> list:
        """Every glossary the Glossaries home lists: book glossaries, Minimal-mode output glossaries,
        manual / imported glossaries and the unified glossaries (newest first within a kind)."""
        ug = self.core.module("unified_glossary")
        found: dict = {}

        def add(path: str, kind: str, **extra: Any) -> None:
            if not path or not os.path.isfile(path):
                return
            key = _norm(path)
            if key not in found:
                found[key] = self._file_row(path, kind, **extra)

        is_book_name = getattr(ug, "is_book_glossary_filename", None) if ug is not None else None

        def book_name_ok(name: str) -> bool:
            if callable(is_book_name):
                try:
                    return bool(is_book_name(name))
                except Exception:
                    return False
            lower = name.lower()
            return lower.endswith(("_glossary.csv", "_glossary.json")) or lower in ("glossary.csv", "glossary.json")

        for shared in self.shared_glossary_dirs():
            if not os.path.isdir(shared):
                continue
            iterate = getattr(ug, "iter_book_glossary_files", None) if ug is not None else None
            books = []
            if callable(iterate):
                try:
                    books = list(iterate(shared) or [])
                except Exception:
                    log.exception("listing book glossaries in %s failed", shared)
            for path in books:
                add(path, "book", book=os.path.basename(os.path.dirname(path)))
            try:
                names = sorted(os.listdir(shared))
            except OSError:
                names = []
            for name in names:  # legacy flat <shared>/<book>_glossary.csv
                path = os.path.join(shared, name)
                if os.path.isfile(path) and book_name_ok(name) and name.lower() not in ("glossary.csv", "glossary.json"):
                    stem = os.path.splitext(name)[0]
                    add(path, "book", book=stem[:-len("_glossary")] if stem.lower().endswith("_glossary") else stem)
            for path, key in self._unified_files(shared):
                add(path, "unified", language_key=key)
        for root in self.output_roots():
            if not os.path.isdir(root):
                continue
            try:
                children = sorted(os.listdir(root))
            except OSError:
                children = []
            for child in children:
                book_dir = os.path.join(root, child)
                if child.lower() == "glossary" or not os.path.isdir(book_dir):
                    continue
                for folder in (book_dir, os.path.join(book_dir, "Glossary")):
                    for ext in (".csv", ".json"):
                        add(os.path.join(folder, f"glossary{ext}"), "output", book=child)
        for path in self.manual_glossary_paths():
            add(path, "manual")
        order = {"book": 0, "output": 1, "manual": 2, "unified": 3}
        return sorted(found.values(), key=lambda g: (order.get(g.kind, 9), -g.mtime, g.name.casefold()))

    def _unified_files(self, shared: str) -> list:
        """(csv path, language key) of every unified glossary under one shared folder
        (``_unified_glossary_editor_files``)."""
        ug = self.core.module("unified_glossary")
        if ug is None or not hasattr(ug, "unified_root") or not hasattr(ug, "unified_paths"):
            return []
        try:
            root = ug.unified_root(shared)
            keys = sorted(os.listdir(root)) if os.path.isdir(root) else []
        except Exception:
            return []
        out = []
        for key in keys:
            try:
                csv_path = ug.unified_paths(shared, key)[2]
            except Exception:
                continue
            if os.path.isfile(csv_path):
                out.append((csv_path, key))
        return out

    def manual_glossary_paths(self) -> list:
        paths: list = []
        for key in ("manual_glossary_path", "additional_glossary_path"):
            value = self.cfg(key, "")
            if isinstance(value, str) and value:
                paths.append(value)
        mapping = self.cfg("manual_glossary_map", {})
        if isinstance(mapping, Mapping):
            paths.extend(str(v) for v in mapping.values() if v)
        prefs = self.prefs
        if prefs is not None and hasattr(prefs, "get"):
            try:
                paths.extend(str(p) for p in (prefs.get("glossary_imports", []) or []) if p)
            except Exception:
                pass
        return [p for p in paths if os.path.splitext(p)[1].lower() in GLOSSARY_EXTENSIONS]

    def record_import(self, path: str) -> None:
        prefs = self.prefs
        if prefs is None or not hasattr(prefs, "set"):
            return
        try:
            current = [p for p in (prefs.get("glossary_imports", []) or []) if _norm(p) != _norm(path)]
            prefs.set("glossary_imports", (current + [os.path.abspath(path)])[-200:])
        except Exception:
            log.exception("recording the imported glossary failed")

    def count_entries(self, path: str) -> Optional[int]:
        """Entry count of a listed file (``unified_glossary.count_entries``), cached by mtime."""
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            return None
        cached = self._counts.get(_norm(path))
        if cached is not None and cached[0] == mtime:
            return cached[1]
        counter = self.core.fn("unified_glossary", "count_entries")
        if counter is None:
            return None
        try:
            count = int(counter(path))
        except Exception:
            return None
        self._counts[_norm(path)] = (mtime, count)
        return count

    def book_for_glossary(self, glossary: GlossaryFile) -> Optional[dict]:
        """The Library book a book/output glossary belongs to (by folder name), when the Library knows it."""
        library = self.library
        if library is None or not glossary.book:
            return None
        snapshot = getattr(library, "snapshot", None)
        books = snapshot.all_books() if snapshot is not None else ()
        target = glossary.book.casefold()
        for book in books:
            folder = str(book.get("output_folder") or "")
            name = os.path.basename(os.path.normpath(folder)) if folder else str(book.get("name") or "")
            if name.casefold() == target or str(book.get("name") or "").casefold() == target:
                return dict(book)
        return None

    def glossaries_for_book(self, book: Mapping[str, Any]) -> list:
        """The glossary files of one Library book: the desktop editor's list for that input
        (``glossary_document.resolve_editor_glossaries``), else the listed book / output glossaries."""
        rows: list = []
        resolve = self.core.fn("glossary_document", "resolve_editor_glossaries")
        source = ""
        if self.library is not None:
            try:
                source = self.library.raw_source(book) or ""
            except Exception:
                source = ""
        if resolve is not None and source:
            cfg = self.config_snapshot()
            try:
                found, _sources = resolve([source], config=cfg, override_dir=(self.output_roots() or [None])[0],
                                          manual_glossary_path=None, base_dir=self.app_dir())
                rows = [self._file_row(path, "book", book=os.path.splitext(os.path.basename(source))[0])
                        for _display, path in found]
            except Exception:
                log.exception("resolving the editor glossaries failed")
        if rows:
            return rows
        folder = str(book.get("output_folder") or "")
        base = os.path.basename(os.path.normpath(folder)) if folder else str(book.get("name") or "")
        return [g for g in self.list_glossaries() if g.kind in ("book", "output") and g.book.casefold() ==
                base.casefold()]

    # ---- documents (blocking) --------------------------------------------------------------------

    def gd(self) -> Any:
        """``glossary_document`` (CoreMissing without it)."""
        module = self.core.module("glossary_document")
        if module is None or not hasattr(module, "GlossaryDocument"):
            raise CoreMissing("glossary_document.GlossaryDocument")
        return module

    def open_document(self, path: str, *, source_path: Optional[str] = None) -> Any:
        """``GlossaryDocument.open``: the desktop editor's background load (parse, gender tracker,
        column fields, the translated-name baseline, the stats line). Backups go through
        :meth:`backup_callback`; log lines to :meth:`log`."""
        gd = self.gd()
        doc = gd.GlossaryDocument(self.config_snapshot(), backup=self.backup_callback, log=self.log)
        doc.load(path)
        self._decorate(doc, source_path)
        return doc

    def reload_document(self, doc: Any) -> Any:
        """Reload / force refresh / the auto-reload (``load_glossary_for_editing``)."""
        doc.load()
        self._decorate(doc, getattr(doc, "source_path", None))
        return doc

    @staticmethod
    def _decorate(doc: Any, source_path: Optional[str]) -> None:
        """Mobile view state kept on the document: its input, the unsaved flag, the file mtime."""
        doc.source_path = source_path
        doc.dirty = False
        try:
            doc.mtime = os.path.getmtime(doc.path)
        except OSError:
            doc.mtime = 0.0

    def mark_saved(self, doc: Any) -> None:
        doc.dirty = False
        try:
            doc.mtime = os.path.getmtime(doc.path)
        except OSError:
            pass

    def row_specs(self, doc: Any) -> list:
        """``editor_row_specs``: (source index, ref, entry copy) of every row."""
        gd = self.gd()
        _fields, specs = gd.editor_row_specs(doc_fields(doc), doc.current_glossary_data, doc.current_glossary_format)
        return [RowSpec(int(idx), ref, entry) for idx, ref, entry in specs]

    def display(self) -> Callable[[Any, str], str]:
        module = self.core.module("glossary_document")
        shared = getattr(module, "editor_display_value", None) if module is not None else None
        return shared if callable(shared) else editor_display_value

    def filter_matcher(self) -> Optional[Callable[..., bool]]:
        return self.core.fn("glossary_document", "row_matches_column_filters")

    def filter_order(self) -> Optional[Callable[..., list]]:
        return self.core.fn("glossary_document", "column_filter_values")

    def visible(self, doc: Any, state: ViewState, specs: Sequence[RowSpec]) -> list:
        return visible_rows(doc, state, specs, display=self.display(), matches=self.filter_matcher())

    def column_values(self, doc: Any, field_name: str, specs: Sequence[RowSpec]) -> list:
        return filter_values(doc, field_name, specs, display=self.display(), order=self.filter_order())

    def stats_text(self, doc: Any) -> str:
        return str(getattr(doc, "stats_text", "") or f"Total entries: {doc_count(doc)}")

    def type_summary(self, doc: Any) -> str:
        if not is_list_doc(doc):
            return ""
        fn = self.core.fn("glossary_document", "glossary_type_count_summary")
        if fn is None:
            return ""
        try:
            configured = doc.entry_type_config() if hasattr(doc, "entry_type_config") else None
            return str(fn(list(doc.current_glossary_data or []), configured) or "")
        except Exception:
            return ""

    # ---- gender ----------------------------------------------------------------------------------------

    def gender_status(self, doc: Any, entry: Mapping[str, Any]) -> Optional[dict]:
        """The row's tracker status (``label``, ``conflict``) - ``GlossaryDocument.gender_status``."""
        try:
            status = doc.gender_status(entry)
        except Exception:
            return None
        return status if isinstance(status, dict) else None

    def gender_model(self, doc: Any, ref: Any) -> Optional[dict]:
        """The Resolve Tracked Gender sheet (``_open_gender_resolution``) as texts + the current decision."""
        gd = self.gd()
        entry = doc.entry(ref)
        if not entry or not gd.can_resolve_gender(self.gender_status(doc, entry)):
            return None
        tracker_entry = gd.tracker_entry_for_entry(entry, doc.current_gender_tracker_data)
        if not tracker_entry:
            return None
        threshold, bias = doc.gender_settings()
        summary = gd.gender_resolution_summary(tracker_entry, entry.get("gender", ""), threshold, bias)
        raw_name = str(entry.get("raw_name", "") or "").strip()
        translated = str(entry.get("translated_name", "") or "").strip()
        threshold_pct, bias_label = gd.gender_settings_labels(threshold, bias)
        total = summary["total"]
        genders = getattr(gd, "BINARY_GENDERS", ("male", "female"))
        return {
            "heading": (translated or raw_name) + (f"  ({raw_name})" if translated and raw_name else ""),
            "overview": (f"Calculated Auto result: {summary['calculated_auto_gender'].title()} · Editor display: "
                         f"{summary['status'].get('label', '')}\nIgnore rare flips: {threshold_pct:.0f}% · "
                         f"Bias: {bias_label}"),
            "history": [gd.gender_history_line(gender, summary, total) for gender in genders],
            "flips": f"Gender flips: {summary['flip_count']}",
            "latest": [gd.gender_flip_line(change) for change in reversed(summary["latest_flips"])],
            "decision": gd.current_gender_decision(tracker_entry),
            "summary": summary,
        }

    def resolve_gender(self, doc: Any, ref: Any, decision: str) -> bool:
        """Resolve Gender › Apply (undoable; written to the tracker on Save)."""
        ok = bool(doc.resolve_gender(ref, decision))
        if ok:
            doc.dirty = True
        return ok

    # ---- edits (in memory; Save writes) ----------------------------------------------------------------

    def edit_field(self, doc: Any, ref: Any, field_name: str, value: Any) -> Any:
        """One cell edit (``GlossaryDocument.edit_cell``: undo snapshot + ``save_edit``); returns the new ref."""
        _stored, new_ref = doc.edit_cell(ref, field_name, "" if value is None else str(value))
        doc.dirty = True
        return new_ref

    def update_entry(self, doc: Any, ref: Any, values: Mapping[str, Any]) -> Any:
        """EntrySheet › Save: the changed columns as cell edits under ONE undo snapshot."""
        gd = self.gd()
        entry = doc.entry(ref) if is_list_doc(doc) else None
        display = self.display()
        changed = [(k, "" if v is None else str(v)) for k, v in values.items()
                   if entry is None or display(entry, k) != ("" if v is None else str(v))]
        if not changed:
            return ref
        doc.push_undo()
        for key, value in changed:
            _row, ref, _ref_changed = gd.apply_entry_edit(doc, ref, key, gd.normalize_edit_value(key, value))
        doc.dirty = True
        return ref

    def change_type(self, doc: Any, refs: Sequence[Any], entry_type: str) -> int:
        """Bulk "Change type" (mobile): the cell edit of every selected row, one undo snapshot."""
        gd = self.gd()
        refs = list(refs)
        if not refs:
            return 0
        doc.push_undo()
        for ref in refs:
            gd.apply_entry_edit(doc, ref, "type", gd.normalize_edit_value("type", entry_type))
        doc.dirty = True
        return len(refs)

    def add_entry(self, doc: Any, values: Mapping[str, Any]) -> Any:
        """＋ Entry (mobile): a new row with the standard columns (undoable); returns its ref."""
        gd = self.gd()
        doc.push_undo()
        if is_list_doc(doc):
            entry: dict = {"type": str(values.get("type") or "character"), "raw_name": "", "translated_name": "",
                           "gender": ""}
            for key, value in values.items():
                text = gd.normalize_edit_value(key, "" if value is None else str(value))
                if text or key in ("type", "raw_name", "translated_name", "gender", "description"):
                    entry[key] = text
            if doc.current_glossary_format == "token_csv":
                same = [e for e in doc.current_glossary_data or [] if isinstance(e, dict) and
                        e.get("type") == entry["type"] and e.get("_section")]
                if same:
                    entry["_section"] = same[0]["_section"]
            doc.current_glossary_data.append(entry)
            doc.dirty = True
            return len(doc.current_glossary_data) - 1
        key = str(values.get("original") or values.get("raw_name") or "").strip()
        if not key:
            raise ValueError("Enter the original term")
        dict_entries(doc)[key] = str(values.get("translated") or values.get("translated_name") or "")
        doc.dirty = True
        return key

    # ---- save (blocking) ---------------------------------------------------------------------------------

    def translated_changes(self, doc: Any) -> list:
        return [tuple(c) for c in (doc.translated_changes() or [])]

    def update_prompt(self, changes: Sequence[tuple]) -> tuple:
        """The "Update output files" question (``update_output_files_prompt``) as plain text."""
        import html as _html

        gd = self.gd()
        title, text, examples = gd.update_output_files_prompt(list(changes))
        plain = _html.unescape(str(examples or "").replace("<br>", "\n"))
        return title, f"{text}\n\n{plain}" if plain else text

    def save_edits(self, doc: Any, *, update_outputs: Optional[bool] = None) -> dict:
        """Save (``save_edited_glossary`` after its question): backup "before_save", the shared save (gender
        resolution, the format writers), the output-file update, the new baseline."""
        report = dict(doc.save_edits(update_output_files=update_outputs) or {})
        if report.get("saved"):
            self.mark_saved(doc)
        return report

    def save_document(self, doc: Any) -> bool:
        """``save_current_glossary`` alone (the swipe-delete Undo writes the restored rows back)."""
        ok = bool(doc.save())
        if ok:
            self.mark_saved(doc)
        return ok

    # ---- tools (blocking): each returns the desktop message box (title, text) or None ---------------------

    @contextlib.contextmanager
    def exclusive(self) -> Iterable[None]:
        """Hold ``job_runner.JOB_LOCK`` while a tool sets process-global env (``remove_duplicates`` sets
        GLOSSARY_DISABLE_HONORIFICS_FILTER): a running job owns ``os.environ``. JobBusy when a job holds it."""
        lock = self.core.value("job_runner", "JOB_LOCK", default=None)
        if lock is None:
            yield
            return
        acquired = lock.acquire(blocking=False)
        if not acquired:
            raise JobBusy("Wait for the running job to finish, then try again.")
        try:
            yield
        finally:
            lock.release()

    def _saved_box(self, doc: Any, box: Any, *, saved: Optional[bool] = None) -> Optional[OpReport]:
        """The tool's message box as an OpReport. The document counts as saved only when the shared call
        wrote and re-read it (a "Success" box; ``saved`` overrides): an "Info" box ("No duplicates found")
        saved nothing, so unsaved edits stay unsaved."""
        if box is None:
            return None
        title, text = box
        if title == "Success" if saved is None else saved:
            self.mark_saved(doc)
        return OpReport(ok=title != "Error", title=str(title), message=str(text), changed=title == "Success")

    def delete(self, doc: Any, refs: Sequence[Any], *, keep_snapshot: bool = False) -> Optional[OpReport]:
        """Delete Selected (after "Confirm Delete"): backup, undo snapshot, delete, save, reload.

        Like the desktop the reload clears the undo history. ``keep_snapshot`` (the mobile swipe
        delete) returns the rows as they were before the delete in ``details['snapshot']`` (the
        shared undo snapshot) for :meth:`restore_snapshot`."""
        snapshot = self.snapshot(doc) if keep_snapshot else None
        report = self._saved_box(doc, doc.delete(list(refs)))
        if report is not None and report.changed and snapshot is not None:
            report.details = {"snapshot": snapshot}
        return report

    def snapshot(self, doc: Any) -> Optional[dict]:
        """The document's undo snapshot as it is now (``push_undo_snapshot``), outside its undo stack."""
        stack: list = []
        if not self.gd().push_undo_snapshot(doc, stack, []):
            return None
        return stack[-1]

    def restore_snapshot(self, doc: Any, snapshot: Mapping[str, Any]) -> bool:
        """Undo of a swipe delete: the snapshot restored into the document (``undo_step``), then saved and
        re-read like the delete itself (``save`` + ``load``). False when nothing was written."""
        kind, _result = self.gd().undo_step(doc, [copy.deepcopy(dict(snapshot))], [])
        if kind != "glossary":
            return False
        try:
            saved = bool(doc.save())
        except Exception:
            doc.dirty = True  # the restored rows are in memory only: Save writes them
            raise
        if not saved:
            doc.dirty = True
            return False
        doc.load()
        self.mark_saved(doc)
        return True

    def prune_filters(self, filters: Mapping[str, Any], fields: Sequence[str]) -> dict:
        """Column filters of columns the re-read glossary still has (``prune_column_filters``)."""
        prune = self.core.fn("glossary_document", "prune_column_filters")
        if prune is None:
            return dict(filters)
        return dict(prune(dict(filters), list(fields)))

    def clean_empty_fields(self, doc: Any) -> Optional[OpReport]:
        return self._saved_box(doc, doc.clean_empty_fields())

    def remove_duplicates(self, doc: Any) -> Optional[OpReport]:
        with self.exclusive():
            return self._saved_box(doc, doc.remove_duplicates())

    def trim_preview(self, doc: Any, top_n: int) -> str:
        return str(doc.trim_preview(int(top_n)))

    def trim(self, doc: Any, top_n: Any) -> Optional[OpReport]:
        return self._saved_box(doc, doc.trim(top_n))

    def filter_types(self, doc: Any) -> list:
        return list(doc.filter_types() or [])

    def preview_filter(self, doc: Any, **choices: Any) -> tuple:
        """Preview Filter → (matching, removed); choices: kept_types {type: bool}, type_limits {type: text},
        search_text, gender_value ('all' / 'Male' / 'Female' / 'Unknown')."""
        return tuple(doc.preview_filter(**choices))

    def apply_filter(self, doc: Any, **choices: Any) -> Optional[OpReport]:
        return self._saved_box(doc, doc.apply_filter(**choices))

    def default_convert_path(self, doc: Any) -> str:
        fn = self.core.fn("glossary_document", "default_convert_path")
        if fn is not None:
            return str(fn(doc.path, doc.config))
        return doc.path.replace(".json", ".csv")

    def convert(self, doc: Any, csv_path: str) -> Optional[OpReport]:
        """Convert Format (backup "before_export"; token-efficient or legacy CSV; reloads on overwrite)."""
        box = doc.convert(csv_path)
        if box is None:
            return None
        # Only a conversion over the open file writes and re-reads it; an export elsewhere leaves edits unsaved.
        reload = _norm(csv_path) == _norm(doc.path)
        report = self._saved_box(doc, box, saved=reload and box[0] == "Success")
        report.details = {"path": csv_path, "reload": reload}
        return report

    def export_selection(self, doc: Any, path: str, refs: Sequence[Any]) -> OpReport:
        title, text = doc.export_selection(path, list(refs))
        return OpReport(True, str(text), str(title), details={"path": path})

    def save_as(self, doc: Any, path: str) -> OpReport:
        """Save As: the document then points at ``path`` (the editor shows the new file)."""
        title, text = doc.save_as(path)
        self.mark_saved(doc)
        return OpReport(True, str(text), str(title), details={"path": path})

    # ---- find / replace ------------------------------------------------------------------------------

    def editor_rows(self, doc: Any, specs: Sequence[RowSpec]) -> list:
        """``EditorRow`` of each shown row (column 0 = row number, then the displayed columns)."""
        gd = self.gd()
        fields = doc_fields(doc)
        return [gd.EditorRow.from_spec(index + 1, spec.ref, spec.entry, fields) for index, spec in enumerate(specs)]

    def find_next(self, doc: Any, text: str, rows: Sequence[Any]) -> Optional[int]:
        """Find Next over the shown rows (``find_next_index``, wrapping after the last hit)."""
        return doc.find_next(text, list(rows))

    def preview_replace(self, doc: Any, find: str, specs: Sequence[RowSpec]) -> tuple:
        """(rows with a match, occurrences) of ``find`` in the shown rows - the sheet's live preview."""
        if not find:
            return 0, 0
        gd = self.gd()
        pattern = re.compile(re.escape(find), re.IGNORECASE)
        rows = occurrences = 0
        for row in self.editor_rows(doc, specs):
            if gd.row_has_match(row, find):
                rows += 1
                occurrences += sum(len(pattern.findall(row.text(c))) for c in range(1, row.columnCount()))
        return rows, occurrences

    def replace_in(self, doc: Any, row: Any, find: str, repl: str) -> int:
        """Replace (one row; undo snapshot only when it matches)."""
        count = int(doc.replace_in(row, find, repl) or 0)
        if count:
            doc.dirty = True
        return count

    def replace_all(self, doc: Any, find: str, repl: str, rows: Optional[Sequence[Any]] = None) -> int:
        """Replace All over the rows the view holds (all, or the used ones while Hide unused is on)."""
        total = int(doc.replace_all(find, repl, list(rows) if rows is not None else None) or 0)
        if total:
            doc.dirty = True
        return total

    def replace_in_outputs(self, doc: Any, old: str, new: str) -> tuple:
        """The "No glossary match" fallback: the output files directly (an undoable output-file step)."""
        return tuple(doc.replace_in_output_files(old, new))

    def undo(self, doc: Any, *, redo: bool = False) -> tuple:
        """Undo / Redo: ``('html', (files, replacements))`` · ``('glossary', save_error)`` (restored, saved
        and re-read like the desktop) · ``(None, None)``."""
        kind, result = doc.redo() if redo else doc.undo()
        if kind == "glossary":
            self.mark_saved(doc)
        return kind, result

    # ---- hide unused (blocking) ----------------------------------------------------------------------

    def used_rows(self, doc: Any, progress: Optional[Callable[[dict], Any]] = None) -> dict:
        """Hide unused entries: the translated output folder (from the input, else the glossary's place) and
        ``GlossaryDocument.used_rows`` (``{'ok', 'used_rows', 'no_files', 'errors', 'total'}``)."""
        output_dir = doc.output_dir(getattr(doc, "source_path", None))
        if not output_dir:
            return {"ok": True, "no_output_dir": True, "total": doc_count(doc)}
        emit = progress if progress is not None else (lambda _payload: None)
        return dict(doc.used_rows(output_dir, emit))

    # ---- backups -----------------------------------------------------------------------------------

    def list_backups(self, doc: Any) -> list:
        """This glossary's backups, newest first: [{'path', 'name', 'mtime', 'size'}]."""
        rows = []
        for path in reversed(list(doc.backups() or [])):
            try:
                stat = os.stat(path)
                mtime, size = stat.st_mtime, stat.st_size
            except OSError:
                mtime, size = 0.0, 0
            rows.append({"path": path, "name": os.path.basename(path), "mtime": mtime, "size": size})
        return rows

    def restore_backup(self, doc: Any, backup_path: str) -> Optional[OpReport]:
        """Restore one editor backup (a backup "before_restore" first; saved in the file's format)."""
        return self._saved_box(doc, doc.restore_backup(backup_path))

    def backup_settings(self) -> tuple:
        return bool(self.cfg("glossary_auto_backup", True)), int(self.cfg("glossary_max_backups", 50) or 0)

    def set_backup_settings(self, enabled: bool, max_backups: int) -> str:
        self.set_many({"glossary_auto_backup": bool(enabled), "glossary_max_backups": int(max_backups)})
        fn = self.core.fn("glossary_document", "backup_settings_message")
        if fn is not None:
            return str(fn(bool(enabled), int(max_backups)))
        return f"Automatic backups {'enabled' if enabled else 'disabled'}"

    def backup_callback(self, doc: Any, operation_name: str = "manual") -> bool:
        """``GlossaryDocument(backup=...)``: the desktop ``create_glossary_backup`` (glossary_files). A failed
        backup asks "Failed to create backup: … Continue anyway?" (``ask_continue``, blocking on the io pool)."""
        if operation_name != "manual" and not self.cfg("glossary_auto_backup", True):
            return True
        fn = self.fn("create_backup")
        if fn is None:
            return self._ask_continue("Backup Failed", "Backups are not available in this build "
                                      "(glossary_files.create_glossary_backup).\n\nContinue anyway?")
        cfg = self.config_snapshot()
        try:
            result = bind_call(fn, doc=doc, document=doc, owner=doc, editor=doc, glossary_path=doc.path,
                               original_path=doc.path, path=doc.path, current_glossary_data=doc.current_glossary_data,
                               data=doc.current_glossary_data, glossary_data=doc.current_glossary_data,
                               operation_name=operation_name, operation=operation_name, config=cfg, settings=cfg,
                               append_log=self.log, log=self.log, ask_continue=self._ask_continue,
                               confirm_continue=self._ask_continue)
        except TypeError as exc:
            if "needs" in str(exc):
                log.warning("glossary_files.create_glossary_backup contract: %s", exc)
                return self._ask_continue("Backup Failed", "Backups are not available in this build "
                                          f"(create_glossary_backup: {exc}).\n\nContinue anyway?")
            raise
        except Exception as exc:
            self.log(f"⚠️ Backup failed: {exc}")
            return self._ask_continue("Backup Failed", f"Failed to create backup: {exc}\n\nContinue anyway?")
        return result is not False

    def _ask_continue(self, title: Any, text: Optional[str] = None) -> bool:
        if text is None:  # called as ask_continue(text) by a shared function
            title, text = "Backup Failed", str(title)
        asker = self.ask_continue
        if asker is None:
            self.log(f"{title}: {text}")
            return True
        try:
            return bool(asker(str(title), str(text)))
        except Exception:
            log.exception("asking to continue failed")
            return False

    def manual_backup(self, doc: Any) -> bool:
        """Backup Settings › Backup Now (``create_glossary_backup("manual")``)."""
        if not doc.current_glossary_data:
            raise ValueError("No glossary loaded")
        return self.backup_callback(doc, "manual")

    # ---- glossary files of inputs (Library: delete / restore) --------------------------------------

    def input_paths_for_books(self, books: Sequence[Mapping[str, Any]]) -> list:
        """The EPUB paths the desktop closures key on (base name = the book): the raw source, else a path
        named after the output folder."""
        out = []
        for book in books:
            path = ""
            if self.library is not None:
                try:
                    path = self.library.raw_source(book)
                except Exception:
                    path = ""
            if not path:
                folder = str(book.get("output_folder") or "")
                name = os.path.basename(os.path.normpath(folder)) if folder else str(book.get("name") or "")
                path = os.path.join(folder or self.app_dir(), f"{name}.epub") if name else ""
            if path:
                out.append(path)
        return out

    def delete_plan(self, inputs: Sequence[str]) -> list:
        """``_delete_current_glossary``'s file list: [(book base, path)] for the inputs, deduplicated."""
        cfg = self.config_snapshot()
        guess = self.fn("guess_glossary")
        result = self.call("delete_plan", inputs=list(inputs), input_files=list(inputs), epubs=list(inputs),
                           paths=list(inputs), config=cfg, settings=cfg,
                           override_dir=os.environ.get("OUTPUT_DIRECTORY") or cfg.get("output_directory"),
                           app_dir=self.app_dir(), guess=guess, guess_glossary=guess,
                           guess_glossary_for_input_file=guess, active_glossary=None, auto_loaded_glossary_path=None,
                           manual_glossary_path=None, manually_loaded=False)
        return [(str(b), str(p)) for b, p in (result or [])]

    def delete_prompt(self, plan: Sequence[tuple]) -> str:
        """The desktop "Delete Glossary" text: "Delete the following files?" + the [book] groups
        (``glossary_files.glossary_delete_display``)."""
        display = self.call("delete_display", all_files=[(str(b), str(p)) for b, p in plan])
        return "Delete the following files?\n\n" + "\n".join(display)

    def delete_files(self, plan: Sequence[tuple]) -> list:
        """Move the files to ``<dir>/Backups/<timestamp>/`` (the desktop delete); returns "book/file" names."""
        result = self.call("delete_files", files=list(plan), all_files=list(plan), plan=list(plan),
                           append_log=self.log, log=self.log)
        deleted = [str(x) for x in (result or [])]
        if deleted:
            self.log(f"🗑️ Deleted ({len(deleted)} files backed up): {', '.join(deleted)}")
        manual = self.cfg("manual_glossary_path", "")
        if manual and any(_norm(manual) == _norm(p) for _b, p in plan):
            self.set_cfg("manual_glossary_path", "")
        return deleted

    def latest_backup(self, inputs: Sequence[str]) -> tuple:
        """``_find_latest_backup``: (backup folder, file names) across the inputs, or (None, [])."""
        cfg = self.config_snapshot()
        result = self.call("latest_backup", inputs=list(inputs), input_files=list(inputs), epubs=list(inputs),
                           paths=list(inputs), config=cfg, settings=cfg,
                           override_dir=os.environ.get("OUTPUT_DIRECTORY") or cfg.get("output_directory"),
                           app_dir=self.app_dir())
        if isinstance(result, tuple) and len(result) == 2:
            return (str(result[0]) if result[0] else None), list(result[1] or [])
        return None, []

    @staticmethod
    def restore_prompt(backup_dir: str, files: Sequence[str]) -> str:
        """The desktop "Restore Glossary" text."""
        return f"Restore from backup ({os.path.basename(backup_dir)})?\n\n" + "\n".join(files)

    def restore_files(self, backup_dir: str, files: Sequence[str]) -> list:
        """Copy the backup's files back next to ``Backups`` (the desktop restore); returns the names."""
        result = self.call("restore_latest", backup_dir=backup_dir, backup_files=list(files), files=list(files),
                           append_log=self.log, log=self.log)
        restored = [str(x) for x in (result or [])]
        if restored:
            self.log(f"↩️ Restored from {os.path.basename(backup_dir)}: {', '.join(restored)}")
        return restored

    # ---- manual glossary -----------------------------------------------------------------------------

    def load_as_manual(self, path: str, *, epub_path: Optional[str] = None) -> dict:
        """Editor "Load" (``load_current_glossary_as_manual``): manual_glossary_path + Append Glossary on;
        in Manual Glossary Only mode the file is also copied to the EPUB's output folder as glossary.csv
        (``glossary_document.copy_glossary_to_epub_output``). Returns {'updates', 'copied_to', 'mode'}."""
        mode = str(self.cfg("auto_glossary_mode", "") or "").lower()
        updates: dict = {"append_glossary": True, "manual_glossary_path": path}
        copied_to = None
        self.set_many(updates)
        self.log(f"\U0001F4D1 Loaded manual glossary: {path}")
        self.log("\u2705 Automatically enabled 'Append Glossary to System Prompt'")
        if mode == "off_no_automap":
            if not epub_path:
                self.log("\u26a0\ufe0f Manual Glossary Only: no EPUB selected \u2014 skipping output-folder copy.")
            else:
                copy_to_output = self.core.fn("glossary_document", "copy_glossary_to_epub_output")
                if copy_to_output is None:
                    raise CoreMissing("glossary_document.copy_glossary_to_epub_output")
                try:
                    copy_to_output(path, epub_path, self.config_snapshot(), self.log)
                    resolve = self.core.fn("glossary_document", "resolve_epub_output_dir")
                    out_dir = resolve(epub_path, self.config_snapshot())[0] if resolve is not None else None
                    copied_to = os.path.join(out_dir, "glossary.csv") if out_dir else None
                except Exception as exc:
                    self.log(f"\u26a0\ufe0f Failed to copy glossary to EPUB output folder: {exc}")
        return {"updates": updates, "copied_to": copied_to, "mode": mode}

    @staticmethod
    def load_prompt(path: str, mode: str) -> tuple:
        """The desktop "Load Glossary" confirmation (text, informative text)."""
        is_manual_only = mode == "off_no_automap"
        display_mode = "Manual Glossary Only" if is_manual_only else (dict(GLOSSARY_MODES).get(mode) or mode
                                                                      or "unknown")
        text = f"Load this glossary for translation?\n\n{os.path.basename(path)}"
        if is_manual_only:
            info = ("Auto Glossary mode is \"Manual Glossary Only\" — the file will be copied into the selected "
                    "EPUB's output folder as glossary.csv so translation picks it up on next run. The output "
                    f"folder will be created if it doesn't exist.\n\nCurrent mode: {display_mode}")
        else:
            info = ("For the loaded glossary to take effect, Auto Glossary mode must be set to \"Manual Glossary "
                    "Only\" — otherwise auto-mapping may override this manual selection.\n\n"
                    f"Current mode: {display_mode}")
        return text, info

    # ---- mode ----------------------------------------------------------------------------------------

    def mode(self) -> str:
        return str(self.cfg("auto_glossary_mode", "off") or "off")

    def modes(self) -> list:
        """(mode, label) in the desktop combo order (``settings_rules.auto_glossary_modes`` /
        ``glossary_mode_label``); the constant table when the rules module is missing."""
        rules = self.core.module("settings_rules")
        if rules is not None and hasattr(rules, "auto_glossary_modes") and hasattr(rules, "glossary_mode_label"):
            try:
                return [(m, rules.glossary_mode_label(m)) for m in rules.auto_glossary_modes()]
            except Exception:
                log.exception("settings_rules.auto_glossary_modes failed")
        return list(GLOSSARY_MODES)

    def mode_label(self, mode: Optional[str] = None) -> str:
        mode = self.mode() if mode is None else mode
        return dict(self.modes()).get(mode, mode)

    def set_mode(self, mode: str) -> dict:
        """The glossary mode selector: ``settings_rules.apply_change(config, 'auto_glossary_mode', mode)`` (the
        desktop shortcut handler) + the Glossary Manager lock pass (``apply_glossary_mode_locks``).
        Writes only the keys that changed; returns them."""
        rules = self.core.module("settings_rules")
        config = self.config_snapshot()
        changed: dict = {}
        if rules is not None and hasattr(rules, "apply_change"):
            changed, _env = rules.apply_change(config, "auto_glossary_mode", mode)
            changed = dict(changed)
            changed.setdefault("auto_glossary_mode", mode)
        else:
            changed = {"auto_glossary_mode": mode}
        changed.update(self.mode_lock_changes(config, mode))
        self.set_many(changed)
        return changed

    def mode_lock_changes(self, config: Optional[dict] = None, mode: Optional[str] = None) -> dict:
        """The Glossary Manager's lock pass on open / mode change (``apply_glossary_mode_locks``):
        the forced toggle values that differ from ``config`` (applied to it)."""
        rules = self.core.module("settings_rules")
        if rules is None or not hasattr(rules, "apply_glossary_mode_locks"):
            return {}
        config = self.config_snapshot() if config is None else config
        try:
            return dict(rules.apply_glossary_mode_locks(config, mode) or {})
        except Exception:
            log.exception("glossary mode lock pass failed")
            return {}

    def apply_mode_locks(self) -> dict:
        changed = self.mode_lock_changes()
        if changed:
            self.set_many(changed)
        return changed

    # ---- jobs ---------------------------------------------------------------------------------------

    def has_job_kind(self, kind: str) -> bool:
        jobs = self.jobs
        checker = getattr(jobs, "has_kind", None) if jobs is not None else None
        if callable(checker):
            try:
                return bool(checker(kind))
            except Exception:
                return False
        return False

    async def submit(self, spec: Any) -> Optional[str]:
        jobs = self.jobs
        if jobs is None:
            return None
        result = jobs.submit(spec)
        if asyncio.iscoroutine(result) or isinstance(result, asyncio.Future):
            result = await result
        return result

    @staticmethod
    def _spec(kind: str, title: str, inputs: Sequence[str] = (), params: Optional[dict] = None,
              origin: Optional[dict] = None) -> Any:
        from glossarion_mobile.services.jobs import JobSpec

        return JobSpec(kind=kind, title=title, inputs=tuple(inputs), params=dict(params or {}),
                       origin=dict(origin or {"type": "glossary", "label": "Glossaries"}))

    def extract_spec(self, inputs: Sequence[str], *, title: Optional[str] = None, origin: Optional[dict] = None,
                     force_balanced_request_merging: bool = False) -> Any:
        """``extract_glossary`` over files: EPUB/TXT/PDF/subtitles, or the images of one folder (the desktop
        groups image files by folder into one combined glossary)."""
        files = [os.path.abspath(p) for p in inputs if p]
        if not files:
            raise ValueError("Pick a file or an image folder to extract from")
        first = files[0]
        label = title or (os.path.basename(os.path.dirname(first)) if _is_image(first) else
                          os.path.splitext(os.path.basename(first))[0])
        if len(files) > 1 and not all(_is_image(p) for p in files):
            label = f"{label} +{len(files) - 1}"
        params = {"force_balanced_request_merging": True} if force_balanced_request_merging else {}
        return self._spec("extract_glossary", label, files, params, origin)

    def refine_supported(self) -> Optional[str]:
        """None when "✨ Refine" can run, else the reason it is disabled."""
        if not self.has_job_kind("glossary_refine"):
            return "The glossary refinement job is not available in this session"
        if not self.available("refine_run"):
            return ("Needs the shared manual-refinement runner (glossary_progress_core."
                    "run_manual_glossary_refinement), not in this build")
        return None

    def refine_spec(self, *, glossary_path: str, progress_path: Optional[str], source_path: Optional[str],
                    selected_types: Sequence[str], target_chunk_count: Optional[int] = None,
                    title: Optional[str] = None, origin: Optional[dict] = None) -> Any:
        """``glossary_refine``: the Glossary Progress "✨ Refine this" / refinement pass (manual refinement
        with ``force`` + ``run_when_disabled``, the desktop RefinementRunOptions of the preview)."""
        if not selected_types:
            raise ValueError("Choose at least one entry type")
        params = {"glossary_path": glossary_path, "progress_path": progress_path or "",
                  "source_path": source_path or "", "selected_types": list(selected_types)}
        if target_chunk_count:
            params["target_chunk_count"] = int(target_chunk_count)
        label = title or os.path.splitext(os.path.basename(glossary_path))[0]
        return self._spec("glossary_refine", label, (glossary_path,), params, origin)

    def refine_types(self, glossary_path: str) -> list:
        """(entry type, entry count) of the active refinement types for the Refine sheet."""
        types = [t for t, cfg in self.custom_entry_types().items() if not isinstance(cfg, Mapping) or cfg.get(
            "enabled", True)]
        if "term" in types and "terms" not in types:
            types[types.index("term")] = "terms"
        types = sorted(types, key=lambda name: (name not in ("character", "terms"), name))
        counts: dict = {}
        parse = self.core.fn("glossary_usage", "parse_glossary_file")
        if parse is not None and glossary_path and os.path.isfile(glossary_path):
            try:
                for entry in parse(glossary_path) or []:
                    key = str((entry or {}).get("type") or "").strip().casefold()
                    counts[key] = counts.get(key, 0) + 1
            except Exception:
                log.exception("counting glossary entry types failed")
        return [(t, counts.get(t.casefold(), 0)) for t in types]

    def unified_spec(self) -> Any:
        """``unified_glossary`` job: Rebuild Now (``unified_glossary.rebuild_now``)."""
        return self._spec("unified_glossary", "Unified glossary", (), {"shared_dir": self.primary_shared_dir()})

    def unified_location(self) -> str:
        """"Location: Glossary/Unified Glossary/<key>/glossary_unified.csv" (``_unified_glossary_folder_key``)."""
        ug = self.core.module("unified_glossary")
        key = ""
        if ug is not None and hasattr(ug, "describe_folder_key"):
            try:
                key = ug.describe_folder_key(self.cfg("unified_glossary_source_language", "auto"),
                                             bool(self.cfg("unified_glossary_combine_all_languages", False)),
                                             self.cfg("output_language", "English") or "English")
            except Exception:
                key = ""
        return f"Location: Glossary/Unified Glossary/{key}/glossary_unified.csv" if key else ""

    def pair_spec(self, result: Mapping[str, Any]) -> Any:
        """``parallel_pair``: build the paired EPUB and extract its glossary (desktop Accept + Extract)."""
        raw = str(result.get("raw_path") or "")
        translated = str(result.get("translated_path") or "")
        if not raw or not translated:
            raise ValueError("Pick both the raw and the translated EPUB")
        label = f"{os.path.splitext(os.path.basename(raw))[0]} ↔ {os.path.splitext(os.path.basename(translated))[0]}"
        selection = self.pair_selection(result)
        params = {"selection": selection, "raw_path": raw, "translated_path": translated,
                  "wrapper_prompt": str(result.get("wrapper_prompt") or ""),
                  "system_prompt": str(result.get("system_prompt") or ""),
                  "profile_name": str(result.get("profile_name") or "")}
        return self._spec("parallel_pair", label, (raw, translated), params)

    def pair_selection(self, result: Mapping[str, Any]) -> dict:
        """The chapter-text-free selection persisted for a pair (``compact_parallel_epub_selection``):
        the job rebuilds the pairs from it (``restore_parallel_epub_pairs``), so no chapter HTML is ever
        checkpointed in the job's spec."""
        compact = None
        if self.available("pair_compact"):
            try:
                compact = self.call("pair_compact", result=dict(result), selection=dict(result))
            except CoreMissing:
                compact = None
        if not isinstance(compact, Mapping):
            mapping = [{"raw_filename": str(_get(p, "raw_filename", "") or ""),
                        "translated_filename": str(_get(p, "translated_filename", "") or "")}
                       for p in (result.get("pairs") or result.get("mapping") or [])]
            compact = {"raw_path": str(result.get("raw_path") or ""),
                       "translated_path": str(result.get("translated_path") or ""), "mapping": mapping,
                       "profile_name": str(result.get("profile_name") or ""),
                       "wrapper_prompt": str(result.get("wrapper_prompt") or ""),
                       "system_prompt": str(result.get("system_prompt") or "")}
        return _jsonable(dict(compact))

    # ---- parallel EPUB pair (blocking) --------------------------------------------------------------

    def pair_chapters(self, epub_path: str) -> Any:
        """``_load_parallel_epub_chapters`` (``parallel_epub_core.load_parallel_epub_chapters``): every
        document with metadata, special files kept."""
        return self.call("pair_chapters", epub_path=epub_path)

    def pair_load(self, epub_path: str) -> dict:
        """One side of the pair as the dialog loads it (``ParallelEpubPairDialog._start_epub_load`` ->
        ``parallel_epub_core.load_parallel_epub_documents``): the documents with readable text as
        {text, filename}, every document's filename in reading order and the dialog's error text."""
        chapters, reading_order, error = self.call("pair_load", chapter_loader=self.pair_chapters, path=epub_path)
        return {"chapters": list(chapters or []), "reading_order": list(reading_order or []), "error": error or "",
                "path": epub_path}

    def special_file_predicate(self) -> Optional[Callable[[str], bool]]:
        """The run's special-file test (``TranslationPipelineMixin._is_special_file`` over the configured
        keyword / exact lists, as TranslatorGUI passes it to the pair dialog)."""
        tp = self.core.module("translation_pipeline")
        mixin = getattr(tp, "TranslationPipelineMixin", None) if tp is not None else None
        test = getattr(mixin, "_is_special_file", None) if mixin is not None else None
        if not callable(test):
            return None
        owner_state = self.core.module("owner_state")
        keywords = self.cfg("special_file_keywords", None)
        exact = self.cfg("special_file_exact", None)
        if keywords is None:
            keywords = getattr(owner_state, "_DEFAULT_SPECIAL_KEYWORDS", "") if owner_state is not None else ""
        if exact is None:
            exact = getattr(owner_state, "_DEFAULT_SPECIAL_EXACT", "") if owner_state is not None else ""
        holder = _Holder(special_file_keywords_var=keywords, special_file_exact_var=exact)
        upgrade = getattr(getattr(owner_state, "ConfigStateMixin", None), "_upgrade_special_file_exact", None)
        if callable(upgrade):
            try:
                holder.special_file_exact_var = upgrade(holder, exact)
            except Exception:
                pass
        return lambda filename: bool(test(holder, filename))

    def pair_auto_map(self, raw_chapters: Any, translated_chapters: Any, *, auto_offset: bool = True,
                      raw_reading_order: Any = None, translated_reading_order: Any = None) -> tuple:
        """(``auto_map_epub_chapters`` result, the special-file predicate it used)."""
        predicate = self.special_file_predicate()
        mappings = self.call("pair_auto_map", raw_chapters=raw_chapters, translated_chapters=translated_chapters,
                             enable_auto_offset=auto_offset, special_file_predicate=predicate,
                             protect_interior_special_files=bool(self.cfg(
                                 "never_consider_in_between_files_as_special", True)),
                             raw_reading_order=raw_reading_order, translated_reading_order=translated_reading_order)
        return list(mappings or []), predicate

    def pair_write_sidecar(self, selection: Mapping[str, Any]) -> Optional[str]:
        """The mapping sidecar beside the raw book's glossary (``_write_parallel_epub_mapping_sidecar``)."""
        if not self.available("pair_sidecar_write"):
            return None
        path = self.call("pair_sidecar_write", selection=dict(selection), config=self.config_snapshot(),
                         output_directory=(self.output_roots() or [None])[0])
        if path:
            self.log(f"💾 Saved Parallel EPUB mapping: {path}")
        return str(path) if path else None

    def pair_saved_selection(self, raw_path: str) -> Optional[dict]:
        """The saved mapping of a pair (sidecar beside the raw book's glossary, else config)."""
        selection = None
        if self.available("pair_sidecar_read"):
            try:
                selection = self.call("pair_sidecar_read", raw_path=raw_path, config=self.config_snapshot(),
                                      output_directory=(self.output_roots() or [None])[0])
            except CoreMissing:
                selection = None
        if not isinstance(selection, Mapping) or not selection.get("mapping"):
            key = self.core.value("parallel_epub_core", "PARALLEL_EPUB_SELECTION_CONFIG_KEY",
                                  default="parallel_epub_pair_selection")
            saved = self.cfg(key, None)
            if isinstance(saved, Mapping) and saved.get("mapping") and _norm(saved.get("raw_path") or "") == _norm(
                    raw_path):
                selection = saved
        return dict(selection) if isinstance(selection, Mapping) else None

    def default_wrapper_prompt(self) -> str:
        return str(self.cfg("parallel_epub_glossary_wrapper_prompt", "") or self.core.value(
            "parallel_epub_core", "DEFAULT_PARALLEL_EPUB_WRAPPER_PROMPT", default="") or "")

    def default_pair_system_prompt(self) -> str:
        if self.available("pair_system_prompt"):
            try:
                return str(self.call("pair_system_prompt") or "")
            except CoreMissing:
                return ""
        return ""

    def pair_profiles(self) -> tuple:
        """(profiles, active profile) the pair dialog opens with (``parallel_epub_profiles`` /
        ``active_parallel_epub_profile``: the saved profiles plus the built-in default)."""
        config = self.config_snapshot()
        profiles = dict(self.call("pair_profiles", config=config) or {})
        return profiles, str(self.call("pair_active_profile", config=config, profiles=profiles) or "")

    def pair_persist_prompts(self, profiles: Mapping[str, Any], active_profile: str, wrapper_prompt: str) -> dict:
        """``_persist_prompt_settings``: the three config values (``parallel_epub_prompt_settings``)."""
        updates = dict(self.call("pair_prompt_settings", profiles=dict(profiles), active_profile=active_profile,
                                 wrapper_prompt=wrapper_prompt) or {})
        self.set_many(updates)
        return updates

    # ---- prompt profiles ---------------------------------------------------------------------------

    def prompt_profiles(self, bucket_id: str, *, default_text: str = "") -> Any:
        """The profile row of one Glossary settings tab, opened like the desktop tab: the shared
        ``glossary_document.GlossaryPromptProfiles`` (Balanced/Full, Minimal) or
        ``RefinementPromptProfiles`` over a copy of config.json (the current prompt is staged into the
        selected profile, then the active profile is applied). Every action the desktop saves calls
        :meth:`persist_prompt_profiles`; ``default_text`` is the prompt's (schema) default when
        config.json has none."""
        if bucket_id not in PROFILE_BUCKETS:
            raise KeyError(bucket_id)
        gd = self.gd()
        holder: dict = {}

        def persist() -> bool:
            return self.persist_prompt_profiles(holder["profiles"])

        owner = ProfileOwner(self.config_snapshot())
        if bucket_id == "refinement":
            cls = getattr(gd, "RefinementPromptProfiles", None)
            if cls is None:
                raise CoreMissing("glossary_document.RefinementPromptProfiles")
            system = str(owner.config.get(cls.SYSTEM_KEY) or default_text or "")
            user = str(owner.config.get(cls.USER_KEY) or "")
            profiles = cls(owner, system, user, persist=persist, log=self.log)
        else:
            cls = getattr(gd, "GlossaryPromptProfiles", None)
            if cls is None:
                raise CoreMissing("glossary_document.GlossaryPromptProfiles")
            meta = gd.glossary_prompt_profile_meta(bucket_id)
            current = str(owner.config.get(meta["config_key"]) or default_text or "")
            setattr(owner, meta["attr"], current)
            profiles = cls(owner, bucket_id, current, persist=persist, log=self.log)
        holder["profiles"] = profiles
        return profiles

    def prompt_profile_updates(self, profiles: Any) -> dict:
        """The config.json values a profile row changed, against config.json now: its own entry of the
        dicts Balanced/Full and Minimal share (the other row's entry is kept) and its prompt keys; for
        Refinement its five keys."""
        mine = profiles.owner.config
        fresh = self.config_snapshot()
        updates: dict = {}
        if hasattr(profiles, "PROFILES_KEY"):  # RefinementPromptProfiles
            for key in (profiles.PROFILES_KEY, profiles.DEFAULT_KEY, profiles.ACTIVE_KEY, profiles.SYSTEM_KEY,
                        profiles.USER_KEY):
                if key in mine and mine[key] != fresh.get(key):
                    updates[key] = copy.deepcopy(mine[key])
            return updates
        bucket = profiles.key
        for key in _GLOSSARY_PROFILE_DICT_KEYS:
            own = mine.get(key) if isinstance(mine.get(key), Mapping) else {}
            current = fresh.get(key)
            merged = dict(current) if isinstance(current, Mapping) else {}
            if bucket in own:
                merged[bucket] = copy.deepcopy(own[bucket])
            else:
                merged.pop(bucket, None)
            if merged != current:
                updates[key] = merged
        meta = self.gd().glossary_prompt_profile_meta(bucket)
        for key in (meta.get("config_key"), meta.get("legacy_config_key")):
            if key and key in mine and mine[key] != fresh.get(key):
                updates[key] = mine[key]
        return updates

    def persist_prompt_profiles(self, profiles: Any) -> bool:
        """Write :meth:`prompt_profile_updates` (the desktop ``save_config`` after a profile action).
        False when config.json did not take them (the shared row then rolls the action back)."""
        updates = self.prompt_profile_updates(profiles)
        if not updates:
            return True
        config = self.config
        if config is None:
            return False
        try:
            if isinstance(config, dict):
                config.update(updates)
            elif callable(getattr(config, "set_many", None)):
                config.set_many(updates)
            else:
                for key, value in updates.items():
                    config.set(key, value)
        except Exception:
            log.exception("saving the prompt profiles failed")
            return False
        return True

class _Holder:
    """Attribute bag for shared methods called with an explicit ``self`` (the special-file test)."""

    def __init__(self, **values: Any) -> None:
        self.__dict__.update(values)


def _is_image(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp")


def _jsonable(value: Any, depth: int = 0) -> Any:
    if depth > 8:
        return repr(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(v, depth + 1) for v in value]
    if hasattr(value, "__dict__"):
        return {k: _jsonable(v, depth + 1) for k, v in vars(value).items() if not k.startswith("_")}
    return repr(value)
