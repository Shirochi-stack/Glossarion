"""Cloud sync (U10): copy the Library's finished books into the user's own cloud storage.

Owner decisions (2026-10-09): the source is the mobile **Library** (finished chat books
auto-migrate into it); its books' compiled outputs (EPUB, PDF, TXT, HTML) are mirrored, one
folder per book like the Library keeps them, into ONE destination the user picks once through
the system document picker. Glossarion never talks to a cloud service: it hands each file to a
document provider on the phone (Android Storage Access Framework, iOS Files / File Provider) and
the user's own cloud app uploads it with the user's own account. No developer account, OAuth
client, API key, server or telemetry; nothing here makes a network request.

Destinations (``Destination.mode``)
  * ``folder``: a folder picked once (``GlossarionNative.pick_folder``: Android
    ``ACTION_OPEN_DOCUMENT_TREE`` + persisted grant; iOS folder picker + bookmark). Each book gets
    its own sub-folder (the Library's per-book layout; ``flat`` when the provider cannot create
    folders), each output its own document, overwritten in place on every recompile.
  * ``files``: one save location per output (``pick_save_location``: Android
    ``ACTION_CREATE_DOCUMENT``; iOS export once + bookmark) for providers not offered as folders
    (Google Drive on many Android phones; Drive / OneDrive / Dropbox / Box on iOS). A new output
    needs one tap ("needs a save location"); later saves update that file.
  * ``phone``: Android ``Downloads/Glossarion/<book>`` through MediaStore, overwriting the same
    entry (``save_to_downloads(replace_uri=...)``); a backup app such as TeraBox can watch it.

Opt-in: a global switch (off by default), a per-book override Default / Always / Never and
per-format toggles (all on). Stored mobile-only: Prefs key ``cloud_sync`` (``mobile_state.json``)
holds the switch, formats and destination; ``<data>/mobile_cloud.json``
(``state/cloud_records.CloudRecordStore``) holds per-book records keyed by destination id + book
+ kind, the per-book overrides and the queue. ``config.json`` gets no key; nothing is secret.

What is uploaded (critic #0/#1): a book's candidates per kind are the union of
``LibraryService.compiled_outputs_blocking`` (``library_core.list_compiled_outputs`` + the
Library-filed EPUB), ``job_kinds.compiled_outputs`` (top-level ``*.epub`` / ``*.pdf``: the PDF an
EPUB book compiles has no ``_translated`` suffix) and the files the finishing job reported; the
choice is deterministic: the newest of the reported files, else the newest candidate. The cloud
file keeps its first name when the title changes (one-way push, always overwrite).

Write pipeline (one book at a time, one file at a time, never on the UI loop):
  1. busy guard: no active / queued job writes the workspace (``JOB_LOCK`` is never taken);
  2. change check: same source + size + mtime -> skip; else a **private snapshot** of the source
     is copied into the app cache on the io pool (critic #6: ebooklib rewrites the same inode),
     hashed, and skipped when the hash matches the last upload;
  3. the book folder and the document are found again or **adopted by name** before anything is
     created (records are lost on reinstall / wipe; Drive allows duplicate names; critic #9); a
     name another book's record holds gets ' (2)' (same-title books);
  4. ``write_file(doc, snapshot)`` with the mode chain ``wt -> rwt -> w`` (the native side uses a
     non-truncating mode only when the new file is not shorter and reads the length back); a
     ``needs_replace`` answer (stale tail / no usable mode) **replaces** the file: create new,
     write, delete old, with a warning that its link changed (critic #5). The new copy is recorded
     (``replacing``, flushed) before its first byte and an old copy whose delete failed stays in
     ``orphans``: later drains delete both, so a kill mid-replace or a failed delete never leaves a
     second copy for good. A provider that refuses sub-folders (``read_only`` from ``create_folder``:
     "Create not supported") makes the destination flat;
  5. ``missing`` is believed only when the destination answers and the document is not in its
     folder (critic #2; the native side proves it, this side checks again); then it is created
     again, once. A permission error on one document affects only that record while the root
     still answers (critic #7); otherwise the destination needs a re-link (one notification).
  6. errors back off: 2 s once, then 1, 5, 15, 60 min, then every 6 h; after 5 attempts the book
     shows "Couldn't save" plus one notification per drain counting the books that wait, and is still
     retried on resume, the next compile and Send now. ``no_space`` is its own error (no tight retry
     loop).
  7. the queue entry a drain took is finished / rescheduled only while no newer trigger came in
     (``gen``): a recompile or a format turned on during the copy is copied next. A record change that
     lands after its book was deleted or its destination was changed is dropped (``_rec``); a write a
     destination change cancels only pauses the drain. The app is kept alive (foreground service / iOS
     background task) right before the first write of a pass, never for a pass with nothing to write,
     and nothing is written after Stop / a service timeout, also when it came during the private copy.

Triggers: a Library job reaching DONE (``on_job_transition``; chat runs wait for the auto-migrate:
``on_workspace_moved``), "Send now" (``send_now``), enabling sync / a per-book Always
(``sync_all`` / ``set_book_override``), app start and resume, a destination linked again.
Android: the 'cloud' ``ServiceHolds`` holder joins the job's foreground service synchronously in
the transition callback, so the service outlives the job until the queue drains (``Keepalive``);
iOS: a background task for the drain plus the holder that asks ``BackgroundExecution`` to keep the
job's grant (``release_kept_background``, patched into background.py). A foreground-service
timeout / destroyed event cancels the write and drops the hold (critic #18).

Book deletion (``forget_book``), "Wipe app data" (``wipe``: releases every persisted grant first)
and disconnecting (``forget_destination``) drop records and queue entries; files already in the
cloud are never deleted. Save locations mode gives a file's persisted grant back as soon as its record
stops pointing at it. Glossarion's own storage is refused as a destination (a save location there is
taken back out on Android) and nothing in Library/Raw is ever a book to copy (a Tools job's raw input
neither). A picker answer that arrives after the process was recreated is taken from
``take_document_results``; auto-migrate moves reported while ``start`` loads the records wait for it.

The native side is reached through ``NativeDocs`` (the ``flet_glossarion_native`` document API:
refs are JSON dicts stored as they come back); host tests use an in-memory fake with the same
normalised answers. The UI talks to the ``ui_state`` / ``book_state`` / action methods
(``ui/screens/cloud_sync.CloudFacade``). Pure asyncio; no Flet import.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import logging
import os
import shutil
import threading
import time
import uuid
import zlib
from dataclasses import dataclass, field, replace
from typing import Any, Awaitable, Callable, Iterable, Mapping, Optional, Sequence

from glossarion_mobile.services import files as files_mod  # pure (no Flet at import): phone folder names, mime
from glossarion_mobile.state.cloud_records import CLOUD_FILE, KINDS, OVERRIDES, CloudRecordStore, book_key, ref_key

__all__ = [
    "BACKOFF",
    "CLOUD_HOLD",
    "CloudNotifier",
    "CloudSettings",
    "CloudSyncService",
    "DEFAULT_MODE_CHAIN",
    "Destination",
    "FAIL_AFTER",
    "KINDS",
    "KIND_LABELS",
    "Keepalive",
    "MODE_FILES",
    "MODE_FOLDER",
    "MODE_PHONE",
    "NativeDocs",
    "OVERRIDES",
    "OWN_FOLDER_REASON",
    "OutputChoice",
    "PREF_KEY",
    "SETTINGS_ROUTE",
    "TRIGGER_KINDS",
    "backoff_delay",
    "choose_outputs",
    "collect_outputs",
    "error_code",
    "is_own_location",
    "output_kind",
    "select_outputs",
]

log = logging.getLogger("glossarion.cloud")

PREF_KEY = "cloud_sync"
SETTINGS_ROUTE = "/settings/cloud"
CLOUD_HOLD = "cloud"  # ServiceHolds holder name (the job runner is "jobs", a sign-in "sign-in")
JOBS_HOLD = "jobs"
HOLD_TITLE = "Glossarion"

MODE_FOLDER, MODE_FILES, MODE_PHONE = "folder", "files", "phone"
MODES = (MODE_FOLDER, MODE_FILES, MODE_PHONE)
LAYOUT_PER_BOOK, LAYOUT_FLAT = "per_book", "flat"
PHONE_TARGET_ID = "phone"
FILES_TARGET_ID = "files"
KIND_LABELS = {"epub": "EPUB", "pdf": "PDF", "txt": "TXT", "html": "HTML"}
DEFAULT_MODE_CHAIN = ("wt", "rwt", "w")

#: Job kinds whose DONE can change a Library book's compiled outputs.
TRIGGER_KINDS = frozenset({"compile_epub", "compile_pdf", "translate", "single_chapter", "retranslate",
                           "resolve_qa", "metadata", "translate_headers", "async_batch"})

#: Error codes (the native-docs ``DocumentError`` values plus the core's own).
E_PERMISSION, E_MISSING, E_MODE, E_PROVIDER, E_SPACE, E_CANCELLED = (
    "permission_lost", "missing", "unsupported_mode", "provider_error", "no_space", "cancelled")
E_UNAVAILABLE, E_BUSY, E_READ_ONLY, E_SIZE, E_EXISTS, E_TIMEOUT = (
    "unavailable", "busy", "read_only", "size_mismatch", "exists", "timeout")
E_SOURCE_MISSING, E_SOURCE_CHANGED, E_BAD_ARGS = "source_missing", "source_changed", "bad_args"
_KNOWN_ERRORS = {E_PERMISSION, E_MISSING, E_MODE, E_PROVIDER, E_SPACE, E_CANCELLED, E_UNAVAILABLE, E_BUSY,
                 E_READ_ONLY, E_SIZE, E_EXISTS, E_TIMEOUT, E_SOURCE_MISSING, E_SOURCE_CHANGED, E_BAD_ARGS}
_ERROR_ALIASES = {"revoked": E_PERMISSION, "security": E_PERMISSION, "stale": E_PERMISSION,
                  "not_found": E_MISSING, "deleted": E_MISSING, "unsupported": E_MODE, "provider": E_PROVIDER,
                  "io": E_PROVIDER, "enospc": E_SPACE, "canceled": E_CANCELLED, "readonly": E_READ_ONLY}

#: Retry delays after the 1st..5th failure (the first one is Drive's flaky first write), then 6 h.
BACKOFF = (2.0, 60.0, 300.0, 900.0, 3600.0)
BACKOFF_MAX = 6 * 3600.0
FAIL_AFTER = 5
SOURCE_RETRY = 5.0  # a vanished workspace is looked up once more (it may be moving into the Library)
BUSY = "busy"  # queue ``last_error`` of a book a job is writing: woken by the next job end
BUSY_RETRY = 300.0  # ...or looked at again after 5 min (no polling while the job runs)
SNAPSHOT_DIR = "cloud_sync"
SPACE_MARGIN = 16 * 1024 * 1024
COPY_CHUNK = 1024 * 1024
IOS_MIN_REMAINING = 15.0
IOS_BYTES_PER_SECOND = 4 * 1024 * 1024
GRANT_LIMIT_OLD, GRANT_LIMIT = 128, 512  # persisted URI grants before / from Android 11
GRANT_MARGIN = 8
NOTIFY_BASE = 42000
NOTIFY_SPAN = 390
PROGRESS_INTERVAL = 0.25  # listener fan-out (4/s)
NOTIFICATION_INTERVAL = 1.0
RECENT_LIMIT = 10
OWN_PREFIXES = ("com.glossarion",)

OWN_FOLDER_REASON = "That folder is Glossarion's own storage; choose a folder of your cloud app"
READ_ONLY_REASON = "Glossarion cannot write to that folder"
NO_DESTINATION_REASON = "Choose where to save books in Settings › Cloud sync"
NOT_IN_LIBRARY_REASON = "Saves once the book is in the Library"
PHONE_ONLY_REASON = "Phone only"
REPLACED_NOTE = "replaced with a new cloud file (this cloud app cannot overwrite a shorter file); its link changed"


# ---------------------------------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------------------------------


def _norm(path: Any) -> str:
    return book_key(path)


def error_code(value: Any) -> str:
    """A native-docs error (code, result dict or text) as one of the known codes."""
    if isinstance(value, Mapping):
        value = value.get("error") or value.get("code") or ""
    value = getattr(value, "value", value)
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if not text:
        return E_PROVIDER
    if text in _KNOWN_ERRORS:
        return text
    if text in _ERROR_ALIASES:
        return _ERROR_ALIASES[text]
    if "security" in text or "permission" in text:
        return E_PERMISSION
    if "space" in text:
        return E_SPACE
    return E_PROVIDER


def backoff_delay(attempts: int) -> float:
    """Delay before the next try after ``attempts`` failures (1-based)."""
    if attempts <= 0:
        return 0.0
    return BACKOFF[attempts - 1] if attempts <= len(BACKOFF) else BACKOFF_MAX


def _mime(path: str) -> str:
    return files_mod.mime_type_for(path)


def _phone_subdir() -> str:
    return files_mod.DOWNLOADS_SUBDIR


def _phone_label() -> str:
    return files_mod.PHONE_FOLDER_LABEL


def _safe(name: str, fallback: str = "Book") -> str:
    return files_mod.safe_name(name, fallback)


def _variant(name: str, n: int) -> str:
    """``Book.epub`` -> ``Book (2).epub`` (``n`` >= 2); a folder name gets the suffix at its end."""
    stem, ext = os.path.splitext(name)
    if ext and 1 < len(ext) <= 6 and stem:
        return f"{stem} ({n}){ext}"
    return f"{name} ({n})"


def _under(path: str, root: str) -> bool:
    if not path or not root or not os.path.isabs(path):
        return False
    try:
        real, base = os.path.normcase(os.path.realpath(path)), os.path.normcase(os.path.realpath(root))
        return os.path.commonpath([real, base]) == base
    except (OSError, ValueError):
        return False


def _same(a: Any, b: Any) -> bool:
    return bool(a) and bool(b) and ref_key(a) == ref_key(b)


def is_own_location(result: Mapping[str, Any], app_roots: Sequence[str] = (),
                    own_prefixes: Sequence[str] = OWN_PREFIXES) -> bool:
    """A picked location inside Glossarion's own storage (refused as a destination): the native side's
    ``own_folder`` (app container, Download/Glossarion), an absolute path inside the app's folders
    (Files › On My iPhone › Glossarion is the output root itself), or the app's own authority."""
    ref = result.get("target") if isinstance(result.get("target"), Mapping) else \
        (result.get("doc") if isinstance(result.get("doc"), Mapping) else {})
    if result.get("own_folder") or (isinstance(ref, Mapping) and ref.get("own_folder")):
        return True
    for source in (result, ref if isinstance(ref, Mapping) else {}):
        path = str(source.get("path") or "")
        if path and any(_under(path, root) for root in app_roots if root):
            return True
        authority = str(source.get("provider") or source.get("authority") or "").lower()
        if authority and any(authority.startswith(prefix) for prefix in own_prefixes):
            return True
    return False


# ---------------------------------------------------------------------------------------------------
# settings and destination
# ---------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Destination:
    """Where books go. ``id`` keys the records (the native ref's stable id, so a refreshed iOS bookmark
    or the same folder picked again keeps them)."""

    id: str
    mode: str
    platform: str = ""
    target: Any = None  # folder mode: the native-docs folder ref (a JSON dict)
    label: str = ""  # folder name / "Downloads/Glossarion"
    provider: str = ""
    provider_label: str = ""  # "Drive"
    can_create: bool = True
    can_write: bool = True
    layout: str = LAYOUT_PER_BOOK
    linked_at: float = 0.0
    needs_relink: str = ""  # '' | 'revoked' | 'missing'
    persisted: bool = True

    @property
    def usable(self) -> bool:
        return not self.needs_relink and self.can_write

    @property
    def display(self) -> str:
        if self.mode == MODE_PHONE:
            return _phone_label()
        if self.mode == MODE_FILES:
            return (self.provider_label or "your save locations")  # the UI calls the mode "Save locations"
        provider, label = self.provider_label.strip(), self.label.strip()
        if provider and label and provider.casefold() != label.casefold():
            return f"{provider} › {label}"
        return label or provider or "the chosen folder"

    def as_dict(self) -> dict:
        return {"id": self.id, "mode": self.mode, "platform": self.platform, "target": self.target,
                "label": self.label, "provider": self.provider, "provider_label": self.provider_label,
                "can_create": self.can_create, "can_write": self.can_write, "layout": self.layout,
                "linked_at": self.linked_at, "needs_relink": self.needs_relink, "persisted": self.persisted}

    def ui(self) -> dict:
        """What ``ui/screens/cloud_sync.destination_text`` reads (no URIs, no bookmarks)."""
        return {"mode": self.mode, "label": self.label, "provider_label": self.provider_label,
                "needs_relink": self.needs_relink or None, "can_create": self.can_create, "display": self.display,
                "layout": self.layout}

    @classmethod
    def from_dict(cls, data: Any) -> Optional["Destination"]:
        if not isinstance(data, Mapping) or data.get("mode") not in MODES or not data.get("id"):
            return None
        if data.get("mode") == MODE_FOLDER and not data.get("target"):
            return None
        try:
            linked = float(data.get("linked_at") or 0.0)
        except (TypeError, ValueError):
            linked = 0.0
        layout = data.get("layout") if data.get("layout") in (LAYOUT_PER_BOOK, LAYOUT_FLAT) else LAYOUT_PER_BOOK
        return cls(id=str(data["id"]), mode=str(data["mode"]), platform=str(data.get("platform") or ""),
                   target=data.get("target"), label=str(data.get("label") or ""),
                   provider=str(data.get("provider") or ""), provider_label=str(data.get("provider_label") or ""),
                   can_create=bool(data.get("can_create", True)), can_write=bool(data.get("can_write", True)),
                   layout=layout, linked_at=linked, needs_relink=str(data.get("needs_relink") or ""),
                   persisted=bool(data.get("persisted", True)))


def folder_target_id(platform: str, target: Any) -> str:
    """Stable id of a picked folder: the native ref's ``id`` (FNV over the tree URI / canonical path)."""
    stable = ""
    if isinstance(target, Mapping):
        stable = str(target.get("id") or "")
    stable = stable or ref_key(target)
    digest = hashlib.sha1(f"{platform}|folder|{stable}".encode("utf-8", "surrogatepass")).hexdigest()[:16]
    return f"folder-{digest}"


@dataclass(frozen=True)
class CloudSettings:
    enabled: bool = False
    kinds: Mapping[str, bool] = field(default_factory=lambda: {k: True for k in KINDS})
    destination: Optional[Destination] = None
    pending_pick: Optional[Mapping[str, Any]] = None

    def kind_on(self, kind: str) -> bool:
        return bool(self.kinds.get(kind, True))

    @classmethod
    def from_raw(cls, raw: Any) -> "CloudSettings":
        raw = raw if isinstance(raw, Mapping) else {}
        kinds_raw = raw.get("kinds") if isinstance(raw.get("kinds"), Mapping) else {}
        kinds = {k: bool(kinds_raw.get(k, True)) for k in KINDS}
        pending = raw.get("pending_pick") if isinstance(raw.get("pending_pick"), Mapping) else None
        return cls(enabled=bool(raw.get("enabled", False)), kinds=kinds,
                   destination=Destination.from_dict(raw.get("destination")), pending_pick=pending)


def _answer(ok: bool, message: str = "", **extra: Any) -> dict:
    """An action's answer for the UI (``CloudFacade._result``): ``{ok, message, ...}``."""
    out = {"ok": bool(ok), "message": str(message or "")}
    out.update(extra)
    return out


# ---------------------------------------------------------------------------------------------------
# what to upload (blocking; io pool)
# ---------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class OutputChoice:
    kind: str
    path: str
    size: int
    mtime_ns: int
    reported: bool = False
    others: tuple = ()  # the other candidates of the same kind (a stale-title EPUB, ...)

    @property
    def name(self) -> str:
        return os.path.basename(self.path)

    @property
    def conflicts(self) -> int:
        return len(self.others)


def output_kind(path: str) -> str:
    """``epub`` / ``pdf`` / ``txt`` / ``html`` for a compiled output path ('' for anything else). A
    ``*_translated.html`` next to a ``*_translated.pdf`` of the same stem is the PDF's debug companion
    (the ``library_core._list_compiled_outputs`` rule) and is not an output."""
    base = os.path.basename(str(path or ""))
    lower = base.lower()
    if lower.endswith(".epub"):
        return "epub"
    if lower.endswith(".pdf"):
        return "pdf"
    if lower.endswith("_translated.txt"):
        return "txt"
    if lower.endswith("_translated.html"):
        stem = base[: -len("_translated.html")]
        if os.path.isfile(os.path.join(os.path.dirname(str(path)), f"{stem}_translated.pdf")):
            return ""
        return "html"
    return ""


def collect_outputs(book: Mapping[str, Any], *, lister: Optional[Callable[[Mapping[str, Any]], Any]] = None,
                    reported: Iterable[str] = (), reported_anywhere: bool = False) -> dict:
    """Blocking: ``{kind: [path, ...]}`` of a book's compiled outputs: the union of the Library's list
    (``lister`` = ``LibraryService.compiled_outputs_blocking``), the top-level EPUB / PDF files
    (``job_kinds.compiled_outputs``) and the files the job reported. Reported paths count when they
    sit in the book's workspace or are the book's own file; ``reported_anywhere`` (a single-book
    job's report) also takes e.g. an organized book's ``Library/Translated`` EPUB."""
    folder = str(book.get("output_folder") or "")
    own = str(book.get("path") or "")
    found: dict = {k: [] for k in KINDS}
    # Never the untranslated source: the row's raw file, nor its ``path`` when that is a source EPUB beside
    # the workspace (desktop-style in-progress rows) rather than the workspace's own output or a
    # Library-filed translation (``LibraryService.compiled_outputs_blocking`` lists ``path`` as is).
    seen: set = {_norm(book.get("raw_source_path"))} if book.get("raw_source_path") else set()
    beside = bool(own and folder) and _norm(os.path.dirname(own)) != _norm(folder)
    if beside and not book.get("in_library") and os.path.isfile(own):
        seen.add(_norm(own))

    def add(path: Any) -> None:
        text = os.fspath(path) if path else ""
        if not text:
            return
        key = _norm(text)
        if key in seen or not os.path.isfile(text):
            return
        kind = output_kind(text)
        if kind not in KINDS:
            return
        seen.add(key)
        found[kind].append(os.path.abspath(text))

    if lister is not None:
        try:
            for item in lister(book) or ():
                add(item[0] if isinstance(item, (tuple, list)) else item)
        except Exception:
            log.debug("listing the compiled outputs failed", exc_info=True)
    if folder and os.path.isdir(folder):
        try:
            from glossarion_mobile.job_kinds import compiled_outputs

            for path in compiled_outputs([folder]):
                add(path)
        except Exception:
            log.debug("job_kinds.compiled_outputs failed", exc_info=True)
    folder_key = _norm(folder) if folder else ""
    own_key = _norm(own) if own else ""
    for path in reported or ():
        if not path:
            continue
        text = os.fspath(path)
        if reported_anywhere or (folder_key and _norm(os.path.dirname(text)) == folder_key) or \
                (own_key and _norm(text) == own_key):
            add(text)
    return {k: v for k, v in found.items() if v}


def choose_outputs(candidates: Mapping[str, Sequence[str]], reported: Iterable[str] = ()) -> dict:
    """Blocking: one deterministic file per kind - the newest of the reported files, else the newest
    candidate (ties: the name), so a stale-title EPUB left in the workspace never wins."""
    reported_keys = {_norm(p) for p in reported or () if p}
    out: dict = {}
    for kind, paths in candidates.items():
        stats = []
        for path in paths:
            try:
                st = os.stat(path)
            except OSError:
                continue
            stats.append((path, st.st_size, st.st_mtime_ns))
        if not stats:
            continue
        pool = [s for s in stats if _norm(s[0]) in reported_keys] or stats
        best = max(pool, key=lambda s: (s[2], os.path.basename(s[0]).casefold(), s[0]))
        others = tuple(sorted(s[0] for s in stats if s[0] != best[0]))
        out[kind] = OutputChoice(kind=kind, path=best[0], size=int(best[1]), mtime_ns=int(best[2]),
                                 reported=_norm(best[0]) in reported_keys, others=others)
    return out


def select_outputs(book: Mapping[str, Any], *, lister: Optional[Callable[[Mapping[str, Any]], Any]] = None,
                   reported: Iterable[str] = (), reported_anywhere: bool = False) -> dict:
    """Blocking: ``{kind: OutputChoice}`` (``collect_outputs`` + ``choose_outputs``)."""
    reported = [os.fspath(p) for p in reported or () if p]
    return choose_outputs(collect_outputs(book, lister=lister, reported=reported,
                                          reported_anywhere=reported_anywhere), reported)


# ---------------------------------------------------------------------------------------------------
# native-docs backend
# ---------------------------------------------------------------------------------------------------


def _plain(value: Any) -> Any:
    from glossarion_mobile.services.native import to_plain

    return to_plain(value)


def _norm_result(raw: Any, *, default_error: str = E_UNAVAILABLE) -> dict:
    """A native answer as ``{ok, error, message, scope, ...}`` (``error`` one of the known codes)."""
    raw = _plain(raw)
    if not isinstance(raw, Mapping):
        return {"ok": False, "error": default_error, "message": "", "scope": None}
    out = {str(k): v for k, v in raw.items()}
    if out.get("ok") is True:
        out["error"] = ""
    else:
        out["ok"] = False
        out["error"] = error_code(out.get("error") or default_error)
    out["message"] = str(out.get("message") or "")
    out.setdefault("scope", None)
    return out


def _ref_name(ref: Any) -> str:
    return str(ref.get("name") or "") if isinstance(ref, Mapping) else ""


def _child(ref: Any) -> Optional[dict]:
    ref = _plain(ref)
    if not isinstance(ref, Mapping) or not ref.get("name"):
        return None
    try:
        size = int(ref.get("size")) if ref.get("size") is not None else None
    except (TypeError, ValueError):
        size = None
    return {"doc": dict(ref), "name": str(ref.get("name")), "is_dir": ref.get("kind") == "folder", "size": size}


class NativeDocs:
    """The native document API of ``flet_glossarion_native`` (``GlossarionNative.pick_folder``,
    ``pick_save_location``, ``pick_document``, ``list_children``, ``create_file``, ``create_folder``,
    ``write_file``, ``stat``, ``delete``, ``query_root``, ``release``, ``list_grants``,
    ``cancel_document_op``, ``take_document_results``, ``save_to_downloads(replace_uri=)``), reached
    through ``NativeBridge.native``. Every answer is normalised (``_norm_result``); the extension
    applies its own timeouts (pickers 1 h, queries 120 s, writes 120 s + 1 s/MiB) and cancels a
    write it stops waiting for, so no outer guard cuts a running copy short (``NativeBridge.call``'s
    45 s would). A stub / desktop answers ``unavailable``."""

    def __init__(self, bridge: Any, *, platform: str = "desktop") -> None:
        self.bridge = bridge
        self.platform = platform

    @property
    def _native(self) -> Any:
        return getattr(self.bridge, "native", self.bridge)

    @property
    def available(self) -> bool:
        if self.platform not in ("android", "ios"):
            return False
        native = self._native
        if native is None or getattr(native, "is_stub", False):
            return False
        if getattr(native, "native_available", True) is False:  # no Dart service (flet run, companion app)
            return False
        return callable(getattr(native, "write_file", None))

    async def _call(self, method: str, *args: Any, default: Any = None, **kwargs: Any) -> Any:
        fn = getattr(self._native, method, None)
        if not callable(fn):
            return default
        try:
            params = inspect.signature(fn).parameters
            if not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
                kwargs = {k: v for k, v in kwargs.items() if k in params}
        except (TypeError, ValueError):
            pass
        try:
            result = await fn(*args, **kwargs)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # the extension never raises; a fake / an old build might
            log.warning("native %s failed: %s", method, type(exc).__name__)
            return default
        return _plain(result)

    async def platform_info(self) -> dict:
        info = await self._call("get_platform_info", default={})
        return info if isinstance(info, dict) else {}

    async def pick_folder(self, initial: Any = None, op_id: Optional[str] = None) -> Optional[dict]:
        """None when cancelled; else ``{ok, target: ref, ...}`` (failure: ``error``)."""
        out = _norm_result(await self._call("pick_folder", initial=initial, op_id=op_id))
        if not out["ok"] and out["error"] == E_CANCELLED:
            return None
        return out

    async def pick_save_location(self, name: str, mime: str, source_path: Optional[str] = None,
                                 op_id: Optional[str] = None) -> Optional[dict]:
        """None when cancelled; else ``{ok, document: ref, write: result | None}``."""
        out = _norm_result(await self._call("pick_save_location", name, mime, source_path, op_id=op_id))
        if not out["ok"] and out["error"] == E_CANCELLED:
            return None
        return out

    async def pick_document(self, mime: str, op_id: Optional[str] = None) -> Optional[dict]:
        out = _norm_result(await self._call("pick_document", [mime] if mime else None, op_id=op_id))
        if not out["ok"] and out["error"] == E_CANCELLED:
            return None
        return out

    async def query_root(self, target: Any) -> dict:
        return _norm_result(await self._call("query_root", target), default_error=E_PROVIDER)

    async def list_children(self, parent: Any, names: Optional[Sequence[str]] = None) -> dict:
        out = _norm_result(await self._call("list_children", parent, names=list(names) if names else None),
                           default_error=E_PROVIDER)
        out["children"] = [c for c in (_child(item) for item in (out.get("children") or ())) if c is not None] \
            if out["ok"] else []
        out["complete"] = bool(out.get("complete", True))
        return out

    async def create_file(self, parent: Any, name: str, mime: str, on_exists: str = "fail") -> dict:
        out = _norm_result(await self._call("create_file", parent, name, mime, on_exists=on_exists),
                           default_error=E_PROVIDER)
        out["doc"] = out.get("document")
        return out

    async def create_dir(self, parent: Any, name: str, on_exists: str = "fail") -> dict:
        out = _norm_result(await self._call("create_folder", parent, name, on_exists=on_exists),
                           default_error=E_PROVIDER)
        out["doc"] = out.get("folder") or out.get("document")
        return out

    async def write_file(self, doc: Any, source_path: str, *, mode_chain: Sequence[str] = DEFAULT_MODE_CHAIN,
                         op_id: str = "") -> dict:
        out = _norm_result(await self._call("write_file", doc, source_path, mode_chain=tuple(mode_chain),
                                            verify=True, op_id=op_id or None), default_error=E_PROVIDER)
        out["doc"] = out.get("document") or doc
        out["size"] = out.get("verified_size")
        return out

    async def rename(self, doc: Any, name: str) -> dict:
        """``rename_document`` where the extension has it (Android ``DocumentsContract.renameDocument``, iOS
        coordinated move); ``unavailable`` otherwise."""
        out = _norm_result(await self._call("rename_document", doc, name), default_error=E_UNAVAILABLE)
        out["doc"] = out.get("document")
        return out

    async def stat(self, doc: Any) -> dict:
        out = _norm_result(await self._call("stat", doc), default_error=E_PROVIDER)
        out["doc"] = out.get("document") or doc
        return out

    async def delete(self, doc: Any) -> bool:
        out = _norm_result(await self._call("delete", doc), default_error=E_PROVIDER)
        return bool(out["ok"] or out["error"] == E_MISSING)

    async def release(self, target: Any) -> bool:
        return bool(await self._call("release", target, default=False))

    async def list_grants(self) -> Optional[list]:
        """Android's persisted grant URIs ([] on iOS); None when the platform cannot say."""
        result = await self._call("list_grants", default=None)
        if not isinstance(result, list):
            return None
        return [str(g.get("uri")) if isinstance(g, Mapping) else str(g) for g in result
                if (g.get("uri") if isinstance(g, Mapping) else g)]

    async def cancel(self, op_id: str) -> bool:
        return bool(await self._call("cancel_document_op", op_id, default=False))

    async def take_results(self) -> list:
        result = await self._call("take_document_results", default=[])
        return [dict(item) for item in result if isinstance(item, Mapping)] if isinstance(result, list) else []

    async def save_to_downloads(self, path: str, display_name: str, mime_type: str, subdir: str = "Glossarion",
                                replace_uri: Optional[str] = None) -> Optional[str]:
        result = await self._call("save_to_downloads", path, display_name, mime_type, subdir, replace_uri=replace_uri)
        return str(result) if result else None


# ---------------------------------------------------------------------------------------------------
# foreground service / background task hold
# ---------------------------------------------------------------------------------------------------


class Keepalive:
    """Keeps the app alive while a drain writes.

    Android: the 'cloud' holder of the bridge's ``ServiceHolds`` through ``native.ServiceLease`` (the
    sign-in's foreground-service logic, shared with share links). After a job the lease joins the job's
    service synchronously (``hold_now`` -> ``ServiceLease.join``) inside the transition callback, before
    ``BackgroundExecution.job_finished`` runs, which then keeps the service with our notification. A
    drain the user started (Send now, resume) starts the service itself while the app is in front
    (``ensure(may_start=True)``; Android 12+ forbids that from the background). ``release`` stops the
    service only when nobody else holds it and no job is active or queued, else gives the remaining
    holder its notification back. iOS: ``begin_background_task('cloud')`` for the drain plus the same
    holder, which makes the patched ``BackgroundExecution.job_finished`` keep the job's background
    grant until ``release`` calls ``release_kept_background``.
    """

    def __init__(self, native: Any = None, *, platform: str = "desktop", jobs: Any = None, background: Any = None,
                 clock: Callable[[], float] = time.monotonic, name: str = CLOUD_HOLD) -> None:
        from glossarion_mobile.services.native import ServiceLease

        self.native = native
        self.name = name  # the ServiceHolds holder
        self.platform = platform
        self.jobs = jobs
        self.background = background
        self.clock = clock
        self.lease = ServiceLease(native, name)  # Android
        self._ios_held = False
        self.bg_task = -1
        self.text = ""
        self._last_update = -1e9
        self.calls: list = []

    @property
    def held(self) -> bool:
        return self._ios_held if self.platform == "ios" else self.lease.held

    @property
    def started(self) -> bool:
        return self.lease.started

    def _holds(self) -> Any:
        return self.lease.holds()

    async def _call(self, method: str, *args: Any, default: Any = None, **kwargs: Any) -> Any:
        self.calls.append(method)
        call = getattr(self.native, "call", None)
        if call is None:
            return default
        try:
            return await call(method, *args, default=default, **kwargs)
        except Exception as exc:
            log.info("native %s failed: %s", method, exc)
            return default

    def hold_now(self, text: str) -> bool:
        """Synchronous: join a running job's service (Android) / ask to keep the job's grant (iOS)."""
        if self.platform == "android":
            if not self.lease.join(HOLD_TITLE, text, holder=JOBS_HOLD):
                return False  # no job service to join; ensure() decides
        elif self.platform == "ios":
            holds = self._holds()
            if holds is None:
                return False
            holds.hold(self.name, HOLD_TITLE, text)
            self._ios_held = True
        else:
            return False
        self.text = text
        self.calls.append("hold_now")
        return True

    async def ensure(self, text: str, *, may_start: bool) -> bool:
        """Hold for a drain; False when Android has no service and may not start one now."""
        self.text = text or self.text
        if self.platform == "android":
            holds = self._holds()
            if self.lease.held and holds is not None and self.name in holds:
                await self.update(self.text, force=True)
                return True
            self.calls.append("acquire")
            if await self.lease.acquire(HOLD_TITLE, self.text, may_start=may_start):
                if not self.lease.started:
                    await self.update(self.text, force=True)  # joined: show our text unless a job holds it
                return True
            if self.lease.failed:
                log.info("the foreground service did not start; saving runs while the app is open")
                return True
            return False
        if self.platform == "ios":
            if self.bg_task < 0:
                task = await self._call("begin_background_task", "cloud", default=-1)
                try:
                    self.bg_task = int(task)
                except (TypeError, ValueError):
                    self.bg_task = -1
            holds = self._holds()
            if holds is not None:
                holds.hold(self.name, HOLD_TITLE, self.text)
            self._ios_held = True
            return True
        return True

    async def update(self, text: str, *, force: bool = False) -> bool:
        """The notification text while we hold the service (1/s; a running job's own text wins)."""
        if not text or self.platform != "android" or not self.lease.held:
            return False
        self.text = text
        now = self.clock()
        if not force and now - self._last_update < NOTIFICATION_INTERVAL:
            self.lease.remember(text, HOLD_TITLE)  # shown when the other holders let go
            return False
        if not await self.lease.update(text, HOLD_TITLE):
            return False
        self._last_update = now
        return True

    def _job_pending(self) -> bool:
        view = getattr(self.jobs, "view", None) if self.jobs is not None else None
        if not callable(view):
            return False
        try:
            current = view()
        except Exception:
            return False
        return getattr(current, "active", None) is not None or bool(getattr(current, "queue", ()))

    async def release(self) -> None:
        if self.platform == "android":
            # A job that is active or queued owns the service again (BackgroundExecution).
            await self.lease.release(keep_service=self._job_pending)
            return
        if self.platform == "ios":
            self._ios_held = False
            holds = self._holds()
            if holds is not None:
                holds.release(self.name)
            if self.bg_task >= 0:
                task, self.bg_task = self.bg_task, -1
                await self._call("end_background_task", task)
            hook = getattr(self.background, "release_kept_background", None)
            if callable(hook):
                try:
                    result = hook()
                    if inspect.isawaitable(result):
                        await result
                except Exception:
                    log.debug("release_kept_background failed", exc_info=True)

    def drop(self) -> None:
        """The service is gone (Android timeout / destroyed): forget the hold without stopping anything."""
        self.lease.drop()
        self._ios_held = False

    def alone(self) -> bool:
        """Android: only the cloud save holds the service (its notification's Stop is ours)."""
        return self.platform == "android" and self.lease.owns_alone()

    async def repost(self) -> bool:
        """The notification was swiped away while only we hold the service: post it again."""
        holds = self._holds()
        if self.platform != "android" or not self.lease.held or (holds is not None and JOBS_HOLD in holds):
            return False
        return await self.update(self.text, force=True)


# ---------------------------------------------------------------------------------------------------
# notifications (failures and "needs a save location" only; routes, never paths)
# ---------------------------------------------------------------------------------------------------


class CloudNotifier:
    """Posts through ``JobNotifications`` (the ``jobs.action`` channel) with ids from 42000."""

    CHANNEL = "jobs.action"

    def __init__(self, notifications: Any = None, native: Any = None) -> None:
        self.notifications = notifications
        self.native = native
        self.posted: list = []

    @staticmethod
    def notification_id(key: str) -> int:
        return NOTIFY_BASE + (zlib.crc32(str(key).encode("utf-8")) % NOTIFY_SPAN)

    async def show(self, key: str, title: str, body: str, route: str) -> bool:
        nid = self.notification_id(key)
        self.posted.append({"id": nid, "title": title, "body": body, "route": route})
        del self.posted[:-20]
        show = getattr(self.notifications, "_show", None) if self.notifications is not None else None
        try:
            if callable(show):
                return bool(await show(nid, title, body, channel=self.CHANNEL, route=route))
            if self.native is not None and hasattr(self.native, "show_notification"):
                return bool(await self.native.show_notification(nid, title, body, channel_id=self.CHANNEL,
                                                                payload="glossarion://app" + route))
        except Exception as exc:
            log.info("cloud notification failed: %s", exc)
        return False

    async def cancel(self, key: str) -> None:
        native = self.native if self.native is not None else getattr(self.notifications, "native", None)
        cancel = getattr(native, "cancel_notification", None) if native is not None else None
        if callable(cancel):
            try:
                await cancel(self.notification_id(key))
            except Exception:
                pass


def _texts() -> Any:
    """The UI's notification texts (``ui/screens/cloud_sync``, u10-ui), else None (no Flet)."""
    try:
        from glossarion_mobile.ui.screens import cloud_sync as screen

        return screen
    except Exception:
        return None


def _progress_text(name: Any, dest_label: Any, written: Any = 0, total: Any = 0) -> str:
    """The foreground notification line while a copy runs (``ui/screens/cloud_sync.cloud_progress_text``)."""
    texts = _texts()
    if texts is not None and hasattr(texts, "cloud_progress_text"):
        return texts.cloud_progress_text(str(name or "books"), str(dest_label or ""), written, total)
    percent = int(100 * int(written or 0) / int(total)) if total else None
    return f"Saving {name} to {dest_label or 'the cloud'}" + (f" · {percent}%" if percent is not None else "…")


# ---------------------------------------------------------------------------------------------------
# the service
# ---------------------------------------------------------------------------------------------------


class _Relink(Exception):
    """The destination lost access as a whole (pause the queue, ask to choose it again)."""


class _RetryLater(Exception):
    """A step failed in a way worth retrying later (the code is ``args[0]``)."""


class _SourceGone(Exception):
    pass


class _SourceChanged(Exception):
    pass


class _NoSpace(Exception):
    pass


@dataclass
class _Book:
    identity: str
    key: str
    exists: bool
    state: str  # library | attachments | outside | missing
    row: dict
    folder_name: str
    title: str
    choices: dict


@dataclass
class _Snapshot:
    path: str
    size: int
    sha1: str
    mtime_ns: int


async def _to_thread(fn: Callable[..., Any], *args: Any) -> Any:
    return await asyncio.to_thread(fn, *args)


def _thread_runner(dispatcher: Any) -> Callable[..., Awaitable[Any]]:
    """``UiDispatcher.run_in_thread`` as the service's io runner (named threads in diagnostics)."""
    async def run(fn: Callable[..., Any], *args: Any) -> Any:
        return await dispatcher.run_in_thread(fn, *args, name="gl-cloud")

    return run


def _error_text(code: str) -> str:
    return {
        E_PROVIDER: "the cloud app reported an error (offline?)",
        E_TIMEOUT: "the cloud app did not answer in time",
        E_SPACE: "out of space",
        E_PERMISSION: "access was refused",
        E_MISSING: "the cloud file is gone",
        E_MODE: "the cloud app cannot overwrite files",
        E_SIZE: "the saved size did not match",
        E_READ_ONLY: "the folder does not accept files",
        E_CANCELLED: "stopped",
        E_SOURCE_CHANGED: "the book changed while it was copied",
        E_SOURCE_MISSING: "the book's file is gone",
        E_BUSY: "the cloud app is busy",
    }.get(code, code or "unknown error")


class CloudSyncService:
    """U10 cloud sync: settings, destination, triggers, the queue and its drain (UI loop + io pool)."""

    def __init__(
        self,
        *,
        docs: Any,
        store: CloudRecordStore,
        prefs: Any = None,
        library: Any = None,
        jobs: Any = None,
        files: Any = None,
        keepalive: Optional[Keepalive] = None,
        notifier: Optional[CloudNotifier] = None,
        run_io: Optional[Callable[..., Awaitable[Any]]] = None,
        spawn: Optional[Callable[[Any], Any]] = None,
        post: Optional[Callable[..., Any]] = None,
        platform: str = "desktop",
        cache_dir: str = "",
        app_roots: Sequence[str] = (),
        own_prefixes: Sequence[str] = OWN_PREFIXES,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.docs = docs
        self.store = store
        self.prefs = prefs
        self.library = library
        self.jobs = jobs
        self.files = files  # FileBridge: phone_folder_reason
        self.keepalive = keepalive or Keepalive(platform=platform)
        self.notifier = notifier or CloudNotifier()
        self._run_io = run_io or _to_thread
        self._spawn_fn = spawn
        self._post_fn = post
        self.platform = platform
        self.cache_dir = cache_dir
        self.app_roots = [str(r) for r in app_roots if r]
        self.own_prefixes = tuple(own_prefixes)
        self.clock = clock
        self.visible = True
        self._memory_settings: dict = {}
        self._listeners: list = []
        self._seen_jobs: list = []
        self._drain_task: Any = None
        self._again = False
        self._timer: Any = None
        self._closed = False
        self._stop_drain = False
        self._inflight: Optional[dict] = None
        self._held: Optional[bool] = None  # this drain pass holds the app alive (asked lazily, before a write)
        self._retire_epoch = 0  # bumped by ``_retire``: a write cancelled for a destination change only pauses
        self._leftovers_due = 0.0  # next sweep of cloud copies a replace still has to delete (0: now)
        self._leftovers_attempts = 0
        self._failures_new: list = []  # error codes of books that reached FAIL_AFTER in this drain
        #: ``install`` / ``start`` until the records are loaded: auto-migrate moves that arrive meanwhile (the
        #: chat's startup sweep runs on a worker thread) wait in ``_early_moves``, else ``load`` would drop them.
        self._loading = False
        self._early_moves: list = []
        self._early_lock = threading.Lock()
        #: Bumped by every change except write progress: the chat cards re-read their state only when it moved
        #: (``ChatFeature._u10_rebind_all``); progress repaints only the card of the book being copied.
        self.state_seq = 0
        self._notified: set = set()
        self._needs_pick_new: list = []
        self._waiting_library: set = set()
        self._job_reports: dict = {}  # book key -> True: a single-book job's report (Library-filed EPUB)
        self._seen_results: list = []  # op ids of late picker answers already handled
        self._last_progress_post = -1e9
        self.last_drain: Optional[dict] = None
        self.events: list = []  # diagnostics: what happened (no paths, no URIs)
        self._unsubs: list = [store.subscribe(self._changed)]
        try:  # the UI loop, for posts from worker threads when no dispatcher is given (tests, tools)
            self._loop: Any = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None

    # ---- plumbing --------------------------------------------------------------------------------

    async def _io(self, fn: Callable[..., Any], *args: Any) -> Any:
        return await self._run_io(fn, *args)

    def _spawn(self, coro: Any) -> Any:
        if self._spawn_fn is not None:
            try:
                return self._spawn_fn(coro)
            except Exception:
                log.exception("spawning a cloud task failed")
        try:
            return asyncio.get_running_loop().create_task(coro)
        except RuntimeError:
            coro.close()
            return None

    def _post(self, fn: Callable[..., Any], *args: Any) -> None:
        if self._post_fn is not None:
            try:
                if self._post_fn(fn, *args) is not False:
                    return
            except Exception:
                pass
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is not None:
            loop.call_soon(fn, *args)
        elif self._loop is not None and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(fn, *args)  # a worker thread (the auto-migrate)
        else:
            fn(*args)

    def _event(self, text: str) -> None:
        self.events.append(f"{time.strftime('%H:%M:%S')} {text}")
        del self.events[:-60]
        log.info("cloud sync: %s", text)

    def subscribe(self, callback: Callable[..., Any]) -> Callable[[], None]:
        """``callback()`` on the UI loop after any change (records, queue, settings, progress)."""
        self._listeners.append(callback)

        def remove() -> None:
            if callback in self._listeners:
                self._listeners.remove(callback)

        return remove

    def _notify_listeners(self) -> None:
        for callback in list(self._listeners):
            try:
                callback()
            except Exception:
                log.exception("cloud sync listener failed")

    def _changed(self) -> None:
        self.state_seq += 1
        self._post(self._notify_listeners)

    def _progress_changed(self) -> None:
        """Write progress only (``state_seq`` stays): listeners repaint, the chat cards skip a full re-read."""
        self._post(self._notify_listeners)

    def inflight_key(self) -> str:
        """The book key being copied right now ('' when none)."""
        inflight = self._inflight
        return str(inflight.get("key") or "") if inflight else ""

    # ---- settings (Prefs ``cloud_sync``) ------------------------------------------------------------

    def _raw_settings(self) -> dict:
        if self.prefs is None:
            return dict(self._memory_settings)
        try:
            raw = self.prefs.get(PREF_KEY, None)
        except Exception:
            raw = None
        return dict(raw) if isinstance(raw, Mapping) else {}

    def _write_settings(self, raw: dict) -> None:
        raw = dict(raw)
        raw["v"] = 1
        if self.prefs is None:
            self._memory_settings = raw
        else:
            self.prefs.set(PREF_KEY, raw)
        self._changed()

    def _update_settings(self, **changes: Any) -> CloudSettings:
        raw = self._raw_settings()
        raw.update(changes)
        self._write_settings(raw)
        return self.settings()

    def settings(self) -> CloudSettings:
        return CloudSettings.from_raw(self._raw_settings())

    def destination(self) -> Optional[Destination]:
        return self.settings().destination

    def _set_destination(self, dest: Optional[Destination]) -> None:
        self._update_settings(destination=dest.as_dict() if dest is not None else None)

    @property
    def supported(self) -> bool:
        """The system pickers exist here (an Android / iOS build with the extension)."""
        return bool(getattr(self.docs, "available", False))

    def set_enabled(self, value: bool) -> dict:
        """The global switch; turning it on queues every Library book (each enabled format once)."""
        before = self.settings().enabled
        self._update_settings(enabled=bool(value))
        if value and not before:
            self._spawn(self.sync_all("enabled"))
        return _answer(True, enabled=bool(value))

    def set_kind_enabled(self, kind: str, value: bool) -> dict:
        if kind not in KINDS:
            return _answer(False, f"Unknown format {kind}")
        raw = self._raw_settings()
        kinds = dict(raw.get("kinds") or {})
        kinds[kind] = bool(value)
        raw["kinds"] = kinds
        self._write_settings(raw)
        if value:
            self._spawn(self.sync_all(f"format:{kind}"))
        return _answer(True)

    def override(self, identity: Any) -> str:
        return self.store.override(identity)

    def set_book_override(self, identity: Any, value: str) -> dict:
        """Per-book Default / Always / Never; Always (or Default with sync on) queues the book."""
        if value not in OVERRIDES:
            return _answer(False, f"Unknown choice {value}")
        self.store.set_override(identity, value)
        if self.destination() is not None and self._wants(str(identity), manual=False):
            self.store.enqueue(str(identity), f"override:{value}")
            self._kick("override")
        return _answer(True, override=value)

    def _wants(self, identity: str, *, manual: bool) -> bool:
        if manual:
            return True
        override = self.store.override(identity)
        if override == "never":
            return False
        if override == "always":
            return True
        return self.settings().enabled

    def _migrate_mirror_pref(self) -> bool:
        """First U10 run on Android with the old "Mirror outputs to Downloads/Glossarion" switch on: the
        phone folder becomes the destination and sync starts on (the user had opted in already). True when
        it migrated: ``start`` then queues every Library book, as turning the switch on by hand does."""
        if self.prefs is None or self.platform != "android":
            return False
        try:
            existing = self.prefs.get(PREF_KEY, None)
            mirror = bool(self.prefs.get(files_mod.MIRROR_PREF, False))
        except Exception:
            return False
        if existing is not None or not mirror:
            return False
        dest = Destination(id=PHONE_TARGET_ID, mode=MODE_PHONE, platform=self.platform, label=_phone_label(),
                           provider="phone", provider_label="Phone", linked_at=self.clock())
        self._write_settings({"enabled": True, "kinds": {k: True for k in KINDS}, "destination": dest.as_dict()})
        self._event("phone folder destination from the mirror switch")
        return True

    # ---- lifecycle -------------------------------------------------------------------------------

    async def start(self) -> None:
        """Load the records (io), clean leftover snapshots, pick up picker answers that outlived their
        call, then drain what waits (app start)."""
        self._loop = asyncio.get_running_loop()
        with self._early_lock:
            self._loading = True
        try:
            await self._io(self.store.load)
        finally:
            with self._early_lock:
                self._loading = False
                moves, self._early_moves = self._early_moves, []
        for old, new, merged in moves:  # chat books the startup sweep moved while the records were loading
            self.on_workspace_moved(old, new, merged=merged)
        await self._io(self._clean_snapshots_blocking)
        migrated = self._migrate_mirror_pref()
        pending = self.settings().pending_pick
        await self._take_late_results()
        if self.settings().pending_pick and pending:
            # The process died while the system picker was open and no answer survived: the native side
            # reports it cancelled; a per-file record keeps asking for its save location.
            self._event(f"picker interrupted ({pending.get('op')})")
            self._update_settings(pending_pick=None)
        if migrated:
            self._sync_all_when_ready("phone folder (mirror switch)")
        self._kick("start")

    def attach(self, *, jobs: Any = None, native: Any = None) -> None:
        """Subscribe to job transitions and the bridge's foreground-service / document events."""
        service = jobs if jobs is not None else self.jobs
        on_transition = getattr(service, "on_transition", None) if service is not None else None
        if callable(on_transition):
            self._unsubs.append(on_transition(self.on_job_transition))
        if native is not None and hasattr(native, "add_listener"):
            for event, handler in (("foreground", self.on_foreground_event), ("document", self.on_document_event)):
                try:
                    native.add_listener(event, handler)
                except Exception:  # an older bridge without the 'document' event (Integrate adds it)
                    log.debug("native %s listener not added", event, exc_info=True)

    def close(self) -> None:
        self._closed = True
        if self._timer is not None:
            try:
                self._timer.cancel()
            except Exception:
                pass
            self._timer = None
        for unsub in self._unsubs:
            try:
                unsub()
            except Exception:
                pass
        self._unsubs = []
        self.store.close()

    def on_lifecycle(self, name: str) -> None:
        """App lifecycle (UI loop): hidden -> Android drains only under a foreground service; resume ->
        drain what waits."""
        self.visible = name not in ("hide", "pause", "inactive", "detach")
        if name == "resume":
            self._kick("resume")

    async def wait_idle(self) -> None:
        """Until no drain runs (tests, shutdown)."""
        while self._drain_task is not None and not self._drain_task.done():
            try:
                await asyncio.shield(self._drain_task)
            except Exception:
                pass

    async def drain_now(self, reason: str = "manual") -> Optional[dict]:
        """Start a drain (or let the running one go round again) and wait for it."""
        self._kick(reason)
        await self.wait_idle()
        return self.last_drain

    def retry_now(self) -> dict:
        """Settings "Retry now": every waiting / failed book is due now (attempts kept)."""
        count = self.store.wake(all_entries=True)
        self._kick("retry")
        return _answer(True, f"Retrying {count} book{'s' if count != 1 else ''}" if count else "Nothing is waiting")

    # ---- triggers ----------------------------------------------------------------------------------

    def _job_books(self, snap: Any) -> list:
        """``[(identity, reported paths, single)]`` of the Library books a job may have changed."""
        spec = getattr(snap, "spec", None)
        if str(getattr(spec, "kind", "") or "") not in TRIGGER_KINDS:
            return []
        # A chat translation (direct_text) is not a trigger: its book is uploaded once the auto-migrate moved
        # it into the Library (on_workspace_moved). A chat card's Compile of a book already in the Library is;
        # one of a workspace still in Attachments is dropped by the drain (it waits for the Library).
        origin = getattr(spec, "origin", None) or {}
        params = getattr(spec, "params", None) or {}
        folders: list = []

        def add(value: Any) -> None:
            if isinstance(value, str) and value and os.path.isabs(value):
                if _norm(value) not in {_norm(f) for f in folders}:
                    folders.append(os.path.abspath(value))

        add(params.get("folder"))
        for name in ("folders", "output_roots"):
            values = params.get(name)
            if isinstance(values, Mapping) or isinstance(values, (str, bytes)):
                # Tools › Headers & metadata maps {raw source: output root}: neither side is a book workspace
                # (iterating it would take the raw EPUB as a book); its bid / output_dirs name the books
                continue
            for value in values or ():
                add(value)
        for value in (getattr(snap, "output_dirs", None) or {}).values():
            add(value)
        add(getattr(snap, "output_dir", None))
        bid = origin.get("bid") if isinstance(origin, Mapping) and origin.get("type") == "library" else None
        if bid and self.library is not None and hasattr(self.library, "identity_for_bid"):
            try:
                add(self.library.identity_for_bid(str(bid)))
            except Exception:
                pass
        # never a job's input file (the untranslated source a Tools job reads) nor anything in Library/Raw
        inputs = {_norm(p) for p in (getattr(spec, "inputs", None) or ()) if isinstance(p, str) and p}
        folders = [f for f in folders if not self._is_raw_source(f)
                   and not (_norm(f) in inputs and os.path.isfile(f))]
        reported = [os.fspath(p) for p in (getattr(snap, "outputs", None) or ()) if p]
        single = len(folders) == 1
        return [(folder, reported, single) for folder in folders]

    def _is_raw_source(self, path: str) -> bool:
        """``path`` lies in the Library's Raw folder (untranslated sources never leave the phone)."""
        library = self.library
        if library is None or not path:
            return False
        try:
            root = library.library_root()
        except Exception:
            return False
        return bool(root) and _under(path, os.path.join(str(root), "Raw"))

    def on_job_transition(self, snap: Any, previous: Any = None) -> None:
        """JobService transition (UI loop, synchronous): a Library job reached DONE -> its books are queued
        and, on Android, the 'cloud' holder joins the job's foreground service *now*, before
        ``BackgroundExecution.job_finished`` (spawned) could stop it. Any job end wakes the books the
        busy guard held back."""
        state = getattr(getattr(snap, "state", None), "value", getattr(snap, "state", None))
        prev = getattr(previous, "value", previous)
        if str(state) not in ("DONE", "FAILED", "CANCELLED", "INTERRUPTED"):
            return
        woken = self.store.wake(errors=(BUSY,))
        if str(state) != "DONE" or previous is None or str(prev) == "QUEUED":
            if woken:
                self._kick("job ended")
            return
        job_id = str(getattr(snap, "id", "") or "")
        if job_id in self._seen_jobs:
            return
        self._seen_jobs.append(job_id)
        del self._seen_jobs[:-500]
        dest = self.destination()
        books = self._job_books(snap) if dest is not None else []
        wanted = [b for b in books if self._wants(b[0], manual=False)]
        if wanted and dest is not None and dest.usable:
            title = os.path.basename(os.path.normpath(wanted[0][0]))
            self.keepalive.hold_now(_progress_text(title, dest.display))
        for identity, reported, single in wanted:
            entry = self.store.enqueue(identity, f"job:{job_id}", reported=reported)
            if single:
                self._job_reports[entry["key"]] = True
        if wanted or woken:
            self._kick("job")

    def on_workspace_moved(self, old: str, new: str, *, merged: bool = False) -> None:
        """Thread-safe (the chat's auto-migrate worker, inside its migrate lock): a chat book moved or was
        merged into the Library. The records follow under the store lock (a merge keeps the Library
        book's documents; critic #10); then the Library book is queued when sync wants it. A move that
        arrives while ``start`` loads the records waits for it (the chat's startup sweep can run first)."""
        with self._early_lock:
            if self._loading:
                self._early_moves.append((str(old), str(new), bool(merged)))
                return
        dropped = self.store.relocate(old, new, merge=merged)
        self._waiting_library.discard(_norm(old))
        self._post(self._after_move, str(new), dropped)

    def _after_move(self, new: str, dropped: list) -> None:
        if dropped:
            self._spawn(self._release_docs(dropped))
        if self.destination() is not None and self._wants(new, manual=False):
            self.store.enqueue(new, "library:moved")
            self._kick("moved")

    def on_migrate_outcome(self, outcome: Mapping[str, Any]) -> None:
        """An auto-migrate outcome (``ui/chat/integration.MIGRATE_*``): DEFERRED / COLLISION leave the book
        waiting for the Library ("Will save when the book moves to the Library")."""
        status = str(outcome.get("status") or "")
        folder = str(outcome.get("folder") or "")
        if status in ("deferred", "collision") and folder:
            self._waiting_library.add(_norm(folder))
            self._changed()

    async def send_now(self, identity: str, kinds: Optional[Iterable[str]] = None) -> dict:
        """The Book page's "Send now" (also past a per-book Never and with the switch off)."""
        dest = self.destination()
        if dest is None:
            return _answer(False, NO_DESTINATION_REASON)
        if dest.needs_relink:
            return _answer(False, f"Glossarion lost access to {dest.display}; choose it again in Settings")
        state = await self._io(self._library_state_blocking, str(identity))
        if state == "attachments":
            return _answer(False, NOT_IN_LIBRARY_REASON)
        if state == "missing":
            return _answer(False, "The book's files are gone")
        if state != "library":
            return _answer(False, "Only Library books are saved to the cloud")
        self.store.enqueue(str(identity), "manual", manual=True, kinds=kinds)
        self._kick("send now")
        return _answer(True, f"Saving to {dest.display}…")

    def _sync_all_when_ready(self, reason: str) -> None:
        """``sync_all`` once the Library has published its books (app start: its first scan may still run; a
        scan is never started from here, the Library's own one is waited for)."""
        if self._library_books():
            self._spawn(self.sync_all(reason, refresh=False))
            return
        subscribe = getattr(self.library, "subscribe", None) if self.library is not None else None
        if not callable(subscribe):
            return
        holder: dict = {}

        def on_snapshot(snapshot: Any) -> None:
            try:
                books = list(snapshot.all_books()) if hasattr(snapshot, "all_books") else []
            except Exception:
                books = []
            if not books or holder.get("done"):
                return
            holder["done"] = True
            unsubscribe = holder.pop("unsubscribe", None)
            if callable(unsubscribe):
                unsubscribe()
            self._spawn(self.sync_all(reason, refresh=False))

        holder["unsubscribe"] = subscribe(on_snapshot)

    async def sync_all(self, reason: str = "all", *, refresh: bool = True) -> int:
        """Queue every Library book sync wants (switch turned on, a format turned on, a destination
        linked). Returns how many were queued; books without outputs are dropped by the drain."""
        if self.destination() is None or self.library is None:
            return 0
        books = self._library_books()
        if not books and refresh and hasattr(self.library, "refresh"):
            try:
                await self.library.refresh(quiet=True, reason="cloud sync")
            except Exception:
                log.debug("library refresh for cloud sync failed", exc_info=True)
            books = self._library_books()
        count = 0
        for row in books:
            identity = self._identity_of(row)
            if not identity or not self._wants(identity, manual=False):
                continue
            self.store.enqueue(identity, reason)
            count += 1
        if count:
            self._kick(reason)
        return count

    def _library_books(self) -> list:
        snapshot = getattr(self.library, "snapshot", None) if self.library is not None else None
        try:
            return list(snapshot.all_books()) if snapshot is not None and hasattr(snapshot, "all_books") else []
        except Exception:
            return []

    @staticmethod
    def _identity_of(row: Mapping[str, Any]) -> str:
        from glossarion_mobile.services.library import book_identity

        return book_identity(row)

    # ---- book removal / wipe -----------------------------------------------------------------------

    async def forget_book(self, identity: str) -> int:
        """A Library book was deleted: drop its records and queue entry (the cloud files stay)."""
        dropped = self.store.forget_book(identity)
        await self._release_docs(dropped)
        self._waiting_library.discard(_norm(identity))
        return len(dropped)

    def forget_books_threadsafe(self, identities: Iterable[str]) -> None:
        """``forget_book`` from a worker thread (``LibraryService.execute_delete_blocking``)."""
        dropped: list = []
        for identity in identities or ():
            dropped.extend(self.store.forget_book(identity))
            self._waiting_library.discard(_norm(identity))
        if dropped:
            self._post(lambda: self._spawn(self._release_docs(dropped)))

    async def _release_docs(self, pairs: Iterable[tuple]) -> None:
        """Release per-file grants (Android); folder documents live under the folder's one grant."""
        for target_id, doc in pairs or ():
            if target_id == FILES_TARGET_ID and doc and self.platform == "android":
                try:
                    await self.docs.release(doc)
                except Exception:
                    pass

    async def wipe(self) -> int:
        """Danger zone "Wipe app data", before the files go: release every persisted grant (they live in
        the system, not in app files, and count toward Android's 128/512 cap), then forget everything."""
        released = 0
        if self._inflight is not None:
            await self._cancel_inflight()
        try:
            grants = await self.docs.list_grants()
        except Exception:
            grants = None
        targets: list = list(grants or [])
        dest = self.destination()
        if dest is not None and dest.mode == MODE_FOLDER and dest.target:
            targets.append(dest.target)
        if dest is not None and dest.mode == MODE_FILES:
            targets.extend(self.store.docs(dest.id))
        for target in targets:
            try:
                if await self.docs.release(target):
                    released += 1
            except Exception:
                pass
        self._update_settings(enabled=False, destination=None, pending_pick=None)
        await self._io(self.store.delete_file)
        await self._io(self._clean_snapshots_blocking)
        self._event(f"wiped ({released} grant(s) released)")
        return released

    # ---- destination flows (user taps) -----------------------------------------------------------

    def _set_pending_pick(self, op: str, **extra: Any) -> str:
        op_id = uuid.uuid4().hex
        self._update_settings(pending_pick={"op": op, "op_id": op_id, "at": self.clock(), **extra})
        return op_id

    def _clear_pending_pick(self) -> None:
        if self.settings().pending_pick:
            self._update_settings(pending_pick=None)

    async def pick_folder(self) -> dict:
        """Settings "Choose folder…" / "Change…" / "Choose again": the system folder picker."""
        if not self.supported:
            return _answer(False, PHONE_ONLY_REASON)
        current = self.destination()
        op_id = self._set_pending_pick("folder")
        try:
            result = await self.docs.pick_folder(current.target if current is not None and current.mode == MODE_FOLDER
                                                 else None, op_id)
        finally:
            self._clear_pending_pick()
        self._seen_results.append(op_id)
        return await self._finish_folder_pick(result)

    async def _finish_folder_pick(self, result: Optional[Mapping[str, Any]]) -> dict:
        if result is None:
            return _answer(False, "", cancelled=True)
        if not result.get("ok"):
            code = error_code(result)
            return _answer(False, "The picker is already open" if code == E_BUSY else
                           f"The folder could not be linked ({_error_text(code)})")
        target = result.get("target")
        if not target:
            return _answer(False, "The folder could not be linked")
        if is_own_location(result, self.app_roots, self.own_prefixes):
            await self._release_target(target)
            self._event("refused Glossarion's own folder")
            return _answer(False, OWN_FOLDER_REASON)
        ref = target if isinstance(target, Mapping) else {}
        if ref.get("can_write") is False:
            await self._release_target(target)
            return _answer(False, READ_ONLY_REASON)
        dest = Destination(
            id=folder_target_id(self.platform, target), mode=MODE_FOLDER, platform=self.platform, target=target,
            label=_ref_name(target) or "Cloud folder", provider=str(ref.get("provider") or ""),
            provider_label=str(ref.get("provider_label") or ""), can_create=ref.get("can_create") is not False,
            can_write=True, linked_at=self.clock(),
            persisted=bool(result.get("persisted", ref.get("persisted", True))))
        old = self.destination()
        relinked = False
        if old is not None and old.id == dest.id:
            dest = replace(dest, layout=old.layout)  # the same folder again: keep its records and layout
            relinked = bool(old.needs_relink)
        elif old is not None:
            await self._retire(old)
        self._set_destination(dest)
        self._notified.discard(("relink", dest.id))
        await self.notifier.cancel(f"relink:{dest.id}")
        self._event("folder linked" + (" again" if relinked else ""))
        notes = []
        if not dest.can_create:
            notes.append("This folder cannot take new files: choose 'Save each file separately' or another folder")
        if not dest.persisted:
            notes.append("The system did not keep the permission: you may have to choose the folder again after a "
                         "restart")
        if relinked:
            self._kick("relinked")
        await self.sync_all("linked")
        return _answer(True, " ".join(notes), destination=dest.ui())

    async def use_save_locations(self) -> dict:
        """"Save each file separately": every new output asks once for its place."""
        if not self.supported:
            return _answer(False, PHONE_ONLY_REASON)
        old = self.destination()
        if old is not None and old.mode == MODE_FILES:
            return _answer(True, destination=old.ui())
        if old is not None:
            await self._retire(old)
        dest = Destination(id=FILES_TARGET_ID, mode=MODE_FILES, platform=self.platform, label="",
                           provider="files", linked_at=self.clock())
        self._set_destination(dest)
        await self.sync_all("files mode")
        return _answer(True, destination=dest.ui())

    async def use_phone_folder(self) -> dict:
        """Android 10+: ``Downloads/Glossarion/<book>`` (MediaStore), overwritten in place."""
        reason = await self._phone_folder_reason()
        if reason:
            return _answer(False, reason)
        old = self.destination()
        if old is not None and old.mode == MODE_PHONE:
            return _answer(True, destination=old.ui())
        if old is not None:
            await self._retire(old)
        dest = Destination(id=PHONE_TARGET_ID, mode=MODE_PHONE, platform=self.platform, label=_phone_label(),
                           provider="phone", provider_label="Phone", linked_at=self.clock())
        self._set_destination(dest)
        await self.sync_all("phone folder")
        return _answer(True, destination=dest.ui())

    async def _phone_folder_reason(self) -> str:
        """Why the phone folder cannot be the destination here ('' when it can): ``FileBridge
        .phone_folder_reason`` (Android 10+, the native service present), else the same facts."""
        check = getattr(self.files, "phone_folder_reason", None) if self.files is not None else None
        if callable(check):
            try:
                return str(await check() or "")
            except Exception:
                log.debug("phone_folder_reason failed", exc_info=True)
        if self.platform != "android":
            return files_mod.PHONE_FOLDER_ONLY_ANDROID
        if not self.supported:
            return files_mod.PHONE_FOLDER_NOT_IN_BUILD
        info = await self.docs.platform_info() if hasattr(self.docs, "platform_info") else {}
        if isinstance(info, Mapping) and not info.get("save_to_downloads", True):
            return files_mod.PHONE_FOLDER_NEEDS_ANDROID_10
        return ""

    async def forget_destination(self) -> dict:
        """"Disconnect": release the grants, forget the records and the queue; cloud files stay."""
        dest = self.destination()
        if dest is not None:
            await self._retire(dest)
        self._set_destination(None)
        self.store.clear_queue()
        self._event("destination forgotten")
        return _answer(True, "Files already saved stay where they are")

    async def _release_target(self, target: Any) -> None:
        if target and self.platform == "android":
            try:
                await self.docs.release(target)
            except Exception:
                pass

    async def _retire(self, dest: Destination) -> None:
        """A destination being replaced or disconnected: its grants are released (Android) and its records
        dropped, so nothing writes to the old place while the UI shows the new one (critic #8). A write it
        cancels only pauses the drain (``_retire_epoch``), which goes round again for the new destination."""
        self._retire_epoch += 1
        if self._inflight is not None:
            await self._cancel_inflight()
        docs = self.store.drop_target(dest.id)
        if dest.mode == MODE_FOLDER:
            await self._release_target(dest.target)
        elif dest.mode == MODE_FILES:
            for doc in docs:
                await self._release_target(doc)
        await self.notifier.cancel(f"relink:{dest.id}")

    async def test_destination(self) -> dict:
        """Settings "Test": create, write and delete a small file in the folder."""
        dest = self.destination()
        if dest is None:
            return _answer(False, NO_DESTINATION_REASON)
        if dest.mode != MODE_FOLDER:
            return _answer(True, "Nothing to test for this destination")
        path = await self._io(self._write_probe_blocking)
        try:
            created = await self.docs.create_file(dest.target, "Glossarion test.txt", "text/plain", on_exists="rename")
            if not created["ok"] or not created.get("doc"):
                return _answer(False, f"The folder refused a new file ({_error_text(created['error'])})")
            written = await self.docs.write_file(created["doc"], path, op_id=uuid.uuid4().hex)
            await self.docs.delete(created["doc"])
            if not written["ok"]:
                return _answer(False, f"The folder refused the write ({_error_text(written['error'])})")
            return _answer(True, "The folder accepts files")
        finally:
            await self._io(self._remove_blocking, path)

    async def choose_save_location(self, identity: str, kind: str) -> dict:
        """Per-file mode: the system Save dialog for one output, written at once (Android) / exported once
        (iOS). The grant count is checked first (Android keeps at most 128 / 512 per app)."""
        dest = self.destination()
        if dest is None or dest.mode != MODE_FILES:
            return _answer(False, "Only when saving each file separately")
        book = await self._io(self._book_blocking, str(identity), [], False)
        choice = book.choices.get(kind)
        if choice is None:
            return _answer(False, f"No {KIND_LABELS.get(kind, kind)} to save yet")
        if self.platform == "android":
            grants = await self.docs.list_grants()
            info = await self.docs.platform_info() if hasattr(self.docs, "platform_info") else {}
            info = info if isinstance(info, Mapping) else {}
            limit = int(info.get("persisted_grant_limit") or 0) or (
                GRANT_LIMIT if int(info.get("sdk_int") or 30) >= 30 else GRANT_LIMIT_OLD)
            if grants is not None and len(grants) >= limit - GRANT_MARGIN:
                return _answer(False, "Too many separately saved files: choose a folder instead")
        try:
            snap = await self._io(self._snapshot_blocking, choice.path)
        except (_SourceGone, _SourceChanged, _NoSpace) as exc:
            return _answer(False, _error_text({_SourceGone: E_SOURCE_MISSING, _SourceChanged: E_SOURCE_CHANGED,
                                               _NoSpace: E_SPACE}[type(exc)]))
        try:
            op_id = self._set_pending_pick("save", key=book.key, kind=kind, identity=book.identity)
            try:
                result = await self.docs.pick_save_location(choice.name, _mime(choice.path), snap.path, op_id)
            finally:
                self._clear_pending_pick()
            self._seen_results.append(op_id)
            return await self._finish_save_pick(dest, book, kind, choice, snap, result)
        finally:
            await self._io(self._remove_blocking, snap.path)

    async def _finish_save_pick(self, dest: Destination, book: _Book, kind: str, choice: Optional[OutputChoice],
                                snap: Optional[_Snapshot], result: Optional[Mapping[str, Any]]) -> dict:
        if result is None:
            return _answer(False, "", cancelled=True)
        doc = result.get("document") or result.get("doc")
        if not result.get("ok") or not doc:
            return _answer(False, f"The file could not be saved there ({_error_text(error_code(result))})")
        write = result.get("write") if isinstance(result.get("write"), Mapping) else None
        if is_own_location({"doc": doc, **dict(result)}, self.app_roots, self.own_prefixes):
            if self.platform == "android":
                # The Save dialog made a new document there (the native side skips the write for its own
                # folder; an older build wrote it first): take it back out, then give the grant back. iOS keeps
                # the exported copy: a "Replace" there may have overwritten one of the app's own files.
                await self.docs.delete(doc)
            await self._release_target(doc)
            self._event("refused a save location in Glossarion's own folder")
            return _answer(False, OWN_FOLDER_REASON)
        old_doc = (self.store.record(dest.id, book.key, kind) or {}).get("doc")
        fields = {"doc": doc, "name": _ref_name(doc) or (choice.name if choice else ""), "parent": "",
                  "status": "pending", "error": "", "note": "", "dirty": True}
        written = bool(write and write.get("ok"))
        verified = write.get("verified_size") if write else None
        if written and snap is not None and choice is not None and (verified is None or verified == snap.size):
            fields.update(status="ok", dirty=False, size=choice.size, mtime_ns=choice.mtime_ns, sha1=snap.sha1,
                          source=choice.path, synced_at=self.clock(), remote_size=verified or snap.size,
                          mode=str(write.get("mode") or "export"))
        self.store.update_record(dest.id, book.key, kind, book.identity, **fields)
        if old_doc and not _same(old_doc, doc):
            await self._release_target(old_doc)  # the record no longer points at it: its grant goes back
        if fields["status"] != "ok":
            self.store.enqueue(book.identity, "save location", manual=True, kinds=[kind])
            self._kick("save location")
        await self.notifier.cancel("needs_pick")
        return _answer(True, f"Saved to {fields['name'] or 'the chosen place'}")

    async def choose_existing_file(self, identity: str, kind: str) -> dict:
        """Per-file mode "Choose cloud file…": keep updating a file that is already in the cloud."""
        dest = self.destination()
        if dest is None or dest.mode != MODE_FILES:
            return _answer(False, "Only when saving each file separately")
        op_id = self._set_pending_pick("document", key=_norm(identity), kind=kind, identity=str(identity))
        try:
            result = await self.docs.pick_document(_mime(f"x.{kind}"), op_id)
        finally:
            self._clear_pending_pick()
        self._seen_results.append(op_id)
        if result is None:
            return _answer(False, "", cancelled=True)
        doc = result.get("document") or result.get("doc")
        if not result.get("ok") or not doc:
            return _answer(False, "That file cannot be used")
        if is_own_location({"doc": doc, **dict(result)}, self.app_roots, self.own_prefixes):
            await self._release_target(doc)
            return _answer(False, OWN_FOLDER_REASON)
        old_doc = (self.store.record(dest.id, _norm(identity), kind) or {}).get("doc")
        self.store.update_record(dest.id, _norm(identity), kind, identity, doc=doc, name=_ref_name(doc), parent="",
                                 status="pending", error="", note="", dirty=True)
        if old_doc and not _same(old_doc, doc):
            await self._release_target(old_doc)  # the record no longer points at it: its grant goes back
        self.store.enqueue(str(identity), "save location", manual=True, kinds=[kind])
        self._kick("existing file")
        return _answer(True)

    async def _take_late_results(self) -> None:
        """Picker answers delivered after their call was gone (activity / process recreated)."""
        take = getattr(self.docs, "take_results", None)
        if not callable(take):
            return
        try:
            items = await take()
        except Exception:
            return
        for item in items:
            await self._late_result(item)

    async def _late_result(self, item: Mapping[str, Any]) -> None:
        op_id = str(item.get("op_id") or "")
        if op_id and op_id in self._seen_results:
            return
        if op_id:
            self._seen_results.append(op_id)
            del self._seen_results[:-50]
        pending = self.settings().pending_pick or {}
        result = item.get("result") if isinstance(item.get("result"), Mapping) else item
        status = str(item.get("status") or ("ok" if result.get("ok") else "cancelled"))
        if pending and pending.get("op_id") and op_id and pending.get("op_id") != op_id:
            return
        self._clear_pending_pick()
        if status != "ok":
            return
        kind = str(item.get("kind") or pending.get("op") or "")
        if kind == "folder":
            await self._finish_folder_pick(_norm_result(result))
            return
        dest = self.destination()
        identity = str(pending.get("identity") or "")
        out_kind = str(pending.get("kind") or "")
        if dest is None or dest.mode != MODE_FILES or not identity or out_kind not in KINDS:
            return
        book = await self._io(self._book_blocking, identity, [], False)
        await self._finish_save_pick(dest, book, out_kind, book.choices.get(out_kind), None, _norm_result(result))

    # ---- native events -------------------------------------------------------------------------------

    async def on_foreground_event(self, event: Mapping[str, Any]) -> None:
        kind = str((event or {}).get("type") or "")
        if kind == "dismissed":
            await self.keepalive.repost()
            return
        if kind == "button" and str((event or {}).get("button_id") or "") == "stop":
            # The notification's Stop while only the cloud save holds the service (a running job owns Stop
            # otherwise): stop writing; the queue stays and drains on the next resume / job.
            if self.keepalive.alone() and self._drain_task is not None and not self._drain_task.done():
                self._stop_drain = True
                await self._cancel_inflight()
                self._event("Stop pressed in the notification: drain stopped")
            return
        if kind in ("timeout", "destroyed") and self.keepalive.held:
            # Android 15 dataSync limit / the app swiped away: stop writing promptly, keep the queue.
            self._stop_drain = True
            await self._cancel_inflight()
            self.keepalive.drop()
            self._event(f"foreground service {kind}: drain stopped")

    async def on_document_event(self, event: Mapping[str, Any]) -> None:
        """``GlossarionNative.on_document``: write progress, or a picker answer that arrived late."""
        kind = str((event or {}).get("type") or "")
        if kind == "progress":
            self.on_native_progress(event)
        elif kind == "pick_result":
            await self._late_result(event)

    async def _cancel_inflight(self) -> None:
        inflight = self._inflight
        if inflight and inflight.get("op_id"):
            try:
                await self.docs.cancel(inflight["op_id"])
            except Exception:
                pass

    def on_native_progress(self, payload: Mapping[str, Any]) -> None:
        """Write progress ``{op_id, written, total}`` -> the row and the notification (throttled)."""
        inflight = self._inflight
        if not inflight or str(payload.get("op_id") or "") != inflight.get("op_id"):
            return
        try:
            inflight["written"] = int(payload.get("written") or 0)
            inflight["total"] = int(payload.get("total") or inflight.get("total") or 0)
        except (TypeError, ValueError):
            return
        now = time.monotonic()
        if now - self._last_progress_post >= PROGRESS_INTERVAL:
            self._last_progress_post = now
            self._progress_changed()
        if inflight["total"]:
            self._spawn(self.keepalive.update(_progress_text(inflight.get("name"), inflight.get("dest"),
                                                             inflight["written"], inflight["total"])))

    # ---- state for the UI (in memory: safe on the loop and on io) ----------------------------------

    def file_entries(self, identity: str) -> dict:
        """``{kind: entry}`` for the Output tab rows / chat card (``ui/screens/cloud_sync.file_status_line``:
        ``status`` ok · writing · pending · waiting · failed · check · needs_pick · needs_relink · missing ·
        source_missing · off · format_off · new, plus ``synced_at``, ``written`` / ``total``, ``reason``,
        ``error``, ``warning``, ``name``). Kinds with nothing to say are left out."""
        settings = self.settings()
        dest = settings.destination
        if dest is None:
            return {}
        key = self.store.resolve(_norm(identity))
        override = self.store.override(identity)
        queued = self.store.queued(identity)
        wants = self._wants(identity, manual=False)
        inflight = self._inflight
        out: dict = {}
        for kind in KINDS:
            record = self.store.record(dest.id, key, kind) or {}
            # ``source``: the local output the record copies (the Output tab puts the line on that row)
            base = {"name": str(record.get("name") or ""), "synced_at": float(record.get("synced_at") or 0.0),
                    "source": str(record.get("source") or "")}
            status = str(record.get("status") or "")
            if inflight and inflight.get("key") == key and inflight.get("kind") == kind:
                out[kind] = dict(base, status="writing", written=inflight.get("written") or 0,
                                 total=inflight.get("total") or 0)
            elif dest.needs_relink:
                out[kind] = dict(base, status="needs_relink")
            elif not settings.kind_on(kind):
                if record:
                    out[kind] = dict(base, status="format_off")
            elif status == "needs_pick":
                out[kind] = dict(base, status="needs_pick", warning=str(record.get("note") or ""))
            elif key in self._waiting_library:
                out[kind] = dict(base, status="waiting", reason="not_in_library")
            elif queued is not None and (queued.get("kinds") is None or kind in (queued.get("kinds") or ())):
                if int(queued.get("attempts") or 0) >= FAIL_AFTER:
                    out[kind] = dict(base, status="failed", error=str(record.get("error") or
                                                                       _error_text(queued.get("last_error") or "")))
                else:
                    out[kind] = dict(base, status="waiting", reason=str(queued.get("last_error") or
                                                                        queued.get("reason") or ""))
            elif status == "ok":
                out[kind] = dict(base, status="ok", warning=str(record.get("note") or ""))
            elif status in ("check", "missing", "source_missing"):
                out[kind] = dict(base, status=status, error=str(record.get("error") or ""))
            elif status in ("failed", "pending", "writing") and record.get("error"):
                out[kind] = dict(base, status="failed", error=str(record.get("error") or ""))
            elif override == "never":
                out[kind] = dict(base, status="off")
            elif wants and record:
                out[kind] = dict(base, status="new")
        return out

    def book_state(self, identity: str) -> dict:
        """The Book page / chat card state (``CloudFacade.book``); call it through the io runner."""
        identity = str(identity or "")
        state = self._library_state_blocking(identity) if identity else "missing"
        waiting = state == "attachments" or self.store.resolve(_norm(identity)) in self._waiting_library
        return {"override": self.store.override(identity), "auto": self._wants(identity, manual=False),
                "files": self.file_entries(identity), "in_library": state == "library", "waiting_library": waiting}

    def ui_state(self) -> dict:
        """Settings › Cloud sync & sharing (``CloudFacade.snapshot``); call it through the io runner."""
        settings = self.settings()
        dest = settings.destination
        queue = []
        failed = 0
        for entry in self.store.queue():
            is_failed = int(entry.get("attempts") or 0) >= FAIL_AFTER
            failed += 1 if is_failed else 0
            title = os.path.splitext(os.path.basename(os.path.normpath(entry["identity"])))[0] or "Book"
            queue.append({"name": title, "title": title, "kind": None, "status": "failed" if is_failed else "waiting",
                          "reason": entry.get("last_error") or entry.get("reason") or "",
                          "error": _error_text(entry["last_error"]) if entry.get("last_error") else "",
                          "at": entry.get("next_at"), "bid": self._bid(entry["identity"]),
                          "attempts": entry.get("attempts", 0)})
        recent: list = []
        last = 0.0
        needs_pick = 0
        if dest is not None:
            for key, entry in self.store.books(dest.id).items():
                title = str(entry.get("title") or "") or \
                    os.path.splitext(os.path.basename(os.path.normpath(str(entry.get("identity") or key))))[0]
                for kind, record in (entry.get("files") or {}).items():
                    status = str(record.get("status") or "")
                    needs_pick += 1 if status == "needs_pick" else 0
                    synced = float(record.get("synced_at") or 0.0)
                    last = max(last, synced)
                    if synced:
                        recent.append({"name": str(record.get("name") or ""), "title": title, "kind": kind,
                                       "status": status, "at": synced, "error": str(record.get("error") or ""),
                                       "warning": str(record.get("note") or ""),
                                       "bid": self._bid(str(entry.get("identity") or ""))})
        recent.sort(key=lambda item: item["at"], reverse=True)
        inflight = self._inflight
        return {
            "supported": self.supported, "reason": None if self.supported else PHONE_ONLY_REASON,
            "enabled": settings.enabled, "kinds": dict(settings.kinds),
            "destination": dest.ui() if dest is not None else None,
            "queue": queue, "recent": recent[:RECENT_LIMIT], "last_saved_at": last or None, "failed": failed,
            "needs_pick": needs_pick,
            "progress": ({"name": inflight.get("name"), "written": inflight.get("written") or 0,
                          "total": inflight.get("total") or 0} if inflight else None),
            "phone_folder": self.platform == "android" and self.supported,
        }

    def _bid(self, identity: str) -> Optional[str]:
        library = self.library
        if library is None or not identity or not hasattr(library, "bid_for"):
            return None
        try:
            return library.bid_for(self._book_row(identity))
        except Exception:
            return None

    # ---- drain -----------------------------------------------------------------------------------------

    def _kick(self, reason: str = "") -> None:
        if self._closed:
            return
        task = self._drain_task
        if task is not None and not task.done():
            self._again = True
            return
        self._drain_task = self._spawn(self._drain(reason))

    def _schedule_timer(self, *, cancel_only: bool = False) -> None:
        if self._timer is not None:
            try:
                self._timer.cancel()
            except Exception:
                pass
            self._timer = None
        dest = self.destination()
        if cancel_only or self._closed or dest is None or not dest.usable:
            return
        next_at = self.store.next_due_at()
        if self._leftovers_due and dest.mode == MODE_FOLDER and self.store.leftovers(dest.id):
            next_at = self._leftovers_due if next_at is None else min(next_at, self._leftovers_due)
        if next_at is None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._timer = loop.call_later(max(0.5, next_at - self.clock()), self._kick, "timer")

    async def _drain(self, reason: str) -> None:
        saved = failed = 0
        blocked = False
        self._stop_drain = False
        try:
            while True:
                self._again = False
                result = await self._drain_pass()
                saved += result["saved"]
                failed += result["failed"]
                blocked = bool(result.get("blocked")) or self._stop_drain
                if not self._again or blocked:
                    break
        except Exception:
            log.exception("cloud sync drain failed")
        finally:
            self._inflight = None
            try:
                await self.keepalive.release()
            except Exception:
                log.debug("releasing the cloud hold failed", exc_info=True)
            await self._post_needs_pick()
            await self._post_failures()
            self.last_drain = {"at": self.clock(), "reason": reason, "saved": saved, "failed": failed,
                               "waiting": len(self.store.queue())}
            self._changed()
            # Blocked (Android in the background without a foreground service, iOS out of background time,
            # the service stopped by the system): no timer spins meanwhile; resume / the next job drains.
            self._schedule_timer(cancel_only=blocked)

    async def _drain_pass(self) -> dict:
        out = {"saved": 0, "failed": 0, "blocked": False}
        skip: set = set()
        self._held = None  # asked once per pass, right before the first write (``_ensure_hold``)
        while not self._stop_drain and not self._closed:
            dest = self.destination()
            if dest is None or not dest.usable:
                break
            entry = self.store.due(self.clock(), skip=skip)
            if entry is None:
                break
            outcome = await self._sync_entry(dest, entry)
            if outcome == "saved":
                out["saved"] += 1
            elif outcome == "retry":
                out["failed"] += 1
            elif outcome in ("stop", "paused"):
                out["blocked"] = outcome == "stop"
                break
            skip.add(entry["key"])
        dest = self.destination()
        if not out["blocked"] and not self._stop_drain and not self._closed and dest is not None and dest.usable:
            await self._sweep_leftovers(dest)
        return out

    async def _ensure_hold(self, dest: Destination) -> bool:
        """Keep the app alive before the first write of this pass (never for a pass that only finds books with
        nothing to write or waiting for a save location). False: Android in the background without a
        foreground service - the drain stops and waits for the app (``_stop_drain``)."""
        if self._held is None:
            held = await self.keepalive.ensure(f"Saving to {dest.display}…", may_start=self.visible)
            if not held and not self.visible and self.platform == "android":
                self._event("drain waits for the app (no foreground service)")
                self._stop_drain = True
                return False
            self._held = True
        return True

    async def _sync_entry(self, dest: Destination, entry: dict) -> str:
        """One book: 'saved' / 'done' (nothing to do) / 'defer' (busy) / 'retry' / 'paused' / 'stop'.

        Only the queue generation taken here is finished / rescheduled (``gen``): a recompile, a format turned
        on or a Send now that arrives while the book is copied keeps the entry due for the next pass."""
        key, identity, manual = entry["key"], entry["identity"], bool(entry.get("manual"))
        gen = entry.get("gen")
        epoch = self._retire_epoch
        if not self._wants(identity, manual=manual):
            self._finish(key, gen)
            return "done"
        reported = list(entry.get("reported") or ())
        anywhere = bool(self._job_reports.get(key) and reported)
        try:
            book = await self._io(self._book_blocking, identity, reported, anywhere)
        except Exception as exc:
            log.warning("reading a book for cloud sync failed: %s", type(exc).__name__)
            return self._retry(entry, E_PROVIDER)
        if not book.exists:
            if int(entry.get("attempts") or 0) == 0:
                self.store.reschedule(key, self.clock() + SOURCE_RETRY, attempts=1, error=E_SOURCE_MISSING, gen=gen)
                return "retry"
            self._finish(key, gen)
            for kind in KINDS:
                if self.store.record(dest.id, key, kind):
                    self._rec(dest, key, kind, status="source_missing", error="")
            return "done"
        if book.state == "attachments":
            self._waiting_library.add(key)
            self._finish(key, gen)
            return "done"
        self._waiting_library.discard(key)
        if book.state != "library":
            self._finish(key, gen)
            return "done"
        if self._busy(identity):
            self._event("book busy: waiting for its job")
            self.store.reschedule(key, self.clock() + BUSY_RETRY, error=BUSY, gen=gen)
            return "defer"
        wanted_kinds = entry.get("kinds")
        codes: dict = {}
        for kind in KINDS:
            settings = self.settings()  # each kind reads the switches again (a format turned on meanwhile)
            if not settings.kind_on(kind) or (wanted_kinds is not None and kind not in wanted_kinds):
                continue
            choice = book.choices.get(kind)
            if choice is None:
                continue
            if self.platform == "ios":
                await self._ensure_hold(dest)  # the background task first: it is what the remaining time measures
            if not await self._time_for(choice.size):
                self._event("not enough background time: the rest waits for the app")
                self._stop_drain = True
                return "stop"
            try:
                codes[kind] = await self._push_kind(dest, book, kind, choice)
            except _Relink:
                await self._mark_relink(dest)
                return "paused"
            if self._retire_epoch != epoch:
                return "paused"  # the destination changed meanwhile: the next pass writes to the new one
            dest = self.destination() or dest
            if self._stop_drain:
                return "stop"
        errors = [c for c in codes.values() if c not in ("ok", "skipped", "needs_pick")]
        if E_CANCELLED in errors:
            # stays due; not an attempt. A write cancelled because the destination was changed only pauses this
            # pass (the drain goes round again for the new one); Stop / a foreground-service timeout stops it.
            return "paused" if self._retire_epoch != epoch else "stop"
        if errors:
            return self._retry(entry, errors[0], book=book)
        self._finish(key, gen)
        return "saved" if "ok" in codes.values() else "done"

    def _finish(self, key: str, gen: Optional[int] = None) -> None:
        if self.store.finish(key, gen) or self.store.queued(key) is None:
            self._job_reports.pop(key, None)

    def _retry(self, entry: dict, code: str, *, book: Optional[_Book] = None) -> str:
        attempts = int(entry.get("attempts") or 0) + 1
        delay = BACKOFF_MAX if code == E_SPACE else backoff_delay(attempts)
        updated = self.store.reschedule(entry["key"], self.clock() + delay, attempts=attempts, error=code,
                                        gen=entry.get("gen"))
        if updated is None:
            return "retry"  # a newer trigger came in meanwhile: it stays due with a fresh start
        if attempts == FAIL_AFTER or (code == E_SPACE and attempts == 1):
            self._failures_new.append(code)  # one notification for the whole drain (``_post_failures``)
        return "retry"

    async def _post_failures(self) -> None:
        """One "Couldn't save" notification per drain (the same id per destination, so a later one replaces
        it): how many books wait after ``FAIL_AFTER`` attempts (or for space on the phone)."""
        codes, self._failures_new = self._failures_new, []
        if not codes:
            return
        dest = self.destination()
        label = dest.display if dest is not None else "the cloud"
        count = sum(1 for e in self.store.queue()
                    if int(e.get("attempts") or 0) >= FAIL_AFTER or e.get("last_error") == E_SPACE)
        count = max(1, count)
        texts = _texts()
        if texts is not None and hasattr(texts, "cloud_failure_notice"):
            title, body, route = texts.cloud_failure_notice(label, count=count, unit="book")
        else:
            title = f"Couldn't save to {label}"
            body = f"{count} book{'s' if count != 1 else ''} waiting · tap for details"
            route = SETTINGS_ROUTE
        if E_SPACE in codes:
            body = "Not enough free space on the phone to prepare the copy"
        await self.notifier.show(f"failed:{dest.id if dest else ''}", title, body, route)

    async def _post_needs_pick(self) -> None:
        books, self._needs_pick_new = self._needs_pick_new, []
        if not books:
            return
        unique = {b.key: b for b in books}
        count = len(unique)
        bid = self._bid(next(iter(unique.values())).identity) if count == 1 else None
        texts = _texts()
        if texts is not None and hasattr(texts, "cloud_needs_location_notice"):
            title, body, route = texts.cloud_needs_location_notice(count, bid)
        else:
            title = "1 book needs a save location" if count == 1 else f"{count} books need a save location"
            body = "Tap to choose where to save it"
            route = f"/library/book/{bid}?tab=output" if bid else SETTINGS_ROUTE
        await self.notifier.show("needs_pick", title, body, route)

    async def _mark_relink(self, dest: Destination) -> None:
        current = self.destination()
        if current is None or current.id != dest.id:
            return
        if not current.needs_relink:
            self._set_destination(replace(current, needs_relink="revoked"))
            self._event("destination lost access")
        if ("relink", dest.id) not in self._notified:
            self._notified.add(("relink", dest.id))
            texts = _texts()
            if texts is not None and hasattr(texts, "cloud_failure_notice"):
                title, body, route = texts.cloud_failure_notice(dest.display, lost_access=True)
            else:
                title, body, route = (f"Glossarion lost access to {dest.display}", "Tap to choose it again",
                                      SETTINGS_ROUTE)
            await self.notifier.show(f"relink:{dest.id}", title, body, route)

    async def _time_for(self, size: int) -> bool:
        """iOS in the background: start a write only when it fits the remaining background time."""
        if self.platform != "ios":
            return True
        call = getattr(self.keepalive.native, "background_time_remaining", None)
        if call is None:
            return True
        try:
            remaining = await call()
        except Exception:
            return True
        if remaining is None:
            return True  # in the foreground
        return float(remaining) >= IOS_MIN_REMAINING + max(0, int(size)) / IOS_BYTES_PER_SECOND

    def _busy(self, identity: str) -> bool:
        """An active or queued job writes the workspace (never read a half-written EPUB)."""
        library = self.library
        if library is not None and hasattr(library, "is_compiling"):
            try:
                if library.is_compiling({"output_folder": identity, "path": identity}):
                    return True
            except Exception:
                pass
        view = getattr(self.jobs, "view", None) if self.jobs is not None else None
        if not callable(view):
            return False
        try:
            current = view()
        except Exception:
            return False
        snaps = ([current.active] if getattr(current, "active", None) is not None else []) + \
            list(getattr(current, "queue", ()) or ())
        if not snaps:
            return False
        try:
            from glossarion_mobile.ui.chat.attachments import job_writes_into
        except Exception:
            job_writes_into = None
        for snap in snaps:
            if job_writes_into is not None:
                if job_writes_into(snap, identity):
                    return True
            else:
                params = getattr(getattr(snap, "spec", None), "params", None) or {}
                if _norm(params.get("folder") or "") == _norm(identity):
                    return True
        return False

    # ---- per-kind write ----------------------------------------------------------------------------

    async def _push_kind(self, dest: Destination, book: _Book, kind: str, choice: OutputChoice) -> str:
        """'ok' / 'skipped' / 'needs_pick' / an error code to retry; raises ``_Relink``."""
        key = book.key
        record = self.store.record(dest.id, key, kind) or {}
        if dest.mode == MODE_FOLDER and (record.get("orphans") or record.get("replacing")):
            # cloud copies a replace still has to delete (killed mid-replace, delete-old failed): first
            await self._clean_leftovers(dest, key, kind, record)
            record = self.store.record(dest.id, key, kind) or {}
        if (record.get("status") == "ok" and not record.get("dirty") and record.get("doc")
                and record.get("source") == choice.path and record.get("size") == choice.size
                and record.get("mtime_ns") == choice.mtime_ns):
            return "skipped"
        if dest.mode == MODE_FILES and not record.get("doc"):
            if record.get("status") != "needs_pick":  # newly waiting for a place: one notification per drain
                self._rec(dest, key, kind, book.identity, status="needs_pick", name=choice.name, error="", note="")
                self._needs_pick_new.append(book)
            return "needs_pick"
        if not await self._ensure_hold(dest):
            return E_CANCELLED  # waits for the app; not an attempt
        try:
            snap = await self._io(self._snapshot_blocking, choice.path)
        except _SourceGone:
            return E_SOURCE_MISSING
        except _SourceChanged:
            return E_SOURCE_CHANGED
        except _NoSpace:
            self._rec(dest, key, kind, book.identity, status="failed", error="not enough free space on the phone")
            return E_SPACE
        op_id = uuid.uuid4().hex
        try:
            current = self.destination()
            if self._stop_drain or self._closed or current is None or current.id != dest.id:
                # Stop / a foreground-service timeout / a destination change while the private copy was made:
                # nothing is written after the hold was dropped (the queue keeps the book)
                return E_CANCELLED
            if (record.get("status") == "ok" and not record.get("dirty") and record.get("doc")
                    and record.get("sha1") == snap.sha1):
                self._rec(dest, key, kind, book.identity, size=choice.size, mtime_ns=choice.mtime_ns,
                          source=choice.path)
                return "skipped"
            name = str(record.get("name") or choice.name)
            self._inflight = {"op_id": op_id, "key": key, "kind": kind, "name": name, "written": 0,
                              "total": snap.size, "dest": dest.display}
            self._changed()
            await self.keepalive.update(_progress_text(name, dest.display))
            if not self._rec(dest, key, kind, book.identity, status="writing"):
                return "skipped"  # the book was deleted meanwhile (a destination change pauses: ``_retire_epoch``)
            await self._io(self.store.flush)  # a kill mid-write must leave 'writing' on disk (rewrite next time)
            if dest.mode == MODE_PHONE:
                return await self._write_phone(dest, book, kind, choice, snap, record)
            return await self._write_doc(dest, book, kind, choice, snap, record, op_id)
        finally:
            self._inflight = None
            await self._io(self._remove_blocking, snap.path)
            self._changed()

    def _rec(self, dest: Destination, key: str, kind: str, identity: Any = None, **fields: Any) -> dict:
        """A record change made by the drain. Nothing is stored ({}) when the destination was changed or
        forgotten meanwhile (its records were retired: no orphan record under the old id) or the book was
        deleted meanwhile (a copy that finishes after the delete must not bring its record back)."""
        current = self.destination()
        if current is None or current.id != dest.id:
            return {}
        return self.store.update_record(dest.id, key, kind, identity, live_only=True, **fields)

    def _ok(self, dest: Destination, book: _Book, kind: str, choice: OutputChoice, snap: _Snapshot, *, doc: Any,
            name: str, parent: Any, size: Optional[int], mode: str, note: str = "", base_name: str = "") -> str:
        """Store a successful write under the book's *current* key (it may have moved meanwhile)."""
        previous = self.store.record(dest.id, book.key, kind) or {}
        self._rec(dest, book.key, kind, None, doc=doc, name=name, parent=parent, size=choice.size,
                  mtime_ns=choice.mtime_ns, sha1=snap.sha1, source=choice.path, synced_at=self.clock(),
                  remote_size=size if size is not None else snap.size, mode=mode, status="ok", error="",
                  note=note, dirty=False, base_name=base_name or previous.get("base_name") or name)
        return "ok"

    def _fail(self, dest: Destination, book: _Book, kind: str, code: str, *, status: str = "pending",
              **fields: Any) -> str:
        self._rec(dest, book.key, kind, None, status=status, error=_error_text(code), **fields)
        return code

    async def _write_phone(self, dest: Destination, book: _Book, kind: str, choice: OutputChoice, snap: _Snapshot,
                           record: dict) -> str:
        name = str(record.get("name") or choice.name)
        folder = self._phone_folder_name(dest, book)
        root = _phone_subdir()
        subdir = root if dest.layout == LAYOUT_FLAT or not folder else f"{root}/{folder}"
        uri = await self.docs.save_to_downloads(snap.path, name, _mime(choice.path), subdir,
                                                replace_uri=record.get("doc") or None)
        if not uri:
            return self._fail(dest, book, kind, E_PROVIDER, dirty=True)
        return self._ok(dest, book, kind, choice, snap, doc=uri, name=name, parent=subdir, size=snap.size,
                        mode="mediastore")

    def _phone_folder_name(self, dest: Destination, book: _Book) -> str:
        """Phone folder: the book's sub-folder name (kept once chosen; ' (2)' when another book has it)."""
        entry = self.store.book(dest.id, book.key) or {}
        folder = entry.get("folder") or {}
        if folder.get("name"):
            return str(folder["name"])
        claimed = self.store.claimed(dest.id, parent="__root__", exclude_key=book.key)
        base = _safe(book.folder_name or book.title or "Book")
        name, n = base, 1
        while name.casefold() in claimed:
            n += 1
            name = _variant(base, n)
        self._set_folder(dest, book, {"doc": "", "name": name})
        return name

    def _set_folder(self, dest: Destination, book: _Book, folder: Mapping[str, Any]) -> None:
        """The book's cloud folder, stored like ``_rec`` (never for a retired destination or a deleted book)."""
        current = self.destination()
        if current is None or current.id != dest.id:
            return
        self.store.set_book_folder(dest.id, book.key, dict(folder), book.identity, book.title, live_only=True)

    async def _query_root(self, dest: Destination) -> dict:
        """``query_root`` of the destination; a refreshed ref (iOS stale bookmark) replaces the stored one."""
        root = await self.docs.query_root(dest.target)
        fresh = root.get("target") if root["ok"] else None
        if fresh and isinstance(fresh, Mapping) and fresh != dest.target:
            current = self.destination()
            if current is not None and current.id == dest.id:
                self._set_destination(replace(current, target=dict(fresh)))
        return root

    async def _check_root(self, dest: Destination, code: str) -> None:
        """A failure that may concern the whole destination: the root no longer answering with a permission
        error (or as gone) means it needs a re-link (``_Relink``); anything else is retried later."""
        if code not in (E_PERMISSION, E_MISSING):
            return
        root = await self._query_root(dest)
        if not root["ok"] and root["error"] in (E_PERMISSION, E_MISSING):
            raise _Relink(code)

    async def _book_parent(self, dest: Destination, book: _Book, *, fresh: bool = False) -> Any:
        """Folder mode: the book's sub-folder ref (adopted by name or created), or the root for a flat
        destination. Raises ``_Relink`` when the root itself is gone, ``_RetryLater`` otherwise."""
        if dest.layout == LAYOUT_FLAT:
            return dest.target
        entry = self.store.book(dest.id, book.key) or {}
        folder = entry.get("folder") or {}
        if folder.get("doc") and not fresh:
            return folder["doc"]
        claimed = self.store.claimed(dest.id, parent="__root__", exclude_key=book.key)
        base = _safe(book.folder_name or book.title or "Book")
        for n in range(1, 50):
            name = base if n == 1 else _variant(base, n)
            if name.casefold() in claimed:
                continue
            listing = await self.docs.list_children(dest.target, [name])
            if not listing["ok"]:
                await self._check_root(dest, listing["error"])
                raise _RetryLater(listing["error"])
            same = [c for c in listing["children"] if c["name"].casefold() == name.casefold()]
            match = next((c for c in same if c["is_dir"]), None)
            if match is not None:
                self._set_folder(dest, book, {"doc": match["doc"], "name": match["name"]})
                self._event("adopted an existing book folder")
                return match["doc"]
            if same:
                continue  # a file has the name: next variant
            created = await self.docs.create_dir(dest.target, name, on_exists="fail")
            if created["ok"] and created.get("doc"):
                made = created["doc"]
                self._set_folder(dest, book, {"doc": made, "name": _ref_name(made) or name})
                return made
            if created["error"] == E_EXISTS and created.get("doc"):
                existing = created["doc"]
                if isinstance(existing, Mapping) and existing.get("kind") == "folder":
                    self._set_folder(dest, book, {"doc": existing, "name": _ref_name(existing) or name})
                    return existing
                continue
            if created["error"] in (E_READ_ONLY, E_MODE, E_BAD_ARGS):
                # The provider cannot make folders: this destination goes flat (names stay unique).
                current = self.destination()
                if current is not None and current.id == dest.id:
                    self._set_destination(replace(current, layout=LAYOUT_FLAT))
                self._event("provider cannot create folders: flat layout")
                return dest.target
            await self._check_root(dest, created["error"])
            raise _RetryLater(created["error"] or E_PROVIDER)
        raise _RetryLater(E_PROVIDER)

    async def _adopt_or_create(self, dest: Destination, book: _Book, parent: Any, name: str, mime: str) -> tuple:
        """``(doc, name)``: an existing same-name file no other book's record uses is adopted (reinstall,
        wiped records: critic #9); a name another book uses gets ' (2)'; else a new document (the native
        side looks the name up again before retrying a failed create)."""
        claimed = self.store.claimed(dest.id, parent=parent, exclude_key=book.key)
        for n in range(1, 50):
            candidate = name if n == 1 else _variant(name, n)
            if candidate.casefold() in claimed:
                continue
            listing = await self.docs.list_children(parent, [candidate])
            if not listing["ok"]:
                await self._check_root(dest, listing["error"])
                raise _RetryLater(listing["error"])
            same = [c for c in listing["children"] if c["name"].casefold() == candidate.casefold()]
            match = next((c for c in same if not c["is_dir"] and c["name"] == candidate), None)
            if match is not None:
                self._event("adopted an existing cloud file")
                return match["doc"], match["name"]
            if same:
                continue
            created = await self.docs.create_file(parent, candidate, mime, on_exists="fail")
            if created["ok"] and created.get("doc"):
                return created["doc"], _ref_name(created["doc"]) or candidate
            if created["error"] == E_EXISTS and isinstance(created.get("doc"), Mapping) and \
                    created["doc"].get("kind") != "folder":
                return created["doc"], _ref_name(created["doc"]) or candidate  # the listing lagged: adopt
            if created["error"] == E_EXISTS:
                continue
            await self._check_root(dest, created["error"])
            raise _RetryLater(created["error"] or E_PROVIDER)
        raise _RetryLater(E_PROVIDER)

    async def _write_doc(self, dest: Destination, book: _Book, kind: str, choice: OutputChoice, snap: _Snapshot,
                         record: dict, op_id: str) -> str:
        mime = _mime(choice.path)
        try:
            parent = record.get("parent") or None
            doc = record.get("doc")
            name = str(record.get("name") or choice.name)
            if dest.mode == MODE_FOLDER and not doc:
                parent = await self._book_parent(dest, book)
                doc, name = await self._adopt_or_create(dest, book, parent, name, mime)
            recreated = False
            for _round in range(3):
                outcome = await self._write_existing(dest, book, kind, choice, snap, doc, name, parent, op_id, mime)
                if not outcome.startswith("recreate"):
                    return outcome
                if recreated or dest.mode != MODE_FOLDER:
                    break
                # proven gone (deleted, moved out of the folder): create it again, once
                recreated = True
                if outcome == "recreate:parent" or not parent:
                    parent = await self._book_parent(dest, book, fresh=True)
                doc, name = await self._adopt_or_create(dest, book, parent, name, mime)
                self._event("recreated a cloud file that was gone")
            return self._fail(dest, book, kind, E_MISSING)
        except _RetryLater as retry:
            code = str(retry.args[0] if retry.args else E_PROVIDER) or E_PROVIDER
            return self._fail(dest, book, kind, code)

    async def _write_existing(self, dest: Destination, book: _Book, kind: str, choice: OutputChoice,
                              snap: _Snapshot, doc: Any, name: str, parent: Any, op_id: str, mime: str,
                              *, adopted: bool = False) -> str:
        """Write into ``doc``; 'ok' / an error code / 'needs_pick' / 'recreate' / 'recreate:parent'."""
        result = await self.docs.write_file(doc, snap.path, mode_chain=DEFAULT_MODE_CHAIN, op_id=op_id)
        if result["ok"]:
            verified = result.get("size")
            if verified is None or verified == snap.size:
                return self._ok(dest, book, kind, choice, snap, doc=result.get("doc") or doc, name=name,
                                parent=parent, size=verified, mode=str(result.get("mode") or ""))
            return self._fail(dest, book, kind, E_SIZE, status="check", dirty=True)
        code = result["error"]
        if (code in (E_MODE, E_SIZE) and result.get("needs_replace")) or (code == E_SIZE and result.get("stale_tail")):
            return await self._replace(dest, book, kind, choice, snap, doc, name, parent, op_id, mime)
        if code == E_MISSING:
            verdict, other = await self._missing_proof(dest, parent, doc, name)
            if verdict in ("absent", "absent:parent"):
                return await self._gone(dest, book, kind, parent_gone=verdict == "absent:parent")
            if verdict == "adopt" and other is not None and not adopted:
                # the same name now has another handle (replaced by the user / the provider): keep updating it
                self._rec(dest, book.key, kind, None, doc=other)
                self._event("adopted a same-name cloud file")
                return await self._write_existing(dest, book, kind, choice, snap, other, name, parent, op_id, mime,
                                                  adopted=True)
            return self._fail(dest, book, kind, E_PROVIDER, dirty=True)
        if code == E_PERMISSION:
            return await self._permission_lost(dest, book, kind, parent, doc, name, result)
        if code == E_CANCELLED:
            self._rec(dest, book.key, kind, None, status="pending", dirty=True)
            return E_CANCELLED
        if code == E_READ_ONLY:
            if dest.mode == MODE_FOLDER:  # this one document refuses writes, its folder does not: a new copy
                return await self._replace(dest, book, kind, choice, snap, doc, name, parent, op_id, mime)
            return await self._needs_pick(dest, book, kind,
                                          "This file no longer accepts changes: choose where to save it")
        if code == E_SPACE:
            return self._fail(dest, book, kind, E_SPACE, dirty=True)
        if code in (E_SOURCE_CHANGED, E_SOURCE_MISSING):
            return self._fail(dest, book, kind, code, dirty=True)
        return self._fail(dest, book, kind, E_TIMEOUT if code == E_TIMEOUT else E_PROVIDER, dirty=True)

    async def _needs_pick(self, dest: Destination, book: _Book, kind: str, note: str) -> str:
        """The record waits for a new save location. Its old document's persisted grant (save locations mode,
        Android) is given back first: it no longer steers writes and counts toward the 128 / 512 cap."""
        old = (self.store.record(dest.id, book.key, kind) or {}).get("doc")
        self._rec(dest, book.key, kind, None, status="needs_pick", doc=None, error="", note=note)
        if old and dest.mode == MODE_FILES:
            await self._release_target(old)
        self._needs_pick_new.append(book)
        return "needs_pick"

    async def _gone(self, dest: Destination, book: _Book, kind: str, *, parent_gone: bool = False) -> str:
        if dest.mode == MODE_FOLDER:
            return "recreate:parent" if parent_gone else "recreate"
        return await self._needs_pick(dest, book, kind, "The saved file is gone: choose where to save it again")

    async def _missing_proof(self, dest: Destination, parent: Any, doc: Any, name: str) -> tuple:
        """``(verdict, doc)``: 'absent' / 'absent:parent' only when the destination answers and the document
        (or its book folder) is not there (critic #2: FileNotFoundException also means 'offline' or
        'unsupported mode'); 'adopt' + the other handle when the same name now has one; else 'unknown'."""
        if dest.mode != MODE_FOLDER:
            stat = await self.docs.stat(doc)
            gone = not stat["ok"] and stat["error"] == E_MISSING and stat.get("proven") is not False
            return ("absent" if gone else "unknown"), None
        root = await self._query_root(dest)
        if not root["ok"]:
            return "unknown", None
        listing = await self.docs.list_children(parent or dest.target, [name])
        if not listing["ok"]:
            if listing["error"] in (E_MISSING, E_PERMISSION) and parent and not _same(parent, dest.target):
                top = await self.docs.list_children(dest.target)
                if top["ok"] and top.get("complete", True) and \
                        not any(_same(c["doc"], parent) for c in top["children"]):
                    return "absent:parent", None
            return "unknown", None
        if not listing.get("complete", True):
            return "unknown", None
        same = [c for c in listing["children"] if not c["is_dir"] and c["name"] == name]
        if any(_same(c["doc"], doc) for c in same):
            return "unknown", None  # still there: the error was not about the file being gone
        if same:
            return "adopt", same[0]["doc"]
        return "absent", None

    async def _permission_lost(self, dest: Destination, book: _Book, kind: str, parent: Any, doc: Any, name: str,
                               result: Mapping[str, Any]) -> str:
        """SecurityException on one document: the destination still answering means only this record is
        affected (moved out of the folder / a pruned per-file grant; critic #7)."""
        if dest.mode == MODE_FOLDER:
            await self._check_root(dest, E_PERMISSION)  # raises _Relink when the folder itself is gone
            verdict, _other = await self._missing_proof(dest, parent, doc, name)
            if verdict in ("absent", "absent:parent"):
                return await self._gone(dest, book, kind, parent_gone=verdict == "absent:parent")
            return self._fail(dest, book, kind, E_PERMISSION, dirty=True)
        return await self._needs_pick(dest, book, kind,
                                      "Glossarion lost access to this file: choose where to save it again")

    async def _replace(self, dest: Destination, book: _Book, kind: str, choice: OutputChoice, snap: _Snapshot,
                       doc: Any, name: str, parent: Any, op_id: str, mime: str) -> str:
        """The provider cannot truncate and the new file is shorter (or takes no write mode): create a new
        document, write it, then delete the old one; its link / id changes and the record says so.

        Never two copies for good: the new document is recorded as ``replacing`` (and flushed) before its first
        byte, so a process killed mid-write leaves a torn copy the next drain deletes; an old copy that cannot
        be deleted now stays in ``orphans`` and is deleted again on later drains (``_clean_leftovers``)."""
        if dest.mode != MODE_FOLDER or not parent:
            return await self._needs_pick(dest, book, kind, "This cloud app cannot replace the file in place: choose "
                                                            "where to save it again (the old file stays)")
        record = self.store.record(dest.id, book.key, kind) or {}
        base = str(record.get("base_name") or name)  # the name the file had first (never ' (1) (1)')
        created = await self.docs.create_file(parent, base, mime, on_exists="rename")
        new_doc = created.get("doc") if created["ok"] else None
        if not new_doc:
            return self._fail(dest, book, kind, created["error"] or E_PROVIDER, dirty=True)
        if not self._rec(dest, book.key, kind, None, replacing=new_doc):
            await self.docs.delete(new_doc)  # the book was deleted meanwhile (a destination change pauses)
            return "skipped"
        await self._io(self.store.flush)
        result = await self.docs.write_file(new_doc, snap.path, mode_chain=DEFAULT_MODE_CHAIN, op_id=op_id)
        verified = result.get("size") if result["ok"] else None
        if not result["ok"] or (verified is not None and verified != snap.size):
            if await self.docs.delete(new_doc):
                self._rec(dest, book.key, kind, None, replacing=None)
            else:  # the partial copy stays until a later drain can delete it
                self._rec(dest, book.key, kind, None, replacing=None,
                          orphans=list(record.get("orphans") or []) + [new_doc])
            return self._fail(dest, book, kind, result["error"] or E_SIZE, dirty=True)
        old_gone = await self.docs.delete(doc)
        self._event("replaced a cloud file (the provider cannot truncate)")
        written = result.get("doc") or new_doc
        final = _ref_name(written) or base
        if final != base and old_gone:
            # the provider named the new copy 'Book (1).epub' while the old one existed: give it the
            # original name back where the native side can rename (else the next replace restores it)
            renamed = await self.docs.rename(written, base)
            if renamed["ok"] and renamed.get("doc"):
                written, final = renamed["doc"], _ref_name(renamed["doc"]) or base
        outcome = self._ok(dest, book, kind, choice, snap, doc=written, name=final, parent=parent, size=verified,
                           mode=str(result.get("mode") or ""), note=REPLACED_NOTE, base_name=base)
        orphans = list((self.store.record(dest.id, book.key, kind) or {}).get("orphans") or [])
        if not old_gone:
            orphans.append(doc)
            self._event("the old cloud copy could not be deleted yet: tried again later")
        self._rec(dest, book.key, kind, None, replacing=None, orphans=orphans or None)
        return outcome

    async def _clean_leftovers(self, dest: Destination, key: str, kind: str, record: Mapping[str, Any]) -> int:
        """Delete the cloud copies a replace left (``replacing``: torn by a killed process; ``orphans``: old
        copies whose delete failed). Returns how many are still there (kept for a later drain)."""
        handles = list(record.get("orphans") or [])
        if record.get("replacing"):
            handles.append(record.get("replacing"))
        handles = [h for h in handles if not _same(h, record.get("doc"))]  # never the copy the record updates
        kept: list = []
        for handle in handles:
            try:
                gone = await self.docs.delete(handle)
            except Exception:
                gone = False
            if not gone:
                kept.append(handle)
        if len(kept) < len(handles):
            self._event(f"deleted {len(handles) - len(kept)} leftover cloud cop{'y' if len(handles) - len(kept) == 1 else 'ies'}")
        self._rec(dest, key, kind, None, replacing=None, orphans=kept or None)
        return len(kept)

    async def _sweep_leftovers(self, dest: Destination) -> None:
        """After the queue: delete leftover copies of books that are not queued (a replace whose delete-old
        failed while the book itself is saved), backing off like the queue while the provider refuses."""
        if dest.mode != MODE_FOLDER:
            return
        now = self.clock()
        if self._leftovers_due and now < self._leftovers_due:
            return
        pending = self.store.leftovers(dest.id)
        kept = 0
        for key, kind, record in pending:
            if self._stop_drain or self._closed:
                return
            kept += await self._clean_leftovers(dest, key, kind, record)
        if kept:
            self._leftovers_attempts += 1
            self._leftovers_due = now + backoff_delay(self._leftovers_attempts)
        else:
            self._leftovers_attempts, self._leftovers_due = 0, 0.0

    # ---- blocking helpers (io pool) ----------------------------------------------------------------

    def _snapshot_dir(self) -> str:
        base = self.cache_dir or os.path.join(os.path.expanduser("~"), ".cache")
        return os.path.join(base, SNAPSHOT_DIR)

    def _clean_snapshots_blocking(self) -> int:
        folder = self._snapshot_dir()
        removed = 0
        if not os.path.isdir(folder):
            return 0
        for name in os.listdir(folder):
            try:
                os.remove(os.path.join(folder, name))
                removed += 1
            except OSError:
                pass
        return removed

    def _snapshot_blocking(self, source: str) -> _Snapshot:
        """Copy ``source`` into the app cache (hashing it) and check it did not change meanwhile."""
        try:
            before = os.stat(source)
        except FileNotFoundError:
            raise _SourceGone(source) from None
        folder = self._snapshot_dir()
        os.makedirs(folder, exist_ok=True)
        try:
            free = shutil.disk_usage(folder).free
        except OSError:
            free = None
        if free is not None and free < before.st_size + SPACE_MARGIN:
            raise _NoSpace(source)
        target = os.path.join(folder, uuid.uuid4().hex + os.path.splitext(source)[1].lower())
        digest = hashlib.sha1()
        copied = 0
        try:
            with open(source, "rb") as src, open(target, "wb") as dst:
                while True:
                    chunk = src.read(COPY_CHUNK)
                    if not chunk:
                        break
                    digest.update(chunk)
                    dst.write(chunk)
                    copied += len(chunk)
                dst.flush()
                os.fsync(dst.fileno())
            after = os.stat(source)
        except FileNotFoundError:
            self._remove_blocking(target)
            raise _SourceGone(source) from None
        except OSError as exc:
            self._remove_blocking(target)
            if getattr(exc, "errno", None) == 28:  # ENOSPC
                raise _NoSpace(source) from exc
            raise
        if copied != before.st_size or (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            self._remove_blocking(target)
            raise _SourceChanged(source)
        return _Snapshot(path=target, size=copied, sha1=digest.hexdigest(), mtime_ns=before.st_mtime_ns)

    def _write_probe_blocking(self) -> str:
        folder = self._snapshot_dir()
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"probe-{uuid.uuid4().hex[:8]}.txt")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("Glossarion can save files here. You can delete this file.\n")
        return path

    @staticmethod
    def _remove_blocking(path: str) -> None:
        try:
            os.remove(path)
        except OSError:
            pass

    def _book_row(self, identity: str) -> dict:
        key = _norm(identity)
        from glossarion_mobile.services.library import LibraryService, book_identity

        for row in self._library_books():
            if _norm(book_identity(row)) == key:
                return dict(row)
        return LibraryService._synthesise(identity) or {"path": identity, "name": os.path.basename(identity)}

    def _in_scan(self, identity: str) -> bool:
        key = _norm(identity)
        from glossarion_mobile.services.library import book_identity

        return any(_norm(book_identity(row)) == key for row in self._library_books())

    def _library_state_blocking(self, identity: str) -> str:
        """library | attachments (a chat book waiting for the auto-migrate) | outside | missing."""
        if not identity or not os.path.exists(identity):
            return "missing"
        parent = os.path.dirname(os.path.normpath(identity))
        if os.path.basename(parent).casefold() == "attachments":
            return "attachments"
        if self._is_raw_source(identity):
            return "outside"  # an untranslated source in Library/Raw is never a book to copy
        if self._in_scan(identity):
            return "library"
        library = self.library
        if library is None:
            return "outside"
        try:
            roots = list(library.output_roots())
        except Exception:
            roots = []
        if any(_norm(parent) == _norm(r) for r in roots if r):
            return "library"
        try:
            root = library.library_root()
        except Exception:
            root = ""
        if root and _under(identity, root):
            return "library"
        return "outside"

    def _book_blocking(self, identity: str, reported: Sequence[str], anywhere: bool) -> _Book:
        key = _norm(identity)
        exists = bool(identity) and os.path.exists(identity)
        state = self._library_state_blocking(identity) if exists else "missing"
        row = self._book_row(identity) if exists else {"path": identity}
        if os.path.isdir(identity):
            folder_name = os.path.basename(os.path.normpath(identity))
        else:
            folder_name = os.path.splitext(os.path.basename(identity))[0]
        title = str(row.get("name") or folder_name or "Book")
        choices: dict = {}
        if exists and state == "library":
            lister = getattr(self.library, "compiled_outputs_blocking", None) if self.library is not None else None
            if not row.get("output_folder") and os.path.isdir(identity):
                row = dict(row, output_folder=identity)
            choices = select_outputs(row, lister=lister, reported=reported, reported_anywhere=anywhere)
        return _Book(identity=os.path.abspath(identity) if identity else "", key=key, exists=exists, state=state,
                     row=row, folder_name=folder_name, title=title, choices=choices)

    # ---- install --------------------------------------------------------------------------------------

    @classmethod
    async def install(cls, app: Any) -> "CloudSyncService":
        """Build the service from the app (after the jobs, the chat and the Library), subscribe it and drain
        what waits. Sets ``app.cloud_sync``."""
        paths = getattr(app, "paths", None)
        page = getattr(app, "page", None)
        platform_value = getattr(getattr(page, "platform", None), "value", getattr(page, "platform", None))
        platform = str(platform_value or "desktop").lower()
        if platform not in ("android", "ios") or getattr(page, "web", False):
            platform = "desktop"
        data_dir = str(getattr(paths, "data", "") or os.getcwd())
        cache_dir = str(getattr(paths, "cache", "") or os.path.join(data_dir, "cache"))
        roots = [str(getattr(paths, name, "") or "") for name in ("data", "docs", "output", "library", "cache",
                                                                   "temp", "home")]
        original_home = getattr(paths, "original_home", None)
        if platform == "ios" and original_home:
            roots.append(str(original_home))  # the whole app container (Documents, Library, tmp)
        native = getattr(app, "native", None)
        jobs_feature = getattr(app, "jobs", None)
        job_service = getattr(app, "job_service", None) or jobs_feature
        background = getattr(jobs_feature, "background", None) if jobs_feature is not None else None
        notifications = getattr(jobs_feature, "notifications", None) if jobs_feature is not None else None
        dispatcher = getattr(app, "dispatcher", None)
        io_runner = spawn = post = None
        if dispatcher is not None and getattr(dispatcher, "bound", False):
            io_runner = _thread_runner(dispatcher)
            spawn = dispatcher.spawn
            post = dispatcher.post
        service = cls(
            docs=NativeDocs(native, platform=platform),
            store=CloudRecordStore(os.path.join(data_dir, CLOUD_FILE)),
            prefs=getattr(app, "prefs", None),
            library=getattr(app, "library", None),
            jobs=job_service,
            files=getattr(app, "files", None),
            keepalive=Keepalive(native, platform=platform, jobs=job_service, background=background),
            notifier=CloudNotifier(notifications, native),
            run_io=io_runner,
            spawn=spawn,
            post=post,
            platform=platform,
            cache_dir=cache_dir,
            app_roots=[r for r in roots if r],
        )
        service.attach(jobs=job_service, native=native)
        service._loading = True  # moves reported from now on wait until ``start`` loaded the records
        app.cloud_sync = service
        library = getattr(app, "library", None)
        if library is not None and hasattr(library, "cloud_sync"):
            library.cloud_sync = service  # the job-end mirror leaves book outputs to this service
        chat = getattr(app, "chat_feature", None)
        handover = getattr(chat, "attach_cloud_sync", None) if chat is not None else None
        if callable(handover):  # chat books the startup sweep moved before this service existed
            try:
                handover(service)
            except Exception:
                log.exception("handing the chat's earlier moves to cloud sync failed")
        await service.start()
        return service
