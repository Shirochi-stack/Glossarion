"""Google Drive sync of a book's outputs, shared by the desktop Library and Glossarion Mobile.

Three modes (one dropdown on both platforms; :data:`MODES`, :data:`MODE_LABELS`):

* ``compiled`` - "Compiled output only" (the default once sync is on): the compiled EPUB (and a
  compiled PDF when that is what was compiled) goes into the chosen Drive folder. Compiling the same
  book again UPDATES the same Drive file (files.update on the remembered id while that file is still
  in the chosen folder; otherwise a file with the same name in that folder that the app made is
  reused, unless another book remembers it, which gets "Name (2).epub"): never a second copy.
* ``full`` - "Full output folder": the book's whole output folder as
  ``<chosen folder>/<output folder name>/...``, file by file (no zip), keeping the folder structure.
  Incremental: a file whose md5 equals Drive's ``md5Checksum`` is not sent again; a changed file
  is updated in place. Temp / lock files are skipped and links out of the folder are not followed.
  Nothing is ever deleted on Drive (a one-way upload).
* ``full_raw`` - "Full output folder + raw source": as ``full``, plus the raw source file in
  ``<chosen folder>/<output folder name>/RAW/<source file name>``.

The chosen folder is a NAME or a path such as ``Books/Glossarion`` (the ``drive.file`` scope cannot
browse the user's folders); it is created when missing and its id remembered.

Layers: :func:`plan_sync` (local and pure: what goes where), :class:`DriveSyncRunner` (carries a
plan out with a ``google_drive_core.DriveClient`` and a caller-owned state dict of remembered ids),
:class:`DriveSyncWorker` (one background thread, coalesced per book, cancellable at shutdown) and
:class:`DriveSyncService` (the glue both platforms use: settings, token, state and log callbacks).
Nothing contacts Google unless sync is enabled and an account is connected. Uploads never raise into
the caller's compile. Pure Python 3.10, no GUI import.
"""

from __future__ import annotations

import copy
import logging
import mimetypes
import os
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

from share_hosting import CancelToken, HttpClient, ShareError

__all__ = [
    "COMPILED_EXTENSIONS",
    "DEFAULT_FOLDER",
    "DEFAULT_MODE",
    "DriveSyncRunner",
    "DriveSyncService",
    "DriveSyncWorker",
    "FOLDER_NOTE",
    "MODES",
    "MODE_COMPILED",
    "MODE_FULL",
    "MODE_FULL_RAW",
    "MODE_LABELS",
    "PlanItem",
    "RAW_FOLDER",
    "SyncJob",
    "SyncPlan",
    "SyncResult",
    "SyncSettings",
    "add_compile_listener",
    "compile_finished",
    "compiled_files_in",
    "folder_parts",
    "is_skipped_name",
    "normalize_mode",
    "plan_sync",
    "remove_compile_listener",
]

log = logging.getLogger("glossarion.drive.sync")

MODE_COMPILED = "compiled"
MODE_FULL = "full"
MODE_FULL_RAW = "full_raw"
MODES = (MODE_COMPILED, MODE_FULL, MODE_FULL_RAW)
MODE_LABELS = {
    MODE_COMPILED: "Compiled output only",
    MODE_FULL: "Full output folder",
    MODE_FULL_RAW: "Full output folder + raw source",
}
DEFAULT_MODE = MODE_COMPILED
DEFAULT_FOLDER = "Glossarion"
RAW_FOLDER = "RAW"
#: The one-line limitation the settings show under the folder field (decision B).
FOLDER_NOTE = "Glossarion can only see folders it made in your Drive, so type a folder name; it is created if missing."
#: What "compiled output" means for the compiled mode.
COMPILED_EXTENSIONS = (".epub", ".pdf")

_SKIP_NAMES = {"thumbs.db", "desktop.ini", ".ds_store", "__pycache__", ".git", ".svn"}
_SKIP_SUFFIXES = (".tmp", ".temp", ".part", ".partial", ".crdownload", ".lock", ".lck", ".swp", ".swx", ".bak~")
_SKIP_PREFIXES = ("~$", ".~lock.", ".#")


def normalize_mode(mode: Any) -> str:
    text = str(mode or "").strip().lower()
    return text if text in MODES else DEFAULT_MODE


def folder_parts(text: Any) -> tuple:
    """``"Books/Glossarion"`` -> ``("Books", "Glossarion")``; empty -> ``(DEFAULT_FOLDER,)``."""
    parts = []
    for raw in str(text or "").replace("\\", "/").split("/"):
        part = raw.strip()
        if part and part not in (".", ".."):
            parts.append(part[:255])
    return tuple(parts) or (DEFAULT_FOLDER,)


def is_skipped_name(name: str) -> bool:
    """Temp, lock and OS clutter files (and VCS / cache folders) a sync never uploads."""
    low = str(name).lower()
    if low in _SKIP_NAMES or low.endswith(_SKIP_SUFFIXES) or low.startswith(_SKIP_PREFIXES):
        return True
    return low.startswith(".") and ".tmp" in low  # atomic-write temp files (".name.json.abc.tmp")


def compiled_files_in(folder: str) -> list:
    """The compiled EPUB / PDF files directly in ``folder`` (``library_core``'s detection rules)."""
    if not folder or not os.path.isdir(folder):
        return []
    try:
        import library_core

        found = [path for path, _kind in library_core._list_compiled_outputs(folder)]
    except Exception:
        found = [entry.path for entry in os.scandir(folder) if entry.is_file(follow_symlinks=False)]
    return [p for p in found if p.lower().endswith(COMPILED_EXTENSIONS)]


@dataclass(frozen=True)
class PlanItem:
    local: str
    parts: tuple  # the remote path below the chosen folder, the file name last
    role: str  # "compiled" | "output" | "raw"


@dataclass
class SyncPlan:
    mode: str
    items: list = field(default_factory=list)
    folders: list = field(default_factory=list)  # remote folder paths (tuples) to ensure, parents first
    skipped: list = field(default_factory=list)  # local paths left out (temp / lock files, links)

    @property
    def empty(self) -> bool:
        return not self.items and not self.folders


def _inside(path: str, root: str) -> bool:
    try:
        return os.path.commonpath([os.path.normcase(path), os.path.normcase(root)]) == os.path.normcase(root)
    except ValueError:
        return False


def _is_link(path: str) -> bool:
    if os.path.islink(path):
        return True
    isjunction = getattr(os.path, "isjunction", None)  # Python 3.12+
    return bool(isjunction and isjunction(path))


def plan_sync(mode: str, *, output_folder: str = "", compiled_files: Sequence[str] = (),
              source_file: str = "") -> SyncPlan:
    """What one book's sync uploads, and where (local only; nothing touches the network)."""
    mode = normalize_mode(mode)
    plan = SyncPlan(mode)
    compiled = []
    for path in compiled_files or ():
        if path and os.path.isfile(path) and os.path.abspath(path) not in compiled:
            compiled.append(os.path.abspath(path))
    if mode == MODE_COMPILED:
        plan.items = [PlanItem(path, (os.path.basename(path),), "compiled") for path in compiled]
        return plan
    if not output_folder or not os.path.isdir(output_folder):
        return plan
    root = os.path.abspath(output_folder)
    real_root = os.path.realpath(root)
    top = os.path.basename(os.path.normpath(root)) or "Book"
    plan.folders.append((top,))
    for current, dirs, files in os.walk(root, followlinks=False):
        rel = os.path.relpath(current, root)
        rel_parts = () if rel in (".", "") else tuple(rel.replace("\\", "/").split("/"))
        kept = []
        for name in sorted(dirs):
            full = os.path.join(current, name)
            if is_skipped_name(name) or _is_link(full) or not _inside(os.path.realpath(full), real_root):
                plan.skipped.append(full)
                continue
            kept.append(name)
            plan.folders.append((top,) + rel_parts + (name,))
        dirs[:] = kept
        for name in sorted(files):
            full = os.path.join(current, name)
            if is_skipped_name(name) or _is_link(full) or not _inside(os.path.realpath(full), real_root):
                plan.skipped.append(full)
                continue
            plan.items.append(PlanItem(full, (top,) + rel_parts + (name,), "output"))
    # a compiled file kept outside the output folder (e.g. organized into the Library) joins the book folder
    for path in compiled:
        if not _inside(path, root):
            plan.items.append(PlanItem(path, (top, os.path.basename(path)), "compiled"))
    if mode == MODE_FULL_RAW and source_file and os.path.isfile(source_file) and not _is_link(source_file):
        plan.folders.append((top, RAW_FOLDER))
        plan.items.append(PlanItem(os.path.abspath(source_file), (top, RAW_FOLDER, os.path.basename(source_file)),
                                   "raw"))
    return plan


@dataclass
class SyncResult:
    mode: str
    label: str = ""
    uploaded: int = 0  # new files on Drive
    updated: int = 0  # files whose content was replaced in place
    unchanged: int = 0  # md5 already equal
    failed: int = 0
    folders_created: int = 0
    errors: list = field(default_factory=list)  # short "name: reason" lines
    error: str = ""  # the reason a sync stopped early (user-facing, no path / token)
    error_code: str = ""
    cancelled: bool = False
    started: float = 0.0
    finished: float = 0.0
    files: dict = field(default_factory=dict)  # remote path ("A/b.epub") -> Drive file id, for uploads/updates

    @property
    def ok(self) -> bool:
        return not self.error and not self.failed and not self.cancelled

    def summary(self) -> str:
        if self.cancelled:
            return "cancelled"
        if self.error:
            return self.error
        counts = [f"{self.uploaded} new", f"{self.updated} updated", f"{self.unchanged} unchanged"]
        if self.failed:
            counts.append(f"{self.failed} failed")
        return ", ".join(counts)

    def as_record(self) -> dict:
        """A small JSON-able record for "last sync" lines (no paths, no ids)."""
        return {"time": self.finished or time.time(), "ok": self.ok, "mode": self.mode, "label": self.label,
                "summary": self.summary(), "uploaded": self.uploaded, "updated": self.updated,
                "unchanged": self.unchanged, "failed": self.failed, "error_code": self.error_code}


def _mime(path: str) -> str:
    low = path.lower()
    if low.endswith(".epub"):
        return "application/epub+zip"
    return mimetypes.guess_type(path)[0] or "application/octet-stream"


class DriveSyncRunner:
    """Carries out one :class:`SyncPlan` with a ``DriveClient``.

    ``state`` is the caller's dict of remembered ids (mutated in place; persist it afterwards):
    ``{"root": {"path": "Books/Glossarion", "id": ...}, "books": {book_key: {"folder_id", "files":
    {remote path: id}}}}``. It holds no secrets.
    """

    def __init__(self, client: Any, state: Optional[dict] = None) -> None:
        self.client = client
        self.state = state if state is not None else {}

    # ---- folders --------------------------------------------------------------------------------------

    def _root(self, folder: Any, cancel: CancelToken, result: SyncResult) -> str:
        parts = folder_parts(folder)
        path = "/".join(parts)
        remembered = self.state.get("root") if isinstance(self.state.get("root"), dict) else {}
        if remembered.get("path") == path and remembered.get("id"):
            found = self.client.get(remembered["id"], cancel)
            if found is not None and found.is_folder and not found.trashed:
                return found.id
        current = "root"
        for part in parts:
            folder_file, created = self.client.ensure_folder(part, current, cancel)
            result.folders_created += int(created)
            current = folder_file.id
        self.state["root"] = {"path": path, "id": current}
        return current

    def _book(self, book_key: str) -> dict:
        books = self.state.setdefault("books", {})
        entry = books.get(book_key)
        if not isinstance(entry, dict):
            entry = books[book_key] = {}
        entry.setdefault("files", {})
        return entry

    # ---- files ----------------------------------------------------------------------------------------

    def _send(self, item: PlanItem, parent: str, existing: Any, cancel: CancelToken, result: SyncResult,
              progress: Optional[Callable[[str, int, int], Any]], remote_name: str = "") -> str:
        """Upload ``item`` into ``parent``: update ``existing`` in place (or skip when its md5 matches).
        ``remote_name``: the name on Drive when it differs from the local one ("Novel (2).epub")."""
        import google_drive_core as gdc

        name = remote_name or item.parts[-1]
        size = os.path.getsize(item.local)
        if existing is not None and existing.md5 and existing.size in (None, size):
            if gdc.file_md5(item.local, cancel) == existing.md5:
                result.unchanged += 1
                return existing.id
        report = (lambda sent, total: progress(name, sent, total)) if progress else None
        if existing is not None:
            try:
                uploaded = self.client.upload(item.local, name=name, mime=_mime(item.local), file_id=existing.id,
                                              size=size, progress=report, cancel=cancel)
                result.updated += 1
                return uploaded.id
            except ShareError as exc:
                if exc.code != "not_found":
                    raise
        uploaded = self.client.upload(item.local, name=name, mime=_mime(item.local), parent=parent, size=size,
                                      progress=report, cancel=cancel)
        result.uploaded += 1
        return uploaded.id

    def run(self, plan: SyncPlan, *, book_key: str, folder: Any = DEFAULT_FOLDER, label: str = "",
            cancel: Optional[CancelToken] = None,
            progress: Optional[Callable[[str, int, int], Any]] = None) -> SyncResult:
        """Run ``plan``. Never raises for Drive / network failures: they end up in the result."""
        cancel = cancel or CancelToken()
        result = SyncResult(plan.mode, label=label, started=time.time())
        try:
            if plan.empty:
                result.error, result.error_code = "Nothing to sync yet (no compiled output found).", "empty"
                return result
            root = self._root(folder, cancel, result)
            entry = self._book(book_key)
            if plan.mode == MODE_COMPILED:
                self._run_compiled(plan, root, entry, cancel, result, progress, book_key=book_key)
            else:
                self._run_full(plan, root, entry, cancel, result, progress)
        except ShareError as exc:
            if exc.code == "cancelled" or cancel.cancelled:
                result.cancelled = True
                result.error_code = "cancelled"
            else:
                result.error, result.error_code = exc.message, exc.code
        except OSError as exc:
            result.error, result.error_code = f"Could not read a local file ({type(exc).__name__}).", "missing_file"
        finally:
            result.finished = time.time()
        return result

    def _fail(self, item: PlanItem, exc: ShareError, result: SyncResult) -> None:
        import google_drive_core as gdc

        if exc.code in gdc.FATAL_CODES:
            raise exc
        result.failed += 1
        result.errors.append(f"{item.parts[-1]}: {exc.message}")

    def _claimed_elsewhere(self, file_id: str, book_key: str) -> bool:
        """True when another book's remembered files include ``file_id`` (never take its file over)."""
        for key, other in (self.state.get("books") or {}).items():
            if key == book_key or not isinstance(other, dict):
                continue
            if file_id in (other.get("files") or {}).values():
                return True
        return False

    def _compiled_target(self, name: str, root: str, book_key: str, remembered: Any,
                         cancel: CancelToken) -> tuple:
        """``(existing DriveFile or None, remote name)`` for one compiled file in the chosen folder.

        The remembered file is reused only while it is still in the chosen folder (the user may have
        changed the folder since). Otherwise a same-named file there is reused unless another book
        remembers it; then the next free "Name (2).epub" is used, as the device sync does."""
        if remembered:
            try:
                found = self.client.get(remembered, cancel)
            except ShareError as exc:
                if exc.code != "not_found":
                    raise
                found = None
            if (found is not None and not found.trashed and not found.is_folder and root in found.parents):
                return found, found.name or name
        stem, ext = os.path.splitext(name)
        for number in range(1, 100):
            candidate = name if number == 1 else f"{stem} ({number}){ext}"
            found = self.client.find_child(root, candidate, cancel)
            if found is None:
                return None, candidate
            if not self._claimed_elsewhere(found.id, book_key):
                return found, candidate
        return None, f"{stem} ({uuid.uuid4().hex[:6]}){ext}"

    def _run_compiled(self, plan, root, entry, cancel, result, progress, book_key: str = "") -> None:
        files = entry["files"]
        for item in plan.items:
            cancel.check()
            name = item.parts[-1]
            existing, remote = self._compiled_target(name, root, book_key, files.get(name), cancel)
            try:
                file_id = self._send(item, root, existing, cancel, result, progress, remote_name=remote)
            except ShareError as exc:
                self._fail(item, exc, result)
                continue
            except OSError as exc:
                result.failed += 1
                result.errors.append(f"{name}: could not be read ({type(exc).__name__})")
                continue
            files[name] = file_id
            result.files[name] = file_id

    def _run_full(self, plan, root, entry, cancel, result, progress) -> None:
        folder_ids: dict = {(): root}
        listings: dict = {}

        def children(folder_id: str) -> dict:
            if folder_id not in listings:
                found: dict = {}
                for item in self.client.list_children(folder_id, cancel):
                    found.setdefault((item.name, item.is_folder), item)
                listings[folder_id] = found
            return listings[folder_id]

        def ensure(parts: tuple) -> str:
            if parts in folder_ids:
                return folder_ids[parts]
            parent = ensure(parts[:-1])
            known = children(parent).get((parts[-1], True))
            if known is None:
                known = self.client.create_folder(parts[-1], parent, cancel)
                result.folders_created += 1
                children(parent)[(parts[-1], True)] = known
                listings[known.id] = {}
            folder_ids[parts] = known.id
            return known.id

        for parts in plan.folders:
            cancel.check()
            ensure(tuple(parts))
        if plan.folders:
            entry["folder_id"] = folder_ids.get(tuple(plan.folders[0]), "")
        for item in plan.items:
            cancel.check()
            parent = ensure(item.parts[:-1])
            existing = children(parent).get((item.parts[-1], False))
            try:
                file_id = self._send(item, parent, existing, cancel, result, progress)
            except ShareError as exc:
                self._fail(item, exc, result)
                continue
            except OSError as exc:
                result.failed += 1
                result.errors.append(f"{item.parts[-1]}: could not be read ({type(exc).__name__})")
                continue
            if item.role == "compiled" or existing is None or existing.id != file_id:
                result.files["/".join(item.parts)] = file_id
            if item.role == "compiled":
                entry["files"][item.parts[-1]] = file_id


# ---- background worker ---------------------------------------------------------------------------------


@dataclass
class SyncJob:
    book_key: str
    label: str = ""
    reason: str = "manual"  # "compile" | "manual"
    output_folder: str = ""
    compiled_files: tuple = ()
    source_file: str = ""


def _merge_jobs(queued: SyncJob, newer: SyncJob) -> SyncJob:
    """One job for a book queued twice: the files of both (an empty list means "all of the book's
    compiled files" and wins), "manual" when either was manual, the newer paths when given."""
    if not queued.compiled_files or not newer.compiled_files:
        files: tuple = ()
    else:
        files = tuple(dict.fromkeys(tuple(queued.compiled_files) + tuple(newer.compiled_files)))
    return SyncJob(book_key=newer.book_key, label=newer.label or queued.label,
                   reason="manual" if "manual" in (queued.reason, newer.reason) else newer.reason,
                   output_folder=newer.output_folder or queued.output_folder, compiled_files=files,
                   source_file=newer.source_file or queued.source_file)


class DriveSyncWorker:
    """One daemon thread running jobs in order; a book queued again while it waits is merged into
    its queued job (both jobs' files). :meth:`shutdown` cancels the running upload and drops the queue."""

    def __init__(self, run_job: Callable[[SyncJob, CancelToken], Any], *, name: str = "drive-sync") -> None:
        self._run_job = run_job
        self._name = name
        self._lock = threading.Lock()
        self._pending: dict = {}  # book_key -> SyncJob, in arrival order
        self._wake = threading.Event()
        self._idle = threading.Event()
        self._idle.set()
        self._closed = False
        self._thread: Optional[threading.Thread] = None
        self._cancel: Optional[CancelToken] = None
        self.current: Optional[SyncJob] = None

    @property
    def busy(self) -> bool:
        return not self._idle.is_set()

    @property
    def queued(self) -> int:
        """Jobs waiting behind the running one."""
        with self._lock:
            return len(self._pending)

    def submit(self, job: SyncJob) -> bool:
        with self._lock:
            if self._closed:
                return False
            queued = self._pending.pop(job.book_key, None)
            self._pending[job.book_key] = _merge_jobs(queued, job) if queued is not None else job
            self._idle.clear()
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._loop, name=self._name, daemon=True)
                self._thread.start()
        self._wake.set()
        return True

    def cancel_current(self) -> None:
        with self._lock:
            token = self._cancel
        if token is not None:
            token.cancel()

    def wait_idle(self, timeout: Optional[float] = None) -> bool:
        return self._idle.wait(timeout)

    def shutdown(self, timeout: float = 5.0) -> None:
        with self._lock:
            self._closed = True
            self._pending.clear()
            token, thread = self._cancel, self._thread
        if token is not None:
            token.cancel()
        self._wake.set()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout)

    def _loop(self) -> None:
        while True:
            with self._lock:
                if self._closed:
                    self._idle.set()
                    return
                if not self._pending:
                    self._wake.clear()
                    self._idle.set()
                    job = None
                else:
                    key = next(iter(self._pending))
                    job = self._pending.pop(key)
                    self._cancel = CancelToken()
                    self.current = job
                    token = self._cancel
            if job is None:
                if not self._wake.wait(30.0):
                    with self._lock:
                        if not self._pending:  # idle for a while: let the thread end
                            self._thread = None
                            self._idle.set()
                            return
                continue
            try:
                self._run_job(job, token)
            except Exception:  # never let one job kill the worker
                log.exception("Drive sync job crashed")
            finally:
                with self._lock:
                    self._cancel = None
                    self.current = None


# ---- the glue both platforms use ----------------------------------------------------------------------


@dataclass(frozen=True)
class SyncSettings:
    enabled: bool = False
    mode: str = DEFAULT_MODE
    folder: str = DEFAULT_FOLDER
    signed_in: bool = False

    @property
    def active(self) -> bool:
        return bool(self.enabled and self.signed_in)


class DriveSyncService:
    """Drive sync for one platform.

    * ``settings()`` -> :class:`SyncSettings` (read at each request, so a change applies at once);
    * ``access()`` -> the ``access(rejected)`` token callable of ``DriveClient`` (called on the worker);
    * ``load_state()`` / ``save_state(state)`` -> the remembered-ids dict (worker thread);
    * ``on_result(job, result)`` -> e.g. store the "last sync" record and refresh the UI (worker thread);
    * ``log(line)`` -> one short line at start, done and failure;
    * ``on_sign_in_lost()`` -> Google refused the stored grant (revoked / expired): forget it.

    :meth:`on_compiled` and :meth:`request_sync` only queue work and never raise; nothing contacts
    Google while sync is off or no account is connected (manual :meth:`request_sync` needs the account).
    """

    def __init__(self, *, settings: Callable[[], SyncSettings], access: Callable[[], Callable[[str], str]],
                 load_state: Callable[[], dict], save_state: Callable[[dict], Any],
                 on_result: Optional[Callable[[SyncJob, SyncResult], Any]] = None,
                 log_line: Optional[Callable[[str], Any]] = None,
                 on_sign_in_lost: Optional[Callable[[], Any]] = None,
                 http: Optional[HttpClient] = None, api_base: Optional[str] = None,
                 upload_base: Optional[str] = None,
                 resolve_source: Optional[Callable[[SyncJob], str]] = None,
                 resolve_compiled: Optional[Callable[[SyncJob], Sequence[str]]] = None) -> None:
        self.resolve_source = resolve_source
        # ``resolve_compiled(job)`` -> the book's compiled files when the job names none (worker thread). Mobile
        # uses it for a Library-filed EPUB outside the workspace and the "<Title>.pdf" that library_core's
        # list leaves out; without it (or when it finds nothing) the compiled mode lists the output folder.
        self.resolve_compiled = resolve_compiled
        self.settings = settings
        self.access = access
        self.load_state = load_state
        self.save_state = save_state
        self.on_result = on_result
        self.log_line = log_line
        self.on_sign_in_lost = on_sign_in_lost
        self.http = http
        self.api_base = api_base
        self.upload_base = upload_base
        self.worker = DriveSyncWorker(self._run_job)

    def _log(self, line: str) -> None:
        log.info("%s", line)
        if self.log_line is not None:
            try:
                self.log_line(line)
            except Exception:
                pass

    def _settings(self) -> SyncSettings:
        try:
            value = self.settings()
        except Exception:
            log.debug("Drive sync settings unavailable", exc_info=True)
            return SyncSettings()
        return value if isinstance(value, SyncSettings) else SyncSettings()

    def request_sync(self, job: SyncJob) -> bool:
        """Queue a sync of one book (manual "Sync now" or a compile). False when sync is not set up."""
        try:
            settings = self._settings()
            if not settings.signed_in or (job.reason != "manual" and not settings.enabled):
                return False
            return self.worker.submit(job)
        except Exception:
            log.debug("Drive sync request failed", exc_info=True)
            return False

    def on_compiled(self, job: SyncJob) -> bool:
        """A compile finished: queue the book when sync is on (never raises into the compile)."""
        job.reason = "compile"
        return self.request_sync(job)

    def shutdown(self, timeout: float = 5.0) -> None:
        self.worker.shutdown(timeout)

    def client(self) -> Any:
        import google_drive_core as gdc

        kwargs = {}
        if self.api_base:
            kwargs["api_base"] = self.api_base
        if self.upload_base:
            kwargs["upload_base"] = self.upload_base
        return gdc.DriveClient(self.access(), self.http, **kwargs)

    def run_blocking(self, job: SyncJob, cancel: Optional[CancelToken] = None) -> SyncResult:
        """Run one job on this thread (the worker uses it; tests may too)."""
        settings = self._settings()
        mode = normalize_mode(settings.mode)
        label = job.label or os.path.basename(os.path.normpath(job.output_folder or "")) or "book"
        cancel = cancel or CancelToken()
        result = SyncResult(mode, label=label, started=time.time())
        if not settings.signed_in or (job.reason != "manual" and not settings.enabled):
            # switched off or signed out while the job waited: nothing is sent
            result.error, result.error_code, result.finished = "Drive sync is off.", "disabled", time.time()
            return result
        if mode != MODE_COMPILED and not job.output_folder:
            # e.g. a Library book whose output folder is gone: never upload its shelf instead
            result.error = f"{label} has no output folder; only \"{MODE_LABELS[MODE_COMPILED]}\" can sync it."
            result.error_code, result.finished = "no_output_folder", time.time()
            self._log(f"⚠️ Drive sync failed: {label}: no output folder")
            if self.on_result is not None:
                try:
                    self.on_result(job, result)
                except Exception:
                    log.debug("Drive sync result callback failed", exc_info=True)
            return result
        self._log(f"☁️ Drive sync: {label} ({MODE_LABELS[mode]})…")
        state: dict = {}
        try:
            compiled = list(job.compiled_files or ())
            if not compiled and self.resolve_compiled is not None:
                try:
                    compiled = [str(p) for p in (self.resolve_compiled(job) or ()) if p]
                except Exception:
                    log.debug("resolving the compiled files failed", exc_info=True)
                    compiled = []
            if not compiled and mode == MODE_COMPILED:
                compiled = compiled_files_in(job.output_folder)
            source = ""
            if mode == MODE_FULL_RAW:
                source = job.source_file
                if not source and self.resolve_source is not None:
                    try:
                        source = self.resolve_source(job) or ""
                    except Exception:
                        source = ""
            plan = plan_sync(mode, output_folder=job.output_folder, compiled_files=compiled, source_file=source)
            try:
                state = copy.deepcopy(self.load_state() or {})
            except Exception:
                state = {}
            result = DriveSyncRunner(self.client(), state).run(plan, book_key=job.book_key, folder=settings.folder,
                                                                label=label, cancel=cancel)
        except ShareError as exc:
            result.error, result.error_code = exc.message, exc.code
            result.finished = time.time()
        except Exception as exc:  # never into the caller
            log.debug("Drive sync failed", exc_info=True)
            result.error, result.error_code = f"Drive sync failed ({type(exc).__name__}).", "error"
            result.finished = time.time()
        if state:
            try:
                self.save_state(state)
            except Exception:
                log.debug("Drive sync state not saved", exc_info=True)
        if result.error_code == "sign_in" and self.on_sign_in_lost is not None:
            try:
                self.on_sign_in_lost()
            except Exception:
                pass
        if result.cancelled:
            self._log(f"☁️ Drive sync cancelled: {label}")
        elif result.ok:
            self._log(f"✅ Drive sync done: {label}: {result.summary()}")
        else:
            reason = result.error or (result.errors[0] if result.errors else result.summary())
            self._log(f"⚠️ Drive sync failed: {label}: {reason}")
        if self.on_result is not None:
            try:
                self.on_result(job, result)
            except Exception:
                log.debug("Drive sync result callback failed", exc_info=True)
        return result

    def _run_job(self, job: SyncJob, cancel: CancelToken) -> None:
        self.run_blocking(job, cancel)


# ---- "a compile finished" listeners ---------------------------------------------------------------------
#
# The compile at the end of a translation run happens inside the translation engine (TransateKRtoEN),
# which knows nothing about the front end. It reports the compiled file here; a front end that wants
# it (the desktop Library's Drive sync) registers a listener. With no listener (mobile, which uses its
# own job DONE hook; the CLI) this does nothing.

_COMPILE_LISTENERS: list = []
_COMPILE_LISTENERS_LOCK = threading.Lock()


def add_compile_listener(listener: Callable[[str, Optional[str], Optional[str]], Any]) -> None:
    """``listener(kind, path, error)`` runs on the compiling thread after each reported compile (once)."""
    with _COMPILE_LISTENERS_LOCK:
        if listener not in _COMPILE_LISTENERS:
            _COMPILE_LISTENERS.append(listener)


def remove_compile_listener(listener: Callable[..., Any]) -> None:
    with _COMPILE_LISTENERS_LOCK:
        try:
            _COMPILE_LISTENERS.remove(listener)
        except ValueError:
            pass


def compile_finished(kind: str, path: Any = None, error: Any = None) -> None:
    """A compile produced ``path`` (``kind``: "epub" / "pdf"), or failed with ``error``. Calls the
    listeners; never raises into the compile."""
    with _COMPILE_LISTENERS_LOCK:
        listeners = list(_COMPILE_LISTENERS)
    for listener in listeners:
        try:
            listener(str(kind or ""), str(path) if path else None, str(error) if error else None)
        except Exception:
            log.debug("compile listener failed", exc_info=True)
