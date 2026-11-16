"""Google Drive sync of Library books on the phone (Settings › Cloud sync & sharing and Library settings › Cloud).

The rules live once in the shared ``drive_sync_core`` (also used by the desktop Library): the three modes
(``compiled`` "Compiled output only", the default; ``full`` "Full output folder"; ``full_raw`` "Full output folder
+ raw source", whose source goes to ``<book folder>/RAW``), the planner, the Drive client and the background
worker. This module is the phone's glue:

* **Account**: the existing native Google sign-in of ``ShareLinkService`` (Android AuthorizationClient, iOS PKCE,
  scope ``drive.file``) and its sealed grant in ``mobile_share_links.json``. ``connect`` signs in for sync without
  turning on the Google Drive share-link site; one grant serves both, and Disconnect ends both.
* **Settings** (``enabled`` off by default, ``mode``, ``folder`` "Glossarion"), the remembered Drive ids and the
  last result live under the ``drive_sync`` key of the same sidecar (no secrets; nothing in config.json).
* **Triggers**: a Library job that reached DONE (``JobService.on_transition``; the books come from
  ``services.cloud_sync.job_books``, the device cloud sync's own rule) and "Sync to Drive now" on a book card.
  In the compiled mode only a job that produced an EPUB / PDF (or a Compile job) uploads; the full modes follow
  every Library job that changed the workspace.
* Nothing contacts Google until the user connected an account AND switched sync on (a manual Sync now needs
  the account only). Uploads run on the core's worker thread, one short log line each (``glossarion.drive.sync``),
  never raise into the job, and stop when the app is closed (``shutdown``).

Pure Python (3.10); no Flet import.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Awaitable, Callable, Mapping, Optional, Sequence

import drive_sync_core as dsc

__all__ = ["COMPILE_KINDS", "MobileDriveSync", "STORE_KEY"]

log = logging.getLogger("glossarion.drive.mobile")

#: The key of the Drive sync settings in ``mobile_share_links.json`` (the share-link service's sidecar).
STORE_KEY = "drive_sync"
#: The ``ServiceHolds`` holder of a Drive upload that a finished job started (the device sync's is "cloud").
DRIVE_HOLD = "drive"
#: Job kinds that compile: their DONE uploads in the compiled mode even when they reported no file.
COMPILE_KINDS = frozenset({"compile_epub", "compile_pdf"})
_COMPILED_SUFFIXES = (".epub", ".pdf")


def _norm(path: Any) -> str:
    text = os.fspath(path) if path else ""
    return os.path.normcase(os.path.normpath(os.path.abspath(text))) if text else ""


def _state_name(value: Any) -> str:
    return str(getattr(value, "value", value) or "")


async def _to_thread(fn: Callable[..., Any], *args: Any) -> Any:
    return await asyncio.to_thread(fn, *args)


class MobileDriveSync:
    """Drive sync for the phone over ``ShareLinkService`` (its store, its Google grant)."""

    def __init__(self, shares: Any, *, library: Any = None, run_io: Optional[Callable[..., Awaitable[Any]]] = None,
                 post: Optional[Callable[..., Any]] = None, log_line: Optional[Callable[[str], Any]] = None,
                 client_factory: Optional[Callable[[Callable[[str], str]], Any]] = None,
                 keepalive: Any = None, spawn: Optional[Callable[..., Any]] = None) -> None:
        self.shares = shares
        self.library = library
        self.run_io = run_io or _to_thread
        self.post = post
        # The device cloud sync's ``Keepalive`` under its own holder (DRIVE_HOLD): an upload a finished job
        # started joins the job's foreground service (Android) / keeps its background grant (iOS).
        self.keepalive = keepalive
        self.spawn = spawn
        self._held = False
        self._listeners: list = []
        self._seen_jobs: list = []
        self._rows: dict = {}  # book key -> the Library row the job was made from
        self._kinds: dict = {}  # book key -> compiled kinds wanted ("epub", "pdf")
        self.core = dsc.DriveSyncService(
            settings=self.sync_settings, access=self._access, load_state=self._load_state,
            save_state=self._save_state, on_result=self._on_result, log_line=log_line,
            resolve_source=self._resolve_source, resolve_compiled=self._resolve_compiled)
        if client_factory is not None:  # tests: an in-memory Drive (the HTTP client is tested in the core)
            self.core.client = lambda: client_factory(self.core.access())
        self._unsub_shares = None
        subscribe = getattr(shares, "subscribe", None)
        if callable(subscribe):
            self._unsub_shares = subscribe(self._on_share_event)

    # ---- wiring --------------------------------------------------------------------------------------

    @classmethod
    def from_app(cls, app: Any) -> Optional["MobileDriveSync"]:
        """The service for ``GlossarionApp`` (after the share links); None without them."""
        shares = getattr(app, "share_links", None)
        if shares is None:
            return None
        dispatcher = getattr(app, "dispatcher", None)
        bound = dispatcher is not None and getattr(dispatcher, "bound", False)

        async def run_in_thread(fn: Callable[..., Any], *args: Any) -> Any:
            return await dispatcher.run_in_thread(fn, *args, name="gl-drive")

        keepalive = None
        platform = str(getattr(shares, "platform", "") or "")
        if platform in ("android", "ios"):
            try:
                from glossarion_mobile.services.cloud_sync import Keepalive

                jobs_feature = getattr(app, "jobs", None)
                keepalive = Keepalive(getattr(app, "native", None), platform=platform,
                                      jobs=getattr(app, "job_service", None) or jobs_feature,
                                      background=getattr(jobs_feature, "background", None), name=DRIVE_HOLD)
            except Exception:
                log.debug("no keepalive for Drive sync", exc_info=True)
        service = cls(shares, library=getattr(app, "library", None), run_io=run_in_thread if bound else None,
                      post=getattr(dispatcher, "post", None) if bound else None, keepalive=keepalive,
                      spawn=getattr(dispatcher, "spawn", None) if bound else None)
        service.attach(getattr(app, "job_service", None) or getattr(app, "jobs", None))
        return service

    def attach(self, jobs: Any) -> None:
        on_transition = getattr(jobs, "on_transition", None) if jobs is not None else None
        if callable(on_transition):
            on_transition(self.on_job_transition)

    def subscribe(self, callback: Callable[..., Any]) -> Callable[[], None]:
        """``callback("drive")`` after a settings change, a sign-in / out, a queued sync and each result."""
        self._listeners.append(callback)

        def unsubscribe() -> None:
            try:
                self._listeners.remove(callback)
            except ValueError:
                pass

        return unsubscribe

    def _emit(self) -> None:
        def deliver() -> None:
            for callback in list(self._listeners):
                try:
                    result = callback("drive")
                    if asyncio.iscoroutine(result):
                        asyncio.ensure_future(result)
                except Exception:
                    log.exception("drive sync listener failed")

        if self.post is not None and self.post(deliver) is not False:
            return
        deliver()

    def _on_share_event(self, kind: Any = None) -> None:
        if kind == "providers":  # the Google grant came or went
            self._emit()

    def shutdown(self) -> None:
        """App closing: cancel the running upload and drop the queue."""
        self.core.shutdown(timeout=2.0)
        if callable(self._unsub_shares):
            self._unsub_shares()
            self._unsub_shares = None

    # ---- settings -------------------------------------------------------------------------------------

    @property
    def supported(self) -> bool:
        return str(getattr(self.shares, "platform", "")) in ("android", "ios")

    def _entry(self) -> dict:
        entry = (self.shares.store.read() or {}).get(STORE_KEY)
        return dict(entry) if isinstance(entry, Mapping) else {}

    def signed_in(self) -> bool:
        try:
            return bool(self.shares.provider_state("gdrive").signed_in)
        except Exception:
            return False

    def sync_settings(self) -> dsc.SyncSettings:
        entry = self._entry()
        return dsc.SyncSettings(enabled=bool(entry.get("enabled")), mode=dsc.normalize_mode(entry.get("mode")),
                                folder="/".join(dsc.folder_parts(entry.get("folder"))), signed_in=self.signed_in())

    def active(self) -> bool:
        """Sync is on and an account is connected (the card's "Sync to Drive now" shows then)."""
        return self.supported and self.sync_settings().active

    def ui_state(self) -> dict:
        """What both Cloud screens show (blocking-safe: a store read)."""
        settings = self.sync_settings()
        entry = self._entry()
        last = entry.get("last_sync")
        current = self.core.worker.current
        return {"available": True, "supported": self.supported, "signed_in": settings.signed_in,
                "enabled": settings.enabled, "mode": settings.mode, "folder": settings.folder,
                "last_sync": dict(last) if isinstance(last, Mapping) else None,
                "busy": bool(self.core.worker.busy), "current": str(getattr(current, "label", "") or "")}

    def _mutate(self, fn: Callable[[dict], Any]) -> Any:
        def change(data: dict) -> Any:
            entry = data.get(STORE_KEY)
            if not isinstance(entry, dict):
                entry = data[STORE_KEY] = {}
            return fn(entry)

        return self.shares.store.mutate(change)

    async def set_enabled(self, value: bool) -> dict:
        await self.run_io(self._mutate, lambda entry: entry.update(enabled=bool(value)))
        self._emit()
        return self.ui_state()

    async def set_mode(self, mode: str) -> dict:
        await self.run_io(self._mutate, lambda entry: entry.update(mode=dsc.normalize_mode(mode)))
        self._emit()
        return self.ui_state()

    async def set_folder(self, text: str) -> dict:
        folder = "/".join(dsc.folder_parts(text))
        await self.run_io(self._mutate, lambda entry: entry.update(folder=folder))
        self._emit()
        return self.ui_state()

    async def connect(self) -> dict:
        """Sign in with the existing native Google sign-in (the caller showed the sync consent first)."""
        await self.shares.connect_drive(for_sync=True)
        now = time.time()
        await self.run_io(self._mutate, lambda entry: entry.setdefault("consent_at", now))
        self._emit()
        return self.ui_state()

    async def disconnect(self) -> dict:
        """Disconnect the Google account; the running Drive sync is cancelled only once that worked (a
        refused disconnect, e.g. while a share-link upload runs, leaves the sync running)."""
        await self.shares.disconnect_drive()
        self.core.worker.cancel_current()
        self._emit()
        return self.ui_state()

    # ---- state the core keeps (worker thread) ----------------------------------------------------------

    def _access(self) -> Callable[[str], str]:
        return self.shares._credentials_blocking("gdrive")["access_token"]

    def _load_state(self) -> dict:
        state = self._entry().get("state")
        return dict(state) if isinstance(state, Mapping) else {}

    def _save_state(self, state: dict) -> None:
        self._mutate(lambda entry: entry.update(state=state))

    def _on_result(self, job: dsc.SyncJob, result: dsc.SyncResult) -> None:
        record = result.as_record()
        if result.error_code != "disabled":
            self._mutate(lambda entry: entry.update(last_sync=record))
        self._kinds.pop(_norm(job.book_key), None)
        if not self.core.worker.queued:
            self._release_hold()
        self._emit()

    # ---- keeping the app alive for an upload a finished job started -------------------------------------

    def _hold(self, label: str) -> None:
        """UI loop, inside the job's transition callback (before ``BackgroundExecution.job_finished``)."""
        if self.keepalive is None:
            return
        try:
            from glossarion_mobile.services.cloud_sync import _progress_text

            if self.keepalive.hold_now(_progress_text(label, "Google Drive")):
                self._held = True
        except Exception:
            log.debug("Drive sync keepalive hold failed", exc_info=True)

    def _release_hold(self) -> None:
        """Worker thread: the queue is empty, so the hold goes back (on the UI loop)."""
        if not self._held or self.keepalive is None:
            return
        self._held = False

        def release() -> None:  # on the UI loop
            try:
                result = self.keepalive.release()
                if asyncio.iscoroutine(result):
                    if self.spawn is not None:
                        self.spawn(result)
                    else:
                        asyncio.ensure_future(result)
            except Exception:
                log.debug("Drive sync keepalive release failed", exc_info=True)

        if self.post is not None and self.post(release) is not False:
            return
        try:  # no app loop (host tests): release here
            result = self.keepalive.release()
            if asyncio.iscoroutine(result):
                asyncio.run(result)
        except Exception:
            log.debug("Drive sync keepalive release failed", exc_info=True)

    # ---- books ---------------------------------------------------------------------------------------

    def _row_for(self, identity: str) -> dict:
        key = _norm(identity)
        row = self._rows.get(key)
        if row is not None:
            return dict(row)
        from glossarion_mobile.services.library import LibraryService, book_identity

        snapshot = getattr(self.library, "snapshot", None) if self.library is not None else None
        try:
            books = list(snapshot.all_books()) if snapshot is not None and hasattr(snapshot, "all_books") else []
        except Exception:
            books = []
        for book in books:
            if _norm(book_identity(book)) == key:
                return dict(book)
        return LibraryService._synthesise(identity) or {"path": identity, "name": os.path.basename(identity)}

    def _workspace(self, row: Mapping[str, Any], identity: str) -> str:
        workspace_for = getattr(self.library, "workspace_for", None) if self.library is not None else None
        folder = ""
        if callable(workspace_for):
            try:
                folder = str(workspace_for(row) or "")
            except Exception:
                folder = ""
        folder = folder or str(row.get("output_folder") or "")
        if not folder and identity and os.path.isdir(identity):
            folder = identity
        return folder

    def _resolve_compiled(self, job: dsc.SyncJob) -> list:
        """The book's newest compiled EPUB / PDF (the device cloud sync's choice rule: a Library-filed EPUB,
        the "<Title>.pdf" of Compile PDF), limited to the kinds the trigger wants."""
        from glossarion_mobile.services.cloud_sync import select_outputs

        row = dict(self._row_for(job.book_key))
        if job.output_folder and not row.get("output_folder"):
            row["output_folder"] = job.output_folder
        lister = getattr(self.library, "compiled_outputs_blocking", None) if self.library is not None else None
        choices = select_outputs(row, lister=lister)
        kinds = self._kinds.get(_norm(job.book_key)) or ("epub", "pdf")
        return [choices[kind].path for kind in kinds if kind in choices]

    def _resolve_source(self, job: dsc.SyncJob) -> str:
        raw_source = getattr(self.library, "raw_source", None) if self.library is not None else None
        if not callable(raw_source):
            return ""
        row = dict(self._row_for(job.book_key))
        if job.output_folder and not row.get("output_folder"):
            row["output_folder"] = job.output_folder
        try:
            return str(raw_source(row) or "")
        except Exception:
            return ""

    def _job(self, identity: str, row: Optional[Mapping[str, Any]], *, reason: str, compiled: Sequence[str] = (),
             kinds: Sequence[str] = ("epub", "pdf")) -> dsc.SyncJob:
        """A job for the core. ``row`` (a card's Library row) is kept for the resolvers; a DONE hook passes
        None: its identity IS the workspace, and the row is looked up on the worker thread."""
        key = _norm(identity)
        if row is not None:
            self._rows[key] = dict(row)
        # a book queued again before it ran is merged by the core's worker: keep the kinds of both
        self._kinds[key] = tuple(dict.fromkeys(tuple(self._kinds.get(key) or ()) + tuple(kinds)))
        label = str((row or {}).get("name") or os.path.basename(os.path.normpath(identity)) or "Book")
        folder = self._workspace(row, identity) if row is not None else identity
        return dsc.SyncJob(book_key=identity, label=label, reason=reason, output_folder=folder,
                           compiled_files=tuple(compiled))

    def sync_book(self, book: Mapping[str, Any]) -> bool:
        """"Sync to Drive now" on a book card: queue it (needs the account; works with the switch off too).
        False when no account is connected (nothing is sent)."""
        from glossarion_mobile.services.library import book_identity

        identity = book_identity(book)
        if not identity or not self.supported:
            return False
        queued = self.core.request_sync(self._job(identity, book, reason="manual"))
        self._emit()
        return queued

    def on_job_transition(self, snap: Any, previous: Any = None) -> None:
        """JobService transition (UI loop): a Library job reached DONE -> its books are queued when sync is on.
        Never raises; does nothing (no store write, no network) while sync is off or signed out."""
        try:
            if _state_name(getattr(snap, "state", None)) != "DONE" or previous is None:
                return
            if _state_name(previous) == "QUEUED":
                return
            job_id = str(getattr(snap, "id", "") or "")
            if job_id in self._seen_jobs:
                return
            self._seen_jobs.append(job_id)
            del self._seen_jobs[:-500]
            if not self.active():
                return
            from glossarion_mobile.services.cloud_sync import job_books

            settings = self.sync_settings()
            kind = str(getattr(getattr(snap, "spec", None), "kind", "") or "")
            queued = False
            for identity, reported, single in job_books(snap, self.library):
                parent = os.path.basename(os.path.dirname(os.path.normpath(identity)))
                if parent.casefold() == "attachments":
                    continue  # a chat book waiting for the move into the Library
                folder_key = _norm(identity)
                compiled = [p for p in reported if str(p).lower().endswith(_COMPILED_SUFFIXES)
                            and (single or _norm(os.path.dirname(p)) == folder_key)]
                if settings.mode == dsc.MODE_COMPILED and not compiled and kind not in COMPILE_KINDS:
                    continue  # nothing was compiled: nothing to update in the compiled mode
                kinds = ("pdf",) if kind == "compile_pdf" else ("epub",)
                job = self._job(identity, None, reason="compile", compiled=compiled, kinds=kinds)
                if not queued and not self._held:
                    self._hold(job.label)  # now, before the job's service / grant can end (and the upload)
                queued = self.core.on_compiled(job) or queued
            if queued:
                self._emit()
            elif self._held and not self.core.worker.busy:
                self._release_hold()
        except Exception:
            log.exception("drive sync job hook failed")

    def on_workspace_moved(self, old: str, new: str, *, merged: bool = False) -> None:
        """The chat's auto-migrate worker (thread, inside its migrate lock): a chat book moved or was merged into
        the Library (the DONE hook skipped it while it was a chat book: a chat translation is a ``direct_text``
        job). Queues the Library book like a compile when sync is on; in the compiled mode only when it has a
        compiled EPUB / PDF. Never raises; nothing happens while sync is off or signed out."""
        try:
            if not new or not self.active():
                return
            settings = self.sync_settings()
            job = self._job(str(new), None, reason="compile", kinds=("epub", "pdf"))
            if settings.mode == dsc.MODE_COMPILED:
                compiled = self._resolve_compiled(job)
                if not compiled:
                    self._kinds.pop(_norm(job.book_key), None)
                    return
                job.compiled_files = tuple(compiled)
            if self.core.on_compiled(job):
                self._emit()
        except Exception:
            log.exception("drive sync move hook failed")
