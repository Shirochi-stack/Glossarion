"""Cloud sync records: ``<data>/mobile_cloud.json`` (U10, UI_SPEC Appendix B). Pure Python.

What Glossarion has copied into the user's own cloud storage (``services/cloud_sync.py``), so a
recompile overwrites the same cloud file instead of adding a second one. Mobile-only: nothing here
is ever written to ``config.json`` and nothing is a secret (document URIs / bookmarks of files the
user picked, file names, sizes, hashes, times).

Layout::

    {version: 1,
     records: {<target id>: {<book key>: {identity, folder: {doc, name} | None, title,
                                          files: {<kind>: Record}}}},
     overrides: {<book key>: "always" | "never"},
     queue: [{key, identity, reason, attempts, next_at, last_error, added_at, manual, kinds, reported, gen}]}

* **target id**: the destination the record belongs to (``Destination.id``: a hash of the picked
  folder / the phone folder / per-file mode). Records of an old destination never steer writes to
  a new one (critic #8); "Change…" retires them.
* **book key**: the normalised identity path of the book (its output workspace, else its file) -
  never the route id (it changes when a chat book moves into the Library) and never the file name
  (titles get translated). ``relocate`` moves a book's records, override and queue entry with its
  workspace; a merge into an existing Library workspace keeps the target's documents (critic #10).
* **Record** (one per target + book + kind): ``doc`` (the provider's handle: an Android document
  URI, an iOS bookmark, a MediaStore URI), ``name`` (the cloud file's name, kept on title changes),
  ``parent`` (the book folder's handle), ``size`` / ``mtime_ns`` / ``sha1`` / ``source`` (what was
  copied last), ``synced_at``, ``remote_size``, ``mode`` (the write mode that worked), ``status``,
  ``error``, ``note`` (a warning such as "replaced: the link changed"), ``dirty`` (a write was cut
  off: rewrite even when the hash matches), ``replacing`` (the new copy of a create-new + delete-old
  replace, flushed before its first byte: a killed process leaves it torn and the next drain deletes
  it) and ``orphans`` (old copies a replace could not delete yet: deleted again on later drains).
* **queue** ``gen``: bumped by every ``enqueue``; a drain finishes / reschedules only the generation it
  took, so a trigger that arrives while the book is being copied stays queued.

Thread-safe: one ``RLock`` (``lock``) guards every read and write, including the relocation the
chat's auto-migrate runs on a worker thread, so a drain never stores a result under a key that
moved meanwhile (``resolve`` follows the session's relocations). Saving is debounced and atomic
(``Prefs.atomic_write_json`` on the ``DebouncedSaver`` worker thread), so callers on the UI loop
never block on disk; ``flush`` saves at once (call it off the loop).
"""

from __future__ import annotations

import copy
import json
import logging
import os
import threading
import time
from typing import Any, Callable, Iterable, Mapping, Optional

from glossarion_mobile.state.config_store import DebouncedSaver
from glossarion_mobile.state.prefs import atomic_write_json

__all__ = [
    "CLOUD_FILE",
    "CLOUD_VERSION",
    "CloudRecordStore",
    "KINDS",
    "OVERRIDES",
    "RECORD_STATES",
    "book_key",
    "ref_key",
]

log = logging.getLogger("glossarion.cloud.records")

CLOUD_FILE = "mobile_cloud.json"
CLOUD_VERSION = 1
#: The compiled outputs U10 copies (owner decision: EPUB, PDF, TXT, HTML; CBZ is not synced).
KINDS = ("epub", "pdf", "txt", "html")
#: Per-book override (owner decision): follow the global switch, always, never.
OVERRIDES = ("default", "always", "never")
#: Record ``status`` values.
RECORD_STATES = ("ok", "pending", "writing", "failed", "check", "needs_pick", "missing")
SAVE_DELAY = 0.25
MAX_QUEUE = 2000


def book_key(identity: Any) -> str:
    """The normalised identity path that keys a book's records ('' for nothing)."""
    text = os.fspath(identity) if identity else ""
    if not text:
        return ""
    try:
        return os.path.normcase(os.path.normpath(os.path.abspath(text)))
    except Exception:
        return str(text)


def ref_key(handle: Any) -> str:
    """A comparable key for a document handle: a native-docs ref (dict, its stable ``id``), or a plain
    URI / bookmark string ('' for nothing)."""
    if not handle:
        return ""
    if isinstance(handle, Mapping):
        for key in ("id", "document", "uri", "bookmark"):
            if handle.get(key):
                return f"{key}:{handle[key]}"
        try:
            return "json:" + json.dumps(handle, sort_keys=True, default=str)
        except Exception:
            return repr(sorted(handle.items(), key=lambda kv: str(kv[0])))
    return f"str:{handle}"


def _empty() -> dict:
    return {"version": CLOUD_VERSION, "records": {}, "overrides": {}, "queue": []}


def _clean_queue_entry(item: Any) -> Optional[dict]:
    if not isinstance(item, Mapping):
        return None
    key = str(item.get("key") or "")
    identity = str(item.get("identity") or "")
    if not key or not identity:
        return None
    try:
        attempts = max(0, int(item.get("attempts") or 0))
    except (TypeError, ValueError):
        attempts = 0
    try:
        next_at = float(item.get("next_at") or 0.0)
    except (TypeError, ValueError):
        next_at = 0.0
    try:
        added = float(item.get("added_at") or 0.0)
    except (TypeError, ValueError):
        added = 0.0
    kinds = item.get("kinds")
    kinds = [k for k in kinds if k in KINDS] if isinstance(kinds, (list, tuple)) else None
    reported = [str(p) for p in (item.get("reported") or ()) if p]
    try:
        gen = max(1, int(item.get("gen") or 1))
    except (TypeError, ValueError):
        gen = 1
    return {"key": key, "identity": identity, "reason": str(item.get("reason") or ""), "attempts": attempts,
            "next_at": next_at, "last_error": str(item.get("last_error") or ""), "added_at": added,
            "manual": bool(item.get("manual")), "kinds": kinds, "reported": reported[:64], "gen": gen}


class CloudRecordStore:
    """``mobile_cloud.json``: per-destination records, per-book overrides and the waiting queue."""

    def __init__(self, path: Any, *, clock: Callable[[], float] = time.time, save_delay: float = SAVE_DELAY) -> None:
        self.path = os.fspath(path)
        self._clock = clock
        self.lock = threading.RLock()
        self._data: dict = _empty()
        self._aliases: dict = {}  # old book key -> new book key (relocations of this session)
        #: Book keys deleted this session (``forget_book``): a copy that was already running when its book was
        #: deleted must not store the deleted book's record again (``update_record(live_only=True)``); a new
        #: trigger for the key (``enqueue``) makes it live again.
        self._forgotten: set = set()
        self._version = 0
        self._saved_version = 0
        self.loaded = False
        self.load_error: Optional[str] = None
        self.save_error: Optional[str] = None
        self.saves = 0
        self._listeners: list = []
        self._saver = DebouncedSaver(self._save_if_dirty, delay=save_delay, name="gl-cloud-save")

    # ---- persistence ----------------------------------------------------------------------------

    def load(self) -> bool:
        """Blocking: read the file (missing -> empty; unreadable -> moved aside, empty). Records left
        ``writing`` by a killed process become ``pending`` + ``dirty`` (the cloud copy may be torn, so the
        next drain rewrites it even when the local hash matches)."""
        data = _empty()
        error = None
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if not isinstance(loaded, dict):
                raise ValueError("not a JSON object")
            records = loaded.get("records")
            if isinstance(records, dict):
                data["records"] = {str(t): v for t, v in records.items() if isinstance(v, dict)}
            overrides = loaded.get("overrides")
            if isinstance(overrides, dict):
                data["overrides"] = {str(k): v for k, v in overrides.items() if v in ("always", "never")}
            queue = [_clean_queue_entry(item) for item in (loaded.get("queue") or ())]
            data["queue"] = [item for item in queue if item is not None][:MAX_QUEUE]
        except FileNotFoundError:
            pass
        except (OSError, ValueError, UnicodeDecodeError) as exc:
            error = f"{os.path.basename(self.path)} was unreadable ({type(exc).__name__}); starting fresh"
            log.warning(error)
            try:
                os.replace(self.path, f"{self.path}.corrupt-{time.strftime('%Y%m%d_%H%M%S')}")
            except OSError:
                pass
        torn = 0
        for books in data["records"].values():
            for book in books.values():
                if not isinstance(book, dict):
                    continue
                files = book.get("files")
                if not isinstance(files, dict):
                    book["files"] = {}
                    continue
                for record in files.values():
                    if isinstance(record, dict) and record.get("status") == "writing":
                        record["status"] = "pending"
                        record["dirty"] = True
                        torn += 1
        with self.lock:
            self._data = data
            self._aliases = {}
            self._forgotten = set()
            self._version += 1
            self._saved_version = self._version if not torn else self._saved_version
            self.loaded = True
            self.load_error = error
        if torn:
            self._schedule()
        return error is None

    def _save_if_dirty(self) -> bool:
        with self.lock:
            if self._version == self._saved_version:
                return False
            version = self._version
            payload = copy.deepcopy(self._data)
        payload["version"] = CLOUD_VERSION
        try:
            atomic_write_json(self.path, payload)
        except Exception as exc:
            with self.lock:
                self.save_error = f"{type(exc).__name__}: {exc}"
            log.error("saving %s failed: %s", CLOUD_FILE, exc)
            return False
        with self.lock:
            self._saved_version = max(self._saved_version, version)
            self.save_error = None
            self.saves += 1
        return True

    def _schedule(self) -> None:
        self._saver.schedule()

    def _changed(self) -> None:
        # caller holds the lock
        self._version += 1

    def flush(self) -> bool:
        """Blocking: save now (off the UI loop)."""
        self._saver.cancel()
        return self._save_if_dirty()

    def wait_idle(self, timeout: Optional[float] = None) -> bool:
        return self._saver.wait_idle(timeout)

    def close(self) -> None:
        try:
            self.flush()
        finally:
            self._saver.close()

    def delete_file(self) -> None:
        """Blocking: forget everything and remove the file (wipe)."""
        with self.lock:
            self._data = _empty()
            self._aliases = {}
            self._forgotten = set()
            self._version += 1
            self._saved_version = self._version
        self._saver.cancel()
        # a save already running (the saver thread) would write the file back after the remove (CI flake)
        self._saver.wait_idle(5.0)
        try:
            os.remove(self.path)
        except FileNotFoundError:
            pass
        except OSError as exc:
            log.info("could not remove %s: %s", CLOUD_FILE, exc)

    def subscribe(self, callback: Callable[[], Any]) -> Callable[[], None]:
        """``callback()`` after every change (on the thread that made it)."""
        self._listeners.append(callback)

        def remove() -> None:
            if callback in self._listeners:
                self._listeners.remove(callback)

        return remove

    def _notify(self) -> None:
        for callback in list(self._listeners):
            try:
                callback()
            except Exception:
                log.exception("cloud records listener failed")

    def _commit(self) -> None:
        # caller holds the lock
        self._changed()
        self._schedule()

    def snapshot(self) -> dict:
        with self.lock:
            return copy.deepcopy(self._data)

    # ---- keys / relocation --------------------------------------------------------------------------

    def resolve(self, key: str) -> str:
        """``key`` after the relocations of this session (a write that finishes after its book moved
        stores its result under the new key)."""
        with self.lock:
            seen = set()
            while key in self._aliases and key not in seen:
                seen.add(key)
                key = self._aliases[key]
            return key

    def relocate(self, old_identity: Any, new_identity: Any, *, merge: bool = False) -> list:
        """Move a book's records, override and queue entry to its new workspace (auto-migrate MOVED,
        merge into an existing Library book). With ``merge`` (or whenever the target already has a
        record of a kind), the target's documents win and the moved book's are dropped; returns the
        dropped ``(target_id, doc)`` pairs so per-file grants can be released."""
        old, new = book_key(old_identity), book_key(new_identity)
        if not old or not new or old == new:
            return []
        dropped: list = []
        with self.lock:
            records = self._data["records"]
            for target_id, books in records.items():
                moving = books.pop(old, None)
                if not isinstance(moving, dict):
                    continue
                existing = books.get(new)
                if not isinstance(existing, dict):
                    moving["identity"] = os.path.abspath(os.fspath(new_identity))
                    books[new] = moving
                    continue
                files = existing.setdefault("files", {})
                for kind, record in (moving.get("files") or {}).items():
                    if kind in files:
                        if isinstance(record, dict) and record.get("doc"):
                            dropped.append((target_id, record.get("doc")))
                    else:
                        files[kind] = record
                if not existing.get("folder") and moving.get("folder"):
                    existing["folder"] = moving.get("folder")
            overrides = self._data["overrides"]
            if old in overrides:
                value = overrides.pop(old)
                if new not in overrides or not merge:
                    overrides.setdefault(new, value)
            queue = self._data["queue"]
            moved_entry = None
            for item in list(queue):
                if item["key"] == old:
                    queue.remove(item)
                    moved_entry = item
            if moved_entry is not None:
                target = next((i for i in queue if i["key"] == new), None)
                if target is None:
                    moved_entry.update(key=new, identity=os.path.abspath(os.fspath(new_identity)))
                    queue.append(moved_entry)
                else:
                    target["manual"] = target["manual"] or moved_entry["manual"]
                    target["next_at"] = min(target["next_at"], moved_entry["next_at"])
                    target["gen"] = int(target.get("gen") or 1) + 1  # new work for a drain that holds the old one
            self._aliases[old] = new
            self._commit()
        self._notify()
        return dropped

    # ---- records ------------------------------------------------------------------------------------

    def book(self, target_id: str, key: str) -> Optional[dict]:
        with self.lock:
            entry = self._data["records"].get(target_id, {}).get(self.resolve(key))
            return copy.deepcopy(entry) if isinstance(entry, dict) else None

    def books(self, target_id: str) -> dict:
        with self.lock:
            return copy.deepcopy(self._data["records"].get(target_id) or {})

    def record(self, target_id: str, key: str, kind: str) -> Optional[dict]:
        with self.lock:
            entry = self._data["records"].get(target_id, {}).get(self.resolve(key))
            record = (entry or {}).get("files", {}).get(kind) if isinstance(entry, dict) else None
            return copy.deepcopy(record) if isinstance(record, dict) else None

    def _book_entry(self, target_id: str, key: str, identity: Any = None) -> dict:
        # caller holds the lock
        books = self._data["records"].setdefault(target_id, {})
        entry = books.get(key)
        if not isinstance(entry, dict):
            entry = books[key] = {"identity": os.path.abspath(os.fspath(identity)) if identity else key,
                                  "folder": None, "title": "", "files": {}}
        entry.setdefault("files", {})
        if identity:
            entry["identity"] = os.path.abspath(os.fspath(identity))
        return entry

    def update_record(self, target_id: str, key: str, kind: str, identity: Any = None, *, live_only: bool = False,
                      **fields: Any) -> dict:
        """Merge ``fields`` into the record (created when missing) under the key's current name. With
        ``live_only`` (a copy finishing in the drain) nothing is stored for a book deleted meanwhile ({})."""
        with self.lock:
            key = self.resolve(key)
            if live_only and key in self._forgotten:
                return {}
            entry = self._book_entry(target_id, key, identity)
            record = entry["files"].setdefault(kind, {})
            record.update(copy.deepcopy(fields))
            result = copy.deepcopy(record)
            self._commit()
        self._notify()
        return result

    def drop_record(self, target_id: str, key: str, kind: str) -> Optional[dict]:
        with self.lock:
            key = self.resolve(key)
            entry = self._data["records"].get(target_id, {}).get(key)
            if not isinstance(entry, dict):
                return None
            record = entry.get("files", {}).pop(kind, None)
            self._commit()
        self._notify()
        return record

    def set_book_folder(self, target_id: str, key: str, folder: Optional[Mapping[str, Any]], identity: Any = None,
                        title: Optional[str] = None, *, live_only: bool = False) -> None:
        with self.lock:
            key = self.resolve(key)
            if live_only and key in self._forgotten:
                return
            entry = self._book_entry(target_id, key, identity)
            entry["folder"] = dict(folder) if folder else None
            if title is not None:
                entry["title"] = str(title)
            self._commit()
        self._notify()

    def claimed(self, target_id: str, *, parent: Any = None, exclude_key: str = "") -> dict:
        """``{name.casefold(): book key}`` of the cloud names other books' records use: the files under
        ``parent`` (a folder handle), or with ``parent='__root__'`` the book folder names. A name
        another book holds is never adopted (two books with the same title get ' (2)')."""
        out: dict = {}
        wanted = ref_key(parent)
        with self.lock:
            exclude = self.resolve(exclude_key) if exclude_key else ""
            for key, entry in (self._data["records"].get(target_id) or {}).items():
                if key == exclude or not isinstance(entry, dict):
                    continue
                if parent == "__root__":
                    folder = entry.get("folder") or {}
                    if folder.get("name"):
                        out[str(folder["name"]).casefold()] = key
                    continue
                for record in (entry.get("files") or {}).values():
                    if isinstance(record, dict) and record.get("name") and ref_key(record.get("parent")) == wanted:
                        out[str(record["name"]).casefold()] = key
        return out

    def docs(self, target_id: str, key: Optional[str] = None) -> list:
        """Every stored document handle of a destination (one book's with ``key``)."""
        out: list = []
        with self.lock:
            books = self._data["records"].get(target_id) or {}
            keys = [self.resolve(key)] if key else list(books)
            for k in keys:
                entry = books.get(k)
                for record in ((entry or {}).get("files") or {}).values() if isinstance(entry, dict) else ():
                    if isinstance(record, dict) and record.get("doc"):
                        out.append(record["doc"])
        return out

    def leftovers(self, target_id: str) -> list:
        """``[(key, kind, record)]`` of a destination's records that still hold cloud copies to delete: the
        old copy a replace could not delete (``orphans``) or a replacement a killed process left half
        written (``replacing``)."""
        out: list = []
        with self.lock:
            for key, entry in (self._data["records"].get(target_id) or {}).items():
                if not isinstance(entry, dict):
                    continue
                for kind, record in (entry.get("files") or {}).items():
                    if isinstance(record, dict) and (record.get("orphans") or record.get("replacing")):
                        out.append((key, kind, copy.deepcopy(record)))
        return out

    def drop_target(self, target_id: str) -> list:
        """Forget a destination's records; returns their document handles (per-file grants to release)."""
        docs = self.docs(target_id)
        with self.lock:
            if self._data["records"].pop(target_id, None) is not None:
                self._commit()
        self._notify()
        return docs

    def forget_book(self, identity: Any) -> list:
        """A deleted Library book: drop its records (every destination), override and queue entry.
        Returns ``(target_id, doc)`` pairs (per-file grants to release); the cloud files stay."""
        key = self.resolve(book_key(identity))
        dropped: list = []
        changed = False
        with self.lock:
            if key:
                self._forgotten.add(key)
            for target_id, books in self._data["records"].items():
                entry = books.pop(key, None)
                if isinstance(entry, dict):
                    changed = True
                    for record in (entry.get("files") or {}).values():
                        if isinstance(record, dict) and record.get("doc"):
                            dropped.append((target_id, record["doc"]))
            if self._data["overrides"].pop(key, None) is not None:
                changed = True
            before = len(self._data["queue"])
            self._data["queue"] = [i for i in self._data["queue"] if i["key"] != key]
            changed = changed or len(self._data["queue"]) != before
            if changed:
                self._commit()
        if changed:
            self._notify()
        return dropped

    # ---- overrides ------------------------------------------------------------------------------------

    def override(self, identity: Any) -> str:
        with self.lock:
            return str(self._data["overrides"].get(self.resolve(book_key(identity))) or "default")

    def set_override(self, identity: Any, value: str) -> str:
        if value not in OVERRIDES:
            raise ValueError(f"unknown override {value!r}")
        key = self.resolve(book_key(identity))
        with self.lock:
            current = self._data["overrides"].get(key)
            if value == "default":
                if current is None:
                    return value
                del self._data["overrides"][key]
            else:
                if current == value:
                    return value
                self._data["overrides"][key] = value
            self._commit()
        self._notify()
        return value

    def overrides(self) -> dict:
        with self.lock:
            return dict(self._data["overrides"])

    # ---- queue ----------------------------------------------------------------------------------------

    def enqueue(self, identity: Any, reason: str, *, manual: bool = False, reported: Iterable[str] = (),
                kinds: Optional[Iterable[str]] = None, now: Optional[float] = None) -> dict:
        """Add (or refresh) a book's queue entry: due now, attempts reset (a new trigger is a fresh
        start). Each book appears once; a manual entry stays manual. Every call bumps the entry's ``gen``:
        a drain that took the entry before this trigger leaves it queued (``finish`` / ``reschedule`` with
        the old ``gen`` do nothing), so a recompile or a format turned on while the book was being copied
        is never lost."""
        key = self.resolve(book_key(identity))
        now = self._clock() if now is None else now
        wanted = [k for k in kinds if k in KINDS] if kinds is not None else None
        with self.lock:
            self._forgotten.discard(key)
            queue = self._data["queue"]
            entry = next((i for i in queue if i["key"] == key), None)
            if entry is None:
                entry = {"key": key, "identity": os.path.abspath(os.fspath(identity)), "reason": reason,
                         "attempts": 0, "next_at": now, "last_error": "", "added_at": now, "manual": bool(manual),
                         "kinds": wanted, "reported": [], "gen": 1}
                queue.append(entry)
                del queue[:-MAX_QUEUE]
            else:
                entry.update(reason=reason, attempts=0, next_at=min(entry["next_at"], now), last_error="",
                             identity=os.path.abspath(os.fspath(identity)), gen=int(entry.get("gen") or 1) + 1)
                entry["manual"] = entry["manual"] or bool(manual)
                if entry.get("kinds") is not None:
                    entry["kinds"] = None if wanted is None else sorted(set(entry["kinds"]) | set(wanted))
            merged = list(entry.get("reported") or [])
            for path in reported or ():
                if path and path not in merged:
                    merged.append(str(path))
            entry["reported"] = merged[-64:]
            result = copy.deepcopy(entry)
            self._commit()
        self._notify()
        return result

    def queue(self) -> list:
        with self.lock:
            return copy.deepcopy(self._data["queue"])

    def queued(self, identity: Any) -> Optional[dict]:
        key = self.resolve(book_key(identity))
        with self.lock:
            entry = next((i for i in self._data["queue"] if i["key"] == key), None)
            return copy.deepcopy(entry) if entry is not None else None

    def due(self, now: Optional[float] = None, *, skip: Iterable[str] = ()) -> Optional[dict]:
        """The oldest entry that is due (``next_at <= now``), skipping ``skip`` keys."""
        now = self._clock() if now is None else now
        skipped = set(skip)
        with self.lock:
            for entry in self._data["queue"]:
                if entry["key"] not in skipped and entry["next_at"] <= now:
                    return copy.deepcopy(entry)
        return None

    def next_due_at(self, *, skip: Iterable[str] = ()) -> Optional[float]:
        skipped = set(skip)
        with self.lock:
            times = [i["next_at"] for i in self._data["queue"] if i["key"] not in skipped]
        return min(times) if times else None

    def finish(self, key: str, gen: Optional[int] = None) -> bool:
        """Remove a book's entry (done, dropped). With ``gen`` only the entry the caller took: a trigger that
        came in meanwhile (another ``gen``) keeps it queued and due."""
        key = self.resolve(key)
        with self.lock:
            before = len(self._data["queue"])
            self._data["queue"] = [i for i in self._data["queue"]
                                   if i["key"] != key or (gen is not None and int(i.get("gen") or 1) != int(gen))]
            if len(self._data["queue"]) == before:
                return False
            self._commit()
        self._notify()
        return True

    def reschedule(self, key: str, next_at: float, *, attempts: Optional[int] = None, error: str = "",
                   reason: Optional[str] = None, gen: Optional[int] = None) -> Optional[dict]:
        """Move a book's entry to ``next_at`` (a retry / backoff). With ``gen``: None (unchanged) when a newer
        trigger came in meanwhile - it stays due now with a fresh start."""
        key = self.resolve(key)
        with self.lock:
            entry = next((i for i in self._data["queue"] if i["key"] == key), None)
            if entry is None:
                return None
            if gen is not None and int(entry.get("gen") or 1) != int(gen):
                return None
            entry["next_at"] = float(next_at)
            if attempts is not None:
                entry["attempts"] = max(0, int(attempts))
            entry["last_error"] = str(error or "")
            if reason is not None:
                entry["reason"] = reason
            result = copy.deepcopy(entry)
            self._commit()
        self._notify()
        return result

    def wake(self, *, errors: Iterable[str] = (), all_entries: bool = False, now: Optional[float] = None) -> int:
        """Make the entries whose ``last_error`` is one of ``errors`` (every entry with ``all_entries``) due
        now: a job ended (the books the busy guard held back), "Retry now". Returns how many."""
        wanted = set(errors)
        now = self._clock() if now is None else now
        count = 0
        with self.lock:
            for entry in self._data["queue"]:
                if (all_entries or entry.get("last_error") in wanted) and entry["next_at"] > now:
                    entry["next_at"] = now
                    count += 1
            if count:
                self._commit()
        if count:
            self._notify()
        return count

    def clear_queue(self) -> int:
        with self.lock:
            count = len(self._data["queue"])
            if count:
                self._data["queue"] = []
                self._commit()
        if count:
            self._notify()
        return count
