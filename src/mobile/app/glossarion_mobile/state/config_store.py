"""MobileConfigStore: the app's view of the shared ``config.json`` (plan §4 services, UI_SPEC §4.15).

It wraps the shared ``src/config_store.py`` (the desktop's own load/save code)
and adds what a phone needs on top:

* **Reads + decrypts only.** ``load()`` calls ``config_store.load_config``;
  nothing is sanitized, migrated or filled with defaults. Sanitizers and
  migrations run inside ``HeadlessOwner`` on the job thread, exactly as the
  desktop runs them in ``TranslatorGUI.__init__``.
* **Schema defaults are display-only.** ``effective(key)`` falls back to the
  ``defaults`` provider (the Settings UI passes the schema's
  ``effective_default``), but a default is never written to disk. An absent
  key therefore behaves like a fresh desktop install, and an imported desktop
  config behaves exactly like on desktop.
* **Sparse writes.** Only keys the user changed differ on disk: unchanged
  encrypted values (``ENC:...`` in ``api_key`` and the key-pool lists) are
  written back byte-for-byte instead of being re-encrypted, every other key
  keeps its value and position. When nothing changed nothing is written, so a
  desktop config round-trips byte-identically. Nested settings
  (``qa_scanner_settings`` / ``manga_settings`` / ``ai_hunter_config``) are
  addressed with path tuples, so one value changes and its siblings stay.
* **Debounced atomic saves.** ``set()`` schedules a save 600 ms after the last
  change on a daemon worker (``config_store.save_config_file`` = encrypt +
  ``app_paths._atomic_json_write``). ``flush()`` saves synchronously on the
  caller's thread (lifecycle INACTIVE/HIDE, job start). The first save of a
  session also makes the usual 72 h ``config_backups`` copy (desktop backs up
  on every save; per-keystroke backups would flood the folder).
* ``snapshot()`` is a deep copy for jobs (the HeadlessOwner is built from it on
  the job thread); per-key observers; ``set_job_running`` drives the
  "Changes apply to the next run" banner.

Observers run synchronously on the thread that changed the value (the UI loop
for edits made in Settings); save listeners run on the saving thread, so UI
code must marshal them through ``UiDispatcher``.

Pure Python (Python 3.10 compatible); never imports Flet. Backend modules
(``config_store``, ``api_key_encryption``) are imported lazily, after the
bootstrap put the backend on ``sys.path`` and the SecureStorage key was
handed to ``api_key_encryption.set_key_material``.
"""

from __future__ import annotations

import copy
import logging
import os
import threading
import time
from typing import Any, Callable, Iterable, Iterator, Mapping, Optional, Union

__all__ = [
    "DEFAULT_DEBOUNCE",
    "DebouncedSaver",
    "MISSING",
    "MobileConfigStore",
    "default_config_path",
    "same_value",
]

log = logging.getLogger("glossarion.config")

DEFAULT_DEBOUNCE = 0.6  # seconds (UI_SPEC §4.15: "Editing auto-saves with a 600 ms debounce")


class _Missing:
    """Sentinel for "key not in config.json" (observers receive it on ``unset``)."""

    _instance: Optional["_Missing"] = None

    def __new__(cls) -> "_Missing":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "MISSING"

    def __bool__(self) -> bool:
        return False


MISSING: Any = _Missing()

Observer = Callable[[str, Any], Any]
PathLike = Union[str, "os.PathLike[str]"]


KeyPath = Union[str, tuple]


def _as_path(key: Any) -> tuple:
    """``"model"`` -> ``("model",)``; tuples are paths into nested objects."""
    if isinstance(key, tuple):
        if not key or not all(isinstance(part, str) and part for part in key):
            raise ValueError(f"invalid config path {key!r}")
        return key
    return (str(key),)


def _lookup(data: Any, path: tuple) -> Any:
    node = data
    for part in path:
        if not isinstance(node, dict) or part not in node:
            return MISSING
        node = node[part]
    return node


def same_value(old: Any, new: Any) -> bool:
    """Equality that also compares types (JSON writes ``1`` and ``true``, ``5`` and ``5.0`` differently)."""
    if old is new:
        return True
    if type(old) is not type(new):
        return False
    if isinstance(old, dict):
        return old.keys() == new.keys() and all(same_value(old[k], new[k]) for k in old)
    if isinstance(old, (list, tuple)):
        return len(old) == len(new) and all(same_value(a, b) for a, b in zip(old, new))
    try:
        return bool(old == new)
    except Exception:
        return False


def default_config_path() -> str:
    """``CONFIG_FILE`` from the bootstrap env contract, else the shared ``app_paths`` location."""
    explicit = os.environ.get("CONFIG_FILE", "").strip()
    if explicit:
        return explicit
    import app_paths  # backend module (on sys.path after bootstrap)

    return app_paths.config_file_path()


class DebouncedSaver:
    """Runs ``save()`` on a daemon worker ``delay`` seconds after the last ``schedule()``.

    ``flush`` callers run their own save synchronously after ``cancel()``; the
    owner serialises the two with its own write lock.
    """

    def __init__(
        self,
        save: Callable[[], Any],
        *,
        delay: float = DEFAULT_DEBOUNCE,
        name: str = "gl-save",
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._save = save
        self.delay = max(0.0, float(delay))
        self.name = name
        self._clock = clock
        self._cond = threading.Condition()
        self._deadline: Optional[float] = None
        self._thread: Optional[threading.Thread] = None
        self._closed = False
        self._busy = False
        self.runs = 0

    @property
    def pending(self) -> bool:
        with self._cond:
            return self._deadline is not None

    @property
    def busy(self) -> bool:
        with self._cond:
            return self._busy

    def schedule(self) -> None:
        with self._cond:
            if self._closed:
                return
            self._deadline = self._clock() + self.delay
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._run, name=self.name, daemon=True)
                self._thread.start()
            self._cond.notify_all()

    def cancel(self) -> None:
        with self._cond:
            self._deadline = None
            self._cond.notify_all()

    def wait_idle(self, timeout: Optional[float] = None) -> bool:
        """Wait until nothing is scheduled and no save is running (tests, shutdown)."""
        end = None if timeout is None else self._clock() + timeout
        with self._cond:
            while self._deadline is not None or self._busy:
                remaining = None if end is None else end - self._clock()
                if remaining is not None and remaining <= 0:
                    return False
                self._cond.wait(0.05 if remaining is None else min(0.05, remaining))
            return True

    def close(self, timeout: float = 5.0) -> None:
        with self._cond:
            self._closed = True
            self._deadline = None
            self._cond.notify_all()
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout)

    def _run(self) -> None:
        while True:
            with self._cond:
                while not self._closed and self._deadline is None:
                    self._cond.wait()
                if self._closed:
                    return
                remaining = self._deadline - self._clock()
                if remaining > 0:
                    self._cond.wait(remaining)
                    continue
                self._deadline = None
                self._busy = True
            try:
                self.runs += 1
                self._save()
            except Exception:  # the save function reports its own errors
                log.exception("%s: debounced save failed", self.name)
            finally:
                with self._cond:
                    self._busy = False
                    self._cond.notify_all()


def _secret_fields() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """(top-level encrypted fields, key-pool list fields) as ``api_key_encryption`` defines them."""
    try:
        import api_key_encryption

        handler = api_key_encryption.get_handler()
        plain = tuple(getattr(handler, "api_key_fields", ()) or ())
        lists_fn = getattr(handler, "multi_key_list_fields", None)
        if lists_fn is None:  # _NullHandler: same names as the real handler
            lists_fn = getattr(api_key_encryption.APIKeyEncryption, "multi_key_list_fields", None)
            lists = tuple(lists_fn(handler)) if lists_fn is not None else ()
        else:
            lists = tuple(lists_fn())
        return plain, lists
    except Exception as exc:  # pragma: no cover - backend missing
        log.debug("api_key_encryption unavailable: %s", exc)
        return (), ()


def _is_enc(value: Any) -> bool:
    return isinstance(value, str) and value.startswith("ENC:")


class MobileConfigStore:
    """Thread-safe, observable, sparsely-written ``config.json``.

    ``path=None`` resolves ``CONFIG_FILE`` (bootstrap env contract) at load time.
    ``defaults(dotted_key)`` supplies display defaults (none when omitted).
    ``writer``/``reader`` default to the shared ``config_store`` functions and
    exist for tests.
    """

    def __init__(
        self,
        path: Optional[PathLike] = None,
        *,
        debounce: float = DEFAULT_DEBOUNCE,
        defaults: Optional[Callable[[str], Any]] = None,
        backup_first_save: bool = True,
        writer: Optional[Callable[..., Any]] = None,
        reader: Optional[Callable[..., dict]] = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._path: Optional[str] = os.fspath(path) if path is not None else None
        self._lock = threading.RLock()
        self._write_lock = threading.Lock()
        self._data: dict = {}
        self._raw: dict = {}  # config.json as on disk (still encrypted)
        self._baseline: dict = {}  # decrypted form of what is on disk
        self._version = 0
        self._saved_version = 0
        self._loaded = False
        self._exists = False
        self._read_only = False
        self._backed_up = not backup_first_save
        self._defaults = defaults
        self._writer = writer
        self._reader = reader
        self._observers: dict[str, list[Observer]] = {}
        self._all_observers: list[Observer] = []
        self._job_observers: list[Callable[[bool], Any]] = []
        self._save_listeners: list[Callable[[bool, Optional[str]], Any]] = []
        self._job_running = False
        self._changed_during_job: set[str] = set()
        self.load_error: Optional[str] = None
        self.save_error: Optional[str] = None
        self.save_count = 0
        self.last_saved_at: Optional[float] = None
        self.corrupt_backup: Optional[str] = None
        self._saver = DebouncedSaver(self._save_if_dirty, delay=debounce, name="gl-config-save", clock=clock)

    # ---- paths / status ---------------------------------------------------------------

    @property
    def path(self) -> str:
        if self._path is None:
            self._path = os.fspath(default_config_path())
        return self._path

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def exists(self) -> bool:
        """config.json existed at load time (or has been written since)."""
        return self._exists

    @property
    def read_only(self) -> bool:
        """The file could not be read (permissions); saves are refused so it is never clobbered."""
        return self._read_only

    @property
    def dirty(self) -> bool:
        with self._lock:
            return self._version != self._saved_version and not same_value(self._data, self._baseline)

    @property
    def save_pending(self) -> bool:
        return self._saver.pending or self._saver.busy

    @property
    def debounce(self) -> float:
        return self._saver.delay

    # ---- load ----------------------------------------------------------------------------

    def _read(self, path: str, *, decrypt: bool) -> dict:
        if self._reader is not None:
            return self._reader(path, decrypt=decrypt)
        import config_store

        return config_store.load_config(path, decrypt=decrypt)

    def load(self) -> dict:
        """Read and decrypt config.json (blocking; call it off the UI loop). Returns a snapshot.

        A missing file is an empty config (fresh install). A file that is not a
        JSON object is copied into ``config_backups`` first (so it can be
        restored from Backup & restore) and then treated as empty, like the
        desktop's ``except: config = {}``. A file that cannot be read at all
        makes the store read-only.
        """
        path = self.path
        raw: dict = {}
        data: dict = {}
        exists = False
        read_only = False
        error: Optional[str] = None
        corrupt_backup: Optional[str] = None
        try:
            raw = self._read(path, decrypt=False)
            data = self._read(path, decrypt=True)
            exists = True
            if not isinstance(raw, dict) or not isinstance(data, dict):
                raise ValueError("config.json does not contain a JSON object")
        except FileNotFoundError:
            raw, data = {}, {}
        except (ValueError, UnicodeDecodeError) as exc:  # json.JSONDecodeError is a ValueError
            exists = True
            raw, data = {}, {}
            error = f"config.json could not be parsed ({type(exc).__name__}: {exc}); starting from defaults"
            corrupt_backup = self._backup_corrupt(path)
        except OSError as exc:
            exists = os.path.exists(path)
            raw, data = {}, {}
            read_only = exists
            error = f"config.json could not be read ({type(exc).__name__}: {exc})"
        if error:
            log.error(error)
        with self._lock:
            old = self._data
            self._raw = raw
            self._data = data
            self._baseline = copy.deepcopy(data)
            self._version += 1
            self._saved_version = self._version
            self._loaded = True
            self._exists = exists
            self._read_only = read_only
            self.load_error = error
            self.corrupt_backup = corrupt_backup
            changed = self._changed_keys(old, data)
        self._saver.cancel()
        for key in changed:
            self._notify(key, self._data.get(key, MISSING))
        return self.snapshot()

    def reload(self) -> dict:
        """Re-read config.json (after a restore/import); unsaved edits are dropped."""
        return self.load()

    def _backup_corrupt(self, path: str) -> Optional[str]:
        try:
            import config_store

            return config_store.backup_config_file(path)
        except Exception as exc:
            log.warning("could not back up the unreadable config.json: %s", exc)
            return None

    @staticmethod
    def _changed_keys(old: Mapping[str, Any], new: Mapping[str, Any]) -> list[str]:
        keys = list(old.keys()) + [k for k in new.keys() if k not in old]
        return [k for k in keys if (k in old) != (k in new) or (k in old and not same_value(old[k], new[k]))]

    # ---- read ----------------------------------------------------------------------------

    def get(self, key: KeyPath, default: Any = None) -> Any:
        """The value stored in config.json (a copy), or ``default`` when absent.

        ``key`` is a top-level key or a path tuple into a nested object, e.g.
        ``("qa_scanner_settings", "min_file_length")``.
        """
        with self._lock:
            value = _lookup(self._data, _as_path(key))
            return default if value is MISSING else copy.deepcopy(value)

    def has(self, key: KeyPath) -> bool:
        with self._lock:
            return _lookup(self._data, _as_path(key)) is not MISSING

    __contains__ = has

    def keys(self) -> list[str]:
        with self._lock:
            return list(self._data.keys())

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())

    def default_for(self, key: KeyPath) -> Any:
        """Display default for ``key`` (schema effective default); never written to disk.

        Nested settings use their dotted schema key (``qa_scanner_settings.min_file_length``).
        """
        provider = self._defaults
        if provider is None:  # no schema wired in: no display defaults
            return None
        try:
            return provider(".".join(_as_path(key)))
        except Exception:
            return None

    def effective(self, key: KeyPath) -> Any:
        """What the Settings UI shows: the stored value, else the schema default (display only)."""
        with self._lock:
            value = _lookup(self._data, _as_path(key))
            if value is not MISSING:
                return copy.deepcopy(value)
        return self.default_for(key)

    def is_modified(self, key: KeyPath) -> bool:
        """The key is stored and differs from its display default (the tile's "modified" dot)."""
        with self._lock:
            value = _lookup(self._data, _as_path(key))
        if value is MISSING:
            return False
        return not same_value(value, self.default_for(key))

    def snapshot(self) -> dict:
        """Deep copy of the decrypted config (jobs build their HeadlessOwner from it)."""
        with self._lock:
            return copy.deepcopy(self._data)

    def undecryptable_keys(self) -> list[str]:
        """Encrypted fields that are still ``ENC:...`` after decryption (key lost / desktop key missing)."""
        plain, lists = _secret_fields()
        out: list[str] = []
        with self._lock:
            for field in plain:
                if _is_enc(self._data.get(field)):
                    out.append(field)
            for field in lists:
                entries = self._data.get(field)
                if isinstance(entries, list) and any(isinstance(e, dict) and _is_enc(e.get("api_key")) for e in entries):
                    out.append(field)
        return out

    # ---- write ---------------------------------------------------------------------------

    def _assign(self, path: tuple[str, ...], value: Any) -> bool:
        # caller holds self._lock; returns True when the stored value changed
        old = _lookup(self._data, path)
        if old is not MISSING and same_value(old, value):
            return False
        top = path[0]
        if len(path) == 1:
            self._data[top] = copy.deepcopy(value)
        else:
            container = self._data.get(top, MISSING)
            if container is MISSING:
                container = {}
            elif not isinstance(container, dict):
                raise ValueError(f"config.json key {top!r} is not an object; cannot set {'.'.join(path)}")
            else:
                container = copy.deepcopy(container)
            node = container
            for part in path[1:-1]:
                child = node.get(part, MISSING)
                if child is MISSING:
                    child = node[part] = {}
                elif not isinstance(child, dict):
                    raise ValueError(f"{'.'.join(path)}: {part!r} is not an object")
                node = child
            node[path[-1]] = copy.deepcopy(value)
            self._data[top] = container
        self._touch(top)
        return True

    def _notify_path(self, path: tuple[str, ...], value: Any) -> None:
        top = path[0]
        if len(path) > 1:
            self._notify(".".join(path), value)
            with self._lock:
                top_value = copy.deepcopy(self._data.get(top, MISSING))
            self._notify(top, top_value)
        else:
            self._notify(top, value)

    def set(self, key: KeyPath, value: Any) -> bool:
        """Store ``value`` (sparse: only this key changes); returns False when it was already set.

        A path tuple updates one value inside a nested object (created when absent)
        and leaves its siblings untouched.
        """
        path = _as_path(key)
        with self._lock:
            if not self._assign(path, value):
                return False
        self._notify_path(path, value)
        self._saver.schedule()
        return True

    def set_many(self, values: Mapping[Any, Any]) -> list[Any]:
        changed: list[Any] = []
        with self._lock:
            for key, value in values.items():
                if self._assign(_as_path(key), value):
                    changed.append(key)
        for key in changed:
            self._notify_path(_as_path(key), values[key])
        if changed:
            self._saver.schedule()
        return changed

    def unset(self, key: KeyPath) -> bool:
        """Remove ``key`` ("Reset to default": the owner then uses its fresh-install default).

        For a nested path only that value is removed; objects left empty are removed
        too when they did not exist in the file before, so set-then-reset writes nothing.
        """
        path = _as_path(key)
        top = path[0]
        with self._lock:
            if _lookup(self._data, path) is MISSING:
                return False
            if len(path) == 1:
                del self._data[top]
            else:
                container = copy.deepcopy(self._data[top])
                nodes = [container]
                for part in path[1:-1]:
                    nodes.append(nodes[-1][part])
                del nodes[-1][path[-1]]
                for depth in range(len(path) - 1, 0, -1):  # prune objects this edit emptied
                    node = nodes[depth - 1]
                    if node or _lookup(self._baseline, path[:depth]) is not MISSING:
                        break
                    if depth == 1:
                        container = MISSING
                    else:
                        del nodes[depth - 2][path[depth - 1]]
                if container is MISSING:
                    del self._data[top]
                else:
                    self._data[top] = container
            self._touch(top)
        self._notify_path(path, MISSING)
        self._saver.schedule()
        return True

    def revert_to(self, snapshot: Mapping[str, Any]) -> list[str]:
        """Replace the config with ``snapshot`` ("Discard changes since opening")."""
        new = copy.deepcopy(dict(snapshot))
        with self._lock:
            changed = self._changed_keys(self._data, new)
            if not changed:
                return []
            self._data = new
            for key in changed:
                self._touch(key)
        for key in changed:
            self._notify(key, new.get(key, MISSING))
        self._saver.schedule()
        return changed

    def _touch(self, key: str) -> None:
        # caller holds self._lock
        self._version += 1
        if self._job_running:
            self._changed_during_job.add(key)

    def _disk_form(self, data: dict) -> dict:
        """``data`` with unchanged encrypted values restored to their on-disk ``ENC:`` form.

        ``api_key_encryption.encrypt_config`` skips values that already start
        with ``ENC:`` (and pool entries whose ``api_key`` does), so unchanged
        secrets are written back byte-for-byte instead of being re-encrypted.
        """
        out = dict(data)
        plain, lists = _secret_fields()
        for field in plain + lists:
            if field in out and field in self._raw and field in self._baseline:
                if same_value(out[field], self._baseline[field]):
                    out[field] = copy.deepcopy(self._raw[field])
        return out

    def _write(self, disk: dict, path: str, backup: bool) -> None:
        if self._writer is not None:
            self._writer(disk, path, backup=backup)
            return
        import config_store

        config_store.save_config_file(disk, path, backup=backup)

    def _save_if_dirty(self) -> bool:
        with self._write_lock:
            with self._lock:
                if self._version == self._saved_version:
                    return False
                if self._read_only:
                    self.save_error = "config.json is not readable; changes are kept in memory only"
                    return False
                version = self._version
                snapshot = copy.deepcopy(self._data)
                if same_value(snapshot, self._baseline):  # edited, then put back
                    self._saved_version = version
                    return False
                disk = self._disk_form(snapshot)
                backup = not self._backed_up
                path = self.path
            try:
                directory = os.path.dirname(os.path.abspath(path))
                if directory:
                    os.makedirs(directory, exist_ok=True)
                self._write(disk, path, backup)
                try:
                    raw = self._read(path, decrypt=False)
                except Exception:
                    raw = disk
            except Exception as exc:
                message = f"Saving config.json failed: {type(exc).__name__}: {exc}"
                log.error(message)
                with self._lock:
                    self.save_error = message
                self._notify_saved(False, message)
                return False
            with self._lock:
                self._raw = raw if isinstance(raw, dict) else {}
                self._baseline = snapshot
                self._saved_version = max(self._saved_version, version)
                self._backed_up = True
                self._exists = True
                self.save_error = None
                self.save_count += 1
                self.last_saved_at = time.time()
        self._notify_saved(True, None)
        return True

    def flush(self) -> bool:
        """Save now on the calling thread (blocking). Returns True when a file was written."""
        self._saver.cancel()
        return self._save_if_dirty()

    def wait_idle(self, timeout: Optional[float] = None) -> bool:
        return self._saver.wait_idle(timeout)

    def backup_now(self) -> Optional[str]:
        """Flush, then copy config.json into ``config_backups`` (Settings › Backup chip)."""
        self.flush()
        import config_store

        return config_store.backup_config_file(self.path)

    def close(self) -> None:
        """Flush pending changes and stop the save worker."""
        try:
            self.flush()
        finally:
            self._saver.close()

    # ---- observers -----------------------------------------------------------------------

    def observe(self, key: str, callback: Observer) -> Callable[[], None]:
        """``callback(key, value)`` after ``key`` changes (value is ``MISSING`` after ``unset``)."""
        with self._lock:
            self._observers.setdefault(key, []).append(callback)

        def unsubscribe() -> None:
            with self._lock:
                callbacks = self._observers.get(key)
                if callbacks and callback in callbacks:
                    callbacks.remove(callback)

        return unsubscribe

    def observe_keys(self, keys: Iterable[str], callback: Observer) -> Callable[[], None]:
        unsubs = [self.observe(key, callback) for key in keys]

        def unsubscribe() -> None:
            for unsub in unsubs:
                unsub()

        return unsubscribe

    def observe_all(self, callback: Observer) -> Callable[[], None]:
        with self._lock:
            self._all_observers.append(callback)

        def unsubscribe() -> None:
            with self._lock:
                if callback in self._all_observers:
                    self._all_observers.remove(callback)

        return unsubscribe

    def observe_saves(self, callback: Callable[[bool, Optional[str]], Any]) -> Callable[[], None]:
        """``callback(ok, error)`` after every save attempt (runs on the saving thread)."""
        with self._lock:
            self._save_listeners.append(callback)

        def unsubscribe() -> None:
            with self._lock:
                if callback in self._save_listeners:
                    self._save_listeners.remove(callback)

        return unsubscribe

    def _notify(self, key: str, value: Any) -> None:
        with self._lock:
            callbacks = list(self._observers.get(key, ())) + list(self._all_observers)
        for callback in callbacks:
            try:
                callback(key, value)
            except Exception:
                log.exception("config observer for %r failed", key)

    def _notify_saved(self, ok: bool, error: Optional[str]) -> None:
        with self._lock:
            listeners = list(self._save_listeners)
        for callback in listeners:
            try:
                callback(ok, error)
            except Exception:
                log.exception("config save listener failed")

    # ---- job state ("Changes apply to the next run") ---------------------------------------

    @property
    def job_running(self) -> bool:
        return self._job_running

    def set_job_running(self, running: bool) -> None:
        """JobService calls this when a job starts/stops (the job uses its own snapshot)."""
        running = bool(running)
        with self._lock:
            if running == self._job_running:
                return
            self._job_running = running
            self._changed_during_job = set()
            observers = list(self._job_observers)
        for callback in observers:
            try:
                callback(running)
            except Exception:
                log.exception("job-state observer failed")

    @property
    def changed_during_job(self) -> frozenset:
        """Keys edited while the current job runs (they apply to the next run)."""
        with self._lock:
            return frozenset(self._changed_during_job)

    def observe_job(self, callback: Callable[[bool], Any]) -> Callable[[], None]:
        with self._lock:
            self._job_observers.append(callback)

        def unsubscribe() -> None:
            with self._lock:
                if callback in self._job_observers:
                    self._job_observers.remove(callback)

        return unsubscribe
