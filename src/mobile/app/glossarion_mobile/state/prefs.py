"""Prefs: mobile-only UI state in ``<data>/mobile_state.json`` (UI_SPEC Appendix B).

Holds what the desktop has no place for and therefore never belongs in the
shared ``config.json``:

* ``last_routes``: the last whitelisted route per slot (``"main"``, tabs);
* ``reader_positions``: ``{bid: {href, fraction, page, mode, updated}}``;
* ``reader_bookmarks``: ``{bid: [{href, fraction, label, created}]}``;
* ``file_refs``: the FileRef registry, an LRU (2,000 entries) mapping opaque
  12-hex ids (``fid``, used in routes such as ``/tools/text/<fid>``) to paths,
  so a route never carries a file path;
* any other top-level key (``set``/``get``: dismissed tips, recent models,
  appearance mirrors); unknown keys from newer builds are preserved.

Writes are debounced (600 ms, ``DebouncedSaver``) and atomic: JSON goes to a
temp file in the same directory, is fsync'ed and then ``os.replace``'d over
the old file, so a crash leaves either the old or the new file, never a torn
one. (``app_paths._atomic_json_write`` is not reused here: it falls back to a
non-atomic direct write on error and does not fsync.) A corrupt file is moved
aside to ``mobile_state.corrupt-<time>.json`` and Prefs start empty.

Pure Python (Python 3.10 compatible); no Flet or backend imports.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Optional, Union

from glossarion_mobile.state.config_store import DEFAULT_DEBOUNCE, DebouncedSaver

__all__ = [
    "MAX_BOOKMARKS_PER_BOOK",
    "MAX_FILE_REFS",
    "PREFS_FILE_NAME",
    "PREFS_VERSION",
    "Prefs",
    "atomic_write_json",
    "file_ref_id",
    "prefs_path_for",
]

log = logging.getLogger("glossarion.prefs")

PREFS_FILE_NAME = "mobile_state.json"
PREFS_VERSION = 1
MAX_FILE_REFS = 2000
MAX_BOOKMARKS_PER_BOOK = 500
_MAX_ID_CHARS = 64

PathLike = Union[str, "os.PathLike[str]"]


def atomic_write_json(path: PathLike, data: Any) -> None:
    """Write ``data`` as JSON atomically (temp file + fsync + ``os.replace``)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=1)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, target)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    if hasattr(os, "O_DIRECTORY"):  # make the rename durable (POSIX)
        try:
            fd = os.open(str(target.parent), os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except OSError:
            pass


def file_ref_id(path: str, salt: int = 0) -> str:
    """Stable opaque 12-hex id for ``path`` (``salt`` resolves the rare collision)."""
    norm = os.path.normcase(os.path.abspath(os.fspath(path)))
    payload = norm if not salt else f"{norm}\x00{salt}"
    return hashlib.sha1(payload.encode("utf-8", "surrogatepass")).hexdigest()[:12]


def _empty() -> dict:
    return {
        "version": PREFS_VERSION,
        "last_routes": {},
        "reader_positions": {},
        "reader_bookmarks": {},
        "file_refs": OrderedDict(),
    }


def _clean_id(value: Any, what: str) -> str:
    text = str(value or "").strip()
    if not text or len(text) > _MAX_ID_CHARS:
        raise ValueError(f"invalid {what}: {value!r}")
    return text


def _fraction(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if number != number:  # NaN
        return 0.0
    return min(1.0, max(0.0, number))


class Prefs:
    """Thread-safe store for ``mobile_state.json``; call ``load()`` off the UI loop."""

    def __init__(
        self,
        path: PathLike,
        *,
        debounce: float = DEFAULT_DEBOUNCE,
        max_file_refs: int = MAX_FILE_REFS,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.path = Path(path)
        self.max_file_refs = max(1, int(max_file_refs))
        self._clock = clock
        self._lock = threading.RLock()
        self._data: dict = _empty()
        self._version = 0
        self._saved_version = 0
        self._loaded = False
        self.load_error: Optional[str] = None
        self.save_error: Optional[str] = None
        self.save_count = 0
        self._saver = DebouncedSaver(self._save_if_dirty, delay=debounce, name="gl-prefs-save")

    # ---- load / save ---------------------------------------------------------------

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def dirty(self) -> bool:
        with self._lock:
            return self._version != self._saved_version

    def load(self) -> dict:
        data = _empty()
        error: Optional[str] = None
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if not isinstance(loaded, dict):
                raise ValueError("mobile_state.json does not contain a JSON object")
            data.update(loaded)
            for name in ("last_routes", "reader_positions", "reader_bookmarks"):
                if not isinstance(data.get(name), dict):
                    data[name] = {}
            refs = data.get("file_refs")
            data["file_refs"] = OrderedDict(
                (str(k), v) for k, v in (refs.items() if isinstance(refs, dict) else ()) if isinstance(v, dict) and v.get("path")
            )
        except FileNotFoundError:
            pass
        except (OSError, ValueError, UnicodeDecodeError) as exc:
            error = f"mobile_state.json was unreadable ({type(exc).__name__}: {exc}); starting fresh"
            log.warning(error)
            self._move_aside()
            data = _empty()
        with self._lock:
            self._data = data
            self._version += 1
            self._saved_version = self._version
            self._loaded = True
            self.load_error = error
        self._saver.cancel()
        return self.snapshot()

    def _move_aside(self) -> None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        target = self.path.with_name(f"{self.path.stem}.corrupt-{stamp}{self.path.suffix}")
        try:
            os.replace(self.path, target)
        except OSError as exc:
            log.warning("could not move the corrupt %s aside: %s", self.path.name, exc)

    def _changed(self) -> None:
        # caller holds the lock
        self._version += 1

    def _save_if_dirty(self) -> bool:
        with self._lock:
            if self._version == self._saved_version:
                return False
            version = self._version
            payload = copy.deepcopy(self._data)
        payload["version"] = PREFS_VERSION
        try:
            atomic_write_json(self.path, payload)
        except Exception as exc:
            with self._lock:
                self.save_error = f"{type(exc).__name__}: {exc}"
            log.error("saving %s failed: %s", self.path.name, exc)
            return False
        with self._lock:
            self._saved_version = max(self._saved_version, version)
            self.save_error = None
            self.save_count += 1
        return True

    def _schedule(self) -> None:
        self._saver.schedule()

    def flush(self) -> bool:
        self._saver.cancel()
        return self._save_if_dirty()

    def wait_idle(self, timeout: Optional[float] = None) -> bool:
        return self._saver.wait_idle(timeout)

    def close(self) -> None:
        try:
            self.flush()
        finally:
            self._saver.close()

    def snapshot(self) -> dict:
        with self._lock:
            out = copy.deepcopy(self._data)
        out["file_refs"] = dict(out.get("file_refs") or {})
        return out

    # ---- generic keys --------------------------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            if key not in self._data:
                return default
            return copy.deepcopy(self._data[key])

    def set(self, key: str, value: Any) -> None:
        if key in ("version", "file_refs"):
            raise ValueError(f"{key!r} is managed by Prefs")
        with self._lock:
            if key in self._data and self._data[key] == value:
                return
            self._data[key] = copy.deepcopy(value)
            self._changed()
        self._schedule()

    # ---- last routes ----------------------------------------------------------------------

    def last_route(self, slot: str = "main") -> Optional[str]:
        with self._lock:
            entry = self._data["last_routes"].get(slot)
        return entry.get("route") if isinstance(entry, dict) else None

    def set_last_route(self, route: str, slot: str = "main") -> bool:
        """Remember ``route`` (only whitelisted routes are kept; deep-link handlers are not)."""
        from glossarion_mobile.ui.router import HANDLED, parse_route

        match = parse_route(route)
        if match is None or match.presentation == HANDLED:
            return False
        with self._lock:
            current = self._data["last_routes"].get(slot)
            if isinstance(current, dict) and current.get("route") == match.route:
                return False
            self._data["last_routes"][slot] = {"route": match.route, "updated": self._clock()}
            self._changed()
        self._schedule()
        return True

    # ---- reader positions and bookmarks --------------------------------------------------

    def reader_position(self, bid: str) -> Optional[dict]:
        with self._lock:
            entry = self._data["reader_positions"].get(str(bid))
            return copy.deepcopy(entry) if isinstance(entry, dict) else None

    def set_reader_position(
        self, bid: str, href: str, fraction: float = 0.0, *, page: Optional[int] = None, mode: Optional[str] = None
    ) -> dict:
        bid = _clean_id(bid, "book id")
        entry = {"href": str(href or ""), "fraction": _fraction(fraction), "page": page, "mode": mode,
                 "updated": self._clock()}
        with self._lock:
            self._data["reader_positions"][bid] = entry
            self._changed()
        self._schedule()
        return dict(entry)

    def bookmarks(self, bid: str) -> list[dict]:
        with self._lock:
            items = self._data["reader_bookmarks"].get(str(bid))
            return copy.deepcopy(items) if isinstance(items, list) else []

    def add_bookmark(self, bid: str, href: str, fraction: float = 0.0, label: str = "") -> dict:
        bid = _clean_id(bid, "book id")
        mark = {"href": str(href or ""), "fraction": _fraction(fraction), "label": str(label or ""),
                "created": self._clock()}
        with self._lock:
            items = self._data["reader_bookmarks"].setdefault(bid, [])
            if not isinstance(items, list):
                items = self._data["reader_bookmarks"][bid] = []
            items.append(mark)
            del items[:-MAX_BOOKMARKS_PER_BOOK]
            self._changed()
        self._schedule()
        return dict(mark)

    def remove_bookmark(self, bid: str, index: int) -> bool:
        with self._lock:
            items = self._data["reader_bookmarks"].get(str(bid))
            if not isinstance(items, list) or not 0 <= index < len(items):
                return False
            del items[index]
            if not items:
                del self._data["reader_bookmarks"][str(bid)]
            self._changed()
        self._schedule()
        return True

    # ---- file refs (opaque ids for routes) -------------------------------------------------

    def file_ref(self, path: PathLike, *, kind: Optional[str] = None) -> str:
        """Register ``path`` and return its ``fid`` (the most recently used entry moves to the end)."""
        text = os.fspath(path)
        if not text:
            raise ValueError("empty path")
        norm = os.path.normcase(os.path.abspath(text))
        with self._lock:
            refs: OrderedDict = self._data["file_refs"]
            salt = 0
            fid = file_ref_id(text)
            while fid in refs and os.path.normcase(os.path.abspath(refs[fid].get("path", ""))) != norm:
                salt += 1
                fid = file_ref_id(text, salt)
            entry = refs.pop(fid, None) or {}
            entry.update({"path": os.path.abspath(text), "touched": self._clock()})
            if kind:
                entry["kind"] = kind
            refs[fid] = entry
            while len(refs) > self.max_file_refs:
                refs.popitem(last=False)
            self._changed()
        self._schedule()
        return fid

    def resolve_file_ref(self, fid: str, *, touch: bool = True) -> Optional[str]:
        with self._lock:
            refs: OrderedDict = self._data["file_refs"]
            entry = refs.get(str(fid))
            if not isinstance(entry, dict):
                return None
            if touch:
                refs.move_to_end(str(fid))
                entry["touched"] = self._clock()
                self._changed()
            path = entry.get("path")
        if touch:
            self._schedule()
        return path

    def forget_file_ref(self, fid: str) -> bool:
        with self._lock:
            if self._data["file_refs"].pop(str(fid), None) is None:
                return False
            self._changed()
        self._schedule()
        return True

    def file_ref_count(self) -> int:
        with self._lock:
            return len(self._data["file_refs"])


def prefs_path_for(data_dir: PathLike) -> Path:
    """``<data>/mobile_state.json``."""
    return Path(data_dir) / PREFS_FILE_NAME

