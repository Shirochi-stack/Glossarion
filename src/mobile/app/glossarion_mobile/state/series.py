"""Series: the optional, mobile-only chat grouping (UI_SPEC §2.15, Appendix B). Pure Python.

A series groups chats (and links Library books) and carries chat defaults. It is not a desktop
feature and adds no backend behaviour: the defaults are the same per-chat override keys the
Chat settings sheet writes (``chat_store_adapter.OVERRIDE_KEYS``: model, prompt profile,
target language, default output mode, glossary policy + manual glossary path and the Direct
Text overrides), applied as the layer between the global config and the chat's own overrides
(Global config -> Series defaults -> per-chat overrides, §2.14). The effective values reach a
run exactly like a chat override (JobSpec overrides); nothing here writes config.json.

Data (Appendix B):

* ``mobile_series.json`` beside the chat sidecar::

      {version: 1, series: [{id, name, color, cover_bid, book_ids, defaults: {...}, created_at}]}

* a chat's series is ``series_id`` in its ``direct_text_chats.mobile.json`` entry (through the
  chat adapter's generic ``meta`` / ``set_meta``); scratch chats never join a series (they are
  never saved). An id that no longer resolves reads as "no series".

``SeriesStore`` is thread-safe (one lock) and writes atomically on every change (the file is
small). Listeners run on the thread that changed the data unless ``post`` is set (the feature
sets it to the UI dispatcher). ``SeriesDefaultsChats`` lets the Chat settings sheet edit a
series' defaults with its own This-chat logic (the series is its "chat").
"""

from __future__ import annotations

import copy
import json
import logging
import os
import re
import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

from glossarion_mobile.state.chat_store_adapter import OVERRIDE_KEYS, is_scratch_cid
from glossarion_mobile.state.prefs import atomic_write_json

__all__ = [
    "DEFAULT_KEYS",
    "MAX_NAME_CHARS",
    "SERIES_COLORS",
    "SERIES_FILE",
    "SERIES_META_KEY",
    "Series",
    "SeriesDefaultsChats",
    "SeriesRow",
    "SeriesStore",
    "chat_series_id",
    "clean_name",
    "color_hex",
    "layered_overrides",
    "member_chats",
    "move_chat",
    "series_choices",
    "series_rows",
    "series_search",
]

log = logging.getLogger("glossarion.series")

SERIES_FILE = "mobile_series.json"
SERIES_VERSION = 1
#: The chat sidecar key that names a chat's series (Appendix B ``series_id``).
SERIES_META_KEY = "series_id"
MAX_NAME_CHARS = 80
#: Series defaults: the per-chat override keys (the Chat settings sheet's This-chat fields).
DEFAULT_KEYS = tuple(OVERRIDE_KEYS)
#: The 8 tonal swatches (UI_SPEC §2.15): Halgakos Rose, Horn Plum, Desktop Blue, Library Violet
#: (the design-system accents, §6.1) and four companions of the same weight.
SERIES_COLORS = (
    ("rose", "#E18F98"),
    ("plum", "#8E6A8A"),
    ("blue", "#5A9FD4"),
    ("violet", "#6C63FF"),
    ("teal", "#3BA99C"),
    ("amber", "#D9A441"),
    ("green", "#6AA84F"),
    ("slate", "#7A8794"),
)
_COLOR_HEX = dict(SERIES_COLORS)
_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")  # router kind "id" (/series/<sid>)


def color_hex(color: Any) -> str:
    """The swatch colour of a series (unknown names fall back to the first swatch)."""
    return _COLOR_HEX.get(str(color or ""), SERIES_COLORS[0][1])


def clean_name(name: Any) -> str:
    """Whitespace collapsed, at most ``MAX_NAME_CHARS`` characters ('' when empty)."""
    return " ".join(str(name or "").split())[:MAX_NAME_CHARS]


def _clean_defaults(values: Any) -> dict:
    if not isinstance(values, Mapping):
        return {}
    return {str(k): copy.deepcopy(v) for k, v in values.items() if k in DEFAULT_KEYS and v is not None}


@dataclass(frozen=True)
class Series:
    id: str
    name: str
    color: str = SERIES_COLORS[0][0]
    cover_bid: str = ""
    book_ids: tuple = ()
    defaults: dict = field(default_factory=dict)
    created_at: float = 0.0

    @property
    def color_hex(self) -> str:
        return color_hex(self.color)

    def as_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "color": self.color, "cover_bid": self.cover_bid,
                "book_ids": list(self.book_ids), "defaults": copy.deepcopy(self.defaults),
                "created_at": self.created_at}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Optional["Series"]:
        sid = str(data.get("id") or "")
        if not _ID_RE.match(sid):
            return None
        books = []
        for bid in data.get("book_ids") or ():
            if isinstance(bid, str) and bid and bid not in books:
                books.append(bid)
        try:
            created = float(data.get("created_at") or 0.0)
        except (TypeError, ValueError):
            created = 0.0
        color = str(data.get("color") or "")
        return cls(
            id=sid,
            name=clean_name(data.get("name")) or "Series",
            color=color if color in _COLOR_HEX else SERIES_COLORS[0][0],
            cover_bid=str(data.get("cover_bid") or ""),
            book_ids=tuple(books),
            defaults=_clean_defaults(data.get("defaults")),
            created_at=created,
        )


class SeriesStore:
    """``mobile_series.json``: the series list (create / edit / delete, defaults, linked books)."""

    def __init__(self, path: str, *, clock: Callable[[], float] = time.time,
                 new_id: Optional[Callable[[], str]] = None) -> None:
        self.path = str(path)
        self._clock = clock
        self._new_id = new_id or (lambda: secrets.token_hex(6))
        self._lock = threading.RLock()
        self._series: dict = {}  # id -> Series, insertion = creation order
        self._listeners: list = []
        self.post: Optional[Callable[[Callable[[], None]], Any]] = None
        self.saves = 0
        self.loaded = False

    # ---- persistence ------------------------------------------------------------------------

    def load(self) -> bool:
        """Read the file (missing or unreadable -> no series). Returns True when it was read."""
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except FileNotFoundError:
            payload = None
        except Exception as exc:
            log.warning("ignoring unreadable %s: %s", os.path.basename(self.path), exc)
            payload = None
        series: dict = {}
        if isinstance(payload, Mapping):
            for item in payload.get("series") or ():
                parsed = Series.from_dict(item) if isinstance(item, Mapping) else None
                if parsed is not None and parsed.id not in series:
                    series[parsed.id] = parsed
        with self._lock:
            self._series = series
            self.loaded = True
        return payload is not None

    def save(self) -> None:
        with self._lock:
            data = {"version": SERIES_VERSION, "series": [s.as_dict() for s in self._series.values()]}
            try:
                atomic_write_json(self.path, data)
                self.saves += 1
            except Exception:
                log.exception("saving %s failed", SERIES_FILE)

    # ---- listeners --------------------------------------------------------------------------

    def subscribe(self, callback: Callable[[], None]) -> Callable[[], None]:
        self._listeners.append(callback)

        def unsubscribe() -> None:
            try:
                self._listeners.remove(callback)
            except ValueError:
                pass

        return unsubscribe

    def _changed(self) -> None:
        self.save()
        post = self.post
        for callback in list(self._listeners):
            try:
                if post is not None:
                    post(callback)
                else:
                    callback()
            except Exception:
                log.exception("series listener failed")

    # ---- queries ----------------------------------------------------------------------------

    def all(self) -> list:
        """Every series, by name (case-insensitive), then creation."""
        with self._lock:
            items = list(self._series.values())
        return sorted(items, key=lambda s: (s.name.casefold(), s.created_at, s.id))

    def get(self, sid: Any) -> Optional[Series]:
        with self._lock:
            return self._series.get(str(sid or ""))

    def has(self, sid: Any) -> bool:
        return self.get(sid) is not None

    def defaults(self, sid: Any) -> dict:
        item = self.get(sid)
        return copy.deepcopy(item.defaults) if item is not None else {}

    def series_for_book(self, bid: Any) -> list:
        key = str(bid or "")
        return [s for s in self.all() if key and key in s.book_ids]

    # ---- edits ------------------------------------------------------------------------------

    def _put(self, item: Series) -> None:
        with self._lock:
            self._series[item.id] = item

    def create(self, name: Any, *, color: Optional[str] = None, cover_bid: str = "",
               book_ids: Iterable[str] = (), defaults: Optional[Mapping[str, Any]] = None) -> Series:
        with self._lock:
            sid = self._new_id()
            while not _ID_RE.match(sid) or sid in self._series:
                sid = secrets.token_hex(6)
            if color not in _COLOR_HEX:
                used = {s.color for s in self._series.values()}
                color = next((c for c, _hex in SERIES_COLORS if c not in used),
                             SERIES_COLORS[len(self._series) % len(SERIES_COLORS)][0])
            books = tuple(dict.fromkeys(str(b) for b in book_ids if b))
            item = Series(id=sid, name=clean_name(name) or "New series", color=str(color),
                          cover_bid=str(cover_bid or (books[0] if books else "")), book_ids=books,
                          defaults=_clean_defaults(defaults), created_at=float(self._clock()))
            self._series[sid] = item
        self._changed()
        return item

    def update(self, sid: Any, *, name: Any = None, color: Optional[str] = None,
               cover_bid: Optional[str] = None) -> bool:
        item = self.get(sid)
        if item is None:
            return False
        changes: dict = {}
        if name is not None and clean_name(name):
            changes["name"] = clean_name(name)
        if color is not None and color in _COLOR_HEX:
            changes["color"] = color
        if cover_bid is not None:
            changes["cover_bid"] = str(cover_bid or "")
        updated = _replace(item, **changes)
        if updated == item:
            return False
        self._put(updated)
        self._changed()
        return True

    def delete(self, sid: Any) -> bool:
        with self._lock:
            removed = self._series.pop(str(sid or ""), None)
        if removed is None:
            return False
        self._changed()
        return True

    def set_default(self, sid: Any, key: str, value: Any) -> bool:
        """One series default (None removes it, the chats then inherit All chats)."""
        if key not in DEFAULT_KEYS:
            raise KeyError(key)
        item = self.get(sid)
        if item is None:
            return False
        values = dict(item.defaults)
        if value is None:
            values.pop(key, None)
        else:
            values[key] = copy.deepcopy(value)
        if values == item.defaults:
            return False
        self._put(_replace(item, defaults=values))
        self._changed()
        return True

    def reset_defaults(self, sid: Any) -> bool:
        item = self.get(sid)
        if item is None or not item.defaults:
            return False
        self._put(_replace(item, defaults={}))
        self._changed()
        return True

    def link_books(self, sid: Any, bids: Iterable[str]) -> int:
        """Link Library books (opaque book ids); returns how many were new."""
        item = self.get(sid)
        if item is None:
            return 0
        books = list(item.book_ids)
        added = 0
        for bid in bids:
            bid = str(bid or "")
            if bid and bid not in books:
                books.append(bid)
                added += 1
        if not added:
            return 0
        self._put(_replace(item, book_ids=tuple(books), cover_bid=item.cover_bid or books[0]))
        self._changed()
        return added

    def unlink_book(self, sid: Any, bid: str) -> bool:
        item = self.get(sid)
        if item is None or bid not in item.book_ids:
            return False
        books = tuple(b for b in item.book_ids if b != bid)
        cover = item.cover_bid if item.cover_bid != bid else (books[0] if books else "")
        self._put(_replace(item, book_ids=books, cover_bid=cover))
        self._changed()
        return True


def _replace(item: Series, **changes: Any) -> Series:
    data = item.as_dict()
    data.update(changes)
    if "book_ids" in changes:
        data["book_ids"] = list(changes["book_ids"])
    return Series(id=item.id, name=data["name"], color=data["color"], cover_bid=data["cover_bid"],
                  book_ids=tuple(data["book_ids"]), defaults=_clean_defaults(data["defaults"]),
                  created_at=item.created_at)


# ---------------------------------------------------------------------------
# Chats (duck-typed ChatStoreAdapter: meta / set_meta / all / get / overrides)
# ---------------------------------------------------------------------------


def chat_series_id(chats: Any, cid: Any, store: Optional[SeriesStore] = None) -> Optional[str]:
    """The chat's series id, or None (scratch chats, no series, or a series that is gone)."""
    if chats is None or cid is None or is_scratch_cid(cid):
        return None
    try:
        sid = (chats.meta(cid) or {}).get(SERIES_META_KEY)
    except Exception:
        return None
    if not isinstance(sid, str) or not sid:
        return None
    if store is not None and not store.has(sid):
        return None
    return sid


def move_chat(chats: Any, cid: Any, sid: Optional[str], store: Optional[SeriesStore] = None) -> bool:
    """Put a chat in a series (``sid``) or take it out (None). Scratch chats cannot join."""
    if chats is None or cid is None or is_scratch_cid(cid):
        return False
    if sid is not None and store is not None and not store.has(sid):
        return False
    if chat_series_id(chats, cid) == (sid or None):
        return False
    chats.set_meta(cid, SERIES_META_KEY, sid or None)
    return True


def member_chats(chats: Any, sid: str, store: Optional[SeriesStore] = None) -> list:
    """The chats of a series (ChatSummary rows), newest first."""
    if chats is None or not sid or (store is not None and not store.has(sid)):
        return []
    rows = []
    for chat in chats.all():
        if getattr(chat, "scratch", False) or is_scratch_cid(chat.cid):
            continue
        if chat_series_id(chats, chat.cid) == sid:
            rows.append(chat)
    return sorted(rows, key=lambda c: c.updated_at, reverse=True)


def layered_overrides(series_defaults: Optional[Mapping[str, Any]], own: Optional[Mapping[str, Any]]) -> dict:
    """Series defaults beneath the chat's own overrides (the chat wins; None = not set)."""
    merged = {k: v for k, v in (series_defaults or {}).items() if k in DEFAULT_KEYS and v is not None}
    merged.update({k: v for k, v in (own or {}).items() if v is not None})
    return merged


@dataclass(frozen=True)
class SeriesRow:
    """One drawer Series entry (UI_SPEC §1.3 item 5)."""

    sid: str
    name: str
    color: str  # hex
    chats: tuple = ()  # ChatSummary rows, newest first

    @property
    def count(self) -> int:
        return len(self.chats)


def series_rows(store: Optional[SeriesStore], chats: Any) -> list:
    """Every series with its chats, for the drawer (series by name)."""
    if store is None:
        return []
    members: dict = {s.id: [] for s in store.all()}
    if chats is not None:
        for chat in chats.all():
            if getattr(chat, "scratch", False) or is_scratch_cid(chat.cid):
                continue
            sid = chat_series_id(chats, chat.cid)
            if sid in members:
                members[sid].append(chat)
    return [SeriesRow(s.id, s.name, s.color_hex,
                      tuple(sorted(members[s.id], key=lambda c: c.updated_at, reverse=True)))
            for s in store.all()]


def series_search(store: Optional[SeriesStore], chats: Any, query: str) -> list:
    """Series-scoped drawer search: every series whose name matches (all its chats), plus the
    series holding chats that match by title or message text (only those chats)."""
    needle = str(query or "").strip().casefold()
    if store is None or not needle:
        return []
    matched_chats: set = set()
    if chats is not None:
        try:
            matched_chats = {c.cid for c in chats.search(query)}
        except Exception:
            matched_chats = set()
        for chat in chats.all():
            if needle in str(chat.title or "").casefold():
                matched_chats.add(chat.cid)
    out = []
    for row in series_rows(store, chats):
        if needle in row.name.casefold():
            out.append(row)
            continue
        hits = tuple(c for c in row.chats if c.cid in matched_chats)
        if hits:
            out.append(SeriesRow(row.sid, row.name, row.color, hits))
    return out


class SeriesDefaultsChats:
    """The chat-adapter surface the Chat settings sheet uses, over one series' defaults.

    ``ChatSettingsSheet(cid=<sid>, chats=SeriesDefaultsChats(store, sid), subject="series")`` edits
    the series defaults with the sheet's This-chat logic: "custom" = set for the series, ↺ resets
    to All chats. Chat-only data (skip plan, text size) is not part of a series."""

    def __init__(self, store: SeriesStore, sid: str) -> None:
        self.store = store
        self.sid = str(sid)

    def overrides(self, cid: Any = None) -> dict:
        return self.store.defaults(self.sid)

    own_overrides = overrides

    def set_override(self, cid: Any, key: str, value: Any) -> None:
        self.store.set_default(self.sid, key, value)

    def reset_overrides(self, cid: Any = None) -> None:
        self.store.reset_defaults(self.sid)

    def meta(self, cid: Any = None) -> dict:
        return {}

    def set_meta(self, cid: Any, key: str, value: Any) -> None:
        return None


def series_choices(store: Optional[SeriesStore]) -> Sequence[tuple]:
    """(sid, name) of every series (filter chips, pickers)."""
    return [(s.id, s.name) for s in store.all()] if store is not None else []
