"""Chat list for the drawer (Pinned / Recents, UI_SPEC §1.3). Pure Python.

``InMemoryChatIndex`` is the U1 placeholder store: it holds the current, empty
"New chat" and lets the drawer pin/unpin rows. In U3 the same interface is
backed by the shared ``direct_text_store.ChatStore`` (``direct_text_chats.json``
v2 plus the mobile sidecar); the drawer does not change.

``group_recents`` implements the Recents headers: Today / Yesterday /
Previous 7 days / "<Month YYYY>" (local time).
"""

from __future__ import annotations

import datetime as _dt
import logging
import threading
import time
from dataclasses import dataclass, replace
from typing import Callable, Iterable, Optional

__all__ = ["ChatSummary", "InMemoryChatIndex", "group_recents", "recents_group_label"]


@dataclass(frozen=True)
class ChatSummary:
    cid: str  # v2 int id as text, or "s<uuid>" for scratch chats (route-safe)
    title: str
    updated_at: float  # epoch seconds
    pinned: bool = False
    pinned_at: float = 0.0
    attachments: int = 0  # attachment workspaces (drawer shows a paperclip count)
    running: bool = False  # owns the running job (drawer shows a ProgressRing)
    scratch: bool = False


def _local_date(ts: float) -> _dt.date:
    return _dt.datetime.fromtimestamp(ts).date()


def recents_group_label(ts: float, now: float) -> str:
    day = _local_date(ts)
    today = _local_date(now)
    delta = (today - day).days
    if delta <= 0:
        return "Today"
    if delta == 1:
        return "Yesterday"
    if delta <= 7:
        return "Previous 7 days"
    return _dt.date(day.year, day.month, 1).strftime("%B %Y")


def group_recents(chats: Iterable[ChatSummary], now: float) -> list[tuple[str, list[ChatSummary]]]:
    """Unpinned chats, newest first, grouped under their Recents header."""
    ordered = sorted((c for c in chats if not c.pinned), key=lambda c: c.updated_at, reverse=True)
    groups: list[tuple[str, list[ChatSummary]]] = []
    for chat in ordered:
        label = recents_group_label(chat.updated_at, now)
        if groups and groups[-1][0] == label:
            groups[-1][1].append(chat)
        else:
            groups.append((label, [chat]))
    return groups


class InMemoryChatIndex:
    """Placeholder chat store with change notification (callbacks run on the caller's thread)."""

    def __init__(
        self, chats: Optional[Iterable[ChatSummary]] = None, *, clock: Optional[Callable[[], float]] = None
    ) -> None:
        self._clock = clock or time.time
        self._lock = threading.RLock()
        if chats is None:
            chats = [ChatSummary("1", "New chat", self._clock())]
        self._chats: dict[str, ChatSummary] = {c.cid: c for c in chats}
        self._listeners: list[Callable[[], None]] = []

    def all(self) -> list[ChatSummary]:
        with self._lock:
            return list(self._chats.values())

    def get(self, cid: str) -> Optional[ChatSummary]:
        with self._lock:
            return self._chats.get(str(cid))

    def pinned(self) -> list[ChatSummary]:
        with self._lock:
            rows = [c for c in self._chats.values() if c.pinned]
        return sorted(rows, key=lambda c: c.pinned_at, reverse=True)

    def recents(self, now: Optional[float] = None) -> list[tuple[str, list[ChatSummary]]]:
        return group_recents(self.all(), self._clock() if now is None else now)

    def search(self, query: str) -> list[ChatSummary]:
        needle = query.strip().casefold()
        if not needle:
            return []
        rows = [c for c in self.all() if needle in c.title.casefold()]
        return sorted(rows, key=lambda c: c.updated_at, reverse=True)

    def set_pinned(self, cid: str, pinned: bool) -> bool:
        with self._lock:
            chat = self._chats.get(str(cid))
            if chat is None or chat.pinned == pinned:
                return False
            self._chats[chat.cid] = replace(chat, pinned=pinned, pinned_at=self._clock() if pinned else 0.0)
        self._changed()
        return True

    def upsert(self, chat: ChatSummary) -> None:
        with self._lock:
            self._chats[chat.cid] = chat
        self._changed()

    def remove(self, cid: str) -> bool:
        with self._lock:
            removed = self._chats.pop(str(cid), None) is not None
        if removed:
            self._changed()
        return removed

    def subscribe(self, callback: Callable[[], None]) -> Callable[[], None]:
        self._listeners.append(callback)

        def unsubscribe() -> None:
            try:
                self._listeners.remove(callback)
            except ValueError:
                pass

        return unsubscribe

    def _changed(self) -> None:
        for callback in list(self._listeners):
            try:
                callback()
            except Exception:
                logging.getLogger("glossarion.state").exception("chat index listener failed")
