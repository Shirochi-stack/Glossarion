"""Route whitelist for ``page.on_route_change`` (pure Python, no Flet import).

Routes reach the app three ways: in-app ``page.push_route()``, deep links
``glossarion://app/<route>`` (Flutter may hand over either the bare path or the
whole URI, e.g. ``data.toString()``), and anything else the OS throws at the
activity (an Open-with ``content://``/``file://`` URI if the share trampoline is
bypassed, ``https://`` App Links, foreign custom schemes). Only whitelisted
paths are acted on; everything else is recorded and ignored, never navigated
to. Routes carry opaque ids and short flags only, never file paths or text.

U0 whitelist: ``/``, ``/__selftest__``, ``/oauth/return``. Later milestones
extend ``ALLOWED_PATHS`` / ``ALLOWED_PREFIXES``.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import parse_qsl, unquote, urlsplit

__all__ = [
    "ALLOWED_PATHS",
    "DEEP_LINK_SCHEME",
    "DEEP_LINK_HOST",
    "RouteMatch",
    "RouteRecord",
    "Router",
    "parse_route",
]

DEEP_LINK_SCHEME = "glossarion"
DEEP_LINK_HOST = "app"

ALLOWED_PATHS = frozenset({"/", "/__selftest__", "/oauth/return"})

# Schemes that must never be routed (file imports arrive through the native
# share/open-with channel, not through page.route).
_BLOCKED_SCHEMES = frozenset({"content", "file", "intent", "data", "javascript", "android-app", "blob"})

_MAX_ROUTE_CHARS = 2048
_MAX_QUERY_ITEMS = 16
_MAX_KEY_CHARS = 64
_MAX_VALUE_CHARS = 512


@dataclass(frozen=True)
class RouteMatch:
    path: str
    query: dict[str, str] = field(default_factory=dict)
    raw: str = ""
    deep_link: bool = False  # arrived as glossarion://app/...

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        return self.query.get(key, default)


@dataclass(frozen=True)
class RouteRecord:
    raw: str
    accepted: bool
    path: Optional[str]
    reason: str
    at: float


def _reject(reason: str) -> tuple[None, str]:
    return None, reason


def _parse(raw: Optional[str]) -> tuple[Optional[RouteMatch], str]:
    if raw is None:
        return RouteMatch("/", {}, "", False), "empty"
    text = str(raw).strip()
    if not text:
        return RouteMatch("/", {}, str(raw), False), "empty"
    if len(text) > _MAX_ROUTE_CHARS:
        return _reject("too long")
    if any(ord(ch) < 0x20 for ch in text):
        return _reject("control characters")

    deep_link = False
    head = text.split("?", 1)[0].split("#", 1)[0]
    if ":" in head and not head.startswith("/"):
        scheme = head.split(":", 1)[0].lower()
        if scheme in _BLOCKED_SCHEMES:
            return _reject(f"blocked scheme {scheme}:")
        try:
            parts = urlsplit(text)
        except ValueError:
            return _reject("unparseable URI")
        if parts.scheme.lower() != DEEP_LINK_SCHEME:
            return _reject(f"foreign scheme {parts.scheme or '?'}:")
        if (parts.hostname or "").lower() != DEEP_LINK_HOST or parts.username or parts.password or parts.port:
            return _reject(f"foreign host {parts.netloc!r}")
        path = parts.path or "/"
        query = parts.query
        deep_link = True
    else:
        if text.startswith("//"):
            return _reject("network-path reference")
        if not text.startswith("/"):
            return _reject("relative route")
        parts = urlsplit(text)
        if parts.scheme or parts.netloc:
            return _reject("unexpected authority")
        path = parts.path or "/"
        query = parts.query

    decoded = unquote(path)
    if decoded != path and any(ch in decoded for ch in ("/", "\\", ".")):
        return _reject("encoded path separators")
    if "\\" in path or "/../" in f"{path}/" or "/./" in f"{path}/":
        return _reject("path traversal")
    if len(path) > 1:
        path = path.rstrip("/") or "/"
    if path not in ALLOWED_PATHS:
        return _reject(f"not whitelisted: {path}")

    params: dict[str, str] = {}
    for key, value in parse_qsl(query, keep_blank_values=True)[:_MAX_QUERY_ITEMS]:
        if len(key) > _MAX_KEY_CHARS or len(value) > _MAX_VALUE_CHARS:
            continue
        params.setdefault(key, value)
    return RouteMatch(path, params, text, deep_link), "ok"


def parse_route(raw: Optional[str]) -> Optional[RouteMatch]:
    """Whitelisted ``RouteMatch`` for ``raw``, or ``None`` when it must be ignored."""
    match, _reason = _parse(raw)
    return match


class Router:
    """Records every route seen and returns a match only for whitelisted ones.

    Thread-safe; ``history`` keeps the last 100 records for diagnostics (the
    U0 spike screen shows it to prove Open-with never changes ``page.route``).
    """

    def __init__(self, history: int = 100) -> None:
        self._lock = threading.Lock()
        self._history: deque[RouteRecord] = deque(maxlen=history)

    def handle(self, raw: Optional[str]) -> Optional[RouteMatch]:
        match, reason = _parse(raw)
        record = RouteRecord(
            raw="" if raw is None else str(raw)[:300],
            accepted=match is not None,
            path=match.path if match is not None else None,
            reason=reason,
            at=time.time(),
        )
        with self._lock:
            self._history.append(record)
        return match

    @property
    def history(self) -> list[RouteRecord]:
        with self._lock:
            return list(self._history)

    def rejected_since(self, since: float) -> list[RouteRecord]:
        return [r for r in self.history if not r.accepted and r.at >= since]

    def seen_since(self, since: float) -> list[RouteRecord]:
        return [r for r in self.history if r.at >= since]
