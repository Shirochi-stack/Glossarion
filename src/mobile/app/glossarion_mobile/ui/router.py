"""Route whitelist for ``page.on_route_change`` (pure Python, no Flet import).

Routes reach the app three ways: in-app navigation, deep links
``glossarion://app/<route>`` (Flutter may hand over either the bare path or the
whole URI, e.g. ``data.toString()``), and anything else the OS throws at the
activity (an Open-with ``content://``/``file://`` URI if the share trampoline is
bypassed, ``https://`` App Links, foreign custom schemes). Only whitelisted
routes are acted on; everything else is recorded and ignored, never navigated
to.

The whitelist is UI_SPEC §1.4. Routes carry opaque ids and enums only, never
file paths, file names or user text (§1.4 "Never in a route"): every path
placeholder and every query value is checked against a strict pattern
(12-hex ids, integers, slugs, enums). Unknown query keys and invalid query
values are dropped; a path that does not match a route exactly is rejected.

``ROUTES`` doubles as the screen table: each ``RouteSpec`` names the surface,
its phone presentation (§1.4 column 3), its parent (for the phone back stack)
and the milestone that ships it (plan §8).
"""

from __future__ import annotations

import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional
from urllib.parse import parse_qsl, quote, unquote, urlencode, urlsplit

__all__ = [
    "ALLOWED_PATHS",
    "DEEP_LINK_SCHEME",
    "DEEP_LINK_HOST",
    "HANDLED",
    "ROUTES",
    "ROUTES_BY_NAME",
    "RouteError",
    "RouteMatch",
    "RouteRecord",
    "RouteSpec",
    "Router",
    "build_route",
    "launch_links",
    "parse_route",
]

DEEP_LINK_SCHEME = "glossarion"
DEEP_LINK_HOST = "app"

# Schemes that must never be routed (file imports arrive through the native
# share/open-with channel, not through page.route).
_BLOCKED_SCHEMES = frozenset({"content", "file", "intent", "data", "javascript", "android-app", "blob"})

_MAX_ROUTE_CHARS = 2048
_MAX_QUERY_ITEMS = 16
_MAX_KEY_CHARS = 64
_MAX_VALUE_CHARS = 512

# Presentations (UI_SPEC §1.4, phone column)
ROOT = "root"
VIEW = "view"
SHEET = "sheet"
FULLSCREEN = "fullscreen"
HANDLED = "handled"  # handled, no View (OAuth return, self-test)

KEY_POOLS = (
    "translation",
    "fallback",
    "glossary",
    "glossary_refinement",
    "qa_vision",
    "metadata",
    "ai_truncation",
    "rolling_summary",
    "truncation_retry",
    "inpainter",
    "tts",
)

# Value patterns. Everything is anchored; nothing allows '/', '\\', '.', '%' or spaces
# except where noted (config keys may contain '.').
_PATTERNS: dict[str, str] = {
    "int": r"\d{1,9}",
    "hex12": r"[0-9a-fA-F]{12}",
    # v2 chat id (int) or scratch chat "s<uuid>"
    "cid": r"\d{1,12}|s[0-9a-fA-F]{8}(?:-?[0-9a-fA-F]{4}){3}-?[0-9a-fA-F]{12}|s[0-9a-fA-F]{8,32}",
    "id": r"[A-Za-z0-9_-]{1,64}",
    "slug": r"[a-z0-9_-]{1,64}",
    # settings_schema section ids are dotted ("other.anti_duplicate.core"); no empty segments
    "section": r"(?=[a-z0-9_.]{1,64}\Z)[a-z0-9_]+(?:\.[a-z0-9_]+)*",
    "key": r"[A-Za-z0-9_.-]{1,96}",
    "token": r"[A-Za-z0-9_-]{1,64}",
    "group": r"[a-z_]{1,32}",
    "root": r"output|library|inbox|chats|payloads|logs|[0-9a-fA-F]{12}",
    "pool": "|".join(KEY_POOLS),
    "shelf": r"progress|completed",
    "book_tab": r"overview|chapters|glossary|output",
    "reader_mode": r"translated|original|bilingual",
    "glossary_tab": r"editor|general|balanced|minimal|refinement",
}
_COMPILED = {name: re.compile(rf"(?:{pattern})\Z") for name, pattern in _PATTERNS.items()}


def _valid(kind: str, value: str) -> bool:
    regex = _COMPILED.get(kind)
    return bool(regex is not None and regex.match(value))


@dataclass(frozen=True)
class RouteSpec:
    name: str
    pattern: str  # "/chat/{cid:cid}/settings"
    title: str
    presentation: str = VIEW
    milestone: str = "U1"
    parent: Optional[str] = None
    query: Mapping[str, str] = field(default_factory=dict)  # key -> pattern kind
    fragment: Optional[str] = None  # pattern kind of "#fragment", when allowed
    alias_of: Optional[str] = None  # canonical route name for aliases

    @property
    def segments(self) -> tuple[tuple[str, Optional[str]], ...]:
        """((literal, None) | (param_name, kind), ...) for each path segment."""
        out = []
        for part in self.pattern.strip("/").split("/") if self.pattern != "/" else ():
            if part.startswith("{") and part.endswith("}"):
                name, _, kind = part[1:-1].partition(":")
                out.append((name, kind or "id"))
            else:
                out.append((part, None))
        return tuple(out)

    @property
    def is_static(self) -> bool:
        return all(kind is None for _name, kind in self.segments)


def _settings_page(slug: str, title: str, milestone: str) -> RouteSpec:
    return RouteSpec(f"settings.{slug}", f"/settings/{slug}", title, VIEW, milestone, parent="settings")


_TOOLS = (
    # slug, title, milestone, query
    ("progress", "Progress manager", "U5"),
    ("progress/glossary", "Glossary progress", "U5"),
    ("qa", "QA scanner", "U6"),
    ("convert", "Compile EPUB / PDF", "U6"),
    ("headers", "Headers & metadata", "U6"),
    ("async", "Async batch", "U7"),
    ("review", "Review generator", "U7"),
    ("sdlxliff", "SDLXLIFF reviewer", "U7"),
    ("manga", "Manga translator", "U8"),
    ("rpgmaker", "RPG Maker", "U7"),
)

ROUTES: tuple[RouteSpec, ...] = (
    RouteSpec("home", "/", "Chat", ROOT, "U1"),
    # Chat (§2)
    RouteSpec("chat", "/chat/{cid:cid}", "Chat", ROOT, "U3"),
    RouteSpec("chat.settings", "/chat/{cid:cid}/settings", "Chat settings", SHEET, "U3", parent="chat"),
    RouteSpec("chat.attachments", "/chat/{cid:cid}/attachments", "Attachments", VIEW, "U7", parent="chat"),
    RouteSpec("chat.compose", "/chat/{cid:cid}/compose", "Composer", FULLSCREEN, "U3", parent="chat"),
    RouteSpec("chat.message", "/chat/{cid:cid}/m/{mid:hex12}", "Message", ROOT, "U3", parent="chat"),
    RouteSpec("chat.message.edit", "/chat/{cid:cid}/m/{mid:hex12}/edit", "Edit output", FULLSCREEN, "U3", parent="chat"),
    RouteSpec("series", "/series/{sid:id}", "Series", VIEW, "U9"),
    # Library and Reader (§3)
    RouteSpec("library", "/library", "Library", VIEW, "U5", query={"shelf": "shelf"}),
    RouteSpec("library.scan_raw", "/library/scan-raw", "Scan for raw", VIEW, "U5", parent="library"),
    RouteSpec(
        "library.book",
        "/library/book/{bid:hex12}",
        "Book",
        VIEW,
        "U5",
        parent="library",
        query={"tab": "book_tab", "filter": "group"},
    ),
    RouteSpec("library.book.metadata", "/library/book/{bid:hex12}/metadata", "Metadata", FULLSCREEN, "U5", parent="library"),
    RouteSpec("reader", "/reader/{bid:hex12}", "Reader", FULLSCREEN, "U5", query={"ch": "int", "mode": "reader_mode"}),
    # Jobs (§1.8)
    RouteSpec("jobs", "/jobs", "Jobs", VIEW, "U3"),
    RouteSpec("jobs.detail", "/jobs/{jid:id}", "Job", VIEW, "U3", parent="jobs"),
    RouteSpec("job", "/job/{jid:id}", "Job", VIEW, "U3", parent="jobs", alias_of="jobs.detail"),
    # Glossaries (§4.1)
    RouteSpec("glossary", "/glossary", "Glossaries", VIEW, "U6"),
    RouteSpec("glossary.unified", "/glossary/unified", "Unified glossary", VIEW, "U6", parent="glossary"),
    RouteSpec("glossary.parallel_pair", "/glossary/parallel-pair", "Parallel EPUB pair", VIEW, "U6", parent="glossary"),
    RouteSpec(
        "glossary.detail", "/glossary/{gid:hex12}", "Glossary", VIEW, "U6", parent="glossary", query={"tab": "glossary_tab"}
    ),
    RouteSpec("glossary.entry", "/glossary/{gid:hex12}/entry/{n:int}", "Glossary entry", SHEET, "U6", parent="glossary"),
    # Tools (§4.2-4.10)
    RouteSpec("tools", "/tools", "Tools", VIEW, "U6"),
    *(
        RouteSpec(
            "tools." + slug.replace("/", "."),
            f"/tools/{slug}",
            title,
            VIEW,
            milestone,
            parent="tools",
            query={"out": "hex12", "tab": "group"},
        )
        for slug, title, milestone in _TOOLS
    ),
    RouteSpec("tools.qa.report", "/tools/qa/report/{rid:hex12}", "QA report", VIEW, "U6", parent="tools.qa"),
    RouteSpec("tools.files", "/tools/files/{root:root}", "Files", VIEW, "U3", parent="tools"),
    RouteSpec("tools.files.folder", "/tools/files/{root:root}/{fid:hex12}", "Files", VIEW, "U3", parent="tools"),
    # UI_SPEC §4.10 (TextEditor with the file tools): U7
    RouteSpec("tools.text", "/tools/text/{fid:hex12}", "Text editor", FULLSCREEN, "U7", query={"hit": "int"}),
    # Settings (§4.11-4.16)
    RouteSpec("settings", "/settings", "Settings", VIEW, "U2"),
    RouteSpec("settings.section", "/settings/s/{section:section}", "Settings", VIEW, "U2", parent="settings", fragment="key"),
    _settings_page("models", "Model manager", "U4"),
    _settings_page("keys", "API keys", "U4"),
    RouteSpec("settings.keys.pool", "/settings/keys/{pool:pool}", "API keys", VIEW, "U4", parent="settings.keys"),
    _settings_page("accounts", "Accounts", "U3"),
    _settings_page("profiles", "Profiles & prompts", "U4"),
    RouteSpec("settings.profiles.detail", "/settings/profiles/{pid:slug}", "Profile", VIEW, "U4", parent="settings.profiles"),
    _settings_page("prefill", "Assistant prefill", "U4"),
    _settings_page("endpoints", "Endpoints", "U4"),
    _settings_page("appearance", "Appearance", "U2"),
    _settings_page("notifications", "Notifications", "U3"),
    _settings_page("storage", "Storage", "U2"),
    _settings_page("cloud", "Cloud sync & sharing", "U10"),
    _settings_page("backup", "Backup & restore", "U2"),
    _settings_page("import", "Import from desktop", "U2"),
    _settings_page("logs", "Logs & diagnostics", "U1"),
    # Diagnostic: the env the next translation run would get (HeadlessOwner + run_env), redacted.
    RouteSpec("settings.env_preview", "/settings/logs/env", "Env preview", VIEW, "U2", parent="settings.logs"),
    _settings_page("updates", "Updates", "U9"),
    _settings_page("about", "About", "U2"),
    _settings_page("danger", "Danger zone", "U2"),
    # Flows and handled routes
    RouteSpec("welcome", "/welcome", "Welcome", FULLSCREEN, "U3"),
    RouteSpec("oauth.return", "/oauth/return", "Sign-in return", HANDLED, "U0", query={"p": "slug", "nonce": "token"}),
    RouteSpec("selftest", "/__selftest__", "Self-test", HANDLED, "U0", query={"suite": "slug"}),
)

ROUTES_BY_NAME: dict[str, RouteSpec] = {spec.name: spec for spec in ROUTES}

# Static (placeholder-free) whitelisted paths; kept for callers of the U0 API.
ALLOWED_PATHS = frozenset(spec.pattern for spec in ROUTES if spec.is_static)


class RouteError(ValueError):
    """``build_route`` was asked for an unknown route or an invalid value."""


@dataclass(frozen=True)
class RouteMatch:
    path: str
    query: dict[str, str] = field(default_factory=dict)
    raw: str = ""
    deep_link: bool = False  # arrived as glossarion://app/...
    name: str = "home"
    params: dict[str, str] = field(default_factory=dict)
    fragment: Optional[str] = None

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        return self.query.get(key, default)

    @property
    def spec(self) -> RouteSpec:
        return ROUTES_BY_NAME[self.name]

    @property
    def presentation(self) -> str:
        return self.spec.presentation

    @property
    def route(self) -> str:
        """Canonical route string (path + validated query + fragment)."""
        return _render(self.path, self.query, self.fragment)


@dataclass(frozen=True)
class RouteRecord:
    raw: str
    accepted: bool
    path: Optional[str]
    reason: str
    at: float


def _render(path: str, query: Mapping[str, str], fragment: Optional[str]) -> str:
    text = path
    if query:
        text += "?" + urlencode(list(query.items()))
    if fragment:
        text += "#" + quote(fragment, safe="")
    return text


def _reject(reason: str) -> tuple[None, str]:
    return None, reason


_SEGMENTS: dict[str, tuple[tuple[str, Optional[str]], ...]] = {spec.name: spec.segments for spec in ROUTES}


def _match_spec(segments: list[str]) -> Optional[tuple[RouteSpec, dict[str, str]]]:
    for spec in ROUTES:
        pattern = _SEGMENTS[spec.name]
        if len(pattern) != len(segments):
            continue
        params: dict[str, str] = {}
        for (name, kind), value in zip(pattern, segments, strict=True):
            if kind is None:
                if value != name:
                    break
            elif _valid(kind, value):
                params[name] = value
            else:
                break
        else:
            return spec, params
    return None


def _home(raw: Any) -> RouteMatch:
    return RouteMatch(path="/", query={}, raw="" if raw is None else str(raw), deep_link=False, name="home")


def _parse(raw: Optional[str]) -> tuple[Optional[RouteMatch], str]:
    if raw is None:
        return _home(raw), "empty"
    text = str(raw).strip()
    if not text:
        return _home(raw), "empty"
    if len(text) > _MAX_ROUTE_CHARS:
        return _reject("too long")
    if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in text):
        return _reject("control characters")

    deep_link = False
    head = text.split("?", 1)[0].split("#", 1)[0]
    if ":" in head and not head.startswith("/"):
        scheme = head.split(":", 1)[0].lower()
        if scheme in _BLOCKED_SCHEMES:
            return _reject(f"blocked scheme {scheme}:")
        try:
            parts = urlsplit(text)
            port = parts.port
        except ValueError:
            return _reject("unparseable URI")
        if parts.scheme.lower() != DEEP_LINK_SCHEME:
            return _reject(f"foreign scheme {parts.scheme or '?'}:")
        if (parts.hostname or "").lower() != DEEP_LINK_HOST or parts.username or parts.password or port:
            return _reject(f"foreign host {parts.netloc!r}")
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

    decoded = unquote(path)
    if decoded != path and any(ch in decoded for ch in ("/", "\\", ".")):
        return _reject("encoded path separators")
    if "\\" in path or "/../" in f"{path}/" or "/./" in f"{path}/" or "//" in path:
        return _reject("path traversal")
    if len(path) > 1:
        path = path.rstrip("/") or "/"

    segments = [] if path == "/" else path.strip("/").split("/")
    found = _match_spec(segments)
    if found is None:
        return _reject(f"not whitelisted: {path}")
    spec, params = found

    query: dict[str, str] = {}
    dropped: list[str] = []
    for key, value in parse_qsl(parts.query, keep_blank_values=True)[:_MAX_QUERY_ITEMS]:
        if len(key) > _MAX_KEY_CHARS or len(value) > _MAX_VALUE_CHARS:
            dropped.append(key[:16])
            continue
        kind = spec.query.get(key)
        if kind is None or not _valid(kind, value):
            dropped.append(key[:16])
            continue
        query.setdefault(key, value)

    fragment: Optional[str] = None
    if parts.fragment:
        candidate = unquote(parts.fragment)
        if spec.fragment is not None and _valid(spec.fragment, candidate):
            fragment = candidate
        else:
            dropped.append("#")

    name = spec.name
    if spec.alias_of is not None:
        canonical = ROUTES_BY_NAME[spec.alias_of]
        name = canonical.name
        path = _fill(canonical, params)
    match = RouteMatch(path=path, query=query, raw=text, deep_link=deep_link, name=name, params=params, fragment=fragment)
    reason = "ok" if not dropped else f"ok (dropped {', '.join(dropped)})"
    return match, reason


def _fill(spec: RouteSpec, params: Mapping[str, Any]) -> str:
    parts = []
    for name, kind in spec.segments:
        if kind is None:
            parts.append(name)
            continue
        if name not in params:
            raise RouteError(f"route {spec.name!r} needs {name!r}")
        value = str(params[name])
        if not _valid(kind, value):
            raise RouteError(f"invalid {name!r} for route {spec.name!r}")
        parts.append(value)
    return "/" + "/".join(parts) if parts else "/"


def parse_route(raw: Optional[str]) -> Optional[RouteMatch]:
    """Whitelisted ``RouteMatch`` for ``raw``, or ``None`` when it must be ignored."""
    match, _reason = _parse(raw)
    return match


def build_route(
    name: str,
    params: Optional[Mapping[str, Any]] = None,
    query: Optional[Mapping[str, Any]] = None,
    fragment: Optional[str] = None,
) -> str:
    """Render a whitelisted route; raises ``RouteError`` for anything not allowed.

    In-app navigation builds routes with this, so a route can never carry a
    file path or user text.
    """
    spec = ROUTES_BY_NAME.get(name)
    if spec is None:
        raise RouteError(f"unknown route {name!r}")
    path = _fill(spec, params or {})
    clean_query: dict[str, str] = {}
    for key, value in (query or {}).items():
        if value is None:
            continue
        kind = spec.query.get(key)
        text = str(value)
        if kind is None or not _valid(kind, text):
            raise RouteError(f"query {key!r} not allowed for route {name!r}")
        clean_query[key] = text
    if fragment is not None:
        if spec.fragment is None or not _valid(spec.fragment, fragment):
            raise RouteError(f"fragment not allowed for route {name!r}")
    return _render(path, clean_query, fragment)


def launch_links(items: Any) -> list[tuple[str, str]]:
    """(id, ``glossarion://`` URI) of cold-start deep links delivered as shared items.

    On iOS receive_sharing_intent claims every UIScene connection, so Flutter
    skips a cold-start ``glossarion://app/...`` link; the native extension
    re-delivers it as ``SharedItem(kind="url", source="launch")``.
    """
    links = []
    for item in items or []:
        if not isinstance(item, dict) or item.get("kind") != "url" or item.get("source") != "launch":
            continue
        text = str(item.get("text") or item.get("uri") or "")
        if text.lower().startswith(DEEP_LINK_SCHEME + ":"):
            links.append((str(item.get("id") or text), text))
    return links


class Router:
    """Records every route seen and returns a match only for whitelisted ones.

    Thread-safe; ``history`` keeps the last 100 records for diagnostics (the
    device-checks screen shows it to prove Open-with never changes ``page.route``).
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
