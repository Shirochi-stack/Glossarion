"""Library view models (UI_SPEC §3.1-§3.3): book cards, density presets, filters. Pure Python, no Flet.

A ``CardModel`` holds exactly what ``BookCard`` renders. Its progress strings come
from the shared ``library_core.card_progress_view(book)`` (computed with the scan on
the io pool); when a build's core does not offer it yet, :func:`card_progress`
renders the same desktop vocabulary (``epub_library._BookCard`` 4927-5085) from
the scanned ``translation_state`` and counts - presentation only, the counts and
the state are always the scanner's. Host tests check both agree.

Exact strings (UI_SPEC §3.2):

* ribbons ``NOT STARTED`` / ``IN PROGRESS`` / ``READY TO COMPILE`` /
  ``OUTDATED PROGRESS`` / ``⚙ COMPILING…`` (none when completed);
* pills ``🆕 Not started``, ``⏳ d/t`` (``⏳ In progress`` without a total) + ``NN%``
  (floored), ``✨ Ready to compile (d/t)``, ``⚠ Outdated Progress file``;
* info ``x.x MB`` / ``N KB`` + type badge ``📕EPUB`` … ``📁FOLDER``;
* warnings ``⚠ missing raw``, ``⚠ +N`` (+ the mobile ``QA ⚠ N``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

from glossarion_mobile.services.library import book_key
from glossarion_mobile.ui.library.colors import type_badge

__all__ = [
    "CardModel",
    "CardWarning",
    "DENSITY_ORDER",
    "DENSITY_LABELS",
    "DEFAULT_DENSITY",
    "DEFAULT_PAGE_SIZE",
    "FORMATS",
    "FilterState",
    "PAGE_SIZES",
    "SORTS",
    "STATE_FILTERS",
    "build_card",
    "card_model_for",
    "card_progress",
    "density_preset",
    "effective_density",
    "info_line",
    "language_badge",
    "next_page_end",
    "next_tristate",
    "page_size_value",
    "SCROLL_APPEND_PX",
    "wants_next_page",
    "size_text",
    "visible_books",
]

#: Desktop ``_ALL_SIZES`` keys (``epub_library_card_size``) in order, with their toolbar labels.
DENSITY_ORDER = ("2xs", "xs", "compact", "normal", "large", "xl", "2xl", "3xl", "4xl", "5xl", "6xl")
DENSITY_LABELS = {
    "2xs": "2XS", "xs": "XS", "compact": "S", "normal": "M", "large": "L", "xl": "XL",
    "2xl": "2XL", "3xl": "3XL", "4xl": "4XL", "5xl": "5XL", "6xl": "6XL",
}
DEFAULT_DENSITY = "compact"  # desktop default (S)

# UI_SPEC §3.1 density table (card_w dp; cover_h keeps the desktop ``_SIZE_PRESETS`` ratio). Used only
# when library_core does not expose ``SIZE_PRESETS`` / ``_SIZE_PRESETS``.
_SPEC_PRESETS = {
    "2xs": (78, 115), "xs": (92, 136), "compact": (110, 162), "normal": (140, 203), "large": (180, 260),
    "xl": (230, 335), "2xl": (290, 422), "3xl": (360, 520), "4xl": (440, 635), "5xl": (530, 762),
    "6xl": (630, 912),
}

#: Card text block below the cover: title (2 lines) + info row + warnings row + pill + padding.
CARD_TEXT_HEIGHT = 112

SORTS = (("date", "Date"), ("name", "A-Z"), ("size", "Size"))  # SORT_DATE / SORT_NAME / SORT_SIZE
FORMATS = (("all", "All"), ("epub", "EPUB"), ("txt", "TXT"), ("pdf", "PDF"), ("html", "HTML"), ("image", "IMG"))
PAGE_SIZES = (("20", "20"), ("50", "50"), ("100", "100"), ("250", "250"), ("500", "500"), ("all", "All"))
DEFAULT_PAGE_SIZE = 20
#: A paged book list appends its next page when the scroll comes this close to the end.
SCROLL_APPEND_PX = 600

#: Tri-state shelf filters (UI_SPEC §3.1 Filter tab): id -> (label, predicate over the scanned row).
STATE_FILTERS: dict[str, tuple[str, Callable[[Mapping[str, Any]], bool]]] = {
    "in_progress": ("In progress", lambda b: b.get("translation_state") == "in_progress"),
    "ready_to_compile": ("Ready to compile", lambda b: b.get("translation_state") == "ready_to_compile"),
    "not_started": ("Not started", lambda b: b.get("translation_state") == "not_started"),
    "outdated_progress": ("Outdated", lambda b: b.get("translation_state") == "outdated_progress"),
    "qa": ("Has QA failures", lambda b: int(b.get("failed_chapters") or 0) > 0),
    "missing_raw": ("Missing raw", lambda b: bool(b.get("missing_raw_file"))),
}


def density_preset(key: str, presets: Optional[Mapping[str, Any]] = None) -> tuple:
    """``(card_w, cover_h)`` in dp for a desktop card-size key (shared ``_SIZE_PRESETS`` when given)."""
    key = key if key in DENSITY_ORDER else DEFAULT_DENSITY
    if presets and key in presets:
        entry = presets[key]
        try:
            return int(entry["card_w"]), int(entry["cover_h"])
        except (KeyError, TypeError, ValueError):
            pass
    return _SPEC_PRESETS[key]


def effective_density(key: str, text_scale: float = 1.0) -> str:
    """UI_SPEC §7.5: at >= 160% text the grid drops one density step (M -> L); the stored key is untouched."""
    key = key if key in DENSITY_ORDER else DEFAULT_DENSITY
    if text_scale >= 1.6:
        index = DENSITY_ORDER.index(key)
        return DENSITY_ORDER[min(index + 1, len(DENSITY_ORDER) - 1)]
    return key


def page_size_value(value: Any) -> int:
    """``epub_library_page_size`` -> append increment (0 = All)."""
    text = str(value if value is not None else DEFAULT_PAGE_SIZE).strip().lower()
    if text in ("all", "0", "-1"):
        return 0
    try:
        number = int(float(text))
    except ValueError:
        return DEFAULT_PAGE_SIZE
    return number if number > 0 else 0


def next_page_end(rendered: int, total: int, page_size: int) -> int:
    """Library paging (Library home, the chat's Library picker): where the next page of ``total`` rows
    ends when ``rendered`` are built; ``page_size`` 0 (``page_size_value`` of "All") builds them all."""
    return min(int(total), int(rendered) + (int(page_size) if page_size > 0 else int(total)))


def wants_next_page(e: Any, rendered: int, total: int) -> bool:
    """An ``on_scroll`` event within ``SCROLL_APPEND_PX`` of the end while rows are still unbuilt."""
    pixels = getattr(e, "pixels", None)
    maximum = getattr(e, "max_scroll_extent", None)
    if pixels is None or maximum is None:
        return False
    return maximum - pixels < SCROLL_APPEND_PX and rendered < total


def size_text(size: Any) -> str:
    """Desktop card size label: ``x.x MB`` from 1 MB, else ``N KB``."""
    try:
        value = float(size or 0)
    except (TypeError, ValueError):
        value = 0.0
    mb = value / (1024 * 1024)
    return f"{mb:.1f} MB" if mb >= 1 else f"{value / 1024:.0f} KB"


def language_badge(book: Mapping[str, Any]) -> Optional[str]:
    meta = book.get("metadata_json") or {}
    lang = str(meta.get("language") or "").strip() if isinstance(meta, Mapping) else ""
    if not lang:
        return None
    code = lang.split("-")[0].split("_")[0]
    return (code if len(code) <= 3 else code[:2]).upper()


@dataclass(frozen=True)
class CardWarning:
    text: str
    role: str  # "missing_raw" | "conflicts" | "qa"
    tooltip: str = ""


@dataclass(frozen=True)
class CardProgress:
    state: str
    pill_text: str
    pct_text: str  # "" unless an in-progress book has a total
    ribbon_text: str
    pct: int
    fraction: Optional[float]


@dataclass(frozen=True)
class CardModel:
    key: str
    bid: str
    title: str
    full_title: str
    size_text: str
    type_kind: str  # TYPE_BADGES key
    type_emoji: str
    type_label: str
    language: Optional[str]
    warnings: tuple = ()
    state: str = "completed"
    pill_text: Optional[str] = None
    pct_text: str = ""
    ribbon_text: Optional[str] = None
    ribbon_state: str = ""  # RIBBON_COLORS key
    progress: Optional[float] = None  # 3 dp cover bar (in progress)
    has_continue: bool = False
    selected: bool = False
    signature: Any = None
    tooltip: str = ""

    @property
    def semantics(self) -> str:
        parts = [self.full_title, self.type_label, self.size_text]
        if self.pill_text:
            parts.append(self.pill_text + (f" {self.pct_text}" if self.pct_text else ""))
        parts.extend(w.text for w in self.warnings)
        return ", ".join(p for p in parts if p)


def card_progress(book: Mapping[str, Any]) -> Optional[CardProgress]:
    """The card's pill and ribbon (``_BookCard`` in-progress indicator); None when there is none."""
    if not book.get("is_in_progress"):
        return None
    total = int(book.get("total_chapters", 0) or 0)
    done = int(book.get("completed_chapters", 0) or 0)
    state = book.get("translation_state") or ("in_progress" if total else "not_started")
    if state == "completed":
        return None
    pct = int((done * 100) // total) if total else 0  # floor: 216/217 never reads 100%
    fraction = (done / total) if total else None
    if state == "outdated_progress":
        return CardProgress(state, "⚠ Outdated Progress file", "", "OUTDATED PROGRESS", pct, fraction)
    if state == "not_started":
        return CardProgress(state, "\U0001f195 Not started", "", "NOT STARTED", pct, fraction)
    if state == "ready_to_compile":
        text = f"✨ Ready to compile ({done}/{total})" if total else "✨ Ready to compile"
        return CardProgress(state, text, "", "READY TO COMPILE", pct, fraction)
    text = f"⏳ {done}/{total}" if total else "⏳ In progress"
    return CardProgress("in_progress", text, f"{pct}%" if total else "", "IN PROGRESS", pct, fraction)


def _from_view(view: Mapping[str, Any], book: Mapping[str, Any]) -> Optional[CardProgress]:
    """A ``library_core.card_progress_view`` dict as ``CardProgress`` (None: no pill)."""
    pill = view.get("pill_text", view.get("pill"))
    ribbon = view.get("ribbon_text", view.get("ribbon"))
    if not pill and not ribbon:
        return None
    state = str(view.get("state") or view.get("pill_role") or view.get("role") or book.get("translation_state")
                or "in_progress")
    state = {"outdated": "outdated_progress", "ready": "ready_to_compile"}.get(state, state)
    try:
        pct = int(view.get("pct") or 0)
    except (TypeError, ValueError):
        pct = 0
    total = int(book.get("total_chapters", 0) or 0)
    done = int(book.get("completed_chapters", 0) or 0)
    if "show_pct" in view:
        pct_text = view.get("pct_text") if view.get("show_pct") else ""
    else:
        pct_text = view.get("pct_text")
        if pct_text is None:
            pct_text = f"{pct}%" if state == "in_progress" and total else ""
    return CardProgress(state, str(pill or ""), str(pct_text or ""), str(ribbon or ""), pct,
                        (done / total) if total else None)


def _warnings(book: Mapping[str, Any], view: Optional[Mapping[str, Any]]) -> tuple:
    out: list[CardWarning] = []
    if book.get("missing_raw_file"):
        out.append(CardWarning(
            "⚠ missing raw", "missing_raw",
            "The raw source file for this book can't be found on disk — Library/Raw, source_epub.txt, and "
            "the raw-inputs registry all came up empty. The compiled output is still readable, but actions that "
            "need the raw source (Read raw, Load for translation) are disabled."))
    conflicts = list(book.get("compiled_conflicts") or [])
    if conflicts:
        lines = ["Extra compiled files in this folder:"]
        for item in conflicts[:8]:
            name, kind = (item[0], item[1]) if isinstance(item, (tuple, list)) and len(item) > 1 else (item, "")
            lines.append(f"  • {name} ({str(kind).upper()})")
        if len(conflicts) > 8:
            lines.append(f"  … and {len(conflicts) - 8} more.")
        out.append(CardWarning(f"⚠ +{len(conflicts)}", "conflicts", "\n".join(lines)))
    failed = int(book.get("failed_chapters", 0) or 0)
    if failed > 0:
        out.append(CardWarning(f"QA ⚠ {failed}", "qa", f"{failed} chapter(s) failed or failed QA"))
    return tuple(out)


def info_line(book: Mapping[str, Any]) -> tuple:
    """``(kind, size text)``; folder cards show their workspace kind's badge."""
    kind = str(book.get("type") or "epub")
    if kind == "in_progress":
        workspace = str(book.get("workspace_kind") or "other").lower()
        kind = workspace if workspace in ("epub", "pdf", "txt", "html", "image") else "in_progress"
    return kind, size_text(book.get("size"))


def build_card(
    book: Mapping[str, Any],
    *,
    key: str,
    bid: str,
    view: Optional[Mapping[str, Any]] = None,
    raw_titles: bool = False,
    raw_title: Optional[str] = None,
    compiling: bool = False,
    has_continue: bool = False,
    selected: bool = False,
    signature: Any = None,
    dark: bool = True,
    badge_text: Optional[str] = None,
    size_label: Optional[str] = None,
) -> CardModel:
    """The ``CardModel`` of one scanned book (``badge_text`` / ``size_label``: the shared
    ``card_type_badge`` / ``card_size_text`` strings when the core has them)."""
    name = str(book.get("name") or "")
    title = (raw_title or name) if raw_titles else name
    progress = _from_view(view, book) if view else None
    if progress is None and not view:
        progress = card_progress(book)
    kind, size = info_line(book)
    emoji, label, _color = type_badge(kind, dark)
    if badge_text:
        emoji, label = "", badge_text
    if size_label:
        size = size_label
    ribbon_text = progress.ribbon_text if progress is not None else None
    ribbon_state = progress.state if progress is not None else ""
    if compiling:
        ribbon_text, ribbon_state = "⚙ COMPILING…", "compiling"
    return CardModel(
        key=key,
        bid=bid,
        title=title,
        full_title=title,
        size_text=size,
        type_kind=kind,
        type_emoji=emoji,
        type_label=label,
        language=language_badge(book),
        warnings=_warnings(book, view),
        state=str(book.get("translation_state") or ("in_progress" if book.get("is_in_progress") else "completed")),
        pill_text=progress.pill_text if progress is not None else None,
        pct_text=progress.pct_text if progress is not None else "",
        ribbon_text=ribbon_text,
        ribbon_state=ribbon_state,
        progress=progress.fraction if progress is not None and progress.state == "in_progress" else None,
        has_continue=has_continue,
        selected=selected,
        signature=signature,
        tooltip=name if not raw_titles else f"{title}\n{name}",
    )


def card_model_for(
    service: Any,
    book: Mapping[str, Any],
    *,
    views: Optional[Mapping[str, Any]],
    raw_titles: bool,
    dark: bool,
    selected: bool = False,
    has_continue: bool = False,
    key: Optional[str] = None,
    signatures: Optional[Mapping[str, Any]] = None,
) -> CardModel:
    """The ``CardModel`` the Library renders for ``book``, from the ``LibraryService`` (route id, the
    shared card badge / size label, the raw title when ``raw_titles`` is on, the compiling ribbon) and
    the last scan's ``views`` (card progress views) / ``signatures`` (default: the service snapshot's),
    both keyed by ``key`` (default ``book_key(book)``). The one card model of the Library home, the
    chat's Library picker and any other Library list."""
    key = key or book_key(book)
    if signatures is None:
        signatures = getattr(getattr(service, "snapshot", None), "signatures", None) or {}
    badge, size = service.card_badge(book)
    return build_card(
        book, key=key, bid=service.bid_for(book), view=(views or {}).get(key), raw_titles=raw_titles,
        raw_title=service.raw_title(book) if raw_titles else None, compiling=service.is_compiling(book),
        has_continue=has_continue, selected=selected, signature=signatures.get(key), dark=dark,
        badge_text=badge, size_label=size)


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


def next_tristate(value: Optional[bool]) -> Optional[bool]:
    """Filter chip cycle: off -> only -> hide -> off."""
    if value is None:
        return True
    if value is True:
        return False
    return None


@dataclass
class FilterState:
    query: str = ""
    fmt: str = "all"  # FORMAT_*
    sort: str = "date"  # SORT_*
    reverse: bool = False
    states: dict = field(default_factory=dict)  # STATE_FILTERS id -> True (only) / False (hide)

    @property
    def active_count(self) -> int:
        return (1 if self.fmt != "all" else 0) + sum(1 for v in self.states.values() if v is not None) + (
            1 if self.reverse else 0)

    def state_ok(self, book: Mapping[str, Any]) -> bool:
        includes = [k for k, v in self.states.items() if v is True and k in STATE_FILTERS]
        excludes = [k for k, v in self.states.items() if v is False and k in STATE_FILTERS]
        if includes and not any(STATE_FILTERS[k][1](book) for k in includes):
            return False
        return not any(STATE_FILTERS[k][1](book) for k in excludes)


def visible_books(
    books: Iterable[Mapping[str, Any]],
    state: FilterState,
    *,
    matches: Callable[[Mapping[str, Any], str], bool],
    format_of: Callable[[Mapping[str, Any]], str],
    sort: Callable[[Sequence[Mapping[str, Any]], str, bool], list],
) -> list:
    """Query (shared ``book_matches_query``), format (``format_of_book``), state chips, then sort."""
    query = state.query.strip()
    out = []
    for book in books:
        if query and not matches(book, query):
            continue
        if state.fmt != "all" and format_of(book) != state.fmt:
            continue
        if not state.state_ok(book):
            continue
        out.append(book)
    return sort(out, state.sort, state.reverse)
