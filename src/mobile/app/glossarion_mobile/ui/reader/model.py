"""Reader model (UI_SPEC §3.11): settings, scopes, CSS overrides, positions, labels.

Pure Python (no Flet, no backend import), host-tested on 3.10+.

Settings
  The desktop reader persists ``epub_reader_font_size`` (14), ``_line_spacing``
  (1.8), ``_theme`` (0 = Dark), ``_font_family`` ("Embedded CSS"), ``_layout``
  (``single_page``), ``_show_raw`` and ``_native_toc`` in ``config.json``
  (epub_library ``EpubReaderDialog.__init__``). The Aa sheet's scope switch
  writes those keys for **All books** (sparse ``MobileConfigStore.set_many``) and
  a per-book override for **This book** (Prefs ``reader_book_settings``).
  Mobile-only switches (follow app theme, tap-zone paging, keep screen on, show
  progress %, lightweight reader) live in Prefs ``reader_prefs`` and never reach
  ``config.json``. Clamps are the desktop ones: size 8-32 pt
  (``_change_font_size``), spacing 1.0-3.0 (``_on_spacing_changed``).

Layout values are the desktop config values (``LAYOUT_SINGLE = "single_page"``,
``LAYOUT_DOUBLE``, ``LAYOUT_SCROLL``, ``LAYOUT_ALL``). Double page needs a
tablet in landscape (>= 900 dp wide and wider than tall); elsewhere it reads as
Single page without rewriting the saved value.

Positions
  ``capture_hint``/``page_from_hint`` are the desktop proportional page hint
  (``_capture_position_hint``/``_apply_pending_page_hint``: the last page stays
  the last page, otherwise ``round(page / (pages - 1) * (count - 1))``), used for
  Original/Translated/Bilingual switches, saved positions and style changes;
  ``clamp_page`` is ``_clamp_page_for_layout`` (spreads start on even pages).
  The page applies a position where the page count is measured: the shared
  ``reader_doc`` mobile shell starts at ``initial_page`` / ``#f=<fraction>``
  (``bridge.position_fragment``) and the Reader's extras restyle with the hint
  rule above.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Callable, Mapping, Optional, Sequence

__all__ = [
    "BILINGUAL",
    "CONFIG_KEYS",
    "DEFAULTS",
    "FONT_FAMILIES",
    "FONT_SIZE_RANGE",
    "LAYOUT_ALL",
    "LAYOUT_DOUBLE",
    "LAYOUT_LABELS",
    "LAYOUT_SCROLL",
    "LAYOUT_SINGLE",
    "LAYOUTS",
    "LINE_SPACING_RANGE",
    "MOBILE_DEFAULTS",
    "ORIGINAL",
    "Position",
    "READER_MODES",
    "ReaderSettings",
    "TocRow",
    "TRANSLATED",
    "book_percent",
    "capture_hint",
    "chapter_label",
    "clamp_font_size",
    "clamp_line_spacing",
    "clamp_page",
    "coerce_setting",
    "config_updates",
    "double_page_allowed",
    "EMBEDDED_CSS",
    "MARGIN_RANGE",
    "MODE_LABELS",
    "MODE_SHORT_LABELS",
    "effective_layout",
    "font_stack",
    "is_paged",
    "mode_availability",
    "override_css",
    "page_from_hint",
    "page_label",
    "progress_label",
    "py_round",
    "resolve_settings",
    "resume_label",
    "spread_for",
    "theme_for",
    "toc_rows",
]

# ---- layouts and modes -------------------------------------------------------------------

LAYOUT_SCROLL = "scroll"  # one chapter, scrollable
LAYOUT_SINGLE = "single_page"  # one chapter, viewport-paginated
LAYOUT_DOUBLE = "double_page"  # two pages side by side (tablet landscape)
LAYOUT_ALL = "all_scroll"  # all chapters concatenated, scrollable
LAYOUTS = (LAYOUT_SINGLE, LAYOUT_DOUBLE, LAYOUT_SCROLL, LAYOUT_ALL)
LAYOUT_LABELS = {
    LAYOUT_SINGLE: "Single page",
    LAYOUT_DOUBLE: "Double page",
    LAYOUT_SCROLL: "Scroll",
    LAYOUT_ALL: "Scroll all",
}

ORIGINAL = "original"
TRANSLATED = "translated"
BILINGUAL = "bilingual"
READER_MODES = (ORIGINAL, TRANSLATED, BILINGUAL)  # the route's ``mode`` enum
MODE_LABELS = {ORIGINAL: "Original", TRANSLATED: "Translated", BILINGUAL: "Bilingual"}
MODE_SHORT_LABELS = {ORIGINAL: "Orig", TRANSLATED: "Trans", BILINGUAL: "Both"}  # under 400 dp

FONT_SIZE_RANGE = (8, 32)
LINE_SPACING_RANGE = (1.0, 3.0)
EMBEDDED_CSS = "Embedded CSS"

#: Aa › Text font families: the desktop "Embedded CSS" plus generic families that exist on
#: every phone (the desktop combo lists the Windows/macOS system fonts by name).
FONT_FAMILIES = (EMBEDDED_CSS, "Serif", "Sans", "Mono")
_FONT_STACKS = {
    "serif": "Georgia, 'Noto Serif', 'Times New Roman', serif",
    "sans": "-apple-system, Roboto, 'Helvetica Neue', 'Noto Sans', Arial, sans-serif",
    "mono": "Menlo, 'Roboto Mono', 'Droid Sans Mono', Consolas, monospace",
}

#: Reader settings stored in config.json (shared with the desktop reader) and their defaults.
CONFIG_KEYS = {
    "font_size": "epub_reader_font_size",
    "line_spacing": "epub_reader_line_spacing",
    "theme": "epub_reader_theme",
    "font_family": "epub_reader_font_family",
    "layout": "epub_reader_layout",
    "show_raw": "epub_reader_show_raw",
    "native_toc": "epub_reader_native_toc",
}
DEFAULTS = {
    "font_size": 14,
    "line_spacing": 1.8,
    "theme": 0,
    "font_family": EMBEDDED_CSS,
    "layout": LAYOUT_SINGLE,
    "show_raw": False,
    "native_toc": False,
}
#: Mobile-only reader switches (Prefs ``reader_prefs``; never written to config.json).
MOBILE_DEFAULTS = {
    "follow_app_theme": False,
    "tap_zones": True,
    "keep_screen_on": False,
    "show_progress": True,
    "lightweight": False,
    "margins": 16,  # dp of horizontal page padding (UI_SPEC: 16-20 dp)
}
MARGIN_RANGE = (8, 40)


def clamp_font_size(value: Any) -> int:
    try:
        size = int(round(float(value)))
    except (TypeError, ValueError):
        size = DEFAULTS["font_size"]
    return max(FONT_SIZE_RANGE[0], min(FONT_SIZE_RANGE[1], size))


def clamp_line_spacing(value: Any) -> float:
    try:
        spacing = float(value)
    except (TypeError, ValueError):
        return DEFAULTS["line_spacing"]
    if spacing != spacing:  # NaN
        return DEFAULTS["line_spacing"]
    return round(max(LINE_SPACING_RANGE[0], min(LINE_SPACING_RANGE[1], spacing)), 2)


def _clamp_theme(value: Any, theme_count: int) -> int:
    try:
        index = int(value)
    except (TypeError, ValueError):
        return 0
    return index if 0 <= index < max(1, theme_count) else 0


def _clamp_margin(value: Any) -> int:
    try:
        margin = int(round(float(value)))
    except (TypeError, ValueError):
        margin = MOBILE_DEFAULTS["margins"]
    return max(MARGIN_RANGE[0], min(MARGIN_RANGE[1], margin))


@dataclass(frozen=True)
class ReaderSettings:
    """Effective reader settings for one book (config keys + per-book override + mobile prefs)."""

    font_size: int = DEFAULTS["font_size"]
    line_spacing: float = DEFAULTS["line_spacing"]
    theme: int = DEFAULTS["theme"]
    font_family: str = DEFAULTS["font_family"]
    layout: str = DEFAULTS["layout"]
    show_raw: bool = DEFAULTS["show_raw"]
    native_toc: bool = DEFAULTS["native_toc"]
    follow_app_theme: bool = MOBILE_DEFAULTS["follow_app_theme"]
    tap_zones: bool = MOBILE_DEFAULTS["tap_zones"]
    keep_screen_on: bool = MOBILE_DEFAULTS["keep_screen_on"]
    show_progress: bool = MOBILE_DEFAULTS["show_progress"]
    lightweight: bool = MOBILE_DEFAULTS["lightweight"]
    margins: int = MOBILE_DEFAULTS["margins"]
    overridden: frozenset = field(default_factory=frozenset)  # keys set for "This book"

    @property
    def embedded_css(self) -> bool:
        return (self.font_family or "").strip() == EMBEDDED_CSS

    def to_dict(self) -> dict:
        data = asdict(self)
        data["overridden"] = sorted(self.overridden)
        return data

    def with_changes(self, **changes: Any) -> "ReaderSettings":
        return replace(self, **changes)


def coerce_setting(key: str, value: Any, theme_count: int = 6) -> Any:
    """One setting value clamped / normalised like the desktop reader does."""
    if key == "font_size":
        return clamp_font_size(value)
    if key == "line_spacing":
        return clamp_line_spacing(value)
    if key == "theme":
        return _clamp_theme(value, theme_count)
    if key == "font_family":
        text = str(value or "").strip()
        return text or EMBEDDED_CSS
    if key == "layout":
        return value if value in LAYOUTS else LAYOUT_SINGLE
    if key == "margins":
        return _clamp_margin(value)
    return bool(value)


def resolve_settings(
    config_get: Callable[[str, Any], Any],
    book_overrides: Optional[Mapping[str, Any]] = None,
    mobile_prefs: Optional[Mapping[str, Any]] = None,
    *,
    theme_count: int = 6,
) -> ReaderSettings:
    """Effective settings: per-book override > config.json key > desktop default (+ mobile prefs)."""
    values: dict[str, Any] = {}
    overrides = dict(book_overrides or {})
    for name, key in CONFIG_KEYS.items():
        if name in overrides:
            raw = overrides[name]
        else:
            try:
                raw = config_get(key, DEFAULTS[name])
            except Exception:
                raw = DEFAULTS[name]
            if raw is None:
                raw = DEFAULTS[name]
        values[name] = coerce_setting(name, raw, theme_count)
    mobile = dict(MOBILE_DEFAULTS)
    mobile.update({k: v for k, v in dict(mobile_prefs or {}).items() if k in MOBILE_DEFAULTS})
    for name in MOBILE_DEFAULTS:
        raw = overrides[name] if name in overrides else mobile[name]
        values[name] = coerce_setting(name, raw, theme_count)
    overridden = frozenset(k for k in overrides if k in CONFIG_KEYS or k in MOBILE_DEFAULTS)
    return ReaderSettings(**values, overridden=overridden)


def config_updates(changes: Mapping[str, Any], *, theme_count: int = 6) -> dict:
    """``{config key: value}`` for the config-backed names in ``changes`` (the "All books" scope)."""
    out = {}
    for name, value in changes.items():
        key = CONFIG_KEYS.get(name)
        if key is not None:
            out[key] = coerce_setting(name, value, theme_count)
    return out


def double_page_allowed(width: Optional[float], height: Optional[float]) -> bool:
    """Double page only on tablets in landscape (>= 900 dp wide and wider than tall)."""
    try:
        w = float(width or 0)
        h = float(height or 0)
    except (TypeError, ValueError):
        return False
    return w >= 900 and w > h


def effective_layout(layout: str, width: Optional[float], height: Optional[float]) -> str:
    if layout == LAYOUT_DOUBLE and not double_page_allowed(width, height):
        return LAYOUT_SINGLE
    return layout if layout in LAYOUTS else LAYOUT_SINGLE


def is_paged(layout: str) -> bool:
    return layout in (LAYOUT_SINGLE, LAYOUT_DOUBLE)


def spread_for(layout: str) -> int:
    return 2 if layout == LAYOUT_DOUBLE else 1


# ---- themes and CSS ------------------------------------------------------------------------


def theme_for(themes: list, settings: ReaderSettings, *, app_dark: Optional[bool] = None) -> dict:
    """The ``READER_THEMES`` entry in use ("Follow app theme" picks Dark / Light by the app's brightness)."""
    if not themes:
        return {"name": "Dark", "bg": "#1e1e1e", "fg": "#d4d4d4", "heading": "#c8c8f0", "link": "#6c9bd2",
                "code_bg": "#252530", "border": "#333333"}
    index = settings.theme
    if settings.follow_app_theme and app_dark is not None:
        wanted = "dark" if app_dark else "light"
        for i, theme in enumerate(themes):
            if str(theme.get("name", "")).strip().lower() == wanted:
                index = i
                break
    if not 0 <= index < len(themes):
        index = 0
    return dict(themes[index])


def font_stack(family: str) -> Optional[str]:
    """CSS font stack for an Aa family (``None`` for Embedded CSS: the book's own CSS decides)."""
    text = (family or "").strip()
    if not text or text == EMBEDDED_CSS:
        return None
    generic = _FONT_STACKS.get(text.lower())
    if generic is not None:
        return generic
    clean = text.replace("'", "").replace('"', "").replace(";", "").replace("{", "").replace("}", "")
    return f"'{clean}', Georgia, 'Noto Serif', serif"


def _css_color(value: Any, fallback: str) -> str:
    text = str(value or "").strip()
    if text.startswith("#") and 4 <= len(text) <= 9 and all(c in "0123456789abcdefABCDEF" for c in text[1:]):
        return text
    return fallback


def override_css(theme: Mapping[str, Any], settings: ReaderSettings, *, layout: Optional[str] = None) -> str:
    """The live ``<style id="glrdr-live">`` text: theme colours + typography + page margins.

    Applied with ``GLRDR.applyStyle(css)`` on every Aa change (no reload) and baked into
    every new document, so a re-rendered chapter looks like the live preview.
    """
    layout = layout or settings.layout
    bg = _css_color(theme.get("bg"), "#1e1e1e")
    fg = _css_color(theme.get("fg"), "#d4d4d4")
    heading = _css_color(theme.get("heading"), fg)
    link = _css_color(theme.get("link"), fg)
    code_bg = _css_color(theme.get("code_bg"), bg)
    border = _css_color(theme.get("border"), fg)
    px = int(round(settings.font_size * 96 / 72))  # desktop: px = pt * 96 / 72
    spacing = clamp_line_spacing(settings.line_spacing)
    stack = font_stack(settings.font_family)
    margin = _clamp_margin(settings.margins)
    text_target = "#columns" if is_paged(layout) else "body"
    typography = f"font-size: {px}px !important; line-height: {spacing} !important;"
    if stack:
        typography += f" font-family: {stack} !important;"
    rules = [
        f"html, body {{ background: {bg} !important; color: {fg} !important; }}",
        f"{text_target} {{ {typography} }}",
        f"h1, h2, h3, h4, h5, h6 {{ color: {heading} !important; }}",
        f"a {{ color: {link} !important; }}",
        f"code {{ background: {code_bg} !important; }}",
        f"hr {{ border-color: {border} !important; }}",
        "mark.glrdr-hit { background: #ffd54f; color: #000; border-radius: 2px; }",
    ]
    # Page margins never go inside the device's safe area (notches, rounded corners).
    sides = (f"padding-left: max({margin}px, env(safe-area-inset-left)) !important; "
             f"padding-right: max({margin}px, env(safe-area-inset-right)) !important;")
    rules.append(f"{'#content' if is_paged(layout) else 'body'} {{ {sides} }}")
    return "\n".join(rules)


# ---- positions --------------------------------------------------------------------------------


def py_round(value: float) -> int:
    """Python's ``round`` (half to even); the page JS uses the same rule."""
    return int(round(value))


def clamp_page(page: Any, count: Any, spread: int = 1) -> int:
    """``_clamp_page_for_layout``: a valid single page, or a spread start (even page) for double page."""
    try:
        page = int(page)
    except (TypeError, ValueError):
        page = 0
    try:
        count = int(count)
    except (TypeError, ValueError):
        count = 0
    count = max(1, count)
    if spread == 2:
        page = max(0, page - (page % 2))
        last_start = max(0, count - 1)
        last_start -= last_start % 2
        return max(0, min(page, last_start))
    return max(0, min(page, count - 1))


def capture_hint(page: Any, pages: Any) -> dict:
    """``_capture_position_hint``: ``{"last": bool, "fraction": page / (pages - 1)}``."""
    try:
        page = int(page or 0)
        pages = int(pages or 0)
    except (TypeError, ValueError):
        page, pages = 0, 0
    last = pages > 0 and page >= pages - 1
    fraction = max(0.0, min(1.0, page / (pages - 1))) if pages > 1 else 0.0
    return {"last": bool(last), "fraction": float(fraction)}


def page_from_hint(hint: Optional[Mapping[str, Any]], count: Any, spread: int = 1) -> int:
    """``_apply_pending_page_hint``: last page stays last, else ``round(fraction * (count - 1))``."""
    try:
        count = max(1, int(count))
    except (TypeError, ValueError):
        count = 1
    hint = hint or {}
    if hint.get("last"):
        target = count - 1
    elif hint.get("fraction") is not None:
        try:
            fraction = float(hint.get("fraction") or 0.0)
        except (TypeError, ValueError):
            fraction = 0.0
        fraction = 0.0 if fraction != fraction else max(0.0, min(1.0, fraction))
        target = py_round(fraction * (count - 1))
    else:
        target = hint.get("page") or 0
    return clamp_page(target, count, spread)


@dataclass
class Position:
    """Where the reader is: chapter index/href, page within the chapter and its fraction."""

    chapter: int = 0
    href: str = ""
    page: int = 0
    pages: int = 0
    fraction: float = 0.0  # page / (pages - 1) in paged layouts; scroll fraction otherwise
    last: bool = False
    mode: str = TRANSLATED

    def hint(self) -> dict:
        return {"last": bool(self.last), "fraction": float(self.fraction)}

    def to_pref(self) -> dict:
        return {"href": self.href, "fraction": self.fraction, "page": self.page, "mode": self.mode,
                "chapter": self.chapter, "pages": self.pages, "last": self.last}

    @classmethod
    def from_pref(cls, data: Optional[Mapping[str, Any]], filenames: list) -> Optional["Position"]:
        """A saved ``reader_positions`` entry resolved against the book's chapter filenames."""
        if not isinstance(data, Mapping):
            return None
        href = str(data.get("href") or "")
        index = _index_for_href(href, filenames)
        if index is None:
            try:
                index = int(data.get("chapter"))
            except (TypeError, ValueError):
                return None
            if not 0 <= index < len(filenames):
                return None
        try:
            fraction = float(data.get("fraction") or 0.0)
        except (TypeError, ValueError):
            fraction = 0.0
        fraction = 0.0 if fraction != fraction else max(0.0, min(1.0, fraction))
        try:
            page = max(0, int(data.get("page") or 0))
        except (TypeError, ValueError):
            page = 0
        try:
            pages = max(0, int(data.get("pages") or 0))
        except (TypeError, ValueError):
            pages = 0
        mode = data.get("mode") if data.get("mode") in READER_MODES else TRANSLATED
        return cls(chapter=index, href=filenames[index] if index < len(filenames) else href, page=page,
                   pages=pages, fraction=fraction, last=bool(data.get("last")), mode=mode)

    @property
    def at_start(self) -> bool:
        return self.chapter == 0 and self.page == 0 and self.fraction <= 0.0 and not (self.last and self.pages > 1)


def _base(name: Any) -> str:
    return str(name or "").replace("\\", "/").rsplit("/", 1)[-1].lower()


def _index_for_href(href: str, filenames: list) -> Optional[int]:
    wanted = _base(href.split("#", 1)[0])
    if not wanted:
        return None
    for index, name in enumerate(filenames):
        if _base(name) == wanted:
            return index
    return None


def book_percent(index: int, fraction: float, total: int) -> int:
    """Whole-book progress (floor), counting the current chapter's fraction."""
    if total <= 0:
        return 0
    value = (max(0, index) + max(0.0, min(1.0, float(fraction or 0.0)))) / float(total)
    return max(0, min(100, int(math.floor(value * 100 + 1e-9))))


def progress_label(display_number: Any, total: int, percent: int) -> str:
    """Bottom bar: ``"Ch 12/48 · 43%"``."""
    return f"Ch {display_number}/{total} · {percent}%"


def page_label(page: int, count: int, spread: int = 1) -> str:
    """Paged layouts: ``"Page 3/9"`` (1-based; a spread shows its first page)."""
    count = max(1, int(count or 1))
    return f"Page {min(count, max(0, int(page or 0)) + 1)}/{count}"


def resume_label(display_number: Any, percent: int) -> str:
    """Snackbar text offered on open when a saved position exists."""
    return f"Resume at Ch {display_number} · {percent}%"


def chapter_label(display_number: Any, title: str) -> str:
    title = (title or "").strip()
    return f"{display_number}. {title}" if title else f"Chapter {display_number}"


def mode_availability(*, has_alternate: bool, chapter_has_raw: bool, chapter_has_translation: bool) -> dict:
    """Which of Original / Translated / Bilingual can be chosen for the current chapter.

    Original and Translated exist when the book has a second flavour (an overlay of
    translated responses, a dual raw/compiled path or raw workspace content);
    Bilingual needs both flavours of this chapter.
    """
    return {
        ORIGINAL: bool(has_alternate and chapter_has_raw),
        TRANSLATED: True,
        BILINGUAL: bool(has_alternate and chapter_has_raw and chapter_has_translation),
    }


# ---- chapters drawer rows ---------------------------------------------------------------


@dataclass(frozen=True)
class TocRow:
    chapter: int  # reader chapter index the row opens
    title: str
    number: Any = None  # display number (chapter rows) or None (native TOC rows)
    status: str = ""  # overlay status of that chapter ("" = none)
    fragment: str = ""  # native TOC anchor inside the chapter


def toc_rows(titles: Sequence[str], numbers: Sequence[Any], statuses: Sequence[str],
             native: Optional[Sequence[dict]] = None) -> list[TocRow]:
    """The drawer rows: mapped native TOC entries when given, else one row per chapter."""
    if native:
        rows = []
        for entry in native:
            try:
                index = int(entry.get("chapter_index", 0))
            except (TypeError, ValueError):
                continue
            if not 0 <= index < len(titles):
                continue
            rows.append(TocRow(chapter=index, title=str(entry.get("title") or "Section"),
                               status=statuses[index] if index < len(statuses) else "",
                               fragment=str(entry.get("fragment") or "")))
        return rows
    return [TocRow(chapter=i, title=str(title or ""), number=numbers[i] if i < len(numbers) else i + 1,
                   status=statuses[i] if i < len(statuses) else "") for i, title in enumerate(titles)]


def now() -> float:
    return time.time()
