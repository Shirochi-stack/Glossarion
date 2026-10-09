"""Book page › Overview (UI_SPEC §3.6; epub_library ``BookDetailsDialog`` hero).

Hero (cover 120 x 180 phone / 240 x 340 tablet, title, author, chips 🌐 language ·
📅 year · type), the in-progress strip "⏳  Translation in progress — d/t chapters
(p%)", the reading actions ("📖  Start reading" / "📖  Read translated" /
"📖  Read raw source" + tonal "📜  Read raw source", "Continue · Ch 12 · 43%"),
the icon row (Translate… · Compile · Translate Metadata · Share · Files; each
dimmed with a reason when its target can't be resolved), SYNOPSIS (4 lines,
expandable; "No synopsis available."), METADATA (📘 Title · ✍️ Author · 🏛️
Publisher · 🌐 Language · 📅 Year, "—" when missing; ✏️ Edit), TAGS and "At a
glance" (the Chapters stats chips, the glossary summary, the last job; a book with no output
workspace shows its EPUB's "📖 N chapters"). The workspace (Compile, Files, ✏️ Edit) is the
book's resolved one (``progress_model.book_workspace``).

Values come from ``library_core.load_book_details`` (``metadata.json`` first, then
the OPF, then the card) and its display helpers when the core offers them.

Until the preview-phase details arrive the hero is a ``Skeleton("hero")`` (UI_SPEC §7.4). On a large
phone in landscape (UI_SPEC §1.1) the same controls are arranged in two columns: hero, strip, read
buttons and icon row | synopsis, metadata, tags and at a glance (``apply_layout``, called from
``BookPageScreen.apply_size_class``; each arrangement gets fresh keys, Flet 1.0.3 freezes a subtree
re-rendered under the same key).
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Optional

import flet as ft

from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.components.empty_state import HALGAKOS_ASSET
from glossarion_mobile.ui.components.skeleton import Skeleton
from glossarion_mobile.ui.library import progress_model as pm
from glossarion_mobile.ui.library.common import (
    OPENS_ELSEWHERE_REASON,
    icon_button,
    opens_in_another_app,
    section_title,
    stat_chip,
)

__all__ = ["OverviewTab", "hero_values", "primary_read_label", "progress_strip_text", "split_tags", "two_columns"]

NO_SYNOPSIS = "No synopsis available."
DASH = "—"
_METADATA_ROWS = (("title", "\U0001f4d8 Title"), ("author", "✍️ Author"), ("publisher", "\U0001f3db️ Publisher"),
                  ("language", "\U0001f310 Language"), ("year", "\U0001f4c5 Year"))


def _first(*values: Any) -> str:
    for value in values:
        if isinstance(value, (list, tuple)):
            value = ", ".join(str(v) for v in value if v)
        if value:
            return str(value).strip()
    return ""


def split_tags(value: Any) -> list:
    """Tag values split on ``#`` / ``,`` / ``;`` / newlines (desktop ``_metadata_subject_values`` shape)."""
    items = value if isinstance(value, (list, tuple)) else [value]
    out: list[str] = []
    for item in items:
        for part in re.split(r"[#,;\n]", str(item or "")):
            part = part.strip()
            if part and part not in out:
                out.append(part)
    return out


def hero_values(book: Mapping[str, Any], details: Optional[Mapping[str, Any]], model: Any = None) -> dict:
    """Title / author / publisher / language / year / synopsis / tags for the hero and the metadata list.

    ``model`` is the shared ``library_core.BookDetailsModel`` over the details payload
    (title, authors, tags and synopsis follow its desktop rules: metadata.json first,
    then the OPF, then the card); without one the same fields are read plainly.
    """
    details = dict(details or {})
    metadata = details.get("metadata_json") or book.get("metadata_json") or {}
    if not isinstance(metadata, Mapping):
        metadata = {}
    opf = details.get("details") or {}
    if not isinstance(opf, Mapping):
        opf = {}
    if model is not None:
        title = str(model.title() or "")
        author = ", ".join(str(a) for a in (model.authors() or []) if a)
        tags = [str(t) for t in (model.tags() or []) if t]
        synopsis = str(model.synopsis() or "")
    else:
        title = _first(metadata.get("title"), opf.get("title"), book.get("name"))
        author = _first(metadata.get("creator"), metadata.get("authors"), opf.get("authors"))
        tags = []
        for key in ("subject", "subjects", "genres", "tags"):
            if metadata.get(key):
                tags = split_tags(metadata.get(key))
                break
        if not tags:
            tags = split_tags(opf.get("subjects") or book.get("subjects") or [])
        synopsis = _first(metadata.get("description"), opf.get("description"))
    synopsis = re.sub(r"\n\s*\n+", "\n\n", synopsis).strip()
    date = _first(metadata.get("date"), opf.get("date"))
    year_match = re.search(r"\d{4}", date)
    return {
        "title": title,
        "author": author,
        "publisher": _first(metadata.get("publisher"), opf.get("publisher")),
        "language": _first(metadata.get("language"), opf.get("language")),
        "year": year_match.group(0) if year_match else "",
        "date": date,
        "synopsis": synopsis,
        "tags": tags,
    }


def primary_read_label(book: Mapping[str, Any], chapters_info: Any) -> tuple:
    """``(primary label, show the tonal "📜 Read raw source")`` (BookDetailsDialog 15770-15805)."""
    has_translated = any((c or {}).get("translated_path") for c in (chapters_info or ()))
    raw = str(book.get("raw_source_path") or "")
    is_pdf_workspace = (raw.lower().endswith(".pdf") or str(book.get("workspace_kind") or "").lower() == "pdf"
                        or str(book.get("compiled_output_kind") or "").lower() == "pdf")
    if book.get("is_in_progress") or is_pdf_workspace:
        if has_translated:
            return "\U0001f4d6  Read translated", True
        return "\U0001f4d6  Read raw source", False
    return "\U0001f4d6  Start reading", False


def progress_strip_text(book: Mapping[str, Any], model: Any = None) -> Optional[str]:
    """The in-progress strip (None: hidden): ``BookDetailsModel.progress_strip_text`` (the desktop
    ``_update_progress_strip`` over the loaded chapters); without a model, the card's scanned counts."""
    if not book.get("is_in_progress") or book.get("translation_state") == "completed":
        return None
    if model is not None:
        try:
            text = model.progress_strip_text()
        except Exception:
            text = None
        return str(text) if text else None
    total = int(book.get("total_chapters", 0) or 0)
    done = int(book.get("completed_chapters", 0) or 0)
    if total and done >= total:
        return None
    if total:
        return f"⏳  Translation in progress — {done}/{total} chapters ({(done * 100) // total}%)"
    return "⏳  Translation in progress"


def two_columns(size_class: Any, width: Optional[float], height: Optional[float]) -> bool:
    """UI_SPEC §1.1 Large phone: "Book page Overview uses two columns in landscape"."""
    name = str(getattr(size_class, "value", size_class) or "")
    try:
        landscape = float(width or 0) > float(height or 0) > 0
    except (TypeError, ValueError):
        landscape = False
    return name == "large_phone" and landscape


class OverviewTab:
    def __init__(self, page: Any) -> None:
        self.page = page
        self.ctx = page.ctx
        self.synopsis_expanded = False
        self.values: dict = {}
        self.columns = False  # two-column landscape arrangement (``apply_layout``)
        self.layout_builds = 0

    # ---- build --------------------------------------------------------------------------------

    def build(self) -> ft.Control:
        tablet = self.ctx.tablet
        cover_w, cover_h = (240, 340) if tablet else (120, 180)
        self.cover = ft.Image(src=HALGAKOS_ASSET, width=cover_w, height=cover_h, fit=ft.BoxFit.CONTAIN,
                              border_radius=tokens.RADII["cover"], key="ov-cover")
        self.title_text = ft.Text("", theme_style=ft.TextThemeStyle.TITLE_LARGE, weight=ft.FontWeight.W_600,
                                  selectable=True, key="ov-title")
        self.author_text = ft.Text("", theme_style=ft.TextThemeStyle.BODY_MEDIUM,
                                   color=ft.Colors.ON_SURFACE_VARIANT, key="ov-author")
        self.hero_chips = ft.Row(wrap=True, spacing=6, run_spacing=4, key="ov-chips")
        self.strip = ft.Container(
            content=ft.Text("", color="#ffd166", weight=ft.FontWeight.W_600, key="ov-strip-text"),
            bgcolor=ft.Colors.with_opacity(0.14, "#6c63ff"), border=ft.Border.all(1, "#6c63ff"),
            border_radius=tokens.RADII["chip"], padding=ft.Padding.symmetric(horizontal=10, vertical=6),
            visible=False, key="ov-strip")
        self.read_button = ft.FilledButton(content="\U0001f4d6  Start reading", on_click=self._on_read, key="ov-read")
        self.raw_button = ft.FilledTonalButton(content="\U0001f4dc  Read raw source", on_click=self._on_read_raw,
                                               visible=False, key="ov-read-raw")
        self.continue_button = ft.OutlinedButton(content="", icon=ft.Icons.PLAY_ARROW, on_click=self._on_continue,
                                                 visible=False, key="ov-continue")
        self.icon_row = ft.Row(wrap=True, spacing=0, key="ov-icons")
        self.synopsis_text = ft.Text(NO_SYNOPSIS, max_lines=4, overflow=ft.TextOverflow.ELLIPSIS, selectable=True,
                                     key="ov-synopsis")
        self.synopsis_toggle = ft.TextButton(content="More", on_click=self._toggle_synopsis, visible=False,
                                             key="ov-synopsis-more")
        self.metadata_rows = ft.Column(spacing=2, key="ov-metadata")
        self.edit_button = ft.TextButton(content="✏️  Edit", on_click=self._on_edit, key="ov-edit")
        self.tags_row = ft.Row(wrap=True, spacing=6, run_spacing=6, key="ov-tags")
        self.glance = ft.Column(spacing=6, key="ov-glance")
        self.error_text = ft.Text("", color=ft.Colors.ERROR, visible=False, key="ov-error")
        self.hero = ft.Row([
            self.cover,
            ft.Column([self.title_text, self.author_text, self.hero_chips], spacing=4, expand=True, tight=True),
        ], vertical_alignment=ft.CrossAxisAlignment.START, spacing=12, key="ov-hero")
        self.skeleton = Skeleton("hero", count=2, label="Loading book…", key="ov-skeleton").control
        self.left_items = [
            self.skeleton,
            self.hero,
            self.strip,
            ft.Row([self.read_button, self.raw_button, self.continue_button], wrap=True, spacing=8, run_spacing=6),
            self.icon_row,
            self.error_text,
        ]
        self.right_items = [
            section_title("SYNOPSIS"),
            self.synopsis_text,
            self.synopsis_toggle,
            ft.Row([section_title("METADATA"), self.edit_button], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
            self.metadata_rows,
            section_title("TAGS"),
            self.tags_row,
            section_title("At a glance"),
            self.glance,
        ]
        self.list = ft.ListView(self._arranged(), spacing=tokens.SPACING["sm"], padding=12, expand=True,
                                key="overview")
        self.render()
        return self.list

    # ---- layout ----------------------------------------------------------------------------------

    def _arranged(self) -> list:
        if not self.columns:
            return [*self.left_items, *self.right_items]
        self.layout_builds += 1
        n = self.layout_builds
        spacing = tokens.SPACING["sm"]
        return [ft.Row([
            ft.Column(list(self.left_items), spacing=spacing, expand=1, tight=True, key=f"ov-left-{n}"),
            ft.Column(list(self.right_items), spacing=spacing, expand=1, tight=True, key=f"ov-right-{n}"),
        ], spacing=tokens.SPACING["md"], vertical_alignment=ft.CrossAxisAlignment.START, key=f"ov-columns-{n}")]

    def apply_layout(self, size_class: Any) -> bool:
        """Two columns on a large phone in landscape, one column otherwise; True when it changed."""
        page = self.ctx.page
        wanted = two_columns(size_class, getattr(page, "width", None), getattr(page, "height", None))
        if wanted == self.columns:
            return False
        self.columns = wanted
        if getattr(self, "list", None) is not None:
            self.list.controls = self._arranged()
            self.ctx.push(self.list)
        return True

    # ---- render ----------------------------------------------------------------------------------

    def render(self) -> None:
        if getattr(self, "list", None) is None:
            return
        page = self.page
        service = page.service
        book = page.book
        details = page.details or {}
        model = page.details_model()
        values = hero_values(book, details, model)
        self.values = values
        loading = page.details is None  # the preview-phase details have not arrived yet
        self.skeleton.visible = loading
        self.hero.visible = not loading
        self.title_text.value = values["title"]
        self.author_text.value = values["author"] or ""
        chips = []
        if values["language"]:
            chips.append(ft.Chip(label=ft.Text(f"\U0001f310 {values['language']}"), key="chip-lang"))
        if values["year"]:
            chips.append(ft.Chip(label=ft.Text(f"\U0001f4c5 {values['year']}"), key="chip-year"))
        kind = str(book.get("workspace_kind") or book.get("type") or "").upper()
        if kind and kind != "IN_PROGRESS":
            chips.append(ft.Chip(label=ft.Text(kind), key="chip-type"))
        self.hero_chips.controls = chips
        cover = service.covers.get(_key(book))
        if cover:
            self.cover.src = cover
            self.cover.fit = ft.BoxFit.COVER
        elif details.get("cover") and isinstance(details.get("cover"), str):
            self.cover.src = details["cover"]
            self.cover.fit = ft.BoxFit.COVER
        strip = progress_strip_text(book, model if details.get("chapters_info") is not None else None)
        self.strip.visible = bool(strip)
        self.strip.content.value = strip or ""
        label, show_raw = primary_read_label(book, details.get("chapters_info"))
        self.read_button.content = label
        elsewhere = opens_in_another_app(book)  # the card menus' rule: no Reader for it, ↗ Share instead
        self.read_button.disabled = elsewhere
        self.read_button.tooltip = OPENS_ELSEWHERE_REASON if elsewhere else None
        self.raw_button.visible = show_raw
        self._render_continue()
        self._render_icons()
        synopsis = values["synopsis"]
        self.synopsis_text.value = synopsis or NO_SYNOPSIS
        self.synopsis_text.max_lines = None if self.synopsis_expanded else 4
        self.synopsis_toggle.visible = bool(synopsis) and (len(synopsis) > 280 or synopsis.count("\n") > 3)
        self.synopsis_toggle.content = "Less" if self.synopsis_expanded else "More"
        self.metadata_rows.controls = [
            ft.Row([ft.Text(label_text, width=120, color=ft.Colors.ON_SURFACE_VARIANT),
                    ft.Text(values.get(key) or DASH, expand=True, selectable=True)], key=f"meta-{key}")
            for key, label_text in _METADATA_ROWS
        ]
        has_workspace = bool(pm.book_workspace(service, book))
        self.edit_button.disabled = not has_workspace
        self.edit_button.tooltip = None if has_workspace else "No output workspace yet"
        self.tags_row.controls = [ft.Chip(label=ft.Text(tag), key=f"tag-{i}") for i, tag in enumerate(values["tags"])]
        if not values["tags"]:
            self.tags_row.controls = [ft.Text(DASH, color=ft.Colors.ON_SURFACE_VARIANT)]
        self._render_glance()
        error = details.get("error")
        self.error_text.visible = bool(error)
        self.error_text.value = str(error or "")
        self.ctx.push(self.list)

    def _render_continue(self) -> None:
        prefs = self.ctx.prefs
        position = None
        if prefs is not None and hasattr(prefs, "reader_position"):
            try:
                position = prefs.reader_position(self.page.bid)
            except Exception:
                position = None
        if not position:
            self.continue_button.visible = False
            return
        chapters = (self.page.details or {}).get("chapters_info") or []
        href = str(position.get("href") or "")
        number = None
        for index, chapter in enumerate(chapters):
            if href and str((chapter or {}).get("filename") or "").split("/")[-1] == href.split("/")[-1].split("#")[0]:
                number = index + 1
                break
        try:
            pct = int(float(position.get("fraction") or 0) * 100)
        except (TypeError, ValueError):
            pct = 0
        parts = ["Continue"]
        if number is not None:
            parts.append(f"Ch {number}")
        parts.append(f"{pct}%")
        self.continue_button.content = " · ".join(parts)
        self.continue_button.visible = True

    def _render_icons(self) -> None:
        page = self.page
        book = page.book
        service = page.service
        raw = str(book.get("raw_source_path") or "")
        has_raw = bool(raw) and not book.get("missing_raw_file")
        has_workspace = bool(pm.book_workspace(service, book))
        metadata_reason = page.metadata_reason()  # shared with Book ⋯

        def icon(name: str, tip: str, handler: Any, reason: Optional[str], key: str) -> ft.Control:
            button = icon_button(name, reason or tip, handler, key=key, disabled=reason is not None)
            button.opacity = 0.35 if reason else 1.0
            return button

        self.icon_row.controls = [
            icon("TRANSLATE", "\U0001f310 Translate…", lambda e: self.ctx.spawn(page.open_translate()),
                 None if has_raw else "The raw source file can't be found", "ov-translate"),
            icon("MENU_BOOK", "\U0001f4d8 Compile", lambda e: page.compile_menu(),
                 None if has_workspace else "No output workspace yet", "ov-compile"),
            icon("LABEL", "\U0001f3f7 Translate Metadata", lambda e: self.ctx.spawn(page.translate_metadata()),
                 metadata_reason, "ov-metadata-job"),
            icon("IOS_SHARE", "↗ Share", lambda e: self.ctx.spawn(page.output.share_primary()), None, "ov-share"),
            icon("FOLDER_OPEN", "\U0001f4c1 Files", lambda e: page.open_files(),
                 None if has_workspace else "No output folder", "ov-files"),
        ]

    def _render_glance(self) -> None:
        page = self.page
        controls: list[ft.Control] = []
        progress = page.progress
        if progress is not None and progress.chips:
            row = [stat_chip(chip.text, chip.status, dark=self.ctx.dark, key=f"glance-{chip.group}",
                             on_select=lambda e, g=chip.group: page.set_tab("chapters", status_filter=g))
                   for chip in progress.chips if chip.visible]
            controls.append(ft.Row(row, wrap=True, spacing=6, run_spacing=6))
            if progress.total_text:
                controls.append(ft.Text(progress.total_text, theme_style=ft.TextThemeStyle.LABEL_SMALL))
        elif progress is not None and progress.no_workspace:
            # no workspace: the EPUB's own chapters (the Chapters tab lists them); nothing while they load
            count = self._epub_chapter_count()
            if count:
                controls.append(ft.TextButton(content=f"\U0001f4d6 {count} chapter{'s' if count != 1 else ''}",
                                              on_click=lambda e: page.set_tab("chapters"), key="glance-chapters"))
            elif count == 0:
                controls.append(ft.Text(progress.error, theme_style=ft.TextThemeStyle.BODY_SMALL,
                                        color=ft.Colors.ON_SURFACE_VARIANT))
        elif progress is not None and progress.error:
            controls.append(ft.Text(progress.error, theme_style=ft.TextThemeStyle.BODY_SMALL,
                                    color=ft.Colors.ON_SURFACE_VARIANT))
        glossary = page.glossary
        if glossary is not None and glossary.path and glossary.chips:
            completed = next((c.count for c in glossary.chips if c.group == "completed"), 0)
            total = glossary.total_text.replace("Total: ", "") if glossary.total_text else ""
            text = f"Glossary {completed}/{total}" if total else f"Glossary {completed} completed"
            controls.append(ft.TextButton(content=text, icon=ft.Icons.SPELLCHECK,
                                          on_click=lambda e: page.set_tab("glossary"), key="glance-glossary"))
        else:
            controls.append(ft.TextButton(content="Glossary progress", icon=ft.Icons.SPELLCHECK,
                                          on_click=lambda e: page.set_tab("glossary"), key="glance-glossary"))
        last = self._last_job()
        if last is not None:
            controls.append(ft.TextButton(content=f"Last job: {last.title} · {last.state_label}",
                                          icon=ft.Icons.WORK_HISTORY,
                                          on_click=lambda e, jid=last.id: self.ctx.go("jobs.detail", {"jid": jid}),
                                          key="glance-job"))
        self.glance.controls = controls

    def _epub_chapter_count(self) -> Optional[int]:
        """The book's EPUB chapters (``BookDetailsModel.counts`` total, the desktop "Chapters (N)"); None while
        the chapter titles are still loading."""
        details = self.page.details or {}
        if details.get("chapters_info") is None and not details.get("error"):
            return None
        model = self.page.details_model()
        if model is None:
            return 0
        try:
            return int(model.counts()[1])
        except Exception:
            return 0

    def _last_job(self) -> Any:
        view = self.page.job_view
        if view is None:
            return None
        for snap in list(getattr(view, "history", ()) or ()):
            origin = getattr(getattr(snap, "spec", None), "origin", {}) or {}
            if origin.get("bid") == self.page.bid:
                return snap
        return None

    # ---- events ------------------------------------------------------------------------------------

    def _on_read(self, e: Any = None) -> None:
        label = str(self.read_button.content or "")
        self.page.open_reader(mode="original" if "raw" in label.lower() else "translated")

    def _on_read_raw(self, e: Any = None) -> None:
        self.page.open_reader(mode="original", raw_only=True)

    def _on_continue(self, e: Any = None) -> None:
        self.page.open_reader(resume=True)  # "Continue · Ch N · P%" opens at that position

    def _on_edit(self, e: Any = None) -> None:
        self.ctx.go("library.book.metadata", {"bid": self.page.bid})

    def _toggle_synopsis(self, e: Any = None) -> None:
        self.synopsis_expanded = not self.synopsis_expanded
        self.render()


def _key(book: Mapping[str, Any]) -> str:
    from glossarion_mobile.services.library import book_key

    return book_key(book)
