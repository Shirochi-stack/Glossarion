"""Book page › Chapters: Progress Manager parity (UI_SPEC §3.7; Retranslation_GUI single-file view).

Header: "📁 <output folder>" chip (Files) + "Mode: Text" badge. Stats row: the
``progress_core`` stats as chips "✅ Completed n" · "🔗 Merged n" · "🔄 In Progress n" ·
"❓ Pending n" · "⬜ Not Translated n" (Not Refined / No TTS by mode) · "❌ Failed n"
(Refine Failed) · "⏭️ Skipped n" - tap filters by ``STATUS_GROUPS`` (toggle),
long-press jumps to the next matching row and wraps (``GestureDetector``, desktop
click behaviour). "Total: N (N-k chapters + k chunks)".

Toolbar: "🔍 Search chapters…" (titles, file names, chunk text, QA issues), a
filter menu (Show special files · Show model info · Show raw titles, persisted as
``epub_details_show_special_files`` / ``retranslation_show_model_info`` /
``epub_details_show_raw_titles``; Rows per page = ``epub_details_chapter_page_size``;
QA failures only; Chunked chapters only) and ⋯ (Manual editing · 🔍 Edit
Translation · 📊 Glossary Progress · ⟳ Refresh · Files).

Rows: StatusAvatar · line 1 · line 2 (output file or model, chunk summary) ·
badges · expandable QA line · expandable chunk children "↳ Chunk i/T". The list
renders a Python-side window (``build_controls_on_demand=False`` so every built
row is a ``ScrollKey`` target, never ``first_item_prototype``) that grows by the
page size while scrolling; a jump first extends the window to the target.

Selection (long-press): "N selected · Select all · Select ▾ (Completed / QA
Failed / Failed group)" and the bottom bar Retranslate (Reset TTS in audio mode) ·
Remove QA mark · More ▾. Every write is a ``progress_actions`` call (lock,
re-read, three-way merge, atomic replace); confirmations use the desktop copy.
Retranslate needs ``plan_retranslation`` (U7) and says so until it exists.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Mapping, Optional, Sequence

import flet as ft

from glossarion_mobile.services.library import CoreMissing
from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.components.action_sheet import ActionItem, ActionSheet
from glossarion_mobile.ui.components.dialogs import ConfirmDialog
from glossarion_mobile.ui.components.empty_state import EmptyState
from glossarion_mobile.ui.components.info_sheet import InfoSheet
from glossarion_mobile.ui.library import progress_model as pm
from glossarion_mobile.ui.library.common import icon_button, mode_badge, stat_chip, status_avatar, tinted
from glossarion_mobile.ui.library.models import PAGE_SIZES, page_size_value
from glossarion_mobile.ui.library.selection_bar import BulkAction, BulkActionBar, SelectionTopBar
from glossarion_mobile.ui.theme import HIT_TARGET, status_color

__all__ = ["ChaptersTab", "RETRANSLATE_U7", "confirm_copy", "row_palette_status"]

log = logging.getLogger("glossarion.library.ui")

RETRANSLATE_U7 = ("Retranslating selected chapters (deleting their outputs and resetting their progress, with the "
                  "Progress Manager's confirmation) arrives in U7. Until then, start a translation from the "
                  "Overview tab: chapters that are not translated are picked up automatically.")
EMPTY_TITLE = "No chapters found yet"
EMPTY_BODY = "Start a translation to see chapter progress."
EDIT_TRANSLATION_REASON = "The SDLXLIFF reviewer arrives in U7"
TEXT_EDITOR_REASON = "The text editor arrives in U6"
ENGINE_REASON = "Needs the translation engine job (arrives in U7)"
# Desktop Book Details "Translate chapter" (BookDetailsDialog._translate_single_chapter)
RETRANSLATE_TITLE = "Retranslate chapter"
ALREADY_RUNNING = "A translation is already running.\nPlease wait for it to finish (or stop it) first."
UNTRACKED_STATUSES = ("", "not_translated")  # rows with no progress entry to reset


def retranslate_now_question(title: str) -> str:
    return f"“{title}” is already translated.\n\nDelete its current translation and retranslate it now?"
_SCROLL_APPEND_PX = 800
# UI_SPEC §7.3 WindowedList: beyond WINDOW_ROWS rows the list mounts one window of at most
# WINDOW_ROWS rows (a selector moves between windows; jumps re-centre first) and, with
# "Rows per page: All", appends WINDOW_STEP rows per scroll step instead of all at once.
WINDOW_ROWS = 1500
WINDOW_STEP = 150


def row_palette_status(status: str) -> str:
    """Display status -> shared palette key (file_missing / unknown read as pending)."""
    return status if status in ("completed", "merged", "in_progress", "pending", "not_translated", "not_refined",
                                "no_tts", "refine_failed", "failed", "qa_failed", "skipped") else "pending"


def confirm_copy(action: str, count: int, extra: Optional[Mapping[str, Any]] = None) -> Optional[tuple]:
    """Desktop confirmation (title, body) for an action, or None when the desktop asks nothing."""
    if action == "remove_qa":
        return "Confirm Remove Failed Mark", f"Remove failed mark from {count} chapters?"
    if action == "remove_refinement":
        return "Confirm Remove Refinement Status", f"Remove refinement status from {count} chapter(s)?"
    if action == "reset_tts":
        return ("Confirm TTS Reset", f"This will delete only generated TTS audio for {count} selected chapter(s), "
                "mark them as No TTS, and leave translated HTML files untouched.\n\nContinue?")
    if action == "delete_audio":
        audio_path = (extra or {}).get("audio_path", "")
        return "Delete Audio File", f"Delete this generated audio file?\n\n{audio_path}"
    return None


class ChaptersTab:
    def __init__(self, page: Any, *, initial_filter: Optional[str] = None) -> None:
        self.page = page
        self.ctx = page.ctx
        cfg = page.service.cfg
        self.filter_group: Optional[str] = initial_filter
        self.query = ""
        self.show_special = bool(cfg("epub_details_show_special_files", cfg("translate_special_files", False)))
        self.show_model = bool(cfg("retranslation_show_model_info", True))
        self.raw_titles = bool(cfg("epub_details_show_raw_titles", False))
        self.page_size_raw = cfg("epub_details_chapter_page_size", 20)
        self.page_size = page_size_value(self.page_size_raw)
        self.qa_only = False
        self.chunked_only = False
        self.view: Optional[pm.ProgressView] = None
        self.rows: list = []  # all RowVMs (top level)
        self.visible: list = []  # filtered RowVMs
        self.rendered = 0  # absolute index (in ``visible``) of the next row to mount
        self.window_start = 0  # first row of the mounted window (WindowedList beyond WINDOW_ROWS)
        self.expanded: set = set()
        self.qa_open: set = set()
        self.selected: set = set()
        self.selecting = False
        self.controls: dict = {}  # key -> row control
        self.jump_cursor: dict = {}  # group -> last jumped visible index
        self.list_view: Optional[ft.ListView] = None
        self.last_result: Any = None
        self.titles: dict = {}  # source file name -> (raw title, translated title)

    # ---- build ----------------------------------------------------------------------------------

    def build(self) -> ft.Control:
        self.folder_chip = ft.Chip(label=ft.Text("\U0001f4c1 —"), on_click=lambda e: self.page.open_files(),
                                   key="ch-folder")
        self.mode = mode_badge(pm.mode_label("text"), key="ch-mode")
        self.stats_row = ft.Row(spacing=6, run_spacing=6, scroll=ft.ScrollMode.AUTO,
                                wrap=self.ctx.text_scale >= tokens.COMPACT_TEXT_SCALE, key="ch-stats")
        self.total_text = ft.Text("", theme_style=ft.TextThemeStyle.LABEL_SMALL, key="ch-total")
        self.search = ft.TextField(hint_text="\U0001f50d Search chapters…", dense=True, expand=True,
                                   on_change=self._on_search, key="ch-search")
        self.filter_menu = ft.PopupMenuButton(icon=ft.Icons.FILTER_LIST, tooltip="Filters", key="ch-filter-menu",
                                              items=self._filter_items())
        self.more_menu = ft.PopupMenuButton(icon=ft.Icons.MORE_VERT, tooltip="More", key="ch-more-menu", items=[
            ft.PopupMenuItem(content="Manual editing · with the SDLXLIFF reviewer (U7)", disabled=True),
            ft.PopupMenuItem(content="\U0001f50d Edit Translation · arrives in U7", disabled=True),
            ft.PopupMenuItem(content="\U0001f4ca Glossary Progress", on_click=lambda e: self.page.set_tab("glossary")),
            ft.PopupMenuItem(content="⟳ Refresh", on_click=lambda e: self.ctx.spawn(self.page.full_refresh())),
            ft.PopupMenuItem(content="Files", on_click=lambda e: self.page.open_files()),
        ])
        self.selection_bar = SelectionTopBar(on_close=self.exit_selection, on_select_all=self.select_all,
                                             select_menu=[("Completed", lambda: self.select_group("completed")),
                                                          ("QA Failed", lambda: self.select_status("qa_failed")),
                                                          ("Failed group", lambda: self.select_group("failed"))],
                                             key="ch-selection")
        self.banner = ft.Container(visible=False, bgcolor=ft.Colors.ERROR_CONTAINER, padding=8,
                                   border_radius=tokens.RADII["card"], key="ch-banner")
        self.loading = ft.ProgressBar(visible=True, key="ch-loading")
        self.list_holder = ft.Container(expand=True, key="ch-list-holder")
        self.bulk_bar = BulkActionBar(page=self.ctx.page, tablet=self.ctx.tablet,
                                      compact=self.ctx.text_scale >= tokens.COMPACT_TEXT_SCALE, key="ch-bulk")
        header = ft.Row([self.folder_chip, self.mode], spacing=8, wrap=True)
        self.root = ft.Column([
            ft.Container(content=ft.Column([header, self.stats_row, self.total_text,
                                            ft.Row([self.search, self.filter_menu, self.more_menu], spacing=0),
                                            self.selection_bar.control, self.banner], spacing=6, tight=True),
                         padding=ft.Padding.only(left=12, right=12, top=8)),
            self.loading,
            self.list_holder,
            self.bulk_bar.control,
        ], spacing=4, expand=True, key="chapters")
        if self.page.progress is not None:
            self.apply(self.page.progress)
        return self.root

    def dispose(self) -> None:
        self.list_view = None

    def _filter_items(self) -> list:
        def check(label: str, value: bool, handler: Any, key: str) -> ft.PopupMenuItem:
            return ft.PopupMenuItem(content=label, checked=value, on_click=lambda e: handler(), key=key)

        return [
            check("Show special files", self.show_special, self.toggle_special, "f-special"),
            check("Show model info", self.show_model, self.toggle_model, "f-model"),
            check("Show raw titles", self.raw_titles, self.toggle_raw_titles, "f-raw"),
            ft.PopupMenuItem(content=f"Rows per page: {self._page_size_label()}",
                             on_click=lambda e: self.open_page_size_sheet(), key="f-page"),
            check("QA failures only", self.qa_only, self.toggle_qa_only, "f-qa"),
            check("Chunked chapters only", self.chunked_only, self.toggle_chunked_only, "f-chunked"),
        ]

    def _page_size_label(self) -> str:
        return "All" if self.page_size <= 0 else str(self.page_size)

    def _refresh_filter_menu(self) -> None:
        if getattr(self, "filter_menu", None) is not None:
            self.filter_menu.items = self._filter_items()
            self.ctx.push(self.filter_menu)

    # ---- data ------------------------------------------------------------------------------------

    def apply(self, view: pm.ProgressView) -> None:
        """New progress model: header, chips, rows (in place when the visible keys are unchanged)."""
        self.view = view
        if getattr(self, "root", None) is None:
            return
        self.loading.visible = False
        folder = view.output_dir or str(self.page.book.get("output_folder") or "")
        self.folder_chip.label = ft.Text(f"\U0001f4c1 {os.path.basename(folder) if folder else '—'}")
        self.mode.content.value = pm.mode_label(view.mode)
        self._render_chips()
        self.total_text.value = view.total_text
        if view.error:
            self.banner.content = ft.Text(view.error if not view.unreadable else
                                          "Progress file could not be read — showing last snapshot",
                                          color=ft.Colors.ON_ERROR_CONTAINER)
            self.banner.visible = True
        else:
            self.banner.visible = False
        old_keys = [r.key for r in self.visible]
        old_rows = {r.key: r for r in self.rows}
        self.rows = list(view.rows)
        new_visible = self._filtered()
        if [r.key for r in new_visible] == old_keys and self.list_view is not None:
            self.visible = new_visible
            for row in new_visible:
                if old_rows.get(row.key) != row and row.key in self.controls:
                    self._replace_row(row)
        else:
            self._render_list(new_visible, window_start=self.window_start)
        self.ctx.push(self.root)

    def _filtered(self) -> list:
        groups = self.view.groups if self.view is not None else {}
        out = []
        for row in self.rows:
            if row.hidden_unless_special and not self.show_special:
                continue
            if row.skipped_special and not self.show_special:
                continue
            if self.filter_group and row.status not in groups.get(self.filter_group, (self.filter_group,)):
                if not any(child.status in groups.get(self.filter_group, ()) for child in row.children):
                    continue
            if self.qa_only and not (row.status == "qa_failed" or any(c.status in ("qa_failed", "failed")
                                                                       for c in row.children)):
                continue
            if self.chunked_only and not row.children:
                continue
            if self.query and not (pm.row_matches(row, self.query) or any(pm.row_matches(c, self.query)
                                                                          for c in row.children)):
                continue
            out.append(row)
        return out

    def _render_chips(self) -> None:
        view = self.view
        chips = []
        for chip in (view.chips if view is not None else ()):
            chips.append(stat_chip(
                chip.text, chip.status, selected=self.filter_group == chip.group, dark=self.ctx.dark,
                visible=chip.visible, key=f"ch-chip-{chip.group}",
                on_select=lambda e, g=chip.group: self.set_filter(None if self.filter_group == g else g),
                on_long_press=lambda e, g=chip.group: self.ctx.spawn(self.jump_next(g))))
        self.stats_row.controls = chips

    def set_filter(self, group: Optional[str]) -> None:
        self.filter_group = group
        if getattr(self, "root", None) is None:
            return
        self._render_chips()
        self._render_list(self._filtered())
        self.ctx.push(self.root)

    # ---- list rendering ---------------------------------------------------------------------------

    def _render_list(self, rows: Sequence[pm.RowVM], *, window_start: int = 0) -> None:
        self.visible = list(rows)
        self.controls = {}
        last = ((len(self.visible) - 1) // WINDOW_ROWS) * WINDOW_ROWS if self.visible else 0
        self.window_start = max(0, min((max(0, int(window_start)) // WINDOW_ROWS) * WINDOW_ROWS, last))
        self.rendered = self.window_start
        if not self.visible:
            self.list_view = None
            if self.view is not None and not self.rows and not self.view.error:
                self.list_holder.content = EmptyState(
                    icon="LIST_ALT", title=EMPTY_TITLE, body=EMPTY_BODY, key="ch-empty",
                    primary=("Translate…", lambda e: self.ctx.spawn(self.page.open_translate())))
            elif self.view is not None and self.rows:
                self.list_holder.content = EmptyState(icon="FILTER_LIST_OFF", title="No chapters match",
                                                      body="Change the search or the filters.", key="ch-empty")
            else:
                self.list_holder.content = None
            return
        self.list_view = ft.ListView(spacing=4, padding=ft.Padding.only(left=8, right=8, bottom=96), expand=True,
                                     build_controls_on_demand=False, on_scroll=self._on_scroll, scroll_interval=120,
                                     key="ch-list")
        if self.windowed:
            self.list_holder.content = ft.Column([self._window_selector(), self.list_view], spacing=0, expand=True)
        else:
            self.list_holder.content = self.list_view
        self._extend_to(self.window_start + self._increment() - 1)

    @property
    def windowed(self) -> bool:
        return len(self.visible) > WINDOW_ROWS

    def _window_end(self) -> int:
        return min(len(self.visible), self.window_start + WINDOW_ROWS)

    def _window_selector(self) -> ft.Control:
        """Beyond WINDOW_ROWS rows: "Rows 1,501–3,000 of 4,210" with previous / next window buttons."""
        total, start, end = len(self.visible), self.window_start, self._window_end()
        self.window_text = ft.Text(f"Rows {start + 1:,}–{end:,} of {total:,}",
                                   theme_style=ft.TextThemeStyle.LABEL_MEDIUM, key="ch-window-text")
        return ft.Row([
            ft.IconButton(icon=ft.Icons.CHEVRON_LEFT, tooltip="Previous rows", disabled=start <= 0,
                          on_click=lambda e: self.set_window(start - WINDOW_ROWS), size_constraints=HIT_TARGET,
                          key="ch-window-prev"),
            self.window_text,
            ft.IconButton(icon=ft.Icons.CHEVRON_RIGHT, tooltip="Next rows", disabled=end >= total,
                          on_click=lambda e: self.set_window(start + WINDOW_ROWS), size_constraints=HIT_TARGET,
                          key="ch-window-next"),
        ], alignment=ft.MainAxisAlignment.CENTER, spacing=4, key="ch-window")

    def set_window(self, start: int) -> None:
        """Mount another window of rows (WindowedList page selector)."""
        self._render_list(self.visible, window_start=start)
        self.ctx.push(self.list_holder)

    def _increment(self) -> int:
        if self.page_size > 0:
            return self.page_size
        return WINDOW_STEP if self.windowed else len(self.visible)

    def _extend_to(self, index: int) -> int:
        if self.list_view is None:
            return 0
        end = min(self._window_end(), max(index + 1, self.rendered))
        added = 0
        for row in self.visible[self.rendered:end]:
            control = self._row_control(row)
            self.controls[row.key] = control
            self.list_view.controls.append(control)
            added += 1
        self.rendered = end
        return added

    def _on_scroll(self, e: Any) -> None:
        event_type = str(getattr(getattr(e, "event_type", None), "value", getattr(e, "event_type", "")))
        if event_type == "overscroll":
            if (getattr(e, "overscroll", 0) or 0) < 0 and (getattr(e, "pixels", 0) or 0) <= 1:
                self.ctx.spawn(self.page.full_refresh())
            return
        pixels, maximum = getattr(e, "pixels", None), getattr(e, "max_scroll_extent", None)
        if pixels is None or maximum is None:
            return
        if maximum - pixels < _SCROLL_APPEND_PX and self.rendered < self._window_end():
            if self._extend_to(self.rendered + self._increment() - 1):
                self.ctx.push(self.list_view)

    def _replace_row(self, row: pm.RowVM) -> None:
        old = self.controls.get(row.key)
        if old is None or self.list_view is None:
            return
        new = self._row_control(row)
        try:
            index = self.list_view.controls.index(old)
        except ValueError:
            return
        self.list_view.controls[index] = new
        self.controls[row.key] = new

    def set_titles(self, chapters_info: Optional[Sequence[Mapping[str, Any]]]) -> None:
        """Raw / translated chapter titles from ``load_book_details`` (by source file name)."""
        titles: dict = {}
        for info in chapters_info or ():
            name = os.path.basename(str((info or {}).get("filename") or ""))
            if name:
                titles[name] = (str(info.get("raw_title") or ""), str(info.get("translated_title") or ""))
        if titles == self.titles:
            return
        self.titles = titles
        if self.list_view is not None and self.visible:
            for row in self.visible:
                if row.kind == "chapter" and row.key in self.controls:
                    self._replace_row(row)
            self.ctx.push(self.list_view)

    def display_title(self, row: pm.RowVM) -> str:
        """UI_SPEC §3.7 line 1: a completed chapter shows its translated title ("Show raw titles" shows
        the raw one); others keep the Progress Manager title ("Ch.012 · chapter0012.xhtml")."""
        if row.kind != "chapter":
            return row.title
        raw_title, translated = self.titles.get(os.path.basename(row.filename), ("", ""))
        prefix = row.title.split(" \u00b7 ")[0]
        if self.raw_titles and raw_title:
            return f"{prefix} \u00b7 {raw_title}"
        if not self.raw_titles and translated and row.status in ("completed", "merged"):
            return f"{prefix} \u00b7 {translated}"
        return row.title

    def _row_control(self, row: pm.RowVM) -> ft.Control:
        dark = self.ctx.dark
        selected = row.key in self.selected
        title = self.display_title(row)
        line2 = row.subtitle
        if row.chunk_text:
            chunk_text = row.chunk_text.strip(" ·")
            line2 = f"{line2} · {chunk_text}" if line2 else chunk_text
        lines: list[ft.Control] = [
            ft.Text(title, theme_style=ft.TextThemeStyle.BODY_MEDIUM, max_lines=2, overflow=ft.TextOverflow.ELLIPSIS,
                    key="line1"),
        ]
        if line2:
            lines.append(ft.Text(line2, theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT,
                                 max_lines=2, overflow=ft.TextOverflow.ELLIPSIS, key="line2"))
        status_line = ft.Text(f"{row.icon} {row.label}", size=11, weight=ft.FontWeight.W_600,
                              color=status_color(row_palette_status(row.status), dark), key="status")
        badges = [status_line] + [ft.Text(b, size=11, key=f"badge-{i}") for i, b in enumerate(row.badges)]
        lines.append(ft.Row(badges, spacing=6, wrap=True, run_spacing=2, key="badges"))
        trailing: list[ft.Control] = []
        if row.children:
            open_ = row.key in self.expanded
            trailing.append(ft.IconButton(icon=ft.Icons.EXPAND_LESS if open_ else ft.Icons.EXPAND_MORE,
                                          tooltip="Hide chunks" if open_ else "Show chunks",
                                          size_constraints=HIT_TARGET, key="chevron",
                                          on_click=lambda e, k=row.key: self.toggle_expand(k)))
        trailing.append(ft.IconButton(icon=ft.Icons.MORE_VERT, tooltip="Chapter actions", size_constraints=HIT_TARGET,
                                      key="row-more", on_click=lambda e, r=row: self.open_row_sheet(r)))
        main = ft.Row([
            status_avatar(row_palette_status(row.status), emoji=row.icon, label=f"{row.title}, {row.label}",
                          dark=dark),
            ft.Column(lines, spacing=1, expand=True, tight=True),
            *trailing,
        ], spacing=8, vertical_alignment=ft.CrossAxisAlignment.START)
        parts: list[ft.Control] = [main]
        if row.qa_lines:
            open_qa = row.key in self.qa_open
            shown = row.qa_lines if open_qa else row.qa_lines[:1]
            parts.append(ft.Container(
                content=ft.Text("\n".join(shown), theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ERROR,
                                max_lines=None if open_qa else 2, overflow=ft.TextOverflow.ELLIPSIS, selectable=open_qa),
                on_click=lambda e, k=row.key: self.toggle_qa(k), padding=ft.Padding.only(left=40), key="qa"))
        if row.children and row.key in self.expanded:
            parts.append(ft.Column([self._child_control(row, child) for child in row.children], spacing=2,
                                   key="children"))
        return ft.Container(
            content=ft.Column(parts, spacing=2, tight=True),
            key=ft.ScrollKey(row.key),
            padding=ft.Padding.symmetric(horizontal=8, vertical=6),
            border_radius=tokens.RADII["card"],
            bgcolor=ft.Colors.SECONDARY_CONTAINER if selected else ft.Colors.SURFACE_CONTAINER_LOW,
            border=ft.Border.all(2, ft.Colors.PRIMARY) if selected else None,
            on_click=lambda e, r=row: self.on_row_tap(r),
            on_long_press=lambda e, r=row: self.on_row_long_press(r),
            ink=True,
        )

    def _child_control(self, parent: pm.RowVM, child: pm.RowVM) -> ft.Control:
        selected = child.key in self.selected
        text = child.title if child.title.startswith("↳") else f"↳ {child.title}"
        detail = " · ".join(p for p in (f"{child.icon} {child.label}", child.model or child.subtitle) if p)
        return ft.Container(
            content=ft.Column([
                ft.Text(text, theme_style=ft.TextThemeStyle.BODY_SMALL),
                ft.Text(detail, size=11, color=status_color(row_palette_status(child.status), self.ctx.dark)),
            ] + ([ft.Text("\n".join(child.qa_lines[:2]), size=11, color=ft.Colors.ERROR)] if child.qa_lines else []),
                spacing=0, tight=True),
            padding=ft.Padding.only(left=44, top=2, bottom=2, right=8),
            bgcolor=tinted(ft.Colors.PRIMARY, 0.12) if selected else None,
            border_radius=tokens.RADII["chip"],
            on_click=lambda e, c=child: self.on_row_tap(c, parent=parent),
            on_long_press=lambda e, c=child: self.on_row_long_press(c),
            key=f"child-{child.key}",
        )

    def toggle_expand(self, key: str) -> None:
        if key in self.expanded:
            self.expanded.discard(key)
        else:
            self.expanded.add(key)
        row = next((r for r in self.visible if r.key == key), None)
        if row is not None:
            self._replace_row(row)
            self.ctx.push(self.list_view)

    def toggle_qa(self, key: str) -> None:
        if key in self.qa_open:
            self.qa_open.discard(key)
        else:
            self.qa_open.add(key)
        row = next((r for r in self.visible if r.key == key), None)
        if row is not None:
            self._replace_row(row)
            self.ctx.push(self.list_view)

    async def jump_next(self, group: str) -> Optional[str]:
        """Long-press on a stats chip: the next visible row with that status group (wraps around)."""
        if self.view is None:
            return None
        members = self.view.groups.get(group, (group,))
        start = self.jump_cursor.get(group, -1) + 1
        order = list(range(start, len(self.visible))) + list(range(0, start))
        for index in order:
            row = self.visible[index]
            if row.status in members or any(c.status in members for c in row.children):
                self.jump_cursor[group] = index
                if not self.window_start <= index < self._window_end():
                    self.set_window(index)  # re-centre the window on the target first (WindowedList)
                if self._extend_to(index + max(5, self._increment() // 2)):
                    self.ctx.push(self.list_view)
                self.ctx.haptic("selection_click")
                if self.list_view is not None:
                    try:
                        await self.list_view.scroll_to(scroll_key=row.key, duration=250)
                    except Exception:
                        log.debug("scroll_to failed", exc_info=True)
                return row.key
        self.ctx.say("No chapter with that status")
        return None

    # ---- toolbar toggles (persisted) -------------------------------------------------------------

    def _on_search(self, e: Any = None) -> None:
        self.query = str(self.search.value or "").strip()
        self._render_list(self._filtered())
        self.ctx.push(self.list_holder)

    def toggle_special(self) -> None:
        self.show_special = not self.show_special
        self.page.service.set_cfg("epub_details_show_special_files", self.show_special)
        self._after_toggle(reload=True)

    def toggle_model(self) -> None:
        self.show_model = not self.show_model
        self.page.service.set_cfg("retranslation_show_model_info", self.show_model)
        self._after_toggle(reload=True)

    def toggle_raw_titles(self) -> None:
        self.raw_titles = not self.raw_titles
        self.page.service.set_cfg("epub_details_show_raw_titles", self.raw_titles)
        self._after_toggle(reload=False)

    def toggle_qa_only(self) -> None:
        self.qa_only = not self.qa_only
        self._after_toggle(reload=False)

    def toggle_chunked_only(self) -> None:
        self.chunked_only = not self.chunked_only
        self._after_toggle(reload=False)

    def set_page_size(self, value: Any) -> None:
        text = str(value).lower()
        self.page_size_raw = "all" if text == "all" else int(text)
        self.page_size = page_size_value(self.page_size_raw)
        self.page.service.set_cfg("epub_details_chapter_page_size", self.page_size_raw)
        self._after_toggle(reload=False)

    def open_page_size_sheet(self) -> ActionSheet:
        sheet = ActionSheet([ActionItem(label, (lambda v=value: self.set_page_size(v)), key=f"page-{value}")
                             for value, label in PAGE_SIZES], title="Rows per page", tablet=self.ctx.tablet)
        self.ctx.show(sheet)
        return sheet

    def _after_toggle(self, *, reload: bool) -> None:
        self._refresh_filter_menu()
        if reload:
            self.ctx.spawn(self.page.reload_progress())
        elif getattr(self, "root", None) is not None:
            self._render_list(self._filtered())
            self.ctx.push(self.list_holder)

    # ---- taps / selection -------------------------------------------------------------------------

    def on_row_tap(self, row: pm.RowVM, parent: Optional[pm.RowVM] = None) -> None:
        if self.selecting:
            self.toggle_select(row.key)
            return
        if row.kind in ("chapter", "chunk") and (row.filename or (parent and parent.filename)):
            target = parent or row
            chapter = target.opf_position
            mode = "translated" if target.status in ("completed", "merged") else "original"
            self.page.open_reader(chapter=chapter, chapter_filename=target.filename or None, mode=mode)
            return
        self.open_row_sheet(row)

    def on_row_long_press(self, row: pm.RowVM) -> None:
        if not self.selecting:
            self.ctx.haptic("medium_impact")
        self.selecting = True
        self.toggle_select(row.key, force=True)

    def _all_rows(self) -> list:
        out = []
        for row in self.visible:
            out.append(row)
            out.extend(row.children)
        return out

    def selected_rows(self) -> list:
        """Selected rows; a selected parent absorbs its own selected chunk rows (desktop normalisation)."""
        rows = [r for r in self._all_rows() if r.key in self.selected]
        parents = {r.key for r in rows if r.children}
        return [r for r in rows if not (r.kind == "chunk" and any(
            r in p.children for p in self.visible if p.key in parents))]

    def toggle_select(self, key: str, force: Optional[bool] = None) -> None:
        on = (key not in self.selected) if force is None else force
        if on:
            self.selected.add(key)
        else:
            self.selected.discard(key)
        if not self.selected:
            self.selecting = False
        self._rerender_key(key)
        self._sync_selection()

    def _rerender_key(self, key: str) -> None:
        for row in self.visible:
            if row.key == key or any(c.key == key for c in row.children):
                self._replace_row(row)
                break
        self.ctx.push(self.list_view)

    # The selection re-renders keep the current 1,500-row window (a long book keeps its place).

    def select_all(self) -> None:
        self.selecting = True
        self.selected = {r.key for r in self.visible}
        self._render_list(self.visible, window_start=self.window_start)
        self._sync_selection()

    def select_group(self, group: str) -> None:
        members = self.view.groups.get(group, (group,)) if self.view is not None else (group,)
        self.selecting = True
        self.selected = {r.key for r in self.visible if r.status in members}
        self._render_list(self.visible, window_start=self.window_start)
        self._sync_selection()

    def select_status(self, status: str) -> None:
        self.selecting = True
        self.selected = {r.key for r in self.visible if r.status == status}
        self._render_list(self.visible, window_start=self.window_start)
        self._sync_selection()

    def exit_selection(self) -> None:
        self.selecting = False
        self.selected = set()
        if self.list_view is not None:
            self._render_list(self.visible, window_start=self.window_start)
        self._sync_selection()

    def _sync_selection(self) -> None:
        if getattr(self, "root", None) is None:
            return
        count = len(self.selected)
        active = self.selecting and count > 0
        self.selection_bar.set_count(count)
        self.selection_bar.show(active)
        self.bulk_bar.show(active)
        if active:
            primary, more = self.bulk_actions(self.selected_rows())
            self.bulk_bar.set_actions(primary, more)
        self.ctx.push(self.selection_bar.control, self.bulk_bar.control, self.list_holder)

    # ---- actions -----------------------------------------------------------------------------------

    def _audio_mode(self) -> bool:
        return self.view is not None and str(self.view.mode).lower() == "audio"

    def _reason(self, action: str, rows: Sequence[pm.RowVM]) -> Optional[str]:
        """Why an action cannot run on this selection (None: try it; the shared plan re-checks)."""
        if not self.page.service.core.has_module("progress_actions"):
            return "Needs progress_actions (not in this build)"
        if self.view is None or self.view.state is None:
            return "The progress could not be read"
        if action == "reset_tts" and not self._audio_mode():
            return "Only in Audio output mode"
        if action in _SINGLE_ROW_ACTIONS and len(rows) != 1:
            return "Select one chapter"
        return None

    def bulk_actions(self, rows: Sequence[pm.RowVM]) -> tuple:
        audio = self._audio_mode()
        primary = [
            BulkAction("retranslate", "Reset TTS" if audio else "Retranslate", "REPLAY",
                       lambda: self.ctx.spawn(self.retranslate(rows)),
                       self._reason("reset_tts", rows) if audio else None),
            BulkAction("remove_qa", "Remove QA mark", "CLEANING_SERVICES",
                       lambda: self.ctx.spawn(self.run_action("remove_qa", rows)), self._reason("remove_qa", rows)),
        ]
        more = []
        for action in ("remove_pending", "remove_refinement", "restore_in_progress", "resolve_qa", "insert_image",
                       "do_not_skip", "delete_audio"):
            more.append(BulkAction(action, pm.ACTION_LABELS[action], "CHEVRON_RIGHT",
                                   (lambda a=action: self.ctx.spawn(self.run_action(a, rows))),
                                   self._reason(action, rows)))
        more.append(BulkAction("edit_translation", "\U0001f50d Edit Translation", "RATE_REVIEW",
                               disabled_reason=EDIT_TRANSLATION_REASON))
        return primary, more

    def open_row_sheet(self, row: pm.RowVM) -> Any:
        return self.ctx.spawn(self.show_row_sheet(row))

    async def show_row_sheet(self, row: pm.RowVM) -> ActionSheet:
        """Row ⋯ (the desktop context menu, ``progress_actions.row_actions``); skipped special rows
        offer only "Do not skip (remove keyword …)"."""
        service = self.page.service
        allowed = None
        keyword = None
        if self.view is not None and self.view.state is not None:
            allowed = await self.ctx.io(pm.row_action_ids, service, self.view, row, [row])
            if row.skipped_special or (allowed is not None and "do_not_skip" in allowed):
                plan = await self.ctx.io(pm.plan_action, service, self.view, "do_not_skip", [row])
                keyword = plan.extra.get("keyword")
        rows = [row]
        if keyword or (allowed is not None and allowed == {"do_not_skip"}):
            label = f"⏭️ Do not skip (remove keyword '{keyword}')" if keyword else "⏭️ Do not skip"
            sheet = ActionSheet([ActionItem(label, lambda: self.ctx.spawn(self.run_action("do_not_skip", rows)),
                                            key="row-do-not-skip")], title=row.title, tablet=self.ctx.tablet)
            self.ctx.show(sheet)
            return sheet

        def gate(action_id: str, reason: str) -> Optional[str]:
            if allowed is None:
                return self._reason(action_id, rows) if action_id in pm.ACTION_LABELS else None
            return None if action_id in allowed else reason

        completed = row.status in ("completed", "merged")
        single_ok = service.has_job_kind("single_chapter")
        items = [
            ActionItem("\U0001f4d6 Open in reader", lambda: self.on_row_tap(row), icon="AUTO_STORIES",
                       disabled_reason=gate("open_reader", "Not an HTML chapter") if allowed is not None else (
                           None if row.filename else "No source file for this row"), key="row-reader"),
            ActionItem("\U0001f4c2 Open file", lambda: self.ctx.spawn(self.share_output(row)), icon="FOLDER_OPEN",
                       disabled_reason=None if row.output_file else "No output file yet", key="row-open-file"),
            ActionItem("✏️ Edit file (find QA issue)" if row.qa_lines else "✏️ Edit file",
                       icon="EDIT", disabled_reason=TEXT_EDITOR_REASON, key="row-edit-file"),
            ActionItem("\U0001f50d Edit Translation", icon="RATE_REVIEW", disabled_reason=EDIT_TRANSLATION_REASON,
                       key="row-edit-translation"),
            ActionItem("\U0001f4cb Copy QA issue", lambda: self.copy_qa(row), icon="CONTENT_COPY",
                       disabled_reason=gate("copy_qa", "No QA issue on this row") if allowed is not None else (
                           None if row.qa_lines else "No QA issue on this row"), key="row-copy-qa"),
            ActionItem("\U0001f50a Open Audio File", icon="PLAY_CIRCLE",
                       disabled_reason=("Inline audio arrives with the media players (U7)"
                                        if allowed is None or "open_audio" in allowed else "No audio file"),
                       key="row-open-audio"),
            ActionItem(pm.ACTION_LABELS["delete_audio"], lambda: self.ctx.spawn(self.run_action("delete_audio", rows)),
                       icon="DELETE_OUTLINE", disabled_reason=gate("delete_audio", "No audio file"),
                       key="row-delete-audio"),
            ActionItem(pm.ACTION_LABELS["resolve_qa"], lambda: self.ctx.spawn(self.run_action("resolve_qa", rows)),
                       icon="BUILD", disabled_reason=gate("resolve_qa", "No resolvable QA issue"), key="row-resolve"),
            ActionItem(pm.ACTION_LABELS["insert_image"], lambda: self.ctx.spawn(self.run_action("insert_image", rows)),
                       icon="IMAGE", disabled_reason=gate("insert_image", "No missing-image QA issue"),
                       key="row-insert-image"),
            ActionItem(pm.ACTION_LABELS["remove_qa"], lambda: self.ctx.spawn(self.run_action("remove_qa", rows)),
                       icon="CLEANING_SERVICES", disabled_reason=gate("remove_qa", "No QA mark"), key="row-remove-qa"),
            ActionItem(pm.ACTION_LABELS["remove_pending"],
                       lambda: self.ctx.spawn(self.run_action("remove_pending", rows)), icon="PENDING_ACTIONS",
                       disabled_reason=gate("remove_pending", "Not a pending row with a saved output"),
                       key="row-remove-pending"),
            ActionItem(pm.ACTION_LABELS["remove_refinement"],
                       lambda: self.ctx.spawn(self.run_action("remove_refinement", rows)), icon="STAR_OUTLINE",
                       disabled_reason=gate("remove_refinement", "No refinement status"), key="row-remove-refinement"),
            ActionItem(pm.ACTION_LABELS["restore_in_progress"],
                       lambda: self.ctx.spawn(self.run_action("restore_in_progress", rows)), icon="RESTORE",
                       disabled_reason=gate("restore_in_progress", "Not in progress"), key="row-restore"),
            ActionItem("\U0001f310 Retranslate this chapter" if completed else "\U0001f310 Translate this chapter",
                       lambda: self.ctx.spawn(self.translate_chapter(row)), icon="TRANSLATE",
                       disabled_reason=None if single_ok else "Single-chapter jobs are not available in this build",
                       key="row-translate"),
        ]
        sheet = ActionSheet(items, title=row.title, subtitle=f"{row.icon} {row.label}", tablet=self.ctx.tablet)
        self.ctx.show(sheet)
        return sheet

    def copy_qa(self, row: pm.RowVM) -> Any:
        copy = self.ctx.copy_text
        if copy is None:
            return None
        return copy(f"{row.output_file or row.filename}: " + ", ".join(row.qa_lines))

    async def share_output(self, row: pm.RowVM) -> bool:
        files = self.ctx.files
        folder = self.view.output_dir if self.view is not None else ""
        path = os.path.join(folder, row.output_file) if folder and row.output_file else ""
        if files is None or not path or not os.path.isfile(path):
            self.ctx.say("The output file is not on disk")
            return False
        return bool(await files.share([path]))

    async def translate_chapter(self, row: pm.RowVM) -> Optional[str]:
        """A SINGLE_CHAPTER job on this chapter (the Reader's live panel shows it streaming).

        The desktop Book Details flow (``BookDetailsDialog._translate_single_chapter``): refused
        while a translation runs; a completed chapter asks first; a chapter with a progress entry
        is reset to pending (``library_core.mark_chapter_pending_for_retranslation``) before the
        job is queued, so the pipeline translates it instead of skipping a chapter it considers
        done (``SINGLE_CHAPTER_FILTER`` only narrows the extraction)."""
        service = self.page.service
        if not service.has_job_kind("single_chapter"):
            self.ctx.say("Single-chapter jobs are not available in this build")
            return None
        from glossarion_mobile.services.jobs import JobSpec

        jobs = getattr(service, "jobs", None)
        busy = getattr(jobs, "busy", False) if jobs is not None else False
        if callable(busy):
            busy = busy()
        if busy:
            self.ctx.say(ALREADY_RUNNING)
            return None
        source = await self.ctx.io(service.raw_source, self.page.book)
        if not source:
            self.ctx.say("The raw source file can't be found")
            return None
        status = "" if row.status in UNTRACKED_STATUSES else str(row.status or "").strip()
        if status == "completed":
            if not await self._confirm(RETRANSLATE_TITLE, retranslate_now_question(row.title or row.filename)):
                return None
        if status:
            folder = (self.view.output_dir if self.view is not None else "") or str(
                self.page.book.get("output_folder") or "")
            reset = service.core.fn("library_core", "mark_chapter_pending_for_retranslation",
                                    "_mark_chapter_pending_for_retranslation")
            if folder and os.path.isdir(folder) and reset is not None:
                try:
                    await self.ctx.io(reset, folder, row.filename)
                except Exception:
                    log.debug("progress reset failed", exc_info=True)
        spec = JobSpec(kind="single_chapter", title=f"{self.page.book.get('name') or ''} · {row.filename}",
                       inputs=(source,), params={"chapter_file": row.filename},
                       origin=service.origin_for(self.page.book))
        job_id = await service.submit(spec)
        self.ctx.say(f"Translating {row.filename}…", "Jobs", lambda: self.ctx.go("jobs"))
        return job_id

    async def retranslate(self, rows: Sequence[pm.RowVM]) -> Any:
        """Retranslate (Reset TTS in audio mode). The retranslation plan/apply arrives in U7."""
        if self._audio_mode():
            return await self.run_action("reset_tts", rows)
        core = self.page.service.core
        if not (core.available("progress_actions", "plan_retranslation")
                and core.available("progress_actions", "apply_retranslation")):
            sheet = InfoSheet(title="Retranslate", body=RETRANSLATE_U7)
            self.ctx.show(sheet)
            return sheet
        self.ctx.say("Retranslate is wired in U7")
        return None

    async def _confirm(self, title: str, body: str, *, destructive: bool = False) -> bool:
        if self.ctx.page is None:
            return True
        loop = asyncio.get_running_loop()
        answer: asyncio.Future = loop.create_future()
        dialog = ConfirmDialog(title=title, body=body, confirm_label="Yes", cancel_label="No",
                               destructive=destructive,
                               on_confirm=lambda: answer.done() or answer.set_result(True),
                               on_cancel=lambda: answer.done() or answer.set_result(False))
        self.ctx.show(dialog)
        return bool(await answer)

    async def run_action(self, action: str, rows: Sequence[pm.RowVM]) -> Optional[str]:
        """Plan (shared selection filter) -> desktop confirmation -> apply on the io pool -> result text."""
        reason = self._reason(action, rows)
        if reason is not None:
            self.ctx.say(reason)
            return None
        service = self.page.service
        try:
            plan = await self.ctx.io(pm.plan_action, service, self.view, action, list(rows))
        except CoreMissing as exc:
            self.ctx.say(f"Not available in this build ({exc.name})")
            return None
        except Exception as exc:
            log.exception("planning %s failed", action)
            self.ctx.say(f"Could not read the progress: {exc}")
            return None
        if plan.refusal:
            self.ctx.say(plan.refusal)
            return None
        copy = confirm_copy(action, plan.count, plan.extra)
        if copy is not None and not await self._confirm(copy[0], copy[1], destructive=action in _DESTRUCTIVE):
            return None
        try:
            message = await self.ctx.io(pm.apply_action, service, self.view, plan)
        except Exception as exc:
            log.exception("progress action %s failed", action)
            self.ctx.say(f"Could not update progress: {exc}")
            return None
        self.last_result = message
        self.ctx.say(message)
        self.exit_selection()
        await self.page.reload_progress(force=True)
        return message


_SINGLE_ROW_ACTIONS = ("delete_audio", "resolve_qa", "insert_image", "do_not_skip")
_DESTRUCTIVE = ("reset_tts", "delete_audio")
