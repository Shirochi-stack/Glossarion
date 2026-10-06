"""Book page › Glossary: Glossary Progress parity (UI_SPEC §3.8; Retranslation_GUI ``_build_gp_panel``).

Header "📖 {book_title}" + the file chip "📁 <base>_glossary_progress.json"; the
"⚠️ Progress file was deleted. Waiting for a new glossary progress file…" banner;
the glossary file card (Open in editor (U6) · Extract / Continue extraction job ·
⋯ Delete glossary files · Restore backup · Load as manual glossary (U6)); the stats
chips (Total · ✅ Completed · ⏭️ Skipped · 🔄 In Progress · ❌ Failed · 🔗 Merged ·
⬜ Not Translated · ✨ Not Refined · 💀 Refine Failed; the last three and Merged are
hidden at 0; tap filters, long-press jumps); the pinned Minimal Pass and Refinement
rows above the chapter rows.

Row ⋯: 📝 Show footnote · ✅ Mark as completed · 🗑️ Remove <Ch.X> from progress ·
✨ Refine this (job). Selection bar: Mark as Completed · Remove from progress ·
Refine · More (Show glossary footnote(s) · Generate completed summary (written to
``glossary_footnotes/`` and shared) · ✅/❌ Skip unmatched entries).

All reads and writes are ``glossary_progress_core`` calls (writes take the
extractor's lock + atomic replace). Empty state: "📊 No glossary extraction
progress found for: <book>" / "Run glossary extraction to see chapter progress.
Refinement entry types are listed below." + Extract glossary.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Optional, Sequence

import flet as ft

from glossarion_mobile.services.library import CoreMissing, first_value
from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.components.action_sheet import ActionItem, ActionSheet
from glossarion_mobile.ui.components.dialogs import ConfirmDialog
from glossarion_mobile.ui.components.reason_chip import ReasonChip
from glossarion_mobile.ui.library import progress_model as pm
from glossarion_mobile.ui.library.colors import GP_STATUS_PALETTE_KEY
from glossarion_mobile.ui.library.common import stat_chip, status_avatar
from glossarion_mobile.ui.library.selection_bar import BulkAction, BulkActionBar, SelectionTopBar
from glossarion_mobile.ui.theme import HIT_TARGET, status_color

__all__ = ["DELETED_BANNER", "EMPTY_BODY", "GlossaryTab", "empty_title"]

log = logging.getLogger("glossarion.library.ui")

DELETED_BANNER = "⚠️ Progress file was deleted. Waiting for a new glossary progress file…"
EMPTY_BODY = "Run glossary extraction to see chapter progress. Refinement entry types are listed below."
GLOSSARY_TOOLS_REASON = "Arrives with the Glossary tools (U6)"
REFINE_REASON = "Glossary refinement jobs arrive in U6"


def empty_title(book_title: str) -> str:
    return f"No glossary extraction progress found for: {book_title}"


def _palette(status: str) -> str:
    return GP_STATUS_PALETTE_KEY.get(status, "not_translated")


class GlossaryTab:
    def __init__(self, page: Any) -> None:
        self.page = page
        self.ctx = page.ctx
        self.view: Optional[pm.GlossaryView] = None
        self.filter_group: Optional[str] = None
        self.selected: set = set()
        self.selecting = False
        self.jump_cursor: dict = {}
        self.list_view: Optional[ft.ListView] = None
        self.last_result: Any = None

    # ---- build -----------------------------------------------------------------------------------

    def build(self) -> ft.Control:
        self.title_text = ft.Text("\U0001f4d6", theme_style=ft.TextThemeStyle.TITLE_SMALL, key="gp-title")
        self.file_chip = ft.Chip(label=ft.Text("\U0001f4c1 —"), on_click=lambda e: self.page.open_files(),
                                 visible=False, key="gp-file")
        self.banner = ft.Container(content=ft.Text(DELETED_BANNER), bgcolor=ft.Colors.TERTIARY_CONTAINER,
                                   border_radius=tokens.RADII["card"], padding=8, visible=False, key="gp-deleted")
        extract_ok = self.page.service.has_job_kind("extract_glossary")
        self.extract_button = ft.FilledTonalButton(content="Extract glossary", icon=ft.Icons.AUTO_AWESOME,
                                                   on_click=lambda e: self.ctx.spawn(self.extract()),
                                                   disabled=not extract_ok, key="gp-extract")
        self.file_card = ft.Container(
            content=ft.Column([
                ft.Text("", key="gp-file-name", theme_style=ft.TextThemeStyle.BODY_MEDIUM),
                ft.Row([
                    ft.OutlinedButton(content="Open in editor", icon=ft.Icons.EDIT_NOTE, disabled=True,
                                      tooltip=GLOSSARY_TOOLS_REASON, key="gp-open-editor"),
                    self.extract_button,
                    ft.PopupMenuButton(icon=ft.Icons.MORE_VERT, tooltip="Glossary file", items=[
                        ft.PopupMenuItem(content="Delete glossary files", disabled=True),
                        ft.PopupMenuItem(content="Restore backup", disabled=True),
                        ft.PopupMenuItem(content="Load as manual glossary", disabled=True),
                    ], key="gp-file-menu"),
                    ReasonChip(reason="Glossary tools: U6", detail=GLOSSARY_TOOLS_REASON),
                ], wrap=True, spacing=6, run_spacing=4),
            ], spacing=4, tight=True),
            bgcolor=ft.Colors.SURFACE_CONTAINER_LOW, border_radius=tokens.RADII["card"], padding=10,
            key="gp-file-card")
        self.stats_row = ft.Row(spacing=6, run_spacing=6, scroll=ft.ScrollMode.AUTO,
                                wrap=self.ctx.text_scale >= tokens.COMPACT_TEXT_SCALE, key="gp-stats")
        self.total_text = ft.Text("", theme_style=ft.TextThemeStyle.LABEL_SMALL, key="gp-total")
        self.path_row = ft.Row([
            ft.TextButton(content="Select All", on_click=lambda e: self.select_all(), key="gp-select-all"),
            ft.TextButton(content="✨ Refinement", disabled=not self._refine_ok(), tooltip=None if
                          self._refine_ok() else REFINE_REASON, key="gp-refinement"),
            ft.TextButton(content="Files", on_click=lambda e: self.page.open_files(), key="gp-files"),
            ft.TextButton(content="✏️ Open Glossary", disabled=True, tooltip=GLOSSARY_TOOLS_REASON,
                          key="gp-open-glossary"),
        ], wrap=True, spacing=0)
        self.selection_bar = SelectionTopBar(on_close=self.exit_selection, on_select_all=self.select_all,
                                             key="gp-selection")
        self.loading = ft.ProgressBar(visible=True, key="gp-loading")
        self.list_holder = ft.Container(expand=True, key="gp-list-holder")
        self.bulk_bar = BulkActionBar(page=self.ctx.page, tablet=self.ctx.tablet,
                                      compact=self.ctx.text_scale >= tokens.COMPACT_TEXT_SCALE, key="gp-bulk")
        self.root = ft.Column([
            ft.Container(content=ft.Column([
                ft.Row([self.title_text, self.file_chip], wrap=True, spacing=8),
                self.banner,
                self.file_card,
                self.stats_row,
                self.total_text,
                self.path_row,
                self.selection_bar.control,
            ], spacing=6, tight=True), padding=ft.Padding.only(left=12, right=12, top=8)),
            self.loading,
            self.list_holder,
            self.bulk_bar.control,
        ], spacing=4, expand=True, key="glossary")
        if self.page.glossary is not None:
            self.apply(self.page.glossary)
        return self.root

    def _refine_ok(self) -> bool:
        return self.page.service.has_job_kind("glossary_refine")

    # ---- data -------------------------------------------------------------------------------------

    def apply(self, view: pm.GlossaryView) -> None:
        self.view = view
        if getattr(self, "root", None) is None:
            return
        self.loading.visible = False
        title = view.book_title or str(self.page.book.get("name") or "")
        self.title_text.value = f"\U0001f4d6 {title}"
        self.banner.visible = view.deleted
        name = os.path.basename(view.path) if view.path else ""
        self.file_chip.visible = bool(name)
        self.file_chip.label = ft.Text(f"\U0001f4c1 {name}")
        file_name = self.file_card.content.controls[0]
        file_name.value = os.path.basename(view.glossary_file) if view.glossary_file else "No glossary file yet"
        self.extract_button.content = "Continue extraction" if view.path else "Extract glossary"
        self._render_chips()
        self.total_text.value = view.total_text
        self._render_rows()
        self.ctx.push(self.root)

    def _render_chips(self) -> None:
        view = self.view
        chips = []
        for chip in (view.chips if view is not None else ()):
            chips.append(stat_chip(
                chip.text, chip.status, selected=self.filter_group == chip.group, dark=self.ctx.dark,
                visible=chip.visible, key=f"gp-chip-{chip.group}",
                on_select=lambda e, g=chip.group: self.set_filter(None if self.filter_group == g else g),
                on_long_press=lambda e, g=chip.group: self.ctx.spawn(self.jump_next(g))))
        self.stats_row.controls = chips

    def set_filter(self, group: Optional[str]) -> None:
        self.filter_group = group
        if getattr(self, "root", None) is None:
            return
        self._render_chips()
        self._render_rows()
        self.ctx.push(self.root)

    def visible_rows(self) -> list:
        view = self.view
        if view is None:
            return []
        rows = list(view.rows)
        if self.filter_group:
            members = view.groups.get(self.filter_group, pm.GP_GROUPS.get(self.filter_group, (self.filter_group,)))
            rows = [r for r in rows if r.status in members]
        return rows

    def _render_rows(self) -> None:
        view = self.view
        if view is None:
            return
        rows = self.visible_rows()
        controls: list[ft.Control] = []
        if view.error:
            controls.append(ft.Text(view.error, color=ft.Colors.ERROR, key="gp-error"))
        if view.empty or (not view.path and not view.deleted):
            controls.append(ft.Container(content=ft.Column([
                ft.Text("\U0001f4ca", size=40),
                ft.Text(empty_title(view.book_title or str(self.page.book.get("name") or "")),
                        theme_style=ft.TextThemeStyle.TITLE_SMALL, text_align=ft.TextAlign.CENTER),
                ft.Text(EMPTY_BODY, text_align=ft.TextAlign.CENTER, color=ft.Colors.ON_SURFACE_VARIANT),
            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, tight=True, spacing=6), padding=16,
                alignment=ft.Alignment.CENTER, key="gp-empty"))
        for row in rows:
            controls.append(self._row_control(row))
        self.list_view = ft.ListView(controls, spacing=4, padding=ft.Padding.only(left=8, right=8, bottom=96),
                                     expand=True, build_controls_on_demand=False, key="gp-list")
        self.list_holder.content = self.list_view

    def _row_control(self, row: pm.GlossaryRowVM) -> ft.Control:
        dark = self.ctx.dark
        selected = row.key in self.selected
        palette = _palette(row.status)
        prefix = "Minimal Pass" if row.kind == "minimal" else ("Refinement" if row.kind == "refinement" else "")
        title = row.title if not prefix or row.title.startswith(prefix) else f"{prefix} · {row.title}"
        lines: list[ft.Control] = [
            ft.Text(title, theme_style=ft.TextThemeStyle.BODY_MEDIUM, max_lines=2, overflow=ft.TextOverflow.ELLIPSIS),
            ft.Row([ft.Text(f"{row.icon} {row.label}", size=11, weight=ft.FontWeight.W_600,
                            color=status_color(palette, dark))]
                   + [ft.Text(b, size=11) for b in row.badges], spacing=6, wrap=True),
        ]
        if row.subtitle:
            lines.append(ft.Text(row.subtitle, theme_style=ft.TextThemeStyle.BODY_SMALL, max_lines=2,
                                 overflow=ft.TextOverflow.ELLIPSIS, color=ft.Colors.ON_SURFACE_VARIANT))
        if row.qa_lines:
            lines.append(ft.Text("\n".join(row.qa_lines[:2]), theme_style=ft.TextThemeStyle.BODY_SMALL,
                                 color=ft.Colors.ERROR, max_lines=3, overflow=ft.TextOverflow.ELLIPSIS))
        return ft.Container(
            content=ft.Row([
                status_avatar(palette, emoji=row.icon, label=f"{title}, {row.label}", dark=dark),
                ft.Column(lines, spacing=1, expand=True, tight=True),
                ft.IconButton(icon=ft.Icons.MORE_VERT, tooltip="Glossary row actions", size_constraints=HIT_TARGET,
                              on_click=lambda e, r=row: self.open_row_sheet(r), key="gp-row-more"),
            ], spacing=8, vertical_alignment=ft.CrossAxisAlignment.START),
            key=ft.ScrollKey(row.key),
            padding=ft.Padding.symmetric(horizontal=8, vertical=6),
            border_radius=tokens.RADII["card"],
            bgcolor=(ft.Colors.SECONDARY_CONTAINER if selected else
                     ft.Colors.SURFACE_CONTAINER_HIGH if row.pinned else ft.Colors.SURFACE_CONTAINER_LOW),
            border=ft.Border.all(2, ft.Colors.PRIMARY) if selected else None,
            on_click=lambda e, r=row: self.on_row_tap(r),
            on_long_press=lambda e, r=row: self.on_row_long_press(r),
            ink=True,
        )

    async def jump_next(self, group: str) -> Optional[str]:
        rows = self.visible_rows()
        groups = self.view.groups if self.view is not None else pm.GP_GROUPS
        members = groups.get(group, pm.GP_GROUPS.get(group, (group,)))
        start = self.jump_cursor.get(group, -1) + 1
        for index in list(range(start, len(rows))) + list(range(0, start)):
            if rows[index].status in members:
                self.jump_cursor[group] = index
                self.ctx.haptic("selection_click")
                if self.list_view is not None:
                    try:
                        await self.list_view.scroll_to(scroll_key=rows[index].key, duration=250)
                    except Exception:
                        log.debug("scroll_to failed", exc_info=True)
                return rows[index].key
        self.ctx.say("No row with that status")
        return None

    # ---- selection ---------------------------------------------------------------------------------

    def on_row_tap(self, row: pm.GlossaryRowVM) -> Any:
        if self.selecting:
            self.toggle_select(row.key)
            return None
        return self.open_row_sheet(row)

    def on_row_long_press(self, row: pm.GlossaryRowVM) -> None:
        if not self.selecting:
            self.ctx.haptic("medium_impact")
        self.selecting = True
        self.toggle_select(row.key, force=True)

    def toggle_select(self, key: str, force: Optional[bool] = None) -> None:
        on = (key not in self.selected) if force is None else force
        if on:
            self.selected.add(key)
        else:
            self.selected.discard(key)
        if not self.selected:
            self.selecting = False
        self._render_rows()
        self._sync_selection()

    def select_all(self) -> None:
        self.selecting = True
        self.selected = {r.key for r in self.visible_rows()}
        self._render_rows()
        self._sync_selection()

    def exit_selection(self) -> None:
        self.selecting = False
        self.selected = set()
        if self.view is not None:
            self._render_rows()
        self._sync_selection()

    def selected_rows(self) -> list:
        return [r for r in (self.view.rows if self.view is not None else ()) if r.key in self.selected]

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
        self.ctx.push(self.root)

    # ---- actions ------------------------------------------------------------------------------------

    def _reason(self, action: str, rows: Sequence[pm.GlossaryRowVM]) -> Optional[str]:
        names = {"mark_completed": "mark_glossary_completed", "remove": "remove_glossary_progress",
                 "footnote": "glossary_footnotes", "summary": "write_glossary_summary"}
        if not self.page.service.core.available("glossary_progress_core", names[action]):
            return f"Needs glossary_progress_core.{names[action]} (not in this build)"
        if self.view is None or self.view.state is None or not self.view.path:
            return "No glossary progress file"
        if self.view.deleted:
            return "The glossary progress file was deleted"
        if action == "mark_completed" and rows and all(r.status == "completed" for r in rows):
            return "Already completed"
        if action == "mark_completed" and rows and all(r.kind == "minimal" for r in rows):
            return "The Minimal pass is completed by running it"
        if action == "footnote" and not any(r.kind == "chapter" for r in rows):
            return "Footnotes are per chapter"
        return None

    def bulk_actions(self, rows: Sequence[pm.GlossaryRowVM]) -> tuple:
        primary = [
            BulkAction("mark_completed", "✅ Mark as Completed", "TASK_ALT",
                       lambda: self.ctx.spawn(self.run_action("mark_completed", rows)),
                       self._reason("mark_completed", rows)),
            BulkAction("remove", "\U0001f5d1️ Remove from progress", "DELETE_OUTLINE",
                       lambda: self.ctx.spawn(self.run_action("remove", rows)), self._reason("remove", rows),
                       destructive=True),
            BulkAction("refine", "✨ Refine", "AUTO_FIX_HIGH", disabled_reason=None if self._refine_ok()
                       else REFINE_REASON),
        ]
        skip = bool(self.page.service.cfg("glossary_progress_skip_unmatched_entries", True))
        more = [
            BulkAction("footnote", "\U0001f4dd Show glossary footnote(s)", "NOTES",
                       lambda: self.ctx.spawn(self.show_footnotes(rows)), self._reason("footnote", rows)),
            BulkAction("summary", "\U0001f4c4 Generate completed summary", "SUMMARIZE",
                       lambda: self.ctx.spawn(self.generate_summary()), self._reason("summary", rows)),
            BulkAction("skip_unmatched", ("✅" if skip else "❌") + " Skip unmatched entries", "RULE",
                       lambda: self.toggle_skip_unmatched()),
        ]
        return primary, more

    def open_row_sheet(self, row: pm.GlossaryRowVM) -> ActionSheet:
        rows = [row]
        label = row.title.split(" · ")[0]
        items = [
            ActionItem("\U0001f4dd Show footnote", lambda: self.ctx.spawn(self.show_footnotes(rows)), icon="NOTES",
                       disabled_reason=self._reason("footnote", rows)),
            ActionItem("✅ Mark as completed", lambda: self.ctx.spawn(self.run_action("mark_completed", rows)),
                       icon="TASK_ALT", disabled_reason=self._reason("mark_completed", rows)),
            ActionItem(f"\U0001f5d1️ Remove {label} from progress ({row.label})",
                       lambda: self.ctx.spawn(self.run_action("remove", rows)), icon="DELETE_OUTLINE",
                       disabled_reason=self._reason("remove", rows), destructive=True),
            ActionItem("✨ Refine this", icon="AUTO_FIX_HIGH",
                       disabled_reason=None if self._refine_ok() else REFINE_REASON),
        ]
        sheet = ActionSheet(items, title=row.title, subtitle=f"{row.icon} {row.label}", tablet=self.ctx.tablet)
        self.ctx.show(sheet)
        return sheet

    async def run_action(self, action: str, rows: Sequence[pm.GlossaryRowVM]) -> Any:
        reason = self._reason(action, rows)
        if reason is not None:
            self.ctx.say(reason)
            return None
        if action == "remove":
            count = len(rows)
            body = (f"Remove {rows[0].title.split(' · ')[0]} from progress ({rows[0].label})?" if count == 1
                    else f"Remove {count} chapters from progress?")
            if self.ctx.page is not None:
                loop = asyncio.get_running_loop()
                answer: asyncio.Future = loop.create_future()
                dialog = ConfirmDialog(title="Remove from progress", body=body, confirm_label="Remove",
                                       destructive=True,
                                       on_confirm=lambda: answer.done() or answer.set_result(True),
                                       on_cancel=lambda: answer.done() or answer.set_result(False))
                self.ctx.show(dialog)
                if not await answer:
                    return None
        try:
            result = await self.ctx.io(pm.run_glossary_action, self.page.service, self.view, action, list(rows))
        except CoreMissing as exc:
            self.ctx.say(f"Not available in this build ({exc.name})")
            return None
        except Exception as exc:
            log.exception("glossary progress action %s failed", action)
            self.ctx.say(f"Could not update the glossary progress: {exc}")
            return None
        self.last_result = result
        changed = bool(first_value(result, "changed", default=True))
        if action == "mark_completed":
            message = (f"✅ Marked {len(rows)} row{'s' if len(rows) != 1 else ''} as completed" if changed
                       else "Nothing to mark as completed")
        else:
            message = (f"🗑️ Removed {len(rows)} row{'s' if len(rows) != 1 else ''} from progress"
                       if changed else "Nothing was removed")
        self.ctx.say(message)
        self.exit_selection()
        await self.page.reload_glossary()
        return result

    async def show_footnotes(self, rows: Sequence[pm.GlossaryRowVM]) -> Any:
        reason = self._reason("footnote", rows)
        if reason is not None:
            self.ctx.say(reason)
            return None
        chapters = [r for r in rows if r.kind == "chapter"]
        try:
            result = await self.ctx.io(pm.run_glossary_action, self.page.service, self.view, "footnote", chapters)
        except Exception as exc:
            self.ctx.say(f"Could not build the footnote: {exc}")
            return None
        error = first_value(result, "error")
        if error:
            self.ctx.say(str(error[2] if isinstance(error, (tuple, list)) and len(error) > 2 else error))
            return None
        markdown = str(first_value(result, "markdown", default="") or "")
        sheet = _MarkdownSheet("📝 Glossary footnote" + ("s" if len(chapters) > 1 else ""),
                               markdown or "_No glossary terms matched this chapter._", copy=self.ctx.copy_text)
        if self.ctx.page is not None:
            sheet.show(self.ctx.page)
        return sheet

    async def generate_summary(self) -> Optional[str]:
        reason = self._reason("summary", [])
        if reason is not None:
            self.ctx.say(reason)
            return None
        try:
            result = await self.ctx.io(pm.run_glossary_action, self.page.service, self.view, "summary", [])
        except Exception as exc:
            self.ctx.say(f"Could not write the summary: {exc}")
            return None
        path = result[0] if isinstance(result, (tuple, list)) and result else first_value(result, "path")
        if path and os.path.isfile(str(path)) and self.ctx.files is not None:
            self.ctx.say(f"📄 Wrote {os.path.basename(str(path))}")
            await self.ctx.files.share([str(path)])
            return str(path)
        self.ctx.say("The summary was not written")
        return None

    def toggle_skip_unmatched(self) -> bool:
        service = self.page.service
        value = not bool(service.cfg("glossary_progress_skip_unmatched_entries", True))
        service.set_cfg("glossary_progress_skip_unmatched_entries", value)
        self.ctx.say(("✅" if value else "❌") + " Skip unmatched entries")
        self._sync_selection()
        return value

    async def extract(self) -> Optional[str]:
        service = self.page.service
        if not service.has_job_kind("extract_glossary"):
            self.ctx.say("The job service is not running")
            return None
        from glossarion_mobile.services.jobs import JobSpec

        source = await self.ctx.io(service.raw_source, self.page.book)
        if not source:
            self.ctx.say("The raw source file can't be found")
            return None
        spec = JobSpec(kind="extract_glossary", title=str(self.page.book.get("name") or os.path.basename(source)),
                       inputs=(source,), origin=service.origin_for(self.page.book))
        job_id = await service.submit(spec)
        self.ctx.say("Extracting the glossary…", "Jobs", lambda: self.ctx.go("jobs"))
        return job_id


class _MarkdownSheet:
    """Footnote sheet: Markdown + Copy · Close (desktop ``Show Glossary Footnote(s)`` dialog)."""

    def __init__(self, title: str, markdown: str, *, copy: Any = None) -> None:
        self.markdown = markdown
        self._page: Any = None
        buttons = [ft.TextButton(content="Close", on_click=lambda e: self.close())]
        if copy is not None:
            buttons.insert(0, ft.TextButton(content="Copy", on_click=lambda e: copy(markdown)))
        self.sheet = ft.BottomSheet(
            content=ft.Container(content=ft.Column([
                ft.Text(title, theme_style=ft.TextThemeStyle.TITLE_MEDIUM, weight=ft.FontWeight.W_600),
                ft.Markdown(markdown, selectable=True, key="footnote-md"),
                ft.Row(buttons, alignment=ft.MainAxisAlignment.END),
            ], tight=True, scroll=ft.ScrollMode.AUTO), padding=16),
            show_drag_handle=True, scrollable=True, bgcolor=ft.Colors.SURFACE_CONTAINER_HIGH)

    def show(self, page: Any) -> None:
        self._page = page
        page.show_dialog(self.sheet)

    def close(self) -> None:
        if self._page is not None and getattr(self.sheet, "open", False):
            self._page.pop_dialog()

