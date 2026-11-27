"""Book page › Images: the Progress Manager's Interactive Images tab (desktop ``interactive_images_gui``
``ImagesFilePanel``; the shared ``interactive_images_progress_core``).

Laid out like the Glossary tab: ▶ Build pending · ⋯ (⟳ Refresh · Select All · Settings · Styles &
preview · Files); the stats chips (✅ Built · ⚠️ Partial · 🕒 Stale · ❌ Failed · 🚫 Rejected · ⬜ Not built;
tap filters, long-press jumps); one line with the totals; one row per translated chapter (icon, chapter,
images, status; a stale row = the chapter changed since its images were built).

Row ⋯ and the selection bar run the desktop context menu's actions, with the desktop's rules
(``interactive_images_progress_core.ROW_ACTIONS`` / ``action_reason``): 🖼 Build / Rebuild · 🔁 Retry
failed · ❓ Mark for rebuild · 🗑 Delete images (confirmed; strips Glossarion's markup, text-checked) ·
🖼 View image(s) · 📄 Open chapter (the Reader) · 📋 Show plan. Builds are ``interactive_images`` jobs
(``job_kinds.interactive_images``: the shared runner on the job's owner); the Book page strip shows them.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Optional, Sequence

import flet as ft

from glossarion_mobile.services.library import CoreMissing
from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.components.action_sheet import ActionItem, ActionSheet
from glossarion_mobile.ui.components.dialogs import ConfirmDialog, close_dialog
from glossarion_mobile.ui.components.pull_to_refresh import PullToRefresh
from glossarion_mobile.ui.components.windowed_list import WindowedList
from glossarion_mobile.ui.library import progress_model as pm
from glossarion_mobile.ui.library.common import BusyBar, stat_chip, status_avatar
from glossarion_mobile.ui.library.selection_bar import BulkAction, BulkActionBar, SelectionTopBar
from glossarion_mobile.ui.theme import HIT_TARGET, status_color

__all__ = ["ACTION_ICONS", "ImagesTab", "SETTINGS_ROUTE"]

log = logging.getLogger("glossarion.library.ui")

SETTINGS_ROUTE = "tools.interactive-images"
#: action id -> Material icon (the labels are the shared desktop ones)
ACTION_ICONS = {"build": "IMAGE", "retry": "REPLAY", "mark": "HELP_OUTLINE", "delete": "DELETE_OUTLINE",
                "open_images": "PHOTO_LIBRARY", "open_chapter": "MENU_BOOK", "show_plan": "LIST_ALT",
                "open_folder": "FOLDER_OPEN"}
#: Bulk bar: these first, the rest under More
PRIMARY_BULK = ("build", "retry", "delete")
REPLACE_LIMIT = 40


class ImagesTab:
    def __init__(self, page: Any) -> None:
        self.page = page
        self.ctx = page.ctx
        self.view: Optional[pm.ImagesView] = None
        self.filter_group: Optional[str] = None
        self.selected: set = set()
        self.selecting = False
        self.jump_cursor: dict = {}
        self.rows_list: Optional[WindowedList] = None
        self._row_keys: list = []
        self.renders = 0
        self.last_result: Any = None
        self.sheet: Any = None
        self.busy = BusyBar(self.ctx, lambda: getattr(self, "loading", None), idle=lambda: self.view is None)

    # ---- build -----------------------------------------------------------------------------------

    def build(self) -> ft.Control:
        self.build_button = ft.FilledTonalButton(content="▶ Build pending", icon=ft.Icons.IMAGE,
                                                 on_click=lambda e: self.ctx.spawn(self.build_pending()),
                                                 key="ix-build")
        self.more_menu = ft.PopupMenuButton(icon=ft.Icons.MORE_VERT, tooltip="More", key="ix-more-menu", items=[
            ft.PopupMenuItem(content="⟳ Refresh", on_click=lambda e: self.ctx.spawn(self.page.reload_images(force=True)),
                             key="ix-refresh"),
            ft.PopupMenuItem(content="Select All", on_click=lambda e: self.select_all(), key="ix-select-all"),
            ft.PopupMenuItem(content="⚙ Settings & references", on_click=lambda e: self.open_settings("models"),
                             key="ix-settings"),
            ft.PopupMenuItem(content="🎨 Styles & preview", on_click=lambda e: self.open_settings("styles"),
                             key="ix-styles"),
            ft.PopupMenuItem(content="📂 Files", on_click=lambda e: self.page.open_files(), key="ix-files"),
        ])
        self.sync_build_button()
        self.stats_row = ft.Row(spacing=6, run_spacing=6, scroll=ft.ScrollMode.AUTO,
                                wrap=self.ctx.text_scale >= tokens.COMPACT_TEXT_SCALE, key="ix-stats")
        self.total_text = ft.Text("", theme_style=ft.TextThemeStyle.LABEL_SMALL, key="ix-total")
        self.selection_bar = SelectionTopBar(on_close=self.exit_selection, on_select_all=self.select_all,
                                             key="ix-selection")
        self.loading = ft.ProgressBar(visible=True, key="ix-loading")
        self.pull = PullToRefresh(lambda: self.page.reload_images(force=True), spawn=self.ctx.spawn,
                                  haptic=getattr(self.ctx, "haptic", None), key="ix-pull")
        self.notice = ft.Column([], spacing=4, tight=True, visible=False, key="ix-notice")
        self.rows_list = WindowedList(build_row=lambda row, _i: self._row_control(row), key_of=lambda row: row.key,
                                      key="ix", on_scroll_extra=self.pull.handle)
        self.bulk_bar = BulkActionBar(page=self.ctx.page, tablet=self.ctx.tablet,
                                      compact=self.ctx.text_scale >= tokens.COMPACT_TEXT_SCALE, key="ix-bulk")
        self.root = ft.Column([
            ft.Container(content=ft.Column([
                ft.Row([ft.Row([self.build_button], wrap=True, spacing=8, run_spacing=4, expand=True), self.more_menu],
                       spacing=0, vertical_alignment=ft.CrossAxisAlignment.START, key="ix-tools"),
                self.stats_row,
                self.total_text,
                self.selection_bar.control,
            ], spacing=6, tight=True), padding=ft.Padding.only(left=12, right=12, top=8)),
            self.loading,
            ft.Container(content=ft.Column([self.pull.bar, self.notice, self.rows_list.control], spacing=0,
                                           expand=True), expand=True, key="ix-list-holder"),
            self.bulk_bar.control,
        ], spacing=4, expand=True, key="images")
        if getattr(self.page, "images", None) is not None:
            self.apply(self.page.images)
        return self.root

    # ---- state ------------------------------------------------------------------------------------

    def running_job(self, view: Any = None) -> Any:
        """The book's queued / running Interactive Images job (None when there is none)."""
        from glossarion_mobile.ui.library.book_page import job_kind_of, jobs_for_book

        jobs_view = view if view is not None else getattr(self.page, "job_view", None)
        for snap in jobs_for_book(jobs_view, self.page.bid, self.page.book, folder=self.page.workspace):
            if job_kind_of(snap) == pm.IMAGES_KIND:
                return snap
        return None

    def build_reason(self, view: Any = None) -> Optional[str]:
        if not self.page.service.has_job_kind(pm.IMAGES_KIND):
            return "The job service is not running"
        if self.running_job(view) is not None:
            return "Interactive Images is running for this book"
        if self.view is not None and self.view.error:
            return self.view.error
        return None

    def sync_build_button(self, view: Any = None) -> Optional[str]:
        """The Book page calls this on every JobsView delivery (like the Glossary tab's Extract)."""
        reason = self.build_reason(view)
        button = getattr(self, "build_button", None)
        if button is not None and (button.disabled != (reason is not None) or button.tooltip != reason):
            button.disabled = reason is not None
            button.tooltip = reason
            self.ctx.push(button)
        return reason

    def apply(self, view: pm.ImagesView) -> None:
        previous = {r.key: r for r in (self.view.rows if self.view is not None else ())}
        self.view = view
        if getattr(self, "root", None) is None:
            return
        self.loading.visible = self.busy.active
        self._render_chips()
        self.total_text.value = view.total_text
        self.sync_build_button()
        self._update_rows(previous)
        self.ctx.push(self.root)

    def _render_chips(self) -> None:
        chips = []
        for chip in (self.view.chips if self.view is not None else ()):
            chips.append(stat_chip(
                chip.text, chip.status, selected=self.filter_group == chip.group, dark=self.ctx.dark,
                visible=chip.visible, key=f"ix-chip-{chip.group}",
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
            members = view.groups.get(self.filter_group, (self.filter_group,))
            rows = [r for r in rows if r.status in members]
        return rows

    def _render_notice(self) -> None:
        view = self.view
        controls: list = []
        if view is not None and view.error:
            controls.append(ft.Text(view.error, color=ft.Colors.ON_SURFACE_VARIANT, key="ix-error"))
        elif view is not None and view.empty:
            controls.append(ft.Container(content=ft.Column([
                ft.Text("\U0001f5bc", size=40),
                ft.Text(pm.IMAGES_EMPTY, text_align=ft.TextAlign.CENTER, color=ft.Colors.ON_SURFACE_VARIANT),
            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, tight=True, spacing=6), padding=16,
                alignment=ft.Alignment.CENTER, key="ix-empty"))
        self.notice.controls = controls
        self.notice.visible = bool(controls)

    def _render_rows(self, *, keep: bool = False) -> None:
        if self.view is None or self.rows_list is None:
            return
        self._render_notice()
        rows = self.visible_rows()
        self._row_keys = [r.key for r in rows]
        self.rows_list.set_items(rows, keep_window=keep, keep_rendered=keep)
        self.renders += 1

    def _update_rows(self, previous: dict) -> None:
        if self.rows_list is None:
            return
        rows = self.visible_rows()
        if [r.key for r in rows] != self._row_keys or not self._row_keys:
            self._render_rows(keep=bool(self._row_keys))
            return
        self._render_notice()
        for row in rows:
            old = previous.get(row.key)
            if old is None or (old.status, old.images, old.error, old.subtitle) != (
                    row.status, row.images, row.error, row.subtitle):
                self.rows_list.replace(row)

    def _replace_keys(self, keys: Any) -> None:
        rows_list = self.rows_list
        if rows_list is None:
            return
        wanted = set(keys)
        mounted = [row for row in self.visible_rows() if row.key in wanted and row.key in rows_list.controls]
        if len(mounted) > REPLACE_LIMIT:
            self._render_rows(keep=True)
            return
        for row in mounted:
            rows_list.replace(row)

    def _row_control(self, row: pm.ImagesRowVM) -> ft.Control:
        dark = self.ctx.dark
        selected = row.key in self.selected
        lines: list = [
            ft.Text(row.title, theme_style=ft.TextThemeStyle.BODY_MEDIUM, max_lines=2,
                    overflow=ft.TextOverflow.ELLIPSIS),
            ft.Text(f"{row.icon} {row.label}", size=11, weight=ft.FontWeight.W_600,
                    color=status_color(row.palette, dark)),
        ]
        if row.subtitle:
            lines.append(ft.Text(row.subtitle, theme_style=ft.TextThemeStyle.BODY_SMALL, max_lines=2,
                                 overflow=ft.TextOverflow.ELLIPSIS, color=ft.Colors.ON_SURFACE_VARIANT))
        return ft.Container(
            content=ft.Row([
                status_avatar(row.palette, emoji=row.icon, label=f"{row.title}, {row.label}", dark=dark),
                ft.Column(lines, spacing=1, expand=True, tight=True),
                ft.IconButton(icon=ft.Icons.MORE_VERT, tooltip="Interactive Images row actions",
                              size_constraints=HIT_TARGET, on_click=lambda e, r=row: self.open_row_sheet(r),
                              key="ix-row-more"),
            ], spacing=8, vertical_alignment=ft.CrossAxisAlignment.START),
            key=ft.ScrollKey(row.key),
            padding=ft.Padding.symmetric(horizontal=8, vertical=6),
            border_radius=tokens.RADII["card"],
            bgcolor=ft.Colors.SECONDARY_CONTAINER if selected else ft.Colors.SURFACE_CONTAINER_LOW,
            border=ft.Border.all(2, ft.Colors.PRIMARY) if selected else None,
            on_click=lambda e, r=row: self.on_row_tap(r),
            on_long_press=lambda e, r=row: self.on_row_long_press(r),
            ink=True,
        )

    async def jump_next(self, group: str) -> Optional[str]:
        rows = self.visible_rows()
        members = (self.view.groups if self.view is not None else {}).get(group, (group,))
        start = self.jump_cursor.get(group, -1) + 1
        for index in list(range(start, len(rows))) + list(range(0, start)):
            if rows[index].status in members:
                self.jump_cursor[group] = index
                self.ctx.haptic("selection_click")
                if self.rows_list is not None:
                    await self.rows_list.jump_to(rows[index].key)
                return rows[index].key
        self.ctx.say("No row with that status")
        return None

    # ---- selection ---------------------------------------------------------------------------------

    def on_row_tap(self, row: pm.ImagesRowVM) -> Any:
        if self.selecting:
            self.toggle_select(row.key)
            return None
        return self.open_row_sheet(row)

    def on_row_long_press(self, row: pm.ImagesRowVM) -> None:
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
        self._replace_keys((key,))
        self._sync_selection()

    def select_all(self) -> None:
        self.selecting = True
        before = set(self.selected)
        self.selected = {r.key for r in self.visible_rows()}
        self._replace_keys(self.selected ^ before)
        self._sync_selection()

    def exit_selection(self) -> None:
        self.selecting = False
        before, self.selected = set(self.selected), set()
        if self.view is not None:
            self._replace_keys(before)
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

    def reason(self, action: str, rows: Sequence[pm.ImagesRowVM]) -> Optional[str]:
        return pm.images_action_reason(self.page.service, self.view, action, rows,
                                       running=self.running_job() is not None)

    def _labels(self, rows: Sequence[pm.ImagesRowVM]) -> list:
        try:
            import interactive_images_progress_core as ipc
        except ImportError:
            return []
        out = []
        for action, _label in ipc.ROW_ACTIONS:
            if action == "open_folder":
                continue  # ⋯ › 📂 Files
            label = ipc.action_label(action, [r.raw for r in rows])
            if action == "open_images":
                label = label.replace("Open", "View")
            out.append((action, label))
        return out

    def bulk_actions(self, rows: Sequence[pm.ImagesRowVM]) -> tuple:
        primary, more = [], []
        for action, label in self._labels(rows):
            item = BulkAction(action, label, ACTION_ICONS[action],
                              lambda a=action: self.ctx.spawn(self.run_action(a, rows)), self.reason(action, rows),
                              destructive=action == "delete")
            (primary if action in PRIMARY_BULK else more).append(item)
        return primary, more

    def open_row_sheet(self, row: pm.ImagesRowVM) -> ActionSheet:
        rows = [row]
        items = [ActionItem(label, lambda a=action: self.ctx.spawn(self.run_action(a, rows)),
                            icon=ACTION_ICONS[action], disabled_reason=self.reason(action, rows),
                            destructive=action == "delete", key=f"ix-act-{action}")
                 for action, label in self._labels(rows)]
        sheet = ActionSheet(items, title=row.title, subtitle=f"{row.icon} {row.label}", tablet=self.ctx.tablet)
        self.sheet = sheet
        self.ctx.show(sheet)
        return sheet

    async def build_pending(self) -> Optional[str]:
        return await self.submit(None)

    async def submit(self, keys: Optional[Sequence[str]]) -> Optional[str]:
        reason = self.build_reason()
        if reason is not None:
            self.ctx.say(reason)
            return None
        service = self.page.service
        try:
            job_id = await service.submit(pm.images_spec(service, self.page.book, keys))
        except Exception as exc:
            self.ctx.say(f"Could not start: {exc}")
            return None
        self.ctx.haptic("medium_impact")
        self.sync_build_button()
        return job_id

    async def run_action(self, action: str, rows: Sequence[pm.ImagesRowVM]) -> Any:
        reason = self.reason(action, rows)
        if reason is not None:
            self.ctx.say(reason)
            return None
        import interactive_images_progress_core as ipc

        targets = ipc.action_targets(action, [r.raw for r in rows])
        keys = [t.key for t in targets]
        if action in ("build", "retry"):
            self.exit_selection() if self.selecting else None
            return await self.submit(keys)
        if action == "open_chapter":
            return self.page.open_reader(chapter_filename=keys[0], mode="translated") if keys else None
        if action in ("show_plan", "open_images"):
            try:
                result = await self.ctx.io(pm.run_images_action, self.page.service, self.view, action, list(rows))
            except Exception as exc:
                self.ctx.say(f"Could not read the plan: {exc}")
                return None
            sheet = (_PlanSheet(result, copy=self.ctx.copy_text) if action == "show_plan"
                     else _ImagesSheet([p for t in result for p in t.get("images", ())]))
            self.sheet = sheet
            if self.ctx.page is not None:
                sheet.show(self.ctx.page)
            return sheet
        if action == "delete":
            if self.ctx.page is not None:
                loop = asyncio.get_running_loop()
                answer: asyncio.Future = loop.create_future()
                dialog = ConfirmDialog(title="Delete Images", body=ipc.delete_confirmation(len(targets)),
                                       confirm_label="Delete", destructive=True,
                                       on_confirm=lambda: answer.done() or answer.set_result(True),
                                       on_cancel=lambda: answer.done() or answer.set_result(False))
                self.ctx.show(dialog)
                if not await answer:
                    return None
        async with self.busy.run():
            try:
                result = await self.ctx.io(pm.run_images_action, self.page.service, self.view, action, list(rows))
            except CoreMissing as exc:
                self.ctx.say(f"Not available in this build ({exc.name})")
                return None
            except Exception as exc:
                log.exception("interactive images action %s failed", action)
                self.ctx.say(f"Could not update the Interactive Images progress: {exc}")
                return None
            self.last_result = result
            self.exit_selection()
            await self.page.reload_images(force=True)
        return result

    def open_settings(self, tab: str = "models") -> None:
        self.ctx.go(SETTINGS_ROUTE, None, {"out": self.page.bid, "tab": tab})


class _PlanSheet:
    """Show plan: the plan text of the chosen chapters (the desktop "Interactive Images plan" window)."""

    def __init__(self, text: str, *, copy: Any = None) -> None:
        self.text = str(text or "")
        self._page: Any = None
        buttons = [ft.TextButton(content="Close", on_click=lambda e: self.close())]
        if copy is not None:
            buttons.insert(0, ft.TextButton(content="Copy", on_click=lambda e: copy(self.text)))
        self.sheet = ft.BottomSheet(
            content=ft.Container(content=ft.Column([
                ft.Text("📋 Interactive Images plan", theme_style=ft.TextThemeStyle.TITLE_MEDIUM,
                        weight=ft.FontWeight.W_600),
                ft.Text(self.text, selectable=True, theme_style=ft.TextThemeStyle.BODY_SMALL, key="ix-plan-text"),
                ft.Row(buttons, alignment=ft.MainAxisAlignment.END),
            ], tight=True, scroll=ft.ScrollMode.AUTO), padding=16),
            show_drag_handle=True, scrollable=True, bgcolor=ft.Colors.SURFACE_CONTAINER_HIGH)

    def show(self, page: Any) -> None:
        self._page = page
        page.show_dialog(self.sheet)

    def close(self) -> None:
        close_dialog(self._page, self.sheet)


class _ImagesSheet:
    """View image(s): the chapter's generated images, read as bytes (``safe_image`` checked)."""

    MAX_IMAGES = 12

    def __init__(self, paths: Sequence[str]) -> None:
        self.paths = [p for p in paths if p and os.path.isfile(p)][: self.MAX_IMAGES]
        self._page: Any = None
        images: list = []
        for path in self.paths:
            data = _image_bytes(path)
            if data is None:
                continue
            images.append(ft.Column([
                ft.Image(src=data, fit=ft.BoxFit.CONTAIN, border_radius=tokens.RADII["card"]),
                ft.Text(os.path.basename(path), theme_style=ft.TextThemeStyle.BODY_SMALL,
                        color=ft.Colors.ON_SURFACE_VARIANT),
            ], spacing=4, tight=True))
        if not images:
            images.append(ft.Text("No images to show."))
        self.sheet = ft.BottomSheet(
            content=ft.Container(content=ft.Column([
                ft.Text("🖼 Interactive images", theme_style=ft.TextThemeStyle.TITLE_MEDIUM, weight=ft.FontWeight.W_600),
                *images,
                ft.Row([ft.TextButton(content="Close", on_click=lambda e: self.close())],
                       alignment=ft.MainAxisAlignment.END),
            ], tight=True, scroll=ft.ScrollMode.AUTO, spacing=10), padding=16, key="ix-images-sheet"),
            show_drag_handle=True, scrollable=True, bgcolor=ft.Colors.SURFACE_CONTAINER_HIGH)

    def show(self, page: Any) -> None:
        self._page = page
        page.show_dialog(self.sheet)

    def close(self) -> None:
        close_dialog(self._page, self.sheet)


def _image_bytes(path: str) -> Optional[bytes]:
    """The file's bytes when it decodes as an image (``safe_image.open_image``: the owner's rule for images
    the app did not draw itself), else None."""
    try:
        from safe_image import open_image

        with open_image(path) as image:
            image.size  # noqa: B018 - decoded header
        with open(path, "rb") as handle:
            return handle.read()
    except Exception:
        log.debug("not showing %s", path, exc_info=True)
        return None
