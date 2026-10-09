"""Library delete confirmation, both desktop levels (UI_SPEC §3.4; epub_library ``_delete_books_prompt``).

The targets, the safe-root gate, the Library/Raw pair, the silent unregister list
and whether the keyword is needed come from ``library_core.plan_delete``
(``LibraryService.plan_delete_blocking``). This module only presents them:

* every target Not started -> a simple ``ConfirmDialog`` with the desktop text
  "Permanently delete N Not Started item(s)? … This cannot be undone.";
* otherwise -> the full-screen ``DeleteConfirmView``: per-target checkboxes with
  the folder contents summary (``summarize_folder_contents``), the desktop warning,
  and a field that must contain "halgakos" or "delete" (case-insensitive,
  whitespace forgiven) before the red Delete button enables.

The delete itself runs on the io pool (``execute_delete``); there is no undo.
"""

from __future__ import annotations

import functools
from typing import Any, Callable, Optional, Sequence

import flet as ft

from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.components.dialogs import ConfirmDialog
from glossarion_mobile.ui.theme import HIT_TARGET

__all__ = [
    "DELETE_VIEW_ROUTE",
    "DeleteConfirmView",
    "DeleteFlow",
    "format_delete_detail",
    "keyword_ok",
    "simple_confirm",
    "simple_confirm_text",
]

#: The keyword view's ``View.route`` (not a router route: no screen or other overlay uses it).
DELETE_VIEW_ROUTE = "/library/delete-confirm"

WARNING_TEXT = ("This removes the checked item(s) permanently — for output folders that's recursive: every "
                "translated chapter, progress tracker, image, and compiled EPUB inside goes with it. This cannot "
                "be undone.")


def keyword_ok(text: Any, keywords: Sequence[str]) -> bool:
    return str(text or "").strip().lower() in {k.lower() for k in keywords}


def _target_lines(target: Any) -> list:
    if target.is_folder:
        lines = [f"▾  {target.label}  —  output folder", f"       {target.path}"]
        lines.extend(target.contents or ["    · (folder is empty)"])
    else:
        lines = [f"▾  {target.label}  —  file ({target.size_text or '?'})", f"       {target.path}"]
    return lines


def format_delete_detail(targets: Sequence[Any]) -> str:
    """Desktop ``_format_delete_detail``: the first 10 targets, then "… and N more item(s)."."""
    lines: list[str] = []
    for target in list(targets)[:10]:
        lines.extend(_target_lines(target))
        lines.append("")
    if len(targets) > 10:
        lines.append(f"… and {len(targets) - 10} more item(s).")
    return "\n".join(lines).rstrip()


def simple_confirm_text(targets: Sequence[Any]) -> str:
    count = len(targets)
    return (f"Permanently delete {count} Not Started item{'s' if count != 1 else ''}?\n\n"
            f"{format_delete_detail(targets)}\n\nThis cannot be undone.")


def simple_confirm(targets: Sequence[Any], on_confirm: Callable[[], Any], text: str = "") -> ConfirmDialog:
    """The desktop Yes/Cancel delete (``_confirm_delete_simple``); ``text`` = the shared plan's prompt."""
    return ConfirmDialog(title="Delete", body=text or simple_confirm_text(targets), confirm_label="Delete",
                         destructive=True, on_confirm=on_confirm)


class DeleteConfirmView:
    """Full-screen typed-keyword confirmation (``_confirm_delete_with_keyword``)."""

    def __init__(self, plan: Any, *, on_delete: Callable[[list], Any], on_close: Callable[[], Any],
                 mono: str = "monospace") -> None:
        self.plan = plan
        self.on_delete = on_delete
        self.on_close = on_close
        self.keywords = tuple(plan.keywords)
        self.deleting = False
        targets = list(plan.targets)
        self.checks: list[ft.Checkbox] = []
        rows: list[ft.Control] = []
        for index, target in enumerate(targets):
            box = ft.Checkbox(value=True, on_change=self._sync, tooltip="Uncheck to exclude this item from the "
                              "delete batch.", key=f"del-check-{index}")
            self.checks.append(box)
            lines = _target_lines(target)
            lines[0] = lines[0].replace("▾  ", "", 1)
            lines[1] = lines[1].strip()
            rows.append(ft.Row([
                box,
                ft.Text("\n".join(lines), font_family=mono, size=12, selectable=True, expand=True),
            ], vertical_alignment=ft.CrossAxisAlignment.START, key=f"del-target-{index}"))
        folders = sum(1 for t in targets if t.is_folder)
        files = len(targets) - folders
        bits = []
        if folders:
            bits.append(f"{folders} output folder{'s' if folders != 1 else ''}")
        if files:
            bits.append(f"{files} file{'s' if files != 1 else ''}")
        self.headline = ft.Text(f"⚠  Permanent delete — {len(targets)} item{'s' if len(targets) != 1 else ''}",
                                theme_style=ft.TextThemeStyle.TITLE_LARGE, color=ft.Colors.ERROR,
                                weight=ft.FontWeight.W_700, key="headline")
        self.subtitle = ft.Text("Uncheck anything you want to keep. Everything still checked below will be removed "
                                "from disk (" + " + ".join(bits) + "):", key="subtitle")
        pretty = " or ".join(k.capitalize() for k in self.keywords)
        self.field = ft.TextField(hint_text=self.keywords[0].capitalize(), on_change=self._sync, autofocus=False,
                                  dense=True, key="keyword")
        self.status = ft.Text("", theme_style=ft.TextThemeStyle.BODY_SMALL, key="status")
        self.progress = ft.ProgressBar(visible=False, key="progress")
        self.delete_button = ft.FilledButton(
            content="\U0001f5d1️  Delete", disabled=True, on_click=self._on_delete,
            style=ft.ButtonStyle(bgcolor=ft.Colors.ERROR, color=ft.Colors.ON_ERROR), key="delete")
        self.cancel_button = ft.TextButton(content="Cancel", on_click=lambda e: self.on_close(), key="cancel")
        body = ft.ListView([
            self.headline,
            self.subtitle,
            ft.Container(content=ft.Column(rows, spacing=6, tight=True), bgcolor=ft.Colors.SURFACE_CONTAINER_LOW,
                         border_radius=tokens.RADII["card"], padding=8),
            ft.Container(content=ft.Text(WARNING_TEXT, color="#b26a00"),
                         bgcolor=ft.Colors.with_opacity(0.12, "#ffb347"), border=ft.Border.all(1, "#ffb347"),
                         border_radius=4, padding=ft.Padding.symmetric(horizontal=10, vertical=6), key="warning"),
            ft.Text(f"Type {pretty} below to unlock the Delete button (case-insensitive):", key="instruction"),
            self.field,
            self.progress,
            self.status,
            ft.Row([self.cancel_button, self.delete_button], alignment=ft.MainAxisAlignment.END),
        ], spacing=tokens.SPACING["md"], padding=16, expand=True)
        # A route of its own: Flet resolves ``view_pop`` to the first View with the popped route,
        # so reusing a screen's route ("/library") made Android back pop that screen instead.
        self.view = ft.View(
            route=DELETE_VIEW_ROUTE,
            appbar=ft.AppBar(
                title=ft.Text("Delete — confirmation required"),
                leading=ft.IconButton(icon=ft.Icons.CLOSE, tooltip="Cancel", size_constraints=HIT_TARGET,
                                      on_click=lambda e: self.on_close()),
                bgcolor=ft.Colors.SURFACE,
            ),
            controls=[ft.SafeArea(content=body, expand=True)],
            padding=0,
        )

    def selected_paths(self) -> list:
        return [t.path for t, box in zip(self.plan.targets, self.checks) if box.value]

    @property
    def can_delete(self) -> bool:
        return not self.deleting and bool(self.selected_paths()) and keyword_ok(self.field.value, self.keywords)

    def _sync(self, e: Any = None) -> None:
        self.delete_button.disabled = not self.can_delete
        try:
            self.delete_button.update()
        except Exception:
            pass

    def set_progress(self, done: int, total: int, label: str = "") -> None:
        self.progress.visible = True
        self.progress.value = (done / total) if total else None
        self.status.value = f"Deleting selected items... ({done}/{total})" + (f"\n{label}" if label else "")
        for control in (self.progress, self.status):
            try:
                control.update()
            except Exception:
                pass

    def unlock(self, message: str = "") -> None:
        """A failed delete: the targets and Cancel are usable again (nothing was confirmed gone)."""
        self.deleting = False
        self.cancel_button.disabled = False
        for box in self.checks:
            box.disabled = False
        self.progress.visible = False
        self.status.value = message
        self.delete_button.disabled = not self.can_delete
        for control in (self.cancel_button, *self.checks, self.progress, self.status, self.delete_button):
            try:
                control.update()
            except Exception:
                pass

    async def _on_delete(self, e: Any = None) -> Any:
        if not self.can_delete:
            return None
        self.deleting = True
        self.delete_button.disabled = True
        self.cancel_button.disabled = True
        for box in self.checks:
            box.disabled = True
        self.set_progress(0, len(self.selected_paths()))
        try:
            result = self.on_delete(self.selected_paths())
            if hasattr(result, "__await__"):
                result = await result
            return result
        finally:
            self.deleting = False


class DeleteFlow:
    """Plan -> (simple confirm | keyword view) -> ``execute_delete`` on the io pool -> summary + rescan.

    Used by the Library selection bar, the single-card sheet and the Book page ⋯ menu.
    ``on_done(report)`` runs after a delete (the Library leaves selection mode, the Book
    page returns to the Library).
    """

    def __init__(self, ctx: Any, *, on_done: Optional[Callable[[Any], Any]] = None) -> None:
        self.ctx = ctx
        self.on_done = on_done
        self.view: Optional[DeleteConfirmView] = None
        self.dialog: Optional[ConfirmDialog] = None
        self.links_sheet: Any = None
        self.delete_remote_links = False  # U10: also delete the books' share-link uploads on their services
        self.plan: Any = None
        self.report: Any = None

    async def start(self, books: Sequence[Any]) -> Any:
        from glossarion_mobile.services.library import CoreMissing

        service = self.ctx.service
        try:
            plan = await self.ctx.io(service.plan_delete_blocking, list(books))
        except CoreMissing as exc:
            self.ctx.say(f"Delete is not available in this build ({exc.name})")
            return None
        self.plan = plan
        if not plan.targets:
            if plan.unregister:
                # Outside the safe roots: the card disappears, the file stays (silent, desktop rule).
                self.report = await self.ctx.io(service.execute_delete_blocking, plan, [])
                await self._finish(say=False)
                return plan
            self.ctx.say("Nothing to delete — none of the selected cards point at a file or folder on disk.")
            return plan
        counter = getattr(service, "deletable_links_blocking", None)
        links = await self.ctx.io(counter, list(books)) if callable(counter) else 0
        if links:
            return self.ask_links(int(links))
        return self.confirm()

    def ask_links(self, count: int) -> Any:
        """U10: the books have share-link uploads Glossarion can delete on their services; once the books are
        gone it no longer can. Delete them too, or keep them online until they expire (then the delete)."""
        from glossarion_mobile.ui.components.action_sheet import ActionItem, ActionSheet

        plural = count != 1

        def choose(delete_remote: bool) -> Any:
            self.delete_remote_links = delete_remote
            return self.confirm()

        sheet = ActionSheet([
            ActionItem(f"Delete {'these uploads' if plural else 'this upload'} too", lambda: choose(True),
                       icon="DELETE_SWEEP", destructive=True, key="delete-links-remote"),
            ActionItem(f"Keep {'them' if plural else 'it'} online until the link expires", lambda: choose(False),
                       icon="LINK", key="delete-links-keep"),
        ], title=f"{count} file{'s' if plural else ''} of {'these books' if len(self.plan.targets) != 1 else 'this book'} "
                 f"{'are' if plural else 'is'} shared via a link",
            subtitle="After the delete Glossarion can no longer remove the upload from the service.",
            tablet=bool(getattr(self.ctx, "tablet", False)))
        self.links_sheet = sheet
        self.ctx.show(sheet)
        return sheet

    def confirm(self) -> Any:
        """The desktop confirmation: the simple Yes / Cancel, or the keyword view."""
        plan = self.plan
        if not plan.needs_keyword:
            self.dialog = simple_confirm(plan.targets, lambda: self.run(None), plan.simple_prompt)
            self.ctx.show(self.dialog)
            return self.dialog
        self.view = DeleteConfirmView(plan, on_delete=self.run, on_close=self.close_view,
                                      mono=getattr(self.ctx, "mono", "monospace"))
        if self.ctx.push_overlay is not None:
            self.ctx.push_overlay(self.view.view)
        return self.view

    def close_view(self) -> None:
        """Remove the keyword view -- unless it is already gone: Android back pops the overlay
        itself (``AppShell.pop_view``), and popping again would pop the Library / Book page."""
        view, self.view = self.view, None
        if view is None or self.ctx.pop_overlay is None:
            return
        shell = getattr(self.ctx, "shell", None)
        overlays = getattr(shell, "overlays", None) if shell is not None else None
        if overlays is not None and view.view not in overlays:
            return
        self.ctx.pop_overlay()

    async def run(self, selected: Optional[Sequence[str]]) -> Any:
        service = self.ctx.service
        view = self.view

        def progress(done: int, total: int, label: str = "") -> None:
            dispatcher = self.ctx.dispatcher
            if view is not None and dispatcher is not None and getattr(dispatcher, "bound", False):
                dispatcher.post(view.set_progress, done, total, label)

        try:
            if self.delete_remote_links:
                execute = functools.partial(service.execute_delete_blocking, delete_remote_links=True)
            else:
                execute = service.execute_delete_blocking
            self.report = await self.ctx.io(execute, self.plan, selected, progress)
        except Exception as exc:
            self.ctx.say(f"Delete failed: {exc}")
            if view is not None:
                view.unlock(f"Delete failed: {exc}")
            return None
        self.close_view()
        self.ctx.haptic("heavy_impact")
        await self._finish(say=True)
        return self.report

    async def _finish(self, *, say: bool) -> None:
        if say and self.report is not None:
            self.ctx.say(self.report.summary)
        if self.on_done is not None:
            result = self.on_done(self.report)
            if hasattr(result, "__await__"):
                await result
        await self.ctx.service.refresh(reason="delete")
