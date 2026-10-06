"""Attachments manager and Migrate (UI_SPEC §2.17, ``/chat/<cid>/attachments``).

The desktop Direct Text "Attachments" manager: one card per ``Attachments/<stem>`` workspace
of the chat (``ChatStore.conversation_attachment_folders``) with **Migrate** - the desktop
Migrate (``ChatStore.migrate_attachment``): the workspace moves beside the Direct Text folder
(or into the configured output override), only the newest top-level EPUB/PDF is kept and the
stored response paths are rewritten. A name collision asks first with the desktop dialog
"Attachment folder already exists" (Merge and replace / Cancel). Migrate and Delete workspace
are blocked while a job may be writing the workspace: this chat's run, or any active / queued
job whose folder is the workspace or inside it (``job_writes_into``: the job card's Compile,
＋ › Retranslate chapters, ...).

Card ⋯: Open in Reader · Progress · Share output · Delete workspace (confirm). Long-press on
Migrate shows the destination.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Optional

import flet as ft

from glossarion_mobile.ui.components.action_sheet import ActionItem, ActionSheet
from glossarion_mobile.ui.components.dialogs import ConfirmDialog
from glossarion_mobile.ui.components.empty_state import EmptyState
from glossarion_mobile.ui.components.reason_chip import ReasonChip
from glossarion_mobile.ui.screens.base import Screen
from glossarion_mobile.ui.theme import HIT_TARGET

__all__ = ["AttachmentsScreen", "BUSY_REASON", "EMPTY_TEXT", "INTRO", "MERGE_TITLE", "job_paths", "job_writes_into"]

log = logging.getLogger("glossarion.chat.attachments")

INTRO = ("Saved attachment workspaces for this conversation. Migrating one moves its complete output tree beside "
         "the Direct Text folder, or into the configured output override.")
EMPTY_TEXT = "No attachment workspaces remain in this conversation."
MERGE_TITLE = "Attachment folder already exists"
BUSY_REASON = "A job is running on this workspace; it may be writing it"


def job_paths(snapshot: Any) -> list:
    """The absolute paths a JobService snapshot works on: its inputs, the path-valued params
    (``folder`` / ``folders`` / ``output_dir`` ...) and the output folders it reported."""
    spec = getattr(snapshot, "spec", None)
    values: list = list(getattr(spec, "inputs", None) or ())
    params = getattr(spec, "params", None) or {}
    for value in params.values() if hasattr(params, "values") else ():
        values.extend(value if isinstance(value, (list, tuple)) else [value])
    values.append(getattr(snapshot, "output_dir", None))
    values.extend((getattr(snapshot, "output_dirs", None) or {}).values())
    return [os.path.abspath(v) for v in values if isinstance(v, str) and v and os.path.isabs(v)]


def job_writes_into(snapshot: Any, folder: str) -> bool:
    """Whether a job's folder is ``folder`` or inside it (it may be writing that workspace)."""
    root = os.path.normcase(os.path.abspath(str(folder or "")))
    if not folder:
        return False
    for path in job_paths(snapshot):
        candidate = os.path.normcase(path)
        try:
            if os.path.commonpath([candidate, root]) == root:
                return True
        except ValueError:  # another drive
            continue
    return False


class AttachmentsScreen(Screen):
    def __init__(
        self,
        match: Any,
        *,
        chats: Any,
        cid: str,
        run_io: Callable[..., Any],
        notify: Optional[Callable[..., Any]] = None,
        page: Any = None,
        busy: Callable[[str], bool] = lambda folder: False,  # a job may be writing this workspace
        open_reader: Optional[Callable[[str, str], Any]] = None,
        open_progress: Optional[Callable[[str, str], Any]] = None,
        share_output: Optional[Callable[[str], Any]] = None,
        on_migrated: Optional[Callable[[str, str], Any]] = None,  # (target folder, attachment source)
        spawn: Optional[Callable[[Any], Any]] = None,
        tablet: bool = False,
    ) -> None:
        super().__init__(match)
        self.chats = chats
        self.cid = str(cid)
        self.run_io = run_io
        self.notify = notify or (lambda *a, **k: None)
        self.page = page
        self.busy = busy
        self.open_reader = open_reader
        self.open_progress = open_progress
        self.share_output = share_output
        self.on_migrated = on_migrated
        self.spawn = spawn
        self.tablet = tablet
        session = chats.session(self.cid) or {}
        self.title = f"Attachments — {session.get('title') or 'Chat'}"
        self.list = ft.Column([], spacing=8, tight=True, key="attachments-list")
        self.cards: dict = {}
        self.folders: list = []
        self.summaries: dict = {}

    # ---- data --------------------------------------------------------------------------------

    def _load(self) -> list:
        folders = list(self.chats.attachment_folders(self.cid) or [])
        from glossarion_mobile.ui.chat.chat_ops import workspace_summary

        self.summaries = {folder: workspace_summary(folder) for folder in folders}
        return folders

    def source_for(self, folder: str) -> str:
        """The attachment the workspace came from (the ``user_file`` turn with the same stem)."""
        stem = os.path.basename(os.path.normpath(folder)).lower()
        for message in reversed(self.chats.messages(self.cid)):
            if message and str(message[0]) == "user_file" and len(message) > 2:
                if os.path.splitext(os.path.basename(str(message[2] or "")))[0].lower() == stem:
                    return str(message[2] or "")
        return ""

    # ---- building ------------------------------------------------------------------------------

    def build_body(self) -> ft.Control:
        # The workspaces are listed (and summarised from their progress files) on the io pool:
        # the first paint shows a progress ring, ``did_show`` loads.
        self.list.controls = [ft.ProgressRing(width=24, height=24, key="attachments-loading")]
        return ft.ListView(
            controls=[ft.Text(INTRO, theme_style=ft.TextThemeStyle.BODY_MEDIUM, color=ft.Colors.ON_SURFACE_VARIANT),
                      self.list],
            spacing=12,
            padding=ft.Padding.all(16),
            expand=True,
        )

    def did_show(self) -> None:
        self._spawn(self.reload())

    def reload_sync(self) -> None:
        """Blocking: list and render the workspaces (tests and callers already off the loop)."""
        try:
            self.folders = self._load()
        except Exception:
            log.exception("listing the attachment workspaces failed")
            self.folders = []
        self._render()

    async def reload(self) -> None:
        try:
            self.folders = await self.run_io(self._load)
        except Exception:
            log.exception("listing the attachment workspaces failed")
            self.folders = []
        self._render()
        try:
            self.list.update()
        except Exception:
            pass

    def _render(self) -> None:
        self.cards = {}
        if not self.folders:
            self.list.controls = [EmptyState(icon="ATTACH_FILE", title="No attachments", body=EMPTY_TEXT,
                                             key="attachments-empty")]
            return
        self.list.controls = [self._card(folder) for folder in self.folders]

    def _card(self, folder: str) -> ft.Control:
        name = os.path.basename(os.path.normpath(folder))
        busy = bool(self.busy(folder))
        migrate: ft.Control = ft.FilledTonalButton(
            content="Migrate", icon=ft.Icons.DRIVE_FILE_MOVE, disabled=busy,
            on_click=lambda e, f=folder: self.migrate(f), key=f"migrate-{name}",
        )
        migrate = ft.GestureDetector(content=migrate, on_long_press_start=lambda e, f=folder: self.show_destination(f))
        row: list = [migrate]
        if busy:
            row.append(ReasonChip(reason=BUSY_REASON))
        card = ft.Container(
            content=ft.Column(
                [
                    ft.Row([ft.Icon(ft.Icons.FOLDER_OUTLINED, color=ft.Colors.PRIMARY),
                            ft.Text(name, theme_style=ft.TextThemeStyle.TITLE_SMALL, expand=True, max_lines=2,
                                    overflow=ft.TextOverflow.ELLIPSIS),
                            ft.IconButton(icon=ft.Icons.MORE_VERT, tooltip="More", size_constraints=HIT_TARGET,
                                          on_click=lambda e, f=folder: self.more(f), key=f"more-{name}")],
                           vertical_alignment=ft.CrossAxisAlignment.CENTER),
                    ft.Text(self.summaries.get(folder, ""), theme_style=ft.TextThemeStyle.LABEL_SMALL,
                            color=ft.Colors.ON_SURFACE_VARIANT),
                    ft.Row(row, wrap=True, spacing=8),
                ],
                spacing=6,
                tight=True,
            ),
            bgcolor=ft.Colors.SURFACE_CONTAINER,
            border_radius=16,
            padding=ft.Padding.all(12),
            key=f"workspace-{name}",
        )
        self.cards[folder] = card
        return card

    # ---- actions ---------------------------------------------------------------------------------

    def _spawn(self, coro: Any) -> Any:
        if self.spawn is not None:
            return self.spawn(coro)
        import asyncio

        try:
            return asyncio.ensure_future(coro)
        except RuntimeError:
            coro.close()
            return None

    def show_destination(self, folder: str) -> None:
        target = self.chats.migration_target(self.cid, folder)
        self.notify(f"Migrate moves it to: {target}" if target else "No migration destination is configured")

    def migrate(self, folder: str) -> Optional[ConfirmDialog]:
        """Migrate (the collision dialog first when the destination exists)."""
        if self.busy(folder):
            self.notify(BUSY_REASON)
            return None
        target = self.chats.migration_target(self.cid, folder)
        if target and os.path.exists(target):
            dialog = ConfirmDialog(
                title=MERGE_TITLE,
                body=(f"The destination folder already exists:\n{target}\n\nMerge this attachment into it and "
                      "replace files with the same names?"),
                confirm_label="Merge and replace", cancel_label="Cancel", destructive=True,
                on_confirm=lambda f=folder: self.run_migrate(f, merge=True),
            )
            if self.page is not None:
                dialog.show(self.page)
            return dialog
        self._spawn(self.run_migrate(folder, merge=False))
        return None

    async def run_migrate(self, folder: str, merge: bool = False) -> dict:
        if self.busy(folder):  # a job started while the collision question was open
            self.notify(BUSY_REASON)
            return {"ok": False, "notices": []}
        source = self.source_for(folder)
        result = await self.run_io(self.chats.migrate_attachment, self.cid, folder, (lambda _t: True) if merge else None)
        notices = list(result.get("notices") or [])
        if result.get("ok"):
            target = self.chats.migration_target(self.cid, folder) or ""
            moved = next((n.get("text", "").split(":\n", 1)[-1] for n in notices if n.get("title") == "Attachment migrated"),
                         "")
            target = moved.strip() or target
            if self.on_migrated is not None:
                self.on_migrated(target, source)
            else:
                self.notify("Attachment migrated")
        elif notices:
            last = notices[-1]
            self.notify(f"{last.get('title')}: {last.get('text')}")
        await self.reload()
        return result

    def more(self, folder: str) -> Optional[ActionSheet]:
        source = self.source_for(folder)
        items = [
            ActionItem("Open in Reader", (lambda: self.open_reader(folder, source)) if self.open_reader else None,
                       icon="AUTO_STORIES", disabled_reason=None if self.open_reader else "The Reader is not available"),
            ActionItem("Progress", (lambda: self.open_progress(folder, source)) if self.open_progress else None,
                       icon="CHECKLIST", disabled_reason=None if self.open_progress else "Not available"),
            ActionItem("Share output", (lambda: self.share_output(folder)) if self.share_output else None,
                       icon="IOS_SHARE", disabled_reason=None if self.share_output else "Not available"),
            ActionItem("Delete workspace", lambda: self.confirm_delete(folder), icon="DELETE_OUTLINE", destructive=True,
                       disabled_reason=BUSY_REASON if self.busy(folder) else None),
        ]
        if self.page is None:
            return None
        sheet = ActionSheet(items, title=os.path.basename(os.path.normpath(folder)), tablet=self.tablet)
        sheet.show(self.page)
        return sheet

    def confirm_delete(self, folder: str) -> Optional[ConfirmDialog]:
        async def go() -> None:
            if self.busy(folder):  # a job started while the question was open
                self.notify(BUSY_REASON)
                return
            ok, error = await self.run_io(self.chats.delete_attachment_workspace, self.cid, folder)
            self.notify("Attachment workspace deleted" if ok else error)
            await self.reload()

        dialog = ConfirmDialog(
            title="Delete workspace?",
            body=(f"Permanently delete this attachment workspace and every file in it?\n\n{folder}\n\n"
                  "This cannot be undone."),
            confirm_label="Delete", cancel_label="Cancel", destructive=True, on_confirm=go,
        )
        if self.page is not None:
            dialog.show(self.page)
        return dialog
