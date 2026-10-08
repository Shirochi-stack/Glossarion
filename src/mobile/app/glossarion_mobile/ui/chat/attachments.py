"""Attachments manager (UI_SPEC §2.17, ``/chat/<cid>/attachments``).

The desktop Direct Text "Attachments" manager: one card per ``Attachments/<stem>`` workspace
of the chat (``ChatStore.conversation_attachment_folders``). On mobile a finished chat book moves
into the Library by itself (``integration.ChatFeature.auto_migrate``: the desktop Migrate,
``ChatStore.migrate_attachment``), so there is no Migrate button: the workspaces listed here are
the ones still waiting - a job is running on them, their run can still be resumed (Resume / Retry
failed), or a different Library book already has their name. For that last case the card ⋯ offers
the desktop "Attachment folder already exists" dialog (Merge and replace / Cancel,
``show_merge_dialog``), which the auto-migrate snackbar opens too.

Delete workspace (and that merge) is blocked while a job may be writing the workspace: this chat's
run, or any active / queued job whose folder is the workspace or inside it (``job_writes_into``:
the job card's Compile, ＋ › Retranslate chapters, ...). Every move runs while no job owns the
process environment (``migrate_when_idle``: the shared Migrate reads the live ``OUTPUT_DIRECTORY``,
which a running job points at its own temporary run root).

Card ⋯: Open in Reader · Progress · Share output · [Merge into Library…] · Delete workspace (confirm).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Optional

import flet as ft

from glossarion_mobile.ui.chat.job_binding import TERMINAL_STATES, state_name
from glossarion_mobile.ui.components.action_sheet import ActionItem, ActionSheet
from glossarion_mobile.ui.components.dialogs import ConfirmDialog
from glossarion_mobile.ui.components.empty_state import EmptyState
from glossarion_mobile.ui.components.reason_chip import ReasonChip
from glossarion_mobile.ui.screens.base import Screen
from glossarion_mobile.ui.theme import HIT_TARGET

__all__ = [
    "AttachmentsScreen", "BUSY_REASON", "EMPTY_TEXT", "INTRO", "JOBS_RUNNING_TEXT", "MERGE_ACTION", "MERGE_TITLE",
    "attachment_source", "job_paths", "job_writes_into", "jobs_lock", "merge_body", "migrate_when_idle",
    "migrated_target", "show_merge_dialog", "workspace_busy",
]

log = logging.getLogger("glossarion.chat.attachments")

INTRO = ("Workspaces waiting to be added to the Library (running, resumable, or a name already in the "
         "Library)")
EMPTY_TEXT = "No attachment workspaces remain in this conversation."
MERGE_TITLE = "Attachment folder already exists"
MERGE_ACTION = "Merge into Library…"
BUSY_REASON = "A job is running on this workspace; it may be writing it"
JOBS_RUNNING_TEXT = "Wait for the running job to finish, then try again"


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


def workspace_busy(runs: Any, jobs: Any, cid: Any, folder: str) -> bool:
    """The Attachments guard (UI_SPEC §2.17): chat ``cid`` has a live run, or an active / queued job
    (``JobsAdapter.pending``) has the workspace or a folder inside it (the job card's Compile,
    ＋ Retranslate chapters). Ended snapshots never count."""
    if runs is not None and runs.live_run(cid) is not None:
        return True
    pending = getattr(jobs, "pending", None) if jobs is not None else None
    try:
        snapshots = pending() if callable(pending) else []
    except Exception:
        snapshots = []
    return any(job_writes_into(snap, folder) for snap in snapshots if state_name(snap) not in TERMINAL_STATES)


def attachment_source(messages: Any, folder: str) -> str:
    """The attachment a workspace came from: the newest ``user_file`` turn whose file has the
    workspace's name as its stem ('' when that turn is gone)."""
    stem = os.path.basename(os.path.normpath(str(folder or ""))).lower()
    for message in reversed(list(messages or [])):
        if message and str(message[0]) == "user_file" and len(message) > 2:
            if os.path.splitext(os.path.basename(str(message[2] or "")))[0].lower() == stem:
                return str(message[2] or "")
    return ""


def merge_body(target: str) -> str:
    """The desktop ``_confirm_attachment_merge`` question."""
    return (f"The destination folder already exists:\n{target}\n\nMerge this attachment into it and "
            "replace files with the same names?")


def show_merge_dialog(page: Any, target: str, on_confirm: Callable[[], Any]) -> ConfirmDialog:
    """The desktop "Attachment folder already exists" dialog (Merge and replace / Cancel); it closes
    itself (``components.dialogs.close_dialog``) once ``on_confirm`` has run."""
    dialog = ConfirmDialog(title=MERGE_TITLE, body=merge_body(target), confirm_label="Merge and replace",
                           cancel_label="Cancel", destructive=True, on_confirm=on_confirm)
    if page is not None:
        dialog.show(page)
    return dialog


def migrated_target(result: Any) -> str:
    """The folder the shared Migrate moved a workspace to (its "Attachment migrated" notice)."""
    for notice in (result or {}).get("notices") or ():
        if isinstance(notice, dict) and notice.get("title") == "Attachment migrated":
            return str(notice.get("text") or "").split(":\n", 1)[-1].strip()
    return ""


def jobs_lock() -> Any:
    """``job_runner.JOB_LOCK``, held by every job for its whole run (None when the backend is absent)."""
    try:
        import job_runner  # shared (U3)

        return job_runner.JOB_LOCK
    except Exception:
        return None


def migrate_when_idle(chats: Any, cid: Any, folder: str, confirm_merge: Optional[Callable[[str], bool]] = None,
                      *, lock: Any = None) -> Optional[dict]:
    """Blocking: the desktop Migrate (``ChatStoreAdapter.migrate_attachment``) while no job runs.

    The shared Migrate picks its destination from the live ``OUTPUT_DIRECTORY``; a running job points
    it at its own temporary run root (deleted when the run ends), so the move would land there. The
    jobs' process lock is taken without waiting and held for the whole move; None when a job holds it.
    """
    lock = jobs_lock() if lock is None else lock
    if lock is not None and not lock.acquire(blocking=False):
        return None
    try:
        return chats.migrate_attachment(cid, folder, confirm_merge)
    finally:
        if lock is not None:
            lock.release()


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
        on_migrated: Optional[Callable[[str, str], Any]] = None,  # (target folder, attachment source) after a merge
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
        return attachment_source(self.chats.messages(self.cid), folder)

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
        rows: list = [
            ft.Row([ft.Icon(ft.Icons.FOLDER_OUTLINED, color=ft.Colors.PRIMARY),
                    ft.Text(name, theme_style=ft.TextThemeStyle.TITLE_SMALL, expand=True, max_lines=2,
                            overflow=ft.TextOverflow.ELLIPSIS),
                    ft.IconButton(icon=ft.Icons.MORE_VERT, tooltip="More", size_constraints=HIT_TARGET,
                                  on_click=lambda e, f=folder: self.more(f), key=f"more-{name}")],
                   vertical_alignment=ft.CrossAxisAlignment.CENTER),
            ft.Text(self.summaries.get(folder, ""), theme_style=ft.TextThemeStyle.LABEL_SMALL,
                    color=ft.Colors.ON_SURFACE_VARIANT),
        ]
        if self.busy(folder):
            rows.append(ft.Row([ReasonChip(reason=BUSY_REASON)], wrap=True, spacing=8))
        card = ft.Container(
            content=ft.Column(
                rows,
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

    def collides(self, folder: str) -> bool:
        """A Library folder already has this workspace's name (only the merge dialog can move it)."""
        target = self.chats.migration_target(self.cid, folder)
        return bool(target) and os.path.exists(target)

    def migrate(self, folder: str) -> Optional[ConfirmDialog]:
        """The desktop Migrate of one workspace (the collision dialog first when the destination exists):
        ⋯ › Merge into Library…, and the job card's action until it opens the Library book instead."""
        if self.busy(folder):
            self.notify(BUSY_REASON)
            return None
        target = self.chats.migration_target(self.cid, folder)
        if target and os.path.exists(target):
            return show_merge_dialog(self.page, target, lambda f=folder: self.run_migrate(f, merge=True))
        self._spawn(self.run_migrate(folder, merge=False))
        return None

    async def run_migrate(self, folder: str, merge: bool = False) -> dict:
        if self.busy(folder):  # a job started while the collision question was open
            self.notify(BUSY_REASON)
            return {"ok": False, "notices": []}
        source = self.source_for(folder)
        result = await self.run_io(migrate_when_idle, self.chats, self.cid, folder,
                                   (lambda _t: True) if merge else None)
        if result is None:  # a job owns the process environment (its OUTPUT_DIRECTORY is its run root)
            self.notify(JOBS_RUNNING_TEXT)
            return {"ok": False, "notices": []}
        notices = list(result.get("notices") or [])
        if result.get("ok"):
            target = migrated_target(result) or self.chats.migration_target(self.cid, folder) or ""
            if self.on_migrated is not None:
                self.on_migrated(target, source)
            else:
                self.notify("Added to the Library")
        elif notices:
            last = notices[-1]
            self.notify(f"{last.get('title')}: {last.get('text')}")
        await self.reload()
        return result

    def more(self, folder: str) -> Optional[ActionSheet]:
        items = self.more_items(folder)
        if self.page is None:
            return None
        sheet = ActionSheet(items, title=os.path.basename(os.path.normpath(folder)), tablet=self.tablet)
        sheet.show(self.page)
        return sheet

    def more_items(self, folder: str) -> list:
        """Card ⋯: Open in Reader · Progress · Share output · [Merge into Library… on a name clash] · Delete."""
        source = self.source_for(folder)
        items = [
            ActionItem("Open in Reader", (lambda: self.open_reader(folder, source)) if self.open_reader else None,
                       icon="AUTO_STORIES", disabled_reason=None if self.open_reader else "The Reader is not available"),
            ActionItem("Progress", (lambda: self.open_progress(folder, source)) if self.open_progress else None,
                       icon="CHECKLIST", disabled_reason=None if self.open_progress else "Not available"),
            ActionItem("Share output", (lambda: self.share_output(folder)) if self.share_output else None,
                       icon="IOS_SHARE", disabled_reason=None if self.share_output else "Not available"),
        ]
        busy = BUSY_REASON if self.busy(folder) else None
        if self.collides(folder):
            items.append(ActionItem(MERGE_ACTION, lambda: self.migrate(folder), icon="CALL_MERGE", disabled_reason=busy))
        items.append(ActionItem("Delete workspace", lambda: self.confirm_delete(folder), icon="DELETE_OUTLINE",
                                destructive=True, disabled_reason=busy))
        return items

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
