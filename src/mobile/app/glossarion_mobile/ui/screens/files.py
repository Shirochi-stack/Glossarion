"""Minimal output file browser (``/tools/files/<root>[/<fid>]``, UI_SPEC §4.10).

Roots (chips): Output (``OUTPUT_DIRECTORY``), Library, Inbox and Chat
workspaces. A sub-folder is addressed by an opaque ``fid`` from the Prefs
FileRef registry, never by a path in the route. Every listed path is checked
against the roots (``realpath``), so a stale or forged ``fid`` cannot leave
them. Rows show icon, name and "size · date"; a folder opens the next level,
a file opens the ExportSheet (Share · Save to… · Save to Downloads / Show in
Files) plus the file tools: **Open with…** (Text editor for text files,
Reader for books when the Reader is installed, the MediaViewer for images,
video and audio, another app through Share),
**Rename** (same folder, name checks) and **Delete** (confirmation; only
files inside a root, never a root itself). Listing runs on a worker thread.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

import flet as ft

from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.components.action_sheet import ActionItem, ActionSheet
from glossarion_mobile.ui.components.empty_state import EmptyState
from glossarion_mobile.ui.router import RouteMatch
from glossarion_mobile.ui.screens.base import Screen
from glossarion_mobile.ui.screens.job_detail import export_sheet
from glossarion_mobile.ui.theme import icon_data

__all__ = ["ROOT_LABELS", "FileBrowserScreen", "FileEntry", "IMAGE_VIEW_EXTENSIONS", "delete_entry", "describe_size",
           "list_folder", "media_kind", "rename_entry", "rename_problem", "resolve_location"]

ROOT_LABELS = (("output", "Output"), ("library", "Library"), ("inbox", "Inbox"), ("chats", "Chat workspaces"))
MAX_ENTRIES = 2000
_ICONS = {
    ".epub": "MENU_BOOK", ".pdf": "PICTURE_AS_PDF", ".txt": "DESCRIPTION", ".md": "DESCRIPTION",
    ".html": "HTML", ".xhtml": "HTML", ".htm": "HTML", ".json": "DATA_OBJECT", ".csv": "TABLE_CHART",
    ".png": "IMAGE", ".jpg": "IMAGE", ".jpeg": "IMAGE", ".webp": "IMAGE", ".gif": "IMAGE",
    ".zip": "FOLDER_ZIP", ".cbz": "FOLDER_ZIP", ".mp3": "AUDIOTRACK", ".wav": "AUDIOTRACK", ".mp4": "MOVIE",
    ".log": "TERMINAL",
}
#: Images the MediaViewer's ``Image`` renders on every platform.
IMAGE_VIEW_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"})


def media_kind(path: str) -> Optional[str]:
    """``image`` / ``video`` / ``audio`` for a file the MediaViewer shows, else None (video and audio:
    the shared ``ChatStoreMixin`` generated-media extensions)."""
    ext = os.path.splitext(str(path or ""))[1].lower()
    if ext in IMAGE_VIEW_EXTENSIONS:
        return "image"
    try:
        from direct_text_store import ChatStoreMixin  # shared (U3)

        video, audio = ChatStoreMixin._VIDEO_OUTPUT_EXTENSIONS, ChatStoreMixin._AUDIO_OUTPUT_EXTENSIONS
    except Exception:
        video, audio = {".mp4", ".mov", ".webm", ".mkv", ".m4v"}, {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac"}
    if ext in video:
        return "video"
    if ext in audio:
        return "audio"
    return None


@dataclass(frozen=True)
class FileEntry:
    name: str
    path: str
    is_dir: bool
    size: int
    mtime: float

    @property
    def icon(self) -> str:
        if self.is_dir:
            return "FOLDER"
        return _ICONS.get(os.path.splitext(self.name)[1].lower(), "INSERT_DRIVE_FILE")


def describe_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{int(value)} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{size} B"


def _inside(path: str, root: str) -> bool:
    try:
        real, base = os.path.realpath(path), os.path.realpath(root)
        return os.path.commonpath([real, base]) == base
    except (OSError, ValueError):
        return False


def resolve_location(roots: Mapping[str, str], root: str, fid: Optional[str],
                     resolve_ref: Optional[Callable[[str], Optional[str]]]) -> tuple:
    """(root path, folder path, error) for a route; the folder must stay inside the root."""
    base = roots.get(root)
    if not base:
        return None, None, "This location is not available"
    if not fid:
        return base, base, None
    folder = resolve_ref(fid) if resolve_ref is not None else None
    if not folder:
        return base, None, "This folder link has expired"
    if not _inside(folder, base):
        return base, None, "This folder is outside the app's storage"
    return base, folder, None


def list_folder(folder: str) -> list[FileEntry]:
    """Folders first, then files, by name (hidden entries and partial copies skipped)."""
    entries: list[FileEntry] = []
    with os.scandir(folder) as it:
        for item in it:
            if item.name.startswith(".") or item.name.endswith(".part"):
                continue
            try:
                stat = item.stat()
                is_dir = item.is_dir()
            except OSError:
                continue
            entries.append(FileEntry(item.name, item.path, is_dir, 0 if is_dir else stat.st_size, stat.st_mtime))
            if len(entries) >= MAX_ENTRIES:
                break
    entries.sort(key=lambda e: (not e.is_dir, e.name.casefold()))
    return entries


_BAD_NAME_CHARS = '<>:"/\\|?*'


def rename_problem(path: str, new_name: str) -> Optional[str]:
    """Why ``path`` cannot be renamed to ``new_name`` (None: it can)."""
    name = str(new_name or "").strip()
    if not name or name in (".", ".."):
        return "Enter a file name"
    if any(ch in name for ch in _BAD_NAME_CHARS) or any(ord(ch) < 32 for ch in name):
        return 'A file name cannot contain < > : " / \\ | ? *'
    if name.startswith("."):
        return "A file name cannot start with a dot"
    if name == os.path.basename(path):
        return "That is already the file's name"
    target = os.path.join(os.path.dirname(path), name)
    if os.path.exists(target) and os.path.normcase(os.path.abspath(target)) != os.path.normcase(os.path.abspath(path)):
        return "A file with that name already exists"
    return None


def rename_entry(path: str, new_name: str, roots: Mapping[str, str]) -> str:
    """Blocking: rename a file inside its folder (inside a root); returns the new path."""
    problem = rename_problem(path, new_name)
    if problem:
        raise ValueError(problem)
    if not any(_inside(path, root) and os.path.realpath(path) != os.path.realpath(root)
               for root in roots.values() if root):
        raise ValueError("This file is outside the app's storage")
    target = os.path.join(os.path.dirname(path), str(new_name).strip())
    os.rename(path, target)
    return target


def delete_entry(path: str, roots: Mapping[str, str]) -> None:
    """Blocking: delete one file inside a root (folders and the roots themselves are refused)."""
    if os.path.isdir(path):
        raise ValueError("Folders are not deleted from the file browser")
    if not any(_inside(path, root) and os.path.realpath(path) != os.path.realpath(root)
               for root in roots.values() if root):
        raise ValueError("This file is outside the app's storage")
    os.remove(path)


class FileBrowserScreen(Screen):
    title = "Files"

    def __init__(
        self,
        match: Optional[RouteMatch],
        *,
        roots: Mapping[str, str],
        files: Any = None,
        page: Any = None,
        navigate: Optional[Callable[..., Any]] = None,
        notify: Optional[Callable[..., Any]] = None,
        file_ref: Optional[Callable[[str], str]] = None,
        resolve_ref: Optional[Callable[[str], Optional[str]]] = None,
        run_io: Optional[Callable[..., Any]] = None,
        tablet: bool = False,
        open_reader: Optional[Callable[..., Any]] = None,
        push_overlay: Optional[Callable[[Any], Any]] = None,
        pop_overlay: Optional[Callable[[Any], Any]] = None,
    ) -> None:
        super().__init__(match)
        self.roots = dict(roots)
        self.files = files
        self.open_reader = open_reader  # (path=...) -> the Reader on a book file, when installed
        self.push_overlay = push_overlay  # the shell's overlay stack (the MediaViewer is a full-screen View)
        self.pop_overlay = pop_overlay
        self.media_viewer: Any = None
        self.last_sheet: Any = None
        self.page = page
        self.navigate = navigate
        self.notify = notify
        self.file_ref = file_ref
        self.resolve_ref = resolve_ref
        self.run_io = run_io
        self.tablet = tablet
        params = match.params if match is not None else {}
        self.root = params.get("root") or "output"
        self.fid = params.get("fid")
        self.base, self.folder, self.error = resolve_location(self.roots, self.root, self.fid, resolve_ref)
        self.entries: list[FileEntry] = []
        self.title = dict(ROOT_LABELS).get(self.root, "Files")

    # ---- body ---------------------------------------------------------------------------

    def build_body(self) -> ft.Control:
        self.root_chips = ft.Row(
            [
                ft.Chip(label=ft.Text(label), selected=key == self.root, show_checkmark=False,
                        on_select=lambda e, k=key: self._go_root(k), key=f"files-root-{key}",
                        disabled=not self.roots.get(key))
                for key, label in ROOT_LABELS
            ],
            scroll=ft.ScrollMode.AUTO,
            spacing=6,
        )
        self.crumbs = ft.Row(self._crumb_controls(), scroll=ft.ScrollMode.AUTO, spacing=2)
        self.status = ft.Text("", theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT)
        self.list_view = ft.ListView(controls=[], expand=True, spacing=0, padding=ft.Padding.only(bottom=24))
        if self.error:
            self.list_view.controls = [EmptyState(icon="FOLDER_OFF", title=self.error, key="files-error")]
        else:
            self.list_view.controls = [ft.ProgressRing(width=24, height=24, stroke_width=2)]
        return ft.Column(
            [
                ft.Container(content=ft.Column([self.root_chips, self.crumbs, self.status], spacing=4, tight=True),
                             padding=ft.Padding.symmetric(horizontal=tokens.SPACING["md"], vertical=6)),
                self.list_view,
            ],
            spacing=0,
            expand=True,
        )

    def _crumb_controls(self) -> list[ft.Control]:
        label = dict(ROOT_LABELS).get(self.root, "Files")
        controls: list[ft.Control] = [ft.TextButton(content=label, on_click=lambda e: self._go_root(self.root),
                                                    key="files-crumb-root")]
        if self.folder and self.base and os.path.realpath(self.folder) != os.path.realpath(self.base):
            rel = os.path.relpath(self.folder, self.base)
            parts = [p for p in rel.replace("\\", "/").split("/") if p and p != "."]
            path = self.base
            for index, part in enumerate(parts):
                path = os.path.join(path, part)
                controls.append(ft.Icon(ft.Icons.CHEVRON_RIGHT, size=16))
                controls.append(ft.TextButton(content=part, on_click=lambda e, p=path: self._open_folder(p),
                                              key=f"files-crumb-{index}"))
        return controls

    def did_show(self) -> None:
        if self.error or self.folder is None:
            return
        if self.run_io is not None:
            try:
                import asyncio

                asyncio.ensure_future(self.reload())
                return
            except RuntimeError:
                pass
        self._show_entries(self._safe_list())

    def _safe_list(self) -> Any:
        try:
            if self.folder == self.base:
                os.makedirs(self.folder, exist_ok=True)  # a root that was never written yet
            return list_folder(self.folder)
        except OSError as exc:
            return exc

    async def reload(self) -> None:
        result = await self.run_io(self._safe_list) if self.run_io is not None else self._safe_list()
        self._show_entries(result)

    def _show_entries(self, result: Any) -> None:
        if isinstance(result, Exception):
            self.entries = []
            self.list_view.controls = [EmptyState(icon="FOLDER_OFF", title="This folder cannot be read",
                                                  body=str(result), key="files-error")]
            self.status.value = ""
        else:
            self.entries = list(result)
            if not self.entries:
                self.list_view.controls = [EmptyState(icon="FOLDER_OPEN", title="Empty folder", key="files-empty")]
            else:
                self.list_view.controls = [self._row(entry, i) for i, entry in enumerate(self.entries)]
            folders = sum(1 for e in self.entries if e.is_dir)
            self.status.value = f"{folders} folders · {len(self.entries) - folders} files"
        self._update()

    def _row(self, entry: FileEntry, index: int) -> ft.Control:
        date = time.strftime("%Y-%m-%d %H:%M", time.localtime(entry.mtime)) if entry.mtime else ""
        subtitle = date if entry.is_dir else f"{describe_size(entry.size)} · {date}"
        return ft.ListTile(
            leading=ft.Icon(icon_data(entry.icon)),
            title=ft.Text(entry.name, max_lines=2, overflow=ft.TextOverflow.ELLIPSIS),
            subtitle=ft.Text(subtitle, theme_style=ft.TextThemeStyle.BODY_SMALL),
            trailing=ft.Icon(ft.Icons.CHEVRON_RIGHT) if entry.is_dir else ft.Icon(ft.Icons.MORE_VERT),
            on_click=lambda e, it=entry: self._on_entry(it),
            min_height=tokens.SIZES["hit_target"],
            key=f"files-row-{index}",
        )

    def _update(self) -> None:
        body = self.body
        if body is not None and getattr(body, "page", None) is not None:
            try:
                body.update()
            except Exception:
                pass

    # ---- navigation / actions ----------------------------------------------------------------

    def _go_root(self, root: str) -> None:
        if self.navigate is not None and self.roots.get(root):
            self.navigate("tools.files", {"root": root})

    def _open_folder(self, path: str) -> None:
        if self.navigate is None or self.file_ref is None or not self.base or not _inside(path, self.base):
            return
        if os.path.realpath(path) == os.path.realpath(self.base):
            self._go_root(self.root)
            return
        self.navigate("tools.files.folder", {"root": self.root, "fid": self.file_ref(path)})

    def _on_entry(self, entry: FileEntry) -> Optional[ActionSheet]:
        if entry.is_dir:
            self._open_folder(entry.path)
            return None
        if self.files is None or self.page is None:
            return None
        tools = [
            ActionItem("Open with…", lambda it=entry: self.open_with(it), icon="OPEN_IN_NEW", key="file-open"),
            ActionItem("Rename", lambda it=entry: self.ask_rename(it), icon="DRIVE_FILE_RENAME_OUTLINE",
                       key="file-rename"),
            ActionItem("Delete", lambda it=entry: self._spawn(self.confirm_delete(it)), icon="DELETE_OUTLINE",
                       destructive=True, key="file-delete"),
        ]
        sheet = export_sheet(self.files, entry.path, page=self.page, notify=self.notify, tablet=self.tablet,
                             extra=tools)
        sheet.show(self.page)
        self.last_sheet = sheet
        return sheet

    # ---- file tools ------------------------------------------------------------------------------

    def open_with(self, entry: FileEntry) -> Optional[ActionSheet]:
        """Open with: Text editor (text files) · Reader (books) · Media viewer (images, video, audio) ·
        another app (Share)."""
        from glossarion_mobile.ui.tools import text_editor

        ext = os.path.splitext(entry.name)[1].lower()
        is_book = ext in (".epub", ".txt", ".pdf", ".html", ".xhtml", ".htm", ".md")
        kind = media_kind(entry.path)
        items = [
            ActionItem("Text editor", lambda: self.open_text(entry), icon="EDIT_NOTE", key="open-text",
                       disabled_reason=None if text_editor.is_text_file(entry.path) else "Not a text file"),
            ActionItem("Reader", lambda: self.open_reader(path=entry.path) if self.open_reader else None,
                       icon="AUTO_STORIES", key="open-reader",
                       disabled_reason=(None if is_book and self.open_reader is not None
                                        else "Not a book file" if not is_book else "The Reader is not available here")),
            ActionItem("Media viewer", lambda: self.open_media(entry), icon="PERM_MEDIA", key="open-media",
                       disabled_reason=(None if kind and self.push_overlay is not None
                                        else "Not an image, video or audio file" if not kind
                                        else "The media viewer is not available here")),
            ActionItem("Another app…", lambda: self._spawn(self.files.share([entry.path])) if self.files else None,
                       icon="IOS_SHARE", key="open-share",
                       disabled_reason=None if self.files is not None else "Sharing is not available"),
        ]
        sheet = ActionSheet(items, title="Open with", subtitle=entry.name, tablet=self.tablet)
        if self.page is not None:
            sheet.show(self.page)
        self.last_sheet = sheet
        return sheet

    def open_media(self, entry: FileEntry) -> Any:
        """The chat's full-screen MediaViewer on one image / video / audio file (zoom, play, Share,
        Save to…, Open externally: the file's own ExportSheet and share sheet)."""
        from glossarion_mobile.ui.chat.media_cards import AudioHub
        from glossarion_mobile.ui.chat.media_model import MediaItem
        from glossarion_mobile.ui.components.media_viewer import MediaViewer

        kind = media_kind(entry.path)
        if kind is None or self.push_overlay is None:
            return None

        def share(path: str) -> Any:
            return self._spawn(self.files.share([path])) if self.files is not None else None

        def save(path: str) -> Any:
            if self.files is None or self.page is None:
                return None
            sheet = export_sheet(self.files, path, page=self.page, notify=self.notify, tablet=self.tablet)
            sheet.show(self.page)
            return sheet

        def close() -> None:
            if self.pop_overlay is not None:
                self.pop_overlay(viewer.view)

        viewer = MediaViewer([MediaItem(kind, entry.path, os.path.isfile(entry.path))], on_close=close,
                             on_save=save, on_share=share, on_open_external=share,
                             audio_hub=AudioHub() if kind == "audio" else None, spawn=self._spawn)
        self.media_viewer = viewer
        self.push_overlay(viewer.view)
        return viewer

    def open_text(self, entry: FileEntry) -> Optional[str]:
        if self.navigate is None or self.file_ref is None:
            return None
        fid = self.file_ref(entry.path)
        self.navigate("tools.text", {"fid": fid})
        return fid

    def ask_rename(self, entry: FileEntry) -> Any:
        from glossarion_mobile.ui.components.dialogs import ConfirmDialog

        field = ft.TextField(label="New name", value=entry.name, autofocus=True, key="rename-field")
        dialog = ConfirmDialog(title="Rename", confirm_label="Rename", cancel_label="Cancel",
                               on_confirm=lambda: self.rename(entry, str(field.value or "")))
        dialog.dialog.content.content.controls.insert(0, field)
        dialog.field = field  # type: ignore[attr-defined]
        if self.page is not None:
            dialog.show(self.page)
        self.last_sheet = dialog
        return dialog

    async def rename(self, entry: FileEntry, new_name: str) -> Optional[str]:
        try:
            target = await self._io(lambda: rename_entry(entry.path, new_name, self.roots))
        except (OSError, ValueError) as exc:
            self._say(str(exc))
            return None
        self._say(f"Renamed to {os.path.basename(target)}")
        await self.reload()
        return target

    async def confirm_delete(self, entry: FileEntry) -> bool:
        from glossarion_mobile.ui.tools.common import ChoiceDialog

        dialog = ChoiceDialog("Delete file", f"Delete {entry.name}?\n\nThis cannot be undone.",
                              [("cancel", "Cancel", "text"), ("delete", "Delete", "destructive")], key="file-delete")
        if self.page is None:
            return False
        dialog.show(self.page)
        if await dialog.wait() != "delete":
            return False
        return await self.delete(entry)

    async def delete(self, entry: FileEntry) -> bool:
        try:
            await self._io(lambda: delete_entry(entry.path, self.roots))
        except (OSError, ValueError) as exc:
            self._say(f"Could not delete: {exc}")
            return False
        self._say(f"Deleted {entry.name}")
        await self.reload()
        return True

    async def _io(self, fn: Callable[..., Any]) -> Any:
        if self.run_io is not None:
            return await self.run_io(fn)
        return fn()

    def _say(self, message: str) -> None:
        if self.notify is not None:
            try:
                self.notify(message)
            except Exception:
                pass

    @staticmethod
    def _spawn(coro: Any) -> Any:
        import asyncio

        try:
            return asyncio.ensure_future(coro)
        except RuntimeError:
            coro.close()
            return None
