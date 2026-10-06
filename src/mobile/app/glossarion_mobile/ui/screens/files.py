"""Minimal output file browser (``/tools/files/<root>[/<fid>]``, UI_SPEC §4.10).

Roots (chips): Output (``OUTPUT_DIRECTORY``), Library, Inbox and Chat
workspaces. A sub-folder is addressed by an opaque ``fid`` from the Prefs
FileRef registry, never by a path in the route. Every listed path is checked
against the roots (``realpath``), so a stale or forged ``fid`` cannot leave
them. Rows show icon, name and "size · date"; a folder opens the next level,
a file opens the ExportSheet (Share · Save to… · Save to Downloads / Show in
Files). Rename / Delete / Open-with arrive with the full file tools (U6) and
are listed disabled. Listing runs on a worker thread.
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

__all__ = ["ROOT_LABELS", "FileBrowserScreen", "FileEntry", "describe_size", "list_folder", "resolve_location"]

ROOT_LABELS = (("output", "Output"), ("library", "Library"), ("inbox", "Inbox"), ("chats", "Chat workspaces"))
MAX_ENTRIES = 2000
_ICONS = {
    ".epub": "MENU_BOOK", ".pdf": "PICTURE_AS_PDF", ".txt": "DESCRIPTION", ".md": "DESCRIPTION",
    ".html": "HTML", ".xhtml": "HTML", ".htm": "HTML", ".json": "DATA_OBJECT", ".csv": "TABLE_CHART",
    ".png": "IMAGE", ".jpg": "IMAGE", ".jpeg": "IMAGE", ".webp": "IMAGE", ".gif": "IMAGE",
    ".zip": "FOLDER_ZIP", ".cbz": "FOLDER_ZIP", ".mp3": "AUDIOTRACK", ".wav": "AUDIOTRACK", ".mp4": "MOVIE",
    ".log": "TERMINAL",
}


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
    ) -> None:
        super().__init__(match)
        self.roots = dict(roots)
        self.files = files
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
        later = [
            ActionItem(label, None, icon=icon, disabled_reason=reason, key=f"file-{key}")
            for key, label, icon, reason in (
                ("open", "Open with…", "OPEN_IN_NEW", "Arrives with the text editor and file tools (U7)"),
                ("rename", "Rename", "DRIVE_FILE_RENAME_OUTLINE", "Arrives with the file tools (U7)"),
                ("delete", "Delete", "DELETE_OUTLINE", "Arrives with the file tools (U7)"),
            )
        ]
        sheet = export_sheet(self.files, entry.path, page=self.page, notify=self.notify, tablet=self.tablet,
                             extra=later)
        sheet.show(self.page)
        return sheet
