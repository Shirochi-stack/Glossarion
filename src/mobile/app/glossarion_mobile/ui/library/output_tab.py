"""Book page › Output (UI_SPEC §3.9).

* **Compiled outputs** (EPUB / PDF / TXT / HTML, ``library_core.list_compiled_outputs``
  priority order, extra files flagged "⚠ +N"): Open (Reader for EPUB, Share
  otherwise) · Share · Save to… · Save to Downloads (Android) / Show in Files (iOS)
  through ``FileBridge`` (unsupported options disabled with their reason).
* **Compile panel:** Compile EPUB · Compile PDF (jobs).
* **Workspace files:** the file browser at the output folder.
* **Raw source row:** file name, Share, "Re-link…" (Scan for raw).
* **Storage line:** "Workspace 84 MB" (measured on the io pool) with a Files link.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import flet as ft

from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.components.action_sheet import ActionItem, ActionSheet
from glossarion_mobile.ui.library.common import section_title
from glossarion_mobile.ui.library.models import size_text
from glossarion_mobile.ui.theme import HIT_TARGET

__all__ = ["OutputTab", "workspace_size_text"]

log = logging.getLogger("glossarion.library.ui")

_KIND_ICONS = {"epub": "MENU_BOOK", "pdf": "PICTURE_AS_PDF", "txt": "DESCRIPTION", "html": "LANGUAGE"}


def workspace_size_text(size: int) -> str:
    return f"Workspace {size_text(size)}"


def _folder_size(folder: str) -> int:
    total = 0
    for root, _dirs, names in os.walk(folder):
        for name in names:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


class OutputTab:
    def __init__(self, page: Any) -> None:
        self.page = page
        self.ctx = page.ctx
        self.outputs: list = []
        self.raw: str = ""
        self.size: Optional[int] = None
        self.stale = True
        self.last_sheet: Optional[ActionSheet] = None

    def build(self) -> ft.Control:
        self.outputs_column = ft.Column(spacing=4, key="out-files")
        self.compile_row = ft.Row([
            ft.FilledTonalButton(content="Compile EPUB", icon=ft.Icons.MENU_BOOK,
                                 on_click=lambda e: self.ctx.spawn(self.page.compile("compile_epub")), key="out-epub"),
            ft.FilledTonalButton(content="Compile PDF", icon=ft.Icons.PICTURE_AS_PDF,
                                 on_click=lambda e: self.ctx.spawn(self.page.compile("compile_pdf")), key="out-pdf"),
        ], wrap=True, spacing=8)
        self.raw_row = ft.ListTile(title=ft.Text("Raw source"), subtitle=ft.Text("—"), dense=True, key="out-raw",
                                   trailing=ft.Row([
                                       ft.IconButton(icon=ft.Icons.IOS_SHARE, tooltip="Share the raw file",
                                                     size_constraints=HIT_TARGET, on_click=self._share_raw,
                                                     key="out-raw-share"),
                                       ft.TextButton(content="Re-link…",
                                                     on_click=lambda e: self.ctx.go("library.scan_raw"),
                                                     key="out-relink"),
                                   ], tight=True, spacing=0))
        self.storage_text = ft.Text("Workspace …", key="out-size")
        self.list = ft.ListView([
            section_title("Compiled outputs"),
            self.outputs_column,
            section_title("Compile"),
            self.compile_row,
            ft.Text("EPUB and PDF output settings are in Settings.", theme_style=ft.TextThemeStyle.BODY_SMALL,
                    color=ft.Colors.ON_SURFACE_VARIANT),
            section_title("Workspace files"),
            ft.ListTile(leading=ft.Icon(ft.Icons.FOLDER_OPEN), title=ft.Text("Browse workspace files"),
                        subtitle=ft.Text("Glossary files, metadata.json, TOC.txt, images, QA reports…"),
                        on_click=lambda e: self.page.open_files(), key="out-browse"),
            self.raw_row,
            ft.Row([self.storage_text, ft.TextButton(content="Files", on_click=lambda e: self.page.open_files())],
                   alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
        ], spacing=tokens.SPACING["sm"], padding=12, expand=True, key="output")
        self._render()
        return self.list

    def mark_stale(self) -> None:
        self.stale = True

    async def reload(self) -> None:
        service = self.page.service
        book = self.page.book

        def gather() -> tuple:
            outputs = service.compiled_outputs_blocking(book)
            raw = service.raw_source(book)
            folder = str(book.get("output_folder") or "")
            size = _folder_size(folder) if folder and os.path.isdir(folder) else None
            return outputs, raw, size

        self.outputs, self.raw, self.size = await self.ctx.io(gather)
        self.stale = False
        self._render()

    def _render(self) -> None:
        if getattr(self, "list", None) is None:
            return
        book = self.page.book
        has_workspace = bool(book.get("output_folder"))
        for button in self.compile_row.controls:
            button.disabled = not has_workspace
        rows: list[ft.Control] = []
        conflicts = list(book.get("compiled_conflicts") or [])
        for index, (path, kind) in enumerate(self.outputs):
            name = os.path.basename(path)
            try:
                meta = f"{kind.upper()} · {size_text(os.path.getsize(path))}"
            except OSError:
                meta = kind.upper()
            if index == 0 and conflicts:
                meta += f" · ⚠ +{len(conflicts)}"
            rows.append(ft.ListTile(
                leading=ft.Icon(getattr(ft.Icons, _KIND_ICONS.get(kind, "INSERT_DRIVE_FILE"))),
                title=ft.Text(name, max_lines=2, overflow=ft.TextOverflow.ELLIPSIS),
                subtitle=ft.Text(meta),
                trailing=ft.IconButton(icon=ft.Icons.MORE_VERT, tooltip="Output actions", size_constraints=HIT_TARGET,
                                       on_click=lambda e, p=path, k=kind: self.open_sheet(p, k)),
                on_click=lambda e, p=path, k=kind: self.open_output(p, k),
                key=f"out-file-{index}",
            ))
        if not rows:
            rows.append(ft.Text("No compiled output yet. Compile the translation to create an EPUB or PDF.",
                                color=ft.Colors.ON_SURFACE_VARIANT, key="out-none"))
        self.outputs_column.controls = rows
        self.raw_row.subtitle = ft.Text(os.path.basename(self.raw) if self.raw else "The raw source file can't be found")
        self.storage_text.value = workspace_size_text(self.size) if self.size is not None else (
            "No output workspace" if not has_workspace else "Workspace …")
        self.ctx.push(self.list)

    # ---- actions ------------------------------------------------------------------------------------

    def open_output(self, path: str, kind: str) -> Any:
        if kind == "epub":
            if path and os.path.abspath(path) != os.path.abspath(str(self.page.book.get("path") or "")):
                return self.ctx.open_reader(path=path, mode="translated")
            return self.page.open_reader(mode="translated")
        return self.ctx.spawn(self._share([path]))

    def open_sheet(self, path: str, kind: str) -> ActionSheet:
        files = self.ctx.files
        items = [ActionItem("Open", lambda: self.open_output(path, kind), icon="OPEN_IN_NEW")]
        if files is not None:
            for option in files.export_options(path):
                items.append(ActionItem(option.label, (lambda o=option.id: self.ctx.spawn(self.export(o, path))),
                                        icon=option.icon, disabled_reason=option.disabled_reason,
                                        key=f"export-{option.id}"))
        items.append(ActionItem("Delete", lambda: self.confirm_delete(path), icon="DELETE_OUTLINE", destructive=True,
                                key="output-delete"))
        sheet = ActionSheet(items, title=os.path.basename(path), tablet=self.ctx.tablet)
        self.last_sheet = sheet
        self.ctx.show(sheet)
        return sheet

    def confirm_delete(self, path: str) -> Any:
        """Delete one compiled output (inside the book's workspace only); the Library rescans."""
        from glossarion_mobile.ui.components.dialogs import ConfirmDialog

        folder = str(self.page.book.get("output_folder") or "")
        inside = bool(folder) and os.path.normcase(os.path.dirname(os.path.abspath(path))) == os.path.normcase(
            os.path.abspath(folder))
        if not inside:
            self.ctx.say("Only compiled files inside the book's workspace can be deleted here")
            return None

        async def run() -> None:
            try:
                await self.ctx.io(os.remove, path)
            except OSError as exc:
                self.ctx.say(f"Delete failed: {exc}")
                return
            self.ctx.say(f"Deleted {os.path.basename(path)}")
            self.page.service.mark_dirty()
            await self.reload()
            await self.page.service.refresh(reason="output deleted")

        dialog = ConfirmDialog(title="Delete", body=f"Delete {os.path.basename(path)}?\n\nThis cannot be undone.",
                               confirm_label="Delete", destructive=True, on_confirm=run)
        return self.ctx.show(dialog)

    async def export(self, option_id: str, path: str) -> Any:
        files = self.ctx.files
        if files is None:
            return None
        result = await files.export(option_id, path)
        needs_confirm = getattr(result, "needs_confirm", False)
        if needs_confirm:
            result = await files.export(option_id, path, confirmed=True)
        if option_id == "downloads":
            self.ctx.say("Saved to Downloads/Glossarion" if result else "Could not save to Downloads")
        elif option_id == "save" and getattr(result, "ok", False):
            self.ctx.say("Saved")
        return result

    async def _share(self, paths: list) -> bool:
        files = self.ctx.files
        if files is None or not paths:
            self.ctx.say("Sharing is not available in this session")
            return False
        return bool(await files.share(paths))

    async def share_primary(self) -> bool:
        if self.stale:
            await self.reload()
        if self.outputs:
            return await self._share([self.outputs[0][0]])
        if self.raw:
            return await self._share([self.raw])
        self.ctx.say("Nothing to share yet")
        return False

    def _share_raw(self, e: Any = None) -> Any:
        if not self.raw:
            self.ctx.say("The raw source file can't be found")
            return None
        return self.ctx.spawn(self._share([self.raw]))
