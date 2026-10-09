"""Book page › Output (UI_SPEC §3.9).

* **Compiled outputs** (EPUB / PDF / TXT / HTML, ``library_core.list_compiled_outputs``
  priority order, extra files flagged "⚠ +N"): Open (Reader for EPUB, Share
  otherwise) · Share · Save to… · Save to Downloads (Android) / Show in Files (iOS)
  through ``FileBridge`` (unsupported options disabled with their reason).
* **Compile panel:** Compile EPUB · Compile PDF (jobs) and links to Settings › EPUB output / PDF.
* **Workspace files** (collapsible groups, collected on the io pool by ``workspace_groups``): glossary
  files · metadata.json / TOC.txt / translated_headers.txt / extraction_report.txt · SDLXLIFF sidecars
  (open the reviewer) · text_to_speech/ (the in-app MediaViewer player) · images (the folder in Files) ·
  QA reports (the QA report viewer); "Browse workspace files" opens the file browser.
* **Raw source row:** file name, Share, "Re-link…" (Scan for raw).
* **Storage line:** "Workspace 84 MB" (measured on the io pool) with a Files link.
* **📝 Review** (UI_SPEC §3.9 / §4.8): when the Review generator wrote ``review/review.md`` (or the
  Volume-mode ``review/combined_review/review.md``) in the workspace, a row opens it in Tools ›
  Review (``tools.review?out=<bid>``).
* **Cloud & sharing** (U10, ``ui/screens/cloud_sync``): each compiled output's row shows its cloud state
  ("Saved to Drive › Glossarion · 2 min ago", "Saving… 45%", "Choose where to save it", "Lost access ·
  Reconnect"; tap it to act); the section has the destination line, **Send now**, "Copy this book:
  Default / Always / Never", **Share file via link** and the book's saved links (Copy · Share · Delete).
  The row sheet adds "Send to cloud now" and "Share file via link". Unavailable actions stay visible with
  their ReasonChip. Cloud / share state is read on the io pool in ``reload`` and follows the services'
  change events while the tab is mounted (``refresh_cloud``).

The workspace is the book's resolved one (``progress_model.book_workspace``): an organized
Library/Translated book lists, compiles and browses the workspace it came from.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Optional

import flet as ft

from glossarion_mobile.services.library import book_identity
from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.components.action_sheet import ActionItem, ActionSheet
from glossarion_mobile.ui.components.reason_chip import ReasonChip
from glossarion_mobile.ui.library import progress_model as pm
from glossarion_mobile.ui.library.common import section_title
from glossarion_mobile.ui.library.models import size_text
from glossarion_mobile.ui.screens import cloud_sync as u10
from glossarion_mobile.ui.theme import HIT_TARGET

__all__ = ["OutputTab", "REVIEW_FILES", "WORKSPACE_GROUPS", "find_review", "workspace_groups", "workspace_size_text"]

#: Where review_generator saves a book's review inside its workspace (``_save_review_text`` /
#: ``review_paths_for``: the single review, then the Volume-mode combined review).
REVIEW_FILES = (("review", "review.md"), ("review", "combined_review", "review.md"))


def find_review(folder: str) -> str:
    """Blocking: the workspace's review file (non-empty), or ""."""
    if not folder:
        return ""
    for parts in REVIEW_FILES:
        path = os.path.join(folder, *parts)
        try:
            if os.path.isfile(path) and os.path.getsize(path) > 0:
                return path
        except OSError:
            continue
    return ""

log = logging.getLogger("glossarion.library.ui")

#: UI_SPEC §3.9 workspace file groups: (id, title, icon).
WORKSPACE_GROUPS = (
    ("glossary", "Glossary files", "MENU_BOOK"),
    ("metadata", "metadata.json · TOC.txt · headers · reports", "DATA_OBJECT"),
    ("sdlxliff", "SDLXLIFF sidecars", "TRANSLATE"),
    ("tts", "Text-to-speech audio", "AUDIOTRACK"),
    ("images", "Images", "IMAGE"),
    ("qa", "QA reports", "FACT_CHECK"),
)
#: Workspace files of the "metadata" group (the header / TOC caches and the extraction report).
METADATA_FILES = ("metadata.json", "TOC.txt", "translated_headers.txt", "extraction_report.txt")
_GLOSSARY_EXTENSIONS = (".csv", ".json", ".txt", ".md", ".xlsx")
_GROUP_LIMIT = 200


def _files_in(folder: str, extensions: Optional[tuple] = None) -> list:
    try:
        names = sorted(os.listdir(folder), key=str.casefold)
    except OSError:
        return []
    out = []
    for name in names:
        path = os.path.join(folder, name)
        if os.path.isfile(path) and (extensions is None or os.path.splitext(name)[1].lower() in extensions):
            out.append(path)
    return out


def workspace_groups(folder: str) -> dict:
    """Blocking: ``{group id: [paths]}`` of a book workspace (UI_SPEC §3.9; empty groups are left out).

    Glossary files: top-level files whose name mentions "glossary" (CSV / JSON / TXT / MD / XLSX);
    metadata: ``METADATA_FILES``; SDLXLIFF: ``*.sdlxliff`` at the top level and under ``SDLXLIFF/``;
    TTS: the audio under ``text_to_speech/``; images: the ``images/`` folder (the folder itself, its
    file count in the title); QA reports: the QA Scanner reports of this folder (``qa_model.list_reports``).
    """
    groups: dict = {}
    if not folder or not os.path.isdir(folder):
        return groups
    top = _files_in(folder)
    glossary = [p for p in top if "glossary" in os.path.basename(p).lower()
                and os.path.splitext(p)[1].lower() in _GLOSSARY_EXTENSIONS]
    if glossary:
        groups["glossary"] = glossary[:_GROUP_LIMIT]
    metadata = [os.path.join(folder, name) for name in METADATA_FILES if os.path.isfile(os.path.join(folder, name))]
    if metadata:
        groups["metadata"] = metadata
    sdlxliff = [p for p in top if p.lower().endswith(".sdlxliff")]
    for sub in ("SDLXLIFF", "sdlxliff"):
        sub_dir = os.path.join(folder, sub)
        if os.path.isdir(sub_dir):
            sdlxliff.extend(p for p in _files_in(sub_dir, (".sdlxliff",)) if p not in sdlxliff)
            break
    if sdlxliff:
        groups["sdlxliff"] = sdlxliff[:_GROUP_LIMIT]
    tts_dir = os.path.join(folder, "text_to_speech")
    if os.path.isdir(tts_dir):
        try:
            from glossarion_mobile.ui.screens.files import media_kind

            audio = [p for p in _files_in(tts_dir) if media_kind(p) == "audio"]
        except Exception:
            audio = _files_in(tts_dir, (".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac", ".opus"))
        if audio:
            groups["tts"] = audio[:_GROUP_LIMIT]
    images_dir = os.path.join(folder, "images")
    if os.path.isdir(images_dir) and _files_in(images_dir):
        groups["images"] = [images_dir]
    try:
        from glossarion_mobile.ui.tools.qa_model import list_reports

        reports = [entry.path for entry in list_reports([folder])]
    except Exception:
        log.debug("QA report lookup failed", exc_info=True)
        reports = []
    if reports:
        groups["qa"] = reports
    return groups


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
        self.review: str = ""
        self.size: Optional[int] = None
        self.groups: dict = {}
        self.stale = True
        self.last_sheet: Optional[ActionSheet] = None
        self.media_viewer: Any = None
        self.group_builds = 0
        # U10: the cloud-sync / share-link state of this book (read on the io pool in ``reload``)
        self.workspace: str = ""
        self.sizes: dict = {}  # compiled output -> bytes (measured on the io pool in ``reload``)
        self.identity: str = ""  # the book's identity path (``book_identity``): cloud records and links key by it
        self.cloud_state: dict = {}
        self.book_cloud: dict = {}
        self.links: list = []
        self.providers: list = []
        self.cloud_builds = 0
        self._u10: Optional[u10.U10Actions] = None
        self._cloud_unsubs: list = []
        self._cloud_refreshing = False
        self._cloud_again = False

    def build(self) -> ft.Control:
        self.outputs_column = ft.Column(spacing=4, key="out-files")
        self.cloud_column = ft.Column(spacing=tokens.SPACING["xs"], key="out-cloud")
        self.groups_column = ft.Column(spacing=0, key="out-groups")
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
        self.review_row = ft.ListTile(leading=ft.Text("📝", size=20), title=ft.Text("Review"),
                                      subtitle=ft.Text("review.md"), visible=False, key="out-review",
                                      trailing=ft.Icon(ft.Icons.CHEVRON_RIGHT), on_click=lambda e: self.open_review())
        self.list = ft.ListView([
            section_title("Compiled outputs"),
            self.outputs_column,
            self.review_row,
            section_title("Cloud & sharing"),
            self.cloud_column,
            section_title("Compile"),
            self.compile_row,
            ft.Row([
                ft.TextButton(content="EPUB output settings", icon=ft.Icons.SETTINGS_OUTLINED, key="out-epub-settings",
                              on_click=lambda e: self.open_settings("epub_output")),
                ft.TextButton(content="PDF settings", icon=ft.Icons.SETTINGS_OUTLINED, key="out-pdf-settings",
                              on_click=lambda e: self.open_settings("pdf")),
            ], wrap=True, spacing=4),
            section_title("Workspace files"),
            self.groups_column,
            ft.ListTile(leading=ft.Icon(ft.Icons.FOLDER_OPEN), title=ft.Text("Browse workspace files"),
                        subtitle=ft.Text("Every file of the output folder"),
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
            # the resolved workspace (an organized book's own folder) and the Library-filed EPUB itself
            outputs = service.compiled_outputs_blocking(pm.workspace_row(service, book))
            raw = service.raw_source(book)
            folder = pm.book_workspace(service, book)
            size = _folder_size(folder) if folder and os.path.isdir(folder) else None
            identity = book_identity(book) or folder or ""
            sizes: dict = {}
            for path, _kind in outputs:
                try:
                    sizes[path] = os.path.getsize(path)
                except OSError:
                    pass
            return (outputs, raw, size, find_review(folder), workspace_groups(folder), folder, identity,
                    self._gather_cloud(identity), sizes)

        (self.outputs, self.raw, self.size, self.review, self.groups, folder, identity,
         cloud, self.sizes) = await self.ctx.io(gather)
        self.workspace = str(folder or "")
        self.identity = str(identity or "")
        self._apply_cloud(cloud)
        self.stale = False
        self._subscribe_cloud()
        self._render()

    def open_review(self) -> Any:
        """📝 Review → Tools › Review generator on this book (``?out=<bid>``)."""
        bid = getattr(self.page, "bid", None) or self.page.service.bid_for(self.page.book)
        return self.ctx.go("tools.review", None, {"out": bid})

    def _render(self) -> None:
        if getattr(self, "list", None) is None:
            return
        book = self.page.book
        self.review_row.visible = bool(self.review)
        if self.review:
            combined = os.path.join("combined_review", "review.md") in self.review
            self.review_row.subtitle = ft.Text("Combined review (Volume mode)" if combined else "review/review.md")
        has_workspace = bool(pm.book_workspace(self.page.service, book))
        for button in self.compile_row.controls:
            button.disabled = not has_workspace
        self._render_outputs()
        self.groups_column.controls = self._group_controls()
        self.raw_row.subtitle = ft.Text(os.path.basename(self.raw) if self.raw else "The raw source file can't be found")
        self.storage_text.value = workspace_size_text(self.size) if self.size is not None else (
            "No output workspace" if not has_workspace else "Workspace …")
        self.ctx.push(self.list)

    def _render_outputs(self) -> None:
        """The compiled-output rows (with their cloud lines) and the Cloud & sharing section; a cloud / share
        change repaints only these (``refresh_cloud``), so expanded workspace groups stay open."""
        book = self.page.book
        rows: list[ft.Control] = []
        conflicts = list(book.get("compiled_conflicts") or [])
        self.cloud_builds += 1
        n = self.cloud_builds
        cloud_rows = self.cloud_rows()
        for index, (path, kind) in enumerate(self.outputs):
            name = os.path.basename(path)
            size = self.sizes.get(path)
            meta = f"{kind.upper()} · {size_text(size)}" if size is not None else kind.upper()
            if index == 0 and conflicts:
                meta += f" · ⚠ +{len(conflicts)}"
            subtitle: ft.Control = ft.Text(meta)
            line = self.cloud_line(index)
            if line is not None:
                icon, text, tone = line
                entry = cloud_rows.get(index) or {}
                subtitle = ft.Column([ft.Text(meta), u10.status_row(
                    icon, text, tone, key=f"out-cloud-line-{index}-{n}",
                    on_click=lambda e, k=kind, en=entry: self.on_cloud_line(k, en))], spacing=2, tight=True)
            rows.append(ft.ListTile(
                leading=ft.Icon(getattr(ft.Icons, _KIND_ICONS.get(kind, "INSERT_DRIVE_FILE"))),
                title=ft.Text(name, max_lines=2, overflow=ft.TextOverflow.ELLIPSIS),
                subtitle=subtitle,
                trailing=ft.IconButton(icon=ft.Icons.MORE_VERT, tooltip="Output actions", size_constraints=HIT_TARGET,
                                       on_click=lambda e, p=path, k=kind: self.open_sheet(p, k)),
                on_click=lambda e, p=path, k=kind: self.open_output(p, k),
                key=f"out-file-{index}-{n}",
            ))
        if not rows:
            rows.append(ft.Text("No compiled output yet. Compile the translation to create an EPUB or PDF.",
                                color=ft.Colors.ON_SURFACE_VARIANT, key=f"out-none-{n}"))
        self.outputs_column.controls = rows
        self.cloud_column.controls = self._cloud_controls(n)

    def _group_controls(self) -> list:
        """One collapsible group per non-empty ``workspace_groups`` entry (fresh keys per render: Flet 1.0.3
        freezes a subtree re-rendered under the key of the one it replaces)."""
        self.group_builds += 1
        n = self.group_builds
        out: list = []
        for group_id, title, icon in WORKSPACE_GROUPS:
            paths = list(self.groups.get(group_id) or ())
            if not paths:
                continue
            rows = [self._file_row(group_id, path, index) for index, path in enumerate(paths)]
            count = len(paths) if group_id != "images" else len(_files_in(paths[0]))
            out.append(ft.ExpansionTile(
                title=ft.Text(f"{title} ({count})"), leading=ft.Icon(getattr(ft.Icons, icon, None)),
                controls=rows, dense=True, key=f"out-group-{group_id}-{n}"))
        return out

    def _file_row(self, group_id: str, path: str, index: int) -> ft.Control:
        name = os.path.basename(path)
        subtitle = None
        if group_id == "qa":
            subtitle = os.path.basename(os.path.dirname(path))
            name = "QA report"
        elif group_id == "images":
            name = "images/"
            subtitle = "Open the folder in Files"
        elif group_id == "tts":
            subtitle = "Play"
        return ft.ListTile(title=ft.Text(name, max_lines=2, overflow=ft.TextOverflow.ELLIPSIS),
                           subtitle=ft.Text(subtitle) if subtitle else None, dense=True,
                           on_click=lambda e, g=group_id, p=path: self.open_workspace_file(g, p),
                           key=f"out-{group_id}-{index}")

    def open_workspace_file(self, group_id: str, path: str) -> Any:
        """Open one workspace file the way its group does: QA report viewer, the MediaViewer player, the
        SDLXLIFF reviewer, the metadata editor, the images folder in Files, else the text editor."""
        prefs = self.ctx.prefs
        bid = getattr(self.page, "bid", None)
        if group_id == "qa" and prefs is not None:
            return self.ctx.go("tools.qa.report", {"rid": prefs.file_ref(path)})
        if group_id == "tts":
            from glossarion_mobile.ui.screens.files import open_media_viewer

            viewer = open_media_viewer(path, push_overlay=self.ctx.push_overlay, pop_overlay=self._pop_media,
                                       files=self.ctx.files, page=self.ctx.page, notify=self.ctx.notify,
                                       tablet=self.ctx.tablet, spawn=self.ctx.spawn)
            if viewer is not None:
                self.media_viewer = viewer
                return viewer
            return self.ctx.spawn(self._share([path]))
        if group_id == "sdlxliff":
            return self.ctx.go("tools.sdlxliff", None, {"out": bid or self.page.service.bid_for(self.page.book)})
        if group_id == "metadata" and os.path.basename(path) == "metadata.json":
            return self.ctx.go("library.book.metadata", {"bid": bid or self.page.service.bid_for(self.page.book)})
        if group_id == "images" and prefs is not None:
            return self.ctx.go("tools.files.folder", {"root": "output", "fid": prefs.file_ref(path)})
        from glossarion_mobile.ui.tools import text_editor

        return text_editor.open_text_editor(self.ctx, path)

    def _pop_media(self, view: Any) -> None:
        shell = getattr(self.ctx, "shell", None)
        overlays = getattr(shell, "overlays", None) if shell is not None else None
        if overlays is not None and view not in overlays:
            return
        if self.ctx.pop_overlay is not None:
            self.ctx.pop_overlay()

    def open_settings(self, section: str) -> Any:
        """Settings › EPUB output / PDF (the compile settings)."""
        return self.ctx.go("settings.section", {"section": section})

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
        # U10: this book to the cloud now, and a share link for this file
        entry = self.cloud_entry(kind) or {}
        if entry.get("status") == "needs_pick":
            items.append(ActionItem("Choose where to save…", lambda: self.ctx.spawn(self.choose_location(kind)),
                                    icon="DRIVE_FILE_MOVE", key="output-cloud-pick"))
        items.append(ActionItem(u10.SEND_NOW_LABEL, lambda: self.ctx.spawn(self.send_now()), icon="CLOUD_UPLOAD",
                                disabled_reason=self.cloud_reason(), key="output-cloud"))
        items.append(ActionItem(u10.SHARE_LABEL, lambda: self.ctx.spawn(self.actions.provider_sheet(path, self.identity)),
                                icon="LINK", disabled_reason=self.share_reason(), key="output-share-link"))
        items.append(ActionItem("Delete", lambda: self.confirm_delete(path), icon="DELETE_OUTLINE", destructive=True,
                                key="output-delete"))
        sheet = ActionSheet(items, title=os.path.basename(path), tablet=self.ctx.tablet)
        self.last_sheet = sheet
        self.ctx.show(sheet)
        return sheet

    def confirm_delete(self, path: str) -> Any:
        """Delete one compiled output (inside the book's workspace only); the Library rescans."""
        from glossarion_mobile.ui.components.dialogs import ConfirmDialog

        folder = pm.book_workspace(self.page.service, self.page.book)
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

    # ---- U10: cloud sync and share links ------------------------------------------------------------

    def _extra(self, name: str) -> Any:
        extras = getattr(self.ctx, "extras", None)
        return extras.get(name) if isinstance(extras, dict) else None

    @property
    def actions(self) -> u10.U10Actions:
        """The shared U10 actions (``ui/screens/cloud_sync``) over this context: the services come from
        ``ctx.extras`` (``cloud_sync`` / ``share_links``, set by the Library feature)."""
        if self._u10 is None:
            files = self.ctx.files
            self._u10 = u10.U10Actions(
                cloud=self._extra("cloud_sync"), shares=self._extra("share_links"), page=self.ctx.page,
                say=self.ctx.say, spawn=self.ctx.spawn, io=self.ctx.io, go=self.ctx.go,
                copy_text=self.ctx.copy_text, share_text=getattr(files, "share_text", None) if files is not None else None,
                show_in_files=getattr(files, "show_in_files", None) if files is not None else None,
                open_url=self._extra("open_url"), read_clipboard=self._extra("read_clipboard"),
                tablet=self.ctx.tablet, platform=self.ctx.platform)
        return self._u10

    def _gather_cloud(self, identity: str) -> dict:
        """Blocking (io pool): the book's cloud and share-link state (keyed by its identity path)."""
        actions = self.actions
        return {
            "state": actions.cloud.snapshot(),
            "book": actions.cloud.book(identity) if identity else {},
            "links": actions.shares.links_for(identity) if identity else [],
            "providers": actions.shares.providers(),
        }

    def _apply_cloud(self, cloud: dict) -> None:
        self.cloud_state = dict(cloud.get("state") or {})
        self.book_cloud = dict(cloud.get("book") or {})
        self.links = list(cloud.get("links") or [])
        self.providers = list(cloud.get("providers") or [])

    def _subscribe_cloud(self) -> None:
        """Follow the services' change events (progress, a finished copy, a new link) while this Book page
        lives; the subscriptions end with the page (``BookPageScreen.dispose`` runs its ``_unsubs``)."""
        if self._cloud_unsubs:
            return
        actions = self.actions
        for facade in (actions.cloud, actions.shares):
            if facade.available:
                self._cloud_unsubs.append(facade.subscribe(self._on_cloud_change))
        owner = getattr(self.page, "_unsubs", None)
        if isinstance(owner, list):
            owner.append(self.unsubscribe_cloud)

    def unsubscribe_cloud(self) -> None:
        for unsub in self._cloud_unsubs:
            try:
                unsub()
            except Exception:
                pass
        self._cloud_unsubs = []

    def _on_cloud_change(self, *args: Any) -> None:
        dispatcher = getattr(self.ctx, "dispatcher", None)
        if dispatcher is not None and getattr(dispatcher, "bound", False) and not dispatcher.on_loop_thread():
            dispatcher.post(self._schedule_cloud)
            return
        self._schedule_cloud()

    def _schedule_cloud(self) -> None:
        if self._cloud_refreshing:
            self._cloud_again = True
            return
        self.ctx.spawn(self.refresh_cloud())

    async def refresh_cloud(self) -> None:
        """Re-read only the cloud / share-link state (no workspace walk) and repaint; a burst of progress
        events is coalesced into one refresh at a time."""
        if self._cloud_refreshing:
            self._cloud_again = True
            return
        self._cloud_refreshing = True
        try:
            while True:
                self._cloud_again = False
                cloud = await self.ctx.io(self._gather_cloud, self.identity)
                self._apply_cloud(cloud)
                if getattr(self, "list", None) is not None:
                    self._render_outputs()
                    self.ctx.push(self.outputs_column, self.cloud_column)
                if not self._cloud_again:
                    break
                await asyncio.sleep(0.25)
        finally:
            self._cloud_refreshing = False

    @property
    def destination_label(self) -> str:
        return u10.destination_text(self.cloud_state.get("destination"))

    def cloud_entry(self, kind: str) -> Optional[dict]:
        entry = dict(self.book_cloud.get("files") or {}).get(kind)
        return dict(entry) if isinstance(entry, dict) else None

    def cloud_rows(self) -> dict:
        """``{output row index: cloud entry}``: each kind's entry goes on the row of the file it copies (the
        entry's ``path`` / ``source``), else on the first row of that kind (UI_SPEC §3.9 priority order)."""
        rows: dict = {}
        files = dict(self.book_cloud.get("files") or {})
        for kind, entry in files.items():
            if not isinstance(entry, dict):
                continue
            wanted = os.path.normcase(os.path.abspath(str(entry.get("path") or entry.get("source") or ""))) \
                if (entry.get("path") or entry.get("source")) else ""
            first = None
            chosen = None
            for index, (path, row_kind) in enumerate(self.outputs):
                if row_kind != kind:
                    continue
                if first is None:
                    first = index
                if wanted and os.path.normcase(os.path.abspath(str(path))) == wanted:
                    chosen = index
                    break
            index = chosen if chosen is not None else first
            if index is not None:
                rows[index] = dict(entry)
        return rows

    def cloud_line(self, index: int) -> Optional[tuple]:
        """``(icon, text, tone)`` under output row ``index`` (None: nothing to show)."""
        if not self.cloud_state.get("destination"):
            return None
        entry = self.cloud_rows().get(index)
        if entry is None:
            return None
        kind = self.outputs[index][1] if 0 <= index < len(self.outputs) else ""
        return u10.file_status_line(entry, dest_label=self.destination_label, kind=kind)

    def on_cloud_line(self, kind: str, entry: dict) -> Any:
        """Tap on a row's cloud line: pick a save location / reconnect / retry."""
        status = str(entry.get("status") or "")
        if status == "needs_pick":
            return self.ctx.spawn(self.choose_location(kind))
        if status in ("needs_relink", "revoked", "permission_lost"):
            return self.actions.open_settings()
        if status in ("failed", "error", "check", "missing"):
            return self.ctx.spawn(self.send_now())
        return None

    def cloud_reason(self) -> Optional[str]:
        return self.actions.cloud_reason(self.cloud_state, has_outputs=bool(self.outputs),
                                         in_library=bool(self.book_cloud.get("in_library", True)))

    def share_reason(self) -> Optional[str]:
        return self.actions.share_reason(self.providers, has_outputs=bool(self.actions.shareable(self.outputs)))

    async def send_now(self) -> dict:
        result = await self.actions.send_now(self.identity, self.cloud_state)
        await self.refresh_cloud()
        return result

    async def choose_location(self, kind: str) -> dict:
        result = await self.actions.choose_save_location(self.identity, kind)
        await self.refresh_cloud()
        return result

    def open_override(self) -> ActionSheet:
        current = str(self.book_cloud.get("override") or "default")
        sheet = self.actions.override_sheet(self.identity, current, on_done=lambda value: self.refresh_cloud())
        self.last_sheet = sheet
        return sheet

    async def share_link(self) -> Any:
        if self.stale:
            await self.reload()
        result = await self.actions.share_link(self.outputs, self.identity)
        if isinstance(result, ActionSheet):
            self.last_sheet = result
        return result

    def _cloud_controls(self, n: int) -> list:
        """The Cloud & sharing section (fresh keys per render)."""
        actions = self.actions
        state = self.cloud_state
        dest = state.get("destination")
        override = str(self.book_cloud.get("override") or "default")
        controls: list = []
        if dest:
            auto = bool(self.book_cloud.get("auto"))
            text = f"Copies go to {self.destination_label} · " + (
                "automatically after each compile" if auto else "only when you tap Send now")
            icon, tone = ("SYNC_PROBLEM", "error") if dest.get("needs_relink") else ("CLOUD_OUTLINED", "ok")
            if dest.get("needs_relink"):
                text = f"Glossarion lost access to {self.destination_label} · Reconnect in Settings"
            controls.append(u10.status_row(icon, text, tone, key=f"out-cloud-dest-{n}",
                                           on_click=lambda e: actions.open_settings()))
        else:
            reason = actions.cloud_reason(state)
            controls.append(ft.Row([
                ft.Text("No cloud destination yet" if reason == u10.NO_DESTINATION_REASON else "Cloud sync",
                        theme_style=ft.TextThemeStyle.BODY_MEDIUM, expand=True),
                ft.TextButton(content="Set up", icon=ft.Icons.SETTINGS_OUTLINED, on_click=lambda e: actions.open_settings(),
                              key=f"out-cloud-setup-{n}"),
            ], wrap=True, key=f"out-cloud-dest-{n}"))
        send_reason = self.cloud_reason()
        override_label = dict((v, l) for v, l, _t in u10.OVERRIDES).get(override, "Default")
        buttons: list = [
            self._u10_button("Send now", "CLOUD_UPLOAD", send_reason, lambda e: self.ctx.spawn(self.send_now()),
                             f"out-cloud-send-{n}"),
            self._u10_button(f"Copy this book: {override_label}", "TUNE",
                             u10.NO_SERVICE_REASON if not actions.cloud.available else None,
                             lambda e: self.open_override(), f"out-cloud-override-{n}", outlined=True),
            self._u10_button(u10.SHARE_LABEL, "LINK", self.share_reason(),
                             lambda e: self.ctx.spawn(self.share_link()), f"out-share-link-{n}"),
        ]
        controls.append(ft.Row(buttons, wrap=True, spacing=8, run_spacing=4, key=f"out-cloud-buttons-{n}"))
        if self.links:
            controls.append(ft.Text(f"Saved links ({len(self.links)})", theme_style=ft.TextThemeStyle.LABEL_LARGE,
                                    key=f"out-links-title-{n}"))
            controls.extend(actions.link_rows(self.links, on_changed=self.refresh_cloud, key_prefix=f"out-link-{n}"))
        return controls

    @staticmethod
    def _u10_button(label: str, icon: str, reason: Optional[str], on_click: Any, key: str,
                    outlined: bool = False) -> ft.Control:
        """An action button, or the disabled button with its ReasonChip (nothing hidden, UI_SPEC §0 item 6)."""
        if reason:
            return ft.Row([ft.OutlinedButton(content=label, icon=getattr(ft.Icons, icon, None), disabled=True),
                           ReasonChip(reason=reason, detail=u10.reason_detail(reason))], spacing=4, tight=True, key=key)
        if outlined:
            return ft.OutlinedButton(content=label, icon=getattr(ft.Icons, icon, None), on_click=on_click, key=key)
        return ft.FilledTonalButton(content=label, icon=getattr(ft.Icons, icon, None), on_click=on_click, key=key)
