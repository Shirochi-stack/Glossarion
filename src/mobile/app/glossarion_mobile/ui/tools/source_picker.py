"""SourcePicker (UI_SPEC §4.2, §5.5): what a tool runs on.

A ``BottomSheet`` with the segments **Recent outputs** · **Library books** · **Chat
workspaces** · **Browse**. Rows are ``targets.ToolTarget`` (output folder + raw source),
loaded on the io pool. Single mode picks on tap; multi mode has checkboxes and a "Use N"
button (a bulk QA scan, metadata for several EPUBs). Each tool passes ``eligible(target)``:
rows it cannot use stay visible, disabled, with the reason (QA skips Direct Text
workspaces, the Converter needs an output folder, metadata needs a raw EPUB...).

Browse picks EPUB / TXT / PDF files through ``FileBridge.pick_files`` (copied into the
Inbox) and finds each file's output folder with the QA Scanner's auto-search
(``targets.target_for_source``).
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional, Sequence

import flet as ft

from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.components.reason_chip import ReasonChip
from glossarion_mobile.ui.tools import targets as tg

__all__ = ["SEGMENTS", "SourcePicker"]

log = logging.getLogger("glossarion.tools")

SEGMENTS = (("recent", "Recent outputs"), ("library", "Library books"), ("chat", "Chat workspaces"),
            ("browse", "Browse"))
BROWSE_EXTENSIONS = ("epub", "txt", "pdf")


def _subtitle(target: tg.ToolTarget) -> str:
    parts = []
    if target.folder:
        parts.append(f"📁 {target.folder_name}")
    else:
        parts.append("No output folder yet")
    if target.source:
        parts.append(f"📖 {target.source_name}")
    elif target.origin != "chat":
        parts.append("No raw source")
    return " · ".join(parts)


class SourcePicker:
    def __init__(self, ctx: Any, *, title: str, multi: bool = False,
                 eligible: Optional[Callable[[tg.ToolTarget], Optional[str]]] = None,
                 on_done: Optional[Callable[[list], Any]] = None, selected: Sequence[tg.ToolTarget] = (),
                 segment: str = "recent", browse_label: str = "Pick a source file…",
                 find_folder: bool = True) -> None:
        self.ctx = ctx
        self.find_folder = find_folder  # Browse: look for the picked file's output folder (QA auto-search)
        self.title = title
        self.multi = multi
        self.eligible = eligible or (lambda t: None)
        self.on_done = on_done
        self.segment = segment if segment in dict(SEGMENTS) else "recent"
        self.rows: dict = {"recent": [], "library": [], "chat": [], "browse": []}
        self.loaded = False
        self.selected: dict = {t.key: t for t in selected}
        self.result: Optional[list] = None
        self.tiles: dict = {}
        self.segmented = ft.SegmentedButton(
            segments=[ft.Segment(value=value, label=ft.Text(label)) for value, label in SEGMENTS],
            selected=[self.segment], allow_multiple_selection=False, show_selected_icon=False,
            on_change=self._on_segment, key="picker-segments")
        self.list_view = ft.ListView(spacing=2, height=360, build_controls_on_demand=True, key="picker-list")
        self.status = ft.Text("", theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT,
                              key="picker-status")
        self.browse_button = ft.FilledTonalButton(content=browse_label, icon=ft.Icons.FILE_OPEN_OUTLINED,
                                                  on_click=self._on_browse, visible=self.segment == "browse",
                                                  key="picker-browse")
        self.done_button = ft.FilledButton(content=self._done_label(), on_click=self._on_done,
                                           visible=multi, disabled=not self.selected, key="picker-done")
        content = ft.Column([
            ft.Text(title, theme_style=ft.TextThemeStyle.TITLE_LARGE, weight=ft.FontWeight.W_600, key="picker-title"),
            ft.Row([self.segmented], scroll=ft.ScrollMode.AUTO),
            self.browse_button,
            self.status,
            self.list_view,
            ft.Row([ft.TextButton(content="Cancel", on_click=lambda e: self.close(), key="picker-cancel"),
                    self.done_button], alignment=ft.MainAxisAlignment.END),
        ], tight=True, spacing=tokens.SPACING["sm"])
        self.sheet = ft.BottomSheet(content=ft.Container(content=content, padding=tokens.SPACING["sheet_padding"]),
                                    show_drag_handle=True, scrollable=True, bgcolor=ft.Colors.SURFACE_CONTAINER_HIGH)
        self._page: Any = None

    # ---- lifecycle -------------------------------------------------------------------------------

    def show(self, page: Any) -> "SourcePicker":
        self._page = page
        page.show_dialog(self.sheet)
        self.ctx.spawn(self.load())
        return self

    def close(self) -> None:
        if self._page is not None and getattr(self.sheet, "open", False):
            self._page.pop_dialog()

    async def load(self) -> None:
        self.status.value = "Loading…"
        self.ctx.push(self.status)
        service = self.ctx.service
        if service is not None:
            snapshot = getattr(service, "snapshot", None)
            if snapshot is not None and not getattr(snapshot, "scanned_at", None) and hasattr(service, "refresh"):
                try:
                    await service.refresh(quiet=True, reason="tools source picker")
                except Exception:
                    log.debug("library refresh for the picker failed", exc_info=True)
        chats_root = self.ctx.chats_root

        def gather() -> dict:
            library = tg.library_targets(service) if service is not None else []
            return {
                "recent": tg.recent_output_targets(service, library_rows=library) if service is not None else [],
                "library": library,
                "chat": tg.chat_workspace_targets(chats_root),
            }

        try:
            found = await self.ctx.io(gather)
        except Exception as exc:
            log.exception("loading the source rows failed")
            found = {}
            self.status.value = f"Could not list sources: {exc}"
        for key, rows in (found or {}).items():
            self.rows[key] = list(rows)
        self.loaded = True
        self.render()

    # ---- rendering ----------------------------------------------------------------------------

    def _done_label(self) -> str:
        count = len(self.selected)
        return f"Use {count}" if count else "Use"

    def render(self) -> None:
        rows = self.rows.get(self.segment, [])
        self.tiles = {}
        controls: list = []
        for target in rows:
            controls.append(self._tile(target))
        self.list_view.controls = controls
        if not rows:
            empty = {"recent": "No translation workspaces yet.", "library": "The Library is empty.",
                     "chat": "No chat attachment workspaces.",
                     "browse": "Pick an EPUB, TXT or PDF file; its output folder is found automatically."}
            self.status.value = empty.get(self.segment, "")
        else:
            self.status.value = f"{len(rows)} item{'s' if len(rows) != 1 else ''}"
        self.browse_button.visible = self.segment == "browse"
        self.done_button.content = self._done_label()
        self.done_button.disabled = not self.selected
        self.ctx.push(self.list_view, self.status, self.browse_button, self.done_button)

    def _tile(self, target: tg.ToolTarget) -> ft.Control:
        reason = self.eligible(target)
        checked = target.key in self.selected
        leading: ft.Control
        if self.multi:
            leading = ft.Checkbox(value=checked, disabled=reason is not None,
                                  on_change=lambda e, t=target: self.toggle(t))
        else:
            leading = ft.Icon(ft.Icons.RADIO_BUTTON_CHECKED if checked else ft.Icons.RADIO_BUTTON_UNCHECKED)
        tile = ft.ListTile(
            leading=leading,
            title=ft.Text(target.title, max_lines=2, overflow=ft.TextOverflow.ELLIPSIS),
            subtitle=ft.Text(_subtitle(target), max_lines=2, overflow=ft.TextOverflow.ELLIPSIS,
                             theme_style=ft.TextThemeStyle.BODY_SMALL),
            trailing=ReasonChip(reason=reason if len(reason) <= 28 else reason[:26] + "…", detail=reason)
            if reason else None,
            disabled=reason is not None,
            on_click=lambda e, t=target: self.toggle(t),
            min_height=tokens.SIZES["hit_target"],
            key=f"pick-{target.key}",
        )
        self.tiles[target.key] = tile
        return tile

    # ---- actions ---------------------------------------------------------------------------------

    def _on_segment(self, e: Any = None) -> None:
        selected = list(getattr(getattr(e, "control", None), "selected", None) or self.segmented.selected or [])
        self.set_segment(selected[0] if selected else "recent")

    def set_segment(self, segment: str) -> None:
        self.segment = segment if segment in dict(SEGMENTS) else "recent"
        self.segmented.selected = [self.segment]
        self.ctx.push(self.segmented)
        self.render()

    def toggle(self, target: tg.ToolTarget) -> None:
        if self.eligible(target) is not None:
            return
        if not self.multi:
            self.selected = {target.key: target}
            self.finish()
            return
        if target.key in self.selected:
            del self.selected[target.key]
        else:
            self.selected[target.key] = target
        self.render()

    def finish(self) -> list:
        self.result = list(self.selected.values())
        self.close()
        if self.on_done is not None:
            from glossarion_mobile.ui.components._handlers import call_handler

            call_handler(self.on_done, list(self.result))
        return self.result

    def _on_done(self, e: Any = None) -> list:
        return self.finish()

    async def browse(self) -> list:
        files = self.ctx.files
        if files is None:
            self.ctx.say("Picking files is not available in this session")
            return []
        try:
            picked = await files.pick_files(target="inbox", allowed_extensions=list(BROWSE_EXTENSIONS),
                                            allow_multiple=self.multi, dialog_title=self.title)
        except Exception as exc:
            self.ctx.say(f"Could not pick the file: {exc}")
            return []
        paths = [getattr(f, "path", None) for f in picked or ()]
        paths = [p for p in paths if p]
        if not paths:
            return []
        output_root = self.ctx.output_root

        find_folder = self.find_folder

        def resolve() -> list:
            if not find_folder:
                return [tg.target_for_source(p, candidates=lambda *a, **k: []) for p in paths]
            return [tg.target_for_source(p, output_root=output_root or None) for p in paths]

        added = await self.ctx.io(resolve)
        for target in added:
            self.rows["browse"] = [t for t in self.rows["browse"] if t.key != target.key] + [target]
        if self.multi:
            for target in added:
                if self.eligible(target) is None:
                    self.selected[target.key] = target
            self.render()
        elif added and self.eligible(added[0]) is None:
            self.selected = {added[0].key: added[0]}
            self.render()
            self.finish()
        else:
            self.render()
        return added

    async def _on_browse(self, e: Any = None) -> list:
        return await self.browse()
