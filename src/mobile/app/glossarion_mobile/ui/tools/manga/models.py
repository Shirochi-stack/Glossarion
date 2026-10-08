"""Model download rows and the model manager sheet (UI_SPEC §5.10 ``ModelDownloadRow``).

A row shows the model name, size and status chip (Not downloaded / Downloading NN% /
Downloaded / Loaded) with Download (Cancel while downloading), Load / Unload and Delete. All of
it is ``services.manga.ModelManager`` over the shared ``manga_models`` registry; downloads run on
the io pool (resume + checksum in the core) and their progress is posted back to the UI loop.
Models are never bundled: the first use asks for the download.
"""

from __future__ import annotations

import itertools
import logging
from typing import Any, Callable, Optional, Sequence

import flet as ft

from glossarion_mobile.services import manga as svc
from glossarion_mobile.ui.components.reason_chip import ReasonChip
from glossarion_mobile.ui.components.sheet import bottom_sheet, scroll_column, sheet_frame
from glossarion_mobile.ui.theme import HIT_TARGET
from glossarion_mobile.ui.tools.manga.common import option_chip, push

__all__ = ["ModelDownloadRow", "ModelManagerSheet", "ensure_models"]

log = logging.getLogger("glossarion.tools.manga")
_SHEETS = itertools.count(1)  # a new key per sheet (never a reused id())


async def ensure_models(ctx: Any, manager: svc.ModelManager, config: dict, *, kinds: Optional[Sequence[str]] = None,
                        action: str = "This run") -> bool:
    """Before a run or an editor step: the models it loads (``manga_models.missing_models`` for the
    config, limited to ``kinds``) that are not on the device yet. The first use asks — Download
    (now, with progress in the model sheet), Download in the run (the job downloads them first,
    verified and resumable, with its progress and Stop: ``services.manga.ensure_run_models``) or
    Cancel. True: go on."""
    if not manager.available:
        return True
    try:
        missing = list(await ctx.io(manager.required, dict(config or {})))
    except Exception:
        log.debug("checking the required models failed", exc_info=True)
        return True
    entries = [manager.status(model_id) for model_id in missing]
    entries = [e for e in entries if kinds is None or e.kind in kinds]
    if not entries:
        return True
    from glossarion_mobile.ui.tools.common import ask

    size = svc.ModelEntry("total", "total", size=sum(int(e.size or 0) for e in entries)).size_label
    names = ", ".join(e.label for e in entries)
    answer = await ask(ctx, "Download models",
                       f"{action} needs {names}{f' ({size})' if size else ''}, not downloaded on this device yet. "
                       "Download now? Models are kept for later runs.",
                       (("cancel", "Cancel", "text"), ("run", "Download in the run", "text"),
                        ("download", "Download", "filled")),
                       key="manga-models-confirm")
    if answer == "run":
        return True
    if answer != "download":
        return False
    sheet = ModelManagerSheet(ctx, manager)
    ctx.extras["manga_models_sheet"] = sheet
    sheet.show(ctx.page)
    for entry in entries:
        row = sheet.rows.get(entry.id) or ModelDownloadRow(ctx, manager, entry.id, key_prefix=f"mm-need-{entry.id}")
        result = await row.download()
        if result.status not in ("ready", "loaded"):
            return False
    return True


class ModelDownloadRow:
    """One registry model: status chip + Download / Cancel / Load / Delete (``key_prefix`` keys).

    ``on_change(entry)`` runs when the status changes (missing → downloading → ready / error,
    loading, loaded, deleted), never for download progress: a progress report re-renders this
    row only (an owner that rebuilds itself on every report would rebuild for the whole
    download, up to ten times a second)."""

    def __init__(self, ctx: Any, manager: svc.ModelManager, model_id: str, *, title: Optional[str] = None,
                 key_prefix: str = "mdl", on_change: Optional[Callable[[svc.ModelEntry], Any]] = None,
                 allow_load: bool = True) -> None:
        self.ctx = ctx
        self.manager = manager
        self.model_id = model_id
        self.title = title
        self.key_prefix = key_prefix
        self.on_change = on_change
        self.allow_load = allow_load
        self._downloading = False  # this row's own download is running (its reports keep it current)
        self.entry: svc.ModelEntry = manager.status(model_id)
        self.title_text = ft.Text("", theme_style=ft.TextThemeStyle.BODY_MEDIUM, weight=ft.FontWeight.W_600,
                                  max_lines=2, overflow=ft.TextOverflow.ELLIPSIS)
        self.sub_text = ft.Text("", theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT)
        self.chip_holder = ft.Container()
        self.bar = ft.ProgressBar(value=0, visible=False, key=f"{key_prefix}-bar")
        self.download_button = ft.IconButton(icon=ft.Icons.DOWNLOAD, tooltip="Download", size_constraints=HIT_TARGET,
                                             on_click=lambda e: self.ctx.spawn(self.download()),
                                             key=f"{key_prefix}-download")
        self.cancel_button = ft.IconButton(icon=ft.Icons.CLOSE, tooltip="Cancel download", size_constraints=HIT_TARGET,
                                           on_click=lambda e: self.cancel(), key=f"{key_prefix}-cancel")
        self.load_button = ft.IconButton(icon=ft.Icons.MEMORY, tooltip="Load", size_constraints=HIT_TARGET,
                                         on_click=lambda e: self.ctx.spawn(self.toggle_load()),
                                         key=f"{key_prefix}-load")
        self.delete_button = ft.IconButton(icon=ft.Icons.DELETE_OUTLINE, tooltip="Delete", size_constraints=HIT_TARGET,
                                           on_click=lambda e: self.ctx.spawn(self.delete()),
                                           key=f"{key_prefix}-delete")
        self.control = ft.Container(
            content=ft.Column([
                ft.Row([
                    ft.Icon(ft.Icons.MODEL_TRAINING, color=ft.Colors.PRIMARY),
                    ft.Column([self.title_text, self.sub_text, self.chip_holder], spacing=2, tight=True, expand=True),
                    self.download_button, self.cancel_button, self.load_button, self.delete_button,
                ], spacing=8, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                self.bar,
            ], spacing=4, tight=True),
            padding=ft.Padding.symmetric(horizontal=4, vertical=4),
            key=f"{key_prefix}-row",
        )
        self.render()

    def render(self) -> None:
        entry = self.entry
        self.title_text.value = self.title or entry.label
        parts = [p for p in (entry.size_label, entry.error if entry.status == "error" else "") if p]
        self.sub_text.value = " · ".join(parts)
        # a new chip per render, under a new key (Flet 1.0.3 freezes a replacement with the same key)
        self._renders = getattr(self, "_renders", 0) + 1
        if entry.status == "unavailable":
            self.chip_holder.content = ReasonChip(reason=entry.chip, detail=entry.reason or entry.chip,
                                                  key=f"{self.key_prefix}-reason-{self._renders}")
        else:
            self.chip_holder.content = option_chip(entry.chip, entry.status,
                                                   key=f"{self.key_prefix}-chip-{self._renders}")
        downloading = entry.status == "downloading"
        have = entry.status in ("ready", "loaded", "loading")
        self.bar.visible = downloading
        self.bar.value = entry.progress or 0.0
        self.download_button.visible = entry.status in ("missing", "error")
        self.cancel_button.visible = downloading
        can_load = self.allow_load and self.manager.can_load(self.model_id)
        self.load_button.visible = have and can_load
        self.load_button.disabled = not can_load or entry.status == "loading"
        self.load_button.icon = ft.Icons.MEMORY if entry.status != "loaded" else ft.Icons.EJECT
        self.load_button.tooltip = ("Unload" if entry.status == "loaded" else "Load") if can_load else \
            "Loading ahead is not available in this build"
        self.delete_button.visible = have
        self.download_button.disabled = entry.status == "unavailable"

    def update(self, entry: Optional[svc.ModelEntry] = None) -> None:
        previous = self.entry.status
        self.entry = entry if entry is not None else self.manager.status(self.model_id)
        self.render()
        push(self.control)
        if self.on_change is not None and self.entry.status != previous:
            try:
                self.on_change(self.entry)
            except Exception:
                log.debug("model row change handler failed", exc_info=True)

    @property
    def downloading(self) -> bool:
        """This row's own download is running."""
        return self._downloading

    def sync(self) -> None:
        """Re-read the registry status, for an owner that keeps this row across a rebuild of its
        own (no push, no ``on_change``); not while this row's download runs (its reports are newer)."""
        if self._downloading:
            return
        self.entry = self.manager.status(self.model_id)
        self.render()

    def _post(self, entry: svc.ModelEntry) -> None:
        """Download progress (io thread) -> the UI loop."""
        dispatcher = getattr(self.ctx, "dispatcher", None)
        post = getattr(dispatcher, "post", None) if dispatcher is not None else None
        if callable(post) and getattr(dispatcher, "bound", False):
            post(self._progress, entry)
        else:
            self._progress(entry)

    def _progress(self, entry: svc.ModelEntry) -> None:
        if self._downloading:  # a report the loop gets after the download returned is stale
            self.update(entry)

    async def download(self) -> svc.ModelEntry:
        self._downloading = True
        try:
            self.update(svc.ModelEntry(self.model_id, self.entry.label, kind=self.entry.kind, size=self.entry.size,
                                       status="downloading", progress=0.0))
            entry = await self.ctx.io(self.manager.download, self.model_id, self._post)
        finally:
            self._downloading = False
        self.update(entry)
        if entry.status == "error":
            self.ctx.say(f"Download failed: {entry.error}")
        elif entry.status in ("ready", "loaded"):
            self.ctx.say(f"Downloaded {entry.label}")
        return entry

    def cancel(self) -> bool:
        return self.manager.cancel(self.model_id)

    async def toggle_load(self) -> svc.ModelEntry:
        if self.entry.status == "loaded":
            entry = await self.ctx.io(self.manager.unload, self.model_id)
        else:
            self.update(svc.ModelEntry(self.model_id, self.entry.label, kind=self.entry.kind, size=self.entry.size,
                                       status="loading", path=self.entry.path))
            entry = await self.ctx.io(self.manager.load, self.model_id)
            if entry.status == "error":
                self.ctx.say(f"Loading failed: {entry.error}")
        self.update(entry)
        return entry

    async def delete(self) -> svc.ModelEntry:
        from glossarion_mobile.ui.tools.common import ask

        answer = await ask(self.ctx, "Delete model", f"Delete {self.entry.label} from this device? "
                           "It downloads again when a run needs it.",
                           (("no", "Cancel", "text"), ("yes", "Delete", "destructive")), key="mdl-delete-confirm")
        if answer != "yes":
            return self.entry
        entry = await self.ctx.io(self.manager.delete, self.model_id)
        self.update(entry)
        return entry


class ModelManagerSheet:
    """Every downloadable model (detector, inpainting, OCR) with its row and the disk usage."""

    def __init__(self, ctx: Any, manager: svc.ModelManager, *, on_change: Optional[Callable[[Any], Any]] = None) -> None:
        self.ctx = ctx
        self.manager = manager
        self.on_change = on_change
        self.rows: dict = {}
        self.usage = ft.Text("", theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT,
                             key="mm-usage")
        controls: list = []
        groups = (("detector", "Bubble detection"), ("inpaint", "Inpainting"), ("ocr", "OCR"))
        entries = manager.entries()
        for kind, title in groups:
            members = [e for e in entries if e.kind == kind]
            if not members:
                continue
            controls.append(ft.Text(title, theme_style=ft.TextThemeStyle.TITLE_SMALL, color=ft.Colors.PRIMARY))
            for entry in members:
                row = ModelDownloadRow(ctx, manager, entry.id, key_prefix=f"mm-{entry.id}",
                                       on_change=self._changed, allow_load=kind != "ocr")
                self.rows[entry.id] = row
                controls.append(row.control)
        others = [e for e in entries if e.kind not in dict(groups)]
        for entry in others:
            row = ModelDownloadRow(ctx, manager, entry.id, key_prefix=f"mm-{entry.id}", on_change=self._changed)
            self.rows[entry.id] = row
            controls.append(row.control)
        if not controls:
            controls.append(ReasonChip(reason="No model registry", detail=svc.MISSING_CORE + " (manga_models)."))
        self._refresh_usage()
        # components.sheet: a scrolling body inset above the navigation bar (owner device fix 57f1835c)
        self.sheet = bottom_sheet(sheet_frame(scroll_column([
            ft.Text("Models", theme_style=ft.TextThemeStyle.TITLE_MEDIUM, weight=ft.FontWeight.W_600),
            self.usage,
            ft.Text("Models download on demand from Hugging Face and stay on this device.",
                    theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT),
            *controls,
        ], spacing=8)), key=f"manga-models-sheet-{next(_SHEETS)}")

    def _refresh_usage(self) -> None:
        try:
            used = int(self.manager.disk_usage() or 0)
        except Exception:
            used = 0
        entry = svc.ModelEntry("usage", "usage", size=used)
        self.usage.value = f"On this device: {entry.size_label or '0 B'}"

    def _changed(self, entry: Any) -> None:
        self._refresh_usage()
        push(self.usage)
        if self.on_change is not None:
            self.on_change(entry)

    def show(self, page: Any) -> "ModelManagerSheet":
        if page is not None:
            page.show_dialog(self.sheet)
        return self
