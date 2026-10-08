"""Manga › Files (UI_SPEC §4.6 Files).

* Sources: **Add files** (images), **Add ZIP/CBZ** (extracted, its pages appended in natural
  order, the CBZ job kept for "Create CBZ at end") and **Add folder** on both platforms
  (``FilePicker.get_directory_path`` + copy into the Inbox; when Android's SAF tree cannot be
  read, ``FolderPickUnavailable`` offers "Pick a .zip instead").
* The list: thumbnail rows in run order, drag to reorder (``ReorderableListView``), Sort
  (Name / Number / Date / Reverse, ascending or descending), a per-file "Process this image"
  switch (the desktop skip marker) and long-press selection with Remove selected · Clear all.
  Order, skips and folder roots persist in the desktop keys (``manga_selected_files``,
  ``manga_skipped_processing_files``, ``manga_selected_folder_roots``).
* Image range ("blank = all, e.g. 3-8") with the desktop status line, process grouping
  (split first-level subfolders into groups; a group picker when there are several).
* Run: Start / Stop (a ``manga`` job; the second Stop forces), progress, the job's LogConsole;
  Create CBZ at end · Auto consolidate (config switches the runner reads). Start first offers
  the download of a model the run loads that is not on the device yet (``ensure_models``).
  **Import OCR** (the editor's import: pages get their OCR / translations / boxes back) is also
  the OCR the next Start reuses, as on the desktop ("Imported OCR (N)", clearable).
* Output: **Create CBZ** (the translated pages packed in run order; pages of an imported CBZ
  go back into ``<name>_translated.cbz`` next to it) · **Download images** (a ZIP of them)
  through the ExportSheet, the CBZ archives the run or Create CBZ wrote (tap to share / save),
  and "Open in editor". Translated pages sit next to their source copy in app storage
  (``services.manga.hide_output_override``). While another job owns the process state the
  lookup of earlier outputs waits ("Wait for the running job") and runs again when a job ends.
* A batch that ended while no Files tab watched it (the screen was closed during the run) is
  applied when the tab shows again: run status, pages, archives. After a batch, and after a
  selection change made while a job ran, the selection's glossary auto-load runs again, so
  Settings › Glossary shows the glossary the run generated.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import flet as ft

from glossarion_mobile.services import manga as svc
from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.theme import HIT_TARGET
from glossarion_mobile.ui.tools.common import JobWatch, action_button, hint_text
from glossarion_mobile.ui.tools.manga.common import JobEnds, MangaTab, export_sheet, push
from glossarion_mobile.ui.tools.manga.models import ensure_models

__all__ = ["FilesTab", "SORT_LABELS"]

log = logging.getLogger("glossarion.tools.manga")

SORT_LABELS = (("name", "Name"), ("numeric", "Number"), ("date", "Date"), ("reverse", "Reverse"))
FOLDER_FALLBACK = "Pick a .zip instead"
RANGE_HINT = "blank = all, e.g. 3-8"
THUMB = 48


class FilesTab(MangaTab):
    key = "manga-files"

    def __init__(self, ctx: Any, session: Any, *, screen: Any = None) -> None:
        super().__init__(ctx, session, screen=screen)
        self.selected: set = set()
        self.selection_mode = False
        self.sort_reverse = False
        self.watch = JobWatch(ctx, self._on_job_end, self._on_job_change)
        self.job_ends = JobEnds(ctx, self._on_any_job_end)
        self._outputs_pending = False  # the earlier-outputs lookup was refused by a running job
        self._unsub_view: Any = None
        self.console: Any = None
        self.console_job: Optional[str] = None
        self.group_index: Optional[int] = None
        self.rows: dict = {}
        self._build_gen = 0

    # ---- build -------------------------------------------------------------------------------------

    def build(self) -> ft.Control:
        files = self.ctx.files
        self.add_files_button = ft.FilledTonalButton(content="Add files", icon=ft.Icons.ADD_PHOTO_ALTERNATE,
                                                     on_click=lambda e: self.ctx.spawn(self.pick_images()),
                                                     disabled=files is None, key="mf-add-files")
        self.add_archive_button = ft.FilledTonalButton(content="Add ZIP/CBZ", icon=ft.Icons.FOLDER_ZIP,
                                                       on_click=lambda e: self.ctx.spawn(self.pick_archive()),
                                                       disabled=files is None, key="mf-add-zip")
        self.add_folder_button = ft.FilledTonalButton(content="Add folder", icon=ft.Icons.CREATE_NEW_FOLDER,
                                                      on_click=lambda e: self.ctx.spawn(self.pick_folder()),
                                                      disabled=files is None, key="mf-add-folder")
        self.summary = ft.Text("", theme_style=ft.TextThemeStyle.BODY_SMALL, key="mf-summary")
        self.sort_buttons = ft.SegmentedButton(
            segments=[ft.Segment(value=v, label=ft.Text(label)) for v, label in SORT_LABELS],
            selected=[], allow_empty_selection=True, show_selected_icon=False,
            on_change=self._on_sort, key="mf-sort")
        self.sort_direction = ft.IconButton(icon=ft.Icons.ARROW_UPWARD, tooltip="Ascending",
                                            on_click=self._toggle_direction, size_constraints=HIT_TARGET,
                                            key="mf-sort-dir")
        self.range_field = ft.TextField(label="Image range", hint_text=RANGE_HINT, dense=True, width=180,
                                        value=self.session.files.image_range, on_change=self._on_range,
                                        key="mf-range")
        self.range_status = ft.Text("All images", theme_style=ft.TextThemeStyle.BODY_SMALL,
                                    color=ft.Colors.ON_SURFACE_VARIANT, key="mf-range-status")
        self.split_switch = ft.Switch(label="Split first-level subfolders into groups",
                                      value=self.session.files.split_first_level, on_change=self._on_split,
                                      key="mf-split")
        self.group_picker = ft.Dropdown(label="Process group", options=[], visible=False, dense=True,
                                        on_select=self._on_group, key="mf-group")
        self.selection_bar = ft.Row([
            ft.Text("", key="mf-sel-count"),
            ft.TextButton(content="Remove selected", icon=ft.Icons.DELETE_OUTLINE, on_click=self._remove_selected,
                          key="mf-remove-selected"),
            ft.TextButton(content="Cancel", on_click=lambda e: self.exit_selection(), key="mf-sel-cancel"),
        ], visible=False, wrap=True, key="mf-selection-bar")
        self.clear_button = ft.TextButton(content="Clear all", icon=ft.Icons.CLEAR_ALL,
                                          on_click=lambda e: self.ctx.spawn(self.clear_all()), key="mf-clear")
        self.list_view = ft.ReorderableListView(controls=[], on_reorder=self._on_reorder, height=360,
                                                show_default_drag_handles=True, key="mf-list")
        self.empty = hint_text("No images yet. Add files, a ZIP / CBZ or a folder.", key="mf-empty")
        cfg = self.ctx.config_snapshot()
        # the tab's defaults (manga_settings_defaults: both on, like the desktop checkboxes)
        self.create_cbz_switch = ft.Switch(label="Create CBZ at end",
                                           value=bool(svc.effective_setting(cfg, svc.K_CREATE_CBZ, True)),
                                           on_change=lambda e: self.ctx.set_cfg(svc.K_CREATE_CBZ, bool(e.control.value)),
                                           key="mf-create-cbz")
        self.consolidate_switch = ft.Switch(label="Auto consolidate images at translation end",
                                            value=bool(svc.effective_setting(cfg, svc.K_CONSOLIDATE, True)),
                                            on_change=lambda e: self.ctx.set_cfg(svc.K_CONSOLIDATE,
                                                                                 bool(e.control.value)),
                                            key="mf-consolidate")
        self.start_button = ft.FilledButton(content="Start", icon=ft.Icons.PLAY_ARROW,
                                            on_click=lambda e: self.ctx.spawn(self.start()), key="mf-start")
        self.stop_button = ft.OutlinedButton(content="Stop", icon=ft.Icons.STOP, visible=False,
                                             on_click=self._on_stop, key="mf-stop")
        self.import_ocr_button = ft.TextButton(content="Import OCR", icon=ft.Icons.FILE_UPLOAD, key="mf-import-ocr",
                                               tooltip="Reuse the OCR (and translations) of an exported session",
                                               on_click=lambda e: self.ctx.spawn(self.import_ocr()))
        self.imported_row = ft.Row([], wrap=True, spacing=4, visible=False, key="mf-imported-ocr",
                                   vertical_alignment=ft.CrossAxisAlignment.CENTER)
        self.progress = ft.ProgressBar(value=None, visible=False, key="mf-progress")
        self.run_status = ft.Text("", theme_style=ft.TextThemeStyle.BODY_SMALL, key="mf-run-status")
        self.console_holder = ft.Container(key="mf-console-holder")
        self.cbz_button = action_button("Create CBZ", "ARCHIVE", lambda e: self.ctx.spawn(self.create_cbz()),
                                        key="mf-cbz", reason="Translate pages first")
        self.download_button = action_button("Download images", "DOWNLOAD",
                                             lambda e: self.ctx.spawn(self.download_images()),
                                             key="mf-download", reason="Translate pages first")
        self.output_holder = ft.Row([self.cbz_button, self.download_button], wrap=True, spacing=8,
                                    key="mf-output-actions")
        self.output_list = ft.Column([], spacing=2, key="mf-output-list")

        sources = self.section("Images", [
            ft.Row([self.add_files_button, self.add_archive_button, self.add_folder_button], wrap=True, spacing=8),
            self.summary,
            ft.Row([self.sort_buttons, self.sort_direction], wrap=True, spacing=4),
            ft.Row([self.range_field, self.range_status], wrap=True, spacing=8,
                   vertical_alignment=ft.CrossAxisAlignment.CENTER),
            self.split_switch,
            self.group_picker,
            ft.Row([self.clear_button], wrap=True),
            self.selection_bar,
            self.empty,
            self.list_view,
            hint_text("Drag to reorder · tap the switch to skip a page · long-press to select"),
        ], icon="PHOTO_LIBRARY", key="mf-images")
        run = self.section("Run", [
            ft.Row([self.start_button, self.stop_button, self.import_ocr_button], wrap=True, spacing=8),
            self.imported_row,
            self.progress,
            self.run_status,
            self.create_cbz_switch,
            self.consolidate_switch,
            self.console_holder,
        ], icon="PLAY_CIRCLE", key="mf-run")
        output = self.section("Output", [self.output_holder, self.output_list], icon="IOS_SHARE", key="mf-output")
        self.root = ft.ListView(controls=[sources, run, output], expand=True, spacing=tokens.SPACING["md"],
                                padding=ft.Padding.symmetric(horizontal=12, vertical=8), key=self.key)
        self.refresh(push_now=False)
        return self.root

    # ---- lifecycle ---------------------------------------------------------------------------------

    def did_show(self) -> None:
        self.job_ends.start()
        for snap in self.watch.adopt((svc.KIND_BATCH,)):
            self.session.batch_job_id = getattr(snap, "id", None)
            self._on_job_change(snap)
        if self.session.batch_job_id and self._unsub_view is None:
            self._subscribe_view()
        self._catch_up_job_end()
        self._retry_lookups()

    def on_session_loaded(self) -> None:
        self.refresh()
        if not self.session.last_outputs and self.session.files.files:
            self.ctx.spawn(self._find_outputs())

    def _catch_up_job_end(self) -> None:
        """The session's batch ended while no Files tab watched it (the screen was closed during the
        run; ``JobWatch.adopt`` only takes live jobs): its end is applied now, once per job (run
        status, translated pages, the archives it wrote, the glossary auto-load)."""
        job_id = self.session.batch_job_id
        if not job_id or job_id == self.session.batch_end_applied or job_id in self.watch.ended:
            return
        jobs = self.ctx.jobs
        snapshot = getattr(jobs, "snapshot", None) if jobs is not None else None
        if not callable(snapshot):
            return
        try:
            snap = snapshot(job_id)
        except Exception:
            snap = None
        if snap is None or not getattr(snap, "is_terminal", False):
            return
        self.watch.ended[job_id] = snap
        self._on_job_end(snap)

    def _retry_lookups(self) -> None:
        """Lookups a running job refused (earlier outputs; the glossary auto-load of a selection
        change), tried again; each stays pending while a job still owns the process state."""
        if self._outputs_pending and self.session.files.files:
            self.ctx.spawn(self._find_outputs())
        if self.session.files.glossary_stale:
            self.ctx.spawn(self._refresh_glossary())

    def _on_any_job_end(self, snap: Any) -> None:
        """A job of any kind ended (``JobEnds``): it no longer owns the process state."""
        if getattr(snap, "id", None) in self.watch.job_ids:
            # this tab's batch: ``_on_job_end`` applies it (and re-runs the glossary auto-load)
            if self._outputs_pending and self.session.files.files:
                self.ctx.spawn(self._find_outputs())
            return
        self._retry_lookups()

    async def _find_outputs(self) -> list:
        """Pages translated by an earlier run (the run's per-page output rule) enable the output
        actions; archives it packed for the selection are listed again. While another job owns
        the process state the lookup waits (the actions say so) and runs again when a job ends."""
        try:
            found = await self.ctx.io(self.session.files.existing_outputs)
            archives = await self.ctx.io(self.session.files.existing_cbz)
        except svc.MangaBusy:
            if not self._outputs_pending:
                self._outputs_pending = True
                self.refresh()
            return []
        except Exception:
            found, archives = [], []
        changed = self._outputs_pending
        self._outputs_pending = False
        if found and not self.session.last_outputs:
            self.session.last_outputs = list(found)
            changed = True
        if archives and not self.session.last_cbz:
            self.session.last_cbz = list(archives)
            changed = True
        if changed:
            self.refresh()
        return found

    def dispose(self) -> None:
        self.watch.stop()
        self.job_ends.stop()
        if self._unsub_view is not None:
            try:
                self._unsub_view()
            except Exception:
                pass
            self._unsub_view = None

    # ---- rendering ---------------------------------------------------------------------------------

    def _row(self, index: int, path: str) -> ft.Control:
        files = self.session.files
        skipped = files.is_skipped(path)
        out_of_range = files.range_skipped(path)
        selected = path in self.selected
        name = os.path.basename(path)
        cbz = files.cbz_job_for(path)
        sub = os.path.basename(cbz) if cbz else os.path.basename(os.path.dirname(path))
        thumb = ft.Container(content=ft.Image(src=path, width=THUMB, height=THUMB, fit=ft.BoxFit.COVER,
                                              cache_width=THUMB * 2, border_radius=6, gapless_playback=True),
                             width=THUMB, height=THUMB)
        texts = ft.Column([
            ft.Text(f"{index + 1}. {name}", max_lines=1, overflow=ft.TextOverflow.ELLIPSIS,
                    theme_style=ft.TextThemeStyle.BODY_MEDIUM,
                    color=ft.Colors.ON_SURFACE_VARIANT if (skipped or out_of_range) else None),
            ft.Text(("⏭️ Skipped · " if skipped else ("Outside the range · " if out_of_range else "")) + sub,
                    max_lines=1, overflow=ft.TextOverflow.ELLIPSIS, theme_style=ft.TextThemeStyle.BODY_SMALL,
                    color=ft.Colors.ON_SURFACE_VARIANT),
        ], spacing=0, tight=True, expand=True)
        switch = ft.Switch(value=not skipped, tooltip="Process this image",
                           on_change=lambda e, p=path: self.toggle_skip(p), key=f"mf-skip-{self._build_gen}-{index}")
        row = ft.Container(
            content=ft.Row([thumb, texts, switch], spacing=10, vertical_alignment=ft.CrossAxisAlignment.CENTER),
            padding=ft.Padding.symmetric(horizontal=8, vertical=4),
            bgcolor=ft.Colors.SECONDARY_CONTAINER if selected else None,
            border_radius=tokens.RADII["field"],
            opacity=0.6 if (skipped or out_of_range) else 1.0,
            on_click=lambda e, p=path: self._on_row_tap(p),
            on_long_press=lambda e, p=path: self._on_row_long_press(p),
            key=f"mf-row-{self._build_gen}-{index}",
        )
        self.rows[path] = row
        return row

    def refresh(self, push_now: bool = True) -> None:
        if self.root is None:
            return
        files = self.session.files
        self._build_gen += 1
        self.rows = {}
        paths = files.files
        groups = files.groups() if paths else []
        if self.group_index is not None and self.group_index >= len(groups):
            self.group_index = None
        shown = set(groups[self.group_index].files) if self.group_index is not None and len(groups) > 1 else None
        self.list_view.controls = [self._row(i, p) for i, p in enumerate(paths) if shown is None or p in shown]
        # drag reorder works on the full visible order; a group view only lists that group's pages
        self.list_view.show_default_drag_handles = shown is None
        self.list_view.visible = bool(paths)
        self.empty.visible = not paths
        run_files, error = files.run_files()
        self.summary.value = (f"{len(paths)} image{'s' if len(paths) != 1 else ''} · {len(run_files)} will run"
                              if paths else "")
        self.range_status.value = files.range_status()
        self.range_status.color = ft.Colors.ERROR if error else ft.Colors.ON_SURFACE_VARIANT
        self.range_field.error_text = None
        self.group_picker.visible = len(groups) > 1
        self.group_picker.options = [ft.DropdownOption(key="", text=f"All groups ({len(groups)})")] + [
            ft.DropdownOption(key=str(i), text=f"{i + 1}/{len(groups)} · {g.name} ({len(g.files)})")
            for i, g in enumerate(groups)]
        self.group_picker.value = str(self.group_index) if self.group_index is not None else ""
        sel_count = self.selection_bar.controls[0]
        sel_count.value = f"{len(self.selected)} selected"
        self.selection_bar.visible = self.selection_mode
        self.clear_button.disabled = not paths
        self.import_ocr_button.disabled = not paths
        self._render_imported()
        self._render_run_state()
        self._render_outputs()
        if push_now:
            push(self.root)

    def _imported_ocr_path(self) -> str:
        info = self.session.imported_ocr or {}
        path = str(info.get("path") or "")
        return path if path and os.path.isfile(path) else ""

    def _render_imported(self) -> None:
        """The imported OCR the next Start reuses (desktop "📥 Imported OCR (N)")."""
        info = self.session.imported_ocr or {}
        path = self._imported_ocr_path()
        self.imported_row.visible = bool(path)
        self.imported_row.controls = [
            ft.Icon(ft.Icons.TEXT_SNIPPET_OUTLINED, size=16),
            hint_text(f"Imported OCR ({info.get('matched', 0)}) · {os.path.basename(path)} · reused by Start"),
            ft.IconButton(icon=ft.Icons.CLOSE, icon_size=16, tooltip="Do not reuse it", size_constraints=HIT_TARGET,
                          on_click=lambda e: self.clear_imported_ocr(), key=f"mf-imported-clear-{self._build_gen}"),
        ] if path else []

    def clear_imported_ocr(self) -> None:
        self.session.imported_ocr = None
        self.refresh()

    async def import_ocr(self) -> Optional[str]:
        """Import OCR (desktop batch "📥 Import OCR"): the editor's import, which restores the pages'
        OCR / translations / boxes and is reused by the next Start."""
        if not self.session.files.files:
            self.ctx.say("Load the manga images before importing OCR text.")
            return None
        editor = getattr(self.screen, "editor_tab", None) if self.screen is not None else None
        if editor is None:
            self.ctx.say("The manga editor is not available")
            return None
        return await editor.import_ocr()

    def _render_run_state(self) -> None:
        snap = self.watch.active()
        running = snap is not None
        files = self.session.files
        run_files, error = files.run_files()
        reason = None
        if not self.ctx.has_kind(svc.KIND_BATCH):
            reason = "Manga jobs are not available in this build"
        elif error:
            reason = str(error)
        elif not run_files:
            reason = "Add images (or switch some back on)"
        self.start_button.disabled = running or reason is not None
        self.start_button.tooltip = reason
        self.stop_button.visible = running
        self.progress.visible = running
        if running:
            progress = getattr(snap, "progress", None)
            fraction = getattr(progress, "fraction", None)
            self.progress.value = fraction
            label = getattr(progress, "label", "") or getattr(snap, "phase", "") or "Queued…"
            state = getattr(getattr(snap, "state", None), "value", "")
            if state in ("STOPPING", "FORCE_STOPPING"):
                label = "Stopping… (tap Stop again to force)" if state == "STOPPING" else "Force stopping…"
            self.run_status.value = label
        elif reason and not self.run_status.value:
            self.run_status.value = ""

    def _render_outputs(self) -> None:
        outputs = [p for p in self.session.last_outputs if os.path.isfile(p)]
        # "not known yet" (a running job refused the lookup) is not "not translated"
        reason = None if outputs else ("Wait for the running job" if self._outputs_pending else "Translate pages first")
        self.cbz_button = action_button("Create CBZ", "ARCHIVE", lambda e: self.ctx.spawn(self.create_cbz()),
                                        key=f"mf-cbz-{self._build_gen}", reason=reason)
        self.download_button = action_button("Download images", "DOWNLOAD",
                                             lambda e: self.ctx.spawn(self.download_images()),
                                             key=f"mf-download-{self._build_gen}", reason=reason)
        editor = ft.TextButton(content="Open in editor", icon=ft.Icons.EDIT, key=f"mf-open-editor-{self._build_gen}",
                               on_click=lambda e: self.screen.select_tab("editor") if self.screen else None,
                               disabled=not self.session.files.files)
        self.output_holder.controls = [self.cbz_button, self.download_button, editor]
        rows = []
        for path in [p for p in self.session.last_cbz if os.path.isfile(p)]:
            rows.append(ft.ListTile(
                leading=ft.Icon(ft.Icons.ARCHIVE), title=ft.Text(os.path.basename(path), max_lines=1,
                                                                 overflow=ft.TextOverflow.ELLIPSIS),
                subtitle=ft.Text("CBZ · tap to share or save", theme_style=ft.TextThemeStyle.BODY_SMALL),
                on_click=lambda e, p=path: export_sheet(self.ctx, p), dense=True,
                key=f"mf-cbz-out-{self._build_gen}-{len(rows)}"))
        for path in outputs[:50]:
            rows.append(ft.ListTile(
                leading=ft.Image(src=path, width=40, height=40, fit=ft.BoxFit.COVER, cache_width=80, border_radius=4),
                title=ft.Text(os.path.basename(path), max_lines=1, overflow=ft.TextOverflow.ELLIPSIS),
                subtitle=ft.Text(os.path.basename(os.path.dirname(path)), theme_style=ft.TextThemeStyle.BODY_SMALL),
                on_click=lambda e, p=path: export_sheet(self.ctx, p), dense=True,
                key=f"mf-out-{self._build_gen}-{len(rows)}"))
        if len(outputs) > 50:
            rows.append(hint_text(f"… and {len(outputs) - 50} more"))
        self.output_list.controls = rows

    # ---- sources -----------------------------------------------------------------------------------

    async def add_paths(self, paths: list) -> int:
        """Add images / archives / folders (blocking work on the io pool), then re-render."""
        files = self.session.files
        try:
            added = await self.ctx.io(files.add_paths, list(paths))
        except Exception as exc:
            self.ctx.say(f"Could not add the files: {exc}")
            return 0
        self._after_selection_change()
        self.refresh()
        if added:
            self.ctx.remember_source("tools.manga", f"{len(files.files)} images")
            self.ctx.say(f"Added {added} image{'s' if added != 1 else ''}")
        elif paths:
            self.ctx.say("No new images found")
        return int(added or 0)

    async def pick_images(self) -> int:
        files = self.ctx.files
        if files is None:
            self.ctx.say("File picking is not available")
            return 0
        picked = await files.pick_files(allowed_extensions=[e.lstrip(".") for e in svc.IMAGE_EXTENSIONS],
                                        allow_multiple=True, dialog_title="Select manga images")
        return await self.add_paths([f.path for f in picked or []])

    async def pick_archive(self) -> int:
        files = self.ctx.files
        if files is None:
            self.ctx.say("File picking is not available")
            return 0
        picked = await files.pick_files(allowed_extensions=["cbz", "zip"], allow_multiple=True,
                                        dialog_title="Select a ZIP or CBZ")
        return await self.add_paths([f.path for f in picked or []])

    async def pick_folder(self) -> int:
        files = self.ctx.files
        if files is None:
            self.ctx.say("File picking is not available")
            return 0
        try:
            folder = await files.pick_folder(dialog_title="Select a folder with manga images")
        except Exception as exc:  # FolderPickUnavailable (Android SAF trees) and picker errors
            reason = getattr(exc, "reason", None) or str(exc)
            if reason == "No folder was chosen":
                return 0
            self.ctx.say(f"{reason} · {FOLDER_FALLBACK}", FOLDER_FALLBACK, lambda: self.ctx.spawn(self.pick_archive()))
            return 0
        return await self.add_paths([folder.path])

    # ---- list actions --------------------------------------------------------------------------------

    async def _mutate(self, fn: Any, *args: Any) -> Any:
        """Run a selection change on the io pool (the moved methods persist and check files),
        then re-render."""
        try:
            return await self.ctx.io(fn, *args)
        except Exception as exc:
            self.ctx.say(f"Could not update the list: {exc}")
            return None
        finally:
            self._after_selection_change()
            self.refresh()

    def _after_selection_change(self) -> None:
        """A selection change made while another job owned the process state could not see the
        folders a mobile job writes the generated glossary to: try again now (the job may be over),
        and otherwise when a job ends (``_on_any_job_end``)."""
        if self.session.files.glossary_stale:
            self.ctx.spawn(self._refresh_glossary())

    def toggle_skip(self, path: str) -> Any:
        return self.ctx.spawn(self._mutate(self.session.files.toggle_skip, path))

    def _on_row_tap(self, path: str) -> None:
        if self.selection_mode:
            self._toggle_selected(path)
            return
        files = self.session.files.files
        if path in files and self.screen is not None:
            self.session.page_index = files.index(path)
            self.screen.select_tab("editor")

    def _on_row_long_press(self, path: str) -> None:
        self.selection_mode = True
        self._toggle_selected(path)
        self.ctx.haptic("selection_click")

    def _toggle_selected(self, path: str) -> None:
        if path in self.selected:
            self.selected.discard(path)
        else:
            self.selected.add(path)
        if not self.selected:
            self.selection_mode = False
        self.refresh()

    def exit_selection(self) -> None:
        self.selected = set()
        self.selection_mode = False
        self.refresh()

    def _remove_selected(self, e: Any = None) -> Any:
        return self.ctx.spawn(self.remove_selected())

    async def remove_selected(self) -> int:
        paths = list(self.selected)
        self.selected = set()
        self.selection_mode = False
        removed = int(await self._mutate(self.session.files.remove, paths) or 0)
        if removed:
            self.ctx.say(f"Removed {removed} image{'s' if removed != 1 else ''}")
        return removed

    async def clear_all(self) -> bool:
        if not self.session.files.files:
            return False
        from glossarion_mobile.ui.tools.common import ask

        answer = await ask(self.ctx, "Clear all", "Remove every image from the list? The files stay on the device.",
                           (("no", "Cancel", "text"), ("yes", "Clear all", "destructive")), key="mf-clear-confirm")
        if answer != "yes":
            return False
        self.selected = set()
        self.selection_mode = False
        self.session.last_outputs = []
        self.session.imported_ocr = None
        await self._mutate(self.session.files.clear)
        return True

    def _on_reorder(self, e: Any) -> Any:
        old, new = int(getattr(e, "old_index", -1) or 0), int(getattr(e, "new_index", -1) or 0)
        self.sort_buttons.selected = []
        return self.ctx.spawn(self._mutate(self.session.files.move, old, new))

    def _on_sort(self, e: Any = None) -> None:
        chosen = list(self.sort_buttons.selected or [])
        if not chosen:
            return
        self.sort(chosen[0])

    def sort(self, sort_type: str) -> Any:
        self.sort_buttons.selected = [] if sort_type == "reverse" else [sort_type]
        return self.ctx.spawn(self._mutate(self.session.files.sort, sort_type, self.sort_reverse))

    def _toggle_direction(self, e: Any = None) -> None:
        self.sort_reverse = not self.sort_reverse
        self.sort_direction.icon = ft.Icons.ARROW_DOWNWARD if self.sort_reverse else ft.Icons.ARROW_UPWARD
        self.sort_direction.tooltip = "Descending" if self.sort_reverse else "Ascending"
        chosen = list(self.sort_buttons.selected or [])
        if chosen:
            self.sort(chosen[0])
        else:
            push(self.sort_direction)

    def _on_range(self, e: Any = None) -> None:
        self.session.files.set_range(self.range_field.value or "")
        self.refresh()

    def _on_split(self, e: Any = None) -> None:
        self.session.files.set_split_first_level(bool(self.split_switch.value))
        self.group_index = None
        self.refresh()

    def _on_group(self, e: Any = None) -> None:
        value = self.group_picker.value
        self.group_index = int(value) if value not in (None, "") else None
        self.refresh()

    # ---- run ---------------------------------------------------------------------------------------

    async def start(self, glossary_only: bool = False) -> Optional[str]:
        """Start (or Settings › Generate glossary: the OCR + glossary pass without translating)."""
        if self.watch.active() is not None:
            self.ctx.say("The manga translator is already running")
            return None
        if not self.ctx.has_kind(svc.KIND_BATCH):
            self.ctx.say("Manga jobs are not available in this build")
            return None
        try:
            spec = svc.batch_spec(self.session.files, output_root=self.ctx.output_root, glossary_only=glossary_only,
                                  editor_session=self.session.editor_token or "",
                                  imported_ocr=self._imported_ocr_path())
        except ValueError as exc:
            self.ctx.say(str(exc))
            return None
        # models are downloaded on first use (the glossary pass detects bubbles, it does not inpaint)
        if not await ensure_models(self.ctx, self.session.models, self.ctx.config_snapshot(),
                                   kinds=("detector",) if glossary_only else None,
                                   action="Generating the glossary" if glossary_only else "Translating"):
            return None
        job_id = await self.ctx.submit(spec)
        if not job_id:
            return None
        self.session.batch_job_id = job_id
        self.watch.watch(job_id)
        self._subscribe_view()
        self._mount_console(job_id)
        self.run_status.value = "Queued…"
        self.refresh()
        return job_id

    def _subscribe_view(self) -> None:
        jobs = self.ctx.jobs
        subscribe = getattr(jobs, "subscribe", None) if jobs is not None else None
        if not callable(subscribe) or self._unsub_view is not None:
            return
        try:
            self._unsub_view = subscribe(lambda view: self._on_view())
        except Exception:
            log.debug("subscribing to the job view failed", exc_info=True)

    def _on_view(self) -> None:
        if self.watch.active() is not None:
            self._render_run_state()
            push(self.progress, self.run_status, self.stop_button, self.start_button)

    def _mount_console(self, job_id: str) -> None:
        jobs = self.ctx.jobs
        getter = getattr(jobs, "log_buffer", None) if jobs is not None else None
        buffer = None
        if callable(getter):
            try:
                buffer = getter(job_id)
            except Exception:
                buffer = None
        if buffer is None:
            self.console_holder.content = hint_text("The log appears here once the job starts.")
            self.console_job = None
            return
        from glossarion_mobile.ui.components.log_console import LogConsole

        self.console = LogConsole(buffer=buffer, dispatcher=self.ctx.dispatcher, list_height=260,
                                  copy_handler=self.ctx.copy_text, key=f"mf-console-{job_id}")
        self.console_holder.content = self.console
        self.console_job = job_id
        push(self.console_holder)

    def _on_stop(self, e: Any = None) -> None:
        snap = self.watch.active()
        jobs = self.ctx.jobs
        if snap is None or jobs is None:
            return
        try:
            jobs.request_stop(snap.id)
        except Exception:
            log.exception("stopping the manga job failed")
        self._render_run_state()
        push(self.run_status)

    def _on_job_change(self, snap: Any) -> None:
        job_id = getattr(snap, "id", None)
        if job_id and self.console_job != job_id and not getattr(snap, "is_terminal", False):
            state = getattr(getattr(snap, "state", None), "value", "")
            if state != "QUEUED":
                self._mount_console(job_id)
        self._render_run_state()
        push(self.progress, self.run_status, self.stop_button, self.start_button)

    def _on_job_end(self, snap: Any) -> None:
        self.session.batch_end_applied = getattr(snap, "id", None)
        result = dict(getattr(snap, "result", {}) or {})
        outputs = [p for p in (result.get("manga_outputs") or getattr(snap, "outputs", ()) or ())
                   if str(p).lower().endswith(svc.IMAGE_EXTENSIONS)]
        if outputs:
            self.session.last_outputs = list(outputs)
        self.session.last_result = result
        cbz = [str(p) for p in (result.get("manga_cbz") or []) if p]
        if cbz:  # the run's own archives (imported CBZs packed back, "Create CBZ at end")
            self.session.last_cbz = cbz + [p for p in self.session.last_cbz if p not in cbz]
        error = getattr(snap, "error", None)
        if getattr(snap, "stopped", False):
            text = f"Stopped · {len(outputs)} page{'s' if len(outputs) != 1 else ''} translated"
        elif error:
            text = f"Failed: {error}"
        else:
            done = result.get("manga_completed", len(outputs))
            failed = result.get("manga_failed", 0)
            text = f"Done · {done} translated" + (f", {failed} failed" if failed else "")
            if cbz:
                text += f" · CBZ: {', '.join(os.path.basename(p) for p in cbz)}"
        self.run_status.value = text
        if self._unsub_view is not None:
            try:
                self._unsub_view()
            except Exception:
                pass
            self._unsub_view = None
        self.refresh()
        if self.screen is not None:
            try:
                self.screen.editor_tab.refresh()
            except Exception:
                pass
        # a glossary pass (or a run with the glossary workflow) wrote <source>/Glossary/...: only the
        # job's config snapshot knew it, so the selection's auto-load runs again for Settings
        self.ctx.spawn(self._refresh_glossary())

    async def _refresh_glossary(self) -> bool:
        """The selection's glossary auto-load as a mobile job sees the folders
        (``MangaFileList.refresh_glossary``; it stores ``manga_generated_glossary_path``), then
        Settings › Glossary re-renders. False while a job still owns the process state."""
        try:
            done = bool(await self.ctx.io(self.session.files.refresh_glossary))
        except Exception:
            log.debug("refreshing the manga glossary state failed", exc_info=True)
            return False
        if done and self.screen is not None:
            try:
                self.screen.settings_tab.refresh()
            except Exception:
                log.debug("refreshing the manga settings failed", exc_info=True)
        return done

    # ---- output --------------------------------------------------------------------------------------

    def _output_name(self) -> str:
        files = self.session.files.files
        if files:
            job = self.session.files.cbz_job_for(files[0])
            if job:
                return os.path.splitext(os.path.basename(job))[0]
            return os.path.basename(os.path.dirname(files[0])) or "manga"
        return "manga"

    async def create_cbz(self) -> Optional[str]:
        """Create CBZ over the run files (``MangaFileList.create_cbz``: pages of an imported CBZ are
        packed back into ``<name>_translated.cbz``, the others by the desktop button's
        ``_create_cbz_from_isolated_folders``); the first archive opens in the ExportSheet, every
        one is listed under Output."""
        if not self.session.files.files:
            self.ctx.say("Add images first")
            return None
        try:
            paths = await self.ctx.io(self.session.files.create_cbz)
        except svc.MangaBusy:
            self.ctx.say("Create the CBZ once the running job has finished")
            return None
        except Exception as exc:
            self.ctx.say(f"Could not create the CBZ: {exc}")
            return None
        paths = [p for p in (paths or []) if p and os.path.isfile(p)]
        if not paths:
            self.ctx.say("No translated images found. Please translate some images first.")
            return None
        self.session.last_cbz = paths + [p for p in self.session.last_cbz if p not in paths]
        self.refresh()
        if len(paths) > 1:
            self.ctx.say(f"Created {len(paths)} CBZ files (listed under Output)")
        export_sheet(self.ctx, paths[0])
        return paths[0]

    async def download_images(self) -> Optional[str]:
        outputs = [p for p in self.session.last_outputs if os.path.isfile(p)]
        if not outputs:
            self.ctx.say("Translate pages first")
            return None
        if len(outputs) == 1:
            export_sheet(self.ctx, outputs[0])
            return outputs[0]
        target = os.path.join(self.session.exports_dir, f"{self._output_name()}_images.zip")
        try:
            path = await self.ctx.io(svc.zip_images, outputs, target)
        except Exception as exc:
            self.ctx.say(f"Could not pack the images: {exc}")
            return None
        export_sheet(self.ctx, path, title=f"{len(outputs)} images · {os.path.basename(path)}")
        return path
