"""Converter / Compile (``/tools/convert``, UI_SPEC §4.5; FEATURE_MAP qa-epub-pdf 41-61).

* **Source**: one translation output folder from the SourcePicker (Library / recent / chat
  workspaces); ``?out=<bid>`` preselects a Library book, ``?tab=validate`` scrolls to Validate.
* **Actions** (jobs, so they queue behind a running translation and log to Jobs):
  Compile EPUB · Compile PDF (a PDF workspace runs the PDF workspace compiler; an EPUB
  workspace compiles its EPUB with "Create PDF after EPUB", rendered by the PyMuPDF HTML shim
  on mobile) · Validate EPUB structure · Rename files (retain source extension).
* **Result card**: the compiled files with Share / Save / Save to Downloads / Open in Reader /
  Add to Library, the validation lines, the rename outcome.
* **EPUB options** (schema tiles bound to config.json; they apply to the next compile):
  EPUB Layout (Auto / EPUB2 / EPUB3), NCX-only, CSS attach, CSS override (Load CSS… through
  the file picker / Clear), custom fonts (Load font… / Clear), HTML method, retain extension,
  gallery / cover / special files / unreferenced images, image compression; links to the PDF
  and Meta Data settings pages.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import flet as ft

from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.components.action_sheet import ActionItem, ActionSheet
from glossarion_mobile.ui.router import RouteMatch
from glossarion_mobile.ui.screens.base import Screen
from glossarion_mobile.ui.theme import HIT_TARGET
from glossarion_mobile.ui.tools import compile_model as cm
from glossarion_mobile.ui.tools import targets as tg
from glossarion_mobile.ui.tools.common import (
    JobWatch,
    action_button,
    card,
    failed_job_card,
    hint_text,
    job_failed,
    schema_tiles,
)
from glossarion_mobile.ui.tools.source_picker import SourcePicker

__all__ = ["ConverterScreen", "converter_eligibility"]

log = logging.getLogger("glossarion.tools.convert")

_KIND_LABELS = {"epub": "EPUB workspace", "pdf": "PDF workspace", "txt": "Text workspace", "other": "Workspace"}
#: The job kinds this screen submits (a reopened screen follows a queued / running one of its folder).
CONVERTER_KINDS = ("compile_epub", "compile_pdf", "validate_epub", "rename_outputs", "md_txt_sidecars",
                   "br_to_paragraphs")
#: Result-card files the Reader opens ("Open in Reader"; a compiled ``*_translated.txt`` included).
READER_OUTPUT_EXTENSIONS = (".epub", ".txt")
#: Result-card files: the compiled EPUB / PDF, and a TXT book when a job lists one.
RESULT_EXTENSIONS = (".epub", ".pdf", ".txt")


def converter_eligibility(target: tg.ToolTarget) -> Optional[str]:
    if not target.folder:
        return "No output folder to compile"
    return None


class ConverterScreen(Screen):
    title = "Compile EPUB / PDF"

    def __init__(self, match: Optional[RouteMatch], ctx: Any) -> None:
        super().__init__(match)
        self.ctx = ctx
        self.state = ctx.tool_state.setdefault("convert", {})
        self.target: Optional[tg.ToolTarget] = self.state.get("target")
        self.watch = JobWatch(ctx, self._on_job_end, self._on_job_change)
        self.active_job: Any = None
        self.last_result: Any = None
        self.option_tiles: dict = {}
        self.picker: Optional[SourcePicker] = None
        self.focus = match.get("tab") if match is not None else None

    # ---- layout --------------------------------------------------------------------------------

    def build_body(self) -> ft.Control:
        self.target_text = ft.Text("", theme_style=ft.TextThemeStyle.BODY_MEDIUM, key="cv-target")
        self.kind_text = hint_text("", key="cv-kind")
        source = card("Output folder", [self.target_text, self.kind_text,
                                        ft.Row([ft.FilledTonalButton(content="Choose…", icon=ft.Icons.FOLDER_OPEN,
                                                                     on_click=lambda e: self.open_picker(),
                                                                     key="cv-choose")])],
                      icon="FOLDER", key="cv-source")
        self.actions_row = ft.Row(wrap=True, spacing=8, run_spacing=8, key="cv-actions")
        self.validate_row = ft.Row(wrap=True, spacing=8, run_spacing=8, key="cv-validate-actions")
        self.run_status = ft.Text("", theme_style=ft.TextThemeStyle.BODY_SMALL, key="cv-status")
        self.stop_button = ft.OutlinedButton(content="Stop", icon=ft.Icons.STOP, on_click=self._on_stop,
                                             visible=False, key="cv-stop")
        self.progress = ft.ProgressBar(visible=False, key="cv-progress")
        build = card("Build", [self.actions_row, ft.Row([self.stop_button]), self.progress, self.run_status],
                     icon="MENU_BOOK", key="cv-build")
        self.validate_card = card("Validate & rename", [self.validate_row], icon="RULE", key=ft.ScrollKey("cv-validate"),
                                  subtitle="Checks the files an EPUB needs (container.xml, OPF, NCX, chapters, "
                                           "metadata.json); Rename applies the retain-extension setting.")
        self.result_column = ft.Column(spacing=4, key="cv-results")
        self.result_card = card("Result", [self.result_column], icon="TASK_ALT", key="cv-result")
        self.result_card.visible = False
        self.options_card = self._options_card()
        self._render_target()
        self.body_list = ft.ListView(
            controls=[source, build, self.validate_card, self.result_card, self.options_card],
            expand=True, spacing=tokens.SPACING["md"], padding=ft.Padding.symmetric(horizontal=12, vertical=8),
            key="cv-body")
        return self.body_list

    def _options_card(self) -> ft.Control:
        layout_value = str(self.ctx.cfg("epub_layout_mode", "auto") or "auto")
        self.layout = ft.SegmentedButton(
            segments=[ft.Segment(value=value, label=ft.Text(label)) for value, label in cm.LAYOUT_CHOICES],
            selected=[layout_value if layout_value in dict(cm.LAYOUT_CHOICES) else "auto"],
            allow_multiple_selection=False, show_selected_icon=False, on_change=self._on_layout, key="cv-layout")
        self.css_text = hint_text("", key="cv-css")
        self.fonts_text = hint_text("", key="cv-fonts")
        css_row = ft.Row([ft.FilledTonalButton(content="Load CSS…", icon=ft.Icons.STYLE, on_click=self._on_load_css,
                                               key="cv-load-css"),
                          ft.TextButton(content="Clear", on_click=lambda e: self.clear_css(), key="cv-clear-css")],
                         wrap=True)
        fonts_row = ft.Row([ft.FilledTonalButton(content="Load Font…", icon=ft.Icons.FONT_DOWNLOAD,
                                                 on_click=self._on_load_fonts, key="cv-load-fonts"),
                            ft.TextButton(content="Clear", on_click=self._on_clear_fonts, key="cv-clear-fonts")],
                           wrap=True)
        tiles, self.option_tiles = schema_tiles(self.ctx, cm.EPUB_OPTION_KEYS)
        image_tiles, image_map = schema_tiles(self.ctx, cm.IMAGE_OPTION_KEYS)
        self.option_tiles.update(image_map)
        if not tiles:
            tiles = [hint_text("Settings are not available in this session.", key="cv-no-settings")]
        links = ft.Row([
            ft.TextButton(content="PDF settings", icon=ft.Icons.PICTURE_AS_PDF,
                          on_click=lambda e: self.open_settings("other.output"), key="cv-pdf-settings"),
            ft.TextButton(content="Meta Data & EPUB settings", icon=ft.Icons.SETTINGS,
                          on_click=lambda e: self.open_settings("other.meta_data"), key="cv-meta-settings"),
            ft.TextButton(content="Special files", icon=ft.Icons.FILTER_LIST,
                          on_click=lambda e: self.open_settings("other.processing.extraction"),
                          key="cv-special-settings"),
        ], wrap=True)
        self._refresh_css_fonts()
        return card("EPUB options", [
            ft.Text("EPUB Layout:", weight=ft.FontWeight.W_600), self.layout,
            hint_text("Auto keeps the source EPUB's folder layout; EPUB2 forces OEBPS/Text/; EPUB3 a flat OEBPS/."),
            ft.Text("CSS override", weight=ft.FontWeight.W_600), self.css_text, css_row,
            ft.Text("Custom fonts", weight=ft.FontWeight.W_600), self.fonts_text, fonts_row,
            *tiles, *image_tiles, links,
        ], icon="TUNE", key="cv-options", subtitle="Saved to config.json; used by the next compile.")

    def _render_target(self) -> None:
        target = self.target
        if target is None:
            self.target_text.value = "No output folder chosen."
            self.kind_text.value = ""
        else:
            self.target_text.value = f"{target.title} · 📁 {target.folder_name}"
            self.kind_text.value = _KIND_LABELS.get(target.kind or "other", "Workspace")
        self._render_actions()
        self.ctx.push(self.target_text, self.kind_text)

    def _render_actions(self) -> None:
        target = self.target
        running = self.active_job is not None and not getattr(self.active_job, "is_terminal", True)
        none = "Choose an output folder first" if target is None else None
        pdf_workspace = target is not None and target.kind == "pdf"
        epub_reason = none or (None if self.ctx.has_kind("compile_epub") else "The job service is not running")
        if pdf_workspace and epub_reason is None:
            epub_reason = "A PDF workspace compiles to PDF"
        pdf_reason = none or (None if self.ctx.has_kind("compile_pdf") else "The job service is not running")
        pdf_label = "Compile PDF" if pdf_workspace or target is None else "Compile PDF (EPUB + PDF)"
        validate_reason = none or (None if self.ctx.has_kind("validate_epub") else "The job service is not running")
        if pdf_workspace and validate_reason is None:
            validate_reason = "EPUB workspaces only"
        rename_reason = none or (None if self.ctx.has_kind("rename_outputs") else "The job service is not running")
        sidecar_reason = none or (None if self.ctx.has_kind("md_txt_sidecars") else "The job service is not running")
        br_reason = none or (None if self.ctx.has_kind("br_to_paragraphs") else "The job service is not running")
        if running:
            sidecar_reason = sidecar_reason or "A job of this folder is running"
            br_reason = br_reason or "A job of this folder is running"
        if running:
            epub_reason = epub_reason or "A compile job is running"
            pdf_reason = pdf_reason or "A compile job is running"
        self.actions_row.controls = [
            action_button("Compile EPUB", "MENU_BOOK", lambda e: self.ctx.spawn(self.compile("epub")),
                          key="cv-compile-epub", reason=epub_reason, filled=True),
            action_button(pdf_label, "PICTURE_AS_PDF", lambda e: self.ctx.spawn(self.compile("pdf")),
                          key="cv-compile-pdf", reason=pdf_reason),
        ]
        self.validate_row.controls = [
            action_button("🔍 Validate EPUB Structure", "RULE", lambda e: self.ctx.spawn(self.validate()),
                          key="cv-validate-btn", reason=validate_reason),
            action_button("Rename Files", "DRIVE_FILE_RENAME_OUTLINE", lambda e: self.ctx.spawn(self.rename()),
                          key="cv-rename", reason=rename_reason),
            # desktop Other Settings › Output MD / TXT "Generate MD" / "Generate TXT" (retroactive sidecars)
            action_button("Generate MD", "DESCRIPTION", lambda e: self.ctx.spawn(self.generate_sidecars("md")),
                          key="cv-gen-md", reason=sidecar_reason),
            action_button("Generate TXT", "TEXT_SNIPPET", lambda e: self.ctx.spawn(self.generate_sidecars("txt")),
                          key="cv-gen-txt", reason=sidecar_reason),
            # desktop "Convert <br> tags to <p> paragraphs" › Apply to Existing Outputs
            action_button("Apply <br> → <p> to outputs", "FORMAT_PARAGRAPH",
                          lambda e: self.ctx.spawn(self.convert_br()), key="cv-br", reason=br_reason),
        ]
        snap = self.active_job
        self.stop_button.visible = running
        self.progress.visible = running
        if running:
            from glossarion_mobile.services.jobs import progress_line

            self.run_status.value = f"{snap.state_label} · {progress_line(snap) or snap.phase}"
        elif snap is not None:
            self.run_status.value = f"{snap.state_label}: {snap.title}" + (f" — {snap.error}" if snap.error else "")
        else:
            self.run_status.value = ""
        self.ctx.push(self.actions_row, self.validate_row, self.stop_button, self.progress, self.run_status)

    # ---- lifecycle ------------------------------------------------------------------------------------

    def did_show(self) -> None:
        self.watch.start()
        self._adopt_jobs()
        out = self.match.get("out") if self.match is not None else None
        if out and self.target is None:
            self.ctx.spawn(self.preselect_book(out))
        if self.focus == "validate":
            self.ctx.spawn(self._scroll_to_validate())

    def dispose(self) -> None:
        self.watch.stop()

    async def _scroll_to_validate(self) -> None:
        try:
            await self.body_list.scroll_to(scroll_key="cv-validate")
        except Exception:
            pass

    async def preselect_book(self, bid: str) -> Optional[tg.ToolTarget]:
        service = self.ctx.service
        if service is None:
            return None
        book = service.book_for_bid(bid)
        if not book:
            return None
        target = await self.ctx.io(tg.target_for_book, service, book)
        if target is not None and converter_eligibility(target) is None:
            self.set_target(target)
        return target

    # ---- source --------------------------------------------------------------------------------------

    def open_picker(self) -> SourcePicker:
        self.picker = SourcePicker(self.ctx, title="Choose an output folder", multi=False,
                                   eligible=converter_eligibility, on_done=lambda rows: self.set_target(rows[0]
                                                                                                         if rows else None))
        if self.ctx.page is not None:
            self.picker.show(self.ctx.page)
        return self.picker

    def set_target(self, target: Optional[tg.ToolTarget]) -> None:
        self.target = target
        self.state["target"] = target
        self._adopt_jobs()
        self._render_target()

    def _adopt_jobs(self) -> None:
        """A compile / validate / rename job of this output folder queued or running from an earlier visit (or
        the Book page): follow it, so Stop shows and the compile buttons stay disabled until it ends."""
        target = self.target
        folder = os.path.normcase(os.path.abspath(target.folder)) if target is not None and target.folder else ""
        if not folder or (self.active_job is not None and not getattr(self.active_job, "is_terminal", True)):
            return

        def same_folder(snap: Any) -> bool:
            inputs = getattr(getattr(snap, "spec", None), "inputs", ()) or ()
            return any(os.path.normcase(os.path.abspath(str(p))) == folder for p in inputs)

        adopted = self.watch.adopt(CONVERTER_KINDS, same_folder)
        if adopted:
            self.active_job = adopted[0]
            if getattr(self, "actions_row", None) is not None:
                self._render_actions()

    # ---- jobs ------------------------------------------------------------------------------------------

    async def _submit(self, spec: Any, label: str) -> Optional[str]:
        job_id = await self.ctx.submit(spec)
        if not job_id:
            return None
        self.watch.watch(job_id)
        snapshot = getattr(self.ctx.jobs, "snapshot", None)
        self.active_job = snapshot(job_id) if callable(snapshot) else None
        self.ctx.remember_source("tools.convert", spec.title)
        self.ctx.haptic("medium_impact")
        self.ctx.say(f"{label} · {spec.title}", "Jobs", lambda: self.ctx.go("jobs.detail", {"jid": job_id}))
        self._render_actions()
        return job_id

    async def retry(self, snap: Any) -> Optional[str]:
        """ErrorCard › Retry: the failed job's own spec again."""
        spec = getattr(snap, "spec", None)
        if spec is None:
            return None
        return await self._submit(spec, "Retrying")

    async def compile(self, fmt: str) -> Optional[str]:
        if self.target is None:
            self.open_picker()
            return None
        try:
            spec = cm.compile_spec(self.target, fmt)
        except ValueError as exc:
            self.ctx.say(str(exc))
            return None
        label = "Compiling PDF" if fmt == "pdf" else "Compiling EPUB"
        return await self._submit(spec, label)

    async def validate(self) -> Optional[str]:
        if self.target is None:
            self.open_picker()
            return None
        try:
            spec = cm.validate_spec([self.target])
        except ValueError as exc:
            self.ctx.say(str(exc))
            return None
        return await self._submit(spec, "Validating EPUB")

    async def rename(self) -> Optional[str]:
        if self.target is None:
            self.open_picker()
            return None
        retain = bool(self.ctx.cfg("retain_source_extension", False))
        return await self._submit(cm.rename_spec(self.target, retain), "Renaming files")

    def _folder_spec(self, kind: str, title_suffix: str, params: Optional[dict] = None) -> Any:
        from glossarion_mobile.services.jobs import JobSpec

        target = self.target
        return JobSpec(kind=kind, title=f"{target.title}{title_suffix}", inputs=(target.folder,),
                       params={"folder": target.folder, **(params or {})},
                       origin={"type": "tool", "label": "Tools · Converter"})

    async def generate_sidecars(self, fmt: str) -> Optional[str]:
        """Generate MD / Generate TXT for the output folder (job ``md_txt_sidecars``)."""
        if self.target is None or not self.target.folder:
            self.open_picker()
            return None
        return await self._submit(self._folder_spec("md_txt_sidecars", "", {"format": fmt}),
                                  f"Generating {fmt.upper()}")

    async def convert_br(self) -> Optional[str]:
        """Apply <br> → <p> to the output folder after the desktop confirmation (job ``br_to_paragraphs``)."""
        if self.target is None or not self.target.folder:
            self.open_picker()
            return None
        from glossarion_mobile.job_kinds.compile import BR_CONFIRM_TEXT, BR_CONFIRM_TITLE
        from glossarion_mobile.ui.tools.common import ask

        answer = await ask(self.ctx, BR_CONFIRM_TITLE,
                           f"Modify existing HTML outputs for 1 selected input file(s)?\n\n{BR_CONFIRM_TEXT}",
                           (("yes", "Yes", "destructive"), ("cancel", "Cancel", "text")), key="cv-br-confirm")
        if answer != "yes":
            return None
        return await self._submit(self._folder_spec("br_to_paragraphs", ""), "Converting <br> to <p>")

    def _on_stop(self, e: Any = None) -> Any:
        snap = self.active_job
        if snap is None or self.ctx.jobs is None:
            return None
        try:
            return self.ctx.jobs.request_stop(snap.id)
        except Exception as exc:
            self.ctx.say(f"Could not stop: {exc}")
            return None

    def _on_job_change(self, snap: Any) -> None:
        self.active_job = snap
        self._render_actions()

    def _on_job_end(self, snap: Any) -> None:
        self.active_job = snap
        self.last_result = snap
        self._render_actions()
        self.render_result(snap)

    def render_result(self, snap: Any) -> None:
        rows: list[ft.Control] = []
        result = dict(getattr(snap, "result", {}) or {})
        kind = getattr(snap, "kind", "")
        state = str(getattr(getattr(snap, "state", None), "value", ""))
        if job_failed(snap):  # UI_SPEC §7.4: an ErrorCard with Retry · Copy error · View log
            rows.append(failed_job_card(self.ctx, snap, key="cv-error",
                                        on_retry=lambda s=snap: self.ctx.spawn(self.retry(s))))
        elif kind == "validate_epub":
            passed = bool(result.get("all_passed"))
            rows.append(ft.Text("✅ All Valid!" if passed else "Validation Results",
                                weight=ft.FontWeight.W_600, key="cv-validate-title"))
            for line in cm.result_lines(result):
                rows.append(ft.Text(line, selectable=True, theme_style=ft.TextThemeStyle.BODY_SMALL))
        elif kind == "rename_outputs":
            rows.append(ft.Text(str(result.get("rename") or snap.state_label), key="cv-rename-result"))
        elif kind == "md_txt_sidecars":
            info = result.get("md_txt") if isinstance(result.get("md_txt"), dict) else {}
            rows.append(ft.Text(str(info.get("message") or snap.state_label), key="cv-md-txt-result",
                                weight=ft.FontWeight.W_600))
            if info:
                rows.append(ft.Text(f"{str(info.get('format') or '').upper()}: {info.get('ok', 0)} written, "
                                    f"{info.get('failed', 0)} failed of {info.get('total', 0)} HTML files",
                                    theme_style=ft.TextThemeStyle.BODY_SMALL))
        elif kind == "br_to_paragraphs":
            rows.append(ft.Text(str(result.get("br_message") or snap.state_label), key="cv-br-result",
                                weight=ft.FontWeight.W_600))
            for audit in result.get("br_audit") or []:
                rows.append(ft.Text(f"{os.path.basename(str(audit.get('output_dir') or ''))}: {audit.get('changed', 0)} "
                                    f"converted, {audit.get('unchanged', 0)} unchanged, {audit.get('failed', 0)} failed "
                                    f"({audit.get('scanned', 0)} scanned)", theme_style=ft.TextThemeStyle.BODY_SMALL))
        else:
            outputs = cm.outputs_of(getattr(snap, "outputs", ()) or (), RESULT_EXTENSIONS)
            if not outputs:
                rows.append(ft.Text(f"{snap.state_label}: no compiled file" + (f" — {snap.error}" if snap.error else ""),
                                    key="cv-no-output"))
            for path in outputs:
                rows.append(ft.ListTile(
                    leading=ft.Icon(ft.Icons.PICTURE_AS_PDF if path.lower().endswith(".pdf") else ft.Icons.MENU_BOOK),
                    title=ft.Text(os.path.basename(path), max_lines=2),
                    subtitle=ft.Text(state.title(), theme_style=ft.TextThemeStyle.BODY_SMALL),
                    trailing=ft.IconButton(icon=ft.Icons.MORE_VERT, tooltip="Actions",
                                           on_click=lambda e, p=path: self.output_actions(p), size_constraints=HIT_TARGET),
                    on_click=lambda e, p=path: self.output_actions(p), dense=True,
                    key=f"cv-output-{os.path.basename(path)}"))
        self.result_column.controls = rows
        self.result_card.visible = True
        self.ctx.push(self.result_card)

    def output_actions(self, path: str) -> ActionSheet:
        files = self.ctx.files
        items: list = []
        if files is not None:
            for option in files.export_options(path):
                items.append(ActionItem(option.label, lambda o=option.id: self.ctx.spawn(self.export(o, path)),
                                        icon=option.icon, disabled_reason=option.disabled_reason,
                                        key=f"export-{option.id}"))
        epub = path.lower().endswith(".epub")
        readable = path.lower().endswith(READER_OUTPUT_EXTENSIONS)  # the Reader opens TXT books too
        items.append(ActionItem("Open in Reader", lambda: self.ctx.open_reader(path=path), icon="AUTO_STORIES",
                                disabled_reason=None if readable else "EPUB and TXT files only", key="open-reader"))
        items.append(ActionItem("Add to Library", lambda: self.ctx.spawn(self.add_to_library(path)),
                                icon="LIBRARY_ADD", disabled_reason=None if (epub and files is not None)
                                else "Only EPUB files go to the Completed shelf", key="add-library"))
        sheet = ActionSheet(items, title=os.path.basename(path), tablet=self.ctx.tablet)
        if self.ctx.page is not None:
            sheet.show(self.ctx.page)
        return sheet

    async def export(self, option_id: str, path: str) -> Any:
        try:
            return await self.ctx.files.export(option_id, path)
        except Exception as exc:
            self.ctx.say(f"Export failed: {exc}")
            return None

    async def add_to_library(self, path: str) -> Any:
        files = self.ctx.files
        try:
            added = await self.ctx.io(lambda: files.add_to_library(path, translated=True))
        except Exception as exc:
            self.ctx.say(f"Could not add: {exc}")
            return None
        service = self.ctx.service
        if service is not None and hasattr(service, "mark_dirty"):
            service.mark_dirty()
        self.ctx.say(f"Added to Library: {added.name}", "Library", lambda: self.ctx.go("library"))
        return added

    # ---- options ----------------------------------------------------------------------------------------

    def open_settings(self, section: str) -> None:
        settings = self.ctx.settings
        if settings is not None and hasattr(settings, "open_setting"):
            settings.open_setting(section)
        else:
            self.ctx.go("settings.section", {"section": section})

    def _on_layout(self, e: Any = None) -> None:
        selected = list(getattr(getattr(e, "control", None), "selected", None) or self.layout.selected or ["auto"])
        self.set_layout(selected[0])

    def set_layout(self, value: str) -> None:
        if value not in dict(cm.LAYOUT_CHOICES):
            return
        self.layout.selected = [value]
        self.ctx.set_cfg("epub_layout_mode", value)
        self.ctx.push(self.layout)

    def _refresh_css_fonts(self) -> None:
        css = str(self.ctx.cfg("epub_css_override_path", "") or "")
        self.css_text.value = (f"✓ {os.path.basename(css)}" if css and os.path.isfile(css)
                               else (f"⚠ Missing: {os.path.basename(css)}" if css else "No CSS override (default CSS)"))
        fonts_dir = self.state.get("fonts_dir")
        count = self.state.get("font_count")
        if fonts_dir is None:
            self.fonts_text.value = "Custom fonts are mirrored into every compiled EPUB."
            self.ctx.spawn(self._count_fonts())
        else:
            self.fonts_text.value = f"✓ {count} font{'s' if count != 1 else ''} loaded" if count else "No fonts loaded"
        self.ctx.push(self.css_text, self.fonts_text)

    async def _count_fonts(self) -> int:
        def count() -> tuple:
            folder = cm.custom_fonts_dir()
            return folder, cm.count_fonts(folder)

        try:
            folder, total = await self.ctx.io(count)
        except Exception:
            return 0
        self.state["fonts_dir"], self.state["font_count"] = folder, total
        self.fonts_text.value = f"✓ {total} font{'s' if total != 1 else ''} loaded" if total else "No fonts loaded"
        self.ctx.push(self.fonts_text)
        return total

    async def load_css(self) -> Optional[str]:
        files = self.ctx.files
        if files is None:
            self.ctx.say("Picking files is not available in this session")
            return None
        try:
            picked = await files.pick_files(target="inbox", allowed_extensions=["css"], allow_multiple=False,
                                            dialog_title="Select CSS file")
        except Exception as exc:
            self.ctx.say(f"Could not pick the file: {exc}")
            return None
        path = next((getattr(f, "path", None) for f in picked or () if getattr(f, "path", None)), None)
        if not path:
            return None
        import_dir = self.ctx.import_dir or os.path.join(self.ctx.data_dir or os.path.dirname(path), "imports")
        try:
            stored = await self.ctx.io(cm.import_css, path, import_dir)
        except Exception as exc:
            self.ctx.say(str(exc))
            return None
        self.ctx.set_cfg("epub_css_override_path", stored)
        self._refresh_css_fonts()
        self.ctx.say(f"CSS override: {os.path.basename(stored)}")
        return stored

    async def _on_load_css(self, e: Any = None) -> Optional[str]:
        return await self.load_css()

    def clear_css(self) -> None:
        self.ctx.set_cfg("epub_css_override_path", "")
        self._refresh_css_fonts()

    async def load_fonts(self) -> int:
        files = self.ctx.files
        if files is None:
            self.ctx.say("Picking files is not available in this session")
            return 0
        try:
            picked = await files.pick_files(target="inbox", allowed_extensions=["ttf", "otf", "woff", "woff2", "zip"],
                                            allow_multiple=True, dialog_title="Select Font Files or ZIP Archives")
        except Exception as exc:
            self.ctx.say(f"Could not pick the files: {exc}")
            return 0
        paths = [getattr(f, "path", None) for f in picked or () if getattr(f, "path", None)]
        if not paths:
            return 0

        def run() -> tuple:
            folder = cm.custom_fonts_dir()
            return folder, cm.import_fonts(paths, folder), cm.count_fonts(folder)

        folder, copied, total = await self.ctx.io(run)
        self.state["fonts_dir"], self.state["font_count"] = folder, total
        self._refresh_css_fonts()
        self.ctx.say(f"✅ {copied} loaded" if copied else "No font files found")
        return copied

    async def _on_load_fonts(self, e: Any = None) -> int:
        return await self.load_fonts()

    async def clear_fonts(self) -> int:
        def run() -> int:
            return cm.clear_fonts(cm.custom_fonts_dir())

        removed = await self.ctx.io(run)
        self.state["font_count"] = 0
        self.state["fonts_dir"] = self.state.get("fonts_dir") or ""
        self._refresh_css_fonts()
        return removed

    async def _on_clear_fonts(self, e: Any = None) -> int:
        return await self.clear_fonts()
