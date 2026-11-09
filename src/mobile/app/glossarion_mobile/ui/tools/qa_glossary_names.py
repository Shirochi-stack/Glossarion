"""Check glossary names (QA Scanner, experimental; desktop QA Scanner settings › "Check glossary names
in existing output…").

Opened from the QA screen or from a chapter's Resolve QA action. It scans the chosen folders on the io pool
(``glossary_name_check.review_folder``: the folder's glossary with the Emergency Glossary Compliance
entry types / minimum length, chapters matched to their sources through translation_progress.json +
chapters_full.json) and lists every name that differs from the glossary as ☐ "Found → Glossary
name ×n" under its chapter file (and translated_headers.txt / TOC.txt / metadata.json for the
chapter titles, TOC and book metadata). Rows are pre-checked when one close spelling of a character name
was found (the automatic-fix case). **Apply checked** asks first, then replaces the checked names in
text (never inside tags), backing each file up to ``glossary_name_check_backup``
(``glossary_name_check.fix_files``).
"""

from __future__ import annotations

import asyncio
import html
import logging
import os
from typing import Any, Optional, Sequence

import flet as ft

from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.components.dialogs import close_dialog
from glossarion_mobile.ui.components.sheet import bottom_sheet, scroll_column, scroll_sheet, sheet_frame
from glossarion_mobile.ui.tools.common import ask, hint_text, section_header

__all__ = ["GlossaryNameFixSheet", "NER_KEY", "chapters_at_once", "name_check_scope", "use_ner"]

log = logging.getLogger("glossarion.tools.qa")

NER_KEY = "glossary_name_check_use_ner"
SCOPE_KEYS = ("emergency_glossary_compliance_mode", "emergency_glossary_compliance_custom_types",
              "glossary_name_check_min_chars", "glossary_name_check_autofix_level",
              "glossary_name_check_match_numbers", "glossary_name_check_plural_exclusion",
              "glossary_name_check_tense_exclusion", "glossary_name_check_inflection_skip_types",
              "output_language")
TITLE = "Check Glossary Names"
INTRO = ("Checked rows are replaced with the glossary name. Rows are pre-checked when one close spelling of a "
         "character name was found; review the others before checking them. Each changed file is backed up to "
         "glossary_name_check_backup first.")


def name_check_scope(ctx: Any) -> dict:
    """Emergency Glossary Compliance's entry types and the name check's minimum length (saved config)."""
    import glossary_name_check as name_check

    config = {}
    for key in SCOPE_KEYS:
        value = ctx.cfg(key, None)
        if value is not None:
            config[key] = value
    return name_check.settings_from_config(config)


def use_ner(ctx: Any) -> bool:
    """Settings › "Use Named Entity Recognition model (NER)" (default on)."""
    import glossary_name_check as name_check

    return name_check.use_ner_from_config({"glossary_name_check_use_ner": ctx.cfg(NER_KEY, True)})


def chapters_at_once(ctx: Any) -> int:
    """Settings › "Chapters checked at once" (glossary_name_check_workers; 0 = Auto)."""
    import glossary_name_check as name_check

    return name_check.workers_from_config({"glossary_name_check_workers": ctx.cfg("glossary_name_check_workers", 0)})


def glossary_layer_settings(ctx: Any) -> dict:
    """Settings › Glossary's Add Additional Glossary / Enable Unified Glossary (and the unified file's
    language keys): the name check uses only the glossaries a translation request would."""
    import glossary_name_check as name_check

    keys = ("add_additional_glossary", "enable_unified_glossary", "unified_glossary_source_language",
            "unified_glossary_combine_all_languages", "output_language")
    return name_check.layer_settings_from_config({key: ctx.cfg(key) for key in keys if ctx.cfg(key) is not None})


def ner_needs_download() -> bool:
    try:
        import ner_names
    except ImportError:
        return False
    return ner_names.runtime_available() and not ner_names.model_installed()


async def offer_ner_download(ctx: Any, manager: Any = None) -> bool:
    """Switching "Check glossary names" on while the NER model is not on the device: the model download
    question the manga tools ask (``ensure_models``: Download now with the Models sheet's row, Download in
    the run = on first use, or Cancel). True when the model is there or will download with the scan."""
    if not ner_needs_download():
        return True
    import manga_models

    from glossarion_mobile.services import manga as svc
    from glossarion_mobile.ui.tools.manga.models import ensure_models

    return await ensure_models(ctx, manager or svc.ModelManager(), {}, model_ids=(manga_models.NER_MODEL_KEY,),
                               action="Check glossary names")


def _review_all(folders: Sequence[str], scope: dict, with_ner: bool = False,
                state: Optional[dict] = None, layer_settings: Optional[dict] = None, chosen: int = 0,
                output_file: str = "", progress_path: Optional[str] = None) -> list:
    """Review every folder on a worker thread; ``state`` gets the progress text and fraction
    (``text`` / ``value``, None = indeterminate) that ``GlossaryNameFixSheet.scan`` shows. ``chosen``:
    Settings › "Chapters checked at once" (``chapters_at_once``; 0 = Auto)."""
    import glossary_name_check as name_check

    state = {} if state is None else state
    mb = 1024 * 1024

    def on_download(report: Any) -> None:
        state.update(text=f"Downloading the named-entity model (NER): {report.downloaded / mb:.1f} / "
                          f"{report.total / mb:.1f} MB", value=report.fraction)

    ner = name_check.resolve_ner(with_ner, log.info, progress=on_download, retry=True)
    workers = name_check.chapter_workers(ner, chosen=chosen)
    reviews = []
    for index, folder in enumerate(folders):
        def on_chapter(done: int, total: int, index: int = index) -> None:
            where = f" (folder {index + 1} of {len(folders)})" if len(folders) > 1 else ""
            state.update(text=f"Checking chapter {done + 1} of {total}{where}…",
                         value=done / total if total else None)

        state.update(text="Scanning…", value=None)
        reviews.append(name_check.review_folder(folder, scope, ner=ner, progress=on_chapter, workers=workers,
                                                settings=layer_settings, progress_path=progress_path,
                                                selected_files=[output_file] if output_file else None))
    return reviews


def _preview_snippets(folder: str, output_file: str, finding: Any, progress_path: Optional[str]) -> tuple:
    import glossary_name_check as name_check

    with open(os.path.join(folder, output_file), encoding="utf-8", errors="ignore") as handle:
        markup = handle.read()
    if output_file in name_check.TITLE_FILES:
        return name_check.finding_snippets(html.escape(markup), finding), None
    source = name_check.source_chapters_for_folder(folder, progress_path).get(output_file, "")
    return name_check.finding_snippets(markup, finding), name_check.source_snippets(source, finding)


class GlossaryNameFixSheet:
    def __init__(self, ctx: Any, folders: Sequence[str], *, output_file: str = "",
                 progress_path: Optional[str] = None) -> None:
        self.ctx = ctx
        self.folders = [str(f) for f in folders if f]
        self.output_file = output_file
        self.progress_path = progress_path
        self.reviews: list = []
        self.rows: list = []  # (checkbox, folder, output file, finding)
        self.applied: Optional[dict] = None
        self.status = ft.Text("Scanning…", key="gn-status")
        self.progress = ft.ProgressBar(key="gn-progress")
        self.results = ft.Column(spacing=2, key="gn-results")
        self.check_all = ft.TextButton(content="Check all", disabled=True, key="gn-check-all",
                                       on_click=lambda e: self.set_all(True))
        self.uncheck_all = ft.TextButton(content="Uncheck all", disabled=True, key="gn-uncheck-all",
                                         on_click=lambda e: self.set_all(False))
        self.apply_button = ft.FilledButton(content="Apply checked", icon=ft.Icons.SPELLCHECK, disabled=True,
                                            on_click=self._on_apply, key="gn-apply")
        footer = ft.Row([self.check_all, self.uncheck_all, self.apply_button], alignment=ft.MainAxisAlignment.END,
                        wrap=True, spacing=8)
        body = scroll_column([
            ft.Text("Resolve QA issue · Glossary names" if output_file else TITLE,
                    theme_style=ft.TextThemeStyle.TITLE_LARGE, weight=ft.FontWeight.W_600),
            hint_text(INTRO),
            self.progress,
            self.status,
            self.results,
        ], footer=[footer], spacing=tokens.SPACING["sm"])
        self.sheet = bottom_sheet(sheet_frame(body), key="gn-sheet")
        self._page: Any = None

    # ---- lifecycle ---------------------------------------------------------------------------------

    def show(self, page: Any) -> "GlossaryNameFixSheet":
        self._page = page
        page.show_dialog(self.sheet)
        self.ctx.spawn(self.scan())
        return self

    def close(self) -> None:
        close_dialog(self._page, self.sheet)

    # ---- scan --------------------------------------------------------------------------------------

    async def scan(self) -> list:
        with_ner = use_ner(self.ctx)
        state: dict = {}
        if with_ner and ner_needs_download():
            with_ner = self._page is not None and await ask(
                self.ctx, TITLE, "Download the 104 MB named-entity model for this check?",
                [("heuristic", "Use capitalization", "text"), ("download", "Download model", "filled")],
                key="gn-ner-download") == "download"
        try:
            task = asyncio.ensure_future(
                self.ctx.io(_review_all, self.folders, name_check_scope(self.ctx), with_ner, state,
                            glossary_layer_settings(self.ctx), chapters_at_once(self.ctx),
                            self.output_file, self.progress_path))
            shown = None
            while not task.done():
                current = (state.get("text"), state.get("value"))
                if current[0] and current != shown:
                    shown = current
                    self.status.value, self.progress.value = current
                    self.ctx.push(self.status, self.progress)
                await asyncio.wait({task}, timeout=0.25)
            self.reviews = task.result()
        except Exception as exc:
            log.exception("glossary name check failed")
            self.reviews = []
            self.status.value = f"Glossary name check failed: {exc}"
            self.progress.visible = False
            self.ctx.push(self.status, self.progress)
            return []
        self.render()
        return self.reviews

    def render(self) -> None:
        import glossary_name_check as name_check

        controls: list[ft.Control] = []
        self.rows = []
        total = chapters = 0
        for review in self.reviews:
            if len(self.reviews) > 1 or review.problem:
                controls.append(section_header(os.path.basename(review.folder.rstrip("/\\")) or review.folder))
            if review.problem:
                controls.append(hint_text(review.problem, key=f"gn-problem-{len(controls)}"))
                continue
            if not review.chapters:
                controls.append(hint_text(name_check.ALL_MATCH))
                continue
            for chapter in review.chapters:
                chapters += 1
                controls.append(ft.Text(chapter.output_file, theme_style=ft.TextThemeStyle.LABEL_LARGE,
                                        color=ft.Colors.ON_SURFACE_VARIANT, selectable=True))
                for finding in chapter.reportable:
                    total += 1
                    box = ft.Checkbox(
                        label=f"{' / '.join(finding.variants)} → {finding.expected}  ×{finding.occurrences}",
                        value=finding.fixable and not self.output_file, key=f"gn-row-{len(self.rows)}")
                    self.rows.append((box, review.folder, chapter.output_file, finding))
                    controls.append(ft.Row([ft.Container(content=box, expand=True), ft.IconButton(
                        icon=ft.Icons.VISIBILITY, tooltip="Preview translation and raw source",
                        on_click=lambda e, folder=review.folder, name=chapter.output_file, found=finding:
                        self.ctx.spawn(self.preview(folder, name, found)), key=f"gn-preview-{len(self.rows) - 1}")]))
        self.results.controls = controls
        self.progress.visible = False
        if total:
            glossaries = {os.path.basename(r.glossary_path) for r in self.reviews if r.glossary_path}
            self.status.value = (f"{total} name(s) in {chapters} file(s) differ from "
                                 f"{', '.join(sorted(glossaries))}.")
        else:
            self.status.value = "Nothing to fix."
        for button in (self.check_all, self.uncheck_all, self.apply_button):
            button.disabled = not self.rows
        self.ctx.push(self.results, self.progress, self.status, self.check_all, self.uncheck_all,
                      self.apply_button)

    async def preview(self, folder: str, output_file: str, finding: Any) -> Any:
        try:
            translated, source = await self.ctx.io(_preview_snippets, folder, output_file, finding,
                                                   self.progress_path)
        except OSError as exc:
            self.ctx.say(f"Could not open the preview: {exc}")
            return None
        if self._page is not None and not self.sheet.open:
            return None
        controls = [ft.Text(f"{' / '.join(finding.variants)} → {finding.expected}", selectable=True),
                    hint_text(f"{finding.raw} · {finding.entry_type} · {finding.origin or 'Book glossary'}")]
        for label, snippets in (("In the translation", translated), ("In the raw source chapter", source)):
            if snippets is None:
                continue
            controls.append(section_header(label))
            if not snippets:
                controls.append(hint_text("No matching text found."))
            for before, match, after in snippets:
                controls.append(ft.Text(spans=[ft.TextSpan(before), ft.TextSpan(
                    match, style=ft.TextStyle(weight=ft.FontWeight.BOLD, color=ft.Colors.PRIMARY)),
                    ft.TextSpan(after)], selectable=True))
        preview = scroll_sheet(f"Preview — {output_file}", controls, key="gn-preview")
        if self.ctx.page is not None:
            self.ctx.page.show_dialog(preview)
        return preview

    def set_all(self, value: bool) -> None:
        for row in self.rows:
            row[0].value = value
        self.ctx.push(*(row[0] for row in self.rows))

    # ---- apply -------------------------------------------------------------------------------------

    def plan(self) -> dict:
        """``{folder: {output file: [Finding]}}`` of the checked rows."""
        out: dict = {}
        for box, folder, output_file, finding in self.rows:
            if box.value:
                out.setdefault(folder, {}).setdefault(output_file, []).append(finding)
        return out

    async def apply(self) -> Optional[dict]:
        import glossary_name_check as name_check

        plan = self.plan()
        if not plan:
            self.ctx.say("No rows are checked.")
            return None
        files = sum(len(chapters) for chapters in plan.values())
        answer = await ask(self.ctx, TITLE,
                           f"Replace the checked names in {files} file(s)?\n"
                           "The original files are backed up to glossary_name_check_backup.",
                           [("no", "Cancel", "text"), ("yes", "Replace", "filled")], key="gn-confirm")
        if answer != "yes":
            return None

        def run() -> dict:
            return {folder: name_check.fix_files(folder, chapters) for folder, chapters in plan.items()}

        changed = await self.ctx.io(run)
        self.applied = changed
        replaced = sum(sum(per_file.values()) for per_file in changed.values())
        changed_files = sum(len(per_file) for per_file in changed.values())
        self.close()  # first: the snackbar must outlive the sheet
        self.ctx.say(f"Replaced {replaced} name(s) in {changed_files} file(s). "
                     "Run a QA scan again to clear these chapters' QA issues.")
        return changed

    async def _on_apply(self, e: Any = None) -> Optional[dict]:
        return await self.apply()
