"""Check glossary names (QA Scanner, experimental; desktop QA Scanner settings › "Check glossary names
in existing output…").

Started from Tools, the QA Scanner, the Book menu or a chapter's Resolve QA action. The job scans the chosen folders
(``glossary_name_check.review_folder``: the folder's glossary with the Emergency Glossary Compliance
entry types / minimum length, chapters matched to their sources through translation_progress.json +
chapters_full.json). The completed job lists every name that differs from the glossary as ☐ "Found → Glossary
name ×n" under its chapter file (and translated_headers.txt / TOC.txt / metadata.json for the
chapter titles, TOC and book metadata). Rows are pre-checked when one close spelling of a character name
was found (the automatic-fix case). **Apply checked** asks first, then replaces the checked names in
text (never inside tags), backing each file up to ``glossary_name_check_backup``
(``glossary_name_check.fix_files``).
"""

from __future__ import annotations

import html
import inspect
import logging
import os
from typing import Any, Optional, Sequence

import flet as ft

from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.components.dialogs import close_dialog
from glossarion_mobile.ui.components.sheet import bottom_sheet, fitted_column, scroll_sheet, sheet_frame
from glossarion_mobile.ui.screens.base import Screen
from glossarion_mobile.ui.theme import HIT_TARGET
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


async def start_name_check(ctx: Any, folders: Sequence[str], *, output_file: str = "",
                           progress_path: Optional[str] = None) -> Optional[str]:
    from glossarion_mobile.services.jobs import JobSpec

    folders = tuple(dict.fromkeys(str(f) for f in folders if f))
    if not folders:
        ctx.say("Choose an output folder first")
        return None
    jobs = ctx.jobs
    if jobs is None or not jobs.has_kind("glossary_names"):
        ctx.say("Glossary name checks are unavailable in this session")
        return None
    with_ner = use_ner(ctx)
    if with_ner and await ctx.io(ner_needs_download):
        answer = await ask(ctx, TITLE, "Download the 104 MB named-entity model for this check?",
                           [("heuristic", "Use capitalization", "text"), ("download", "Download model", "filled")],
                           key="gn-ner-download")
        if answer not in ("heuristic", "download"):
            return None
        with_ner = answer == "download"
    label = os.path.basename(folders[0].rstrip("/\\")) if len(folders) == 1 else f"{len(folders)} folders"
    params = {"scope": name_check_scope(ctx), "ner": with_ner, "layers": glossary_layer_settings(ctx),
              "workers": chapters_at_once(ctx), "output_file": output_file, "progress_path": progress_path}
    result = jobs.submit(JobSpec(kind="glossary_names", title=label, inputs=folders, params=params))
    job_id = await result if inspect.isawaitable(result) else result
    if job_id:
        ctx.go("jobs.detail", {"jid": job_id})
    return job_id


def _preview_snippets(folder: str, output_file: str, finding: Any, progress_path: Optional[str]) -> tuple:
    import glossary_name_check as name_check

    with open(os.path.join(folder, output_file), encoding="utf-8", errors="ignore") as handle:
        markup = handle.read()
    if output_file in name_check.TITLE_FILES:
        return name_check.finding_snippets(html.escape(markup), finding), None
    source = name_check.source_chapters_for_folder(folder, progress_path).get(output_file, "")
    return name_check.finding_snippets(markup, finding), name_check.source_snippets(source, finding)


class GlossaryNamesScreen(Screen):
    title = "Glossary name mismatches"

    def __init__(self, match: Any, ctx: Any) -> None:
        super().__init__(match)
        self.ctx = ctx
        self.target = None
        self._visible = False
        self._selection_version = 0
        self._label = hint_text("Choose an output folder")
        self._run = ft.FilledButton("Check glossary names", icon=ft.Icons.SPELLCHECK,
                                    disabled=True, on_click=self._scan)

    def build_body(self) -> ft.Control:
        return ft.ListView([
            hint_text("Compare the translation with its glossary and preview name mismatches before fixing them."),
            self._label,
            ft.Row([ft.OutlinedButton("Choose output", icon=ft.Icons.FOLDER_OPEN, on_click=self._pick),
                    self._run], wrap=True),
        ], padding=12, spacing=12, expand=True)

    @staticmethod
    def _ineligible(target: Any) -> Optional[str]:
        from glossarion_mobile.ui.tools.targets import folder_has_html

        if target.direct_text or not target.folder or not folder_has_html(target.folder):
            return "Needs an output folder with translated HTML"
        return None

    def did_show(self) -> None:
        self._visible = True
        if self.target is None:
            self.ctx.spawn(self._select_recent())

    def dispose(self) -> None:
        self._visible = False

    async def _select_recent(self) -> None:
        from glossarion_mobile.ui.tools import targets as tg

        service = self.ctx.service
        version = self._selection_version
        snapshot = getattr(service, "snapshot", None)
        try:
            if snapshot is not None and not getattr(snapshot, "scanned_at", None):
                await service.refresh(quiet=True, reason="glossary names default output")

            def latest() -> Any:
                chat = tg.chat_workspace_targets(self.ctx.chats_root, service=service) if self.ctx.chats_root else []
                return next((t for t in tg.recent_output_targets(service, chat_rows=chat)
                             if self._ineligible(t) is None), None)

            target = await self.ctx.io(latest)
        except Exception:
            log.debug("finding the glossary name check workspace failed", exc_info=True)
            return
        if target is not None and self._visible and version == self._selection_version:
            self._select([target])

    def _select(self, targets: Sequence[Any]) -> None:
        if not targets:
            return
        self._selection_version += 1
        self.target = targets[0]
        self._label.value = self.target.title or os.path.basename(self.target.folder)
        self._run.disabled = False
        self.ctx.push(self._label, self._run)

    def _pick(self, _event: Any) -> None:
        from glossarion_mobile.ui.tools.source_picker import SourcePicker

        picker = SourcePicker(self.ctx, title="Choose output to check", multi=False,
                              eligible=self._ineligible, on_done=self._select, find_folder=True)
        if self.ctx.page is not None:
            picker.show(self.ctx.page)

    def _scan(self, _event: Any) -> None:
        if self.target is not None:
            self.ctx.spawn(start_name_check(self.ctx, [self.target.folder]))


class GlossaryNameFixSheet:
    def __init__(self, ctx: Any, reviews: Sequence[Any], *, output_file: str = "",
                 progress_path: Optional[str] = None) -> None:
        self.ctx = ctx
        self.output_file = output_file
        self.progress_path = progress_path
        self.reviews = list(reviews)
        self.rows: list = []  # (checkbox, folder, output file, finding)
        self.applied: Optional[dict] = None
        self.status = ft.Text("", key="gn-status")
        self.results = ft.Column(spacing=2, key="gn-results")
        self.check_all = ft.TextButton(content="Check all", disabled=True, key="gn-check-all",
                                       on_click=lambda e: self.set_all(True))
        self.uncheck_all = ft.TextButton(content="Uncheck all", disabled=True, key="gn-uncheck-all",
                                         on_click=lambda e: self.set_all(False))
        self.apply_button = ft.FilledButton(content="Apply checked", icon=ft.Icons.SPELLCHECK, disabled=True,
                                            on_click=self._on_apply, key="gn-apply")
        self.footer = ft.Row([self.check_all, self.uncheck_all, self.apply_button], alignment=ft.MainAxisAlignment.END,
                             wrap=True, spacing=8)
        self.intro = ft.Text(INTRO, theme_style=ft.TextThemeStyle.BODY_SMALL, visible=False, key="gn-intro")
        body = fitted_column([
            ft.Text("Resolve QA issue · Glossary names" if output_file else TITLE,
                    theme_style=ft.TextThemeStyle.TITLE_LARGE, weight=ft.FontWeight.W_600),
            self.intro,
            self.status,
            self.results,
            self.footer,
        ], spacing=tokens.SPACING["sm"], estimate=240)
        self.sheet = bottom_sheet(sheet_frame(body), key="gn-sheet")
        self._page: Any = None

    # ---- lifecycle ---------------------------------------------------------------------------------

    def show(self, page: Any) -> "GlossaryNameFixSheet":
        self._page = page
        self.render()
        page.show_dialog(self.sheet)
        return self

    def close(self) -> None:
        close_dialog(self._page, self.sheet)

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
                        icon=ft.Icons.VISIBILITY, tooltip="Preview translation and raw source", size_constraints=HIT_TARGET,
                        on_click=lambda e, folder=review.folder, name=chapter.output_file, found=finding:
                        self.ctx.spawn(self.preview(folder, name, found)), key=f"gn-preview-{len(self.rows) - 1}")]))
        self.results.controls = controls
        if total:
            glossaries = {os.path.basename(r.glossary_path) for r in self.reviews if r.glossary_path}
            self.status.value = (f"{total} name(s) in {chapters} file(s) differ from "
                                 f"{', '.join(sorted(glossaries))}.")
        else:
            self.status.value = "Nothing to fix."
        self.footer.visible = self.intro.visible = bool(self.rows)
        for button in (self.check_all, self.uncheck_all, self.apply_button):
            button.disabled = not self.rows

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
