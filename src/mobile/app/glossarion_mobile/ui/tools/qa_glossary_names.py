"""Check glossary names (QA Scanner, experimental; desktop QA Scanner settings › "Check glossary names
in existing output…").

Opened from the QA screen when Settings › QA Scanner Settings › "Check glossary names with NER
(Experimental)" is on. It scans the chosen folders on the io pool
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
import logging
import os
from typing import Any, Optional, Sequence

import flet as ft

from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.components.dialogs import close_dialog
from glossarion_mobile.ui.components.sheet import bottom_sheet, scroll_column, sheet_frame
from glossarion_mobile.ui.tools.common import ask, hint_text, section_header

__all__ = ["CHECK_KEY", "GlossaryNameFixSheet", "NER_KEY", "chapters_at_once", "check_enabled", "name_check_scope",
           "use_ner"]

log = logging.getLogger("glossarion.tools.qa")

CHECK_KEY = ("qa_scanner_settings", "check_glossary_names")
NER_KEY = "glossary_name_check_use_ner"
SCOPE_KEYS = ("emergency_glossary_compliance_mode", "emergency_glossary_compliance_custom_types",
              "glossary_name_check_min_chars", "glossary_name_check_autofix_level",
              "glossary_name_check_match_numbers")
TITLE = "Check Glossary Names"
INTRO = ("Checked rows are replaced with the glossary name. Rows are pre-checked when one close spelling of a "
         "character name was found; review the others before checking them. Each changed file is backed up to "
         "glossary_name_check_backup first.")


def check_enabled(ctx: Any) -> bool:
    return bool(ctx.cfg(CHECK_KEY, False))


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


def _review_all(folders: Sequence[str], scope: dict, with_ner: bool = False,
                state: Optional[dict] = None, layer_settings: Optional[dict] = None, chosen: int = 0) -> list:
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
                                                settings=layer_settings))
    return reviews


class GlossaryNameFixSheet:
    def __init__(self, ctx: Any, folders: Sequence[str]) -> None:
        self.ctx = ctx
        self.folders = [str(f) for f in folders if f]
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
            ft.Text(TITLE, theme_style=ft.TextThemeStyle.TITLE_LARGE, weight=ft.FontWeight.W_600),
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
            state.update(text="Downloading the named-entity model (NER, 104 MB)…", value=None)
        try:
            task = asyncio.ensure_future(
                self.ctx.io(_review_all, self.folders, name_check_scope(self.ctx), with_ner, state,
                            glossary_layer_settings(self.ctx), chapters_at_once(self.ctx)))
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
                        value=finding.fixable, key=f"gn-row-{len(self.rows)}")
                    self.rows.append((box, review.folder, chapter.output_file, finding))
                    controls.append(box)
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
