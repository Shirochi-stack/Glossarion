"""Fix glossary names (QA Scanner, experimental; desktop QA Scanner settings › "Fix glossary names in
existing chapters…").

Opened from the QA screen when Settings › QA Scanner Settings › "Check glossary names
(Experimental)" is on. It scans the chosen folders on the io pool
(``glossary_name_check.review_folder``: the folder's glossary with the Emergency Glossary Compliance
entry types / minimum length, chapters matched to their sources through translation_progress.json +
chapters_full.json) and lists every name that differs from the glossary as ☐ "Found → Glossary
name ×n" under its chapter file. Rows are pre-checked when one close spelling of a character name
was found (the automatic-fix case). **Apply checked** asks first, then replaces the checked names in
text (never inside tags), backing each file up to ``glossary_name_check_backup``
(``glossary_name_check.fix_files``).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional, Sequence

import flet as ft

from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.components.dialogs import close_dialog
from glossarion_mobile.ui.components.sheet import bottom_sheet, scroll_column, sheet_frame
from glossarion_mobile.ui.tools.common import ask, hint_text, section_header

__all__ = ["CHECK_KEY", "GlossaryNameFixSheet", "check_enabled", "name_check_scope"]

log = logging.getLogger("glossarion.tools.qa")

CHECK_KEY = ("qa_scanner_settings", "check_glossary_names")
SCOPE_KEYS = ("emergency_glossary_compliance_mode", "emergency_glossary_compliance_custom_types",
              "emergency_glossary_compliance_min_chars")
TITLE = "Fix Glossary Names"
INTRO = ("Checked rows are replaced with the glossary name. Rows are pre-checked when one close spelling of a "
         "character name was found; review the others before checking them. Each changed file is backed up to "
         "glossary_name_check_backup first.")


def check_enabled(ctx: Any) -> bool:
    return bool(ctx.cfg(CHECK_KEY, False))


def name_check_scope(ctx: Any) -> dict:
    """Emergency Glossary Compliance's entry types / minimum length from the saved config."""
    import glossary_name_check as name_check

    config = {}
    for key in SCOPE_KEYS:
        value = ctx.cfg(key, None)
        if value is not None:
            config[key] = value
    return name_check.settings_from_config(config)


def _review_all(folders: Sequence[str], scope: dict) -> list:
    import glossary_name_check as name_check

    return [name_check.review_folder(folder, scope) for folder in folders]


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
        try:
            self.reviews = await self.ctx.io(_review_all, self.folders, name_check_scope(self.ctx))
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
            self.status.value = (f"{total} name(s) in {chapters} chapter(s) differ from "
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
                           f"Replace the checked names in {files} chapter file(s)?\n"
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
