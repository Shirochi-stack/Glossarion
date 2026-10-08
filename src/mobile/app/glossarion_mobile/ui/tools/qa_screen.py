"""QA Scanner (``/tools/qa``, UI_SPEC §4.4; FEATURE_MAP qa-epub-pdf 0-40).

* **Mode cards**: Quick Scan (Recommended) · Aggressive · AI Hunter · Custom - the desktop
  mode dialog's cards (``qa_model.MODE_CARDS``); Custom opens the Custom Mode Settings sheet.
* **Fields**: "Quick Scan duplicate check sample size (characters)" and "Auto-search
  output" (``qa_scanner_settings.quick_scan_sample_size`` / ``qa_auto_search_output``,
  saved like the desktop dialog saves them). On mobile the sample size shows 0 (duplicate
  check off) when none is saved and a saved desktop 1000 is turned into 0 once (owner
  2026-10-08, ``qa_model.migrate_quick_sample_size``); the chat's scans use the same value.
* **Source**: the SourcePicker (multi-select = bulk scan). Library rows carry the output
  folder and the raw source; Browse finds a picked file's folder with the auto-search.
  Direct Text workspaces are listed but not picked here (desktop rule); a chat scans its own
  workspace from the chat (``qa_model.chat_qa_job``).
* **Start**: the desktop pre-run questions with their texts (word count without a source:
  continue without word count / pick a source; source/folder name mismatch), then a
  ``qa_scan`` job (``job_kinds.qa``). While it runs: phase, Stop (the JobService stop:
  graceful first, then force) and the job log.
* **Settings**: Settings › QA Scanner Settings / AI Hunter (the schema sections, with the
  phrase editors and "Reset this section to defaults"); silent-truncation embeddings and the
  thread executor are shown with their mobile reasons.
* **Reports**: the reports of the Library's output folders (newest first) and the shared
  latest-report search (``qa_scan_runtime.find_latest_qa_report``); a report opens the QA
  report viewer.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Optional

import flet as ft

from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.components.action_sheet import ActionItem, ActionSheet
from glossarion_mobile.ui.components.reason_chip import ReasonChip
from glossarion_mobile.ui.router import RouteMatch
from glossarion_mobile.ui.screens.base import Screen
from glossarion_mobile.ui.theme import HIT_TARGET
from glossarion_mobile.ui.tools import qa_model as qm
from glossarion_mobile.ui.tools import targets as tg
from glossarion_mobile.ui.tools.common import JobWatch, action_button, ask, card, hint_text
from glossarion_mobile.ui.tools.qa_custom import CustomModeSheet
from glossarion_mobile.ui.tools.source_picker import SourcePicker

__all__ = ["QaScannerScreen", "open_qa_report", "qa_eligibility"]

log = logging.getLogger("glossarion.tools.qa")

EMBEDDINGS_REASON = ("The embeddings method of silent truncation needs sentence-transformers (torch), which "
                     "cannot run on Android/iOS. The heuristic method always works.")
THREADS_REASON = "Android and iOS have no process pools: QA scans always use threads (at most 2 workers)."


def qa_eligibility(target: tg.ToolTarget) -> Optional[str]:
    """Why a row cannot be QA-scanned (None when it can)."""
    if target.direct_text:
        return "Direct Text workspaces are excluded from QA scans"
    if not target.folder:
        return "No output folder to scan"
    return None


def open_qa_report(ctx: Any, path: str) -> Optional[str]:
    """Open a ``validation_results.html`` in the QA report viewer (``/tools/qa/report/<rid>``).

    ``ctx`` needs ``prefs`` (the FileRef registry: a route never carries a path; also found as
    ``ctx.env.prefs``), ``go`` / ``navigate(name, params)`` and ``say`` / ``notify(message)`` - a
    ToolsContext / LibraryContext, or the ChatView: the QA screen's report rows and the chat's QA
    card use it. Returns the FileRef id, or None (no Prefs, or no path).
    """
    prefs = getattr(ctx, "prefs", None)
    if prefs is None:
        prefs = getattr(getattr(ctx, "env", None), "prefs", None)
    go = getattr(ctx, "go", None) or getattr(ctx, "navigate", None)
    say = getattr(ctx, "say", None) or getattr(ctx, "notify", None)
    if prefs is None or not path or go is None:
        if say is not None:
            say("Reports cannot be opened in this session")
        return None
    rid = prefs.file_ref(path, kind="qa_report")
    go("tools.qa.report", {"rid": rid})
    return rid


def _when(mtime: float) -> str:
    if not mtime:
        return ""
    now = time.time()
    day = time.strftime("%Y-%m-%d", time.localtime(mtime))
    if day == time.strftime("%Y-%m-%d", time.localtime(now)):
        return "Today " + time.strftime("%H:%M", time.localtime(mtime))
    return time.strftime("%b %d, %H:%M", time.localtime(mtime))


class QaScannerScreen(Screen):
    title = "QA scanner"

    def __init__(self, match: Optional[RouteMatch], ctx: Any) -> None:
        super().__init__(match)
        self.ctx = ctx
        state = ctx.tool_state.setdefault("qa", {})
        self.state = state
        self.mode = state.get("mode") or "quick-scan"
        self.targets: list = list(state.get("targets") or [])
        self.watch = JobWatch(ctx, self._on_job_end, self._on_job_change)
        self.reports: list = []
        self.summaries: dict = {}
        self.mode_cards: dict = {}
        self.report_rows: dict = {}
        self.active_job: Any = None
        self.last_spec: Any = None
        self.picker: Optional[SourcePicker] = None
        self.custom_sheet: Optional[CustomModeSheet] = None

    # ---- layout -----------------------------------------------------------------------------------

    def actions(self) -> list:
        return [ft.IconButton(icon=ft.Icons.TUNE, tooltip="QA Scanner Settings", key="qa-settings-action",
                              on_click=lambda e: self.open_settings("qa.settings"), size_constraints=HIT_TARGET)]

    def build_body(self) -> ft.Control:
        self.mode_row = ft.ResponsiveRow(spacing=8, run_spacing=8, key="qa-modes")
        self._render_modes()
        # owner 2026-10-08: a saved desktop 1000 becomes 0 once (also done at app start); nothing
        # saved shows the mobile default 0 (duplicate check off) - the value the job scans with
        if self.ctx.store is not None:
            qm.migrate_quick_sample_size(self.ctx.cfg, self.ctx.set_cfg, self.ctx.prefs)
        sample = self.ctx.cfg(qm.QUICK_SAMPLE_KEY, None)
        if sample is None:
            sample = qm.MOBILE_QUICK_SAMPLE_SIZE
        self.sample_field = ft.TextField(label=qm.QUICK_SAMPLE_LABEL, value=str(sample), dense=True,
                                         keyboard_type=ft.KeyboardType.NUMBER, helper=qm.QUICK_SAMPLE_HINT,
                                         on_blur=self._on_sample, on_submit=self._on_sample, key="qa-sample")
        self.auto_search = ft.Switch(label="Auto-search output", value=bool(self.ctx.cfg("qa_auto_search_output", True)),
                                     on_change=self._on_auto_search, key="qa-auto-search")
        fields = card("Scan options", [self.sample_field, self.auto_search], icon="TUNE", key="qa-fields")
        self.targets_column = ft.Column(spacing=4, key="qa-targets")
        self.choose_button = ft.FilledTonalButton(content="Choose…", icon=ft.Icons.FOLDER_OPEN,
                                                  on_click=lambda e: self.open_picker(), key="qa-choose")
        self.clear_button = ft.TextButton(content="Clear", on_click=lambda e: self.set_targets([]), key="qa-clear")
        source = card("Folders to scan", [self.targets_column, ft.Row([self.choose_button, self.clear_button],
                                                                         wrap=True)],
                      icon="FOLDER", key="qa-source",
                      subtitle="Pick several books for a bulk scan; each folder is checked against its raw source.")
        self.start_button = ft.FilledButton(content="Start QA scan", icon=ft.Icons.PLAY_ARROW,
                                            on_click=self._on_start, key="qa-start")
        self.run_status = ft.Text("", theme_style=ft.TextThemeStyle.BODY_MEDIUM, key="qa-run-status")
        self.stop_button = ft.OutlinedButton(content="Stop", icon=ft.Icons.STOP, on_click=self._on_stop,
                                             visible=False, key="qa-stop")
        self.log_button = ft.TextButton(content="View log", icon=ft.Icons.TERMINAL, on_click=self._on_view_log,
                                        visible=False, key="qa-log")
        self.progress = ft.ProgressBar(visible=False, key="qa-progress")
        run = card("Run", [ft.Row([self.start_button, self.stop_button, self.log_button], wrap=True),
                           self.progress, self.run_status], icon="PLAY_CIRCLE", key="qa-run")
        settings = card("Settings", [
            ft.Row([
                action_button("QA Scanner Settings", "SETTINGS", lambda e: self.open_settings("qa.settings"),
                              key="qa-open-settings"),
                action_button("AI Hunter", "SMART_TOY", lambda e: self.open_settings("qa.ai_hunter"),
                              key="qa-open-ai-hunter"),
                action_button("Custom thresholds", "TUNE", lambda e: self.open_custom(),
                              key="qa-open-custom",
                              reason=None if qm.custom_defaults() is not None
                              else "Needs the shared Custom-mode defaults"),
            ], wrap=True, spacing=8, run_spacing=8),
            ft.Row([hint_text("Silent truncation: heuristic method"),
                    ReasonChip(reason="Embeddings: not on mobile", detail=EMBEDDINGS_REASON)], wrap=True,
                   key="qa-embeddings"),
            ft.Row([hint_text("Use threads instead of processes: on"),
                    ReasonChip(reason="Locked on mobile", detail=THREADS_REASON)], wrap=True, key="qa-threads"),
        ], icon="SETTINGS", key="qa-settings",
            subtitle="Phrase editors, word-count multipliers and Reset to default live in the settings pages.")
        self.reports_column = ft.Column(spacing=2, key="qa-reports")
        self.latest_button = ft.TextButton(content="📁 Open QA Report", on_click=self._on_open_latest,
                                           key="qa-open-latest")
        reports = card("Reports", [self.reports_column, self.latest_button], icon="DESCRIPTION", key="qa-reports-card")
        self._render_targets()
        self._render_run()
        self.body_list = ft.ListView(
            controls=[ft.Text("Detection mode", theme_style=ft.TextThemeStyle.TITLE_SMALL, color=ft.Colors.PRIMARY,
                              weight=ft.FontWeight.W_600),
                      hint_text("Choose how sensitive the duplicate detection should be"),
                      self.mode_row, fields, source, run, settings, reports],
            expand=True, spacing=tokens.SPACING["md"], padding=ft.Padding.symmetric(horizontal=12, vertical=8),
            key="qa-body")
        return self.body_list

    def _render_modes(self) -> None:
        cards = []
        self.mode_cards = {}
        for value in qm.DISPLAY_ORDER:
            mode = qm.MODES_BY_VALUE[value]
            selected = value == self.mode
            texts: list[ft.Control] = [
                ft.Row([ft.Text(mode.emoji, size=22), ft.Text(mode.title, weight=ft.FontWeight.W_700,
                                                               theme_style=ft.TextThemeStyle.TITLE_SMALL)],
                       spacing=6),
                ft.Text(mode.subtitle, theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT),
            ]
            if mode.recommendation:
                texts.append(ft.Container(
                    content=ft.Text(mode.recommendation, theme_style=ft.TextThemeStyle.LABEL_SMALL,
                                    color=ft.Colors.ON_PRIMARY_CONTAINER),
                    bgcolor=ft.Colors.PRIMARY_CONTAINER, border_radius=tokens.RADII["badge"],
                    padding=ft.Padding.symmetric(horizontal=6, vertical=2)))
            texts.append(ft.Text("\n".join(mode.features[:3]), theme_style=ft.TextThemeStyle.BODY_SMALL,
                                 max_lines=3))
            container = ft.Container(
                content=ft.Column(texts, spacing=4, tight=True),
                padding=10,
                border_radius=tokens.RADII["card"],
                bgcolor=ft.Colors.SECONDARY_CONTAINER if selected else ft.Colors.SURFACE_CONTAINER_LOW,
                border=ft.Border.all(2, ft.Colors.PRIMARY) if selected else ft.Border.all(1, ft.Colors.OUTLINE_VARIANT),
                on_click=lambda e, v=value: self.select_mode(v),
                ink=True,
                col={"xs": 6, "md": 3},
                key=f"qa-mode-{value}",
            )
            self.mode_cards[value] = container
            cards.append(ft.Semantics(content=container, selected=selected, button=True,
                                      label=f"{mode.title} mode", col={"xs": 6, "md": 3}))
        self.mode_row.controls = cards

    def _render_targets(self) -> None:
        rows: list[ft.Control] = []
        settings = qm.effective_settings(self.ctx.config_snapshot())
        for target in self.targets:
            lines = [f"📁 {target.folder_name or '(no output folder)'}"]
            lines.append(f"📖 {target.source_name}" if target.source else "📖 No source file")
            trailing = None
            if target.source and target.folder and not qm.names_match(target.source, target.folder, settings):
                trailing = ReasonChip(reason="Name mismatch", detail=qm.mismatch_text(target.source, target.folder))
            rows.append(ft.ListTile(
                leading=ft.Icon(ft.Icons.MENU_BOOK), title=ft.Text(target.title, max_lines=2),
                subtitle=ft.Text(" · ".join(lines), max_lines=2, theme_style=ft.TextThemeStyle.BODY_SMALL),
                trailing=trailing or ft.IconButton(icon=ft.Icons.CLOSE, tooltip="Remove",
                                                   on_click=lambda e, t=target: self.remove_target(t), size_constraints=HIT_TARGET),
                on_click=lambda e, t=target: self.target_actions(t),
                dense=True, key=f"qa-target-{target.key}"))
        if not rows:
            rows.append(hint_text("No folder chosen yet.", key="qa-no-target"))
        elif len(rows) > 1:
            rows.insert(0, hint_text(f"Bulk scan: {len(rows)} folders", key="qa-bulk"))
        self.targets_column.controls = rows
        self.clear_button.disabled = not self.targets
        self.ctx.push(self.targets_column, self.clear_button)

    def _render_run(self) -> None:
        snap = self.active_job
        running = snap is not None and not getattr(snap, "is_terminal", True)
        kind_ok = self.ctx.has_kind("qa_scan")
        reason = None if kind_ok else "The job service is not running"
        if not self.targets:
            reason = reason or "Choose a folder to scan"
        self.start_button.disabled = reason is not None or running
        self.start_button.content = (f"Start QA scan ({len(self.targets)})" if len(self.targets) > 1
                                     else "Start QA scan")
        self.stop_button.visible = running
        self.log_button.visible = snap is not None
        self.progress.visible = running
        if running:
            from glossarion_mobile.services.jobs import progress_line

            self.run_status.value = f"{snap.state_label} · {progress_line(snap) or snap.phase}"
        elif snap is not None:
            text = f"{snap.state_label}: {snap.title}"
            if getattr(snap, "error", None):
                text += f" — {snap.error}"
            self.run_status.value = text
        else:
            self.run_status.value = reason or f"Mode: {qm.MODES_BY_VALUE[self.mode].title}"
        self.ctx.push(self.start_button, self.stop_button, self.log_button, self.progress, self.run_status)

    # ---- lifecycle --------------------------------------------------------------------------------

    def did_show(self) -> None:
        self.watch.start()
        # a QA scan queued or running from an earlier visit (Stop, Start disabled until it ends)
        adopted = self.watch.adopt(("qa_scan",))
        if adopted and (self.active_job is None or getattr(self.active_job, "is_terminal", True)):
            self.active_job = adopted[0]
            self._render_run()
        out = self.match.get("out") if self.match is not None else None
        if out and not self.targets:
            self.ctx.spawn(self.preselect_book(out))
        self.ctx.spawn(self.load_reports())

    def dispose(self) -> None:
        self.watch.stop()

    async def preselect_book(self, bid: str) -> Optional[tg.ToolTarget]:
        service = self.ctx.service
        if service is None:
            return None
        book = service.book_for_bid(bid)
        if not book:
            return None
        target = await self.ctx.io(tg.target_for_book, service, book)
        if target is not None and qa_eligibility(target) is None:
            self.set_targets([target])
        return target

    # ---- modes / fields ---------------------------------------------------------------------------

    def select_mode(self, value: str) -> None:
        if value not in qm.MODES_BY_VALUE:
            return
        self.mode = value
        self.state["mode"] = value
        self._render_modes()
        self.ctx.push(self.mode_row)
        self._render_run()
        if value == "custom":
            self.open_custom()

    def open_custom(self, on_saved: Any = None) -> Optional[CustomModeSheet]:
        if qm.custom_defaults() is None:
            self.ctx.say("This build has no Custom-mode defaults")
            return None
        self.custom_sheet = CustomModeSheet(self.ctx, on_saved=on_saved)
        if self.ctx.page is not None:
            self.custom_sheet.show(self.ctx.page)
        return self.custom_sheet

    def _on_sample(self, e: Any = None) -> bool:
        """Persist the sample size field (blur / submit, and at Start: a touch on Start does not unfocus
        the field on Android / iOS). False when the text is not a whole number."""
        raw = str(self.sample_field.value or "").strip()
        try:
            value = int(raw)
        except ValueError:
            self.sample_field.error = "Whole number (-1 = all text, 0 = off)"
            self.ctx.push(self.sample_field)
            return False
        self.sample_field.error = None
        self.ctx.push(self.sample_field)
        # Desktop: the mode handler persists ONLY this key (never the whole settings snapshot).
        # Mobile: the untouched default (nothing saved, 0 shown) stays unsaved, so config.json gets
        # no value the owner never chose.
        if self.ctx.cfg(qm.QUICK_SAMPLE_KEY, None) is None and value == qm.MOBILE_QUICK_SAMPLE_SIZE:
            return True
        self.ctx.set_cfg(qm.QUICK_SAMPLE_KEY, value)
        return True

    def _on_auto_search(self, e: Any = None) -> None:
        self.ctx.set_cfg("qa_auto_search_output", bool(self.auto_search.value))

    def open_settings(self, section: str) -> None:
        settings = self.ctx.settings
        if settings is not None and hasattr(settings, "open_setting"):
            settings.open_setting(section)
            return
        self.ctx.go("settings.section", {"section": section})

    # ---- sources ---------------------------------------------------------------------------------

    def open_picker(self) -> SourcePicker:
        self.picker = SourcePicker(self.ctx, title="Choose folders to scan", multi=True, eligible=qa_eligibility,
                                   on_done=self.set_targets, selected=self.targets,
                                   find_folder=bool(self.ctx.cfg("qa_auto_search_output", True)))
        if self.ctx.page is not None:
            self.picker.show(self.ctx.page)
        return self.picker

    def set_targets(self, targets: Any) -> None:
        self.targets = [t for t in targets or () if qa_eligibility(t) is None]
        self.state["targets"] = list(self.targets)
        self._render_targets()
        self._render_run()

    def remove_target(self, target: tg.ToolTarget) -> None:
        self.set_targets([t for t in self.targets if t.key != target.key])

    def target_actions(self, target: tg.ToolTarget) -> ActionSheet:
        """Row menu: change the source file used for the word-count / truncation checks, or remove."""
        sheet = ActionSheet([
            ActionItem("Select source file…", lambda: self.ctx.spawn(self.pick_source_for(target)),
                       icon="FILE_OPEN", key="qa-target-source"),
            ActionItem("Scan without a source", lambda: self.set_targets(
                [t.with_source("") if t.key == target.key else t for t in self.targets]),
                icon="LINK_OFF", disabled_reason=None if target.source else "No source file set",
                key="qa-target-no-source"),
            ActionItem("Remove", lambda: self.remove_target(target), icon="CLOSE", destructive=True,
                       key="qa-target-remove"),
        ], title=target.title, subtitle=target.source_name or "No source file", tablet=self.ctx.tablet)
        if self.ctx.page is not None:
            sheet.show(self.ctx.page)
        return sheet

    async def pick_source_for(self, target: tg.ToolTarget) -> Optional[tg.ToolTarget]:
        """"Select a different source": a picked EPUB/TXT/PDF/HTML becomes the target's source."""
        files = self.ctx.files
        if files is None:
            self.ctx.say("Picking files is not available in this session")
            return None
        try:
            picked = await files.pick_files(target="inbox", allowed_extensions=["epub", "html", "htm", "xhtml",
                                                                                "txt", "pdf"],
                                            allow_multiple=False, dialog_title="Select Source EPUB or HTML File")
        except Exception as exc:
            self.ctx.say(f"Could not pick the file: {exc}")
            return None
        path = next((getattr(f, "path", None) for f in picked or () if getattr(f, "path", None)), None)
        if not path:
            return None
        updated = target.with_source(path)
        self.set_targets([updated if t.key == target.key else t for t in self.targets])
        return updated

    # ---- run ----------------------------------------------------------------------------------------

    async def start(self) -> Optional[str]:
        """Pre-run questions (desktop texts), then the ``qa_scan`` job."""
        if not self.targets:
            self.ctx.say("Choose a folder to scan")
            self.open_picker()
            return None
        if not self.ctx.has_kind("qa_scan"):
            self.ctx.say("The job service is not running")
            return None
        if not self._on_sample():  # the value typed last, even when the field still has focus
            self.ctx.say("Fix the Quick Scan sample size first")
            return None
        if self.mode == "custom" and not self.ctx.cfg(("qa_scanner_settings", "custom_mode_settings"), None):
            # Desktop: Custom always goes through its dialog; nothing saved yet means "Save" first.
            self.open_custom(on_saved=lambda _saved: self.ctx.spawn(self.start()))
            return None
        settings = qm.effective_settings(self.ctx.config_snapshot())
        disable_word_count = False
        word_count = bool(settings.get("check_word_count_ratio", False))
        if word_count and not any(t.source for t in self.targets):
            answer = await ask(self.ctx, qm.NO_SOURCE_TITLE,
                               qm.no_source_text() + "\n\nWould you like to:\n"
                               "• Continue scan without word count analysis\n"
                               "• Select a source EPUB/HTML file now\n"
                               "• Cancel the scan",
                               [("cancel", "Cancel", "text"), ("pick", "Select source…", "text"),
                                ("continue", "Continue without word count", "filled")], key="qa-no-source")
            if answer == "pick" and len(self.targets) == 1:
                updated = await self.pick_source_for(self.targets[0])
                if updated is None:
                    retry = await ask(self.ctx, "No File Selected",
                                      "No EPUB/HTML file was selected.\n\n"
                                      "Do you want to continue the scan without word count analysis?",
                                      [("no", "No", "text"), ("yes", "Yes", "filled")], key="qa-no-file")
                    if retry != "yes":
                        self.ctx.say("⚠️ QA scan canceled.")
                        return None
                    disable_word_count = True
            elif answer == "continue" or (answer == "pick" and len(self.targets) > 1):
                disable_word_count = True
            else:
                self.ctx.say("⚠️ QA scan canceled.")
                return None
        if (len(self.targets) == 1 and not disable_word_count and word_count
                and settings.get("warn_name_mismatch", True)):
            target = self.targets[0]
            if target.source and not qm.names_match(target.source, target.folder, settings):
                answer = await ask(self.ctx, qm.MISMATCH_TITLE,
                                   qm.mismatch_text(target.source, target.folder) + "\n\nWould you like to:\n"
                                   "• Continue anyway (I'm sure these match)\n"
                                   "• Select a different source file\n"
                                   "• Cancel the scan",
                                   [("cancel", "Cancel", "text"), ("pick", "Select different…", "text"),
                                    ("continue", "Continue anyway", "filled")], key="qa-mismatch")
                if answer == "pick":
                    updated = await self.pick_source_for(target)
                    if updated is None:
                        proceed = await ask(self.ctx, "No File Selected",
                                            "No EPUB file was selected.\n\n"
                                            "Continue scan without word count analysis?",
                                            [("no", "No", "text"), ("yes", "Yes", "filled")], key="qa-no-file")
                        if proceed != "yes":
                            self.ctx.say("⚠️ QA scan canceled.")
                            return None
                        disable_word_count = True
                elif answer != "continue":
                    self.ctx.say("⚠️ QA scan canceled due to source/folder mismatch.")
                    return None
        spec = qm.qa_spec(self.targets, self.mode, disable_word_count=disable_word_count)
        self.last_spec = spec
        job_id = await self.ctx.submit(spec)
        if not job_id:
            return None
        self.watch.watch(job_id)
        jobs = self.ctx.jobs
        snapshot = getattr(jobs, "snapshot", None) if jobs is not None else None
        self.active_job = snapshot(job_id) if callable(snapshot) else None
        self.ctx.remember_source("tools.qa", spec.title)
        self.ctx.haptic("medium_impact")
        self.ctx.say(f"QA scan · {spec.title}", "Jobs", lambda: self.ctx.go("jobs.detail", {"jid": job_id}))
        self._render_run()
        return job_id

    async def _on_start(self, e: Any = None) -> Optional[str]:
        try:
            return await self.start()
        except Exception as exc:
            log.exception("starting the QA scan failed")
            self.ctx.say(f"Could not start: {exc}")
            return None

    def _on_stop(self, e: Any = None) -> Any:
        snap = self.active_job
        jobs = self.ctx.jobs
        if snap is None or jobs is None:
            return None
        try:
            mode = jobs.request_stop(snap.id)
        except Exception as exc:
            self.ctx.say(f"Could not stop: {exc}")
            return None
        if mode == "graceful":
            self.ctx.say("⏳ Graceful stop — waiting for in-flight QA scan API calls to complete... "
                         "tap Stop again to force")
        else:
            self.ctx.say("⛔ QA scan stop requested.")
        return mode

    def _on_view_log(self, e: Any = None) -> None:
        if self.active_job is not None:
            self.ctx.go("jobs.detail", {"jid": self.active_job.id})

    def _on_job_change(self, snap: Any) -> None:
        self.active_job = snap
        self._render_run()

    def _on_job_end(self, snap: Any) -> None:
        self.active_job = snap
        self._render_run()
        reports = list((getattr(snap, "result", {}) or {}).get("qa_reports") or ())
        self.ctx.spawn(self.load_reports())
        if reports and str(getattr(getattr(snap, "state", None), "value", "")) == "DONE":
            self.ctx.say("✅ QA scan completed", "Open report", lambda: self.open_report(reports[0]))

    # ---- reports ------------------------------------------------------------------------------------

    async def load_reports(self) -> list:
        service = self.ctx.service
        roots = [self.ctx.output_root] if self.ctx.output_root else []

        def gather() -> tuple:
            folders: list = []
            if service is not None:
                try:
                    books = service.snapshot.all_books()
                except Exception:
                    books = ()
                folders = [str(b.get("output_folder") or "") for b in books or () if b.get("output_folder")]
            entries = qm.list_reports(folders, roots, limit=30)
            return entries, {e.path: qm.load_report_summary(e.path) for e in entries[:30]}

        try:
            self.reports, self.summaries = await self.ctx.io(gather)
        except Exception:
            log.exception("listing the QA reports failed")
            self.reports, self.summaries = [], {}
        self._render_reports()
        return self.reports

    def _render_reports(self) -> None:
        rows: list[ft.Control] = []
        self.report_rows = {}
        for entry in self.reports:
            summary = self.summaries.get(entry.path)
            subtitle = _when(entry.mtime)
            if summary is not None:
                subtitle = f"{subtitle} · {summary.headline}" if subtitle else summary.headline
            tile = ft.ListTile(leading=ft.Icon(ft.Icons.DESCRIPTION_OUTLINED), title=ft.Text(entry.title, max_lines=1),
                               subtitle=ft.Text(subtitle, theme_style=ft.TextThemeStyle.BODY_SMALL, max_lines=2),
                               on_click=lambda e, p=entry.path: self.open_report(p), dense=True,
                               key=f"qa-report-{entry.title}")
            self.report_rows[entry.path] = tile
            rows.append(tile)
        if not rows:
            rows.append(hint_text("No QA reports yet.", key="qa-no-reports"))
        self.reports_column.controls = rows
        self.ctx.push(self.reports_column)

    def open_report(self, path: str) -> Optional[str]:
        return open_qa_report(self.ctx, path)

    async def open_latest(self) -> Optional[str]:
        """Desktop "📁 Open QA Report": the newest ``validation_results.html`` under the output root."""
        try:
            import qa_scan_runtime
        except Exception:
            qa_scan_runtime = None  # type: ignore[assignment]
        find = getattr(qa_scan_runtime, "find_latest_qa_report", None) if qa_scan_runtime is not None else None
        root = self.ctx.output_root or os.environ.get("OUTPUT_DIRECTORY") or ""
        last = self.state.get("last_report")
        path = None
        if callable(find):
            path = await self.ctx.io(find, root or None, last)
        elif self.reports:
            path = self.reports[0].path
        if not path:
            self.ctx.say("No QA report found. Run a scan first.")
            return None
        self.state["last_report"] = path
        return self.open_report(path)

    async def _on_open_latest(self, e: Any = None) -> Optional[str]:
        return await self.open_latest()
