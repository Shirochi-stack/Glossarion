"""Headers & metadata (``/tools/headers``, UI_SPEC §4.5; FEATURE_MAP qa-epub-pdf 54-58).

* **Books**: one or more Library books (raw EPUB + output folder) from the SourcePicker, each
  with its cache status (translated_headers.txt / TOC.txt / metadata.json).
* **Chapter headers**: "Translate Headers Now" (a ``translate_headers`` job; Stop while it
  runs; the EPUB is rebuilt afterwards like the desktop button) · "Delete Header Files" ·
  "Delete TOC.txt" (desktop summaries and the RECYCLED-link question) · the header settings
  as schema tiles.
* **Metadata**: Translate Book Title / Metadata, the translation mode (together / metadata
  separately / parallel - the desktop radio labels), "Metadata fields…" (the desktop
  "Configure Metadata Translation" field picker: detected standard + custom fields per EPUB),
  "Configure All Prompts" (the prompt tiles of its tabs) and "Translate Metadata (N EPUBs)"
  (a ``metadata`` job; "Metadata Already Exists" asks first, as in the desktop Library).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import flet as ft

from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.components.dialogs import ConfirmDialog, close_dialog
from glossarion_mobile.ui.router import RouteMatch
from glossarion_mobile.ui.screens.base import Screen
from glossarion_mobile.ui.theme import HIT_TARGET
from glossarion_mobile.ui.tools import headers_model as hm
from glossarion_mobile.ui.tools import targets as tg
from glossarion_mobile.ui.tools.common import (
    JobWatch,
    action_button,
    ask,
    card,
    failed_job_card,
    hint_text,
    job_failed,
    schema_tiles,
)
from glossarion_mobile.ui.tools.source_picker import SourcePicker

__all__ = ["HeadersScreen", "MetadataFieldsSheet", "PromptsSheet", "header_help_markdown", "headers_eligibility"]


def header_help_markdown() -> tuple:
    """``(title, markdown)`` of the desktop "Header Translation - Help" guide (the shared
    ``metadata_defaults.HEADER_HELP_SECTIONS``, moved out of other_settings.HeaderTranslationHelpDialog)."""
    try:
        from metadata_defaults import HEADER_HELP_SECTIONS, HEADER_HELP_TITLE
    except Exception:
        return "Header Translation - Help", "The header translation guide is not in this build."
    parts = []
    for section in HEADER_HELP_SECTIONS:
        parts.append(f"**{section.get('title', '')}**")
        parts.extend(str(line) for line in section.get("content", ()))
        parts.append("")
    return HEADER_HELP_TITLE, "\n\n".join(p for p in parts if p is not None).strip()

log = logging.getLogger("glossarion.tools.headers")

HEADER_SETTING_KEYS = (
    "batch_translate_headers",
    "headers_per_batch",
    "update_html_headers",
    "save_header_translations",
    "failed_translation_retry_attempts",
    "use_toc_ncx",
    "toc_ncx_per_batch",
)
METADATA_SETTING_KEYS = ("translate_book_title",)


def headers_eligibility(target: tg.ToolTarget) -> Optional[str]:
    if not target.source:
        return "Needs the raw EPUB"
    if not target.is_epub_source:
        return "EPUB books only"
    return None


class PromptsSheet:
    """"Configure All Prompts": the prompt tiles of the desktop dialog's tabs (Book Title, Chapter Headers,
    Metadata Fields, ⚙️ Advanced) and its "Reset all prompts to defaults"."""

    RESET_TITLE = "Reset All Prompts"
    RESET_BODY = "Are you sure you want to reset ALL prompts to their default values?\n\nThis cannot be undone."

    def __init__(self, ctx: Any) -> None:
        self.ctx = ctx
        self.tiles: dict = {}
        sections: list[ft.Control] = []
        for title, keys in hm.PROMPT_GROUPS:
            controls, tiles = schema_tiles(ctx, keys)
            self.tiles.update(tiles)
            sections.append(card(title, controls or [hint_text("Settings are not available in this session.")],
                                 key=f"prompts-{title}"))
        content = ft.Column([ft.Text("Configure All Prompts", theme_style=ft.TextThemeStyle.TITLE_LARGE,
                                     weight=ft.FontWeight.W_600),
                             hint_text("{target_lang} is replaced with the output language."),
                             *sections,
                             ft.Row([ft.TextButton(content="Reset all prompts to defaults", icon=ft.Icons.RESTART_ALT,
                                                   on_click=lambda e: self.ctx.spawn(self.confirm_reset()),
                                                   key="prompts-reset"),
                                     ft.TextButton(content="Close", on_click=lambda e: self.close(),
                                                   key="prompts-close")],
                                    alignment=ft.MainAxisAlignment.END, wrap=True)],
                            tight=True, spacing=tokens.SPACING["sm"], scroll=ft.ScrollMode.AUTO)
        self.sheet = ft.BottomSheet(content=ft.Container(content=content, padding=tokens.SPACING["sheet_padding"]),
                                    show_drag_handle=True, scrollable=True, bgcolor=ft.Colors.SURFACE_CONTAINER_HIGH)
        self._page: Any = None

    def show(self, page: Any) -> "PromptsSheet":
        self._page = page
        page.show_dialog(self.sheet)
        return self

    def close(self) -> None:
        close_dialog(self._page, self.sheet)

    async def confirm_reset(self) -> bool:
        """The desktop "Reset all prompts to defaults" (Yes / No, default No) -> ``reset()``."""
        answer = await ask(self.ctx, self.RESET_TITLE, self.RESET_BODY,
                           (("yes", "Yes", "destructive"), ("no", "No", "text")), key="prompts-reset")
        if answer != "yes":
            return False
        return self.reset()

    def reset(self) -> bool:
        """Remove the prompt keys, blank the book title prompt and re-seed the defaults
        (``headers_model.reset_prompt_changes``), then refresh the tiles."""
        settings = getattr(self.ctx, "settings", None)
        store = getattr(settings, "store", None) if settings is not None else None
        if store is None:
            return False
        try:
            removed, written = hm.reset_prompt_changes(store.snapshot())
        except Exception:
            log.exception("resetting the metadata prompts failed")
            return False
        for key in removed:
            store.unset(key)
        if written:
            store.set_many(dict(written))
        for tile in self.tiles.values():
            try:
                tile.refresh()
            except Exception:
                pass
        say = getattr(self.ctx, "say", None)
        if callable(say):
            say("All prompts reset to their defaults")
        return True


class MetadataFieldsSheet:
    """"Configure Metadata Translation" field picker (per EPUB, desktop rules in ``headers_model``)."""

    def __init__(self, ctx: Any, epubs: list) -> None:
        self.ctx = ctx
        self.epubs = [t.source for t in epubs if t.source]
        self.titles = {t.source: t.title for t in epubs if t.source}
        self.previous = dict(ctx.cfg("translate_metadata_fields", {}) or {})
        self.sync: dict = {}
        self.selections: dict = {}  # epub path -> {field: bool} (visited EPUBs, like the desktop dialog)
        self.detected: dict = {}
        self.current = self.epubs[0] if self.epubs else ""
        self.checkboxes: dict = {}
        self.fields_column = ft.Column(spacing=2, key="mf-fields")
        self.selector = ft.Dropdown(
            options=[ft.DropdownOption(key=path, text=self.titles.get(path) or os.path.basename(path))
                     for path in self.epubs],
            value=self.current or None, on_select=self._on_select, visible=len(self.epubs) > 1, key="mf-epub")
        self.counter = hint_text("", key="mf-counter")
        content = ft.Column([
            ft.Text("Select Metadata Fields to Translate", theme_style=ft.TextThemeStyle.TITLE_LARGE,
                    weight=ft.FontWeight.W_600),
            hint_text("These fields will be translated along with or separately from the book title:"),
            ft.Row([self.selector, self.counter], wrap=True),
            self.fields_column,
            ft.Row([ft.TextButton(content="↺ Reset", on_click=self._on_reset, key="mf-reset"),
                    ft.TextButton(content="Cancel", on_click=lambda e: self.close(), key="mf-cancel"),
                    ft.FilledButton(content="💾 Save", on_click=self._on_save, key="mf-save")],
                   alignment=ft.MainAxisAlignment.END, wrap=True),
        ], tight=True, spacing=tokens.SPACING["sm"], scroll=ft.ScrollMode.AUTO)
        self.sheet = ft.BottomSheet(content=ft.Container(content=content, padding=tokens.SPACING["sheet_padding"]),
                                    show_drag_handle=True, scrollable=True, bgcolor=ft.Colors.SURFACE_CONTAINER_HIGH)
        self._page: Any = None
        self.saved: Optional[dict] = None

    def show(self, page: Any) -> "MetadataFieldsSheet":
        self._page = page
        page.show_dialog(self.sheet)
        self.ctx.spawn(self.load(self.current))
        return self

    def close(self) -> None:
        close_dialog(self._page, self.sheet)

    def _remember(self) -> None:
        if self.current and self.checkboxes:
            self.selections[self.current] = {name: bool(box.value) for name, box in self.checkboxes.items()}

    async def load(self, epub: str) -> dict:
        self._remember()
        self.current = epub
        if epub not in self.detected:
            try:
                self.detected[epub] = await self.ctx.io(hm.detect_fields, epub)
            except Exception as exc:
                log.exception("detecting metadata fields failed")
                self.detected[epub] = {}
                self.ctx.say(f"Error reading EPUB metadata: {exc}")
        self.render()
        return self.detected[epub]

    def render(self) -> None:
        epub = self.current
        detected = self.detected.get(epub, {})
        saved = self.selections.get(epub) or hm.saved_selection(self.previous, epub)
        checks = hm.initial_checks(detected, saved, self.sync)
        if epub in self.selections:
            checks.update(self.selections[epub])
        rows: list[ft.Control] = []
        self.checkboxes = {}
        if not detected:
            rows.append(hint_text("No metadata fields found in this EPUB.", key="mf-empty"))
        else:
            rows.append(ft.Text("Standard Metadata Fields:", weight=ft.FontWeight.W_600))
            for name, (label, _desc) in hm.STANDARD_FIELDS.items():
                if name in detected:
                    rows.append(self._row(name, f"{label}:", detected[name], checks.get(name, False)))
            custom = [n for n in detected if n not in hm.STANDARD_FIELDS]
            if custom:
                rows.append(ft.Text("Custom Metadata Fields:", weight=ft.FontWeight.W_600))
                rows.append(hint_text("(Non-standard fields found in your EPUB)"))
                for name in custom:
                    rows.append(self._row(name, f"{name}:", detected[name], checks.get(name, False)))
        self.fields_column.controls = rows
        if len(self.epubs) > 1 and epub in self.epubs:
            self.counter.value = f"{self.epubs.index(epub) + 1} / {len(self.epubs)}"
        self.ctx.push(self.fields_column, self.counter)

    def _row(self, name: str, label: str, value: Any, checked: bool) -> ft.Control:
        text = str(value)
        if len(text) > 50:
            text = text[:47] + "..."
        box = ft.Checkbox(label=label, value=bool(checked), on_change=lambda e, n=name: self._on_toggle(n),
                          key=f"mf-{name}")
        self.checkboxes[name] = box
        return ft.Row([box, ft.Text(text, theme_style=ft.TextThemeStyle.BODY_SMALL, max_lines=1, expand=True,
                                    overflow=ft.TextOverflow.ELLIPSIS)], spacing=6)

    def _on_toggle(self, name: str) -> None:
        box = self.checkboxes.get(name)
        if box is not None:
            self.sync[name] = bool(box.value)  # shared across the selected EPUBs (desktop field sync)

    def _on_select(self, e: Any = None) -> None:
        value = getattr(getattr(e, "control", None), "value", None) or self.selector.value
        if value and value != self.current:
            self.ctx.spawn(self.load(value))

    def reset(self) -> None:
        for name in list(self.sync):
            self.sync[name] = name in hm.DEFAULT_ENABLED_FIELDS
        for name, box in self.checkboxes.items():
            box.value = name in hm.DEFAULT_ENABLED_FIELDS
        self.ctx.push(*self.checkboxes.values())

    def _on_reset(self, e: Any = None) -> None:
        dialog = ConfirmDialog(
            title="Reset Metadata Fields",
            body="Are you sure you want to reset all metadata field selections to defaults?\n\n"
                 "Description and Subject will be enabled; all others will be unchecked.",
            confirm_label="Yes", cancel_label="No", on_confirm=self.reset)
        if self._page is not None:
            dialog.show(self._page)
        else:
            self.reset()

    def save(self) -> dict:
        self._remember()
        value = hm.fields_config(self.previous, self.selections, self.epubs)
        self.ctx.set_cfg("translate_metadata_fields", value)
        self.saved = value
        self.close()
        self.ctx.say("Metadata fields saved")
        return value

    def _on_save(self, e: Any = None) -> dict:
        return self.save()


class HeadersScreen(Screen):
    title = "Headers & metadata"

    def __init__(self, match: Optional[RouteMatch], ctx: Any) -> None:
        super().__init__(match)
        self.ctx = ctx
        self.state = ctx.tool_state.setdefault("headers", {})
        self.targets: list = list(self.state.get("targets") or [])
        self.status: dict = {}
        self.watch = JobWatch(ctx, self._on_job_end, self._on_job_change)
        self.active_job: Any = None
        self.picker: Optional[SourcePicker] = None
        self.last_plan: Optional[hm.ArtifactPlan] = None
        self.last_delete: Optional[tuple] = None
        self.mode_radio: Optional[ft.RadioGroup] = None

    # ---- layout --------------------------------------------------------------------------------

    def build_body(self) -> ft.Control:
        self.books_column = ft.Column(spacing=2, key="hd-books")
        books = card("Books", [self.books_column,
                               ft.Row([ft.FilledTonalButton(content="Choose…", icon=ft.Icons.LIBRARY_BOOKS,
                                                            on_click=lambda e: self.open_picker(), key="hd-choose"),
                                       ft.TextButton(content="Clear", on_click=lambda e: self.set_targets([]),
                                                     key="hd-clear")], wrap=True)],
                     icon="LIBRARY_BOOKS", key="hd-books-card")
        self.rebuild = ft.Switch(label="Rebuild the EPUB afterwards", value=bool(self.state.get("rebuild", True)),
                                 on_change=self._on_rebuild, key="hd-rebuild")
        self.header_actions = ft.Row(wrap=True, spacing=8, run_spacing=8, key="hd-header-actions")
        self.run_status = ft.Text("", theme_style=ft.TextThemeStyle.BODY_SMALL, key="hd-status")
        # UI_SPEC §7.4: a failed headers / metadata job as an ErrorCard (Retry · Copy error · View log)
        self.header_error = ft.Container(visible=False, key="hd-header-error")
        self.metadata_error = ft.Container(visible=False, key="hd-metadata-error")
        self.failed: dict = {}  # job kind -> the failed snapshot shown
        self.error_builds = 0
        self.stop_button = ft.OutlinedButton(content="⏹ Stop Headers", on_click=self._on_stop, visible=False,
                                             key="hd-stop")
        self.progress = ft.ProgressBar(visible=False, key="hd-progress")
        header_tiles, self.header_tiles = schema_tiles(self.ctx, HEADER_SETTING_KEYS)
        headers = card("Chapter headers & TOC", [
            self.header_actions, self.rebuild, ft.Row([self.stop_button]), self.progress, self.run_status,
            self.header_error,
            *header_tiles,
            ft.TextButton(content="All Meta Data settings", icon=ft.Icons.SETTINGS,
                          on_click=lambda e: self.open_settings("other.meta_data"), key="hd-meta-settings"),
        ], icon="TITLE", key="hd-headers",
            subtitle="Translates the chapter titles from the raw EPUB's spine and updates the chapter files.",
            trailing=ft.IconButton(icon=ft.Icons.INFO_OUTLINE, tooltip="Header translation help", key="hd-help",
                                   on_click=lambda e: self.open_help(), size_constraints=HIT_TARGET))
        meta_tiles, self.meta_tiles = schema_tiles(self.ctx, METADATA_SETTING_KEYS)
        mode = str(self.ctx.cfg("metadata_translation_mode", "together") or "together")
        self.mode_radio = ft.RadioGroup(
            value=mode if mode in [m[0] for m in hm.METADATA_MODES] else "together",
            on_change=self._on_mode,
            content=ft.Column([ft.Radio(value=value, label=label, tooltip=tip or None, key=f"hd-mode-{value}")
                               for value, label, tip in hm.METADATA_MODES], spacing=0, tight=True),
            key="hd-mode")
        self.metadata_actions = ft.Row(wrap=True, spacing=8, run_spacing=8, key="hd-metadata-actions")
        metadata = card("Metadata", [
            *meta_tiles,
            ft.Text("Translation Mode", weight=ft.FontWeight.W_600), self.mode_radio,
            ft.Row([ft.FilledTonalButton(content="Metadata fields…", icon=ft.Icons.CHECKLIST,
                                         on_click=lambda e: self.open_fields(), key="hd-fields"),
                    ft.FilledTonalButton(content="Configure All Prompts", icon=ft.Icons.EDIT_NOTE,
                                         on_click=lambda e: self.open_prompts(), key="hd-prompts")], wrap=True),
            self.metadata_actions,
            self.metadata_error,
        ], icon="LABEL", key="hd-metadata", subtitle="Book title, author, description… into metadata.json and the EPUB.")
        self._render_books()
        return ft.ListView(controls=[books, headers, metadata], expand=True, spacing=tokens.SPACING["md"],
                           padding=ft.Padding.symmetric(horizontal=12, vertical=8), key="hd-body")

    def _render_books(self) -> None:
        rows: list[ft.Control] = []
        for target in self.targets:
            status = self.status.get(target.key) or {}
            chips = []
            if status.get("headers"):
                chips.append("✓ translated_headers.txt")
            if status.get("toc"):
                chips.append("✓ TOC.txt")
            if status.get("metadata"):
                chips.append("✓ metadata.json")
            subtitle = f"📖 {target.source_name}" + (f" · 📁 {target.folder_name}" if target.folder else
                                                      " · no output folder yet")
            if chips:
                subtitle += "\n" + "  ".join(chips)
            rows.append(ft.ListTile(leading=ft.Icon(ft.Icons.MENU_BOOK), title=ft.Text(target.title, max_lines=2),
                                    subtitle=ft.Text(subtitle, theme_style=ft.TextThemeStyle.BODY_SMALL, max_lines=3),
                                    trailing=ft.IconButton(icon=ft.Icons.CLOSE, tooltip="Remove",
                                                           on_click=lambda e, t=target: self.remove_target(t), size_constraints=HIT_TARGET),
                                    dense=True, key=f"hd-book-{target.key}"))
        if not rows:
            rows.append(hint_text("No book chosen yet.", key="hd-no-books"))
        self.books_column.controls = rows
        self._render_actions()
        self.ctx.push(self.books_column)

    def _render_actions(self) -> None:
        targets = self.targets
        running = self.active_job is not None and not getattr(self.active_job, "is_terminal", True)
        none = "Choose a book first" if not targets else None
        with_folder = [t for t in targets if t.folder]
        jobs_reason = None if self.ctx.has_kind("translate_headers") else "The job service is not running"
        headers_reason = none or jobs_reason or (None if with_folder else "No output folder yet (translate first)")
        if running and headers_reason is None:
            headers_reason = "A header job is running"
        delete_reason = none or (None if with_folder else "No output folder yet")
        self.header_actions.controls = [
            action_button("Translate Headers Now", "TRANSLATE", lambda e: self.ctx.spawn(self.translate_headers()),
                          key="hd-translate", reason=headers_reason, filled=True),
            action_button("Delete Header Files", "DELETE_OUTLINE", lambda e: self.ctx.spawn(self.delete("headers")),
                          key="hd-delete-headers", reason=delete_reason, destructive=True),
            action_button("Delete TOC.txt", "DELETE_SWEEP", lambda e: self.ctx.spawn(self.delete("toc")),
                          key="hd-delete-toc", reason=delete_reason, destructive=True),
        ]
        meta_reason = none or (None if self.ctx.has_kind("metadata") else "The job service is not running")
        count = len(hm.epub_targets(targets))
        label = f"🌐 Translate Metadata for {count} EPUB{'s' if count != 1 else ''}" if count > 1 \
            else "🌐 Translate Metadata"
        self.metadata_actions.controls = [
            action_button(label, "LABEL", lambda e: self.ctx.spawn(self.translate_metadata()),
                          key="hd-translate-metadata", reason=meta_reason, filled=True),
        ]
        snap = self.active_job
        self.stop_button.visible = running
        self.progress.visible = running
        if running:
            from glossarion_mobile.services.jobs import progress_line

            self.run_status.value = f"{snap.state_label} · {progress_line(snap) or snap.phase}"
        elif snap is not None and not job_failed(snap):
            self.run_status.value = f"{snap.state_label}: {snap.title}" + (f" — {snap.error}" if snap.error else "")
        else:
            self.run_status.value = ""
        self._render_errors()
        self.ctx.push(self.header_actions, self.metadata_actions, self.stop_button, self.progress, self.run_status,
                      self.header_error, self.metadata_error)

    def _render_errors(self) -> None:
        """The ErrorCards of the last failed headers / metadata job (fresh keys per build)."""
        self.error_builds += 1
        for kind, slot in (("translate_headers", self.header_error), ("metadata", self.metadata_error)):
            snap = self.failed.get(kind)
            if snap is None:
                slot.content, slot.visible = None, False
                continue
            slot.content = failed_job_card(self.ctx, snap, key=f"hd-error-{kind}-{self.error_builds}",
                                           on_retry=lambda s=snap: self.ctx.spawn(self.retry(s)))
            slot.visible = True

    async def retry(self, snap: Any) -> Optional[str]:
        """ErrorCard › Retry: resubmit the failed job's spec."""
        spec = getattr(snap, "spec", None)
        if spec is None:
            return None
        self.failed.pop(getattr(snap, "kind", ""), None)
        job_id = await self.ctx.submit(spec)
        if not job_id:
            self._render_actions()
            return None
        self.watch.watch(job_id)
        snapshot = getattr(self.ctx.jobs, "snapshot", None)
        if getattr(snap, "kind", "") == "translate_headers" and callable(snapshot):
            self.active_job = snapshot(job_id)
        self._render_actions()
        return job_id

    # ---- lifecycle ------------------------------------------------------------------------------------

    def did_show(self) -> None:
        self.watch.start()
        self._adopt_jobs()
        out = self.match.get("out") if self.match is not None else None
        if out and not self.targets:
            self.ctx.spawn(self.preselect_book(out))
        elif self.targets:
            self.ctx.spawn(self.refresh_status())

    def _adopt_jobs(self) -> None:
        """Reopened while a header / metadata job of an earlier visit is queued or running: follow it (Stop,
        Translate Headers Now disabled, the cache status refreshed when it ends)."""
        adopted = self.watch.adopt(("translate_headers", "metadata"))
        headers = next((s for s in adopted if getattr(s, "kind", "") == "translate_headers"), None)
        if headers is not None and (self.active_job is None or getattr(self.active_job, "is_terminal", True)):
            self.active_job = headers
            if getattr(self, "header_actions", None) is not None:
                self._render_actions()

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
        if target is not None and headers_eligibility(target) is None:
            self.set_targets([target])
        return target

    async def refresh_status(self) -> dict:
        targets = list(self.targets)

        def gather() -> dict:
            out = {}
            for target in targets:
                status = hm.artifact_status(target.folder)
                status["metadata"] = bool(target.folder) and os.path.isfile(os.path.join(target.folder,
                                                                                         "metadata.json"))
                out[target.key] = status
            return out

        try:
            self.status = await self.ctx.io(gather)
        except Exception:
            log.exception("reading the cache status failed")
        self._render_books()
        return self.status

    # ---- sources ----------------------------------------------------------------------------------------

    def open_picker(self) -> SourcePicker:
        self.picker = SourcePicker(self.ctx, title="Choose books", multi=True, eligible=headers_eligibility,
                                   on_done=self.set_targets, selected=self.targets, segment="library",
                                   browse_label="Pick an EPUB…")
        if self.ctx.page is not None:
            self.picker.show(self.ctx.page)
        return self.picker

    def set_targets(self, targets: Any) -> None:
        self.targets = [t for t in targets or () if headers_eligibility(t) is None]
        self.state["targets"] = list(self.targets)
        self._render_books()
        if self.targets:
            self.ctx.spawn(self.refresh_status())

    def remove_target(self, target: tg.ToolTarget) -> None:
        self.set_targets([t for t in self.targets if t.key != target.key])

    # ---- headers ---------------------------------------------------------------------------------------

    def _on_rebuild(self, e: Any = None) -> None:
        self.state["rebuild"] = bool(self.rebuild.value)

    async def translate_headers(self) -> Optional[str]:
        targets = [t for t in self.targets if t.folder]
        if not targets:
            self.ctx.say("No output directory found for the selected books")
            return None
        try:
            spec = hm.headers_spec(targets, rebuild_epub=bool(self.rebuild.value))
        except ValueError as exc:
            self.ctx.say(str(exc))
            return None
        job_id = await self.ctx.submit(spec)
        if not job_id:
            return None
        self.watch.watch(job_id)
        snapshot = getattr(self.ctx.jobs, "snapshot", None)
        self.active_job = snapshot(job_id) if callable(snapshot) else None
        self.ctx.remember_source("tools.headers", spec.title)
        self.ctx.say("🌐 Starting standalone header translation in background...", "Jobs",
                     lambda: self.ctx.go("jobs.detail", {"jid": job_id}))
        self._render_actions()
        return job_id

    def _on_stop(self, e: Any = None) -> Any:
        snap = self.active_job
        if snap is None or self.ctx.jobs is None:
            return None
        try:
            return self.ctx.jobs.request_stop(snap.id)
        except Exception as exc:
            self.ctx.say(f"Could not stop: {exc}")
            return None

    async def delete(self, kind: str) -> Optional[tuple]:
        """Delete Header Files / Delete TOC.txt with the desktop confirmation flow."""
        targets = list(self.targets)
        if not targets:
            self.ctx.say("No EPUB file(s) selected. Please select EPUB file(s) first.")
            return None
        plan = await self.ctx.io(hm.plan_artifact_delete, targets, kind)
        self.last_plan = plan
        if not plan.found and not plan.not_found and not plan.errors:
            self.ctx.say("No EPUB files were processed.")
            return None
        if not plan.found:
            await ask(self.ctx, "No Files to Delete", plan.summary_text(), [("ok", "OK", "filled")],
                      key="hd-nothing")
            return None
        texts = hm.ARTIFACT_TEXTS[kind]
        if plan.has_linked:
            answer = await ask(self.ctx, "Confirm Deletion", plan.question_text(),
                               [("cancel", "Cancel", "text"), ("only", texts["only"], "text"),
                                ("both", texts["both"], "destructive")], key="hd-confirm")
        else:
            answer = await ask(self.ctx, "Confirm Deletion", plan.question_text(),
                               [("no", "No", "text"), ("yes", "Yes", "destructive")], key="hd-confirm")
        if answer not in ("yes", "only", "both"):
            return None
        message, ok = await self.ctx.io(lambda: hm.execute_artifact_delete(plan, delete_linked=answer == "both"))
        self.last_delete = (message, ok)
        await ask(self.ctx, "Success" if ok else "Error", message, [("ok", "OK", "filled")], key="hd-result")
        await self.refresh_status()
        return message, ok

    # ---- metadata ------------------------------------------------------------------------------------------

    def _on_mode(self, e: Any = None) -> None:
        value = getattr(getattr(e, "control", None), "value", None) or (self.mode_radio.value if self.mode_radio
                                                                         else None)
        if value in [m[0] for m in hm.METADATA_MODES]:
            self.ctx.set_cfg("metadata_translation_mode", value)

    def open_fields(self) -> Optional[MetadataFieldsSheet]:
        epubs = hm.epub_targets(self.targets)
        if not epubs:
            self.ctx.say("Choose a book with a raw EPUB first")
            return None
        sheet = MetadataFieldsSheet(self.ctx, epubs)
        if self.ctx.page is not None:
            sheet.show(self.ctx.page)
        return sheet

    def open_prompts(self) -> PromptsSheet:
        sheet = PromptsSheet(self.ctx)
        if self.ctx.page is not None:
            sheet.show(self.ctx.page)
        return sheet

    async def translate_metadata(self) -> Optional[str]:
        epubs = hm.epub_targets(self.targets)
        if not epubs:
            self.ctx.say("Could not resolve an original EPUB source for the selection.")
            return None
        warning = await self.ctx.io(hm.existing_metadata_warning, [t.folder for t in epubs])
        if warning:
            answer = await ask(self.ctx, "Metadata Already Exists", warning,
                               [("cancel", "Cancel", "text"), ("yes", "Yes", "filled")], key="hd-meta-exists")
            if answer != "yes":
                return None
        try:
            spec = hm.metadata_spec(epubs)
        except ValueError as exc:
            self.ctx.say(str(exc))
            return None
        job_id = await self.ctx.submit(spec)
        if not job_id:
            return None
        self.watch.watch(job_id)
        self.ctx.remember_source("tools.headers", spec.title)
        self.ctx.say(f"🌐 Translating metadata · {spec.title}", "Jobs",
                     lambda: self.ctx.go("jobs.detail", {"jid": job_id}))
        return job_id

    def open_settings(self, section: str) -> None:
        settings = self.ctx.settings
        if settings is not None and hasattr(settings, "open_setting"):
            settings.open_setting(section)
        else:
            self.ctx.go("settings.section", {"section": section})

    def open_help(self) -> Any:
        """ⓘ on "Chapter headers & TOC": the desktop Header Translation help guide (InfoSheet)."""
        from glossarion_mobile.ui.components.info_sheet import InfoSheet

        title, body = header_help_markdown()
        sheet = InfoSheet(title=title, body=body, markdown=True)
        if self.ctx.page is not None:
            sheet.show(self.ctx.page)
        return sheet

    # ---- jobs ----------------------------------------------------------------------------------------------

    def _on_job_change(self, snap: Any) -> None:
        if getattr(snap, "kind", "") == "translate_headers":
            self.active_job = snap
            self._render_actions()

    def _on_job_end(self, snap: Any) -> None:
        kind = getattr(snap, "kind", "")
        if kind in ("translate_headers", "metadata"):
            if job_failed(snap):
                self.failed[kind] = snap
            else:
                self.failed.pop(kind, None)
        if kind == "translate_headers":
            self.active_job = snap
            self._render_actions()
        elif kind == "metadata":
            self._render_actions()
        self.ctx.spawn(self.refresh_status())
