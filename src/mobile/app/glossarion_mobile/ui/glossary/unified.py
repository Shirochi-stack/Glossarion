"""Unified glossary (``/glossary/unified``; UI_SPEC §4.1; desktop "Unified Glossary Settings" dialog).

* the description and the switches Enable Unified Glossary · Generate Unified Glossary ·
  Combine all languages · Exclude gendered active entries (schema tiles, auto-saved);
* Source language (Auto · Korean · … · Other; stored lower-case like the desktop combo);
* "Location: Glossary/Unified Glossary/<key>/glossary_unified.csv";
* **🔄 Rebuild Now** - a ``unified_glossary`` job (``unified_glossary.rebuild_now`` with the
  desktop settings snapshot), disabled while it is queued or runs (a ``JobWatch`` on the job,
  also on a rebuild a previous visit of the screen started);
  while another run is active the desktop warning is shown and the rebuild waits in the queue;
  the unified files with Open.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import flet as ft

from glossarion_mobile.ui.router import RouteMatch
from glossarion_mobile.ui.screens.base import Screen
from glossarion_mobile.ui.tools.common import JobWatch

__all__ = ["DESCRIPTION", "SOURCE_LANGUAGES", "UnifiedGlossaryScreen"]

log = logging.getLogger("glossarion.glossary.ui")

DESCRIPTION = ("One deduplicated glossary shared by every novel, kept per source and target language. It is updated "
               "at the start and end of each glossary run and sent alongside the main glossary with the same "
               "compression settings.")
#: GlossaryManager ``_UNIFIED_SOURCE_LANGUAGE_OPTIONS`` (the shared constant wins when a core exposes it).
SOURCE_LANGUAGES = ("Auto", "Korean", "Japanese", "Chinese", "English", "Spanish", "French", "German", "Italian",
                    "Portuguese", "Russian", "Arabic", "Hindi", "Turkish", "Hebrew", "Thai", "Other")
SWITCH_KEYS = ("enable_unified_glossary", "generate_unified_glossary", "unified_glossary_combine_all_languages",
               "unified_glossary_exclude_gender_entries")
REBUILD_HINT = "(Rebuilds from every book glossary now; progress appears in the job log)"
BUSY_WARNING = "⚠️ Unified glossary: wait for the current translation/glossary run to finish before rebuilding."


class UnifiedGlossaryScreen(Screen):
    title = "Unified glossary"

    def __init__(self, match: Optional[RouteMatch], ctx: Any) -> None:
        super().__init__(match)
        self.ctx = ctx
        self.service = ctx.service
        self.page_view: Any = None
        self.job_id: Optional[str] = None
        self._unsubs: list = []
        self.watch = JobWatch(ctx, self._on_job_end)
        self.reason: Optional[str] = None

    def languages(self) -> tuple:
        shared = self.service.core.value("glossary_document", "UNIFIED_SOURCE_LANGUAGE_OPTIONS", default=None)
        return tuple(shared) if shared else SOURCE_LANGUAGES

    def build_body(self) -> ft.Control:
        controls: list = [ft.Text(DESCRIPTION, theme_style=ft.TextThemeStyle.BODY_SMALL,
                                  color=ft.Colors.ON_SURFACE_VARIANT, key="ug-desc")]
        settings = self.ctx.settings
        if settings is not None and getattr(settings.schema, "available", False):
            from glossarion_mobile.ui.settings.tiles import EffectiveConfig, make_tile

            config_view = EffectiveConfig(settings.store)
            self.tiles = {}
            for key in SWITCH_KEYS:
                spec = settings.schema.spec(key)
                if spec is None:
                    continue
                tile = make_tile(spec, settings, config=config_view)
                tile.refresh(push=False)
                self.tiles[key] = tile
                controls.append(tile.control)
        saved = str(self.service.cfg("unified_glossary_source_language", "auto") or "auto").strip().lower()
        options = self.languages()
        value = next((o for o in options if o.lower() == saved), options[0])
        self.language = ft.Dropdown(label="Source language", value=value, key="ug-language",
                                    options=[ft.DropdownOption(key=o, text=o) for o in options],
                                    on_select=lambda e: self._on_language())
        controls.append(self.language)
        controls.append(ft.Text("(Auto reuses the source language detected during extraction)",
                                theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT))
        self.location = ft.Text(self.service.unified_location(), theme_style=ft.TextThemeStyle.BODY_SMALL,
                                selectable=True, key="ug-location")
        controls.append(self.location)
        reason = None if self.service.has_job_kind("unified_glossary") else "The job service is not running"
        self.reason = reason
        self.rebuild_button = ft.FilledButton(content="🔄 Rebuild Now", on_click=lambda e: self.ctx.spawn(self.rebuild()),
                                              disabled=reason is not None, tooltip=reason, key="ug-rebuild")
        self.status = ft.Text(REBUILD_HINT, theme_style=ft.TextThemeStyle.BODY_SMALL, key="ug-status")
        controls.append(ft.Row([self.rebuild_button], wrap=True))
        controls.append(self.status)
        self.files_column = ft.Column(spacing=0, tight=True, key="ug-files")
        controls.append(ft.Text("Unified glossaries", theme_style=ft.TextThemeStyle.TITLE_SMALL,
                                color=ft.Colors.PRIMARY))
        controls.append(self.files_column)
        self.root = ft.ListView(controls=controls, spacing=8, padding=ft.Padding.symmetric(horizontal=16, vertical=12),
                                expand=True, key="unified")
        return self.root

    def did_show(self) -> None:
        self.watch.start()
        # Reopened while a rebuild is queued or running (started from an earlier visit): follow it, so the
        # button stays disabled and a second tap queues nothing.
        adopted = self.watch.adopt(("unified_glossary",))
        if adopted and (self.job_id is None or self.job_id in self.watch.ended):
            self.job_id = adopted[0].id
            queued = str(getattr(adopted[0].state, "value", adopted[0].state)) == "QUEUED"
            self.status.value = ("(Queued — the rebuild starts when the current run finishes)" if queued else
                                 "(Running in the background — progress appears in the job log)")
            self.ctx.push(self.status)
        self._sync_rebuild()
        store = getattr(self.ctx.settings, "store", None)
        if store is not None and not self._unsubs:
            try:
                self._unsubs.append(store.observe_all(lambda key, value: self.ctx.post_ui(self._refresh_tiles)))
            except Exception:
                pass
        self.ctx.spawn(self.load_files())

    def dispose(self) -> None:
        self.watch.stop()
        for unsub in self._unsubs:
            try:
                unsub()
            except Exception:
                pass
        self._unsubs = []

    def _refresh_tiles(self) -> None:
        for tile in getattr(self, "tiles", {}).values():
            tile.refresh(push=False)
        self.location.value = self.service.unified_location()
        self.ctx.push(self.root)

    def _on_language(self) -> str:
        language = str(self.language.value or "Auto").strip().lower() or "auto"
        self.service.set_cfg("unified_glossary_source_language", language)
        self.location.value = self.service.unified_location()
        self.ctx.push(self.location)
        return language

    async def load_files(self) -> list:
        try:
            rows = await self.ctx.io(self.service.list_glossaries)
        except Exception:
            rows = []
        unified = [r for r in rows if r.kind == "unified"]
        self.files_column.controls = [
            ft.ListTile(leading=ft.Icon(ft.Icons.MERGE_TYPE), title=ft.Text(r.language_key or r.name),
                        subtitle=ft.Text(r.name), on_click=lambda e, r=r: self.ctx.go("glossary.detail", {"gid": r.gid}),
                        key=f"ug-file-{r.gid}") for r in unified
        ] or [ft.Text("No unified glossary yet — it is generated during glossary runs, or with Rebuild Now.",
                      theme_style=ft.TextThemeStyle.BODY_SMALL, key="ug-none")]
        self.ctx.push(self.files_column)
        return unified

    def rebuild_active(self) -> bool:
        """The rebuild this screen started is still queued or running."""
        return self.watch.active() is not None

    def _sync_rebuild(self) -> None:
        running = self.rebuild_active()
        self.rebuild_button.disabled = self.reason is not None or running
        self.rebuild_button.tooltip = self.reason or ("Rebuilding…" if running else None)
        self.ctx.push(self.rebuild_button)

    def _on_job_end(self, snap: Any) -> None:
        state = str(getattr(getattr(snap, "state", None), "value", getattr(snap, "state", "")) or "")
        error = getattr(snap, "error", None)
        if state == "DONE":
            self.status.value = "(Rebuild finished — the unified glossaries are updated)"
        else:
            self.status.value = f"(Rebuild ended: {state.lower() or 'stopped'}{f' — {error}' if error else ''})"
        self._sync_rebuild()
        self.ctx.push(self.status)
        self.ctx.spawn(self.load_files())

    async def rebuild(self) -> Optional[str]:
        if self.rebuild_active():
            self.ctx.say("The unified glossary rebuild is already running")
            return self.job_id
        jobs = self.ctx.jobs
        try:
            running = bool(getattr(jobs, "busy", False)) if jobs is not None else False  # JobService.busy (property)
        except Exception:
            running = False
        if running:
            self.ctx.say(BUSY_WARNING)
        spec = self.service.unified_spec()
        try:
            self.job_id = await self.service.submit(spec)
        except Exception as exc:
            self.ctx.say(f"Could not start the rebuild: {exc}")
            return None
        self.watch.watch(self.job_id)
        self._sync_rebuild()
        self.status.value = ("(Queued — the rebuild starts when the current run finishes)" if running else
                             "(Running in the background — progress appears in the job log)")
        self.ctx.push(self.status)
        self.ctx.say("📚 Unified glossary: Rebuild Now started", "Jobs", lambda: self.ctx.go("jobs"))
        return self.job_id
