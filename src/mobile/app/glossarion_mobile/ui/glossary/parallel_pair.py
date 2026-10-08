"""Parallel EPUB pair (``/glossary/parallel-pair``; UI_SPEC §4.1; desktop ``ParallelEpubPairDialog``).

Steps on one full-screen view:

1. **EPUBs** - pick the raw and the translated EPUB (FileBridge picker, or a Library book).
   Both are read on the io pool (``extract_chapters_from_epub`` with special files kept; the
   dialog's loader drops text-less documents and keeps the reading order). Reopening a
   pair restores its saved mapping (the sidecar beside the raw book's glossary, else the
   ``parallel_epub_pair_selection`` config value).
2. **Mapping** - ``parallel_epub_core.auto_map_epub_chapters`` (Auto-offset switch
   ``parallel_epub_auto_offset_enabled``; Re-map runs it again), an offset −/+ stepper
   (overflow and special files become unmapped), the status line ("N mapped • offset +1 •
   2 raw unmatched …"), tap a row to pick its translated file, long-press to select rows
   and **Set unmapped**.
3. **Prompts** - the system-prompt profile (``parallel_epub_glossary_profiles``, the built-in
   "Parallel EPUB Glossary" can be reset, others deleted; New asks a name) and the pair
   wrapper prompt with its placeholders ``{raw_text}`` ``{translated_text}``
   ``{raw_filename}`` ``{translated_filename}``.
4. **Accept** - the dialog's checks and texts ("EPUBs Required", "Two EPUBs Required",
   "Wrapper Placeholders Required", "System Prompt Required", "Mapping Required",
   "Duplicate Mapping", the "Unmapped HTML Files" question), then a ``parallel_pair`` job
   (glossary only; the chapter-text-free selection travels in the job spec).

``PairMapping`` holds the table state the desktop dialog keeps in its QTableWidget
(assignments, Match texts, offset); every data step is the dialog's own shared function in
``parallel_epub_core`` (``_apply_mapping_offset``, ``_set_rows_unmapped``, ``_selected_mapping``,
``_unpaired_file_counts``, ``_update_mapping_status``, ``_apply_pending_persisted_mapping``,
the Accept checks, the prompt profiles and their persisted settings).
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Callable, Optional, Sequence

import flet as ft

from glossarion_mobile.services.glossary import CoreMissing
from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.components.action_sheet import ActionItem, ActionSheet
from glossarion_mobile.ui.components.windowed_list import WindowedList
from glossarion_mobile.ui.glossary.common import ask, prompt_text
from glossarion_mobile.ui.router import RouteMatch
from glossarion_mobile.ui.screens.base import Screen
from glossarion_mobile.ui.theme import HIT_TARGET

__all__ = ["DEFAULT_PROFILE", "PairMapping", "ParallelPairScreen", "UNMAPPED"]

log = logging.getLogger("glossarion.glossary.ui")

UNMAPPED = "— Unmapped —"  # parallel_epub_core.translated_mapping_label(-1)
DEFAULT_PROFILE = "Parallel EPUB Glossary"  # parallel_epub_core.DEFAULT_PARALLEL_EPUB_PROFILE
PLACEHOLDERS = ("{raw_text}", "{translated_text}", "{raw_filename}", "{translated_filename}")


class PairMapping:
    """The mapping table state (one row per raw document) as the desktop dialog keeps it in its
    QTableWidget: each row's translated index (``Qt.UserRole``) and Match text, and the running
    ``_mapping_offset``. Only that storage is local; every step is the dialog's shared function in
    ``parallel_epub_core`` (``core``): ``offset_parallel_epub_mapping`` (``_apply_mapping_offset``),
    ``valid_parallel_epub_rows`` (``_set_rows_unmapped``), ``persisted_parallel_epub_rows``
    (``_apply_pending_persisted_mapping``), ``selected_parallel_epub_mapping`` (``_selected_mapping``),
    ``unpaired_file_counts`` / ``unpaired_warning_text``, ``parallel_epub_mapping_status``
    (``_update_mapping_status``), ``translated_mapping_label`` and ``build_parallel_epub_pairs``.
    """

    def __init__(self, raw: Sequence[dict], translated: Sequence[dict], auto: Sequence[dict], *, core: Any,
                 special_file_predicate: Optional[Callable[[str], bool]] = None, protect_interior: bool = True,
                 reading_order: Optional[Sequence[str]] = None) -> None:
        self.core = core
        self.raw = list(raw)
        self.translated = list(translated)
        self.auto = [dict(m) for m in auto]
        self.special_file_predicate = special_file_predicate
        self.protect_interior = bool(protect_interior)
        self.reading_order = list(reading_order) if reading_order is not None else None
        self.offset = 0
        self.assign: list = []
        self.strategy: list = []
        self.reset()

    def reset(self) -> None:
        """``_rebuild_mapping`` / ``_populate_mapping_rows``: the automatic assignment and Match text."""
        self.offset = 0
        self.assign = []
        self.strategy = []
        for row in range(len(self.raw)):
            automatic = self.auto[row] if row < len(self.auto) else {}
            index = automatic.get("translated_index")
            self.assign.append(-1 if index is None else int(index))
            self.strategy.append(str(automatic.get("strategy")) if row < len(self.auto) else "Unmatched")

    def label(self, translated_index: int) -> str:
        return self.core.translated_mapping_label(self.translated, translated_index)

    def apply_offset(self, delta: int) -> None:
        """``_apply_mapping_offset``: the automatic indexes shifted by the running offset."""
        if not self.auto or not self.translated:
            return
        self.offset += int(delta)
        rows = self.core.offset_parallel_epub_mapping(self.auto, self.offset, self.translated,
                                                      self.special_file_predicate,
                                                      protect_interior=self.protect_interior,
                                                      reading_order=self.reading_order)
        for row, (translated_index, strategy) in enumerate(rows):
            if row < len(self.assign):
                self.assign[row] = translated_index
                self.strategy[row] = strategy

    def set_row(self, row: int, translated_index: int) -> None:
        """A manual choice (``_mapping_changed``: the Match column reads "Manual")."""
        self.assign[row] = int(translated_index)
        self.strategy[row] = "Manual"

    def set_unmapped(self, rows: Sequence[int]) -> int:
        """``_set_rows_unmapped``."""
        valid = self.core.valid_parallel_epub_rows(rows, len(self.assign))
        for row in valid:
            self.assign[row] = -1
            self.strategy[row] = "Manual — Unmapped"
        return len(valid)

    def restore(self, restored_pairs: Sequence[dict]) -> None:
        """``_apply_pending_persisted_mapping``: a saved mapping over every row."""
        rows = self.core.persisted_parallel_epub_rows(len(self.assign), restored_pairs)
        for row, (translated_index, strategy) in enumerate(rows):
            self.assign[row] = translated_index
            self.strategy[row] = strategy
        self.offset = 0

    def selected(self) -> list:
        return self.core.selected_parallel_epub_mapping(self.assign)

    def unpaired_counts(self, mapping: Optional[list] = None) -> tuple:
        mapping = self.selected() if mapping is None else mapping
        return self.core.unpaired_file_counts(mapping, len(self.raw), len(self.translated))

    def status(self) -> tuple:
        """(status text, duplicate assignment count) of the mapping status line."""
        return self.core.parallel_epub_mapping_status(self.selected(), self.offset, self.auto, len(self.raw),
                                                      len(self.translated))

    def status_text(self) -> str:
        return self.status()[0]

    def duplicate_count(self) -> int:
        return int(self.status()[1])

    def unpaired_warning(self) -> str:
        return self.core.unpaired_warning_text(self.selected(), len(self.raw), len(self.translated))

    def pairs(self) -> list:
        """The accepted pairs (``build_parallel_epub_pairs``)."""
        return self.core.build_parallel_epub_pairs(self.selected(), self.raw, self.translated)


class ParallelPairScreen(Screen):
    title = "Parallel EPUB pair"

    def __init__(self, match: Optional[RouteMatch], ctx: Any) -> None:
        super().__init__(match)
        self.ctx = ctx
        self.service = ctx.service
        self.raw_path = ""
        self.translated_path = ""
        self.raw: dict = {}
        self.translated: dict = {}
        self.mapping: Optional[PairMapping] = None
        self.selected_rows: set = set()
        self.selecting = False
        self.loading = ""
        self.job_id: Optional[str] = None
        # The profiles load in did_show on the io pool: the built-in default prompt imports the
        # glossary extractor (and its API clients), seconds on a cold start.
        self.profiles: dict = {}
        self.profile = ""
        self.profiles_ready = False
        self._profiles_task: Any = None
        self._pending_selection: Optional[dict] = None

    # ---- body -------------------------------------------------------------------------------------------

    def build_body(self) -> ft.Control:
        self.raw_card = self._epub_card("raw", "Raw EPUB (source language)")
        self.translated_card = self._epub_card("translated", "Translated EPUB")
        self.auto_offset = ft.Switch(label="Auto-offset", value=bool(self.service.cfg(
            "parallel_epub_auto_offset_enabled", True)), on_change=lambda e: self._on_auto_offset(), key="pp-auto-offset")
        self.offset_text = ft.Text("offset +0", key="pp-offset")
        self.status = ft.Text("Load both EPUBs to create a map.", theme_style=ft.TextThemeStyle.BODY_SMALL,
                              color=ft.Colors.ON_SURFACE_VARIANT, key="pp-status")
        self.unmap_button = ft.TextButton(content="Set unmapped", icon=ft.Icons.LINK_OFF, visible=False,
                                          on_click=lambda e: self.set_selected_unmapped(), key="pp-unmap")
        self.rows = WindowedList(build_row=self._row_control, key_of=lambda row: f"r{row}", key="pp-rows",
                                 padding=ft.Padding.symmetric(horizontal=4))
        self.rows_holder = ft.Container(content=self.rows.control, height=420, key="pp-rows-holder")
        self.profile_dropdown = ft.Dropdown(label="System prompt profile", value=self.profile or None, dense=True,
                                            expand=True, disabled=not self.profiles_ready,
                                            hint_text=None if self.profiles_ready else "Loading profiles…",
                                            on_select=lambda e: self.load_profile(self.profile_dropdown.value),
                                            key="pp-profile")
        self.system_prompt = ft.TextField(label="System prompt", multiline=True, min_lines=4, max_lines=10,
                                          key="pp-system")
        self.delete_button = ft.IconButton(icon=ft.Icons.DELETE_OUTLINE, tooltip="Delete Profile",
                                           size_constraints=HIT_TARGET, on_click=lambda e: self.ctx.spawn(
                                               self.delete_or_reset_profile()), key="pp-delete")
        wrapper = str(self.service.cfg("parallel_epub_glossary_wrapper_prompt", "") or self.service.default_wrapper_prompt())
        self.wrapper = ft.TextField(label="Pair wrapper prompt", value=wrapper, multiline=True, min_lines=4,
                                    max_lines=10, key="pp-wrapper")
        self.placeholder_chips = ft.Row([ft.Chip(label=ft.Text(p), on_click=lambda e, p=p: self._insert(p))
                                         for p in PLACEHOLDERS], wrap=True, spacing=4)
        reason = None if self.service.has_job_kind("parallel_pair") else "The job service is not running"
        self.accept_button = ft.FilledButton(content="Use mapped pair · Extract glossary", icon=ft.Icons.CHECK,
                                             disabled=reason is not None, tooltip=reason,
                                             on_click=lambda e: self.ctx.spawn(self.accept()), key="pp-accept")
        if self.profiles_ready:
            self._render_profiles()
        self.root = ft.ListView(controls=[
            ft.Text("1. EPUBs", theme_style=ft.TextThemeStyle.TITLE_SMALL, color=ft.Colors.PRIMARY),
            self.raw_card, self.translated_card,
            ft.Text("2. Mapping", theme_style=ft.TextThemeStyle.TITLE_SMALL, color=ft.Colors.PRIMARY),
            ft.Row([self.auto_offset,
                    ft.TextButton(content="Re-map", icon=ft.Icons.AUTORENEW, on_click=lambda e: self.remap(),
                                  key="pp-remap")], wrap=True),
            ft.Row([ft.IconButton(icon=ft.Icons.REMOVE, tooltip="Offset −1", size_constraints=HIT_TARGET,
                                  on_click=lambda e: self.apply_offset(-1), key="pp-minus"),
                    self.offset_text,
                    ft.IconButton(icon=ft.Icons.ADD, tooltip="Offset +1", size_constraints=HIT_TARGET,
                                  on_click=lambda e: self.apply_offset(1), key="pp-plus"),
                    self.unmap_button], wrap=True),
            self.status,
            self.rows_holder,
            ft.Text("3. Prompts", theme_style=ft.TextThemeStyle.TITLE_SMALL, color=ft.Colors.PRIMARY),
            ft.Row([self.profile_dropdown,
                    ft.IconButton(icon=ft.Icons.ADD, tooltip="New Profile", size_constraints=HIT_TARGET,
                                  on_click=lambda e: self.ctx.spawn(self.new_profile()), key="pp-new"),
                    ft.IconButton(icon=ft.Icons.SAVE_OUTLINED, tooltip="Save Profile", size_constraints=HIT_TARGET,
                                  on_click=lambda e: self.save_profile(), key="pp-save"),
                    self.delete_button], spacing=0),
            self.system_prompt,
            ft.Text("Available placeholders: {raw_text}, {translated_text}, {raw_filename}, {translated_filename}",
                    theme_style=ft.TextThemeStyle.BODY_SMALL),
            self.placeholder_chips,
            self.wrapper,
            ft.Text("4. Accept", theme_style=ft.TextThemeStyle.TITLE_SMALL, color=ft.Colors.PRIMARY),
            ft.Row([self.accept_button,
                    ft.TextButton(content="Glossary progress", icon=ft.Icons.PLAYLIST_ADD_CHECK, key="pp-progress",
                                  on_click=lambda e: self.open_progress())], wrap=True),
        ], spacing=8, padding=ft.Padding.symmetric(horizontal=12, vertical=8), expand=True, key="parallel-pair")
        return self.root

    def _core(self) -> Any:
        core = self.service.core.module("parallel_epub_core")
        if core is None:
            raise CoreMissing("parallel_epub_core")
        return core

    def _epub_card(self, side: str, label: str) -> ft.Container:
        name = ft.Text("No EPUB selected", key=f"pp-{side}-name", max_lines=2, overflow=ft.TextOverflow.ELLIPSIS)
        info = ft.Text("", theme_style=ft.TextThemeStyle.BODY_SMALL, key=f"pp-{side}-info")
        card = ft.Container(content=ft.Column([
            ft.Text(label, theme_style=ft.TextThemeStyle.LABEL_LARGE),
            name, info,
            ft.Row([ft.TextButton(content="Choose EPUB…", icon=ft.Icons.FOLDER_OPEN,
                                  on_click=lambda e: self.ctx.spawn(self.pick(side)), key=f"pp-{side}-pick"),
                    ft.TextButton(content="From Library…", icon=ft.Icons.LOCAL_LIBRARY,
                                  on_click=lambda e: self.ctx.spawn(self.pick_from_library(side)),
                                  key=f"pp-{side}-library")],
                   wrap=True),
        ], spacing=2, tight=True), bgcolor=ft.Colors.SURFACE_CONTAINER_LOW, border_radius=tokens.RADII["card"],
            padding=10, key=f"pp-{side}")
        card.data = {"name": name, "info": info}
        return card

    def did_show(self) -> None:
        self.ctx.spawn(self._startup())

    async def _startup(self) -> None:
        await self.load_profiles()
        if self.raw_path or self._pending_selection is not None:
            return
        last_raw = str(self.service.cfg("parallel_epub_glossary_last_raw_epub", "") or "")
        saved = self.service.pair_saved_selection(last_raw) if last_raw else None
        if saved is None:
            key = self.service.core.value("parallel_epub_core", "PARALLEL_EPUB_SELECTION_CONFIG_KEY",
                                          default="parallel_epub_pair_selection")
            candidate = self.service.cfg(key, None)
            saved = dict(candidate) if isinstance(candidate, dict) and candidate.get("mapping") else None
        if saved is not None:
            await self.restore_selection(saved)

    async def load_profiles(self) -> dict:
        """(profiles, active profile) the pair dialog opens with (``parallel_epub_profiles``), off the UI
        loop; the dropdown stays disabled until then. Concurrent callers share one load."""
        if self.profiles_ready:
            return self.profiles
        task = self._profiles_task
        if task is None:
            task = self._profiles_task = asyncio.ensure_future(self._read_profiles())
        await asyncio.shield(task)
        return self.profiles

    async def _read_profiles(self) -> None:
        service = self.service

        def saved_profiles() -> tuple:
            saved = service.cfg("parallel_epub_glossary_profiles", {})
            return (dict(saved) if isinstance(saved, dict) else {},
                    str(service.cfg("parallel_epub_glossary_active_profile", "") or DEFAULT_PROFILE))

        def read() -> tuple:
            try:
                profiles, profile = service.pair_profiles()
            except CoreMissing:
                profiles, profile = saved_profiles()
            profiles = dict(profiles)
            if DEFAULT_PROFILE not in profiles:
                try:
                    profiles[DEFAULT_PROFILE] = service.default_pair_system_prompt()
                except Exception as exc:
                    log.warning("the built-in Parallel EPUB prompt is unavailable: %s", exc)
                    profiles[DEFAULT_PROFILE] = ""
            return profiles, profile

        try:
            profiles, profile = await self.ctx.io(read)
        except Exception as exc:  # keep the saved profiles: Accept / Save persist this dict
            log.warning("loading the Parallel EPUB profiles failed: %s", exc)
            profiles, profile = saved_profiles()
            profiles.setdefault(DEFAULT_PROFILE, "")
        self.profiles, self.profile = profiles, profile
        self.profiles_ready = True
        if getattr(self, "profile_dropdown", None) is not None:
            self.profile_dropdown.disabled = False
            self.profile_dropdown.hint_text = None
            self._render_profiles()
            self.ctx.push(self.profile_dropdown, self.system_prompt, self.delete_button)

    # ---- loading ------------------------------------------------------------------------------------------

    async def pick(self, side: str) -> Optional[str]:
        files = self.ctx.files
        if files is None:
            self.ctx.say("Picking files is not available in this session")
            return None
        try:
            picked = await files.pick_files(target="inbox", allowed_extensions=["epub"], allow_multiple=False,
                                            dialog_title="Choose the raw EPUB" if side == "raw" else
                                            "Choose the translated EPUB")
        except Exception as exc:
            self.ctx.say(f"Could not pick the file: {exc}")
            return None
        if not picked:
            return None
        await self.load(side, picked[0].path)
        return picked[0].path

    def library_epubs(self, side: str) -> list:
        """Blocking (io pool): ``(book name, EPUB path)`` of the Library books with an EPUB for ``side`` - the raw
        source, or for the translated side the workspace's compiled EPUB (``compiled_outputs_blocking``), else
        the book's own EPUB when it sits on the Completed shelf."""
        library = self.ctx.library
        snapshot = getattr(library, "snapshot", None) if library is not None else None
        books = list(snapshot.all_books()) if snapshot is not None else []
        compiled = getattr(library, "compiled_outputs_blocking", None)
        found = []
        for book in books:
            own = str(book.get("path") or "")
            path = str(book.get("raw_source_path") or "") if side == "raw" else ""
            if side == "translated":
                own_key = os.path.normcase(os.path.abspath(own)) if own else ""
                for candidate, kind in (compiled(book) if callable(compiled) else []):
                    # compiled_outputs_blocking also lists the Library file itself (a raw EPUB on the shelf)
                    if str(kind).lower() == "epub" and os.path.normcase(os.path.abspath(candidate)) != own_key:
                        path = candidate
                        break
                if not path and own.lower().endswith(".epub") and book.get("type") == "completed":
                    path = own
            if path and path.lower().endswith(".epub") and os.path.isfile(path):
                found.append((str(book.get("name") or os.path.basename(path)), path))
        return found

    async def pick_from_library(self, side: str) -> Optional[ActionSheet]:
        try:
            found = await self.ctx.io(self.library_epubs, side)
        except Exception:
            log.exception("listing the Library EPUBs failed")
            found = []
        items = [ActionItem(name, lambda p=path: self.ctx.spawn(self.load(side, p)), icon="MENU_BOOK")
                 for name, path in found]
        if not items:
            self.ctx.say("No Library book has an EPUB for this side")
            return None
        sheet = ActionSheet(items[:200], title="Raw EPUB from Library" if side == "raw" else
                            "Translated EPUB from Library", tablet=self.ctx.tablet)
        self.ctx.show(sheet)
        return sheet

    async def load(self, side: str, path: str) -> Optional[dict]:
        card = self.raw_card if side == "raw" else self.translated_card
        card.data["name"].value = os.path.basename(path)
        card.data["info"].value = f"Reading {'raw' if side == 'raw' else 'translated'} EPUB HTML in the background…"
        self.ctx.push(card)
        self.loading = side  # Accept answers "EPUB Still Loading" meanwhile (the dialog's _active_load)
        try:
            loaded = await self.ctx.io(self.service.pair_load, path)
        except CoreMissing as exc:
            card.data["info"].value = f"Not available in this build ({exc.name})"
            self.ctx.push(card)
            return None
        except Exception as exc:
            card.data["info"].value = str(exc)
            self.ctx.push(card)
            return None
        finally:
            self.loading = ""
        if side == "raw":
            self.raw_path, self.raw = path, loaded
            self.service.set_cfg("parallel_epub_glossary_last_raw_epub", path)
        else:
            self.translated_path, self.translated = path, loaded
            self.service.set_cfg("parallel_epub_glossary_last_translated_epub", path)
        error = loaded.get("error")
        card.data["info"].value = error or f"{len(loaded.get('chapters') or [])} HTML files"
        self.ctx.push(card)
        if self.raw_path and self.translated_path and not error:
            await self.remap_async()
        return loaded

    async def restore_selection(self, selection: dict) -> bool:
        """``restore_persisted_selection``: both EPUBs, the prompts, then the saved mapping."""
        await self.load_profiles()
        try:
            pending = self._core().prepare_persisted_parallel_epub_selection(selection)
        except CoreMissing:
            return False
        if pending is None:
            return False
        raw, translated = pending["raw_path"], pending["translated_path"]
        self._pending_selection = pending
        if selection.get("wrapper_prompt"):
            self.wrapper.value = str(selection["wrapper_prompt"])
        name = str(selection.get("profile_name") or "").strip()
        if name and name in self._profiles():
            self.profile = name
            self._render_profiles()
        if selection.get("system_prompt"):
            self.system_prompt.value = str(selection["system_prompt"])
        await self.load("raw", raw)
        await self.load("translated", translated)
        return True

    # ---- mapping -------------------------------------------------------------------------------------------

    def _on_auto_offset(self) -> None:
        self.service.set_cfg("parallel_epub_auto_offset_enabled", bool(self.auto_offset.value))
        self.remap()

    def remap(self) -> None:
        self.ctx.spawn(self.remap_async())

    async def remap_async(self) -> Optional[PairMapping]:
        if not self.raw.get("chapters") or not self.translated.get("chapters"):
            self.status.value = "Load both EPUBs to create a map."
            self.ctx.push(self.status)
            return None
        self.status.value = f"Building HTML mapping… 0/{len(self.raw['chapters'])}"
        self.ctx.push(self.status)
        raw, translated = self.raw, self.translated
        try:
            auto, predicate = await self.ctx.io(lambda: self.service.pair_auto_map(
                raw["chapters"], translated["chapters"], auto_offset=bool(self.auto_offset.value),
                raw_reading_order=raw.get("reading_order"), translated_reading_order=translated.get("reading_order")))
        except CoreMissing as exc:
            self.status.value = f"Mapping needs {exc.name} (not in this build)"
            self.ctx.push(self.status)
            return None
        core = self._core()
        self.mapping = PairMapping(raw["chapters"], translated["chapters"], auto, core=core,
                                   special_file_predicate=predicate,
                                   protect_interior=bool(self.service.cfg(
                                       "never_consider_in_between_files_as_special", True)),
                                   reading_order=translated.get("reading_order"))
        pending = self._pending_selection
        if isinstance(pending, dict) and core.parallel_epub_selection_matches(pending, self.raw_path,
                                                                              self.translated_path):
            restored, _skipped = core.restore_parallel_epub_pairs(raw["chapters"], translated["chapters"],
                                                                  pending.get("mapping") or [])
            self.mapping.restore(restored)
            self._pending_selection = None
        self.selected_rows = set()
        self.selecting = False
        self.render_mapping()
        return self.mapping

    def apply_offset(self, delta: int) -> None:
        if self.mapping is None:
            return
        self.mapping.apply_offset(delta)
        self.render_mapping()

    def set_selected_unmapped(self) -> int:
        if self.mapping is None:
            return 0
        count = self.mapping.set_unmapped(self.selected_rows)
        self.selected_rows = set()
        self.selecting = False
        self.render_mapping()
        return count

    def render_mapping(self) -> None:
        mapping = self.mapping
        if mapping is None:
            return
        self.offset_text.value = f"offset {mapping.offset:+d}"
        status_text, duplicate_count = mapping.status()
        self.status.value = status_text
        self.status.color = ft.Colors.ERROR if duplicate_count else ft.Colors.ON_SURFACE_VARIANT
        self.unmap_button.visible = bool(self.selected_rows)
        self.unmap_button.content = (f"Set {len(self.selected_rows)} Selected Rows as Unmapped"
                                     if len(self.selected_rows) > 1 else "Set This Row as Unmapped")
        self.rows.set_items(list(range(len(mapping.raw))), keep_window=True, keep_rendered=True)
        self.ctx.push(self.root)

    def _row_control(self, row: int, position: int) -> ft.Control:
        mapping = self.mapping
        raw_name = str(mapping.raw[row].get("filename") or "") if mapping else ""
        index = mapping.assign[row] if mapping else -1
        selected = row in self.selected_rows
        return ft.Container(
            content=ft.Column([
                ft.Text(raw_name, weight=ft.FontWeight.W_600, max_lines=1, overflow=ft.TextOverflow.ELLIPSIS),
                ft.Text(f"→ {mapping.label(index) if mapping else UNMAPPED}", max_lines=1,
                        overflow=ft.TextOverflow.ELLIPSIS, color=ft.Colors.ERROR if index < 0 else None),
                ft.Text(mapping.strategy[row] if mapping else "", size=11, color=ft.Colors.ON_SURFACE_VARIANT),
            ], spacing=0, tight=True),
            padding=ft.Padding.symmetric(horizontal=10, vertical=6),
            border_radius=tokens.RADII["card"],
            bgcolor=ft.Colors.SECONDARY_CONTAINER if selected else ft.Colors.SURFACE_CONTAINER_LOW,
            on_click=lambda e, r=row: self.on_row_tap(r),
            on_long_press=lambda e, r=row: self.on_row_long_press(r),
            ink=True,
        )

    def on_row_tap(self, row: int) -> Optional[ActionSheet]:
        if self.selecting:
            self.selected_rows ^= {row}
            if not self.selected_rows:
                self.selecting = False
            self.render_mapping()
            return None
        return self.open_row_picker(row)

    def on_row_long_press(self, row: int) -> None:
        self.selecting = True
        self.selected_rows.add(row)
        self.render_mapping()

    def open_row_picker(self, row: int) -> Optional[ActionSheet]:
        mapping = self.mapping
        if mapping is None:
            return None
        items = [ActionItem(UNMAPPED, lambda: self.choose(row, -1), icon="LINK_OFF")]
        for index, chapter in enumerate(mapping.translated):
            items.append(ActionItem(str(chapter.get("filename") or ""), lambda i=index: self.choose(row, i),
                                    icon="CHECK" if mapping.assign[row] == index else "DESCRIPTION",
                                    key=f"pp-pick-{index}"))
        sheet = ActionSheet(items, title=str(mapping.raw[row].get("filename") or ""),
                            subtitle="Choose the translated HTML file", tablet=self.ctx.tablet)
        self.ctx.show(sheet)
        return sheet

    def choose(self, row: int, translated_index: int) -> None:
        if self.mapping is None:
            return
        self.mapping.set_row(row, translated_index)
        self.render_mapping()

    # ---- prompts ------------------------------------------------------------------------------------------

    def _profiles(self) -> dict:
        """The profiles (``parallel_epub_profiles`` added the built-in default at open; a Reset
        re-adds it the same way)."""
        if DEFAULT_PROFILE not in self.profiles:
            self.profiles[DEFAULT_PROFILE] = self.service.default_pair_system_prompt()
        return self.profiles

    def _render_profiles(self) -> None:
        profiles = self._profiles()
        if self.profile not in profiles:
            self.profile = DEFAULT_PROFILE
        self.profile_dropdown.options = [ft.DropdownOption(key=name, text=name) for name in profiles]
        self.profile_dropdown.value = self.profile
        if not self.system_prompt.value:
            self.system_prompt.value = str(profiles.get(self.profile) or "")
        is_default = self.profile == DEFAULT_PROFILE
        self.delete_button.icon = ft.Icons.RESTART_ALT if is_default else ft.Icons.DELETE_OUTLINE
        self.delete_button.tooltip = "Reset Profile" if is_default else "Delete Profile"

    def load_profile(self, name: Optional[str]) -> None:
        profiles = self._profiles()
        if not name or name not in profiles:
            return
        self.profile = name
        self.system_prompt.value = str(profiles.get(name) or "")
        self._render_profiles()
        self.ctx.push(self.root)

    def _persist_prompts(self) -> None:
        """``_persist_prompt_settings`` (``parallel_epub_prompt_settings``)."""
        self.service.pair_persist_prompts(self.profiles, self.profile, self.wrapper.value or "")

    async def new_profile(self) -> Optional[str]:
        await self.load_profiles()
        name = await prompt_text(self.ctx, title="New Profile", label="Profile name")
        if not name:
            return None
        if name in self._profiles():
            await ask(self.ctx, title="Profile Exists", body=f"A profile named '{name}' already exists.", confirm="OK",
                      cancel="Close")
            return None
        self.profiles[name] = self.system_prompt.value or ""
        self.profile = name
        self._persist_prompts()
        self._render_profiles()
        self.ctx.push(self.root)
        return name

    def save_profile(self) -> Optional[str]:
        if not self.profiles_ready:  # never persist before the saved profiles are known
            self.ctx.say("Loading profiles…")
            return None
        name = self.profile or DEFAULT_PROFILE
        self.profiles[name] = self.system_prompt.value or ""
        self._persist_prompts()
        self.ctx.say(f"Saved profile “{name}”")
        return name

    async def delete_or_reset_profile(self) -> Optional[str]:
        await self.load_profiles()
        name = self.profile
        if not name:
            return None
        if name == DEFAULT_PROFILE:
            if not await ask(self.ctx, title="Reset Profile",
                             body="Reset the built-in Parallel EPUB Glossary profile?\n\nThe current prompt text will "
                                  "be replaced with the default pair-specific and glossary extraction instructions.",
                             confirm="Yes", cancel="No"):
                return None
            self.profiles[name] = self.service.default_pair_system_prompt()
            self.system_prompt.value = self.profiles[name]
        else:
            if not await ask(self.ctx, title="Delete Profile", body=f"Delete the profile '{name}'?", confirm="Yes",
                             cancel="No", destructive=True):
                return None
            self.profiles.pop(name, None)
            self.profile = DEFAULT_PROFILE
            self.system_prompt.value = str(self._profiles().get(DEFAULT_PROFILE) or "")
        self._persist_prompts()
        self._render_profiles()
        self.ctx.push(self.root)
        return name

    def _insert(self, placeholder: str) -> None:
        self.wrapper.value = (self.wrapper.value or "") + placeholder
        self.ctx.push(self.wrapper)

    # ---- accept --------------------------------------------------------------------------------------------

    def validate(self) -> Optional[tuple]:
        """The ``_accept_pair`` checks (``validate_parallel_epub_pair``): (title, message) of the first
        failure, or None."""
        problem = self._core().validate_parallel_epub_pair(
            loading=bool(self.loading), raw_path=self.raw_path or "", translated_path=self.translated_path or "",
            raw_chapters=self.raw.get("chapters") or [], translated_chapters=self.translated.get("chapters") or [],
            wrapper_prompt=self.wrapper.value or "", system_prompt=(self.system_prompt.value or "").strip(),
            mapping=self.mapping.selected() if self.mapping is not None else [])
        if problem is None:
            return None
        _kind, title, message = problem
        return title, message

    async def accept(self) -> Optional[str]:
        await self.load_profiles()  # the prompts persisted below include the saved profiles
        failure = self.validate()
        if failure is not None:
            await ask(self.ctx, title=failure[0], body=failure[1], confirm="OK", cancel="Close")
            return None
        mapping = self.mapping
        unmatched_raw, unused_translated = mapping.unpaired_counts()
        if (unmatched_raw or unused_translated) and not await ask(self.ctx, title="Unmapped HTML Files",
                                                                   body=mapping.unpaired_warning(), confirm="Yes",
                                                                   cancel="No"):
            return None
        system_prompt = (self.system_prompt.value or "").strip()
        profile_name = self.profile or DEFAULT_PROFILE
        self.profiles[profile_name] = system_prompt
        self.service.set_many({"parallel_epub_glossary_last_raw_epub": self.raw_path,
                               "parallel_epub_glossary_last_translated_epub": self.translated_path})
        self._persist_prompts()
        result = {"raw_path": self.raw_path, "translated_path": self.translated_path, "pairs": mapping.pairs(),
                  "wrapper_prompt": self.wrapper.value or "", "system_prompt": system_prompt,
                  "profile_name": profile_name}  # the dialog's result_data
        selection = self.service.pair_selection(result)
        key = self.service.core.value("parallel_epub_core", "PARALLEL_EPUB_SELECTION_CONFIG_KEY",
                                      default="parallel_epub_pair_selection")
        self.service.set_cfg(key, selection)
        try:
            await self.ctx.io(self.service.pair_write_sidecar, selection)
        except Exception as exc:
            self.service.log(f"⚠️ Could not save the Parallel EPUB mapping beside its glossary: {exc}")
        try:
            self.job_id = await self.service.submit(self.service.pair_spec(result))
        except Exception as exc:
            self.ctx.say(f"Could not start the pair extraction: {exc}")
            return None
        self.ctx.say("Extracting the pair glossary…", "Jobs", lambda: self.ctx.go("jobs"))
        return self.job_id

    def progress_context(self) -> Optional[dict]:
        """The desktop ``_parallel_epub_progress_manager_context`` of this pair (raw EPUB, the working EPUB
        name the extraction used, the mapped raw filenames, the cache key)."""
        if not self.raw_path or not self.translated_path:
            return None
        core = self._core()
        naming = getattr(core, "parallel_epub_working_filename", None) if core is not None else None
        name = naming(self.raw_path) if callable(naming) else os.path.basename(self.raw_path)
        generated = os.path.join(os.path.dirname(os.path.abspath(self.raw_path)), "_parallel_pair", name)
        mapping = getattr(self, "mapping", None)
        pairs = mapping.pairs() if mapping is not None else []
        return {
            "raw_path": self.raw_path,
            "generated_path": generated,
            "raw_filenames": [str(p.get("raw_filename") or "") for p in pairs if isinstance(p, dict)],
            "cache_key": ("parallel-epub::"
                          f"{os.path.normcase(os.path.abspath(self.raw_path))}::"
                          f"{os.path.normcase(os.path.abspath(generated))}"),
        }

    def open_progress(self) -> Optional[str]:
        """Glossary progress of the pair: the raw book's Glossary progress with the pair's context."""
        context = self.progress_context()
        library = getattr(self.ctx, "library", None)
        if context is None or library is None or not hasattr(library, "bid_for"):
            self.ctx.say("Load both EPUBs first")
            return None
        from glossarion_mobile.ui.library import progress_model as pm

        pm.set_parallel_context(context)
        stem = os.path.splitext(os.path.basename(self.raw_path))[0]
        bid = library.bid_for({"name": stem, "path": self.raw_path, "raw_source_path": self.raw_path})
        self.ctx.go("tools.progress.glossary", None, {"out": bid})
        return bid

    def handle_back(self) -> bool:
        if self.selecting:
            self.selecting = False
            self.selected_rows = set()
            self.render_mapping()
            return True
        return False
