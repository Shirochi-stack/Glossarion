"""Review generator (``/tools/review``, UI_SPEC §4.8; desktop ``review_dialog.ReviewDialog``).

* **Books**: the SourcePicker (several books for Volume mode or "📚 Review all Files"); ◀ / ▶
  step through them, the output pane shows the current one's review.
* **Modes** (persisted like the dialog: ``review_spoiler_mode`` / ``review_chunk_mode`` /
  ``review_chunk_wrap`` / ``review_volume_mode``): "50/50 Split Mode" · "Full Review Mode"
  (+ "Wrap Chunks" and the Final prompt) · "Volume Mode" with "↕ File Order…" (a reorderable
  list; the run reviews the files in that order as one book).
* **Prompts**: the Settings tiles of ``review_system_prompt`` / ``review_final_prompt``
  (empty = ``review_generator``'s default; ``{target_lang}`` becomes the output language),
  "↺ Reset" restores the defaults.
* **Run**: "🚀 Start Review" / "📚 Review all Files" / Stop queue a ``review`` job (the shared
  review orchestration, ``job_kinds.review``); "~N tokens" is ``review_generator.count_review_tokens``.
* **Output pane**: the review Markdown (``<output>/review/review.md``, Volume mode
  ``review/combined_review/review.md``) styled by the ⋯ Display sheet (``review_font_family`` /
  ``review_font_size`` / ``review_line_height`` / ``review_font_color`` / ``review_header_spacing``
  / ``review_spacing`` / ``review_list_gap``, desktop defaults, Reset). 💾 Save rewrites the review
  with ``review_generator._save_review_text``; 🗑️ Delete / ↩️ Restore use the shared review
  backup helpers when the build has them.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional, Sequence

import flet as ft

from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.components.sheet import fits_compact, scroll_column, sheet_frame
from glossarion_mobile.ui.router import RouteMatch
from glossarion_mobile.ui.screens.base import Screen
from glossarion_mobile.ui.tools import targets as tg
from glossarion_mobile.ui.tools.common import JobWatch, action_button, ask, card, hint_text, schema_tiles
from glossarion_mobile.ui.tools.source_picker import SourcePicker

__all__ = ["DISPLAY_DEFAULTS", "MODE_KEYS", "ReviewScreen", "display_style", "read_review", "review_spec"]

log = logging.getLogger("glossarion.tools.review")

KIND = "review"
#: Dialog checkbox -> config key (``_load_saved_prompt`` / ``_save_prompt_to_config``) and default.
MODE_KEYS = {
    "spoiler_mode": ("review_spoiler_mode", False, "50/50 Split Mode"),
    "chunk_mode": ("review_chunk_mode", False, "Full Review Mode"),
    "wrap_chunks": ("review_chunk_wrap", True, "Wrap Chunks"),
    "volume_mode": ("review_volume_mode", False, "Volume Mode"),
}
PROMPT_KEYS = ("review_system_prompt", "review_final_prompt")
#: ``review_dialog`` Font Settings defaults (``_reset_font_settings``; config keys of ``_on_review_font_changed``).
DISPLAY_DEFAULTS = {
    "review_font_family": "Segoe UI",
    "review_font_size": 9,
    "review_line_height": 100,
    "review_font_color": "#e0e0e0",
    "review_header_spacing": 6,
    "review_spacing": 8,
    "review_list_gap": 10,
}
OVERWRITE_TITLE = "Overwrite Review?"
OVERWRITE_TEXT = ("A review is already displayed in the output.\nStarting a new review will replace it.\n\n"
                  "Continue?")
GENERATE_ALL_TITLE = "Generate All Reviews"
#: review_generator helpers for 🗑️ Delete / ↩️ Restore (moved out of review_dialog._on_delete /
#: _on_restore at the U7 integration; the desktop dialog calls the same functions).
DELETE_HELPERS = ("move_review_to_backups",)
RESTORE_HELPERS = ("restore_review_backup",)
BACKUP_HELPERS = ("latest_review_backup",)
QUESTION_HELPERS = ("review_restore_question",)
HELPER_REASON = "Needs the shared review backup helpers (review_generator)"


def read_review(paths: Sequence[str]) -> tuple:
    """Blocking: ``(path, text)`` of the first existing review file (``("", "")`` when none)."""
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                text = handle.read()
        except OSError:
            continue
        if text.strip():
            return path, text
    return "", ""


def _int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def display_style(cfg: Any, *, dark: bool = False) -> ft.MarkdownStyleSheet:
    """The Display sheet's settings as a Markdown style sheet (font family/size in pt, line height %,
    colour, header / paragraph spacing, list gap)."""
    family = str(cfg("review_font_family", DISPLAY_DEFAULTS["review_font_family"]) or "") or None
    size = _int(cfg("review_font_size", DISPLAY_DEFAULTS["review_font_size"]), 9) * 4 / 3  # pt -> logical px
    height = _int(cfg("review_line_height", DISPLAY_DEFAULTS["review_line_height"]), 100) / 100.0
    color_value = str(cfg("review_font_color", DISPLAY_DEFAULTS["review_font_color"]) or "")
    # The desktop default is a light grey for its dark panel; it follows the theme here.
    color = None if color_value.lower() == DISPLAY_DEFAULTS["review_font_color"] else (color_value or None)
    header = _int(cfg("review_header_spacing", DISPLAY_DEFAULTS["review_header_spacing"]), 6)
    spacing = _int(cfg("review_spacing", DISPLAY_DEFAULTS["review_spacing"]), 8)
    gap = _int(cfg("review_list_gap", DISPLAY_DEFAULTS["review_list_gap"]), 10)
    text = ft.TextStyle(font_family=family, size=size, height=height, color=color)

    def heading(scale: float) -> ft.TextStyle:
        return ft.TextStyle(font_family=family, size=size * scale, height=height, color=color,
                            weight=ft.FontWeight.W_700)

    pad = ft.Padding.only(top=header, bottom=header)
    return ft.MarkdownStyleSheet(
        p_text_style=text, block_spacing=spacing, list_indent=gap * 2,
        list_bullet_padding=ft.Padding.only(right=gap / 2),
        h1_text_style=heading(1.6), h1_padding=pad, h2_text_style=heading(1.4), h2_padding=pad,
        h3_text_style=heading(1.2), h3_padding=pad, h4_text_style=heading(1.1), h4_padding=pad,
    )


def review_spec(paths: Sequence[str], *, mode: str, options: dict, title: str = "") -> Any:
    from glossarion_mobile.services.jobs import JobSpec

    name = title or (os.path.basename(paths[0]) if paths else "")
    if mode == "all" and len(paths) > 1:
        name = f"{len(paths)} files"
    return JobSpec(kind=KIND, title=name, inputs=tuple(paths), params={"mode": mode, **options},
                   origin={"type": "tool", "label": "Tools · Review generator"})


def _review_generator() -> Any:
    try:
        import review_generator
    except Exception:
        return None
    return review_generator


def _helper(names: Sequence[str]) -> Any:
    module = _review_generator()
    if module is None:
        return None
    for name in names:
        fn = getattr(module, name, None)
        if callable(fn):
            return fn
    return None


class ReviewScreen(Screen):
    title = "Review generator"

    def __init__(self, match: Optional[RouteMatch], ctx: Any) -> None:
        super().__init__(match)
        self.ctx = ctx
        self.state = ctx.tool_state.setdefault("review", {})
        self.targets: list = list(self.state.get("targets") or [])
        self.index = int(self.state.get("index") or 0)
        self.watch = JobWatch(ctx, self._on_job_end, self._on_job_change)
        self.review_path = ""
        self.review_text = ""
        self.switches: dict = {}
        self.picker: Optional[SourcePicker] = None
        self.display_sheet: Any = None
        self.order_sheet: Any = None

    # ---- layout -----------------------------------------------------------------------------------

    def actions(self) -> list:
        return [ft.IconButton(icon=ft.Icons.TEXT_FIELDS, tooltip="Display", key="review-display-action",
                              on_click=lambda e: self.open_display_sheet())]

    def build_body(self) -> ft.Control:
        self.book_text = ft.Text("", theme_style=ft.TextThemeStyle.BODY_MEDIUM, expand=True, key="review-book")
        self.nav_prev = ft.IconButton(icon=ft.Icons.CHEVRON_LEFT, tooltip="Previous", key="review-prev",
                                      on_click=lambda e: self.step(-1))
        self.nav_next = ft.IconButton(icon=ft.Icons.CHEVRON_RIGHT, tooltip="Next", key="review-next",
                                      on_click=lambda e: self.step(1))
        self.token_text = ft.Text("", theme_style=ft.TextThemeStyle.BODY_SMALL, key="review-tokens")
        books = card("Books", [
            ft.Row([self.nav_prev, self.book_text, self.nav_next], vertical_alignment=ft.CrossAxisAlignment.CENTER),
            ft.Row([ft.FilledTonalButton(content="Choose…", icon=ft.Icons.FOLDER_OPEN, key="review-choose",
                                         on_click=lambda e: self.open_picker())], wrap=True),
            self.token_text,
        ], icon="MENU_BOOK", key="review-books")
        for name, (key, default, label) in MODE_KEYS.items():
            self.switches[name] = ft.Switch(label=label, value=bool(self.ctx.cfg(key, default)),
                                            on_change=lambda e, n=name: self.set_mode(n, bool(e.control.value)),
                                            key=f"review-{name}")
        self.order_button = ft.TextButton(content="↕ File Order…", on_click=lambda e: self.open_order_sheet(),
                                          key="review-order")
        modes = card("Modes", [self.switches["spoiler_mode"], self.switches["chunk_mode"],
                               self.switches["wrap_chunks"], ft.Row([self.switches["volume_mode"], self.order_button],
                                                                    wrap=True)],
                     icon="TUNE", key="review-modes")
        tiles, self.prompt_tiles = schema_tiles(self.ctx, PROMPT_KEYS)
        prompts = card("Prompts", (tiles or [hint_text("Settings › Review generator")]) + [
            hint_text("Empty uses the default prompt; {target_lang} becomes the output language.",
                      key="review-prompt-hint"),
            ft.TextButton(content="↺ Reset prompts to default", key="review-reset-prompts",
                          on_click=lambda e: self.ctx.spawn(self.reset_prompts())),
        ], icon="EDIT_NOTE", key="review-prompts")
        self.start_button = ft.FilledButton(content="🚀 Start Review", key="review-start",
                                            on_click=lambda e: self.ctx.spawn(self.start("single")))
        self.all_button = ft.FilledTonalButton(content="📚 Review all Files", key="review-all",
                                               on_click=lambda e: self.ctx.spawn(self.start("all")))
        self.stop_button = ft.OutlinedButton(content="Stop Review", icon=ft.Icons.STOP, visible=False,
                                             on_click=self._on_stop, key="review-stop")
        self.progress = ft.ProgressBar(visible=False, key="review-progress")
        self.run_status = ft.Text("", theme_style=ft.TextThemeStyle.BODY_SMALL, key="review-run-status")
        run = card("Run", [ft.Row([self.start_button, self.all_button, self.stop_button], wrap=True, spacing=8),
                           self.progress, self.run_status], icon="PLAY_CIRCLE", key="review-run")
        self.output = ft.Markdown("", selectable=True, extension_set=ft.MarkdownExtensionSet.GITHUB_WEB,
                                  md_style_sheet=display_style(self.ctx.cfg, dark=self.ctx.dark), key="review-output")
        self.output_hint = hint_text("Generated review will appear here...", key="review-output-hint")
        self.file_actions = ft.Row(wrap=True, spacing=6, run_spacing=6, key="review-file-actions")
        output = card("Review", [self.file_actions, self.output_hint, self.output], icon="ARTICLE",
                      key="review-output-card")
        self._render()
        return ft.ListView(controls=[books, modes, prompts, run, output], expand=True, spacing=tokens.SPACING["md"],
                           padding=ft.Padding.symmetric(horizontal=12, vertical=8), key="review-screen")

    def did_show(self) -> None:
        for snap in self.watch.adopt((KIND,)):
            self._on_job_change(snap)
        self.ctx.spawn(self.load_current())

    def dispose(self) -> None:
        self.watch.stop()

    # ---- state ------------------------------------------------------------------------------------

    @property
    def volume(self) -> bool:
        return bool(self.switches["volume_mode"].value) if self.switches else bool(
            self.ctx.cfg("review_volume_mode", False))

    def current(self) -> Optional[tg.ToolTarget]:
        if not self.targets:
            return None
        self.index = max(0, min(self.index, len(self.targets) - 1))
        return self.targets[self.index]

    def review_paths(self) -> list:
        """The dialog's review files (``review_generator.review_paths_for``: ``review/review.md`` in the
        book's output folder; Volume mode ``review/combined_review/review.md`` in every volume)."""
        rg = _review_generator()
        paths_for = getattr(rg, "review_paths_for", None) if rg is not None else None
        sources = [t.source for t in self.targets if t.source]
        target = self.current()
        if not callable(paths_for):
            return []
        config = self.ctx.config_snapshot()
        try:
            if self.volume and sources:
                return list(paths_for(sources[0], sources, True, config))
            if target is not None and target.source:
                return list(paths_for(target.source, [], False, config))
        except Exception:
            log.debug("review_paths_for failed", exc_info=True)
        return []

    def _render(self) -> None:
        target = self.current()
        if target is None:
            self.book_text.value = "Choose a book"
        elif self.volume and len(self.targets) > 1:
            self.book_text.value = f"Volume mode · {len(self.targets)} files · first: {target.source_name or target.title}"
        else:
            self.book_text.value = (f"{self.index + 1} / {len(self.targets)} · " if len(self.targets) > 1 else "") + (
                target.source_name or target.title)
        self.nav_prev.disabled = len(self.targets) <= 1 or self.volume
        self.nav_next.disabled = self.nav_prev.disabled
        chunk = bool(self.switches["chunk_mode"].value)
        self.switches["wrap_chunks"].visible = chunk
        if "review_final_prompt" in getattr(self, "prompt_tiles", {}):
            self.prompt_tiles["review_final_prompt"].control.visible = chunk
        self.order_button.visible = self.volume
        busy = self.watch.active() is not None
        has_source = target is not None and bool(target.source)
        self.start_button.disabled = busy or not has_source
        self.all_button.disabled = busy or len(self.targets) <= 1 or self.volume
        self.start_button.content = "🚀 Start Volume Review" if self.volume else "🚀 Start Review"
        self.stop_button.content = "Stop Volume Review" if self.volume else "Stop Review"
        has_review = bool(self.review_text.strip())
        self.output.value = self.review_text
        self.output.visible = has_review
        self.output_hint.visible = not has_review
        delete_fn, restore_fn = _helper(DELETE_HELPERS), _helper(RESTORE_HELPERS)
        self.file_actions.controls = [
            action_button("💾 Save", "SAVE", lambda e: self.ctx.spawn(self.save()), key="review-save",
                          reason=None if has_review and not busy else "No review to save"),
            action_button("🗑️ Delete", "DELETE_OUTLINE", lambda e: self.ctx.spawn(self.delete()), key="review-delete",
                          destructive=True, reason=HELPER_REASON if delete_fn is None else (
                              None if has_review and not busy else "No review to delete")),
            action_button("↩️ Restore", "RESTORE", lambda e: self.ctx.spawn(self.restore()), key="review-restore",
                          reason=HELPER_REASON if restore_fn is None else ("A review is running" if busy else None)),
        ]

    def set_mode(self, name: str, value: bool) -> None:
        key = MODE_KEYS[name][0]
        self.ctx.set_cfg(key, bool(value))
        switch = self.switches.get(name)
        if switch is not None and switch.value != bool(value):
            switch.value = bool(value)
        if name == "volume_mode":
            self.index = 0
            self.ctx.spawn(self.load_current())
        self._render()
        self._push(self.body)

    def step(self, offset: int) -> None:
        if not self.targets:
            return
        self.index = (self.index + offset) % len(self.targets)
        self.state["index"] = self.index
        self._render()
        self._push(self.body)
        self.ctx.spawn(self.load_current())

    def open_picker(self) -> SourcePicker:
        picker = SourcePicker(self.ctx, title="Review books", multi=True,
                              eligible=lambda t: None if t.source else "No raw source file",
                              on_done=self.set_targets, selected=self.targets, segment="library")
        self.picker = picker
        if self.ctx.page is not None:
            picker.show(self.ctx.page)
            self.ctx.spawn(picker.load())
        return picker

    def set_targets(self, targets: Sequence[tg.ToolTarget]) -> None:
        self.targets = list(targets)
        self.index = 0
        self.state.update(targets=self.targets, index=0)
        self._render()
        self._push(self.body)
        self.ctx.spawn(self.load_current())

    async def load_current(self) -> str:
        paths = self.review_paths()
        path, text = await self.ctx.io(read_review, paths) if paths else ("", "")
        self.review_path, self.review_text = path, text
        target = self.current()
        self.token_text.value = ""
        rg = _review_generator()
        if target is not None and target.source and rg is not None and hasattr(rg, "count_review_tokens"):
            review_input = [t.source for t in self.targets if t.source] if self.volume else target.source
            self.token_text.value = "⏳ Counting tokens..."
            self._push(self.token_text)
            try:
                count = await self.ctx.io(lambda: rg.count_review_tokens(review_input, lambda *_a: None))
                self.token_text.value = f"~{int(count):,} tokens"
            except Exception:
                self.token_text.value = ""
        self._render()
        self._push(self.body)
        return text

    # ---- prompts / display ------------------------------------------------------------------------

    async def reset_prompts(self) -> None:
        answer = await ask(self.ctx, "Reset prompts", "Reset system prompt to default?",
                           [("no", "No", "text"), ("yes", "Yes", "filled")], key="review-reset")
        if answer != "yes":
            return
        rg = _review_generator()
        self.ctx.set_cfg("review_system_prompt", getattr(rg, "DEFAULT_REVIEW_PROMPT", "") if rg else "")
        self.ctx.set_cfg("review_final_prompt", getattr(rg, "DEFAULT_FINAL_REVIEW_PROMPT", "") if rg else "")
        for tile in getattr(self, "prompt_tiles", {}).values():
            try:
                tile.refresh()
            except Exception:
                pass

    def open_display_sheet(self) -> Any:
        fields = {}

        def field(key: str, label: str, numeric: bool = True) -> ft.TextField:
            control = ft.TextField(label=label, value=str(self.ctx.cfg(key, DISPLAY_DEFAULTS[key])), dense=True,
                                   keyboard_type=ft.KeyboardType.NUMBER if numeric else ft.KeyboardType.TEXT,
                                   on_blur=lambda e, k=key, n=numeric: self.set_display(k, e.control.value, n),
                                   on_submit=lambda e, k=key, n=numeric: self.set_display(k, e.control.value, n),
                                   key=f"review-display-{key}")
            fields[key] = control
            return control

        controls = [
            ft.Text("Display", theme_style=ft.TextThemeStyle.TITLE_LARGE, weight=ft.FontWeight.W_600),
            field("review_font_family", "Font family", numeric=False),
            field("review_font_size", "Font size (pt)"),
            field("review_line_height", "Line height (%)"),
            field("review_font_color", "Font colour (#rrggbb)", numeric=False),
            field("review_header_spacing", "Header spacing"),
            field("review_spacing", "Paragraph spacing"),
            field("review_list_gap", "List gap"),
            ft.TextButton(content="↺ Reset", key="review-display-reset", on_click=lambda e: self.reset_display(fields)),
        ]
        # Scrolls: with the keyboard up the lower fields and Reset are past the sheet, and a focused
        # field is scrolled into view only inside a scroll view.
        sheet = ft.BottomSheet(content=sheet_frame(scroll_column(controls, spacing=8),
                                                   padding=ft.Padding.only(left=16, right=16, bottom=24)),
                               show_drag_handle=True, scrollable=True, key="review-display-sheet")
        self.display_sheet = sheet
        self.display_fields = fields
        if self.ctx.page is not None:
            self.ctx.page.show_dialog(sheet)
        return sheet

    def set_display(self, key: str, value: Any, numeric: bool = True) -> None:
        text = str(value or "").strip()
        if numeric:
            try:
                parsed: Any = int(float(text))
            except ValueError:
                return
        else:
            parsed = text or DISPLAY_DEFAULTS[key]
        self.ctx.set_cfg(key, parsed)
        self.output.md_style_sheet = display_style(self.ctx.cfg, dark=self.ctx.dark)
        self._push(self.output)

    def reset_display(self, fields: Optional[dict] = None) -> None:
        for key, default in DISPLAY_DEFAULTS.items():
            self.ctx.set_cfg(key, default)
            if fields and key in fields:
                fields[key].value = str(default)
        self.output.md_style_sheet = display_style(self.ctx.cfg, dark=self.ctx.dark)
        self._push(self.output, *(fields or {}).values())

    def open_order_sheet(self) -> Any:
        """Volume mode "↕ File Order…": drag to reorder (the run reviews the files in this order)."""
        rows = [ft.ListTile(title=ft.Text(t.source_name or t.title), leading=ft.Icon(ft.Icons.DRAG_HANDLE),
                            key=f"order-{i}") for i, t in enumerate(self.targets)]
        # Unless the rows surely fit (estimate at 200 % text), the list takes the rest of the sheet's
        # height and scrolls itself: shrink-wrapped in a tight Column it never scrolled, so volumes
        # past the screen could not be seen or reordered.
        fits = fits_compact(self.ctx.page, 200 + 80 * len(rows))
        listing = ft.ReorderableListView(controls=rows, on_reorder=self._on_reorder, expand=not fits,
                                         key="review-order-list")
        self.order_listing = listing
        sheet = ft.BottomSheet(content=sheet_frame(ft.Column([
            ft.Text("File order", theme_style=ft.TextThemeStyle.TITLE_LARGE, weight=ft.FontWeight.W_600),
            hint_text("The combined review is saved in every volume under review/combined_review/."),
            listing], tight=fits, spacing=8), padding=ft.Padding.only(left=16, right=16, bottom=24)),
            show_drag_handle=True, scrollable=True, key="review-order-sheet")
        self.order_sheet = sheet
        if self.ctx.page is not None:
            self.ctx.page.show_dialog(sheet)
        return sheet

    def _on_reorder(self, e: Any) -> None:
        """A row was dragged: Flet does not reorder ``controls`` itself (1.0.3 ``on_reorder``), so the
        sheet's rows follow ``targets`` here (the order the Volume review runs in)."""
        old, new = int(getattr(e, "old_index", 0) or 0), int(getattr(e, "new_index", 0) or 0)
        listing = getattr(e, "control", None) or getattr(self, "order_listing", None)
        controls = getattr(listing, "controls", None)
        if isinstance(controls, list) and 0 <= old < len(controls):
            controls.insert(max(0, min(new, len(controls) - 1)), controls.pop(old))
            try:
                listing.update()
            except Exception:
                pass
        self.move_target(old, new)

    def move_target(self, old: int, new: int) -> None:
        if not (0 <= old < len(self.targets)):
            return
        item = self.targets.pop(old)
        new = max(0, min(new, len(self.targets)))
        self.targets.insert(new, item)
        self.state["targets"] = self.targets
        self._render()
        self._push(self.body)

    # ---- run ---------------------------------------------------------------------------------------

    def options(self) -> dict:
        return {name: bool(self.switches[name].value) for name in ("spoiler_mode", "chunk_mode", "wrap_chunks")}

    async def start(self, which: str) -> Optional[str]:
        if not self.ctx.has_kind(KIND):
            self.ctx.say("Review jobs are not available in this build")
            return None
        target = self.current()
        if target is None or not target.source:
            self.ctx.say("Choose a book first")
            return None
        if which == "all" and not self.volume and len(self.targets) > 1:
            paths = [t.source for t in self.targets if t.source]
            answer = await ask(self.ctx, GENERATE_ALL_TITLE,
                               f"Generate reviews for all {len(paths)} EPUBs?\n\n"
                               "This will process each EPUB and save the review automatically.",
                               [("no", "No", "text"), ("yes", "Yes", "filled")], key="review-all-confirm")
            if answer != "yes":
                return None
            mode = "all"
        else:
            existing = await self.ctx.io(lambda: any(os.path.exists(p) for p in self.review_paths()))
            if existing:
                answer = await ask(self.ctx, OVERWRITE_TITLE, OVERWRITE_TEXT,
                                   [("no", "No", "text"), ("yes", "Yes", "filled")], key="review-overwrite")
                if answer != "yes":
                    return None
            if self.volume:
                paths, mode = [t.source for t in self.targets if t.source], "volume"
            else:
                paths, mode = [target.source], "single"
        spec = review_spec(paths, mode=mode, options=self.options(), title=target.source_name or target.title)
        job_id = await self.ctx.submit(spec)
        if job_id:
            self.watch.watch(job_id)
            self.ctx.remember_source("tools.review", target.title)
            self._set_running(True, "Queued…")
        return job_id

    def _set_running(self, running: bool, text: str) -> None:
        self.progress.visible = running
        self.stop_button.visible = running
        self.run_status.value = text
        self._render()
        self._push(self.body)

    def _on_stop(self, e: Any = None) -> None:
        snap = self.watch.active()
        if snap is not None and self.ctx.jobs is not None:
            try:
                self.ctx.jobs.request_stop(snap.id)
            except Exception:
                log.exception("stopping the review failed")

    def _on_job_change(self, snap: Any) -> None:
        if not getattr(snap, "is_terminal", False):
            self._set_running(True, str(getattr(snap, "phase", "") or "Reviewing…"))

    def _on_job_end(self, snap: Any) -> None:
        error = getattr(snap, "error", None)
        text = "Stopped" if getattr(snap, "stopped", False) else (f"Failed: {error}" if error else "Done")
        self._set_running(False, text)
        self.ctx.spawn(self.load_current())

    # ---- review file actions ---------------------------------------------------------------------------

    async def save(self) -> bool:
        rg = _review_generator()
        text = self.review_text.strip()
        paths = self.review_paths()
        if rg is None or not text or not paths:
            return False
        try:
            await self.ctx.io(lambda: rg._save_review_text(text, os.path.dirname(os.path.dirname(paths[0])),
                                                           lambda *_a: None, review_output_paths=paths))
        except OSError as exc:
            self.ctx.say(f"❌ Failed to save review: {exc}")
            return False
        self.ctx.say("✅ Saved!")
        return True

    async def delete(self) -> bool:
        fn = _helper(DELETE_HELPERS)
        if fn is None:
            self.ctx.say(HELPER_REASON)
            return False
        paths = [p for p in self.review_paths() if os.path.exists(p)]
        if not paths:
            self.ctx.say("📭 No Review")
            return False
        try:
            await self.ctx.io(fn, paths)
        except Exception as exc:
            self.ctx.say(f"❌ Failed to delete review: {exc}")
            return False
        self.ctx.say("✅ Copies moved to backups" if len(paths) > 1 else "✅ Moved to backups")
        await self.load_current()
        return True

    async def restore(self) -> bool:
        fn = _helper(RESTORE_HELPERS)
        if fn is None:
            self.ctx.say(HELPER_REASON)
            return False
        paths = self.review_paths()
        latest_fn, question_fn = _helper(BACKUP_HELPERS), _helper(QUESTION_HELPERS)
        latest = await self.ctx.io(latest_fn, paths) if latest_fn is not None else None
        if latest_fn is not None and not latest:
            self.ctx.say("No backup to restore")  # desktop: the button is hidden without a backup
            return False
        if any(os.path.exists(p) for p in paths):
            question = (question_fn(latest[1]) if question_fn is not None and latest else
                        "Your current review will be overwritten.\n\nAre you sure you want to restore?")
            answer = await ask(self.ctx, "Restore Backup", question,
                               [("no", "No", "text"), ("yes", "Yes", "filled")], key="review-restore-confirm")
            if answer != "yes":
                return False
        try:
            restored = await self.ctx.io(fn, paths)
        except Exception as exc:
            self.ctx.say(f"❌ Failed to restore: {exc}")
            return False
        if restored is False:
            self.ctx.say("No backup to restore")
            return False
        self.ctx.say("✅ Restored!")
        await self.load_current()
        return True

    def _push(self, *controls: Any) -> None:
        for control in controls:
            if control is None:
                continue
            try:
                control.update()
            except Exception:
                pass
