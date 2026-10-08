"""Map glossaries (desktop "Map Glossaries to EPUBs", ``TranslatorGUI._open_glossary_mapping_dialog``).

A full-screen sheet with one row per EPUB of a batch / multi-book translation: the glossary the
row will append (prefilled from the saved ``manual_glossary_map`` through
``glossary_files.mapped_glossary_for_input``, else the desktop guess), "📂 Select File" (FileBridge,
glossary extensions) and "Clear"; the desktop footer "Use one Glossary" · "Clear All" ·
"Auto-Fill" · Cancel · Save. Save runs ``GlossaryService.save_glossary_map`` on the io pool
(``build_glossary_mapping``: a missing file refuses the save with the desktop "Missing glossary
file" text; then ``manual_glossary_map``, the global manual glossary cleared, Append Glossary on,
and ``copy_mapped_glossaries_to_outputs`` in Manual Glossary Only mode). Texts are the dialog's.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Optional, Sequence

import flet as ft

from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.components._handlers import call_handler
from glossarion_mobile.ui.glossary.common import SheetHost
from glossarion_mobile.ui.theme import HIT_TARGET

__all__ = ["HEADER_TEXT", "MapGlossariesSheet", "MISSING_TITLE", "NO_MATCH_TEXT"]

HEADER_TEXT = "Select which glossary should be appended for each EPUB:"
MISSING_TITLE = "Missing glossary file"
NO_MATCH_TEXT = "No matching glossaries were found."


class MapGlossariesSheet:
    """``rows``: [(epub path, glossary path or "")]. ``pick()`` (async) returns a picked glossary path
    or None; ``guess(epub)`` (blocking, io pool) the Auto-Fill guess; ``save(rows)`` (blocking, io
    pool) returns ``{"missing": [...], "mapping": {...}}``; ``on_saved(result)`` after a save."""

    def __init__(self, ctx: Any, rows: Sequence[tuple], *, pick: Callable[[], Any], guess: Callable[[str], str],
                 save: Callable[[list], dict], on_saved: Optional[Callable[[dict], Any]] = None) -> None:
        self.ctx = ctx
        self.host = SheetHost(ctx)
        self.pick = pick
        self.guess = guess
        self.save_fn = save
        self.on_saved = on_saved
        self.values: list = [[str(epub), str(glossary or "")] for epub, glossary in rows]
        self.fields: list = []
        self.saving = False
        self.result: Optional[dict] = None
        self.error_text = ft.Text("", color=ft.Colors.ERROR, theme_style=ft.TextThemeStyle.BODY_SMALL,
                                  visible=False, key="map-error")
        row_controls: list = []
        for index, (epub, glossary) in enumerate(self.values):
            field = ft.TextField(value=glossary, read_only=True, hint_text="(no glossary)", dense=True, expand=True,
                                 border_radius=tokens.RADII["field"], color=ft.Colors.ON_SURFACE,
                                 key=f"map-field-{index}")
            self.fields.append(field)
            row_controls.append(ft.Container(
                content=ft.Column([
                    ft.Text(os.path.basename(epub), theme_style=ft.TextThemeStyle.TITLE_SMALL, max_lines=2,
                            overflow=ft.TextOverflow.ELLIPSIS),
                    ft.Row([field], spacing=4),
                    ft.Row([
                        ft.TextButton(content="📂 Select File", on_click=lambda e, i=index: self.ctx.spawn(self.pick_for(i)),
                                      key=f"map-pick-{index}"),
                        ft.TextButton(content="Clear", on_click=lambda e, i=index: self.set_row(i, ""),
                                      key=f"map-clear-{index}"),
                    ], spacing=4, wrap=True),
                ], spacing=2, tight=True),
                padding=ft.Padding.symmetric(horizontal=8, vertical=6),
                bgcolor=ft.Colors.SURFACE_CONTAINER_LOW,
                border_radius=tokens.RADII["card"],
                key=f"map-row-{index}",
            ))
        self.save_button = ft.FilledButton(content="Save", on_click=lambda e: self.ctx.spawn(self.save()), key="map-save")
        footer = ft.Row([
            ft.TextButton(content="Use one Glossary", on_click=lambda e: self.ctx.spawn(self.apply_to_all()),
                          key="map-all"),
            ft.TextButton(content="Clear All", on_click=lambda e: self.clear_all(), key="map-clear-all"),
            ft.TextButton(content="Auto-Fill", on_click=lambda e: self.ctx.spawn(self.autofill()), key="map-autofill"),
        ], wrap=True, spacing=4)
        header = ft.Row([
            ft.IconButton(icon=ft.Icons.CLOSE, tooltip="Cancel", on_click=lambda e: self.close(),
                          size_constraints=HIT_TARGET, key="map-cancel"),
            ft.Text("Map Glossaries to EPUBs", theme_style=ft.TextThemeStyle.TITLE_MEDIUM, expand=True,
                    weight=ft.FontWeight.W_600),
            self.save_button,
        ], vertical_alignment=ft.CrossAxisAlignment.CENTER)
        self.dialog = ft.BottomSheet(
            content=ft.SafeArea(content=ft.Container(
                padding=ft.Padding.only(left=8, right=12, top=8, bottom=12),
                content=ft.Column([
                    header,
                    ft.Text(HEADER_TEXT, weight=ft.FontWeight.W_600, theme_style=ft.TextThemeStyle.BODY_MEDIUM),
                    ft.ListView(controls=row_controls, expand=True, spacing=6, key="map-list"),
                    self.error_text,
                    footer,
                ], spacing=tokens.SPACING["sm"], expand=True),
            ), expand=True),
            fullscreen=True,
            scrollable=False,
            show_drag_handle=False,
            bgcolor=ft.Colors.SURFACE,
            key="map-sheet",
        )

    # ---- rows ------------------------------------------------------------------------------------

    def set_row(self, index: int, path: str) -> None:
        if not 0 <= index < len(self.values):
            return
        self.values[index][1] = str(path or "")
        field = self.fields[index]
        field.value = self.values[index][1]
        self._push(field)

    def rows(self) -> list:
        return [(epub, glossary) for epub, glossary in self.values]

    async def pick_for(self, index: int) -> Optional[str]:
        path = await self._call(self.pick)
        if path:
            self.set_row(index, path)
        return path

    async def apply_to_all(self) -> Optional[str]:
        path = await self._call(self.pick)
        if path:
            for index in range(len(self.values)):
                self.set_row(index, path)
        return path

    def clear_all(self) -> None:
        for index in range(len(self.values)):
            self.set_row(index, "")

    async def autofill(self) -> int:
        epubs = [epub for epub, _g in self.values]
        guesses = await self.ctx.io(lambda: [self.guess(epub) for epub in epubs])
        matched = 0
        for index, guess in enumerate(guesses):
            if guess:
                self.set_row(index, guess)
                matched += 1
        if matched == 0:
            self.ctx.say(NO_MATCH_TEXT)
        return matched

    # ---- save ------------------------------------------------------------------------------------

    async def save(self) -> Optional[dict]:
        if self.saving:
            return None
        self.saving = True
        self.save_button.disabled = True
        self._push(self.save_button)
        try:
            result = await self.ctx.io(self.save_fn, self.rows())
        except Exception as exc:
            self._error(f"Could not save the glossary mapping: {exc}")
            return None
        finally:
            self.saving = False
            self.save_button.disabled = False
            self._push(self.save_button)
        missing = list((result or {}).get("missing") or [])
        if missing:
            self._error(f"{MISSING_TITLE}: These mappings point to files that don’t exist:\n\n" + "\n".join(missing))
            return result
        self.result = dict(result or {})
        self.close()
        count = len(self.result.get("mapping") or {})
        self.ctx.say(f"📑 Saved glossary mapping for {count} EPUB(s)")
        if self.on_saved is not None:
            call_handler(self.on_saved, self.result)
        return self.result

    def _error(self, text: str) -> None:
        self.error_text.value = text
        self.error_text.visible = bool(text)
        self._push(self.error_text)

    # ---- presentation ----------------------------------------------------------------------------

    def show(self, page: Any = None) -> "MapGlossariesSheet":
        self.host.open(self.dialog)
        return self

    def close(self) -> None:
        self.host.close()

    async def _call(self, fn: Callable[[], Any]) -> Any:
        result = fn()
        if hasattr(result, "__await__"):
            result = await result
        return result

    @staticmethod
    def _push(*controls: Any) -> None:
        for control in controls:
            try:
                control.update()
            except Exception:
                pass
