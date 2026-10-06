"""Edit raw (UI_SPEC §4.1 "⋯ › Edit raw"; desktop "✏️ Edit in Notepad").

A full-screen mono editor over the glossary file. Reading and saving reuse the chat's
generated-glossary editor helpers (``ui.chat.cards.glossary_preview`` reads with the BOM
detected; ``save_text_keep_bom`` replaces the file atomically and keeps a UTF-8 BOM), so a
raw edit never changes the file's encoding. The editor reloads the table after a save.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Optional

import flet as ft

from glossarion_mobile.ui.theme import HIT_TARGET

__all__ = ["RawGlossaryEditor"]


class RawGlossaryEditor:
    def __init__(self, ctx: Any, path: str, text: str, *, has_bom: bool,
                 on_saved: Optional[Callable[[str], Any]] = None) -> None:
        from glossarion_mobile.ui.chat.cards import save_text_keep_bom

        self.ctx = ctx
        self.path = path
        self.has_bom = has_bom
        self.on_saved = on_saved
        self._save = save_text_keep_bom
        self.saved = False
        self.error_text = ft.Text("", color=ft.Colors.ERROR, visible=False, key="raw-error")
        self.editor = ft.TextField(value=text, multiline=True, min_lines=20, expand=True,
                                   text_style=ft.TextStyle(font_family=getattr(ctx, "mono", "monospace"), size=13),
                                   border=ft.OutlineInputBorder(), key="raw-editor")
        self.view = ft.View(
            route="/glossary/raw",
            appbar=ft.AppBar(
                title=ft.Text(os.path.basename(path), max_lines=1, overflow=ft.TextOverflow.ELLIPSIS),
                leading=ft.IconButton(icon=ft.Icons.CLOSE, tooltip="Cancel", on_click=lambda e: self.close(),
                                      size_constraints=HIT_TARGET),
                actions=[ft.TextButton(content="Save", on_click=lambda e: ctx.spawn(self.save()), key="raw-save")],
            ),
            controls=[ft.SafeArea(expand=True, content=ft.Column([
                ft.Text(path, theme_style=ft.TextThemeStyle.LABEL_SMALL, selectable=True,
                        color=ft.Colors.ON_SURFACE_VARIANT),
                self.editor,
                self.error_text,
            ], expand=True, spacing=8))],
            padding=12,
        )

    @classmethod
    async def open(cls, ctx: Any, path: str, *, on_saved: Optional[Callable[[str], Any]] = None
                   ) -> Optional["RawGlossaryEditor"]:
        from glossarion_mobile.ui.chat.cards import glossary_preview

        info = await ctx.io(glossary_preview, path)
        if not info.get("exists"):
            ctx.say("The glossary file could not be found.")
            return None
        editor = cls(ctx, path, info.get("text") or "", has_bom=bool(info.get("bom")), on_saved=on_saved)
        if ctx.push_overlay is not None:
            ctx.push_overlay(editor.view)
        return editor

    async def save(self) -> bool:
        text = self.editor.value or ""
        try:
            await self.ctx.io(self._save, self.path, text, self.has_bom)
        except Exception as exc:
            self.error_text.value = f"Could not save the glossary:\n\n{exc}"
            self.error_text.visible = True
            self.ctx.push(self.error_text)
            return False
        self.saved = True
        self.ctx.say(f"Saved {os.path.basename(self.path)}")
        self.close()
        if self.on_saved is not None:
            self.on_saved(self.path)
        return True

    def close(self) -> None:
        pop = getattr(self.ctx, "pop_overlay", None)
        if callable(pop):
            try:
                pop()
            except Exception:
                pass
