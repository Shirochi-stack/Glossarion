"""FindReplaceSheet (UI_SPEC §4.1, §5.6): the desktop editor "Find / Replace" (Ctrl+F) dialog.

Find · Replace with · a live preview ("12 rows · 15 matches", case-insensitive like the
desktop) and Find Next · Replace · Replace All · Close. Every action is the shared
``GlossaryDocument`` step: Find Next (``find_next_index`` over the shown rows, wrapping)
jumps the editor list to the hit; Replace changes the current row (an undo snapshot only
when it matches); Replace All snapshots once. When Replace All finds nothing in the
glossary, the desktop "No glossary match" question offers to update the output files
directly; that replacement is undoable (an output-file undo step). Status texts are the
desktop dialog's.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional

import flet as ft

from glossarion_mobile.ui.glossary.common import SheetHost, ask, sheet

__all__ = ["FindReplaceSheet", "NO_MATCH_BODY", "NO_MATCH_TITLE", "no_match_texts"]

#: Fallbacks of ``glossary_document.NO_GLOSSARY_MATCH_TITLE`` / ``_TEXT``.
NO_MATCH_TITLE = "No glossary match"
NO_MATCH_BODY = "No entry found in the glossary. Update output files directly?"
PREVIEW_DEBOUNCE = 0.3


def no_match_texts(service: Any) -> tuple:
    core = getattr(service, "core", None)
    title = core.value("glossary_document", "NO_GLOSSARY_MATCH_TITLE", default=None) if core else None
    text = core.value("glossary_document", "NO_GLOSSARY_MATCH_TEXT", default=None) if core else None
    return str(title or NO_MATCH_TITLE), str(text or NO_MATCH_BODY)


class FindReplaceSheet:
    def __init__(self, editor: Any) -> None:
        self.editor = editor
        self.ctx = editor.ctx
        self.service = editor.service
        self.host = SheetHost(self.ctx)
        self._preview_task: Any = None
        self.find_field = ft.TextField(label="Find", value=editor.last_find, autofocus=True, dense=True,
                                       on_change=lambda e: self._schedule_preview(),
                                       on_submit=lambda e: self.ctx.spawn(self.find_next()), key="fr-find")
        self.replace_field = ft.TextField(label="Replace with", value=editor.last_replace, dense=True, key="fr-replace")
        self.preview = ft.Text("", theme_style=ft.TextThemeStyle.LABEL_MEDIUM, color=ft.Colors.ON_SURFACE_VARIANT,
                               key="fr-preview")
        self.status = ft.Text("", theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT,
                              key="fr-status")
        self.dialog = sheet("Find / Replace", [self.find_field, self.replace_field, self.preview, self.status], actions=[
            ft.TextButton(content="Find Next", on_click=lambda e: self.ctx.spawn(self.find_next()), key="fr-next"),
            ft.TextButton(content="Replace", on_click=lambda e: self.replace_current(), key="fr-one"),
            ft.FilledTonalButton(content="Replace All", on_click=lambda e: self.ctx.spawn(self.replace_all()),
                                 key="fr-all"),
            ft.TextButton(content="Close", on_click=lambda e: self.close(), key="fr-close"),
        ], key="fr-sheet")
        self.update_preview()

    def show(self, page: Any = None) -> "FindReplaceSheet":
        self.host.open(self.dialog)
        return self

    def close(self) -> None:
        self.editor.last_find = self.find_field.value or ""
        self.editor.last_replace = self.replace_field.value or ""
        self.host.close()

    def _say(self, text: str) -> None:
        self.status.value = text
        self.ctx.push(self.status)

    # ---- preview ---------------------------------------------------------------------------------------

    def update_preview(self) -> tuple:
        doc = self.editor.doc
        find = self.find_field.value or ""
        if doc is None or not find:
            self.preview.value = ""
            return 0, 0
        try:
            rows, occurrences = self.service.preview_replace(doc, find, self.editor.view_rows())
        except Exception:
            self.preview.value = ""
            return 0, 0
        self.preview.value = (f"{rows:,} row{'s' if rows != 1 else ''} · {occurrences:,} match"
                              f"{'es' if occurrences != 1 else ''}" if rows else "No matches in the glossary")
        return rows, occurrences

    def _schedule_preview(self) -> None:
        task = self._preview_task
        if task is not None and not task.done():
            task.cancel()

        async def later() -> None:
            await asyncio.sleep(PREVIEW_DEBOUNCE)
            self.update_preview()
            self.ctx.push(self.preview)

        try:
            self._preview_task = self.ctx.spawn(later())
        except RuntimeError:
            self.update_preview()

    # ---- actions ---------------------------------------------------------------------------------------

    async def find_next(self) -> Optional[int]:
        """``find_next``: the next shown row (after the last hit, wrapping) whose columns contain the text."""
        text = self.find_field.value or ""
        doc = self.editor.doc
        rows = list(self.editor.visible)
        if not text or doc is None or not rows:
            return None
        index = self.service.find_next(doc, text, self.service.editor_rows(doc, rows))
        if index is None:
            self._say("No matches found.")
            return None
        self.editor.last_find = text
        self._say(f"Found at row {index + 1}")
        await self.editor.jump_to(rows[index].key)
        return index

    def replace_current(self) -> int:
        find = self.find_field.value or ""
        key = self.editor.current_key
        spec = self.editor.spec_for_key(key) if key else None
        if not find or spec is None:
            self._say("No matches in current row.")
            return 0
        count = self.editor.replace_in_row(spec, find, self.replace_field.value or "")
        self._say(f"Replaced {count} occurrence(s) in current row" if count else "No matches in current row.")
        self.update_preview()
        self.ctx.push(self.preview)
        return count

    async def replace_all(self) -> int:
        find = self.find_field.value or ""
        repl = self.replace_field.value or ""
        if not find or self.editor.doc is None:
            return 0
        total = await self.editor.replace_all(find, repl)
        self.editor.last_find, self.editor.last_replace = find, repl
        self._say(f"Replaced {total} occurrence(s) across all entries.")
        self.update_preview()
        self.ctx.push(self.preview)
        if total == 0:
            title, text = no_match_texts(self.service)
            if await ask(self.ctx, title=title, body=f"{text}\n\n{find} -> {repl}", confirm="Yes", cancel="No"):
                files_updated, replacements = await self.editor.replace_in_outputs(find, repl)
                self._say(f"Updated {files_updated} files directly ({replacements} replacements).")
        return total
