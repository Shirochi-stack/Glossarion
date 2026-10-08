"""Metadata editor (``/library/book/<bid>/metadata``; UI_SPEC §3.6 METADATA ✏️ Edit, §3.5 ⋯ Edit metadata.json).

Full-screen form with the desktop ``_BookMetadataEditDialog`` fields (Title, Author,
Publisher, Language, Date, Tags (comma / newline), Synopsis) and its note: "Changes
are saved to the output workspace's metadata.json. The original EPUB is not
modified." The form opens with ``BookDetailsModel.editor_values()``; Save sends only
the changed fields (``metadata_changed_values``) to ``save_metadata_json_atomic``
(``original_*`` kept, ``<field>_translated`` set so a later compile does not
translate over a manual value). A **metadata.json** segment edits the raw file
(validated JSON, atomic replace). The workspace is the book's resolved one
(``LibraryService.workspace_for``), so an organized Library/Translated book edits the
metadata.json of the workspace it came from.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Mapping, Optional

import flet as ft

from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.components.empty_state import EmptyState
from glossarion_mobile.ui.library import progress_model as pm
from glossarion_mobile.ui.library.common import LibraryContext
from glossarion_mobile.ui.library.overview_tab import hero_values, split_tags
from glossarion_mobile.ui.screens.base import Screen

__all__ = ["EDITOR_FIELDS", "EDITOR_NOTE", "MetadataEditorScreen", "changed_values", "editor_values"]

log = logging.getLogger("glossarion.library.ui")

EDITOR_NOTE = "Changes are saved to the output workspace's metadata.json. The original EPUB is not modified."
#: (metadata.json field, form label, multiline) - the desktop dialog's rows.
EDITOR_FIELDS = (("title", "Title", False), ("creator", "Author", False), ("publisher", "Publisher", False),
                 ("language", "Language", False), ("date", "Date", False), ("subject", "Tags", True),
                 ("description", "Synopsis", True))


def editor_values(book: Mapping[str, Any], payload: Optional[Mapping[str, Any]], model: Any = None) -> dict:
    """The values the form opens with (``BookDetailsModel.editor_values`` when available)."""
    if model is not None:
        try:
            values = model.editor_values()
            if isinstance(values, Mapping):
                return {field: str(values.get(field) or "") for field, _l, _m in EDITOR_FIELDS}
        except Exception:
            log.debug("editor_values failed", exc_info=True)
    hero = hero_values(book, payload, None)
    return {"title": hero["title"], "creator": hero["author"], "publisher": hero["publisher"],
            "language": hero["language"], "date": hero["date"], "subject": ", ".join(hero["tags"]),
            "description": hero["synopsis"]}


def changed_values(initial: Mapping[str, Any], current: Mapping[str, Any], core: Any = None) -> dict:
    """Only the edited fields (``metadata_changed_values``: tags compared as tag sets)."""
    fn = core.fn("library_core", "metadata_changed_values") if core is not None else None
    if fn is not None:
        try:
            return dict(fn(dict(initial), dict(current)))
        except Exception:
            log.debug("metadata_changed_values failed", exc_info=True)
    out = {}
    for field, value in current.items():
        if field == "subject":
            if split_tags(initial.get(field)) != split_tags(value):
                out[field] = value
        elif str(initial.get(field) or "").strip() != str(value or "").strip():
            out[field] = value
    return out


class MetadataEditorScreen(Screen):
    title = "Edit metadata"

    def __init__(self, match: Any, ctx: LibraryContext) -> None:
        super().__init__(match)
        self.ctx = ctx
        self.service = ctx.service
        self.bid = str(match.params.get("bid") or "") if match is not None else ""
        self.book: dict = self.service.book_for_bid(self.bid) or {}
        self.payload: Optional[dict] = None
        self.initial: dict = {}
        self.fields: dict = {}
        self.saved: Any = None
        self.mode = "form"
        self.json_text = ""
        self.save_button = ft.TextButton(content="Save", on_click=self._on_save, disabled=True, key="meta-save")

    def actions(self) -> list:
        return [self.save_button]

    @property
    def workspace(self) -> str:
        """The book's resolved output workspace (``LibraryService.workspace_for``; an organized
        Library/Translated book edits the workspace it came from)."""
        return pm.book_workspace(self.service, self.book) if self.book else ""

    def _target(self) -> dict:
        """The row carrying its resolved workspace, for the shared metadata calls (the row keeps its id)."""
        return pm.workspace_row(self.service, self.book)

    def build_body(self) -> ft.Control:
        if not self.book or not self.workspace:
            return EmptyState(icon="EDIT_OFF", title="No output workspace",
                              body="Metadata can be edited once this book has an output workspace.",
                              key="meta-none")
        for field, label, multiline in EDITOR_FIELDS:
            self.fields[field] = ft.TextField(
                label=label, multiline=multiline, min_lines=3 if field == "description" else (2 if multiline else 1),
                max_lines=12 if field == "description" else (4 if multiline else 1),
                hint_text="Separate tags with commas or new lines" if field == "subject" else None,
                on_change=self._on_change, key=f"meta-{field}")
        self.mode_buttons = ft.SegmentedButton(
            segments=[ft.Segment(value="form", label=ft.Text("Form")),
                      ft.Segment(value="json", label=ft.Text("metadata.json"))],
            selected=["form"], show_selected_icon=False, on_change=self._on_mode, key="meta-mode")
        self.json_field = ft.TextField(multiline=True, min_lines=12, max_lines=40,
                                       text_style=ft.TextStyle(size=13,
                                                               font_family=getattr(self.ctx, "mono", "monospace")),
                                       visible=False, on_change=self._on_change, key="meta-json")
        self.error = ft.Text("", color=ft.Colors.ERROR, visible=False, key="meta-error")
        self.form = ft.Column(list(self.fields.values()), spacing=tokens.SPACING["md"], tight=True, key="meta-form")
        return ft.ListView([
            self.mode_buttons,
            ft.Text(EDITOR_NOTE, theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT,
                    key="meta-note"),
            self.form,
            self.json_field,
            self.error,
        ], spacing=tokens.SPACING["md"], padding=16, expand=True)

    def did_show(self) -> None:
        if self.book and self.workspace:
            self.ctx.spawn(self.load())

    async def load(self) -> None:
        target = self._target()  # the workspace's metadata.json, also for an organized book
        try:
            self.payload = await self.ctx.io(self.service.load_details_blocking, target, "preview")
        except Exception as exc:
            log.info("metadata editor details failed: %s", exc)
            self.payload = {}
        model = self.service.details_model(target, self.payload)
        self.initial = editor_values(target, self.payload, model)
        for field, control in self.fields.items():
            control.value = self.initial.get(field, "")
        path = os.path.join(self.workspace, "metadata.json")

        def read() -> str:
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    return handle.read()
            except OSError:
                return "{}"

        self.json_text = await self.ctx.io(read)
        if getattr(self, "json_field", None) is not None:
            self.json_field.value = self.json_text
        self.ctx.push(*self.fields.values(), getattr(self, "json_field", None))

    def current_values(self) -> dict:
        return {field: str(control.value or "") for field, control in self.fields.items()}

    def edits(self) -> dict:
        return changed_values(self.initial, self.current_values(), self.service.core)

    def _on_change(self, e: Any = None) -> None:
        dirty = (str(self.json_field.value or "") != self.json_text) if self.mode == "json" else bool(self.edits())
        self.save_button.disabled = not dirty
        self.ctx.push(self.save_button)

    def _on_mode(self, e: Any = None) -> None:
        selected = list(getattr(self.mode_buttons, "selected", []) or ["form"])
        self.mode = selected[0]
        self.form.visible = self.mode == "form"
        self.json_field.visible = self.mode == "json"
        self._on_change()
        self.ctx.push(self.form, self.json_field)

    async def save(self) -> Any:
        self.error.visible = False
        try:
            if self.mode == "json":
                text = str(self.json_field.value or "")
                json.loads(text)
                result = await self.ctx.io(self.service.save_metadata_json_text_blocking, self._target(), text)
                self.json_text = text
            else:
                edits = self.edits()
                if not edits:
                    return None
                result = await self.ctx.io(self.service.save_metadata_blocking, self._target(), edits, self.payload)
                if isinstance(result, Mapping):
                    self.book["metadata_json"] = dict(result)
                self.initial = self.current_values()
        except Exception as exc:
            self.error.value = str(exc) if "metadata.json" in str(exc) else f"Could not save metadata.json:\n{exc}"
            self.error.visible = True
            self.ctx.push(self.error)
            return None
        self.saved = result
        self.save_button.disabled = True
        self.ctx.push(self.save_button)
        self.ctx.say("Saved metadata.json")
        await self.service.refresh(reason="metadata")
        return result

    async def _on_save(self, e: Any = None) -> Any:
        return await self.save()
