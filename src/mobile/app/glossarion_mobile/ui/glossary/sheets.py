"""Glossary sheets: Extract glossary, ✨ Refine, the glossary mode, PlanGlossarySheet (UI_SPEC §2.5, §3.8, §4.1, §5.4).

* ``ExtractSheet`` - the desktop "Extract Glossary" source: this book, a Library book, a file
  (EPUB / TXT / PDF / subtitles), an image folder (the folder's images become one combined
  glossary, like the desktop image-folder path) or the Parallel EPUB pair; it submits an
  ``extract_glossary`` job (the Balanced/Full engine, ``extract_glossary_from_epub``).
* ``RefineSheet`` - the Glossary Progress refinement preview: the active entry types with
  their entry counts ("Already refined: …" warning), an optional exact chunk count, "✨ Refine
  Now" → a ``glossary_refine`` job.
* ``ModeSheet`` - the 8 glossary modes (desktop combo order and labels); choosing one runs
  the shortcut handler and the mode lock pass (``GlossaryService.set_mode``).
* ``PlanGlossarySheet`` - what a translation will use: the effective mode line and the loaded
  glossary, Glossary mode ▾ · Load file… · Use book glossary · Clear ✕ · Review glossary (the
  desktop main-window glossary row: 📄 Load Glossary, the status label and ✕).
"""

from __future__ import annotations

import os
from typing import Any, Callable, Mapping, Optional, Sequence

import flet as ft

from glossarion_mobile.ui.components._handlers import call_handler
from glossarion_mobile.ui.glossary.common import SheetHost, sheet

__all__ = ["ExtractSheet", "IMAGE_EXTENSIONS", "ModeSheet", "PlanGlossarySheet", "RefineSheet"]

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp")
SOURCE_EXTENSIONS = ["epub", "txt", "pdf", "srt", "ass", "lrc", "zip", "sdlxliff", "html", "htm", "md", "cbz"]


class ExtractSheet:
    def __init__(self, ctx: Any, *, source_path: Optional[str] = None, title: Optional[str] = None,
                 on_submit: Callable[[list, Optional[str]], Any], on_pair: Optional[Callable[[], Any]] = None) -> None:
        self.ctx = ctx
        self.host = SheetHost(ctx)
        self.on_submit = on_submit
        self.on_pair = on_pair
        service = ctx.service
        reason = None if service.has_job_kind("extract_glossary") else "The job service is not running"
        detail = None
        if reason is None:  # U9 preflight: the extraction runs the main model; an excluded route cannot start
            from glossarion_mobile.services.model_catalog import job_model_block

            block = job_model_block("extract_glossary", {}, service.cfg)
            if block is not None:
                reason, detail = block
        self.start_reason = reason
        tiles: list = [ft.Text(f"Glossary mode: {service.mode_label()} · Extract Glossary runs the Balanced/Full "
                               "extractor with the Glossary settings.", theme_style=ft.TextThemeStyle.BODY_SMALL,
                               color=ft.Colors.ON_SURFACE_VARIANT, key="ex-mode")]
        if source_path:
            tiles.append(self._tile(f"This book: {title or os.path.basename(source_path)}", "MENU_BOOK",
                                    lambda: self.submit([source_path], title), "ex-this", reason, detail))
        tiles.append(self._tile("Library book…", "LOCAL_LIBRARY", lambda: ctx.spawn(self.pick_library_book()),
                                "ex-library", reason, detail))
        tiles.append(self._tile("File… (EPUB, TXT, PDF, subtitles, CBZ)", "DESCRIPTION",
                                lambda: ctx.spawn(self.pick_file()), "ex-file", reason, detail))
        tiles.append(self._tile("Image folder…", "PHOTO_LIBRARY", lambda: ctx.spawn(self.pick_images()), "ex-images",
                                reason, detail))
        if on_pair is not None:
            tiles.append(self._tile("Parallel EPUB pair…", "COMPARE_ARROWS", self._pair, "ex-pair", None))
        self.dialog = sheet("Extract glossary", tiles, actions=[
            ft.TextButton(content="Cancel", on_click=lambda e: self.host.close())], key="ex-sheet")

    @staticmethod
    def _tile(label: str, icon: str, handler: Callable[[], Any], key: str, reason: Optional[str],
              detail: Optional[str] = None) -> ft.ListTile:
        from glossarion_mobile.ui.components.reason_chip import unavailable_tile
        from glossarion_mobile.ui.theme import icon_data

        if reason:  # never a disabled ListTile: Flet would disable the ReasonChip too (UI_SPEC §5.2)
            return unavailable_tile(label, reason=reason, detail=detail, leading=ft.Icon(icon_data(icon)), key=key,
                                    min_height=48, dense=False)
        return ft.ListTile(leading=ft.Icon(icon_data(icon)), title=ft.Text(label), trailing=None,
                           on_click=lambda e: handler(), key=key, min_height=48)

    def show(self, page: Any = None) -> "ExtractSheet":
        self.host.open(self.dialog)
        return self

    def submit(self, inputs: list, title: Optional[str] = None) -> Any:
        self.host.close()
        return call_handler(self.on_submit, list(inputs), title)

    def _pair(self) -> Any:
        self.host.close()
        return call_handler(self.on_pair)

    async def pick_library_book(self) -> Any:
        """The Library books with a raw source (resolved on the io pool: the Library resolvers read files)."""
        from glossarion_mobile.ui.components.action_sheet import ActionItem, ActionSheet

        library = self.ctx.library
        snapshot = getattr(library, "snapshot", None) if library is not None else None
        books = list(snapshot.all_books()) if snapshot is not None else []

        def resolve() -> list:
            found = []
            for book in books:
                try:
                    source = library.raw_source(book)
                except Exception:
                    source = ""
                if source:
                    found.append((str(book.get("name") or os.path.basename(source)), source))
            return found

        try:
            found = await self.ctx.io(resolve) if books else []
        except Exception:
            found = []
        items = [ActionItem(name, lambda s=source, n=name: self.submit([s], n), icon="MENU_BOOK")
                 for name, source in found]
        if not items:
            self.ctx.say("No Library book has a raw source file")
            return None
        self.host.close()
        picker = ActionSheet(items[:300], title="Extract from a Library book", tablet=self.ctx.tablet)
        self.ctx.show(picker)
        return picker

    async def pick_file(self) -> Optional[list]:
        files = self.ctx.files
        if files is None:
            self.ctx.say("Picking files is not available in this session")
            return None
        picked = await files.pick_files(target="inbox", allowed_extensions=SOURCE_EXTENSIONS, allow_multiple=True,
                                        dialog_title="Extract glossary from…")
        if not picked:
            return None
        paths = [p.path for p in picked]
        archives = [p for p in paths if p.lower().endswith(".cbz")]
        if archives:
            # a CBZ is its page images, as one image group named after the archive (like an image folder)
            for archive in archives:
                try:
                    images = await self.ctx.io(self.ctx.service.expand_cbz, archive)
                except Exception as exc:
                    self.ctx.say(f"Could not read {os.path.basename(archive)}: {exc}")
                    continue
                if not images:
                    self.ctx.say(f"{os.path.basename(archive)} has no images")
                    continue
                self.submit(images, os.path.splitext(os.path.basename(archive))[0])
            paths = [p for p in paths if p not in archives]
            if not paths:
                return archives
        self.submit(paths)
        return paths

    async def pick_images(self) -> Optional[list]:
        """A folder of images (copied into the Inbox); Android without folder access: pick the images."""
        files = self.ctx.files
        if files is None:
            self.ctx.say("Picking files is not available in this session")
            return None
        folder = None
        try:
            imported = await files.pick_folder(dialog_title="Image folder")
            folder = imported.path
        except Exception as exc:  # FolderPickUnavailable on Android SAF trees
            reason = getattr(exc, "reason", str(exc))
            self.ctx.say(f"{reason} — pick the images instead")
        if folder:
            images = sorted(os.path.join(folder, n) for n in os.listdir(folder)
                            if os.path.splitext(n)[1].lower() in IMAGE_EXTENSIONS)
            if not images:
                self.ctx.say("The folder has no images")
                return None
            self.submit(images, os.path.basename(folder))
            return images
        picked = await files.pick_files(target="inbox", allowed_extensions=[e.lstrip(".") for e in IMAGE_EXTENSIONS],
                                        allow_multiple=True, dialog_title="Images")
        images = [p.path for p in picked or [] if os.path.splitext(p.path)[1].lower() in IMAGE_EXTENSIONS]
        if images:
            self.submit(images, "images")
        return images or None


class RefineSheet:
    """The refinement preview (``_confirm_manual_glossary_refinement``) without the token metrics."""

    def __init__(self, ctx: Any, *, glossary_path: str, types: Sequence[tuple], selected: Sequence[str] = (),
                 completed: Sequence[str] = (), on_refine: Callable[[list, Optional[int]], Any],
                 reason: Optional[str] = None) -> None:
        self.ctx = ctx
        self.host = SheetHost(ctx)
        self.on_refine = on_refine
        wanted = {str(t).casefold() for t in selected}
        self.completed = {str(t).casefold() for t in completed}
        self.checks: dict = {}
        rows: list = [ft.Text(os.path.basename(glossary_path), theme_style=ft.TextThemeStyle.LABEL_LARGE,
                              key="rf-file")]
        for name, count in types:
            check = ft.Checkbox(label=f"{name}  ({count:,} entries)" if count else f"{name}  (empty)",
                                value=(not wanted or name.casefold() in wanted) and count > 0, disabled=count <= 0,
                                on_change=lambda e: self._sync(), key=f"rf-type-{name}")
            self.checks[name] = check
            rows.append(check)
        self.warning = ft.Text("", color=ft.Colors.ERROR, theme_style=ft.TextThemeStyle.BODY_SMALL, visible=False,
                               key="rf-warning")
        self.chunks = ft.TextField(label="Exact total chunk count (optional)", dense=True,
                                   keyboard_type=ft.KeyboardType.NUMBER, key="rf-chunks")
        self.summary = ft.Text("", theme_style=ft.TextThemeStyle.BODY_SMALL, key="rf-summary")
        rows += [self.summary, self.warning, self.chunks]
        if reason:
            from glossarion_mobile.ui.components.reason_chip import ReasonChip

            rows.append(ReasonChip(reason="Refinement unavailable", detail=reason))
        self.refine_button = ft.FilledButton(content="✨  Refine Now", on_click=lambda e: self.refine(),
                                             disabled=reason is not None, key="rf-refine")
        self.counts = {name: count for name, count in types}
        self.dialog = sheet("Confirm Glossary Refinement", rows, actions=[
            ft.TextButton(content="Cancel", on_click=lambda e: self.host.close()), self.refine_button],
            key="rf-sheet")
        self.reason = reason
        self._sync()

    def selected(self) -> list:
        return [name for name, check in self.checks.items() if check.value]

    def _sync(self) -> None:
        names = self.selected()
        total = sum(self.counts.get(n, 0) for n in names)
        self.summary.value = (f"{len(names)} type{'s' if len(names) != 1 else ''} selected • {total:,} entries"
                              if names else "No entry types selected.")
        already = [n for n in names if n.casefold() in self.completed]
        self.warning.value = ("⚠️ Already refined: " + ", ".join(already) +
                              ". This run will replace them with fresh results.") if already else ""
        self.warning.visible = bool(already)
        self.refine_button.disabled = self.reason is not None or total <= 0
        self.ctx.push(self.dialog)

    def refine(self) -> Any:
        names = self.selected()
        if not names:
            return None
        try:
            target = int(str(self.chunks.value or "").strip()) if str(self.chunks.value or "").strip() else None
        except ValueError:
            target = None
        self.host.close()
        return call_handler(self.on_refine, names, target)

    def show(self, page: Any = None) -> "RefineSheet":
        self.host.open(self.dialog)
        return self


class ModeSheet:
    """The glossary mode selector (8 modes) with the lock note."""

    def __init__(self, ctx: Any, *, on_changed: Optional[Callable[[str], Any]] = None) -> None:
        self.ctx = ctx
        self.host = SheetHost(ctx)
        self.on_changed = on_changed
        service = ctx.service
        self.group = ft.RadioGroup(value=service.mode(), on_change=lambda e: self.choose(self.group.value),
                                   content=ft.Column([ft.Radio(value=mode, label=label, key=f"mode-{mode}")
                                                      for mode, label in service.modes()], spacing=0, tight=True),
                                   key="mode-group")
        self.dialog = sheet("Glossary mode", [
            self.group,
            ft.Text("Append Glossary, Auto-Mapping and Fuzzy Auto-Mapping follow the mode (🔒 in Settings › Glossary "
                    "› General).", theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT),
        ], actions=[ft.TextButton(content="Done", on_click=lambda e: self.host.close())], key="mode-sheet")

    def choose(self, mode: Optional[str]) -> dict:
        if not mode:
            return {}
        changed = self.ctx.service.set_mode(mode)
        call_handler(self.on_changed, mode)
        return changed

    def show(self, page: Any = None) -> "ModeSheet":
        self.host.open(self.dialog)
        return self


class PlanGlossarySheet:
    """What a translation will use (the desktop main-window glossary row) for the chat Plan / Library."""

    def __init__(self, ctx: Any, *, book: Optional[Mapping[str, Any]] = None, on_load_file: Callable[[], Any],
                 on_use_book: Optional[Callable[[], Any]] = None, on_clear: Callable[[], Any],
                 on_review: Optional[Callable[[], Any]] = None, on_mode: Callable[[], Any],
                 effective: str = "", on_map: Optional[Callable[[], Any]] = None,
                 on_settings: Optional[Callable[[], Any]] = None) -> None:
        self.ctx = ctx
        self.host = SheetHost(ctx)
        service = ctx.service
        loaded = str(service.cfg("manual_glossary_path", "") or "")
        self.mode_tile = ft.ListTile(leading=ft.Icon(ft.Icons.RULE), title=ft.Text(f"Glossary mode: {service.mode_label()}"),
                                     subtitle=ft.Text(effective) if effective else None,
                                     trailing=ft.Icon(ft.Icons.ARROW_DROP_DOWN), key="pg-mode",
                                     on_click=lambda e: self._run(on_mode))
        self.loaded_text = ft.Text(f"Loaded: {os.path.basename(loaded)}" if loaded else "No glossary loaded (auto-mapping "
                                   "picks the book glossary)", theme_style=ft.TextThemeStyle.BODY_SMALL, key="pg-loaded")
        tiles = [
            self.mode_tile,
            self.loaded_text,
            ft.ListTile(leading=ft.Icon(ft.Icons.FILE_OPEN), title=ft.Text("Load file…"), key="pg-load",
                        on_click=lambda e: self._run(on_load_file)),
            ft.ListTile(leading=ft.Icon(ft.Icons.MENU_BOOK), title=ft.Text("Use book glossary"), key="pg-book",
                        disabled=on_use_book is None, on_click=lambda e: self._run(on_use_book)),
            ft.ListTile(leading=ft.Icon(ft.Icons.CLOSE, color=ft.Colors.ERROR), title=ft.Text("Clear ✕"),
                        key="pg-clear", disabled=not loaded, on_click=lambda e: self._run(on_clear)),
            ft.ListTile(leading=ft.Icon(ft.Icons.EDIT_NOTE), title=ft.Text("Review glossary"), key="pg-review",
                        disabled=on_review is None, on_click=lambda e: self._run(on_review)),
        ]
        if on_map is not None:  # batch / several EPUBs: desktop "Map Glossaries to EPUBs"
            tiles.append(ft.ListTile(leading=ft.Icon(ft.Icons.ACCOUNT_TREE), title=ft.Text("Map glossaries…"),
                                     subtitle=ft.Text("One glossary per EPUB"), key="pg-map",
                                     on_click=lambda e: self._run(on_map)))
        if on_settings is not None:
            tiles.append(ft.TextButton(content="Change glossary mode settings", icon=ft.Icons.SETTINGS,
                                       on_click=lambda e: self._run(on_settings), key="pg-settings"))
        title = "Glossary" + (f" · {book.get('name')}" if book else "")
        self.dialog = sheet(title, tiles, actions=[ft.TextButton(content="Done", on_click=lambda e: self.host.close())],
                            key="pg-sheet")

    def _run(self, handler: Optional[Callable[[], Any]]) -> Any:
        self.host.close()
        return call_handler(handler)

    def show(self, page: Any = None) -> "PlanGlossarySheet":
        self.host.open(self.dialog)
        return self
