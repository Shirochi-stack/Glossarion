"""SDLXLIFF reviewer, compact layout (``/tools/sdlxliff?out=<oid>``, UI_SPEC §4.9).

Desktop ``SDLXLIFFReviewDialog`` (Retranslation_GUI) over the shared reviewer core:
``sdlxliff_review_core.open_sdlxliff_review`` builds an ``SdlxliffReviewSession`` - the dialog's
data half (``SdlxliffReviewCoreMixin``, moved verbatim) without widgets - for the output folder,
with the Progress Manager context (raw source for sidecar generation, the loaded progress);
the session's methods run on the io pool:

* **Book / piece navigation**: ``session.books`` (another discovered book: ``switch_book``) and
  ``session.pieces`` with the dialog's sidebar labels (``piece_summary``): a ``Dropdown`` and
  ``‹ ›`` steppers.
* **Legend filter chips** (the dialog's legend: "green ok" · "yellow density/tag-level" ·
  "purple MT inaccurate" · "red dropped/added/empty/untranslated") filter the rows.
* **Row cards**: source (small, muted) over an editable output field with the status colour
  bar and reason; a changed field saves on blur through ``save_row`` (the shared save
  semantics incl. Manual editing). A save replaces only that row's card (and the header
  counts): the other fields keep their focus and text. Full re-renders (piece / book / filter /
  layout changes, an action's result) first save any typed text, and the 2 s poll does not
  reload while a field has focus or unsaved text.
* **Machine translation**: provider sheet (Auto / Google / DeepL / Bing / Yandex; Argos shown
  disabled: "Offline engine (ctranslate2) not available on mobile"; the free fallback chain
  skips it), "🔑 Configure … API Key…" (``set_machine_translation_credentials``: encrypted like
  the desktop's), "🌐 Generate Machine Translation Preview" (``machine_translation_preview``)
  and per-row **Inject MT** (``inject_machine_translation``).
* **🟣 Flag Inaccurate** (``flag_inaccurate``) with "Set Score Threshold…" / "Reset Threshold".
* **Piece ⋯**: Mark as Completed / Undo (``translation_progress.json`` +
  ``SDLXLIFF/review_status_overrides.json``), Edit Output (the text editor on the output file).
* **↻ Refresh** (``refresh(force=True)``: missing / stale sidecars are regenerated) and a 2 s
  ``changed_on_disk`` poll while the screen is visible: on top of the stack and the app in the
  foreground (like the Library and Book page pollers).
* **Notepad layout** (tablets only): the piece's whole output document (``notepad_document``,
  built on the io pool) in a code editor (XML) saved with ``edit_document`` + ``flush_edits``;
  on phones the layout toggle is shown disabled with the ReasonChip "Tablet layout only".
* **Edits are never dropped**: like the dialog's ``closeEvent`` (Notepad HTML captured, queued
  edits flushed), leaving the reviewer saves an edited Notepad document and any typed row text;
  a re-render (piece / book / filter / layout / refresh) saves them first too.

The Chapters tab opens this screen with ``open_reviewer`` (the route carries only the folder's
``fid``; the piece to focus, the raw source and the loaded progress travel in-process).
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional

import flet as ft

from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.components.reason_chip import ReasonChip
from glossarion_mobile.ui.foreground import poll_sleep
from glossarion_mobile.ui.router import RouteMatch
from glossarion_mobile.ui.screens.base import Screen
from glossarion_mobile.ui.theme import HIT_TARGET
from glossarion_mobile.ui.tools.common import hint_text

__all__ = ["ARGOS_REASON", "LEGEND", "NOTEPAD_REASON", "PROVIDERS", "ReviewerBinding", "ReviewerRequest",
           "SdlxliffScreen", "open_reviewer", "request_open", "sidecar_path_for", "take_request"]

log = logging.getLogger("glossarion.tools.sdlxliff")

#: The dialog's legend (``_legend_status_label`` texts and statuses).
LEGEND = (("green", "green ok"), ("yellow", "yellow density/tag-level"), ("purple", "purple MT inaccurate"),
          ("red", "red dropped/added/empty/untranslated"))
STATUS_COLORS = {"green": "#28a745", "yellow": "#d39e00", "purple": "#b967ff", "red": "#dc3545"}
#: ``MACHINE_TRANSLATION_PROVIDER_LABELS`` order (the session's labels win when present).
PROVIDERS = (("auto", "Auto"), ("google", "Google"), ("deepl", "DeepL"), ("bing", "Bing"),
             ("argos", "Argos Translate"), ("yandex", "Yandex"))
ARGOS_REASON = "Offline engine (ctranslate2) not available on mobile"
NOTEPAD_REASON = "Tablet layout only"
NO_CORE = "The SDLXLIFF reviewer core is not available in this build"
#: The dialog's credential prompts per provider: (session keyword, label, secret).
CREDENTIAL_FIELDS = {
    "deepl": (("api_key", "DeepL API key:", True),),
    "bing": (("api_key", "Microsoft Translator API key:", True), ("region", "Azure region (optional):", False)),
    "yandex": (("api_key", "Yandex Cloud API key:", True), ("folder_id", "Yandex Cloud folder ID:", False)),
}
POLL_SECONDS = 2.0


@dataclass
class ReviewerRequest:
    source: Optional[str] = None
    focus: Optional[str] = None  # output file name of the piece to show first
    manual_editing: bool = False
    progress_data: Any = field(default=None, repr=False)  # the Progress Manager's loaded progress
    # Manual editing: the Not Translated / Pending rows (progress_core.untranslated_manual_entries) that
    # get source-only sidecars (U9; desktop autogen_manual_entries)
    manual_entries: Any = field(default=None, repr=False)


_REQUESTS: dict = {}
_LOCK = threading.Lock()


def _key(folder: str) -> str:
    return os.path.normcase(os.path.abspath(str(folder or "")))


def request_open(folder: str, *, source: Optional[str] = None, focus: Optional[str] = None,
                 manual_editing: bool = False, progress_data: Any = None, manual_entries: Any = None) -> None:
    with _LOCK:
        _REQUESTS[_key(folder)] = ReviewerRequest(source=source, focus=focus, manual_editing=bool(manual_editing),
                                                  progress_data=progress_data,
                                                  manual_entries=list(manual_entries) if manual_entries else None)


def take_request(folder: str) -> ReviewerRequest:
    with _LOCK:
        return _REQUESTS.pop(_key(folder), None) or ReviewerRequest()


def open_reviewer(ctx: Any, folder: str, *, source: Optional[str] = None, focus: Optional[str] = None,
                  manual_editing: bool = False, progress_data: Any = None, manual_entries: Any = None) -> Optional[str]:
    """Tools › SDLXLIFF reviewer on ``folder`` (``?out=<fid>``; the rest in-process)."""
    prefs = getattr(ctx, "prefs", None)
    if prefs is None or not folder:
        return None
    request_open(folder, source=source, focus=focus, manual_editing=manual_editing, progress_data=progress_data,
                 manual_entries=manual_entries)
    fid = prefs.file_ref(folder)
    ctx.go("tools.sdlxliff", None, {"out": fid})
    return fid


def sidecar_path_for(output_dir: str, output_file: Optional[str]) -> Optional[str]:
    """The row's sidecar (desktop ``_sdlxliff_sidecar_path_for_output_file``): ``SDLXLIFF/<output>.sdlxliff``."""
    name = os.path.basename(str(output_file or "").replace("\\", "/"))
    return os.path.join(output_dir, "SDLXLIFF", f"{name}.sdlxliff") if name else None


class ConfigParent:
    """The session's context parent: ``config`` + ``save_config`` (``_persist_review_config_value``
    saves through it); the keys that changed go to the mobile config store."""

    def __init__(self, config: dict, save: Optional[Callable[[str, Any], Any]] = None) -> None:
        self.config = config
        self._save = save
        self._saved = dict(config)

    def save_config(self, show_message: bool = False) -> bool:
        changed = {k: v for k, v in self.config.items() if self._saved.get(k, object()) != v}
        for key, value in changed.items():
            if self._save is not None:
                self._save(key, value)
        self._saved = dict(self.config)
        return True


class ReviewerBinding:
    """The shared review session for one output folder (blocking calls: run them on the io pool)."""

    def __init__(self, core: Any, output_dir: str, *, source: Optional[str] = None, focus: Optional[str] = None,
                 config: Optional[dict] = None, save: Optional[Callable[[str, Any], Any]] = None,
                 progress_data: Any = None, manual_entries: Any = None) -> None:
        opener = getattr(core, "open_sdlxliff_review", None) if core is not None else None
        if not callable(opener):
            raise LookupError(NO_CORE)
        self.output_dir = output_dir
        self.focus = focus
        self.parent = ConfigParent(dict(config or {}), save)
        kwargs: dict = {"manual_entries": list(manual_entries)} if manual_entries else {}
        self.session = opener(output_dir, self.parent.config, current_path=sidecar_path_for(output_dir, focus),
                              context_parent=self.parent, source_path=source, progress_data=progress_data,
                              load=False, **kwargs)

    # ---- data ----------------------------------------------------------------------------------

    @property
    def status(self) -> str:
        try:
            return str(self.session.status or "")
        except Exception:
            return ""

    @property
    def pieces(self) -> list:
        return list(getattr(self.session, "pieces", None) or [])

    @property
    def books(self) -> list:
        try:
            return list(self.session.books or [])
        except Exception:
            return []

    def load(self, *, force: bool = False) -> list:
        """The dialog's refresh scan: generate missing / stale sidecars, reload when changed."""
        self.session.refresh(force=force)
        return self.pieces

    def changed_on_disk(self) -> bool:
        return bool(self.session.changed_on_disk())

    def focus_index(self) -> int:
        row = getattr(self.session, "_displayed_review_row", -1)
        pieces = self.pieces
        if isinstance(row, int) and 0 <= row < len(pieces):
            return row
        name = str(self.focus or "")
        for index, piece in enumerate(pieces):
            if name and name in (piece.get("output_name"), piece.get("original_name"), piece.get("name")):
                return index
        return 0

    def label(self, index: int) -> str:
        try:
            return str(self.session.piece_summary(index).get("label") or "")
        except Exception:
            piece = self.pieces[index]
            return str(piece.get("review_label") or piece.get("output_name") or piece.get("name") or index + 1)

    def switch_book(self, index: int) -> bool:
        return bool(self.session.switch_book(index))

    def select(self, index: int) -> None:
        try:
            self.session.select_piece(index)
        except Exception:
            pass

    # ---- edits / completion ------------------------------------------------------------------------

    def save_row(self, piece_index: int, row_index: int, text: str) -> str:
        return str(self.session.save_row(piece_index, row_index, text) or "")

    def notepad_document(self, piece_index: int) -> str:
        return str(self.session.notepad_document(piece_index) or "")

    def save_document(self, piece_index: int, html_text: str) -> str:
        self.session.edit_document(piece_index, html_text)
        return str(self.session.flush_edits() or "")

    def mark_completed(self, piece_index: int) -> str:
        return str(self.session.mark_completed([piece_index]) or "")

    def undo_completed(self, piece_index: int) -> str:
        return str(self.session.undo_completed([piece_index]) or "")

    def output_path(self, piece_index: int) -> str:
        return str(self.session.output_path(piece_index) or "")

    # ---- machine translation / flag ------------------------------------------------------------------

    def provider(self) -> str:
        try:
            return str(self.session.provider or "auto")
        except Exception:
            return "auto"

    def provider_labels(self) -> dict:
        """The dialog's ``MACHINE_TRANSLATION_PROVIDER_LABELS`` (provider -> menu label)."""
        labels = getattr(self.session, "MACHINE_TRANSLATION_PROVIDER_LABELS", None)
        return dict(labels) if isinstance(labels, Mapping) and labels else dict(PROVIDERS)

    def set_credentials(self, provider: str, values: Mapping[str, Any]) -> bool:
        return bool(self.session.set_machine_translation_credentials(provider, **dict(values)))

    def set_provider(self, provider: str) -> str:
        return str(self.session.set_provider(provider) or "")

    def translate_piece(self, piece_index: int) -> str:
        result = self.session.machine_translation_preview(piece_index) or {}
        return str(result.get("message") or result.get("error") or "")

    def inject(self, piece_index: int, row_index: int) -> str:
        return str(self.session.inject_machine_translation(piece_index, row_index) or "")

    def flag_inaccurate(self, piece_index: int) -> str:
        return str(self.session.flag_inaccurate(piece_index) or "")

    def threshold(self) -> float:
        return float(self.session.inaccuracy_threshold)

    def default_threshold(self) -> float:
        return float(getattr(self.session, "MACHINE_TRANSLATION_INACCURACY_THRESHOLD", 150.0))

    def set_threshold(self, value: Any) -> float:
        return float(self.session.set_inaccuracy_threshold(value))

    def reset_threshold(self) -> float:
        return float(self.session.reset_inaccuracy_threshold())


class SdlxliffScreen(Screen):
    title = "SDLXLIFF reviewer"

    def __init__(self, match: Optional[RouteMatch], ctx: Any) -> None:
        super().__init__(match)
        self.ctx = ctx
        query = match.query if match is not None else {}
        prefs = getattr(ctx, "prefs", None)
        fid = query.get("out") if query else None
        self.folder = (prefs.resolve_file_ref(fid) if prefs is not None and fid else None) or ""
        self.request = take_request(self.folder) if self.folder else ReviewerRequest()
        self.binding: Optional[ReviewerBinding] = None
        self.error: Optional[str] = None if self.folder else (
            "Choose a workspace with SDLXLIFF sidecars, or open the reviewer from a book's Chapters tab")
        self.piece_index = 0
        self.filter_status: Optional[str] = None
        self.notepad = False
        self.row_fields: dict = {}  # row index -> its output TextField (the rows shown now)
        self.row_base: dict = {}  # row index -> the text its field was built with / last saved
        self.row_cards: dict = {}  # row index -> its card in ``rows_column``
        self.rows_piece_index = 0  # the piece the row cards show
        self._row_gen = 0  # row card keys are unique per build: a rebuilt card is a new control
        self._focused_row: Optional[int] = None
        self.notepad_editor: Any = None
        self.notepad_base: Optional[str] = None  # the document the Notepad editor was loaded with
        self.notepad_piece = 0  # the piece that document belongs to
        self._notepad_token: Any = None
        self._render_pending = False  # a polled refresh came in during an edit: re-render after it
        self._poll_task: Any = None
        self._visible = False
        # tablets (UI_SPEC §4.9): the books and pieces as a persistent side list (MasterDetail)
        self.side_by_side = bool(getattr(ctx, "tablet", False))
        self.md: Any = None
        self.root: Optional[ft.Container] = None
        self.side_list: Optional[ft.ListView] = None
        self._layout_gen = 0

    # ---- layout -----------------------------------------------------------------------------------

    def actions(self) -> list:
        return [ft.IconButton(icon=ft.Icons.REFRESH, tooltip="↻ Refresh", key="sdl-refresh",
                              on_click=lambda e: self.ctx.spawn(self.reload(force=True)), size_constraints=HIT_TARGET)]

    def build_body(self) -> ft.Control:
        self.book_dropdown = ft.Dropdown(label="Book", options=[], dense=True, visible=False,
                                         on_select=self._on_book_select, key="sdl-book")
        self.piece_dropdown = ft.Dropdown(label="Piece", options=[], dense=True, expand=True,
                                          on_select=self._on_piece_select, key="sdl-piece")
        self.prev_button = ft.IconButton(icon=ft.Icons.CHEVRON_LEFT, tooltip="Previous piece",
                                         on_click=lambda e: self.step(-1), key="sdl-prev", size_constraints=HIT_TARGET)
        self.next_button = ft.IconButton(icon=ft.Icons.CHEVRON_RIGHT, tooltip="Next piece",
                                         on_click=lambda e: self.step(1), key="sdl-next", size_constraints=HIT_TARGET)
        self.piece_menu = ft.PopupMenuButton(icon=ft.Icons.MORE_VERT, tooltip="Piece actions", key="sdl-piece-menu",
                                             items=self._piece_menu_items())
        self.legend = ft.Row(scroll=ft.ScrollMode.AUTO, spacing=6, key="sdl-legend")
        self.summary = ft.Text("", theme_style=ft.TextThemeStyle.BODY_SMALL, key="sdl-summary")
        self.status_text = ft.Text("", theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.PRIMARY,
                                   key="sdl-status")
        self.mt_button = ft.FilledTonalButton(content="🌐 Generate Machine Translation Preview", key="sdl-mt",
                                              on_click=lambda e: self.ctx.spawn(self.translate_piece()))
        self.mt_provider = ft.TextButton(content="Provider: Auto", on_click=lambda e: self.open_provider_sheet(),
                                         key="sdl-provider")
        self.flag_button = ft.FilledTonalButton(content="🟣 Flag Inaccurate", key="sdl-flag",
                                                on_click=lambda e: self.ctx.spawn(self.flag()))
        self.threshold_button = ft.TextButton(content="Threshold", key="sdl-threshold",
                                              on_click=lambda e: self.open_threshold_sheet())
        tablet = bool(getattr(self.ctx, "tablet", False))
        self.layout_switch = ft.Switch(label="Notepad layout", value=False, disabled=not tablet,
                                       on_change=self._on_layout, key="sdl-layout")
        layout_row: list = [self.layout_switch]
        if not tablet:
            layout_row.append(ReasonChip(reason=NOTEPAD_REASON, detail="The Notepad layout edits the whole output "
                                                                       "document and needs a tablet-sized screen."))
        self.rows_column = ft.Column(spacing=8, key="sdl-rows")
        self.body_holder = ft.Container(content=self.rows_column, key="sdl-holder")
        header = ft.Column([
            self.book_dropdown,
            ft.Row([self.prev_button, self.piece_dropdown, self.next_button, self.piece_menu],
                   vertical_alignment=ft.CrossAxisAlignment.CENTER),
            self.summary, self.legend,
            ft.Row([self.mt_button, self.mt_provider, self.flag_button, self.threshold_button], wrap=True, spacing=6,
                   run_spacing=6),
            ft.Row(layout_row, wrap=True, spacing=6), self.status_text,
        ], spacing=6, tight=True)
        self.header_column = header
        self.list_view = ft.ListView(controls=[header, self.body_holder], expand=True, spacing=tokens.SPACING["sm"],
                                     padding=ft.Padding.symmetric(horizontal=12, vertical=8), key="sdl-screen")
        if self.error:
            self.rows_column.controls = [hint_text(self.error, key="sdl-error")]
            if not self.folder:  # the Tools hub tile: choose a workspace (SourcePicker), like the Progress manager
                self.rows_column.controls.append(ft.FilledTonalButton(
                    content="Choose a workspace…", icon=ft.Icons.FOLDER_OPEN, key="sdl-choose",
                    on_click=lambda e: self.open_picker()))
        self.root = ft.Container(expand=True, key="sdl-root")
        self._layout()
        return self.root

    # ---- tablet: side list of books and pieces (UI_SPEC §4.9) --------------------------------------

    def _layout(self) -> None:
        """Phone: the reviewer list (Dropdown + ‹ › selectors). Tablet: MasterDetail with the books and
        pieces as a persistent side list and the reviewer as the detail. Every rebuild gets fresh
        wrapper keys (Flet 1.0.3 freezes a subtree re-rendered under the key it replaces)."""
        if self.root is None:
            return
        self._layout_gen += 1
        gen = self._layout_gen
        main = ft.Container(content=self.list_view, expand=True, key=f"sdl-main-{gen}")
        compact_selectors = not self.side_by_side
        for control in (self.piece_dropdown, self.prev_button, self.next_button):
            control.visible = compact_selectors
        if self.side_by_side:
            from glossarion_mobile.ui.components.master_detail import MasterDetail

            self.side_list = ft.ListView(spacing=2, expand=True, padding=ft.Padding.symmetric(vertical=8),
                                         key=f"sdl-side-{gen}")
            self._render_side_list()
            self.md = MasterDetail(self.side_list, placeholder=main, two_pane=True, master_width=320,
                                   key=f"sdl-md-{gen}")
            self.root.content = self.md.control
        else:
            self.md = None
            self.side_list = None
            self.root.content = main

    def _piece_color(self, index: int, piece: Mapping[str, Any]) -> str:
        summary: Mapping[str, Any] = {}
        session = getattr(self.binding, "session", None)
        try:
            summary = session.piece_summary(index) if session is not None else {}
        except Exception:
            summary = {}
        summary = summary or piece
        if summary.get("completed") or piece.get("manual_green_override"):
            return STATUS_COLORS["green"]
        for status in ("red", "yellow", "purple"):
            if int(summary.get(f"{status}_count") or 0):
                return STATUS_COLORS[status]
        return STATUS_COLORS["green"]

    def _render_side_list(self) -> None:
        if self.side_list is None:
            return
        binding = self.binding
        controls: list = []
        books = binding.books if binding is not None else []
        if len(books) > 1:
            controls.append(ft.Text("Books", theme_style=ft.TextThemeStyle.TITLE_SMALL, color=ft.Colors.PRIMARY))
            current = str(self.book_dropdown.value or "0")
            for i, book in enumerate(books):
                label = str(book.get("label") or os.path.basename(str(book.get("output_dir") or "")))
                controls.append(ft.ListTile(title=ft.Text(label, max_lines=2), dense=True, selected=str(i) == current,
                                            on_click=lambda e, i=i: self._pick_book(i),
                                            key=f"sdl-side-book-{self._layout_gen}-{i}"))
        pieces = binding.pieces if binding is not None else []
        if pieces:
            controls.append(ft.Text("Pieces", theme_style=ft.TextThemeStyle.TITLE_SMALL, color=ft.Colors.PRIMARY))
        for i, piece in enumerate(pieces):
            controls.append(ft.ListTile(
                leading=ft.Container(width=12, height=12, border_radius=6, bgcolor=self._piece_color(i, piece)),
                title=ft.Text(binding.label(i), max_lines=2, overflow=ft.TextOverflow.ELLIPSIS),
                dense=True, selected=i == self.piece_index, on_click=lambda e, i=i: self.select_piece(i),
                key=f"sdl-side-piece-{self._layout_gen}-{i}"))
        if not controls:
            controls.append(hint_text("No SDLXLIFF pieces yet", key=f"sdl-side-empty-{self._layout_gen}"))
        self.side_list.controls = controls
        self._push(self.side_list)

    def _pick_book(self, index: int) -> None:
        self.book_dropdown.value = str(index)
        self.ctx.spawn(self.switch_book(index))

    def select_piece(self, index: int) -> None:
        """A side-list tap (tablet): the piece the Dropdown would pick."""
        pieces = self.binding.pieces if self.binding is not None else []
        if not 0 <= index < len(pieces):
            return
        self.piece_index = index
        self._rerender()

    def apply_size_class(self, size_class: Any) -> None:
        tablet = bool(getattr(size_class, "persistent_sidebar", False))
        if tablet == self.side_by_side:
            return
        self.side_by_side = tablet
        self._layout()
        self._push(self.root)

    def open_picker(self) -> Any:
        """The SourcePicker (eligible: an output folder with SDLXLIFF sidecars), then the reviewer on it."""
        from glossarion_mobile.ui.tools.source_picker import SourcePicker

        def eligible(target: Any) -> Optional[str]:
            folder = str(getattr(target, "folder", "") or "")
            if not folder:
                return "Needs an output folder"
            return None if os.path.isdir(os.path.join(folder, "SDLXLIFF")) else "No SDLXLIFF sidecars in this workspace"

        def chosen(targets: list) -> None:
            if targets:
                target = targets[0]
                open_reviewer(self.ctx, target.folder, source=target.source or None)

        picker = SourcePicker(self.ctx, title="Review SDLXLIFF of…", multi=False, eligible=eligible, on_done=chosen,
                              segment="library", find_folder=True)
        if self.ctx.page is not None:
            picker.show(self.ctx.page)
        return picker

    def did_show(self) -> None:
        self._visible = True
        if self.folder:
            self.ctx.spawn(self.open())

    def dispose(self) -> None:
        self._visible = False
        task, self._poll_task = self._poll_task, None
        if task is not None:
            try:
                task.cancel()
            except Exception:
                pass
        self._save_on_leave()

    def _save_on_leave(self) -> Any:
        """The dialog's ``closeEvent``: the edited Notepad document and typed row texts are saved
        when the reviewer is left (Back, a drawer destination, ...), off the loop."""
        binding = self.binding
        if binding is None:
            return None
        rows = [(self.rows_piece_index, index, str(self.row_fields[index].value or "")) for index in self._dirty_rows()]
        document = (self.notepad_piece, str(self.notepad_editor.value or "")) if self._document_dirty() else None
        if not rows and document is None:
            return None

        async def save() -> None:
            try:
                for piece_index, row_index, text in rows:
                    await self.ctx.io(binding.save_row, piece_index, row_index, text)
                if document is not None:
                    await self.ctx.io(binding.save_document, *document)
            except Exception as exc:
                log.exception("saving the SDLXLIFF edits on leave failed")
                self.ctx.say(f"Could not save the SDLXLIFF edits: {exc}")

        return self.ctx.spawn(save())

    # ---- session -------------------------------------------------------------------------------------

    def _make_binding(self) -> ReviewerBinding:
        try:
            import sdlxliff_review_core as core
        except Exception:
            core = None
        config = self.ctx.config_snapshot() if hasattr(self.ctx, "config_snapshot") else {}
        return ReviewerBinding(core, self.folder, source=self.request.source, focus=self.request.focus,
                               config=config, save=self.ctx.set_cfg, progress_data=self.request.progress_data,
                               manual_entries=self.request.manual_entries if self.request.manual_editing else None)

    async def open(self) -> Optional[ReviewerBinding]:
        try:
            self.binding = await self.ctx.io(self._make_binding)
        except LookupError as exc:
            self.error = str(exc)
        except Exception as exc:
            log.exception("opening the SDLXLIFF review failed")
            self.error = f"The review could not be opened: {exc}"
        if self.binding is None:
            self.rows_column.controls = [hint_text(self.error or NO_CORE, key="sdl-error")]
            self._push(self.rows_column)
            return None
        await self.reload(focus=True)
        if self._poll_task is None:
            self._poll_task = self.ctx.spawn(self._poll())
        return self.binding

    async def reload(self, *, force: bool = False, focus: bool = False, polled: bool = False) -> list:
        """Refresh the session (``focus``: then show the piece the screen was opened for).

        ``polled``: the 2 s poll's silent refresh; when an edit started meanwhile, the re-render
        waits until it is saved (``_render_pending``)."""
        binding = self.binding
        if binding is None:
            return []
        await self._flush_edits()
        try:
            pieces = await self.ctx.io(lambda: binding.load(force=force))
        except Exception as exc:
            log.exception("refreshing the SDLXLIFF review failed")
            self.ctx.say(f"SDLXLIFF refresh failed: {exc}")
            pieces = binding.pieces
        if focus:
            self.piece_index = binding.focus_index()
        self.piece_index = max(0, min(self.piece_index, len(pieces) - 1)) if pieces else 0
        if polled and self._editing():
            self._render_pending = True
            return pieces
        self._render_pending = False
        self.render()
        return pieces

    def _shown(self) -> bool:
        """On top of the stack (not covered by e.g. its own Edit Output editor) and the app in the
        foreground: the Library / Book page poller's ``visible`` + ``foreground``."""
        try:
            is_top = getattr(self.ctx, "is_top", None)
            foreground = getattr(self.ctx, "foreground", None)
            return ((not callable(is_top) or bool(is_top(self)))
                    and (not callable(foreground) or bool(foreground())))
        except Exception:
            return True

    async def _poll(self) -> None:
        """2 s poll while visible (the dialog's silent auto refresh)."""
        while self._visible and self.binding is not None:
            await poll_sleep(getattr(self.ctx, "page", None), POLL_SECONDS)  # parks while the app is hidden
            if not self._visible or self.binding is None:
                return
            if not self._shown():
                continue
            if self._editing():
                continue  # a reload re-renders every field: wait until the edit is saved
            if self._render_pending:
                self._render_pending = False
                self.render()
                continue
            try:
                changed = await self.ctx.io(self.binding.changed_on_disk)
            except Exception:
                continue
            if changed and not self._editing():
                await self.reload(polled=True)

    # ---- rendering -------------------------------------------------------------------------------------

    def current_piece(self) -> Optional[dict]:
        pieces = self.binding.pieces if self.binding is not None else []
        if not pieces:
            return None
        self.piece_index = max(0, min(self.piece_index, len(pieces) - 1))
        return pieces[self.piece_index]

    def render(self) -> None:
        """Full re-render (header + rows or the Notepad document)."""
        piece = self._render_header()
        binding = self.binding
        self.row_fields, self.row_base, self.row_cards = {}, {}, {}
        self._focused_row = None
        self.notepad_editor, self.notepad_base = None, None
        self._notepad_token = None
        if self.notepad and piece is not None and binding is not None:
            # the document is built from the output HTML (BeautifulSoup): on the io pool
            self.body_holder.content = ft.ProgressRing(width=24, height=24, key="sdl-notepad-loading")
            self._notepad_token = token = object()
            self.ctx.spawn(self._load_notepad(token, self.piece_index))
        else:
            self.rows_piece_index = self.piece_index
            self.rows_column.controls = self._row_cards(piece)
            self.body_holder.content = self.rows_column
        self._render_side_list()
        self._push(self.list_view)

    def _render_header(self) -> Optional[dict]:
        """Book / piece selectors, legend counts, summary, provider and status; returns the piece."""
        binding = self.binding
        pieces = binding.pieces if binding is not None else []
        books = binding.books if binding is not None else []
        self.book_dropdown.visible = len(books) > 1
        self.book_dropdown.options = [
            ft.DropdownOption(key=str(i), text=str(b.get("label") or os.path.basename(str(b.get("output_dir") or ""))))
            for i, b in enumerate(books)]
        self.piece_dropdown.options = [ft.DropdownOption(key=str(i), text=binding.label(i)) for i in range(len(pieces))]
        self.piece_dropdown.value = str(self.piece_index) if pieces else None
        self.prev_button.disabled = self.next_button.disabled = len(pieces) <= 1
        piece = self.current_piece()
        if binding is not None and piece is not None:
            binding.select(self.piece_index)
        counts: dict = {}
        for row in (piece or {}).get("rows") or ():
            status = str(row.get("status") or "green")
            counts[status] = counts.get(status, 0) + 1
        self.legend.controls = [
            ft.Chip(label=ft.Text(f"{text} ({counts.get(status, 0)})"), selected=self.filter_status == status,
                    show_checkmark=False, key=f"sdl-legend-{status}",
                    leading=ft.Container(width=10, height=10, bgcolor=STATUS_COLORS[status], border_radius=5),
                    on_select=lambda e, s=status: self.set_filter(None if self.filter_status == s else s))
            for status, text in LEGEND]
        if piece is None:
            self.summary.value = "No SDLXLIFF sidecars yet" if binding is not None else ""
        else:
            completed = " · ✅ Completed" if piece.get("manual_green_override") else ""
            manual = " · Manual editing" if piece.get("manual_editing") else ""
            self.summary.value = (f"{piece.get('source_count', 0)} source · {piece.get('target_count', 0)} output"
                                  f"{manual}{completed}")
        provider = binding.provider() if binding is not None else "auto"
        labels = binding.provider_labels() if binding is not None else dict(PROVIDERS)
        self.mt_provider.content = f"Provider: {labels.get(provider, 'Auto')}"
        self.status_text.value = binding.status if binding is not None else ""
        self.piece_menu.items = self._piece_menu_items()
        return piece

    async def _load_notepad(self, token: Any, piece_index: int) -> Optional[str]:
        binding = self.binding
        error = ""
        try:
            text: Optional[str] = await self.ctx.io(binding.notepad_document, piece_index)
        except Exception as exc:
            log.exception("building the Notepad document failed")
            text, error = None, str(exc)
        if token is not self._notepad_token or not self.notepad:
            return None  # another render replaced this one meanwhile
        if text is None:
            self.body_holder.content = hint_text(f"The output document could not be read: {error}",
                                                 key="sdl-notepad-error")
        else:
            self.notepad_base = text
            self.notepad_piece = piece_index
            self.body_holder.content = self._notepad(text)
        self._push(self.body_holder)
        return text

    def _row_cards(self, piece: Optional[dict]) -> list:
        if piece is None:
            return [hint_text("Generate SDLXLIFF sidecars from the Chapters tab (Manual editing) or ↻ Refresh.",
                              key="sdl-empty")]
        cards = []
        for index, row in enumerate(piece.get("rows") or ()):
            status = str(row.get("status") or "green")
            if self.filter_status and status != self.filter_status:
                continue
            cards.append(self._row_card(index, row, status))
        return cards or [hint_text("No rows with that status", key="sdl-no-rows")]

    def _row_card(self, index: int, row: Mapping[str, Any], status: str) -> ft.Control:
        self._row_gen += 1
        gen = self._row_gen
        target = str(row.get("target") or "")
        piece_index = self.rows_piece_index
        field_control = ft.TextField(value=target, multiline=True, min_lines=1, dense=True,
                                     label=str(row.get("target_tag_label") or "Output"),
                                     on_focus=lambda e, i=index: self._set_focused(i, True),
                                     key=f"sdl-target-{index}-{gen}")
        # the blur saves the text of this field into this piece, whatever was re-rendered meanwhile
        field_control.on_blur = lambda e, i=index, f=field_control, p=piece_index: (
            self._set_focused(i, False), self.ctx.spawn(self.save_row(i, f, p)))
        self.row_fields[index] = field_control
        self.row_base[index] = target
        parts: list = [
            ft.Text(f"{row.get('source_tag_label') or 'Source'} · {row.get('reason') or status}",
                    theme_style=ft.TextThemeStyle.LABEL_SMALL, color=STATUS_COLORS.get(status)),
            ft.Text(str(row.get("source") or ""), theme_style=ft.TextThemeStyle.BODY_SMALL,
                    color=ft.Colors.ON_SURFACE_VARIANT, selectable=True),
            field_control,
        ]
        preview = str(row.get("tooltip_translation_error") or row.get("tooltip_translation") or "")
        if row.get("tooltip_translation_pending"):
            preview = str(row.get("tooltip_translation_status") or "⏳ Generating machine translation preview...")
        if preview:
            parts.append(ft.Row([
                ft.Text(f"MT: {preview}", theme_style=ft.TextThemeStyle.BODY_SMALL, expand=True, selectable=True),
                ft.TextButton(content="Inject MT", key=f"sdl-inject-{index}",
                              disabled=not row.get("tooltip_translation") or bool(row.get("tooltip_translation_pending")),
                              on_click=lambda e, i=index: self.ctx.spawn(self.inject(i))),
            ], vertical_alignment=ft.CrossAxisAlignment.CENTER))
        card = ft.Container(
            content=ft.Row([
                ft.Container(width=4, bgcolor=STATUS_COLORS.get(status, STATUS_COLORS["green"]), border_radius=2),
                ft.Column(parts, spacing=4, expand=True, tight=True),
            ], vertical_alignment=ft.CrossAxisAlignment.STRETCH, spacing=8),
            padding=ft.Padding.symmetric(horizontal=8, vertical=8),
            bgcolor=ft.Colors.SURFACE_CONTAINER_LOW,
            border_radius=tokens.RADII["card"],
            key=f"sdl-row-{index}-{gen}",
        )
        self.row_cards[index] = card
        return card

    def _set_focused(self, index: int, focused: bool) -> None:
        if focused:
            self._focused_row = index
        elif self._focused_row == index:
            self._focused_row = None

    def _dirty_rows(self) -> list:
        """Row indices whose field text differs from what it was built with / last saved."""
        return [i for i, f in self.row_fields.items() if str(f.value or "") != self.row_base.get(i, "")]

    def _document_dirty(self) -> bool:
        """Notepad: the document in the editor differs from the one it was loaded with / last saved."""
        editor = self.notepad_editor
        return bool(self.notepad and editor is not None and self.notepad_base is not None
                    and str(editor.value or "") != self.notepad_base)

    def _editing(self) -> bool:
        """A field has focus or unsaved text (Notepad: the document was edited)."""
        if self._focused_row is not None or self._dirty_rows():
            return True
        return self._document_dirty()

    async def _flush_rows(self) -> None:
        """Save the typed row texts a full re-render would drop."""
        piece_index = self.rows_piece_index
        for index in self._dirty_rows():
            field_control = self.row_fields.get(index)
            if field_control is not None:
                await self.save_row(index, field_control, piece_index, refresh=False)

    async def _flush_edits(self) -> None:
        """Save everything a full re-render would drop: typed row texts and an edited Notepad document."""
        await self._flush_rows()
        if self._document_dirty() and self.binding is not None:
            text = str(self.notepad_editor.value or "")
            try:
                await self.ctx.io(self.binding.save_document, self.notepad_piece, text)
            except Exception as exc:
                log.exception("saving the Notepad document failed")
                self.ctx.say(f"Could not save: {exc}")
                return
            self.notepad_base = text

    def _rerender(self) -> None:
        """A full re-render after a navigation; typed row text / the edited document is saved first."""
        if self._dirty_rows() or self._document_dirty():
            self.ctx.spawn(self._flush_then_render())
        else:
            self.render()

    async def _flush_then_render(self) -> None:
        await self._flush_edits()
        self.render()

    def _notepad(self, text: str) -> ft.Control:
        try:
            import flet_code_editor as fce

            self.notepad_editor = fce.CodeEditor(value=text, language=getattr(fce.CodeLanguage, "XML", None),
                                                 expand=True)
        except Exception:
            self.notepad_editor = ft.TextField(value=text, multiline=True, min_lines=20, key="sdl-notepad",
                                               text_style=ft.TextStyle(font_family=getattr(self.ctx, "mono", None),
                                                                       size=13))
        return ft.Column([
            self.notepad_editor,
            ft.Row([ft.FilledButton(content="Save document", key="sdl-notepad-save",
                                    on_click=lambda e: self.ctx.spawn(self.save_document()))]),
        ], spacing=6)

    def _piece_menu_items(self) -> list:
        piece = self.current_piece() if self.binding is not None else None
        completed = bool(piece and piece.get("manual_green_override"))
        return [
            ft.PopupMenuItem(content="Undo Completed" if completed else "Mark as Completed", key="sdl-complete",
                             on_click=lambda e: self.ctx.spawn(self.toggle_completed())),
            ft.PopupMenuItem(content="Edit Output", key="sdl-edit-output", on_click=lambda e: self.edit_output()),
        ]

    # ---- navigation / filters -------------------------------------------------------------------------

    def _on_piece_select(self, e: Any = None) -> None:
        try:
            self.piece_index = int(self.piece_dropdown.value or 0)
        except (TypeError, ValueError):
            return
        self._rerender()

    def _on_book_select(self, e: Any = None) -> None:
        try:
            index = int(self.book_dropdown.value or 0)
        except (TypeError, ValueError):
            return
        self.ctx.spawn(self.switch_book(index))

    async def switch_book(self, index: int) -> bool:
        if self.binding is None:
            return False
        await self._flush_edits()
        switched = await self.ctx.io(self.binding.switch_book, index)
        if switched:
            self.piece_index = 0
            await self.reload()
        return bool(switched)

    def step(self, offset: int) -> None:
        pieces = self.binding.pieces if self.binding is not None else []
        if not pieces:
            return
        self.piece_index = (self.piece_index + offset) % len(pieces)
        self._rerender()

    def set_filter(self, status: Optional[str]) -> None:
        self.filter_status = status
        self._rerender()

    def _on_layout(self, e: Any = None) -> None:
        if not getattr(self.ctx, "tablet", False):
            self.layout_switch.value = False
            self.ctx.say(NOTEPAD_REASON)
            return
        self.notepad = bool(self.layout_switch.value)
        self._rerender()

    # ---- actions --------------------------------------------------------------------------------------

    def _done(self, message: str) -> str:
        if message:
            self.ctx.say(message)
        self.render()
        return message

    async def save_row(self, row_index: int, field_control: Any = None, piece_index: Optional[int] = None,
                       *, refresh: bool = True) -> bool:
        """Save one row's output (on blur). ``field_control`` / ``piece_index``: the field and piece the
        edit was made in (default: the rows shown now). Only that row's card is rebuilt afterwards."""
        binding = self.binding
        field_control = field_control if field_control is not None else self.row_fields.get(row_index)
        piece_index = self.rows_piece_index if piece_index is None else piece_index
        pieces = binding.pieces if binding is not None else []
        if field_control is None or not 0 <= piece_index < len(pieces):
            return False
        rows = pieces[piece_index].get("rows") or []
        text = str(field_control.value or "")
        if row_index >= len(rows) or text == str(rows[row_index].get("target") or ""):
            if self.row_fields.get(row_index) is field_control:
                self.row_base[row_index] = text
            return False
        try:
            await self.ctx.io(binding.save_row, piece_index, row_index, text)
        except Exception as exc:
            log.exception("saving the SDLXLIFF edit failed")
            self.ctx.say(f"Could not save: {exc}")
            return False
        if self.row_fields.get(row_index) is field_control:
            self.row_base[row_index] = text
        if refresh:
            self._row_saved(piece_index, row_index)
        return True

    def _row_saved(self, piece_index: int, row_index: int) -> None:
        """After a row save: the header counts and that row's card only (status bar, reason, text).
        Every other field keeps its control (and so its focus and typed text)."""
        if self.notepad or piece_index != self.rows_piece_index or self.binding is None:
            return
        self._render_header()
        pieces = self.binding.pieces
        rows = (pieces[piece_index].get("rows") or []) if piece_index < len(pieces) else []
        card = self.row_cards.get(row_index)
        controls = self.rows_column.controls
        position = next((i for i, c in enumerate(controls) if c is card), None)
        if row_index < len(rows) and position is not None and self._focused_row != row_index:
            row = rows[row_index]
            controls[position] = self._row_card(row_index, row, str(row.get("status") or "green"))
        self._push(self.legend, self.summary, self.status_text, self.piece_dropdown, self.rows_column)

    async def save_document(self) -> bool:
        binding, editor = self.binding, self.notepad_editor
        if binding is None or editor is None:
            return False
        try:
            message = await self.ctx.io(binding.save_document, self.notepad_piece, str(editor.value or ""))
        except Exception as exc:
            self.ctx.say(f"Could not save: {exc}")
            return False
        self._done(message or "Saved")
        return True

    async def toggle_completed(self) -> str:
        binding, piece = self.binding, self.current_piece()
        if binding is None or piece is None:
            return ""
        await self._flush_edits()
        fn = binding.undo_completed if piece.get("manual_green_override") else binding.mark_completed
        return self._done(await self.ctx.io(fn, self.piece_index))

    def edit_output(self) -> Optional[str]:
        from glossarion_mobile.ui.tools import text_editor

        if self.binding is None or self.current_piece() is None:
            return None
        path = self.binding.output_path(self.piece_index)
        if not path or not os.path.isfile(path):
            self.ctx.say("This piece has no output file yet")
            return None
        return text_editor.open_text_editor(self.ctx, path)

    async def translate_piece(self) -> str:
        if self.binding is None or self.current_piece() is None:
            return ""
        await self._flush_edits()
        self.status_text.value = "⏳ Generating machine translation preview..."
        self._push(self.status_text)
        try:
            message = await self.ctx.io(self.binding.translate_piece, self.piece_index)
        except Exception as exc:
            message = f"Machine translation preview failed: {exc}"
        return self._done(message)

    async def inject(self, row_index: int) -> str:
        if self.binding is None:
            return ""
        await self._flush_edits()
        return self._done(await self.ctx.io(self.binding.inject, self.piece_index, row_index))

    async def flag(self) -> str:
        if self.binding is None or self.current_piece() is None:
            return ""
        await self._flush_edits()
        return self._done(await self.ctx.io(self.binding.flag_inaccurate, self.piece_index))

    # ---- sheets ---------------------------------------------------------------------------------------

    def open_provider_sheet(self) -> Any:
        from glossarion_mobile.ui.components.action_sheet import ActionItem, ActionSheet

        current = self.binding.provider() if self.binding is not None else "auto"
        items = []
        labels = self.binding.provider_labels() if self.binding is not None else dict(PROVIDERS)
        for provider, label in labels.items():
            items.append(ActionItem(("✓ " if provider == current else "") + label,
                                    (lambda p=provider: self.ctx.spawn(self.choose_provider(p))),
                                    key=f"sdl-provider-{provider}",
                                    disabled_reason=ARGOS_REASON if provider == "argos" else None))
        for provider, label in (("deepl", "🔑 Configure DeepL API Key..."), ("bing", "🔑 Configure Bing API Key..."),
                                ("yandex", "🔑 Configure Yandex API Key...")):
            items.append(ActionItem(label, (lambda p=provider: self.ctx.spawn(self.configure(p))),
                                    key=f"sdl-configure-{provider}"))
        sheet = ActionSheet(items, title="Machine translation provider", tablet=bool(getattr(self.ctx, "tablet", False)))
        if self.ctx.page is not None:
            sheet.show(self.ctx.page)
        return sheet

    async def _credentials(self, provider: str) -> Optional[dict]:
        """The dialog's credential prompts for ``provider`` in one sheet (None: cancelled)."""
        fields = CREDENTIAL_FIELDS.get(provider)
        if not fields:
            return {}
        scripted = self.ctx.extras.get("credentials") if hasattr(self.ctx, "extras") else None
        if isinstance(scripted, list):
            return scripted.pop(0) if scripted else None
        if self.ctx.page is None:
            return None
        from glossarion_mobile.ui.components.dialogs import ConfirmDialog

        controls = {name: ft.TextField(label=label, password=secret, can_reveal_password=secret, dense=True,
                                       key=f"sdl-cred-{name}") for name, label, secret in fields}
        loop = asyncio.get_running_loop()
        answer: asyncio.Future = loop.create_future()
        labels = self.binding.provider_labels() if self.binding is not None else dict(PROVIDERS)
        dialog = ConfirmDialog(title=f"{labels.get(provider, provider)} credentials", confirm_label="Save",
                               on_confirm=lambda: answer.done() or answer.set_result(
                                   {name: str(controls[name].value or "").strip() for name in controls}),
                               on_cancel=lambda: answer.done() or answer.set_result(None))
        for control in reversed(list(controls.values())):
            dialog.dialog.content.content.controls.insert(0, control)
        dialog.show(self.ctx.page)
        return await answer

    async def choose_provider(self, provider: str) -> str:
        """The dialog's provider change: a key-based provider asks for its credentials first."""
        if self.binding is None:
            return "auto"
        if provider == "argos":
            self.ctx.say(ARGOS_REASON)
            return self.binding.provider()
        if provider in CREDENTIAL_FIELDS:
            values = await self._credentials(provider)
            if values:
                await self.ctx.io(self.binding.set_credentials, provider, values)
        self._done(await self.ctx.io(self.binding.set_provider, provider))
        return self.binding.provider()

    async def configure(self, provider: str) -> bool:
        """"🔑 Configure … API Key…" (the dialog's ``force=True`` prompts)."""
        if self.binding is None:
            return False
        values = await self._credentials(provider)
        if values is None:
            return False
        ok = await self.ctx.io(self.binding.set_credentials, provider, values)
        self.ctx.say(self.binding.status or ("Saved" if ok else "Not saved"))
        return bool(ok)

    def open_threshold_sheet(self) -> Any:
        from glossarion_mobile.ui.components.action_sheet import ActionItem, ActionSheet

        if self.binding is None:
            return None
        current = self.binding.threshold()
        default = self.binding.default_threshold()
        sheet = ActionSheet([
            ActionItem(f"🟣 Set Score Threshold... ({current:g})", lambda: self.ctx.spawn(self.ask_threshold()),
                       key="sdl-threshold-set"),
            ActionItem(f"↺ Reset Threshold ({default:g})", lambda: self.ctx.spawn(self.reset_threshold()),
                       key="sdl-threshold-reset"),
        ], title="Flag Inaccurate Threshold", tablet=bool(getattr(self.ctx, "tablet", False)))
        if self.ctx.page is not None:
            sheet.show(self.ctx.page)
        return sheet

    async def ask_threshold(self) -> Optional[float]:
        scripted = self.ctx.extras.get("threshold") if hasattr(self.ctx, "extras") else None
        if scripted is not None:
            return await self.set_threshold(scripted)
        if self.ctx.page is None or self.binding is None:
            return None
        from glossarion_mobile.ui.components.dialogs import ConfirmDialog

        field_control = ft.TextField(label="Score threshold (lower flags more rows, higher flags fewer):",
                                     value=f"{self.binding.threshold():g}", keyboard_type=ft.KeyboardType.NUMBER,
                                     key="sdl-threshold-field")
        loop = asyncio.get_running_loop()
        answer: asyncio.Future = loop.create_future()
        dialog = ConfirmDialog(title="Flag Inaccurate Threshold", confirm_label="Set",
                               on_confirm=lambda: answer.done() or answer.set_result(field_control.value),
                               on_cancel=lambda: answer.done() or answer.set_result(None))
        dialog.dialog.content.content.controls.insert(0, field_control)
        dialog.show(self.ctx.page)
        value = await answer
        return None if value is None else await self.set_threshold(value)

    async def set_threshold(self, value: Any) -> Optional[float]:
        if self.binding is None:
            return None
        try:
            threshold = await self.ctx.io(self.binding.set_threshold, value)
        except Exception as exc:
            self.ctx.say(f"Could not set the threshold: {exc}")
            return None
        self._done(self.binding.status)
        return threshold

    async def reset_threshold(self) -> Optional[float]:
        if self.binding is None:
            return None
        threshold = await self.ctx.io(self.binding.reset_threshold)
        self._done(self.binding.status)
        return threshold

    def _push(self, *controls: Any) -> None:
        for control in controls:
            if control is None:
                continue
            try:
                control.update()
            except Exception:
                pass
