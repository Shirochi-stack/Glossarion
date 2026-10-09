"""Manga › Editor (UI_SPEC §4.6 Editor, §5.10 ``MangaEditor``).

The screen of ``manga_editor_core.MangaEditorSession`` (ImageRenderer's GUI-free editor, the
same code the desktop preview runs):

* **Page strip** of the Files selection (skipped pages dimmed with ⏭️; long-press a thumbnail for
  the desktop thumbnail menu's "⏭️ Skip Processing" / "▶️ Process This Image");
  **Source / Translated** (side by side on tablets).
* **Pan** mode: ``InteractiveViewer`` (pinch zoom, pan). **Edit** modes: ``Stack(Image,
  canvas.Canvas, GestureDetector(drag_interval=24))`` with Select/Move (drag a box, drag its
  bottom-right corner to resize), Box, Circle and Lasso; Delete and Exclude from Clean act on
  the selected box. Brush / Eraser stay visible, disabled (no shared mask editor yet).
* **Workflow buttons** Detect · Clean · Recognize · Translate · Translate all · Save & Update
  Overlay, and the per-box OCR / Translate / Clean actions — ``manga_step`` jobs on the
  JobService: OCR and translation need the job's HeadlessOwner (plan §1), which only exists on
  the job thread. Box edits (add, move, resize, delete, texts, flags) are session calls on the
  io pool; editing waits while a step runs (the session runs one step at a time). A step that
  loads a model not on the device yet (detector / ONNX inpainter) offers its download first.
* **Long-press a box** → ``BoxSheet``. Import OCR (renders imported translations: a job) /
  Export OCR (``manga_ocr_io`` JSON in the OCR Text folder) / the auto-saved OCR files.
* The editor rewrites the same output file on every render, so the Translated view shows a
  versioned copy (``page_<id>_v3_<mtime>.png``): Flet's ``Image`` cache never shows an old render.

Box geometry is in image pixels (the session's scene coordinates); the canvas maps it to the
displayed size (``scale = displayed width / image width``).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import flet as ft
import flet.canvas as cv

from glossarion_mobile.services import manga as svc
from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.components.reason_chip import ReasonChip
from glossarion_mobile.ui.theme import HIT_TARGET
from glossarion_mobile.ui.components.empty_state import faded_mascot
from glossarion_mobile.ui.tools.common import JobWatch, hint_text
from glossarion_mobile.ui.tools.manga.box_sheet import BoxSheet
from glossarion_mobile.ui.tools.manga.common import JobEnds, MangaTab, export_sheet, push
from glossarion_mobile.ui.tools.manga.models import ensure_models

__all__ = ["EditorTab", "Geometry", "TOOLS", "hit_test", "lasso_bounds", "normalize_rect"]

log = logging.getLogger("glossarion.tools.manga")

MASK_REASON = "Brush and eraser are desktop only (experimental mask painting there; not planned for mobile)"
#: (tool id, label, icon, disabled reason)
TOOLS = (
    ("pan", "Pan / zoom", "PAN_TOOL", None),
    ("select", "Select / move", "OPEN_WITH", None),
    ("box", "Box", "CROP_SQUARE", None),
    ("circle", "Circle", "CIRCLE_OUTLINED", None),
    ("lasso", "Lasso", "GESTURE", None),
    ("brush", "Brush", "BRUSH", MASK_REASON),
    ("eraser", "Eraser", "AUTO_FIX_OFF", MASK_REASON),
)
STEP_BUTTONS = (("detect", "Detect", "CENTER_FOCUS_STRONG"), ("clean", "Clean", "CLEANING_SERVICES"),
                ("recognize", "Recognize", "DOCUMENT_SCANNER"), ("translate", "Translate", "TRANSLATE"),
                ("translate_all", "Translate all", "LIBRARY_BOOKS"), ("render", "Update overlay", "AUTO_FIX_NORMAL"))
HANDLE = 24.0  # dp: the resize handle
MIN_BOX = 8.0  # image px
DRAG_INTERVAL = 24
BOX_COLOR = "#E18F98"
SELECTED_COLOR = "#5A9FD4"
EXCLUDED_COLOR = "#9E9E9E"
SHAPES = {"box": "rect", "circle": "ellipse", "lasso": "polygon"}


# ---------------------------------------------------------------------------
# Geometry (pure; image pixels <-> displayed dp)
# ---------------------------------------------------------------------------


@dataclass
class Geometry:
    image_w: float = 0.0
    image_h: float = 0.0
    display_w: float = 0.0

    @property
    def scale(self) -> float:
        if not self.image_w or not self.display_w:
            return 1.0
        return self.display_w / self.image_w

    def to_image(self, x: float, y: float) -> tuple:
        s = self.scale or 1.0
        ix, iy = x / s, y / s
        if self.image_w:
            ix = max(0.0, min(self.image_w, ix))
        if self.image_h:
            iy = max(0.0, min(self.image_h, iy))
        return (ix, iy)


def normalize_rect(x1: float, y1: float, x2: float, y2: float) -> tuple:
    """(x, y, w, h) of two corners in any order."""
    return (min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))


def _contains(box: dict, x: float, y: float) -> bool:
    return (float(box.get("x", 0)) <= x <= float(box.get("x", 0)) + float(box.get("width", 0))
            and float(box.get("y", 0)) <= y <= float(box.get("y", 0)) + float(box.get("height", 0)))


def hit_test(boxes: Sequence[dict], x: float, y: float) -> Optional[int]:
    """The topmost (last drawn) box containing image point (x, y)."""
    for index in range(len(boxes) - 1, -1, -1):
        if _contains(boxes[index], x, y):
            return index
    return None


def on_resize_handle(box: dict, x: float, y: float, tolerance: float) -> bool:
    """Image point (x, y) is on ``box``'s bottom-right corner handle."""
    right = float(box.get("x", 0)) + float(box.get("width", 0))
    bottom = float(box.get("y", 0)) + float(box.get("height", 0))
    return abs(x - right) <= tolerance and abs(y - bottom) <= tolerance


def lasso_bounds(points: Sequence[tuple]) -> Optional[tuple]:
    """((x, y, w, h), polygon) of a lasso stroke in image px; None for a stroke too small to keep."""
    if len(points) < 3:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x, y, w, h = min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)
    if w < MIN_BOX or h < MIN_BOX:
        return None
    return (x, y, w, h), [[round(px, 1), round(py, 1)] for px, py in points]


def image_size(path: str) -> tuple:
    """Blocking: (width, height) of an image file, (0, 0) when unreadable.

    The chat's header reader (``media_model.image_size``: ``safe_image.open_image``, the owner's
    restricted Pillow parsers + pixel-bomb guard for untrusted images, cached per file)."""
    try:
        from glossarion_mobile.ui.chat.media_model import image_size as header_size

        return tuple(header_size(path) or (0, 0))
    except Exception:
        return (0, 0)


# ---------------------------------------------------------------------------
# The tab
# ---------------------------------------------------------------------------


class EditorTab(MangaTab):
    key = "manga-editor"

    def __init__(self, ctx: Any, session: Any, *, screen: Any = None) -> None:
        super().__init__(ctx, session, screen=screen)
        self.snapshot: dict = {}
        self.boxes: list = []  # box dicts of the open page (page_snapshot), edited locally while dragging
        self.image_path: str = ""
        self.translated_view: str = ""
        self.geometry = Geometry()
        self.sizes: dict = {}
        self.tool = "pan"
        self.view = "source"
        self.selected: Optional[int] = None
        self.drag: Optional[dict] = None
        self.watch = JobWatch(ctx, self._on_step_end, self._on_step_change)
        self.job_ends = JobEnds(ctx, self._on_any_job_end)
        self._view_pending = False  # the translated page's lookup was refused by a running job
        self.sheet: Optional[BoxSheet] = None
        self.step_progress_text = ""
        self._gen = 0
        self._editing = False  # a box edit is in flight on the io pool
        self._listening = False
        # the source viewer (and the tablet row holding it) kept across refreshes while the page
        # and the Pan / Edit mode stay the same, so the InteractiveViewer keeps its zoom and pan
        self._source_cache: Optional[tuple] = None  # (signature, control)
        self._dual_cache: Optional[tuple] = None  # (source control, row)

    # ---- build --------------------------------------------------------------------------------------

    def build(self) -> ft.Control:
        self._source_cache = self._dual_cache = None  # a new tree: nothing to keep in place
        self.strip = ft.ListView(controls=[], horizontal=True, height=76, spacing=6, key="me-strip")
        self.view_buttons = ft.SegmentedButton(
            segments=[ft.Segment(value="source", label=ft.Text("Source")),
                      ft.Segment(value="translated", label=ft.Text("Translated"))],
            selected=["source"], show_selected_icon=False, on_change=self._on_view, key="me-view")
        self.page_label = ft.Text("", theme_style=ft.TextThemeStyle.BODY_SMALL, key="me-page-label")
        self.prev_button = ft.IconButton(icon=ft.Icons.CHEVRON_LEFT, tooltip="Previous page", key="me-prev",
                                         size_constraints=HIT_TARGET,
                                         on_click=lambda e: self.ctx.spawn(self.step_page(-1)))
        self.next_button = ft.IconButton(icon=ft.Icons.CHEVRON_RIGHT, tooltip="Next page", key="me-next",
                                         size_constraints=HIT_TARGET,
                                         on_click=lambda e: self.ctx.spawn(self.step_page(1)))
        self.tool_buttons: dict = {}
        tool_controls: list = []
        for tool_id, label, icon, reason in TOOLS:
            button = ft.IconButton(icon=getattr(ft.Icons, icon, ft.Icons.EDIT), tooltip=reason or label,
                                   size_constraints=HIT_TARGET, selected=tool_id == self.tool,
                                   disabled=reason is not None, key=f"me-tool-{tool_id}",
                                   on_click=lambda e, t=tool_id: self.set_tool(t))
            self.tool_buttons[tool_id] = button
            tool_controls.append(button)
        self.edit_button = ft.IconButton(icon=ft.Icons.EDIT_NOTE, tooltip="Edit box", key="me-edit-box",
                                         size_constraints=HIT_TARGET, on_click=lambda e: self.open_box_sheet())
        self.exclude_button = ft.IconButton(icon=ft.Icons.BLOCK, tooltip="Exclude from Clean", key="me-exclude",
                                            size_constraints=HIT_TARGET,
                                            on_click=lambda e: self.ctx.spawn(self.toggle_exclude()))
        self.delete_button = ft.IconButton(icon=ft.Icons.DELETE_OUTLINE, tooltip="Delete box", key="me-delete",
                                           size_constraints=HIT_TARGET,
                                           on_click=lambda e: self.ctx.spawn(self.delete_selected()))
        self.clear_button = ft.IconButton(icon=ft.Icons.CLEAR_ALL, tooltip="Clear boxes", key="me-clear-boxes",
                                          size_constraints=HIT_TARGET,
                                          on_click=lambda e: self.ctx.spawn(self.clear_boxes()))
        self.toolbar = ft.Row([*tool_controls, ft.VerticalDivider(width=8), self.edit_button, self.exclude_button,
                               self.delete_button, self.clear_button], scroll=ft.ScrollMode.AUTO, spacing=0,
                              key="me-toolbar")
        self.viewer_holder = ft.Container(expand=True, key="me-viewer-holder")
        self.step_buttons: dict = {}
        for step, label, icon in STEP_BUTTONS:
            self.step_buttons[step] = ft.FilledTonalButton(content=label,
                                                           icon=getattr(ft.Icons, icon, ft.Icons.PLAY_ARROW),
                                                           key=f"me-step-{step}",
                                                           on_click=lambda e, s=step: self.ctx.spawn(self.run_step(s)))
        self.step_status = ft.Text("", theme_style=ft.TextThemeStyle.BODY_SMALL, key="me-step-status")
        self.step_bar = ft.ProgressBar(visible=False, key="me-step-progress")
        self.stop_button = ft.TextButton(content="Stop", icon=ft.Icons.STOP, visible=False, key="me-step-stop",
                                         on_click=lambda e: self._stop())
        self.step_reason = ft.Container(key="me-step-reason")
        more = ft.Row([
            ft.TextButton(content="Import OCR", icon=ft.Icons.FILE_UPLOAD, key="me-import-ocr",
                          on_click=lambda e: self.ctx.spawn(self.import_ocr())),
            ft.TextButton(content="Export OCR", icon=ft.Icons.FILE_DOWNLOAD, key="me-export-ocr",
                          on_click=lambda e: self.ctx.spawn(self.export_ocr())),
            ft.TextButton(content="Auto-saved OCR", icon=ft.Icons.HISTORY, key="me-ocr-files",
                          on_click=lambda e: self.ctx.spawn(self.open_ocr_files())),
            ft.TextButton(content="Files", icon=ft.Icons.PHOTO_LIBRARY, key="me-files",
                          on_click=lambda e: self.screen.select_tab("files") if self.screen else None),
        ], wrap=True, spacing=4)
        # the non-chibi Halgakos, faded, where a page will appear (owner, 2026-10-09)
        self.empty = ft.Container(
            content=ft.Column([faded_mascot(), hint_text("Add images in the Files tab, then open a page here.")],
                              horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=8),
            alignment=ft.Alignment.CENTER, padding=ft.Padding.symmetric(vertical=12), key="me-empty",
        )
        self.root = ft.Column([
            self.empty,
            self.strip,
            ft.Row([self.view_buttons, self.prev_button, self.page_label, self.next_button], wrap=True, spacing=4,
                   vertical_alignment=ft.CrossAxisAlignment.CENTER),
            self.toolbar,
            self.viewer_holder,
            ft.Row(list(self.step_buttons.values()), scroll=ft.ScrollMode.AUTO, spacing=6, key="me-steps"),
            ft.Row([self.step_status, self.stop_button, self.step_reason], wrap=True, spacing=6),
            self.step_bar,
            more,
        ], expand=True, spacing=tokens.SPACING["sm"], key=self.key)
        self.refresh(push_now=False)
        return ft.Container(content=self.root, padding=ft.Padding.symmetric(horizontal=8, vertical=6), expand=True)

    # ---- lifecycle ----------------------------------------------------------------------------------

    def did_show(self) -> None:
        self.job_ends.start()
        for snap in self.watch.adopt((svc.KIND_STEP,)):
            self.session.step_job_id = getattr(snap, "id", None)
            self._on_step_change(snap)
        if not self._listening:
            self.session.editor_listeners.append(self._on_session_event)
            self._listening = True
        files = self.session.files.files
        wanted = self._wanted_path()
        if files and wanted and (wanted != self.image_path):
            self.ctx.spawn(self.open_page(self.session.page_index))
        else:
            self.refresh()
            if self._view_pending:
                self.ctx.spawn(self._retry_translated())

    def on_session_loaded(self) -> None:
        self.refresh()

    def _on_any_job_end(self, snap: Any) -> None:
        """A job of any kind ended (``JobEnds``): a translated-page lookup it refused runs again."""
        if self._view_pending:
            self.ctx.spawn(self._retry_translated())

    async def _retry_translated(self) -> str:
        """The open page's translated view, looked up again after a running job refused it
        (``MangaBusy``); still pending while a job owns the process state."""
        path = self.image_path
        if not path:
            self._view_pending = False
            return ""
        try:
            view = await self.ctx.io(self._display_translated)
        except svc.MangaBusy:
            return ""
        if path != self.image_path:  # another page opened meanwhile: its own lookup applies
            return ""
        self._view_pending = False
        self.translated_view = view
        self.refresh()
        return view

    def dispose(self) -> None:
        self.watch.stop()
        self.job_ends.stop()
        if self._listening and self._on_session_event in self.session.editor_listeners:
            self.session.editor_listeners.remove(self._on_session_event)
        self._listening = False

    def handle_back(self) -> bool:
        if self.selected is not None:
            self.selected = None
            self.refresh()
            return True
        if self.tool != "pan":
            self.set_tool("pan")
            return True
        return False

    def _wanted_path(self) -> str:
        files = self.session.files.files
        if not files:
            return ""
        index = max(0, min(self.session.page_index, len(files) - 1))
        return files[index]

    @property
    def busy(self) -> bool:
        return self.watch.active() is not None

    def _busy_reason(self) -> Optional[str]:
        return "Wait for the running step" if self.busy else None

    # ---- session ------------------------------------------------------------------------------------

    async def _editor(self) -> Any:
        es = await self.ctx.io(self.session.ensure_editor)
        if es is None:
            self.ctx.say(self.session.editor_error or "The manga editor is not available in this build")
        return es

    async def _snapshot(self) -> dict:
        es = self.session.editor
        if es is None or not self.image_path:
            return {}
        snap = await self.ctx.io(es.page_snapshot, self.image_path)
        return dict(snap or {})

    async def _apply_snapshot(self, snap: dict) -> None:
        self.snapshot = dict(snap or {})
        self.boxes = [dict(b) for b in (self.snapshot.get("boxes") or [])]
        if self.selected is not None and self.selected >= len(self.boxes):
            self.selected = None
        try:
            self.translated_view = await self.ctx.io(self._display_translated)
            self._view_pending = False
        except svc.MangaBusy:  # not known while another job runs: looked up again when it ends
            self.translated_view = ""
            self._view_pending = True
        self.refresh()

    def _display_translated(self) -> str:
        """Blocking: the versioned display copy of the page's translated output. Raises
        ``MangaBusy`` when the page's output path cannot be looked up while a job runs."""
        snap = self.snapshot
        path = str(snap.get("translated_path") or snap.get("rendered_path") or "")
        if not path or not os.path.isfile(path):
            candidate = self.session.files.output_path_for(self.image_path) if self.image_path else ""
            path = candidate if candidate and os.path.isfile(candidate) else ""
        if not path:
            return ""
        try:
            return svc.display_copy(path, int(snap.get("revision") or 0), self.session.view_cache)
        except OSError:
            return path

    def _on_session_event(self, kind: str, data: Any) -> None:
        """MangaEditorSession events (job thread): Translate all progress, new renders."""
        def run() -> None:
            if kind == "progress" and isinstance(data, dict):
                self.step_progress_text = f"Page {data.get('current')}/{data.get('total')}"
                self.step_status.value = self.step_progress_text
                push(self.step_status)

        dispatcher = getattr(self.ctx, "dispatcher", None)
        if dispatcher is not None and getattr(dispatcher, "bound", False):
            dispatcher.post(run)
        else:
            run()

    # ---- pages --------------------------------------------------------------------------------------

    async def open_page(self, index: int) -> dict:
        files = self.session.files.files
        if not files:
            self.snapshot, self.boxes, self.image_path = {}, [], ""
            self.refresh()
            return {}
        if self.busy:
            self.ctx.say("Wait for the running step")
            return self.snapshot
        index = max(0, min(int(index), len(files) - 1))
        es = await self._editor()
        if es is None:
            self.refresh()
            return {}
        path = files[index]
        self.session.page_index = index
        if path not in self.sizes:
            self.sizes[path] = await self.ctx.io(image_size, path)
        snap = await self.ctx.io(es.open_page, path)
        self.image_path = path
        self.geometry.image_w, self.geometry.image_h = (float(v) for v in self.sizes.get(path, (0, 0)))
        self.selected = None
        self.drag = None
        await self._apply_snapshot(snap)
        return self.snapshot

    async def step_page(self, delta: int) -> dict:
        return await self.open_page(self.session.page_index + delta)

    # ---- rendering ----------------------------------------------------------------------------------

    def refresh(self, push_now: bool = True) -> None:
        if self.root is None:
            return
        self._gen += 1
        files = self.session.files.files
        has_page = bool(files) and bool(self.image_path)
        self.empty.visible = not files
        self.strip.controls = [self._thumb(i, p) for i, p in enumerate(files)]
        self.strip.visible = bool(files)
        self.page_label.value = f"{self.session.page_index + 1} / {len(files)}" if files else ""
        busy = self.busy
        self.prev_button.disabled = busy or not files or self.session.page_index <= 0
        self.next_button.disabled = busy or not files or self.session.page_index >= len(files) - 1
        for tool_id, button in self.tool_buttons.items():
            button.selected = tool_id == self.tool
        box_selected = has_page and self.selected is not None and 0 <= self.selected < len(self.boxes)
        for button in (self.delete_button, self.exclude_button, self.edit_button):
            button.disabled = busy or not box_selected
        self.clear_button.disabled = busy or not has_page or not self.boxes
        if has_page:
            self.viewer_holder.content = self._viewer()
        else:
            self._source_cache = self._dual_cache = None
            self.viewer_holder.content = hint_text(self.session.editor_error) if self.session.editor_error else None
        self._render_steps(has_page)
        if push_now:
            push(self.root)

    def _thumb(self, index: int, path: str) -> ft.Control:
        current = index == self.session.page_index
        skipped = self.is_skipped(path)
        image: ft.Control = ft.Image(src=path, width=52, height=68, fit=ft.BoxFit.COVER, cache_width=104,
                                     border_radius=4, gapless_playback=True)
        if skipped:  # the desktop skip marker (⏭️), dimmed like the Files tab row
            image = ft.Stack([image, ft.Container(content=ft.Text("⏭️", size=12), right=1, top=1,
                                                  bgcolor=ft.Colors.with_opacity(0.7, ft.Colors.SURFACE),
                                                  border_radius=4, padding=1)], width=52, height=68)
        return ft.Container(
            content=image,
            border=ft.Border.all(3 if current else 1, ft.Colors.PRIMARY if current else ft.Colors.OUTLINE_VARIANT),
            border_radius=6, opacity=0.5 if skipped else 1.0,
            on_click=lambda e, i=index: self.ctx.spawn(self.open_page(i)),
            on_long_press=lambda e, p=path: self.thumb_menu(p),
            key=f"me-thumb-{self._gen}-{index}",
            tooltip=os.path.basename(path) + (" · skipped" if skipped else ""))

    def is_skipped(self, path: str) -> bool:
        try:
            return bool(self.session.files.is_skipped(path))
        except Exception:
            return False

    def thumb_menu(self, path: str) -> Any:
        """Long-press on a page thumbnail: the desktop preview's thumbnail menu "⏭️ Skip Processing" /
        "▶️ Process This Image" (the Files tab's per-file switch, ``MangaFileList.toggle_skip``)."""
        from glossarion_mobile.ui.components.action_sheet import ActionItem, ActionSheet

        skipped = self.is_skipped(path)
        label = "▶️ Process This Image" if skipped else "⏭️ Skip Processing"
        items = [ActionItem(label, lambda p=path: self.ctx.spawn(self.toggle_skip(p)),
                            icon="PLAY_ARROW" if skipped else "SKIP_NEXT", key="me-thumb-skip")]
        sheet = ActionSheet(items, title=os.path.basename(path), tablet=bool(getattr(self.ctx, "tablet", False)))
        self.ctx.extras["manga_thumb_sheet"] = sheet
        if self.ctx.page is not None:
            sheet.show(self.ctx.page)
        return sheet

    async def toggle_skip(self, path: str) -> Optional[bool]:
        """Skip / process a page from the editor strip; both the Files tab and the strip re-render."""
        files_tab = getattr(self.screen, "files_tab", None)
        mutate = getattr(files_tab, "_mutate", None)
        if callable(mutate):
            result = await mutate(self.session.files.toggle_skip, path)
        else:
            try:
                result = await self.ctx.io(self.session.files.toggle_skip, path)
            except Exception as exc:
                self.ctx.say(f"Could not update the list: {exc}")
                result = None
        self.refresh()
        return result

    def _viewer(self) -> ft.Control:
        if getattr(self.ctx, "tablet", False):
            if self._dual_cache is None:
                self._source_cache = None  # a new row gets a new source viewer (never re-parented)
            source = self._source_viewer()
            translated = ft.Container(content=self._translated_viewer(), expand=True)
            cached = self._dual_cache
            if cached is not None and cached[0] is source:
                row = cached[1]
                row.controls[1] = translated  # the source side (and its zoom) stays in place
                return row
            row = ft.Row([ft.Container(content=source, expand=True), translated],
                         expand=True, spacing=8, key=f"me-dual-{self._gen}")
            self._dual_cache = (source, row)
            return row
        if self._dual_cache is not None:  # the tablet row is gone: so is the source viewer in it
            self._dual_cache = self._source_cache = None
        if self.view == "source":
            return self._source_viewer()
        self._source_cache = None  # leaving the tree: its zoom state goes with it
        return self._translated_viewer()

    def _shapes(self) -> list:
        shapes: list = []
        scale = self.geometry.scale
        for index, box in enumerate(self.boxes):
            excluded = bool(box.get("exclude_from_clean"))
            color = SELECTED_COLOR if index == self.selected else (EXCLUDED_COLOR if excluded else BOX_COLOR)
            paint = ft.Paint(color=color, stroke_width=3 if index == self.selected else 2,
                             style=ft.PaintingStyle.STROKE)
            x, y = float(box.get("x", 0)) * scale, float(box.get("y", 0)) * scale
            w, h = float(box.get("width", 0)) * scale, float(box.get("height", 0)) * scale
            polygon = box.get("polygon") or []
            if box.get("shape") == "polygon" and len(polygon) >= 3:
                points = [(float(p[0]) * scale, float(p[1]) * scale) for p in polygon]
                elements = [cv.Path.MoveTo(*points[0])] + [cv.Path.LineTo(px, py) for px, py in points[1:]]
                elements.append(cv.Path.Close())
                shapes.append(cv.Path(elements=elements, paint=paint))
            elif box.get("shape") == "ellipse":
                shapes.append(cv.Oval(x, y, w, h, paint=paint))
            else:
                shapes.append(cv.Rect(x, y, w, h, paint=paint))
            marks = ("✓" if box.get("translation") else ("·" if box.get("ocr_text") else "")) + (" ✕" if excluded else "")
            shapes.append(cv.Text(x + 3, y + 2, f"{index + 1}{marks}",
                                  style=ft.TextStyle(size=11, color=color, weight=ft.FontWeight.W_700)))
            if index == self.selected:
                shapes.append(cv.Rect(x + w - 8, y + h - 8, 16, 16,
                                      paint=ft.Paint(color=SELECTED_COLOR, style=ft.PaintingStyle.FILL)))
        drag = self.drag or {}
        paint = ft.Paint(color=SELECTED_COLOR, stroke_width=2, style=ft.PaintingStyle.STROKE)
        if drag.get("mode") == "draw" and drag.get("current"):
            (sx, sy), (cx, cy) = drag["start"], drag["current"]
            x, y, w, h = normalize_rect(sx, sy, cx, cy)
            shapes.append(cv.Oval(x, y, w, h, paint=paint) if self.tool == "circle" else cv.Rect(x, y, w, h, paint=paint))
        elif drag.get("mode") == "lasso" and len(drag.get("points") or ()) >= 2:
            points = drag["points"]
            elements = [cv.Path.MoveTo(*points[0])] + [cv.Path.LineTo(px, py) for px, py in points[1:]]
            shapes.append(cv.Path(elements=elements, paint=paint))
        return shapes

    def _source_viewer(self) -> ft.Control:
        """The page with its boxes. Pan mode: the InteractiveViewer pans and zooms, and the
        gesture surface on top only takes long-presses (box sheet): a pan recognizer there would
        win Flutter's gesture arena against the viewer's own and swallow one-finger panning. Edit
        modes: the surface takes taps and drags, the viewer stays put. The control is kept (same
        objects, same keys, updated in place) while the page and the mode stay the same, so a
        refresh (a box selected, a step finished) keeps the zoom; a new page or mode rebuilds it
        under new keys (Flet freezes a subtree re-created under an old key)."""
        geometry = self.geometry
        aspect = (geometry.image_w / geometry.image_h) if geometry.image_w and geometry.image_h else 0.7
        editing = self.tool != "pan"
        signature = (self.image_path, editing, round(aspect, 6))
        cached = self._source_cache
        if cached is not None and cached[0] == signature:
            self.canvas.shapes = self._shapes()
            return cached[1]
        # every layer fills the aspect-ratio box: image, shapes, then the gesture surface on top
        self.canvas = cv.Canvas(shapes=self._shapes(), on_resize=self._on_canvas_resize, left=0, top=0, right=0,
                                bottom=0, key=f"me-canvas-{self._gen}")
        image = ft.Image(src=self.image_path, fit=ft.BoxFit.FILL, gapless_playback=True, left=0, top=0, right=0,
                         bottom=0, key=f"me-image-{self._gen}")
        handlers: dict = {"on_long_press_start": self._on_long_press}
        if editing:
            handlers.update(on_tap_down=self._on_tap, on_pan_start=self._on_pan_start,
                            on_pan_update=self._on_pan_update, on_pan_end=self._on_pan_end)
        self.gestures = ft.GestureDetector(
            content=ft.Container(bgcolor=ft.Colors.TRANSPARENT), drag_interval=DRAG_INTERVAL,
            left=0, top=0, right=0, bottom=0, key=f"me-gestures-{self._gen}", **handlers)
        stack = ft.Stack([image, self.canvas, self.gestures], aspect_ratio=aspect, key=f"me-stack-{self._gen}")
        self.interactive = ft.InteractiveViewer(content=stack, min_scale=1.0, max_scale=6.0, pan_enabled=not editing,
                                                scale_enabled=not editing, key=f"me-iv-{self._gen}")
        container = ft.Container(content=self.interactive, expand=True, bgcolor=ft.Colors.SURFACE_CONTAINER_LOWEST,
                                 border_radius=tokens.RADII["field"], key=f"me-source-{self._gen}")
        self._source_cache = (signature, container)
        return container

    def _translated_viewer(self) -> ft.Control:
        path = self.translated_view
        if not path:
            text = ("Wait for the running job: the translated page shows once it has finished"
                    if self._view_pending else "Not translated yet: run Translate (or Start in Files).")
            return ft.Container(content=hint_text(text), alignment=ft.Alignment.CENTER, expand=True,
                                key=f"me-translated-empty-{self._gen}")
        source = str(self.snapshot.get("translated_path") or self.snapshot.get("rendered_path") or path)
        return ft.Container(
            content=ft.InteractiveViewer(content=ft.Image(src=path, fit=ft.BoxFit.CONTAIN, gapless_playback=True),
                                         min_scale=1.0, max_scale=6.0),
            expand=True, bgcolor=ft.Colors.SURFACE_CONTAINER_LOWEST, border_radius=tokens.RADII["field"],
            on_long_press=lambda e, p=source: export_sheet(self.ctx, p), key=f"me-translated-{self._gen}",
            tooltip="Long-press to share or save")

    def _steps_reason(self) -> Optional[str]:
        if not self.ctx.has_kind(svc.KIND_STEP):
            return "Editor steps are not available in this build"
        # find_spec only: importing the editor core takes many seconds (it runs on the io pool,
        # in MangaSession.ensure_editor), never on the UI loop
        if self.session.editor is None:
            if self.session.editor_error:
                return str(self.session.editor_error)
            if not svc.importable("manga_editor_core"):
                return svc.MISSING_CORE + " (manga_editor_core)"
        return None

    def _render_steps(self, has_page: bool) -> None:
        snap = self.watch.active()
        running = snap is not None
        reason = self._steps_reason()
        for step, button in self.step_buttons.items():
            needs_boxes = step == "render"
            button.disabled = running or not has_page or reason is not None or (needs_boxes and not self.boxes)
        self.step_reason.content = ReasonChip(reason=svc.chip_text(reason), detail=reason) if reason else None
        self.stop_button.visible = running
        self.step_bar.visible = running
        if running:
            state = getattr(getattr(snap, "state", None), "value", "")
            self.step_status.value = (self.step_progress_text or getattr(snap, "phase", "")
                                      or ("Queued…" if state == "QUEUED" else "Running…"))

    # ---- view / tools ---------------------------------------------------------------------------------

    def _on_view(self, e: Any = None) -> None:
        chosen = list(self.view_buttons.selected or ["source"])
        self.view = chosen[0]
        self.refresh()

    def set_tool(self, tool: str) -> None:
        reason = {t: r for t, _l, _i, r in TOOLS}.get(tool)
        if reason:
            self.ctx.say(reason)
            return
        self.tool = tool
        self.drag = None
        if tool != "pan" and self.view != "source":
            self.view = "source"
            self.view_buttons.selected = ["source"]
        self.refresh()

    def _on_canvas_resize(self, e: Any) -> None:
        width = float(getattr(e, "width", 0) or 0)
        if width and abs(width - self.geometry.display_w) > 0.5:
            self.geometry.display_w = width
            self._redraw()

    @staticmethod
    def _local(e: Any) -> tuple:
        pos = getattr(e, "local_position", None)
        if pos is None:
            return (0.0, 0.0)
        return (float(getattr(pos, "x", 0.0) or 0.0), float(getattr(pos, "y", 0.0) or 0.0))

    def _redraw(self) -> None:
        canvas = getattr(self, "canvas", None)
        if canvas is not None:
            canvas.shapes = self._shapes()
            push(canvas)

    def _can_edit(self) -> bool:
        return bool(self.image_path) and self.session.editor is not None and not self.busy and not self._editing

    # ---- gestures (display coords in, image px stored) --------------------------------------------------

    def tap_at(self, dx: float, dy: float) -> Optional[int]:
        if not self.image_path or self.tool == "pan":
            return None
        ix, iy = self.geometry.to_image(dx, dy)
        self.selected = hit_test(self.boxes, ix, iy)
        self.refresh()
        return self.selected

    def long_press_at(self, dx: float, dy: float) -> Optional[BoxSheet]:
        if not self.image_path:
            return None
        ix, iy = self.geometry.to_image(dx, dy)
        index = hit_test(self.boxes, ix, iy)
        if index is None:
            return None
        self.selected = index
        self.refresh()
        return self.open_box_sheet(index)

    def _on_tap(self, e: Any) -> None:
        self.tap_at(*self._local(e))

    def _on_long_press(self, e: Any) -> None:
        self.long_press_at(*self._local(e))

    def pan_start(self, dx: float, dy: float) -> None:
        if self.tool == "pan" or not self._can_edit():
            return
        if self.tool in ("box", "circle"):
            self.drag = {"mode": "draw", "start": (dx, dy), "current": (dx, dy)}
        elif self.tool == "lasso":
            self.drag = {"mode": "lasso", "points": [(dx, dy)]}
        elif self.tool == "select":
            ix, iy = self.geometry.to_image(dx, dy)
            tolerance = HANDLE / max(self.geometry.scale, 1e-6)
            if self.selected is not None and 0 <= self.selected < len(self.boxes) and \
                    on_resize_handle(self.boxes[self.selected], ix, iy, tolerance):
                self.drag = {"mode": "resize", "index": self.selected, "last": (ix, iy)}
                return
            index = hit_test(self.boxes, ix, iy)
            self.selected = index
            self.drag = {"mode": "move", "index": index, "last": (ix, iy)} if index is not None else None

    def pan_update(self, dx: float, dy: float) -> None:
        drag = self.drag
        if not drag:
            return
        mode = drag.get("mode")
        if mode == "draw":
            drag["current"] = (dx, dy)
        elif mode == "lasso":
            drag["points"].append((dx, dy))
        elif mode in ("move", "resize"):
            index = drag.get("index")
            if index is None or not (0 <= index < len(self.boxes)):
                return
            ix, iy = self.geometry.to_image(dx, dy)
            lx, ly = drag["last"]
            box = self.boxes[index]
            old = (float(box["x"]), float(box["y"]), float(box["width"]), float(box["height"]))
            if mode == "move":
                box["x"], box["y"] = max(0.0, old[0] + ix - lx), max(0.0, old[1] + iy - ly)
            else:
                box["width"], box["height"] = max(MIN_BOX, old[2] + ix - lx), max(MIN_BOX, old[3] + iy - ly)
            if box.get("polygon") and old[2] > 0 and old[3] > 0:  # EditorBox.set_geometry's polygon rule
                sx, sy = float(box["width"]) / old[2], float(box["height"]) / old[3]
                box["polygon"] = [[box["x"] + (px - old[0]) * sx, box["y"] + (py - old[1]) * sy]
                                  for px, py in box["polygon"]]
            drag["last"] = (ix, iy)
        self._redraw()

    async def pan_end(self) -> Optional[int]:
        drag, self.drag = self.drag, None
        if not drag or self.session.editor is None:
            self.refresh()
            return None
        es = self.session.editor
        mode = drag.get("mode")
        result = None
        self._editing = True
        try:
            if mode == "draw" and drag.get("current"):
                (sx, sy), (cx, cy) = drag["start"], drag["current"]
                ax, ay = self.geometry.to_image(sx, sy)
                bx, by = self.geometry.to_image(cx, cy)
                x, y, w, h = normalize_rect(ax, ay, bx, by)
                if w >= MIN_BOX and h >= MIN_BOX:
                    shape = SHAPES.get(self.tool, "rect")
                    await self.ctx.io(lambda: es.add_box(x, y, w, h, shape=shape))
                    result = len(self.boxes)
            elif mode == "lasso":
                found = lasso_bounds([self.geometry.to_image(px, py) for px, py in drag.get("points") or ()])
                if found is not None:
                    (x, y, w, h), polygon = found
                    await self.ctx.io(lambda: es.add_box(x, y, w, h, shape="polygon", polygon=polygon))
                    result = len(self.boxes)
            elif mode in ("move", "resize") and drag.get("index") is not None:
                index = int(drag["index"])
                box = self.boxes[index]
                await self.ctx.io(lambda: es.update_box(index, box["x"], box["y"], box["width"], box["height"],
                                                        rerender=False))
                result = index
        except Exception as exc:
            self.ctx.say(f"Could not change the box: {exc}")
        finally:
            self._editing = False
        if result is not None:
            self.selected = result
        await self._apply_snapshot(await self._snapshot())
        return result

    def _on_pan_start(self, e: Any) -> None:
        self.pan_start(*self._local(e))

    def _on_pan_update(self, e: Any) -> None:
        self.pan_update(*self._local(e))

    def _on_pan_end(self, e: Any) -> None:
        self.ctx.spawn(self.pan_end())

    # ---- box actions ------------------------------------------------------------------------------------

    async def _box_call(self, fn: Any) -> bool:
        if not self._can_edit():
            self.ctx.say("Wait for the running step" if self.busy else "Open a page first")
            return False
        self._editing = True
        try:
            await self.ctx.io(fn)
            return True
        except Exception as exc:
            self.ctx.say(f"Could not change the box: {exc}")
            return False
        finally:
            self._editing = False
            await self._apply_snapshot(await self._snapshot())

    async def delete_selected(self) -> bool:
        index = self.selected
        es = self.session.editor
        if index is None or es is None:
            return False
        self.selected = None
        return await self._box_call(lambda: es.delete_box(index))

    async def clear_boxes(self, *, confirm: bool = True) -> bool:
        """Clear Boxes (desktop ``_on_clear_boxes_clicked``, shared through manga_editor_core): every box
        of the page with its OCR text and translation, and the page's translated image."""
        es = self.session.editor
        if es is None or not self.image_path:
            return False
        if confirm:
            from glossarion_mobile.ui.tools.common import ask

            answer = await ask(self.ctx, "Clear boxes",
                               "Remove every box on this page with its OCR text and translation? The page's "
                               "translated image is deleted; the cleaned image stays.",
                               (("no", "Cancel", "text"), ("yes", "Clear", "filled")), key="me-clear-confirm")
            if answer != "yes":
                return False
        self.selected = None
        return await self._box_call(lambda: es.clear_page())

    async def toggle_exclude(self) -> bool:
        index = self.selected
        es = self.session.editor
        if index is None or es is None or not (0 <= index < len(self.boxes)):
            return False
        value = not bool(self.boxes[index].get("exclude_from_clean"))
        return await self._box_call(lambda: es.set_box_excluded(index, value))

    def open_box_sheet(self, index: Optional[int] = None) -> Optional[BoxSheet]:
        index = self.selected if index is None else index
        if index is None or not (0 <= index < len(self.boxes)):
            return None
        box = self.boxes[index]
        self.sheet = BoxSheet(box, index, on_action=lambda name, i, changes: self.ctx.spawn(
            self.box_action(name, i, changes)), busy_reason=self._busy_reason(), steps_reason=self._steps_reason(),
            tab="translation" if box.get("translation") else "ocr")
        self.ctx.extras["manga_box_sheet"] = self.sheet
        return self.sheet.show(self.ctx.page)

    async def box_action(self, name: str, index: int, changes: dict) -> Optional[str]:
        """BoxSheet actions: apply the edits (session calls), then the action itself."""
        es = self.session.editor
        if es is None:
            return None
        if changes:
            def apply() -> None:
                if "ocr_text" in changes or "translation" in changes:
                    es.edit_box_text(index, ocr_text=changes.get("ocr_text"), translation=changes.get("translation"))
                if "free_text" in changes:
                    es.set_box_free_text(index, changes["free_text"])
                if "exclude_from_clean" in changes:
                    es.set_box_excluded(index, changes["exclude_from_clean"])
                if "inpaint_iterations" in changes:
                    es.set_box_iterations(index, changes["inpaint_iterations"])

            if not await self._box_call(apply):
                return None
        if name == "save":
            return None
        if name == "delete":
            self.selected = index
            await self.delete_selected()
            return None
        step = {"save_render": "render", "ocr": "ocr_box", "translate": "translate_box", "clean": "clean_box"}.get(name)
        if step is None:
            return None
        return await self.run_step(step, index=index if step != "render" else None)

    # ---- steps --------------------------------------------------------------------------------------

    async def run_step(self, step: str, *, index: Optional[int] = None, extra: Optional[dict] = None,
                       images: Optional[Sequence[str]] = None) -> Optional[str]:
        if not self.image_path:
            self.ctx.say("Open a page first")
            return None
        if self.busy:
            self.ctx.say("Wait for the running step")
            return None
        reason = self._steps_reason()
        if reason:
            self.ctx.say(reason)
            return None
        es = await self._editor()
        if es is None:
            return None
        if step == "detect" and self.boxes:
            from glossarion_mobile.ui.tools.common import ask

            answer = await ask(self.ctx, "Detect text", f"Replace the {len(self.boxes)} boxes on this page with "
                               "freshly detected ones?", (("no", "Cancel", "text"), ("yes", "Replace", "filled")),
                               key="me-detect-confirm")
            if answer != "yes":
                return None
        if step == "translate_all" and images is None:
            run_files, error = self.session.files.run_files()
            if error or not run_files:
                self.ctx.say(error or "No pages to translate")
                return None
            images = run_files
        kinds = self.step_model_kinds(step)
        if kinds and not await ensure_models(self.ctx, self.session.models, self.ctx.config_snapshot(), kinds=kinds,
                                             action=svc.STEP_LABELS.get(step, step)):
            return None
        if kinds:  # the job fetches what is still missing first (verified, resumable, Stop cancels)
            extra = {**dict(extra or {}), "model_kinds": list(kinds)}
        spec = svc.step_spec(step, self.session.editor_token, self.image_path, images=images, index=index,
                             extra=extra)
        job_id = await self.ctx.submit(spec)
        if job_id:
            self.session.step_job_id = job_id
            self.watch.watch(job_id)
            self.step_progress_text = ""
            self.step_status.value = "Queued…"
            self.refresh()
        return job_id

    def step_model_kinds(self, step: str) -> tuple:
        """The downloadable models a step loads: the detector when it detects (Detect; Recognize /
        Clean / Translate on a page without boxes; Translate all), the inpainter when it cleans."""
        detect = () if self.boxes and step in ("recognize", "clean", "translate") else ("detector",)
        return {"detect": ("detector",), "recognize": detect, "clean": detect + ("inpaint",), "clean_box": ("inpaint",),
                "translate": detect + ("inpaint",), "translate_all": ("detector", "inpaint")}.get(step, ())

    def _stop(self) -> None:
        snap = self.watch.active()
        jobs = self.ctx.jobs
        if snap is not None and jobs is not None:
            try:
                jobs.request_stop(snap.id)
            except Exception:
                log.exception("stopping the editor step failed")

    def _on_step_change(self, snap: Any) -> None:
        self._render_steps(bool(self.image_path))
        push(self.step_status, self.step_bar, self.stop_button)

    def _on_step_end(self, snap: Any) -> None:
        error = getattr(snap, "error", None)
        result = dict(getattr(snap, "result", {}) or {})
        spec = getattr(snap, "spec", None)
        step = str(result.get("manga_step") or (spec.params.get("step") if spec is not None else "") or "")
        label = svc.STEP_LABELS.get(step, step)
        if getattr(snap, "stopped", False):
            text = f"{label} stopped"
        elif error:
            text = f"{label} failed: {error}"
        else:
            text = f"{label} done"
            imported = result.get("manga_import")
            if isinstance(imported, dict):
                matched = int(imported.get("matched") or 0)
                text = f"Imported OCR for {matched} of {imported.get('files', 0)} pages"
                path = str((spec.params.get("path") if spec is not None else "") or "")
                if matched and path:
                    # the desktop import is also the OCR the next Start reuses (Files › Run)
                    self.session.imported_ocr = {"path": path, "matched": matched,
                                                 "files": int(imported.get("files") or 0)}
                    if self.screen is not None:
                        try:
                            self.screen.files_tab.refresh()
                        except Exception:
                            pass
                elif not matched:
                    text = "The OCR file does not match any loaded image"
        self.step_status.value = text
        self.step_progress_text = ""
        self.ctx.spawn(self._reload_after_step(step))

    async def _reload_after_step(self, step: str) -> None:
        await self._apply_snapshot(await self._snapshot())
        if step in ("translate", "render", "translate_all", "translate_box", "import_ocr") and self.translated_view:
            if not getattr(self.ctx, "tablet", False):
                self.view = "translated"
                self.view_buttons.selected = ["translated"]
            self.refresh()
        if self.screen is not None and step in ("translate_all", "translate"):
            try:
                self.screen.files_tab.refresh()
            except Exception:
                pass

    # ---- OCR JSON ---------------------------------------------------------------------------------------

    async def export_ocr(self) -> Optional[str]:
        files = self.session.files.files
        if not files:
            self.ctx.say("Add images first")
            return None
        if self.busy:
            self.ctx.say("Wait for the running step")
            return None
        es = await self._editor()
        if es is None:
            return None

        def run() -> dict:
            target = self.session.files.ocr_export_path() or os.path.join(
                self.session.exports_dir, f"manga_ocr_{svc.now_stamp()}.json")
            root = os.path.commonpath([os.path.dirname(p) for p in files]) if len(files) > 1 else os.path.dirname(files[0])
            return es.export_ocr(target, files, source_root=root)

        try:
            result = await self.ctx.io(run)
        except Exception as exc:
            self.ctx.say(f"Export failed: {exc}")
            return None
        path = (result or {}).get("path")
        if not path:
            self.ctx.say("Nothing to export yet: recognize or translate a page first")
            return None
        self.ctx.say(f"Exported OCR for {result.get('pages', 0)} pages")
        export_sheet(self.ctx, path)
        return path

    async def import_ocr(self, path: Optional[str] = None) -> Optional[str]:
        files = self.session.files.files
        if not files:
            self.ctx.say("Add the matching images first")
            return None
        if path is None:
            picker = self.ctx.files
            if picker is None:
                self.ctx.say("File picking is not available")
                return None
            picked = await picker.pick_files(allowed_extensions=["json"], allow_multiple=False,
                                             dialog_title="Import manga OCR JSON")
            if not picked:
                return None
            path = picked[0].path
        if not self.image_path:  # (from the Files tab) the step runs on a page of the session
            await self.open_page(self.session.page_index)
        return await self.run_step("import_ocr", images=files, extra={"path": path})

    async def open_ocr_files(self) -> Any:
        """The auto-saved OCR exports (the OCR Text folder of the output root), newest first."""
        from glossarion_mobile.ui.components.action_sheet import ActionItem, ActionSheet

        def scan() -> list:
            return svc.list_ocr_files(self.session.files.ocr_dir())[:30]

        try:
            paths = await self.ctx.io(scan)
        except svc.MangaBusy as exc:  # the folder is not known while another job runs
            self.ctx.say(str(exc))
            return None
        if not paths:
            self.ctx.say("No auto-saved OCR files yet")
            return None
        items = [ActionItem(os.path.basename(p), (lambda p=p: self.import_ocr(p)), icon="DESCRIPTION",
                            key=f"me-ocr-file-{i}") for i, p in enumerate(paths)]
        sheet = ActionSheet(items, title="Auto-saved OCR files", subtitle="Tap to import",
                            tablet=bool(getattr(self.ctx, "tablet", False)))
        self.ctx.extras["manga_ocr_files_sheet"] = sheet
        if self.ctx.page is not None:
            sheet.show(self.ctx.page)
        return sheet
