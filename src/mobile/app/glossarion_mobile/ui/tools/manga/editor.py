"""Manga › Editor (UI_SPEC §4.6 Editor, §5.10 ``MangaEditor``).

The screen of ``manga_editor_core.MangaEditorSession`` (ImageRenderer's GUI-free editor, the
same code the desktop preview runs):

* **Page strip** of the Files selection (skipped pages dimmed with ⏭️; long-press a thumbnail for
  the desktop thumbnail menu's "⏭️ Skip Processing" / "▶️ Process This Image").
* **One page view** (the desktop preview's merged source / output view) over the full width and
  every dp of height the controls leave. Its toolbar row starts with the desktop's **display-mode
  button** (``MangaImagePreviewWidget.cleaned_toggle_btn``): one button showing the current state,
  ✒️ translated → 🧽 cleaned → 📄 original on each tap; the same boxes and tools stay over the
  translated render, the cleaned page or the original page. A finished Translate / Translate all /
  Start switches to ✒️, a finished Clean to 🧽 (desktop ``_restore_clean_button``); the mode is also
  the session preview's ``source_display_mode`` the shared Clean-this-box code reads.
* The page sits in an ``InteractiveViewer`` (pinch zoom 1-8x, pan) with its box overlay in the
  same transformed ``Stack(Image, canvas.Canvas, GestureDetector)``, so the boxes stay on the
  image at every zoom; the viewer (and its zoom) is kept across the tools. The page is always the
  viewer's full width and does not move at 1x unless it is taller than the viewer. **Pan** mode:
  the viewer pans and zooms. **Edit** modes (the zoom stays where Pan left it): Select/Move (drag
  a box or its top-left move handle; its bottom-right handle resizes; the handles show in this
  tool only), Box, Circle and Lasso; Delete and Exclude from Clean act on the selected box. A
  box stays selected until another box is picked, ✕ / Back, or Select/Move is left. A moved
  translated box carries its rendered text along at once; the page is re-rendered (Save & Update
  Overlay) a moment after the last move.
* **Workflow buttons** Detect · Clean · Recognize · Translate · Translate all · Save & Update
  Overlay, and the per-box OCR / Translate / Clean actions — ``manga_step`` jobs on the
  JobService: OCR and translation need the job's HeadlessOwner (plan §1), which only exists on
  the job thread. Box edits (add, move, resize, delete, texts, flags) are session calls on the
  io pool; editing waits while a step runs (the session runs one step at a time). A step that
  loads a model not on the device yet (detector / ONNX inpainter) offers its download first.
* **The selected box** (tap in an edit mode, long-press in Pan) → ``BoxPanel`` floating over the
  page, at the bottom or the top, away from the box (the page keeps its size): OCR and
  Translated tabs, each text edited in place, and Move (exact position / size, nudges). The toolbar's ⋯ holds Share, Import OCR
  (renders imported translations: a job) / Export OCR (``manga_ocr_io`` JSON in the OCR Text
  folder) / the auto-saved OCR files and Files.
* The editor rewrites the same output file on every render, so the cleaned / translated views show
  a versioned copy (``page_<id>_v3_<mtime>.png``): Flet's ``Image`` cache never shows an old render.

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
from glossarion_mobile.ui.tools.manga.box_sheet import COMPACT_BUTTON, ITERATION_CHOICES, BoxPanel, box_signature
from glossarion_mobile.ui.tools.manga.common import JobEnds, MangaTab, export_sheet, push
from glossarion_mobile.ui.tools.manga.models import ensure_models

__all__ = ["EditorTab", "Geometry", "PageLayout", "TOOLS", "VIEWS", "ViewTransform", "crop_patches", "handle_at",
           "hit_test", "lasso_bounds", "next_view", "normalize_rect", "place_box"]

log = logging.getLogger("glossarion.tools.manga")

#: (tool id, label, icon, disabled reason)
TOOLS = (
    ("pan", "Pan / zoom", "PAN_TOOL", None),
    ("select", "Select / move", "OPEN_WITH", None),
    ("box", "Box", "CROP_SQUARE", None),
    ("circle", "Circle", "CIRCLE_OUTLINED", None),
    ("lasso", "Lasso", "GESTURE", None),
)
STEP_BUTTONS = (("detect", "Detect", "CENTER_FOCUS_STRONG"), ("clean", "Clean", "CLEANING_SERVICES"),
                ("recognize", "Recognize", "DOCUMENT_SCANNER"), ("translate", "Translate", "TRANSLATE"),
                ("translate_all", "Translate all", "LIBRARY_BOOKS"), ("render", "Update overlay", "AUTO_FIX_NORMAL"))
HANDLE = 26.0  # dp on screen: the move (top-left) / resize (bottom-right) handle of the selected box
HANDLE_REACH = 22.0  # dp on screen from a handle's centre that still grabs it
RENDER_DELAY = 1.2  # s after the last move / resize before the page is re-rendered (Save & Update Overlay)
THUMB_W, THUMB_H = 40, 52  # dp: a page strip thumbnail
MIN_BOX = 8.0  # image px
DRAG_INTERVAL = 24
BOX_COLOR = "#E18F98"
SELECTED_COLOR = "#5A9FD4"
EXCLUDED_COLOR = "#9E9E9E"
SHAPES = {"box": "rect", "circle": "ellipse", "lasso": "polygon"}
#: The page view's display mode, in the order the desktop button cycles it
#: (``MangaImagePreviewWidget._cycle_source_display_mode``: translated -> cleaned -> original):
#: (``source_display_mode``, the button's emoji, its tooltip, its colour).
VIEWS = (("translated", "✒️", "Showing translated output (click to cycle)", "#28a745"),
         ("cleaned", "🧽", "Showing cleaned images (click to cycle)", "#4a7ba7"),
         ("original", "📄", "Showing original images (click to cycle)", "#6c757d"))
VIEW_MODES = tuple(view[0] for view in VIEWS)
#: Steps after which the desktop shows the translated output.
TRANSLATED_STEPS = ("translate", "render", "translate_all", "translate_box", "import_ocr")
MIN_ZOOM, MAX_ZOOM = 1.0, 8.0
GUTTER = 2  # dp around the editor: the page takes the full width
#: The viewer's fling friction (Flutter's default ``interactionEndFrictionCoefficient``): a fling
#: glides velocity / -ln(friction) further.
FRICTION = 0.0000135
DEFAULT_VIEWPORT = (360.0, 480.0)  # dp, until the viewer reports its size


def next_view(view: str) -> str:
    """The display mode after ``view`` (the desktop button's cycle)."""
    index = VIEW_MODES.index(view) if view in VIEW_MODES else -1
    return VIEW_MODES[(index + 1) % len(VIEW_MODES)]


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
    return handle_at(box, x, y, tolerance) == "resize"


def handle_at(box: dict, x: float, y: float, tolerance: float) -> Optional[str]:
    """``move`` (top-left corner) / ``resize`` (bottom-right corner): the handle of ``box`` at image
    point (x, y) within ``tolerance`` (the nearer one when both are), else None."""
    left, top = float(box.get("x", 0)), float(box.get("y", 0))
    right, bottom = left + float(box.get("width", 0)), top + float(box.get("height", 0))
    found = []
    for name, (hx, hy) in (("move", (left, top)), ("resize", (right, bottom))):
        distance = max(abs(x - hx), abs(y - hy))
        if distance <= tolerance:
            found.append((distance, name))
    return min(found)[1] if found else None


def place_box(box: dict, x: float, y: float, width: float, height: float,
              image_w: float = 0.0, image_h: float = 0.0) -> dict:
    """Move / resize ``box`` in place (image px), kept on the page and at least ``MIN_BOX``; a lasso
    polygon is translated and scaled with it (``EditorBox.set_geometry``'s rule)."""
    width, height = max(MIN_BOX, float(width)), max(MIN_BOX, float(height))
    if image_w:
        width = min(width, image_w)
        x = max(0.0, min(float(x), image_w - width))
    else:
        x = max(0.0, float(x))
    if image_h:
        height = min(height, image_h)
        y = max(0.0, min(float(y), image_h - height))
    else:
        y = max(0.0, float(y))
    old = (float(box.get("x", 0)), float(box.get("y", 0)), float(box.get("width", 0)), float(box.get("height", 0)))
    if box.get("polygon") and old[2] > 0 and old[3] > 0:
        sx, sy = width / old[2], height / old[3]
        box["polygon"] = [[x + (px - old[0]) * sx, y + (py - old[1]) * sy] for px, py in box["polygon"]]
    box["x"], box["y"], box["width"], box["height"] = x, y, width, height
    return box


@dataclass
class PageLayout:
    """The page in the viewer: always the viewer's full width (``page_w`` × ``page_h``), in a content
    frame at least as tall as the viewer with the page centred in it. A page taller than the
    viewer pans vertically; one that fits does not move until it is zoomed."""
    viewport_w: float
    viewport_h: float
    page_w: float
    page_h: float
    content_h: float

    @property
    def page_top(self) -> float:
        return (self.content_h - self.page_h) / 2

    @classmethod
    def fit(cls, viewport_w: float, viewport_h: float, aspect: float) -> "PageLayout":
        page_h = viewport_w / aspect if aspect else viewport_h
        return cls(viewport_w, viewport_h, viewport_w, page_h, max(viewport_h, page_h))


@dataclass
class ViewTransform:
    """The viewer's pan / zoom, followed from its interaction events (Flutter keeps the matrix; Flet
    only reports the gestures): scale ``scale`` and translation (``tx``, ``ty``) of the content in the
    viewport, clamped like ``InteractiveViewer`` with a zero boundary margin. It places the box
    panel away from the selected box and keeps the handles a finger's size at every zoom."""
    scale: float = 1.0
    tx: float = 0.0
    ty: float = 0.0
    viewport_w: float = 0.0
    viewport_h: float = 0.0
    content_w: float = 0.0
    content_h: float = 0.0
    _focal: Optional[tuple] = None
    _start_scale: float = 1.0
    _pointers: int = 1

    def resize(self, viewport_w: float, viewport_h: float, content_w: float, content_h: float) -> None:
        self.viewport_w, self.viewport_h = viewport_w, viewport_h
        self.content_w, self.content_h = content_w, content_h
        self.clamp()

    def reset(self) -> None:
        self.scale, self.tx, self.ty, self._focal = 1.0, 0.0, 0.0, None

    def clamp(self) -> None:
        self.scale = max(MIN_ZOOM, min(MAX_ZOOM, self.scale))
        low_x = min(0.0, self.viewport_w - self.content_w * self.scale)
        low_y = min(0.0, self.viewport_h - self.content_h * self.scale)
        self.tx = max(low_x, min(0.0, self.tx))
        self.ty = max(low_y, min(0.0, self.ty))

    def start(self, fx: float, fy: float, pointers: int = 1) -> None:
        self._focal, self._start_scale, self._pointers = (fx, fy), self.scale, max(1, int(pointers or 1))

    def update(self, fx: float, fy: float, gesture_scale: float = 1.0) -> None:
        """The gesture moved: the content point under the previous focal point is scaled about it,
        then follows the focal point."""
        if self._focal is None:
            self.start(fx, fy)
        px, py = self._focal
        scale = max(MIN_ZOOM, min(MAX_ZOOM, self._start_scale * float(gesture_scale or 1.0)))
        cx, cy = (px - self.tx) / self.scale, (py - self.ty) / self.scale
        self.tx, self.ty = px - cx * scale + (fx - px), py - cy * scale + (fy - py)
        self.scale = scale
        self._focal = (fx, fy)
        self.clamp()

    def end(self, vx: float = 0.0, vy: float = 0.0) -> None:
        """The gesture ended: a one-finger fling glides on (velocity / -ln(friction))."""
        import math

        if self._pointers <= 1:
            glide = -1.0 / math.log(FRICTION)
            self.tx += vx * glide
            self.ty += vy * glide
        self._focal = None
        self.clamp()

    def to_view(self, x: float, y: float) -> tuple:
        """Content (unzoomed) point -> viewport point."""
        return (self.tx + x * self.scale, self.ty + y * self.scale)


def crop_patches(translated: str, background: str, rect: tuple, image_size: tuple) -> tuple:
    """Blocking: PNG bytes of ``rect`` (x, y, w, h in page px) cut from the translated render (the
    box's rendered text) and from ``background`` (the cleaned / original page: what the text covers),
    each scaled from ``image_size`` to its file's own size. (None, None) when either cannot be read."""
    import io as _io

    try:
        from safe_image import open_page_image  # the owner's guarded Pillow open (pages are untrusted)
    except Exception:
        return (None, None)
    out = []
    iw, ih = (float(v) for v in image_size)
    x, y, w, h = rect
    for path in (translated, background):
        try:
            with open_page_image(path) as im:
                sx = im.width / iw if iw else 1.0
                sy = im.height / ih if ih else 1.0
                box = (int(x * sx), int(y * sy), max(int(x * sx) + 1, int((x + w) * sx)),
                       max(int(y * sy) + 1, int((y + h) * sy)))
                buffer = _io.BytesIO()
                im.crop(box).save(buffer, format="PNG")
                out.append(buffer.getvalue())
        except Exception:
            return (None, None)
    return tuple(out)


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
        self.cleaned_view: str = ""  # display copy of the page's cleaned image (the Source view)
        self.geometry = Geometry()
        self.sizes: dict = {}
        self.tool = "pan"
        self.view = "translated"  # the desktop preview's default display mode
        self.selected: Optional[int] = None
        self.drag: Optional[dict] = None
        self.watch = JobWatch(ctx, self._on_step_end, self._on_step_change)
        self.job_ends = JobEnds(ctx, self._on_any_job_end)
        self._view_pending = False  # the translated page's lookup was refused by a running job
        self.panel: Optional[BoxPanel] = None  # the selected box's inline editor
        self.panel_tab = "translated"  # the BoxPanel tab last shown (OCR / Translated)
        self.step_progress_text = ""
        self._gen = 0
        self._editing = False  # a box edit is in flight on the io pool
        self._listening = False
        # the page viewer kept across refreshes, display-mode switches and tool changes while the
        # page stays the same, so the InteractiveViewer keeps its zoom and pan
        self._source_cache: Optional[tuple] = None  # (signature, control)
        self._surface_mode: Optional[bool] = None  # the gesture surface's mode (True: an edit tool)
        self.view_transform = ViewTransform()  # the viewer's pan / zoom (from its gestures)
        self.layout: Optional[PageLayout] = None
        self._viewport: Optional[tuple] = None  # the viewer's reported (width, height)
        self.panel_side = "bottom"  # where the box panel floats over the viewer (away from the box)
        self.nudge_step: Optional[int] = None  # the Move tab's step, kept across panels
        # moved / resized translated boxes until the page is rendered again: {index: {"origin": the
        # box's (x, y, w, h) in the render, "hole": cv.Image, "text": cv.Image}} — the text cut from
        # the render follows the box, the page under it covers where it was (instant, like the
        # desktop's overlay moving with its rectangle), then Save & Update Overlay renders it for real
        self.live: dict = {}
        self._render_token = 0

    # ---- build --------------------------------------------------------------------------------------

    def build(self) -> ft.Control:
        self._source_cache = None  # a new tree: nothing to keep in place
        self.panel = None
        self.strip = ft.ListView(controls=[], horizontal=True, height=THUMB_H + 8, spacing=4, expand=True,
                                 key="me-strip")
        # the desktop display-mode button: one button, the current state, cycling on each tap
        self.view_emoji = ft.Text(VIEWS[0][1], size=16)
        self.view_button = ft.IconButton(icon=self.view_emoji, tooltip=VIEWS[0][2],
                                         size_constraints=HIT_TARGET, key="me-view",
                                         on_click=lambda e: self.cycle_view())
        self.view_hint = ft.Text("", theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT,
                                 visible=False, key="me-view-hint")
        # the selected box's panel floats over the page (top or bottom, away from the box): the page
        # keeps its size when a box is picked
        self.panel_holder = ft.Container(left=0, right=0, bottom=0, key="me-box-panel-holder")
        self.page_label = ft.Text("", theme_style=ft.TextThemeStyle.LABEL_SMALL, key="me-page-label")
        self.prev_button = ft.IconButton(icon=ft.Icons.CHEVRON_LEFT, tooltip="Previous page", key="me-prev",
                                         size_constraints=HIT_TARGET,
                                         on_click=lambda e: self.ctx.spawn(self.step_page(-1)))
        self.next_button = ft.IconButton(icon=ft.Icons.CHEVRON_RIGHT, tooltip="Next page", key="me-next",
                                         size_constraints=HIT_TARGET,
                                         on_click=lambda e: self.ctx.spawn(self.step_page(1)))
        self.pages_row = ft.Row([self.prev_button, self.page_label, self.next_button, self.strip], spacing=0,
                                vertical_alignment=ft.CrossAxisAlignment.CENTER, key="me-pages")
        self.tool_buttons: dict = {}
        tool_controls: list = []
        for tool_id, label, icon, reason in TOOLS:
            button = ft.IconButton(icon=getattr(ft.Icons, icon, ft.Icons.EDIT), tooltip=reason or label,
                                   size_constraints=HIT_TARGET, selected=tool_id == self.tool,
                                   disabled=reason is not None, key=f"me-tool-{tool_id}",
                                   on_click=lambda e, t=tool_id: self.set_tool(t))
            self.tool_buttons[tool_id] = button
            tool_controls.append(button)
        self.exclude_button = ft.IconButton(icon=ft.Icons.BLOCK, tooltip="Exclude from Clean", key="me-exclude",
                                            size_constraints=HIT_TARGET,
                                            on_click=lambda e: self.ctx.spawn(self.toggle_exclude()))
        self.delete_button = ft.IconButton(icon=ft.Icons.DELETE_OUTLINE, tooltip="Delete box", key="me-delete",
                                           size_constraints=HIT_TARGET,
                                           on_click=lambda e: self.ctx.spawn(self.delete_selected()))
        self.clear_button = ft.IconButton(icon=ft.Icons.CLEAR_ALL, tooltip="Clear boxes", key="me-clear-boxes",
                                          size_constraints=HIT_TARGET,
                                          on_click=lambda e: self.ctx.spawn(self.clear_boxes()))
        self.share_item = ft.PopupMenuItem(content="Share or save the translated page", icon=ft.Icons.IOS_SHARE,
                                           key="me-share", on_click=lambda e: self.share_translated())
        self.more_menu = ft.PopupMenuButton(
            icon=ft.Icons.MORE_VERT, tooltip="More editor actions", key="me-more", size_constraints=HIT_TARGET,
            items=[
                self.share_item,
                ft.PopupMenuItem(),
                ft.PopupMenuItem(content="Import OCR", icon=ft.Icons.FILE_UPLOAD, key="me-import-ocr",
                                 on_click=lambda e: self.ctx.spawn(self.import_ocr())),
                ft.PopupMenuItem(content="Export OCR", icon=ft.Icons.FILE_DOWNLOAD, key="me-export-ocr",
                                 on_click=lambda e: self.ctx.spawn(self.export_ocr())),
                ft.PopupMenuItem(content="Auto-saved OCR", icon=ft.Icons.HISTORY, key="me-ocr-files",
                                 on_click=lambda e: self.ctx.spawn(self.open_ocr_files())),
                ft.PopupMenuItem(content="Files", icon=ft.Icons.PHOTO_LIBRARY, key="me-files",
                                 on_click=lambda e: self.screen.select_tab("files") if self.screen else None),
            ])
        # the preview toolbar: one row, the display-mode button first and the ⋯ last (pinned), the
        # tools between them scroll on a narrow phone
        self.tools_row = ft.Row([*tool_controls, ft.VerticalDivider(width=8), self.exclude_button,
                                 self.delete_button, self.clear_button], scroll=ft.ScrollMode.AUTO, spacing=0,
                                expand=True, key="me-tools")
        self.toolbar = ft.Row([self.view_button, self.tools_row, self.more_menu], spacing=0,
                              vertical_alignment=ft.CrossAxisAlignment.CENTER, key="me-toolbar")
        self.viewer_holder = ft.Container(left=0, top=0, right=0, bottom=0, on_size_change=self._on_viewport_size,
                                          key="me-viewer-holder")
        self.viewer_area = ft.Stack([self.viewer_holder, self.panel_holder], expand=True, key="me-viewer-area")
        self.step_buttons: dict = {}
        for step, label, icon in STEP_BUTTONS:
            self.step_buttons[step] = ft.FilledTonalButton(content=label,
                                                           icon=getattr(ft.Icons, icon, ft.Icons.PLAY_ARROW),
                                                           key=f"me-step-{step}", style=COMPACT_BUTTON,
                                                           on_click=lambda e, s=step: self.ctx.spawn(self.run_step(s)))
        self.step_status = ft.Text("", theme_style=ft.TextThemeStyle.BODY_SMALL, key="me-step-status")
        self.step_bar = ft.ProgressBar(visible=False, key="me-step-progress")
        self.stop_button = ft.TextButton(content="Stop", icon=ft.Icons.STOP, visible=False, key="me-step-stop",
                                         style=COMPACT_BUTTON, on_click=lambda e: self._stop())
        self.step_reason = ft.Container(key="me-step-reason")
        self.status_row = ft.Row([self.step_status, self.stop_button, self.step_reason], wrap=True, spacing=6,
                                 key="me-status")
        # the non-chibi Halgakos, faded, where a page will appear (owner, 2026-10-09)
        self.empty = ft.Container(
            content=ft.Column([faded_mascot(), hint_text("Add images in the Files tab, then open a page here.")],
                              horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=8),
            alignment=ft.Alignment.CENTER, padding=ft.Padding.symmetric(vertical=12), key="me-empty",
        )
        self.root = ft.Column([
            self.empty,
            self.pages_row,
            self.toolbar,
            self.view_hint,
            self.viewer_area,
            ft.Row(list(self.step_buttons.values()), scroll=ft.ScrollMode.AUTO, spacing=4, key="me-steps"),
            self.status_row,
            self.step_bar,
        ], expand=True, spacing=tokens.SPACING["xs"], key=self.key)
        self.refresh(push_now=False)
        return ft.Container(content=self.root, padding=ft.Padding.symmetric(horizontal=GUTTER, vertical=GUTTER),
                            expand=True)

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
            self.cleaned_view = await self.ctx.io(self._display_cleaned)
        except Exception:
            self.cleaned_view = ""
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
        path = str(snap.get("translated_path") or "")
        cleaned = str(snap.get("cleaned_path") or "")
        if path and cleaned and os.path.normcase(os.path.abspath(path)) == os.path.normcase(os.path.abspath(cleaned)):
            # a Clean points the session's output at the cleaned image (the desktop output viewer
            # shows it): that is the cleaned view, the translated one stays the rendered page
            path = ""
        path = path or str(snap.get("rendered_path") or "")
        if not path or not os.path.isfile(path):
            candidate = self.session.files.output_path_for(self.image_path) if self.image_path else ""
            path = candidate if candidate and os.path.isfile(candidate) else ""
        if not path:
            return ""
        try:
            return svc.display_copy(path, int(snap.get("revision") or 0), self.session.view_cache)
        except OSError:
            return path

    def _display_cleaned(self) -> str:
        """Blocking: the versioned display copy of the page's cleaned image ('' when not cleaned)."""
        snap = self.snapshot
        path = str(snap.get("cleaned_path") or "")
        if not path or not os.path.isfile(path):
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
        self.pages_row.visible = bool(files)
        self.page_label.value = f"{self.session.page_index + 1} / {len(files)}" if files else ""
        busy = self.busy
        self.prev_button.disabled = busy or not files or self.session.page_index <= 0
        self.next_button.disabled = busy or not files or self.session.page_index >= len(files) - 1
        for tool_id, button in self.tool_buttons.items():
            button.selected = tool_id == self.tool
        box_selected = has_page and self.selected is not None and 0 <= self.selected < len(self.boxes)
        for button in (self.delete_button, self.exclude_button):
            button.disabled = busy or not box_selected
        self.clear_button.disabled = busy or not has_page or not self.boxes
        self._render_view_button()
        self.share_item.disabled = not (has_page and self.translated_view)
        hint = self._view_hint_text() if has_page else ""
        self.view_hint.value = hint
        self.view_hint.visible = bool(hint)
        if has_page:
            self.viewer_holder.content = self._viewer()
        else:
            self._source_cache = None
            self.viewer_holder.content = hint_text(self.session.editor_error) if self.session.editor_error else None
        self._render_panel(box_selected)
        self._render_steps(has_page)
        if push_now:
            push(self.root)

    def _render_panel(self, box_selected: bool) -> None:
        """The selected box's ``BoxPanel`` under the page. Kept (with what is being typed) while the
        box shows the same texts and flags; rebuilt, under a new key, when they change."""
        if not box_selected:
            if self.panel is not None:
                self.panel_tab = self.panel.tab
                self.nudge_step = self.panel.step
            self.panel = None
            self.panel_holder.content = None
            self.panel_holder.visible = False
            return
        self.panel_holder.visible = True
        self._place_panel()
        index = int(self.selected)
        box = self.boxes[index]
        reasons = (self._busy_reason(), self._steps_reason())
        panel = self.panel
        if panel is not None and panel.signature == box_signature(box, index) and \
                getattr(panel, "_reasons", None) == reasons:
            panel.set_geometry(box)
            self.panel_holder.content = panel.control
            return
        typing = panel if panel is not None and \
            panel.signature == box_signature(box, index) else None  # only the step state changed
        if panel is not None:
            self.panel_tab = panel.tab
            self.nudge_step = panel.step
        panel = BoxPanel(box, index, on_action=lambda name, i, changes: self.ctx.spawn(self.box_action(name, i, changes)),
                         on_iterations=lambda i: self.open_iterations(i),
                         on_geometry=lambda i, values: self.ctx.spawn(self.set_box_geometry(i, values)),
                         on_close=self.deselect, busy_reason=reasons[0], steps_reason=reasons[1],
                         tab=self.panel_tab, **({"step": self.nudge_step} if self.nudge_step else {}))
        if typing is not None:
            panel.keep_typing(typing)
            panel.set_geometry(box)
        panel._reasons = reasons
        self.panel = panel
        self.ctx.extras["manga_box_panel"] = panel
        self.panel_holder.content = panel.control

    def _place_panel(self) -> str:
        """The box panel floats at the bottom of the viewer, or at its top when the selected box
        is shown in the viewer's lower half (it never covers the box being edited)."""
        side = "bottom"
        layout = self.layout
        if layout is not None and self.selected is not None and 0 <= self.selected < len(self.boxes):
            box = self.boxes[self.selected]
            scale = self.geometry.scale
            centre = (float(box.get("y", 0)) + float(box.get("height", 0)) / 2) * scale + layout.page_top
            _x, y = self.view_transform.to_view(0.0, centre)
            side = "top" if y > layout.viewport_h / 2 else "bottom"
        self.panel_side = side
        self.panel_holder.top = 0 if side == "top" else None
        self.panel_holder.bottom = 0 if side == "bottom" else None
        return side

    def deselect(self) -> None:
        """✕ on the box panel (and Back): no box selected."""
        if self.selected is not None:
            self.selected = None
            self.refresh()

    def _thumb(self, index: int, path: str) -> ft.Control:
        current = index == self.session.page_index
        skipped = self.is_skipped(path)
        image: ft.Control = ft.Image(src=path, width=THUMB_W, height=THUMB_H, fit=ft.BoxFit.COVER,
                                     cache_width=THUMB_W * 2, border_radius=4, gapless_playback=True)
        if skipped:  # the desktop skip marker (⏭️), dimmed like the Files tab row
            image = ft.Stack([image, ft.Container(content=ft.Text("⏭️", size=10), right=1, top=1,
                                                  bgcolor=ft.Colors.with_opacity(0.7, ft.Colors.SURFACE),
                                                  border_radius=4, padding=1)], width=THUMB_W, height=THUMB_H)
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
        """One page view on every device (the desktop preview's merged source / output view): the
        display mode only changes the image under the boxes."""
        return self._source_viewer()

    def background(self) -> str:
        """The image under the boxes for the display mode: original = the page; cleaned = its
        cleaned image (the page until it is cleaned); translated = the translated render (else the
        cleaned image, else the page)."""
        if self.view == "translated" and self.translated_view:
            return self.translated_view
        if self.view in ("translated", "cleaned") and self.cleaned_view:
            return self.cleaned_view
        return self.image_path

    def _view_hint_text(self) -> str:
        if self.view == "translated" and not self.translated_view:
            if self._view_pending:
                return "Wait for the running job: the translated page shows once it has finished"
            return "Not translated yet: run Translate (or Start in Files). Showing the " + (
                "cleaned page." if self.cleaned_view else "original page.")
        if self.view == "cleaned" and not self.cleaned_view:
            return "Not cleaned yet: run Clean. Showing the original page."
        return ""

    def share_translated(self) -> Any:
        """Share / save the translated page (the ExportSheet of the output file, not the display copy)."""
        if not self.translated_view:
            self.ctx.say("Not translated yet: run Translate (or Start in Files)")
            return None
        source = str(self.snapshot.get("translated_path") or self.snapshot.get("rendered_path") or self.translated_view)
        return export_sheet(self.ctx, source)

    def _shapes(self) -> list:
        shapes: list = []
        scale = self.geometry.scale
        if self.live and self.translated_view and self.background() == self.translated_view:
            for index, entry in self.live.items():
                if not (0 <= index < len(self.boxes)) or entry.get("text") is None:
                    continue
                ox, oy, ow, oh = entry["origin"]
                hole, text = entry["hole"], entry["text"]
                hole.x, hole.y, hole.width, hole.height = ox * scale, oy * scale, ow * scale, oh * scale
                box = self.boxes[index]
                text.x, text.y = float(box.get("x", 0)) * scale, float(box.get("y", 0)) * scale
                text.width, text.height = float(box.get("width", 0)) * scale, float(box.get("height", 0)) * scale
                shapes.extend((hole, text))
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
            if index == self.selected and self.tool == "select":
                shapes.extend(self._handle_shapes(x, y, x + w, y + h))
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

    def _handle_shapes(self, left: float, top: float, right: float, bottom: float) -> list:
        """The selected box's handles (Select / move tool only): a move icon (four arrows) on the
        top-left corner, a resize icon (a diagonal double arrow) on the bottom-right one, the same
        size on screen at every zoom."""
        r = HANDLE / 2 / max(self.view_transform.scale, 1e-6)
        fill = ft.Paint(color=SELECTED_COLOR, style=ft.PaintingStyle.FILL)
        ink = ft.Paint(color="#FFFFFF", stroke_width=max(r / 6, 0.5), style=ft.PaintingStyle.STROKE,
                       stroke_cap=ft.StrokeCap.ROUND)
        a, head = r * 0.62, r * 0.26  # arrow half-length, arrowhead size
        shapes: list = [cv.Circle(left, top, r, paint=fill), cv.Circle(right, bottom, r, paint=fill)]
        cx, cy = left, top
        move = [cv.Path.MoveTo(cx - a, cy), cv.Path.LineTo(cx + a, cy), cv.Path.MoveTo(cx, cy - a),
                cv.Path.LineTo(cx, cy + a)]
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            tx, ty = cx + dx * a, cy + dy * a
            move += [cv.Path.MoveTo(tx - dx * head - dy * head, ty - dy * head - dx * head), cv.Path.LineTo(tx, ty),
                     cv.Path.LineTo(tx - dx * head + dy * head, ty - dy * head + dx * head)]
        shapes.append(cv.Path(elements=move, paint=ink))
        cx, cy, d = right, bottom, a * 0.75
        resize = [cv.Path.MoveTo(cx - d, cy - d), cv.Path.LineTo(cx + d, cy + d),
                  cv.Path.MoveTo(cx - d + head * 1.3, cy - d), cv.Path.LineTo(cx - d, cy - d),
                  cv.Path.LineTo(cx - d, cy - d + head * 1.3),
                  cv.Path.MoveTo(cx + d - head * 1.3, cy + d), cv.Path.LineTo(cx + d, cy + d),
                  cv.Path.LineTo(cx + d, cy + d - head * 1.3)]
        shapes.append(cv.Path(elements=resize, paint=ink))
        return shapes

    def _source_viewer(self) -> ft.Control:
        """The page (the display mode's image) with its boxes, in an ``InteractiveViewer`` (pinch
        zoom, pan). Image, box canvas and gesture surface share one ``Stack`` inside the viewer, so
        the viewer's transform moves them together: the boxes stay on the image at every zoom, and
        the surface's local positions stay in the unzoomed page coordinates the ``Geometry`` maps.
        The page is always the viewer's full width (``PageLayout``: the viewer is unconstrained and
        the stack sized from the viewer's reported size), centred in a frame at least the viewer's
        height, with no boundary margin: a page that fits does not move until it is pinch-zoomed,
        a taller one pans vertically. The floating box panel never resizes it.

        Pan mode: the viewer pans and zooms, and the surface only takes long-presses (box panel):
        a pan recognizer there would win Flutter's gesture arena against the viewer's own and
        swallow one-finger panning. Edit modes: the viewer holds the zoom Pan left, and the surface
        takes taps and drags. The viewer is kept (same objects, same keys, updated in place) while
        the page stays the same, so a refresh, a display-mode switch and a tool change keep the
        zoom; only the gesture surface is swapped (under a new key) when the mode changes. A new
        page rebuilds it under new keys (Flet freezes a subtree re-created under an old key)."""
        geometry = self.geometry
        aspect = (geometry.image_w / geometry.image_h) if geometry.image_w and geometry.image_h else 0.7
        editing = self.tool != "pan"
        signature = (self.image_path, round(aspect, 6))
        cached = self._source_cache
        background = self.background()
        if cached is not None and cached[0] == signature:
            self.canvas.shapes = self._shapes()
            if self.page_image.src != background:  # another display mode: same viewer, same zoom
                self.page_image.src = background
            self._set_surface(editing)
            return cached[1]
        # every layer fills the page-sized stack: image, shapes, then the gesture surface on top
        self.canvas = cv.Canvas(shapes=[], on_resize=self._on_canvas_resize, left=0, top=0, right=0,
                                bottom=0, key=f"me-canvas-{self._gen}")
        image = ft.Image(src=background, fit=ft.BoxFit.FILL, gapless_playback=True, left=0, top=0, right=0,
                         bottom=0, key=f"me-image-{self._gen}")
        self.page_image = image
        self.stack = ft.Stack([image, self.canvas], key=f"me-stack-{self._gen}")
        self.frame = ft.Container(content=self.stack, alignment=ft.Alignment.CENTER, key=f"me-frame-{self._gen}")
        self.interactive = ft.InteractiveViewer(
            content=self.frame, constrained=False, min_scale=MIN_ZOOM, max_scale=MAX_ZOOM,
            boundary_margin=ft.Margin.all(0), interaction_update_interval=50,
            on_interaction_start=self._on_view_start, on_interaction_update=self._on_view_update,
            on_interaction_end=self._on_view_end, key=f"me-iv-{self._gen}")
        self.view_transform.reset()
        self.live.clear()
        self._apply_layout(*(self._viewport or self._estimated_viewport()), push_now=False)
        self.canvas.shapes = self._shapes()
        self._surface_mode = None
        self._set_surface(editing)
        container = ft.Container(content=self.interactive, expand=True, bgcolor=ft.Colors.SURFACE_CONTAINER_LOWEST,
                                 border_radius=tokens.RADII["field"], clip_behavior=ft.ClipBehavior.HARD_EDGE,
                                 key=f"me-source-{self._gen}")
        self._source_cache = (signature, container)
        return container

    # ---- the viewer's size and transform ----------------------------------------------------------------

    def _estimated_viewport(self) -> tuple:
        """The viewer's size before it reports it: the page's width (the editor has the full width)."""
        width = getattr(self.ctx.page, "width", None) if getattr(self.ctx, "page", None) is not None else None
        try:
            width = float(width) - 2 * GUTTER if width else DEFAULT_VIEWPORT[0]
        except (TypeError, ValueError):
            width = DEFAULT_VIEWPORT[0]
        return (max(1.0, width), DEFAULT_VIEWPORT[1])

    def _aspect(self) -> float:
        g = self.geometry
        return (g.image_w / g.image_h) if g.image_w and g.image_h else 0.7

    def _apply_layout(self, viewport_w: float, viewport_h: float, push_now: bool = True) -> PageLayout:
        """Size the page stack and its frame for a viewer of this size (full width, centred)."""
        layout = PageLayout.fit(float(viewport_w), float(viewport_h), self._aspect())
        self.layout = layout
        stack, frame = getattr(self, "stack", None), getattr(self, "frame", None)
        if stack is not None and frame is not None:
            stack.width, stack.height = layout.page_w, layout.page_h
            frame.width, frame.height = layout.viewport_w, layout.content_h
        self.view_transform.resize(layout.viewport_w, layout.viewport_h, layout.viewport_w, layout.content_h)
        if push_now and frame is not None:
            push(frame)
        return layout

    def _on_viewport_size(self, e: Any) -> None:
        width, height = float(getattr(e, "width", 0) or 0), float(getattr(e, "height", 0) or 0)
        if width <= 0 or height <= 0 or (width, height) == self._viewport:
            return
        self._viewport = (width, height)
        if getattr(self, "stack", None) is not None and self._source_cache is not None:
            layout = self._apply_layout(width, height)
            if abs(layout.page_w - self.geometry.display_w) > 0.5:  # (the canvas reports it too)
                self.geometry.display_w = layout.page_w
                self._redraw()
            if self.panel is not None and self._place_panel():
                push(self.panel_holder)

    @staticmethod
    def _point(value: Any) -> tuple:
        return (float(getattr(value, "x", 0.0) or 0.0), float(getattr(value, "y", 0.0) or 0.0))

    def _on_view_start(self, e: Any) -> None:
        self.view_transform.start(*self._point(getattr(e, "local_focal_point", None)),
                                  pointers=int(getattr(e, "pointer_count", 1) or 1))

    def _on_view_update(self, e: Any) -> None:
        self.view_transform.update(*self._point(getattr(e, "local_focal_point", None)),
                                   float(getattr(e, "scale", 1.0) or 1.0))

    def _on_view_end(self, e: Any) -> None:
        self.view_transform.end(*self._point(getattr(e, "velocity", None)))
        self._after_view_change()

    def _after_view_change(self) -> None:
        """Pan / zoom ended: the handles keep their size, the box panel moves off the box."""
        if self.selected is not None:
            self._redraw()
            if self.panel is not None:
                self._place_panel()
                push(self.panel_holder)

    def _set_surface(self, editing: bool) -> None:
        """The gesture surface for the Pan / Edit mode (a new one, under a new key, when the mode
        changes) and the viewer's own pan / zoom (Pan mode only)."""
        if self._surface_mode is editing:
            return
        self._surface_mode = editing
        handlers: dict = {"on_long_press_start": self._on_long_press}
        if editing:
            handlers.update(on_tap_down=self._on_tap, on_pan_start=self._on_pan_start,
                            on_pan_update=self._on_pan_update, on_pan_end=self._on_pan_end)
        self.gestures = ft.GestureDetector(
            content=ft.Container(bgcolor=ft.Colors.TRANSPARENT), drag_interval=DRAG_INTERVAL,
            left=0, top=0, right=0, bottom=0, key=f"me-gestures-{self._gen}", **handlers)
        self.stack.controls = [self.page_image, self.canvas, self.gestures]
        self.interactive.pan_enabled = not editing
        self.interactive.scale_enabled = not editing

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

    def cycle_view(self) -> str:
        """The display-mode button: translated -> cleaned -> original -> translated (desktop
        ``_cycle_source_display_mode``)."""
        self.set_view(next_view(self.view))
        return self.view

    def set_view(self, view: str) -> None:
        if view not in VIEW_MODES:
            return
        self.view = view
        self._sync_session_view()
        self.refresh()

    def _sync_session_view(self) -> None:
        """The display mode is also the session preview's ``source_display_mode``: the shared
        Clean-this-box result handler reads it (desktop ``_update_image_preview_with_result``:
        re-render in translated, else show the cleaned page)."""
        preview = getattr(self.session.editor, "image_preview_widget", None)
        if preview is not None:
            try:
                preview.source_display_mode = self.view
            except Exception:
                pass

    def _render_view_button(self) -> None:
        """The display-mode button shows the current mode (the desktop button's emoji, tooltip and colour)."""
        _mode, emoji, tooltip, color = VIEWS[VIEW_MODES.index(self.view)] if self.view in VIEW_MODES else VIEWS[0]
        button = self.view_button
        self.view_emoji.value = emoji
        button.tooltip = tooltip
        button.bgcolor = color

    def _session_view(self) -> str:
        """The session preview's display mode (set by the shared step code), '' when unknown."""
        preview = getattr(self.session.editor, "image_preview_widget", None)
        mode = getattr(preview, "source_display_mode", "") if preview is not None else ""
        return mode if mode in VIEW_MODES else ""

    def set_tool(self, tool: str) -> None:
        reason = {t: r for t, _l, _i, r in TOOLS}.get(tool)
        if reason:
            self.ctx.say(reason)
            return
        if self.tool == "select" and tool != "select":
            self.selected = None  # leaving Select / move ends the box being moved
        self.tool = tool
        self.drag = None
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
        """Tap in an edit tool: a box is selected; a tap beside every box keeps the selection (it
        ends with ✕, Back, another box or leaving Select / move)."""
        if not self.image_path or self.tool == "pan":
            return None
        ix, iy = self.geometry.to_image(dx, dy)
        index = hit_test(self.boxes, ix, iy)
        if index is not None:
            self.selected = index
        self.refresh()
        return self.selected

    def long_press_at(self, dx: float, dy: float) -> Optional[BoxPanel]:
        """Long-press a box (any mode, also Pan): select it; its ``BoxPanel`` floats over the page."""
        if not self.image_path:
            return None
        ix, iy = self.geometry.to_image(dx, dy)
        index = hit_test(self.boxes, ix, iy)
        if index is None:
            return None
        self.selected = index
        self.refresh()
        return self.panel

    def _on_tap(self, e: Any) -> None:
        self.tap_at(*self._local(e))

    def _on_long_press(self, e: Any) -> None:
        self.long_press_at(*self._local(e))

    def _handle_reach(self) -> float:
        """How far (image px) from a handle's centre a press still grabs it."""
        return HANDLE_REACH / max(self.geometry.scale * self.view_transform.scale, 1e-6)

    def pan_start(self, dx: float, dy: float) -> None:
        if self.tool == "pan" or not self._can_edit():
            return
        self._render_token += 1  # a pending re-render waits for this edit
        if self.tool in ("box", "circle"):
            self.drag = {"mode": "draw", "start": (dx, dy), "current": (dx, dy)}
        elif self.tool == "lasso":
            self.drag = {"mode": "lasso", "points": [(dx, dy)]}
        elif self.tool == "select":
            ix, iy = self.geometry.to_image(dx, dy)
            selected = self.selected
            if selected is not None and 0 <= selected < len(self.boxes):
                handle = handle_at(self.boxes[selected], ix, iy, self._handle_reach())
                if handle is not None:
                    self.drag = {"mode": handle, "index": selected, "last": (ix, iy)}
                    self._begin_live(selected)
                    return
            index = hit_test(self.boxes, ix, iy)
            if index is None:  # beside every box: nothing moves, the selection stays
                self.drag = None
                return
            if index != selected:
                self.selected = index
                self.refresh()
            self.drag = {"mode": "move", "index": index, "last": (ix, iy)}
            self._begin_live(index)

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
            x, y, w, h = (float(box["x"]), float(box["y"]), float(box["width"]), float(box["height"]))
            g = self.geometry
            if mode == "move":
                place_box(box, x + ix - lx, y + iy - ly, w, h, g.image_w, g.image_h)
            else:
                width, height = w + ix - lx, h + iy - ly
                place_box(box, x, y, min(width, g.image_w - x) if g.image_w else width,
                          min(height, g.image_h - y) if g.image_h else height, g.image_w, g.image_h)
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
        moved = False
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
                moved = True
        except Exception as exc:
            self.ctx.say(f"Could not change the box: {exc}")
        finally:
            self._editing = False
        if result is not None:
            self.selected = result
        await self._apply_snapshot(await self._snapshot())
        if moved:
            self._schedule_render()
        return result

    def _on_pan_start(self, e: Any) -> None:
        self.pan_start(*self._local(e))

    def _on_pan_update(self, e: Any) -> None:
        self.pan_update(*self._local(e))

    def _on_pan_end(self, e: Any) -> None:
        self.ctx.spawn(self.pan_end())

    async def set_box_geometry(self, index: int, values: dict) -> bool:
        """The Move tab: place box ``index`` at {x, y, width, height} (image px, kept on the page),
        saved like a drag (the page is re-rendered shortly after, a translated box follows at once)."""
        es = self.session.editor
        if es is None or not (0 <= index < len(self.boxes)):
            return False
        if not self._can_edit():
            self.ctx.say("Wait for the running step" if self.busy else "Open a page first")
            return False
        self._render_token += 1
        self._begin_live(index)
        box = self.boxes[index]
        g = self.geometry
        place_box(box, values.get("x", box["x"]), values.get("y", box["y"]), values.get("width", box["width"]),
                  values.get("height", box["height"]), g.image_w, g.image_h)
        if self.panel is not None and self.panel.index == index:
            self.panel.set_geometry(box)
        self._redraw()
        if not await self._box_call(lambda: es.update_box(index, box["x"], box["y"], box["width"], box["height"],
                                                          rerender=False)):
            return False
        self._schedule_render()
        return True

    # ---- the moved box's text, live ---------------------------------------------------------------------

    def _begin_live(self, index: int) -> None:
        """A translated box starts moving: cut its rendered text (and the page under it) from the
        translated render once, so the text follows the box until the page is rendered again."""
        if index in self.live or not (0 <= index < len(self.boxes)) or not self.translated_view:
            return
        box = self.boxes[index]
        if not str(box.get("translation") or "").strip():
            return
        origin = (float(box["x"]), float(box["y"]), float(box["width"]), float(box["height"]))
        entry: dict = {"origin": origin, "hole": None, "text": None}
        self.live[index] = entry
        translated = self.translated_view
        background = self.cleaned_view or self.image_path
        size = (self.geometry.image_w, self.geometry.image_h)

        async def cut() -> None:
            try:
                hole, text = await self.ctx.io(crop_patches, translated, background, origin, size)
            except Exception:
                hole = text = None
            if self.live.get(index) is not entry or hole is None or text is None:
                return
            entry["hole"] = cv.Image(src=hole, x=0, y=0, width=1, height=1)
            entry["text"] = cv.Image(src=text, x=0, y=0, width=1, height=1)
            self._redraw()

        self.ctx.spawn(cut())

    def _schedule_render(self) -> None:
        """Re-render the page (Save & Update Overlay, the desktop's rectangle-moved auto save)
        ``RENDER_DELAY`` seconds after the last move of a translated box; another edit first
        postpones it."""
        if not self.live:
            return
        self._render_token += 1
        self.ctx.spawn(self._render_later(self._render_token))

    async def _render_later(self, token: int) -> Optional[str]:
        import asyncio

        await asyncio.sleep(RENDER_DELAY)
        if token != self._render_token or self.drag or self.busy or self._editing or not self.live:
            return None
        return await self.run_step("render")

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
        self.live.clear()
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
        self.live.clear()
        return await self._box_call(lambda: es.clear_page())

    async def toggle_exclude(self) -> bool:
        index = self.selected
        es = self.session.editor
        if index is None or es is None or not (0 <= index < len(self.boxes)):
            return False
        value = not bool(self.boxes[index].get("exclude_from_clean"))
        return await self._box_call(lambda: es.set_box_excluded(index, value))

    def show_box_tab(self, tab: str, index: Optional[int] = None) -> Optional[BoxPanel]:
        """Select a box (default: the selected one) and show its panel's ``ocr`` / ``translated`` tab
        (the desktop's "📝 OCR Recognition Result" / "🌍 Translation Result")."""
        index = self.selected if index is None else index
        if index is None or not (0 <= index < len(self.boxes)):
            return None
        if self.selected != index:
            self.selected = index
            self.refresh()
        panel = self.panel
        if panel is not None:
            panel.select_tab(tab)
            self.panel_tab = panel.tab
        return panel

    def open_iterations(self, index: int) -> Any:
        """Set Inpainting Iterations (Auto or 0-50, the desktop dialog's range) for a box."""
        from glossarion_mobile.ui.components.action_sheet import ActionItem, ActionSheet

        if not (0 <= index < len(self.boxes)):
            return None
        current = self.boxes[index].get("inpaint_iterations")
        current = -1 if current in (None, "") else current

        def choose(value: int) -> Any:
            panel = self.panel
            if panel is not None and panel.index == index:
                return panel.set_flag("inpaint_iterations", value)  # with any typed translation
            return self.ctx.spawn(self.box_action("save", index, {"inpaint_iterations": value}))

        items = [ActionItem(("Auto" if v == -1 else str(v)) + (" ✓" if v == current else ""),
                            (lambda v=v: choose(v)), key=f"me-iterations-{v}") for v in ITERATION_CHOICES]
        sheet = ActionSheet(items, title=f"Box {index + 1} · Inpainting iterations",
                            tablet=bool(getattr(self.ctx, "tablet", False)))
        self.ctx.extras["manga_iterations_sheet"] = sheet
        if self.ctx.page is not None:
            sheet.show(self.ctx.page)
        return sheet

    async def box_action(self, name: str, index: int, changes: dict) -> Optional[str]:
        """BoxPanel actions: apply the edits (session calls), then the action itself."""
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
        self._sync_session_view()
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
        # the tool's own model / API key (Files › Translation model), like Start
        extra = {**dict(extra or {}), **svc.translation_params(getattr(self.ctx, "prefs", None))} or None
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
        if step in TRANSLATED_STEPS or step in ("detect", "clean"):
            self.live.clear()  # the page was rendered (or its boxes replaced): the render shows it
        elif self.live:
            self._schedule_render()  # a move waited for this step
        self.ctx.spawn(self._reload_after_step(step))

    async def _reload_after_step(self, step: str) -> None:
        await self._apply_snapshot(await self._snapshot())
        if step == "clean" and self.cleaned_view:
            self.set_view("cleaned")  # desktop _restore_clean_button: the cleaned page is shown
        elif step in TRANSLATED_STEPS and self.translated_view:
            self.set_view("translated")  # the page just translated: where the manual editing happens
        elif self._session_view() and self._session_view() != self.view:
            # the shared step code switched the preview (Clean this box outside the translated
            # view shows the cleaned page)
            self.set_view(self._session_view())
        if self.screen is not None and step in ("translate_all", "translate"):
            try:
                self.screen.files_tab.refresh()
            except Exception:
                pass

    def after_batch(self) -> Any:
        """A Files › Start batch ended: the runner wrote the pages' state and renders (this session's
        state manager), so the open page is restored from it again and, once translated, shown on
        the Translated view, like a finished Translate."""
        if not self.image_path or self.busy:
            self.refresh()
            return None
        return self.ctx.spawn(self._reopen_after_batch())

    async def _reopen_after_batch(self) -> None:
        await self.open_page(self.session.page_index)
        if self.translated_view:
            self.set_view("translated")

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
