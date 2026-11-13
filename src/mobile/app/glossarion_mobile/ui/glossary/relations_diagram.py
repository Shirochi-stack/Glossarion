"""Relationship diagram of one glossary (desktop ``relations_diagram_dialog.RelationsDiagramDialog``).

A full-screen overlay View over the glossary's relations tracker (``<glossary>_relations_tracker.json``):
``relations_tracking.build_graph`` + ``layout_graph`` drawn on a ``canvas.Canvas`` inside an
``InteractiveViewer`` (pinch zoom, pan; the manga editor's ``_source_viewer`` pattern). Edges are lines
with "A / B" at their midpoint (what B is to A, then what A is to B), amber when the relation changed;
nodes are rounded name labels tinted by gender (female pink, male blue, others grey; unlisted names
paler). Controls: the chapter slider ("Up to chapter N", the last chapter by default), the focus
character (radial layout around it), "Former relations" (earlier labels in brackets) and "Unlisted
names". Node positions come from the all-chapters graph, so nodes stay put while the slider moves.
Tapping a node opens a sheet with that character's relation history (``character_history``). A
glossary without a tracker shows an empty state.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any, Optional

import flet as ft
import flet.canvas as cv

from glossarion_mobile.ui.components.dialogs import close_dialog
from glossarion_mobile.ui.components.empty_state import EmptyState
from glossarion_mobile.ui.glossary.common import quiet, sheet
from glossarion_mobile.ui.theme import HIT_TARGET, semantic, toggle

__all__ = ["ALL_CHARACTERS", "EMPTY_BODY", "GENDER_TINTS", "RelationsDiagram", "edge_label_text"]

log = logging.getLogger("glossarion.glossary.ui")

ALL_CHARACTERS = "__all__"
EMPTY_BODY = ("Turn on Enable Relations Tracker in Glossary › General, then run a Balanced or Full glossary "
              "extraction.")
#: the desktop diagram's tints (female: the editor's tracked-gender pink; male: the dialog accent)
GENDER_TINTS = {"female": "#ec4899", "male": "#5a9fd4"}
NEUTRAL_TINT = "#6b7280"
NODE_WIDTH = 120
NODE_HEIGHT = 30


def edge_label_text(edge: Any, show_former: bool = False) -> str:
    """``"Father / Daughter"``: what b is to a, then what a is to b; earlier labels in brackets
    (``relations_tracking.edge_label_text``, shared with the desktop diagram)."""
    return _core(None).edge_label_text(edge, show_former)


def _gender_key(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text.startswith(("f", "여")):
        return "female"
    if text.startswith(("m", "남")):
        return "male"
    return ""


def _chapter_text(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value or "?")
    return str(int(number)) if number.is_integer() else f"{number:g}"


def _core(ctx: Any) -> Any:
    from glossarion_mobile.ui.glossary.relations import relations_core

    return relations_core(ctx)


class RelationsDiagram:
    def __init__(self, ctx: Any, path: Optional[str], *, tracker: Any = None, entries: Any = None) -> None:
        self.ctx = ctx
        self.path = path
        self.core = _core(ctx)
        self.tracker = tracker if isinstance(tracker, dict) else {"entries": {}}
        self.entries = entries if isinstance(entries, list) else None
        low, high = self.core.chapter_bounds(self.tracker) if self.core is not None else (None, None)
        self.low, self.high = low, high
        self.chapter = high
        self.focus: Optional[str] = None
        self.show_former = True
        self.show_unlisted = False
        self.positions: dict = {}
        self.graph: dict = {"nodes": [], "edges": []}
        self.drawn_edges: list = []
        self.history_sheet: Any = None
        count = len(self.tracker.get("entries") or {})
        self.height = float(max(520, int(150 * math.sqrt(max(count, 1)))))
        self.width = round(self.height * 1.3)
        self.view = self._build_view()
        if not self.empty:
            self._layout()
            self.rebuild(push=False)

    # ---- loading -----------------------------------------------------------------------------------

    @staticmethod
    def load(ctx: Any, path: Optional[str]) -> tuple:
        """``(tracker, glossary entries)`` of ``path`` (blocking: the io pool)."""
        core = _core(ctx)
        if core is None or not path:
            return {"entries": {}}, None
        tracker = core.load_tracker(core.tracker_path_for_glossary(path))
        entries = None
        service = getattr(ctx, "service", None)
        try:
            doc = service.open_document(path) if service is not None and os.path.isfile(path) else None
            data = getattr(doc, "current_glossary_data", None)
            entries = [e for e in data if isinstance(e, dict)] if isinstance(data, list) else None
        except Exception:
            log.debug("reading %s for the diagram failed", path, exc_info=True)
        return tracker, entries

    @classmethod
    async def open(cls, ctx: Any, path: Optional[str]) -> "RelationsDiagram":
        tracker, entries = await ctx.io(cls.load, ctx, path)
        diagram = cls(ctx, path, tracker=tracker, entries=entries)
        push = getattr(ctx, "push_overlay", None)
        if callable(push):
            push(diagram.view)
        return diagram

    @property
    def empty(self) -> bool:
        return self.core is None or not (self.tracker.get("entries") or {})

    # ---- view --------------------------------------------------------------------------------------

    def _build_view(self) -> ft.View:
        title = "Relations" + (f" · {os.path.basename(self.path)}" if self.path else "")
        appbar = ft.AppBar(title=ft.Text(title, max_lines=1, overflow=ft.TextOverflow.ELLIPSIS),
                           leading=ft.IconButton(icon=ft.Icons.CLOSE, tooltip="Close", size_constraints=HIT_TARGET,
                                                 on_click=quiet(self.close), key="rel-close"))
        if self.empty:
            self.body = EmptyState(icon="ACCOUNT_TREE_OUTLINED", title="No relations tracked for this glossary yet",
                                   body=EMPTY_BODY, key="rel-empty")
            return ft.View(route="/glossary/relations", appbar=appbar,
                           controls=[ft.SafeArea(expand=True, content=self.body)], padding=12)
        single = self.low is None or self.high is None or self.low == self.high
        low = self.low if self.low is not None else 0
        high = self.high if self.high is not None and not single else low + 1
        self.chapter_label = ft.Text(self._chapter_label(), theme_style=ft.TextThemeStyle.LABEL_LARGE,
                                     key="rel-chapter-label")
        self.slider = ft.Slider(min=low, max=high, value=self.high if self.high is not None else low,
                                divisions=max(1, int(round(high - low))), round=0, disabled=single,
                                expand=True, on_change=self._on_slider, on_change_end=self._on_slider_end,
                                key="rel-slider")
        self.focus_picker = ft.Dropdown(label="Focus", value=ALL_CHARACTERS, dense=True, expand=True,
                                        enable_filter=True, editable=True, options=self._focus_options(),
                                        on_select=lambda e: self.set_focus(self.focus_picker.value),
                                        key="rel-focus")
        self.former_switch = toggle(value=self.show_former, label="Former relations",
                                    on_change=lambda e: self.set_former(bool(e.control.value)), key="rel-former")
        self.unlisted_switch = toggle(value=self.show_unlisted, label="Unlisted names",
                                      on_change=lambda e: self.set_unlisted(bool(e.control.value)), key="rel-unlisted")
        self.summary = ft.Text("", theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT,
                               key="rel-summary")
        self.canvas = cv.Canvas(shapes=[], left=0, top=0, width=self.width, height=self.height, key="rel-canvas")
        self.stack = ft.Stack([self.canvas], width=self.width, height=self.height, key="rel-stack")
        self.viewer = ft.InteractiveViewer(content=self.stack, constrained=False, min_scale=0.25, max_scale=4,
                                           boundary_margin=ft.Margin.all(200), key="rel-viewer")
        self.body = ft.Column([
            ft.Row([self.chapter_label, self.slider], vertical_alignment=ft.CrossAxisAlignment.CENTER),
            ft.Row([self.focus_picker]),
            ft.Row([self.former_switch, self.unlisted_switch], wrap=True, spacing=12),
            self.summary,
            ft.Container(content=self.viewer, expand=True, bgcolor=ft.Colors.SURFACE_CONTAINER_LOWEST,
                         border_radius=12, clip_behavior=ft.ClipBehavior.HARD_EDGE, key="rel-frame"),
        ], spacing=6, expand=True, key="rel-body")
        return ft.View(route="/glossary/relations", appbar=appbar,
                       controls=[ft.SafeArea(expand=True, content=self.body)], padding=12)

    def _chapter_label(self) -> str:
        return f"Up to chapter {_chapter_text(self.chapter)}" if self.chapter is not None else "All chapters"

    def _all_graph(self) -> dict:
        return self.core.build_graph(self.tracker, self.entries, None, self.show_unlisted, "full")

    def _focus_options(self) -> list:
        options = [ft.DropdownOption(key=ALL_CHARACTERS, text="All characters")]
        if self.core is not None:
            for node in self._all_graph().get("nodes", []):
                options.append(ft.DropdownOption(key=node["id"], text=str(node.get("label") or node["id"])))
        return options

    def _layout(self) -> None:
        """Node positions from the all-chapters graph (stable while the slider moves)."""
        graph = self._all_graph()
        focus = self.focus if any(n["id"] == self.focus for n in graph.get("nodes", [])) else None
        self.positions = self.core.layout_graph(graph, width=self.width, height=self.height, focus=focus)

    # ---- drawing -----------------------------------------------------------------------------------

    def rebuild(self, push: bool = True) -> None:
        if self.empty:
            return
        mode = "full" if self.show_former else "current"
        self.graph = self.core.build_graph(self.tracker, self.entries, self.chapter, self.show_unlisted, mode)
        dark = bool(getattr(self.ctx, "dark", False))
        amber = semantic("warning", dark)
        line = ft.Colors.with_opacity(0.55, ft.Colors.ON_SURFACE)
        shapes: list = []
        labels: list = []
        self.drawn_edges = []
        for edge in self.graph.get("edges", []):
            a, b = self.positions.get(edge["a"]), self.positions.get(edge["b"])
            if a is None or b is None:
                continue
            color = amber if edge.get("changed") else line
            shapes.append(cv.Line(a[0], a[1], b[0], b[1],
                                  paint=ft.Paint(color=color, stroke_width=2 if edge.get("changed") else 1.4,
                                                 style=ft.PaintingStyle.STROKE)))
            labels.append(cv.Text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, edge_label_text(edge, self.show_former),
                                  style=ft.TextStyle(size=11, color=amber if edge.get("changed") else ft.Colors.ON_SURFACE),
                                  alignment=ft.Alignment.CENTER, text_align=ft.TextAlign.CENTER, max_width=180,
                                  max_lines=2, ellipsis="…"))
            self.drawn_edges.append(edge)
        self.canvas.shapes = shapes + labels  # labels over every line
        nodes = []
        for node in self.graph.get("nodes", []):
            point = self.positions.get(node["id"])
            if point is None:
                continue
            tint = GENDER_TINTS.get(_gender_key(node.get("gender")), NEUTRAL_TINT)
            listed = node.get("listed", True)
            selected = node["id"] == self.focus
            nodes.append(ft.Container(
                content=ft.Text(str(node.get("label") or node["id"]), size=12, weight=ft.FontWeight.W_600,
                                max_lines=1, overflow=ft.TextOverflow.ELLIPSIS, text_align=ft.TextAlign.CENTER,
                                color=ft.Colors.WHITE),
                left=point[0] - NODE_WIDTH / 2, top=point[1] - NODE_HEIGHT / 2, width=NODE_WIDTH, height=NODE_HEIGHT,
                alignment=ft.Alignment.CENTER, border_radius=NODE_HEIGHT / 2,
                bgcolor=ft.Colors.with_opacity(0.9 if listed else 0.45, tint),
                border=ft.Border.all(2 if selected else 1, amber if selected else tint),
                tooltip=str(node.get("raw_name") or node["id"]),
                on_click=quiet(lambda n=node: self.show_history(n["id"])), key=f"rel-node-{node['id']}"))
        self.node_controls = nodes
        self.stack.controls = [self.canvas, *nodes]
        count_nodes, count_edges = len(nodes), len(self.drawn_edges)
        self.summary.value = (f"{count_nodes} character{'s' if count_nodes != 1 else ''} · "
                              f"{count_edges} relation{'s' if count_edges != 1 else ''}")
        if push:
            self.ctx.push(self.stack, self.summary, self.chapter_label)

    def edge_count(self) -> int:
        return len(self.drawn_edges)

    def node_count(self) -> int:
        return len(getattr(self, "node_controls", []) or [])

    # ---- controls ----------------------------------------------------------------------------------

    def _on_slider(self, e: Any) -> None:
        self.chapter = e.control.value
        self.chapter_label.value = self._chapter_label()
        self.ctx.push(self.chapter_label)

    def _on_slider_end(self, e: Any) -> None:
        self.set_chapter(e.control.value)

    def set_chapter(self, value: Any) -> None:
        self.chapter = value
        if hasattr(self, "slider") and value is not None:
            self.slider.value = value
        if hasattr(self, "chapter_label"):
            self.chapter_label.value = self._chapter_label()
        self.rebuild()

    def set_focus(self, node_id: Any) -> None:
        self.focus = None if node_id in (None, "", ALL_CHARACTERS) else str(node_id)
        self._layout()
        self.rebuild()

    def set_former(self, on: bool) -> None:
        self.show_former = bool(on)
        self.rebuild()

    def set_unlisted(self, on: bool) -> None:
        self.show_unlisted = bool(on)
        self.focus_picker.options = self._focus_options()
        self._layout()
        self.rebuild()
        self.ctx.push(self.focus_picker)

    # ---- history sheet -----------------------------------------------------------------------------

    def history_rows(self, node_id: str) -> list:
        node = next((n for n in self.graph.get("nodes", []) if n["id"] == node_id), None)
        raw = (node or {}).get("raw_name") or node_id
        return self.core.character_history(self.tracker, raw)

    def show_history(self, node_id: str) -> Any:
        """The character's relation history (``character_history``) in a sheet."""
        node = next((n for n in self.graph.get("nodes", []) if n["id"] == node_id), {"label": node_id})
        controls: list = []
        for item in self.history_rows(node_id):
            name = item.get("target_translated") or item.get("target_raw") or item.get("target_key")
            spans = []
            for run in item.get("timeline", []):
                first, last = _chapter_text(run.get("from_chapter")), _chapter_text(run.get("to_chapter"))
                where = f"ch {first}" if first == last else f"ch {first}–{last}"
                spans.append(f"{run.get('label')} ({where})")
            lines = [ft.Text(f"{name}: {item.get('current') or '?'}", weight=ft.FontWeight.W_600)]
            if len(spans) > 1:
                lines.append(ft.Text(" → ".join(spans), theme_style=ft.TextThemeStyle.BODY_SMALL, selectable=True))
            elif spans:
                lines.append(ft.Text(spans[0], theme_style=ft.TextThemeStyle.BODY_SMALL,
                                     color=ft.Colors.ON_SURFACE_VARIANT))
            for blip in item.get("blips", []):
                lines.append(ft.Text(f"{'Ignored reversed' if blip.get('flip') else 'Ignored one-off'}: {blip.get('label')} (ch {_chapter_text(blip.get('chapter_num'))})",
                                     theme_style=ft.TextThemeStyle.BODY_SMALL, italic=True,
                                     color=ft.Colors.ON_SURFACE_VARIANT))
            controls.append(ft.Column(lines, spacing=2, tight=True))
        if not controls:
            controls.append(ft.Text("No relations recorded for this character.",
                                    color=ft.Colors.ON_SURFACE_VARIANT))
        page = getattr(self.ctx, "page", None)
        if self.history_sheet is not None:  # one history sheet at a time (one key)
            close_dialog(page, self.history_sheet)
        self.history_sheet = sheet(str(node.get("label") or node_id), controls, key="rel-history-sheet")
        if page is not None:
            page.show_dialog(self.history_sheet)
        return self.history_sheet

    def close(self) -> None:
        page = getattr(self.ctx, "page", None)
        if self.history_sheet is not None:
            close_dialog(page, self.history_sheet)
        pop = getattr(self.ctx, "pop_overlay", None)
        if callable(pop):
            try:
                pop()
            except Exception:
                pass
