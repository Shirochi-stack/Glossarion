"""Relations & Gender Tracker of one glossary (desktop ``relations_diagram_dialog.RelationsDiagramDialog``).

A full-screen overlay View over the glossary's relations tracker (``<glossary>_relations_tracker.json``)
and gender tracker (``<glossary>_gender_tracker.json``): ``relations_tracking.build_graph`` +
``layout_graph`` drawn on a ``canvas.Canvas`` inside an ``InteractiveViewer`` (pinch zoom, pan; the manga
editor's ``_source_viewer`` pattern). Edges are lines with their label at the midpoint
(``relations_tracking.edge_label_text``: "Father / Daughter", one side alone when only one is known, no
label when none is), amber when the relation changed; nodes are rounded name labels tinted by the gender
known at the slider's chapter (female pink, male blue, others grey; unlisted names paler). Names tracked
for gender but without relations are drawn without lines while Gender is shown.

Controls: the chapter slider ("Up to chapter N", the last chapter by default), the focus character
(radial layout around it), "Former relations" (earlier labels in brackets), "Unlisted names", and the
view filters "Relations" / "Gender" / "Hide" (a searchable checklist of names with "Show all"). The
filters are stored per book in the tracker's ``view`` key (``relations_tracking.save_view``, shared with
the desktop). Node positions come from the all-chapters graph, so nodes stay put while the slider or a
filter changes.

Tapping a node opens a sheet with that character's description at the slider's chapter, gender history
and relation history; tapping a name there opens that character's sheet. The sheet rows edit and delete
relations, history entries and descriptions; "Add relation" adds one. A long press on a node offers Add
relation, Edit description, Set gender, Hide from view and Remove from tracker. Every write is one locked
load -> modify -> write of the core (``relations_tracking.edit_tracker`` / ``set_gender``); every delete
asks first (``common.ask``); the last edits can be undone (app bar Undo, or the snackbar's Undo).

While the screen is open the three files (relations tracker, gender tracker, glossary) are checked once a
second (``file_stamp``: one ``os.stat`` each); a change is read on the io pool and redrawn, keeping the
slider, focus, filters, the open sheet's character and the zoom. A file being written is skipped until
the next check. A glossary without a tracker shows an empty state.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import math
import os
from typing import Any, Callable, Optional

import flet as ft
import flet.canvas as cv

from glossarion_mobile.ui.components.dialogs import close_dialog
from glossarion_mobile.ui.components.empty_state import EmptyState
from glossarion_mobile.ui.glossary.common import ask, quiet, sheet
from glossarion_mobile.ui.theme import HIT_TARGET, semantic, toggle

__all__ = ["ALL_CHARACTERS", "EMPTY_BODY", "GENDER_TINTS", "POLL_SECONDS", "RelationsDiagram", "TITLE",
           "edge_label_text"]

log = logging.getLogger("glossarion.glossary.ui")

TITLE = "Relations & Gender Tracker"
ALL_CHARACTERS = "__all__"
EMPTY_BODY = ("Turn on Enable Relations Tracker in Glossary › General, then run a Balanced or Full glossary "
              "extraction.")
#: the desktop diagram's tints (female: the editor's tracked-gender pink; male: the dialog accent)
GENDER_TINTS = {"female": "#ec4899", "male": "#5a9fd4"}
NEUTRAL_TINT = "#6b7280"
NODE_WIDTH = 120
NODE_HEIGHT = 30
#: the auto-update check (decision B: once a second)
POLL_SECONDS = 1.0
UNDO_LIMIT = 20
GENDER_CHOICES = (("male", "Male"), ("female", "Female"), ("auto", "Not set (automatic)"))
_SMALL = dict(theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT)


def edge_label_text(edge: Any, show_former: bool = False) -> str:
    """``"Father / Daughter"``: what b is to a, then what a is to b; one side alone when only one is
    known, "" when none is; earlier labels in brackets (``relations_tracking.edge_label_text``, shared
    with the desktop diagram)."""
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
        return str(value or "")
    return str(int(number)) if number.is_integer() else f"{number:g}"


def _span(first: Any, last: Any) -> str:
    first, last = _chapter_text(first), _chapter_text(last)
    return f"ch {first}" if first == last else f"ch {first}–{last}"


def _core(ctx: Any) -> Any:
    from glossarion_mobile.ui.glossary.relations import relations_core

    return relations_core(ctx)


def _icon_button(icon: Any, tooltip: str, handler: Callable[[], Any], key: str) -> ft.IconButton:
    return ft.IconButton(icon=icon, tooltip=tooltip, icon_size=18, size_constraints=HIT_TARGET,
                         on_click=quiet(handler), key=key)


class RelationsDiagram:
    def __init__(self, ctx: Any, path: Optional[str], *, tracker: Any = None, entries: Any = None,
                 gender: Any = None, stamps: Any = None) -> None:
        self.ctx = ctx
        self.path = path
        self.core = _core(ctx)
        self.tracker = tracker if isinstance(tracker, dict) else {"entries": {}}
        self.entries = entries if isinstance(entries, list) else None
        self.gender = gender if isinstance(gender, dict) else None
        self.tracker_path = self.core.tracker_path_for_glossary(path) if self.core is not None and path else ""
        self.gender_path = (self.core.gender_tracker_path_for_glossary(path)
                            if self.core is not None and path else "")
        #: (relations tracker, gender tracker, glossary) file stamps seen at the last load
        self.stamps = tuple(stamps) if isinstance(stamps, (tuple, list)) and len(stamps) == 3 else (None, None, None)
        self.view_choice = self._view_settings()
        low, high = self._bounds()
        self.low, self.high = low, high
        self.chapter = high
        self.focus: Optional[str] = None
        self.show_former = True
        self.show_unlisted = False
        self.positions: dict = {}
        self.graph: dict = {"nodes": [], "edges": []}
        self.all_graph: dict = {"nodes": [], "edges": []}
        self.drawn_edges: list = []
        self.node_controls: list = []
        self.history_sheet: Any = None
        self.form_sheet: Any = None  # the open edit form / menu / Hide sheet
        self.selected: Optional[str] = None  # the character whose sheet is open
        self.undo_stack: list = []  # [(kind "tracker" | "gender", edit result)]
        self.editing = 0  # edit forms / confirmations / writes of this screen in progress
        self.loading = False
        self.closed = False
        self.watch_task: Any = None
        self.task: Any = None  # the last edit started from a form (tests await it)
        self.reloads = 0
        self._layout_key: Any = None
        self.all_graph = self._all_graph() if self.core is not None else {"nodes": [], "edges": []}
        self._size()
        self.view = self._build_view()
        if not self.empty:
            self._layout()
            self.rebuild(push=False)

    # ---- loading -----------------------------------------------------------------------------------

    @staticmethod
    def read_state(ctx: Any, path: Optional[str], previous: Any = None) -> dict:
        """Read the files whose stamp moved since ``previous`` (all of them without it; blocking: the io
        pool). Returns ``{"stamps": (tracker, gender, glossary), "tracker"?, "gender"?, "entries"?}``: a
        part is present only when it was read; a file that cannot be read right now (mid-write, does not
        parse) is left out and keeps its previous stamp, so the next check tries it again."""
        core = _core(ctx)
        previous = tuple(previous) if isinstance(previous, (tuple, list)) and len(previous) == 3 else None
        old = previous or (None, None, None)
        state: dict = {"stamps": old}
        if core is None or not path:
            if previous is None:
                state.update(tracker={"entries": {}}, entries=None, gender=None)
            return state
        stamps = list(old)
        files = (core.tracker_path_for_glossary(path), core.gender_tracker_path_for_glossary(path))
        for index, (name, reader) in enumerate(zip(("tracker", "gender"), (core.read_tracker, core.read_gender_tracker))):
            stamp = core.file_stamp(files[index])
            if previous is not None and stamp == previous[index]:
                continue
            data = reader(files[index])
            if data is None or core.file_stamp(files[index]) != stamp:
                if previous is None:  # first load: show what can be shown, try the file again next time
                    state[name] = {"entries": {}} if name == "tracker" else None
                    stamps[index] = None
                continue
            state[name] = data
            stamps[index] = stamp
        stamp = core.file_stamp(path)
        if previous is None or stamp != previous[2]:
            if stamp is None:
                state["entries"] = None
                stamps[2] = None
            else:
                entries = _read_entries(ctx, path, stamp, core)
                if entries is not False:
                    state["entries"] = entries
                    stamps[2] = stamp
                elif previous is None:
                    state["entries"] = None
                    stamps[2] = None
        state["stamps"] = tuple(stamps)
        return state

    @staticmethod
    def load(ctx: Any, path: Optional[str]) -> tuple:
        """``(tracker, glossary entries)`` of ``path`` (blocking: the io pool)."""
        state = RelationsDiagram.read_state(ctx, path)
        return state.get("tracker") or {"entries": {}}, state.get("entries")

    @classmethod
    async def open(cls, ctx: Any, path: Optional[str]) -> "RelationsDiagram":
        state = await ctx.io(cls.read_state, ctx, path, None)
        diagram = cls(ctx, path, tracker=state.get("tracker"), entries=state.get("entries"),
                      gender=state.get("gender"), stamps=state.get("stamps"))
        push = getattr(ctx, "push_overlay", None)
        if callable(push):
            push(diagram.view)
        diagram.start_watching()
        return diagram

    @property
    def empty(self) -> bool:
        if self.core is None:
            return True
        entries = self.tracker.get("entries") or {}
        if any(isinstance(item, dict) and item.get("relations") for item in entries.values()):
            return False
        return not (self.gender or {}).get("entries")

    # ---- auto-update -------------------------------------------------------------------------------

    def current_stamps(self) -> tuple:
        """One ``os.stat`` per file: what the once-a-second check costs when nothing changed."""
        return tuple(self.core.file_stamp(p) if p else None for p in (self.tracker_path, self.gender_path, self.path))

    def start_watching(self) -> Any:
        if self.watch_task is None and self.path and self.core is not None and not self.closed:
            self.watch_task = self.ctx.spawn(self._watch())
        return self.watch_task

    async def _watch(self) -> None:
        try:
            while not self.closed:
                await asyncio.sleep(POLL_SECONDS)
                if self.closed:
                    break
                if self._left():  # Android back popped the View: stop watching
                    self.closed = True
                    self.watch_task = None
                    break
                try:
                    await self.check_for_update()
                except Exception:
                    log.debug("relations diagram update check failed", exc_info=True)
        except asyncio.CancelledError:
            pass

    async def check_for_update(self) -> bool:
        """Reload and redraw when one of the files changed since the last load (not while an edit form
        or a write of this screen is open, nor while the app is in the background)."""
        if self.closed or self.editing or self.loading or not self.path or self.core is None:
            return False
        foreground = getattr(self.ctx, "foreground", None)
        if callable(foreground) and not foreground():
            return False
        if self.current_stamps() == self.stamps:
            return False
        self.loading = True
        try:
            state = await self.ctx.io(self.read_state, self.ctx, self.path, self.stamps)
        finally:
            self.loading = False
        if self.closed or self.editing:
            return False
        parts = {k: state[k] for k in ("tracker", "gender", "entries") if k in state}
        self.stamps = tuple(state.get("stamps") or self.stamps)
        if not parts:
            return False
        self.reloads += 1
        self._apply_state(**parts)
        return True

    def _bounds(self) -> tuple:
        """The slider's range: the relations tracker's chapters and the gender tracker's (as on desktop)."""
        if self.core is None:
            return None, None
        return self.core.diagram_chapter_bounds(self.tracker, self.gender)

    def _left(self) -> bool:
        """True once this screen's View is off the page (Android back pops it without ``close``)."""
        shell = getattr(self.ctx, "shell", None)
        overlays = getattr(shell, "overlays", None) if shell is not None else None
        return isinstance(overlays, list) and self.view not in overlays

    def _view_settings(self) -> dict:
        if self.core is None:
            return {"hidden": [], "show_relations": True, "show_gender": True}
        return self.core.view_settings(self.tracker)

    def _apply_state(self, **parts: Any) -> None:
        """New file contents in: redraw keeping the slider (at the last chapter it stays at the new last
        one), focus, filters, the open sheet's character and the zoom (the viewer is not replaced)."""
        at_end = self.chapter is None or self.high is None or self._num(self.chapter) >= self._num(self.high)
        was_empty = self.empty
        if "tracker" in parts:
            self.tracker = parts["tracker"] if isinstance(parts["tracker"], dict) else {"entries": {}}
        if "entries" in parts:
            self.entries = parts["entries"] if isinstance(parts["entries"], list) else None
        if "gender" in parts:
            self.gender = parts["gender"] if isinstance(parts["gender"], dict) else None
        self.view_choice = self._view_settings()
        self.low, self.high = self._bounds()
        if at_end or self.chapter is None:
            self.chapter = self.high
        elif self.high is not None and self._num(self.chapter) > self._num(self.high):
            self.chapter = self.high
        elif self.low is not None and self._num(self.chapter) < self._num(self.low):
            self.chapter = self.low
        self.all_graph = self._all_graph()
        if self.empty != was_empty:
            self.safe.content = self._build_body()
            if not self.empty:
                self._layout_key = None
        if not self.empty:
            self._update_controls()
            self._size()
            self._layout(force=False)
            self.rebuild(push=False)
        self._refresh_undo(push=False)
        self.ctx.push(self.safe, self.undo_button)
        if self.selected is not None and self.history_sheet is not None and getattr(self.history_sheet, "open", False):
            if self._known(self.selected):
                self.show_history(self.selected)
            else:
                self._close_history()

    @staticmethod
    def _num(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("-inf")

    # ---- view --------------------------------------------------------------------------------------

    def _build_view(self) -> ft.View:
        name = os.path.basename(self.path) if self.path else ""
        title = ft.Column([ft.Text(TITLE, max_lines=1, overflow=ft.TextOverflow.ELLIPSIS, key="rel-title")]
                          + ([ft.Text(name, max_lines=1, overflow=ft.TextOverflow.ELLIPSIS, **_SMALL)] if name else []),
                          spacing=0, tight=True)
        self.undo_button = ft.IconButton(icon=ft.Icons.UNDO, tooltip="Undo the last edit", disabled=True,
                                         size_constraints=HIT_TARGET, on_click=quiet(self.undo), key="rel-undo")
        self.add_button = ft.IconButton(icon=ft.Icons.ADD_LINK, tooltip="Add relation", size_constraints=HIT_TARGET,
                                        disabled=not self.tracker_path,
                                        on_click=quiet(lambda: self.add_relation_form(None)), key="rel-add")
        appbar = ft.AppBar(title=title,
                           leading=ft.IconButton(icon=ft.Icons.CLOSE, tooltip="Close", size_constraints=HIT_TARGET,
                                                 on_click=quiet(self.close), key="rel-close"),
                           actions=[self.add_button, self.undo_button])
        self.safe = ft.SafeArea(expand=True, content=self._build_body())
        return ft.View(route="/glossary/relations", appbar=appbar, controls=[self.safe], padding=12)

    def _build_body(self) -> ft.Control:
        if self.empty:
            self.body = EmptyState(icon="ACCOUNT_TREE_OUTLINED", title="No relations tracked for this glossary yet",
                                   body=EMPTY_BODY, key="rel-empty")
            return self.body
        if getattr(self, "diagram_body", None) is None:
            self.diagram_body = self._build_diagram_body()
        else:
            self._update_controls()
        self.body = self.diagram_body
        return self.body

    def _slider_range(self) -> tuple:
        single = self.low is None or self.high is None or self.low == self.high
        low = self.low if self.low is not None else 0
        high = self.high if self.high is not None and not single else low + 1
        return single, low, high

    def _build_diagram_body(self) -> ft.Control:
        single, low, high = self._slider_range()
        self.chapter_label = ft.Text(self._chapter_label(), theme_style=ft.TextThemeStyle.LABEL_LARGE,
                                     key="rel-chapter-label")
        self.slider = ft.Slider(min=low, max=high, value=self.chapter if self.chapter is not None else low,
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
        view = self.view_choice
        self.relations_chip = ft.Chip(label=ft.Text("Relations"), selected=view["show_relations"],
                                      on_select=quiet(lambda: self.set_relations_shown(
                                          not self.view_choice["show_relations"])),
                                      key="rel-show-relations")
        self.gender_chip = ft.Chip(label=ft.Text("Gender"), selected=view["show_gender"],
                                   on_select=quiet(lambda: self.set_gender_shown(not self.view_choice["show_gender"])),
                                   key="rel-show-gender")
        self.hide_chip = ft.Chip(label=ft.Text(self._hide_label()), leading=ft.Icon(ft.Icons.VISIBILITY_OFF_OUTLINED),
                                 on_click=quiet(self.hide_sheet), key="rel-hide")
        self.summary = ft.Text("", theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT,
                               key="rel-summary")
        self.canvas = cv.Canvas(shapes=[], left=0, top=0, width=self.width, height=self.height, key="rel-canvas")
        self.stack = ft.Stack([self.canvas], width=self.width, height=self.height, key="rel-stack")
        self.viewer = ft.InteractiveViewer(content=self.stack, constrained=False, min_scale=0.25, max_scale=4,
                                           boundary_margin=ft.Margin.all(200), key="rel-viewer")
        return ft.Column([
            ft.Row([self.chapter_label, self.slider], vertical_alignment=ft.CrossAxisAlignment.CENTER),
            ft.Row([self.focus_picker]),
            ft.Row([self.former_switch, self.unlisted_switch], wrap=True, spacing=12),
            ft.Row([self.relations_chip, self.gender_chip, self.hide_chip], wrap=True, spacing=8, key="rel-filters"),
            self.summary,
            ft.Container(content=self.viewer, expand=True, bgcolor=ft.Colors.SURFACE_CONTAINER_LOWEST,
                         border_radius=12, clip_behavior=ft.ClipBehavior.HARD_EDGE, key="rel-frame"),
        ], spacing=6, expand=True, key="rel-body")

    def _update_controls(self) -> None:
        """Slider range, focus list and filter chips after new file contents."""
        if not hasattr(self, "slider"):
            return
        single, low, high = self._slider_range()
        self.slider.min, self.slider.max = low, high
        self.slider.divisions = max(1, int(round(high - low)))
        self.slider.disabled = single
        self.slider.value = self.chapter if self.chapter is not None else low
        self.chapter_label.value = self._chapter_label()
        self.focus_picker.options = self._focus_options()
        if self.focus is not None and not any(n["id"] == self.focus for n in self.all_graph.get("nodes", [])):
            self.focus = None
            self.focus_picker.value = ALL_CHARACTERS
        self._update_filters()

    def _update_filters(self) -> None:
        if not hasattr(self, "relations_chip"):
            return
        self.relations_chip.selected = self.view_choice["show_relations"]
        self.gender_chip.selected = self.view_choice["show_gender"]
        self.hide_chip.label = ft.Text(self._hide_label())

    def _hide_label(self) -> str:
        count = len(self.view_choice.get("hidden") or [])
        return f"Hidden ({count})" if count else "Hide"

    def _chapter_label(self) -> str:
        return f"Up to chapter {_chapter_text(self.chapter)}" if self.chapter is not None else "All chapters"

    def _all_graph(self) -> dict:
        return self.core.build_graph(self.tracker, self.entries, None, self.show_unlisted, "full",
                                     gender_tracker=self.gender, include_gender_only=True)

    def _size(self) -> None:
        # Names that have a relation (or gender tracking): an entry may hold only its descriptions.
        count = len(self.all_graph.get("nodes") or [])
        self.height = float(max(520, int(150 * math.sqrt(max(count, 1)))))
        self.width = round(self.height * 1.3)
        if hasattr(self, "canvas"):
            self.canvas.width = self.stack.width = self.width
            self.canvas.height = self.stack.height = self.height

    def _focus_options(self) -> list:
        options = [ft.DropdownOption(key=ALL_CHARACTERS, text="All characters")]
        for node in self.all_graph.get("nodes", []):
            options.append(ft.DropdownOption(key=node["id"], text=str(node.get("label") or node["id"])))
        return options

    def _layout(self, force: bool = True) -> None:
        """Node positions from the all-chapters graph (stable while the slider and the filters move); an
        update that leaves the names and pairs as they were keeps the positions."""
        graph = self.all_graph = self._all_graph()
        focus = self.focus if any(n["id"] == self.focus for n in graph.get("nodes", [])) else None
        key = (tuple(sorted(n["id"] for n in graph.get("nodes", []))),
               tuple(sorted((e["a"], e["b"]) for e in graph.get("edges", []))), focus, self.width, self.height)
        if not force and key == self._layout_key and self.positions:
            return
        self._layout_key = key
        # a node is drawn centred on its point, so the layout gets the canvas minus half a node on every
        # side: a node at the layout's edge would otherwise be cut off by the Stack
        pad_x, pad_y = NODE_WIDTH / 2 + 8, NODE_HEIGHT / 2 + 8
        inner = self.core.layout_graph(graph, width=max(self.width - 2 * pad_x, 1), height=max(self.height - 2 * pad_y, 1),
                                       focus=focus)
        self.positions = {key: (point[0] + pad_x, point[1] + pad_y) for key, point in inner.items()}

    # ---- drawing -----------------------------------------------------------------------------------

    def rebuild(self, push: bool = True) -> None:
        if self.empty:
            return
        mode = "full" if self.show_former else "current"
        view = self.view_choice
        self.graph = self.core.build_graph(self.tracker, self.entries, self.chapter, self.show_unlisted, mode,
                                           gender_tracker=self.gender, include_gender_only=True,
                                           hidden=view["hidden"], show_relations=view["show_relations"],
                                           show_gender=view["show_gender"])
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
            text = edge_label_text(edge, self.show_former)
            if text:  # neither side known: no label
                labels.append(cv.Text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, text,
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
                on_click=quiet(lambda n=node: self.show_history(n["id"])),
                on_long_press=quiet(lambda n=node: self.node_menu(n["id"])), key=f"rel-node-{node['id']}"))
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
        self.all_graph = self._all_graph()
        self.focus_picker.options = self._focus_options()
        self._layout()
        self.rebuild()
        self.ctx.push(self.focus_picker)

    # ---- view filters (stored per book in the tracker's "view" key) --------------------------------

    def set_relations_shown(self, on: bool) -> Any:
        return self.save_view(show_relations=bool(on))

    def set_gender_shown(self, on: bool) -> Any:
        return self.save_view(show_gender=bool(on))

    def set_hidden(self, node_id: str, hide: bool = True) -> Any:
        hidden = [h for h in self.view_choice["hidden"] if h != node_id] + ([node_id] if hide else [])
        return self.save_view(hidden=hidden)

    def save_view(self, **choice: Any) -> Any:
        """Show the choice now, then store it in the tracker file (one locked write of the ``view`` key;
        its stamp is kept, so the update check does not redraw this write again)."""
        # A copy: the tracker object may be an undo record's "after" snapshot, which must not take in
        # this view choice (undoing that edit would then drop it).
        self.tracker = copy.deepcopy(self.tracker)
        self.core.set_view(self.tracker, **choice)
        self.view_choice = self._view_settings()
        self._update_filters()
        self.rebuild(push=False)
        self.ctx.push(self.stack, self.summary, getattr(self, "relations_chip", None),
                      getattr(self, "gender_chip", None), getattr(self, "hide_chip", None))
        if self.selected is not None and self.history_sheet is not None and getattr(self.history_sheet, "open", False):
            self.show_history(self.selected)
        if not self.tracker_path:
            return None
        self.task = self.ctx.spawn(self._write_view(choice))
        return self.task

    async def _write_view(self, choice: dict) -> Any:
        prior = self.stamps[0]
        core, path = self.core, self.tracker_path

        def write() -> tuple:
            before = core.file_stamp(path)
            return before, core.save_view(path, **choice)

        self.editing += 1
        try:
            before, result = await self.ctx.io(write)
        except Exception as exc:
            log.debug("saving the diagram view failed", exc_info=True)
            self.ctx.say(f"Could not save the view: {exc}")
            return None
        finally:
            self.editing -= 1
        if before != prior:  # someone else wrote the file since the last load: show their changes too
            self._apply_state(tracker=result["tracker"])
        else:
            self.tracker = result["tracker"]
            self.view_choice = self._view_settings()
        self.stamps = (result.get("stamp"), self.stamps[1], self.stamps[2])
        return result

    def hide_sheet(self) -> Any:
        """A searchable checklist of names: a checked name is hidden from the diagram; "Show all"."""
        names = list(self.all_graph.get("nodes", []))
        ids = {n["id"] for n in names}
        names += [self._node_info(h) for h in self.view_choice["hidden"] if h not in ids]
        boxes: list = []

        def on_box(e: Any) -> None:
            self.set_hidden(e.control.data, bool(e.control.value))

        for node in names:
            boxes.append(ft.Checkbox(label=str(node.get("label") or node["id"]), data=node["id"],
                                     value=node["id"] in self.view_choice["hidden"], on_change=quiet_event(on_box),
                                     key=f"rel-hide-{node['id']}"))
        listing = ft.Column(boxes, spacing=0, tight=True, key="rel-hide-list")

        def on_search(e: Any) -> None:
            text = str(search.value or "").strip().casefold()
            for box in boxes:
                box.visible = not text or text in str(box.label).casefold() or text in str(box.data).casefold()
            self.ctx.push(listing)

        def show_all() -> None:
            for box in boxes:
                box.value = False
            self.ctx.push(listing)
            self.save_view(hidden=[])

        search = ft.TextField(label="Search names", dense=True, on_change=quiet_event(on_search), key="rel-hide-search")
        actions = [ft.TextButton(content="Show all", on_click=quiet(show_all), key="rel-hide-all"),
                   ft.FilledButton(content="Done", on_click=quiet(self._close_form), key="rel-hide-done")]
        return self._open_form("Hide from view", [search, listing], actions, key="rel-hide-sheet", editing=False)

    # ---- names -------------------------------------------------------------------------------------

    def _node_info(self, node_id: Any) -> dict:
        node_id = str(node_id or "")
        for graph in (self.graph, self.all_graph):
            for node in graph.get("nodes", []):
                if node["id"] == node_id:
                    return node
        entries = self.tracker.get("entries") or {}
        item = entries.get(node_id)
        if isinstance(item, dict):
            return {"id": node_id, "raw_name": str(item.get("raw_name") or node_id),
                    "label": str(item.get("translated_name") or item.get("raw_name") or node_id)}
        for name in self._known_names():
            if name["id"] == node_id:
                return dict(name)
        return {"id": node_id, "raw_name": node_id, "label": node_id}

    def _known(self, node_id: str) -> bool:
        return any(name["id"] == node_id for name in self._known_names())

    def _known_names(self) -> list:
        return self.core.known_names(self.tracker, self.entries, self.gender) if self.core is not None else []

    def _raw(self, node_id: str) -> str:
        return str(self._node_info(node_id).get("raw_name") or node_id)

    def _label(self, node_id: str) -> str:
        return str(self._node_info(node_id).get("label") or node_id)

    def _glossary_gender(self, raw: str) -> str:
        key = self.core.tracker_key(raw)
        for entry in self.entries or []:
            if self.core.tracker_key(entry.get("raw_name")) == key:
                return str(entry.get("gender") or "")
        return ""

    def _resolve_name(self, text: str, preset: Optional[dict] = None) -> tuple:
        """``(raw_name, translated)`` of a typed name: a known name by its label, raw name or key; else a
        new name."""
        text = str(text or "").strip()
        if not text:
            return "", ""
        fold = text.casefold()
        if preset and fold in (str(preset.get("label") or "").casefold(), str(preset.get("raw_name") or "").casefold()):
            return str(preset.get("raw_name") or text), str(preset.get("label") or "")
        key = self.core.tracker_key(text)
        for name in self._known_names():
            if name["id"] == key or fold in (name["label"].casefold(), name["raw_name"].casefold()):
                label = name["label"] if name["label"] != name["raw_name"] else ""
                return name["raw_name"], label
        return text, ""

    # ---- history sheet -----------------------------------------------------------------------------

    def history_rows(self, node_id: str) -> list:
        # Like the picture, the history stops at the slider's chapter.
        return self.core.character_history(self.tracker, self._raw(node_id), self.chapter)

    def description_rows(self, node_id: str) -> tuple:
        """``(written, earlier, source)``: the description written at the slider's chapter
        (core ``descriptions_shown``, shared with the desktop diagram)."""
        return self.core.descriptions_shown(self.tracker, self._raw(node_id), self.chapter, self.entries)

    def gender_rows(self, node_id: str) -> dict:
        """The character's gender history up to the slider's chapter (core ``gender_history``)."""
        raw = self._raw(node_id)
        return self.core.gender_history(self.gender, raw, self.chapter, glossary_gender=self._glossary_gender(raw))

    def _description_controls(self, node_id: str) -> list:
        written, earlier, source = self.description_rows(node_id)
        if written is None:
            if source != "chapter":
                return []
            return [ft.Text(f"No description written up to chapter {_chapter_text(self.chapter)}.", **_SMALL)]
        heading = (f"Description written at chapter {_chapter_text(written.get('chapter_num'))}"
                   if source == "chapter" else "Description (from the glossary; not recorded per chapter yet)")
        lines = [ft.Text(heading, **_SMALL), ft.Text(written["text"], selectable=True)]
        for old in earlier:
            lines.append(ft.Text(f"ch {_chapter_text(old.get('chapter_num'))}: {old['text']}", selectable=True, **_SMALL))
        return [ft.Column(lines, spacing=2, tight=True, key="rel-history-description")]

    def _description_edit_row(self, node_id: str) -> list:
        """Edit / delete buttons for every description entry shown (written and earlier ones)."""
        if not self.tracker_path:
            return []
        written, earlier, source = self.description_rows(node_id)
        buttons: list = []
        if source == "chapter" and written is not None:
            for index, item in enumerate([written] + list(earlier)):
                where = f"ch {_chapter_text(item.get('chapter_num'))}"
                buttons.append(ft.Row([
                    ft.Text(f"Description, {where}", expand=True, **_SMALL),
                    _icon_button(ft.Icons.EDIT_OUTLINED, f"Edit the description of {where}",
                                 lambda d=item: self.description_form(node_id, d), f"rel-desc-edit-{index}"),
                    _icon_button(ft.Icons.DELETE_OUTLINE, f"Delete the description of {where}",
                                 lambda d=item: self.delete_description(node_id, d), f"rel-desc-delete-{index}"),
                ], spacing=0, tight=True))
        return buttons

    def _gender_controls(self, node_id: str) -> list:
        if not self.view_choice["show_gender"]:
            return []
        text = self.core.gender_history_text(self.gender_rows(node_id)) or "Not known"
        row = [ft.Text(f"Gender: {text}", expand=True, selectable=True, key="rel-history-gender")]
        if self.path:
            row.append(_icon_button(ft.Icons.EDIT_OUTLINED, "Set gender", lambda: self.gender_form(node_id),
                                    "rel-gender-edit"))
        return [ft.Row(row, spacing=0, vertical_alignment=ft.CrossAxisAlignment.CENTER)]

    def _relation_controls(self, node_id: str) -> list:
        raw = self._raw(node_id)
        own = self._label(node_id)
        editable = bool(self.tracker_path)
        controls: list = []
        for item in self.history_rows(node_id):
            tkey = item.get("target_key")
            name = item.get("target_translated") or item.get("target_raw") or tkey
            current = item.get("current") or ""
            title = ft.TextButton(content=f"{name}: {current}" if current else str(name),
                                  tooltip=f"Open {name}", on_click=quiet(lambda k=tkey: self.show_history(k)),
                                  key=f"rel-open-{tkey}")
            head = [ft.Container(content=title, expand=True, alignment=ft.Alignment.CENTER_LEFT)]
            if editable:
                head += [_icon_button(ft.Icons.EDIT_OUTLINED, "Edit relation",
                                      lambda k=tkey: self.relation_form(node_id, k), f"rel-edit-{tkey}"),
                         _icon_button(ft.Icons.DELETE_OUTLINE, "Delete relation",
                                      lambda k=tkey: self.delete_relation(node_id, k), f"rel-delete-{tkey}")]
            lines: list = [ft.Row(head, spacing=0, vertical_alignment=ft.CrossAxisAlignment.CENTER)]
            label_ab, label_ba = self.core.relation_labels(self.tracker, raw, item.get("target_raw") or tkey, self.chapter)
            edge = {"a": node_id, "b": tkey, "label_ab": label_ab or current, "label_ba": label_ba}
            for text in self.core.edge_description_lines(edge, lambda k: own if k == node_id else name):
                lines.append(ft.Text(text, **_SMALL))
            for index, run in enumerate(item.get("timeline", [])):
                lines.append(self._history_row(f"{run.get('label')} ({_span(run.get('from_chapter'), run.get('to_chapter'))})",
                                               node_id, tkey, run, f"{tkey}-{index}", editable))
            for index, blip in enumerate(item.get("blips", [])):
                kind = "Ignored reversed" if blip.get("flip") else "Ignored one-off"
                text = f"{kind}: {blip.get('label')} (ch {_chapter_text(blip.get('chapter_num'))})"
                run = {"label": blip.get("label"), "from_chapter": blip.get("chapter_num"),
                       "to_chapter": blip.get("chapter_num"), "chapter_file": blip.get("chapter_file", "")}
                lines.append(self._history_row(text, node_id, tkey, run, f"{tkey}-blip-{index}", editable, italic=True))
            controls.append(ft.Column(lines, spacing=2, tight=True, key=f"rel-relation-{tkey}"))
        return controls

    def _history_row(self, text: str, node_id: str, tkey: str, run: dict, key: str, editable: bool,
                     italic: bool = False) -> ft.Control:
        row: list = [ft.Text(text, expand=True, selectable=True, italic=italic, **_SMALL)]
        if editable:
            row += [_icon_button(ft.Icons.EDIT_OUTLINED, "Edit this label",
                                 lambda: self.label_form(node_id, tkey, run), f"rel-run-edit-{key}"),
                    _icon_button(ft.Icons.DELETE_OUTLINE, "Delete this history entry",
                                 lambda: self.delete_history(node_id, tkey, run), f"rel-run-delete-{key}")]
        return ft.Row(row, spacing=0, vertical_alignment=ft.CrossAxisAlignment.CENTER)

    def show_history(self, node_id: str) -> Any:
        """The character's description at the slider's chapter, gender history and relation history,
        with edit / delete buttons, in a sheet."""
        node_id = str(node_id)
        same = self.selected == node_id
        self.selected = node_id
        controls: list = self._description_controls(node_id) + self._description_edit_row(node_id)
        controls += self._gender_controls(node_id)
        if self.view_choice["show_relations"]:
            relations = self._relation_controls(node_id)
            if not relations:
                upto = f" up to chapter {_chapter_text(self.chapter)}" if self.chapter is not None else ""
                relations = [ft.Text(f"No relations recorded for this character{upto}.",
                                     color=ft.Colors.ON_SURFACE_VARIANT)]
            controls += relations
        actions = []
        if self.tracker_path:
            actions = [ft.TextButton(content="More…", icon=ft.Icons.MORE_HORIZ,
                                     on_click=quiet(lambda: self.node_menu(node_id)), key="rel-history-more"),
                       ft.FilledButton(content="Add relation", icon=ft.Icons.ADD_LINK,
                                       on_click=quiet(lambda: self.add_relation_form(node_id)), key="rel-history-add")]
        page = getattr(self.ctx, "page", None)
        self._form_seq = getattr(self, "_form_seq", 0) + 1
        fresh = sheet(self._label(node_id), controls, actions=actions, key=f"rel-history-sheet-{self._form_seq}")
        current = self.history_sheet
        if same and current is not None and getattr(current, "open", False):
            # the same character after an edit or an update: the open sheet's content is replaced
            current.content = fresh.content
            self.ctx.push(current)
            return current
        if current is not None:  # one history sheet at a time
            close_dialog(page, current)
        self.history_sheet = mine = fresh

        def dismissed(e: Any = None) -> None:
            if self.history_sheet is mine:
                self.selected = None

        mine.on_dismiss = dismissed
        if page is not None:
            page.show_dialog(mine)
        return mine

    def _close_history(self) -> None:
        if self.history_sheet is not None:
            close_dialog(getattr(self.ctx, "page", None), self.history_sheet)
        self.selected = None

    # ---- forms -------------------------------------------------------------------------------------

    def _open_form(self, title: str, controls: list, actions: list, *, key: str, editing: bool = True) -> Any:
        """A sheet over the diagram; while an edit form is open the update check waits."""
        self._close_form()
        # A new key each time: Flet freezes a dialog shown under the key of one whose dismiss has not
        # arrived yet, and a frozen form's fields cannot be filled in (the name suggestions do that).
        self._form_seq = getattr(self, "_form_seq", 0) + 1
        form = sheet(title, controls, actions=actions, key=f"{key}-{self._form_seq}")
        state = {"open": True}

        def finished(e: Any = None) -> None:
            if state["open"]:
                state["open"] = False
                if editing:
                    self.editing = max(0, self.editing - 1)

        form.data = finished
        form.on_dismiss = finished
        if editing:
            self.editing += 1
        self.form_sheet = form
        page = getattr(self.ctx, "page", None)
        if page is not None:
            page.show_dialog(form)
        return form

    def _close_form(self) -> None:
        form, self.form_sheet = self.form_sheet, None
        if form is None:
            return
        close_dialog(getattr(self.ctx, "page", None), form)
        if callable(getattr(form, "data", None)):
            form.data()

    def _form_actions(self, submit: Callable[[], Any], error: ft.Text, prefix: str, save: str = "Save") -> list:
        """Cancel / Save: ``submit()`` returns an error text (the form stays open) or a coroutine (the
        form closes and the edit runs)."""

        def on_save() -> None:
            outcome = submit()
            if isinstance(outcome, str):
                error.value = outcome
                error.visible = True
                self.ctx.push(error)
                return
            self._close_form()
            if asyncio.iscoroutine(outcome):
                self.task = self.ctx.spawn(outcome)

        return [ft.TextButton(content="Cancel", on_click=quiet(self._close_form), key=f"{prefix}-cancel"),
                ft.FilledButton(content=save, on_click=quiet(on_save), key=f"{prefix}-save")]

    @staticmethod
    def _error_text(key: str) -> ft.Text:
        return ft.Text("", color=ft.Colors.ERROR, visible=False, key=key)

    def _chapter_value(self, text: Any) -> tuple:
        """``(ok, chapter)`` of a chapter field ('' -> the slider's chapter's absence: the last one)."""
        text = str(text or "").strip()
        if not text:
            return True, None
        try:
            number = float(text)
        except ValueError:
            return False, None
        return True, int(number) if number.is_integer() else number

    def relation_form(self, node_id: str, target: str) -> Any:
        """Edit both directions of a relation at the slider's chapter; a side may be cleared."""
        raw, target_raw = self._raw(node_id), self._raw(target)
        own, other = self._label(node_id), self._label(target)
        old_ab, old_ba = self.core.relation_labels(self.tracker, raw, target_raw, self.chapter)
        field_ab = ft.TextField(label=f"What {other} is to {own}", value=old_ab, dense=True, key="rel-form-ab")
        field_ba = ft.TextField(label=f"What {own} is to {other}", value=old_ba, dense=True, key="rel-form-ba")
        error = self._error_text("rel-form-error")
        chapter = self.chapter

        def submit() -> Any:
            new_ab, new_ba = str(field_ab.value or "").strip(), str(field_ba.value or "").strip()
            if new_ab == old_ab and new_ba == old_ba:
                return "Nothing changed."
            change_ab = new_ab if new_ab != old_ab else None
            change_ba = new_ba if new_ba != old_ba else None
            cleared = [text for text, new, old in ((f"what {other} is to {own}", change_ab, old_ab),
                                                   (f"what {own} is to {other}", change_ba, old_ba))
                       if new == "" and old]

            async def run() -> Any:
                if cleared and not await self._confirm(
                        "Clear relation side", f"Clear {' and '.join(cleared)}? Later extractions will not add "
                                               "that label again."):
                    return None
                return await self.edit(
                    lambda t: self.core.edit_relation(t, raw, target_raw, change_ab, change_ba, old_ab=old_ab,
                                                      old_ba=old_ba, chapter=chapter, translated_a=own,
                                                      translated_b=other),
                    f"Relation between {own} and {other} saved")

            return run()

        hint = ft.Text("Leave a side empty to clear it.", **_SMALL)
        return self._open_form(f"{own} – {other}", [field_ab, field_ba, hint, error],
                               self._form_actions(submit, error, "rel-form"), key="rel-form-sheet")

    def label_form(self, node_id: str, target: str, run: dict) -> Any:
        """Rename one label of a direction (every record of it: ``set_relation_label``)."""
        raw, target_raw = self._raw(node_id), self._raw(target)
        own, other = self._label(node_id), self._label(target)
        old = str(run.get("label") or "")
        field = ft.TextField(label=f"What {other} is to {own}", value=old, dense=True, key="rel-label-value")
        error = self._error_text("rel-label-error")

        def submit() -> Any:
            new = str(field.value or "").strip()
            if new == old:
                return "Nothing changed."
            if not new:
                return "Type a label, or delete the entry instead."
            return self.edit(lambda t: self.core.set_relation_label(t, raw, target_raw, new, old_label=old,
                                                                    chapter=self.chapter, translated_name=own,
                                                                    target_translated=other),
                             f"“{old}” is now “{new}”")

        return self._open_form("Edit label", [field, error], self._form_actions(submit, error, "rel-label"),
                               key="rel-label-sheet")

    def add_relation_form(self, node_id: Optional[str]) -> Any:
        """Add a relation: the two characters (a known name picked from the suggestions, or a new one), the
        label of one or both directions and the chapter (default: the slider's)."""
        preset = self._node_info(node_id) if node_id else None
        names = self._known_names()
        fields: dict = {}
        error = self._error_text("rel-add-error")

        def suggestions(which: str) -> ft.Row:
            return ft.Row([], wrap=True, spacing=4, run_spacing=4, visible=False, key=f"rel-add-suggest-{which}")

        rows = {"a": suggestions("a"), "b": suggestions("b")}

        def relabel() -> None:
            a = str(fields["a"].value or "").strip() or "the first character"
            b = str(fields["b"].value or "").strip() or "the other character"
            fields["ab"].label = f"What {b} is to {a}"
            fields["ba"].label = f"What {a} is to {b}"

        def pick(which: str, name: dict) -> None:
            fields[which].value = name["label"]
            rows[which].visible = False
            relabel()
            self.ctx.push(fields[which], rows[which], fields["ab"], fields["ba"])

        def suggest(which: str) -> None:
            text = str(fields[which].value or "").strip().casefold()
            matches = [] if not text else [
                n for n in names if (text in n["label"].casefold() or text in n["raw_name"].casefold())
                and text not in (n["label"].casefold(), n["raw_name"].casefold())][:6]
            rows[which].controls = [ft.Chip(label=ft.Text(n["label"]), on_click=quiet(lambda w=which, n=n: pick(w, n)),
                                            key=f"rel-add-pick-{which}-{n['id']}") for n in matches]
            rows[which].visible = bool(matches)
            relabel()
            self.ctx.push(rows[which], fields["ab"], fields["ba"])

        fields["a"] = ft.TextField(label="Character", value=str(preset["label"]) if preset else "", dense=True,
                                   on_change=quiet_event(lambda e: suggest("a")), key="rel-add-name-a")
        fields["b"] = ft.TextField(label="Other character (pick or type a new name)", dense=True,
                                   on_change=quiet_event(lambda e: suggest("b")), key="rel-add-name-b")
        fields["ab"] = ft.TextField(dense=True, key="rel-add-label-ab")
        fields["ba"] = ft.TextField(dense=True, key="rel-add-label-ba")
        fields["chapter"] = ft.TextField(label="Chapter", value=_chapter_text(self.chapter) if self.chapter is not None else "",
                                         dense=True, keyboard_type=ft.KeyboardType.NUMBER, key="rel-add-chapter")
        relabel()
        self.add_fields = fields

        def submit() -> Any:
            raw_a, tr_a = self._resolve_name(fields["a"].value, preset)
            raw_b, tr_b = self._resolve_name(fields["b"].value)
            label_ab, label_ba = str(fields["ab"].value or "").strip(), str(fields["ba"].value or "").strip()
            if not raw_a or not raw_b:
                return "Pick or type both characters."
            if self.core.tracker_key(raw_a) == self.core.tracker_key(raw_b):
                return "Pick two different characters."
            if not label_ab and not label_ba:
                return "Type the label of at least one direction."
            ok, chapter = self._chapter_value(fields["chapter"].value)
            if not ok:
                return "The chapter must be a number."
            own, other = tr_a or raw_a, tr_b or raw_b
            return self.edit(lambda t: self.core.add_relation(t, raw_a, raw_b, label_ab or None, label_ba or None,
                                                              chapter=chapter, translated_name=tr_a,
                                                              target_translated=tr_b),
                             f"Relation between {own} and {other} added" + self._unlisted_note(raw_a, raw_b))

        controls = [fields["a"], rows["a"], fields["b"], rows["b"], fields["ab"], fields["ba"], fields["chapter"], error]
        return self._open_form("Add relation", controls, self._form_actions(submit, error, "rel-add", "Add"),
                               key="rel-add-sheet")

    def _unlisted_note(self, *raws: str) -> str:
        if self.entries is None or self.show_unlisted:
            return ""
        listed = {self.core.tracker_key(e.get("raw_name")) for e in self.entries}
        missing = [r for r in raws if self.core.tracker_key(r) not in listed]
        return f" (turn on Unlisted names to see {', '.join(missing)})" if missing else ""

    def description_form(self, node_id: str, item: Optional[dict] = None) -> Any:
        """Edit the description written at a chapter (default: the one shown, else the slider's chapter);
        an empty text deletes it."""
        raw, own = self._raw(node_id), self._label(node_id)
        recorded = item is not None
        if item is None:
            written, _earlier, source = self.description_rows(node_id)
            recorded = source == "chapter" and written is not None
            # nothing recorded yet: start from the glossary's text, written at the slider's chapter
            item = written if recorded else {"text": (written or {}).get("text", ""), "chapter_num": self.chapter,
                                             "chapter_file": ""}
        chapter, file = item.get("chapter_num"), item.get("chapter_file") or ""
        old = str(item.get("text") or "")
        where = f"chapter {_chapter_text(chapter)}" if chapter is not None else "the last chapter"
        field = ft.TextField(label=f"Description at {where}", value=old, multiline=True, min_lines=2, max_lines=8,
                             key="rel-desc-text")
        error = self._error_text("rel-desc-error")

        def submit() -> Any:
            text = " ".join(str(field.value or "").split())
            if recorded and text == " ".join(old.split()):
                return "Nothing changed."
            if not text:
                if not recorded:
                    return "Type a description."
                return self.delete_description(node_id, item, spawn=False)
            return self.edit(lambda t: self.core.set_description(t, raw, text, chapter, chapter_file=file,
                                                                 translated_name=own),
                             f"Description of {own} saved")

        return self._open_form(f"{own}: description", [field, error], self._form_actions(submit, error, "rel-desc"),
                               key="rel-desc-sheet")

    def gender_form(self, node_id: str) -> Any:
        """Set the gender for the slider's chapter or for all chapters (the gender tracker's own writer)."""
        raw, own = self._raw(node_id), self._label(node_id)
        history = self.gender_rows(node_id)
        current = history.get("current") or "auto"
        choice = ft.Dropdown(label="Gender", value=current if current in ("male", "female") else "auto", dense=True,
                             options=[ft.DropdownOption(key=k, text=t) for k, t in GENDER_CHOICES], key="rel-gender-value")
        chapter = self.chapter
        every = toggle(value=chapter is None or history.get("source") == "decision",
                       label="All chapters" if chapter is None else f"All chapters (off: chapter {_chapter_text(chapter)} only)",
                       disabled=chapter is None, key="rel-gender-all")
        error = self._error_text("rel-gender-error")

        def submit() -> Any:
            value = str(choice.value or "auto")
            scope = None if (every.value or chapter is None) else chapter

            async def run() -> Any:
                if value == "auto" and scope is not None and not await self._confirm(
                        "Clear gender", f"Delete the gender recorded for {own} at chapter {_chapter_text(scope)}?"):
                    return None
                where = "all chapters" if scope is None else f"chapter {_chapter_text(scope)}"
                return await self.edit_gender(raw, value, scope, own, f"Gender of {own} set for {where}")

            return run()

        return self._open_form(f"{own}: gender", [choice, every, error],
                               self._form_actions(submit, error, "rel-gender"), key="rel-gender-sheet")

    def node_menu(self, node_id: str) -> Any:
        """Long press on a name: Add relation, Edit description, Set gender, Hide from view, Remove."""
        node_id = str(node_id)
        own = self._label(node_id)

        def item(icon: Any, text: str, handler: Callable[[], Any], key: str, enabled: bool = True) -> ft.ListTile:
            def run() -> Any:
                self._close_form()
                return handler()

            return ft.ListTile(leading=ft.Icon(icon), title=ft.Text(text), disabled=not enabled,
                               on_click=quiet(run), key=key)

        editable = bool(self.tracker_path)
        rows = [item(ft.Icons.ADD_LINK, "Add relation…", lambda: self.add_relation_form(node_id), "rel-menu-add", editable),
                item(ft.Icons.NOTES, "Edit description…", lambda: self.description_form(node_id), "rel-menu-description",
                     editable),
                item(ft.Icons.WC, "Set gender…", lambda: self.gender_form(node_id), "rel-menu-gender", bool(self.path)),
                item(ft.Icons.VISIBILITY_OFF_OUTLINED, "Hide from view", lambda: self.set_hidden(node_id, True),
                     "rel-menu-hide"),
                item(ft.Icons.PERSON_REMOVE_OUTLINED, "Remove from tracker…", lambda: self.remove_character(node_id),
                     "rel-menu-remove", editable)]
        return self._open_form(own, rows, [], key="rel-menu-sheet", editing=False)

    # ---- deletes (each asks first) -----------------------------------------------------------------

    async def _confirm(self, title: str, body: str) -> bool:
        self.editing += 1
        try:
            return await ask(self.ctx, title=title, body=body, confirm="Delete", cancel="Cancel", destructive=True)
        finally:
            self.editing -= 1

    def _run(self, coro: Any, spawn: bool = True) -> Any:
        if not spawn:
            return coro
        self.task = self.ctx.spawn(coro)
        return self.task

    def delete_relation(self, node_id: str, target: str) -> Any:
        raw, target_raw = self._raw(node_id), self._raw(target)
        own, other = self._label(node_id), self._label(target)

        async def run() -> Any:
            if not await self._confirm("Delete relation", f"Delete the relation between {own} and {other}, both "
                                                          "directions and its history? Later extractions will not "
                                                          "add it again."):
                return None
            return await self.edit(lambda t: self.core.delete_relation(t, raw, target_raw),
                                   f"Relation between {own} and {other} deleted")

        return self._run(run())

    def delete_history(self, node_id: str, target: str, run_row: dict) -> Any:
        raw, target_raw = self._raw(node_id), self._raw(target)
        own, other = self._label(node_id), self._label(target)
        label = str(run_row.get("label") or "")
        span = _span(run_row.get("from_chapter"), run_row.get("to_chapter"))

        async def run() -> Any:
            if not await self._confirm("Delete history entry", f"Delete “{label}” ({span}): what {other} is to "
                                                               f"{own}?"):
                return None
            return await self.edit(
                lambda t: self.core.delete_history_entry(t, raw, target_raw, label, run_row.get("from_chapter"),
                                                         run_row.get("to_chapter"), run_row.get("chapter_file", "")),
                f"“{label}” ({span}) deleted")

        return self._run(run())

    def delete_description(self, node_id: str, item: dict, spawn: bool = True) -> Any:
        raw, own = self._raw(node_id), self._label(node_id)
        chapter, file = item.get("chapter_num"), item.get("chapter_file") or ""
        where = f"chapter {_chapter_text(chapter)}"

        async def run() -> Any:
            if not await self._confirm("Delete description", f"Delete the description of {own} written at {where}? "
                                                             "Later extractions of that chapter will not add it again."):
                return None
            return await self.edit(lambda t: self.core.delete_description(t, raw, chapter, chapter_file=file),
                                   f"Description of {own} at {where} deleted")

        return self._run(run(), spawn)

    def remove_character(self, node_id: str) -> Any:
        raw, own = self._raw(node_id), self._label(node_id)
        count = sum(1 for item in self.core.character_history(self.tracker, raw) if item)

        async def run() -> Any:
            relations = f" and its {count} relation{'s' if count != 1 else ''}" if count else ""
            if not await self._confirm("Remove from tracker", f"Remove {own}{relations} from the relations tracker? "
                                                              "Later extractions will not add this name again."):
                return None
            result = await self.edit(lambda t: self.core.remove_character(t, raw), f"{own} removed from the tracker")
            if result and result.get("changed") and self.selected == node_id:
                self._close_history()
            return result

        return self._run(run())

    # ---- writes and undo ---------------------------------------------------------------------------

    async def edit(self, mutate: Callable[[dict], Any], message: str) -> Optional[dict]:
        """One locked load -> ``mutate`` -> write of the tracker file (``relations_tracking.edit_tracker``);
        the result is kept for Undo and its stamp, so the update check does not reload this write."""
        if not self.tracker_path:
            self.ctx.say("This glossary has no file to keep a relations tracker beside")
            return None
        self.editing += 1
        try:
            result = await self.ctx.io(self.core.edit_tracker, self.tracker_path, mutate)
        except Exception as exc:
            log.debug("editing the relations tracker failed", exc_info=True)
            self.ctx.say(f"Could not save the relations tracker: {exc}")
            return None
        finally:
            self.editing -= 1
        if not result.get("changed"):
            self.ctx.say("Nothing changed")
            return result
        self.stamps = (result.get("stamp"), self.stamps[1], self.stamps[2])
        self._push_undo("tracker", result)
        self._apply_state(tracker=result["tracker"])
        self.ctx.say(message, "Undo", self._undo_soon)
        return result

    async def edit_gender(self, raw: str, gender: str, chapter: Any, translated: str, message: str) -> Optional[dict]:
        """A gender entry through the gender tracker's own writer (``relations_tracking.set_gender``)."""
        if not self.path:
            return None
        self.editing += 1
        try:
            result = await self.ctx.io(lambda: self.core.set_gender(self.path, raw, gender, chapter,
                                                                    translated_name=translated))
        except Exception as exc:
            log.debug("editing the gender tracker failed", exc_info=True)
            self.ctx.say(f"Could not save the gender tracker: {exc}")
            return None
        finally:
            self.editing -= 1
        if not result.get("changed"):
            self.ctx.say("Nothing changed")
            return result
        self.stamps = (self.stamps[0], result.get("stamp"), self.stamps[2])
        self._push_undo("gender", result)
        self._apply_state(gender=result["tracker"])
        self.ctx.say(message, "Undo", self._undo_soon)
        return result

    def _push_undo(self, kind: str, result: dict) -> None:
        self.undo_stack.append((kind, result))
        del self.undo_stack[:-UNDO_LIMIT]
        self._refresh_undo()

    def _refresh_undo(self, push: bool = True) -> None:
        self.undo_button.disabled = not self.undo_stack
        if push:
            self.ctx.push(self.undo_button)

    def _undo_soon(self, *_args: Any) -> Any:
        self.task = self.ctx.spawn(self.undo())
        return self.task

    async def undo(self) -> bool:
        """Undo the last edit of this screen (``undo_edit`` / ``undo_gender_edit``): refused when another
        writer changed the same characters since."""
        if not self.undo_stack:
            self.ctx.say("Nothing to undo")
            return False
        kind, edit = self.undo_stack.pop()
        self._refresh_undo()
        self.editing += 1
        try:
            if kind == "gender":
                result = await self.ctx.io(self.core.undo_gender_edit, self.path, edit)
            else:
                result = await self.ctx.io(self.core.undo_edit, self.tracker_path, edit)
        except Exception as exc:
            log.debug("undo failed", exc_info=True)
            self.ctx.say(f"Could not undo: {exc}")
            return False
        finally:
            self.editing -= 1
        if not result.get("undone"):
            reason = result.get("reason")
            self.ctx.say("Cannot undo: the tracker changed since (an extraction may be updating it)"
                         if reason == "changed" else "Cannot undo: the tracker cannot be read right now"
                         if reason == "unreadable" else "Nothing to undo")
            return False
        if kind == "gender":
            self.stamps = (self.stamps[0], result.get("stamp"), self.stamps[2])
            self._apply_state(gender=result["tracker"])
        else:
            self.stamps = (result.get("stamp"), self.stamps[1], self.stamps[2])
            self._apply_state(tracker=result["tracker"])
        self.ctx.say("Undone")
        return True

    def close(self) -> None:
        self.closed = True
        task, self.watch_task = self.watch_task, None
        cancel = getattr(task, "cancel", None)
        if callable(cancel):
            try:
                cancel()
            except Exception:
                pass
        page = getattr(self.ctx, "page", None)
        self._close_form()
        if self.history_sheet is not None:
            close_dialog(page, self.history_sheet)
        pop = getattr(self.ctx, "pop_overlay", None)
        if callable(pop):
            try:
                pop()
            except Exception:
                pass


def quiet_event(handler: Callable[[Any], Any]) -> Callable[..., None]:
    """``quiet`` for a handler that needs the event (a chip's new ``selected``, a field's text)."""

    def run(e: Any = None) -> None:
        quiet(lambda: handler(e))(e)

    return run


def _read_entries(ctx: Any, path: str, stamp: Any, core: Any) -> Any:
    """The glossary's entries, None when there is no service, False when the file is being written (empty,
    or its stamp moved during the read) or does not parse right now."""
    if stamp is not None and stamp[1] == 0:
        return False
    service = getattr(ctx, "service", None)
    if service is None:
        return None
    try:
        doc = service.open_document(path)
        data = getattr(doc, "current_glossary_data", None)
    except Exception:
        log.debug("reading %s for the diagram failed", path, exc_info=True)
        return False
    if core.file_stamp(path) != stamp:
        return False
    return [e for e in data if isinstance(e, dict)] if isinstance(data, list) else None
