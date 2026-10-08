"""LogConsole (UI_SPEC §5.6, basic U1 version).

Filter chips All / Errors / Thinking / API, a follow toggle and Copy. Lines
arrive from a ``LogBuffer`` through ``UiDispatcher.subscribe_log`` (the pump
hands over at most 400 new lines per 120 ms tick, on the loop thread) and are
coalesced into 40-line selectable mono ``Text`` blocks, at most 100 mounted
(§7.3). Following scrolls with ``scroll_to(offset=-1)``; ``auto_scroll`` stays
off. Search and Share arrive with the job detail view (U3).
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import field
from typing import Any, Callable, Optional

import flet as ft

from glossarion_mobile.services.dispatcher import LogBuffer, LogLine, UiDispatcher
from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.theme import HIT_TARGET, mono_family

__all__ = ["FILTERS", "LogConsole"]

# (filter id, chip label, LogLine.kind values shown)
FILTERS = (
    ("all", "All", None),
    ("errors", "Errors", {"error"}),
    ("thinking", "Thinking", {"thinking"}),
    ("api", "API", {"api"}),
)


@ft.control
class LogConsole(ft.Column):
    buffer: Optional[LogBuffer] = field(default=None, metadata={"skip": True})
    dispatcher: Optional[UiDispatcher] = field(default=None, metadata={"skip": True})
    block_lines: int = field(default=40, metadata={"skip": True})
    max_blocks: int = field(default=100, metadata={"skip": True})
    backlog: int = field(default=500, metadata={"skip": True})
    list_height: Optional[float] = field(default=320, metadata={"skip": True})
    copy_handler: Optional[Callable[[str], Any]] = field(default=None, metadata={"skip": True})

    def init(self) -> None:
        super().init()
        self.filter_id = "all"
        self.following = True
        self.lines: deque[LogLine] = deque(maxlen=self.block_lines * self.max_blocks)
        self.gap_total = 0
        self._unsubscribe: Optional[Callable[[], None]] = None
        self._block_counts: list[int] = []
        self.filter_chips = {
            fid: ft.Chip(
                label=ft.Text(label),
                selected=fid == "all",
                show_checkmark=False,
                on_select=lambda e, f=fid: self.set_filter(f),
                key=f"log-filter-{fid}",
            )
            for fid, label, _kinds in FILTERS
        }
        self.follow_button = ft.IconButton(
            icon=ft.Icons.VERTICAL_ALIGN_BOTTOM,
            selected=True,
            tooltip="Follow new lines",
            on_click=self._toggle_follow,
            size_constraints=HIT_TARGET,
        )
        self.copy_button = ft.IconButton(
            icon=ft.Icons.CONTENT_COPY,
            tooltip="Copy log",
            on_click=self._copy,
            size_constraints=HIT_TARGET,
            visible=self.copy_handler is not None,
        )
        self.empty_text = ft.Text("No log lines yet.", theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT)
        self.list_view = ft.ListView(
            controls=[self.empty_text],
            spacing=0,
            padding=ft.Padding.all(8),
            height=self.list_height,
            expand=self.list_height is None,
            auto_scroll=False,
        )
        self.controls = [
            ft.Row(
                [
                    ft.Row(list(self.filter_chips.values()), scroll=ft.ScrollMode.AUTO, spacing=6, expand=True),
                    self.follow_button,
                    self.copy_button,
                ],
                spacing=4,
            ),
            ft.Container(
                content=self.list_view,
                bgcolor=ft.Colors.SURFACE_CONTAINER_LOWEST,
                border_radius=tokens.RADII["field"],
                expand=self.list_height is None,
            ),
        ]
        self.spacing = tokens.SPACING["sm"]

    # ---- lifecycle -----------------------------------------------------------

    def did_mount(self) -> None:
        super().did_mount()
        self.attach()

    def will_unmount(self) -> None:
        self.detach()
        super().will_unmount()

    def attach(self) -> None:
        if self._unsubscribe is None and self.buffer is not None and self.dispatcher is not None:
            self._unsubscribe = self.dispatcher.subscribe_log(self.buffer, self.on_lines, backlog=self.backlog)

    def detach(self) -> None:
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None

    # ---- data ------------------------------------------------------------------

    def _kinds(self) -> Optional[set]:
        for fid, _label, kinds in FILTERS:
            if fid == self.filter_id:
                return kinds
        return None

    def _visible(self, line: LogLine) -> bool:
        kinds = self._kinds()
        return kinds is None or line.kind in kinds

    def on_lines(self, lines: list[LogLine], gap: int) -> None:
        """Dispatcher callback (loop thread): append new lines to the blocks."""
        self.gap_total += gap
        if gap:
            self._append_text(f"… {gap} earlier lines were dropped", force_new_block=False)
        for line in lines:
            self.lines.append(line)
            if self._visible(line):
                self._append_text(line.text)
        self._after_change()

    def _append_text(self, text: str, force_new_block: bool = False) -> None:
        # identity, not ``in``: Flet controls are dataclasses whose ``==`` compares every field, and
        # this runs once per log line (50k lines took 47 s on the host with ``in``; U9 budget)
        controls = self.list_view.controls
        if controls and controls[0] is self.empty_text:
            controls.clear()
        blocks = self.list_view.controls
        if force_new_block or not blocks or self._block_counts[-1] >= self.block_lines:
            blocks.append(self._new_block())
            self._block_counts.append(0)
        block = blocks[-1]
        block.value = f"{block.value}\n{text}" if block.value else text
        self._block_counts[-1] += 1
        while len(blocks) > self.max_blocks:
            blocks.pop(0)
            self._block_counts.pop(0)

    def _new_block(self) -> ft.Text:
        return ft.Text(
            "",
            selectable=True,
            font_family=mono_family(self._page_or_none()),
            size=tokens.MONO_STYLE.size,
        )

    def _page_or_none(self) -> Any:
        try:
            return self.page
        except Exception:
            return None

    def _rebuild(self) -> None:
        self.list_view.controls = []
        self._block_counts = []
        for line in self.lines:
            if self._visible(line):
                self._append_text(line.text)
        if not self.list_view.controls:
            self.list_view.controls = [self.empty_text]

    def _after_change(self) -> None:
        if self._page_or_none() is None:
            return
        if self.dispatcher is not None and self.dispatcher.on_loop_thread():
            self.dispatcher.mark_dirty(self)
        else:
            self.update()
        if self.following:
            try:
                asyncio.ensure_future(self.list_view.scroll_to(offset=-1, duration=0))
            except Exception:
                pass

    # ---- actions -------------------------------------------------------------------

    def set_filter(self, filter_id: str) -> None:
        self.filter_id = filter_id
        for fid, chip in self.filter_chips.items():
            chip.selected = fid == filter_id
        self._rebuild()
        self._after_change()

    def _toggle_follow(self, e: Any = None) -> None:
        self.following = not self.following
        self.follow_button.selected = self.following
        self._after_change()

    def visible_text(self) -> str:
        return "\n".join(line.text for line in self.lines if self._visible(line))

    def _copy(self, e: Any = None) -> None:
        if self.copy_handler is not None:
            result = self.copy_handler(self.visible_text())
            if asyncio.iscoroutine(result):
                asyncio.ensure_future(result)
