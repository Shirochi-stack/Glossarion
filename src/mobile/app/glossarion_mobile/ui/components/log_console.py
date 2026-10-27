"""LogConsole (UI_SPEC §5.6, basic U1 version).

Filter chips All / Errors / Thinking / API, a follow toggle and Copy. Lines
arrive from a ``LogBuffer`` through ``UiDispatcher.subscribe_log`` (the pump
hands over at most 400 new lines per 120 ms tick, on the loop thread) and are
coalesced into 40-line selectable ``Text`` blocks in the mono family at the log
size (``theme.log_text``, §6.2), at most 100 mounted
(§7.3). Search and Share arrive with the job detail view (U3).

Following (hotfix 5, owner: "why are the logs not scrolled to the bottom by default. There should be a
button to jump to bottom"): the list follows with Flet's ``auto_scroll`` (the client jumps to the end after
it laid the new lines out; a ``scroll_to(offset=-1)`` sent from here ran before that layout, against the
previous end, so a log opened on its backlog stayed near its top). Every console opens on its newest lines,
also while auto-scroll is paused for the session. Scrolling up pauses following and shows a ↓ button
(``jump_button``, key ``log-jump-end``) over the list's corner; ↓, the auto-scroll button or scrolling back
to the end follows again.

Owner 2026-10-18: the follow toggle is an "auto-scroll" button (pause / resume icons, a tooltip naming
the state, key ``log-autoscroll``) whose choice lasts for the app session (``SESSION``); with
``header_follow`` the host shows ``follow_button`` in its card header (next to Share) instead of the
filter row. The filter chips and Copy are compact so All / Errors / Thinking / API and Copy fit a
360 dp phone at 100 % text; the row still scrolls sideways at larger text.
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import field
from typing import Any, Callable, Optional

import flet as ft

from glossarion_mobile.services.dispatcher import LogBuffer, LogLine, UiDispatcher
from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.theme import HIT_TARGET, log_text, text_style

__all__ = ["CHIP_LABEL", "CHIP_LABEL_PADDING", "CHIP_PADDING", "COMPACT_BUTTON", "FILTERS", "LogConsole",
           "SESSION"]

#: per-viewer choices kept for the rest of the app session (every LogConsole opens with them)
SESSION = {"following": True}

#: compact filter chips: LABEL_MEDIUM text, 2 + 4 dp padding a side (the chip's tap target stays 48 dp)
CHIP_LABEL = "label_medium"
CHIP_PADDING = ft.Padding.symmetric(horizontal=2)
CHIP_LABEL_PADDING = ft.Padding.symmetric(horizontal=4)
#: Copy / follow in the filter row: 40 dp wide, still 48 dp tall
COMPACT_BUTTON = ft.BoxConstraints(min_width=40, min_height=tokens.SIZES["hit_target"])
#: how close to the list's end (dp) still counts as "at the end" (Flet's auto_scroll re-pins within 40 px)
END_SLACK = 40.0
#: a console opened while auto-scroll is paused stops following this long after its backlog arrived (the
#: client pins the last backlog batch after laying it out)
OPEN_SETTLE_SECONDS = 0.6

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
    header_follow: bool = field(default=False, metadata={"skip": True})
    #: called after each change the console painted (new lines, a filter, follow on/off). Not for pinning the
    #: host's page to its end: a ``scroll_to`` sent from here runs on the client before the new content is laid
    #: out (the Job page pins its list with ``auto_scroll`` instead, hotfix 2)
    on_after_change: Optional[Callable[[], Any]] = field(default=None, metadata={"skip": True})

    def init(self) -> None:
        super().init()
        self.filter_id = "all"
        self.following = bool(SESSION["following"])
        self.lines: deque[LogLine] = deque(maxlen=self.block_lines * self.max_blocks)
        self.gap_total = 0
        self.at_end = True  # the list shows its newest lines (on_scroll keeps it current)
        self._opening_seq: Optional[int] = None  # the backlog's last line while the console opens on its end
        self._scroll_paused = False  # a scroll up paused following (back at the end follows again)
        self._unsubscribe: Optional[Callable[[], None]] = None
        self._block_counts: list[int] = []
        self.filter_chips = {
            fid: ft.Chip(
                label=ft.Text(label),
                label_text_style=text_style(CHIP_LABEL),
                padding=CHIP_PADDING,
                label_padding=CHIP_LABEL_PADDING,
                visual_density=ft.VisualDensity.COMPACT,
                selected=fid == "all",
                show_checkmark=False,
                on_select=lambda e, f=fid: self.set_filter(f),
                key=f"log-filter-{fid}",
            )
            for fid, label, _kinds in FILTERS
        }
        self.follow_button = ft.IconButton(
            tooltip="Auto-scroll",  # _paint_follow names the state
            on_click=self._toggle_follow,
            size_constraints=HIT_TARGET if self.header_follow else COMPACT_BUTTON,
            key="log-autoscroll",
        )
        self._paint_follow()
        self.copy_button = ft.IconButton(
            icon=ft.Icons.CONTENT_COPY,
            tooltip="Copy log",
            on_click=self._copy,
            size_constraints=COMPACT_BUTTON,
            visible=self.copy_handler is not None,
            key="log-copy",
        )
        self.empty_text = ft.Text("No log lines yet.", theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT)
        self.list_view = ft.ListView(
            controls=[self.empty_text],
            spacing=0,
            padding=ft.Padding.all(8),
            height=self.list_height,
            expand=self.list_height is None,
            # following: the client pins the end after layout (instant, no 1 s animation); ``attach`` turns it
            # on for the backlog whatever the session's choice
            auto_scroll=self.following,
            auto_scroll_animation=ft.Animation(0, ft.AnimationCurve.LINEAR),
            on_scroll=self._on_scroll,
            scroll_interval=100,
        )
        self.jump_button = ft.IconButton(
            icon=ft.Icons.ARROW_DOWNWARD,
            tooltip="Jump to the newest lines",
            size_constraints=HIT_TARGET,
            on_click=self.jump_to_end,
            style=ft.ButtonStyle(bgcolor=ft.Colors.SECONDARY_CONTAINER, color=ft.Colors.ON_SECONDARY_CONTAINER,
                                 elevation=2),
            key="log-jump-end",
        )
        self.jump_holder = ft.Container(content=self.jump_button, right=8, bottom=8, visible=False)
        self.filter_row = ft.Row(list(self.filter_chips.values()), scroll=ft.ScrollMode.AUTO, spacing=4, expand=True,
                                 key="log-filters")
        self.toolbar = ft.Row(
            [self.filter_row] + ([] if self.header_follow else [self.follow_button]) + [self.copy_button],
            spacing=2,
            key="log-toolbar",
        )
        self.controls = [
            self.toolbar,
            ft.Container(
                content=ft.Stack([self.list_view, self.jump_holder],
                                 fit=ft.StackFit.EXPAND if self.list_height is None else ft.StackFit.LOOSE,
                                 expand=self.list_height is None, key="log-list-stack"),
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
            # the console opens on its newest lines, also while auto-scroll is paused for the session
            last = int(getattr(self.buffer, "last_seq", 0) or 0)
            self._opening_seq = last if last > 0 else None
            self.list_view.auto_scroll = True
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
        opening = self._opening_seq
        if opening is not None and lines and lines[-1].seq >= opening:
            self._opening_seq = None  # the backlog is in: the session's auto-scroll choice applies again
            if not self.following:
                self._settle_paused()
        self._after_change()

    def _settle_paused(self) -> None:
        """A console opened while auto-scroll is paused: stop following once the client pinned the backlog."""
        def settle() -> None:
            if not self.following and self._opening_seq is None:
                self.list_view.auto_scroll = False
                self._push(self.list_view)

        try:
            asyncio.get_running_loop().call_later(OPEN_SETTLE_SECONDS, settle)
        except RuntimeError:  # no loop (host tests drive it directly)
            settle()

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
        return log_text("", page=self._page_or_none())

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
        if self.on_after_change is not None:
            try:
                self.on_after_change()
            except Exception:
                pass

    # ---- actions -------------------------------------------------------------------

    def set_filter(self, filter_id: str) -> None:
        self.filter_id = filter_id
        for fid, chip in self.filter_chips.items():
            chip.selected = fid == filter_id
        self._rebuild()
        self._after_change()

    def _paint_follow(self) -> None:
        # the icon is the tap's action (pause while following, jump-to-end while paused)
        button = self.follow_button
        button.selected = self.following
        if self.following:
            button.icon = ft.Icons.PAUSE_CIRCLE_OUTLINE
            button.tooltip = "Auto-scroll is on: tap to pause"
        else:
            button.icon = ft.Icons.KEYBOARD_DOUBLE_ARROW_DOWN
            button.tooltip = "Auto-scroll is paused: tap to follow new lines"

    def set_following(self, following: bool, *, remember: bool = True) -> None:
        """Auto-scroll on (jumps to the end now) or off; ``remember``: the choice lasts for the app session
        (a scroll up pauses this console only)."""
        self.following = bool(following)
        if remember:
            SESSION["following"] = self.following
        self._paint_follow()
        if self.following:
            self.list_view.auto_scroll = True
            self._jump()
        elif self._opening_seq is None:
            self.list_view.auto_scroll = False
        self._after_change()

    def _jump(self) -> None:
        """To the end now: the lines are laid out already, so a ``scroll_to`` lands on the real end (the
        client's own pin waits for new lines after a scroll up)."""
        self._set_at_end(True)
        if self._page_or_none() is None:
            return
        try:
            asyncio.ensure_future(self.list_view.scroll_to(offset=-1, duration=200))
        except Exception:
            pass

    def jump_to_end(self, e: Any = None) -> None:
        """↓: the newest lines, and this console follows again (the session's auto-scroll choice stays)."""
        self._scroll_paused = False
        self.set_following(True, remember=False)

    def _set_at_end(self, at_end: bool) -> None:
        self.at_end = bool(at_end)
        visible = not self.at_end and bool(self.lines)
        if self.jump_holder.visible != visible:
            self.jump_holder.visible = visible
            self._push(self.jump_holder)

    def _on_scroll(self, e: Any) -> None:
        """Scrolling up pauses following (this console only) and shows ↓; back at the end it follows again
        when a scroll paused it."""
        pixels, maximum = getattr(e, "pixels", None), getattr(e, "max_scroll_extent", None)
        if pixels is None or maximum is None:
            return
        try:
            at_end = float(pixels) >= float(maximum) - END_SLACK
        except (TypeError, ValueError):
            return
        was = self.at_end
        self._set_at_end(at_end)
        if was and not at_end and self.following and self._opening_seq is None:
            # auto_scroll stays on: the client unpinned itself on the scroll up and re-pins back at the end
            self._scroll_paused = True
            self.following = False
            self._paint_follow()
            self._push(self.follow_button)
        elif at_end and not was and self._scroll_paused:
            self._scroll_paused = False
            self.following = True
            self._paint_follow()
            self._push(self.follow_button)

    def _push(self, control: Any) -> None:
        if self._page_or_none() is None:
            return
        try:
            control.update()
        except Exception:
            pass

    def _toggle_follow(self, e: Any = None) -> None:
        self._scroll_paused = False
        self.set_following(not self.following)

    def visible_text(self) -> str:
        return "\n".join(line.text for line in self.lines if self._visible(line))

    def _copy(self, e: Any = None) -> None:
        if self.copy_handler is not None:
            result = self.copy_handler(self.visible_text())
            if asyncio.iscoroutine(result):
                asyncio.ensure_future(result)
