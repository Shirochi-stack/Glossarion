"""WindowedList (UI_SPEC §5.6, §7.3): a Python-side window over a long list (glossaries of 10k entries).

* At most ``WINDOW_ROWS`` (1,500) rows are mounted at a time; beyond that a page selector
  ("Rows 1,501–3,000 of 10,240" with previous / next) moves between windows.
* Inside a window rows are appended ``STEP`` (150) at a time: the first step when the list
  is (re)built, the next ones when ``on_scroll`` reports the end is near.
* Every row is keyed ``ft.ScrollKey(<row key>)`` inside ``ListView(build_controls_on_demand=False)``
  so each built row is a valid ``scroll_to`` target; :meth:`jump_to` first re-centres the
  window on the target and builds up to it, then scrolls (Flet 1.0.3: ``scroll_to`` only
  reaches built items).
* :meth:`replace` re-renders one row in place (selection, edits) without rebuilding the list.

The same behaviour as the Book page Chapters list (U5), as a reusable component; no
``first_item_prototype`` / ``item_extent`` because rows wrap at large text sizes.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Optional, Sequence

import flet as ft

from glossarion_mobile.ui.theme import HIT_TARGET

__all__ = ["STEP", "WINDOW_ROWS", "WindowedList"]

log = logging.getLogger("glossarion.ui")

WINDOW_ROWS = 1500
STEP = 150
SCROLL_APPEND_PX = 600


class WindowedList:
    def __init__(
        self,
        *,
        build_row: Callable[[Any, int], ft.Control],
        key_of: Callable[[Any], str],
        window_rows: int = WINDOW_ROWS,
        step: int = STEP,
        spacing: int = 4,
        padding: Any = None,
        key: str = "windowed",
        on_scroll_extra: Optional[Callable[[Any], Any]] = None,
    ) -> None:
        self.build_row = build_row
        self.key_of = key_of
        self.window_rows = max(1, int(window_rows))
        self.step = max(1, int(step))
        self.key = key
        self.on_scroll_extra = on_scroll_extra
        self.items: list = []
        self.window_start = 0
        self.rendered = 0  # absolute index of the next row to build (window_start <= rendered <= window_end)
        self.controls: dict = {}  # row key -> mounted control
        self.positions: dict = {}  # row key -> absolute index in ``items``
        self.list_view = ft.ListView(spacing=spacing, padding=padding if padding is not None else
                                     ft.Padding.only(left=8, right=8, bottom=96), expand=True,
                                     build_controls_on_demand=False, on_scroll=self._on_scroll, scroll_interval=120,
                                     auto_scroll=False, key=f"{key}-list")
        self.window_text = ft.Text("", theme_style=ft.TextThemeStyle.LABEL_MEDIUM, key=f"{key}-window-text")
        self.prev_button = ft.IconButton(icon=ft.Icons.CHEVRON_LEFT, tooltip="Previous rows",
                                         on_click=lambda e: self.set_window(self.window_start - self.window_rows),
                                         size_constraints=HIT_TARGET, key=f"{key}-window-prev")
        self.next_button = ft.IconButton(icon=ft.Icons.CHEVRON_RIGHT, tooltip="Next rows",
                                         on_click=lambda e: self.set_window(self.window_start + self.window_rows),
                                         size_constraints=HIT_TARGET, key=f"{key}-window-next")
        self.selector = ft.Row([self.prev_button, self.window_text, self.next_button],
                               alignment=ft.MainAxisAlignment.CENTER, spacing=4, visible=False, key=f"{key}-window")
        self.control = ft.Column([self.selector, self.list_view], spacing=0, expand=True, key=key)
        self.jumps: list = []

    # ---- state ------------------------------------------------------------------------------------

    @property
    def windowed(self) -> bool:
        return len(self.items) > self.window_rows

    @property
    def window_end(self) -> int:
        return min(len(self.items), self.window_start + self.window_rows)

    @property
    def mounted_count(self) -> int:
        return len(self.list_view.controls)

    def mounted_keys(self) -> list:
        return [self.key_of(item) for item in self.items[self.window_start:self.rendered]]

    # ---- building ----------------------------------------------------------------------------------

    def set_items(self, items: Sequence[Any], *, keep_window: bool = False, keep_rendered: bool = False) -> None:
        """Replace the rows. ``keep_window`` keeps the current window page; ``keep_rendered`` also
        rebuilds as many rows as were mounted (selection / edits keep the scroll position)."""
        previous_rendered = self.rendered - self.window_start
        self.items = list(items)
        self.positions = {self.key_of(item): index for index, item in enumerate(self.items)}
        start = self.window_start if keep_window else 0
        self._mount_window(start, minimum=previous_rendered if keep_rendered else 0)

    def _mount_window(self, start: int, minimum: int = 0) -> None:
        total = len(self.items)
        last = ((total - 1) // self.window_rows) * self.window_rows if total else 0
        self.window_start = max(0, min((max(0, int(start)) // self.window_rows) * self.window_rows, last))
        self.rendered = self.window_start
        self.controls = {}
        self.list_view.controls = []
        self._sync_selector()
        self.extend_to(self.window_start + max(self.step, minimum) - 1)

    def _sync_selector(self) -> None:
        self.selector.visible = self.windowed
        if self.windowed:
            total, start, end = len(self.items), self.window_start, self.window_end
            self.window_text.value = f"Rows {start + 1:,}–{end:,} of {total:,}"
            self.prev_button.disabled = start <= 0
            self.next_button.disabled = end >= total

    def extend_to(self, index: int) -> int:
        """Build rows up to ``index`` (absolute, clamped to the window); returns how many were added."""
        end = min(self.window_end, max(int(index) + 1, self.rendered))
        added = 0
        for position in range(self.rendered, end):
            item = self.items[position]
            control = self._build(item, position)
            self.controls[self.key_of(item)] = control
            self.list_view.controls.append(control)
            added += 1
        self.rendered = max(self.rendered, end)
        return added

    def _build(self, item: Any, position: int) -> ft.Control:
        control = self.build_row(item, position)
        control.key = ft.ScrollKey(self.key_of(item))
        return control

    def set_window(self, start: int) -> None:
        self._mount_window(start)
        self.push()

    def replace(self, item: Any) -> bool:
        """Re-render one mounted row in place (False when it is not mounted)."""
        key = self.key_of(item)
        old = self.controls.get(key)
        if old is None:
            return False
        position = self.positions.get(key)
        if position is not None:
            self.items[position] = item
        try:
            index = self.list_view.controls.index(old)
        except ValueError:
            return False
        new = self._build(item, position if position is not None else index)
        self.list_view.controls[index] = new
        self.controls[key] = new
        return True

    def push(self) -> None:
        for control in (self.control,):
            try:
                control.update()
            except Exception:
                pass

    # ---- scrolling ----------------------------------------------------------------------------------

    def _on_scroll(self, e: Any) -> None:
        if self.on_scroll_extra is not None:
            try:
                self.on_scroll_extra(e)
            except Exception:
                log.debug("scroll hook failed", exc_info=True)
        event_type = str(getattr(getattr(e, "event_type", None), "value", getattr(e, "event_type", "")))
        if event_type == "overscroll":
            return
        pixels, maximum = getattr(e, "pixels", None), getattr(e, "max_scroll_extent", None)
        if pixels is None or maximum is None:
            return
        if maximum - pixels < SCROLL_APPEND_PX and self.rendered < self.window_end:
            if self.extend_to(self.rendered + self.step - 1):
                try:
                    self.list_view.update()
                except Exception:
                    pass

    def load_more(self) -> int:
        """Append the next step (what ``on_scroll`` does near the end; tests and the "More" fallback)."""
        added = self.extend_to(self.rendered + self.step - 1)
        return added

    async def jump_to(self, key: str, *, settle: float = 0.05, duration: int = 250) -> bool:
        """Re-centre the window on ``key`` (page + build up to it), then ``scroll_to`` its ScrollKey."""
        position = self.positions.get(key)
        if position is None:
            return False
        if not self.window_start <= position < self.window_end:
            self._mount_window(position)
            self.push()
            await asyncio.sleep(settle)
        if position >= self.rendered:
            self.extend_to(position + self.step // 2)
            try:
                self.list_view.update()
            except Exception:
                pass
            await asyncio.sleep(settle)
        self.jumps.append(key)
        try:
            await self.list_view.scroll_to(scroll_key=ft.ScrollKey(key), duration=duration)
        except Exception as exc:  # not mounted (tests) / client gone
            log.debug("scroll_to(%s) failed: %s", key, exc)
        return True
