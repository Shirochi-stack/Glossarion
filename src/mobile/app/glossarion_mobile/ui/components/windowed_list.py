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

U6 built it for the Glossary editor (``ui/glossary/windowed_list.py``, which now re-exports this
module); U9 moved it to ``components`` (UI_SPEC Appendix A) unchanged.

Options for lists whose rows are heavier or reorderable (the Model Manager, issue 19); the
defaults keep the behaviour above:

* ``list_view=``: the list control to fill (e.g. ``ft.ReorderableListView``) instead of the
  plain ``ListView``; its ``on_scroll`` is hooked unless it already has one.
* ``scroll_keys=False``: rows keep a plain string key (``Dismissible`` / reorderable children
  need a stable value key, and Flutter compares it) instead of ``ft.ScrollKey``.
* ``show_more=True``: a deterministic "Show 100 more (N left)" button after the mounted rows
  (``list_view.footer`` when the list has one), for clients where ``on_scroll`` does not reach
  Python; at the end of a window it moves to the next window.
* :meth:`absolute` maps a mounted (window-relative) index to an index in ``items``;
  :meth:`remove_key` / :meth:`insert_item` change one row without rebuilding the window.
* ``quiet_scroll=True``: a scroll event that appends nothing tells Flet the handler is done
  (``ft.context.mark_update_called``). Flet 1.0.3 otherwise auto-updates the nearest isolated
  ancestor after a handler that called no ``update()``: the Page, a whole-page diff on every
  scroll event (``scroll_interval``).
* ``reset_scroll=True``: a window change (page selector, the footer's next window) scrolls the
  list back to its first row. Flutter keeps the scroll offset when the children change and only
  clamps it, so a jump from the end of one window would open the next one at its end.
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
        list_view: Optional[ft.ListView] = None,
        scroll_keys: bool = True,
        show_more: bool = False,
        quiet_scroll: bool = False,
        reset_scroll: bool = False,
    ) -> None:
        self.build_row = build_row
        self.key_of = key_of
        self.window_rows = max(1, int(window_rows))
        self.step = max(1, int(step))
        self.key = key
        self.on_scroll_extra = on_scroll_extra
        self.scroll_keys = scroll_keys
        self.quiet_scroll = quiet_scroll
        self.reset_scroll = reset_scroll
        self.scroll_resets = 0  # window changes that asked the client to scroll back to the first row
        self._scroll_task: Any = None
        self.items: list = []
        self.window_start = 0
        self.rendered = 0  # absolute index of the next row to build (window_start <= rendered <= window_end)
        self.controls: dict = {}  # row key -> mounted control
        self.positions: dict = {}  # row key -> absolute index in ``items``
        if list_view is None:
            list_view = ft.ListView(spacing=spacing, padding=padding if padding is not None else
                                    ft.Padding.only(left=8, right=8, bottom=96), expand=True,
                                    build_controls_on_demand=False, on_scroll=self._on_scroll, scroll_interval=120,
                                    auto_scroll=False, key=f"{key}-list")
        else:
            if getattr(list_view, "on_scroll", None) is None:
                list_view.on_scroll = self._on_scroll
            if getattr(list_view, "key", None) is None:
                list_view.key = f"{key}-list"
        self.list_view = list_view
        self.more_button: Optional[ft.TextButton] = None
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
        self.more_holder: Optional[ft.Container] = None
        if show_more:
            self.more_button = ft.TextButton(content="", on_click=lambda e: self._on_more(), key=f"{key}-more")
            self.more_holder = ft.Container(content=self.more_button, alignment=ft.Alignment.CENTER, visible=False,
                                            padding=ft.Padding.symmetric(vertical=4))
            if hasattr(self.list_view, "footer"):
                self.list_view.footer = self.more_holder
            else:
                self.control.controls.append(self.more_holder)
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

    def absolute(self, index: int) -> int:
        """The index in ``items`` of the mounted row at ``index`` (window-relative, e.g. a reorder event's)."""
        return self.window_start + int(index)

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

    def _sync_more(self) -> None:
        """The ``show_more`` button: the next step of this window, else the next window."""
        if self.more_button is None or self.more_holder is None:
            return
        total, left = len(self.items), len(self.items) - self.rendered
        if self.rendered < self.window_end:
            self.more_button.content = f"Show {min(self.step, self.window_end - self.rendered):,} more ({left:,} left)"
        elif self.window_end < total:
            self.more_button.content = (f"Show rows {self.window_end + 1:,}–"
                                        f"{min(total, self.window_end + self.window_rows):,} ({left:,} left)")
        self.more_holder.visible = left > 0

    def _on_more(self) -> int:
        if self.rendered < self.window_end:
            added = self.load_more()
            try:
                self.list_view.update()
            except Exception:
                pass
            return added
        if self.window_end < len(self.items):
            self.set_window(self.window_start + self.window_rows)
        return 0

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
        self._sync_more()
        return added

    def _build(self, item: Any, position: int) -> ft.Control:
        control = self.build_row(item, position)
        if self.scroll_keys:
            control.key = ft.ScrollKey(self.key_of(item))
        else:
            key = str(self.key_of(item))
            if getattr(control, "key", None) != key:  # a reused row keeps its key (no change to send)
                control.key = key
        return control

    def set_window(self, start: int) -> None:
        self._mount_window(start)
        self.push()
        if self.reset_scroll:
            self._scroll_to_first_row()

    def _scroll_to_first_row(self) -> None:
        """``reset_scroll``: the new window opens at its first row (a no-op off a page / outside a loop)."""
        try:
            page = self.list_view.page  # raises while the list is not on a page (tests building it alone)
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        if page is None:
            return

        async def scroll() -> None:
            try:
                await self.list_view.scroll_to(offset=0, duration=0)
            except Exception as exc:  # the client went away
                log.debug("scroll_to(0) failed: %s", exc)

        self.scroll_resets += 1
        self._scroll_task = loop.create_task(scroll())  # referenced until it is done

    def replace(self, item: Any) -> bool:
        """Re-render one mounted row in place (False when it is not mounted)."""
        key = self.key_of(item)
        old = self.controls.get(key)
        if old is None:
            return False
        position = self.positions.get(key)
        if position is not None:
            self.items[position] = item
        # by identity: ``list.index`` would compare up to 1,500 rows field by field (dataclass ``==``)
        index = next((i for i, control in enumerate(self.list_view.controls) if control is old), None)
        if index is None:
            return False
        new = self._build(item, position if position is not None else index)
        self.list_view.controls[index] = new
        self.controls[key] = new
        return True

    def _reindex(self) -> None:
        self.positions = {self.key_of(item): index for index, item in enumerate(self.items)}

    def remove_key(self, key: str) -> bool:
        """Take one row out of ``items`` and, when it is mounted, out of the list at once (a swiped
        ``Dismissible`` must leave the tree in the same update); the other rows stay as they are."""
        position = self.positions.get(key)
        if position is None:
            return False
        if position < self.window_start:  # above the mounted window: every mounted row shifts
            items = self.items[:position] + self.items[position + 1:]
            self.set_items(items, keep_window=True, keep_rendered=True)
            return True
        del self.items[position]
        control = self.controls.pop(key, None)
        if control is not None:
            index = next((i for i, c in enumerate(self.list_view.controls) if c is control), None)
            if index is not None:
                del self.list_view.controls[index]
            self.rendered = max(self.window_start, self.rendered - 1)
        self._reindex()
        if self.items and self.window_start >= len(self.items):  # the last window emptied: show the one before
            self._mount_window(self.window_start - self.window_rows)
            return True
        self._sync_selector()
        self._sync_more()
        return True

    def insert_item(self, index: int, item: Any) -> bool:
        """Put ``item`` at ``index`` of ``items``; mounts it in place when that is inside the mounted
        rows (an Undo, a failed edit putting its row back). True when it was mounted."""
        index = max(0, min(int(index), len(self.items)))
        if index < self.window_start:
            self.set_items(self.items[:index] + [item] + self.items[index:], keep_window=True, keep_rendered=True)
            return False
        self.items.insert(index, item)
        self._reindex()
        mounted = False
        if index <= self.rendered and index < self.window_start + self.window_rows:
            control = self._build(item, index)
            self.controls[self.key_of(item)] = control
            self.list_view.controls.insert(index - self.window_start, control)
            self.rendered += 1
            mounted = True
            if self.rendered > self.window_start + self.window_rows:  # the window is full: its last row leaves
                last = self.list_view.controls.pop()
                self.controls = {k: c for k, c in self.controls.items() if c is not last}
                self.rendered -= 1
        self._sync_selector()
        self._sync_more()
        return mounted

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
        pixels, maximum = getattr(e, "pixels", None), getattr(e, "max_scroll_extent", None)
        if (event_type != "overscroll" and pixels is not None and maximum is not None
                and maximum - pixels < SCROLL_APPEND_PX and self.rendered < self.window_end):
            if self.extend_to(self.rendered + self.step - 1):
                try:
                    self.list_view.update()
                    return
                except Exception:
                    pass
        if self.quiet_scroll:  # nothing to send: no auto-update of the whole page after this event
            try:
                ft.context.mark_update_called()
            except Exception:
                pass

    def load_more(self) -> int:
        """Append the next step (what ``on_scroll`` does near the end; tests and the ``show_more`` button)."""
        added = self.extend_to(self.rendered + self.step - 1)
        return added

    async def jump_to(self, key: str, *, settle: float = 0.05, duration: int = 250) -> bool:
        """Re-centre the window on ``key`` (page + build up to it), then ``scroll_to`` its ScrollKey
        (a ``scroll_keys=False`` list has no ScrollKeys to reach: it only re-centres)."""
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
