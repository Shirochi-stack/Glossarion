"""JumpToSheet (UI_SPEC §2.18): the desktop Input / Output navigators as a sheet.

Step header "Input 3/7 ▲ ▼ · Output 5/12 ▲ ▼" (stepping jumps to the previous / next input
or output card) and two lists, Inputs (N) / Outputs (N), whose rows ("1. preview…",
"📎 name — prompt") replace the desktop hover previews. A tap runs the chat's jump procedure
(window re-centred on the card, then ``scroll_to(scroll_key=)``, §2.8) and closes the sheet.
The rows come from ``chat_ops.jump_entries`` (hidden version members excluded).
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

import flet as ft

from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.chat.chat_ops import JumpEntry
from glossarion_mobile.ui.components.dialogs import close_dialog
from glossarion_mobile.ui.theme import HIT_TARGET

__all__ = ["JumpToSheet", "step_target"]


def step_target(entries: Sequence[JumpEntry], current: Optional[int], delta: int) -> Optional[int]:
    """The message index one step before / after ``current`` (a message index) among ``entries``."""
    indices = [e.index for e in entries]
    if not indices:
        return None
    if current is None:
        return indices[-1] if delta < 0 else indices[0]
    if delta < 0:
        earlier = [i for i in indices if i < current]
        return earlier[-1] if earlier else None
    later = [i for i in indices if i > current]
    return later[0] if later else None


def _position(entries: Sequence[JumpEntry], current: Optional[int]) -> int:
    """1-based position of the entry at or before ``current`` (0 when none)."""
    if current is None:
        return 0
    position = 0
    for n, entry in enumerate(entries, start=1):
        if entry.index <= current:
            position = n
    return position


class JumpToSheet:
    def __init__(
        self,
        inputs: Sequence[JumpEntry],
        outputs: Sequence[JumpEntry],
        *,
        on_jump: Callable[[int], Any],
        current: Optional[int] = None,
        tab: str = "inputs",
    ) -> None:
        self.inputs = list(inputs)
        self.outputs = list(outputs)
        self.on_jump = on_jump
        self.current = current
        self.tab = tab if tab in ("inputs", "outputs") else "inputs"
        self._page: Any = None
        self.input_label = ft.Text("", theme_style=ft.TextThemeStyle.LABEL_LARGE)
        self.output_label = ft.Text("", theme_style=ft.TextThemeStyle.LABEL_LARGE)
        header = ft.Row(
            [
                self.input_label,
                ft.IconButton(icon=ft.Icons.KEYBOARD_ARROW_UP, tooltip="Previous input", size_constraints=HIT_TARGET,
                              on_click=lambda e: self.step("inputs", -1), key="jump-input-up"),
                ft.IconButton(icon=ft.Icons.KEYBOARD_ARROW_DOWN, tooltip="Next input", size_constraints=HIT_TARGET,
                              on_click=lambda e: self.step("inputs", 1), key="jump-input-down"),
                ft.Container(width=8),
                self.output_label,
                ft.IconButton(icon=ft.Icons.KEYBOARD_ARROW_UP, tooltip="Previous output", size_constraints=HIT_TARGET,
                              on_click=lambda e: self.step("outputs", -1), key="jump-output-up"),
                ft.IconButton(icon=ft.Icons.KEYBOARD_ARROW_DOWN, tooltip="Next output", size_constraints=HIT_TARGET,
                              on_click=lambda e: self.step("outputs", 1), key="jump-output-down"),
            ],
            wrap=True,
            spacing=0,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )
        self.input_tab = ft.TextButton(content=f"Inputs ({len(self.inputs)})", on_click=lambda e: self.show_tab("inputs"),
                                       key="jump-tab-inputs")
        self.output_tab = ft.TextButton(content=f"Outputs ({len(self.outputs)})",
                                        on_click=lambda e: self.show_tab("outputs"), key="jump-tab-outputs")
        self.rows = ft.Column([], spacing=0, tight=True)
        self._refresh()
        self.dialog = ft.BottomSheet(
            content=ft.Container(
                padding=ft.Padding.only(left=12, right=12, bottom=16),
                content=ft.Column(
                    [ft.Text("Jump to…", theme_style=ft.TextThemeStyle.TITLE_MEDIUM, weight=ft.FontWeight.W_600),
                     header, ft.Row([self.input_tab, self.output_tab], spacing=4), self.rows],
                    spacing=6, tight=True, scroll=ft.ScrollMode.AUTO,
                ),
            ),
            show_drag_handle=True,
            scrollable=True,
            draggable=True,
            bgcolor=ft.Colors.SURFACE_CONTAINER_HIGH,
        )

    def _entries(self, tab: str) -> list:
        return self.inputs if tab == "inputs" else self.outputs

    def _refresh(self) -> None:
        self.input_label.value = f"Input {_position(self.inputs, self.current)}/{len(self.inputs)}"
        self.output_label.value = f"Output {_position(self.outputs, self.current)}/{len(self.outputs)}"
        entries = self._entries(self.tab)
        self.rows.controls = [
            ft.ListTile(title=ft.Text(entry.label, max_lines=2, overflow=ft.TextOverflow.ELLIPSIS,
                                      theme_style=ft.TextThemeStyle.BODY_MEDIUM),
                        on_click=lambda e, i=entry.index: self.jump(i), min_height=tokens.SIZES["hit_target"],
                        key=f"jump-row-{entry.index}")
            for entry in entries
        ] or [ft.Text("Nothing here yet", theme_style=ft.TextThemeStyle.BODY_SMALL)]

    def show_tab(self, tab: str) -> None:
        self.tab = tab
        self._refresh()
        try:
            self.dialog.update()
        except Exception:
            pass

    def step(self, tab: str, delta: int) -> Optional[int]:
        target = step_target(self._entries(tab), self.current, delta)
        if target is None:
            return None
        self.current = target
        self._refresh()
        self.on_jump(target)
        try:
            self.dialog.update()
        except Exception:
            pass
        return target

    def jump(self, index: int) -> None:
        self.current = index
        self.close()
        self.on_jump(index)

    def show(self, page: Any) -> None:
        self._page = page
        page.show_dialog(self.dialog)

    def close(self) -> None:
        close_dialog(self._page, self.dialog)
