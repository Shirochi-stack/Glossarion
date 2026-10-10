"""Status components (UI_SPEC §5.3): status is always icon + text + colour.

``StatusChip`` is the filter / status chip used by Progress, Library, Jobs and
Keys: leading status icon, label and optional count, all in the status colour
from the shared palette (``tokens.STATUS_PALETTE``).
"""

from __future__ import annotations

from dataclasses import field
from typing import Any, Optional

import flet as ft

from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.theme import icon_data, status_color

__all__ = ["StatusChip", "count_badge", "status_label"]


def count_badge(label: str) -> ft.Badge:
    """A small count badge drawn inside its control's top-right corner (U11 item 6): the default
    Material badge hangs over the edge, and a scrolling chip row or a short strip cropped its red dot."""
    return ft.Badge(label=str(label), large_size=14, text_style=ft.TextStyle(size=9),
                    padding=ft.Padding.symmetric(horizontal=3), offset=ft.Offset(-4, 3))

def status_label(status: str, label: Optional[str] = None, count: Optional[int] = None) -> str:
    text = label or tokens.status_style(status).label
    return f"{text} · {count}" if count is not None else text


@ft.control
class StatusChip(ft.Chip):
    status: str = field(default="pending", metadata={"skip": True})
    text: Optional[str] = field(default=None, metadata={"skip": True})
    count: Optional[int] = field(default=None, metadata={"skip": True})
    dark: bool = field(default=False, metadata={"skip": True})
    label: Any = ""

    def init(self) -> None:
        super().init()
        self.icon_control = ft.Icon(ft.Icons.INFO_OUTLINE, size=16)
        self.label_text = ft.Text(theme_style=ft.TextThemeStyle.LABEL_MEDIUM)
        self.leading = self.icon_control
        self.label = self.label_text
        self.show_checkmark = False
        self.visual_density = ft.VisualDensity.COMPACT
        self.shape = ft.RoundedRectangleBorder(radius=tokens.RADII["chip"])
        self._sync()

    def before_update(self) -> None:
        super().before_update()
        self._sync()

    def _sync(self) -> None:
        color = status_color(self.status, self.dark)
        self.icon_control.icon = icon_data(tokens.status_style(self.status).icon)
        self.icon_control.color = color
        self.label_text.value = status_label(self.status, self.text, self.count)
        self.label_text.color = color
        self.border_side = ft.BorderSide(1, color)
        # zero counts are hidden unless the chip is a pinned group (§5.3)
        self.visible = not (self.count == 0 and not self.selected and self.data != "pinned")

    def status_color(self) -> str:
        return status_color(self.status, self.dark)
