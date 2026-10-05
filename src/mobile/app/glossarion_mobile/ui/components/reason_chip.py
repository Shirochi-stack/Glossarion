"""ReasonChip (UI_SPEC §5.2): why a visible feature is unavailable.

Small outlined chip ("Not on mobile", "Needs flet-camera", "Arrives in U3").
Tapping it opens an InfoSheet with the full reason. Nothing is ever hidden
(§0 item 6): unavailable features stay visible, disabled, with this chip.
"""

from __future__ import annotations

from dataclasses import field
from typing import Any, Optional

import flet as ft

from glossarion_mobile.ui import tokens

__all__ = ["ReasonChip", "NOT_ON_MOBILE"]

NOT_ON_MOBILE = "Not available on mobile"


@ft.control
class ReasonChip(ft.Chip):
    reason: str = field(default="", metadata={"skip": True})
    detail: Optional[str] = field(default=None, metadata={"skip": True})
    label: Any = ""

    def init(self) -> None:
        super().init()
        self.label = ft.Text(self.reason, theme_style=ft.TextThemeStyle.LABEL_SMALL)
        self.leading = ft.Icon(ft.Icons.INFO_OUTLINE, size=14)
        self.visual_density = ft.VisualDensity.COMPACT
        self.padding = ft.Padding.symmetric(horizontal=4)
        self.shape = ft.RoundedRectangleBorder(radius=tokens.RADII["chip"])
        self.tooltip = self.detail or self.reason
        if self.on_click is None:
            self.on_click = self._open_detail

    async def _open_detail(self, e: Any = None) -> None:
        from glossarion_mobile.ui.components.info_sheet import InfoSheet

        InfoSheet(title=self.reason, body=self.detail or self.reason).show(self.page)
