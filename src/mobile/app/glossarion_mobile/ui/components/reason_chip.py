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

__all__ = ["ReasonChip", "NOT_ON_MOBILE", "reason_subtitle", "tile_reason_chip", "unavailable_tile"]

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


def reason_subtitle(chip: "ReasonChip", subtitle: Any = None) -> ft.Column:
    """A ListTile ``subtitle`` holding ``chip`` under the row's label (and its own subtitle, if any).

    Never ``ListTile(trailing=ReasonChip(...))``: Flutter lays out ``trailing`` at its full width
    before the title, so a long reason ("Not an image, video or audio file") leaves the title a
    few dp at 320 dp or at large text and the label wraps one letter per line. Here the chip
    sits in a wrapping row below the label and the label keeps the row's width."""
    if isinstance(subtitle, str):
        subtitle = ft.Text(subtitle, theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT)
    return ft.Column(([subtitle] if subtitle is not None else []) + [ft.Row([chip], wrap=True)],
                     spacing=2, tight=True)


def tile_reason_chip(tile: Any) -> Optional["ReasonChip"]:
    """The ReasonChip a ``reason_subtitle`` put in ``tile``'s subtitle (None: the row is available)."""
    subtitle = getattr(tile, "subtitle", None)
    for control in getattr(subtitle, "controls", None) or []:
        for chip in getattr(control, "controls", None) or []:
            if isinstance(chip, ReasonChip):
                return chip
    return None


def unavailable_tile(
    title: str,
    *,
    reason: str,
    detail: Optional[str] = None,
    subtitle: Any = None,
    leading: Any = None,
    key: Any = None,
    dense: bool = True,
    min_height: Optional[float] = None,
) -> ft.ListTile:
    """A row for an unavailable feature with its ReasonChip (UI_SPEC §5.2).

    Never ``ListTile(disabled=True)``: Flet passes ``disabled`` down to every child (Dart
    ``Control.disabled`` follows the parent; the chip's ``onPressed`` needs ``!disabled``), so a
    chip inside a disabled row is greyed out and its InfoSheet with the full reason never opens.
    The row stays enabled with no tap action and muted text; the chip opens the reason."""
    if leading is not None and isinstance(leading, ft.Icon) and leading.color is None:
        leading.color = ft.Colors.ON_SURFACE_VARIANT
    return ft.ListTile(
        title=ft.Text(title, color=ft.Colors.ON_SURFACE_VARIANT),
        subtitle=reason_subtitle(ReasonChip(reason=reason, detail=detail), subtitle),
        leading=leading,
        on_click=None,
        dense=dense,
        min_height=min_height,
        key=key,
    )
