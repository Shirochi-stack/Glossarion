"""SectionCard (UI_SPEC §5.7 / mobile-app design §8.4): a tonal group card.

Header (optional icon, title in titleSmall/primary, subtitle, trailing control,
"modified" dot) above its child controls; ``collapsible`` turns it into an
``ExpansionTile``. Tonal surface (surfaceContainerLow, radius 12), no
dividers, no shadow.
"""

from __future__ import annotations

from dataclasses import field
from typing import Any, Optional, Sequence

import flet as ft

from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.theme import icon_data

__all__ = ["SectionCard"]


@ft.control
class SectionCard(ft.Container):
    title: str = field(default="", metadata={"skip": True})
    children: Sequence[ft.Control] = field(default=(), metadata={"skip": True})
    icon: Any = field(default=None, metadata={"skip": True})
    subtitle: Optional[str] = field(default=None, metadata={"skip": True})
    trailing: Optional[ft.Control] = field(default=None, metadata={"skip": True})
    collapsible: bool = field(default=False, metadata={"skip": True})
    expanded: bool = field(default=True, metadata={"skip": True})
    modified_count: int = field(default=0, metadata={"skip": True})

    def init(self) -> None:
        super().init()
        self.title_text = ft.Text(
            self.title, theme_style=ft.TextThemeStyle.TITLE_SMALL, color=ft.Colors.PRIMARY, weight=ft.FontWeight.W_600
        )
        self.modified_dot = ft.Container(
            width=8,
            height=8,
            border_radius=4,
            bgcolor=ft.Colors.PRIMARY,
            visible=self.modified_count > 0,
            tooltip=f"{self.modified_count} modified" if self.modified_count else None,
        )
        leading = ft.Icon(icon_data(self.icon), color=ft.Colors.PRIMARY, size=20) if self.icon is not None else None
        subtitle = (
            ft.Text(self.subtitle, theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT)
            if self.subtitle
            else None
        )
        self.bgcolor = ft.Colors.SURFACE_CONTAINER_LOW
        self.border_radius = tokens.RADII["card"]
        if self.collapsible:
            self.padding = ft.Padding.symmetric(vertical=4)
            self.content = ft.ExpansionTile(
                title=ft.Row([self.title_text, self.modified_dot], spacing=6, tight=True),
                subtitle=subtitle,
                leading=leading,
                trailing=self.trailing,
                expanded=self.expanded,
                controls=list(self.children),
                controls_padding=ft.Padding.only(left=12, right=12, bottom=12),
                shape=ft.RoundedRectangleBorder(radius=tokens.RADII["card"]),
                collapsed_shape=ft.RoundedRectangleBorder(radius=tokens.RADII["card"]),
            )
            return
        header_text: list[ft.Control] = [ft.Row([self.title_text, self.modified_dot], spacing=6, tight=True)]
        if subtitle is not None:
            header_text.append(subtitle)
        header: list[ft.Control] = []
        if leading is not None:
            header.append(leading)
        header.append(ft.Column(header_text, spacing=2, tight=True, expand=True))
        if self.trailing is not None:
            header.append(self.trailing)
        self.padding = tokens.SPACING["card_padding"]
        self.content = ft.Column(
            [ft.Row(header, spacing=8, vertical_alignment=ft.CrossAxisAlignment.CENTER), *self.children],
            spacing=tokens.SPACING["sm"],
            tight=True,
        )
