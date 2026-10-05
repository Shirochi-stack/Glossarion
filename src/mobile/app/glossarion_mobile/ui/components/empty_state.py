"""EmptyState (UI_SPEC §5.2): art · title · body · actions · optional suggestion chips."""

from __future__ import annotations

from dataclasses import field
from typing import Any, Callable, Optional, Sequence

import flet as ft

from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.theme import icon_data

__all__ = ["EmptyState", "HALGAKOS_ASSET"]

HALGAKOS_ASSET = "icon.png"  # app/assets/icon.png (the mascot)

Action = tuple[str, Callable[..., Any]]


@ft.control
class EmptyState(ft.Container):
    title: str = field(default="", metadata={"skip": True})
    body: Optional[str] = field(default=None, metadata={"skip": True})
    icon: Any = field(default=None, metadata={"skip": True})  # Material icon; None + image_src -> art
    image_src: Optional[str] = field(default=None, metadata={"skip": True})
    primary: Optional[Action] = field(default=None, metadata={"skip": True})
    secondary: Optional[Action] = field(default=None, metadata={"skip": True})
    suggestions: Sequence[Action] = field(default=(), metadata={"skip": True})

    def init(self) -> None:
        super().init()
        art_size = tokens.SIZES["empty_state_art"]
        parts: list[ft.Control] = []
        if self.image_src:
            parts.append(ft.Image(src=self.image_src, width=art_size, height=art_size, semantics_label="Halgakos"))
        elif self.icon is not None:
            parts.append(ft.Icon(icon_data(self.icon), size=art_size * 0.75, color=ft.Colors.ON_SURFACE_VARIANT))
        self.title_text = ft.Text(
            self.title,
            theme_style=ft.TextThemeStyle.TITLE_MEDIUM,
            weight=ft.FontWeight.W_600,
            text_align=ft.TextAlign.CENTER,
        )
        parts.append(self.title_text)
        if self.body:
            parts.append(
                ft.Text(
                    self.body,
                    theme_style=ft.TextThemeStyle.BODY_MEDIUM,
                    color=ft.Colors.ON_SURFACE_VARIANT,
                    text_align=ft.TextAlign.CENTER,
                )
            )
        buttons: list[ft.Control] = []
        if self.primary is not None:
            buttons.append(ft.FilledTonalButton(content=self.primary[0], on_click=self.primary[1]))
        if self.secondary is not None:
            buttons.append(ft.TextButton(content=self.secondary[0], on_click=self.secondary[1]))
        if buttons:
            parts.append(ft.Row(buttons, alignment=ft.MainAxisAlignment.CENTER, wrap=True, spacing=8))
        self.suggestion_chips = [
            ft.Chip(label=ft.Text(label), on_click=handler, key=f"suggest-{label}") for label, handler in self.suggestions
        ]
        if self.suggestion_chips:
            parts.append(
                ft.Row(self.suggestion_chips, alignment=ft.MainAxisAlignment.CENTER, wrap=True, spacing=8, run_spacing=8)
            )
        self.content = ft.Column(
            parts,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=tokens.SPACING["md"],
            tight=True,
        )
        self.padding = ft.Padding.symmetric(horizontal=tokens.SPACING["xxl"], vertical=tokens.SPACING["xxxl"])
        self.alignment = ft.Alignment.CENTER
