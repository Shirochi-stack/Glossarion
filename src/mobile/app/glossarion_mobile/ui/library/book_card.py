"""BookCard and BookListRow (UI_SPEC §3.2, §5.6): render a ``CardModel``.

Grid card: cover stack (cached thumb or the Halgakos placeholder, the corner
ribbon, a 3 dp progress bar for in-progress books, the ▶ Continue button when a
reading position exists, the selection check) · title (2-line clamp, 3 at
>= 160% text) · info row ("x.x MB" + type badge + optional language chip) ·
warning chips · the pill pinned to the bottom. List row: 72 dp, 48 x 72 cover.

Tap / long-press / ▶ / ⋯ are callbacks; ``set_model`` updates the controls in
place (the home screen's diff replaces only cards whose signature changed).
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import flet as ft

from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.components.empty_state import HALGAKOS_ASSET
from glossarion_mobile.ui.library.colors import pill_colors, ribbon_colors, type_badge, warning_colors
from glossarion_mobile.ui.library.common import tinted
from glossarion_mobile.ui.library.models import CardModel
from glossarion_mobile.ui.theme import HIT_TARGET, resolve_color

__all__ = ["BookCard", "BookListRow", "LIST_ROW_HEIGHT"]

LIST_ROW_HEIGHT = 72
_LIST_COVER = (48, 72)


def _ribbon(model: CardModel) -> Optional[ft.Control]:
    if not model.ribbon_text:
        return None
    text_color, background = ribbon_colors(model.ribbon_state)
    return ft.Container(
        content=ft.Text(model.ribbon_text, size=tokens.RIBBON_STYLE.size, weight=ft.FontWeight.W_700,
                        color=text_color, no_wrap=True),
        bgcolor=background,
        padding=ft.Padding.symmetric(horizontal=5, vertical=1),
        border_radius=ft.BorderRadius.only(bottom_right=3, top_left=tokens.RADII["cover"]),
        left=0,
        top=0,
        key="ribbon",
    )


def _warning_chip(text: str, role: str, tooltip: str, dark: bool) -> ft.Control:
    color, background = warning_colors(role, dark)
    color = resolve_color(color)
    if background.startswith("role:") or background == color:
        background = tinted(color, 0.12)
    return ft.Container(
        content=ft.Text(text, size=10, weight=ft.FontWeight.W_700, color=color, no_wrap=True),
        bgcolor=background,
        border=ft.Border.all(1, color),
        border_radius=3,
        padding=ft.Padding.symmetric(horizontal=4, vertical=0),
        tooltip=tooltip or None,
        key=f"warn-{role}",
    )


def _pill_row(model: CardModel, dark: bool) -> Optional[ft.Control]:
    if not model.pill_text:
        return None
    colors = pill_colors(model.state if model.state in ("outdated_progress", "not_started", "ready_to_compile")
                         else "in_progress", dark)
    pill_box = ft.Container(
        content=ft.Text(model.pill_text, size=10, weight=ft.FontWeight.W_700, color=colors.text, no_wrap=True,
                        overflow=ft.TextOverflow.ELLIPSIS),
        bgcolor=colors.background,
        border=ft.Border.all(1, colors.border),
        border_radius=3,
        padding=ft.Padding.only(left=5, right=5, bottom=2),
        key="pill",
    )
    parts: list[ft.Control] = [ft.Container(content=pill_box, expand=True if not model.pct_text else False)]
    if model.pct_text:
        parts.append(ft.Text(model.pct_text, size=10, weight=ft.FontWeight.W_700, color=colors.pct, key="pct"))
    return ft.Row(parts, spacing=4, tight=True)


def _info_row(model: CardModel, dark: bool, *, show_language: bool) -> ft.Control:
    emoji, label, color = type_badge(model.type_kind, dark)
    parts: list[ft.Control] = [
        ft.Text(model.size_text, size=11, color=ft.Colors.ON_SURFACE_VARIANT, no_wrap=True, key="size"),
        ft.Text(f"{emoji}{label}", size=11, weight=ft.FontWeight.W_700, color=color, no_wrap=True, key="type"),
    ]
    if show_language and model.language:
        parts.append(ft.Container(
            content=ft.Text(model.language, size=9, weight=ft.FontWeight.W_600, color=ft.Colors.ON_SURFACE_VARIANT),
            border=ft.Border.all(1, ft.Colors.OUTLINE_VARIANT),
            border_radius=4,
            padding=ft.Padding.symmetric(horizontal=3),
            key="lang",
        ))
    return ft.Row(parts, spacing=4, tight=True, wrap=False)


class _CardBase:
    def __init__(self, model: CardModel, *, cover_src: Optional[str] = None, dark: bool = False,
                 on_open: Optional[Callable[[CardModel], Any]] = None,
                 on_long_press: Optional[Callable[[CardModel], Any]] = None,
                 on_continue: Optional[Callable[[CardModel], Any]] = None,
                 on_more: Optional[Callable[[CardModel], Any]] = None,
                 show_language: bool = False, show_progress: bool = True) -> None:
        self.model = model
        self.cover_src = cover_src
        self.dark = dark
        self.on_open = on_open
        self.on_long_press = on_long_press
        self.on_continue = on_continue
        self.on_more = on_more
        self.show_language = show_language
        self.show_progress = show_progress
        self.control: ft.Container = ft.Container()

    def _tap(self, e: Any = None) -> Any:
        return self.on_open(self.model) if self.on_open is not None else None

    def _long(self, e: Any = None) -> Any:
        return self.on_long_press(self.model) if self.on_long_press is not None else None

    def _continue(self, e: Any = None) -> Any:
        return self.on_continue(self.model) if self.on_continue is not None else None

    def _more(self, e: Any = None) -> Any:
        return self.on_more(self.model) if self.on_more is not None else None

    def _cover_image(self, width: Optional[float], height: Optional[float]) -> ft.Control:
        src = self.cover_src or HALGAKOS_ASSET
        fallback = ft.Container(content=ft.Text("\U0001f4d6", size=max(14, int((width or 60) * 0.3))),
                                alignment=ft.Alignment.CENTER, bgcolor=ft.Colors.SURFACE_CONTAINER_HIGH)
        return ft.Image(src=src, width=width, height=height, fit=ft.BoxFit.COVER if self.cover_src else
                        ft.BoxFit.CONTAIN, border_radius=tokens.RADII["cover"], error_content=fallback,
                        semantics_label=self.model.full_title, gapless_playback=True, key="cover")

    def set_cover(self, src: Optional[str]) -> None:
        self.cover_src = src
        self.rebuild()

    def set_model(self, model: CardModel) -> None:
        self.model = model
        self.rebuild()

    def rebuild(self) -> None:  # pragma: no cover - subclasses
        raise NotImplementedError

    def update(self) -> None:
        try:
            self.control.update()
        except Exception:
            pass


class BookCard(_CardBase):
    """Grid card of fixed width ``card_w`` (the GridView sizes cells by ``max_extent``)."""

    def __init__(self, model: CardModel, *, card_w: int, cover_h: int, title_lines: int = 2, **kwargs: Any) -> None:
        super().__init__(model, **kwargs)
        self.card_w = card_w
        self.cover_h = cover_h
        self.title_lines = title_lines
        self.control = ft.Container(
            on_click=self._tap,
            on_long_press=self._long,
            border_radius=tokens.RADII["card"],
            padding=6,
            ink=True,
            key=f"book-{model.bid}",
        )
        self.rebuild()

    def rebuild(self) -> None:
        model = self.model
        # The GridView sizes the cell (max_extent + aspect ratio); the cover takes what the text rows leave.
        stack: list[ft.Control] = [ft.Container(content=self._cover_image(None, None), expand=True,
                                                left=0, top=0, right=0, bottom=0)]
        ribbon = _ribbon(model)
        if ribbon is not None:
            stack.append(ribbon)
        if self.show_progress and model.progress is not None:
            stack.append(ft.ProgressBar(value=max(0.0, min(1.0, model.progress)), bar_height=3, left=0, right=0,
                                        bottom=0, color=resolve_color("#6c63ff"), bgcolor=tinted("#000000", 0.25),
                                        semantics_label="Translation progress", key="bar"))
        if model.has_continue:
            stack.append(ft.Container(
                content=ft.FilledIconButton(icon=ft.Icons.PLAY_ARROW, icon_size=16, tooltip="Continue reading",
                                            on_click=self._continue, width=28, height=28,
                                            style=ft.ButtonStyle(padding=0)),
                right=0, bottom=4, width=tokens.SIZES["hit_target"], height=tokens.SIZES["hit_target"],
                alignment=ft.Alignment.BOTTOM_RIGHT, key="continue"))
        if model.selected:
            stack.append(ft.Container(content=ft.Icon(ft.Icons.CHECK_CIRCLE, color=ft.Colors.PRIMARY, size=22),
                                      right=4, top=4, bgcolor=ft.Colors.SURFACE, border_radius=12, key="check"))
        rows: list[ft.Control] = [
            ft.Stack(stack, expand=True, key="cover-stack"),
            ft.Text(model.title, max_lines=self.title_lines, overflow=ft.TextOverflow.ELLIPSIS,
                    theme_style=ft.TextThemeStyle.LABEL_LARGE, tooltip=model.tooltip or model.full_title,
                    key="title"),
            _info_row(model, self.dark, show_language=self.show_language),
        ]
        if model.warnings:
            rows.append(ft.Row([_warning_chip(w.text, w.role, w.tooltip, self.dark) for w in model.warnings],
                               spacing=4, wrap=True, run_spacing=2, key="warnings"))
        pill_row = _pill_row(model, self.dark)
        if pill_row is not None:
            rows.append(pill_row)
        self.control.content = ft.Semantics(content=ft.Column(rows, spacing=3, expand=True),
                                            label=model.semantics, selected=model.selected, button=True)
        self.control.bgcolor = (ft.Colors.SECONDARY_CONTAINER if model.selected else ft.Colors.SURFACE_CONTAINER_LOW)
        self.control.border = ft.Border.all(2, ft.Colors.PRIMARY) if model.selected else None


class BookListRow(_CardBase):
    """List row (72 dp, 48 x 72 cover) with a trailing ⋯ for the single-card action sheet."""

    def __init__(self, model: CardModel, **kwargs: Any) -> None:
        super().__init__(model, **kwargs)
        self.control = ft.Container(
            on_click=self._tap,
            on_long_press=self._long,
            border_radius=tokens.RADII["card"],
            padding=ft.Padding.symmetric(horizontal=8, vertical=4),
            ink=True,
            key=f"book-{model.bid}",
        )
        self.rebuild()

    def rebuild(self) -> None:
        model = self.model
        width, height = _LIST_COVER
        cover_stack: list[ft.Control] = [self._cover_image(width, height)]
        if self.show_progress and model.progress is not None:
            cover_stack.append(ft.ProgressBar(value=max(0.0, min(1.0, model.progress)), bar_height=3, left=0,
                                              right=0, bottom=0, color=resolve_color("#6c63ff"), key="bar"))
        if model.selected:
            cover_stack.append(ft.Container(content=ft.Icon(ft.Icons.CHECK_CIRCLE, color=ft.Colors.PRIMARY,
                                                            size=18), right=2, top=2, key="check"))
        lines: list[ft.Control] = [
            ft.Text(model.title, max_lines=1, overflow=ft.TextOverflow.ELLIPSIS,
                    theme_style=ft.TextThemeStyle.BODY_MEDIUM, weight=ft.FontWeight.W_600, key="title"),
            _info_row(model, self.dark, show_language=self.show_language),
        ]
        extras: list[ft.Control] = []
        if model.ribbon_text and model.ribbon_state == "compiling":
            text_color, background = ribbon_colors("compiling")
            extras.append(ft.Container(content=ft.Text(model.ribbon_text, size=9, weight=ft.FontWeight.W_700,
                                                       color=text_color), bgcolor=background, border_radius=3,
                                       padding=ft.Padding.symmetric(horizontal=4)))
        pill_row = _pill_row(model, self.dark)
        if pill_row is not None:
            extras.append(pill_row)
        extras.extend(_warning_chip(w.text, w.role, w.tooltip, self.dark) for w in model.warnings)
        if extras:
            lines.append(ft.Row(extras, spacing=4, wrap=True, run_spacing=2, key="extras"))
        more = ft.IconButton(icon=ft.Icons.MORE_VERT, tooltip="Book actions", on_click=self._more,
                             size_constraints=HIT_TARGET, key="more")
        self.control.content = ft.Semantics(
            content=ft.Row([
                ft.Stack(cover_stack, width=width, height=height),
                ft.Column(lines, spacing=2, expand=True, tight=True),
                more,
            ], spacing=10, vertical_alignment=ft.CrossAxisAlignment.CENTER),
            label=model.semantics, selected=model.selected, button=True)
        self.control.bgcolor = ft.Colors.SECONDARY_CONTAINER if model.selected else None
