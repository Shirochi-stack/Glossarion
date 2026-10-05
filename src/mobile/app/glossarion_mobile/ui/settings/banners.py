"""Inline notices for the settings screens (job running, keys not decryptable, schema missing)."""

from __future__ import annotations

from typing import Any, Callable, Optional

import flet as ft

from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.theme import semantic

__all__ = ["JOB_BANNER_TEXT", "JobBanner", "notice"]

JOB_BANNER_TEXT = "Changes apply to the next run"


def notice(text: str, *, role: str = "info", icon: str = "INFO_OUTLINE", key: Optional[str] = None,
           visible: bool = True) -> ft.Container:
    color = semantic(role, False) if role in ("success", "warning", "info", "locked") else ft.Colors.ERROR
    return ft.Container(
        key=key,
        visible=visible,
        padding=ft.Padding.symmetric(horizontal=12, vertical=8),
        margin=ft.Margin.symmetric(horizontal=12),
        border_radius=tokens.RADII["card"],
        bgcolor=ft.Colors.with_opacity(0.12, color),
        content=ft.Row(
            [ft.Icon(getattr(ft.Icons, icon, ft.Icons.INFO_OUTLINE), size=18, color=color),
             ft.Text(text, theme_style=ft.TextThemeStyle.BODY_SMALL, expand=True)],
            spacing=8,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        ),
    )


class JobBanner:
    """"Changes apply to the next run" while a job runs (UI_SPEC §4.15); follows ``store.observe_job``."""

    def __init__(self, ctx: Any) -> None:
        self.ctx = ctx
        self.control = notice(JOB_BANNER_TEXT + " — the running job keeps the settings it started with.",
                              role="info", icon="SCHEDULE", key="settings-job-banner",
                              visible=ctx.store.job_running)
        self._unsub: Optional[Callable[[], None]] = None

    @property
    def visible(self) -> bool:
        return bool(self.control.visible)

    def attach(self) -> None:
        if self._unsub is None:
            self._unsub = self.ctx.store.observe_job(lambda running: self.ctx.on_ui(self._set, running))
        self._set(self.ctx.store.job_running, push=False)

    def detach(self) -> None:
        if self._unsub is not None:
            self._unsub()
            self._unsub = None

    def _set(self, running: bool, push: bool = True) -> None:
        self.control.visible = bool(running)
        if push:
            self.ctx.push(self.control)
