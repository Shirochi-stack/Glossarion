"""``PageScreen``: a Screen bound to a ``SettingsContext`` / ``ChatEnv`` (U4 settings pages).

The Data / About / Profiles pages share the context helpers the U2 settings
screens use (``run_io``, ``say``, ``push``, ``go``, ``show_dialog``) instead of
taking the whole app; host tests pass a fake context.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Optional

import flet as ft

from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.screens.base import Screen

__all__ = ["PageScreen", "human_size", "section"]

log = logging.getLogger("glossarion.pages")


def human_size(size: Any) -> str:
    try:
        value = float(size)
    except (TypeError, ValueError):
        return "—"
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} GB"


def section(title: str, controls: list, *, key: Optional[str] = None, subtitle: Optional[str] = None) -> ft.Control:
    """A tonal card with a primary-coloured title (UI_SPEC §5.7 SectionCard look)."""
    head: list[ft.Control] = [ft.Text(title, theme_style=ft.TextThemeStyle.TITLE_SMALL, color=ft.Colors.PRIMARY,
                                      weight=ft.FontWeight.W_600)]
    if subtitle:
        head.append(ft.Text(subtitle, theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT))
    return ft.Container(
        content=ft.Column(head + list(controls), spacing=tokens.SPACING["sm"], tight=True),
        padding=tokens.SPACING["card_padding"],
        border_radius=tokens.RADII["card"],
        bgcolor=ft.Colors.SURFACE_CONTAINER_LOW,
        key=key,
    )


class PageScreen(Screen):
    def __init__(self, match: Any, ctx: Any) -> None:
        super().__init__(match)
        self.ctx = ctx
        self.dialog: Any = None

    @property
    def page(self) -> Any:
        return getattr(self.ctx, "page", None)

    @property
    def store(self) -> Any:
        return getattr(self.ctx, "store", None)

    @property
    def prefs(self) -> Any:
        return getattr(self.ctx, "prefs", None)

    def say(self, message: str) -> None:
        say = getattr(self.ctx, "say", None)
        if callable(say):
            say(message)
        else:
            log.info("%s: %s", type(self).__name__, message)

    def push(self, *controls: Any) -> None:
        for control in controls:
            if control is None:
                continue
            try:
                control.update()
            except Exception:
                pass

    async def io(self, fn: Callable[..., Any], *args: Any) -> Any:
        run_io = getattr(self.ctx, "run_io", None)
        if callable(run_io):
            return await run_io(fn, *args)
        return await asyncio.to_thread(fn, *args)

    def spawn(self, coro: Any) -> Any:
        spawn = getattr(self.ctx, "spawn", None)
        if callable(spawn):
            try:
                return spawn(coro)
            except Exception:
                pass
        try:
            return asyncio.ensure_future(coro)
        except RuntimeError:
            coro.close()
            return None

    def show(self, dialog: Any) -> Any:
        """Open a ConfirmDialog / ActionSheet / InfoSheet (anything with ``show(page)``)."""
        self.dialog = dialog
        if self.page is not None:
            dialog.show(self.page)
        return dialog

    def scaffold(self, controls: list) -> ft.Control:
        return ft.ListView(controls=controls, expand=True, padding=12, spacing=tokens.SPACING["md"])
