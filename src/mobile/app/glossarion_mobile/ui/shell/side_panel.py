"""SidePanel (UI_SPEC §1.1, §5.1): the 380 dp right panel on tablets.

Title row (title, pin on wide screens, ✕) over swappable content: chat settings, job
detail, compare and the glossary term sheet (``components.surface``). Hidden until a
surface opens it.

Each content has an *owner* (the sheet or screen it belongs to) and an optional
``on_close``: replacing the content or closing the panel calls it once (a job-detail screen
disposes, a sheet forgets it was shown). ``on_change`` tells the shell the panel opened or
closed, so the chat column and the composer's output-mode row follow the narrower main
area. The content switch is a cross-fade (it is already the reduced-motion form).
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import flet as ft

from glossarion_mobile.ui import motion, tokens
from glossarion_mobile.ui.theme import HIT_TARGET

__all__ = ["SidePanel"]

log = logging.getLogger("glossarion.shell")


class SidePanel:
    def __init__(self, width: int = tokens.SIZES["side_panel"], *,
                 on_change: Optional[Callable[[bool], Any]] = None) -> None:
        self.pinned = False
        self.owner: Any = None
        self.on_close: Optional[Callable[[], Any]] = None
        self.on_change = on_change
        self.opened = 0  # contents shown so far (per-build keys: Flet freezes a re-keyed subtree)
        self.title_text = ft.Text("", theme_style=ft.TextThemeStyle.TITLE_MEDIUM, weight=ft.FontWeight.W_600, expand=True,
                                  max_lines=2, overflow=ft.TextOverflow.ELLIPSIS)
        self.pin_button = ft.IconButton(
            icon=ft.Icons.PUSH_PIN_OUTLINED, tooltip="Pin panel", on_click=self._toggle_pin, size_constraints=HIT_TARGET
        )
        self.close_button = ft.IconButton(icon=ft.Icons.CLOSE, tooltip="Close panel", on_click=self._close, size_constraints=HIT_TARGET)
        self.switcher = ft.AnimatedSwitcher(
            content=ft.Container(),
            duration=motion.duration(tokens.MOTION["state_ms"]),
            reverse_duration=motion.duration(tokens.MOTION["state_ms"]),
            transition=ft.AnimatedSwitcherTransition.FADE,
            expand=True,
        )
        self.control = ft.Container(
            width=width,
            visible=False,
            bgcolor=ft.Colors.SURFACE_CONTAINER_LOW,
            padding=ft.Padding.only(left=12, right=4, top=4, bottom=8),
            content=ft.SafeArea(
                content=ft.Column(
                    [ft.Row([self.title_text, self.pin_button, self.close_button], spacing=0,
                            vertical_alignment=ft.CrossAxisAlignment.CENTER), self.switcher],
                    spacing=4,
                    expand=True,
                ),
                avoid_intrusions_left=False,
                expand=True,
            ),
        )

    @property
    def is_open(self) -> bool:
        return bool(self.control.visible)

    def hosts(self, owner: Any) -> bool:
        """True while the panel shows ``owner``'s content."""
        return self.is_open and owner is not None and self.owner is owner

    def open(self, title: str, content: ft.Control, *, owner: Any = None,
             on_close: Optional[Callable[[], Any]] = None) -> None:
        """Show ``content`` (the previous owner's ``on_close`` runs first)."""
        was_open = self.is_open
        if self.owner is not None and self.owner is not owner:
            self._release()
        self.opened += 1
        self.owner = owner if owner is not None else content
        self.on_close = on_close
        self.title_text.value = title
        # a fresh wrapper per content: the switcher fades between two distinct children
        self.switcher.content = ft.Container(content=content, expand=True, key=f"side-panel-{self.opened}")
        self.control.visible = True
        self._push()
        if not was_open:
            self._changed(True)

    def close(self) -> None:
        was_open = self.is_open
        self._release()
        self.control.visible = False
        self.pinned = False
        self.pin_button.icon = ft.Icons.PUSH_PIN_OUTLINED
        self.switcher.content = ft.Container()
        self._push()
        if was_open:
            self._changed(False)

    def dismiss(self, owner: Any) -> bool:
        """Close when the panel shows ``owner``; True when it did."""
        if not self.hosts(owner):
            return False
        self.close()
        return True

    def _release(self) -> None:
        callback, self.on_close, self.owner = self.on_close, None, None
        if callback is not None:
            try:
                callback()
            except Exception:
                log.exception("side panel close callback failed")

    def set_pin_available(self, available: bool) -> None:
        """Pinning (three panes) only on wide screens (>= 1200 dp)."""
        self.pin_button.visible = available

    def _toggle_pin(self, e: Any = None) -> None:
        self.pinned = not self.pinned
        self.pin_button.icon = ft.Icons.PUSH_PIN if self.pinned else ft.Icons.PUSH_PIN_OUTLINED
        self.pin_button.tooltip = "Unpin panel" if self.pinned else "Pin panel"
        self._push()

    def _close(self, e: Any = None) -> None:
        self.close()

    def _changed(self, is_open: bool) -> None:
        if self.on_change is not None:
            try:
                self.on_change(is_open)
            except Exception:
                log.exception("side panel change handler failed")

    def _push(self) -> None:
        try:
            self.control.update()
        except Exception:
            pass

    @property
    def content(self) -> Optional[ft.Control]:
        wrapper = self.switcher.content
        return getattr(wrapper, "content", None) if self.is_open else None
