"""MasterDetail (UI_SPEC §1.1, §5.1): list pane + detail pane at >= 1200 dp.

Wide screens lay Settings (sections | section page), the Book page (Overview | Chapters ·
Glossary · Output) and Manga (Files | Editor · Settings) out side by side. Narrower classes
keep the single pane the screen always had: ``show_detail`` then returns False and the caller
navigates as before.

``control`` is a Container whose content is either the master alone or
``Row[master (fixed width), detail (expand)]``; ``set_two_pane`` swaps it when the size class
changes (``Screen.apply_size_class``) without rebuilding the master or the detail. The detail
pane has a title row (title · actions) over the content; replacing the detail runs the previous
detail's ``on_close`` (a section page disposes). Each detail gets a fresh wrapper key: Flet
1.0.3 freezes a subtree re-rendered under the key of the one it replaces.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional, Sequence

import flet as ft

from glossarion_mobile.ui import motion, tokens

__all__ = ["MASTER_WIDTH", "MasterDetail"]

log = logging.getLogger("glossarion.ui")

MASTER_WIDTH = 380


class MasterDetail:
    def __init__(
        self,
        master: ft.Control,
        *,
        placeholder: Optional[ft.Control] = None,
        two_pane: bool = False,
        master_width: float = MASTER_WIDTH,
        key: str = "md",
    ) -> None:
        self.master = master
        self.placeholder = placeholder or ft.Container()
        self.master_width = float(master_width)
        self.key = key
        self.two_pane = bool(two_pane)
        self.detail: Optional[ft.Control] = None
        self.detail_owner: Any = None
        self.detail_title = ""
        self._on_close: Optional[Callable[[], Any]] = None
        self.shown = 0
        self.title_text = ft.Text("", theme_style=ft.TextThemeStyle.TITLE_MEDIUM, weight=ft.FontWeight.W_600,
                                  expand=True, max_lines=2, overflow=ft.TextOverflow.ELLIPSIS)
        self.actions_row = ft.Row([], spacing=0, tight=True)
        self.header = ft.Container(
            content=ft.Row([self.title_text, self.actions_row], spacing=4,
                           vertical_alignment=ft.CrossAxisAlignment.CENTER),
            padding=ft.Padding.only(left=16, right=4, top=4, bottom=4),
            visible=False,
        )
        self.switcher = ft.AnimatedSwitcher(
            content=self.placeholder,
            duration=motion.duration(tokens.MOTION["state_ms"]),
            reverse_duration=motion.duration(tokens.MOTION["state_ms"]),
            transition=ft.AnimatedSwitcherTransition.FADE,
            expand=True,
        )
        self.detail_pane = ft.Container(
            content=ft.Column([self.header, ft.Container(content=self.switcher, expand=True)], spacing=0, expand=True),
            expand=True,
            bgcolor=ft.Colors.SURFACE,
        )
        self.master_pane = ft.Container(width=self.master_width, bgcolor=ft.Colors.SURFACE_CONTAINER_LOW)
        self.control = ft.Container(expand=True, key=key)
        self._layout()

    # ---- layout --------------------------------------------------------------------------------

    def _layout(self) -> None:
        if self.two_pane:
            self.master_pane.content = self.master
            self.control.content = ft.Row([self.master_pane, self.detail_pane], spacing=0, expand=True,
                                          vertical_alignment=ft.CrossAxisAlignment.STRETCH)
        else:
            self.master_pane.content = None
            self.control.content = self.master

    def set_two_pane(self, two_pane: bool) -> bool:
        """Swap between one and two panes; True when it changed. Leaving two panes keeps the
        detail (its owner is closed only by ``clear_detail`` / ``dispose``)."""
        two_pane = bool(two_pane)
        if two_pane == self.two_pane:
            return False
        self.two_pane = two_pane
        self._layout()
        self._push()
        return True

    # ---- detail ----------------------------------------------------------------------------------

    def show_detail(
        self,
        title: str,
        content: ft.Control,
        *,
        actions: Sequence[ft.Control] = (),
        owner: Any = None,
        on_close: Optional[Callable[[], Any]] = None,
    ) -> bool:
        """Show ``content`` in the detail pane; False in single-pane mode (the caller navigates)."""
        if not self.two_pane:
            return False
        self._release()
        self.shown += 1
        self.detail = content
        self.detail_owner = owner if owner is not None else content
        self.detail_title = title
        self._on_close = on_close
        self.title_text.value = title
        self.actions_row.controls = list(actions)
        self.header.visible = True
        self.switcher.content = ft.Container(content=content, expand=True, key=f"{self.key}-detail-{self.shown}")
        self._push()
        return True

    def shows(self, owner: Any) -> bool:
        return owner is not None and self.detail_owner is owner

    def clear_detail(self) -> None:
        self._release()
        self.detail = None
        self.detail_owner = None
        self.detail_title = ""
        self.header.visible = False
        self.actions_row.controls = []
        self.switcher.content = self.placeholder
        self._push()

    def dispose(self) -> None:
        self._release()

    def _release(self) -> None:
        callback, self._on_close = self._on_close, None
        if callback is not None:
            try:
                callback()
            except Exception:
                log.exception("master-detail close callback failed")

    def _push(self) -> None:
        try:
            self.control.update()
        except Exception:  # not mounted (tests) / the screen is gone
            pass
