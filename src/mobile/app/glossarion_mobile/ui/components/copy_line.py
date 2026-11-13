"""CopyLine: one line of text the user should paste somewhere, selectable, with a Copy button.

The ErrorCard's copy pattern (UI_SPEC §5.2: mono selectable text, ``on_copy(text)`` through
``call_handler``, which runs a returned coroutine as a task) as a small row of its own, for
notices that ask the user to add a line to a prompt (Glossary › General › Enable Relations
Tracker: "- {relations_instruction}"). ``on_copy`` is usually ``ctx.copy_text``.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import flet as ft

from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.components._handlers import call_handler
from glossarion_mobile.ui.theme import mono_family

__all__ = ["copy_line"]


def copy_line(text: str, *, on_copy: Optional[Callable[[str], Any]] = None, key: Optional[str] = None,
              label: str = "Copy") -> ft.Container:
    """A rounded row: ``text`` in mono (selectable) and a Copy button (hidden without ``on_copy``)."""
    prefix = key or "copy-line"
    button = ft.TextButton(content=label, icon=ft.Icons.CONTENT_COPY, key=f"{prefix}-copy",
                           visible=on_copy is not None,
                           on_click=lambda e: call_handler(on_copy, text) if on_copy is not None else None)
    return ft.Container(
        key=key,
        padding=ft.Padding.only(left=10, right=2, top=2, bottom=2),
        border_radius=tokens.RADII["field"],
        bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST,
        content=ft.Row(
            [ft.Text(text, selectable=True, size=tokens.MONO_STYLE.size, font_family=mono_family(None),
                     expand=True, key=f"{prefix}-text"),
             button],
            spacing=4,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        ),
    )
