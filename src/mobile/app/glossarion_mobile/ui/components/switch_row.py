"""A Switch whose label wraps (owner, 2026-10-28: the Manga translator's "Include page image in translation
requests (visual context)" toggle was cut off at the screen edge).

Flet's ``Switch(label=...)`` lays the label out in a Row as wide as its children (Flet 1.0.3 ``switch.dart``:
``Row(mainAxisSize: min, children: [switch, Text(label)])``), so a label longer than the room beside the
switch runs past the edge instead of wrapping. ``switch_row`` puts the label beside the switch as its own
expanded Text, which wraps; a tap on it toggles the switch like the built-in label does. Chat settings rows
(owner, 2026-10-19: a label ran under the "Inherited from" badge) are built with it too.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional, Sequence

import flet as ft

__all__ = ["switch_row"]


def switch_row(switch: ft.Switch, label: str, *, trailing: Sequence[ft.Control] = (),
               key: Optional[str] = None) -> ft.Row:
    """``switch`` with ``label`` wrapping beside it (``trailing`` after the label). The switch keeps its own
    key, value and ``on_change``; a tap on the label flips the value and calls that ``on_change``."""

    def toggle(e: Any = None) -> None:
        if switch.disabled:
            return
        switch.value = not bool(switch.value)
        try:
            switch.update()
        except Exception:  # not on a page yet (rows built offscreen)
            pass
        handler = switch.on_change
        if handler is not None:
            result = handler(ft.Event("change", switch))
            if asyncio.iscoroutine(result):
                asyncio.ensure_future(result)

    text = ft.Container(content=ft.Text(label), expand=True, on_click=toggle)
    return ft.Row([switch, text, *trailing], spacing=8, vertical_alignment=ft.CrossAxisAlignment.CENTER, key=key)
