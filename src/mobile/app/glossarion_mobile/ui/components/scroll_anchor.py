"""A ``scroll_to(scroll_key=…)`` target that never holds the content it marks.

Flet 1.0.3's ``ControlWidget.build`` (``packages/flet/lib/src/controls/control_widget.dart``) turns a
control's ``ft.ScrollKey`` into a **new** Flutter ``GlobalKey()`` on every build. Every rebuild of
that control's parent (a ``ListView``'s ``LayoutBuilder`` when the soft keyboard shrinks the
viewport, a bottom sheet following the keyboard inset, any ``update()`` of an ancestor) therefore
gives the keyed widget a different key, and Flutter unmounts the whole keyed subtree and mounts a
fresh one. A ``TextField`` inside it loses its ``FocusNode`` and the keyboard closes the moment it
opens (owner: "the keyboard pops out as soon as it pops in" in the chat card's Run options).

``anchored`` keeps the ScrollKey on a zero-size marker *beside* the content: the marker is the
only widget that is re-created, ``Scrollable.ensureVisible`` still finds the content's top edge
through it, and the content keeps its Flutter elements (focus, caret, scroll state) across rebuilds.
"""

from __future__ import annotations

from typing import Any, Sequence

import flet as ft

__all__ = ["anchored_column", "scroll_anchor"]


def scroll_anchor(key: Any) -> ft.Container:
    """A zero-height marker keyed ``ft.ScrollKey(key)`` (put it first, at the top of what it marks)."""
    return ft.Container(key=ft.ScrollKey(str(key)), height=0, width=0)


def anchored_column(key: Any, controls: Sequence[ft.Control], **kwargs: Any) -> ft.Column:
    """``controls`` in a Column whose first child is ``scroll_anchor(key)``: no gap, the children
    stretched to the column width as a single keyed child would be."""
    kwargs.setdefault("spacing", 0)
    kwargs.setdefault("tight", True)
    kwargs.setdefault("horizontal_alignment", ft.CrossAxisAlignment.STRETCH)
    return ft.Column([scroll_anchor(key), *controls], **kwargs)
