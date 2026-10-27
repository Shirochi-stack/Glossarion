"""DraggableSheet: Flutter's ``DraggableScrollableSheet`` for a ``BottomSheet``'s content (hotfix 5).

Flet 1.0.3 ships no draggable sheet, and a height driven from Python follows the finger one round trip
late. The Dart side (``src/flutter/flet_glossarion_native/lib/src/draggable_sheet.dart``) builds it
natively: the sheet opens at ``initial_size`` of the height the BottomSheet gives it (``scrollable=True``),
a swipe up on the rows grows it to ``max_size`` before they scroll, a swipe down shrinks it (it snaps back
to ``initial_size`` when let go above it) and closes the BottomSheet at ``min_size``. ``header`` scrolls
with the rows; ``footer`` stays pinned under them.

The control exists only in apps built with this extension (Android / iOS). Elsewhere the client shows an
unknown-control box, so callers check ``available`` on their page first and build a plain column there.
"""

from __future__ import annotations

from dataclasses import field
from typing import Any, Optional

import flet as ft

__all__ = ["DraggableSheet", "native_widgets_available"]

_MOBILE_PLATFORMS = ("android", "ios")


@ft.control("GlossarionDraggableSheet")
class DraggableSheet(ft.LayoutControl):
    """Rows in a sheet that grows with a swipe up and closes with a swipe down."""

    controls: list[ft.Control] = field(default_factory=list)
    """The rows (built lazily by the client, like ``ListView``)."""

    header: Optional[ft.Control] = None
    """Above the rows, scrolling with them."""

    footer: Optional[ft.Control] = None
    """Pinned under the rows."""

    initial_size: ft.Number = 0.5
    """Height when the sheet opens, as a fraction of the height the BottomSheet offers."""

    min_size: ft.Number = 0.25
    """A swipe down to this fraction closes the BottomSheet."""

    max_size: ft.Number = 1.0
    """A swipe up grows the sheet up to this fraction."""

    snap: bool = True
    """Let go between sizes: settle at ``initial_size``, ``max_size`` or close."""


def native_widgets_available(page: Any) -> bool:
    """True when ``page``'s client has this extension's widgets: the built Android / iOS app (never the
    desktop or web client of ``flet run``, nor a run with ``GLOSSARION_NATIVE_DISABLE`` set)."""
    import os

    if os.environ.get("GLOSSARION_NATIVE_DISABLE", "").strip().lower() in ("1", "true", "yes", "on"):
        return False
    if page is None or getattr(page, "web", False):
        return False
    platform = getattr(page, "platform", None)
    value = str(getattr(platform, "value", platform) or "").lower()
    return value in _MOBILE_PLATFORMS
