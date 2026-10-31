"""TapAway: Flutter's ``TapRegion`` around a control, for a dropdown that closes when tapped away from.

Flet 1.0.3 exposes ``TapRegion`` only as a text field's ``on_tap_outside``. The Dart side
(``src/flutter/flet_glossarion_native/lib/src/tap_away.dart``) sends ``on_tap_away`` for a tap on the page
outside ``content``: not for a scroll that starts there, nor for a tap on a sheet, a dialog or an open dropdown
menu. ``enabled`` is off while the dropdown is shut, so other taps send nothing.

The control exists only in apps built with this extension (Android / iOS): callers check
``native_widgets_available`` on their page first and use the bare ``content`` elsewhere.
"""

# No ``from __future__ import annotations``: Flet resolves an event's type from the live ``on_tap_away``
# annotation, and a string there makes every event fail ("[resolve error] on_tap_away").
from typing import Optional

import flet as ft

__all__ = ["TapAway"]


@ft.control("GlossarionTapAway")
class TapAway(ft.LayoutControl):
    """``content``, reporting taps on the page outside it."""

    content: Optional[ft.Control] = None
    """The control a tap is away from."""

    enabled: bool = True
    """Off: taps are not looked at."""

    on_tap_away: Optional[ft.ControlEventHandler["TapAway"]] = None
    """Called for a tap on the page outside ``content``."""

    def before_update(self) -> None:
        # takes its content's visibility: a hidden dropdown leaves no spacing gap in its Column
        super().before_update()
        if self.content is not None:
            self.visible = self.content.visible
