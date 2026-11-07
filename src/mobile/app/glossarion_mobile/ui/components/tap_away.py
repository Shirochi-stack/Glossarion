"""A dropdown closes when the user taps away from it (owner 2026-10-31: the response card's Run options).

``closes_on_tap_away(tile)`` returns what goes in the tile's place: the tile inside the extension's ``TapAway``
on the phone app, the bare tile on the desktop and web clients of ``flet run``, which do not have that control
(``install_tap_away`` decides once, at app start, like ``keyboard_dismiss``).

Only a dropdown the user opened closes this way: the running card's Requests list opens by itself for the user
to watch, and stays open until the user has toggled it. The region is armed by the tile's own ``on_change``.

The region (and its tile) is built once and re-added by every repaint of its card. When a card is rebuilt around it
under the same key (a new Plan card, a Result card whose rows changed), Flet diffs the keyed pair as frozen and
marks the region it meets at a new place in that card frozen for good; setting ``enabled`` then raised "Frozen
controls cannot be updated", which Flet reports as a crashed app (owner 2026-11-07: the numeric keyboard of
Run options' Output token limit). The handlers lift that marker first, as ``dialogs.close_dialog`` does for a
keyed sheet and Flet's own ``use_dialog`` hook does for its dialogs.
"""

from __future__ import annotations

from typing import Any, Optional

import flet as ft

from glossarion_mobile.ui.components._handlers import call_handler
from glossarion_mobile.ui.components.dialogs import _unmark
from glossarion_mobile.ui.components.drag_sheet import native_widget_class

__all__ = ["closes_on_tap_away", "install_tap_away"]

_REGION = "_glossarion_tap_away_region"
#: the extension's ``TapAway``; None until the app installs it, and on a client without it
_region_class: Any = None


def install_tap_away(page: Any) -> None:
    """Look up the extension's control for ``page``'s client; once, at app start."""
    global _region_class
    _region_class = native_widget_class(page, "TapAway")


def _thaw(*controls: Any) -> None:
    """Lift the frozen marker a keyed rebuild of the card left (see the module docstring)."""
    for control in controls:
        _unmark(control, "_frozen")


def _push(control: ft.Control) -> None:
    try:
        control.update()
    except RuntimeError:  # not on a page (yet): it goes out with its first build
        pass


def closes_on_tap_away(tile: ft.ExpansionTile) -> ft.Control:
    """``tile`` in a region that closes it on a tap away, the same region every time; ``tile`` without the
    extension's control."""
    region: Optional[Any] = getattr(tile, _REGION, None)
    if region is not None or _region_class is None:
        return region if region is not None else tile
    own_change = tile.on_change

    def on_change(e: Any) -> None:
        call_handler(own_change, e)
        opened = str(getattr(e, "data", "")).strip().lower() == "true"
        if region.enabled != opened:
            _thaw(region)
            region.enabled = opened
            _push(region)

    def on_tap_away(_e: Any = None) -> None:
        _thaw(region, tile)
        region.enabled = False
        tile.expanded = False
        _push(region)

    tile.on_change = on_change
    region = _region_class(content=tile, enabled=False, on_tap_away=on_tap_away)
    object.__setattr__(tile, _REGION, region)
    return region
