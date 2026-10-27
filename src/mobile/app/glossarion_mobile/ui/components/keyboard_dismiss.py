"""A tap outside a text field closes the keyboard (hotfix 5, owner: "Update it so tapping a blank area
closes the keyboard").

Flutter takes the focus off a field on a tap outside it only for a mouse; on Android / iOS the field keeps
the focus and its keyboard (``field_commit`` works around the same gap for its commit), so the keyboard
stayed up until the system back gesture. ``install_tap_outside_dismiss(unfocus)`` hooks
``ft.TextField.before_update`` (Flet runs it for every field it sends, like ``sheet.install_close_buttons``
does for sheets), so every field's ``on_tap_outside`` runs the field's own handler first (``field_commit``'s
commit) and then ``unfocus()``: the extension's ``unfocus``, which drops the focus of the field that held it
at that touch and never that of a field the same tap focused (``NativeBridge.unfocus``).

"Outside" is Flutter's own test (``TextFieldTapRegion``): the field with its decoration, selection handles
and copy / paste toolbar is inside; blank space, the chat, another control and the start of a scroll are
outside. A field that keeps its keyboard for some outside taps says so with ``keeps_keyboard`` (the
composer while its slash-command list is open: a tap on a command fills the field).
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import flet as ft

from glossarion_mobile.ui.components._handlers import call_handler

__all__ = ["dismiss_keyboard", "install_tap_outside_dismiss", "keeps_keyboard"]

log = logging.getLogger("glossarion.ui")

_MARK = "_glossarion_tap_outside"
_KEEP = "_glossarion_keeps_keyboard"
_OWN = "_glossarion_own_tap_outside"
#: what drops the keyboard (``NativeBridge.unfocus``); None until the app installs the hook
_unfocus: Optional[Callable[[], Any]] = None


def keeps_keyboard(field: ft.TextField, keep: Callable[[], bool]) -> ft.TextField:
    """``field`` keeps its keyboard on a tap outside while ``keep()`` is true; returns the field."""
    object.__setattr__(field, _KEEP, keep)
    return field


def dismiss_keyboard() -> Any:
    """Drop the keyboard of the field focused at the last touch (a no-op before the hook is installed)."""
    return call_handler(_unfocus)


def _keeps(field: Any) -> bool:
    keep = getattr(field, _KEEP, None)
    if keep is None:
        return False
    try:
        return bool(keep())
    except Exception:
        log.exception("keeps_keyboard check failed")
        return False


def wire(field: Any) -> None:
    """Give ``field`` the dismissing ``on_tap_outside`` (idempotent; a handler set later is wrapped too)."""
    current = getattr(field, "on_tap_outside", None)
    if getattr(current, _MARK, False):
        return

    def on_tap_outside(e: Any = None) -> None:
        own = getattr(on_tap_outside, _OWN, None)
        if own is not None:
            call_handler(own, e)
        if not _keeps(field):
            dismiss_keyboard()

    setattr(on_tap_outside, _MARK, True)
    setattr(on_tap_outside, _OWN, current)
    field.on_tap_outside = on_tap_outside


def install_tap_outside_dismiss(unfocus: Callable[[], Any]) -> None:
    """Every ``ft.TextField`` closes its keyboard on a tap outside it, through ``unfocus``; once per process,
    at app start (a later call only swaps ``unfocus``)."""
    global _unfocus
    _unfocus = unfocus
    original = ft.TextField.before_update
    if getattr(original, _MARK, False):
        return

    def before_update(self: ft.TextField) -> None:
        wire(self)
        original(self)

    setattr(before_update, _MARK, True)
    ft.TextField.before_update = before_update
