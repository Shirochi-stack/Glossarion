"""Hardware-keyboard shortcuts (UI_SPEC §2.4 / §7.6; desktop zoom shortcuts, FEATURE_MAP main-window #73).

Tablets, Chromebooks and the desktop dev window: ``page.on_keyboard_event`` (installed by
``GlossarionApp._install_keyboard``) maps

* Ctrl+= (or Ctrl++ / numpad +) → one text-size step up,
* Ctrl+- (numpad -) → one step down,
* Ctrl+0 (numpad 0) → back to the default size,

to whatever is on screen: the open Reader changes its Aa font size
(``ReaderScreen.keyboard_font_size``), the chat changes the current chat's Text size
(``ChatView.keyboard_text_scale``, the per-chat sidecar ``text_scale`` clamped to
``CHAT_TEXT_SCALE_RANGE``). Other screens ignore the keys. Cmd (meta) works like Ctrl on
macOS / iPadOS keyboards. Enter-to-send stays the composer's own ``shift_enter``.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

__all__ = ["KeyboardShortcuts", "zoom_step"]

log = logging.getLogger("glossarion.keyboard")

_ZOOM_IN = frozenset({"=", "+", "add", "numpad add", "equal", "plus"})
_ZOOM_OUT = frozenset({"-", "minus", "subtract", "numpad subtract", "_"})
_ZOOM_RESET = frozenset({"0", "numpad 0", "digit 0", ")"})


def zoom_step(key: Any, *, ctrl: bool = False, meta: bool = False, alt: bool = False) -> Optional[int]:
    """+1 / -1 / 0 for a zoom shortcut, None for any other key (Alt combinations are not zoom)."""
    if not (ctrl or meta) or alt:
        return None
    name = str(key or "").strip().lower()
    if name in _ZOOM_IN:
        return 1
    if name in _ZOOM_OUT:
        return -1
    if name in _ZOOM_RESET:
        return 0
    return None


class KeyboardShortcuts:
    def __init__(self, app: Any) -> None:
        self.app = app
        self.handled: list = []  # (target, step) of applied shortcuts (diagnostics / tests)

    def target(self) -> tuple:
        """("reader", screen) / ("chat", chat_view) / (None, None) for what is on screen."""
        shell = getattr(self.app, "shell", None)
        top = getattr(shell, "top_screen", None) if shell is not None else None
        if top is not None:
            if callable(getattr(top, "keyboard_font_size", None)):
                return "reader", top
            return None, None
        chat_view = getattr(self.app, "chat_view", None)
        if chat_view is not None and callable(getattr(chat_view, "keyboard_text_scale", None)):
            return "chat", chat_view
        return None, None

    def on_keyboard_event(self, e: Any) -> Optional[int]:
        step = zoom_step(getattr(e, "key", None), ctrl=bool(getattr(e, "ctrl", False)),
                         meta=bool(getattr(e, "meta", False)), alt=bool(getattr(e, "alt", False)))
        if step is None:
            return None
        kind, target = self.target()
        if target is None:
            return None
        try:
            if kind == "reader":
                target.keyboard_font_size(step)
            else:
                target.keyboard_text_scale(step)
        except Exception:
            log.exception("keyboard text-size shortcut failed")
            return None
        self.handled.append((kind, step))
        return step
