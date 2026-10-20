"""Value fields commit on Done *and* on tapping away (owner: "why do all input fields require me to
click done. Shouldn't clicking away count as well").

``commit_on_blur(field, commit)`` wires one commit to the three ways an edit ends:

* ``on_submit`` — the keyboard's Done / Enter;
* ``on_tap_outside`` — a touch anywhere outside the focused field. On Android / iOS a touch outside
  does not take the focus away (Flutter unfocuses only for a mouse), so ``on_blur`` alone never ran
  when the owner tapped away;
* ``on_blur`` — the focus moved (another field, Done closing the keyboard, a dialog).

Only for VALUE fields (a setting, number, name or path that is stored). ACTION fields (send, search,
find, add-and-clear, sign-in) keep ``on_submit`` only: tapping away never runs an action.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Optional

__all__ = ["commit_on_blur"]

log = logging.getLogger("glossarion.ui")

_UNSET = object()


def commit_on_blur(field: Any, commit: Callable[[str], Any], *,
                   current: Optional[Callable[[], Any]] = None) -> Any:
    """Wire ``field`` (an ``ft.TextField``) so Done, a tap outside it and the loss of focus all run
    ``commit(text)``, and return the field.

    * One commit per edit: the text already handed to ``commit`` is not committed again (Done and
      the blur that follows it, several taps outside). Focusing the field again starts a new edit.
    * ``current()`` (optional) is the stored value: while the field shows it (and no error) nothing
      is written. ``None`` from it means "always commit".
    * ``commit`` keeps its own validation and error display (an invalid text shows the same error as
      on Done and leaves the stored value alone).
    * The field's own ``on_focus`` / ``on_blur`` (focus tracking) keep running first."""
    prior_focus = getattr(field, "on_focus", None)
    prior_blur = getattr(field, "on_blur", None)
    state = {"last": _UNSET}

    def run(e: Any = None) -> Any:
        text = str(getattr(field, "value", "") or "")
        if state["last"] is not _UNSET and state["last"] == text:
            return None
        state["last"] = text
        if current is not None and not getattr(field, "error", None):
            try:
                stored = current()
            except Exception:
                stored = None
            if stored is not None and text.strip() == str(stored).strip():
                return None
        return _call(commit, text)

    def on_focus(e: Any = None) -> Any:
        state["last"] = _UNSET
        return _call(prior_focus, e) if prior_focus is not None else None

    def on_blur(e: Any = None) -> Any:
        if prior_blur is not None:
            _call(prior_blur, e)
        return run(e)

    field.on_submit = run
    field.on_tap_outside = run
    field.on_blur = on_blur
    field.on_focus = on_focus
    return field


def _call(handler: Callable[..., Any], *args: Any) -> Any:
    try:
        result = handler(*args)
    except Exception:
        log.exception("field commit %s failed", getattr(handler, "__qualname__", handler))
        return None
    if asyncio.iscoroutine(result):
        return asyncio.ensure_future(result)
    return result
