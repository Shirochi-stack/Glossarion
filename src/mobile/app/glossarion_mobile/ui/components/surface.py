"""Where a surface opens on each size class (UI_SPEC §1.1, §5.1, §7.2).

On tablets (>= 900 dp) persistent tasks open in the shell's right **SidePanel** instead of a
bottom sheet: chat settings, job detail, compare and the glossary term sheet. Wide screens
(>= 1200 dp) also lay Settings, the Book page and Manga out master-detail.

The shell registers itself as the page's *host* (``register(page, host)``); components and
screens ask through this module, so ``components`` never imports the shell:

* ``size_class(page)`` / ``is_tablet(page)`` / ``is_wide(page)``: the shell's current size
  class (without a host: from ``page.width``);
* ``present_sheet(page, dialog, title=…)``: a ``BottomSheet``'s content in the SidePanel on
  tablets (True), else nothing happens (False: the caller shows the sheet);
* ``dismiss(page, owner)``: closes the panel when it holds ``owner``;
  ``components.dialogs.close_dialog`` calls it first, so every existing close path (a
  sheet's Save / Cancel / ✕) also closes the panel copy;
* ``open_route_in_panel(page, match)``: a whole screen (job detail) in the SidePanel.

The host protocol (``AppShell``): ``size_class``, ``present(content, *, title, owner,
on_close) -> bool``, ``dismiss(owner) -> bool``, ``hosts(owner) -> bool`` and
``open_screen_in_panel(match) -> bool``.
"""

from __future__ import annotations

import logging
import weakref
from typing import Any, Callable, Optional

from glossarion_mobile.ui.responsive import SizeClass, size_class_for

__all__ = [
    "dismiss",
    "host_for",
    "hosts",
    "is_tablet",
    "is_wide",
    "open_route_in_panel",
    "present_sheet",
    "register",
    "size_class",
    "unregister",
]

log = logging.getLogger("glossarion.ui")

_HOSTS: dict[int, Any] = {}


def register(page: Any, host: Any) -> None:
    """``host`` (the AppShell) answers for ``page`` from now on."""
    if page is None:
        return
    try:
        _HOSTS[id(page)] = weakref.ref(host)
    except TypeError:
        _HOSTS[id(page)] = lambda h=host: h


def unregister(page: Any, host: Any = None) -> None:
    ref = _HOSTS.get(id(page))
    if ref is not None and (host is None or ref() is host):
        _HOSTS.pop(id(page), None)


def host_for(page: Any) -> Any:
    if page is None:
        return None
    ref = _HOSTS.get(id(page))
    return ref() if ref is not None else None


def size_class(page: Any) -> SizeClass:
    host = host_for(page)
    value = getattr(host, "size_class", None) if host is not None else None
    if isinstance(value, SizeClass):
        return value
    return size_class_for(getattr(page, "width", None) or 0)


def is_tablet(page: Any) -> bool:
    """A persistent sidebar and the SidePanel (>= 900 dp)."""
    return size_class(page).persistent_sidebar


def is_wide(page: Any) -> bool:
    """Master-detail layouts (>= 1200 dp)."""
    return size_class(page) is SizeClass.WIDE


def _sheet_content(dialog: Any) -> Any:
    content = getattr(dialog, "content", None)
    return content if content is not None else dialog


def present_sheet(
    page: Any,
    dialog: Any,
    *,
    title: str,
    owner: Any = None,
    on_close: Optional[Callable[[], Any]] = None,
) -> bool:
    """Show ``dialog``'s content in the SidePanel on tablets; False when the caller should open
    the sheet itself (phones, or no shell)."""
    host = host_for(page)
    if host is None or not is_tablet(page):
        return False
    present = getattr(host, "present", None)
    if not callable(present):
        return False
    try:
        return bool(present(_sheet_content(dialog), title=title, owner=owner if owner is not None else dialog,
                            on_close=on_close))
    except Exception:
        log.exception("presenting %s in the side panel failed", title)
        return False


def hosts(page: Any, owner: Any) -> bool:
    """True while the SidePanel shows ``owner``'s content."""
    host = host_for(page)
    check = getattr(host, "hosts", None) if host is not None else None
    try:
        return bool(check(owner)) if callable(check) and owner is not None else False
    except Exception:
        return False


def dismiss(page: Any, owner: Any) -> bool:
    """Close the SidePanel when it shows ``owner``; True when it did."""
    host = host_for(page)
    if host is None or owner is None:
        return False
    close = getattr(host, "dismiss", None)
    try:
        return bool(close(owner)) if callable(close) else False
    except Exception:
        log.debug("dismissing from the side panel failed", exc_info=True)
        return False


def open_route_in_panel(page: Any, match: Any) -> bool:
    """Open a route's screen in the SidePanel on tablets (job detail); False: navigate instead."""
    host = host_for(page)
    if host is None or match is None or not is_tablet(page):
        return False
    opener = getattr(host, "open_screen_in_panel", None)
    try:
        return bool(opener(match)) if callable(opener) else False
    except Exception:
        log.exception("opening %s in the side panel failed", getattr(match, "route", match))
        return False
