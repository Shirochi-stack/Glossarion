"""HiddenWebViewHost: WebViews that must run but never be seen (the WebViewBridge's pages).

A flet-webview page only runs while its WebView is mounted, laid out and painted:

* an unmounted, ``visible=False`` or ``Offstage`` WebView is never created;
* a WebView that is laid out but never painted (``opacity=0``, or placed fully off-screen
  where the view system skips drawing it) can be detached from the window (iOS) or stop
  producing frames (Android), which pauses timers and ``requestAnimationFrame`` that hCaptcha
  and Google's page rely on.

So each hidden page gets its own ``page.overlay`` entry (positioned children are allowed
there): a 1 x 1 logical-pixel container at the top-left corner whose Stack lays the WebView
out at its full viewport size and clips it to that single pixel. The WebView is therefore
mounted, sized and painted, but shows at most one pixel, at 1 % opacity. The entry ignores
pointer input and is excluded from the accessibility tree. Every entry has its own key (a
Flet 1.0.3 control re-rendered under an old key keeps its old state).

All methods run on the Flet loop thread (the WebViewBridge posts them there).
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

import flet as ft

__all__ = ["HIDDEN_OPACITY", "HiddenWebViewHost"]

log = logging.getLogger("glossarion.hidden_webview")

#: Opacity of an entry: painted (a zero-opacity subtree is never painted) but invisible.
HIDDEN_OPACITY = 0.01


class HiddenWebViewHost:
    """Mounts hidden controls (WebViews) in ``page.overlay``; loop thread only."""

    def __init__(self, page: Any) -> None:
        self.page = page
        self.entries: dict[str, ft.Control] = {}

    def build_entry(self, control: ft.Control, *, size: Tuple[int, int], key: str) -> ft.Container:
        """The overlay entry that keeps ``control`` laid out at ``size`` but hidden."""
        width, height = (max(1, int(size[0])), max(1, int(size[1])))
        frame = ft.Container(content=control, left=0, top=0, width=width, height=height)
        return ft.Container(
            key=key,
            left=0,
            top=0,
            width=1,
            height=1,
            opacity=HIDDEN_OPACITY,
            ignore_interactions=True,
            content=ft.Semantics(
                exclude_semantics=True,
                content=ft.Stack(controls=[frame], clip_behavior=ft.ClipBehavior.HARD_EDGE),
            ),
        )

    def add(self, control: ft.Control, *, size: Tuple[int, int], key: str) -> ft.Container:
        """Mount ``control`` hidden; returns its entry (pass it to :meth:`remove`)."""
        old = self.entries.pop(key, None)
        if old is not None:
            self._detach(old)
        entry = self.build_entry(control, size=size, key=key)
        self.entries[key] = entry
        self.page.overlay.append(entry)
        self._update()
        return entry

    def remove(self, entry: Optional[ft.Control]) -> bool:
        """Unmount an entry (disposes its WebView); False when it was not mounted."""
        if entry is None:
            return False
        key = getattr(entry, "key", None)
        if key is not None and self.entries.get(key) is entry:
            del self.entries[key]
        if not self._detach(entry):
            return False
        self._update()
        return True

    def clear(self) -> int:
        """Unmount every entry; returns how many were mounted."""
        entries = list(self.entries.values())
        self.entries.clear()
        removed = sum(1 for entry in entries if self._detach(entry))
        if removed:
            self._update()
        return removed

    def __len__(self) -> int:
        return len(self.entries)

    def _detach(self, entry: ft.Control) -> bool:
        try:
            self.page.overlay.remove(entry)
        except ValueError:
            return False
        return True

    def _update(self) -> None:
        try:
            self.page.update()
        except Exception as exc:  # the page is closing; the overlay change is moot then
            log.debug("hidden webview host update failed: %s", exc)
