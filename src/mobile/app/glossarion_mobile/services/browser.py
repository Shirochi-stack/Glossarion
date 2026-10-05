"""``webbrowser.open`` -> in-app browser (Custom Tabs / SFSafariViewController).

``runtime_bootstrap`` installs a preferred ``webbrowser`` controller whose
``open(url)`` calls the opener registered with ``rb.set_url_opener``. This is
that opener: it may be called from any thread (the unchanged desktop OAuth
flows call ``webbrowser.open`` on a worker thread) and hands the URL to
``UrlLauncher.launch_url`` on the Flet loop, in ``IN_APP_BROWSER_VIEW`` mode on
phones, falling back to the platform default.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import flet as ft

from glossarion_mobile.services.dispatcher import UiDispatcher

__all__ = ["InAppUrlOpener"]

log = logging.getLogger("glossarion.browser")


class InAppUrlOpener:
    def __init__(self, dispatcher: UiDispatcher, url_launcher: Any, *, is_mobile: bool) -> None:
        self.dispatcher = dispatcher
        self.url_launcher = url_launcher
        self.is_mobile = is_mobile
        self.opened: list[str] = []

    def open_threadsafe(self, url: str) -> None:
        """``rb.set_url_opener`` target (any thread)."""
        self.dispatcher.submit(self.launch, url)

    async def launch(self, url: str) -> None:
        self.opened.append(url)
        mode = ft.LaunchMode.IN_APP_BROWSER_VIEW if self.is_mobile else ft.LaunchMode.PLATFORM_DEFAULT
        try:
            await asyncio.wait_for(self.url_launcher.launch_url(url, mode=mode), 20)
        except Exception as exc:
            log.warning("launch_url(%s, %s) failed: %s; retrying with the platform default", url, mode, exc)
            try:
                await asyncio.wait_for(self.url_launcher.launch_url(url), 20)
            except Exception as exc2:
                log.error("launch_url fallback failed: %s", exc2)

    async def close_in_app_view(self) -> None:
        """Close SFSafariViewController on iOS (Android Custom Tabs cannot be closed)."""
        if not self.is_mobile:
            return
        try:
            await asyncio.wait_for(self.url_launcher.close_in_app_web_view(), 5)
        except Exception as exc:
            log.info("close_in_app_web_view: %s", exc)
