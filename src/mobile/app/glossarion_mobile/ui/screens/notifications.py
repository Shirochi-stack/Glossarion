"""Settings › Notifications & background (``/settings/notifications``, UI_SPEC §4.15 General, §1.9).

What the phone needs so long jobs keep going and report back:

* the notification permission (Android 13+ asks; iOS asks when notifications are set up) through the
  same request the Welcome step 4 and the first Run tap use (``BackgroundExecution``);
* the three job channels (``services.notifications``): ``jobs.progress`` (the ongoing Android
  foreground-service notification), ``jobs.done`` (Done / Stopped / Failed with Open and Share) and
  ``jobs.action`` (glossary review needed, sign-in required, paused by the system);
* Android: the battery-optimisation exemption (``IGNORE_BATTERY_OPTIMIZATIONS``) and the "keep the
  screen on during jobs" preference ``BackgroundExecution`` reads (``PREF_KEEP_SCREEN_ON``);
* iOS: the background limits (a job pauses about 30 s after the app leaves the screen, iOS 26 can
  continue it with a system progress indicator; a paused job resumes from its saved progress).

No logic lives here: the buttons call the feature's ``request_notifications`` / ``request_battery``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Optional

import flet as ft

from glossarion_mobile.ui.components.reason_chip import ReasonChip
from glossarion_mobile.ui.screens.page_base import PageScreen, section

__all__ = ["CHANNEL_ROWS", "IOS_NOTE", "NotificationsScreen"]

log = logging.getLogger("glossarion.pages")

#: (channel id, title, what it shows) - UI_SPEC §1.9.
CHANNEL_ROWS = (
    ("jobs.progress", "Job progress", "The ongoing notification while a job runs (Android keeps the job alive with "
                                      "it). Stop in the notification stops the job gracefully."),
    ("jobs.done", "Job finished", "Done / Stopped / Failed, with Open and Share when the job made one output."),
    ("jobs.action", "Needs you", "Glossary ready for review, sign-in required, paused by the system (tap to "
                                 "resume)."),
)
IOS_NOTE = ("iPhone and iPad: iOS pauses a running translation about 30 seconds after you leave the app. On iOS 26 "
            "the job can continue in the background with a system progress indicator. A paused job resumes from its "
            "saved progress (translation_progress.json) when you come back.")
NO_NATIVE_REASON = "Phone only"


class NotificationsScreen(PageScreen):
    title = "Notifications & background"

    def __init__(self, match: Any, ctx: Any, *, background: Any = None,
                 request_notifications: Optional[Callable[[], Any]] = None,
                 request_battery: Optional[Callable[[], Any]] = None) -> None:
        super().__init__(match, ctx)
        self.background = background
        self.request_notifications = request_notifications
        self.request_battery = request_battery
        self.notification_status = ft.Text("", theme_style=ft.TextThemeStyle.BODY_SMALL,
                                           color=ft.Colors.ON_SURFACE_VARIANT, key="notif-status")
        self.battery_status = ft.Text("", theme_style=ft.TextThemeStyle.BODY_SMALL,
                                      color=ft.Colors.ON_SURFACE_VARIANT, key="notif-battery-status")
        self.results: dict = {}

    @property
    def is_android(self) -> bool:
        return bool(getattr(self.background, "is_android", False))

    @property
    def is_ios(self) -> bool:
        return bool(getattr(self.background, "is_ios", False))

    @property
    def native(self) -> bool:
        return self.is_android or self.is_ios

    def _keep_screen_on(self) -> bool:
        keep = getattr(self.background, "keep_screen_on", None)
        try:
            return bool(keep()) if callable(keep) else False
        except Exception:
            return False

    def build_body(self) -> ft.Control:
        from glossarion_mobile.services.background import PREF_NOTIFICATION_ASKED

        asked = False
        reader = getattr(self.background, "_pref", None)
        if callable(reader):
            try:
                asked = bool(reader(PREF_NOTIFICATION_ASKED, False))
            except Exception:
                asked = False
        self.notification_status.value = ("Glossarion asked for notification permission before. Tap again if you "
                                          "turned it off in the system settings." if asked else
                                          "Not asked yet: the first Run asks too.")
        allow = ft.FilledTonalButton(content="Allow notifications", icon=ft.Icons.NOTIFICATIONS_ACTIVE_OUTLINED,
                                     on_click=lambda e: self.spawn(self.allow_notifications()),
                                     disabled=not (self.native and self.request_notifications is not None),
                                     key="notif-allow")
        channels = [
            ft.ListTile(title=ft.Text(title), subtitle=ft.Text(text, theme_style=ft.TextThemeStyle.BODY_SMALL),
                        leading=ft.Icon(ft.Icons.NOTIFICATIONS_OUTLINED), dense=True, key=f"notif-channel-{cid}")
            for cid, title, text in CHANNEL_ROWS
        ]
        rows: list = [allow, self.notification_status]
        if not self.native:
            rows.append(ReasonChip(reason=NO_NATIVE_REASON, detail="Notifications and background execution are run by "
                                                                   "the Android / iOS app."))
        controls = [section("Notifications", rows + channels, key="notif-card",
                            subtitle="Progress, finished jobs and jobs that need you.")]
        background_rows: list = []
        if self.is_android or not self.native:
            background_rows += [
                ft.Text("Long translations keep running with the screen off when battery optimisation is disabled "
                        "for Glossarion.", theme_style=ft.TextThemeStyle.BODY_MEDIUM),
                ft.FilledTonalButton(content="Disable battery optimisation", icon=ft.Icons.BATTERY_CHARGING_FULL,
                                     on_click=lambda e: self.spawn(self.allow_battery()),
                                     disabled=not (self.is_android and self.request_battery is not None),
                                     key="notif-battery"),
                self.battery_status,
            ]
        self.keep_awake = ft.Switch(label="Keep the screen on while a job runs", value=self._keep_screen_on(),
                                    on_change=self._on_keep_awake, disabled=self.background is None,
                                    key="notif-keep-awake")
        background_rows.append(self.keep_awake)
        background_rows.append(ft.Text(IOS_NOTE, theme_style=ft.TextThemeStyle.BODY_SMALL,
                                       color=ft.Colors.ON_SURFACE_VARIANT, key="notif-ios-note"))
        controls.append(section("Background", background_rows, key="notif-background"))
        return self.scaffold(controls)

    async def _await(self, fn: Optional[Callable[[], Any]]) -> str:
        if fn is None:
            return "unavailable"
        try:
            result = fn()
            if asyncio.iscoroutine(result):
                result = await result
            return str(result)
        except Exception as exc:
            log.info("permission request failed: %s", exc)
            return f"error: {exc}"

    async def allow_notifications(self) -> str:
        status = await self._await(self.request_notifications)
        self.results["notifications"] = status
        self.notification_status.value = f"Notification permission: {status}"
        self.push(self.notification_status)
        return status

    async def allow_battery(self) -> str:
        status = await self._await(self.request_battery)
        self.results["battery"] = status
        self.battery_status.value = f"Battery optimisation exemption: {status}"
        self.push(self.battery_status)
        return status

    def _on_keep_awake(self, e: Any = None) -> None:
        from glossarion_mobile.services.background import PREF_KEEP_SCREEN_ON

        value = bool(getattr(getattr(e, "control", None), "value", self.keep_awake.value))
        setter = getattr(self.background, "_set_pref", None)
        if callable(setter):
            setter(PREF_KEEP_SCREEN_ON, value)
