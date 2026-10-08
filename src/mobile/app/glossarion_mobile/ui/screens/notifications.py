"""Settings › Notifications & background (``/settings/notifications``, UI_SPEC §4.15 General, §1.9).

What the phone needs so long jobs keep going and report back:

* the notification permission (Android 13+ asks; iOS asks when notifications are set up) through the
  same request the Welcome step 4 and the first Run tap use (``BackgroundExecution``); the status line
  shows the real state (``BackgroundExecution.notification_status``: On / Off / Blocked in system
  settings), **Open system settings** appears while it is off or blocked (a permanently denied
  permission is only changed there) and **Send a test notification** posts one on ``jobs.done``;
* the three job channels (``services.notifications``): ``jobs.progress`` (the ongoing Android
  foreground-service notification), ``jobs.done`` (Done / Stopped / Failed with Open and Share) and
  ``jobs.action`` (glossary review needed with Accept / Review, sign-in required, paused by the system);
* Glossary review: "Always accept generated glossaries", the global value of the chat setting (Prefs
  ``AUTO_ACCEPT_GLOSSARY_PREF``; a chat can override it in its settings sheet);
* Android: the battery-optimisation exemption (``IGNORE_BATTERY_OPTIMIZATIONS``) and the "keep the
  screen on during jobs" preference ``BackgroundExecution`` reads (``PREF_KEEP_SCREEN_ON``);
* iOS: the background limits (a job pauses about 30 s after the app leaves the screen, iOS 26 can
  continue it with a system progress indicator; a paused job resumes from its saved progress).

No logic lives here: the buttons call the feature's ``request_notifications`` / ``request_battery`` and
``BackgroundExecution`` (status, system settings, test notification, Prefs).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Optional

import flet as ft

from glossarion_mobile.ui.components.reason_chip import ReasonChip
from glossarion_mobile.ui.screens.page_base import PageScreen, section

__all__ = ["CHANNEL_ROWS", "IOS_NOTE", "NotificationsScreen", "STATE_LABELS"]

log = logging.getLogger("glossarion.pages")

#: (channel id, title, what it shows) - UI_SPEC §1.9.
CHANNEL_ROWS = (
    ("jobs.progress", "Job progress", "The ongoing notification while a job runs (Android keeps the job alive with "
                                      "it, and posts it again if you swipe it away). Stop in the notification stops "
                                      "the job gracefully."),
    ("jobs.done", "Job finished", "Done / Stopped / Failed, with Open and Share when the job made one output."),
    ("jobs.action", "Needs you", "Glossary ready for review (Accept or Review), a question to answer, sign-in "
                                 "required, paused by the system (tap to resume)."),
)
#: ``notification_state`` -> the status line (UI_SPEC §4.15).
STATE_LABELS = {
    "on": "On",
    "off": "Off: Allow notifications asks again",
    "blocked": "Blocked in system settings: open them to turn notifications on",
    "unknown": "Unknown",
    "unavailable": "Phone only",
}
AUTO_ACCEPT_HELP = ("Skip the Edit / Yes / No card: translation starts with the generated glossary. Applies to new "
                    "sends; a chat can change it in its settings. The desktop always asks.")
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
        self.test_status = ft.Text("", theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT,
                                   visible=False, key="notif-test-status")
        self.results: dict = {}
        self.permission_state: Optional[str] = None  # notification_state of the last status read

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

    def _pref(self, key: str, default: Any = None) -> Any:
        reader = getattr(self.background, "_pref", None)
        if not callable(reader):
            return default
        try:
            return reader(key, default)
        except Exception:
            return default

    def _has(self, name: str) -> bool:
        return callable(getattr(self.background, name, None))

    def build_body(self) -> ft.Control:
        from glossarion_mobile.services.background import PREF_NOTIFICATION_ASKED
        from glossarion_mobile.ui.chat.direct_text_rules import AUTO_ACCEPT_GLOSSARY_PREF

        if self._has("notification_status"):
            self.notification_status.value = "Checking the notification permission…"  # did_show reads it
        else:
            asked = bool(self._pref(PREF_NOTIFICATION_ASKED, False))
            self.notification_status.value = ("Glossarion asked for notification permission before. Tap again if "
                                              "you turned it off in the system settings." if asked else
                                              "Not asked yet: the first Run asks too.")
        allow = ft.FilledTonalButton(content="Allow notifications", icon=ft.Icons.NOTIFICATIONS_ACTIVE_OUTLINED,
                                     on_click=lambda e: self.spawn(self.allow_notifications()),
                                     disabled=not (self.native and self.request_notifications is not None),
                                     key="notif-allow")
        self.open_settings_button = ft.OutlinedButton(
            content="Open system settings", icon=ft.Icons.SETTINGS_OUTLINED,
            on_click=lambda e: self.spawn(self.open_system_settings()),
            visible=False, key="notif-open-settings")
        self.test_button = ft.TextButton(
            content="Send a test notification", icon=ft.Icons.NOTIFICATION_ADD_OUTLINED,
            on_click=lambda e: self.spawn(self.send_test()),
            disabled=not (self.native and self._has("send_test_notification")), key="notif-test")
        channels = [
            ft.ListTile(title=ft.Text(title), subtitle=ft.Text(text, theme_style=ft.TextThemeStyle.BODY_SMALL),
                        leading=ft.Icon(ft.Icons.NOTIFICATIONS_OUTLINED), dense=True, key=f"notif-channel-{cid}")
            for cid, title, text in CHANNEL_ROWS
        ]
        rows: list = [ft.Row([allow, self.open_settings_button], wrap=True, spacing=8), self.notification_status,
                      self.test_button, self.test_status]
        if not self.native:
            rows.append(ReasonChip(reason=NO_NATIVE_REASON, detail="Notifications and background execution are run by "
                                                                   "the Android / iOS app."))
        controls = [section("Notifications", rows + channels, key="notif-card",
                            subtitle="Progress, finished jobs and jobs that need you.")]
        self.auto_accept = ft.Switch(label="Always accept generated glossaries",
                                     value=bool(self._pref(AUTO_ACCEPT_GLOSSARY_PREF, False)),
                                     on_change=self._on_auto_accept, disabled=not self._has("_set_pref"),
                                     key="notif-auto-accept")
        controls.append(section("Glossary review", [
            self.auto_accept,
            ft.Text(AUTO_ACCEPT_HELP, theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT),
        ], key="notif-glossary-review", subtitle="When a chat's generated glossary is ready (\"Glossary ready: "
                                                 "review needed\")."))
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

    def did_show(self) -> None:
        if self._has("notification_status"):
            self.spawn(self.refresh_status())

    def app_resumed(self) -> None:
        """Back from "Open system settings" (or anywhere else): the permission may have changed."""
        self.did_show()

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

    async def refresh_status(self) -> Optional[str]:
        """The status line from the real permission state (read fresh: the user may have changed it in the
        system settings); "Open system settings" while it is off or blocked."""
        from glossarion_mobile.services.background import notification_state

        reader = getattr(self.background, "notification_status", None)
        if not callable(reader):
            return None
        try:
            info = await reader()
        except Exception as exc:
            log.info("reading the notification status failed: %s", exc)
            info = {}
        state = notification_state(info)
        self.permission_state = state
        self.results["status"] = dict(info) if isinstance(info, dict) else {}
        self.notification_status.value = f"Notifications: {STATE_LABELS.get(state, state)}"
        if getattr(self, "open_settings_button", None) is not None:
            self.open_settings_button.visible = self.native and state in ("off", "blocked")
        self.push(self.notification_status, getattr(self, "open_settings_button", None))
        return state

    async def allow_notifications(self) -> str:
        status = await self._await(self.request_notifications)
        self.results["notifications"] = status
        if await self.refresh_status() is None:
            self.notification_status.value = f"Notification permission: {status}"
            self.push(self.notification_status)
        return status

    async def open_system_settings(self) -> bool:
        opener = getattr(self.background, "open_notification_settings", None)
        opened = False
        if callable(opener):
            try:
                opened = bool(await opener())
            except Exception as exc:
                log.info("opening the system settings failed: %s", exc)
        self.results["settings"] = opened
        if not opened:
            self.say("The system settings could not be opened: open Settings › Apps › Glossarion › Notifications")
        return opened

    async def send_test(self) -> bool:
        sender = getattr(self.background, "send_test_notification", None)
        ok = False
        if callable(sender):
            try:
                ok = bool(await sender())
            except Exception as exc:
                log.info("the test notification failed: %s", exc)
        self.results["test"] = ok
        self.test_status.value = ("Test notification sent." if ok else
                                  "The system did not show it: notifications are off or blocked for Glossarion.")
        self.test_status.visible = True
        self.push(self.test_status)
        if not ok:
            await self.refresh_status()
        return ok

    def _on_auto_accept(self, e: Any = None) -> None:
        from glossarion_mobile.ui.chat.direct_text_rules import AUTO_ACCEPT_GLOSSARY_PREF

        value = bool(getattr(getattr(e, "control", None), "value", self.auto_accept.value))
        setter = getattr(self.background, "_set_pref", None)
        if callable(setter):
            setter(AUTO_ACCEPT_GLOSSARY_PREF, value)

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
