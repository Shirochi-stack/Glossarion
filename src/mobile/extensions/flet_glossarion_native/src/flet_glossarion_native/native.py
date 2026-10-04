"""GlossarionNative: Flet service for the Android/iOS pieces Flet does not ship.

The Dart side lives in ``src/flutter/flet_glossarion_native`` and only exists in
apps built with ``flet build`` (the extension is compiled into the Flutter
shell). Under ``flet run`` on Windows/macOS/Linux, on the web, or with the Flet
companion app, there is no Dart service: every method then returns a safe
default without touching the client, and never raises.
"""

import logging
import os
from typing import Any, Optional, Sequence, Union

import flet as ft

from flet_glossarion_native.types import (
    CHANNEL_JOBS_PROGRESS,
    DEFAULT_NOTIFICATION_CHANNELS,
    JOB_SERVICE_NOTIFICATION_ID,
    BackgroundTaskEvent,
    ForegroundEvent,
    NotificationAction,
    NotificationButton,
    NotificationChannel,
    NotificationEvent,
    ShareEvent,
    SharedItem,
)

__all__ = ["GlossarionNative", "EXTENSION_VERSION"]

logger = logging.getLogger("flet_glossarion_native")

EXTENSION_VERSION = "0.1.0"

#: Set to 1/true to force the safe-default behaviour even in a built app.
DISABLE_ENV = "GLOSSARION_NATIVE_DISABLE"

_DEFAULT_TIMEOUT = 20.0
_LONG_TIMEOUT = 900.0

_MOBILE_PLATFORMS = ("android", "ios")

# Messages Flet 1.0.3 produces when the control type has no Dart counterpart
# (flet_backend.dart _onInvokeMethod / control.dart invokeMethod).
_MISSING_SERVICE_MARKERS = (
    "invoke method listener",
    "no invoke method listeners",
    "inexistent control",
)


def _platform_name(platform: Any) -> Optional[str]:
    if platform is None:
        return None
    value = getattr(platform, "value", platform)
    return str(value).lower()


def _as_map(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    to_map = getattr(value, "to_map", None)
    if callable(to_map):
        return to_map()
    if isinstance(value, dict):
        return dict(value)
    raise TypeError(f"Cannot convert {type(value).__name__} to a map")


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    return bool(value)


@ft.control("GlossarionNative")
class GlossarionNative(ft.Service):
    """Android foreground service, notifications, share/open-with intake,
    MediaStore Downloads export and iOS background tasks.

    Create it once inside the Flet app (``main(page)``) with the event handlers
    set in the constructor so they are registered before the Dart service
    starts::

        native = GlossarionNative(on_share=..., on_foreground=..., on_notification=...)

    Shared items: everything received before Python subscribed (cold start) is
    returned by :meth:`get_initial_shared`; later arrivals fire ``on_share``.
    Items stay queued on the Dart side until :meth:`clear_shared` is called, so
    call it once the items have been imported. Items carry a unique ``id``.
    """

    on_share: Optional[ft.EventHandler[ShareEvent]] = None
    """Files/text received through Open-with / Share (Android) or Open in (iOS)."""

    on_foreground: Optional[ft.EventHandler[ForegroundEvent]] = None
    """Android foreground-service events: notification buttons, tap, timeout, destroyed."""

    on_background_task: Optional[ft.EventHandler[BackgroundTaskEvent]] = None
    """iOS background-task expiry and BGContinuedProcessingTask lifecycle."""

    on_notification: Optional[ft.EventHandler[NotificationEvent]] = None
    """A notification from show_notification() was tapped or one of its actions pressed."""

    def init(self):
        super().init()
        # Python-only state; deliberately not dataclass fields so it is never
        # serialized to the client.
        self._native_unavailable_reason: Optional[str] = None

    # ------------------------------------------------------------------ helpers

    def _page_or_none(self) -> Optional[Any]:
        try:
            return self.page
        except Exception:
            return None

    def _platform(self) -> Optional[str]:
        page = self._page_or_none()
        if page is None:
            return None
        return _platform_name(getattr(page, "platform", None))

    def _disabled_by_env(self) -> bool:
        return os.environ.get(DISABLE_ENV, "").strip().lower() in ("1", "true", "yes", "on")

    def _unavailable_reason(self) -> Optional[str]:
        """Why native calls are skipped right now, or None when they may run."""
        if self._disabled_by_env():
            return f"disabled by {DISABLE_ENV}"
        reason = getattr(self, "_native_unavailable_reason", None)
        if reason:
            return reason
        page = self._page_or_none()
        if page is None:
            return "service is not attached to a page"
        if getattr(page, "web", False):
            return "web client"
        platform = _platform_name(getattr(page, "platform", None))
        if platform not in _MOBILE_PLATFORMS:
            return f"unsupported platform: {platform}"
        return None

    @property
    def native_available(self) -> bool:
        """True when calls are forwarded to the Dart service (built Android/iOS app)."""
        return self._unavailable_reason() is None

    async def _call(
        self,
        method: str,
        arguments: Optional[dict[str, Any]] = None,
        *,
        default: Any = None,
        timeout: Optional[float] = _DEFAULT_TIMEOUT,
    ) -> Any:
        if self._unavailable_reason() is not None:
            return default
        try:
            result = await self._invoke_method(method, arguments, timeout=timeout)
        except Exception as ex:  # never let a native failure reach the caller
            message = str(ex)
            lowered = message.lower()
            if any(marker in lowered for marker in _MISSING_SERVICE_MARKERS):
                # The client has no GlossarionNative Dart service (flet run,
                # companion app): stop trying for the rest of the session.
                self._native_unavailable_reason = f"no Dart service: {message}"
                logger.info("GlossarionNative unavailable: %s", message)
            else:
                logger.warning("GlossarionNative.%s failed: %s", method, message)
            return default
        return default if result is None else result

    # ------------------------------------------------------------ platform info

    async def get_platform_info(self) -> dict[str, Any]:
        """Platform facts: ``platform``, ``sdk_int``/``system_version``,
        ``notifications_enabled``, ``ignoring_battery_optimizations``,
        ``fgs_types``, ``continued_processing``, ``save_to_downloads`` ...

        Always contains ``native`` (bool) and ``platform``.
        """
        base: dict[str, Any] = {
            "platform": self._platform(),
            "native": False,
            "extension_version": EXTENSION_VERSION,
            "continued_processing": False,
            "notifications_enabled": False,
            "save_to_downloads": False,
        }
        reason = self._unavailable_reason()
        if reason is not None:
            base["unavailable_reason"] = reason
            return base
        result = await self._call("get_platform_info", default=None)
        if not isinstance(result, dict):
            reason = self._unavailable_reason()
            if reason:
                base["unavailable_reason"] = reason
            return base
        info = dict(base)
        info.update(result)
        info["native"] = True
        return info

    # ------------------------------------------------------------------ sharing

    async def get_initial_shared(self) -> list[SharedItem]:
        """Items received before this call (cold-start Open-with/Share and any
        not yet cleared). Does not clear them; call :meth:`clear_shared`."""
        result = await self._call("get_initial_shared", default=[])
        if not isinstance(result, list):
            return []
        return [SharedItem.from_map(item) for item in result]

    async def clear_shared(self, delete_files: bool = False) -> None:
        """Forget queued shared items. With ``delete_files`` the platform also
        deletes its ``shared`` copy directory (only after you imported the files)."""
        await self._call("clear_shared", {"delete_files": bool(delete_files)})

    # ------------------------------------------------------------ notifications

    async def init_notifications(
        self,
        channels: Optional[Sequence[Union[NotificationChannel, dict[str, Any]]]] = None,
        *,
        request_permission: bool = False,
    ) -> bool:
        """Create Android channels (defaults: jobs.progress / jobs.done / jobs.action).

        On iOS ``request_permission=True`` asks for alert/sound/badge permission;
        Android 13+ POST_NOTIFICATIONS is requested through flet-permission-handler.
        Returns whether notifications are currently enabled.
        """
        if channels is None:
            channels = DEFAULT_NOTIFICATION_CHANNELS
        payload = {
            "channels": [_as_map(c) for c in channels],
            "request_permission": bool(request_permission),
        }
        return _as_bool(await self._call("init_notifications", payload, default=False))

    async def show_notification(
        self,
        id: int,
        title: str,
        body: str,
        *,
        channel_id: str,
        payload: Optional[str] = None,
        ongoing: bool = False,
        progress: Optional[tuple[int, int]] = None,
        indeterminate: bool = False,
        actions: Optional[Sequence[Union[NotificationAction, dict[str, Any]]]] = None,
        auto_cancel: Optional[bool] = None,
        silent: bool = False,
    ) -> bool:
        """Post or replace notification ``id``. ``progress=(done, total)`` shows a
        progress bar on Android (iOS shows ``done/total`` as the subtitle).
        Taps and actions arrive through ``on_notification`` with ``payload``.
        Returns True when the platform accepted the notification."""
        if int(id) == JOB_SERVICE_NOTIFICATION_ID:
            logger.warning("Notification id %s is reserved for the job service", id)
            return False
        args: dict[str, Any] = {
            "id": int(id),
            "title": title,
            "body": body,
            "channel_id": channel_id,
            "payload": payload,
            "ongoing": bool(ongoing),
            "indeterminate": bool(indeterminate),
            "silent": bool(silent),
        }
        if progress is not None:
            done, total = progress
            args["progress"] = [int(done), int(total)]
        if actions:
            args["actions"] = [_as_map(a) for a in actions]
        if auto_cancel is not None:
            args["auto_cancel"] = bool(auto_cancel)
        return _as_bool(await self._call("show_notification", args, default=False))

    async def cancel_notification(self, id: int) -> None:
        await self._call("cancel_notification", {"id": int(id)})

    async def get_launch_notification(self) -> Optional[NotificationEvent]:
        """The notification tap that cold-started the app, if any."""
        result = await self._call("get_launch_notification", default=None)
        if not isinstance(result, dict):
            return None
        raw_id = result.get("notification_id")
        try:
            notification_id = int(raw_id) if raw_id is not None else None
        except (TypeError, ValueError):
            notification_id = None
        return NotificationEvent(
            name="notification",
            control=self,
            notification_id=notification_id,
            action_id=result.get("action_id"),
            payload=result.get("payload"),
            launched_app=True,
        )

    # ------------------------------------------- Android foreground service

    async def start_job_service(
        self,
        title: str,
        text: str,
        *,
        buttons: Optional[Sequence[Union[NotificationButton, dict[str, Any]]]] = None,
        wake_lock: bool = True,
        wifi_lock: bool = True,
        channel_id: str = CHANNEL_JOBS_PROGRESS,
        channel_name: str = "Job progress",
    ) -> bool:
        """Start the Android dataSync foreground service (keeps the process and
        the Python job thread alive with the screen off). Button presses arrive
        as ``on_foreground`` events of type ``button``. Android only; False elsewhere."""
        args = {
            "title": title,
            "text": text,
            "buttons": [_as_map(b) for b in (buttons or ())],
            "wake_lock": bool(wake_lock),
            "wifi_lock": bool(wifi_lock),
            "channel_id": channel_id,
            "channel_name": channel_name,
            "service_id": JOB_SERVICE_NOTIFICATION_ID,
        }
        return _as_bool(await self._call("start_job_service", args, default=False, timeout=30.0))

    async def update_job_service(
        self, title: Optional[str] = None, text: Optional[str] = None
    ) -> None:
        """Update the service notification. Throttle calls (about 1/s) on the caller side."""
        await self._call("update_job_service", {"title": title, "text": text})

    async def stop_job_service(self) -> None:
        await self._call("stop_job_service", timeout=30.0)

    async def is_job_service_running(self) -> bool:
        return _as_bool(await self._call("is_job_service_running", default=False))

    # ------------------------------------------------ iOS background execution

    async def begin_background_task(
        self,
        name: str,
        *,
        expiration_title: Optional[str] = None,
        expiration_body: Optional[str] = None,
        expiration_payload: Optional[str] = None,
    ) -> int:
        """iOS ``beginBackgroundTask``. Returns the task id, or -1 when unsupported.

        On expiry the native side posts the optional local notification
        (``expiration_*``), fires ``on_background_task(type="expiring")`` and
        ends the task itself (iOS kills apps that do not)."""
        args = {
            "name": name,
            "expiration_title": expiration_title,
            "expiration_body": expiration_body,
            "expiration_payload": expiration_payload,
        }
        result = await self._call("begin_background_task", args, default=-1)
        try:
            return int(result)
        except (TypeError, ValueError):
            return -1

    async def end_background_task(self, task_id: int) -> None:
        if task_id is None or int(task_id) < 0:
            return
        await self._call("end_background_task", {"task_id": int(task_id)})

    async def background_time_remaining(self) -> Optional[float]:
        """Seconds left while backgrounded (iOS); None in the foreground or elsewhere."""
        result = await self._call("background_time_remaining", default=None)
        if result is None:
            return None
        try:
            return float(result)
        except (TypeError, ValueError):
            return None

    async def start_continued_processing(
        self,
        identifier: Optional[str] = None,
        title: str = "Glossarion",
        subtitle: str = "",
        *,
        strategy: str = "fail",
        expiration_title: Optional[str] = None,
        expiration_body: Optional[str] = None,
        expiration_payload: Optional[str] = None,
    ) -> bool:
        """Submit an iOS 26 BGContinuedProcessingTaskRequest. Must be called in
        direct response to a user action (the Run tap).

        ``identifier`` must start with ``<bundle id>.job.`` and be unique per
        job (iOS kills apps that register an identifier twice); pass None to let
        the native side generate ``com.glossarion.app.job.<uuid>``. ``strategy``
        is ``fail`` (only run if it can start now) or ``queue``. Returns False
        on iOS < 26, on other platforms, or when submission fails."""
        args = {
            "identifier": identifier,
            "title": title,
            "subtitle": subtitle,
            "strategy": strategy,
            "expiration_title": expiration_title,
            "expiration_body": expiration_body,
            "expiration_payload": expiration_payload,
        }
        return _as_bool(await self._call("start_continued_processing", args, default=False))

    async def update_continued_processing(
        self, completed: int, total: int, subtitle: Optional[str] = None
    ) -> None:
        """Report progress to the iOS Live Activity (keep it moving: the system
        expires tasks that show no progress)."""
        await self._call(
            "update_continued_processing",
            {"completed": int(completed), "total": int(total), "subtitle": subtitle},
        )

    async def finish_continued_processing(self, success: bool) -> None:
        await self._call("finish_continued_processing", {"success": bool(success)})

    # ------------------------------------------------------------- Downloads

    async def save_to_downloads(
        self,
        path: str,
        display_name: str,
        mime_type: str,
        subdir: str = "Glossarion",
    ) -> Optional[str]:
        """Android: copy ``path`` into public Downloads/<subdir> (MediaStore on
        API 29+, legacy file write on 26-28 when WRITE_EXTERNAL_STORAGE is
        granted). Returns the content URI / file path, or None."""
        args = {
            "path": str(path),
            "display_name": display_name,
            "mime_type": mime_type,
            "subdir": subdir,
        }
        result = await self._call("save_to_downloads", args, default=None, timeout=_LONG_TIMEOUT)
        return str(result) if result else None
