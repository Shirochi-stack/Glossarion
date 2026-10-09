"""GlossarionNative: Flet service for the Android/iOS pieces Flet does not ship.

The Dart side lives in ``src/flutter/flet_glossarion_native`` and only exists in
apps built with ``flet build`` (the extension is compiled into the Flutter
shell). Under ``flet run`` on Windows/macOS/Linux, on the web, or with the Flet
companion app, there is no Dart service: every method then returns a safe
default without touching the client, and never raises.
"""

import asyncio
import logging
import os
import uuid
from typing import Any, Mapping, Optional, Sequence, Union

import flet as ft

from flet_glossarion_native.documents import (
    DEFAULT_MODE_CHAIN,
    PICKER_TIMEOUT,
    QUERY_TIMEOUT,
    DocumentError,
    error_result,
    normalize_modes,
    normalize_result,
    write_timeout,
)
from flet_glossarion_native.types import (
    CHANNEL_JOBS_PROGRESS,
    DEFAULT_NOTIFICATION_CHANNELS,
    JOB_SERVICE_NOTIFICATION_ID,
    BackgroundTaskEvent,
    DocumentEvent,
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


def _new_op_id() -> str:
    return uuid.uuid4().hex


def _ref_arg(value: Any) -> Optional[dict[str, Any]]:
    """A document reference as a plain map for the channel.

    Refs are the dicts the native side returned (see ``documents.REF_KEYS``). A bare string is
    accepted for convenience: an Android tree URI (``.../tree/<id>`` without ``/document/``) is a
    folder, any other string a document URI.
    """
    if value is None:
        return None
    if isinstance(value, Mapping):
        return {str(k): v for k, v in value.items()}
    if isinstance(value, str) and value:
        if "/tree/" in value and "/document/" not in value:
            return {"kind": "folder", "uri": value}
        return {"kind": "file", "document": value}
    return None


async def _file_size(path: Optional[str]) -> int:
    if not path:
        return 0
    try:
        return int(await asyncio.to_thread(os.path.getsize, path))
    except (OSError, TypeError, ValueError):
        return 0


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

    on_document: Optional[ft.EventHandler[DocumentEvent]] = None
    """Document destinations: write progress and picker answers that arrived late."""

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
            "documents": False,
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
        replace_uri: Optional[str] = None,
    ) -> Optional[str]:
        """Android: copy ``path`` into public Downloads/<subdir> (MediaStore on
        API 29+, legacy file write on 26-28 when WRITE_EXTERNAL_STORAGE is
        granted). Returns the content URI / file path, or None.

        ``replace_uri``: the URI an earlier call returned. When that entry still
        belongs to Glossarion it is overwritten in place (same URI returned, its
        file name is never changed, so a backup app watching the folder sees an
        update, not a new file). Otherwise a new entry is added as without it.
        A failed in-place write raises on the native side (None here) instead of
        adding a second copy."""
        args = {
            "path": str(path),
            "display_name": display_name,
            "mime_type": mime_type,
            "subdir": subdir,
            "replace_uri": str(replace_uri) if replace_uri else None,
        }
        result = await self._call("save_to_downloads", args, default=None, timeout=_LONG_TIMEOUT)
        return str(result) if result else None

    async def save_to_downloads_entry(
        self,
        path: str,
        display_name: str,
        mime_type: str,
        subdir: str = "Glossarion",
        replace_uri: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        """``save_to_downloads`` that also says which name the entry has: ``{"uri", "name"}`` (MediaStore keeps
        names unique in a folder, so a new entry may be ``name (1).ext``; an entry overwritten in place keeps
        its first name). ``name`` is None when the platform did not say (an older build answers the URI only).
        None when nothing was saved."""
        args = {
            "path": str(path),
            "display_name": display_name,
            "mime_type": mime_type,
            "subdir": subdir,
            "replace_uri": str(replace_uri) if replace_uri else None,
            "report_name": True,
        }
        result = await self._call("save_to_downloads", args, default=None, timeout=_LONG_TIMEOUT)
        if isinstance(result, dict):
            uri = result.get("uri")
            if not uri:
                return None
            name = result.get("name")
            return {"uri": str(uri), "name": str(name) if name else None}
        return {"uri": str(result), "name": None} if result else None

    # ---------------------------------------------------- document destinations
    #
    # A folder or file the user picked once in the system picker (Android SAF,
    # iOS Files), written to later without asking again. See documents.py for the
    # reference / result shapes and the error codes. Every method returns a
    # result dict and never raises; elsewhere the answer is
    # {"ok": False, "error": "unavailable"}.

    async def _doc_call(
        self, method: str, arguments: dict[str, Any], *, timeout: Optional[float]
    ) -> dict[str, Any]:
        reason = self._unavailable_reason()
        if reason is not None:
            return error_result(DocumentError.UNAVAILABLE, reason)
        try:
            result = await self._invoke_method(method, arguments, timeout=timeout)
        except (TimeoutError, asyncio.TimeoutError):
            logger.warning("GlossarionNative.%s timed out", method)
            return error_result(DocumentError.TIMEOUT, f"{method} did not answer in time")
        except Exception as ex:
            message = str(ex)
            lowered = message.lower()
            if any(marker in lowered for marker in _MISSING_SERVICE_MARKERS):
                self._native_unavailable_reason = f"no Dart service: {message}"
                logger.info("GlossarionNative unavailable: %s", message)
                return error_result(DocumentError.UNAVAILABLE, self._native_unavailable_reason)
            # The message can carry content URIs: log the type only.
            logger.warning("GlossarionNative.%s failed: %s", method, type(ex).__name__)
            code = DocumentError.BAD_ARGS if "bad_args" in lowered else DocumentError.PROVIDER_ERROR
            return error_result(code, message)
        return normalize_result(result)

    async def pick_folder(
        self, *, initial: Any = None, op_id: Optional[str] = None
    ) -> dict[str, Any]:
        """Let the user pick a folder once; Glossarion keeps write access to it.

        Android: ``ACTION_OPEN_DOCUMENT_TREE`` + ``takePersistableUriPermission``
        (only the flags actually granted). iOS: Files folder picker + minimal
        bookmark. Answer: ``{"ok": True, "target": ref, "persisted": bool}`` or
        ``{"ok": False, "error": "cancelled" | "busy" | "unavailable" | ...}``.
        ``target["own_folder"]`` is True for Glossarion's own storage, which the
        app must refuse as a cloud destination. ``initial`` (a ref) opens the
        picker there on Android 8+."""
        args = {"op_id": op_id or _new_op_id(), "initial": _ref_arg(initial)}
        return await self._doc_call("pick_folder", args, timeout=PICKER_TIMEOUT)

    async def pick_save_location(
        self,
        name: str,
        mime_type: str,
        source_path: Optional[str] = None,
        *,
        initial: Any = None,
        mode_chain: Sequence[str] = DEFAULT_MODE_CHAIN,
        op_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Ask once where one file goes (for cloud apps that do not offer folders).

        Android: ``ACTION_CREATE_DOCUMENT`` (title ``name``) + persisted grant;
        with ``source_path`` the file is written right away through the same
        mode chain as :meth:`write_file`. iOS: the export picker moves a copy of
        ``source_path`` (an empty file without it) to the chosen place and keeps
        a bookmark to it. Answer: ``{"ok": True, "document": ref, "write":
        write_result | None}``; later updates use ``write_file(document, ...)``."""
        args = {
            "op_id": op_id or _new_op_id(),
            "name": str(name),
            "mime_type": mime_type,
            "source_path": str(source_path) if source_path else None,
            "initial": _ref_arg(initial),
            "mode_chain": list(normalize_modes(mode_chain) or DEFAULT_MODE_CHAIN),
        }
        result = await self._doc_call("pick_save_location", args, timeout=PICKER_TIMEOUT)
        if isinstance(result.get("write"), Mapping):
            result["write"] = normalize_result(result["write"])
        return result

    async def pick_document(
        self,
        mime_types: Optional[Sequence[str]] = None,
        *,
        initial: Any = None,
        op_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Pick an existing file to keep updating (re-link a cloud copy).

        Android ``ACTION_OPEN_DOCUMENT`` + persisted grant; iOS open picker +
        bookmark. Answer: ``{"ok": True, "document": ref}``."""
        args = {
            "op_id": op_id or _new_op_id(),
            "mime_types": [str(m) for m in (mime_types or ()) if m],
            "initial": _ref_arg(initial),
        }
        return await self._doc_call("pick_document", args, timeout=PICKER_TIMEOUT)

    async def list_children(
        self, folder: Any, *, names: Optional[Sequence[str]] = None
    ) -> dict[str, Any]:
        """Items directly inside ``folder``: ``{"ok": True, "children": [ref, ...],
        "complete": bool}``. ``names`` limits the answer to those display names
        (adopt-by-name before creating; cloud listings may lag, ``complete`` is
        False while the provider is still loading)."""
        args = {"folder": _ref_arg(folder), "names": list(names) if names is not None else None}
        return await self._doc_call("list_children", args, timeout=QUERY_TIMEOUT)

    async def create_file(
        self, folder: Any, name: str, mime_type: str, *, on_exists: str = "rename"
    ) -> dict[str, Any]:
        """Create an empty file in ``folder``: ``{"ok": True, "document": ref,
        "created": bool, "adopted": bool}``.

        ``on_exists``: ``rename`` (the provider / iOS picks ``name (1).ext``),
        ``adopt`` (return the existing item instead) or ``fail`` (error
        ``exists`` with the existing ``document``). A first create that throws is
        retried once after the name was looked up again, so an eventually
        consistent provider (Drive) does not end up with two copies."""
        args = {"folder": _ref_arg(folder), "name": str(name), "mime_type": mime_type, "on_exists": on_exists}
        return await self._doc_call("create_file", args, timeout=QUERY_TIMEOUT)

    async def create_folder(
        self, parent: Any, name: str, *, on_exists: str = "adopt"
    ) -> dict[str, Any]:
        """Create (or by default adopt) a sub-folder: ``{"ok": True, "folder": ref,
        "created": bool, "adopted": bool}``."""
        args = {"folder": _ref_arg(parent), "name": str(name), "on_exists": on_exists}
        return await self._doc_call("create_folder", args, timeout=QUERY_TIMEOUT)

    async def write_file(
        self,
        target_or_doc: Any,
        source_path: str,
        *,
        name: Optional[str] = None,
        mime_type: Optional[str] = None,
        on_exists: str = "rename",
        mode_chain: Sequence[str] = DEFAULT_MODE_CHAIN,
        verify: bool = True,
        op_id: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> dict[str, Any]:
        """Stream ``source_path`` (a private snapshot copy) into a document.

        ``target_or_doc`` is a file ref (overwrite it) or a folder ref (create
        ``name`` in it first, like :meth:`create_file`). Android tries the modes
        of ``mode_chain`` in order; ``w``/``rw`` (non-truncating) are used only
        when the new file is not shorter than the cloud copy, the written length
        is read back (``verified_size``) and a stale tail is reported as
        ``size_mismatch`` with ``needs_replace``. iOS stages the copy and swaps
        it in under file coordination. ``on_document`` progress events carry
        ``op_id``. Answer on success: ``{"ok": True, "document": ref, "created":
        bool, "mode": str, "written": int, "verified_size": int | None, ...}``.

        ``timeout`` defaults to 120 s + 1 s per MiB (max 2 h). When Python stops
        waiting, the native copy is cancelled through :meth:`cancel_document_op`
        and the answer is error ``timeout``."""
        modes = normalize_modes(mode_chain)
        op = op_id or _new_op_id()
        if not modes:
            return error_result(DocumentError.BAD_ARGS, "mode_chain lists no valid mode", op_id=op)
        if timeout is None:
            timeout = write_timeout(await _file_size(source_path))
        args = {
            "op_id": op,
            "ref": _ref_arg(target_or_doc),
            "source_path": str(source_path) if source_path else None,
            "name": name,
            "mime_type": mime_type,
            "on_exists": on_exists,
            "mode_chain": list(modes),
            "verify": bool(verify),
        }
        result = await self._doc_call("write_file", args, timeout=timeout)
        if result.get("error") == DocumentError.TIMEOUT.value:
            await self.cancel_document_op(op)
        result.setdefault("op_id", op)
        return result

    async def rename_document(self, document: Any, name: str) -> dict[str, Any]:
        """Rename a document inside its folder: ``{"ok": True, "document": ref}`` with the new name.

        Android ``DocumentsContract.renameDocument`` (the provider may answer with a new URI, so keep the
        returned ref); iOS a coordinated move inside the same picked folder. The cloud sync gives a copy
        it had to replace (create new + delete old) its first name back. Errors: ``exists`` (the name is
        taken), ``unavailable`` (the provider or a single exported file cannot be renamed), else like
        :meth:`stat` (``missing`` / ``permission_lost`` with their scope)."""
        args = {"document": _ref_arg(document), "name": str(name)}
        return await self._doc_call("rename_document", args, timeout=QUERY_TIMEOUT)

    async def stat(self, document: Any) -> dict[str, Any]:
        """Fresh name / size / mtime / flags: ``{"ok": True, "document": ref}``."""
        return await self._doc_call("stat", {"document": _ref_arg(document)}, timeout=QUERY_TIMEOUT)

    async def delete(self, document: Any) -> dict[str, Any]:
        """Delete a document (cloud apps usually keep it in their trash).
        ``{"ok": True}``; a document that is already gone answers error ``missing``."""
        return await self._doc_call("delete", {"document": _ref_arg(document)}, timeout=QUERY_TIMEOUT)

    async def query_root(self, target: Any) -> dict[str, Any]:
        """Is the destination still usable? ``{"ok": True, "target": ref}`` with
        fresh ``can_write`` / ``can_create`` / ``persisted`` (iOS: a refreshed
        bookmark when the old one went stale), else ``permission_lost`` /
        ``missing`` with ``scope == "target"`` or ``provider_error``."""
        return await self._doc_call("query_root", {"target": _ref_arg(target)}, timeout=QUERY_TIMEOUT)

    async def release(self, target: Any) -> bool:
        """Give up a kept permission (Android ``releasePersistableUriPermission`` of
        the folder or single file; a URI string from :meth:`list_grants` works
        too). iOS keeps nothing outside the app's data: True. Files already in the
        cloud are not touched."""
        if isinstance(target, str):
            args = {"target": target}
        else:
            args = {"target": _ref_arg(target)}
        return _as_bool(await self._call("release", args, default=False, timeout=QUERY_TIMEOUT))

    async def list_grants(self) -> list[dict[str, Any]]:
        """Android's persisted permissions (``uri``, ``read``, ``write``,
        ``persisted_time``, ``tree``) for clean-up and the per-app limit
        (``get_platform_info()["persisted_grant_limit"]``). iOS: []."""
        result = await self._call("list_grants", {}, default=[], timeout=QUERY_TIMEOUT)
        if not isinstance(result, list):
            return []
        return [dict(item) for item in result if isinstance(item, Mapping)]

    async def cancel_document_op(self, op_id: str) -> bool:
        """Stop a running write_file() (checked between 1 MiB chunks)."""
        if not op_id:
            return False
        return _as_bool(await self._call("cancel_document_op", {"op_id": str(op_id)}, default=False))

    async def take_document_results(self) -> list[dict[str, Any]]:
        """Picker answers that arrived after their call was gone (see
        ``DocumentEventType.PICK_RESULT``), oldest first; they are removed.
        Call once at start-up and on each ``pick_result`` event."""
        result = await self._call("take_document_results", default=[])
        if not isinstance(result, list):
            return []
        return [dict(item) for item in result if isinstance(item, Mapping)]
