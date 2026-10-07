"""Bridge to the in-repo Flet extension ``flet_glossarion_native``.

``create_native(page, handlers)`` returns the real ``GlossarionNative`` service
on Android/iOS when the extension imports, otherwise ``NativeStub`` (desktop
dev, host tests, or a build where the extension failed to load). Both expose
the same async API with safe defaults, so callers never branch on the platform.
The event handlers are passed to the constructor, as the extension requires,
so they are registered before the Dart service starts.

``NativeBridge`` wraps either one: it normalises results and event payloads to
plain dicts/lists, applies an outer timeout to every native call (a Dart side
that never answers must not hang the UI), and fans events out to Python
callbacks. Its ``service_holds`` (``ServiceHolds``) records who needs the
Android foreground service (jobs, a sign-in waiting in the browser), so one of
them finishing never stops the service under the other.

The extension is imported lazily: importing this module never imports
``flet_glossarion_native``.
"""

from __future__ import annotations

import asyncio
import dataclasses
import enum
import importlib
import logging
import os
from typing import Any, Awaitable, Callable, Optional

log = logging.getLogger("glossarion.native")

__all__ = ["NativeStub", "NativeBridge", "ServiceHolds", "create_native", "load_extension", "service_holds", "to_plain"]

EVENTS = ("share", "foreground", "background_task", "notification")
# Outer guard only; the extension applies its own per-call timeouts (20-30 s).
DEFAULT_TIMEOUT = 45.0


def to_plain(value: Any, _depth: int = 0) -> Any:
    """JSON-friendly copy of extension results/events (dataclasses, enums, lists)."""
    if _depth > 6:
        return repr(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, dict):
        return {str(k): to_plain(v, _depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_plain(v, _depth + 1) for v in value]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        out = {}
        for f in dataclasses.fields(value):
            if f.name in ("control",) or f.name.startswith("_"):
                continue
            out[f.name] = to_plain(getattr(value, f.name, None), _depth + 1)
        return out
    return str(value)


class NativeStub:
    """Desktop/host stand-in for ``GlossarionNative``: every call is a safe no-op.

    Return values mirror the extension's own defaults on unsupported platforms.
    """

    is_stub = True

    def __init__(self, reason: str = "desktop", **handlers: Any) -> None:
        self.reason = reason
        self.on_share = handlers.get("on_share")
        self.on_foreground = handlers.get("on_foreground")
        self.on_background_task = handlers.get("on_background_task")
        self.on_notification = handlers.get("on_notification")

    async def get_platform_info(self) -> dict:
        return {"platform": "desktop", "native": False, "stub": True, "unavailable_reason": self.reason}

    async def get_initial_shared(self) -> list:
        return []

    async def clear_shared(self, delete_files: bool = False) -> None:
        return None

    async def init_notifications(self, *args: Any, **kwargs: Any) -> bool:
        return False

    async def show_notification(self, *args: Any, **kwargs: Any) -> bool:
        return False

    async def cancel_notification(self, *args: Any, **kwargs: Any) -> None:
        return None

    async def get_launch_notification(self) -> None:
        return None

    async def start_job_service(self, *args: Any, **kwargs: Any) -> bool:
        return False

    async def update_job_service(self, *args: Any, **kwargs: Any) -> None:
        return None

    async def stop_job_service(self) -> None:
        return None

    async def is_job_service_running(self) -> bool:
        return False

    async def begin_background_task(self, *args: Any, **kwargs: Any) -> int:
        return -1

    async def end_background_task(self, *args: Any, **kwargs: Any) -> None:
        return None

    async def background_time_remaining(self) -> Optional[float]:
        return None

    async def start_continued_processing(self, *args: Any, **kwargs: Any) -> bool:
        return False

    async def update_continued_processing(self, *args: Any, **kwargs: Any) -> None:
        return None

    async def finish_continued_processing(self, *args: Any, **kwargs: Any) -> None:
        return None

    async def save_to_downloads(self, *args: Any, **kwargs: Any) -> Optional[str]:
        return None


def load_extension():
    """The ``flet_glossarion_native`` module, or ``None`` when it cannot be imported."""
    try:
        return importlib.import_module("flet_glossarion_native")
    except Exception as exc:  # ImportError, or a broken extension package
        log.info("flet_glossarion_native unavailable: %s", exc)
        return None


def _is_mobile_page(page: Any) -> bool:
    platform = getattr(page, "platform", None)
    value = getattr(platform, "value", platform)
    return str(value or "").lower() in ("android", "ios") and not getattr(page, "web", False)


def create_native(page: Any, handlers: Optional[dict[str, Any]] = None) -> Any:
    """Real ``GlossarionNative`` on Android/iOS, ``NativeStub`` elsewhere.

    Must be called with the page as the current Flet context (inside ``main``
    or an event handler) so the service auto-registers. Keep a strong reference
    to the result: Flet unregisters services nobody references. The desktop
    Flutter client has no GlossarionNative service, so desktop always gets the
    stub (the extension would only return defaults there anyway).
    """
    handlers = dict(handlers or {})
    if os.environ.get("GLOSSARION_FORCE_NATIVE_STUB") == "1":
        return NativeStub("forced by GLOSSARION_FORCE_NATIVE_STUB", **handlers)
    if not _is_mobile_page(page):
        return NativeStub("not a mobile platform", **handlers)
    module = load_extension()
    cls = getattr(module, "GlossarionNative", None) if module is not None else None
    if cls is None:
        return NativeStub("flet_glossarion_native not importable", **handlers)
    try:
        return cls(**handlers)
    except Exception as exc:
        log.exception("GlossarionNative() failed")
        return NativeStub(f"GlossarionNative() failed: {exc}", **handlers)


def _make_type(type_name: str, **values: Any) -> Any:
    """Build an extension dataclass (e.g. NotificationButton) or fall back to a dict."""
    module = load_extension()
    cls = getattr(module, type_name, None) if module is not None else None
    if cls is None:
        return dict(values)
    if dataclasses.is_dataclass(cls):
        names = {f.name for f in dataclasses.fields(cls)}
        values = {k: v for k, v in values.items() if k in names}
    try:
        return cls(**values)
    except Exception:
        return dict(values)


class ServiceHolds:
    """Who needs the Android foreground service right now, with the notification each one shows.

    The service is one per app: the job runner (``BackgroundExecution``) and a sign-in
    waiting in the browser (``OAuthBridge``) share it. Each ``hold``s it while it needs it
    and ``release``s it when done; whoever releases last stops the service, anyone else
    gets the remaining holder's notification back. Loop-thread only (no locking).
    """

    def __init__(self) -> None:
        self._holds: dict = {}

    def hold(self, name: str, title: str, text: str) -> None:
        """Add (or refresh the notification of) holder *name*."""
        self._holds.pop(name, None)
        self._holds[name] = (str(title), str(text))

    def release(self, name: str) -> Optional[tuple]:
        """Drop *name*; the ``(title, text)`` of the most recent remaining holder, or None when none is left."""
        self._holds.pop(name, None)
        if not self._holds:
            return None
        return list(self._holds.values())[-1]

    def names(self) -> list:
        return list(self._holds)

    def __contains__(self, name: object) -> bool:
        return name in self._holds

    def __len__(self) -> int:
        return len(self._holds)


def service_holds(native: Any) -> Optional[ServiceHolds]:
    """The ``ServiceHolds`` of a native bridge, or None (stubs and test fakes without one)."""
    holds = getattr(native, "service_holds", None)
    return holds if isinstance(holds, ServiceHolds) else None


class NativeBridge:
    """Timeout-guarded, normalising wrapper around ``GlossarionNative``/``NativeStub``."""

    def __init__(self, page: Any = None, *, native: Any = None, timeout: float = DEFAULT_TIMEOUT) -> None:
        self.timeout = timeout
        self.service_holds = ServiceHolds()
        self._listeners: dict[str, list[Callable[[dict], Any]]] = {name: [] for name in EVENTS}
        handlers = {f"on_{name}": self._make_handler(name) for name in EVENTS}
        if native is None:
            native = create_native(page, handlers)
        else:
            for key, handler in handlers.items():
                setattr(native, key, handler)
        self.native = native  # strong reference keeps the Flet service registered

    @property
    def is_stub(self) -> bool:
        return bool(getattr(self.native, "is_stub", False))

    # ---- events -----------------------------------------------------------

    def add_listener(self, event: str, callback: Callable[[dict], Any]) -> None:
        self._listeners[event].append(callback)

    def _make_handler(self, name: str) -> Callable[[Any], Awaitable[None]]:
        async def handler(e: Any = None) -> None:
            payload = to_plain(e) if e is not None else {}
            if not isinstance(payload, dict):
                payload = {"data": payload}
            payload.setdefault("event", name)
            for callback in list(self._listeners[name]):
                try:
                    result = callback(payload)
                    if asyncio.iscoroutine(result):
                        await result
                except Exception:
                    log.exception("native %s listener failed", name)

        return handler

    # ---- calls ------------------------------------------------------------

    async def call(self, method: str, *args: Any, default: Any = None, **kwargs: Any) -> Any:
        func = getattr(self.native, method, None)
        if func is None:
            return default
        try:
            result = await asyncio.wait_for(func(*args, **kwargs), self.timeout)
        except asyncio.TimeoutError:
            log.warning("native %s timed out after %ss", method, self.timeout)
            return default
        return to_plain(result)

    async def platform_info(self) -> dict:
        info = await self.call("get_platform_info", default={})
        return info if isinstance(info, dict) else {"value": info}

    async def initial_shared(self) -> list:
        items = await self.call("get_initial_shared", default=[])
        return items if isinstance(items, list) else [items]

    async def clear_shared(self, delete_files: bool = False) -> None:
        await self.call("clear_shared", delete_files)

    async def init_notifications(self) -> bool:
        # channels=None -> the extension's DEFAULT_NOTIFICATION_CHANNELS
        # (jobs.progress / jobs.done / jobs.action); iOS asks for permission here.
        return bool(await self.call("init_notifications", None, request_permission=True, default=False))

    async def show_notification(
        self, notification_id: int, title: str, body: str, *, channel_id: str = "jobs.action", payload: Optional[str] = None
    ) -> bool:
        return bool(
            await self.call(
                "show_notification", notification_id, title, body, channel_id=channel_id, payload=payload, default=False
            )
        )

    async def cancel_notification(self, notification_id: int) -> None:
        await self.call("cancel_notification", notification_id)

    async def launch_notification(self) -> Any:
        return await self.call("get_launch_notification")

    async def start_job_service(self, title: str, text: str) -> bool:
        buttons = [
            _make_type("NotificationButton", id="stop", text="Stop"),
            _make_type("NotificationButton", id="open", text="Open"),
        ]
        return bool(await self.call("start_job_service", title, text, buttons=buttons, default=False))

    async def update_job_service(self, text: Optional[str] = None, title: Optional[str] = None) -> None:
        await self.call("update_job_service", title=title, text=text)

    async def stop_job_service(self) -> None:
        await self.call("stop_job_service")

    async def is_job_service_running(self) -> bool:
        return bool(await self.call("is_job_service_running", default=False))

    async def begin_background_task(self, name: str) -> int:
        result = await self.call("begin_background_task", name, default=-1)
        try:
            return int(result)
        except (TypeError, ValueError):
            return -1

    async def end_background_task(self, task_id: int) -> None:
        await self.call("end_background_task", task_id)

    async def background_time_remaining(self) -> Optional[float]:
        return await self.call("background_time_remaining")

    async def start_continued_processing(self, identifier: str, title: str, subtitle: str) -> bool:
        return bool(await self.call("start_continued_processing", identifier, title, subtitle, default=False))

    async def update_continued_processing(self, completed: int, total: int, subtitle: Optional[str] = None) -> None:
        await self.call("update_continued_processing", completed, total, subtitle)

    async def finish_continued_processing(self, success: bool) -> None:
        await self.call("finish_continued_processing", success)
