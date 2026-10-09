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
Android foreground service (jobs, a sign-in waiting in the browser, a cloud
save or a share-link upload), so one of them finishing never stops the service
under the other; ``ServiceLease`` is one such holder's join / start / hand back
/ stop logic (sign-in, U10 cloud sync and share links use it).

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

__all__ = ["DOCUMENT_METHODS", "METHOD_TIMEOUTS", "NativeStub", "NativeBridge", "ServiceHolds", "ServiceLease",
           "create_native", "load_extension", "service_holds", "to_plain"]

# ``document``: U10 document-destination events (write progress, picker answers that outlived their call).
EVENTS = ("share", "foreground", "background_task", "notification", "document")
# Outer guard only; the extension applies its own per-call timeouts (20-30 s).
DEFAULT_TIMEOUT = 45.0
#: Calls the extension bounds with its own longer timeout (``save_to_downloads`` copies a whole file: 900 s,
#: ``flet_glossarion_native._LONG_TIMEOUT``): the outer guard waits longer, or Python would report a failure
#: while the native copy still runs (the copy-once mirror, Save to Downloads, the transfer.it handoff).
METHOD_TIMEOUTS = {"save_to_downloads": 960.0, "save_to_downloads_entry": 960.0}
#: Document destinations (U10, ``flet_glossarion_native`` documents API): off-device every one answers
#: ``{"ok": False, "error": "unavailable"}``. The cloud sync calls them on the extension directly
#: (``cloud_sync.NativeDocs``): the extension bounds each one itself (pickers 1 h, writes by size).
DOCUMENT_METHODS = ("pick_folder", "pick_save_location", "pick_document", "list_children", "create_file",
                    "create_folder", "write_file", "rename_document", "stat", "delete", "query_root")


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
        self.on_document = handlers.get("on_document")

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

    # ---- document destinations (U10): the extension's answers off-device -------------------------------

    def _no_documents(self) -> dict:
        return {"ok": False, "error": "unavailable", "message": self.reason, "scope": None, "retryable": False}

    async def pick_folder(self, *args: Any, **kwargs: Any) -> dict:
        return self._no_documents()

    async def pick_save_location(self, *args: Any, **kwargs: Any) -> dict:
        return self._no_documents()

    async def pick_document(self, *args: Any, **kwargs: Any) -> dict:
        return self._no_documents()

    async def list_children(self, *args: Any, **kwargs: Any) -> dict:
        return self._no_documents()

    async def create_file(self, *args: Any, **kwargs: Any) -> dict:
        return self._no_documents()

    async def create_folder(self, *args: Any, **kwargs: Any) -> dict:
        return self._no_documents()

    async def write_file(self, *args: Any, **kwargs: Any) -> dict:
        return self._no_documents()

    async def rename_document(self, *args: Any, **kwargs: Any) -> dict:
        return self._no_documents()

    async def stat(self, *args: Any, **kwargs: Any) -> dict:
        return self._no_documents()

    async def delete(self, *args: Any, **kwargs: Any) -> dict:
        return self._no_documents()

    async def query_root(self, *args: Any, **kwargs: Any) -> dict:
        return self._no_documents()

    async def release(self, *args: Any, **kwargs: Any) -> bool:
        return False

    async def list_grants(self) -> list:
        return []

    async def cancel_document_op(self, *args: Any, **kwargs: Any) -> bool:
        return False

    async def take_document_results(self) -> list:
        return []


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


class ServiceLease:
    """One holder's share of the Android foreground service (the sign-in's logic, extracted so the U10
    cloud sync and share links reuse it instead of copying it).

    ``acquire`` joins a running service through ``ServiceHolds`` (a job's) or starts it; ``join`` does
    the same synchronously for a service another holder keeps running (the cloud sync joins a finishing
    job's service inside the transition callback, before ``BackgroundExecution.job_finished`` runs);
    ``update`` remembers this holder's notification text and shows it while nobody else holds the
    service; ``release`` gives the notification back to the remaining holder, or stops the service when
    nobody is left (unless ``keep_service`` says a queued job takes it over); ``drop`` forgets the hold
    when the service is already gone. Loop-thread only. ``native`` is a ``NativeBridge`` (or a fake with
    its ``is_job_service_running`` / ``start_job_service`` / ``update_job_service`` / ``stop_job_service``).
    """

    def __init__(self, native: Any, name: str) -> None:
        self.native = native
        self.name = name
        self.title = ""
        self.text = ""
        self.started = False  # this lease started the service
        self.held = False  # this lease relies on the service (started or shared)
        self.failed = False  # the service could not be started

    def holds(self) -> Optional[ServiceHolds]:
        return service_holds(self.native)

    def join(self, title: str, text: str, *, holder: str) -> bool:
        """Synchronous: share the service *holder* holds (or this lease holds already); False otherwise."""
        holds = self.holds()
        if holds is None or (holder not in holds and self.name not in holds):
            return False
        holds.hold(self.name, title, text)
        self.title, self.text = str(title), str(text)
        self.held = True
        return True

    async def acquire(self, title: str, text: str, *, may_start: bool = True) -> bool:
        """Hold the service; False when it could not be started (``failed``), may not be started now
        (``may_start=False``: Android 12+ forbids starting one from the background) or runs without
        ``ServiceHolds`` (someone else's service that cannot be shared)."""
        self.started = self.held = self.failed = False
        self.title, self.text = str(title), str(text)
        if self.native is None:
            return False
        holds = self.holds()
        try:
            if await self.native.is_job_service_running():
                if holds is None:
                    return False  # someone else's service, and no way to share it
                holds.hold(self.name, title, text)
                self.held = True
                return True
            if not may_start:
                return False
            started = bool(await self.native.start_job_service(title, text))
        except Exception as exc:
            log.warning("foreground service for %s unavailable: %s", self.name, exc)
            self.failed = True
            return False
        if not started:
            log.warning("the foreground service for %s did not start", self.name)
            self.failed = True
            return False
        self.started = self.held = True
        if holds is not None:
            holds.hold(self.name, title, text)
        return True

    def owns_alone(self) -> bool:
        """This lease is the only holder (its notification is the one shown)."""
        if not self.held:
            return False
        holds = self.holds()
        if holds is None:
            return self.started
        return holds.names() == [self.name]

    def remember(self, text: str, title: Optional[str] = None) -> None:
        """This holder's notification text, shown again when the other holders release the service."""
        if not self.held:
            return
        self.title = self.title if title is None else str(title)
        self.text = str(text)
        holds = self.holds()
        if holds is not None and self.name in holds:
            holds.hold(self.name, self.title, self.text)

    async def update(self, text: str, title: Optional[str] = None) -> bool:
        """New notification text; shown only while this lease holds the service alone (a job's progress
        keeps the notification while it runs). True when it was sent to the notification."""
        if not self.held:
            return False
        self.remember(text, title)
        if not self.owns_alone():
            return False
        try:
            await self.native.update_job_service(text=self.text, title=self.title)
        except Exception:
            return False
        return True

    async def release(self, *, keep_service: Optional[Callable[[], bool]] = None) -> None:
        """Give the service up: the remaining holder gets its notification back, else the service stops
        (not when ``keep_service()`` says a queued job takes it over)."""
        held, started = self.held, self.started
        self.held = self.started = False
        if not held or self.native is None:
            return
        holds = self.holds()
        if holds is not None:
            remaining = holds.release(self.name)
            if remaining is not None:
                # Another holder still needs the service: give it back its notification.
                title, text = remaining
                try:
                    await self.native.update_job_service(text=text, title=title)
                except Exception:
                    pass
                return
        elif not started:
            return
        if keep_service is not None:
            try:
                if keep_service():
                    return
            except Exception:
                log.debug("keep_service check failed", exc_info=True)
        try:
            await self.native.stop_job_service()
        except Exception:
            pass

    def drop(self) -> None:
        """The service is gone (``destroyed`` / ``timeout``): forget the hold without stopping anything."""
        self.held = self.started = False
        holds = self.holds()
        if holds is not None:
            holds.release(self.name)


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
        limit = max(self.timeout, METHOD_TIMEOUTS.get(method, 0.0))
        try:
            result = await asyncio.wait_for(func(*args, **kwargs), limit)
        except asyncio.TimeoutError:
            log.warning("native %s timed out after %ss", method, limit)
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
