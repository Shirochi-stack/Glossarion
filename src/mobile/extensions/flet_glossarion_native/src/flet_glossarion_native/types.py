"""Data types exchanged with the GlossarionNative Flutter service.

Event payload keys sent by the Dart side (``lib/src/native_service.dart``) must
match the dataclass field names below: Flet builds event objects with
``flet.utils.from_dict``, which reads only declared fields and ignores unknown
keys. Event payloads never use the keys ``name``, ``data`` or ``control``,
because those are fields of ``ft.Event`` itself.

Annotations are kept as real objects (no ``from __future__ import
annotations``) and ``Optional[...]`` is used instead of ``X | None``:
``from_dict`` only converts nested dataclasses inside ``typing.Union``.
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional

import flet as ft

__all__ = [
    "BackgroundTaskEvent",
    "BackgroundTaskEventType",
    "CHANNEL_JOBS_ACTION",
    "CHANNEL_JOBS_DONE",
    "CHANNEL_JOBS_PROGRESS",
    "DEFAULT_NOTIFICATION_CHANNELS",
    "DocumentEvent",
    "DocumentEventType",
    "ForegroundEvent",
    "ForegroundEventType",
    "JOB_SERVICE_NOTIFICATION_ID",
    "NotificationAction",
    "NotificationButton",
    "NotificationChannel",
    "NotificationEvent",
    "NotificationImportance",
    "ShareEvent",
    "SharedItem",
    "SharedItemKind",
]

#: Notification id used by the Android foreground service (flutter_foreground_task
#: ``serviceId``). Do not reuse it for show_notification().
JOB_SERVICE_NOTIFICATION_ID = 41100

#: Channel ids from the UI spec (plan section 5, "Notification channels").
CHANNEL_JOBS_PROGRESS = "jobs.progress"
CHANNEL_JOBS_DONE = "jobs.done"
CHANNEL_JOBS_ACTION = "jobs.action"


class SharedItemKind(str, Enum):
    """``SharedItem.kind`` values."""

    FILE = "file"
    TEXT = "text"
    URL = "url"


class ForegroundEventType(str, Enum):
    """``ForegroundEvent.type`` values (Android foreground service)."""

    STARTED = "started"
    """The service's task handler started (``TaskHandler.onStart``)."""
    BUTTON = "button"
    """A notification button was pressed; see ``ForegroundEvent.button_id``."""
    TAP = "tap"
    """The service notification itself was tapped."""
    DISMISSED = "dismissed"
    """The service notification was dismissed (Android 14+)."""
    TIMEOUT = "timeout"
    """Android 15+ ended the dataSync service after its 6h/24h budget."""
    DESTROYED = "destroyed"
    """The service stopped (by stop_job_service() or by the system)."""


class BackgroundTaskEventType(str, Enum):
    """``BackgroundTaskEvent.type`` values (iOS)."""

    EXPIRING = "expiring"
    """A beginBackgroundTask grant is about to expire; the task was ended."""
    CONTINUED_STARTED = "continued_started"
    """iOS 26 BGContinuedProcessingTask launch handler ran."""
    CONTINUED_EXPIRED = "continued_expired"
    """The system expired the continued-processing task or the user cancelled it."""
    CONTINUED_FAILED = "continued_failed"
    """Submitting the continued-processing request failed."""


class DocumentEventType(str, Enum):
    """``DocumentEvent.type`` values (document destinations)."""

    PROGRESS = "progress"
    """A write_file() copy advanced: ``written`` of ``total`` bytes (at most 4 per second)."""
    PICK_RESULT = "pick_result"
    """A picker answered after the call that opened it was gone (the activity or the whole process
    was recreated while the system picker was open) or Glossarion restarted while a picker was open
    (``status == "cancelled"``). ``result`` is the answer the call would have returned; the same
    items are also kept for ``take_document_results()``: de-duplicate by ``op_id``."""


class NotificationImportance(str, Enum):
    """Android channel importance (ignored on iOS)."""

    MIN = "min"
    LOW = "low"
    DEFAULT = "default"
    HIGH = "high"
    MAX = "max"


def _known_fields(cls: type, data: dict[str, Any]) -> dict[str, Any]:
    names = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    return {k: v for k, v in data.items() if k in names}


@dataclass
class SharedItem:
    """One item received through Android "Open with"/"Share" or iOS "Open in".

    Files are already copied into app-private storage (Android
    ``cacheDir/shared/``, iOS ``tmp/shared/``); ``path`` is that copy.
    """

    id: str = ""
    """Unique id of this item (use it to de-duplicate events and initial items)."""
    kind: str = SharedItemKind.FILE.value
    """``file``, ``text`` or ``url`` (see :class:`SharedItemKind`)."""
    path: Optional[str] = None
    """Local path of the copied file (``kind == "file"``), else None."""
    name: Optional[str] = None
    """Display file name."""
    mime_type: Optional[str] = None
    size: Optional[int] = None
    text: Optional[str] = None
    """Shared text (``kind == "text"``) or URL (``kind == "url"``)."""
    subject: Optional[str] = None
    uri: Optional[str] = None
    """Original content:// / file:// URI (diagnostics only; never route on it)."""
    source: Optional[str] = None
    """``view``, ``send``, ``send_multiple`` (Android trampoline), ``open_url`` /
    ``launch`` (iOS Open in, cold-start deep link) or ``share`` (receive_sharing_intent)."""
    error: Optional[str] = None
    """Set when the platform could not copy the item; ``path`` is then None."""

    @classmethod
    def from_map(cls, data: Any) -> "SharedItem":
        if isinstance(data, SharedItem):
            return data
        if not isinstance(data, dict):
            return cls(kind=SharedItemKind.TEXT.value, text=str(data))
        values = _known_fields(cls, data)
        for key in ("id", "kind"):
            if values.get(key) is None:
                values.pop(key, None)
        size = values.get("size")
        if size is not None:
            try:
                values["size"] = int(size)
            except (TypeError, ValueError):
                values["size"] = None
        return cls(**values)

    @property
    def is_file(self) -> bool:
        return self.kind == SharedItemKind.FILE.value and bool(self.path)


@dataclass
class NotificationChannel:
    """Android notification channel (created once; settings are then owned by the user)."""

    id: str
    name: str
    description: Optional[str] = None
    importance: str = NotificationImportance.DEFAULT.value
    show_badge: bool = True
    vibration: bool = False
    sound: bool = True

    def to_map(self) -> dict[str, Any]:
        data = asdict(self)
        if isinstance(self.importance, Enum):
            data["importance"] = self.importance.value
        return data


#: Channels from the UI spec. ``jobs.progress`` is also the foreground-service channel.
DEFAULT_NOTIFICATION_CHANNELS: tuple[NotificationChannel, ...] = (
    NotificationChannel(
        id=CHANNEL_JOBS_PROGRESS,
        name="Job progress",
        description="Ongoing translation and tool jobs",
        importance=NotificationImportance.LOW.value,
        show_badge=False,
        sound=False,
    ),
    NotificationChannel(
        id=CHANNEL_JOBS_DONE,
        name="Finished jobs",
        description="A job finished or failed",
        importance=NotificationImportance.DEFAULT.value,
    ),
    NotificationChannel(
        id=CHANNEL_JOBS_ACTION,
        name="Action needed",
        description="Glossary review, sign-in required, paused by a system limit",
        importance=NotificationImportance.HIGH.value,
    ),
)


@dataclass
class NotificationButton:
    """Button on the Android foreground-service notification (e.g. ``stop``/``open``)."""

    id: str
    text: str

    def to_map(self) -> dict[str, Any]:
        return {"id": self.id, "text": self.text}


@dataclass
class NotificationAction:
    """Action button on a regular notification posted with show_notification()."""

    id: str
    title: str

    def to_map(self) -> dict[str, Any]:
        return {"id": self.id, "title": self.title}


@dataclass
class ShareEvent(ft.Event["GlossarionNative"]):
    """Fired when files or text arrive through Open-with / Share / Open in."""

    items: list[SharedItem] = field(default_factory=list)


@dataclass
class ForegroundEvent(ft.Event["GlossarionNative"]):
    """Android foreground-service event (see :class:`ForegroundEventType`)."""

    type: str = ""
    button_id: Optional[str] = None
    is_timeout: bool = False


@dataclass
class BackgroundTaskEvent(ft.Event["GlossarionNative"]):
    """iOS background-execution event (see :class:`BackgroundTaskEventType`)."""

    type: str = ""
    task_id: Optional[int] = None
    task_name: Optional[str] = None
    identifier: Optional[str] = None
    reason: Optional[str] = None


@dataclass
class DocumentEvent(ft.Event["GlossarionNative"]):
    """Document-destination event (see :class:`DocumentEventType`)."""

    type: str = ""
    op_id: Optional[str] = None
    """The ``op_id`` of the write_file() / pick call it belongs to."""
    written: Optional[int] = None
    total: Optional[int] = None
    kind: Optional[str] = None
    """Picker kind for ``pick_result``: ``folder``, ``save_location`` or ``document``."""
    status: Optional[str] = None
    """``ok``, ``cancelled`` or ``error`` for ``pick_result``."""
    result: Optional[dict[str, Any]] = None
    """The full pick answer (``{"ok": ..., "target"/"document": ...}``) for ``pick_result``."""


@dataclass
class NotificationEvent(ft.Event["GlossarionNative"]):
    """A notification posted by show_notification() was tapped or an action pressed."""

    notification_id: Optional[int] = None
    action_id: Optional[str] = None
    """None for a tap on the notification body."""
    payload: Optional[str] = None
    launched_app: bool = False
    """True when the tap cold-started the app."""
