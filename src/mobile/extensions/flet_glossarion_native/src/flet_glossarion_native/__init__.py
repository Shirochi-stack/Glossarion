from flet_glossarion_native.native import EXTENSION_VERSION, GlossarionNative
from flet_glossarion_native.types import (
    CHANNEL_JOBS_ACTION,
    CHANNEL_JOBS_DONE,
    CHANNEL_JOBS_PROGRESS,
    DEFAULT_NOTIFICATION_CHANNELS,
    JOB_SERVICE_NOTIFICATION_ID,
    BackgroundTaskEvent,
    BackgroundTaskEventType,
    ForegroundEvent,
    ForegroundEventType,
    NotificationAction,
    NotificationButton,
    NotificationChannel,
    NotificationEvent,
    NotificationImportance,
    ShareEvent,
    SharedItem,
    SharedItemKind,
)

__version__ = EXTENSION_VERSION

__all__ = [
    "BackgroundTaskEvent",
    "BackgroundTaskEventType",
    "CHANNEL_JOBS_ACTION",
    "CHANNEL_JOBS_DONE",
    "CHANNEL_JOBS_PROGRESS",
    "DEFAULT_NOTIFICATION_CHANNELS",
    "EXTENSION_VERSION",
    "ForegroundEvent",
    "ForegroundEventType",
    "GlossarionNative",
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
