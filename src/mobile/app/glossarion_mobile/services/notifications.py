"""Job notifications (UI_SPEC §1.9) over the GlossarionNative extension.

Channels (created by ``init_notifications`` from the extension's
``DEFAULT_NOTIFICATION_CHANNELS``):

* ``jobs.progress``: the ongoing Android foreground-service notification
  (owned by ``BackgroundExecution``: ``start/update/stop_job_service``);
* ``jobs.done``: "Done: *Book*" / "Stopped: *Book* (12/80)" / "Failed: *Book*";
  tap -> the job's route, actions **Open** and **Share** (when the job has one
  compiled output);
* ``jobs.action``: "Glossary ready: review needed", "Sign-in required for
  ChatGPT", "Paused (system limit): tap to resume".

Payloads are app routes as deep links (``glossarion://app/job/<jid>``; a chat
job's glossary question opens its chat, ``/chat/<cid>``), never paths or user text. ``parse_event`` turns a tap/action event back into
``(action, route, job_id)`` for the app. All calls go through ``NativeBridge``
(safe defaults on desktop / unsupported platforms). Pure asyncio, no Flet import.
"""

from __future__ import annotations

import logging
import zlib
from dataclasses import dataclass
from typing import Any, Mapping, Optional

__all__ = [
    "ACTION_OPEN",
    "ACTION_RESUME",
    "ACTION_SHARE",
    "CHANNEL_ACTION",
    "CHANNEL_DONE",
    "CHANNEL_PROGRESS",
    "JobNotifications",
    "NotificationTap",
    "chat_of",
    "done_text",
    "job_route",
    "question_route",
]

log = logging.getLogger("glossarion.notifications")

CHANNEL_PROGRESS = "jobs.progress"
CHANNEL_DONE = "jobs.done"
CHANNEL_ACTION = "jobs.action"

ACTION_OPEN = "open"
ACTION_SHARE = "share"
ACTION_RESUME = "resume"

DEEP_LINK_PREFIX = "glossarion://app"
_DONE_BASE = 41200  # 41100 is the foreground-service notification (JOB_SERVICE_NOTIFICATION_ID)
_ACTION_BASE = 41600
_ID_SPAN = 390


def job_route(job_id: str) -> str:
    return f"/job/{job_id}"


def chat_of(snap: Any) -> Optional[str]:
    """The chat a job belongs to (``spec.origin`` ``{"type": "chat", "cid": ...}``), else None."""
    origin = getattr(getattr(snap, "spec", None), "origin", None) or {}
    if isinstance(origin, Mapping) and origin.get("type") == "chat" and origin.get("cid") not in (None, ""):
        return str(origin.get("cid"))
    return None


def question_route(snap: Any) -> str:
    """Where a job's blocking question is answered: the owning chat (its approval card), else the job."""
    cid = chat_of(snap)
    return f"/chat/{cid}" if cid is not None else job_route(snap.id)


def _deep_link(route: str) -> str:
    return DEEP_LINK_PREFIX + route


def _notification_id(base: int, key: str) -> int:
    return base + (zlib.crc32(key.encode("utf-8")) % _ID_SPAN)


def done_text(snap: Any) -> tuple:
    """(title, body) for the ``jobs.done`` notification."""
    from glossarion_mobile.services.jobs import JobState

    progress = snap.progress
    counts = f"{progress.completed}/{progress.total}" if progress.total else ""
    if snap.state is JobState.FAILED:
        return f"Failed: {snap.title}", (snap.error or "Open the job for details")[:200]
    if snap.state is JobState.CANCELLED:
        return f"Stopped: {snap.title}" + (f" ({counts})" if counts else ""), "Tap to resume"
    body = f"{counts} chapters" if counts else "Finished"
    return f"Done: {snap.title}", body


@dataclass(frozen=True)
class NotificationTap:
    action: str  # "open" (body tap or Open), "share", "resume"
    route: Optional[str]  # whitelisted app route ("/job/<jid>")
    job_id: Optional[str]


class JobNotifications:
    def __init__(self, native: Any, *, platform: str = "desktop") -> None:
        self.native = native
        self.platform = platform
        self._initialised: Optional[bool] = None
        self.posted: list[tuple] = []  # (id, channel, title) for diagnostics/tests

    async def ensure_init(self) -> bool:
        """Create the channels once (iOS also asks for permission here)."""
        if self._initialised is None:
            try:
                self._initialised = bool(await self.native.init_notifications())
            except Exception as exc:
                log.warning("init_notifications failed: %s", exc)
                self._initialised = False
        return bool(self._initialised)

    async def _show(self, notification_id: int, title: str, body: str, *, channel: str, route: Optional[str],
                    actions: Optional[list] = None) -> bool:
        await self.ensure_init()
        payload = _deep_link(route) if route else None
        kwargs: dict = {"channel_id": channel, "payload": payload}
        if actions:
            kwargs["actions"] = actions
        try:
            ok = bool(await self.native.call("show_notification", notification_id, title, body, default=False, **kwargs))
        except Exception as exc:
            log.warning("show_notification failed: %s", exc)
            ok = False
        self.posted.append((notification_id, channel, title))
        del self.posted[:-20]
        return ok

    async def job_finished(self, snap: Any) -> bool:
        title, body = done_text(snap)
        actions = [{"id": ACTION_OPEN, "title": "Open"}]
        if len(snap.outputs) == 1:
            actions.append({"id": ACTION_SHARE, "title": "Share"})
        await self.cancel_action(snap.id)
        return await self._show(_notification_id(_DONE_BASE, snap.id), title, body, channel=CHANNEL_DONE,
                                route=job_route(snap.id), actions=actions)

    async def glossary_review_needed(self, snap: Any) -> bool:
        """Tap -> the approval card (the owning chat; UI_SPEC §1.9), else the job."""
        return await self._show(_notification_id(_ACTION_BASE, snap.id), "Glossary ready: review needed",
                                f"{snap.title}: choose Edit, Yes or No", channel=CHANNEL_ACTION,
                                route=question_route(snap))

    async def paused(self, snap: Any, reason: str = "system limit") -> bool:
        return await self._show(_notification_id(_ACTION_BASE, snap.id), f"Paused ({reason}): tap to resume",
                                snap.title, channel=CHANNEL_ACTION, route=job_route(snap.id),
                                actions=[{"id": ACTION_RESUME, "title": "Resume"}])

    async def sign_in_required(self, account_label: str = "ChatGPT", route: str = "/settings/accounts") -> bool:
        return await self._show(_notification_id(_ACTION_BASE, "sign-in:" + account_label),
                                f"Sign-in required for {account_label}", "Tap to sign in", channel=CHANNEL_ACTION,
                                route=route)

    async def cancel_action(self, job_id: str) -> None:
        try:
            await self.native.cancel_notification(_notification_id(_ACTION_BASE, job_id))
        except Exception:
            pass

    @staticmethod
    def parse_event(event: Mapping[str, Any]) -> Optional[NotificationTap]:
        """A ``notification`` event -> what to do; None for payloads that are not ours."""
        payload = str((event or {}).get("payload") or "")
        if not payload.startswith(DEEP_LINK_PREFIX + "/"):
            return None
        route = payload[len(DEEP_LINK_PREFIX):]
        job_id = None
        if route.startswith("/job/"):
            job_id = route[len("/job/"):].split("?", 1)[0].split("/", 1)[0] or None
        action = str(event.get("action_id") or ACTION_OPEN)
        if action not in (ACTION_OPEN, ACTION_SHARE, ACTION_RESUME):
            action = ACTION_OPEN
        return NotificationTap(action=action, route=route, job_id=job_id)
