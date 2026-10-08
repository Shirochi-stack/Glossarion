"""Job notifications (UI_SPEC §1.9) over the GlossarionNative extension.

Channels (created by ``init_notifications`` from the extension's
``DEFAULT_NOTIFICATION_CHANNELS``):

* ``jobs.progress``: the ongoing Android foreground-service notification
  (owned by ``BackgroundExecution``: ``start/update/stop_job_service``);
* ``jobs.done``: "Done: *Book*" / "Stopped: *Book* (12/80)" / "Failed: *Book*";
  tap -> the job's route, actions **Open** and **Share** (when the job has one
  compiled output);
* ``jobs.action``: "Glossary ready: review needed" (actions **Accept** /
  **Review**), "Answer needed: *question*" (Tools › Async batch), "Sign-in
  required for ChatGPT", "Paused (system limit): tap to resume".

Payloads are app routes as deep links (``glossarion://app/job/<jid>``; a chat
job's glossary question opens its chat, ``/chat/<cid>``), never paths or user text. ``parse_event`` turns a tap/action event back into
``(action, route, job_id, notification_id)`` for the app. Android action buttons
always bring the app forward (``PendingIntent.getActivity``), so **Accept** is
answered by the app (``JobsFeature._on_notification``), never in the background.
All calls go through ``NativeBridge`` (safe defaults on desktop / unsupported
platforms); a post the platform refuses (no permission, notifications off) is
logged and kept in ``last_result``. Pure asyncio, no Flet import.
"""

from __future__ import annotations

import logging
import zlib
from dataclasses import dataclass
from typing import Any, Mapping, Optional

__all__ = [
    "ACTION_ACCEPT",
    "ACTION_OPEN",
    "ACTION_RESUME",
    "ACTION_SHARE",
    "ASYNC_BATCH_QUESTION",
    "ASYNC_BATCH_ROUTE",
    "CHANNEL_ACTION",
    "CHANNEL_DONE",
    "CHANNEL_PROGRESS",
    "JobNotifications",
    "NotificationTap",
    "action_notification_id",
    "chat_of",
    "done_text",
    "job_route",
    "question_route",
    "question_title",
]

log = logging.getLogger("glossarion.notifications")

CHANNEL_PROGRESS = "jobs.progress"
CHANNEL_DONE = "jobs.done"
CHANNEL_ACTION = "jobs.action"

ACTION_OPEN = "open"
ACTION_SHARE = "share"
ACTION_RESUME = "resume"
ACTION_ACCEPT = "accept"  # "Glossary ready" -> answer the pending glossary gate Yes (JobsFeature)
_ACTIONS = (ACTION_OPEN, ACTION_SHARE, ACTION_RESUME, ACTION_ACCEPT)

#: Tools › Async batch asks the dialog's questions through the job (``ui/tools/async_batch.py``).
ASYNC_BATCH_QUESTION = "async_batch_question"
ASYNC_BATCH_ROUTE = "/tools/async"

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


def _question_of(snap: Any) -> Mapping:
    question = getattr(snap, "question", None)
    return question if isinstance(question, Mapping) else {}


def question_route(snap: Any) -> str:
    """Where a job's blocking question is answered: Tools › Async batch for its dialog questions, the
    owning chat (its approval card), a Library book's page (U9: the review gate's approval sheet, UI_SPEC
    §3.10), else the job."""
    if str(_question_of(snap).get("kind") or "") == ASYNC_BATCH_QUESTION:
        return ASYNC_BATCH_ROUTE
    cid = chat_of(snap)
    if cid is not None:
        return f"/chat/{cid}"
    origin = getattr(getattr(snap, "spec", None), "origin", None) or {}
    bid = str(origin.get("bid") or "") if isinstance(origin, Mapping) and origin.get("type") == "library" else ""
    if bid:
        return f"/library/book/{bid}"
    return job_route(snap.id)


def question_title(snap: Any, question: Any = None) -> str:
    """What a non-glossary question asks (the Async batch dialog's title), else the job's title."""
    question = question if isinstance(question, Mapping) else _question_of(snap)
    data = question.get("data")
    title = str((data if isinstance(data, Mapping) else {}).get("title") or "").strip()
    return title or str(getattr(snap, "title", "") or "Glossarion")


def _deep_link(route: str) -> str:
    return DEEP_LINK_PREFIX + route


def _notification_id(base: int, key: str) -> int:
    return base + (zlib.crc32(key.encode("utf-8")) % _ID_SPAN)


def action_notification_id(job_id: str) -> int:
    """The id of job ``job_id``'s ``jobs.action`` notification (glossary review, answer needed, paused)."""
    return _notification_id(_ACTION_BASE, str(job_id))


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
    action: str  # "open" (body tap or Open / Review), "share", "resume", "accept"
    route: Optional[str]  # whitelisted app route ("/job/<jid>")
    job_id: Optional[str]
    notification_id: Optional[int] = None  # the tapped notification (None when the platform did not say)


class JobNotifications:
    def __init__(self, native: Any, *, platform: str = "desktop") -> None:
        self.native = native
        self.platform = platform
        self._initialised: Optional[bool] = None
        self.posted: list[tuple] = []  # (id, channel, title) for diagnostics/tests
        #: The last post: {"id", "channel", "title", "ok"} (Settings › Notifications and Diagnostics show it).
        self.last_result: Optional[dict] = None

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
        if not ok:
            # Android refuses without POST_NOTIFICATIONS or with the app's notifications turned off.
            log.info("notification %r on %s was not shown (notifications off or not permitted)", title, channel)
        self.posted.append((notification_id, channel, title))
        del self.posted[:-20]
        self.last_result = {"id": notification_id, "channel": channel, "title": title, "ok": ok}
        return ok

    async def test_notification(self) -> bool:
        """Settings › Notifications & background › "Send a test notification" (on ``jobs.done``)."""
        return await self._show(_notification_id(_DONE_BASE, "test"), "Glossarion test notification",
                                "Notifications work: finished jobs and reviews will show here.",
                                channel=CHANNEL_DONE, route="/settings/notifications")

    async def job_finished(self, snap: Any) -> bool:
        title, body = done_text(snap)
        actions = [{"id": ACTION_OPEN, "title": "Open"}]
        if len(snap.outputs) == 1:
            actions.append({"id": ACTION_SHARE, "title": "Share"})
        await self.cancel_action(snap.id)
        return await self._show(_notification_id(_DONE_BASE, snap.id), title, body, channel=CHANNEL_DONE,
                                route=job_route(snap.id), actions=actions)

    async def glossary_review_needed(self, snap: Any) -> bool:
        """Tap / **Review** -> the approval card (the owning chat, a Library book's page; UI_SPEC §1.9), else
        the job; **Accept** answers Yes (``JobsFeature._on_notification``)."""
        return await self._show(action_notification_id(snap.id), "Glossary ready: review needed",
                                f"{snap.title}: choose Edit, Yes or No", channel=CHANNEL_ACTION,
                                route=question_route(snap),
                                actions=[{"id": ACTION_ACCEPT, "title": "Accept"}, {"id": ACTION_OPEN, "title": "Review"}])

    async def answer_needed(self, snap: Any, title: Optional[str] = None) -> bool:
        """Any other blocking question (Tools › Async batch): "Answer needed: <question>" -> where it is
        answered (``question_route``)."""
        return await self._show(action_notification_id(snap.id), f"Answer needed: {title or question_title(snap)}",
                                f"{snap.title}: tap to answer", channel=CHANNEL_ACTION, route=question_route(snap))

    async def paused(self, snap: Any, reason: str = "system limit") -> bool:
        return await self._show(action_notification_id(snap.id), f"Paused ({reason}): tap to resume",
                                snap.title, channel=CHANNEL_ACTION, route=job_route(snap.id),
                                actions=[{"id": ACTION_RESUME, "title": "Resume"}])

    async def sign_in_required(self, account_label: str = "ChatGPT", route: str = "/settings/accounts") -> bool:
        return await self._show(_notification_id(_ACTION_BASE, "sign-in:" + account_label),
                                f"Sign-in required for {account_label}", "Tap to sign in", channel=CHANNEL_ACTION,
                                route=route)

    async def cancel_action(self, job_id: str) -> None:
        try:
            await self.native.cancel_notification(action_notification_id(job_id))
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
        if action not in _ACTIONS:
            action = ACTION_OPEN
        try:
            raw_id = event.get("notification_id")
            notification_id = int(raw_id) if raw_id is not None and int(raw_id) >= 0 else None
        except (TypeError, ValueError):
            notification_id = None
        return NotificationTap(action=action, route=route, job_id=job_id, notification_id=notification_id)
