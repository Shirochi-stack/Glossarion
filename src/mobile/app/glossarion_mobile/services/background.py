"""BackgroundExecution: keep a job running when the app leaves the screen (plan §4, UI_SPEC §1.9/§7.6).

Android
  * first run: ask for the notification permission (``flet_permission_handler``,
    Android 13+), and once, on the first long job, explain and request the
    battery-optimisation exemption (``Permission.IGNORE_BATTERY_OPTIMIZATIONS``;
    remembered in Prefs ``jobs_battery_prompt_done``);
  * job start: ``GlossarionNative.start_job_service(title, text)`` (dataSync
    foreground service with wake + Wi-Fi locks, buttons Stop / Open);
  * progress: ``update_job_service`` at most once per second;
  * notification **Stop** -> ``JobService.request_stop()`` (graceful first, a
    second tap forces); **Open** / tap -> the job's route;
  * ``on_foreground(type=timeout)`` (Android 15 dataSync 6h/24h budget) ->
    graceful stop (never escalates) + "Paused (system limit): tap to resume";
  * job end: ``stop_job_service`` (the ``jobs.done`` notification follows).

iOS
  * the Run tap: ``start_continued_processing(com.glossarion.app.job.<id>)``
    (iOS 26+, must be user-initiated, so ``prepare_for_run`` is awaited before
    ``JobService.submit``);
  * job start: ``begin_background_task`` (about 30 s after backgrounding on
    iOS < 26) with an expiry notification;
  * progress: ``update_continued_processing(done, total, subtitle)`` (1/s);
  * ``on_background_task(expiring | continued_expired)`` -> graceful stop +
    local "Paused: tap to resume" notification; jobs resume from
    ``translation_progress.json``;
  * job end: ``finish_continued_processing`` + ``end_background_task``.

Queued jobs inherit the running job's foreground service, background grant
and continued-processing task (Android refuses to start a foreground service
from the background), so they are only released when the queue drains. The
service is shared with a sign-in waiting in the browser (``OAuthBridge``)
through the bridge's ``ServiceHolds``: the last one to finish stops it.

Both: ``Wakelock`` while a job runs when "Keep screen on during jobs" is on
(Prefs ``keep_screen_on_during_jobs``; default on for iOS, off on Android
where the foreground service holds a CPU wake lock with the screen off).

Native calls go through ``NativeBridge`` (stub on desktop); permission and
wakelock objects are injected (tests use fakes). Event handlers run on the UI
loop. Pure asyncio, no Flet import at module scope.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any, Awaitable, Callable, Mapping, Optional

from glossarion_mobile.services.native import service_holds

__all__ = [
    "BackgroundExecution",
    "CONTINUED_ID_PREFIX",
    "IOS_BACKGROUND_NOTICE",
    "PREF_BATTERY_PROMPT",
    "PREF_KEEP_SCREEN_ON",
    "PREF_NOTIFICATION_ASKED",
]

log = logging.getLogger("glossarion.background")

CONTINUED_ID_PREFIX = "com.glossarion.app.job."
PREF_BATTERY_PROMPT = "jobs_battery_prompt_done"
PREF_NOTIFICATION_ASKED = "jobs_notification_permission_asked"
PREF_KEEP_SCREEN_ON = "keep_screen_on_during_jobs"
UPDATE_INTERVAL = 1.0  # seconds between foreground-service / Live Activity updates
JOBS_HOLD = "jobs"  # ServiceHolds name of the job runner

#: UI_SPEC §7.6: shown on iOS where jobs start (Jobs page, Plan card).
IOS_BACKGROUND_NOTICE = (
    "iOS may pause translation about 30 s after you leave the app (iOS 26+ continues in the background). "
    "A paused job resumes from its saved progress: open Jobs and tap Resume."
)

BATTERY_EXPLANATION = (
    "Long translations keep running with the screen off. Android may still pause them to save battery; "
    "allow Glossarion to ignore battery optimisation so jobs are not cut short."
)


class _PermissionRequester:
    """``flet_permission_handler.PermissionHandler`` created lazily (must be built on the loop)."""

    def __init__(self) -> None:
        self._handler: Any = None

    async def request(self, name: str) -> str:
        try:
            from flet_permission_handler import Permission, PermissionHandler
        except ImportError as exc:
            return f"unavailable ({exc})"
        if self._handler is None:
            self._handler = PermissionHandler()
        status = await asyncio.wait_for(self._handler.request(getattr(Permission, name)), 120)
        return str(getattr(status, "value", status))


class BackgroundExecution:
    def __init__(
        self,
        native: Any,
        *,
        platform: str = "desktop",
        jobs: Any = None,
        notifications: Any = None,
        prefs: Any = None,
        permissions: Any = None,
        wakelock: Any = None,
        confirm: Optional[Callable[[str, str], Awaitable[bool]]] = None,
        navigate_route: Optional[Callable[[str], Any]] = None,
        clock: Callable[[], float] = time.monotonic,
        update_interval: float = UPDATE_INTERVAL,
    ) -> None:
        self.native = native
        self.platform = platform
        self.jobs = jobs
        self.notifications = notifications
        self.prefs = prefs
        self.permissions = permissions if permissions is not None else _PermissionRequester()
        self.wakelock = wakelock
        self.confirm = confirm
        self.navigate_route = navigate_route
        self.clock = clock
        self.update_interval = update_interval

        self.service_running = False
        self.service_job: Optional[str] = None
        self.bg_task_id = -1
        self.continued_id: Optional[str] = None
        self.continued_started = False
        self.wakelock_on = False
        self._last_update = -1e9
        self._last_text: Optional[str] = None
        self._pending_continued: Optional[str] = None
        self.calls: list[str] = []  # what happened, for diagnostics and tests
        self.app_visible = True
        # Transitions are handled one at a time, in delivery order: a job that ends while its
        # start (ensure_init + start_job_service round trips) is still pending must not see
        # the service as not started yet and leave it running.
        self._transition_lock: Optional[asyncio.Lock] = None
        self._transition_loop: Any = None

    # ---- helpers --------------------------------------------------------------------------

    @property
    def is_android(self) -> bool:
        return self.platform == "android"

    @property
    def is_ios(self) -> bool:
        return self.platform == "ios"

    def _pref(self, key: str, default: Any = None) -> Any:
        if self.prefs is None:
            return default
        try:
            return self.prefs.get(key, default)
        except Exception:
            return default

    def _set_pref(self, key: str, value: Any) -> None:
        if self.prefs is not None:
            try:
                self.prefs.set(key, value)
            except Exception:
                log.debug("pref %s not saved", key, exc_info=True)

    def keep_screen_on(self) -> bool:
        return bool(self._pref(PREF_KEEP_SCREEN_ON, self.is_ios))

    async def _native(self, method: str, *args: Any, default: Any = None, **kwargs: Any) -> Any:
        self.calls.append(method)
        try:
            return await self.native.call(method, *args, default=default, **kwargs)
        except Exception as exc:
            log.warning("native %s failed: %s", method, exc)
            return default

    # ---- before submit (the Run tap) ---------------------------------------------------------

    async def prepare_for_run(self, spec: Any, *, long_job: bool = True) -> None:
        """Called from the Run/Send tap before ``JobService.submit``.

        Android: notification permission (once) and the battery-optimisation
        sheet (once, for long jobs). iOS: submits the BGContinuedProcessingTask
        request, which iOS only accepts in direct response to a user action.
        """
        if self.is_android:
            if not self._pref(PREF_NOTIFICATION_ASKED, False):
                self._set_pref(PREF_NOTIFICATION_ASKED, True)
                status = await self._request_permission("NOTIFICATION")
                log.info("notification permission: %s", status)
            if long_job and not self._pref(PREF_BATTERY_PROMPT, False):
                self._set_pref(PREF_BATTERY_PROMPT, True)
                proceed = True
                if self.confirm is not None:
                    try:
                        proceed = bool(await self.confirm("Keep translations running", BATTERY_EXPLANATION))
                    except Exception:
                        proceed = False
                if proceed:
                    status = await self._request_permission("IGNORE_BATTERY_OPTIMIZATIONS")
                    log.info("battery optimisation exemption: %s", status)
        elif self.is_ios:
            identifier = CONTINUED_ID_PREFIX + uuid.uuid4().hex[:12]
            title = str(getattr(spec, "title", "") or "Glossarion")
            started = await self._native(
                "start_continued_processing",
                identifier,
                title,
                "Starting…",
                default=False,
                expiration_title="Glossarion paused",
                expiration_body=f"{title}: tap to resume",
            )
            if started:
                self.continued_id = identifier
                self.continued_started = True
            if self.notifications is not None:
                await self.notifications.ensure_init()  # iOS asks for notification permission here

    async def _request_permission(self, name: str) -> str:
        self.calls.append(f"permission:{name}")
        try:
            return await self.permissions.request(name)
        except Exception as exc:
            return f"error {type(exc).__name__}: {exc}"

    # ---- job lifecycle (transitions from JobService, on the loop) ---------------------------------

    async def on_transition(self, snap: Any, previous: Any) -> None:
        """One JobService transition; transitions run strictly one after another (the callers
        spawn a task per transition, so without the lock a fast STARTING -> FAILED could run
        ``job_finished`` before ``job_started`` recorded the foreground service)."""
        from glossarion_mobile.services.jobs import JobState

        loop = asyncio.get_running_loop()
        if self._transition_lock is None or self._transition_loop is not loop:
            self._transition_lock = asyncio.Lock()  # one per event loop (tests run several)
            self._transition_loop = loop
        async with self._transition_lock:
            if snap.state is JobState.STARTING:
                await self.job_started(snap)
            elif snap.is_terminal and previous is not None and previous is not JobState.QUEUED:
                await self.job_finished(snap)
            elif snap.is_active:
                await self.job_progress(snap, force=True)

    async def job_started(self, snap: Any) -> None:
        from glossarion_mobile.services.jobs import notification_text

        text = notification_text(snap)
        if self.is_android:
            if self.notifications is not None:
                await self.notifications.ensure_init()
            if self.service_running:
                # The service stayed up for this queued job (Android does not allow starting a
                # foreground service while the app is in the background).
                await self._native("update_job_service", title="Glossarion", text=text)
            else:
                started = await self._native("start_job_service", "Glossarion", text, default=False,
                                             buttons=[{"id": "stop", "text": "Stop"}, {"id": "open", "text": "Open"}])
                self.service_running = bool(started)
            self.service_job = snap.id
            holds = service_holds(self.native)
            if holds is not None and self.service_running:
                holds.hold(JOBS_HOLD, "Glossarion", text)
        elif self.is_ios and self.bg_task_id < 0:
            task_id = await self._native("begin_background_task", f"job-{snap.id}", default=-1,
                                         expiration_title="Glossarion paused",
                                         expiration_body=f"{snap.title}: tap to resume",
                                         expiration_payload=f"glossarion://app/job/{snap.id}")
            try:
                self.bg_task_id = int(task_id)
            except (TypeError, ValueError):
                self.bg_task_id = -1
        if self.keep_screen_on() and self.wakelock is not None and not self.wakelock_on:
            try:
                await self.wakelock.enable()
                self.wakelock_on = True
                self.calls.append("wakelock:on")
            except Exception as exc:
                log.info("wakelock enable failed: %s", exc)
        self._last_update = self.clock()
        self._last_text = text

    async def job_progress(self, snap: Any, *, force: bool = False) -> bool:
        """Throttled (1/s) progress to the FGS notification / iOS Live Activity."""
        from glossarion_mobile.services.jobs import notification_text

        now = self.clock()
        if not force and now - self._last_update < self.update_interval:
            return False
        text = notification_text(snap)
        progress = snap.progress
        if text == self._last_text and not force:
            return False
        self._last_update = now
        self._last_text = text
        if self.is_android and self.service_running:
            holds = service_holds(self.native)
            if holds is not None:
                holds.hold(JOBS_HOLD, "Glossarion", text)  # what to show again after a sign-in releases it
            await self._native("update_job_service", title="Glossarion", text=text)
            return True
        if self.is_ios and self.continued_started:
            await self._native("update_continued_processing", int(progress.completed or 0),
                               int(progress.total or 0), text)
            return True
        return False

    async def job_finished(self, snap: Any) -> None:
        from glossarion_mobile.services.jobs import JobState

        if self.notifications is not None:
            await self.notifications.job_finished(snap)
        if self._next_job_waiting():
            if self.is_android and self.service_running:
                await self._native("update_job_service", title="Glossarion", text="Starting the next job…")
            return  # the next queued job inherits the service / background grant / wakelock
        if self.is_android and (self.service_running or self.service_job == snap.id):
            holds = service_holds(self.native)
            remaining = holds.release(JOBS_HOLD) if holds is not None else None
            if remaining is not None:
                # A sign-in waiting in the browser still needs the service: keep it, with its notification.
                title, text = remaining
                await self._native("update_job_service", title=title, text=text)
            else:
                await self._native("stop_job_service")
            self.service_running = False
            self.service_job = None
        if self.is_ios:
            if self.continued_started:
                await self._native("finish_continued_processing", snap.state is JobState.DONE)
                self.continued_started = False
                self.continued_id = None
            if self.bg_task_id >= 0:
                await self._native("end_background_task", self.bg_task_id)
                self.bg_task_id = -1
        if self.wakelock_on and self.wakelock is not None:
            try:
                await self.wakelock.disable()
                self.calls.append("wakelock:off")
            except Exception as exc:
                log.info("wakelock disable failed: %s", exc)
            self.wakelock_on = False

    # ---- native events (UI loop) ---------------------------------------------------------------

    def _active(self) -> Any:
        jobs = self.jobs
        return jobs.snapshot() if jobs is not None else None

    def _next_job_waiting(self) -> bool:
        """A queued job starts right after this one: keep the service / background grant."""
        view = getattr(self.jobs, "view", None) if self.jobs is not None else None
        if not callable(view):
            return False
        try:
            current = view()
        except Exception:
            return False
        return bool(getattr(current, "queue", ())) and not getattr(current, "paused", False)

    async def on_foreground_event(self, event: Mapping[str, Any]) -> None:
        """Android foreground-service events: Stop / Open buttons, tap, system timeout."""
        kind = str(event.get("type") or "")
        button = str(event.get("button_id") or "")
        snap = self._active()
        if kind == "button" and button == "stop":
            if self.jobs is not None:
                self.jobs.request_stop(reason="Stop pressed in the notification")
            return
        if (kind == "button" and button == "open") or kind == "tap":
            if snap is not None and self.navigate_route is not None:
                self.navigate_route(f"/job/{snap.id}")
            return
        if kind in ("timeout", "destroyed"):
            holds = service_holds(self.native)
            # Every stop ends with "destroyed"; a late one may follow a service that already runs again.
            if holds is not None and JOBS_HOLD in holds and not await self._native("is_job_service_running",
                                                                                    default=False):
                holds.release(JOBS_HOLD)  # the service is gone (OAuthBridge drops its own hold too)
        if kind == "timeout" or (kind == "destroyed" and event.get("is_timeout")):
            self.service_running = False
            if snap is not None and self.jobs is not None:
                self.jobs.request_stop(escalate=False, reason="Android ended the background service (time limit)")
                if self.notifications is not None:
                    await self.notifications.paused(snap, "system limit")
            return
        if kind == "destroyed":
            self.service_running = False

    async def on_background_task_event(self, event: Mapping[str, Any]) -> None:
        """iOS: the background grant or the continued-processing task is ending."""
        kind = str(event.get("type") or "")
        if kind == "expiring":
            if event.get("task_id") is not None and event.get("task_id") == self.bg_task_id:
                self.bg_task_id = -1  # the native side already ended it
        elif kind in ("continued_expired", "continued_failed"):
            self.continued_started = False
            if kind == "continued_failed":
                return
        else:
            return
        snap = self._active()
        if snap is not None and self.jobs is not None:
            self.jobs.request_stop(escalate=False, reason="iOS paused the app in the background")
            if self.notifications is not None and kind == "continued_expired":
                await self.notifications.paused(snap, "background time ended")

    def on_lifecycle(self, name: str) -> None:
        self.app_visible = name not in ("hide", "pause", "inactive", "detach")
