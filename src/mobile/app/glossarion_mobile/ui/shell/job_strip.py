"""JobStrip (UI_SPEC §1.7, §5.1): the 44 dp "mini-player" for the running job.

Leading ``ProgressRing(28)`` with the job-kind icon in its centre
(indeterminate until a total is known) · title (labelLarge) over subtitle
(labelSmall; warning colour for "Waiting for your glossary decision") · "+N"
queued badge · Stop (same state vocabulary as Send). Tap opens job detail.
``Semantics(live_region=True)`` so screen readers announce progress.

``AppState.job_strip`` is fed by the jobs feature (``JobService`` snapshots,
``services.jobs.strip_model_for``); it is hidden while no job runs. After a
job ends the strip shows "Done · Book" / "Stopped · Book" / "Failed · Book"
with an **Open** / **View** button (the feature clears it after 10 s).
Swiping it down hides it until the job's state changes (UI_SPEC §1.7).

44 dp is a minimum, not a fixed height (UI_SPEC §7.5: nothing but images has a fixed
height): at 200 % text the two lines need more and the strip grows instead of clipping them.
The live region announces at most once per ``ANNOUNCE_SECONDS`` (§7.5); in between only
the visible text changes.
"""

from __future__ import annotations

import time
from dataclasses import field
from typing import Any, Callable, Optional

import flet as ft

from glossarion_mobile.state.app_state import JobStripModel
from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.theme import HIT_TARGET, icon_data, semantic
from glossarion_mobile.ui.components.status import count_badge

__all__ = ["JobStrip"]


@ft.control
class JobStrip(ft.Container):
    model: Optional[JobStripModel] = field(default=None, metadata={"skip": True})
    on_open: Optional[Callable[[], Any]] = field(default=None, metadata={"skip": True})
    on_stop: Optional[Callable[[], Any]] = field(default=None, metadata={"skip": True})
    on_dismiss: Optional[Callable[[], Any]] = field(default=None, metadata={"skip": True})
    dark: bool = field(default=False, metadata={"skip": True})

    #: model.state values after the job ended (no Stop; an Open/View button instead).
    ENDED_STATES = ("done", "failed")
    #: Downward fling speed (logical px/s) that dismisses the strip.
    DISMISS_VELOCITY = 250.0
    #: Screen readers hear the strip at most this often (UI_SPEC §7.5).
    ANNOUNCE_SECONDS = 10.0

    def init(self) -> None:
        super().init()
        self.ring = ft.ProgressRing(width=28, height=28, stroke_width=3)
        self.kind_icon = ft.Icon(ft.Icons.TRANSLATE, size=14)
        self.title_text = ft.Text("", theme_style=ft.TextThemeStyle.LABEL_LARGE, max_lines=1, overflow=ft.TextOverflow.ELLIPSIS)
        self.subtitle_text = ft.Text("", theme_style=ft.TextThemeStyle.LABEL_SMALL, max_lines=1, overflow=ft.TextOverflow.ELLIPSIS)
        self.stop_button = ft.IconButton(
            icon=ft.Icons.STOP_CIRCLE,
            tooltip="Stop",
            on_click=self._stop,
            size_constraints=HIT_TARGET,
        )
        self.end_icon = ft.Icon(ft.Icons.TASK_ALT, size=24, visible=False)
        self.open_button = ft.TextButton(content="Open", on_click=self._open, visible=False)
        self.dismissed_state: Optional[str] = None
        self._announced_at = 0.0
        self._announced_state: Optional[str] = None
        self.semantics = ft.Semantics(
            live_region=True,
            content=ft.Row(
                [
                    # 44 dp minimum height: the row grows with the text (200 % scale) instead of clipping
                    ft.Container(width=0, height=tokens.SIZES["job_strip"]),
                    ft.Stack(
                        [
                            self.ring,
                            ft.Container(width=28, height=28, alignment=ft.Alignment.CENTER, content=self.kind_icon),
                            ft.Container(width=28, height=28, alignment=ft.Alignment.CENTER, content=self.end_icon),
                        ],
                        width=28,
                        height=28,
                    ),
                    ft.Column([self.title_text, self.subtitle_text], spacing=0, tight=True, expand=True),
                    self.open_button,
                    self.stop_button,
                ],
                spacing=10,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
        )
        self.gestures = ft.GestureDetector(content=self.semantics, on_vertical_drag_end=self._on_drag_end)
        self.content = self.gestures
        self.bgcolor = ft.Colors.SURFACE_CONTAINER_HIGHEST
        self.border_radius = tokens.RADII["job_strip"]
        self.margin = ft.Margin.symmetric(horizontal=8, vertical=4)
        self.padding = ft.Padding.only(left=8)
        self.on_click = self._open
        self.ink = True
        self._sync()

    def before_update(self) -> None:
        super().before_update()
        self._sync()

    def _sync(self) -> None:
        model = self.model
        if model is not None and self.dismissed_state is not None and model.state != self.dismissed_state:
            self.dismissed_state = None  # a state change brings the strip back
        self.visible = model is not None and model.state not in ("hidden",) and self.dismissed_state is None
        if model is None:
            self.dismissed_state = None
            return
        ended = model.state in self.ENDED_STATES
        self.ring.visible = not ended
        self.kind_icon.visible = not ended
        self.end_icon.visible = ended
        if ended:
            failed = model.state == "failed"
            self.end_icon.icon = ft.Icons.ERROR if failed else ft.Icons.TASK_ALT
            self.end_icon.color = ft.Colors.ERROR if failed else semantic("success", self.dark)
            self.open_button.content = "View" if failed else "Open"
        self.open_button.visible = ended
        self.ring.value = None if model.progress is None else max(0.0, min(1.0, model.progress))
        self.kind_icon.icon = icon_data(model.kind_icon)
        self.title_text.value = model.title
        self.subtitle_text.value = model.subtitle
        self.subtitle_text.color = semantic("warning", self.dark) if model.warning else None
        self.stop_button.badge = count_badge(f"+{model.queued}") if model.queued else None
        running = model.state in ("running", "finishing")
        self.stop_button.visible = running or model.state == "stopping"
        self.stop_button.disabled = model.state == "stopping"
        self.stop_button.icon = ft.Icons.HOURGLASS_BOTTOM if model.state == "finishing" else ft.Icons.STOP_CIRCLE
        self.stop_button.tooltip = {
            "finishing": "Graceful stop requested. Tap again to force stop.",
            "stopping": "Force stop requested",
        }.get(model.state, "Stop")
        self._announce(model)

    def _announce(self, model: JobStripModel) -> None:
        """Update the spoken label on a state change, else at most once per ``ANNOUNCE_SECONDS``
        (a live region re-announces every label change: progress ticks would flood TalkBack)."""
        now = time.monotonic()
        if model.state != self._announced_state or now - self._announced_at >= self.ANNOUNCE_SECONDS:
            self.semantics.label = f"{model.title}. {model.subtitle}".strip()
            self._announced_state = model.state
            self._announced_at = now

    def set_model(self, model: Optional[JobStripModel]) -> None:
        self.model = model
        self._sync()
        try:
            self.update()
        except Exception:
            pass

    def _open(self, e: Any = None) -> None:
        if self.on_open is not None:
            self.on_open()

    def _stop(self, e: Any = None) -> None:
        if self.on_stop is not None:
            self.on_stop()

    def dismiss(self) -> None:
        """Hide until the job's state changes (swipe down)."""
        if self.model is None:
            return
        self.dismissed_state = self.model.state
        self._sync()
        try:
            self.update()
        except Exception:
            pass
        if self.on_dismiss is not None:
            self.on_dismiss()

    def _on_drag_end(self, e: Any = None) -> None:
        velocity = getattr(e, "primary_velocity", None)
        if velocity is None:
            velocity = getattr(getattr(e, "velocity", None), "y", 0.0)
        try:
            if float(velocity or 0.0) >= self.DISMISS_VELOCITY:
                self.dismiss()
        except (TypeError, ValueError):
            pass
