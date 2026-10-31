"""LivePanel (UI_SPEC §3.11, §5.8): the native live "Translate this chapter" view.

A half-height draggable ``BottomSheet`` over the Reader, never inside the
WebView: the status line ("🛰️ Translating “f” — waiting for stream…"), the
streamed chapter rendered as ``Markdown`` (fed by ``live.LiveFeed``; it follows
the stream unless the reader scrolls up), a "🧠 Thinking (n)" ``ExpansionTile``
holding the thinking text and the pipeline log (at the log size, ``theme.log_text``),
"⏹ Stop" and "✕ Hide" (the job keeps running; the Reader's 🌐 becomes
"🛰️ Live view" and reopens this sheet). States: ``waiting`` · ``streaming`` ·
``finished`` · ``stopped`` · ``failed``.

The chapter area and the sheet's full height wait for the first streamed text (owner 2026-10-31: no empty
space while the model thinks), and ``sync_job`` keeps the status and the job line under it on what the
progress strip and the Jobs page say about the same job.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import flet as ft

from glossarion_mobile.ui.components._handlers import call_handler
from glossarion_mobile.ui.components.dialogs import close_dialog
from glossarion_mobile.ui.reader.live import (
    LiveFeed,
    status_queued,
    status_stopping,
    status_streaming,
    status_waiting,
    thinking_label,
)
from glossarion_mobile.ui.theme import log_text

__all__ = ["LivePanel"]

WAITING = "waiting"
STREAMING = "streaming"
FINISHED = "finished"
STOPPED = "stopped"
FAILED = "failed"
_RENDER_EVERY = 0.25  # seconds between Markdown re-renders while streaming


class LivePanel:
    def __init__(
        self,
        *,
        chapter_file: str,
        feed: LiveFeed,
        on_stop: Optional[Callable[[], Any]] = None,
        on_hide: Optional[Callable[[], Any]] = None,
        mono_family: str = "monospace",
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        import time

        self.chapter_file = chapter_file
        self.feed = feed
        self.on_stop = on_stop
        self.on_hide = on_hide
        self.state = WAITING
        self._clock = clock or time.monotonic
        self._last_render = 0.0
        self._rendered_version = -1
        self._rendered_side = -1
        self._follow = True
        self._page: Any = None
        self.is_open = False
        self.full_height = 440.0
        self.waiting_text = status_waiting(chapter_file)

        self.status = ft.Text(self.waiting_text, theme_style=ft.TextThemeStyle.LABEL_LARGE,
                              color=ft.Colors.TERTIARY, key="live-status")
        self.job_line = ft.Text("", theme_style=ft.TextThemeStyle.LABEL_SMALL, color=ft.Colors.ON_SURFACE_VARIANT,
                                visible=False, key="live-job")
        self.progress = ft.ProgressBar(height=2, visible=True)
        self.content_md = ft.Markdown("", selectable=True, extension_set=ft.MarkdownExtensionSet.GITHUB_WEB,
                                      key="live-content")
        self.content_column = ft.Column([self.content_md], scroll=ft.ScrollMode.AUTO, auto_scroll=True,
                                        expand=True, on_scroll=self._on_scroll, key="live-scroll")
        self.side_text = log_text("", family=mono_family, color=ft.Colors.ON_SURFACE_VARIANT, key="live-side")
        self.thinking_tile = ft.ExpansionTile(
            title=ft.Text(thinking_label(0), theme_style=ft.TextThemeStyle.LABEL_LARGE),
            controls=[ft.Container(content=ft.Column([self.side_text], scroll=ft.ScrollMode.AUTO, auto_scroll=True,
                                                     height=160), padding=ft.Padding.symmetric(horizontal=12))],
            expanded=False,
            dense=True,
            key="live-thinking",
        )
        self.stop_button = ft.FilledTonalButton(content="⏹ Stop", on_click=self._on_stop, key="live-stop")
        self.hide_button = ft.TextButton(content="✕ Hide", on_click=self._on_hide, key="live-hide")
        self.content_box = ft.Container(content=self.content_column, expand=True, visible=False,
                                        padding=ft.Padding.symmetric(vertical=4), key="live-content-box")
        self.body_column = ft.Column(
            [
                ft.Row([ft.Container(ft.Column([self.status, self.job_line], spacing=2, tight=True), expand=True),
                        self.hide_button],
                       vertical_alignment=ft.CrossAxisAlignment.CENTER),
                self.progress,
                self.content_box,
                self.thinking_tile,
                ft.Row([self.stop_button], alignment=ft.MainAxisAlignment.END),
            ],
            spacing=6,
            tight=True,
        )
        body = ft.Container(padding=ft.Padding.only(left=16, right=16, bottom=12), content=self.body_column)
        self.sheet = ft.BottomSheet(
            content=ft.Container(content=body),
            draggable=True,
            show_drag_handle=True,
            dismissible=True,
            bgcolor=ft.Colors.SURFACE_CONTAINER_HIGH,
            on_dismiss=self._on_dismiss,
        )

    # ---- visibility ------------------------------------------------------------------------

    def show(self, page: Any, height: Optional[float] = None) -> None:
        self._page = page
        if height:
            try:
                from glossarion_mobile.ui.components.sheet import sheet_body

                self.full_height = max(280.0, float(height) * 0.55)
                self._size_sheet(sheet_body(self.sheet))
            except Exception:
                pass
        self.render(force=True)
        self.is_open = True
        page.show_dialog(self.sheet)

    def close(self) -> None:
        page = self._page
        close_dialog(page, self.sheet)
        self.is_open = False

    def _on_dismiss(self, e: Any = None) -> None:
        self.is_open = False
        call_handler(self.on_hide)

    def _on_hide(self, e: Any = None) -> None:
        self.close()
        call_handler(self.on_hide)

    def _on_stop(self, e: Any = None) -> None:
        if self.state not in (WAITING, STREAMING):
            return
        self._show_stopping()
        call_handler(self.on_stop)

    def _show_stopping(self) -> None:
        self.status.value = status_stopping()
        self.stop_button.disabled = True
        self._push(self.status, self.stop_button)

    def _on_scroll(self, e: Any) -> None:
        try:
            pixels = float(getattr(e, "pixels", 0.0) or 0.0)
            extent = float(getattr(e, "max_scroll_extent", 0.0) or 0.0)
        except (TypeError, ValueError):
            return
        # only noted: Flet's client suspends the column's auto_scroll pin itself while the reader is scrolled
        # up. Turning auto_scroll off and on rebuilt the scroll view at the chapter's top (v9.15.7 hotfix 1)
        self._follow = pixels >= extent - 8

    # ---- feeding ---------------------------------------------------------------------------

    def add_lines(self, lines: Any) -> None:
        """New job log lines (UI loop): classify, then re-render at most every 250 ms."""
        grew = self.feed.feed(lines)
        if grew and self.state == WAITING:
            self.state = STREAMING
            if not self.stop_button.disabled:  # not while it is stopping
                self.status.value = status_streaming(self.chapter_file)
                self._push(self.status)
        self.render()

    def sync_job(self, snap: Any) -> None:
        """The job changed (UI loop): the panel says what the strip and the Jobs page say about it - queued,
        stopping (a Stop from there too), its progress line. The job's end is ``finish``."""
        if self.state not in (WAITING, STREAMING) or getattr(snap, "is_terminal", False):
            return
        from glossarion_mobile.services.jobs import JobState, progress_line

        state = getattr(snap, "state", None)
        if state in (JobState.STOPPING, JobState.FORCE_STOPPING):
            if not self.stop_button.disabled:
                self._show_stopping()
        elif state is JobState.QUEUED:
            self.status.value = status_queued(self.chapter_file)
            self._push(self.status)
        elif self.state == WAITING and self.status.value != self.waiting_text:
            self.status.value = self.waiting_text
            self._push(self.status)
        line = progress_line(snap, verb=True, elapsed=False)
        if line != self.job_line.value:
            self.job_line.value = line
            self.job_line.visible = bool(line)
            self._push(self.job_line)

    def render(self, *, force: bool = False) -> None:
        now = self._clock()
        if not force and now - self._last_render < _RENDER_EVERY:
            return
        self._last_render = now
        changed = []
        if self.feed.content_version != self._rendered_version:
            self._rendered_version = self.feed.content_version
            self.content_md.value = self.feed.markdown()
            changed.append(self.content_md)
            if self.content_md.value and not self.content_box.visible:
                self._open_chapter_area()
        if self.feed.side_version != self._rendered_side:
            self._rendered_side = self.feed.side_version
            self.side_text.value = self.feed.side_text[-20000:]
            title = self.thinking_tile.title
            if isinstance(title, ft.Text):
                title.value = thinking_label(self.feed.side_count)
            changed.append(self.thinking_tile)
        if changed:
            self._push(*changed)

    def _size_sheet(self, holder: Any) -> None:
        """The sheet's height: as much as the lines above need until there is text to show, then its share of
        the screen with the chapter area filling it."""
        holder.height = self.full_height if self.content_box.visible else None

    def _open_chapter_area(self) -> None:
        """The first streamed text arrived."""
        from glossarion_mobile.ui.components.sheet import sheet_body

        self.content_box.visible = True
        self.body_column.expand = True
        self.body_column.tight = False
        holder = sheet_body(self.sheet)
        self._size_sheet(holder)
        self._push(holder)

    def finish(self, text: str, *, stopped: bool = False, completed: bool = False) -> None:
        self.render(force=True)
        self.state = FINISHED if completed else (STOPPED if stopped else FAILED)
        self.status.value = text
        self.status.color = ft.Colors.PRIMARY if completed else (ft.Colors.TERTIARY if stopped else ft.Colors.ERROR)
        self.progress.visible = False
        self.stop_button.disabled = True
        self._push(self.status, self.progress, self.stop_button)

    def _push(self, *controls: Any) -> None:
        for control in controls:
            try:
                control.update()
            except Exception:  # not mounted yet (tests) or sheet closed
                pass

