"""LivePanel (UI_SPEC §3.11, §5.8): the native live "Translate this chapter" view.

A half-height draggable ``BottomSheet`` over the Reader, never inside the
WebView: the status line ("🛰️ Translating “f” — waiting for stream…"), the
streamed chapter rendered as ``Markdown`` (fed by ``live.LiveFeed``; it follows
the stream unless the reader scrolls up), a "🧠 Thinking (n)" ``ExpansionTile``
holding the thinking text and the pipeline log (at the log size, ``theme.log_text``),
"⏹ Stop" and "✕ Hide" (the job keeps running; the Reader's 🌐 becomes
"🛰️ Live view" and reopens this sheet). States: ``waiting`` · ``streaming`` ·
``finished`` · ``stopped`` · ``failed``.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import flet as ft

from glossarion_mobile.ui.components._handlers import call_handler
from glossarion_mobile.ui.components.dialogs import close_dialog
from glossarion_mobile.ui.reader.live import LiveFeed, status_stopping, status_waiting, thinking_label
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

        self.status = ft.Text(status_waiting(chapter_file), theme_style=ft.TextThemeStyle.LABEL_LARGE,
                              color=ft.Colors.TERTIARY, key="live-status")
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
        body = ft.Container(
            padding=ft.Padding.only(left=16, right=16, bottom=12),
            content=ft.Column(
                [
                    ft.Row([ft.Container(self.status, expand=True), self.hide_button],
                           vertical_alignment=ft.CrossAxisAlignment.CENTER),
                    self.progress,
                    ft.Container(content=self.content_column, expand=True, padding=ft.Padding.symmetric(vertical=4)),
                    self.thinking_tile,
                    ft.Row([self.stop_button], alignment=ft.MainAxisAlignment.END),
                ],
                spacing=6,
                expand=True,
            ),
        )
        self.sheet = ft.BottomSheet(
            content=ft.Container(content=body, height=440),
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
                self.sheet.content.height = max(280.0, float(height) * 0.55)
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
        self.status.value = status_stopping()
        self.stop_button.disabled = True
        self._push(self.status, self.stop_button)
        call_handler(self.on_stop)

    def _on_scroll(self, e: Any) -> None:
        try:
            pixels = float(getattr(e, "pixels", 0.0) or 0.0)
            extent = float(getattr(e, "max_scroll_extent", 0.0) or 0.0)
        except (TypeError, ValueError):
            return
        follow = pixels >= extent - 8
        if follow != self._follow:
            self._follow = follow
            self.content_column.auto_scroll = follow
            self._push(self.content_column)

    # ---- feeding ---------------------------------------------------------------------------

    def add_lines(self, lines: Any) -> None:
        """New job log lines (UI loop): classify, then re-render at most every 250 ms."""
        grew = self.feed.feed(lines)
        if grew and self.state == WAITING:
            self.state = STREAMING
        self.render()

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
        if self.feed.side_version != self._rendered_side:
            self._rendered_side = self.feed.side_version
            self.side_text.value = self.feed.side_text[-20000:]
            title = self.thinking_tile.title
            if isinstance(title, ft.Text):
                title.value = thinking_label(self.feed.side_count)
            changed.append(self.thinking_tile)
        if changed:
            self._push(*changed)

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

