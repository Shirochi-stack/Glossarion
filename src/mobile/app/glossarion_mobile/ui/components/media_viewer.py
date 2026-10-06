"""MediaViewer (UI_SPEC §5.5): the full-screen view of generated media.

* images: ``InteractiveViewer`` (pinch zoom 1-6x, pan) around ``Image(fit=CONTAIN)`` on black;
  several images of one response page with a swipe (``PageView``), "2 / 5" in the title;
* video: the ``flet_video.Video`` player (default controls) filling the view;
* audio: the chat's AudioCard (same ``AudioHub``).

App bar: ✕ · title · Save · Share · Open externally (the chat view's callbacks, which use
FileBridge's ExportSheet / share sheet; "Open externally" hands the file to the system).
It is a plain ``ft.View`` pushed with the shell's overlay stack (``ChatEnv.push_overlay``).
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

import flet as ft

from glossarion_mobile.ui.chat.media_model import MediaItem
from glossarion_mobile.ui.theme import HIT_TARGET

__all__ = ["MAX_ZOOM", "MediaViewer"]

MAX_ZOOM = 6.0


class MediaViewer:
    def __init__(
        self,
        items: Sequence[MediaItem],
        index: int = 0,
        *,
        on_close: Optional[Callable[[], Any]] = None,
        on_save: Optional[Callable[[str], Any]] = None,
        on_share: Optional[Callable[[str], Any]] = None,
        on_open_external: Optional[Callable[[str], Any]] = None,
        audio_hub: Any = None,
        spawn: Optional[Callable[[Any], Any]] = None,
    ) -> None:
        self.items = [i for i in items if i.exists] or list(items)
        self.index = max(0, min(int(index), max(0, len(self.items) - 1)))
        self.on_close = on_close
        self.on_save = on_save
        self.on_share = on_share
        self.on_open_external = on_open_external
        self.audio_hub = audio_hub
        self.spawn = spawn
        self.title = ft.Text("", max_lines=1, overflow=ft.TextOverflow.ELLIPSIS, color=ft.Colors.WHITE)
        self.pages: Optional[ft.PageView] = None
        self.viewers: list = []
        self.body = self._body()
        self._refresh_title()
        actions = [
            ft.IconButton(icon=ft.Icons.SAVE_ALT, tooltip="Save", icon_color=ft.Colors.WHITE, size_constraints=HIT_TARGET,
                          on_click=lambda e: self._act(self.on_save), disabled=on_save is None, key="viewer-save"),
            ft.IconButton(icon=ft.Icons.IOS_SHARE, tooltip="Share", icon_color=ft.Colors.WHITE, size_constraints=HIT_TARGET,
                          on_click=lambda e: self._act(self.on_share), disabled=on_share is None, key="viewer-share"),
            ft.IconButton(icon=ft.Icons.OPEN_IN_NEW, tooltip="Open externally", icon_color=ft.Colors.WHITE,
                          size_constraints=HIT_TARGET, on_click=lambda e: self._act(self.on_open_external),
                          disabled=on_open_external is None, key="viewer-open-external"),
        ]
        self.view = ft.View(
            route="/chat/media",
            appbar=ft.AppBar(
                leading=ft.IconButton(icon=ft.Icons.CLOSE, tooltip="Close", icon_color=ft.Colors.WHITE,
                                      size_constraints=HIT_TARGET, on_click=lambda e: self.close(), key="viewer-close"),
                title=self.title,
                actions=actions,
                bgcolor=ft.Colors.BLACK,
                automatically_imply_leading=False,
            ),
            controls=[ft.SafeArea(content=self.body, expand=True)],
            bgcolor=ft.Colors.BLACK,
            padding=0,
        )

    # ---- building -------------------------------------------------------------------------------

    @property
    def current(self) -> Optional[MediaItem]:
        return self.items[self.index] if self.items else None

    def _image(self, item: MediaItem) -> ft.Control:
        viewer = ft.InteractiveViewer(
            content=ft.Image(src=item.path, fit=ft.BoxFit.CONTAIN, semantics_label=item.name,
                             error_content=ft.Text("Generated image unavailable.", color=ft.Colors.WHITE)),
            min_scale=1.0,
            max_scale=MAX_ZOOM,
            pan_enabled=True,
            scale_enabled=True,
            expand=True,
        )
        self.viewers.append(viewer)
        return ft.Container(content=viewer, alignment=ft.Alignment.CENTER, expand=True, bgcolor=ft.Colors.BLACK)

    def _body(self) -> ft.Control:
        item = self.current
        if item is None:
            return ft.Container(content=ft.Text("Nothing to show", color=ft.Colors.WHITE), alignment=ft.Alignment.CENTER,
                                expand=True)
        images = [i for i in self.items if i.kind == "image"]
        if item.kind == "image" and len(images) > 1:
            self.items = images
            self.pages = ft.PageView(controls=[self._image(i) for i in images], selected_index=self.index,
                                     on_change=self._on_page, expand=True)
            return self.pages
        if item.kind == "image":
            return self._image(item)
        if item.kind == "video":
            from glossarion_mobile.ui.chat.media_cards import VideoCard

            return ft.Container(content=VideoCard(item, available_width=920), alignment=ft.Alignment.CENTER,
                                expand=True, padding=8)
        from glossarion_mobile.ui.chat.media_cards import AudioCard

        return ft.Container(content=AudioCard(item, hub=self.audio_hub, spawn=self.spawn), alignment=ft.Alignment.CENTER,
                            expand=True, padding=16)

    def _refresh_title(self) -> None:
        item = self.current
        name = item.name if item is not None else ""
        self.title.value = f"{self.index + 1} / {len(self.items)} · {name}" if len(self.items) > 1 else name

    # ---- events ---------------------------------------------------------------------------------

    def _on_page(self, e: Any = None) -> None:
        value = getattr(getattr(e, "control", None), "selected_index", None)
        if value is None and self.pages is not None:
            value = self.pages.selected_index
        self.show_index(int(value or 0))

    def show_index(self, index: int) -> None:
        self.index = max(0, min(int(index), len(self.items) - 1))
        self._refresh_title()
        try:
            self.title.update()
        except Exception:
            pass

    def _act(self, handler: Optional[Callable[[str], Any]]) -> Any:
        item = self.current
        if handler is None or item is None:
            return None
        return handler(item.path)

    def close(self) -> None:
        if self.on_close is not None:
            self.on_close()
