"""Media in assistant cards (UI_SPEC §2.9 item 4, §5.5): image gallery, VideoCard, AudioCard,
and the Refine card's "Compare with original" sheet.

The media of a response come from the shared desktop rule (``ChatStoreAdapter.message_media``
-> ``media_model.MediaItem``); these controls only show them:

* ``ImageGallery`` - one image at <= 76 % of the width (280-980 dp, max height 760, radius 12)
  or a wrap of thumbnails; tap opens the full-screen ``MediaViewer``; a missing file reads
  "Generated image unavailable." plus its name (desktop text);
* ``VideoCard`` - ``flet_video.Video(playlist=[VideoMedia(path)], aspect_ratio=16/9)`` with the
  package's default controls (1.0.3 has no ``show_controls``); ⋯ Save as… / Share / Open
  externally; without flet-video (or when playback fails) a ReasonChip and Open externally;
* ``AudioCard`` - "🔊 Generated audio", play/pause, seek ``Slider``, "m:ss / m:ss", a volume
  button revealing a volume ``Slider`` (default 75 %, the desktop media player's), ⋯ Save /
  Share / Open externally; playback through ``AudioHub`` (one ``flet_audio.Audio`` service per
  page). The hub follows the playing FILE, not a card instance: every card built for that file
  (a transcript re-render, the MediaViewer) adopts the playback state and receives its events;
  "Native audio playback is unavailable" otherwise;
* ``CompareSheet`` - the refined text stacked over the original, paragraph by paragraph
  (``media_model.compare_blocks``).

Actions (save / share / open externally / open viewer) are callbacks the chat view provides.
"""

from __future__ import annotations

import logging
import os
import weakref
from typing import Any, Callable, Optional, Sequence

import flet as ft

from glossarion_mobile.ui.chat.media_model import (
    AUDIO_DEFAULT_VOLUME,
    VIDEO_ASPECT_RATIO,
    MediaItem,
    capped_image_height,
    compare_blocks,
    format_clock,
    image_size,
)
from glossarion_mobile.ui.components.action_sheet import ActionItem, ActionSheet
from glossarion_mobile.ui.components.reason_chip import ReasonChip
from glossarion_mobile.ui.theme import HIT_TARGET

__all__ = [
    "AUDIO_UNAVAILABLE",
    "AudioCard",
    "AudioHub",
    "CompareSheet",
    "IMAGE_UNAVAILABLE",
    "ImageGallery",
    "MediaActions",
    "VIDEO_UNAVAILABLE",
    "VideoCard",
    "image_width",
    "media_section",
    "video_available",
]

log = logging.getLogger("glossarion.chat.media")

IMAGE_UNAVAILABLE = "Generated image unavailable."
AUDIO_UNAVAILABLE = "Native audio playback is unavailable"
VIDEO_UNAVAILABLE = "Video playback needs flet-video (not in this build)"
_MUTED = ft.Colors.with_opacity(0.6, ft.Colors.ON_SURFACE)


def image_width(available: Optional[float]) -> float:
    """76 % of the transcript width, clamped to 280-980 dp (UI_SPEC §2.9)."""
    width = float(available or 400) * 0.76
    return max(280.0, min(980.0, width))


def video_available() -> bool:
    try:
        import flet_video  # noqa: F401  (availability probe)

        return True
    except Exception:
        return False


class MediaActions:
    """Callbacks a media card / the viewer use (all optional; missing ones disable their item)."""

    def __init__(
        self,
        *,
        open_viewer: Optional[Callable[[Sequence[MediaItem], int], Any]] = None,
        save: Optional[Callable[[str], Any]] = None,
        share: Optional[Callable[[str], Any]] = None,
        open_external: Optional[Callable[[str], Any]] = None,
        page: Any = None,
        tablet: bool = False,
    ) -> None:
        self.open_viewer = open_viewer
        self.save = save
        self.share = share
        self.open_external = open_external
        self.page = page
        self.tablet = tablet

    def menu_items(self, item: MediaItem) -> list:
        missing = None if item.exists else "The generated file is missing"
        return [
            ActionItem("Save as…", (lambda: self.save(item.path)) if self.save else None, icon="SAVE_ALT",
                       disabled_reason=missing or (None if self.save else "Saving is not available here")),
            ActionItem("Share", (lambda: self.share(item.path)) if self.share else None, icon="IOS_SHARE",
                       disabled_reason=missing or (None if self.share else "Sharing is not available here")),
            ActionItem("Open externally", (lambda: self.open_external(item.path)) if self.open_external else None,
                       icon="OPEN_IN_NEW",
                       disabled_reason=missing or (None if self.open_external else "Not available here")),
        ]

    def show_menu(self, item: MediaItem) -> Optional[ActionSheet]:
        if self.page is None:
            return None
        sheet = ActionSheet(self.menu_items(item), title=item.name, tablet=self.tablet)
        sheet.show(self.page)
        return sheet


def _missing(kind: str, item: MediaItem) -> ft.Control:
    return ft.Container(
        content=ft.Column(
            [
                ft.Text(f"Generated {kind} unavailable." if kind != "image" else IMAGE_UNAVAILABLE,
                        weight=ft.FontWeight.W_600, theme_style=ft.TextThemeStyle.BODY_MEDIUM),
                ft.Text(item.name, theme_style=ft.TextThemeStyle.LABEL_SMALL, color=_MUTED, selectable=True),
            ],
            spacing=2,
            tight=True,
        ),
        padding=ft.Padding.all(10),
        border_radius=12,
        bgcolor=ft.Colors.SURFACE_CONTAINER_LOW,
        key=f"media-missing-{item.name}",
    )


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------


class ImageGallery(ft.Column):
    """Generated image(s) of one response; tap opens the full-screen MediaViewer."""

    def __init__(self, items: Sequence[MediaItem], *, actions: Optional[MediaActions] = None,
                 available_width: Optional[float] = None, key: Any = None) -> None:
        super().__init__(spacing=6, tight=True, key=key)
        self.items = [i for i in items if i.kind == "image"]
        self.actions = actions or MediaActions()
        self.images: list = []
        width = image_width(available_width)
        present = [i for i in self.items if i.exists]
        controls: list = []
        if len(present) == 1:  # max height 760 (UI_SPEC §2.9): the image's aspect from its header
            height = capped_image_height(width, image_size(present[0].path))
            controls.append(self._tile(present[0], 0, width=width, height=height, fit=ft.BoxFit.CONTAIN))
        elif present:
            thumb = max(120.0, (width - 6) / 2)
            controls.append(ft.Row([self._tile(item, n, width=thumb, height=thumb, fit=ft.BoxFit.COVER)
                                    for n, item in enumerate(present)], wrap=True, spacing=6, run_spacing=6))
        controls.extend(_missing("image", item) for item in self.items if not item.exists)
        self.controls = controls

    def _tile(self, item: MediaItem, position: int, *, width: float, height: Optional[float], fit: Any) -> ft.Control:
        image = ft.Image(src=item.path, fit=fit, width=width, height=height, border_radius=12, gapless_playback=True,
                         semantics_label=f"Generated image {item.name}",
                         error_content=_missing("image", MediaItem("image", item.path, False)))
        self.images.append(image)
        return ft.Container(
            content=image,
            border_radius=12,
            on_click=lambda e, n=position: self.open(n),
            on_long_press=lambda e, it=item: self.actions.show_menu(it),
            tooltip=item.name,
            key=f"image-{position}",
            height=height,
            width=width,
        )

    def open(self, position: int = 0) -> Any:
        present = [i for i in self.items if i.exists]
        if self.actions.open_viewer is None or not present:
            return None
        return self.actions.open_viewer(present, max(0, min(position, len(present) - 1)))


# ---------------------------------------------------------------------------
# Video
# ---------------------------------------------------------------------------


class VideoCard(ft.Container):
    """flet-video player (default controls, 16:9) + ⋯ Save as… / Share / Open externally."""

    def __init__(self, item: MediaItem, *, actions: Optional[MediaActions] = None,
                 available_width: Optional[float] = None, key: Any = None) -> None:
        super().__init__(key=key)
        self.item = item
        self.actions = actions or MediaActions()
        self.player: Any = None
        self.error_text = ft.Text("", theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ERROR, visible=False)
        header = ft.Row(
            [
                ft.Text("🎬 Generated video", theme_style=ft.TextThemeStyle.LABEL_LARGE, weight=ft.FontWeight.W_600),
                ft.Text(item.name, theme_style=ft.TextThemeStyle.LABEL_SMALL, color=_MUTED, expand=True,
                        max_lines=1, overflow=ft.TextOverflow.ELLIPSIS),
                ft.IconButton(icon=ft.Icons.MORE_HORIZ, tooltip="More", size_constraints=HIT_TARGET, icon_size=18,
                              on_click=lambda e: self.actions.show_menu(self.item), key="video-more"),
            ],
            spacing=8,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )
        body: ft.Control
        if not item.exists:
            body = _missing("video", item)
        else:
            body = self._player(available_width)
        self.content = ft.Column([header, body, self.error_text], spacing=6, tight=True)
        self.border_radius = 12
        self.padding = ft.Padding.all(8)
        self.bgcolor = ft.Colors.SURFACE_CONTAINER_LOW

    def _player(self, available_width: Optional[float]) -> ft.Control:
        try:
            import flet_video as fv

            self.player = fv.Video(
                playlist=[fv.VideoMedia(self.item.path)],
                aspect_ratio=VIDEO_ASPECT_RATIO,
                autoplay=False,
                width=min(920.0, float(available_width or 400)),
                on_error=self._on_error,
                key="video-player",
            )
            return self.player
        except Exception as exc:  # flet-video not in this build / the control could not be built
            log.info("video player unavailable: %s", exc)
            return self._fallback(VIDEO_UNAVAILABLE)

    def _fallback(self, reason: str) -> ft.Control:
        return ft.Row(
            [ReasonChip(reason=reason),
             ft.TextButton(content="Open externally", icon=ft.Icons.OPEN_IN_NEW,
                           on_click=lambda e: self.actions.open_external(self.item.path) if self.actions.open_external
                           else None, key="video-open-external")],
            wrap=True,
            spacing=6,
        )

    def _on_error(self, e: Any = None) -> None:
        self.error_text.value = "Video playback failed. Use ⋯ › Open externally."
        self.error_text.visible = True
        try:
            self.update()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------


class AudioHub:
    """One ``flet_audio.Audio`` service for the chat; the cards of the playing file receive its events.

    Flet services are registered on the page when constructed, so the chat keeps a single one
    (created on the first Play) instead of one per AudioCard. Playback belongs to a file
    (``active_path``): a transcript re-render or the MediaViewer builds new AudioCards for it,
    and each one ``attach``es - it adopts the state, position, duration and volume and from then
    on follows the events like the card that started the playback.
    """

    def __init__(self) -> None:
        self.audio: Any = None
        self.active: Optional["AudioCard"] = None  # the newest card of the playing file
        self.active_path: Optional[str] = None
        self.state = "stopped"
        self.position_ms = 0
        self.duration_ms = 0
        self.volume = AUDIO_DEFAULT_VOLUME
        self.error: Optional[str] = None
        self._followers: list = []  # weak references to every live card of the playing file

    @staticmethod
    def _path_key(path: str) -> str:
        return os.path.normcase(os.path.abspath(str(path or "")))

    def is_playing_file(self, card: "AudioCard") -> bool:
        return self.active_path is not None and self._path_key(card.item.path) == self.active_path

    def _cards(self) -> list:
        cards = []
        for ref in list(self._followers):
            card = ref()
            if card is None:
                self._followers.remove(ref)
            elif all(card is not c for c in cards):
                cards.append(card)
        return cards

    def _each(self, apply: Callable[["AudioCard"], Any]) -> None:
        """``apply`` to every card of the playing file; a card that cannot take it stops following
        (one stale card must not keep the others, or the playback itself, from updating)."""
        for card in self._cards():
            try:
                apply(card)
            except Exception:
                log.debug("audio card dropped from the playback", exc_info=True)
                self._followers = [ref for ref in self._followers if ref() is not card]
                if self.active is card:
                    self.active = None

    def attach(self, card: "AudioCard") -> None:
        """A card built for the playing file adopts the playback and follows its events."""
        if not self.is_playing_file(card):
            return
        if all(ref() is not card for ref in self._followers):
            self._followers.append(weakref.ref(card))
        self.active = card
        card.adopt(self.state, self.position_ms, self.duration_ms, self.volume)

    def _ensure(self) -> Any:
        if self.audio is not None:
            return self.audio
        import flet_audio as fa

        self.audio = fa.Audio(
            src="",
            autoplay=False,
            volume=AUDIO_DEFAULT_VOLUME,
            release_mode=fa.ReleaseMode.STOP,
            on_duration_change=self._on_duration,
            on_position_change=self._on_position,
            on_state_change=self._on_state,
        )
        return self.audio

    def _update(self) -> None:
        try:
            self.audio.update()
        except Exception:
            pass

    async def toggle(self, card: "AudioCard") -> str:
        """Play / pause ``card``; another card's playback is paused first. Returns the new state."""
        try:
            audio = self.audio = self._ensure()
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            card.set_unavailable()
            return "unavailable"
        try:
            if self.is_playing_file(card) and self.state == "playing":
                self.attach(card)
                await audio.pause()
                self._set_state("paused")
            elif self.is_playing_file(card) and self.state == "paused":
                self.attach(card)
                await audio.resume()
                self._set_state("playing")
            else:
                if self.active_path is not None and not self.is_playing_file(card):
                    try:
                        await audio.pause()
                    except Exception:
                        pass
                    self._each(lambda other: other.on_state("stopped"))
                self.active_path = self._path_key(card.item.path)
                self._followers = [weakref.ref(card)]
                self.active = card
                self.position_ms = card.position_ms
                self.duration_ms = card.duration_ms
                self.volume = card.volume
                audio.src = card.item.path
                audio.volume = card.volume
                self._update()
                await audio.play(card.position_ms)
                self._set_state("playing")
        except Exception as exc:
            log.info("audio playback failed: %s", exc)
            self.error = f"{type(exc).__name__}: {exc}"
            try:
                card.set_unavailable()
            except Exception:
                log.debug("audio card could not show the failure", exc_info=True)
            return "unavailable"
        return self.state

    async def seek(self, card: "AudioCard", milliseconds: int) -> None:
        card.position_ms = int(milliseconds)
        if self.is_playing_file(card) and self.audio is not None:
            self.attach(card)
            self.position_ms = int(milliseconds)
            try:
                await self.audio.seek(int(milliseconds))
            except Exception:
                pass

    def set_volume(self, card: "AudioCard", volume: float) -> None:
        card.volume = max(0.0, min(1.0, float(volume)))
        if self.is_playing_file(card) and self.audio is not None:
            self.volume = card.volume
            self.audio.volume = card.volume
            self._update()

    def _set_state(self, state: str) -> None:
        self.state = state
        self._each(lambda card: card.on_state(state))

    # events (the loop delivers them; every card of the playing file follows)
    def _on_duration(self, e: Any) -> None:
        if self.active_path is None:
            return
        duration = getattr(e, "duration", None)
        ms = getattr(duration, "in_milliseconds", None)
        if ms is None:
            try:
                ms = int(duration.total_seconds() * 1000)  # datetime.timedelta
            except Exception:
                ms = int(duration or 0)
        self.duration_ms = int(ms or 0)
        self._each(lambda card: card.on_duration(self.duration_ms))

    def _on_position(self, e: Any) -> None:
        if self.active_path is None:
            return
        self.position_ms = int(getattr(e, "position", 0) or 0)
        self._each(lambda card: card.on_position(self.position_ms))

    def _on_state(self, e: Any) -> None:
        value = getattr(getattr(e, "state", None), "value", None) or str(getattr(e, "state", "") or "")
        state = {"playing": "playing", "paused": "paused", "completed": "stopped", "stopped": "stopped",
                 "disposed": "stopped"}.get(str(value).lower(), self.state)
        if state == "stopped":
            self.position_ms = 0
            for card in self._cards():
                card.position_ms = 0
        self._set_state(state)


class AudioCard(ft.Container):
    """"🔊 Generated audio": play/pause · seek · "m:ss / m:ss" · volume (75 %) · ⋯."""

    def __init__(self, item: MediaItem, *, hub: Optional[AudioHub] = None, actions: Optional[MediaActions] = None,
                 spawn: Optional[Callable[[Any], Any]] = None, key: Any = None) -> None:
        super().__init__(key=key)
        self.item = item
        self.hub = hub or AudioHub()
        self.actions = actions or MediaActions()
        self.spawn = spawn
        self.volume = AUDIO_DEFAULT_VOLUME
        self.position_ms = 0
        self.duration_ms = 0
        self.state = "stopped"
        self.play_button = ft.IconButton(icon=ft.Icons.PLAY_ARROW, tooltip="Play", size_constraints=HIT_TARGET,
                                         on_click=self._toggle, disabled=not item.exists, key="audio-play")
        self.seek = ft.Slider(min=0, max=1, value=0, expand=True, on_change_end=self._seek, disabled=not item.exists,
                              key="audio-seek")
        self.time_text = ft.Text("0:00 / 0:00", theme_style=ft.TextThemeStyle.LABEL_SMALL, color=_MUTED)
        self.volume_slider = ft.Slider(min=0, max=1, value=self.volume, visible=False, on_change=self._volume,
                                       key="audio-volume")
        self.volume_button = ft.IconButton(icon=ft.Icons.VOLUME_UP, tooltip="Volume", size_constraints=HIT_TARGET,
                                           on_click=self._toggle_volume, key="audio-volume-button")
        self.unavailable = ft.Column(
            [ReasonChip(reason=AUDIO_UNAVAILABLE),
             ft.Row([ft.TextButton(content="Share", icon=ft.Icons.IOS_SHARE,
                                   on_click=lambda e: self.actions.share(item.path) if self.actions.share else None),
                     ft.TextButton(content="Open externally", icon=ft.Icons.OPEN_IN_NEW,
                                   on_click=lambda e: self.actions.open_external(item.path)
                                   if self.actions.open_external else None, key="audio-open-external")],
                    wrap=True, spacing=4)],
            spacing=4, tight=True, visible=False,
        )
        header = ft.Row(
            [
                ft.Text("🔊 Generated audio", theme_style=ft.TextThemeStyle.LABEL_LARGE, weight=ft.FontWeight.W_600),
                ft.Text(item.name, theme_style=ft.TextThemeStyle.LABEL_SMALL, color=_MUTED, expand=True, max_lines=1,
                        overflow=ft.TextOverflow.ELLIPSIS),
                ft.IconButton(icon=ft.Icons.MORE_HORIZ, tooltip="More", size_constraints=HIT_TARGET, icon_size=18,
                              on_click=lambda e: self.actions.show_menu(self.item), key="audio-more"),
            ],
            spacing=8,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )
        controls = [header]
        if item.exists:
            controls += [
                ft.Row([self.play_button, self.seek, self.time_text, self.volume_button], spacing=4,
                       vertical_alignment=ft.CrossAxisAlignment.CENTER),
                self.volume_slider,
                self.unavailable,
            ]
        else:
            controls.append(_missing("audio", item))
        self.content = ft.Column(controls, spacing=4, tight=True)
        self.border_radius = 12
        self.padding = ft.Padding.all(8)
        self.bgcolor = ft.Colors.SURFACE_CONTAINER_LOW
        if item.exists:
            self.hub.attach(self)  # built while its file plays (a re-render, the viewer): follow it

    def adopt(self, state: str, position_ms: int, duration_ms: int, volume: float) -> None:
        """Take over the hub's playback state (no update: the card may not be on the page yet)."""
        self.state = state
        self.position_ms = max(0, int(position_ms or 0))
        self.duration_ms = max(0, int(duration_ms or 0))
        self.volume = float(volume)
        playing = state == "playing"
        self.play_button.icon = ft.Icons.PAUSE if playing else ft.Icons.PLAY_ARROW
        self.play_button.tooltip = "Pause" if playing else "Play"
        self.seek.max = max(1, self.duration_ms)
        self.seek.value = min(float(self.position_ms), float(self.seek.max))
        self.volume_slider.value = self.volume
        self._refresh_time()

    # ---- user events -------------------------------------------------------------------------

    def _run(self, coro: Any) -> Any:
        if self.spawn is not None:
            return self.spawn(coro)
        import asyncio

        try:
            return asyncio.ensure_future(coro)
        except RuntimeError:
            coro.close()
            return None

    def _toggle(self, e: Any = None) -> Any:
        return self._run(self.hub.toggle(self))

    def _seek(self, e: Any = None) -> Any:
        value = float(getattr(getattr(e, "control", None), "value", self.seek.value) or 0)
        return self._run(self.hub.seek(self, int(value)))

    def _volume(self, e: Any = None) -> None:
        value = float(getattr(getattr(e, "control", None), "value", self.volume_slider.value) or 0)
        self.hub.set_volume(self, value)
        self.volume_button.icon = ft.Icons.VOLUME_OFF if value <= 0 else ft.Icons.VOLUME_UP
        self._push()

    def _toggle_volume(self, e: Any = None) -> None:
        self.volume_slider.visible = not self.volume_slider.visible
        self._push()

    # ---- hub events ----------------------------------------------------------------------------

    def on_state(self, state: str) -> None:
        self.state = state
        playing = state == "playing"
        self.play_button.icon = ft.Icons.PAUSE if playing else ft.Icons.PLAY_ARROW
        self.play_button.tooltip = "Pause" if playing else "Play"
        if state == "stopped":
            self.seek.value = 0
        self._refresh_time()
        self._push()

    def on_duration(self, milliseconds: int) -> None:
        self.duration_ms = max(0, int(milliseconds))
        self.seek.max = max(1, self.duration_ms)
        self._refresh_time()
        self._push()

    def on_position(self, milliseconds: int) -> None:
        self.position_ms = max(0, int(milliseconds))
        self.seek.value = min(float(self.position_ms), float(self.seek.max or 1))
        self._refresh_time()
        self._push()

    def set_unavailable(self) -> None:
        self.unavailable.visible = True
        self.play_button.disabled = True
        self.seek.disabled = True
        self._push()

    def _refresh_time(self) -> None:
        self.time_text.value = f"{format_clock(self.position_ms)} / {format_clock(self.duration_ms)}"

    def _push(self) -> None:
        try:
            self.update()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Section builder and the Refine compare sheet
# ---------------------------------------------------------------------------


def media_section(items: Sequence[MediaItem], *, actions: MediaActions, hub: Optional[AudioHub] = None,
                  available_width: Optional[float] = None, spawn: Optional[Callable[[Any], Any]] = None) -> Optional[ft.Control]:
    """The media block of one assistant card (images first, then one video / audio card each)."""
    if not items:
        return None
    controls: list = []
    images = [i for i in items if i.kind == "image"]
    if images:
        controls.append(ImageGallery(images, actions=actions, available_width=available_width, key="media-images"))
    for n, item in enumerate(i for i in items if i.kind == "video"):
        controls.append(VideoCard(item, actions=actions, available_width=available_width, key=f"media-video-{n}"))
    for n, item in enumerate(i for i in items if i.kind == "audio"):
        controls.append(AudioCard(item, hub=hub, actions=actions, spawn=spawn, key=f"media-audio-{n}"))
    return ft.Column(controls, spacing=8, tight=True, key="media-section")


_BLOCK_COLOURS = {
    "replace": (ft.Colors.ERROR_CONTAINER, ft.Colors.TERTIARY_CONTAINER),
    "delete": (ft.Colors.ERROR_CONTAINER, None),
    "insert": (None, ft.Colors.TERTIARY_CONTAINER),
}


class CompareSheet:
    """"Compare with original": each refined paragraph stacked under the original it replaced."""

    def __init__(self, original: str, refined: str, *, title: str = "Compare with original") -> None:
        from glossarion_mobile.ui.chat.direct_text_rules import display_markdown

        self.blocks = compare_blocks(display_markdown(original), display_markdown(refined))
        changed = sum(1 for tag, _o, _r in self.blocks if tag != "equal")
        rows: list = [
            ft.Text(title, theme_style=ft.TextThemeStyle.TITLE_MEDIUM, weight=ft.FontWeight.W_600),
            ft.Text(f"{changed} changed paragraph{'s' if changed != 1 else ''} · original above, refined below",
                    theme_style=ft.TextThemeStyle.LABEL_SMALL, color=_MUTED),
        ]
        for tag, old, new in self.blocks:
            if tag == "equal":
                rows.append(ft.Text(new, theme_style=ft.TextThemeStyle.BODY_MEDIUM, color=_MUTED, selectable=True))
                continue
            old_bg, new_bg = _BLOCK_COLOURS.get(tag, (None, None))
            if old:
                rows.append(ft.Container(content=ft.Text(old, theme_style=ft.TextThemeStyle.BODY_MEDIUM, selectable=True),
                                         bgcolor=old_bg, border_radius=6, padding=ft.Padding.all(6),
                                         tooltip="Original"))
            if new:
                rows.append(ft.Container(content=ft.Text(new, theme_style=ft.TextThemeStyle.BODY_MEDIUM, selectable=True),
                                         bgcolor=new_bg, border_radius=6, padding=ft.Padding.all(6),
                                         tooltip="Refined"))
        self.dialog = ft.BottomSheet(
            content=ft.Container(padding=ft.Padding.only(left=16, right=16, bottom=24),
                                 content=ft.Column(rows, spacing=8, tight=True, scroll=ft.ScrollMode.AUTO)),
            show_drag_handle=True,
            scrollable=True,
            draggable=True,
            bgcolor=ft.Colors.SURFACE_CONTAINER_HIGH,
        )

    def show(self, page: Any) -> None:
        page.show_dialog(self.dialog)
