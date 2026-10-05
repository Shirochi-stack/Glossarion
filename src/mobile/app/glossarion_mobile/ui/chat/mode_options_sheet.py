"""ModeOptionsSheet (UI_SPEC §2.6): opened by tapping the active output mode.

Title "Output: <mode>", the "This chat only" switch (it scopes the mode itself) and the
mode's options. Vision and Refine render their schema-bound settings tiles (the shared
settings tiles of Settings, bound to the global config keys, as on the desktop main window /
Other Settings); Image / Video / Audio show a ReasonChip until their pipelines arrive (U7)
instead of hiding anything.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import flet as ft

from glossarion_mobile.ui.chat.output_modes import mode_label, normalize_mode, output_mode
from glossarion_mobile.ui.components.reason_chip import ReasonChip
from glossarion_mobile.ui.theme import icon_data

__all__ = ["MODE_HINTS", "MODE_MILESTONES", "MODE_OPTION_KEYS", "ModeOptionsSheet"]

MODE_HINTS = {
    "text": "Translate text, documents, subtitles and books",
    "vision": "Vision OCR prompt · Skip translation (OCR only) · Batch Vision API requests · Keep OCR image · "
    "Process long images · Hide labels · Vision keys",
    "image": "Output resolution 1K / 2K / 4K · Batch requests · Image keys · Custom image-edit endpoint",
    "video": "Duration 5 / 10 / 15 / 20 / 30 / 60 s · Resolution 360p / 480p / 720p / 1080p",
    "audio": "TTS voice · TTS keys",
    "refinement": "Refinement mode (Full / Full + raw / Failed / Partial / Partial.b / Partial.b2) · "
    "Raw prompt role · Refine prompt",
}
MODE_MILESTONES = {"text": None, "vision": None, "refinement": None, "image": "U7", "video": "U7", "audio": "U7"}
#: Schema keys rendered as settings tiles per mode (UI_SPEC §2.6 "Options (bound schema keys)").
MODE_OPTION_KEYS = {
    "vision": (
        "vision_ocr_prompt", "vision_ocr_user_prompt", "vision_ocr_skip_translation",
        "vision_ocr_batch_translation", "vision_ocr_batch_size", "vision_ocr_keep_images",
        "process_webnovel_images", "hide_image_translation_label",
    ),
    "refinement": (
        "multipass_refinement_mode", "refinement_full_with_raw_raw_role",
        "refinement_system_prompt", "refinement_user_prompt",
    ),
}
GENERATIVE = ("image", "video", "audio")


class ModeOptionsSheet:
    def __init__(self, mode: str, *, on_dismiss: Optional[Callable[..., Any]] = None, ctx: Any = None) -> None:
        self.mode = normalize_mode(mode)
        info = output_mode(self.mode)
        self._page: Any = None
        self.ctx = ctx  # SettingsContext for the option tiles (None: the options show a reason chip)
        self.tiles: dict = {}
        self.this_chat_switch = ft.Switch(label="This chat only", value=False)
        controls: list[ft.Control] = [
            ft.Row(
                [
                    ft.Icon(icon_data(info.icon)),
                    ft.Text(mode_label(self.mode), theme_style=ft.TextThemeStyle.TITLE_LARGE, weight=ft.FontWeight.W_600),
                ],
                spacing=8,
            ),
            self.this_chat_switch,
            ft.Text(MODE_HINTS[self.mode], theme_style=ft.TextThemeStyle.BODY_MEDIUM),
        ]
        milestone = MODE_MILESTONES.get(self.mode)
        if milestone:
            controls.append(ReasonChip(reason=f"Options arrive in {milestone}"))
        keys = MODE_OPTION_KEYS.get(self.mode, ())
        if keys:
            controls.extend(self._option_tiles(keys))
        if self.mode in GENERATIVE:
            self.generate_button = ft.FilledTonalButton(content="Generate from prompt (no input)", disabled=True)
            controls.append(
                ft.Row([self.generate_button, ReasonChip(reason="Type a prompt in the composer first")], wrap=True, spacing=8)
            )
        self.dialog = ft.BottomSheet(
            content=ft.Container(
                padding=ft.Padding.only(left=16, right=16, bottom=24),
                content=ft.Column(controls, tight=True, spacing=12),
            ),
            show_drag_handle=True,
            scrollable=True,
            on_dismiss=on_dismiss,
            bgcolor=ft.Colors.SURFACE_CONTAINER_HIGH,
        )

    def _option_tiles(self, keys: tuple) -> list:
        """The shared settings tiles of ``keys`` (each skipped when the schema lacks it)."""
        ctx = self.ctx
        if ctx is None:
            return [ReasonChip(reason="These options are in Settings (settings unavailable in this session)")]
        from glossarion_mobile.ui.settings.tiles import make_tile

        out = []
        for key in keys:
            try:
                spec = ctx.schema.spec(key)
            except Exception:
                spec = None
            if spec is None:
                continue
            try:
                tile = make_tile(spec, ctx)
            except Exception:
                continue
            self.tiles[key] = tile
            out.append(tile.control)
        return out

    def show(self, page: Any) -> None:
        self._page = page
        page.show_dialog(self.dialog)
