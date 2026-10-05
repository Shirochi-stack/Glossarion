"""ModeOptionsSheet skeleton (UI_SPEC §2.6): opened by tapping the active output mode.

Title "Output: <mode>", the "This chat only" switch and, per mode, what its
options will be. The schema-bound option tiles arrive with the settings
schema (U2) and the pipelines (U3 Vision/Refine, U7 Image/Video/Audio); until
then each mode shows a ReasonChip instead of hiding anything.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import flet as ft

from glossarion_mobile.ui.chat.output_modes import mode_label, normalize_mode, output_mode
from glossarion_mobile.ui.components.reason_chip import ReasonChip
from glossarion_mobile.ui.theme import icon_data

__all__ = ["MODE_HINTS", "MODE_MILESTONES", "ModeOptionsSheet"]

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
MODE_MILESTONES = {"text": None, "vision": "U3", "refinement": "U3", "image": "U7", "video": "U7", "audio": "U7"}
GENERATIVE = ("image", "video", "audio")


class ModeOptionsSheet:
    def __init__(self, mode: str, *, on_dismiss: Optional[Callable[..., Any]] = None) -> None:
        self.mode = normalize_mode(mode)
        info = output_mode(self.mode)
        self._page: Any = None
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

    def show(self, page: Any) -> None:
        self._page = page
        page.show_dialog(self.dialog)
