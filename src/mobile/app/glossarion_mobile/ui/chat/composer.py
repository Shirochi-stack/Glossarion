"""Composer (UI_SPEC §2.3, §5.4): the radius-24 tonal card at the bottom of the chat.

Rows, top to bottom (desktop Direct Text order): chips row (attachment /
pasted-text chips; hidden when empty) · auto-growing TextField (1-6 lines) ·
the always-visible OutputModeRow · action row (＋ · option pills · token hint ·
Send/Stop). ``StatusCaption`` is the one-line caption shown directly above the
composer whenever the Send state is not Ready, with the fix buttons of §2.4.

Paste-to-chip: when one ``on_change`` adds more than 10,000 characters the
pasted part moves into a "Pasted text · N chars" chip and the field keeps what
was typed before it.

U1 skeleton: no attachments, drafts or token counting yet (U3).
"""

from __future__ import annotations

import math
from dataclasses import field
from typing import Any, Callable, Optional

import flet as ft

from glossarion_mobile.state.store import Signal
from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.chat.output_mode_row import OutputModeRow
from glossarion_mobile.ui.chat.send_button import SendStopButton
from glossarion_mobile.ui.chat.send_state import BlockReason, SendAction, SendInputs, SendState
from glossarion_mobile.ui.theme import HIT_TARGET

__all__ = ["Composer", "PASTE_CHIP_THRESHOLD", "PastedTextChip", "StatusCaption", "HINT_EMPTY", "HINT_ATTACHMENT"]

PASTE_CHIP_THRESHOLD = 10_000
EXPAND_ICON_LINES = 3
HINT_EMPTY = "Message to translate…"  # desktop placeholder without the drop hint
HINT_ATTACHMENT = "Add optional instructions for the attached file…"  # desktop string


@ft.control
class PastedTextChip(ft.Container):
    text: str = field(default="", metadata={"skip": True})
    on_remove: Optional[Callable[["PastedTextChip"], Any]] = field(default=None, metadata={"skip": True})

    def init(self) -> None:
        super().init()
        self.label = ft.Text(
            f"Pasted text · {len(self.text):,} chars", theme_style=ft.TextThemeStyle.LABEL_MEDIUM, no_wrap=True
        )
        self.remove_button = ft.IconButton(
            icon=ft.Icons.CLOSE,
            icon_size=16,
            tooltip="Remove pasted text",
            on_click=self._remove,
            size_constraints=HIT_TARGET,
        )
        self.content = ft.Row(
            [ft.Icon(ft.Icons.CONTENT_PASTE, size=16), self.label, self.remove_button],
            spacing=4,
            tight=True,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )
        self.padding = ft.Padding.only(left=8)
        self.border_radius = tokens.RADII["chip"]
        self.bgcolor = ft.Colors.SURFACE_CONTAINER_HIGHEST

    def _remove(self, e: Any = None) -> None:
        if self.on_remove is not None:
            self.on_remove(self)


@ft.control
class StatusCaption(ft.Container):
    """One line above the composer; fix buttons for a blocked Send (§2.3, §2.4)."""

    on_fix: Optional[Callable[[str], Any]] = field(default=None, metadata={"skip": True})

    def init(self) -> None:
        super().init()
        self.text = ft.Text(
            "",
            theme_style=ft.TextThemeStyle.LABEL_SMALL,
            color=ft.Colors.with_opacity(0.6, ft.Colors.ON_SURFACE),
            max_lines=2,
            overflow=ft.TextOverflow.ELLIPSIS,
            expand=True,
        )
        self.fix_button = ft.FilledTonalButton(content="", visible=False, on_click=self._on_fix)
        self.secondary_button = ft.TextButton(content="", visible=False, on_click=self._on_secondary)
        self.content = ft.Row(
            [self.text, self.fix_button, self.secondary_button],
            spacing=8,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
            wrap=True,
        )
        self.padding = ft.Padding.symmetric(horizontal=16)
        self.visible = False
        self._block: Optional[BlockReason] = None

    def show(self, caption: Optional[str], block: Optional[BlockReason] = None) -> None:
        self._block = block
        self.visible = bool(caption)
        self.text.value = caption or ""
        self.fix_button.visible = bool(block and block.fix_label)
        self.fix_button.content = (block.fix_label if block and block.fix_label else "")
        self.secondary_button.visible = bool(block and block.secondary_label)
        self.secondary_button.content = (block.secondary_label if block and block.secondary_label else "")

    def _on_fix(self, e: Any = None) -> None:
        if self._block is not None and self._block.fix_action and self.on_fix is not None:
            self.on_fix(self._block.fix_action)

    def _on_secondary(self, e: Any = None) -> None:
        if self._block is not None and self._block.secondary_action and self.on_fix is not None:
            self.on_fix(self._block.secondary_action)


@ft.control
class Composer(ft.Container):
    mode_signal: Optional[Signal] = field(default=None, metadata={"skip": True})
    row_style: str = field(default="label", metadata={"skip": True})
    on_plus: Optional[Callable[[], Any]] = field(default=None, metadata={"skip": True})
    on_plus_long_press: Optional[Callable[[], Any]] = field(default=None, metadata={"skip": True})
    on_send_action: Optional[Callable[[SendAction], Any]] = field(default=None, metadata={"skip": True})
    on_content_changed: Optional[Callable[[bool], Any]] = field(default=None, metadata={"skip": True})
    on_expand: Optional[Callable[[], Any]] = field(default=None, metadata={"skip": True})
    on_open_mode_options: Optional[Callable[[str], Any]] = field(default=None, metadata={"skip": True})

    def init(self) -> None:
        super().init()
        self._previous_text = ""
        self.has_attachment = False
        self.chips_row = ft.Row([], scroll=ft.ScrollMode.AUTO, spacing=6, visible=False)
        self.text_field = ft.TextField(
            multiline=True,
            min_lines=1,
            max_lines=6,
            shift_enter=True,
            border=ft.NoInputBorder(),
            dense=True,
            content_padding=ft.Padding.all(4),
            hint_text=HINT_EMPTY,
            on_change=self._on_text_change,
            on_submit=self._on_submit,
            expand=True,
            text_style=ft.TextStyle(size=tokens.TYPE_SCALE["body_large"].size),
        )
        self.expand_button = ft.IconButton(
            icon=ft.Icons.OPEN_IN_FULL,
            icon_size=20,
            tooltip="Open the full-screen editor",
            visible=False,
            on_click=self._on_expand,
            size_constraints=HIT_TARGET,
        )
        self.output_row = OutputModeRow(
            mode_signal=self.mode_signal,
            style_name=self.row_style,
            on_open_options=self._open_mode_options,
        )
        self.mode_signal = self.output_row.mode_signal
        self.plus_button = ft.IconButton(
            icon=ft.Icons.ADD,
            tooltip="Attach, output mode and tools",
            on_click=self._on_plus,
            on_long_press=self._on_plus_long,
            size_constraints=HIT_TARGET,
            rotate=ft.Rotate(angle=0),
            animate_rotation=tokens.MOTION["sheet_ms"],
        )
        self.pills_row = ft.Row([], spacing=6, scroll=ft.ScrollMode.AUTO, visible=False)
        self.token_hint = ft.Text("", theme_style=ft.TextThemeStyle.LABEL_SMALL, visible=False)
        self.send_button = SendStopButton(on_action=self._on_send_action)
        self.content = ft.Column(
            [
                self.chips_row,
                ft.Row([self.text_field, self.expand_button], vertical_alignment=ft.CrossAxisAlignment.START, spacing=0),
                self.output_row,
                ft.Row(
                    [
                        self.plus_button,
                        ft.Container(content=self.pills_row, expand=True),
                        self.token_hint,
                        self.send_button,
                    ],
                    spacing=4,
                    height=tokens.SIZES["composer_row"] + 8,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
            ],
            spacing=2,
            tight=True,
        )
        self.bgcolor = ft.Colors.SURFACE_CONTAINER_HIGH
        self.border_radius = tokens.RADII["composer"]
        self.padding = ft.Padding.symmetric(vertical=8, horizontal=12)
        self.margin = ft.Margin.only(left=8, right=8, bottom=8)

    # ---- content -------------------------------------------------------------------

    @property
    def text(self) -> str:
        return self.text_field.value or ""

    @property
    def pasted_chips(self) -> list[PastedTextChip]:
        return [c for c in self.chips_row.controls if isinstance(c, PastedTextChip)]

    @property
    def has_content(self) -> bool:
        return bool(self.text.strip()) or bool(self.pasted_chips) or self.has_attachment

    def _line_estimate(self, text: str) -> int:
        return sum(max(1, math.ceil(len(line) / 40)) for line in text.split("\n")) if text else 1

    def _on_text_change(self, e: Any = None) -> None:
        self.handle_text(self.text_field.value or "")

    def handle_text(self, value: str) -> None:
        """Process a new field value (the ``on_change`` path; also used by tests)."""
        previous = self._previous_text
        if len(value) - len(previous) > PASTE_CHIP_THRESHOLD:
            if value.startswith(previous):
                pasted, kept = value[len(previous):], previous
            else:
                pasted, kept = value, ""
            self.text_field.value = kept
            value = kept
            self.add_pasted_text(pasted)
        self._previous_text = value
        self.text_field.value = value
        self.expand_button.visible = self._line_estimate(value) >= EXPAND_ICON_LINES
        self._content_changed()

    def add_pasted_text(self, text: str) -> PastedTextChip:
        chip = PastedTextChip(text=text, on_remove=self._remove_chip)
        self.chips_row.controls.append(chip)
        self.chips_row.visible = True
        return chip

    def _remove_chip(self, chip: Any) -> None:
        if chip in self.chips_row.controls:
            self.chips_row.controls.remove(chip)
        self.chips_row.visible = bool(self.chips_row.controls)
        self._content_changed()

    def clear(self) -> None:
        self.text_field.value = ""
        self._previous_text = ""
        self.chips_row.controls.clear()
        self.chips_row.visible = False
        self.expand_button.visible = False
        self._content_changed()

    def set_attachment_hint(self, attached: bool) -> None:
        self.has_attachment = attached
        self.text_field.hint_text = HINT_ATTACHMENT if attached else HINT_EMPTY

    def _content_changed(self) -> None:
        if self.on_content_changed is not None:
            self.on_content_changed(self.has_content)
        self._push()

    def _push(self) -> None:
        try:
            self.update()
        except Exception:
            pass

    # ---- layout --------------------------------------------------------------------

    def set_row_style(self, style_name: str) -> bool:
        changed = self.output_row.set_style(style_name)
        self.row_style = style_name
        return changed

    def set_compact_text(self, compact: bool) -> None:
        """>= 160% text scale: max 4 lines (§7.5)."""
        self.text_field.max_lines = 4 if compact else 6

    def set_plus_open(self, is_open: bool) -> None:
        """＋ rotates 45° into × while the ＋ sheet is open."""
        self.plus_button.rotate = ft.Rotate(angle=math.pi / 4 if is_open else 0)
        self.plus_button.tooltip = "Close" if is_open else "Attach, output mode and tools"
        self._push()

    # ---- send state -------------------------------------------------------------------

    def apply_send_inputs(self, inputs: SendInputs) -> bool:
        return self.send_button.apply(inputs)

    @property
    def send_state(self) -> SendState:
        return self.send_button.state

    # ---- events ---------------------------------------------------------------------

    def _on_send_action(self, action: SendAction) -> None:
        if self.on_send_action is not None:
            self.on_send_action(action)

    def _on_submit(self, e: Any = None) -> None:
        # Hardware Enter (shift_enter=True): same as tapping the button.
        if self.send_button.state in (SendState.IDLE_READY, SendState.QUEUE, SendState.BLOCKED):
            self.send_button.tap()

    def _on_plus(self, e: Any = None) -> None:
        if self.on_plus is not None:
            self.on_plus()

    def _on_plus_long(self, e: Any = None) -> None:
        if self.on_plus_long_press is not None:
            self.on_plus_long_press()

    def _on_expand(self, e: Any = None) -> None:
        if self.on_expand is not None:
            self.on_expand()

    def _open_mode_options(self, mode_id: str) -> None:
        if self.on_open_mode_options is not None:
            self.on_open_mode_options(mode_id)
