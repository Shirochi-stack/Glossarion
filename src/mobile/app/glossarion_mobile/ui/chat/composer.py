"""Composer (UI_SPEC §2.3, §5.4): the radius-24 tonal card at the bottom of the chat.

Rows, top to bottom (desktop Direct Text order): chips row (attachment /
pasted-text chips; hidden when empty) · auto-growing TextField (1-6 lines) ·
action row (＋ · output mode · option pills · token hint · Send/Stop). The output
mode (``OutputModeRow(inline=True)``) has no line of its own (owner request, so
the text field keeps that line): a chip with a menu on phones, the six toggles
from 600 dp, labelled toggles once they fit (``responsive.output_row_style``).
The token hint hides at >= 160% text scale (§7.5) so the row never overflows.
``StatusCaption`` is the one-line caption shown directly above the
composer whenever the Send state is not Ready, with the fix buttons of §2.4.

Paste-to-chip: when one ``on_change`` adds more than 10,000 characters the
pasted part moves into a "Pasted text · N chars" chip and the field keeps what
was typed before it.

U3: one attachment chip per turn (desktop parity; × removes it, the hint switches
to "Add optional instructions for the attached file…"), draft autosave through
``on_draft_changed`` (the store debounces the save by 450 ms), option pills that
differ from the defaults (tap -> sheet, × -> reset) and the token hint.

U9: slash commands (§2.7, ``slash``): "/" at the start of the field shows the command popover
(``self.slash.control``, which the chat view places directly above the composer card); a tap
inserts a command that takes an argument, or runs it (``on_slash``), and Send / Enter on a
complete command runs it instead of sending the text.
"""

from __future__ import annotations

import math
from dataclasses import field
from typing import Any, Callable, Optional

import flet as ft

from glossarion_mobile.state.store import Signal
from glossarion_mobile.ui import motion, tokens
from glossarion_mobile.ui.chat.direct_text_rules import attachment_icon, attachment_kind_label, format_attachment_size
from glossarion_mobile.ui.chat.output_mode_row import OutputModeRow
from glossarion_mobile.ui.chat.send_button import SendStopButton
from glossarion_mobile.ui.chat.slash import SlashCommand, SlashPopover, completion, parse_command
from glossarion_mobile.ui.chat.send_state import BlockReason, SendAction, SendInputs, SendState
from glossarion_mobile.ui.responsive import ACTION_ROW_GAP
from glossarion_mobile.ui.theme import HIT_TARGET, icon_data

__all__ = [
    "AttachmentChip",
    "Composer",
    "PASTE_CHIP_THRESHOLD",
    "PastedTextChip",
    "StatusCaption",
    "HINT_EMPTY",
    "HINT_ATTACHMENT",
]

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
class AttachmentChip(ft.Container):
    """FileChip for the composer: type icon · name · "EPUB · 1.2 MB" · × (UI_SPEC §2.3, §5.3)."""

    record: dict = field(default_factory=dict, metadata={"skip": True})
    missing: bool = field(default=False, metadata={"skip": True})
    on_remove: Optional[Callable[["AttachmentChip"], Any]] = field(default=None, metadata={"skip": True})
    on_open: Optional[Callable[["AttachmentChip"], Any]] = field(default=None, metadata={"skip": True})

    def init(self) -> None:
        super().init()
        extension = str(self.record.get("extension") or "")
        self.name_text = ft.Text(
            str(self.record.get("name") or "file"), theme_style=ft.TextThemeStyle.LABEL_MEDIUM, no_wrap=True,
            max_lines=1, overflow=ft.TextOverflow.ELLIPSIS,
        )
        meta = f"{attachment_kind_label(extension)} · {format_attachment_size(self.record.get('size'))}"
        self.meta_text = ft.Text(
            "Attachment missing" if self.missing else meta,
            theme_style=ft.TextThemeStyle.LABEL_SMALL,
            color=ft.Colors.ERROR if self.missing else ft.Colors.with_opacity(0.6, ft.Colors.ON_SURFACE),
        )
        self.remove_button = ft.IconButton(
            icon=ft.Icons.CLOSE, icon_size=16, tooltip="Remove attachment", on_click=self._remove, size_constraints=HIT_TARGET
        )
        self.content = ft.Row(
            [
                ft.Icon(icon_data(attachment_icon(extension)), size=20, color=ft.Colors.PRIMARY),
                ft.Column([self.name_text, self.meta_text], spacing=0, tight=True),
                self.remove_button,
            ],
            spacing=6,
            tight=True,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )
        self.padding = ft.Padding.only(left=8)
        self.border_radius = tokens.RADII["chip"]
        self.bgcolor = ft.Colors.ERROR_CONTAINER if self.missing else ft.Colors.SURFACE_CONTAINER_HIGHEST
        self.tooltip = str(self.record.get("path") or "")
        self.on_click = self._open

    def _remove(self, e: Any = None) -> None:
        if self.on_remove is not None:
            self.on_remove(self)

    def _open(self, e: Any = None) -> None:
        if self.on_open is not None:
            self.on_open(self)


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
            # no expand: Flutter's Wrap (Row(wrap=True)) rejects Expanded children and the
            # whole chat column then renders as an error box; the text wraps at the row width.
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
    row_style: str = field(default="chip", metadata={"skip": True})  # responsive.output_row_style
    on_plus: Optional[Callable[[], Any]] = field(default=None, metadata={"skip": True})
    on_plus_long_press: Optional[Callable[[], Any]] = field(default=None, metadata={"skip": True})
    on_send_action: Optional[Callable[[SendAction], Any]] = field(default=None, metadata={"skip": True})
    on_content_changed: Optional[Callable[[bool], Any]] = field(default=None, metadata={"skip": True})
    on_expand: Optional[Callable[[], Any]] = field(default=None, metadata={"skip": True})
    on_open_mode_options: Optional[Callable[[str], Any]] = field(default=None, metadata={"skip": True})
    on_draft_changed: Optional[Callable[[str], Any]] = field(default=None, metadata={"skip": True})
    on_attachment_removed: Optional[Callable[[], Any]] = field(default=None, metadata={"skip": True})
    on_attachment_open: Optional[Callable[[dict], Any]] = field(default=None, metadata={"skip": True})
    on_pill: Optional[Callable[[str], Any]] = field(default=None, metadata={"skip": True})
    on_pill_reset: Optional[Callable[[str], Any]] = field(default=None, metadata={"skip": True})
    on_slash: Optional[Callable[[str], Any]] = field(default=None, metadata={"skip": True})

    def init(self) -> None:
        super().init()
        self._previous_text = ""
        self._compact_text = False
        self.pills: list = []  # [(id, label)] of the options that differ from the defaults
        self.has_attachment = False
        self.attachment: Optional[dict] = None
        self.attachment_chip: Optional[AttachmentChip] = None
        self.chips_row = ft.Row([], scroll=ft.ScrollMode.AUTO, spacing=6, visible=False)
        self.slash = SlashPopover(on_pick=self._on_slash_pick)
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
            inline=True,
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
            animate_rotation=motion.rotation_animation(tokens.MOTION["sheet_ms"]),
        )
        self.pills_row = ft.Row([], spacing=6, scroll=ft.ScrollMode.AUTO, visible=False)
        self.token_hint = ft.Text("", theme_style=ft.TextThemeStyle.LABEL_SMALL, visible=False, no_wrap=True)
        self.send_button = SendStopButton(on_action=self._on_send_action)
        # ＋ · output mode · option pills (take the rest, scroll) · token hint · Send; one line, never
        # wrapping (responsive.action_row_width estimates it).
        self.action_row = ft.Row(
            [
                self.plus_button,
                self.output_row,
                ft.Container(content=self.pills_row, expand=True),
                self.token_hint,
                self.send_button,
            ],
            spacing=ACTION_ROW_GAP,
            height=tokens.SIZES["composer_row"] + 8,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )
        self.content = ft.Column(
            [
                self.chips_row,
                ft.Row([self.text_field, self.expand_button], vertical_alignment=ft.CrossAxisAlignment.START, spacing=0),
                self.action_row,
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
        self._update_slash(value)
        if self.on_draft_changed is not None:
            self.on_draft_changed(value)
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

    @property
    def pasted_text(self) -> str:
        """Text of the pasted-text chips (sent as the ``["user", text]`` content)."""
        return "\n".join(chip.text for chip in self.pasted_chips)

    def send_text(self) -> str:
        """Field text plus pasted chips, as the desktop reads ``toPlainText().strip()``."""
        parts = [part for part in (self.text, self.pasted_text) if part]
        return "\n".join(parts).strip()

    def set_text(self, value: str) -> None:
        """Programmatic set (draft restore) without paste-to-chip and without a draft write."""
        self._previous_text = value or ""
        self.text_field.value = value or ""
        self.expand_button.visible = self._line_estimate(self._previous_text) >= EXPAND_ICON_LINES
        self._update_slash(self._previous_text)
        self._content_changed()

    # ---- attachment (one per turn) ------------------------------------------------------

    def set_attachment(self, record: Optional[dict], *, missing: bool = False) -> None:
        if self.attachment_chip is not None and self.attachment_chip in self.chips_row.controls:
            self.chips_row.controls.remove(self.attachment_chip)
        self.attachment_chip = None
        self.attachment = dict(record) if record else None
        if record:
            self.attachment_chip = AttachmentChip(
                record=dict(record), missing=missing, on_remove=self._remove_attachment, on_open=self._open_attachment
            )
            self.chips_row.controls.insert(0, self.attachment_chip)
        self.chips_row.visible = bool(self.chips_row.controls)
        self.set_attachment_hint(bool(record))
        self._content_changed()

    def _remove_attachment(self, chip: Any = None) -> None:
        self.set_attachment(None)
        if self.on_attachment_removed is not None:
            self.on_attachment_removed()

    def _open_attachment(self, chip: Any = None) -> None:
        if self.on_attachment_open is not None and self.attachment:
            self.on_attachment_open(dict(self.attachment))

    # ---- option pills and token hint -----------------------------------------------------

    def set_pills(self, pills: list) -> None:
        """``[(id, label)]`` of options that differ from the defaults (§2.3 action row). At >= 160 %
        text they collapse into one "Options (n)" chip that opens Chat settings (§2.3, §7.5)."""
        self.pills = list(pills)
        self._render_pills()

    def _render_pills(self) -> None:
        pills = self.pills
        if self._compact_text and pills:
            count = len(pills)
            self.pills_row.controls = [
                ft.Chip(
                    label=ft.Text(f"Options ({count})", theme_style=ft.TextThemeStyle.LABEL_MEDIUM),
                    leading=ft.Icon(ft.Icons.TUNE, size=16),
                    visual_density=ft.VisualDensity.COMPACT,
                    tooltip=", ".join(label for _pill_id, label in pills),
                    on_click=lambda e: self.on_pill("options") if self.on_pill else None,
                    key="pill-options",
                )
            ]
        else:
            self.pills_row.controls = [
                ft.Chip(
                    label=ft.Text(label),
                    on_click=lambda e, p=pill_id: self.on_pill(p) if self.on_pill else None,
                    on_delete=lambda e, p=pill_id: self.on_pill_reset(p) if self.on_pill_reset else None,
                    key=f"pill-{pill_id}",
                )
                for pill_id, label in pills
            ]
        self.pills_row.visible = bool(pills)

    def set_token_hint(self, text: str) -> None:
        """"≈1.2k tok"; hidden at >= 160% text scale (§7.5), where it would crowd the action row."""
        self.token_hint.value = text
        self.token_hint.visible = bool(text) and not self._compact_text

    def clear(self) -> None:
        self.text_field.value = ""
        self._previous_text = ""
        self.slash.hide()
        self.chips_row.controls.clear()
        self.chips_row.visible = False
        self.expand_button.visible = False
        self.attachment = None
        self.attachment_chip = None
        self.set_attachment_hint(False)
        self.set_token_hint("")
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
        """The output-mode control for a layout (``Layout.output_row``: chip / icons / full)."""
        changed = self.output_row.set_style(style_name)
        self.row_style = style_name
        return changed

    def set_text_scale(self, scale: float) -> None:
        """The field's own size (an explicit 15 sp style, so the theme's scale does not reach it):
        Appearance text size × the chat's Text size (UI_SPEC §2.14)."""
        try:
            value = max(0.5, float(scale or 1.0))
        except (TypeError, ValueError):
            value = 1.0
        self.text_field.text_style = ft.TextStyle(size=round(tokens.TYPE_SCALE["body_large"].size * value, 2))

    def set_compact_text(self, compact: bool) -> None:
        """>= 160% text scale: max 4 lines, no token hint, the pills as "Options (n)" (§7.5)."""
        changed = bool(compact) != self._compact_text
        self._compact_text = bool(compact)
        self.text_field.max_lines = 4 if compact else 6
        self.set_token_hint(self.token_hint.value or "")
        if changed:
            self._render_pills()

    def set_plus_open(self, is_open: bool) -> None:
        """＋ rotates 45° into × while the ＋ sheet is open; under reduce motion it turns at once
        (no rotation animation, UI_SPEC §6.3)."""
        self.plus_button.animate_rotation = motion.rotation_animation(tokens.MOTION["sheet_ms"])
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
        if action in (SendAction.SEND, SendAction.QUEUE) and self.run_slash_text():
            return
        if self.on_send_action is not None:
            self.on_send_action(action)

    # ---- slash commands (UI_SPEC §2.7) ---------------------------------------------------------

    def _slash_eligible(self) -> bool:
        return self.on_slash is not None and not self.pasted_chips

    def _update_slash(self, value: str) -> None:
        if self._slash_eligible():
            self.slash.update_for(value)
        elif self.slash.visible:
            self.slash.hide()

    def run_slash_text(self) -> bool:
        """Run the field's text as a command when it is a complete one (Send / Enter); False otherwise."""
        if not self._slash_eligible() or parse_command(self.text) is None:
            return False
        text = self.text.strip()
        self.set_text("")
        if self.on_draft_changed is not None:
            self.on_draft_changed("")
        self.on_slash(text)
        return True

    def _on_slash_pick(self, command: SlashCommand) -> None:
        """A popover tap: a command that takes an argument goes into the field (with the typed
        argument kept); one without runs."""
        parsed = parse_command(self.text)
        has_arg = parsed is not None and parsed[0] == command and bool(parsed[1])
        if command.arg and not has_arg:
            self.set_text(completion(command))
            try:
                self.text_field.focus()
            except Exception:
                pass
            return
        if not has_arg:
            self.set_text(f"/{command.name}")
        self.run_slash_text()

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
