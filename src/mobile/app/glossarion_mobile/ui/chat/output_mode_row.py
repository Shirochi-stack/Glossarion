"""OutputModeRow (UI_SPEC §2.3 item 3, §5.4): the desktop Direct Text "Output:" control.

In the composer (``inline=True``) it sits in the action row right after ＋, so it has no
line of its own (owner request: the old separate row took a line from the text field):

- ``"chip"`` (chat column < 600 dp, or >= 160% text scale): one compact chip with the
  active mode's emoji and a ▾ in a 48 dp target. Tapping it opens a menu of the six modes
  (emoji + label, a check on the active one) and "Options for <Mode>…". After an automatic
  switch to Vision the chip carries a small dot, and its tooltip reads "· auto".
- ``"icons"`` (from 600 dp): the six toggles 📝 👁️ 🖼️ 🎬 🔊 ✨ inline, no "Output:" label.
- ``"full"`` (once they fit, ``responsive.output_row_style``): toggles that also show their
  text ("📝 Text").

In the ＋ sheet (``inline=False``, the default) the row keeps its own line: "Output: Text"
label + six toggles (``"label"``), icon toggles only for ``"icons"`` and ``"chip"``, label +
labelled toggles for ``"full"``.

Each toggle is a 32 dp visual in a 48 dp hit target with a tooltip and a Semantics label that
includes "selected". Tapping an inactive toggle selects that mode; tapping the active one
opens its options sheet (``on_open_options``). The selection lives in an ``OutputModeState``
Signal, so the composer control and the ＋ sheet row stay in sync.
"""

from __future__ import annotations

from dataclasses import field
from typing import Any, Callable, Optional

import flet as ft

from glossarion_mobile.state.store import Signal
from glossarion_mobile.ui.chat.output_modes import (
    OUTPUT_MODES,
    OutputModeState,
    chip_semantics_label,
    mode_label,
    mode_tooltip,
    normalize_mode,
    output_mode,
    semantics_label,
)
from glossarion_mobile.ui.responsive import CHIP_ARROW, CHIP_PADDING, FULL_TOGGLE_PADDING, MODE_EMOJI_SIZE, TOGGLE_GAP
from glossarion_mobile.ui.theme import HIT_TARGET
from glossarion_mobile.ui.tokens import SIZES

__all__ = ["OutputModeRow", "STYLES"]

STYLES = ("chip", "icons", "label", "full")

_SELECTED_BG = ft.Colors.SECONDARY_CONTAINER


def _auto_badge() -> ft.Badge:
    """The "· auto" cue where no "Output: Vision · auto" label is shown: a small dot."""
    return ft.Badge(small_size=8, bgcolor=ft.Colors.PRIMARY)


@ft.control
class OutputModeRow(ft.Row):
    mode_signal: Optional[Signal] = field(default=None, metadata={"skip": True})  # Signal[OutputModeState]
    style_name: str = field(default="label", metadata={"skip": True})  # chip | icons | label | full
    inline: bool = field(default=False, metadata={"skip": True})  # True: inside the composer's action row
    on_open_options: Optional[Callable[[str], Any]] = field(default=None, metadata={"skip": True})
    on_mode_changed: Optional[Callable[[str], Any]] = field(default=None, metadata={"skip": True})

    def init(self) -> None:
        super().init()
        if self.mode_signal is None:
            self.mode_signal = Signal(OutputModeState(), name="output_mode")
        self._unsubscribe: Optional[Callable[[], None]] = None
        self.label_text = ft.Text("Output: Text", theme_style=ft.TextThemeStyle.LABEL_MEDIUM, no_wrap=True)
        self.toggles: dict[str, ft.Control] = {}
        self.semantics: dict[str, ft.Semantics] = {}
        self.chip: Optional[ft.PopupMenuButton] = None
        self.chip_semantics: Optional[ft.Semantics] = None
        self.menu_items: dict[str, ft.PopupMenuItem] = {}
        self.options_item: Optional[ft.PopupMenuItem] = None
        self.spacing = TOGGLE_GAP
        # No fixed height: the toggles keep their 48 dp hit targets (UI_SPEC §0 item 4).
        self.vertical_alignment = ft.CrossAxisAlignment.CENTER
        self._build()

    # ---- building ---------------------------------------------------------------

    @property
    def effective_style(self) -> str:
        """What is drawn: inline "label" -> "icons" (no label in the action row); in the ＋ sheet
        "chip" -> "icons" (the sheet has a line of its own for the six toggles)."""
        style = self.style_name if self.style_name in STYLES else "label"
        if self.inline and style == "label":
            return "icons"
        if not self.inline and style == "chip":
            return "icons"
        return style

    @property
    def label_shown(self) -> bool:
        return not self.inline and self.effective_style in ("label", "full")

    def _toggle(self, style: str, mode_id: str, emoji: str, label: str) -> ft.Control:
        if style == "full":
            visual = ft.Container(
                content=ft.Text(f"{emoji} {label}", theme_style=ft.TextThemeStyle.LABEL_MEDIUM, no_wrap=True),
                padding=ft.Padding.symmetric(horizontal=FULL_TOGGLE_PADDING),
                height=32,
                alignment=ft.Alignment.CENTER,
                border_radius=16,
            )
            # the pill is the 32 dp visual; the 48 dp tall container around it is the hit target
            return ft.Container(
                content=visual,
                height=SIZES["hit_target"],
                alignment=ft.Alignment.CENTER,
                on_click=lambda e, m=mode_id: self.tap(m),
                tooltip=mode_tooltip(mode_id),
                key=f"mode-full-{mode_id}",
            )
        return ft.IconButton(
            icon=ft.Text(emoji, size=MODE_EMOJI_SIZE),
            tooltip=mode_tooltip(mode_id),
            on_click=lambda e, m=mode_id: self.tap(m),
            size_constraints=HIT_TARGET,
            style=ft.ButtonStyle(
                bgcolor={ft.ControlState.SELECTED: _SELECTED_BG, ft.ControlState.DEFAULT: ft.Colors.TRANSPARENT},
                padding=ft.Padding.all(4),
            ),
            selected=False,
            key=f"mode-{style}-{mode_id}",
        )

    def _build_chip(self) -> ft.Control:
        self.chip_emoji = ft.Text(output_mode(self.mode).emoji, size=MODE_EMOJI_SIZE)
        self.chip_visual = ft.Container(
            content=ft.Row(
                [self.chip_emoji, ft.Icon(ft.Icons.ARROW_DROP_DOWN, size=CHIP_ARROW)],
                spacing=0,
                tight=True,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            height=32,
            border_radius=16,
            bgcolor=_SELECTED_BG,
            padding=ft.Padding.only(left=CHIP_PADDING[0], right=CHIP_PADDING[1]),
            alignment=ft.Alignment.CENTER,
        )
        self.menu_items = {
            mode.id: ft.PopupMenuItem(
                content=f"{mode.emoji}  {mode.label}",
                checked=False,
                on_click=lambda e, m=mode.id: self.select(m),
            )
            for mode in OUTPUT_MODES
        }
        self.options_item = ft.PopupMenuItem(content="Options…", icon=ft.Icons.TUNE, on_click=lambda e: self.open_options())
        # PopupMenuButton opens on tap only (it cannot be opened from code, UI_SPEC §5.0); the 48 dp
        # tall content is its hit target.
        self.chip = ft.PopupMenuButton(
            content=ft.Container(content=self.chip_visual, height=SIZES["hit_target"], alignment=ft.Alignment.CENTER),
            items=[*self.menu_items.values(), ft.PopupMenuItem(), self.options_item],  # an empty item is a divider
            padding=0,
            tooltip=mode_tooltip(self.mode),
            key="mode-chip",
        )
        self.chip_semantics = ft.Semantics(content=self.chip, label=chip_semantics_label(self.mode), button=True)
        return self.chip_semantics

    def _build(self) -> None:
        style = self.effective_style
        self.toggles = {}
        self.semantics = {}
        self.chip = None
        self.chip_semantics = None
        self.menu_items = {}
        self.options_item = None
        controls: list[ft.Control] = []
        if style == "chip":
            controls.append(self._build_chip())
        else:
            if self.label_shown:
                controls.append(ft.Container(content=self.label_text, padding=ft.Padding.only(left=4, right=6)))
            for mode in OUTPUT_MODES:
                toggle = self._toggle(style, mode.id, mode.emoji, mode.label)
                wrapper = ft.Semantics(content=toggle, label=semantics_label(mode.id, False), button=True, selected=False)
                self.toggles[mode.id] = toggle
                self.semantics[mode.id] = wrapper
                controls.append(wrapper)
        self.controls = controls
        # Inline the control hugs its content (the action row's pills take the rest); the ＋ sheet's
        # row may scroll at narrow widths.
        self.tight = self.inline
        self.scroll = None if self.inline else ft.ScrollMode.HIDDEN
        self._sync()

    def set_style(self, style_name: str) -> bool:
        """Re-layout for a width class (``responsive.output_row_style``); True if changed."""
        if style_name == self.style_name:
            return False
        self.style_name = style_name
        self._build()
        return True

    # ---- state ---------------------------------------------------------------------

    @property
    def state(self) -> OutputModeState:
        return self.mode_signal.value

    @property
    def mode(self) -> str:
        return self.state.mode

    def _sync(self, _value: Any = None) -> None:
        state = self.state
        self.label_text.value = mode_label(state.mode, state.automatic)
        auto_cue = state.automatic and not self.label_shown
        for mode_id, toggle in self.toggles.items():
            selected = mode_id == state.mode
            if isinstance(toggle, ft.IconButton):
                toggle.selected = selected
                self._set_badge(toggle, selected and auto_cue)
            else:  # labelled toggle: the pill inside the 48 dp target carries fill and dot
                toggle.content.bgcolor = _SELECTED_BG if selected else None
                self._set_badge(toggle.content, selected and auto_cue)
            toggle.tooltip = mode_tooltip(mode_id, selected and state.automatic)
            wrapper = self.semantics[mode_id]
            wrapper.selected = selected
            wrapper.label = semantics_label(mode_id, selected)
        if self.chip is not None:
            self.chip_emoji.value = output_mode(state.mode).emoji
            self._set_badge(self.chip_visual, state.automatic)
            self.chip.tooltip = mode_tooltip(state.mode, state.automatic)
            self.chip_semantics.label = chip_semantics_label(state.mode, state.automatic)
            for mode_id, item in self.menu_items.items():
                item.checked = mode_id == state.mode
            self.options_item.content = f"Options for {output_mode(state.mode).label}…"

    @staticmethod
    def _set_badge(control: ft.Control, shown: bool) -> None:
        if shown and control.badge is None:
            control.badge = _auto_badge()
        elif not shown and control.badge is not None:
            control.badge = None

    def did_mount(self) -> None:
        super().did_mount()
        if self._unsubscribe is None:
            self._unsubscribe = self.mode_signal.subscribe(self._on_signal)

    def will_unmount(self) -> None:
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None
        super().will_unmount()

    def _on_signal(self, _value: Any) -> None:
        self._sync()
        try:
            self.update()
        except Exception:
            pass

    # ---- actions -------------------------------------------------------------------

    def select(self, mode_id: str) -> str:
        """Make ``mode_id`` the output mode (a menu pick or an inactive toggle).

        Returns ``"selected"``, or ``"unchanged"`` when it already is the mode (for tests).
        """
        mode_id = normalize_mode(mode_id)
        if mode_id == self.state.mode:
            return "unchanged"
        self.mode_signal.set(self.state.select(mode_id))
        if self._unsubscribe is None:  # not mounted: keep the visuals in step anyway
            self._sync()
        if self.on_mode_changed is not None:
            self.on_mode_changed(mode_id)
        return "selected"

    def open_options(self, mode_id: Optional[str] = None) -> None:
        """The mode's options sheet ("Options for <Mode>…", or a tap on the active toggle)."""
        if self.on_open_options is not None:
            self.on_open_options(normalize_mode(mode_id or self.mode))

    def tap(self, mode_id: str) -> str:
        """Toggle tap: select an inactive mode, or open the active mode's options.

        Returns ``"selected"`` or ``"options"`` (for tests).
        """
        mode_id = normalize_mode(mode_id)
        if mode_id == self.state.mode:
            self.open_options(mode_id)
            return "options"
        return self.select(mode_id)
