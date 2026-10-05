"""OutputModeRow (UI_SPEC §2.3 item 3, §5.4): the desktop Direct Text "Output:" row.

"Output: Text" label followed by six compact toggles 📝 👁️ 🖼️ 🎬 🔊 ✨, always
visible. Width rules: icons only under 400 dp or at >= 160% text scale; label
+ icons from 400 dp; label + toggles that also show their text ("📝 Text") on
tablets (>= 900 dp). Each toggle is a 32 dp visual in a 48 dp hit target with
a tooltip and a Semantics label that includes "selected".

Tapping an inactive toggle selects that mode; tapping the active one opens its
options sheet (``on_open_options``). The selection lives in an
``OutputModeState`` Signal so the composer row and the ＋ sheet row stay in sync.
"""

from __future__ import annotations

from dataclasses import field
from typing import Any, Callable, Optional

import flet as ft

from glossarion_mobile.state.store import Signal
from glossarion_mobile.ui.chat.output_modes import (
    OUTPUT_MODES,
    OutputModeState,
    mode_label,
    mode_tooltip,
    normalize_mode,
    semantics_label,
)
from glossarion_mobile.ui.theme import HIT_TARGET

__all__ = ["OutputModeRow"]

_SELECTED_BG = ft.Colors.SECONDARY_CONTAINER


@ft.control
class OutputModeRow(ft.Row):
    mode_signal: Optional[Signal] = field(default=None, metadata={"skip": True})  # Signal[OutputModeState]
    style_name: str = field(default="label", metadata={"skip": True})  # icons | label | full
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
        self.spacing = 2
        # No fixed height: the toggles keep their 48 dp hit targets (UI_SPEC §0 item 4).
        self.vertical_alignment = ft.CrossAxisAlignment.CENTER
        self.scroll = ft.ScrollMode.HIDDEN
        self._build()

    # ---- building ---------------------------------------------------------------

    def _toggle(self, mode_id: str, emoji: str, label: str) -> ft.Control:
        if self.style_name == "full":
            return ft.Container(
                content=ft.Text(f"{emoji} {label}", theme_style=ft.TextThemeStyle.LABEL_MEDIUM),
                padding=ft.Padding.symmetric(horizontal=10),
                height=32,
                alignment=ft.Alignment.CENTER,
                border_radius=16,
                on_click=lambda e, m=mode_id: self.tap(m),
                tooltip=mode_tooltip(mode_id),
                key=f"mode-{mode_id}",
            )
        return ft.IconButton(
            icon=ft.Text(emoji, size=16),
            tooltip=mode_tooltip(mode_id),
            on_click=lambda e, m=mode_id: self.tap(m),
            size_constraints=HIT_TARGET,
            style=ft.ButtonStyle(
                bgcolor={ft.ControlState.SELECTED: _SELECTED_BG, ft.ControlState.DEFAULT: ft.Colors.TRANSPARENT},
                padding=ft.Padding.all(4),
            ),
            selected=False,
            key=f"mode-{mode_id}",
        )

    def _build(self) -> None:
        self.toggles = {}
        self.semantics = {}
        controls: list[ft.Control] = []
        if self.style_name != "icons":
            controls.append(ft.Container(content=self.label_text, padding=ft.Padding.only(left=4, right=6)))
        for mode in OUTPUT_MODES:
            toggle = self._toggle(mode.id, mode.emoji, mode.label)
            wrapper = ft.Semantics(content=toggle, label=semantics_label(mode.id, False), button=True, selected=False)
            self.toggles[mode.id] = toggle
            self.semantics[mode.id] = wrapper
            controls.append(wrapper)
        self.controls = controls
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
        for mode_id, toggle in self.toggles.items():
            selected = mode_id == state.mode
            if isinstance(toggle, ft.IconButton):
                toggle.selected = selected
            else:
                toggle.bgcolor = _SELECTED_BG if selected else None
            wrapper = self.semantics[mode_id]
            wrapper.selected = selected
            wrapper.label = semantics_label(mode_id, selected)

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

    def tap(self, mode_id: str) -> str:
        """Toggle tap: select an inactive mode, or open the active mode's options.

        Returns ``"selected"`` or ``"options"`` (for tests).
        """
        mode_id = normalize_mode(mode_id)
        if mode_id == self.state.mode:
            if self.on_open_options is not None:
                self.on_open_options(mode_id)
            return "options"
        self.mode_signal.set(self.state.select(mode_id))
        if self._unsubscribe is None:  # not mounted: keep the visuals in step anyway
            self._sync()
        if self.on_mode_changed is not None:
            self.on_mode_changed(mode_id)
        return "selected"
