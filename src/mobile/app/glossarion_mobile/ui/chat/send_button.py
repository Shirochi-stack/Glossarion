"""SendStopButton (UI_SPEC §2.4, §5.4): the view over ``SendStopMachine``.

A 40 dp circle inside a 48 dp target; an ``AnimatedSwitcher`` (scale, 200 ms)
swaps the visual per state. ``blocked`` is rendered muted but NOT disabled: a
disabled control receives no taps, and on touch a tooltip only appears on
long-press, so the tap must still arrive to explain the reason. The long-press
menu is a ``ContextMenu(primary_trigger=LONG_PRESS)`` (``PopupMenuButton``
cannot be opened from code) whose items are rebuilt on every state change.

``on_action(SendAction)`` is called with what the tap or menu item means; the
chat view maps it onto ``ChatRuns`` / JobService (send, queue, graceful stop,
force stop).
"""

from __future__ import annotations

import math
from dataclasses import field
from typing import Any, Callable, Optional

import flet as ft

from glossarion_mobile.ui import motion, tokens
from glossarion_mobile.ui.chat.send_state import SendAction, SendInputs, SendState, SendStopMachine
from glossarion_mobile.ui.components.empty_state import HALGAKOS_ASSET
from glossarion_mobile.ui.theme import HIT_TARGET, icon_data, semantic

__all__ = ["SendStopButton"]

_MUTED = ft.Colors.with_opacity(0.38, ft.Colors.ON_SURFACE)

#: owner #17: while a job runs, Stop is the Halgakos mascot turning (one turn per SPIN_TURN_MS). Each
#: linear rotation covers SPIN_TURNS turns; its ``on_animation_end`` queues the next, so the client turns
#: the image smoothly and the server only hears from it once per SPIN_TURNS turns.
SPIN_TURN_MS = 1600
SPIN_TURNS = 8
SPIN_SIZE = 36


@ft.control
class SendStopButton(ft.Container):
    machine: Optional[SendStopMachine] = field(default=None, metadata={"skip": True})
    on_action: Optional[Callable[[SendAction], Any]] = field(default=None, metadata={"skip": True})
    dark: bool = field(default=False, metadata={"skip": True})

    def init(self) -> None:
        super().init()
        if self.machine is None:
            self.machine = SendStopMachine()
        self.switcher = ft.AnimatedSwitcher(
            content=ft.Container(),
            duration=tokens.MOTION["send_morph_ms"],
            reverse_duration=tokens.MOTION["send_morph_ms"],
            transition=motion.switcher_transition(ft.AnimatedSwitcherTransition.SCALE),
        )
        self.menu = ft.ContextMenu(
            content=self.switcher,
            primary_trigger=ft.ContextMenuTrigger.LONG_PRESS,
            secondary_trigger=None,
            tertiary_trigger=None,
        )
        self.content = self.menu
        self.width = tokens.SIZES["hit_target"]
        self.height = tokens.SIZES["hit_target"]
        self.alignment = ft.Alignment.CENTER
        self.rendered_state: Optional[SendState] = None
        self.spin_image: Optional[ft.Image] = None
        self.render()

    def did_mount(self) -> None:
        super().did_mount()
        self._start_spin()

    # ---- rendering -----------------------------------------------------------------

    @property
    def state(self) -> SendState:
        return self.machine.state

    def _spinner(self, key: str, tooltip: str) -> ft.Control:
        """RUNNING: the mascot turning (still under reduce motion), tappable to stop, read out as "Stop"."""
        self.spin_image = ft.Image(
            src=HALGAKOS_ASSET, width=SPIN_SIZE, height=SPIN_SIZE, fit=ft.BoxFit.CONTAIN,
            semantics_label="Stop", gapless_playback=True, rotate=ft.Rotate(angle=0),
            animate_rotation=motion.spin_animation(SPIN_TURN_MS * SPIN_TURNS),
            on_animation_end=self._spin_more, key=f"{key}-mascot",
        )
        return ft.Container(
            key=key,
            width=tokens.SIZES["hit_target"],
            height=tokens.SIZES["hit_target"],
            alignment=ft.Alignment.CENTER,
            tooltip=tooltip,
            on_click=self._on_tap,
            ink=True,
            border_radius=tokens.RADII["full"],
            content=self.spin_image,
        )

    def _spin_more(self, e: Any = None) -> None:
        """Queue the next SPIN_TURNS turns while the spinner is on screen (not under reduce motion)."""
        image = self.spin_image
        if image is None or motion.reduced() or self.rendered_state is not SendState.RUNNING:
            return
        before = image.rotate
        image.rotate = ft.Rotate(angle=float(getattr(before, "angle", 0) or 0) + SPIN_TURNS * 2 * math.pi)
        try:
            image.update()
        except Exception:  # not mounted (yet / any more): turn from the angle the client has once mounted
            image.rotate = before

    def _visual_control(self, state: SendState) -> ft.Control:
        visual = self.machine.visual
        key = f"send-{state.value}"
        if state is SendState.RUNNING:
            return self._spinner(key, visual.tooltip)
        if state is SendState.STOPPING:
            return ft.Container(
                key=key,
                width=tokens.SIZES["send_visual"],
                height=tokens.SIZES["send_visual"],
                alignment=ft.Alignment.CENTER,
                tooltip=visual.tooltip,
                content=ft.ProgressRing(width=18, height=18, stroke_width=2, color=ft.Colors.OUTLINE),
            )
        if state is SendState.FINISHING:
            warning = semantic("warning", self.dark)
            return ft.Container(
                key=key,
                width=tokens.SIZES["send_visual"],
                height=tokens.SIZES["send_visual"],
                alignment=ft.Alignment.CENTER,
                tooltip=visual.tooltip,
                on_click=self._on_tap,
                ink=True,
                border_radius=tokens.RADII["full"],
                content=ft.Stack(
                    [
                        ft.ProgressRing(width=34, height=34, stroke_width=2, color=warning),
                        ft.Container(
                            width=34,
                            height=34,
                            alignment=ft.Alignment.CENTER,
                            content=ft.Icon(icon_data(visual.icon), color=warning, size=18),
                        ),
                    ],
                    width=34,
                    height=34,
                ),
            )
        common: dict[str, Any] = dict(
            icon=icon_data(visual.icon),
            tooltip=visual.tooltip,
            on_click=self._on_tap,
            size_constraints=HIT_TARGET,
            key=key,
        )
        if visual.style == "filled":
            return ft.FilledIconButton(**common)
        if visual.style == "tonal":
            return ft.FilledTonalIconButton(**common)
        if visual.style == "error":
            return ft.FilledIconButton(
                style=ft.ButtonStyle(bgcolor=ft.Colors.ERROR, icon_color=ft.Colors.ON_ERROR), **common
            )
        return ft.IconButton(icon_color=_MUTED, **common)  # muted: idle_empty, blocked

    def _menu_items(self) -> list[ft.PopupMenuItem]:
        return [
            ft.PopupMenuItem(content=label, on_click=lambda e, a=action: self._on_menu(a), key=f"send-menu-{action.value}")
            for action, label in self.machine.long_press_items()
        ]

    def render(self) -> bool:
        """Sync the visual with the machine; True when the state changed."""
        state = self.machine.state
        changed = state is not self.rendered_state
        if changed:
            self.spin_image = None
            # scale morph; a cross-fade under reduce motion (UI_SPEC §6.3)
            self.switcher.transition = motion.switcher_transition(ft.AnimatedSwitcherTransition.SCALE)
            self.switcher.content = self._visual_control(state)
            self.menu.primary_items = self._menu_items()
            self.rendered_state = state
        else:
            current = self.switcher.content
            if hasattr(current, "tooltip"):
                current.tooltip = self.machine.visual.tooltip
        return changed

    def apply(self, inputs: SendInputs) -> bool:
        """New app inputs; re-render and push the change when mounted."""
        self.machine.apply(inputs)
        changed = self.render()
        self._push()
        self._start_spin()
        return changed

    def _start_spin(self) -> None:
        """A freshly shown spinner starts turning once the client has it (an implicit animation runs on a
        change, so the first angle goes out after the push that showed the image at 0)."""
        image = self.spin_image
        if image is not None and not float(getattr(image.rotate, "angle", 0) or 0):
            self._spin_more()

    def _push(self) -> None:
        try:
            self.update()
        except Exception:  # not mounted yet
            pass

    # ---- events ------------------------------------------------------------------------

    def tap(self) -> SendAction:
        action = self.machine.tap()
        self.render()
        self._push()
        self._start_spin()
        if self.on_action is not None and action is not SendAction.NONE:
            self.on_action(action)
        return action

    def _on_tap(self, e: Any = None) -> None:
        self.tap()

    def _on_menu(self, action: SendAction) -> None:
        self.machine.select(action)
        self.render()
        self._push()
        if self.on_action is not None:
            self.on_action(action)
