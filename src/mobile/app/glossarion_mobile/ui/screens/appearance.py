"""Appearance (``/settings/appearance``; UI_SPEC §4.15 General › Appearance, §6).

Theme System / Light / Dark / AMOLED, accent (Halgakos Rose · Desktop Blue ·
Library Violet), text scale 85–130 %, reduce motion and haptics. These are
mobile-only, so they live in ``mobile_state.json`` (``Prefs`` key ``appearance``),
never in the shared config.json. The desktop "Auto DPI / GUI scale" settings are
listed disabled with a ReasonChip (the OS scales the UI); their config values
round-trip untouched.

``apply_appearance`` re-themes the page (``theme.apply_theme``), sets the app text
scale signal (layout uses it) and the haptics switch; the app calls it at start
(``AccountsProfilesFeature``) and the page calls it on every change.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import flet as ft

from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.components.reason_chip import ReasonChip
from glossarion_mobile.ui.screens.page_base import PageScreen, section
from glossarion_mobile.ui.theme import Appearance, apply_theme

__all__ = [
    "ACCENT_LABELS",
    "APPEARANCE_PREF",
    "AppearanceScreen",
    "DEFAULT_APPEARANCE",
    "TEXT_SCALE_RANGE",
    "apply_appearance",
    "normalize_appearance",
]

APPEARANCE_PREF = "appearance"
TEXT_SCALE_RANGE = (0.85, 1.30)
ACCENT_LABELS = {"halgakos_rose": "Halgakos Rose", "desktop_blue": "Desktop Blue", "library_violet": "Library Violet"}
THEME_LABELS = {"system": "System", "light": "Light", "dark": "Dark", "amoled": "AMOLED"}
DEFAULT_APPEARANCE = {"theme": "system", "accent": "halgakos_rose", "text_scale": 1.0, "reduce_motion": False,
                      "haptics": True}
#: Desktop scaling settings that do not apply on a phone (shown disabled, values kept).
DESKTOP_SCALING_KEYS = ("auto_dpi_scale", "gui_scale_factor", "gui_font_scale")


def normalize_appearance(value: Any) -> dict:
    """Validated appearance prefs (unknown values fall back to the defaults)."""
    data = dict(DEFAULT_APPEARANCE)
    if isinstance(value, Mapping):
        theme = str(value.get("theme") or "").lower()
        if theme in THEME_LABELS:
            data["theme"] = theme
        accent = str(value.get("accent") or "")
        if accent in tokens.ACCENTS:
            data["accent"] = accent
        try:
            scale = round(float(value.get("text_scale", 1.0)), 2)
            data["text_scale"] = min(TEXT_SCALE_RANGE[1], max(TEXT_SCALE_RANGE[0], scale))
        except (TypeError, ValueError):
            pass
        data["reduce_motion"] = bool(value.get("reduce_motion", False))
        data["haptics"] = bool(value.get("haptics", True))
    return data


def apply_appearance(page: Any, value: Any, *, state: Any = None, haptics: Any = None, update: bool = True) -> dict:
    """Apply appearance prefs to the page / app state / haptics; returns the normalised prefs."""
    prefs = normalize_appearance(value)
    if page is not None:
        apply_theme(page, Appearance(prefs["theme"]), seed=tokens.ACCENTS[prefs["accent"]], text_scale=prefs["text_scale"])
        if prefs["reduce_motion"]:
            for theme in (getattr(page, "theme", None), getattr(page, "dark_theme", None)):
                if theme is not None:
                    theme.page_transitions = ft.PageTransitionsTheme(
                        android=ft.PageTransitionTheme.NONE, ios=ft.PageTransitionTheme.NONE,
                        macos=ft.PageTransitionTheme.NONE, windows=ft.PageTransitionTheme.NONE,
                        linux=ft.PageTransitionTheme.NONE)
    if state is not None and hasattr(state, "text_scale"):
        try:
            state.text_scale.set(float(prefs["text_scale"]))
        except Exception:
            pass
    if haptics is not None:
        haptics.enabled = bool(prefs["haptics"])
    if update and page is not None:
        try:
            page.update()
        except Exception:
            pass
    return prefs


class AppearanceScreen(PageScreen):
    title = "Appearance"

    def __init__(self, match: Any, ctx: Any, *, state: Any = None, haptics: Any = None) -> None:
        super().__init__(match, ctx)
        self.state = state
        self.haptics = haptics
        self.values = normalize_appearance(self.prefs.get(APPEARANCE_PREF) if self.prefs is not None else None)

    def build_body(self) -> ft.Control:
        values = self.values
        self.theme_buttons = ft.SegmentedButton(
            segments=[ft.Segment(value=k, label=ft.Text(v)) for k, v in THEME_LABELS.items()],
            selected=[values["theme"]],
            on_change=lambda e: self.set_value("theme", next(iter(e.control.selected or ["system"]))),
            key="appearance-theme",
        )
        self.accent_chips = {
            key: ft.Chip(
                label=ft.Text(label),
                leading=ft.Container(width=14, height=14, border_radius=7, bgcolor=tokens.ACCENTS[key]),
                selected=key == values["accent"],
                on_select=lambda e, k=key: self.set_value("accent", k),
                key=f"accent-{key}",
            )
            for key, label in ACCENT_LABELS.items()
        }
        self.scale_label = ft.Text(f"Text size {int(round(values['text_scale'] * 100))}%",
                                   theme_style=ft.TextThemeStyle.LABEL_MEDIUM)
        self.scale_slider = ft.Slider(min=TEXT_SCALE_RANGE[0], max=TEXT_SCALE_RANGE[1], divisions=9,
                                      value=values["text_scale"], key="appearance-scale",
                                      on_change_end=lambda e: self.set_value("text_scale", e.control.value))
        self.motion_switch = ft.Switch(label="Reduce motion", value=values["reduce_motion"],
                                       on_change=lambda e: self.set_value("reduce_motion", bool(e.control.value)))
        self.haptics_switch = ft.Switch(label="Haptic feedback", value=values["haptics"],
                                        on_change=lambda e: self.set_value("haptics", bool(e.control.value)))
        dpi = ft.ListTile(
            title=ft.Text("Auto DPI / GUI scale"),
            subtitle=ft.Text("The phone scales the interface; Text size above replaces the desktop GUI scale.",
                             theme_style=ft.TextThemeStyle.BODY_SMALL),
            trailing=ReasonChip(reason="Not on mobile", detail="DPI scaling is handled by Android / iOS. The desktop "
                                "auto_dpi_scale, gui_scale_factor and gui_font_scale values are kept untouched."),
            disabled=True,
            key="appearance-dpi",
        )
        return self.scaffold([
            section("Theme", [self.theme_buttons], key="appearance-theme-card"),
            section("Accent", [ft.Row(list(self.accent_chips.values()), wrap=True, spacing=6)]),
            section("Text", [self.scale_label, self.scale_slider, dpi]),
            section("Motion & feedback", [self.motion_switch, self.haptics_switch]),
        ])

    def set_value(self, key: str, value: Any) -> dict:
        values = dict(self.values)
        values[key] = value
        self.values = normalize_appearance(values)
        if self.prefs is not None:
            self.prefs.set(APPEARANCE_PREF, dict(self.values))
        if key == "accent":
            for name, chip in getattr(self, "accent_chips", {}).items():
                chip.selected = name == self.values["accent"]
                self.push(chip)
        if key == "text_scale" and hasattr(self, "scale_label"):
            self.scale_label.value = f"Text size {int(round(self.values['text_scale'] * 100))}%"
            self.push(self.scale_label)
        apply_appearance(self.page, self.values, state=self.state, haptics=self.haptics)
        return self.values

    def current(self) -> Optional[dict]:
        return dict(self.values)
