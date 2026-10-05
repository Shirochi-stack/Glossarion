"""Material 3 theme for the app (UI_SPEC §6): Halgakos Rose seed, compact density.

``build_theme`` turns ``tokens`` into an ``ft.Theme``; ``apply_theme`` sets the
page's light/dark themes and theme mode for an ``Appearance`` choice
(System / Light / Dark / AMOLED). Semantic and status colours are app
constants (Flet themes have no custom roles); ``status_color``/``semantic``
resolve them for the current brightness, including theme-role references
such as ``"role:outline"`` -> ``ft.Colors.OUTLINE``.

Hit targets: Material 3 compact density shrinks visuals, never touch areas;
``HIT_TARGET`` constraints keep every icon button at 48 x 48 dp (§0 item 4).
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

import flet as ft

from glossarion_mobile.ui import tokens

__all__ = [
    "Appearance",
    "HIT_TARGET",
    "apply_theme",
    "build_theme",
    "build_text_theme",
    "hit_target",
    "icon_data",
    "is_dark",
    "mono_family",
    "page_platform",
    "resolve_color",
    "semantic",
    "status_color",
    "text_style",
]


class Appearance(str, Enum):
    SYSTEM = "system"
    LIGHT = "light"
    DARK = "dark"
    AMOLED = "amoled"


def hit_target() -> ft.BoxConstraints:
    size = tokens.SIZES["hit_target"]
    return ft.BoxConstraints(min_width=size, min_height=size)


HIT_TARGET = hit_target()

_WEIGHTS = {
    400: ft.FontWeight.W_400,
    500: ft.FontWeight.W_500,
    600: ft.FontWeight.W_600,
    700: ft.FontWeight.W_700,
}


def text_style(name: str, *, scale: float = 1.0, color: Optional[str] = None, family: Optional[str] = None) -> ft.TextStyle:
    """``ft.TextStyle`` for a token name (``"label_small"``, ``"mono"``, ``"ribbon"``, ...)."""
    if name == "mono":
        spec = tokens.MONO_STYLE
    elif name == "ribbon":
        spec = tokens.RIBBON_STYLE
    else:
        spec = tokens.TYPE_SCALE[name]
    return ft.TextStyle(
        size=round(spec.size * scale, 2),
        height=round(spec.line / spec.size, 4),
        weight=_WEIGHTS.get(spec.weight, ft.FontWeight.NORMAL),
        letter_spacing=spec.letter_spacing,
        color=color,
        font_family=family,
    )


def build_text_theme(scale: float = 1.0) -> ft.TextTheme:
    return ft.TextTheme(**{name: text_style(name, scale=scale) for name in tokens.TYPE_SCALE})


def build_theme(
    *,
    dark: bool = False,
    amoled: bool = False,
    seed: str = tokens.SEED_COLOR,
    text_scale: float = 1.0,
) -> ft.Theme:
    """Theme for one brightness. ``amoled`` only applies to the dark theme."""
    scheme_overrides: dict[str, Any] = {}
    if not dark and seed == tokens.SEED_COLOR:
        scheme_overrides["tertiary"] = tokens.TERTIARY_LIGHT  # Horn Plum; tertiaryContainer is generated
    if dark and amoled:
        for key in ("surface", "surface_container_lowest", "surface_container_low", "surface_container"):
            scheme_overrides[key] = tokens.AMOLED[key]
    return ft.Theme(
        color_scheme_seed=seed,
        color_scheme=ft.ColorScheme(**scheme_overrides) if scheme_overrides else None,
        use_material3=True,
        visual_density=ft.VisualDensity.COMPACT,
        text_theme=build_text_theme(text_scale),
        scaffold_bgcolor=tokens.AMOLED["scaffold"] if (dark and amoled) else None,
        page_transitions=ft.PageTransitionsTheme(
            android=ft.PageTransitionTheme.FADE_FORWARDS,  # fade-through (§6.3 motion)
            ios=ft.PageTransitionTheme.CUPERTINO,
            macos=ft.PageTransitionTheme.CUPERTINO,
            windows=ft.PageTransitionTheme.FADE_FORWARDS,
            linux=ft.PageTransitionTheme.FADE_FORWARDS,
        ),
    )


def apply_theme(
    page: Any,
    appearance: Appearance = Appearance.SYSTEM,
    *,
    seed: str = tokens.SEED_COLOR,
    text_scale: float = 1.0,
) -> None:
    """Set ``page.theme``/``dark_theme``/``theme_mode`` (no ``update()``)."""
    appearance = Appearance(appearance)
    page.theme = build_theme(dark=False, seed=seed, text_scale=text_scale)
    page.dark_theme = build_theme(dark=True, amoled=appearance is Appearance.AMOLED, seed=seed, text_scale=text_scale)
    page.theme_mode = {
        Appearance.SYSTEM: ft.ThemeMode.SYSTEM,
        Appearance.LIGHT: ft.ThemeMode.LIGHT,
        Appearance.DARK: ft.ThemeMode.DARK,
        Appearance.AMOLED: ft.ThemeMode.DARK,
    }[appearance]


def is_dark(page: Any) -> bool:
    mode = getattr(page, "theme_mode", None)
    if mode == ft.ThemeMode.DARK:
        return True
    if mode == ft.ThemeMode.LIGHT:
        return False
    return getattr(page, "platform_brightness", None) == ft.Brightness.DARK


_ROLE_COLORS = {
    "outline": ft.Colors.OUTLINE,
    "outlineVariant": ft.Colors.OUTLINE_VARIANT,
    "error": ft.Colors.ERROR,
    "primary": ft.Colors.PRIMARY,
    "tertiary": ft.Colors.TERTIARY,
    "onSurfaceVariant": ft.Colors.ON_SURFACE_VARIANT,
}


def resolve_color(value: str) -> str:
    """Hex values pass through; ``"role:<name>"`` becomes the ``ft.Colors`` role."""
    if value.startswith("role:"):
        return _ROLE_COLORS.get(value[5:], ft.Colors.OUTLINE)
    return value


def status_color(status: str, dark: bool = False) -> str:
    return resolve_color(tokens.status_color(status, dark))


def semantic(role: str, dark: bool = False) -> str:
    return tokens.semantic_color(role, dark)


def icon_data(name: Any) -> Any:
    """``ft.Icons`` member for a token icon name (``"CHECK_CIRCLE"``); IconData passes through."""
    if isinstance(name, str):
        return getattr(ft.Icons, name.upper(), ft.Icons.HELP_OUTLINE)
    return name


def page_platform(page: Any) -> str:
    platform = getattr(page, "platform", None)
    return str(getattr(platform, "value", platform) or "")


def mono_family(page: Any = None) -> str:
    return tokens.mono_family(page_platform(page) if page is not None else None)
