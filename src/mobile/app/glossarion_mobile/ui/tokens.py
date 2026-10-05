"""Design tokens (UI_SPEC §6): colours, type scale, spacing, radii, sizes, motion.

Pure data, no Flet import, so host tests and non-UI code can read it. ``theme.py``
turns these into ``ft.Theme`` objects and resolves the theme-role references
(``"role:outline"``) to ``ft.Colors`` values.

Flet themes have no custom colour roles, so the semantic colours (success,
warning, info, locked) and the status palette are app constants keyed by
brightness: ``(light, dark)`` pairs. Status is always rendered as icon + text +
colour, never colour alone (§6.1 rule), hence ``STATUS_STYLES`` carries an icon
name and a label for every status.
"""

from __future__ import annotations

from typing import NamedTuple, Optional, Union

__all__ = [
    "ACCENTS",
    "AMOLED",
    "COMPACT_TEXT_SCALE",
    "MONO_FAMILIES",
    "MOTION",
    "RADII",
    "SEED_COLOR",
    "SEMANTIC_COLORS",
    "SIZES",
    "SPACING",
    "STATUS_PALETTE",
    "STATUS_STYLES",
    "TERTIARY_LIGHT",
    "SECONDARY_HINT",
    "TYPE_SCALE",
    "MONO_STYLE",
    "RIBBON_STYLE",
    "TypeStyle",
    "StatusStyle",
    "gutter",
    "mono_family",
    "semantic_color",
    "status_color",
    "status_style",
]

# --------------------------------------------------------------------------
# Colour (§6.1)
# --------------------------------------------------------------------------

SEED_COLOR = "#E18F98"  # "Halgakos Rose": measured from assets/Halgakos.png
TERTIARY_LIGHT = "#5B3D57"  # "Horn Plum": tertiary override in the light theme; brand ink
SECONDARY_HINT = "#404765"  # "Skirt Navy": secondary hint

ACCENTS = {
    "halgakos_rose": SEED_COLOR,
    "desktop_blue": "#5A9FD4",  # Direct Text accent
    "library_violet": "#6C63FF",
}

# AMOLED dark: surfaces forced to pure black / near black.
AMOLED = {
    "surface": "#000000",
    "surface_container_lowest": "#000000",
    "surface_container_low": "#0A0A0A",
    "surface_container": "#0A0A0A",
    "scaffold": "#000000",
}

ColorPair = tuple[str, str]  # (light, dark)

SEMANTIC_COLORS: dict[str, ColorPair] = {
    "success": ("#2E7D4F", "#6FD49A"),
    "warning": ("#B26A00", "#FFB74D"),
    "info": ("#0E7C8C", "#5FD0DF"),
    "locked": ("#7C3AED", "#B388FF"),  # desktop purple lock
}

# A palette entry is a (light, dark) pair, "semantic:<name>" or "role:<flet color role>".
PaletteValue = Union[ColorPair, str]

STATUS_PALETTE: dict[str, PaletteValue] = {
    "completed": ("#2E7D4F", "#27AE60"),  # success; #27AE60 = desktop dark parity
    "merged": ("#0E7C8C", "#17A2B8"),  # info; #17A2B8 = desktop parity
    "in_progress": ("#F59E0B", "#F59E0B"),
    "pending": "role:outline",
    "not_translated": ("#2B6CB0", "#7FB2F0"),
    "not_refined": ("#8A63D2", "#B79CFF"),
    "no_tts": ("#8A63D2", "#B79CFF"),
    "refine_failed": ("#7F5F00", "#D8B24A"),
    "failed": "role:error",
    "qa_failed": "role:error",
    "skipped": ("#9AA0A6", "#9AA0A6"),
    "cooling": "semantic:warning",
    "disabled": "role:outlineVariant",
    # Job / generic states used by chips and the JobStrip
    "running": ("#F59E0B", "#F59E0B"),
    "queued": "role:outline",
    "done": "semantic:success",
    "stopped": "semantic:warning",
    "interrupted": "semantic:warning",
    "info": "semantic:info",
    "locked": "semantic:locked",
}


class StatusStyle(NamedTuple):
    icon: str  # Material icon name (ft.Icons member)
    label: str


# Default icon + label per status. Labels for Progress/Library statuses come from
# progress_core.present in U5; these are the generic fallbacks.
STATUS_STYLES: dict[str, StatusStyle] = {
    "completed": StatusStyle("CHECK_CIRCLE", "Completed"),
    "merged": StatusStyle("CALL_MERGE", "Merged"),
    "in_progress": StatusStyle("HOURGLASS_TOP", "In Progress"),
    "pending": StatusStyle("SCHEDULE", "Pending"),
    "not_translated": StatusStyle("TRANSLATE", "Not Translated"),
    "not_refined": StatusStyle("AUTO_FIX_OFF", "Not Refined"),
    "no_tts": StatusStyle("VOLUME_OFF", "No TTS"),
    "refine_failed": StatusStyle("AUTO_FIX_HIGH", "Refine Failed"),
    "failed": StatusStyle("ERROR", "Failed"),
    "qa_failed": StatusStyle("REPORT", "QA Failed"),
    "skipped": StatusStyle("SKIP_NEXT", "Skipped"),
    "cooling": StatusStyle("AC_UNIT", "Cooling"),
    "disabled": StatusStyle("BLOCK", "Disabled"),
    "running": StatusStyle("PLAY_CIRCLE", "Running"),
    "queued": StatusStyle("QUEUE", "Queued"),
    "done": StatusStyle("TASK_ALT", "Done"),
    "stopped": StatusStyle("STOP_CIRCLE", "Stopped"),
    "interrupted": StatusStyle("RESTART_ALT", "Interrupted"),
    "info": StatusStyle("INFO_OUTLINE", "Info"),
    "locked": StatusStyle("LOCK", "Locked"),
}


def semantic_color(role: str, dark: bool) -> str:
    light_value, dark_value = SEMANTIC_COLORS[role]
    return dark_value if dark else light_value


def status_color(status: str, dark: bool) -> str:
    """Hex colour or ``"role:<flet role>"`` for ``status`` (unknown -> outline role)."""
    value = STATUS_PALETTE.get(status, "role:outline")
    if isinstance(value, tuple):
        return value[1] if dark else value[0]
    if value.startswith("semantic:"):
        return semantic_color(value.split(":", 1)[1], dark)
    return value


def status_style(status: str) -> StatusStyle:
    return STATUS_STYLES.get(status, StatusStyle("INFO_OUTLINE", status.replace("_", " ").title()))


# --------------------------------------------------------------------------
# Typography (§6.2): size / line height (sp) / weight
# --------------------------------------------------------------------------


class TypeStyle(NamedTuple):
    size: float
    line: float
    weight: int
    letter_spacing: Optional[float] = None


TYPE_SCALE: dict[str, TypeStyle] = {
    "headline_small": TypeStyle(22, 28, 600),
    "title_large": TypeStyle(18, 24, 600),
    "title_medium": TypeStyle(16, 22, 600),
    "title_small": TypeStyle(14, 20, 600),
    "body_large": TypeStyle(15, 22, 400),
    "body_medium": TypeStyle(14, 20, 400),
    "body_small": TypeStyle(12, 16, 400),
    "label_large": TypeStyle(14, 20, 600),
    "label_medium": TypeStyle(12, 16, 600),
    "label_small": TypeStyle(11, 14, 500),
}
RIBBON_STYLE = TypeStyle(10, 12, 700, 0.6)  # caps
MONO_STYLE = TypeStyle(13, 18, 400)

MONO_FAMILIES = {
    "android": "monospace",
    "ios": "Menlo",
    "macos": "Menlo",
    "windows": "Consolas",
    "linux": "monospace",
}


def mono_family(platform: Optional[str]) -> str:
    return MONO_FAMILIES.get(str(platform or "").lower(), "monospace")


# Text scale at which compact fallbacks kick in (§7.5: subtitle -> model only,
# output-mode row icons only, option pills -> "Options (n)").
COMPACT_TEXT_SCALE = 1.6

# --------------------------------------------------------------------------
# Spacing, radii, sizes, motion (§6.3)
# --------------------------------------------------------------------------

SPACING = {
    "none": 0,
    "xxs": 2,
    "xs": 4,
    "sm": 8,
    "md": 12,
    "lg": 16,
    "xl": 20,
    "xxl": 24,
    "xxxl": 32,
    "card_padding": 12,
    "sheet_padding": 16,
    "list_item_v": 8,
    "list_item_h": 12,
}

RADII = {
    "badge": 6,
    "chip": 8,
    "field": 8,
    "cover": 8,
    "card": 12,
    "tile": 12,
    "job_strip": 12,
    "plan_card": 16,
    "user_file_card": 16,
    "bubble": 18,
    "sheet": 20,
    "composer": 24,
    "full": 999,
}

SIZES = {
    "hit_target": 48,  # minimum touch target everywhere (§0 item 4)
    "icon_button_visual": 40,
    "send_visual": 40,
    "chip": 32,
    "composer_chip": 28,
    "app_bar": 56,
    "composer_row": 40,
    "row_one_line": 48,
    "row_two_line": 60,
    "chapter_row_min": 64,
    "drawer_row_visual": 44,
    "drawer_header": 56,
    "drawer_footer": 56,
    "drawer_max": 360,
    "job_strip": 44,
    "bottom_bar": 64,
    "sidebar": 300,
    "sidebar_wide": 320,
    "side_panel": 380,
    "chat_max": 860,
    "large_phone_chat_max": 760,
    "dialog_max": 560,
    "avatar_small": 28,
    "empty_state_art": 64,
}

MOTION = {
    "state_ms": 150,
    "sheet_ms": 200,
    "send_morph_ms": 200,
    "route_ms": 250,
}

_GUTTERS = {"phone": 12, "large_phone": 16, "tablet": 24, "wide": 24}


def gutter(size_class: str) -> int:
    """Page gutter for a size class value (``responsive.SizeClass``)."""
    return _GUTTERS.get(str(getattr(size_class, "value", size_class)), 12)
