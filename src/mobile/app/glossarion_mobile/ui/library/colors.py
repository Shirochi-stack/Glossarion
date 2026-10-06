"""Library colour constants (UI_SPEC §3.2, §6.1): ribbons, pills, type badges, warnings.

Flet themes have no custom colour roles, so these are app constants keyed by
brightness. Dark values are the desktop hex values verbatim (epub_library
``_BookCard``); light values use the same hue at about M3 tone 40 so they read on
light surfaces. Ribbons sit on cover images, so they keep the desktop colours in
both modes. Colours with opacity use Flet's ``"<color>,<opacity>"`` string form.

Status always reads as icon/emoji + text + colour (§6.1 rule): these colours are
never the only signal. Pure data; no Flet import.
"""

from __future__ import annotations

from typing import NamedTuple

__all__ = [
    "PILL_COLORS",
    "RIBBON_COLORS",
    "TYPE_BADGES",
    "WARNING_COLORS",
    "GP_STATUS_PALETTE_KEY",
    "PillColors",
    "pill_colors",
    "ribbon_colors",
    "type_badge",
    "warning_colors",
    "with_opacity",
]


def with_opacity(color: str, opacity: float) -> str:
    """Flet ``Colors.with_opacity`` without importing Flet."""
    return f"{color},{opacity}"


class PillColors(NamedTuple):
    text: str
    background: str
    border: str
    pct: str = ""  # the "NN%" part (in progress only)


# (dark, light)
PILL_COLORS: dict[str, tuple[PillColors, PillColors]] = {
    "outdated_progress": (
        PillColors("#ffb347", "rgba", "#ffb347"),
        PillColors("#9a5b00", "rgba", "#9a5b00"),
    ),
    "not_started": (
        PillColors("#8ab4d0", "rgba", "#8ab4d0"),
        PillColors("#2f6a8f", "rgba", "#2f6a8f"),
    ),
    "ready_to_compile": (
        PillColors("#6ee8a0", "rgba", "#6ee8a0"),
        PillColors("#1e7a48", "rgba", "#1e7a48"),
    ),
    "in_progress": (
        PillColors("#ffd166", "rgba", "#6c63ff", "#8ab4d0"),
        PillColors("#7a5a00", "rgba", "#4b44c9", "#2f6a8f"),
    ),
}

# Pill backgrounds (desktop rgba tints): (hex, opacity) per state; light mode uses the light text hue.
_PILL_TINT = {
    "outdated_progress": ("#ffb347", 0.18),
    "not_started": ("#8ab4d0", 0.15),
    "ready_to_compile": ("#6ee8a0", 0.16),
    "in_progress": ("#6c63ff", 0.18),
}

#: Ribbon text / background per state (desktop ribbons; same on light and dark).
RIBBON_COLORS: dict[str, tuple[str, str]] = {
    "not_started": ("#ffffff", with_opacity("#8ab4d0", 0.92)),
    "in_progress": ("#ffffff", with_opacity("#6c63ff", 0.92)),
    "ready_to_compile": ("#ffffff", with_opacity("#3caa6e", 0.95)),
    "outdated_progress": ("#ffffff", with_opacity("#ffb347", 0.92)),
    "compiling": ("#1e1616", with_opacity("#ffd166", 0.95)),
}

#: Type badge: kind -> (emoji, label, dark colour, light colour) (desktop ``type_info``).
TYPE_BADGES: dict[str, tuple[str, str, str, str]] = {
    "epub": ("\U0001f4d5", "EPUB", "#6c63ff", "#4b44c9"),
    "pdf": ("\U0001f4c4", "PDF", "#e74c3c", "#b3261e"),
    "txt": ("\U0001f4d7", "TXT", "#2ecc71", "#1e7a48"),
    "html": ("\U0001f310", "HTML", "#3498db", "#1f6fa8"),
    "image": ("\U0001f5bc️", "IMG", "#f39c12", "#9a5b00"),
    "in_progress": ("\U0001f4c1", "FOLDER", "#ffd166", "#7a5a00"),
}

#: Warning chips: role -> (dark text, light text, tint opacity).
WARNING_COLORS: dict[str, tuple[str, str, float]] = {
    "missing_raw": ("#ff9e6d", "#a24a1d", 0.15),
    "conflicts": ("#ffb347", "#9a5b00", 0.15),
    "qa": ("role:error", "role:error", 0.12),
}

#: Glossary Progress statuses -> the shared status palette key (tokens.STATUS_PALETTE, UI_SPEC §6.1).
GP_STATUS_PALETTE_KEY: dict[str, str] = {
    "completed": "completed",
    "skipped": "skipped",
    "skipped_empty": "skipped",
    "skipped_image_only": "skipped",
    "skipped_title_header_only": "skipped",
    "failed": "failed",
    "qa_failed": "qa_failed",
    "error": "failed",
    "merged": "merged",
    "in_progress": "in_progress",
    "partially_in_progress": "in_progress",
    "not_completed": "not_translated",
    "not_translated": "not_translated",
    "not_refined": "not_refined",
    "refine_failed": "refine_failed",
    "pending": "pending",
}


def pill_colors(state: str, dark: bool) -> PillColors:
    pair = PILL_COLORS.get(state) or PILL_COLORS["in_progress"]
    base = pair[0] if dark else pair[1]
    tint_hex, opacity = _PILL_TINT.get(state, _PILL_TINT["in_progress"])
    if not dark:
        tint_hex = base.text if state != "in_progress" else base.border
    return PillColors(base.text, with_opacity(tint_hex, opacity), base.border, base.pct)


def ribbon_colors(state: str) -> tuple[str, str]:
    return RIBBON_COLORS.get(state) or RIBBON_COLORS["in_progress"]


def type_badge(kind: str, dark: bool) -> tuple[str, str, str]:
    """``(emoji, label, colour)`` for a book type / workspace kind (unknown -> EPUB, like desktop)."""
    emoji, label, dark_color, light_color = TYPE_BADGES.get(kind, TYPE_BADGES["epub"])
    return emoji, label, dark_color if dark else light_color


def warning_colors(role: str, dark: bool) -> tuple[str, str]:
    """``(text, background)`` of a warning chip."""
    dark_text, light_text, opacity = WARNING_COLORS.get(role, WARNING_COLORS["conflicts"])
    text = dark_text if dark else light_text
    if text.startswith("role:"):
        return text, text
    return text, with_opacity(text, opacity)
