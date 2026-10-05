"""Direct Text output modes (UI_SPEC §2.3, §2.6). Pure Python, no Flet import.

The six modes, their emoji and labels are the desktop ``_InputOutputDialog``
``_OUTPUT_MODE_CHOICES`` (translator_gui.py) verbatim: the composer shows the
same "Output: Text" label followed by 📝 👁️ 🖼️ 🎬 🔊 ✨. Desktop normalises
``refine`` to ``refinement`` and anything unknown to ``text``; so does
``normalize_mode``. ``tests_host/test_ui_foundations.py`` checks the tuple
against the desktop source so the two cannot drift.

Persistence (``direct_text_output_mode`` or the chat override, never the global
``output_mode``) arrives with MobileConfigStore in U2/U3.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import NamedTuple, Optional

__all__ = [
    "IMAGE_ATTACHMENT_EXTENSIONS",
    "OUTPUT_MODES",
    "OUTPUT_MODE_CHOICES",
    "OutputMode",
    "OutputModeState",
    "VISION_ARCHIVE_ATTACHMENT_EXTENSIONS",
    "is_vision_attachment",
    "mode_label",
    "mode_tooltip",
    "normalize_mode",
    "output_mode",
    "semantics_label",
]

# Desktop _InputOutputDialog._OUTPUT_MODE_CHOICES (mode id, emoji, label).
OUTPUT_MODE_CHOICES = (
    ("text", "📝", "Text"),
    ("vision", "👁️", "Vision"),
    ("image", "🖼️", "Image"),
    ("video", "🎬", "Video"),
    ("audio", "🔊", "Audio"),
    ("refinement", "✨", "Refine"),
)

# Desktop _InputOutputDialog._IMAGE_ATTACHMENT_EXTENSIONS / _VISION_ARCHIVE_ATTACHMENT_EXTENSIONS:
# attaching one of these switches the mode to Vision automatically (§2.3 "Auto-switch").
IMAGE_ATTACHMENT_EXTENSIONS = frozenset(
    {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tif", ".tiff", ".svg", ".ico", ".heic", ".heif", ".avif", ".jxl"}
)
VISION_ARCHIVE_ATTACHMENT_EXTENSIONS = frozenset({".cbz"})

# Material icon used for semantics / the options sheet header (§2.6 table).
_MODE_ICONS = {
    "text": "NOTES",
    "vision": "VISIBILITY",
    "image": "IMAGE",
    "video": "MOVIE",
    "audio": "VOLUME_UP",
    "refinement": "AUTO_FIX_HIGH",
}


class OutputMode(NamedTuple):
    id: str
    emoji: str
    label: str
    icon: str


OUTPUT_MODES: tuple[OutputMode, ...] = tuple(
    OutputMode(mode_id, emoji, label, _MODE_ICONS[mode_id]) for mode_id, emoji, label in OUTPUT_MODE_CHOICES
)
_BY_ID = {mode.id: mode for mode in OUTPUT_MODES}

AUTO_SUFFIX = " · auto"  # desktop: " · automatic for visual attachment" (shortened for phones, §2.3)


def normalize_mode(mode: Optional[str]) -> str:
    """Desktop ``_set_direct_output_mode`` normalisation."""
    value = str(mode or "text").strip().lower()
    if value == "refine":
        value = "refinement"
    return value if value in _BY_ID else "text"


def output_mode(mode: Optional[str]) -> OutputMode:
    return _BY_ID[normalize_mode(mode)]


def mode_label(mode: Optional[str], automatic: bool = False) -> str:
    """The row label, e.g. "Output: Text" or "Output: Vision · auto"."""
    return f"Output: {output_mode(mode).label}{AUTO_SUFFIX if automatic else ''}"


def mode_tooltip(mode: Optional[str]) -> str:
    return f"Output mode: {output_mode(mode).label}"


def semantics_label(mode: Optional[str], selected: bool) -> str:
    """Spoken label (desktop accessible name "<Label> output mode", plus "selected")."""
    text = f"{output_mode(mode).label} output mode"
    return f"{text}, selected" if selected else text


def is_vision_attachment(path: Optional[str]) -> bool:
    extension = os.path.splitext(str(path or ""))[1].lower()
    return extension in IMAGE_ATTACHMENT_EXTENSIONS or extension in VISION_ARCHIVE_ATTACHMENT_EXTENSIONS


@dataclass(frozen=True)
class OutputModeState:
    """Selected mode, whether it was switched automatically, and the mode to restore."""

    mode: str = "text"
    automatic: bool = False
    previous: Optional[str] = None

    @property
    def label(self) -> str:
        return mode_label(self.mode, self.automatic)

    def select(self, mode: Optional[str]) -> "OutputModeState":
        """A manual choice clears the automatic flag and the restore target."""
        return OutputModeState(normalize_mode(mode), False, None)

    def attachment_changed(self, path: Optional[str]) -> "OutputModeState":
        """Auto-switch rule (desktop 5336-5383): visual attachment -> Vision · auto;
        removing it (or attaching a non-visual file) restores the previous mode."""
        if path and is_vision_attachment(path):
            if self.mode == "vision":
                return self
            return OutputModeState("vision", True, self.mode)
        if self.automatic:
            return OutputModeState(normalize_mode(self.previous), False, None)
        return self
