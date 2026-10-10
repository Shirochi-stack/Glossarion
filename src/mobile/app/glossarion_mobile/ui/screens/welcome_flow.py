"""Welcome flow state (pure Python, no Flet, Python 3.10): steps, glossary-mode cards, config writes.

``WelcomeScreen`` (``welcome.py``) renders this; host tests drive it directly. The
glossary-mode cards and the config writes on finish are the desktop first-run
welcome's (``translator_gui._show_glossary_mode_welcome``); ``tests_host/test_chat.py``
compares them with the desktop source.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

__all__ = [
    "DEFAULT_GLOSSARY_MODE",
    "GLOSSARY_MODE_CARDS",
    "NOTE",
    "OFF_GLOSSARY_MODES",
    "STEPS",
    "STEP_TITLES",
    "TAGLINE",
    "WelcomeFlow",
    "welcome_glossary_updates",
]

STEPS = ("sign_in", "providers", "language_glossary", "permissions", "done")
STEP_TITLES = {
    "sign_in": "Welcome to Glossarion",
    "providers": "Other providers",
    "language_glossary": "Choose Your Glossary Mode",
    "permissions": "Permissions",
    "done": "You're ready",
}
TAGLINE = "Translate novels, manga and documents with your own AI accounts and keys"

#: Desktop ``_show_glossary_mode_welcome`` page 1 cards: (value, emoji, title, subtitle, features, recommendation).
GLOSSARY_MODE_CARDS = (
    ("off", "🚫", "OFF", "Manual control + Auto-Mapping ON",
     ("✓ No automatic extraction", "✓ Enables Auto-Mapping", "✓ Zero extra API cost"), None),
    ("off_fuzzy_automap", "🔍", "OFF (Fuzzy Mapping)", "Auto-Mapping + Fuzzy name matching",
     ("✓ No automatic extraction", "✓ Fuzzy filename matching", "✓ Zero extra API cost"), None),
    ("off_no_automap", "🔒", "MANUAL GLOSSARY ONLY", "Off + disables auto-mapping",
     ("✓ No automatic extraction", "✓ No auto-mapping", "✓ Use editor's Load Glossary button"), None),
    ("no_glossary", "📭", "NO GLOSSARY", "Translate without any glossary",
     ("✓ Skips glossary entirely", "✓ Fastest translation", "✓ Zero extra API cost"), None),
    ("minimal", "⚡", "MINIMAL", "Compact batch extraction",
     ("✓ Batch extracts names & terms", "✓ Lightweight and cost-efficient", "⚠ May miss uncommon terms"), None),
    ("balanced", "⭐", "BALANCED", "Glossary merging + splitting",
     ("✓ Merges chapters, splits long ones", "✓ Best quality-to-cost ratio", "✓ Smart deduplication"),
     "✅ Recommended for most users"),
    ("full", "🔬", "FULL", "Per-chapter extraction",
     ("✓ Per-chapter extraction", "✓ Maximum term capture", "💰 Higher API cost"), "⚡ Best for important novels"),
    ("single_pass", "📑", "SINGLE PASS", "Inline glossary during translation",
     ("✓ Extracts while translating", "✓ No separate glossary pass", "⚠ Adds prompt/output overhead"),
     "⚡ Fast setup, live glossary"),
)
DEFAULT_GLOSSARY_MODE = "balanced"  # desktop: selected_mode = ['balanced']
NOTE = "⚠️ AI models may produce smaller glossaries due to training biases. Full mode captures the most terms but costs more."
OFF_GLOSSARY_MODES = ("off", "off_fuzzy_automap", "off_no_automap", "no_glossary")


def welcome_glossary_updates(mode: str) -> dict:
    """Config writes of the desktop welcome's glossary page on "Get Started"."""
    mode_val = str(mode or DEFAULT_GLOSSARY_MODE)
    updates: dict = {
        "auto_glossary_mode": mode_val,
        "enable_auto_glossary": mode_val not in OFF_GLOSSARY_MODES,
    }
    if mode_val not in ("off", "off_no_automap", "no_glossary"):
        updates["append_glossary"] = True
        updates["append_glossary_auto_load"] = True
    if mode_val == "off_no_automap":
        updates["append_glossary_auto_load"] = False
    return updates


@dataclass
class WelcomeFlow:
    """Pure state of the Welcome steps."""

    step: int = 0
    glossary_mode: str = DEFAULT_GLOSSARY_MODE
    target_language: str = "English"
    signed_in: bool = False
    skipped_sign_in: bool = False
    api_key_set: bool = False
    # U13: the run budget the user sets here (None: leave the setting alone)
    max_output_tokens: Optional[int] = None
    chunk_size: Optional[str] = None  # "" = auto (manual_chunk_size cleared)
    finished: bool = False
    skipped: bool = False
    history: list = field(default_factory=list)

    @property
    def step_id(self) -> str:
        return STEPS[self.step]

    @property
    def is_last(self) -> bool:
        return self.step == len(STEPS) - 1

    def go(self, step_id: str) -> str:
        self.history.append(self.step)
        self.step = STEPS.index(step_id)
        return self.step_id

    def next(self) -> str:
        if self.step_id == "sign_in" and not self.signed_in:
            self.skipped_sign_in = True
        if self.step_id == "sign_in" and self.signed_in:
            return self.go("language_glossary")  # step 2 is optional once signed in
        if not self.is_last:
            return self.go(STEPS[self.step + 1])
        return self.step_id

    def back(self) -> str:
        if self.history:
            self.step = self.history.pop()
        elif self.step > 0:
            self.step -= 1
        return self.step_id

    def mark_signed_in(self) -> str:
        self.signed_in = True
        self.skipped_sign_in = False
        return self.go("language_glossary") if self.step_id == "sign_in" else self.step_id

    def select_mode(self, mode: str) -> None:
        if mode in {card[0] for card in GLOSSARY_MODE_CARDS}:
            self.glossary_mode = mode

    def finish_updates(self) -> dict:
        updates = welcome_glossary_updates(self.glossary_mode)
        if self.target_language:
            updates["output_language"] = self.target_language
        if self.max_output_tokens:
            updates["max_output_tokens"] = int(self.max_output_tokens)
        if self.chunk_size is not None:
            updates["manual_chunk_size"] = str(self.chunk_size)
        updates["glossary_mode_dialog_shown"] = True
        self.finished = True
        return updates

    def skip_updates(self) -> dict:
        self.skipped = True
        return {"glossary_mode_dialog_shown": True}
