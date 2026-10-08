"""Slash commands (UI_SPEC §2.7): "/" at the start of the composer field opens a popover of the
matching commands (at most six rows visible) right above the composer; tapping one inserts it (a
command that takes an argument) or runs it (also a command whose argument is optional, e.g.
``/library [title]``). Send / Enter on a complete command runs it instead of sending the text.

``SLASH_COMMANDS`` is the §2.7 table; :func:`match_commands` filters it for the typed prefix and
:func:`parse_command` splits a complete command line into (command, argument). The chat view runs
the command (``ChatView.run_slash``) through the handlers the ＋ sheet, the Plan card chips and
the drawer already use, so a command does exactly what its button does.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Callable, Optional

import flet as ft

__all__ = [
    "MAX_VISIBLE",
    "MODE_ARGS",
    "POLICY_ARGS",
    "SLASH_COMMANDS",
    "SlashCommand",
    "SlashPopover",
    "completion",
    "is_slash_text",
    "parse_chapter_range",
    "match_commands",
    "parse_command",
]

MAX_VISIBLE = 6
ROW_HEIGHT = 56
#: /mode argument -> output mode id (``output_modes.normalize_mode`` takes "refine" too)
MODE_ARGS = ("text", "vision", "image", "video", "audio", "refine")
#: /policy argument -> the chat's ``glossary_override_mode`` (direct_text_store.GLOSSARY_OVERRIDE_MODES)
POLICY_ARGS = {"none": "none", "attachments": "attachments_only", "off": "no_glossary", "manual": "manual"}


@dataclass(frozen=True)
class SlashCommand:
    name: str  # what follows the "/" ("compile epub" has a space)
    description: str
    arg: str = ""  # argument placeholder shown in the popover ("<range>", "text|vision|…")
    optional: bool = False  # the argument may be left out: a popover tap runs the bare command
    #: it sends text to the model (or opens a tool whose job does); the others also run from Send while
    #: Send is blocked (not signed in, no key, ...): ``/library``, ``/model``, ``/qa`` need no model
    uses_model: bool = False

    @property
    def label(self) -> str:
        return f"/{self.name}" + (f" {self.arg}" if self.arg else "")


SLASH_COMMANDS: tuple[SlashCommand, ...] = (
    SlashCommand("glossary", "Extract a glossary", uses_model=True),
    SlashCommand("qa", "QA scan this chat's book (Quick Scan)"),
    SlashCommand("compile epub", "Compile an EPUB"),
    SlashCommand("compile pdf", "Compile a PDF"),
    SlashCommand("headers", "Translate headers", uses_model=True),
    SlashCommand("metadata", "Translate metadata", uses_model=True),
    SlashCommand("manga", "Manga translator", uses_model=True),
    SlashCommand("review", "Review generator", uses_model=True),
    SlashCommand("async", "Async batch", uses_model=True),
    SlashCommand("progress", "Progress manager"),
    SlashCommand("retranslate", "Retranslate chapters", "<range>", uses_model=True),
    SlashCommand("mode", "Switch the output mode", "|".join(MODE_ARGS)),
    SlashCommand("model", "Pick a model", "<query>"),
    SlashCommand("profile", "Prompt profile", "<name>"),
    SlashCommand("lang", "Target language", "<language>"),
    SlashCommand("policy", "Glossary policy of this chat", "|".join(POLICY_ARGS)),
    SlashCommand("scratch", "New scratch chat"),
    SlashCommand("export", "Export this chat"),
    SlashCommand("library", "Attach a Library book", "[title]", optional=True),
    SlashCommand("jobs", "Open Jobs"),
    SlashCommand("settings", "Search settings", "<query>"),
)


def is_slash_text(text: Any) -> bool:
    """A one-line field value that starts with "/" (the popover's trigger)."""
    value = str(text or "")
    return value.startswith("/") and "\n" not in value


def match_commands(text: Any) -> list:
    """The commands for the typed ``/prefix``: every command for "/", a prefix match on the name
    while it is typed, and the command itself once its argument is being typed."""
    if not is_slash_text(text):
        return []
    typed = " ".join(str(text)[1:].lower().split())
    trailing_space = str(text).endswith(" ")
    out = []
    for command in SLASH_COMMANDS:
        name = command.name
        if not typed or name.startswith(typed):
            out.append(command)
        elif command.arg and (typed.startswith(name + " ") or (typed == name and trailing_space)):
            out.append(command)
    return out


def parse_command(text: Any) -> Optional[tuple]:
    """``(SlashCommand, argument)`` of a complete command line (the longest name that matches),
    else None."""
    if not is_slash_text(text):
        return None
    body = " ".join(str(text)[1:].split())
    lowered = body.lower()
    best = None
    for command in SLASH_COMMANDS:
        name = command.name
        if lowered == name or (lowered.startswith(name + " ") and command.arg):
            if best is None or len(name) > len(best.name):
                best = command
    if best is None:
        return None
    return best, body[len(best.name):].strip()


def parse_chapter_range(text: Any) -> Optional[tuple]:
    """``/retranslate <range>``: the desktop chapter-range syntax (``RunEnvMixin._parse_chapter_range_text``:
    ``N`` or ``N-M``) as ``(start, end)``, else None."""
    try:
        from run_env import RunEnvMixin

        parsed = RunEnvMixin._parse_chapter_range_text(None, text)
    except Exception:
        return None
    return (int(parsed[0]), int(parsed[1])) if parsed else None


def completion(command: SlashCommand) -> str:
    """What tapping a command that takes an argument puts in the field."""
    return f"/{command.name} "


_BUILDS = itertools.count(1)


class SlashPopover:
    """The popover above the composer field: one row per matching command (six visible, the rest
    scroll). ``on_pick(command)`` runs for a tap."""

    def __init__(self, on_pick: Callable[[SlashCommand], Any]) -> None:
        self.on_pick = on_pick
        self.commands: list = []
        self.list = ft.ListView(controls=[], spacing=0, padding=0)
        self.control = ft.Container(
            content=self.list,
            visible=False,
            bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST,
            border_radius=12,
            padding=ft.Padding.symmetric(vertical=4),
            margin=ft.Margin.only(left=8, right=8, bottom=4),  # aligned with the composer card
            key="slash-popover",
        )

    @property
    def visible(self) -> bool:
        return bool(self.control.visible)

    def update_for(self, text: Any) -> list:
        """Show the commands matching ``text`` (hidden when none match or the field is not a command)."""
        self.commands = match_commands(text)
        build = next(_BUILDS)  # per-build keys: Flet 1.0.3 freezes a re-keyed replacement
        self.list.controls = [
            ft.ListTile(
                title=ft.Text(command.label, weight=ft.FontWeight.W_600),
                subtitle=ft.Text(command.description, theme_style=ft.TextThemeStyle.BODY_SMALL),
                dense=True,
                on_click=lambda e, c=command: self.on_pick(c),
                key=f"slash-{command.name.replace(' ', '-')}-{build}",
            )
            for command in self.commands
        ]
        self.list.height = ROW_HEIGHT * min(len(self.commands), MAX_VISIBLE) if self.commands else None
        self.control.visible = bool(self.commands)
        return self.commands

    def hide(self) -> None:
        self.commands = []
        self.list.controls = []
        self.control.visible = False
