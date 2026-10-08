"""Clear saved raw link flow and the Library's choice dialogs (UI_SPEC §3.4; epub_library
``_clear_saved_raw_link``, ``_ask_collision_policy``).

The plan, prompt and execution are ``library_core.LibraryShelf``'s
(``plan_clear_raw_link`` / ``execute_clear_raw_link``, through ``LibraryService``).
This module only asks the desktop questions: the Clear-saved-raw-link confirmation
and the one-time collision policy (Replace / Keep Both / Skip, "… All" for several,
Cancel: ``collision_text`` / ``collision_choices``).

The mobile Library has no manual Organize (n) / Undo (n): finished chat books reach
the Library by themselves and imports are copied into Library/Raw, so the desktop
counters would only move imported copies around (UI_SPEC §3.4 "Automatic"). The
shelf plans stay ``LibraryService`` API (``plan_organize_blocking`` …).
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Optional, Sequence

import flet as ft

from glossarion_mobile.services.library import CoreMissing
from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.components.dialogs import close_dialog

__all__ = [
    "ChoiceDialog",
    "clear_raw_link_flow",
    "collision_choices",
    "collision_text",
]


class ChoiceDialog:
    """An ``AlertDialog`` with N labelled choices; ``await choose(page)`` returns the chosen id or None."""

    def __init__(self, title: str, body: str, choices: Sequence[tuple], *, cancel_label: str = "Cancel") -> None:
        self.title = title
        self.body = body
        self.choices = list(choices)
        self.result: Optional[str] = None
        self._future: Optional[asyncio.Future] = None
        self._page: Any = None
        buttons: list[ft.Control] = [ft.TextButton(content=cancel_label, on_click=lambda e: self._done(None),
                                                   key="choice-cancel")]
        for choice_id, label in self.choices:
            buttons.append(ft.TextButton(content=label, on_click=lambda e, c=choice_id: self._done(c),
                                         key=f"choice-{choice_id}"))
        self.dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text(title),
            content=ft.Container(width=tokens.SIZES["dialog_max"],
                                 content=ft.Column([ft.Text(body, selectable=True)], tight=True,
                                                   scroll=ft.ScrollMode.AUTO)),
            actions=buttons,
            actions_alignment=ft.MainAxisAlignment.END,
        )

    def _done(self, value: Optional[str]) -> None:
        self.result = value
        close_dialog(self._page, self.dialog)
        if self._future is not None and not self._future.done():
            self._future.set_result(value)

    async def choose(self, page: Any) -> Optional[str]:
        self._page = page
        self._future = asyncio.get_running_loop().create_future()
        if page is None:
            return None
        page.show_dialog(self.dialog)
        return await self._future


def collision_text(collisions: Sequence[Any], dest_label: str = "the Library") -> str:
    """Desktop ``_ask_collision_policy`` text."""
    names = []
    for item in collisions:
        existing = item[1] if isinstance(item, (list, tuple)) and len(item) > 1 else item
        names.append(os.path.basename(str(existing)))
    n = len(names)
    if n == 1:
        return f"A file named “{names[0]}” already exists in {dest_label}.\n\nWhat would you like to do?"
    preview = [f"  • {name}" for name in names[:6]]
    if n > 6:
        preview.append(f"  … and {n - 6} more.")
    return (f"{n} files being imported already exist in {dest_label}:\n\n" + "\n".join(preview)
            + "\n\nWhat would you like to do with the duplicates?")


def collision_choices(n: int) -> list:
    suffix = " All" if n > 1 else ""
    return [("replace", "Replace" + suffix), ("keep_both", "Keep Both" + suffix), ("skip", "Skip" + suffix)]


async def clear_raw_link_flow(ctx: Any, books: Sequence[Any]) -> Optional[Any]:
    service = ctx.service
    try:
        plan = await ctx.io(service.plan_clear_raw_link_blocking, list(books))
    except CoreMissing as exc:
        ctx.say(f"Clearing raw links is not available in this build ({exc.name})")
        return None
    if not plan.get("targets"):
        ctx.say("Nothing to clear — the selected books have no saved raw link outside Library/Raw.")
        return None
    if await ChoiceDialog("Clear saved raw link", str(plan.get("prompt") or ""), [("yes", "Yes")]
                          ).choose(ctx.page) != "yes":
        return None
    cleared = await ctx.io(service.execute_clear_raw_link_blocking, plan)
    if cleared:
        ctx.say(f"Cleared the saved raw link of {cleared} workspace{'s' if cleared != 1 else ''}.")
    else:
        ctx.say("None of the selected workspaces could be updated. Check that the folders still exist on disk.")
    await service.refresh(reason="clear raw link")
    return cleared
