"""Organize (n) / Undo (n) / Clear saved raw link flows (UI_SPEC §3.4; epub_library ``_organize_into_library``,
``_undo_organize_prompt``, ``_clear_saved_raw_link``).

The plans, prompts, moves and summaries are ``library_core.LibraryShelf``'s
(``plan_organize`` / ``execute_organize``, ``plan_undo`` / ``undo_collisions`` /
``execute_undo``, ``plan_clear_raw_link`` / ``execute_clear_raw_link``, through
``LibraryService``). This module only asks the desktop questions: the Organize
preview (Yes / No), the one-time collision policy (Replace / Keep Both / Skip,
"… All" for several, Cancel), the Undo category (Raw / Translated / All) and the
Clear-saved-raw-link confirmation. Since mobile imports copy files into the
Library, these mostly apply to imported desktop data and migrated workspaces.
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
    "organize_flow",
    "undo_flow",
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


def _summary(result: Any, fallback: str) -> str:
    if isinstance(result, dict) and result.get("summary"):
        return str(result["summary"])
    return fallback


async def organize_flow(ctx: Any, books: Optional[Sequence[Any]] = None) -> Optional[Any]:
    """Organize (n); ``books``: the selection bar's "Organize selected" (only their files move)."""
    service = ctx.service
    try:
        if books is None:
            plan = await ctx.io(service.plan_organize_blocking)
        else:
            selected = [dict(b) for b in books]
            plan = await ctx.io(lambda: service.plan_organize_blocking(selected))
    except CoreMissing as exc:
        ctx.say(f"Organize is not available in this build ({exc.name})")
        return None
    if not plan.get("raw_moves") and not plan.get("translated_moves"):
        ctx.say("All resolvable files are already in Library/Raw or Library/Translated. Nothing to move."
                if books is None else "The selected books' files are already in the Library. Nothing to move.")
        return None
    body = ("Move the following files into the Library?\n\n"
            + "\n".join("  • " + line for line in plan.get("preview") or ())
            + "\n\nThis is reversible via the Undo Move button.")
    if await ChoiceDialog("Organize Files into Library", body, [("yes", "Yes")], cancel_label="No"
                          ).choose(ctx.page) != "yes":
        return None
    collisions = list(plan.get("collisions") or ())
    policy = "keep_both"
    if collisions:
        policy = await ChoiceDialog("Duplicate files", collision_text(collisions),
                                    collision_choices(len(collisions))).choose(ctx.page)
        if policy is None:
            return None
    result = await ctx.io(service.execute_organize_blocking, plan, policy)
    ctx.say(_summary(result, "Organized the Library"))
    await service.refresh(reason="organize")
    return result


async def undo_flow(ctx: Any) -> Optional[Any]:
    service = ctx.service
    try:
        plan = await ctx.io(service.plan_undo_blocking)
    except CoreMissing as exc:
        ctx.say(f"Undo Move is not available in this build ({exc.name})")
        return None
    if not plan.get("raw_map") and not plan.get("trans_map"):
        ctx.say("No files to undo — Library/Raw and Library/Translated are both empty and the origins "
                "registry is clean.")
        return None
    kind = await ChoiceDialog("Undo Move", str(plan.get("prompt") or ""),
                              [("raw", "Raw"), ("translated", "Translated"), ("all", "All")]).choose(ctx.page)
    if kind is None:
        return None
    restore_raw = kind in ("raw", "all")
    restore_trans = kind in ("translated", "all")
    collisions = await ctx.io(service.undo_collisions_blocking, plan, restore_raw, restore_trans)
    policy = "keep_both"
    if collisions:
        policy = await ChoiceDialog("Duplicate files", collision_text(collisions, "the original location"),
                                    collision_choices(len(collisions))).choose(ctx.page)
        if policy is None:
            return None
    result = await ctx.io(service.execute_undo_blocking, plan, restore_raw, restore_trans, policy, collisions)
    ctx.say(_summary(result, "Undo Move finished"))
    await service.refresh(reason="undo")
    return result


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
