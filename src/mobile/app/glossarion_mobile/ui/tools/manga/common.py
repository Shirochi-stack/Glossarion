"""Small pieces the manga tabs share: the tab base class, the status chip of an ``OptionRow`` /
``ModelEntry`` and the ExportSheet for a produced file."""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Optional

import flet as ft

from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.components.action_sheet import ActionItem, ActionSheet
from glossarion_mobile.ui.components.reason_chip import ReasonChip
from glossarion_mobile.ui.theme import icon_data

__all__ = ["CHIP_STATUS", "JobEnds", "MangaTab", "export_sheet", "option_chip", "push", "reason_or_chip"]

log = logging.getLogger("glossarion.tools.manga")

#: OptionRow / ModelEntry status -> ``tokens.STATUS_STYLES`` key (icon + colour; text from the row).
CHIP_STATUS = {
    "ready": "completed",
    "loaded": "completed",
    "needs_key": "failed",
    "not_downloaded": "pending",
    "missing": "pending",
    "downloading": "in_progress",
    "loading": "in_progress",
    "unavailable": "disabled",
    "error": "failed",
    "off": "skipped",
}


def push(*controls: Any) -> None:
    for control in controls:
        if control is None:
            continue
        try:
            control.update()
        except Exception:  # not mounted (host tests) / detached
            pass


class JobEnds:
    """``on_end(snapshot)`` whenever a job of any kind reaches a terminal state (the JobService's
    ``on_transition``; ``JobWatch`` only follows the jobs a tab started). A lookup that a running
    job refused (``services.manga.MangaBusy``: the job owned the process state) runs again once
    that job is over; the JobService reports the end after the job has let go of it."""

    def __init__(self, ctx: Any, on_end: Callable[[Any], Any]) -> None:
        self.ctx = ctx
        self.on_end = on_end
        self._unsub: Optional[Callable[[], Any]] = None

    def start(self) -> None:
        if self._unsub is not None:
            return
        jobs = self.ctx.jobs
        on_transition = getattr(jobs, "on_transition", None) if jobs is not None else None
        if callable(on_transition):
            try:
                self._unsub = on_transition(self._on_transition)
            except Exception:
                log.debug("subscribing to job transitions failed", exc_info=True)

    def stop(self) -> None:
        unsub, self._unsub = self._unsub, None
        if unsub is not None:
            try:
                unsub()
            except Exception:
                pass

    def _on_transition(self, snap: Any, previous: Any) -> None:
        if getattr(snap, "is_terminal", False):
            try:
                self.on_end(snap)
            except Exception:
                log.exception("manga job end handler failed")


def option_chip(text: str, status: str, *, key: Optional[str] = None, dark: bool = False) -> ft.Control:
    """Status chip: icon + text + colour (UI_SPEC §5.3), never colour alone."""
    from glossarion_mobile.ui.theme import status_color

    style_key = CHIP_STATUS.get(status, "info")
    color = status_color(style_key, dark)
    return ft.Container(
        content=ft.Row([ft.Icon(icon_data(tokens.status_style(style_key).icon), size=14, color=color),
                        ft.Text(text, theme_style=ft.TextThemeStyle.LABEL_SMALL, color=color)],
                       spacing=4, tight=True),
        padding=ft.Padding.symmetric(horizontal=6, vertical=2),
        border=ft.Border.all(1, color),
        border_radius=tokens.RADII["chip"],
        key=key,
    )


def reason_or_chip(row: Any, *, key: Optional[str] = None, dark: bool = False) -> ft.Control:
    """A disabled row's ReasonChip, else its status chip."""
    if getattr(row, "reason", None):
        from glossarion_mobile.services.manga import chip_text

        return ReasonChip(reason=str(row.chip or chip_text(row.reason)), detail=row.reason
                          + (f"\n\n{row.detail}" if getattr(row, "detail", "") else ""), key=key)
    return option_chip(str(getattr(row, "chip", "") or ""), str(getattr(row, "status", "info")), key=key, dark=dark)


def export_sheet(ctx: Any, path: str, *, title: Optional[str] = None) -> Optional[ActionSheet]:
    """FileBridge export options for ``path`` (Share… · Save to… · Save to Downloads · Show in Files)."""
    files = ctx.files
    if files is None:
        ctx.say("Exporting is not available")
        return None

    async def run(option_id: str) -> Any:
        try:
            result = await files.export(option_id, path)
        except Exception as exc:
            ctx.say(f"Export failed: {exc}")
            return None
        location = getattr(result, "location", None)
        if getattr(result, "needs_confirm", False):
            from glossarion_mobile.ui.tools.common import ask

            answer = await ask(ctx, "Large file", f"{os.path.basename(path)} is larger than 200 MB. Save it anyway?",
                               (("no", "Cancel", "text"), ("yes", "Save", "filled")), key="manga-save-confirm")
            if answer == "yes":
                result = await files.save_as(path, confirmed=True)
                location = getattr(result, "location", None)
        if location:
            ctx.say(f"Saved {os.path.basename(path)}")
        elif option_id == "downloads" and isinstance(result, str):
            ctx.say(f"Saved to Downloads/Glossarion: {os.path.basename(path)}")
        elif getattr(result, "error", None):
            ctx.say(f"Saving failed: {result.error}")
        return result

    items = [ActionItem(option.label, (lambda oid=option.id: run(oid)), icon=option.icon,
                        disabled_reason=option.disabled_reason, key=f"manga-export-{option.id}")
             for option in files.export_options(path)]
    sheet = ActionSheet(items, title=title or os.path.basename(path), tablet=bool(getattr(ctx, "tablet", False)))
    ctx.extras["manga_last_sheet"] = sheet
    if ctx.page is not None:
        sheet.show(ctx.page)
    return sheet


class MangaTab:
    """A tab of the manga screen: ``build()`` once, ``did_show()`` when selected,
    ``on_session_loaded()`` after the persisted selection is back, ``refresh()`` on outside
    changes, ``dispose()`` when the screen leaves."""

    key = "manga-tab"

    def __init__(self, ctx: Any, session: Any, *, screen: Any = None) -> None:
        self.ctx = ctx
        self.session = session
        self.screen = screen
        self.root: Optional[ft.Control] = None

    def build(self) -> ft.Control:
        raise NotImplementedError

    def did_show(self) -> None:
        pass

    def on_session_loaded(self) -> None:
        self.refresh()

    def refresh(self) -> None:
        pass

    def dispose(self) -> None:
        pass

    def section(self, title: str, controls: list, *, icon: Any = None, key: Optional[str] = None,
                subtitle: Optional[str] = None, trailing: Optional[ft.Control] = None) -> ft.Container:
        from glossarion_mobile.ui.tools.common import card

        return card(title, controls, icon=icon, key=key, subtitle=subtitle, trailing=trailing)

