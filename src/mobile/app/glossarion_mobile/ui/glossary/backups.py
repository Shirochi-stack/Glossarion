"""BackupsSheet (UI_SPEC §4.1 "⋯ › Backups (list → restore)"; desktop editor "📂 Backups").

The desktop button opens ``<glossary folder>/Backups`` in the file manager; on mobile the
backups of this glossary are listed (``GlossaryDocument.backups``: the JSON snapshots
``create_glossary_backup`` writes, newest first) with Restore (``restore_backup``: a
"before_restore" backup first, then the backup's entries saved in this file's format) and
Share, plus Backup settings / Backup now. An empty or missing folder shows the desktop
"No Backups" text.
"""

from __future__ import annotations

import os
import time
from typing import Any, Callable, Optional, Sequence

import flet as ft

from glossarion_mobile.ui.components._handlers import call_handler
from glossarion_mobile.ui.glossary.common import SheetHost, sheet
from glossarion_mobile.ui.theme import HIT_TARGET

__all__ = ["BackupsSheet", "no_backups_text"]


def no_backups_text(glossary_path: str) -> str:
    backup_dir = os.path.join(os.path.dirname(glossary_path), "Backups")
    return ("No backups folder found yet.\n\n"
            f"Expected location:\n{backup_dir}\n\n"
            "Backups are created automatically when you perform destructive operations (delete, filter, etc).")


def _size(n: int) -> str:
    if n >= 1024 * 1024:
        return f"{n / (1024 * 1024):.1f} MB"
    if n >= 1024:
        return f"{n / 1024:.0f} KB"
    return f"{n} B"


class BackupsSheet:
    def __init__(self, ctx: Any, *, glossary_path: str, backups: Sequence[dict],
                 on_restore: Callable[[str], Any], on_share: Optional[Callable[[str], Any]] = None,
                 on_settings: Optional[Callable[[], Any]] = None, on_backup_now: Optional[Callable[[], Any]] = None,
                 error: Optional[str] = None) -> None:
        self.ctx = ctx
        self.host = SheetHost(ctx)
        self.backups = list(backups)
        self.on_restore = on_restore
        self.on_share = on_share
        controls: list = []
        if error:
            controls.append(ft.Text(error, color=ft.Colors.ERROR, key="backups-error"))
        elif not self.backups:
            controls.append(ft.Text(no_backups_text(glossary_path), selectable=True, key="backups-empty"))
        rows = []
        for row in self.backups:
            path = str(row.get("path") or "")
            mtime = float(row.get("mtime") or 0)
            stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(mtime)) if mtime else ""
            trailing = [ft.IconButton(icon=ft.Icons.RESTORE, tooltip="Restore this backup",
                                      size_constraints=HIT_TARGET, on_click=lambda e, p=path: self.restore(p),
                                      key=f"backup-restore-{row.get('name')}")]
            if on_share is not None:
                trailing.insert(0, ft.IconButton(icon=ft.Icons.IOS_SHARE, tooltip="Share", size_constraints=HIT_TARGET,
                                                 on_click=lambda e, p=path: call_handler(self.on_share, p)))
            rows.append(ft.ListTile(
                leading=ft.Icon(ft.Icons.HISTORY),
                title=ft.Text(str(row.get("name") or os.path.basename(path)), max_lines=2,
                              overflow=ft.TextOverflow.ELLIPSIS),
                subtitle=ft.Text(" · ".join(x for x in (stamp, _size(int(row.get("size") or 0)),
                                                        str(row.get("operation") or "")) if x)),
                trailing=ft.Row(trailing, tight=True, spacing=0),
                key=f"backup-{row.get('name')}",
            ))
        controls.extend(rows)
        actions: list = []
        if on_backup_now is not None:
            actions.append(ft.TextButton(content="Backup now", on_click=lambda e: call_handler(on_backup_now),
                                         key="backups-now"))
        if on_settings is not None:
            actions.append(ft.TextButton(content="Backup settings", on_click=lambda e: self._settings(on_settings),
                                         key="backups-settings"))
        actions.append(ft.TextButton(content="Close", on_click=lambda e: self.host.close()))
        self.dialog = sheet(f"Backups ({len(self.backups)})", controls, actions=actions, key="backups-sheet")

    def _settings(self, handler: Callable[[], Any]) -> Any:
        self.host.close()
        return call_handler(handler)

    def restore(self, path: str) -> Any:
        self.host.close()
        return call_handler(self.on_restore, path)

    def show(self, page: Any = None) -> "BackupsSheet":
        self.host.open(self.dialog)
        return self
