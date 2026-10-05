"""Backup & restore (``/settings/backup``; UI_SPEC §4.16 Data › Backup & restore).

The desktop Config Backup Manager over the shared ``config_store`` core: the
automatic backups in ``config_backups`` beside config.json (72 h retention, newest
first), **Create backup**, **Restore** and **Delete** with the desktop
confirmation texts. Restore validates the backup, makes a safety backup of the
current config and replaces config.json atomically
(``config_store.restore_config_backup_file``); where the desktop then restarts, the
phone reloads ``MobileConfigStore`` in place (a running job keeps its snapshot).
Pending edits are flushed first so the safety backup holds them.

Also: export of config.json without API keys (Share), and links to profile and key
pool import / export.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import time
from typing import Any, Callable, Optional

import flet as ft

from glossarion_mobile.ui.components.dialogs import ConfirmDialog
from glossarion_mobile.ui.screens.page_base import PageScreen, human_size, section

__all__ = ["BackupScreen", "config_without_keys", "delete_backup", "list_backups", "restore_backup"]

log = logging.getLogger("glossarion.backup")


def list_backups(config_path: str) -> list:
    """Blocking: ``config_store.list_config_backups`` (newest first)."""
    import config_store

    return list(config_store.list_config_backups(config_path))


def restore_backup(store: Any, backup_path: str) -> str:
    """Blocking: flush, safety-backup + atomic restore (shared core), then reload the store."""
    import config_store

    store.flush()
    config_store.restore_config_backup_file(
        store.path, backup_path, safety_backup=lambda: config_store.backup_config_file(store.path))
    store.reload()
    return os.path.basename(backup_path)


def delete_backup(config_path: str, name: str) -> None:
    """Blocking: remove one backup (desktop ``delete_selected``); only files named like backups."""
    import config_store

    if os.path.basename(name) != name or not config_store._is_backup_name(name):
        raise ValueError(f"Not a configuration backup: {name}")
    os.remove(os.path.join(config_store.config_backup_dir(config_path), name))


def config_without_keys(config: dict) -> dict:
    """A copy of ``config`` with every API key field removed (``api_key_encryption`` field lists)."""
    out = copy.deepcopy(dict(config or {}))
    try:
        import api_key_encryption

        handler = api_key_encryption.get_handler()
        plain = list(getattr(handler, "api_key_fields", []) or [])
        lists_fn = getattr(handler, "multi_key_list_fields", None)
        if lists_fn is None:
            lists_fn = lambda: api_key_encryption.APIKeyEncryption.multi_key_list_fields(handler)  # noqa: E731
        lists = list(lists_fn())
    except Exception:
        plain, lists = ["api_key"], []
    for key in plain:
        out.pop(key, None)
    for key in lists:
        entries = out.get(key)
        if isinstance(entries, list):
            out[key] = [{k: v for k, v in e.items() if k != "api_key"} if isinstance(e, dict) else e for e in entries]
    return out


def _when(mtime: Any) -> str:
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(mtime)))
    except (TypeError, ValueError):
        return "unknown time"


class BackupScreen(PageScreen):
    title = "Backup & restore"

    def __init__(self, match: Any, ctx: Any, *, share_file: Optional[Callable[[str], Any]] = None,
                 temp_dir: Optional[str] = None) -> None:
        super().__init__(match, ctx)
        self.share_file = share_file
        self.temp_dir = temp_dir
        self.backups: list = []
        self.list_column = ft.Column(spacing=0, tight=True)
        self.status = ft.Text("", theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT)

    def build_body(self) -> ft.Control:
        self.create_button = ft.FilledButton(content="Create backup", icon=ft.Icons.SAVE_OUTLINED,
                                             on_click=self._on_create, key="backup-create")
        return self.scaffold([
            section("Configuration backups", [
                ft.Text("A backup of config.json is made automatically before settings are saved; backups older than "
                        "72 hours are removed.", theme_style=ft.TextThemeStyle.BODY_SMALL),
                self.create_button,
                self.status,
                self.list_column,
            ], key="backup-list-card"),
            section("Export & import", [
                ft.ListTile(title=ft.Text("Export settings (without API keys)"), leading=ft.Icon(ft.Icons.IOS_SHARE),
                            on_click=self._on_export, disabled=self.share_file is None, key="backup-export"),
                ft.ListTile(title=ft.Text("Prompt profiles import / export"), leading=ft.Icon(ft.Icons.SWAP_VERT),
                            on_click=lambda e: self.ctx.go("settings.profiles"), key="backup-profiles"),
                ft.ListTile(title=ft.Text("Key pools import / export"), leading=ft.Icon(ft.Icons.KEY),
                            on_click=lambda e: self.ctx.go("settings.keys"), key="backup-keys"),
                ft.ListTile(title=ft.Text("Import from desktop"), leading=ft.Icon(ft.Icons.DOWNLOAD),
                            on_click=lambda e: self.ctx.go("settings.import"), key="backup-import"),
            ]),
        ])

    def did_show(self) -> None:
        self.spawn(self.refresh())

    async def refresh(self) -> list:
        store = self.store
        if store is None:
            return []
        try:
            self.backups = await self.io(list_backups, store.path)
        except Exception as exc:
            self.status.value = f"Could not list backups: {exc}"
            self.push(self.status)
            return []
        self.status.value = f"{len(self.backups)} backup{'s' if len(self.backups) != 1 else ''}"
        self.list_column.controls = [self._row(entry) for entry in self.backups]
        self.push(self.status, self.list_column)
        return self.backups

    def _row(self, entry: dict) -> ft.ListTile:
        name = entry.get("name", "")
        return ft.ListTile(
            leading=ft.Icon(ft.Icons.HISTORY),
            title=ft.Text(_when(entry.get("mtime"))),
            subtitle=ft.Text(f"{name} · {human_size(entry.get('size'))}", theme_style=ft.TextThemeStyle.BODY_SMALL),
            trailing=ft.Row([
                ft.TextButton(content="Restore", on_click=lambda e, en=entry: self.confirm_restore(en)),
                ft.IconButton(icon=ft.Icons.DELETE_OUTLINE, tooltip="Delete",
                              on_click=lambda e, en=entry: self.confirm_delete(en)),
            ], tight=True, spacing=0),
            dense=True,
            key=f"backup-{name}",
        )

    async def create(self) -> Optional[str]:
        store = self.store
        try:
            path = await self.io(store.backup_now)
        except Exception as exc:
            self.say(f"Failed to create backup: {exc}")
            return None
        if path:
            self.say("New configuration backup created successfully!")
        else:
            self.say("Nothing to back up yet: config.json has not been written")
        await self.refresh()
        return path

    async def _on_create(self, e: Any = None) -> None:
        await self.create()

    def confirm_restore(self, entry: dict) -> ConfirmDialog:
        name = entry.get("name", "")
        return self.show(ConfirmDialog(
            title="Confirm Restore",
            body=(f"This will replace your current configuration with the backup from:\n\n{_when(entry.get('mtime'))}\n"
                  f"{name}\n\nA backup of your current config will be created first.\n\n"
                  "Are you sure you want to continue?"),
            confirm_label="Yes", cancel_label="No",
            on_confirm=lambda: self.restore(entry),
        ))

    async def restore(self, entry: dict) -> bool:
        try:
            name = await self.io(restore_backup, self.store, entry.get("path", ""))
        except Exception as exc:
            self.say(f"Failed to restore backup: {exc}")
            return False
        self.say(f"Configuration restored from: {name}")
        await self.refresh()
        return True

    def confirm_delete(self, entry: dict) -> ConfirmDialog:
        name = entry.get("name", "")
        return self.show(ConfirmDialog(
            title="Confirm Delete",
            body=f"Delete backup from {_when(entry.get('mtime'))}?\n\n{name}\n\nThis action cannot be undone.",
            confirm_label="Yes", cancel_label="No", destructive=True,
            on_confirm=lambda: self.delete(entry),
        ))

    async def delete(self, entry: dict) -> bool:
        try:
            await self.io(delete_backup, self.store.path, entry.get("name", ""))
        except Exception as exc:
            self.say(f"Failed to delete backup: {exc}")
            return False
        self.say("Backup deleted successfully.")
        await self.refresh()
        return True

    async def export_without_keys(self) -> Optional[str]:
        store = self.store
        directory = self.temp_dir or os.path.dirname(store.path)
        path = os.path.join(directory, "glossarion_settings_no_keys.json")

        def write() -> None:
            os.makedirs(directory, exist_ok=True)
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(config_without_keys(store.snapshot()), handle, ensure_ascii=False, indent=2)

        try:
            await self.io(write)
        except Exception as exc:
            self.say(f"Export failed: {exc}")
            return None
        if self.share_file is not None:
            result = self.share_file(path)
            if hasattr(result, "__await__"):
                await result
        return path

    async def _on_export(self, e: Any = None) -> None:
        await self.export_without_keys()
