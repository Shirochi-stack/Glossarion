"""Backup & restore (``/settings/backup``; UI_SPEC §4.16 Data › Backup & restore).

The desktop Config Backup Manager over the shared ``config_store`` core: the
automatic backups in ``config_backups`` beside config.json (72 h retention, newest
first), **Create backup**, **Restore** and **Delete** with the desktop
confirmation texts. Restore validates the backup, makes a safety backup of the
current config and replaces config.json atomically
(``config_store.restore_config_backup_file``); where the desktop then restarts, the
phone reloads ``MobileConfigStore`` in place (a running job keeps its snapshot).
Pending edits are flushed first so the safety backup holds them.

Also: export of config.json without API keys (Share; every credential removed:
``config_export.secret_fields``), **Export with API keys (passphrase)** and **Import config…**
(``services.config_export``: the same credentials re-encrypted with a passphrase-derived key in the
shared ``api_key_encryption`` ENC: format, so the keys can move to another device; U9), and links to
profile and key pool import / export.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Callable, Optional

import flet as ft

from glossarion_mobile.ui.components.dialogs import ConfirmDialog
from glossarion_mobile.ui.screens.page_base import PageScreen, human_size, section
from glossarion_mobile.ui.theme import HIT_TARGET

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
    """A copy of ``config`` with every credential removed (``services.config_export.without_secrets``: the
    ``api_key_encryption`` field lists, the settings_schema ``secret`` keys and the Azure OCR keys)."""
    from glossarion_mobile.services.config_export import without_secrets

    return without_secrets(config)


def _when(mtime: Any) -> str:
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(mtime)))
    except (TypeError, ValueError):
        return "unknown time"


class BackupScreen(PageScreen):
    title = "Backup & restore"

    def __init__(self, match: Any, ctx: Any, *, share_file: Optional[Callable[[str], Any]] = None,
                 temp_dir: Optional[str] = None, pick_files: Optional[Callable[..., Any]] = None) -> None:
        super().__init__(match, ctx)
        self.share_file = share_file
        self.temp_dir = temp_dir
        self.pick_files = pick_files
        self.passphrase_dialog: Any = None
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
                ft.ListTile(title=ft.Text("Export with API keys (passphrase)"), leading=ft.Icon(ft.Icons.ENHANCED_ENCRYPTION),
                            subtitle=ft.Text("Keys re-encrypted with a passphrase you choose, for another device",
                                             theme_style=ft.TextThemeStyle.BODY_SMALL),
                            on_click=lambda e: self.ask_passphrase(export=True), disabled=self.share_file is None,
                            key="backup-export-keys"),
                ft.ListTile(title=ft.Text("Import config…"), leading=ft.Icon(ft.Icons.FILE_OPEN),
                            subtitle=ft.Text("A passphrase export; a safety backup is made first",
                                             theme_style=ft.TextThemeStyle.BODY_SMALL),
                            on_click=lambda e: self.spawn(self.pick_import()), disabled=self.pick_files is None,
                            key="backup-import-config"),
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
                              on_click=lambda e, en=entry: self.confirm_delete(en), size_constraints=HIT_TARGET),
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

    # ---- passphrase export / import (U9) ------------------------------------------------------------

    def ask_passphrase(self, *, export: bool, path: str = "") -> ft.AlertDialog:
        """The passphrase dialog: twice for an export, once for an import."""
        from glossarion_mobile.services.config_export import MIN_PASSPHRASE
        from glossarion_mobile.ui.components.dialogs import close_dialog

        first = ft.TextField(label="Passphrase", password=True, can_reveal_password=True, autofocus=True,
                             key="backup-pass-1")
        second = ft.TextField(label="Repeat passphrase", password=True, can_reveal_password=True,
                              visible=export, key="backup-pass-2")
        error = ft.Text("", color=ft.Colors.ERROR, visible=False, key="backup-pass-error")

        def fail(text: str) -> None:
            error.value = text
            error.visible = True
            try:
                error.update()
            except Exception:
                pass

        def ok(e: Any = None) -> None:
            value = str(first.value or "")
            if len(value) < MIN_PASSPHRASE:
                fail(f"Use at least {MIN_PASSPHRASE} characters")
                return
            if export and value != str(second.value or ""):
                fail("The passphrases do not match")
                return
            close_dialog(self.page, dialog)
            self.spawn(self.export_with_keys(value) if export else self.import_config(path, value))

        dialog = ft.AlertDialog(
            title=ft.Text("Export with API keys" if export else "Import config"),
            content=ft.Column([
                ft.Text("Anyone with this file and the passphrase can use your API keys. The passphrase is not "
                        "stored; it cannot be recovered." if export else
                        "Enter the passphrase the export was made with.", theme_style=ft.TextThemeStyle.BODY_SMALL),
                first, second, error,
            ], tight=True, spacing=8),
            actions=[ft.TextButton(content="Cancel", on_click=lambda e: close_dialog(self.page, dialog)),
                     ft.FilledButton(content="Export" if export else "Import", on_click=ok, key="backup-pass-ok")],
            key="backup-pass-dialog",
        )
        self.passphrase_dialog = dialog
        if self.page is not None:
            self.page.show_dialog(dialog)
        return dialog

    async def export_with_keys(self, passphrase: str) -> Optional[str]:
        from glossarion_mobile.services import config_export as ce

        store = self.store
        directory = self.temp_dir or os.path.dirname(store.path)
        path = os.path.join(directory, time.strftime("glossarion_config_%Y%m%d-%H%M%S.json"))

        def write() -> str:
            os.makedirs(directory, exist_ok=True)
            return ce.write_export(path, ce.export_config(store.snapshot(), passphrase))

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

    async def pick_import(self) -> Optional[str]:
        if self.pick_files is None:
            return None
        result = self.pick_files(["json"], False)
        if hasattr(result, "__await__"):
            result = await result
        paths = [str(getattr(item, "path", item)) for item in (result or []) if item]
        if not paths:
            return None
        self.ask_passphrase(export=False, path=paths[0])
        return paths[0]

    async def import_config(self, path: str, passphrase: str) -> bool:
        """Read, decrypt (passphrase), safety-backup the current config, then replace it in place."""
        from glossarion_mobile.services import config_export as ce

        store = self.store

        def run() -> int:
            config = ce.import_config(ce.read_export(path), passphrase)
            store.flush()
            store.backup_now()
            changed = store.revert_to(config)
            store.flush()
            return len(changed)

        try:
            changed = await self.io(run)
        except ce.WrongPassphrase:
            self.say("Wrong passphrase")
            return False
        except Exception as exc:
            self.say(f"Import failed: {exc}")
            return False
        self.say(f"Config imported ({changed} setting{'s' if changed != 1 else ''} changed); "
                 "the previous config is in the backups")
        await self.refresh()
        return True
