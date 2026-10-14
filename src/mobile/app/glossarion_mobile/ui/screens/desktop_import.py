"""Import from desktop (``/settings/import``; UI_SPEC §4.16 Data › Import from desktop).

1. **Settings** - pick the desktop ``config.json`` and, optionally, its key file
   (``.glossarion_key`` / ``glossarion_key.txt`` next to the desktop app). The API keys
   in a desktop config are Fernet ``ENC:`` values encrypted with that desktop key: they
   are decrypted with the shared ``api_key_encryption.APIKeyEncryption.decrypt_config``
   (the same field lists) using the desktop key, then merged into ``MobileConfigStore``
   - whose save re-encrypts them with this device's SecureStorage key. Without the key
   file the ``ENC:`` values are kept as they are and Settings shows "re-enter keys".
   Every top-level key of the desktop config is merged as-is (values of routes that are
   excluded on mobile round-trip untouched); keys only the phone has stay. A backup of
   the current config is made first.
2. **Prompt profiles** - a desktop "Export Profiles" JSON, merged like desktop
   "Import Profiles" (``ProfileService.import_json``).
3. **Glossaries** - glossary files (CSV / JSON / TXT / MD) or a ZIP of the desktop
   ``Glossary`` folder, copied into the shared Glossary folder
   (``glossary_paths.resolve_shared_glossary_dir``); the backend's legacy-layout
   migration files root-level ``<book>_glossary.*`` into book folders on the next run.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import zipfile
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import flet as ft

from glossarion_mobile.ui.screens.page_base import PageScreen, section

__all__ = [
    "DesktopImportScreen",
    "GLOSSARY_EXTENSIONS",
    "ImportPreview",
    "decrypt_desktop_config",
    "import_glossaries",
    "merge_desktop_config",
    "read_desktop_config",
    "read_desktop_key",
]

log = logging.getLogger("glossarion.import")

GLOSSARY_EXTENSIONS = (".csv", ".json", ".txt", ".md")


@dataclass
class ImportPreview:
    path: str
    config: dict  # decrypted where possible
    settings: int = 0
    secrets: int = 0  # ENC: values in the desktop file
    decrypted: int = 0
    undecryptable: list = field(default_factory=list)

    @property
    def summary(self) -> str:
        parts = [f"{self.settings} setting{'s' if self.settings != 1 else ''}"]
        if self.secrets:
            parts.append(f"{self.decrypted}/{self.secrets} API keys decrypted")
        if self.undecryptable:
            parts.append(f"re-enter: {', '.join(self.undecryptable)}")
        return " · ".join(parts)


def read_desktop_config(path: str) -> dict:
    """Blocking: the desktop config.json as stored (still encrypted)."""
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("This file is not a Glossarion config.json (no settings object).")
    return data


def read_desktop_key(path: str) -> bytes:
    """Blocking: the desktop Fernet key file (``.glossarion_key`` / ``glossarion_key.txt``), validated."""
    import api_key_encryption

    with open(path, "rb") as handle:
        raw = handle.read().strip()
    return api_key_encryption._normalize_key_material(raw)


def _enc(value: Any) -> bool:
    return isinstance(value, str) and value.startswith("ENC:")


def _secret_fields() -> tuple:
    import api_key_encryption

    handler = api_key_encryption.get_handler()
    plain = tuple(getattr(handler, "api_key_fields", None) or api_key_encryption._NullHandler.api_key_fields)
    lists = tuple(api_key_encryption.APIKeyEncryption.multi_key_list_fields(handler))
    return plain, lists


def _count_enc(config: dict, plain: tuple, lists: tuple) -> tuple:
    found: list = []
    count = 0
    for key in plain:
        if _enc(config.get(key)):
            count += 1
            found.append(key)
    for key in lists:
        entries = config.get(key)
        if isinstance(entries, list):
            n = sum(1 for e in entries if isinstance(e, dict) and _enc(e.get("api_key")))
            if n:
                count += n
                found.append(key)
    return count, found


def decrypt_desktop_config(raw: dict, key: Optional[bytes] = None) -> tuple:
    """``(config, secrets, decrypted, undecryptable_fields)``: ``ENC:`` values decrypted with the desktop key.

    Uses the shared ``APIKeyEncryption.decrypt_config`` with a cipher built from ``key``
    (never the device's own key). Values that do not decrypt stay ``ENC:``.
    """
    plain, lists = _secret_fields()
    secrets, _fields = _count_enc(raw, plain, lists)
    config = dict(raw)
    if key is not None and secrets:
        import api_key_encryption
        from cryptography.fernet import Fernet

        handler = api_key_encryption.APIKeyEncryption.__new__(api_key_encryption.APIKeyEncryption)
        handler.cipher = Fernet(key)
        handler.key_file = None
        handler.api_key_fields = list(plain)
        config = handler.decrypt_config(raw)
    remaining, undecryptable = _count_enc(config, plain, lists)
    return config, secrets, secrets - remaining, undecryptable


def preview_import(path: str, key: Optional[bytes] = None) -> ImportPreview:
    """Blocking: read + decrypt with the desktop key (bytes from ``read_desktop_key``); nothing is written."""
    raw = read_desktop_config(path)
    config, secrets, decrypted, undecryptable = decrypt_desktop_config(raw, key)
    return ImportPreview(path=path, config=config, settings=len(config), secrets=secrets, decrypted=decrypted,
                         undecryptable=undecryptable)


def merge_desktop_config(store: Any, config: dict) -> list:
    """Blocking: back up, then merge every desktop key into the store (sparse set_many); returns changed keys."""
    try:
        store.backup_now()
    except Exception as exc:
        log.info("backup before import failed: %s", exc)
    changed = store.set_many(dict(config))
    store.flush()
    return list(changed)


def import_glossaries(paths: list, target_dir: Optional[str] = None) -> list:
    """Blocking: copy glossary files (or the files of a ZIP) into the shared Glossary folder."""
    if target_dir is None:
        import glossary_paths

        target_dir = glossary_paths.resolve_shared_glossary_dir(create=True)
    os.makedirs(target_dir, exist_ok=True)
    root = os.path.realpath(target_dir)
    copied: list = []
    for path in paths:
        if str(path).lower().endswith(".zip"):
            with zipfile.ZipFile(path) as archive:
                for info in archive.infolist():
                    name = info.filename.replace("\\", "/")
                    if info.is_dir() or not name.lower().endswith(GLOSSARY_EXTENSIONS):
                        continue
                    parts = [p for p in name.split("/") if p not in ("", ".")]
                    if parts and parts[0].lower() == "glossary":
                        parts = parts[1:]  # a zipped desktop Glossary folder
                    if not parts or any(p == ".." for p in parts):
                        continue
                    dest = os.path.realpath(os.path.join(target_dir, *parts))
                    if not dest.startswith(root + os.sep):
                        continue
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    with archive.open(info) as source, open(dest, "wb") as out:
                        shutil.copyfileobj(source, out)
                    copied.append(dest)
            continue
        if not str(path).lower().endswith(GLOSSARY_EXTENSIONS):
            continue
        dest = os.path.join(target_dir, os.path.basename(path))
        if os.path.abspath(path) != os.path.abspath(dest):
            shutil.copy2(path, dest)
        copied.append(dest)
    return copied


class DesktopImportScreen(PageScreen):
    title = "Import from desktop"

    def __init__(self, match: Any, ctx: Any, *, pick_files: Optional[Callable[..., Any]] = None,
                 profiles: Any = None, glossary_dir: Optional[str] = None, scrub_dirs: tuple = ()) -> None:
        super().__init__(match, ctx)
        self.pick_files = pick_files
        self.profiles = profiles  # ProfileService
        self.glossary_dir = glossary_dir
        self.scrub_dirs = tuple(str(d) for d in scrub_dirs if d)  # picked copies of the key file are deleted here
        self.config_path: Optional[str] = None
        self.key_bytes: Optional[bytes] = None
        self.key_name: str = ""
        self.preview: Optional[ImportPreview] = None
        self.config_text = ft.Text("No file chosen", theme_style=ft.TextThemeStyle.BODY_SMALL)
        self.key_text = ft.Text("Optional: without it, API keys must be re-entered",
                                theme_style=ft.TextThemeStyle.BODY_SMALL)
        self.preview_text = ft.Text("", theme_style=ft.TextThemeStyle.BODY_SMALL, selectable=True)
        self.result_text = ft.Text("", theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.PRIMARY)

    def build_body(self) -> ft.Control:
        self.import_button = ft.FilledButton(content="Import settings", icon=ft.Icons.DOWNLOAD, disabled=True,
                                             on_click=self._on_import, key="import-run")
        return self.scaffold([
            section("Settings (config.json)", [
                ft.Text("Copy config.json from the desktop Glossarion folder to the phone, then pick it here. "
                        "Your current settings are backed up first.", theme_style=ft.TextThemeStyle.BODY_SMALL),
                ft.Row([ft.FilledTonalButton(content="Pick config.json", on_click=self._on_pick_config,
                                             key="import-pick-config"), self.config_text], wrap=True),
                ft.Row([ft.TextButton(content="Pick key file (.glossarion_key)", on_click=self._on_pick_key,
                                      key="import-pick-key"), self.key_text], wrap=True),
                self.preview_text,
                self.import_button,
                self.result_text,
            ], key="import-settings"),
            section("Prompt profiles", [
                ft.Text("A JSON file from the desktop “Export Profiles”.", theme_style=ft.TextThemeStyle.BODY_SMALL),
                ft.FilledTonalButton(content="Import profiles…", on_click=self._on_profiles,
                                     disabled=self.profiles is None or not getattr(self.profiles, "available", False),
                                     key="import-profiles"),
            ]),
            section("Glossaries", [
                ft.Text("Glossary files (CSV, JSON, TXT, MD) or a ZIP of the desktop Glossary folder.",
                        theme_style=ft.TextThemeStyle.BODY_SMALL),
                ft.FilledTonalButton(content="Import glossaries…", on_click=self._on_glossaries, key="import-glossaries"),
            ]),
        ])

    async def _pick(self, extensions: list, multiple: bool = False) -> list:
        if self.pick_files is None:
            self.say("The file picker is not available in this session")
            return []
        result = self.pick_files(extensions, multiple)
        if hasattr(result, "__await__"):
            result = await result
        return [str(p) for p in (result or [])]

    async def choose_config(self, path: str) -> Optional[ImportPreview]:
        self.config_path = path
        self.config_text.value = os.path.basename(path)
        return await self.refresh_preview()

    async def choose_key(self, path: str) -> Optional[ImportPreview]:
        """Read the desktop key once (kept in memory only); an app-owned picked copy is deleted."""
        try:
            self.key_bytes = await self.io(read_desktop_key, path)
        except Exception as exc:
            self.key_bytes = None
            self.key_text.value = f"Not a Glossarion key file: {exc}"
            self.push(self.key_text)
            return None
        finally:
            self._scrub(path)
        self.key_name = os.path.basename(path)
        self.key_text.value = f"{self.key_name} (read; the copy on this device was deleted)"
        return await self.refresh_preview()

    def _scrub(self, path: str) -> None:
        try:
            target = os.path.normcase(os.path.realpath(path))
            for root in self.scrub_dirs:
                root = os.path.normcase(os.path.realpath(root))
                if target.startswith(root.rstrip(os.sep) + os.sep) and os.path.isfile(path):
                    os.remove(path)
                    return
        except OSError as exc:
            log.info("could not delete the picked key copy: %s", exc)

    async def refresh_preview(self) -> Optional[ImportPreview]:
        if not self.config_path:
            self.push(self.config_text, self.key_text)
            return None
        try:
            self.preview = await self.io(preview_import, self.config_path, self.key_bytes)
        except Exception as exc:
            self.preview = None
            self.preview_text.value = f"Cannot import this file: {exc}"
            self.import_button.disabled = True
            self.push(self.config_text, self.key_text, self.preview_text, self.import_button)
            return None
        self.preview_text.value = self.preview.summary
        self.import_button.disabled = False
        self.push(self.config_text, self.key_text, self.preview_text, self.import_button)
        return self.preview

    async def run_import(self) -> list:
        preview = self.preview
        if preview is None or self.store is None:
            return []
        try:
            changed = await self.io(merge_desktop_config, self.store, preview.config)
        except Exception as exc:
            self.say(f"Import failed: {exc}")
            return []
        message = f"Imported {len(changed)} changed setting{'s' if len(changed) != 1 else ''} from the desktop"
        if preview.secrets:
            message += f"; {preview.decrypted} API key{'s' if preview.decrypted != 1 else ''} re-encrypted for this device"
        if preview.undecryptable:
            message += f"; re-enter: {', '.join(preview.undecryptable)}"
        self.result_text.value = message
        self.push(self.result_text)
        self.say(message)
        return changed

    async def _on_pick_config(self, e: Any = None) -> None:
        picked = await self._pick(["json"])
        if picked:
            await self.choose_config(picked[0])

    async def _on_pick_key(self, e: Any = None) -> None:
        picked = await self._pick([])
        if picked:
            await self.choose_key(picked[0])

    async def _on_import(self, e: Any = None) -> None:
        await self.run_import()

    async def import_profiles_from(self, path: str) -> int:
        def read() -> str:
            with open(path, "r", encoding="utf-8") as handle:
                return handle.read()

        try:
            count = self.profiles.import_json(await self.io(read))
        except Exception as exc:
            self.say(str(exc) if str(exc).startswith("Failed") else f"Failed to import profiles: {exc}")
            return 0
        self.say(f"Imported {count} profiles.")
        return count

    async def _on_profiles(self, e: Any = None) -> None:
        picked = await self._pick(["json"])
        if picked:
            await self.import_profiles_from(picked[0])

    async def import_glossary_files(self, paths: list) -> list:
        try:
            copied = await self.io(import_glossaries, list(paths), self.glossary_dir)
        except Exception as exc:
            self.say(f"Glossary import failed: {exc}")
            return []
        self.say(f"Imported {len(copied)} glossary file{'s' if len(copied) != 1 else ''}")
        return copied

    async def _on_glossaries(self, e: Any = None) -> None:
        picked = await self._pick(["csv", "json", "txt", "md", "zip"], True)
        if picked:
            await self.import_glossary_files(picked)
