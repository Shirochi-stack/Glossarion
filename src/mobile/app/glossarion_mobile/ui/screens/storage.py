"""Storage (``/settings/storage``; UI_SPEC §4.16 Data › Storage).

* **Folders:** where the app keeps its data (``runtime_bootstrap.AppPaths``): app data
  (config.json, chats, tokens), Output, Library, Inbox, cache, temp, logs, with the
  disk usage of each (measured on a worker thread). The output root is app storage
  (Android) or the Files-visible ``Documents/Glossarion`` (iOS); arbitrary SAF
  folders are not supported (ReasonChip).
* **Clear caches:** empties the cache and temp folders, keeping the seeded tiktoken
  cache (offline token counting) - the OS may purge them anyway.
* **Android "Mirror outputs":** a switch (``Prefs`` ``mirror_outputs``) that copies
  every finished output file into the public ``Downloads/Glossarion`` folder through
  ``FileBridge.save_to_downloads`` (MediaStore). ``mirror_output`` is the call the job
  layer makes when a job finishes.
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import flet as ft

from glossarion_mobile.ui.components.reason_chip import ReasonChip
from glossarion_mobile.ui.screens.page_base import PageScreen, human_size, section

__all__ = [
    "KEEP_IN_CACHE",
    "MIRROR_PREF",
    "StorageFolder",
    "StorageScreen",
    "clear_folder",
    "folder_usage",
    "mirror_output",
    "storage_folders",
]

log = logging.getLogger("glossarion.storage")

MIRROR_PREF = "mirror_outputs"
KEEP_IN_CACHE = ("tiktoken",)  # seeded at boot from the app assets; offline token counting needs it


@dataclass(frozen=True)
class StorageFolder:
    id: str
    label: str
    path: str
    clearable: bool = False
    note: str = ""


def storage_folders(paths: Any) -> list:
    """The folders the Storage page lists, from ``AppPaths`` (or anything with the same attributes)."""
    if paths is None:
        return []
    data = str(getattr(paths, "data", "") or "")
    out = [
        StorageFolder("data", "App data", data, note="config.json, chats, sign-in tokens, backups"),
        StorageFolder("output", "Output", str(getattr(paths, "output", "") or "")),
        StorageFolder("library", "Library", str(getattr(paths, "library", "") or "")),
        StorageFolder("inbox", "Inbox", os.path.join(data, "Inbox") if data else "", note="Imported files"),
        StorageFolder("cache", "Cache", str(getattr(paths, "cache", "") or ""), clearable=True),
        StorageFolder("temp", "Temporary files", str(getattr(paths, "temp", "") or ""), clearable=True),
        StorageFolder("logs", "Logs", str(getattr(paths, "logs", "") or "")),
        # U9: the API client's request/response dumps and the HTTP request log (Logs & diagnostics)
        StorageFolder("payloads", "Payloads", os.path.join(data, "Payloads") if data else "", clearable=True,
                      note="API request/response dumps (Logs & diagnostics › Save payloads)"),
        StorageFolder("http_requests", "HTTP requests",
                      os.path.join(str(getattr(paths, "logs", "") or ""), "http_requests")
                      if getattr(paths, "logs", "") else "", clearable=True,
                      note="Logs & diagnostics › HTTP logging"),
    ]
    return [folder for folder in out if folder.path]


def folder_usage(path: str) -> tuple:
    """Blocking: ``(bytes, files)`` under ``path`` (0, 0 when missing)."""
    total = 0
    files = 0
    if not path or not os.path.isdir(path):
        return 0, 0
    for root, _dirs, names in os.walk(path):
        for name in names:
            try:
                total += os.path.getsize(os.path.join(root, name))
                files += 1
            except OSError:
                pass
    return total, files


def clear_folder(path: str, keep: Iterable[str] = KEEP_IN_CACHE) -> int:
    """Blocking: delete the contents of ``path`` except the names in ``keep``; returns entries removed."""
    removed = 0
    if not path or not os.path.isdir(path):
        return 0
    keep = set(keep)
    for name in os.listdir(path):
        if name in keep:
            continue
        target = os.path.join(path, name)
        try:
            if os.path.isdir(target) and not os.path.islink(target):
                shutil.rmtree(target)
            else:
                os.remove(target)
            removed += 1
        except OSError as exc:
            log.info("could not remove %s: %s", target, exc)
    return removed


async def mirror_output(files: Any, path: str, prefs: Any = None) -> list:
    """Copy a finished output (file or folder) into Downloads/Glossarion when the switch is on (Android).

    Returns the saved URIs; nothing happens on other platforms or when the switch is off.
    """
    if prefs is not None and not prefs.get(MIRROR_PREF, False):
        return []
    saver = getattr(files, "save_to_downloads", None)
    if saver is None or getattr(files, "platform", "") != "android" or not path:
        return []
    targets = [path]
    if os.path.isdir(path):
        targets = [os.path.join(root, name) for root, _dirs, names in os.walk(path) for name in sorted(names)]
    saved = []
    for target in targets:
        try:
            uri = await saver(target)
        except Exception as exc:
            log.info("mirroring %s failed: %s", target, exc)
            continue
        if uri:
            saved.append(uri)
    return saved


class StorageScreen(PageScreen):
    title = "Storage"

    def __init__(self, match: Any, ctx: Any, *, paths: Any = None, platform: str = "desktop",
                 open_files: Any = None) -> None:
        super().__init__(match, ctx)
        self.paths = paths
        self.platform = platform
        self.open_files = open_files
        self.folders = storage_folders(paths)
        self.usage: dict = {}
        self.usage_texts: dict[str, ft.Text] = {}

    def build_body(self) -> ft.Control:
        rows: list[ft.Control] = []
        for folder in self.folders:
            text = ft.Text("Measuring…", theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT)
            self.usage_texts[folder.id] = text
            rows.append(ft.ListTile(
                title=ft.Text(folder.label),
                subtitle=ft.Column([ft.Text(folder.path, selectable=True, theme_style=ft.TextThemeStyle.BODY_SMALL),
                                    text] + ([ft.Text(folder.note, theme_style=ft.TextThemeStyle.BODY_SMALL)]
                                             if folder.note else []), spacing=0, tight=True),
                dense=True,
                key=f"storage-{folder.id}",
            ))
        output_note = ("Outputs are saved in Documents/Glossarion, visible in the Files app."
                       if self.platform == "ios" else "Outputs are saved in the app's storage; share or export them "
                                                       "from the chat, the Library or Files.")
        self.mirror_switch = ft.Switch(
            label="Mirror finished outputs to Downloads/Glossarion",
            value=bool(self.prefs.get(MIRROR_PREF, False)) if self.prefs is not None else False,
            disabled=self.platform != "android",
            on_change=lambda e: self.set_mirror(bool(e.control.value)),
            key="storage-mirror",
        )
        mirror_row: list[ft.Control] = [self.mirror_switch]
        if self.platform != "android":
            mirror_row.append(ReasonChip(reason="Android only", detail="iOS keeps outputs in the Files-visible "
                                         "Documents/Glossarion folder already."))
        self.clear_button = ft.FilledTonalButton(content="Clear caches", icon=ft.Icons.CLEANING_SERVICES,
                                                 on_click=self._on_clear, key="storage-clear")
        return self.scaffold([
            section("Folders", rows),
            section("Output folder", [
                ft.Text(output_note, theme_style=ft.TextThemeStyle.BODY_SMALL),
                ft.Row([ft.Text("Choose another folder", expand=True),
                        ReasonChip(reason="Not on mobile", detail="Arbitrary folders (SAF) are not supported; the "
                                   "output root is app storage or iOS Documents. The desktop output_directory value is "
                                   "kept untouched.")]),
                ft.Row(mirror_row, wrap=True),
            ]),
            section("Caches", [
                ft.Text("Reader, cover and temporary files. The bundled tokenizer data is kept.",
                        theme_style=ft.TextThemeStyle.BODY_SMALL),
                self.clear_button,
            ]),
        ])

    def did_show(self) -> None:
        self.spawn(self.measure())

    async def measure(self) -> dict:
        for folder in self.folders:
            size, files = await self.io(folder_usage, folder.path)
            self.usage[folder.id] = (size, files)
            text = self.usage_texts.get(folder.id)
            if text is not None:
                text.value = f"{human_size(size)} · {files:,} file{'s' if files != 1 else ''}"
                self.push(text)
        return dict(self.usage)

    def set_mirror(self, value: bool) -> None:
        if self.prefs is not None:
            self.prefs.set(MIRROR_PREF, bool(value))

    async def clear_caches(self) -> int:
        removed = 0
        for folder in self.folders:
            if folder.clearable:
                removed += await self.io(clear_folder, folder.path)
        self.say(f"Cleared caches ({removed} item{'s' if removed != 1 else ''})")
        await self.measure()
        return removed

    async def _on_clear(self, e: Any = None) -> None:
        await self.clear_caches()

    def folder(self, folder_id: str) -> Optional[StorageFolder]:
        return next((f for f in self.folders if f.id == folder_id), None)
