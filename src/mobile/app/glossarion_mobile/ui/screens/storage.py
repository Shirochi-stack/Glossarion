"""Storage (``/settings/storage``; UI_SPEC §4.16 Data › Storage).

* **Folders:** where the app keeps its data (``runtime_bootstrap.AppPaths``): app data
  (config.json, chats, tokens), Output, Library, Inbox, cache, temp, logs, with the
  disk usage of each (measured on a worker thread). The output root is app storage
  (Android) or the Files-visible ``Documents/Glossarion`` (iOS); arbitrary SAF
  folders are not supported (ReasonChip).
* **Clear caches:** empties the cache and temp folders, keeping the seeded tiktoken
  cache (offline token counting) - the OS may purge them anyway.
* **Phone folder (Android 10+, U10):** the public ``Downloads/Glossarion`` folder. The
  pre-U10 "Mirror outputs" switch is folded into U10: books (EPUB, PDF, TXT, HTML) go
  there when the phone folder is the destination in Settings › Cloud sync & sharing
  (``services/cloud_sync``: one entry per output in ``Downloads/Glossarion/<book>/``,
  overwritten on every recompile, so a backup app watching the folder sees one file per
  output). This page says where books go (the cloud sync's ``ui_state``, read through the
  Cloud sync screen's ``CloudFacade``), links there and shows its "My cloud app isn't
  listed" help (TeraBox and other folder-backup apps). The ``Prefs`` switch
  ``mirror_outputs`` now copies only the other finished outputs (images, subtitles,
  glossaries, reports) there once: ``mirror_output``, the call the Library makes when a
  job finishes. iOS and Android 9 or older show the switch disabled with a ReasonChip.
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional

import flet as ft

from glossarion_mobile.services.files import (
    MIRROR_PREF,
    PHONE_FOLDER_LABEL,
    PHONE_FOLDER_NEEDS_ANDROID_10,
    PHONE_FOLDER_NOT_IN_BUILD,
    PHONE_FOLDER_ONLY_ANDROID,
)
from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.components.reason_chip import ReasonChip
from glossarion_mobile.ui.router import ROUTES_BY_NAME
from glossarion_mobile.ui.screens.page_base import PageScreen, human_size, section

__all__ = [
    "KEEP_IN_CACHE",
    "MIRROR_PREF",
    "NOT_LISTED_TITLE",
    "OTHER_OUTPUTS_EXPLAINER",
    "OTHER_OUTPUTS_LABEL",
    "StorageFolder",
    "StorageScreen",
    "clear_folder",
    "folder_usage",
    "mirror_output",
    "phone_folder_books_text",
    "phone_folder_reason_detail",
    "storage_folders",
]

log = logging.getLogger("glossarion.storage")

KEEP_IN_CACHE = ("tiktoken",)  # seeded at boot from the app assets; offline token counting needs it

OTHER_OUTPUTS_LABEL = f"Copy other outputs to {PHONE_FOLDER_LABEL}"
OTHER_OUTPUTS_EXPLAINER = (
    "Images, subtitles, glossaries and reports are copied there once when a job finishes (a second copy gets a "
    "\"(1)\" name). Books are not copied by this switch: they follow Cloud sync & sharing."
)
NOT_LISTED_TITLE = "My cloud app isn't listed"
_REASON_DETAILS = {
    PHONE_FOLDER_ONLY_ANDROID: "iOS keeps outputs in the Files-visible Documents/Glossarion folder already (Files › "
                               "On My iPhone › Glossarion); use Share to send a file to another app.",
    PHONE_FOLDER_NEEDS_ANDROID_10: "Android 9 and older need a storage permission Glossarion does not ask for. Use "
                                   "Share or Save to… instead.",
    PHONE_FOLDER_NOT_IN_BUILD: "The native Glossarion service is not in this build (flet run, or a build without "
                               "the extension).",
}


def phone_folder_reason_detail(reason: str) -> str:
    """The ReasonChip detail for a ``FileBridge.phone_folder_reason`` result."""
    return _REASON_DETAILS.get(reason, reason)


def _cloud_screen() -> Any:
    """``ui/screens/cloud_sync`` (U10 Settings › Cloud sync & sharing), or None in a build without it."""
    try:
        from glossarion_mobile.ui.screens import cloud_sync
    except Exception as exc:  # ImportError, or a broken module: this page still works without it
        log.info("cloud sync screen unavailable: %s", exc)
        return None
    return cloud_sync


def _cloud_route() -> Optional[str]:
    """The Cloud sync & sharing route name, when the router has it."""
    screen = _cloud_screen()
    name = getattr(screen, "ROUTE_NAME", None) if screen is not None else None
    return name if name in ROUTES_BY_NAME else None


def phone_folder_books_text(state: Optional[Mapping[str, Any]], platform: str = "android") -> str:
    """Where books go, from the cloud sync service's ``ui_state()`` (None: no cloud sync in this session)."""
    if platform != "android":
        return _REASON_DETAILS[PHONE_FOLDER_ONLY_ANDROID]
    if not state:
        return (f"Books are not copied to {PHONE_FOLDER_LABEL} automatically. Share or Save to Downloads copies one "
                "file at a time.")
    destination = state.get("destination") if isinstance(state.get("destination"), Mapping) else None
    mode = str((destination or {}).get("mode") or "")
    if mode == "phone":
        if state.get("enabled"):
            return (f"Books (EPUB, PDF, TXT, HTML) are kept in {PHONE_FOLDER_LABEL}/<book>/ and replaced in place "
                    "when you recompile, under the name they were first saved with.")
        return ("The phone folder is chosen in Cloud sync & sharing, but automatic copies are off: a book is copied "
                "when you tap Send now on its page.")
    if destination:
        screen = _cloud_screen()
        describe = getattr(screen, "destination_text", None) if screen is not None else None
        label = str(describe(dict(destination)) or "") if callable(describe) else ""
        label = label or str(destination.get("label") or destination.get("provider_label") or "your cloud folder")
        return f"Books are copied to {label} (Cloud sync & sharing), not to the phone folder."
    return (f"To keep books updated in {PHONE_FOLDER_LABEL}, choose Phone folder in Cloud sync & sharing (off until "
            "you choose it).")


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


async def mirror_output(files: Any, path: str, prefs: Any = None, *, skip_books: bool = False) -> list:
    """Copy a finished output (file or folder) into Downloads/Glossarion once when the switch is on (Android).

    ``skip_books`` (the U10 cloud sync is installed): the outputs the cloud sync copies (its
    ``output_kind``: EPUB, PDF, ``*_translated.txt`` / ``.html``) are left to its destination, so every
    output goes to exactly one place; only the others are copied here (MediaStore names a taken name
    ``name (1).ext``). Returns the saved URIs; nothing happens on other platforms or when the switch is
    off. The listing (and the book check, which looks for a PDF's companion HTML) runs off the loop
    (``files.run_io``).
    """
    if prefs is not None and not prefs.get(MIRROR_PREF, False):
        return []
    saver = getattr(files, "save_to_downloads", None)
    if saver is None or getattr(files, "platform", "") != "android" or not path:
        return []
    is_book: Any = None
    if skip_books:
        try:
            from glossarion_mobile.services.cloud_sync import output_kind as is_book
        except ImportError as exc:
            log.info("cloud sync rules unavailable, copying every output: %s", exc)

    def listing() -> list:
        if os.path.isdir(path):
            found = [os.path.join(root, name) for root, _dirs, names in os.walk(path) for name in sorted(names)]
        else:
            found = [path]
        return [target for target in found if not (is_book and is_book(target))]

    run_io = getattr(files, "run_io", None)
    targets = await run_io(listing) if callable(run_io) else listing()
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


def _cloud_state(cloud: Any) -> Optional[dict]:
    """Blocking (the service may read its sidecar): the cloud sync ``ui_state`` through the Cloud sync screen's
    ``CloudFacade``, or None without a cloud sync service."""
    if cloud is None:
        return None
    screen = _cloud_screen()
    facade_cls = getattr(screen, "CloudFacade", None) if screen is not None else None
    if facade_cls is None:
        return None
    facade = facade_cls(cloud)
    if not facade.available:
        return None
    state = facade.snapshot()
    return dict(state) if isinstance(state, Mapping) else None


class StorageScreen(PageScreen):
    title = "Storage"

    def __init__(self, match: Any, ctx: Any, *, paths: Any = None, platform: str = "desktop",
                 open_files: Any = None, files: Any = None, cloud: Any = None) -> None:
        super().__init__(match, ctx)
        self.paths = paths
        self.platform = platform
        self.open_files = open_files
        #: FileBridge: ``phone_folder_reason`` disables the copy switch where it cannot work (Android 9).
        self.files = files
        #: The U10 cloud sync service, or a function returning it (``app.cloud_sync``): where books go.
        self.cloud = cloud
        self.cloud_state: Optional[dict] = None
        self.phone_folder_reason: Optional[str] = None if platform == "android" else PHONE_FOLDER_ONLY_ANDROID
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
        self.clear_button = ft.FilledTonalButton(content="Clear caches", icon=ft.Icons.CLEANING_SERVICES,
                                                 on_click=self._on_clear, key="storage-clear")
        return self.scaffold([
            section("Folders", rows),
            section("Output folder", [
                ft.Text(output_note, theme_style=ft.TextThemeStyle.BODY_SMALL),
                ft.Row([ft.Text("Choose another folder", expand=True),
                        ReasonChip(reason="Not on mobile", detail="Arbitrary folders (SAF) are not supported; the "
                                   "output root is app storage or iOS Documents. The desktop output_directory value is "
                                   "kept untouched. To keep copies of your books in a cloud folder or the phone folder, "
                                   "use Settings › Cloud sync & sharing.")]),
            ]),
            self._phone_folder_section(),
            section("Caches", [
                ft.Text("Reader, cover and temporary files. The bundled tokenizer data is kept.",
                        theme_style=ft.TextThemeStyle.BODY_SMALL),
                self.clear_button,
            ]),
        ])

    def _phone_folder_section(self) -> ft.Control:
        """Downloads/Glossarion: where books go (the U10 destination), its help, the copy-once switch."""
        self.books_text = ft.Text(phone_folder_books_text(self.cloud_state, self.platform),
                                  theme_style=ft.TextThemeStyle.BODY_SMALL, key="storage-phone-books")
        links: list[ft.Control] = []
        if _cloud_route() is not None:
            links.append(ft.TextButton(content="Cloud sync & sharing", icon=ft.Icons.CLOUD_OUTLINED,
                                       on_click=self.open_cloud_settings, key="storage-phone-cloud"))
        if self.platform == "android" and self._not_listed_help():
            links.append(ft.TextButton(content=NOT_LISTED_TITLE, icon=ft.Icons.HELP_OUTLINE,
                                       on_click=self.show_not_listed_help, key="storage-phone-help"))
        self.mirror_switch = ft.Switch(
            value=bool(self.prefs.get(MIRROR_PREF, False)) if self.prefs is not None else False,
            disabled=self.phone_folder_reason is not None,
            tooltip=OTHER_OUTPUTS_LABEL,
            on_change=lambda e: self.set_mirror(bool(e.control.value)),
            key="storage-mirror",
        )
        reason = self.phone_folder_reason
        self.mirror_reason_row = ft.Row(
            [ReasonChip(reason=reason, detail=phone_folder_reason_detail(reason), key="storage-mirror-reason")]
            if reason else [],
            wrap=True,
            visible=bool(reason),
        )
        return section("Phone folder", [
            self.books_text,
            ft.Row(links, wrap=True, spacing=tokens.SPACING["sm"], visible=bool(links)),
            ft.Row([ft.Text(OTHER_OUTPUTS_LABEL, expand=True), self.mirror_switch],
                   vertical_alignment=ft.CrossAxisAlignment.CENTER),
            ft.Text(OTHER_OUTPUTS_EXPLAINER, theme_style=ft.TextThemeStyle.BODY_SMALL,
                    color=ft.Colors.ON_SURFACE_VARIANT),
            self.mirror_reason_row,
        ], key="storage-phone-folder", subtitle=PHONE_FOLDER_LABEL)

    @staticmethod
    def _not_listed_help() -> str:
        screen = _cloud_screen()
        return str(getattr(screen, "NOT_LISTED_HELP_ANDROID", "") or "") if screen is not None else ""

    def did_show(self) -> None:
        self.spawn(self.measure())
        if self.cloud is not None:
            self.spawn(self.refresh_cloud_state())
        if self.platform == "android" and self.files is not None:
            self.spawn(self.check_phone_folder())

    def app_resumed(self) -> None:
        if self.cloud is not None:
            self.spawn(self.refresh_cloud_state())

    async def refresh_cloud_state(self) -> Optional[dict]:
        try:
            self.cloud_state = await self.io(_cloud_state, self.cloud)
        except Exception as exc:
            log.info("reading the cloud sync state failed: %s", exc)
            return None
        text = phone_folder_books_text(self.cloud_state, self.platform)
        if getattr(self, "books_text", None) is not None and self.books_text.value != text:
            self.books_text.value = text
            self.push(self.books_text)
        return self.cloud_state

    async def check_phone_folder(self) -> Optional[str]:
        """Disable the copy switch with its reason where Downloads/Glossarion cannot be written (Android 9, no
        native service)."""
        checker = getattr(self.files, "phone_folder_reason", None)
        if not callable(checker):
            return None
        try:
            reason = await checker()
        except Exception as exc:
            log.info("phone folder check failed: %s", exc)
            return None
        if reason:
            self._show_phone_folder_reason(reason)
        return reason

    def _show_phone_folder_reason(self, reason: str) -> None:
        if reason == self.phone_folder_reason and self.mirror_reason_row.controls:
            return  # already shown (a keyed chip is never rebuilt under the same key)
        self.phone_folder_reason = reason
        self.mirror_switch.disabled = True
        self.mirror_reason_row.controls = [ReasonChip(reason=reason, detail=phone_folder_reason_detail(reason),
                                                      key="storage-mirror-reason")]
        self.mirror_reason_row.visible = True
        self.push(self.mirror_switch, self.mirror_reason_row)

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
        """The copy-once switch for the outputs that are not books (``mirror_outputs``)."""
        if self.prefs is not None:
            self.prefs.set(MIRROR_PREF, bool(value))

    def show_not_listed_help(self, e: Any = None) -> Any:
        """The Cloud sync screen's "My cloud app isn't listed" help (folder-backup apps such as TeraBox)."""
        from glossarion_mobile.ui.components.info_sheet import InfoSheet

        body = self._not_listed_help()
        return self.show(InfoSheet(title=NOT_LISTED_TITLE, body=body, markdown=True)) if body else None

    def open_cloud_settings(self, e: Any = None) -> Any:
        route = _cloud_route()
        go = getattr(self.ctx, "go", None)
        return go(route) if route is not None and callable(go) else None

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
